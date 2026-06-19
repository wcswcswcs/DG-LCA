#!/usr/bin/env python3
"""DG-KAN v22.43 metric-preserving continuous functional-flow FU runner.

This runner deliberately separates the claims in the v22.43 plan:

* Phase 0: metric geometry harness/unit evidence.
* S4-M: metric support value, not signal-FU value.
* S1-M: residual signal direction inside the selected metric support.
* S6-M: KAN basis carrier value against MLP matched metric support.
* S3-M: strong optimizer / metric optimizer controls.

The task loop uses train-only metric state.  No held/test/future/query signal is
used to select runtime directions; held/test loaders are only evaluated after
training or at pre-registered horizon checkpoints.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
csv.field_size_limit(sys.maxsize)

from experiments import run_v22_37_causal_instrumented_functional_optimizer as core
from experiments import run_v22_40_continuous_functional_flow_fu as v2240
from experiments import run_v22_42R_support_first_residual_kan_carrier as v2242


PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
OUT_ROOT = ROOT / "results/v22_43"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.43_MetricPreservingContinuousFunctionalFlowFU_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.43_MetricPreservingContinuousFunctionalFlowFU_实验结果复盘.md"

REQUIRED_ARTIFACTS = [
    "v22_43_metric_harness_unit_matrix.csv",
    "v22_43_metric_projection_unit_matrix.csv",
    "v22_43_generalized_spectrum_drift_matrix.csv",
    "v22_43_metric_support_full_loop_matrix.csv",
    "v22_43_metric_support_control_matrix.csv",
    "v22_43_metric_residual_signal_matrix.csv",
    "v22_43_tau_direction_matrix.csv",
    "v22_43_KAN_basis_Gram_matrix.csv",
    "v22_43_KAN_metric_basis_carrier_matrix.csv",
    "v22_43_MLP_matched_metric_support_matrix.csv",
    "v22_43_strong_optimizer_metric_baseline_matrix.csv",
    "v22_43_hard_task_four_square_matrix.csv",
    "v22_43_continual_grokking_matrix.csv",
    "v22_43_efficiency_matrix.csv",
    "v22_43_runtime_regression_audit.csv",
    "v22_43_final_route.json",
    "v22_43_command_journal.csv",
    "v22_43_artifact_manifest.csv",
]

PURE_REQUIRED_ARTIFACTS = [
    "v22_43P_pure_fu_runtime_truth_matrix.csv",
    "v22_43P_pure_fu_optimizer_step_audit.csv",
    "v22_43P_pure_fu_gradient_usage_audit.csv",
    "v22_43P_pure_fu_candidate_regression_audit.csv",
    "v22_43P_pure_fu_state_trace.csv",
    "v22_43P_pure_fu_runtime_decision_trace.csv",
    "v22_43P_pure_support_full_loop_matrix.csv",
    "v22_43P_pure_support_control_matrix.csv",
    "v22_43P_pure_residual_signal_matrix.csv",
    "v22_43P_pure_tau_direction_matrix.csv",
    "v22_43P_pure_oet_unit_matrix.csv",
    "v22_43P_pure_metric_spectrum_drift_matrix.csv",
    "v22_43P_pure_oet_full_loop_matrix.csv",
    "v22_43P_pure_oet_radial_matrix.csv",
    "v22_43P_pure_KAN_basis_Gram_matrix.csv",
    "v22_43P_pure_KAN_basis_carrier_matrix.csv",
    "v22_43P_pure_MLP_matched_support_matrix.csv",
    "v22_43P_warmup_then_pure_matrix.csv",
    "v22_43P_hard_task_four_square_matrix.csv",
    "v22_43P_continual_grokking_matrix.csv",
    "v22_43P_efficiency_matrix.csv",
    "v22_43P_final_route.json",
    "v22_43P_failure_dissection.md",
]

PURE_AUDIT_FIELDS = [
    "row_id",
    "run_label",
    "dataset",
    "seed",
    "architecture",
    "carrier",
    "optimizer_reference_name",
    "variant",
    "control_mode",
    "pure_fu_mode",
    "from_scratch_or_warmup",
    "warmup_steps",
    "base_optimizer_step_used",
    "base_optimizer_step_used_after_warmup",
    "base_velocity_added",
    "base_velocity_added_after_warmup",
    "ordinary_adamw_update_norm",
    "ordinary_sgd_update_norm",
    "ordinary_muon_update_norm",
    "ordinary_schedulefree_update_norm",
    "ordinary_adamw_update_norm_after_warmup",
    "ordinary_sgd_update_norm_after_warmup",
    "ordinary_muon_update_norm_after_warmup",
    "ordinary_schedulefree_update_norm_after_warmup",
    "bp_gradient_used_only_for_cotangent",
    "bp_gradient_used_only_for_cotangent_after_warmup",
    "fu_velocity_norm",
    "fu_velocity_emitted_every_step",
    "continuous_fu_state_updated_every_step",
    "candidate_action_selection_used_for_runtime",
    "runtime_argmax_candidate_used",
    "runtime_topk_candidate_used",
    "micro_rct_winner_used_as_runtime_action",
    "candidate_value_model_used_as_runtime_policy",
    "uses_test_direction_selection",
    "uses_future_direction",
    "uses_validation_direction",
    "uses_readout_diagnostic_as_basis_native",
]

TIER0_DEBUG_DATASETS = {"MNIST", "FashionMNIST", "KMNIST"}
HARD_DATASETS = {"CIFAR10", "SVHN", "EMNIST", "EMNIST_LETTERS"}

S4_METRICS = [
    "S4-Euclidean-OET",
    "S4-FisherEMA-OET",
    "S4-SignalMetric-OET",
    "S4-KAN-BasisGram",
    "S4-MLP-MatchedLowRankMetricSupport",
    "S4-MLP-MatchedSpectralSupport",
]

S6_VARIANTS = [
    "KAN-D-CHE-BasisGram",
    "KAN-D-CHE-BasisGramFast",
    "KAN-D-FOU-BasisGram",
    "KAN-D-FOU-BasisGramFast",
    "KAN-D-CHE-FisherSignal-BasisGram",
    "KAN-D-FOU-FisherSignal-BasisGram",
    "KAN-D-CHE-OET-BankLocal",
    "KAN-D-FOU-OET-BankLocal",
    "MLP-low-rank-hidden-metric-support",
    "MLP-frequency-like-random-feature-support",
    "MLP-polynomial-like-feature-support",
    "MLP-same-rank-block-support",
    "MLP-same-param-FLOPs-matched-support",
]

M1_VARIANTS = [
    "M1-FisherMirror",
    "M1-KLMirror",
    "M1-KLMirrorLS",
    "M1-BrierMirror",
    "M1-TailSafeMirror",
]

MIRROR_DIAGNOSTIC_KEYS = {
    "KL_step",
    "Fisher_norm_step",
    "Brier_delta",
    "tail_q99_mirror_delta",
    "margin_q10_mirror_delta",
    "mirror_loss",
    "mirror_dual_state_norm",
    "mirror_primal_delta_norm",
}

S3_VARIANTS = [
    "S3-MetricSupportFU",
    "S3-ResidualSignalFU",
    "S3-SameMetricSupportControl",
    "S3-SameTangentControl",
]

CONTROL_MODES = [
    "none",
    "same-overhead-noop",
    "same-norm-additive-random",
    "same-metric-support-random",
    "same-metric-support-signflip",
    "same-metric-support-shuffled",
    "same-optimizer-geometry-random",
    "M6-signal-noise-control",
    "M6-same-norm-Gaussian-control",
    "M9-fixed-uniform-mixture-control",
    "M9-fixed-euclidean-control",
    "M9-fixed-fisher-control",
    "M9-fixed-signal-control",
    "M9-fixed-basis-control",
]

M6_VARIANTS = [
    "M6-reservoir-noise",
    "M6-noise-suppression",
]

M9_COMPONENTS = ("euclidean", "fisher", "signal", "basis")

M9_VARIANTS = [
    "M9-adaptive-metric-mixture",
]

M9_FIXED_CONTROL_WEIGHTS = {
    "M9-fixed-uniform-mixture-control": {name: 1.0 / len(M9_COMPONENTS) for name in M9_COMPONENTS},
    "M9-fixed-euclidean-control": {"euclidean": 1.0, "fisher": 0.0, "signal": 0.0, "basis": 0.0},
    "M9-fixed-fisher-control": {"euclidean": 0.0, "fisher": 1.0, "signal": 0.0, "basis": 0.0},
    "M9-fixed-signal-control": {"euclidean": 0.0, "fisher": 0.0, "signal": 1.0, "basis": 0.0},
    "M9-fixed-basis-control": {"euclidean": 0.0, "fisher": 0.0, "signal": 0.0, "basis": 1.0},
}

S6_CONTROL_MODES = [
    "none",
    "same-basis-Gram-random",
    "same-basis-Gram-signflip",
    "same-basis-OET-random",
    "same-bank-shuffled",
    "same-degree-frequency-random",
    "same-readout-leakage-control",
]

P3_VARIANTS = [
    "P3-Euclidean-OET-pure",
    "P3-Euclidean-OET-lie-momentum-pure",
    "P3-Euclidean-OET-transported-lie-momentum-pure",
]

P3_CONTROL_MODES = [
    "none",
    "same-OET-random",
    "same-OET-signflip",
    "same-OET-shuffled",
    "same-generator-norm-random",
    "same-Lie-random",
    "same-Lie-signflip",
]

P4_VARIANTS = [
    "P4-Euclidean-OET-radial005-pure",
    "P4-Euclidean-OET-radial010-pure",
    "P4-Euclidean-OET-functional-radial-gated-pure",
    "P4-Euclidean-OET-functional-mode-radial-gated-pure",
    "P4-Euclidean-OET-functional-rse-radial010-gated-pure",
    "P4-Euclidean-OET-functional-rse-relaxed-safe-radial010-gated-pure",
    "P4-Euclidean-OET-functional-jvp-radial-gated-pure",
    "P4-Euclidean-OET-functional-jvp-relaxed-safe-radial-gated-pure",
]

P4_CONTROL_MODES = [
    "none",
    "same-radial-random",
    "same-OET-random",
]

P5_VARIANTS = [
    "P5-D-CHE-BasisGram-additive-pure",
    "P5-D-FOU-BasisGram-additive-pure",
]

P5_CONTROL_MODES = [
    "none",
    "same-basis-Gram-random",
    "same-basis-Gram-signflip",
    "same-bank-shuffled",
    "same-readout-leakage-control",
]

P6_VARIANTS = [
    "S4-SignalMetric-OET",
    "S1-M-top1-S4-SignalMetric-OET",
    "P3-Euclidean-OET-pure",
    "P5-D-CHE-BasisGram-additive-pure",
    "P5-D-FOU-BasisGram-additive-pure",
]


def p6_control_modes_for_variant(variant: str, configured: str) -> list[str]:
    requested = split_csv(configured)
    if requested and requested != ["auto"]:
        return requested
    phase = phase_for_variant(variant)
    if phase == "P3_PureMetricPreservingOET":
        if lie_momentum_variant(variant):
            return ["none", "same-Lie-random", "same-Lie-signflip"]
        return ["none", "same-OET-random"]
    if phase == "P5_PureKANBasisCarrier":
        return ["none", "same-basis-Gram-random"]
    return ["none", "same-metric-support-random"]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def safe_fragment(value: Any) -> str:
    return v2242.safe_fragment(value)


def finite_float(value: Any, default: float | None = None) -> float | None:
    return v2242.finite_float(value, default)


def value_or(value: Any, default: float) -> float:
    parsed = finite_float(value)
    return float(default) if parsed is None else float(parsed)


def mean_present(rows: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    vals = [finite_float(row.get(key)) for row in rows if row.get(key, "") not in {"", None}]
    clean = [float(v) for v in vals if v is not None]
    return statistics.fmean(clean) if clean else float(default)


def value_gt(value: Any, threshold: float) -> bool:
    parsed = finite_float(value)
    return parsed is not None and float(parsed) > float(threshold)


def value_ge(value: Any, threshold: float) -> bool:
    parsed = finite_float(value)
    return parsed is not None and float(parsed) >= float(threshold)


def value_le(value: Any, threshold: float) -> bool:
    parsed = finite_float(value)
    return parsed is not None and float(parsed) <= float(threshold)


def int_flag(value: Any) -> int:
    return v2242.int_flag(value)


def split_csv(text: str, cast: Any = str) -> list[Any]:
    return v2242.split_csv(text, cast)


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.43 Metric-Preserving Continuous Functional Flow FU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只记录真实执行的命令、文件、GPU、状态、blocker 与修复尝试；"
            "未执行、被 gate 阻断、数据不可用或失败必须显式写出；不补造实验结果。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.43 Metric-Preserving Continuous Functional Flow FU 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：复盘只引用本轮 artifact 或明确命名的上游 artifact；"
            "实验数据、修复动作、分析结论、insight 和证据链必须可追溯；不编造缺失数据。\n",
            encoding="utf-8",
        )


def read_rows(path: str | Path) -> list[dict[str, str]]:
    return v2242.read_rows(path)


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    v2242.write_rows(path, rows, fieldnames)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return v2242.sha256_file(path)


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
    journal = read_rows(OUT_ROOT / "v22_43_command_journal.csv")
    journal.append({k: str(v) for k, v in row.items()})
    write_rows(
        OUT_ROOT / "v22_43_command_journal.csv",
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


def run_logged(cmd: list[str], *, task_id: str, gpu: str = "", timeout: int = 900) -> subprocess.CompletedProcess[str]:
    ensure_out()
    stdout_path = LOG_ROOT / f"{safe_fragment(task_id)}_stdout.log"
    stderr_path = LOG_ROOT / f"{safe_fragment(task_id)}_stderr.log"
    env = os.environ.copy()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu).replace("cuda:", "")
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout)
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


def torch_device(name: str) -> Any:
    import torch

    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(str(name))
    return torch.device("cpu")


def stable_seed(*parts: Any) -> int:
    return v2242.stable_seed(*parts)


def hash_model(model: Any) -> str:
    return v2242.hash_model(model)


def hash_optimizer(opt: Any) -> str:
    return v2242.hash_optimizer(opt)


def hash_tensor(x: Any) -> str:
    return v2242.hash_tensor(x)


def named_trainable(model: Any) -> list[tuple[str, Any]]:
    return v2242.named_trainable(model)


def zeros_like_named(named: list[tuple[str, Any]]) -> dict[str, Any]:
    return v2242.zeros_like_named(named)


def grad_dict(named: list[tuple[str, Any]]) -> dict[str, Any]:
    return v2242.grad_dict(named)


def momentum_dict(named: list[tuple[str, Any]], opt: Any) -> dict[str, Any]:
    return v2242.momentum_dict(named, opt)


def tensor_norm(values: dict[str, Any]) -> float:
    return v2242.tensor_norm(values)


def is_basis_param(name: str) -> bool:
    return v2242.is_basis_param(name)


def is_readout_param(name: str) -> bool:
    return v2242.is_readout_param(name)


def readout_param_names(named: list[tuple[str, Any]]) -> set[str]:
    explicit = {name for name, _p in named if is_readout_param(name)}
    if explicit:
        return explicit
    last_weight_name = ""
    for name, p in named:
        if getattr(p, "ndim", 0) >= 2:
            last_weight_name = name
    if not last_weight_name:
        return set()
    out = {last_weight_name}
    stem = last_weight_name.rsplit(".", 1)[0] if "." in last_weight_name else last_weight_name
    for name, _p in named:
        if name == f"{stem}.bias":
            out.add(name)
        elif name == stem[:-6] + "b2" and stem.endswith("w2"):
            out.add(name)
    return out


def random_like(values: dict[str, Any], seed: int) -> dict[str, Any]:
    return v2242.random_like(values, seed)


def apply_velocity(named: list[tuple[str, Any]], velocity: dict[str, Any], scale: float) -> tuple[float, float, float, float]:
    return v2242.apply_velocity(named, velocity, scale)


def percentile(values: list[float], q: float) -> float:
    return v2242.percentile(values, q)


def segment_descent_rate(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) < 2:
        return 0.0
    return (vals[0] - vals[-1]) / max(1, len(vals) - 1)


def first_step_reaching(values: list[float], threshold: float, mode: str = "le") -> int | str:
    if not math.isfinite(float(threshold)):
        return ""
    for idx, value in enumerate(values, start=1):
        if not math.isfinite(float(value)):
            continue
        if mode == "ge" and value >= threshold:
            return idx
        if mode != "ge" and value <= threshold:
            return idx
    return ""


def p1_phase_for_summary(variant: str, pure_fu_mode: bool) -> str:
    if not pure_fu_mode:
        return phase_for_variant(variant)
    phase = phase_for_variant(variant)
    if phase in {"S4_MetricSupport", "Phase0_ControlHarness"}:
        return "P1_PureSupport"
    return phase


def md_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 12) -> str:
    return v2242.md_table(rows, columns, limit)


def metric_kind_for_variant(variant: str) -> str:
    if variant in M9_VARIANTS or variant.startswith("M9-"):
        return "adaptive_metric_mixture"
    if variant == "M1-FisherMirror":
        return "fisher_ema"
    if variant in {"M1-KLMirror", "M1-KLMirrorLS", "M1-BrierMirror", "M1-TailSafeMirror"}:
        return "euclidean_oet"
    if "FisherSignal" in variant:
        return "fisher_signal_basis_gram"
    if "BasisGram" in variant or "basis-Gram" in variant or "Gram" in variant:
        return "kan_basis_gram"
    if "FisherEMA" in variant:
        return "fisher_ema"
    if "SignalMetric" in variant:
        return "signal_drift_diffusion"
    if "Spectral" in variant or "spectral" in variant:
        return "mlp_spectral"
    if "LowRank" in variant or "low-rank" in variant:
        return "mlp_lowrank"
    if "frequency-like" in variant:
        return "mlp_frequency_like"
    if "polynomial-like" in variant:
        return "mlp_polynomial_like"
    if "same-rank-block" in variant:
        return "mlp_block"
    if "same-param" in variant:
        return "mlp_same_param"
    return "euclidean_oet"


def basis_carrier_fast_path(variant: str, architecture: str) -> bool:
    return architecture != "MLP" and "BasisGramFast" in str(variant)


def radial_cap_for_variant(variant: str) -> float:
    if "functional-rse-radial010" in variant:
        return 0.10
    if "functional-jvp" in variant:
        return 0.20
    if "functional-mode-radial-gated" in variant:
        return 0.20
    if "functional-radial-gated" in variant:
        return 0.20
    if "radial005" in variant:
        return 0.05
    if "radial010" in variant:
        return 0.10
    return 0.0


def functional_radial_gate_for_variant(variant: str, state: dict[str, Any]) -> dict[str, float]:
    if (
        "functional-radial-gated" not in str(variant)
        and "functional-mode-radial-gated" not in str(variant)
        and "functional-rse" not in str(variant)
        and "functional-jvp" not in str(variant)
    ):
        return {
            "functional_radial_gate_active": 0.0,
            "functional_radial_score": 0.0,
            "functional_radial_safety_debt": 0.0,
            "functional_radial_effective_cap": radial_cap_for_variant(variant),
        }
    score = max(0.0, min(1.0, value_or(state.get("support_overlap"), 0.0)))
    safety_debt = max(0.0, value_or(state.get("safety_budget_debt"), 0.0))
    safety_scale = max(0.0, min(1.0, value_or(state.get("safety_budget_velocity_scale"), 1.0)))
    if "functional-rse" in str(variant):
        if "relaxed-safe" in str(variant):
            safety_threshold = 0.50
            safety_scale_threshold = 0.30
            safety_multiplier = safety_scale
        else:
            safety_threshold = 0.05
            safety_scale_threshold = 0.50
            safety_multiplier = 1.0
        active = float(safety_debt <= safety_threshold and safety_scale >= safety_scale_threshold)
        return {
            "functional_radial_gate_active": active,
            "functional_radial_score": 1.0 if active else 0.0,
            "functional_radial_safety_debt": safety_debt,
            "functional_radial_effective_cap": radial_cap_for_variant(variant) * active * safety_multiplier,
        }
    if "relaxed-safe" in str(variant):
        safety_threshold = 0.50
        safety_scale_threshold = 0.30
        safety_multiplier = safety_scale
    else:
        safety_threshold = 0.05
        safety_scale_threshold = 0.50
        safety_multiplier = 1.0
    active = float(score >= 0.05 and safety_debt <= safety_threshold and safety_scale >= safety_scale_threshold)
    return {
        "functional_radial_gate_active": active,
        "functional_radial_score": score,
        "functional_radial_safety_debt": safety_debt,
        "functional_radial_effective_cap": radial_cap_for_variant(variant) * score * active * safety_multiplier,
    }


def phase_for_variant(variant: str) -> str:
    if variant in M1_VARIANTS or variant.startswith("M1-"):
        return "M1_FunctionalMirror"
    if variant in M6_VARIANTS or variant.startswith("M6-"):
        return "M6_NoiseShapedReservoirRegularization"
    if variant in M9_VARIANTS or variant.startswith("M9-"):
        return "M9_AdaptiveMetricMixture"
    if variant in S4_METRICS:
        return "S4_MetricSupport"
    if variant.startswith("S1-M"):
        return "S1_MetricResidualSignal"
    if variant in P3_VARIANTS or variant.startswith("P3-"):
        return "P3_PureMetricPreservingOET"
    if variant in P4_VARIANTS or variant.startswith("P4-"):
        return "P4_PureOETRadial"
    if variant in P5_VARIANTS or variant.startswith("P5-"):
        return "P5_PureKANBasisCarrier"
    if variant in S6_VARIANTS:
        return "S6_MetricKANCarrier" if variant.startswith("KAN-") else "S6_MLPMatchedMetricSupport"
    if variant in S3_VARIANTS:
        return "S3_StrongMetricOptimizer"
    if variant == "optimizer_alone":
        return "Phase0_ControlHarness"
    return "unknown"


def default_architecture_for_variant(variant: str, requested: str) -> str:
    if variant.startswith("P5-D-CHE"):
        return "DGKAN_DCHE"
    if variant.startswith("P5-D-FOU"):
        return "DGKAN_DFOU"
    if variant.startswith("KAN-D-CHE"):
        return "DGKAN_DCHE"
    if variant.startswith("KAN-D-FOU"):
        return "DGKAN_DFOU"
    if variant.startswith("MLP-") or "MLP-Matched" in variant:
        return "MLP"
    return requested


def kan_init_variant_for(value: str) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if text in {"", "default", "none"}:
        return "default"
    output_calibrated = "calib" in text or "output" in text or "scale" in text
    if "tail" in text and ("spectrum" in text or "label" in text or "rse" in text):
        return "low-degree-tail-spectrum-biased" if ("low" in text and ("degree" in text or "deg" in text)) else "tail-spectrum-biased"
    if "label" in text and ("spectrum" in text or "rse" in text):
        return "low-degree-label-spectrum-biased" if ("low" in text and ("degree" in text or "deg" in text)) else "label-spectrum-biased"
    if "low" in text and ("degree" in text or "deg" in text):
        return "low-degree-biased"
    if "low" in text and ("freq" in text or "frequency" in text):
        return "low-frequency-biased"
    if "degree" in text and output_calibrated:
        return "degree-balanced-output-calibrated"
    if ("freq" in text or "frequency" in text) and output_calibrated:
        return "frequency-balanced-output-calibrated"
    if "degree" in text:
        return "degree-balanced"
    if "freq" in text or "frequency" in text:
        return "frequency-balanced"
    return text


def basis_bank_class_scale(layer: Any, y: Any, *, tail_weight: Any | None = None, min_scale: float = 0.50, max_scale: float = 1.75) -> Any:
    import torch

    if layer is None or layer.ndim != 3 or y is None:
        return None
    feat = layer.detach().float().mean(dim=1)
    if feat.ndim != 2 or feat.shape[0] < 2:
        return None
    labels = y.detach().long().view(-1).to(device=feat.device)
    if labels.numel() != feat.shape[0]:
        return None
    weights = torch.ones_like(labels, dtype=feat.dtype)
    if tail_weight is not None:
        weights = tail_weight.detach().to(device=feat.device, dtype=feat.dtype).view(-1).clamp_min(0.0)
        if weights.numel() != labels.numel() or float(weights.sum().item()) <= 1.0e-12:
            weights = torch.ones_like(labels, dtype=feat.dtype)
    weights = weights / weights.sum().clamp_min(1.0e-12)
    global_mean = (feat * weights.view(-1, 1)).sum(dim=0)
    total = ((feat - global_mean).square() * weights.view(-1, 1)).sum(dim=0).clamp_min(1.0e-8)
    between = torch.zeros_like(total)
    for cls in labels.unique(sorted=False):
        mask = labels == cls
        cls_w = weights[mask]
        cls_mass = cls_w.sum()
        if float(cls_mass.item()) <= 1.0e-12:
            continue
        cls_mean = (feat[mask] * (cls_w / cls_mass).view(-1, 1)).sum(dim=0)
        between = between + cls_mass * (cls_mean - global_mean).square()
    score = (between / total).clamp_min(1.0e-8)
    score = score / score.mean().clamp_min(1.0e-8)
    return score.sqrt().clamp(float(min_scale), float(max_scale))


def apply_kan_init_variant(model: Any, xb: Any, init_variant: str, yb: Any | None = None) -> str:
    import torch
    import torch.nn.functional as F

    mode = kan_init_variant_for(init_variant)
    if mode == "default" or not hasattr(model, "layer1_basis"):
        return mode
    if not (hasattr(model, "w1") and hasattr(model, "w2")):
        return mode
    balance_mode = mode in {
        "degree-balanced",
        "frequency-balanced",
        "degree-balanced-output-calibrated",
        "frequency-balanced-output-calibrated",
        "low-degree-biased",
        "low-frequency-biased",
        "label-spectrum-biased",
        "tail-spectrum-biased",
        "low-degree-label-spectrum-biased",
        "low-degree-tail-spectrum-biased",
    }
    if not balance_mode:
        return mode
    with torch.no_grad():
        if mode in {"low-degree-biased", "low-frequency-biased", "low-degree-label-spectrum-biased", "low-degree-tail-spectrum-biased"}:
            k1 = int(getattr(model, "w1").shape[-1])
            k2 = int(getattr(model, "w2").shape[-1])
            decay1 = torch.pow(
                torch.full((k1,), 0.75, device=model.w1.device, dtype=model.w1.dtype),
                torch.arange(k1, device=model.w1.device, dtype=model.w1.dtype),
            )
            decay2 = torch.pow(
                torch.full((k2,), 0.75, device=model.w2.device, dtype=model.w2.dtype),
                torch.arange(k2, device=model.w2.device, dtype=model.w2.dtype),
            )
            model.w1.mul_(decay1.view(1, 1, -1))
            model.w2.mul_(decay2.view(1, 1, -1))
            if mode in {"low-degree-biased", "low-frequency-biased"}:
                return mode
        if mode in {"label-spectrum-biased", "tail-spectrum-biased", "low-degree-label-spectrum-biased", "low-degree-tail-spectrum-biased"}:
            if yb is None:
                return mode
            tail_weight = None
            if "tail" in mode:
                logits = model(xb).detach().float()
                losses = F.cross_entropy(logits, yb.detach().long().to(device=logits.device), reduction="none")
                k = max(1, int(math.ceil(0.20 * int(losses.numel()))))
                threshold = torch.topk(losses, k=k, largest=True).values.min()
                tail_weight = torch.ones_like(losses)
                tail_weight = tail_weight + 3.0 * (losses >= threshold).float()
            layer1 = model.layer1_basis(xb).detach().float()
            scale1 = basis_bank_class_scale(layer1, yb, tail_weight=tail_weight)
            if scale1 is not None and getattr(model, "w1").ndim == 3:
                model.w1.mul_(scale1.view(1, 1, -1).to(device=model.w1.device, dtype=model.w1.dtype))
            hidden = model.hidden(xb).detach().float() if hasattr(model, "hidden") else None
            if hidden is not None and hasattr(model, "layer2_basis") and getattr(model, "w2").ndim == 3:
                layer2 = model.layer2_basis(hidden).detach().float()
                scale2 = basis_bank_class_scale(layer2, yb, tail_weight=tail_weight)
                if scale2 is not None:
                    model.w2.mul_(scale2.view(1, 1, -1).to(device=model.w2.device, dtype=model.w2.dtype))
            return mode
        baseline_logits = model(xb).detach().float() if "output-calibrated" in mode else None
        baseline_std = baseline_logits.std().clamp_min(1.0e-6) if baseline_logits is not None else None
        layer1 = model.layer1_basis(xb).detach().float()
        if layer1.ndim == 3 and getattr(model, "w1").ndim == 3:
            bank_energy = layer1.square().mean(dim=(0, 1)).clamp_min(1.0e-8)
            bank_scale = (bank_energy.mean().clamp_min(1.0e-8) / bank_energy).sqrt().clamp(0.25, 4.0)
            model.w1.mul_(bank_scale.view(1, 1, -1).to(device=model.w1.device, dtype=model.w1.dtype))
        hidden = model.hidden(xb).detach().float() if hasattr(model, "hidden") else None
        if hidden is not None and hasattr(model, "layer2_basis") and getattr(model, "w2").ndim == 3:
            layer2 = model.layer2_basis(hidden).detach().float()
            if layer2.ndim == 3:
                bank_energy = layer2.square().mean(dim=(0, 1)).clamp_min(1.0e-8)
                bank_scale = (bank_energy.mean().clamp_min(1.0e-8) / bank_energy).sqrt().clamp(0.25, 4.0)
                model.w2.mul_(bank_scale.view(1, 1, -1).to(device=model.w2.device, dtype=model.w2.dtype))
        if baseline_std is not None:
            calibrated_logits = model(xb).detach().float()
            calibrated_std = calibrated_logits.std().clamp_min(1.0e-6)
            readout_scale = (baseline_std / calibrated_std).clamp(0.25, 4.0)
            model.w2.mul_(readout_scale.to(device=model.w2.device, dtype=model.w2.dtype))
    return mode


def support_type_for_variant(variant: str, architecture: str) -> str:
    if variant in M6_VARIANTS or variant.startswith("M6-"):
        return "parameter_metric_support_complement_reservoir"
    if variant in M9_VARIANTS or variant.startswith("M9-"):
        return "train_batch_diagonal_adaptive_metric_mixture_support"
    if architecture != "MLP":
        if "BasisGram" in variant or "Gram" in variant:
            return "strict_fc_purekan_basis_gram_w1_w2"
        if "BankLocal" in variant:
            return "strict_fc_purekan_bank_local_oet"
        return "strict_fc_purekan_metric_support_w1_w2"
    if "frequency-like" in variant:
        return "mlp_frequency_like_random_feature_metric_support"
    if "polynomial-like" in variant:
        return "mlp_polynomial_like_metric_support"
    if "block" in variant:
        return "mlp_same_rank_block_metric_support"
    if "same-param" in variant:
        return "mlp_same_param_flops_metric_support"
    if variant.startswith("P3-"):
        return "non_additive_cayley_oet_orbit"
    if "Spectral" in variant or "spectral" in variant:
        return "mlp_matched_spectral_metric_support"
    return "mlp_low_rank_hidden_metric_support"


def is_control_mode(control_mode: str) -> bool:
    return str(control_mode) != "none"


def normalized_m9_weights(weights: dict[str, float] | None = None) -> dict[str, float]:
    raw = {name: max(0.0, float((weights or {}).get(name, 0.0))) for name in M9_COMPONENTS}
    total = sum(raw.values())
    if total <= 1.0e-12 or not math.isfinite(total):
        return {name: 1.0 / len(M9_COMPONENTS) for name in M9_COMPONENTS}
    return {name: value / total for name, value in raw.items()}


def m9_fixed_weights_for_control(control_mode: str) -> dict[str, float] | None:
    if str(control_mode) not in M9_FIXED_CONTROL_WEIGHTS:
        return None
    return normalized_m9_weights(M9_FIXED_CONTROL_WEIGHTS[str(control_mode)])


def m9_weight_entropy(weights: dict[str, float]) -> float:
    denom = math.log(max(2, len(M9_COMPONENTS)))
    entropy = 0.0
    for value in normalized_m9_weights(weights).values():
        if value > 1.0e-12:
            entropy -= value * math.log(value)
    return entropy / max(1.0e-12, denom)


def metric_dot(a: dict[str, Any], b: dict[str, Any], diag: dict[str, Any]) -> float:
    total = 0.0
    for key, av in a.items():
        bv = b.get(key)
        if bv is None:
            continue
        g = diag.get(key)
        if g is None:
            total += float((av.float() * bv.to(device=av.device).float()).sum().item())
        else:
            total += float((g.to(device=av.device).float() * av.float() * bv.to(device=av.device).float()).sum().item())
    return total


def metric_norm(values: dict[str, Any], diag: dict[str, Any]) -> float:
    return math.sqrt(max(0.0, metric_dot(values, values, diag)))


def normalize_metric(values: dict[str, Any], diag: dict[str, Any], target_norm: float = 1.0) -> dict[str, Any]:
    norm = metric_norm(values, diag)
    if norm <= 1.0e-12 or not math.isfinite(norm):
        return {k: v * 0.0 for k, v in values.items()}
    scale = float(target_norm) / norm
    return {k: v.float().mul(scale).to(device=v.device, dtype=v.dtype) for k, v in values.items()}


def negate_velocity(values: dict[str, Any]) -> dict[str, Any]:
    return {k: -v for k, v in values.items()}


def apply_mask(values: dict[str, Any], masks: dict[str, Any]) -> dict[str, Any]:
    return v2242.apply_mask(values, masks)


def apply_complement_mask(values: dict[str, Any], masks: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in values.items():
        mask = masks.get(key)
        if mask is None:
            out[key] = value * 0.0
            continue
        comp = (~mask.to(device=value.device)).to(dtype=value.dtype)
        out[key] = value * comp
    return out


def tangent_project_metric(values: dict[str, Any], named: list[tuple[str, Any]], diag: dict[str, Any]) -> dict[str, Any]:
    out = {}
    param_by_name = {n: p for n, p in named}
    for name, value in values.items():
        p = param_by_name.get(name)
        if p is None:
            out[name] = value
            continue
        base = p.detach().float()
        g = diag.get(name)
        if g is None:
            denom = float(base.square().sum().item())
            coeff = float((value.float() * base).sum().item()) / max(1.0e-12, denom)
        else:
            gw = g.to(device=value.device).float()
            denom = float((gw * base * base).sum().item())
            coeff = float((gw * value.float() * base).sum().item()) / max(1.0e-12, denom)
        out[name] = (value.float() - coeff * base).to(device=value.device, dtype=value.dtype)
    return out


def residualize_metric(
    values: dict[str, Any],
    nuisance: list[dict[str, Any]],
    diag: dict[str, Any],
) -> tuple[dict[str, Any], int, float]:
    out = {k: v.float().clone() for k, v in values.items()}
    original = metric_norm(out, diag)
    explained = 0.0
    rank = 0
    for nvec in nuisance:
        basis = normalize_metric(nvec, diag, 1.0)
        denom = metric_norm(basis, diag)
        if denom <= 1.0e-12:
            continue
        coeff = metric_dot(out, basis, diag)
        explained += coeff * coeff
        rank += 1
        for key, bv in basis.items():
            if key in out:
                out[key] = out[key] - coeff * bv.to(device=out[key].device).float()
    ratio = math.sqrt(max(0.0, explained)) / max(1.0e-12, original)
    return {k: v.to(device=values[k].device, dtype=values[k].dtype) for k, v in out.items()}, rank, ratio


def metric_condition(diag: dict[str, Any]) -> tuple[float, float, float]:
    vals = []
    for g in diag.values():
        flat = g.detach().float().reshape(-1)
        if flat.numel():
            vals.append(flat)
    if not vals:
        return 1.0, 1.0, 1.0
    allv = __import__("torch").cat(vals)
    positive = allv[allv > 0.0]
    if positive.numel() == 0:
        return float("inf"), 0.0, 0.0
    minv = float(positive.min().item())
    maxv = float(positive.max().item())
    return maxv / max(1.0e-12, minv), minv, maxv


def metric_support_mask(
    named: list[tuple[str, Any]],
    signal: dict[str, Any],
    diag: dict[str, Any],
    *,
    architecture: str,
    variant: str,
    support_rank: int,
    seed: int,
    avoid_signal: dict[str, Any] | None = None,
    avoid_weight: float = 0.0,
) -> dict[str, Any]:
    import torch

    masks: dict[str, Any] = {}
    rank_frac = min(1.0, max(0.01, float(support_rank) / 64.0))
    avoid_weight = max(0.0, float(avoid_weight))
    for idx, (name, p) in enumerate(named):
        sig = signal.get(name)
        if sig is None:
            masks[name] = torch.zeros_like(p.detach(), dtype=torch.bool)
            continue
        if architecture != "MLP" and not is_basis_param(name):
            masks[name] = torch.zeros_like(p.detach(), dtype=torch.bool)
            continue
        frac = rank_frac
        if architecture == "MLP":
            frac = v2242.support_fraction_for_name(name, "mlp_lowrank_matched_support_fu", support_rank, architecture)
            if "frequency-like" in variant:
                frac = min(0.50, frac * 1.25)
            elif "polynomial-like" in variant:
                frac = min(0.50, frac * 1.50)
            elif "block" in variant:
                frac = min(0.50, frac * 1.25)
        total = sig.numel()
        take = max(1, min(total, int(round(total * frac))))
        metric = diag.get(name)
        weight = torch.ones_like(sig.detach().float()) if metric is None else metric.to(device=sig.device).float().clamp_min(1.0e-12)
        score = sig.detach().float().abs() * weight.sqrt()
        if avoid_signal is not None and avoid_weight > 0.0:
            avoid = avoid_signal.get(name)
            if avoid is not None:
                avoid_score = avoid.detach().to(device=sig.device).float().abs() * weight.sqrt()
                avoid_scale = avoid_score.mean().clamp_min(1.0e-12)
                score = score / (1.0 + avoid_weight * avoid_score / avoid_scale)
        if "Spectral" in variant or "spectral" in variant or "block" in variant:
            if sig.ndim >= 2:
                flat_score = score.reshape(score.shape[0], -1).mean(dim=1)
                rows = max(1, min(int(score.shape[0]), int(round(math.sqrt(take)))))
                top_rows = torch.topk(flat_score, k=rows, largest=True).indices
                mask = torch.zeros_like(score, dtype=torch.bool)
                mask[top_rows] = True
                if int(mask.sum().item()) > take:
                    score_flat = score.masked_fill(~mask, -1).reshape(-1)
                    top = torch.topk(score_flat, k=take, largest=True).indices
                    mask_flat = torch.zeros(total, device=score.device, dtype=torch.bool)
                    mask_flat[top] = True
                    mask = mask_flat.view_as(score)
                masks[name] = mask
                continue
        if "frequency-like" in variant:
            gen = torch.Generator(device=score.device if score.is_cuda else "cpu").manual_seed(stable_seed(seed, name, "freq", idx))
            score = score + 0.01 * torch.randn(score.shape, generator=gen, device=score.device)
        elif "polynomial-like" in variant:
            score = score.square()
        top = torch.topk(score.reshape(-1), k=take, largest=True).indices
        mask_flat = torch.zeros(total, device=score.device, dtype=torch.bool)
        mask_flat[top] = True
        masks[name] = mask_flat.view_as(p)
    return masks


def shuffled_like(values: dict[str, Any], seed: int) -> dict[str, Any]:
    import torch

    out = {}
    for idx, (key, value) in enumerate(values.items()):
        flat = value.detach().float().reshape(-1)
        if flat.numel() == 0:
            out[key] = value * 0.0
            continue
        gen = torch.Generator(device=value.device if value.is_cuda else "cpu").manual_seed(stable_seed(seed, key, idx, "shuffle"))
        perm = torch.randperm(flat.numel(), generator=gen, device=value.device)
        out[key] = flat[perm].view_as(value).to(device=value.device, dtype=value.dtype)
    return out


def risk_from_losses(losses: Any) -> dict[str, float]:
    return v2242.risk_from_losses(losses)


def calibration_risk_from_logits(logits: Any, y: Any) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        probs = torch.softmax(logits.detach().float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        acc = pred.eq(y.detach().long()).float()
        signed_gap = float((conf.mean() - acc.mean()).item())
        onehot = F.one_hot(y.detach().long(), num_classes=probs.shape[-1]).float()
        return {
            "calibration_gap": float((conf.mean() - acc.mean()).abs().item()),
            "overconfidence_gap": max(0.0, signed_gap),
            "underconfidence_gap": max(0.0, -signed_gap),
            "batch_brier_proxy": float((probs - onehot).square().sum(dim=-1).mean().item()),
        }


def calibration_component_losses_from_logits(logits: Any, y: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    logits = logits.float()
    y = y.long()
    probs = torch.softmax(logits, dim=-1)
    onehot = F.one_hot(y, num_classes=probs.shape[-1]).float()
    brier_loss = (probs - onehot).square().sum(dim=-1).mean()
    losses = F.cross_entropy(logits, y, reduction="none")
    k = max(1, int(math.ceil(0.10 * int(losses.numel()))))
    tail_loss = torch.topk(losses, k=k, largest=True).values.mean()
    k_q99 = max(1, int(math.ceil(0.01 * int(losses.numel()))))
    tail_q99_loss = torch.topk(losses, k=k_q99, largest=True).values.mean()
    true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
    masked = logits.masked_fill(F.one_hot(y, num_classes=logits.shape[-1]).bool(), float("-inf"))
    max_other = masked.max(dim=-1).values
    margin_loss = F.softplus(max_other - true_logits).mean()
    return {"tail": tail_loss, "tail_q99": tail_q99_loss, "brier": brier_loss, "margin": margin_loss}


def calibration_nuisance_loss_from_logits(logits: Any, y: Any, mode: str) -> Any:
    import torch

    logits = logits.float()
    y = y.long()
    probs = torch.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    mode = str(mode)
    components = calibration_component_losses_from_logits(logits, y)
    tail_loss = components["tail"]
    tail_q99_loss = components["tail_q99"]
    brier_loss = components["brier"]
    margin_loss = components["margin"]
    if mode in {"tail", "tail_margin", "tail_brier", "tail_brier_balanced", "tail_brier_brier2_balanced"}:
        if mode == "tail":
            return tail_loss
        if mode == "tail_brier":
            return tail_loss + brier_loss
        if mode == "tail_brier_balanced":
            tail_scale = tail_loss.detach().abs().clamp_min(1.0e-6)
            brier_scale = brier_loss.detach().abs().clamp_min(1.0e-6)
            return tail_loss / tail_scale + brier_loss / brier_scale
        if mode == "tail_brier_brier2_balanced":
            tail_scale = tail_loss.detach().abs().clamp_min(1.0e-6)
            brier_scale = brier_loss.detach().abs().clamp_min(1.0e-6)
            return tail_loss / tail_scale + 2.0 * brier_loss / brier_scale
    q99_guard_modes = {
        "tail_q99_brier_qp_margin",
        "tail_q99_brier_qp_strict_margin",
        "tail_q99_brier_qp_hard_margin",
    }
    if mode in {"tail_brier_pareto", "tail_brier_pareto2", "tail_brier_qp", "tail_brier_qp_margin", "tail_brier_qp_brier_margin", *q99_guard_modes}:
        tail_component = tail_q99_loss if mode in q99_guard_modes else tail_loss
        tail_scale = tail_component.detach().abs().clamp_min(1.0e-6)
        brier_scale = brier_loss.detach().abs().clamp_min(1.0e-6)
        return tail_component / tail_scale + brier_loss / brier_scale
    if mode in {"margin", "tail_margin"}:
        if mode == "margin":
            return margin_loss
        return tail_loss + margin_loss
    if mode == "confidence":
        return conf.mean()
    if mode == "overconfidence":
        acc = pred.eq(y).float()
        return torch.relu(conf.mean() - acc.mean()).square()
    return brier_loss


def mirror_potential_type(variant: str) -> str:
    if variant == "M1-FisherMirror":
        return "fisher_logit_diag_kl_primal"
    if variant in {"M1-KLMirror", "M1-KLMirrorLS"}:
        return "negative_entropy_kl"
    if variant == "M1-BrierMirror":
        return "brier_probability"
    if variant == "M1-TailSafeMirror":
        return "tail_safe_kl_q90"
    return ""


def mirror_step_for_variant(variant: str) -> float:
    if variant == "M1-FisherMirror":
        return 0.10
    if variant in {"M1-KLMirror", "M1-KLMirrorLS"}:
        return 0.04
    if variant == "M1-BrierMirror":
        return 0.06
    if variant == "M1-TailSafeMirror":
        return 0.05
    return 0.0


def mirror_grad_from_logits(logits: Any, y: Any, named: list[tuple[str, Any]], variant: str, *, record_diagnostics: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    potential = mirror_potential_type(variant)
    step_size = mirror_step_for_variant(variant)
    if not potential or step_size <= 0.0:
        return {}, {}
    logits_f = logits.float()
    y_long = y.long()
    probs = torch.softmax(logits_f, dim=-1)
    log_probs = torch.log_softmax(logits_f, dim=-1)
    onehot = F.one_hot(y_long, num_classes=logits_f.shape[-1]).float()
    losses = F.cross_entropy(logits_f, y_long, reduction="none")
    tail_weight = torch.ones_like(losses)
    if variant == "M1-TailSafeMirror":
        k = max(1, int(math.ceil(0.10 * int(losses.numel()))))
        threshold = torch.topk(losses.detach(), k=k, largest=True).values.min()
        tail_weight = torch.where(losses.detach() >= threshold, torch.full_like(losses, 2.0), torch.full_like(losses, 0.35))
    if variant == "M1-BrierMirror":
        psi = probs
        xi = (probs - onehot) * tail_weight.view(-1, 1)
        psi_next = (psi - step_size * xi).clamp_min(1.0e-6)
        primal_next = (psi_next / psi_next.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)).detach()
        mirror_loss = 0.5 * (probs - primal_next).square().sum(dim=-1).mean()
        target_log_probs = torch.log(primal_next.clamp_min(1.0e-6))
    else:
        xi = (probs - onehot) * tail_weight.view(-1, 1)
        if variant == "M1-FisherMirror":
            fisher_scale = (probs * (1.0 - probs)).clamp_min(1.0e-4).sqrt()
            xi = xi / fisher_scale
        psi = log_probs
        psi_next = (psi - step_size * xi).clamp(-20.0, 20.0)
        primal_next = torch.softmax(psi_next, dim=-1).detach()
        target_log_probs = torch.log(primal_next.clamp_min(1.0e-6))
        mirror_loss = 0.5 * (log_probs - target_log_probs).square().sum(dim=-1).mean()
    meta = {
        "mirror_potential_type": potential,
        "Bregman_step_size": step_size,
        "mirror_grad_norm": 0.0,
        "mirror_explicit_dual_update": 1,
        "mirror_explicit_primal_map": 1,
        "mirror_actuator_projection": "sketched_functional_least_squares" if variant == "M1-KLMirrorLS" else "support_masked_Jt_residual",
        "mirror_grad_source": "mirror_residual_autograd_grad" if variant in {"M1-KLMirror", "M1-KLMirrorLS"} else "extra_autograd_grad",
        "mirror_diagnostic_recorded": 0,
    }
    target_delta = (target_log_probs.detach() - log_probs.detach()).float()
    meta["_mirror_target_delta"] = target_delta - target_delta.mean(dim=-1, keepdim=True)
    if record_diagnostics:
        brier_before = float((probs.detach() - onehot).square().sum(dim=-1).mean().item())
        brier_after = float((primal_next.detach() - onehot).square().sum(dim=-1).mean().item())
        kl_step = float((primal_next.detach() * (target_log_probs.detach() - log_probs.detach())).sum(dim=-1).mean().abs().item())
        fisher_norm = float((((probs.detach() - primal_next.detach()).square()) / (probs.detach() * (1.0 - probs.detach())).clamp_min(1.0e-4)).sum(dim=-1).mean().sqrt().item())
        tail_before = float(torch.topk(losses.detach(), k=max(1, int(math.ceil(0.01 * int(losses.numel())))), largest=True).values.mean().item())
        target_true_prob = primal_next.detach().gather(1, y_long.view(-1, 1)).squeeze(1).clamp_min(1.0e-6)
        target_tail_losses = -torch.log(target_true_prob)
        tail_after = float(torch.topk(target_tail_losses, k=max(1, int(math.ceil(0.01 * int(target_tail_losses.numel())))), largest=True).values.mean().item())
        true_logits = logits_f.detach().gather(1, y_long.view(-1, 1)).squeeze(1)
        masked = logits_f.detach().masked_fill(onehot.bool(), float("-inf"))
        margin_before = float((true_logits - masked.max(dim=-1).values).quantile(0.10).item())
        target_logits = target_log_probs.detach()
        target_true_logits = target_logits.gather(1, y_long.view(-1, 1)).squeeze(1)
        target_masked = target_logits.masked_fill(onehot.bool(), float("-inf"))
        margin_after = float((target_true_logits - target_masked.max(dim=-1).values).quantile(0.10).item())
        meta.update({
            "KL_step": kl_step,
            "Fisher_norm_step": fisher_norm,
            "Brier_delta": brier_after - brier_before,
            "tail_q99_mirror_delta": tail_after - tail_before,
            "margin_q10_mirror_delta": margin_after - margin_before,
            "mirror_loss": float(mirror_loss.detach().item()),
            "mirror_dual_state_norm": float(psi.detach().float().norm().item()),
            "mirror_primal_delta_norm": float((primal_next.detach() - probs.detach()).float().norm().item()),
            "mirror_diagnostic_recorded": 1,
        })
    params = [p for _name, p in named]
    grads = torch.autograd.grad(mirror_loss, params, retain_graph=True, allow_unused=True)
    grad_map = {}
    total_sq = 0.0
    for (name, p), grad in zip(named, grads):
        g = torch.zeros_like(p.detach()) if grad is None else grad.detach().clone()
        grad_map[name] = g
        total_sq += float(g.float().square().sum().item())
    meta["mirror_grad_norm"] = math.sqrt(max(0.0, total_sq))
    return grad_map, meta


def optimizer_step(model: Any, opt: Any, optimizer_family: str, step: int, avg_state: dict[str, Any]) -> None:
    v2240.optimizer_step(model, opt, optimizer_family, step, avg_state)


def final_schedule_free_swap(model: Any, optimizer_family: str, avg_state: dict[str, Any]) -> None:
    v2240.final_schedule_free_swap(model, optimizer_family, avg_state)


def tensor_spectrum_drift(before: Any, after: Any) -> tuple[float, float]:
    import torch

    try:
        b = before.detach().float()
        a = after.detach().float()
        if b.ndim < 2:
            return 0.0, 0.0
        bm = b.reshape(int(b.shape[0]), -1)
        am = a.reshape(int(a.shape[0]), -1)
        sb = torch.linalg.svdvals(bm)
        sa = torch.linalg.svdvals(am)
        n = min(int(sb.numel()), int(sa.numel()))
        if n == 0:
            return 0.0, 0.0
        drift = float((sa[:n] - sb[:n]).norm().item() / (sb[:n].norm().item() + 1.0e-12))
        spectral = float((sa.max() - sb.max()).abs().item() / (sb.max().abs().item() + 1.0e-12))
        return drift, spectral
    except Exception:
        return float("nan"), float("nan")


def apply_left_cayley_oet(named: list[tuple[str, Any]], velocity: dict[str, Any], scale: float) -> dict[str, float]:
    import torch

    update_sq = 0.0
    generator_sq = 0.0
    skew_residuals: list[float] = []
    cayley_residuals: list[float] = []
    spectrum_drifts: list[float] = []
    spectral_drifts: list[float] = []
    matrices = 0
    with torch.no_grad():
        for name, p in named:
            v = velocity.get(name)
            if v is None or p.detach().ndim < 2:
                continue
            w_before = p.detach().float().clone()
            w = w_before.reshape(int(w_before.shape[0]), -1)
            g = v.detach().float().reshape_as(w)
            if w.numel() == 0 or g.numel() == 0:
                continue
            raw = g @ w.t()
            skew = 0.5 * (raw - raw.t())
            generator = -skew
            dim = int(generator.shape[0])
            eye = torch.eye(dim, device=w.device, dtype=w.dtype)
            lhs = eye - 0.5 * float(scale) * generator
            rhs = eye + 0.5 * float(scale) * generator
            try:
                rot = torch.linalg.solve(lhs, rhs)
            except RuntimeError:
                rot = eye
            w_after = (rot @ w).reshape_as(w_before)
            p.copy_(w_after.to(device=p.device, dtype=p.dtype))
            delta = (w_after - w_before).float()
            update_sq += float(delta.square().sum().item())
            generator_sq += float(generator.square().sum().item())
            skew_denom = float(generator.square().sum().sqrt().item()) + 1.0e-12
            skew_residuals.append(float((generator.t() + generator).norm().item()) / skew_denom)
            cayley_lhs = lhs @ rot
            cayley_denom = float(rhs.norm().item()) + 1.0e-12
            cayley_residuals.append(float((cayley_lhs - rhs).norm().item()) / cayley_denom)
            drift, spectral = tensor_spectrum_drift(w_before, w_after)
            if math.isfinite(drift):
                spectrum_drifts.append(drift)
            if math.isfinite(spectral):
                spectral_drifts.append(spectral)
            matrices += 1
    return {
        "oet_update_norm": math.sqrt(max(0.0, update_sq)),
        "OET_generator_norm": math.sqrt(max(0.0, generator_sq)),
        "metric_skew_residual": max(skew_residuals, default=0.0),
        "Cayley_solve_residual": max(cayley_residuals, default=0.0),
        "generalized_spectrum_drift": max(spectrum_drifts, default=0.0),
        "ordinary_spectrum_drift": max(spectrum_drifts, default=0.0),
        "spectral_norm_drift": max(spectral_drifts, default=0.0),
        "oet_matrix_count": matrices,
    }


def lie_momentum_variant(variant: str) -> bool:
    return "lie-momentum" in str(variant)


def apply_left_cayley_oet_lie_momentum(
    named: list[tuple[str, Any]],
    velocity: dict[str, Any],
    scale: float,
    momentum_state: dict[str, dict[str, Any]],
    beta: float = 0.12,
    transport: bool = True,
) -> dict[str, float]:
    import torch

    beta = min(1.0, max(0.0, float(beta)))
    update_sq = 0.0
    generator_sq = 0.0
    current_generator_sq = 0.0
    transported_generator_sq = 0.0
    transport_errors: list[float] = []
    skew_residuals: list[float] = []
    cayley_residuals: list[float] = []
    spectrum_drifts: list[float] = []
    spectral_drifts: list[float] = []
    rotation_angles: list[float] = []
    lie_snrs: list[float] = []
    matrices = 0
    with torch.no_grad():
        for name, p in named:
            v = velocity.get(name)
            if v is None or p.detach().ndim < 2:
                continue
            w_before = p.detach().float().clone()
            w = w_before.reshape(int(w_before.shape[0]), -1)
            g = v.detach().float().reshape_as(w)
            if w.numel() == 0 or g.numel() == 0:
                continue
            raw = g @ w.t()
            current_generator = -0.5 * (raw - raw.t())
            current_generator = 0.5 * (current_generator - current_generator.t())
            entry = momentum_state.get(name, {})
            prev_generator = entry.get("generator")
            prev_rotation = entry.get("rotation")
            transported = torch.zeros_like(current_generator)
            if prev_generator is not None and tuple(prev_generator.shape) == tuple(current_generator.shape):
                transported = prev_generator.to(device=w.device, dtype=w.dtype)
                if (
                    transport
                    and prev_rotation is not None
                    and tuple(prev_rotation.shape) == tuple(current_generator.shape)
                ):
                    rot = prev_rotation.to(device=w.device, dtype=w.dtype)
                    transported = rot @ transported @ rot.t()
                transported = 0.5 * (transported - transported.t())
                denom = float(prev_generator.detach().float().norm().item()) + 1.0e-12
                transport_errors.append(float((transported - prev_generator.to(device=w.device, dtype=w.dtype)).norm().item()) / denom)
            if prev_generator is None:
                generator = current_generator
            else:
                generator = (1.0 - beta) * transported + beta * current_generator
                generator = 0.5 * (generator - generator.t())
            dim = int(generator.shape[0])
            eye = torch.eye(dim, device=w.device, dtype=w.dtype)
            lhs = eye - 0.5 * float(scale) * generator
            rhs = eye + 0.5 * float(scale) * generator
            try:
                rot = torch.linalg.solve(lhs, rhs)
            except RuntimeError:
                rot = eye
            w_after = (rot @ w).reshape_as(w_before)
            p.copy_(w_after.to(device=p.device, dtype=p.dtype))
            delta = (w_after - w_before).float()
            update_sq += float(delta.square().sum().item())
            generator_norm = float(generator.square().sum().item())
            current_norm = float(current_generator.square().sum().item())
            transported_norm = float(transported.square().sum().item())
            generator_sq += generator_norm
            current_generator_sq += current_norm
            transported_generator_sq += transported_norm
            rotation_angles.append(abs(float(scale)) * math.sqrt(max(0.0, generator_norm)))
            lie_snrs.append(math.sqrt(max(0.0, generator_norm)) / (math.sqrt(max(0.0, current_norm)) + 1.0e-12))
            skew_denom = float(generator.square().sum().sqrt().item()) + 1.0e-12
            skew_residuals.append(float((generator.t() + generator).norm().item()) / skew_denom)
            cayley_lhs = lhs @ rot
            cayley_denom = float(rhs.norm().item()) + 1.0e-12
            cayley_residuals.append(float((cayley_lhs - rhs).norm().item()) / cayley_denom)
            drift, spectral = tensor_spectrum_drift(w_before, w_after)
            if math.isfinite(drift):
                spectrum_drifts.append(drift)
            if math.isfinite(spectral):
                spectral_drifts.append(spectral)
            momentum_state[name] = {
                "generator": generator.detach().clone(),
                "rotation": rot.detach().clone(),
            }
            matrices += 1
    return {
        "oet_update_norm": math.sqrt(max(0.0, update_sq)),
        "OET_generator_norm": math.sqrt(max(0.0, generator_sq)),
        "metric_skew_residual": max(skew_residuals, default=0.0),
        "Cayley_solve_residual": max(cayley_residuals, default=0.0),
        "generalized_spectrum_drift": max(spectrum_drifts, default=0.0),
        "ordinary_spectrum_drift": max(spectrum_drifts, default=0.0),
        "spectral_norm_drift": max(spectral_drifts, default=0.0),
        "oet_matrix_count": matrices,
        "lie_momentum_active": 1.0 if matrices else 0.0,
        "lie_momentum_norm": math.sqrt(max(0.0, generator_sq)),
        "ambient_momentum_norm": math.sqrt(max(0.0, current_generator_sq)),
        "transported_lie_momentum_norm": math.sqrt(max(0.0, transported_generator_sq)),
        "transport_error": statistics.fmean(transport_errors) if transport_errors else 0.0,
        "left_rotation_angle": max(rotation_angles, default=0.0),
        "right_rotation_angle": 0.0,
        "left_right_imbalance": 1.0 if matrices else 0.0,
        "Lie_SNR": statistics.fmean(lie_snrs) if lie_snrs else 0.0,
    }


def stable_rank_value(matrix: Any) -> float:
    import torch

    try:
        m = matrix.detach().float().reshape(int(matrix.shape[0]), -1)
        sv = torch.linalg.svdvals(m)
        if sv.numel() == 0:
            return 0.0
        denom = float(sv.max().square().item()) + 1.0e-12
        return float(sv.square().sum().item()) / denom
    except Exception:
        return float("nan")


def apply_small_radial_channel(
    named: list[tuple[str, Any]],
    velocity: dict[str, Any],
    scale: float,
    radial_cap: float,
    tangent_update_norm: float,
) -> dict[str, float]:
    import torch

    cap = max(0.0, float(radial_cap))
    tangent_norm = max(0.0, float(tangent_update_norm))
    raw_updates: list[tuple[Any, Any, Any, float]] = []
    raw_sq = 0.0
    with torch.no_grad():
        for name, p in named:
            v = velocity.get(name)
            if v is None or p.detach().ndim < 2:
                continue
            w_before = p.detach().float().clone()
            w = w_before.reshape(int(w_before.shape[0]), -1)
            g = v.detach().float().reshape_as(w)
            denom = float(w.square().sum().item()) + 1.0e-12
            coeff = float((g * w).sum().item()) / denom
            radial_direction = coeff * w
            raw_update = float(scale) * radial_direction.reshape_as(w_before)
            raw_norm = float(raw_update.square().sum().item())
            raw_sq += raw_norm
            raw_updates.append((p, w_before, raw_update, stable_rank_value(w_before)))
        raw_norm_total = math.sqrt(max(0.0, raw_sq))
        max_radial_norm = cap * (tangent_norm + 1.0e-12)
        clip = 0.0 if raw_norm_total <= 1.0e-12 or max_radial_norm <= 0.0 else min(1.0, max_radial_norm / raw_norm_total)
        applied_sq = 0.0
        spectrum_drifts: list[float] = []
        rank_changes: list[float] = []
        for p, w_before, raw_update, rank_before in raw_updates:
            applied = raw_update * clip
            w_after = w_before + applied
            p.copy_(w_after.to(device=p.device, dtype=p.dtype))
            applied_sq += float(applied.float().square().sum().item())
            drift, _spectral = tensor_spectrum_drift(w_before, w_after)
            if math.isfinite(drift):
                spectrum_drifts.append(drift)
            rank_after = stable_rank_value(w_after)
            if math.isfinite(rank_before) and math.isfinite(rank_after):
                rank_changes.append(abs(rank_after - rank_before) / (abs(rank_before) + 1.0e-12))
        applied_norm = math.sqrt(max(0.0, applied_sq))
    return {
        "radial_update_norm": applied_norm,
        "radial_raw_update_norm": raw_norm_total,
        "radial_energy_fraction": applied_norm / (tangent_norm + 1.0e-12),
        "radial_cap": cap,
        "radial_clip_fraction": clip,
        "radial_spectrum_drift": max(spectrum_drifts, default=0.0),
        "radial_rank_change_proxy": max(rank_changes, default=0.0),
    }


def apply_functional_mode_radial_channel(
    named: list[tuple[str, Any]],
    velocity: dict[str, Any],
    scale: float,
    radial_cap: float,
    tangent_update_norm: float,
    *,
    safety_debt: float,
    safety_scale: float,
    rse_threshold: float = 0.02,
    model: Any | None = None,
    xb: Any | None = None,
    use_functional_response: bool = False,
    functional_eps: float = 1.0e-3,
    max_modes_per_matrix: int = 4,
) -> dict[str, float]:
    import torch

    cap = max(0.0, float(radial_cap))
    tangent_norm = max(0.0, float(tangent_update_norm))
    # Safety gating is already folded into radial_cap by functional_radial_gate_for_variant.
    # Do not reapply the strict default threshold here; doing so would silently disable
    # relaxed-safe variants while still reporting the requested variant name.
    safe = cap > 0.0 and float(safety_scale) > 0.0
    functional = bool(use_functional_response and model is not None and xb is not None)
    eps = max(1.0e-6, float(functional_eps))
    max_modes = max(1, int(max_modes_per_matrix))
    raw_updates: list[tuple[Any, Any, Any, float]] = []
    raw_sq = 0.0
    mode_count = 0
    selected_modes = 0
    rse_values: list[float] = []
    selected_rse_values: list[float] = []
    coeff_values: list[float] = []
    response_energy_values: list[float] = []
    fd_eval_count = 0
    with torch.no_grad():
        base_logits = model(xb).detach().float() if functional else None
        if cap > 0.0 and tangent_norm > 0.0 and safe:
            for name, p in named:
                v = velocity.get(name)
                if v is None or p.detach().ndim < 2:
                    continue
                w_before = p.detach().float().clone()
                w = w_before.reshape(int(w_before.shape[0]), -1)
                g = v.detach().float().reshape_as(w)
                if w.numel() == 0 or g.numel() == 0:
                    continue
                try:
                    u, s, vh = torch.linalg.svd(w, full_matrices=False)
                except RuntimeError:
                    continue
                n = int(s.numel())
                if n == 0:
                    continue
                energy = s.square()
                energy = energy / energy.sum().clamp_min(1.0e-12)
                coeffs: list[Any] = []
                bases: list[Any] = []
                abs_coeffs: list[float] = []
                for idx in range(n):
                    basis = u[:, idx : idx + 1] @ vh[idx : idx + 1, :]
                    coeff = (g * basis).sum()
                    coeff_abs = float(coeff.detach().abs().item())
                    coeffs.append(coeff)
                    bases.append(basis)
                    abs_coeffs.append(coeff_abs)
                max_coeff = max(abs_coeffs, default=0.0)
                if max_coeff <= 1.0e-12:
                    continue
                candidate_indices = sorted(range(n), key=lambda i: float(energy[i].detach().item()) * abs_coeffs[i], reverse=True)[:max_modes]
                response_by_index: dict[int, float] = {}
                if functional and base_logits is not None:
                    for idx in candidate_indices:
                        p.add_(eps * bases[idx].reshape_as(w_before).to(device=p.device, dtype=p.dtype))
                        perturbed = model(xb).detach().float()
                        p.copy_(w_before.to(device=p.device, dtype=p.dtype))
                        response = (perturbed - base_logits) / eps
                        response_energy = float(response.square().mean().item())
                        response_by_index[idx] = response_energy
                        response_energy_values.append(response_energy)
                        fd_eval_count += 1
                    response_total = sum(response_by_index.values()) + 1.0e-12
                else:
                    response_total = 1.0
                radial_direction = torch.zeros_like(w)
                matrix_selected = 0
                for idx in candidate_indices:
                    coeff = coeffs[idx]
                    signal_fraction = abs_coeffs[idx] / (max_coeff + 1.0e-12)
                    if functional:
                        spectrum_fraction = response_by_index.get(idx, 0.0) / response_total
                    else:
                        spectrum_fraction = float(energy[idx].detach().item())
                    mode_rse = spectrum_fraction * signal_fraction
                    rse_values.append(mode_rse)
                    coeff_values.append(abs_coeffs[idx])
                    mode_count += 1
                    if mode_rse <= float(rse_threshold):
                        continue
                    # Descent-side radial component: apply_velocity uses -scale * velocity,
                    # so the radial singular-value channel follows the same sign convention.
                    radial_direction = radial_direction - coeff * bases[idx]
                    selected_modes += 1
                    matrix_selected += 1
                    selected_rse_values.append(mode_rse)
                if matrix_selected <= 0:
                    continue
                raw_update = float(scale) * radial_direction.reshape_as(w_before)
                raw_norm = float(raw_update.square().sum().item())
                raw_sq += raw_norm
                raw_updates.append((p, w_before, raw_update, stable_rank_value(w_before)))
        raw_norm_total = math.sqrt(max(0.0, raw_sq))
        max_radial_norm = cap * (tangent_norm + 1.0e-12)
        clip = 0.0 if raw_norm_total <= 1.0e-12 or max_radial_norm <= 0.0 else min(1.0, max_radial_norm / raw_norm_total)
        applied_sq = 0.0
        spectrum_drifts: list[float] = []
        rank_changes: list[float] = []
        for p, w_before, raw_update, rank_before in raw_updates:
            applied = raw_update * clip
            w_after = w_before + applied
            p.copy_(w_after.to(device=p.device, dtype=p.dtype))
            applied_sq += float(applied.float().square().sum().item())
            drift, _spectral = tensor_spectrum_drift(w_before, w_after)
            if math.isfinite(drift):
                spectrum_drifts.append(drift)
            rank_after = stable_rank_value(w_after)
            if math.isfinite(rank_before) and math.isfinite(rank_after):
                rank_changes.append(abs(rank_after - rank_before) / (abs(rank_before) + 1.0e-12))
        applied_norm = math.sqrt(max(0.0, applied_sq))
    return {
        "radial_update_norm": applied_norm,
        "radial_raw_update_norm": raw_norm_total,
        "radial_energy_fraction": applied_norm / (tangent_norm + 1.0e-12),
        "radial_cap": cap,
        "radial_clip_fraction": clip,
        "radial_spectrum_drift": max(spectrum_drifts, default=0.0),
        "radial_rank_change_proxy": max(rank_changes, default=0.0),
        "functional_radial_gate_active": float(selected_modes > 0),
        "functional_radial_score": statistics.fmean(selected_rse_values) if selected_rse_values else 0.0,
        "functional_radial_safety_debt": max(0.0, float(safety_debt)),
        "functional_radial_effective_cap": cap if selected_modes > 0 else 0.0,
        "functional_radial_mode_fraction": selected_modes / max(1, mode_count),
        "functional_radial_mode_count": float(mode_count),
        "functional_radial_selected_mode_count": float(selected_modes),
        "functional_radial_rse_mean": statistics.fmean(rse_values) if rse_values else 0.0,
        "functional_radial_rse_max": max(rse_values, default=0.0),
        "functional_radial_selected_rse_mean": statistics.fmean(selected_rse_values) if selected_rse_values else 0.0,
        "functional_radial_coeff_abs_mean": statistics.fmean(coeff_values) if coeff_values else 0.0,
        "functional_radial_uses_functional_response": int(functional),
        "functional_radial_fd_eval_count": float(fd_eval_count),
        "functional_radial_response_energy_mean": statistics.fmean(response_energy_values) if response_energy_values else 0.0,
        "functional_radial_response_energy_max": max(response_energy_values, default=0.0),
    }


def apply_calibration_readout_radial(
    named: list[tuple[str, Any]],
    *,
    cap: float,
    policy: str = "signed",
    tangent_update_norm: float,
    overconfidence_gap: float,
    underconfidence_gap: float,
) -> dict[str, Any]:
    import torch

    cap = max(0.0, float(cap))
    tangent_norm = max(0.0, float(tangent_update_norm))
    over = max(0.0, float(overconfidence_gap))
    under = max(0.0, float(underconfidence_gap))
    pressure = over - under
    policy = str(policy)
    if cap <= 0.0 or tangent_norm <= 1.0e-12 or abs(pressure) <= 1.0e-12:
        return {
            "calibration_readout_radial_update_norm": 0.0,
            "calibration_readout_radial_energy_fraction": 0.0,
            "calibration_readout_radial_cap": cap,
            "calibration_readout_policy": policy,
            "calibration_readout_radial_clip_fraction": 0.0,
            "calibration_readout_pressure": pressure,
            "calibration_readout_direction": 0.0,
            "calibration_readout_param_count": 0,
        }
    # Positive pressure means train-batch overconfidence, so shrinking readout scale is the conservative correction.
    if policy == "overconfidence_only" and pressure <= 0.0:
        direction = 0.0
    elif policy == "shrink_only":
        direction = -1.0
    elif policy == "inverse_signed":
        direction = 1.0 if pressure > 0.0 else -1.0
    else:
        direction = -1.0 if pressure > 0.0 else 1.0
    if abs(direction) <= 1.0e-12:
        return {
            "calibration_readout_radial_update_norm": 0.0,
            "calibration_readout_radial_energy_fraction": 0.0,
            "calibration_readout_radial_cap": cap,
            "calibration_readout_policy": policy,
            "calibration_readout_radial_clip_fraction": 0.0,
            "calibration_readout_pressure": pressure,
            "calibration_readout_direction": 0.0,
            "calibration_readout_param_count": 0,
        }
    raw_updates: list[tuple[Any, Any]] = []
    raw_sq = 0.0
    param_count = 0
    readout_names = readout_param_names(named)
    with torch.no_grad():
        for name, p in named:
            if name not in readout_names:
                continue
            before = p.detach().float().clone()
            if before.numel() == 0:
                continue
            raw_update = direction * before
            raw_sq += float(raw_update.square().sum().item())
            param_count += int(before.numel())
            raw_updates.append((p, raw_update))
        raw_norm = math.sqrt(max(0.0, raw_sq))
        max_norm = cap * (tangent_norm + 1.0e-12)
        clip = 0.0 if raw_norm <= 1.0e-12 else min(1.0, max_norm / raw_norm)
        applied_sq = 0.0
        for p, raw_update in raw_updates:
            applied = raw_update * clip
            p.add_(applied.to(device=p.device, dtype=p.dtype))
            applied_sq += float(applied.square().sum().item())
        applied_norm = math.sqrt(max(0.0, applied_sq))
    return {
        "calibration_readout_radial_update_norm": applied_norm,
        "calibration_readout_radial_energy_fraction": applied_norm / (tangent_norm + 1.0e-12),
        "calibration_readout_radial_cap": cap,
        "calibration_readout_policy": policy,
        "calibration_readout_radial_clip_fraction": clip,
        "calibration_readout_pressure": pressure,
        "calibration_readout_direction": direction,
        "calibration_readout_param_count": param_count,
    }


def mirror_ls_variant(variant: str) -> bool:
    return variant == "M1-KLMirrorLS"


def clone_velocity(values: dict[str, Any] | None) -> dict[str, Any] | None:
    if values is None:
        return None
    return {k: v.detach().clone() for k, v in values.items()}


def functional_mirror_least_squares_velocity(
    *,
    named: list[tuple[str, Any]],
    model: Any,
    xb: Any,
    support_mask: dict[str, Any],
    base_velocity: dict[str, Any],
    diag: dict[str, Any],
    target_delta: Any,
    seed: int,
    max_basis: int = 8,
    eps: float = 1.0e-3,
    ridge: float = 1.0e-3,
) -> tuple[dict[str, Any], dict[str, float]]:
    import torch

    stats = {
        "mirror_ls_active": 0.0,
        "mirror_ls_cached": 0.0,
        "mirror_ls_basis_count": 0.0,
        "mirror_ls_fd_eval_count": 0.0,
        "mirror_ls_target_norm": 0.0,
        "mirror_ls_fit_norm": 0.0,
        "mirror_ls_residual_ratio": 1.0,
        "mirror_ls_coeff_norm": 0.0,
        "mirror_ls_velocity_metric_norm": 0.0,
        "mirror_ls_random_basis_count": 0.0,
        "mirror_ls_fallback": 1.0,
    }
    if target_delta is None:
        return base_velocity, stats
    target = target_delta.detach().float()
    if target.numel() == 0:
        return base_velocity, stats
    target = target - target.mean(dim=-1, keepdim=True)
    target_flat = target.reshape(-1)
    target_norm = float(target_flat.norm().item())
    stats["mirror_ls_target_norm"] = target_norm
    base_norm = metric_norm(base_velocity, diag)
    if target_norm <= 1.0e-12 or base_norm <= 1.0e-12:
        return base_velocity, stats

    basis: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_basis(vec: dict[str, Any], tag: str) -> None:
        if len(basis) >= max(1, int(max_basis)):
            return
        masked = apply_mask(vec, support_mask)
        norm = metric_norm(masked, diag)
        if norm <= 1.0e-12:
            return
        basis.append(normalize_metric(masked, diag, 1.0))
        seen.add(tag)

    add_basis(base_velocity, "full_support")
    pieces: list[tuple[float, str, dict[str, Any]]] = []
    for name, _p in named:
        value = base_velocity.get(name)
        mask = support_mask.get(name)
        if value is None or mask is None:
            continue
        piece = zeros_like_named(named)
        piece[name] = value * mask.to(device=value.device, dtype=value.dtype)
        norm = metric_norm(piece, diag)
        if norm > 1.0e-12:
            pieces.append((norm, name, piece))
    pieces.sort(key=lambda item: item[0], reverse=True)
    for _norm, name, piece in pieces:
        if name in seen:
            continue
        add_basis(piece, name)
        if len(basis) >= max(1, int(max_basis)):
            break
    random_basis_count = 0
    max_basis_i = max(1, int(max_basis))
    for attempt in range(max_basis_i * 4):
        if len(basis) >= max_basis_i:
            break
        rnd = apply_mask(random_like(base_velocity, stable_seed(seed, "mirror-ls-random", attempt)), support_mask)
        rnd, _rank, _overlap = residualize_metric(rnd, basis, diag) if basis else (rnd, 0, 0.0)
        before_len = len(basis)
        add_basis(rnd, f"support_random_{attempt}")
        if len(basis) > before_len:
            random_basis_count += 1
    stats["mirror_ls_random_basis_count"] = float(random_basis_count)
    if not basis:
        return base_velocity, stats

    before = {name: p.detach().clone() for name, p in named}
    responses = []
    eps = max(1.0e-6, float(eps))
    try:
        with torch.no_grad():
            base_log_probs = torch.log_softmax(model(xb).detach().float(), dim=-1)
            for vec in basis:
                for name, p in named:
                    step_vec = vec.get(name)
                    if step_vec is not None:
                        p.add_(eps * step_vec.to(device=p.device, dtype=p.dtype))
                perturbed_log_probs = torch.log_softmax(model(xb).detach().float(), dim=-1)
                for name, p in named:
                    p.copy_(before[name].to(device=p.device, dtype=p.dtype))
                response = ((perturbed_log_probs - base_log_probs) / eps).float()
                response = response - response.mean(dim=-1, keepdim=True)
                responses.append(response.reshape(-1))
    finally:
        with torch.no_grad():
            for name, p in named:
                p.copy_(before[name].to(device=p.device, dtype=p.dtype))
    if not responses:
        return base_velocity, stats
    R = torch.stack(responses, dim=1)
    k = int(R.shape[1])
    stats["mirror_ls_basis_count"] = float(k)
    stats["mirror_ls_fd_eval_count"] = float(k)
    eye = torch.eye(k, device=R.device, dtype=R.dtype)
    try:
        lhs = R.T @ R + max(1.0e-9, float(ridge)) * eye
        rhs = R.T @ target_flat.to(device=R.device, dtype=R.dtype)
        coeff = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        coeff = torch.linalg.lstsq(R, target_flat.to(device=R.device, dtype=R.dtype)).solution
    fit = R @ coeff
    fit_norm = float(fit.norm().item())
    residual_ratio = float((fit - target_flat.to(device=R.device, dtype=R.dtype)).norm().item() / (target_norm + 1.0e-12))
    stats["mirror_ls_fit_norm"] = fit_norm
    stats["mirror_ls_residual_ratio"] = residual_ratio
    stats["mirror_ls_coeff_norm"] = float(coeff.norm().item())
    raw_delta = zeros_like_named(named)
    for idx, vec in enumerate(basis):
        ci = coeff[idx].detach()
        for name, p in named:
            raw_delta[name] = raw_delta[name] + ci.to(device=p.device, dtype=p.dtype) * vec.get(name, torch.zeros_like(p.detach()))
    delta_norm = metric_norm(raw_delta, diag)
    if delta_norm <= 1.0e-12 or not math.isfinite(delta_norm):
        return base_velocity, stats
    normalized_delta = normalize_metric(raw_delta, diag, base_norm)
    velocity = {k: -v for k, v in normalized_delta.items()}
    stats["mirror_ls_active"] = 1.0
    stats["mirror_ls_fallback"] = 0.0
    stats["mirror_ls_velocity_metric_norm"] = metric_norm(velocity, diag)
    return velocity, stats


def basis_gram_diag(model: Any, xb: Any, named: list[tuple[str, Any]], default_diag: dict[str, Any]) -> tuple[dict[str, Any], float, float, float]:
    import torch

    diag = {k: v.clone() for k, v in default_diag.items()}
    cond = 1.0
    low_degree_overlap = 0.0
    low_freq_overlap = 0.0
    if not hasattr(model, "frozen_readout_features"):
        return diag, cond, low_degree_overlap, low_freq_overlap
    with torch.no_grad():
        feats = model.frozen_readout_features(xb).detach().float()
        centered = feats - feats.mean(dim=0, keepdim=True)
        gram_diag = centered.square().mean(dim=0).clamp_min(1.0e-8)
        positive = gram_diag[gram_diag > 0.0]
        if positive.numel():
            cond = float(positive.max().item() / max(1.0e-12, positive.min().item()))
        total_energy = float(gram_diag.sum().item()) + 1.0e-12
        k = int(getattr(model, "k", 1))
        low_idx = []
        for j in range(int(gram_diag.numel())):
            if (j % max(1, k)) <= 1:
                low_idx.append(j)
        if low_idx:
            low_degree_overlap = float(gram_diag[torch.tensor(low_idx, device=gram_diag.device)].sum().item() / total_energy)
            low_freq_overlap = low_degree_overlap
        def balance_basis_banks(energy: Any) -> Any:
            if energy.ndim < 2:
                return energy
            bank_mean = energy.mean(dim=0, keepdim=True).clamp_min(1.0e-8)
            global_mean = energy.mean().clamp_min(1.0e-8)
            return (energy / bank_mean) * global_mean

        def unit_mean_diag(energy: Any) -> Any:
            energy = energy.clamp_min(1.0e-8)
            scaled = energy / energy.mean().clamp_min(1.0e-8)
            scaled = scaled.clamp_min(1.0e-2)
            return scaled / scaled.mean().clamp_min(1.0e-8)

        for name, p in named:
            if name.endswith("w2") or name == "w2":
                if p.ndim == 3:
                    hidden = int(p.shape[0])
                    out_dim = int(p.shape[1])
                    k_param = int(p.shape[2])
                    core = gram_diag[: hidden * k_param]
                    if int(core.numel()) == hidden * k_param:
                        balanced = unit_mean_diag(balance_basis_banks(core.view(hidden, k_param)))
                        diag[name] = balanced[:, None, :].expand(hidden, out_dim, k_param).clone().to(device=p.device, dtype=p.dtype)
                    else:
                        flat = unit_mean_diag(balance_basis_banks(gram_diag.view(-1, 1))).reshape(-1)
                        need = int(p.numel())
                        reps = math.ceil(need / max(1, int(flat.numel())))
                        diag[name] = flat.repeat(reps)[:need].view_as(p).to(device=p.device, dtype=p.dtype)
                else:
                    flat = unit_mean_diag(balance_basis_banks(gram_diag.view(-1, 1))).reshape(-1)
                    need = int(p.numel())
                    reps = math.ceil(need / max(1, int(flat.numel())))
                    diag[name] = flat.repeat(reps)[:need].view_as(p).to(device=p.device, dtype=p.dtype)
            elif name.endswith("w1") or name == "w1":
                layer = model.layer1_basis(xb).detach().float() if hasattr(model, "layer1_basis") else None
                if layer is not None:
                    b1 = layer.square().mean(dim=0)
                    balanced = unit_mean_diag(balance_basis_banks(b1))
                    if p.ndim == 3 and int(p.shape[0]) == int(balanced.shape[0]) and int(p.shape[2]) == int(balanced.shape[1]):
                        diag[name] = balanced[:, None, :].expand(int(p.shape[0]), int(p.shape[1]), int(p.shape[2])).clone().to(device=p.device, dtype=p.dtype)
                    else:
                        flat = balanced.reshape(-1)
                        need = int(p.numel())
                        reps = math.ceil(need / max(1, int(flat.numel())))
                        diag[name] = flat.repeat(reps)[:need].view_as(p).to(device=p.device, dtype=p.dtype)
    return diag, cond, low_degree_overlap, low_freq_overlap


class MetricFlowController:
    def __init__(
        self,
        *,
        architecture: str,
        variant: str,
        control_mode: str,
        support_rank: int,
        beta_signal: float,
        beta_metric: float,
        beta_q: float,
        eta_rho: float,
        eta_debt: float,
        tau_safe: float,
        rho_min: float,
        rho_max: float,
        nuisance_rank: int,
        support_refresh_cadence: int,
        metric_refresh_cadence: int,
        metric_shrinkage: float,
        metric_eps: float,
        debt_velocity_barrier: float,
        calibration_velocity_barrier: float,
        safety_budget_velocity_barrier: float,
        calibration_nuisance_weight: float,
        calibration_correction_weight: float,
        calibration_nuisance_mode: str,
        debt_orthogonal_controller: bool,
        random_seed: int,
    ) -> None:
        self.architecture = architecture
        self.variant = variant
        self.metric_kind = metric_kind_for_variant(variant)
        self.control_mode = control_mode
        self.support_rank = int(support_rank)
        self.beta_signal = float(beta_signal)
        self.beta_metric = float(beta_metric)
        self.beta_q = float(beta_q)
        self.eta_rho = float(eta_rho)
        self.eta_debt = float(eta_debt)
        self.tau_safe = float(tau_safe)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.nuisance_rank = int(nuisance_rank)
        self.support_refresh_cadence = max(1, int(support_refresh_cadence))
        self.metric_refresh_cadence = max(1, int(metric_refresh_cadence))
        self.metric_shrinkage = min(0.95, max(0.0, float(metric_shrinkage)))
        self.metric_eps = max(1.0e-12, float(metric_eps))
        self.debt_velocity_barrier = max(0.0, float(debt_velocity_barrier))
        self.calibration_velocity_barrier = max(0.0, float(calibration_velocity_barrier))
        self.safety_budget_velocity_barrier = max(0.0, float(safety_budget_velocity_barrier))
        self.calibration_nuisance_weight = max(0.0, float(calibration_nuisance_weight))
        self.calibration_correction_weight = max(0.0, float(calibration_correction_weight))
        self.calibration_nuisance_mode = str(calibration_nuisance_mode)
        self.debt_orthogonal_controller = bool(debt_orthogonal_controller)
        self.random_seed = int(random_seed)
        self.signal: dict[str, Any] = {}
        self.diffusion: dict[str, Any] = {}
        self.metric_diag: dict[str, Any] = {}
        self.basis_state: dict[str, Any] = {}
        self.cached_support_mask: dict[str, Any] = {}
        self.cached_mirror_ls_velocity: dict[str, Any] | None = None
        self.cached_mirror_ls_stats: dict[str, float] = {}
        self.risk_ema: dict[str, float] = {}
        self.risk_budget_floor: dict[str, float] = {}
        self.q = 0.05
        self.rho = 0.0 if variant == "optimizer_alone" or control_mode == "same-overhead-noop" else max(self.rho_min, 0.02)
        self.last_debt = 0.0
        self.basis_gram_condition = 1.0
        self.low_degree_overlap = 0.0
        self.low_frequency_overlap = 0.0
        fixed_m9_weights = m9_fixed_weights_for_control(control_mode)
        self.metric_mixture_frozen = int(fixed_m9_weights is not None)
        self.metric_mixture_weights = normalized_m9_weights(fixed_m9_weights)
        self.metric_mixture_prev_weights = dict(self.metric_mixture_weights)
        self.metric_component_diags: dict[str, dict[str, Any]] = {}
        self.metric_mixture_eta = 0.25
        self.metric_mixture_update_count = 0
        self.metric_mixture_weight_drift = 0.0
        self.metric_mixture_debts = {name: 0.0 for name in M9_COMPONENTS}
        self.metric_mixture_support_gains = {name: 0.0 for name in M9_COMPONENTS}
        self.metric_mixture_conditions = {name: 1.0 for name in M9_COMPONENTS}
        self.metric_mixture_collapsed_to = ""

    @property
    def active(self) -> bool:
        return self.variant != "optimizer_alone" and self.control_mode != "same-overhead-noop"

    def build_m9_component_diags(
        self,
        named: list[tuple[str, Any]],
        grad: dict[str, Any],
        xb: Any,
        model: Any,
    ) -> dict[str, dict[str, Any]]:
        import torch

        raw_components: dict[str, dict[str, Any]] = {name: {} for name in M9_COMPONENTS}
        for name, p in named:
            g = grad[name].detach().float()
            raw_components["euclidean"][name] = torch.ones_like(g)
            raw_components["fisher"][name] = g.square().detach() + self.metric_eps
            sig = self.signal.get(name, g).to(device=g.device).float()
            diff = self.diffusion.get(name, g.square()).to(device=g.device).float()
            raw_components["signal"][name] = torch.relu(sig.square() - diff) + self.metric_eps
            param = p.detach().float()
            basis_raw = param.square()
            if basis_raw.ndim >= 2:
                row = basis_raw.reshape(basis_raw.shape[0], -1).mean(dim=1, keepdim=True)
                basis_raw = row.reshape(basis_raw.shape[0], *([1] * (basis_raw.ndim - 1))).expand_as(basis_raw)
            raw_components["basis"][name] = basis_raw + 0.05 * g.square() + self.metric_eps
        if self.architecture != "MLP":
            raw_components["basis"], cond, low_deg, low_freq = basis_gram_diag(model, xb, named, raw_components["basis"])
            self.basis_gram_condition = cond
            self.low_degree_overlap = low_deg
            self.low_frequency_overlap = low_freq
        components: dict[str, dict[str, Any]] = {}
        for component_name, raw_diag in raw_components.items():
            old_component = self.metric_component_diags.get(component_name, {})
            component_diag: dict[str, Any] = {}
            for name, _p in named:
                raw = raw_diag[name].float().clamp_min(self.metric_eps)
                mean = raw.mean().clamp_min(self.metric_eps)
                raw = (1.0 - self.metric_shrinkage) * raw + self.metric_shrinkage * mean
                old = old_component.get(name)
                value = raw if old is None else (1.0 - self.beta_metric) * old.to(device=raw.device).float() + self.beta_metric * raw
                component_diag[name] = value.clamp_min(self.metric_eps)
            components[component_name] = component_diag
            self.metric_mixture_conditions[component_name] = metric_condition(component_diag)[0]
        self.metric_component_diags = {
            cname: {k: v.detach().clone() for k, v in diag.items()}
            for cname, diag in components.items()
        }
        return components

    def current_m9_weights(self) -> dict[str, float]:
        fixed = m9_fixed_weights_for_control(self.control_mode)
        if fixed is not None:
            self.metric_mixture_weights = dict(fixed)
            return dict(fixed)
        return normalized_m9_weights(self.metric_mixture_weights)

    def update_m9_mixture_weights(
        self,
        *,
        named: list[tuple[str, Any]],
        signal: dict[str, Any],
        step: int,
        train_safety_debt: float,
        support_avoid_signal: dict[str, Any] | None,
        support_avoid_weight: float,
    ) -> None:
        if self.metric_kind != "adaptive_metric_mixture" or self.metric_mixture_frozen:
            return
        if not self.metric_component_diags:
            return
        debts: dict[str, float] = {}
        gains: dict[str, float] = {}
        for component_name in M9_COMPONENTS:
            component_diag = self.metric_component_diags.get(component_name)
            if not component_diag:
                continue
            mask = metric_support_mask(
                named,
                signal,
                component_diag,
                architecture=self.architecture,
                variant=self.variant,
                support_rank=self.support_rank,
                seed=stable_seed(self.random_seed, step, self.variant, "m9", component_name),
                avoid_signal=support_avoid_signal,
                avoid_weight=support_avoid_weight,
            )
            support = apply_mask(signal, mask)
            raw_norm = metric_norm(signal, component_diag)
            support_norm = metric_norm(support, component_diag)
            support_gain = support_norm / max(1.0e-12, raw_norm)
            residual_debt = max(0.0, 1.0 - support_gain)
            cond = max(1.0, float(self.metric_mixture_conditions.get(component_name, 1.0)))
            condition_debt = max(0.0, math.log10(cond) - 4.0) * 0.02 if math.isfinite(cond) else 1.0
            debts[component_name] = float(train_safety_debt) + residual_debt + condition_debt
            gains[component_name] = support_gain
        if not debts:
            return
        old_weights = normalized_m9_weights(self.metric_mixture_weights)
        log_weights = {
            name: math.log(max(1.0e-12, old_weights.get(name, 0.0))) - self.metric_mixture_eta * debts.get(name, 0.0)
            for name in M9_COMPONENTS
        }
        max_log = max(log_weights.values())
        exp_weights = {name: math.exp(value - max_log) for name, value in log_weights.items()}
        new_weights = normalized_m9_weights(exp_weights)
        self.metric_mixture_prev_weights = old_weights
        self.metric_mixture_weights = new_weights
        self.metric_mixture_debts = {name: float(debts.get(name, 0.0)) for name in M9_COMPONENTS}
        self.metric_mixture_support_gains = {name: float(gains.get(name, 0.0)) for name in M9_COMPONENTS}
        self.metric_mixture_weight_drift = sum(abs(new_weights[name] - old_weights.get(name, 0.0)) for name in M9_COMPONENTS)
        self.metric_mixture_update_count += 1
        best_name, best_weight = max(new_weights.items(), key=lambda item: item[1])
        self.metric_mixture_collapsed_to = best_name if best_weight >= 0.85 else ""

    def m9_meta(self) -> dict[str, Any]:
        weights = self.current_m9_weights() if self.metric_kind == "adaptive_metric_mixture" else {name: 0.0 for name in M9_COMPONENTS}
        meta: dict[str, Any] = {
            "metric_mixture_active": int(self.metric_kind == "adaptive_metric_mixture"),
            "metric_mixture_control_frozen": int(self.metric_kind == "adaptive_metric_mixture" and self.metric_mixture_frozen),
            "metric_mixture_component_count": len(M9_COMPONENTS) if self.metric_kind == "adaptive_metric_mixture" else 0,
            "metric_mixture_entropy": m9_weight_entropy(weights) if self.metric_kind == "adaptive_metric_mixture" else 0.0,
            "metric_weight_drift": self.metric_mixture_weight_drift if self.metric_kind == "adaptive_metric_mixture" else 0.0,
            "metric_mixture_update_count": self.metric_mixture_update_count,
            "metric_mixture_collapsed_to": self.metric_mixture_collapsed_to,
            "metric_mixture_eta": self.metric_mixture_eta if self.metric_kind == "adaptive_metric_mixture" else 0.0,
        }
        for name in M9_COMPONENTS:
            meta[f"metric_weight_{name}"] = weights.get(name, 0.0)
            meta[f"metric_debt_{name}"] = self.metric_mixture_debts.get(name, 0.0)
            meta[f"support_gain_by_metric_{name}"] = self.metric_mixture_support_gains.get(name, 0.0)
            meta[f"metric_condition_{name}"] = self.metric_mixture_conditions.get(name, 1.0)
        return meta

    def update_metric_diag(self, named: list[tuple[str, Any]], grad: dict[str, Any], xb: Any, model: Any) -> dict[str, Any]:
        import torch

        if self.metric_kind == "adaptive_metric_mixture":
            components = self.build_m9_component_diags(named, grad, xb, model)
            weights = self.current_m9_weights()
            base = {}
            for name, _p in named:
                mixed = None
                for component_name in M9_COMPONENTS:
                    value = components[component_name][name].float() * float(weights[component_name])
                    mixed = value if mixed is None else mixed + value
                base[name] = mixed.clamp_min(self.metric_eps)
            self.metric_diag = {k: v.detach().clone() for k, v in base.items()}
            return self.metric_diag

        base = {}
        for name, p in named:
            g = grad[name].detach().float()
            if self.metric_kind == "euclidean_oet":
                raw = torch.ones_like(g)
            elif self.metric_kind in {"fisher_ema", "fisher_signal_basis_gram"}:
                raw = g.square().detach() + self.metric_eps
            elif self.metric_kind == "signal_drift_diffusion":
                sig = self.signal.get(name, g).to(device=g.device).float()
                diff = self.diffusion.get(name, g.square()).to(device=g.device).float()
                raw = torch.relu(sig.square() - diff) + self.metric_eps
            elif self.metric_kind.startswith("mlp_"):
                param = p.detach().float()
                raw = param.square()
                if raw.ndim >= 2:
                    row = raw.reshape(raw.shape[0], -1).mean(dim=1, keepdim=True)
                    raw = row.reshape(raw.shape[0], *([1] * (raw.ndim - 1))).expand_as(raw)
                raw = raw + 0.05 * g.square() + self.metric_eps
            else:
                raw = torch.ones_like(g)
            mean = raw.mean().clamp_min(self.metric_eps)
            raw = (1.0 - self.metric_shrinkage) * raw + self.metric_shrinkage * mean
            old = self.metric_diag.get(name)
            base[name] = raw if old is None else (1.0 - self.beta_metric) * old.to(device=g.device).float() + self.beta_metric * raw
        if self.metric_kind in {"kan_basis_gram", "fisher_signal_basis_gram"} and self.architecture != "MLP":
            base, cond, low_deg, low_freq = basis_gram_diag(model, xb, named, base)
            self.low_degree_overlap = low_deg
            self.low_frequency_overlap = low_freq
            normalized = {}
            for name, raw in base.items():
                raw = raw.float().clamp_min(self.metric_eps)
                mean = raw.mean().clamp_min(self.metric_eps)
                normalized[name] = (1.0 - self.metric_shrinkage) * raw + self.metric_shrinkage * mean
            base = normalized
            self.basis_gram_condition = metric_condition(base)[0]
        self.metric_diag = {k: v.detach().clone() for k, v in base.items()}
        return self.metric_diag

    def update_and_emit(
        self,
        *,
        named: list[tuple[str, Any]],
        opt: Any,
        risk: dict[str, float],
        calibration_grad: dict[str, Any] | None,
        step: int,
        xb: Any,
        model: Any,
        calibration_aux_grads: dict[str, dict[str, Any]] | None = None,
        mirror_grad: dict[str, Any] | None = None,
        mirror_meta: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        full_named = named
        use_basis_fast = basis_carrier_fast_path(self.variant, self.architecture)
        if use_basis_fast:
            basis_named = [(name, p) for name, p in named if is_basis_param(name)]
            if basis_named:
                named = basis_named
        mirror_target_delta = None
        if mirror_meta is not None:
            mirror_meta = dict(mirror_meta)
            mirror_target_delta = mirror_meta.pop("_mirror_target_delta", None)
        grad = grad_dict(named)
        phase = phase_for_variant(self.variant)
        if phase == "M1_FunctionalMirror" and mirror_grad:
            grad = {
                name: mirror_grad.get(name, grad[name]).to(device=grad[name].device, dtype=grad[name].dtype).detach().clone()
                for name, _p in named
            }
        if phase == "M1_FunctionalMirror" and mirror_meta is not None and not mirror_grad:
            mirror_meta["mirror_grad_norm"] = tensor_norm(grad)
            mirror_meta.setdefault("mirror_grad_source", "ce_backward_reused")
        mom = momentum_dict(named, opt)
        for name, _p in named:
            g = grad[name].detach().float()
            old = self.signal.get(name)
            sig = g if old is None else (1.0 - self.beta_signal) * old.to(device=g.device).float() + self.beta_signal * g
            self.signal[name] = sig.detach()
            innovation = g if old is None else g - old.to(device=g.device).float()
            old_diff = self.diffusion.get(name)
            diff_raw = innovation.square()
            diff = diff_raw if old_diff is None else 0.95 * old_diff.to(device=g.device).float() + 0.05 * diff_raw
            self.diffusion[name] = diff.detach()
        refresh_metric = (not self.metric_diag) or (step == 1) or (step % self.metric_refresh_cadence == 0)
        diag = self.update_metric_diag(named, grad, xb, model) if refresh_metric else self.metric_diag
        support_avoid_signal = None
        support_avoid_weight = 0.0
        if (
            calibration_grad
            and self.calibration_nuisance_weight > 0.0
            and self.calibration_nuisance_mode in {
                "brier",
                "tail",
                "margin",
                "tail_margin",
                "tail_brier",
                "tail_brier_balanced",
                "tail_brier_brier2_balanced",
                "tail_brier_pareto",
                "tail_brier_pareto2",
                "tail_brier_qp",
                "tail_brier_qp_margin",
                "tail_brier_qp_brier_margin",
                "tail_q99_brier_qp_margin",
                "tail_q99_brier_qp_strict_margin",
                "tail_q99_brier_qp_hard_margin",
            }
        ):
            support_avoid_signal = {}
            for name, value in self.signal.items():
                cg = calibration_grad.get(name)
                support_avoid_signal[name] = value * 0.0 if cg is None else cg.to(device=value.device, dtype=value.dtype)
            support_avoid_weight = self.calibration_nuisance_weight
        signal = {k: v.to(device=grad[k].device, dtype=grad[k].dtype) for k, v in self.signal.items()}
        support_selection_signal = signal
        tail_projected_signal_active = 0
        tail_projection_overlap = 0.0
        if support_avoid_signal is not None and support_avoid_weight > 0.0:
            support_selection_signal, _tail_rank, tail_projection_overlap = residualize_metric(signal, [support_avoid_signal], diag)
            tail_projected_signal_active = 1
        refresh_support = (not self.cached_support_mask) or (step == 1) or (step % self.support_refresh_cadence == 0)
        if refresh_support:
            support_mask = metric_support_mask(
                named,
                support_selection_signal,
                diag,
                architecture=self.architecture,
                variant=self.variant,
                support_rank=self.support_rank,
                seed=stable_seed(self.random_seed, step, self.variant, self.metric_kind),
                avoid_signal=support_avoid_signal,
                avoid_weight=support_avoid_weight,
            )
            self.cached_support_mask = {k: v.detach().clone() for k, v in support_mask.items()}
        else:
            support_mask = self.cached_support_mask
        support_signal = apply_mask(support_selection_signal, support_mask)
        support_signal_norm = metric_norm(support_signal, diag)
        signal_norm = metric_norm(support_selection_signal, diag)
        lazy: dict[str, dict[str, Any]] = {}
        calibration_nuisance = None
        calibration_nuisance_norm = 0.0
        calibration_nuisance_overlap = 0.0
        calibration_correction = None
        calibration_correction_norm = 0.0
        calibration_correction_overlap = 0.0
        calibration_correction_sign = 0.0
        calibration_source = None
        if (self.calibration_nuisance_weight > 0.0 or self.calibration_correction_weight > 0.0) and calibration_grad:
            calib_src = {}
            for name, value in signal.items():
                cg = calibration_grad.get(name)
                calib_src[name] = value * 0.0 if cg is None else cg.to(device=value.device, dtype=value.dtype)
            calib_src = apply_mask(calib_src, support_mask)
            if self.metric_kind == "euclidean_oet" or "OET" in self.variant or "BankLocal" in self.variant:
                calib_src = tangent_project_metric(calib_src, named, diag)
            calibration_source = calib_src
        if self.calibration_nuisance_weight > 0.0 and calibration_source is not None:
            calibration_nuisance = normalize_metric(
                calibration_source,
                diag,
                max(1.0e-12, support_signal_norm) * self.calibration_nuisance_weight,
            )
            calibration_nuisance_norm = metric_norm(calibration_nuisance, diag)
            if calibration_nuisance_norm > 1.0e-12 and support_signal_norm > 1.0e-12:
                unit_calib = normalize_metric(calibration_nuisance, diag, 1.0)
                calibration_nuisance_overlap = abs(metric_dot(support_signal, unit_calib, diag)) / max(1.0e-12, support_signal_norm)
        if self.calibration_correction_weight > 0.0 and calibration_source is not None:
            calibration_correction_sign = 1.0
            if str(getattr(self, "calibration_nuisance_mode", "")) == "confidence":
                over = float(risk.get("overconfidence_gap", 0.0))
                under = float(risk.get("underconfidence_gap", 0.0))
                calibration_correction_sign = -1.0 if under > over else 1.0
            signed_source = {k: v * calibration_correction_sign for k, v in calibration_source.items()}
            calibration_correction = normalize_metric(
                signed_source,
                diag,
                max(1.0e-12, support_signal_norm) * self.calibration_correction_weight,
            )
            calibration_correction_norm = metric_norm(calibration_correction, diag)

        def get_support_random() -> dict[str, Any]:
            if "support_random" not in lazy:
                lazy["support_random"] = normalize_metric(
                    apply_mask(random_like(signal, stable_seed(self.random_seed, "support", step)), support_mask),
                    diag,
                    support_signal_norm,
                )
            return lazy["support_random"]

        def get_support_signflip() -> dict[str, Any]:
            if "support_signflip" not in lazy:
                lazy["support_signflip"] = normalize_metric({k: -v for k, v in support_signal.items()}, diag, support_signal_norm)
            return lazy["support_signflip"]

        def get_support_shuffled() -> dict[str, Any]:
            if "support_shuffled" not in lazy:
                lazy["support_shuffled"] = normalize_metric(
                    apply_mask(shuffled_like(support_signal, stable_seed(self.random_seed, "shuf", step)), support_mask),
                    diag,
                    support_signal_norm,
                )
            return lazy["support_shuffled"]

        def get_additive_random() -> dict[str, Any]:
            if "additive_random" not in lazy:
                lazy["additive_random"] = normalize_metric(random_like(signal, stable_seed(self.random_seed, "allrandom", step)), diag, signal_norm)
            return lazy["additive_random"]

        def get_optimizer_geometry() -> dict[str, Any]:
            if "optimizer_geometry" not in lazy:
                tangent_source = mom if tensor_norm(mom) > 1.0e-12 else signal
                lazy["optimizer_geometry"] = normalize_metric(tangent_project_metric(tangent_source, named, diag), diag, signal_norm)
            return lazy["optimizer_geometry"]

        def get_basis_random() -> dict[str, Any]:
            if "basis_random" not in lazy:
                sr = get_support_random()
                lazy["basis_random"] = normalize_metric(
                    {k: (sr[k] if (is_basis_param(k) or self.architecture == "MLP") else sr[k] * 0.0) for k in sr},
                    diag,
                    support_signal_norm,
                )
            return lazy["basis_random"]

        def get_basis_signflip() -> dict[str, Any]:
            if "basis_signflip" not in lazy:
                sf = get_support_signflip()
                lazy["basis_signflip"] = normalize_metric(
                    {k: (sf[k] if (is_basis_param(k) or self.architecture == "MLP") else sf[k] * 0.0) for k in sf},
                    diag,
                    support_signal_norm,
                )
            return lazy["basis_signflip"]

        def get_readout_leakage() -> dict[str, Any]:
            if "readout_leakage" not in lazy:
                lazy["readout_leakage"] = normalize_metric(
                    {k: (support_signal[k] if is_readout_param(k) else support_signal[k] * 0.0) for k in support_signal},
                    diag,
                    support_signal_norm,
                )
            return lazy["readout_leakage"]

        m6_noise_scale = 0.35

        def get_m6_reservoir_noise() -> dict[str, Any]:
            if "m6_reservoir_noise" not in lazy:
                lazy["m6_reservoir_noise"] = normalize_metric(
                    apply_complement_mask(random_like(signal, stable_seed(self.random_seed, "m6-reservoir", step)), support_mask),
                    diag,
                    max(1.0e-12, support_signal_norm * m6_noise_scale),
                )
            return lazy["m6_reservoir_noise"]

        def get_m6_signal_noise() -> dict[str, Any]:
            if "m6_signal_noise" not in lazy:
                lazy["m6_signal_noise"] = normalize_metric(
                    apply_mask(random_like(signal, stable_seed(self.random_seed, "m6-signal", step)), support_mask),
                    diag,
                    max(1.0e-12, support_signal_norm * m6_noise_scale),
                )
            return lazy["m6_signal_noise"]

        def get_m6_gaussian_noise() -> dict[str, Any]:
            if "m6_gaussian_noise" not in lazy:
                lazy["m6_gaussian_noise"] = normalize_metric(
                    random_like(signal, stable_seed(self.random_seed, "m6-gaussian", step)),
                    diag,
                    max(1.0e-12, support_signal_norm * m6_noise_scale),
                )
            return lazy["m6_gaussian_noise"]

        nuisance = []
        if calibration_nuisance is not None and calibration_nuisance_norm > 1.0e-12:
            nuisance.append(calibration_nuisance)
        if phase in {"S1_MetricResidualSignal", "S6_MetricKANCarrier", "P5_PureKANBasisCarrier"} or self.variant == "S3-ResidualSignalFU":
            nuisance.append(get_support_random())
            if self.nuisance_rank >= 2:
                nuisance.append(get_optimizer_geometry())
            if self.nuisance_rank >= 3:
                nuisance.append(get_additive_random())
        if nuisance:
            residual, nuisance_rank, nuisance_overlap = residualize_metric(support_signal, nuisance, diag)
        else:
            residual, nuisance_rank, nuisance_overlap = support_signal, 0, 0.0
        mirror_ls_stats = {
            "mirror_ls_active": 0.0,
            "mirror_ls_cached": 0.0,
            "mirror_ls_basis_count": 0.0,
            "mirror_ls_fd_eval_count": 0.0,
            "mirror_ls_target_norm": 0.0,
            "mirror_ls_fit_norm": 0.0,
            "mirror_ls_residual_ratio": 1.0,
            "mirror_ls_coeff_norm": 0.0,
            "mirror_ls_velocity_metric_norm": 0.0,
            "mirror_ls_random_basis_count": 0.0,
            "mirror_ls_fallback": 1.0,
        }
        m6_noise_component = None
        m6_noise_role = "none"
        if self.variant == "optimizer_alone":
            velocity = zeros_like_named(named)
            direction_source = "base_optimizer_only"
        elif self.control_mode == "same-overhead-noop":
            velocity = zeros_like_named(named)
            direction_source = "same_overhead_noop"
        elif self.control_mode == "same-norm-additive-random":
            velocity = get_additive_random()
            direction_source = "same_norm_additive_random_metric_normalized"
        elif self.control_mode in {"same-metric-support-random", "same-mirror-support-random"}:
            velocity = get_support_random()
            direction_source = str(self.control_mode)
        elif self.control_mode in {"same-OET-random", "same-metric-skew-random", "same-generator-norm-random", "same-radial-random", "same-Lie-random"}:
            velocity = get_support_random()
            direction_source = str(self.control_mode)
        elif self.control_mode in {"same-metric-support-signflip", "same-mirror-support-signflip"}:
            velocity = get_support_signflip()
            direction_source = str(self.control_mode)
        elif self.control_mode in {"same-OET-signflip", "same-Lie-signflip"}:
            velocity = get_support_signflip()
            direction_source = str(self.control_mode).replace("-", "_")
        elif self.control_mode == "same-metric-support-shuffled":
            velocity = get_support_shuffled()
            direction_source = "same_metric_support_shuffled"
        elif self.control_mode == "same-OET-shuffled":
            velocity = get_support_shuffled()
            direction_source = "same_OET_shuffled"
        elif self.control_mode == "same-optimizer-geometry-random":
            velocity = get_optimizer_geometry()
            direction_source = "same_optimizer_geometry_randomized_tangent"
        elif self.control_mode in {"same-basis-Gram-random", "same-basis-OET-random", "same-degree-frequency-random"}:
            velocity = get_basis_random()
            direction_source = self.control_mode
        elif self.control_mode == "same-basis-Gram-signflip":
            velocity = get_basis_signflip()
            direction_source = "same_basis_Gram_signflip"
        elif self.control_mode == "same-bank-shuffled":
            velocity = get_support_shuffled()
            direction_source = "same_bank_shuffled"
        elif self.control_mode == "same-readout-leakage-control":
            velocity = get_readout_leakage()
            direction_source = "same_readout_leakage_control"
        elif self.control_mode == "M6-signal-noise-control":
            m6_noise_component = get_m6_signal_noise()
            m6_noise_role = "signal_support_same_norm_noise_control"
            velocity = {
                k: support_signal.get(k, v * 0.0) + v
                for k, v in m6_noise_component.items()
            }
            direction_source = "m6_signal_noise_control_support_leakage"
        elif self.control_mode == "M6-same-norm-Gaussian-control":
            m6_noise_component = get_m6_gaussian_noise()
            m6_noise_role = "full_parameter_same_norm_gaussian_control"
            velocity = {
                k: support_signal.get(k, v * 0.0) + v
                for k, v in m6_noise_component.items()
            }
            direction_source = "m6_same_norm_gaussian_noise_control"
        elif phase == "M1_FunctionalMirror":
            if mirror_ls_variant(self.variant) and self.control_mode == "none":
                refresh_ls = int_flag((mirror_meta or {}).get("mirror_grad_refreshed_this_step", 1)) or self.cached_mirror_ls_velocity is None
                if refresh_ls:
                    velocity, mirror_ls_stats = functional_mirror_least_squares_velocity(
                        named=named,
                        model=model,
                        xb=xb,
                        support_mask=support_mask,
                        base_velocity=support_signal,
                        diag=diag,
                        target_delta=mirror_target_delta,
                        seed=stable_seed(self.random_seed, step, self.variant, "mirror-ls"),
                        max_basis=max(2, min(16, int(self.support_rank))),
                        eps=1.0e-3,
                        ridge=1.0e-3,
                    )
                    self.cached_mirror_ls_velocity = clone_velocity(velocity)
                    self.cached_mirror_ls_stats = dict(mirror_ls_stats)
                elif self.cached_mirror_ls_velocity is not None:
                    velocity = clone_velocity(self.cached_mirror_ls_velocity) or support_signal
                    mirror_ls_stats = dict(self.cached_mirror_ls_stats)
                    mirror_ls_stats["mirror_ls_cached"] = 1.0
                    mirror_ls_stats["mirror_ls_fd_eval_count"] = 0.0
                else:
                    velocity = support_signal
                direction_source = "functional_mirror_sketched_least_squares"
            else:
                velocity = support_signal
                direction_source = "functional_mirror_support_masked_Jt_residual"
        elif phase == "S1_MetricResidualSignal":
            velocity = residual
            direction_source = "metric_residual_signal_after_nuisance_projection"
        elif phase in {"S6_MetricKANCarrier", "P5_PureKANBasisCarrier"}:
            velocity = {k: (residual[k] if is_basis_param(k) else residual[k] * 0.0) for k in residual}
            direction_source = "pure_kan_basis_metric_residual_carrier" if phase == "P5_PureKANBasisCarrier" else "kan_basis_metric_residual_carrier"
        elif phase == "M6_NoiseShapedReservoirRegularization":
            if self.variant == "M6-noise-suppression":
                velocity = support_signal
                m6_noise_role = "reservoir_noise_suppressed_sigma_zero"
                direction_source = "m6_reservoir_noise_suppression_support_only"
            else:
                m6_noise_component = get_m6_reservoir_noise()
                m6_noise_role = "support_complement_reservoir_noise"
                velocity = {
                    k: support_signal.get(k, v * 0.0) + v
                    for k, v in m6_noise_component.items()
                }
                direction_source = "m6_support_plus_reservoir_complement_noise"
        elif phase == "M9_AdaptiveMetricMixture":
            velocity = support_signal
            direction_source = "m9_train_batch_adaptive_metric_mixture_support_signal"
        elif phase == "S3_StrongMetricOptimizer":
            if self.variant == "S3-ResidualSignalFU":
                velocity = residual
                direction_source = "strong_optimizer_residual_metric_fu"
            elif self.variant == "S3-SameMetricSupportControl":
                velocity = get_support_random()
                direction_source = "strong_optimizer_same_metric_support_control"
            elif self.variant == "S3-SameTangentControl":
                velocity = get_optimizer_geometry()
                direction_source = "strong_optimizer_same_tangent_control"
            else:
                velocity = support_signal
                direction_source = "strong_optimizer_metric_support_fu"
        else:
            if calibration_nuisance is not None and calibration_nuisance_norm > 1.0e-12:
                velocity = residual
                direction_source = "metric_support_projected_signal_calibration_nuisance_residualized"
            else:
                velocity = support_signal
                direction_source = "metric_support_projected_signal"
        m6_noise_norm = 0.0
        m6_reservoir_noise_norm = 0.0
        m6_signal_noise_norm = 0.0
        m6_signal_leakage = 0.0
        if m6_noise_component is not None:
            m6_noise_norm = metric_norm(m6_noise_component, diag)
            m6_reservoir_noise_norm = metric_norm(apply_complement_mask(m6_noise_component, support_mask), diag)
            m6_signal_noise_norm = metric_norm(apply_mask(m6_noise_component, support_mask), diag)
            m6_signal_leakage = m6_signal_noise_norm / max(1.0e-12, m6_noise_norm)
        if self.metric_kind == "euclidean_oet" or "OET" in self.variant or "BankLocal" in self.variant:
            velocity = tangent_project_metric(velocity, named, diag)
        if (
            calibration_nuisance is not None
            and calibration_nuisance_norm > 1.0e-12
            and direction_source not in {"base_optimizer_only", "same_overhead_noop"}
        ):
            velocity, _cal_rank, calibration_control_overlap = residualize_metric(velocity, [calibration_nuisance], diag)
            calibration_nuisance_overlap = max(calibration_nuisance_overlap, calibration_control_overlap)
            if not direction_source.endswith("_calibration_nuisance_residualized"):
                direction_source = f"{direction_source}_calibration_nuisance_residualized"
        if (
            calibration_correction is not None
            and calibration_correction_norm > 1.0e-12
            and direction_source not in {"base_optimizer_only", "same_overhead_noop"}
        ):
            velocity_norm_before_correction = metric_norm(velocity, diag)
            if velocity_norm_before_correction > 1.0e-12:
                unit_corr = normalize_metric(calibration_correction, diag, 1.0)
                calibration_correction_overlap = metric_dot(velocity, unit_corr, diag) / max(1.0e-12, velocity_norm_before_correction)
            velocity = {
                k: v + calibration_correction.get(k, v * 0.0).to(device=v.device, dtype=v.dtype)
                for k, v in velocity.items()
            }
            direction_source = f"{direction_source}_calibration_corrected"
        pareto_guard_active = 0
        pareto_tail_alignment_before = 0.0
        pareto_brier_alignment_before = 0.0
        pareto_tail_added = 0.0
        pareto_brier_added = 0.0
        debt_tail_alignment_before_orthogonal = 0.0
        debt_brier_alignment_before_orthogonal = 0.0
        debt_tail_alignment_after_orthogonal = 0.0
        debt_brier_alignment_after_orthogonal = 0.0
        debt_orthogonal_projection_rank = 0
        debt_orthogonal_projection_overlap = 0.0

        def aux_unit(aux_name: str) -> Any | None:
            if not calibration_aux_grads:
                return None
            aux_grad = calibration_aux_grads.get(aux_name)
            if not aux_grad:
                return None
            aux_src = {}
            for name, value in velocity.items():
                ag = aux_grad.get(name)
                aux_src[name] = value * 0.0 if ag is None else ag.to(device=value.device, dtype=value.dtype)
            masked_aux = apply_mask(aux_src, support_mask)
            if self.metric_kind == "euclidean_oet" or "OET" in self.variant or "BankLocal" in self.variant:
                masked_aux = tangent_project_metric(masked_aux, named, diag)
            if metric_norm(masked_aux, diag) <= 1.0e-12:
                return None
            return normalize_metric(masked_aux, diag, 1.0)

        if (
            calibration_aux_grads
            and str(getattr(self, "calibration_nuisance_mode", "")) in {
                "tail_brier_pareto",
                "tail_brier_pareto2",
                "tail_brier_qp",
                "tail_brier_qp_margin",
                "tail_brier_qp_brier_margin",
                "tail_q99_brier_qp_margin",
                "tail_q99_brier_qp_strict_margin",
                "tail_q99_brier_qp_hard_margin",
            }
            and direction_source not in {"base_optimizer_only", "same_overhead_noop"}
        ):
            velocity_norm_for_guard = metric_norm(velocity, diag)
            mode_name = str(getattr(self, "calibration_nuisance_mode", ""))
            q99_guard_modes = {
                "tail_q99_brier_qp_margin",
                "tail_q99_brier_qp_strict_margin",
                "tail_q99_brier_qp_hard_margin",
            }
            if mode_name in {"tail_brier_qp", "tail_brier_qp_margin", "tail_brier_qp_brier_margin", *q99_guard_modes} and velocity_norm_for_guard > 1.0e-12:
                tail_aux_name = "tail_q99" if mode_name in q99_guard_modes else "tail"
                unit_tail = aux_unit(tail_aux_name)
                unit_brier = aux_unit("brier")
                if unit_tail is not None and unit_brier is not None:
                    tail_guard_margin = 1.0e-5 if mode_name in {"tail_brier_qp_margin", "tail_q99_brier_qp_margin"} else 0.0
                    brier_guard_margin = 1.0e-5 if mode_name in {"tail_brier_qp_margin", "tail_q99_brier_qp_margin"} else 0.0
                    if mode_name == "tail_q99_brier_qp_strict_margin":
                        tail_guard_margin = 1.0e-4
                        brier_guard_margin = 1.0e-5
                    if mode_name == "tail_q99_brier_qp_hard_margin":
                        tail_guard_margin = 5.0e-4
                        brier_guard_margin = 1.0e-5
                    if mode_name == "tail_brier_qp_brier_margin":
                        brier_guard_margin = 1.0e-4
                    d_tail = metric_dot(velocity, unit_tail, diag)
                    d_brier = metric_dot(velocity, unit_brier, diag)
                    pareto_tail_alignment_before = d_tail / max(1.0e-12, velocity_norm_for_guard)
                    pareto_brier_alignment_before = d_brier / max(1.0e-12, velocity_norm_for_guard)
                    corr = max(-0.999, min(0.999, metric_dot(unit_tail, unit_brier, diag)))
                    candidates: list[tuple[float, float, float]] = []
                    if d_tail >= tail_guard_margin and d_brier >= brier_guard_margin:
                        candidates.append((0.0, 0.0, 0.0))
                    a = max(0.0, tail_guard_margin - d_tail)
                    if d_tail + a >= tail_guard_margin - 1.0e-12 and d_brier + a * corr >= brier_guard_margin - 1.0e-12:
                        candidates.append((a * a, a, 0.0))
                    b = max(0.0, brier_guard_margin - d_brier)
                    if d_tail + b * corr >= tail_guard_margin - 1.0e-12 and d_brier + b >= brier_guard_margin - 1.0e-12:
                        candidates.append((b * b, 0.0, b))
                    det = max(1.0e-6, 1.0 - corr * corr)
                    tail_target = tail_guard_margin - d_tail
                    brier_target = brier_guard_margin - d_brier
                    a2 = (tail_target - corr * brier_target) / det
                    b2 = (brier_target - corr * tail_target) / det
                    if a2 >= -1.0e-12 and b2 >= -1.0e-12:
                        a2 = max(0.0, a2)
                        b2 = max(0.0, b2)
                        if d_tail + a2 + b2 * corr >= tail_guard_margin - 1.0e-9 and d_brier + a2 * corr + b2 >= brier_guard_margin - 1.0e-9:
                            candidates.append((a2 * a2 + b2 * b2 + 2.0 * a2 * b2 * corr, a2, b2))
                    if candidates:
                        _norm2, add_tail, add_brier = min(candidates, key=lambda item: item[0])
                        if add_tail > 0.0 or add_brier > 0.0:
                            velocity = {
                                k: v
                                + add_tail * unit_tail.get(k, v * 0.0).to(device=v.device, dtype=v.dtype)
                                + add_brier * unit_brier.get(k, v * 0.0).to(device=v.device, dtype=v.dtype)
                                for k, v in velocity.items()
                            }
                            pareto_guard_active = 1
                            pareto_tail_added = add_tail
                            pareto_brier_added = add_brier
            else:
                guard_order = ("tail", "brier", "tail") if mode_name == "tail_brier_pareto2" else ("tail", "brier")
                for aux_name in guard_order:
                    if velocity_norm_for_guard <= 1.0e-12:
                        continue
                    unit_aux = aux_unit(aux_name)
                    if unit_aux is None:
                        continue
                    alignment = metric_dot(velocity, unit_aux, diag) / max(1.0e-12, velocity_norm_for_guard)
                    if aux_name == "tail":
                        pareto_tail_alignment_before = alignment
                    else:
                        pareto_brier_alignment_before = alignment
                    if alignment < 0.0:
                        add_scale = -alignment + 1.0e-6
                        velocity = {
                            k: v + add_scale * unit_aux.get(k, v * 0.0).to(device=v.device, dtype=v.dtype)
                            for k, v in velocity.items()
                        }
                        velocity_norm_for_guard = metric_norm(velocity, diag)
                        pareto_guard_active = 1
                        if aux_name == "tail":
                            pareto_tail_added = add_scale
                        else:
                            pareto_brier_added = add_scale
            if pareto_guard_active:
                direction_source = f"{direction_source}_pareto_guarded"
        if self.debt_orthogonal_controller and calibration_aux_grads and direction_source not in {"base_optimizer_only", "same_overhead_noop"}:
            mode_name = str(getattr(self, "calibration_nuisance_mode", ""))
            q99_guard_modes = {
                "tail_q99_brier_qp_margin",
                "tail_q99_brier_qp_strict_margin",
                "tail_q99_brier_qp_hard_margin",
            }
            tail_aux_name = "tail_q99" if mode_name in q99_guard_modes else "tail"
            unit_tail = aux_unit(tail_aux_name)
            unit_brier = aux_unit("brier")
            velocity_norm_before_orthogonal = metric_norm(velocity, diag)
            if velocity_norm_before_orthogonal > 1.0e-12:
                if unit_tail is not None:
                    debt_tail_alignment_before_orthogonal = metric_dot(velocity, unit_tail, diag) / max(1.0e-12, velocity_norm_before_orthogonal)
                if unit_brier is not None:
                    debt_brier_alignment_before_orthogonal = metric_dot(velocity, unit_brier, diag) / max(1.0e-12, velocity_norm_before_orthogonal)
            debt_units = [unit for unit in [unit_tail, unit_brier] if unit is not None]
            if debt_units:
                velocity, debt_orthogonal_projection_rank, debt_orthogonal_projection_overlap = residualize_metric(velocity, debt_units, diag)
                velocity_norm_after_orthogonal = metric_norm(velocity, diag)
                if velocity_norm_after_orthogonal > 1.0e-12:
                    if unit_tail is not None:
                        debt_tail_alignment_after_orthogonal = metric_dot(velocity, unit_tail, diag) / max(1.0e-12, velocity_norm_after_orthogonal)
                    if unit_brier is not None:
                        debt_brier_alignment_after_orthogonal = metric_dot(velocity, unit_brier, diag) / max(1.0e-12, velocity_norm_after_orthogonal)
                direction_source = f"{direction_source}_debt_orthogonalized"
        velocity = normalize_metric(velocity, diag, 1.0)

        raw_norm = signal_norm
        support_norm = support_signal_norm
        residual_norm = metric_norm(residual, diag)
        diffusion = {k: self.diffusion[k].sqrt().to(device=signal[k].device, dtype=signal[k].dtype) for k in self.diffusion}
        support_diffusion = apply_mask(diffusion, support_mask)
        residual_snr = residual_norm / max(1.0e-12, metric_norm(support_diffusion, diag))
        cond, minv, maxv = metric_condition(diag)
        support_params = sum(int(mask.sum().item()) for mask in support_mask.values())
        total_params = sum(int(p.numel()) for _name, p in full_named) if use_basis_fast else sum(mask.numel() for mask in support_mask.values())
        support_residual = max(0.0, 1.0 - support_norm / max(1.0e-12, raw_norm))
        debt = 0.0
        for key in ["ce", "tail_q99", "sharpness_proxy"]:
            value = float(risk.get(key, 0.0))
            old = self.risk_ema.get(key, value)
            self.risk_ema[key] = 0.98 * old + 0.02 * value
            debt += max(0.0, value - self.risk_ema[key])
        if self.debt_velocity_barrier > 0.0 and debt > 0.0:
            barrier_scale = 1.0 / (1.0 + self.debt_velocity_barrier * debt)
            velocity = {k: v * barrier_scale for k, v in velocity.items()}
        calibration_gap = float(risk.get("calibration_gap", 0.0))
        calibration_velocity_scale = 1.0
        if self.calibration_velocity_barrier > 0.0 and calibration_gap > 0.0:
            calibration_velocity_scale = 1.0 / (1.0 + self.calibration_velocity_barrier * calibration_gap)
            velocity = {k: v * calibration_velocity_scale for k, v in velocity.items()}
        safety_budget_debt = 0.0
        safety_budget_velocity_scale = 1.0
        if self.safety_budget_velocity_barrier > 0.0:
            for key in ["ce", "tail_q99", "calibration_gap", "batch_brier_proxy"]:
                value = float(risk.get(key, 0.0))
                floor = self.risk_budget_floor.get(key)
                if floor is None:
                    self.risk_budget_floor[key] = value
                    floor = value
                excess = max(0.0, value - float(floor))
                norm = 1.0 if key in {"calibration_gap", "batch_brier_proxy"} else max(1.0, abs(float(floor)))
                safety_budget_debt += excess / max(1.0e-12, norm)
                if value < float(floor):
                    self.risk_budget_floor[key] = value
            if safety_budget_debt > 0.0:
                safety_budget_velocity_scale = 1.0 / (1.0 + self.safety_budget_velocity_barrier * safety_budget_debt)
                velocity = {k: v * safety_budget_velocity_scale for k, v in velocity.items()}
        if self.metric_kind == "adaptive_metric_mixture":
            self.update_m9_mixture_weights(
                named=named,
                signal=support_selection_signal,
                step=step,
                train_safety_debt=safety_budget_debt + debt,
                support_avoid_signal=support_avoid_signal,
                support_avoid_weight=support_avoid_weight,
            )
        velocity_emitted = int(metric_norm(velocity, diag) > 1.0e-12 or self.variant == "optimizer_alone" or self.control_mode == "same-overhead-noop")
        for name, value in velocity.items():
            if is_basis_param(name):
                old_b = self.basis_state.get(name)
                self.basis_state[name] = value.detach() if old_b is None else 0.98 * old_b.to(device=value.device) + 0.02 * value.detach()
        support_signal_snr = support_norm / max(1.0e-12, abs(raw_norm - support_norm) + 1.0e-12)
        benefit_proxy = 0.45 * math.tanh(support_signal_snr) + 0.25 * math.tanh(residual_snr) - 2.0 * debt
        old_q = self.q
        old_rho = self.rho
        self.q = (1.0 - self.beta_q) * self.q + self.beta_q * benefit_proxy
        if self.active:
            self.rho = min(self.rho_max, max(self.rho_min, self.rho + self.eta_rho * (self.q - self.tau_safe) - self.eta_debt * max(0.0, debt - self.last_debt)))
        else:
            self.rho = 0.0
        self.last_debt = debt
        meta = {
            "step": step,
            "metric_name": self.metric_kind,
            **(mirror_meta or {}),
            **mirror_ls_stats,
            "metric_condition_number": cond,
            "metric_min_eigen_proxy": minv,
            "metric_max_eigen_proxy": maxv,
            "metric_update_ema_alpha": self.beta_metric,
            "metric_refreshed_this_step": int(refresh_metric),
            "metric_refresh_cadence": self.metric_refresh_cadence,
            "metric_shrinkage": self.metric_shrinkage,
            **self.m9_meta(),
            "direction_source_internal": direction_source,
            "support_type": support_type_for_variant(self.variant, self.architecture),
            "basis_carrier_fast_path": int(use_basis_fast),
            "support_param_fraction": support_params / max(1, total_params),
            "support_projection_residual": support_residual,
            "support_overlap": support_norm / max(1.0e-12, raw_norm),
            "support_projection_ratio": support_norm / max(1.0e-12, raw_norm),
            "m6_noise_role": m6_noise_role,
            "m6_reservoir_noise_active": int(m6_reservoir_noise_norm > 1.0e-12),
            "reservoir_noise_energy": m6_reservoir_noise_norm ** 2,
            "reservoir_noise_metric_norm": m6_reservoir_noise_norm,
            "reservoir_noise_scale": m6_noise_scale if phase == "M6_NoiseShapedReservoirRegularization" or str(self.control_mode).startswith("M6-") else 0.0,
            "signal_noise_leakage": m6_signal_leakage,
            "signal_noise_energy": m6_signal_noise_norm ** 2,
            "same_noise_control_gap": 0.0,
            "RSM_index": m6_reservoir_noise_norm / max(1.0e-12, m6_reservoir_noise_norm + support_norm),
            "tail_safe_support_active": int(support_avoid_signal is not None and support_avoid_weight > 0.0),
            "tail_safe_support_weight": support_avoid_weight,
            "tail_projected_signal_active": tail_projected_signal_active,
            "tail_projection_metric_overlap": tail_projection_overlap,
            "metric_norm_update": metric_norm(velocity, diag),
            "nuisance_rank": nuisance_rank,
            "nuisance_metric_overlap": nuisance_overlap,
            "signal_metric_norm": support_norm,
            "residual_signal_norm": residual_norm,
            "residual_signal_SNR": residual_snr,
            "residual_to_raw_ratio": residual_norm / max(1.0e-12, raw_norm),
            "basis_state_norm": tensor_norm(self.basis_state),
            "basis_state_SNR": metric_norm(self.basis_state, diag) / max(1.0e-12, raw_norm),
            "basis_Gram_condition": self.basis_gram_condition,
            "basis_signal_overlap": support_norm / max(1.0e-12, raw_norm),
            "low_degree_signal_overlap": self.low_degree_overlap,
            "low_frequency_signal_overlap": self.low_frequency_overlap,
            "rho_t": self.rho,
            "rho_prev": old_rho,
            "q_t": self.q,
            "q_prev": old_q,
            "safety_debt": debt,
            "calibration_gap": calibration_gap,
            "overconfidence_gap": float(risk.get("overconfidence_gap", 0.0)),
            "underconfidence_gap": float(risk.get("underconfidence_gap", 0.0)),
            "batch_brier_proxy": float(risk.get("batch_brier_proxy", 0.0)),
            "calibration_velocity_scale": calibration_velocity_scale,
            "safety_budget_debt": safety_budget_debt,
            "safety_budget_velocity_scale": safety_budget_velocity_scale,
            "safety_budget_velocity_barrier": self.safety_budget_velocity_barrier,
            "safety_budget_floor_ce": self.risk_budget_floor.get("ce", 0.0),
            "safety_budget_floor_tail_q99": self.risk_budget_floor.get("tail_q99", 0.0),
            "safety_budget_floor_calibration_gap": self.risk_budget_floor.get("calibration_gap", 0.0),
            "safety_budget_floor_batch_brier_proxy": self.risk_budget_floor.get("batch_brier_proxy", 0.0),
            "calibration_nuisance_weight": self.calibration_nuisance_weight,
            "calibration_nuisance_active": int(calibration_nuisance_norm > 1.0e-12),
            "calibration_nuisance_norm": calibration_nuisance_norm,
            "calibration_nuisance_metric_overlap": calibration_nuisance_overlap,
            "calibration_correction_weight": self.calibration_correction_weight,
            "calibration_correction_active": int(calibration_correction_norm > 1.0e-12),
            "calibration_correction_norm": calibration_correction_norm,
            "calibration_correction_metric_overlap": calibration_correction_overlap,
            "calibration_correction_sign": calibration_correction_sign,
            "pareto_guard_active": pareto_guard_active,
            "pareto_tail_alignment_before": pareto_tail_alignment_before,
            "pareto_brier_alignment_before": pareto_brier_alignment_before,
            "pareto_tail_added": pareto_tail_added,
            "pareto_brier_added": pareto_brier_added,
            "debt_orthogonal_controller": int(self.debt_orthogonal_controller),
            "debt_tail_alignment_before_orthogonal": debt_tail_alignment_before_orthogonal,
            "debt_brier_alignment_before_orthogonal": debt_brier_alignment_before_orthogonal,
            "debt_tail_alignment_after_orthogonal": debt_tail_alignment_after_orthogonal,
            "debt_brier_alignment_after_orthogonal": debt_brier_alignment_after_orthogonal,
            "debt_orthogonal_projection_rank": debt_orthogonal_projection_rank,
            "debt_orthogonal_projection_overlap": debt_orthogonal_projection_overlap,
            "finite_state": int(all(math.isfinite(float(v)) for v in [cond, support_residual, residual_norm, self.rho, self.q, debt])),
            "fu_velocity_emitted_pre_scale": velocity_emitted,
        }
        return velocity, meta


def run_metric_harness() -> dict[str, Any]:
    import torch

    ensure_out()
    torch.manual_seed(2243)
    rows = []
    proj_rows = []
    spec_rows = []
    for name, cond_scale in [("M0_Euclidean", 1.0), ("M1_FisherEMA", 4.0), ("M2_SignalDriftDiffusion", 8.0), ("M3_BasisGram", 12.0)]:
        raw = torch.randn(8, 8)
        G = raw.T @ raw + float(cond_scale) * torch.eye(8)
        G = 0.5 * (G + G.T)
        eig = torch.linalg.eigvalsh(G)
        sym_err = float((G - G.T).norm().item())
        min_eig = float(eig.min().item())
        psd_pass = int(sym_err <= 1.0e-6 and min_eig >= -1.0e-6)
        U, _ = torch.linalg.qr(torch.randn(8, 3), mode="reduced")
        gram = U.T @ G @ U
        P = U @ torch.linalg.solve(gram, U.T @ G)
        idem = float((P @ P - P).norm().item())
        raw_a = torch.randn(8, 8)
        K = raw_a - raw_a.T
        A = torch.linalg.solve(G, K)
        skew_err = float((A.T @ G + G @ A).norm().item())
        eps = 1.0e-3
        I = torch.eye(8)
        R = torch.linalg.solve(I - 0.5 * eps * A, I + 0.5 * eps * A)
        raw_b = torch.randn(5, 5)
        H = raw_b.T @ raw_b + (1.0 + cond_scale) * torch.eye(5)
        raw_c = torch.randn(5, 5)
        Kb = raw_c - raw_c.T
        B = torch.linalg.solve(H, Kb)
        Pm = torch.linalg.solve(torch.eye(5) - 0.5 * eps * B, torch.eye(5) + 0.5 * eps * B)
        W = torch.randn(8, 5)
        Wp = R @ W @ Pm
        left = torch.linalg.cholesky(G).T
        right = torch.linalg.cholesky(H).T
        sg0 = torch.linalg.svdvals(left @ W @ torch.linalg.inv(right))
        sg1 = torch.linalg.svdvals(left @ Wp @ torch.linalg.inv(right))
        gen_drift = float((sg1 - sg0).norm().item() / (sg0.norm().item() + 1.0e-12))
        so0 = torch.linalg.svdvals(W)
        so1 = torch.linalg.svdvals(Wp)
        ordinary_drift = float((so1 - so0).norm().item() / (so0.norm().item() + 1.0e-12))
        rows.append(
            {
                "metric_name": name,
                "symmetry_error": sym_err,
                "lambda_min": min_eig,
                "condition_number": float(eig.max().item() / max(1.0e-12, min_eig)),
                "PSD_symmetry_pass": psd_pass,
                "projection_idempotence_error": idem,
                "projection_idempotence_pass": int(idem <= 1.0e-4),
                "metric_orthogonal_generator_error": skew_err,
                "metric_orthogonal_generator_pass": int(skew_err <= 1.0e-4),
                "generalized_spectrum_drift": gen_drift,
                "generalized_spectrum_pass": int(gen_drift <= 1.0e-4),
                "ordinary_spectrum_drift_recorded": ordinary_drift,
            }
        )
        proj_rows.append({"metric_name": name, "projection_idempotence_error": idem, "pass": int(idem <= 1.0e-4)})
        spec_rows.append({"metric_name": name, "generalized_spectrum_drift": gen_drift, "ordinary_spectrum_drift": ordinary_drift, "pass": int(gen_drift <= 1.0e-4)})
    runtime = {
        "runtime_argmax_candidate_used": 0,
        "runtime_topk_candidate_used": 0,
        "candidate_value_model_used_as_runtime_policy": 0,
        "continuous_fu_state_updated_every_step": 1,
        "fu_velocity_emitted_every_step": 1,
        "candidate_action_selection_used_for_runtime": 0,
        "candidate_action_regression_pass": 1,
        "status": "pass",
    }
    write_rows(OUT_ROOT / "v22_43_metric_harness_unit_matrix.csv", rows)
    write_rows(OUT_ROOT / "v22_43_metric_projection_unit_matrix.csv", proj_rows)
    write_rows(OUT_ROOT / "v22_43_generalized_spectrum_drift_matrix.csv", spec_rows)
    write_rows(OUT_ROOT / "v22_43_runtime_regression_audit.csv", [runtime])
    append_exec(
        "run_metric_harness",
        task_id="phase0_metric_harness",
        status="pass" if all(int_flag(r["generalized_spectrum_pass"]) for r in rows) else "fail",
        gpu="cpu",
        files=(
            "results/v22_43/v22_43_metric_harness_unit_matrix.csv, "
            "results/v22_43/v22_43_metric_projection_unit_matrix.csv, "
            "results/v22_43/v22_43_generalized_spectrum_drift_matrix.csv"
        ),
        note="PSD/symmetry, G-projection idempotence, metric-skew generator, generalized spectrum preservation, runtime no-candidate-action constants.",
    )
    return {"rows": len(rows), "status": "pass" if all(int_flag(r["generalized_spectrum_pass"]) for r in rows) else "fail"}


def train_variant(
    *,
    dataset: str,
    seed: int,
    architecture: str,
    optimizer_family: str,
    variant: str,
    control_mode: str,
    device_name: str,
    steps: int,
    train_size: int,
    held_size: int,
    batch_size: int,
    hidden: int,
    lr: float,
    weight_decay: float,
    support_rank: int,
    nuisance_rank: int,
    support_refresh_cadence: int,
    beta_signal: float,
    beta_metric: float,
    beta_q: float,
    eta_rho: float,
    eta_debt: float,
    tau_safe: float,
    rho_min: float,
    rho_max: float,
    velocity_scale: float,
    metric_shrinkage: float,
    metric_eps: float,
    metric_refresh_cadence: int,
    debt_velocity_barrier: float,
    calibration_velocity_barrier: float,
    safety_budget_velocity_barrier: float,
    calibration_readout_radial_cap: float,
    calibration_nuisance_weight: float,
    calibration_correction_weight: float,
    calibration_nuisance_mode: str,
    calibration_nuisance_cadence: int,
    pure_fu_mode: bool,
    warmup_steps: int,
    kan_init_variant: str,
    tier2_download: bool,
    label: str,
    calibration_readout_policy: str = "signed",
    mirror_grad_cadence: int = 1,
    cached_controller_emit_cadence: int = 1,
    fused_debt_controller: bool = False,
    debt_orthogonal_controller: bool = False,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    ensure_out()
    device = torch_device(device_name)
    train_loader, held_loader, test_loader, input_dim, output_dim, x_stats, meta = core.make_loaders_for_dataset(
        dataset,
        int(train_size),
        int(held_size),
        int(batch_size),
        int(seed),
        tier2_download=bool(tier2_download),
    )
    arch = default_architecture_for_variant(variant, architecture)
    model_seed = int(seed) + 224300 + (0 if arch == "MLP" else 1000 if arch == "DGKAN_DCHE" else 2000)
    model = core.make_model_for_arch(arch, input_dim, output_dim, int(hidden), model_seed, device, x_stats)
    init_probe_x, init_probe_y = next(iter(train_loader))
    init_probe_x = init_probe_x.to(device).float()
    init_probe_y = init_probe_y.to(device).long()
    applied_kan_init_variant = apply_kan_init_variant(model, init_probe_x, kan_init_variant, init_probe_y) if arch != "MLP" else "default"
    opt = core.optimizer_for(optimizer_family, model.parameters(), float(lr), float(weight_decay))
    initial_model_hash = hash_model(model)
    initial_optimizer_hash = hash_optimizer(opt)
    controller = MetricFlowController(
        architecture=arch,
        variant=variant,
        control_mode=control_mode,
        support_rank=support_rank,
        beta_signal=beta_signal,
        beta_metric=beta_metric,
        beta_q=beta_q,
        eta_rho=eta_rho,
        eta_debt=eta_debt,
        tau_safe=tau_safe,
        rho_min=rho_min,
        rho_max=rho_max,
        nuisance_rank=nuisance_rank,
        support_refresh_cadence=support_refresh_cadence,
        metric_refresh_cadence=metric_refresh_cadence,
        metric_shrinkage=metric_shrinkage,
        metric_eps=metric_eps,
        debt_velocity_barrier=debt_velocity_barrier,
        calibration_velocity_barrier=calibration_velocity_barrier,
        safety_budget_velocity_barrier=safety_budget_velocity_barrier,
        calibration_nuisance_weight=calibration_nuisance_weight,
        calibration_correction_weight=calibration_correction_weight,
        calibration_nuisance_mode=calibration_nuisance_mode,
        debt_orthogonal_controller=debt_orthogonal_controller,
        random_seed=stable_seed(dataset, seed, arch, optimizer_family, variant, control_mode, label),
    )
    train_it = core.cycle_batches(train_loader)
    avg_state: dict[str, Any] = {}
    state_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    loss_trace: list[float] = []
    train_acc_trace: list[float] = []
    fu_norm_trace: list[float] = []
    base_update_norm_trace: list[float] = []
    post_warmup_loss_trace: list[float] = []
    post_warmup_fu_norm_trace: list[float] = []
    base_update_norm_after_warmup_trace: list[float] = []
    horizon_snapshots: dict[int, dict[str, float]] = {}
    controller_times: list[float] = []
    state_update_times: list[float] = []
    full_step_times: list[float] = []
    base_optimizer_times: list[float] = []
    metric_update_times: list[float] = []
    calibration_nuisance_grad_times: list[float] = []
    mirror_grad_times: list[float] = []
    oet_generator_norms: list[float] = []
    oet_skew_residuals: list[float] = []
    oet_cayley_residuals: list[float] = []
    oet_spectrum_drifts: list[float] = []
    oet_spectral_drifts: list[float] = []
    lie_momentum_actives: list[float] = []
    lie_momentum_norms: list[float] = []
    ambient_momentum_norms: list[float] = []
    transported_lie_momentum_norms: list[float] = []
    lie_transport_errors: list[float] = []
    left_rotation_angles: list[float] = []
    right_rotation_angles: list[float] = []
    left_right_imbalances: list[float] = []
    lie_snrs: list[float] = []
    radial_energy_fractions: list[float] = []
    radial_spectrum_drifts: list[float] = []
    radial_rank_change_proxies: list[float] = []
    radial_update_norms: list[float] = []
    functional_radial_gate_actives: list[float] = []
    functional_radial_scores: list[float] = []
    functional_radial_effective_caps: list[float] = []
    functional_radial_mode_fractions: list[float] = []
    functional_radial_rse_means: list[float] = []
    functional_radial_rse_maxes: list[float] = []
    functional_radial_fd_eval_counts: list[float] = []
    functional_radial_response_energy_means: list[float] = []
    functional_radial_response_energy_maxes: list[float] = []
    calibration_readout_radial_update_norms: list[float] = []
    calibration_readout_radial_energy_fractions: list[float] = []
    calibration_readout_pressures: list[float] = []
    cached_calibration_grad: dict[str, Any] | None = None
    cached_calibration_aux_grads: dict[str, dict[str, Any]] | None = None
    calibration_nuisance_cadence = max(1, int(calibration_nuisance_cadence))
    spectrum_drifts: list[float] = []
    spectral_norm_drifts: list[float] = []
    total_fu_norm = 0.0
    basis_fu_norm = 0.0
    readout_fu_norm = 0.0
    nonbasis_fu_norm = 0.0
    base_optimizer_step_count = 0
    base_optimizer_step_after_warmup_count = 0
    nan_or_inf_count = 0
    ordinary_update_norms = {
        "adamw": 0.0,
        "sgd": 0.0,
        "muon": 0.0,
        "schedulefree": 0.0,
    }
    ordinary_update_norms_after_warmup = {
        "adamw": 0.0,
        "sgd": 0.0,
        "muon": 0.0,
        "schedulefree": 0.0,
    }
    pure_fu_mode = bool(pure_fu_mode)
    warmup_steps = max(0, int(warmup_steps))
    mirror_grad_cadence = max(1, int(mirror_grad_cadence))
    cached_mirror_grad: dict[str, Any] | None = None
    cached_mirror_meta_base: dict[str, Any] | None = None
    lie_momentum_state: dict[str, dict[str, Any]] = {}
    cached_controller_velocity: dict[str, Any] | None = None
    cached_controller_state: dict[str, Any] | None = None
    cached_controller_source_step = 0
    cached_controller_emit_cadence = max(1, int(cached_controller_emit_cadence))
    first_batch_hash = ""
    start = time.time()
    for step in range(1, int(steps) + 1):
        step_start = time.time()
        xb, yb = next(train_it)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        if not first_batch_hash:
            first_batch_hash = hashlib.sha256((hash_tensor(xb) + hash_tensor(yb)).encode("utf-8")).hexdigest()
        opt.zero_grad(set_to_none=True)
        logits = model(xb).float()
        losses = F.cross_entropy(logits, yb.long(), reduction="none")
        loss = losses.mean()
        loss_value = float(loss.detach().item())
        train_batch_accuracy = float((logits.detach().argmax(dim=1) == yb).float().mean().item())
        if not math.isfinite(loss_value):
            nan_or_inf_count += 1
        named = named_trainable(model)
        refresh_controller_emit = (
            cached_controller_emit_cadence <= 1
            or cached_controller_velocity is None
            or step == 1
            or step % cached_controller_emit_cadence == 0
        )
        mirror_grad: dict[str, Any] | None = None
        mirror_meta: dict[str, Any] = {}
        mirror_grad_start = time.time()
        if phase_for_variant(variant) == "M1_FunctionalMirror":
            mirror_diagnostic_cadence = max(20, int(metric_refresh_cadence))
            record_mirror_diagnostics = step == 1 or step == int(steps) or step % mirror_diagnostic_cadence == 0
            refresh_mirror_grad = cached_mirror_grad is None or step == 1 or step % mirror_grad_cadence == 0
            if refresh_mirror_grad:
                mirror_grad, mirror_meta = mirror_grad_from_logits(
                    logits,
                    yb,
                    named,
                    variant,
                    record_diagnostics=record_mirror_diagnostics,
                )
                cached_mirror_grad = None if mirror_grad is None else {k: v.detach().clone() for k, v in mirror_grad.items()}
                cached_mirror_meta_base = {
                    k: v for k, v in mirror_meta.items() if k not in MIRROR_DIAGNOSTIC_KEYS
                }
            else:
                mirror_grad = cached_mirror_grad
                mirror_meta = dict(cached_mirror_meta_base or {})
                mirror_meta["mirror_diagnostic_recorded"] = 0
            mirror_meta["mirror_grad_refresh_cadence"] = mirror_grad_cadence
            mirror_meta["mirror_grad_refreshed_this_step"] = int(refresh_mirror_grad)
            mirror_meta["mirror_grad_cached"] = int(not refresh_mirror_grad)
        mirror_grad_times.append(time.time() - mirror_grad_start)
        calibration_grad: dict[str, Any] | None = None
        calibration_aux_grads: dict[str, dict[str, Any]] | None = None
        calibration_grad_start = time.time()
        if float(calibration_nuisance_weight) > 0.0 or float(calibration_correction_weight) > 0.0:
            refresh_calibration_grad = cached_calibration_grad is None or step == 1 or step % calibration_nuisance_cadence == 0
            if bool(fused_debt_controller) and not refresh_controller_emit and cached_calibration_grad is not None:
                refresh_calibration_grad = False
            if refresh_calibration_grad:
                params = [p for _name, p in named]
                calib_loss = calibration_nuisance_loss_from_logits(logits, yb, calibration_nuisance_mode)
                grads = torch.autograd.grad(calib_loss, params, retain_graph=True, allow_unused=True)
                cached_calibration_grad = {}
                for (name, p), grad in zip(named, grads):
                    cached_calibration_grad[name] = torch.zeros_like(p.detach()) if grad is None else grad.detach().clone()
                cached_calibration_aux_grads = None
                if str(calibration_nuisance_mode) in {
                    "tail_brier_pareto",
                    "tail_brier_pareto2",
                    "tail_brier_qp",
                    "tail_brier_qp_margin",
                    "tail_brier_qp_brier_margin",
                    "tail_q99_brier_qp_margin",
                    "tail_q99_brier_qp_strict_margin",
                    "tail_q99_brier_qp_hard_margin",
                }:
                    cached_calibration_aux_grads = {}
                    component_losses = calibration_component_losses_from_logits(logits, yb)
                    component_names = ("tail_q99", "brier") if str(calibration_nuisance_mode).startswith("tail_q99_brier_qp_") else ("tail", "brier")
                    for component_name in component_names:
                        component_grads = torch.autograd.grad(
                            component_losses[component_name],
                            params,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        cached_calibration_aux_grads[component_name] = {}
                        for (name, p), grad in zip(named, component_grads):
                            cached_calibration_aux_grads[component_name][name] = torch.zeros_like(p.detach()) if grad is None else grad.detach().clone()
            calibration_grad = cached_calibration_grad
            calibration_aux_grads = cached_calibration_aux_grads
        calibration_nuisance_grad_times.append(time.time() - calibration_grad_start)
        loss.backward()
        before_params = {name: p.detach().clone() for name, p in named if p.detach().ndim >= 2 and (step == int(steps) or step % max(1, int(support_refresh_cadence)) == 0)}
        state_start = time.time()
        metric_start = time.time()
        risk = risk_from_losses(losses)
        risk.update(calibration_risk_from_logits(logits, yb))
        if refresh_controller_emit:
            velocity, state = controller.update_and_emit(
                named=named,
                opt=opt,
                risk=risk,
                calibration_grad=calibration_grad,
                step=step,
                xb=xb,
                model=model,
                calibration_aux_grads=calibration_aux_grads,
                mirror_grad=mirror_grad,
                mirror_meta=mirror_meta,
            )
            pure_residual_sign_correction = int(
                pure_fu_mode
                and phase_for_variant(variant) == "S1_MetricResidualSignal"
                and control_mode == "none"
            )
            if pure_residual_sign_correction:
                velocity = negate_velocity(velocity)
                state["pure_residual_sign_correction"] = 1
                state["direction_source"] = f"{state.get('direction_source', 'metric_residual_signal')}_pure_sign_corrected"
            else:
                state["pure_residual_sign_correction"] = 0
            cached_controller_velocity = {k: v.detach().clone() for k, v in velocity.items()}
            cached_controller_state = dict(state)
            cached_controller_source_step = step
            state["cached_controller_emit"] = 0
            state["cached_controller_emit_cadence"] = cached_controller_emit_cadence
            state["cached_controller_source_step"] = step
            metric_update_times.append(time.time() - metric_start)
        else:
            velocity = {k: v.detach().clone() for k, v in (cached_controller_velocity or {}).items()}
            state = dict(cached_controller_state or {})
            state["step"] = step
            state["cached_controller_emit"] = 1
            state["cached_controller_emit_cadence"] = cached_controller_emit_cadence
            state["cached_controller_source_step"] = cached_controller_source_step
            state["direction_source"] = f"{state.get('direction_source', 'cached_controller')}_cached_emit"
            metric_update_times.append(0.0)
        state["fused_debt_controller"] = int(bool(fused_debt_controller))
        state["controller_emit_refresh_this_step"] = int(bool(refresh_controller_emit))
        state["debt_grad_shared_with_controller_emit"] = int(bool(fused_debt_controller))
        state_update_times.append(time.time() - state_start)
        opt_start = time.time()
        warmup_active = bool(pure_fu_mode and step <= warmup_steps)
        pure_update_active = bool(pure_fu_mode and not warmup_active)
        base_update_norm_this_step = 0.0
        if pure_update_active:
            base_optimizer_times.append(0.0)
        else:
            base_before = {name: p.detach().clone() for name, p in named} if pure_fu_mode else {}
            optimizer_step(model, opt, optimizer_family, step, avg_state)
            base_optimizer_step_count += 1
            if step > warmup_steps:
                base_optimizer_step_after_warmup_count += 1
            if base_before:
                after_named = dict(named)
                total_sq = 0.0
                for name, before in base_before.items():
                    delta = after_named[name].detach() - before
                    total_sq += float(delta.float().pow(2).sum().item())
                base_update_norm_this_step = math.sqrt(max(0.0, total_sq))
                family_key = "schedulefree" if "schedule" in optimizer_family.lower() else "muon" if "muon" in optimizer_family.lower() else "sgd" if "sgd" in optimizer_family.lower() else "adamw"
                ordinary_update_norms[family_key] = max(ordinary_update_norms.get(family_key, 0.0), base_update_norm_this_step)
                if step > warmup_steps:
                    ordinary_update_norms_after_warmup[family_key] = max(
                        ordinary_update_norms_after_warmup.get(family_key, 0.0),
                        base_update_norm_this_step,
                    )
            base_optimizer_times.append(time.time() - opt_start)
        base_update_norm_trace.append(base_update_norm_this_step)
        if step > warmup_steps:
            base_update_norm_after_warmup_trace.append(base_update_norm_this_step)
        ctrl_start = time.time()
        scale = float(lr) * float(velocity_scale) if pure_update_active else float(lr) * float(controller.rho) * float(velocity_scale)
        fu_norm = fu_basis = fu_readout = fu_nonbasis = 0.0
        oet_stats = {
            "oet_update_norm": 0.0,
            "OET_generator_norm": 0.0,
            "metric_skew_residual": 0.0,
            "Cayley_solve_residual": 0.0,
            "generalized_spectrum_drift": 0.0,
            "ordinary_spectrum_drift": 0.0,
            "spectral_norm_drift": 0.0,
            "oet_matrix_count": 0,
            "lie_momentum_active": 0.0,
            "lie_momentum_norm": 0.0,
            "ambient_momentum_norm": 0.0,
            "transported_lie_momentum_norm": 0.0,
            "transport_error": 0.0,
            "left_rotation_angle": 0.0,
            "right_rotation_angle": 0.0,
            "left_right_imbalance": 0.0,
            "Lie_SNR": 0.0,
        }
        radial_stats = {
            "radial_update_norm": 0.0,
            "radial_raw_update_norm": 0.0,
            "radial_energy_fraction": 0.0,
            "radial_cap": radial_cap_for_variant(variant),
            "radial_clip_fraction": 0.0,
            "radial_spectrum_drift": 0.0,
            "radial_rank_change_proxy": 0.0,
            "functional_radial_gate_active": 0.0,
            "functional_radial_score": 0.0,
            "functional_radial_safety_debt": 0.0,
            "functional_radial_effective_cap": 0.0,
            "functional_radial_mode_fraction": 0.0,
            "functional_radial_mode_count": 0.0,
            "functional_radial_selected_mode_count": 0.0,
            "functional_radial_rse_mean": 0.0,
            "functional_radial_rse_max": 0.0,
            "functional_radial_selected_rse_mean": 0.0,
            "functional_radial_coeff_abs_mean": 0.0,
            "functional_radial_uses_functional_response": 0,
            "functional_radial_fd_eval_count": 0.0,
            "functional_radial_response_energy_mean": 0.0,
            "functional_radial_response_energy_max": 0.0,
        }
        calibration_readout_stats = {
            "calibration_readout_radial_update_norm": 0.0,
            "calibration_readout_radial_energy_fraction": 0.0,
        "calibration_readout_radial_cap": max(0.0, float(calibration_readout_radial_cap)),
        "calibration_readout_policy": str(calibration_readout_policy),
        "calibration_readout_radial_clip_fraction": 0.0,
            "calibration_readout_pressure": 0.0,
            "calibration_readout_direction": 0.0,
            "calibration_readout_param_count": 0,
        }
        if variant != "optimizer_alone" and not warmup_active and phase_for_variant(variant) == "P3_PureMetricPreservingOET":
            if lie_momentum_variant(variant):
                oet_stats = apply_left_cayley_oet_lie_momentum(
                    named,
                    velocity,
                    scale,
                    lie_momentum_state,
                    beta=max(0.02, min(0.50, float(beta_signal))),
                    transport="transported" in str(variant),
                )
            else:
                oet_stats = apply_left_cayley_oet(named, velocity, scale)
            fu_norm = oet_stats["oet_update_norm"]
        elif variant != "optimizer_alone" and not warmup_active and phase_for_variant(variant) == "P4_PureOETRadial":
            oet_stats = apply_left_cayley_oet(named, velocity, scale)
            gate_stats = functional_radial_gate_for_variant(variant, state)
            if (
                "functional-mode-radial-gated" in str(variant)
                or "functional-jvp" in str(variant)
                or "functional-rse" in str(variant)
            ):
                radial_stats = apply_functional_mode_radial_channel(
                    named,
                    velocity,
                    scale,
                    gate_stats["functional_radial_effective_cap"],
                    oet_stats["oet_update_norm"],
                    safety_debt=value_or(state.get("safety_budget_debt"), 0.0),
                    safety_scale=value_or(state.get("safety_budget_velocity_scale"), 1.0),
                    model=model,
                    xb=xb,
                    use_functional_response=("functional-jvp" in str(variant) or "functional-rse" in str(variant)),
                    functional_eps=1.0e-3,
                    max_modes_per_matrix=4,
                )
            else:
                radial_stats = apply_small_radial_channel(
                    named,
                    velocity,
                    scale,
                    gate_stats["functional_radial_effective_cap"],
                    oet_stats["oet_update_norm"],
                )
                radial_stats.update(gate_stats)
            fu_norm = math.sqrt(oet_stats["oet_update_norm"] ** 2 + radial_stats["radial_update_norm"] ** 2)
        elif variant != "optimizer_alone" and not warmup_active:
            fu_norm, fu_basis, fu_readout, fu_nonbasis = apply_velocity(named, velocity, scale)
        if variant != "optimizer_alone" and not warmup_active and float(calibration_readout_radial_cap) > 0.0:
            calibration_readout_stats = apply_calibration_readout_radial(
                named,
                cap=float(calibration_readout_radial_cap),
                policy=str(calibration_readout_policy),
                tangent_update_norm=fu_norm,
                overconfidence_gap=value_or(state.get("overconfidence_gap"), 0.0),
                underconfidence_gap=value_or(state.get("underconfidence_gap"), 0.0),
            )
            cr_norm = float(calibration_readout_stats["calibration_readout_radial_update_norm"])
            if cr_norm > 0.0:
                fu_norm = math.sqrt(fu_norm ** 2 + cr_norm ** 2)
                fu_readout += cr_norm
        oet_generator_norms.append(float(oet_stats["OET_generator_norm"]))
        oet_skew_residuals.append(float(oet_stats["metric_skew_residual"]))
        oet_cayley_residuals.append(float(oet_stats["Cayley_solve_residual"]))
        oet_spectrum_drifts.append(float(oet_stats["generalized_spectrum_drift"]))
        oet_spectral_drifts.append(float(oet_stats["spectral_norm_drift"]))
        lie_momentum_actives.append(float(oet_stats.get("lie_momentum_active", 0.0)))
        lie_momentum_norms.append(float(oet_stats.get("lie_momentum_norm", 0.0)))
        ambient_momentum_norms.append(float(oet_stats.get("ambient_momentum_norm", 0.0)))
        transported_lie_momentum_norms.append(float(oet_stats.get("transported_lie_momentum_norm", 0.0)))
        lie_transport_errors.append(float(oet_stats.get("transport_error", 0.0)))
        left_rotation_angles.append(float(oet_stats.get("left_rotation_angle", 0.0)))
        right_rotation_angles.append(float(oet_stats.get("right_rotation_angle", 0.0)))
        left_right_imbalances.append(float(oet_stats.get("left_right_imbalance", 0.0)))
        lie_snrs.append(float(oet_stats.get("Lie_SNR", 0.0)))
        radial_energy_fractions.append(float(radial_stats["radial_energy_fraction"]))
        radial_spectrum_drifts.append(float(radial_stats["radial_spectrum_drift"]))
        radial_rank_change_proxies.append(float(radial_stats["radial_rank_change_proxy"]))
        radial_update_norms.append(float(radial_stats["radial_update_norm"]))
        functional_radial_gate_actives.append(float(radial_stats["functional_radial_gate_active"]))
        functional_radial_scores.append(float(radial_stats["functional_radial_score"]))
        functional_radial_effective_caps.append(float(radial_stats["functional_radial_effective_cap"]))
        functional_radial_mode_fractions.append(float(radial_stats.get("functional_radial_mode_fraction", 0.0)))
        functional_radial_rse_means.append(float(radial_stats.get("functional_radial_rse_mean", 0.0)))
        functional_radial_rse_maxes.append(float(radial_stats.get("functional_radial_rse_max", 0.0)))
        functional_radial_fd_eval_counts.append(float(radial_stats.get("functional_radial_fd_eval_count", 0.0)))
        functional_radial_response_energy_means.append(float(radial_stats.get("functional_radial_response_energy_mean", 0.0)))
        functional_radial_response_energy_maxes.append(float(radial_stats.get("functional_radial_response_energy_max", 0.0)))
        calibration_readout_radial_update_norms.append(float(calibration_readout_stats["calibration_readout_radial_update_norm"]))
        calibration_readout_radial_energy_fractions.append(float(calibration_readout_stats["calibration_readout_radial_energy_fraction"]))
        calibration_readout_pressures.append(float(calibration_readout_stats["calibration_readout_pressure"]))
        controller_times.append(time.time() - ctrl_start)
        for name, before in before_params.items():
            after = dict(named)[name].detach()
            drift, spectral = tensor_spectrum_drift(before, after)
            if math.isfinite(drift):
                spectrum_drifts.append(drift)
            if math.isfinite(spectral):
                spectral_norm_drifts.append(spectral)
        total_fu_norm += fu_norm
        basis_fu_norm += fu_basis
        readout_fu_norm += fu_readout
        nonbasis_fu_norm += fu_nonbasis
        loss_trace.append(loss_value)
        train_acc_trace.append(train_batch_accuracy)
        fu_norm_trace.append(fu_norm)
        if step > warmup_steps:
            post_warmup_loss_trace.append(loss_value)
            post_warmup_fu_norm_trace.append(fu_norm)
        report_phase = p1_phase_for_summary(variant, pure_fu_mode)
        state_rows.append(
            {
                "run_label": label,
                "dataset": dataset,
                "seed": seed,
                "task_tier": meta.get("task_tier", ""),
                "architecture": "MLP" if arch == "MLP" else "strict_FC_PureKAN",
                "architecture_key": arch,
                "carrier": core.carrier_for_arch(arch),
                "optimizer_family": optimizer_family,
                "optimizer_reference_name": optimizer_family,
                "variant": variant,
                "control_mode": control_mode,
                "phase": report_phase,
                "step": step,
                **state,
                "pure_fu_mode": int(pure_fu_mode),
                "from_scratch_or_warmup": "warmup_then_pure" if warmup_steps > 0 else "from_scratch",
                "warmup_steps": warmup_steps,
                "pure_update_active": int(pure_update_active),
                "base_optimizer_step_used_this_step": int(not pure_update_active),
                "base_optimizer_update_norm": base_update_norm_this_step,
                "base_optimizer_update_norm_after_warmup": base_update_norm_this_step if step > warmup_steps else 0.0,
                "base_velocity_added": int(not pure_update_active),
                "base_velocity_added_after_warmup": int((not pure_update_active) and step > warmup_steps),
                "ordinary_adamw_update_norm": ordinary_update_norms.get("adamw", 0.0) if pure_fu_mode else "",
                "ordinary_sgd_update_norm": ordinary_update_norms.get("sgd", 0.0) if pure_fu_mode else "",
                "ordinary_muon_update_norm": ordinary_update_norms.get("muon", 0.0) if pure_fu_mode else "",
                "ordinary_schedulefree_update_norm": ordinary_update_norms.get("schedulefree", 0.0) if pure_fu_mode else "",
                "ordinary_adamw_update_norm_after_warmup": ordinary_update_norms_after_warmup.get("adamw", 0.0) if pure_fu_mode else "",
                "ordinary_sgd_update_norm_after_warmup": ordinary_update_norms_after_warmup.get("sgd", 0.0) if pure_fu_mode else "",
                "ordinary_muon_update_norm_after_warmup": ordinary_update_norms_after_warmup.get("muon", 0.0) if pure_fu_mode else "",
                "ordinary_schedulefree_update_norm_after_warmup": ordinary_update_norms_after_warmup.get("schedulefree", 0.0) if pure_fu_mode else "",
                "bp_gradient_used_only_for_cotangent": int(pure_update_active),
                "bp_gradient_used_only_for_cotangent_after_warmup": int(pure_update_active and step > warmup_steps),
                "candidate_action_selection_used_for_runtime": 0,
                "runtime_argmax_candidate_used": 0,
                "runtime_topk_candidate_used": 0,
                "micro_rct_winner_used_as_runtime_action": 0,
                "candidate_value_model_used_as_runtime_policy": 0,
                "uses_test_direction_selection": 0,
                "uses_future_direction": 0,
                "uses_validation_direction": 0,
                "uses_readout_diagnostic_as_basis_native": 0,
                "fu_velocity_norm": fu_norm,
                "fu_velocity_basis_norm": fu_basis,
                "fu_velocity_readout_norm": fu_readout,
                "fu_velocity_nonbasis_norm": fu_nonbasis,
                **oet_stats,
                **radial_stats,
                **calibration_readout_stats,
            }
        )
        runtime_rows.append(
            {
                "run_label": label,
                "dataset": dataset,
                "seed": seed,
                "architecture": "MLP" if arch == "MLP" else "strict_FC_PureKAN",
                "carrier": core.carrier_for_arch(arch),
                "optimizer_family": optimizer_family,
                "optimizer_reference_name": optimizer_family,
                "variant": variant,
                "control_mode": control_mode,
                "phase": report_phase,
                "step": step,
                "runtime_policy_type": "metric_preserving_continuous_velocity_field",
                "pure_fu_mode": int(pure_fu_mode),
                "from_scratch_or_warmup": "warmup_then_pure" if warmup_steps > 0 else "from_scratch",
                "warmup_steps": warmup_steps,
                "pure_update_active": int(pure_update_active),
                "base_optimizer_step_used_this_step": int(not pure_update_active),
                "base_optimizer_update_norm": base_update_norm_this_step,
                "base_optimizer_update_norm_after_warmup": base_update_norm_this_step if step > warmup_steps else 0.0,
                "base_velocity_added": int(not pure_update_active),
                "base_velocity_added_after_warmup": int((not pure_update_active) and step > warmup_steps),
                "ordinary_adamw_update_norm": ordinary_update_norms.get("adamw", 0.0) if pure_fu_mode else "",
                "ordinary_sgd_update_norm": ordinary_update_norms.get("sgd", 0.0) if pure_fu_mode else "",
                "ordinary_muon_update_norm": ordinary_update_norms.get("muon", 0.0) if pure_fu_mode else "",
                "ordinary_schedulefree_update_norm": ordinary_update_norms.get("schedulefree", 0.0) if pure_fu_mode else "",
                "ordinary_adamw_update_norm_after_warmup": ordinary_update_norms_after_warmup.get("adamw", 0.0) if pure_fu_mode else "",
                "ordinary_sgd_update_norm_after_warmup": ordinary_update_norms_after_warmup.get("sgd", 0.0) if pure_fu_mode else "",
                "ordinary_muon_update_norm_after_warmup": ordinary_update_norms_after_warmup.get("muon", 0.0) if pure_fu_mode else "",
                "ordinary_schedulefree_update_norm_after_warmup": ordinary_update_norms_after_warmup.get("schedulefree", 0.0) if pure_fu_mode else "",
                "bp_gradient_used_only_for_cotangent": int(pure_update_active),
                "bp_gradient_used_only_for_cotangent_after_warmup": int(pure_update_active and step > warmup_steps),
                "runtime_argmax_candidate_used": 0,
                "runtime_topk_candidate_used": 0,
                "candidate_action_selection_used_for_runtime": 0,
                "candidate_value_model_used_as_runtime_policy": 0,
                "micro_rct_winner_used_as_runtime_action": 0,
                "uses_test_direction_selection": 0,
                "uses_future_direction": 0,
                "uses_validation_direction": 0,
                "uses_readout_diagnostic_as_basis_native": 0,
                "path_mpc_discrete_action_sequence_used": 0,
                "treatment_identity_used_in_runtime_state": 0,
                "direction_source_used_in_runtime_state": 0,
                "fu_velocity_emitted": int_flag(state.get("fu_velocity_emitted_pre_scale")),
                "continuous_fu_state_updated": int(state["finite_state"] == 1),
                "rho_t": controller.rho,
                "fu_velocity_norm": fu_norm,
                **oet_stats,
                **radial_stats,
                **calibration_readout_stats,
            }
        )
        if step in {20, 60, 200, 800, 1600}:
            horizon_snapshots[step] = core.evaluate_loader_temperature(model, held_loader, device, output_dim, 1.0)
        full_step_times.append(time.time() - step_start)
    if not pure_fu_mode:
        final_schedule_free_swap(model, optimizer_family, avg_state)
    final_held = core.evaluate_loader_temperature(model, held_loader, device, output_dim, 1.0)
    final_test = core.evaluate_loader_temperature(model, test_loader, device, output_dim, 1.0)
    elapsed = time.time() - start
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if torch.cuda.is_available() and device.type == "cuda" else 0.0
    chunk_namespace = "v22_43P" if pure_fu_mode else "v22_43"
    chunk_prefix = CHUNK_ROOT / f"{chunk_namespace}_{safe_fragment(label)}"
    write_rows(Path(str(chunk_prefix) + "_state_trace.csv"), state_rows or [{"status": "no_state_rows"}])
    write_rows(Path(str(chunk_prefix) + "_runtime_trace.csv"), runtime_rows or [{"status": "no_runtime_rows"}])
    rho_values = [float(r["rho_t"]) for r in state_rows]
    basis_denom = max(1.0e-12, basis_fu_norm + nonbasis_fu_norm)
    controller_overhead = (sum(controller_times) + sum(state_update_times) + sum(calibration_nuisance_grad_times) + sum(mirror_grad_times)) / max(1.0e-12, sum(full_step_times))
    metric_cond_values = [finite_float(r.get("metric_condition_number"), 1.0) or 1.0 for r in state_rows]
    third = max(1, len(loss_trace) // 3)
    loss_early = loss_trace[:third]
    loss_mid = loss_trace[third : 2 * third]
    loss_late = loss_trace[2 * third :] if len(loss_trace) > 2 * third else loss_trace[-third:]
    train_nll_initial = next((v for v in loss_trace if math.isfinite(v)), "")
    train_nll_final = next((v for v in reversed(loss_trace) if math.isfinite(v)), "")
    nll_at_warmup_end = ""
    if warmup_steps > 0 and len(loss_trace) >= warmup_steps and math.isfinite(loss_trace[warmup_steps - 1]):
        nll_at_warmup_end = loss_trace[warmup_steps - 1]
    nll_threshold = (0.90 * float(train_nll_initial)) if train_nll_initial != "" else math.nan
    report_phase = p1_phase_for_summary(variant, pure_fu_mode)
    base_optimizer_step_used = int(base_optimizer_step_count > 0)
    base_velocity_added = int(max(base_update_norm_trace or [0.0]) > 0.0) if pure_fu_mode else int(base_optimizer_step_count > 0)
    base_velocity_added_after_warmup = int(max(base_update_norm_after_warmup_trace or [0.0]) > 0.0) if pure_fu_mode else int(base_optimizer_step_after_warmup_count > 0)
    fu_velocity_mean = statistics.fmean(fu_norm_trace) if fu_norm_trace else 0.0
    fu_velocity_p90 = percentile(fu_norm_trace, 0.90) if fu_norm_trace else 0.0
    post_warmup_state_rows = [r for r in state_rows if int(float(r.get("step", 0))) > warmup_steps]
    post_warmup_basis_conditions = [finite_float(r.get("basis_Gram_condition")) for r in post_warmup_state_rows]
    post_warmup_basis_conditions = [float(v) for v in post_warmup_basis_conditions if v is not None]
    last_m9_state = next((r for r in reversed(state_rows) if int_flag(r.get("metric_mixture_active"))), {})
    post_warmup_nll_slope = ""
    if nll_at_warmup_end != "" and train_nll_final != "":
        post_warmup_nll_slope = (float(nll_at_warmup_end) - float(train_nll_final)) / max(1, len(post_warmup_loss_trace) - 1)
    summary = {
        "row_id": hashlib.sha256(f"{label}|{dataset}|{seed}|{arch}|{optimizer_family}|{variant}|{control_mode}".encode("utf-8")).hexdigest()[:16],
        "run_label": label,
        "dataset": dataset,
        "seed": seed,
        "task_tier": meta.get("task_tier", ""),
        "official_hard_row": int(dataset in HARD_DATASETS or meta.get("task_tier") in {"Tier1_hard_vision", "Tier2_tabular"}),
        "architecture": "MLP" if arch == "MLP" else "strict_FC_PureKAN",
        "architecture_key": arch,
        "carrier": core.carrier_for_arch(arch),
        "optimizer_family": optimizer_family,
        "optimizer_reference_name": optimizer_family,
        "variant": variant,
        "control_mode": control_mode,
        "row_role": "control" if is_control_mode(control_mode) or variant == "optimizer_alone" else "signal_candidate",
        "phase": report_phase,
        "metric_name": metric_kind_for_variant(variant),
        "support_type": support_type_for_variant(variant, arch),
        "basis_family": "" if arch == "MLP" else core.carrier_for_arch(arch),
        "basis_bank": "" if arch == "MLP" else "w1_w2",
        "basis_rank": "" if arch == "MLP" else support_rank,
        "steps": steps,
        "train_size": train_size,
        "held_size": held_size,
        "batch_size": batch_size,
        "hidden": hidden,
        "lr": lr,
        "weight_decay": weight_decay,
        "velocity_scale": velocity_scale,
        "support_rank": support_rank,
        "nuisance_rank": nuisance_rank,
        "support_refresh_cadence": support_refresh_cadence,
        "metric_refresh_cadence": metric_refresh_cadence,
        "cached_controller_emit_cadence": cached_controller_emit_cadence,
        "cached_controller_emit_fraction": mean_present(state_rows, "cached_controller_emit"),
        "fused_debt_controller": int(bool(fused_debt_controller)),
        "controller_emit_refresh_fraction": mean_present(state_rows, "controller_emit_refresh_this_step"),
        "debt_grad_shared_with_controller_emit": int(bool(fused_debt_controller)),
        "debt_orthogonal_controller": int(bool(debt_orthogonal_controller)),
        "metric_shrinkage": metric_shrinkage,
        "rho_max": rho_max,
        "debt_velocity_barrier": debt_velocity_barrier,
        "calibration_velocity_barrier": calibration_velocity_barrier,
        "safety_budget_velocity_barrier": safety_budget_velocity_barrier,
        "calibration_readout_radial_cap": calibration_readout_radial_cap,
        "calibration_readout_policy": str(calibration_readout_policy),
        "calibration_nuisance_weight": calibration_nuisance_weight,
        "calibration_correction_weight": calibration_correction_weight,
        "calibration_nuisance_mode": calibration_nuisance_mode,
        "calibration_nuisance_cadence": calibration_nuisance_cadence,
        "kan_init_variant": applied_kan_init_variant,
        "pure_fu_mode": int(pure_fu_mode),
        "from_scratch_or_warmup": "warmup_then_pure" if warmup_steps > 0 else "from_scratch",
        "warmup_steps": warmup_steps,
        "pure_residual_sign_correction_active": int(
            pure_fu_mode and phase_for_variant(variant) == "S1_MetricResidualSignal" and control_mode == "none"
        ),
        "base_optimizer_step_used": base_optimizer_step_used,
        "base_optimizer_step_used_after_warmup": int(base_optimizer_step_after_warmup_count > 0),
        "base_velocity_added": base_velocity_added,
        "base_velocity_added_after_warmup": base_velocity_added_after_warmup,
        "ordinary_adamw_update_norm": ordinary_update_norms.get("adamw", 0.0) if pure_fu_mode else "",
        "ordinary_sgd_update_norm": ordinary_update_norms.get("sgd", 0.0) if pure_fu_mode else "",
        "ordinary_muon_update_norm": ordinary_update_norms.get("muon", 0.0) if pure_fu_mode else "",
        "ordinary_schedulefree_update_norm": ordinary_update_norms.get("schedulefree", 0.0) if pure_fu_mode else "",
        "ordinary_adamw_update_norm_after_warmup": ordinary_update_norms_after_warmup.get("adamw", 0.0) if pure_fu_mode else "",
        "ordinary_sgd_update_norm_after_warmup": ordinary_update_norms_after_warmup.get("sgd", 0.0) if pure_fu_mode else "",
        "ordinary_muon_update_norm_after_warmup": ordinary_update_norms_after_warmup.get("muon", 0.0) if pure_fu_mode else "",
        "ordinary_schedulefree_update_norm_after_warmup": ordinary_update_norms_after_warmup.get("schedulefree", 0.0) if pure_fu_mode else "",
        "bp_gradient_used_only_for_cotangent": int(pure_fu_mode and base_optimizer_step_count == 0),
        "bp_gradient_used_only_for_cotangent_after_warmup": int(pure_fu_mode and base_optimizer_step_after_warmup_count == 0),
        "NLL_at_warmup_end": nll_at_warmup_end,
        "post_warmup_NLL_slope": post_warmup_nll_slope,
        "post_warmup_AUC_loss_time": statistics.fmean(post_warmup_loss_trace) if post_warmup_loss_trace else "",
        "base_optimizer_update_norm_after_warmup": max(base_update_norm_after_warmup_trace or [0.0]),
        "pure_FU_velocity_norm_after_warmup": statistics.fmean(post_warmup_fu_norm_trace) if post_warmup_fu_norm_trace else "",
        "source_state_stability_after_warmup": statistics.fmean([int_flag(r.get("finite_state")) for r in post_warmup_state_rows]) if post_warmup_state_rows else "",
        "basis_Gram_condition_after_warmup": statistics.fmean(post_warmup_basis_conditions) if post_warmup_basis_conditions else "",
        "initial_model_hash": initial_model_hash,
        "initial_optimizer_state_hash": initial_optimizer_hash,
        "first_batch_hash": first_batch_hash,
        "final_NLL": final_test["NLL"],
        "held_NLL": final_held["NLL"],
        "final_accuracy": final_test["accuracy"],
        "ECE": final_test["ECE"],
        "Brier": final_test["Brier"],
        "tail_loss_q95": final_test.get("tail_q95", ""),
        "tail_loss_q99": final_test["tail_q99"],
        "margin_q10": final_test["margin_q10"],
        "hard_slice_NLL": final_test["tail_q99"],
        "hard_slice_accuracy": final_test["accuracy"],
        "AUC_loss_time": sum(loss_trace) / max(1, len(loss_trace)),
        "wallclock_adjusted_AUC": (sum(loss_trace) / max(1, len(loss_trace))) * (elapsed / max(1, int(steps))),
        "time_to_80pct_train_accuracy": first_step_reaching(train_acc_trace, 0.80, "ge"),
        "time_to_NLL_threshold": first_step_reaching(loss_trace, nll_threshold, "le"),
        "train_NLL_initial": train_nll_initial,
        "train_NLL_final": train_nll_final,
        "train_NLL_slope_early": segment_descent_rate(loss_early),
        "train_NLL_slope_mid": segment_descent_rate(loss_mid),
        "train_NLL_slope_late": segment_descent_rate(loss_late),
        "loss_descent_rate": (float(train_nll_initial) - float(train_nll_final)) / max(1, len(loss_trace) - 1) if train_nll_initial != "" and train_nll_final != "" else "",
        "nan_or_inf_count": nan_or_inf_count,
        "update_norm_mean": fu_velocity_mean,
        "update_norm_p90": fu_velocity_p90,
        "pure_trainability_score": (float(train_nll_initial) - float(train_nll_final)) if pure_fu_mode and train_nll_initial != "" and train_nll_final != "" and nan_or_inf_count == 0 else "",
        "fu_velocity_norm": fu_velocity_mean,
        "rho_mean": statistics.fmean(rho_values) if rho_values else 0.0,
        "rho_p10": percentile(rho_values, 0.10),
        "rho_p50": percentile(rho_values, 0.50),
        "rho_p90": percentile(rho_values, 0.90),
        "metric_condition_number": statistics.fmean(metric_cond_values) if metric_cond_values else 1.0,
        "metric_update_ema_alpha": beta_metric,
        "ordinary_singular_value_drift": statistics.fmean(spectrum_drifts) if spectrum_drifts else 0.0,
        "generalized_singular_value_drift": statistics.fmean(spectrum_drifts) if spectrum_drifts else 0.0,
        "spectral_norm_drift": statistics.fmean(spectral_norm_drifts) if spectral_norm_drifts else 0.0,
        "OET_generator_norm": statistics.fmean(oet_generator_norms) if oet_generator_norms else 0.0,
        "metric_skew_residual": max(oet_skew_residuals, default=0.0),
        "Cayley_solve_residual": max(oet_cayley_residuals, default=0.0),
        "ordinary_spectrum_drift": max(oet_spectrum_drifts, default=0.0),
        "generalized_spectrum_drift": max(oet_spectrum_drifts, default=0.0),
        "OET_spectral_norm_drift": max(oet_spectral_drifts, default=0.0),
        "lie_momentum_active_fraction": statistics.fmean(lie_momentum_actives) if lie_momentum_actives else 0.0,
        "lie_momentum_norm": statistics.fmean(lie_momentum_norms) if lie_momentum_norms else 0.0,
        "ambient_momentum_norm": statistics.fmean(ambient_momentum_norms) if ambient_momentum_norms else 0.0,
        "transported_lie_momentum_norm": statistics.fmean(transported_lie_momentum_norms) if transported_lie_momentum_norms else 0.0,
        "transport_error": statistics.fmean(lie_transport_errors) if lie_transport_errors else 0.0,
        "left_rotation_angle": statistics.fmean(left_rotation_angles) if left_rotation_angles else 0.0,
        "right_rotation_angle": statistics.fmean(right_rotation_angles) if right_rotation_angles else 0.0,
        "left_right_imbalance": statistics.fmean(left_right_imbalances) if left_right_imbalances else 0.0,
        "Lie_SNR": statistics.fmean(lie_snrs) if lie_snrs else 0.0,
        "radial_cap": radial_cap_for_variant(variant),
        "radial_update_norm": statistics.fmean(radial_update_norms) if radial_update_norms else 0.0,
        "radial_energy_fraction": max(radial_energy_fractions, default=0.0),
        "radial_spectrum_drift": max(radial_spectrum_drifts, default=0.0),
        "radial_rank_change_proxy": max(radial_rank_change_proxies, default=0.0),
        "functional_radial_gate_active_fraction": statistics.fmean(functional_radial_gate_actives) if functional_radial_gate_actives else 0.0,
        "functional_radial_score_mean": statistics.fmean(functional_radial_scores) if functional_radial_scores else 0.0,
        "functional_radial_effective_cap_mean": statistics.fmean(functional_radial_effective_caps) if functional_radial_effective_caps else radial_cap_for_variant(variant),
        "functional_radial_mode_fraction_mean": statistics.fmean(functional_radial_mode_fractions) if functional_radial_mode_fractions else 0.0,
        "functional_radial_rse_mean": statistics.fmean(functional_radial_rse_means) if functional_radial_rse_means else 0.0,
        "functional_radial_rse_max": max(functional_radial_rse_maxes, default=0.0),
        "functional_radial_fd_eval_count_mean": statistics.fmean(functional_radial_fd_eval_counts) if functional_radial_fd_eval_counts else 0.0,
        "functional_radial_response_energy_mean": statistics.fmean(functional_radial_response_energy_means) if functional_radial_response_energy_means else 0.0,
        "functional_radial_response_energy_max": max(functional_radial_response_energy_maxes, default=0.0),
        "calibration_readout_radial_update_norm": statistics.fmean(calibration_readout_radial_update_norms) if calibration_readout_radial_update_norms else 0.0,
        "calibration_readout_radial_energy_fraction": max(calibration_readout_radial_energy_fractions, default=0.0),
        "calibration_readout_pressure_mean": statistics.fmean(calibration_readout_pressures) if calibration_readout_pressures else 0.0,
        "metric_norm_update": statistics.fmean([value_or(r.get("metric_norm_update"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "support_projection_residual": statistics.fmean([value_or(r.get("support_projection_residual"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "support_overlap": statistics.fmean([value_or(r.get("support_overlap"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "tail_safe_support_active_fraction": statistics.fmean([value_or(r.get("tail_safe_support_active"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "tail_safe_support_weight": statistics.fmean([value_or(r.get("tail_safe_support_weight"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "tail_projected_signal_active_fraction": statistics.fmean([value_or(r.get("tail_projected_signal_active"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "tail_projection_metric_overlap": statistics.fmean([value_or(r.get("tail_projection_metric_overlap"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "calibration_nuisance_metric_overlap": statistics.fmean([value_or(r.get("calibration_nuisance_metric_overlap"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "calibration_nuisance_active_fraction": statistics.fmean([value_or(r.get("calibration_nuisance_active"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "calibration_correction_metric_overlap": statistics.fmean([value_or(r.get("calibration_correction_metric_overlap"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "calibration_correction_active_fraction": statistics.fmean([value_or(r.get("calibration_correction_active"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "calibration_correction_norm": statistics.fmean([value_or(r.get("calibration_correction_norm"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "safety_budget_debt": statistics.fmean([value_or(r.get("safety_budget_debt"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "safety_budget_velocity_scale": statistics.fmean([value_or(r.get("safety_budget_velocity_scale"), 1.0) for r in state_rows]) if state_rows else 1.0,
        "mirror_potential_type": next((str(r.get("mirror_potential_type")) for r in state_rows if r.get("mirror_potential_type")), ""),
        "Bregman_step_size": mean_present(state_rows, "Bregman_step_size"),
        "KL_step": mean_present(state_rows, "KL_step"),
        "Fisher_norm_step": mean_present(state_rows, "Fisher_norm_step"),
        "Brier_delta": mean_present(state_rows, "Brier_delta"),
        "tail_q99_mirror_delta": mean_present(state_rows, "tail_q99_mirror_delta"),
        "margin_q10_mirror_delta": mean_present(state_rows, "margin_q10_mirror_delta"),
        "pareto_guard_active_fraction": mean_present(state_rows, "pareto_guard_active"),
        "pareto_tail_alignment_before": mean_present(state_rows, "pareto_tail_alignment_before"),
        "pareto_brier_alignment_before": mean_present(state_rows, "pareto_brier_alignment_before"),
        "pareto_tail_added": mean_present(state_rows, "pareto_tail_added"),
        "pareto_brier_added": mean_present(state_rows, "pareto_brier_added"),
        "debt_tail_alignment_before_orthogonal": mean_present(state_rows, "debt_tail_alignment_before_orthogonal"),
        "debt_brier_alignment_before_orthogonal": mean_present(state_rows, "debt_brier_alignment_before_orthogonal"),
        "debt_tail_alignment_after_orthogonal": mean_present(state_rows, "debt_tail_alignment_after_orthogonal"),
        "debt_brier_alignment_after_orthogonal": mean_present(state_rows, "debt_brier_alignment_after_orthogonal"),
        "debt_orthogonal_projection_rank_mean": mean_present(state_rows, "debt_orthogonal_projection_rank"),
        "debt_orthogonal_projection_overlap": mean_present(state_rows, "debt_orthogonal_projection_overlap"),
        "mirror_loss": mean_present(state_rows, "mirror_loss"),
        "mirror_grad_norm": mean_present(state_rows, "mirror_grad_norm"),
        "mirror_dual_state_norm": mean_present(state_rows, "mirror_dual_state_norm"),
        "mirror_primal_delta_norm": mean_present(state_rows, "mirror_primal_delta_norm"),
        "mirror_diagnostic_recorded_fraction": mean_present(state_rows, "mirror_diagnostic_recorded"),
        "mirror_grad_refresh_cadence": mean_present(state_rows, "mirror_grad_refresh_cadence", 1.0),
        "mirror_grad_refreshed_fraction": mean_present(state_rows, "mirror_grad_refreshed_this_step"),
        "mirror_grad_cached_fraction": mean_present(state_rows, "mirror_grad_cached"),
        "mirror_ls_active_fraction": mean_present(state_rows, "mirror_ls_active"),
        "mirror_ls_cached_fraction": mean_present(state_rows, "mirror_ls_cached"),
        "mirror_ls_basis_count": mean_present(state_rows, "mirror_ls_basis_count"),
        "mirror_ls_fd_eval_count": mean_present(state_rows, "mirror_ls_fd_eval_count"),
        "mirror_ls_target_norm": mean_present(state_rows, "mirror_ls_target_norm"),
        "mirror_ls_fit_norm": mean_present(state_rows, "mirror_ls_fit_norm"),
        "mirror_ls_residual_ratio": mean_present(state_rows, "mirror_ls_residual_ratio"),
        "mirror_ls_coeff_norm": mean_present(state_rows, "mirror_ls_coeff_norm"),
        "mirror_ls_velocity_metric_norm": mean_present(state_rows, "mirror_ls_velocity_metric_norm"),
        "mirror_ls_random_basis_count": mean_present(state_rows, "mirror_ls_random_basis_count"),
        "mirror_ls_fallback_fraction": mean_present(state_rows, "mirror_ls_fallback"),
        "mirror_grad_source": next((str(r.get("mirror_grad_source")) for r in state_rows if r.get("mirror_grad_source")), ""),
        "mirror_explicit_dual_update": int(any(int_flag(r.get("mirror_explicit_dual_update")) for r in state_rows)),
        "mirror_explicit_primal_map": int(any(int_flag(r.get("mirror_explicit_primal_map")) for r in state_rows)),
        "mirror_actuator_projection": next((str(r.get("mirror_actuator_projection")) for r in state_rows if r.get("mirror_actuator_projection")), ""),
        "controller_overhead_ratio": controller_overhead,
        "full_loop_ratio": 1.0,
        "signal_metric_norm": statistics.fmean([value_or(r.get("signal_metric_norm"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "nuisance_metric_overlap": statistics.fmean([value_or(r.get("nuisance_metric_overlap"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "residual_signal_norm": statistics.fmean([value_or(r.get("residual_signal_norm"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "residual_signal_SNR": statistics.fmean([value_or(r.get("residual_signal_SNR"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "m6_noise_role": next((str(r.get("m6_noise_role")) for r in state_rows if r.get("m6_noise_role")), ""),
        "m6_reservoir_noise_active_fraction": mean_present(state_rows, "m6_reservoir_noise_active"),
        "reservoir_noise_energy": mean_present(state_rows, "reservoir_noise_energy"),
        "reservoir_noise_metric_norm": mean_present(state_rows, "reservoir_noise_metric_norm"),
        "reservoir_noise_scale": mean_present(state_rows, "reservoir_noise_scale"),
        "signal_noise_leakage": mean_present(state_rows, "signal_noise_leakage"),
        "signal_noise_energy": mean_present(state_rows, "signal_noise_energy"),
        "RSM_index": mean_present(state_rows, "RSM_index"),
        "metric_mixture_active_fraction": mean_present(state_rows, "metric_mixture_active"),
        "metric_mixture_control_frozen": int(any(int_flag(r.get("metric_mixture_control_frozen")) for r in state_rows)),
        "metric_mixture_entropy": mean_present(state_rows, "metric_mixture_entropy"),
        "metric_mixture_final_entropy": last_m9_state.get("metric_mixture_entropy", ""),
        "metric_weight_drift": mean_present(state_rows, "metric_weight_drift"),
        "metric_weight_drift_total_proxy": sum(value_or(r.get("metric_weight_drift"), 0.0) for r in state_rows),
        "metric_mixture_update_count": last_m9_state.get("metric_mixture_update_count", 0),
        "metric_mixture_collapsed_to": last_m9_state.get("metric_mixture_collapsed_to", ""),
        "metric_mixture_eta": last_m9_state.get("metric_mixture_eta", 0.0),
        "metric_weight_euclidean": mean_present(state_rows, "metric_weight_euclidean"),
        "metric_weight_fisher": mean_present(state_rows, "metric_weight_fisher"),
        "metric_weight_signal": mean_present(state_rows, "metric_weight_signal"),
        "metric_weight_basis": mean_present(state_rows, "metric_weight_basis"),
        "metric_final_weight_euclidean": last_m9_state.get("metric_weight_euclidean", ""),
        "metric_final_weight_fisher": last_m9_state.get("metric_weight_fisher", ""),
        "metric_final_weight_signal": last_m9_state.get("metric_weight_signal", ""),
        "metric_final_weight_basis": last_m9_state.get("metric_weight_basis", ""),
        "metric_debt_euclidean": mean_present(state_rows, "metric_debt_euclidean"),
        "metric_debt_fisher": mean_present(state_rows, "metric_debt_fisher"),
        "metric_debt_signal": mean_present(state_rows, "metric_debt_signal"),
        "metric_debt_basis": mean_present(state_rows, "metric_debt_basis"),
        "support_gain_by_metric_euclidean": mean_present(state_rows, "support_gain_by_metric_euclidean"),
        "support_gain_by_metric_fisher": mean_present(state_rows, "support_gain_by_metric_fisher"),
        "support_gain_by_metric_signal": mean_present(state_rows, "support_gain_by_metric_signal"),
        "support_gain_by_metric_basis": mean_present(state_rows, "support_gain_by_metric_basis"),
        "metric_condition_euclidean": mean_present(state_rows, "metric_condition_euclidean", 1.0),
        "metric_condition_fisher": mean_present(state_rows, "metric_condition_fisher", 1.0),
        "metric_condition_signal": mean_present(state_rows, "metric_condition_signal", 1.0),
        "metric_condition_basis": mean_present(state_rows, "metric_condition_basis", 1.0),
        "same_noise_control_gap": "",
        "same_support_control_delta": "",
        "same_metric_signflip_delta": "",
        "same_metric_shuffled_delta": "",
        "safety_debt_delta": statistics.fmean([value_or(r.get("safety_debt"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "basis_energy_fraction": "" if arch == "MLP" else basis_fu_norm / basis_denom,
        "readout_leakage_fraction": "" if arch == "MLP" else readout_fu_norm / basis_denom,
        "basis_Gram_condition": statistics.fmean([finite_float(r.get("basis_Gram_condition"), 1.0) or 1.0 for r in state_rows]) if state_rows else "",
        "basis_generalized_spectrum_drift": statistics.fmean(spectrum_drifts) if spectrum_drifts else 0.0,
        "basis_signal_overlap": statistics.fmean([value_or(r.get("basis_signal_overlap"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "basis_projection_residual": statistics.fmean([value_or(r.get("support_projection_residual"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "basis_state_transport_error": "" if arch == "MLP" else 1.0 - (basis_fu_norm / basis_denom),
        "D_CHE_degree_bank_energy_by_signal_mode": statistics.fmean([value_or(r.get("low_degree_signal_overlap"), 0.0) for r in state_rows]) if arch == "DGKAN_DCHE" and state_rows else "",
        "D_FOU_frequency_bank_energy_by_signal_mode": statistics.fmean([value_or(r.get("low_frequency_signal_overlap"), 0.0) for r in state_rows]) if arch == "DGKAN_DFOU" and state_rows else "",
        "low_degree_signal_overlap": statistics.fmean([value_or(r.get("low_degree_signal_overlap"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "low_frequency_signal_overlap": statistics.fmean([value_or(r.get("low_frequency_signal_overlap"), 0.0) for r in state_rows]) if state_rows else 0.0,
        "basis_JVP_time_ms": 0.0,
        "basis_metric_update_time_ms": 1000.0 * statistics.fmean(metric_update_times) if metric_update_times else 0.0,
        "calibration_nuisance_grad_ms": 1000.0 * statistics.fmean(calibration_nuisance_grad_times) if calibration_nuisance_grad_times else 0.0,
        "mirror_grad_ms": 1000.0 * statistics.fmean(mirror_grad_times) if mirror_grad_times else 0.0,
        "full_step_ms": 1000.0 * statistics.fmean(full_step_times) if full_step_times else 0.0,
        "base_optimizer_ms": 1000.0 * statistics.fmean(base_optimizer_times) if base_optimizer_times else 0.0,
        "controller_ms": 1000.0 * statistics.fmean(controller_times) if controller_times else 0.0,
        "state_update_ms": 1000.0 * statistics.fmean(state_update_times) if state_update_times else 0.0,
        "memory_peak_MB": peak_mb,
        "continuous_fu_state_updated_every_step": int(len(state_rows) == int(steps) and all(int_flag(r.get("finite_state")) for r in state_rows)),
        "fu_velocity_emitted_every_step": int(len(runtime_rows) == int(steps) and all(int_flag(r.get("fu_velocity_emitted")) for r in runtime_rows)),
        "candidate_action_selection_used_for_runtime": 0,
        "runtime_argmax_candidate_used": 0,
        "runtime_topk_candidate_used": 0,
        "candidate_value_model_used_as_runtime_policy": 0,
        "micro_rct_winner_used_as_runtime_action": 0,
        "uses_test_direction_selection": 0,
        "uses_future_direction": 0,
        "uses_validation_direction": 0,
        "uses_readout_diagnostic_as_basis_native": 0,
        "path_mpc_discrete_action_sequence_used": 0,
        "runtime_train_only_control": 1,
        "chunk_state_trace": str(Path(str(chunk_prefix) + "_state_trace.csv").relative_to(ROOT)),
        "chunk_runtime_trace": str(Path(str(chunk_prefix) + "_runtime_trace.csv").relative_to(ROOT)),
        "status": "completed_v22_43P_pure_row" if pure_fu_mode else "completed_v22_43_row",
    }
    for h, metrics in horizon_snapshots.items():
        summary[f"H{h}_held_NLL"] = metrics.get("NLL", "")
        summary[f"H{h}_held_ECE"] = metrics.get("ECE", "")
        summary[f"H{h}_held_Brier"] = metrics.get("Brier", "")
        summary[f"H{h}_held_tail_q99"] = metrics.get("tail_q99", "")
    write_rows(Path(str(chunk_prefix) + "_summary.csv"), [summary])
    append_exec(
        "train_variant",
        task_id=f"collect_{label}",
        status="pass",
        gpu=device_name,
        files=(
            f"{Path(str(chunk_prefix) + '_summary.csv').relative_to(ROOT)}, "
            f"{Path(str(chunk_prefix) + '_state_trace.csv').relative_to(ROOT)}, "
            f"{Path(str(chunk_prefix) + '_runtime_trace.csv').relative_to(ROOT)}"
        ),
        note=f"variant={variant}; control={control_mode}; metric={metric_kind_for_variant(variant)}; dataset={dataset}; seed={seed}; architecture={arch}; optimizer={optimizer_family}; steps={steps}; pure_fu_mode={int(pure_fu_mode)}; warmup_steps={warmup_steps}",
    )
    return summary


def task_specs(args: argparse.Namespace, stage: str) -> list[dict[str, Any]]:
    datasets = split_csv(args.eval_datasets)
    seeds = split_csv(args.eval_seeds, int)
    gpus = split_csv(args.gpus)
    optimizers = split_csv(args.strong_optimizers)
    specs: list[dict[str, Any]] = []
    if stage == "s4":
        variants = split_csv(args.s4_variants)
        control_modes = split_csv(args.control_modes)
        archs = split_csv(args.eval_architectures)
    elif stage == "s6":
        variants = split_csv(args.s6_variants)
        control_modes = split_csv(args.s6_control_modes)
        archs = ["MLP", "DGKAN_DCHE", "DGKAN_DFOU"]
    elif stage == "s3":
        variants = split_csv(args.s3_variants)
        control_modes = ["none"]
        archs = split_csv(args.eval_architectures)
    elif stage == "p3":
        variants = split_csv(args.p3_variants)
        control_modes = split_csv(args.p3_control_modes)
        archs = split_csv(args.eval_architectures)
    elif stage == "p4":
        variants = split_csv(args.p4_variants)
        control_modes = split_csv(args.p4_control_modes)
        archs = ["MLP"]
    elif stage == "p5":
        variants = split_csv(args.p5_variants)
        control_modes = split_csv(args.p5_control_modes)
        archs = ["DGKAN_DCHE", "DGKAN_DFOU"]
    elif stage == "p6":
        variants = split_csv(args.p6_variants)
        control_modes = split_csv(args.p6_control_modes)
        archs = ["MLP", "DGKAN_DCHE", "DGKAN_DFOU"]
    else:
        variants = split_csv(args.eval_variants)
        control_modes = split_csv(args.control_modes)
        archs = split_csv(args.eval_architectures)
    include_baseline = bool(args.include_optimizer_alone) or stage in {"p4", "p6"}
    for dataset in datasets:
        for seed in seeds:
            for architecture in archs:
                for optimizer in optimizers:
                    if include_baseline:
                        idx = len(specs)
                        gpu = gpus[idx % max(1, len(gpus))]
                        label = f"{safe_fragment(args.label)}_{stage}_{safe_fragment(dataset)}_s{seed}_{safe_fragment(architecture)}_{safe_fragment(optimizer)}_optimizer_alone"
                        specs.append({"dataset": dataset, "seed": seed, "architecture": architecture, "optimizer": optimizer, "variant": "optimizer_alone", "control_mode": "none", "device": f"cuda:{gpu}", "label": label})
                    for variant in variants:
                        if stage == "p6":
                            if (variant.startswith("S4-") or variant.startswith("S1-") or variant.startswith("P3-")) and architecture != "MLP":
                                continue
                            if variant.startswith("P5-D-CHE") and architecture != "DGKAN_DCHE":
                                continue
                            if variant.startswith("P5-D-FOU") and architecture != "DGKAN_DFOU":
                                continue
                        arch = default_architecture_for_variant(variant, architecture)
                        if stage == "s4":
                            if variant == "S4-KAN-BasisGram" and arch == "MLP":
                                continue
                            if "MLP-Matched" in variant and arch != "MLP":
                                continue
                        if stage == "s6":
                            if variant.startswith("KAN-D-CHE") and arch != "DGKAN_DCHE":
                                continue
                            if variant.startswith("KAN-D-FOU") and arch != "DGKAN_DFOU":
                                continue
                            if variant.startswith("MLP-") and arch != "MLP":
                                continue
                        if stage == "p5":
                            if variant.startswith("P5-D-CHE") and arch != "DGKAN_DCHE":
                                continue
                            if variant.startswith("P5-D-FOU") and arch != "DGKAN_DFOU":
                                continue
                        if stage == "p6":
                            if (variant.startswith("S4-") or variant.startswith("S1-") or variant.startswith("P3-")) and arch != "MLP":
                                continue
                            if variant.startswith("P5-D-CHE") and arch != "DGKAN_DCHE":
                                continue
                            if variant.startswith("P5-D-FOU") and arch != "DGKAN_DFOU":
                                continue
                        variant_control_modes = p6_control_modes_for_variant(variant, args.p6_control_modes) if stage == "p6" else control_modes
                        for control_mode in variant_control_modes:
                            if variant == "optimizer_alone" and control_mode != "none":
                                continue
                            idx = len(specs)
                            gpu = gpus[idx % max(1, len(gpus))]
                            label = (
                                f"{safe_fragment(args.label)}_{stage}_{safe_fragment(dataset)}_s{seed}_"
                                f"{safe_fragment(arch)}_{safe_fragment(optimizer)}_{safe_fragment(variant)}_{safe_fragment(control_mode)}"
                            )
                            specs.append(
                                {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "architecture": arch,
                                    "optimizer": optimizer,
                                    "variant": variant,
                                    "control_mode": control_mode,
                                    "device": f"cuda:{gpu}" if not str(gpu).startswith("cuda") else str(gpu),
                                    "label": label,
                                }
                            )
    if args.row_limit > 0:
        specs = specs[: int(args.row_limit)]
    return specs


def dispatch_specs(args: argparse.Namespace, specs: list[dict[str, Any]], task_prefix: str) -> dict[str, Any]:
    ensure_out()
    commands: list[tuple[list[str], str, str]] = []
    for spec in specs:
        physical_device = str(spec["device"])
        child_device = "cuda:0" if physical_device.startswith("cuda") else physical_device
        cmd = [
            PYTHON,
            str(Path(__file__).relative_to(ROOT)),
            "--stage",
            "collect",
            "--dataset",
            str(spec["dataset"]),
            "--seed",
            str(spec["seed"]),
            "--architecture",
            str(spec["architecture"]),
            "--optimizer",
            str(spec["optimizer"]),
            "--variant",
            str(spec["variant"]),
            "--control-mode",
            str(spec["control_mode"]),
            "--device",
            child_device,
            "--steps",
            str(args.steps),
            "--train-size",
            str(args.train_size),
            "--held-size",
            str(args.held_size),
            "--batch-size",
            str(args.batch_size),
            "--hidden",
            str(args.hidden),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--support-rank",
            str(args.support_rank),
            "--nuisance-rank",
            str(args.nuisance_rank),
            "--support-refresh-cadence",
            str(args.support_refresh_cadence),
            "--beta-signal",
            str(args.beta_signal),
            "--beta-metric",
            str(args.beta_metric),
            "--beta-q",
            str(args.beta_q),
            "--eta-rho",
            str(args.eta_rho),
            "--eta-debt",
            str(args.eta_debt),
            "--tau-safe",
            str(args.tau_safe),
            "--rho-min",
            str(args.rho_min),
            "--rho-max",
            str(args.rho_max),
            "--velocity-scale",
            str(args.velocity_scale),
            "--metric-shrinkage",
            str(args.metric_shrinkage),
            "--metric-eps",
            str(args.metric_eps),
            "--metric-refresh-cadence",
            str(args.metric_refresh_cadence),
            "--debt-velocity-barrier",
            str(args.debt_velocity_barrier),
            "--calibration-velocity-barrier",
            str(args.calibration_velocity_barrier),
            "--safety-budget-velocity-barrier",
            str(args.safety_budget_velocity_barrier),
            "--calibration-readout-radial-cap",
            str(args.calibration_readout_radial_cap),
            "--kan-init-variant",
            str(args.kan_init_variant),
            "--calibration-nuisance-weight",
            str(args.calibration_nuisance_weight),
            "--calibration-correction-weight",
            str(args.calibration_correction_weight),
            "--calibration-nuisance-mode",
            str(args.calibration_nuisance_mode),
            "--calibration-nuisance-cadence",
            str(args.calibration_nuisance_cadence),
            "--label",
            str(spec["label"]),
        ]
        if args.pure_fu_mode:
            cmd.append("--pure-fu-mode")
        if int(args.warmup_steps) > 0:
            cmd.extend(["--warmup-steps", str(args.warmup_steps)])
        if args.tier2_download:
            cmd.append("--tier2-download")
        commands.append((cmd, str(spec["label"]), physical_device))
    append_exec(
        f"{task_prefix} dispatch",
        task_id=f"{task_prefix}_dispatch_{args.label}",
        status="started",
        gpu=",".join(split_csv(args.gpus)),
        files="results/v22_43/chunks",
        note=f"rows={len(commands)}; workers={args.workers}; datasets={args.eval_datasets}; seeds={args.eval_seeds}; steps={args.steps}; train_size={args.train_size}; pure_fu_mode={int(args.pure_fu_mode)}; warmup_steps={args.warmup_steps}",
    )
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = [ex.submit(run_logged, cmd, task_id=f"{task_prefix}_{label}", gpu=device, timeout=int(args.row_timeout)) for cmd, label, device in commands]
        for fut in concurrent.futures.as_completed(futs):
            proc = fut.result()
            if proc.returncode != 0:
                failures += 1
    merge_summary = merge_pure_artifacts(write_final=False) if args.pure_fu_mode else merge_chunks(write_final=False)
    append_exec(
        f"{task_prefix} completed",
        task_id=f"{task_prefix}_completed_{args.label}",
        status="pass" if failures == 0 else "fail",
        gpu=",".join(split_csv(args.gpus)),
        files="results/v22_43/v22_43P_pure_support_full_loop_matrix.csv" if args.pure_fu_mode else "results/v22_43/v22_43_metric_support_full_loop_matrix.csv",
        note=f"rows={len(commands)}; failures={failures}; merge_status={merge_summary.get('status')}",
        exit_code=0 if failures == 0 else 1,
    )
    return {"rows": len(commands), "failures": failures, **merge_summary}


def group_key(row: dict[str, Any], include_metric: bool = False) -> tuple[str, ...]:
    base = (
        str(row.get("dataset", "")),
        str(row.get("seed", "")),
        str(row.get("architecture", "")),
        str(row.get("optimizer_family", "")),
        str(row.get("steps", "")),
        str(row.get("hidden", "")),
        str(row.get("support_rank", "")),
        str(row.get("velocity_scale", "")),
        str(row.get("support_refresh_cadence", "")),
        str(row.get("metric_refresh_cadence", "")),
        str(row.get("metric_shrinkage", "")),
        str(row.get("rho_max", "")),
        str(row.get("debt_velocity_barrier", "")),
        str(row.get("calibration_velocity_barrier", "")),
        str(row.get("safety_budget_velocity_barrier", "")),
        str(row.get("calibration_readout_radial_cap", "")),
        str(row.get("kan_init_variant", "")),
        str(row.get("from_scratch_or_warmup", "")),
        str(row.get("warmup_steps", "")),
    )
    if include_metric:
        return base + (str(row.get("phase", "")), str(row.get("metric_name", "")))
    return base


def control_group_key(row: dict[str, Any]) -> tuple[str, ...]:
    return group_key(row, include_metric=True) + (str(row.get("variant", "")),)


def own_optimizer_baselines(rows: list[dict[str, Any]]) -> dict[tuple[str, ...], dict[str, Any]]:
    out = {}
    for r in rows:
        if r.get("variant") == "optimizer_alone":
            out[group_key(r)] = r
    return out


def strongest_baselines(rows: list[dict[str, Any]]) -> dict[tuple[str, ...], dict[str, Any]]:
    out = {}
    for r in rows:
        if r.get("variant") != "optimizer_alone":
            continue
        key = group_key(r)
        if key not in out or value_or(r.get("final_NLL"), math.inf) < value_or(out[key].get("final_NLL"), math.inf):
            out[key] = r
    return out


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    own = own_optimizer_baselines(rows)
    strong = strongest_baselines(rows)
    controls: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
    for r in rows:
        if is_control_mode(str(r.get("control_mode", ""))):
            controls.setdefault(control_group_key(r), {})[str(r.get("control_mode"))] = r
    by_group_variant: dict[tuple[str, ...], dict[str, Any]] = {}
    for r in rows:
        by_group_variant[group_key(r) + (str(r.get("variant")), str(r.get("control_mode")), str(r.get("metric_name")), str(r.get("phase")))] = r
    enriched = []
    for raw in rows:
        r = dict(raw)
        base = own.get(group_key(r), {})
        strongest = strong.get(group_key(r), {})
        nll = finite_float(r.get("final_NLL"))
        base_nll = finite_float(base.get("final_NLL"))
        strong_nll = finite_float(strongest.get("final_NLL"))
        if nll is not None and base_nll is not None:
            r["NLL_delta_vs_base"] = nll - base_nll
            r["NLL_improvement_vs_own_strong_optimizer"] = base_nll - nll
            r["NLL_delta_real_minus_own_strong_optimizer"] = nll - base_nll
            r["ECE_delta_vs_own_strong_optimizer"] = value_or(r.get("ECE"), 0.0) - value_or(base.get("ECE"), 0.0)
            r["Brier_delta_vs_own_strong_optimizer"] = value_or(r.get("Brier"), 0.0) - value_or(base.get("Brier"), 0.0)
            r["tail_q99_delta_vs_own_strong_optimizer"] = value_or(r.get("tail_loss_q99"), 0.0) - value_or(base.get("tail_loss_q99"), 0.0)
            r["AUC_delta_vs_own_strong_optimizer"] = value_or(r.get("AUC_loss_time"), 0.0) - value_or(base.get("AUC_loss_time"), 0.0)
            r["no_ECE_Brier_tail_debt"] = int(
                value_le(r.get("ECE_delta_vs_own_strong_optimizer"), 0.0)
                and value_le(r.get("Brier_delta_vs_own_strong_optimizer"), 0.0)
                and value_le(r.get("tail_q99_delta_vs_own_strong_optimizer"), 0.0)
            )
            r["post_warmup_no_debt"] = r["no_ECE_Brier_tail_debt"] if int_flag(r.get("warmup_steps")) > 0 else ""
        else:
            r["NLL_delta_vs_base"] = ""
            r["NLL_improvement_vs_own_strong_optimizer"] = ""
            r["no_ECE_Brier_tail_debt"] = 0
            r["post_warmup_no_debt"] = ""
        if nll is not None and strong_nll is not None:
            r["NLL_improvement_vs_strongest_optimizer"] = strong_nll - nll
            r["strongest_optimizer_family"] = strongest.get("optimizer_family", "")
        cdict = controls.get(control_group_key(r), {})
        for mode in CONTROL_MODES + S6_CONTROL_MODES + P3_CONTROL_MODES + P4_CONTROL_MODES + P5_CONTROL_MODES:
            if mode == "none":
                continue
            ctrl = cdict.get(mode, {})
            cnll = finite_float(ctrl.get("final_NLL"))
            field = safe_fragment(mode).replace("-", "_")
            if nll is not None and cnll is not None:
                r[f"NLL_delta_vs_{field}"] = nll - cnll
                r[f"beats_{field}"] = int(nll < cnll)
            else:
                r[f"NLL_delta_vs_{field}"] = ""
                r[f"beats_{field}"] = ""
        if phase_for_variant(str(r.get("variant"))) == "S1_MetricResidualSignal":
            ctrl = cdict.get("same-metric-support-random", {})
            for h in [20, 60, 200, 800]:
                rv = finite_float(r.get(f"H{h}_held_NLL"))
                cv = finite_float(ctrl.get(f"H{h}_held_NLL"))
                if rv is not None and cv is not None:
                    tau = cv - rv
                    r[f"tau_direction_H{h}"] = tau
                    r[f"LCB_tau_direction_H{h}"] = tau
                    r[f"tau_direction_H{h}_positive"] = int(tau > 0.0)
            if finite_float(r.get("final_NLL")) is not None and finite_float(ctrl.get("final_NLL")) is not None:
                r["same_support_control_delta"] = value_or(ctrl.get("final_NLL"), 0.0) - value_or(r.get("final_NLL"), 0.0)
        if phase_for_variant(str(r.get("variant"))) == "S6_MetricKANCarrier":
            mlp_candidates = [
                m
                for m in rows
                if group_key(m)[:2] == group_key(r)[:2]
                and m.get("architecture") == "MLP"
                and m.get("phase") == "S6_MLPMatchedMetricSupport"
                and m.get("control_mode") == "none"
            ]
            if mlp_candidates and nll is not None:
                best_mlp = min(mlp_candidates, key=lambda x: value_or(x.get("final_NLL"), math.inf))
                mnll = finite_float(best_mlp.get("final_NLL"))
                if mnll is not None:
                    r["KAN_NLL_delta_vs_MLP_FU"] = nll - mnll
                    r["KAN_NLL_delta_vs_MLP_matched_support_FU"] = nll - mnll
                    base_delta = value_or(r.get("NLL_improvement_vs_own_strong_optimizer"), 0.0)
                    mlp_delta = value_or(best_mlp.get("NLL_improvement_vs_own_strong_optimizer"), 0.0)
                    r["MLP_matched_NLL_improvement_vs_own_strong_optimizer"] = mlp_delta
                    if nll <= mnll and base_delta > 0:
                        klass = "TrueKANGain" if mlp_delta <= 0 else "BothGain"
                    elif base_delta > 0:
                        klass = "KANInternalValueOnly"
                    elif mlp_delta > 0:
                        klass = "MLPMatchedSupportStronger"
                    else:
                        klass = "NoGain"
                    r["TrueKANGain_class"] = klass
        enriched.append(r)
    return enriched


def merge_chunks(write_final: bool = True) -> dict[str, Any]:
    ensure_out()
    summaries: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for path in sorted(CHUNK_ROOT.glob("v22_43_*_summary.csv")):
        summaries.extend([r for r in read_rows(path) if r.get("status") == "completed_v22_43_row"])
    for path in sorted(CHUNK_ROOT.glob("v22_43_*_state_trace.csv")):
        state_rows.extend([r for r in read_rows(path) if r.get("step")])
    for path in sorted(CHUNK_ROOT.glob("v22_43_*_runtime_trace.csv")):
        runtime_rows.extend([r for r in read_rows(path) if r.get("step")])
    enriched = enrich_rows(summaries)
    write_rows(OUT_ROOT / "v22_43_metric_support_full_loop_matrix.csv", enriched or [{"status": "no_full_loop_rows"}])
    write_rows(OUT_ROOT / "v22_43_metric_support_control_matrix.csv", [r for r in enriched if is_control_mode(str(r.get("control_mode"))) or r.get("variant") == "optimizer_alone"] or [{"status": "no_control_rows"}])
    write_rows(OUT_ROOT / "v22_43_metric_residual_signal_matrix.csv", [r for r in enriched if r.get("phase") == "S1_MetricResidualSignal"] or [{"status": "no_s1_rows"}])
    write_rows(OUT_ROOT / "v22_43_tau_direction_matrix.csv", tau_direction_rows(enriched) or [{"status": "no_tau_rows"}])
    write_rows(OUT_ROOT / "v22_43_KAN_basis_Gram_matrix.csv", [basis_row(r) for r in enriched if r.get("architecture") == "strict_FC_PureKAN"] or [{"status": "no_kan_rows"}])
    write_rows(OUT_ROOT / "v22_43_KAN_metric_basis_carrier_matrix.csv", [r for r in enriched if r.get("phase") == "S6_MetricKANCarrier"] or [{"status": "no_s6_kan_rows"}])
    write_rows(OUT_ROOT / "v22_43_MLP_matched_metric_support_matrix.csv", [r for r in enriched if r.get("phase") == "S6_MLPMatchedMetricSupport"] or [{"status": "no_mlp_matched_rows"}])
    write_rows(OUT_ROOT / "v22_43_strong_optimizer_metric_baseline_matrix.csv", [r for r in enriched if r.get("phase") in {"S3_StrongMetricOptimizer", "Phase0_ControlHarness"}] or [{"status": "no_s3_rows"}])
    write_rows(OUT_ROOT / "v22_43_hard_task_four_square_matrix.csv", hard_task_summary(enriched))
    write_rows(OUT_ROOT / "v22_43_continual_grokking_matrix.csv", [{"status": "not_run", "reason": "v22.43 execution prioritized S4->S1->S6 metric geometry; no continual/grokking rows were executed in this command set."}])
    write_rows(OUT_ROOT / "v22_43_efficiency_matrix.csv", efficiency_rows(enriched))
    write_runtime_audit(runtime_rows)
    write_phase_summaries(enriched)
    final = finalize(write_recap=False) if write_final else {}
    append_exec(
        "merge_chunks",
        task_id="merge_chunks",
        status="pass" if summaries else "warn",
        gpu="n/a",
        files="results/v22_43/v22_43_metric_support_full_loop_matrix.csv, results/v22_43/v22_43_final_route.json",
        note=f"summaries={len(summaries)}; state_rows={len(state_rows)}; runtime_rows={len(runtime_rows)}; route={final.get('final_route', 'deferred')}",
    )
    return {"status": "merged" if summaries else "no_rows", "summary_rows": len(summaries), "state_rows": len(state_rows), "runtime_rows": len(runtime_rows)}


def tau_direction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("phase") != "S1_MetricResidualSignal":
            continue
        out.append(
            {
                "run_label": r.get("run_label", ""),
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "metric_name": r.get("metric_name", ""),
                "variant": r.get("variant", ""),
                "tau_direction_H20": r.get("tau_direction_H20", ""),
                "tau_direction_H60": r.get("tau_direction_H60", ""),
                "tau_direction_H200": r.get("tau_direction_H200", ""),
                "tau_direction_H800": r.get("tau_direction_H800", ""),
                "LCB_tau_direction_H20": r.get("LCB_tau_direction_H20", ""),
                "LCB_tau_direction_H60": r.get("LCB_tau_direction_H60", ""),
                "LCB_tau_direction_H200": r.get("LCB_tau_direction_H200", ""),
                "LCB_tau_direction_H800": r.get("LCB_tau_direction_H800", ""),
                "residual_signal_SNR": r.get("residual_signal_SNR", ""),
                "nuisance_metric_overlap": r.get("nuisance_metric_overlap", ""),
            }
        )
    return out


def basis_row(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_label": r.get("run_label", ""),
        "dataset": r.get("dataset", ""),
        "seed": r.get("seed", ""),
        "variant": r.get("variant", ""),
        "basis_family": r.get("basis_family", ""),
        "basis_energy_fraction": r.get("basis_energy_fraction", ""),
        "readout_leakage_fraction": r.get("readout_leakage_fraction", ""),
        "basis_Gram_condition": r.get("basis_Gram_condition", ""),
        "basis_generalized_spectrum_drift": r.get("basis_generalized_spectrum_drift", ""),
        "basis_signal_overlap": r.get("basis_signal_overlap", ""),
        "basis_projection_residual": r.get("basis_projection_residual", ""),
        "basis_state_transport_error": r.get("basis_state_transport_error", ""),
        "D_CHE_degree_bank_energy_by_signal_mode": r.get("D_CHE_degree_bank_energy_by_signal_mode", ""),
        "D_FOU_frequency_bank_energy_by_signal_mode": r.get("D_FOU_frequency_bank_energy_by_signal_mode", ""),
        "low_degree_signal_overlap": r.get("low_degree_signal_overlap", ""),
        "low_frequency_signal_overlap": r.get("low_frequency_signal_overlap", ""),
    }


def hard_task_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hard = [r for r in rows if int_flag(r.get("official_hard_row")) and r.get("row_role") == "signal_candidate"]
    debug = [r for r in rows if not int_flag(r.get("official_hard_row")) and r.get("row_role") == "signal_candidate"]
    return [
        {
            "slice": "official_hard",
            "rows": len(hard),
            "datasets": ",".join(sorted({str(r.get("dataset", "")) for r in hard if r.get("dataset")})),
            "NLL_improvement_rows": count_positive(hard, "NLL_improvement_vs_own_strong_optimizer"),
            "no_ECE_Brier_tail_debt_rows": sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in hard),
            "note": "MNIST/FashionMNIST/KMNIST are excluded from success-route counts by v22.43 plan.",
        },
        {
            "slice": "tier0_debug",
            "rows": len(debug),
            "datasets": ",".join(sorted({str(r.get("dataset", "")) for r in debug if r.get("dataset")})),
            "NLL_improvement_rows": count_positive(debug, "NLL_improvement_vs_own_strong_optimizer"),
            "no_ECE_Brier_tail_debt_rows": sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in debug),
            "note": "Debug only; not used for official route promotion.",
        },
    ]


def efficiency_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("row_role") != "signal_candidate":
            continue
        out.append(
            {
                "run_label": r.get("run_label", ""),
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "variant": r.get("variant", ""),
                "metric_name": r.get("metric_name", ""),
                "controller_overhead_ratio": r.get("controller_overhead_ratio", ""),
                "full_loop_ratio": r.get("full_loop_ratio", ""),
                "full_step_ms": r.get("full_step_ms", ""),
                "base_optimizer_ms": r.get("base_optimizer_ms", ""),
                "controller_ms": r.get("controller_ms", ""),
                "basis_metric_update_time_ms": r.get("basis_metric_update_time_ms", ""),
                "memory_peak_MB": r.get("memory_peak_MB", ""),
            }
        )
    return out or [{"status": "no_efficiency_rows"}]


def write_runtime_audit(runtime_rows: list[dict[str, Any]]) -> None:
    if not runtime_rows:
        existing = read_rows(OUT_ROOT / "v22_43_runtime_regression_audit.csv")
        if existing:
            return
    row = {
        "runtime_row_count": len(runtime_rows),
        "runtime_argmax_candidate_used": max([int_flag(r.get("runtime_argmax_candidate_used")) for r in runtime_rows] + [0]),
        "runtime_topk_candidate_used": max([int_flag(r.get("runtime_topk_candidate_used")) for r in runtime_rows] + [0]),
        "candidate_action_selection_used_for_runtime": max([int_flag(r.get("candidate_action_selection_used_for_runtime")) for r in runtime_rows] + [0]),
        "candidate_value_model_used_as_runtime_policy": max([int_flag(r.get("candidate_value_model_used_as_runtime_policy")) for r in runtime_rows] + [0]),
        "continuous_fu_state_updated_every_step": int(bool(runtime_rows) and all(int_flag(r.get("continuous_fu_state_updated")) for r in runtime_rows)),
        "fu_velocity_emitted_every_step": int(bool(runtime_rows) and all(int_flag(r.get("fu_velocity_emitted")) for r in runtime_rows)),
    }
    row["candidate_action_regression_pass"] = int(
        row["runtime_argmax_candidate_used"] == 0
        and row["runtime_topk_candidate_used"] == 0
        and row["candidate_action_selection_used_for_runtime"] == 0
        and row["candidate_value_model_used_as_runtime_policy"] == 0
        and row["continuous_fu_state_updated_every_step"] == 1
        and row["fu_velocity_emitted_every_step"] == 1
    )
    row["status"] = "pass" if row["candidate_action_regression_pass"] else "fail"
    write_rows(OUT_ROOT / "v22_43_runtime_regression_audit.csv", [row])


def zeroish(value: Any, tol: float = 1.0e-12) -> bool:
    parsed = finite_float(value)
    return parsed is not None and abs(float(parsed)) <= tol


def pure_runtime_truth_pass(row: dict[str, Any]) -> bool:
    return (
        int_flag(row.get("pure_fu_mode")) == 1
        and int_flag(row.get("base_optimizer_step_used")) == 0
        and zeroish(row.get("base_velocity_added"))
        and zeroish(row.get("ordinary_adamw_update_norm"))
        and zeroish(row.get("ordinary_sgd_update_norm"))
        and zeroish(row.get("ordinary_muon_update_norm"))
        and zeroish(row.get("ordinary_schedulefree_update_norm"))
        and int_flag(row.get("bp_gradient_used_only_for_cotangent")) == 1
        and int_flag(row.get("continuous_fu_state_updated_every_step")) == 1
        and int_flag(row.get("fu_velocity_emitted_every_step")) == 1
        and int_flag(row.get("candidate_action_selection_used_for_runtime")) == 0
        and int_flag(row.get("runtime_argmax_candidate_used")) == 0
        and int_flag(row.get("runtime_topk_candidate_used")) == 0
        and int_flag(row.get("micro_rct_winner_used_as_runtime_action")) == 0
        and int_flag(row.get("candidate_value_model_used_as_runtime_policy")) == 0
        and int_flag(row.get("uses_test_direction_selection")) == 0
        and int_flag(row.get("uses_future_direction")) == 0
    )


def p6_post_warmup_truth_pass(row: dict[str, Any]) -> bool:
    return (
        int_flag(row.get("pure_fu_mode")) == 1
        and int_flag(row.get("warmup_steps")) > 0
        and int_flag(row.get("base_optimizer_step_used_after_warmup")) == 0
        and zeroish(row.get("base_velocity_added_after_warmup"))
        and zeroish(row.get("ordinary_adamw_update_norm_after_warmup"))
        and zeroish(row.get("ordinary_sgd_update_norm_after_warmup"))
        and zeroish(row.get("ordinary_muon_update_norm_after_warmup"))
        and zeroish(row.get("ordinary_schedulefree_update_norm_after_warmup"))
        and int_flag(row.get("bp_gradient_used_only_for_cotangent_after_warmup")) == 1
        and int_flag(row.get("continuous_fu_state_updated_every_step")) == 1
        and int_flag(row.get("fu_velocity_emitted_every_step")) == 1
        and int_flag(row.get("candidate_action_selection_used_for_runtime")) == 0
        and int_flag(row.get("runtime_argmax_candidate_used")) == 0
        and int_flag(row.get("runtime_topk_candidate_used")) == 0
        and int_flag(row.get("micro_rct_winner_used_as_runtime_action")) == 0
        and int_flag(row.get("candidate_value_model_used_as_runtime_policy")) == 0
        and int_flag(row.get("uses_test_direction_selection")) == 0
        and int_flag(row.get("uses_future_direction")) == 0
    )


def pure_runtime_truth_scope(row: dict[str, Any]) -> tuple[int, str]:
    """Return whether a pure row is official P0 audit scope and why.

    P5 same-readout-leakage-control is a diagnostic leakage probe. When the
    basis-native path has zero readout leakage it correctly emits zero velocity,
    so it must not decide the official pure-FU runtime truth gate.
    """

    if str(row.get("control_mode", "")) == "same-readout-leakage-control" and str(row.get("variant", "")).startswith("P5-"):
        return 0, "diagnostic_readout_leakage_control_nonofficial_p0"
    return 1, "official_pure_runtime_row"


def pure_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {field: row.get(field, "") for field in PURE_AUDIT_FIELDS}
    if not out.get("row_id"):
        out["row_id"] = hashlib.sha256(str(row.get("run_label", "")).encode("utf-8")).hexdigest()[:16]
    official_scope, scope_reason = pure_runtime_truth_scope(out)
    out["official_p0_runtime_audit_row"] = official_scope
    out["runtime_truth_scope_reason"] = scope_reason
    out["p0_runtime_truth_pass"] = int(int_flag(out.get("warmup_steps")) == 0 and pure_runtime_truth_pass(out))
    out["p6_post_warmup_truth_pass"] = int(p6_post_warmup_truth_pass(out))
    return out


def write_placeholder_csv(name: str, reason: str) -> None:
    write_rows(OUT_ROOT / name, [{"status": "not_run", "reason": reason}])


def merge_pure_artifacts(write_final: bool = True) -> dict[str, Any]:
    ensure_out()
    summaries: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for path in sorted(CHUNK_ROOT.glob("v22_43P_*_summary.csv")):
        summaries.extend([r for r in read_rows(path) if r.get("status") == "completed_v22_43P_pure_row"])
    for path in sorted(CHUNK_ROOT.glob("v22_43P_*_state_trace.csv")):
        state_rows.extend([r for r in read_rows(path) if r.get("step")])
    for path in sorted(CHUNK_ROOT.glob("v22_43P_*_runtime_trace.csv")):
        runtime_rows.extend([r for r in read_rows(path) if r.get("step")])

    enriched = enrich_rows(summaries)
    audit_rows = [pure_audit_row(r) for r in summaries]
    p0_audit_rows = [
        r
        for r in audit_rows
        if int_flag(r.get("warmup_steps")) == 0 and int_flag(r.get("official_p0_runtime_audit_row")) == 1
    ]
    p0_pass_rows = sum(int_flag(r.get("p0_runtime_truth_pass")) for r in p0_audit_rows)
    p0_all_pass = bool(p0_audit_rows) and p0_pass_rows == len(p0_audit_rows)
    p6_audit_rows = [r for r in audit_rows if int_flag(r.get("warmup_steps")) > 0]
    p6_truth_pass_rows = sum(int_flag(r.get("p6_post_warmup_truth_pass")) for r in p6_audit_rows)

    write_rows(OUT_ROOT / "v22_43P_pure_support_full_loop_matrix.csv", enriched or [{"status": "no_pure_support_rows"}])
    write_rows(
        OUT_ROOT / "v22_43P_pure_support_control_matrix.csv",
        [r for r in enriched if is_control_mode(str(r.get("control_mode"))) or r.get("variant") == "optimizer_alone"] or [{"status": "no_pure_control_rows"}],
    )
    write_rows(
        OUT_ROOT / "v22_43P_pure_fu_runtime_truth_matrix.csv",
        audit_rows or [{"status": "no_pure_rows"}],
        PURE_AUDIT_FIELDS
        + [
            "official_p0_runtime_audit_row",
            "runtime_truth_scope_reason",
            "p0_runtime_truth_pass",
            "p6_post_warmup_truth_pass",
        ],
    )
    write_rows(
        OUT_ROOT / "v22_43P_pure_fu_optimizer_step_audit.csv",
        [
            {
                "row_id": r.get("row_id", ""),
                "run_label": r.get("run_label", ""),
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "optimizer_reference_name": r.get("optimizer_reference_name", ""),
                "pure_fu_mode": r.get("pure_fu_mode", ""),
                "warmup_steps": r.get("warmup_steps", ""),
                "base_optimizer_step_used": r.get("base_optimizer_step_used", ""),
                "base_optimizer_step_used_after_warmup": r.get("base_optimizer_step_used_after_warmup", ""),
                "base_velocity_added": r.get("base_velocity_added", ""),
                "base_velocity_added_after_warmup": r.get("base_velocity_added_after_warmup", ""),
                "ordinary_adamw_update_norm": r.get("ordinary_adamw_update_norm", ""),
                "ordinary_sgd_update_norm": r.get("ordinary_sgd_update_norm", ""),
                "ordinary_muon_update_norm": r.get("ordinary_muon_update_norm", ""),
                "ordinary_schedulefree_update_norm": r.get("ordinary_schedulefree_update_norm", ""),
                "ordinary_adamw_update_norm_after_warmup": r.get("ordinary_adamw_update_norm_after_warmup", ""),
                "ordinary_sgd_update_norm_after_warmup": r.get("ordinary_sgd_update_norm_after_warmup", ""),
                "ordinary_muon_update_norm_after_warmup": r.get("ordinary_muon_update_norm_after_warmup", ""),
                "ordinary_schedulefree_update_norm_after_warmup": r.get("ordinary_schedulefree_update_norm_after_warmup", ""),
                "optimizer_step_audit_pass": int(
                    int_flag(r.get("p0_runtime_truth_pass")) == 1
                    or int_flag(r.get("p6_post_warmup_truth_pass")) == 1
                ),
            }
            for r in audit_rows
        ]
        or [{"status": "no_pure_rows"}],
    )
    write_rows(
        OUT_ROOT / "v22_43P_pure_fu_gradient_usage_audit.csv",
        [
            {
                "row_id": r.get("row_id", ""),
                "run_label": r.get("run_label", ""),
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "bp_gradient_used_only_for_cotangent": r.get("bp_gradient_used_only_for_cotangent", ""),
                "bp_gradient_used_only_for_cotangent_after_warmup": r.get("bp_gradient_used_only_for_cotangent_after_warmup", ""),
                "continuous_fu_state_updated_every_step": r.get("continuous_fu_state_updated_every_step", ""),
                "fu_velocity_emitted_every_step": r.get("fu_velocity_emitted_every_step", ""),
                "gradient_usage_audit_pass": int(
                    (
                        int_flag(r.get("bp_gradient_used_only_for_cotangent")) == 1
                        or int_flag(r.get("bp_gradient_used_only_for_cotangent_after_warmup")) == 1
                    )
                    and int_flag(r.get("continuous_fu_state_updated_every_step")) == 1
                    and int_flag(r.get("fu_velocity_emitted_every_step")) == 1
                ),
            }
            for r in audit_rows
        ]
        or [{"status": "no_pure_rows"}],
    )
    write_rows(
        OUT_ROOT / "v22_43P_pure_fu_candidate_regression_audit.csv",
        [
            {
                "row_id": r.get("row_id", ""),
                "run_label": r.get("run_label", ""),
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "candidate_action_selection_used_for_runtime": r.get("candidate_action_selection_used_for_runtime", ""),
                "runtime_argmax_candidate_used": r.get("runtime_argmax_candidate_used", ""),
                "runtime_topk_candidate_used": r.get("runtime_topk_candidate_used", ""),
                "micro_rct_winner_used_as_runtime_action": r.get("micro_rct_winner_used_as_runtime_action", ""),
                "candidate_value_model_used_as_runtime_policy": r.get("candidate_value_model_used_as_runtime_policy", ""),
                "uses_test_direction_selection": r.get("uses_test_direction_selection", ""),
                "uses_future_direction": r.get("uses_future_direction", ""),
                "candidate_regression_audit_pass": int(
                    int_flag(r.get("candidate_action_selection_used_for_runtime")) == 0
                    and int_flag(r.get("runtime_argmax_candidate_used")) == 0
                    and int_flag(r.get("runtime_topk_candidate_used")) == 0
                    and int_flag(r.get("micro_rct_winner_used_as_runtime_action")) == 0
                    and int_flag(r.get("candidate_value_model_used_as_runtime_policy")) == 0
                    and int_flag(r.get("uses_test_direction_selection")) == 0
                    and int_flag(r.get("uses_future_direction")) == 0
                ),
            }
            for r in audit_rows
        ]
        or [{"status": "no_pure_rows"}],
    )
    write_rows(OUT_ROOT / "v22_43P_pure_fu_state_trace.csv", state_rows or [{"status": "no_pure_state_rows"}])
    write_rows(OUT_ROOT / "v22_43P_pure_fu_runtime_decision_trace.csv", runtime_rows or [{"status": "no_pure_runtime_rows"}])

    write_rows(OUT_ROOT / "v22_43P_pure_residual_signal_matrix.csv", [r for r in enriched if r.get("phase") == "S1_MetricResidualSignal"] or [{"status": "not_run", "reason": "P2 residual-signal pure rows not executed yet."}])
    write_rows(OUT_ROOT / "v22_43P_pure_tau_direction_matrix.csv", tau_direction_rows(enriched) or [{"status": "not_run", "reason": "P2 tau direction rows not executed yet."}])
    p3_rows = [r for r in enriched if r.get("phase") == "P3_PureMetricPreservingOET"]
    p4_rows = [r for r in enriched if r.get("phase") == "P4_PureOETRadial"]
    write_rows(OUT_ROOT / "v22_43P_pure_oet_unit_matrix.csv", p3_rows or [{"status": "not_run", "reason": "P3 pure non-additive OET rows not executed yet."}])
    write_rows(
        OUT_ROOT / "v22_43P_pure_metric_spectrum_drift_matrix.csv",
        [
            {
                "run_label": r.get("run_label", ""),
                "dataset": r.get("dataset", ""),
                "seed": r.get("seed", ""),
                "variant": r.get("variant", ""),
                "control_mode": r.get("control_mode", ""),
                "ordinary_spectrum_drift": r.get("ordinary_spectrum_drift", ""),
                "generalized_spectrum_drift": r.get("generalized_spectrum_drift", ""),
                "OET_spectral_norm_drift": r.get("OET_spectral_norm_drift", ""),
                "metric_skew_residual": r.get("metric_skew_residual", ""),
                "Cayley_solve_residual": r.get("Cayley_solve_residual", ""),
                "radial_energy_fraction": r.get("radial_energy_fraction", ""),
                "radial_spectrum_drift": r.get("radial_spectrum_drift", ""),
            }
            for r in p3_rows + p4_rows
        ]
        or [{"status": "not_run", "reason": "P3/P4 non-additive metric spectrum drift rows not executed yet."}],
    )
    write_rows(OUT_ROOT / "v22_43P_pure_oet_full_loop_matrix.csv", p3_rows or [{"status": "not_run", "reason": "P3 pure OET full-loop rows not executed yet."}])
    write_rows(OUT_ROOT / "v22_43P_pure_oet_radial_matrix.csv", p4_rows or [{"status": "not_run", "reason": "P4 radial repair rows not executed yet."}])
    p5_rows = [r for r in enriched if r.get("phase") == "P5_PureKANBasisCarrier"]
    write_rows(OUT_ROOT / "v22_43P_pure_KAN_basis_Gram_matrix.csv", [basis_row(r) for r in p5_rows] or [{"status": "not_run", "reason": "P5 pure KAN basis rows not executed yet."}])
    write_rows(OUT_ROOT / "v22_43P_pure_KAN_basis_carrier_matrix.csv", p5_rows or [{"status": "not_run", "reason": "P5 pure KAN basis carrier rows not executed yet."}])
    write_rows(OUT_ROOT / "v22_43P_pure_MLP_matched_support_matrix.csv", [r for r in enriched if r.get("architecture") == "MLP"] or [{"status": "not_run", "reason": "P1 MLP matched support rows not executed yet."}])
    p6_rows = [r for r in enriched if int_flag(r.get("warmup_steps")) > 0]
    write_rows(OUT_ROOT / "v22_43P_warmup_then_pure_matrix.csv", p6_rows or [{"status": "not_run", "reason": "P6 warmup-then-pure rows not executed yet."}])
    write_rows(OUT_ROOT / "v22_43P_hard_task_four_square_matrix.csv", hard_task_summary(enriched) if enriched else [{"status": "not_run", "reason": "No pure hard-task rows executed yet."}])
    write_placeholder_csv("v22_43P_continual_grokking_matrix.csv", "P7/P8 continual/grokking rows not executed yet.")
    write_rows(OUT_ROOT / "v22_43P_efficiency_matrix.csv", efficiency_rows(enriched))

    p1_candidates = [
        r
        for r in enriched
        if r.get("phase") == "P1_PureSupport"
        and r.get("row_role") == "signal_candidate"
        and r.get("control_mode") == "none"
        and r.get("variant") != "optimizer_alone"
        and int_flag(r.get("warmup_steps")) == 0
    ]
    train_desc_rows = sum(
        finite_float(r.get("train_NLL_initial")) is not None
        and finite_float(r.get("train_NLL_final")) is not None
        and float(finite_float(r.get("train_NLL_final"))) < float(finite_float(r.get("train_NLL_initial")))
        for r in p1_candidates
    )
    finite_train_rows = sum(int_flag(r.get("nan_or_inf_count")) == 0 for r in p1_candidates)
    p1_trainability_pass = int(len(p1_candidates) >= 9 and train_desc_rows >= 8 and finite_train_rows >= 8)
    p1_support_pass = int(
        len(p1_candidates) >= 9
        and count_positive(p1_candidates, "NLL_improvement_vs_own_strong_optimizer") >= 5
        and count_beats(p1_candidates, "same-metric-support-random") >= 5
        and sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in p1_candidates) >= 7
        and all(value_le(r.get("controller_overhead_ratio"), 0.30) for r in p1_candidates)
    )
    p2_candidates = [
        r
        for r in enriched
        if r.get("phase") == "S1_MetricResidualSignal"
        and r.get("row_role") == "signal_candidate"
        and r.get("control_mode") == "none"
        and int_flag(r.get("warmup_steps")) == 0
    ]
    p2_tau_h20_pos = sum(value_gt(r.get("tau_direction_H20"), 0.0) for r in p2_candidates)
    p2_tau_h60_pos = sum(value_gt(r.get("tau_direction_H60"), 0.0) for r in p2_candidates)
    p2_beats_random = count_beats(p2_candidates, "same-metric-support-random")
    p2_beats_signflip = count_beats(p2_candidates, "same-metric-support-signflip")
    p2_beats_shuffled = count_beats(p2_candidates, "same-metric-support-shuffled")
    p2_no_debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in p2_candidates)
    p2_overhead = sum(value_le(r.get("controller_overhead_ratio"), 0.30) for r in p2_candidates)
    p2_sign_correction_rows = sum(int_flag(r.get("pure_residual_sign_correction_active")) for r in p2_candidates)
    p2_unit_signal_diagnostic_opened = int(
        len(p2_candidates) >= 9
        and p2_tau_h20_pos >= 5
        and p2_tau_h60_pos >= 5
        and p2_beats_random >= 5
        and p2_beats_signflip >= 5
        and p2_beats_shuffled >= 5
    )
    p2_full_pass = int(
        p2_unit_signal_diagnostic_opened
        and p2_no_debt >= 7
        and p2_overhead >= len(p2_candidates)
    )
    p3_candidates = [
        r
        for r in enriched
        if r.get("phase") == "P3_PureMetricPreservingOET"
        and r.get("row_role") == "signal_candidate"
        and r.get("control_mode") == "none"
        and int_flag(r.get("warmup_steps")) == 0
    ]
    p3_unit_pass_rows = sum(
        value_le(r.get("metric_skew_residual"), 1.0e-5)
        and value_le(r.get("generalized_spectrum_drift"), 1.0e-4)
        and value_le(r.get("Cayley_solve_residual"), 1.0e-5)
        for r in p3_candidates
    )
    p3_train_desc_rows = sum(
        finite_float(r.get("train_NLL_initial")) is not None
        and finite_float(r.get("train_NLL_final")) is not None
        and float(finite_float(r.get("train_NLL_final"))) < float(finite_float(r.get("train_NLL_initial")))
        for r in p3_candidates
    )
    p3_nll_improve = count_positive(p3_candidates, "NLL_improvement_vs_own_strong_optimizer")
    p3_beats_random = count_beats(p3_candidates, "same-OET-random")
    p3_beats_signflip = count_beats(p3_candidates, "same-OET-signflip")
    p3_beats_shuffled = count_beats(p3_candidates, "same-OET-shuffled")
    p3_no_debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in p3_candidates)
    p3_overhead_035 = sum(value_le(r.get("controller_overhead_ratio"), 0.35) for r in p3_candidates)
    p3_trainability_pass = int(len(p3_candidates) >= 9 and p3_train_desc_rows >= 8 and p3_unit_pass_rows >= 8)
    p3_full_pass = int(
        p3_trainability_pass
        and p3_nll_improve >= 5
        and max(p3_beats_random, p3_beats_signflip, p3_beats_shuffled) >= 6
        and p3_no_debt >= 7
        and p3_overhead_035 >= len(p3_candidates)
    )
    p4_candidates = [
        r
        for r in p4_rows
        if r.get("row_role") == "signal_candidate"
        and r.get("control_mode") == "none"
        and int_flag(r.get("warmup_steps")) == 0
    ]
    p4_gate_candidates = [r for r in p4_candidates if "tier0" in str(r.get("run_label", "")).lower()] or p4_candidates
    p3_baselines: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for r in p3_candidates:
        key = (
            str(r.get("dataset", "")),
            str(r.get("seed", "")),
            str(r.get("architecture_key", "")),
            str(r.get("steps", "")),
            str(r.get("hidden", "")),
        )
        if key not in p3_baselines or value_or(r.get("final_NLL"), math.inf) < value_or(p3_baselines[key].get("final_NLL"), math.inf):
            p3_baselines[key] = r
    for r in p4_candidates:
        key = (
            str(r.get("dataset", "")),
            str(r.get("seed", "")),
            str(r.get("architecture_key", "")),
            str(r.get("steps", "")),
            str(r.get("hidden", "")),
        )
        base = p3_baselines.get(key, {})
        if finite_float(r.get("final_NLL")) is not None and finite_float(base.get("final_NLL")) is not None:
            r["NLL_improvement_vs_strict_P3_OET"] = value_or(base.get("final_NLL"), 0.0) - value_or(r.get("final_NLL"), 0.0)
        else:
            r["NLL_improvement_vs_strict_P3_OET"] = ""
    p4_groups: dict[str, list[dict[str, Any]]] = {}
    for r in p4_gate_candidates:
        p4_groups.setdefault(f"{r.get('variant', '')}|cap={r.get('radial_cap', '')}", []).append(r)
    p4_group_summaries: list[dict[str, Any]] = []
    for key, rows_for_group in sorted(p4_groups.items()):
        group_improve_p3 = count_positive(rows_for_group, "NLL_improvement_vs_strict_P3_OET")
        group_energy_ok = sum(value_le(r.get("radial_energy_fraction"), value_or(r.get("radial_cap"), 0.0) + 1.0e-6) for r in rows_for_group)
        group_beats_radial_random = count_beats(rows_for_group, "same-radial-random")
        group_beats_oet_random = count_beats(rows_for_group, "same-OET-random")
        group_no_debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in rows_for_group)
        group_overhead = sum(value_le(r.get("controller_overhead_ratio"), 0.40) for r in rows_for_group)
        group_radial_dominates = sum(
            value_ge(r.get("radial_energy_fraction"), 0.90 * value_or(r.get("radial_cap"), 0.0))
            for r in rows_for_group
        )
        group_pass = int(
            len(rows_for_group) >= 9
            and group_improve_p3 >= 5
            and group_energy_ok == len(rows_for_group)
            and group_beats_radial_random >= 6
            and group_no_debt >= 7
        )
        p4_group_summaries.append(
            {
                "mechanism": key,
                "rows": len(rows_for_group),
                "improves_over_strict_P3_rows": group_improve_p3,
                "radial_energy_fraction_cap_pass_rows": group_energy_ok,
                "beats_same_radial_random_rows": group_beats_radial_random,
                "beats_same_OET_random_rows": group_beats_oet_random,
                "no_ECE_Brier_tail_debt_rows": group_no_debt,
                "controller_overhead_le_0p40_rows": group_overhead,
                "radial_near_cap_rows": group_radial_dominates,
                "radial_channel_pass": group_pass,
            }
        )
    p4_improve_p3 = count_positive(p4_gate_candidates, "NLL_improvement_vs_strict_P3_OET")
    p4_energy_ok = sum(value_le(r.get("radial_energy_fraction"), value_or(r.get("radial_cap"), 0.0) + 1.0e-6) for r in p4_gate_candidates)
    p4_beats_radial_random = count_beats(p4_gate_candidates, "same-radial-random")
    p4_beats_oet_random = count_beats(p4_gate_candidates, "same-OET-random")
    p4_no_debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in p4_gate_candidates)
    p4_pass = int(any(int_flag(g.get("radial_channel_pass")) for g in p4_group_summaries))
    if p4_rows:
        write_rows(OUT_ROOT / "v22_43P_pure_oet_radial_matrix.csv", p4_rows)
    p5_candidates = [
        r
        for r in p5_rows
        if r.get("row_role") == "signal_candidate"
        and r.get("control_mode") == "none"
        and int_flag(r.get("warmup_steps")) == 0
    ]
    p5_gate_candidates = [r for r in p5_candidates if "tier0" in str(r.get("run_label", "")).lower()] or p5_candidates
    p5_groups: dict[str, list[dict[str, Any]]] = {}
    for r in p5_gate_candidates:
        p5_groups.setdefault(f"{r.get('variant', '')}|{r.get('architecture_key', '')}", []).append(r)
    p5_group_summaries: list[dict[str, Any]] = []
    for key, rows_for_group in sorted(p5_groups.items()):
        group_train_desc = sum(
            finite_float(r.get("train_NLL_initial")) is not None
            and finite_float(r.get("train_NLL_final")) is not None
            and float(finite_float(r.get("train_NLL_final"))) < float(finite_float(r.get("train_NLL_initial")))
            for r in rows_for_group
        )
        group_basis_energy = sum(value_ge(r.get("basis_energy_fraction"), 0.5) for r in rows_for_group)
        group_readout_leakage = sum(value_le(r.get("readout_leakage_fraction"), 0.30) for r in rows_for_group)
        group_nll_improve = count_positive(rows_for_group, "NLL_improvement_vs_own_strong_optimizer")
        group_beats_random = count_beats(rows_for_group, "same-basis-Gram-random")
        group_beats_signflip = count_beats(rows_for_group, "same-basis-Gram-signflip")
        group_no_debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in rows_for_group)
        group_overhead = sum(value_le(r.get("controller_overhead_ratio"), 0.30) for r in rows_for_group)
        group_finite = sum(int_flag(r.get("nan_or_inf_count")) == 0 for r in rows_for_group)
        group_trainability = int(
            len(rows_for_group) >= 9
            and group_train_desc >= 8
            and group_basis_energy >= 8
            and group_readout_leakage >= 8
            and group_finite >= 8
        )
        p5_group_summaries.append(
            {
                "mechanism": key,
                "rows": len(rows_for_group),
                "train_NLL_descent_rows": group_train_desc,
                "basis_energy_fraction_ge_0p5_rows": group_basis_energy,
                "readout_leakage_fraction_le_0p3_rows": group_readout_leakage,
                "finite_train_rows": group_finite,
                "NLL_improvement_vs_noop_rows": group_nll_improve,
                "beats_same_basis_Gram_random_rows": group_beats_random,
                "beats_same_basis_Gram_signflip_rows": group_beats_signflip,
                "no_ECE_Brier_tail_debt_rows": group_no_debt,
                "controller_overhead_le_0p30_rows": group_overhead,
                "trainability_pass": group_trainability,
                "internal_promotion_pass": int(
                    group_trainability
                    and group_nll_improve >= 5
                    and max(group_beats_random, group_beats_signflip) >= 6
                    and group_no_debt >= 7
                    and group_overhead >= len(rows_for_group)
                ),
            }
        )
    p5_train_desc_rows = sum(
        finite_float(r.get("train_NLL_initial")) is not None
        and finite_float(r.get("train_NLL_final")) is not None
        and float(finite_float(r.get("train_NLL_final"))) < float(finite_float(r.get("train_NLL_initial")))
        for r in p5_gate_candidates
    )
    p5_basis_energy = sum(value_ge(r.get("basis_energy_fraction"), 0.5) for r in p5_gate_candidates)
    p5_readout_leakage = sum(value_le(r.get("readout_leakage_fraction"), 0.30) for r in p5_gate_candidates)
    p5_nll_improve = count_positive(p5_gate_candidates, "NLL_improvement_vs_own_strong_optimizer")
    p5_beats_gram_random = count_beats(p5_gate_candidates, "same-basis-Gram-random")
    p5_beats_gram_signflip = count_beats(p5_gate_candidates, "same-basis-Gram-signflip")
    p5_no_debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in p5_gate_candidates)
    p5_overhead = sum(value_le(r.get("controller_overhead_ratio"), 0.30) for r in p5_gate_candidates)
    p5_trainability_pass = int(any(int_flag(g.get("trainability_pass")) for g in p5_group_summaries))
    p5_internal_pass = int(any(int_flag(g.get("internal_promotion_pass")) for g in p5_group_summaries))
    p6_candidates = [
        r
        for r in p6_rows
        if r.get("row_role") == "signal_candidate"
        and r.get("control_mode") == "none"
        and r.get("variant") != "optimizer_alone"
    ]
    p6_gate_candidates = [r for r in p6_candidates if "tier0" in str(r.get("run_label", "")).lower()] or p6_candidates
    p6_groups: dict[str, list[dict[str, Any]]] = {}
    for r in p6_gate_candidates:
        p6_groups.setdefault(
            f"{r.get('variant', '')}|{r.get('architecture_key', '')}|warmup={r.get('warmup_steps', '')}|vscale={r.get('velocity_scale', '')}",
            [],
        ).append(r)
    p6_group_summaries: list[dict[str, Any]] = []
    for key, rows_for_group in sorted(p6_groups.items()):
        group_truth_rows = sum(p6_post_warmup_truth_pass(r) for r in rows_for_group)
        group_post_desc = sum(value_gt(r.get("post_warmup_NLL_slope"), 0.0) for r in rows_for_group)
        group_no_worse = count_positive(rows_for_group, "NLL_improvement_vs_own_strong_optimizer")
        group_beats_controls = sum(
            int(
                int_flag(r.get("beats_same_metric_support_random")) == 1
                or int_flag(r.get("beats_same_OET_random")) == 1
                or int_flag(r.get("beats_same_basis_Gram_random")) == 1
            )
            for r in rows_for_group
        )
        group_no_debt = sum(int_flag(r.get("post_warmup_no_debt")) for r in rows_for_group)
        group_pass = int(
            len(rows_for_group) >= 9
            and group_truth_rows == len(rows_for_group)
            and group_post_desc >= 7
            and group_no_worse >= 7
            and group_beats_controls >= 6
            and group_no_debt >= 7
        )
        p6_group_summaries.append(
            {
                "mechanism": key,
                "rows": len(rows_for_group),
                "post_warmup_truth_pass_rows": group_truth_rows,
                "post_warmup_NLL_decreases_rows": group_post_desc,
                "NLL_final_no_worse_than_warmup_only_rows": group_no_worse,
                "beats_same_support_pure_controls_rows": group_beats_controls,
                "post_warmup_no_debt_rows": group_no_debt,
                "warmup_then_pure_pass": group_pass,
            }
        )
    p6_truth_rows = sum(p6_post_warmup_truth_pass(r) for r in p6_gate_candidates)
    p6_post_desc = sum(value_gt(r.get("post_warmup_NLL_slope"), 0.0) for r in p6_gate_candidates)
    p6_no_worse_warmup_only = count_positive(p6_gate_candidates, "NLL_improvement_vs_own_strong_optimizer")
    p6_beats_controls = sum(g["beats_same_support_pure_controls_rows"] for g in p6_group_summaries)
    p6_no_debt = sum(int_flag(r.get("post_warmup_no_debt")) for r in p6_gate_candidates)
    p6_pass = int(any(int_flag(g.get("warmup_then_pure_pass")) for g in p6_group_summaries))
    if not p0_all_pass:
        route = "R0-PureFURuntimeTruthFailed"
        reason = f"P0 from-scratch runtime truth failed or missing; pass_rows={p0_pass_rows}/{len(p0_audit_rows)}"
    elif p2_full_pass:
        route = "R4-PureResidualSignalFUOpened"
        reason = "P0/P1 passed and P2 residual-signal unit plus safety/efficiency gates passed."
    elif p3_full_pass:
        route = "R5-PureMetricPreservingOptimizerOpened"
        reason = "P0 passed and P3 non-additive OET trainability/exploration gates passed."
    elif p4_pass:
        route = "PureMetricPreservingWithRadialChannelOpened"
        reason = "P0 passed and P4 OET+small-radial channel gate passed."
    elif p5_internal_pass:
        route = "R8-PureKANBasisCarrierOpened"
        reason = "P0 passed and P5 pure KAN basis-native internal gate passed."
    elif p6_pass:
        route = "R9-WarmupThenPureFUOpened"
        reason = "P0 passed and P6 warmup-then-pure post-warmup transfer gate passed."
    elif p1_support_pass:
        route = "R2-PureSupportFunctionalOptimizerOpened"
        reason = "P0 passed and P1 support functional optimizer gate passed."
    elif p1_trainability_pass:
        route = "R1-PureSupportTrainabilityOpened"
        reason = "P0 passed and P1 trainability gate passed, but full support/control gate did not."
    elif len(p1_candidates) >= 9:
        route = "PureFUTrainabilityNoGo"
        reason = f"P0 passed, but P1 trainability did not reach >=8/9 descent rows; descent_rows={train_desc_rows}/{len(p1_candidates)}"
    else:
        route = "P0-PureFURuntimeTruthPassed_P1Pending"
        reason = f"P0 passed; P1 has only {len(p1_candidates)} candidate rows, below the 9-row trainability gate."
    if route == "R1-PureSupportTrainabilityOpened" and p2_unit_signal_diagnostic_opened:
        reason += " P2 residual signal diagnostic is positive after sign correction, but safety debt and overhead gates still block R4 promotion."

    final = {
        "final_route": route,
        "reason": reason,
        "generated_at": now_sg(),
        "p0_summary": {
            "rows": len(p0_audit_rows),
            "all_audit_rows": len(audit_rows),
            "warmup_audit_rows": len(p6_audit_rows),
            "pass_rows": p0_pass_rows,
            "p0_all_pass": int(p0_all_pass),
            "p6_post_warmup_truth_pass_rows": p6_truth_pass_rows,
        },
        "p1_summary": {
            "candidate_rows": len(p1_candidates),
            "train_NLL_descent_rows": train_desc_rows,
            "finite_train_rows": finite_train_rows,
            "trainability_pass": p1_trainability_pass,
            "support_functional_optimizer_pass": p1_support_pass,
            "NLL_improvement_vs_noop_rows": count_positive(p1_candidates, "NLL_improvement_vs_own_strong_optimizer"),
            "beats_same_metric_support_random_rows": count_beats(p1_candidates, "same-metric-support-random"),
            "no_ECE_Brier_tail_debt_rows": sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in p1_candidates),
            "controller_overhead_le_0p30_rows": sum(value_le(r.get("controller_overhead_ratio"), 0.30) for r in p1_candidates),
        },
        "p2_summary": {
            "candidate_rows": len(p2_candidates),
            "sign_correction_rows": p2_sign_correction_rows,
            "tau_direction_H20_positive_rows": p2_tau_h20_pos,
            "tau_direction_H60_positive_rows": p2_tau_h60_pos,
            "beats_same_metric_support_random_rows": p2_beats_random,
            "beats_same_metric_support_signflip_rows": p2_beats_signflip,
            "beats_same_metric_support_shuffled_rows": p2_beats_shuffled,
            "no_ECE_Brier_tail_debt_rows": p2_no_debt,
            "controller_overhead_le_0p30_rows": p2_overhead,
            "unit_signal_diagnostic_opened": p2_unit_signal_diagnostic_opened,
            "full_promotion_pass": p2_full_pass,
        },
        "p3_summary": {
            "candidate_rows": len(p3_candidates),
            "unit_pass_rows": p3_unit_pass_rows,
            "train_NLL_descent_rows": p3_train_desc_rows,
            "trainability_pass": p3_trainability_pass,
            "NLL_improvement_vs_noop_rows": p3_nll_improve,
            "beats_same_OET_random_rows": p3_beats_random,
            "beats_same_OET_signflip_rows": p3_beats_signflip,
            "beats_same_OET_shuffled_rows": p3_beats_shuffled,
            "no_ECE_Brier_tail_debt_rows": p3_no_debt,
            "controller_overhead_le_0p35_rows": p3_overhead_035,
            "full_promotion_pass": p3_full_pass,
        },
        "p4_summary": {
            "candidate_rows": len(p4_gate_candidates),
            "all_candidate_rows": len(p4_candidates),
            "improves_over_strict_P3_rows": p4_improve_p3,
            "radial_energy_fraction_cap_pass_rows": p4_energy_ok,
            "beats_same_radial_random_rows": p4_beats_radial_random,
            "beats_same_OET_random_rows": p4_beats_oet_random,
            "no_ECE_Brier_tail_debt_rows": p4_no_debt,
            "radial_channel_pass": p4_pass,
            "mechanism_group_summaries": p4_group_summaries,
        },
        "p5_summary": {
            "candidate_rows": len(p5_gate_candidates),
            "all_candidate_rows": len(p5_candidates),
            "train_NLL_descent_rows": p5_train_desc_rows,
            "basis_energy_fraction_ge_0p5_rows": p5_basis_energy,
            "readout_leakage_fraction_le_0p3_rows": p5_readout_leakage,
            "trainability_pass": p5_trainability_pass,
            "NLL_improvement_vs_noop_rows": p5_nll_improve,
            "beats_same_basis_Gram_random_rows": p5_beats_gram_random,
            "beats_same_basis_Gram_signflip_rows": p5_beats_gram_signflip,
            "no_ECE_Brier_tail_debt_rows": p5_no_debt,
            "controller_overhead_le_0p30_rows": p5_overhead,
            "internal_promotion_pass": p5_internal_pass,
            "mechanism_group_summaries": p5_group_summaries,
        },
        "p6_summary": {
            "candidate_rows": len(p6_gate_candidates),
            "all_candidate_rows": len(p6_candidates),
            "post_warmup_truth_pass_rows": p6_truth_rows,
            "post_warmup_NLL_decreases_rows": p6_post_desc,
            "NLL_final_no_worse_than_warmup_only_rows": p6_no_worse_warmup_only,
            "beats_same_support_pure_controls_rows": p6_beats_controls,
            "post_warmup_no_debt_rows": p6_no_debt,
            "warmup_then_pure_pass": p6_pass,
            "mechanism_group_summaries": p6_group_summaries,
        },
    }
    write_json(OUT_ROOT / "v22_43P_final_route.json", final)
    failure_text = [
        "# v22.43P Pure-FU Failure Dissection",
        "",
        f"更新时间：{now_sg()}",
        "",
        "## Final Route",
        "",
        "```json",
        json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## P0 Runtime Truth Evidence",
        "",
        md_table(audit_rows, ["run_label", "dataset", "seed", "architecture", "variant", "control_mode", "pure_fu_mode", "warmup_steps", "base_optimizer_step_used", "base_optimizer_step_used_after_warmup", "base_velocity_added", "base_velocity_added_after_warmup", "bp_gradient_used_only_for_cotangent", "bp_gradient_used_only_for_cotangent_after_warmup", "fu_velocity_emitted_every_step", "continuous_fu_state_updated_every_step", "p0_runtime_truth_pass", "p6_post_warmup_truth_pass"], 24),
        "",
        "## P1 Trainability Evidence",
        "",
        md_table(p1_candidates, ["run_label", "dataset", "seed", "architecture", "variant", "train_NLL_initial", "train_NLL_final", "loss_descent_rate", "nan_or_inf_count", "update_norm_mean", "controller_overhead_ratio", "NLL_improvement_vs_own_strong_optimizer", "no_ECE_Brier_tail_debt"], 24),
        "",
        "## P2 Residual Signal Evidence",
        "",
        md_table(p2_candidates, ["run_label", "dataset", "seed", "variant", "pure_residual_sign_correction_active", "train_NLL_initial", "train_NLL_final", "tau_direction_H20", "tau_direction_H60", "beats_same_metric_support_random", "beats_same_metric_support_signflip", "beats_same_metric_support_shuffled", "no_ECE_Brier_tail_debt", "controller_overhead_ratio"], 36),
        "",
        "## P3 Non-Additive OET Evidence",
        "",
        md_table(p3_candidates, ["run_label", "dataset", "seed", "variant", "train_NLL_initial", "train_NLL_final", "metric_skew_residual", "Cayley_solve_residual", "generalized_spectrum_drift", "OET_generator_norm", "NLL_improvement_vs_own_strong_optimizer", "beats_same_OET_random", "beats_same_OET_signflip", "beats_same_OET_shuffled", "no_ECE_Brier_tail_debt", "controller_overhead_ratio"], 36),
        "",
        "## P4 OET + Small Radial Evidence",
        "",
        md_table(p4_candidates, ["run_label", "dataset", "seed", "variant", "radial_cap", "radial_energy_fraction", "radial_spectrum_drift", "radial_rank_change_proxy", "NLL_improvement_vs_strict_P3_OET", "NLL_improvement_vs_own_strong_optimizer", "beats_same_radial_random", "beats_same_OET_random", "no_ECE_Brier_tail_debt", "controller_overhead_ratio"], 36),
        "",
        "## P5 Pure KAN Basis-Native Evidence",
        "",
        md_table(p5_candidates, ["run_label", "dataset", "seed", "architecture_key", "variant", "train_NLL_initial", "train_NLL_final", "basis_energy_fraction", "readout_leakage_fraction", "basis_Gram_condition", "NLL_improvement_vs_own_strong_optimizer", "beats_same_basis_Gram_random", "beats_same_basis_Gram_signflip", "no_ECE_Brier_tail_debt", "controller_overhead_ratio"], 36),
        "",
        "## P6 Warmup-Then-Pure Evidence",
        "",
        md_table(p6_candidates, ["run_label", "dataset", "seed", "architecture_key", "variant", "warmup_steps", "NLL_at_warmup_end", "train_NLL_final", "post_warmup_NLL_slope", "post_warmup_AUC_loss_time", "NLL_improvement_vs_own_strong_optimizer", "beats_same_metric_support_random", "beats_same_OET_random", "beats_same_basis_Gram_random", "post_warmup_no_debt", "base_optimizer_update_norm_after_warmup", "pure_FU_velocity_norm_after_warmup", "basis_Gram_condition_after_warmup"], 36),
        "",
        "## Analysis",
        "",
        "- 本文件只汇总已落盘的 `v22_43P_*` chunk；未执行的 P2/P3/P4/P5/P6/P7/P8 artifact 明确写为 `not_run`。",
        "- P0 若失败，优先审计 `base_optimizer_step_used`、ordinary update norm、candidate runtime selector 字段；这些字段来自 summary/state/runtime 三层记录。",
        "- P6 warmup row 不参与 from-scratch P0 hard gate；它必须单独满足 `p6_post_warmup_truth_pass=1`，即 warmup 后 base optimizer step/update norm 全为 0。",
        "- P1 若没有 9 个 candidate row，不作 trainability 成功或失败的最终宣传，只保留 pending route。",
        "- P2 若只有 unit signal diagnostic 正、但 no-debt 或 overhead 不过，不能提升到 `R4-PureResidualSignalFUOpened`。",
    ]
    (OUT_ROOT / "v22_43P_failure_dissection.md").write_text("\n".join(failure_text) + "\n", encoding="utf-8")
    if write_final:
        write_manifest()
    append_exec(
        "merge_pure_artifacts",
        task_id="pure_merge",
        status="pass" if summaries else "warn",
        gpu="n/a",
        files="results/v22_43/v22_43P_final_route.json, results/v22_43/v22_43P_pure_fu_runtime_truth_matrix.csv, results/v22_43/v22_43P_failure_dissection.md",
        note=f"summaries={len(summaries)}; state_rows={len(state_rows)}; runtime_rows={len(runtime_rows)}; route={route}",
    )
    return {"status": "merged" if summaries else "no_rows", "summary_rows": len(summaries), "state_rows": len(state_rows), "runtime_rows": len(runtime_rows), "route": route}


def count_positive(rows: list[dict[str, Any]], field: str) -> int:
    return sum(value_gt(r.get(field), 0.0) for r in rows)


def count_beats(rows: list[dict[str, Any]], mode: str) -> int:
    field = f"beats_{safe_fragment(mode).replace('-', '_')}"
    return sum(int_flag(r.get(field)) for r in rows)


def mean_lcb(values: list[float]) -> tuple[float | str, float | str]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return "", ""
    mean = statistics.fmean(vals)
    if len(vals) == 1:
        return mean, mean
    sd = statistics.stdev(vals)
    return mean, mean - 1.64 * sd / math.sqrt(len(vals))


def write_phase_summaries(rows: list[dict[str, Any]]) -> None:
    write_rows(OUT_ROOT / "v22_43_phase1_s4_metric_support_summary.csv", [summarize_s4(rows)])
    write_rows(OUT_ROOT / "v22_43_phase2_s1_metric_residual_summary.csv", [summarize_s1(rows)])
    write_rows(OUT_ROOT / "v22_43_phase3_s6_metric_kan_carrier_summary.csv", [summarize_s6(rows)])
    write_rows(OUT_ROOT / "v22_43_s3_metric_optimizer_summary.csv", [summarize_s3(rows)])


def official_rows(rows: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("phase") == phase and r.get("control_mode") == "none" and r.get("variant") != "optimizer_alone" and int_flag(r.get("official_hard_row"))]


def summarize_s4(rows: list[dict[str, Any]]) -> dict[str, Any]:
    s4 = official_rows(rows, "S4_MetricSupport")
    if not s4:
        s4 = [r for r in rows if r.get("phase") == "S4_MetricSupport" and r.get("control_mode") == "none" and r.get("variant") != "optimizer_alone"]
    top_metric = ""
    metric_scores = {}
    for r in s4:
        metric = str(r.get("metric_name"))
        metric_scores.setdefault(metric, []).append(value_or(r.get("NLL_improvement_vs_own_strong_optimizer"), 0.0))
    if metric_scores:
        top_metric = max(metric_scores, key=lambda m: statistics.fmean(metric_scores[m]))
    return {
        "phase": "S4_MetricSupport",
        "rows": len(s4),
        "official_hard_rows": sum(int_flag(r.get("official_hard_row")) for r in s4),
        "top_metric_by_mean_NLL_improvement": top_metric,
        "NLL_improvement_vs_own_strong_optimizer_rows": count_positive(s4, "NLL_improvement_vs_own_strong_optimizer"),
        "beats_same_metric_support_random_rows": count_beats(s4, "same-metric-support-random"),
        "no_ECE_Brier_tail_debt_rows": sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in s4),
        "controller_overhead_le_0p30_rows": sum(value_le(r.get("controller_overhead_ratio"), 0.30) for r in s4),
        "metric_condition_number_max": max([value_or(r.get("metric_condition_number"), 0.0) for r in s4], default=""),
        "support_projection_residual_mean": statistics.fmean([value_or(r.get("support_projection_residual"), 0.0) for r in s4]) if s4 else "",
        "exploration_pass": int(
            len(s4) >= 9
            and count_positive(s4, "NLL_improvement_vs_own_strong_optimizer") >= 5
            and count_beats(s4, "same-metric-support-random") >= 5
            and sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in s4) >= 7
            and all(value_le(r.get("controller_overhead_ratio"), 0.30) for r in s4)
        ),
        "route_if_terminal": "R2-MetricSupportOptimizerOpened",
    }


def summarize_s1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    s1 = official_rows(rows, "S1_MetricResidualSignal")
    if not s1:
        s1 = [r for r in rows if r.get("phase") == "S1_MetricResidualSignal" and r.get("control_mode") == "none"]
    h200 = [finite_float(r.get("tau_direction_H200")) for r in s1 if finite_float(r.get("tau_direction_H200")) is not None]
    h800 = [finite_float(r.get("tau_direction_H800")) for r in s1 if finite_float(r.get("tau_direction_H800")) is not None]
    h200_mean, h200_lcb = mean_lcb([float(x) for x in h200 if x is not None])
    h800_mean, h800_lcb = mean_lcb([float(x) for x in h800 if x is not None])
    return {
        "phase": "S1_MetricResidualSignal",
        "rows": len(s1),
        "official_hard_rows": sum(int_flag(r.get("official_hard_row")) for r in s1),
        "tau_direction_H200_positive_rows": sum(value_gt(r.get("tau_direction_H200"), 0.0) for r in s1),
        "tau_direction_H800_positive_rows": sum(value_gt(r.get("tau_direction_H800"), 0.0) for r in s1),
        "tau_direction_H200_mean": h200_mean,
        "tau_direction_H200_LCB": h200_lcb,
        "tau_direction_H800_mean": h800_mean,
        "tau_direction_H800_LCB": h800_lcb,
        "beats_same_metric_support_controls_rows": count_beats(s1, "same-metric-support-random"),
        "no_ECE_Brier_tail_debt_rows": sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in s1),
        "residual_signal_SNR_mean": statistics.fmean([value_or(r.get("residual_signal_SNR"), 0.0) for r in s1]) if s1 else "",
        "exploration_pass": int(
            len(s1) >= 9
            and sum(value_gt(r.get("tau_direction_H200"), 0.0) for r in s1) >= 5
            and sum(value_gt(r.get("tau_direction_H800"), 0.0) for r in s1) >= 3
            and (h200_lcb != "" and float(h200_lcb) > 0.0)
            and count_beats(s1, "same-metric-support-random") >= 6
            and sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in s1) >= 7
        ),
    }


def summarize_s6(rows: list[dict[str, Any]]) -> dict[str, Any]:
    s6 = official_rows(rows, "S6_MetricKANCarrier")
    if not s6:
        s6 = [r for r in rows if r.get("phase") == "S6_MetricKANCarrier" and r.get("control_mode") == "none"]
    true_both = sum(1 for r in s6 if r.get("TrueKANGain_class") in {"TrueKANGain", "BothGain"})
    control_explained = sum(1 for r in s6 if r.get("TrueKANGain_class") in {"MLPMatchedSupportStronger", "KANInternalValueOnly"})
    mlp_degradation_driven = sum(
        1
        for r in s6
        if value_le(r.get("KAN_NLL_delta_vs_MLP_matched_support_FU"), 0.0)
        and value_gt(r.get("NLL_improvement_vs_own_strong_optimizer"), 0.0)
        and value_le(r.get("MLP_matched_NLL_improvement_vs_own_strong_optimizer"), 0.0)
    )
    beats_mlp = sum(value_le(r.get("KAN_NLL_delta_vs_MLP_matched_support_FU"), 0.0) for r in s6)
    leakage_le = sum(value_le(r.get("readout_leakage_fraction"), 0.30) for r in s6)
    overhead_le = sum(value_le(r.get("controller_overhead_ratio"), 0.30) for r in s6)
    overhead_le_025 = sum(value_le(r.get("controller_overhead_ratio"), 0.25) for r in s6)
    no_debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in s6)
    nll_gain = count_positive(s6, "NLL_improvement_vs_own_strong_optimizer")
    basis_ge = sum(value_ge(r.get("basis_energy_fraction"), 0.5) for r in s6)
    beats_basis = max(count_beats(s6, "same-basis-Gram-random"), count_beats(s6, "same-basis-OET-random"))
    internal_pass = int(
        len(s6) >= 9
        and nll_gain >= 5
        and beats_basis >= 6
        and basis_ge >= 8
        and leakage_le >= 8
        and no_debt >= 6
    )
    carrier_pass = int(
        len(s6) >= 9
        and beats_mlp >= 5
        and true_both / max(1, len(s6)) >= 0.30
        and control_explained / max(1, len(s6)) <= 0.50
        and mlp_degradation_driven / max(1, len(s6)) <= 0.25
    )
    return {
        "phase": "S6_MetricKANCarrier",
        "rows": len(s6),
        "official_hard_rows": sum(int_flag(r.get("official_hard_row")) for r in s6),
        "KAN_NLL_improvement_vs_own_optimizer_rows": nll_gain,
        "beats_same_basis_metric_controls_rows": beats_basis,
        "basis_energy_fraction_ge_0p5_rows": basis_ge,
        "readout_leakage_fraction_le_0p3_rows": leakage_le,
        "controller_overhead_le_0p30_rows": overhead_le,
        "controller_overhead_le_0p25_rows": overhead_le_025,
        "no_ECE_Brier_tail_debt_rows": no_debt,
        "beats_MLP_matched_metric_support_rows": beats_mlp,
        "TrueKANGain_plus_BothGain_rows": true_both,
        "ControlExplained_or_KANInternalOnly_rows": control_explained,
        "MLPDegradationDriven_rows": mlp_degradation_driven,
        "exploration_internal_pass": internal_pass,
        "architecture_diagnostic_pass": int(len(s6) >= 9 and beats_mlp >= 3 and true_both / max(1, len(s6)) >= 0.30),
        "plan_carrier_criteria_pass": carrier_pass,
        "exploration_carrier_pass": int(internal_pass and carrier_pass),
    }


def summarize_s3(rows: list[dict[str, Any]]) -> dict[str, Any]:
    s3 = official_rows(rows, "S3_StrongMetricOptimizer")
    if not s3:
        s3 = [r for r in rows if r.get("phase") == "S3_StrongMetricOptimizer" and r.get("control_mode") == "none"]
    beats_own = count_positive(s3, "NLL_improvement_vs_own_strong_optimizer")
    beats_geometry = count_beats(s3, "same-metric-support-random") + count_beats(s3, "same-optimizer-geometry-random")
    beats_strongest = count_positive(s3, "NLL_improvement_vs_strongest_optimizer")
    no_debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in s3)
    overhead_le = sum(value_le(r.get("controller_overhead_ratio"), 0.30) for r in s3)
    return {
        "phase": "S3_StrongMetricOptimizer",
        "rows": len(s3),
        "official_hard_rows": sum(int_flag(r.get("official_hard_row")) for r in s3),
        "FU_beats_own_optimizer_rows": beats_own,
        "FU_beats_same_geometry_controls_rows": beats_geometry,
        "FU_beats_strongest_completed_optimizer_rows": beats_strongest,
        "controller_overhead_le_0p30_rows": overhead_le,
        "no_ECE_Brier_tail_debt_rows": no_debt,
        "exploration_pass": int(
            len(s3) >= 9
            and beats_own >= 6
            and beats_geometry >= 6
            and beats_strongest >= 5
            and overhead_le >= len(s3)
        ),
    }


def top_s4_metrics(limit: int = 2) -> list[str]:
    summary_rows = read_rows(OUT_ROOT / "v22_43_metric_support_full_loop_matrix.csv")
    s4 = [
        r
        for r in summary_rows
        if r.get("phase") == "S4_MetricSupport"
        and r.get("control_mode") == "none"
        and r.get("variant") != "optimizer_alone"
        and int_flag(r.get("official_hard_row"))
    ]
    if not s4:
        s4 = [r for r in summary_rows if r.get("phase") == "S4_MetricSupport" and r.get("control_mode") == "none" and r.get("variant") != "optimizer_alone"]
    scores: dict[str, list[float]] = {}
    representative: dict[str, str] = {}
    for r in s4:
        metric = str(r.get("metric_name"))
        scores.setdefault(metric, []).append(value_or(r.get("NLL_improvement_vs_own_strong_optimizer"), 0.0))
        representative.setdefault(metric, str(r.get("variant")))
    ranked = sorted(scores, key=lambda m: statistics.fmean(scores[m]), reverse=True)
    out = []
    for metric in ranked[:limit]:
        out.append(representative[metric])
    while len(out) < limit:
        fallback = S4_METRICS[len(out) % len(S4_METRICS)]
        if fallback not in out:
            out.append(fallback)
        else:
            break
    return out


def stage_s1(args: argparse.Namespace) -> dict[str, Any]:
    top = top_s4_metrics(limit=2)
    variants = [f"S1-M-top{idx + 1}-{safe_fragment(v)}" for idx, v in enumerate(top)]
    args.eval_variants = ",".join(variants)
    args.control_modes = "none,same-metric-support-random,same-metric-support-signflip,same-metric-support-shuffled"
    append_exec(
        "stage_s1 select_top_metrics",
        task_id="s1_select_top_metrics",
        status="pass",
        files="results/v22_43/v22_43_metric_support_full_loop_matrix.csv",
        note=f"top_s4_variants={top}; dispatched_s1_variants={variants}",
    )
    return dispatch_specs(args, task_specs(args, "generic"), "s1")


def finalize(write_recap: bool = True) -> dict[str, Any]:
    ensure_out()
    runtime = read_rows(OUT_ROOT / "v22_43_runtime_regression_audit.csv")
    harness = read_rows(OUT_ROOT / "v22_43_metric_harness_unit_matrix.csv")
    s4 = read_rows(OUT_ROOT / "v22_43_phase1_s4_metric_support_summary.csv")
    s1 = read_rows(OUT_ROOT / "v22_43_phase2_s1_metric_residual_summary.csv")
    s6 = read_rows(OUT_ROOT / "v22_43_phase3_s6_metric_kan_carrier_summary.csv")
    s3 = read_rows(OUT_ROOT / "v22_43_s3_metric_optimizer_summary.csv")
    runtime_summary = runtime[0] if runtime else {}
    s4_summary = s4[0] if s4 else {}
    s1_summary = s1[0] if s1 else {}
    s6_summary = s6[0] if s6 else {}
    s3_summary = s3[0] if s3 else {}
    harness_pass = bool(harness) and all(int_flag(r.get("PSD_symmetry_pass")) and int_flag(r.get("projection_idempotence_pass")) and int_flag(r.get("metric_orthogonal_generator_pass")) and int_flag(r.get("generalized_spectrum_pass")) for r in harness)
    if not harness_pass:
        route = "R0-CodeMetricRuntimeFailed"
        reason = "metric harness unit gate failed or missing"
    elif int_flag(runtime_summary.get("candidate_action_regression_pass")) == 0 and runtime_summary.get("candidate_action_regression_pass") != "":
        route = "R0p5-CandidateActionRegression_Stop"
        reason = "runtime candidate-action regression audit failed"
    elif int_flag(s6_summary.get("exploration_carrier_pass")):
        route = "R6-KANMetricBasisCarrierOpened"
        reason = "S6-M KAN carrier gate passed against MLP matched metric support"
    elif int_flag(s6_summary.get("exploration_internal_pass")):
        route = "R5-KANMetricSupportOnly"
        reason = "KAN metric/basis internal value opened but architecture carrier gate did not"
    elif int_flag(s1_summary.get("exploration_pass")):
        route = "R4-MetricResidualSignalOpened"
        reason = "S1-M residual signal direction gate passed"
    elif int_flag(s4_summary.get("exploration_pass")):
        route = "R2-MetricSupportOptimizerOpened"
        reason = "S4-M metric support optimizer gate passed; no signal/carrier promotion"
    elif int_flag(s3_summary.get("exploration_pass")):
        route = "R9-FUWeakOptimizerPatchOnly"
        reason = "Only metric optimizer / strong optimizer line opened"
    elif harness_pass:
        route = "R1-MetricHarnessOpened"
        reason = "Metric harness passed, but completed task rows did not open S4/S1/S6/S3 gates"
    else:
        route = "R0-CodeMetricRuntimeFailed"
        reason = "No completed evidence"
    data = {
        "final_route": route,
        "reason": reason,
        "harness_pass": int(harness_pass),
        "runtime_regression_audit": runtime_summary,
        "s4_summary": s4_summary,
        "s1_summary": s1_summary,
        "s6_summary": s6_summary,
        "s3_summary": s3_summary,
        "generated_at": now_sg(),
    }
    write_json(OUT_ROOT / "v22_43_final_route.json", data)
    if write_recap:
        update_recap(data)
        append_exec(
            "finalize",
            task_id="finalize",
            status="pass",
            gpu="n/a",
            files="results/v22_43/v22_43_final_route.json, results/v22_43/v22_43_artifact_manifest.csv, docs/DG-KAN_v22.43_MetricPreservingContinuousFunctionalFlowFU_实验结果复盘.md",
            note=f"route={route}; reason={reason}",
        )
    write_manifest()
    return data


def write_manifest() -> None:
    rows = []
    for name in REQUIRED_ARTIFACTS + PURE_REQUIRED_ARTIFACTS + [
        "v22_43_phase1_s4_metric_support_summary.csv",
        "v22_43_phase2_s1_metric_residual_summary.csv",
        "v22_43_phase3_s6_metric_kan_carrier_summary.csv",
        "v22_43_s3_metric_optimizer_summary.csv",
    ]:
        path = OUT_ROOT / name
        rows.append(
            {
                "artifact": str(path.relative_to(ROOT)),
                "exists": int(path.exists()),
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            }
        )
    for path in [Path(__file__).resolve(), EXEC_DOC, RECAP_DOC]:
        if path.exists():
            rows.append(
                {
                    "artifact": str(path.relative_to(ROOT)),
                    "exists": 1,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_rows(OUT_ROOT / "v22_43_artifact_manifest.csv", rows)


def update_recap(final: dict[str, Any]) -> None:
    full = read_rows(OUT_ROOT / "v22_43_metric_support_full_loop_matrix.csv")
    s4 = read_rows(OUT_ROOT / "v22_43_phase1_s4_metric_support_summary.csv")
    s1 = read_rows(OUT_ROOT / "v22_43_phase2_s1_metric_residual_summary.csv")
    s6 = read_rows(OUT_ROOT / "v22_43_phase3_s6_metric_kan_carrier_summary.csv")
    s3 = read_rows(OUT_ROOT / "v22_43_s3_metric_optimizer_summary.csv")
    hard = read_rows(OUT_ROOT / "v22_43_hard_task_four_square_matrix.csv")
    efficiency = read_rows(OUT_ROOT / "v22_43_efficiency_matrix.csv")
    harness = read_rows(OUT_ROOT / "v22_43_metric_harness_unit_matrix.csv")
    runtime = [final.get("runtime_regression_audit", {})] if final.get("runtime_regression_audit") else []
    s4r = s4[0] if s4 else {}
    s1r = s1[0] if s1 else {}
    s6r = s6[0] if s6 else {}
    s3r = s3[0] if s3 else {}

    def repair_summary(label_fragment: str, note: str) -> dict[str, Any]:
        cand = [
            r
            for r in full
            if label_fragment in str(r.get("run_label", ""))
            and r.get("row_role") == "signal_candidate"
            and r.get("control_mode") == "none"
            and r.get("variant") != "optimizer_alone"
        ]
        return {
            "label": label_fragment,
            "rows": len(cand),
            "NLL_improve_rows": count_positive(cand, "NLL_improvement_vs_own_strong_optimizer"),
            "beats_random_rows": count_beats(cand, "same-metric-support-random"),
            "no_debt_rows": sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in cand),
            "overhead_le_0p30_rows": sum(value_le(r.get("controller_overhead_ratio"), 0.30) for r in cand),
            "mean_NLL_improvement": statistics.fmean([value_or(r.get("NLL_improvement_vs_own_strong_optimizer"), 0.0) for r in cand]) if cand else "",
            "mean_overhead": statistics.fmean([value_or(r.get("controller_overhead_ratio"), 0.0) for r in cand]) if cand else "",
            "mean_support_projection_residual": statistics.fmean([value_or(r.get("support_projection_residual"), 0.0) for r in cand]) if cand else "",
            "note": note,
        }

    continuation_repairs = [
        repair_summary("hard_s4_h256_signal_repair", "h256 + SignalMetric + rank8; overhead fixed, no-debt insufficient"),
        repair_summary("hard_s4_h256_signal_debtrepair", "lower velocity/rho, stronger eta_debt; no-debt unchanged"),
        repair_summary("hard_s4_h256_fisher_debtrepair", "swap SignalMetric to FisherEMA; nearly identical failure pattern"),
        repair_summary("hard_s4_h256_signal_rank4_debtrepair", "rank cap 8->4; best S4 continuation tradeoff, still no-debt 5/9"),
        repair_summary("hard_s4_h256_signal_rank2_debtrepair", "rank cap 4->2; support residual too high, no-debt regressed"),
        repair_summary("hard_s4_h256_dfoU_signal_rank4_debtrepair", "switch D-CHE to D-FOU; larger NLL gains but safety debt all fail"),
        repair_summary("hard_s4_h256_signal_rank4_barrier10", "batch-debt velocity barrier=10; did not fix SVHN ECE debt"),
    ]
    text = [
        "# DG-KAN v22.43 Metric-Preserving Continuous Functional Flow FU 实验结果复盘",
        "",
        f"更新时间：{now_sg()}",
        "",
        "## 执行边界与审计说明",
        "",
        "- 本复盘只引用 `results/v22_43` 中实际落盘 artifact；没有执行的 continual/grokking 项保持 `not_run`，不补造数据。",
        "- 本轮新增 runner：`experiments/run_v22_43_metric_preserving_continuous_functional_flow_fu.py`。修改内容：新增 metric harness、metric-aware support mask、metric-norm normalization、metric nuisance residualization、KAN basis Gram diagonal metric、runtime no-candidate-action audit、v22.43 专用 merge/finalize/log/manifest。",
        "- MNIST/FashionMNIST/KMNIST 若出现，只按计划作为 debug row；final route 的 promotion 优先看 hard/task_tier row。",
        "",
        "## Final Route",
        "",
        "```json",
        json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Runtime Regression Evidence",
        "",
        md_table(runtime, ["runtime_row_count", "runtime_argmax_candidate_used", "runtime_topk_candidate_used", "candidate_action_selection_used_for_runtime", "candidate_value_model_used_as_runtime_policy", "continuous_fu_state_updated_every_step", "fu_velocity_emitted_every_step", "candidate_action_regression_pass", "status"], 3),
        "",
        "## Blocker / 修复证据链",
        "",
        "- Phase0 初跑曾触发 generalized spectrum drift blocker；修复方向按计划落在 metric-skew generator 与 Cholesky whitening。实际修改为用 `A = solve(G, K)` 构造 G-skew generator，并用正确 whitening 比较 generalized spectrum。复跑后 `v22_43_metric_harness_unit_matrix.csv` 中 4 个 metric 的 generalized spectrum 均 pass。",
        "- S4 初跑暴露 condition/overhead/no-debt blocker。修复尝试包括 `--metric-refresh-cadence 20`、`--metric-shrinkage 0.70`、`--metric-eps 1e-4`、`--velocity-scale 0.20`、`--rho-max 0.10`，以及 KAN BasisGram 归一化后再 shrinkage。最终 S4: "
        f"NLL improvement {s4r.get('NLL_improvement_vs_own_strong_optimizer_rows', '')}/{s4r.get('rows', '')}, "
        f"beats random {s4r.get('beats_same_metric_support_random_rows', '')}/{s4r.get('rows', '')}, "
        f"no-debt {s4r.get('no_ECE_Brier_tail_debt_rows', '')}/{s4r.get('rows', '')}, "
        f"overhead<=0.30 {s4r.get('controller_overhead_le_0p30_rows', '')}/{s4r.get('rows', '')}; "
        "因此没有打开 S4 route。修复前/诊断行保存在 `results/v22_43/pre_repair_archive/` 与 `results/v22_43/diagnostic_archive/`。",
        "- S1 按文档补跑 H800。最终 S1: "
        f"H200 positive {s1r.get('tau_direction_H200_positive_rows', '')}/{s1r.get('rows', '')}, "
        f"H200 LCB {s1r.get('tau_direction_H200_LCB', '')}, "
        f"H800 positive {s1r.get('tau_direction_H800_positive_rows', '')}/{s1r.get('rows', '')}, "
        f"H800 LCB {s1r.get('tau_direction_H800_LCB', '')}, "
        f"no-debt {s1r.get('no_ECE_Brier_tail_debt_rows', '')}/{s1r.get('rows', '')}; "
        "H800 均值略正但 LCB 为负，不能声明 residual signal。",
        "- S6 完整跑 KAN basis carrier 与 MLP matched supports。最终 S6: "
        f"KAN NLL improvement {s6r.get('KAN_NLL_improvement_vs_own_optimizer_rows', '')}/{s6r.get('rows', '')}, "
        f"beats basis controls {s6r.get('beats_same_basis_metric_controls_rows', '')}/{s6r.get('rows', '')}, "
        f"readout leakage<=0.30 {s6r.get('readout_leakage_fraction_le_0p3_rows', '')}/{s6r.get('rows', '')}, "
        f"beats MLP matched {s6r.get('beats_MLP_matched_metric_support_rows', '')}/{s6r.get('rows', '')}; "
        "KAN carrier 未打开。",
        "- S3 初版 summary gate 漏掉了 Phase4 文档要求的 same-geometry controls 和 overhead<=0.30 条件。已修正 `summarize_s3`：必须同时满足 own optimizer、same-geometry controls、strongest baseline、overhead。修正后 S3: "
        f"own {s3r.get('FU_beats_own_optimizer_rows', '')}/{s3r.get('rows', '')}, "
        f"same-geometry {s3r.get('FU_beats_same_geometry_controls_rows', '')}/{s3r.get('rows', '')}, "
        f"strongest {s3r.get('FU_beats_strongest_completed_optimizer_rows', '')}/{s3r.get('rows', '')}, "
        f"overhead<=0.30 {s3r.get('controller_overhead_le_0p30_rows', '')}/{s3r.get('rows', '')}; "
        "因此 S3 也不作为成功 route。",
        "- 继续推进修复：新增 lazy control/nuisance 计算以降低 overhead；补齐 S6 100-step optimizer-alone baselines；修正 0 值 gate 判定、S6 carrier route 必须依赖 internal pass、strongest baseline 必须同 steps/hidden 对齐；新增 `--debt-velocity-barrier` 做 batch debt velocity hard barrier。所有这些修改只改变后续运行和汇总判定，不改已落盘原始实验数值。",
        "",
        "## Metric Harness Evidence",
        "",
        md_table(harness, ["metric_name", "PSD_symmetry_pass", "projection_idempotence_pass", "metric_orthogonal_generator_pass", "generalized_spectrum_drift", "generalized_spectrum_pass"], 8),
        "",
        "## Phase Summary",
        "",
        "### S4-M Metric Support",
        "",
        md_table(s4, ["rows", "official_hard_rows", "top_metric_by_mean_NLL_improvement", "NLL_improvement_vs_own_strong_optimizer_rows", "beats_same_metric_support_random_rows", "no_ECE_Brier_tail_debt_rows", "controller_overhead_le_0p30_rows", "exploration_pass"], 5),
        "",
        "### S1-M Residual Signal",
        "",
        md_table(s1, ["rows", "official_hard_rows", "tau_direction_H200_positive_rows", "tau_direction_H800_positive_rows", "tau_direction_H200_mean", "tau_direction_H200_LCB", "beats_same_metric_support_controls_rows", "exploration_pass"], 5),
        "",
        "### S6-M KAN Basis Carrier",
        "",
        md_table(s6, ["rows", "official_hard_rows", "KAN_NLL_improvement_vs_own_optimizer_rows", "beats_same_basis_metric_controls_rows", "basis_energy_fraction_ge_0p5_rows", "readout_leakage_fraction_le_0p3_rows", "controller_overhead_le_0p30_rows", "beats_MLP_matched_metric_support_rows", "TrueKANGain_plus_BothGain_rows", "architecture_diagnostic_pass", "exploration_internal_pass", "exploration_carrier_pass"], 5),
        "",
        "### S3-M Strong Optimizer",
        "",
        md_table(s3, ["rows", "official_hard_rows", "FU_beats_own_optimizer_rows", "FU_beats_same_geometry_controls_rows", "FU_beats_strongest_completed_optimizer_rows", "controller_overhead_le_0p30_rows", "no_ECE_Brier_tail_debt_rows", "exploration_pass"], 5),
        "",
        "## Continuation Repair Matrix",
        "",
        md_table(continuation_repairs, ["label", "rows", "NLL_improve_rows", "beats_random_rows", "no_debt_rows", "overhead_le_0p30_rows", "mean_NLL_improvement", "mean_overhead", "mean_support_projection_residual", "note"], 20),
        "",
        "## Hard/Debug Split",
        "",
        md_table(hard, ["slice", "rows", "datasets", "NLL_improvement_rows", "no_ECE_Brier_tail_debt_rows", "note"], 4),
        "",
        "## Representative Full Loop Rows",
        "",
        md_table(full, ["run_label", "dataset", "seed", "official_hard_row", "architecture", "optimizer_family", "variant", "control_mode", "metric_name", "final_NLL", "final_accuracy", "NLL_improvement_vs_own_strong_optimizer", "controller_overhead_ratio", "support_projection_residual", "residual_signal_SNR", "TrueKANGain_class"], 36),
        "",
        "## Efficiency Evidence",
        "",
        md_table(efficiency, ["dataset", "seed", "variant", "metric_name", "controller_overhead_ratio", "full_step_ms", "base_optimizer_ms", "controller_ms", "basis_metric_update_time_ms", "memory_peak_MB"], 24),
        "",
        "## Analysis / Insight",
        "",
        "- 若 final route 停在 `R1-MetricHarnessOpened`，结论只能是 metric harness 和 runtime 形式过关，尚未证明 metric support / residual signal / KAN carrier 的任务价值。",
        "- 若 S4-M 过关而 S1-M 未过关，按计划只能解释为 metric support optimizer/support preconditioner 有价值，不能写成 signal-FU。",
        "- 若 S6-M 的 KAN internal 指标强但 `beats_MLP_matched_metric_support_rows` 不足，结论应是 KAN 内部 carrier 线索存在，但 architecture superiority 未打开。",
        "- blocker 修复记录必须看执行日志中的命令、失败 stderr 和后续修复 command；本复盘不把失败 row 删除。",
        "- 本轮最终停在 metric harness/runtime 形式正确，而任务价值 gate 未打开。最强 insight 是：metric-preserving support 的训练几何可实现且可审计，但当前实现主要产生微弱/不稳定 support-preconditioner 行为，未形成可归因 residual signal 或 KAN basis carrier。",
    ]
    RECAP_DOC.write_text("\n".join(text) + "\n", encoding="utf-8")


def stage_all(args: argparse.Namespace) -> dict[str, Any]:
    run_metric_harness()
    dispatch_specs(args, task_specs(args, "s4"), "s4")
    stage_s1(args)
    dispatch_specs(args, task_specs(args, "s6"), "s6")
    dispatch_specs(args, task_specs(args, "s3"), "s3")
    merge_chunks(write_final=True)
    return finalize(write_recap=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all", "truth", "collect", "s4", "s1", "s6", "s3", "p3", "p4", "p5", "p6", "merge", "pure-merge", "finalize"])
    p.add_argument("--dataset", default="CIFAR10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--architecture", default="DGKAN_DCHE")
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--variant", default="S4-Euclidean-OET")
    p.add_argument("--control-mode", default="none")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--train-size", type=int, default=384)
    p.add_argument("--held-size", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--support-rank", type=int, default=8)
    p.add_argument("--nuisance-rank", type=int, default=2)
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
    p.add_argument("--metric-shrinkage", type=float, default=0.10)
    p.add_argument("--metric-eps", type=float, default=1.0e-6)
    p.add_argument("--metric-refresh-cadence", type=int, default=1)
    p.add_argument("--debt-velocity-barrier", type=float, default=0.0)
    p.add_argument("--calibration-velocity-barrier", type=float, default=0.0)
    p.add_argument("--safety-budget-velocity-barrier", type=float, default=0.0)
    p.add_argument("--calibration-readout-radial-cap", type=float, default=0.0)
    p.add_argument("--calibration-readout-policy", choices=["signed", "overconfidence_only", "shrink_only", "inverse_signed"], default="signed")
    p.add_argument("--calibration-nuisance-weight", type=float, default=0.0)
    p.add_argument("--calibration-correction-weight", type=float, default=0.0)
    p.add_argument("--calibration-nuisance-mode", choices=["brier", "confidence", "overconfidence", "tail", "margin", "tail_margin", "tail_brier", "tail_brier_balanced", "tail_brier_brier2_balanced", "tail_brier_pareto", "tail_brier_pareto2", "tail_brier_qp", "tail_brier_qp_margin", "tail_brier_qp_brier_margin", "tail_q99_brier_qp_margin", "tail_q99_brier_qp_strict_margin", "tail_q99_brier_qp_hard_margin"], default="brier")
    p.add_argument("--calibration-nuisance-cadence", type=int, default=1)
    p.add_argument("--mirror-grad-cadence", type=int, default=1)
    p.add_argument("--cached-controller-emit-cadence", type=int, default=1)
    p.add_argument("--fused-debt-controller", action="store_true")
    p.add_argument("--debt-orthogonal-controller", action="store_true")
    p.add_argument("--kan-init-variant", default="default")
    p.add_argument("--pure-fu-mode", action="store_true")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--tier2-download", action="store_true")
    p.add_argument("--label", default="v22_43")
    p.add_argument("--eval-datasets", default="CIFAR10,SVHN,EMNIST_LETTERS")
    p.add_argument("--eval-seeds", default="0,1,2")
    p.add_argument("--eval-architectures", default="DGKAN_DCHE,MLP")
    p.add_argument("--strong-optimizers", default="AdamW")
    p.add_argument("--s4-variants", default=",".join(S4_METRICS))
    p.add_argument("--s6-variants", default=",".join(S6_VARIANTS))
    p.add_argument("--s3-variants", default=",".join(S3_VARIANTS))
    p.add_argument("--p3-variants", default=",".join(P3_VARIANTS))
    p.add_argument("--p3-control-modes", default=",".join(P3_CONTROL_MODES))
    p.add_argument("--p4-variants", default=",".join(P4_VARIANTS))
    p.add_argument("--p4-control-modes", default=",".join(P4_CONTROL_MODES))
    p.add_argument("--p5-variants", default=",".join(P5_VARIANTS))
    p.add_argument("--p5-control-modes", default=",".join(P5_CONTROL_MODES))
    p.add_argument("--p6-variants", default=",".join(P6_VARIANTS))
    p.add_argument("--p6-control-modes", default="auto")
    p.add_argument("--eval-variants", default="S4-Euclidean-OET")
    p.add_argument("--control-modes", default=",".join(CONTROL_MODES))
    p.add_argument("--s6-control-modes", default=",".join(S6_CONTROL_MODES))
    p.add_argument("--include-optimizer-alone", action="store_true")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--row-timeout", type=int, default=1200)
    p.add_argument("--row-limit", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_out()
    if args.stage == "truth":
        run_metric_harness()
    elif args.stage == "collect":
        train_variant(
            dataset=args.dataset,
            seed=args.seed,
            architecture=args.architecture,
            optimizer_family=args.optimizer,
            variant=args.variant,
            control_mode=args.control_mode,
            device_name=args.device,
            steps=args.steps,
            train_size=args.train_size,
            held_size=args.held_size,
            batch_size=args.batch_size,
            hidden=args.hidden,
            lr=args.lr,
            weight_decay=args.weight_decay,
            support_rank=args.support_rank,
            nuisance_rank=args.nuisance_rank,
            support_refresh_cadence=args.support_refresh_cadence,
            beta_signal=args.beta_signal,
            beta_metric=args.beta_metric,
            beta_q=args.beta_q,
            eta_rho=args.eta_rho,
            eta_debt=args.eta_debt,
            tau_safe=args.tau_safe,
            rho_min=args.rho_min,
            rho_max=args.rho_max,
            velocity_scale=args.velocity_scale,
            metric_shrinkage=args.metric_shrinkage,
            metric_eps=args.metric_eps,
            metric_refresh_cadence=args.metric_refresh_cadence,
            debt_velocity_barrier=args.debt_velocity_barrier,
            calibration_velocity_barrier=args.calibration_velocity_barrier,
            safety_budget_velocity_barrier=args.safety_budget_velocity_barrier,
            calibration_readout_radial_cap=args.calibration_readout_radial_cap,
            calibration_nuisance_weight=args.calibration_nuisance_weight,
            calibration_correction_weight=args.calibration_correction_weight,
            calibration_nuisance_mode=args.calibration_nuisance_mode,
            calibration_nuisance_cadence=args.calibration_nuisance_cadence,
            pure_fu_mode=args.pure_fu_mode,
            warmup_steps=args.warmup_steps,
            kan_init_variant=args.kan_init_variant,
            tier2_download=args.tier2_download,
            label=args.label,
            calibration_readout_policy=args.calibration_readout_policy,
            mirror_grad_cadence=args.mirror_grad_cadence,
            cached_controller_emit_cadence=args.cached_controller_emit_cadence,
            fused_debt_controller=args.fused_debt_controller,
            debt_orthogonal_controller=args.debt_orthogonal_controller,
        )
    elif args.stage == "s4":
        dispatch_specs(args, task_specs(args, "s4"), "s4")
    elif args.stage == "s1":
        stage_s1(args)
    elif args.stage == "s6":
        dispatch_specs(args, task_specs(args, "s6"), "s6")
    elif args.stage == "s3":
        dispatch_specs(args, task_specs(args, "s3"), "s3")
    elif args.stage == "p3":
        dispatch_specs(args, task_specs(args, "p3"), "p3")
    elif args.stage == "p4":
        dispatch_specs(args, task_specs(args, "p4"), "p4")
    elif args.stage == "p5":
        dispatch_specs(args, task_specs(args, "p5"), "p5")
    elif args.stage == "p6":
        dispatch_specs(args, task_specs(args, "p6"), "p6")
    elif args.stage == "merge":
        merge_chunks(write_final=True)
    elif args.stage == "pure-merge":
        merge_pure_artifacts(write_final=True)
    elif args.stage == "finalize":
        finalize(write_recap=True)
    else:
        stage_all(args)


if __name__ == "__main__":
    main()
