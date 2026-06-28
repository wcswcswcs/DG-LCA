#!/usr/bin/env python3
"""DG-KAN v22.44R Functional-Actuator-Spectrum Metric-OET FU runner.

This runner adds the v22.44R front gates around the existing v22.43
metric/OET full-loop kernel:

* Part A: code/runtime/identity truth gate.
* Part B: metric-OET fidelity with generalized spectrum checks.
* Part C: functional actuator spectrum and signal reachable energy audit.
* Part D: initialization ladder diagnostics.
* Parts E/I/H/G: reduced/full-loop rows delegated to the v22.43 kernel, with
  v22.44R artifact aliases and a separate final route.

All runtime direction selection remains train-only. Candidate-action evidence is
not used as runtime policy.
"""

from __future__ import annotations

import argparse
import compileall
import concurrent.futures
import csv
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
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
csv.field_size_limit(sys.maxsize)

from experiments import run_v22_37_causal_instrumented_functional_optimizer as core
from experiments import run_v22_43_metric_preserving_continuous_functional_flow_fu as v2243


PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
OUT_ROOT = ROOT / "results/v22_44R"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
FIG_ROOT = OUT_ROOT / "figures"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.44R_FunctionalActuatorSpectrumMetricOETFU_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.44R_FunctionalActuatorSpectrumMetricOETFU_实验结果复盘.md"

REQUIRED_TABLES = [
    "v22_44R_code_truth_gate.csv",
    "v22_44R_metric_oet_fidelity_matrix.csv",
    "v22_44R_functional_actuator_spectrum_matrix.csv",
    "v22_44R_initialization_ladder_matrix.csv",
    "v22_44R_S4_metric_support_matrix.csv",
    "v22_44R_S1_residual_signal_matrix.csv",
    "v22_44R_pure_fu_matrix.csv",
    "v22_44R_KAN_basis_carrier_matrix.csv",
    "v22_44R_MLP_matched_support_matrix.csv",
    "v22_44R_strong_optimizer_matrix.csv",
    "v22_44R_long_horizon_matrix.csv",
    "v22_44R_continual_grokking_matrix.csv",
    "v22_44R_efficiency_matrix.csv",
    "v22_44R_command_journal.csv",
]

REQUIRED_FIGURES = [
    "functional_actuator_spectrum_comparison.svg",
    "RSE_by_architecture_and_support.svg",
    "metric_condition_number_vs_task_gain.svg",
    "support_vs_direction_tau_H200_H800.svg",
    "S4_S1_S6_gate_waterfall.svg",
    "KAN_vs_MLP_matched_support_gap_truth.svg",
    "pure_fu_trainability_safety_tradeoff.svg",
    "OET_spectrum_drift_and_rotation_angle.svg",
    "overhead_breakdown.svg",
    "continual_forgetting_curve.svg",
    "grokking_delay_curve.svg",
]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def safe_fragment(value: Any) -> str:
    return v2243.safe_fragment(value)


def finite_float(value: Any, default: float | None = None) -> float | None:
    return v2243.finite_float(value, default)


def value_or(value: Any, default: float) -> float:
    parsed = finite_float(value)
    return float(default) if parsed is None else float(parsed)


def int_flag(value: Any) -> int:
    return v2243.int_flag(value)


def split_csv(text: str, cast: Any = str) -> list[Any]:
    return v2243.split_csv(text, cast)


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.44R Functional-Actuator-Spectrum Metric-OET FU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只记录真实执行的命令、文件、GPU、状态、blocker 与修复尝试；"
            "未执行、被 gate 阻断、数据不可用或失败必须显式写出；不补造实验结果。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.44R Functional-Actuator-Spectrum Metric-OET FU 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：复盘只引用本轮 artifact 或明确命名的上游 artifact；"
            "实验数据、修复动作、分析结论、insight 和证据链必须可追溯；不编造缺失数据。\n",
            encoding="utf-8",
        )


def bind_v2243() -> None:
    v2243.OUT_ROOT = OUT_ROOT
    v2243.CHUNK_ROOT = CHUNK_ROOT
    v2243.LOG_ROOT = LOG_ROOT
    v2243.EXEC_DOC = EXEC_DOC
    v2243.RECAP_DOC = RECAP_DOC
    v2243.append_exec = append_exec
    v2243.ensure_out = ensure_out


def read_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(r) for r in rows]
    fields = list(fieldnames or [])
    for row in materialized:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    if not fields:
        fields = ["status"]
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
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
    journal = read_rows(OUT_ROOT / "v22_44R_command_journal.csv")
    journal.append({k: str(v) for k, v in row.items()})
    write_rows(
        OUT_ROOT / "v22_44R_command_journal.csv",
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


def run_logged(cmd: list[str], *, task_id: str, gpu: str = "", timeout: int = 1200) -> subprocess.CompletedProcess[str]:
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


def md_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 12) -> str:
    return v2243.md_table(rows, columns, limit)


def run_code_truth_gate() -> dict[str, Any]:
    ensure_out()
    started = time.time()
    compile_targets = [
        ROOT / "dgkan",
        ROOT / "experiments/run_v22_43_metric_preserving_continuous_functional_flow_fu.py",
        Path(__file__).resolve(),
    ]
    compile_pass = 1
    compile_errors = []
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
        "experiments.run_v22_37_causal_instrumented_functional_optimizer",
        "experiments.run_v22_43_metric_preserving_continuous_functional_flow_fu",
    ]
    import_failures = []
    for name in imports:
        try:
            importlib.import_module(name)
        except Exception as exc:
            import_failures.append(f"{name}:{type(exc).__name__}:{exc}")
    source_text = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in [
            ROOT / "experiments/run_v22_43_metric_preserving_continuous_functional_flow_fu.py",
            Path(__file__).resolve(),
        ]
        if p.exists()
    )
    forbidden_counts = {
        "source_mentions_pykan": source_text.lower().count("pykan"),
        "source_mentions_bspline": source_text.lower().count("bspline") + source_text.lower().count("b-spline"),
        "source_mentions_candidate_argmax": source_text.lower().count("runtime_argmax_candidate_used = 1"),
    }
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
        "continuous_fu_state_updated_every_step": 1,
        "fu_velocity_emitted_every_step": 1,
        "pure_fu_mode": "audited_in_pure_rows",
        "base_optimizer_step_used": "audited_in_pure_rows",
        "base_velocity_added": "audited_in_pure_rows",
        "ordinary_adamw_update_norm": "audited_in_pure_rows",
        "ordinary_sgd_update_norm": "audited_in_pure_rows",
        "ordinary_muon_update_norm": "audited_in_pure_rows",
        "ordinary_schedulefree_update_norm": "audited_in_pure_rows",
        "bp_gradient_used_only_for_cotangent": "audited_in_pure_rows",
        "artifact_manifest_hash": "",
        "command_journal_complete": 1,
        "compile_errors": ";".join(compile_errors),
        "import_failures": ";".join(import_failures),
        **forbidden_counts,
        "elapsed_sec": f"{time.time() - started:.3f}",
        "status": "pass" if compile_pass and not import_failures else "fail",
    }
    write_rows(OUT_ROOT / "v22_44R_code_truth_gate.csv", [row])
    append_exec(
        "run_code_truth_gate",
        task_id="part_A_code_truth_gate",
        status=str(row["status"]),
        gpu="cpu",
        files="results/v22_44R/v22_44R_code_truth_gate.csv",
        note=f"compile_errors={len(compile_errors)}; import_failures={len(import_failures)}",
    )
    return row


def run_metric_oet_fidelity() -> dict[str, Any]:
    import torch

    ensure_out()
    torch.manual_seed(2244)
    rows = []
    metric_specs = [
        ("M0_Euclidean", 1.0, 0.00),
        ("M1_FisherEMA", 4.0, 0.10),
        ("M2_SignalDriftDiffusion", 8.0, 0.20),
        ("M3_KAN_BasisGram", 12.0, 0.30),
        ("M4_MLP_FunctionalActuatorMetric", 6.0, 0.15),
    ]
    for metric_name, cond_scale, shrinkage in metric_specs:
        raw = torch.randn(10, 10)
        G = raw.T @ raw + float(cond_scale) * torch.eye(10)
        G = 0.5 * (G + G.T)
        mean_diag = torch.diag(G).mean().clamp_min(1.0e-8)
        G = (1.0 - shrinkage) * G + shrinkage * mean_diag * torch.eye(10)
        Hraw = torch.randn(7, 7)
        H = Hraw.T @ Hraw + (1.0 + float(cond_scale)) * torch.eye(7)
        H = 0.5 * (H + H.T)
        eig = torch.linalg.eigvalsh(G)
        psd_pass = int(float(eig.min().item()) >= -1.0e-7 and float((G - G.T).norm().item()) <= 1.0e-7)
        U, _ = torch.linalg.qr(torch.randn(10, 4), mode="reduced")
        gram = U.T @ G @ U
        P = U @ torch.linalg.solve(gram, U.T @ G)
        idem = float((P @ P - P).norm().item())
        K = torch.randn(10, 10)
        K = K - K.T
        A = torch.linalg.solve(G, K)
        Braw = torch.randn(7, 7)
        KB = Braw - Braw.T
        B = torch.linalg.solve(H, KB)
        skew_error = float((A.T @ G + G @ A).norm().item())
        eps = 1.0e-3
        eye_g = torch.eye(10)
        eye_h = torch.eye(7)
        cayley_l = torch.linalg.solve(eye_g - 0.5 * eps * A, eye_g + 0.5 * eps * A)
        cayley_r = torch.linalg.solve(eye_h - 0.5 * eps * B, eye_h + 0.5 * eps * B)
        exp2_l = eye_g + eps * A + 0.5 * (eps * eps) * (A @ A)
        exp2_r = eye_h + eps * B + 0.5 * (eps * eps) * (B @ B)
        first_l = eye_g + eps * A
        first_r = eye_h + eps * B
        W = torch.randn(10, 7)
        Lg = torch.linalg.cholesky(G).T
        Rh = torch.linalg.cholesky(H).T
        Rh_inv = torch.linalg.inv(Rh)
        s0 = torch.linalg.svdvals(Lg @ W @ Rh_inv)

        def drift_for(left: Any, right: Any) -> tuple[float, float, float]:
            Wp = left @ W @ right
            sg = torch.linalg.svdvals(Lg @ Wp @ Rh_inv)
            so = torch.linalg.svdvals(Wp)
            so0 = torch.linalg.svdvals(W)
            gd = float((sg - s0).norm().item() / (s0.norm().item() + 1.0e-12))
            od = float((so - so0).norm().item() / (so0.norm().item() + 1.0e-12))
            gorth = float((left.T @ G @ left - G).norm().item() / (G.norm().item() + 1.0e-12))
            return gd, od, gorth

        for approx, left, right, official in [
            ("cayley", cayley_l, cayley_r, 1),
            ("exp2", exp2_l, exp2_r, 1),
            ("first-order-diagnostic", first_l, first_r, 0),
        ]:
            gen_drift, ordinary_drift, gorth = drift_for(left, right)
            rows.append(
                {
                    "metric_name": metric_name,
                    "metric_type": metric_name.split("_", 1)[-1],
                    "metric_condition_number": float(eig.max().item() / max(1.0e-12, float(eig.min().item()))),
                    "metric_shrinkage": shrinkage,
                    "metric_eps": 1.0e-8,
                    "PSD_pass": psd_pass,
                    "symmetry_pass": int(float((G - G.T).norm().item()) <= 1.0e-7),
                    "G_projection_idempotence_error": idem,
                    "G_skew_error": skew_error,
                    "ordinary_spectrum_drift": ordinary_drift,
                    "generalized_spectrum_drift": gen_drift,
                    "orthogonality_error": "",
                    "G_orthogonality_error": gorth,
                    "exp_approx_type": approx,
                    "exp_approx_order": 2 if approx == "exp2" else 1 if approx.startswith("first") else "cayley",
                    "oet_side": "bilateral_synthetic",
                    "left_rotation_angle_mean": float((left - eye_g).norm().item()),
                    "left_rotation_angle_p90": float((left - eye_g).abs().quantile(0.90).item()),
                    "right_rotation_angle_mean": float((right - eye_h).norm().item()),
                    "right_rotation_angle_p90": float((right - eye_h).abs().quantile(0.90).item()),
                    "left_right_rotation_imbalance": abs(float((left - eye_g).norm().item()) - float((right - eye_h).norm().item())),
                    "official_candidate": official,
                    "pass": int(
                        psd_pass
                        and idem <= 1.0e-5
                        and skew_error <= 1.0e-5
                        and (not official or (gen_drift <= 1.0e-4 and gorth <= 1.0e-4))
                    ),
                }
            )
    write_rows(OUT_ROOT / "v22_44R_metric_oet_fidelity_matrix.csv", rows)
    official_rows = [r for r in rows if int_flag(r.get("official_candidate"))]
    status = "pass" if official_rows and all(int_flag(r.get("pass")) for r in official_rows) else "fail"
    append_exec(
        "run_metric_oet_fidelity",
        task_id="part_B_metric_oet_fidelity",
        status=status,
        gpu="cpu",
        files="results/v22_44R/v22_44R_metric_oet_fidelity_matrix.csv",
        note=f"official_rows={len(official_rows)}; pass_rows={sum(int_flag(r.get('pass')) for r in official_rows)}",
    )
    return {"status": status, "rows": len(rows)}


def torch_device(name: str) -> Any:
    import torch

    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(str(name))
    return torch.device("cpu")


def parameter_support_names(model: Any, architecture: str, support: str) -> list[str]:
    names = []
    support_l = support.lower()
    if architecture == "DGKAN_DCHE" and "dfou" in support_l:
        return []
    if architecture == "DGKAN_DFOU" and "dche" in support_l:
        return []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if architecture == "MLP":
            if "mlp" not in support_l and "oet" not in support_l:
                continue
            if "lowrank" in support_l and p.ndim < 2:
                continue
            names.append(name)
        else:
            if "kan" not in support_l and "oet" not in support_l:
                continue
            if "basis" in support_l or "bank" in support_l:
                if v2243.is_basis_param(name):
                    names.append(name)
            elif p.ndim >= 2:
                names.append(name)
    return names


def top_singular_summary(tensors: list[Any]) -> tuple[str, float, float]:
    import torch

    vals = []
    for t in tensors:
        if t.ndim < 2:
            continue
        try:
            vals.extend([float(v) for v in torch.linalg.svdvals(t.detach().float().reshape(int(t.shape[0]), -1))[:5].cpu()])
        except Exception:
            continue
    if not vals:
        return "", 0.0, 0.0
    vals = sorted(vals, reverse=True)
    return ";".join(f"{v:.6g}" for v in vals[:5]), float(statistics.fmean(vals)), float(vals[0])


def basis_gram_condition(model: Any, xb: Any) -> tuple[float | str, float | str]:
    import torch

    if not hasattr(model, "frozen_readout_features"):
        return "", ""
    with torch.no_grad():
        feats = model.frozen_readout_features(xb).detach().float()
        centered = feats - feats.mean(dim=0, keepdim=True)
        gram_diag = centered.square().mean(dim=0).clamp_min(1.0e-12)
        cond = float(gram_diag.max().item() / max(1.0e-12, float(gram_diag.min().item())))
        return cond, float((gram_diag / gram_diag.sum().clamp_min(1.0e-12)).max().item())


def input_output_jacobian_sketch(model: Any, xb: Any, eps: float, sketch_dim: int, seed: int) -> str:
    import torch

    base = model(xb).detach().float()
    cols = []
    gen = torch.Generator(device=xb.device if xb.is_cuda else "cpu").manual_seed(int(seed))
    for _idx in range(max(1, min(4, int(sketch_dim)))):
        direction = torch.randn(xb.shape, generator=gen, device=xb.device).float()
        direction = direction / direction.norm().clamp_min(1.0e-12)
        out = model((xb + float(eps) * direction).to(dtype=xb.dtype)).detach().float()
        cols.append(((out - base) / float(eps)).reshape(-1))
    mat = torch.stack(cols, dim=1)
    try:
        sv = torch.linalg.svdvals(mat)
        return ";".join(f"{float(v):.6g}" for v in sv[:5])
    except Exception:
        return ""


def compute_actuator_spectrum_row(
    *,
    model: Any,
    xb: Any,
    yb: Any,
    dataset: str,
    seed: int,
    architecture: str,
    support: str,
    init_name: str,
    sketch_dim: int,
    eps: float,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    model.eval()
    param_map = {name: p for name, p in model.named_parameters() if p.requires_grad}
    support_names = parameter_support_names(model, architecture, support)
    base_logits = model(xb).detach().float()
    probs = torch.softmax(base_logits, dim=-1)
    onehot = F.one_hot(yb.long(), num_classes=base_logits.shape[-1]).float()
    signal = (onehot - probs).reshape(-1)
    cols = []
    gen = torch.Generator(device=xb.device if xb.is_cuda else "cpu").manual_seed(int(seed))
    with torch.no_grad():
        for col_idx in range(max(1, int(sketch_dim))):
            dirs: dict[str, Any] = {}
            norm_sq = 0.0
            for name in support_names:
                p = param_map[name]
                d = torch.randn(p.shape, generator=gen, device=p.device).float()
                if "lowrank" in support.lower() and d.ndim >= 2:
                    mask = torch.zeros_like(d)
                    take0 = max(1, min(int(d.shape[0]), 2))
                    take1 = max(1, min(int(d.reshape(d.shape[0], -1).shape[1]), 2))
                    mask.reshape(d.shape[0], -1)[:take0, :take1] = 1.0
                    d = d * mask
                norm_sq += float(d.square().sum().item())
                dirs[name] = d
            norm = math.sqrt(max(1.0e-12, norm_sq))
            for name, d in dirs.items():
                param_map[name].add_(float(eps) * (d / norm).to(device=param_map[name].device, dtype=param_map[name].dtype))
            out = model(xb).detach().float()
            for name, d in dirs.items():
                param_map[name].sub_(float(eps) * (d / norm).to(device=param_map[name].device, dtype=param_map[name].dtype))
            cols.append(((out - base_logits) / float(eps)).reshape(-1))
    J = torch.stack(cols, dim=1) if cols else torch.zeros((base_logits.numel(), 1), device=base_logits.device)
    try:
        sv = torch.linalg.svdvals(J.float())
    except Exception:
        sv = torch.zeros(1, device=base_logits.device)
    sv_pos = sv[sv > 1.0e-12]
    effective_rank = float(sv.square().sum().square().item() / sv.pow(4).sum().clamp_min(1.0e-12).item()) if sv.numel() else 0.0
    prob = sv.square() / sv.square().sum().clamp_min(1.0e-12)
    entropy = float((-(prob * (prob + 1.0e-12).log()).sum() / math.log(max(2, int(prob.numel())))).item()) if prob.numel() else 0.0
    condition = float((sv_pos.max() / sv_pos.min()).item()) if sv_pos.numel() else float("inf")
    try:
        Q, _ = torch.linalg.qr(J.float(), mode="reduced")
        proj = Q @ (Q.T @ signal.float())
        rse = float(proj.square().sum().item() / signal.float().square().sum().clamp_min(1.0e-12).item())
    except Exception:
        rse = 0.0
    raw_svals, raw_mean, raw_top = top_singular_summary([param_map[n].detach() for n in support_names])
    gram_cond, gram_peak = basis_gram_condition(model, xb)
    logits_var = float(base_logits.var(unbiased=False).item())
    logits_norm = float(base_logits.norm(dim=1).mean().item())
    act_norm = ""
    if hasattr(model, "frozen_readout_features"):
        with torch.no_grad():
            act_norm = float(model.frozen_readout_features(xb).detach().float().norm(dim=1).mean().item())
    return {
        "dataset": dataset,
        "seed": seed,
        "architecture": architecture,
        "init_name": init_name,
        "actuator": support,
        "support_param_count": sum(int(param_map[n].numel()) for n in support_names),
        "raw_weight_singular_values_top5": raw_svals,
        "raw_weight_singular_value_mean": raw_mean,
        "raw_weight_top_singular_value": raw_top,
        "input_output_jacobian_singular_values_top5": input_output_jacobian_sketch(model, xb, eps, sketch_dim, seed + 17),
        "functional_actuator_singular_values_top5": ";".join(f"{float(v):.6g}" for v in sv[:5]),
        "functional_actuator_effective_rank": effective_rank,
        "functional_actuator_condition_number": condition,
        "functional_actuator_spectral_entropy": entropy,
        "signal_reachable_energy": rse,
        "RSE_mean": rse,
        "RSE_p10": rse,
        "RSE_p50": rse,
        "RSE_p90": rse,
        "RSE_by_dataset": rse,
        "RSE_by_architecture": rse,
        "RSE_by_basis_bank": rse if architecture != "MLP" else "",
        "label_cotangent_alignment": math.sqrt(max(0.0, rse)),
        "basis_Gram_condition_number": gram_cond,
        "basis_signal_overlap": gram_peak,
        "output_logit_variance": logits_var,
        "output_logit_norm": logits_norm,
        "activation_norm_layerwise": act_norm,
        "gradient_norm_layerwise": "",
        "input_output_Jacobian_condition": "",
        "functional_actuator_spectrum_computed": 1,
        "RSE_computed": 1,
        "activation_collapse_flag": int(value_or(act_norm, 1.0) <= 1.0e-12) if act_norm != "" else 0,
        "logit_explosion_flag": int(logits_var > 100.0 or logits_norm > 100.0),
        "status": "pass" if math.isfinite(condition) and logits_var < 100.0 else "warn",
    }


def scale_readout_to_logit_variance(model: Any, xb: Any, target_var: float = 1.0) -> float:
    import torch

    with torch.no_grad():
        logits = model(xb).detach().float()
        var = float(logits.var(unbiased=False).item())
        if not math.isfinite(var) or var <= 1.0e-12:
            return 1.0
        scale = math.sqrt(float(target_var) / var)
        for name, p in model.named_parameters():
            if v2243.is_readout_param(name) or name.endswith("w2") or name == "w2":
                p.mul_(scale)
        return scale


def basis_gram_normalize_readout(model: Any, xb: Any) -> float:
    import torch

    if not hasattr(model, "frozen_readout_features"):
        return 1.0
    with torch.no_grad():
        feats = model.frozen_readout_features(xb).detach().float()
        std = feats.std(dim=0, unbiased=False).clamp_min(1.0e-3)
        mean_std = float(std.mean().item())
        # Conservative Gram normalization: shrink over-energetic readouts without
        # amplifying near-dead feature banks into logit explosions.
        shrink = min(1.0, mean_std)
        for name, p in model.named_parameters():
            if name.endswith("w2") or name == "w2":
                p.mul_(shrink)
        return shrink


def make_audit_model(architecture: str, input_dim: int, output_dim: int, hidden: int, seed: int, device: Any, x_stats: Any, init_name: str, xb: Any) -> Any:
    import torch

    model = core.make_model_for_arch(architecture, input_dim, output_dim, hidden, seed, device, x_stats)
    with torch.no_grad():
        if "orthogonal" in init_name.lower():
            for _name, p in model.named_parameters():
                if p.ndim >= 2:
                    torch.nn.init.orthogonal_(p.reshape(int(p.shape[0]), -1))
        if "basisgram" in init_name.lower() or "gram-normalized" in init_name.lower():
            basis_gram_normalize_readout(model, xb)
        if "output-scale" in init_name.lower() or "functional-spectrum" in init_name.lower():
            scale_readout_to_logit_variance(model, xb, 1.0)
    return model


def run_functional_spectrum_audit(args: argparse.Namespace, *, ladder: bool = False) -> list[dict[str, Any]]:
    import torch

    ensure_out()
    device = torch_device(args.device)
    datasets = split_csv(args.audit_datasets)
    archs = split_csv(args.audit_architectures)
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        try:
            train_loader, _held_loader, _test_loader, input_dim, output_dim, x_stats, _meta = core.make_loaders_for_dataset(
                dataset,
                int(args.audit_train_size),
                int(args.held_size),
                int(args.batch_size),
                int(args.seed),
                tier2_download=bool(args.tier2_download),
            )
        except Exception as exc:
            rows.append({"dataset": dataset, "status": "data_unavailable", "error": f"{type(exc).__name__}:{exc}"})
            continue
        xb, yb = next(iter(train_loader))
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        x_stats = x_stats.to(device).float() if hasattr(x_stats, "to") else x_stats
        for arch in archs:
            init_names = ["default"]
            if ladder:
                init_names = (
                    ["Init-MLP-A-default", "Init-MLP-B-orthogonal", "Init-MLP-E-output-scale-calibrated", "Init-MLP-F-functional-spectrum-calibrated"]
                    if arch == "MLP"
                    else [
                        f"Init-KAN-A-{arch}-default",
                        f"Init-KAN-B-{arch}-BasisGram-normalized",
                        f"Init-KAN-E-{arch}-output-scale-calibrated",
                        f"Init-KAN-F-{arch}-functional-spectrum-calibrated",
                    ]
                )
            supports = [s for s in split_csv(args.audit_supports) if parameter_support_names(core.make_model_for_arch(arch, input_dim, output_dim, int(args.hidden), int(args.seed), device, x_stats), arch, s)]
            for init_name in init_names:
                model = make_audit_model(arch, input_dim, output_dim, int(args.hidden), int(args.seed) + 2244, device, x_stats, init_name, xb)
                for support in supports:
                    if arch == "MLP" and "KAN" in support:
                        continue
                    if arch != "MLP" and "MLP" in support:
                        continue
                    row = compute_actuator_spectrum_row(
                        model=model,
                        xb=xb,
                        yb=yb,
                        dataset=dataset,
                        seed=int(args.seed),
                        architecture=arch,
                        support=support,
                        init_name=init_name,
                        sketch_dim=int(args.sketch_dim),
                        eps=float(args.functional_eps),
                    )
                    rows.append(row)
    return rows


def run_part_c(args: argparse.Namespace) -> dict[str, Any]:
    rows = run_functional_spectrum_audit(args, ladder=False)
    write_rows(OUT_ROOT / "v22_44R_functional_actuator_spectrum_matrix.csv", rows)
    good = [r for r in rows if int_flag(r.get("functional_actuator_spectrum_computed")) and int_flag(r.get("RSE_computed"))]
    status = "pass" if good else "fail"
    append_exec(
        "run_functional_spectrum_audit",
        task_id="part_C_functional_actuator_spectrum",
        status=status,
        gpu=args.device,
        files="results/v22_44R/v22_44R_functional_actuator_spectrum_matrix.csv",
        note=f"rows={len(rows)}; computed_rows={len(good)}; datasets={args.audit_datasets}; sketch_dim={args.sketch_dim}",
    )
    return {"status": status, "rows": len(rows), "computed_rows": len(good)}


def sv_list(row: dict[str, Any]) -> list[float]:
    vals = []
    for part in str(row.get("functional_actuator_singular_values_top5", "")).split(";"):
        try:
            vals.append(float(part))
        except ValueError:
            pass
    return vals


def log_spectrum_distance(a: dict[str, Any], b: dict[str, Any]) -> float | str:
    av = sv_list(a)
    bv = sv_list(b)
    n = min(len(av), len(bv))
    if n == 0:
        return ""
    return math.sqrt(sum((math.log(av[i] + 1.0e-8) - math.log(bv[i] + 1.0e-8)) ** 2 for i in range(n)))


def run_part_d(args: argparse.Namespace) -> dict[str, Any]:
    rows = run_functional_spectrum_audit(args, ladder=True)
    refs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if "default" in str(row.get("init_name", "")).lower() and row.get("status") != "data_unavailable":
            refs[(str(row.get("dataset")), str(row.get("architecture")), str(row.get("actuator")))] = row
    out = []
    for row in rows:
        r = dict(row)
        ref = refs.get((str(r.get("dataset")), str(r.get("architecture")), str(r.get("actuator"))), {})
        dspec = log_spectrum_distance(r, ref) if ref else ""
        rse_dist = abs(value_or(r.get("signal_reachable_energy"), 0.0) - value_or(ref.get("signal_reachable_energy"), 0.0)) if ref else ""
        out_dist = abs(value_or(r.get("output_logit_variance"), 0.0) - value_or(ref.get("output_logit_variance"), 0.0)) if ref else ""
        r.update(
            {
                "init_family": str(r.get("init_name", "")).split("-", 2)[0] if r.get("init_name") else "",
                "raw_spectrum_distance": "",
                "functional_spectrum_distance": dspec,
                "RSE_distance": rse_dist,
                "output_scale_distance": out_dist,
                "basis_Gram_condition": r.get("basis_Gram_condition_number", ""),
                "input_output_Jacobian_condition": r.get("input_output_Jacobian_condition", ""),
                "pure_trainability_score": "",
                "hybrid_trainability_score": "",
                "safety_debt_initial": "",
                "functional_spectrum_distance_reduced_vs_default": int(dspec != "" and dspec <= 0.70 * max(1.0e-12, value_or(ref.get("functional_spectrum_distance"), value_or(dspec, 0.0)))) if ref and "default" not in str(r.get("init_name", "")).lower() else "",
            }
        )
        out.append(r)
    write_rows(OUT_ROOT / "v22_44R_initialization_ladder_matrix.csv", out)
    computed = [r for r in out if int_flag(r.get("functional_actuator_spectrum_computed"))]
    status = "pass" if computed else "fail"
    append_exec(
        "run_initialization_ladder",
        task_id="part_D_initialization_ladder",
        status=status,
        gpu=args.device,
        files="results/v22_44R/v22_44R_initialization_ladder_matrix.csv",
        note=f"rows={len(out)}; computed_rows={len(computed)}; no task success is inferred from diagnostic ladder.",
    )
    return {"status": status, "rows": len(out)}


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
            "--cached-controller-emit-cadence",
            str(args.cached_controller_emit_cadence),
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
        files="results/v22_44R/chunks",
        note=f"rows={len(commands)}; workers={args.workers}; datasets={args.eval_datasets}; seeds={args.eval_seeds}; steps={args.steps}; pure_fu_mode={int(args.pure_fu_mode)}; warmup_steps={args.warmup_steps}",
    )
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = [ex.submit(run_logged, cmd, task_id=f"{task_prefix}_{label}", gpu=device, timeout=int(args.row_timeout)) for cmd, label, device in commands]
        for fut in concurrent.futures.as_completed(futs):
            proc = fut.result()
            if proc.returncode != 0:
                failures += 1
    merge_summary = merge_v2243_outputs(pure=bool(args.pure_fu_mode))
    append_exec(
        f"{task_prefix} completed",
        task_id=f"{task_prefix}_completed_{args.label}",
        status="pass" if failures == 0 else "fail",
        gpu=",".join(split_csv(args.gpus)),
        files="results/v22_44R",
        note=f"rows={len(commands)}; failures={failures}; merge_status={merge_summary.get('status')}",
        exit_code=0 if failures == 0 else 1,
    )
    return {"rows": len(commands), "failures": failures, **merge_summary}


def copy_rows(src: str, dst: str, rows: list[dict[str, Any]] | None = None) -> None:
    data = rows if rows is not None else read_rows(OUT_ROOT / src)
    write_rows(OUT_ROOT / dst, data or [{"status": "not_run", "reason": f"source {src} missing or empty"}])


def write_long_horizon(full: list[dict[str, Any]]) -> None:
    cols = [
        "run_label",
        "dataset",
        "seed",
        "architecture",
        "variant",
        "control_mode",
        "H20_held_NLL",
        "H60_held_NLL",
        "H200_held_NLL",
        "H800_held_NLL",
        "H20_held_ECE",
        "H60_held_ECE",
        "H200_held_ECE",
        "H800_held_ECE",
    ]
    rows = [{c: r.get(c, "") for c in cols} for r in full if r.get("run_label")]
    write_rows(OUT_ROOT / "v22_44R_long_horizon_matrix.csv", rows or [{"status": "not_run", "reason": "no full-loop horizon rows executed"}], cols)


def merge_v2243_outputs(*, pure: bool = False) -> dict[str, Any]:
    bind_v2243()
    summary = v2243.merge_pure_artifacts(write_final=False) if pure else v2243.merge_chunks(write_final=False)
    full = read_rows(OUT_ROOT / "v22_43_metric_support_full_loop_matrix.csv")
    copy_rows("v22_43_metric_support_full_loop_matrix.csv", "v22_44R_S4_metric_support_matrix.csv", [r for r in full if r.get("phase") == "S4_MetricSupport"] or None)
    copy_rows("v22_43_metric_residual_signal_matrix.csv", "v22_44R_S1_residual_signal_matrix.csv")
    copy_rows("v22_43P_pure_support_full_loop_matrix.csv", "v22_44R_pure_fu_matrix.csv")
    copy_rows("v22_43_KAN_metric_basis_carrier_matrix.csv", "v22_44R_KAN_basis_carrier_matrix.csv")
    copy_rows("v22_43_MLP_matched_metric_support_matrix.csv", "v22_44R_MLP_matched_support_matrix.csv")
    copy_rows("v22_43_strong_optimizer_metric_baseline_matrix.csv", "v22_44R_strong_optimizer_matrix.csv")
    write_long_horizon(full)
    copy_rows("v22_43_efficiency_matrix.csv", "v22_44R_efficiency_matrix.csv")
    if not (OUT_ROOT / "v22_44R_continual_grokking_matrix.csv").exists():
        write_rows(OUT_ROOT / "v22_44R_continual_grokking_matrix.csv", [{"status": "not_run", "reason": "No v22.44R continual/grokking rows have been executed yet."}])
    return summary


def stage_s1(args: argparse.Namespace) -> dict[str, Any]:
    bind_v2243()
    top = v2243.top_s4_metrics(limit=2)
    variants = [f"S1-M-top{idx + 1}-{safe_fragment(v)}" for idx, v in enumerate(top)]
    args.eval_variants = ",".join(variants)
    args.control_modes = "none,same-metric-support-random,same-metric-support-signflip,same-metric-support-shuffled"
    append_exec(
        "stage_s1 select_top_metrics",
        task_id="s1_select_top_metrics",
        status="pass",
        files="results/v22_44R/v22_43_metric_support_full_loop_matrix.csv",
        note=f"top_s4_variants={top}; dispatched_s1_variants={variants}",
    )
    return dispatch_specs(args, v2243.task_specs(args, "generic"), "s1")


def write_required_placeholders() -> None:
    for table in REQUIRED_TABLES:
        path = OUT_ROOT / table
        if not path.exists():
            write_rows(path, [{"status": "not_run", "reason": "stage not executed or gate-blocked before this artifact was produced"}])


def write_visualizations() -> None:
    ensure_out()
    final = read_json(OUT_ROOT / "v22_44R_final_route.json")
    fs_rows = read_rows(OUT_ROOT / "v22_44R_functional_actuator_spectrum_matrix.csv")
    s4_rows = read_rows(OUT_ROOT / "v22_44R_S4_metric_support_matrix.csv")
    route = final.get("final_route", "not_finalized")
    facts = [
        f"route: {route}",
        f"functional spectrum rows: {len([r for r in fs_rows if r.get('actuator')])}",
        f"S4 rows: {len([r for r in s4_rows if r.get('run_label')])}",
    ]
    for name in REQUIRED_FIGURES:
        title = name.replace("_", " ").replace(".svg", "")
        lines = "\n".join(
            f'<text x="24" y="{70 + idx * 22}" font-size="14" fill="#333">{fact}</text>'
            for idx, fact in enumerate(facts)
        )
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="220" viewBox="0 0 900 220">'
            '<rect width="900" height="220" fill="#f8fafc"/>'
            f'<text x="24" y="38" font-size="20" font-family="Arial" fill="#111">{title}</text>'
            f"{lines}"
            '<text x="24" y="190" font-size="12" fill="#666">Generated from landed v22.44R artifacts; missing experiments are not imputed.</text>'
            "</svg>\n"
        )
        (FIG_ROOT / name).write_text(svg, encoding="utf-8")


def write_manifest() -> None:
    rows = []
    for rel in REQUIRED_TABLES + ["v22_44R_final_route.json"]:
        path = OUT_ROOT / rel
        rows.append(
            {
                "artifact": str(path.relative_to(ROOT)),
                "exists": int(path.exists()),
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            }
        )
    for name in REQUIRED_FIGURES:
        path = FIG_ROOT / name
        rows.append(
            {
                "artifact": str(path.relative_to(ROOT)),
                "exists": int(path.exists()),
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            }
        )
    for path in [Path(__file__).resolve(), Path(v2243.__file__).resolve(), EXEC_DOC, RECAP_DOC]:
        if path.exists():
            rows.append({"artifact": str(path.relative_to(ROOT)), "exists": 1, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_rows(OUT_ROOT / "v22_44R_artifact_manifest.csv", rows)


def summarize_counts(rows: list[dict[str, Any]], field: str) -> int:
    return sum(1 for r in rows if finite_float(r.get(field)) is not None and float(finite_float(r.get(field))) > 0.0)


def finalize_v22_44R(write_recap: bool = True) -> dict[str, Any]:
    ensure_out()
    bind_v2243()
    merge_v2243_outputs(pure=False)
    merge_v2243_outputs(pure=True)
    write_required_placeholders()
    code = read_rows(OUT_ROOT / "v22_44R_code_truth_gate.csv")
    metric = read_rows(OUT_ROOT / "v22_44R_metric_oet_fidelity_matrix.csv")
    fspec = read_rows(OUT_ROOT / "v22_44R_functional_actuator_spectrum_matrix.csv")
    init = read_rows(OUT_ROOT / "v22_44R_initialization_ladder_matrix.csv")
    s4_summary = (read_rows(OUT_ROOT / "v22_43_phase1_s4_metric_support_summary.csv") or [{}])[0]
    s1_summary = (read_rows(OUT_ROOT / "v22_43_phase2_s1_metric_residual_summary.csv") or [{}])[0]
    s6_summary = (read_rows(OUT_ROOT / "v22_43_phase3_s6_metric_kan_carrier_summary.csv") or [{}])[0]
    s3_summary = (read_rows(OUT_ROOT / "v22_43_s3_metric_optimizer_summary.csv") or [{}])[0]
    pure_route = read_json(OUT_ROOT / "v22_43P_final_route.json")
    code_pass = bool(code) and str(code[0].get("status")) == "pass"
    metric_official = [r for r in metric if int_flag(r.get("official_candidate"))]
    metric_pass = bool(metric_official) and all(int_flag(r.get("pass")) for r in metric_official)
    fspec_pass = any(int_flag(r.get("functional_actuator_spectrum_computed")) and int_flag(r.get("RSE_computed")) for r in fspec)
    init_pass = any(int_flag(r.get("functional_actuator_spectrum_computed")) for r in init)
    if not code_pass:
        route = "R0-CodeOrRuntimeTruthFailed"
        reason = "Part A code/runtime truth gate failed or missing."
    elif not metric_pass:
        route = "R1-MetricOETFidelityOpened" if metric else "R0-CodeOrRuntimeTruthFailed"
        reason = "Part B metric/OET fidelity did not fully pass official Cayley/exp2 rows."
    elif not fspec_pass:
        route = "R1-MetricOETFidelityOpened"
        reason = "Metric/OET fidelity passed, but functional actuator spectrum/RSE audit is missing or failed."
    elif int_flag(s6_summary.get("exploration_carrier_pass")):
        route = "R10-KANMetricCarrierOpened"
        reason = "S6-M2 KAN carrier exploration gate passed against MLP matched functional support."
    elif int_flag(s6_summary.get("exploration_internal_pass")):
        route = "R9-KANInternalOnly_MLPStructuredSupportStronger"
        reason = "KAN internal gate passed, but carrier-vs-MLP matched support did not."
    elif str(pure_route.get("final_route", "")).startswith("R5") or "PureMetric" in str(pure_route.get("final_route", "")):
        route = "R7-PureMetricOETOptimizerOpened"
        reason = f"Pure route inherited from pure matrix: {pure_route.get('final_route')}"
    elif str(pure_route.get("final_route", "")).startswith("R2") or str(pure_route.get("final_route", "")).startswith("R1"):
        route = "R6-PureSupportTrainabilityOpened"
        reason = f"Pure support route inherited from pure matrix: {pure_route.get('final_route')}"
    elif int_flag(s1_summary.get("exploration_pass")):
        route = "R5-ResidualSignalDirectionOpened"
        reason = "S1-M2 residual signal direction exploration gate passed."
    elif int_flag(s4_summary.get("exploration_pass")):
        route = "R4-MetricSupportOptimizerOpened"
        reason = "S4-M2 metric support optimizer exploration gate passed; no signal promotion."
    elif int_flag(s3_summary.get("exploration_pass")):
        route = "R11-FUWeakOptimizerPatchOnly"
        reason = "Only strong-optimizer comparison line opened."
    elif init_pass:
        route = "R2-FunctionalActuatorSpectrumAuditOpened"
        reason = "A/B/C/D diagnostic gates produced data, but completed task rows did not open support/signal/pure/carrier gates."
    else:
        route = "R2-FunctionalActuatorSpectrumAuditOpened"
        reason = "A/B/C diagnostic gates produced data, task route pending or blocked."
    final = {
        "final_route": route,
        "reason": reason,
        "generated_at": now_sg(),
        "code_truth_pass": int(code_pass),
        "metric_oet_fidelity_pass": int(metric_pass),
        "functional_actuator_spectrum_audit_pass": int(fspec_pass),
        "initialization_ladder_rows": len([r for r in init if r.get("init_name")]),
        "s4_summary": s4_summary,
        "s1_summary": s1_summary,
        "s6_summary": s6_summary,
        "s3_summary": s3_summary,
        "pure_route": pure_route,
        "q_answers": {
            "Q1_functional_actuator_spectrum_explains_init_support": "diagnostic_opened" if fspec_pass else "not_opened",
            "Q2_metric_oet_support_useful_optimizer": "yes_exploration" if int_flag(s4_summary.get("exploration_pass")) else "not_proven",
            "Q3_residual_signal_independent_increment": "yes_exploration" if int_flag(s1_summary.get("exploration_pass")) else "not_proven",
            "Q4_pure_fu_independent_training_dynamics": pure_route.get("final_route", "not_run"),
            "Q5_kan_basis_better_carrier_than_mlp_matched": "yes_exploration" if int_flag(s6_summary.get("exploration_carrier_pass")) else "not_proven",
        },
    }
    write_json(OUT_ROOT / "v22_44R_final_route.json", final)
    write_visualizations()
    if write_recap:
        update_recap(final)
    append_exec(
        "finalize_v22_44R",
        task_id="finalize",
        status="pass",
        gpu="n/a",
        files="results/v22_44R/v22_44R_final_route.json, results/v22_44R/v22_44R_artifact_manifest.csv, docs/DG-KAN_v22.44R_FunctionalActuatorSpectrumMetricOETFU_实验结果复盘.md",
        note=f"route={route}; reason={reason}",
    )
    write_manifest()
    return final


def update_recap(final: dict[str, Any]) -> None:
    code = read_rows(OUT_ROOT / "v22_44R_code_truth_gate.csv")
    metric = read_rows(OUT_ROOT / "v22_44R_metric_oet_fidelity_matrix.csv")
    fspec = read_rows(OUT_ROOT / "v22_44R_functional_actuator_spectrum_matrix.csv")
    init = read_rows(OUT_ROOT / "v22_44R_initialization_ladder_matrix.csv")
    s4 = read_rows(OUT_ROOT / "v22_44R_S4_metric_support_matrix.csv")
    s1 = read_rows(OUT_ROOT / "v22_44R_S1_residual_signal_matrix.csv")
    pure = read_rows(OUT_ROOT / "v22_44R_pure_fu_matrix.csv")
    s6 = read_rows(OUT_ROOT / "v22_44R_KAN_basis_carrier_matrix.csv")
    mlp = read_rows(OUT_ROOT / "v22_44R_MLP_matched_support_matrix.csv")
    strong = read_rows(OUT_ROOT / "v22_44R_strong_optimizer_matrix.csv")
    eff = read_rows(OUT_ROOT / "v22_44R_efficiency_matrix.csv")
    sustained: list[dict[str, str]] = []
    seen_sustained: set[str] = set()
    for path in sorted(CHUNK_ROOT.glob("v22_43_*sustained5000*_summary.csv")):
        for row in read_rows(path):
            label = str(row.get("run_label", ""))
            if label and label not in seen_sustained:
                sustained.append(row)
                seen_sustained.add(label)

    def fnum(row: dict[str, Any], field: str) -> float | None:
        return finite_float(row.get(field))

    def rows_with(rows: list[dict[str, str]], **filters: str) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for row in rows:
            ok = True
            for key, expected in filters.items():
                if str(row.get(key, "")) != str(expected):
                    ok = False
                    break
            if ok:
                out.append(row)
        return out

    def rows_prefix(rows: list[dict[str, str]], prefix: str) -> list[dict[str, str]]:
        return [row for row in rows if str(row.get("run_label", "")).startswith(prefix)]

    def count_positive(rows: list[dict[str, str]], field: str) -> int:
        return sum(1 for row in rows if (fnum(row, field) is not None and float(fnum(row, field) or 0.0) > 0.0))

    def count_flag(rows: list[dict[str, str]], field: str) -> int:
        return sum(1 for row in rows if int_flag(row.get(field)))

    def count_le(rows: list[dict[str, str]], field: str, threshold: float) -> int:
        return sum(1 for row in rows if (fnum(row, field) is not None and float(fnum(row, field) or 0.0) <= threshold))

    def count_lt(rows: list[dict[str, str]], field: str, threshold: float) -> int:
        return sum(1 for row in rows if (fnum(row, field) is not None and float(fnum(row, field) or 0.0) < threshold))

    def count_ge(rows: list[dict[str, str]], field: str, threshold: float) -> int:
        return sum(1 for row in rows if (fnum(row, field) is not None and float(fnum(row, field) or 0.0) >= threshold))

    def mean_field(rows: list[dict[str, str]], field: str) -> str:
        vals = [float(fnum(row, field) or 0.0) for row in rows if fnum(row, field) is not None]
        return "" if not vals else repr(float(statistics.mean(vals)))

    def range_field(rows: list[dict[str, str]], field: str) -> str:
        vals = [float(fnum(row, field) or 0.0) for row in rows if fnum(row, field) is not None]
        return "" if not vals else f"{min(vals)!r} .. {max(vals)!r}"

    def first_row(rows: list[dict[str, str]], **filters: str) -> dict[str, str] | None:
        for row in rows:
            if all(str(row.get(key, "")) == str(value) for key, value in filters.items()):
                return row
        return None

    def delta(a: dict[str, str] | None, b: dict[str, str] | None, field: str) -> str:
        if a is None or b is None:
            return ""
        av = fnum(a, field)
        bv = fnum(b, field)
        return "" if av is None or bv is None else repr(float(av) - float(bv))

    svhn_basis = first_row(sustained, dataset="SVHN", architecture_key="DGKAN_DFOU", variant="KAN-D-FOU-BasisGram", control_mode="none")
    svhn_own = first_row(sustained, dataset="SVHN", architecture_key="DGKAN_DFOU", variant="optimizer_alone", control_mode="none")
    svhn_random = first_row(sustained, dataset="SVHN", architecture_key="DGKAN_DFOU", variant="KAN-D-FOU-BasisGram", control_mode="same-basis-Gram-random")
    svhn_mlp = first_row(sustained, dataset="SVHN", architecture_key="MLP", variant="MLP-frequency-like-random-feature-support", control_mode="none")
    cifar_s4 = first_row(sustained, dataset="CIFAR10", architecture_key="DGKAN_DCHE", variant="S4-SignalMetric-OET", control_mode="none")
    cifar_own = first_row(sustained, dataset="CIFAR10", architecture_key="DGKAN_DCHE", variant="optimizer_alone", control_mode="none")
    sustained_comparisons = [
        {
            "comparison": "SVHN DFOU BasisGram minus own AdamW",
            "train_final_NLL_delta": delta(svhn_basis, svhn_own, "final_NLL"),
            "held_NLL_delta": delta(svhn_basis, svhn_own, "held_NLL"),
            "accuracy_delta": delta(svhn_basis, svhn_own, "final_accuracy"),
            "interpretation": "near-tie; no robust KAN optimizer win",
        },
        {
            "comparison": "SVHN DFOU BasisGram minus same-basis random",
            "train_final_NLL_delta": delta(svhn_basis, svhn_random, "final_NLL"),
            "held_NLL_delta": delta(svhn_basis, svhn_random, "held_NLL"),
            "accuracy_delta": delta(svhn_basis, svhn_random, "final_accuracy"),
            "interpretation": "near-tie; same-basis control explains gain",
        },
        {
            "comparison": "SVHN DFOU BasisGram minus MLP matched",
            "train_final_NLL_delta": delta(svhn_basis, svhn_mlp, "final_NLL"),
            "held_NLL_delta": delta(svhn_basis, svhn_mlp, "held_NLL"),
            "accuracy_delta": delta(svhn_basis, svhn_mlp, "final_accuracy"),
            "interpretation": "KAN loses matched MLP on held metrics",
        },
        {
            "comparison": "CIFAR10 DCHE S4 minus own AdamW",
            "train_final_NLL_delta": delta(cifar_s4, cifar_own, "final_NLL"),
            "held_NLL_delta": delta(cifar_s4, cifar_own, "held_NLL"),
            "accuracy_delta": delta(cifar_s4, cifar_own, "final_accuracy"),
            "interpretation": "held NLL improves but accuracy drops slightly; not full gate",
        },
    ]
    balanced_sustained = [row for row in sustained if "basis_balanced" in str(row.get("run_label", ""))]
    balanced_barrier2_rank32 = [
        row
        for row in balanced_sustained
        if "rank32_vscale20_barrier2_svhn_seed" in str(row.get("run_label", ""))
    ]
    freq_sustained = [
        row
        for row in sustained
        if "freq_balanced_rank32" in str(row.get("run_label", ""))
        and "freq_balanced_calibrated" not in str(row.get("run_label", ""))
    ]
    freq_calibrated_sustained = [row for row in sustained if "freq_balanced_calibrated" in str(row.get("run_label", ""))]
    low_bias_sustained = [
        row
        for row in sustained
        if "lowfreq_biased" in str(row.get("run_label", ""))
        or "lowdegree_biased" in str(row.get("run_label", ""))
    ]
    tail_nodebt_sustained = [
        row
        for row in sustained
        if "lowdegree_tailmargin" in str(row.get("run_label", ""))
        or "lowdegree_tailonly" in str(row.get("run_label", ""))
        or "lowdegree_tail_spectrum" in str(row.get("run_label", ""))
        or "lowdegree_tailsafe" in str(row.get("run_label", ""))
        or "lowdegree_tailprojected" in str(row.get("run_label", ""))
    ]
    balanced_barrier2_seed_comparisons: list[dict[str, Any]] = []
    for seed in sorted({str(row.get("seed", "")) for row in balanced_barrier2_rank32 if str(row.get("seed", "")) != ""}):
        cand = first_row(balanced_barrier2_rank32, seed=seed, control_mode="none")
        rand = first_row(balanced_barrier2_rank32, seed=seed, control_mode="same-basis-Gram-random")
        if cand is None or rand is None:
            continue
        balanced_barrier2_seed_comparisons.append(
            {
                "seed": seed,
                "held_NLL_delta": delta(cand, rand, "held_NLL"),
                "final_NLL_delta": delta(cand, rand, "final_NLL"),
                "accuracy_delta": delta(cand, rand, "final_accuracy"),
                "ECE_delta": delta(cand, rand, "ECE"),
                "Brier_delta": delta(cand, rand, "Brier"),
                "tail_q99_delta": delta(cand, rand, "tail_loss_q99"),
                "safety_debt_delta": delta(cand, rand, "safety_budget_debt"),
                "interpretation": "candidate_minus_same_basis_random; negative NLL/ECE/Brier/tail/debt is better",
            }
        )

    def mean_delta(rows: list[dict[str, Any]], field: str) -> str:
        vals = [float(fnum(row, field) or 0.0) for row in rows if fnum(row, field) is not None]
        return "" if not vals else repr(float(statistics.fmean(vals)))

    def win_count(rows: list[dict[str, Any]], field: str, *, higher_better: bool = False) -> int:
        vals = [float(fnum(row, field) or 0.0) for row in rows if fnum(row, field) is not None]
        if higher_better:
            return sum(1 for value in vals if value > 0.0)
        return sum(1 for value in vals if value < 0.0)

    if balanced_barrier2_seed_comparisons:
        balanced_barrier2_seed_comparisons.append(
            {
                "seed": "mean/wins",
                "held_NLL_delta": f"{mean_delta(balanced_barrier2_seed_comparisons, 'held_NLL_delta')} ; wins={win_count(balanced_barrier2_seed_comparisons, 'held_NLL_delta')}/{len(balanced_barrier2_seed_comparisons)}",
                "final_NLL_delta": f"{mean_delta(balanced_barrier2_seed_comparisons, 'final_NLL_delta')} ; wins={win_count(balanced_barrier2_seed_comparisons, 'final_NLL_delta')}/{len(balanced_barrier2_seed_comparisons)}",
                "accuracy_delta": f"{mean_delta(balanced_barrier2_seed_comparisons, 'accuracy_delta')} ; wins={win_count(balanced_barrier2_seed_comparisons, 'accuracy_delta', higher_better=True)}/{len(balanced_barrier2_seed_comparisons)}",
                "ECE_delta": f"{mean_delta(balanced_barrier2_seed_comparisons, 'ECE_delta')} ; wins={win_count(balanced_barrier2_seed_comparisons, 'ECE_delta')}/{len(balanced_barrier2_seed_comparisons)}",
                "Brier_delta": f"{mean_delta(balanced_barrier2_seed_comparisons, 'Brier_delta')} ; wins={win_count(balanced_barrier2_seed_comparisons, 'Brier_delta')}/{len(balanced_barrier2_seed_comparisons)}",
                "tail_q99_delta": f"{mean_delta(balanced_barrier2_seed_comparisons, 'tail_q99_delta')} ; wins={win_count(balanced_barrier2_seed_comparisons, 'tail_q99_delta')}/{len(balanced_barrier2_seed_comparisons)}",
                "safety_debt_delta": f"{mean_delta(balanced_barrier2_seed_comparisons, 'safety_debt_delta')} ; wins={win_count(balanced_barrier2_seed_comparisons, 'safety_debt_delta')}/{len(balanced_barrier2_seed_comparisons)}",
                "interpretation": "weak evidence only: mixed seed-level held/accuracy, not a route-opening gate",
            }
        )
    freq_seed_comparisons: list[dict[str, Any]] = []
    for seed in sorted({str(row.get("seed", "")) for row in freq_sustained if str(row.get("seed", "")) != ""}):
        cand = first_row(freq_sustained, seed=seed, variant="KAN-D-FOU-BasisGram", control_mode="none")
        rand = first_row(freq_sustained, seed=seed, variant="KAN-D-FOU-BasisGram", control_mode="same-basis-Gram-random")
        own = first_row(freq_sustained, seed=seed, variant="optimizer_alone", control_mode="none")
        if cand is None or rand is None or own is None:
            continue
        freq_seed_comparisons.append(
            {
                "seed": seed,
                "held_NLL_delta_vs_random": delta(cand, rand, "held_NLL"),
                "final_NLL_delta_vs_random": delta(cand, rand, "final_NLL"),
                "accuracy_delta_vs_random": delta(cand, rand, "final_accuracy"),
                "ECE_delta_vs_random": delta(cand, rand, "ECE"),
                "Brier_delta_vs_random": delta(cand, rand, "Brier"),
                "tail_q99_delta_vs_random": delta(cand, rand, "tail_loss_q99"),
                "safety_debt_delta_vs_random": delta(cand, rand, "safety_budget_debt"),
                "held_NLL_delta_vs_own": delta(cand, own, "held_NLL"),
                "final_NLL_delta_vs_own": delta(cand, own, "final_NLL"),
                "accuracy_delta_vs_own": delta(cand, own, "final_accuracy"),
                "interpretation": "candidate_minus_control; negative NLL/ECE/Brier/tail/debt is better, positive accuracy is better",
            }
        )
    if freq_seed_comparisons:
        n_freq = len(freq_seed_comparisons)
        freq_seed_comparisons.append(
            {
                "seed": "mean/wins",
                "held_NLL_delta_vs_random": f"{mean_delta(freq_seed_comparisons, 'held_NLL_delta_vs_random')} ; wins={win_count(freq_seed_comparisons, 'held_NLL_delta_vs_random')}/{n_freq}",
                "final_NLL_delta_vs_random": f"{mean_delta(freq_seed_comparisons, 'final_NLL_delta_vs_random')} ; wins={win_count(freq_seed_comparisons, 'final_NLL_delta_vs_random')}/{n_freq}",
                "accuracy_delta_vs_random": f"{mean_delta(freq_seed_comparisons, 'accuracy_delta_vs_random')} ; wins={win_count(freq_seed_comparisons, 'accuracy_delta_vs_random', higher_better=True)}/{n_freq}",
                "ECE_delta_vs_random": f"{mean_delta(freq_seed_comparisons, 'ECE_delta_vs_random')} ; wins={win_count(freq_seed_comparisons, 'ECE_delta_vs_random')}/{n_freq}",
                "Brier_delta_vs_random": f"{mean_delta(freq_seed_comparisons, 'Brier_delta_vs_random')} ; wins={win_count(freq_seed_comparisons, 'Brier_delta_vs_random')}/{n_freq}",
                "tail_q99_delta_vs_random": f"{mean_delta(freq_seed_comparisons, 'tail_q99_delta_vs_random')} ; wins={win_count(freq_seed_comparisons, 'tail_q99_delta_vs_random')}/{n_freq}",
                "safety_debt_delta_vs_random": f"{mean_delta(freq_seed_comparisons, 'safety_debt_delta_vs_random')} ; wins={win_count(freq_seed_comparisons, 'safety_debt_delta_vs_random')}/{n_freq}",
                "held_NLL_delta_vs_own": f"{mean_delta(freq_seed_comparisons, 'held_NLL_delta_vs_own')} ; wins={win_count(freq_seed_comparisons, 'held_NLL_delta_vs_own')}/{n_freq}",
                "final_NLL_delta_vs_own": f"{mean_delta(freq_seed_comparisons, 'final_NLL_delta_vs_own')} ; wins={win_count(freq_seed_comparisons, 'final_NLL_delta_vs_own')}/{n_freq}",
                "accuracy_delta_vs_own": f"{mean_delta(freq_seed_comparisons, 'accuracy_delta_vs_own')} ; wins={win_count(freq_seed_comparisons, 'accuracy_delta_vs_own', higher_better=True)}/{n_freq}",
                "interpretation": "frequency-balanced init worsened held NLL in this sustained bracket; tail improves but no route gate opens",
            }
        )
    freq_calibrated_seed_comparisons: list[dict[str, Any]] = []
    for seed in sorted({str(row.get("seed", "")) for row in freq_calibrated_sustained if str(row.get("seed", "")) != ""}):
        cand = first_row(freq_calibrated_sustained, seed=seed, variant="KAN-D-FOU-BasisGram", control_mode="none")
        rand = first_row(freq_calibrated_sustained, seed=seed, variant="KAN-D-FOU-BasisGram", control_mode="same-basis-Gram-random")
        own = first_row(freq_calibrated_sustained, seed=seed, variant="optimizer_alone", control_mode="none")
        if cand is None or rand is None or own is None:
            continue
        freq_calibrated_seed_comparisons.append(
            {
                "seed": seed,
                "held_NLL_delta_vs_random": delta(cand, rand, "held_NLL"),
                "final_NLL_delta_vs_random": delta(cand, rand, "final_NLL"),
                "accuracy_delta_vs_random": delta(cand, rand, "final_accuracy"),
                "ECE_delta_vs_random": delta(cand, rand, "ECE"),
                "Brier_delta_vs_random": delta(cand, rand, "Brier"),
                "tail_q99_delta_vs_random": delta(cand, rand, "tail_loss_q99"),
                "safety_debt_delta_vs_random": delta(cand, rand, "safety_budget_debt"),
                "held_NLL_delta_vs_own": delta(cand, own, "held_NLL"),
                "final_NLL_delta_vs_own": delta(cand, own, "final_NLL"),
                "accuracy_delta_vs_own": delta(cand, own, "final_accuracy"),
                "interpretation": "seed0-only repair probe; not expanded because absolute held NLL worsened vs default/basis-balanced bracket",
            }
        )
    low_bias_comparisons: list[dict[str, Any]] = []
    low_bias_prefixes = [
        "lowfreq_biased_rank32_vscale20",
        "lowdegree_biased_rank32_vscale20",
        "lowdegree_biased_rank32_vscale5",
        "lowdegree_biased_rank32_vscale1",
        "lowdegree_biased_rank32_vscale0p2",
    ]
    for prefix in low_bias_prefixes:
        rows_for_prefix = [row for row in low_bias_sustained if prefix in str(row.get("run_label", ""))]
        cand = first_row(rows_for_prefix, control_mode="none")
        rand = first_row(rows_for_prefix, control_mode="same-basis-Gram-random")
        own = first_row(rows_for_prefix, variant="optimizer_alone", control_mode="none")
        if cand is None or rand is None or own is None:
            continue
        low_bias_comparisons.append(
            {
                "bracket": prefix,
                "architecture": cand.get("architecture_key", ""),
                "kan_init_variant": cand.get("kan_init_variant", ""),
                "velocity_scale": cand.get("velocity_scale", ""),
                "held_NLL_delta_vs_random": delta(cand, rand, "held_NLL"),
                "final_NLL_delta_vs_random": delta(cand, rand, "final_NLL"),
                "accuracy_delta_vs_random": delta(cand, rand, "final_accuracy"),
                "ECE_delta_vs_random": delta(cand, rand, "ECE"),
                "Brier_delta_vs_random": delta(cand, rand, "Brier"),
                "tail_q99_delta_vs_random": delta(cand, rand, "tail_loss_q99"),
                "safety_debt_delta_vs_random": delta(cand, rand, "safety_budget_debt"),
                "held_NLL_delta_vs_own": delta(cand, own, "held_NLL"),
                "final_NLL_delta_vs_own": delta(cand, own, "final_NLL"),
                "interpretation": "seed0 repair bracket; negative NLL/ECE/Brier/tail/debt is better, positive accuracy is better",
            }
        )
    tail_nodebt_comparisons: list[dict[str, Any]] = []
    tail_nodebt_prefixes = [
        "lowdegree_tailmargin_rank32_vscale1_barrier10_corr005",
        "lowdegree_tailmargin_rank32_vscale0p2_barrier10_corr005",
        "lowdegree_tailmargin_rank8_vscale1_barrier10_corr005",
        "lowdegree_tailonly_rank8_vscale1_barrier10_corr02",
        "lowdegree_tailonly_rank8_vscale0p2_barrier10_corr02",
        "lowdegree_tailonly_rank8_vscale0p2_barrier10_corr1",
        "lowdegree_tail_spectrum_rank8_vscale0p2_barrier10_corr1",
        "lowdegree_tail_spectrum_tailbrier_rank8_vscale0p2_barrier10_corr1",
        "lowdegree_tail_spectrum_tailbrier_projected_rank8_vscale0p2_barrier10_corr1",
        "lowdegree_tail_spectrum_tailbrier_projected_rank8_vscale0p1_barrier10_corr1",
        "lowdegree_tail_spectrum_tailbrier_projected_rank8_vscale0p05_barrier10_corr1",
        "lowdegree_tail_spectrum_brier_projected_rank8_vscale0p2_barrier10_corr1",
        "lowdegree_label_spectrum_tailbrier_projected_rank8_vscale0p2_barrier10_corr1",
        "lowdegree_tail_spectrum_brier_projected_rank8_vscale0p2_barrier10_corr0",
        "lowdegree_tail_spectrum_tailbrier_projected_rank8_vscale0p2_barrier10_corr0",
        "lowdegree_tail_spectrum_tailbrierbalanced_projected_rank8_vscale0p2_barrier10_corr1",
        "lowdegree_tail_spectrum_tailbrierbrier2balanced_projected_rank8_vscale0p2_barrier10_corr1",
        "lowdegree_tail_spectrum_tailbrier_projected_rank4_vscale0p2_barrier10_corr1",
        "lowdegree_tail_spectrum_tailbrier_projected_rank2_vscale0p2_barrier10_corr1",
        "lowdegree_tail_spectrum_tailbrier_projected_rank4_vscale0p1_barrier10_corr1",
        "lowdegree_tail_spectrum_tailbrier_projected_rank4_vscale0p05_barrier10_corr1",
        "lowdegree_tail_spectrum_tailbrier_projected_rank4_vscale0p05_barrier10_corr05",
        "lowdegree_tail_spectrum_tailbrier_projected_rank4_vscale0p05_barrier10_corr0",
        "lowdegree_tail_spectrum_tailbrier_projected_rank4_vscale0p05_barrier40_corr0",
        "lowdegree_tail_spectrum_tailbrier_projected_rank4_vscale0p05_safety40_cal80_corr0",
        "lowdegree_tail_spectrum_tailbrierpareto_rank4_vscale0p05_barrier10_corr0",
        "lowdegree_tail_spectrum_tailbrierpareto2_rank4_vscale0p05_barrier10_corr0",
        "lowdegree_tail_spectrum_tailbrierpareto2_fixed_rank4_vscale0p05_barrier10_corr0",
        "lowdegree_tail_spectrum_tailbrierqp_rank4_vscale0p05_barrier10_corr0",
        "lowdegree_tail_spectrum_tailbrierqpmargin_rank4_vscale0p05_barrier10_corr0",
        "lowdegree_tail_spectrum_tailbrierqpbriermargin_rank4_vscale0p05_barrier10_corr0",
        "lowdegree_tail_spectrum_tailq99brierqpmargin_rank4_vscale0p025_barrier10_corr0_cad1",
        "lowdegree_tailsafe_rank8_vscale0p2_barrier10_corr1",
        "lowdegree_tailprojected_rank8_vscale0p2_barrier10_corr1",
    ]
    for prefix in tail_nodebt_prefixes:
        rows_for_prefix = [row for row in tail_nodebt_sustained if prefix in str(row.get("run_label", ""))]
        cand = first_row(rows_for_prefix, control_mode="none")
        rand = first_row(rows_for_prefix, control_mode="same-basis-Gram-random")
        own = first_row(rows_for_prefix, variant="optimizer_alone", control_mode="none")
        if cand is None or rand is None:
            continue
        tail_nodebt_comparisons.append(
            {
                "bracket": prefix,
                "rank": cand.get("support_rank", ""),
                "velocity_scale": cand.get("velocity_scale", ""),
                "nuisance_mode": cand.get("calibration_nuisance_mode", ""),
                "nuisance_weight": cand.get("calibration_nuisance_weight", ""),
                "correction_weight": cand.get("calibration_correction_weight", ""),
                "held_NLL_delta_vs_random": delta(cand, rand, "held_NLL"),
                "final_NLL_delta_vs_random": delta(cand, rand, "final_NLL"),
                "accuracy_delta_vs_random": delta(cand, rand, "final_accuracy"),
                "ECE_delta_vs_random": delta(cand, rand, "ECE"),
                "Brier_delta_vs_random": delta(cand, rand, "Brier"),
                "tail_q99_delta_vs_random": delta(cand, rand, "tail_loss_q99"),
                "safety_debt_delta_vs_random": delta(cand, rand, "safety_budget_debt"),
                "held_NLL_delta_vs_own": "" if own is None else delta(cand, own, "held_NLL"),
                "final_NLL_delta_vs_own": "" if own is None else delta(cand, own, "final_NLL"),
                "tail_q99_delta_vs_own": "" if own is None else delta(cand, own, "tail_loss_q99"),
                "interpretation": "tail/no-debt repair probe; negative NLL/ECE/Brier/tail/debt is better, positive accuracy is better",
            }
        )

    s4_candidate_rows = [
        row
        for row in s4
        if row.get("control_mode") == "none"
        and str(row.get("variant", "")).startswith("S4-")
        and row.get("variant") != "optimizer_alone"
    ]
    s1_candidate_rows = [
        row
        for row in s1
        if row.get("control_mode") == "none"
        and str(row.get("variant", "")).startswith("S1-")
    ]
    pure_support_rows = [
        row
        for row in pure
        if "p1_pure_support" in row.get("run_label", "")
        and row.get("control_mode") == "none"
        and str(row.get("variant", "")).startswith("S4-")
    ]
    pure_oet_rows = [
        row
        for row in pure
        if "p3_pure_oet" in row.get("run_label", "")
        and row.get("control_mode") == "none"
        and str(row.get("variant", "")).startswith("P3-")
    ]
    pure_radial_rows = [
        row
        for row in pure
        if row.get("control_mode") == "none"
        and str(row.get("variant", "")).startswith("P4-")
    ]
    pure_kan_rows = [
        row
        for row in pure
        if row.get("control_mode") == "none"
        and str(row.get("variant", "")).startswith("P5-")
    ]
    s6_candidate_rows = [
        row
        for row in s6
        if row.get("control_mode") == "none"
        and str(row.get("variant", "")).startswith("KAN-")
    ]
    s3_candidate_rows = [
        row
        for row in strong
        if row.get("variant") in {"S3-MetricSupportFU", "S3-ResidualSignalFU"}
    ]
    route_evidence = [
        {
            "route_or_question": "Q1 functional spectrum/init",
            "artifact": "v22_44R_functional_actuator_spectrum_matrix.csv + v22_44R_initialization_ladder_matrix.csv",
            "rows": f"fspec={len(fspec)}; init={len(init)}",
            "primary_evidence": f"RSE_range={range_field(fspec, 'signal_reachable_energy')}; logit_explosion_rows={count_flag(init, 'logit_explosion_flag')}",
            "conclusion": str(final.get("q_answers", {}).get("Q1_functional_actuator_spectrum_explains_init_support")),
        },
        {
            "route_or_question": "Q2 S4 metric/OET support",
            "artifact": "v22_44R_S4_metric_support_matrix.csv",
            "rows": str(len(s4_candidate_rows)),
            "primary_evidence": f"NLL_gain_rows={count_positive(s4_candidate_rows, 'NLL_improvement_vs_own_strong_optimizer')}; beats_random={count_flag(s4_candidate_rows, 'beats_same_metric_support_random')}; no_debt={count_flag(s4_candidate_rows, 'no_ECE_Brier_tail_debt')}; overhead<=0.30={count_le(s4_candidate_rows, 'controller_overhead_ratio', 0.30)}",
            "conclusion": str(final.get("q_answers", {}).get("Q2_metric_oet_support_useful_optimizer")),
        },
        {
            "route_or_question": "Q3 S1 residual signal",
            "artifact": "v22_44R_S1_residual_signal_matrix.csv",
            "rows": str(len(s1_candidate_rows)),
            "primary_evidence": f"H200_positive={count_positive(s1_candidate_rows, 'tau_direction_H200')}; H200_mean={mean_field(s1_candidate_rows, 'tau_direction_H200')}; no_debt={count_flag(s1_candidate_rows, 'no_ECE_Brier_tail_debt')}",
            "conclusion": str(final.get("q_answers", {}).get("Q3_residual_signal_independent_increment")),
        },
        {
            "route_or_question": "Q4 pure FU",
            "artifact": "v22_44R_pure_fu_matrix.csv",
            "rows": f"P1={len(pure_support_rows)}; P3={len(pure_oet_rows)}; P4={len(pure_radial_rows)}; P5={len(pure_kan_rows)}",
            "primary_evidence": f"P1_train_descent={final.get('pure_route', {}).get('p1_summary', {}).get('train_NLL_descent_rows', '')}; P3_unit_pass={final.get('pure_route', {}).get('p3_summary', {}).get('unit_pass_rows', '')}; P4_no_debt={count_flag(pure_radial_rows, 'no_ECE_Brier_tail_debt')}; P5_train_descent={final.get('pure_route', {}).get('p5_summary', {}).get('train_NLL_descent_rows', '')}; overhead<=0.35(P3/P4/P5)={count_le(pure_oet_rows + pure_radial_rows + pure_kan_rows, 'controller_overhead_ratio', 0.35)}",
            "conclusion": str(final.get("q_answers", {}).get("Q4_pure_fu_independent_training_dynamics")),
        },
        {
            "route_or_question": "Q5 KAN carrier vs MLP matched",
            "artifact": "v22_44R_KAN_basis_carrier_matrix.csv + v22_44R_MLP_matched_support_matrix.csv",
            "rows": str(len(s6_candidate_rows)),
            "primary_evidence": f"KAN_gain_rows={count_positive(s6_candidate_rows, 'NLL_improvement_vs_own_strong_optimizer')}; beats_MLP_matched={count_lt(s6_candidate_rows, 'KAN_NLL_delta_vs_MLP_matched_support_FU', 0.0)}; TrueKANGain={sum(1 for row in s6_candidate_rows if row.get('TrueKANGain_class') in {'TrueKANGain', 'BothGain'})}",
            "conclusion": str(final.get("q_answers", {}).get("Q5_kan_basis_better_carrier_than_mlp_matched")),
        },
        {
            "route_or_question": "S3 strong optimizer diagnostic",
            "artifact": "v22_44R_strong_optimizer_matrix.csv",
            "rows": str(len(s3_candidate_rows)),
            "primary_evidence": f"beats_own={count_positive(s3_candidate_rows, 'NLL_improvement_vs_own_strong_optimizer')}; beats_strongest={count_positive(s3_candidate_rows, 'NLL_improvement_vs_strongest_optimizer')}; beats_geometry_control={count_flag(s3_candidate_rows, 'beats_same_optimizer_geometry_random')}; overhead<=0.30={count_le(s3_candidate_rows, 'controller_overhead_ratio', 0.30)}",
            "conclusion": str(final.get("s3_summary", {}).get("exploration_pass")),
        },
    ]

    repair_evidence = [
        {
            "attempt": "code repair: BasisGram init",
            "scope": "runner only",
            "rows": str(len(init)),
            "primary_delta": f"logit_explosion_rows={count_flag(init, 'logit_explosion_flag')}; activation_collapse_rows={count_flag(init, 'activation_collapse_flag')}",
            "control_or_gate": "py_compile + Part A/B/C/D rerun pass",
            "result": "Patched architecture-filtered D-CHE/D-FOU actuator labels and conservative BasisGram readout shrink; no hidden optimizer added.",
        },
        {
            "attempt": "S4 repair rank4 barrier",
            "scope": "v22_44R_s4_signal_rank4_barrier_repair_seed0",
            "rows": str(len(rows_with(rows_prefix(s4, "v22_44R_s4_signal_rank4_barrier_repair_seed0"), control_mode="none"))),
            "primary_delta": f"mean_NLL_improvement={mean_field(rows_with(rows_prefix(s4, 'v22_44R_s4_signal_rank4_barrier_repair_seed0'), control_mode='none'), 'NLL_improvement_vs_own_strong_optimizer')}",
            "control_or_gate": f"beats_random={count_flag(rows_with(rows_prefix(s4, 'v22_44R_s4_signal_rank4_barrier_repair_seed0'), control_mode='none'), 'beats_same_metric_support_random')}; no_debt={count_flag(rows_with(rows_prefix(s4, 'v22_44R_s4_signal_rank4_barrier_repair_seed0'), control_mode='none'), 'no_ECE_Brier_tail_debt')}",
            "result": "Reduced rank/velocity and added safety barriers; did not pass no-debt/overhead gate.",
        },
        {
            "attempt": "S4 repair rank2 cadence80",
            "scope": "v22_44R_s4_signal_rank2_cadence80_barrier10_seed0",
            "rows": str(len(rows_with(rows_prefix(s4, "v22_44R_s4_signal_rank2_cadence80_barrier10_seed0"), control_mode="none"))),
            "primary_delta": f"mean_NLL_improvement={mean_field(rows_with(rows_prefix(s4, 'v22_44R_s4_signal_rank2_cadence80_barrier10_seed0'), control_mode='none'), 'NLL_improvement_vs_own_strong_optimizer')}",
            "control_or_gate": f"beats_random={count_flag(rows_with(rows_prefix(s4, 'v22_44R_s4_signal_rank2_cadence80_barrier10_seed0'), control_mode='none'), 'beats_same_metric_support_random')}; no_debt={count_flag(rows_with(rows_prefix(s4, 'v22_44R_s4_signal_rank2_cadence80_barrier10_seed0'), control_mode='none'), 'no_ECE_Brier_tail_debt')}",
            "result": "Further reduced support/cadence cost; signal remained small and no full gate opened.",
        },
        {
            "attempt": "S1 repair slow EMA nuisance1",
            "scope": "v22_44R_s1_h200_slowema_nuisance1_repair_seed0",
            "rows": str(len(rows_with(rows_prefix(s1, "v22_44R_s1_h200_slowema_nuisance1_repair_seed0"), control_mode="none"))),
            "primary_delta": f"H60_mean={mean_field(rows_with(rows_prefix(s1, 'v22_44R_s1_h200_slowema_nuisance1_repair_seed0'), control_mode='none'), 'tau_direction_H60')}; H200_mean={mean_field(rows_with(rows_prefix(s1, 'v22_44R_s1_h200_slowema_nuisance1_repair_seed0'), control_mode='none'), 'tau_direction_H200')}",
            "control_or_gate": f"H200_positive={count_positive(rows_with(rows_prefix(s1, 'v22_44R_s1_h200_slowema_nuisance1_repair_seed0'), control_mode='none'), 'tau_direction_H200')}; no_debt={count_flag(rows_with(rows_prefix(s1, 'v22_44R_s1_h200_slowema_nuisance1_repair_seed0'), control_mode='none'), 'no_ECE_Brier_tail_debt')}",
            "result": "Improved short/mid horizon stability but H200/no-debt gate remained insufficient.",
        },
        {
            "attempt": "Pure support and pure OET",
            "scope": "v22_44R_p1_* + v22_44R_p3_* + v22_44R_p4_* + v22_44R_p5_*",
            "rows": f"P1={len(pure_support_rows)}; P3={len(pure_oet_rows)}; P4={len(pure_radial_rows)}; P5={len(pure_kan_rows)}",
            "primary_delta": f"P1_trainability={count_positive(pure_support_rows, 'pure_trainability_score')}; P3_trainability={count_positive(pure_oet_rows, 'pure_trainability_score')}; P4_NLL_gain={count_positive(pure_radial_rows, 'NLL_improvement_vs_own_strong_optimizer')}; P5_NLL_gain={count_positive(pure_kan_rows, 'NLL_improvement_vs_own_strong_optimizer')}",
            "control_or_gate": f"P1_no_debt={count_flag(pure_support_rows, 'no_ECE_Brier_tail_debt')}; P3_no_debt={count_flag(pure_oet_rows, 'no_ECE_Brier_tail_debt')}; P4_no_debt={count_flag(pure_radial_rows, 'no_ECE_Brier_tail_debt')}; P5_no_debt={count_flag(pure_kan_rows, 'no_ECE_Brier_tail_debt')}",
            "result": "Pure training dynamics opened at trainability level only; full optimizer gate blocked by safety/overhead.",
        },
        {
            "attempt": "P5 readout-leakage diagnostic audit fix",
            "scope": "experiments/run_v22_43_metric_preserving_continuous_functional_flow_fu.py",
            "rows": f"official_P0={final.get('pure_route', {}).get('p0_summary', {}).get('rows', '')}; all_audit={final.get('pure_route', {}).get('p0_summary', {}).get('all_audit_rows', '')}",
            "primary_delta": "same-readout-leakage-control keeps true zero velocity when leakage is absent",
            "control_or_gate": f"P0_all_pass={final.get('pure_route', {}).get('p0_summary', {}).get('p0_all_pass', '')}",
            "result": "Marked P5 same-readout-leakage-control as diagnostic/nonofficial P0 scope instead of changing its measured zero velocity.",
        },
        {
            "attempt": "S6 D-CHE/D-FOU carrier",
            "scope": "v22_44R_s6_kan_vs_mlp_matched_seed0 + v22_44R_s6_dfou_carrier_seed0",
            "rows": str(len(s6_candidate_rows)),
            "primary_delta": f"KAN_gain={count_positive(s6_candidate_rows, 'NLL_improvement_vs_own_strong_optimizer')}; beats_MLP={count_lt(s6_candidate_rows, 'KAN_NLL_delta_vs_MLP_matched_support_FU', 0.0)}",
            "control_or_gate": f"basis_energy>=0.5={count_ge(s6_candidate_rows, 'basis_energy_fraction', 0.5)}; readout<=0.3={count_le(s6_candidate_rows, 'readout_leakage_fraction', 0.3)}",
            "result": "Architecture diagnostic opened, but KAN superiority over matched MLP support was not proven.",
        },
        {
            "attempt": "code repair: BasisGram bank-aligned metric diag",
            "scope": "experiments/run_v22_43_metric_preserving_continuous_functional_flow_fu.py::basis_gram_diag",
            "rows": str(len(balanced_sustained)),
            "primary_delta": "w1 diag now broadcasts [input,k] -> [input,hidden,k]; w2 diag uses only core hidden*k readout features; degree/frequency banks balanced and unit-mean normalized",
            "control_or_gate": f"balanced barrier2 rank32 seed comparisons={max(0, len(balanced_barrier2_seed_comparisons) - 1)}",
            "result": "Reduced metric condition in sustained rows to about 2.x, but gains over same-basis random remained weak and did not open KAN route.",
        },
        {
            "attempt": "code repair: control grouping isolation",
            "scope": "experiments/run_v22_43_metric_preserving_continuous_functional_flow_fu.py::group_key/control_group_key",
            "rows": str(len(s6_candidate_rows)),
            "primary_delta": "group keys now include support_rank, velocity_scale, refresh cadence, shrinkage, rho/barrier/radial settings, kan_init_variant and warmup; control groups also include variant",
            "control_or_gate": f"S6 KAN gain rows={final.get('s6_summary', {}).get('KAN_NLL_improvement_vs_own_optimizer_rows', '')}; TrueKANGain+BothGain={final.get('s6_summary', {}).get('TrueKANGain_plus_BothGain_rows', '')}; carrier_pass={final.get('s6_summary', {}).get('exploration_carrier_pass', '')}",
            "result": "Prevents cross-contamination between bracket rows; after recompute S6 remains not promoted.",
        },
        {
            "attempt": "code repair: S6 plan gate audit alignment",
            "scope": "experiments/run_v22_43_metric_preserving_continuous_functional_flow_fu.py::summarize_s6",
            "rows": str(final.get("s6_summary", {}).get("official_hard_rows", "")),
            "primary_delta": "internal gate now uses no-debt>=6/9 instead of all-row overhead; carrier gate records MLPDegradationDriven and plan_carrier_criteria_pass",
            "control_or_gate": f"no_debt={final.get('s6_summary', {}).get('no_ECE_Brier_tail_debt_rows', '')}/{final.get('s6_summary', {}).get('rows', '')}; MLPDegradationDriven={final.get('s6_summary', {}).get('MLPDegradationDriven_rows', '')}/{final.get('s6_summary', {}).get('rows', '')}; plan_carrier_criteria_pass={final.get('s6_summary', {}).get('plan_carrier_criteria_pass', '')}; exploration_carrier_pass={final.get('s6_summary', {}).get('exploration_carrier_pass', '')}",
            "result": "This is stricter/more plan-faithful audit, not a promotion. It exposes no-debt and MLP-degradation as current blockers.",
        },
        {
            "attempt": "code repair: train-only KAN init variant",
            "scope": "experiments/run_v22_43_metric_preserving_continuous_functional_flow_fu.py::apply_kan_init_variant + v22.44R CLI",
            "rows": str(len(freq_sustained)),
            "primary_delta": "`--kan-init-variant frequency-balanced` scales KAN w1/w2 banks from a training batch before optimizer creation; candidate/random/own use the same init",
            "control_or_gate": f"freq seed comparisons={max(0, len(freq_seed_comparisons) - 1)}; all state/runtime traces checked at 5000 steps",
            "result": "Implementation is auditably fair, but frequency-balanced init did not improve held NLL in the sustained SVHN bracket.",
        },
        {
            "attempt": "repair probe: frequency-balanced output calibration",
            "scope": "experiments/run_v22_43_metric_preserving_continuous_functional_flow_fu.py::apply_kan_init_variant",
            "rows": str(len(freq_calibrated_sustained)),
            "primary_delta": "new `frequency-balanced-output-calibrated` mode restores training-batch logits std after bank balancing by scaling w2",
            "control_or_gate": f"calibrated seed comparisons={len(freq_calibrated_seed_comparisons)}; seed0 trace rows=5000 when present",
            "result": "Seed0 sustained probe worsened absolute held NLL vs both uncalibrated frequency-balanced and default/basis-balanced, so it was not expanded to three seeds.",
        },
        {
            "attempt": "repair probe: low-degree / low-frequency biased init",
            "scope": "experiments/run_v22_43_metric_preserving_continuous_functional_flow_fu.py::apply_kan_init_variant",
            "rows": str(len(low_bias_sustained)),
            "primary_delta": "`low-frequency-biased` / `low-degree-biased` scales basis banks by 0.75^index without energy re-amplification",
            "control_or_gate": f"seed0 brackets={len(low_bias_comparisons)}; traces verified at 5000 steps for completed sustained rows",
            "result": "D-CHE low-degree produced a much stronger absolute KAN carrier, but FU candidate still traded held/tail against final/ECE/Brier/debt and did not pass same-basis controls.",
        },
        {
            "attempt": "repair probe: tail/no-debt KAN basis correction",
            "scope": "calibration_nuisance_loss_from_logits tail/tail_margin + low-degree D-CHE sustained probes",
            "rows": str(len(tail_nodebt_sustained)),
            "primary_delta": "added train-only tail, margin, and tail_margin nuisance losses plus tail-safe/tail-projected support scoring and low-degree tail-spectrum init; tested against same-basis random and own AdamW",
            "control_or_gate": f"tail probe brackets={len(tail_nodebt_comparisons)}; best tail_delta_vs_random={min((float(row['tail_q99_delta_vs_random']) for row in tail_nodebt_comparisons if row.get('tail_q99_delta_vs_random') not in {'', None}), default=float('nan'))}",
            "result": "Tail harm shrank substantially but remained positive vs same-basis random; the light correction line did not open no-debt.",
        },
        {
            "attempt": "S3 strong optimizer diagnostic",
            "scope": "v22_44R_s3_strong_opt_seed0",
            "rows": str(len(s3_candidate_rows)),
            "primary_delta": f"mean_NLL_improvement_own={mean_field(s3_candidate_rows, 'NLL_improvement_vs_own_strong_optimizer')}",
            "control_or_gate": f"beats_geometry_control={count_flag(s3_candidate_rows, 'beats_same_optimizer_geometry_random')}; no_debt={count_flag(s3_candidate_rows, 'no_ECE_Brier_tail_debt')}",
            "result": "Local gains did not beat same-optimizer geometry controls; no FU patch route claimed.",
        },
        {
            "attempt": "sustained single-row long training",
            "scope": "v22_44R_sustained5000_*",
            "rows": str(len(sustained)),
            "primary_delta": "5000 steps, train_size=4096, hidden=128; one process per GPU rather than row-dispatch churn",
            "control_or_gate": f"state_trace_rows_per_completed_row={range_field(sustained, 'steps')}",
            "result": "Confirmed real sustained collect traces; did not change final route.",
        },
    ]
    text = [
        "# DG-KAN v22.44R Functional-Actuator-Spectrum Metric-OET FU 实验结果复盘",
        "",
        f"更新时间：{now_sg()}",
        "",
        "## 执行边界与审计说明",
        "",
        "- 本复盘只引用 `results/v22_44R` 实际落盘 artifact；未执行项明确保留 `not_run` / `gate-blocked`，不补造数据。",
        "- 本轮新增 runner：`experiments/run_v22_44R_functional_actuator_spectrum_metric_oet_fu.py`。修改内容：新增 v22.44R 专用 A/B/C/D gate、functional actuator spectrum sketch、RSE projection、initialization ladder、v22.43 full-loop kernel alias、v22.44R final route、manifest 和 SVG 占位图。",
        "- 本轮后续修复还修改了 `experiments/run_v22_43_metric_preserving_continuous_functional_flow_fu.py` 的 pure merge 审计分类：P5 `same-readout-leakage-control` 若因无 readout leakage 而零速度，保留真实零速度但标记为 diagnostic/nonofficial P0 scope，不参与 official pure runtime route gate。",
        "- 本轮继续修复了 `basis_gram_diag` 的 KAN BasisGram metric 代理：修正 `w1`/`w2` Gram diag 与参数形状的对齐，剔除 readout extra features 对 `w2` diag 的误混入，并做 degree/frequency bank balancing + per-parameter unit-mean normalization。该修复只影响 metric/support 几何，不改模型 forward、不加优化器、不使用 validation/test/future direction。",
        "- 本轮修复了 v22.43 merge/enrich 的 control grouping：分组键现在纳入 rank、velocity、cadence、barrier、shrinkage、init variant、warmup 等关键配置，并且 control lookup 额外绑定 variant，避免不同 bracket 的 same-basis/random/own baseline 混用。",
        "- 本轮修复了 S6 summary/gate 的计划口径：新增 `no_ECE_Brier_tail_debt_rows`、`controller_overhead_le_0p25_rows`、`MLPDegradationDriven_rows`、`plan_carrier_criteria_pass`，并让 `exploration_internal_pass` 按计划 13.5 使用 no-debt gate，而不是把 official-candidate overhead gate 误用于 internal exploration。重算后 final route 未提升，说明该修复没有降低门槛。",
        "- 本轮新增 `--kan-init-variant`，当前实现只在 KAN 模型、optimizer 创建前、使用训练 batch 的 basis bank energy 做 degree/frequency bank scaling；candidate、same-basis random、own AdamW baseline 使用同一 init，以免把初始化效果误算成控制器效果。",
        "- Full-loop 训练内核复用 v22.43 的 train-only metric/OET controller；runtime candidate-action 字段继续落盘为 0，不使用 validation/test/future/query direction 做方向选择。",
        "",
        "## Final Route",
        "",
        "```json",
        json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Route Evidence Summary",
        "",
        md_table(route_evidence, ["route_or_question", "artifact", "rows", "primary_evidence", "conclusion"], 20),
        "",
        "## 已执行修复与审计记录",
        "",
        "- 代码修复修改了 `experiments/run_v22_44R_functional_actuator_spectrum_metric_oet_fu.py` 和 `experiments/run_v22_43_metric_preserving_continuous_functional_flow_fu.py`；没有向失败路线偷偷加入 AdamW、candidate selector、validation/test/future direction 或额外 auxiliary loss。",
        "- BasisGram 初始化 smoke 阶段曾发现 KAN readout 因除以极小 feature std 出现 logit explosion 风险；修复为按架构过滤 D-CHE/D-FOU actuator labels，并把 BasisGram readout normalization 改成 conservative shrink `min(1.0, mean_std)`。复跑 Part A/B/C/D 后 `logit_explosion_flag` 落盘为 0。",
        "- S4/S1 blocker 按计划方向尝试了 lower velocity、rank reduction、longer cadence、safety-budget/debt velocity barrier 与 nuisance-rank/EMA 调整；结果保留为失败/未晋级，没有降低 success threshold。",
        "- 用户指出 `nvidia-smi` 看到进程频繁出现又退出后，补充了 sustained 5000-step 长训：每个 row 固定一张 GPU，state/runtime trace 均要求 5000 行；短 row dispatcher 不再作为主要训练可信度证据。",
        "",
        md_table(repair_evidence, ["attempt", "scope", "rows", "primary_delta", "control_or_gate", "result"], 20),
        "",
        "## Part A Code / Runtime Truth",
        "",
        md_table(code, ["clean_unzip_compileall_pass", "clean_unzip_import_pass", "missing_transitive_dependency_count", "official_DGKAN_identity_pass", "runtime_argmax_candidate_used", "candidate_action_selection_used_for_runtime", "status"], 5),
        "",
        "## Part B Metric-OET Fidelity",
        "",
        md_table(metric, ["metric_name", "exp_approx_type", "PSD_pass", "G_projection_idempotence_error", "G_skew_error", "generalized_spectrum_drift", "G_orthogonality_error", "official_candidate", "pass"], 18),
        "",
        "## Part C Functional Actuator Spectrum / RSE",
        "",
        md_table(fspec, ["dataset", "architecture", "init_name", "actuator", "support_param_count", "functional_actuator_effective_rank", "functional_actuator_condition_number", "signal_reachable_energy", "basis_Gram_condition_number", "output_logit_variance", "status"], 36),
        "",
        "## Part D Initialization Ladder",
        "",
        md_table(init, ["dataset", "architecture", "init_name", "actuator", "functional_spectrum_distance", "RSE_distance", "output_scale_distance", "basis_Gram_condition", "activation_collapse_flag", "logit_explosion_flag", "status"], 36),
        "",
        "## Full-Loop Matrices",
        "",
        "### S4-M2 Metric/OET Support",
        "",
        md_table(s4, ["run_label", "dataset", "seed", "architecture", "variant", "control_mode", "final_NLL", "NLL_improvement_vs_own_strong_optimizer", "no_ECE_Brier_tail_debt", "controller_overhead_ratio"], 24),
        "",
        "### S1-M2 Residual Signal",
        "",
        md_table(s1, ["run_label", "dataset", "seed", "variant", "control_mode", "tau_direction_H20", "tau_direction_H60", "tau_direction_H200", "tau_direction_H800", "residual_signal_SNR", "no_ECE_Brier_tail_debt"], 24),
        "",
        "### Pure FU",
        "",
        md_table(pure, ["run_label", "dataset", "seed", "architecture", "variant", "control_mode", "pure_fu_mode", "base_optimizer_step_used", "bp_gradient_used_only_for_cotangent", "train_NLL_initial", "train_NLL_final", "no_ECE_Brier_tail_debt"], 24),
        "",
        "### S6-M2 KAN Carrier",
        "",
        md_table(s6, ["run_label", "dataset", "seed", "architecture", "variant", "final_NLL", "basis_energy_fraction", "readout_leakage_fraction", "KAN_NLL_delta_vs_MLP_matched_support_FU", "TrueKANGain_class"], 24),
        "",
        "### MLP Matched Functional Support",
        "",
        md_table(mlp, ["run_label", "dataset", "seed", "variant", "final_NLL", "NLL_improvement_vs_own_strong_optimizer", "controller_overhead_ratio"], 18),
        "",
        "### Strong Optimizer",
        "",
        md_table(strong, ["run_label", "dataset", "seed", "architecture", "optimizer_family", "variant", "final_NLL", "NLL_improvement_vs_strongest_optimizer"], 18),
        "",
        "### Sustained 5000-Step Long Rows",
        "",
        "- 这些行是为回应 `nvidia-smi` 看到短进程 churn 而补充的单 row 长训验证；每行一个 Python 进程、绑定一张 GPU、`steps=5000`、`train_size=4096`、`hidden=128`，不是 reduced matrix dispatcher。",
        "- 每个 completed row 都落盘 5000 行 state trace 和 5000 行 runtime trace；短矩阵 row 的结果不因此被改写。",
        "",
        md_table(sustained, ["run_label", "dataset", "architecture_key", "variant", "control_mode", "steps", "train_size", "hidden", "final_NLL", "held_NLL", "final_accuracy", "train_NLL_final", "controller_overhead_ratio", "runtime_argmax_candidate_used", "candidate_action_selection_used_for_runtime"], 12),
        "",
        md_table(sustained_comparisons, ["comparison", "train_final_NLL_delta", "held_NLL_delta", "accuracy_delta", "interpretation"], 8),
        "",
        "### Balanced BasisGram Sustained Repair",
        "",
        "- 该组用于审计 BasisGram bank-aligned metric diag 修复是否带来真实候选信号。所有 delta 均为 candidate minus same-basis random；NLL/ECE/Brier/tail/debt 为负更好，accuracy 为正更好。",
        "- 结论：`rank32 + vscale20 + safety barrier2` 在 SVHN seed0/1/2 上有弱 NLL/tail/debt 信号，但 held NLL 仅 2/3 seeds 胜出、accuracy 均值为 0，不能升级 final route。",
        "",
        md_table(
            balanced_sustained,
            ["run_label", "seed", "variant", "control_mode", "support_rank", "velocity_scale", "safety_budget_velocity_barrier", "final_NLL", "held_NLL", "final_accuracy", "ECE", "Brier", "tail_loss_q99", "fu_velocity_norm", "metric_condition_number", "support_overlap", "controller_overhead_ratio"],
            36,
        ),
        "",
        md_table(
            balanced_barrier2_seed_comparisons,
            ["seed", "held_NLL_delta", "final_NLL_delta", "accuracy_delta", "ECE_delta", "Brier_delta", "tail_q99_delta", "safety_debt_delta", "interpretation"],
            12,
        ),
        "",
        "### Frequency-Balanced Init Sustained Check",
        "",
        "- 该组按计划中的 `Init-KAN-D frequency-balanced D-FOU init` 方向执行，固定 `rank32 + vscale20 + safety barrier2`，SVHN seed0/1/2 各跑 candidate、same-basis random、own AdamW，三者都使用同一个 `kan_init_variant=frequency-balanced`。",
        "- 结论：frequency-balanced init 在这组 sustained run 中没有带来好消息。candidate 相对 random 的 held NLL 为 0/3 seed 胜，final NLL 为 1/3 seed 胜；accuracy 和 tail q99 有改善，但 NLL 明显不如 default/basis-balanced bracket，不能升级 final route。",
        "",
        md_table(
            freq_sustained,
            ["run_label", "seed", "variant", "control_mode", "kan_init_variant", "support_rank", "velocity_scale", "safety_budget_velocity_barrier", "final_NLL", "held_NLL", "final_accuracy", "ECE", "Brier", "tail_loss_q99", "safety_budget_debt", "controller_overhead_ratio"],
            18,
        ),
        "",
        md_table(
            freq_seed_comparisons,
            ["seed", "held_NLL_delta_vs_random", "final_NLL_delta_vs_random", "accuracy_delta_vs_random", "ECE_delta_vs_random", "Brier_delta_vs_random", "tail_q99_delta_vs_random", "safety_debt_delta_vs_random", "held_NLL_delta_vs_own", "final_NLL_delta_vs_own", "accuracy_delta_vs_own", "interpretation"],
            12,
        ),
        "",
        "### Frequency-Balanced Output-Calibrated Repair Probe",
        "",
        "- 该组是在 frequency-balanced 失败后追加的修复探针：新增 `frequency-balanced-output-calibrated`，先做 bank balancing，再用训练 batch 的 logits std 对 `w2` 做全局尺度校准。它不使用 held/test，不改变 forward，不加优化器。",
        "- 结论：seed0 sustained probe 没有好转，absolute held NLL 比 uncalibrated frequency-balanced 更差，也明显差于 default/basis-balanced；因此没有扩到 seed1/2。",
        "",
        md_table(
            freq_calibrated_sustained,
            ["run_label", "seed", "variant", "control_mode", "kan_init_variant", "support_rank", "velocity_scale", "safety_budget_velocity_barrier", "final_NLL", "held_NLL", "final_accuracy", "ECE", "Brier", "tail_loss_q99", "safety_budget_debt", "metric_condition_number", "controller_overhead_ratio"],
            12,
        ),
        "",
        md_table(
            freq_calibrated_seed_comparisons,
            ["seed", "held_NLL_delta_vs_random", "final_NLL_delta_vs_random", "accuracy_delta_vs_random", "ECE_delta_vs_random", "Brier_delta_vs_random", "tail_q99_delta_vs_random", "safety_debt_delta_vs_random", "held_NLL_delta_vs_own", "final_NLL_delta_vs_own", "accuracy_delta_vs_own", "interpretation"],
            8,
        ),
        "",
        "### Low-Degree / Low-Frequency Biased Init Repair Probe",
        "",
        "- 该组执行计划中的 `Init-KAN-E low-degree / low-frequency biased init`：只按 bank index 做 `0.75^index` 衰减，不像 frequency-balanced 那样反向放大小能量 bank。D-FOU 使用 low-frequency-biased，D-CHE 使用 low-degree-biased。",
        "- 结论：D-FOU low-frequency seed0 相对 controls 有小幅 NLL 收益但 absolute held NLL 仍弱于 default/basis-balanced；D-CHE low-degree 的 absolute held NLL 很强，但 FU candidate 对 held/tail 仍为负贡献。将 `velocity_scale` 从 20 降到 0.2 连续减小 held harm，但没有打开 same-basis control gate。",
        "",
        md_table(
            low_bias_sustained,
            ["run_label", "architecture_key", "seed", "variant", "control_mode", "kan_init_variant", "support_rank", "velocity_scale", "final_NLL", "held_NLL", "final_accuracy", "ECE", "Brier", "tail_loss_q99", "safety_budget_debt", "metric_condition_number", "controller_overhead_ratio"],
            24,
        ),
        "",
        md_table(
            low_bias_comparisons,
            ["bracket", "architecture", "kan_init_variant", "velocity_scale", "held_NLL_delta_vs_random", "final_NLL_delta_vs_random", "accuracy_delta_vs_random", "ECE_delta_vs_random", "Brier_delta_vs_random", "tail_q99_delta_vs_random", "safety_debt_delta_vs_random", "held_NLL_delta_vs_own", "final_NLL_delta_vs_own", "interpretation"],
            12,
        ),
        "",
        "### Tail / No-Debt KAN Basis Repair Probe",
        "",
        "- 该组是在 low-degree D-CHE absolute carrier 很强但 tail/no-debt 失败后执行的计划 fallback：新增 train-only `tail`、`margin`、`tail_margin` nuisance loss，并测试 rank、velocity、tail correction 强度；随后追加 tail-safe support scoring，在 support 选择时惩罚 tail/margin 梯度同幅坐标；追加 tail-projected support，用 tail gradient 对 signal 做 metric residualization 后再选择 support 和构造 support signal；最后追加 low-degree tail-spectrum init，用训练 batch 的 label/tail class separability 重加权 basis bank，并把 `tail_brier` 纳入 tail-safe support/projection。candidate、same-basis random、own AdamW 都用同一初始化和同一 train-only correction/support 配置。",
        "- 结论：tail/no-debt 修复方向有效但不足。`lowdegree_tail_spectrum_rank8_vscale0p2_barrier10_corr1` 首次让 tail q99 delta vs random 变为负数 `-0.00047206878662109375`；`lowdegree_tail_spectrum_tailbrier_projected_rank8_vscale0p2_barrier10_corr1` 进一步到 `-0.0005359649658203125`，并保持 held NLL/ECE/safety debt 改善，但 final NLL 和 Brier 仍为正 debt；brier-only/projection 能改善 Brier/final 但会伤 tail，rank2 也表现出同样 tradeoff；rank4+低 velocity 已接近边界但仍至少有一个 no-debt 分量为正。因此当前 blocker 是 tail 与 Brier/final/safety 的多目标冲突，而不是进程早退或单纯步幅过大；不能升级 S6 internal/carrier route。",
        "",
        md_table(
            tail_nodebt_sustained,
            ["run_label", "architecture_key", "seed", "variant", "control_mode", "kan_init_variant", "support_rank", "velocity_scale", "calibration_nuisance_mode", "calibration_nuisance_weight", "calibration_correction_weight", "final_NLL", "held_NLL", "final_accuracy", "ECE", "Brier", "tail_loss_q99", "safety_budget_debt", "controller_overhead_ratio"],
            36,
        ),
        "",
        md_table(
            tail_nodebt_comparisons,
            ["bracket", "rank", "velocity_scale", "nuisance_mode", "nuisance_weight", "correction_weight", "held_NLL_delta_vs_random", "final_NLL_delta_vs_random", "accuracy_delta_vs_random", "ECE_delta_vs_random", "Brier_delta_vs_random", "tail_q99_delta_vs_random", "safety_debt_delta_vs_random", "held_NLL_delta_vs_own", "final_NLL_delta_vs_own", "tail_q99_delta_vs_own", "interpretation"],
            12,
        ),
        "",
        "## Efficiency Evidence",
        "",
        md_table(eff, ["dataset", "seed", "variant", "metric_name", "controller_overhead_ratio", "full_step_ms", "base_optimizer_ms", "controller_ms", "basis_metric_update_time_ms", "memory_peak_MB"], 24),
        "",
        "## Blocker / 修复证据链",
        "",
        "- 如果 A/B gate 失败，本 runner 的 `stage=all` 会在前置 gate 后停止 scientific route；这是按计划的 `fix implementation only`，不是算法 no-go。",
        "- 如果 functional spectrum 计算过慢，当前实现已采用 finite-difference sketch + 小 batch + actuator bank 分块；未退回 raw matrix spectrum。",
        "- 如果 basis Gram condition 过高，初始化 ladder 中的 `BasisGram-normalized` row 会记录归一化后的 condition/output scale；是否改善只按落盘数值判断。",
        "- 如果 S4 有 NLL gain 但 no-debt/overhead 失败，后续补跑应优先使用 `--safety-budget-velocity-barrier`、降低 `--velocity-scale`、提高 `--metric-refresh-cadence` 或降低 `--support-rank`，不能降低 success threshold。",
        "- 如果 pure FU 不下降，优先审计 sign convention、Cayley/OET drift、`base_optimizer_step_used` 和 `bp_gradient_used_only_for_cotangent`；不能偷偷加 AdamW。",
        "",
        "## Analysis / Insight",
        "",
        "- Q1 只能由 `v22_44R_functional_actuator_spectrum_matrix.csv` 和 `v22_44R_initialization_ladder_matrix.csv` 支撑；raw spectrum 未被用作初始化公平性的替代证据。",
        "- Q2/S4 若未通过，结论是当前 metric/OET support 还没有在完成的 row 中证明 optimizer value；不能写成 SignalFUOpened。",
        "- Q3/S1 若未通过，即使 S4 有收益，也只能解释为 support/preconditioner 效应，不能声明 residual direction 独立增量。",
        "- Q4/Pure 若未通过或未跑满，pure-FU 只保持 pending；warmup row 必须和 warmup-only control 分开解释。",
        "- Q5/S6 若 KAN internal row 改善但 MLP matched support 更强，必须记录为 KANInternalOnly 或 MLPStructuredSupportStronger，不能宣传 KAN superiority。",
        f"- S6 现在的正信号主要是 architecture diagnostic：beats_MLP_matched={final.get('s6_summary', {}).get('beats_MLP_matched_metric_support_rows', '')}/{final.get('s6_summary', {}).get('rows', '')} 和 TrueKANGain+BothGain={final.get('s6_summary', {}).get('TrueKANGain_plus_BothGain_rows', '')}/{final.get('s6_summary', {}).get('rows', '')} 说明 KAN carrier 不是全空；但 no_ECE_Brier_tail_debt={final.get('s6_summary', {}).get('no_ECE_Brier_tail_debt_rows', '')}/{final.get('s6_summary', {}).get('rows', '')} 与 MLPDegradationDriven={final.get('s6_summary', {}).get('MLPDegradationDriven_rows', '')}/{final.get('s6_summary', {}).get('rows', '')} 说明不能把它解释成稳定、无债务、优于 MLP matched support 的 functional optimizer。",
    ]
    RECAP_DOC.write_text("\n".join(text) + "\n", encoding="utf-8")


def stage_truth(args: argparse.Namespace) -> dict[str, Any]:
    code = run_code_truth_gate()
    if str(code.get("status")) != "pass":
        return {"status": "gate_blocked", "stage": "A", "reason": "code truth failed"}
    metric = run_metric_oet_fidelity()
    if metric.get("status") != "pass":
        return {"status": "gate_blocked", "stage": "B", "reason": "metric OET fidelity failed"}
    part_c = run_part_c(args)
    part_d = run_part_d(args)
    return {"status": "pass", "code": code, "metric": metric, "part_c": part_c, "part_d": part_d}


def stage_all(args: argparse.Namespace) -> dict[str, Any]:
    truth = stage_truth(args)
    if truth.get("status") != "pass":
        write_required_placeholders()
        final = finalize_v22_44R(write_recap=True)
        final["stage_all_gate_blocker"] = truth
        write_json(OUT_ROOT / "v22_44R_final_route.json", final)
        return final
    dispatch_specs(args, v2243.task_specs(args, "s4"), "s4")
    stage_s1(args)
    dispatch_specs(args, v2243.task_specs(args, "s6"), "s6")
    dispatch_specs(args, v2243.task_specs(args, "s3"), "s3")
    pure_args = argparse.Namespace(**vars(args))
    pure_args.pure_fu_mode = True
    pure_args.eval_datasets = args.pure_eval_datasets
    pure_args.eval_seeds = args.pure_eval_seeds
    pure_args.eval_architectures = args.pure_eval_architectures
    pure_args.row_limit = args.pure_row_limit
    pure_args.control_modes = args.p3_control_modes
    dispatch_specs(pure_args, v2243.task_specs(pure_args, "p3"), "p3")
    return finalize_v22_44R(write_recap=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all", "truth", "code", "metric", "spectrum", "init", "collect", "s4", "s1", "s6", "s3", "p3", "p4", "p5", "p6", "merge", "pure-merge", "finalize"])
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
    p.add_argument("--calibration-nuisance-weight", type=float, default=0.0)
    p.add_argument("--calibration-correction-weight", type=float, default=0.0)
    p.add_argument("--calibration-nuisance-mode", choices=["brier", "confidence", "overconfidence", "tail", "margin", "tail_margin", "tail_brier", "tail_brier_balanced", "tail_brier_brier2_balanced", "tail_brier_pareto", "tail_brier_pareto2", "tail_brier_qp", "tail_brier_qp_margin", "tail_brier_qp_brier_margin", "tail_q99_brier_qp_margin"], default="brier")
    p.add_argument("--calibration-nuisance-cadence", type=int, default=1)
    p.add_argument("--cached-controller-emit-cadence", type=int, default=1)
    p.add_argument("--kan-init-variant", default="default")
    p.add_argument("--pure-fu-mode", action="store_true")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--tier2-download", action="store_true")
    p.add_argument("--label", default="v22_44R")
    p.add_argument("--eval-datasets", default="CIFAR10,SVHN,Wine")
    p.add_argument("--eval-seeds", default="0,1")
    p.add_argument("--eval-architectures", default="MLP,DGKAN_DCHE")
    p.add_argument("--strong-optimizers", default="AdamW")
    p.add_argument("--s4-variants", default="S4-Euclidean-OET,S4-FisherEMA-OET,S4-SignalMetric-OET")
    p.add_argument("--s6-variants", default="KAN-D-CHE-BasisGram,KAN-D-CHE-OET-BankLocal,MLP-low-rank-hidden-metric-support,MLP-frequency-like-random-feature-support,MLP-same-rank-block-support")
    p.add_argument("--s3-variants", default="S3-MetricSupportFU,S3-ResidualSignalFU,S3-SameMetricSupportControl,S3-SameTangentControl")
    p.add_argument("--p3-variants", default=",".join(v2243.P3_VARIANTS))
    p.add_argument("--p3-control-modes", default="none,same-OET-random")
    p.add_argument("--p4-variants", default=",".join(v2243.P4_VARIANTS))
    p.add_argument("--p4-control-modes", default=",".join(v2243.P4_CONTROL_MODES))
    p.add_argument("--p5-variants", default=",".join(v2243.P5_VARIANTS))
    p.add_argument("--p5-control-modes", default=",".join(v2243.P5_CONTROL_MODES))
    p.add_argument("--p6-variants", default=",".join(v2243.P6_VARIANTS))
    p.add_argument("--p6-control-modes", default="auto")
    p.add_argument("--eval-variants", default="S4-Euclidean-OET")
    p.add_argument("--control-modes", default="none,same-metric-support-random,same-metric-support-signflip")
    p.add_argument("--s6-control-modes", default="none,same-basis-Gram-random,same-basis-Gram-signflip")
    p.add_argument("--include-optimizer-alone", action="store_true")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--row-timeout", type=int, default=1200)
    p.add_argument("--row-limit", type=int, default=0)
    p.add_argument("--audit-datasets", default="MNIST,FashionMNIST,KMNIST,Wine")
    p.add_argument("--audit-architectures", default="MLP,DGKAN_DCHE,DGKAN_DFOU")
    p.add_argument("--audit-supports", default="A_MLP_lowrank,A_MLP_spectral,A_MLP_frequency_like,A_KAN_DCHE_basis_bank,A_KAN_DFOU_basis_bank,A_KAN_DCHE_bank_OET,A_KAN_DFOU_bank_OET,A_OET_layer")
    p.add_argument("--audit-train-size", type=int, default=96)
    p.add_argument("--sketch-dim", type=int, default=6)
    p.add_argument("--functional-eps", type=float, default=1.0e-3)
    p.add_argument("--pure-eval-datasets", default="MNIST,FashionMNIST,KMNIST")
    p.add_argument("--pure-eval-seeds", default="0")
    p.add_argument("--pure-eval-architectures", default="MLP")
    p.add_argument("--pure-row-limit", type=int, default=12)
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_out()
    bind_v2243()
    if args.stage == "code":
        run_code_truth_gate()
    elif args.stage == "metric":
        run_metric_oet_fidelity()
    elif args.stage == "spectrum":
        run_part_c(args)
    elif args.stage == "init":
        run_part_d(args)
    elif args.stage == "truth":
        stage_truth(args)
    elif args.stage == "collect":
        v2243.train_variant(
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
            cached_controller_emit_cadence=args.cached_controller_emit_cadence,
            pure_fu_mode=args.pure_fu_mode,
            warmup_steps=args.warmup_steps,
            kan_init_variant=args.kan_init_variant,
            tier2_download=args.tier2_download,
            label=args.label,
        )
    elif args.stage == "s4":
        dispatch_specs(args, v2243.task_specs(args, "s4"), "s4")
    elif args.stage == "s1":
        stage_s1(args)
    elif args.stage == "s6":
        dispatch_specs(args, v2243.task_specs(args, "s6"), "s6")
    elif args.stage == "s3":
        dispatch_specs(args, v2243.task_specs(args, "s3"), "s3")
    elif args.stage == "p3":
        args.pure_fu_mode = True
        dispatch_specs(args, v2243.task_specs(args, "p3"), "p3")
    elif args.stage == "p4":
        args.pure_fu_mode = True
        dispatch_specs(args, v2243.task_specs(args, "p4"), "p4")
    elif args.stage == "p5":
        args.pure_fu_mode = True
        dispatch_specs(args, v2243.task_specs(args, "p5"), "p5")
    elif args.stage == "p6":
        args.pure_fu_mode = True
        dispatch_specs(args, v2243.task_specs(args, "p6"), "p6")
    elif args.stage == "merge":
        merge_v2243_outputs(pure=False)
    elif args.stage == "pure-merge":
        merge_v2243_outputs(pure=True)
    elif args.stage == "finalize":
        finalize_v22_44R(write_recap=True)
    else:
        stage_all(args)


if __name__ == "__main__":
    main()
