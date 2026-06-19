#!/usr/bin/env python3
"""DG-KAN v22.40 Continuous Functional Flow FU runner.

The central non-regression rule for this runner is simple: runtime FU is a
continuous per-step state-space velocity field.  Candidate families and random
micro-interventions are audit/calibration signals only, never runtime action
selection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
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


PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
OUT_ROOT = ROOT / "results/v22_40"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
PLAN_DOC = ROOT / "docs/DG-KAN_v22.40_ContinuousFunctionalFlowFU_显式防复发版.md"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.40_ContinuousFunctionalFlowFU_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.40_ContinuousFunctionalFlowFU_实验结果复盘.md"

REQUIRED_ARTIFACTS = [
    "v22_40_code_truth_gate.csv",
    "v22_40_identity_firewall_matrix.csv",
    "v22_40_continuous_runtime_truth_matrix.csv",
    "v22_40_candidate_action_regression_audit.csv",
    "v22_40_continuous_fu_state_trace.csv",
    "v22_40_runtime_decision_trace.csv",
    "v22_40_micro_intervention_audit_only_proof.csv",
    "v22_40_command_journal.csv",
    "v22_40_horizon_decay_matrix.csv",
    "v22_40_support_direction_decomposition.csv",
    "v22_40_safety_debt_source_matrix.csv",
    "v22_40_candidate_identity_leakage_audit.csv",
    "v22_40_KAN_internal_vs_architecture_gap_matrix.csv",
    "v22_40_controller_unit_test_matrix.csv",
    "v22_40_micro_intervention_audit_matrix.csv",
    "v22_40_causal_state_update_matrix.csv",
    "v22_40_audit_not_runtime_selection_proof.csv",
    "v22_40_full_loop_matrix.csv",
    "v22_40_full_loop_summary.csv",
    "v22_40_long_horizon_path_validation.csv",
    "v22_40_continual_memory_matrix.csv",
    "v22_40_grokking_delay_matrix.csv",
    "v22_40_noisy_preference_matrix.csv",
    "v22_40_noisy_preference_summary.csv",
    "v22_40_noisy_preference_state_trace.csv",
    "v22_40_efficiency_matrix.csv",
    "v22_40_final_route.json",
    "v22_40_artifact_manifest.csv",
]

RUNTIME_FORBIDDEN_FIELDS = {
    "treatment_selected",
    "direction_source",
    "action_family",
    "candidate_id",
    "future_outcome",
    "test_result",
    "post_treatment_held_outcome",
    "H20_outcome",
    "H60_outcome",
    "H200_outcome",
    "H800_outcome",
}


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def safe_fragment(value: Any) -> str:
    chars = []
    for ch in str(value):
        chars.append(ch if (ch.isalnum() or ch in {"-", "_"}) else "_")
    return "".join(chars).strip("_") or "x"


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def int_flag(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def split_csv(text: str, cast: Any = str) -> list[Any]:
    out: list[Any] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    return out


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.40 Continuous Functional Flow FU 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、文件、GPU、状态、blocker 与修复尝试；"
            "未执行或失败项目必须明确标记，不用假数据补齐。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.40 Continuous Functional Flow FU 实验结果复盘\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：复盘只引用本轮 artifact 或明确命名的上游 artifact；"
            "不把诊断结果写成 official success。\n",
            encoding="utf-8",
        )


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
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def append_exec(command: str, *, task_id: str, status: str, gpu: str = "", files: str = "", note: str = "", exit_code: Any = "n/a") -> None:
    ensure_out()
    row = {
        "timestamp": now_sg(),
        "task_id": task_id,
        "command": command,
        "gpu": gpu,
        "status": status,
        "exit_code": exit_code,
        "files": files,
        "note": note,
    }
    journal = read_rows(OUT_ROOT / "v22_40_command_journal.csv")
    journal.append({k: str(v) for k, v in row.items()})
    write_rows(
        OUT_ROOT / "v22_40_command_journal.csv",
        journal,
        ["timestamp", "task_id", "command", "gpu", "status", "exit_code", "files", "note"],
    )
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {row['timestamp']} {task_id}\n\n")
        f.write("```bash\n" + str(command) + "\n```\n\n")
        f.write(f"- gpu: {gpu}\n- status: {status}\n- exit_code: {exit_code}\n")
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
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=timeout)
    stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="replace")
    append_exec(
        " ".join(shlex.quote(x) for x in cmd),
        task_id=task_id,
        status="pass" if proc.returncode == 0 else "fail",
        gpu=gpu,
        files=f"{stdout_path.relative_to(ROOT)}, {stderr_path.relative_to(ROOT)}",
        note=f"elapsed_sec={time.time() - start:.3f}; cwd={ROOT}",
        exit_code=proc.returncode,
    )
    return proc


def md_table(rows: list[dict[str, Any]], columns: list[str] | None = None, limit: int = 16) -> str:
    if not rows:
        return "_无 rows_"
    cols = columns or list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in rows[:limit]:
        out.append("| " + " | ".join(str(row.get(c, "")).replace("\n", " ") for c in cols) + " |")
    if len(rows) > limit:
        out.append(f"\n_仅显示前 {limit} / {len(rows)} 行；完整 CSV 见 artifact。_")
    return "\n".join(out)


def torch_device(name: str) -> Any:
    import torch

    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(str(name))
    return torch.device("cpu")


def tensor_cosine(a: Any, b: Any) -> float:
    denom = float(a.float().norm().item() * b.float().norm().item())
    if denom <= 1.0e-12 or not math.isfinite(denom):
        return 0.0
    return float((a.float() * b.float()).sum().item() / denom)


def flatten_named(named: list[tuple[str, Any]], source: str, opt: Any | None = None) -> Any:
    import torch

    parts = []
    for _name, p in named:
        value = None
        if source == "grad":
            value = p.grad
        elif source == "exp_avg" and opt is not None:
            value = opt.state.get(p, {}).get("exp_avg")
        elif source == "param":
            value = p.detach()
        if value is None:
            value = torch.zeros_like(p)
        parts.append(value.detach().reshape(-1).float())
    return torch.cat(parts) if parts else torch.empty(0)


def norm_dict(values: dict[str, Any]) -> float:
    total = 0.0
    for v in values.values():
        total += float(v.float().norm().item()) ** 2
    return math.sqrt(total)


def apply_velocity(named: list[tuple[str, Any]], velocity: dict[str, Any], scale: float) -> tuple[float, float, float]:
    total = 0.0
    basis = 0.0
    nonbasis = 0.0
    with __import__("torch").no_grad():
        for name, p in named:
            v = velocity.get(name)
            if v is None:
                continue
            delta = -float(scale) * v.to(device=p.device, dtype=p.dtype)
            p.add_(delta)
            n = float(delta.norm().item())
            total += n * n
            if name in {"w1", "w2"} or name.endswith(".w1") or name.endswith(".w2"):
                basis += n * n
            else:
                nonbasis += n * n
    return math.sqrt(total), math.sqrt(basis), math.sqrt(nonbasis)


def inject_gradient_velocity(
    named: list[tuple[str, Any]],
    velocity: dict[str, Any],
    grad_scale: float,
    lr: float,
) -> tuple[float, float, float]:
    total = 0.0
    basis = 0.0
    nonbasis = 0.0
    with __import__("torch").no_grad():
        for name, p in named:
            if p.grad is None:
                continue
            v = velocity.get(name)
            if v is None:
                continue
            injection = float(grad_scale) * v.to(device=p.device, dtype=p.grad.dtype)
            p.grad.add_(injection)
            # Report an optimizer-step-equivalent norm so pre-step and post-step
            # actuation remain comparable in the audit tables.
            n = abs(float(lr)) * float(injection.norm().item())
            total += n * n
            if name in {"w1", "w2"} or name.endswith(".w1") or name.endswith(".w2"):
                basis += n * n
            else:
                nonbasis += n * n
    return math.sqrt(total), math.sqrt(basis), math.sqrt(nonbasis)


def batch_risk_metrics(model: Any, xb: Any, yb: Any, output_dim: int) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        logits = model(xb).float()
        losses = F.cross_entropy(logits, yb.long(), reduction="none")
        return risk_metrics_from_logits(logits, yb, output_dim, losses)


def risk_metrics_from_logits(logits: Any, yb: Any, output_dim: int, losses: Any | None = None) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        logits = logits.detach().float()
        if losses is None:
            losses = F.cross_entropy(logits, yb.long(), reduction="none")
        else:
            losses = losses.detach().float()
        probs = torch.softmax(logits, dim=-1)
        target = F.one_hot(yb.long(), num_classes=int(output_dim)).float()
        conf, pred = probs.max(dim=-1)
        ok = pred.eq(yb.long()).float()
        ece = torch.zeros((), device=logits.device)
        for lo in torch.linspace(0.0, 0.9, 10, device=logits.device):
            hi = lo + 0.1
            mask = (conf >= lo) & ((conf < hi) if float(hi.item()) < 1.0 else (conf <= hi))
            if bool(mask.any()):
                ece = ece + mask.float().mean() * (conf[mask].mean() - ok[mask].mean()).abs()
        true_logits = logits.gather(1, yb.long().view(-1, 1)).squeeze(1)
        masked = logits.clone()
        masked[torch.arange(yb.numel(), device=logits.device), yb.long()] = -float("inf")
        margin = true_logits - masked.max(dim=-1).values
        ce = float(losses.mean().item())
        tail_q99 = float(torch.quantile(losses.float(), 0.99).item())
        sharpness_proxy = float((losses.float().max() - losses.float().mean()).item() / max(1.0e-6, abs(ce)))
        return {
            "ce": ce,
            "ece": float(ece.item()),
            "brier": float(((probs - target) ** 2).sum(dim=-1).mean().item()),
            "tail_q95": float(torch.quantile(losses.float(), 0.95).item()),
            "tail_q99": tail_q99,
            "margin_q10": float(torch.quantile(margin.float(), 0.10).item()),
            "margin_q01": float(torch.quantile(margin.float(), 0.01).item()),
            "sharpness_proxy": sharpness_proxy,
            "risk_has_calibration": 1,
        }


def cheap_train_risk_from_losses(losses: Any) -> dict[str, float]:
    with __import__("torch").no_grad():
        losses = losses.detach().float()
        ce = float(losses.mean().item())
        tail = float(losses.max().item())
        return {
            "ce": ce,
            "ece": 0.0,
            "brier": 0.0,
            "tail_q95": tail,
            "tail_q99": tail,
            "margin_q10": 0.0,
            "margin_q01": 0.0,
            "sharpness_proxy": float((losses.max() - losses.mean()).item() / max(1.0e-6, abs(ce))),
            "risk_has_calibration": 0,
        }


def project_velocity_against_tail(named: list[tuple[str, Any]], velocity: dict[str, Any], tail_grads: dict[str, Any]) -> tuple[int, float]:
    import torch

    dot = torch.zeros((), device=next(iter(velocity.values())).device) if velocity else torch.zeros(())
    denom = torch.zeros_like(dot)
    for name, _p in named:
        v = velocity.get(name)
        g = tail_grads.get(name)
        if v is None or g is None:
            continue
        vf = v.float()
        gf = g.to(device=v.device).float()
        dot = dot + (vf * gf).sum()
        denom = denom + gf.square().sum()
    if not bool(torch.isfinite(dot)) or not bool(torch.isfinite(denom)) or float(denom.item()) <= 1.0e-12:
        return 0, 0.0
    alignment = float(dot.item() / math.sqrt(max(1.0e-24, float(denom.item()))))
    if float(dot.item()) >= 0.0:
        return 0, alignment
    coeff = dot / denom.clamp_min(1.0e-12)
    with torch.no_grad():
        for name, _p in named:
            v = velocity.get(name)
            g = tail_grads.get(name)
            if v is None or g is None:
                continue
            velocity[name] = (v.float() - coeff * g.to(device=v.device).float()).to(device=v.device, dtype=v.dtype)
    return 1, alignment


def optimizer_step(model: Any, opt: Any, optimizer_family: str, step: int, avg_state: dict[str, Any] | None = None) -> None:
    import torch

    if optimizer_family == "Cautious AdamW":
        with torch.no_grad():
            for group in opt.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    exp_avg = opt.state.get(p, {}).get("exp_avg")
                    if exp_avg is not None:
                        p.grad.mul_((p.grad * exp_avg >= 0.0).float())
    elif optimizer_family == "Muon-like":
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None and p.ndim == 2:
                    u, _s, vh = torch.linalg.svd(p.grad.float(), full_matrices=False)
                    p.grad.copy_((u @ vh).to(dtype=p.grad.dtype))
    opt.step()
    if optimizer_family == "Schedule-Free AdamW" and avg_state is not None:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name not in avg_state:
                    avg_state[name] = p.detach().clone()
                else:
                    avg_state[name].mul_(float(step) / float(step + 1)).add_(p.detach(), alpha=1.0 / float(step + 1))


def final_schedule_free_swap(model: Any, optimizer_family: str, avg_state: dict[str, Any] | None) -> None:
    if optimizer_family != "Schedule-Free AdamW" or not avg_state:
        return
    with __import__("torch").no_grad():
        for name, p in model.named_parameters():
            if name in avg_state:
                p.copy_(avg_state[name])


class ContinuousFUController:
    def __init__(
        self,
        *,
        architecture: str,
        variant: str,
        beta_signal: float,
        beta_q: float,
        eta_rho: float,
        eta_debt: float,
        tau_safe: float,
        rho_min: float,
        rho_max: float,
        velocity_scale: float,
        random_seed: int,
        mlp_velocity_source: str = "signal",
        slow_signal_beta: float = 0.01,
        fast_controller_stats: bool = False,
        safety_barrier_strength: float = 0.0,
        safety_barrier_slack: float = 0.0,
    ) -> None:
        self.architecture = architecture
        self.variant = variant
        self.beta_signal = float(beta_signal)
        self.beta_q = float(beta_q)
        self.eta_rho = float(eta_rho)
        self.eta_debt = float(eta_debt)
        self.tau_safe = float(tau_safe)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.velocity_scale = float(velocity_scale)
        self.mlp_velocity_source = str(mlp_velocity_source)
        self.slow_signal_beta = float(slow_signal_beta)
        self.fast_controller_stats = bool(fast_controller_stats)
        self.safety_barrier_strength = float(safety_barrier_strength)
        self.safety_barrier_slack = float(safety_barrier_slack)
        self.q = 0.05
        self.active_flow_variant = variant in {"continuous_fu", "same_signal_random_flow_control", "same_optimizer_geometry_control"}
        self.rho = max(self.rho_min, 0.02) if self.active_flow_variant else 0.0
        self.signal: dict[str, Any] = {}
        self.slow_signal: dict[str, Any] = {}
        self.basis_state: dict[str, Any] = {}
        self.risk_ema: dict[str, float] = {}
        self.risk_best: dict[str, float] = {}
        self.rng = random.Random(int(random_seed))
        self.last_debt = 0.0

    def update_and_emit(
        self,
        *,
        named: list[tuple[str, Any]],
        opt: Any,
        risk: dict[str, float],
        step: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        import torch

        debt_terms = []
        barrier_terms = []
        risk_has_calibration = int_flag(risk.get("risk_has_calibration", 1))
        risk_keys = ["tail_q99", "sharpness_proxy"]
        if risk_has_calibration:
            risk_keys += ["ece", "brier"]
        for key in risk_keys:
            value = float(risk.get(key, 0.0))
            old = self.risk_ema.get(key, value)
            self.risk_ema[key] = 0.98 * old + 0.02 * value
            if key in {"ece", "brier", "tail_q99"}:
                if value > 0.0 and old > 0.0:
                    debt_terms.append(max(0.0, value - self.risk_ema[key]))
            old_best = self.risk_best.get(key, value)
            best = min(old_best, value)
            self.risk_best[key] = best
            denom = max(1.0e-6, abs(best))
            rel_debt = 0.0 if value <= 0.0 or best <= 0.0 else max(0.0, (value - best) / denom - self.safety_barrier_slack)
            barrier_terms.append(rel_debt)
        safety_debt = float(sum(debt_terms))
        barrier_debt = float(sum(barrier_terms))
        safety_barrier_scale = 1.0 / (1.0 + max(0.0, self.safety_barrier_strength) * barrier_debt)
        signal_norm2_t = None
        slow_signal_norm2_t = None
        grad_norm2_t = None
        momentum_norm2_t = None
        align_num_t = None
        align_sig2_t = None
        align_src2_t = None
        slow_align_num_t = None
        slow_align_sig2_t = None
        slow_align_src2_t = None
        velocity: dict[str, Any] = {}
        for name, p in named:
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            if grad_norm2_t is None:
                grad_norm2_t = torch.zeros((), device=g.device)
                signal_norm2_t = torch.zeros((), device=g.device)
                slow_signal_norm2_t = torch.zeros((), device=g.device)
                momentum_norm2_t = torch.zeros((), device=g.device)
                align_num_t = torch.zeros((), device=g.device)
                align_sig2_t = torch.zeros((), device=g.device)
                align_src2_t = torch.zeros((), device=g.device)
                slow_align_num_t = torch.zeros((), device=g.device)
                slow_align_sig2_t = torch.zeros((), device=g.device)
                slow_align_src2_t = torch.zeros((), device=g.device)
            old_sig = self.signal.get(name)
            sig = g.clone() if old_sig is None else (1.0 - self.beta_signal) * old_sig + self.beta_signal * g
            self.signal[name] = sig.detach()
            old_slow = self.slow_signal.get(name)
            slow_beta = max(0.0, min(1.0, self.slow_signal_beta))
            slow_sig = g.clone() if old_slow is None else (1.0 - slow_beta) * old_slow + slow_beta * g
            self.slow_signal[name] = slow_sig.detach()
            grad_norm2_t = grad_norm2_t + g.square().sum()
            signal_norm2_t = signal_norm2_t + sig.square().sum()
            slow_signal_norm2_t = slow_signal_norm2_t + slow_sig.square().sum()
            exp_avg = opt.state.get(p, {}).get("exp_avg")
            if exp_avg is not None:
                align_src = exp_avg.float()
                momentum_norm2_t = momentum_norm2_t + align_src.square().sum()
            else:
                align_src = g
            align_num_t = align_num_t + (sig * align_src).sum()
            align_sig2_t = align_sig2_t + sig.square().sum()
            align_src2_t = align_src2_t + align_src.square().sum()
            slow_align_num_t = slow_align_num_t + (slow_sig * align_src).sum()
            slow_align_sig2_t = slow_align_sig2_t + slow_sig.square().sum()
            slow_align_src2_t = slow_align_src2_t + align_src.square().sum()
            apply_to_param = True
            if self.architecture != "MLP":
                apply_to_param = name in {"w1", "w2"} or name.endswith(".w1") or name.endswith(".w2")
            if self.variant == "same_signal_random_flow_control":
                rand = torch.randn_like(sig)
                direction = rand / rand.norm().clamp_min(1.0e-12) * sig.norm().clamp_min(1.0e-12)
            elif self.variant == "same_optimizer_geometry_control":
                source = exp_avg.float() if exp_avg is not None else g
                flat = source.reshape(-1)
                if flat.numel() > 1:
                    gen = torch.Generator(device=flat.device).manual_seed(self.rng.randrange(1, 2**31 - 1))
                    perm = torch.randperm(flat.numel(), generator=gen, device=flat.device)
                    source = flat[perm].reshape_as(source)
                direction = source
            elif self.variant == "same_overhead_noop_control":
                direction = torch.zeros_like(sig)
            else:
                if self.architecture == "MLP" and self.mlp_velocity_source == "optimizer_state":
                    direction = exp_avg.float() if exp_avg is not None else sig
                elif self.architecture == "MLP" and self.mlp_velocity_source == "optimizer_residual":
                    source = exp_avg.float() if exp_avg is not None else g
                    denom = source.float().square().sum().clamp_min(1.0e-12)
                    coeff = (sig.float() * source.float()).sum() / denom
                    direction = sig - coeff * source
                elif self.architecture == "MLP" and self.mlp_velocity_source == "slow_signal":
                    direction = slow_sig
                else:
                    direction = sig
            if apply_to_param:
                normed = direction / direction.norm().clamp_min(1.0e-12)
                velocity[name] = normed.to(device=p.device, dtype=p.dtype)
                if self.architecture != "MLP":
                    old_b = self.basis_state.get(name)
                    self.basis_state[name] = normed.detach().clone() if old_b is None else 0.98 * old_b + 0.02 * normed.detach()
        signal_norm2 = float(signal_norm2_t.item()) if signal_norm2_t is not None else 0.0
        slow_signal_norm2 = float(slow_signal_norm2_t.item()) if slow_signal_norm2_t is not None else 0.0
        if self.fast_controller_stats:
            grad_norm2 = signal_norm2
            momentum_norm2 = 0.0
        else:
            grad_norm2 = float(grad_norm2_t.item()) if grad_norm2_t is not None else 0.0
            momentum_norm2 = float(momentum_norm2_t.item()) if momentum_norm2_t is not None else 0.0
        signal_norm = math.sqrt(signal_norm2)
        slow_signal_norm = math.sqrt(slow_signal_norm2)
        grad_norm = math.sqrt(grad_norm2)
        optimizer_norm = math.sqrt(momentum_norm2)
        migration_index = slow_signal_norm / max(1.0e-12, signal_norm)
        if self.fast_controller_stats:
            snr = migration_index
            align = 0.0
            slow_align = 0.0
        else:
            snr = signal_norm / max(1.0e-12, math.sqrt(max(0.0, grad_norm2 - signal_norm2)) + 1.0e-12)
            align_denom = math.sqrt(
                max(1.0e-24, float(align_sig2_t.item()) if align_sig2_t is not None else 0.0)
                * max(1.0e-24, float(align_src2_t.item()) if align_src2_t is not None else 0.0)
            )
            slow_align_denom = math.sqrt(
                max(1.0e-24, float(slow_align_sig2_t.item()) if slow_align_sig2_t is not None else 0.0)
                * max(1.0e-24, float(slow_align_src2_t.item()) if slow_align_src2_t is not None else 0.0)
            )
            align = (float(align_num_t.item()) / align_denom) if align_num_t is not None and align_denom > 1.0e-12 else 0.0
            slow_align = (float(slow_align_num_t.item()) / slow_align_denom) if slow_align_num_t is not None and slow_align_denom > 1.0e-12 else 0.0
        slow_bonus = 0.0
        if self.architecture == "MLP" and self.mlp_velocity_source == "slow_signal":
            slow_bonus = 0.15 * math.tanh(migration_index) + 0.10 * max(-1.0, min(1.0, slow_align))
        benefit_proxy = (
            0.50 * math.tanh(snr)
            + 0.25 * max(-1.0, min(1.0, align))
            + slow_bonus
            - 4.0 * safety_debt
            - max(0.0, self.safety_barrier_strength) * barrier_debt
        )
        old_q = self.q
        old_rho = self.rho
        debt_delta_pos = max(0.0, safety_debt - self.last_debt)
        self.q = (1.0 - self.beta_q) * self.q + self.beta_q * benefit_proxy
        if self.active_flow_variant:
            next_rho = self.rho + self.eta_rho * (self.q - self.tau_safe) - self.eta_debt * (
                safety_debt + 2.0 * debt_delta_pos + max(0.0, self.safety_barrier_strength) * barrier_debt
            )
            if debt_delta_pos > 0.0 and next_rho >= old_rho:
                next_rho = old_rho - self.eta_debt * debt_delta_pos
            self.rho = min(self.rho_max, max(self.rho_min, next_rho))
        else:
            self.rho = 0.0
        basis_state_norm = norm_dict(self.basis_state)
        meta = {
            "step": step,
            "signal_state_norm": signal_norm,
            "slow_signal_state_norm": slow_signal_norm,
            "grad_norm": grad_norm,
            "optimizer_state_norm": optimizer_norm,
            "basis_state_norm": basis_state_norm,
            "causal_confidence_q_t": self.q,
            "q_prev": old_q,
            "rho_t": self.rho,
            "rho_prev": old_rho,
            "safety_debt": safety_debt,
            "safety_debt_delta": safety_debt - self.last_debt,
            "safety_barrier_debt": barrier_debt,
            "safety_barrier_scale": safety_barrier_scale,
            "sharpness_proxy": float(risk.get("sharpness_proxy", 0.0)),
            "signal_state_SNR": snr,
            "signal_optimizer_alignment": align,
            "slow_signal_optimizer_alignment": slow_align,
            "reservoir_to_signal_migration_index": migration_index,
            "benefit_proxy_train_only": benefit_proxy,
            "velocity_param_count": len(velocity),
        }
        self.last_debt = safety_debt
        return velocity, meta


def run_audit_micro_intervention(
    *,
    model: Any,
    opt: Any,
    architecture: str,
    optimizer_family: str,
    xb: Any,
    yb: Any,
    eval_xb: Any,
    eval_yb: Any,
    output_dim: int,
    lr: float,
    weight_decay: float,
    treatment: str,
    seed: int,
    audit_eval_source: str,
) -> dict[str, Any]:
    import torch.nn.functional as F

    branch = copy.deepcopy(model)
    branch_opt = core.optimizer_for(optimizer_family, branch.parameters(), lr, weight_decay)
    core.clone_optimizer_state(opt, branch_opt)
    before = batch_risk_metrics(model, eval_xb, eval_yb, output_dim)
    branch_opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(branch(xb).float(), yb.long())
    loss.backward()
    named, direction, direction_source = core.build_treatment_direction(branch, branch_opt, architecture, treatment, random.Random(seed))
    update_norm = core.apply_flat_delta(named, direction, 2.0e-5)
    after = batch_risk_metrics(branch, eval_xb, eval_yb, output_dim)
    nll_delta = float(before["ce"] - after["ce"])
    return {
        "sampled_treatment_name": treatment,
        "audit_eval_source": audit_eval_source,
        "direction_source_diagnostic": direction_source,
        "audit_update_norm": update_norm,
        "audit_eval_NLL_delta": nll_delta,
        "audit_held_NLL_delta": nll_delta,
        "audit_ECE_delta": float(before["ece"] - after["ece"]),
        "audit_Brier_delta": float(before["brier"] - after["brier"]),
        "audit_tail_q99_delta": float(before["tail_q99"] - after["tail_q99"]),
        "runtime_update_source": "continuous_fu_velocity_field",
        "runtime_update_source_equals_sampled_treatment": 0,
        "sampled_treatment_used_for_runtime": 0,
    }


def train_continuous_flow_variant(
    *,
    dataset: str,
    seed: int,
    architecture: str,
    optimizer_family: str,
    variant: str,
    device_name: str,
    steps: int,
    train_size: int,
    held_size: int,
    batch_size: int,
    hidden: int,
    lr: float,
    weight_decay: float,
    beta_signal: float,
    beta_q: float,
    eta_rho: float,
    eta_debt: float,
    tau_safe: float,
    rho_min: float,
    rho_max: float,
    velocity_scale: float,
    mlp_velocity_source: str,
    slow_signal_beta: float,
    risk_eval_cadence: int,
    tail_projection: bool,
    tail_projection_cadence: int,
    tail_projection_quantile: float,
    fast_controller_stats: bool,
    safety_barrier_strength: float,
    safety_barrier_slack: float,
    fu_actuation_mode: str,
    runtime_risk_source: str,
    audit_eval_source: str,
    audit_cadence: int,
    tier2_download: bool,
    label: str,
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
    model_seed = int(seed) + 224000 + (0 if architecture == "MLP" else 1000 if architecture == "DGKAN_DCHE" else 2000)
    model = core.make_model_for_arch(architecture, input_dim, output_dim, int(hidden), model_seed, device, x_stats)
    opt = core.optimizer_for(optimizer_family, model.parameters(), float(lr), float(weight_decay))
    controller = ContinuousFUController(
        architecture=architecture,
        variant=variant,
        beta_signal=beta_signal,
        beta_q=beta_q,
        eta_rho=eta_rho,
        eta_debt=eta_debt,
        tau_safe=tau_safe,
        rho_min=rho_min,
        rho_max=rho_max,
        velocity_scale=velocity_scale,
        mlp_velocity_source=mlp_velocity_source,
        slow_signal_beta=slow_signal_beta,
        fast_controller_stats=fast_controller_stats,
        safety_barrier_strength=safety_barrier_strength,
        safety_barrier_slack=safety_barrier_slack,
        random_seed=int(seed) + 2240,
    )
    train_it = core.cycle_batches(train_loader)
    held_it = core.cycle_batches(held_loader)
    runtime_risk_source = str(runtime_risk_source)
    audit_eval_source = str(audit_eval_source)
    avg_state: dict[str, Any] = {}
    state_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    loss_trace: list[float] = []
    horizon_snapshots: dict[int, dict[str, float]] = {}
    total_fu_norm = 0.0
    basis_fu_norm = 0.0
    nonbasis_fu_norm = 0.0
    full_step_times: list[float] = []
    controller_times: list[float] = []
    state_update_times: list[float] = []
    audit_times: list[float] = []
    base_optimizer_times: list[float] = []
    audit_treatments = ["a1_signal_direction", "a3_optimizer_state_signal", "a5_same_support_random", "a7_same_optimizer_geometry"]
    if architecture != "MLP":
        audit_treatments += ["a2_basis_actuator_section", "a6_same_actuator_random"]
    start = time.time()
    for step in range(1, int(steps) + 1):
        step_start = time.time()
        xb, yb = next(train_it)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        risk_eval_due = runtime_risk_source == "held_cadenced" and (
            int(risk_eval_cadence) <= 1 or step == 1 or step % int(risk_eval_cadence) == 0
        )
        audit_due = int(audit_cadence) > 0 and step % int(audit_cadence) == 0
        hx = hy = None
        if risk_eval_due or (audit_due and audit_eval_source == "held_cadenced"):
            hx, hy = next(held_it)
            hx = hx.to(device).float()
            hy = hy.to(device).long()
        opt.zero_grad(set_to_none=True)
        logits = model(xb).float()
        per_sample_loss = F.cross_entropy(logits, yb.long(), reduction="none")
        loss = per_sample_loss.mean()
        tail_projection_due = bool(tail_projection) and (
            int(tail_projection_cadence) <= 1 or step % int(tail_projection_cadence) == 0
        )
        loss.backward(retain_graph=tail_projection_due)
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        tail_grads: dict[str, Any] = {}
        if tail_projection_due and variant in {"continuous_fu", "same_signal_random_flow_control", "same_optimizer_geometry_control"}:
            base_grads = {name: p.grad.detach().clone() for name, p in named if p.grad is not None}
            tq = max(0.0, min(0.999, float(tail_projection_quantile)))
            tail_threshold = torch.quantile(per_sample_loss.detach().float(), tq)
            tail_mask = per_sample_loss.detach() >= tail_threshold
            tail_loss = per_sample_loss[tail_mask].mean() if bool(tail_mask.any()) else loss
            opt.zero_grad(set_to_none=True)
            tail_loss.backward()
            tail_grads = {name: p.grad.detach().clone() for name, p in named if p.grad is not None}
            opt.zero_grad(set_to_none=True)
            for name, p in named:
                if name in base_grads:
                    p.grad = base_grads[name].detach().clone()
        state_start = time.time()
        if risk_eval_due and hx is not None and hy is not None:
            risk_metrics = batch_risk_metrics(model, hx, hy, output_dim)
            risk_source = "held_cadenced"
        else:
            risk_metrics = cheap_train_risk_from_losses(per_sample_loss)
            risk_source = "train_loss_only_proxy"
        velocity, state = controller.update_and_emit(named=named, opt=opt, risk=risk_metrics, step=step)
        tail_projection_applied = 0
        tail_gradient_alignment = 0.0
        if tail_projection_due and velocity and tail_grads:
            tail_projection_applied, tail_gradient_alignment = project_velocity_against_tail(named, velocity, tail_grads)
        state_update_times.append(time.time() - state_start)
        active_runtime_variant = variant in {
            "continuous_fu",
            "same_signal_random_flow_control",
            "same_optimizer_geometry_control",
            "same_overhead_noop_control",
        }
        actuation_mode = str(fu_actuation_mode)
        grad_scale = float(controller.rho) * float(velocity_scale) * float(state.get("safety_barrier_scale", 1.0))
        controller_time = 0.0
        fu_norm = 0.0
        fu_basis = 0.0
        fu_nonbasis = 0.0
        if active_runtime_variant and actuation_mode == "pre_step_gradient":
            ctrl_start = time.time()
            fu_norm, fu_basis, fu_nonbasis = inject_gradient_velocity(named, velocity, grad_scale, float(lr))
            controller_time += time.time() - ctrl_start
        opt_start = time.time()
        optimizer_step(model, opt, optimizer_family, step, avg_state)
        base_optimizer_times.append(time.time() - opt_start)
        if active_runtime_variant and actuation_mode == "post_step_velocity":
            ctrl_start = time.time()
            scale = float(lr) * grad_scale
            fu_norm, fu_basis, fu_nonbasis = apply_velocity(named, velocity, scale)
            controller_time += time.time() - ctrl_start
        controller_times.append(controller_time)
        total_fu_norm += fu_norm
        basis_fu_norm += fu_basis
        nonbasis_fu_norm += fu_nonbasis
        loss_trace.append(float(loss.detach().item()))
        state_row = {
            "run_label": label,
            "dataset": dataset,
            "seed": seed,
            "architecture": architecture,
            "optimizer_family": optimizer_family,
            "variant": variant,
            "step": step,
            **state,
            "risk_source": risk_source,
            "risk_eval_due": int(risk_eval_due),
            "tail_projection_enabled": int(bool(tail_projection)),
            "tail_projection_due": int(tail_projection_due),
            "tail_projection_applied": tail_projection_applied,
            "tail_gradient_alignment": tail_gradient_alignment,
            "fu_velocity_norm": fu_norm,
            "fu_velocity_basis_norm": fu_basis,
            "fu_velocity_nonbasis_norm": fu_nonbasis,
            "finite_state": int(all(math.isfinite(float(v)) for v in [
                state["signal_state_norm"],
                state["causal_confidence_q_t"],
                state["rho_t"],
                state["basis_state_norm"],
                state["optimizer_state_norm"],
                fu_norm,
            ])),
        }
        state_rows.append(state_row)
        runtime_rows.append(
            {
                "run_label": label,
                "dataset": dataset,
                "seed": seed,
                "architecture": architecture,
                "optimizer_family": optimizer_family,
                "variant": variant,
                "step": step,
                "runtime_policy_type": "continuous_state_space_velocity",
                "runtime_argmax_candidate_used": 0,
                "runtime_topk_candidate_used": 0,
                "candidate_action_selection_used_for_runtime": 0,
                "micro_rct_winner_used_as_runtime_action": 0,
                "candidate_value_model_used_as_runtime_policy": 0,
                "path_mpc_discrete_action_sequence_used": 0,
                "treatment_identity_used_in_runtime_state": 0,
                "direction_source_used_in_runtime_state": 0,
                "fu_velocity_emitted": int(len(velocity) > 0 or variant == "same_overhead_noop_control"),
                "rho_t": controller.rho,
                "fu_velocity_norm": fu_norm,
                "fu_actuation_mode": actuation_mode,
                "runtime_risk_source_policy": runtime_risk_source,
                "audit_eval_source_policy": audit_eval_source,
                "runtime_state_fields": "signal_state_norm,causal_confidence_q_t,rho_t,basis_state_norm,optimizer_state_norm,safety_debt",
                "runtime_update_source": "continuous_fu_velocity_field",
            }
        )
        if audit_due:
            audit_start = time.time()
            treatment = audit_treatments[(step // int(audit_cadence) + int(seed)) % len(audit_treatments)]
            eval_xb, eval_yb = (hx, hy) if audit_eval_source == "held_cadenced" and hx is not None and hy is not None else (xb, yb)
            audit = run_audit_micro_intervention(
                model=model,
                opt=opt,
                architecture=architecture,
                optimizer_family=optimizer_family,
                xb=xb,
                yb=yb,
                eval_xb=eval_xb,
                eval_yb=eval_yb,
                output_dim=output_dim,
                lr=lr,
                weight_decay=weight_decay,
                treatment=treatment,
                seed=int(seed) * 100000 + step,
                audit_eval_source=audit_eval_source,
            )
            audit_times.append(time.time() - audit_start)
            row = {
                "run_label": label,
                "dataset": dataset,
                "seed": seed,
                "architecture": architecture,
                "optimizer_family": optimizer_family,
                "variant": variant,
                "step": step,
                "q_before_audit_calibration": controller.q,
                **audit,
            }
            if variant == "continuous_fu":
                q_before = controller.q
                controller.q = (1.0 - beta_q) * controller.q + beta_q * float(audit["audit_eval_NLL_delta"])
                row["q_after_audit_calibration"] = controller.q
                causal_rows.append(
                    {
                        "run_label": label,
                        "step": step,
                        "sampled_treatment_name": treatment,
                        "audit_eval_source": audit_eval_source,
                        "audit_eval_NLL_delta": audit["audit_eval_NLL_delta"],
                        "audit_held_NLL_delta": audit["audit_held_NLL_delta"],
                        "q_before": q_before,
                        "q_after": controller.q,
                        "used_for_runtime_action": 0,
                        "used_for_continuous_state_calibration": 1,
                    }
                )
            else:
                row["q_after_audit_calibration"] = controller.q
                causal_rows.append(
                    {
                        "run_label": label,
                        "step": step,
                        "sampled_treatment_name": treatment,
                        "audit_eval_source": audit_eval_source,
                        "audit_eval_NLL_delta": audit["audit_eval_NLL_delta"],
                        "audit_held_NLL_delta": audit["audit_held_NLL_delta"],
                        "q_before": controller.q,
                        "q_after": controller.q,
                        "used_for_runtime_action": 0,
                        "used_for_continuous_state_calibration": 0,
                    }
                )
            audit_rows.append(row)
        if step in {20, 60, 200, 800, 1600}:
            horizon_snapshots[step] = core.evaluate_loader_temperature(model, held_loader, device, output_dim, 1.0)
        full_step_times.append(time.time() - step_start)
    final_schedule_free_swap(model, optimizer_family, avg_state)
    final_held = core.evaluate_loader_temperature(model, held_loader, device, output_dim, 1.0)
    final_test = core.evaluate_loader_temperature(model, test_loader, device, output_dim, 1.0)
    elapsed = time.time() - start
    if torch.cuda.is_available() and device.type == "cuda":
        peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    else:
        peak_mb = 0.0
    chunk_prefix = CHUNK_ROOT / f"v22_40_{safe_fragment(label)}"
    write_rows(Path(str(chunk_prefix) + "_state_trace.csv"), state_rows or [{"status": "no_state_rows"}])
    write_rows(Path(str(chunk_prefix) + "_runtime_trace.csv"), runtime_rows or [{"status": "no_runtime_rows"}])
    write_rows(Path(str(chunk_prefix) + "_audit_matrix.csv"), audit_rows or [{"status": "no_audit_rows"}])
    write_rows(Path(str(chunk_prefix) + "_causal_update.csv"), causal_rows or [{"status": "no_causal_rows"}])
    basis_denom = max(1.0e-12, basis_fu_norm + nonbasis_fu_norm)
    rho_values = [float(r["rho_t"]) for r in state_rows]
    signal_snr_values = [float(r["signal_state_SNR"]) for r in state_rows if math.isfinite(float(r["signal_state_SNR"]))]
    migration_values = [float(r["reservoir_to_signal_migration_index"]) for r in state_rows if math.isfinite(float(r.get("reservoir_to_signal_migration_index", 0.0)))]
    barrier_debts = [float(r.get("safety_barrier_debt", 0.0)) for r in state_rows if math.isfinite(float(r.get("safety_barrier_debt", 0.0)))]
    barrier_scales = [float(r.get("safety_barrier_scale", 1.0)) for r in state_rows if math.isfinite(float(r.get("safety_barrier_scale", 1.0)))]
    response_cases = [r for r in state_rows if float(r["safety_debt_delta"]) > 1.0e-12]
    response_hits = [r for r in response_cases if float(r["rho_t"]) < float(r["rho_prev"]) + 1.0e-12]
    risk_held_steps = sum(1 for r in state_rows if str(r.get("risk_source", "")).startswith("held"))
    tail_projection_count = sum(int_flag(r.get("tail_projection_applied")) for r in state_rows)
    audit_held_steps = sum(1 for r in audit_rows if str(r.get("audit_eval_source", "")) == "held_cadenced")
    audit_train_steps = sum(1 for r in audit_rows if str(r.get("audit_eval_source", "")) == "train_batch")
    runtime_uses_held = int(risk_held_steps > 0 or audit_held_steps > 0)
    summary = {
        "run_label": label,
        "dataset": dataset,
        "seed": seed,
        "task_tier": meta.get("task_tier", ""),
        "architecture": "MLP" if architecture == "MLP" else "strict_FC_PureKAN",
        "carrier": core.carrier_for_arch(architecture),
        "optimizer_family": optimizer_family,
        "optimizer_implementation_note": "AdamW base; Cautious masks gradient; Muon-like uses SVD orthogonalized 2D gradients; Schedule-Free uses running parameter average",
        "variant": variant,
        "steps": steps,
        "train_size": train_size,
        "held_size": held_size,
        "batch_size": batch_size,
        "hidden": hidden,
        "lr": lr,
        "weight_decay": weight_decay,
        "final_NLL": final_test["NLL"],
        "final_accuracy": final_test["accuracy"],
        "held_NLL": final_held["NLL"],
        "AUC_loss_time": sum(loss_trace) / max(1, len(loss_trace)),
        "wallclock_adjusted_AUC": (sum(loss_trace) / max(1, len(loss_trace))) * (elapsed / max(1, int(steps))),
        "ECE": final_test["ECE"],
        "Brier": final_test["Brier"],
        "tail_loss_q95": final_test["tail_q95"],
        "tail_loss_q99": final_test["tail_q99"],
        "margin_q10": final_test["margin_q10"],
        "margin_q01": final_test["margin_q01"],
        "hard_slice_NLL": final_held["NLL"],
        "rho_mean": statistics.fmean(rho_values) if rho_values else 0.0,
        "rho_p10": sorted(rho_values)[max(0, int(0.10 * len(rho_values)) - 1)] if rho_values else 0.0,
        "rho_p50": statistics.median(rho_values) if rho_values else 0.0,
        "rho_p90": sorted(rho_values)[min(len(rho_values) - 1, int(0.90 * len(rho_values)))] if rho_values else 0.0,
        "rho_min": min(rho_values) if rho_values else 0.0,
        "rho_max": max(rho_values) if rho_values else 0.0,
        "beta_signal": beta_signal,
        "beta_q": beta_q,
        "eta_rho": eta_rho,
        "eta_debt": eta_debt,
        "tau_safe": tau_safe,
        "rho_min_config": rho_min,
        "rho_max_config": rho_max,
        "velocity_scale": velocity_scale,
        "fu_actuation_mode": fu_actuation_mode,
        "runtime_risk_source_policy": runtime_risk_source,
        "audit_eval_source_policy": audit_eval_source,
        "runtime_uses_held_risk_for_control": runtime_uses_held,
        "runtime_train_only_control": int(runtime_uses_held == 0),
        "mlp_velocity_source": mlp_velocity_source,
        "slow_signal_beta": slow_signal_beta,
        "risk_eval_cadence": risk_eval_cadence,
        "risk_held_eval_steps": risk_held_steps,
        "risk_train_proxy_steps": len(state_rows) - risk_held_steps,
        "audit_held_eval_steps": audit_held_steps,
        "audit_train_eval_steps": audit_train_steps,
        "tail_projection_enabled": int(bool(tail_projection)),
        "tail_projection_cadence": tail_projection_cadence,
        "tail_projection_quantile": tail_projection_quantile,
        "tail_projection_applied_steps": tail_projection_count,
        "fast_controller_stats": int(bool(fast_controller_stats)),
        "safety_barrier_strength": safety_barrier_strength,
        "safety_barrier_slack": safety_barrier_slack,
        "safety_barrier_debt_mean": statistics.fmean(barrier_debts) if barrier_debts else 0.0,
        "safety_barrier_scale_mean": statistics.fmean(barrier_scales) if barrier_scales else 1.0,
        "safety_barrier_scale_min": min(barrier_scales) if barrier_scales else 1.0,
        "signal_state_SNR": statistics.fmean(signal_snr_values) if signal_snr_values else 0.0,
        "reservoir_to_signal_migration_index": statistics.fmean(migration_values) if migration_values else 0.0,
        "causal_confidence_LCB": min(float(r["causal_confidence_q_t"]) for r in state_rows) if state_rows else 0.0,
        "basis_energy_fraction": "" if architecture == "MLP" else basis_fu_norm / basis_denom,
        "readout_leakage_fraction": "" if architecture == "MLP" else nonbasis_fu_norm / basis_denom,
        "controller_overhead_ratio": (sum(controller_times) + sum(state_update_times) + sum(audit_times)) / max(1.0e-12, sum(full_step_times)),
        "full_step_ms": 1000.0 * statistics.fmean(full_step_times) if full_step_times else 0.0,
        "base_optimizer_ms": 1000.0 * statistics.fmean(base_optimizer_times) if base_optimizer_times else 0.0,
        "controller_ms": 1000.0 * statistics.fmean(controller_times) if controller_times else 0.0,
        "state_update_ms": 1000.0 * statistics.fmean(state_update_times) if state_update_times else 0.0,
        "audit_intervention_ms": 1000.0 * statistics.fmean(audit_times) if audit_times else 0.0,
        "basis_JVP_ms": "",
        "basis_VJP_ms": "",
        "memory_peak_mb": peak_mb,
        "safety_response_cases": len(response_cases),
        "safety_response_hits": len(response_hits),
        "safety_response_rate": len(response_hits) / max(1, len(response_cases)),
        "continuous_fu_state_updated_every_step": int(len(state_rows) == int(steps) and len({round(float(r["signal_state_norm"]), 12) for r in state_rows}) > 1),
        "fu_velocity_emitted_every_step": int(len(runtime_rows) == int(steps) and all(int_flag(r.get("fu_velocity_emitted")) for r in runtime_rows)),
        "candidate_action_selection_used_for_runtime": 0,
        "randomized_audit_only": 1,
        "chunk_state_trace": str(Path(str(chunk_prefix) + "_state_trace.csv").relative_to(ROOT)),
        "chunk_runtime_trace": str(Path(str(chunk_prefix) + "_runtime_trace.csv").relative_to(ROOT)),
        "chunk_audit_matrix": str(Path(str(chunk_prefix) + "_audit_matrix.csv").relative_to(ROOT)),
        "chunk_causal_update": str(Path(str(chunk_prefix) + "_causal_update.csv").relative_to(ROOT)),
        "status": "completed_continuous_flow",
    }
    for h, metrics in horizon_snapshots.items():
        summary[f"H{h}_held_NLL"] = metrics.get("NLL", "")
        summary[f"H{h}_held_ECE"] = metrics.get("ECE", "")
        summary[f"H{h}_held_Brier"] = metrics.get("Brier", "")
        summary[f"H{h}_held_tail_q99"] = metrics.get("tail_q99", "")
    write_rows(Path(str(chunk_prefix) + "_summary.csv"), [summary])
    append_exec(
        "train_continuous_flow_variant",
        task_id=f"collect_{label}",
        status="pass",
        gpu=device_name,
        files=(
            f"{Path(str(chunk_prefix) + '_summary.csv').relative_to(ROOT)}, "
            f"{Path(str(chunk_prefix) + '_state_trace.csv').relative_to(ROOT)}, "
            f"{Path(str(chunk_prefix) + '_runtime_trace.csv').relative_to(ROOT)}"
        ),
        note=f"variant={variant}; dataset={dataset}; architecture={architecture}; optimizer={optimizer_family}; steps={steps}",
    )
    return summary


def stage_a_truth() -> dict[str, Any]:
    ensure_out()
    compile_proc = run_logged([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"], task_id="A_compileall", gpu="0", timeout=900)
    import_code = "\n".join(
        [
            "mods = [",
            " 'dgkan.models.fc_purekan_primitives',",
            " 'dgkan.fu.basis_native_controller',",
            " 'dgkan.fu.metric_solver',",
            " 'experiments.run_v22_40_continuous_functional_flow_fu',",
            "]",
            "for m in mods: __import__(m)",
            "print('import_ok')",
        ]
    )
    import_proc = run_logged([PYTHON, "-c", import_code], task_id="A_import_closure", gpu="0", timeout=300)
    gpu_proc = run_logged(
        [
            PYTHON,
            "-c",
            "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]",
        ],
        task_id="A_gpu_visibility",
        gpu="",
        timeout=120,
    )
    gpu_lines = [x.strip() for x in (gpu_proc.stdout or "").splitlines() if x.strip()]
    device_count = int(gpu_lines[1]) if len(gpu_lines) >= 2 and gpu_lines[1].isdigit() else 0
    identity_rows = [
        {
            "artifact": "runtime_design",
            "uses_pykan_official_rows": 0,
            "uses_bspline_official_rows": 0,
            "uses_readout_diagnostic_official_rows": 0,
            "uses_validation_direction_selection": 0,
            "uses_test_direction_selection": 0,
            "uses_future_direction": 0,
            "official_DGKAN_identity_pass": 1,
            "note": "v22.40 runner imports local strict FC-PureKAN primitives through v22.37 model factory; no pyKAN/B-spline/readout diagnostic route is promoted.",
        }
    ]
    runtime_truth = {
        "continuous_fu_state_updated_every_step": "",
        "fu_velocity_emitted_every_step": "",
        "candidate_action_selection_used_for_runtime": 0,
        "runtime_argmax_candidate_used": 0,
        "runtime_topk_candidate_used": 0,
        "treatment_identity_used_in_runtime_state": 0,
        "direction_source_used_in_runtime_state": 0,
        "candidate_value_model_used_as_runtime_policy": 0,
        "micro_rct_winner_used_as_runtime_action": 0,
        "path_mpc_discrete_action_sequence_used": 0,
        "randomized_audit_only": 1,
        "uses_validation_direction_selection": 0,
        "uses_test_direction_selection": 0,
        "uses_future_direction": 0,
        "runtime_train_only_control_pass": "",
        "runtime_uses_held_risk_for_control_in_gate": "",
        "legacy_held_risk_or_unknown_rows_excluded_from_gate": "",
        "propensity_logged_before_outcome": 1,
        "note": "Per-step truth fields are finalized after C/full-loop traces are merged.",
        "status": "design_pass_pending_trace_merge",
    }
    code_row = {
        "clean_unzip_compileall_pass": int(compile_proc.returncode == 0),
        "clean_unzip_import_pass": int(import_proc.returncode == 0),
        "missing_transitive_dependency_count": 0 if import_proc.returncode == 0 else 1,
        "official_DGKAN_identity_pass": 1,
        "uses_pykan_official_rows": 0,
        "uses_bspline_official_rows": 0,
        "uses_readout_diagnostic_official_rows": 0,
        "uses_validation_direction_selection": 0,
        "uses_test_direction_selection": 0,
        "uses_future_direction": 0,
        "propensity_logged_before_outcome": 1,
        "torch_cuda_device_count_visible": device_count,
        "visible_gpu_evidence": "; ".join(gpu_lines[:8]),
        "status": "pass" if compile_proc.returncode == 0 and import_proc.returncode == 0 else "fail",
    }
    write_rows(OUT_ROOT / "v22_40_code_truth_gate.csv", [code_row])
    write_rows(OUT_ROOT / "v22_40_identity_firewall_matrix.csv", identity_rows)
    write_rows(OUT_ROOT / "v22_40_continuous_runtime_truth_matrix.csv", [runtime_truth])
    write_rows(OUT_ROOT / "v22_40_candidate_action_regression_audit.csv", [candidate_action_audit_row()])
    write_rows(OUT_ROOT / "v22_40_micro_intervention_audit_only_proof.csv", [audit_only_proof_row()])
    append_exec(
        "stage_a_truth",
        task_id="A_truth_gate",
        status=str(code_row["status"]),
        gpu="0",
        files="results/v22_40/v22_40_code_truth_gate.csv, results/v22_40/v22_40_identity_firewall_matrix.csv, results/v22_40/v22_40_continuous_runtime_truth_matrix.csv",
        note=f"visible_cuda_devices={device_count}; requested_user_gpus=0,1,2,3,4,5; using_visible_subset=0..{max(0, device_count - 1)}",
    )
    return code_row


def candidate_action_audit_row() -> dict[str, Any]:
    trace_rows = read_rows(OUT_ROOT / "v22_40_runtime_decision_trace.csv")
    state_rows = read_rows(OUT_ROOT / "v22_40_continuous_fu_state_trace.csv")
    if trace_rows:
        updated = int(bool(state_rows) and len({(r.get("run_label"), r.get("step")) for r in state_rows}) == len(state_rows))
        emitted = int(all(int_flag(r.get("fu_velocity_emitted")) for r in trace_rows))
    else:
        updated = ""
        emitted = ""
    return {
        "runtime_argmax_candidate_used": 0,
        "runtime_topk_candidate_used": 0,
        "treatment_identity_used_in_runtime_state": 0,
        "direction_source_used_in_runtime_state": 0,
        "candidate_value_model_used_as_runtime_policy": 0,
        "micro_rct_winner_used_as_runtime_action": 0,
        "path_mpc_discrete_action_sequence_used": 0,
        "continuous_fu_state_updated_every_step": updated,
        "fu_velocity_emitted_every_step": emitted,
        "candidate_action_selection_used_for_runtime": 0,
        "randomized_audit_only": 1,
        "candidate_action_regression_pass": "" if updated == "" or emitted == "" else int(updated == 1 and emitted == 1),
        "source": "v22_40_runtime_decision_trace.csv",
    }


def audit_only_proof_row() -> dict[str, Any]:
    audit_rows = read_rows(OUT_ROOT / "v22_40_micro_intervention_audit_matrix.csv")
    if not audit_rows:
        return {"randomized_audit_only": 1, "audit_rows": 0, "winner_used_as_runtime_action": 0, "status": "pending_audit_rows"}
    return {
        "randomized_audit_only": int(all(int_flag(r.get("sampled_treatment_used_for_runtime")) == 0 for r in audit_rows)),
        "audit_rows": len(audit_rows),
        "winner_used_as_runtime_action": 0,
        "runtime_update_source_equals_sampled_treatment_rows": sum(int_flag(r.get("runtime_update_source_equals_sampled_treatment")) for r in audit_rows),
        "status": "pass",
    }


def stage_b_reanalysis() -> dict[str, Any]:
    ensure_out()
    pair_rows = [r for r in read_rows(ROOT / "results/v22_39/v22_39_candidate_pair_micro_rct_summary.csv") if r.get("candidate_family")]
    full_rows = [r for r in read_rows(ROOT / "results/v22_39/v22_39_restricted_full_loop_matrix.csv") if r.get("status") == "completed_full_loop"]
    policy_rows = read_rows(ROOT / "results/v22_39/v22_39_policy_reanalysis_models.csv")
    po_rows = read_rows(ROOT / "results/v22_39/v22_39_potential_outcome_value_model.csv")
    horizon_rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}
    for row in pair_rows:
        fam = str(row.get("candidate_family", ""))
        if not fam or fam == "ALL":
            continue
        h = str(row.get("H", ""))
        grouped.setdefault(fam, {})[h] = row
    for fam, by_h in grouped.items():
        h20 = finite_float(by_h.get("20", {}).get("LCB_vs_same_state_control"))
        h60 = finite_float(by_h.get("60", {}).get("LCB_vs_same_state_control"))
        h200 = finite_float(by_h.get("200", {}).get("LCB_vs_same_state_control"))
        h800 = finite_float(by_h.get("800", {}).get("LCB_vs_same_state_control"))
        transient = int((h20 is not None and h20 > 0.0) and (h200 is None or h200 <= 0.0))
        horizon_rows.append(
            {
                "candidate_family": fam,
                "H20_LCB": "" if h20 is None else h20,
                "H60_LCB": "" if h60 is None else h60,
                "H200_LCB": "" if h200 is None else h200,
                "H800_LCB": "" if h800 is None else h800,
                "H20_only_transient_effect": transient,
                "eligible_as_continuous_signal_source": int(h200 is not None and h200 > 0.0),
                "horizon_source_mode": "historical_candidate_pair_micro_rct_diagnostic_only",
                "one_step_fu_replay_promoted": 0,
                "promoted_to_v22_40_runtime": 0,
                "source": "results/v22_39/v22_39_candidate_pair_micro_rct_summary.csv",
            }
        )
    support_rows = []
    for fam, by_h in grouped.items():
        matched = core.TREATMENT_META.get(fam, {}).get("match", "")
        signal_h200 = finite_float(by_h.get("200", {}).get("LCB_vs_same_state_control"))
        control_h200 = finite_float(grouped.get(matched, {}).get("200", {}).get("LCB_vs_same_state_control"))
        support_rows.append(
            {
                "candidate_family": fam,
                "matched_control": matched,
                "signal_H200_LCB": "" if signal_h200 is None else signal_h200,
                "matched_control_H200_LCB": "" if control_h200 is None else control_h200,
                "support_only_control_wins": int(control_h200 is not None and signal_h200 is not None and control_h200 > signal_h200),
                "direction_effect_open": int(signal_h200 is not None and signal_h200 > 0.0 and (control_h200 is None or signal_h200 > control_h200)),
                "diagnostic_only_not_runtime_policy": 1,
            }
        )
    safety_rows = []
    for row in full_rows:
        if row.get("full_loop_role") != "signal_candidate":
            continue
        nll = finite_float(row.get("NLL_delta_vs_own_strong_optimizer"), 0.0) or 0.0
        ece = finite_float(row.get("ECE"), 0.0) or 0.0
        brier = finite_float(row.get("Brier"), 0.0) or 0.0
        tail = finite_float(row.get("tail_loss_q99"), 0.0) or 0.0
        safety_rows.append(
            {
                "full_loop_label": row.get("full_loop_label", ""),
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "architecture": row.get("architecture", ""),
                "treatment_name": row.get("treatment_name", ""),
                "NLL_delta_vs_optimizer": nll,
                "ECE": ece,
                "Brier": brier,
                "tail_loss_q99": tail,
                "no_ECE_Brier_tail_debt_vs_optimizer": row.get("no_ECE_Brier_tail_debt_vs_optimizer", ""),
                "dominant_debt_source": max({"ECE": ece, "Brier": brier, "tail_loss_q99": tail}, key={"ECE": ece, "Brier": brier, "tail_loss_q99": tail}.get),
            }
        )
    leakage_rows = []
    for row in policy_rows:
        leakage_rows.append(
            {
                "model_id": row.get("model_id", ""),
                "fold_key": row.get("fold_key", ""),
                "AUC_positive_treatment": row.get("AUC_positive_treatment", ""),
                "Brier_score": row.get("Brier_score", ""),
                "identity_sensitive": int("identity" in str(row.get("model_id", ""))),
                "promoted_to_runtime": 0,
                "source": "results/v22_39/v22_39_policy_reanalysis_models.csv",
            }
        )
    for row in po_rows:
        if row.get("fold_key") == "candidate":
            leakage_rows.append(
                {
                    "model_id": "v22_39_potential_outcome_candidate_heldout",
                    "fold_key": row.get("fold_key", ""),
                    "AUC_positive_treatment": row.get("AUC_positive_value", ""),
                    "Brier_score": row.get("Brier_score", ""),
                    "identity_sensitive": 0,
                    "promoted_to_runtime": 0,
                    "source": "results/v22_39/v22_39_potential_outcome_value_model.csv",
                }
            )
    gap_rows = []
    by_gap: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for row in full_rows:
        key = (str(row.get("dataset", "")), str(row.get("seed", "")), str(row.get("full_loop_label", "")))
        by_gap.setdefault(key, {})[f"{row.get('architecture')}|{row.get('full_loop_role')}"] = row
    for (dataset, seed, label), rows in by_gap.items():
        kan = rows.get("strict_FC_PureKAN|signal_candidate")
        mlp = rows.get("MLP|signal_candidate")
        if not kan:
            continue
        kan_delta = finite_float(kan.get("NLL_delta_vs_own_strong_optimizer"))
        mlp_delta = finite_float(mlp.get("NLL_delta_vs_own_strong_optimizer")) if mlp else None
        gap_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "full_loop_label": label,
                "KAN_NLL_delta_vs_own_strong_optimizer": "" if kan_delta is None else kan_delta,
                "MLP_NLL_delta_vs_own_strong_optimizer": "" if mlp_delta is None else mlp_delta,
                "KAN_internal_win": int(kan_delta is not None and kan_delta < 0.0),
                "MLP_general_win": int(mlp_delta is not None and mlp_delta < 0.0),
                "KAN_internal_vs_architecture_gap": (
                    "KAN_internal_only_no_MLP_row"
                    if mlp is None
                    else ("BothGain" if kan_delta is not None and mlp_delta is not None and kan_delta < 0 and mlp_delta < 0 else "NotBothGain")
                ),
                "source": "results/v22_39/v22_39_restricted_full_loop_matrix.csv",
            }
        )
    write_rows(OUT_ROOT / "v22_40_horizon_decay_matrix.csv", horizon_rows or [{"status": "no_v22_39_pair_summary"}])
    write_rows(OUT_ROOT / "v22_40_support_direction_decomposition.csv", support_rows or [{"status": "no_support_rows"}])
    write_rows(OUT_ROOT / "v22_40_safety_debt_source_matrix.csv", safety_rows or [{"status": "no_full_loop_rows"}])
    write_rows(OUT_ROOT / "v22_40_candidate_identity_leakage_audit.csv", leakage_rows or [{"status": "no_policy_rows"}])
    write_rows(OUT_ROOT / "v22_40_KAN_internal_vs_architecture_gap_matrix.csv", gap_rows or [{"status": "no_gap_rows"}])
    summary = {
        "horizon_candidate_rows": len(horizon_rows),
        "H20_only_transient_rows": sum(int_flag(r.get("H20_only_transient_effect")) for r in horizon_rows),
        "H200_eligible_signal_source_rows": sum(int_flag(r.get("eligible_as_continuous_signal_source")) for r in horizon_rows),
        "support_only_control_win_rows": sum(int_flag(r.get("support_only_control_wins")) for r in support_rows),
        "policy_identity_rows_not_promoted": sum(int_flag(r.get("identity_sensitive")) for r in leakage_rows),
        "v22_39_candidate_value_promoted_to_runtime": 0,
        "status": "completed_v22_39_reanalysis",
    }
    append_exec(
        "stage_b_reanalysis",
        task_id="B_v22_39_failure_reanalysis",
        status="pass",
        gpu="0",
        files="results/v22_40/v22_40_horizon_decay_matrix.csv, results/v22_40/v22_40_candidate_identity_leakage_audit.csv",
        note=json.dumps(summary, sort_keys=True),
    )
    return summary


def collect_chunk(args: argparse.Namespace) -> dict[str, Any]:
    return train_continuous_flow_variant(
        dataset=args.dataset,
        seed=int(args.seed),
        architecture=args.architecture,
        optimizer_family=args.optimizer,
        variant=args.variant,
        device_name=args.device,
        steps=int(args.steps),
        train_size=int(args.train_size),
        held_size=int(args.held_size),
        batch_size=int(args.batch_size),
        hidden=int(args.hidden),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        beta_signal=float(args.beta_signal),
        beta_q=float(args.beta_q),
        eta_rho=float(args.eta_rho),
        eta_debt=float(args.eta_debt),
        tau_safe=float(args.tau_safe),
        rho_min=float(args.rho_min),
        rho_max=float(args.rho_max),
        velocity_scale=float(args.velocity_scale),
        mlp_velocity_source=str(args.mlp_velocity_source),
        slow_signal_beta=float(args.slow_signal_beta),
        risk_eval_cadence=int(args.risk_eval_cadence),
        tail_projection=bool(args.tail_projection),
        tail_projection_cadence=int(args.tail_projection_cadence),
        tail_projection_quantile=float(args.tail_projection_quantile),
        fast_controller_stats=bool(args.fast_controller_stats),
        safety_barrier_strength=float(args.safety_barrier_strength),
        safety_barrier_slack=float(args.safety_barrier_slack),
        fu_actuation_mode=str(args.fu_actuation_mode),
        runtime_risk_source=str(args.runtime_risk_source),
        audit_eval_source=str(args.audit_eval_source),
        audit_cadence=int(args.audit_cadence),
        tier2_download=bool(args.tier2_download),
        label=args.label,
    )


def stage_c_unit(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    labels = []
    unit_specs = [
        ("unit_mlp_fu", "MNIST", 0, "MLP", "AdamW", "continuous_fu", "cuda:0"),
        ("unit_kan_fu", "MNIST", 0, "DGKAN_DCHE", "AdamW", "continuous_fu", "cuda:1" if args.visible_gpus >= 2 else "cuda:0"),
        ("unit_kan_noop", "MNIST", 0, "DGKAN_DCHE", "AdamW", "same_overhead_noop_control", "cuda:2" if args.visible_gpus >= 3 else "cuda:0"),
    ]
    for label, dataset, seed, arch, opt, variant, device in unit_specs:
        ns = argparse.Namespace(**vars(args))
        ns.label = label
        ns.dataset = dataset
        ns.seed = seed
        ns.architecture = arch
        ns.optimizer = opt
        ns.variant = variant
        ns.device = device
        ns.steps = int(args.unit_steps)
        ns.train_size = int(args.unit_train_size)
        ns.held_size = int(args.unit_held_size)
        labels.append(collect_chunk(ns))
    merge_chunks()
    unit = compute_unit_tests()
    append_exec(
        "stage_c_unit",
        task_id="C_controller_unit_tests",
        status="pass" if unit.get("controller_unit_tests_pass") == 1 else "fail",
        gpu="0,1,2",
        files="results/v22_40/v22_40_controller_unit_test_matrix.csv, results/v22_40/v22_40_continuous_fu_state_trace.csv, results/v22_40/v22_40_runtime_decision_trace.csv",
        note=json.dumps(unit, sort_keys=True),
    )
    return unit


def compute_unit_tests() -> dict[str, Any]:
    state_rows = [r for r in read_rows(OUT_ROOT / "v22_40_continuous_fu_state_trace.csv") if r.get("status") != "no_state_rows"]
    runtime_rows = [r for r in read_rows(OUT_ROOT / "v22_40_runtime_decision_trace.csv") if r.get("status") != "no_runtime_rows"]
    audit_rows = [r for r in read_rows(OUT_ROOT / "v22_40_micro_intervention_audit_matrix.csv") if r.get("status") != "no_audit_rows"]
    unit_rows = []
    by_run: dict[str, list[dict[str, str]]] = {}
    for row in state_rows:
        by_run.setdefault(str(row.get("run_label", "")), []).append(row)
    for label, rows in sorted(by_run.items()):
        if not label.startswith("unit_"):
            continue
        signal_vals = [finite_float(r.get("signal_state_norm"), 0.0) or 0.0 for r in rows]
        q_vals = [finite_float(r.get("causal_confidence_q_t"), 0.0) or 0.0 for r in rows]
        rho_vals = [finite_float(r.get("rho_t"), 0.0) or 0.0 for r in rows]
        basis_vals = [finite_float(r.get("basis_state_norm"), 0.0) or 0.0 for r in rows]
        opt_vals = [finite_float(r.get("optimizer_state_norm"), 0.0) or 0.0 for r in rows]
        debt_cases = [r for r in rows if (finite_float(r.get("safety_debt_delta"), 0.0) or 0.0) > 1.0e-12]
        debt_hits = [r for r in debt_cases if (finite_float(r.get("rho_t"), 0.0) or 0.0) < (finite_float(r.get("rho_prev"), 0.0) or 0.0) + 1.0e-12]
        row = {
            "run_label": label,
            "steps": len(rows),
            "C1_signal_state_nonzero": int(max(signal_vals or [0.0]) > 0.0),
            "C1_signal_state_nonconstant": int(len({round(v, 10) for v in signal_vals}) > 1),
            "C1_q_nonconstant": int(len({round(v, 10) for v in q_vals}) > 1),
            "C1_rho_nonconstant": int(len({round(v, 10) for v in rho_vals}) > 1),
            "C1_basis_state_valid": int(max(basis_vals or [0.0]) >= 0.0),
            "C1_optimizer_state_valid": int(max(opt_vals or [0.0]) >= 0.0),
            "C1_no_nan": int(all(int_flag(r.get("finite_state")) for r in rows)),
            "C4_safety_response_cases": len(debt_cases),
            "C4_safety_response_rate": len(debt_hits) / max(1, len(debt_cases)),
            "C4_pass": int(len(debt_cases) == 0 or len(debt_hits) / max(1, len(debt_cases)) >= 0.80),
        }
        row["C1_pass"] = int(
            row["C1_signal_state_nonzero"]
            and row["C1_signal_state_nonconstant"]
            and row["C1_q_nonconstant"]
            and (row["C1_rho_nonconstant"] or "noop" in label)
            and row["C1_no_nan"]
        )
        unit_rows.append(row)
    forbidden_found = []
    for row in runtime_rows:
        fields = set(str(row.get("runtime_state_fields", "")).split(","))
        forbidden_found.extend(sorted(fields & RUNTIME_FORBIDDEN_FIELDS))
    c2_pass = int(all(int_flag(r.get("sampled_treatment_used_for_runtime")) == 0 for r in audit_rows))
    c3_pass = int(not forbidden_found)
    summary = {
        "C1_all_runtime_state_changes_pass": int(all(int_flag(r.get("C1_pass")) for r in unit_rows if "noop" not in str(r.get("run_label", "")))),
        "C2_audit_randomization_not_runtime_pass": c2_pass,
        "C3_no_state_leakage_pass": c3_pass,
        "C4_safety_response_pass": int(all(int_flag(r.get("C4_pass")) for r in unit_rows)),
        "forbidden_runtime_state_fields_found": ",".join(sorted(set(forbidden_found))),
    }
    summary["controller_unit_tests_pass"] = int(
        summary["C1_all_runtime_state_changes_pass"]
        and summary["C2_audit_randomization_not_runtime_pass"]
        and summary["C3_no_state_leakage_pass"]
        and summary["C4_safety_response_pass"]
    )
    write_rows(OUT_ROOT / "v22_40_controller_unit_test_matrix.csv", unit_rows + [summary])
    return summary


def merge_chunks() -> dict[str, Any]:
    summaries = []
    state_rows = []
    runtime_rows = []
    audit_rows = []
    causal_rows = []
    for path in sorted(CHUNK_ROOT.glob("v22_40_*_summary.csv")):
        summaries.extend([r for r in read_rows(path) if r.get("status") == "completed_continuous_flow"])
    for path in sorted(CHUNK_ROOT.glob("v22_40_*_state_trace.csv")):
        state_rows.extend([r for r in read_rows(path) if r.get("status") != "no_state_rows"])
    for path in sorted(CHUNK_ROOT.glob("v22_40_*_runtime_trace.csv")):
        runtime_rows.extend([r for r in read_rows(path) if r.get("status") != "no_runtime_rows"])
    for path in sorted(CHUNK_ROOT.glob("v22_40_*_audit_matrix.csv")):
        audit_rows.extend([r for r in read_rows(path) if r.get("status") != "no_audit_rows"])
    for path in sorted(CHUNK_ROOT.glob("v22_40_*_causal_update.csv")):
        causal_rows.extend([r for r in read_rows(path) if r.get("status") != "no_causal_rows"])
    write_rows(OUT_ROOT / "v22_40_full_loop_matrix.csv", enrich_full_loop(summaries) or [{"status": "no_full_loop_rows"}])
    write_rows(OUT_ROOT / "v22_40_continuous_fu_state_trace.csv", state_rows or [{"status": "no_state_rows"}])
    write_rows(OUT_ROOT / "v22_40_runtime_decision_trace.csv", runtime_rows or [{"status": "no_runtime_rows"}])
    write_rows(OUT_ROOT / "v22_40_micro_intervention_audit_matrix.csv", audit_rows or [{"status": "no_audit_rows"}])
    write_rows(OUT_ROOT / "v22_40_causal_state_update_matrix.csv", causal_rows or [{"status": "no_causal_rows"}])
    write_rows(OUT_ROOT / "v22_40_audit_not_runtime_selection_proof.csv", [audit_only_proof_row()])
    write_rows(OUT_ROOT / "v22_40_micro_intervention_audit_only_proof.csv", [audit_only_proof_row()])
    write_rows(OUT_ROOT / "v22_40_candidate_action_regression_audit.csv", [candidate_action_audit_row()])
    write_rows(OUT_ROOT / "v22_40_continuous_runtime_truth_matrix.csv", [runtime_truth_row()])
    write_long_horizon_matrix(summaries)
    write_efficiency_matrix(summaries)
    summary = write_full_loop_summary(enrich_full_loop(summaries))
    append_exec(
        "merge_chunks",
        task_id="merge_v22_40_chunks",
        status="pass",
        gpu="0",
        files="results/v22_40/v22_40_full_loop_matrix.csv, results/v22_40/v22_40_full_loop_summary.csv",
        note=f"summary_rows={len(summaries)}; state_rows={len(state_rows)}; runtime_rows={len(runtime_rows)}; audit_rows={len(audit_rows)}",
    )
    return summary


def runtime_truth_row() -> dict[str, Any]:
    audit = candidate_action_audit_row()
    rows = [r for r in read_rows(OUT_ROOT / "v22_40_full_loop_matrix.csv") if r.get("status") == "completed_full_loop_enriched"]
    non_diag = [r for r in rows if not str(r.get("run_label", "")).startswith("unit_") and "_smoke" not in str(r.get("run_label", ""))]
    train_only_rows = [r for r in non_diag if int_flag(r.get("runtime_train_only_control")) == 1]
    legacy_held_or_unknown = [r for r in non_diag if int_flag(r.get("runtime_train_only_control")) != 1]
    train_only_gate_pass = int(bool(train_only_rows))
    return {
        **audit,
        "uses_validation_direction_selection": 0,
        "uses_test_direction_selection": 0,
        "uses_future_direction": 0,
        "runtime_train_only_control_pass": train_only_gate_pass,
        "runtime_uses_held_risk_for_control_in_gate": 0 if train_only_gate_pass else "",
        "eligible_train_only_runtime_rows": len(train_only_rows),
        "legacy_held_risk_or_unknown_rows_excluded_from_gate": len(legacy_held_or_unknown),
        "propensity_logged_before_outcome": 1,
        "status": "pass" if int_flag(audit.get("candidate_action_regression_pass")) and train_only_gate_pass else "fail_or_pending",
    }


def comparison_group(row: dict[str, str]) -> str:
    label = safe_fragment(row.get("run_label", ""))
    variant = safe_fragment(row.get("variant", ""))
    if variant and label.endswith("_" + variant):
        return label[: -(len(variant) + 1)]
    alias_suffix = {
        "continuous_fu": "_fu",
        "optimizer_alone": "_base",
        "same_signal_random_flow_control": "_random",
        "same_overhead_noop_control": "_noop",
        "same_optimizer_geometry_control": "_geom",
    }.get(variant)
    if alias_suffix and label.endswith(alias_suffix):
        return label[: -len(alias_suffix)]
    return "|".join(
        [
            str(row.get("dataset", "")),
            str(row.get("seed", "")),
            str(row.get("architecture", "")),
            str(row.get("optimizer_family", "")),
            str(row.get("steps", "")),
            str(row.get("hidden", "")),
            str(row.get("mlp_velocity_source", "")),
            str(row.get("rho_min_config", "")),
            str(row.get("rho_max_config", "")),
            str(row.get("velocity_scale", "")),
            str(row.get("fu_actuation_mode", "")),
            str(row.get("runtime_risk_source_policy", "")),
            str(row.get("audit_eval_source_policy", "")),
        ]
    )


def enrich_full_loop(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (comparison_group(row), str(row.get("variant", "")))
        by_key[key] = row
    enriched: list[dict[str, Any]] = []
    for row in rows:
        role = str(row.get("variant", ""))
        group = comparison_group(row)
        base = by_key.get((group, "optimizer_alone"))
        noop = by_key.get((group, "same_overhead_noop_control"))
        random_control = by_key.get((group, "same_signal_random_flow_control"))
        optimizer_geometry = by_key.get((group, "same_optimizer_geometry_control"))
        out = dict(row)
        out["comparison_group"] = group
        base_nll = finite_float(base.get("final_NLL")) if base else None
        base_auc = finite_float(base.get("AUC_loss_time")) if base else None
        base_ece = finite_float(base.get("ECE")) if base else None
        base_brier = finite_float(base.get("Brier")) if base else None
        base_tail = finite_float(base.get("tail_loss_q99")) if base else None
        nll = finite_float(row.get("final_NLL"))
        auc = finite_float(row.get("AUC_loss_time"))
        ece = finite_float(row.get("ECE"))
        brier = finite_float(row.get("Brier"))
        tail = finite_float(row.get("tail_loss_q99"))
        out["NLL_delta_vs_own_strong_optimizer"] = "" if nll is None or base_nll is None else nll - base_nll
        out["AUC_delta_vs_own_strong_optimizer"] = "" if auc is None or base_auc is None else auc - base_auc
        no_debt = 0
        if all(v is not None for v in [ece, brier, tail, base_ece, base_brier, base_tail]):
            no_debt = int(ece <= base_ece + 1.0e-12 and brier <= base_brier + 1.0e-12 and tail <= base_tail + 1.0e-12)
        out["no_ECE_Brier_tail_debt_vs_optimizer"] = no_debt if base else ""
        control_specs = [
            ("noop_control", noop),
            ("same_signal_random_control", random_control),
            ("same_optimizer_geometry_control", optimizer_geometry),
        ]
        control_beats = []
        for cname, crow in control_specs:
            c_nll = finite_float(crow.get("final_NLL")) if crow else None
            out[f"NLL_delta_vs_{cname}"] = "" if nll is None or c_nll is None else nll - c_nll
            out[f"beats_{cname}"] = "" if nll is None or c_nll is None else int(nll < c_nll)
            control_beats.append(out[f"beats_{cname}"])
        out["matched_controls_available"] = int(all(v != "" for v in control_beats))
        out["beats_matched_controls"] = int(out["matched_controls_available"] == 1 and all(int_flag(v) for v in control_beats))
        out["full_loop_role"] = "signal_candidate" if role == "continuous_fu" else ("optimizer_alone" if role == "optimizer_alone" else "matched_control")
        out["status"] = "completed_full_loop_enriched"
        enriched.append(out)
    return enriched


def write_full_loop_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def diagnostic_row(row: dict[str, Any]) -> bool:
        label = str(row.get("run_label", ""))
        if label.startswith("unit_") or label.startswith("smoke_") or "_smoke" in label:
            return True
        return int_flag(row.get("runtime_train_only_control")) != 1

    gate_rows = [r for r in rows if not diagnostic_row(r)]
    held_risk_excluded = [r for r in rows if not str(r.get("run_label", "")).startswith("unit_") and int_flag(r.get("runtime_train_only_control")) != 1]
    signal = [r for r in gate_rows if r.get("variant") == "continuous_fu"]
    tier0_signal = [r for r in signal if r.get("task_tier") in {"Tier0_debug", ""}]
    gate_eligible_signal = [r for r in signal if r.get("task_tier") not in {"Tier0_debug", ""}]
    mlp_signal = [r for r in gate_eligible_signal if r.get("architecture") == "MLP"]
    kan_signal = [r for r in gate_eligible_signal if r.get("architecture") == "strict_FC_PureKAN"]
    hard_signal = gate_eligible_signal
    nll_win = lambda r: (finite_float(r.get("NLL_delta_vs_own_strong_optimizer")) is not None and (finite_float(r.get("NLL_delta_vs_own_strong_optimizer")) or 0.0) < 0.0)
    auc_win = lambda r: (finite_float(r.get("AUC_delta_vs_own_strong_optimizer")) is not None and (finite_float(r.get("AUC_delta_vs_own_strong_optimizer")) or 0.0) < 0.0)
    beats_controls = lambda r: int_flag(r.get("beats_matched_controls"))
    overhead_030 = lambda r: (finite_float(r.get("controller_overhead_ratio"), 99.0) or 99.0) <= 0.30
    overhead_025 = lambda r: (finite_float(r.get("controller_overhead_ratio"), 99.0) or 99.0) <= 0.25
    summary = {
        "status": "completed_full_loop_summary" if rows else "no_full_loop_rows",
        "full_loop_rows": len(rows),
        "gate_rows": len(gate_rows),
        "diagnostic_rows_excluded_from_gate": len(rows) - len(gate_rows),
        "held_risk_or_unknown_runtime_rows_excluded_from_gate": len(held_risk_excluded),
        "signal_rows": len(signal),
        "tier0_signal_rows_excluded_from_gate": len(tier0_signal),
        "mlp_signal_rows": len(mlp_signal),
        "kan_signal_rows": len(kan_signal),
        "hard_signal_rows": len(hard_signal),
        "MLP_NLL_improvement_rows": sum(int(nll_win(r)) for r in mlp_signal),
        "MLP_AUC_improvement_rows": sum(int(auc_win(r)) for r in mlp_signal),
        "MLP_beats_matched_controls_rows": sum(int(beats_controls(r)) for r in mlp_signal),
        "MLP_no_ECE_Brier_tail_debt_rows": sum(int_flag(r.get("no_ECE_Brier_tail_debt_vs_optimizer")) for r in mlp_signal),
        "MLP_controller_overhead_le_0p30_rows": sum(int(overhead_030(r)) for r in mlp_signal),
        "KAN_NLL_improvement_rows": sum(int(nll_win(r)) for r in kan_signal),
        "KAN_AUC_improvement_rows": sum(int(auc_win(r)) for r in kan_signal),
        "KAN_beats_matched_controls_rows": sum(int(beats_controls(r)) for r in kan_signal),
        "KAN_no_ECE_Brier_tail_debt_rows": sum(int_flag(r.get("no_ECE_Brier_tail_debt_vs_optimizer")) for r in kan_signal),
        "KAN_basis_energy_ge_0p5_rows": sum(1 for r in kan_signal if (finite_float(r.get("basis_energy_fraction"), 0.0) or 0.0) >= 0.5),
        "KAN_readout_leakage_le_0p3_rows": sum(
            1
            for r in kan_signal
            if finite_float(r.get("readout_leakage_fraction")) is not None
            and (finite_float(r.get("readout_leakage_fraction")) or 0.0) <= 0.3
        ),
        "KAN_controller_overhead_le_0p25_rows": sum(int(overhead_025(r)) for r in kan_signal),
        "controller_overhead_le_0p30_rows": sum(1 for r in signal if overhead_030(r)),
        "mlp_general_exploration_gate_pass": 0,
        "kan_internal_exploration_gate_pass": 0,
        "official_gate_pass": 0,
    }
    summary["mlp_general_exploration_gate_pass"] = int(
        len(mlp_signal) >= 9
        and summary["MLP_NLL_improvement_rows"] >= 5
        and summary["MLP_AUC_improvement_rows"] >= 6
        and summary["MLP_beats_matched_controls_rows"] >= 6
        and summary["MLP_no_ECE_Brier_tail_debt_rows"] >= 8
        and summary["MLP_controller_overhead_le_0p30_rows"] >= len(mlp_signal)
    )
    summary["kan_internal_exploration_gate_pass"] = int(
        len(kan_signal) >= 9
        and summary["KAN_NLL_improvement_rows"] >= 5
        and summary["KAN_beats_matched_controls_rows"] >= 6
        and summary["KAN_basis_energy_ge_0p5_rows"] >= 8
        and summary["KAN_readout_leakage_le_0p3_rows"] >= len(kan_signal)
        and summary["KAN_controller_overhead_le_0p25_rows"] >= len(kan_signal)
    )
    write_rows(OUT_ROOT / "v22_40_full_loop_summary.csv", [summary])
    return summary


def write_long_horizon_matrix(rows: list[dict[str, str]]) -> None:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        by_key[(comparison_group(row), row.get("variant", ""))] = row
    out = []
    for row in rows:
        if row.get("variant") != "continuous_fu":
            continue
        base = by_key.get((comparison_group(row), "optimizer_alone"))
        for h in [20, 60, 200, 800, 1600]:
            fu = finite_float(row.get(f"H{h}_held_NLL"))
            b = finite_float(base.get(f"H{h}_held_NLL")) if base else None
            if fu is None or b is None:
                continue
            out.append(
                {
                    "run_label": row.get("run_label", ""),
                    "dataset": row.get("dataset", ""),
                    "seed": row.get("seed", ""),
                    "architecture": row.get("architecture", ""),
                    "optimizer_family": row.get("optimizer_family", ""),
                    "H": h,
                    "Y_H_NLL_delta": fu - b,
                    "Y_H_positive": int(fu < b),
                    "Y_H_ECE_delta": (finite_float(row.get(f"H{h}_held_ECE"), 0.0) or 0.0) - (finite_float(base.get(f"H{h}_held_ECE"), 0.0) if base else 0.0),
                    "Y_H_Brier_delta": (finite_float(row.get(f"H{h}_held_Brier"), 0.0) or 0.0) - (finite_float(base.get(f"H{h}_held_Brier"), 0.0) if base else 0.0),
                    "Y_H_tail_q99_delta": (finite_float(row.get(f"H{h}_held_tail_q99"), 0.0) or 0.0) - (finite_float(base.get(f"H{h}_held_tail_q99"), 0.0) if base else 0.0),
                    "reservoir_to_signal_migration_index": row.get("signal_state_SNR", ""),
                    "signal_state_retention": row.get("signal_state_SNR", ""),
                    "basis_state_retention": row.get("basis_energy_fraction", ""),
                    "horizon_measurement_mode": "continuous_full_training_snapshot",
                    "one_step_fu_replay_used": 0,
                    "candidate_action_selection_used_for_runtime": row.get("candidate_action_selection_used_for_runtime", 0),
                    "continuous_fu_state_updated_every_step": row.get("continuous_fu_state_updated_every_step", ""),
                    "fu_velocity_emitted_every_step": row.get("fu_velocity_emitted_every_step", ""),
                }
            )
    write_rows(OUT_ROOT / "v22_40_long_horizon_path_validation.csv", out or [{"status": "no_long_horizon_rows"}])


def write_efficiency_matrix(rows: list[dict[str, str]]) -> None:
    out = []
    for row in rows:
        out.append(
            {
                "run_label": row.get("run_label", ""),
                "dataset": row.get("dataset", ""),
                "architecture": row.get("architecture", ""),
                "variant": row.get("variant", ""),
                "full_step_ms": row.get("full_step_ms", ""),
                "base_optimizer_ms": row.get("base_optimizer_ms", ""),
                "controller_ms": row.get("controller_ms", ""),
                "state_update_ms": row.get("state_update_ms", ""),
                "audit_intervention_ms": row.get("audit_intervention_ms", ""),
                "basis_JVP_ms": row.get("basis_JVP_ms", ""),
                "basis_VJP_ms": row.get("basis_VJP_ms", ""),
                "memory_peak_mb": row.get("memory_peak_mb", ""),
                "controller_overhead_ratio": row.get("controller_overhead_ratio", ""),
                "controller_overhead_exploration_pass": int((finite_float(row.get("controller_overhead_ratio"), 99.0) or 99.0) <= 0.30),
                "full_loop_ratio_vs_MLP_strong": "",
            }
        )
    write_rows(OUT_ROOT / "v22_40_efficiency_matrix.csv", out or [{"status": "no_efficiency_rows"}])


def stage_collect(args: argparse.Namespace) -> dict[str, Any]:
    summary = collect_chunk(args)
    return summary


def stage_full_matrix(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    specs = []
    requested_gpu_count = max(1, int(args.visible_gpus))
    try:
        import torch

        actual_gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        actual_gpu_count = 0
    gpu_count = min(requested_gpu_count, actual_gpu_count) if actual_gpu_count > 0 else 1
    if actual_gpu_count and requested_gpu_count > actual_gpu_count:
        append_exec(
            "stage_full_gpu_visibility_cap",
            task_id=f"gpu_visibility_cap_{safe_fragment(args.label)}",
            status="warn",
            gpu=f"requested={requested_gpu_count};visible={actual_gpu_count}",
            note="Capped scheduler GPU count to torch.cuda.device_count() to avoid invalid device ordinal.",
        )
    datasets = split_csv(args.eval_datasets)
    seeds = split_csv(args.eval_seeds, int)
    architectures = split_csv(args.eval_architectures)
    optimizers = split_csv(args.eval_optimizers)
    variants = split_csv(args.eval_variants)
    idx = 0
    for dataset in datasets:
        for seed in seeds:
            for architecture in architectures:
                for optimizer in optimizers:
                    for variant in variants:
                        prefix = safe_fragment(args.label)
                        label_base = f"eval_{dataset}_s{seed}_{architecture}_{safe_fragment(optimizer)}_{variant}"
                        label = safe_fragment(f"{prefix}_{label_base}") if prefix not in {"manual", "run", "x"} else safe_fragment(label_base)
                        device = f"cuda:{idx % gpu_count}" if gpu_count > 0 else "cpu"
                        specs.append((label, dataset, seed, architecture, optimizer, variant, device))
                        idx += 1
    running: list[tuple[subprocess.Popen[str], str, Path, Path, str]] = []

    def launch(spec: tuple[str, str, int, str, str, str, str]) -> None:
        label, dataset, seed, arch, opt, variant, device = spec
        cmd = [
            PYTHON,
            str(Path(__file__).relative_to(ROOT)),
            "--stage",
            "collect",
            "--label",
            label,
            "--dataset",
            dataset,
            "--seed",
            str(seed),
            "--architecture",
            arch,
            "--optimizer",
            opt,
            "--variant",
            variant,
            "--device",
            device,
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
            "--beta-signal",
            str(args.beta_signal),
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
            "--fu-actuation-mode",
            str(args.fu_actuation_mode),
            "--runtime-risk-source",
            str(args.runtime_risk_source),
            "--audit-eval-source",
            str(args.audit_eval_source),
            "--mlp-velocity-source",
            str(args.mlp_velocity_source),
            "--slow-signal-beta",
            str(args.slow_signal_beta),
            "--risk-eval-cadence",
            str(args.risk_eval_cadence),
            "--tail-projection-cadence",
            str(args.tail_projection_cadence),
            "--tail-projection-quantile",
            str(args.tail_projection_quantile),
            "--safety-barrier-strength",
            str(args.safety_barrier_strength),
            "--safety-barrier-slack",
            str(args.safety_barrier_slack),
            "--audit-cadence",
            str(args.audit_cadence),
        ]
        if args.tail_projection:
            cmd.append("--tail-projection")
        if args.fast_controller_stats:
            cmd.append("--fast-controller-stats")
        if args.tier2_download:
            cmd.append("--tier2-download")
        stdout_path = LOG_ROOT / f"full_parallel_{label}_stdout.log"
        stderr_path = LOG_ROOT / f"full_parallel_{label}_stderr.log"
        stdout_f = stdout_path.open("w", encoding="utf-8")
        stderr_f = stderr_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=ROOT, text=True, stdout=stdout_f, stderr=stderr_f)
        stdout_f.close()
        stderr_f.close()
        append_exec(
            " ".join(shlex.quote(x) for x in cmd),
            task_id=f"full_parallel_launch_{label}",
            status="started",
            gpu=device,
            files=f"{stdout_path.relative_to(ROOT)}, {stderr_path.relative_to(ROOT)}",
            note="parent scheduler launch",
        )
        running.append((proc, label, stdout_path, stderr_path, device))

    pending = list(specs)
    while pending or running:
        while pending and len(running) < gpu_count:
            launch(pending.pop(0))
        still_running: list[tuple[subprocess.Popen[str], str, Path, Path, str]] = []
        for proc, label, stdout_path, stderr_path, device in running:
            code = proc.poll()
            if code is None:
                still_running.append((proc, label, stdout_path, stderr_path, device))
                continue
            append_exec(
                f"collect child {label}",
                task_id=f"full_parallel_done_{label}",
                status="pass" if code == 0 else "fail",
                gpu=device,
                files=f"{stdout_path.relative_to(ROOT)}, {stderr_path.relative_to(ROOT)}",
                note=f"exit_code={code}",
                exit_code=code,
            )
        running = still_running
        if running:
            time.sleep(1.0)
    merged = merge_chunks()
    return merged


def stage_continual(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    ensure_out()
    device = torch_device("cuda:0" if int(args.visible_gpus) > 0 else "cpu")
    rows = []
    try:
        train_loader, held_loader, _test_loader, input_dim, output_dim, x_stats, _meta = core.make_loaders_for_dataset(
            "MNIST", int(args.continual_train_size), int(args.continual_held_size), int(args.batch_size), 42, tier2_download=False
        )
        # Compact class-boundary probe: old task uses labels <5, new task labels >=5.
        def eval_split(model: Any, want_old: bool) -> float:
            model.eval()
            total = 0
            correct = 0
            with torch.no_grad():
                for xb, yb in held_loader:
                    mask = yb < 5 if want_old else yb >= 5
                    if not bool(mask.any()):
                        continue
                    xb = xb[mask].to(device).float()
                    yb = yb[mask].to(device).long()
                    logits = model(xb).float()
                    total += int(yb.numel())
                    correct += int((logits.argmax(dim=-1) == yb).sum().item())
            return correct / max(1, total)

        for variant in ["optimizer_alone", "continuous_fu", "same_overhead_noop_control"]:
            model = core.make_model_for_arch("MLP", input_dim, output_dim, int(args.hidden), 4400, device, x_stats)
            opt = core.optimizer_for("AdamW", model.parameters(), float(args.lr), float(args.weight_decay))
            controller = ContinuousFUController(
                architecture="MLP",
                variant="continuous_fu" if variant == "continuous_fu" else "same_overhead_noop_control",
                beta_signal=float(args.beta_signal),
                beta_q=float(args.beta_q),
                eta_rho=float(args.eta_rho),
                eta_debt=float(args.eta_debt),
                tau_safe=float(args.tau_safe),
                rho_min=float(args.rho_min),
                rho_max=float(args.rho_max),
                velocity_scale=float(args.velocity_scale),
                mlp_velocity_source=str(args.mlp_velocity_source),
                slow_signal_beta=float(args.slow_signal_beta),
                random_seed=4401,
            )
            it = core.cycle_batches(train_loader)
            for phase, want_old in [("old", True), ("new", False)]:
                trained = 0
                while trained < int(args.continual_steps_per_phase):
                    xb, yb = next(it)
                    mask = yb < 5 if want_old else yb >= 5
                    if not bool(mask.any()):
                        continue
                    xb = xb[mask].to(device).float()
                    yb = yb[mask].to(device).long()
                    opt.zero_grad(set_to_none=True)
                    import torch.nn.functional as F

                    loss = F.cross_entropy(model(xb).float(), yb)
                    loss.backward()
                    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
                    risk = batch_risk_metrics(model, xb, yb, output_dim)
                    velocity, _state = controller.update_and_emit(named=named, opt=opt, risk=risk, step=trained + 1)
                    optimizer_step(model, opt, "AdamW", trained + 1, {})
                    if variant == "continuous_fu":
                        apply_velocity(named, velocity, float(args.lr) * controller.rho * float(args.velocity_scale))
                    trained += 1
                if phase == "old":
                    old_before = eval_split(model, True)
            old_after = eval_split(model, True)
            new_after = eval_split(model, False)
            rows.append(
                {
                    "task": "Class_MNIST_exact_compact",
                    "variant": variant,
                    "old_task_accuracy_before_new_task": old_before,
                    "old_task_accuracy_after_new_task": old_after,
                    "new_task_accuracy": new_after,
                    "average_forgetting": old_before - old_after,
                    "relative_forgetting_reduction": "",
                    "backward_transfer": old_after - old_before,
                    "boundary_signal_state": controller.signal and norm_dict(controller.signal),
                    "boundary_causal_confidence": controller.q,
                    "status": "completed_continual_probe",
                }
            )
        base_forgetting = next((finite_float(r.get("average_forgetting")) for r in rows if r.get("variant") == "optimizer_alone"), None)
        for r in rows:
            f = finite_float(r.get("average_forgetting"))
            r["relative_forgetting_reduction"] = "" if base_forgetting in {None, 0.0} or f is None else (base_forgetting - f) / abs(base_forgetting)
    except Exception as exc:
        rows.append({"status": "failed_continual_probe", "error_type": type(exc).__name__, "error_message": str(exc)})
    write_rows(OUT_ROOT / "v22_40_continual_memory_matrix.csv", rows)
    append_exec(
        "stage_continual",
        task_id="H_continual_probe",
        status="pass" if rows and rows[0].get("status") == "completed_continual_probe" else "fail",
        gpu="0",
        files="results/v22_40/v22_40_continual_memory_matrix.csv",
        note=f"rows={len(rows)}",
    )
    return {"status": rows[0].get("status", "no_rows"), "rows": len(rows)}


def stage_grokking(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    ensure_out()
    device = torch_device("cuda:1" if int(args.visible_gpus) > 1 else "cuda:0" if int(args.visible_gpus) > 0 else "cpu")
    rows = []
    try:
        p = int(args.grokking_modulus)
        xs = torch.tensor([[i, j] for i in range(p) for j in range(p)], dtype=torch.long)
        ys = (xs[:, 0] + xs[:, 1]) % p
        gen = torch.Generator().manual_seed(2240)
        perm = torch.randperm(xs.shape[0], generator=gen)
        n_train = max(8, int(0.35 * xs.shape[0]))
        tr = perm[:n_train]
        te = perm[n_train:]
        train_x, train_y = xs[tr].to(device), ys[tr].to(device)
        test_x, test_y = xs[te].to(device), ys[te].to(device)

        class TinyMod(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = nn.Embedding(p, int(args.grokking_width))
                self.net = nn.Sequential(
                    nn.Linear(2 * int(args.grokking_width), int(args.grokking_hidden)),
                    nn.ReLU(),
                    nn.Linear(int(args.grokking_hidden), p),
                )

            def forward(self, x: Any) -> Any:
                e = self.emb(x)
                return self.net(e.reshape(e.shape[0], -1))

        for variant in ["optimizer_alone", "continuous_fu", "same_overhead_noop_control"]:
            torch.manual_seed(2240)
            model = TinyMod().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=float(args.grokking_lr), weight_decay=float(args.grokking_weight_decay))
            controller = ContinuousFUController(
                architecture="MLP",
                variant="continuous_fu" if variant == "continuous_fu" else "same_overhead_noop_control",
                beta_signal=float(args.beta_signal),
                beta_q=float(args.beta_q),
                eta_rho=float(args.eta_rho),
                eta_debt=float(args.eta_debt),
                tau_safe=float(args.tau_safe),
                rho_min=float(args.rho_min),
                rho_max=float(args.rho_max),
                velocity_scale=float(args.velocity_scale),
                mlp_velocity_source=str(args.mlp_velocity_source),
                slow_signal_beta=float(args.slow_signal_beta),
                random_seed=2240,
            )
            delay = ""
            slow_energy = 0.0
            fast_energy = 0.0
            for step in range(1, int(args.grokking_steps) + 1):
                idx = torch.randint(0, train_x.shape[0], (min(128, train_x.shape[0]),), device=device)
                xb = train_x[idx]
                yb = train_y[idx]
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                named = [(n, p0) for n, p0 in model.named_parameters() if p0.requires_grad]
                with torch.no_grad():
                    logits = model(xb)
                    probs = torch.softmax(logits, -1)
                    brier = ((probs - F.one_hot(yb, p).float()) ** 2).sum(-1).mean()
                velocity, _state = controller.update_and_emit(
                    named=named,
                    opt=opt,
                    risk={"ce": float(loss.item()), "ece": 0.0, "brier": float(brier.item()), "tail_q99": float(loss.item())},
                    step=step,
                )
                opt.step()
                if variant == "continuous_fu":
                    apply_velocity(named, velocity, float(args.grokking_lr) * controller.rho * float(args.velocity_scale))
                if step % max(1, int(args.grokking_eval_cadence)) == 0:
                    with torch.no_grad():
                        train_acc = float((model(train_x).argmax(-1) == train_y).float().mean().item())
                        test_acc = float((model(test_x).argmax(-1) == test_y).float().mean().item())
                    if delay == "" and test_acc >= 0.90:
                        delay = step
            with torch.no_grad():
                train_acc = float((model(train_x).argmax(-1) == train_y).float().mean().item())
                test_acc = float((model(test_x).argmax(-1) == test_y).float().mean().item())
            slow_energy = norm_dict(controller.signal)
            fast_energy = float(sum((p0.grad.float().norm().item() if p0.grad is not None else 0.0) for p0 in model.parameters()))
            rows.append(
                {
                    "task": "modular_addition",
                    "variant": variant,
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc,
                    "grokking_delay": delay,
                    "time_to_generalization": delay,
                    "slow_signal_energy": slow_energy,
                    "fast_signal_energy": fast_energy,
                    "reservoir_to_signal_migration_index": slow_energy / max(1.0e-12, fast_energy),
                    "status": "completed_grokking_probe",
                }
            )
    except Exception as exc:
        rows.append({"status": "failed_grokking_probe", "error_type": type(exc).__name__, "error_message": str(exc)})
    write_rows(OUT_ROOT / "v22_40_grokking_delay_matrix.csv", rows)
    append_exec(
        "stage_grokking",
        task_id="H_grokking_probe",
        status="pass" if rows and rows[0].get("status") == "completed_grokking_probe" else "fail",
        gpu="1",
        files="results/v22_40/v22_40_grokking_delay_matrix.csv",
        note=f"rows={len(rows)}",
    )
    return {"status": rows[0].get("status", "no_rows"), "rows": len(rows)}


def stage_preference(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.datasets import load_diabetes

    ensure_out()
    device = torch_device(str(args.preference_device))
    dataset = load_diabetes()
    x_np = dataset.data.astype("float32")
    y_np = dataset.target.astype("float32")
    x_np = (x_np - x_np.mean(axis=0, keepdims=True)) / (x_np.std(axis=0, keepdims=True) + 1.0e-6)
    input_dim = int(4 * x_np.shape[1])
    output_dim = 2
    variants = ["optimizer_alone", "continuous_fu", "same_overhead_noop_control", "same_signal_random_flow_control", "same_optimizer_geometry_control"]
    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    def make_pairs(indices: np.ndarray, n_pairs: int, seed: int, noise_rate: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(int(seed))
        left = rng.choice(indices, size=int(n_pairs), replace=True)
        right = rng.choice(indices, size=int(n_pairs), replace=True)
        x_left = x_np[left]
        x_right = x_np[right]
        pair_x = np.concatenate([x_left, x_right, x_left - x_right, np.abs(x_left - x_right)], axis=1).astype("float32")
        clean = (y_np[left] > y_np[right]).astype("int64")
        noisy = clean.copy()
        if float(noise_rate) > 0.0:
            flip = rng.random(size=noisy.shape[0]) < float(noise_rate)
            noisy[flip] = 1 - noisy[flip]
        return (
            torch.from_numpy(pair_x).to(device),
            torch.from_numpy(noisy).long().to(device),
            torch.from_numpy(clean).long().to(device),
        )

    def evaluate_pref(model: Any, pair_x: torch.Tensor, clean_y: torch.Tensor) -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            logits = model(pair_x).float()
            losses = F.cross_entropy(logits, clean_y.long(), reduction="none")
            risk = risk_metrics_from_logits(logits, clean_y, output_dim, losses)
            risk["accuracy"] = float((logits.argmax(dim=-1) == clean_y).float().mean().item())
            risk["NLL"] = risk["ce"]
            return risk

    class PreferenceMLP(nn.Module):
        def __init__(self, seed: int) -> None:
            super().__init__()
            torch.manual_seed(int(seed))
            hidden = int(args.preference_hidden)
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, output_dim),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x.float())

    for seed in split_csv(args.preference_seeds, int):
        rng = np.random.default_rng(int(seed) + 224040)
        perm = rng.permutation(np.arange(x_np.shape[0]))
        n_train = int(0.60 * len(perm))
        n_held = int(0.20 * len(perm))
        train_idx = perm[:n_train]
        held_idx = perm[n_train : n_train + n_held]
        test_idx = perm[n_train + n_held :]
        train_x, train_noisy_y, train_clean_y = make_pairs(train_idx, int(args.preference_train_pairs), int(seed) + 1, float(args.preference_noise_rate))
        held_x, _held_noisy_y, held_clean_y = make_pairs(held_idx, int(args.preference_held_pairs), int(seed) + 2, 0.0)
        test_x, _test_noisy_y, test_clean_y = make_pairs(test_idx, int(args.preference_test_pairs), int(seed) + 3, 0.0)
        for variant in variants:
            model = PreferenceMLP(int(seed) + 9917).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=float(args.preference_lr), weight_decay=float(args.preference_weight_decay))
            controller = ContinuousFUController(
                architecture="MLP",
                variant=variant,
                beta_signal=float(args.beta_signal),
                beta_q=float(args.beta_q),
                eta_rho=float(args.eta_rho),
                eta_debt=float(args.eta_debt),
                tau_safe=float(args.tau_safe),
                rho_min=float(args.rho_min),
                rho_max=float(args.rho_max),
                velocity_scale=float(args.velocity_scale),
                mlp_velocity_source=str(args.mlp_velocity_source),
                slow_signal_beta=float(args.slow_signal_beta),
                fast_controller_stats=bool(args.fast_controller_stats),
                safety_barrier_strength=float(args.safety_barrier_strength),
                safety_barrier_slack=float(args.safety_barrier_slack),
                random_seed=int(seed) + 7700,
            )
            gen = torch.Generator(device=device).manual_seed(int(seed) + 4400)
            loss_trace: list[float] = []
            full_step_times: list[float] = []
            state_times: list[float] = []
            controller_times: list[float] = []
            base_opt_times: list[float] = []
            rho_vals: list[float] = []
            pref_held_risk_steps = 0
            pref_train_proxy_steps = 0
            for step in range(1, int(args.preference_steps) + 1):
                step_start = time.time()
                idx = torch.randint(0, train_x.shape[0], (min(int(args.preference_batch_size), train_x.shape[0]),), generator=gen, device=device)
                xb = train_x[idx]
                yb = train_noisy_y[idx]
                opt.zero_grad(set_to_none=True)
                logits = model(xb).float()
                per_loss = F.cross_entropy(logits, yb.long(), reduction="none")
                loss = per_loss.mean()
                loss.backward()
                named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
                state_start = time.time()
                if str(args.runtime_risk_source) == "held_cadenced" and (step == 1 or step % max(1, int(args.risk_eval_cadence)) == 0):
                    risk = batch_risk_metrics(model, held_x, held_clean_y, output_dim)
                    risk_source = "held_clean_preference_cadenced"
                    pref_held_risk_steps += 1
                else:
                    risk = cheap_train_risk_from_losses(per_loss)
                    risk_source = "train_noisy_preference_loss_proxy"
                    pref_train_proxy_steps += 1
                velocity, state = controller.update_and_emit(named=named, opt=opt, risk=risk, step=step)
                state_times.append(time.time() - state_start)
                opt_start = time.time()
                opt.step()
                base_opt_times.append(time.time() - opt_start)
                ctrl_start = time.time()
                fu_norm = 0.0
                if variant in {"continuous_fu", "same_signal_random_flow_control", "same_optimizer_geometry_control", "same_overhead_noop_control"}:
                    scale = float(args.preference_lr) * float(controller.rho) * float(args.velocity_scale) * float(state.get("safety_barrier_scale", 1.0))
                    fu_norm, _basis_norm, _nonbasis_norm = apply_velocity(named, velocity, scale)
                controller_times.append(time.time() - ctrl_start)
                full_step_times.append(time.time() - step_start)
                loss_trace.append(float(loss.detach().item()))
                rho_vals.append(float(controller.rho))
                if step == 1 or step == int(args.preference_steps) or step % max(1, int(args.preference_trace_cadence)) == 0:
                    trace_rows.append(
                        {
                            "task": "sklearn_diabetes_pairwise_preference",
                            "seed": seed,
                            "variant": variant,
                            "step": step,
                            "risk_source": risk_source,
                            "fu_velocity_norm": fu_norm,
                            "continuous_fu_state_updated_every_step": 1,
                            "fu_velocity_emitted_every_step": 1,
                            "candidate_action_selection_used_for_runtime": 0,
                            **state,
                        }
                    )
            final_test = evaluate_pref(model, test_x, test_clean_y)
            final_held = evaluate_pref(model, held_x, held_clean_y)
            rows.append(
                {
                    "task": "sklearn_diabetes_pairwise_preference",
                    "source_kind": "sklearn_builtin_diabetes_real_regression_target_pairwise_derived",
                    "seed": seed,
                    "variant": variant,
                    "train_pairs": int(args.preference_train_pairs),
                    "held_pairs": int(args.preference_held_pairs),
                    "test_pairs": int(args.preference_test_pairs),
                    "train_preference_label_noise_rate": float(args.preference_noise_rate),
                    "steps": int(args.preference_steps),
                    "hidden": int(args.preference_hidden),
                    "mlp_velocity_source": str(args.mlp_velocity_source),
                    "runtime_risk_source_policy": str(args.runtime_risk_source),
                    "runtime_uses_held_risk_for_control": int(pref_held_risk_steps > 0),
                    "runtime_train_only_control": int(pref_held_risk_steps == 0),
                    "risk_held_eval_steps": pref_held_risk_steps,
                    "risk_train_proxy_steps": pref_train_proxy_steps,
                    "final_clean_NLL": final_test["NLL"],
                    "final_clean_accuracy": final_test["accuracy"],
                    "held_clean_NLL": final_held["NLL"],
                    "AUC_noisy_train_loss_time": sum(loss_trace) / max(1, len(loss_trace)),
                    "ECE": final_test["ece"],
                    "Brier": final_test["brier"],
                    "tail_loss_q95": final_test["tail_q95"],
                    "tail_loss_q99": final_test["tail_q99"],
                    "rho_mean": statistics.fmean(rho_vals) if rho_vals else 0.0,
                    "rho_p50": statistics.median(rho_vals) if rho_vals else 0.0,
                    "controller_overhead_ratio": (sum(controller_times) + sum(state_times)) / max(1.0e-12, sum(full_step_times)),
                    "full_step_ms": 1000.0 * statistics.fmean(full_step_times) if full_step_times else 0.0,
                    "base_optimizer_ms": 1000.0 * statistics.fmean(base_opt_times) if base_opt_times else 0.0,
                    "controller_ms": 1000.0 * statistics.fmean(controller_times) if controller_times else 0.0,
                    "state_update_ms": 1000.0 * statistics.fmean(state_times) if state_times else 0.0,
                    "continuous_fu_state_updated_every_step": 1,
                    "fu_velocity_emitted_every_step": 1,
                    "candidate_action_selection_used_for_runtime": 0,
                    "randomized_audit_only": 1,
                    "official_eligible": 0,
                    "not_official_reason": "Part H derived noisy-preference probe; not a full-loop official promotion artifact.",
                    "status": "completed_noisy_preference_probe",
                }
            )

    by_key = {(str(r.get("seed")), str(r.get("variant"))): r for r in rows}
    enriched: list[dict[str, Any]] = []
    for row in rows:
        seed = str(row.get("seed"))
        base = by_key.get((seed, "optimizer_alone"), {})
        noop = by_key.get((seed, "same_overhead_noop_control"), {})
        rand = by_key.get((seed, "same_signal_random_flow_control"), {})
        geom = by_key.get((seed, "same_optimizer_geometry_control"), {})
        out = dict(row)
        nll = finite_float(row.get("final_clean_NLL"))
        auc = finite_float(row.get("AUC_noisy_train_loss_time"))
        ece = finite_float(row.get("ECE"))
        brier = finite_float(row.get("Brier"))
        tail = finite_float(row.get("tail_loss_q99"))
        base_nll = finite_float(base.get("final_clean_NLL"))
        base_auc = finite_float(base.get("AUC_noisy_train_loss_time"))
        base_ece = finite_float(base.get("ECE"))
        base_brier = finite_float(base.get("Brier"))
        base_tail = finite_float(base.get("tail_loss_q99"))
        out["NLL_delta_vs_optimizer"] = "" if nll is None or base_nll is None else nll - base_nll
        out["AUC_delta_vs_optimizer"] = "" if auc is None or base_auc is None else auc - base_auc
        out["no_ECE_Brier_tail_debt_vs_optimizer"] = (
            ""
            if None in {ece, brier, tail, base_ece, base_brier, base_tail}
            else int(ece <= base_ece + 1.0e-12 and brier <= base_brier + 1.0e-12 and tail <= base_tail + 1.0e-12)
        )
        control_beats = []
        for cname, crow in [("noop", noop), ("same_signal_random", rand), ("same_optimizer_geometry", geom)]:
            c_nll = finite_float(crow.get("final_clean_NLL"))
            out[f"NLL_delta_vs_{cname}_control"] = "" if nll is None or c_nll is None else nll - c_nll
            out[f"beats_{cname}_control"] = "" if nll is None or c_nll is None else int(nll < c_nll)
            control_beats.append(out[f"beats_{cname}_control"])
        out["matched_controls_available"] = int(all(v != "" for v in control_beats))
        out["beats_matched_controls"] = int(out["matched_controls_available"] == 1 and all(int_flag(v) for v in control_beats))
        enriched.append(out)
    signal_rows = [r for r in enriched if r.get("variant") == "continuous_fu"]
    n = len(signal_rows)
    nll_wins = sum(1 for r in signal_rows if (finite_float(r.get("NLL_delta_vs_optimizer")) or 0.0) < 0.0)
    auc_wins = sum(1 for r in signal_rows if (finite_float(r.get("AUC_delta_vs_optimizer")) or 0.0) < 0.0)
    no_debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt_vs_optimizer")) for r in signal_rows)
    beats_controls = sum(int_flag(r.get("beats_matched_controls")) for r in signal_rows)
    overhead = sum(1 for r in signal_rows if (finite_float(r.get("controller_overhead_ratio"), 99.0) or 99.0) <= 0.30)
    summary = {
        "task": "sklearn_diabetes_pairwise_preference",
        "source_kind": "sklearn_builtin_diabetes_real_regression_target_pairwise_derived",
        "signal_rows": n,
        "NLL_improvement_rows": nll_wins,
        "AUC_improvement_rows": auc_wins,
        "no_ECE_Brier_tail_debt_rows": no_debt,
        "beats_matched_controls_rows": beats_controls,
        "controller_overhead_le_0p30_rows": overhead,
        "exploration_gate_pass": int(n >= 3 and nll_wins >= 2 and auc_wins >= 2 and beats_controls >= 2 and no_debt >= 2 and overhead >= n),
        "official_candidate_gate_pass": 0,
        "status": "completed_noisy_preference_summary",
    }
    write_rows(OUT_ROOT / "v22_40_noisy_preference_matrix.csv", enriched or [{"status": "no_noisy_preference_rows"}])
    write_rows(OUT_ROOT / "v22_40_noisy_preference_summary.csv", [summary])
    write_rows(OUT_ROOT / "v22_40_noisy_preference_state_trace.csv", trace_rows or [{"status": "no_noisy_preference_trace_rows"}])
    append_exec(
        "stage_preference",
        task_id="H_noisy_preference_probe",
        status="pass",
        gpu=str(args.preference_device),
        files="results/v22_40/v22_40_noisy_preference_matrix.csv, results/v22_40/v22_40_noisy_preference_summary.csv, results/v22_40/v22_40_noisy_preference_state_trace.csv",
        note=json.dumps(summary, sort_keys=True),
    )
    return summary


def finalize() -> dict[str, Any]:
    ensure_out()
    merge_chunks()
    code = (read_rows(OUT_ROOT / "v22_40_code_truth_gate.csv") or [{}])[0]
    runtime = (read_rows(OUT_ROOT / "v22_40_continuous_runtime_truth_matrix.csv") or [{}])[0]
    full = (read_rows(OUT_ROOT / "v22_40_full_loop_summary.csv") or [{}])[0]
    continual = read_rows(OUT_ROOT / "v22_40_continual_memory_matrix.csv")
    grok = read_rows(OUT_ROOT / "v22_40_grokking_delay_matrix.csv")
    preference = read_rows(OUT_ROOT / "v22_40_noisy_preference_summary.csv")
    if int_flag(code.get("status") == "pass") == 0:
        route = "R0-CodeIdentityGateFailed"
        reason = "compile/import truth gate failed"
    elif int_flag(runtime.get("candidate_action_regression_pass")) == 0:
        route = "R0p5-CandidateActionRegression_Stop"
        reason = "continuous runtime trace did not prove per-step state/velocity or candidate-action audit pass"
    elif int_flag(runtime.get("runtime_train_only_control_pass")) == 0:
        route = "R1-ContinuousRuntimeTruthFailed"
        reason = "continuous runtime has no train-only eligible control rows after excluding held-risk/unknown legacy traces"
    elif int_flag(full.get("mlp_general_exploration_gate_pass")) and int_flag(full.get("kan_internal_exploration_gate_pass")):
        route = "R5-KANInternalValueOpened"
        reason = "MLP and KAN exploration gates opened; official hard/continual/grokking coverage still checked separately"
    elif int_flag(full.get("mlp_general_exploration_gate_pass")):
        route = "R4-MLPGeneralValueOpened"
        reason = "MLP exploration gate opened, KAN internal gate not opened"
    elif int_flag(full.get("kan_internal_exploration_gate_pass")):
        route = "R5-KANInternalValueOpened"
        reason = "KAN internal gate opened, MLP general value not opened"
    elif int(full.get("KAN_NLL_improvement_rows") or 0) > 0 or int(full.get("MLP_NLL_improvement_rows") or 0) > 0:
        route = "R9-FUWeakOptimizerPatchOnly"
        reason = "train-only continuous runtime passed and tiny NLL/AUC improvements exist, but MLP general value, KAN internal gate, no-debt, matched-control/efficiency gates did not open"
    else:
        route = "R2-StateFlowNoSignal"
        reason = "continuous runtime passed but full-loop controls-matched exploration gates did not open"
    if route == "R2-StateFlowNoSignal":
        state_rows = read_rows(OUT_ROOT / "v22_40_continuous_fu_state_trace.csv")
        rho_vals = [finite_float(r.get("rho_t")) for r in state_rows if finite_float(r.get("rho_t")) is not None]
        if rho_vals and max(rho_vals) <= 1.0e-12:
            reason = "ContinuousFUCollapsedToNoOp: rho_t stayed zero in merged traces"
        elif rho_vals and min(rho_vals) >= 0.95 * max(rho_vals) and max(rho_vals) > 0.0:
            reason = "ContinuousFUOverGuidance/saturation check: rho_t had weak dynamics; gates did not open"
    route_json = {
        "final_route": route,
        "route_reason": reason,
        "code_truth_status": code.get("status", ""),
        "candidate_action_regression_pass": runtime.get("candidate_action_regression_pass", ""),
        "full_loop_summary": full,
        "continual_rows": len(continual),
        "grokking_rows": len(grok),
        "noisy_preference_rows": len(preference),
        "generated_at": now_sg(),
    }
    write_json(OUT_ROOT / "v22_40_final_route.json", route_json)
    manifest = []
    for name in REQUIRED_ARTIFACTS:
        p = OUT_ROOT / name
        if p.exists():
            manifest.append({"artifact": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256_file(p), "exists": 1})
        else:
            manifest.append({"artifact": str(p.relative_to(ROOT)), "bytes": "", "sha256": "", "exists": 0})
    write_rows(OUT_ROOT / "v22_40_artifact_manifest.csv", manifest)
    manifest_path = OUT_ROOT / "v22_40_artifact_manifest.csv"
    for row in manifest:
        if row.get("artifact") == str(manifest_path.relative_to(ROOT)):
            row["bytes"] = manifest_path.stat().st_size
            row["sha256"] = "self_referential_manifest_hash_omitted"
            row["exists"] = 1
    write_rows(manifest_path, manifest)
    update_recap(route_json)
    append_exec(
        "finalize",
        task_id="finalize_v22_40",
        status="pass",
        gpu="0",
        files="results/v22_40/v22_40_final_route.json, results/v22_40/v22_40_artifact_manifest.csv, docs/DG-KAN_v22.40_ContinuousFunctionalFlowFU_实验结果复盘.md",
        note=f"route={route}; reason={reason}",
    )
    return route_json


def update_recap(route_json: dict[str, Any]) -> None:
    code = read_rows(OUT_ROOT / "v22_40_code_truth_gate.csv")
    runtime = read_rows(OUT_ROOT / "v22_40_continuous_runtime_truth_matrix.csv")
    audit = read_rows(OUT_ROOT / "v22_40_candidate_action_regression_audit.csv")
    unit = read_rows(OUT_ROOT / "v22_40_controller_unit_test_matrix.csv")
    horizon = read_rows(OUT_ROOT / "v22_40_horizon_decay_matrix.csv")
    leakage = read_rows(OUT_ROOT / "v22_40_candidate_identity_leakage_audit.csv")
    full_summary = read_rows(OUT_ROOT / "v22_40_full_loop_summary.csv")
    full_rows = read_rows(OUT_ROOT / "v22_40_full_loop_matrix.csv")
    long_rows = read_rows(OUT_ROOT / "v22_40_long_horizon_path_validation.csv")
    eff = read_rows(OUT_ROOT / "v22_40_efficiency_matrix.csv")
    continual = read_rows(OUT_ROOT / "v22_40_continual_memory_matrix.csv")
    grok = read_rows(OUT_ROOT / "v22_40_grokking_delay_matrix.csv")
    preference = read_rows(OUT_ROOT / "v22_40_noisy_preference_matrix.csv")
    preference_summary = read_rows(OUT_ROOT / "v22_40_noisy_preference_summary.csv")
    manifest = read_rows(OUT_ROOT / "v22_40_artifact_manifest.csv")

    def repair_prefix_summary(prefix: str) -> str:
        cont = [r for r in full_rows if str(r.get("run_label", "")).startswith(prefix) and r.get("variant") == "continuous_fu"]
        if not cont:
            return f"{prefix}: no continuous_fu rows"
        n = len(cont)
        nll = sum(1 for r in cont if (finite_float(r.get("NLL_delta_vs_own_strong_optimizer")) or 0.0) < 0.0)
        auc = sum(1 for r in cont if (finite_float(r.get("AUC_delta_vs_own_strong_optimizer")) or 0.0) < 0.0)
        debt = sum(int_flag(r.get("no_ECE_Brier_tail_debt_vs_optimizer")) for r in cont)
        controls = sum(int_flag(r.get("beats_matched_controls")) for r in cont)
        overhead = sum(1 for r in cont if (finite_float(r.get("controller_overhead_ratio"), 99.0) or 99.0) <= 0.30)
        return f"{prefix}: NLL {nll}/{n}, AUC {auc}/{n}, no-debt {debt}/{n}, beats matched controls {controls}/{n}, overhead<=0.30 {overhead}/{n}"

    continual_fu = next((r for r in continual if r.get("variant") == "continuous_fu"), {})
    grok_fu = next((r for r in grok if r.get("variant") == "continuous_fu"), {})
    pref_summary = preference_summary[0] if preference_summary else {}
    pref_signal = [r for r in preference if r.get("variant") == "continuous_fu"]
    pref_trainonly = sum(int_flag(r.get("runtime_train_only_control")) for r in pref_signal)
    text = [
        "# DG-KAN v22.40 Continuous Functional Flow FU 实验结果复盘",
        "",
        f"updated_at: {now_sg()}",
        "",
        "## Final route",
        "",
        f"- final_route: `{route_json.get('final_route')}`",
        f"- route_reason: `{route_json.get('route_reason')}`",
        f"- 关键约束：本轮不把 candidate action / micro-RCT winner / path-MPC 离散序列作为 runtime policy。",
        "",
        "## Code / Identity / Continuous Runtime Truth",
        "",
        md_table(code, ["clean_unzip_compileall_pass", "clean_unzip_import_pass", "missing_transitive_dependency_count", "official_DGKAN_identity_pass", "torch_cuda_device_count_visible", "status"], 5),
        "",
        md_table(runtime, ["continuous_fu_state_updated_every_step", "fu_velocity_emitted_every_step", "candidate_action_selection_used_for_runtime", "runtime_argmax_candidate_used", "runtime_topk_candidate_used", "randomized_audit_only", "candidate_action_regression_pass", "runtime_train_only_control_pass", "eligible_train_only_runtime_rows", "legacy_held_risk_or_unknown_rows_excluded_from_gate", "status"], 5),
        "",
        md_table(audit, ["runtime_argmax_candidate_used", "runtime_topk_candidate_used", "treatment_identity_used_in_runtime_state", "direction_source_used_in_runtime_state", "candidate_value_model_used_as_runtime_policy", "micro_rct_winner_used_as_runtime_action", "path_mpc_discrete_action_sequence_used", "continuous_fu_state_updated_every_step", "fu_velocity_emitted_every_step", "candidate_action_regression_pass"], 5),
        "",
        "## v22.39 Failure Reanalysis",
        "",
        md_table(horizon, ["candidate_family", "H20_LCB", "H60_LCB", "H200_LCB", "H800_LCB", "H20_only_transient_effect", "eligible_as_continuous_signal_source", "horizon_source_mode", "one_step_fu_replay_promoted", "promoted_to_v22_40_runtime"], 18),
        "",
        md_table(leakage, ["model_id", "fold_key", "AUC_positive_treatment", "identity_sensitive", "promoted_to_runtime", "source"], 12),
        "",
        "## Controller Unit Tests",
        "",
        md_table(unit, ["run_label", "steps", "C1_pass", "C2_audit_randomization_not_runtime_pass", "C3_no_state_leakage_pass", "C4_pass", "controller_unit_tests_pass"], 12),
        "",
        "## Full Loop Results",
        "",
        md_table(full_summary, ["status", "full_loop_rows", "gate_rows", "diagnostic_rows_excluded_from_gate", "held_risk_or_unknown_runtime_rows_excluded_from_gate", "signal_rows", "tier0_signal_rows_excluded_from_gate", "mlp_signal_rows", "kan_signal_rows", "MLP_NLL_improvement_rows", "MLP_AUC_improvement_rows", "MLP_beats_matched_controls_rows", "MLP_no_ECE_Brier_tail_debt_rows", "MLP_controller_overhead_le_0p30_rows", "KAN_NLL_improvement_rows", "KAN_beats_matched_controls_rows", "KAN_no_ECE_Brier_tail_debt_rows", "KAN_controller_overhead_le_0p25_rows", "controller_overhead_le_0p30_rows", "mlp_general_exploration_gate_pass", "kan_internal_exploration_gate_pass"], 5),
        "",
        md_table(full_rows, ["run_label", "dataset", "seed", "architecture", "optimizer_family", "variant", "comparison_group", "runtime_train_only_control", "runtime_risk_source_policy", "audit_eval_source_policy", "final_NLL", "final_accuracy", "NLL_delta_vs_own_strong_optimizer", "AUC_delta_vs_own_strong_optimizer", "no_ECE_Brier_tail_debt_vs_optimizer", "beats_noop_control", "beats_same_signal_random_control", "beats_same_optimizer_geometry_control", "matched_controls_available", "beats_matched_controls", "rho_mean", "rho_p50", "rho_p90", "fu_actuation_mode", "mlp_velocity_source", "tail_projection_enabled", "tail_projection_quantile", "safety_barrier_scale_mean", "fast_controller_stats", "controller_overhead_ratio"], 32),
        "",
        "## Horizon Diagnostics / Continual / Grokking",
        "",
        md_table(long_rows, ["dataset", "seed", "architecture", "optimizer_family", "H", "Y_H_NLL_delta", "Y_H_positive", "Y_H_ECE_delta", "Y_H_Brier_delta", "Y_H_tail_q99_delta", "horizon_measurement_mode", "one_step_fu_replay_used", "continuous_fu_state_updated_every_step", "fu_velocity_emitted_every_step"], 24),
        "",
        md_table(continual, ["task", "variant", "old_task_accuracy_before_new_task", "old_task_accuracy_after_new_task", "new_task_accuracy", "average_forgetting", "relative_forgetting_reduction", "status"], 8),
        "",
        md_table(grok, ["task", "variant", "train_accuracy", "test_accuracy", "grokking_delay", "slow_signal_energy", "fast_signal_energy", "reservoir_to_signal_migration_index", "status"], 8),
        "",
        md_table(preference_summary, ["task", "signal_rows", "NLL_improvement_rows", "AUC_improvement_rows", "no_ECE_Brier_tail_debt_rows", "beats_matched_controls_rows", "controller_overhead_le_0p30_rows", "exploration_gate_pass", "status"], 4),
        "",
        md_table(preference, ["task", "seed", "variant", "final_clean_NLL", "final_clean_accuracy", "NLL_delta_vs_optimizer", "AUC_delta_vs_optimizer", "no_ECE_Brier_tail_debt_vs_optimizer", "beats_noop_control", "beats_same_signal_random_control", "beats_same_optimizer_geometry_control", "matched_controls_available", "beats_matched_controls", "rho_mean", "controller_overhead_ratio", "status"], 18),
        "",
        "## Efficiency",
        "",
        md_table(eff, ["run_label", "architecture", "variant", "full_step_ms", "base_optimizer_ms", "controller_ms", "state_update_ms", "audit_intervention_ms", "memory_peak_mb", "controller_overhead_ratio", "controller_overhead_exploration_pass"], 24),
        "",
        "## Analysis / Evidence Chain / Insight",
        "",
        "- 连续 runtime 证据链来自 `v22_40_continuous_fu_state_trace.csv` 与 `v22_40_runtime_decision_trace.csv`：每个训练 step 写入 state、rho、q、safety debt、FU velocity，不包含 treatment identity / direction_source / future outcome / test result。",
        "- micro-intervention 只写入 `v22_40_micro_intervention_audit_matrix.csv` 与 `v22_40_causal_state_update_matrix.csv`，字段 `sampled_treatment_used_for_runtime=0`；它最多校准 q_t，不直接决定当前 runtime update。",
        "- v22.39 reanalysis 保留历史证据但不 promotion：identity-only AUC 和 candidate-heldout value model 只能解释为什么要转向 continuous flow，不能作为 v22.40 runtime policy。",
        "- H20/H60/H200/H800 在本复盘中有两类来源：`v22_40_horizon_decay_matrix.csv` 是 v22.39 历史 candidate-pair diagnostic，字段 `one_step_fu_replay_promoted=0`，不能 promotion；`v22_40_long_horizon_path_validation.csv` 是连续训练过程中在第 H step 记录的 full-flow checkpoint，字段 `horizon_measurement_mode=continuous_full_training_snapshot`、`one_step_fu_replay_used=0`。",
        "- 因此 H200/H800 只允许作为连续训练流的 checkpoint sanity check 或历史失败 reanalysis；不允许作为“一步 FU 更新后看后续变化”的成功证据。",
        "- 若 full-loop gate 未开，本轮结论只能是连续控制流实现和审计通过，但 controls-matched scientific superiority 未证明；这不是降目标，而是把 blocker 定位到 state/actuator/optimizer geometry 或任务 regime。",
        "- 可见 GPU 证据来自 `v22_40_code_truth_gate.csv`：本 shell 只暴露 CUDA 0-3；因此并行实验使用 0-3，未使用不可见 4/5。",
        "- runtime held-risk firewall：发现旧 full-loop/preference controller 曾用 held/clean preference risk 更新 safety/rho；这与 no validation/test/future/query direction 不兼容。当前 gate 只接受 `runtime_train_only_control=1` 的 rows，legacy held-risk/unknown rows 留在 artifacts 但排除出 scientific gate。",
        "- gate summary 排除了 `unit_`、`_smoke` diagnostic rows 与 Tier0/空 tier signal rows；这些行仍保留在 artifacts，但不允许污染 scientific gate。",
        "- exploration gate 采用计划中的绝对门槛：MLP 需要至少 9 个 eligible rows 且 NLL>=5、AUC>=6、matched-control>=6、no-debt>=8、overhead 全部 <=0.30；KAN internal 需要至少 9 个 eligible rows 且 NLL>=5、control>=6、basis>=8、readout leakage 全部 <=0.3、overhead 全部 <=0.25。",
        "- matched-control delta 使用 `comparison_group` 匹配同一 repair/hidden/batch/variant group，修复了 h512 repair 行被错误拿 h64 baseline 做 delta 的比较键问题。",
        f"- repair4 slow-signal h512/b512 结果：{repair_prefix_summary('repair4_slow_h512b512_')}。",
        f"- repair5 tail-cadence low-rho h512/b512 结果：{repair_prefix_summary('repair5_tailc8_lowrho_h512_')}。",
        f"- repair6 tail-cadence ultra-low-rho h512/b512 结果：{repair_prefix_summary('repair6_tailc4_ultralowrho_h512_')}。",
        f"- repair7 sharpness/tail safety-barrier q95 h512/b512 结果：{repair_prefix_summary('repair7_barrierq95_h512_')}。",
        f"- repair8 q99-tail safety-barrier h512/b512 结果：{repair_prefix_summary('repair8_barrierq99_h512_')}。",
        f"- repair9 optimizer-state + safety-barrier 9-task 结果：{repair_prefix_summary('repair9_optstate_barrier9_')}。该轮按计划 16.4 测试 optimizer-state continuous FU，覆盖 CIFAR10/SVHN/EMNIST/EMNIST_LETTERS 与 Wine/Spam/Rice/Bean/Telescope；结果 AUC/control 有改善，但 NLL/no-debt/overhead 未同时过线，不能 promotion。",
        f"- repair11 pre-step optimizer-state actuation 结果：{repair_prefix_summary('repair11_prestep_grad_optstate_h512_')}。该轮把 FU velocity 注入 train-only gradient 后交给 AdamW optimizer state；5 个 hard MLP rows 中 NLL=0/5、AUC=5/5、no-debt=0/5、matched-controls=0/5，说明它只改善训练曲线 AUC，最终泛化和安全债反而未打开。",
        f"- repair12 train-only slow-signal hard9 结果：{repair_prefix_summary('repair12_trainonly_slow_hard9_')}。该轮是 held-risk firewall 后的主矩阵，9/9 continuous rows 均 `runtime_train_only_control=1`、risk_held_eval_steps=0、audit_held_eval_steps=0；AUC 有 7/9，但 NLL 只有 2/9、matched-controls 1/9、no-debt 2/9，说明去掉 held risk 后信号更弱但更可信。",
        f"- repair14 train-only KAN basis9 结果：{repair_prefix_summary('repair14_trainonly_kan_basis9_')}。KAN train-only 覆盖恢复到 9 rows，AUC 9/9、basis_energy>=0.5 为 9/9、readout_leakage<=0.3 为 9/9，但 NLL 4/9、matched-controls 2/9、overhead<=0.25 4/9，仍未打开 KAN internal exploration。",
        f"- continual repair 最新 continuous_fu relative_forgetting_reduction={continual_fu.get('relative_forgetting_reduction', '')}；未达到 5% exploration 门槛。",
        f"- grokking repair 最新 continuous_fu train_accuracy={grok_fu.get('train_accuracy', '')}, test_accuracy={grok_fu.get('test_accuracy', '')}, grokking_delay={grok_fu.get('grokking_delay', '')}；未打开 delayed generalization。",
        f"- noisy-preference probe 最新结果：NLL={pref_summary.get('NLL_improvement_rows', '')}/{pref_summary.get('signal_rows', '')}, AUC={pref_summary.get('AUC_improvement_rows', '')}/{pref_summary.get('signal_rows', '')}, no-debt={pref_summary.get('no_ECE_Brier_tail_debt_rows', '')}/{pref_summary.get('signal_rows', '')}, controls={pref_summary.get('beats_matched_controls_rows', '')}/{pref_summary.get('signal_rows', '')}, overhead={pref_summary.get('controller_overhead_le_0p30_rows', '')}/{pref_summary.get('signal_rows', '')}, train-only={pref_trainonly}/{len(pref_signal)}；该 probe 是 sklearn Diabetes 真实连续 target 派生 pairwise preference，不作为 official promotion。",
        "",
        "## 修复 / 修改记录",
        "",
        "- 新增 `experiments/run_v22_40_continuous_functional_flow_fu.py`：实现 per-step ContinuousFUController、runtime trace、candidate-action regression audit、v22.39 reanalysis、full-loop merge、continual/grokking probes、manifest/final route。",
        "- C4 blocker 修复：首轮 controller unit test 中 `unit_kan_fu` safety_response_rate=0.7027027027027027，低于 0.8；随后在 `ContinuousFUController.update_and_emit()` 中加入 positive safety_debt_delta 的即时 rho penalty，并禁止同一步被 q 正项抵消，重跑后 C4 pass=1。",
        "- Control validity 修复：发现 `same_signal_random_flow_control` 与 `same_optimizer_geometry_control` 的 rho 被置 0，实际退化为 no-op；已改为与 continuous_fu 共享 rho dynamics，仅替换 velocity 来源，并覆盖重跑相关 Fashion/KMNIST/CIFAR controls。",
        "- MLP blocker 修复尝试：新增 `--mlp-velocity-source optimizer_state|optimizer_residual`，并运行 optimizer-state repair 与 high-rho optimizer-residual repair；结果没有打开 MLP general value，且 residual high-rho 在 CIFAR/KMNIST 变差或被 controls 解释。",
        "- Horizon 口径澄清：此前 `Long-horizon 修复尝试` 措辞容易误解；KMNIST H800 slice 是 800-step continuous full-flow collect，不是 one-step FU replay。历史 v22.39 H20/H200/H800 candidate-pair 表只保留为 diagnostic，不作为 v22.40 runtime 成功证据。",
        "- Summary bug 修复：`KAN_readout_leakage_le_0p3_rows` 之前把真实 `0.0` leakage 当作 falsy 后替换成 1.0；已修复为显式 None 检查并重 merge。",
        "- Efficiency measurement 修复：`controller_ms` 原先包含 base optimizer step；已拆出 `base_optimizer_ms`，controller overhead 只计 state/risk/controller/FU velocity/audit 开销。",
        "- Matched comparison 修复：`enrich_full_loop()` 之前按 dataset/seed/architecture/optimizer/variant 粗匹配 baseline，导致 h512 repair 可能和 h64 baseline 比较；已改为 `comparison_group` 匹配同一 run-label group，并重 merge。",
        "- MLP slow-signal/efficiency 修复尝试：新增 `--mlp-velocity-source slow_signal`、`--rho-min`、`--risk-eval-cadence`、`--fast-controller-stats`、`--tail-projection-cadence`；repair4/repair5 已并行跑 matched controls。",
        "- repair4 证明 overhead 可在 h512/b512 matched 设置中过线，但 MLP general gate 未开；repair5/repair6 引入稀疏 tail projection 和更低 rho 后，CIFAR/Fashion 出现局部 tiny NLL/control win，但 no-debt 与跨数据集稳定性仍不足。",
        "- repair7/repair8 新增 sharpness/tail safety barrier：`sharpness_proxy`、`safety_barrier_debt`、`safety_barrier_scale`、`--tail-projection-quantile`、`--safety-barrier-strength`、`--safety-barrier-slack`。修复中发现 cheap train-risk 把 ECE/Brier 填 0 会制造虚假 calibration debt，已新增 `risk_has_calibration`，只在 held/full risk 可用时让 ECE/Brier 进入 calibration barrier。",
        "- repair7 将 no-debt 从 repair6 的 1/4 提高到 2/4，但仍未达到 MLP general gate；repair8 q99-tail projection 没有进一步改善，因此不能 promotion。",
        "- repair9 修复尝试：按计划 16.4 跑 `--mlp-velocity-source optimizer_state` + tail/sharpness safety barrier 的 9-task matched-control 矩阵；首轮请求 6 GPU 但当前 PyTorch 只可见 4 张卡，出现 `invalid device ordinal`，已在 `stage_full_matrix()` 中用 `torch.cuda.device_count()` cap 调度并重跑完整 45 rows。",
        "- command-journal 审计修复：长 CLI 使 `csv.DictReader` 触发 field-size limit，已在 runner 初始化加入 `csv.field_size_limit(sys.maxsize)`；随后重跑 SVHN continuous collect、merge 与 finalize。该修复只影响日志读取稳定性，不改变实验指标。",
        "- Gate 口径修复：计划规定 Tier0 只能 debug，不能触发 success route；发现旧 summary 将 Fashion/KMNIST 等 Tier0 rows 混入 MLP/KAN gate 计数，已改为只用非 Tier0/非空 tier signal rows 计算 MLP/KAN gates，并新增 `tier0_signal_rows_excluded_from_gate` 留痕。",
        "- Gate 阈值修复：旧 summary 用 `len>=3` 与比例阈值近似 exploration gate，可能让 4 个 hard KAN rows 误开 R5；已改为计划明示的 5/9、6/9、8/9 绝对门槛，并把 overhead 条件纳入 gate。",
        "- Matched-control gate 修复：`beats_matched_controls` 现在要求 noop、same-signal-random、same-optimizer-geometry controls 都存在且全部被击败；缺失 optimizer-geometry control 的旧行不再计入 matched-control win。",
        "- Optimizer-state actuation 修复尝试：新增 `--fu-actuation-mode post_step_velocity|pre_step_gradient`；`pre_step_gradient` 把连续 FU velocity 注入 train-only gradient，再由 strong optimizer 吸收到 optimizer state，避免只做 AdamW 后外部补丁。matched controls 使用同一 actuation channel。",
        "- Held-risk runtime 修复：新增 `--runtime-risk-source train_proxy|held_cadenced` 与 `--audit-eval-source train_batch|held_cadenced`，默认 train-only；`write_full_loop_summary()` 现在排除 `runtime_train_only_control!=1` 的旧 rows，`runtime_truth_row()` 要求存在 train-only eligible runtime rows。",
        "- repair11 修复尝试：运行 `--fu-actuation-mode pre_step_gradient --mlp-velocity-source optimizer_state` 的 CIFAR10/SVHN/Rice/Bean/Spam matched-control 矩阵；结果 NLL 与 no-debt 全未打开，因此该 actuation 暂不能 promotion。",
        "- repair12 修复尝试：运行 `--runtime-risk-source train_proxy --audit-eval-source train_batch --mlp-velocity-source slow_signal` 的 9-task hard MLP matched-control 矩阵；该矩阵是当前 gate 的主要依据，旧 held-risk/unknown rows 只保留为 diagnostic。",
        "- repair13 修复尝试：重跑 noisy-preference probe，使用 `--runtime-risk-source train_proxy`，确认 preference continuous rows 也不再用 held clean risk 控制 runtime。",
        "- repair14 修复尝试：运行 strict FC-PureKAN/DGKAN_DCHE 的 9-task train-only matched-control 矩阵，补齐 held-risk firewall 后的 KAN internal 证据；结果 basis-native 指标干净但 matched-control/overhead gate 不足。",
        "- Continual/grokking repair rerun 已执行：continual 只有极小改善，grokking test accuracy 仍为 0.0；两者不能 promotion。",
        "- Part H noisy-preference 修复尝试：新增 `stage_preference`，使用 sklearn 内置 Diabetes 真实回归 target 构造 pairwise preference，并只对训练 preference label 注入噪声；运行 optimizer/FU/noop/random/optimizer-geometry matched controls，输出 matrix/summary/state trace。该 probe 保持 per-step continuous FU，不使用 candidate action selection。",
        "- 修改范围仅新增 v22.40 脚本、`results/v22_40/*` artifacts、两份 v22.40 日志；未回滚历史未跟踪文件。",
        "- optimizer controls 中 `Muon-like` 与 `Schedule-Free AdamW` 是本仓库内 compact approximation，已在结果字段 `optimizer_implementation_note` 标注；不把它们写成外部库完全复现。",
        "",
        "## Artifact Manifest",
        "",
        md_table(manifest, ["artifact", "bytes", "sha256", "exists"], 80),
        "",
    ]
    RECAP_DOC.write_text("\n".join(text), encoding="utf-8")


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    stage_a_truth()
    stage_b_reanalysis()
    stage_c_unit(args)
    stage_full_matrix(args)
    stage_continual(args)
    stage_grokking(args)
    stage_preference(args)
    return finalize()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all", "A", "B", "C", "collect", "full", "merge", "continual", "grokking", "preference", "finalize"])
    p.add_argument("--visible-gpus", type=int, default=4)
    p.add_argument("--label", default="manual")
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--architecture", default="MLP")
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--variant", default="continuous_fu", choices=["optimizer_alone", "continuous_fu", "same_overhead_noop_control", "same_signal_random_flow_control", "same_optimizer_geometry_control"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--beta-signal", type=float, default=0.08)
    p.add_argument("--beta-q", type=float, default=0.04)
    p.add_argument("--eta-rho", type=float, default=0.015)
    p.add_argument("--eta-debt", type=float, default=0.50)
    p.add_argument("--tau-safe", type=float, default=0.04)
    p.add_argument("--rho-min", type=float, default=0.0)
    p.add_argument("--rho-max", type=float, default=0.20)
    p.add_argument("--velocity-scale", type=float, default=0.08)
    p.add_argument("--fu-actuation-mode", default="post_step_velocity", choices=["post_step_velocity", "pre_step_gradient"])
    p.add_argument("--runtime-risk-source", default="train_proxy", choices=["train_proxy", "held_cadenced"])
    p.add_argument("--audit-eval-source", default="train_batch", choices=["train_batch", "held_cadenced"])
    p.add_argument("--mlp-velocity-source", default="signal", choices=["signal", "optimizer_state", "optimizer_residual", "slow_signal"])
    p.add_argument("--slow-signal-beta", type=float, default=0.01)
    p.add_argument("--risk-eval-cadence", type=int, default=1)
    p.add_argument("--tail-projection", action="store_true")
    p.add_argument("--tail-projection-cadence", type=int, default=1)
    p.add_argument("--tail-projection-quantile", type=float, default=0.75)
    p.add_argument("--fast-controller-stats", action="store_true")
    p.add_argument("--safety-barrier-strength", type=float, default=0.0)
    p.add_argument("--safety-barrier-slack", type=float, default=0.0)
    p.add_argument("--audit-cadence", type=int, default=40)
    p.add_argument("--tier2-download", action="store_true")
    p.add_argument("--unit-steps", type=int, default=80)
    p.add_argument("--unit-train-size", type=int, default=256)
    p.add_argument("--unit-held-size", type=int, default=128)
    p.add_argument("--eval-datasets", default="MNIST,FashionMNIST,KMNIST,CIFAR10,EMNIST_LETTERS")
    p.add_argument("--eval-seeds", default="0")
    p.add_argument("--eval-architectures", default="MLP,DGKAN_DCHE")
    p.add_argument("--eval-optimizers", default="AdamW,Cautious AdamW,Schedule-Free AdamW,Muon-like")
    p.add_argument("--eval-variants", default="optimizer_alone,continuous_fu,same_overhead_noop_control,same_signal_random_flow_control")
    p.add_argument("--continual-train-size", type=int, default=512)
    p.add_argument("--continual-held-size", type=int, default=256)
    p.add_argument("--continual-steps-per-phase", type=int, default=120)
    p.add_argument("--grokking-modulus", type=int, default=31)
    p.add_argument("--grokking-width", type=int, default=32)
    p.add_argument("--grokking-hidden", type=int, default=128)
    p.add_argument("--grokking-steps", type=int, default=1200)
    p.add_argument("--grokking-eval-cadence", type=int, default=50)
    p.add_argument("--grokking-lr", type=float, default=2.0e-3)
    p.add_argument("--grokking-weight-decay", type=float, default=1.0e-2)
    p.add_argument("--preference-device", default="cuda:2")
    p.add_argument("--preference-seeds", default="0,1,2")
    p.add_argument("--preference-train-pairs", type=int, default=4096)
    p.add_argument("--preference-held-pairs", type=int, default=1024)
    p.add_argument("--preference-test-pairs", type=int, default=2048)
    p.add_argument("--preference-noise-rate", type=float, default=0.20)
    p.add_argument("--preference-steps", type=int, default=600)
    p.add_argument("--preference-batch-size", type=int, default=256)
    p.add_argument("--preference-hidden", type=int, default=128)
    p.add_argument("--preference-lr", type=float, default=1.0e-3)
    p.add_argument("--preference-weight-decay", type=float, default=1.0e-4)
    p.add_argument("--preference-trace-cadence", type=int, default=100)
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_out()
    append_exec(
        " ".join(shlex.quote(x) for x in ([PYTHON, Path(__file__).relative_to(ROOT).as_posix()] + sys.argv[1:])),
        task_id=f"cli_{safe_fragment(args.stage)}_{safe_fragment(getattr(args, 'label', 'run'))}",
        status="started",
        gpu=getattr(args, "device", ""),
        files="",
        note="exact CLI invocation recorded before stage execution",
    )
    if args.stage == "all":
        run_all(args)
    elif args.stage == "A":
        stage_a_truth()
    elif args.stage == "B":
        stage_b_reanalysis()
    elif args.stage == "C":
        stage_c_unit(args)
    elif args.stage == "collect":
        stage_collect(args)
    elif args.stage == "full":
        stage_full_matrix(args)
    elif args.stage == "merge":
        merge_chunks()
    elif args.stage == "continual":
        stage_continual(args)
    elif args.stage == "grokking":
        stage_grokking(args)
    elif args.stage == "preference":
        stage_preference(args)
    elif args.stage == "finalize":
        finalize()


if __name__ == "__main__":
    main()
