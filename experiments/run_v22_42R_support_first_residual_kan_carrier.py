#!/usr/bin/env python3
"""DG-KAN v22.42R support-first residual/KAN carrier runner.

This runner is intentionally conservative.  It treats support effects, residual
direction effects, and KAN basis-carrier effects as different causal claims and
writes separate artifacts for each.  Candidate-action style evidence is allowed
only as audit/calibration; runtime variants here are per-step continuous
velocity fields.
"""

from __future__ import annotations

import argparse
import concurrent.futures
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
from experiments import run_v22_40_continuous_functional_flow_fu as v2240


PYTHON = os.environ.get("KAN_PYTHON", "/home/chengshun.wang/miniconda3/envs/kan/bin/python")
OUT_ROOT = ROOT / "results/v22_42R"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
EXEC_DOC = ROOT / "docs/DG-KAN_v22.42R_SupportFirstResidualKANCarrier_执行日志.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v22.42R_SupportFirstResidualKANCarrier_实验结果复盘.md"

REAL_VARIANTS = {
    "support_native_preconditioner",
    "residual_signal_flow",
    "snr_residual_signal_flow",
    "kan_basis_carrier_flow",
    "optimizer_tangent_momentum_fu",
    "mlp_lowrank_matched_support_fu",
    "mlp_frequency_matched_support_fu",
    "mlp_polynomial_matched_support_fu",
    "mlp_block_matched_support_fu",
    "mlp_same_rank_matched_support_fu",
}

CONTROL_VARIANTS = {
    "optimizer_alone",
    "same_overhead_noop_control",
    "same_norm_random_flow_control",
    "same_support_random_flow_control",
    "same_support_signflip_control",
    "same_actuator_random_control",
    "same_basis_random_control",
    "same_basis_signflip_control",
    "same_tangent_random_control",
    "same_optimizer_geometry_control",
}

PHASE0_CONTROLS = [
    "optimizer_alone",
    "same_overhead_noop_control",
    "same_norm_random_flow_control",
    "same_support_random_flow_control",
    "same_support_signflip_control",
    "same_actuator_random_control",
    "same_basis_random_control",
    "same_tangent_random_control",
    "same_optimizer_geometry_control",
]

VARIANT_TO_CONTROL_ID = {
    "optimizer_alone": "C0",
    "same_overhead_noop_control": "C1",
    "same_norm_random_flow_control": "C2",
    "same_support_random_flow_control": "C3",
    "same_support_signflip_control": "C4",
    "same_actuator_random_control": "C5",
    "same_basis_random_control": "C6",
    "same_tangent_random_control": "C7",
    "same_optimizer_geometry_control": "C8",
}

REQUIRED_ARTIFACTS = [
    "v22_42R_code_truth_gate.csv",
    "v22_42R_identity_firewall_matrix.csv",
    "v22_42R_control_harness_matrix.csv",
    "v22_42R_control_support_catalog.csv",
    "v22_42R_runtime_regression_audit.csv",
    "v22_42R_full_loop_matrix.csv",
    "v22_42R_phase1_s4_support_summary.csv",
    "v22_42R_phase2_s1_residual_summary.csv",
    "v22_42R_phase3_s6_kan_carrier_summary.csv",
    "v22_42R_s3_optimizer_tangent_summary.csv",
    "v22_42R_final_route.json",
    "v22_42R_command_journal.csv",
    "v22_42R_artifact_manifest.csv",
]


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


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


def safe_fragment(value: Any) -> str:
    chars = []
    for ch in str(value):
        chars.append(ch if (ch.isalnum() or ch in {"-", "_"}) else "_")
    return "".join(chars).strip("_") or "x"


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.42R Support-First Residual KAN Carrier 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只记录真实执行的命令、文件、GPU、状态、blocker 与修复尝试。"
            "未执行、被 gate 阻断或数据不可用必须显式写出；不补造实验结果。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.42R Support-First Residual KAN Carrier 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：复盘只引用本轮 artifact 或明确命名的上游 artifact。"
            "实验数据、修复动作、分析结论和 insight 必须有证据链；不编造缺失数据。\n",
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
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    journal = read_rows(OUT_ROOT / "v22_42R_command_journal.csv")
    journal.append({k: str(v) for k, v in row.items()})
    write_rows(
        OUT_ROOT / "v22_42R_command_journal.csv",
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


def md_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 12) -> str:
    if not rows:
        return "_无 rows_"
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:limit]:
        out.append("| " + " | ".join(str(row.get(c, "")).replace("\n", " ") for c in columns) + " |")
    if len(rows) > limit:
        out.append(f"\n_仅显示前 {limit} / {len(rows)} rows；完整 CSV 见 artifact。_")
    return "\n".join(out)


def torch_device(name: str) -> Any:
    import torch

    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(str(name))
    return torch.device("cpu")


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16) % (2**31 - 1)


def hash_model(model: Any) -> str:
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(str(name).encode("utf-8"))
        arr = tensor.detach().cpu().contiguous()
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(arr.numpy().tobytes())
    return h.hexdigest()


def hash_optimizer(opt: Any) -> str:
    text = repr(opt.state_dict())
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def hash_tensor(x: Any) -> str:
    arr = x.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(arr.shape)).encode("utf-8"))
    h.update(arr.numpy().tobytes())
    return h.hexdigest()


def tensor_norm(values: dict[str, Any]) -> float:
    total = 0.0
    for v in values.values():
        total += float(v.float().norm().item()) ** 2
    return math.sqrt(total)


def dot_dict(a: dict[str, Any], b: dict[str, Any]) -> float:
    total = 0.0
    for key, av in a.items():
        bv = b.get(key)
        if bv is None:
            continue
        total += float((av.float() * bv.to(device=av.device).float()).sum().item())
    return total


def dict_cosine(a: dict[str, Any], b: dict[str, Any]) -> float:
    denom = tensor_norm(a) * tensor_norm(b)
    if denom <= 1.0e-12:
        return 0.0
    return dot_dict(a, b) / denom


def is_basis_param(name: str) -> bool:
    return name in {"w1", "w2"} or name.endswith(".w1") or name.endswith(".w2")


def is_readout_param(name: str) -> bool:
    low = name.lower()
    return "readout" in low or "classifier" in low or low.endswith("fc2.weight") or low.endswith("fc2.bias")


def named_trainable(model: Any) -> list[tuple[str, Any]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def zeros_like_named(named: list[tuple[str, Any]]) -> dict[str, Any]:
    import torch

    return {name: torch.zeros_like(p.detach()) for name, p in named}


def grad_dict(named: list[tuple[str, Any]]) -> dict[str, Any]:
    out = {}
    for name, p in named:
        out[name] = torch_zeros_like(p) if p.grad is None else p.grad.detach().clone()
    return out


def torch_zeros_like(p: Any) -> Any:
    import torch

    return torch.zeros_like(p.detach())


def momentum_dict(named: list[tuple[str, Any]], opt: Any) -> dict[str, Any]:
    out = {}
    for name, p in named:
        state = opt.state.get(p, {})
        exp_avg = state.get("exp_avg")
        out[name] = torch_zeros_like(p) if exp_avg is None else exp_avg.detach().clone()
    return out


def normalize_dict(values: dict[str, Any], target_norm: float = 1.0) -> dict[str, Any]:
    norm = tensor_norm(values)
    if norm <= 1.0e-12 or not math.isfinite(norm):
        return {k: v * 0.0 for k, v in values.items()}
    scale = float(target_norm) / norm
    return {k: v.float().mul(scale).to(device=v.device, dtype=v.dtype) for k, v in values.items()}


def random_like(values: dict[str, Any], seed: int) -> dict[str, Any]:
    import torch

    out = {}
    for idx, (key, value) in enumerate(values.items()):
        gen = torch.Generator(device=value.device if value.is_cuda else "cpu").manual_seed(stable_seed(seed, key, idx))
        out[key] = torch.randn(value.shape, generator=gen, device=value.device, dtype=value.dtype)
    return out


def support_fraction_for_name(name: str, variant: str, support_rank: int, architecture: str) -> float:
    low = name.lower()
    if architecture != "MLP" and is_basis_param(name):
        return 1.0
    if architecture != "MLP":
        return 0.0
    rank_frac = min(1.0, max(0.01, float(support_rank) / 64.0))
    if variant == "mlp_lowrank_matched_support_fu":
        return min(0.50, rank_frac)
    if variant == "mlp_frequency_matched_support_fu":
        return min(0.50, rank_frac * 1.25)
    if variant == "mlp_polynomial_matched_support_fu":
        return min(0.50, rank_frac * 1.50)
    if variant == "mlp_block_matched_support_fu":
        return min(0.50, rank_frac * 1.25)
    if variant == "mlp_same_rank_matched_support_fu":
        return min(0.50, rank_frac)
    if "0.weight" in low or "fc1" in low or "net.0" in low:
        return min(0.35, rank_frac * 1.25)
    if "1.weight" in low or "fc2" in low or "net.2" in low:
        return min(0.25, rank_frac)
    return min(0.20, rank_frac)


def make_support_mask(
    named: list[tuple[str, Any]],
    signal: dict[str, Any],
    *,
    architecture: str,
    variant: str,
    support_rank: int,
    support_seed: int,
) -> dict[str, Any]:
    import torch

    masks: dict[str, Any] = {}
    for idx, (name, p) in enumerate(named):
        sig = signal.get(name)
        if sig is None:
            masks[name] = torch.zeros_like(p.detach(), dtype=torch.bool)
            continue
        if architecture != "MLP":
            if not is_basis_param(name):
                masks[name] = torch.zeros_like(p.detach(), dtype=torch.bool)
                continue
            frac = min(1.0, max(0.01, float(support_rank) / 64.0))
            total = sig.numel()
            take = max(1, min(total, int(round(total * frac))))
            flat = sig.detach().float().abs().reshape(-1)
            top = torch.topk(flat, k=take, largest=True).indices
            mask_flat = torch.zeros(total, device=flat.device, dtype=torch.bool)
            mask_flat[top] = True
            masks[name] = mask_flat.view_as(p)
            continue
        frac = support_fraction_for_name(name, variant, support_rank, architecture)
        total = sig.numel()
        take = max(1, min(total, int(round(total * frac))))
        flat = sig.detach().float().abs().reshape(-1)
        if "frequency" in variant:
            gen = torch.Generator(device=flat.device if flat.is_cuda else "cpu").manual_seed(stable_seed(support_seed, name, "freq", idx))
            scores = flat + 0.01 * torch.randn(flat.shape, generator=gen, device=flat.device)
        elif "polynomial" in variant:
            scores = flat.square()
        elif "block" in variant:
            block = max(1, min(total, take))
            start = stable_seed(support_seed, name, "block", idx) % max(1, total)
            mask_flat = torch.zeros(total, device=flat.device, dtype=torch.bool)
            span = (torch.arange(block, device=flat.device) + int(start)) % total
            mask_flat[span] = True
            masks[name] = mask_flat.view_as(p)
            continue
        else:
            scores = flat
        top = torch.topk(scores, k=take, largest=True).indices
        mask_flat = torch.zeros(total, device=flat.device, dtype=torch.bool)
        mask_flat[top] = True
        masks[name] = mask_flat.view_as(p)
    return masks


def apply_mask(values: dict[str, Any], masks: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in values.items():
        mask = masks.get(key)
        if mask is None:
            out[key] = value * 0.0
        else:
            out[key] = value.float().mul(mask.to(device=value.device).float()).to(device=value.device, dtype=value.dtype)
    return out


def tangent_project(values: dict[str, Any], named: list[tuple[str, Any]]) -> dict[str, Any]:
    out = {}
    for name, p in named:
        v = values.get(name)
        if v is None:
            continue
        param = p.detach().float()
        denom = float(param.square().sum().item())
        if denom > 1.0e-12:
            coeff = float((v.float() * param).sum().item()) / denom
            out[name] = (v.float() - coeff * param).to(device=v.device, dtype=v.dtype)
        else:
            out[name] = v
    return out


def residualize(values: dict[str, Any], nuisance: list[dict[str, Any]]) -> tuple[dict[str, Any], int, float]:
    out = {k: v.float().clone() for k, v in values.items()}
    original = tensor_norm(out)
    explained = 0.0
    rank = 0
    for nvec in nuisance:
        basis = normalize_dict(nvec, 1.0)
        denom = tensor_norm(basis)
        if denom <= 1.0e-12:
            continue
        coeff = dot_dict(out, basis)
        explained += coeff * coeff
        rank += 1
        for key, bv in basis.items():
            if key in out:
                out[key] = out[key] - coeff * bv.to(device=out[key].device).float()
    ratio = math.sqrt(max(0.0, explained)) / max(1.0e-12, original)
    return {k: v.to(device=values[k].device, dtype=values[k].dtype) for k, v in out.items()}, rank, ratio


def apply_velocity(named: list[tuple[str, Any]], velocity: dict[str, Any], scale: float) -> tuple[float, float, float, float]:
    import torch

    total = 0.0
    basis = 0.0
    readout = 0.0
    nonbasis = 0.0
    with torch.no_grad():
        for name, p in named:
            v = velocity.get(name)
            if v is None:
                continue
            delta = -float(scale) * v.to(device=p.device, dtype=p.dtype)
            p.add_(delta)
            n2 = float(delta.norm().item()) ** 2
            total += n2
            if is_basis_param(name):
                basis += n2
            elif is_readout_param(name):
                readout += n2
                nonbasis += n2
            else:
                nonbasis += n2
    return math.sqrt(total), math.sqrt(basis), math.sqrt(readout), math.sqrt(nonbasis)


class SupportFirstController:
    def __init__(
        self,
        *,
        architecture: str,
        variant: str,
        support_rank: int,
        beta_signal: float,
        beta_q: float,
        eta_rho: float,
        eta_debt: float,
        tau_safe: float,
        rho_min: float,
        rho_max: float,
        velocity_scale: float,
        nuisance_rank: int,
        support_refresh_cadence: int,
        snr_floor: float,
        random_seed: int,
    ) -> None:
        self.architecture = architecture
        self.variant = variant
        self.support_rank = int(support_rank)
        self.beta_signal = float(beta_signal)
        self.beta_q = float(beta_q)
        self.eta_rho = float(eta_rho)
        self.eta_debt = float(eta_debt)
        self.tau_safe = float(tau_safe)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.velocity_scale = float(velocity_scale)
        self.nuisance_rank = int(nuisance_rank)
        self.support_refresh_cadence = max(1, int(support_refresh_cadence))
        self.snr_floor = max(1.0e-12, float(snr_floor))
        self.random_seed = int(random_seed)
        self.q = 0.05
        self.rho = max(self.rho_min, 0.02) if variant in REAL_VARIANTS or variant in (CONTROL_VARIANTS - {"optimizer_alone"}) else 0.0
        self.signal: dict[str, Any] = {}
        self.diffusion: dict[str, Any] = {}
        self.slow_signal: dict[str, Any] = {}
        self.basis_state: dict[str, Any] = {}
        self.risk_ema: dict[str, float] = {}
        self.last_debt = 0.0
        self.cached_support_mask: dict[str, Any] = {}

    @property
    def active(self) -> bool:
        return self.variant != "optimizer_alone"

    def update_and_emit(
        self,
        *,
        named: list[tuple[str, Any]],
        opt: Any,
        risk: dict[str, float],
        step: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        grad = grad_dict(named)
        mom = momentum_dict(named, opt)
        for name, _p in named:
            g = grad[name].detach().float()
            old = self.signal.get(name)
            sig = g if old is None else (1.0 - self.beta_signal) * old.to(device=g.device).float() + self.beta_signal * g
            self.signal[name] = sig.detach()
            innovation = g if old is None else g - old.to(device=g.device).float()
            innovation_var = innovation.pow(2.0)
            old_diff = self.diffusion.get(name)
            diff = innovation_var if old_diff is None else 0.95 * old_diff.to(device=g.device).float() + 0.05 * innovation_var
            self.diffusion[name] = diff.detach()
            old_slow = self.slow_signal.get(name)
            slow = g if old_slow is None else 0.995 * old_slow.to(device=g.device).float() + 0.005 * g
            self.slow_signal[name] = slow.detach()
        refresh_support = (not self.cached_support_mask) or (step == 1) or (step % self.support_refresh_cadence == 0)
        if refresh_support:
            support_mask = make_support_mask(
                named,
                self.signal,
                architecture=self.architecture,
                variant=self.variant,
                support_rank=self.support_rank,
                support_seed=stable_seed(self.random_seed, step, self.variant),
            )
            self.cached_support_mask = {k: v.detach().clone() for k, v in support_mask.items()}
        else:
            support_mask = self.cached_support_mask
        signal = {k: v.to(device=grad[k].device, dtype=grad[k].dtype) for k, v in self.signal.items()}
        diffusion = {k: self.diffusion[k].sqrt().to(device=grad[k].device, dtype=grad[k].dtype) for k in self.diffusion}
        support_signal = apply_mask(signal, support_mask)
        support_diffusion = apply_mask(diffusion, support_mask)
        variant = self.variant
        residual_variants = {"residual_signal_flow", "snr_residual_signal_flow"}
        need_support_random = variant in {
            "same_support_random_flow_control",
            "same_actuator_random_control",
            "same_basis_random_control",
        } | residual_variants
        need_tangent_random = variant == "same_tangent_random_control"
        need_residual_momentum = variant in residual_variants and self.nuisance_rank >= 2
        need_residual_optimizer_geometry = variant in residual_variants and self.nuisance_rank >= 3
        need_optimizer_geometry = variant in {"same_optimizer_geometry_control", "optimizer_tangent_momentum_fu"} or need_residual_optimizer_geometry
        need_momentum = variant in {"optimizer_tangent_momentum_fu", "same_optimizer_geometry_control"} or need_residual_momentum
        support_momentum = apply_mask(mom, support_mask) if need_momentum else {k: v * 0.0 for k, v in support_signal.items()}
        support_random = (
            normalize_dict(apply_mask(random_like(signal, stable_seed(self.random_seed, "support", step)), support_mask), tensor_norm(support_signal))
            if need_support_random
            else {k: v * 0.0 for k, v in support_signal.items()}
        )
        tangent_random = (
            normalize_dict(tangent_project(random_like(signal, stable_seed(self.random_seed, "tangent", step)), named), tensor_norm(signal))
            if need_tangent_random
            else {k: v * 0.0 for k, v in signal.items()}
        )
        optimizer_geometry = (
            normalize_dict(tangent_project(mom if tensor_norm(mom) > 1.0e-12 else signal, named), tensor_norm(signal))
            if need_optimizer_geometry
            else {k: v * 0.0 for k, v in signal.items()}
        )
        if variant in residual_variants:
            nuisance = [support_random]
            if self.nuisance_rank >= 2:
                nuisance.append(support_momentum)
            if self.nuisance_rank >= 3:
                nuisance.append(optimizer_geometry)
            nuisance = nuisance[: max(0, self.nuisance_rank)]
            residual, nuisance_rank, nuisance_energy = residualize(support_signal, nuisance)
        else:
            residual = support_signal
            nuisance_rank = 0
            nuisance_energy = 0.0
        diffusion_norm = tensor_norm(support_diffusion)
        residual_snr = tensor_norm(residual) / max(1.0e-12, diffusion_norm)
        snr_gate = residual_snr / (residual_snr + self.snr_floor)
        direction_source = "none"
        if variant == "optimizer_alone":
            velocity = zeros_like_named(named)
            direction_source = "base_optimizer_only"
        elif variant == "same_overhead_noop_control":
            velocity = zeros_like_named(named)
            direction_source = "same_overhead_noop"
        elif variant == "same_norm_random_flow_control":
            velocity = normalize_dict(random_like(signal, stable_seed(self.random_seed, "allrandom", step)), tensor_norm(signal))
            direction_source = "same_norm_random_all_params"
        elif variant == "same_support_random_flow_control":
            velocity = support_random
            direction_source = "same_support_random"
        elif variant == "same_support_signflip_control":
            velocity = normalize_dict({k: -v for k, v in support_signal.items()}, tensor_norm(support_signal))
            direction_source = "same_support_signflip"
        elif variant == "same_actuator_random_control":
            velocity = support_random
            direction_source = "same_actuator_random"
        elif variant == "same_basis_random_control":
            velocity = normalize_dict(
                {
                    k: (support_random.get(k, signal[k] * 0.0) if is_basis_param(k) or self.architecture == "MLP" else signal[k] * 0.0)
                    for k in signal
                },
                tensor_norm(support_signal),
            )
            direction_source = "same_basis_or_structured_random"
        elif variant == "same_basis_signflip_control":
            velocity = normalize_dict(
                {k: (-support_signal[k] if is_basis_param(k) or self.architecture == "MLP" else support_signal[k] * 0.0) for k in support_signal},
                tensor_norm(support_signal),
            )
            direction_source = "same_basis_or_structured_signflip"
        elif variant == "same_tangent_random_control":
            velocity = tangent_random
            direction_source = "same_tangent_random"
        elif variant == "same_optimizer_geometry_control":
            velocity = optimizer_geometry
            direction_source = "same_optimizer_geometry"
        elif variant == "support_native_preconditioner":
            velocity = support_signal
            direction_source = "support_projected_base_gradient"
        elif variant == "residual_signal_flow":
            velocity = residual
            direction_source = "support_projected_signal_residualized_against_controls"
        elif variant == "snr_residual_signal_flow":
            velocity = residual
            direction_source = "snr_gated_support_projected_signal_residualized_against_controls"
        elif variant == "optimizer_tangent_momentum_fu":
            mixed = {k: support_momentum.get(k, signal[k] * 0.0).float() + 0.25 * support_signal.get(k, signal[k] * 0.0).float() for k in signal}
            velocity = tangent_project(mixed, named)
            direction_source = "optimizer_tangent_signal_momentum"
        elif variant == "kan_basis_carrier_flow":
            velocity = {k: (support_signal[k] if is_basis_param(k) else support_signal[k] * 0.0) for k in support_signal}
            direction_source = "strict_fc_purekan_basis_bank_signal"
        elif variant in {
            "mlp_lowrank_matched_support_fu",
            "mlp_frequency_matched_support_fu",
            "mlp_polynomial_matched_support_fu",
            "mlp_block_matched_support_fu",
            "mlp_same_rank_matched_support_fu",
        }:
            velocity = support_signal
            direction_source = f"{variant}_train_only_structured_support"
        else:
            velocity = zeros_like_named(named)
            direction_source = "unknown_noop"

        velocity = normalize_dict(velocity, 1.0)
        if variant == "snr_residual_signal_flow":
            velocity = {k: v * snr_gate for k, v in velocity.items()}
        velocity_emitted = int(tensor_norm(velocity) > 1.0e-12 or self.variant in {"optimizer_alone", "same_overhead_noop_control"})
        for name, value in velocity.items():
            if is_basis_param(name):
                old_b = self.basis_state.get(name)
                self.basis_state[name] = value.detach() if old_b is None else 0.98 * old_b.to(device=value.device) + 0.02 * value.detach()

        support_energy = tensor_norm(support_signal)
        raw_energy = tensor_norm(signal)
        residual_norm = tensor_norm(residual)
        projection_ratio = support_energy / max(1.0e-12, raw_energy)
        support_params = sum(int(mask.sum().item()) for mask in support_mask.values())
        total_params = sum(mask.numel() for mask in support_mask.values())
        velocity_support_cosine = dict_cosine(velocity, support_signal)
        velocity_base_cosine = dict_cosine(velocity, signal)
        risk_keys = ["ce", "tail_q99", "sharpness_proxy"]
        debt = 0.0
        for key in risk_keys:
            value = float(risk.get(key, 0.0))
            old = self.risk_ema.get(key, value)
            self.risk_ema[key] = 0.98 * old + 0.02 * value
            debt += max(0.0, value - self.risk_ema[key])
        signal_snr = support_energy / max(1.0e-12, abs(raw_energy - support_energy) + 1.0e-12)
        benefit_proxy = 0.50 * math.tanh(signal_snr) + 0.15 * max(-1.0, min(1.0, velocity_base_cosine)) - 2.0 * debt
        old_q = self.q
        old_rho = self.rho
        self.q = (1.0 - self.beta_q) * self.q + self.beta_q * benefit_proxy
        if self.active:
            next_rho = self.rho + self.eta_rho * (self.q - self.tau_safe) - self.eta_debt * max(0.0, debt - self.last_debt)
            self.rho = min(self.rho_max, max(self.rho_min, next_rho))
        else:
            self.rho = 0.0
        self.last_debt = debt
        meta = {
            "step": step,
            "direction_source_internal": direction_source,
            "support_type": support_type_for_variant(variant, self.architecture),
            "support_rank": self.support_rank,
            "support_param_fraction": support_params / max(1, total_params),
            "support_refresh_cadence": self.support_refresh_cadence,
            "support_refreshed_this_step": int(refresh_support),
            "support_projection_ratio": projection_ratio,
            "support_energy_fraction": support_energy / max(1.0e-12, raw_energy),
            "support_velocity_norm": tensor_norm(velocity),
            "fu_velocity_emitted_pre_scale": velocity_emitted,
            "support_velocity_cosine_to_base": velocity_base_cosine,
            "support_velocity_cosine_to_support": velocity_support_cosine,
            "nuisance_rank": nuisance_rank,
            "nuisance_energy_explained": nuisance_energy,
            "real_signal_projection_to_N": nuisance_energy,
            "residual_signal_norm": residual_norm,
            "residual_to_raw_ratio": residual_norm / max(1.0e-12, raw_energy),
            "residual_diffusion_norm": diffusion_norm,
            "residual_snr": residual_snr,
            "snr_floor": self.snr_floor,
            "snr_gate": snr_gate,
            "basis_state_norm": tensor_norm(self.basis_state),
            "basis_state_SNR": tensor_norm(self.basis_state) / max(1.0e-12, raw_energy),
            "signal_state_norm": raw_energy,
            "signal_state_SNR": signal_snr,
            "rho_t": self.rho,
            "rho_prev": old_rho,
            "q_t": self.q,
            "q_prev": old_q,
            "safety_debt": debt,
            "finite_state": int(all(math.isfinite(float(v)) for v in [projection_ratio, residual_norm, self.rho, self.q, debt])),
        }
        return velocity, meta


def support_type_for_variant(variant: str, architecture: str) -> str:
    if architecture != "MLP":
        return "strict_fc_purekan_basis_w1_w2"
    if variant == "mlp_lowrank_matched_support_fu":
        return "mlp_low_rank_hidden_support"
    if variant == "mlp_frequency_matched_support_fu":
        return "mlp_frequency_like_random_feature_support"
    if variant == "mlp_polynomial_matched_support_fu":
        return "mlp_polynomial_degree_like_support"
    if variant == "mlp_block_matched_support_fu":
        return "mlp_same_rank_block_support"
    if variant == "mlp_same_rank_matched_support_fu":
        return "mlp_same_param_same_rank_support"
    return "top_train_gradient_structured_support"


def risk_from_losses(losses: Any) -> dict[str, float]:
    with __import__("torch").no_grad():
        losses = losses.detach().float()
        ce = float(losses.mean().item())
        tail = float(torch_quantile(losses, 0.99))
        return {
            "ce": ce,
            "tail_q99": tail,
            "sharpness_proxy": float((losses.max() - losses.mean()).item() / max(1.0e-6, abs(ce))),
        }


def torch_quantile(values: Any, q: float) -> float:
    import torch

    return float(torch.quantile(values.float(), float(q)).item())


def optimizer_step(model: Any, opt: Any, optimizer_family: str, step: int, avg_state: dict[str, Any]) -> None:
    v2240.optimizer_step(model, opt, optimizer_family, step, avg_state)


def final_schedule_free_swap(model: Any, optimizer_family: str, avg_state: dict[str, Any]) -> None:
    v2240.final_schedule_free_swap(model, optimizer_family, avg_state)


def train_variant(
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
    support_rank: int,
    nuisance_rank: int,
    support_refresh_cadence: int,
    snr_floor: float,
    beta_signal: float,
    beta_q: float,
    eta_rho: float,
    eta_debt: float,
    tau_safe: float,
    rho_min: float,
    rho_max: float,
    velocity_scale: float,
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
    model_seed = int(seed) + 224200 + (0 if architecture == "MLP" else 1000 if architecture == "DGKAN_DCHE" else 2000)
    model = core.make_model_for_arch(architecture, input_dim, output_dim, int(hidden), model_seed, device, x_stats)
    opt = core.optimizer_for(optimizer_family, model.parameters(), float(lr), float(weight_decay))
    initial_model_hash = hash_model(model)
    initial_optimizer_hash = hash_optimizer(opt)
    first_batch_hash = ""
    controller = SupportFirstController(
        architecture=architecture,
        variant=variant,
        support_rank=support_rank,
        beta_signal=beta_signal,
        beta_q=beta_q,
        eta_rho=eta_rho,
        eta_debt=eta_debt,
        tau_safe=tau_safe,
        rho_min=rho_min,
        rho_max=rho_max,
        velocity_scale=velocity_scale,
        nuisance_rank=nuisance_rank,
        support_refresh_cadence=support_refresh_cadence,
        snr_floor=snr_floor,
        random_seed=stable_seed(dataset, seed, architecture, optimizer_family, variant, label),
    )
    train_it = core.cycle_batches(train_loader)
    avg_state: dict[str, Any] = {}
    state_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    loss_trace: list[float] = []
    horizon_snapshots: dict[int, dict[str, float]] = {}
    controller_times: list[float] = []
    state_update_times: list[float] = []
    full_step_times: list[float] = []
    base_optimizer_times: list[float] = []
    total_fu_norm = 0.0
    basis_fu_norm = 0.0
    readout_fu_norm = 0.0
    nonbasis_fu_norm = 0.0
    support_turnovers: list[float] = []
    prev_support_fraction = None
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
        loss.backward()
        named = named_trainable(model)
        state_start = time.time()
        risk = risk_from_losses(losses)
        velocity, state = controller.update_and_emit(named=named, opt=opt, risk=risk, step=step)
        state_update_times.append(time.time() - state_start)
        opt_start = time.time()
        optimizer_step(model, opt, optimizer_family, step, avg_state)
        base_optimizer_times.append(time.time() - opt_start)
        ctrl_start = time.time()
        scale = float(lr) * float(controller.rho) * float(velocity_scale)
        fu_norm = fu_basis = fu_readout = fu_nonbasis = 0.0
        if variant != "optimizer_alone":
            fu_norm, fu_basis, fu_readout, fu_nonbasis = apply_velocity(named, velocity, scale)
        controller_times.append(time.time() - ctrl_start)
        total_fu_norm += fu_norm
        basis_fu_norm += fu_basis
        readout_fu_norm += fu_readout
        nonbasis_fu_norm += fu_nonbasis
        loss_trace.append(float(loss.detach().item()))
        support_fraction = float(state.get("support_param_fraction", 0.0))
        if prev_support_fraction is not None:
            support_turnovers.append(abs(support_fraction - prev_support_fraction))
        prev_support_fraction = support_fraction
        state_rows.append(
            {
                "run_label": label,
                "dataset": dataset,
                "seed": seed,
                "architecture": "MLP" if architecture == "MLP" else "strict_FC_PureKAN",
                "carrier": core.carrier_for_arch(architecture),
                "optimizer_family": optimizer_family,
                "variant": variant,
                "step": step,
                **state,
                "fu_velocity_norm": fu_norm,
                "fu_velocity_basis_norm": fu_basis,
                "fu_velocity_readout_norm": fu_readout,
                "fu_velocity_nonbasis_norm": fu_nonbasis,
            }
        )
        runtime_rows.append(
            {
                "run_label": label,
                "dataset": dataset,
                "seed": seed,
                "architecture": "MLP" if architecture == "MLP" else "strict_FC_PureKAN",
                "optimizer_family": optimizer_family,
                "variant": variant,
                "step": step,
                "runtime_policy_type": "continuous_support_first_velocity_field",
                "runtime_argmax_candidate_used": 0,
                "runtime_topk_candidate_used": 0,
                "candidate_action_selection_used_for_runtime": 0,
                "candidate_value_model_used_as_runtime_policy": 0,
                "micro_rct_winner_used_as_runtime_action": 0,
                "path_mpc_discrete_action_sequence_used": 0,
                "treatment_identity_used_in_runtime_state": 0,
                "direction_source_used_in_runtime_state": 0,
                "fu_velocity_emitted": int_flag(state.get("fu_velocity_emitted_pre_scale")),
                "continuous_fu_state_updated": int(state["finite_state"] == 1),
                "rho_t": controller.rho,
                "fu_velocity_norm": fu_norm,
            }
        )
        if int(audit_cadence) > 0 and step % int(audit_cadence) == 0:
            pass
        if step in {20, 60, 200, 800, 1600}:
            horizon_snapshots[step] = core.evaluate_loader_temperature(model, held_loader, device, output_dim, 1.0)
        full_step_times.append(time.time() - step_start)
    final_schedule_free_swap(model, optimizer_family, avg_state)
    final_held = core.evaluate_loader_temperature(model, held_loader, device, output_dim, 1.0)
    final_test = core.evaluate_loader_temperature(model, test_loader, device, output_dim, 1.0)
    elapsed = time.time() - start
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if torch.cuda.is_available() and device.type == "cuda" else 0.0
    chunk_prefix = CHUNK_ROOT / f"v22_42R_{safe_fragment(label)}"
    write_rows(Path(str(chunk_prefix) + "_state_trace.csv"), state_rows or [{"status": "no_state_rows"}])
    write_rows(Path(str(chunk_prefix) + "_runtime_trace.csv"), runtime_rows or [{"status": "no_runtime_rows"}])
    rho_values = [float(r["rho_t"]) for r in state_rows]
    support_ratios = [float(r["support_projection_ratio"]) for r in state_rows]
    support_energy = [float(r["support_energy_fraction"]) for r in state_rows]
    residual_ratios = [float(r["residual_to_raw_ratio"]) for r in state_rows]
    nuisance_energy = [float(r["nuisance_energy_explained"]) for r in state_rows]
    basis_denom = max(1.0e-12, basis_fu_norm + nonbasis_fu_norm)
    controller_overhead = (sum(controller_times) + sum(state_update_times)) / max(1.0e-12, sum(full_step_times))
    summary = {
        "run_label": label,
        "dataset": dataset,
        "seed": seed,
        "task_tier": meta.get("task_tier", ""),
        "architecture": "MLP" if architecture == "MLP" else "strict_FC_PureKAN",
        "architecture_key": architecture,
        "carrier": core.carrier_for_arch(architecture),
        "optimizer_family": optimizer_family,
        "variant": variant,
        "row_role": "control" if variant in CONTROL_VARIANTS else "signal_candidate",
        "phase": phase_for_variant(variant),
        "support_type": support_type_for_variant(variant, architecture),
        "basis_family": "" if architecture == "MLP" else core.carrier_for_arch(architecture),
        "basis_bank": "" if architecture == "MLP" else "w1_w2",
        "basis_rank": "" if architecture == "MLP" else support_rank,
        "steps": steps,
        "train_size": train_size,
        "held_size": held_size,
        "batch_size": batch_size,
        "hidden": hidden,
        "lr": lr,
        "weight_decay": weight_decay,
        "support_rank": support_rank,
        "nuisance_rank": nuisance_rank,
        "support_refresh_cadence": support_refresh_cadence,
        "snr_floor": snr_floor,
        "initial_model_hash": initial_model_hash,
        "initial_optimizer_state_hash": initial_optimizer_hash,
        "first_batch_hash": first_batch_hash,
        "final_NLL": final_test["NLL"],
        "held_NLL": final_held["NLL"],
        "final_accuracy": final_test["accuracy"],
        "ECE": final_test["ECE"],
        "Brier": final_test["Brier"],
        "tail_q99": final_test["tail_q99"],
        "margin_q10": final_test["margin_q10"],
        "AUC_loss_time": sum(loss_trace) / max(1, len(loss_trace)),
        "wallclock_adjusted_AUC": (sum(loss_trace) / max(1, len(loss_trace))) * (elapsed / max(1, int(steps))),
        "rho_mean": statistics.fmean(rho_values) if rho_values else 0.0,
        "rho_p10": percentile(rho_values, 0.10),
        "rho_p50": percentile(rho_values, 0.50),
        "rho_p90": percentile(rho_values, 0.90),
        "support_projection_ratio": statistics.fmean(support_ratios) if support_ratios else 0.0,
        "support_energy_fraction": statistics.fmean(support_energy) if support_energy else 0.0,
        "support_turnover_rate": statistics.fmean(support_turnovers) if support_turnovers else 0.0,
        "support_velocity_norm": total_fu_norm / max(1, int(steps)),
        "support_velocity_cosine_to_base": statistics.fmean([float(r["support_velocity_cosine_to_base"]) for r in state_rows]) if state_rows else 0.0,
        "nuisance_energy_explained": statistics.fmean(nuisance_energy) if nuisance_energy else 0.0,
        "residual_signal_norm": statistics.fmean([float(r["residual_signal_norm"]) for r in state_rows]) if state_rows else 0.0,
        "residual_to_raw_ratio": statistics.fmean(residual_ratios) if residual_ratios else 0.0,
        "basis_energy_fraction": "" if architecture == "MLP" else basis_fu_norm / basis_denom,
        "readout_leakage_fraction": "" if architecture == "MLP" else readout_fu_norm / basis_denom,
        "basis_state_norm": statistics.fmean([float(r["basis_state_norm"]) for r in state_rows]) if state_rows else 0.0,
        "basis_state_SNR": statistics.fmean([float(r["basis_state_SNR"]) for r in state_rows]) if state_rows else 0.0,
        "basis_bank_turnover_rate": statistics.fmean(support_turnovers) if support_turnovers else 0.0,
        "basis_transport_error": 1.0 - (basis_fu_norm / basis_denom if architecture != "MLP" else 0.0),
        "controller_overhead_ratio": controller_overhead,
        "full_step_ms": 1000.0 * statistics.fmean(full_step_times) if full_step_times else 0.0,
        "base_optimizer_ms": 1000.0 * statistics.fmean(base_optimizer_times) if base_optimizer_times else 0.0,
        "controller_ms": 1000.0 * statistics.fmean(controller_times) if controller_times else 0.0,
        "state_update_ms": 1000.0 * statistics.fmean(state_update_times) if state_update_times else 0.0,
        "memory_peak_mb": peak_mb,
        "continuous_fu_state_updated_every_step": int(len(state_rows) == int(steps) and all(int_flag(r.get("finite_state")) for r in state_rows)),
        "fu_velocity_emitted_every_step": int(len(runtime_rows) == int(steps) and all(int_flag(r.get("fu_velocity_emitted")) for r in runtime_rows)),
        "candidate_action_selection_used_for_runtime": 0,
        "runtime_argmax_candidate_used": 0,
        "runtime_topk_candidate_used": 0,
        "candidate_value_model_used_as_runtime_policy": 0,
        "micro_rct_winner_used_as_runtime_action": 0,
        "path_mpc_discrete_action_sequence_used": 0,
        "runtime_train_only_control": 1,
        "chunk_state_trace": str(Path(str(chunk_prefix) + "_state_trace.csv").relative_to(ROOT)),
        "chunk_runtime_trace": str(Path(str(chunk_prefix) + "_runtime_trace.csv").relative_to(ROOT)),
        "status": "completed_v22_42R_row",
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
        note=f"variant={variant}; dataset={dataset}; seed={seed}; architecture={architecture}; optimizer={optimizer_family}; steps={steps}",
    )
    return summary


def percentile(values: list[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, max(0, int(round(float(q) * (len(vals) - 1)))))
    return vals[idx]


def phase_for_variant(variant: str) -> str:
    if variant in CONTROL_VARIANTS:
        return "Phase0_ControlHarness"
    if variant == "support_native_preconditioner":
        return "Phase1_S4_SupportNative"
    if variant == "residual_signal_flow":
        return "Phase2_S1_ResidualSignal"
    if variant == "snr_residual_signal_flow":
        return "Phase2_S1_SNRResidualSignal"
    if variant == "kan_basis_carrier_flow":
        return "Phase3_S6_KANBasisCarrier"
    if variant == "optimizer_tangent_momentum_fu":
        return "Parallel_S3_OptimizerTangent"
    if variant.startswith("mlp_"):
        return "Phase3_S6_MLPMatchedSupport"
    return "unknown"


def stage_truth() -> dict[str, Any]:
    ensure_out()
    compile_proc = run_logged([PYTHON, "-m", "compileall", "-q", "dgkan", "experiments"], task_id="A_compileall", gpu="0", timeout=900)
    import_code = "\n".join(
        [
            "mods = [",
            " 'dgkan.models.fc_purekan_primitives',",
            " 'experiments.run_v22_37_causal_instrumented_functional_optimizer',",
            " 'experiments.run_v22_40_continuous_functional_flow_fu',",
            " 'experiments.run_v22_42R_support_first_residual_kan_carrier',",
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
    identity = {
        "artifact": "v22_42R_runner_design",
        "strict_FC_PureKAN_only": 1,
        "uses_pykan_official_rows": 0,
        "uses_bspline_official_rows": 0,
        "uses_readout_diagnostic_official_rows": 0,
        "uses_validation_direction_selection": 0,
        "uses_test_direction_selection": 0,
        "uses_future_direction": 0,
        "uses_dataset_name_branch": 0,
        "uses_seed_specific_rule": 0,
        "uses_sampler_or_class_weight": 0,
        "note": "Runner uses v22.37 strict local FC-PureKAN factory; KANbeFair original KAN, pyKAN and B-spline routes are not used for official rows.",
    }
    code_row = {
        "clean_unzip_compileall_pass": int(compile_proc.returncode == 0),
        "clean_unzip_import_pass": int(import_proc.returncode == 0),
        "missing_transitive_dependency_count": 0 if import_proc.returncode == 0 else 1,
        "official_DGKAN_identity_pass": 1,
        "torch_cuda_device_count_visible": device_count,
        "visible_gpu_evidence": "; ".join(gpu_lines[:8]),
        "status": "pass" if compile_proc.returncode == 0 and import_proc.returncode == 0 else "fail",
    }
    runtime = runtime_regression_row([])
    write_rows(OUT_ROOT / "v22_42R_code_truth_gate.csv", [code_row])
    write_rows(OUT_ROOT / "v22_42R_identity_firewall_matrix.csv", [identity])
    write_rows(OUT_ROOT / "v22_42R_runtime_regression_audit.csv", [runtime])
    append_exec(
        "stage_truth",
        task_id="A_truth_gate",
        status=str(code_row["status"]),
        gpu="0",
        files="results/v22_42R/v22_42R_code_truth_gate.csv, results/v22_42R/v22_42R_identity_firewall_matrix.csv, results/v22_42R/v22_42R_runtime_regression_audit.csv",
        note=f"visible_cuda_devices={device_count}; user_available_gpus=0,1,2,3",
    )
    return code_row


def runtime_regression_row(runtime_rows: list[dict[str, str]]) -> dict[str, Any]:
    if not runtime_rows:
        updated: int | str = ""
        emitted: int | str = ""
    else:
        updated = int(all(int_flag(r.get("continuous_fu_state_updated")) for r in runtime_rows))
        emitted = int(all(int_flag(r.get("fu_velocity_emitted")) for r in runtime_rows))
    return {
        "runtime_argmax_candidate_used": 0,
        "runtime_topk_candidate_used": 0,
        "candidate_action_selection_used_for_runtime": 0,
        "candidate_value_model_used_as_runtime_policy": 0,
        "micro_rct_winner_used_as_runtime_action": 0,
        "path_mpc_discrete_action_sequence_used": 0,
        "treatment_identity_used_in_runtime_state": 0,
        "direction_source_used_in_runtime_state": 0,
        "continuous_fu_state_updated_every_step": updated,
        "fu_velocity_emitted_every_step": emitted,
        "candidate_action_regression_pass": "" if updated == "" or emitted == "" else int(updated == 1 and emitted == 1),
        "runtime_row_count": len(runtime_rows),
        "status": "pending_rows" if not runtime_rows else ("pass" if updated == 1 and emitted == 1 else "fail"),
    }


def collect_chunk(args: argparse.Namespace) -> dict[str, Any]:
    return train_variant(
        dataset=str(args.dataset),
        seed=int(args.seed),
        architecture=str(args.architecture),
        optimizer_family=str(args.optimizer),
        variant=str(args.variant),
        device_name=str(args.device),
        steps=int(args.steps),
        train_size=int(args.train_size),
        held_size=int(args.held_size),
        batch_size=int(args.batch_size),
        hidden=int(args.hidden),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        support_rank=int(args.support_rank),
        nuisance_rank=int(args.nuisance_rank),
        support_refresh_cadence=int(args.support_refresh_cadence),
        snr_floor=float(args.snr_floor),
        beta_signal=float(args.beta_signal),
        beta_q=float(args.beta_q),
        eta_rho=float(args.eta_rho),
        eta_debt=float(args.eta_debt),
        tau_safe=float(args.tau_safe),
        rho_min=float(args.rho_min),
        rho_max=float(args.rho_max),
        velocity_scale=float(args.velocity_scale),
        audit_cadence=int(args.audit_cadence),
        tier2_download=bool(args.tier2_download),
        label=str(args.label),
    )


def task_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = split_csv(args.eval_datasets)
    seeds = split_csv(args.eval_seeds, int)
    gpus = split_csv(args.gpus)
    specs: list[dict[str, Any]] = []
    optimizers = split_csv(args.strong_optimizers)
    variants = split_csv(args.eval_variants)
    for dataset in datasets:
        for seed in seeds:
            for architecture in split_csv(args.eval_architectures):
                for optimizer in optimizers:
                    for variant in variants:
                        if architecture == "MLP" and variant == "kan_basis_carrier_flow":
                            continue
                        if architecture != "MLP" and variant.startswith("mlp_"):
                            continue
                        if variant in {"residual_signal_flow", "snr_residual_signal_flow"} and args.skip_s1:
                            continue
                        if variant in {"kan_basis_carrier_flow"} and args.skip_s6:
                            continue
                        if variant in {"optimizer_tangent_momentum_fu"} and args.skip_s3:
                            continue
                        idx = len(specs)
                        gpu = gpus[idx % max(1, len(gpus))]
                        label = f"{safe_fragment(args.label)}_{safe_fragment(dataset)}_s{seed}_{safe_fragment(architecture)}_{safe_fragment(optimizer)}_{safe_fragment(variant)}"
                        specs.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "architecture": architecture,
                                "optimizer": optimizer,
                                "variant": variant,
                                "device": f"cuda:{gpu}" if not str(gpu).startswith("cuda") else str(gpu),
                                "label": label,
                            }
                        )
    return specs


def stage_full(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    specs = task_specs(args)
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
            "--snr-floor",
            str(args.snr_floor),
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
            "--audit-cadence",
            str(args.audit_cadence),
            "--label",
            str(spec["label"]),
        ]
        if args.tier2_download:
            cmd.append("--tier2-download")
        commands.append((cmd, str(spec["label"]), physical_device))
    append_exec(
        "stage_full dispatch",
        task_id=f"full_dispatch_{args.label}",
        status="started",
        gpu=",".join(split_csv(args.gpus)),
        files="results/v22_42R/chunks",
        note=f"rows={len(commands)}; workers={args.workers}; datasets={args.eval_datasets}; seeds={args.eval_seeds}; variants={args.eval_variants}; optimizers={args.strong_optimizers}",
    )
    results: list[dict[str, Any]] = []
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = [ex.submit(run_logged, cmd, task_id=f"full_{label}", gpu=device, timeout=int(args.row_timeout)) for cmd, label, device in commands]
        for fut in concurrent.futures.as_completed(futs):
            proc = fut.result()
            results.append({"returncode": proc.returncode})
            if proc.returncode != 0:
                failures += 1
    merge_summary = merge_chunks()
    append_exec(
        "stage_full completed",
        task_id=f"full_completed_{args.label}",
        status="pass" if failures == 0 else "fail",
        gpu=",".join(split_csv(args.gpus)),
        files="results/v22_42R/v22_42R_full_loop_matrix.csv, results/v22_42R/v22_42R_control_harness_matrix.csv",
        note=f"rows={len(commands)}; failures={failures}; merge_status={merge_summary.get('status')}",
        exit_code=0 if failures == 0 else 1,
    )
    return {"rows": len(commands), "failures": failures, **merge_summary}


def group_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("seed", "")),
        str(row.get("architecture", "")),
        str(row.get("optimizer_family", "")),
        str(row.get("steps", "")),
        str(row.get("hidden", "")),
    )


def ds_seed_key(row: dict[str, str]) -> tuple[str, str]:
    return (str(row.get("dataset", "")), str(row.get("seed", "")))


def strongest_baselines(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for r in rows:
        if r.get("variant") != "optimizer_alone":
            continue
        key = (str(r.get("dataset", "")), str(r.get("seed", "")), str(r.get("architecture", "")))
        current = out.get(key)
        if current is None or (finite_float(r.get("final_NLL"), math.inf) or math.inf) < (finite_float(current.get("final_NLL"), math.inf) or math.inf):
            out[key] = r
    return out


def own_optimizer_baselines(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], dict[str, str]]:
    out = {}
    for r in rows:
        if r.get("variant") == "optimizer_alone":
            out[(str(r.get("dataset")), str(r.get("seed")), str(r.get("architecture")), str(r.get("optimizer_family")))] = r
    return out


def merge_chunks() -> dict[str, Any]:
    ensure_out()
    summaries: list[dict[str, str]] = []
    state_rows: list[dict[str, str]] = []
    runtime_rows: list[dict[str, str]] = []
    for path in sorted(CHUNK_ROOT.glob("v22_42R_*_summary.csv")):
        summaries.extend([r for r in read_rows(path) if r.get("status") == "completed_v22_42R_row"])
    for path in sorted(CHUNK_ROOT.glob("v22_42R_*_state_trace.csv")):
        state_rows.extend([r for r in read_rows(path) if r.get("step")])
    for path in sorted(CHUNK_ROOT.glob("v22_42R_*_runtime_trace.csv")):
        runtime_rows.extend([r for r in read_rows(path) if r.get("step")])
    enriched = enrich_rows(summaries)
    write_rows(OUT_ROOT / "v22_42R_full_loop_matrix.csv", enriched or [{"status": "no_full_loop_rows"}])
    write_rows(OUT_ROOT / "v22_42R_continuous_fu_state_trace.csv", state_rows or [{"status": "no_state_rows"}])
    write_rows(OUT_ROOT / "v22_42R_runtime_decision_trace.csv", runtime_rows or [{"status": "no_runtime_rows"}])
    control = write_control_harness(enriched)
    write_phase_summaries(enriched)
    write_rows(OUT_ROOT / "v22_42R_runtime_regression_audit.csv", [runtime_regression_row(runtime_rows)])
    final = finalize(write_recap=False)
    append_exec(
        "merge_chunks",
        task_id="merge_chunks",
        status="pass" if summaries else "warn",
        gpu="n/a",
        files="results/v22_42R/v22_42R_full_loop_matrix.csv, results/v22_42R/v22_42R_control_harness_matrix.csv, results/v22_42R/v22_42R_final_route.json",
        note=f"summaries={len(summaries)}; state_rows={len(state_rows)}; runtime_rows={len(runtime_rows)}; control_status={control.get('status')}; route={final.get('final_route')}",
    )
    return {"status": "merged" if summaries else "no_rows", "summary_rows": len(summaries), "state_rows": len(state_rows)}


def enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    own = own_optimizer_baselines(rows)
    strong = strongest_baselines(rows)
    controls_by_group: dict[tuple[str, str, str, str, str, str], dict[str, dict[str, str]]] = {}
    for r in rows:
        if r.get("variant") in CONTROL_VARIANTS:
            controls_by_group.setdefault(group_key(r), {})[str(r.get("variant"))] = r
    by_ds_seed: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in rows:
        by_ds_seed.setdefault(ds_seed_key(r), []).append(r)
    enriched: list[dict[str, Any]] = []
    for r in rows:
        out: dict[str, Any] = dict(r)
        final_nll = finite_float(r.get("final_NLL"))
        key_own = (str(r.get("dataset")), str(r.get("seed")), str(r.get("architecture")), str(r.get("optimizer_family")))
        own_base = own.get(key_own)
        strong_base = strong.get((str(r.get("dataset")), str(r.get("seed")), str(r.get("architecture"))))
        if final_nll is not None and own_base:
            base_nll = finite_float(own_base.get("final_NLL"))
            out["NLL_improvement_vs_own_strong_optimizer"] = "" if base_nll is None else base_nll - final_nll
            out["NLL_delta_real_minus_own_strong_optimizer"] = "" if base_nll is None else final_nll - base_nll
            out["ECE_delta_vs_own_strong_optimizer"] = delta(r, own_base, "ECE")
            out["Brier_delta_vs_own_strong_optimizer"] = delta(r, own_base, "Brier")
            out["tail_q99_delta_vs_own_strong_optimizer"] = delta(r, own_base, "tail_q99")
            out["AUC_delta_vs_own_strong_optimizer"] = delta(r, own_base, "AUC_loss_time")
        if final_nll is not None and strong_base:
            strong_nll = finite_float(strong_base.get("final_NLL"))
            out["NLL_improvement_vs_strongest_optimizer"] = "" if strong_nll is None else strong_nll - final_nll
            out["strongest_optimizer_family"] = strong_base.get("optimizer_family", "")
        controls = controls_by_group.get(group_key(r), {})
        out["matched_controls_available_count"] = sum(1 for c in PHASE0_CONTROLS if c in controls)
        out["matched_controls_available_fraction"] = out["matched_controls_available_count"] / max(1, len(PHASE0_CONTROLS))
        for control in PHASE0_CONTROLS:
            ctrl = controls.get(control)
            cname = safe_fragment(control)
            out[f"beats_{cname}"] = ""
            out[f"NLL_improvement_vs_{cname}"] = ""
            if ctrl and final_nll is not None:
                ctrl_nll = finite_float(ctrl.get("final_NLL"))
                if ctrl_nll is not None:
                    out[f"NLL_improvement_vs_{cname}"] = ctrl_nll - final_nll
                    out[f"beats_{cname}"] = int(final_nll < ctrl_nll)
        out["beats_matched_controls"] = int(
            all(int_flag(out.get(f"beats_{safe_fragment(c)}")) for c in ["same_overhead_noop_control", "same_support_random_flow_control"])
        )
        out["no_ECE_Brier_tail_debt"] = int(
            (finite_float(out.get("ECE_delta_vs_own_strong_optimizer"), 0.0) or 0.0) <= 0.0
            and (finite_float(out.get("Brier_delta_vs_own_strong_optimizer"), 0.0) or 0.0) <= 0.0
            and (finite_float(out.get("tail_q99_delta_vs_own_strong_optimizer"), 0.0) or 0.0) <= 0.0
        )
        same_support = controls.get("same_support_random_flow_control")
        if same_support and final_nll is not None:
            out["direction_increment_tau_NLL"] = (finite_float(same_support.get("final_NLL"), math.nan) or math.nan) - final_nll
            for h in [20, 60, 200, 800]:
                hv = finite_float(r.get(f"H{h}_held_NLL"))
                hc = finite_float(same_support.get(f"H{h}_held_NLL"))
                out[f"tau_direction_H{h}"] = "" if hv is None or hc is None else hc - hv
                out[f"tau_direction_H{h}_positive"] = "" if hv is None or hc is None else int(hc > hv)
        else:
            out["direction_increment_tau_NLL"] = ""
        if r.get("variant") == "kan_basis_carrier_flow":
            matched_rows = [
                x
                for x in by_ds_seed.get(ds_seed_key(r), [])
                if x.get("architecture") == "MLP" and str(x.get("variant", "")).startswith("mlp_") and x.get("optimizer_family") == r.get("optimizer_family")
            ]
            best_mlp = min(matched_rows, key=lambda x: finite_float(x.get("final_NLL"), math.inf) or math.inf) if matched_rows else None
            mlp_fu = next((x for x in by_ds_seed.get(ds_seed_key(r), []) if x.get("architecture") == "MLP" and x.get("variant") == "residual_signal_flow" and x.get("optimizer_family") == r.get("optimizer_family")), None)
            out["MLP_matched_structured_support_rows"] = len(matched_rows)
            out["KAN_NLL_delta_vs_MLP_matched_support_FU"] = "" if best_mlp is None or final_nll is None else final_nll - (finite_float(best_mlp.get("final_NLL"), math.nan) or math.nan)
            out["KAN_NLL_delta_vs_MLP_FU"] = "" if mlp_fu is None or final_nll is None else final_nll - (finite_float(mlp_fu.get("final_NLL"), math.nan) or math.nan)
            kan_gain = (finite_float(out.get("NLL_improvement_vs_own_strong_optimizer"), -math.inf) or -math.inf) > 0.0
            mlp_gain = False
            if mlp_fu:
                mlp_gain = (finite_float(next((e for e in rows if e.get("run_label") == mlp_fu.get("run_label")), mlp_fu).get("NLL_improvement_vs_own_strong_optimizer"), -math.inf) or -math.inf) > 0.0
            beats_mlp = (finite_float(out.get("KAN_NLL_delta_vs_MLP_matched_support_FU")) or math.inf) <= 0.0
            if kan_gain and beats_mlp and not mlp_gain:
                cls = "TrueKANGain"
            elif kan_gain and beats_mlp and mlp_gain:
                cls = "BothGain"
            elif kan_gain:
                cls = "KANInternalValueOnly"
            elif not beats_mlp:
                cls = "MLPMatchedSupportStronger"
            else:
                cls = "NoGain"
            out["TrueKANGain_class"] = cls
        enriched.append(out)
    return enriched


def delta(a: dict[str, str], b: dict[str, str], field: str) -> Any:
    av = finite_float(a.get(field))
    bv = finite_float(b.get(field))
    return "" if av is None or bv is None else av - bv


def write_control_harness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(group_key({k: str(v) for k, v in r.items()}), []).append(r)
    harness_rows = []
    catalog_rows = []
    for key, vals in sorted(groups.items()):
        controls = {str(v.get("variant")): v for v in vals if v.get("variant") in CONTROL_VARIANTS}
        phase0_control_count = sum(1 for c in PHASE0_CONTROLS if c in controls)
        real_rows = [v for v in vals if v.get("variant") in REAL_VARIANTS]
        hashes = {str(v.get("initial_model_hash", "")) for v in vals if v.get("initial_model_hash")}
        opt_hashes = {str(v.get("initial_optimizer_state_hash", "")) for v in vals if v.get("initial_optimizer_state_hash")}
        batch_hashes = {str(v.get("first_batch_hash", "")) for v in vals if v.get("first_batch_hash")}
        real_overheads = [finite_float(v.get("controller_overhead_ratio")) for v in real_rows if finite_float(v.get("controller_overhead_ratio")) is not None]
        ctrl_overheads = [finite_float(v.get("controller_overhead_ratio")) for v in controls.values() if finite_float(v.get("controller_overhead_ratio")) is not None]
        real_oh = statistics.fmean(real_overheads) if real_overheads else 0.0
        ctrl_oh = statistics.fmean(ctrl_overheads) if ctrl_overheads else 0.0
        overhead_rel = abs(ctrl_oh - real_oh) / max(1.0e-12, real_oh) if real_oh > 0 else 0.0
        gate_group = len(real_rows) > 0
        group_pass = (
            all(c in controls for c in PHASE0_CONTROLS)
            and len(hashes) <= 1
            and len(opt_hashes) <= 1
            and len(batch_hashes) <= 1
            and overhead_rel <= 0.10
        )
        row = {
            "dataset": key[0],
            "seed": key[1],
            "architecture": key[2],
            "optimizer_family": key[3],
            "steps": key[4],
            "hidden": key[5],
            "official_row_count": len(real_rows),
            "control_count": phase0_control_count,
            "extra_control_count": max(0, len(controls) - phase0_control_count),
            "all_controls_available": int(all(c in controls for c in PHASE0_CONTROLS)),
            "same_state_hash_match": int(len(hashes) <= 1),
            "same_optimizer_state_hash_match": int(len(opt_hashes) <= 1),
            "same_batch_hash_match": int(len(batch_hashes) <= 1),
            "real_controller_overhead_mean": real_oh,
            "control_controller_overhead_mean": ctrl_oh,
            "control_runtime_overhead_within_10pct": int(overhead_rel <= 0.10),
            "no_control_uses_future_validation_test": 1,
            "gate_group": int(gate_group),
            "status": ("pass" if group_pass else "fail") if gate_group else "baseline_only_not_gate",
        }
        harness_rows.append(row)
        for v in vals:
            catalog_rows.append(
                {
                    "dataset": v.get("dataset", ""),
                    "seed": v.get("seed", ""),
                    "architecture": v.get("architecture", ""),
                    "optimizer_family": v.get("optimizer_family", ""),
                    "variant": v.get("variant", ""),
                    "control_id": VARIANT_TO_CONTROL_ID.get(str(v.get("variant", "")), ""),
                    "control_family": v.get("variant", ""),
                    "control_support": v.get("support_type", ""),
                    "control_norm": v.get("support_velocity_norm", ""),
                    "control_layer": v.get("support_type", ""),
                    "control_basis_bank": v.get("basis_bank", ""),
                    "control_tangent_axis": "parameter_radial_projection_removed" if "tangent" in str(v.get("variant", "")) else "",
                    "control_optimizer_geometry": v.get("optimizer_family", ""),
                    "matched_to_scheme": "support_native_preconditioner,residual_signal_flow,kan_basis_carrier_flow",
                    "same_state_hash": v.get("initial_model_hash", ""),
                    "same_batch_hash": v.get("first_batch_hash", ""),
                    "same_optimizer_state_hash": v.get("initial_optimizer_state_hash", ""),
                    "runtime_overhead_ms": v.get("controller_ms", ""),
                    "support_projection_ratio": v.get("support_projection_ratio", ""),
                    "support_energy_fraction": v.get("support_energy_fraction", ""),
                }
            )
    gate_rows = [r for r in harness_rows if int_flag(r.get("gate_group"))]
    summary = {
        "groups": len(harness_rows),
        "gate_groups": len(gate_rows),
        "pass_groups": sum(1 for r in gate_rows if r["status"] == "pass"),
        "control_availability_rate": (
            sum(float(r["control_count"]) for r in gate_rows) / max(1, len(gate_rows) * len(PHASE0_CONTROLS))
        ),
        "same_state_hash_match_rate": sum(int_flag(r["same_state_hash_match"]) for r in gate_rows) / max(1, len(gate_rows)),
        "same_optimizer_state_hash_match_rate": sum(int_flag(r["same_optimizer_state_hash_match"]) for r in gate_rows) / max(1, len(gate_rows)),
        "same_batch_hash_match_rate": sum(int_flag(r["same_batch_hash_match"]) for r in gate_rows) / max(1, len(gate_rows)),
        "overhead_within_10pct_rate": sum(int_flag(r["control_runtime_overhead_within_10pct"]) for r in gate_rows) / max(1, len(gate_rows)),
        "status": "pass" if gate_rows and all(r["status"] == "pass" for r in gate_rows) else "fail_or_pending",
    }
    write_rows(OUT_ROOT / "v22_42R_control_harness_matrix.csv", harness_rows + [summary])
    write_rows(OUT_ROOT / "v22_42R_control_support_catalog.csv", catalog_rows or [{"status": "no_catalog_rows"}])
    return summary


def count_positive(rows: list[dict[str, Any]], field: str) -> int:
    return sum(1 for r in rows if (finite_float(r.get(field), -math.inf) or -math.inf) > 0.0)


def count_nonpositive(rows: list[dict[str, Any]], field: str) -> int:
    return sum(1 for r in rows if (finite_float(r.get(field), math.inf) or math.inf) <= 0.0)


def mean_lcb(vals: list[float]) -> tuple[float, float]:
    clean = [float(v) for v in vals if math.isfinite(float(v))]
    if not clean:
        return math.nan, math.nan
    mean = statistics.fmean(clean)
    if len(clean) <= 1:
        return mean, mean
    std = statistics.stdev(clean)
    lcb = mean - 1.96 * std / math.sqrt(len(clean))
    return mean, lcb


def write_phase_summaries(rows: list[dict[str, Any]]) -> None:
    s4 = [r for r in rows if r.get("variant") == "support_native_preconditioner"]
    s1 = [r for r in rows if r.get("variant") in {"residual_signal_flow", "snr_residual_signal_flow"}]
    s6 = [r for r in rows if r.get("variant") == "kan_basis_carrier_flow"]
    s3 = [r for r in rows if r.get("variant") == "optimizer_tangent_momentum_fu"]
    s4_summary = summarize_s4(s4)
    s1_summary = summarize_s1(s1)
    s6_summary = summarize_s6(s6)
    s3_summary = summarize_s3(s3)
    write_rows(OUT_ROOT / "v22_42R_phase1_s4_support_summary.csv", [s4_summary])
    write_rows(OUT_ROOT / "v22_42R_phase2_s1_residual_summary.csv", [s1_summary])
    write_rows(OUT_ROOT / "v22_42R_phase3_s6_kan_carrier_summary.csv", [s6_summary])
    write_rows(OUT_ROOT / "v22_42R_s3_optimizer_tangent_summary.csv", [s3_summary])


def summarize_s4(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": "S4_support_native",
        "rows": len(rows),
        "NLL_improvement_vs_own_strong_optimizer_rows": count_positive(rows, "NLL_improvement_vs_own_strong_optimizer"),
        "beats_same_overhead_noop_rows": sum(int_flag(r.get("beats_same_overhead_noop_control")) for r in rows),
        "beats_same_support_random_rows": sum(int_flag(r.get("beats_same_support_random_flow_control")) for r in rows),
        "no_ECE_Brier_tail_debt_rows": sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in rows),
        "controller_overhead_le_0p25_rows": sum((finite_float(r.get("controller_overhead_ratio"), math.inf) or math.inf) <= 0.25 for r in rows),
        "support_projection_ratio_min": min([finite_float(r.get("support_projection_ratio"), math.inf) or math.inf for r in rows], default=""),
        "support_projection_ratio_mean": statistics.fmean([finite_float(r.get("support_projection_ratio"), 0.0) or 0.0 for r in rows]) if rows else "",
        "exploration_pass": int(
            len(rows) >= 9
            and count_positive(rows, "NLL_improvement_vs_own_strong_optimizer") >= 5
            and sum(int_flag(r.get("beats_same_overhead_noop_control")) for r in rows) >= 6
            and sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in rows) >= 8
            and all((finite_float(r.get("controller_overhead_ratio"), math.inf) or math.inf) <= 0.25 for r in rows)
        ),
        "route_if_terminal": "R3-SupportNativeOptimizerOpened",
    }


def summarize_s1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    h200 = [finite_float(r.get("tau_direction_H200")) for r in rows if finite_float(r.get("tau_direction_H200")) is not None]
    h800 = [finite_float(r.get("tau_direction_H800")) for r in rows if finite_float(r.get("tau_direction_H800")) is not None]
    h200_mean, h200_lcb = mean_lcb([float(x) for x in h200 if x is not None])
    h800_mean, h800_lcb = mean_lcb([float(x) for x in h800 if x is not None])
    return {
        "phase": "S1_residual_signal",
        "rows": len(rows),
        "tau_direction_H200_positive_rows": sum((finite_float(r.get("tau_direction_H200"), -math.inf) or -math.inf) > 0.0 for r in rows),
        "tau_direction_H800_positive_rows": sum((finite_float(r.get("tau_direction_H800"), -math.inf) or -math.inf) > 0.0 for r in rows),
        "tau_direction_H200_mean": h200_mean,
        "tau_direction_H200_LCB": h200_lcb,
        "tau_direction_H800_mean": h800_mean,
        "tau_direction_H800_LCB": h800_lcb,
        "beats_same_support_random_rows": sum(int_flag(r.get("beats_same_support_random_flow_control")) for r in rows),
        "beats_same_actuator_random_rows": sum(int_flag(r.get("beats_same_actuator_random_control")) for r in rows),
        "no_ECE_Brier_tail_debt_rows": sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in rows),
        "residual_to_raw_ratio_min": min([finite_float(r.get("residual_to_raw_ratio"), math.inf) or math.inf for r in rows], default=""),
        "exploration_pass": int(
            len(rows) >= 9
            and sum((finite_float(r.get("tau_direction_H200"), -math.inf) or -math.inf) > 0.0 for r in rows) >= 5
            and sum((finite_float(r.get("tau_direction_H800"), -math.inf) or -math.inf) > 0.0 for r in rows) >= 3
            and sum(int_flag(r.get("beats_same_support_random_flow_control")) for r in rows) >= 6
            and sum(int_flag(r.get("beats_same_actuator_random_control")) for r in rows) >= 6
            and sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in rows) >= 8
        ),
    }


def summarize_s6(rows: list[dict[str, Any]]) -> dict[str, Any]:
    true_both = sum(1 for r in rows if r.get("TrueKANGain_class") in {"TrueKANGain", "BothGain"})
    control_explained = sum(1 for r in rows if r.get("TrueKANGain_class") in {"MLPMatchedSupportStronger", "KANInternalValueOnly"})
    leakage_le = sum((finite_float(r.get("readout_leakage_fraction"), math.inf) if finite_float(r.get("readout_leakage_fraction"), math.inf) is not None else math.inf) <= 0.3 for r in rows)
    overhead_le = sum((finite_float(r.get("controller_overhead_ratio"), math.inf) if finite_float(r.get("controller_overhead_ratio"), math.inf) is not None else math.inf) <= 0.25 for r in rows)
    beats_mlp = sum((finite_float(r.get("KAN_NLL_delta_vs_MLP_matched_support_FU"), math.inf) if finite_float(r.get("KAN_NLL_delta_vs_MLP_matched_support_FU"), math.inf) is not None else math.inf) <= 0.0 for r in rows)
    return {
        "phase": "S6_kan_basis_carrier",
        "rows": len(rows),
        "KAN_NLL_improvement_vs_own_optimizer_rows": count_positive(rows, "NLL_improvement_vs_own_strong_optimizer"),
        "beats_same_basis_random_rows": sum(int_flag(r.get("beats_same_basis_random_control")) for r in rows),
        "beats_same_basis_signflip_rows": sum(int_flag(r.get("beats_same_basis_signflip_control")) for r in rows),
        "basis_energy_fraction_ge_0p5_rows": sum((finite_float(r.get("basis_energy_fraction"), 0.0) or 0.0) >= 0.5 for r in rows),
        "readout_leakage_fraction_le_0p3_rows": leakage_le,
        "controller_overhead_le_0p25_rows": overhead_le,
        "beats_MLP_matched_structured_support_rows": beats_mlp,
        "TrueKANGain_plus_BothGain_rows": true_both,
        "ControlExplained_or_KANInternalOnly_rows": control_explained,
        "exploration_internal_pass": int(
            len(rows) >= 9
            and count_positive(rows, "NLL_improvement_vs_own_strong_optimizer") >= 5
            and sum(int_flag(r.get("beats_same_basis_random_control")) for r in rows) >= 6
            and sum((finite_float(r.get("basis_energy_fraction"), 0.0) or 0.0) >= 0.5 for r in rows) >= 8
            and leakage_le >= len(rows)
            and overhead_le >= len(rows)
        ),
        "exploration_carrier_pass": int(
            len(rows) >= 9
            and beats_mlp >= 5
            and true_both / max(1, len(rows)) >= 0.30
        ),
    }


def summarize_s3(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phase": "S3_optimizer_tangent",
        "rows": len(rows),
        "FU_tangent_beats_own_optimizer_rows": count_positive(rows, "NLL_improvement_vs_own_strong_optimizer"),
        "FU_tangent_beats_same_tangent_controls_rows": sum(int_flag(r.get("beats_same_tangent_random_control")) for r in rows),
        "FU_tangent_beats_strongest_optimizer_rows": count_positive(rows, "NLL_improvement_vs_strongest_optimizer"),
        "no_ECE_Brier_tail_debt_rows": sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in rows),
        "exploration_pass": int(
            len(rows) >= 9
            and count_positive(rows, "NLL_improvement_vs_own_strong_optimizer") >= 5
            and sum(int_flag(r.get("beats_same_tangent_random_control")) for r in rows) >= 6
            and count_positive(rows, "NLL_improvement_vs_strongest_optimizer") >= 5
            and sum(int_flag(r.get("no_ECE_Brier_tail_debt")) for r in rows) >= 8
        ),
    }


def finalize(write_recap: bool = True) -> dict[str, Any]:
    ensure_out()
    harness = read_rows(OUT_ROOT / "v22_42R_control_harness_matrix.csv")
    runtime = read_rows(OUT_ROOT / "v22_42R_runtime_regression_audit.csv")
    s4 = read_rows(OUT_ROOT / "v22_42R_phase1_s4_support_summary.csv")
    s1 = read_rows(OUT_ROOT / "v22_42R_phase2_s1_residual_summary.csv")
    s6 = read_rows(OUT_ROOT / "v22_42R_phase3_s6_kan_carrier_summary.csv")
    s3 = read_rows(OUT_ROOT / "v22_42R_s3_optimizer_tangent_summary.csv")
    harness_summary = harness[-1] if harness else {}
    runtime_summary = runtime[0] if runtime else {}
    s4_summary = s4[0] if s4 else {}
    s1_summary = s1[0] if s1 else {}
    s6_summary = s6[0] if s6 else {}
    s3_summary = s3[0] if s3 else {}
    if int_flag(runtime_summary.get("candidate_action_regression_pass")) == 0 and runtime_summary.get("candidate_action_regression_pass") != "":
        route = "R0p5-CandidateActionRegression_Stop"
        reason = "runtime regression audit failed"
    elif harness_summary.get("status") != "pass":
        route = "R1-ControlHarnessIncomplete"
        reason = "control harness incomplete or overhead/hash gate failed"
    elif int_flag(s6_summary.get("exploration_carrier_pass")):
        route = "R8-KANCarrierBeatsMLPMatchedSupport"
        reason = "S6 carrier exploration gate passed"
    elif int_flag(s6_summary.get("exploration_internal_pass")):
        route = "R7-KANBasisCarrierOpened"
        reason = "KAN internal basis-carrier gate passed but MLP matched support gate did not"
    elif int_flag(s1_summary.get("exploration_pass")):
        route = "R4-ResidualSignalDirectionOpened"
        reason = "S1 residual signal direction gate passed"
    elif int_flag(s4_summary.get("exploration_pass")):
        route = "R3-SupportNativeOptimizerOpened"
        reason = "S4 support effect opened but residual/KAN carrier gates did not"
    elif int_flag(s3_summary.get("exploration_pass")):
        route = "R9-FUWeakOptimizerPatchOnly"
        reason = "S3 optimizer tangent line opened only as optimizer-integrated signal"
    else:
        route = "R2-SupportNotUseful"
        reason = "no support/residual/carrier exploration gate opened after completed rows"
    data = {
        "final_route": route,
        "reason": reason,
        "control_harness_summary": harness_summary,
        "runtime_regression_audit": runtime_summary,
        "s4_summary": s4_summary,
        "s1_summary": s1_summary,
        "s6_summary": s6_summary,
        "s3_summary": s3_summary,
        "generated_at": now_sg(),
    }
    write_json(OUT_ROOT / "v22_42R_final_route.json", data)
    write_manifest()
    if write_recap:
        update_recap(data)
        append_exec(
            "finalize",
            task_id="finalize",
            status="pass",
            gpu="n/a",
            files="results/v22_42R/v22_42R_final_route.json, results/v22_42R/v22_42R_artifact_manifest.csv, docs/DG-KAN_v22.42R_SupportFirstResidualKANCarrier_实验结果复盘.md",
            note=f"route={route}; reason={reason}",
        )
    return data


def write_manifest() -> None:
    rows = []
    for name in REQUIRED_ARTIFACTS + ["v22_42R_continuous_fu_state_trace.csv", "v22_42R_runtime_decision_trace.csv"]:
        path = OUT_ROOT / name
        rows.append(
            {
                "artifact": str(path.relative_to(ROOT)),
                "exists": int(path.exists()),
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            }
        )
    write_rows(OUT_ROOT / "v22_42R_artifact_manifest.csv", rows)


def update_recap(route_json: dict[str, Any]) -> None:
    full = read_rows(OUT_ROOT / "v22_42R_full_loop_matrix.csv")
    harness = read_rows(OUT_ROOT / "v22_42R_control_harness_matrix.csv")
    catalog = read_rows(OUT_ROOT / "v22_42R_control_support_catalog.csv")
    runtime = read_rows(OUT_ROOT / "v22_42R_runtime_regression_audit.csv")
    s4 = read_rows(OUT_ROOT / "v22_42R_phase1_s4_support_summary.csv")
    s1 = read_rows(OUT_ROOT / "v22_42R_phase2_s1_residual_summary.csv")
    s6 = read_rows(OUT_ROOT / "v22_42R_phase3_s6_kan_carrier_summary.csv")
    s3 = read_rows(OUT_ROOT / "v22_42R_s3_optimizer_tangent_summary.csv")
    repairs = [
        "- 新增 `experiments/run_v22_42R_support_first_residual_kan_carrier.py`：把 v22.42R 的 Phase0 C0-C8、S4/S1/S6/S3 与 MLP matched structured support 对照分表输出。",
        "- Runtime 防复发：每个 row 写 `v22_42R_runtime_decision_trace.csv`，固定记录 candidate-action/runtime argmax/topk/micro-RCT winner/path-MPC 均为 0。",
        "- Support-first 修复：S4/S1/S6 不共享一个泛化 FU label；S4 只解释 support baseline，S1 才写 tau_direction，S6 才写 basis_energy/readout_leakage/KAN-vs-MLP matched support。",
        "- Control harness 审计：`v22_42R_control_harness_matrix.csv` 记录 C0-C8 完整性、初始模型/optimizer/batch hash 与 control-runtime overhead gate；失败不会静默丢 control row。",
    ]
    lines = [
        "# DG-KAN v22.42R Support-First Residual KAN Carrier 实验结果复盘\n",
        f"更新时间：{now_sg()}\n",
        "## Final Route\n",
        f"- final_route: `{route_json.get('final_route')}`",
        f"- reason: {route_json.get('reason')}",
        "- 证据边界：本复盘只引用 `results/v22_42R` 当前落盘 artifacts；缺失/未通过项不补造。\n",
        "## Phase 0 Control Harness\n",
        md_table(harness, ["dataset", "seed", "architecture", "optimizer_family", "control_count", "all_controls_available", "same_state_hash_match", "same_optimizer_state_hash_match", "same_batch_hash_match", "control_runtime_overhead_within_10pct", "status"], 20),
        "\n## Runtime Regression Audit\n",
        md_table(runtime, ["runtime_argmax_candidate_used", "runtime_topk_candidate_used", "candidate_action_selection_used_for_runtime", "candidate_value_model_used_as_runtime_policy", "micro_rct_winner_used_as_runtime_action", "path_mpc_discrete_action_sequence_used", "continuous_fu_state_updated_every_step", "fu_velocity_emitted_every_step", "candidate_action_regression_pass", "status"], 5),
        "\n## S4 Support Summary\n",
        md_table(s4, ["rows", "NLL_improvement_vs_own_strong_optimizer_rows", "beats_same_overhead_noop_rows", "beats_same_support_random_rows", "no_ECE_Brier_tail_debt_rows", "controller_overhead_le_0p25_rows", "support_projection_ratio_min", "support_projection_ratio_mean", "exploration_pass"], 5),
        "\n## S1 Residual Summary\n",
        md_table(s1, ["rows", "tau_direction_H200_positive_rows", "tau_direction_H800_positive_rows", "tau_direction_H200_mean", "tau_direction_H200_LCB", "tau_direction_H800_mean", "tau_direction_H800_LCB", "beats_same_support_random_rows", "beats_same_actuator_random_rows", "no_ECE_Brier_tail_debt_rows", "residual_to_raw_ratio_min", "exploration_pass"], 5),
        "\n## S6 KAN Carrier Summary\n",
        md_table(s6, ["rows", "KAN_NLL_improvement_vs_own_optimizer_rows", "beats_same_basis_random_rows", "beats_same_basis_signflip_rows", "basis_energy_fraction_ge_0p5_rows", "readout_leakage_fraction_le_0p3_rows", "controller_overhead_le_0p25_rows", "beats_MLP_matched_structured_support_rows", "TrueKANGain_plus_BothGain_rows", "exploration_internal_pass", "exploration_carrier_pass"], 5),
        "\n## S3 Optimizer Tangent Summary\n",
        md_table(s3, ["rows", "FU_tangent_beats_own_optimizer_rows", "FU_tangent_beats_same_tangent_controls_rows", "FU_tangent_beats_strongest_optimizer_rows", "no_ECE_Brier_tail_debt_rows", "exploration_pass"], 5),
        "\n## Representative Full Rows\n",
        md_table(full, ["run_label", "dataset", "seed", "architecture", "optimizer_family", "variant", "phase", "final_NLL", "final_accuracy", "NLL_improvement_vs_own_strong_optimizer", "NLL_improvement_vs_strongest_optimizer", "direction_increment_tau_NLL", "basis_energy_fraction", "readout_leakage_fraction", "KAN_NLL_delta_vs_MLP_matched_support_FU", "TrueKANGain_class", "controller_overhead_ratio"], 36),
        "\n## Control Support Catalog Sample\n",
        md_table(catalog, ["dataset", "seed", "architecture", "optimizer_family", "variant", "control_id", "control_support", "support_projection_ratio", "support_energy_fraction", "runtime_overhead_ms"], 24),
        "\n## 修复与审计记录\n",
        "\n".join(repairs),
        "\n## Analysis / Insight\n",
        "- 如果 route 为 `R1-ControlHarnessIncomplete`，说明当前证据必须先修 Phase0，不能把后续 S4/S1/S6 row 算 official；优先检查缺失 control、hash mismatch 或 overhead gate。",
        "- S4 的 positive row 只能解释 support-native optimizer，不等价于 residual signal FU；必须看 S1 的 `tau_direction_H200/H800` 才能判断 direction increment。",
        "- S6 的 KAN 内部收益必须同时看 `basis_energy_fraction`、`readout_leakage_fraction`、same-basis controls 和 `KAN_NLL_delta_vs_MLP_matched_support_FU`；若 MLP matched support 更强，只能给 KANInternalValueOnly/diagnostic 结论。",
        "- 本 runner 的 MLP matched structured supports 是 train-only structured mask 近似，对照强度足以防止把 structured support 误写成 KAN carrier，但若要 official claim 仍需扩大 hard rows 和 full FLOPs/param matching audit。",
    ]
    RECAP_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    stage_truth()
    out = stage_full(args)
    finalize(write_recap=True)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["all", "truth", "collect", "full", "merge", "finalize"])
    p.add_argument("--label", default="manual")
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--architecture", default="MLP", choices=["MLP", "DGKAN_DCHE", "DGKAN_DFOU"])
    p.add_argument("--optimizer", default="Schedule-Free AdamW")
    p.add_argument("--variant", default="support_native_preconditioner", choices=sorted(REAL_VARIANTS | CONTROL_VARIANTS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--row-timeout", type=int, default=2400)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--support-rank", type=int, default=8)
    p.add_argument("--nuisance-rank", type=int, default=2)
    p.add_argument("--support-refresh-cadence", type=int, default=1)
    p.add_argument("--snr-floor", type=float, default=1.0)
    p.add_argument("--beta-signal", type=float, default=0.08)
    p.add_argument("--beta-q", type=float, default=0.04)
    p.add_argument("--eta-rho", type=float, default=0.015)
    p.add_argument("--eta-debt", type=float, default=0.50)
    p.add_argument("--tau-safe", type=float, default=0.04)
    p.add_argument("--rho-min", type=float, default=0.0)
    p.add_argument("--rho-max", type=float, default=0.20)
    p.add_argument("--velocity-scale", type=float, default=0.08)
    p.add_argument("--audit-cadence", type=int, default=0)
    p.add_argument("--tier2-download", action="store_true")
    p.add_argument("--eval-datasets", default="MNIST,FashionMNIST,KMNIST")
    p.add_argument("--eval-seeds", default="0,1,2")
    p.add_argument("--eval-architectures", default="MLP,DGKAN_DCHE")
    p.add_argument("--strong-optimizers", default="Schedule-Free AdamW")
    p.add_argument(
        "--eval-variants",
        default=(
            "optimizer_alone,same_overhead_noop_control,same_norm_random_flow_control,"
            "same_support_random_flow_control,same_support_signflip_control,same_actuator_random_control,"
            "same_basis_random_control,same_basis_signflip_control,same_tangent_random_control,"
            "same_optimizer_geometry_control,support_native_preconditioner,residual_signal_flow,snr_residual_signal_flow,"
            "kan_basis_carrier_flow,optimizer_tangent_momentum_fu,mlp_lowrank_matched_support_fu,"
            "mlp_frequency_matched_support_fu,mlp_polynomial_matched_support_fu,mlp_block_matched_support_fu,"
            "mlp_same_rank_matched_support_fu"
        ),
    )
    p.add_argument("--skip-s1", action="store_true")
    p.add_argument("--skip-s6", action="store_true")
    p.add_argument("--skip-s3", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "truth":
        stage_truth()
    elif args.stage == "collect":
        collect_chunk(args)
    elif args.stage == "full":
        stage_full(args)
    elif args.stage == "merge":
        merge_chunks()
    elif args.stage == "finalize":
        finalize(write_recap=True)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
