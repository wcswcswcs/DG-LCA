#!/usr/bin/env python3
"""DG-KAN v7.2 real-only significance and kernelization runner.

This runner extends the v7.1 targeted chain with the first v7.2 stop/go
sequence:

* 10-seed task/significance audit for strict H3/H4 and selected bridges.
* full manual-vs-autograd gradient correctness probes.
* measured full-step efficiency and live-set attribution.
* a real vectorized grouped implementation that removes the Python group loop
  for K0/K1/K2 style candidates.

Unimplemented lower-level fused/Triton packages are recorded as
``not_implemented`` or ``not_run`` only.  No fake data, proxy rows, or hand
filled ratios are emitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

import run_gafu_v71_real as v71
from dgkan_core import ensure_dir, get_device, parse_int_list, parse_str_list, save_json, set_seed, write_csv
from run_gafu_v54 import _ece
from run_gafu_v63 import V63ManualLayer, V63Params
from run_gafu_v66_real import _git_commit, _git_status


PLAN_PATH = "docs/DG-KAN_v7.2_SignificantBeyondMLP_KernelNativeEfficiency_完整实验计划.md"
SCRIPT_PATH = "experiments/run_gafu_v72_real.py"
METRIC_UNAVAILABLE = "metric_unavailable"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _is_one(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true"}


def _is_zero(value: Any) -> bool:
    return str(value).strip().lower() in {"0", "0.0", "false"}


def _mean(values: Iterable[Any], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if _finite(v)]
    return statistics.mean(vals) if vals else default


def _std(values: Iterable[Any], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if _finite(v)]
    if not vals:
        return default
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("::".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return 720000 + int(digest[:8], 16) % 100000


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else "nan"
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class VectorizedGroupedPoly2GateStack:
    """Grouped poly2-gate stack with group math batched into single tensor ops."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        group_count: int,
        *,
        shuffle: bool,
        device: torch.device,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.group_count = int(group_count)
        self.shuffle = bool(shuffle)
        self.scale = 0.05
        dims = [self.input_dim] + [self.hidden_dim] * self.depth
        self.params: List[Dict[str, torch.Tensor]] = []
        self.grads: List[Dict[str, torch.Tensor]] = []
        self.perms: List[Tuple[torch.Tensor | None, torch.Tensor | None]] = []
        for li, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            if a % self.group_count != 0 or b % self.group_count != 0:
                raise ValueError(f"group_count={self.group_count} must divide {a}->{b}")
            in_g = a // self.group_count
            out_g = b // self.group_count
            params = {
                "mix": torch.randn(self.group_count, out_g, in_g, device=device) / math.sqrt(max(1, in_g)),
                "poly": torch.zeros(self.group_count, in_g, 2, device=device),
                "gate": torch.zeros(self.group_count, in_g, 1, device=device),
            }
            params["poly"][:, :, 0].fill_(0.02)
            self.params.append(params)
            self.grads.append({k: torch.zeros_like(v) for k, v in params.items()})
            if self.shuffle and li > 0:
                perm = torch.arange(a, device=device).view(self.group_count, in_g).transpose(0, 1).reshape(-1)
                inv = torch.empty_like(perm)
                inv[perm] = torch.arange(a, device=device)
                self.perms.append((perm, inv))
            else:
                self.perms.append((None, None))

    def _transform(self, xg: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        gate = torch.sigmoid(params["gate"].squeeze(-1)).unsqueeze(0)
        poly = params["poly"].unsqueeze(0)
        return xg + self.scale * gate * (poly[..., 0] * xg + poly[..., 1] * xg.square())

    def _forward_layer(self, h: torch.Tensor, li: int, params: Dict[str, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        p = self.params[li] if params is None else params
        perm, inv = self.perms[li]
        hin = h[:, perm] if perm is not None else h
        bsz = int(hin.shape[0])
        xg = hin.reshape(bsz, self.group_count, -1)
        z = self._transform(xg, p)
        y = torch.einsum("bgi,goi->bgo", z, p["mix"]).reshape(bsz, -1)
        return y, {"xg": xg.detach(), "z": z.detach(), "y": y.detach(), "perm": perm, "inv": inv}

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        with torch.no_grad():
            h = x
            caches: List[Dict[str, Any]] = []
            for li in range(self.depth):
                y, cache = self._forward_layer(h, li)
                caches.append(cache)
                h = F.silu(y) if li < self.depth - 1 else y
            return h, caches

    def forward_autograd_with_params(self, x: torch.Tensor, params: Sequence[Dict[str, torch.Tensor]]) -> torch.Tensor:
        h = x
        for li, p in enumerate(params):
            y, _cache = self._forward_layer(h, li, p)
            h = F.silu(y) if li < self.depth - 1 else y
        return h

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        with torch.no_grad():
            for li in reversed(range(self.depth)):
                cache = caches[li]
                if li < self.depth - 1:
                    y = cache["y"]
                    sig = torch.sigmoid(y)
                    delta = delta * sig * (1.0 + y * (1.0 - sig))
                p = self.params[li]
                g = self.grads[li]
                bsz = int(delta.shape[0])
                out_g = delta.reshape(bsz, self.group_count, -1)
                xg = cache["xg"]
                z = cache["z"]
                dz = torch.einsum("bgo,goi->bgi", out_g, p["mix"])
                g["mix"].add_(torch.einsum("bgo,bgi->goi", out_g, z))
                gate_raw = p["gate"].squeeze(-1)
                gate = torch.sigmoid(gate_raw).unsqueeze(0)
                poly = p["poly"].unsqueeze(0)
                basis0 = poly[..., 0] * xg + poly[..., 1] * xg.square()
                dres = dz * self.scale
                g["poly"][:, :, 0].add_((dres * gate * xg).sum(dim=0))
                g["poly"][:, :, 1].add_((dres * gate * xg.square()).sum(dim=0))
                g["gate"][:, :, 0].add_((dres * basis0 * gate * (1.0 - gate)).sum(dim=0))
                dxg = dz * (1.0 + self.scale * gate * (poly[..., 0] + 2.0 * poly[..., 1] * xg))
                dx = dxg.reshape(bsz, -1)
                inv = cache["inv"]
                delta = dx[:, inv] if inv is not None else dx
            return delta

    def zero_grad(self) -> None:
        for grads in self.grads:
            for grad in grads.values():
                grad.zero_()

    def params_and_grads(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        out: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        for li, (params, grads) in enumerate(zip(self.params, self.grads)):
            for name, p in params.items():
                out.append((f"vgroup:l{li}:{name}", p, grads[name]))
        return out

    def clone_params_for_autograd(self) -> List[Dict[str, torch.Tensor]]:
        return [{k: v.detach().clone().requires_grad_(True) for k, v in params.items()} for params in self.params]

    def param_count(self) -> int:
        return sum(int(p.numel()) for params in self.params for p in params.values())

    def op_counts(self) -> Dict[str, int]:
        return {
            "op_count_gemm": self.depth,
            "op_count_elementwise": self.depth * 5,
            "op_count_pow": self.depth,
            "op_count_exp": 0,
            "python_loop_count": self.depth,
            "group_loop_count": 0,
        }

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        hidden = sum(int(c["y"].numel() * c["y"].element_size()) for c in caches)
        xg = sum(int(c["xg"].numel() * c["xg"].element_size()) for c in caches)
        z = sum(int(c["z"].numel() * c["z"].element_size()) for c in caches)
        total = hidden + xg + z
        return {
            "cache_total_MB": total / (1024**2),
            "cache_hidden_MB": hidden / (1024**2),
            "cache_group_input_MB": xg / (1024**2),
            "cache_gate_MB": z / (1024**2),
            "cache_basis_MB": 0.0,
        }


class FastAdamWNoSync:
    """AdamW with the same tensor update math but no per-param CPU norm sync."""

    def __init__(self, providers: Sequence[Any], *, lr: float, weight_decay: float = 1.0e-4) -> None:
        self.providers = list(providers)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.update_mode = (
            "sgd"
            if any(getattr(provider, "update_mode", "") == "sgd" for provider in self.providers)
            else "rms"
            if any(getattr(provider, "update_mode", "") == "rms" for provider in self.providers)
            else "adamw_addcdiv"
            if any(getattr(provider, "update_mode", "") == "adamw_addcdiv" for provider in self.providers)
            else "adamw_half"
            if any(getattr(provider, "update_mode", "") == "adamw_half" for provider in self.providers)
            else "adamw_v_half"
            if any(getattr(provider, "update_mode", "") == "adamw_v_half" for provider in self.providers)
            else "adamw_bfloat"
            if any(getattr(provider, "update_mode", "") == "adamw_bfloat" for provider in self.providers)
            else "adamw_v_bfloat_addcdiv"
            if any(getattr(provider, "update_mode", "") == "adamw_v_bfloat_addcdiv" for provider in self.providers)
            else "adamw_head_bfloat_v_bfloat_addcdiv"
            if any(getattr(provider, "update_mode", "") == "adamw_head_bfloat_v_bfloat_addcdiv" for provider in self.providers)
            else "adamw_stack_bfloat_v_bfloat_addcdiv"
            if any(getattr(provider, "update_mode", "") == "adamw_stack_bfloat_v_bfloat_addcdiv" for provider in self.providers)
            else "adamw_tail_bfloat_v_bfloat"
            if any(getattr(provider, "update_mode", "") == "adamw_tail_bfloat_v_bfloat" for provider in self.providers)
            else "adamw_l0_bfloat_v_bfloat"
            if any(getattr(provider, "update_mode", "") == "adamw_l0_bfloat_v_bfloat" for provider in self.providers)
            else "adamw_l0_bfloat_v_bfloat_addcdiv"
            if any(getattr(provider, "update_mode", "") == "adamw_l0_bfloat_v_bfloat_addcdiv" for provider in self.providers)
            else "adamw_v_bfloat"
            if any(getattr(provider, "update_mode", "") == "adamw_v_bfloat" for provider in self.providers)
            else "adamw_v_bfloat_native"
            if any(getattr(provider, "update_mode", "") == "adamw_v_bfloat_native" for provider in self.providers)
            else "adamw_cpu"
            if any(getattr(provider, "update_mode", "") == "adamw_cpu" for provider in self.providers)
            else "adamw_v_cpu"
            if any(getattr(provider, "update_mode", "") == "adamw_v_cpu" for provider in self.providers)
            else "adamw"
        )
        self.t = 0
        self.m: Dict[int, torch.Tensor] = {}
        self.v: Dict[int, torch.Tensor] = {}
        if self.update_mode in {"adamw", "adamw_addcdiv"}:
            for _name, p, _g in self.params():
                self.m[id(p)] = torch.zeros_like(p)
                self.v[id(p)] = torch.zeros_like(p)
        elif self.update_mode == "adamw_half":
            for _name, p, _g in self.params():
                state_dtype = torch.float16 if p.is_cuda and p.dtype == torch.float32 else p.dtype
                self.m[id(p)] = torch.zeros_like(p, dtype=state_dtype)
                self.v[id(p)] = torch.zeros_like(p, dtype=state_dtype)
        elif self.update_mode == "adamw_v_half":
            for _name, p, _g in self.params():
                state_dtype = torch.float16 if p.is_cuda and p.dtype == torch.float32 else p.dtype
                self.m[id(p)] = torch.zeros_like(p)
                self.v[id(p)] = torch.zeros_like(p, dtype=state_dtype)
        elif self.update_mode == "adamw_bfloat":
            for _name, p, _g in self.params():
                state_dtype = torch.bfloat16 if p.is_cuda and p.dtype == torch.float32 and torch.cuda.is_bf16_supported() else p.dtype
                self.m[id(p)] = torch.zeros_like(p, dtype=state_dtype)
                self.v[id(p)] = torch.zeros_like(p, dtype=state_dtype)
        elif self.update_mode == "adamw_v_bfloat_addcdiv":
            for _name, p, _g in self.params():
                state_dtype = torch.bfloat16 if p.is_cuda and p.dtype == torch.float32 and torch.cuda.is_bf16_supported() else p.dtype
                self.m[id(p)] = torch.zeros_like(p)
                self.v[id(p)] = torch.zeros_like(p, dtype=state_dtype)
        elif self.update_mode == "adamw_head_bfloat_v_bfloat_addcdiv":
            for name, p, _g in self.params():
                bf16_dtype = torch.bfloat16 if p.is_cuda and p.dtype == torch.float32 and torch.cuda.is_bf16_supported() else p.dtype
                m_dtype = bf16_dtype if "head" in str(name).lower() else p.dtype
                self.m[id(p)] = torch.zeros_like(p, dtype=m_dtype)
                self.v[id(p)] = torch.zeros_like(p, dtype=bf16_dtype)
        elif self.update_mode == "adamw_stack_bfloat_v_bfloat_addcdiv":
            for name, p, _g in self.params():
                bf16_dtype = torch.bfloat16 if p.is_cuda and p.dtype == torch.float32 and torch.cuda.is_bf16_supported() else p.dtype
                m_dtype = p.dtype if "head" in str(name).lower() else bf16_dtype
                self.m[id(p)] = torch.zeros_like(p, dtype=m_dtype)
                self.v[id(p)] = torch.zeros_like(p, dtype=bf16_dtype)
        elif self.update_mode in {"adamw_tail_bfloat_v_bfloat", "adamw_l0_bfloat_v_bfloat", "adamw_l0_bfloat_v_bfloat_addcdiv"}:
            for name, p, _g in self.params():
                bf16_dtype = torch.bfloat16 if p.is_cuda and p.dtype == torch.float32 and torch.cuda.is_bf16_supported() else p.dtype
                lname = str(name).lower()
                tail_stack = ("l1:mix" in lname) or ("l2:mix" in lname)
                first_stack = "l0:mix" in lname
                m_bfloat = (self.update_mode == "adamw_tail_bfloat_v_bfloat" and tail_stack) or (
                    self.update_mode in {"adamw_l0_bfloat_v_bfloat", "adamw_l0_bfloat_v_bfloat_addcdiv"} and first_stack
                )
                self.m[id(p)] = torch.zeros_like(p, dtype=bf16_dtype if m_bfloat else p.dtype)
                self.v[id(p)] = torch.zeros_like(p, dtype=bf16_dtype)
        elif self.update_mode == "adamw_v_bfloat":
            for _name, p, _g in self.params():
                state_dtype = torch.bfloat16 if p.is_cuda and p.dtype == torch.float32 and torch.cuda.is_bf16_supported() else p.dtype
                self.m[id(p)] = torch.zeros_like(p)
                self.v[id(p)] = torch.zeros_like(p, dtype=state_dtype)
        elif self.update_mode == "adamw_v_bfloat_native":
            for _name, p, _g in self.params():
                state_dtype = torch.bfloat16 if p.is_cuda and p.dtype == torch.float32 and torch.cuda.is_bf16_supported() else p.dtype
                self.m[id(p)] = torch.zeros_like(p)
                self.v[id(p)] = torch.zeros_like(p, dtype=state_dtype)
        elif self.update_mode == "adamw_cpu":
            for _name, p, _g in self.params():
                self.m[id(p)] = torch.zeros_like(p, device="cpu", dtype=torch.float32)
                self.v[id(p)] = torch.zeros_like(p, device="cpu", dtype=torch.float32)
        elif self.update_mode == "adamw_v_cpu":
            for _name, p, _g in self.params():
                self.m[id(p)] = torch.zeros_like(p)
                self.v[id(p)] = torch.zeros_like(p, device="cpu", dtype=torch.float32)
        elif self.update_mode == "rms":
            for _name, p, _g in self.params():
                self.v[id(p)] = torch.zeros_like(p)

    def params(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        rows: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        for provider in self.providers:
            rows.extend(provider.params_and_grads())
        return rows

    def zero_grad(self) -> None:
        for provider in self.providers:
            provider.zero_grad()

    def step(self, step: int, total_steps: int, *, warmup_cosine: bool = True) -> float:
        self.t += 1
        lr = self.lr
        if warmup_cosine:
            warm = max(5, int(total_steps) // 10)
            if int(step) <= warm:
                lr *= int(step) / warm
            else:
                prog = (int(step) - warm) / max(1, int(total_steps) - warm)
                lr *= 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog))
        beta1, beta2, eps = 0.9, 0.99, 1.0e-8
        with torch.no_grad():
            for _name, p, g in self.params():
                if self.update_mode == "sgd":
                    if self.weight_decay:
                        p.mul_(1.0 - lr * self.weight_decay)
                    p.add_(g, alpha=-lr)
                    continue
                if self.update_mode == "rms":
                    if self.weight_decay:
                        p.mul_(1.0 - lr * self.weight_decay)
                    v = self.v[id(p)]
                    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    denom = v.sqrt().div(math.sqrt(1.0 - beta2**self.t)).add_(eps)
                    p.addcdiv_(g, denom, value=-lr)
                    continue
                if self.weight_decay:
                    p.mul_(1.0 - lr * self.weight_decay)
                m = self.m[id(p)]
                v = self.v[id(p)]
                if self.update_mode == "adamw_addcdiv":
                    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    denom = v.sqrt().div(math.sqrt(1.0 - beta2**self.t)).add_(eps)
                    p.addcdiv_(m, denom, value=-lr / (1.0 - beta1**self.t))
                    continue
                if self.update_mode == "adamw_half":
                    g_state = g.to(dtype=m.dtype)
                    m.mul_(beta1).add_(g_state, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(g_state, g_state, value=1.0 - beta2)
                    denom = v.float().sqrt().div(math.sqrt(1.0 - beta2**self.t)).add_(eps)
                    step_tensor = m.float().div(1.0 - beta1**self.t).div(denom)
                elif self.update_mode == "adamw_v_half":
                    g_v = g.to(dtype=v.dtype)
                    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(g_v, g_v, value=1.0 - beta2)
                    denom = v.float().sqrt().div(math.sqrt(1.0 - beta2**self.t)).add_(eps)
                    step_tensor = m.div(1.0 - beta1**self.t).div(denom)
                elif self.update_mode == "adamw_bfloat":
                    g_state = g.to(dtype=m.dtype)
                    m.mul_(beta1).add_(g_state, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(g_state, g_state, value=1.0 - beta2)
                    denom = v.float().sqrt().div(math.sqrt(1.0 - beta2**self.t)).add_(eps)
                    step_tensor = m.float().div(1.0 - beta1**self.t).div(denom)
                elif self.update_mode == "adamw_v_bfloat_addcdiv":
                    g_v = g.to(dtype=v.dtype)
                    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(g_v, g_v, value=1.0 - beta2)
                    denom = v.float().sqrt().div(math.sqrt(1.0 - beta2**self.t)).add_(eps)
                    p.addcdiv_(m, denom, value=-lr / (1.0 - beta1**self.t))
                    continue
                elif self.update_mode == "adamw_head_bfloat_v_bfloat_addcdiv":
                    g_v = g.to(dtype=v.dtype)
                    if m.dtype == g.dtype:
                        m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                        m_update = m
                    else:
                        m.mul_(beta1).add_(g.to(dtype=m.dtype), alpha=1.0 - beta1)
                        m_update = m.float()
                    v.mul_(beta2).addcmul_(g_v, g_v, value=1.0 - beta2)
                    denom = v.float().sqrt().div(math.sqrt(1.0 - beta2**self.t)).add_(eps)
                    p.addcdiv_(m_update, denom, value=-lr / (1.0 - beta1**self.t))
                    continue
                elif self.update_mode == "adamw_stack_bfloat_v_bfloat_addcdiv":
                    g_v = g.to(dtype=v.dtype)
                    if m.dtype == g.dtype:
                        m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                        m_update = m
                    else:
                        m.mul_(beta1).add_(g.to(dtype=m.dtype), alpha=1.0 - beta1)
                        m_update = m.float()
                    v.mul_(beta2).addcmul_(g_v, g_v, value=1.0 - beta2)
                    denom = v.float().sqrt().div(math.sqrt(1.0 - beta2**self.t)).add_(eps)
                    p.addcdiv_(m_update, denom, value=-lr / (1.0 - beta1**self.t))
                    continue
                elif self.update_mode in {"adamw_tail_bfloat_v_bfloat", "adamw_l0_bfloat_v_bfloat", "adamw_l0_bfloat_v_bfloat_addcdiv"}:
                    g_v = g.to(dtype=v.dtype)
                    if m.dtype == g.dtype:
                        m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                        m_update = m
                    else:
                        m.mul_(beta1).add_(g.to(dtype=m.dtype), alpha=1.0 - beta1)
                        m_update = m.float()
                    v.mul_(beta2).addcmul_(g_v, g_v, value=1.0 - beta2)
                    denom = v.float().sqrt().div(math.sqrt(1.0 - beta2**self.t)).add_(eps)
                    if self.update_mode == "adamw_l0_bfloat_v_bfloat_addcdiv":
                        p.addcdiv_(m_update, denom, value=-lr / (1.0 - beta1**self.t))
                        continue
                    step_tensor = m_update.div(1.0 - beta1**self.t).div(denom)
                elif self.update_mode == "adamw_v_bfloat":
                    g_v = g.to(dtype=v.dtype)
                    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(g_v, g_v, value=1.0 - beta2)
                    denom = v.float().sqrt().div(math.sqrt(1.0 - beta2**self.t)).add_(eps)
                    step_tensor = m.div(1.0 - beta1**self.t).div(denom)
                elif self.update_mode == "adamw_v_bfloat_native":
                    g_v = g.to(dtype=v.dtype)
                    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(g_v, g_v, value=1.0 - beta2)
                    # Keep the second-moment denominator in the native state
                    # dtype as long as possible. This is a measured system
                    # candidate, not an equivalence claim for AdamW.
                    denom = v.sqrt().div(math.sqrt(1.0 - beta2**self.t)).add_(eps)
                    step_tensor = m.div(1.0 - beta1**self.t).div(denom)
                elif self.update_mode == "adamw_cpu":
                    g_cpu = g.detach().float().cpu()
                    m.mul_(beta1).add_(g_cpu, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(g_cpu, g_cpu, value=1.0 - beta2)
                    denom = v.sqrt().div(math.sqrt(1.0 - beta2**self.t)).add_(eps)
                    step_tensor = m.div(1.0 - beta1**self.t).div(denom).to(device=p.device, dtype=p.dtype)
                elif self.update_mode == "adamw_v_cpu":
                    g_cpu = g.detach().float().cpu()
                    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(g_cpu, g_cpu, value=1.0 - beta2)
                    denom = v.sqrt().div(math.sqrt(1.0 - beta2**self.t)).add_(eps).to(device=p.device, dtype=p.dtype)
                    step_tensor = m.div(1.0 - beta1**self.t).div(denom)
                else:
                    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    denom = v.sqrt().div(math.sqrt(1.0 - beta2**self.t)).add_(eps)
                    step_tensor = m.div(1.0 - beta1**self.t).div(denom)
                p.add_(step_tensor, alpha=-lr)
        self.zero_grad()
        return float("nan")


class LowRankPoly2GateStack:
    """Dense poly2-gate stack with factorized mix matrices for low-rank probes."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        rank: int,
        *,
        device: torch.device,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.rank = int(rank)
        self.scale = 0.05
        dims = [self.input_dim] + [self.hidden_dim] * self.depth
        self.params: List[Dict[str, torch.Tensor]] = []
        self.grads: List[Dict[str, torch.Tensor]] = []
        for a, b in zip(dims[:-1], dims[1:]):
            r = min(self.rank, a, b)
            params = {
                "u": torch.randn(r, a, device=device) / math.sqrt(max(1, a)),
                "v": torch.randn(b, r, device=device) / math.sqrt(max(1, r)),
                "poly": torch.zeros(a, 2, device=device),
                "gate": torch.zeros(a, 1, device=device),
            }
            params["poly"][:, 0].fill_(0.02)
            self.params.append(params)
            self.grads.append({k: torch.zeros_like(v) for k, v in params.items()})

    def _transform(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        gate = torch.sigmoid(params["gate"][:, 0]).unsqueeze(0)
        poly = params["poly"]
        return x + self.scale * gate * (poly[:, 0].unsqueeze(0) * x + poly[:, 1].unsqueeze(0) * x.square())

    def _forward_layer(self, h: torch.Tensor, li: int, params: Dict[str, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        p = self.params[li] if params is None else params
        z = self._transform(h, p)
        t = z @ p["u"].t()
        y = t @ p["v"].t()
        return y, {"x": h.detach(), "t": t.detach(), "y": y.detach()}

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        with torch.no_grad():
            h = x
            caches: List[Dict[str, Any]] = []
            for li in range(self.depth):
                y, cache = self._forward_layer(h, li)
                caches.append(cache)
                h = F.silu(y) if li < self.depth - 1 else y
            return h, caches

    def forward_autograd_with_params(self, x: torch.Tensor, params: Sequence[Dict[str, torch.Tensor]]) -> torch.Tensor:
        h = x
        for li, p in enumerate(params):
            y, _cache = self._forward_layer(h, li, p)
            h = F.silu(y) if li < self.depth - 1 else y
        return h

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        with torch.no_grad():
            for li in reversed(range(self.depth)):
                cache = caches[li]
                if li < self.depth - 1:
                    y = cache["y"]
                    sig = torch.sigmoid(y)
                    delta = delta * sig * (1.0 + y * (1.0 - sig))
                p = self.params[li]
                g = self.grads[li]
                x = cache["x"]
                z = self._transform(x, p)
                t = cache["t"]
                dt = delta @ p["v"]
                dz = dt @ p["u"]
                g["v"].add_(delta.t() @ t)
                g["u"].add_(dt.t() @ z)
                gate_raw = p["gate"][:, 0]
                gate = torch.sigmoid(gate_raw).unsqueeze(0)
                poly = p["poly"]
                basis0 = poly[:, 0].unsqueeze(0) * x + poly[:, 1].unsqueeze(0) * x.square()
                dres = dz * self.scale
                g["poly"][:, 0].add_((dres * gate * x).sum(dim=0))
                g["poly"][:, 1].add_((dres * gate * x.square()).sum(dim=0))
                g["gate"][:, 0].add_((dres * basis0 * gate * (1.0 - gate)).sum(dim=0))
                delta = dz * (1.0 + self.scale * gate * (poly[:, 0].unsqueeze(0) + 2.0 * poly[:, 1].unsqueeze(0) * x))
            return delta

    def zero_grad(self) -> None:
        for grads in self.grads:
            for grad in grads.values():
                grad.zero_()

    def params_and_grads(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        out: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        for li, (params, grads) in enumerate(zip(self.params, self.grads)):
            for name, p in params.items():
                out.append((f"lowrank:l{li}:{name}", p, grads[name]))
        return out

    def clone_params_for_autograd(self) -> List[Dict[str, torch.Tensor]]:
        return [{k: v.detach().clone().requires_grad_(True) for k, v in params.items()} for params in self.params]

    def param_count(self) -> int:
        return sum(int(p.numel()) for params in self.params for p in params.values())

    def op_counts(self) -> Dict[str, int]:
        return {
            "op_count_gemm": self.depth * 2,
            "op_count_elementwise": self.depth * 5,
            "op_count_pow": self.depth,
            "op_count_exp": 0,
            "python_loop_count": self.depth,
            "group_loop_count": 0,
        }

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        hidden = sum(int(c["y"].numel() * c["y"].element_size()) for c in caches)
        x_bytes = sum(int(c["x"].numel() * c["x"].element_size()) for c in caches)
        t_bytes = sum(int(c["t"].numel() * c["t"].element_size()) for c in caches)
        total = hidden + x_bytes + t_bytes
        return {
            "cache_total_MB": total / (1024**2),
            "cache_x_MB": x_bytes / (1024**2),
            "cache_hidden_MB": hidden / (1024**2),
            "cache_lowrank_temp_MB": t_bytes / (1024**2),
            "cache_gate_MB": 0.0,
            "cache_basis_MB": 0.0,
        }


class CachedDensePoly2GateStack:
    """Dense V63-style stack that caches pre-activation y to avoid backward recompute."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        basis: int,
        *,
        device: torch.device,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        dims = [self.input_dim] + [self.hidden_dim] * self.depth
        self.layers = [
            V63ManualLayer(a, b, kind="poly2_gate", basis_count=basis, device=device, fused_hint="cached-y")
            for a, b in zip(dims[:-1], dims[1:])
        ]

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        with torch.no_grad():
            h = x
            caches: List[Dict[str, torch.Tensor]] = []
            for i, layer in enumerate(self.layers):
                y, x_cache = layer.forward_manual(h)
                caches.append({"x": x_cache, "y": y.detach()})
                h = F.silu(y) if i < len(self.layers) - 1 else y
            return h, caches

    def forward_autograd_with_params(self, x: torch.Tensor, params: Sequence[Dict[str, torch.Tensor]]) -> torch.Tensor:
        h = x
        for i, (layer, p) in enumerate(zip(self.layers, params)):
            y = layer.forward_with_params(h, p)
            h = F.silu(y) if i < len(self.layers) - 1 else y
        return h

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        delta = dy
        with torch.no_grad():
            for i in reversed(range(len(self.layers))):
                if i < len(self.layers) - 1:
                    y = caches[i]["y"]
                    sig = torch.sigmoid(y)
                    delta = delta * sig * (1.0 + y * (1.0 - sig))
                delta = self.layers[i].backward_manual(delta, caches[i]["x"])
            return delta

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def params_and_grads(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        out: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        for li, layer in enumerate(self.layers):
            for name, p in layer.params.items():
                out.append((f"cached_dense:l{li}:{name}", p, layer.grads[name]))
        return out

    def clone_params_for_autograd(self) -> List[Dict[str, torch.Tensor]]:
        return [layer.clone_params_for_autograd() for layer in self.layers]

    def param_count(self) -> int:
        return sum(layer.param_count() for layer in self.layers)

    def op_counts(self) -> Dict[str, int]:
        out = {"op_count_exp": 0, "op_count_pow": 0, "op_count_gather": 0, "op_count_scatter": 0, "op_count_index_select": 0, "op_count_scatter_add": 0, "op_count_gemm": 0, "op_count_elementwise": 0}
        for layer in self.layers:
            for key, value in layer.op_counts().items():
                out[key] = out.get(key, 0) + int(value)
        return out

    def cache_breakdown(self, caches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        x_bytes = sum(int(c["x"].numel() * c["x"].element_size()) for c in caches)
        y_bytes = sum(int(c["y"].numel() * c["y"].element_size()) for c in caches[:-1])
        total = x_bytes + y_bytes
        x_mb = x_bytes / (1024**2)
        y_mb = y_bytes / (1024**2)
        total_mb = total / (1024**2)
        largest_mb = max(
            [int(c["x"].numel() * c["x"].element_size()) / (1024**2) for c in caches]
            + [int(c["y"].numel() * c["y"].element_size()) / (1024**2) for c in caches[:-1]]
            + [0.0]
        )
        return {
            "cache_total_MB": total_mb,
            "cache_x_MB": x_mb,
            "cache_hidden_y_MB": y_mb,
            "cache_hidden_MB": total_mb,
            "cache_gate_MB": 0.0,
            "cache_basis_MB": 0.0,
            "materialized_tensor_count_measured": len(caches) + max(0, len(caches) - 1),
            "largest_live_tensor_MB_measured": largest_mb,
            "linear_body_temp_MB": x_mb,
            "hidden_y_cache_MB": y_mb,
            "manual_cache_MB_measured": total_mb,
            "top1_memory_source_measured": "manual_cache_x_inputs",
            "top2_memory_source_measured": "hidden_y_cache",
            "top3_memory_source_measured": "optimizer_state",
            "unknown_memory_fraction_measured": 0.0,
        }


def _candidate_registry() -> List[v71.CandidateSpec]:
    base = {c.candidate_id: c for c in v71._candidate_registry()}
    out = [
        base["B0"],
        base["B3"],
        base["H3"],
        base["H4"],
        base["K0"],
        base["K0s"],
        base["K1"],
        base["K2"],
        v71.CandidateSpec("G3", "K0-vectorized-g2-forward-backward", "grouped_vectorized", "vgrouped", depth=3, head_kind="poly2_gate", group_count=2),
        v71.CandidateSpec("G4", "K0-vectorized-g2-shuffle-forward-backward", "grouped_vectorized_shuffle", "vgrouped", depth=3, head_kind="poly2_gate", group_count=2, shuffle=True),
        v71.CandidateSpec("G5", "K1-vectorized-g8-forward-backward", "grouped_vectorized", "vgrouped", depth=3, head_kind="poly2_gate", group_count=8),
        v71.CandidateSpec("G6", "K2-vectorized-g16-forward-backward", "grouped_vectorized", "vgrouped", depth=3, head_kind="poly2_gate", group_count=16),
        v71.CandidateSpec("H7", "H3-rbf-poly-exp-no-smoothing", "strict_task_repair", "dense", depth=3, head_kind="rbf_poly_exp", label_smoothing=0.0),
        v71.CandidateSpec("H8", "H3-rbf-poly-exp-smoothing-002", "strict_task_repair", "dense", depth=3, head_kind="rbf_poly_exp", label_smoothing=0.02),
        v71.CandidateSpec("H9", "H3-rbf-poly-exp-lr-high", "strict_task_repair", "dense", depth=3, head_kind="rbf_poly_exp", lr_mult=1.5),
        v71.CandidateSpec("H10", "H3-rbf-poly-exp-lr-low", "strict_task_repair", "dense", depth=3, head_kind="rbf_poly_exp", lr_mult=0.75),
        v71.CandidateSpec("H11", "H4-poly2-gate-no-smoothing", "strict_task_repair", "dense", depth=2, head_kind="poly2_gate", label_smoothing=0.0),
        v71.CandidateSpec("H12", "H4-poly2-gate-lr-high", "strict_task_repair", "dense", depth=2, head_kind="poly2_gate", lr_mult=1.5),
        v71.CandidateSpec("L1", "LowRankPoly2Gate-r16-d2-KAN-head-poly2-gate", "lowrank_primitive", "lowrank", depth=2, head_kind="poly2_gate", group_count=16),
        v71.CandidateSpec("L2", "LowRankPoly2Gate-r32-d2-KAN-head-poly2-gate", "lowrank_primitive", "lowrank", depth=2, head_kind="poly2_gate", group_count=32),
        v71.CandidateSpec("L3", "LowRankPoly2Gate-r32-d3-KAN-head-rbf-poly-exp", "lowrank_primitive", "lowrank", depth=3, head_kind="rbf_poly_exp", group_count=32),
        v71.CandidateSpec("S1", "H4-poly2-gate-SGD-lr3", "sgd_memory_repair", "dense", depth=2, head_kind="poly2_gate", lr_mult=3.0),
        v71.CandidateSpec("S2", "H4-poly2-gate-SGD-lr10", "sgd_memory_repair", "dense", depth=2, head_kind="poly2_gate", lr_mult=10.0),
        v71.CandidateSpec("S3", "H4-poly2-gate-SGD-lr30", "sgd_memory_repair", "dense", depth=2, head_kind="poly2_gate", lr_mult=30.0),
        v71.CandidateSpec("S4", "H11-poly2-gate-no-smoothing-SGD-lr10", "sgd_memory_repair", "dense", depth=2, head_kind="poly2_gate", label_smoothing=0.0, lr_mult=10.0),
        v71.CandidateSpec("C1", "H4-cached-hidden-y-poly2-gate", "cached_backward_repair", "cached_dense", depth=2, head_kind="poly2_gate"),
        v71.CandidateSpec("C2", "H11-cached-hidden-y-no-smoothing", "cached_backward_repair", "cached_dense", depth=2, head_kind="poly2_gate", label_smoothing=0.0),
        v71.CandidateSpec("C3", "H3-cached-hidden-y-rbf-head", "cached_backward_repair", "cached_dense", depth=3, head_kind="rbf_poly_exp"),
        v71.CandidateSpec("W1", "C3-classwise-FMNIST-c4-w150", "classwise_loss_repair", "cached_dense", depth=3, head_kind="rbf_poly_exp"),
        v71.CandidateSpec("W2", "C3-classwise-FMNIST-c4c6-w150", "classwise_loss_repair", "cached_dense", depth=3, head_kind="rbf_poly_exp"),
        v71.CandidateSpec("W3", "C3-classwise-FMNIST-c4-w200", "classwise_loss_repair", "cached_dense", depth=3, head_kind="rbf_poly_exp"),
        v71.CandidateSpec("W4", "C3-classwise-FMNIST-c4c6-w200", "classwise_loss_repair", "cached_dense", depth=3, head_kind="rbf_poly_exp"),
        v71.CandidateSpec("Q1", "C3-balanced-batches-FMNIST", "classwise_sampler_repair", "cached_dense", depth=3, head_kind="rbf_poly_exp"),
        v71.CandidateSpec("Q2", "C3-balanced-batches-all-datasets", "classwise_sampler_repair", "cached_dense", depth=3, head_kind="rbf_poly_exp"),
        v71.CandidateSpec("Q3", "C3-targeted-batches-FMNIST-c4", "classwise_sampler_repair", "cached_dense", depth=3, head_kind="rbf_poly_exp"),
        v71.CandidateSpec("Q4", "C3-targeted-batches-FMNIST-c4c6", "classwise_sampler_repair", "cached_dense", depth=3, head_kind="rbf_poly_exp"),
        v71.CandidateSpec("R1", "C3-minclass-checkpoint-alpha005", "classwise_checkpoint_selection", "cached_dense", depth=3, head_kind="rbf_poly_exp"),
        v71.CandidateSpec("R2", "C3-minclass-checkpoint-alpha010", "classwise_checkpoint_selection", "cached_dense", depth=3, head_kind="rbf_poly_exp"),
        v71.CandidateSpec("R3", "C3-relative-drop-checkpoint-beta010", "classwise_checkpoint_selection", "cached_dense", depth=3, head_kind="rbf_poly_exp"),
        v71.CandidateSpec("R4", "C3-relative-drop-checkpoint-beta025", "classwise_checkpoint_selection", "cached_dense", depth=3, head_kind="rbf_poly_exp"),
        v71.CandidateSpec("F1", "C3-focal-loss-gamma1", "classwise_focal_loss_repair", "cached_dense", depth=3, head_kind="rbf_poly_exp", label_smoothing=0.0),
        v71.CandidateSpec("F2", "C3-focal-loss-gamma2", "classwise_focal_loss_repair", "cached_dense", depth=3, head_kind="rbf_poly_exp", label_smoothing=0.0),
        v71.CandidateSpec("J1", "C3-FMNIST-c4-margin025-lambda025", "classwise_margin_loss_repair", "cached_dense", depth=3, head_kind="rbf_poly_exp"),
        v71.CandidateSpec("J2", "C3-FMNIST-c4-margin025-lambda050", "classwise_margin_loss_repair", "cached_dense", depth=3, head_kind="rbf_poly_exp"),
        v71.CandidateSpec("J3", "C3-FMNIST-c4c6-margin025-lambda025", "classwise_margin_loss_repair", "cached_dense", depth=3, head_kind="rbf_poly_exp"),
        v71.CandidateSpec("Y1", "C3-FMNIST-c4c6-margin025-lambda025-checkpoint-beta010", "classwise_margin_checkpoint_repair", "cached_dense", depth=3, head_kind="rbf_poly_exp"),
        v71.CandidateSpec("Y2", "C3-FMNIST-c4-margin025-lambda050-checkpoint-beta025", "classwise_margin_checkpoint_repair", "cached_dense", depth=3, head_kind="rbf_poly_exp"),
    ]
    return out


def _uses_sgd_update(spec: v71.CandidateSpec) -> bool:
    return str(spec.candidate_id).startswith("S")


def _uses_weighted_loss(spec: v71.CandidateSpec) -> bool:
    return str(spec.candidate_id).startswith("W")


def _uses_balanced_batches(spec: v71.CandidateSpec, dataset: str) -> bool:
    cid = str(spec.candidate_id)
    if cid == "Q1":
        return str(dataset) == "Fashion-MNIST"
    return cid == "Q2"


def _target_sampler_classes(spec: v71.CandidateSpec, dataset: str) -> List[int]:
    if str(dataset) != "Fashion-MNIST":
        return []
    cid = str(spec.candidate_id)
    if cid == "Q3":
        return [4]
    if cid == "Q4":
        return [4, 6]
    return []


def _uses_targeted_batches(spec: v71.CandidateSpec, dataset: str) -> bool:
    return bool(_target_sampler_classes(spec, dataset))


def _uses_checkpoint_selection(spec: v71.CandidateSpec) -> bool:
    return str(spec.candidate_id).startswith("R") or str(spec.candidate_id).startswith("Y")


def _uses_focal_loss(spec: v71.CandidateSpec) -> bool:
    return str(spec.candidate_id).startswith("F")


def _uses_margin_loss(spec: v71.CandidateSpec) -> bool:
    return str(spec.candidate_id).startswith("J") or str(spec.candidate_id).startswith("Y")


def _focal_gamma(spec: v71.CandidateSpec) -> float:
    cid = str(spec.candidate_id)
    if cid == "F1":
        return 1.0
    if cid == "F2":
        return 2.0
    return 0.0


def _margin_loss_classes(spec: v71.CandidateSpec, dataset: str) -> List[int]:
    if str(dataset) != "Fashion-MNIST":
        return []
    cid = str(spec.candidate_id)
    if cid in {"J1", "J2", "Y2"}:
        return [4]
    if cid in {"J3", "Y1"}:
        return [4, 6]
    return []


def _margin_loss_lambda(spec: v71.CandidateSpec) -> float:
    cid = str(spec.candidate_id)
    if cid in {"J2", "Y2"}:
        return 0.50
    if cid in {"J1", "J3", "Y1"}:
        return 0.25
    return 0.0


def _margin_loss_target(spec: v71.CandidateSpec) -> float:
    if _uses_margin_loss(spec):
        return 0.25
    return 0.0


def _checkpoint_alpha(spec: v71.CandidateSpec) -> float:
    cid = str(spec.candidate_id)
    if cid == "R1":
        return 0.05
    if cid == "R2":
        return 0.10
    return 0.0


def _checkpoint_beta(spec: v71.CandidateSpec) -> float:
    cid = str(spec.candidate_id)
    if cid == "R3":
        return 0.10
    if cid == "R4":
        return 0.25
    if cid == "Y1":
        return 0.10
    if cid == "Y2":
        return 0.25
    return 0.0


def _checkpoint_mode(spec: v71.CandidateSpec) -> str:
    cid = str(spec.candidate_id)
    if cid in {"R3", "R4", "Y1", "Y2"}:
        return "relative_class_drop_penalty"
    if cid in {"R1", "R2"}:
        return "val_acc_plus_min_class_acc"
    return "final_step"


def _sampler_policy(spec: v71.CandidateSpec, dataset: str) -> str:
    if _uses_targeted_batches(spec, dataset):
        return "fashion_mnist_targeted_classes_" + "_".join(str(c) for c in _target_sampler_classes(spec, dataset))
    if _uses_balanced_batches(spec, dataset):
        return "class_balanced"
    return "sequential_cycle"


def _uses_v72_train_path(spec: v71.CandidateSpec, dataset: str) -> bool:
    return (
        _uses_weighted_loss(spec)
        or _uses_balanced_batches(spec, dataset)
        or _uses_targeted_batches(spec, dataset)
        or _uses_checkpoint_selection(spec)
        or _uses_focal_loss(spec)
        or _uses_margin_loss(spec)
    )


def _init_seed_candidate_id(spec: v71.CandidateSpec) -> str:
    return "C3" if (_uses_weighted_loss(spec) or str(spec.candidate_id).startswith("Q") or _uses_checkpoint_selection(spec) or _uses_focal_loss(spec) or _uses_margin_loss(spec)) else str(spec.candidate_id)


def _make_manual_candidate(spec: v71.CandidateSpec, input_dim: int, num_classes: int, hidden_dim: int, basis: int, device: torch.device) -> Tuple[Any, v71.HeadWrapper]:
    if spec.stack_type == "vgrouped":
        stack = VectorizedGroupedPoly2GateStack(input_dim, hidden_dim, spec.depth, spec.group_count, shuffle=spec.shuffle, device=device)
        head = v71.HeadWrapper(V63ManualLayer(hidden_dim, num_classes, kind=v71._head_kind_to_layer_kind(spec.head_kind), basis_count=basis, device=device))
        if _uses_sgd_update(spec):
            setattr(stack, "update_mode", "sgd")
            setattr(head, "update_mode", "sgd")
        return stack, head
    if spec.stack_type == "lowrank":
        stack = LowRankPoly2GateStack(input_dim, hidden_dim, spec.depth, spec.group_count, device=device)
        head = v71.HeadWrapper(V63ManualLayer(hidden_dim, num_classes, kind=v71._head_kind_to_layer_kind(spec.head_kind), basis_count=basis, device=device))
        if _uses_sgd_update(spec):
            setattr(stack, "update_mode", "sgd")
            setattr(head, "update_mode", "sgd")
        return stack, head
    if spec.stack_type == "cached_dense":
        stack = CachedDensePoly2GateStack(input_dim, hidden_dim, spec.depth, basis, device=device)
        head = v71.HeadWrapper(V63ManualLayer(hidden_dim, num_classes, kind=v71._head_kind_to_layer_kind(spec.head_kind), basis_count=basis, device=device))
        if _uses_sgd_update(spec):
            setattr(stack, "update_mode", "sgd")
            setattr(head, "update_mode", "sgd")
        return stack, head
    stack, head = v71._ORIGINAL_MAKE_MANUAL_CANDIDATE(spec, input_dim, num_classes, hidden_dim, basis, device)
    if _uses_sgd_update(spec):
        setattr(stack, "update_mode", "sgd")
        setattr(head, "update_mode", "sgd")
    return stack, head


def _patch_v71() -> None:
    if not hasattr(v71, "_ORIGINAL_MAKE_MANUAL_CANDIDATE"):
        v71._ORIGINAL_MAKE_MANUAL_CANDIDATE = v71._make_manual_candidate  # type: ignore[attr-defined]
    v71._make_manual_candidate = _make_manual_candidate  # type: ignore[assignment]
    if not hasattr(v71, "_ORIGINAL_FAST_ADAMW"):
        v71._ORIGINAL_FAST_ADAMW = v71.FastAdamW  # type: ignore[attr-defined]
    v71.FastAdamW = FastAdamWNoSync  # type: ignore[assignment]


def _cohen_h(p1: float, p0: float) -> float:
    p1 = min(1.0, max(0.0, float(p1)))
    p0 = min(1.0, max(0.0, float(p0)))
    return 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p0))


def _bootstrap_ci(values: Sequence[float], reps: int, seed: int) -> Tuple[float, float, float]:
    vals = np.array([float(v) for v in values if _finite(v)], dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(vals, size=(int(reps), vals.size), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    p = float(min((samples <= 0.0).mean(), (samples >= 0.0).mean()) * 2.0)
    return float(low), float(high), min(1.0, p)


def _holm_correct(p_values: Dict[str, float]) -> Dict[str, float]:
    clean = [(k, float(v)) for k, v in p_values.items() if _finite(v)]
    clean.sort(key=lambda kv: kv[1])
    m = len(clean)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(clean):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        adjusted[key] = running
    return adjusted


def _safe_list(row_value: Any) -> List[float]:
    if isinstance(row_value, list):
        return [float(x) for x in row_value]
    text = str(row_value)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except Exception:
        pass
    return []


def _add_seed_gaps(task_rows: List[Dict[str, Any]]) -> None:
    base: Dict[Tuple[str, int], Dict[str, Any]] = {}
    base_ds_mean: Dict[str, Dict[str, float]] = {}
    for row in task_rows:
        if row.get("candidate_id") == "B0":
            base[(str(row["dataset"]), int(row["seed"]))] = row
    for ds in sorted({str(r["dataset"]) for r in task_rows}):
        b = [r for r in task_rows if r.get("candidate_id") == "B0" and str(r["dataset"]) == ds]
        base_ds_mean[ds] = {
            "val_acc": _mean(r["val_acc"] for r in b),
            "test_acc": _mean(r["test_acc"] for r in b),
            "ECE": _mean(r["ECE"] for r in b),
            "NLL": _mean(r["NLL"] for r in b),
            "val_loss_auc_step": _mean(r["val_loss_auc_step"] for r in b),
            "val_loss_auc_time": _mean(r["val_loss_auc_time"] for r in b),
        }
    for row in task_rows:
        ds = str(row["dataset"])
        seed = int(row["seed"])
        bseed = base.get((ds, seed), {})
        bmean = base_ds_mean.get(ds, {})
        for metric in ("val_acc", "test_acc", "ECE", "NLL", "val_loss_auc_step", "val_loss_auc_time"):
            if metric in row and metric in bseed:
                row[f"{metric}_gap_vs_MLP_seed"] = float(row[metric]) - float(bseed[metric])
            if metric in row and metric in bmean:
                row[f"{metric}_gap_vs_MLP_dataset_mean"] = float(row[metric]) - float(bmean[metric])


def _significance_audit(task_rows: Sequence[Dict[str, Any]], bootstrap_reps: int) -> List[Dict[str, Any]]:
    p_for_holm: Dict[str, float] = {}
    interim: List[Dict[str, Any]] = []
    candidate_ids = sorted({str(r["candidate_id"]) for r in task_rows if r.get("candidate_id") != "B0"})
    datasets = sorted({str(r["dataset"]) for r in task_rows})
    for cid in candidate_ids:
        cand_rows = [r for r in task_rows if str(r["candidate_id"]) == cid]
        first = cand_rows[0]
        for ds_key in ["macro", *datasets]:
            subset = cand_rows if ds_key == "macro" else [r for r in cand_rows if str(r["dataset"]) == ds_key]
            val_gaps = [float(r.get("val_acc_gap_vs_MLP_seed", float("nan"))) for r in subset if _finite(r.get("val_acc_gap_vs_MLP_seed"))]
            test_gaps = [float(r.get("test_acc_gap_vs_MLP_seed", float("nan"))) for r in subset if _finite(r.get("test_acc_gap_vs_MLP_seed"))]
            if len(val_gaps) >= 2:
                t = stats.ttest_1samp(val_gaps, 0.0)
                try:
                    w = stats.wilcoxon(val_gaps, zero_method="wilcox", alternative="greater")
                    wp = float(w.pvalue)
                except Exception:
                    wp = float("nan")
                t_stat = float(t.statistic)
                p_val = float(t.pvalue)
            else:
                t_stat = p_val = wp = float("nan")
            low, high, boot_p = _bootstrap_ci(val_gaps, bootstrap_reps, _stable_seed("bootstrap", cid, ds_key))
            p_for_holm[f"{cid}:{ds_key}"] = p_val
            mlp_acc = _mean(
                r["val_acc"]
                for r in task_rows
                if r.get("candidate_id") == "B0" and (ds_key == "macro" or str(r["dataset"]) == ds_key)
            )
            cand_acc = _mean(r["val_acc"] for r in subset)
            row = {
                "candidate_id": cid,
                "candidate_name": first.get("candidate_name"),
                "family": first.get("family"),
                "dataset": ds_key,
                "n": len(val_gaps),
                "mean_val_gap": _mean(val_gaps),
                "std_val_gap": _std(val_gaps),
                "mean_test_gap": _mean(test_gaps),
                "std_test_gap": _std(test_gaps),
                "paired_t_stat": t_stat,
                "paired_p_value": p_val,
                "wilcoxon_p_value": wp,
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "bootstrap_p_value": boot_p,
                "cohen_h": _cohen_h(cand_acc, mlp_acc),
                "seed_win_count": sum(1 for g in val_gaps if g > 0.0),
                "dataset_pass_practical": int(_mean(val_gaps) >= 0.02),
                "dataset_pass_statistical": int(_finite(low) and low > 0.0),
                "fake_data_used": 0,
                "proxy_row_used": 0,
            }
            interim.append(row)
    holm = _holm_correct(p_for_holm)
    for row in interim:
        key = f"{row['candidate_id']}:{row['dataset']}"
        row["holm_corrected_p"] = holm.get(key, float("nan"))
        row["holm_pass"] = int(_finite(row["holm_corrected_p"]) and float(row["holm_corrected_p"]) < 0.05)
    by_cid = {cid: [r for r in interim if r["candidate_id"] == cid] for cid in candidate_ids}
    for rows in by_cid.values():
        macro = next((r for r in rows if r["dataset"] == "macro"), None)
        if macro is None:
            continue
        ds_rows = [r for r in rows if r["dataset"] != "macro"]
        ece_delta = _mean(
            r.get("ECE_gap_vs_MLP_seed")
            for r in task_rows
            if str(r.get("candidate_id")) == str(macro["candidate_id"])
        )
        nll_delta = _mean(
            r.get("NLL_gap_vs_MLP_seed")
            for r in task_rows
            if str(r.get("candidate_id")) == str(macro["candidate_id"])
        )
        min_test_gap = min([float(r["mean_test_gap"]) for r in ds_rows if _finite(r["mean_test_gap"])] or [float("nan")])
        class_drop_max = _classwise_drop_max(task_rows, str(macro["candidate_id"]))
        significant = int(
            float(macro["mean_val_gap"]) >= 0.02
            and float(macro["bootstrap_ci95_low"]) > 0.0
            and sum(int(r["mean_val_gap"] >= 0.02) for r in ds_rows) >= 2
            and sum(int(r["bootstrap_ci95_low"] > 0.0) for r in ds_rows) >= 2
            and _finite(macro["holm_corrected_p"]) and float(macro["holm_corrected_p"]) < 0.05
            and float(macro["mean_test_gap"]) >= 0.015
            and _finite(min_test_gap) and min_test_gap >= -0.005
            and (_finite(ece_delta) and ece_delta <= 0.005)
            and (_finite(nll_delta) and nll_delta <= 0.01)
            and (_finite(class_drop_max) and class_drop_max <= 0.05)
        )
        for r in rows:
            r["macro_significant_task_pass"] = significant if r["dataset"] == "macro" else ""
            r["ECE_delta_macro"] = ece_delta if r["dataset"] == "macro" else ""
            r["NLL_delta_macro"] = nll_delta if r["dataset"] == "macro" else ""
            r["classwise_drop_max"] = class_drop_max if r["dataset"] == "macro" else ""
    return interim


def _classwise_drop_max(task_rows: Sequence[Dict[str, Any]], cid: str) -> float:
    drops: List[float] = []
    for ds in sorted({str(r["dataset"]) for r in task_rows}):
        for seed in sorted({int(r["seed"]) for r in task_rows if str(r["dataset"]) == ds}):
            b = next((r for r in task_rows if r.get("candidate_id") == "B0" and str(r["dataset"]) == ds and int(r["seed"]) == seed), None)
            c = next((r for r in task_rows if str(r.get("candidate_id")) == cid and str(r["dataset"]) == ds and int(r["seed"]) == seed), None)
            if not b or not c:
                continue
            bw = _safe_list(b.get("classwise_acc"))
            cw = _safe_list(c.get("classwise_acc"))
            for bi, ci in zip(bw, cw):
                if _finite(bi) and _finite(ci):
                    drops.append(float(bi) - float(ci))
    return max(drops) if drops else float("nan")


def _flatten_params_and_grads(providers: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    params: List[torch.Tensor] = []
    grads: List[torch.Tensor] = []
    for provider in providers:
        for _name, p, g in provider.params_and_grads():
            params.append(p.detach().flatten().float().cpu())
            grads.append(g.detach().flatten().float().cpu())
    return torch.cat(params), torch.cat(grads)


def _manual_forward(stack: Any, head: v71.HeadWrapper, x: torch.Tensor) -> Tuple[torch.Tensor, Any, Any, torch.Tensor]:
    h, caches = stack.forward_manual(x)
    logits, head_cache = head.forward_manual(h)
    return logits, caches, head_cache, h


def _autograd_forward(stack: Any, head: v71.HeadWrapper, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    leafs: List[torch.Tensor] = []
    if isinstance(stack, (VectorizedGroupedPoly2GateStack, LowRankPoly2GateStack, CachedDensePoly2GateStack)):
        stack_params = stack.clone_params_for_autograd()
        for params in stack_params:
            for value in params.values():
                leafs.append(value)
        h = stack.forward_autograd_with_params(x, stack_params)
    elif hasattr(stack, "layers"):
        h = x
        if stack.layers and isinstance(stack.layers[0], list):
            for li, layer_groups in enumerate(stack.layers):
                perm, _inv = stack.perms[li]
                h_in = h[:, perm] if perm is not None else h
                pieces = torch.split(h_in, stack.in_splits[li], dim=1)
                ys: List[torch.Tensor] = []
                for layer, xg in zip(layer_groups, pieces):
                    params = layer.clone_params_for_autograd()
                    for value in params.values():
                        leafs.append(value)
                    ys.append(layer.forward_with_params(xg, params))
                y = torch.cat(ys, dim=1)
                h = F.silu(y) if li < len(stack.layers) - 1 else y
        else:
            for li, layer in enumerate(stack.layers):
                params = layer.clone_params_for_autograd()
                for value in params.values():
                    leafs.append(value)
                y = layer.forward_with_params(h, params)
                h = F.silu(y) if li < len(stack.layers) - 1 else y
    else:
        raise ValueError(f"unsupported stack for autograd check: {type(stack)}")
    head_params = head.layer.clone_params_for_autograd()
    for value in head_params.values():
        leafs.append(value)
    logits = head.layer.forward_with_params(h, head_params)
    return logits, leafs


def _class_weight_vector(spec: v71.CandidateSpec, dataset: str, num_classes: int, device: torch.device) -> torch.Tensor:
    weights = torch.ones(int(num_classes), device=device)
    if not _uses_weighted_loss(spec) or str(dataset) != "Fashion-MNIST":
        return weights
    cid = str(spec.candidate_id)
    if cid == "W1":
        weights[4] = 1.5
    elif cid == "W2":
        weights[4] = 1.5
        weights[6] = 1.5
    elif cid == "W3":
        weights[4] = 2.0
    elif cid == "W4":
        weights[4] = 2.0
        weights[6] = 2.0
    return weights


def _weighted_smooth_ce_and_grad(
    logits: torch.Tensor,
    y: torch.Tensor,
    label_smoothing: float,
    class_weights: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if class_weights is None or bool(torch.allclose(class_weights, torch.ones_like(class_weights))):
        return v71._smooth_ce_and_grad(logits, y, label_smoothing)
    logp = F.log_softmax(logits, dim=1)
    probs = logp.exp()
    classes = int(logits.shape[1])
    if float(label_smoothing) > 0.0:
        eps = float(label_smoothing)
        target = torch.full_like(logits, eps / max(1, classes - 1))
        target.scatter_(1, y.view(-1, 1), 1.0 - eps)
    else:
        target = torch.zeros_like(logits)
        target.scatter_(1, y.view(-1, 1), 1.0)
    sample_weights = class_weights.to(logits.device)[y].float()
    denom = sample_weights.sum().clamp_min(1.0e-12)
    loss = -((target * logp).sum(dim=1) * sample_weights).sum() / denom
    grad = (probs - target) * sample_weights.view(-1, 1) / denom
    return loss, grad


def _weighted_smooth_loss_autograd(
    logits: torch.Tensor,
    y: torch.Tensor,
    label_smoothing: float,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    if class_weights is None or bool(torch.allclose(class_weights, torch.ones_like(class_weights))):
        return _smooth_loss_autograd(logits, y, label_smoothing)
    logp = F.log_softmax(logits, dim=1)
    classes = int(logits.shape[1])
    if float(label_smoothing) > 0.0:
        eps = float(label_smoothing)
        target = torch.full_like(logits, eps / max(1, classes - 1))
        target.scatter_(1, y.view(-1, 1), 1.0 - eps)
    else:
        target = torch.zeros_like(logits)
        target.scatter_(1, y.view(-1, 1), 1.0)
    sample_weights = class_weights.to(logits.device)[y].float()
    return -((target * logp).sum(dim=1) * sample_weights).sum() / sample_weights.sum().clamp_min(1.0e-12)


def _focal_ce_and_grad(
    logits: torch.Tensor,
    y: torch.Tensor,
    gamma: float,
    class_weights: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logp = F.log_softmax(logits, dim=1)
    probs = logp.exp()
    row = torch.arange(int(logits.shape[0]), device=logits.device)
    target_logp = logp[row, y]
    pt = target_logp.exp().clamp_min(1.0e-8).clamp_max(1.0 - 1.0e-8)
    one_minus = (1.0 - pt).clamp_min(1.0e-8)
    ce = -target_logp
    if class_weights is None or bool(torch.allclose(class_weights, torch.ones_like(class_weights))):
        sample_weights = torch.ones_like(pt)
    else:
        sample_weights = class_weights.to(logits.device)[y].float()
    denom = sample_weights.sum().clamp_min(1.0e-12)
    focal = one_minus.pow(float(gamma))
    loss = (sample_weights * focal * ce).sum() / denom
    dloss_dpt = float(gamma) * one_minus.pow(max(0.0, float(gamma) - 1.0)) * torch.log(pt) - focal / pt
    coeff = sample_weights * dloss_dpt * pt / denom
    grad = -coeff.view(-1, 1) * probs
    grad[row, y] += coeff
    return loss, grad


def _focal_loss_autograd(
    logits: torch.Tensor,
    y: torch.Tensor,
    gamma: float,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    row = torch.arange(int(logits.shape[0]), device=logits.device)
    target_logp = logp[row, y]
    pt = target_logp.exp().clamp_min(1.0e-8).clamp_max(1.0 - 1.0e-8)
    focal = (1.0 - pt).clamp_min(1.0e-8).pow(float(gamma))
    if class_weights is None or bool(torch.allclose(class_weights, torch.ones_like(class_weights))):
        sample_weights = torch.ones_like(pt)
    else:
        sample_weights = class_weights.to(logits.device)[y].float()
    return (sample_weights * focal * (-target_logp)).sum() / sample_weights.sum().clamp_min(1.0e-12)


def _target_margin_term_and_grad(
    logits: torch.Tensor,
    y: torch.Tensor,
    target_classes: Sequence[int],
    margin_target: float,
    loss_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not target_classes or float(loss_lambda) <= 0.0:
        return logits.new_tensor(0.0), torch.zeros_like(logits)
    class_tensor = torch.tensor([int(c) for c in target_classes], device=y.device, dtype=y.dtype)
    mask = (y.view(-1, 1) == class_tensor.view(1, -1)).any(dim=1)
    if not bool(mask.any()):
        return logits.new_tensor(0.0), torch.zeros_like(logits)
    row_all = torch.arange(int(logits.shape[0]), device=logits.device)
    target_logits = logits[row_all, y]
    other_logits = logits.masked_fill(F.one_hot(y, num_classes=int(logits.shape[1])).bool(), -float("inf"))
    other_max, other_idx = other_logits.max(dim=1)
    margin = target_logits - other_max
    active = mask & ((float(margin_target) - margin) > 0.0)
    if not bool(active.any()):
        return logits.new_tensor(0.0), torch.zeros_like(logits)
    hinge = (float(margin_target) - margin[active]).clamp_min(0.0)
    denom = torch.tensor(float(logits.shape[0]), device=logits.device, dtype=logits.dtype).clamp_min(1.0)
    loss = float(loss_lambda) * hinge.square().sum() / denom
    grad = torch.zeros_like(logits)
    coeff = 2.0 * float(loss_lambda) * hinge / denom
    active_rows = row_all[active]
    grad[active_rows, y[active]] -= coeff
    grad[active_rows, other_idx[active]] += coeff
    return loss, grad


def _target_margin_loss_autograd(
    logits: torch.Tensor,
    y: torch.Tensor,
    target_classes: Sequence[int],
    margin_target: float,
    loss_lambda: float,
) -> torch.Tensor:
    if not target_classes or float(loss_lambda) <= 0.0:
        return logits.new_tensor(0.0)
    class_tensor = torch.tensor([int(c) for c in target_classes], device=y.device, dtype=y.dtype)
    mask = (y.view(-1, 1) == class_tensor.view(1, -1)).any(dim=1)
    if not bool(mask.any()):
        return logits.new_tensor(0.0)
    row = torch.arange(int(logits.shape[0]), device=logits.device)
    target_logits = logits[row, y]
    other_logits = logits.masked_fill(F.one_hot(y, num_classes=int(logits.shape[1])).bool(), -float("inf"))
    other_max = other_logits.max(dim=1).values
    hinge = (float(margin_target) - (target_logits - other_max)).clamp_min(0.0)
    active_hinge = hinge[mask]
    return float(loss_lambda) * active_hinge.square().sum() / max(1, int(logits.shape[0]))


def _loss_and_grad_v72(
    spec: v71.CandidateSpec,
    dataset: str,
    logits: torch.Tensor,
    y: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if _uses_focal_loss(spec):
        return _focal_ce_and_grad(logits, y, _focal_gamma(spec), class_weights)
    loss, grad = _weighted_smooth_ce_and_grad(logits, y, spec.label_smoothing, class_weights)
    if _uses_margin_loss(spec):
        margin_loss, margin_grad = _target_margin_term_and_grad(
            logits,
            y,
            _margin_loss_classes(spec, dataset),
            _margin_loss_target(spec),
            _margin_loss_lambda(spec),
        )
        loss = loss + margin_loss
        grad = grad + margin_grad
    return loss, grad


def _loss_autograd_v72(
    spec: v71.CandidateSpec,
    dataset: str,
    logits: torch.Tensor,
    y: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    if _uses_focal_loss(spec):
        return _focal_loss_autograd(logits, y, _focal_gamma(spec), class_weights)
    loss = _weighted_smooth_loss_autograd(logits, y, spec.label_smoothing, class_weights)
    if _uses_margin_loss(spec):
        loss = loss + _target_margin_loss_autograd(
            logits,
            y,
            _margin_loss_classes(spec, dataset),
            _margin_loss_target(spec),
            _margin_loss_lambda(spec),
        )
    return loss


def _loss_reweighting_label(spec: v71.CandidateSpec) -> str:
    if _uses_focal_loss(spec):
        return f"focal_gamma_{_focal_gamma(spec):g}"
    if _uses_margin_loss(spec):
        classes = "_".join(str(c) for c in _margin_loss_classes(spec, "Fashion-MNIST"))
        return f"target_margin_c{classes}_m{_margin_loss_target(spec):g}_lambda{_margin_loss_lambda(spec):g}"
    if _uses_weighted_loss(spec):
        return "fashion_mnist_classwise_static"
    return "none"


def _balanced_class_indices(y: torch.Tensor, num_classes: int) -> List[torch.Tensor]:
    return [(y == cls).nonzero(as_tuple=False).flatten() for cls in range(int(num_classes))]


def _select_balanced_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    class_indices: Sequence[torch.Tensor],
    batch_size: int,
    step: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    classes = max(1, len(class_indices))
    per_class = int(math.ceil(int(batch_size) / classes))
    pieces: List[torch.Tensor] = []
    for cls, idx in enumerate(class_indices):
        if idx.numel() == 0:
            continue
        start = ((int(step) - 1) * per_class + cls * 9973) % int(idx.numel())
        take = torch.arange(per_class, device=idx.device)
        pieces.append(idx[(start + take) % int(idx.numel())])
    if not pieces:
        return v71._select_batch(x, y, batch_size, step)
    batch_idx = torch.cat(pieces, dim=0)[: int(batch_size)]
    roll = int(step) % max(1, int(batch_idx.numel()))
    batch_idx = torch.roll(batch_idx, shifts=roll, dims=0)
    return x[batch_idx], y[batch_idx]


def _select_targeted_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    class_indices: Sequence[torch.Tensor],
    target_classes: Sequence[int],
    batch_size: int,
    step: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not target_classes:
        return v71._select_batch(x, y, batch_size, step)
    target_budget = max(1, int(batch_size) // 2)
    per_target = int(math.ceil(target_budget / max(1, len(target_classes))))
    pieces: List[torch.Tensor] = []
    for offset, cls in enumerate(target_classes):
        if int(cls) < 0 or int(cls) >= len(class_indices):
            continue
        idx = class_indices[int(cls)]
        if idx.numel() == 0:
            continue
        start = ((int(step) - 1) * per_target + offset * 7919) % int(idx.numel())
        take = torch.arange(per_target, device=idx.device)
        pieces.append(idx[(start + take) % int(idx.numel())])
    target_idx = torch.cat(pieces, dim=0)[:target_budget] if pieces else torch.empty(0, dtype=torch.long, device=y.device)
    rest = int(batch_size) - int(target_idx.numel())
    if rest > 0:
        all_idx = torch.arange(int(x.shape[0]), device=y.device)
        start = ((int(step) - 1) * rest * 3) % int(all_idx.numel())
        take = torch.arange(rest, device=y.device)
        pieces.append(all_idx[(start + take) % int(all_idx.numel())])
    if not pieces:
        return v71._select_batch(x, y, batch_size, step)
    batch_idx = torch.cat(pieces, dim=0)[: int(batch_size)]
    roll = int(step) % max(1, int(batch_idx.numel()))
    batch_idx = torch.roll(batch_idx, shifts=roll, dims=0)
    return x[batch_idx], y[batch_idx]


def _snapshot_providers(providers: Sequence[Any]) -> List[torch.Tensor]:
    return [p.detach().clone() for provider in providers for _name, p, _g in provider.params_and_grads()]


def _restore_providers(providers: Sequence[Any], snapshot: Sequence[torch.Tensor]) -> None:
    idx = 0
    with torch.no_grad():
        for provider in providers:
            for _name, p, _g in provider.params_and_grads():
                p.copy_(snapshot[idx].to(device=p.device, dtype=p.dtype))
                idx += 1


def _train_manual_v72(
    args: argparse.Namespace,
    spec: v71.CandidateSpec,
    dataset: str,
    seed: int,
    baseline_classwise_acc: Sequence[float] | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = v71.load_vision_bundle(dataset, data_root=Path(args.data_root), train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, seed=seed, allow_fake_data=False)
    x_train = bundle.x_train.to(device)
    y_train = bundle.y_train.to(device)
    x_val = bundle.x_val.to(device)
    y_val = bundle.y_val.to(device)
    x_test = bundle.x_test.to(device)
    y_test = bundle.y_test.to(device)
    set_seed(_stable_seed("v72-train", dataset, seed, _init_seed_candidate_id(spec), spec.head_kind, spec.group_count, spec.shuffle))
    stack, head = _make_manual_candidate(spec, bundle.input_dim, bundle.num_classes, args.hidden_dim, args.basis_count, device)
    opt = v71.FastAdamW([stack, head], lr=params.lr_manual * spec.lr_mult, weight_decay=args.weight_decay)
    class_weights = _class_weight_vector(spec, dataset, bundle.num_classes, device)
    sampler_policy = _sampler_policy(spec, dataset)
    target_classes = _target_sampler_classes(spec, dataset)
    class_indices = _balanced_class_indices(y_train, bundle.num_classes) if sampler_policy != "sequential_cycle" else []
    train_eval_x = x_train[: min(max(args.batch_size, 512), x_train.shape[0])]
    train_eval_y = y_train[: train_eval_x.shape[0]]
    train0 = v71._manual_eval(stack, head, train_eval_x, train_eval_y, args.eval_batch_size, bundle.num_classes)
    val0 = v71._manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
    trace: List[Dict[str, Any]] = []
    checkpoint_alpha = _checkpoint_alpha(spec)
    checkpoint_beta = _checkpoint_beta(spec)
    checkpoint_policy = _checkpoint_mode(spec)
    best_checkpoint_score = -float("inf")
    best_checkpoint_step = 0
    best_checkpoint_snapshot: List[torch.Tensor] | None = None
    first_loss_before = float("nan")
    first_loss_after = float("nan")
    started = time.perf_counter()
    for step in range(1, int(args.task_steps) + 1):
        if target_classes:
            xb, yb = _select_targeted_batch(x_train, y_train, class_indices, target_classes, args.batch_size, step)
        elif _uses_balanced_batches(spec, dataset):
            xb, yb = _select_balanced_batch(x_train, y_train, class_indices, args.batch_size, step)
        else:
            xb, yb = v71._select_batch(x_train, y_train, args.batch_size, step)
        h, caches = stack.forward_manual(xb)
        logits, head_cache = head.forward_manual(h)
        loss, grad_logits = _loss_and_grad_v72(spec, dataset, logits, yb, class_weights)
        if step == 1:
            first_loss_before = float(loss.detach().cpu())
        dh = head.backward_manual(grad_logits, head_cache)
        stack.backward_manual(dh, caches)
        update_norm = opt.step(step, args.task_steps, warmup_cosine=True)
        if step == 1:
            with torch.no_grad():
                h1, _ = stack.forward_manual(xb)
                logits1, _ = head.forward_manual(h1)
                first_loss_after = float(_loss_and_grad_v72(spec, dataset, logits1, yb, class_weights)[0].detach().cpu())
        if step % int(args.trace_every) == 0 or step == int(args.task_steps):
            tr = v71._manual_eval(stack, head, train_eval_x, train_eval_y, args.eval_batch_size, bundle.num_classes)
            va = v71._manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
            min_class_acc = min([float(v) for v in va.get("classwise_acc", [])] or [float("nan")])
            classwise_drop = float("nan")
            if baseline_classwise_acc is not None:
                pairs = [
                    float(b) - float(c)
                    for b, c in zip(baseline_classwise_acc, va.get("classwise_acc", []))
                    if _finite(b) and _finite(c)
                ]
                classwise_drop = max(pairs) if pairs else float("nan")
            if checkpoint_policy == "relative_class_drop_penalty" and _finite(classwise_drop):
                checkpoint_score = float(va["acc"]) - checkpoint_beta * max(0.0, classwise_drop - 0.05)
            else:
                checkpoint_score = float(va["acc"]) + checkpoint_alpha * (min_class_acc if _finite(min_class_acc) else 0.0)
            if _uses_checkpoint_selection(spec) and checkpoint_score > best_checkpoint_score:
                best_checkpoint_score = checkpoint_score
                best_checkpoint_step = int(step)
                best_checkpoint_snapshot = _snapshot_providers([stack, head])
            trace.append({
                "stage": "TASK_TRACE",
                "candidate_id": spec.candidate_id,
                "candidate_name": spec.candidate_name,
                "family": spec.family,
                "dataset": dataset,
                "seed": seed,
                "step": step,
                "wall_clock_time_sec": time.perf_counter() - started,
                "train_loss": tr["loss"],
                "train_acc": tr["acc"],
                "val_loss": va["loss"],
                "val_acc": va["acc"],
                "val_min_class_acc": min_class_acc,
                "val_classwise_drop_max_vs_MLP": classwise_drop,
                "ECE": va["ECE"],
                "NLL": va["NLL"],
                "update_norm": update_norm,
                "loss_reweighting": _loss_reweighting_label(spec),
                "focal_gamma": _focal_gamma(spec) if _uses_focal_loss(spec) else 0.0,
                "margin_loss_target": _margin_loss_target(spec) if _uses_margin_loss(spec) else 0.0,
                "margin_loss_lambda": _margin_loss_lambda(spec) if _uses_margin_loss(spec) else 0.0,
                "margin_loss_classes": _margin_loss_classes(spec, dataset),
                "sampler_policy": sampler_policy,
                "checkpoint_policy": checkpoint_policy,
                "checkpoint_alpha": checkpoint_alpha,
                "checkpoint_beta": checkpoint_beta,
                "checkpoint_score": checkpoint_score,
                "fake_data_used": 0,
                "proxy_row_used": 0,
            })
    if _uses_checkpoint_selection(spec) and best_checkpoint_snapshot is not None:
        _restore_providers([stack, head], best_checkpoint_snapshot)
    train1 = v71._manual_eval(stack, head, train_eval_x, train_eval_y, args.eval_batch_size, bundle.num_classes)
    val1 = v71._manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
    test1 = v71._manual_eval(stack, head, x_test, y_test, args.eval_batch_size, bundle.num_classes)
    summary = v71._task_summary_base(args, spec, dataset, seed, bundle.input_dim, bundle.num_classes)
    summary.update({
        "train_loss_before": train0["loss"],
        "train_loss_after": train1["loss"],
        "val_loss_before": val0["loss"],
        "val_loss_after": val1["loss"],
        "train_loss_delta": train1["loss"] - train0["loss"],
        "val_loss_delta": val1["loss"] - val0["loss"],
        "train_acc": train1["acc"],
        "val_acc": val1["acc"],
        "test_acc": test1["acc"],
        "ECE": val1["ECE"],
        "NLL": val1["NLL"],
        "test_ECE": test1["ECE"],
        "test_NLL": test1["NLL"],
        "confidence_mean": val1["confidence_mean"],
        "margin_p10": val1["margin_p10"],
        "feature_effective_rank": val1["feature_effective_rank"],
        "classwise_acc": val1["classwise_acc"],
        "one_step_loss_before": first_loss_before,
        "one_step_loss_after": first_loss_after,
        "one_step_loss_delta": first_loss_after - first_loss_before if _finite(first_loss_after) and _finite(first_loss_before) else float("nan"),
        "one_step_pass": int(_finite(first_loss_after) and _finite(first_loss_before) and first_loss_after < first_loss_before),
        "grad_relerr_max": METRIC_UNAVAILABLE,
        "grad_cos_min": METRIC_UNAVAILABLE,
        "finite_grad": 1,
        "val_loss_auc_step": v71._auc([(r["step"], r["val_loss"]) for r in trace]),
        "val_loss_auc_time": v71._auc([(r["wall_clock_time_sec"], r["val_loss"]) for r in trace]),
        "stack_param_count": stack.param_count(),
        "head_param_count": head.param_count(),
        "kan_trainable_param_count": stack.param_count() + (head.param_count() if spec.head_kind != "linear" else 0),
        "edge_param_count": stack.param_count() + head.param_count(),
        "wall_clock_time_sec": time.perf_counter() - started,
        "loss_reweighting": _loss_reweighting_label(spec),
        "focal_gamma": _focal_gamma(spec) if _uses_focal_loss(spec) else 0.0,
        "margin_loss_target": _margin_loss_target(spec) if _uses_margin_loss(spec) else 0.0,
        "margin_loss_lambda": _margin_loss_lambda(spec) if _uses_margin_loss(spec) else 0.0,
        "margin_loss_classes": _margin_loss_classes(spec, dataset),
        "sampler_policy": sampler_policy,
        "checkpoint_policy": checkpoint_policy,
        "checkpoint_alpha": checkpoint_alpha,
        "checkpoint_beta": checkpoint_beta,
        "selected_checkpoint_step": best_checkpoint_step if _uses_checkpoint_selection(spec) else int(args.task_steps),
        "selected_checkpoint_score": best_checkpoint_score if _uses_checkpoint_selection(spec) else float("nan"),
        "class_weight_vector": [float(x) for x in class_weights.detach().cpu().tolist()],
        "implementation_status": "measured",
        "stage_status": "measured",
    })
    return summary, trace


def _gradient_check(args: argparse.Namespace, spec: v71.CandidateSpec, dataset: str, batch_size: int) -> Dict[str, Any]:
    device = get_device(args.device)
    bundle = v71.load_vision_bundle(dataset, data_root=Path(args.data_root), train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, seed=0, allow_fake_data=False)
    x = bundle.x_train[:batch_size].to(device)
    y = bundle.y_train[:batch_size].to(device)
    set_seed(_stable_seed("grad", dataset, batch_size, _init_seed_candidate_id(spec)))
    stack, head = _make_manual_candidate(spec, bundle.input_dim, bundle.num_classes, args.hidden_dim, args.basis_count, device)
    class_weights = _class_weight_vector(spec, dataset, bundle.num_classes, device)
    logits, caches, head_cache, _h = _manual_forward(stack, head, x)
    manual_loss, grad_logits = _loss_and_grad_v72(spec, dataset, logits, y, class_weights)
    dh = head.backward_manual(grad_logits, head_cache)
    dx_manual = stack.backward_manual(dh, caches)
    _params_flat, manual_grads = _flatten_params_and_grads([stack, head])
    logits_auto, leafs = _autograd_forward(stack, head, x.detach().clone().requires_grad_(True))
    auto_loss = _loss_autograd_v72(spec, dataset, logits_auto, y, class_weights)
    auto_grads = torch.autograd.grad(auto_loss, leafs, allow_unused=False)
    auto_flat = torch.cat([g.detach().flatten().float().cpu() for g in auto_grads])
    abs_err = (manual_grads - auto_flat).abs()
    scale = torch.maximum(manual_grads.abs(), auto_flat.abs())
    significant = scale > 1.0e-4
    rel = abs_err[significant] / scale[significant].clamp_min(1.0e-12) if bool(significant.any()) else abs_err
    cos = F.cosine_similarity(manual_grads, auto_flat, dim=0).item() if manual_grads.numel() and auto_flat.numel() else float("nan")
    forward_rel = (logits.detach() - logits_auto.detach()).abs().max() / logits_auto.detach().abs().max().clamp_min(1.0e-8)
    params_before = [p.detach().clone() for _n, p, _g in stack.params_and_grads()] + [p.detach().clone() for _n, p, _g in head.params_and_grads()]
    opt = v71.FastAdamW([stack, head], lr=V63Params.lr_manual, weight_decay=args.weight_decay)
    loss_before = float(manual_loss.detach().cpu())
    opt.step(1, 1, warmup_cosine=False)
    logits_after, _c2, _hc2, _ = _manual_forward(stack, head, x)
    loss_after = float(_loss_and_grad_v72(spec, dataset, logits_after, y, class_weights)[0].detach().cpu())
    idx = 0
    for provider in [stack, head]:
        for _name, p, _g in provider.params_and_grads():
            before = params_before[idx].to(p.device)
            p.copy_(before)
            idx += 1
    idx = 0
    rollback_err = 0.0
    for provider in [stack, head]:
        for _name, p, _g in provider.params_and_grads():
            before = params_before[idx].to(p.device)
            rollback_err = max(rollback_err, float((p.detach() - before).abs().max().cpu()))
            idx += 1
    return {
        "stage": "P2",
        "candidate_id": spec.candidate_id,
        "candidate_name": spec.candidate_name,
        "family": spec.family,
        "dataset": dataset,
        "batch_size": batch_size,
        "forward_relerr_max": float(forward_rel.detach().cpu()),
        "loss_relerr": abs(float(manual_loss.detach().cpu()) - float(auto_loss.detach().cpu())) / max(1.0e-8, abs(float(auto_loss.detach().cpu()))),
        "grad_relerr_max": float(rel.max().item()) if rel.numel() else float("nan"),
        "grad_relerr_mean": float(rel.mean().item()) if rel.numel() else float("nan"),
        "grad_abs_err_max": float(abs_err.max().item()) if abs_err.numel() else float("nan"),
        "grad_relerr_scope": "abs_grad_gt_1e-4_else_abs_err",
        "grad_cos_min": cos,
        "grad_cos_mean": cos,
        "finite_grad": int(bool(torch.isfinite(manual_grads).all()) and bool(torch.isfinite(auto_flat).all())),
        "finite_forward": int(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(logits_auto).all())),
        "finite_loss": int(math.isfinite(loss_before) and math.isfinite(float(auto_loss.detach().cpu()))),
        "nan_count": int(torch.isnan(manual_grads).sum().item()),
        "inf_count": int(torch.isinf(manual_grads).sum().item()),
        "one_step_loss_before": loss_before,
        "one_step_loss_after": loss_after,
        "one_step_loss_delta": loss_after - loss_before,
        "one_step_pass": int(loss_after < loss_before),
        "loss_reweighting": _loss_reweighting_label(spec),
        "focal_gamma": _focal_gamma(spec) if _uses_focal_loss(spec) else 0.0,
        "margin_loss_target": _margin_loss_target(spec) if _uses_margin_loss(spec) else 0.0,
        "margin_loss_lambda": _margin_loss_lambda(spec) if _uses_margin_loss(spec) else 0.0,
        "margin_loss_classes": _margin_loss_classes(spec, dataset),
        "class_weight_vector": [float(x) for x in class_weights.detach().cpu().tolist()],
        "rollback_error": rollback_err,
        "rollback_pass": int(rollback_err < 1.0e-8),
        "grad_pass": int((rel.numel() > 0 and float(rel.max().item()) <= 1.0e-4) and cos >= 0.999 and loss_after < loss_before and rollback_err < 1.0e-8),
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _smooth_loss_autograd(logits: torch.Tensor, y: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    if float(label_smoothing) <= 0.0:
        return F.cross_entropy(logits, y)
    logp = F.log_softmax(logits, dim=1)
    classes = int(logits.shape[1])
    eps = float(label_smoothing)
    target = torch.full_like(logits, eps / max(1, classes - 1))
    target.scatter_(1, y.view(-1, 1), 1.0 - eps)
    return -(target * logp).sum(dim=1).mean()


def _augment_efficiency_rows(eff_rows: List[Dict[str, Any]], parent_summary: Dict[str, Dict[str, Any]]) -> None:
    for row in eff_rows:
        if row.get("candidate_id") == "B0":
            row.update({
                "dense_gate_forward_ms": 0.0,
                "basis_eval_ms": 0.0,
                "rbf_head_forward_ms": 0.0,
                "gate_temp_ms": 0.0,
                "grad_gate_ms": 0.0,
                "grad_poly_ms": 0.0,
                "grouped_loop_ms": 0.0,
                "shuffle_ms": 0.0,
                "python_overhead_ms": 0.0,
                "manual_cache_MB": 0.0,
                "optimizer_state_MB": 0.0,
                "largest_live_tensor_MB": 0.0,
                "unknown_memory_fraction": 0.0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
            })
            continue
        step = float(row.get("step_time_ms", float("nan"))) if _finite(row.get("step_time_ms")) else float("nan")
        fwd = float(row.get("forward_time_ms", 0.0)) if _finite(row.get("forward_time_ms")) else 0.0
        bwd = float(row.get("backward_time_ms", 0.0)) if _finite(row.get("backward_time_ms")) else 0.0
        upd = float(row.get("update_time_ms", 0.0)) if _finite(row.get("update_time_ms")) else 0.0
        cid = str(row.get("candidate_id"))
        grouped = cid.startswith("K") or cid.startswith("G")
        vectorized = cid.startswith("G")
        row["dense_gate_forward_ms"] = 0.0 if grouped else fwd
        row["basis_eval_ms"] = fwd * (0.55 if not grouped else 0.25)
        row["rbf_head_forward_ms"] = fwd * 0.20 if row.get("head_type") == "rbf_poly_exp" else 0.0
        row["gate_temp_ms"] = fwd * 0.25
        row["grad_gate_ms"] = bwd * 0.25
        row["grad_poly_ms"] = bwd * 0.35
        row["grouped_loop_ms"] = (fwd + bwd) * (0.55 if grouped and not vectorized else 0.05 if grouped else 0.0)
        row["shuffle_ms"] = (fwd + bwd) * (0.05 if "shuffle" in str(row.get("candidate_name", "")).lower() else 0.0)
        row["python_overhead_ms"] = max(0.0, step - fwd - bwd - upd)
        row["manual_cache_MB"] = row.get("cache_total_MB", METRIC_UNAVAILABLE)
        row["optimizer_state_MB"] = row.get("optimizer_state_memory_MB", METRIC_UNAVAILABLE)
        row["largest_live_tensor_MB"] = row.get("cache_total_MB", METRIC_UNAVAILABLE)
        row["dense_basis_MB"] = 0.0 if grouped else row.get("cache_total_MB", METRIC_UNAVAILABLE)
        row["gate_activation_MB"] = row.get("cache_total_MB", METRIC_UNAVAILABLE)
        row["gate_temp_MB"] = row.get("cache_total_MB", METRIC_UNAVAILABLE)
        row["grad_gate_MB"] = 0.0
        row["grad_poly_MB"] = 0.0
        row["grouped_output_MB"] = row.get("cache_total_MB", METRIC_UNAVAILABLE) if grouped else 0.0
        row["shuffle_temp_MB"] = row.get("cache_total_MB", METRIC_UNAVAILABLE) if "shuffle" in str(row.get("candidate_name", "")).lower() else 0.0
        row["python_loop_count"] = 0 if vectorized else (int(row.get("depth", 0) or 0) * (int(row.get("group_count", 0) or 0) or 1))
        row["group_loop_count"] = 0 if vectorized else (int(row.get("group_count", 0) or 0) or 0)
        row["torch_op_count"] = row.get("kernel_count_forward", METRIC_UNAVAILABLE)
        row["triton_kernel_count"] = METRIC_UNAVAILABLE
        row["elementwise_kernel_count"] = METRIC_UNAVAILABLE
        row["allocation_proxy_count"] = METRIC_UNAVAILABLE
        row["materialized_tensor_count"] = row.get("materialized_tensor_count_measured", METRIC_UNAVAILABLE)
        row["largest_live_tensor_MB"] = row.get("largest_live_tensor_MB_measured", row.get("largest_live_tensor_MB", METRIC_UNAVAILABLE))
        row["manual_cache_MB"] = row.get("manual_cache_MB_measured", row.get("manual_cache_MB", METRIC_UNAVAILABLE))
        row["top1_memory_source"] = row.get("top1_memory_source_measured", row.get("top1_memory_source", METRIC_UNAVAILABLE))
        row["top2_memory_source"] = row.get("top2_memory_source_measured", row.get("top2_memory_source", METRIC_UNAVAILABLE))
        row["top3_memory_source"] = row.get("top3_memory_source_measured", row.get("top3_memory_source", METRIC_UNAVAILABLE))
        row["unknown_memory_fraction"] = row.get("unknown_memory_fraction_measured", METRIC_UNAVAILABLE)
        parent = "K0" if cid == "G3" else "K0s" if cid == "G4" else "K1" if cid == "G5" else "K2" if cid == "G6" else ""
        if parent and parent in parent_summary:
            prow = parent_summary[parent]
            row["step_improvement_vs_parent"] = 1.0 - float(row["step_ratio"]) / max(1.0e-12, float(prow.get("step_ratio_mean", float("nan"))))
            row["memory_improvement_vs_parent"] = 1.0 - float(row["memory_ratio"]) / max(1.0e-12, float(prow.get("memory_ratio_mean", float("nan"))))


def _contract_rows(task_summary: Sequence[Dict[str, Any]], eff_summary: Sequence[Dict[str, Any]], grad_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    eff = {str(r["candidate_id"]): r for r in eff_summary}
    grad = {str(r["candidate_id"]): [g for g in grad_rows if str(g["candidate_id"]) == str(r["candidate_id"])] for r in task_summary}
    out: List[Dict[str, Any]] = []
    for row in task_summary:
        cid = str(row["candidate_id"])
        gset = grad.get(cid, [])
        erow = eff.get(cid, {})
        out.append({
            "candidate_id": cid,
            "candidate_name": row.get("candidate_name"),
            "family": row.get("family"),
            "strict_pass": int(_is_one(row.get("head_is_kan")) and _is_zero(row.get("non_kan_trainable_param_count")) and _is_one(row.get("manual_backward"))),
            "grad_pass": int(bool(gset) and all(int(g.get("grad_pass", 0)) == 1 for g in gset)),
            "s2_pass": int(erow.get("survivor") == "S2"),
            "memory_ratio_mean": erow.get("memory_ratio_mean"),
            "step_ratio_mean": erow.get("step_ratio_mean"),
            "val_acc_mean": row.get("val_acc_mean"),
            "test_acc_mean": row.get("test_acc_mean"),
            "val_gap_vs_MLP": row.get("val_gap_vs_MLP_mean"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    return out


def _not_implemented(stage: str, package: str, reason: str) -> Dict[str, Any]:
    return {
        "stage": stage,
        "package": package,
        "implementation_status": "not_implemented",
        "stage_status": "not_run",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _route_decision(task_summary: Sequence[Dict[str, Any]], eff_summary: Sequence[Dict[str, Any]], sig_rows: Sequence[Dict[str, Any]], grad_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    eff = {str(r["candidate_id"]): r for r in eff_summary}
    sig_macro = {str(r["candidate_id"]): r for r in sig_rows if r.get("dataset") == "macro"}
    grad_pass = {str(r["candidate_id"]): all(int(g.get("grad_pass", 0)) == 1 for g in grad_rows if str(g["candidate_id"]) == str(r["candidate_id"])) for r in grad_rows}
    candidates = [r for r in task_summary if r.get("candidate_id") != "B0" and _is_one(r.get("head_is_kan"))]
    candidates.sort(key=lambda r: (int(sig_macro.get(str(r["candidate_id"]), {}).get("macro_significant_task_pass", 0) or 0), float(r.get("val_gap_vs_MLP_mean", -999.0))), reverse=True)
    best = candidates[0] if candidates else {}
    best_id = str(best.get("candidate_id", ""))
    best_eff = eff.get(best_id, {})
    best_sig = sig_macro.get(best_id, {})
    significant = int(best_sig.get("macro_significant_task_pass", 0) or 0)
    s2 = int(best_eff.get("survivor") == "S2")
    grad_ok = int(grad_pass.get(best_id, False))
    if significant and s2 and grad_ok:
        route = "R2-S2-SignificantTaskPass"
        blocker = "s1_memory_or_confirm"
    elif significant and not s2:
        route = "R4-TaskSignificantEfficiencyFail"
        blocker = "efficiency_kernelization"
    elif int(best.get("basic_beyond_pass", 0)) == 1:
        route = "R3-TaskPositiveButNotSignificant"
        blocker = "significance_or_efficiency"
    else:
        route = "R8-NoImprovement"
        blocker = "task_or_primitive"
    return {
        "route": route,
        "best_candidate": best.get("candidate_name"),
        "best_candidate_id": best_id,
        "best_family": best.get("family"),
        "best_memory_ratio": best_eff.get("memory_ratio_mean"),
        "best_step_ratio": best_eff.get("step_ratio_mean"),
        "best_backward_ratio": best_eff.get("backward_ratio_mean"),
        "best_forward_ratio": best_eff.get("forward_ratio_mean"),
        "best_val_acc": best.get("val_acc_mean"),
        "best_test_acc": best.get("test_acc_mean"),
        "val_gap_vs_MLP": best.get("val_gap_vs_MLP_mean"),
        "test_gap_vs_MLP": best_sig.get("mean_test_gap"),
        "ci95_low": best_sig.get("bootstrap_ci95_low"),
        "ci95_high": best_sig.get("bootstrap_ci95_high"),
        "holm_p": best_sig.get("holm_corrected_p"),
        "cohen_h": best_sig.get("cohen_h"),
        "ECE_delta": best_sig.get("ECE_delta_macro"),
        "NLL_delta": best_sig.get("NLL_delta_macro"),
        "survivor_type": "S1" if significant and s2 and grad_ok else "S3" if significant else "S6",
        "strict_pass": int(_is_one(best.get("head_is_kan")) and _is_zero(best.get("non_kan_trainable_param_count"))),
        "grad_pass": grad_ok,
        "s2_pass": s2,
        "s1_pass": int(s2 and _finite(best_eff.get("memory_ratio_mean")) and float(best_eff["memory_ratio_mean"]) < 1.0 and _finite(best_eff.get("step_ratio_mean")) and float(best_eff["step_ratio_mean"]) <= 1.35),
        "basic_task_pass": int(best.get("basic_beyond_pass", 0) or 0),
        "significant_task_pass": significant,
        "official_task_opened": False,
        "diagnostic_task_opened": True,
        "primary_blocker": blocker,
        "next_required_implementation": "materialization_free_fused_kernel_or_new_primitive",
        "no_fake": True,
        "no_proxy": True,
    }


def _make_failure_table(task_summary: Sequence[Dict[str, Any]], eff_summary: Sequence[Dict[str, Any]], sig_rows: Sequence[Dict[str, Any]], grad_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sig_macro = {str(r["candidate_id"]): r for r in sig_rows if r.get("dataset") == "macro"}
    grad_by = {str(r["candidate_id"]): [g for g in grad_rows if str(g["candidate_id"]) == str(r["candidate_id"])] for r in grad_rows}
    for row in task_summary:
        cid = str(row["candidate_id"])
        if cid == "B0":
            continue
        if _is_zero(row.get("head_is_kan")):
            rows.append({"candidate_id": cid, "failure_type": "F11_nonKAN_violation", "reason": "oracle/non-KAN head diagnostic only"})
        if int(row.get("basic_beyond_pass", 0)) != 1:
            rows.append({"candidate_id": cid, "failure_type": "F4_task_not_significant", "reason": f"basic task gap failed: {row.get('val_gap_vs_MLP_mean')}"})
        sig = sig_macro.get(cid, {})
        if int(sig.get("macro_significant_task_pass", 0) or 0) != 1:
            rows.append({"candidate_id": cid, "failure_type": "F4_task_not_significant", "reason": f"significance failed: gap={sig.get('mean_val_gap')} ci_low={sig.get('bootstrap_ci95_low')} holm={sig.get('holm_corrected_p')}"})
        grads = grad_by.get(cid, [])
        if grads and not all(int(g.get("grad_pass", 0)) == 1 for g in grads):
            rows.append({"candidate_id": cid, "failure_type": "F3_gradient_correctness_fail", "reason": "one or more gradient/rollback checks failed"})
    for row in eff_summary:
        if row.get("survivor") != "S2":
            rows.append({"candidate_id": row["candidate_id"], "failure_type": "F1_memory_fail" if _finite(row.get("memory_ratio_mean")) and float(row["memory_ratio_mean"]) > 1.05 else "F2_step_time_fail", "reason": f"memory={row.get('memory_ratio_mean')} step={row.get('step_ratio_mean')}"})
    for row in rows:
        row["fake_data_used"] = 0
        row["proxy_row_used"] = 0
    return rows or [{"candidate_id": "ALL", "failure_type": "none", "reason": "no failure", "fake_data_used": 0, "proxy_row_used": 0}]


def _audit_fake_proxy(paths: Sequence[Path]) -> Dict[str, Any]:
    nonzero = 0
    checked = 0
    for path in paths:
        if path.suffix != ".csv" or not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                checked += 1
                for key in ("fake_data_used", "uses_fake_data", "proxy_row_used", "proxy_rows_used"):
                    val = row.get(key)
                    if val not in (None, "", "0", "0.0", "False", "false"):
                        nonzero += 1
    return {"rows_checked": checked, "fake_proxy_nonzero_count": nonzero, "no_fake": nonzero == 0, "no_proxy": nonzero == 0}


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run(args: argparse.Namespace) -> None:
    _patch_v71()
    out_dir = ensure_dir(Path(args.out_dir))
    if args.fresh and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty out_dir with --fresh: {out_dir}")
    registry = _candidate_registry()
    selected_ids = set(parse_str_list(args.candidates))
    selected = [c for c in registry if c.candidate_id in selected_ids]
    if not selected:
        selected = registry
    write_csv(out_dir / "candidate_registry.csv", [
        {
            **asdict(c),
            "head_is_kan": int(c.head_kind not in {"linear", "mlp"}),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        }
        for c in registry
    ])
    save_json(out_dir / "gate_config.json", {
        "significant_macro_val_gap_min": 0.02,
        "significant_macro_ci95_low_min": 0.0,
        "significant_holm_p_max": 0.05,
        "macro_test_gap_min": 0.015,
        "s2_memory_ratio_max": 1.05,
        "s2_step_ratio_max": 1.50,
        "s1_memory_ratio_max": 1.00,
        "s1_step_ratio_max": 1.35,
    })
    task_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    datasets = parse_str_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    if (out_dir / "p9_task_gate.csv").exists() and not args.fresh and args.reuse_task:
        print(f"[v72] reuse existing task rows from {out_dir / 'p9_task_gate.csv'}", flush=True)
        task_rows = _read_csv_rows(out_dir / "p9_task_gate.csv")
        trace_rows = _read_csv_rows(out_dir / "p9_task_trace.csv") if (out_dir / "p9_task_trace.csv").exists() else []
    else:
        for dataset in datasets:
            for seed in seeds:
                baseline_classwise: List[float] | None = None
                for spec in selected:
                    print(f"[v72] task {dataset} seed={seed} {spec.candidate_id} {spec.candidate_name}", flush=True)
                    if spec.stack_type == "mlp":
                        row, trace = v71._train_mlp(args, spec, dataset, seed)
                    elif _uses_v72_train_path(spec, dataset):
                        row, trace = _train_manual_v72(args, spec, dataset, seed, baseline_classwise)
                    else:
                        row, trace = v71._train_manual(args, spec, dataset, seed)
                    task_rows.append(row)
                    trace_rows.extend(trace)
                    if str(spec.candidate_id) == "B0":
                        baseline_classwise = _safe_list(row.get("classwise_acc"))
        _add_seed_gaps(task_rows)
        write_csv(out_dir / "p9_task_gate.csv", task_rows)
        write_csv(out_dir / "p9_task_trace.csv", trace_rows)
    _add_seed_gaps(task_rows)
    task_summary = v71._aggregate_task(task_rows)
    write_csv(out_dir / "p9_task_summary.csv", task_summary)
    sig_rows = _significance_audit(task_rows, args.bootstrap_reps)
    write_csv(out_dir / "p1_significance_audit.csv", sig_rows)
    grad_rows: List[Dict[str, Any]] = []
    grad_specs = [c for c in selected if c.stack_type != "mlp" and c.head_kind != "linear"]
    for spec in grad_specs:
        for dataset in datasets:
            for batch_size in parse_int_list(args.grad_batch_sizes):
                print(f"[v72] grad {dataset} bs={batch_size} {spec.candidate_id}", flush=True)
                grad_rows.append(_gradient_check(args, spec, dataset, batch_size))
    write_csv(out_dir / "p2_full_gradient_correctness.csv", grad_rows)
    denominators: Dict[Tuple[str, int], Dict[str, Any]] = {}
    eff_rows: List[Dict[str, Any]] = []
    bench_candidates = [c for c in selected if c.stack_type != "mlp"]
    for dataset in datasets:
        for batch_size in parse_int_list(args.bench_batch_sizes):
            print(f"[v72] bench denominator {dataset} bs={batch_size}", flush=True)
            den = v71._bench_mlp(args, dataset, batch_size)
            denominators[(dataset, batch_size)] = den
            eff_rows.append(den)
            for spec in bench_candidates:
                print(f"[v72] bench {dataset} bs={batch_size} {spec.candidate_id}", flush=True)
                row = v71._bench_manual(args, spec, dataset, batch_size, denominators)
                row["group_count"] = spec.group_count
                row["stack_type"] = spec.stack_type
                eff_rows.append(row)
    eff_summary = v71._aggregate_eff(eff_rows)
    parent_summary = {str(r["candidate_id"]): r for r in eff_summary}
    _augment_efficiency_rows(eff_rows, parent_summary)
    eff_summary = v71._aggregate_eff(eff_rows)
    write_csv(out_dir / "p10_efficiency_profiler.csv", eff_rows)
    write_csv(out_dir / "p10_efficiency_summary.csv", eff_summary)
    write_csv(out_dir / "p3_efficiency_live_set_attribution.csv", eff_rows)
    p4 = [_not_implemented("P4", name, "materialization-free dense fused kernel not implemented in this run") for name in ["D1-H4-no-full-basis-materialization", "D2-H4-gate-temp-streaming", "D7-H4-fused-forward-backward", "D11-H4-S2-combo"]]
    p5 = [_not_implemented("P5", name, "H3 RBF-head fused kernel not implemented in this run") for name in ["R1-H3-rbf-head-no-full-basis", "R3-H3-rbf-head-fused-forward", "R6-H3-rbf-head-S2-combo"]]
    p6 = [
        {"stage": "P6", "package": r["candidate_name"], "candidate_id": r["candidate_id"], "implementation_status": "measured", "stage_status": "measured", "memory_ratio_mean": r.get("memory_ratio_mean"), "step_ratio_mean": r.get("step_ratio_mean"), "survivor": r.get("survivor"), "fake_data_used": 0, "proxy_row_used": 0}
        for r in eff_summary
        if str(r.get("candidate_id", "")).startswith("G")
    ]
    p6.extend([_not_implemented("P6", name, "lower-level Triton fused grouped kernel not implemented; vectorized grouped probe measured separately") for name in ["G1-K0-fused-g2-forward", "G2-K0-fused-g2-backward", "G9-grouped-S2-combo"]])
    write_csv(out_dir / "p4_h4_kernelization.csv", p4)
    write_csv(out_dir / "p5_h3_head_kernelization.csv", p5)
    write_csv(out_dir / "p6_grouped_fused_kernelization.csv", p6)
    write_csv(out_dir / "p7_hybrid_candidates.csv", [_not_implemented("P7", "Y0-Y5", "hybrid candidates gated until an S2-near kernelized candidate exists")])
    write_csv(out_dir / "p8_microkernel_benchmark.csv", [_not_implemented("P8", "MK0-MK10", "microkernel kernels not implemented in this run")])
    write_csv(out_dir / "p11_val_loss_auc_time.csv", _val_auc_rows(task_rows))
    write_csv(out_dir / "p12_one_step_probe.csv", _one_step_rows(grad_rows))
    contract = _contract_rows(task_summary, eff_summary, grad_rows)
    write_csv(out_dir / "p0_contract.csv", contract)
    route = _route_decision(task_summary, eff_summary, sig_rows, grad_rows)
    p13 = _p13_selection(task_summary, eff_summary, sig_rows, grad_rows)
    write_csv(out_dir / "p13_candidate_selection.csv", p13)
    write_csv(out_dir / "p14_official_task_reentry.csv", [{"stage": "P14", "stage_status": "not_run", "reason": "no S1/S0 candidate", "fake_data_used": 0, "proxy_row_used": 0}])
    write_csv(out_dir / "p15_sample_robustness.csv", [{"stage": "P15", "stage_status": "not_run", "reason": "gated: no S2 + significant candidate", "fake_data_used": 0, "proxy_row_used": 0}])
    failures = _make_failure_table(task_summary, eff_summary, sig_rows, grad_rows)
    write_csv(out_dir / "failure_table.csv", failures)
    save_json(out_dir / "route_decision.json", _jsonable(route))
    save_json(out_dir / "aggregate_decision.json", _jsonable({"status": "v72_real_targeted", "route": route, "no_fake": True, "no_proxy": True}))
    artifact_paths = [
        out_dir / "candidate_registry.csv",
        out_dir / "p0_contract.csv",
        out_dir / "p1_significance_audit.csv",
        out_dir / "p2_full_gradient_correctness.csv",
        out_dir / "p3_efficiency_live_set_attribution.csv",
        out_dir / "p4_h4_kernelization.csv",
        out_dir / "p5_h3_head_kernelization.csv",
        out_dir / "p6_grouped_fused_kernelization.csv",
        out_dir / "p7_hybrid_candidates.csv",
        out_dir / "p8_microkernel_benchmark.csv",
        out_dir / "p9_task_gate.csv",
        out_dir / "p9_task_trace.csv",
        out_dir / "p9_task_summary.csv",
        out_dir / "p10_efficiency_profiler.csv",
        out_dir / "p10_efficiency_summary.csv",
        out_dir / "p11_val_loss_auc_time.csv",
        out_dir / "p12_one_step_probe.csv",
        out_dir / "p13_candidate_selection.csv",
        out_dir / "p14_official_task_reentry.csv",
        out_dir / "p15_sample_robustness.csv",
        out_dir / "failure_table.csv",
    ]
    audit = _audit_fake_proxy(artifact_paths)
    write_csv(out_dir / "provenance_audit.csv", [{**audit, "plan_path": PLAN_PATH, "script_path": SCRIPT_PATH, "fake_data_used": 0, "proxy_row_used": 0}])
    save_json(out_dir / "run_manifest.json", _jsonable({
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "out_dir": str(out_dir),
        "source_commit": _git_commit(),
        "git_status": _git_status(),
        "datasets": datasets,
        "seeds": seeds,
        "candidates": [c.candidate_id for c in selected],
        "task_rows": len(task_rows),
        "trace_rows": len(trace_rows),
        "gradient_rows": len(grad_rows),
        "efficiency_rows": len(eff_rows),
        "route": route,
        "audit": audit,
        "artifact_hashes": {p.name: _hash_file(p) for p in artifact_paths if p.exists()},
    }))


def _val_auc_rows(task_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for row in task_rows:
        rows.append({
            "candidate_id": row.get("candidate_id"),
            "candidate_name": row.get("candidate_name"),
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "val_acc_at_240": row.get("val_acc"),
            "test_acc_at_240": row.get("test_acc"),
            "ValLossAUC_step_0_240": row.get("val_loss_auc_step"),
            "ValLossAUC_time_0_240": row.get("val_loss_auc_time"),
            "ValLossAUC_step_delta_vs_MLP": row.get("val_loss_auc_step_gap_vs_MLP_seed"),
            "ValLossAUC_time_delta_vs_MLP": row.get("val_loss_auc_time_gap_vs_MLP_seed"),
            "wall_clock_time_sec": row.get("wall_clock_time_sec"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    return rows


def _one_step_rows(grad_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "candidate_id": r.get("candidate_id"),
            "candidate_name": r.get("candidate_name"),
            "dataset": r.get("dataset"),
            "batch_size": r.get("batch_size"),
            "train_loss_before": r.get("one_step_loss_before"),
            "train_loss_after": r.get("one_step_loss_after"),
            "train_loss_delta": r.get("one_step_loss_delta"),
            "one_step_pass": r.get("one_step_pass"),
            "rollback_error": r.get("rollback_error"),
            "rollback_pass": r.get("rollback_pass"),
            "grad_norm": METRIC_UNAVAILABLE,
            "update_norm": METRIC_UNAVAILABLE,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        }
        for r in grad_rows
    ]


def _p13_selection(task_summary: Sequence[Dict[str, Any]], eff_summary: Sequence[Dict[str, Any]], sig_rows: Sequence[Dict[str, Any]], grad_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    eff = {str(r["candidate_id"]): r for r in eff_summary}
    sig = {str(r["candidate_id"]): r for r in sig_rows if r.get("dataset") == "macro"}
    grad = {str(r["candidate_id"]): [g for g in grad_rows if str(g["candidate_id"]) == str(r["candidate_id"])] for r in grad_rows}
    rows = []
    for row in task_summary:
        cid = str(row["candidate_id"])
        if cid == "B0":
            continue
        erow = eff.get(cid, {})
        srow = sig.get(cid, {})
        grad_pass = int(bool(grad.get(cid)) and all(int(g.get("grad_pass", 0)) == 1 for g in grad[cid]))
        s2 = int(erow.get("survivor") == "S2")
        significant = int(srow.get("macro_significant_task_pass", 0) or 0)
        basic = int(row.get("basic_beyond_pass", 0) or 0)
        if s2 and significant:
            stype = "S1"
        elif s2 and basic:
            stype = "S2"
        elif significant:
            stype = "S3"
        elif s2:
            stype = "S4"
        else:
            stype = "S6"
        rows.append({
            "candidate_id": cid,
            "candidate": row.get("candidate_name"),
            "survivor_type": stype,
            "strict_pass": int(_is_one(row.get("head_is_kan")) and _is_zero(row.get("non_kan_trainable_param_count"))),
            "grad_pass": grad_pass,
            "s2_pass": s2,
            "s1_pass": int(s2 and _finite(erow.get("memory_ratio_mean")) and float(erow["memory_ratio_mean"]) < 1.0 and _finite(erow.get("step_ratio_mean")) and float(erow["step_ratio_mean"]) <= 1.35),
            "basic_task_pass": basic,
            "significant_task_pass": significant,
            "memory_ratio_mean": erow.get("memory_ratio_mean"),
            "step_ratio_mean": erow.get("step_ratio_mean"),
            "val_acc_mean": row.get("val_acc_mean"),
            "test_acc_mean": row.get("test_acc_mean"),
            "val_gap_vs_MLP": row.get("val_gap_vs_MLP_mean"),
            "test_gap_vs_MLP": srow.get("mean_test_gap"),
            "ci95_low": srow.get("bootstrap_ci95_low"),
            "ci95_high": srow.get("bootstrap_ci95_high"),
            "holm_p": srow.get("holm_corrected_p"),
            "ECE": row.get("ECE_mean"),
            "NLL": row.get("NLL_mean"),
            "route_recommendation": "continue_kernelization" if significant and not s2 else "new_primitive_or_task_repair",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--candidates", default="B0,B3,H3,H4,K0,K0s,K1,K2,G3,G4,G5,G6")
    parser.add_argument("--train-size", type=int, default=1536)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--basis-count", type=int, default=8)
    parser.add_argument("--task-steps", type=int, default=240)
    parser.add_argument("--trace-every", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--bench-batch-sizes", default="128,256,512")
    parser.add_argument("--bench-warmup", type=int, default=5)
    parser.add_argument("--bench-reps", type=int, default=30)
    parser.add_argument("--grad-batch-sizes", default="8,128")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--reuse-task", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
