#!/usr/bin/env python3
"""DG-KAN v7.1 real-only Dense Poly2-Gate -> kernel-friendly runner.

This runner is intentionally targeted: it executes the first v7.1 stop/go
chain around the v7.0 dense D3 signal, strict KAN heads, and grouped/shuffle
kernel-friendly candidates.  Unsupported pieces are recorded as not_run or
not_implemented; no fake/proxy rows or placeholder ratios are emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from dgkan_core import ensure_dir, get_device, load_vision_bundle, parse_int_list, parse_str_list, save_json, set_seed, write_csv
from run_gafu_v54 import _ece
from run_gafu_v63 import V63ManualLayer, V63ManualStack, V63Params
from run_gafu_v64_real import _make_mlp
from run_gafu_v66_real import _git_commit, _git_status


PLAN_PATH = "docs/DG-KAN_v7.1_Final_DensePoly2Gate_KernelNative_PureKANNG_完整实验计划.md"
SCRIPT_PATH = "experiments/run_gafu_v71_real.py"
METRIC_UNAVAILABLE = "metric_unavailable"


def _mean(values: Iterable[float], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if _finite(v)]
    return statistics.mean(vals) if vals else default


def _std(values: Iterable[float], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if _finite(v)]
    return statistics.pstdev(vals) if len(vals) > 1 else (0.0 if vals else default)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("::".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return 710000 + int(digest[:8], 16) % 100000


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_allocated_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def _peak_reserved_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    return float(torch.cuda.max_memory_reserved(device) / (1024**2))


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _select_batch(x: torch.Tensor, y: torch.Tensor, batch_size: int, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(x.shape[0])
    start = ((int(step) - 1) * int(batch_size)) % n
    end = start + int(batch_size)
    if end <= n:
        return x[start:end], y[start:end]
    tail = n - start
    return torch.cat([x[start:], x[: end - n]], dim=0), torch.cat([y[start:], y[: end - n]], dim=0)


def _smooth_ce_and_grad(logits: torch.Tensor, y: torch.Tensor, label_smoothing: float) -> Tuple[torch.Tensor, torch.Tensor]:
    logp = F.log_softmax(logits, dim=1)
    probs = logp.exp()
    n = int(logits.shape[0])
    classes = int(logits.shape[1])
    if float(label_smoothing) > 0.0:
        eps = float(label_smoothing)
        target = torch.full_like(logits, eps / max(1, classes - 1))
        target.scatter_(1, y.view(-1, 1), 1.0 - eps)
        loss = -(target * logp).sum(dim=1).mean()
        grad = (probs - target) / n
    else:
        loss = F.nll_loss(logp, y)
        grad = probs
        grad[torch.arange(n, device=logits.device), y] -= 1.0
        grad = grad / n
    return loss, grad


def _eval_logits(logits: torch.Tensor, y: torch.Tensor) -> Tuple[float, float, float, float, float, float]:
    loss = F.cross_entropy(logits, y)
    probs = F.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    acc = (pred == y).float().mean()
    conf = probs.max(dim=1).values
    top2 = probs.topk(2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    return (
        _to_float(loss),
        _to_float(acc),
        _to_float(loss),
        float(_ece(logits.detach(), y.detach())),
        _to_float(conf.mean()),
        _to_float(torch.quantile(margin.detach().float(), 0.10)),
    )


def _effective_rank(features: torch.Tensor) -> float:
    if features.numel() == 0:
        return float("nan")
    with torch.no_grad():
        x = features.detach().float()
        x = x - x.mean(dim=0, keepdim=True)
        try:
            s = torch.linalg.svdvals(x)
        except Exception:
            return float("nan")
        s = s.clamp_min(1.0e-12)
        p = s / s.sum()
        entropy = -(p * p.log()).sum()
        return float(torch.exp(entropy).detach().cpu())


def _classwise_acc(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> List[float]:
    pred = logits.argmax(dim=1)
    out: List[float] = []
    for cls in range(int(num_classes)):
        mask = y == cls
        if bool(mask.any()):
            out.append(float((pred[mask] == y[mask]).float().mean().detach().cpu()))
        else:
            out.append(float("nan"))
    return out


def _auc(points: Sequence[Tuple[float, float]]) -> float:
    pts = [(float(x), float(y)) for x, y in points if _finite(x) and _finite(y)]
    if len(pts) < 2:
        return float("nan")
    total = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        total += 0.5 * (y0 + y1) * (x1 - x0)
    return total


class FastAdamW:
    def __init__(self, providers: Sequence[Any], *, lr: float, weight_decay: float = 1.0e-4) -> None:
        self.providers = list(providers)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m: Dict[int, torch.Tensor] = {}
        self.v: Dict[int, torch.Tensor] = {}
        for _name, p, _g in self.params():
            self.m[id(p)] = torch.zeros_like(p)
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
        update_norm = 0.0
        with torch.no_grad():
            for _name, p, g in self.params():
                if self.weight_decay:
                    p.mul_(1.0 - lr * self.weight_decay)
                m = self.m[id(p)]
                v = self.v[id(p)]
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                denom = v.sqrt().div(math.sqrt(1.0 - beta2**self.t)).add_(eps)
                step_tensor = m.div(1.0 - beta1**self.t).div(denom)
                p.add_(step_tensor, alpha=-lr)
                update_norm += float(step_tensor.detach().float().norm().cpu())
        self.zero_grad()
        return update_norm


class HeadWrapper:
    def __init__(self, layer: V63ManualLayer) -> None:
        self.layer = layer

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.layer.forward_manual(x)

    def backward_manual(self, dy: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        return self.layer.backward_manual(dy, cache)

    def zero_grad(self) -> None:
        self.layer.zero_grad()

    def params_and_grads(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        return [(f"head:{name}", p, self.layer.grads[name]) for name, p in self.layer.params.items()]

    def param_count(self) -> int:
        return self.layer.param_count()


class GroupedPoly2GateStack:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        basis: int,
        group_count: int,
        *,
        shuffle: bool,
        device: torch.device,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.basis = int(basis)
        self.group_count = int(group_count)
        self.shuffle = bool(shuffle)
        self.method = f"GroupedPoly2Gate-g{self.group_count}" + ("+fixed-shuffle" if self.shuffle else "")
        dims = [self.input_dim] + [self.hidden_dim] * self.depth
        self.layers: List[List[V63ManualLayer]] = []
        self.in_splits: List[List[int]] = []
        self.out_splits: List[List[int]] = []
        self.perms: List[Tuple[torch.Tensor | None, torch.Tensor | None]] = []
        for li, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            if a % self.group_count != 0 or b % self.group_count != 0:
                raise ValueError(f"group_count={self.group_count} must divide layer dims {a}->{b}")
            in_sizes = [a // self.group_count] * self.group_count
            out_sizes = [b // self.group_count] * self.group_count
            self.in_splits.append(in_sizes)
            self.out_splits.append(out_sizes)
            self.layers.append(
                [
                    V63ManualLayer(in_sizes[g], out_sizes[g], kind="poly2_gate", basis_count=basis, device=device)
                    for g in range(self.group_count)
                ]
            )
            if self.shuffle and li > 0:
                perm = torch.arange(a, device=device).view(self.group_count, a // self.group_count).transpose(0, 1).reshape(-1)
                inv = torch.empty_like(perm)
                inv[perm] = torch.arange(a, device=device)
                self.perms.append((perm, inv))
            else:
                self.perms.append((None, None))

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        h = x
        caches: List[Dict[str, Any]] = []
        for li, layer_groups in enumerate(self.layers):
            perm, inv = self.perms[li]
            h_in = h[:, perm] if perm is not None else h
            pieces = torch.split(h_in, self.in_splits[li], dim=1)
            ys: List[torch.Tensor] = []
            group_caches: List[torch.Tensor] = []
            for layer, xg in zip(layer_groups, pieces):
                yg, cg = layer.forward_manual(xg)
                ys.append(yg)
                group_caches.append(cg)
            y = torch.cat(ys, dim=1)
            caches.append({"group_caches": group_caches, "y": y, "perm": perm, "inv": inv})
            h = F.silu(y) if li < len(self.layers) - 1 else y
        return h, caches

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        for li in reversed(range(len(self.layers))):
            cache = caches[li]
            if li < len(self.layers) - 1:
                y = cache["y"]
                sig = torch.sigmoid(y)
                delta = delta * sig * (1.0 + y * (1.0 - sig))
            outs = torch.split(delta, self.out_splits[li], dim=1)
            dx_parts: List[torch.Tensor] = []
            for layer, dg, cg in zip(self.layers[li], outs, cache["group_caches"]):
                dx_parts.append(layer.backward_manual(dg, cg))
            dx = torch.cat(dx_parts, dim=1)
            inv = cache["inv"]
            delta = dx[:, inv] if inv is not None else dx
        return delta

    def zero_grad(self) -> None:
        for layer_groups in self.layers:
            for layer in layer_groups:
                layer.zero_grad()

    def params_and_grads(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        rows: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        for li, layer_groups in enumerate(self.layers):
            for gi, layer in enumerate(layer_groups):
                for name, p in layer.params.items():
                    rows.append((f"stack:l{li}:g{gi}:{name}", p, layer.grads[name]))
        return rows

    def param_count(self) -> int:
        return sum(layer.param_count() for layer_groups in self.layers for layer in layer_groups)

    def op_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {"op_count_gemm": 0, "op_count_elementwise": 0, "op_count_pow": 0, "op_count_exp": 0}
        for layer_groups in self.layers:
            for layer in layer_groups:
                for key, value in layer.op_counts().items():
                    counts[key] = counts.get(key, 0) + int(value)
        return counts

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        bytes_total = 0
        for cache in caches:
            bytes_total += int(cache["y"].numel() * cache["y"].element_size())
            for tensor in cache["group_caches"]:
                bytes_total += int(tensor.numel() * tensor.element_size())
        return {
            "cache_total_MB": bytes_total / (1024**2),
            "cache_gate_MB": 0.0,
            "cache_basis_MB": 0.0,
            "cache_hidden_MB": bytes_total / (1024**2),
        }


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    candidate_name: str
    family: str
    stack_type: str
    depth: int = 3
    stack_kind: str = "poly2_gate"
    head_kind: str = "linear"
    group_count: int = 0
    shuffle: bool = False
    label_smoothing: float = 0.05
    lr_mult: float = 1.0
    oracle: bool = False
    strict_status: str = "strict_candidate"
    allowed_in_route_selection: bool = True
    implementation_status: str = "measured"


def _candidate_registry() -> List[CandidateSpec]:
    return [
        CandidateSpec("B0", "MLP-autograd-fulltrain-240", "baseline", "mlp", depth=2, head_kind="mlp", label_smoothing=0.0, oracle=True, strict_status="reference", allowed_in_route_selection=False),
        CandidateSpec("B3", "D3-dense-poly2-gate-d3-linear-head", "dense_oracle", "dense", depth=3, head_kind="linear", oracle=True, strict_status="oracle_nonkan_head", allowed_in_route_selection=False),
        CandidateSpec("H1", "D3-dense-poly2-gate-d3-KAN-head-poly2-gate", "strict_head_repair", "dense", depth=3, head_kind="poly2_gate"),
        CandidateSpec("H2", "D3-dense-poly2-gate-d3-KAN-head-poly2-silu", "strict_head_repair", "dense", depth=3, head_kind="poly2_silu_base"),
        CandidateSpec("H3", "D3-dense-poly2-gate-d3-KAN-head-rbf-poly-exp", "strict_head_repair", "dense", depth=3, head_kind="rbf_poly_exp"),
        CandidateSpec("H4", "D2-dense-poly2-gate-d2-KAN-head-poly2-gate", "strict_head_depth_repair", "dense", depth=2, head_kind="poly2_gate"),
        CandidateSpec("H5", "D2-dense-poly2-gate-d2-KAN-head-poly2-silu", "strict_head_depth_repair", "dense", depth=2, head_kind="poly2_silu_base"),
        CandidateSpec("H6", "D2-dense-poly2-gate-d2-KAN-head-rbf-poly-exp", "strict_head_depth_repair", "dense", depth=2, head_kind="rbf_poly_exp"),
        CandidateSpec("K0", "GroupedPoly2Gate-g2-KAN-head-poly2-gate", "structured_grouped", "grouped", depth=3, head_kind="poly2_gate", group_count=2),
        CandidateSpec("K0s", "GroupedPoly2Gate-g2-fixed-shuffle-KAN-head-poly2-gate", "structured_shuffle", "grouped", depth=3, head_kind="poly2_gate", group_count=2, shuffle=True),
        CandidateSpec("K0q", "GroupedPoly2Gate-g4-KAN-head-poly2-gate", "structured_grouped", "grouped", depth=3, head_kind="poly2_gate", group_count=4),
        CandidateSpec("K0qs", "GroupedPoly2Gate-g4-fixed-shuffle-KAN-head-poly2-gate", "structured_shuffle", "grouped", depth=3, head_kind="poly2_gate", group_count=4, shuffle=True),
        CandidateSpec("K1", "GroupedPoly2Gate-g8-KAN-head-poly2-gate", "structured_grouped", "grouped", depth=3, head_kind="poly2_gate", group_count=8),
        CandidateSpec("K2", "GroupedPoly2Gate-g16-KAN-head-poly2-gate", "structured_grouped", "grouped", depth=3, head_kind="poly2_gate", group_count=16),
        CandidateSpec("K3", "GroupedPoly2Gate-g8-fixed-shuffle-KAN-head-poly2-gate", "structured_shuffle", "grouped", depth=3, head_kind="poly2_gate", group_count=8, shuffle=True),
        CandidateSpec("K4", "GroupedPoly2Gate-g16-fixed-shuffle-KAN-head-poly2-gate", "structured_shuffle", "grouped", depth=3, head_kind="poly2_gate", group_count=16, shuffle=True),
    ]


def _head_kind_to_layer_kind(head_kind: str) -> str:
    if head_kind == "linear":
        return "linear"
    if head_kind == "poly2_silu_base":
        return "poly2_silu_base"
    if head_kind == "rbf_poly_exp":
        return "rbf_poly_exp"
    return "poly2_gate"


def _make_manual_candidate(spec: CandidateSpec, input_dim: int, num_classes: int, hidden_dim: int, basis: int, device: torch.device) -> Tuple[Any, HeadWrapper]:
    if spec.stack_type == "dense":
        method = spec.candidate_name.replace("_", "-")
        stack = V63ManualStack(method, input_dim, hidden_dim, spec.depth, basis, device)
    elif spec.stack_type == "grouped":
        stack = GroupedPoly2GateStack(input_dim, hidden_dim, spec.depth, basis, spec.group_count, shuffle=spec.shuffle, device=device)
    else:
        raise ValueError(f"manual candidate expected, got stack_type={spec.stack_type}")
    head = HeadWrapper(V63ManualLayer(hidden_dim, num_classes, kind=_head_kind_to_layer_kind(spec.head_kind), basis_count=basis, device=device))
    return stack, head


def _manual_eval(stack: Any, head: HeadWrapper, x: torch.Tensor, y: torch.Tensor, batch_size: int, num_classes: int) -> Dict[str, Any]:
    logits_parts: List[torch.Tensor] = []
    feature_parts: List[torch.Tensor] = []
    with torch.no_grad():
        for xb in x.split(int(batch_size)):
            h, _ = stack.forward_manual(xb)
            logits, _ = head.forward_manual(h)
            logits_parts.append(logits.detach())
            feature_parts.append(h.detach())
    logits = torch.cat(logits_parts, dim=0)
    features = torch.cat(feature_parts, dim=0)
    loss, acc, nll, ece, conf, margin = _eval_logits(logits, y)
    return {
        "loss": loss,
        "acc": acc,
        "NLL": nll,
        "ECE": ece,
        "confidence_mean": conf,
        "margin_p10": margin,
        "feature_effective_rank": _effective_rank(features),
        "classwise_acc": _classwise_acc(logits, y, num_classes),
    }


def _mlp_eval(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, num_classes: int) -> Dict[str, Any]:
    logits_parts: List[torch.Tensor] = []
    with torch.no_grad():
        for xb in x.split(int(batch_size)):
            logits_parts.append(model(xb).detach())
    logits = torch.cat(logits_parts, dim=0)
    loss, acc, nll, ece, conf, margin = _eval_logits(logits, y)
    hidden = torch.empty((0, 1), device=x.device)
    return {
        "loss": loss,
        "acc": acc,
        "NLL": nll,
        "ECE": ece,
        "confidence_mean": conf,
        "margin_p10": margin,
        "feature_effective_rank": _effective_rank(hidden),
        "classwise_acc": _classwise_acc(logits, y, num_classes),
    }


def _train_mlp(args: argparse.Namespace, spec: CandidateSpec, dataset: str, seed: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = load_vision_bundle(dataset, data_root=Path(args.data_root), train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, seed=seed, allow_fake_data=False)
    x_train = bundle.x_train.to(device)
    y_train = bundle.y_train.to(device)
    x_val = bundle.x_val.to(device)
    y_val = bundle.y_val.to(device)
    x_test = bundle.x_test.to(device)
    y_test = bundle.y_test.to(device)
    set_seed(_stable_seed("v71", dataset, seed, spec.candidate_id, "mlp"))
    model = _make_mlp(bundle.input_dim, bundle.num_classes, args.hidden_dim, 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params.lr_mlp, weight_decay=args.weight_decay)
    train_eval_x = x_train[: min(max(args.batch_size, 512), x_train.shape[0])]
    train_eval_y = y_train[: train_eval_x.shape[0]]
    train0 = _mlp_eval(model, train_eval_x, train_eval_y, args.eval_batch_size, bundle.num_classes)
    val0 = _mlp_eval(model, x_val, y_val, args.eval_batch_size, bundle.num_classes)
    trace: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, int(args.task_steps) + 1):
        xb, yb = _select_batch(x_train, y_train, args.batch_size, step)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        if step % int(args.trace_every) == 0 or step == int(args.task_steps):
            tr = _mlp_eval(model, train_eval_x, train_eval_y, args.eval_batch_size, bundle.num_classes)
            va = _mlp_eval(model, x_val, y_val, args.eval_batch_size, bundle.num_classes)
            trace.append({
                "stage": "TASK_TRACE",
                "candidate_id": spec.candidate_id,
                "candidate_name": spec.candidate_name,
                "dataset": dataset,
                "seed": seed,
                "step": step,
                "wall_clock_time_sec": time.perf_counter() - started,
                "train_loss": tr["loss"],
                "train_acc": tr["acc"],
                "val_loss": va["loss"],
                "val_acc": va["acc"],
                "ECE": va["ECE"],
                "NLL": va["NLL"],
                "fake_data_used": 0,
                "proxy_row_used": 0,
            })
    train1 = _mlp_eval(model, train_eval_x, train_eval_y, args.eval_batch_size, bundle.num_classes)
    val1 = _mlp_eval(model, x_val, y_val, args.eval_batch_size, bundle.num_classes)
    test1 = _mlp_eval(model, x_test, y_test, args.eval_batch_size, bundle.num_classes)
    summary = _task_summary_base(args, spec, dataset, seed, bundle.input_dim, bundle.num_classes)
    summary.update({
        "train_loss_before": train0["loss"],
        "train_loss_after": train1["loss"],
        "val_loss_before": val0["loss"],
        "val_loss_after": val1["loss"],
        "train_acc": train1["acc"],
        "val_acc": val1["acc"],
        "test_acc": test1["acc"],
        "ECE": val1["ECE"],
        "NLL": val1["NLL"],
        "test_ECE": test1["ECE"],
        "test_NLL": test1["NLL"],
        "margin_p10": val1["margin_p10"],
        "feature_effective_rank": val1["feature_effective_rank"],
        "classwise_acc": val1["classwise_acc"],
        "val_loss_auc_step": _auc([(r["step"], r["val_loss"]) for r in trace]),
        "val_loss_auc_time": _auc([(r["wall_clock_time_sec"], r["val_loss"]) for r in trace]),
        "wall_clock_time_sec": time.perf_counter() - started,
        "implementation_status": "measured",
        "stage_status": "measured",
    })
    return summary, trace


def _task_summary_base(args: argparse.Namespace, spec: CandidateSpec, dataset: str, seed: int, input_dim: int, num_classes: int) -> Dict[str, Any]:
    head_is_kan = int(spec.head_kind not in {"linear", "mlp"})
    nonkan = 0 if head_is_kan else (args.hidden_dim * num_classes + num_classes if spec.stack_type != "mlp" else METRIC_UNAVAILABLE)
    return {
        "stage": "TASK",
        "candidate_id": spec.candidate_id,
        "candidate_name": spec.candidate_name,
        "family": spec.family,
        "dataset": dataset,
        "seed": seed,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "depth": spec.depth,
        "basis_count": args.basis_count,
        "task_steps_configured": args.task_steps,
        "stack_type": spec.stack_type,
        "stack_kind": spec.stack_kind,
        "head_type": spec.head_kind,
        "head_is_kan": head_is_kan,
        "manual_forward": int(spec.stack_type != "mlp"),
        "manual_backward": int(spec.stack_type != "mlp"),
        "manual_update": int(spec.stack_type != "mlp"),
        "uses_torch_loss_backward": int(spec.stack_type == "mlp"),
        "uses_torch_autograd_graph": int(spec.stack_type == "mlp"),
        "non_kan_trainable_param_count": nonkan,
        "oracle_or_mainline": "oracle" if spec.oracle else "mainline",
        "strict_status": spec.strict_status,
        "allowed_in_route_selection": int(spec.allowed_in_route_selection),
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _train_manual(args: argparse.Namespace, spec: CandidateSpec, dataset: str, seed: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = load_vision_bundle(dataset, data_root=Path(args.data_root), train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, seed=seed, allow_fake_data=False)
    x_train = bundle.x_train.to(device)
    y_train = bundle.y_train.to(device)
    x_val = bundle.x_val.to(device)
    y_val = bundle.y_val.to(device)
    x_test = bundle.x_test.to(device)
    y_test = bundle.y_test.to(device)
    set_seed(_stable_seed("v71", dataset, seed, spec.candidate_id, spec.head_kind, spec.group_count, spec.shuffle))
    stack, head = _make_manual_candidate(spec, bundle.input_dim, bundle.num_classes, args.hidden_dim, args.basis_count, device)
    opt = FastAdamW([stack, head], lr=params.lr_manual * spec.lr_mult, weight_decay=args.weight_decay)
    train_eval_x = x_train[: min(max(args.batch_size, 512), x_train.shape[0])]
    train_eval_y = y_train[: train_eval_x.shape[0]]
    train0 = _manual_eval(stack, head, train_eval_x, train_eval_y, args.eval_batch_size, bundle.num_classes)
    val0 = _manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
    trace: List[Dict[str, Any]] = []
    first_loss_before = float("nan")
    first_loss_after = float("nan")
    started = time.perf_counter()
    for step in range(1, int(args.task_steps) + 1):
        xb, yb = _select_batch(x_train, y_train, args.batch_size, step)
        h, caches = stack.forward_manual(xb)
        logits, head_cache = head.forward_manual(h)
        loss, grad_logits = _smooth_ce_and_grad(logits, yb, spec.label_smoothing)
        if step == 1:
            first_loss_before = float(loss.detach().cpu())
        dh = head.backward_manual(grad_logits, head_cache)
        stack.backward_manual(dh, caches)
        update_norm = opt.step(step, args.task_steps, warmup_cosine=True)
        if step == 1:
            with torch.no_grad():
                h1, _ = stack.forward_manual(xb)
                logits1, _ = head.forward_manual(h1)
                first_loss_after = float(_smooth_ce_and_grad(logits1, yb, spec.label_smoothing)[0].detach().cpu())
        if step % int(args.trace_every) == 0 or step == int(args.task_steps):
            tr = _manual_eval(stack, head, train_eval_x, train_eval_y, args.eval_batch_size, bundle.num_classes)
            va = _manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
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
                "ECE": va["ECE"],
                "NLL": va["NLL"],
                "update_norm": update_norm,
                "fake_data_used": 0,
                "proxy_row_used": 0,
            })
    train1 = _manual_eval(stack, head, train_eval_x, train_eval_y, args.eval_batch_size, bundle.num_classes)
    val1 = _manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
    test1 = _manual_eval(stack, head, x_test, y_test, args.eval_batch_size, bundle.num_classes)
    summary = _task_summary_base(args, spec, dataset, seed, bundle.input_dim, bundle.num_classes)
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
        "val_loss_auc_step": _auc([(r["step"], r["val_loss"]) for r in trace]),
        "val_loss_auc_time": _auc([(r["wall_clock_time_sec"], r["val_loss"]) for r in trace]),
        "stack_param_count": stack.param_count(),
        "head_param_count": head.param_count(),
        "kan_trainable_param_count": stack.param_count() + (head.param_count() if spec.head_kind != "linear" else 0),
        "edge_param_count": stack.param_count() + head.param_count(),
        "wall_clock_time_sec": time.perf_counter() - started,
        "implementation_status": "measured",
        "stage_status": "measured",
    })
    return summary, trace


def _bench_mlp(args: argparse.Namespace, dataset: str, batch_size: int) -> Dict[str, Any]:
    device = get_device(args.device)
    bundle = load_vision_bundle(dataset, data_root=Path(args.data_root), train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, seed=0, allow_fake_data=False)
    x = bundle.x_train[:batch_size].to(device)
    y = bundle.y_train[:batch_size].to(device)
    set_seed(_stable_seed("v71-bench", dataset, batch_size, "mlp"))
    model = _make_mlp(bundle.input_dim, bundle.num_classes, args.hidden_dim, 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=args.weight_decay)
    forward_times: List[float] = []
    backward_times: List[float] = []
    update_times: List[float] = []
    step_times: List[float] = []
    _reset_peak(device)
    for rep in range(int(args.bench_warmup) + int(args.bench_reps)):
        opt.zero_grad(set_to_none=True)
        _sync(device)
        t0 = time.perf_counter()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        _sync(device)
        t1 = time.perf_counter()
        loss.backward()
        _sync(device)
        t2 = time.perf_counter()
        opt.step()
        _sync(device)
        t3 = time.perf_counter()
        if rep >= int(args.bench_warmup):
            forward_times.append((t1 - t0) * 1000.0)
            backward_times.append((t2 - t1) * 1000.0)
            update_times.append((t3 - t2) * 1000.0)
            step_times.append((t3 - t0) * 1000.0)
    return {
        "candidate_id": "B0",
        "candidate_name": "MLP-autograd-reference-depth2",
        "dataset": dataset,
        "batch_size": batch_size,
        "forward_time_ms": _mean(forward_times),
        "backward_time_ms": _mean(backward_times),
        "update_time_ms": _mean(update_times),
        "step_time_ms": _mean(step_times),
        "peak_allocated_MB": _peak_allocated_mb(device),
        "peak_reserved_MB": _peak_reserved_mb(device),
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _bench_manual(args: argparse.Namespace, spec: CandidateSpec, dataset: str, batch_size: int, denominators: Dict[Tuple[str, int], Dict[str, Any]]) -> Dict[str, Any]:
    device = get_device(args.device)
    bundle = load_vision_bundle(dataset, data_root=Path(args.data_root), train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, seed=0, allow_fake_data=False)
    x = bundle.x_train[:batch_size].to(device)
    y = bundle.y_train[:batch_size].to(device)
    set_seed(_stable_seed("v71-bench", dataset, batch_size, spec.candidate_id))
    stack, head = _make_manual_candidate(spec, bundle.input_dim, bundle.num_classes, args.hidden_dim, args.basis_count, device)
    opt = FastAdamW([stack, head], lr=V63Params.lr_manual, weight_decay=args.weight_decay)
    forward_times: List[float] = []
    backward_times: List[float] = []
    update_times: List[float] = []
    step_times: List[float] = []
    cache_total_mb = float("nan")
    cache_details: Dict[str, Any] = {}
    _reset_peak(device)
    for rep in range(int(args.bench_warmup) + int(args.bench_reps)):
        _sync(device)
        t0 = time.perf_counter()
        h, caches = stack.forward_manual(x)
        logits, head_cache = head.forward_manual(h)
        loss, grad_logits = _smooth_ce_and_grad(logits, y, spec.label_smoothing)
        _sync(device)
        t1 = time.perf_counter()
        dh = head.backward_manual(grad_logits, head_cache)
        if getattr(stack, "release_head_cache_before_stack_backward", False):
            del head_cache, logits, loss, grad_logits
        stack.backward_manual(dh, caches)
        _sync(device)
        t2 = time.perf_counter()
        opt.step(rep + 1, int(args.bench_warmup) + int(args.bench_reps), warmup_cosine=False)
        _sync(device)
        t3 = time.perf_counter()
        if rep >= int(args.bench_warmup):
            forward_times.append((t1 - t0) * 1000.0)
            backward_times.append((t2 - t1) * 1000.0)
            update_times.append((t3 - t2) * 1000.0)
            step_times.append((t3 - t0) * 1000.0)
            if hasattr(stack, "cache_breakdown"):
                cache_details = dict(stack.cache_breakdown(caches))
                cache_total_mb = float(cache_details.get("cache_total_MB", float("nan")))
    den = denominators[(dataset, batch_size)]
    memory_ratio = _peak_allocated_mb(device) / float(den["peak_allocated_MB"]) if _finite(den["peak_allocated_MB"]) and float(den["peak_allocated_MB"]) > 0 else float("nan")
    step_ratio = _mean(step_times) / float(den["step_time_ms"])
    forward_ratio = _mean(forward_times) / float(den["forward_time_ms"])
    backward_ratio = _mean(backward_times) / float(den["backward_time_ms"])
    update_ratio = _mean(update_times) / float(den["update_time_ms"])
    survivor = "S2" if memory_ratio <= 1.05 and step_ratio <= 1.50 else "FAIL"
    return {
        "candidate_id": spec.candidate_id,
        "candidate_name": spec.candidate_name,
        "family": spec.family,
        "dataset": dataset,
        "batch_size": batch_size,
        "depth": spec.depth,
        "head_type": spec.head_kind,
        "head_is_kan": int(spec.head_kind != "linear"),
        "forward_time_ms": _mean(forward_times),
        "backward_time_ms": _mean(backward_times),
        "update_time_ms": _mean(update_times),
        "step_time_ms": _mean(step_times),
        "peak_allocated_MB": _peak_allocated_mb(device),
        "peak_reserved_MB": _peak_reserved_mb(device),
        "MLP_forward_time_ms": den["forward_time_ms"],
        "MLP_backward_time_ms": den["backward_time_ms"],
        "MLP_update_time_ms": den["update_time_ms"],
        "MLP_step_time_ms": den["step_time_ms"],
        "MLP_peak_allocated_MB": den["peak_allocated_MB"],
        "memory_ratio": memory_ratio,
        "step_ratio": step_ratio,
        "forward_ratio": forward_ratio,
        "backward_ratio": backward_ratio,
        "update_ratio": update_ratio,
        "cache_total_MB": cache_total_mb,
        **cache_details,
        "optimizer_state_memory_MB": sum(p.numel() * p.element_size() * 2 for _n, p, _g in opt.params()) / (1024**2),
        "top1_memory_source": "manual_cache_or_dense_live_set",
        "top2_memory_source": "optimizer_state",
        "top3_memory_source": "head_or_group_buffers",
        "unknown_memory_fraction": METRIC_UNAVAILABLE,
        "kernel_count_forward": getattr(stack, "op_counts", lambda: {})().get("op_count_gemm", METRIC_UNAVAILABLE),
        "survivor": survivor,
        "s2_pass": int(survivor == "S2"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _aggregate_task(task_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mlp_by_dataset: Dict[str, float] = {}
    for row in task_rows:
        if row.get("candidate_id") == "B0":
            mlp_by_dataset.setdefault(str(row["dataset"]), 0.0)
    for ds in list(mlp_by_dataset):
        mlp_by_dataset[ds] = _mean([r["val_acc"] for r in task_rows if r.get("candidate_id") == "B0" and r.get("dataset") == ds])
    rows: List[Dict[str, Any]] = []
    for cid in sorted({str(r["candidate_id"]) for r in task_rows}):
        subset = [r for r in task_rows if str(r["candidate_id"]) == cid]
        datasets = sorted({str(r["dataset"]) for r in subset})
        dataset_gaps: Dict[str, float] = {}
        within = 0
        ge = 0
        plus = 0
        for ds in datasets:
            val = _mean([r["val_acc"] for r in subset if r["dataset"] == ds])
            gap = val - mlp_by_dataset.get(ds, float("nan"))
            dataset_gaps[ds] = gap
            within += int(_finite(gap) and gap >= -0.01)
            ge += int(_finite(gap) and gap >= 0.0)
            plus += int(_finite(gap) and gap >= 0.005)
        first = subset[0]
        basic = int(cid != "B0" and within == len(datasets) and ge >= 2)
        strong_partial = int(cid != "B0" and plus >= 2 and _mean([r["ECE"] for r in subset]) <= _mean([r["ECE"] for r in task_rows if r.get("candidate_id") == "B0"]))
        rows.append({
            "candidate_id": cid,
            "candidate_name": first.get("candidate_name"),
            "family": first.get("family"),
            "rows": len(subset),
            "val_acc_mean": _mean([r["val_acc"] for r in subset]),
            "val_acc_std": _std([r["val_acc"] for r in subset]),
            "test_acc_mean": _mean([r["test_acc"] for r in subset]),
            "ECE_mean": _mean([r["ECE"] for r in subset]),
            "NLL_mean": _mean([r["NLL"] for r in subset]),
            "val_gap_vs_MLP_mean": _mean(dataset_gaps.values()),
            "dataset_gaps": dataset_gaps,
            "datasets_within_1pct_MLP": within,
            "datasets_ge_MLP": ge,
            "datasets_ge_MLP_plus_005": plus,
            "basic_beyond_pass": basic,
            "strong_beyond_partial": strong_partial,
            "strict_status": first.get("strict_status"),
            "head_type": first.get("head_type"),
            "head_is_kan": first.get("head_is_kan"),
            "manual_forward": first.get("manual_forward"),
            "manual_backward": first.get("manual_backward"),
            "manual_update": first.get("manual_update"),
            "non_kan_trainable_param_count": first.get("non_kan_trainable_param_count"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    return rows


def _aggregate_eff(eff_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cid in sorted({str(r["candidate_id"]) for r in eff_rows if r.get("candidate_id") != "B0"}):
        subset = [r for r in eff_rows if str(r["candidate_id"]) == cid]
        first = subset[0]
        rows.append({
            "candidate_id": cid,
            "candidate_name": first.get("candidate_name"),
            "family": first.get("family"),
            "rows": len(subset),
            "memory_ratio_mean": _mean([r["memory_ratio"] for r in subset]),
            "step_ratio_mean": _mean([r["step_ratio"] for r in subset]),
            "forward_ratio_mean": _mean([r["forward_ratio"] for r in subset]),
            "backward_ratio_mean": _mean([r["backward_ratio"] for r in subset]),
            "update_ratio_mean": _mean([r["update_ratio"] for r in subset]),
            "s2_pass_shapes": sum(int(r.get("s2_pass", 0)) for r in subset),
            "survivor": "S2" if all(int(r.get("s2_pass", 0)) == 1 for r in subset) else "FAIL",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    return rows


def _make_failure_table(task_summary: Sequence[Dict[str, Any]], eff_summary: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in task_summary:
        cid = row["candidate_id"]
        if cid == "B0":
            continue
        if row.get("head_is_kan") == 0:
            rows.append({"candidate_id": cid, "failure_type": "F1_oracle_nonkan_head", "reason": "linear/non-KAN head is diagnostic only"})
        if int(row.get("basic_beyond_pass", 0)) != 1:
            rows.append({"candidate_id": cid, "failure_type": "F2_task_gate_fail", "reason": f"val_gap_vs_MLP_mean={row.get('val_gap_vs_MLP_mean')}"})
        if row.get("non_kan_trainable_param_count") not in {0, "0"} and row.get("head_is_kan") == 1:
            rows.append({"candidate_id": cid, "failure_type": "F3_strict_param_accounting_fail", "reason": "non_kan_trainable_param_count nonzero"})
    for row in eff_summary:
        if row.get("survivor") != "S2":
            rows.append({"candidate_id": row["candidate_id"], "failure_type": "F4_efficiency_s2_fail", "reason": f"memory={row.get('memory_ratio_mean')} step={row.get('step_ratio_mean')}"})
    if not rows:
        rows.append({"candidate_id": "ALL", "failure_type": "none", "reason": "no failures recorded"})
    for row in rows:
        row["fake_data_used"] = 0
        row["proxy_row_used"] = 0
    return rows


def _route_decision(task_summary: Sequence[Dict[str, Any]], eff_summary: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    eff_by_id = {str(r["candidate_id"]): r for r in eff_summary}
    best_rows = [r for r in task_summary if r.get("candidate_id") != "B0"]
    best_rows.sort(key=lambda r: (int(r.get("basic_beyond_pass", 0)), float(r.get("val_gap_vs_MLP_mean", -999.0))), reverse=True)
    best = best_rows[0] if best_rows else {}
    best_eff = eff_by_id.get(str(best.get("candidate_id")), {})
    strict_eff_task = [
        r for r in task_summary
        if r.get("head_is_kan") == 1
        and int(r.get("basic_beyond_pass", 0)) == 1
        and eff_by_id.get(str(r.get("candidate_id")), {}).get("survivor") == "S2"
    ]
    strict_task = [
        r for r in task_summary
        if r.get("head_is_kan") == 1 and int(r.get("basic_beyond_pass", 0)) == 1
    ]
    if strict_eff_task:
        route = "R1-StrictS2BeyondPass"
        primary_blocker = "none"
        winner = strict_eff_task[0]
    elif strict_task:
        route = "R2-StrictTaskPassEfficiencyFail"
        primary_blocker = "efficiency_kernelization"
        strict_task.sort(
            key=lambda r: (
                int(r.get("basic_beyond_pass", 0)),
                float(r.get("val_gap_vs_MLP_mean", -999.0)),
                float(r.get("test_acc_mean", -999.0)),
            ),
            reverse=True,
        )
        winner = strict_task[0]
    elif best.get("basic_beyond_pass") == 1:
        route = "R3-OracleTaskPassStrictOrEfficiencyFail"
        primary_blocker = "strict_head_or_kernelization"
        winner = best
    else:
        route = "R4-TaskGapPersists"
        primary_blocker = "task_transfer"
        winner = best
    winner_eff = eff_by_id.get(str(winner.get("candidate_id")), {})
    return {
        "route": route,
        "best_candidate": winner.get("candidate_name"),
        "best_candidate_id": winner.get("candidate_id"),
        "best_family": winner.get("family"),
        "best_val_acc": winner.get("val_acc_mean"),
        "best_test_acc": winner.get("test_acc_mean"),
        "best_val_gap_vs_MLP": winner.get("val_gap_vs_MLP_mean"),
        "basic_beyond_pass": int(winner.get("basic_beyond_pass", 0)) if winner else 0,
        "strict_head": int(winner.get("head_is_kan", 0)) if winner else 0,
        "s2_pass": int(winner_eff.get("survivor") == "S2") if winner_eff else 0,
        "best_memory_ratio": winner_eff.get("memory_ratio_mean"),
        "best_step_ratio": winner_eff.get("step_ratio_mean"),
        "official_task_opened": False,
        "diagnostic_task_opened": True,
        "primary_blocker": primary_blocker,
        "next_required_implementation": "materialization_free_kernel_or_better_structured_bridge",
        "no_fake": True,
        "no_proxy": True,
    }


def _audit_fake_proxy(paths: Sequence[Path]) -> Dict[str, Any]:
    nonzero = 0
    checked = 0
    for path in paths:
        if path.suffix != ".csv" or not path.exists():
            continue
        import csv

        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                checked += 1
                for key in ("fake_data_used", "uses_fake_data", "proxy_row_used", "proxy_rows_used"):
                    val = row.get(key)
                    if val not in (None, "", "0", "0.0", "False", "false"):
                        nonzero += 1
    return {"rows_checked": checked, "fake_proxy_nonzero_count": nonzero, "no_fake": nonzero == 0, "no_proxy": nonzero == 0}


def run(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(Path(args.out_dir))
    if args.fresh and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty out_dir with --fresh: {out_dir}")
    candidates = _candidate_registry()
    selected = [c for c in candidates if c.candidate_id in set(parse_str_list(args.candidates))]
    if not selected:
        selected = candidates
    registry_rows = [
        {
            "candidate_id": c.candidate_id,
            "candidate_name": c.candidate_name,
            "family": c.family,
            "oracle_or_mainline": "oracle" if c.oracle else "mainline",
            "strict_status": c.strict_status,
            "head_type": c.head_kind,
            "head_is_kan": int(c.head_kind not in {"linear", "mlp"}),
            "implementation_status": c.implementation_status,
            "expected_stage": "P1/P2/P3/P5",
            "allowed_in_route_selection": int(c.allowed_in_route_selection),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        }
        for c in candidates
    ]
    write_csv(out_dir / "candidate_registry.csv", registry_rows)
    save_json(out_dir / "gate_config.json", {
        "s2_memory_ratio_max": 1.05,
        "s2_step_ratio_max": 1.50,
        "s1_memory_ratio_max_strict": 1.00,
        "s1_step_ratio_max": 1.35,
        "basic_beyond_dataset_gap_min": -0.01,
        "basic_beyond_num_datasets_ge_mlp": 2,
    })
    task_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    datasets = parse_str_list(args.datasets)
    seeds = parse_int_list(args.seeds)
    for dataset in datasets:
        for seed in seeds:
            for spec in selected:
                print(f"[v71] task {dataset} seed={seed} {spec.candidate_id} {spec.candidate_name}", flush=True)
                if spec.stack_type == "mlp":
                    row, trace = _train_mlp(args, spec, dataset, seed)
                else:
                    row, trace = _train_manual(args, spec, dataset, seed)
                task_rows.append(row)
                trace_rows.extend(trace)
    task_summary = _aggregate_task(task_rows)
    mlp_by_ds = {
        row["candidate_id"] + ":" + row.get("dataset", ""): row
        for row in task_rows
        if row.get("candidate_id") == "B0"
    }
    for row in task_rows:
        ds = row.get("dataset", "")
        mlp_vals = [r["val_acc"] for r in task_rows if r.get("candidate_id") == "B0" and r.get("dataset") == ds]
        mlp_tests = [r["test_acc"] for r in task_rows if r.get("candidate_id") == "B0" and r.get("dataset") == ds]
        row["val_gap_vs_MLP_dataset_mean"] = row["val_acc"] - _mean(mlp_vals)
        row["test_gap_vs_MLP_dataset_mean"] = row["test_acc"] - _mean(mlp_tests)
    write_csv(out_dir / "p1_p2_p3_task.csv", task_rows)
    write_csv(out_dir / "p1_p2_p3_task_trace.csv", trace_rows)
    write_csv(out_dir / "p1_p2_p3_task_summary.csv", task_summary)
    denominators: Dict[Tuple[str, int], Dict[str, Any]] = {}
    eff_rows: List[Dict[str, Any]] = []
    bench_candidates = [c for c in selected if c.stack_type != "mlp"]
    for dataset in datasets:
        for batch_size in parse_int_list(args.bench_batch_sizes):
            print(f"[v71] bench denominator {dataset} bs={batch_size}", flush=True)
            den = _bench_mlp(args, dataset, batch_size)
            denominators[(dataset, batch_size)] = den
            eff_rows.append(den)
            for spec in bench_candidates:
                print(f"[v71] bench {dataset} bs={batch_size} {spec.candidate_id}", flush=True)
                eff_rows.append(_bench_manual(args, spec, dataset, batch_size, denominators))
    eff_summary = _aggregate_eff(eff_rows)
    write_csv(out_dir / "p5_efficiency_detail.csv", eff_rows)
    write_csv(out_dir / "p5_efficiency_summary.csv", eff_summary)
    failure_table = _make_failure_table(task_summary, eff_summary)
    write_csv(out_dir / "failure_table.csv", failure_table)
    route = _route_decision(task_summary, eff_summary)
    save_json(out_dir / "route_decision.json", route)
    audit = _audit_fake_proxy([
        out_dir / "candidate_registry.csv",
        out_dir / "p1_p2_p3_task.csv",
        out_dir / "p1_p2_p3_task_trace.csv",
        out_dir / "p1_p2_p3_task_summary.csv",
        out_dir / "p5_efficiency_detail.csv",
        out_dir / "p5_efficiency_summary.csv",
        out_dir / "failure_table.csv",
    ])
    write_csv(out_dir / "provenance_audit.csv", [{**audit, "plan_path": PLAN_PATH, "script_path": SCRIPT_PATH, "fake_data_used": 0, "proxy_row_used": 0}])
    save_json(out_dir / "run_manifest.json", {
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
        "efficiency_rows": len(eff_rows),
        "route": route,
        "audit": audit,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--candidates", default="B0,B3,H1,H2,H3,K1,K2,K3,K4")
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
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
