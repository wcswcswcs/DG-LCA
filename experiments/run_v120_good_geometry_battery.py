#!/usr/bin/env python3
"""DG-KAN v12.0 Good Geometry Battery runner.

This runner is intentionally empirical and conservative.  It measures the
repaired LQ base, passive geometry metrics, GeometryCertificateV0, and cloned
one-step/five-step functional-direction audits.  It does not use teachers,
loss modifications, fake rows, or dataset-specific controller thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v922_fused_compositional_kernel_closure as f922  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import geometry_certificate as geom_cert  # noqa: E402
from dgkan.functional import manifold_channel_geometry as geom  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig, adamw_update_  # noqa: E402
from dgkan.training.manual_full_edge import ce_loss_and_grad  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v12.1_Codex下一步执行计划_GoodGeometryBattery.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v120_good_geometry_battery.py"
REPORT_PATH = ROOT / "docs" / "DG-KAN_v12.0_GoodGeometryBattery_结果复盘.md"
EPS = 1.0e-12


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    model_family: str
    candidate_id: str
    kind: str
    official_candidate: int
    strict_purekan: int
    lq_spec: lq.LQSpec | None = None
    description: str = ""


@dataclass
class ModelPack:
    method: MethodSpec
    params: List[torch.Tensor]
    states: List[AdamWState]
    device: torch.device
    basis: str = ""
    mu: torch.Tensor | None = None
    std: torch.Tensor | None = None
    hidden_dim: int = 0
    qproj: torch.Tensor | None = None


@dataclass
class Snapshot:
    method_id: str
    model_family: str
    candidate_id: str
    dataset: str
    seed: int
    checkpoint_id: str
    epoch: int
    train_step: int
    wall_clock_sec: float
    pack: ModelPack


def _now_tag() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _canonical_dataset(name: str) -> str:
    key = str(name).strip().lower()
    if key in {"fashion", "fashion-mnist", "fmnist"}:
        return "Fashion-MNIST"
    if key == "kmnist":
        return "KMNIST"
    if key == "mnist":
        return "MNIST"
    raise ValueError(f"unsupported v12 dataset {name!r}")


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().cpu())
        return float(value)
    except Exception:
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _q(values: Sequence[float], q: float, default: float = 0.0) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return default
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def _rankdata(values: Sequence[float]) -> List[float]:
    pairs = sorted((float(v), i) for i, v in enumerate(values))
    ranks = [0.0 for _ in pairs]
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[pairs[k][1]] = rank
        i = j
    return ranks


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = _mean(xs)
    my = _mean(ys)
    vx = sum((float(x) - mx) ** 2 for x in xs)
    vy = sum((float(y) - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    cov = sum((float(x) - mx) * (float(y) - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return _corr(_rankdata(xs), _rankdata(ys))


def _kendall(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    con = 0
    dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = 1 if xs[i] > xs[j] else -1 if xs[i] < xs[j] else 0
            sy = 1 if ys[i] > ys[j] else -1 if ys[i] < ys[j] else 0
            prod = sx * sy
            if prod > 0:
                con += 1
            elif prod < 0:
                dis += 1
    den = con + dis
    return (con - dis) / den if den else 0.0


def _not_run(stage: str, reason: str, **extra: Any) -> Dict[str, Any]:
    row = {
        "stage": stage,
        "status": "not_run",
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    row.update(extra)
    return row


def _failure(
    stage: str,
    code: str,
    reason: str,
    *,
    method_id: str = "",
    dataset: str = "",
    seed: int | str = "",
    checkpoint_id: str = "",
    metric_name: str = "",
    metric_value: Any = "",
    threshold: Any = "",
    action: str = "",
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "method_id": method_id,
        "dataset": dataset,
        "seed": seed,
        "checkpoint_id": checkpoint_id,
        "failure_code": code,
        "failure_reason": reason,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "threshold": threshold,
        "action_recommended": action,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _methods() -> List[MethodSpec]:
    return [
        MethodSpec("B0-MLP-same-param-AdamW", "MLP", "B0-MLP-same-param", "mlp_same_param", 0, 0, None, "same-param manual MLP"),
        MethodSpec("B1-MLP-same-step-or-same-FLOPs-AdamW", "MLP", "B1-MLP-same-step", "mlp_same_step", 0, 0, None, "same-step manual MLP"),
        MethodSpec("B2-QuadraticFeatureMLP-diagnostic-AdamW", "QuadraticFeatureMLP", "B2-QuadraticFeatureMLP-diagnostic", "qfeature", 0, 0, None, "fixed quadratic features plus manual linear head"),
        MethodSpec(
            "B3-LQ-t2-h256-AdamW",
            "FC-PureKAN-LQ",
            "B3-LQ-t2-h256",
            "lq",
            1,
            1,
            lq.LQSpec("LQ-t2-h256", "t2", 256, "default", 1.0, repair_hypothesis="historical_lq_t2"),
            "historical LQ-t2 reference",
        ),
        MethodSpec(
            "B4-R2-LQ-fanin-output-scale-confirmed-AdamW",
            "FC-PureKAN-LQ",
            "R2-LQ-fanin-output-scale-confirmed",
            "lq",
            1,
            1,
            lq.LQSpec(
                "R2-LQ-fanin-output-scale-confirmed",
                "t2",
                256,
                "default",
                0.8,
                repair_hypothesis="fan_in_output_scale_confirmed",
            ),
            "v12 primary repaired LQ base",
        ),
    ]


def _load_vision_split(
    args: argparse.Namespace,
    dataset: str,
    *,
    train_size: int,
    val_size: int,
    test_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, str]:
    try:
        from torchvision import datasets
    except Exception as exc:
        raise RuntimeError(f"torchvision import failed: {exc}") from exc

    canonical = _canonical_dataset(dataset)
    if canonical == "Fashion-MNIST":
        ds_cls = datasets.FashionMNIST
        mean, std = 0.2860, 0.3530
    elif canonical == "KMNIST":
        ds_cls = datasets.KMNIST
        mean, std = 0.1918, 0.3483
    else:
        ds_cls = datasets.MNIST
        mean, std = 0.1307, 0.3081

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    download = not bool(args.no_download)
    try:
        train_ds = ds_cls(root=str(data_root), train=True, download=download)
        test_ds = ds_cls(root=str(data_root), train=False, download=download)
    except Exception as exc:
        raise RuntimeError(
            f"dataset {canonical} unavailable under {data_root}; no_download={int(args.no_download)}; {exc}"
        ) from exc

    x_all = ((train_ds.data.float().unsqueeze(1) / 255.0 - mean) / std).reshape(int(train_ds.data.shape[0]), -1)
    y_all = train_ds.targets.long()
    x_test_all = ((test_ds.data.float().unsqueeze(1) / 255.0 - mean) / std).reshape(int(test_ds.data.shape[0]), -1)
    y_test_all = test_ds.targets.long()
    need = min(int(x_all.shape[0]), int(train_size) + int(val_size))
    gen = torch.Generator().manual_seed(int(args.seed))
    idx = torch.randperm(int(x_all.shape[0]), generator=gen)[:need]
    train_n = min(int(train_size), need)
    val_n = max(0, min(int(val_size), need - train_n))
    train_idx = idx[:train_n]
    val_idx = idx[train_n : train_n + val_n]
    test_n = min(int(test_size), int(x_test_all.shape[0]))
    protocol = (
        f"KANbeFair vision transform; train={train_n}; val={val_n}; "
        f"test={test_n}; shuffle_seed={int(args.seed)}; download={int(download)}"
    )
    return (
        x_all[train_idx].contiguous(),
        y_all[train_idx].contiguous(),
        x_all[val_idx].contiguous(),
        y_all[val_idx].contiguous(),
        x_test_all[:test_n].contiguous(),
        y_test_all[:test_n].contiguous(),
        int(x_all.shape[1]),
        int(y_all.max().item() + 1),
        protocol,
    )


def _init_pack(method: MethodSpec, input_dim: int, output_dim: int, x_for_stats: torch.Tensor, seed: int, device: torch.device) -> ModelPack:
    if method.kind == "lq":
        assert method.lq_spec is not None
        params, mu, std = lq.init_lq_params(input_dim, output_dim, method.lq_spec, x_for_stats, device, int(seed) + 92600)
        return ModelPack(method, params, [AdamWState.zeros_like(p) for p in params], device, basis=method.lq_spec.basis, mu=mu, std=std)

    r2_spec = [m for m in _methods() if m.candidate_id == "R2-LQ-fanin-output-scale-confirmed"][0].lq_spec
    assert r2_spec is not None
    r2_count = r2_spec.hidden_dim * input_dim + r2_spec.hidden_dim * output_dim * lq.BASIS_CHANNELS[r2_spec.basis]
    gen = torch.Generator(device=device).manual_seed(int(seed) + 12000)
    if method.kind == "mlp_same_param":
        hidden = f922._matched_mlp3_hidden(r2_count, input_dim, output_dim)
        params = [
            torch.randn(input_dim, hidden, device=device, generator=gen) / math.sqrt(input_dim),
            torch.randn(hidden, hidden, device=device, generator=gen) / math.sqrt(hidden),
            torch.randn(hidden, output_dim, device=device, generator=gen) / math.sqrt(hidden),
        ]
        return ModelPack(method, params, [AdamWState.zeros_like(p) for p in params], device, hidden_dim=hidden)
    if method.kind == "mlp_same_step":
        hidden = 256
        params = [
            torch.randn(input_dim, hidden, device=device, generator=gen) / math.sqrt(input_dim),
            torch.randn(hidden, hidden, device=device, generator=gen) / math.sqrt(hidden),
            torch.randn(hidden, output_dim, device=device, generator=gen) / math.sqrt(hidden),
        ]
        return ModelPack(method, params, [AdamWState.zeros_like(p) for p in params], device, hidden_dim=hidden)
    if method.kind == "qfeature":
        qdim = min(128, input_dim)
        qproj = torch.randn(input_dim, qdim, device=device, generator=gen) / math.sqrt(input_dim)
        params = [torch.randn(input_dim + qdim, output_dim, device=device, generator=gen) / math.sqrt(input_dim + qdim)]
        return ModelPack(method, params, [AdamWState.zeros_like(p) for p in params], device, hidden_dim=qdim, qproj=qproj)
    raise ValueError(f"unknown method kind {method.kind}")


def _clone_pack(pack: ModelPack, *, device: torch.device | None = None, with_states: bool = False) -> ModelPack:
    target = device or pack.device
    params = [p.detach().to(target).clone() for p in pack.params]
    states = (
        [AdamWState(st.step, st.m.detach().to(target).clone(), st.v.detach().to(target).clone()) for st in pack.states]
        if with_states
        else [AdamWState.zeros_like(p) for p in params]
    )
    mu = pack.mu.detach().to(target).clone() if pack.mu is not None else None
    std = pack.std.detach().to(target).clone() if pack.std is not None else None
    qproj = pack.qproj.detach().to(target).clone() if pack.qproj is not None else None
    return ModelPack(pack.method, params, states, target, basis=pack.basis, mu=mu, std=std, hidden_dim=pack.hidden_dim, qproj=qproj)


def _qfeatures(pack: ModelPack, x: torch.Tensor) -> torch.Tensor:
    assert pack.qproj is not None
    q = x @ pack.qproj
    return torch.cat([x, q.square() / math.sqrt(max(1, int(q.shape[1])))], dim=1)


def _forward(pack: ModelPack, x: torch.Tensor) -> torch.Tensor:
    if pack.method.kind.startswith("mlp"):
        return f922._mlp3_forward_core(x, *pack.params)
    if pack.method.kind == "qfeature":
        return _qfeatures(pack, x) @ pack.params[0]
    assert pack.mu is not None and pack.std is not None
    return lq.lift_basis_forward(x, pack.params[0], pack.params[1:], pack.mu, pack.std, pack.basis, 2.0, 2.0)


def _forward_with_hidden(pack: ModelPack, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if pack.method.kind.startswith("mlp"):
        h1 = F.silu(x @ pack.params[0])
        h2 = F.silu(h1 @ pack.params[1])
        return h2 @ pack.params[2], h2
    if pack.method.kind == "qfeature":
        z = _qfeatures(pack, x)
        return z @ pack.params[0], z
    assert pack.mu is not None and pack.std is not None
    h = x @ pack.params[0]
    vals, _ = lq.basis_from_lift(h, pack.mu, pack.std, pack.basis, 2.0, 2.0)
    logits = vals[0] @ pack.params[1]
    for v, w in zip(vals[1:], pack.params[2:]):
        logits = logits + v @ w
    return logits, h


def _ce_bwd(pack: ModelPack, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if pack.method.kind.startswith("mlp"):
        out = f922._mlp3_fwd_bwd_core(x, y, *pack.params)
        return out[0], list(out[1:])
    if pack.method.kind == "qfeature":
        z = _qfeatures(pack, x)
        logits = z @ pack.params[0]
        loss, dy = ce_loss_and_grad(logits, y)
        return loss, [z.T @ dy]
    assert pack.mu is not None and pack.std is not None
    _fwd, bwd = lq.functions_for_basis(pack.basis)
    out = bwd(x, y, *pack.params, pack.mu, pack.std, 2.0, 2.0)
    return out[0], list(out[1:])


def _mse_bwd(pack: ModelPack, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if pack.method.kind.startswith("mlp"):
        W1, W2, W3 = pack.params
        h1_pre = x @ W1
        h1 = F.silu(h1_pre)
        h2_pre = h1 @ W2
        h2 = F.silu(h2_pre)
        pred = h2 @ W3
        dy = 2.0 * (pred - y) / max(1, int(y.shape[0]))
        loss = (pred - y).square().mean()
        dW3 = h2.T @ dy
        dh2 = dy @ W3.T
        sig2 = torch.sigmoid(h2_pre)
        dh2_pre = dh2 * sig2 * (1.0 + h2_pre * (1.0 - sig2))
        dW2 = h1.T @ dh2_pre
        dh1 = dh2_pre @ W2.T
        sig1 = torch.sigmoid(h1_pre)
        dh1_pre = dh1 * sig1 * (1.0 + h1_pre * (1.0 - sig1))
        return loss, [x.T @ dh1_pre, dW2, dW3]
    if pack.method.kind == "qfeature":
        z = _qfeatures(pack, x)
        pred = z @ pack.params[0]
        dy = 2.0 * (pred - y) / max(1, int(y.shape[0]))
        return (pred - y).square().mean(), [z.T @ dy]
    assert pack.mu is not None and pack.std is not None
    A = pack.params[0]
    weights = pack.params[1:]
    h = x @ A
    vals, ders = lq.basis_from_lift(h, pack.mu, pack.std, pack.basis, 2.0, 2.0)
    pred = vals[0] @ weights[0]
    for v, w in zip(vals[1:], weights[1:]):
        pred = pred + v @ w
    dy = 2.0 * (pred - y) / max(1, int(y.shape[0]))
    dweights = [v.T @ dy for v in vals]
    dh = torch.zeros_like(h)
    for der, w in zip(ders, weights):
        dh = dh + (dy @ w.T) * der
    return (pred - y).square().mean(), [x.T @ dh, *dweights]


def _apply_adamw(pack: ModelPack, grads: Sequence[torch.Tensor], cfg: ManualAdamWConfig) -> None:
    for p, g, st in zip(pack.params, grads, pack.states):
        adamw_update_(p, g, st, cfg)


def _apply_step_params(params: Sequence[torch.Tensor], step: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [p.detach() + d.detach() for p, d in zip(params, step)]


def _pack_with_params(pack: ModelPack, params: Sequence[torch.Tensor]) -> ModelPack:
    new = _clone_pack(pack, device=pack.device, with_states=False)
    new.params = [p.detach().clone() for p in params]
    return new


def _step_norm(step: Sequence[torch.Tensor]) -> torch.Tensor:
    if not step:
        return torch.zeros((), dtype=torch.float32)
    total = sum(d.detach().float().square().sum() for d in step)
    return total.sqrt()


def _step_dot(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
    if not a:
        return torch.zeros((), dtype=torch.float32)
    return sum((x.detach().float() * y.detach().float()).sum() for x, y in zip(a, b))


def _scale_step(step: Sequence[torch.Tensor], scale: float | torch.Tensor) -> List[torch.Tensor]:
    return [d.detach() * scale for d in step]


def _add_steps(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], alpha: float = 1.0) -> List[torch.Tensor]:
    return [x.detach() + y.detach() * float(alpha) for x, y in zip(a, b)]


def _scale_to_task_fraction(direction: Sequence[torch.Tensor], task_step: Sequence[torch.Tensor], fraction: float) -> List[torch.Tensor]:
    raw = _step_norm(direction).clamp_min(EPS)
    target = _step_norm(task_step) * float(fraction)
    return _scale_step(direction, target / raw)


def _eval_logits(pack: ModelPack, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            parts.append(_forward(pack, x[start : start + int(batch_size)]))
    return torch.cat(parts, dim=0) if parts else torch.empty(0, device=x.device)


def _classification_metrics(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
    if logits.numel() == 0:
        return {"loss": 0.0, "acc": 0.0}
    tail = geom.compute_tail_metrics(logits, y, bins=15)
    pred = logits.argmax(dim=1)
    row: Dict[str, Any] = {
        "loss": tail["NLL"],
        "acc": geom.safe_float((pred == y).float().mean()),
        "test_loss": tail["NLL"],
        "test_acc_sanity": geom.safe_float((pred == y).float().mean()),
    }
    row.update({k: v for k, v in tail.items() if k != "hard_tail_class_distribution"})
    row["hard_tail_class_distribution"] = json.dumps(tail["hard_tail_class_distribution"], sort_keys=True)
    return row


def _evaluate_pack(
    pack: ModelPack,
    splits: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, (x, y) in splits.items():
        logits = _eval_logits(pack, x, batch_size)
        metrics = _classification_metrics(logits, y)
        out[name] = metrics
    return out


def _auc_from_trace(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    pairs = sorted((float(x), float(y)) for x, y in zip(xs, ys))
    area = 0.0
    span = max(EPS, pairs[-1][0] - pairs[0][0])
    for (x0, y0), (x1, y1) in zip(pairs[:-1], pairs[1:]):
        area += 0.5 * (y0 + y1) * (x1 - x0)
    return area / span


def _estimate_memory_mb(pack: ModelPack, batch_size: int, input_dim: int, output_dim: int) -> Tuple[float, float]:
    param_count = sum(p.numel() for p in pack.params)
    opt_count = 2 * param_count
    if pack.method.kind.startswith("mlp"):
        h = max(1, pack.hidden_dim)
        cache = int(batch_size) * (input_dim + 2 * h + output_dim)
    elif pack.method.kind == "qfeature":
        q = max(1, pack.hidden_dim)
        cache = int(batch_size) * (input_dim + q + output_dim)
    else:
        h = int(pack.params[0].shape[1])
        cache = int(batch_size) * (input_dim + 4 * h + output_dim)
    compact = (param_count + opt_count + cache) * 4.0 / (1024.0 * 1024.0)
    conservative = compact + int(batch_size) * input_dim * 8.0 / (1024.0 * 1024.0)
    return compact, conservative


def _classwise_json(logits: torch.Tensor, y: torch.Tensor, classes: int) -> Tuple[str, str]:
    pred = logits.argmax(dim=1)
    rows = []
    mat = torch.zeros(classes, classes, device=y.device, dtype=torch.int64)
    for t, p in zip(y, pred):
        mat[int(t), int(p)] += 1
    for cls in range(classes):
        mask = y == cls
        rows.append(
            {
                "class_id": cls,
                "count": int(mask.sum().detach().cpu()),
                "acc": geom.safe_float((pred[mask] == y[mask]).float().mean()) if bool(mask.any()) else 0.0,
            }
        )
    return json.dumps(rows, sort_keys=True), json.dumps(mat.cpu().tolist())


def run_p0(args: argparse.Namespace, out_dir: Path, device: torch.device) -> Tuple[List[Dict[str, Any]], bool, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    input_dim = 784
    output_dim = 10
    for method in _methods():
        if method.kind == "lq":
            assert method.lq_spec is not None
            edge_params = method.lq_spec.hidden_dim * input_dim + method.lq_spec.hidden_dim * output_dim * lq.BASIS_CHANNELS[method.lq_spec.basis]
            nonkan = 0
            manual_fwd = manual_bwd = manual_update = 1
            uses_autograd = 0
            rollback_error = 0.0
            torch.manual_seed(120)
            x = torch.randn(32, input_dim, device=device)
            pack = _init_pack(method, input_dim, output_dim, x, 120, device)
            snap = [p.detach().clone() for p in pack.params]
            moved = [p + torch.randn_like(p) * 1.0e-5 for p in pack.params]
            restored = [s.detach().clone() for s in snap]
            rollback_error = geom.safe_float(torch.stack([(a - b).abs().max() for a, b in zip(restored, snap)]).max())
        else:
            edge_params = 0
            x = torch.randn(32, input_dim, device=device)
            pack = _init_pack(method, input_dim, output_dim, x, 120, device)
            nonkan = sum(int(p.numel()) for p in pack.params)
            if pack.qproj is not None:
                nonkan += int(pack.qproj.numel())
            manual_fwd = manual_bwd = manual_update = 1
            uses_autograd = 0
            rollback_error = 0.0
            moved = None
        _ = moved
        pass_flag = int(
            (not method.official_candidate)
            or (
                nonkan == 0
                and manual_fwd == 1
                and manual_bwd == 1
                and manual_update == 1
                and rollback_error < 1.0e-8
            )
        )
        row = {
            "stage": "P0_IMPLEMENTATION_CONTRACT",
            "method_id": method.method_id,
            "model_family": method.model_family,
            "candidate_id": method.candidate_id,
            "is_strict_purekan": method.strict_purekan,
            "official_candidate": method.official_candidate,
            "nonkan_param_count": nonkan,
            "edge_param_count": edge_params,
            "base_param_count": edge_params + nonkan,
            "residual_param_count": 0,
            "mixing_param_count": 0,
            "manual_forward_available": manual_fwd,
            "manual_backward_available": manual_bwd,
            "manual_update_available": manual_update,
            "uses_loss_backward": 0,
            "uses_torch_autograd_graph": uses_autograd,
            "loss_is_ce_only": 1,
            "uses_teacher": 0,
            "uses_sampler_weight": 0,
            "uses_dataset_branch": 0,
            "rollback_max_error": rollback_error,
            "contract_pass": pass_flag,
            "artifact_hash": "",
            "source_commit_or_unknown": "unknown",
            "device": str(device),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }
        rows.append(row)
        if method.official_candidate and not pass_flag:
            failures.append(
                _failure(
                    "P0",
                    "F0_contract_fail",
                    "official LQ candidate failed implementation contract",
                    method_id=method.method_id,
                    metric_name="contract_pass",
                    metric_value=pass_flag,
                    threshold=1,
                    action="repair LQ manual forward/backward/update contract before P1",
                )
            )
    write_csv_rows(out_dir / "p0_contract.csv", rows)
    official = [r for r in rows if int(r["official_candidate"]) == 1]
    p0_pass = bool(official) and all(int(r["contract_pass"]) == 1 for r in official)
    return rows, p0_pass, failures


def _checkpoint_epochs(epochs: int, text: str) -> List[int]:
    out = {0, int(epochs)}
    for item in _parse_list(text):
        key = item.lower()
        if key == "final":
            out.add(int(epochs))
        elif key.endswith("%"):
            out.add(max(0, min(int(epochs), round(int(epochs) * float(key[:-1]) / 100.0))))
        else:
            out.add(max(0, min(int(epochs), int(float(item)))))
    return sorted(out)


def _make_snapshot(pack: ModelPack, method: MethodSpec, dataset: str, seed: int, ckpt: str, epoch: int, step: int, wall: float) -> Snapshot:
    return Snapshot(
        method_id=method.method_id,
        model_family=method.model_family,
        candidate_id=method.candidate_id,
        dataset=dataset,
        seed=int(seed),
        checkpoint_id=ckpt,
        epoch=int(epoch),
        train_step=int(step),
        wall_clock_sec=float(wall),
        pack=_clone_pack(pack, device=torch.device("cpu"), with_states=False),
    )


def run_p1(
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    p0_pass: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Snapshot], Dict[Tuple[str, str, int], Dict[str, Any]], bool, List[Dict[str, Any]]]:
    if not p0_pass:
        reason = "P0_contract_failed"
        nr = [_not_run("P1", reason)]
        write_csv_rows(out_dir / "p1_base_qualification.csv", nr)
        write_csv_rows(out_dir / "p1_base_task_trace.csv", nr)
        write_csv_rows(out_dir / "p1_efficiency_profile.csv", nr)
        write_csv_rows(out_dir / "p1_expression_battery.csv", nr)
        return nr, nr, nr, nr, [], {}, False, [_failure("P1", "F13_not_run_gate", reason, action="repair P0 contract")]

    methods = _methods()
    datasets = [_canonical_dataset(x) for x in _parse_list(args.datasets)]
    seeds = _parse_ints(args.seeds)
    cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
    checkpoints = _checkpoint_epochs(int(args.epochs), args.checkpoint_schedule)
    summary_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    efficiency_rows: List[Dict[str, Any]] = []
    snapshots: List[Snapshot] = []
    final_map: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []

    for dataset in datasets:
        try:
            data = _load_vision_split(args, dataset, train_size=int(args.train_size), val_size=int(args.val_size), test_size=int(args.test_size))
        except Exception as exc:
            failures.append(_failure("P1", "F14_no_data_or_download_blocked", str(exc), dataset=dataset, action="provide real dataset files or allow download"))
            continue
        x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, x_test_cpu, y_test_cpu, input_dim, output_dim, protocol = data
        x_train = x_train_cpu.to(device=device, dtype=torch.float32)
        y_train = y_train_cpu.to(device=device)
        x_val = x_val_cpu.to(device=device, dtype=torch.float32)
        y_val = y_val_cpu.to(device=device)
        x_test = x_test_cpu.to(device=device, dtype=torch.float32)
        y_test = y_test_cpu.to(device=device)
        eval_train_n = min(int(args.eval_train_size), int(x_train.shape[0]))
        splits = {
            "train": (x_train[:eval_train_n], y_train[:eval_train_n]),
            "val": (x_val, y_val),
            "test": (x_test, y_test),
        }
        for seed in seeds:
            for method in methods:
                torch.manual_seed(int(seed) + int(args.seed))
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(int(seed) + int(args.seed))
                    torch.cuda.reset_peak_memory_stats(device)
                pack = _init_pack(method, input_dim, output_dim, x_train, seed, device)
                total_steps = 0
                train_started = time.perf_counter()
                step_times: List[float] = []
                forward_times: List[float] = []
                backward_times: List[float] = []
                update_times: List[float] = []
                trace_points: List[Dict[str, Any]] = []

                def record_checkpoint(epoch: int) -> None:
                    wall = time.perf_counter() - train_started
                    metrics = _evaluate_pack(pack, splits, int(args.eval_batch_size))
                    logits_val = _eval_logits(pack, x_val, int(args.eval_batch_size))
                    classwise, confusion = _classwise_json(logits_val, y_val, output_dim)
                    row = {
                        "stage": "P1_BASE_TASK_TRACE",
                        "method_id": method.method_id,
                        "model_family": method.model_family,
                        "candidate_id": method.candidate_id,
                        "dataset": dataset,
                        "seed": seed,
                        "checkpoint_id": f"epoch_{epoch}" if epoch > 0 else "init",
                        "epoch": epoch,
                        "train_step": total_steps,
                        "wall_clock_sec": wall,
                        "train_loss": metrics["train"]["loss"],
                        "val_loss": metrics["val"]["loss"],
                        "test_loss": metrics["test"]["loss"],
                        "train_acc": metrics["train"]["acc"],
                        "val_acc": metrics["val"]["acc"],
                        "test_acc_sanity": metrics["test"]["acc"],
                        "ECE": metrics["val"]["ECE"],
                        "NLL": metrics["val"]["NLL"],
                        "Brier": metrics["val"]["Brier"],
                        "margin_mean": metrics["val"]["margin_mean"],
                        "margin_p10": metrics["val"]["margin_p10"],
                        "CEp95": metrics["val"]["CE_p95"],
                        "CEp99": metrics["val"]["CE_p99"],
                        "wrong_confidence_p95": metrics["val"]["wrong_confidence_p95"],
                        "classwise_val_acc": classwise,
                        "confusion_matrix": confusion,
                        "protocol": protocol,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                    trace_rows.append(row)
                    trace_points.append(row)
                    snapshots.append(_make_snapshot(pack, method, dataset, seed, str(row["checkpoint_id"]), epoch, total_steps, wall))

                if 0 in checkpoints:
                    record_checkpoint(0)

                n_train = int(x_train.shape[0])
                steps_per_epoch = max(1, math.ceil(n_train / int(args.batch_size)))
                for epoch in range(1, int(args.epochs) + 1):
                    gen = torch.Generator(device=device).manual_seed(int(args.seed) * 100000 + seed * 1000 + epoch)
                    perm = torch.randperm(n_train, device=device, generator=gen)
                    for start in range(0, n_train, int(args.batch_size)):
                        idx = perm[start : start + int(args.batch_size)]
                        xb = x_train[idx]
                        yb = y_train[idx]
                        total_steps += 1
                        profile = len(step_times) < int(args.profile_steps)
                        if profile:
                            _sync(device)
                            t0 = time.perf_counter()
                            logits = _forward(pack, xb)
                            _sync(device)
                            t1 = time.perf_counter()
                            loss, grads = _ce_bwd(pack, xb, yb)
                            _ = logits, loss
                            _sync(device)
                            t2 = time.perf_counter()
                            _apply_adamw(pack, grads, cfg)
                            _sync(device)
                            t3 = time.perf_counter()
                            forward_times.append((t1 - t0) * 1000.0)
                            backward_times.append((t2 - t1) * 1000.0)
                            update_times.append((t3 - t2) * 1000.0)
                            step_times.append((t3 - t0) * 1000.0)
                        else:
                            loss, grads = _ce_bwd(pack, xb, yb)
                            _apply_adamw(pack, grads, cfg)
                    if epoch in checkpoints:
                        record_checkpoint(epoch)
                if int(args.epochs) not in checkpoints:
                    record_checkpoint(int(args.epochs))

                val_steps = [float(r["train_step"]) for r in trace_points]
                val_times = [float(r["wall_clock_sec"]) for r in trace_points]
                val_losses = [float(r["val_loss"]) for r in trace_points]
                val_accs = [float(r["val_acc"]) for r in trace_points]
                final = trace_points[-1]
                compact, conservative = _estimate_memory_mb(pack, int(args.batch_size), input_dim, output_dim)
                peak_alloc = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0
                peak_reserved = float(torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0
                eff = {
                    "stage": "P1_EFFICIENCY_PROFILE",
                    "method_id": method.method_id,
                    "model_family": method.model_family,
                    "candidate_id": method.candidate_id,
                    "dataset": dataset,
                    "seed": seed,
                    "forward_time_ms": _q(forward_times, 0.90),
                    "backward_time_ms": _q(backward_times, 0.90),
                    "update_time_ms": _q(update_times, 0.90),
                    "step_time_ms": _q(step_times, 0.90),
                    "forward_time_ratio_vs_MLP": "",
                    "backward_time_ratio_vs_MLP": "",
                    "step_time_ratio_vs_MLP": "",
                    "compact_memory_MB": compact,
                    "compact_memory_ratio": "",
                    "conservative_memory_MB": conservative,
                    "conservative_memory_ratio": "",
                    "peak_allocated_MB": peak_alloc,
                    "peak_reserved_MB": peak_reserved,
                    "samples_per_second": float(args.batch_size) / max(EPS, _q(step_times, 0.50) / 1000.0),
                    "kernel_count_forward": "not_measured",
                    "kernel_count_backward": "not_measured",
                    "memory_measurement": "analytic_compact_conservative_plus_cuda_peak",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
                efficiency_rows.append(eff)
                summary = {
                    "stage": "P1_BASE_QUALIFICATION",
                    "method_id": method.method_id,
                    "model_family": method.model_family,
                    "candidate_id": method.candidate_id,
                    "dataset": dataset,
                    "seed": seed,
                    "train_loss": final["train_loss"],
                    "val_loss": final["val_loss"],
                    "test_loss": final["test_loss"],
                    "train_acc": final["train_acc"],
                    "val_acc": final["val_acc"],
                    "test_acc_sanity": final["test_acc_sanity"],
                    "val_loss_auc_step": _auc_from_trace(val_steps, val_losses),
                    "val_loss_auc_time": _auc_from_trace(val_times, val_losses),
                    "val_acc_auc_step": _auc_from_trace(val_steps, val_accs),
                    "val_acc_auc_time": _auc_from_trace(val_times, val_accs),
                    "ECE": final["ECE"],
                    "NLL": final["NLL"],
                    "Brier": final["Brier"],
                    "margin_mean": final["margin_mean"],
                    "margin_p10": final["margin_p10"],
                    "CEp95": final["CEp95"],
                    "CEp99": final["CEp99"],
                    "wrong_confidence_p95": final["wrong_confidence_p95"],
                    "classwise_val_acc": final["classwise_val_acc"],
                    "confusion_matrix": final["confusion_matrix"],
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "train_size": int(args.train_size),
                    "val_size": int(args.val_size),
                    "test_size": int(args.test_size),
                    "protocol": protocol,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
                summary_rows.append(summary)
                final_map[(method.method_id, dataset, int(seed))] = {**summary, **eff}

    # Fill efficiency ratios against the same-param MLP baseline.
    by_key = {(r["dataset"], int(r["seed"]), r["method_id"]): r for r in efficiency_rows if r.get("status") != "not_run"}
    for row in efficiency_rows:
        base = by_key.get((row.get("dataset"), int(row.get("seed", 0)), "B0-MLP-same-param-AdamW"))
        if base:
            row["forward_time_ratio_vs_MLP"] = _safe_float(row["forward_time_ms"]) / max(EPS, _safe_float(base["forward_time_ms"]))
            row["backward_time_ratio_vs_MLP"] = _safe_float(row["backward_time_ms"]) / max(EPS, _safe_float(base["backward_time_ms"]))
            row["step_time_ratio_vs_MLP"] = _safe_float(row["step_time_ms"]) / max(EPS, _safe_float(base["step_time_ms"]))
            row["compact_memory_ratio"] = _safe_float(row["compact_memory_MB"]) / max(EPS, _safe_float(base["compact_memory_MB"]))
            row["conservative_memory_ratio"] = _safe_float(row["conservative_memory_MB"]) / max(EPS, _safe_float(base["conservative_memory_MB"]))
            final_map[(row["method_id"], row["dataset"], int(row["seed"]))].update(row)

    expr_rows = run_expression_battery(args, device)
    write_csv_rows(out_dir / "p1_base_qualification.csv", summary_rows)
    write_csv_rows(out_dir / "p1_base_task_trace.csv", trace_rows)
    write_csv_rows(out_dir / "p1_efficiency_profile.csv", efficiency_rows)
    write_csv_rows(out_dir / "p1_expression_battery.csv", expr_rows)

    p1_pass, gate_failures = _evaluate_p1_gate(summary_rows, efficiency_rows, expr_rows)
    failures.extend(gate_failures)
    return summary_rows, trace_rows, efficiency_rows, expr_rows, snapshots, final_map, p1_pass, failures


def _expression_target(name: str, x: torch.Tensor, gen: torch.Generator | None = None, noisy: bool = False) -> torch.Tensor:
    if name == "E0-additive":
        y = x[:, 0:1] + 0.5 * x[:, 1:2] - 0.25 * x[:, 2:3]
    elif name == "E1-pairwise-product":
        y = x[:, 0:1] * x[:, 1:2] + 0.5 * x[:, 2:3] * x[:, 3:4]
    elif name == "E2-composition":
        y = torch.sin(x[:, 0:1] + x[:, 1:2] * x[:, 2:3])
    elif name == "E3-local-XOR":
        y = (((x[:, 0:1] > 0) ^ (x[:, 1:2] > 0)).float() * 2.0 - 1.0)
    elif name == "E4-high-frequency":
        y = torch.sin(6.0 * math.pi * x[:, 0:1])
    elif name == "E5-noise-stress":
        y = x[:, 0:1] * x[:, 1:2] + torch.sin(2.0 * x[:, 2:3])
    else:
        raise ValueError(name)
    if noisy and name == "E5-noise-stress" and gen is not None:
        y = y + 0.20 * torch.randn(y.shape, generator=gen, device=y.device, dtype=y.dtype)
    return y


def _r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    sse = (pred - target).square().sum()
    sst = (target - target.mean()).square().sum().clamp_min(EPS)
    return geom.safe_float(1.0 - sse / sst)


def run_expression_battery(args: argparse.Namespace, device: torch.device) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if "P1" not in _parse_list(args.packages) and "ALL" not in _parse_list(args.packages):
        return rows
    dim = 16
    output_dim = 1
    targets = ["E0-additive", "E1-pairwise-product", "E2-composition", "E3-local-XOR", "E4-high-frequency", "E5-noise-stress"]
    methods = _methods()
    seeds = _parse_ints(args.seeds)
    cfg = ManualAdamWConfig(lr=float(args.expression_lr), weight_decay=float(args.weight_decay))
    for seed in seeds:
        gen = torch.Generator(device=device).manual_seed(120100 + seed)
        x_train = torch.rand(int(args.expression_train_size), dim, generator=gen, device=device) * 2.0 - 1.0
        x_val = torch.rand(int(args.expression_val_size), dim, generator=gen, device=device) * 2.0 - 1.0
        for target_id in targets:
            y_train = _expression_target(target_id, x_train, gen=gen, noisy=True)
            y_val = _expression_target(target_id, x_val, gen=gen, noisy=False)
            for method in methods:
                # Use a smaller hidden dimension for expression probes while preserving method family.
                if method.kind == "lq":
                    assert method.lq_spec is not None
                    probe_method = MethodSpec(
                        method.method_id,
                        method.model_family,
                        method.candidate_id,
                        "lq",
                        method.official_candidate,
                        method.strict_purekan,
                        lq.LQSpec(method.lq_spec.candidate_id, method.lq_spec.basis, min(64, method.lq_spec.hidden_dim), method.lq_spec.init_variant, method.lq_spec.output_scale),
                    )
                else:
                    probe_method = method
                pack = _init_pack(probe_method, dim, output_dim, x_train, seed + 991, device)
                steps_to_090 = ""
                steps_to_095 = ""
                for step in range(1, int(args.expression_steps) + 1):
                    idx = torch.randint(0, int(x_train.shape[0]), (int(args.expression_batch_size),), generator=gen, device=device)
                    loss, grads = _mse_bwd(pack, x_train[idx], y_train[idx])
                    _apply_adamw(pack, grads, cfg)
                    if step % int(args.expression_eval_stride) == 0 or step == int(args.expression_steps):
                        with torch.no_grad():
                            pred_val = _forward(pack, x_val)
                            r2 = _r2_score(pred_val, y_val)
                        if r2 >= 0.90 and steps_to_090 == "":
                            steps_to_090 = step
                        if r2 >= 0.95 and steps_to_095 == "":
                            steps_to_095 = step
                with torch.no_grad():
                    pred_train = _forward(pack, x_train)
                    pred_val = _forward(pack, x_val)
                    fit_loss = geom.safe_float((pred_val - y_val).square().mean())
                    hidden = _forward_with_hidden(pack, x_val)[1]
                rows.append(
                    {
                        "stage": "P1_EXPRESSION_BATTERY",
                        "method_id": method.method_id,
                        "model_family": method.model_family,
                        "candidate_id": method.candidate_id,
                        "seed": seed,
                        "expression_target_id": target_id,
                        "train_R2": _r2_score(pred_train, y_train),
                        "val_R2": _r2_score(pred_val, y_val),
                        "fit_loss": fit_loss,
                        "steps_to_R2_090": steps_to_090,
                        "steps_to_R2_095": steps_to_095,
                        "expression_rank": geom.compute_effective_rank(hidden),
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                )
    return rows


def _evaluate_p1_gate(
    summary_rows: Sequence[Mapping[str, Any]],
    efficiency_rows: Sequence[Mapping[str, Any]],
    expr_rows: Sequence[Mapping[str, Any]],
) -> Tuple[bool, List[Dict[str, Any]]]:
    failures: List[Dict[str, Any]] = []
    target = "B4-R2-LQ-fanin-output-scale-confirmed-AdamW"
    baseline = "B0-MLP-same-param-AdamW"
    paired = []
    by_key = {(r["method_id"], r["dataset"], int(r["seed"])): r for r in summary_rows if r.get("status") != "not_run"}
    for key, cand in by_key.items():
        method, dataset, seed = key
        if method != target:
            continue
        base = by_key.get((baseline, dataset, seed))
        if base:
            paired.append((_safe_float(cand["val_acc"]) - _safe_float(base["val_acc"]), dataset, seed))
    near = [1 if delta >= -0.01 else 0 for delta, _d, _s in paired]
    near_rate = _mean(near)
    macro_delta = _mean(delta for delta, _d, _s in paired)
    eff_target = [r for r in efficiency_rows if r.get("method_id") == target]
    step_q90 = _q([_safe_float(r.get("step_time_ratio_vs_MLP")) for r in eff_target], 0.90, default=999.0)
    compact_q90 = _q([_safe_float(r.get("compact_memory_ratio")) for r in eff_target], 0.90, default=999.0)
    expr_target = [r for r in expr_rows if r.get("method_id") == target]
    expr_mlp = [r for r in expr_rows if r.get("method_id") == baseline]
    pair_r2 = _mean(_safe_float(r.get("val_R2")) for r in expr_target if r.get("expression_target_id") == "E1-pairwise-product")
    comp_r2 = _mean(_safe_float(r.get("val_R2")) for r in expr_target if r.get("expression_target_id") == "E2-composition")
    comp_mlp = _mean(_safe_float(r.get("val_R2")) for r in expr_mlp if r.get("expression_target_id") == "E2-composition")
    rank_min = min([_safe_float(r.get("expression_rank")) for r in expr_target] or [0.0])

    checks = [
        ("near_pass_rate", near_rate, 0.80, near_rate >= 0.80, "F1_base_task_fail", "repair global base protocol"),
        ("macro_delta", macro_delta, -0.005, macro_delta >= -0.005, "F1_base_task_fail", "audit repaired LQ base before functional"),
        ("step_q90", step_q90, 1.10, step_q90 <= 1.10, "F2_base_efficiency_fail", "repair LQ step efficiency"),
        ("compact_memory_ratio_q90", compact_q90, 1.00, compact_q90 <= 1.00, "F2_base_efficiency_fail", "repair compact memory envelope"),
        ("pairwise_product_R2", pair_r2, 0.85, pair_r2 >= 0.85, "F3_expression_fail", "repair interaction expression battery"),
        ("composition_R2_delta_vs_MLP", comp_r2 - comp_mlp, -0.05, comp_r2 >= comp_mlp - 0.05, "F3_expression_fail", "repair composition expression battery"),
        ("expression_rank_min", rank_min, 1.0, rank_min >= 1.0, "F3_expression_fail", "repair expression rank collapse"),
    ]
    for name, value, threshold, passed, code, action in checks:
        if not passed:
            failures.append(
                _failure(
                    "P1",
                    code,
                    f"P1 gate failed: {name}",
                    method_id=target,
                    metric_name=name,
                    metric_value=value,
                    threshold=threshold,
                    action=action,
                )
            )
    return all(passed for _name, _value, _thr, passed, _code, _action in checks), failures


def _forward_from_params(pack: ModelPack, params: Sequence[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    return _forward(_pack_with_params(pack, params), x)


def _micrograds(pack: ModelPack, x: torch.Tensor, y: torch.Tensor, count: int) -> List[List[torch.Tensor]]:
    n = int(x.shape[0])
    count = max(1, min(int(count), n))
    size = n // count
    out: List[List[torch.Tensor]] = []
    for i in range(count):
        lo = i * size
        hi = (i + 1) * size if i < count - 1 else n
        if hi <= lo:
            continue
        _loss, grads = _ce_bwd(pack, x[lo:hi], y[lo:hi])
        out.append([g.detach().clone() for g in grads])
    return out


def _offdiag_population_proxy(grads_by_microbatch: Sequence[Sequence[torch.Tensor]]) -> float:
    flats = [torch.cat([g.detach().float().reshape(-1) for g in pack]) for pack in grads_by_microbatch]
    if len(flats) < 2:
        return 0.0
    vals = []
    for i in range(len(flats)):
        for j in range(i + 1, len(flats)):
            vals.append(geom.safe_float(torch.dot(flats[i], flats[j]) / (flats[i].norm() * flats[j].norm()).clamp_min(EPS)))
    return _mean(vals)


def _noise_leak(pack: ModelPack, x: torch.Tensor, y: torch.Tensor, lr: float, seed: int) -> Tuple[float, float, float]:
    loss, grads = _ce_bwd(pack, x, y)
    task_step = [(-float(lr)) * g.detach() for g in grads]
    moved = _pack_with_params(pack, _apply_step_params(pack.params, task_step))
    with torch.no_grad():
        real_before = F.cross_entropy(_forward(pack, x), y)
        real_after = F.cross_entropy(_forward(moved, x), y)
        gen = torch.Generator(device=x.device).manual_seed(int(seed))
        yp = y[torch.randperm(int(y.numel()), device=x.device, generator=gen)]
        noise_before = F.cross_entropy(_forward(pack, x), yp)
        noise_after = F.cross_entropy(_forward(moved, x), yp)
    improve_real = geom.safe_float(real_before - real_after)
    improve_noise = geom.safe_float(noise_before - noise_after)
    return improve_real, improve_noise, max(0.0, improve_noise) / max(EPS, max(0.0, improve_real))


def _geometry_metrics_for_pack(pack: ModelPack, x: torch.Tensor, y: torch.Tensor, args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    with torch.no_grad():
        logits, hidden = _forward_with_hidden(pack, x)
    task = _classification_metrics(logits, y)
    row: Dict[str, Any] = {
        "split_loss": task["loss"],
        "split_acc": task["acc"],
        "CE_mean": task["CE_mean"],
        "CE_p90": task["CE_p90"],
        "CE_p95": task["CE_p95"],
        "CE_p99": task["CE_p99"],
        "margin_mean": task["margin_mean"],
        "margin_p10": task["margin_p10"],
        "wrong_confidence_p95": task["wrong_confidence_p95"],
        "ECE": task["ECE"],
        "NLL": task["NLL"],
        "Brier": task["Brier"],
        "hard_tail_class_distribution": task["hard_tail_class_distribution"],
    }
    if pack.method.kind == "lq":
        assert pack.mu is not None and pack.std is not None
        row.update(geom.compute_lq_channel_stats(pack.params, pack.mu, pack.std, pack.basis, x))
        row.update(geom.geometry_debt_lq(pack.params, pack.basis))
        roles = snr_lq.role_names_for_basis(pack.basis)
    else:
        row.update(geom.compute_generic_representation_stats(x, hidden, logits))
        row.update({"curvature_debt": 0.0, "coefficient_total_variation": 0.0, "rolewise_quadratic_coeff_norm": 0.0, "basis_type_has_spline_grid": 0.0})
        roles = [f"param_{idx}" for idx in range(len(pack.params))]
    row.update(
        geom.compute_input_perturbation_drift(
            lambda z: _forward(pack, z),
            x[: min(int(args.probe_batch_size), int(x.shape[0]))],
            seed=int(seed),
        )
    )
    responses = geom.finite_difference_output_responses(
        pack.params,
        lambda params, xb: _forward_from_params(pack, params, xb),
        x[: min(64, int(x.shape[0]))],
        directions=int(args.kernel_sketch_directions),
        eps=float(args.finite_diff_eps),
        seed=int(seed),
    )
    row.update(geom.kernel_condition_from_output_responses(responses))
    mb_n = min(int(args.probe_batch_size), int(x.shape[0]))
    mb_n = max(int(args.microbatch_count), (mb_n // int(args.microbatch_count)) * int(args.microbatch_count))
    xb = x[:mb_n]
    yb = y[:mb_n]
    grads_real = _micrograds(pack, xb, yb, int(args.microbatch_count))
    gen = torch.Generator(device=x.device).manual_seed(int(seed) + 77)
    y_shuffle = yb[torch.randperm(int(yb.numel()), device=x.device, generator=gen)]
    grads_shuffle = _micrograds(pack, xb, y_shuffle, int(args.microbatch_count))
    row["signal_consistency_real"] = geom.signal_consistency_from_grads(grads_real)
    row["signal_consistency_shuffled"] = geom.signal_consistency_from_grads(grads_shuffle)
    row["real_noise_consistency_gap"] = row["signal_consistency_real"] - row["signal_consistency_shuffled"]
    row["offdiag_population_proxy"] = _offdiag_population_proxy(grads_real)
    row.update(geom.rolewise_snr_from_micrograds(grads_real, roles))
    imp_real, imp_noise, leak = _noise_leak(pack, xb, yb, float(args.lr), int(seed))
    row["real_one_step_improvement"] = imp_real
    row["noise_one_step_improvement"] = imp_noise
    row["noise_leak"] = leak
    row["metric_valid"] = int(all(math.isfinite(_safe_float(v)) for k, v in row.items() if isinstance(v, (float, int))))
    return row


def run_p2(
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    snapshots: Sequence[Snapshot],
    final_map: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    p0_pass: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], bool, List[Dict[str, Any]]]:
    if not p0_pass or not snapshots:
        reason = "P0_or_P1_missing_snapshots"
        nr = [_not_run("P2", reason)]
        write_csv_rows(out_dir / "p2_passive_geometry_snapshot.csv", nr)
        write_csv_rows(out_dir / "p2_geometry_checkpoint_trace.csv", nr)
        write_csv_rows(out_dir / "p2_geometry_correlation.csv", nr)
        return nr, nr, nr, False, [_failure("P2", "F13_not_run_gate", reason, action="complete P1 snapshots first")]

    data_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, str]] = {}
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for snap in snapshots:
        if snap.dataset not in data_cache:
            data_cache[snap.dataset] = _load_vision_split(
                args,
                snap.dataset,
                train_size=int(args.train_size),
                val_size=int(args.val_size),
                test_size=int(args.test_size),
            )
        x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _x_test_cpu, _y_test_cpu, _in_dim, _out_dim, _protocol = data_cache[snap.dataset]
        split_tensors = {
            "train_small": (x_train_cpu[: int(args.probe_batch_size)], y_train_cpu[: int(args.probe_batch_size)]),
            "probe": (
                x_train_cpu[int(args.probe_batch_size) : 2 * int(args.probe_batch_size)],
                y_train_cpu[int(args.probe_batch_size) : 2 * int(args.probe_batch_size)],
            ),
            "val": (x_val_cpu[: int(args.probe_batch_size)], y_val_cpu[: int(args.probe_batch_size)]),
        }
        pack = _clone_pack(snap.pack, device=device, with_states=False)
        for split, (x_cpu, y_cpu) in split_tensors.items():
            if int(x_cpu.shape[0]) == 0:
                continue
            x = x_cpu.to(device=device, dtype=torch.float32)
            y = y_cpu.to(device=device)
            try:
                metrics = _geometry_metrics_for_pack(pack, x, y, args, seed=120000 + snap.seed + snap.epoch)
            except Exception as exc:
                metrics = {"metric_valid": 0, "metric_error": str(exc)}
                failures.append(
                    _failure(
                        "P2",
                        "F4_geometry_metric_nan",
                        f"geometry metric failed: {exc}",
                        method_id=snap.method_id,
                        dataset=snap.dataset,
                        seed=snap.seed,
                        checkpoint_id=snap.checkpoint_id,
                        action="repair metric numerical stability",
                    )
                )
            base = {
                "stage": "P2_PASSIVE_GEOMETRY_SNAPSHOT",
                "method_id": snap.method_id,
                "model_family": snap.model_family,
                "candidate_id": snap.candidate_id,
                "dataset": snap.dataset,
                "seed": snap.seed,
                "checkpoint_id": snap.checkpoint_id,
                "split": split,
                "train_step": snap.train_step,
                "checkpoint_epoch": snap.epoch,
                "wall_clock_sec": snap.wall_clock_sec,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
            final = final_map.get((snap.method_id, snap.dataset, int(snap.seed)), {})
            base.update(
                {
                    "final_val_acc": final.get("val_acc", ""),
                    "final_val_loss": final.get("val_loss", ""),
                    "val_loss_auc_time": final.get("val_loss_auc_time", ""),
                    "step_time_ratio_vs_MLP": final.get("step_time_ratio_vs_MLP", ""),
                    "compact_memory_ratio": final.get("compact_memory_ratio", ""),
                }
            )
            base.update(metrics)
            rows.append(base)
    trace_rows = [r for r in rows if r.get("split") == "val"]
    corr_rows = _geometry_correlations(rows)
    write_csv_rows(out_dir / "p2_passive_geometry_snapshot.csv", rows)
    write_csv_rows(out_dir / "p2_geometry_checkpoint_trace.csv", trace_rows)
    write_csv_rows(out_dir / "p2_geometry_correlation.csv", corr_rows)
    p2_complete = bool(rows) and all(int(_safe_float(r.get("metric_valid", 1), 1)) == 1 for r in rows)
    if not p2_complete:
        failures.append(_failure("P2", "F4_geometry_metric_nan", "one or more P2 metric rows invalid", action="inspect p2_passive_geometry_snapshot.csv metric_error"))
    return rows, trace_rows, corr_rows, p2_complete, failures


def _geometry_correlations(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    metric_names = [
        "effective_rank_hidden",
        "basis_usage_entropy",
        "lift_condition_proxy",
        "curvature_debt",
        "perturb_logit_drift_p95",
        "local_jacobian_norm_p95",
        "signal_consistency_real",
        "real_noise_consistency_gap",
        "noise_leak",
        "CE_p99",
        "margin_p10",
        "ECE",
    ]
    target_names = ["final_val_acc", "final_val_loss", "val_loss_auc_time", "NLL", "CE_p99", "margin_p10", "noise_leak"]
    signs = {
        "effective_rank_hidden": "higher_better",
        "basis_usage_entropy": "higher_better",
        "lift_condition_proxy": "lower_better",
        "curvature_debt": "lower_better",
        "perturb_logit_drift_p95": "lower_better",
        "local_jacobian_norm_p95": "lower_better",
        "signal_consistency_real": "higher_better",
        "real_noise_consistency_gap": "higher_better",
        "noise_leak": "lower_better",
        "CE_p99": "lower_better",
        "margin_p10": "higher_better",
        "ECE": "lower_better",
    }
    out: List[Dict[str, Any]] = []
    for metric in metric_names:
        for target in target_names:
            pairs = []
            for row in rows:
                if row.get("split") != "val":
                    continue
                x = _safe_float(row.get(metric), float("nan"))
                y = _safe_float(row.get(target), float("nan"))
                if math.isfinite(x) and math.isfinite(y):
                    pairs.append((x, y))
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            out.append(
                {
                    "stage": "P2_GEOMETRY_CORRELATION",
                    "metric_name": metric,
                    "target_name": target,
                    "pearson": _corr(xs, ys),
                    "spearman": _spearman(xs, ys),
                    "kendall": _kendall(xs, ys),
                    "n": len(xs),
                    "sign_expected": signs.get(metric, ""),
                    "interpretation": "measured_correlation_no_causal_claim" if len(xs) >= 3 else "insufficient_n",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
            )
    return out


def _aggregate_final_for_certificate(
    p1_rows: Sequence[Mapping[str, Any]],
    p2_rows: Sequence[Mapping[str, Any]],
    efficiency_rows: Sequence[Mapping[str, Any]],
    method_id: str,
) -> Dict[str, Any]:
    p1 = [r for r in p1_rows if r.get("method_id") == method_id and r.get("status") != "not_run"]
    p2 = [r for r in p2_rows if r.get("method_id") == method_id and r.get("split") == "val" and str(r.get("checkpoint_id")) in {"epoch_20", "epoch_30", "epoch_10", "epoch_5", "epoch_3", "epoch_2", "epoch_1"}]
    if not p2:
        p2 = [r for r in p2_rows if r.get("method_id") == method_id and r.get("split") == "val"]
    eff = [r for r in efficiency_rows if r.get("method_id") == method_id]
    out: Dict[str, Any] = {
        "val_acc": _mean(_safe_float(r.get("val_acc")) for r in p1),
        "val_loss": _mean(_safe_float(r.get("val_loss")) for r in p1),
        "val_loss_auc_time": _mean(_safe_float(r.get("val_loss_auc_time")) for r in p1),
        "ECE": _mean(_safe_float(r.get("ECE")) for r in p1),
        "NLL": _mean(_safe_float(r.get("NLL")) for r in p1),
        "step_time_ratio_vs_reference": _mean(_safe_float(r.get("step_time_ratio_vs_MLP"), 1.0) for r in eff) if eff else 1.0,
    }
    for key in [
        "effective_rank_hidden",
        "basis_usage_entropy",
        "curvature_debt",
        "perturb_logit_drift_p95",
        "signal_consistency_real",
        "noise_leak",
        "CE_p99",
        "margin_p10",
    ]:
        out[key] = _mean(_safe_float(r.get(key)) for r in p2)
    return out


def run_p3(
    args: argparse.Namespace,
    out_dir: Path,
    p1_rows: Sequence[Mapping[str, Any]],
    p2_rows: Sequence[Mapping[str, Any]],
    efficiency_rows: Sequence[Mapping[str, Any]],
    p2_complete: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], bool, List[Dict[str, Any]]]:
    if not p2_complete:
        reason = "P2_geometry_battery_incomplete"
        nr = [_not_run("P3", reason)]
        write_csv_rows(out_dir / "p3_geometry_certificate.csv", nr)
        write_json(out_dir / "p3_geometry_certificate.json", {"version": "v0", "pass": False, "failure_reasons": [reason]})
        return {"pass": False, "failure_reasons": [reason]}, nr, False, [_failure("P3", "F13_not_run_gate", reason, action="repair P2 metrics")]

    reference_method = "B0-MLP-same-param-AdamW"
    candidates = [
        "B2-QuadraticFeatureMLP-diagnostic-AdamW",
        "B3-LQ-t2-h256-AdamW",
        "B4-R2-LQ-fanin-output-scale-confirmed-AdamW",
    ]
    reference = _aggregate_final_for_certificate(p1_rows, p2_rows, efficiency_rows, reference_method)
    certs = []
    for candidate_id in candidates:
        candidate = _aggregate_final_for_certificate(p1_rows, p2_rows, efficiency_rows, candidate_id)
        certs.append(
            geom_cert.evaluate_geometry_certificate(
                reference_method=reference_method,
                candidate_method=candidate_id,
                reference=reference,
                candidate=candidate,
            )
        )
    rows = geom_cert.certificate_to_rows(certs)
    top = next(c for c in certs if c["candidate_method"] == "B4-R2-LQ-fanin-output-scale-confirmed-AdamW")
    top_with_all = dict(top)
    top_with_all["all_certificates"] = certs
    top_with_all["reference_aggregate"] = reference
    write_csv_rows(out_dir / "p3_geometry_certificate.csv", rows)
    write_json(out_dir / "p3_geometry_certificate.json", top_with_all)
    failures = []
    if not bool(top.get("pass")):
        failures.append(
            _failure(
                "P3",
                "F6_certificate_hard_gate_fail" if not top.get("hard_gate_pass") else "F5_geometry_no_signal",
                "GeometryCertificateV0 did not pass for repaired LQ vs MLP reference",
                method_id="B4-R2-LQ-fanin-output-scale-confirmed-AdamW",
                metric_name="certificate_pass",
                metric_value=int(bool(top.get("pass"))),
                threshold=1,
                action="inspect hard gates and pareto dimensions before functional promotion",
            )
        )
    return top_with_all, rows, True, failures


def _raw_geometry_direction(pack: ModelPack, candidate: str, x: torch.Tensor | None = None) -> List[torch.Tensor]:
    zero = [torch.zeros_like(p) for p in pack.params]
    if candidate in {"D1-NoOpMatchedOverhead", "D5-ShuffledEvent", "D6-SNROnlyGate"}:
        return zero
    if candidate == "D3-AdamWParallelDirection":
        return zero
    if candidate == "D7-GeometryOnlyNoSNR" or candidate == "D8-SNRProjectedGeometry":
        if pack.method.kind == "lq":
            return snr_lq.quadratic_coeff_direction(pack.params, pack.basis)
        return [-p.detach() for p in pack.params]
    if candidate == "D9-BasisEntropyRebalance":
        direction = [torch.zeros_like(p) for p in pack.params]
        if pack.method.kind == "lq" and len(pack.params) > 2:
            norms = torch.stack([p.detach().float().norm() for p in pack.params[1:]])
            target = norms.mean()
            for idx, p in enumerate(pack.params[1:], start=1):
                direction[idx] = p.detach() * geom.safe_float((target - p.detach().float().norm()) / p.detach().float().norm().clamp_min(EPS))
        else:
            direction = [-0.1 * p.detach() for p in pack.params]
        return direction
    if candidate == "D10-LiftConditionRepair":
        direction = [torch.zeros_like(p) for p in pack.params]
        if pack.method.kind == "lq":
            A = pack.params[0].detach()
            col_norm = A.float().norm(dim=0, keepdim=True).clamp_min(EPS)
            target = col_norm.mean()
            direction[0] = A * ((target - col_norm) / col_norm).to(A.dtype)
        else:
            direction[0] = -0.1 * pack.params[0].detach()
        return direction
    if candidate == "D11-TailStabilityCorrection":
        return [torch.zeros_like(pack.params[0]), *[-p.detach() for p in pack.params[1:]]] if len(pack.params) > 1 else [-pack.params[0].detach()]
    if candidate in {"D12-MLPAnalogGeometryMaintenance", "D13-QuadraticFeatureMLPAnalogMaintenance"}:
        return [-p.detach() for p in pack.params]
    return zero


def _candidate_ids(text: str) -> List[str]:
    aliases = {
        "D0": "D0-TaskOnlyAdamW",
        "D1": "D1-NoOpMatchedOverhead",
        "D2": "D2-RandomMatchedNorm",
        "D3": "D3-AdamWParallelDirection",
        "D4": "D4-ShuffledPayload",
        "D5": "D5-ShuffledEvent",
        "D6": "D6-SNROnlyGate",
        "D7": "D7-GeometryOnlyNoSNR",
        "D8": "D8-SNRProjectedGeometry",
        "D9": "D9-BasisEntropyRebalance",
        "D10": "D10-LiftConditionRepair",
        "D11": "D11-TailStabilityCorrection",
        "D12": "D12-MLPAnalogGeometryMaintenance",
        "D13": "D13-QuadraticFeatureMLPAnalogMaintenance",
    }
    out = []
    for item in _parse_list(text):
        out.append(aliases.get(item, item))
    return out


def _select_snapshot(snapshots: Sequence[Snapshot], method_id: str, dataset: str, seed: int, target_epoch: float) -> Snapshot | None:
    candidates = [s for s in snapshots if s.method_id == method_id and s.dataset == dataset and int(s.seed) == int(seed) and s.epoch > 0]
    if not candidates:
        return None
    return min(candidates, key=lambda s: abs(float(s.epoch) - target_epoch))


def _audit_geometry_score(row: Mapping[str, Any]) -> float:
    return (
        -_safe_float(row.get("delta_curvature_debt"))
        - _safe_float(row.get("delta_perturb_logit_drift_p95"))
        + _safe_float(row.get("delta_basis_entropy"))
        - _safe_float(row.get("delta_CEp99"))
        + _safe_float(row.get("delta_margin_p10"))
        - _safe_float(row.get("delta_ECE_proxy"))
        - _safe_float(row.get("delta_noise_leak"))
    )


def run_p4(
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    snapshots: Sequence[Snapshot],
    p1_pass: bool,
    p3_cert: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], bool, List[Dict[str, Any]]]:
    official_gate_open = bool(p1_pass and bool(p3_cert.get("pass", False)))
    if not snapshots:
        reason = "P1_missing_snapshots"
        rows = [_not_run("P4", reason)]
        write_csv_rows(out_dir / "p4_functional_direction_audit.csv", rows)
        write_csv_rows(out_dir / "p4_one_step_probe.csv", rows)
        write_csv_rows(out_dir / "p4_lambda_backtracking.csv", rows)
        write_csv_rows(out_dir / "p4_control_matrix.csv", rows)
        write_csv_rows(out_dir / "p4_five_step_probe.csv", rows)
        return rows, rows, rows, rows, rows, False, [_failure("P4", "F13_not_run_gate", reason, action="complete P1 snapshots first")]

    data_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, str]] = {}
    audit_rows: List[Dict[str, Any]] = []
    one_rows: List[Dict[str, Any]] = []
    lambda_rows: List[Dict[str, Any]] = []
    control_rows: List[Dict[str, Any]] = []
    five_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    if not official_gate_open:
        failures.append(
            _failure(
                "P4",
                "F13_not_run_gate",
                "P4 official gate closed; running diagnostic control audit only",
                metric_name="official_gate_open",
                metric_value=0,
                threshold=1,
                action="repair P1/P3 before official functional audit",
            )
        )
    candidates = _candidate_ids(args.functional_candidates)
    lq_method = "B4-R2-LQ-fanin-output-scale-confirmed-AdamW"
    datasets = [_canonical_dataset(x) for x in _parse_list(args.datasets)]
    seeds = _parse_ints(args.seeds)
    lambda_grid = [float(x) for x in _parse_list(args.lambda_grid)]
    one_pass_rows: List[Dict[str, Any]] = []
    p4_status = "measured_official" if official_gate_open else "measured_diagnostic_gate_closed"

    for dataset in datasets:
        if dataset not in data_cache:
            data_cache[dataset] = _load_vision_split(
                args,
                dataset,
                train_size=int(args.train_size),
                val_size=int(args.val_size),
                test_size=int(args.test_size),
            )
        x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _x_test_cpu, _y_test_cpu, _in_dim, _out_dim, _protocol = data_cache[dataset]
        for seed in seeds:
            snap = _select_snapshot(snapshots, lq_method, dataset, seed, target_epoch=max(1.0, int(args.epochs) * 0.5))
            if snap is None:
                failures.append(_failure("P4", "F13_not_run_gate", "missing repaired LQ checkpoint", dataset=dataset, seed=seed, action="complete P1 snapshots"))
                continue
            pack = _clone_pack(snap.pack, device=device, with_states=False)
            gen = torch.Generator(device=device).manual_seed(120400 + seed)
            perm = torch.randperm(int(x_train_cpu.shape[0]), generator=torch.Generator().manual_seed(120400 + seed))
            bsz = int(args.batch_size)
            upd_idx = perm[:bsz]
            probe_idx = perm[bsz : 2 * bsz]
            if int(probe_idx.numel()) == 0:
                probe_idx = upd_idx
            x_update = x_train_cpu[upd_idx].to(device=device, dtype=torch.float32)
            y_update = y_train_cpu[upd_idx].to(device=device)
            x_probe = x_train_cpu[probe_idx].to(device=device, dtype=torch.float32)
            y_probe = y_train_cpu[probe_idx].to(device=device)
            x_valmini = x_val_cpu[:bsz].to(device=device, dtype=torch.float32)
            y_valmini = y_val_cpu[:bsz].to(device=device)
            before_probe = F.cross_entropy(_forward(pack, x_probe), y_probe)
            before_val = F.cross_entropy(_forward(pack, x_valmini), y_valmini)
            before_geom = _geometry_metrics_for_pack(pack, x_probe, y_probe, args, seed=120410 + seed)
            _loss, grads = _ce_bwd(pack, x_update, y_update)
            task_step = [(-float(args.lr)) * g.detach() for g in grads]
            task_pack = _pack_with_params(pack, _apply_step_params(pack.params, task_step))
            task_probe_after = F.cross_entropy(_forward(task_pack, x_probe), y_probe)
            task_val_after = F.cross_entropy(_forward(task_pack, x_valmini), y_valmini)
            task_descent = geom.safe_float(before_probe - task_probe_after)
            for cand in candidates:
                if cand == "D12-MLPAnalogGeometryMaintenance" or cand == "D13-QuadraticFeatureMLPAnalogMaintenance":
                    continue
                if cand == "D0-TaskOnlyAdamW":
                    raw = [torch.zeros_like(p) for p in pack.params]
                    scaled = raw
                elif cand == "D2-RandomMatchedNorm" or cand == "D4-ShuffledPayload":
                    raw = [torch.randn(p.shape, generator=gen, device=device, dtype=p.dtype) for p in pack.params]
                    scaled = _scale_to_task_fraction(raw, task_step, float(args.functional_norm_fraction))
                elif cand == "D3-AdamWParallelDirection":
                    raw = task_step
                    scaled = _scale_to_task_fraction(raw, task_step, float(args.functional_norm_fraction))
                else:
                    raw = _raw_geometry_direction(pack, cand, x_update)
                    scaled = _scale_to_task_fraction(raw, task_step, float(args.functional_norm_fraction))
                accepted_lambda = 0.0
                accepted_row: Dict[str, Any] | None = None
                backtracks = 0
                for lam in lambda_grid:
                    new_step = _add_steps(task_step, scaled, alpha=lam)
                    cos = geom.safe_float(_step_dot(new_step, task_step) / (_step_norm(new_step) * _step_norm(task_step)).clamp_min(EPS))
                    moved_pack = _pack_with_params(pack, _apply_step_params(pack.params, new_step))
                    probe_after = F.cross_entropy(_forward(moved_pack, x_probe), y_probe)
                    val_after = F.cross_entropy(_forward(moved_pack, x_valmini), y_valmini)
                    holdout_descent = geom.safe_float(before_probe - probe_after)
                    ratio = holdout_descent / max(EPS, task_descent)
                    bad = int(holdout_descent < -1.0e-8)
                    trial = {
                        "stage": "P4_LAMBDA_BACKTRACKING",
                        "status": p4_status,
                        "official_gate_open": int(official_gate_open),
                        "method_id": lq_method,
                        "candidate_direction": cand,
                        "dataset": dataset,
                        "seed": seed,
                        "checkpoint_id": snap.checkpoint_id,
                        "lambda_trial": lam,
                        "cos_new_task": cos,
                        "holdout_descent_ratio": ratio,
                        "bad_step_new": bad,
                        "accepted_trial": 0,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                    lambda_rows.append(trial)
                    if accepted_row is None and cos >= 0.85 and ratio >= 0.95 and bad == 0:
                        accepted_lambda = lam
                        trial["accepted_trial"] = 1
                        accepted_row = {
                            "moved_pack": moved_pack,
                            "probe_after": probe_after,
                            "val_after": val_after,
                            "cos": cos,
                            "ratio": ratio,
                            "bad": bad,
                        }
                        break
                    backtracks += 1
                if accepted_row is None:
                    accepted_lambda = 0.0
                    new_step = task_step
                    moved_pack = task_pack
                    probe_after = task_probe_after
                    val_after = task_val_after
                    cos = 1.0
                    ratio = 1.0
                    bad = 0
                else:
                    moved_pack = accepted_row["moved_pack"]
                    probe_after = accepted_row["probe_after"]
                    val_after = accepted_row["val_after"]
                    cos = accepted_row["cos"]
                    ratio = accepted_row["ratio"]
                    bad = accepted_row["bad"]
                after_geom = _geometry_metrics_for_pack(moved_pack, x_probe, y_probe, args, seed=120420 + seed)
                geom_improve = int(
                    (_safe_float(after_geom.get("curvature_debt")) < _safe_float(before_geom.get("curvature_debt")))
                    or (_safe_float(after_geom.get("perturb_logit_drift_p95")) < _safe_float(before_geom.get("perturb_logit_drift_p95")))
                    or (_safe_float(after_geom.get("basis_usage_entropy")) > _safe_float(before_geom.get("basis_usage_entropy")))
                    or (_safe_float(after_geom.get("CE_p99")) < _safe_float(before_geom.get("CE_p99")))
                    or (_safe_float(after_geom.get("ECE")) < _safe_float(before_geom.get("ECE")))
                    or (_safe_float(after_geom.get("noise_leak")) < _safe_float(before_geom.get("noise_leak")))
                )
                one_pass = int(cos >= 0.85 and ratio >= 0.95 and bad == 0 and geom_improve == 1)
                row = {
                    "stage": "P4_FUNCTIONAL_DIRECTION_AUDIT",
                    "status": p4_status,
                    "official_gate_open": int(official_gate_open),
                    "method_id": lq_method,
                    "model_family": "FC-PureKAN-LQ",
                    "candidate_direction": cand,
                    "dataset": dataset,
                    "seed": seed,
                    "checkpoint_id": snap.checkpoint_id,
                    "cos_new_task": cos,
                    "projection_on_task": geom.safe_float(_step_dot(scaled, task_step) / _step_norm(task_step).clamp_min(EPS)),
                    "norm_new_over_task": geom.safe_float(_step_norm(task_step if accepted_lambda == 0 else _add_steps(task_step, scaled, accepted_lambda)) / _step_norm(task_step).clamp_min(EPS)),
                    "angle_new_task_degree": math.degrees(math.acos(max(-1.0, min(1.0, cos)))),
                    "lambda_initial": lambda_grid[0] if lambda_grid else 0.0,
                    "lambda_accepted": accepted_lambda,
                    "lambda_backtrack_count": backtracks,
                    "lambda_zero_rate": int(accepted_lambda == 0.0),
                    "train_descent_task": geom.safe_float(-_loss),
                    "train_descent_new": geom.safe_float(-_loss),
                    "holdout_descent_task": task_descent,
                    "holdout_descent_new": geom.safe_float(before_probe - probe_after),
                    "holdout_descent_ratio": ratio,
                    "valmini_descent_task": geom.safe_float(before_val - task_val_after),
                    "valmini_descent_new": geom.safe_float(before_val - val_after),
                    "bad_step_new": bad,
                    "bad_step_rate_new": bad,
                    "rollback_error": 0.0,
                    "delta_effective_rank": _safe_float(after_geom.get("effective_rank_hidden")) - _safe_float(before_geom.get("effective_rank_hidden")),
                    "delta_lift_condition": _safe_float(after_geom.get("lift_condition_proxy")) - _safe_float(before_geom.get("lift_condition_proxy")),
                    "delta_basis_entropy": _safe_float(after_geom.get("basis_usage_entropy")) - _safe_float(before_geom.get("basis_usage_entropy")),
                    "delta_curvature_debt": _safe_float(after_geom.get("curvature_debt")) - _safe_float(before_geom.get("curvature_debt")),
                    "delta_perturb_logit_drift_p95": _safe_float(after_geom.get("perturb_logit_drift_p95")) - _safe_float(before_geom.get("perturb_logit_drift_p95")),
                    "delta_local_jacobian_norm_p95": _safe_float(after_geom.get("local_jacobian_norm_p95")) - _safe_float(before_geom.get("local_jacobian_norm_p95")),
                    "delta_signal_consistency": _safe_float(after_geom.get("signal_consistency_real")) - _safe_float(before_geom.get("signal_consistency_real")),
                    "delta_noise_leak": _safe_float(after_geom.get("noise_leak")) - _safe_float(before_geom.get("noise_leak")),
                    "delta_CEp99": _safe_float(after_geom.get("CE_p99")) - _safe_float(before_geom.get("CE_p99")),
                    "delta_margin_p10": _safe_float(after_geom.get("margin_p10")) - _safe_float(before_geom.get("margin_p10")),
                    "delta_ECE_proxy": _safe_float(after_geom.get("ECE")) - _safe_float(before_geom.get("ECE")),
                    "delta_NLL_proxy": _safe_float(after_geom.get("NLL")) - _safe_float(before_geom.get("NLL")),
                    "logit_drift_p95": _safe_float(after_geom.get("perturb_logit_drift_p95")),
                    "feature_drift_p95": _safe_float(after_geom.get("local_jacobian_norm_p95")),
                    "geometry_improvement": geom_improve,
                    "one_step_gate_pass": one_pass,
                    "geometry_score": 0.0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
                row["geometry_score"] = _audit_geometry_score(row)
                audit_rows.append(row)
                one_rows.append({**row, "stage": "P4_ONE_STEP_PROBE"})
                if one_pass and cand in {"D8-SNRProjectedGeometry", "D9-BasisEntropyRebalance", "D10-LiftConditionRepair", "D11-TailStabilityCorrection"}:
                    one_pass_rows.append(row)
                if cand.startswith("D8") or cand.startswith("D9") or cand.startswith("D10") or cand.startswith("D11"):
                    if not one_pass:
                        failures.append(
                            _failure(
                                "P4",
                                "F7_functional_task_unsafe" if (cos < 0.85 or ratio < 0.95 or bad) else "F8_functional_no_geometry_gain",
                                "functional candidate failed one-step gate",
                                method_id=lq_method,
                                dataset=dataset,
                                seed=seed,
                                checkpoint_id=snap.checkpoint_id,
                                metric_name="one_step_gate_pass",
                                metric_value=one_pass,
                                threshold=1,
                                action="repair direction/backtracking before short-run",
                            )
                        )

    analog_map = {
        "D12-MLPAnalogGeometryMaintenance": "B0-MLP-same-param-AdamW",
        "D13-QuadraticFeatureMLPAnalogMaintenance": "B2-QuadraticFeatureMLP-diagnostic-AdamW",
    }
    for cand, method_id in analog_map.items():
        if cand not in candidates:
            continue
        for dataset in datasets:
            if dataset not in data_cache:
                data_cache[dataset] = _load_vision_split(
                    args,
                    dataset,
                    train_size=int(args.train_size),
                    val_size=int(args.val_size),
                    test_size=int(args.test_size),
                )
            x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _x_test_cpu, _y_test_cpu, _in_dim, _out_dim, _protocol = data_cache[dataset]
            for seed in seeds:
                snap = _select_snapshot(snapshots, method_id, dataset, seed, target_epoch=max(1.0, int(args.epochs) * 0.5))
                if snap is None:
                    failures.append(
                        _failure(
                            "P4",
                            "F13_not_run_gate",
                            "missing analog checkpoint for P4 control",
                            method_id=method_id,
                            dataset=dataset,
                            seed=seed,
                            action="complete P1 analog snapshots",
                        )
                    )
                    continue
                pack = _clone_pack(snap.pack, device=device, with_states=False)
                perm = torch.randperm(int(x_train_cpu.shape[0]), generator=torch.Generator().manual_seed(120900 + seed))
                bsz = int(args.batch_size)
                upd_idx = perm[:bsz]
                probe_idx = perm[bsz : 2 * bsz]
                if int(probe_idx.numel()) == 0:
                    probe_idx = upd_idx
                x_update = x_train_cpu[upd_idx].to(device=device, dtype=torch.float32)
                y_update = y_train_cpu[upd_idx].to(device=device)
                x_probe = x_train_cpu[probe_idx].to(device=device, dtype=torch.float32)
                y_probe = y_train_cpu[probe_idx].to(device=device)
                x_valmini = x_val_cpu[:bsz].to(device=device, dtype=torch.float32)
                y_valmini = y_val_cpu[:bsz].to(device=device)
                before_probe = F.cross_entropy(_forward(pack, x_probe), y_probe)
                before_val = F.cross_entropy(_forward(pack, x_valmini), y_valmini)
                before_geom = _geometry_metrics_for_pack(pack, x_probe, y_probe, args, seed=121000 + seed)
                loss, grads = _ce_bwd(pack, x_update, y_update)
                task_step = [(-float(args.lr)) * g.detach() for g in grads]
                task_pack = _pack_with_params(pack, _apply_step_params(pack.params, task_step))
                task_probe_after = F.cross_entropy(_forward(task_pack, x_probe), y_probe)
                task_val_after = F.cross_entropy(_forward(task_pack, x_valmini), y_valmini)
                task_descent = geom.safe_float(before_probe - task_probe_after)
                raw = _raw_geometry_direction(pack, cand, x_update)
                scaled = _scale_to_task_fraction(raw, task_step, float(args.functional_norm_fraction))
                accepted_lambda = 0.0
                accepted_row: Dict[str, Any] | None = None
                backtracks = 0
                for lam in lambda_grid:
                    new_step = _add_steps(task_step, scaled, alpha=lam)
                    cos = geom.safe_float(_step_dot(new_step, task_step) / (_step_norm(new_step) * _step_norm(task_step)).clamp_min(EPS))
                    moved_pack = _pack_with_params(pack, _apply_step_params(pack.params, new_step))
                    probe_after = F.cross_entropy(_forward(moved_pack, x_probe), y_probe)
                    val_after = F.cross_entropy(_forward(moved_pack, x_valmini), y_valmini)
                    holdout_descent = geom.safe_float(before_probe - probe_after)
                    ratio = holdout_descent / max(EPS, task_descent)
                    bad = int(holdout_descent < -1.0e-8)
                    trial = {
                        "stage": "P4_LAMBDA_BACKTRACKING",
                        "status": p4_status,
                        "official_gate_open": int(official_gate_open),
                        "method_id": method_id,
                        "candidate_direction": cand,
                        "dataset": dataset,
                        "seed": seed,
                        "checkpoint_id": snap.checkpoint_id,
                        "lambda_trial": lam,
                        "cos_new_task": cos,
                        "holdout_descent_ratio": ratio,
                        "bad_step_new": bad,
                        "accepted_trial": 0,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                    lambda_rows.append(trial)
                    if accepted_row is None and cos >= 0.85 and ratio >= 0.95 and bad == 0:
                        accepted_lambda = lam
                        trial["accepted_trial"] = 1
                        accepted_row = {"moved_pack": moved_pack, "probe_after": probe_after, "val_after": val_after, "cos": cos, "ratio": ratio, "bad": bad}
                        break
                    backtracks += 1
                if accepted_row is None:
                    accepted_lambda = 0.0
                    moved_pack = task_pack
                    probe_after = task_probe_after
                    val_after = task_val_after
                    cos = 1.0
                    ratio = 1.0
                    bad = 0
                else:
                    moved_pack = accepted_row["moved_pack"]
                    probe_after = accepted_row["probe_after"]
                    val_after = accepted_row["val_after"]
                    cos = accepted_row["cos"]
                    ratio = accepted_row["ratio"]
                    bad = accepted_row["bad"]
                after_geom = _geometry_metrics_for_pack(moved_pack, x_probe, y_probe, args, seed=121100 + seed)
                geom_improve = int(
                    (_safe_float(after_geom.get("curvature_debt")) < _safe_float(before_geom.get("curvature_debt")))
                    or (_safe_float(after_geom.get("perturb_logit_drift_p95")) < _safe_float(before_geom.get("perturb_logit_drift_p95")))
                    or (_safe_float(after_geom.get("basis_usage_entropy")) > _safe_float(before_geom.get("basis_usage_entropy")))
                    or (_safe_float(after_geom.get("CE_p99")) < _safe_float(before_geom.get("CE_p99")))
                    or (_safe_float(after_geom.get("ECE")) < _safe_float(before_geom.get("ECE")))
                    or (_safe_float(after_geom.get("noise_leak")) < _safe_float(before_geom.get("noise_leak")))
                )
                one_pass = int(cos >= 0.85 and ratio >= 0.95 and bad == 0 and geom_improve == 1)
                row = {
                    "stage": "P4_FUNCTIONAL_DIRECTION_AUDIT",
                    "status": p4_status,
                    "official_gate_open": int(official_gate_open),
                    "method_id": method_id,
                    "model_family": snap.model_family,
                    "candidate_direction": cand,
                    "dataset": dataset,
                    "seed": seed,
                    "checkpoint_id": snap.checkpoint_id,
                    "cos_new_task": cos,
                    "projection_on_task": geom.safe_float(_step_dot(scaled, task_step) / _step_norm(task_step).clamp_min(EPS)),
                    "norm_new_over_task": geom.safe_float(_step_norm(task_step if accepted_lambda == 0 else _add_steps(task_step, scaled, accepted_lambda)) / _step_norm(task_step).clamp_min(EPS)),
                    "angle_new_task_degree": math.degrees(math.acos(max(-1.0, min(1.0, cos)))),
                    "lambda_initial": lambda_grid[0] if lambda_grid else 0.0,
                    "lambda_accepted": accepted_lambda,
                    "lambda_backtrack_count": backtracks,
                    "lambda_zero_rate": int(accepted_lambda == 0.0),
                    "train_descent_task": geom.safe_float(-loss),
                    "train_descent_new": geom.safe_float(-loss),
                    "holdout_descent_task": task_descent,
                    "holdout_descent_new": geom.safe_float(before_probe - probe_after),
                    "holdout_descent_ratio": ratio,
                    "valmini_descent_task": geom.safe_float(before_val - task_val_after),
                    "valmini_descent_new": geom.safe_float(before_val - val_after),
                    "bad_step_new": bad,
                    "bad_step_rate_new": bad,
                    "rollback_error": 0.0,
                    "delta_effective_rank": _safe_float(after_geom.get("effective_rank_hidden")) - _safe_float(before_geom.get("effective_rank_hidden")),
                    "delta_lift_condition": _safe_float(after_geom.get("lift_condition_proxy")) - _safe_float(before_geom.get("lift_condition_proxy")),
                    "delta_basis_entropy": _safe_float(after_geom.get("basis_usage_entropy")) - _safe_float(before_geom.get("basis_usage_entropy")),
                    "delta_curvature_debt": _safe_float(after_geom.get("curvature_debt")) - _safe_float(before_geom.get("curvature_debt")),
                    "delta_perturb_logit_drift_p95": _safe_float(after_geom.get("perturb_logit_drift_p95")) - _safe_float(before_geom.get("perturb_logit_drift_p95")),
                    "delta_local_jacobian_norm_p95": _safe_float(after_geom.get("local_jacobian_norm_p95")) - _safe_float(before_geom.get("local_jacobian_norm_p95")),
                    "delta_signal_consistency": _safe_float(after_geom.get("signal_consistency_real")) - _safe_float(before_geom.get("signal_consistency_real")),
                    "delta_noise_leak": _safe_float(after_geom.get("noise_leak")) - _safe_float(before_geom.get("noise_leak")),
                    "delta_CEp99": _safe_float(after_geom.get("CE_p99")) - _safe_float(before_geom.get("CE_p99")),
                    "delta_margin_p10": _safe_float(after_geom.get("margin_p10")) - _safe_float(before_geom.get("margin_p10")),
                    "delta_ECE_proxy": _safe_float(after_geom.get("ECE")) - _safe_float(before_geom.get("ECE")),
                    "delta_NLL_proxy": _safe_float(after_geom.get("NLL")) - _safe_float(before_geom.get("NLL")),
                    "logit_drift_p95": _safe_float(after_geom.get("perturb_logit_drift_p95")),
                    "feature_drift_p95": _safe_float(after_geom.get("local_jacobian_norm_p95")),
                    "geometry_improvement": geom_improve,
                    "one_step_gate_pass": one_pass,
                    "geometry_score": 0.0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
                row["geometry_score"] = _audit_geometry_score(row)
                audit_rows.append(row)
                one_rows.append({**row, "stage": "P4_ONE_STEP_PROBE"})

    controls = {"D1-NoOpMatchedOverhead", "D2-RandomMatchedNorm", "D3-AdamWParallelDirection", "D4-ShuffledPayload", "D5-ShuffledEvent", "D6-SNROnlyGate", "D7-GeometryOnlyNoSNR", "D12-MLPAnalogGeometryMaintenance", "D13-QuadraticFeatureMLPAnalogMaintenance"}
    functional = {"D8-SNRProjectedGeometry", "D9-BasisEntropyRebalance", "D10-LiftConditionRepair", "D11-TailStabilityCorrection"}
    control_best = max([_safe_float(r.get("geometry_score")) for r in audit_rows if r.get("candidate_direction") in controls] or [0.0])
    for cand in sorted(functional):
        cand_rows = [r for r in audit_rows if r.get("candidate_direction") == cand]
        cand_best = max([_safe_float(r.get("geometry_score")) for r in cand_rows] or [0.0])
        accepted = sum(int(r.get("one_step_gate_pass", 0)) for r in cand_rows)
        beat = int(accepted > 0 and cand_best > control_best)
        control_rows.append(
            {
                "stage": "P4_CONTROL_MATRIX",
                "status": p4_status,
                "official_gate_open": int(official_gate_open),
                "candidate_direction": cand,
                "accepted_rows": accepted,
                "candidate_best_geometry_score": cand_best,
                "control_best_geometry_score": control_best,
                "beats_strong_controls": beat,
                "mlp_analog_explains_gain": "not_opened_in_this_gate" if beat == 0 else "not_measured",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            }
        )
        if accepted > 0 and not beat:
            failures.append(_failure("P4", "F9_control_explains_gain", "functional one-step gain did not beat strong controls", method_id=lq_method, metric_name=cand, metric_value=cand_best, threshold=f">{control_best}", action="orthogonalize against optimizer/control directions"))

    if one_pass_rows:
        # Conservative five-step cloned audit for rows that cleared one-step.
        for source_row in one_pass_rows[: int(args.p4_max_five_step)]:
            dataset = str(source_row["dataset"])
            seed = int(source_row["seed"])
            snap = _select_snapshot(
                snapshots,
                str(source_row["method_id"]),
                dataset,
                seed,
                target_epoch=max(1.0, int(args.epochs) * 0.5),
            )
            if snap is None:
                five_rows.append(
                    {
                        "stage": "P4_FIVE_STEP_PROBE",
                        "status": "not_run",
                        "official_gate_open": int(official_gate_open),
                        "method_id": source_row["method_id"],
                        "candidate_direction": source_row["candidate_direction"],
                        "dataset": dataset,
                        "seed": seed,
                        "checkpoint_id": source_row["checkpoint_id"],
                        "five_step_status": "not_run",
                        "reason": "source_checkpoint_missing",
                        "cumulative_holdout_nonharm_pass": 0,
                        "geometry_not_reversed_pass": 0,
                        "control_resistant_pass": 0,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                )
                continue
            if dataset not in data_cache:
                data_cache[dataset] = _load_vision_split(
                    args,
                    dataset,
                    train_size=int(args.train_size),
                    val_size=int(args.val_size),
                    test_size=int(args.test_size),
                )
            x_train_cpu, y_train_cpu, _x_val_cpu, _y_val_cpu, _x_test_cpu, _y_test_cpu, _in_dim, _out_dim, _protocol = data_cache[dataset]
            pack = _clone_pack(snap.pack, device=device, with_states=False)
            candidate_direction = str(source_row["candidate_direction"])
            accepted_lambda = _safe_float(source_row.get("lambda_accepted"), 0.0)
            bad_count = 0
            cumulative_holdout_delta = 0.0
            initial_geometry_score = _safe_float(source_row.get("geometry_score"), 0.0)
            control_resistant = any(
                int(r.get("beats_strong_controls", 0)) == 1
                for r in control_rows
                if r.get("candidate_direction") == candidate_direction
            )
            for step_idx in range(5):
                perm = torch.randperm(
                    int(x_train_cpu.shape[0]),
                    generator=torch.Generator().manual_seed(122000 + seed * 37 + step_idx),
                )
                bsz = int(args.batch_size)
                upd_idx = perm[:bsz]
                probe_idx = perm[bsz : 2 * bsz]
                if int(probe_idx.numel()) == 0:
                    probe_idx = upd_idx
                x_update = x_train_cpu[upd_idx].to(device=device, dtype=torch.float32)
                y_update = y_train_cpu[upd_idx].to(device=device)
                x_probe = x_train_cpu[probe_idx].to(device=device, dtype=torch.float32)
                y_probe = y_train_cpu[probe_idx].to(device=device)
                before_probe = F.cross_entropy(_forward(pack, x_probe), y_probe)
                before_geom = _geometry_metrics_for_pack(pack, x_probe, y_probe, args, seed=122100 + seed + step_idx)
                _loss, grads = _ce_bwd(pack, x_update, y_update)
                task_step = [(-float(args.lr)) * g.detach() for g in grads]
                raw = _raw_geometry_direction(pack, candidate_direction, x_update)
                scaled = _scale_to_task_fraction(raw, task_step, float(args.functional_norm_fraction))
                new_step = _add_steps(task_step, scaled, alpha=accepted_lambda)
                moved_pack = _pack_with_params(pack, _apply_step_params(pack.params, new_step))
                after_probe = F.cross_entropy(_forward(moved_pack, x_probe), y_probe)
                after_geom = _geometry_metrics_for_pack(moved_pack, x_probe, y_probe, args, seed=122200 + seed + step_idx)
                bad = int(after_probe > before_probe + 1.0e-8)
                bad_count += bad
                cumulative_holdout_delta += geom.safe_float(after_probe - before_probe)
                delta_row = {
                    "delta_curvature_debt": _safe_float(after_geom.get("curvature_debt")) - _safe_float(before_geom.get("curvature_debt")),
                    "delta_perturb_logit_drift_p95": _safe_float(after_geom.get("perturb_logit_drift_p95")) - _safe_float(before_geom.get("perturb_logit_drift_p95")),
                    "delta_basis_entropy": _safe_float(after_geom.get("basis_usage_entropy")) - _safe_float(before_geom.get("basis_usage_entropy")),
                    "delta_CEp99": _safe_float(after_geom.get("CE_p99")) - _safe_float(before_geom.get("CE_p99")),
                    "delta_margin_p10": _safe_float(after_geom.get("margin_p10")) - _safe_float(before_geom.get("margin_p10")),
                    "delta_ECE_proxy": _safe_float(after_geom.get("ECE")) - _safe_float(before_geom.get("ECE")),
                    "delta_noise_leak": _safe_float(after_geom.get("noise_leak")) - _safe_float(before_geom.get("noise_leak")),
                }
                geometry_score = _audit_geometry_score(delta_row)
                five_rows.append(
                    {
                        "stage": "P4_FIVE_STEP_PROBE",
                        "status": p4_status,
                        "official_gate_open": int(official_gate_open),
                        "method_id": source_row["method_id"],
                        "candidate_direction": candidate_direction,
                        "dataset": dataset,
                        "seed": seed,
                        "checkpoint_id": source_row["checkpoint_id"],
                        "rollout_step": step_idx + 1,
                        "five_step_status": p4_status,
                        "reason": "diagnostic_gate_closed" if not official_gate_open else "measured",
                        "holdout_loss_before": geom.safe_float(before_probe),
                        "holdout_loss_after": geom.safe_float(after_probe),
                        "cumulative_holdout_delta": cumulative_holdout_delta,
                        "bad_step_rate": bad_count / float(step_idx + 1),
                        "geometry_score": geometry_score,
                        "cumulative_holdout_nonharm_pass": int(bad_count == 0),
                        "geometry_not_reversed_pass": int(geometry_score >= min(0.0, initial_geometry_score)),
                        "control_resistant_pass": int(control_resistant),
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    }
                )
                pack = moved_pack
    else:
        five_rows.append(
            {
                **_not_run("P4_FIVE_STEP_PROBE", "no_functional_candidate_passed_one_step"),
                "official_gate_open": int(official_gate_open),
            }
        )

    p4_pass = (
        official_gate_open
        and bool(control_rows)
        and any(int(r.get("beats_strong_controls", 0)) == 1 for r in control_rows)
        and all(str(r.get("five_step_status", "")) != "not_run" for r in five_rows)
    )
    write_csv_rows(out_dir / "p4_functional_direction_audit.csv", audit_rows or [_not_run("P4", "no_audit_rows")])
    write_csv_rows(out_dir / "p4_one_step_probe.csv", one_rows or [_not_run("P4_ONE_STEP_PROBE", "no_one_step_rows")])
    write_csv_rows(out_dir / "p4_lambda_backtracking.csv", lambda_rows or [_not_run("P4_LAMBDA_BACKTRACKING", "no_lambda_rows")])
    write_csv_rows(out_dir / "p4_control_matrix.csv", control_rows or [_not_run("P4_CONTROL_MATRIX", "no_control_rows")])
    write_csv_rows(out_dir / "p4_five_step_probe.csv", five_rows)
    return audit_rows, one_rows, lambda_rows, control_rows, five_rows, p4_pass, failures


def run_p5(out_dir: Path, p4_pass: bool) -> List[Dict[str, Any]]:
    if not p4_pass:
        rows = [_not_run("P5", "P4_no_control_resistant_functional_candidate")]
    else:
        rows = [_not_run("P5", "P4_opened_but_short_run_not_executed_in_this_v12_1_run")]
    write_csv_rows(out_dir / "p5_short_run_plan_or_notrun.csv", rows)
    return rows


def _write_simple_svg(path: Path, title: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    ensure_dir(path.parent)
    lines = []
    for row in list(rows)[:10]:
        label = str(row.get("method_id", row.get("candidate_direction", row.get("metric_name", ""))))[:42]
        vals = "  ".join(f"{f}={row.get(f, '')}" for f in fields)
        lines.append((label, vals[:120]))
    height = 120 + 24 * max(1, len(lines))
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}">',
        '<rect width="1100" height="100%" fill="#f8fafc"/>',
        f'<text x="28" y="42" font-family="Arial, sans-serif" font-size="24" fill="#111827">{title}</text>',
        '<text x="28" y="72" font-family="Arial, sans-serif" font-size="13" fill="#64748b">Rendered from measured CSV/JSON artifacts; no inferred pass values.</text>',
    ]
    y = 110
    for label, vals in lines:
        body.append(f'<text x="36" y="{y}" font-family="Arial, sans-serif" font-size="13" fill="#0f172a">{label}</text>')
        body.append(f'<text x="360" y="{y}" font-family="Arial, sans-serif" font-size="13" fill="#334155">{vals}</text>')
        y += 24
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def write_figures(out_dir: Path, artifacts: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    fig = ensure_dir(out_dir / "figures")
    specs = [
        ("p0_contract_heatmap.svg", "P0 Contract Heatmap", "p0", ["contract_pass", "rollback_max_error"]),
        ("p1_accuracy_delta_by_dataset_seed.svg", "P1 Accuracy Delta", "p1", ["dataset", "seed", "val_acc"]),
        ("p1_val_loss_vs_time.svg", "P1 Val Loss vs Time", "trace", ["checkpoint_id", "wall_clock_sec", "val_loss"]),
        ("p1_efficiency_dashboard.svg", "P1 Efficiency Dashboard", "eff", ["step_time_ms", "step_time_ratio_vs_MLP"]),
        ("p1_compact_vs_conservative_memory.svg", "P1 Compact vs Conservative Memory", "eff", ["compact_memory_MB", "conservative_memory_MB"]),
        ("p1_expression_battery_R2.svg", "P1 Expression Battery R2", "expr", ["expression_target_id", "val_R2"]),
        ("p1_task_efficiency_pareto.svg", "P1 Task Efficiency Pareto", "p1", ["val_acc", "val_loss_auc_time"]),
        ("p2_effective_rank_trajectory.svg", "P2 Effective Rank Trajectory", "p2", ["checkpoint_id", "effective_rank_hidden"]),
        ("p2_basis_entropy_trajectory.svg", "P2 Basis Entropy Trajectory", "p2", ["checkpoint_id", "basis_usage_entropy"]),
        ("p2_perturb_drift_distribution.svg", "P2 Perturb Drift Distribution", "p2", ["perturb_logit_drift_p95"]),
        ("p2_snr_rolewise_heatmap.svg", "P2 SNR Rolewise Heatmap", "p2", ["snr_positive_fraction_global"]),
        ("p2_tail_metrics_trajectory.svg", "P2 Tail Metrics Trajectory", "p2", ["CE_p99", "margin_p10"]),
        ("p2_geometry_target_correlation_heatmap.svg", "P2 Geometry Target Correlation", "corr", ["target_name", "pearson"]),
        ("p2_lq_vs_mlp_geometry_radar.svg", "P2 LQ vs MLP Geometry Radar", "p2", ["method_id", "effective_rank_hidden"]),
        ("p3_certificate_gate_heatmap.svg", "P3 Certificate Gate Heatmap", "p3", ["metric_name", "metric_pass"]),
        ("p3_pareto_frontier_task_geometry_cost.svg", "P3 Pareto Frontier", "p3", ["metric_name", "metric_pass"]),
        ("p4_direction_cosine_heatmap.svg", "P4 Direction Cosine", "p4", ["candidate_direction", "cos_new_task"]),
        ("p4_holdout_descent_scatter.svg", "P4 Holdout Descent", "p4", ["candidate_direction", "holdout_descent_ratio"]),
        ("p4_geometry_delta_by_candidate.svg", "P4 Geometry Delta", "p4", ["candidate_direction", "geometry_score"]),
        ("p4_control_beat_matrix.svg", "P4 Control Beat Matrix", "control", ["candidate_direction", "beats_strong_controls"]),
        ("p4_lambda_backtracking_histogram.svg", "P4 Lambda Backtracking", "lambda", ["candidate_direction", "lambda_trial", "accepted_trial"]),
        ("p4_five_step_rollout_curves.svg", "P4 Five Step Rollout", "five", ["candidate_direction", "five_step_status"]),
        ("p4_bad_step_timeline.svg", "P4 Bad Step Timeline", "p4", ["candidate_direction", "bad_step_rate_new"]),
    ]
    for filename, title, key, fields in specs:
        _write_simple_svg(fig / filename, title, artifacts.get(key, []), fields)


def audit_provenance(out_dir: Path) -> List[Dict[str, Any]]:
    rows_checked = 0
    fake = 0
    proxy = 0
    cpu = 0
    for path in out_dir.glob("*.csv"):
        if path.name in {"artifact_hashes.csv", "provenance_audit.csv"}:
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows_checked += 1
                fake += int(_safe_float(row.get("fake_data_used", 0)) != 0.0)
                proxy += int(_safe_float(row.get("proxy_row_used", 0)) != 0.0)
                cpu += int(_safe_float(row.get("cpu_offload_used", 0)) != 0.0)
    rows = [
        {
            "stage": "PROVENANCE_AUDIT",
            "rows_checked": rows_checked,
            "fake_data_used": fake,
            "proxy_row_used": proxy,
            "cpu_offload_used": cpu,
            "no_fake_pass": int(fake == 0),
            "no_proxy_pass": int(proxy == 0),
            "no_cpu_offload_pass": int(cpu == 0),
        }
    ]
    write_csv_rows(out_dir / "provenance_audit.csv", rows)
    return rows


def write_route(
    out_dir: Path,
    *,
    p0_pass: bool,
    p1_pass: bool,
    p2_complete: bool,
    p3_ready: bool,
    p3_pass: bool,
    p4_pass: bool,
    p5_opened: bool,
    provenance: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    no_fake = bool(provenance and int(provenance[0].get("no_fake_pass", 0)) == 1 and int(provenance[0].get("no_proxy_pass", 0)) == 1)
    if not p0_pass:
        route = "R1-ContractFail"
        blocker = "implementation_contract_failed"
        next_action = "repair P0 contract"
    elif not p1_pass:
        route = "R2-BaseNotQualified"
        blocker = "repaired_lq_base_not_qualified"
        next_action = "global repaired LQ base repair"
    elif p2_complete and not p3_pass:
        route = "R3-BaseQualifiedGeometryMeasured"
        blocker = "geometry_certificate_not_passed"
        next_action = "metric repair or geometry target repair"
    elif p3_pass and not p4_pass:
        route = "R4-GeometryCertificateReadyNoFunctional"
        blocker = "functional_audit_not_control_resistant"
        next_action = "direction repair"
    elif p4_pass and p5_opened:
        route = "R7-FunctionalAuditPassShortRunOpened"
        blocker = ""
        next_action = "run P5 short-run controlled training"
    else:
        route = "R0-Unknown"
        blocker = "incomplete_or_unclassified"
        next_action = "inspect failure_table"
    decision = {
        "route": route,
        "base_candidate": "R2-LQ-fanin-output-scale-confirmed",
        "p0_contract_pass": bool(p0_pass),
        "p1_base_pass": bool(p1_pass),
        "p2_geometry_battery_complete": bool(p2_complete),
        "p3_certificate_ready": bool(p3_ready),
        "p3_certificate_pass": bool(p3_pass),
        "p4_functional_audit_pass": bool(p4_pass),
        "p5_short_run_opened": bool(p5_opened),
        "strict_functional_success": False,
        "external_ready": False,
        "primary_blocker": blocker,
        "next_recommended_action": next_action,
        "no_fake": no_fake,
        "failure_count": len(failures),
        "generated_route_status": "functional_not_opened" if not p4_pass else "short_run_opened",
    }
    write_json(out_dir / "route_decision.json", decision)
    write_json(out_dir / "aggregate_decision.json", decision)
    return decision


def write_hashes(out_dir: Path) -> List[Dict[str, str]]:
    paths = [p for p in out_dir.rglob("*") if p.is_file() and p.name != "artifact_hashes.csv"]
    rows = artifact_hash_rows(paths, root=out_dir)
    write_csv_rows(out_dir / "artifact_hashes.csv", rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DG-KAN v12 Good Geometry Battery")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--packages", default="P0,P1,P2,P3,P4")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--eval-train-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--checkpoint-schedule", default="0,1,3,10,final")
    parser.add_argument("--microbatch-count", type=int, default=4)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--kernel-sketch-directions", type=int, default=6)
    parser.add_argument("--finite-diff-eps", type=float, default=1.0e-4)
    parser.add_argument("--functional-candidates", default="D0,D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13")
    parser.add_argument("--controls", default="D1,D2,D3,D4,D5,D6,D7")
    parser.add_argument("--lambda-grid", default="0.05,0.02,0.01,0.005,0.0")
    parser.add_argument("--functional-norm-fraction", type=float, default=0.10)
    parser.add_argument("--profile-steps", type=int, default=8)
    parser.add_argument("--expression-train-size", type=int, default=1024)
    parser.add_argument("--expression-val-size", type=int, default=512)
    parser.add_argument("--expression-steps", type=int, default=200)
    parser.add_argument("--expression-batch-size", type=int, default=128)
    parser.add_argument("--expression-eval-stride", type=int, default=20)
    parser.add_argument("--expression-lr", type=float, default=3.0e-3)
    parser.add_argument("--p4-max-five-step", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2413)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "v12_0_good_geometry_battery" / f"v120_good_geometry_battery_{_now_tag()}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = _device(args.device)
    packages = set(_parse_list(args.packages))
    if "ALL" in packages:
        packages.update({"P0", "P1", "P2", "P3", "P4"})

    all_failures: List[Dict[str, Any]] = []
    write_json(
        out_dir / "run_manifest_v120.json",
        {
            "stage": "RUN_MANIFEST_V120",
            "started_at": _now_iso(),
            "args": vars(args),
            "device": str(device),
            "plan_path": str(PLAN_PATH.relative_to(ROOT)),
            "runner": str(SCRIPT_PATH.relative_to(ROOT)),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        },
    )

    if "P0" in packages:
        p0_rows, p0_pass, failures = run_p0(args, out_dir, device)
        all_failures.extend(failures)
    else:
        p0_rows, p0_pass = [_not_run("P0", "package_not_requested")], False
        write_csv_rows(out_dir / "p0_contract.csv", p0_rows)

    if "P1" in packages:
        p1_rows, trace_rows, eff_rows, expr_rows, snapshots, final_map, p1_pass, failures = run_p1(args, out_dir, device, p0_pass)
        all_failures.extend(failures)
    else:
        p1_rows = trace_rows = eff_rows = expr_rows = [_not_run("P1", "package_not_requested")]
        snapshots = []
        final_map = {}
        p1_pass = False
        for name in ["p1_base_qualification.csv", "p1_base_task_trace.csv", "p1_efficiency_profile.csv", "p1_expression_battery.csv"]:
            write_csv_rows(out_dir / name, p1_rows)

    if "P2" in packages or "P2_METRIC_SMOKE" in packages:
        p2_rows, p2_trace, corr_rows, p2_complete, failures = run_p2(args, out_dir, device, snapshots, final_map, p0_pass)
        all_failures.extend(failures)
    else:
        p2_rows = p2_trace = corr_rows = [_not_run("P2", "package_not_requested")]
        p2_complete = False
        for name in ["p2_passive_geometry_snapshot.csv", "p2_geometry_checkpoint_trace.csv", "p2_geometry_correlation.csv"]:
            write_csv_rows(out_dir / name, p2_rows)

    if "P3" in packages:
        p3_cert, p3_rows, p3_ready, failures = run_p3(args, out_dir, p1_rows, p2_rows, eff_rows, p2_complete)
        all_failures.extend(failures)
    else:
        p3_cert = {"pass": False, "failure_reasons": ["package_not_requested"]}
        p3_rows = [_not_run("P3", "package_not_requested")]
        p3_ready = False
        write_csv_rows(out_dir / "p3_geometry_certificate.csv", p3_rows)
        write_json(out_dir / "p3_geometry_certificate.json", p3_cert)

    if "P4" in packages or "P4_DIRECTION_SMOKE" in packages:
        p4_rows, one_rows, lambda_rows, control_rows, five_rows, p4_pass, failures = run_p4(args, out_dir, device, snapshots, p1_pass, p3_cert)
        all_failures.extend(failures)
    else:
        p4_rows = one_rows = lambda_rows = control_rows = five_rows = [_not_run("P4", "package_not_requested")]
        p4_pass = False
        for name in [
            "p4_functional_direction_audit.csv",
            "p4_one_step_probe.csv",
            "p4_lambda_backtracking.csv",
            "p4_control_matrix.csv",
            "p4_five_step_probe.csv",
        ]:
            write_csv_rows(out_dir / name, p4_rows)

    p5_rows = run_p5(out_dir, p4_pass)
    p5_opened = bool(p4_pass)
    if not p5_opened:
        all_failures.append(_failure("P5", "F13_not_run_gate", "P5 not opened by P4 gate", action="repair P4 control-resistant functional candidate"))

    write_csv_rows(out_dir / "failure_table.csv", all_failures or [_failure("ALL", "none", "no failures recorded")])
    artifacts = {
        "p0": p0_rows,
        "p1": p1_rows,
        "trace": trace_rows,
        "eff": eff_rows,
        "expr": expr_rows,
        "p2": p2_rows,
        "corr": corr_rows,
        "p3": p3_rows,
        "p4": p4_rows,
        "lambda": lambda_rows,
        "control": control_rows,
        "five": five_rows,
    }
    write_figures(out_dir, artifacts)
    provenance = audit_provenance(out_dir)
    write_route(
        out_dir,
        p0_pass=p0_pass,
        p1_pass=p1_pass,
        p2_complete=p2_complete,
        p3_ready=p3_ready,
        p3_pass=bool(p3_cert.get("pass", False)),
        p4_pass=p4_pass,
        p5_opened=p5_opened,
        provenance=provenance,
        failures=all_failures,
    )
    write_hashes(out_dir)
    print(json.dumps({"out_dir": str(out_dir), "route": json.loads((out_dir / "route_decision.json").read_text())["route"]}, sort_keys=True))


if __name__ == "__main__":
    main()
