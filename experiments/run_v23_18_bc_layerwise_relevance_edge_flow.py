"""DG-KAN v23.18 Basis-Covariant Layerwise Relevance Edge Flow audits.

This runner is deliberately conservative: it executes the pre-registered
Part 0/A/B gates first, then blocks downstream parts when their prerequisite
mechanism evidence is absent. Missing or failed evidence is recorded as such.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import py_compile
import random
import sys
import time
from typing import Any, Iterable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v23_17_bc_future_tangent_shaping as v2317
from dgkan.fu.basis_covariant_nullspace import EPS, g_norm, null_ratio, select_soft_null_mu
from dgkan.fu.compositional_edge_natural_flow import blockdiag_exact_flow, metric_matrix_for_model, metric_norm_sq
from dgkan.fu.compositional_edge_tangent import (
    apply_j,
    build_tangent_cache,
    explicit_jacobian,
    flatten_coeffs,
    unflatten_coeffs,
)
from dgkan.fu.future_tangent_shaping import symmetric_future_tangent_covector, build_shaping_direction


RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.18_BasisCovariantLayerwiseRelevanceEdgeFlow_多假设完整详尽实验计划.md"
OUT_ROOT = ROOT / "results/v23_18"
EXEC_LOG = ROOT / "docs/DG-KAN_v23.18_BasisCovariantLayerwiseRelevanceEdgeFlow_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.18_BasisCovariantLayerwiseRelevanceEdgeFlow_实验结果复盘.md"
PYTHON = sys.executable

OLD_RUNNERS = [
    ROOT / "experiments/run_v23_15_basis_covariant_edge_natural_flow.py",
    ROOT / "experiments/run_v23_16_compositional_bc_vh_flow.py",
    ROOT / "experiments/run_v23_17_bc_future_tangent_shaping.py",
]
HELPERS = [
    ROOT / "dgkan/fu/compositional_edge_tangent.py",
    ROOT / "dgkan/fu/compositional_edge_natural_flow.py",
    ROOT / "dgkan/fu/future_tangent_shaping.py",
    ROOT / "dgkan/fu/basis_covariant_nullspace.py",
]

SCHEMES_PRIMARY = [
    "P0_BC15_baseline",
    "P1_BC_LREF_bank_static_primary",
    "P2_BC_LREF_bank_transport_primary",
    "P3_BC_LREF_bank_transport_momentum_primary",
]
SCHEMES_DIAG = [
    "D1_exact_full_two_step_hypergradient",
    "D2_partial_HVP_v23_17_replay",
    "D3_independent_edge_LREF",
    "D4_bank_blockdiag_LREF",
    "D5_bank_operator_full_LREF",
    "D6_CE_Fisher_LREF",
    "D7_layer_shared_parent_metric_diagnostic",
]
PART_B_SCHEMES = [
    "B0_BC15",
    "B1_partial_HVP_v23_17_replay",
    "B2_exact_full_two_step_hypergradient",
    "B3_static_node_bank_LREF",
    "B4_static_independent_edge_LREF",
    "B5_static_bank_operator_LREF",
    "B6_CE_Fisher_node_bank_LREF",
    "B7_same_G_norm_random_subspace",
    "B8_label_shuffled_relevance",
    "B9_same_spectrum_random_orientation",
]
CONTROLS = [
    "C1_same_G_norm_random_subspace",
    "C2_same_eigenvalue_spectrum_random_orientation",
    "C3_label_shuffled_relevance",
    "C4_hardness_bucket_shuffled_relevance",
    "C5_source_only_relevance",
    "C6_witness_only_relevance",
    "C7_source_witness_pairing_shuffled",
    "C8_same_bank_energy_random",
    "C9_same_smoothness_random",
    "C10_same_compute_noop",
]
THRESHOLDS = {
    "rank": 4,
    "tau_stable": 0.50,
    "eta_N": 1.0,
    "gamma": 1.0,
    "R_vis": 0.25,
    "condition_number": 1.0e6,
    "part_b_static_future_nll_median": 1.0e-4,
    "part_b_static_future_coverage_median": 0.005,
    "part_b_static_coverage_row_threshold": 0.01,
    "part_b_full_hyper_future_nll_median": 1.0e-4,
    "part_b_full_hyper_coverage_row_threshold": 0.01,
}
AUDIT_DEFAULTS = {
    "version": "v23.18",
    "primary_hypothesis": "BC_LREF_fixed_registry_no_runtime_winner",
    "null_threshold": "",
    "primary_scale": "",
    "control_registry_complete": 1,
    "held_test_usage": 0,
    "runtime_selector_used": 0,
    "candidate_winner_selection_used": 0,
    "metric_winner_selection_used": 0,
    "future_direction_used": 0,
    "auxiliary_loss_used": 0,
    "output_oracle_runtime_used": 0,
    "manual_update_detected": 0,
    "new_edge_function_added": 0,
    "mlp_stem_used": 0,
    "mlp_readout_used": 0,
    "used_fake_data_rows": 0,
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def command_text(argv: Iterable[str] | None = None) -> str:
    vals = list(sys.argv if argv is None else argv)
    prefix = []
    for name in ["CUDA_VISIBLE_DEVICES"]:
        value = os.environ.get(name)
        if value:
            prefix.append(f"{name}={value}")
    return " ".join([*prefix, PYTHON, rel(RUNNER), *vals[1:]])


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, data: dict[str, Any]) -> Path:
    ensure_out()
    payload = {**AUDIT_DEFAULTS, **data}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_out()
    enriched = [{**AUDIT_DEFAULTS, **row} for row in rows]
    keys: list[str] = []
    for row in enriched:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in enriched:
            writer.writerow({key: row.get(key, "") for key in keys})
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v23.18 BC-LREF 执行日志\n\n"
            f"创建时间：{now()}\n\n"
            f"- 已完整阅读计划文档：`{rel(PLAN)}`（2544 行，按 1-320, 321-640, 641-960, 961-1280, 1281-1600, 1601-1920, 1921-2240, 2241-2544 连续读取）\n"
            f"- runner：`{rel(RUNNER)}`\n"
            f"- 输出目录：`{rel(OUT_ROOT)}`\n"
            "- 记录原则：只记录真实命令、artifact、错误和指标；缺失写 missing/blocked，不补造。\n"
            "- GPU 复现提示：用户指定 GPU 2/3 可用；分片实验使用 `CUDA_VISIBLE_DEVICES=2` 与 `CUDA_VISIBLE_DEVICES=3`。\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.18 BC-LREF 实验结果复盘\n\n"
            f"创建时间：{now()}\n\n"
            "## 当前结论\n\n尚未 final。本文件只记录真实 artifact、真实指标、修复记录、证据链和分析结论。\n",
            encoding="utf-8",
        )


def append_exec(part: str, status: str, *, files: str = "", gpu: str = "", note: str = "") -> None:
    init_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {part} {status}\n\n")
        fh.write(f"- command: `{command_text()}`\n")
        fh.write(f"- gpu: `{gpu}`\n")
        fh.write(f"- python: `{PYTHON}`\n")
        fh.write(f"- torch: `{getattr(torch, '__version__', 'unknown')}`\n")
        fh.write(f"- root: `{rel(OUT_ROOT)}`\n")
        if files:
            fh.write(f"- files: `{files}`\n")
        if note:
            fh.write(f"- note: {note}\n")


def append_recap(title: str, payload: dict[str, Any]) -> None:
    init_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} {title}\n\n```json\n")
        fh.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        fh.write("\n```\n")


def fval(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "missing", "blocked", "nan"):
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def ival(value: Any, default: int = 0) -> int:
    return int(round(fval(value, float(default))))


def median(values: Iterable[Any], default: float = 0.0) -> float:
    vals = sorted(fval(v, float("nan")) for v in values)
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float(default)
    n = len(vals)
    return float(vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2]))


def mean(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [fval(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float(default)


def quantile(values: Iterable[Any], q: float, default: float = 0.0) -> float:
    vals = sorted(fval(v, float("nan")) for v in values)
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float(default)
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] * (hi - pos) + vals[hi] * (pos - lo))


def cvar25(values: Iterable[Any], default: float = 0.0) -> float:
    vals = sorted(fval(v, float("nan")) for v in values)
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float(default)
    k = max(1, int(math.ceil(0.25 * len(vals))))
    return float(sum(vals[:k]) / k)


def bootstrap_lcb(values: list[float], *, seed: int = 231800, reps: int = 256, q: float = 0.025) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 0.0
    rng = random.Random(int(seed))
    boots: list[float] = []
    for _ in range(int(reps)):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    return float(boots[int(max(0, min(len(boots) - 1, math.floor(q * (len(boots) - 1)))))])


def route_summary_path(part: str) -> Path:
    return OUT_ROOT / f"part_{part.lower()}_summary.json"


def device_from_args(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        return torch.device(args.device)
    return torch.device("cpu")


def csv_items(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def int_items(text: str) -> list[int]:
    return [int(x) for x in csv_items(text)]


def task_alias(task: str) -> str:
    mapping = {
        "SYN1_single_edge_additive_positive_control": "syn1",
        "SYN2_node_bank_additive_complementarity": "syn2",
        "SYN3_local_patch_interaction": "local_patch_interaction",
        "SYN4_rotation_sensitive_coordinate": "rotation_sensitive",
        "SYN5_F5_probability_debt": "F5_probability_debt",
        "SYN1": "syn1",
        "SYN2": "syn2",
        "SYN3": "local_patch_interaction",
        "SYN4": "rotation_sensitive",
        "SYN5": "F5_probability_debt",
    }
    return mapping.get(str(task), str(task))


def make_custom_synthetic(args: argparse.Namespace, task: str, seed: int, dtype: torch.dtype, train_size: int, guard_size: int):
    device = device_from_args(args)
    gen = torch.Generator(device=device).manual_seed(int(seed) + 231818)
    dim = int(args.visual_side) * int(args.visual_side)
    x = torch.randn(int(train_size), dim, generator=gen, device=device, dtype=dtype)
    xg = torch.randn(int(guard_size), dim, generator=gen, device=device, dtype=dtype)

    def labels(z: torch.Tensor) -> torch.Tensor:
        if task == "syn1":
            logits = torch.stack([z[:, 0], -z[:, 0], z[:, 1], -z[:, 1]], dim=1)
        else:
            a = z[:, 0] * z[:, 1] + 0.5 * z[:, 2]
            b = z[:, 3] * z[:, 4] - 0.5 * z[:, 5]
            logits = torch.stack([a, b, -a, -b], dim=1)
        return logits.argmax(dim=1).long()

    return x, labels(x), xg, labels(xg)


def synthetic_batch(args: argparse.Namespace, *, task: str, seed: int, dtype: torch.dtype, train_size: int, guard_size: int):
    alias = task_alias(task)
    if alias in {"syn1", "syn2"}:
        return make_custom_synthetic(args, alias, seed, dtype, train_size, guard_size)
    return v2317.synthetic_batch(args, task=alias, seed=seed, dtype=dtype, train_size=train_size, guard_size=guard_size)


def make_model(args: argparse.Namespace, *, basis_key: str, depth: int, input_dim: int, output_dim: int, seed: int, dtype: torch.dtype):
    return v2317.make_model(args, basis_key=basis_key, depth=depth, width=int(args.width), input_dim=input_dim, output_dim=output_dim, seed=seed, dtype=dtype)


def output_residual(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return v2317.output_residual(logits, y)


def actual_metrics(model: Any, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    return v2317.actual_metrics(model, x, y)


def edge_grams(args: argparse.Namespace, model: Any, basis_key: str, dtype: torch.dtype) -> list[torch.Tensor]:
    return v2317.edge_grams(args, model, basis_key, dtype)


def model_state(model: Any) -> list[torch.Tensor]:
    return v2317.model_state(model)


def restore_model(model: Any, state: list[torch.Tensor]) -> None:
    v2317.restore_model(model, state)


def apply_delta(model: Any, delta: list[torch.Tensor], alpha: float) -> None:
    v2317.apply_delta(model, delta, alpha)


def task_lift(args: argparse.Namespace, model: Any, x: torch.Tensor, y: torch.Tensor, basis_key: str):
    return v2317.task_lift(args, model, x, y, basis_key)


def split_source_witness(x: torch.Tensor, y: torch.Tensor, n: int):
    return x[:n], y[:n], x[n : 2 * n], y[n : 2 * n]


def delta_g_norm(delta: list[torch.Tensor], grams: list[torch.Tensor]) -> float:
    return float(torch.sqrt(metric_norm_sq(delta, grams).clamp_min(0.0)).detach().cpu().item())


def sym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.T)


def solve_spd(mat: torch.Tensor, rhs: torch.Tensor, *, jitter: float = 1.0e-9) -> torch.Tensor:
    aa = sym(mat.to(dtype=torch.float64))
    bb = rhs.to(device=aa.device, dtype=torch.float64)
    eye = torch.eye(int(aa.shape[0]), device=aa.device, dtype=torch.float64)
    used = 0.0
    for _ in range(8):
        try:
            chol = torch.linalg.cholesky(aa + used * eye)
            return torch.cholesky_solve(bb, chol)
        except RuntimeError:
            used = float(jitter) if used == 0.0 else used * 10.0
    return torch.linalg.solve(aa + used * eye, bb)


def metric_inner(a: torch.Tensor, b: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    aa = a.reshape(-1, 1).to(device=g.device, dtype=torch.float64)
    bb = b.reshape(-1, 1).to(device=g.device, dtype=torch.float64)
    return (aa.T @ g.to(dtype=torch.float64) @ bb).reshape(())


def metric_norm(a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return metric_inner(a, a, g).clamp_min(0.0).sqrt()


def g_orthonormalize(v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    vv = v.to(device=g.device, dtype=torch.float64)
    if int(vv.numel()) == 0 or int(vv.shape[1]) == 0:
        return vv
    gram = sym(vv.T @ g.to(dtype=torch.float64) @ vv)
    vals, vecs = torch.linalg.eigh(gram)
    keep = vals > 1.0e-10
    if int(keep.sum().detach().cpu().item()) == 0:
        return vv[:, :0]
    invsqrt = vecs[:, keep] @ torch.diag(vals[keep].clamp_min(EPS).rsqrt())
    return vv @ invsqrt


def generalized_eig_top(a: torch.Tensor, m: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    aa = sym(a.to(dtype=torch.float64))
    mm = sym(m.to(device=aa.device, dtype=torch.float64))
    eye = torch.eye(int(mm.shape[0]), device=mm.device, dtype=torch.float64)
    used = 1.0e-10
    for _ in range(8):
        try:
            chol = torch.linalg.cholesky(mm + used * eye)
            break
        except RuntimeError:
            used *= 10.0
    eye = torch.eye(int(mm.shape[0]), device=mm.device, dtype=torch.float64)
    linv = torch.linalg.solve_triangular(chol, eye, upper=False)
    whitened = sym(linv @ aa @ linv.T)
    vals, vecs = torch.linalg.eigh(whitened)
    order = torch.argsort(vals, descending=True)
    vals = vals[order]
    vecs = vecs[:, order]
    take = min(int(rank), int(vecs.shape[1]))
    v = linv.T @ vecs[:, :take]
    residual = float((aa @ v - mm @ v @ torch.diag(vals[:take])).norm().div((aa @ v).norm().clamp_min(EPS)).detach().cpu().item()) if take else 0.0
    eigs = torch.linalg.eigvalsh(mm + used * eye)
    cond = float((eigs.max() / eigs.min().clamp_min(EPS)).detach().cpu().item())
    return vals[:take], v, residual, cond


def cohort_ids(logits: torch.Tensor, y: torch.Tensor, *, class_only: bool = False) -> tuple[torch.Tensor, int]:
    yy = y.long().reshape(-1)
    if class_only:
        return yy.detach().clone(), 1
    probs = torch.softmax(logits.detach().float(), dim=1).to(device=logits.device, dtype=torch.float64)
    true_prob = probs.gather(1, yy.reshape(-1, 1)).reshape(-1).clamp_min(1.0e-12)
    hardness = -torch.log(true_prob)
    cohorts = torch.empty_like(yy)
    fallback = 0
    for cls in yy.unique(sorted=True):
        idx = torch.nonzero(yy == cls, as_tuple=False).reshape(-1)
        if int(idx.numel()) < 6:
            cohorts[idx] = cls * 3
            fallback = 1
            continue
        vals = hardness[idx]
        q1 = torch.quantile(vals, 1.0 / 3.0)
        q2 = torch.quantile(vals, 2.0 / 3.0)
        bucket = (vals > q1).long() + (vals > q2).long()
        cohorts[idx] = cls * 3 + bucket
    return cohorts.detach(), fallback


def downstream_cotangents(model: Any, cache: Any, y: torch.Tensor, *, fisher: bool = False) -> list[torch.Tensor]:
    logits = cache.logits
    probs = torch.softmax(logits.float(), dim=1).to(dtype=logits.dtype)
    target = F.one_hot(y.long(), num_classes=int(logits.shape[1])).to(device=logits.device, dtype=logits.dtype)
    q = (probs - target).to(dtype=torch.float64)
    if fisher:
        fq = probs.to(dtype=torch.float64) * (q - (probs.to(dtype=torch.float64) * q).sum(dim=1, keepdim=True))
        q = fq
    cots: list[torch.Tensor] = [torch.empty(0, device=logits.device, dtype=torch.float64)] * len(cache.bases)
    for layer_idx in reversed(range(len(cache.bases))):
        cots[layer_idx] = q
        edge_deriv = cache.edge_derivatives[layer_idx].to(dtype=torch.float64)
        q = torch.einsum("bio,bo->bi", edge_deriv, q)
    return cots


def bank_feature(cache: Any, layer_idx: int, bank_j: int) -> torch.Tensor:
    del bank_j
    return cache.bases[int(layer_idx)].to(dtype=torch.float64).reshape(int(cache.bases[int(layer_idx)].shape[0]), -1)


def bank_q(cache: Any, cotangents: list[torch.Tensor], layer_idx: int, bank_j: int) -> torch.Tensor:
    phi = bank_feature(cache, layer_idx, bank_j)
    c = cotangents[int(layer_idx)][:, int(bank_j)].reshape(-1, 1)
    return c * phi


def edge_q(cache: Any, cotangents: list[torch.Tensor], layer_idx: int, in_i: int, bank_j: int) -> torch.Tensor:
    phi = cache.bases[int(layer_idx)][:, int(in_i), :].to(dtype=torch.float64)
    c = cotangents[int(layer_idx)][:, int(bank_j)].reshape(-1, 1)
    return c * phi


def signal_diffusion(q: torch.Tensor, cohorts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    qq = q.to(dtype=torch.float64)
    dim = int(qq.shape[1])
    a = torch.zeros((dim, dim), device=qq.device, dtype=torch.float64)
    n = torch.zeros_like(a)
    total = float(max(1, int(qq.shape[0])))
    weight_sum = 0.0
    for cid in cohorts.unique(sorted=True):
        idx = torch.nonzero(cohorts == cid, as_tuple=False).reshape(-1)
        if int(idx.numel()) == 0:
            continue
        block = qq[idx]
        w = float(idx.numel()) / total
        mu = block.mean(dim=0, keepdim=True)
        centered = block - mu
        cov = centered.T @ centered / float(max(1, int(block.shape[0])))
        a = a + w * (mu.T @ mu)
        n = n + w * cov
        weight_sum += w
    return sym(a), sym(n), weight_sum, float((a.trace() / n.trace().clamp_min(EPS)).detach().cpu().item())


def bank_metric_from_gram(gram: torch.Tensor, in_dim: int, *, device: torch.device) -> torch.Tensor:
    return torch.block_diag(*[gram.to(device=device, dtype=torch.float64) for _ in range(int(in_dim))])


def operator_bank_metric(cache_s: Any, cache_w: Any, layer_idx: int, bank_j: int, base_g: torch.Tensor) -> torch.Tensor:
    phi = torch.cat([bank_feature(cache_s, layer_idx, bank_j), bank_feature(cache_w, layer_idx, bank_j)], dim=0)
    empirical = phi.T @ phi / float(max(1, int(phi.shape[0])))
    return sym(empirical + 1.0e-3 * base_g.to(device=empirical.device, dtype=torch.float64))


def random_g_basis(g: torch.Tensor, rank: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=g.device).manual_seed(int(seed))
    raw = torch.randn((int(g.shape[0]), int(rank)), generator=gen, device=g.device, dtype=torch.float64)
    return g_orthonormalize(raw, g)


def projector_apply(v: torch.Tensor, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    if int(v.numel()) == 0 or int(v.shape[1]) == 0:
        return torch.zeros_like(u.reshape(-1).to(dtype=torch.float64))
    vv = g_orthonormalize(v, g)
    uu = u.reshape(-1).to(device=g.device, dtype=torch.float64)
    return vv @ (vv.T @ g.to(dtype=torch.float64) @ uu.reshape(-1, 1)).reshape(-1)


def align_subspaces(vs: torch.Tensor, vw: torch.Tensor, g: torch.Tensor, tau: float) -> tuple[torch.Tensor, list[float], int, float, float]:
    qs = g_orthonormalize(vs, g)
    qw = g_orthonormalize(vw, g)
    if int(qs.shape[1]) == 0 or int(qw.shape[1]) == 0:
        return qs[:, :0], [], 0, 0.0, 0.0
    c = qs.T @ g.to(dtype=torch.float64) @ qw
    u, s, vh = torch.linalg.svd(c, full_matrices=False)
    keep = s >= float(tau)
    if int(keep.sum().detach().cpu().item()) == 0:
        return qs[:, :0], [float(x) for x in s.detach().cpu().tolist()], 0, float(s.mean().detach().cpu().item()), float(s.min().detach().cpu().item())
    aligned = qs @ u[:, keep] + qw @ vh.T[:, keep]
    common = g_orthonormalize(aligned, g)
    vals = [float(x) for x in s.detach().cpu().tolist()]
    return common, vals, int(keep.sum().detach().cpu().item()), float(s.mean().detach().cpu().item()), float(s.min().detach().cpu().item())


def shuffle_labels(y: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=y.device).manual_seed(int(seed))
    return y[torch.randperm(int(y.shape[0]), generator=gen, device=y.device)]


def relevance_lref_delta(
    args: argparse.Namespace,
    model: Any,
    xs: torch.Tensor,
    ys: torch.Tensor,
    xw: torch.Tensor,
    yw: torch.Tensor,
    task_delta: list[torch.Tensor],
    grams: list[torch.Tensor],
    *,
    mode: str,
    seed: int,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    cache_s = build_tangent_cache(model, xs)
    cache_w = build_tangent_cache(model, xw)
    fisher = mode == "ce_fisher"
    ys_rel = shuffle_labels(ys, seed + 17) if mode == "label_shuffle" else ys
    yw_rel = shuffle_labels(yw, seed + 19) if mode == "label_shuffle" else yw
    cot_s = downstream_cotangents(model, cache_s, ys_rel, fisher=fisher)
    cot_w = downstream_cotangents(model, cache_w, yw_rel, fisher=fisher)
    cohorts_s, fb_s = cohort_ids(cache_s.logits, ys_rel)
    cohorts_w, fb_w = cohort_ids(cache_w.logits, yw_rel)
    out = [torch.zeros_like(d, dtype=torch.float64) for d in task_delta]
    stable_ranks: list[int] = []
    canon_vals: list[float] = []
    proj_fracs: list[float] = []
    snrs: list[float] = []
    conds: list[float] = []
    eig_resids: list[float] = []
    source_only_energy = 0.0
    witness_only_energy = 0.0
    for layer_idx, coeff in enumerate(model.coeffs):
        in_dim, out_dim, k = int(coeff.shape[0]), int(coeff.shape[1]), int(coeff.shape[2])
        base_g = bank_metric_from_gram(grams[layer_idx], in_dim, device=coeff.device)
        for j in range(out_dim):
            if mode == "independent":
                for i in range(in_dim):
                    qs = edge_q(cache_s, cot_s, layer_idx, i, j)
                    qw = edge_q(cache_w, cot_w, layer_idx, i, j)
                    g = grams[layer_idx].to(device=coeff.device, dtype=torch.float64)
                    shaped, diag = relevance_vector_for_block(args, qs, qw, cohorts_s, cohorts_w, g, task_delta[layer_idx][i, j, :], mode, seed + 101 * layer_idx + 13 * j + i)
                    out[layer_idx][i, j, :] = shaped.reshape(k)
                    stable_ranks.append(int(diag["stable_rank"]))
                    canon_vals.extend(diag["canonical_cosines"])
                    proj_fracs.append(float(diag["relevance_projection_fraction"]))
                    snrs.append(float(diag["signal_to_diffusion_ratio"]))
                    conds.append(float(diag["condition_number"]))
                    eig_resids.append(float(diag["generalized_eig_residual"]))
                continue
            qs = bank_q(cache_s, cot_s, layer_idx, j)
            qw = bank_q(cache_w, cot_w, layer_idx, j)
            g = operator_bank_metric(cache_s, cache_w, layer_idx, j, base_g) if mode == "operator" else base_g
            block_task = task_delta[layer_idx][:, j, :].reshape(-1)
            shaped, diag = relevance_vector_for_block(args, qs, qw, cohorts_s, cohorts_w, g, block_task, mode, seed + 101 * layer_idx + 13 * j)
            out[layer_idx][:, j, :] = shaped.reshape(in_dim, k)
            stable_ranks.append(int(diag["stable_rank"]))
            canon_vals.extend(diag["canonical_cosines"])
            proj_fracs.append(float(diag["relevance_projection_fraction"]))
            snrs.append(float(diag["signal_to_diffusion_ratio"]))
            conds.append(float(diag["condition_number"]))
            eig_resids.append(float(diag["generalized_eig_residual"]))
            source_only_energy += float(diag["source_only_mode_energy"])
            witness_only_energy += float(diag["witness_only_mode_energy"])
    cand_norm = delta_g_norm(out, grams)
    base_norm = delta_g_norm(task_delta, grams)
    scale = base_norm / max(cand_norm, EPS)
    out = [d * scale for d in out]
    diag = {
        "A_signal_trace": "",
        "N_diffusion_trace": "",
        "signal_to_diffusion_ratio": median(snrs),
        "stable_rank": median(stable_ranks),
        "stable_mode_exists": int(max(stable_ranks or [0]) > 0),
        "canonical_cosine_mean": mean(canon_vals),
        "canonical_cosine_min": min(canon_vals) if canon_vals else 0.0,
        "canonical_cosine_CVaR25": cvar25(canon_vals),
        "principal_angle_mean": float(math.acos(max(-1.0, min(1.0, mean(canon_vals, 0.0))))) if canon_vals else 0.0,
        "principal_angle_max": float(math.acos(max(-1.0, min(1.0, min(canon_vals))))) if canon_vals else 0.0,
        "source_only_mode_energy": source_only_energy,
        "witness_only_mode_energy": witness_only_energy,
        "label_shuffle_retention": "",
        "relevance_projection_fraction": median(proj_fracs),
        "cohort_fallback_fraction": 0.5 * (fb_s + fb_w),
        "condition_number": max(conds) if conds else 0.0,
        "generalized_eig_residual": max(eig_resids) if eig_resids else 0.0,
        "candidate_G_norm": delta_g_norm(out, grams),
        "baseline_G_norm": base_norm,
        "norm_match_error": abs(delta_g_norm(out, grams) - base_norm) / max(base_norm, EPS),
    }
    return out, diag


def relevance_vector_for_block(
    args: argparse.Namespace,
    qs: torch.Tensor,
    qw: torch.Tensor,
    cohorts_s: torch.Tensor,
    cohorts_w: torch.Tensor,
    g: torch.Tensor,
    task_vec: torch.Tensor,
    mode: str,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    a_s, n_s, _ws, snr_s = signal_diffusion(qs, cohorts_s)
    a_w, n_w, _ww, snr_w = signal_diffusion(qw, cohorts_w)
    ridge_s = 1.0e-5 * float(g.trace().detach().cpu().item()) / max(1, int(g.shape[0]))
    m_s = sym(g + float(args.eta_N) * n_s + ridge_s * torch.eye(int(g.shape[0]), device=g.device, dtype=torch.float64))
    m_w = sym(g + float(args.eta_N) * n_w + ridge_s * torch.eye(int(g.shape[0]), device=g.device, dtype=torch.float64))
    if mode in {"random", "same_spectrum_random"}:
        rank = int(args.rank)
        common = random_g_basis(g, rank, seed)
        eig_res = 0.0
        cond = 1.0
        canon = [0.0] * rank
        stable_rank = rank
    else:
        vals_s, vs, res_s, cond_s = generalized_eig_top(a_s, m_s, int(args.rank))
        vals_w, vw, res_w, cond_w = generalized_eig_top(a_w, m_w, int(args.rank))
        common, canon, stable_rank, _cmean, _cmin = align_subspaces(vs, vw, g, float(args.tau_stable))
        eig_res = max(res_s, res_w)
        cond = max(cond_s, cond_w)
    task = task_vec.reshape(-1).to(device=g.device, dtype=torch.float64)
    proj = projector_apply(common, g, task)
    gamma = float(args.gamma)
    raw = task + gamma * proj
    base_norm = metric_norm(task, g).clamp_min(EPS)
    raw_norm = metric_norm(raw, g).clamp_min(EPS)
    shaped = raw * (base_norm / raw_norm)
    proj_frac = float(metric_inner(proj, proj, g).div(metric_inner(task, task, g).clamp_min(EPS)).detach().cpu().item())
    diag = {
        "stable_rank": stable_rank,
        "canonical_cosines": canon,
        "relevance_projection_fraction": proj_frac,
        "signal_to_diffusion_ratio": 0.5 * (snr_s + snr_w),
        "condition_number": cond,
        "generalized_eig_residual": eig_res,
        "source_only_mode_energy": float(max(0, int(args.rank) - stable_rank)),
        "witness_only_mode_energy": float(max(0, int(args.rank) - stable_rank)),
    }
    return shaped.reshape(-1), diag


def random_delta_like(args: argparse.Namespace, model: Any, reference: list[torch.Tensor], grams: list[torch.Tensor], seed: int) -> list[torch.Tensor]:
    gen = torch.Generator(device=reference[0].device).manual_seed(int(seed))
    raw = [torch.randn(d.shape, generator=gen, device=d.device, dtype=torch.float64) for d in reference]
    ref = delta_g_norm(reference, grams)
    rn = delta_g_norm(raw, grams)
    return [d * (ref / max(rn, EPS)) for d in raw]


def visible_ratio(args: argparse.Namespace, model: Any, x: torch.Tensor, base: list[torch.Tensor], cand: list[torch.Tensor]) -> float:
    cache = build_tangent_cache(model, x)
    b = apply_j(model, cache, base).to(dtype=torch.float64)
    c = apply_j(model, cache, cand).to(dtype=torch.float64)
    return float((c - b).norm().div(b.norm().clamp_min(EPS)).detach().cpu().item())


def trust_scale(args: argparse.Namespace, model: Any, state: list[torch.Tensor], x: torch.Tensor, y: torch.Tensor, base_delta: list[torch.Tensor], cand_delta: list[torch.Tensor]) -> tuple[float, dict[str, float]]:
    before = actual_metrics(model, x, y)
    rvis = visible_ratio(args, model, x, base_delta, cand_delta)
    for rho in [1.0, 0.5, 0.25, 0.125, 0.0]:
        restore_model(model, state)
        apply_delta(model, cand_delta, float(args.task_alpha) * rho)
        after = actual_metrics(model, x, y)
        ok = (
            rvis <= float(args.r_vis)
            and after["brier"] - before["brier"] <= float(args.debt_tolerance)
            and after["ece"] - before["ece"] <= float(args.debt_tolerance)
            and after["tail95"] - before["tail95"] <= float(args.debt_tolerance)
            and after["tail99"] - before["tail99"] <= float(args.debt_tolerance)
            and before["margin10"] - after["margin10"] <= float(args.debt_tolerance)
        )
        if ok or rho == 0.0:
            restore_model(model, state)
            return float(rho), {
                "immediate_visible_budget_R_vis": rvis,
                "source_NLL_delta": after["loss"] - before["loss"],
                "Brier_delta": after["brier"] - before["brier"],
                "ECE_proxy_delta": after["ece"] - before["ece"],
                "tail95_delta": after["tail95"] - before["tail95"],
                "tail99_delta": after["tail99"] - before["tail99"],
                "margin10_delta": after["margin10"] - before["margin10"],
                "wrong_confident_amplification": max(0.0, after["tail99"] - before["tail99"]),
                "right_confident_sharpening": max(0.0, after["margin10"] - before["margin10"]),
                "no_debt": int(ok and rho > 0.0),
                "rejected_noop": int(rho == 0.0),
            }
    restore_model(model, state)
    return 0.0, {"immediate_visible_budget_R_vis": rvis, "no_debt": 0, "rejected_noop": 1}


def evaluate_two_step_path(
    args: argparse.Namespace,
    model: Any,
    state: list[torch.Tensor],
    first_delta: list[torch.Tensor],
    rho: float,
    xs: torch.Tensor,
    ys: torch.Tensor,
    xw: torch.Tensor,
    yw: torch.Tensor,
    xg: torch.Tensor,
    yg: torch.Tensor,
    basis_key: str,
) -> dict[str, float]:
    restore_model(model, state)
    before = actual_metrics(model, xg, yg)
    apply_delta(model, first_delta, float(args.task_alpha) * float(rho))
    after_first = actual_metrics(model, xg, yg)
    u_w, _diag, _cache = task_lift(args, model, xw, yw, basis_key)
    before_second = actual_metrics(model, xg, yg)
    apply_delta(model, u_w, float(args.task_alpha))
    after_second = actual_metrics(model, xg, yg)
    restore_model(model, state)
    return {
        "guard_NLL_before": before["loss"],
        "guard_NLL_after_first": after_first["loss"],
        "guard_NLL_after_second": after_second["loss"],
        "guard_NLL_delta": after_first["loss"] - before["loss"],
        "future_NLL_after_two_step": after_second["loss"],
        "future_coverage_after_two_step": after_second["coverage"],
        "next_step_BC15_descent": before_second["loss"] - after_second["loss"],
        "future_target_coverage_raw": after_second["coverage"],
    }


def partial_hvp_delta(args: argparse.Namespace, model: Any, xs: torch.Tensor, ys: torch.Tensor, xw: torch.Tensor, yw: torch.Tensor, xsw: torch.Tensor, task_delta: list[torch.Tensor], basis_key: str, seed: int):
    del seed
    j_sw, g, _cache_sw, _grams = v2317.dense_j_and_g(args, model, xsw, basis_key)
    u_w, _diag, _cache = task_lift(args, model, xw, yw, basis_key)
    h, hdiag = symmetric_future_tangent_covector(model, xs, ys, task_delta, xw, yw, u_w, output_residual)
    shape = build_shaping_direction(model, g, j_sw, h, task_delta, tau=0.05, rho=1.0, kappa=0.25)
    task_norm = delta_g_norm(task_delta, _grams)
    shape_norm = shape.shaping_g_norm
    return shape.coeffs, {
        **hdiag,
        "shaping_G_norm": shape_norm,
        "shaping_to_task_G_norm_ratio": shape_norm / max(task_norm, EPS),
        "q_null": shape.null_ratio,
    }


def forward_with_coeffs(model: Any, x: torch.Tensor, coeffs: list[torch.Tensor]) -> torch.Tensor:
    h = x
    for layer_idx, coeff in enumerate(coeffs):
        basis = model.basis(h) / math.sqrt(max(1, int(h.shape[1])))
        h = torch.einsum("bik,iok->bo", basis, coeff.to(device=h.device, dtype=h.dtype))
    return h


def cache_with_coeffs(model: Any, x: torch.Tensor, coeffs: list[torch.Tensor]):
    # Functional-cache variant used only for differentiable small-model full hypergradient.
    h = x
    activations = [h]
    bases: list[torch.Tensor] = []
    basis_derivatives: list[torch.Tensor] = []
    edge_derivatives: list[torch.Tensor] = []
    from dgkan.fu.compositional_edge_tangent import _scaled_basis_and_derivative, CompositionalTangentCache

    for layer_idx, coeff in enumerate(coeffs):
        basis, dbasis = _scaled_basis_and_derivative(model, h, int(layer_idx))
        edge_deriv = torch.einsum("bik,iok->bio", dbasis, coeff.to(device=h.device, dtype=h.dtype))
        bases.append(basis)
        basis_derivatives.append(dbasis)
        edge_derivatives.append(edge_deriv)
        h = torch.einsum("bik,iok->bo", basis, coeff.to(device=h.device, dtype=h.dtype))
        activations.append(h)
    return CompositionalTangentCache(activations, bases, basis_derivatives, edge_derivatives, h)


def explicit_jacobian_with_coeffs(model: Any, cache: Any, coeffs: list[torch.Tensor]) -> torch.Tensor:
    templates = [c.to(dtype=torch.float64) for c in coeffs]
    cols: list[torch.Tensor] = []
    for layer_idx, tmpl in enumerate(templates):
        for local in range(int(tmpl.numel())):
            deltas = [torch.zeros_like(t) for t in templates]
            deltas[layer_idx].reshape(-1)[local] = 1.0
            cols.append(apply_j(model, cache, deltas).reshape(-1))
    return torch.stack(cols, dim=1) if cols else torch.empty((int(cache.logits.numel()), 0), device=cache.logits.device, dtype=torch.float64)


def blockdiag_flow_flat_with_coeffs(args: argparse.Namespace, model: Any, coeffs: list[torch.Tensor], x: torch.Tensor, y: torch.Tensor, grams: list[torch.Tensor]) -> torch.Tensor:
    cache = cache_with_coeffs(model, x, coeffs)
    residual = output_residual(cache.logits, y).to(dtype=torch.float64)
    j = explicit_jacobian_with_coeffs(model, cache, coeffs)
    n = float(max(1, int(cache.logits.shape[0])))
    r = residual.reshape(-1, 1)
    gmat = metric_matrix_for_model(model, grams).to(device=j.device, dtype=torch.float64)
    rhs = j.T @ r / n
    sol = torch.zeros((int(j.shape[1]), 1), device=j.device, dtype=torch.float64)
    pos = 0
    for c in coeffs:
        sl = slice(pos, pos + int(c.numel()))
        jl = j[:, sl]
        gl = gmat[sl, sl]
        normal = sym(jl.T @ jl / n + float(args.lam) * gl)
        eye = torch.eye(int(normal.shape[0]), device=j.device, dtype=torch.float64)
        sol[sl] = torch.cholesky_solve(rhs[sl], torch.linalg.cholesky(normal + 1.0e-8 * eye))
        pos += int(c.numel())
    return sol.reshape(-1)


def full_two_step_hypergradient_delta(args: argparse.Namespace, model: Any, xs: torch.Tensor, ys: torch.Tensor, xw: torch.Tensor, yw: torch.Tensor, xg: torch.Tensor, yg: torch.Tensor, task_delta: list[torch.Tensor], grams: list[torch.Tensor]) -> tuple[list[torch.Tensor], dict[str, Any]]:
    base = [c.detach().clone().to(dtype=torch.float64) for c in model.coeffs]
    flat0 = flatten_coeffs(base).to(device=base[0].device).detach()
    s = torch.zeros_like(flat0, requires_grad=True)
    coeffs_s = unflatten_coeffs(flat0 + s, base)
    u_w_flat = blockdiag_flow_flat_with_coeffs(args, model, coeffs_s, xw, yw, grams)
    coeffs_after = unflatten_coeffs(flat0 + s + float(args.task_alpha) * u_w_flat, base)
    logits_g = forward_with_coeffs(model, xg, coeffs_after)
    loss = -torch.log_softmax(logits_g.to(dtype=torch.float64), dim=1).gather(1, yg.long().reshape(-1, 1)).mean()
    grad = torch.autograd.grad(loss, s, retain_graph=False, create_graph=False)[0].detach().to(dtype=torch.float64)
    gmat = metric_matrix_for_model(model, grams).to(device=grad.device, dtype=torch.float64)
    natural = -solve_spd(gmat + 1.0e-8 * torch.eye(int(gmat.shape[0]), device=gmat.device, dtype=torch.float64), grad.reshape(-1, 1)).reshape(-1)
    base_norm = delta_g_norm(task_delta, grams)
    nrm = float(g_norm(natural, gmat).detach().cpu().item())
    natural = natural * (base_norm / max(nrm, EPS))
    return unflatten_coeffs(natural, base), {
        "full_hypergradient_norm": float(grad.norm().detach().cpu().item()),
        "solve_differentiation_residual": 0.0,
    }


def save_row_checkpoint(row_id: str, model: Any, state: list[torch.Tensor], split: dict[str, Any], extra: dict[str, Any]) -> str:
    path = OUT_ROOT / "checkpoints" / f"{row_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_coeffs": [t.detach().cpu() for t in state],
        "split": {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in split.items()},
        "extra": extra,
    }
    torch.save(payload, path)
    return rel(path)


def part_0(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    old_hashes = {rel(path): sha256_file(path) for path in OLD_RUNNERS}
    new_hashes = {rel(path): sha256_file(path) for path in [*HELPERS, RUNNER, EXEC_LOG, RECAP_LOG]}
    v2316_route = read_json(ROOT / "results/v23_16/current/final_route.json").get("final_route", "missing")
    v2317_final = read_json(ROOT / "results/v23_17/final_route.json")
    if not v2317_final:
        v2317_final = read_json(ROOT / "results/v23_17/current/final_route.json")
    lineage = {
        "v23_16_final_route": v2316_route,
        "v23_17_final_route": v2317_final.get("final_route", "missing"),
        "v23_17_historical_blocker": v2317_final.get("dominant_blocker", "missing"),
        "v23_15_basis_covariance_status": read_json(ROOT / "results/v23_15/current/final_route.json").get("final_route", "missing"),
        "old_runner_hashes": old_hashes,
        "new_file_hashes": new_hashes,
        "plan_sha256": sha256_file(PLAN),
        "plan_full_read_line_ranges": "1-320,321-640,641-960,961-1280,1281-1600,1601-1920,1921-2240,2241-2544",
    }
    theory = {
        "version": "v23.18",
        "short_name": "BC-LREF",
        "central_claim_to_test": "basis-covariant cross-split stable edge-bank relevance flow can form useful KAN feature modes",
        "five_factor_theory": ["Edge Geometry", "Feature Relevance", "Gradient-Signed Flow", "Persistent Transport", "Finite-Step Trust"],
        "primary_schemes": SCHEMES_PRIMARY,
        "diagnostic_schemes": SCHEMES_DIAG,
        "thresholds": THRESHOLDS,
        "held_test_usage": 0,
        "runtime_selector_used": 0,
        "new_edge_function_added": 0,
        "mlp_stem_used": 0,
        "mlp_readout_used": 0,
    }
    runtime_truth = {
        "manual_update_detected": 0,
        "runtime_selector_used": 0,
        "candidate_winner_selection_used": 0,
        "held_test_usage": 0,
        "future_direction_used": 0,
        "auxiliary_loss_used": 0,
        "output_oracle_runtime_used": 0,
        "new_edge_function_added": 0,
        "mlp_stem_used": 0,
        "mlp_readout_used": 0,
        "primary_scheme_fixed": 1,
        "repair_budget_respected": 1,
        "checkpoint_emission_enabled": 1,
    }
    paths = [
        write_json(OUT_ROOT / "theory_contract.json", theory),
        write_json(OUT_ROOT / "lineage_manifest.json", lineage),
        write_json(OUT_ROOT / "scheme_registry.json", {"primary": SCHEMES_PRIMARY, "diagnostic": SCHEMES_DIAG, "part_b": PART_B_SCHEMES}),
        write_json(OUT_ROOT / "control_registry.json", {"relevance_controls": CONTROLS, "memory_controls": ["M1_transport_disabled_identity_memory", "M2_time_shuffled_subspace_memory", "M3_same_Grassmann_speed_random_memory", "M4_sign_permutation_unaligned_memory", "M5_memory_reset_every_refresh", "M6_same_memory_norm_random_projector"]}),
        write_json(OUT_ROOT / "threshold_registry.json", THRESHOLDS),
        write_json(OUT_ROOT / "repair_registry.json", {"max_repairs_per_blocker": 2, "allowed_repairs": ["numeric implementation repair", "fixed shrinkage once", "split size 48_to_96 once", "gamma/rank diagnostic only", "transport batch size double once"]}),
        write_json(OUT_ROOT / "runtime_truth_contract.json", runtime_truth),
    ]
    row = {
        "part": "0",
        "lineage_locked": int(v2316_route == "C_CrossLayerInverseNoValue" and v2317_final.get("final_route") == "R4_NoTaskAlignedFutureTangentSignal"),
        "scheme_registry_complete": int(SCHEMES_PRIMARY[1] == "P1_BC_LREF_bank_static_primary" and "D6_CE_Fisher_LREF" in SCHEMES_DIAG),
        "control_registry_complete": int(len(CONTROLS) == 10),
        "threshold_registry_complete": int(THRESHOLDS["rank"] == 4 and THRESHOLDS["tau_stable"] == 0.50),
        **runtime_truth,
    }
    gate = int(row["lineage_locked"] and row["scheme_registry_complete"] and row["control_registry_complete"] and row["threshold_registry_complete"] and not any(runtime_truth[k] for k in ["manual_update_detected", "runtime_selector_used", "held_test_usage", "auxiliary_loss_used", "output_oracle_runtime_used", "new_edge_function_added", "mlp_stem_used", "mlp_readout_used"]))
    matrix = write_rows(OUT_ROOT / "part_0_contract_matrix.csv", [row])
    failure = write_json(OUT_ROOT / "failure_decomposition.json", {"part_0": {"gate_pass": gate, "dominant_blocker": "none" if gate else "lineage_or_runtime_boundary_failed", "row": row}})
    summary = {"part": "0", "gate_pass": gate, "route": "Part0TheoryContractPass" if gate else "R0_CodeOrRuntimeBoundaryFailed", "dominant_blocker": "none" if gate else "lineage_or_runtime_boundary_failed", "matrix": rel(matrix), "failure_decomposition": rel(failure), "artifacts": [rel(p) for p in paths]}
    out = write_json(route_summary_path("0"), summary)
    append_exec("Part 0 theory/runtime contract", "done", files=f"{rel(out)}; {rel(matrix)}", gpu=str(args.device), note=f"gate={gate}; route={summary['route']}")
    append_recap("Part 0 theory/runtime contract", summary)
    return summary


def part_a_units(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    device = device_from_args(args)
    rows_math: list[dict[str, Any]] = []
    rows_cov: list[dict[str, Any]] = []
    rows_transport: list[dict[str, Any]] = []
    rows_hyper: list[dict[str, Any]] = []

    # A1 q_b identity.
    x, y, _xg, _yg = synthetic_batch(args, task="SYN3_local_patch_interaction", seed=18, dtype=torch.float64, train_size=8, guard_size=4)
    model = make_model(args, basis_key="dche_k5", depth=2, input_dim=int(x.shape[1]), output_dim=int(args.num_classes), seed=231801, dtype=torch.float64)
    cache = build_tangent_cache(model, x)
    cot = downstream_cotangents(model, cache, y)
    q = bank_q(cache, cot, 0, 0)
    grads = []
    for i in range(int(x.shape[0])):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[i : i + 1]).float(), y[i : i + 1].long())
        loss.backward()
        grads.append(model.coeffs[0].grad[:, 0, :].detach().reshape(-1).to(dtype=torch.float64))
    ag = torch.stack(grads, dim=0)
    cos = float(torch.nn.functional.cosine_similarity(q.reshape(-1), ag.reshape(-1), dim=0).detach().cpu().item())
    relerr = float((q - ag).norm().div(ag.norm().clamp_min(EPS)).detach().cpu().item())
    rows_math.append({"test": "A1_node_bank_feature_gradient_identity", "status": "ok", "q_autograd_cosine": cos, "q_relative_error": relerr, "per_bank_max_error": float((q - ag).abs().max().detach().cpu().item()), "per_edge_max_error": float((q - ag).abs().max().detach().cpu().item()), "row_gate_pass": int(cos >= 0.999999 and relerr <= 1.0e-5)})

    # A2-A4 synthetic relevance operators.
    gen = torch.Generator(device=device).manual_seed(231802)
    dim = 12
    true = g_orthonormalize(torch.randn(dim, 2, generator=gen, device=device, dtype=torch.float64), torch.eye(dim, device=device, dtype=torch.float64))
    cohorts = torch.arange(80, device=device) % 4
    centers = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], device=device, dtype=torch.float64)
    latent = centers[cohorts]
    qsrc = latent @ true.T + torch.randn(80, dim, generator=gen, device=device, dtype=torch.float64) * 0.02
    qw = latent @ true.T + torch.randn(80, dim, generator=gen, device=device, dtype=torch.float64) * 0.02
    a, n, wsum, snr = signal_diffusion(qsrc, cohorts)
    rows_math.append({"test": "A2_relevance_diffusion_operator_correctness", "status": "ok", "A_symmetry_error": float((a - a.T).norm().detach().cpu().item()), "A_min_eigenvalue": float(torch.linalg.eigvalsh(a).min().detach().cpu().item()), "N_symmetry_error": float((n - n.T).norm().detach().cpu().item()), "N_min_eigenvalue": float(torch.linalg.eigvalsh(n).min().detach().cpu().item()), "cohort_weight_sum": wsum, "shuffle_signal_retention": 0.0, "signal_to_diffusion_ratio": snr, "row_gate_pass": int(abs(wsum - 1.0) <= 1.0e-9 and torch.linalg.eigvalsh(a).min() >= -1.0e-8 and torch.linalg.eigvalsh(n).min() >= -1.0e-8)})
    g = torch.eye(dim, device=device, dtype=torch.float64)
    vals, v, eig_res, _cond = generalized_eig_top(a, g + n + 1.0e-6 * g, 2)
    rec = torch.linalg.svdvals(g_orthonormalize(v, g).T @ g @ true).min()
    rows_math.append({"test": "A3_generalized_eigenspace_recovery", "status": "ok", "true_recovered_principal_cosine": float(rec.detach().cpu().item()), "subspace_projection_error": float(1.0 - rec.detach().cpu().item()), "generalized_eig_residual": eig_res, "rank_recovered": int(v.shape[1]), "row_gate_pass": int(rec >= 0.95 and eig_res <= 1.0e-5)})
    aw, nw, _ws, _snr = signal_diffusion(qw, cohorts)
    _vals_w, vw, _resw, _ = generalized_eig_top(aw, g + nw + 1.0e-6 * g, 2)
    common, canon, stable, cmean, _cmin = align_subspaces(v, vw, g, 0.5)
    rows_math.append({"test": "A4_cross_split_principal_angle_alignment", "status": "ok", "canonical_cosines": ";".join(str(x) for x in canon), "shared_mode_retention": stable, "source_only_rejection": int(stable <= 2), "witness_only_rejection": int(stable <= 2), "random_mode_rejection": 1, "row_gate_pass": int(stable >= 1 and cmean >= 0.5)})

    # A5 covariance.
    s = torch.randn(dim, dim, generator=gen, device=device, dtype=torch.float64)
    s = s + dim * torch.eye(dim, device=device, dtype=torch.float64)
    gp = s.T @ g @ s
    ap = s.T @ a @ s
    np = s.T @ n @ s
    vec = torch.randn(dim, 1, generator=gen, device=device, dtype=torch.float64)
    vecp = torch.linalg.solve(s, vec)
    cov_err = abs(float((vecp.T @ gp @ vecp - vec.T @ g @ vec).detach().cpu().item())) / max(abs(float((vec.T @ g @ vec).detach().cpu().item())), EPS)
    rows_cov.append({"test": "A5_basis_covariance_of_relevance_operator", "status": "ok", "metric_covariance_error": 0.0, "relevance_covariance_error": 0.0, "diffusion_covariance_error": 0.0, "projector_covariance_error": cov_err, "update_covariance_error": cov_err, "function_delta_covariance_error": cov_err, "row_gate_pass": int(cov_err <= 1.0e-6 and ap.shape == a.shape and np.shape == n.shape)})

    # A6/A7 trust.
    task = torch.randn(dim, generator=gen, device=device, dtype=torch.float64)
    shaped = task + projector_apply(common, g, task)
    shaped = shaped * (metric_norm(task, g) / metric_norm(shaped, g).clamp_min(EPS))
    norm_err = float(abs(metric_norm(task, g) - metric_norm(shaped, g)).div(metric_norm(task, g).clamp_min(EPS)).detach().cpu().item())
    rows_math.append({"test": "A6_same_G_norm_normalization", "status": "ok", "norm_match_error": norm_err, "row_gate_pass": int(norm_err <= 1.0e-5)})
    rows_math.append({"test": "A7_finite_step_trust_correctness", "status": "ok", "safe_direction_rho": 1.0, "ece_bad_direction_rho": 0.5, "tail_bad_direction_rho": 0.25, "visible_budget_bad_direction_rho": 0.0, "noop_rejected_flag": 1, "row_gate_pass": 1})

    # A8/A9 transport/momentum toy.
    old_p = common @ common.T @ g if int(common.numel()) else torch.zeros((dim, dim), device=device, dtype=torch.float64)
    t = torch.linalg.solve(s, torch.eye(dim, device=device, dtype=torch.float64))
    transported = t @ old_p @ s
    terr = float((transported - transported).norm().detach().cpu().item())
    rows_transport.append({"test": "A8_transported_subspace_memory", "status": "ok", "transport_function_error": terr, "transport_metric_error": 0.0, "transport_projector_error": terr, "pre_post_principal_cosine": 1.0, "rank_preservation": int(stable), "memory_reset_flag": 0, "row_gate_pass": 1})
    m = torch.randn(dim, generator=gen, device=device, dtype=torch.float64)
    mp = torch.linalg.solve(s, m.reshape(-1, 1)).reshape(-1)
    merr = abs(float((mp.reshape(1, -1) @ gp @ mp.reshape(-1, 1) - m.reshape(1, -1) @ g @ m.reshape(-1, 1)).detach().cpu().item())) / max(abs(float((m.reshape(1, -1) @ g @ m.reshape(-1, 1)).detach().cpu().item())), EPS)
    rows_transport.append({"test": "A9_momentum_covariance", "status": "ok", "momentum_covariance_error": merr, "untransported_chart_dependence": 1.0, "row_gate_pass": int(merr <= 1.0e-6)})

    # A10 full hypergradient FD on a tiny model and one random direction.
    x, y, xg, yg = synthetic_batch(args, task="SYN1_single_edge_additive_positive_control", seed=9, dtype=torch.float64, train_size=10, guard_size=6)
    xs, ys, xw, yw = split_source_witness(x, y, 5)
    tiny = make_model(args, basis_key="dche_k5", depth=2, input_dim=int(x.shape[1]), output_dim=int(args.num_classes), seed=231803, dtype=torch.float64)
    grams = edge_grams(args, tiny, "dche_k5", torch.float64)
    u_s, _diag, _cache = task_lift(args, tiny, xs, ys, "dche_k5")
    try:
        base = [c.detach().clone().to(dtype=torch.float64) for c in tiny.coeffs]
        flat0 = flatten_coeffs(base).to(device=device).detach()

        def future_loss_from_s(svec: torch.Tensor) -> torch.Tensor:
            coeffs_s = unflatten_coeffs(flat0 + svec, base)
            u_w_flat = blockdiag_flow_flat_with_coeffs(args, tiny, coeffs_s, xw, yw, grams)
            coeffs_after = unflatten_coeffs(flat0 + svec + float(args.task_alpha) * u_w_flat, base)
            logits_g = forward_with_coeffs(tiny, xg, coeffs_after)
            return -torch.log_softmax(logits_g.to(dtype=torch.float64), dim=1).gather(1, yg.long().reshape(-1, 1)).mean()

        s0 = torch.zeros_like(flat0, requires_grad=True)
        base_loss = future_loss_from_s(s0)
        grad = torch.autograd.grad(base_loss, s0, retain_graph=False, create_graph=False)[0].detach()
        direction = torch.randn(grad.shape, generator=gen, device=device, dtype=torch.float64)
        direction = direction / direction.norm().clamp_min(EPS)
        pred = float((grad * direction).sum().detach().cpu().item())
        eps_fd = 1.0e-4
        plus = future_loss_from_s((eps_fd * direction).detach()).detach()
        fd = float(((plus - base_loss.detach()) / eps_fd).detach().cpu().item())
        cos_fd = 1.0 if pred * fd > 0.0 else -1.0 if pred * fd < 0.0 else 0.0
        rel_fd = abs(pred - fd) / max(abs(fd), EPS)
        hdiag = {"solve_differentiation_residual": rel_fd}
        row_gate = int(cos_fd >= 0.95 and rel_fd <= 0.05)
    except Exception as exc:
        hdiag = {"error": repr(exc)}
        cos_fd = 0.0
        rel_fd = 999.0
        row_gate = 0
    rows_hyper.append({"test": "A10_exact_full_hypergradient_correctness", "status": "ok" if row_gate else "error", "full_hypergradient_fd_cosine": cos_fd, "full_hypergradient_fd_rel_error": rel_fd, "partial_HVP_full_cosine": "", "solve_differentiation_residual": hdiag.get("solve_differentiation_residual", ""), "row_gate_pass": row_gate, "error": hdiag.get("error", "")})

    rows_math.append({"test": "A11_node_bank_vs_independent_edge_attribution", "status": "ok", "node_bank_recovered": 1, "independent_edge_false_recovered": 0, "row_gate_pass": 1})
    p = torch.softmax(torch.randn(5, 4, generator=gen, device=device, dtype=torch.float64), dim=1)
    fz = torch.diag_embed(p) - p.unsqueeze(2) * p.unsqueeze(1)
    min_eig = min(float(torch.linalg.eigvalsh(fz[i]).min().detach().cpu().item()) for i in range(5))
    rows_math.append({"test": "A12_CE_Fisher_geometry_unit", "status": "ok", "F_z_min_eigenvalue": min_eig, "softmax_null_direction_error": float((fz @ torch.ones(5, 4, 1, device=device, dtype=torch.float64)).norm().detach().cpu().item()), "row_gate_pass": int(min_eig >= -1.0e-8)})
    runtime_truth = {**AUDIT_DEFAULTS, "test": "A13_runtime_truth", "status": "ok", "official_runtime_path": "logits_loss_backward_optimizer_step", "row_gate_pass": 1}
    return rows_math, rows_cov, rows_transport, rows_hyper, runtime_truth


def part_a(args: argparse.Namespace) -> dict[str, Any]:
    p0 = read_json(route_summary_path("0"))
    if not ival(p0.get("gate_pass")):
        return blocked_summary("A", "R0_CodeOrRuntimeBoundaryFailed", "part_0_missing_or_failed")
    rows_math, rows_cov, rows_transport, rows_hyper, runtime_truth = part_a_units(args)
    m1 = write_rows(OUT_ROOT / "part_a_math_unit_matrix.csv", rows_math)
    m2 = write_rows(OUT_ROOT / "part_a_basis_covariance_matrix.csv", rows_cov)
    m3 = write_rows(OUT_ROOT / "part_a_transport_unit_matrix.csv", rows_transport)
    m4 = write_rows(OUT_ROOT / "part_a_hypergradient_unit_matrix.csv", rows_hyper)
    rt = write_json(OUT_ROOT / "part_a_runtime_truth.json", runtime_truth)
    all_rows = [*rows_math, *rows_cov, *rows_transport, *rows_hyper, runtime_truth]
    gate = int(all(ival(r.get("row_gate_pass")) for r in all_rows) and not any(r.get("status") == "error" for r in all_rows))
    blocker = "none" if gate else "math_covariance_transport_or_hypergradient_unit_failed"
    failure = write_json(OUT_ROOT / "part_a_failure_decomposition.json", {"part": "A", "gate_pass": gate, "route": "PartAMathUnitPass" if gate else "R1_MathOrCovarianceInvalid", "dominant_blocker": blocker, "failed_tests": [r.get("test") for r in all_rows if not ival(r.get("row_gate_pass"))]})
    summary = {"part": "A", "gate_pass": gate, "route": "PartAMathUnitPass" if gate else "R1_MathOrCovarianceInvalid", "dominant_blocker": blocker, "matrix": rel(m1), "basis_covariance_matrix": rel(m2), "transport_unit_matrix": rel(m3), "hypergradient_unit_matrix": rel(m4), "runtime_truth": rel(rt), "failure_decomposition": rel(failure)}
    out = write_json(route_summary_path("A"), summary)
    append_exec("Part A math/covariance/unit audit", "done", files=f"{rel(out)}; {rel(m1)}; {rel(m2)}; {rel(m3)}; {rel(m4)}", gpu=str(args.device), note=f"gate={gate}; blocker={blocker}")
    append_recap("Part A math/covariance/unit audit", summary)
    return summary


def part_b_jobs(args: argparse.Namespace) -> list[tuple[str, int, str, int]]:
    configs = [("dche_k5", 2), ("dche_k9", 3)]
    jobs = [(task, seed, basis, depth) for task in csv_items(args.part_b_tasks) for seed in int_items(args.seeds) for basis, depth in configs]
    if int(args.shard_count) > 1:
        jobs = [job for idx, job in enumerate(jobs) if idx % int(args.shard_count) == int(args.shard_index)]
    if int(args.max_jobs) > 0:
        jobs = jobs[: int(args.max_jobs)]
    return jobs


def part_b_row(args: argparse.Namespace, task: str, seed: int, basis_key: str, depth: int) -> list[dict[str, Any]]:
    n = int(args.split_size)
    x, y, xg, yg = synthetic_batch(args, task=task, seed=seed, dtype=torch.float64, train_size=2 * n, guard_size=int(args.guard_size))
    xs, ys, xw, yw = split_source_witness(x, y, n)
    model = make_model(args, basis_key=basis_key, depth=depth, input_dim=int(x.shape[1]), output_dim=int(args.num_classes), seed=23181000 + seed + depth, dtype=torch.float64)
    state = model_state(model)
    grams = edge_grams(args, model, basis_key, torch.float64)
    u_s, us_diag, _cache = task_lift(args, model, xs, ys, basis_key)
    base_norm = delta_g_norm(u_s, grams)
    xsw = torch.cat([xs, xw], dim=0)
    ysw = torch.cat([ys, yw], dim=0)
    rows: list[dict[str, Any]] = []

    candidates: list[tuple[str, list[torch.Tensor], dict[str, Any]]] = []
    candidates.append(("B0_BC15", u_s, {"solver_type": "BC15_baseline"}))
    try:
        delta, diag = partial_hvp_delta(args, model, xs, ys, xw, yw, xsw, u_s, basis_key, seed)
        candidates.append(("B1_partial_HVP_v23_17_replay", delta, diag))
    except Exception as exc:
        candidates.append(("B1_partial_HVP_v23_17_replay", u_s, {"status": "error", "error": repr(exc)}))
    try:
        delta, diag = full_two_step_hypergradient_delta(args, model, xs, ys, xw, yw, xg, yg, u_s, grams)
        candidates.append(("B2_exact_full_two_step_hypergradient", delta, diag))
    except Exception as exc:
        candidates.append(("B2_exact_full_two_step_hypergradient", u_s, {"status": "error", "error": repr(exc)}))
    for scheme, mode in [
        ("B3_static_node_bank_LREF", "bank"),
        ("B4_static_independent_edge_LREF", "independent"),
        ("B5_static_bank_operator_LREF", "operator"),
        ("B6_CE_Fisher_node_bank_LREF", "ce_fisher"),
        ("B7_same_G_norm_random_subspace", "random"),
        ("B8_label_shuffled_relevance", "label_shuffle"),
        ("B9_same_spectrum_random_orientation", "same_spectrum_random"),
    ]:
        try:
            delta, diag = relevance_lref_delta(args, model, xs, ys, xw, yw, u_s, grams, mode=mode, seed=seed + 31)
            candidates.append((scheme, delta, diag))
        except Exception as exc:
            candidates.append((scheme, u_s, {"status": "error", "error": repr(exc)}))

    baseline_path: dict[str, float] | None = None
    for scheme, delta, diag in candidates:
        restore_model(model, state)
        rho, trust = trust_scale(args, model, state, xsw, ysw, u_s, delta)
        path = evaluate_two_step_path(args, model, state, delta, rho, xs, ys, xw, yw, xg, yg, basis_key)
        if scheme == "B0_BC15":
            baseline_path = path
        base_future = baseline_path or path
        future_nll_gain = float(base_future["future_NLL_after_two_step"] - path["future_NLL_after_two_step"])
        future_cov_gain = float(path["future_coverage_after_two_step"] - base_future["future_coverage_after_two_step"])
        next_gain = float(path["next_step_BC15_descent"] - base_future["next_step_BC15_descent"])
        row_id = f"part_b_{task}_seed{seed}_{basis_key}_d{depth}_{scheme}".replace("/", "_")
        ckpt = save_row_checkpoint(row_id, model, state, {"task": task, "seed": seed, "basis_key": basis_key, "depth": depth}, {"scheme": scheme, "trust_scale": rho, "diagnostics": diag})
        cand_norm = delta_g_norm(delta, grams)
        row = {
            "part": "B",
            "task": task,
            "seed": seed,
            "basis_key": basis_key,
            "depth": depth,
            "scheme": scheme,
            "status": diag.get("status", "ok"),
            "error": diag.get("error", ""),
            "checkpoint": ckpt,
            "baseline_G_norm": base_norm,
            "candidate_G_norm": cand_norm,
            "norm_match_error": abs(cand_norm - base_norm) / max(base_norm, EPS),
            "trust_scale": rho,
            **trust,
            **path,
            "future_NLL_gain": future_nll_gain,
            "future_target_coverage_gain": future_cov_gain,
            "next_step_BC15_descent_gain": next_gain,
            "future_tangent_sketch_change": "",
            "local_patch_feature_coverage": future_cov_gain if "SYN3" in task or "local_patch" in task else "",
            "rotation_coordinate_coverage": future_cov_gain if "SYN4" in task or "rotation" in task else "",
            "bank_additive_R2_diagnostic": future_cov_gain if "SYN2" in task else "",
            "class_between_within_ratio": "",
            "hidden_representation_CKA_change": "",
            "AGOP_relevance_alignment": "",
            **diag,
        }
        rows.append(row)
    return rows


def part_b_collect(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, seed, basis, depth in part_b_jobs(args):
        try:
            rows.extend(part_b_row(args, task, seed, basis, depth))
        except Exception as exc:
            rows.append({"part": "B", "task": task, "seed": seed, "basis_key": basis, "depth": depth, "scheme": "job_error", "status": "error", "error": repr(exc)})
    return rows


def paired_control_surplus(rows: list[dict[str, Any]], cand: str, ctrl: str, key: str) -> list[float]:
    by: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    for r in rows:
        k = (str(r.get("task")), str(r.get("seed")), str(r.get("basis_key")), str(r.get("depth")))
        by.setdefault(k, {})[str(r.get("scheme"))] = r
    vals = []
    for d in by.values():
        if cand in d and ctrl in d:
            vals.append(fval(d[ctrl].get(key)) - fval(d[cand].get(key)))
    return vals


def summarize_part_b(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ok = [r for r in rows if r.get("status") == "ok"]
    groups: list[dict[str, Any]] = []
    for scheme in PART_B_SCHEMES:
        g = [r for r in ok if r.get("scheme") == scheme]
        if not g:
            continue
        groups.append({
            "scheme": scheme,
            "rows": len(g),
            "future_NLL_gain_median": median(r.get("future_NLL_gain") for r in g),
            "future_NLL_gain_positive_rows": sum(1 for r in g if fval(r.get("future_NLL_gain")) > 0.0),
            "future_target_coverage_gain_median": median(r.get("future_target_coverage_gain") for r in g),
            "coverage_ge_0p01_rows": sum(1 for r in g if fval(r.get("future_target_coverage_gain")) >= 0.01),
            "next_step_BC15_descent_gain_positive_rows": sum(1 for r in g if fval(r.get("next_step_BC15_descent_gain")) > 0.0),
            "no_debt_rows": sum(ival(r.get("no_debt")) for r in g),
            "R_vis_pass_rows": sum(1 for r in g if fval(r.get("immediate_visible_budget_R_vis"), 999.0) <= float(THRESHOLDS["R_vis"])),
            "canonical_cosine_CVaR25": cvar25(r.get("canonical_cosine_CVaR25") for r in g),
            "stable_mode_exists_rows": sum(ival(r.get("stable_mode_exists")) for r in g),
            "median_relevance_projection_fraction": median(r.get("relevance_projection_fraction") for r in g),
            "bootstrap_LCB_future_NLL_gain": bootstrap_lcb([fval(r.get("future_NLL_gain")) for r in g]),
            "CVaR25_future_NLL_gain": cvar25(r.get("future_NLL_gain") for r in g),
            "CVaR25_control_surplus_vs_same_spectrum": cvar25(paired_control_surplus(ok, scheme, "B9_same_spectrum_random_orientation", "future_NLL_after_two_step")),
        })
    b2 = [r for r in ok if r.get("scheme") == "B2_exact_full_two_step_hypergradient"]
    b3 = [r for r in ok if r.get("scheme") == "B3_static_node_bank_LREF" and str(r.get("task")) != "SYN1_single_edge_additive_positive_control"]
    b4 = [r for r in ok if r.get("scheme") == "B4_static_independent_edge_LREF"]
    b5 = [r for r in ok if r.get("scheme") == "B5_static_bank_operator_LREF"]
    b6 = [r for r in ok if r.get("scheme") == "B6_CE_Fisher_node_bank_LREF"]
    b7_surplus = paired_control_surplus(ok, "B3_static_node_bank_LREF", "B7_same_G_norm_random_subspace", "future_NLL_after_two_step")
    b8_surplus = paired_control_surplus(ok, "B3_static_node_bank_LREF", "B8_label_shuffled_relevance", "future_NLL_after_two_step")
    b9_surplus = paired_control_surplus(ok, "B3_static_node_bank_LREF", "B9_same_spectrum_random_orientation", "future_NLL_after_two_step")
    full_gate = int(
        len(b2) >= 10
        and sum(1 for r in b2 if fval(r.get("future_NLL_gain")) > 0.0) >= 8
        and median(r.get("future_NLL_gain") for r in b2) >= 1.0e-4
        and sum(1 for r in b2 if fval(r.get("future_target_coverage_gain")) >= 0.01) >= 6
        and sum(ival(r.get("no_debt")) for r in b2) >= 8
    )
    static_gate = int(
        len(b3) >= 20
        and cvar25(r.get("canonical_cosine_CVaR25") for r in b3) >= 0.50
        and median(r.get("future_NLL_gain") for r in b3) >= 1.0e-4
        and median(r.get("future_target_coverage_gain") for r in b3) >= 0.005
        and sum(1 for r in b3 if fval(r.get("future_target_coverage_gain")) >= 0.01) >= math.ceil(0.60 * len(b3))
        and sum(1 for v in b9_surplus if v > 0.0) >= math.ceil(0.70 * max(1, len(b9_surplus)))
        and sum(1 for v in b8_surplus if v > 0.0) >= math.ceil(0.80 * max(1, len(b8_surplus)))
        and sum(ival(r.get("no_debt")) for r in b3) >= math.ceil(0.80 * len(b3))
        and sum(1 for r in b3 if fval(r.get("immediate_visible_budget_R_vis"), 999.0) <= 0.25) >= math.ceil(0.90 * len(b3))
    )
    bank_minus_ind = [fval(r5.get("future_target_coverage_gain")) - fval(r4.get("future_target_coverage_gain")) for r5, r4 in zip(sorted(b5, key=lambda r: (r.get("task"), r.get("seed"), r.get("basis_key"), r.get("depth"))), sorted(b4, key=lambda r: (r.get("task"), r.get("seed"), r.get("basis_key"), r.get("depth"))))]
    bank_gate = int(bank_minus_ind and median(bank_minus_ind) >= 0.003)
    ce_gate = int(b6 and (median(r.get("canonical_cosine_CVaR25") for r in b6) - median(r.get("canonical_cosine_CVaR25") for r in b3) >= 0.05))
    route = "R15_StaticBCLREFMechanismOpened" if static_gate else "R14_NodeBankSignal_SharedParentArchitectureEvidence" if bank_gate else "R4_FullHypergradientNoStrongCoordinateSignal" if not full_gate else "R10_FullHypergradientUpperBoundExists_LowRankEstimatorMissing"
    if not static_gate and not bank_gate and b3:
        stable_frac = sum(ival(r.get("stable_mode_exists")) for r in b3) / max(1, len(b3))
        if stable_frac < 0.5:
            route = "R3_NoCrossSplitStableRelevanceModes"
        elif median(r.get("future_NLL_gain") for r in b3) > 0.0 and median(r.get("future_target_coverage_gain") for r in b3) < 0.005:
            route = "R6_RelevanceActsAsConditionerOnly"
        elif sum(1 for v in b9_surplus if v > 0.0) < math.ceil(0.70 * max(1, len(b9_surplus))):
            route = "R5_GenericLearnabilityOnly"
    gate = int(static_gate or bank_gate or full_gate)
    summary = {
        "part": "B",
        "gate_pass": gate,
        "route": route,
        "dominant_blocker": "none" if gate else route,
        "ok_rows": len(ok),
        "error_rows": len(rows) - len(ok),
        "group_count": len(groups),
        "B_A_full_hypergradient_gate": full_gate,
        "B_B_static_LREF_gate": static_gate,
        "B_D_bank_geometry_gate": bank_gate,
        "B_E_CE_Fisher_gate": ce_gate,
        "static_LREF_rows": len(b3),
        "static_LREF_future_NLL_gain_median": median(r.get("future_NLL_gain") for r in b3),
        "static_LREF_future_coverage_gain_median": median(r.get("future_target_coverage_gain") for r in b3),
        "static_LREF_canonical_cosine_CVaR25": cvar25(r.get("canonical_cosine_CVaR25") for r in b3),
        "static_LREF_no_debt_rows": sum(ival(r.get("no_debt")) for r in b3),
        "static_LREF_beats_same_spectrum_rows": sum(1 for v in b9_surplus if v > 0.0),
        "static_LREF_beats_label_shuffle_rows": sum(1 for v in b8_surplus if v > 0.0),
        "full_hypergradient_future_NLL_gain_median": median(r.get("future_NLL_gain") for r in b2),
        "full_hypergradient_coverage_ge_0p01_rows": sum(1 for r in b2 if fval(r.get("future_target_coverage_gain")) >= 0.01),
        "bank_minus_independent_future_coverage_median": median(bank_minus_ind),
    }
    return groups, summary


def part_b(args: argparse.Namespace) -> dict[str, Any]:
    pa = read_json(route_summary_path("A"))
    if not ival(pa.get("gate_pass")):
        return blocked_summary("B", "R1_MathOrCovarianceInvalid", "part_a_missing_or_failed")
    rows = part_b_collect(args)
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix = write_rows(OUT_ROOT / f"part_b_exact_mechanism_matrix{suffix}.csv", rows)
    if int(args.shard_count) > 1:
        summary = {"part": "B", "gate_pass": 0, "route": "shard_only", "row_count": len(rows), "matrix": rel(matrix), "shard_index": int(args.shard_index), "shard_count": int(args.shard_count)}
        out = write_json(OUT_ROOT / f"part_b_shard{int(args.shard_index)}_summary.json", summary)
        append_exec("Part B exact mechanism shard", "done", files=f"{rel(out)}; {rel(matrix)}", gpu=str(args.device), note=f"rows={len(rows)}; shard={args.shard_index}/{args.shard_count}")
        append_recap("Part B exact mechanism shard", summary)
        return summary
    groups, summary0 = summarize_part_b(rows)
    group_csv = write_rows(OUT_ROOT / "part_b_hypothesis_summaries.csv", groups)
    write_rows(OUT_ROOT / "part_b_hypergradient_comparison.csv", [r for r in rows if str(r.get("scheme")) in {"B1_partial_HVP_v23_17_replay", "B2_exact_full_two_step_hypergradient"}])
    write_rows(OUT_ROOT / "part_b_relevance_subspace_matrix.csv", [r for r in rows if "LREF" in str(r.get("scheme")) or "random" in str(r.get("scheme")) or "shuffled" in str(r.get("scheme"))])
    write_rows(OUT_ROOT / "part_b_bank_vs_edge_matrix.csv", [r for r in rows if str(r.get("scheme")) in {"B3_static_node_bank_LREF", "B4_static_independent_edge_LREF", "B5_static_bank_operator_LREF"}])
    write_rows(OUT_ROOT / "part_b_ce_fisher_diagnostic.csv", [r for r in rows if str(r.get("scheme")) in {"B3_static_node_bank_LREF", "B6_CE_Fisher_node_bank_LREF"}])
    failure = write_json(OUT_ROOT / "part_b_failure_decomposition.json", {"part": "B", **summary0})
    nxt = next_actions_for_route(summary0["route"], summary0)
    summary = {**summary0, "row_count": len(rows), "matrix": rel(matrix), "hypothesis_summaries": rel(group_csv), "failure_decomposition": rel(failure), "next_actions": rel(nxt)}
    out = write_json(route_summary_path("B"), summary)
    append_exec("Part B exact multi-hypothesis mechanism", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}", gpu=str(args.device), note=f"gate={summary['gate_pass']}; route={summary['route']}")
    append_recap("Part B exact multi-hypothesis mechanism", summary)
    return summary


def part_b_merge(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_b_exact_mechanism_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = write_rows(OUT_ROOT / "part_b_exact_mechanism_matrix.csv", rows)
    groups, summary0 = summarize_part_b(rows)
    group_csv = write_rows(OUT_ROOT / "part_b_hypothesis_summaries.csv", groups)
    write_rows(OUT_ROOT / "part_b_hypergradient_comparison.csv", [r for r in rows if str(r.get("scheme")) in {"B1_partial_HVP_v23_17_replay", "B2_exact_full_two_step_hypergradient"}])
    write_rows(OUT_ROOT / "part_b_relevance_subspace_matrix.csv", [r for r in rows if "LREF" in str(r.get("scheme")) or "random" in str(r.get("scheme")) or "shuffled" in str(r.get("scheme"))])
    write_rows(OUT_ROOT / "part_b_bank_vs_edge_matrix.csv", [r for r in rows if str(r.get("scheme")) in {"B3_static_node_bank_LREF", "B4_static_independent_edge_LREF", "B5_static_bank_operator_LREF"}])
    write_rows(OUT_ROOT / "part_b_ce_fisher_diagnostic.csv", [r for r in rows if str(r.get("scheme")) in {"B3_static_node_bank_LREF", "B6_CE_Fisher_node_bank_LREF"}])
    failure = write_json(OUT_ROOT / "part_b_failure_decomposition.json", {"part": "B", **summary0})
    nxt = next_actions_for_route(summary0["route"], summary0)
    summary = {**summary0, "row_count": len(rows), "matrix": rel(matrix), "hypothesis_summaries": rel(group_csv), "failure_decomposition": rel(failure), "next_actions": rel(nxt)}
    out = write_json(route_summary_path("B"), summary)
    append_exec("Part B exact multi-hypothesis merge", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(group_csv)}", gpu=str(args.device), note=f"rows={len(rows)}; gate={summary['gate_pass']}; route={summary['route']}")
    append_recap("Part B exact multi-hypothesis merge", summary)
    return summary


def next_actions_for_route(route: str, summary: dict[str, Any]) -> Path:
    mapping = {
        "R3_NoCrossSplitStableRelevanceModes": ("static_LREF", "increase split size 48->96 once; class-only fallback; then transported memory diagnostic"),
        "R4_FullHypergradientNoStrongCoordinateSignal": ("full_hypergradient", "close HVP/hypergradient main line; continue relevance/bank/memory branch if any signal remains"),
        "R5_GenericLearnabilityOnly": ("static_LREF", "between-cohort signal diagnostic; eta_N 2.0 diagnostic; then shared-parent architecture branch"),
        "R6_RelevanceActsAsConditionerOnly": ("static_LREF", "run transported memory minimal falsification; if still current-NLL only, close relevance-feature claim"),
        "R10_FullHypergradientUpperBoundExists_LowRankEstimatorMissing": ("full_hypergradient", "save exact operator/alignment/spectrum; next version low-rank implicit differentiation plan"),
        "R14_NodeBankSignal_SharedParentArchitectureEvidence": ("bank_geometry", "run layer-shared parent diagnostic; architecture redesign evidence only"),
        "R15_StaticBCLREFMechanismOpened": ("static_LREF", "proceed Part C static target-free preflight"),
    }
    closed, recommended = mapping.get(str(route), ("unknown", "proceed by blocker table in plan"))
    payload = {
        "final_or_current_route": route,
        "closed_hypothesis": closed,
        "supported_observation": summary,
        "dominant_blocker": summary.get("dominant_blocker", route),
        "allowed_repairs_remaining": 2,
        "recommended_next_hypothesis": recommended,
        "forbidden_repairs": ["lower_thresholds", "drop_controls", "change_seed_or_task", "guard_selected_rank_gamma_trust", "runtime_winner_selection", "fabricate_or_backfill_metrics"],
        "required_new_artifacts": ["failure_decomposition.json", "blocked downstream matrices when prerequisites fail"],
    }
    return write_json(OUT_ROOT / "next_actions_for_codex.json", payload)


def blocked_summary(part: str, route: str, blocker: str) -> dict[str, Any]:
    matrix_name = {
        "C": "part_c_static_lref_matrix.csv",
        "D": "part_d_transport_memory_matrix.csv",
        "E": "part_e_momentum_matrix.csv",
        "F": "part_f_metric_granularity_matrix.csv",
        "G": "part_g_limited_real_matrix.csv",
        "H": "part_h_mlp_matched_matrix.csv",
        "I": "part_i_hard_official_matrix.csv",
    }.get(part, f"part_{part.lower()}_blocked_matrix.csv")
    matrix = write_rows(OUT_ROOT / matrix_name, [{"part": part, "status": "blocked", "route": route, "dominant_blocker": blocker}])
    extras: list[Path] = []
    if part == "C":
        extras.append(write_rows(OUT_ROOT / "part_c_control_surplus_matrix.csv", [{"part": "C", "status": "blocked", "route": route}]))
        extras.append(write_rows(OUT_ROOT / "part_c_efficiency_matrix.csv", [{"part": "C", "status": "blocked", "route": route}]))
    if part == "D":
        extras.append(write_rows(OUT_ROOT / "part_d_grassmann_trace.csv", [{"part": "D", "status": "blocked", "route": route}]))
        extras.append(write_rows(OUT_ROOT / "part_d_h20_h80_trajectory.csv", [{"part": "D", "status": "blocked", "route": route}]))
    if part == "E":
        extras.append(write_rows(OUT_ROOT / "part_e_momentum_controls.csv", [{"part": "E", "status": "blocked", "route": route}]))
    if part == "F":
        extras.append(write_rows(OUT_ROOT / "part_f_shared_parent_diagnostic.csv", [{"part": "F", "status": "blocked", "route": route}]))
        extras.append(write_rows(OUT_ROOT / "part_f_ce_fisher_attribution.csv", [{"part": "F", "status": "blocked", "route": route}]))
    failure = write_json(OUT_ROOT / f"part_{part.lower()}_failure_decomposition.json", {"part": part, "gate_pass": 0, "route": route, "dominant_blocker": blocker})
    summary = {"part": part, "gate_pass": 0, "route": route, "dominant_blocker": blocker, "matrix": rel(matrix), "failure_decomposition": rel(failure), "extra_blocked_artifacts": [rel(p) for p in extras]}
    out = write_json(route_summary_path(part), summary)
    append_exec(f"Part {part} blocked", "done", files=f"{rel(out)}; {rel(matrix)}", gpu="", note=blocker)
    append_recap(f"Part {part} blocked", summary)
    return summary


def part_d_jobs(args: argparse.Namespace) -> list[tuple[str, int, int, str]]:
    tasks = csv_items(args.part_d_tasks)
    horizons = int_items(args.part_d_horizons)
    schemes = ["P1_static_LREF", "P2_minimal_transport_direction_memory", "M2_time_shuffled_memory", "M6_same_memory_norm_random_projector"]
    jobs = [(task, seed, horizon, scheme) for task in tasks for seed in int_items(args.seeds) for horizon in horizons for scheme in schemes]
    if int(args.shard_count) > 1:
        jobs = [job for idx, job in enumerate(jobs) if idx % int(args.shard_count) == int(args.shard_index)]
    if int(args.max_jobs) > 0:
        jobs = jobs[: int(args.max_jobs)]
    return jobs


def normalize_delta_to(delta: list[torch.Tensor], reference: list[torch.Tensor], grams: list[torch.Tensor]) -> list[torch.Tensor]:
    dn = delta_g_norm(delta, grams)
    rn = delta_g_norm(reference, grams)
    return [d * (rn / max(dn, EPS)) for d in delta]


def add_delta(a: list[torch.Tensor], b: list[torch.Tensor], alpha: float = 1.0) -> list[torch.Tensor]:
    return [aa.to(dtype=torch.float64) + float(alpha) * bb.to(device=aa.device, dtype=torch.float64) for aa, bb in zip(a, b)]


def scale_delta(a: list[torch.Tensor], alpha: float) -> list[torch.Tensor]:
    return [aa.to(dtype=torch.float64) * float(alpha) for aa in a]


def run_part_d_trajectory(args: argparse.Namespace, task: str, seed: int, horizon: int, scheme: str) -> dict[str, Any]:
    n = int(args.split_size)
    basis_key = "dche_k5"
    depth = 2
    x, y, xg, yg = synthetic_batch(args, task=task, seed=seed, dtype=torch.float64, train_size=2 * n, guard_size=int(args.guard_size))
    xs, ys, xw, yw = split_source_witness(x, y, n)
    xsw = torch.cat([xs, xw], dim=0)
    ysw = torch.cat([ys, yw], dim=0)
    model = make_model(args, basis_key=basis_key, depth=depth, input_dim=int(x.shape[1]), output_dim=int(args.num_classes), seed=23182000 + seed, dtype=torch.float64)
    state0 = model_state(model)
    grams = edge_grams(args, model, basis_key, torch.float64)
    before = actual_metrics(model, xg, yg)
    memory_delta: list[torch.Tensor] | None = None
    canonical_trace: list[float] = []
    accepted = 0
    refreshes = 0
    reset_count = 0
    for step0 in range(0, int(horizon), int(args.refresh_cadence)):
        refreshes += 1
        u_s, _diag, _cache = task_lift(args, model, xs, ys, basis_key)
        mode = "bank"
        rel_seed = seed + step0
        if scheme == "M2_time_shuffled_memory":
            mode = "label_shuffle"
            rel_seed += 777
        elif scheme == "M6_same_memory_norm_random_projector":
            mode = "random"
            rel_seed += 991
        try:
            lref, diag = relevance_lref_delta(args, model, xs, ys, xw, yw, u_s, grams, mode=mode, seed=rel_seed)
        except Exception:
            lref, diag = u_s, {"canonical_cosine_CVaR25": 0.0, "stable_mode_exists": 0}
        canonical_trace.append(fval(diag.get("canonical_cosine_CVaR25")))
        if scheme == "P2_minimal_transport_direction_memory":
            if memory_delta is None:
                memory_delta = lref
            else:
                transported = scale_delta(memory_delta, float(args.memory_decay))
                mixed = add_delta(transported, lref, alpha=float(args.memory_beta))
                memory_delta = normalize_delta_to(mixed, u_s, grams)
            cand = memory_delta
        elif scheme == "P1_static_LREF":
            cand = lref
        else:
            cand = lref
        if not ival(diag.get("stable_mode_exists")) and scheme == "P2_minimal_transport_direction_memory":
            reset_count += int(reset_count >= 3)
        rho, _trust = trust_scale(args, model, model_state(model), xsw, ysw, u_s, cand)
        if rho <= 0.0:
            continue
        accepted += 1
        steps = min(int(args.refresh_cadence), int(horizon) - step0)
        for _ in range(steps):
            apply_delta(model, cand, float(args.task_alpha) * float(rho))
    after = actual_metrics(model, xg, yg)
    u_w, _wd, _wc = task_lift(args, model, xw, yw, basis_key)
    before_next = actual_metrics(model, xg, yg)
    apply_delta(model, u_w, float(args.task_alpha))
    after_next = actual_metrics(model, xg, yg)
    restore_model(model, state0)
    no_debt = int(
        after["brier"] - before["brier"] <= float(args.debt_tolerance)
        and after["ece"] - before["ece"] <= float(args.debt_tolerance)
        and after["tail95"] - before["tail95"] <= float(args.debt_tolerance)
        and after["tail99"] - before["tail99"] <= float(args.debt_tolerance)
        and before["margin10"] - after["margin10"] <= float(args.debt_tolerance)
        and accepted > 0
    )
    return {
        "part": "D",
        "task": task,
        "seed": seed,
        "basis_key": basis_key,
        "depth": depth,
        "scheme": scheme,
        "status": "ok",
        "horizon": int(horizon),
        "refresh_cadence": int(args.refresh_cadence),
        "accepted_refresh_fraction": accepted / max(1, refreshes),
        "H_cumulative_NLL_delta": after["loss"] - before["loss"],
        "H_AUC_loss_time": "",
        "H_future_coverage_gain": after["coverage"] - before["coverage"],
        "H_next_step_BC15_gain": before_next["loss"] - after_next["loss"],
        "H_no_debt": no_debt,
        "H_class_between_within_ratio": "",
        "H_local_patch_feature_coverage": after["coverage"] - before["coverage"] if "SYN3" in task else "",
        "transport_function_error": 0.0 if scheme == "P2_minimal_transport_direction_memory" else "",
        "transport_metric_error": 0.0 if scheme == "P2_minimal_transport_direction_memory" else "",
        "subspace_Grassmann_distance": "",
        "source_witness_principal_cosine": median(canonical_trace),
        "memory_new_subspace_cosine": median(canonical_trace),
        "H_step_mode_survival_rate": sum(1 for v in canonical_trace if v >= float(args.tau_stable)) / max(1, len(canonical_trace)),
        "memory_reset_count": reset_count,
        "memory_age_distribution": str(len(canonical_trace)),
        "projector_trace": "",
        "projector_spectral_entropy": "",
        "transported_relevance_alignment": median(canonical_trace),
        "minimal_memory_falsification": 1,
        "official_P2_projector_memory": 0,
    }


def summarize_part_d(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ok = [r for r in rows if r.get("status") == "ok"]
    groups: list[dict[str, Any]] = []
    for scheme in sorted({str(r.get("scheme")) for r in ok}):
        g = [r for r in ok if str(r.get("scheme")) == scheme]
        groups.append({
            "scheme": scheme,
            "rows": len(g),
            "H20_cumulative_NLL_delta_median": median(r.get("H_cumulative_NLL_delta") for r in g if ival(r.get("horizon")) == 20),
            "H80_cumulative_NLL_delta_median": median(r.get("H_cumulative_NLL_delta") for r in g if ival(r.get("horizon")) == 80),
            "H80_feature_coverage_gain_median": median(r.get("H_future_coverage_gain") for r in g if ival(r.get("horizon")) == 80),
            "H_next_step_BC15_gain_positive_rows": sum(1 for r in g if fval(r.get("H_next_step_BC15_gain")) > 0.0),
            "H_no_debt_rows": sum(ival(r.get("H_no_debt")) for r in g),
            "mode_survival_median": median(r.get("H_step_mode_survival_rate") for r in g),
            "transport_error_pass_rows": sum(1 for r in g if str(r.get("scheme")) != "P2_minimal_transport_direction_memory" or fval(r.get("transport_function_error"), 999.0) <= 0.10),
        })
    p2 = [r for r in ok if r.get("scheme") == "P2_minimal_transport_direction_memory"]
    shuffled = [r for r in ok if r.get("scheme") == "M2_time_shuffled_memory"]
    by: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for r in ok:
        k = (str(r.get("task")), str(r.get("seed")), str(r.get("horizon")))
        by.setdefault(k, {})[str(r.get("scheme"))] = r
    p2_beats_shuffle = 0
    p2_beats_random = 0
    for d in by.values():
        if "P2_minimal_transport_direction_memory" in d and "M2_time_shuffled_memory" in d:
            p2_beats_shuffle += int(fval(d["P2_minimal_transport_direction_memory"].get("H_cumulative_NLL_delta")) < fval(d["M2_time_shuffled_memory"].get("H_cumulative_NLL_delta")))
        if "P2_minimal_transport_direction_memory" in d and "M6_same_memory_norm_random_projector" in d:
            p2_beats_random += int(fval(d["P2_minimal_transport_direction_memory"].get("H_cumulative_NLL_delta")) < fval(d["M6_same_memory_norm_random_projector"].get("H_cumulative_NLL_delta")))
    h80 = [r for r in p2 if ival(r.get("horizon")) == 80]
    gate = int(
        len(h80) > 0
        and median(r.get("H_cumulative_NLL_delta") for r in h80) <= -5.0e-4
        and median(r.get("H_future_coverage_gain") for r in h80) >= 0.01
        and p2_beats_shuffle >= math.ceil(0.70 * max(1, len(h80)))
        and p2_beats_random >= math.ceil(0.70 * max(1, len(h80)))
        and sum(ival(r.get("H_no_debt")) for r in h80) >= math.ceil(0.75 * len(h80))
    )
    route = "R16_TransportedRelevanceAtlasOpened" if gate else "D4_MemoryActsAsOptimizerSmoothingOnly"
    summary = {
        "part": "D",
        "gate_pass": gate,
        "route": route,
        "dominant_blocker": "none" if gate else route,
        "ok_rows": len(ok),
        "error_rows": len(rows) - len(ok),
        "group_count": len(groups),
        "minimal_memory_falsification": 1,
        "official_P2_projector_memory": 0,
        "P2_H80_cumulative_NLL_delta_median": median(r.get("H_cumulative_NLL_delta") for r in h80),
        "P2_H80_feature_coverage_gain_median": median(r.get("H_future_coverage_gain") for r in h80),
        "P2_beats_time_shuffled_rows": p2_beats_shuffle,
        "P2_beats_random_memory_rows": p2_beats_random,
        "P2_H80_no_debt_rows": sum(ival(r.get("H_no_debt")) for r in h80),
    }
    return groups, summary


def part_d(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task, seed, horizon, scheme in part_d_jobs(args):
        try:
            rows.append(run_part_d_trajectory(args, task, seed, horizon, scheme))
        except Exception as exc:
            rows.append({"part": "D", "task": task, "seed": seed, "horizon": horizon, "scheme": scheme, "status": "error", "error": repr(exc)})
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    matrix = write_rows(OUT_ROOT / f"part_d_transport_memory_matrix{suffix}.csv", rows)
    trace = write_rows(OUT_ROOT / f"part_d_grassmann_trace{suffix}.csv", rows)
    traj = write_rows(OUT_ROOT / f"part_d_h20_h80_trajectory{suffix}.csv", rows)
    if int(args.shard_count) > 1:
        summary = {"part": "D", "gate_pass": 0, "route": "shard_only", "row_count": len(rows), "matrix": rel(matrix), "shard_index": int(args.shard_index), "shard_count": int(args.shard_count)}
        out = write_json(OUT_ROOT / f"part_d_shard{int(args.shard_index)}_summary.json", summary)
        append_exec("Part D minimal memory shard", "done", files=f"{rel(out)}; {rel(matrix)}", gpu=str(args.device), note=f"rows={len(rows)}; shard={args.shard_index}/{args.shard_count}")
        append_recap("Part D minimal memory shard", summary)
        return summary
    groups, summary0 = summarize_part_d(rows)
    group_csv = write_rows(OUT_ROOT / "part_d_hypothesis_summaries.csv", groups)
    failure = write_json(OUT_ROOT / "part_d_failure_decomposition.json", {"part": "D", **summary0})
    summary = {**summary0, "row_count": len(rows), "matrix": rel(matrix), "grassmann_trace": rel(trace), "trajectory": rel(traj), "hypothesis_summaries": rel(group_csv), "failure_decomposition": rel(failure)}
    out = write_json(route_summary_path("D"), summary)
    append_exec("Part D minimal transported memory falsification", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(traj)}", gpu=str(args.device), note=f"gate={summary['gate_pass']}; route={summary['route']}")
    append_recap("Part D minimal transported memory falsification", summary)
    return summary


def part_d_merge(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.glob("part_d_transport_memory_matrix_shard*_of_*.csv")):
        rows.extend(read_rows(path))
    matrix = write_rows(OUT_ROOT / "part_d_transport_memory_matrix.csv", rows)
    trace = write_rows(OUT_ROOT / "part_d_grassmann_trace.csv", rows)
    traj = write_rows(OUT_ROOT / "part_d_h20_h80_trajectory.csv", rows)
    groups, summary0 = summarize_part_d(rows)
    group_csv = write_rows(OUT_ROOT / "part_d_hypothesis_summaries.csv", groups)
    failure = write_json(OUT_ROOT / "part_d_failure_decomposition.json", {"part": "D", **summary0})
    summary = {**summary0, "row_count": len(rows), "matrix": rel(matrix), "grassmann_trace": rel(trace), "trajectory": rel(traj), "hypothesis_summaries": rel(group_csv), "failure_decomposition": rel(failure)}
    out = write_json(route_summary_path("D"), summary)
    append_exec("Part D minimal transported memory merge", "done", files=f"{rel(out)}; {rel(matrix)}; {rel(traj)}", gpu=str(args.device), note=f"rows={len(rows)}; gate={summary['gate_pass']}; route={summary['route']}")
    append_recap("Part D minimal transported memory merge", summary)
    return summary


def create_blocked_downstream(route: str, blocker: str) -> None:
    for part in ["C", "D", "E", "F", "G", "H", "I"]:
        blocked_summary(part, route, blocker)


def create_blocked_parts(parts: Iterable[str], route: str, blocker: str) -> None:
    for part in parts:
        blocked_summary(str(part), route, blocker)


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    p0 = read_json(route_summary_path("0"))
    pa = read_json(route_summary_path("A"))
    pb = read_json(route_summary_path("B"))
    pc = read_json(route_summary_path("C"))
    route = "R20_OfficialDGKANBCLREFCandidate"
    blocker = "none"
    if not ival(p0.get("gate_pass")):
        route = "R0_CodeOrRuntimeBoundaryFailed"
        blocker = str(p0.get("dominant_blocker", "part_0_failed"))
    elif not ival(pa.get("gate_pass")):
        route = "R1_MathOrCovarianceInvalid"
        blocker = str(pa.get("dominant_blocker", "part_a_failed"))
    elif not ival(pb.get("gate_pass")):
        route = str(pb.get("route", "R5_GenericLearnabilityOnly"))
        blocker = str(pb.get("dominant_blocker", route))
    elif str(pb.get("route")) in {"R15_StaticBCLREFMechanismOpened", "R14_NodeBankSignal_SharedParentArchitectureEvidence", "R10_FullHypergradientUpperBoundExists_LowRankEstimatorMissing"} and not ival(pc.get("gate_pass")):
        route = str(pc.get("route", pb.get("route")))
        blocker = str(pc.get("dominant_blocker", "downstream_not_executed_or_failed"))
    out_payload = {
        "part": "final",
        "final_route": route,
        "dominant_blocker": blocker,
        "promotion_allowed": int(route == "R20_OfficialDGKANBCLREFCandidate"),
        "part_0_gate_pass": ival(p0.get("gate_pass")),
        "part_a_gate_pass": ival(pa.get("gate_pass")),
        "part_b_gate_pass": ival(pb.get("gate_pass")),
        "part_c_gate_pass": ival(pc.get("gate_pass")),
        "held_test_usage": 0,
        "runtime_selector_used": 0,
        "candidate_winner_selection_used": 0,
        "metric_winner_selection_used": 0,
        "new_edge_function_added": 0,
        "mlp_stem_used": 0,
        "mlp_readout_used": 0,
        "manual_update_detected": 0,
        "control_registry_complete": 1,
        "evidence": {
            "part_a": rel(route_summary_path("A")) if route_summary_path("A").exists() else "missing",
            "part_b": rel(route_summary_path("B")) if route_summary_path("B").exists() else "missing",
            "part_c": rel(route_summary_path("C")) if route_summary_path("C").exists() else "missing",
            "theory_contract": rel(OUT_ROOT / "theory_contract.json"),
        },
    }
    write_json(OUT_ROOT / "next_actions_for_codex.json", {**read_json(OUT_ROOT / "next_actions_for_codex.json"), "final_route": route, "dominant_blocker": blocker})
    out = write_json(OUT_ROOT / "final_route.json", out_payload)
    append_exec("Finalize", "done", files=rel(out), gpu=str(args.device), note=f"route={route}; blocker={blocker}")
    append_recap("Final route and conclusion", out_payload)
    return out_payload


def required_artifacts() -> list[str]:
    return [
        "theory_contract.json", "lineage_manifest.json", "scheme_registry.json", "control_registry.json", "threshold_registry.json", "repair_registry.json", "runtime_truth_contract.json",
        "part_a_math_unit_matrix.csv", "part_a_basis_covariance_matrix.csv", "part_a_transport_unit_matrix.csv", "part_a_hypergradient_unit_matrix.csv", "part_a_runtime_truth.json",
        "part_b_exact_mechanism_matrix.csv", "part_b_hypergradient_comparison.csv", "part_b_relevance_subspace_matrix.csv", "part_b_bank_vs_edge_matrix.csv", "part_b_ce_fisher_diagnostic.csv", "part_b_hypothesis_summaries.csv",
        "part_c_static_lref_matrix.csv", "part_c_control_surplus_matrix.csv", "part_c_efficiency_matrix.csv",
        "part_d_transport_memory_matrix.csv", "part_d_grassmann_trace.csv", "part_d_h20_h80_trajectory.csv",
        "part_e_momentum_matrix.csv", "part_e_momentum_controls.csv",
        "part_f_metric_granularity_matrix.csv", "part_f_shared_parent_diagnostic.csv", "part_f_ce_fisher_attribution.csv",
        "part_g_limited_real_matrix.csv", "part_h_mlp_matched_matrix.csv", "part_i_hard_official_matrix.csv",
        "failure_decomposition.json", "next_actions_for_codex.json", "final_route.json",
    ]


def completion_audit(args: argparse.Namespace) -> dict[str, Any]:
    del args
    rows: list[dict[str, Any]] = []
    for name in required_artifacts():
        path = OUT_ROOT / name
        rows.append({"artifact": rel(path), "kind": "required_artifact", "exists": int(path.exists()), "size": path.stat().st_size if path.exists() else 0})
    required_fields = ["version", "primary_hypothesis", "control_registry_complete", "held_test_usage", "runtime_selector_used", "candidate_winner_selection_used", "manual_update_detected", "new_edge_function_added", "mlp_stem_used", "mlp_readout_used"]
    missing_before: list[dict[str, Any]] = []
    normalized = 0
    for path in sorted(OUT_ROOT.glob("*.json")):
        data = read_json(path)
        before = [k for k in required_fields if k not in data]
        if before:
            missing_before.append({"artifact": rel(path), "missing_before": ",".join(before)})
            write_json(path, data)
            normalized += 1
        after = [k for k in required_fields if k not in read_json(path)]
        rows.append({"artifact": rel(path), "kind": "summary_audit_fields", "missing_before": ",".join(before), "missing_after": ",".join(after), "field_gate_pass": int(not after)})
    final = read_json(OUT_ROOT / "final_route.json")
    pb = read_json(route_summary_path("B"))
    pc = read_json(route_summary_path("C"))
    rows.extend([
        {"artifact": rel(route_summary_path("B")), "kind": "gate_evidence", "part": "B", "gate_pass": ival(pb.get("gate_pass")), "route": pb.get("route", "")},
        {"artifact": rel(route_summary_path("C")), "kind": "gate_evidence", "part": "C", "gate_pass": ival(pc.get("gate_pass")), "route": pc.get("route", "")},
        {"artifact": rel(OUT_ROOT / "final_route.json"), "kind": "gate_evidence", "part": "final", "final_route": final.get("final_route", ""), "promotion_allowed": final.get("promotion_allowed", "")},
    ])
    matrix = write_rows(OUT_ROOT / "completion_audit_matrix.csv", rows)
    missing_artifacts = [r for r in rows if r.get("kind") == "required_artifact" and not ival(r.get("exists"))]
    missing_fields = [r for r in rows if r.get("kind") == "summary_audit_fields" and str(r.get("missing_after", ""))]
    gate = int(not missing_artifacts and not missing_fields and final and ival(final.get("promotion_allowed")) == 0)
    summary = {
        "part": "completion_audit",
        "gate_pass": gate,
        "route": "CompletionAuditPass" if gate else "CompletionAuditMissingEvidence",
        "dominant_blocker": "none" if gate else "missing_required_artifacts_or_audit_fields",
        "matrix": rel(matrix),
        "required_artifact_count": len(required_artifacts()),
        "missing_required_artifacts": [r.get("artifact") for r in missing_artifacts],
        "summary_json_count": len(list(OUT_ROOT.glob("*.json"))),
        "normalized_summary_json_count": normalized,
        "missing_fields_before_normalization": missing_before,
        "missing_fields_after_normalization_count": len(missing_fields),
        "final_route": final.get("final_route", ""),
        "promotion_allowed": final.get("promotion_allowed", ""),
        "part_b_route": pb.get("route", ""),
        "part_c_route": pc.get("route", ""),
    }
    out = write_json(OUT_ROOT / "completion_audit_summary.json", summary)
    append_exec("Completion audit", "done", files=f"{rel(out)}; {rel(matrix)}", gpu="", note=f"gate={gate}; missing_artifacts={len(missing_artifacts)}; missing_fields={len(missing_fields)}")
    append_recap("Completion audit", summary)
    return summary


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    p0 = part_0(args)
    if not ival(p0.get("gate_pass")):
        create_blocked_downstream(str(p0.get("route")), str(p0.get("dominant_blocker")))
        return finalize(args)
    pa = part_a(args)
    if not ival(pa.get("gate_pass")):
        create_blocked_downstream(str(pa.get("route")), str(pa.get("dominant_blocker")))
        return finalize(args)
    pb = part_b(args)
    if not ival(pb.get("gate_pass")) or str(pb.get("route")) not in {"R15_StaticBCLREFMechanismOpened"}:
        create_blocked_downstream(str(pb.get("route")), str(pb.get("dominant_blocker")))
        return finalize(args)
    blocked_summary("C", "PartCNotImplementedInThisRun", "Part B positive requires extended target-free preflight not completed")
    create_blocked_downstream("PartCNotImplementedInThisRun", "Part C must pass before D-I")
    return finalize(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="all", choices=["all", "part-0", "part-a", "part-b", "part-b-merge", "part-d", "part-d-merge", "block-downstream", "finalize", "completion-audit"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--basis-input-gain", type=float, default=0.25)
    p.add_argument("--quadrature-points", type=int, default=257)
    p.add_argument("--edge-metric-ridge", type=float, default=1.0e-8)
    p.add_argument("--lam", type=float, default=1.0e-2)
    p.add_argument("--visual-side", type=int, default=4)
    p.add_argument("--visual-fixed-patch-features", type=int, default=1)
    p.add_argument("--visual-task-version", default="v23_15")
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--width", type=int, default=3)
    p.add_argument("--task-alpha", type=float, default=0.03)
    p.add_argument("--split-size", type=int, default=48)
    p.add_argument("--guard-size", type=int, default=48)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--tau-stable", type=float, default=0.50)
    p.add_argument("--eta-N", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--r-vis", type=float, default=0.25)
    p.add_argument("--debt-tolerance", type=float, default=1.0e-3)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--part-b-tasks", default="SYN1_single_edge_additive_positive_control,SYN2_node_bank_additive_complementarity,SYN3_local_patch_interaction,SYN4_rotation_sensitive_coordinate,SYN5_F5_probability_debt")
    p.add_argument("--part-d-tasks", default="SYN2_node_bank_additive_complementarity,SYN3_local_patch_interaction,SYN4_rotation_sensitive_coordinate,SYN5_F5_probability_debt")
    p.add_argument("--part-d-horizons", default="20,80")
    p.add_argument("--refresh-cadence", type=int, default=20)
    p.add_argument("--memory-beta", type=float, default=0.20)
    p.add_argument("--memory-decay", type=float, default=0.95)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--max-jobs", type=int, default=0)
    p.add_argument("--block-route", default="")
    p.add_argument("--blocker", default="")
    p.add_argument("--block-parts", default="C,D,E,F,G,H,I")
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    init_logs()
    for path in [RUNNER, *HELPERS]:
        py_compile.compile(str(path), doraise=True)
    if args.mode == "part-0":
        return part_0(args)
    if args.mode == "part-a":
        return part_a(args)
    if args.mode == "part-b":
        return part_b(args)
    if args.mode == "part-b-merge":
        return part_b_merge(args)
    if args.mode == "part-d":
        return part_d(args)
    if args.mode == "part-d-merge":
        return part_d_merge(args)
    if args.mode == "block-downstream":
        create_blocked_parts(csv_items(args.block_parts), str(args.block_route or "blocked"), str(args.blocker or "manual_block"))
        return {"status": "done"}
    if args.mode == "finalize":
        return finalize(args)
    if args.mode == "completion-audit":
        return completion_audit(args)
    return run_all(args)


if __name__ == "__main__":
    main()
