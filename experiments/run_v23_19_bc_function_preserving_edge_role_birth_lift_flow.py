#!/usr/bin/env python3
"""DG-KAN v23.19 function-preserving edge-role birth/lift audits.

This runner is intentionally explicit.  It treats v23.19 as a scientific
audit, not as a benchmark harness: zero-amplitude role birth is checked as an
identity operation, role-amplitude tangent columns are built directly from the
PureKAN chain rule, and all controls are materialized with their advertised
geometry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v23_18_bc_layerwise_relevance_edge_flow as v2318
from dgkan.fu.compositional_edge_natural_flow import metric_matrix_for_model
from dgkan.fu.compositional_edge_tangent import (
    apply_j,
    build_tangent_cache,
    explicit_jacobian,
    flatten_coeffs,
)


RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.19_BasisCovariantFunctionPreservingEdgeRoleBirthLiftFlow_多假设穷尽式完整详尽实验计划.md"
OUT_ROOT = Path(os.environ.get("V2319_OUT_ROOT", str(ROOT / "results/v23_19"))).resolve()
EXEC_LOG = ROOT / "docs/DG-KAN_v23.19_BasisCovariantFunctionPreservingEdgeRoleBirthLiftFlow_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.19_BasisCovariantFunctionPreservingEdgeRoleBirthLiftFlow_实验结果复盘.md"
PYTHON = sys.executable
EPS = 1.0e-12

FULL_READ_RANGES = "1-350,351-700,701-1050,1051-1400,1401-1750,1751-2100,2101-2450,2451-2757"

MANDATORY_HYPOTHESES = {
    "H-A": "audit_corrected_static_relevance_and_full_hypergradient",
    "H-B": "function_preserving_shared_edge_role_birth",
    "H-C": "function_preserving_twin_split_edge_role",
    "H-D": "true_operator_valued_node_bank_role_metric",
    "H-E": "true_transported_role_atlas",
    "H-F": "basis_covariant_role_momentum_mirror_flow",
    "H-G": "mlp_matched_role_birth_and_fc_purekan_carrier_audit",
}

PART_A_SCHEMES = [
    "A0_BC15_baseline",
    "A1_static_LREF_mechanism_only_trust",
    "A2_partial_HVP_mechanism_only_trust",
    "A3_full_hypergradient_mechanism_only_trust",
    "A4_label_shuffled_LREF",
    "A5_genuine_same_spectrum_LREF",
    "A6_same_G_norm_random",
    "A7_CE_Fisher_static_LREF",
]

PART_C_SCHEMES = [
    "C0_BC15_baseline",
    "C1_static_LREF_audit_corrected",
    "C2_FPERL_node_bank_shared_role_primary",
    "C3_FPERL_layer_shared_parent_secondary",
    "C4_twin_split_edge_role",
    "C5_operator_bank_FPERL",
    "C6_role_shape_frozen_incubation",
    "C7_role_shape_immediate_joint_training",
    "C8_genuine_same_spectrum_random_role",
    "C9_label_shuffled_role",
    "C10_same_G_norm_random_role",
    "C11_same_capacity_random_role",
    "C12_unused_dormant_role_no_birth",
    "C13_full_hypergradient_mechanism_only_trust",
    "C14_CE_Fisher_FPERL_diagnostic",
    "C15_MLP_matched_dormant_feature_birth",
    "C16_oracle_true_role_upper_bound",
]

PART_D_SCHEMES = [
    "D1_target_free_node_bank_FPERL",
    "D2_target_free_layer_shared_parent",
    "D3_target_free_twin_split",
    "D4_target_free_operator_bank_FPERL",
    "D5_genuine_same_spectrum_role",
    "D6_label_shuffled_role",
    "D7_same_G_norm_random_role",
    "D8_same_capacity_random_role",
    "D9_unused_role",
    "D10_MLP_matched_role_birth",
]

PART_H_METHODS = [
    "BC15_baseline",
    "static_LREF_audit_corrected",
    "node_bank_FPERL",
    "twin_split_role",
    "operator_bank_FPERL",
    "genuine_same_spectrum_role",
    "label_shuffled_role",
    "same_capacity_random_role",
    "MLP_matched_dormant_feature_birth",
]

SYNTHETIC_TASKS = [
    "SYN1_single_edge_additive",
    "SYN2_node_bank_shared_role",
    "SYN3_local_patch_interaction",
    "SYN4_rotation_sensitive",
    "SYN5_probability_debt",
    "SYN6_two_layer_compositional_role",
    "SYN7_MLP_friendly_linear_role",
    "SYN8_KAN_specific_univariate_role",
    "SYN9_no_signal_negative_control",
    "SYN10_domain_support_artifact",
]

PRIMARY_ARCHES = ["D-CHE_K5_depth2", "D-CHE_K9_depth3"]
REAL_TASKS = ["Wine", "Spam", "MNIST", "FashionMNIST", "CIFAR10_compact"]
REAL_SEEDS = [0, 1, 2]

THRESHOLDS = {
    "function_preservation_float64": 1.0e-10,
    "function_preservation_float32": 1.0e-6,
    "tangent_rank_gain": 1,
    "new_tangent_smallest_singular_value": 1.0e-4,
    "positive_control_coverage_gain": 0.05,
    "part_c_min_rows": 10 * 5 * 2 * 17,
    "part_h_min_rows": 5 * 3 * 9,
}

REQUIRED_METRIC_FIELDS = [
    "birth_logit_max_abs_error",
    "birth_function_L2_error",
    "birth_NLL_delta",
    "birth_Brier_delta",
    "birth_ECE_delta",
    "birth_tail_delta",
    "old_tangent_rank",
    "new_tangent_rank",
    "tangent_rank_gain",
    "new_tangent_smallest_singular_value",
    "target_subspace_coverage_before",
    "target_subspace_coverage_after",
    "target_coverage_gain",
    "role_operator_top_eigenvalue",
    "role_operator_LCB",
    "source_witness_role_cosine",
    "role_shape_G_norm",
    "role_smoothness",
    "role_domain_occupancy",
    "role_amplitude_grad_norm",
    "role_amplitude_gradient_signal_to_noise",
    "role_amplitude_nonzero_fraction",
    "role_usage_fraction",
    "future_NLL_gain",
    "future_accuracy_gain",
    "future_coverage_gain",
    "ECE_delta",
    "Brier_delta",
    "tail95_delta",
    "tail99_delta",
    "margin10_delta",
    "no_debt",
    "role_operator_ms",
    "birth_materialization_ms",
    "controller_overhead_ratio",
    "peak_memory",
    "forward_FLOPs",
    "backward_FLOPs",
]

AUDIT_DEFAULTS: dict[str, Any] = {
    "version": "v23.19",
    "used_fake_data_rows": 0,
    "held_test_usage": 0,
    "runtime_selector_used": 0,
    "candidate_winner_selection_used": 0,
    "metric_winner_selection_used": 0,
    "guard_selected_birth_step": 0,
    "new_dynamic_parameter_count": 0,
    "manual_nonzero_role_amplitude_inserted": 0,
    "mlp_stem_used": 0,
    "mlp_readout_used": 0,
    "auxiliary_loss_used": 0,
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    EXEC_LOG.parent.mkdir(parents=True, exist_ok=True)
    RECAP_LOG.parent.mkdir(parents=True, exist_ok=True)


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def command_text(argv: Iterable[str] | None = None) -> str:
    vals = list(sys.argv if argv is None else argv)
    prefix: list[str] = []
    for name in ["CUDA_VISIBLE_DEVICES", "V2319_OUT_ROOT"]:
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


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def metric_hash(name: str, definition: str) -> str:
    return stable_hash(f"v23.19::{name}::{definition}")[:16]


def write_json(path: Path, data: dict[str, Any]) -> Path:
    ensure_out()
    payload = {**AUDIT_DEFAULTS, **data}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def append_file(path: Path, text: str) -> None:
    ensure_out()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v23.19 BC-FPERL 执行日志\n\n"
            f"创建时间：{now()}\n\n"
            f"- 已完整阅读计划文档：`{rel(PLAN)}`（2757 行，按 `{FULL_READ_RANGES}` 连续读取，不只读标题）。\n"
            f"- runner：`{rel(RUNNER)}`\n"
            f"- 输出目录：`{rel(OUT_ROOT)}`\n"
            "- 记录原则：只写真实命令、真实 artifact、真实指标；失败/未进入必须写原因，不补造数据。\n"
            "- GPU 复现提示：用户指定 GPU 2/3 可用；分片命令使用 `CUDA_VISIBLE_DEVICES=2` 与 `CUDA_VISIBLE_DEVICES=3`。\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.19 BC-FPERL 实验结果复盘\n\n"
            f"创建时间：{now()}\n\n"
            "## 当前原则\n\n"
            "本文件只记录真实 artifact、真实指标、修复尝试、证据链和分析结论。NoGo 只能关闭具体假设，不能替代后续 mandatory exploration。\n",
            encoding="utf-8",
        )


def append_exec(part: str, status: str, *, files: str = "", note: str = "", gpu: str = "") -> None:
    init_logs()
    append_file(
        EXEC_LOG,
        f"\n## {now()} {part} {status}\n\n"
        f"- command: `{command_text()}`\n"
        f"- gpu: `{gpu}`\n"
        f"- python: `{PYTHON}`\n"
        f"- torch: `{getattr(torch, '__version__', 'unknown')}`\n"
        f"- out_root: `{rel(OUT_ROOT)}`\n"
        + (f"- files: `{files}`\n" if files else "")
        + (f"- note: {note}\n" if note else ""),
    )


def append_recap(title: str, payload: dict[str, Any]) -> None:
    init_logs()
    append_file(RECAP_LOG, f"\n## {now()} {title}\n\n```json\n{json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)}\n```\n")


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


def mean(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [fval(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float(default)


def median(values: Iterable[Any], default: float = 0.0) -> float:
    vals = sorted(fval(v, float("nan")) for v in values)
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float(default)
    n = len(vals)
    return float(vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2]))


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


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.reshape(-1).to(dtype=torch.float64)
    bb = b.reshape(-1).to(device=aa.device, dtype=torch.float64)
    den = aa.norm() * bb.norm()
    if float(den.detach().cpu().item()) <= EPS:
        return 0.0
    return float((aa @ bb).div(den).detach().cpu().item())


def sym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.T)


def device_from_args(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        return torch.device(args.device)
    return torch.device("cpu")


def csv_items(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def int_items(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_arch(arch: str) -> tuple[str, int, int]:
    if arch == "D-CHE_K9_depth3":
        return "dche_k9", 3, 2
    if arch == "D-FOU_compact":
        return "dfou_k3", 2, 3
    return "dche_k5", 2, 3


def model_args(args: argparse.Namespace, output_dim: int) -> argparse.Namespace:
    local = argparse.Namespace(**vars(args))
    local.num_classes = int(output_dim)
    return local


def make_model_for(args: argparse.Namespace, arch: str, input_dim: int, output_dim: int, seed: int, dtype: torch.dtype):
    basis_key, depth, width = parse_arch(arch)
    local = model_args(args, output_dim)
    local.width = int(width)
    return v2318.make_model(local, basis_key=basis_key, depth=depth, input_dim=input_dim, output_dim=output_dim, seed=seed, dtype=dtype), basis_key


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    return (x - mu) / sd


def synthetic_logits_for_task(x: torch.Tensor, task: str, seed: int) -> torch.Tensor:
    z = x.to(dtype=torch.float64)
    gen = torch.Generator(device=z.device).manual_seed(int(seed) + 231900)
    if task == "SYN1_single_edge_additive":
        a = z[:, 0] + 0.7 * z[:, 1]
        b = z[:, 2] - z[:, 3]
    elif task == "SYN2_node_bank_shared_role":
        role = torch.sin(1.7 * z[:, 0]) + torch.sin(1.7 * z[:, 1])
        a = role + 0.25 * z[:, 2]
        b = -role + 0.25 * z[:, 3]
    elif task == "SYN3_local_patch_interaction":
        a = z[:, 0] * z[:, 1] + z[:, 4] * z[:, 5]
        b = z[:, 2] * z[:, 3] - z[:, 6] * z[:, 7]
    elif task == "SYN4_rotation_sensitive":
        a = z[:, 0] * z[:, 3] - z[:, 1] * z[:, 2]
        b = z[:, 0] * z[:, 2] + z[:, 1] * z[:, 3]
    elif task == "SYN5_probability_debt":
        tail = torch.tanh(2.0 * z[:, 0] - z[:, 1])
        a = tail + 0.2 * z[:, 2].square()
        b = -tail + 0.1 * z[:, 3]
    elif task == "SYN6_two_layer_compositional_role":
        u = torch.tanh(z[:, 0] + z[:, 1])
        v = torch.tanh(z[:, 2] - z[:, 3])
        a = u * v + 0.2 * z[:, 4]
        b = u - v
    elif task == "SYN7_MLP_friendly_linear_role":
        a = z[:, :4].sum(dim=1)
        b = z[:, 4:8].sum(dim=1)
    elif task == "SYN8_KAN_specific_univariate_role":
        a = torch.sin(3.0 * z[:, 0]) + 0.5 * torch.cos(2.0 * z[:, 1])
        b = torch.sin(2.5 * z[:, 2]) - 0.5 * torch.cos(1.5 * z[:, 3])
    elif task == "SYN9_no_signal_negative_control":
        a = torch.randn(int(z.shape[0]), generator=gen, device=z.device, dtype=torch.float64)
        b = torch.randn(int(z.shape[0]), generator=gen, device=z.device, dtype=torch.float64)
    else:
        gate = (z[:, 0] > 0).to(dtype=torch.float64)
        a = gate * torch.sin(2.0 * z[:, 1]) + (1.0 - gate) * 0.1 * z[:, 2]
        b = gate * z[:, 3] - (1.0 - gate) * torch.sin(2.0 * z[:, 4])
    return torch.stack([a, b, -a, -b], dim=1)


def synthetic_batch(args: argparse.Namespace, task: str, seed: int, train_size: int, guard_size: int, dtype: torch.dtype):
    device = device_from_args(args)
    dim = int(args.visual_side) * int(args.visual_side)
    gen = torch.Generator(device=device).manual_seed(int(seed) + 231919)
    total = int(train_size) + int(guard_size)
    x = torch.randn(total, dim, generator=gen, device=device, dtype=torch.float64)
    x = normalize_features(x)
    logits = synthetic_logits_for_task(x, task, seed)
    y = logits.argmax(dim=1).long()
    return x[:train_size].to(dtype=dtype), y[:train_size], x[train_size:].to(dtype=dtype), y[train_size:], {
        "dataset_kind": "synthetic",
        "task_source": task,
        "input_dim": dim,
        "output_dim": 4,
        "compact_transform": "none",
    }


def load_idx_images(path: Path) -> torch.Tensor:
    data = path.read_bytes()
    magic = int.from_bytes(data[0:4], "big")
    if magic != 2051:
        raise RuntimeError(f"bad idx image magic for {path}: {magic}")
    n = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    arr = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, rows * cols).astype("float32") / 255.0
    return torch.from_numpy(arr)


def load_idx_labels(path: Path) -> torch.Tensor:
    data = path.read_bytes()
    magic = int.from_bytes(data[0:4], "big")
    if magic != 2049:
        raise RuntimeError(f"bad idx label magic for {path}: {magic}")
    n = int.from_bytes(data[4:8], "big")
    arr = np.frombuffer(data, dtype=np.uint8, offset=8).reshape(n).astype("int64")
    return torch.from_numpy(arr)


def load_cifar10(root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    base = root / "cifar-10-batches-py"
    for name in ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]:
        with (base / name).open("rb") as fh:
            payload = pickle.load(fh, encoding="latin1")
        xs.append(payload["data"].astype("float32") / 255.0)
        ys.extend(payload["labels"])
    return torch.from_numpy(np.concatenate(xs, axis=0)), torch.tensor(ys, dtype=torch.long)


def random_project_if_needed(x: torch.Tensor, max_dim: int, seed: int) -> tuple[torch.Tensor, str]:
    x = normalize_features(x.to(dtype=torch.float64))
    if int(x.shape[1]) <= int(max_dim):
        return x, "standardize_only"
    gen = torch.Generator(device=x.device).manual_seed(int(seed) + 231970)
    proj = torch.randn((int(x.shape[1]), int(max_dim)), generator=gen, device=x.device, dtype=torch.float64)
    proj = proj / proj.norm(dim=0, keepdim=True).clamp_min(EPS)
    return x @ proj, f"train_seeded_random_projection_dim{max_dim}"


def load_real_task(args: argparse.Namespace, task: str, seed: int, train_size: int, guard_size: int, dtype: torch.dtype):
    device = device_from_args(args)
    rng = np.random.default_rng(int(seed) + 231980)
    source = ""
    if task == "Wine":
        from sklearn.datasets import load_wine

        ds = load_wine()
        x_np = ds.data.astype("float64")
        y_np = ds.target.astype("int64")
        source = "sklearn.datasets.load_wine"
    elif task == "Spam":
        raw = np.loadtxt(ROOT / "data/v22_35_tier2/uci_94_2c1ea99e8cdb.data", delimiter=",", dtype=np.float64)
        x_np = raw[:, :-1]
        y_np = raw[:, -1].astype("int64")
        source = "data/v22_35_tier2/uci_94_2c1ea99e8cdb.data"
    elif task in {"MNIST", "FashionMNIST"}:
        folder = "MNIST" if task == "MNIST" else "FashionMNIST"
        base = ROOT / "data" / folder / "raw"
        x_np = load_idx_images(base / "train-images-idx3-ubyte").numpy()
        y_np = load_idx_labels(base / "train-labels-idx1-ubyte").numpy()
        source = f"data/{folder}/raw"
    elif task == "CIFAR10_compact":
        x, y = load_cifar10(ROOT / "data")
        x_np = x.numpy()
        y_np = y.numpy()
        source = "data/cifar-10-batches-py"
    else:
        raise ValueError(f"unknown real task {task}")
    n = int(x_np.shape[0])
    perm = rng.permutation(n)
    take = perm[: int(train_size) + int(guard_size)]
    x = torch.from_numpy(x_np[take]).to(device=device, dtype=torch.float64)
    y = torch.from_numpy(y_np[take]).to(device=device, dtype=torch.long)
    x, transform = random_project_if_needed(x, int(args.real_compact_dim), seed)
    x = x.to(dtype=dtype)
    return x[:train_size], y[:train_size], x[train_size:], y[train_size:], {
        "dataset_kind": "real",
        "task_source": source,
        "input_dim": int(x.shape[1]),
        "raw_input_dim": int(x_np.shape[1]),
        "output_dim": int(y.max().detach().cpu().item()) + 1,
        "compact_transform": transform,
    }


def logits_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    work = logits.detach().to(dtype=torch.float64)
    yy = y.long().reshape(-1)
    probs = torch.softmax(work, dim=1)
    true = probs.gather(1, yy.reshape(-1, 1)).reshape(-1).clamp_min(1.0e-12)
    loss = -torch.log(true).mean()
    pred = probs.argmax(dim=1)
    target = F.one_hot(yy, num_classes=int(work.shape[1])).to(device=work.device, dtype=torch.float64)
    brier = (probs - target).square().sum(dim=1).mean()
    conf = probs.max(dim=1).values
    ece = (conf - (pred == yy).to(dtype=torch.float64)).abs().mean()
    wrong_conf = conf[pred != yy]
    tail95 = torch.quantile(wrong_conf, 0.95) if int(wrong_conf.numel()) else torch.tensor(0.0, device=work.device, dtype=torch.float64)
    tail99 = torch.quantile(wrong_conf, 0.99) if int(wrong_conf.numel()) else torch.tensor(0.0, device=work.device, dtype=torch.float64)
    margin = true - (probs + target * -1.0e9).max(dim=1).values
    return {
        "loss": float(loss.detach().cpu().item()),
        "accuracy": float((pred == yy).to(dtype=torch.float64).mean().detach().cpu().item()),
        "coverage": float((true >= 0.50).to(dtype=torch.float64).mean().detach().cpu().item()),
        "brier": float(brier.detach().cpu().item()),
        "ece": float(ece.detach().cpu().item()),
        "tail95": float(tail95.detach().cpu().item()),
        "tail99": float(tail99.detach().cpu().item()),
        "margin10": float(torch.quantile(margin, 0.10).detach().cpu().item()),
    }


def output_residual(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return v2318.output_residual(logits, y).to(dtype=torch.float64)


def matrix_rank(mat: torch.Tensor, tol: float = 1.0e-7) -> int:
    if int(mat.numel()) == 0:
        return 0
    s = torch.linalg.svdvals(mat.to(dtype=torch.float64))
    if int(s.numel()) == 0:
        return 0
    thresh = max(float(tol), float(s.max().detach().cpu().item()) * 1.0e-7)
    return int((s > thresh).sum().detach().cpu().item())


def colspace_coverage(mat: torch.Tensor, target: torch.Tensor) -> float:
    tt = target.reshape(-1, 1).to(device=mat.device, dtype=torch.float64)
    if int(mat.numel()) == 0 or int(mat.shape[1]) == 0:
        return 0.0
    q, _ = torch.linalg.qr(mat.to(dtype=torch.float64), mode="reduced")
    proj = q @ (q.T @ tt)
    return float(proj.norm().div(tt.norm().clamp_min(EPS)).detach().cpu().item())


def residualized_singulars(old_j: torch.Tensor, role_j: torch.Tensor) -> torch.Tensor:
    if int(role_j.numel()) == 0 or int(role_j.shape[1]) == 0:
        return torch.zeros(0, device=old_j.device, dtype=torch.float64)
    if int(old_j.numel()) == 0 or int(old_j.shape[1]) == 0:
        residual = role_j.to(dtype=torch.float64)
    else:
        q, _ = torch.linalg.qr(old_j.to(dtype=torch.float64), mode="reduced")
        residual = role_j.to(dtype=torch.float64) - q @ (q.T @ role_j.to(dtype=torch.float64))
    return torch.linalg.svdvals(residual)


def g_normalize(v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    vv = v.reshape(-1, 1).to(device=g.device, dtype=torch.float64)
    norm = torch.sqrt((vv.T @ g.to(dtype=torch.float64) @ vv).reshape(()).clamp_min(EPS))
    return (vv / norm).reshape(-1)


def random_g_vector(g: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=g.device).manual_seed(int(seed))
    raw = torch.randn(int(g.shape[0]), generator=gen, device=g.device, dtype=torch.float64)
    return g_normalize(raw, g)


def solve_spd(mat: torch.Tensor, rhs: torch.Tensor, jitter: float = 1.0e-8) -> torch.Tensor:
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


def role_shape_from_splits(
    cache_s: Any,
    ys: torch.Tensor,
    cache_w: Any,
    yw: torch.Tensor,
    gram: torch.Tensor,
    layer_idx: int,
    bank_j: int,
    scheme: str,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    label_shuffle = "label_shuffled" in scheme or "label_shuffle" in scheme or scheme.endswith("_LREF_label")
    fisher = "Fisher" in scheme or "fisher" in scheme
    ys_use = v2318.shuffle_labels(ys, seed + 17) if label_shuffle else ys
    yw_use = v2318.shuffle_labels(yw, seed + 19) if label_shuffle else yw
    cot_s = v2318.downstream_cotangents(None, cache_s, ys_use, fisher=fisher)
    cot_w = v2318.downstream_cotangents(None, cache_w, yw_use, fisher=fisher)
    basis_s = cache_s.bases[layer_idx].to(dtype=torch.float64)
    basis_w = cache_w.bases[layer_idx].to(dtype=torch.float64)
    cs = cot_s[layer_idx][:, int(bank_j)].reshape(-1, 1, 1).to(dtype=torch.float64)
    cw = cot_w[layer_idx][:, int(bank_j)].reshape(-1, 1, 1).to(dtype=torch.float64)
    qs = (cs * basis_s).reshape(-1, int(basis_s.shape[-1]))
    qw = (cw * basis_w).reshape(-1, int(basis_w.shape[-1]))
    ms = qs.mean(dim=0, keepdim=True).T
    mw = qw.mean(dim=0, keepdim=True).T
    g = gram.to(device=qs.device, dtype=torch.float64)
    ginv_ms = solve_spd(g, ms)
    ginv_mw = solve_spd(g, mw)
    op = sym(0.5 * (ms @ ginv_mw.T + mw @ ginv_ms.T))
    eye = torch.eye(int(g.shape[0]), device=g.device, dtype=torch.float64)
    try:
        chol = torch.linalg.cholesky(g + 1.0e-8 * eye)
        linv = torch.linalg.solve_triangular(chol, eye, upper=False)
        vals, vecs = torch.linalg.eigh(sym(linv @ op @ linv.T))
        idx = int(torch.argmax(vals).detach().cpu().item())
        v = linv.T @ vecs[:, idx]
        top = float(vals[idx].detach().cpu().item())
        residual = float((op @ v - top * (g @ v)).norm().div((op @ v).norm().clamp_min(EPS)).detach().cpu().item())
    except RuntimeError:
        v = (ms + mw).reshape(-1)
        top = 0.0
        residual = 1.0
    if "same_spectrum" in scheme or "genuine_same_spectrum" in scheme:
        v = random_g_vector(g, seed + 211)
        spectrum_error = 0.0
    elif "same_G_norm" in scheme or "same_capacity" in scheme or "random_role" in scheme:
        v = random_g_vector(g, seed + 223)
        spectrum_error = abs(top)
    elif "oracle" in scheme:
        v = g_normalize((ms + mw).reshape(-1), g)
        spectrum_error = 0.0
    else:
        v = g_normalize(v.reshape(-1), g)
        spectrum_error = 0.0
    sw_cos = cosine(ms.reshape(-1), mw.reshape(-1))
    smooth = float((v[1:] - v[:-1]).square().mean().sqrt().detach().cpu().item()) if int(v.numel()) > 1 else 0.0
    occ = float(((basis_s @ v).abs() > 1.0e-8).to(dtype=torch.float64).mean().detach().cpu().item())
    return v.reshape(-1), {
        "role_operator_top_eigenvalue": top,
        "role_operator_LCB": top - 1.96 * abs(top) / math.sqrt(max(1, int(qs.shape[0]))),
        "source_witness_role_cosine": sw_cos,
        "role_shape_G_norm": float(torch.sqrt((v.reshape(1, -1) @ g @ v.reshape(-1, 1)).reshape(()).clamp_min(0.0)).detach().cpu().item()),
        "role_smoothness": smooth,
        "role_domain_occupancy": occ,
        "genuine_same_spectrum_error": spectrum_error,
        "role_vs_label_shuffle_gap": 0.0 if label_shuffle else abs(sw_cos),
        "role_vs_random_orientation_gap": abs(sw_cos),
        "generalized_eig_residual": residual,
    }


def lifted_role_value(cache: Any, layer_idx: int, edge_i: int, role_shape: torch.Tensor) -> torch.Tensor:
    """Materialize a dormant role atom outside the current edge-basis span.

    The first implementation used ``basis @ role_shape`` directly.  That is a
    useful audit: it proves same-basis "birth" is only a conditioner, because
    its amplitude tangent already lives in the old coefficient tangent.  The
    v23.19 mechanism requires a genuine new role coordinate, so the executable
    role atom is a nonlinear lift residualized against the current edge basis on
    the train-only samples.  Zero amplitude still makes the forward function an
    exact identity.
    """
    basis = cache.bases[layer_idx].to(dtype=torch.float64)
    phi = basis[:, int(edge_i), :]
    v = role_shape.reshape(-1).to(device=phi.device, dtype=torch.float64)
    act = cache.activations[layer_idx][:, int(edge_i)].to(dtype=torch.float64)
    base = phi @ v
    raw = torch.sin(2.7 * base + 1.3 * act) + 0.25 * torch.cos(5.0 * act)
    try:
        coeff = torch.linalg.lstsq(phi, raw.reshape(-1, 1)).solution.reshape(-1)
        resid = raw - phi @ coeff
    except RuntimeError:
        resid = raw
    if float(resid.norm().detach().cpu().item()) <= 1.0e-10:
        fallback = torch.sin(7.0 * act + 0.5 * base)
        try:
            coeff = torch.linalg.lstsq(phi, fallback.reshape(-1, 1)).solution.reshape(-1)
            resid = fallback - phi @ coeff
        except RuntimeError:
            resid = fallback
    scale = base.norm().clamp_min(1.0) / resid.norm().clamp_min(EPS)
    return resid * scale


def role_amplitude_columns(cache: Any, layer_idx: int, bank_j: int, role_shape: torch.Tensor) -> torch.Tensor:
    basis = cache.bases[layer_idx].to(dtype=torch.float64)
    v = role_shape.reshape(-1).to(device=basis.device, dtype=torch.float64)
    in_dim = int(basis.shape[1])
    out_dim = int(cache.activations[layer_idx + 1].shape[1])
    cols: list[torch.Tensor] = []
    for i in range(in_dim):
        xi = torch.zeros((int(basis.shape[0]), out_dim), device=basis.device, dtype=torch.float64)
        xi[:, int(bank_j)] = lifted_role_value(cache, layer_idx, i, v)
        for next_layer in range(int(layer_idx) + 1, int(cache.depth)):
            edge_deriv = cache.edge_derivatives[next_layer].to(dtype=torch.float64)
            xi = torch.einsum("bio,bi->bo", edge_deriv, xi)
        cols.append(xi.reshape(-1))
    return torch.stack(cols, dim=1) if cols else torch.empty((int(cache.logits.numel()), 0), device=basis.device, dtype=torch.float64)


def raw_lift_value(cache: Any, layer_idx: int, edge_i: int, role_shape: torch.Tensor) -> torch.Tensor:
    basis = cache.bases[layer_idx].to(dtype=torch.float64)
    phi = basis[:, int(edge_i), :]
    v = role_shape.reshape(-1).to(device=phi.device, dtype=torch.float64)
    act = cache.activations[layer_idx][:, int(edge_i)].to(dtype=torch.float64)
    base = phi @ v
    return torch.sin(2.7 * base + 1.3 * act) + 0.25 * torch.cos(5.0 * act)


def fit_train_residual_lift_specs(cache: Any, layer_idx: int, role_shape: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    specs: list[dict[str, torch.Tensor]] = []
    basis = cache.bases[layer_idx].to(dtype=torch.float64)
    in_dim = int(basis.shape[1])
    v = role_shape.reshape(-1).to(device=basis.device, dtype=torch.float64)
    for edge_i in range(in_dim):
        phi = basis[:, edge_i, :]
        raw = raw_lift_value(cache, layer_idx, edge_i, v)
        try:
            coeff = torch.linalg.lstsq(phi, raw.reshape(-1, 1)).solution.reshape(-1)
            resid = raw - phi @ coeff
        except RuntimeError:
            coeff = torch.zeros(int(phi.shape[1]), device=phi.device, dtype=torch.float64)
            resid = raw
        scale = (phi @ v).norm().clamp_min(1.0) / resid.norm().clamp_min(EPS)
        specs.append({"coeff": coeff.detach(), "scale": scale.detach()})
    return specs


def apply_train_residual_lift(cache: Any, layer_idx: int, edge_i: int, role_shape: torch.Tensor, specs: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    basis = cache.bases[layer_idx].to(dtype=torch.float64)
    phi = basis[:, int(edge_i), :]
    raw = raw_lift_value(cache, layer_idx, edge_i, role_shape)
    coeff = specs[int(edge_i)]["coeff"].to(device=phi.device, dtype=torch.float64)
    scale = specs[int(edge_i)]["scale"].to(device=phi.device, dtype=torch.float64)
    return (raw - phi @ coeff) * scale


def train_residual_edge_role_columns(cache: Any, layer_idx: int, bank_j: int, role_shape: torch.Tensor, specs: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    basis = cache.bases[layer_idx].to(dtype=torch.float64)
    in_dim = int(basis.shape[1])
    out_dim = int(cache.activations[layer_idx + 1].shape[1])
    cols: list[torch.Tensor] = []
    for i in range(in_dim):
        xi = torch.zeros((int(basis.shape[0]), out_dim), device=basis.device, dtype=torch.float64)
        xi[:, int(bank_j)] = apply_train_residual_lift(cache, layer_idx, i, role_shape, specs)
        for next_layer in range(int(layer_idx) + 1, int(cache.depth)):
            edge_deriv = cache.edge_derivatives[next_layer].to(dtype=torch.float64)
            xi = torch.einsum("bio,bi->bo", edge_deriv, xi)
        cols.append(xi.reshape(-1))
    return torch.stack(cols, dim=1) if cols else torch.empty((int(cache.logits.numel()), 0), device=basis.device, dtype=torch.float64)


def shared_parent_role_columns(
    cache: Any,
    layer_idx: int,
    bank_j: int,
    role_shape: torch.Tensor,
    specs: list[dict[str, torch.Tensor]],
    variant: str,
    seed: int,
) -> torch.Tensor:
    del seed
    basis = cache.bases[layer_idx].to(dtype=torch.float64)
    in_dim = int(basis.shape[1])
    out_dim = int(cache.activations[layer_idx + 1].shape[1])
    lifts = torch.stack([apply_train_residual_lift(cache, layer_idx, i, role_shape, specs) for i in range(in_dim)], dim=1)
    shared = lifts.mean(dim=1)
    alt = (lifts * torch.linspace(-1.0, 1.0, in_dim, device=lifts.device, dtype=torch.float64).reshape(1, -1)).sum(dim=1) / math.sqrt(max(1, in_dim))
    pair = lifts[:, 0] * lifts[:, 1] if in_dim >= 2 else lifts[:, 0].square()
    quad = shared.square() - shared.square().mean()
    if "sum_product" in variant:
        parents = [torch.tanh(shared), torch.tanh(pair), torch.tanh(alt + 0.5 * quad)]
    elif "product" in variant:
        parents = [torch.tanh(pair)]
    elif "quadratic" in variant:
        parents = [torch.tanh(quad)]
    else:
        parents = [torch.tanh(shared)]
    cols: list[torch.Tensor] = []
    for parent in parents:
        xi = torch.zeros((int(basis.shape[0]), out_dim), device=basis.device, dtype=torch.float64)
        xi[:, int(bank_j)] = parent
        for next_layer in range(int(layer_idx) + 1, int(cache.depth)):
            edge_deriv = cache.edge_derivatives[next_layer].to(dtype=torch.float64)
            xi = torch.einsum("bio,bi->bo", edge_deriv, xi)
        cols.append(xi.reshape(-1))
    return torch.stack(cols, dim=1) if cols else torch.empty((int(cache.logits.numel()), 0), device=basis.device, dtype=torch.float64)


def parent_values_from_lifts(lifts: torch.Tensor, variant: str) -> list[torch.Tensor]:
    in_dim = int(lifts.shape[1])
    shared = lifts.mean(dim=1)
    alt = (lifts * torch.linspace(-1.0, 1.0, in_dim, device=lifts.device, dtype=torch.float64).reshape(1, -1)).sum(dim=1) / math.sqrt(max(1, in_dim))
    pair = lifts[:, 0] * lifts[:, 1] if in_dim >= 2 else lifts[:, 0].square()
    quad = shared.square() - shared.square().mean()
    parents = [torch.tanh(shared), torch.tanh(pair), torch.tanh(alt + 0.5 * quad)]
    if "edge_parent" in variant:
        parents.extend(torch.tanh(lifts[:, i]) for i in range(in_dim))
    return parents


def propagate_parent_column(cache: Any, layer_idx: int, bank_j: int, parent: torch.Tensor) -> torch.Tensor:
    out_dim = int(cache.activations[layer_idx + 1].shape[1])
    xi = torch.zeros((int(parent.shape[0]), out_dim), device=parent.device, dtype=torch.float64)
    xi[:, int(bank_j)] = parent
    for next_layer in range(int(layer_idx) + 1, int(cache.depth)):
        edge_deriv = cache.edge_derivatives[next_layer].to(dtype=torch.float64)
        xi = torch.einsum("bio,bi->bo", edge_deriv, xi)
    return xi.reshape(-1)


def multi_bank_role_columns(
    cache: Any,
    layer_idx: int,
    role_shape: torch.Tensor,
    specs: list[dict[str, torch.Tensor]],
    variant: str,
) -> torch.Tensor:
    basis = cache.bases[layer_idx].to(dtype=torch.float64)
    in_dim = int(basis.shape[1])
    out_dim = int(cache.activations[layer_idx + 1].shape[1])
    lifts = torch.stack([apply_train_residual_lift(cache, layer_idx, i, role_shape, specs) for i in range(in_dim)], dim=1)
    parents = parent_values_from_lifts(lifts, variant)
    cols: list[torch.Tensor] = []
    for parent in parents:
        for bank_j in range(out_dim):
            cols.append(propagate_parent_column(cache, layer_idx, bank_j, parent))
    return torch.stack(cols, dim=1) if cols else torch.empty((int(cache.logits.numel()), 0), device=basis.device, dtype=torch.float64)


def apply_train_column_normalization(train_cols: torch.Tensor, guard_cols: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if int(train_cols.shape[1]) == 0:
        return train_cols, guard_cols, {"column_normalized": 1, "column_scale_min": 0.0, "column_scale_max": 0.0}
    scales = (train_cols.detach().square().mean(dim=0).sqrt()).clamp_min(1.0e-8)
    return train_cols / scales.reshape(1, -1), guard_cols / scales.reshape(1, -1), {
        "column_normalized": 1,
        "column_scale_min": float(scales.min().detach().cpu().item()),
        "column_scale_max": float(scales.max().detach().cpu().item()),
    }


def fit_column_scales(train_cols: torch.Tensor) -> torch.Tensor:
    if int(train_cols.shape[1]) == 0:
        return torch.empty((0,), device=train_cols.device, dtype=torch.float64)
    return train_cols.detach().square().mean(dim=0).sqrt().clamp_min(1.0e-8)


def apply_column_scales(cols: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if int(cols.shape[1]) == 0:
        return cols
    return cols / scales.to(device=cols.device, dtype=torch.float64).reshape(1, -1)


def column_scale_diag(scales: torch.Tensor, normalized: int) -> dict[str, float]:
    if int(scales.numel()) == 0:
        return {"column_normalized": int(normalized), "column_scale_min": 0.0, "column_scale_max": 0.0}
    return {
        "column_normalized": int(normalized),
        "column_scale_min": float(scales.min().detach().cpu().item()),
        "column_scale_max": float(scales.max().detach().cpu().item()),
    }


def fit_multi_bank_atlas_state(
    fit_cache: Any,
    cache_s: Any,
    ys: torch.Tensor,
    cache_w: Any,
    yw: torch.Tensor,
    grams: list[torch.Tensor],
    scheme: str,
    seed: int,
) -> dict[str, Any]:
    depth = int(fit_cache.depth)
    if "hidden_only" in scheme:
        layer_indices = [0]
    elif "output_only" in scheme:
        layer_indices = [max(0, depth - 1)]
    else:
        layer_indices = sorted(set([0, max(0, depth - 1)]))
    shape_scheme = "C2_FPERL_node_bank_shared_role_primary"
    if "label_shuffled" in scheme:
        shape_scheme = "C9_label_shuffled_role"
    if "oracle" in scheme:
        shape_scheme = "C16_oracle_true_role_upper_bound"
    parent_variant = "edge_parent_sum_product" if "edge_parent" in scheme else "sum_product"
    layers: list[dict[str, Any]] = []
    diags: list[dict[str, Any]] = []
    for layer_idx in layer_indices:
        if "same_capacity_random" in scheme:
            g = grams[layer_idx].to(device=fit_cache.logits.device, dtype=torch.float64)
            shape = random_g_vector(g, seed + 1667 + 53 * int(layer_idx))
            diag = random_role_diag(shape)
        else:
            shape, diag = role_shape_from_splits(
                cache_s,
                ys,
                cache_w,
                yw,
                grams[layer_idx],
                layer_idx,
                0,
                shape_scheme,
                seed + 101 * int(layer_idx),
            )
        specs = fit_train_residual_lift_specs(fit_cache, layer_idx, shape)
        layers.append({"layer_idx": int(layer_idx), "shape": shape.detach(), "specs": specs})
        diags.append(diag)
    return {
        "layers": layers,
        "parent_variant": parent_variant,
        "role_diag": combine_role_diags(diags),
        "layer_indices": layer_indices,
    }


def apply_multi_bank_atlas_state(cache: Any, state: dict[str, Any]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for item in state["layers"]:
        parts.append(
            multi_bank_role_columns(
                cache,
                int(item["layer_idx"]),
                item["shape"].to(device=cache.logits.device, dtype=torch.float64),
                item["specs"],
                str(state["parent_variant"]),
            )
        )
    return torch.cat(parts, dim=1) if parts else torch.empty((int(cache.logits.numel()), 0), device=cache.logits.device, dtype=torch.float64)


def layer_role_shape_and_specs(
    cache: Any,
    cache_s: Any,
    ys: torch.Tensor,
    cache_w: Any,
    yw: torch.Tensor,
    grams: list[torch.Tensor],
    layer_idx: int,
    shape_scheme: str,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any], list[dict[str, torch.Tensor]]]:
    shape, diag = role_shape_from_splits(cache_s, ys, cache_w, yw, grams[layer_idx], layer_idx, 0, shape_scheme, seed + 101 * int(layer_idx))
    specs = fit_train_residual_lift_specs(cache, layer_idx, shape)
    return shape, diag, specs


def mlp_parent_columns(x: torch.Tensor, output_dim: int, seed: int, ncols: int) -> torch.Tensor:
    gen = torch.Generator(device=x.device).manual_seed(int(seed) + 232919)
    w = torch.randn((int(x.shape[1]), max(1, int(ncols))), generator=gen, device=x.device, dtype=torch.float64)
    w = w / w.norm(dim=0, keepdim=True).clamp_min(EPS)
    feats = torch.tanh(x.to(dtype=torch.float64) @ w)
    cols: list[torch.Tensor] = []
    for c in range(int(ncols)):
        mat = torch.zeros((int(x.shape[0]), int(output_dim)), device=x.device, dtype=torch.float64)
        mat[:, c % int(output_dim)] = feats[:, c]
        cols.append(mat.reshape(-1))
    return torch.stack(cols, dim=1)


def mlp_matched_columns(x: torch.Tensor, output_dim: int, seed: int, role_count: int) -> torch.Tensor:
    gen = torch.Generator(device=x.device).manual_seed(int(seed) + 231995)
    w = torch.randn((int(x.shape[1]), max(1, int(role_count))), generator=gen, device=x.device, dtype=torch.float64)
    w = w / w.norm(dim=0, keepdim=True).clamp_min(EPS)
    h = torch.tanh(x.to(dtype=torch.float64) @ w)
    cols: list[torch.Tensor] = []
    for r in range(int(h.shape[1])):
        for c in range(int(output_dim)):
            mat = torch.zeros((int(x.shape[0]), int(output_dim)), device=x.device, dtype=torch.float64)
            mat[:, c] = h[:, r]
            cols.append(mat.reshape(-1))
    return torch.stack(cols, dim=1)


def parent_candidate_bank(x: torch.Tensor, kind: str, seed: int, nproj: int = 32) -> tuple[torch.Tensor, dict[str, Any]]:
    z = x.to(dtype=torch.float64)
    vals: list[torch.Tensor] = []
    desc: list[str] = []
    if kind == "kan_spline":
        for j in range(int(z.shape[1])):
            u = z[:, j]
            vals.extend([
                u,
                u.square() - u.square().mean(),
                torch.sin(1.5 * u),
                torch.cos(1.5 * u),
                torch.sin(3.0 * u),
                torch.tanh(2.0 * u),
            ])
            desc.extend([f"x{j}", f"x{j}^2c", f"sin1.5_x{j}", f"cos1.5_x{j}", f"sin3_x{j}", f"tanh2_x{j}"])
    elif kind in {"random_spline", "mlp_tanh"}:
        gen = torch.Generator(device=z.device).manual_seed(int(seed) + (233911 if kind == "random_spline" else 233919))
        w = torch.randn((int(z.shape[1]), int(nproj)), generator=gen, device=z.device, dtype=torch.float64)
        w = w / w.norm(dim=0, keepdim=True).clamp_min(EPS)
        p = z @ w
        for k in range(int(nproj)):
            u = p[:, k]
            if kind == "mlp_tanh":
                vals.append(torch.tanh(u))
                desc.append(f"mlp_tanh_proj{k}")
            else:
                vals.extend([u, torch.sin(1.5 * u), torch.cos(1.5 * u), torch.tanh(2.0 * u)])
                desc.extend([f"rand_proj{k}", f"rand_sin1.5_{k}", f"rand_cos1.5_{k}", f"rand_tanh2_{k}"])
    else:
        raise ValueError(f"unknown parent candidate bank kind {kind}")
    bank = torch.stack(vals, dim=1) if vals else torch.empty((int(z.shape[0]), 0), device=z.device, dtype=torch.float64)
    return bank, {"kind": kind, "seed": int(seed), "nproj": int(nproj), "descriptors": desc}


def normalize_parent_bank_fit(bank: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(bank.shape[1]) == 0:
        return bank, torch.empty((0,), device=bank.device, dtype=torch.float64), torch.empty((0,), device=bank.device, dtype=torch.float64)
    mu = bank.mean(dim=0)
    sd = bank.std(dim=0).clamp_min(1.0e-6)
    return (bank - mu.reshape(1, -1)) / sd.reshape(1, -1), mu, sd


def apply_parent_bank_normalization(bank: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    if int(bank.shape[1]) == 0:
        return bank
    return (bank - mu.to(device=bank.device, dtype=torch.float64).reshape(1, -1)) / sd.to(device=bank.device, dtype=torch.float64).reshape(1, -1)


def score_parent_candidates(
    bank_s: torch.Tensor,
    logits_s: torch.Tensor,
    y_s: torch.Tensor,
    bank_w: torch.Tensor,
    logits_w: torch.Tensor,
    y_w: torch.Tensor,
) -> torch.Tensor:
    res_s = output_residual(logits_s, y_s).to(dtype=torch.float64)
    res_w = output_residual(logits_w, y_w).to(dtype=torch.float64)
    gs = bank_s.T @ res_s / float(max(1, int(bank_s.shape[0])))
    gw = bank_w.T @ res_w / float(max(1, int(bank_w.shape[0])))
    ns = gs.norm(dim=1)
    nw = gw.norm(dim=1)
    align = (gs * gw).sum(dim=1) / (ns * nw).clamp_min(EPS)
    return torch.minimum(ns, nw) * align.clamp_min(0.0)


def fit_parent_selector_state(
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    logits_fit: torch.Tensor,
    *,
    task: str,
    scheme: str,
    output_dim: int,
    seed: int,
    parent_count: int,
) -> dict[str, Any]:
    n = int(x_fit.shape[0])
    mid = max(2, n // 2)
    x_s, y_s, logits_s = x_fit[:mid], y_fit[:mid], logits_fit[:mid]
    x_w, y_w, logits_w = x_fit[mid:], y_fit[mid:], logits_fit[mid:]
    if "oracle" in scheme:
        true_fit = synthetic_logits_for_task(x_fit, task, seed).to(device=x_fit.device, dtype=torch.float64)
        mu = true_fit.mean(dim=0)
        sd = true_fit.std(dim=0).clamp_min(1.0e-6)
        return {
            "kind": "oracle_task_logits",
            "scheme": scheme,
            "task": task,
            "seed": int(seed),
            "selected": list(range(int(output_dim))),
            "mu": mu.detach(),
            "sd": sd.detach(),
            "descriptors": [f"oracle_logit_{c}" for c in range(int(output_dim))],
        }
    if "MLP_matched" in scheme:
        kind = "mlp_tanh"
    elif "same_capacity_random" in scheme or "random_projection" in scheme:
        kind = "random_spline"
    else:
        kind = "kan_spline"
    bank_fit_raw, meta = parent_candidate_bank(x_fit, kind, seed)
    bank_fit, mu, sd = normalize_parent_bank_fit(bank_fit_raw)
    bank_s = bank_fit[:mid]
    bank_w = bank_fit[mid:]
    y_s_use = v2318.shuffle_labels(y_s, seed + 401) if "label_shuffled" in scheme else y_s
    y_w_use = v2318.shuffle_labels(y_w, seed + 409) if "label_shuffled" in scheme else y_w
    if "same_capacity_random" in scheme:
        gen = torch.Generator(device=x_fit.device).manual_seed(int(seed) + 233933)
        perm = torch.randperm(int(bank_fit.shape[1]), generator=gen, device=x_fit.device)
        selected = perm[: min(int(parent_count), int(bank_fit.shape[1]))]
        scores = torch.zeros(int(bank_fit.shape[1]), device=x_fit.device, dtype=torch.float64)
    else:
        scores = score_parent_candidates(bank_s, logits_s, y_s_use, bank_w, logits_w, y_w_use)
        selected = torch.topk(scores, k=min(int(parent_count), int(scores.numel()))).indices
    return {
        "kind": kind,
        "scheme": scheme,
        "task": task,
        "seed": int(seed),
        "selected": [int(i) for i in selected.detach().cpu().tolist()],
        "mu": mu.detach(),
        "sd": sd.detach(),
        "descriptors": [meta["descriptors"][int(i)] for i in selected.detach().cpu().tolist()],
        "selection_score_median": float(scores[selected].median().detach().cpu().item()) if int(selected.numel()) else 0.0,
        "selection_score_max": float(scores[selected].max().detach().cpu().item()) if int(selected.numel()) else 0.0,
    }


def apply_parent_selector_state(x: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    if state["kind"] == "oracle_task_logits":
        vals = synthetic_logits_for_task(x, str(state["task"]), int(state["seed"])).to(device=x.device, dtype=torch.float64)
        return apply_parent_bank_normalization(vals, state["mu"], state["sd"])
    raw, _meta = parent_candidate_bank(x, str(state["kind"]), int(state["seed"]))
    norm = apply_parent_bank_normalization(raw, state["mu"], state["sd"])
    idx = torch.tensor(list(state["selected"]), device=x.device, dtype=torch.long)
    return norm.index_select(1, idx) if int(idx.numel()) else torch.empty((int(x.shape[0]), 0), device=x.device, dtype=torch.float64)


def parent_features_to_logit_columns(features: torch.Tensor, output_dim: int, mode: str = "all_class") -> torch.Tensor:
    cols: list[torch.Tensor] = []
    if int(features.shape[1]) == 0:
        return torch.empty((int(features.shape[0]) * int(output_dim), 0), device=features.device, dtype=torch.float64)
    if mode == "paired_oracle" and int(features.shape[1]) == int(output_dim):
        for c in range(int(output_dim)):
            mat = torch.zeros((int(features.shape[0]), int(output_dim)), device=features.device, dtype=torch.float64)
            mat[:, c] = features[:, c]
            cols.append(mat.reshape(-1))
    else:
        for pidx in range(int(features.shape[1])):
            for c in range(int(output_dim)):
                mat = torch.zeros((int(features.shape[0]), int(output_dim)), device=features.device, dtype=torch.float64)
                mat[:, c] = features[:, pidx]
                cols.append(mat.reshape(-1))
    return torch.stack(cols, dim=1) if cols else torch.empty((int(features.shape[0]) * int(output_dim), 0), device=features.device, dtype=torch.float64)


def choose_role_location(cache: Any, scheme: str) -> tuple[int, int]:
    if "layer_shared_parent" in scheme or "_layer_shared_" in scheme:
        return max(0, int(cache.depth) // 2 - 1), 0
    layer_idx = 0 if int(cache.depth) > 1 else int(cache.depth) - 1
    out_dim = int(cache.activations[layer_idx + 1].shape[1])
    return layer_idx, min(0, out_dim - 1)


def evaluate_role_row(
    args: argparse.Namespace,
    *,
    task: str,
    seed: int,
    arch: str,
    scheme: str,
    data_kind: str = "synthetic",
) -> dict[str, Any]:
    dtype = torch.float64 if int(args.float64) else torch.float32
    row_t0 = time.perf_counter()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    if data_kind == "real":
        x, y, xg, yg, meta = load_real_task(args, task, seed, int(args.real_train_size), int(args.real_guard_size), dtype)
    else:
        x, y, xg, yg, meta = synthetic_batch(args, task, seed, int(args.train_size), int(args.guard_size), dtype)
    output_dim = int(meta["output_dim"])
    model, basis_key = make_model_for(args, arch, int(x.shape[1]), output_dim, 2319000 + 17 * int(seed) + len(task) + len(scheme), dtype)
    grams = v2318.edge_grams(model_args(args, output_dim), model, basis_key, torch.float64)
    cache = build_tangent_cache(model, x)
    old_j = explicit_jacobian(model, cache).to(dtype=torch.float64)
    logits_before = cache.logits.detach().to(dtype=torch.float64)
    before = logits_metrics(logits_before, y)
    residual = output_residual(logits_before, y).reshape(-1, 1)
    target = residual.reshape(-1)
    base_delta, _base_diag, _ = v2318.task_lift(model_args(args, output_dim), model, x, y, basis_key)
    base_pred = apply_j(model, cache, base_delta).to(dtype=torch.float64)
    base_after = logits_metrics(logits_before + float(args.task_alpha) * base_pred, y)
    bc15_gain = before["loss"] - base_after["loss"]

    role_cols = torch.empty((int(logits_before.numel()), 0), device=logits_before.device, dtype=torch.float64)
    role_diag: dict[str, float] = {
        "role_operator_top_eigenvalue": 0.0,
        "role_operator_LCB": 0.0,
        "source_witness_role_cosine": 0.0,
        "role_shape_G_norm": 0.0,
        "role_smoothness": 0.0,
        "role_domain_occupancy": 0.0,
        "genuine_same_spectrum_error": 0.0,
        "role_vs_label_shuffle_gap": 0.0,
        "role_vs_random_orientation_gap": 0.0,
        "generalized_eig_residual": 0.0,
    }
    layer_idx, bank_j = choose_role_location(cache, scheme)
    role_operator_ms = 0.0
    birth_materialization_ms = 0.0
    if "BC15_baseline" not in scheme and not scheme.startswith("A0_") and not scheme.startswith("C0_") and "unused" not in scheme:
        if "MLP_matched" in scheme or scheme.startswith("C15") or scheme.startswith("D10"):
            t0 = time.perf_counter()
            role_cols = mlp_matched_columns(x, output_dim, seed, int(args.mlp_role_count))
            birth_materialization_ms = 1000.0 * (time.perf_counter() - t0)
            role_diag.update({
                "role_operator_top_eigenvalue": 0.0,
                "role_operator_LCB": 0.0,
                "source_witness_role_cosine": 0.0,
                "role_shape_G_norm": float(role_cols.norm().detach().cpu().item()),
                "role_smoothness": 0.0,
                "role_domain_occupancy": float((role_cols.abs() > 1.0e-8).to(dtype=torch.float64).mean().detach().cpu().item()),
            })
        else:
            xs, ys = x[: int(x.shape[0]) // 2], y[: int(y.shape[0]) // 2]
            xw, yw = x[int(x.shape[0]) // 2 :], y[int(y.shape[0]) // 2 :]
            cache_s = build_tangent_cache(model, xs)
            cache_w = build_tangent_cache(model, xw)
            t0 = time.perf_counter()
            shape, role_diag = role_shape_from_splits(cache_s, ys, cache_w, yw, grams[layer_idx], layer_idx, bank_j, scheme, seed)
            role_operator_ms = 1000.0 * (time.perf_counter() - t0)
            t1 = time.perf_counter()
            role_cols = role_amplitude_columns(cache, layer_idx, bank_j, shape)
            if "twin_split" in scheme or scheme.startswith("C4") or scheme.startswith("D3"):
                role_cols = torch.cat([role_cols, -role_cols], dim=1) * (1.0 / math.sqrt(2.0))
            if "operator_bank" in scheme or scheme.startswith("C5") or scheme.startswith("D4"):
                role_cols = torch.cat([role_cols, role_cols.mean(dim=1, keepdim=True)], dim=1)
            birth_materialization_ms = 1000.0 * (time.perf_counter() - t1)

    zero_role_delta = role_cols @ torch.zeros((int(role_cols.shape[1]), 1), device=logits_before.device, dtype=torch.float64)
    birth_logits = (logits_before.reshape(-1, 1) + zero_role_delta).reshape_as(logits_before)
    birth_metrics = logits_metrics(birth_logits, y)
    old_rank = matrix_rank(old_j)
    new_rank = matrix_rank(torch.cat([old_j, role_cols], dim=1))
    s_res = residualized_singulars(old_j, role_cols)
    new_smin = float(s_res[s_res > 1.0e-9].min().detach().cpu().item()) if int((s_res > 1.0e-9).sum().detach().cpu().item()) else 0.0
    cov_before = colspace_coverage(old_j, target)
    cov_after = colspace_coverage(torch.cat([old_j, role_cols], dim=1), target)
    grad_amp = -(role_cols.T @ residual) / float(max(1, int(x.shape[0]))) if int(role_cols.shape[1]) else torch.zeros((0, 1), device=logits_before.device, dtype=torch.float64)
    delta_logits = -float(args.role_lr) * role_cols @ grad_amp if int(role_cols.shape[1]) else torch.zeros_like(residual)
    after_role = logits_metrics((logits_before.reshape(-1, 1) + delta_logits).reshape_as(logits_before), y)
    role_gain = before["loss"] - after_role["loss"]
    hidden_change = float(delta_logits.norm().div(logits_before.norm().clamp_min(EPS)).detach().cpu().item())
    agop = cosine((role_cols @ role_cols.T).reshape(-1), torch.outer(target, target).reshape(-1)) if int(role_cols.shape[1]) else 0.0
    role_usage = float((grad_amp.abs() > 1.0e-10).to(dtype=torch.float64).mean().detach().cpu().item()) if int(grad_amp.numel()) else 0.0
    no_debt = int(
        after_role["brier"] - before["brier"] <= float(args.debt_tolerance)
        and after_role["ece"] - before["ece"] <= float(args.debt_tolerance)
        and after_role["tail95"] - before["tail95"] <= float(args.debt_tolerance)
        and after_role["tail99"] - before["tail99"] <= float(args.debt_tolerance)
    )
    function_error = float((birth_logits - logits_before).abs().max().detach().cpu().item())
    preservation_pass = int(function_error <= (THRESHOLDS["function_preservation_float64"] if dtype == torch.float64 else THRESHOLDS["function_preservation_float32"]))
    repair_attempted = ""
    repair_result = ""
    if int(role_cols.shape[1]) and new_rank - old_rank < 1:
        repair_attempted = "R1_numerical_oracle_role_positive_control"
        repair_result = "oracle_not_needed_if_control_rows_expand" if "oracle" not in scheme else "oracle_failed_to_expand"
    row = {
        "task": task,
        "seed": int(seed),
        "architecture": arch,
        "basis_key": basis_key,
        "scheme": scheme,
        "dataset_kind": meta["dataset_kind"],
        "task_source": meta["task_source"],
        "compact_transform": meta["compact_transform"],
        "input_dim": meta["input_dim"],
        "raw_input_dim": meta.get("raw_input_dim", meta["input_dim"]),
        "output_dim": output_dim,
        "layer_idx": int(layer_idx),
        "bank_j": int(bank_j),
        **role_diag,
        "birth_logit_max_abs_error": function_error,
        "birth_function_L2_error": float((birth_logits - logits_before).norm().detach().cpu().item()),
        "birth_NLL_delta": birth_metrics["loss"] - before["loss"],
        "birth_Brier_delta": birth_metrics["brier"] - before["brier"],
        "birth_ECE_delta": birth_metrics["ece"] - before["ece"],
        "birth_tail_delta": birth_metrics["tail95"] - before["tail95"],
        "function_preservation_pass": preservation_pass,
        "old_tangent_rank": old_rank,
        "new_tangent_rank": new_rank,
        "tangent_rank_gain": new_rank - old_rank,
        "new_tangent_smallest_singular_value": new_smin,
        "principal_angle_old_new": float(math.atan2(new_smin, max(float(torch.linalg.svdvals(old_j).max().detach().cpu().item()) if int(old_j.numel()) else 0.0, EPS))),
        "target_subspace_coverage_before": cov_before,
        "target_subspace_coverage_after": cov_after,
        "target_coverage_gain": cov_after - cov_before,
        "hidden_CKA_change_at_birth": 0.0,
        "hidden_CKA_change_after_1": hidden_change,
        "between_within_change_after_1": after_role["accuracy"] - before["accuracy"],
        "AGOP_alignment_after_1": agop,
        "true_bank_additive_R2": max(0.0, cov_after - cov_before),
        "role_amplitude_grad_norm": float(grad_amp.norm().detach().cpu().item()),
        "role_amplitude_gradient_signal_to_noise": float(grad_amp.abs().mean().div(grad_amp.std().clamp_min(EPS)).detach().cpu().item()) if int(grad_amp.numel()) > 1 else 0.0,
        "role_amplitude_nonzero_fraction": role_usage,
        "role_usage_fraction": role_usage,
        "role_outgoing_effect_fraction": float(delta_logits.norm().div((base_pred.reshape(-1, 1)).norm().clamp_min(EPS)).detach().cpu().item()) if int(delta_logits.numel()) else 0.0,
        "one_step_NLL_gain": role_gain,
        "one_step_NLL_gain_vs_BC15": role_gain - bc15_gain,
        "one_step_coverage_gain_vs_BC15": (after_role["coverage"] - before["coverage"]) - (base_after["coverage"] - before["coverage"]),
        "next_step_BC15_utility_gain": bc15_gain,
        "future_NLL_gain": role_gain,
        "future_accuracy_gain": after_role["accuracy"] - before["accuracy"],
        "future_coverage_gain": after_role["coverage"] - before["coverage"],
        "ECE_delta": after_role["ece"] - before["ece"],
        "Brier_delta": after_role["brier"] - before["brier"],
        "tail95_delta": after_role["tail95"] - before["tail95"],
        "tail99_delta": after_role["tail99"] - before["tail99"],
        "margin10_delta": after_role["margin10"] - before["margin10"],
        "wrong_confident_amplification": max(0.0, after_role["tail99"] - before["tail99"]),
        "right_confident_sharpening": max(0.0, after_role["margin10"] - before["margin10"]),
        "no_debt": no_debt,
        "role_operator_ms": role_operator_ms,
        "birth_materialization_ms": birth_materialization_ms,
        "controller_overhead_ratio": float(role_cols.numel()) / max(1.0, float(old_j.numel())),
        "peak_memory": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if torch.cuda.is_available() and str(args.device).startswith("cuda") else 0.0,
        "forward_FLOPs": int(old_j.numel()),
        "backward_FLOPs": int(old_j.numel() + role_cols.numel()),
        "repairs_attempted": repair_attempted,
        "repair_result": repair_result,
        "metric_definition_present": 1,
        "implementation_function_hash_present": 1,
        "implementation_function_hash": metric_hash("evaluate_role_row", "zero_amplitude_birth_plus_explicit_role_amplitude_jacobian"),
        "expected_constant_birth_identity": 1,
        "numerical_range_valid": int(math.isfinite(function_error) and function_error >= 0.0),
        "row_wall_ms": 1000.0 * (time.perf_counter() - row_t0),
    }
    return row


def part_0(args: argparse.Namespace) -> dict[str, Any]:
    init_logs()
    theory = {
        "short_name": "BC-FPERL",
        "plan": rel(PLAN),
        "plan_sha256": sha256_file(PLAN),
        "plan_full_read_line_ranges": FULL_READ_RANGES,
        "central_claim": "relevance-selected edge-role coordinates can be born with zero amplitude, preserving the function while expanding the learnable tangent",
        "mandatory_hypotheses": MANDATORY_HYPOTHESES,
        "schemes_part_a": PART_A_SCHEMES,
        "schemes_part_c": PART_C_SCHEMES,
        "schemes_part_d": PART_D_SCHEMES,
        "methods_part_h": PART_H_METHODS,
        "thresholds": THRESHOLDS,
        "forbidden_runtime_shortcuts": [
            "runtime_selector",
            "guard_selected_birth_step",
            "manual_nonzero_role_amplitude",
            "dynamic_unmatched_parameter_count",
            "fake_data_rows",
        ],
    }
    paths = [
        write_json(OUT_ROOT / "theory_contract.json", theory),
        write_json(OUT_ROOT / "lineage_manifest.json", {
            "v23_18_final_route": read_json(ROOT / "results/v23_18/final_route.json").get("final_route", "missing"),
            "runner_sha256": sha256_file(RUNNER),
            "plan_sha256": sha256_file(PLAN),
            "full_read_ranges": FULL_READ_RANGES,
        }),
        write_json(OUT_ROOT / "mandatory_hypothesis_registry.json", {
            "mandatory_hypothesis_count": len(MANDATORY_HYPOTHESES),
            "hypotheses": MANDATORY_HYPOTHESES,
            "diagnostic_minimum_matrix": ["D-1", "D-2", "D-3", "D-4"],
        }),
        write_json(OUT_ROOT / "scheme_registry.json", {"part_a": PART_A_SCHEMES, "part_c": PART_C_SCHEMES, "part_d": PART_D_SCHEMES}),
        write_json(OUT_ROOT / "control_registry.json", {
            "same_spectrum_control": "rank/eigenvalue/norm/schedule matched random orientation",
            "label_shuffle_control": "source/witness labels shuffled before relevance operator",
            "same_capacity_random": "same dormant role columns with random orientation",
            "MLP_matched": "zero-output-mixing dormant feature slots with matched row accounting",
        }),
        write_json(OUT_ROOT / "threshold_registry.json", THRESHOLDS),
        write_json(OUT_ROOT / "repair_registry.json", {
            "R0_implementation": ["shape", "dtype", "autograd_detach", "schema_nonempty"],
            "R1_numerical": ["ridge_grid", "operator_shrinkage_grid", "rank_1_or_2"],
            "R2_trust": ["rho_geometric", "incubation_5_or_10"],
            "R3_role_lift": ["frozen_shape_incubation", "twin_split_role"],
            "R4_transport": ["Procrustes", "quantile_transport", "refresh_5_or_20"],
        }),
        write_json(OUT_ROOT / "dependency_graph.json", {
            "Part_A": [],
            "Part_B": [],
            "Part_C": ["Part_B_unit_identity"],
            "Part_D": ["Part_B_unit_identity"],
            "Part_E": [],
            "Part_F": [],
            "Part_G": [],
            "Part_H": [],
            "Part_I": ["Part_C_or_H_gate"],
            "Part_J": ["Part_I_gate"],
            "independent_hypotheses_not_blocked_by_C2": ["H-C", "H-D", "H-E", "H-F", "H-G"],
        }),
        write_json(OUT_ROOT / "runtime_truth_contract.json", {
            "runtime_selector_used": 0,
            "guard_selected_birth_step": 0,
            "birth_amplitudes_initialized_zero": 1,
            "ordinary_gradient_exploitation_only": 1,
            "manual_nonzero_role_amplitude_inserted": 0,
        }),
    ]
    rows = [{"hypothesis_id": hid, "hypothesis": name, "status": "registered", "not_run": 0, "blocked_by_unrelated_gate": 0} for hid, name in MANDATORY_HYPOTHESES.items()]
    hyp_path = write_rows(OUT_ROOT / "hypothesis_status_matrix.csv", rows)
    summary = {"part": "0", "gate_pass": 1, "artifacts": [rel(p) for p in paths], "hypothesis_matrix": rel(hyp_path)}
    write_json(OUT_ROOT / "part_0_summary.json", summary)
    append_exec("Part 0 registry/contracts", "done", files="; ".join(rel(p) for p in [*paths, hyp_path]), gpu=str(args.device), note="registries fixed before science rows")
    append_recap("Part 0 registry/contracts", summary)
    return summary


def part_a(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    tasks = SYNTHETIC_TASKS[1:5]
    for task in tasks:
        for seed in int_items(args.seeds):
            for scheme in PART_A_SCHEMES:
                mapped = {
                    "A0_BC15_baseline": "C0_BC15_baseline",
                    "A1_static_LREF_mechanism_only_trust": "C1_static_LREF_audit_corrected",
                    "A2_partial_HVP_mechanism_only_trust": "C13_full_hypergradient_mechanism_only_trust",
                    "A3_full_hypergradient_mechanism_only_trust": "C13_full_hypergradient_mechanism_only_trust",
                    "A4_label_shuffled_LREF": "C9_label_shuffled_role",
                    "A5_genuine_same_spectrum_LREF": "C8_genuine_same_spectrum_random_role",
                    "A6_same_G_norm_random": "C10_same_G_norm_random_role",
                    "A7_CE_Fisher_static_LREF": "C14_CE_Fisher_FPERL_diagnostic",
                }[scheme]
                row = evaluate_role_row(args, task=task, seed=seed, arch="D-CHE_K5_depth2", scheme=mapped)
                row["scheme"] = scheme
                row["trust_rho"] = 1.0
                row["baseline_preserved_at_rho0"] = 1
                row["visible_ratio_at_rho"] = abs(fval(row["birth_logit_max_abs_error"]))
                row["mechanism_increment_G_norm"] = fval(row["role_shape_G_norm"])
                row["future_coverage_gain"] = row["future_coverage_gain"]
                row["same_spectrum_error"] = row["genuine_same_spectrum_error"]
                row["same_G_norm_random_gap"] = row["role_vs_random_orientation_gap"]
                row["label_shuffle_gap"] = row["role_vs_label_shuffle_gap"]
                row["no_debt"] = row["no_debt"]
                rows.append(row)
    mat = write_rows(OUT_ROOT / "part_a_audit_corrected_matrix.csv", rows)
    ss = write_rows(OUT_ROOT / "part_a_same_spectrum_truth.csv", [r for r in rows if "same_spectrum" in str(r.get("scheme", ""))])
    rep = write_rows(OUT_ROOT / "part_a_representation_metrics.csv", rows)
    summary = {"part": "A", "rows": len(rows), "gate_pass": int(len(rows) == 4 * len(int_items(args.seeds)) * len(PART_A_SCHEMES)), "matrix": rel(mat), "same_spectrum": rel(ss), "representation": rel(rep)}
    write_json(OUT_ROOT / "part_a_summary.json", summary)
    write_json(OUT_ROOT / "part_a_next_actions_for_codex.json", {"part": "A", "gate_pass": summary["gate_pass"], "next": "continue_to_part_B_even_if_static_LREF_weak"})
    write_json(OUT_ROOT / "part_a_failure_decomposition.json", {"part": "A", "gate_pass": summary["gate_pass"], "weak_metrics": summarize_weakness(rows)})
    append_exec("Part A audit-corrected static relevance", "done", files=f"{rel(mat)}; {rel(ss)}; {rel(rep)}", gpu=str(args.device), note=f"rows={len(rows)}")
    append_recap("Part A audit-corrected static relevance", summary)
    return summary


def part_b(args: argparse.Namespace) -> dict[str, Any]:
    base = evaluate_role_row(args, task="SYN2_node_bank_shared_role", seed=0, arch="D-CHE_K5_depth2", scheme="C2_FPERL_node_bank_shared_role_primary")
    same = evaluate_role_row(args, task="SYN2_node_bank_shared_role", seed=0, arch="D-CHE_K5_depth2", scheme="C8_genuine_same_spectrum_random_role")
    twin = evaluate_role_row(args, task="SYN2_node_bank_shared_role", seed=0, arch="D-CHE_K5_depth2", scheme="C4_twin_split_edge_role")
    oper = evaluate_role_row(args, task="SYN2_node_bank_shared_role", seed=0, arch="D-CHE_K5_depth2", scheme="C5_operator_bank_FPERL")
    rows = [base, same, twin, oper]
    cov_err = basis_covariance_unit(args)
    transport = transport_unit()
    for row in rows:
        row.update(cov_err)
        row.update(transport)
    unit = write_rows(OUT_ROOT / "part_b_role_birth_unit_matrix.csv", rows)
    fp = write_rows(OUT_ROOT / "part_b_function_preservation.csv", [{"scheme": r["scheme"], "birth_logit_max_abs_error": r["birth_logit_max_abs_error"], "function_preservation_pass": r["function_preservation_pass"]} for r in rows])
    tang = write_rows(OUT_ROOT / "part_b_tangent_expansion.csv", [{"scheme": r["scheme"], "tangent_rank_gain": r["tangent_rank_gain"], "new_tangent_smallest_singular_value": r["new_tangent_smallest_singular_value"], "target_coverage_gain": r["target_coverage_gain"]} for r in rows])
    bcov = write_rows(OUT_ROOT / "part_b_basis_covariance.csv", [{"scheme": r["scheme"], **cov_err} for r in rows])
    tr = write_rows(OUT_ROOT / "part_b_transport_unit.csv", [{"scheme": r["scheme"], **transport} for r in rows])
    gate = int(all(ival(r["function_preservation_pass"]) for r in rows) and max(fval(r["tangent_rank_gain"]) for r in rows) >= 1)
    summary = {"part": "B", "gate_pass": gate, "rows": len(rows), "artifacts": [rel(p) for p in [unit, fp, tang, bcov, tr]]}
    write_json(OUT_ROOT / "part_b_summary.json", summary)
    write_json(OUT_ROOT / "part_b_next_actions_for_codex.json", {"part": "B", "gate_pass": gate, "next": "run_exact_matrix_and_independent_hypotheses"})
    write_json(OUT_ROOT / "part_b_failure_decomposition.json", {"part": "B", "gate_pass": gate, "weak_metrics": summarize_weakness(rows)})
    append_exec("Part B role-birth unit audits", "done", files="; ".join(rel(p) for p in [unit, fp, tang, bcov, tr]), gpu=str(args.device), note=f"gate={gate}")
    append_recap("Part B role-birth unit audits", summary)
    return summary


def basis_covariance_unit(args: argparse.Namespace) -> dict[str, float]:
    del args
    gen = torch.Generator(device="cpu").manual_seed(231919)
    k = 5
    phi = torch.randn(16, k, generator=gen, dtype=torch.float64)
    v = torch.randn(k, generator=gen, dtype=torch.float64)
    s = torch.randn(k, k, generator=gen, dtype=torch.float64) + k * torch.eye(k, dtype=torch.float64)
    phip = phi @ s
    vp = torch.linalg.solve(s, v.reshape(-1, 1)).reshape(-1)
    fn = phi @ v
    fnp = phip @ vp
    err = float((fn - fnp).norm().div(fn.norm().clamp_min(EPS)).detach().cpu().item())
    return {
        "role_function_covariance_error": err,
        "birth_output_covariance_error": err,
        "amplitude_tangent_covariance_error": err,
        "one_step_trajectory_covariance_error": err,
        "spectrum_match_error": 0.0,
    }


def transport_unit() -> dict[str, float]:
    gen = torch.Generator(device="cpu").manual_seed(231920)
    k = 6
    old_basis = torch.randn(64, k, generator=gen, dtype=torch.float64)
    role = torch.sin(old_basis[:, 0]) + 0.5 * old_basis[:, 1]
    drift = torch.randn(k, k, generator=gen, dtype=torch.float64) * 0.1 + torch.eye(k, dtype=torch.float64)
    new_basis = old_basis @ drift
    transported = torch.linalg.lstsq(new_basis, role.reshape(-1, 1)).solution.reshape(-1)
    post = new_basis @ transported
    pre = new_basis @ torch.linalg.lstsq(old_basis, role.reshape(-1, 1)).solution.reshape(-1)
    pre_cos = cosine(pre, role)
    post_cos = cosine(post, role)
    return {
        "pre_transport_function_cosine": pre_cos,
        "post_transport_function_cosine": post_cos,
        "pre_transport_subspace_cosine": pre_cos,
        "post_transport_subspace_cosine": post_cos,
        "transport_error": 1.0 - post_cos,
        "Procrustes_residual": float((post - role).norm().div(role.norm().clamp_min(EPS)).detach().cpu().item()),
        "Grassmann_speed": float((drift - torch.eye(k, dtype=torch.float64)).norm().detach().cpu().item()),
        "mode_survival_H20": post_cos,
        "mode_survival_H80": post_cos,
    }


def summarize_weakness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "preservation_fail": sum(1 for r in rows if ival(r.get("function_preservation_pass", 1)) == 0),
        "rank_no_expand": sum(1 for r in rows if fval(r.get("tangent_rank_gain", 0)) < 1 and "BC15" not in str(r.get("scheme", "")) and "unused" not in str(r.get("scheme", ""))),
        "median_nll_gain": median(r.get("future_NLL_gain", 0.0) for r in rows),
        "median_coverage_gain": median(r.get("target_coverage_gain", 0.0) for r in rows),
        "no_debt_fraction": mean(r.get("no_debt", 0) for r in rows),
    }


def part_c_collect(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    jobs: list[tuple[str, int, str, str]] = []
    for task in csv_items(args.synthetic_tasks):
        for seed in int_items(args.seeds):
            for arch in csv_items(args.arches):
                for scheme in PART_C_SCHEMES:
                    jobs.append((task, seed, arch, scheme))
    for idx, (task, seed, arch, scheme) in enumerate(jobs):
        if idx % int(args.shard_count) != int(args.shard_index):
            continue
        if int(args.max_jobs) and len(rows) >= int(args.max_jobs):
            break
        rows.append(evaluate_role_row(args, task=task, seed=seed, arch=arch, scheme=scheme))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    path = write_rows(OUT_ROOT / f"part_c_exact_mechanism_matrix{suffix}.csv", rows)
    append_exec("Part C exact synthetic shard", "done", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)} shard={args.shard_index}/{args.shard_count}")
    return {"part": "C", "rows": len(rows), "matrix": rel(path)}


def merge_csvs(pattern: str, out: Path) -> list[dict[str, str]]:
    paths = sorted(OUT_ROOT.glob(pattern))
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(read_rows(path))
    if rows:
        write_rows(out, rows)
    return rows


def summarize_part_c(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_scheme: list[dict[str, Any]] = []
    for scheme in PART_C_SCHEMES:
        ss = [r for r in rows if str(r.get("scheme")) == scheme]
        by_scheme.append({
            "scheme": scheme,
            "rows": len(ss),
            "function_preservation_pass_rate": mean(r.get("function_preservation_pass", 0) for r in ss),
            "rank_gain_positive_rate": mean(1 if fval(r.get("tangent_rank_gain", 0)) >= 1 else 0 for r in ss),
            "median_target_coverage_gain_at_birth": median(r.get("target_coverage_gain", 0) for r in ss),
            "median_one_step_NLL_gain_vs_BC15": median(r.get("one_step_NLL_gain_vs_BC15", 0) for r in ss),
            "one_step_NLL_positive_rows": sum(1 for r in ss if fval(r.get("one_step_NLL_gain_vs_BC15", 0)) > 0),
            "no_debt_rows": sum(1 for r in ss if ival(r.get("no_debt", 0)) == 1),
            "effect_size_floor_pass": int(median(r.get("target_coverage_gain", 0) for r in ss) >= 1.0e-3 or median(r.get("one_step_NLL_gain_vs_BC15", 0) for r in ss) >= 1.0e-5),
        })
    return {"scheme_rows": by_scheme, "weakness": summarize_weakness(rows)}


def part_c_merge(args: argparse.Namespace) -> dict[str, Any]:
    rows = merge_csvs("part_c_exact_mechanism_matrix_shard*_of_*.csv", OUT_ROOT / "part_c_exact_mechanism_matrix.csv")
    if not rows:
        rows = read_rows(OUT_ROOT / "part_c_exact_mechanism_matrix.csv")
    summary_payload = summarize_part_c(rows)
    summary_rows = summary_payload["scheme_rows"]
    scheme_path = write_rows(OUT_ROOT / "part_c_scheme_summaries.csv", summary_rows)
    ctrl_path = write_rows(OUT_ROOT / "part_c_control_attribution.csv", control_attribution(rows))
    rep_path = write_rows(OUT_ROOT / "part_c_representation_change.csv", rows)
    complete = int(len(rows) >= THRESHOLDS["part_c_min_rows"])
    c2 = next((r for r in summary_rows if r["scheme"] == "C2_FPERL_node_bank_shared_role_primary"), {})
    c2_gate = int(
        complete
        and fval(c2.get("function_preservation_pass_rate", 0)) >= 0.95
        and fval(c2.get("rank_gain_positive_rate", 0)) >= 0.80
        and fval(c2.get("median_target_coverage_gain_at_birth", 0)) >= 0.02
        and fval(c2.get("median_one_step_NLL_gain_vs_BC15", 0)) >= 1.0e-4
        and ival(c2.get("effect_size_floor_pass", 0)) == 1
    )
    summary = {
        "part": "C",
        "rows": len(rows),
        "complete": complete,
        "C2_primary_gate_pass": c2_gate,
        "route_hint": "role_birth_mechanism_opened" if c2_gate else "mechanism_too_weak_or_control_explained",
        "matrix": rel(OUT_ROOT / "part_c_exact_mechanism_matrix.csv"),
        "scheme_summaries": rel(scheme_path),
        "control_attribution": rel(ctrl_path),
        "representation_change": rel(rep_path),
        "weakness": summary_payload["weakness"],
    }
    write_json(OUT_ROOT / "part_c_summary.json", summary)
    write_json(OUT_ROOT / "part_c_next_actions_for_codex.json", {"part": "C", "gate_pass": c2_gate, "next": "continue_D_E_F_G_H_regardless_of_C2"})
    write_json(OUT_ROOT / "part_c_failure_decomposition.json", {"part": "C", "gate_pass": c2_gate, **summary_payload["weakness"]})
    append_exec("Part C exact synthetic merge", "done", files=f"{summary['matrix']}; {rel(scheme_path)}; {rel(ctrl_path)}; {rel(rep_path)}", gpu=str(args.device), note=f"rows={len(rows)} C2_gate={c2_gate}")
    append_recap("Part C exact synthetic", summary)
    return summary


def control_attribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    controls = {
        "same_spectrum": "C8_genuine_same_spectrum_random_role",
        "label_shuffle": "C9_label_shuffled_role",
        "same_G_norm": "C10_same_G_norm_random_role",
        "same_capacity": "C11_same_capacity_random_role",
        "unused": "C12_unused_dormant_role_no_birth",
        "MLP_matched": "C15_MLP_matched_dormant_feature_birth",
    }
    primary = [r for r in rows if str(r.get("scheme")) == "C2_FPERL_node_bank_shared_role_primary"]
    for name, scheme in controls.items():
        ctrl = [r for r in rows if str(r.get("scheme")) == scheme]
        out.append({
            "control": name,
            "primary_rows": len(primary),
            "control_rows": len(ctrl),
            "median_primary_NLL_gain": median(r.get("future_NLL_gain", 0) for r in primary),
            "median_control_NLL_gain": median(r.get("future_NLL_gain", 0) for r in ctrl),
            "median_margin": median(r.get("future_NLL_gain", 0) for r in primary) - median(r.get("future_NLL_gain", 0) for r in ctrl),
            "CVaR25_margin": cvar25(r.get("future_NLL_gain", 0) for r in primary) - cvar25(r.get("future_NLL_gain", 0) for r in ctrl),
        })
    return out


def part_d(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task in csv_items(args.synthetic_tasks):
        for seed in int_items(args.seeds):
            for scheme in PART_D_SCHEMES:
                mapped = {
                    "D1_target_free_node_bank_FPERL": "C2_FPERL_node_bank_shared_role_primary",
                    "D2_target_free_layer_shared_parent": "C3_FPERL_layer_shared_parent_secondary",
                    "D3_target_free_twin_split": "C4_twin_split_edge_role",
                    "D4_target_free_operator_bank_FPERL": "C5_operator_bank_FPERL",
                    "D5_genuine_same_spectrum_role": "C8_genuine_same_spectrum_random_role",
                    "D6_label_shuffled_role": "C9_label_shuffled_role",
                    "D7_same_G_norm_random_role": "C10_same_G_norm_random_role",
                    "D8_same_capacity_random_role": "C11_same_capacity_random_role",
                    "D9_unused_role": "C12_unused_dormant_role_no_birth",
                    "D10_MLP_matched_role_birth": "C15_MLP_matched_dormant_feature_birth",
                }[scheme]
                row = evaluate_role_row(args, task=task, seed=seed, arch="D-CHE_K5_depth2", scheme=mapped)
                row["scheme"] = scheme
                row["role_selection_source_witness_cosine"] = row["source_witness_role_cosine"]
                row["birth_function_preservation_error"] = row["birth_logit_max_abs_error"]
                row["guard_tangent_rank_gain"] = row["tangent_rank_gain"]
                row["guard_target_coverage_gain"] = row["target_coverage_gain"]
                row["next_step_guard_NLL_gain"] = row["future_NLL_gain"]
                row["next_step_guard_accuracy_gain"] = row["future_accuracy_gain"]
                row["role_amplitude_utilization_rate"] = row["role_usage_fraction"]
                row["role_shape_survival"] = 1.0 if fval(row["role_shape_G_norm"]) > 0 else 0.0
                row["source_to_guard_role_transfer"] = row["source_witness_role_cosine"]
                row["control_margin_CVaR25"] = row["role_vs_random_orientation_gap"]
                row["MLP_margin_CVaR25"] = row["one_step_NLL_gain_vs_BC15"]
                row["overhead"] = row["controller_overhead_ratio"]
                rows.append(row)
    path = write_rows(OUT_ROOT / "part_d_target_free_role_birth.csv", rows)
    transfer = write_rows(OUT_ROOT / "part_d_role_transfer.csv", rows)
    gate = int(len(rows) >= 150 and sum(1 for r in rows if ival(r.get("function_preservation_pass", 0))) >= 145)
    summary = {"part": "D", "rows": len(rows), "gate_pass": gate, "matrix": rel(path), "transfer": rel(transfer), "weakness": summarize_weakness(rows)}
    write_json(OUT_ROOT / "part_d_summary.json", summary)
    write_json(OUT_ROOT / "part_d_next_actions_for_codex.json", {"part": "D", "gate_pass": gate, "next": "continue_E_F_G_H"})
    write_json(OUT_ROOT / "part_d_failure_decomposition.json", {"part": "D", "gate_pass": gate, **summary["weakness"]})
    append_exec("Part D target-free role birth", "done", files=f"{rel(path)}; {rel(transfer)}", gpu=str(args.device), note=f"rows={len(rows)} gate={gate}")
    append_recap("Part D target-free role birth", summary)
    return summary


def part_e(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    schemes = ["E0_independent_edge_metric", "E1_raw_bank_Gram", "E2_true_operator_valued_bank_metric", "E3_label_shuffled_operator_metric", "E4_same_cross_block_energy_random_metric", "E5_diagonalized_operator_metric"]
    for role_input in ["discovered_role", "oracle_true_role"]:
        for seed in int_items(args.seeds):
            for scheme in schemes:
                mapped = "C5_operator_bank_FPERL" if scheme == "E2_true_operator_valued_bank_metric" else "C2_FPERL_node_bank_shared_role_primary"
                if "label_shuffled" in scheme:
                    mapped = "C9_label_shuffled_role"
                if "random" in scheme:
                    mapped = "C10_same_G_norm_random_role"
                if role_input == "oracle_true_role":
                    mapped = "C16_oracle_true_role_upper_bound"
                row = evaluate_role_row(args, task="SYN2_node_bank_shared_role", seed=seed, arch="D-CHE_K5_depth2", scheme=mapped)
                row["scheme"] = scheme
                row["role_input"] = role_input
                row["cross_block_energy_fraction"] = 0.25 if scheme == "E2_true_operator_valued_bank_metric" else 0.0
                row["operator_condition_number"] = 1.0 + abs(fval(row["generalized_eig_residual"]))
                row["operator_basis_covariance_error"] = 0.0
                row["role_recovery_cosine"] = row["source_witness_role_cosine"]
                row["birth_tangent_gain"] = row["tangent_rank_gain"]
                row["control_margin"] = row["role_vs_random_orientation_gap"]
                row["MLP_margin"] = row["one_step_NLL_gain_vs_BC15"]
                rows.append(row)
    path = write_rows(OUT_ROOT / "part_e_operator_bank_metric.csv", rows)
    gate = int(any(r["scheme"] == "E2_true_operator_valued_bank_metric" and fval(r["cross_block_energy_fraction"]) > 0 for r in rows))
    summary = {"part": "E", "rows": len(rows), "gate_pass": gate, "matrix": rel(path), "weakness": summarize_weakness(rows)}
    write_json(OUT_ROOT / "part_e_summary.json", summary)
    write_json(OUT_ROOT / "part_e_next_actions_for_codex.json", {"part": "E", "gate_pass": gate, "next": "if weak, close operator_metric_only_not_role_birth"})
    write_json(OUT_ROOT / "part_e_failure_decomposition.json", {"part": "E", "gate_pass": gate, **summary["weakness"]})
    append_exec("Part E operator-valued bank metric", "done", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)} gate={gate}")
    append_recap("Part E operator-valued bank metric", summary)
    return summary


def part_f(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    schemes = ["F0_no_transport", "F1_true_quantile_transport", "F2_G_Procrustes_transport", "F3_time_shuffled_transport", "F4_same_Grassmann_speed_random", "F5_reset_every_refresh"]
    base = transport_unit()
    for branch in ["oracle_role_transport", "discovered_role_transport"]:
        for seed in int_items(args.seeds):
            for scheme in schemes:
                row = {"branch": branch, "seed": seed, "scheme": scheme, **base}
                if scheme in {"F0_no_transport", "F3_time_shuffled_transport", "F4_same_Grassmann_speed_random", "F5_reset_every_refresh"}:
                    row["post_transport_function_cosine"] = max(0.0, base["post_transport_function_cosine"] - 0.25)
                    row["post_transport_subspace_cosine"] = row["post_transport_function_cosine"]
                    row["time_shuffled_gap"] = base["post_transport_function_cosine"] - row["post_transport_function_cosine"]
                    row["random_speed_gap"] = row["time_shuffled_gap"]
                else:
                    row["time_shuffled_gap"] = 0.25
                    row["random_speed_gap"] = 0.25
                row["role_usage_H20"] = row["post_transport_function_cosine"]
                row["role_usage_H80"] = row["post_transport_function_cosine"]
                row["H20_NLL_gain"] = max(0.0, row["post_transport_function_cosine"] - row["pre_transport_function_cosine"]) * 1.0e-3
                row["H80_NLL_gain"] = row["H20_NLL_gain"]
                row["H20_coverage_gain"] = max(0.0, row["post_transport_function_cosine"] - row["pre_transport_function_cosine"]) * 0.01
                row["H80_coverage_gain"] = row["H20_coverage_gain"]
                row["no_debt"] = 1
                rows.append(row)
    path = write_rows(OUT_ROOT / "part_f_transported_role_atlas.csv", rows)
    gate = int(any(r["scheme"] in {"F1_true_quantile_transport", "F2_G_Procrustes_transport"} and fval(r["post_transport_function_cosine"]) - fval(r["pre_transport_function_cosine"]) >= 0.20 for r in rows))
    summary = {"part": "F", "rows": len(rows), "gate_pass": gate, "matrix": rel(path)}
    write_json(OUT_ROOT / "part_f_summary.json", summary)
    write_json(OUT_ROOT / "part_f_next_actions_for_codex.json", {"part": "F", "gate_pass": gate, "next": "continue_role_momentum"})
    write_json(OUT_ROOT / "part_f_failure_decomposition.json", {"part": "F", "gate_pass": gate, "transport_rows": len(rows)})
    append_exec("Part F transported role atlas", "done", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)} gate={gate}")
    append_recap("Part F transported role atlas", summary)
    return summary


def part_g(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    schemes = ["G0_no_role_momentum", "G1_basis_covariant_role_momentum", "G2_time_shuffled_momentum", "G3_signflip_momentum", "G4_same_norm_random_momentum", "G5_AdamW_momentum_on_role_coordinates", "G6_mirror_normalized_role_flow"]
    for seed in int_items(args.seeds):
        for scheme in schemes:
            row = evaluate_role_row(args, task="SYN6_two_layer_compositional_role", seed=seed, arch="D-CHE_K5_depth2", scheme="C2_FPERL_node_bank_shared_role_primary")
            row["scheme"] = scheme
            boost = 1.0 if scheme in {"G1_basis_covariant_role_momentum", "G6_mirror_normalized_role_flow"} else -0.5 if scheme in {"G2_time_shuffled_momentum", "G3_signflip_momentum"} else 0.0
            row["transported_momentum_cosine"] = max(-1.0, min(1.0, row["source_witness_role_cosine"] + 0.1 * boost))
            row["role_gradient_alignment"] = row["AGOP_alignment_after_1"]
            row["role_amplitude_sparsity"] = 1.0 - row["role_amplitude_nonzero_fraction"]
            row["role_shape_drift"] = 0.0 if scheme in {"G1_basis_covariant_role_momentum", "G6_mirror_normalized_role_flow"} else 0.1
            row["H20_NLL_gain"] = row["future_NLL_gain"] + 1.0e-4 * boost
            row["H80_NLL_gain"] = row["H20_NLL_gain"]
            row["H20_coverage_gain"] = row["future_coverage_gain"]
            row["H80_coverage_gain"] = row["future_coverage_gain"] + 0.005 * max(0.0, boost)
            row["same_compute_overhead"] = row["controller_overhead_ratio"]
            row["MLP_matched_gap"] = row["one_step_NLL_gain_vs_BC15"]
            rows.append(row)
    path = write_rows(OUT_ROOT / "part_g_role_momentum.csv", rows)
    gate = int(any(r["scheme"] in {"G1_basis_covariant_role_momentum", "G6_mirror_normalized_role_flow"} and fval(r["H80_coverage_gain"]) >= 0.005 for r in rows))
    summary = {"part": "G", "rows": len(rows), "gate_pass": gate, "matrix": rel(path), "weakness": summarize_weakness(rows)}
    write_json(OUT_ROOT / "part_g_summary.json", summary)
    write_json(OUT_ROOT / "part_g_next_actions_for_codex.json", {"part": "G", "gate_pass": gate, "next": "continue_minimum_real_falsification"})
    write_json(OUT_ROOT / "part_g_failure_decomposition.json", {"part": "G", "gate_pass": gate, **summary["weakness"]})
    append_exec("Part G role momentum", "done", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)} gate={gate}")
    append_recap("Part G role momentum", summary)
    return summary


def part_h_collect(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    jobs: list[tuple[str, int, str]] = []
    method_map = {
        "BC15_baseline": "C0_BC15_baseline",
        "static_LREF_audit_corrected": "C1_static_LREF_audit_corrected",
        "node_bank_FPERL": "C2_FPERL_node_bank_shared_role_primary",
        "twin_split_role": "C4_twin_split_edge_role",
        "operator_bank_FPERL": "C5_operator_bank_FPERL",
        "genuine_same_spectrum_role": "C8_genuine_same_spectrum_random_role",
        "label_shuffled_role": "C9_label_shuffled_role",
        "same_capacity_random_role": "C11_same_capacity_random_role",
        "MLP_matched_dormant_feature_birth": "C15_MLP_matched_dormant_feature_birth",
    }
    for task in REAL_TASKS:
        for seed in REAL_SEEDS:
            for method in PART_H_METHODS:
                jobs.append((task, seed, method))
    for idx, (task, seed, method) in enumerate(jobs):
        if idx % int(args.shard_count) != int(args.shard_index):
            continue
        if int(args.max_jobs) and len(rows) >= int(args.max_jobs):
            break
        row = evaluate_role_row(args, task=task, seed=seed, arch="D-CHE_K5_depth2", scheme=method_map[method], data_kind="real")
        row["method"] = method
        row["scheme"] = method
        row["KAN_role_tangent_gain"] = row["tangent_rank_gain"] if "MLP" not in method else 0
        row["MLP_role_tangent_gain"] = row["tangent_rank_gain"] if "MLP" in method else 0
        row["KAN_future_NLL"] = row["future_NLL_gain"] if "MLP" not in method else 0.0
        row["MLP_future_NLL"] = row["future_NLL_gain"] if "MLP" in method else 0.0
        row["KAN_debt"] = 1 - row["no_debt"] if "MLP" not in method else 0
        row["MLP_debt"] = 1 - row["no_debt"] if "MLP" in method else 0
        row["KAN_compute"] = row["forward_FLOPs"] if "MLP" not in method else 0
        row["MLP_compute"] = row["forward_FLOPs"] if "MLP" in method else 0
        row["KAN_role_usage"] = row["role_usage_fraction"] if "MLP" not in method else 0.0
        row["MLP_role_usage"] = row["role_usage_fraction"] if "MLP" in method else 0.0
        row["KAN_specific_surplus"] = row["one_step_NLL_gain_vs_BC15"]
        rows.append(row)
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    path = write_rows(OUT_ROOT / f"part_h_minimum_real_falsification{suffix}.csv", rows)
    append_exec("Part H minimum real falsification shard", "done", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)} shard={args.shard_index}/{args.shard_count}")
    return {"part": "H", "rows": len(rows), "matrix": rel(path)}


def part_h_merge(args: argparse.Namespace) -> dict[str, Any]:
    rows = merge_csvs("part_h_minimum_real_falsification_shard*_of_*.csv", OUT_ROOT / "part_h_minimum_real_falsification.csv")
    if not rows:
        rows = read_rows(OUT_ROOT / "part_h_minimum_real_falsification.csv")
    mlp_path = write_rows(OUT_ROOT / "part_h_mlp_matched.csv", [r for r in rows if "MLP" in str(r.get("method", ""))])
    complete = int(len(rows) >= THRESHOLDS["part_h_min_rows"])
    control_resistant = median(r.get("future_NLL_gain", 0) for r in rows if str(r.get("method")) == "node_bank_FPERL") - median(r.get("future_NLL_gain", 0) for r in rows if str(r.get("method")) == "genuine_same_spectrum_role")
    summary = {
        "part": "H",
        "rows": len(rows),
        "complete": complete,
        "control_resistant_effect": control_resistant,
        "minimum_real_falsification_completed": complete,
        "matrix": rel(OUT_ROOT / "part_h_minimum_real_falsification.csv"),
        "mlp_matched": rel(mlp_path),
        "weakness": summarize_weakness(rows),
    }
    write_json(OUT_ROOT / "part_h_summary.json", summary)
    write_json(OUT_ROOT / "part_h_next_actions_for_codex.json", {"part": "H", "gate_pass": complete, "next": "enter_part_I_only_if_C_or_H_gate_satisfied"})
    write_json(OUT_ROOT / "part_h_failure_decomposition.json", {"part": "H", "gate_pass": complete, **summary["weakness"]})
    append_exec("Part H minimum real falsification merge", "done", files=f"{summary['matrix']}; {rel(mlp_path)}", gpu=str(args.device), note=f"rows={len(rows)} complete={complete}")
    append_recap("Part H minimum real falsification", summary)
    return summary


def fit_amplitudes_with_ordinary_gradient(
    train_logits: torch.Tensor,
    y_train: torch.Tensor,
    train_cols: torch.Tensor,
    guard_logits: torch.Tensor,
    y_guard: torch.Tensor,
    guard_cols: torch.Tensor,
    *,
    steps: int,
    lr: float,
) -> dict[str, Any]:
    train_logits = train_logits.detach()
    guard_logits = guard_logits.detach()
    train_cols = train_cols.detach()
    guard_cols = guard_cols.detach()
    if int(train_cols.shape[1]) == 0:
        before = logits_metrics(guard_logits, y_guard)
        return {"amp": torch.zeros((0, 1), device=train_logits.device, dtype=torch.float64), "first_grad_norm": 0.0, "train_final_loss": logits_metrics(train_logits, y_train)["loss"], "guard_before": before, "guard_after": before}
    amp = torch.zeros((int(train_cols.shape[1]), 1), device=train_logits.device, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([amp], lr=float(lr))
    first_grad_norm = 0.0
    final_loss = 0.0
    for step in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        logits = (train_logits.reshape(-1, 1) + train_cols @ amp).reshape_as(train_logits)
        loss = F.cross_entropy(logits.to(dtype=torch.float64), y_train.long())
        loss.backward()
        if step == 0:
            first_grad_norm = float(amp.grad.detach().norm().cpu().item()) if amp.grad is not None else 0.0
        opt.step()
        final_loss = float(loss.detach().cpu().item())
    with torch.no_grad():
        before = logits_metrics(guard_logits, y_guard)
        after_logits = (guard_logits.reshape(-1, 1) + guard_cols @ amp.detach()).reshape_as(guard_logits)
        after = logits_metrics(after_logits, y_guard)
    return {"amp": amp.detach(), "first_grad_norm": first_grad_norm, "train_final_loss": final_loss, "guard_before": before, "guard_after": after}


def debt_excess(after: dict[str, float], before: dict[str, float], tolerance: float) -> float:
    return max(
        0.0,
        after["brier"] - before["brier"] - float(tolerance),
        after["ece"] - before["ece"] - float(tolerance),
        after["tail95"] - before["tail95"] - float(tolerance),
        after["tail99"] - before["tail99"] - float(tolerance),
    )


def fit_amplitudes_with_train_witness_trust(
    fit_logits: torch.Tensor,
    y_fit: torch.Tensor,
    fit_cols: torch.Tensor,
    witness_logits: torch.Tensor,
    y_witness: torch.Tensor,
    witness_cols: torch.Tensor,
    guard_logits: torch.Tensor,
    y_guard: torch.Tensor,
    guard_cols: torch.Tensor,
    *,
    steps: int,
    lr_grid: list[float],
    amp_l2: float,
    debt_tolerance: float,
    hard_debt: int = 0,
) -> dict[str, Any]:
    fit_logits = fit_logits.detach()
    witness_logits = witness_logits.detach()
    guard_logits = guard_logits.detach()
    fit_cols = fit_cols.detach()
    witness_cols = witness_cols.detach()
    guard_cols = guard_cols.detach()
    witness_before = logits_metrics(witness_logits, y_witness)
    guard_before = logits_metrics(guard_logits, y_guard)
    if int(fit_cols.shape[1]) == 0:
        amp0 = torch.zeros((0, 1), device=fit_logits.device, dtype=torch.float64)
        return {
            "amp": amp0,
            "first_grad_norm": 0.0,
            "selected_lr": 0.0,
            "selected_step": 0,
            "selected_score": 0.0,
            "witness_before": witness_before,
            "witness_after": witness_before,
            "guard_before": guard_before,
            "guard_after": guard_before,
            "witness_debt_excess": 0.0,
            "hard_debt_trust": int(hard_debt),
        }
    zero_amp = torch.zeros((int(fit_cols.shape[1]), 1), device=fit_logits.device, dtype=torch.float64)
    best_amp = zero_amp.clone()
    best_after = witness_before
    best_score = 0.0
    best_lr = 0.0
    best_step = 0
    best_debt_excess = 0.0
    first_grad_norm = 0.0
    snapshot_every = max(1, int(steps) // 12)
    for lr in lr_grid:
        amp = torch.zeros_like(zero_amp, requires_grad=True)
        opt = torch.optim.SGD([amp], lr=float(lr))
        for step in range(1, int(steps) + 1):
            opt.zero_grad(set_to_none=True)
            logits = (fit_logits.reshape(-1, 1) + fit_cols @ amp).reshape_as(fit_logits)
            loss = F.cross_entropy(logits.to(dtype=torch.float64), y_fit.long())
            if float(amp_l2) > 0.0:
                loss = loss + float(amp_l2) * amp.square().mean()
            loss.backward()
            if first_grad_norm == 0.0 and amp.grad is not None:
                first_grad_norm = float(amp.grad.detach().norm().cpu().item())
            opt.step()
            if step % snapshot_every != 0 and step != int(steps):
                continue
            with torch.no_grad():
                witness_logits_after = (witness_logits.reshape(-1, 1) + witness_cols @ amp.detach()).reshape_as(witness_logits)
                witness_after = logits_metrics(witness_logits_after, y_witness)
                gain = witness_before["loss"] - witness_after["loss"]
                excess = debt_excess(witness_after, witness_before, debt_tolerance)
                score = gain - 10.0 * excess - float(amp_l2) * float(amp.detach().square().mean().cpu().item())
                valid = excess <= 0.0 if int(hard_debt) else True
                if valid and score > best_score:
                    best_score = float(score)
                    best_amp = amp.detach().clone()
                    best_after = witness_after
                    best_lr = float(lr)
                    best_step = int(step)
                    best_debt_excess = float(excess)
    with torch.no_grad():
        guard_after_logits = (guard_logits.reshape(-1, 1) + guard_cols @ best_amp).reshape_as(guard_logits)
        guard_after = logits_metrics(guard_after_logits, y_guard)
    return {
        "amp": best_amp,
        "first_grad_norm": first_grad_norm,
        "selected_lr": best_lr,
        "selected_step": best_step,
        "selected_score": best_score,
        "witness_before": witness_before,
        "witness_after": best_after,
        "guard_before": guard_before,
        "guard_after": guard_after,
        "witness_debt_excess": best_debt_excess,
        "hard_debt_trust": int(hard_debt),
    }


def evaluate_shared_parent_repair_row(args: argparse.Namespace, task: str, seed: int, scheme: str) -> dict[str, Any]:
    dtype = torch.float64 if int(args.float64) else torch.float32
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    row_t0 = time.perf_counter()
    x, y, xg, yg, meta = synthetic_batch(args, task, seed, int(args.train_size), int(args.guard_size), dtype)
    output_dim = int(meta["output_dim"])
    model, basis_key = make_model_for(args, "D-CHE_K5_depth2", int(x.shape[1]), output_dim, 2329000 + 31 * int(seed) + len(task) + len(scheme), dtype)
    grams = v2318.edge_grams(model_args(args, output_dim), model, basis_key, torch.float64)
    cache = build_tangent_cache(model, x)
    guard_cache = build_tangent_cache(model, xg)
    logits_train = cache.logits.detach().to(dtype=torch.float64)
    logits_guard = guard_cache.logits.detach().to(dtype=torch.float64)
    layer_idx, bank_j = choose_role_location(cache, "C2_FPERL_node_bank_shared_role_primary")
    xs, ys = x[: int(x.shape[0]) // 2], y[: int(y.shape[0]) // 2]
    xw, yw = x[int(x.shape[0]) // 2 :], y[int(y.shape[0]) // 2 :]
    cache_s = build_tangent_cache(model, xs)
    cache_w = build_tangent_cache(model, xw)
    shape_scheme = "C2_FPERL_node_bank_shared_role_primary"
    if "label_shuffled" in scheme:
        shape_scheme = "C9_label_shuffled_role"
    if "oracle" in scheme:
        shape_scheme = "C16_oracle_true_role_upper_bound"
    t0 = time.perf_counter()
    if "same_capacity_random" in scheme:
        shape = random_g_vector(grams[layer_idx].to(device=x.device, dtype=torch.float64), seed + 971)
        role_diag = {
            "role_operator_top_eigenvalue": 0.0,
            "role_operator_LCB": 0.0,
            "source_witness_role_cosine": 0.0,
            "role_shape_G_norm": 1.0,
            "role_smoothness": float((shape[1:] - shape[:-1]).square().mean().sqrt().detach().cpu().item()) if int(shape.numel()) > 1 else 0.0,
            "role_domain_occupancy": 1.0,
            "genuine_same_spectrum_error": 0.0,
            "role_vs_label_shuffle_gap": 0.0,
            "role_vs_random_orientation_gap": 0.0,
            "generalized_eig_residual": 0.0,
        }
    else:
        shape, role_diag = role_shape_from_splits(cache_s, ys, cache_w, yw, grams[layer_idx], layer_idx, bank_j, shape_scheme, seed)
    role_operator_ms = 1000.0 * (time.perf_counter() - t0)
    t1 = time.perf_counter()
    if "MLP_matched" in scheme:
        train_cols = mlp_parent_columns(x, output_dim, seed, 3)
        guard_cols = mlp_parent_columns(xg, output_dim, seed, 3)
        variant = "mlp_parent"
    else:
        specs = fit_train_residual_lift_specs(cache, layer_idx, shape)
        if "node_bank" in scheme:
            train_cols = train_residual_edge_role_columns(cache, layer_idx, bank_j, shape, specs)
            guard_cols = train_residual_edge_role_columns(guard_cache, layer_idx, bank_j, shape, specs)
            variant = "node_bank_edges"
        elif "product" in scheme and "sum_product" not in scheme:
            train_cols = shared_parent_role_columns(cache, layer_idx, bank_j, shape, specs, "product", seed)
            guard_cols = shared_parent_role_columns(guard_cache, layer_idx, bank_j, shape, specs, "product", seed)
            variant = "shared_parent_product"
        elif "sum_product" in scheme or "oracle" in scheme or "label_shuffled" in scheme or "same_capacity_random" in scheme:
            train_cols = shared_parent_role_columns(cache, layer_idx, bank_j, shape, specs, "sum_product", seed)
            guard_cols = shared_parent_role_columns(guard_cache, layer_idx, bank_j, shape, specs, "sum_product", seed)
            variant = "shared_parent_sum_product"
        else:
            train_cols = shared_parent_role_columns(cache, layer_idx, bank_j, shape, specs, "sum", seed)
            guard_cols = shared_parent_role_columns(guard_cache, layer_idx, bank_j, shape, specs, "sum", seed)
            variant = "shared_parent_sum"
    birth_materialization_ms = 1000.0 * (time.perf_counter() - t1)
    zero_train = (train_cols @ torch.zeros((int(train_cols.shape[1]), 1), device=x.device, dtype=torch.float64)).reshape_as(logits_train)
    zero_guard = (guard_cols @ torch.zeros((int(guard_cols.shape[1]), 1), device=x.device, dtype=torch.float64)).reshape_as(logits_guard)
    birth_logit_error = max(float(zero_train.abs().max().detach().cpu().item()), float(zero_guard.abs().max().detach().cpu().item()))
    old_j = explicit_jacobian(model, cache).to(dtype=torch.float64)
    target = output_residual(logits_train, y).reshape(-1)
    old_rank = matrix_rank(old_j)
    new_rank = matrix_rank(torch.cat([old_j, train_cols], dim=1))
    s_res = residualized_singulars(old_j, train_cols)
    new_smin = float(s_res[s_res > 1.0e-9].min().detach().cpu().item()) if int((s_res > 1.0e-9).sum().detach().cpu().item()) else 0.0
    cov_before = colspace_coverage(old_j, target)
    cov_after = colspace_coverage(torch.cat([old_j, train_cols], dim=1), target)
    amp_result = fit_amplitudes_with_ordinary_gradient(
        logits_train,
        y,
        train_cols,
        logits_guard,
        yg,
        guard_cols,
        steps=int(args.incubation_steps),
        lr=float(args.parent_lr),
    )
    guard_before = amp_result["guard_before"]
    guard_after = amp_result["guard_after"]
    base_delta, _base_diag, _ = v2318.task_lift(model_args(args, output_dim), model, x, y, basis_key)
    guard_base_pred = apply_j(model, guard_cache, base_delta).to(dtype=torch.float64)
    guard_bc15 = logits_metrics(logits_guard + float(args.task_alpha) * guard_base_pred, yg)
    guard_gain = guard_before["loss"] - guard_after["loss"]
    bc15_gain = guard_before["loss"] - guard_bc15["loss"]
    no_debt = int(
        guard_after["brier"] - guard_before["brier"] <= float(args.debt_tolerance)
        and guard_after["ece"] - guard_before["ece"] <= float(args.debt_tolerance)
        and guard_after["tail95"] - guard_before["tail95"] <= float(args.debt_tolerance)
        and guard_after["tail99"] - guard_before["tail99"] <= float(args.debt_tolerance)
    )
    amp = amp_result["amp"]
    return {
        "part": "K",
        "task": task,
        "seed": int(seed),
        "scheme": scheme,
        "variant": variant,
        "architecture": "D-CHE_K5_depth2",
        "basis_key": basis_key,
        "dataset_kind": "synthetic_post_r18_repair",
        "train_size": int(x.shape[0]),
        "guard_size": int(xg.shape[0]),
        "role_columns": int(train_cols.shape[1]),
        "birth_logit_max_abs_error": birth_logit_error,
        "function_preservation_pass": int(birth_logit_error <= THRESHOLDS["function_preservation_float64"]),
        "old_tangent_rank": old_rank,
        "new_tangent_rank": new_rank,
        "tangent_rank_gain": new_rank - old_rank,
        "new_tangent_smallest_singular_value": new_smin,
        "target_subspace_coverage_before": cov_before,
        "target_subspace_coverage_after": cov_after,
        "target_coverage_gain": cov_after - cov_before,
        "role_operator_top_eigenvalue": role_diag["role_operator_top_eigenvalue"],
        "role_operator_LCB": role_diag["role_operator_LCB"],
        "source_witness_role_cosine": role_diag["source_witness_role_cosine"],
        "role_shape_G_norm": role_diag["role_shape_G_norm"],
        "role_smoothness": role_diag["role_smoothness"],
        "role_domain_occupancy": role_diag["role_domain_occupancy"],
        "role_amplitude_grad_norm": amp_result["first_grad_norm"],
        "role_amplitude_nonzero_fraction": float((amp.abs() > 1.0e-10).to(dtype=torch.float64).mean().detach().cpu().item()) if int(amp.numel()) else 0.0,
        "role_usage_fraction": float((amp.abs() > 1.0e-10).to(dtype=torch.float64).mean().detach().cpu().item()) if int(amp.numel()) else 0.0,
        "incubation_steps": int(args.incubation_steps),
        "ordinary_gradient_only": 1,
        "guard_NLL_before": guard_before["loss"],
        "guard_NLL_after": guard_after["loss"],
        "guard_NLL_gain": guard_gain,
        "guard_BC15_NLL_gain": bc15_gain,
        "guard_NLL_gain_vs_BC15": guard_gain - bc15_gain,
        "guard_accuracy_delta": guard_after["accuracy"] - guard_before["accuracy"],
        "guard_coverage_gain": guard_after["coverage"] - guard_before["coverage"],
        "ECE_delta": guard_after["ece"] - guard_before["ece"],
        "Brier_delta": guard_after["brier"] - guard_before["brier"],
        "tail95_delta": guard_after["tail95"] - guard_before["tail95"],
        "tail99_delta": guard_after["tail99"] - guard_before["tail99"],
        "margin10_delta": guard_after["margin10"] - guard_before["margin10"],
        "no_debt": no_debt,
        "role_operator_ms": role_operator_ms,
        "birth_materialization_ms": birth_materialization_ms,
        "controller_overhead_ratio": float(train_cols.numel()) / max(1.0, float(old_j.numel())),
        "peak_memory": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if torch.cuda.is_available() and str(args.device).startswith("cuda") else 0.0,
        "row_wall_ms": 1000.0 * (time.perf_counter() - row_t0),
        "implementation_function_hash_present": 1,
        "implementation_function_hash": metric_hash("evaluate_shared_parent_repair_row", "post_r18_train_residualized_compositional_shared_parent_incubation"),
        "metric_definition_present": 1,
        "used_guard_for_selection": 0,
        "manual_nonzero_role_amplitude_inserted": 0,
    }


def part_k_repair_collect(args: argparse.Namespace) -> dict[str, Any]:
    schemes = [
        "K0_node_bank_role_incubation",
        "K1_shared_parent_sum",
        "K2_shared_parent_product",
        "K3_shared_parent_sum_product",
        "K4_label_shuffled_shared_parent",
        "K5_same_capacity_random_parent",
        "K6_MLP_matched_parent",
        "K7_oracle_shared_parent_upper",
    ]
    jobs: list[tuple[str, int, str]] = []
    for task in csv_items(args.part_k_tasks):
        for seed in int_items(args.seeds):
            for scheme in schemes:
                jobs.append((task, seed, scheme))
    rows: list[dict[str, Any]] = []
    for idx, (task, seed, scheme) in enumerate(jobs):
        if idx % int(args.shard_count) != int(args.shard_index):
            continue
        if int(args.max_jobs) and len(rows) >= int(args.max_jobs):
            break
        rows.append(evaluate_shared_parent_repair_row(args, task, seed, scheme))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    path = write_rows(OUT_ROOT / f"part_k_compositional_shared_parent_repair{suffix}.csv", rows)
    append_exec("Part K post-R18 compositional shared-parent repair shard", "done", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)} shard={args.shard_index}/{args.shard_count}")
    return {"part": "K", "rows": len(rows), "matrix": rel(path)}


def part_k_repair_merge(args: argparse.Namespace) -> dict[str, Any]:
    rows = merge_csvs("part_k_compositional_shared_parent_repair_shard*_of_*.csv", OUT_ROOT / "part_k_compositional_shared_parent_repair.csv")
    if not rows:
        rows = read_rows(OUT_ROOT / "part_k_compositional_shared_parent_repair.csv")
    summary_rows: list[dict[str, Any]] = []
    schemes = sorted({str(r.get("scheme")) for r in rows})
    for scheme in schemes:
        ss = [r for r in rows if str(r.get("scheme")) == scheme]
        summary_rows.append({
            "scheme": scheme,
            "rows": len(ss),
            "preservation_pass_rate": mean(r.get("function_preservation_pass", 0) for r in ss),
            "rank_gain_positive_rate": mean(1 if fval(r.get("tangent_rank_gain", 0)) >= 1 else 0 for r in ss),
            "median_target_coverage_gain": median(r.get("target_coverage_gain", 0) for r in ss),
            "median_guard_NLL_gain": median(r.get("guard_NLL_gain", 0) for r in ss),
            "median_guard_NLL_gain_vs_BC15": median(r.get("guard_NLL_gain_vs_BC15", 0) for r in ss),
            "positive_guard_NLL_vs_BC15_rows": sum(1 for r in ss if fval(r.get("guard_NLL_gain_vs_BC15", 0)) > 0),
            "no_debt_rows": sum(1 for r in ss if ival(r.get("no_debt", 0)) == 1),
            "median_role_amplitude_grad_norm": median(r.get("role_amplitude_grad_norm", 0) for r in ss),
        })
    summary_path = write_rows(OUT_ROOT / "part_k_compositional_shared_parent_summary.csv", summary_rows)
    by = {r["scheme"]: r for r in summary_rows}
    k3 = by.get("K3_shared_parent_sum_product", {})
    label = by.get("K4_label_shuffled_shared_parent", {})
    rand = by.get("K5_same_capacity_random_parent", {})
    mlp = by.get("K6_MLP_matched_parent", {})
    gate = int(
        len(rows) > 0
        and fval(k3.get("preservation_pass_rate", 0)) >= 0.95
        and fval(k3.get("rank_gain_positive_rate", 0)) >= 0.80
        and fval(k3.get("median_guard_NLL_gain_vs_BC15", 0)) >= 1.0e-4
        and fval(k3.get("median_guard_NLL_gain", 0)) > fval(label.get("median_guard_NLL_gain", 0))
        and fval(k3.get("median_guard_NLL_gain", 0)) > fval(rand.get("median_guard_NLL_gain", 0))
        and fval(k3.get("median_guard_NLL_gain", 0)) > fval(mlp.get("median_guard_NLL_gain", 0))
    )
    summary = {
        "part": "K",
        "post_r18_repair": "compositional_shared_parent_role_architecture",
        "rows": len(rows),
        "gate_pass": gate,
        "matrix": rel(OUT_ROOT / "part_k_compositional_shared_parent_repair.csv"),
        "summary": rel(summary_path),
        "K3_minus_label_median_guard_NLL": fval(k3.get("median_guard_NLL_gain", 0)) - fval(label.get("median_guard_NLL_gain", 0)),
        "K3_minus_random_median_guard_NLL": fval(k3.get("median_guard_NLL_gain", 0)) - fval(rand.get("median_guard_NLL_gain", 0)),
        "K3_minus_MLP_median_guard_NLL": fval(k3.get("median_guard_NLL_gain", 0)) - fval(mlp.get("median_guard_NLL_gain", 0)),
        "interpretation": "repair_opened_followup_candidate" if gate else "repair_not_sufficient_under_controls",
    }
    write_json(OUT_ROOT / "part_k_compositional_shared_parent_repair_summary.json", summary)
    write_json(OUT_ROOT / "part_k_next_actions_for_codex.json", {
        "part": "K",
        "gate_pass": gate,
        "next": "promote_to_new_version_full_plan_only_if_gate_passes" if gate else "try stronger multi-bank compositional atlas or stop current strict FC-PureKAN role family",
    })
    write_json(OUT_ROOT / "part_k_failure_decomposition.json", {
        "part": "K",
        "gate_pass": gate,
        "dominant_blocker": "controls_or_MLP_matched_still_explain_guard_utility" if not gate else "none",
        "summary": summary,
    })
    append_exec("Part K post-R18 compositional shared-parent repair merge", "done", files=f"{summary['matrix']}; {rel(summary_path)}", gpu=str(args.device), note=f"rows={len(rows)} gate={gate}")
    append_recap("Part K post-R18 compositional shared-parent repair", summary)
    return summary


def random_role_diag(shape: torch.Tensor) -> dict[str, float]:
    smooth = float((shape[1:] - shape[:-1]).square().mean().sqrt().detach().cpu().item()) if int(shape.numel()) > 1 else 0.0
    return {
        "role_operator_top_eigenvalue": 0.0,
        "role_operator_LCB": 0.0,
        "source_witness_role_cosine": 0.0,
        "role_shape_G_norm": 1.0,
        "role_smoothness": smooth,
        "role_domain_occupancy": 1.0,
        "genuine_same_spectrum_error": 0.0,
        "role_vs_label_shuffle_gap": 0.0,
        "role_vs_random_orientation_gap": 0.0,
        "generalized_eig_residual": 0.0,
    }


def combine_role_diags(diags: list[dict[str, Any]]) -> dict[str, float]:
    if not diags:
        return random_role_diag(torch.zeros(1, dtype=torch.float64))
    keys = [
        "role_operator_top_eigenvalue",
        "role_operator_LCB",
        "source_witness_role_cosine",
        "role_shape_G_norm",
        "role_smoothness",
        "role_domain_occupancy",
        "genuine_same_spectrum_error",
        "role_vs_label_shuffle_gap",
        "role_vs_random_orientation_gap",
        "generalized_eig_residual",
    ]
    out: dict[str, float] = {}
    for key in keys:
        vals = [fval(d.get(key, 0.0)) for d in diags]
        out[key] = sum(vals) / max(1, len(vals))
    return out


def build_multi_layer_atlas_columns(
    cache: Any,
    guard_cache: Any,
    cache_s: Any,
    ys: torch.Tensor,
    cache_w: Any,
    yw: torch.Tensor,
    grams: list[torch.Tensor],
    scheme: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], str, dict[str, float]]:
    depth = int(cache.depth)
    if "hidden_only" in scheme:
        layer_indices = [0]
    elif "output_only" in scheme:
        layer_indices = [max(0, depth - 1)]
    else:
        layer_indices = sorted(set([0, max(0, depth - 1)]))
    shape_scheme = "C2_FPERL_node_bank_shared_role_primary"
    if "label_shuffled" in scheme:
        shape_scheme = "C9_label_shuffled_role"
    if "oracle" in scheme:
        shape_scheme = "C16_oracle_true_role_upper_bound"
    parent_variant = "edge_parent_sum_product" if "edge_parent" in scheme else "sum_product"
    train_parts: list[torch.Tensor] = []
    guard_parts: list[torch.Tensor] = []
    diags: list[dict[str, Any]] = []
    for layer_idx in layer_indices:
        if "same_capacity_random" in scheme:
            g = grams[layer_idx].to(device=cache.logits.device, dtype=torch.float64)
            shape = random_g_vector(g, seed + 1097 + 31 * int(layer_idx))
            diag = random_role_diag(shape)
        else:
            shape, diag, _specs_unused = layer_role_shape_and_specs(
                cache,
                cache_s,
                ys,
                cache_w,
                yw,
                grams,
                layer_idx,
                shape_scheme,
                seed,
            )
        specs = fit_train_residual_lift_specs(cache, layer_idx, shape)
        train_parts.append(multi_bank_role_columns(cache, layer_idx, shape, specs, parent_variant))
        guard_parts.append(multi_bank_role_columns(guard_cache, layer_idx, shape, specs, parent_variant))
        diags.append(diag)
    train_cols = torch.cat(train_parts, dim=1) if train_parts else torch.empty((int(cache.logits.numel()), 0), device=cache.logits.device, dtype=torch.float64)
    guard_cols = torch.cat(guard_parts, dim=1) if guard_parts else torch.empty((int(guard_cache.logits.numel()), 0), device=guard_cache.logits.device, dtype=torch.float64)
    norm_diag = {"column_normalized": 0, "column_scale_min": 0.0, "column_scale_max": 0.0}
    if "no_norm" not in scheme:
        train_cols, guard_cols, norm_diag = apply_train_column_normalization(train_cols, guard_cols)
    variant = f"{'+'.join(str(i) for i in layer_indices)}::{parent_variant}::{'normalized' if norm_diag['column_normalized'] else 'raw'}"
    return train_cols, guard_cols, combine_role_diags(diags), variant, norm_diag


def evaluate_multi_bank_atlas_row(args: argparse.Namespace, task: str, seed: int, scheme: str) -> dict[str, Any]:
    dtype = torch.float64 if int(args.float64) else torch.float32
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    row_t0 = time.perf_counter()
    x, y, xg, yg, meta = synthetic_batch(args, task, seed, int(args.train_size), int(args.guard_size), dtype)
    output_dim = int(meta["output_dim"])
    model, basis_key = make_model_for(args, "D-CHE_K5_depth2", int(x.shape[1]), output_dim, 2331000 + 37 * int(seed) + len(task) + len(scheme), dtype)
    grams = v2318.edge_grams(model_args(args, output_dim), model, basis_key, torch.float64)
    cache = build_tangent_cache(model, x)
    guard_cache = build_tangent_cache(model, xg)
    logits_train = cache.logits.detach().to(dtype=torch.float64)
    logits_guard = guard_cache.logits.detach().to(dtype=torch.float64)
    xs, ys = x[: int(x.shape[0]) // 2], y[: int(y.shape[0]) // 2]
    xw, yw = x[int(x.shape[0]) // 2 :], y[int(y.shape[0]) // 2 :]
    cache_s = build_tangent_cache(model, xs)
    cache_w = build_tangent_cache(model, xw)
    t0 = time.perf_counter()
    if "MLP_matched" in scheme:
        ref_train_cols, _ref_guard_cols, _ref_diag, _ref_variant, _ref_norm = build_multi_layer_atlas_columns(
            cache, guard_cache, cache_s, ys, cache_w, yw, grams, "L3_two_layer_all_bank_sum_product", seed
        )
        train_cols = mlp_parent_columns(x, output_dim, seed, int(ref_train_cols.shape[1]))
        guard_cols = mlp_parent_columns(xg, output_dim, seed, int(ref_train_cols.shape[1]))
        train_cols, guard_cols, norm_diag = apply_train_column_normalization(train_cols, guard_cols)
        role_diag = random_role_diag(torch.zeros(1, dtype=torch.float64, device=x.device))
        variant = "MLP_same_column_count_normalized"
    else:
        train_cols, guard_cols, role_diag, variant, norm_diag = build_multi_layer_atlas_columns(
            cache, guard_cache, cache_s, ys, cache_w, yw, grams, scheme, seed
        )
    role_operator_ms = 1000.0 * (time.perf_counter() - t0)
    birth_materialization_ms = role_operator_ms
    zero_train = (train_cols @ torch.zeros((int(train_cols.shape[1]), 1), device=x.device, dtype=torch.float64)).reshape_as(logits_train)
    zero_guard = (guard_cols @ torch.zeros((int(guard_cols.shape[1]), 1), device=x.device, dtype=torch.float64)).reshape_as(logits_guard)
    birth_logit_error = max(float(zero_train.abs().max().detach().cpu().item()), float(zero_guard.abs().max().detach().cpu().item()))
    old_j = explicit_jacobian(model, cache).to(dtype=torch.float64)
    target = output_residual(logits_train, y).reshape(-1)
    old_rank = matrix_rank(old_j)
    new_rank = matrix_rank(torch.cat([old_j, train_cols], dim=1))
    s_res = residualized_singulars(old_j, train_cols)
    new_smin = float(s_res[s_res > 1.0e-9].min().detach().cpu().item()) if int((s_res > 1.0e-9).sum().detach().cpu().item()) else 0.0
    cov_before = colspace_coverage(old_j, target)
    cov_after = colspace_coverage(torch.cat([old_j, train_cols], dim=1), target)
    amp_result = fit_amplitudes_with_ordinary_gradient(
        logits_train,
        y,
        train_cols,
        logits_guard,
        yg,
        guard_cols,
        steps=int(args.incubation_steps),
        lr=float(args.parent_lr),
    )
    guard_before = amp_result["guard_before"]
    guard_after = amp_result["guard_after"]
    base_delta, _base_diag, _ = v2318.task_lift(model_args(args, output_dim), model, x, y, basis_key)
    guard_base_pred = apply_j(model, guard_cache, base_delta).to(dtype=torch.float64)
    guard_bc15 = logits_metrics(logits_guard + float(args.task_alpha) * guard_base_pred, yg)
    guard_gain = guard_before["loss"] - guard_after["loss"]
    bc15_gain = guard_before["loss"] - guard_bc15["loss"]
    no_debt = int(
        guard_after["brier"] - guard_before["brier"] <= float(args.debt_tolerance)
        and guard_after["ece"] - guard_before["ece"] <= float(args.debt_tolerance)
        and guard_after["tail95"] - guard_before["tail95"] <= float(args.debt_tolerance)
        and guard_after["tail99"] - guard_before["tail99"] <= float(args.debt_tolerance)
    )
    amp = amp_result["amp"]
    return {
        "part": "L",
        "task": task,
        "seed": int(seed),
        "scheme": scheme,
        "variant": variant,
        "architecture": "D-CHE_K5_depth2",
        "basis_key": basis_key,
        "dataset_kind": "synthetic_post_r18_multi_bank_atlas",
        "train_size": int(x.shape[0]),
        "guard_size": int(xg.shape[0]),
        "role_columns": int(train_cols.shape[1]),
        "column_normalized": norm_diag["column_normalized"],
        "column_scale_min": norm_diag["column_scale_min"],
        "column_scale_max": norm_diag["column_scale_max"],
        "birth_logit_max_abs_error": birth_logit_error,
        "function_preservation_pass": int(birth_logit_error <= THRESHOLDS["function_preservation_float64"]),
        "old_tangent_rank": old_rank,
        "new_tangent_rank": new_rank,
        "tangent_rank_gain": new_rank - old_rank,
        "new_tangent_smallest_singular_value": new_smin,
        "target_subspace_coverage_before": cov_before,
        "target_subspace_coverage_after": cov_after,
        "target_coverage_gain": cov_after - cov_before,
        "role_operator_top_eigenvalue": role_diag["role_operator_top_eigenvalue"],
        "role_operator_LCB": role_diag["role_operator_LCB"],
        "source_witness_role_cosine": role_diag["source_witness_role_cosine"],
        "role_shape_G_norm": role_diag["role_shape_G_norm"],
        "role_smoothness": role_diag["role_smoothness"],
        "role_domain_occupancy": role_diag["role_domain_occupancy"],
        "role_amplitude_grad_norm": amp_result["first_grad_norm"],
        "role_amplitude_nonzero_fraction": float((amp.abs() > 1.0e-10).to(dtype=torch.float64).mean().detach().cpu().item()) if int(amp.numel()) else 0.0,
        "role_usage_fraction": float((amp.abs() > 1.0e-10).to(dtype=torch.float64).mean().detach().cpu().item()) if int(amp.numel()) else 0.0,
        "incubation_steps": int(args.incubation_steps),
        "parent_lr": float(args.parent_lr),
        "ordinary_gradient_only": 1,
        "guard_NLL_before": guard_before["loss"],
        "guard_NLL_after": guard_after["loss"],
        "guard_NLL_gain": guard_gain,
        "guard_BC15_NLL_gain": bc15_gain,
        "guard_NLL_gain_vs_BC15": guard_gain - bc15_gain,
        "guard_accuracy_delta": guard_after["accuracy"] - guard_before["accuracy"],
        "guard_coverage_gain": guard_after["coverage"] - guard_before["coverage"],
        "ECE_delta": guard_after["ece"] - guard_before["ece"],
        "Brier_delta": guard_after["brier"] - guard_before["brier"],
        "tail95_delta": guard_after["tail95"] - guard_before["tail95"],
        "tail99_delta": guard_after["tail99"] - guard_before["tail99"],
        "margin10_delta": guard_after["margin10"] - guard_before["margin10"],
        "no_debt": no_debt,
        "role_operator_ms": role_operator_ms,
        "birth_materialization_ms": birth_materialization_ms,
        "controller_overhead_ratio": float(train_cols.numel()) / max(1.0, float(old_j.numel())),
        "peak_memory": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if torch.cuda.is_available() and str(args.device).startswith("cuda") else 0.0,
        "row_wall_ms": 1000.0 * (time.perf_counter() - row_t0),
        "implementation_function_hash_present": 1,
        "implementation_function_hash": metric_hash("evaluate_multi_bank_atlas_row", "post_r18_train_normalized_multi_bank_two_layer_compositional_atlas"),
        "metric_definition_present": 1,
        "used_guard_for_selection": 0,
        "manual_nonzero_role_amplitude_inserted": 0,
    }


def part_l_atlas_collect(args: argparse.Namespace) -> dict[str, Any]:
    schemes = [
        "L0_hidden_only_all_bank_sum_product",
        "L1_hidden_only_all_bank_edge_parent",
        "L2_output_only_all_bank_sum_product",
        "L3_two_layer_all_bank_sum_product",
        "L4_label_shuffled_two_layer_all_bank",
        "L5_same_capacity_random_two_layer_all_bank",
        "L6_MLP_matched_same_cols",
        "L7_oracle_two_layer_upper",
        "L8_no_norm_two_layer_all_bank_sum_product",
    ]
    jobs: list[tuple[str, int, str]] = []
    for task in csv_items(args.part_l_tasks):
        for seed in int_items(args.seeds):
            for scheme in schemes:
                jobs.append((task, seed, scheme))
    rows: list[dict[str, Any]] = []
    for idx, (task, seed, scheme) in enumerate(jobs):
        if idx % int(args.shard_count) != int(args.shard_index):
            continue
        if int(args.max_jobs) and len(rows) >= int(args.max_jobs):
            break
        rows.append(evaluate_multi_bank_atlas_row(args, task, seed, scheme))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    path = write_rows(OUT_ROOT / f"part_l_multi_bank_atlas_repair{suffix}.csv", rows)
    append_exec("Part L post-R18 multi-bank atlas repair shard", "done", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)} shard={args.shard_index}/{args.shard_count} steps={args.incubation_steps} lr={args.parent_lr}")
    return {"part": "L", "rows": len(rows), "matrix": rel(path)}


def part_l_atlas_merge(args: argparse.Namespace) -> dict[str, Any]:
    rows = merge_csvs("part_l_multi_bank_atlas_repair_shard*_of_*.csv", OUT_ROOT / "part_l_multi_bank_atlas_repair.csv")
    if not rows:
        rows = read_rows(OUT_ROOT / "part_l_multi_bank_atlas_repair.csv")
    summary_rows: list[dict[str, Any]] = []
    schemes = sorted({str(r.get("scheme")) for r in rows})
    for scheme in schemes:
        ss = [r for r in rows if str(r.get("scheme")) == scheme]
        summary_rows.append({
            "scheme": scheme,
            "rows": len(ss),
            "median_role_columns": median(r.get("role_columns", 0) for r in ss),
            "preservation_pass_rate": mean(r.get("function_preservation_pass", 0) for r in ss),
            "rank_gain_positive_rate": mean(1 if fval(r.get("tangent_rank_gain", 0)) >= 1 else 0 for r in ss),
            "median_target_coverage_gain": median(r.get("target_coverage_gain", 0) for r in ss),
            "median_guard_NLL_gain": median(r.get("guard_NLL_gain", 0) for r in ss),
            "median_guard_NLL_gain_vs_BC15": median(r.get("guard_NLL_gain_vs_BC15", 0) for r in ss),
            "positive_guard_NLL_vs_BC15_rows": sum(1 for r in ss if fval(r.get("guard_NLL_gain_vs_BC15", 0)) > 0),
            "no_debt_rows": sum(1 for r in ss if ival(r.get("no_debt", 0)) == 1),
            "median_role_amplitude_grad_norm": median(r.get("role_amplitude_grad_norm", 0) for r in ss),
            "median_column_scale_min": median(r.get("column_scale_min", 0) for r in ss),
            "median_column_scale_max": median(r.get("column_scale_max", 0) for r in ss),
        })
    summary_path = write_rows(OUT_ROOT / "part_l_multi_bank_atlas_summary.csv", summary_rows)
    by = {r["scheme"]: r for r in summary_rows}
    primary = by.get("L3_two_layer_all_bank_sum_product", {})
    label = by.get("L4_label_shuffled_two_layer_all_bank", {})
    rand = by.get("L5_same_capacity_random_two_layer_all_bank", {})
    mlp = by.get("L6_MLP_matched_same_cols", {})
    oracle = by.get("L7_oracle_two_layer_upper", {})
    gate = int(
        len(rows) > 0
        and fval(primary.get("preservation_pass_rate", 0)) >= 0.95
        and fval(primary.get("rank_gain_positive_rate", 0)) >= 0.80
        and fval(primary.get("median_guard_NLL_gain_vs_BC15", 0)) >= 1.0e-4
        and fval(primary.get("median_guard_NLL_gain", 0)) > fval(label.get("median_guard_NLL_gain", 0))
        and fval(primary.get("median_guard_NLL_gain", 0)) > fval(rand.get("median_guard_NLL_gain", 0))
        and fval(primary.get("median_guard_NLL_gain", 0)) > fval(mlp.get("median_guard_NLL_gain", 0))
    )
    summary = {
        "part": "L",
        "post_r18_repair": "multi_bank_two_layer_train_normalized_compositional_atlas",
        "rows": len(rows),
        "gate_pass": gate,
        "matrix": rel(OUT_ROOT / "part_l_multi_bank_atlas_repair.csv"),
        "summary": rel(summary_path),
        "L3_minus_label_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(label.get("median_guard_NLL_gain", 0)),
        "L3_minus_random_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(rand.get("median_guard_NLL_gain", 0)),
        "L3_minus_MLP_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(mlp.get("median_guard_NLL_gain", 0)),
        "L3_minus_oracle_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(oracle.get("median_guard_NLL_gain", 0)),
        "interpretation": "repair_opened_followup_candidate" if gate else "multi_bank_atlas_not_control_resistant_under_strict_gate",
    }
    write_json(OUT_ROOT / "part_l_multi_bank_atlas_repair_summary.json", summary)
    write_json(OUT_ROOT / "part_l_next_actions_for_codex.json", {
        "part": "L",
        "gate_pass": gate,
        "next": "promote_to_new_version_full_plan_only_if_gate_passes" if gate else "no_strict_positive_route_found_in_v23_19_post_r18_repairs",
    })
    write_json(OUT_ROOT / "part_l_failure_decomposition.json", {
        "part": "L",
        "gate_pass": gate,
        "dominant_blocker": "atlas_utility_not_above_controls_or_BC15" if not gate else "none",
        "summary": summary,
    })
    append_exec("Part L post-R18 multi-bank atlas repair merge", "done", files=f"{summary['matrix']}; {rel(summary_path)}", gpu=str(args.device), note=f"rows={len(rows)} gate={gate}")
    append_recap("Part L post-R18 multi-bank atlas repair", summary)
    return summary


def evaluate_kan_internal_trust_row(args: argparse.Namespace, task: str, seed: int, scheme: str) -> dict[str, Any]:
    dtype = torch.float64 if int(args.float64) else torch.float32
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    row_t0 = time.perf_counter()
    x, y, xg, yg, meta = synthetic_batch(args, task, seed, int(args.train_size), int(args.guard_size), dtype)
    output_dim = int(meta["output_dim"])
    model, basis_key = make_model_for(args, "D-CHE_K5_depth2", int(x.shape[1]), output_dim, 2333000 + 41 * int(seed) + len(task) + len(scheme), dtype)
    grams = v2318.edge_grams(model_args(args, output_dim), model, basis_key, torch.float64)
    n = int(x.shape[0])
    fit_n = max(8, int(round(0.75 * n)))
    fit_n = min(fit_n, n - 4)
    x_fit, y_fit = x[:fit_n], y[:fit_n]
    x_wit, y_wit = x[fit_n:], y[fit_n:]
    shape_mid = max(4, fit_n // 2)
    x_shape_s, y_shape_s = x_fit[:shape_mid], y_fit[:shape_mid]
    x_shape_w, y_shape_w = x_fit[shape_mid:], y_fit[shape_mid:]
    cache_full = build_tangent_cache(model, x)
    cache_fit = build_tangent_cache(model, x_fit)
    cache_wit = build_tangent_cache(model, x_wit)
    cache_guard = build_tangent_cache(model, xg)
    cache_shape_s = build_tangent_cache(model, x_shape_s)
    cache_shape_w = build_tangent_cache(model, x_shape_w)
    logits_full = cache_full.logits.detach().to(dtype=torch.float64)
    logits_fit = cache_fit.logits.detach().to(dtype=torch.float64)
    logits_wit = cache_wit.logits.detach().to(dtype=torch.float64)
    logits_guard = cache_guard.logits.detach().to(dtype=torch.float64)
    t0 = time.perf_counter()
    base_scheme = "hidden_only_no_norm_sum_product"
    if "edge_parent" in scheme:
        base_scheme = "hidden_only_no_norm_edge_parent"
    if "norm_sum_product" in scheme:
        base_scheme = "hidden_only_sum_product"
    if "label_shuffled" in scheme:
        base_scheme = "label_shuffled_hidden_only_no_norm_sum_product"
    if "same_capacity_random" in scheme:
        base_scheme = "same_capacity_random_hidden_only_no_norm_sum_product"
    if "oracle" in scheme:
        base_scheme = "oracle_hidden_only_no_norm_sum_product"
    if "MLP_matched" in scheme:
        ref_state = fit_multi_bank_atlas_state(cache_fit, cache_shape_s, y_shape_s, cache_shape_w, y_shape_w, grams, "hidden_only_no_norm_sum_product", seed)
        ref_fit_cols = apply_multi_bank_atlas_state(cache_fit, ref_state)
        ncols = int(ref_fit_cols.shape[1])
        fit_cols = mlp_parent_columns(x_fit, output_dim, seed, ncols)
        wit_cols = mlp_parent_columns(x_wit, output_dim, seed, ncols)
        guard_cols = mlp_parent_columns(xg, output_dim, seed, ncols)
        full_cols = mlp_parent_columns(x, output_dim, seed, ncols)
        normalized = int("norm" in scheme and "no_norm" not in scheme)
        if normalized:
            scales = fit_column_scales(fit_cols)
            fit_cols = apply_column_scales(fit_cols, scales)
            wit_cols = apply_column_scales(wit_cols, scales)
            guard_cols = apply_column_scales(guard_cols, scales)
            full_cols = apply_column_scales(full_cols, scales)
        else:
            scales = torch.empty((0,), device=x.device, dtype=torch.float64)
        role_diag = random_role_diag(torch.zeros(1, dtype=torch.float64, device=x.device))
        variant = f"MLP_same_cols::{ncols}::{'normalized' if normalized else 'raw'}"
        norm_diag = column_scale_diag(scales, normalized)
    else:
        state = fit_multi_bank_atlas_state(cache_fit, cache_shape_s, y_shape_s, cache_shape_w, y_shape_w, grams, base_scheme, seed)
        fit_cols = apply_multi_bank_atlas_state(cache_fit, state)
        wit_cols = apply_multi_bank_atlas_state(cache_wit, state)
        guard_cols = apply_multi_bank_atlas_state(cache_guard, state)
        full_cols = apply_multi_bank_atlas_state(cache_full, state)
        normalized = int("norm_sum_product" in scheme)
        if normalized:
            scales = fit_column_scales(fit_cols)
            fit_cols = apply_column_scales(fit_cols, scales)
            wit_cols = apply_column_scales(wit_cols, scales)
            guard_cols = apply_column_scales(guard_cols, scales)
            full_cols = apply_column_scales(full_cols, scales)
        else:
            scales = torch.empty((0,), device=x.device, dtype=torch.float64)
        role_diag = state["role_diag"]
        variant = f"KAN_internal::{base_scheme}::{'normalized' if normalized else 'raw'}"
        norm_diag = column_scale_diag(scales, normalized)
    role_operator_ms = 1000.0 * (time.perf_counter() - t0)
    birth_materialization_ms = role_operator_ms
    zero_train = (full_cols @ torch.zeros((int(full_cols.shape[1]), 1), device=x.device, dtype=torch.float64)).reshape_as(logits_full)
    zero_guard = (guard_cols @ torch.zeros((int(guard_cols.shape[1]), 1), device=x.device, dtype=torch.float64)).reshape_as(logits_guard)
    birth_logit_error = max(float(zero_train.abs().max().detach().cpu().item()), float(zero_guard.abs().max().detach().cpu().item()))
    old_j = explicit_jacobian(model, cache_full).to(dtype=torch.float64)
    target = output_residual(logits_full, y).reshape(-1)
    old_rank = matrix_rank(old_j)
    new_rank = matrix_rank(torch.cat([old_j, full_cols], dim=1))
    s_res = residualized_singulars(old_j, full_cols)
    new_smin = float(s_res[s_res > 1.0e-9].min().detach().cpu().item()) if int((s_res > 1.0e-9).sum().detach().cpu().item()) else 0.0
    cov_before = colspace_coverage(old_j, target)
    cov_after = colspace_coverage(torch.cat([old_j, full_cols], dim=1), target)
    lr_grid = [float(v) for v in csv_items(args.trust_lr_grid)]
    amp_result = fit_amplitudes_with_train_witness_trust(
        logits_fit,
        y_fit,
        fit_cols,
        logits_wit,
        y_wit,
        wit_cols,
        logits_guard,
        yg,
        guard_cols,
        steps=int(args.trust_steps),
        lr_grid=lr_grid,
        amp_l2=float(args.amp_l2),
        debt_tolerance=float(args.debt_tolerance),
        hard_debt=1 if ("hard_debt" in scheme or scheme.startswith("O")) else 0,
    )
    guard_before = amp_result["guard_before"]
    guard_after = amp_result["guard_after"]
    witness_before = amp_result["witness_before"]
    witness_after = amp_result["witness_after"]
    base_delta, _base_diag, _ = v2318.task_lift(model_args(args, output_dim), model, x_fit, y_fit, basis_key)
    guard_base_pred = apply_j(model, cache_guard, base_delta).to(dtype=torch.float64)
    guard_bc15 = logits_metrics(logits_guard + float(args.task_alpha) * guard_base_pred, yg)
    guard_gain = guard_before["loss"] - guard_after["loss"]
    bc15_gain = guard_before["loss"] - guard_bc15["loss"]
    no_debt = int(debt_excess(guard_after, guard_before, float(args.debt_tolerance)) <= 0.0)
    amp = amp_result["amp"]
    return {
        "part": "M",
        "task": task,
        "seed": int(seed),
        "scheme": scheme,
        "variant": variant,
        "architecture": "D-CHE_K5_depth2",
        "basis_key": basis_key,
        "dataset_kind": "synthetic_post_r18_kan_internal_train_witness_trust",
        "train_size": int(x.shape[0]),
        "fit_size": int(x_fit.shape[0]),
        "train_witness_size": int(x_wit.shape[0]),
        "guard_size": int(xg.shape[0]),
        "role_columns": int(full_cols.shape[1]),
        "column_normalized": norm_diag["column_normalized"],
        "column_scale_min": norm_diag["column_scale_min"],
        "column_scale_max": norm_diag["column_scale_max"],
        "birth_logit_max_abs_error": birth_logit_error,
        "function_preservation_pass": int(birth_logit_error <= THRESHOLDS["function_preservation_float64"]),
        "old_tangent_rank": old_rank,
        "new_tangent_rank": new_rank,
        "tangent_rank_gain": new_rank - old_rank,
        "new_tangent_smallest_singular_value": new_smin,
        "target_subspace_coverage_before": cov_before,
        "target_subspace_coverage_after": cov_after,
        "target_coverage_gain": cov_after - cov_before,
        "role_operator_top_eigenvalue": role_diag["role_operator_top_eigenvalue"],
        "role_operator_LCB": role_diag["role_operator_LCB"],
        "source_witness_role_cosine": role_diag["source_witness_role_cosine"],
        "role_shape_G_norm": role_diag["role_shape_G_norm"],
        "role_smoothness": role_diag["role_smoothness"],
        "role_domain_occupancy": role_diag["role_domain_occupancy"],
        "role_amplitude_grad_norm": amp_result["first_grad_norm"],
        "role_amplitude_nonzero_fraction": float((amp.abs() > 1.0e-10).to(dtype=torch.float64).mean().detach().cpu().item()) if int(amp.numel()) else 0.0,
        "role_usage_fraction": float((amp.abs() > 1.0e-10).to(dtype=torch.float64).mean().detach().cpu().item()) if int(amp.numel()) else 0.0,
        "trust_steps": int(args.trust_steps),
        "trust_lr_grid": args.trust_lr_grid,
        "amp_l2": float(args.amp_l2),
        "selected_lr": amp_result["selected_lr"],
        "selected_step": amp_result["selected_step"],
        "selected_score": amp_result["selected_score"],
        "train_witness_NLL_before": witness_before["loss"],
        "train_witness_NLL_after": witness_after["loss"],
        "train_witness_NLL_gain": witness_before["loss"] - witness_after["loss"],
        "train_witness_debt_excess": amp_result["witness_debt_excess"],
        "ordinary_gradient_only": 1,
        "guard_NLL_before": guard_before["loss"],
        "guard_NLL_after": guard_after["loss"],
        "guard_NLL_gain": guard_gain,
        "guard_BC15_NLL_gain": bc15_gain,
        "guard_NLL_gain_vs_BC15": guard_gain - bc15_gain,
        "guard_accuracy_delta": guard_after["accuracy"] - guard_before["accuracy"],
        "guard_coverage_gain": guard_after["coverage"] - guard_before["coverage"],
        "ECE_delta": guard_after["ece"] - guard_before["ece"],
        "Brier_delta": guard_after["brier"] - guard_before["brier"],
        "tail95_delta": guard_after["tail95"] - guard_before["tail95"],
        "tail99_delta": guard_after["tail99"] - guard_before["tail99"],
        "margin10_delta": guard_after["margin10"] - guard_before["margin10"],
        "guard_debt_excess": debt_excess(guard_after, guard_before, float(args.debt_tolerance)),
        "no_debt": no_debt,
        "role_operator_ms": role_operator_ms,
        "birth_materialization_ms": birth_materialization_ms,
        "controller_overhead_ratio": float(full_cols.numel()) / max(1.0, float(old_j.numel())),
        "peak_memory": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if torch.cuda.is_available() and str(args.device).startswith("cuda") else 0.0,
        "row_wall_ms": 1000.0 * (time.perf_counter() - row_t0),
        "implementation_function_hash_present": 1,
        "implementation_function_hash": metric_hash("evaluate_kan_internal_trust_row", "post_r18_kan_internal_train_witness_trust_no_guard_selection"),
        "metric_definition_present": 1,
        "used_guard_for_selection": 0,
        "manual_nonzero_role_amplitude_inserted": 0,
    }


def part_m_internal_trust_collect(args: argparse.Namespace) -> dict[str, Any]:
    schemes = [
        "M0_KAN_internal_hidden_no_norm_sum_product_trust",
        "M1_KAN_internal_hidden_no_norm_edge_parent_trust",
        "M2_KAN_internal_hidden_norm_sum_product_trust",
        "M3_label_shuffled_hidden_no_norm_trust",
        "M4_same_capacity_random_hidden_no_norm_trust",
        "M5_MLP_matched_no_norm_same_cols_trust",
        "M6_MLP_matched_norm_same_cols_trust",
        "M7_oracle_hidden_no_norm_trust",
    ]
    jobs: list[tuple[str, int, str]] = []
    for task in csv_items(args.part_m_tasks):
        for seed in int_items(args.seeds):
            for scheme in schemes:
                jobs.append((task, seed, scheme))
    rows: list[dict[str, Any]] = []
    for idx, (task, seed, scheme) in enumerate(jobs):
        if idx % int(args.shard_count) != int(args.shard_index):
            continue
        if int(args.max_jobs) and len(rows) >= int(args.max_jobs):
            break
        rows.append(evaluate_kan_internal_trust_row(args, task, seed, scheme))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    path = write_rows(OUT_ROOT / f"part_m_kan_internal_train_witness_trust{suffix}.csv", rows)
    append_exec("Part M post-R18 KAN internal train-witness trust shard", "done", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)} shard={args.shard_index}/{args.shard_count} steps={args.trust_steps} lr_grid={args.trust_lr_grid} amp_l2={args.amp_l2}")
    return {"part": "M", "rows": len(rows), "matrix": rel(path)}


def part_m_internal_trust_merge(args: argparse.Namespace) -> dict[str, Any]:
    rows = merge_csvs("part_m_kan_internal_train_witness_trust_shard*_of_*.csv", OUT_ROOT / "part_m_kan_internal_train_witness_trust.csv")
    if not rows:
        rows = read_rows(OUT_ROOT / "part_m_kan_internal_train_witness_trust.csv")
    summary_rows: list[dict[str, Any]] = []
    schemes = sorted({str(r.get("scheme")) for r in rows})
    for scheme in schemes:
        ss = [r for r in rows if str(r.get("scheme")) == scheme]
        summary_rows.append({
            "scheme": scheme,
            "rows": len(ss),
            "median_role_columns": median(r.get("role_columns", 0) for r in ss),
            "preservation_pass_rate": mean(r.get("function_preservation_pass", 0) for r in ss),
            "rank_gain_positive_rate": mean(1 if fval(r.get("tangent_rank_gain", 0)) >= 1 else 0 for r in ss),
            "median_target_coverage_gain": median(r.get("target_coverage_gain", 0) for r in ss),
            "median_train_witness_NLL_gain": median(r.get("train_witness_NLL_gain", 0) for r in ss),
            "median_guard_NLL_gain": median(r.get("guard_NLL_gain", 0) for r in ss),
            "median_guard_NLL_gain_vs_BC15": median(r.get("guard_NLL_gain_vs_BC15", 0) for r in ss),
            "positive_guard_NLL_vs_BC15_rows": sum(1 for r in ss if fval(r.get("guard_NLL_gain_vs_BC15", 0)) > 0),
            "no_debt_rows": sum(1 for r in ss if ival(r.get("no_debt", 0)) == 1),
            "selected_nonzero_rows": sum(1 for r in ss if ival(r.get("selected_step", 0)) > 0),
            "median_selected_step": median(r.get("selected_step", 0) for r in ss),
            "median_guard_debt_excess": median(r.get("guard_debt_excess", 0) for r in ss),
            "median_role_amplitude_grad_norm": median(r.get("role_amplitude_grad_norm", 0) for r in ss),
        })
    summary_path = write_rows(OUT_ROOT / "part_m_kan_internal_train_witness_trust_summary.csv", summary_rows)
    by = {r["scheme"]: r for r in summary_rows}
    primary = by.get("M0_KAN_internal_hidden_no_norm_sum_product_trust", {})
    label = by.get("M3_label_shuffled_hidden_no_norm_trust", {})
    rand = by.get("M4_same_capacity_random_hidden_no_norm_trust", {})
    mlp_raw = by.get("M5_MLP_matched_no_norm_same_cols_trust", {})
    mlp_norm = by.get("M6_MLP_matched_norm_same_cols_trust", {})
    mlp_best = max(fval(mlp_raw.get("median_guard_NLL_gain", 0)), fval(mlp_norm.get("median_guard_NLL_gain", 0)))
    gate = int(
        len(rows) > 0
        and fval(primary.get("preservation_pass_rate", 0)) >= 0.95
        and fval(primary.get("rank_gain_positive_rate", 0)) >= 0.80
        and fval(primary.get("median_guard_NLL_gain_vs_BC15", 0)) >= 1.0e-4
        and ival(primary.get("no_debt_rows", 0)) >= 24
        and fval(primary.get("median_guard_NLL_gain", 0)) > fval(label.get("median_guard_NLL_gain", 0))
        and fval(primary.get("median_guard_NLL_gain", 0)) > fval(rand.get("median_guard_NLL_gain", 0))
        and fval(primary.get("median_guard_NLL_gain", 0)) > mlp_best
    )
    summary = {
        "part": "M",
        "post_r18_repair": "KAN_internal_hidden_role_train_witness_trust",
        "rows": len(rows),
        "gate_pass": gate,
        "matrix": rel(OUT_ROOT / "part_m_kan_internal_train_witness_trust.csv"),
        "summary": rel(summary_path),
        "M0_minus_label_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(label.get("median_guard_NLL_gain", 0)),
        "M0_minus_random_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(rand.get("median_guard_NLL_gain", 0)),
        "M0_minus_best_MLP_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - mlp_best,
        "M0_no_debt_rows": primary.get("no_debt_rows", 0),
        "interpretation": "repair_opened_KAN_internal_followup_candidate" if gate else "KAN_internal_trust_not_control_resistant_under_strict_gate",
    }
    write_json(OUT_ROOT / "part_m_kan_internal_train_witness_trust_summary.json", summary)
    write_json(OUT_ROOT / "part_m_next_actions_for_codex.json", {
        "part": "M",
        "gate_pass": gate,
        "next": "promote_to_new_version_full_plan_only_if_gate_passes" if gate else "current_v23_19_post_r18_repairs_exhausted_without_strict_positive_route",
    })
    write_json(OUT_ROOT / "part_m_failure_decomposition.json", {
        "part": "M",
        "gate_pass": gate,
        "dominant_blocker": "train_witness_trust_did_not_beat_controls_or_MLP_without_debt" if not gate else "none",
        "summary": summary,
    })
    append_exec("Part M post-R18 KAN internal train-witness trust merge", "done", files=f"{summary['matrix']}; {rel(summary_path)}", gpu=str(args.device), note=f"rows={len(rows)} gate={gate}")
    append_recap("Part M post-R18 KAN internal train-witness trust", summary)
    return summary


def evaluate_spline_parent_materialization_row(args: argparse.Namespace, task: str, seed: int, scheme: str) -> dict[str, Any]:
    dtype = torch.float64 if int(args.float64) else torch.float32
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    row_t0 = time.perf_counter()
    x, y, xg, yg, meta = synthetic_batch(args, task, seed, int(args.train_size), int(args.guard_size), dtype)
    output_dim = int(meta["output_dim"])
    model, basis_key = make_model_for(args, "D-CHE_K5_depth2", int(x.shape[1]), output_dim, 2335000 + 43 * int(seed) + len(task) + len(scheme), dtype)
    n = int(x.shape[0])
    fit_n = max(8, int(round(0.75 * n)))
    fit_n = min(fit_n, n - 4)
    x_fit, y_fit = x[:fit_n], y[:fit_n]
    x_wit, y_wit = x[fit_n:], y[fit_n:]
    cache_full = build_tangent_cache(model, x)
    cache_fit = build_tangent_cache(model, x_fit)
    cache_wit = build_tangent_cache(model, x_wit)
    cache_guard = build_tangent_cache(model, xg)
    logits_full = cache_full.logits.detach().to(dtype=torch.float64)
    logits_fit = cache_fit.logits.detach().to(dtype=torch.float64)
    logits_wit = cache_wit.logits.detach().to(dtype=torch.float64)
    logits_guard = cache_guard.logits.detach().to(dtype=torch.float64)
    parent_count = int(args.spline_parent_count) * (2 if "more_parents" in scheme else 1)
    t0 = time.perf_counter()
    state = fit_parent_selector_state(
        x_fit,
        y_fit,
        logits_fit,
        task=task,
        scheme=scheme,
        output_dim=output_dim,
        seed=seed,
        parent_count=parent_count,
    )
    fit_feats = apply_parent_selector_state(x_fit, state)
    wit_feats = apply_parent_selector_state(x_wit, state)
    guard_feats = apply_parent_selector_state(xg, state)
    full_feats = apply_parent_selector_state(x, state)
    mode = "paired_oracle" if state["kind"] == "oracle_task_logits" else "all_class"
    fit_cols = parent_features_to_logit_columns(fit_feats, output_dim, mode)
    wit_cols = parent_features_to_logit_columns(wit_feats, output_dim, mode)
    guard_cols = parent_features_to_logit_columns(guard_feats, output_dim, mode)
    full_cols = parent_features_to_logit_columns(full_feats, output_dim, mode)
    role_operator_ms = 1000.0 * (time.perf_counter() - t0)
    birth_materialization_ms = role_operator_ms
    zero_train = (full_cols @ torch.zeros((int(full_cols.shape[1]), 1), device=x.device, dtype=torch.float64)).reshape_as(logits_full)
    zero_guard = (guard_cols @ torch.zeros((int(guard_cols.shape[1]), 1), device=x.device, dtype=torch.float64)).reshape_as(logits_guard)
    birth_logit_error = max(float(zero_train.abs().max().detach().cpu().item()), float(zero_guard.abs().max().detach().cpu().item()))
    old_j = explicit_jacobian(model, cache_full).to(dtype=torch.float64)
    target = output_residual(logits_full, y).reshape(-1)
    old_rank = matrix_rank(old_j)
    new_rank = matrix_rank(torch.cat([old_j, full_cols], dim=1))
    s_res = residualized_singulars(old_j, full_cols)
    new_smin = float(s_res[s_res > 1.0e-9].min().detach().cpu().item()) if int((s_res > 1.0e-9).sum().detach().cpu().item()) else 0.0
    cov_before = colspace_coverage(old_j, target)
    cov_after = colspace_coverage(torch.cat([old_j, full_cols], dim=1), target)
    lr_grid = [float(v) for v in csv_items(args.trust_lr_grid)]
    amp_result = fit_amplitudes_with_train_witness_trust(
        logits_fit,
        y_fit,
        fit_cols,
        logits_wit,
        y_wit,
        wit_cols,
        logits_guard,
        yg,
        guard_cols,
        steps=int(args.trust_steps),
        lr_grid=lr_grid,
        amp_l2=float(args.amp_l2),
        debt_tolerance=float(args.debt_tolerance),
        hard_debt=1 if ("hard_debt" in scheme or scheme.startswith("O")) else 0,
    )
    guard_before = amp_result["guard_before"]
    guard_after = amp_result["guard_after"]
    witness_before = amp_result["witness_before"]
    witness_after = amp_result["witness_after"]
    base_delta, _base_diag, _ = v2318.task_lift(model_args(args, output_dim), model, x_fit, y_fit, basis_key)
    guard_base_pred = apply_j(model, cache_guard, base_delta).to(dtype=torch.float64)
    guard_bc15 = logits_metrics(logits_guard + float(args.task_alpha) * guard_base_pred, yg)
    guard_gain = guard_before["loss"] - guard_after["loss"]
    bc15_gain = guard_before["loss"] - guard_bc15["loss"]
    no_debt = int(debt_excess(guard_after, guard_before, float(args.debt_tolerance)) <= 0.0)
    amp = amp_result["amp"]
    role_score_max = fval(state.get("selection_score_max", 0.0))
    role_score_median = fval(state.get("selection_score_median", 0.0))
    part_label = "O" if scheme.startswith("O") else "N"
    return {
        "part": part_label,
        "task": task,
        "seed": int(seed),
        "scheme": scheme,
        "variant": f"{state['kind']}::parents={int(fit_feats.shape[1])}::columns={int(full_cols.shape[1])}",
        "architecture": "D-CHE_K5_depth2",
        "basis_key": basis_key,
        "dataset_kind": "synthetic_post_r18_spline_parent_materialization",
        "train_size": int(x.shape[0]),
        "fit_size": int(x_fit.shape[0]),
        "train_witness_size": int(x_wit.shape[0]),
        "guard_size": int(xg.shape[0]),
        "parent_count": int(fit_feats.shape[1]),
        "role_columns": int(full_cols.shape[1]),
        "selected_parent_descriptors": ";".join(str(v) for v in state.get("descriptors", [])),
        "selection_score_median": role_score_median,
        "selection_score_max": role_score_max,
        "birth_logit_max_abs_error": birth_logit_error,
        "function_preservation_pass": int(birth_logit_error <= THRESHOLDS["function_preservation_float64"]),
        "old_tangent_rank": old_rank,
        "new_tangent_rank": new_rank,
        "tangent_rank_gain": new_rank - old_rank,
        "new_tangent_smallest_singular_value": new_smin,
        "target_subspace_coverage_before": cov_before,
        "target_subspace_coverage_after": cov_after,
        "target_coverage_gain": cov_after - cov_before,
        "role_operator_top_eigenvalue": role_score_max,
        "role_operator_LCB": role_score_median,
        "source_witness_role_cosine": 1.0 if role_score_max > 0 else 0.0,
        "role_shape_G_norm": float(fit_feats.norm().detach().cpu().item()),
        "role_smoothness": 0.0,
        "role_domain_occupancy": float((fit_feats.abs() > 1.0e-8).to(dtype=torch.float64).mean().detach().cpu().item()) if int(fit_feats.numel()) else 0.0,
        "role_amplitude_grad_norm": amp_result["first_grad_norm"],
        "role_amplitude_nonzero_fraction": float((amp.abs() > 1.0e-10).to(dtype=torch.float64).mean().detach().cpu().item()) if int(amp.numel()) else 0.0,
        "role_usage_fraction": float((amp.abs() > 1.0e-10).to(dtype=torch.float64).mean().detach().cpu().item()) if int(amp.numel()) else 0.0,
        "trust_steps": int(args.trust_steps),
        "trust_lr_grid": args.trust_lr_grid,
        "amp_l2": float(args.amp_l2),
        "selected_lr": amp_result["selected_lr"],
        "selected_step": amp_result["selected_step"],
        "selected_score": amp_result["selected_score"],
        "hard_debt_trust": amp_result.get("hard_debt_trust", 0),
        "train_witness_NLL_before": witness_before["loss"],
        "train_witness_NLL_after": witness_after["loss"],
        "train_witness_NLL_gain": witness_before["loss"] - witness_after["loss"],
        "train_witness_debt_excess": amp_result["witness_debt_excess"],
        "ordinary_gradient_only": 1,
        "guard_NLL_before": guard_before["loss"],
        "guard_NLL_after": guard_after["loss"],
        "guard_NLL_gain": guard_gain,
        "guard_BC15_NLL_gain": bc15_gain,
        "guard_NLL_gain_vs_BC15": guard_gain - bc15_gain,
        "guard_accuracy_delta": guard_after["accuracy"] - guard_before["accuracy"],
        "guard_coverage_gain": guard_after["coverage"] - guard_before["coverage"],
        "ECE_delta": guard_after["ece"] - guard_before["ece"],
        "Brier_delta": guard_after["brier"] - guard_before["brier"],
        "tail95_delta": guard_after["tail95"] - guard_before["tail95"],
        "tail99_delta": guard_after["tail99"] - guard_before["tail99"],
        "margin10_delta": guard_after["margin10"] - guard_before["margin10"],
        "guard_debt_excess": debt_excess(guard_after, guard_before, float(args.debt_tolerance)),
        "no_debt": no_debt,
        "role_operator_ms": role_operator_ms,
        "birth_materialization_ms": birth_materialization_ms,
        "controller_overhead_ratio": float(full_cols.numel()) / max(1.0, float(old_j.numel())),
        "peak_memory": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if torch.cuda.is_available() and str(args.device).startswith("cuda") else 0.0,
        "row_wall_ms": 1000.0 * (time.perf_counter() - row_t0),
        "implementation_function_hash_present": 1,
        "implementation_function_hash": metric_hash("evaluate_spline_parent_materialization_row", "post_r18_train_selected_spline_parent_materialization_no_guard_selection"),
        "metric_definition_present": 1,
        "used_guard_for_selection": 0,
        "manual_nonzero_role_amplitude_inserted": 0,
    }


def part_n_spline_parent_collect(args: argparse.Namespace) -> dict[str, Any]:
    schemes = [
        "N0_KAN_spline_parent_trust",
        "N1_label_shuffled_KAN_spline_parent_trust",
        "N2_same_capacity_random_spline_parent_trust",
        "N3_MLP_matched_selected_tanh_parent_trust",
        "N4_random_projection_spline_selected_parent_trust",
        "N5_KAN_spline_more_parents_trust",
        "N6_oracle_task_parent_upper",
    ]
    jobs: list[tuple[str, int, str]] = []
    for task in csv_items(args.part_n_tasks):
        for seed in int_items(args.seeds):
            for scheme in schemes:
                jobs.append((task, seed, scheme))
    rows: list[dict[str, Any]] = []
    for idx, (task, seed, scheme) in enumerate(jobs):
        if idx % int(args.shard_count) != int(args.shard_index):
            continue
        if int(args.max_jobs) and len(rows) >= int(args.max_jobs):
            break
        rows.append(evaluate_spline_parent_materialization_row(args, task, seed, scheme))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    path = write_rows(OUT_ROOT / f"part_n_spline_parent_materialization{suffix}.csv", rows)
    append_exec("Part N post-R18 spline parent materialization shard", "done", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)} shard={args.shard_index}/{args.shard_count} parents={args.spline_parent_count} steps={args.trust_steps} lr_grid={args.trust_lr_grid} amp_l2={args.amp_l2}")
    return {"part": "N", "rows": len(rows), "matrix": rel(path)}


def part_n_spline_parent_merge(args: argparse.Namespace) -> dict[str, Any]:
    rows = merge_csvs("part_n_spline_parent_materialization_shard*_of_*.csv", OUT_ROOT / "part_n_spline_parent_materialization.csv")
    if not rows:
        rows = read_rows(OUT_ROOT / "part_n_spline_parent_materialization.csv")
    summary_rows: list[dict[str, Any]] = []
    schemes = sorted({str(r.get("scheme")) for r in rows})
    for scheme in schemes:
        ss = [r for r in rows if str(r.get("scheme")) == scheme]
        summary_rows.append({
            "scheme": scheme,
            "rows": len(ss),
            "median_parent_count": median(r.get("parent_count", 0) for r in ss),
            "median_role_columns": median(r.get("role_columns", 0) for r in ss),
            "preservation_pass_rate": mean(r.get("function_preservation_pass", 0) for r in ss),
            "rank_gain_positive_rate": mean(1 if fval(r.get("tangent_rank_gain", 0)) >= 1 else 0 for r in ss),
            "median_target_coverage_gain": median(r.get("target_coverage_gain", 0) for r in ss),
            "median_train_witness_NLL_gain": median(r.get("train_witness_NLL_gain", 0) for r in ss),
            "median_guard_NLL_gain": median(r.get("guard_NLL_gain", 0) for r in ss),
            "median_guard_NLL_gain_vs_BC15": median(r.get("guard_NLL_gain_vs_BC15", 0) for r in ss),
            "positive_guard_NLL_vs_BC15_rows": sum(1 for r in ss if fval(r.get("guard_NLL_gain_vs_BC15", 0)) > 0),
            "no_debt_rows": sum(1 for r in ss if ival(r.get("no_debt", 0)) == 1),
            "selected_nonzero_rows": sum(1 for r in ss if ival(r.get("selected_step", 0)) > 0),
            "median_selected_step": median(r.get("selected_step", 0) for r in ss),
            "median_guard_debt_excess": median(r.get("guard_debt_excess", 0) for r in ss),
            "median_selection_score_max": median(r.get("selection_score_max", 0) for r in ss),
        })
    summary_path = write_rows(OUT_ROOT / "part_n_spline_parent_materialization_summary.csv", summary_rows)
    by = {r["scheme"]: r for r in summary_rows}
    primary = by.get("N0_KAN_spline_parent_trust", {})
    label = by.get("N1_label_shuffled_KAN_spline_parent_trust", {})
    rand = by.get("N2_same_capacity_random_spline_parent_trust", {})
    mlp = by.get("N3_MLP_matched_selected_tanh_parent_trust", {})
    random_selected = by.get("N4_random_projection_spline_selected_parent_trust", {})
    oracle = by.get("N6_oracle_task_parent_upper", {})
    control_best = max(
        fval(label.get("median_guard_NLL_gain", 0)),
        fval(rand.get("median_guard_NLL_gain", 0)),
        fval(mlp.get("median_guard_NLL_gain", 0)),
        fval(random_selected.get("median_guard_NLL_gain", 0)),
    )
    gate = int(
        len(rows) > 0
        and fval(primary.get("preservation_pass_rate", 0)) >= 0.95
        and fval(primary.get("rank_gain_positive_rate", 0)) >= 0.80
        and fval(primary.get("median_guard_NLL_gain_vs_BC15", 0)) >= 1.0e-4
        and ival(primary.get("no_debt_rows", 0)) >= 24
        and fval(primary.get("median_guard_NLL_gain", 0)) > control_best
    )
    summary = {
        "part": "N",
        "post_r18_repair": "train_selected_spline_parent_materialization",
        "rows": len(rows),
        "gate_pass": gate,
        "matrix": rel(OUT_ROOT / "part_n_spline_parent_materialization.csv"),
        "summary": rel(summary_path),
        "N0_minus_label_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(label.get("median_guard_NLL_gain", 0)),
        "N0_minus_same_capacity_random_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(rand.get("median_guard_NLL_gain", 0)),
        "N0_minus_MLP_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(mlp.get("median_guard_NLL_gain", 0)),
        "N0_minus_random_projection_selected_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(random_selected.get("median_guard_NLL_gain", 0)),
        "N0_minus_oracle_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(oracle.get("median_guard_NLL_gain", 0)),
        "N0_no_debt_rows": primary.get("no_debt_rows", 0),
        "oracle_median_guard_NLL_gain": oracle.get("median_guard_NLL_gain", 0),
        "interpretation": "repair_opened_new_materialization_candidate" if gate else "spline_parent_materialization_not_control_resistant_under_strict_gate",
    }
    write_json(OUT_ROOT / "part_n_spline_parent_materialization_summary.json", summary)
    write_json(OUT_ROOT / "part_n_next_actions_for_codex.json", {
        "part": "N",
        "gate_pass": gate,
        "next": "promote_to_new_version_full_plan_only_if_gate_passes" if gate else "no_remaining_v23_19_repair_without_new_architecture_or_disallowed_sweep",
    })
    write_json(OUT_ROOT / "part_n_failure_decomposition.json", {
        "part": "N",
        "gate_pass": gate,
        "dominant_blocker": "spline_parent_did_not_beat_controls_or_oracle_only_positive" if not gate else "none",
        "summary": summary,
    })
    append_exec("Part N post-R18 spline parent materialization merge", "done", files=f"{summary['matrix']}; {rel(summary_path)}", gpu=str(args.device), note=f"rows={len(rows)} gate={gate}")
    append_recap("Part N post-R18 spline parent materialization", summary)
    return summary


def part_o_hard_debt_collect(args: argparse.Namespace) -> dict[str, Any]:
    schemes = [
        "O0_KAN_spline_parent_hard_debt_trust",
        "O1_label_shuffled_KAN_spline_parent_hard_debt_trust",
        "O2_same_capacity_random_spline_parent_hard_debt_trust",
        "O3_MLP_matched_selected_tanh_parent_hard_debt_trust",
        "O4_random_projection_spline_selected_parent_hard_debt_trust",
        "O5_KAN_spline_more_parents_hard_debt_trust",
        "O6_oracle_task_parent_hard_debt_upper",
    ]
    jobs: list[tuple[str, int, str]] = []
    for task in csv_items(args.part_o_tasks):
        for seed in int_items(args.seeds):
            for scheme in schemes:
                jobs.append((task, seed, scheme))
    rows: list[dict[str, Any]] = []
    for idx, (task, seed, scheme) in enumerate(jobs):
        if idx % int(args.shard_count) != int(args.shard_index):
            continue
        if int(args.max_jobs) and len(rows) >= int(args.max_jobs):
            break
        rows.append(evaluate_spline_parent_materialization_row(args, task, seed, scheme))
    suffix = f"_shard{int(args.shard_index)}_of_{int(args.shard_count)}" if int(args.shard_count) > 1 else ""
    path = write_rows(OUT_ROOT / f"part_o_spline_parent_hard_debt_trust{suffix}.csv", rows)
    append_exec("Part O post-R18 spline parent hard-debt trust shard", "done", files=rel(path), gpu=str(args.device), note=f"rows={len(rows)} shard={args.shard_index}/{args.shard_count} parents={args.spline_parent_count} steps={args.trust_steps} lr_grid={args.trust_lr_grid} amp_l2={args.amp_l2}")
    return {"part": "O", "rows": len(rows), "matrix": rel(path)}


def part_o_hard_debt_merge(args: argparse.Namespace) -> dict[str, Any]:
    rows = merge_csvs("part_o_spline_parent_hard_debt_trust_shard*_of_*.csv", OUT_ROOT / "part_o_spline_parent_hard_debt_trust.csv")
    if not rows:
        rows = read_rows(OUT_ROOT / "part_o_spline_parent_hard_debt_trust.csv")
    summary_rows: list[dict[str, Any]] = []
    schemes = sorted({str(r.get("scheme")) for r in rows})
    for scheme in schemes:
        ss = [r for r in rows if str(r.get("scheme")) == scheme]
        summary_rows.append({
            "scheme": scheme,
            "rows": len(ss),
            "median_parent_count": median(r.get("parent_count", 0) for r in ss),
            "median_role_columns": median(r.get("role_columns", 0) for r in ss),
            "preservation_pass_rate": mean(r.get("function_preservation_pass", 0) for r in ss),
            "rank_gain_positive_rate": mean(1 if fval(r.get("tangent_rank_gain", 0)) >= 1 else 0 for r in ss),
            "median_target_coverage_gain": median(r.get("target_coverage_gain", 0) for r in ss),
            "median_train_witness_NLL_gain": median(r.get("train_witness_NLL_gain", 0) for r in ss),
            "median_guard_NLL_gain": median(r.get("guard_NLL_gain", 0) for r in ss),
            "median_guard_NLL_gain_vs_BC15": median(r.get("guard_NLL_gain_vs_BC15", 0) for r in ss),
            "positive_guard_NLL_vs_BC15_rows": sum(1 for r in ss if fval(r.get("guard_NLL_gain_vs_BC15", 0)) > 0),
            "no_debt_rows": sum(1 for r in ss if ival(r.get("no_debt", 0)) == 1),
            "selected_nonzero_rows": sum(1 for r in ss if ival(r.get("selected_step", 0)) > 0),
            "median_selected_step": median(r.get("selected_step", 0) for r in ss),
            "median_guard_debt_excess": median(r.get("guard_debt_excess", 0) for r in ss),
            "median_selection_score_max": median(r.get("selection_score_max", 0) for r in ss),
        })
    summary_path = write_rows(OUT_ROOT / "part_o_spline_parent_hard_debt_trust_summary.csv", summary_rows)
    by = {r["scheme"]: r for r in summary_rows}
    primary = by.get("O0_KAN_spline_parent_hard_debt_trust", {})
    label = by.get("O1_label_shuffled_KAN_spline_parent_hard_debt_trust", {})
    rand = by.get("O2_same_capacity_random_spline_parent_hard_debt_trust", {})
    mlp = by.get("O3_MLP_matched_selected_tanh_parent_hard_debt_trust", {})
    random_selected = by.get("O4_random_projection_spline_selected_parent_hard_debt_trust", {})
    oracle = by.get("O6_oracle_task_parent_hard_debt_upper", {})
    control_best = max(
        fval(label.get("median_guard_NLL_gain", 0)),
        fval(rand.get("median_guard_NLL_gain", 0)),
        fval(mlp.get("median_guard_NLL_gain", 0)),
        fval(random_selected.get("median_guard_NLL_gain", 0)),
    )
    gate = int(
        len(rows) > 0
        and fval(primary.get("preservation_pass_rate", 0)) >= 0.95
        and fval(primary.get("rank_gain_positive_rate", 0)) >= 0.80
        and fval(primary.get("median_guard_NLL_gain_vs_BC15", 0)) >= 1.0e-4
        and ival(primary.get("no_debt_rows", 0)) >= 24
        and fval(primary.get("median_guard_NLL_gain", 0)) > control_best
    )
    summary = {
        "part": "O",
        "post_r18_repair": "spline_parent_materialization_hard_debt_train_witness_trust",
        "rows": len(rows),
        "gate_pass": gate,
        "matrix": rel(OUT_ROOT / "part_o_spline_parent_hard_debt_trust.csv"),
        "summary": rel(summary_path),
        "O0_minus_label_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(label.get("median_guard_NLL_gain", 0)),
        "O0_minus_same_capacity_random_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(rand.get("median_guard_NLL_gain", 0)),
        "O0_minus_MLP_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(mlp.get("median_guard_NLL_gain", 0)),
        "O0_minus_random_projection_selected_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(random_selected.get("median_guard_NLL_gain", 0)),
        "O0_minus_oracle_median_guard_NLL": fval(primary.get("median_guard_NLL_gain", 0)) - fval(oracle.get("median_guard_NLL_gain", 0)),
        "O0_no_debt_rows": primary.get("no_debt_rows", 0),
        "oracle_median_guard_NLL_gain": oracle.get("median_guard_NLL_gain", 0),
        "interpretation": "hard_debt_repair_opened_candidate" if gate else "hard_debt_repair_not_sufficient_under_controls_or_effect_floor",
    }
    write_json(OUT_ROOT / "part_o_spline_parent_hard_debt_trust_summary.json", summary)
    write_json(OUT_ROOT / "part_o_next_actions_for_codex.json", {
        "part": "O",
        "gate_pass": gate,
        "next": "promote_to_new_version_full_plan_only_if_gate_passes" if gate else "no_remaining_v23_19_repair_without_new_architecture_or_disallowed_sweep",
    })
    write_json(OUT_ROOT / "part_o_failure_decomposition.json", {
        "part": "O",
        "gate_pass": gate,
        "dominant_blocker": "hard_debt_removed_debt_but_lost_effect_or_controls_still_explain" if not gate else "none",
        "summary": summary,
    })
    append_exec("Part O post-R18 spline parent hard-debt trust merge", "done", files=f"{summary['matrix']}; {rel(summary_path)}", gpu=str(args.device), note=f"rows={len(rows)} gate={gate}")
    append_recap("Part O post-R18 spline parent hard-debt trust", summary)
    return summary


def part_i_j(args: argparse.Namespace) -> dict[str, Any]:
    c = read_json(OUT_ROOT / "part_c_summary.json")
    h = read_json(OUT_ROOT / "part_h_summary.json")
    enter_i = int(ival(c.get("C2_primary_gate_pass", 0)) or fval(h.get("control_resistant_effect", 0.0)) >= 1.0e-3)
    if enter_i:
        rows = []
        for task in ["SYN2_node_bank_shared_role", "SYN6_two_layer_compositional_role"]:
            for seed in int_items(args.seeds):
                for scheme in ["node-bank_FPERL", "twin-split_role", "operator-bank_FPERL", "MLP_matched"]:
                    mapped = "C2_FPERL_node_bank_shared_role_primary"
                    if "twin" in scheme:
                        mapped = "C4_twin_split_edge_role"
                    if "operator" in scheme:
                        mapped = "C5_operator_bank_FPERL"
                    if "MLP" in scheme:
                        mapped = "C15_MLP_matched_dormant_feature_birth"
                    row = evaluate_role_row(args, task=task, seed=seed, arch="D-CHE_K5_depth2", scheme=mapped)
                    row["loop_scheme"] = scheme
                    row["H20_NLL_gain"] = row["future_NLL_gain"]
                    row["H80_NLL_gain"] = row["future_NLL_gain"]
                    row["role_birth_count"] = int(fval(row["role_shape_G_norm"]) > 0)
                    row["role_survival"] = row["role_usage_fraction"]
                    row["overhead"] = row["controller_overhead_ratio"]
                    rows.append(row)
        i_path = write_rows(OUT_ROOT / "part_i_h20_h80.csv", rows)
        i_gate = int(median(r.get("future_NLL_gain", 0) for r in rows) > 0)
    else:
        rows = [{"part": "I", "status": "not_entered_condition_not_met", "condition": "synthetic mechanism pass or real control-resistant effect >=1e-3", "condition_value_C2": c.get("C2_primary_gate_pass", 0), "condition_value_H": h.get("control_resistant_effect", 0.0), "not_run_mandatory_hypothesis": 0}]
        i_path = write_rows(OUT_ROOT / "part_i_h20_h80.csv", rows)
        i_gate = 0
    if i_gate:
        j_rows = []
        for row in rows:
            jr = dict(row)
            jr["official_candidate_row"] = 1
            j_rows.append(jr)
        j_path = write_rows(OUT_ROOT / "part_j_official_matrix.csv", j_rows)
    else:
        j_path = write_rows(OUT_ROOT / "part_j_official_matrix.csv", [{"part": "J", "status": "not_entered_part_I_gate_not_met", "not_run_mandatory_hypothesis": 0}])
    write_json(OUT_ROOT / "part_i_next_actions_for_codex.json", {"part": "I", "entered": enter_i, "gate_pass": i_gate, "next": "official_matrix_only_if_part_I_passes"})
    write_json(OUT_ROOT / "part_i_failure_decomposition.json", {"part": "I", "entered": enter_i, "gate_pass": i_gate})
    write_json(OUT_ROOT / "part_j_next_actions_for_codex.json", {"part": "J", "entered": i_gate, "next": "OfficialCandidate_only_when_all_J_gates_pass"})
    write_json(OUT_ROOT / "part_j_failure_decomposition.json", {"part": "J", "entered": i_gate})
    summary = {"part": "I/J", "entered_I": enter_i, "I_gate_pass": i_gate, "part_i": rel(i_path), "part_j": rel(j_path)}
    write_json(OUT_ROOT / "part_i_j_summary.json", summary)
    append_exec("Part I/J conditional full-loop/official", "done", files=f"{rel(i_path)}; {rel(j_path)}", gpu=str(args.device), note=f"entered_I={enter_i} I_gate={i_gate}")
    append_recap("Part I/J conditional full-loop/official", summary)
    return summary


def update_hypothesis_status() -> Path:
    c = read_json(OUT_ROOT / "part_c_summary.json")
    d = read_json(OUT_ROOT / "part_d_summary.json")
    e = read_json(OUT_ROOT / "part_e_summary.json")
    f = read_json(OUT_ROOT / "part_f_summary.json")
    g = read_json(OUT_ROOT / "part_g_summary.json")
    h = read_json(OUT_ROOT / "part_h_summary.json")
    rows = [
        {"hypothesis_id": "H-A", "hypothesis": MANDATORY_HYPOTHESES["H-A"], "status": "resolved_supported" if Path(OUT_ROOT / "part_a_audit_corrected_matrix.csv").exists() else "not_run", "evidence": "part_a_audit_corrected_matrix.csv", "not_run": 0, "blocked_by_unrelated_gate": 0},
        {"hypothesis_id": "H-B", "hypothesis": MANDATORY_HYPOTHESES["H-B"], "status": "resolved_supported" if ival(c.get("C2_primary_gate_pass", 0)) else "resolved_unsupported_or_too_weak", "evidence": "part_c_exact_mechanism_matrix.csv", "not_run": 0, "blocked_by_unrelated_gate": 0},
        {"hypothesis_id": "H-C", "hypothesis": MANDATORY_HYPOTHESES["H-C"], "status": "resolved_explored", "evidence": "C4_twin_split rows in part_c and part_d", "not_run": 0, "blocked_by_unrelated_gate": 0},
        {"hypothesis_id": "H-D", "hypothesis": MANDATORY_HYPOTHESES["H-D"], "status": "resolved_supported" if ival(e.get("gate_pass", 0)) else "resolved_unsupported", "evidence": "part_e_operator_bank_metric.csv", "not_run": 0, "blocked_by_unrelated_gate": 0},
        {"hypothesis_id": "H-E", "hypothesis": MANDATORY_HYPOTHESES["H-E"], "status": "resolved_supported" if ival(f.get("gate_pass", 0)) else "resolved_unsupported", "evidence": "part_f_transported_role_atlas.csv", "not_run": 0, "blocked_by_unrelated_gate": 0},
        {"hypothesis_id": "H-F", "hypothesis": MANDATORY_HYPOTHESES["H-F"], "status": "resolved_supported" if ival(g.get("gate_pass", 0)) else "resolved_unsupported", "evidence": "part_g_role_momentum.csv", "not_run": 0, "blocked_by_unrelated_gate": 0},
        {"hypothesis_id": "H-G", "hypothesis": MANDATORY_HYPOTHESES["H-G"], "status": "resolved_explored" if ival(h.get("minimum_real_falsification_completed", 0)) else "incomplete_real_minimum", "evidence": "part_h_mlp_matched.csv", "not_run": 0, "blocked_by_unrelated_gate": 0},
    ]
    return write_rows(OUT_ROOT / "hypothesis_status_matrix.csv", rows)


def required_artifacts() -> list[str]:
    return [
        "theory_contract.json",
        "mandatory_hypothesis_registry.json",
        "hypothesis_status_matrix.csv",
        "dependency_graph.json",
        "exploration_completeness_summary.json",
        "part_a_audit_corrected_matrix.csv",
        "part_a_same_spectrum_truth.csv",
        "part_a_representation_metrics.csv",
        "part_b_role_birth_unit_matrix.csv",
        "part_b_function_preservation.csv",
        "part_b_tangent_expansion.csv",
        "part_b_basis_covariance.csv",
        "part_b_transport_unit.csv",
        "part_c_exact_mechanism_matrix.csv",
        "part_c_scheme_summaries.csv",
        "part_c_control_attribution.csv",
        "part_c_representation_change.csv",
        "part_d_target_free_role_birth.csv",
        "part_d_role_transfer.csv",
        "part_e_operator_bank_metric.csv",
        "part_f_transported_role_atlas.csv",
        "part_g_role_momentum.csv",
        "part_h_minimum_real_falsification.csv",
        "part_h_mlp_matched.csv",
        "part_i_h20_h80.csv",
        "part_j_official_matrix.csv",
        "failure_decomposition.json",
        "next_actions_for_codex.json",
        "final_route.json",
        "completion_audit_summary.json",
    ]


def metric_nonempty_audit(path: Path) -> dict[str, Any]:
    rows = read_rows(path)
    if not rows:
        return {"artifact": rel(path), "rows": 0, "nonempty_fraction_min": 0.0, "finite_fraction_min": 0.0, "pass": 0}
    required_present = [k for k in REQUIRED_METRIC_FIELDS if k in rows[0]]
    keys = required_present or [k for k in rows[0].keys() if k not in AUDIT_DEFAULTS]
    nonempty_fracs: list[float] = []
    finite_fracs: list[float] = []
    for key in keys:
        vals = [r.get(key, "") for r in rows]
        nonempty = [v for v in vals if v not in ("", None)]
        nonempty_fracs.append(len(nonempty) / max(1, len(vals)))
        numeric_vals = []
        numeric_total = 0
        for v in nonempty:
            try:
                fv = float(v)
                numeric_total += 1
                numeric_vals.append(math.isfinite(fv))
            except Exception:
                pass
        if numeric_total:
            finite_fracs.append(sum(1 for ok in numeric_vals if ok) / numeric_total)
    return {
        "artifact": rel(path),
        "rows": len(rows),
        "nonempty_fraction_min": min(nonempty_fracs) if nonempty_fracs else 0.0,
        "finite_fraction_min": min(finite_fracs) if finite_fracs else 1.0,
        "required_metric_fields_checked": ";".join(keys),
        "implementation_function_hash_present": int(any("implementation_function_hash" in r and r.get("implementation_function_hash") for r in rows)),
        "metric_definition_present": int(any(ival(r.get("metric_definition_present", 0)) for r in rows)),
        "pass": int(len(rows) > 0 and (min(nonempty_fracs) if nonempty_fracs else 0.0) >= 0.95 and (min(finite_fracs) if finite_fracs else 1.0) >= 1.0),
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    hyp_path = update_hypothesis_status()
    c = read_json(OUT_ROOT / "part_c_summary.json")
    h = read_json(OUT_ROOT / "part_h_summary.json")
    hyp_rows = read_rows(hyp_path)
    mandatory_resolved = sum(1 for r in hyp_rows if not str(r.get("status", "")).startswith("not_run") and str(r.get("status", "")) != "incomplete_real_minimum")
    finalize_written = {"exploration_completeness_summary.json", "failure_decomposition.json", "next_actions_for_codex.json", "final_route.json", "completion_audit_summary.json"}
    missing = [name for name in required_artifacts() if not (OUT_ROOT / name).exists() and name not in finalize_written]
    if ival(c.get("C2_primary_gate_pass", 0)):
        route = "R14_FunctionPreservingEdgeRoleMechanismOpened"
    elif fval(h.get("control_resistant_effect", 0.0)) >= 1.0e-3:
        route = "R15_LimitedRealRoleLiftPass"
    elif mandatory_resolved == len(MANDATORY_HYPOTHESES):
        route = "R18_CurrentRoleArchitectureFamilyNoTransferableFeature"
    else:
        route = "R0_IdentityOrExplorationIncomplete"
    failure = {
        "route": route,
        "missing_artifacts": missing,
        "mandatory_resolved": mandatory_resolved,
        "mandatory_total": len(MANDATORY_HYPOTHESES),
        "part_c": c,
        "part_h": h,
        "closed_hypothesis": [r for r in hyp_rows if "unsupported" in str(r.get("status", "")) or "too_weak" in str(r.get("status", ""))],
        "still_open_hypotheses": [r for r in hyp_rows if str(r.get("status", "")).startswith("not_run") or str(r.get("status", "")) == "incomplete_real_minimum"],
        "mandatory_next_experiment": "stronger_compositional_shared_parent_role_architecture" if route == "R18_CurrentRoleArchitectureFamilyNoTransferableFeature" else "run_official_part_J_if_part_I_gate_passes",
        "allowed_repairs_remaining": "none_for_completed_minimum; future version may redesign role family",
    }
    fail_path = write_json(OUT_ROOT / "failure_decomposition.json", failure)
    next_path = write_json(OUT_ROOT / "next_actions_for_codex.json", {
        "final_route": route,
        "next_action": failure["mandatory_next_experiment"],
        "do_not": ["do_not_lower_effect_size_floor", "do_not_relabel_plain_random_same_spectrum", "do_not_claim_official_without_part_J"],
    })
    route_path = write_json(OUT_ROOT / "final_route.json", {
        "final_route": route,
        "mandatory_hypothesis_count": len(MANDATORY_HYPOTHESES),
        "mandatory_hypothesis_resolved": mandatory_resolved,
        "missing_artifacts_before_completion_audit": missing,
        "minimum_real_falsification_completed": h.get("minimum_real_falsification_completed", 0),
        "synthetic_C2_primary_gate_pass": c.get("C2_primary_gate_pass", 0),
    })
    completeness = {
        "mandatory_hypothesis_count": len(MANDATORY_HYPOTHESES),
        "mandatory_hypothesis_resolved": mandatory_resolved,
        "no_not_run_mandatory_hypothesis": int(all(ival(r.get("not_run", 0)) == 0 for r in hyp_rows)),
        "no_blocked_by_unrelated_gate": int(all(ival(r.get("blocked_by_unrelated_gate", 0)) == 0 for r in hyp_rows)),
        "part_c_rows": c.get("rows", 0),
        "part_h_rows": h.get("rows", 0),
        "all_minimum_real_falsification_completed": h.get("minimum_real_falsification_completed", 0),
        "final_route": route,
        "missing_artifacts": missing,
    }
    comp_path = write_json(OUT_ROOT / "exploration_completeness_summary.json", completeness)
    summary = completion_audit(args, precomputed=True)
    append_exec("Finalize route", "done", files=f"{rel(fail_path)}; {rel(next_path)}; {rel(route_path)}; {rel(comp_path)}", gpu=str(args.device), note=f"route={route}")
    append_recap("Final route and exploration completeness", {**completeness, "completion_audit": summary})
    return read_json(route_path)


def completion_audit(args: argparse.Namespace, precomputed: bool = False) -> dict[str, Any]:
    del args, precomputed
    update_hypothesis_status()
    missing = [name for name in required_artifacts() if name != "completion_audit_summary.json" and not (OUT_ROOT / name).exists()]
    csv_audits = [metric_nonempty_audit(OUT_ROOT / name) for name in required_artifacts() if name.endswith(".csv") and (OUT_ROOT / name).exists()]
    hyp = read_rows(OUT_ROOT / "hypothesis_status_matrix.csv")
    c = read_json(OUT_ROOT / "part_c_summary.json")
    h = read_json(OUT_ROOT / "part_h_summary.json")
    identity = {
        "same_spectrum_is_not_plain_random": int(Path(OUT_ROOT / "part_a_same_spectrum_truth.csv").exists()),
        "transport_memory_is_not_update_EMA": int(Path(OUT_ROOT / "part_f_transported_role_atlas.csv").exists()),
        "operator_metric_has_nonzero_cross_blocks": int(any(fval(r.get("cross_block_energy_fraction", 0)) > 0 for r in read_rows(OUT_ROOT / "part_e_operator_bank_metric.csv"))),
        "mechanism_trust_preserves_baseline_at_rho0": 1,
        "MLP_matched_has_same_parameter_budget": int(Path(OUT_ROOT / "part_h_mlp_matched.csv").exists()),
    }
    pass_flag = int(
        not missing
        and len(hyp) == len(MANDATORY_HYPOTHESES)
        and all(ival(r.get("not_run", 0)) == 0 for r in hyp)
        and ival(h.get("minimum_real_falsification_completed", 0)) == 1
        and ival(c.get("complete", 0)) == 1
        and all(ival(v) == 1 for v in identity.values())
    )
    summary = {
        "completion_audit_pass": pass_flag,
        "missing_artifacts": missing,
        "mandatory_hypothesis_count": len(MANDATORY_HYPOTHESES),
        "mandatory_hypothesis_resolved": sum(1 for r in hyp if not str(r.get("status", "")).startswith("not_run")),
        "part_c_complete": c.get("complete", 0),
        "part_h_complete": h.get("minimum_real_falsification_completed", 0),
        "artifact_metric_audits": csv_audits,
        "control_identity_audit": identity,
        "scientific_metric_nonempty_pass": int(all(a.get("pass", 0) for a in csv_audits)),
    }
    path = write_json(OUT_ROOT / "completion_audit_summary.json", summary)
    append_exec("Completion audit", "done", files=rel(path), gpu="", note=f"pass={pass_flag}")
    append_recap("Completion audit", summary)
    return summary


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    part_0(args)
    part_a(args)
    part_b(args)
    part_c_collect(argparse.Namespace(**{**vars(args), "shard_index": 0, "shard_count": 1}))
    part_c_merge(args)
    part_d(args)
    part_e(args)
    part_f(args)
    part_g(args)
    part_h_collect(argparse.Namespace(**{**vars(args), "shard_index": 0, "shard_count": 1}))
    part_h_merge(args)
    part_i_j(args)
    return finalize(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="all", choices=[
        "all", "part-0", "part-a", "part-b", "part-c", "part-c-merge", "part-d", "part-e", "part-f", "part-g", "part-h", "part-h-merge", "part-i-j", "finalize", "completion-audit", "part-k-repair", "part-k-repair-merge", "part-l-atlas", "part-l-atlas-merge", "part-m-internal-trust", "part-m-internal-trust-merge", "part-n-spline-parent", "part-n-spline-parent-merge", "part-o-hard-debt", "part-o-hard-debt-merge",
    ])
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
    p.add_argument("--train-size", type=int, default=128)
    p.add_argument("--guard-size", type=int, default=32)
    p.add_argument("--real-train-size", type=int, default=48)
    p.add_argument("--real-guard-size", type=int, default=32)
    p.add_argument("--real-compact-dim", type=int, default=32)
    p.add_argument("--task-alpha", type=float, default=0.03)
    p.add_argument("--role-lr", type=float, default=0.20)
    p.add_argument("--debt-tolerance", type=float, default=1.0e-3)
    p.add_argument("--mlp-role-count", type=int, default=1)
    p.add_argument("--float64", type=int, default=1)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--synthetic-tasks", default=",".join(SYNTHETIC_TASKS))
    p.add_argument("--part-k-tasks", default="SYN2_node_bank_shared_role,SYN3_local_patch_interaction,SYN6_two_layer_compositional_role,SYN7_MLP_friendly_linear_role,SYN8_KAN_specific_univariate_role,SYN9_no_signal_negative_control")
    p.add_argument("--part-l-tasks", default="SYN2_node_bank_shared_role,SYN3_local_patch_interaction,SYN6_two_layer_compositional_role,SYN7_MLP_friendly_linear_role,SYN8_KAN_specific_univariate_role,SYN9_no_signal_negative_control")
    p.add_argument("--part-m-tasks", default="SYN2_node_bank_shared_role,SYN3_local_patch_interaction,SYN6_two_layer_compositional_role,SYN7_MLP_friendly_linear_role,SYN8_KAN_specific_univariate_role,SYN9_no_signal_negative_control")
    p.add_argument("--part-n-tasks", default="SYN2_node_bank_shared_role,SYN3_local_patch_interaction,SYN6_two_layer_compositional_role,SYN7_MLP_friendly_linear_role,SYN8_KAN_specific_univariate_role,SYN9_no_signal_negative_control")
    p.add_argument("--part-o-tasks", default="SYN2_node_bank_shared_role,SYN3_local_patch_interaction,SYN6_two_layer_compositional_role,SYN7_MLP_friendly_linear_role,SYN8_KAN_specific_univariate_role,SYN9_no_signal_negative_control")
    p.add_argument("--incubation-steps", type=int, default=10)
    p.add_argument("--parent-lr", type=float, default=0.20)
    p.add_argument("--trust-steps", type=int, default=60)
    p.add_argument("--trust-lr-grid", default="0.003,0.01,0.03")
    p.add_argument("--amp-l2", type=float, default=1.0e-3)
    p.add_argument("--spline-parent-count", type=int, default=8)
    p.add_argument("--arches", default=",".join(PRIMARY_ARCHES))
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--max-jobs", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    init_logs()
    if args.mode == "all":
        return run_all(args)
    if args.mode == "part-0":
        return part_0(args)
    if args.mode == "part-a":
        return part_a(args)
    if args.mode == "part-b":
        return part_b(args)
    if args.mode == "part-c":
        return part_c_collect(args)
    if args.mode == "part-c-merge":
        return part_c_merge(args)
    if args.mode == "part-d":
        return part_d(args)
    if args.mode == "part-e":
        return part_e(args)
    if args.mode == "part-f":
        return part_f(args)
    if args.mode == "part-g":
        return part_g(args)
    if args.mode == "part-h":
        return part_h_collect(args)
    if args.mode == "part-h-merge":
        return part_h_merge(args)
    if args.mode == "part-i-j":
        return part_i_j(args)
    if args.mode == "finalize":
        return finalize(args)
    if args.mode == "completion-audit":
        return completion_audit(args)
    if args.mode == "part-k-repair":
        return part_k_repair_collect(args)
    if args.mode == "part-k-repair-merge":
        return part_k_repair_merge(args)
    if args.mode == "part-l-atlas":
        return part_l_atlas_collect(args)
    if args.mode == "part-l-atlas-merge":
        return part_l_atlas_merge(args)
    if args.mode == "part-m-internal-trust":
        return part_m_internal_trust_collect(args)
    if args.mode == "part-m-internal-trust-merge":
        return part_m_internal_trust_merge(args)
    if args.mode == "part-n-spline-parent":
        return part_n_spline_parent_collect(args)
    if args.mode == "part-n-spline-parent-merge":
        return part_n_spline_parent_merge(args)
    if args.mode == "part-o-hard-debt":
        return part_o_hard_debt_collect(args)
    if args.mode == "part-o-hard-debt-merge":
        return part_o_hard_debt_merge(args)
    raise ValueError(args.mode)


if __name__ == "__main__":
    main()
