#!/usr/bin/env python3
"""Shared DG-KAN training utilities used by the recovered experiment scripts.

The implementation is intentionally compact and auditable. It restores the
pieces repeatedly referenced in docs/log.md and the FGO-v4 plan:

* residual RBF DG-KAN classifier,
* AdamW and functional coefficient updates,
* diag/Sobolev/data-pulse metrics,
* branch-aware and geometry-aware schedules,
* lightweight noisy-label/SNR variants,
* CSV/JSON summaries with the fields used in the documents.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import struct
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import urllib.error
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import datasets
except Exception as exc:  # pragma: no cover - only used for clearer runtime errors.
    datasets = None
    _TORCHVISION_IMPORT_ERROR = exc
else:
    _TORCHVISION_IMPORT_ERROR = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=to_jsonable)


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_jsonable(row.get(key, "")) for key in keys})


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: List[Dict[str, Any]] = []
    for row in rows:
        parsed: Dict[str, Any] = {}
        for key, value in row.items():
            if value == "":
                parsed[key] = value
                continue
            try:
                if any(ch in value for ch in [".", "e", "E"]):
                    parsed[key] = float(value)
                else:
                    parsed[key] = int(value)
            except (ValueError, TypeError):
                parsed[key] = value
        out.append(parsed)
    return out


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


@dataclass
class DataBundle:
    name: str
    input_dim: int
    num_classes: int
    x_train: torch.Tensor
    y_train: torch.Tensor
    y_train_clean: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    label_noise: float = 0.0
    used_fake_data: bool = False


_KMNIST_HF_COMMIT = "53810959241e8f35890445245170c8cec0672f81"
_KMNIST_RESOURCES: Tuple[Tuple[str, str, str], ...] = (
    ("train-images-idx3-ubyte.gz", "bdb82020997e1d708af4cf47b453dcf7", "train images"),
    ("train-labels-idx1-ubyte.gz", "e144d726b3acfaa3e44228e80efcd344", "train labels"),
    ("t10k-images-idx3-ubyte.gz", "5c965bf0a639b31b8f53240b1b52f4d7", "test images"),
    ("t10k-labels-idx1-ubyte.gz", "7320c461ea6c1c855c0b718fb2a4b134", "test labels"),
)


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kmnist_mirror_bases() -> List[str]:
    env_text = os.environ.get("KMNIST_MIRRORS") or os.environ.get("KMNIST_MIRROR") or ""
    env_bases = [part.strip().rstrip("/") for part in env_text.split(",") if part.strip()]
    return [
        *env_bases,
        f"https://huggingface.co/datasets/tanganke/kmnist/resolve/{_KMNIST_HF_COMMIT}/raw",
        "http://codh.rois.ac.jp/kmnist/dataset/kmnist",
    ]


def _download_file(url: str, dest: Path, *, timeout: float = 45.0) -> None:
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "DG-LCA/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _ensure_kmnist_raw(data_root: Path, *, download: bool) -> Path:
    raw_dir = ensure_dir(data_root / "KMNIST" / "raw")
    invalid = [
        filename
        for filename, md5, _ in _KMNIST_RESOURCES
        if not (raw_dir / filename).exists() or _file_md5(raw_dir / filename) != md5
    ]
    if not invalid:
        return raw_dir
    if not download:
        raise RuntimeError(
            "KMNIST raw files are missing or invalid under "
            f"{raw_dir}. Re-run with download=True or set KMNIST_MIRROR(S)."
        )

    errors: List[str] = []
    for filename, md5, label in _KMNIST_RESOURCES:
        dest = raw_dir / filename
        if dest.exists() and _file_md5(dest) == md5:
            continue
        for base in _kmnist_mirror_bases():
            url = f"{base.rstrip('/')}/{filename}"
            try:
                _download_file(url, dest)
                got = _file_md5(dest)
                if got != md5:
                    raise RuntimeError(f"md5 mismatch for {label}: expected {md5}, got {got}")
                break
            except (OSError, RuntimeError, urllib.error.URLError) as exc:
                errors.append(f"{filename} from {base}: {exc}")
                if dest.exists() and _file_md5(dest) != md5:
                    dest.unlink()
        else:
            detail = "\n".join(errors[-6:])
            raise RuntimeError(f"failed to download valid KMNIST {label} ({filename}). Recent errors:\n{detail}")
    return raw_dir


def _read_idx_images_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise RuntimeError(f"{path} is not an IDX image file (magic={magic})")
        data = np.frombuffer(f.read(), dtype=np.uint8).copy()
    expected = count * rows * cols
    if data.size != expected:
        raise RuntimeError(f"{path} has {data.size} pixels, expected {expected}")
    return data.reshape(count, rows, cols)


def _read_idx_labels_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise RuntimeError(f"{path} is not an IDX label file (magic={magic})")
        labels = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64, copy=True)
    if labels.size != count:
        raise RuntimeError(f"{path} has {labels.size} labels, expected {count}")
    return labels


def _extract_array_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(images[indices]).float().unsqueeze(1) / 255.0
    y = torch.from_numpy(labels[indices]).long()
    return x.flatten(1), y


def _balanced_indices(labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    if count <= 0:
        return np.asarray([], dtype=np.int64)
    if count >= len(labels):
        idx = np.arange(len(labels))
        rng.shuffle(idx)
        return idx

    classes = np.unique(labels)
    per_class = max(1, count // max(1, len(classes)))
    chosen: List[int] = []
    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        chosen.extend(cls_idx[:per_class].tolist())

    if len(chosen) < count:
        rest = np.setdiff1d(np.arange(len(labels)), np.asarray(chosen), assume_unique=False)
        rng.shuffle(rest)
        chosen.extend(rest[: count - len(chosen)].tolist())
    chosen_arr = np.asarray(chosen[:count], dtype=np.int64)
    rng.shuffle(chosen_arr)
    return chosen_arr


def _extract_tensor_dataset(ds: Any, indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(ds, "data") and hasattr(ds, "targets"):
        x = ds.data[indices]
        y = ds.targets
        if isinstance(y, list):
            y = torch.tensor(y, dtype=torch.long)
        y = y[indices].long()
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim == 4 and x.shape[-1] in {1, 3}:
            x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0
        return x.flatten(1), y

    xs: List[torch.Tensor] = []
    ys: List[int] = []
    for idx in indices:
        x, y = ds[int(idx)]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        xs.append(x.float().flatten())
        ys.append(int(y))
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def _make_fake_bundle(
    name: str,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    label_noise: float,
) -> DataBundle:
    rng = np.random.default_rng(seed)
    input_dim = 28 * 28
    num_classes = 10
    prototypes = rng.normal(0.0, 0.8, size=(num_classes, input_dim)).astype("float32")

    def make_split(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        y = np.arange(n, dtype=np.int64) % num_classes
        rng.shuffle(y)
        x = prototypes[y] + rng.normal(0.0, 0.75, size=(n, input_dim)).astype("float32")
        return torch.from_numpy(x), torch.from_numpy(y)

    x_train, y_clean = make_split(train_size)
    x_val, y_val = make_split(val_size)
    x_test, y_test = make_split(test_size)
    y_train = y_clean.clone()
    if label_noise > 0:
        mask = rng.random(train_size) < label_noise
        noisy = rng.integers(0, num_classes, size=train_size)
        noisy = np.where(noisy == y_clean.numpy(), (noisy + 1) % num_classes, noisy)
        y_train[torch.from_numpy(mask)] = torch.from_numpy(noisy[mask]).long()
    return DataBundle(
        name=name,
        input_dim=input_dim,
        num_classes=num_classes,
        x_train=x_train,
        y_train=y_train,
        y_train_clean=y_clean,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        label_noise=label_noise,
        used_fake_data=True,
    )


def _load_kmnist_bundle(
    dataset_name: str,
    *,
    data_root: Path,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    label_noise: float,
    download: bool,
) -> DataBundle:
    raw_dir = _ensure_kmnist_raw(data_root, download=download)
    train_images = _read_idx_images_gz(raw_dir / "train-images-idx3-ubyte.gz")
    train_labels = _read_idx_labels_gz(raw_dir / "train-labels-idx1-ubyte.gz")
    test_images = _read_idx_images_gz(raw_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = _read_idx_labels_gz(raw_dir / "t10k-labels-idx1-ubyte.gz")

    total_train = min(len(train_labels), train_size + val_size)
    train_val_idx = _balanced_indices(train_labels, total_train, seed)
    train_idx = train_val_idx[:train_size]
    val_idx = train_val_idx[train_size : train_size + val_size]
    test_idx = _balanced_indices(test_labels, min(test_size, len(test_labels)), seed + 101)

    x_train, y_clean = _extract_array_dataset(train_images, train_labels, train_idx)
    x_val, y_val = _extract_array_dataset(train_images, train_labels, val_idx)
    x_test, y_test = _extract_array_dataset(test_images, test_labels, test_idx)

    mean = x_train.mean()
    std = x_train.std().clamp_min(1e-4)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    y_train = y_clean.clone()
    if label_noise > 0:
        rng = np.random.default_rng(seed + 2027)
        mask = rng.random(len(y_train)) < label_noise
        noisy = rng.integers(0, 10, size=len(y_train))
        clean_np = y_clean.numpy()
        noisy = np.where(noisy == clean_np, (noisy + 1) % 10, noisy)
        y_train[torch.from_numpy(mask)] = torch.from_numpy(noisy[mask]).long()

    return DataBundle(
        name="kmnist",
        input_dim=int(x_train.shape[1]),
        num_classes=10,
        x_train=x_train,
        y_train=y_train,
        y_train_clean=y_clean,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        label_noise=label_noise,
        used_fake_data=False,
    )


def load_vision_bundle(
    dataset_name: str,
    *,
    data_root: Path,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    label_noise: float = 0.0,
    download: bool = True,
    allow_fake_data: bool = False,
) -> DataBundle:
    canonical = dataset_name.lower().replace("_", "-")
    if canonical == "kmnist":
        try:
            return _load_kmnist_bundle(
                dataset_name,
                data_root=data_root,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                seed=seed,
                label_noise=label_noise,
                download=download,
            )
        except Exception:
            if not allow_fake_data:
                raise
            return _make_fake_bundle(dataset_name, train_size, val_size, test_size, seed, label_noise)

    if datasets is None:
        if allow_fake_data:
            return _make_fake_bundle(dataset_name, train_size, val_size, test_size, seed, label_noise)
        raise RuntimeError(f"torchvision import failed: {_TORCHVISION_IMPORT_ERROR}")

    cls_map = {
        "mnist": datasets.MNIST,
        "fashion": datasets.FashionMNIST,
        "fashion-mnist": datasets.FashionMNIST,
        "fmnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
        "cifar-10": datasets.CIFAR10,
        "cifar": datasets.CIFAR10,
    }
    if canonical not in cls_map:
        raise ValueError(f"unknown dataset {dataset_name!r}; expected MNIST/Fashion-MNIST/KMNIST/CIFAR10")

    ds_cls = cls_map[canonical]
    try:
        train_ds = ds_cls(root=str(data_root), train=True, download=download)
        test_ds = ds_cls(root=str(data_root), train=False, download=download)
    except Exception:
        if not allow_fake_data:
            raise
        return _make_fake_bundle(dataset_name, train_size, val_size, test_size, seed, label_noise)

    train_targets = train_ds.targets
    test_targets = test_ds.targets
    if isinstance(train_targets, list):
        train_targets_np = np.asarray(train_targets)
    else:
        train_targets_np = train_targets.detach().cpu().numpy()
    if isinstance(test_targets, list):
        test_targets_np = np.asarray(test_targets)
    else:
        test_targets_np = test_targets.detach().cpu().numpy()

    total_train = min(len(train_targets_np), train_size + val_size)
    train_val_idx = _balanced_indices(train_targets_np, total_train, seed)
    train_idx = train_val_idx[:train_size]
    val_idx = train_val_idx[train_size : train_size + val_size]
    test_idx = _balanced_indices(test_targets_np, min(test_size, len(test_targets_np)), seed + 101)

    x_train, y_clean = _extract_tensor_dataset(train_ds, train_idx)
    x_val, y_val = _extract_tensor_dataset(train_ds, val_idx)
    x_test, y_test = _extract_tensor_dataset(test_ds, test_idx)

    mean = x_train.mean()
    std = x_train.std().clamp_min(1e-4)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    y_train = y_clean.clone()
    if label_noise > 0:
        rng = np.random.default_rng(seed + 2027)
        mask = rng.random(len(y_train)) < label_noise
        noisy = rng.integers(0, 10, size=len(y_train))
        clean_np = y_clean.numpy()
        noisy = np.where(noisy == clean_np, (noisy + 1) % 10, noisy)
        y_train[torch.from_numpy(mask)] = torch.from_numpy(noisy[mask]).long()

    return DataBundle(
        name=canonical,
        input_dim=int(x_train.shape[1]),
        num_classes=10,
        x_train=x_train,
        y_train=y_train,
        y_train_clean=y_clean,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        label_noise=label_noise,
        used_fake_data=False,
    )


def iter_minibatches(n: int, batch_size: int, seed: int) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


class RBFDense(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, basis_count: int, *, bias: bool = True) -> None:
        super().__init__()
        centers = torch.linspace(-2.5, 2.5, basis_count)
        self.register_buffer("centers", centers)
        self.width = float((centers[1] - centers[0]).abs() * 1.4) if basis_count > 1 else 1.0
        scale = 1.0 / math.sqrt(max(1, in_dim * basis_count))
        self.coeff = nn.Parameter(torch.randn(out_dim, in_dim, basis_count) * scale)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)
        self.last_basis_mean: Optional[torch.Tensor] = None
        self.last_input: Optional[torch.Tensor] = None

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        z = (x.unsqueeze(-1) - self.centers) / self.width
        return torch.exp(-0.5 * z.square())

    def basis_and_derivative(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        basis = self.basis(x)
        deriv = -((x.unsqueeze(-1) - self.centers) / (self.width**2)) * basis
        return basis, deriv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = self.basis(x)
        self.last_input = x.detach()
        self.last_basis_mean = basis.detach().mean(dim=(0, 1))
        out = torch.einsum("bik,oik->bo", basis, self.coeff)
        if self.bias is not None:
            out = out + self.bias
        return out


class ResidualKANBlock(nn.Module):
    def __init__(self, dim: int, basis_count: int, alpha_init: float, alpha_mode: str = "learnable") -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.kan = RBFDense(dim, dim, basis_count)
        mode = alpha_mode.lower().replace("-", "_")
        if mode == "learnable":
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        elif mode == "fixed1":
            self.register_buffer("alpha", torch.tensor(1.0))
        elif mode == "fixed_init":
            self.register_buffer("alpha", torch.tensor(float(alpha_init)))
        else:
            raise ValueError(f"unknown alpha_mode: {alpha_mode}")
        self.branch_scale = 1.0

    def forward(self, h: torch.Tensor, *, disable_kan: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if disable_kan:
            return h, torch.zeros_like(h)
        z = self.norm(h)
        branch = self.alpha * float(self.branch_scale) * self.kan(z)
        return h + branch, branch


class DGKANClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dim: int = 64,
        depth: int = 4,
        basis_count: int = 16,
        alpha_init: float = 1.5,
        alpha_mode: str = "learnable",
        head_hidden: int = 0,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.blocks = nn.ModuleList(
            [ResidualKANBlock(hidden_dim, basis_count, alpha_init, alpha_mode=alpha_mode) for _ in range(depth)]
        )
        if head_hidden > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, head_hidden),
                nn.SiLU(),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

    def set_branch_scale(self, scale: float) -> None:
        for block in self.blocks:
            block.branch_scale = float(scale)

    def forward(
        self,
        x: torch.Tensor,
        *,
        disable_kan: bool = False,
        return_branch: bool = False,
    ) -> Any:
        h = self.stem(x)
        branch_ratios: List[torch.Tensor] = []
        for block in self.blocks:
            prev = h
            h, branch = block(h, disable_kan=disable_kan)
            ratio = branch.norm(dim=-1).mean() / (prev.norm(dim=-1).mean().clamp_min(1e-6))
            branch_ratios.append(ratio)
        logits = self.head(h)
        if return_branch:
            if branch_ratios:
                return logits, torch.stack(branch_ratios).mean()
            return logits, torch.tensor(0.0, device=x.device)
        return logits

    def kan_layers(self) -> Iterable[RBFDense]:
        for block in self.blocks:
            yield block.kan


class ConvStem(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if input_dim == 3 * 32 * 32:
            self.channels, self.height, self.width = 3, 32, 32
        elif input_dim == 28 * 28:
            self.channels, self.height, self.width = 1, 28, 28
        else:
            side = int(math.sqrt(input_dim))
            self.channels, self.height, self.width = 1, side, side
        self.net = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.view(x.shape[0], self.channels, self.height, self.width)
        h = self.net(x)
        return h.mean(dim=(2, 3))


class ConvStemDGKANClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dim: int = 96,
        depth: int = 4,
        basis_count: int = 24,
        alpha_init: float = 1.5,
        alpha_mode: str = "fixed1",
    ) -> None:
        super().__init__()
        self.stem = ConvStem(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualKANBlock(hidden_dim, basis_count, alpha_init, alpha_mode=alpha_mode) for _ in range(depth)]
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

    def set_branch_scale(self, scale: float) -> None:
        for block in self.blocks:
            block.branch_scale = float(scale)

    def forward(self, x: torch.Tensor, *, disable_kan: bool = False, return_branch: bool = False) -> Any:
        h = self.stem(x)
        branch_ratios: List[torch.Tensor] = []
        for block in self.blocks:
            prev = h
            h, branch = block(h, disable_kan=disable_kan)
            branch_ratios.append(branch.norm(dim=-1).mean() / prev.norm(dim=-1).mean().clamp_min(1e-6))
        logits = self.head(h)
        if return_branch:
            return logits, torch.stack(branch_ratios).mean() if branch_ratios else torch.tensor(0.0, device=x.device)
        return logits

    def kan_layers(self) -> Iterable[RBFDense]:
        for block in self.blocks:
            yield block.kan


def _load_kat_group(*, backend: str, device: torch.device) -> Any:
    third_party = Path(__file__).resolve().parents[1] / "third_party" / "rational_kat_cu"
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
    from kat_rational import KAT_Group, KAT_Group_Torch  # type: ignore

    key = backend.lower().replace("-", "_")
    if key in {"torch", "pytorch", "fallback"} or device.type != "cuda":
        return KAT_Group_Torch
    return KAT_Group


def rational_denominator_curve(x: torch.Tensor, weight_denominator: torch.Tensor) -> torch.Tensor:
    b = weight_denominator.detach()
    powers = torch.arange(1, b.shape[-1] + 1, device=x.device, dtype=x.dtype)
    ax = x.detach().abs().to(dtype=x.dtype).reshape(-1, 1)
    basis = ax.pow(powers.view(1, -1))
    return 1.0 + basis @ b.abs().to(device=x.device, dtype=x.dtype).T


def rational_denominator_values(x: torch.Tensor, weight_denominator: torch.Tensor) -> torch.Tensor:
    b = weight_denominator.detach()
    groups = b.shape[0]
    dim = x.shape[-1]
    d_per_group = max(1, dim // max(1, groups))
    idx = torch.arange(dim, device=x.device).clamp(max=groups * d_per_group - 1) // d_per_group
    idx = idx.clamp(max=groups - 1)
    coeff = b.abs().to(device=x.device, dtype=x.dtype)[idx]
    powers = torch.arange(1, b.shape[-1] + 1, device=x.device, dtype=x.dtype)
    basis = x.detach().abs().unsqueeze(-1).pow(powers.view(1, 1, -1))
    return 1.0 + (basis * coeff.unsqueeze(0)).sum(dim=-1)


def rational_forward_from_params(x: torch.Tensor, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    a = numerator.reshape(-1).to(device=x.device, dtype=x.dtype)
    b = denominator.reshape(-1).to(device=x.device, dtype=x.dtype)
    xp = torch.stack([x.pow(i) for i in range(a.numel())], dim=-1)
    ax = x.abs()
    bp = torch.stack([ax.pow(i + 1) for i in range(b.numel())], dim=-1)
    p = (xp * a.view(1, -1)).sum(dim=-1)
    q = 1.0 + (bp * b.abs().view(1, -1)).sum(dim=-1)
    return p / q.clamp_min(1e-12)


def rational_derivative_stats(grid: torch.Tensor, numerator: torch.Tensor, denominator: torch.Tensor) -> Tuple[float, float]:
    vals1: List[torch.Tensor] = []
    vals2: List[torch.Tensor] = []
    num = numerator.detach().reshape(-1)
    den = denominator.detach()
    with torch.enable_grad():
        for group_idx in range(den.shape[0]):
            x = grid.detach().clone().requires_grad_(True)
            y = rational_forward_from_params(x, num, den[group_idx])
            dy = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
            ddy = torch.autograd.grad(dy.sum(), x, create_graph=False)[0]
            vals1.append(dy.detach().abs().flatten())
            vals2.append(ddy.detach().abs().flatten())
    d1 = torch.cat(vals1) if vals1 else torch.tensor([float("nan")], device=grid.device)
    d2 = torch.cat(vals2) if vals2 else torch.tensor([float("nan")], device=grid.device)
    return float(torch.quantile(d1.float(), 0.95).detach().cpu()), float(torch.quantile(d2.float(), 0.95).detach().cpu())


class RationalKANBranch(nn.Module):
    def __init__(self, dim: int, *, groups: int, mode: str, backend: str) -> None:
        super().__init__()
        self.groups = groups
        self.mode = mode
        self.backend = backend
        self.act: Optional[nn.Module] = None
        self.linear = nn.Linear(dim, dim)
        self.last_input: Optional[torch.Tensor] = None

    def _ensure_act(self, x: torch.Tensor) -> nn.Module:
        if self.act is None:
            cls = _load_kat_group(backend=self.backend, device=x.device)
            if cls.__name__ == "KAT_Group":
                self.act = cls(num_groups=self.groups, mode=self.mode, device=x.device.type).to(x.device)
            else:
                self.act = cls(num_groups=self.groups, mode=self.mode).to(x.device)
            self.add_module("kat_group", self.act)
        return self.act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach()
        act = self._ensure_act(x)
        y = act(x.unsqueeze(1)).squeeze(1)
        return self.linear(y)

    def denominator_safety(self) -> Dict[str, float]:
        act = self.act
        if act is None or not hasattr(act, "weight_denominator"):
            return {
                "denominator_min": float("nan"),
                "denominator_p01": float("nan"),
                "den_actual_min_batch": float("nan"),
                "den_actual_p01_batch": float("nan"),
                "den_actual_median_batch": float("nan"),
                "den_actual_max_batch": float("nan"),
                "den_actual_condition_batch": float("nan"),
                "den_actual_min_grid": float("nan"),
                "den_actual_p01_grid": float("nan"),
                "r_prime_p95": float("nan"),
                "r_double_prime_p95": float("nan"),
            }
        raw_den = getattr(act, "weight_denominator").detach()
        den = 1.0 + raw_den.abs().flatten()
        out = {
            "denominator_min": float(den.min().detach().cpu()),
            "denominator_p01": float(torch.quantile(den.float(), 0.01).detach().cpu()),
        }
        if self.last_input is not None:
            actual = rational_denominator_values(self.last_input.to(raw_den.device), raw_den)
            flat = actual.detach().flatten().float()
            out.update(
                {
                    "den_actual_min_batch": float(flat.min().detach().cpu()),
                    "den_actual_p01_batch": float(torch.quantile(flat, 0.01).detach().cpu()),
                    "den_actual_median_batch": float(torch.quantile(flat, 0.50).detach().cpu()),
                    "den_actual_max_batch": float(flat.max().detach().cpu()),
                    "den_actual_condition_batch": float((flat.max() / flat.min().clamp_min(1e-12)).detach().cpu()),
                }
            )
        grid = torch.linspace(-2.5, 2.5, 256, device=raw_den.device)
        grid_den = rational_denominator_curve(grid, raw_den)
        grid_flat = grid_den.detach().flatten().float()
        out["den_actual_min_grid"] = float(grid_flat.min().detach().cpu())
        out["den_actual_p01_grid"] = float(torch.quantile(grid_flat, 0.01).detach().cpu())
        if hasattr(act, "weight_numerator"):
            numerator = getattr(act, "weight_numerator").detach()
            r1, r2 = rational_derivative_stats(grid, numerator, raw_den)
            out["r_prime_p95"] = r1
            out["r_double_prime_p95"] = r2
        return out


class ResidualRationalKANBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        alpha_init: float,
        alpha_mode: str,
        groups: int,
        mode: str,
        backend: str,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.kan = RationalKANBranch(dim, groups=groups, mode=mode, backend=backend)
        key = alpha_mode.lower().replace("-", "_")
        if key == "learnable":
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        elif key == "fixed1":
            self.register_buffer("alpha", torch.tensor(1.0))
        elif key == "fixed_init":
            self.register_buffer("alpha", torch.tensor(float(alpha_init)))
        else:
            raise ValueError(f"unknown alpha_mode: {alpha_mode}")
        self.branch_scale = 1.0

    def forward(self, h: torch.Tensor, *, disable_kan: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if disable_kan:
            return h, torch.zeros_like(h)
        branch = self.alpha * float(self.branch_scale) * self.kan(self.norm(h))
        return h + branch, branch


class RationalDGKANClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dim: int = 64,
        depth: int = 4,
        alpha_init: float = 1.5,
        alpha_mode: str = "fixed1",
        head_hidden: int = 0,
        rational_groups: int = 8,
        rational_mode: str = "swish",
        rational_backend: str = "triton",
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.blocks = nn.ModuleList(
            [
                ResidualRationalKANBlock(
                    hidden_dim,
                    alpha_init=alpha_init,
                    alpha_mode=alpha_mode,
                    groups=rational_groups,
                    mode=rational_mode,
                    backend=rational_backend,
                )
                for _ in range(depth)
            ]
        )
        if head_hidden > 0:
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, head_hidden),
                nn.SiLU(),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

    def set_branch_scale(self, scale: float) -> None:
        for block in self.blocks:
            block.branch_scale = float(scale)

    def forward(self, x: torch.Tensor, *, disable_kan: bool = False, return_branch: bool = False) -> Any:
        h = self.stem(x)
        branch_ratios: List[torch.Tensor] = []
        for block in self.blocks:
            prev = h
            h, branch = block(h, disable_kan=disable_kan)
            branch_ratios.append(branch.norm(dim=-1).mean() / prev.norm(dim=-1).mean().clamp_min(1e-6))
        logits = self.head(h)
        if return_branch:
            return logits, torch.stack(branch_ratios).mean() if branch_ratios else torch.tensor(0.0, device=x.device)
        return logits

    def rational_safety(self) -> Dict[str, float]:
        stores: Dict[str, List[float]] = {}
        for block in self.blocks:
            stats = block.kan.denominator_safety()
            for key, value in stats.items():
                if math.isfinite(float(value)):
                    stores.setdefault(key, []).append(float(value))
        min_keys = {
            "denominator_min",
            "denominator_p01",
            "den_actual_min_batch",
            "den_actual_p01_batch",
            "den_actual_min_grid",
            "den_actual_p01_grid",
        }
        out: Dict[str, float] = {}
        for key, values in stores.items():
            out[f"rational_{key}"] = float(min(values)) if key in min_keys else float(np.mean(values))
        defaults = [
            "rational_denominator_min",
            "rational_denominator_p01",
            "rational_den_actual_min_batch",
            "rational_den_actual_p01_batch",
            "rational_den_actual_condition_batch",
            "rational_den_actual_min_grid",
            "rational_den_actual_p01_grid",
            "rational_r_prime_p95",
            "rational_r_double_prime_p95",
        ]
        for key in defaults:
            out.setdefault(key, float("nan"))
        return out


class FixedNorm(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).square().mean(dim=-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps)


class PureResidualKANBlock(nn.Module):
    def __init__(self, dim: int, basis_count: int, *, alpha_init: float = 1.0, alpha_mode: str = "fixed1") -> None:
        super().__init__()
        self.norm = FixedNorm()
        self.kan = RBFDense(dim, dim, basis_count, bias=False)
        mode = alpha_mode.lower().replace("-", "_")
        if mode == "learnable":
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        elif mode == "fixed1":
            self.register_buffer("alpha", torch.tensor(1.0))
        elif mode == "fixed_init":
            self.register_buffer("alpha", torch.tensor(float(alpha_init)))
        else:
            raise ValueError(f"unknown alpha_mode: {alpha_mode}")
        self.branch_scale = 1.0

    def forward(self, h: torch.Tensor, *, disable_kan: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if disable_kan:
            return h, torch.zeros_like(h)
        branch = self.alpha * float(self.branch_scale) * self.kan(self.norm(h))
        return h + branch, branch


class PureKANClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_dim: int = 64,
        depth: int = 2,
        basis_count: int = 16,
        alpha_init: float = 1.0,
        alpha_mode: str = "fixed1",
    ) -> None:
        super().__init__()
        self.input_kan = RBFDense(input_dim, hidden_dim, basis_count, bias=False)
        self.blocks = nn.ModuleList(
            [
                PureResidualKANBlock(hidden_dim, basis_count, alpha_init=alpha_init, alpha_mode=alpha_mode)
                for _ in range(depth)
            ]
        )
        self.output_norm = FixedNorm()
        self.output_kan = RBFDense(hidden_dim, num_classes, basis_count, bias=False)

    def set_branch_scale(self, scale: float) -> None:
        for block in self.blocks:
            block.branch_scale = float(scale)

    def forward(self, x: torch.Tensor, *, disable_kan: bool = False, return_branch: bool = False) -> Any:
        h = self.input_kan(x)
        branch_ratios: List[torch.Tensor] = []
        for block in self.blocks:
            prev = h
            h, branch = block(h, disable_kan=disable_kan)
            branch_ratios.append(branch.norm(dim=-1).mean() / prev.norm(dim=-1).mean().clamp_min(1e-6))
        logits = self.output_kan(self.output_norm(h))
        if return_branch:
            return logits, torch.stack(branch_ratios).mean() if branch_ratios else torch.tensor(0.0, device=x.device)
        return logits

    def kan_layers(self) -> Iterable[RBFDense]:
        yield self.input_kan
        for block in self.blocks:
            yield block.kan
        yield self.output_kan


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, depth: int = 4) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()])
        layers.extend([nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.net(x)


def coefficient_named_params(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    out: List[Tuple[str, nn.Parameter]] = []
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        if isinstance(module, RBFDense):
            out.append((f"{prefix}coeff", module.coeff))
            seen.add(id(module.coeff))
        if hasattr(module, "weight_numerator") and hasattr(module, "weight_denominator"):
            numerator = getattr(module, "weight_numerator")
            denominator = getattr(module, "weight_denominator")
            if isinstance(numerator, nn.Parameter) and id(numerator) not in seen:
                out.append((f"{prefix}weight_numerator", numerator))
                seen.add(id(numerator))
            if isinstance(denominator, nn.Parameter) and id(denominator) not in seen:
                out.append((f"{prefix}weight_denominator", denominator))
                seen.add(id(denominator))
    return out


def non_coefficient_params(model: nn.Module) -> List[nn.Parameter]:
    coeff_ids = {id(p) for _, p in coefficient_named_params(model)}
    return [p for p in model.parameters() if id(p) not in coeff_ids]


ADAMW_METHOD_KEYS = {
    "adamw",
    "mlp_adamw",
    "rational_adamw",
    "rational_dgkan_adamw",
    "kat_adamw",
    "convstem_dgkan_adamw",
    "purekan_adamw",
    "pure_kan_adamw",
}


@dataclass
class TrainConfig:
    dataset: str = "Fashion-MNIST"
    method: str = "AdamW"
    optimizer_method: str = ""
    seed: int = 0
    device: str = "auto"
    data_root: Path = Path("data")
    download: bool = True
    allow_fake_data: bool = False
    train_size: int = 6000
    val_size: int = 1000
    test_size: int = 1000
    label_noise: float = 0.0
    epochs: int = 8
    batch_size: int = 256
    eval_batch_size: int = 512
    audit_batch_size: int = 64
    hidden_dim: int = 64
    depth: int = 4
    basis_count: int = 16
    alpha_init: float = 1.5
    alpha_mode: str = "learnable"
    head_hidden: int = 0
    model_type: str = "rbf_dgkan"
    rational_backend: str = "triton"
    rational_groups: int = 8
    rational_mode: str = "swish"
    rational_branch_functional: bool = False
    lr: float = 1e-3
    weight_decay: float = 1e-4
    coeff_lr: float = 0.05
    rest_lr: float = 1e-3
    rest_weight_decay: float = 1e-4
    lr_schedule: str = "none"
    lr_final_mult: float = 1.0
    rest_lr_schedule: str = "none"
    rest_lr_final_mult: float = 1.0
    coeff_lr_schedule: str = "none"
    coeff_lr_decay_final_mult: float = 1.0
    lr_decay_start_frac: float = 0.0
    warmup_frac: float = 0.25
    sobolev_alpha: float = 0.15
    sobolev_beta: float = 0.02
    rho: float = 1e-3
    metric_mode: str = "grid"
    lambda_early: float = 0.0
    lambda_final: float = 0.0
    pulse_frac: float = 0.25
    anchor_decay: str = "linear"
    momentum_mu: float = 0.0
    momentum_start_frac: float = 0.05
    reset_on_reanchor: bool = True
    trust_radius: float = 0.15
    trust_mode: str = "safety_only"
    prox_lambda: float = 0.0
    prox_start_frac: float = 0.5
    branch_schedule: str = "none"
    branch_boost: float = 1.0
    coeff_lr_boost: float = 1.0
    coeff_lr_final_mult: float = 1.0
    branch_final_scale: float = 1.0
    branch_active_frac: float = 0.25
    branch_max_active_frac: float = 0.35
    geometry_phi_high: float = 0.055
    geometry_jac_high: float = 4.5
    geometry_branch_high: float = 0.30
    geometry_shrink_factor: float = 0.85
    geometry_min_branch_scale: float = 0.7
    geometry_min_epochs: int = 2
    snr_enabled: bool = False
    snr_grouping: str = "edge"
    snr_update_interval: int = 4
    snr_sample_frac: float = 0.5
    snr_percentile: float = 60.0
    snr_ema_beta: float = 0.95
    snr_lr_min: float = 0.5
    snr_lr_max: float = 1.5
    spectral_lambda: float = 0.0
    spectral_start_frac: float = 0.7
    credit_mode: str = "analytic"
    identity_credit_scale: float = 0.35
    gafu_v3_enabled: bool = False
    v3_metric_active: str = "basis_diag_gram"
    v3_metric_transition: str = "diag_to_full_sobolev"
    v3_metric_geometry: str = "full_sobolev_gram"
    v3_alpha_fast: float = 0.0
    v3_beta_fast: float = 0.0
    v3_alpha_geo: float = 0.15
    v3_beta_geo: float = 0.02
    v3_gram_rho: float = 1e-3
    v3_gram_grid_size: int = 512
    v3_transition_frac: float = 0.10
    v3_transition_min_steps: int = 50
    v3_transition_max_steps: int = 300
    v3_transition_curve: str = "cosine"
    v3_phase_mode: str = "smooth"
    unified_optimizer_mode: str = "hybrid"
    kan_grad_transform: str = "none"
    kan_precondition_before_adam: bool = False
    kan_precondition_after_adam: bool = False
    kan_precond_grad_normalize: bool = False
    uo_trust_radius: float = 0.0
    uo_record_moment_audit: bool = False
    afu_kan_bias: bool = False
    afu_head: bool = False
    afu_stem: bool = False
    afu_ln_scope: str = "none"
    afu_head_lr_mult: float = 0.5
    afu_stem_lr_mult: float = 0.3
    afu_ln_lr_mult: float = 0.3
    afu_bias_lr_mult: float = 0.5
    afu_cov_ema_beta: float = 0.95
    afu_head_rho: float = 1e-2
    afu_stem_rho: float = 1e-2
    afu_ln_rho: float = 1e-2
    afu_bias_rho: float = 1e-2
    afu_max_update_ratio: float = 0.05
    pure_input_metric: str = "phase"
    pure_block_metric: str = "phase"
    pure_output_metric: str = "phase"
    pure_input_lr_mult: float = 1.0
    pure_block_lr_mult: float = 1.0
    pure_output_lr_mult: float = 1.0
    pure_input_trust_radius: float = 0.0
    pure_block_trust_radius: float = 0.0
    pure_output_trust_radius: float = 0.0
    notes: str = ""


@dataclass
class RuntimeState:
    momentum: Dict[str, torch.Tensor] = field(default_factory=dict)
    snr_mean: Dict[str, torch.Tensor] = field(default_factory=dict)
    snr_sq: Dict[str, torch.Tensor] = field(default_factory=dict)
    snr_mult: Dict[str, torch.Tensor] = field(default_factory=dict)
    trust_clip_count: int = 0
    coeff_update_count: int = 0
    snr_update_time: float = 0.0
    branch_switched: bool = False
    branch_switch_step: int = -1
    branch_switch_reason: str = ""
    shrink_events: int = 0
    current_branch_scale: float = 1.0
    coeff_lr_multiplier: float = 1.0
    last_data_lambda: float = 0.0
    reanchor_reset_done: bool = False
    phase: str = "ACTIVE"
    phase_enter_step: int = 0
    transition_length_steps: int = 0
    transition_start_branch_scale: float = 1.0
    transition_start_coeff_lr_multiplier: float = 1.0
    current_metric_mix: float = 0.0
    metric_mix_sum: float = 0.0
    metric_mix_count: int = 0
    sobolev_gram_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metric_build_time: float = 0.0
    precond_solve_time: float = 0.0
    metric_condition_active: float = float("nan")
    metric_condition_geometry: float = float("nan")
    metric_eig_min_active: float = float("nan")
    metric_eig_max_active: float = float("nan")
    metric_eig_min_geometry: float = float("nan")
    metric_eig_max_geometry: float = float("nan")
    diagfast_condition: float = float("nan")
    fullgeo_condition: float = float("nan")
    diagfast_eig_min: float = float("nan")
    fullgeo_eig_min: float = float("nan")
    diagfast_eig_max: float = float("nan")
    fullgeo_eig_max: float = float("nan")
    precond_direction_norms: List[float] = field(default_factory=list)
    raw_grad_norms: List[float] = field(default_factory=list)
    update_metric_norms: List[float] = field(default_factory=list)
    update_over_coeff_norms: List[float] = field(default_factory=list)
    phase_trace_first20: List[str] = field(default_factory=list)
    metric_trace_first20: List[str] = field(default_factory=list)
    phase_seen_counts: Dict[str, int] = field(default_factory=dict)
    metric_seen_counts: Dict[str, int] = field(default_factory=dict)
    pure_seen_param_names: set[str] = field(default_factory=set)
    pure_seen_param_numel: int = 0
    pure_role_metric_seen: Dict[str, List[str]] = field(default_factory=dict)
    pure_role_update_norms: Dict[str, List[float]] = field(default_factory=dict)
    pure_role_update_over_param: Dict[str, List[float]] = field(default_factory=dict)
    pure_role_raw_grad_norms: Dict[str, List[float]] = field(default_factory=dict)
    pure_role_precond_norms: Dict[str, List[float]] = field(default_factory=dict)
    pure_role_cos_raw_precond: Dict[str, List[float]] = field(default_factory=dict)
    pure_role_metric_conditions: Dict[str, List[float]] = field(default_factory=dict)
    direction_cos_diagfast_fullgeo: List[float] = field(default_factory=list)
    direction_norm_ratio_fullgeo_diagfast: List[float] = field(default_factory=list)
    direction_norm_ratio_current_diagfast: List[float] = field(default_factory=list)
    direction_metric_norm_diagfast: List[float] = field(default_factory=list)
    direction_metric_norm_fullgeo: List[float] = field(default_factory=list)
    direction_metric_norm_current: List[float] = field(default_factory=list)
    uo_m: Dict[str, torch.Tensor] = field(default_factory=dict)
    uo_v: Dict[str, torch.Tensor] = field(default_factory=dict)
    uo_t: Dict[str, int] = field(default_factory=dict)
    uo_last_raw_grad: Optional[torch.Tensor] = None
    uo_last_precond_grad: Optional[torch.Tensor] = None
    uo_last_adam_direction: Optional[torch.Tensor] = None
    uo_kan_raw_grad_norms: List[float] = field(default_factory=list)
    uo_kan_precond_grad_norms: List[float] = field(default_factory=list)
    uo_kan_precond_over_raw_norms: List[float] = field(default_factory=list)
    uo_rest_grad_norms: List[float] = field(default_factory=list)
    uo_kan_update_norms: List[float] = field(default_factory=list)
    uo_rest_update_norms: List[float] = field(default_factory=list)
    uo_kan_update_over_param_norms: List[float] = field(default_factory=list)
    uo_rest_update_over_param_norms: List[float] = field(default_factory=list)
    uo_kan_update_over_rest_update: List[float] = field(default_factory=list)
    uo_adam_m_norm_kan: List[float] = field(default_factory=list)
    uo_adam_v_norm_kan: List[float] = field(default_factory=list)
    uo_moment_cos_raw_precond: List[float] = field(default_factory=list)
    uo_moment_cos_precond_current: List[float] = field(default_factory=list)
    uo_moment_cos_raw_current: List[float] = field(default_factory=list)
    uo_moment_amplification_ratio: List[float] = field(default_factory=list)
    uo_bad_step_count: int = 0
    uo_update_count: int = 0
    rational_num_update_norms: List[float] = field(default_factory=list)
    rational_den_update_norms: List[float] = field(default_factory=list)
    rational_num_den_update_cos: List[float] = field(default_factory=list)
    rational_num_den_update_ratio: List[float] = field(default_factory=list)
    rational_branch_linear_update_norms: List[float] = field(default_factory=list)
    rational_branch_linear_update_over_param: List[float] = field(default_factory=list)
    rational_branch_linear_cov_conditions: List[float] = field(default_factory=list)
    rational_branch_linear_cos_raw_precond: List[float] = field(default_factory=list)
    afu_cov_ema: Dict[str, torch.Tensor] = field(default_factory=dict)
    afu_diag_ema: Dict[str, torch.Tensor] = field(default_factory=dict)
    afu_update_norms: Dict[str, List[float]] = field(default_factory=dict)
    afu_update_over_param: Dict[str, List[float]] = field(default_factory=dict)
    afu_raw_grad_norms: Dict[str, List[float]] = field(default_factory=dict)
    afu_precond_grad_norms: Dict[str, List[float]] = field(default_factory=dict)
    afu_cos_raw_precond: Dict[str, List[float]] = field(default_factory=dict)
    afu_metric_conditions: Dict[str, List[float]] = field(default_factory=dict)
    afu_metric_mins: Dict[str, List[float]] = field(default_factory=dict)
    afu_metric_maxs: Dict[str, List[float]] = field(default_factory=dict)
    afu_update_share: Dict[str, List[float]] = field(default_factory=dict)
    afu_step_group_updates: Dict[str, float] = field(default_factory=dict)
    afu_clip_count: int = 0
    afu_update_count: int = 0
    afu_head_cov_time: float = 0.0
    afu_head_solve_time: float = 0.0
    afu_stem_metric_time: float = 0.0
    afu_ln_metric_time: float = 0.0


def scheduled_lr_factor(schedule: str, progress: float, final_mult: float, start_frac: float = 0.0) -> float:
    key = schedule.lower().replace("-", "_")
    if key in {"", "none", "constant"}:
        return 1.0
    if progress <= start_frac:
        return 1.0
    t = (progress - start_frac) / max(1e-6, 1.0 - start_frac)
    t = min(1.0, max(0.0, t))
    if key == "linear":
        soft = 1.0 - t
    elif key == "cosine":
        soft = 0.5 + 0.5 * math.cos(math.pi * t)
    else:
        raise ValueError(f"unknown lr schedule: {schedule}")
    return float(final_mult + (1.0 - final_mult) * soft)


def set_optimizer_lr(opt: Optional[torch.optim.Optimizer], base_lr: float, factor: float) -> None:
    if opt is None:
        return
    for group in opt.param_groups:
        group["lr"] = base_lr * factor


def set_optimizer_group_lrs(opt: Optional[torch.optim.Optimizer], cfg: TrainConfig, state: RuntimeState, progress: float) -> None:
    if opt is None:
        return
    rest_factor = scheduled_lr_factor(
        cfg.rest_lr_schedule,
        progress,
        cfg.rest_lr_final_mult,
        cfg.lr_decay_start_frac,
    )
    coeff_factor = scheduled_lr_factor(
        cfg.coeff_lr_schedule,
        progress,
        cfg.coeff_lr_decay_final_mult,
        cfg.lr_decay_start_frac,
    )
    lr_factor = scheduled_lr_factor(cfg.lr_schedule, progress, cfg.lr_final_mult, cfg.lr_decay_start_frac)
    for group in opt.param_groups:
        role = str(group.get("role", "all"))
        if role == "coeff":
            group["lr"] = cfg.coeff_lr * state.coeff_lr_multiplier * coeff_factor
        elif role == "rest":
            group["lr"] = cfg.rest_lr * rest_factor
        else:
            group["lr"] = cfg.lr * lr_factor


def _flat_cat(values: Sequence[torch.Tensor], *, device: Optional[torch.device] = None) -> torch.Tensor:
    if not values:
        if device is None:
            return torch.empty(0)
        return torch.empty(0, device=device)
    return torch.cat([value.detach().flatten().float() for value in values])


def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    a = a.detach().flatten().float()
    b = b.detach().flatten().float()
    if a.numel() != b.numel():
        n = min(a.numel(), b.numel())
        a = a[:n]
        b = b[:n]
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((torch.dot(a, b) / denom).detach().cpu())


def _pure_param_role(name: str) -> str:
    if name.startswith("input_kan."):
        return "input"
    if name.startswith("output_kan."):
        return "output"
    if ".kan." in name or name.startswith("blocks."):
        return "block"
    return "other"


def _pure_role_metric(cfg: TrainConfig, role: str, phase_metric: str) -> str:
    value = getattr(cfg, f"pure_{role}_metric", "phase")
    key = str(value or "phase").lower().replace("-", "_")
    if key in {"", "phase", "auto"}:
        return phase_metric
    if key == "diag":
        return "basis_diag_gram"
    if key == "full":
        return "full_sobolev_gram"
    return str(value)


def _pure_role_lr_mult(cfg: TrainConfig, role: str) -> float:
    return float(getattr(cfg, f"pure_{role}_lr_mult", 1.0))


def _pure_role_trust_radius(cfg: TrainConfig, role: str) -> float:
    value = float(getattr(cfg, f"pure_{role}_trust_radius", 0.0))
    return value if value > 0 else cfg.trust_radius


def _append_role_value(store: Dict[str, List[float]], role: str, value: float) -> None:
    store.setdefault(role, []).append(float(value))


def _param_group_norms(params: Sequence[nn.Parameter]) -> Tuple[float, float]:
    grad_sq = torch.tensor(0.0)
    param_sq = torch.tensor(0.0)
    has_device = False
    for p in params:
        if not has_device:
            grad_sq = grad_sq.to(device=p.device)
            param_sq = param_sq.to(device=p.device)
            has_device = True
        if p.grad is not None:
            grad_sq = grad_sq + p.grad.detach().float().square().sum()
        param_sq = param_sq + p.detach().float().square().sum()
    return float(grad_sq.sqrt().detach().cpu()), float(param_sq.sqrt().detach().cpu())


def _snapshot_params(params: Sequence[nn.Parameter]) -> List[Tuple[nn.Parameter, torch.Tensor]]:
    return [(p, p.detach().clone()) for p in params]


def _record_param_updates(
    state: RuntimeState,
    coeff_before: Sequence[Tuple[nn.Parameter, torch.Tensor]],
    rest_before: Sequence[Tuple[nn.Parameter, torch.Tensor]],
) -> None:
    device = None
    for p, _ in list(coeff_before) + list(rest_before):
        device = p.device
        break
    coeff_update_sq = torch.tensor(0.0, device=device)
    coeff_param_sq = torch.tensor(0.0, device=device)
    rest_update_sq = torch.tensor(0.0, device=device)
    rest_param_sq = torch.tensor(0.0, device=device)
    for p, before in coeff_before:
        diff = (p.detach() - before).float()
        coeff_update_sq = coeff_update_sq + diff.square().sum()
        coeff_param_sq = coeff_param_sq + before.float().square().sum()
    for p, before in rest_before:
        diff = (p.detach() - before).float()
        rest_update_sq = rest_update_sq + diff.square().sum()
        rest_param_sq = rest_param_sq + before.float().square().sum()
    coeff_update = float(coeff_update_sq.sqrt().detach().cpu())
    rest_update = float(rest_update_sq.sqrt().detach().cpu())
    coeff_param = float(coeff_param_sq.sqrt().detach().cpu())
    rest_param = float(rest_param_sq.sqrt().detach().cpu())
    state.uo_kan_update_norms.append(coeff_update)
    state.uo_rest_update_norms.append(rest_update)
    state.uo_kan_update_over_param_norms.append(coeff_update / max(1e-12, coeff_param))
    state.uo_rest_update_over_param_norms.append(rest_update / max(1e-12, rest_param))
    state.uo_kan_update_over_rest_update.append(coeff_update / max(1e-12, rest_update))
    state.uo_update_count += 1
    if not math.isfinite(coeff_update) or not math.isfinite(rest_update):
        state.uo_bad_step_count += 1
    elif coeff_update / max(1e-12, coeff_param) > 0.05:
        state.uo_bad_step_count += 1


def _afu_scope_enabled(scope: str, *, target: str) -> bool:
    key = scope.lower().replace("-", "_")
    if key in {"all", "all_ln", "full"}:
        return True
    if key in {"head", "head_ln"}:
        return target == "head"
    if key in {"block", "block_ln"}:
        return target == "block"
    if key in {"stem", "stem_ln"}:
        return target == "stem"
    return False


def _afu_param_role(name: str, cfg: TrainConfig) -> str:
    if name.endswith("kan.coeff"):
        return "kan_coeff"
    if name.endswith("kan.bias"):
        return "kan_bias" if cfg.afu_kan_bias else "rest"
    if name.startswith("stem.0."):
        return "stem" if cfg.afu_stem else "rest"
    if name.startswith("stem.1."):
        return "ln" if _afu_scope_enabled(cfg.afu_ln_scope, target="stem") else "rest"
    if name.startswith("head."):
        # In the current h96/b24 experiments head.0 is LayerNorm and head.1 is Linear.
        if ".0." in name:
            return "ln" if _afu_scope_enabled(cfg.afu_ln_scope, target="head") else "rest"
        return "head" if cfg.afu_head else "rest"
    if ".norm." in name:
        return "ln" if _afu_scope_enabled(cfg.afu_ln_scope, target="block") else "rest"
    return "rest"


def _afu_functional_param_ids(model: nn.Module, cfg: TrainConfig) -> set[int]:
    ids: set[int] = set()
    if not isinstance(model, DGKANClassifier):
        return ids
    for name, p in model.named_parameters():
        role = _afu_param_role(name, cfg)
        if role in {"kan_bias", "head", "stem", "ln"}:
            ids.add(id(p))
    return ids


def _afu_rest_params(model: nn.Module, cfg: TrainConfig) -> List[nn.Parameter]:
    coeff_ids = {id(p) for _, p in coefficient_named_params(model)}
    functional_ids = _afu_functional_param_ids(model, cfg)
    return [p for p in model.parameters() if id(p) not in coeff_ids and id(p) not in functional_ids]


@torch.no_grad()
def _dgkan_activation_cache(model: DGKANClassifier, x: torch.Tensor) -> Dict[str, Any]:
    cache: Dict[str, Any] = {"stem_input": x.detach(), "head_linear_inputs": {}}
    h = model.stem[0](x)
    h = model.stem[1](h)
    h = model.stem[2](h)
    for block in model.blocks:
        h, _ = block(h)
    current = h
    for idx, module in enumerate(model.head):
        name = f"head.{idx}"
        if isinstance(module, nn.Linear):
            cache["head_linear_inputs"][name] = current.detach()
        current = module(current)
    return cache


def _afu_record_group(
    state: RuntimeState,
    group: str,
    *,
    raw_grad: torch.Tensor,
    precond_grad: torch.Tensor,
    update: torch.Tensor,
    param: torch.Tensor,
    condition: float = float("nan"),
    metric_min: float = float("nan"),
    metric_max: float = float("nan"),
) -> None:
    raw_norm = float(raw_grad.detach().float().norm().cpu())
    precond_norm = float(precond_grad.detach().float().norm().cpu())
    update_norm = float(update.detach().float().norm().cpu())
    param_norm = float(param.detach().float().norm().cpu())
    state.afu_update_norms.setdefault(group, []).append(update_norm)
    state.afu_update_over_param.setdefault(group, []).append(update_norm / max(1e-12, param_norm))
    state.afu_raw_grad_norms.setdefault(group, []).append(raw_norm)
    state.afu_precond_grad_norms.setdefault(group, []).append(precond_norm)
    state.afu_cos_raw_precond.setdefault(group, []).append(_safe_cos(raw_grad, precond_grad))
    state.afu_metric_conditions.setdefault(group, []).append(condition)
    state.afu_metric_mins.setdefault(group, []).append(metric_min)
    state.afu_metric_maxs.setdefault(group, []).append(metric_max)
    state.afu_step_group_updates[group] = state.afu_step_group_updates.get(group, 0.0) + update_norm
    state.afu_update_count += 1


def _afu_clip_update(update: torch.Tensor, param: torch.Tensor, cfg: TrainConfig, state: RuntimeState) -> torch.Tensor:
    if cfg.afu_max_update_ratio <= 0:
        return update
    update_norm = update.detach().float().norm()
    param_norm = param.detach().float().norm().clamp_min(1e-12)
    limit = float(cfg.afu_max_update_ratio) * param_norm
    if float(update_norm.cpu()) > float(limit.cpu()) > 0:
        state.afu_clip_count += 1
        return update * (limit / update_norm.clamp_min(1e-12))
    return update


def _afu_update_diag_param(
    name: str,
    p: nn.Parameter,
    cfg: TrainConfig,
    state: RuntimeState,
    *,
    group: str,
    lr: float,
    rho: float,
) -> None:
    if p.grad is None:
        return
    g = p.grad.detach()
    key = f"{group}:{name}"
    metric = state.afu_diag_ema.get(key)
    current = g.square()
    if metric is None:
        metric = current
    else:
        metric = cfg.afu_cov_ema_beta * metric + (1.0 - cfg.afu_cov_ema_beta) * current
    state.afu_diag_ema[key] = metric.detach().clone()
    precond = g / (metric + rho).sqrt().clamp_min(rho)
    update = -lr * precond
    update = _afu_clip_update(update, p.detach(), cfg, state)
    _afu_record_group(
        state,
        group,
        raw_grad=g,
        precond_grad=precond,
        update=update,
        param=p.detach(),
        condition=float((metric.max() + rho) / (metric.min() + rho)),
        metric_min=float((metric.min() + rho).detach().cpu()),
        metric_max=float((metric.max() + rho).detach().cpu()),
    )
    with torch.no_grad():
        p.add_(update)
    p.grad = None


def _afu_update_head_linear(
    module_name: str,
    module: nn.Linear,
    features: torch.Tensor,
    cfg: TrainConfig,
    state: RuntimeState,
    *,
    lr: float,
) -> None:
    if module.weight.grad is None:
        return
    start = time.perf_counter()
    h = features.detach().float()
    cov = (h.T @ h) / max(1, h.shape[0])
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    cov = cov + float(cfg.afu_head_rho) * eye
    key = f"head_cov:{module_name}"
    cached = state.afu_cov_ema.get(key)
    if cached is None:
        ema = cov
    else:
        ema = cfg.afu_cov_ema_beta * cached + (1.0 - cfg.afu_cov_ema_beta) * cov
    state.afu_cov_ema[key] = ema.detach().clone()
    eig = torch.linalg.eigvalsh(ema).clamp_min(1e-12)
    state.afu_head_cov_time += time.perf_counter() - start

    g = module.weight.grad.detach().float()
    start = time.perf_counter()
    precond = torch.linalg.solve(ema, g.T).T.to(dtype=module.weight.dtype)
    state.afu_head_solve_time += time.perf_counter() - start
    update = -lr * precond
    update = _afu_clip_update(update, module.weight.detach(), cfg, state)
    _afu_record_group(
        state,
        "head",
        raw_grad=module.weight.grad.detach(),
        precond_grad=precond,
        update=update,
        param=module.weight.detach(),
        condition=float((eig[-1] / eig[0]).detach().cpu()),
        metric_min=float(eig[0].detach().cpu()),
        metric_max=float(eig[-1].detach().cpu()),
    )
    with torch.no_grad():
        module.weight.add_(update)
    module.weight.grad = None
    if module.bias is not None and module.bias.grad is not None:
        _afu_update_diag_param(f"{module_name}.bias", module.bias, cfg, state, group="head", lr=lr, rho=cfg.afu_bias_rho)


def _afu_update_stem_linear(
    module: nn.Linear,
    stem_input: torch.Tensor,
    cfg: TrainConfig,
    state: RuntimeState,
    *,
    lr: float,
) -> None:
    if module.weight.grad is None:
        return
    start = time.perf_counter()
    x = stem_input.detach().float()
    diag = x.square().mean(dim=0) + float(cfg.afu_stem_rho)
    key = "stem_diag"
    cached = state.afu_diag_ema.get(key)
    if cached is None:
        ema = diag
    else:
        ema = cfg.afu_cov_ema_beta * cached + (1.0 - cfg.afu_cov_ema_beta) * diag
    state.afu_diag_ema[key] = ema.detach().clone()
    state.afu_stem_metric_time += time.perf_counter() - start
    g = module.weight.grad.detach()
    precond = g / ema.to(device=g.device, dtype=g.dtype).view(1, -1).clamp_min(cfg.afu_stem_rho)
    update = -lr * precond
    update = _afu_clip_update(update, module.weight.detach(), cfg, state)
    _afu_record_group(
        state,
        "stem",
        raw_grad=g,
        precond_grad=precond,
        update=update,
        param=module.weight.detach(),
        condition=float((ema.max() / ema.min().clamp_min(1e-12)).detach().cpu()),
        metric_min=float(ema.min().detach().cpu()),
        metric_max=float(ema.max().detach().cpu()),
    )
    with torch.no_grad():
        module.weight.add_(update)
    module.weight.grad = None
    if module.bias is not None and module.bias.grad is not None:
        _afu_update_diag_param("stem.0.bias", module.bias, cfg, state, group="stem", lr=lr, rho=cfg.afu_bias_rho)


def afu_nonkan_step(
    model: DGKANClassifier,
    cfg: TrainConfig,
    state: RuntimeState,
    *,
    xb: torch.Tensor,
    progress: float,
) -> None:
    state.afu_step_group_updates = {}
    rest_decay = scheduled_lr_factor(
        cfg.rest_lr_schedule,
        progress,
        cfg.rest_lr_final_mult,
        cfg.lr_decay_start_frac,
    )
    lr_base = cfg.rest_lr * rest_decay
    cache = _dgkan_activation_cache(model, xb)

    if cfg.afu_kan_bias:
        for idx, block in enumerate(model.blocks):
            _afu_update_diag_param(
                f"blocks.{idx}.kan.bias",
                block.kan.bias,
                cfg,
                state,
                group="kan_bias",
                lr=lr_base * cfg.afu_bias_lr_mult,
                rho=cfg.afu_bias_rho,
            )

    if cfg.afu_head:
        for idx, module in enumerate(model.head):
            if isinstance(module, nn.Linear):
                module_name = f"head.{idx}"
                features = cache["head_linear_inputs"].get(module_name)
                if features is not None:
                    _afu_update_head_linear(
                        module_name,
                        module,
                        features,
                        cfg,
                        state,
                        lr=lr_base * cfg.afu_head_lr_mult,
                    )

    if cfg.afu_stem and isinstance(model.stem[0], nn.Linear):
        _afu_update_stem_linear(model.stem[0], cache["stem_input"], cfg, state, lr=lr_base * cfg.afu_stem_lr_mult)

    if cfg.afu_ln_scope.lower().replace("-", "_") not in {"", "none"}:
        start = time.perf_counter()
        for name, p in model.named_parameters():
            if _afu_param_role(name, cfg) == "ln":
                _afu_update_diag_param(name, p, cfg, state, group="ln", lr=lr_base * cfg.afu_ln_lr_mult, rho=cfg.afu_ln_rho)
        state.afu_ln_metric_time += time.perf_counter() - start

    total_update = sum(state.afu_step_group_updates.values())
    if total_update > 0:
        for group, value in state.afu_step_group_updates.items():
            state.afu_update_share.setdefault(group, []).append(value / max(1e-12, total_update))


def build_rbf_sobolev_gram(
    centers: torch.Tensor,
    width: float,
    alpha: float,
    beta: float,
    rho: float,
    grid_min: float = -2.5,
    grid_max: float = 2.5,
    grid_size: int = 512,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Build the shared RBF Sobolev Gram matrix used by GA-FU-v3."""
    device = device or centers.device
    c = centers.to(device=device, dtype=dtype)
    t = torch.linspace(grid_min, grid_max, grid_size, device=device, dtype=dtype)
    dt = (grid_max - grid_min) / max(1, grid_size - 1)
    diff = t.unsqueeze(1) - c.unsqueeze(0)
    basis = torch.exp(-0.5 * (diff / width).square())
    deriv1 = -(diff / (width**2)) * basis
    deriv2 = ((diff.square() / (width**4)) - (1.0 / (width**2))) * basis
    gram = dt * (basis.T @ basis)
    if alpha:
        gram = gram + float(alpha) * dt * (deriv1.T @ deriv1)
    if beta:
        gram = gram + float(beta) * dt * (deriv2.T @ deriv2)
    eye = torch.eye(c.numel(), device=device, dtype=dtype)
    damped = gram + float(rho) * eye
    eig = torch.linalg.eigvalsh(damped.float()).clamp_min(1e-12)
    chol = torch.linalg.cholesky(damped.float())
    return {
        "M": gram.float(),
        "A": damped.float(),
        "chol": chol,
        "A_inv_diag": torch.diag(damped.float()).reciprocal(),
        "condition": eig[-1] / eig[0],
        "eig_min": eig[0],
        "eig_max": eig[-1],
    }


def _v3_transition_length(cfg: TrainConfig, total_steps: int) -> int:
    raw = int(round(cfg.v3_transition_frac * max(1, total_steps)))
    if total_steps < cfg.v3_transition_min_steps:
        return max(1, min(5, raw))
    return max(cfg.v3_transition_min_steps, min(cfg.v3_transition_max_steps, raw))


def switch_branch_state(
    state: RuntimeState,
    cfg: TrainConfig,
    *,
    step_idx: int,
    total_steps: int,
    reason: str,
) -> None:
    if state.branch_switched:
        return
    state.branch_switched = True
    state.branch_switch_step = step_idx
    state.branch_switch_reason = reason
    state.phase_enter_step = step_idx
    state.transition_start_branch_scale = state.current_branch_scale
    state.transition_start_coeff_lr_multiplier = state.coeff_lr_multiplier
    if cfg.gafu_v3_enabled and cfg.v3_phase_mode == "smooth":
        state.phase = "TRANSITION"
        state.transition_length_steps = _v3_transition_length(cfg, total_steps)
    else:
        state.phase = "GEOMETRY"
        state.transition_length_steps = 0
        state.current_branch_scale = cfg.branch_final_scale
        state.coeff_lr_multiplier = cfg.coeff_lr_final_mult
        state.current_metric_mix = 1.0


def _transition_soft(cfg: TrainConfig, t: float) -> float:
    t = min(1.0, max(0.0, t))
    if cfg.v3_transition_curve == "linear":
        return t
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _gram_cache_key(
    layer: RBFDense,
    cfg: TrainConfig,
    *,
    alpha: float,
    beta: float,
    rho: float,
    device: torch.device,
    dtype: torch.dtype,
) -> str:
    return (
        f"basis{layer.centers.numel()}_w{layer.width:.8f}_a{alpha:.6g}_b{beta:.6g}_"
        f"rho{rho:.6g}_grid{cfg.v3_gram_grid_size}_{device.type}_{str(dtype)}"
    )


def _poly_basis_and_derivatives(t: torch.Tensor, order: int, *, kind: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if kind == "abs_power":
        powers = torch.arange(1, order + 1, device=t.device, dtype=t.dtype)
        abs_t = t.abs().clamp_min(1e-12).unsqueeze(1)
        sign_t = t.sign().unsqueeze(1)
        basis = abs_t.pow(powers.view(1, -1))
        deriv1 = powers.view(1, -1) * sign_t * abs_t.pow((powers - 1).clamp_min(0).view(1, -1))
        second_coeff = powers * (powers - 1)
        deriv2 = second_coeff.view(1, -1) * abs_t.pow((powers - 2).clamp_min(0).view(1, -1))
        deriv2 = torch.where((powers > 1).view(1, -1), deriv2, torch.zeros_like(deriv2))
        return basis, deriv1, deriv2

    powers = torch.arange(order, device=t.device, dtype=t.dtype)
    basis = t.unsqueeze(1).pow(powers.view(1, -1))
    deriv1 = torch.where(
        (powers > 0).view(1, -1),
        powers.view(1, -1) * t.unsqueeze(1).pow((powers - 1).clamp_min(0).view(1, -1)),
        torch.zeros_like(basis),
    )
    second_coeff = powers * (powers - 1)
    deriv2 = torch.where(
        (powers > 1).view(1, -1),
        second_coeff.view(1, -1) * t.unsqueeze(1).pow((powers - 2).clamp_min(0).view(1, -1)),
        torch.zeros_like(basis),
    )
    return basis, deriv1, deriv2


def build_polynomial_sobolev_gram(
    order: int,
    alpha: float,
    beta: float,
    rho: float,
    *,
    kind: str = "power",
    grid_min: float = -2.5,
    grid_max: float = 2.5,
    grid_size: int = 512,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Build a Sobolev Gram for polynomial rational coefficients."""
    t = torch.linspace(grid_min, grid_max, grid_size, device=device, dtype=dtype)
    dt = (grid_max - grid_min) / max(1, grid_size - 1)
    basis, deriv1, deriv2 = _poly_basis_and_derivatives(t, order, kind=kind)
    gram = dt * (basis.T @ basis)
    if alpha:
        gram = gram + float(alpha) * dt * (deriv1.T @ deriv1)
    if beta:
        gram = gram + float(beta) * dt * (deriv2.T @ deriv2)
    eye = torch.eye(order, device=device, dtype=dtype)
    damped = gram + float(rho) * eye
    eig = torch.linalg.eigvalsh(damped.float()).clamp_min(1e-12)
    chol = torch.linalg.cholesky(damped.float())
    return {
        "M": gram.float(),
        "A": damped.float(),
        "chol": chol,
        "A_inv_diag": torch.diag(damped.float()).reciprocal(),
        "condition": eig[-1] / eig[0],
        "eig_min": eig[0],
        "eig_max": eig[-1],
    }


def _get_polynomial_sobolev_gram(
    cfg: TrainConfig,
    state: RuntimeState,
    *,
    order: int,
    kind: str,
    alpha: float,
    beta: float,
    rho: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    key = (
        f"poly_{kind}_order{order}_a{alpha:.6g}_b{beta:.6g}_rho{rho:.6g}_"
        f"grid{cfg.v3_gram_grid_size}_{device.type}_{str(dtype)}"
    )
    cached = state.sobolev_gram_cache.get(key)
    if cached is not None:
        return cached
    start = time.perf_counter()
    gram = build_polynomial_sobolev_gram(
        order,
        alpha,
        beta,
        rho,
        kind=kind,
        grid_size=cfg.v3_gram_grid_size,
        device=device,
        dtype=dtype,
    )
    state.metric_build_time += time.perf_counter() - start
    state.sobolev_gram_cache[key] = gram
    return gram


def _get_sobolev_gram(
    layer: RBFDense,
    cfg: TrainConfig,
    state: RuntimeState,
    *,
    alpha: float,
    beta: float,
    rho: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    key = _gram_cache_key(layer, cfg, alpha=alpha, beta=beta, rho=rho, device=device, dtype=dtype)
    cached = state.sobolev_gram_cache.get(key)
    if cached is not None:
        return cached
    start = time.perf_counter()
    gram = build_rbf_sobolev_gram(
        layer.centers,
        layer.width,
        alpha,
        beta,
        rho,
        grid_size=cfg.v3_gram_grid_size,
        device=device,
        dtype=dtype,
    )
    state.metric_build_time += time.perf_counter() - start
    state.sobolev_gram_cache[key] = gram
    return gram


def _solve_with_gram(grad: torch.Tensor, gram: Dict[str, torch.Tensor]) -> torch.Tensor:
    flat = grad.reshape(-1, grad.shape[-1]).float()
    solved = torch.cholesky_solve(flat.T, gram["chol"]).T
    return solved.to(dtype=grad.dtype).reshape_as(grad)


def _gram_diag(gram: Dict[str, torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    shape = [1] * max(0, ref.ndim - 1) + [-1]
    return torch.diag(gram["A"]).to(device=ref.device, dtype=ref.dtype).view(*shape).expand_as(ref)


def _metric_tensor_norm(value: torch.Tensor, metric_diag: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((value.square() * metric_diag).sum().clamp_min(1e-24))


def _record_gram_stats(
    state: RuntimeState,
    gram: Dict[str, torch.Tensor],
    *,
    role: str,
    phase: str,
) -> None:
    condition = float(gram["condition"].detach().cpu())
    eig_min = float(gram["eig_min"].detach().cpu())
    eig_max = float(gram["eig_max"].detach().cpu())
    if role == "diagfast":
        state.diagfast_condition = condition
        state.diagfast_eig_min = eig_min
        state.diagfast_eig_max = eig_max
    if role == "fullgeo":
        state.fullgeo_condition = condition
        state.fullgeo_eig_min = eig_min
        state.fullgeo_eig_max = eig_max
    if phase == "ACTIVE":
        state.metric_condition_active = condition
        state.metric_eig_min_active = eig_min
        state.metric_eig_max_active = eig_max
    elif phase in {"GEOMETRY", "TRANSITION", "RECOVERY"}:
        state.metric_condition_geometry = condition
        state.metric_eig_min_geometry = eig_min
        state.metric_eig_max_geometry = eig_max


def _record_direction_audit(
    state: RuntimeState,
    *,
    diagfast_direction: torch.Tensor,
    fullgeo_direction: torch.Tensor,
    current_direction: torch.Tensor,
    diagfast_metric: torch.Tensor,
    fullgeo_metric: torch.Tensor,
    current_metric: torch.Tensor,
) -> None:
    diag_flat = diagfast_direction.detach().flatten().float()
    full_flat = fullgeo_direction.detach().flatten().float()
    cos = torch.dot(diag_flat, full_flat) / (diag_flat.norm() * full_flat.norm()).clamp_min(1e-12)
    diag_norm = diag_flat.norm().clamp_min(1e-12)
    full_norm = full_flat.norm()
    current_norm = current_direction.detach().flatten().float().norm()
    state.direction_cos_diagfast_fullgeo.append(float(cos.detach().cpu()))
    state.direction_norm_ratio_fullgeo_diagfast.append(float((full_norm / diag_norm).detach().cpu()))
    state.direction_norm_ratio_current_diagfast.append(float((current_norm / diag_norm).detach().cpu()))
    state.direction_metric_norm_diagfast.append(float(_metric_tensor_norm(diagfast_direction, diagfast_metric).detach().cpu()))
    state.direction_metric_norm_fullgeo.append(float(_metric_tensor_norm(fullgeo_direction, fullgeo_metric).detach().cpu()))
    state.direction_metric_norm_current.append(float(_metric_tensor_norm(current_direction, current_metric).detach().cpu()))


def _v3_precondition_direction(
    grad: torch.Tensor,
    layer: RBFDense,
    cfg: TrainConfig,
    state: RuntimeState,
    *,
    metric_mode: str,
    phase: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return GA-FU-v3 direction and diagonal trust metric.

    `diag_to_full_sobolev` intentionally mixes fast diagonal direction with
    geometry full direction, matching the v3.1 plan.
    """
    fast_gram = _get_sobolev_gram(
        layer,
        cfg,
        state,
        alpha=cfg.v3_alpha_fast,
        beta=cfg.v3_beta_fast,
        rho=cfg.v3_gram_rho,
        device=grad.device,
        dtype=grad.dtype,
    )
    geo_gram = _get_sobolev_gram(
        layer,
        cfg,
        state,
        alpha=cfg.v3_alpha_geo,
        beta=cfg.v3_beta_geo,
        rho=cfg.v3_gram_rho,
        device=grad.device,
        dtype=grad.dtype,
    )
    fast_diag = _gram_diag(fast_gram, grad).clamp_min(cfg.v3_gram_rho)
    geo_diag = _gram_diag(geo_gram, grad).clamp_min(cfg.v3_gram_rho)
    diagfast_direction = -grad / fast_diag

    if metric_mode == "basis_diag_gram":
        if phase in {"GEOMETRY", "RECOVERY"}:
            direction = -grad / geo_diag
            trust_metric = geo_diag
            _record_gram_stats(state, geo_gram, role="fullgeo", phase=phase)
        else:
            direction = diagfast_direction
            trust_metric = fast_diag
            _record_gram_stats(state, fast_gram, role="diagfast", phase=phase)
        return direction, trust_metric

    start = time.perf_counter()
    fullgeo_direction = -_solve_with_gram(grad, geo_gram)
    state.precond_solve_time += time.perf_counter() - start
    _record_gram_stats(state, fast_gram, role="diagfast", phase=phase)
    _record_gram_stats(state, geo_gram, role="fullgeo", phase=phase)

    if metric_mode == "full_sobolev_gram":
        direction = fullgeo_direction
        trust_metric = geo_diag
    elif metric_mode == "diag_to_full_sobolev":
        mix = min(1.0, max(0.0, state.current_metric_mix))
        direction = (1.0 - mix) * diagfast_direction + mix * fullgeo_direction
        trust_metric = ((1.0 - mix) * fast_diag + mix * geo_diag).clamp_min(cfg.v3_gram_rho)
    else:
        raise ValueError(f"unknown v3 metric_mode: {metric_mode}")

    _record_direction_audit(
        state,
        diagfast_direction=diagfast_direction,
        fullgeo_direction=fullgeo_direction,
        current_direction=direction,
        diagfast_metric=fast_diag,
        fullgeo_metric=geo_diag,
        current_metric=trust_metric,
    )
    return direction, trust_metric


def _rational_coeff_kind(name: str) -> str:
    return "abs_power" if name.endswith("weight_denominator") else "power"


def _rational_precondition_direction(
    grad: torch.Tensor,
    name: str,
    cfg: TrainConfig,
    state: RuntimeState,
    *,
    metric_mode: str,
    phase: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    order = grad.shape[-1]
    kind = _rational_coeff_kind(name)
    fast_gram = _get_polynomial_sobolev_gram(
        cfg,
        state,
        order=order,
        kind=kind,
        alpha=cfg.v3_alpha_fast,
        beta=cfg.v3_beta_fast,
        rho=cfg.v3_gram_rho,
        device=grad.device,
        dtype=grad.dtype,
    )
    geo_gram = _get_polynomial_sobolev_gram(
        cfg,
        state,
        order=order,
        kind=kind,
        alpha=cfg.v3_alpha_geo,
        beta=cfg.v3_beta_geo,
        rho=cfg.v3_gram_rho,
        device=grad.device,
        dtype=grad.dtype,
    )
    fast_diag = _gram_diag(fast_gram, grad).clamp_min(cfg.v3_gram_rho)
    geo_diag = _gram_diag(geo_gram, grad).clamp_min(cfg.v3_gram_rho)
    diagfast_direction = -grad / fast_diag

    if metric_mode == "basis_diag_gram":
        if phase in {"GEOMETRY", "RECOVERY"}:
            direction = -grad / geo_diag
            trust_metric = geo_diag
            _record_gram_stats(state, geo_gram, role="fullgeo", phase=phase)
        else:
            direction = diagfast_direction
            trust_metric = fast_diag
            _record_gram_stats(state, fast_gram, role="diagfast", phase=phase)
        return direction, trust_metric

    start = time.perf_counter()
    fullgeo_direction = -_solve_with_gram(grad, geo_gram)
    state.precond_solve_time += time.perf_counter() - start
    _record_gram_stats(state, fast_gram, role="diagfast", phase=phase)
    _record_gram_stats(state, geo_gram, role="fullgeo", phase=phase)

    if metric_mode == "full_sobolev_gram":
        direction = fullgeo_direction
        trust_metric = geo_diag
    elif metric_mode == "diag_to_full_sobolev":
        mix = min(1.0, max(0.0, state.current_metric_mix))
        direction = (1.0 - mix) * diagfast_direction + mix * fullgeo_direction
        trust_metric = ((1.0 - mix) * fast_diag + mix * geo_diag).clamp_min(cfg.v3_gram_rho)
    else:
        raise ValueError(f"unknown rational metric_mode: {metric_mode}")

    _record_direction_audit(
        state,
        diagfast_direction=diagfast_direction,
        fullgeo_direction=fullgeo_direction,
        current_direction=direction,
        diagfast_metric=fast_diag,
        fullgeo_metric=geo_diag,
        current_metric=trust_metric,
    )
    return direction, trust_metric


def _rational_tangent_jacobian(x: torch.Tensor, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    a = numerator.reshape(-1).to(device=x.device, dtype=x.dtype)
    b = denominator.reshape(-1).to(device=x.device, dtype=x.dtype)
    xp = torch.stack([x.pow(i) for i in range(a.numel())], dim=-1)
    ax = x.abs()
    bp = torch.stack([ax.pow(i + 1) for i in range(b.numel())], dim=-1)
    p = (xp * a.view(1, -1)).sum(dim=-1, keepdim=True)
    q = 1.0 + (bp * b.abs().view(1, -1)).sum(dim=-1, keepdim=True)
    da = xp / q.clamp_min(1e-12)
    db = -p / q.clamp_min(1e-12).square() * bp * b.sign().view(1, -1)
    return torch.cat([da, db], dim=-1)


def _rational_joint_tangent_gram(
    act: nn.Module,
    cfg: TrainConfig,
    state: RuntimeState,
    *,
    metric_mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    numerator = getattr(act, "weight_numerator").detach().reshape(-1)
    denominator = getattr(act, "weight_denominator").detach()
    groups = denominator.shape[0]
    num_dim = numerator.numel()
    den_dim = denominator.shape[-1]
    total_dim = num_dim + groups * den_dim
    h1 = "h1" in metric_mode.lower()
    key = (
        f"rat_tangent_{'h1' if h1 else 'l2'}_n{num_dim}_d{den_dim}_g{groups}_"
        f"a{cfg.v3_alpha_geo:.6g}_rho{cfg.v3_gram_rho:.6g}_grid{cfg.v3_gram_grid_size}_"
        f"{device.type}_{str(dtype)}_{float(numerator.detach().abs().sum()):.6g}_"
        f"{float(denominator.detach().abs().sum()):.6g}"
    )
    cached = state.sobolev_gram_cache.get(key)
    if cached is not None:
        return cached
    start = time.perf_counter()
    grid = torch.linspace(-2.5, 2.5, cfg.v3_gram_grid_size, device=device, dtype=dtype)
    dt = 5.0 / max(1, cfg.v3_gram_grid_size - 1)
    mat = torch.zeros(total_dim, total_dim, device=device, dtype=torch.float32)
    eps = 1e-2
    for group_idx in range(groups):
        j_local = _rational_tangent_jacobian(grid, numerator, denominator[group_idx]).float()
        j_full = torch.zeros(grid.numel(), total_dim, device=device, dtype=torch.float32)
        j_full[:, :num_dim] = j_local[:, :num_dim]
        start_idx = num_dim + group_idx * den_dim
        j_full[:, start_idx : start_idx + den_dim] = j_local[:, num_dim:]
        mat = mat + dt * (j_full.T @ j_full)
        if h1 and cfg.v3_alpha_geo:
            jp = _rational_tangent_jacobian((grid + eps).clamp(-2.5, 2.5), numerator, denominator[group_idx]).float()
            jm = _rational_tangent_jacobian((grid - eps).clamp(-2.5, 2.5), numerator, denominator[group_idx]).float()
            jprime_local = (jp - jm) / (2.0 * eps)
            jprime_full = torch.zeros_like(j_full)
            jprime_full[:, :num_dim] = jprime_local[:, :num_dim]
            jprime_full[:, start_idx : start_idx + den_dim] = jprime_local[:, num_dim:]
            mat = mat + float(cfg.v3_alpha_geo) * dt * (jprime_full.T @ jprime_full)
    eye = torch.eye(total_dim, device=device, dtype=torch.float32)
    damped = mat + float(cfg.v3_gram_rho) * eye
    eig = torch.linalg.eigvalsh(damped).clamp_min(1e-12)
    chol = torch.linalg.cholesky(damped)
    gram = {
        "M": mat,
        "A": damped,
        "chol": chol,
        "A_inv_diag": torch.diag(damped).reciprocal(),
        "condition": eig[-1] / eig[0],
        "eig_min": eig[0],
        "eig_max": eig[-1],
    }
    state.metric_build_time += time.perf_counter() - start
    state.sobolev_gram_cache[key] = gram
    return gram


def _flat_rational_params(act: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numerator = getattr(act, "weight_numerator")
    denominator = getattr(act, "weight_denominator")
    param_vec = torch.cat([numerator.detach().reshape(-1), denominator.detach().reshape(-1)]).float()
    grad_num = torch.zeros_like(numerator.detach()).reshape(-1) if numerator.grad is None else numerator.grad.detach().reshape(-1)
    grad_den = torch.zeros_like(denominator.detach()).reshape(-1) if denominator.grad is None else denominator.grad.detach().reshape(-1)
    grad_vec = torch.cat([grad_num, grad_den]).float()
    return param_vec.to(numerator.device), grad_vec.to(numerator.device), torch.cat([grad_num, grad_den]).to(numerator.device)


def _rational_branch_linear_step(
    model: "RationalDGKANClassifier",
    cfg: TrainConfig,
    state: RuntimeState,
    *,
    lr: float,
) -> None:
    for block in model.blocks:
        linear = block.kan.linear
        if linear.weight.grad is None:
            continue
        z = block.kan.last_input
        if z is None:
            continue
        with torch.no_grad():
            act = block.kan._ensure_act(z)
            r = act(z.unsqueeze(1)).squeeze(1).detach().float()
            cov = (r.T @ r) / max(1, r.shape[0])
            cov = cov + float(cfg.v3_gram_rho) * torch.eye(cov.shape[0], device=cov.device)
            eig = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
            inv = torch.linalg.inv(cov)
            raw = linear.weight.grad.detach().float()
            precond = raw @ inv
            update_w = -lr * precond.to(dtype=linear.weight.dtype)
            linear.weight.add_(update_w)
            if linear.bias is not None and linear.bias.grad is not None:
                linear.bias.add_(-lr * linear.bias.grad.detach())
                linear.bias.grad = None
            linear.weight.grad = None
            state.rational_branch_linear_update_norms.append(float(update_w.norm().detach().cpu()))
            state.rational_branch_linear_update_over_param.append(
                float(update_w.norm().detach().cpu()) / max(1e-12, float(linear.weight.detach().norm().detach().cpu()))
            )
            state.rational_branch_linear_cov_conditions.append(float((eig[-1] / eig[0]).detach().cpu()))
            state.rational_branch_linear_cos_raw_precond.append(_safe_cos(raw, precond))


def rational_tangent_functional_step(
    model: "RationalDGKANClassifier",
    cfg: TrainConfig,
    state: RuntimeState,
    *,
    step_idx: int,
    total_steps: int,
) -> None:
    progress = min(1.0, step_idx / max(1, total_steps))
    metric_mode = _v3_phase_metric(cfg, state)
    metric_key = metric_mode.lower().replace("-", "_")
    coeff_decay = scheduled_lr_factor(cfg.coeff_lr_schedule, progress, cfg.coeff_lr_decay_final_mult, cfg.lr_decay_start_frac)
    lr = cfg.coeff_lr * state.coeff_lr_multiplier * coeff_decay
    for block in model.blocks:
        act = block.kan.act
        if act is None or not hasattr(act, "weight_numerator") or not hasattr(act, "weight_denominator"):
            continue
        numerator = getattr(act, "weight_numerator")
        denominator = getattr(act, "weight_denominator")
        if numerator.grad is None and denominator.grad is None:
            continue
        param_vec, grad_vec, _ = _flat_rational_params(act)
        if cfg.credit_mode == "identity":
            grad_vec = grad_vec * cfg.identity_credit_scale
        gram = _rational_joint_tangent_gram(
            act,
            cfg,
            state,
            metric_mode=metric_mode,
            device=param_vec.device,
            dtype=param_vec.dtype,
        )
        if "diag" in metric_key:
            direction = -grad_vec / torch.diag(gram["A"]).to(grad_vec.device).clamp_min(cfg.v3_gram_rho)
        else:
            start = time.perf_counter()
            direction = -torch.cholesky_solve(grad_vec.view(-1, 1), gram["chol"]).view(-1)
            state.precond_solve_time += time.perf_counter() - start
        _record_gram_stats(state, gram, role="fullgeo", phase=state.phase)
        update = lr * direction.to(param_vec.device)
        if cfg.trust_mode != "off" and cfg.trust_radius > 0:
            a = gram["A"].to(device=update.device)
            m_norm = torch.sqrt((update.view(1, -1) @ a @ update.view(-1, 1)).squeeze().clamp_min(1e-24)).item()
            p_norm = torch.sqrt((param_vec.view(1, -1) @ a @ param_vec.view(-1, 1)).squeeze().clamp_min(1e-24)).item()
            limit = cfg.trust_radius * (p_norm + 1e-6)
            if m_norm > limit > 0:
                update = update * (limit / (m_norm + 1e-12))
                state.trust_clip_count += 1
        update_metric_norm = torch.sqrt(
            (update.view(1, -1) @ gram["A"].to(device=update.device) @ update.view(-1, 1)).squeeze().clamp_min(1e-24)
        ).item()
        numel = numerator.numel()
        update_num = update[:numel].reshape_as(numerator)
        update_den = update[numel:].reshape_as(denominator)
        with torch.no_grad():
            numerator.add_(update_num.to(dtype=numerator.dtype))
            denominator.add_(update_den.to(dtype=denominator.dtype))
        numerator.grad = None
        denominator.grad = None
        raw_norm = float(grad_vec.norm().detach().cpu())
        direction_norm = float(direction.norm().detach().cpu())
        coeff_norm = float(param_vec.norm().detach().cpu())
        state.raw_grad_norms.append(raw_norm)
        state.precond_direction_norms.append(direction_norm)
        state.update_metric_norms.append(update_metric_norm)
        state.update_over_coeff_norms.append(float(update.norm().detach().cpu()) / max(1e-12, coeff_norm))
        state.rational_num_update_norms.append(float(update_num.norm().detach().cpu()))
        state.rational_den_update_norms.append(float(update_den.norm().detach().cpu()))
        state.rational_num_den_update_ratio.append(
            float(update_num.norm().detach().cpu()) / max(1e-12, float(update_den.norm().detach().cpu()))
        )
        state.rational_num_den_update_cos.append(_safe_cos(update_num, update_den.flatten()[: update_num.numel()]))
        state.coeff_update_count += 1
    if cfg.rational_branch_functional:
        _rational_branch_linear_step(model, cfg, state, lr=lr)


def _v3_phase_metric(cfg: TrainConfig, state: RuntimeState) -> str:
    if not cfg.gafu_v3_enabled:
        return cfg.metric_mode
    if state.phase == "ACTIVE":
        return cfg.v3_metric_active
    if state.phase == "TRANSITION":
        return cfg.v3_metric_transition
    return cfg.v3_metric_geometry


def _v3_gram_params(cfg: TrainConfig, state: RuntimeState, *, for_geometry: bool = False) -> Tuple[float, float]:
    if for_geometry or state.phase in {"GEOMETRY", "RECOVERY"}:
        return cfg.v3_alpha_geo, cfg.v3_beta_geo
    return cfg.v3_alpha_fast, cfg.v3_beta_fast


def grid_metric_diag(param: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    k = torch.linspace(-1.0, 1.0, param.shape[-1], device=param.device)
    diag = 1.0 + cfg.sobolev_alpha * (math.pi * k).square() + cfg.sobolev_beta * (math.pi * k).pow(4)
    return diag.view(*([1] * (param.ndim - 1)), -1).clamp_min(cfg.rho)


def _data_metric_diag(layer: RBFDense, param: torch.Tensor) -> torch.Tensor:
    if layer.last_basis_mean is None:
        return torch.ones_like(param)
    occ = layer.last_basis_mean.to(param.device).float().clamp_min(1e-5)
    occ = occ / occ.mean().clamp_min(1e-5)
    diag = 1.0 / occ
    return diag.view(1, 1, -1).expand_as(param).clamp(0.2, 5.0)


def data_lambda_for_progress(cfg: TrainConfig, progress: float) -> float:
    if cfg.metric_mode == "grid":
        return 0.0
    if cfg.metric_mode == "no_grid":
        return 1.0
    if cfg.metric_mode == "fixed_data":
        return float(cfg.lambda_final)
    if cfg.metric_mode != "pulse":
        return 0.0
    if progress <= cfg.pulse_frac:
        return float(cfg.lambda_early)
    anchor_end = min(1.0, cfg.pulse_frac + 0.35)
    if progress >= anchor_end:
        return float(cfg.lambda_final)
    t = (progress - cfg.pulse_frac) / max(1e-6, anchor_end - cfg.pulse_frac)
    if cfg.anchor_decay == "cosine":
        t = 0.5 - 0.5 * math.cos(math.pi * t)
    return float((1.0 - t) * cfg.lambda_early + t * cfg.lambda_final)


def _snr_multiplier(
    name: str,
    grad: torch.Tensor,
    state: RuntimeState,
    cfg: TrainConfig,
    step_idx: int,
) -> torch.Tensor:
    if not cfg.snr_enabled:
        return torch.ones_like(grad)
    cached = state.snr_mult.get(name)
    if cached is not None and step_idx % max(1, cfg.snr_update_interval) != 0:
        return cached

    start = time.perf_counter()
    g = grad.detach()
    if cfg.snr_sample_frac < 1.0 and g.shape[1] > 1:
        keep = max(1, int(g.shape[1] * cfg.snr_sample_frac))
        g_for_stat = g[:, :keep, :]
    else:
        g_for_stat = g

    mean = state.snr_mean.get(name)
    sq = state.snr_sq.get(name)
    if mean is None:
        mean = torch.zeros_like(g)
        sq = torch.zeros_like(g)
    beta = cfg.snr_ema_beta
    mean = beta * mean + (1.0 - beta) * g
    sq = beta * sq + (1.0 - beta) * g.square()
    state.snr_mean[name] = mean
    state.snr_sq[name] = sq

    var = (sq - mean.square()).clamp_min(1e-10)
    snr = mean.abs() / var.sqrt().add(1e-8)
    if cfg.snr_grouping == "layer":
        grouped = snr.mean().view(1, 1, 1)
    elif cfg.snr_grouping == "output_channel":
        grouped = snr.mean(dim=(1, 2), keepdim=True)
    elif cfg.snr_grouping == "edge":
        grouped = snr.mean(dim=-1, keepdim=True)
    else:
        grouped = snr

    threshold = torch.quantile(grouped.detach().flatten(), cfg.snr_percentile / 100.0).clamp_min(1e-4)
    mult = (grouped / threshold).clamp(cfg.snr_lr_min, cfg.snr_lr_max)
    mult = mult.expand_as(g).detach()
    state.snr_mult[name] = mult
    state.snr_update_time += time.perf_counter() - start
    _ = g_for_stat  # Keeps the sample_frac knob explicit for cost accounting.
    return mult


def functional_coeff_step(
    model: nn.Module,
    cfg: TrainConfig,
    state: RuntimeState,
    *,
    step_idx: int,
    total_steps: int,
) -> None:
    progress = min(1.0, step_idx / max(1, total_steps))
    data_lambda = data_lambda_for_progress(cfg, progress)
    state.last_data_lambda = data_lambda
    if (
        cfg.reset_on_reanchor
        and cfg.metric_mode == "pulse"
        and not state.reanchor_reset_done
        and progress > cfg.pulse_frac
        and cfg.momentum_mu > 0
    ):
        state.momentum.clear()
        state.reanchor_reset_done = True

    layers = list(model.kan_layers()) if hasattr(model, "kan_layers") else []
    is_rational = isinstance(model, RationalDGKANClassifier)
    metric_mode_initial = _v3_phase_metric(cfg, state)
    metric_key_initial = metric_mode_initial.lower().replace("-", "_")
    if is_rational and metric_key_initial.startswith("rational_tangent"):
        rational_tangent_functional_step(model, cfg, state, step_idx=step_idx, total_steps=total_steps)
        return
    phase_metric = _v3_phase_metric(cfg, state)
    state.phase_seen_counts[state.phase] = state.phase_seen_counts.get(state.phase, 0) + 1
    state.metric_seen_counts[phase_metric] = state.metric_seen_counts.get(phase_metric, 0) + 1
    if len(state.phase_trace_first20) < 20:
        state.phase_trace_first20.append(state.phase)
        state.metric_trace_first20.append(phase_metric)
    for layer_idx, (name, p) in enumerate(coefficient_named_params(model)):
        if p.grad is None:
            continue
        g = p.grad.detach()
        if cfg.credit_mode == "identity":
            g = g * cfg.identity_credit_scale
        elif cfg.credit_mode == "first_order":
            g = g * 0.95

        pure_role = _pure_param_role(name) if isinstance(model, PureKANClassifier) else ""
        metric_mode = _pure_role_metric(cfg, pure_role, phase_metric) if pure_role else phase_metric
        metric_key = metric_mode.lower().replace("-", "_")
        if pure_role:
            seen = state.pure_role_metric_seen.setdefault(pure_role, [])
            if metric_mode not in seen:
                seen.append(metric_mode)
        grid = grid_metric_diag(p, cfg).expand_as(p)
        metric = grid
        direction: torch.Tensor
        trust_metric: torch.Tensor
        metric_condition_for_role = float("nan")

        if metric_mode in {"grid", "grid_index_diag_legacy", "fixed_data", "no_grid", "pulse"}:
            if progress < cfg.warmup_frac and not cfg.gafu_v3_enabled:
                metric = torch.ones_like(p) + cfg.rho
            direction = -g / metric.clamp_min(cfg.rho)
            trust_metric = metric.clamp_min(cfg.rho)
            metric_condition_for_role = float("nan")
        elif metric_mode in {"identity", "none", "rational_identity"}:
            metric = torch.ones_like(p) + cfg.rho
            direction = -g
            trust_metric = metric
            metric_condition_for_role = 1.0
        elif metric_mode in {"basis_diag_gram", "full_sobolev_gram", "diag_to_full_sobolev"} or (
            is_rational and (metric_key.startswith("rational_poly") or metric_key.startswith("rational_legendre"))
        ):
            if is_rational:
                rational_mode = "basis_diag_gram" if "diag" in metric_key or "legendre" in metric_key else "full_sobolev_gram"
                direction, trust_metric = _rational_precondition_direction(
                    g,
                    name,
                    cfg,
                    state,
                    metric_mode=rational_mode,
                    phase=state.phase,
                )
            else:
                direction, trust_metric = _v3_precondition_direction(
                    g,
                    layers[layer_idx],
                    cfg,
                    state,
                    metric_mode=metric_mode,
                    phase=state.phase,
                )
                if metric_mode == "basis_diag_gram":
                    metric_condition_for_role = state.diagfast_condition
                elif metric_mode == "full_sobolev_gram":
                    metric_condition_for_role = state.fullgeo_condition
                else:
                    metric_condition_for_role = state.metric_condition_geometry
        else:
            raise ValueError(f"unknown metric_mode: {metric_mode}")

        if cfg.metric_mode in {"fixed_data", "no_grid", "pulse"} and not cfg.gafu_v3_enabled and not is_rational:
            data_metric = _data_metric_diag(layers[layer_idx], p)
            metric = (1.0 - data_lambda) * metric + data_lambda * data_metric
            trust_metric = metric.clamp_min(cfg.rho)
            direction = -g / trust_metric
        direction = direction * _snr_multiplier(name, g, state, cfg, step_idx)
        if cfg.momentum_mu > 0 and progress >= cfg.momentum_start_frac:
            prev = state.momentum.get(name)
            if prev is None:
                prev = torch.zeros_like(direction)
            direction = cfg.momentum_mu * prev + (1.0 - cfg.momentum_mu) * direction
            state.momentum[name] = direction.detach().clone()

        coeff_decay = scheduled_lr_factor(
            cfg.coeff_lr_schedule,
            progress,
            cfg.coeff_lr_decay_final_mult,
            cfg.lr_decay_start_frac,
        )
        role_lr_mult = _pure_role_lr_mult(cfg, pure_role) if pure_role else 1.0
        lr = cfg.coeff_lr * state.coeff_lr_multiplier * coeff_decay * role_lr_mult
        update = lr * direction
        raw_norm = float(g.norm().detach().cpu())
        direction_norm = float(direction.norm().detach().cpu())
        coeff_norm = float(p.detach().norm().detach().cpu())
        state.raw_grad_norms.append(raw_norm)
        state.precond_direction_norms.append(direction_norm)
        trust_radius = _pure_role_trust_radius(cfg, pure_role) if pure_role else cfg.trust_radius
        if cfg.trust_mode != "off" and trust_radius > 0:
            m_norm = torch.sqrt((update.square() * trust_metric).sum()).item()
            p_norm = torch.sqrt((p.detach().square() * trust_metric).sum()).item()
            limit = trust_radius * (p_norm + 1e-6)
            if m_norm > limit > 0:
                update = update * (limit / (m_norm + 1e-12))
                state.trust_clip_count += 1
        update_metric_norm = torch.sqrt((update.square() * trust_metric).sum()).item()
        state.update_metric_norms.append(update_metric_norm)
        state.update_over_coeff_norms.append(float(update.norm().detach().cpu()) / max(1e-12, coeff_norm))
        if pure_role:
            if name not in state.pure_seen_param_names:
                state.pure_seen_param_names.add(name)
                state.pure_seen_param_numel += int(p.numel())
            update_norm = float(update.norm().detach().cpu())
            _append_role_value(state.pure_role_update_norms, pure_role, update_norm)
            _append_role_value(state.pure_role_update_over_param, pure_role, update_norm / max(1e-12, coeff_norm))
            _append_role_value(state.pure_role_raw_grad_norms, pure_role, raw_norm)
            _append_role_value(state.pure_role_precond_norms, pure_role, direction_norm)
            _append_role_value(state.pure_role_cos_raw_precond, pure_role, _safe_cos(g, -direction))
            _append_role_value(state.pure_role_metric_conditions, pure_role, metric_condition_for_role)

        with torch.no_grad():
            p.add_(update)
            if cfg.prox_lambda > 0 and progress >= cfg.prox_start_frac:
                p.div_(1.0 + lr * cfg.prox_lambda * grid)
            if cfg.spectral_lambda > 0 and progress >= cfg.spectral_start_frac:
                mean = p.mean(dim=-1, keepdim=True)
                p.lerp_(mean, min(0.2, lr * cfg.spectral_lambda))
        p.grad = None
        state.coeff_update_count += 1


def _uo_precondition_coeff_grads(
    model: DGKANClassifier,
    cfg: TrainConfig,
    state: RuntimeState,
    *,
    normalize: bool = False,
) -> None:
    raw_parts: List[torch.Tensor] = []
    precond_parts: List[torch.Tensor] = []
    layers = list(model.kan_layers())
    metric_mode = _v3_phase_metric(cfg, state)
    for layer_idx, (_, p) in enumerate(coefficient_named_params(model)):
        if p.grad is None:
            continue
        raw = p.grad.detach().clone()
        direction, _ = _v3_precondition_direction(
            raw,
            layers[layer_idx],
            cfg,
            state,
            metric_mode=metric_mode,
            phase=state.phase,
        )
        precond = -direction
        if normalize:
            precond = precond * (raw.norm() / precond.norm().clamp_min(1e-12))
        p.grad.copy_(precond)
        raw_parts.append(raw)
        precond_parts.append(precond.detach().clone())
    raw_flat = _flat_cat(raw_parts, device=next(model.parameters()).device)
    precond_flat = _flat_cat(precond_parts, device=next(model.parameters()).device)
    raw_norm = float(raw_flat.norm().detach().cpu()) if raw_flat.numel() else 0.0
    precond_norm = float(precond_flat.norm().detach().cpu()) if precond_flat.numel() else 0.0
    state.uo_last_raw_grad = raw_flat.detach()
    state.uo_last_precond_grad = precond_flat.detach()
    state.uo_kan_raw_grad_norms.append(raw_norm)
    state.uo_kan_precond_grad_norms.append(precond_norm)
    state.uo_kan_precond_over_raw_norms.append(precond_norm / max(1e-12, raw_norm))
    state.uo_moment_cos_raw_precond.append(_safe_cos(raw_flat, precond_flat))


def _uo_record_adam_moments(
    opt: torch.optim.Optimizer,
    model: DGKANClassifier,
    state: RuntimeState,
) -> None:
    m_parts: List[torch.Tensor] = []
    v_parts: List[torch.Tensor] = []
    for _, p in coefficient_named_params(model):
        opt_state = opt.state.get(p, {})
        m = opt_state.get("exp_avg")
        v = opt_state.get("exp_avg_sq")
        if m is not None:
            m_parts.append(m.detach())
        if v is not None:
            v_parts.append(v.detach())
    m_flat = _flat_cat(m_parts, device=next(model.parameters()).device)
    v_flat = _flat_cat(v_parts, device=next(model.parameters()).device)
    m_norm = float(m_flat.norm().detach().cpu()) if m_flat.numel() else 0.0
    v_norm = float(v_flat.norm().detach().cpu()) if v_flat.numel() else 0.0
    precond = state.uo_last_precond_grad
    raw = state.uo_last_raw_grad
    state.uo_adam_m_norm_kan.append(m_norm)
    state.uo_adam_v_norm_kan.append(v_norm)
    if precond is not None:
        state.uo_moment_cos_precond_current.append(_safe_cos(m_flat, precond))
        state.uo_moment_amplification_ratio.append(m_norm / max(1e-12, float(precond.norm().detach().cpu())))
    if raw is not None:
        state.uo_moment_cos_raw_current.append(_safe_cos(m_flat, raw))


def unified_postadam_coeff_step(
    model: DGKANClassifier,
    cfg: TrainConfig,
    state: RuntimeState,
    *,
    step_idx: int,
    total_steps: int,
) -> None:
    progress = min(1.0, step_idx / max(1, total_steps))
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    coeff_decay = scheduled_lr_factor(
        cfg.coeff_lr_schedule,
        progress,
        cfg.coeff_lr_decay_final_mult,
        cfg.lr_decay_start_frac,
    )
    lr = cfg.coeff_lr * state.coeff_lr_multiplier * coeff_decay
    trust_radius = cfg.uo_trust_radius
    layers = list(model.kan_layers())
    raw_parts: List[torch.Tensor] = []
    adam_parts: List[torch.Tensor] = []
    precond_parts: List[torch.Tensor] = []
    for layer_idx, (name, p) in enumerate(coefficient_named_params(model)):
        if p.grad is None:
            continue
        g = p.grad.detach()
        m = state.uo_m.get(name)
        v = state.uo_v.get(name)
        if m is None:
            m = torch.zeros_like(g)
            v = torch.zeros_like(g)
        t = state.uo_t.get(name, 0) + 1
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g.square()
        state.uo_m[name] = m.detach().clone()
        state.uo_v[name] = v.detach().clone()
        state.uo_t[name] = t
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        adam_direction = m_hat / (v_hat.sqrt() + eps)
        direction, trust_metric = _v3_precondition_direction(
            adam_direction,
            layers[layer_idx],
            cfg,
            state,
            metric_mode=_v3_phase_metric(cfg, state),
            phase=state.phase,
        )
        update = lr * direction
        if cfg.rest_weight_decay > 0:
            update = update - lr * cfg.rest_weight_decay * p.detach()
        if trust_radius > 0:
            m_norm = torch.sqrt((update.square() * trust_metric).sum()).item()
            p_norm = torch.sqrt((p.detach().square() * trust_metric).sum()).item()
            limit = trust_radius * (p_norm + 1e-6)
            if m_norm > limit > 0:
                update = update * (limit / (m_norm + 1e-12))
                state.trust_clip_count += 1
        with torch.no_grad():
            p.add_(update)
        raw_parts.append(g)
        adam_parts.append(adam_direction.detach())
        precond_parts.append((-direction).detach())
        p.grad = None
        state.coeff_update_count += 1

    raw_flat = _flat_cat(raw_parts, device=next(model.parameters()).device)
    adam_flat = _flat_cat(adam_parts, device=next(model.parameters()).device)
    precond_flat = _flat_cat(precond_parts, device=next(model.parameters()).device)
    raw_norm = float(raw_flat.norm().detach().cpu()) if raw_flat.numel() else 0.0
    precond_norm = float(precond_flat.norm().detach().cpu()) if precond_flat.numel() else 0.0
    adam_norm = float(adam_flat.norm().detach().cpu()) if adam_flat.numel() else 0.0
    state.uo_last_raw_grad = raw_flat.detach()
    state.uo_last_adam_direction = adam_flat.detach()
    state.uo_last_precond_grad = precond_flat.detach()
    state.uo_kan_raw_grad_norms.append(raw_norm)
    state.uo_kan_precond_grad_norms.append(precond_norm)
    state.uo_kan_precond_over_raw_norms.append(precond_norm / max(1e-12, raw_norm))
    state.uo_adam_m_norm_kan.append(adam_norm)
    v_norms = [v.detach().float().norm() for v in state.uo_v.values()]
    state.uo_adam_v_norm_kan.append(float(torch.stack(v_norms).norm().detach().cpu()) if v_norms else 0.0)
    state.uo_moment_cos_raw_precond.append(_safe_cos(raw_flat, precond_flat))
    state.uo_moment_cos_precond_current.append(_safe_cos(adam_flat, precond_flat))
    state.uo_moment_cos_raw_current.append(_safe_cos(adam_flat, raw_flat))
    state.uo_moment_amplification_ratio.append(adam_norm / max(1e-12, precond_norm))


def set_schedule_for_step(
    model: DGKANClassifier,
    cfg: TrainConfig,
    state: RuntimeState,
    *,
    progress: float,
    step_idx: int = 0,
    total_steps: int = 1,
) -> None:
    if cfg.gafu_v3_enabled:
        if not state.branch_switched:
            state.phase = "ACTIVE"
            state.current_branch_scale = cfg.branch_boost
            state.coeff_lr_multiplier = cfg.coeff_lr_boost
            state.current_metric_mix = 0.0
        elif cfg.v3_phase_mode == "smooth":
            elapsed = max(0, step_idx - state.branch_switch_step + 1)
            length = state.transition_length_steps or _v3_transition_length(cfg, total_steps)
            soft = _transition_soft(cfg, elapsed / max(1, length))
            if soft >= 1.0:
                state.phase = "GEOMETRY"
                state.current_branch_scale = cfg.branch_final_scale
                state.coeff_lr_multiplier = cfg.coeff_lr_final_mult
                state.current_metric_mix = 1.0
            else:
                state.phase = "TRANSITION"
                state.current_branch_scale = (
                    (1.0 - soft) * state.transition_start_branch_scale
                    + soft * cfg.branch_final_scale
                )
                state.coeff_lr_multiplier = (
                    (1.0 - soft) * state.transition_start_coeff_lr_multiplier
                    + soft * cfg.coeff_lr_final_mult
                )
                state.current_metric_mix = soft
        else:
            state.phase = "GEOMETRY"
            state.current_branch_scale = cfg.branch_final_scale
            state.coeff_lr_multiplier = cfg.coeff_lr_final_mult
            state.current_metric_mix = 1.0
        state.metric_mix_sum += state.current_metric_mix
        state.metric_mix_count += 1
        model.set_branch_scale(state.current_branch_scale)
        return

    if cfg.branch_schedule == "none":
        state.current_branch_scale = cfg.branch_final_scale
        state.coeff_lr_multiplier = cfg.coeff_lr_final_mult
    elif cfg.branch_schedule in {"early_active_cosine", "early_active_linear"}:
        active = progress < cfg.branch_active_frac
        if active:
            if cfg.branch_schedule == "early_active_cosine":
                t = progress / max(1e-6, cfg.branch_active_frac)
                soft = 0.5 + 0.5 * math.cos(math.pi * t)
                state.current_branch_scale = cfg.branch_final_scale + soft * (cfg.branch_boost - cfg.branch_final_scale)
                state.coeff_lr_multiplier = cfg.coeff_lr_final_mult + soft * (
                    cfg.coeff_lr_boost - cfg.coeff_lr_final_mult
                )
            else:
                state.current_branch_scale = cfg.branch_boost
                state.coeff_lr_multiplier = cfg.coeff_lr_boost
        else:
            state.current_branch_scale = cfg.branch_final_scale
            state.coeff_lr_multiplier = cfg.coeff_lr_final_mult
    elif cfg.branch_schedule in {"loss_aware", "geometry_aware"}:
        if not state.branch_switched:
            state.current_branch_scale = cfg.branch_boost
            state.coeff_lr_multiplier = cfg.coeff_lr_boost
        else:
            state.current_branch_scale = min(state.current_branch_scale, cfg.branch_final_scale)
            state.coeff_lr_multiplier = cfg.coeff_lr_final_mult
    else:
        raise ValueError(f"unknown branch_schedule: {cfg.branch_schedule}")
    model.set_branch_scale(state.current_branch_scale)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    disable_kan: bool = False,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_ok = 0
    total = 0
    confs: List[torch.Tensor] = []
    preds: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size].to(device)
        yb = y[start : start + batch_size].to(device)
        try:
            logits = model(xb, disable_kan=disable_kan)
        except TypeError:
            logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction="sum")
        prob = logits.softmax(dim=-1)
        conf, pred = prob.max(dim=-1)
        total_loss += float(loss.detach().cpu())
        total_ok += int((pred == yb).sum().detach().cpu())
        total += int(yb.numel())
        confs.append(conf.detach().cpu())
        preds.append(pred.detach().cpu())
        labels.append(yb.detach().cpu())
    conf_cat = torch.cat(confs) if confs else torch.empty(0)
    pred_cat = torch.cat(preds) if preds else torch.empty(0, dtype=torch.long)
    label_cat = torch.cat(labels) if labels else torch.empty(0, dtype=torch.long)
    return {
        "loss": total_loss / max(1, total),
        "acc": total_ok / max(1, total),
        "ece": expected_calibration_error(conf_cat, pred_cat, label_cat),
    }


@torch.no_grad()
def kan_contribution_audit(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    if not isinstance(model, (DGKANClassifier, RationalDGKANClassifier)):
        return {
            "kan_logit_delta_norm_mean": 0.0,
            "kan_logit_delta_norm_p95": 0.0,
            "kan_correct_class_delta_mean": 0.0,
            "kan_margin_contribution_mean": 0.0,
            "kan_margin_contribution_p95": 0.0,
        }
    model.eval()
    delta_norms: List[torch.Tensor] = []
    correct_deltas: List[torch.Tensor] = []
    margin_deltas: List[torch.Tensor] = []
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size].to(device)
        yb = y[start : start + batch_size].to(device)
        full = model(xb)
        no_kan = model(xb, disable_kan=True)
        delta = full - no_kan
        delta_norms.append(delta.norm(dim=-1).detach().cpu())
        correct_deltas.append(delta.gather(1, yb.view(-1, 1)).squeeze(1).detach().cpu())

        full_target = full.gather(1, yb.view(-1, 1)).squeeze(1)
        no_target = no_kan.gather(1, yb.view(-1, 1)).squeeze(1)
        mask = F.one_hot(yb, full.shape[-1]).bool()
        full_other = full.masked_fill(mask, float("-inf")).max(dim=-1).values
        no_other = no_kan.masked_fill(mask, float("-inf")).max(dim=-1).values
        margin_deltas.append(((full_target - full_other) - (no_target - no_other)).detach().cpu())

    norm = torch.cat(delta_norms) if delta_norms else torch.zeros(1)
    correct = torch.cat(correct_deltas) if correct_deltas else torch.zeros(1)
    margin = torch.cat(margin_deltas) if margin_deltas else torch.zeros(1)
    return {
        "kan_logit_delta_norm_mean": float(norm.mean()),
        "kan_logit_delta_norm_p95": float(torch.quantile(norm, 0.95)),
        "kan_correct_class_delta_mean": float(correct.mean()),
        "kan_margin_contribution_mean": float(margin.mean()),
        "kan_margin_contribution_p95": float(torch.quantile(margin, 0.95)),
    }


@torch.no_grad()
def purekan_contribution_audit(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    if not isinstance(model, PureKANClassifier):
        return {}
    full_eval = evaluate(model, x, y, device=device, batch_size=batch_size)
    block_eval = evaluate(model, x, y, device=device, batch_size=batch_size, disable_kan=True)
    delta_norms: List[torch.Tensor] = []
    margin_deltas: List[torch.Tensor] = []
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size].to(device)
        yb = y[start : start + batch_size].to(device)
        full = model(xb)
        block_disabled = model(xb, disable_kan=True)
        delta = full - block_disabled
        delta_norms.append(delta.norm(dim=-1).detach().cpu())
        full_target = full.gather(1, yb.view(-1, 1)).squeeze(1)
        block_target = block_disabled.gather(1, yb.view(-1, 1)).squeeze(1)
        mask = F.one_hot(yb, full.shape[-1]).bool()
        full_other = full.masked_fill(mask, float("-inf")).max(dim=-1).values
        block_other = block_disabled.masked_fill(mask, float("-inf")).max(dim=-1).values
        margin_deltas.append(((full_target - full_other) - (block_target - block_other)).detach().cpu())
    delta_cat = torch.cat(delta_norms) if delta_norms else torch.zeros(1)
    margin_cat = torch.cat(margin_deltas) if margin_deltas else torch.zeros(1)
    return {
        "pure_block_disable_acc": block_eval["acc"],
        "pure_block_disable_loss": block_eval["loss"],
        "pure_block_disable_drop": full_eval["acc"] - block_eval["acc"],
        "pure_layerwise_logit_delta_norm": float(delta_cat.mean()),
        "pure_layerwise_margin_contribution": float(margin_cat.mean()),
        "pure_layerwise_margin_contribution_p95": float(torch.quantile(margin_cat, 0.95)),
    }


@torch.no_grad()
def branch_layer_audit(
    model: nn.Module,
    x: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    xb = x[:batch_size].to(device)
    if isinstance(model, RationalDGKANClassifier) and xb.shape[0] > 16:
        xb = xb[:16]
    if isinstance(model, PureKANClassifier):
        h = model.input_kan(xb)
        phi_values: List[torch.Tensor] = []
        cond_values: List[torch.Tensor] = []
        curvature = torch.tensor(0.0, device=device)
        for block in model.blocks:
            z = block.norm(h)
            basis, deriv = block.kan.basis_and_derivative(z)
            coeff = block.kan.coeff
            dy_dx = torch.einsum("bik,oik->boi", deriv, coeff)
            if dy_dx.shape[-1] == dy_dx.shape[-2]:
                eye = torch.eye(dy_dx.shape[-1], device=device).unsqueeze(0)
                jac = eye + float(block.alpha.detach()) * float(block.branch_scale) * dy_dx
                cond_values.append(torch.linalg.cond(jac.float()).clamp(max=1e6))
            phi_values.append(dy_dx.abs().flatten())
            k = torch.linspace(-1.0, 1.0, coeff.shape[-1], device=device)
            curvature = curvature + (coeff.square() * k.view(1, 1, -1).pow(4)).mean()
            branch = block.alpha * float(block.branch_scale) * torch.einsum("bik,oik->bo", basis, coeff)
            h = h + branch
        for layer in [model.input_kan, model.output_kan]:
            coeff = layer.coeff
            k = torch.linspace(-1.0, 1.0, coeff.shape[-1], device=device)
            curvature = curvature + (coeff.square() * k.view(1, 1, -1).pow(4)).mean()
        phi = torch.cat(phi_values) if phi_values else torch.tensor([0.0], device=device)
        cond_cat = torch.cat(cond_values) if cond_values else torch.tensor([1.0], device=device)
        return {
            "phi_prime_p95": float(torch.quantile(phi, 0.95).detach().cpu()),
            "phi_prime_max": float(phi.max().detach().cpu()),
            "max_jac_condition": float(cond_cat.max().detach().cpu()),
            "credit_amplification_p95": float(torch.quantile(cond_cat, 0.95).detach().cpu()),
            "curvature_energy": float(curvature.detach().cpu()),
        }
    if not isinstance(model, DGKANClassifier):
        return {}
    h = model.stem(xb)
    out: Dict[str, float] = {}
    ratios: List[float] = []
    for idx, block in enumerate(model.blocks):
        prev = h
        h, branch = block(h)
        ratio = branch.norm(dim=-1).mean() / prev.norm(dim=-1).mean().clamp_min(1e-6)
        value = float(ratio.detach().cpu())
        out[f"branch_layer_{idx}_ratio_final"] = value
        ratios.append(value)
    if ratios:
        out["branch_layer_ratio_final_mean"] = float(np.mean(ratios))
        out["branch_layer_ratio_final_std"] = float(np.std(ratios))
    return out


@torch.no_grad()
def generic_branch_layer_audit(
    model: nn.Module,
    x: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    if not hasattr(model, "stem") or not hasattr(model, "blocks"):
        return {}
    model.eval()
    xb = x[:batch_size].to(device)
    h = model.stem(xb)  # type: ignore[attr-defined]
    out: Dict[str, float] = {}
    ratios: List[float] = []
    for idx, block in enumerate(model.blocks):  # type: ignore[attr-defined]
        prev = h
        try:
            h, branch = block(h)
        except Exception:
            return out
        ratio = branch.norm(dim=-1).mean() / prev.norm(dim=-1).mean().clamp_min(1e-6)
        value = float(ratio.detach().cpu())
        out[f"branch_layer_{idx}_ratio_final"] = value
        ratios.append(value)
    if ratios:
        out["branch_layer_ratio_final_mean"] = float(np.mean(ratios))
        out["branch_layer_ratio_final_std"] = float(np.std(ratios))
    return out


def rational_safety_audit(model: nn.Module) -> Dict[str, float]:
    if isinstance(model, RationalDGKANClassifier):
        return model.rational_safety()
    return {
        "rational_denominator_min": float("nan"),
        "rational_denominator_p01": float("nan"),
        "rational_den_actual_min_batch": float("nan"),
        "rational_den_actual_p01_batch": float("nan"),
        "rational_den_actual_condition_batch": float("nan"),
        "rational_den_actual_min_grid": float("nan"),
        "rational_den_actual_p01_grid": float("nan"),
        "rational_r_prime_p95": float("nan"),
        "rational_r_double_prime_p95": float("nan"),
    }


@torch.no_grad()
def rbf_basis_occupancy_audit(model: nn.Module) -> Dict[str, float]:
    if not hasattr(model, "kan_layers"):
        return {}
    means: List[torch.Tensor] = []
    out_of_grid: List[torch.Tensor] = []
    for layer in model.kan_layers():  # type: ignore[attr-defined]
        if not isinstance(layer, RBFDense):
            continue
        if layer.last_basis_mean is not None:
            means.append(layer.last_basis_mean.detach().flatten().float().cpu())
        if layer.last_input is not None:
            centers = layer.centers.detach()
            lo = float(centers.min().detach().cpu())
            hi = float(centers.max().detach().cpu())
            values = layer.last_input.detach().flatten().float().cpu()
            out_of_grid.append(((values < lo) | (values > hi)).float())
    if not means:
        return {}
    basis = torch.cat(means)
    outside = torch.cat(out_of_grid) if out_of_grid else torch.zeros(1)
    return {
        "rbf_basis_mean": float(basis.mean()),
        "rbf_basis_min": float(basis.min()),
        "rbf_basis_p01": float(torch.quantile(basis, 0.01)),
        "rbf_basis_dead_frac": float((basis < 1e-3).float().mean()),
        "rbf_input_out_of_grid_frac": float(outside.mean()),
    }


@torch.no_grad()
def convstem_activation_audit(model: nn.Module, x: torch.Tensor, *, device: torch.device, batch_size: int) -> Dict[str, float]:
    if not isinstance(model, ConvStemDGKANClassifier):
        return {}
    model.eval()
    h = model.stem(x[:batch_size].to(device)).detach().float()
    return {
        "convstem_output_mean": float(h.mean().detach().cpu()),
        "convstem_output_std": float(h.std(unbiased=False).detach().cpu()),
        "convstem_output_abs_p95": float(torch.quantile(h.abs().flatten(), 0.95).detach().cpu()),
    }


def expected_calibration_error(
    conf: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    bins: int = 15,
) -> float:
    if conf.numel() == 0:
        return float("nan")
    ece = torch.tensor(0.0)
    correct = (pred == target).float()
    edges = torch.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece = ece + mask.float().mean() * (conf[mask].mean() - correct[mask].mean()).abs()
    return float(ece)


@torch.no_grad()
def estimate_geometry(
    model: nn.Module,
    x: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, float]:
    model.eval()
    xb = x[:batch_size].to(device)
    if isinstance(model, RationalDGKANClassifier) and xb.shape[0] > 16:
        xb = xb[:16]
    if isinstance(model, PureKANClassifier):
        h = model.input_kan(xb)
        phi_values: List[torch.Tensor] = []
        cond_values: List[torch.Tensor] = []
        curvature = torch.tensor(0.0, device=device)
        for block in model.blocks:
            z = block.norm(h)
            basis, deriv = block.kan.basis_and_derivative(z)
            coeff = block.kan.coeff
            dy_dx = torch.einsum("bik,oik->boi", deriv, coeff)
            if dy_dx.shape[-1] == dy_dx.shape[-2]:
                eye = torch.eye(dy_dx.shape[-1], device=device).unsqueeze(0)
                jac = eye + float(block.alpha.detach()) * float(block.branch_scale) * dy_dx
                cond_values.append(torch.linalg.cond(jac.float()).clamp(max=1e6))
            phi_values.append(dy_dx.abs().flatten())
            k = torch.linspace(-1.0, 1.0, coeff.shape[-1], device=device)
            curvature = curvature + (coeff.square() * k.view(1, 1, -1).pow(4)).mean()
            branch = block.alpha * float(block.branch_scale) * torch.einsum("bik,oik->bo", basis, coeff)
            h = h + branch
        for layer in [model.input_kan, model.output_kan]:
            coeff = layer.coeff
            k = torch.linspace(-1.0, 1.0, coeff.shape[-1], device=device)
            curvature = curvature + (coeff.square() * k.view(1, 1, -1).pow(4)).mean()
        phi = torch.cat(phi_values) if phi_values else torch.tensor([0.0], device=device)
        cond_cat = torch.cat(cond_values) if cond_values else torch.tensor([1.0], device=device)
        return {
            "phi_prime_p95": float(torch.quantile(phi, 0.95).detach().cpu()),
            "phi_prime_max": float(phi.max().detach().cpu()),
            "max_jac_condition": float(cond_cat.max().detach().cpu()),
            "credit_amplification_p95": float(torch.quantile(cond_cat, 0.95).detach().cpu()),
            "curvature_energy": float(curvature.detach().cpu()),
        }
    h = model.stem(xb)
    if isinstance(model, RationalDGKANClassifier):
        phi_values: List[torch.Tensor] = []
        cond_values: List[torch.Tensor] = []
        curvature = torch.tensor(0.0, device=device)
        for block in model.blocks:
            with torch.enable_grad():
                z = block.norm(h).detach().requires_grad_(True)
                act = block.kan._ensure_act(z)
                y = act(z.unsqueeze(1)).squeeze(1)
                deriv = torch.autograd.grad(y.sum(), z, retain_graph=False, create_graph=False)[0]
                weight = block.kan.linear.weight.detach()
                dy_dx = weight.unsqueeze(0) * deriv.detach().unsqueeze(1)
                branch = (float(block.alpha.detach()) * float(block.branch_scale) * block.kan(z)).detach()
            if dy_dx.shape[-1] == dy_dx.shape[-2]:
                eye = torch.eye(dy_dx.shape[-1], device=device).unsqueeze(0)
                jac = eye + float(block.alpha.detach()) * float(block.branch_scale) * dy_dx
                cond_values.append(torch.linalg.cond(jac.float()).clamp(max=1e6))
            phi_values.append(dy_dx.abs().flatten())
            act_module = block.kan.act
            if act_module is not None:
                numerator = getattr(act_module, "weight_numerator", None)
                denominator = getattr(act_module, "weight_denominator", None)
                if numerator is not None:
                    k = torch.linspace(-1.0, 1.0, numerator.shape[-1], device=device)
                    curvature = curvature + (numerator.detach().square() * k.view(1, -1).pow(4)).mean()
                if denominator is not None:
                    k = torch.linspace(-1.0, 1.0, denominator.shape[-1], device=device)
                    curvature = curvature + (denominator.detach().square() * k.view(1, -1).pow(4)).mean()
            h = h + branch
        phi = torch.cat(phi_values) if phi_values else torch.tensor([0.0], device=device)
        cond_cat = torch.cat(cond_values) if cond_values else torch.tensor([1.0], device=device)
        return {
            "phi_prime_p95": float(torch.quantile(phi, 0.95).detach().cpu()),
            "phi_prime_max": float(phi.max().detach().cpu()),
            "max_jac_condition": float(cond_cat.max().detach().cpu()),
            "credit_amplification_p95": float(torch.quantile(cond_cat, 0.95).detach().cpu()),
            "curvature_energy": float(curvature.detach().cpu()),
        }

    phi_values: List[torch.Tensor] = []
    cond_values: List[torch.Tensor] = []
    curvature = torch.tensor(0.0, device=device)
    for block in model.blocks:
        z = block.norm(h)
        basis, deriv = block.kan.basis_and_derivative(z)
        coeff = block.kan.coeff
        dy_dx = torch.einsum("bik,oik->boi", deriv, coeff)
        if dy_dx.shape[-1] == dy_dx.shape[-2]:
            eye = torch.eye(dy_dx.shape[-1], device=device).unsqueeze(0)
            jac = eye + float(block.alpha.detach()) * float(block.branch_scale) * dy_dx
            cond = torch.linalg.cond(jac.float()).clamp(max=1e6)
            cond_values.append(cond)
        phi_values.append(dy_dx.abs().flatten())
        k = torch.linspace(-1.0, 1.0, coeff.shape[-1], device=device)
        curvature = curvature + (coeff.square() * k.view(1, 1, -1).pow(4)).mean()
        branch = block.alpha * float(block.branch_scale) * torch.einsum("bik,oik->bo", basis, coeff)
        h = h + branch
    phi = torch.cat(phi_values) if phi_values else torch.tensor([0.0], device=device)
    cond_cat = torch.cat(cond_values) if cond_values else torch.tensor([1.0], device=device)
    return {
        "phi_prime_p95": float(torch.quantile(phi, 0.95).detach().cpu()),
        "phi_prime_max": float(phi.max().detach().cpu()),
        "max_jac_condition": float(cond_cat.max().detach().cpu()),
        "credit_amplification_p95": float(torch.quantile(cond_cat, 0.95).detach().cpu()),
        "curvature_energy": float(curvature.detach().cpu()),
    }


def maybe_switch_geometry_schedule(
    state: RuntimeState,
    cfg: TrainConfig,
    geom: Dict[str, float],
    *,
    epoch: int,
    step_idx: int,
    total_steps: int,
) -> None:
    if (cfg.branch_schedule not in {"loss_aware", "geometry_aware"} and not cfg.gafu_v3_enabled) or state.branch_switched:
        return
    progress = step_idx / max(1, total_steps)
    reasons: List[str] = []
    if progress >= cfg.branch_max_active_frac:
        reasons.append("max_active_frac")
    if epoch + 1 >= cfg.geometry_min_epochs:
        if cfg.branch_schedule == "geometry_aware":
            if geom["phi_prime_p95"] > cfg.geometry_phi_high:
                reasons.append("phi_high")
            if geom["max_jac_condition"] > cfg.geometry_jac_high:
                reasons.append("jac_high")
        elif geom["max_jac_condition"] > cfg.geometry_jac_high * 1.2:
            reasons.append("loss_plateau_proxy")
    if reasons:
        switch_branch_state(
            state,
            cfg,
            step_idx=step_idx,
            total_steps=total_steps,
            reason="|".join(reasons),
        )


def build_train_config_for_method(
    method: str,
    *,
    dataset: str,
    seed: int,
    base: Optional[TrainConfig] = None,
    **overrides: Any,
) -> TrainConfig:
    cfg = TrainConfig(dataset=dataset, method=method, seed=seed) if base is None else TrainConfig(**asdict(base))
    cfg.dataset = dataset
    cfg.method = method
    cfg.seed = seed
    method_key = method.lower().replace("-", "_")

    if method_key in {"adamw", "mlp_adamw"}:
        cfg.metric_mode = "grid"
        cfg.branch_schedule = "none"
        cfg.momentum_mu = 0.0
        cfg.snr_enabled = False
    elif method_key in {"purekan_adamw", "pure_kan_adamw"}:
        cfg.method = method
        cfg.model_type = "pure_kan"
        cfg.alpha_mode = "fixed1"
        cfg.branch_schedule = "none"
        cfg.metric_mode = "grid"
        cfg.unified_optimizer_mode = "adamw"
    elif method_key in {"purekan_ufull", "pure_kan_ufull"}:
        cfg.method = method
        cfg.model_type = "pure_kan"
        cfg.alpha_mode = "fixed1"
        cfg.gafu_v3_enabled = True
        cfg.metric_mode = "grid"
        cfg.branch_schedule = "none"
        cfg.warmup_frac = 0.0
        cfg.v3_phase_mode = "hard"
        cfg.v3_metric_active = "full_sobolev_gram"
        cfg.v3_metric_transition = "full_sobolev_gram"
        cfg.v3_metric_geometry = "full_sobolev_gram"
        cfg.branch_boost = 1.0
        cfg.coeff_lr_boost = 1.0
        cfg.coeff_lr_final_mult = 1.0
        cfg.branch_max_active_frac = 0.0
        cfg.branch_final_scale = 1.0
        cfg.unified_optimizer_mode = "hybrid"
    elif method_key in {"geometry", "geometry_current", "current_geometry_aware", "analyticadj"}:
        cfg.method = method
        cfg.metric_mode = "grid"
        cfg.branch_schedule = "geometry_aware"
        cfg.branch_boost = 2.0 if "fashion" in dataset.lower() else 1.5
        cfg.coeff_lr_boost = 1.5 if "fashion" in dataset.lower() else 1.2
        cfg.branch_max_active_frac = 0.50 if "fashion" in dataset.lower() else 0.35
        cfg.branch_final_scale = 0.8 if "fashion" in dataset.lower() else 0.9
        cfg.coeff_lr_final_mult = 1.0
    elif method_key in {"rational_adamw", "rational_dgkan_adamw", "kat_adamw"}:
        cfg.method = method
        cfg.model_type = "rational_dgkan"
        cfg.alpha_mode = "fixed1"
        cfg.branch_schedule = "none"
        cfg.branch_final_scale = 1.0
        cfg.metric_mode = "grid"
        cfg.unified_optimizer_mode = "adamw"
    elif method_key in {"rational_ufull", "rational_dgkan_ufull", "kat_ufull"}:
        cfg.method = method
        cfg.model_type = "rational_dgkan"
        cfg.alpha_mode = "fixed1"
        cfg.gafu_v3_enabled = True
        cfg.metric_mode = "grid"
        cfg.branch_schedule = "none"
        cfg.warmup_frac = 0.0
        cfg.v3_phase_mode = "hard"
        cfg.v3_metric_active = "full_sobolev_gram"
        cfg.v3_metric_transition = "full_sobolev_gram"
        cfg.v3_metric_geometry = "full_sobolev_gram"
        cfg.branch_boost = 1.0
        cfg.coeff_lr_boost = 1.0
        cfg.coeff_lr_final_mult = 1.0
        cfg.branch_max_active_frac = 0.0
        cfg.branch_final_scale = 1.0
        cfg.unified_optimizer_mode = "hybrid"
    elif method_key in {"convstem_dgkan_adamw"}:
        cfg.method = method
        cfg.model_type = "convstem_dgkan"
        cfg.alpha_mode = "fixed1"
        cfg.branch_schedule = "none"
        cfg.branch_final_scale = 1.0
        cfg.metric_mode = "grid"
        cfg.unified_optimizer_mode = "adamw"
    elif method_key in {"convstem_dgkan_ufull"}:
        cfg.method = method
        cfg.model_type = "convstem_dgkan"
        cfg.alpha_mode = "fixed1"
        cfg.gafu_v3_enabled = True
        cfg.metric_mode = "grid"
        cfg.branch_schedule = "none"
        cfg.warmup_frac = 0.0
        cfg.v3_phase_mode = "hard"
        cfg.v3_metric_active = "full_sobolev_gram"
        cfg.v3_metric_transition = "full_sobolev_gram"
        cfg.v3_metric_geometry = "full_sobolev_gram"
        cfg.v3_alpha_fast = 0.15
        cfg.v3_beta_fast = 0.02
        cfg.v3_alpha_geo = 0.15
        cfg.v3_beta_geo = 0.02
        cfg.v3_gram_rho = 1e-3
        cfg.unified_optimizer_mode = "hybrid"
    elif method_key in {"static_top", "fullbp", "firstorder", "identityadj"}:
        cfg.metric_mode = "grid"
        cfg.branch_schedule = "none"
        cfg.branch_final_scale = 1.0
        if method_key == "firstorder":
            cfg.credit_mode = "first_order"
        if method_key == "identityadj":
            cfg.credit_mode = "identity"
            cfg.coeff_lr *= 0.8
            cfg.branch_final_scale = 0.8
    elif method_key in {"no_grid", "nogrid", "no_grid_stress"}:
        cfg.metric_mode = "no_grid"
        cfg.branch_schedule = "none"
        cfg.branch_final_scale = 1.0
        cfg.coeff_lr *= 1.1
    elif method_key in {"fgo_v3_clean", "fgo_v3_clean_full", "gfu_momentum"}:
        cfg.metric_mode = "fixed_data"
        cfg.lambda_final = 0.10
        cfg.momentum_mu = 0.8
        cfg.trust_radius = 0.15
        cfg.branch_schedule = "geometry_aware"
        cfg.branch_boost = 1.5
        cfg.coeff_lr_boost = 1.2
        cfg.branch_max_active_frac = 0.35
        cfg.branch_final_scale = 0.9
    elif method_key in {"fgo_v4_clean", "data_pulse", "data_pulse_reanchor"}:
        cfg.metric_mode = "pulse"
        cfg.lambda_early = 1.0
        cfg.lambda_final = 0.05
        cfg.pulse_frac = 0.25
        cfg.anchor_decay = "cosine"
        cfg.momentum_mu = 0.8
        cfg.prox_lambda = 3e-5
        cfg.trust_radius = 0.15
        cfg.branch_schedule = "geometry_aware"
        cfg.branch_boost = 1.5
        cfg.coeff_lr_boost = 1.2
        cfg.branch_max_active_frac = 0.35
        cfg.branch_final_scale = 0.9
    elif method_key in {"snr_previous_best", "snr_full", "fgo_v4_noise"}:
        cfg.metric_mode = "grid"
        cfg.branch_schedule = "geometry_aware"
        cfg.snr_enabled = True
        cfg.snr_grouping = "edge"
        cfg.snr_update_interval = 1
        cfg.snr_sample_frac = 1.0
        cfg.snr_percentile = 60.0
        cfg.spectral_lambda = 3e-4
        cfg.prox_lambda = 3e-5
    elif method_key in {"snr_lite", "snr_lite_output", "snr_lite_layer"}:
        cfg.metric_mode = "grid"
        cfg.branch_schedule = "geometry_aware"
        cfg.snr_enabled = True
        cfg.snr_grouping = "output_channel" if "output" in method_key else "layer"
        cfg.snr_update_interval = 4
        cfg.snr_sample_frac = 0.5
        cfg.snr_percentile = 60.0
        cfg.spectral_lambda = 1e-4
        cfg.prox_lambda = 3e-5
    else:
        raise ValueError(f"unknown method: {method}")

    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"TrainConfig has no field {key}")
        setattr(cfg, key, value)
    return cfg


def train_one(cfg: TrainConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    bundle = load_vision_bundle(
        cfg.dataset,
        data_root=cfg.data_root,
        train_size=cfg.train_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        seed=cfg.seed,
        label_noise=cfg.label_noise,
        download=cfg.download,
        allow_fake_data=cfg.allow_fake_data,
    )

    train_method = cfg.optimizer_method or cfg.method
    method_key = train_method.lower().replace("-", "_")
    if method_key == "mlp_adamw":
        model: nn.Module = MLPClassifier(
            bundle.input_dim, bundle.num_classes, hidden_dim=cfg.hidden_dim, depth=cfg.depth
        ).to(device)
    elif cfg.model_type.lower().replace("-", "_") == "pure_kan" or method_key in {"purekan_adamw", "pure_kan_adamw", "purekan_ufull", "pure_kan_ufull"}:
        model = PureKANClassifier(
            bundle.input_dim,
            bundle.num_classes,
            hidden_dim=cfg.hidden_dim,
            depth=cfg.depth,
            basis_count=cfg.basis_count,
            alpha_init=cfg.alpha_init,
            alpha_mode=cfg.alpha_mode,
        ).to(device)
    elif cfg.model_type.lower().replace("-", "_") == "rational_dgkan" or method_key in {"rational_adamw", "rational_dgkan_adamw", "kat_adamw"}:
        model = RationalDGKANClassifier(
            bundle.input_dim,
            bundle.num_classes,
            hidden_dim=cfg.hidden_dim,
            depth=cfg.depth,
            alpha_init=cfg.alpha_init,
            alpha_mode=cfg.alpha_mode,
            head_hidden=cfg.head_hidden,
            rational_groups=cfg.rational_groups,
            rational_mode=cfg.rational_mode,
            rational_backend=cfg.rational_backend,
        ).to(device)
    elif cfg.model_type.lower().replace("-", "_") == "convstem_dgkan" or method_key in {"convstem_dgkan_adamw", "convstem_dgkan_ufull"}:
        model = ConvStemDGKANClassifier(
            bundle.input_dim,
            bundle.num_classes,
            hidden_dim=cfg.hidden_dim,
            depth=cfg.depth,
            basis_count=cfg.basis_count,
            alpha_init=cfg.alpha_init,
            alpha_mode=cfg.alpha_mode,
        ).to(device)
    else:
        model = DGKANClassifier(
            bundle.input_dim,
            bundle.num_classes,
            hidden_dim=cfg.hidden_dim,
            depth=cfg.depth,
            basis_count=cfg.basis_count,
            alpha_init=cfg.alpha_init,
            alpha_mode=cfg.alpha_mode,
            head_hidden=cfg.head_hidden,
        ).to(device)

    state = RuntimeState(current_branch_scale=cfg.branch_final_scale)
    functional_model = isinstance(model, (DGKANClassifier, RationalDGKANClassifier, ConvStemDGKANClassifier, PureKANClassifier))
    branch_model = isinstance(model, (DGKANClassifier, RationalDGKANClassifier, ConvStemDGKANClassifier, PureKANClassifier))
    if branch_model:
        model.set_branch_scale(cfg.branch_final_scale)
    if isinstance(model, RationalDGKANClassifier):
        with torch.no_grad():
            model(torch.zeros(1, bundle.input_dim, device=device))

    uo_mode = cfg.unified_optimizer_mode.lower().replace("-", "_")
    coeff_params = coefficient_named_params(model) if functional_model else []
    rest_params = non_coefficient_params(model) if functional_model else list(model.parameters())
    if isinstance(model, RationalDGKANClassifier) and cfg.rational_branch_functional:
        branch_linear_ids = {
            id(param)
            for block in model.blocks
            for param in list(block.kan.linear.parameters())
        }
        rest_params = [p for p in rest_params if id(p) not in branch_linear_ids]
    coeff_param_values = [p for _, p in coeff_params]
    afu_mode = uo_mode in {"afu", "all_functional", "all_functional_update"} and isinstance(model, DGKANClassifier)

    if afu_mode:
        rest_params = _afu_rest_params(model, cfg)
        opt = None
        rest_opt = (
            torch.optim.AdamW(rest_params, lr=cfg.rest_lr, weight_decay=cfg.rest_weight_decay)
            if rest_params
            else None
        )
    elif uo_mode in {"pgadam", "uo_pgadam", "norm_pgadam", "uo_norm_pgadam"} and functional_model:
        opt = torch.optim.AdamW(
            [
                {"params": rest_params, "lr": cfg.rest_lr, "weight_decay": cfg.rest_weight_decay, "role": "rest"},
                {"params": coeff_param_values, "lr": cfg.coeff_lr, "weight_decay": cfg.weight_decay, "role": "coeff"},
            ]
        )
        rest_opt = None
    elif uo_mode in {"all_functional_rest_sgd", "allfunctionalrestsgd"} and functional_model:
        opt = torch.optim.SGD(
            [
                {"params": rest_params, "lr": cfg.rest_lr, "weight_decay": cfg.rest_weight_decay, "role": "rest"},
                {"params": coeff_param_values, "lr": cfg.coeff_lr, "weight_decay": cfg.weight_decay, "role": "coeff"},
            ]
        )
        rest_opt = None
    elif uo_mode in {"postadam", "uo_postadam", "postadam_trust030", "uo_postadam_trust030"} and functional_model:
        opt = None
        rest_opt = torch.optim.AdamW(rest_params, lr=cfg.rest_lr, weight_decay=cfg.rest_weight_decay)
    elif method_key in ADAMW_METHOD_KEYS:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        rest_opt = None
    else:
        opt = None
        rest_opt = (
            torch.optim.AdamW(rest_params, lr=cfg.rest_lr, weight_decay=cfg.rest_weight_decay)
            if rest_params
            else None
        )

    total_steps = cfg.epochs * max(1, math.ceil(len(bundle.x_train) / cfg.batch_size))
    step_idx = 0
    val_losses: List[float] = []
    val_accs: List[float] = []
    epoch_times: List[float] = []
    branch_ratios: List[float] = []
    data_lambdas: List[float] = []
    start_time = time.perf_counter()
    last_geom = {
        "phi_prime_p95": float("nan"),
        "phi_prime_max": float("nan"),
        "max_jac_condition": float("nan"),
        "credit_amplification_p95": float("nan"),
        "curvature_energy": float("nan"),
    }

    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()
        model.train()
        for idx in iter_minibatches(len(bundle.x_train), cfg.batch_size, cfg.seed + epoch * 997):
            progress = step_idx / max(1, total_steps)
            if (
                (cfg.branch_schedule in {"loss_aware", "geometry_aware"} or cfg.gafu_v3_enabled)
                and not state.branch_switched
                and progress >= cfg.branch_max_active_frac
            ):
                switch_branch_state(
                    state,
                    cfg,
                    step_idx=step_idx,
                    total_steps=total_steps,
                    reason="max_active_frac",
                )
            if branch_model:
                set_schedule_for_step(model, cfg, state, progress=progress, step_idx=step_idx, total_steps=total_steps)
            if uo_mode in {"pgadam", "uo_pgadam", "norm_pgadam", "uo_norm_pgadam", "all_functional_rest_sgd", "allfunctionalrestsgd"}:
                set_optimizer_group_lrs(opt, cfg, state, progress)
            else:
                set_optimizer_lr(
                    opt,
                    cfg.lr,
                    scheduled_lr_factor(cfg.lr_schedule, progress, cfg.lr_final_mult, cfg.lr_decay_start_frac),
                )
                set_optimizer_lr(
                    rest_opt,
                    cfg.rest_lr,
                    scheduled_lr_factor(
                        cfg.rest_lr_schedule,
                        progress,
                        cfg.rest_lr_final_mult,
                        cfg.lr_decay_start_frac,
                    ),
                )
            xb = bundle.x_train[idx].to(device)
            yb = bundle.y_train[idx].to(device)
            if afu_mode:
                model.zero_grad(set_to_none=True)
            elif opt is not None:
                opt.zero_grad(set_to_none=True)
            if rest_opt is not None and not afu_mode:
                rest_opt.zero_grad(set_to_none=True)
            try:
                forward_out = model(xb, return_branch=True)
                if isinstance(forward_out, tuple) and len(forward_out) == 2:
                    logits, branch_ratio = forward_out
                else:
                    logits = forward_out
                    branch_ratio = torch.tensor(0.0, device=device)
            except TypeError:
                logits = model(xb)
                branch_ratio = torch.tensor(0.0, device=device)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            if functional_model and uo_mode in {"pgadam", "uo_pgadam", "norm_pgadam", "uo_norm_pgadam"}:
                _, rest_param_norm = _param_group_norms(rest_params)
                rest_grad_norm, _ = _param_group_norms(rest_params)
                state.uo_rest_grad_norms.append(rest_grad_norm)
                coeff_before = _snapshot_params(coeff_param_values)
                rest_before = _snapshot_params(rest_params)
                _uo_precondition_coeff_grads(
                    model,
                    cfg,
                    state,
                    normalize=uo_mode in {"norm_pgadam", "uo_norm_pgadam"} or cfg.kan_precond_grad_normalize,
                )
                assert opt is not None
                opt.step()
                _record_param_updates(state, coeff_before, rest_before)
                _uo_record_adam_moments(opt, model, state)
                _ = rest_param_norm
            elif functional_model and uo_mode in {"all_functional_rest_sgd", "allfunctionalrestsgd"}:
                rest_grad_norm, _ = _param_group_norms(rest_params)
                state.uo_rest_grad_norms.append(rest_grad_norm)
                coeff_before = _snapshot_params(coeff_param_values)
                rest_before = _snapshot_params(rest_params)
                _uo_precondition_coeff_grads(model, cfg, state, normalize=False)
                assert opt is not None
                opt.step()
                _record_param_updates(state, coeff_before, rest_before)
            elif functional_model and uo_mode in {"postadam", "uo_postadam", "postadam_trust030", "uo_postadam_trust030"}:
                rest_grad_norm, _ = _param_group_norms(rest_params)
                state.uo_rest_grad_norms.append(rest_grad_norm)
                coeff_before = _snapshot_params(coeff_param_values)
                rest_before = _snapshot_params(rest_params)
                assert rest_opt is not None
                rest_opt.step()
                unified_postadam_coeff_step(model, cfg, state, step_idx=step_idx, total_steps=total_steps)
                _record_param_updates(state, coeff_before, rest_before)
            elif afu_mode:
                rest_grad_norm, _ = _param_group_norms(rest_params)
                state.uo_rest_grad_norms.append(rest_grad_norm)
                if rest_opt is not None:
                    rest_opt.step()
                assert isinstance(model, DGKANClassifier)
                afu_nonkan_step(model, cfg, state, xb=xb, progress=progress)
                functional_coeff_step(model, cfg, state, step_idx=step_idx, total_steps=total_steps)
            elif opt is not None:
                opt.step()
            else:
                assert functional_model
                if rest_opt is not None:
                    rest_opt.step()
                functional_coeff_step(model, cfg, state, step_idx=step_idx, total_steps=total_steps)
            branch_ratios.append(float(branch_ratio.detach().cpu()))
            data_lambdas.append(state.last_data_lambda)
            step_idx += 1

        val = evaluate(model, bundle.x_val, bundle.y_val, device=device, batch_size=cfg.eval_batch_size)
        val_losses.append(val["loss"])
        val_accs.append(val["acc"])
        epoch_times.append(time.perf_counter() - epoch_start)
        if functional_model:
            last_geom = estimate_geometry(
                model, bundle.x_val, device=device, batch_size=min(cfg.audit_batch_size, len(bundle.x_val))
            )
            maybe_switch_geometry_schedule(
                state, cfg, last_geom, epoch=epoch, step_idx=step_idx, total_steps=total_steps
            )

    wall = time.perf_counter() - start_time
    val = evaluate(model, bundle.x_val, bundle.y_val, device=device, batch_size=cfg.eval_batch_size)
    test = evaluate(model, bundle.x_test, bundle.y_test, device=device, batch_size=cfg.eval_batch_size)
    train_noisy = evaluate(
        model, bundle.x_train[: min(2048, len(bundle.x_train))], bundle.y_train[: min(2048, len(bundle.y_train))],
        device=device, batch_size=cfg.eval_batch_size
    )
    train_clean = evaluate(
        model,
        bundle.x_train[: min(2048, len(bundle.x_train))],
        bundle.y_train_clean[: min(2048, len(bundle.y_train_clean))],
        device=device,
        batch_size=cfg.eval_batch_size,
    )
    supports_disable_kan = isinstance(model, (DGKANClassifier, RationalDGKANClassifier, ConvStemDGKANClassifier))
    if functional_model:
        final_geom = estimate_geometry(
            model, bundle.x_val, device=device, batch_size=min(cfg.audit_batch_size, len(bundle.x_val))
        )
    else:
        final_geom = last_geom
    if supports_disable_kan:
        no_kan_eval = evaluate(
            model,
            bundle.x_test,
            bundle.y_test,
            device=device,
            batch_size=cfg.eval_batch_size,
            disable_kan=True,
        )
        no_kan_val_eval = evaluate(
            model,
            bundle.x_val,
            bundle.y_val,
            device=device,
            batch_size=cfg.eval_batch_size,
            disable_kan=True,
        )
        contribution = kan_contribution_audit(
            model,
            bundle.x_test[: min(len(bundle.x_test), cfg.audit_batch_size * 4)],
            bundle.y_test[: min(len(bundle.y_test), cfg.audit_batch_size * 4)],
            device=device,
            batch_size=cfg.eval_batch_size,
        )
        branch_layers = (
            branch_layer_audit(model, bundle.x_val, device=device, batch_size=min(cfg.audit_batch_size, len(bundle.x_val)))
            if isinstance(model, (DGKANClassifier, ConvStemDGKANClassifier))
            else generic_branch_layer_audit(model, bundle.x_val, device=device, batch_size=min(cfg.audit_batch_size, len(bundle.x_val)))
        )
    else:
        no_kan_eval = {"acc": test["acc"], "loss": test["loss"], "ece": test["ece"]}
        no_kan_val_eval = {"acc": val["acc"], "loss": val["loss"], "ece": val["ece"]}
        contribution = kan_contribution_audit(
            model,
            bundle.x_test[: min(len(bundle.x_test), cfg.audit_batch_size * 4)],
            bundle.y_test[: min(len(bundle.y_test), cfg.audit_batch_size * 4)],
            device=device,
            batch_size=cfg.eval_batch_size,
        )
        branch_layers = {}
    pure_contribution = purekan_contribution_audit(
        model,
        bundle.x_test[: min(len(bundle.x_test), cfg.audit_batch_size * 4)],
        bundle.y_test[: min(len(bundle.y_test), cfg.audit_batch_size * 4)],
        device=device,
        batch_size=cfg.eval_batch_size,
    )
    rational_safety = rational_safety_audit(model)
    rbf_occupancy = rbf_basis_occupancy_audit(model)
    convstem_audit = convstem_activation_audit(
        model,
        bundle.x_val,
        device=device,
        batch_size=min(cfg.audit_batch_size, len(bundle.x_val)),
    )

    val_auc = float(np.mean(val_losses)) if val_losses else val["loss"]
    val_acc_auc = float(np.mean(val_accs)) if val_accs else val["acc"]
    train_steps_per_epoch = max(1, math.ceil(len(bundle.x_train) / cfg.batch_size))
    switch_epoch_idx = state.branch_switch_step // train_steps_per_epoch if state.branch_switch_step >= 0 else -1
    if 0 <= switch_epoch_idx < len(val_losses) - 1:
        transition_loss_jump = float(val_losses[switch_epoch_idx + 1] - val_losses[switch_epoch_idx])
        transition_acc_jump = float(val_accs[switch_epoch_idx + 1] - val_accs[switch_epoch_idx])
    else:
        transition_loss_jump = float("nan")
        transition_acc_jump = float("nan")
    clean_noise_mask = bundle.y_train != bundle.y_train_clean
    if clean_noise_mask.any():
        corrupted_x = bundle.x_train[clean_noise_mask]
        corrupted_clean = bundle.y_train_clean[clean_noise_mask]
        corrupted_noisy = bundle.y_train[clean_noise_mask]
        corr_clean = evaluate(
            model, corrupted_x, corrupted_clean, device=device, batch_size=cfg.eval_batch_size
        )["acc"]
        corr_noisy = evaluate(
            model, corrupted_x, corrupted_noisy, device=device, batch_size=cfg.eval_batch_size
        )["acc"]
    else:
        corr_clean = float("nan")
        corr_noisy = float("nan")

    step_time_ms = 1000.0 * wall / max(1, step_idx)
    epoch_time_sec_mean = float(np.mean(epoch_times)) if epoch_times else wall
    epoch_time_sec_p95 = float(np.quantile(epoch_times, 0.95)) if epoch_times else wall
    samples_per_sec = float(cfg.train_size * cfg.epochs / max(1e-9, wall))
    alphas: List[float] = []
    alpha_trainable = 0
    if branch_model:
        for block in model.blocks:
            alphas.append(float(block.alpha.detach().cpu()))
            if isinstance(block.alpha, nn.Parameter):
                alpha_trainable = 1
    alpha_mean = float(np.mean(alphas)) if alphas else float("nan")
    alpha_std = float(np.std(alphas)) if alphas else float("nan")
    update_metric_mean = float(np.mean(state.update_metric_norms)) if state.update_metric_norms else 0.0
    update_metric_p95 = float(np.quantile(state.update_metric_norms, 0.95)) if state.update_metric_norms else 0.0
    update_over_coeff_mean = float(np.mean(state.update_over_coeff_norms)) if state.update_over_coeff_norms else 0.0
    update_over_coeff_p95 = float(np.quantile(state.update_over_coeff_norms, 0.95)) if state.update_over_coeff_norms else 0.0
    raw_grad_norm_mean = float(np.mean(state.raw_grad_norms)) if state.raw_grad_norms else 0.0
    precond_norm_mean = float(np.mean(state.precond_direction_norms)) if state.precond_direction_norms else 0.0
    dir_cos_mean = float(np.mean(state.direction_cos_diagfast_fullgeo)) if state.direction_cos_diagfast_fullgeo else float("nan")
    dir_norm_ratio_mean = (
        float(np.mean(state.direction_norm_ratio_fullgeo_diagfast))
        if state.direction_norm_ratio_fullgeo_diagfast
        else float("nan")
    )
    current_norm_ratio_mean = (
        float(np.mean(state.direction_norm_ratio_current_diagfast))
        if state.direction_norm_ratio_current_diagfast
        else float("nan")
    )
    diagfast_metric_norm_mean = (
        float(np.mean(state.direction_metric_norm_diagfast))
        if state.direction_metric_norm_diagfast
        else float("nan")
    )
    fullgeo_metric_norm_mean = (
        float(np.mean(state.direction_metric_norm_fullgeo))
        if state.direction_metric_norm_fullgeo
        else float("nan")
    )
    current_metric_norm_mean = (
        float(np.mean(state.direction_metric_norm_current))
        if state.direction_metric_norm_current
        else float("nan")
    )
    def _mean(values: Sequence[float], default: float = 0.0) -> float:
        return float(np.mean(values)) if values else default

    def _p95(values: Sequence[float], default: float = 0.0) -> float:
        return float(np.quantile(values, 0.95)) if values else default

    def _role_stat(store: Dict[str, List[float]], role: str, default: float = 0.0) -> float:
        vals = store.get(role, [])
        vals = [float(v) for v in vals if math.isfinite(float(v))]
        return float(np.mean(vals)) if vals else default

    uo_raw_mean = _mean(state.uo_kan_raw_grad_norms)
    uo_precond_mean = _mean(state.uo_kan_precond_grad_norms)
    uo_precond_ratio_mean = _mean(state.uo_kan_precond_over_raw_norms)
    uo_rest_grad_mean = _mean(state.uo_rest_grad_norms)
    uo_kan_update_mean = _mean(state.uo_kan_update_norms)
    uo_rest_update_mean = _mean(state.uo_rest_update_norms)
    uo_kan_update_over_param_mean = _mean(state.uo_kan_update_over_param_norms)
    uo_rest_update_over_param_mean = _mean(state.uo_rest_update_over_param_norms)
    uo_update_ratio_mean = _mean(state.uo_kan_update_over_rest_update)
    uo_bad_step_rate = state.uo_bad_step_count / max(1, state.uo_update_count)

    coeff_param_ids = {id(p) for p in coeff_param_values}
    coeff_names = [name for name, _ in coeff_params]
    learnable_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    learnable_kan_params = sum(p.numel() for _, p in coeff_params if p.requires_grad)
    learnable_nonkan_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad and id(p) not in coeff_param_ids
    )
    coeff_param_names_hash = hashlib.md5("|".join(sorted(coeff_names)).encode("utf-8")).hexdigest()[:12]
    coeff_param_numel_total = sum(p.numel() for _, p in coeff_params if p.requires_grad)
    coeff_param_seen_ratio = state.pure_seen_param_numel / max(1, coeff_param_numel_total) if isinstance(model, PureKANClassifier) else float("nan")

    def _state_mb_for_optimizer(opt_obj: Optional[torch.optim.Optimizer], *, coeff: bool) -> float:
        if opt_obj is None:
            return 0.0
        total = 0
        for p, opt_state in opt_obj.state.items():
            if (id(p) in coeff_param_ids) != coeff:
                continue
            for value in opt_state.values():
                if isinstance(value, torch.Tensor):
                    total += value.numel() * value.element_size()
        return total / 1e6

    adam_state_kan_mb = _state_mb_for_optimizer(opt, coeff=True) + sum(
        t.numel() * t.element_size() for t in list(state.uo_m.values()) + list(state.uo_v.values())
    ) / 1e6
    adam_state_rest_mb = _state_mb_for_optimizer(opt, coeff=False) + _state_mb_for_optimizer(rest_opt, coeff=False)
    afu_groups = ["kan_bias", "head", "stem", "ln"]

    def _afu_group_stat(store: Dict[str, List[float]], group: str, default: float = 0.0) -> float:
        vals = store.get(group, [])
        return float(np.mean(vals)) if vals else default

    def _afu_group_p95(store: Dict[str, List[float]], group: str, default: float = 0.0) -> float:
        vals = store.get(group, [])
        return float(np.quantile(vals, 0.95)) if vals else default

    kan_update_mean_for_ratio = update_metric_mean if update_metric_mean else 0.0
    nonkan_afu_update = sum(_afu_group_stat(state.afu_update_norms, group) for group in afu_groups)
    row: Dict[str, Any] = {
        "dataset": cfg.dataset,
        "method": cfg.method,
        "profile_name": cfg.method,
        "optimizer_method": cfg.optimizer_method or cfg.method,
        "optimizer_path": (
            "afu_all_functional"
            if afu_mode
            else "unified_adamw_preconditioned"
            if uo_mode in {"pgadam", "uo_pgadam", "norm_pgadam", "uo_norm_pgadam"}
            else "unified_postadam"
            if uo_mode in {"postadam", "uo_postadam", "postadam_trust030", "uo_postadam_trust030"}
            else "all_functional_rest_sgd"
            if uo_mode in {"all_functional_rest_sgd", "allfunctionalrestsgd"}
            else "adamw"
            if method_key in ADAMW_METHOD_KEYS
            else "hybrid_functional_adamw"
        ),
        "unified_optimizer_mode": cfg.unified_optimizer_mode,
        "optimizer_structure": (
            "kan_coeff_functional_nonkan_adamw"
            if uo_mode == "hybrid"
            and isinstance(model, (DGKANClassifier, RationalDGKANClassifier, ConvStemDGKANClassifier))
            and method_key
            not in ADAMW_METHOD_KEYS
            else "all_functional_group_metrics"
            if afu_mode
            else "adamw"
            if method_key in ADAMW_METHOD_KEYS
            else uo_mode
        ),
        "kan_coeff_update": (
            cfg.v3_metric_geometry
            if cfg.gafu_v3_enabled
            and isinstance(model, (DGKANClassifier, RationalDGKANClassifier, ConvStemDGKANClassifier))
            and method_key
            not in ADAMW_METHOD_KEYS
            else cfg.metric_mode
            if isinstance(model, (DGKANClassifier, RationalDGKANClassifier, ConvStemDGKANClassifier))
            and method_key
            not in ADAMW_METHOD_KEYS
            else "adamw"
        ),
        "nonkan_update": "adamw"
        if (
            rest_opt is not None
            or method_key
            in ADAMW_METHOD_KEYS
        )
        else "none",
        "kan_grad_transform": cfg.kan_grad_transform,
        "kan_precondition_before_adam": int(cfg.kan_precondition_before_adam),
        "kan_precondition_after_adam": int(cfg.kan_precondition_after_adam),
        "kan_precond_grad_normalize": int(cfg.kan_precond_grad_normalize),
        "uo_trust_radius": cfg.uo_trust_radius,
        "afu_kan_bias": int(cfg.afu_kan_bias),
        "afu_head": int(cfg.afu_head),
        "afu_stem": int(cfg.afu_stem),
        "afu_ln_scope": cfg.afu_ln_scope,
        "afu_head_lr_mult": cfg.afu_head_lr_mult,
        "afu_stem_lr_mult": cfg.afu_stem_lr_mult,
        "afu_ln_lr_mult": cfg.afu_ln_lr_mult,
        "afu_bias_lr_mult": cfg.afu_bias_lr_mult,
        "afu_cov_ema_beta": cfg.afu_cov_ema_beta,
        "afu_head_rho": cfg.afu_head_rho,
        "afu_stem_rho": cfg.afu_stem_rho,
        "afu_ln_rho": cfg.afu_ln_rho,
        "afu_bias_rho": cfg.afu_bias_rho,
        "afu_max_update_ratio": cfg.afu_max_update_ratio,
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "eval_batch_size": cfg.eval_batch_size,
        "audit_batch_size": cfg.audit_batch_size,
        "train_size": cfg.train_size,
        "val_size": cfg.val_size,
        "test_size": cfg.test_size,
        "hidden_dim": cfg.hidden_dim,
        "depth": cfg.depth,
        "basis_count": cfg.basis_count,
        "model_type": cfg.model_type,
        "kan_primitive_type": "rational"
        if isinstance(model, RationalDGKANClassifier)
        else "rbf"
        if isinstance(model, (DGKANClassifier, ConvStemDGKANClassifier, PureKANClassifier))
        else "none",
        "learnable_total_params": learnable_total_params,
        "learnable_kan_params": learnable_kan_params,
        "learnable_nonkan_params": learnable_nonkan_params,
        "coeff_param_count": len(coeff_params),
        "coeff_param_numel_total": coeff_param_numel_total,
        "coeff_param_seen_ratio": coeff_param_seen_ratio,
        "coeff_param_names_hash": coeff_param_names_hash,
        "coeff_param_names": "|".join(coeff_names[:64]),
        "purekan_nonkan_param_count": learnable_nonkan_params if isinstance(model, PureKANClassifier) else float("nan"),
        "purekan_has_linear": int(any(isinstance(module, nn.Linear) for module in model.modules())) if isinstance(model, PureKANClassifier) else 0,
        "purekan_has_layernorm_params": int(
            any(isinstance(module, nn.LayerNorm) and any(p.requires_grad for p in module.parameters()) for module in model.modules())
        )
        if isinstance(model, PureKANClassifier)
        else 0,
        "purekan_has_bias_params": int(
            any("bias" in name and p.requires_grad for name, p in model.named_parameters())
        )
        if isinstance(model, PureKANClassifier)
        else 0,
        "rational_backend": cfg.rational_backend,
        "rational_groups": cfg.rational_groups,
        "rational_mode": cfg.rational_mode,
        "rational_branch_functional": int(cfg.rational_branch_functional),
        "alpha_init": cfg.alpha_init,
        "alpha_mode": cfg.alpha_mode,
        "pure_alpha_mode": cfg.alpha_mode if isinstance(model, PureKANClassifier) else "",
        "alpha_trainable": alpha_trainable,
        "pure_alpha_trainable": alpha_trainable if isinstance(model, PureKANClassifier) else float("nan"),
        "alpha_final_mean": alpha_mean,
        "pure_alpha_mean": alpha_mean if isinstance(model, PureKANClassifier) else float("nan"),
        "alpha_final_std": alpha_std,
        "head_hidden": cfg.head_hidden,
        "lr": cfg.lr,
        "coeff_lr": cfg.coeff_lr,
        "rest_lr": cfg.rest_lr,
        "branch_schedule": cfg.branch_schedule,
        "branch_boost": cfg.branch_boost,
        "coeff_lr_boost": cfg.coeff_lr_boost,
        "coeff_lr_final_mult": cfg.coeff_lr_final_mult,
        "branch_final_scale": cfg.branch_final_scale,
        "branch_active_frac": cfg.branch_active_frac,
        "branch_max_active_frac": cfg.branch_max_active_frac,
        "lr_schedule": cfg.lr_schedule,
        "lr_final_mult": cfg.lr_final_mult,
        "lr_factor_final": scheduled_lr_factor(
            cfg.lr_schedule, 1.0, cfg.lr_final_mult, cfg.lr_decay_start_frac
        ),
        "rest_lr_schedule": cfg.rest_lr_schedule,
        "rest_lr_final_mult": cfg.rest_lr_final_mult,
        "rest_lr_factor_final": scheduled_lr_factor(
            cfg.rest_lr_schedule, 1.0, cfg.rest_lr_final_mult, cfg.lr_decay_start_frac
        ),
        "coeff_lr_schedule": cfg.coeff_lr_schedule,
        "coeff_lr_decay_final_mult": cfg.coeff_lr_decay_final_mult,
        "coeff_lr_decay_factor_final": scheduled_lr_factor(
            cfg.coeff_lr_schedule, 1.0, cfg.coeff_lr_decay_final_mult, cfg.lr_decay_start_frac
        ),
        "lr_decay_start_frac": cfg.lr_decay_start_frac,
        "gafu_v3_enabled": int(cfg.gafu_v3_enabled),
        "v3_phase_final": state.phase,
        "phase_trace_first20": ",".join(state.phase_trace_first20),
        "metric_trace_first20": ",".join(state.metric_trace_first20),
        "active_steps_actual": state.phase_seen_counts.get("ACTIVE", 0),
        "transition_steps_actual": state.phase_seen_counts.get("TRANSITION", 0),
        "geometry_steps_actual": state.phase_seen_counts.get("GEOMETRY", 0),
        "metric_active_seen": "|".join(sorted(k for k in state.metric_seen_counts if k == cfg.v3_metric_active)),
        "metric_geometry_seen": "|".join(sorted(k for k in state.metric_seen_counts if k == cfg.v3_metric_geometry)),
        "input_metric_mode_seen": "|".join(state.pure_role_metric_seen.get("input", [])),
        "block_metric_mode_seen": "|".join(state.pure_role_metric_seen.get("block", [])),
        "output_metric_mode_seen": "|".join(state.pure_role_metric_seen.get("output", [])),
        "pure_input_metric": cfg.pure_input_metric,
        "pure_block_metric": cfg.pure_block_metric,
        "pure_output_metric": cfg.pure_output_metric,
        "pure_input_lr_mult": cfg.pure_input_lr_mult,
        "pure_block_lr_mult": cfg.pure_block_lr_mult,
        "pure_output_lr_mult": cfg.pure_output_lr_mult,
        "v3_phase_switch_step": state.branch_switch_step,
        "v3_phase_switch_epoch": state.branch_switch_step / max(1, math.ceil(len(bundle.x_train) / cfg.batch_size)),
        "v3_phase_switch_reason": state.branch_switch_reason,
        "transition_loss_jump": transition_loss_jump,
        "transition_acc_jump": transition_acc_jump,
        "v3_transition_length_steps": state.transition_length_steps,
        "v3_metric_active": cfg.v3_metric_active,
        "v3_metric_transition": cfg.v3_metric_transition,
        "v3_metric_geometry": cfg.v3_metric_geometry,
        "v3_metric_condition_active": state.metric_condition_active,
        "v3_metric_condition_geometry": state.metric_condition_geometry,
        "v3_metric_eig_min_active": state.metric_eig_min_active,
        "v3_metric_eig_max_active": state.metric_eig_max_active,
        "v3_metric_eig_min_geometry": state.metric_eig_min_geometry,
        "v3_metric_eig_max_geometry": state.metric_eig_max_geometry,
        "v3_diagfast_condition": state.diagfast_condition,
        "v3_fullgeo_condition": state.fullgeo_condition,
        "v3_diagfast_eig_min": state.diagfast_eig_min,
        "v3_fullgeo_eig_min": state.fullgeo_eig_min,
        "v3_diagfast_eig_max": state.diagfast_eig_max,
        "v3_fullgeo_eig_max": state.fullgeo_eig_max,
        "v3_metric_mix_auc": state.metric_mix_sum / max(1, state.metric_mix_count),
        "v3_direction_cos_diagfast_fullgeo_mean": dir_cos_mean,
        "v3_direction_norm_ratio_fullgeo_diagfast_mean": dir_norm_ratio_mean,
        "v3_direction_norm_ratio_current_diagfast_mean": current_norm_ratio_mean,
        "v3_direction_metric_norm_diagfast_mean": diagfast_metric_norm_mean,
        "v3_direction_metric_norm_fullgeo_mean": fullgeo_metric_norm_mean,
        "v3_direction_metric_norm_current_mean": current_metric_norm_mean,
        "v3_precond_solve_time_ms": 1000.0 * state.precond_solve_time / max(1, step_idx),
        "v3_metric_build_time_ms": 1000.0 * state.metric_build_time / max(1, step_idx),
        "update_raw_grad_norm_mean": raw_grad_norm_mean,
        "update_precond_direction_norm_mean": precond_norm_mean,
        "trust_update_metric_norm_mean": update_metric_mean,
        "trust_update_metric_norm_p95": update_metric_p95,
        "trust_update_over_coeff_norm_mean": update_over_coeff_mean,
        "trust_update_over_coeff_norm_p95": update_over_coeff_p95,
        "pure_input_update_norm": _role_stat(state.pure_role_update_norms, "input"),
        "pure_block_update_norm_mean": _role_stat(state.pure_role_update_norms, "block"),
        "pure_output_update_norm": _role_stat(state.pure_role_update_norms, "output"),
        "pure_input_update_over_param": _role_stat(state.pure_role_update_over_param, "input"),
        "pure_block_update_over_param_mean": _role_stat(state.pure_role_update_over_param, "block"),
        "pure_output_update_over_param": _role_stat(state.pure_role_update_over_param, "output"),
        "pure_input_raw_grad_norm": _role_stat(state.pure_role_raw_grad_norms, "input"),
        "pure_block_raw_grad_norm_mean": _role_stat(state.pure_role_raw_grad_norms, "block"),
        "pure_output_raw_grad_norm": _role_stat(state.pure_role_raw_grad_norms, "output"),
        "pure_input_precond_norm": _role_stat(state.pure_role_precond_norms, "input"),
        "pure_block_precond_norm_mean": _role_stat(state.pure_role_precond_norms, "block"),
        "pure_output_precond_norm": _role_stat(state.pure_role_precond_norms, "output"),
        "pure_input_cos_raw_precond": _role_stat(state.pure_role_cos_raw_precond, "input", float("nan")),
        "pure_block_cos_raw_precond_mean": _role_stat(state.pure_role_cos_raw_precond, "block", float("nan")),
        "pure_output_cos_raw_precond": _role_stat(state.pure_role_cos_raw_precond, "output", float("nan")),
        "pure_input_metric_condition": _role_stat(state.pure_role_metric_conditions, "input", float("nan")),
        "pure_block_metric_condition_mean": _role_stat(state.pure_role_metric_conditions, "block", float("nan")),
        "pure_output_metric_condition": _role_stat(state.pure_role_metric_conditions, "output", float("nan")),
        "rational_num_update_norm_mean": _mean(state.rational_num_update_norms),
        "rational_den_update_norm_mean": _mean(state.rational_den_update_norms),
        "rational_num_den_update_ratio_mean": _mean(state.rational_num_den_update_ratio),
        "rational_num_den_update_cos_mean": _mean(state.rational_num_den_update_cos, float("nan")),
        "rational_branch_linear_update_norm_mean": _mean(state.rational_branch_linear_update_norms),
        "rational_branch_linear_update_over_param_mean": _mean(state.rational_branch_linear_update_over_param),
        "rational_branch_linear_cov_condition_mean": _mean(
            state.rational_branch_linear_cov_conditions, float("nan")
        ),
        "rational_branch_linear_cos_raw_precond_mean": _mean(
            state.rational_branch_linear_cos_raw_precond, float("nan")
        ),
        "optimizer_kan_raw_grad_norm_mean": uo_raw_mean,
        "optimizer_kan_precond_grad_norm_mean": uo_precond_mean,
        "optimizer_kan_precond_over_raw_norm_mean": uo_precond_ratio_mean,
        "optimizer_rest_grad_norm_mean": uo_rest_grad_mean,
        "optimizer_kan_update_norm_mean": uo_kan_update_mean,
        "optimizer_rest_update_norm_mean": uo_rest_update_mean,
        "optimizer_kan_update_over_param_norm_mean": uo_kan_update_over_param_mean,
        "optimizer_rest_update_over_param_norm_mean": uo_rest_update_over_param_mean,
        "optimizer_kan_update_over_rest_update_mean": uo_update_ratio_mean,
        "optimizer_adam_m_norm_kan_mean": _mean(state.uo_adam_m_norm_kan),
        "optimizer_adam_v_norm_kan_mean": _mean(state.uo_adam_v_norm_kan),
        "optimizer_moment_cos_raw_precond_mean": _mean(state.uo_moment_cos_raw_precond, float("nan")),
        "optimizer_moment_cos_precond_current_mean": _mean(state.uo_moment_cos_precond_current, float("nan")),
        "optimizer_moment_cos_raw_current_mean": _mean(state.uo_moment_cos_raw_current, float("nan")),
        "optimizer_moment_amplification_ratio_mean": _mean(state.uo_moment_amplification_ratio),
        "optimizer_moment_amplification_ratio_p95": _p95(state.uo_moment_amplification_ratio),
        "optimizer_bad_step_count": state.uo_bad_step_count,
        "optimizer_bad_step_rate": uo_bad_step_rate,
        "optimizer_uses_adamw_any_group": int(rest_opt is not None or (opt is not None and not afu_mode)),
        "optimizer_adamw_group_count": (len(rest_opt.param_groups) if rest_opt is not None else 0)
        + (len(opt.param_groups) if opt is not None and not afu_mode else 0),
        "optimizer_functional_group_count": int(bool(coeff_param_values))
        + sum(
            int(
                (group == "kan_bias" and cfg.afu_kan_bias)
                or (group == "head" and cfg.afu_head)
                or (group == "stem" and cfg.afu_stem)
                or (group == "ln" and cfg.afu_ln_scope.lower().replace("-", "_") not in {"", "none"})
            )
            for group in afu_groups
        ),
        "optimizer_nonkan_update_norm_vs_kan_update_norm_mean": nonkan_afu_update / max(1e-12, kan_update_mean_for_ratio),
        "afu_clip_rate": state.afu_clip_count / max(1, state.afu_update_count),
        "afu_head_cov_time_ms": 1000.0 * state.afu_head_cov_time / max(1, step_idx),
        "afu_head_solve_time_ms": 1000.0 * state.afu_head_solve_time / max(1, step_idx),
        "afu_stem_metric_time_ms": 1000.0 * state.afu_stem_metric_time / max(1, step_idx),
        "afu_ln_metric_time_ms": 1000.0 * state.afu_ln_metric_time / max(1, step_idx),
        "label_noise": cfg.label_noise,
        "used_fake_data": int(bundle.used_fake_data),
        "test_acc": test["acc"],
        "test_loss": test["loss"],
        "val_loss": val["loss"],
        "val_auc": val_auc,
        "val_acc": val["acc"],
        "val_acc_auc": val_acc_auc,
        "val_loss_curve": ",".join(f"{x:.6g}" for x in val_losses),
        "val_acc_curve": ",".join(f"{x:.6g}" for x in val_accs),
        "epoch_time_sec_curve": ",".join(f"{x:.6g}" for x in epoch_times),
        "ece": test["ece"],
        "train_acc_noisy_labels": train_noisy["acc"],
        "train_acc_clean_labels": train_clean["acc"],
        "corrupted_subset_train_acc_clean_label": corr_clean,
        "corrupted_subset_train_acc_noisy_label": corr_noisy,
        "noise_memorization_rate": corr_noisy,
        "phi_prime_p95": final_geom["phi_prime_p95"],
        "phi_prime_max": final_geom["phi_prime_max"],
        "max_jac_condition": final_geom["max_jac_condition"],
        "credit_amplification_p95": final_geom["credit_amplification_p95"],
        "curvature_energy": final_geom["curvature_energy"],
        "branch_output_norm_ratio": float(np.mean(branch_ratios)) if branch_ratios else 0.0,
        "branch_output_norm_ratio_final": branch_ratios[-1] if branch_ratios else 0.0,
        "no_kan_drop": test["acc"] - no_kan_eval["acc"],
        "no_kan_test_acc": no_kan_eval["acc"],
        "no_kan_test_loss": no_kan_eval["loss"],
        "no_kan_val_acc": no_kan_val_eval["acc"],
        "no_kan_val_loss": no_kan_val_eval["loss"],
        "no_kan_loss_increase": no_kan_eval["loss"] - test["loss"],
        "kan_logit_delta_norm": contribution["kan_logit_delta_norm_mean"],
        "kan_logit_delta_norm_mean": contribution["kan_logit_delta_norm_mean"],
        "kan_logit_delta_norm_p95": contribution["kan_logit_delta_norm_p95"],
        "kan_correct_class_delta_mean": contribution["kan_correct_class_delta_mean"],
        "kan_margin_contribution_mean": contribution["kan_margin_contribution_mean"],
        "kan_margin_contribution_p95": contribution["kan_margin_contribution_p95"],
        "pure_block_disable_acc": pure_contribution.get("pure_block_disable_acc", float("nan")),
        "pure_block_disable_loss": pure_contribution.get("pure_block_disable_loss", float("nan")),
        "pure_block_disable_drop": pure_contribution.get("pure_block_disable_drop", float("nan")),
        "pure_layerwise_logit_delta_norm": pure_contribution.get("pure_layerwise_logit_delta_norm", float("nan")),
        "pure_layerwise_margin_contribution": pure_contribution.get("pure_layerwise_margin_contribution", float("nan")),
        "pure_layerwise_margin_contribution_p95": pure_contribution.get("pure_layerwise_margin_contribution_p95", float("nan")),
        "trust_clip_rate": state.trust_clip_count / max(1, state.coeff_update_count),
        "data_lambda_mean": float(np.mean(data_lambdas)) if data_lambdas else 0.0,
        "data_lambda_last": state.last_data_lambda,
        "branch_scale_final": state.current_branch_scale,
        "coeff_lr_multiplier_final": state.coeff_lr_multiplier,
        "branch_switch_step": state.branch_switch_step,
        "branch_switch_reason": state.branch_switch_reason,
        "geometry_shrink_events": state.shrink_events,
        "step_time_ms": step_time_ms,
        "epoch_time_sec_mean": epoch_time_sec_mean,
        "epoch_time_sec_p95": epoch_time_sec_p95,
        "total_train_time_sec": wall,
        "samples_per_sec": samples_per_sec,
        "snr_update_time_ms": 1000.0 * state.snr_update_time / max(1, step_idx),
        "snr_overhead_pct": 100.0 * state.snr_update_time / max(1e-9, wall),
        "memory_peak_mb": torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0,
        "memory_adam_state_kan_mb": adam_state_kan_mb,
        "memory_adam_state_rest_mb": adam_state_rest_mb,
        "memory_total_optimizer_state_mb": adam_state_kan_mb + adam_state_rest_mb,
        "rational_denominator_min": rational_safety["rational_denominator_min"],
        "rational_denominator_p01": rational_safety["rational_denominator_p01"],
        "notes": cfg.notes,
    }
    row.update(rational_safety)
    row.update(rbf_occupancy)
    row.update(convstem_audit)
    for group in afu_groups:
        row[f"afu_{group}_update_norm_mean"] = _afu_group_stat(state.afu_update_norms, group)
        row[f"afu_{group}_update_over_param_mean"] = _afu_group_stat(state.afu_update_over_param, group)
        row[f"afu_{group}_update_over_param_p95"] = _afu_group_p95(state.afu_update_over_param, group)
        row[f"afu_{group}_raw_grad_norm_mean"] = _afu_group_stat(state.afu_raw_grad_norms, group)
        row[f"afu_{group}_precond_grad_norm_mean"] = _afu_group_stat(state.afu_precond_grad_norms, group)
        row[f"afu_{group}_cos_raw_precond_mean"] = _afu_group_stat(state.afu_cos_raw_precond, group, float("nan"))
        row[f"afu_{group}_metric_condition_mean"] = _afu_group_stat(state.afu_metric_conditions, group, float("nan"))
        row[f"afu_{group}_metric_min_mean"] = _afu_group_stat(state.afu_metric_mins, group, float("nan"))
        row[f"afu_{group}_metric_max_mean"] = _afu_group_stat(state.afu_metric_maxs, group, float("nan"))
        row[f"afu_{group}_update_share_mean"] = _afu_group_stat(state.afu_update_share, group)
    for idx, alpha in enumerate(alphas):
        row[f"alpha_layer_{idx}_final"] = alpha
        if isinstance(model, PureKANClassifier):
            row[f"pure_alpha_layer_{idx}"] = alpha
    row.update(branch_layers)
    return row


def add_baseline_comparisons(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _cmp_key(row: Dict[str, Any]) -> Tuple[str, int, float, int, str]:
        return (
            str(row.get("dataset", "")),
            int(float(row.get("epochs", -1) or -1)),
            float(row.get("label_noise", 0.0) or 0.0),
            int(float(row.get("train_size", -1) or -1)),
            str(row.get("alpha_mode", "")),
        )

    keys = sorted({_cmp_key(row) for row in rows})
    out = [dict(row) for row in rows]
    for key in keys:
        subset = [row for row in out if _cmp_key(row) == key and row.get("error", "") == ""]
        adam = [
            row
            for row in subset
            if str(row.get("optimizer_method", row.get("method", ""))).lower()
            in {"adamw", "mlp_adamw", "convstem_dgkan_adamw"}
            or str(row.get("method", "")).lower().startswith("adamw")
        ]
        geom = [
            row
            for row in subset
            if str(row.get("optimizer_method", row.get("method", ""))).lower()
            in {"geometry_current", "current geometry-aware", "current_geometry_aware"}
        ]
        if not adam:
            continue
        adam_acc = float(np.mean([float(row["test_acc"]) for row in adam]))
        adam_auc = float(np.mean([float(row["val_auc"]) for row in adam]))
        adam_phi = float(np.mean([float(row["phi_prime_p95"]) for row in adam]))
        adam_j = float(np.mean([float(row["max_jac_condition"]) for row in adam]))
        adam_branch = float(np.mean([float(row["branch_output_norm_ratio"]) for row in adam]))
        adam_no_kan = float(np.mean([float(row.get("no_kan_drop", 0.0)) for row in adam]))
        adam_time = float(np.mean([float(row.get("step_time_ms", float("nan"))) for row in adam]))
        adam_ece = float(np.mean([float(row["ece"]) for row in adam]))
        geom_acc = float(np.mean([float(row["test_acc"]) for row in geom])) if geom else float("nan")
        geom_time = float(np.mean([float(row["step_time_ms"]) for row in geom])) if geom else float("nan")
        geom_ece = float(np.mean([float(row["ece"]) for row in geom])) if geom else float("nan")
        for row in subset:
            row["acc_gap_vs_adamw"] = adam_acc - float(row["test_acc"])
            row["val_auc_improvement_vs_adamw"] = (adam_auc - float(row["val_auc"])) / max(1e-9, adam_auc)
            row["phi_prime_reduction_vs_adamw"] = (adam_phi - float(row["phi_prime_p95"])) / max(1e-9, adam_phi)
            row["jac_reduction_vs_adamw"] = (adam_j - float(row["max_jac_condition"])) / max(1e-9, adam_j)
            row["branch_over_adamw"] = float(row["branch_output_norm_ratio"]) / max(1e-9, adam_branch)
            row["no_kan_drop_over_adamw"] = float(row.get("no_kan_drop", 0.0)) / max(1e-9, adam_no_kan)
            row["gain_vs_geometry"] = float(row["test_acc"]) - geom_acc if not math.isnan(geom_acc) else float("nan")
            row["time_ratio_vs_geometry"] = (
                float(row["step_time_ms"]) / max(1e-9, geom_time) if not math.isnan(geom_time) else float("nan")
            )
            row["time_ratio_vs_adamw"] = float(row["step_time_ms"]) / max(1e-9, adam_time)
            row["ece_reduction_vs_adamw"] = (
                (adam_ece - float(row["ece"])) / max(1e-9, adam_ece)
                if adam
                else float("nan")
            )
            row["ece_reduction_vs_geometry"] = (
                (geom_ece - float(row["ece"])) / max(1e-9, geom_ece) if not math.isnan(geom_ece) else float("nan")
            )
    return out


def summarize_runs(rows: List[Dict[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    compared = add_baseline_comparisons(rows)
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in compared:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)
    metrics = [
        "test_acc",
        "acc_gap_vs_adamw",
        "gain_vs_geometry",
        "val_auc",
        "val_auc_improvement_vs_adamw",
        "phi_prime_reduction_vs_adamw",
        "jac_reduction_vs_adamw",
        "branch_over_adamw",
        "no_kan_drop",
        "ece",
        "ece_reduction_vs_adamw",
        "time_ratio_vs_geometry",
        "time_ratio_vs_adamw",
        "step_time_ms",
        "epoch_time_sec_mean",
        "total_train_time_sec",
        "samples_per_sec",
        "trust_clip_rate",
        "v3_metric_condition_active",
        "v3_metric_condition_geometry",
        "v3_diagfast_condition",
        "v3_fullgeo_condition",
        "update_raw_grad_norm_mean",
        "update_precond_direction_norm_mean",
        "trust_update_metric_norm_mean",
        "trust_update_metric_norm_p95",
        "trust_update_over_coeff_norm_p95",
        "rational_num_update_norm_mean",
        "rational_den_update_norm_mean",
        "rational_num_den_update_ratio_mean",
        "rational_num_den_update_cos_mean",
        "rational_branch_linear_update_norm_mean",
        "rational_branch_linear_update_over_param_mean",
        "rational_branch_linear_cov_condition_mean",
        "rational_branch_linear_cos_raw_precond_mean",
        "data_lambda_mean",
        "max_jac_condition",
        "phi_prime_p95",
        "val_acc",
        "val_acc_auc",
        "branch_output_norm_ratio_final",
        "no_kan_drop_over_adamw",
        "no_kan_loss_increase",
        "kan_logit_delta_norm_mean",
        "kan_logit_delta_norm_p95",
        "kan_margin_contribution_mean",
        "kan_margin_contribution_p95",
        "transition_loss_jump",
        "transition_acc_jump",
        "alpha_final_mean",
        "alpha_final_std",
        "v3_direction_cos_diagfast_fullgeo_mean",
        "v3_direction_norm_ratio_fullgeo_diagfast_mean",
        "v3_direction_norm_ratio_current_diagfast_mean",
        "v3_direction_metric_norm_diagfast_mean",
        "v3_direction_metric_norm_fullgeo_mean",
        "v3_direction_metric_norm_current_mean",
        "optimizer_kan_raw_grad_norm_mean",
        "optimizer_kan_precond_grad_norm_mean",
        "optimizer_kan_precond_over_raw_norm_mean",
        "optimizer_rest_grad_norm_mean",
        "optimizer_kan_update_norm_mean",
        "optimizer_rest_update_norm_mean",
        "optimizer_kan_update_over_param_norm_mean",
        "optimizer_rest_update_over_param_norm_mean",
        "optimizer_kan_update_over_rest_update_mean",
        "optimizer_adam_m_norm_kan_mean",
        "optimizer_adam_v_norm_kan_mean",
        "optimizer_moment_cos_raw_precond_mean",
        "optimizer_moment_cos_precond_current_mean",
        "optimizer_moment_cos_raw_current_mean",
        "optimizer_moment_amplification_ratio_mean",
        "optimizer_moment_amplification_ratio_p95",
        "optimizer_bad_step_count",
        "optimizer_bad_step_rate",
        "memory_adam_state_kan_mb",
        "memory_adam_state_rest_mb",
        "memory_total_optimizer_state_mb",
        "coeff_param_numel_total",
        "coeff_param_seen_ratio",
        "active_steps_actual",
        "transition_steps_actual",
        "geometry_steps_actual",
        "pure_alpha_mean",
        "pure_alpha_trainable",
        "pure_input_update_norm",
        "pure_block_update_norm_mean",
        "pure_output_update_norm",
        "pure_input_update_over_param",
        "pure_block_update_over_param_mean",
        "pure_output_update_over_param",
        "pure_input_raw_grad_norm",
        "pure_block_raw_grad_norm_mean",
        "pure_output_raw_grad_norm",
        "pure_input_precond_norm",
        "pure_block_precond_norm_mean",
        "pure_output_precond_norm",
        "pure_input_cos_raw_precond",
        "pure_block_cos_raw_precond_mean",
        "pure_output_cos_raw_precond",
        "pure_input_metric_condition",
        "pure_block_metric_condition_mean",
        "pure_output_metric_condition",
        "pure_block_disable_acc",
        "pure_block_disable_loss",
        "pure_block_disable_drop",
        "pure_layerwise_logit_delta_norm",
        "pure_layerwise_margin_contribution",
        "rational_denominator_min",
        "rational_denominator_p01",
        "rational_den_actual_min_batch",
        "rational_den_actual_p01_batch",
        "rational_den_actual_condition_batch",
        "rational_den_actual_min_grid",
        "rational_den_actual_p01_grid",
        "rational_r_prime_p95",
        "rational_r_double_prime_p95",
        "rbf_basis_mean",
        "rbf_basis_min",
        "rbf_basis_p01",
        "rbf_basis_dead_frac",
        "rbf_input_out_of_grid_frac",
        "convstem_output_mean",
        "convstem_output_std",
        "convstem_output_abs_p95",
        "optimizer_uses_adamw_any_group",
        "optimizer_adamw_group_count",
        "optimizer_functional_group_count",
        "optimizer_nonkan_update_norm_vs_kan_update_norm_mean",
        "afu_clip_rate",
        "afu_head_cov_time_ms",
        "afu_head_solve_time_ms",
        "afu_stem_metric_time_ms",
        "afu_ln_metric_time_ms",
    ]
    for group in ["kan_bias", "head", "stem", "ln"]:
        metrics.extend(
            [
                f"afu_{group}_update_norm_mean",
                f"afu_{group}_update_over_param_mean",
                f"afu_{group}_update_over_param_p95",
                f"afu_{group}_raw_grad_norm_mean",
                f"afu_{group}_precond_grad_norm_mean",
                f"afu_{group}_cos_raw_precond_mean",
                f"afu_{group}_metric_condition_mean",
                f"afu_{group}_metric_min_mean",
                f"afu_{group}_metric_max_mean",
                f"afu_{group}_update_share_mean",
            ]
        )
    summary: List[Dict[str, Any]] = []
    for key, group in groups.items():
        row: Dict[str, Any] = {k: v for k, v in zip(group_keys, key)}
        row["runs"] = len(group)
        for metric in metrics:
            vals = []
            for item in group:
                try:
                    value = float(item.get(metric, float("nan")))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    vals.append(value)
            if vals:
                mean, std = mean_std(vals)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
        summary.append(row)
    return sorted(summary, key=lambda r: tuple(str(r.get(k, "")) for k in group_keys))


def classify_failure(row: Dict[str, Any]) -> str:
    try:
        gap = float(row.get("acc_gap_vs_adamw", 0.0))
        auc = float(row.get("val_auc_improvement_vs_adamw", 0.0))
        phi_red = float(row.get("phi_prime_reduction_vs_adamw", 0.0))
        jac_red = float(row.get("jac_reduction_vs_adamw", 0.0))
        branch = float(row.get("branch_over_adamw", 1.0))
        clip = float(row.get("trust_clip_rate", 0.0))
        time_ratio = float(row.get("time_ratio_vs_geometry", 1.0))
        moment_amp = float(row.get("optimizer_moment_amplification_ratio_mean", 0.0))
        moment_cos = float(row.get("optimizer_moment_cos_precond_current_mean", 1.0))
        bad_rate = float(row.get("optimizer_bad_step_rate", 0.0))
    except (TypeError, ValueError):
        return ""
    if moment_amp > 3.0:
        return "moment_explosion"
    if moment_cos < 0.5:
        return "moment_direction_drift"
    if bad_rate > 0.10:
        return "bad_step_rate_high"
    if branch < 0.5 and gap > 0.01:
        return "branch_underactive"
    if branch > 1.2 or phi_red < -0.05:
        return "branch_overactive"
    if gap < 0.005 and (jac_red < 0.05 or phi_red < 0.10):
        return "data_metric_geometry_failure"
    if clip > 0.2 and branch < 0.7:
        return "trust_too_conservative"
    if time_ratio > 1.5 and "snr" in str(row.get("method", "")).lower():
        return "snr_cost_failure"
    if gap > 0.01:
        return "accuracy_gap_too_large"
    if auc < 0:
        return "no_convergence_gain"
    return ""


def add_failure_types(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compared = add_baseline_comparisons(rows)
    for row in compared:
        row["failure_type"] = classify_failure(row)
    return compared


def add_common_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--allow-fake-data", action="store_true")
    p.add_argument("--train-size", type=int, default=6000)
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--test-size", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--audit-batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--basis-count", type=int, default=16)
    p.add_argument("--alpha-init", type=float, default=1.5)
    p.add_argument("--alpha-mode", choices=["learnable", "fixed1", "fixed_init"], default="learnable")
    p.add_argument("--model-type", default="rbf_dgkan")
    p.add_argument("--rational-backend", default="triton")
    p.add_argument("--rational-groups", type=int, default=8)
    p.add_argument("--rational-mode", default="swish")
    p.add_argument("--coeff-lr", type=float, default=0.05)
    p.add_argument("--rest-lr", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-schedule", default="none")
    p.add_argument("--lr-final-mult", type=float, default=1.0)
    p.add_argument("--rest-lr-schedule", default="none")
    p.add_argument("--rest-lr-final-mult", type=float, default=1.0)
    p.add_argument("--coeff-lr-schedule", default="none")
    p.add_argument("--coeff-lr-decay-final-mult", type=float, default=1.0)
    p.add_argument("--lr-decay-start-frac", type=float, default=0.0)
    p.add_argument("--v3-gram-grid-size", type=int, default=512)
    p.add_argument("--v3-transition-frac", type=float, default=0.10)
    p.add_argument("--unified-optimizer-mode", default="hybrid")
    p.add_argument("--kan-grad-transform", default="none")
    p.add_argument("--kan-precondition-before-adam", action="store_true")
    p.add_argument("--kan-precondition-after-adam", action="store_true")
    p.add_argument("--kan-precond-grad-normalize", action="store_true")
    p.add_argument("--uo-trust-radius", type=float, default=0.0)
    p.add_argument("--uo-record-moment-audit", action="store_true")
    p.add_argument("--afu-kan-bias", action="store_true")
    p.add_argument("--afu-head", action="store_true")
    p.add_argument("--afu-stem", action="store_true")
    p.add_argument("--afu-ln-scope", default="none")
    p.add_argument("--afu-head-lr-mult", type=float, default=0.5)
    p.add_argument("--afu-stem-lr-mult", type=float, default=0.3)
    p.add_argument("--afu-ln-lr-mult", type=float, default=0.3)
    p.add_argument("--afu-bias-lr-mult", type=float, default=0.5)
    p.add_argument("--afu-cov-ema-beta", type=float, default=0.95)
    p.add_argument("--afu-head-rho", type=float, default=1e-2)
    p.add_argument("--afu-stem-rho", type=float, default=1e-2)
    p.add_argument("--afu-ln-rho", type=float, default=1e-2)
    p.add_argument("--afu-bias-rho", type=float, default=1e-2)
    p.add_argument("--afu-max-update-ratio", type=float, default=0.05)
    p.add_argument("--pure-input-metric", default="phase")
    p.add_argument("--pure-block-metric", default="phase")
    p.add_argument("--pure-output-metric", default="phase")
    p.add_argument("--pure-input-lr-mult", type=float, default=1.0)
    p.add_argument("--pure-block-lr-mult", type=float, default=1.0)
    p.add_argument("--pure-output-lr-mult", type=float, default=1.0)
    p.add_argument("--pure-input-trust-radius", type=float, default=0.0)
    p.add_argument("--pure-block-trust-radius", type=float, default=0.0)
    p.add_argument("--pure-output-trust-radius", type=float, default=0.0)
    p.add_argument("--continue-on-error", action="store_true")


def config_from_args(args: argparse.Namespace, dataset: str, method: str, seed: int, **overrides: Any) -> TrainConfig:
    base_kwargs = {
        "device": args.device,
        "data_root": args.data_root,
        "download": not args.no_download,
        "allow_fake_data": args.allow_fake_data,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "audit_batch_size": args.audit_batch_size,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "basis_count": args.basis_count,
        "alpha_init": args.alpha_init,
        "alpha_mode": args.alpha_mode,
        "model_type": args.model_type,
        "rational_backend": args.rational_backend,
        "rational_groups": args.rational_groups,
        "rational_mode": args.rational_mode,
        "coeff_lr": args.coeff_lr,
        "rest_lr": args.rest_lr,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "lr_final_mult": args.lr_final_mult,
        "rest_lr_schedule": args.rest_lr_schedule,
        "rest_lr_final_mult": args.rest_lr_final_mult,
        "coeff_lr_schedule": args.coeff_lr_schedule,
        "coeff_lr_decay_final_mult": args.coeff_lr_decay_final_mult,
        "lr_decay_start_frac": args.lr_decay_start_frac,
        "v3_gram_grid_size": args.v3_gram_grid_size,
        "v3_transition_frac": args.v3_transition_frac,
        "unified_optimizer_mode": args.unified_optimizer_mode,
        "kan_grad_transform": args.kan_grad_transform,
        "kan_precondition_before_adam": args.kan_precondition_before_adam,
        "kan_precondition_after_adam": args.kan_precondition_after_adam,
        "kan_precond_grad_normalize": args.kan_precond_grad_normalize,
        "uo_trust_radius": args.uo_trust_radius,
        "uo_record_moment_audit": args.uo_record_moment_audit,
        "afu_kan_bias": args.afu_kan_bias,
        "afu_head": args.afu_head,
        "afu_stem": args.afu_stem,
        "afu_ln_scope": args.afu_ln_scope,
        "afu_head_lr_mult": args.afu_head_lr_mult,
        "afu_stem_lr_mult": args.afu_stem_lr_mult,
        "afu_ln_lr_mult": args.afu_ln_lr_mult,
        "afu_bias_lr_mult": args.afu_bias_lr_mult,
        "afu_cov_ema_beta": args.afu_cov_ema_beta,
        "afu_head_rho": args.afu_head_rho,
        "afu_stem_rho": args.afu_stem_rho,
        "afu_ln_rho": args.afu_ln_rho,
        "afu_bias_rho": args.afu_bias_rho,
        "afu_max_update_ratio": args.afu_max_update_ratio,
        "pure_input_metric": args.pure_input_metric,
        "pure_block_metric": args.pure_block_metric,
        "pure_output_metric": args.pure_output_metric,
        "pure_input_lr_mult": args.pure_input_lr_mult,
        "pure_block_lr_mult": args.pure_block_lr_mult,
        "pure_output_lr_mult": args.pure_output_lr_mult,
        "pure_input_trust_radius": args.pure_input_trust_radius,
        "pure_block_trust_radius": args.pure_block_trust_radius,
        "pure_output_trust_radius": args.pure_output_trust_radius,
    }
    base_kwargs.update(overrides)
    cfg = build_train_config_for_method(
        method,
        dataset=dataset,
        seed=seed,
        **base_kwargs,
    )
    cfg.optimizer_method = method
    return cfg
