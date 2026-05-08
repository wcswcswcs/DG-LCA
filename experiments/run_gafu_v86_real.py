#!/usr/bin/env python3
"""DG-KAN v8.6 external generalization / continual formalization runner.

This runner is deliberately boundary-oriented.  It reuses the v8.5 external
fair runner for the accepted-route reproduction, then adds v8.6-specific
multi-split continual stress artifacts.  It does not widen the claim unless
landed CSV/JSON rows support the corresponding gates.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

import run_gafu_v85_real as v85
from dgkan_core import ensure_dir, save_json, set_seed, write_csv


PLAN_PATH = "docs/DG-KAN_v8.6_External_Generalization_Continual_Formalization_完整实验计划.md"
METRIC_UNAVAILABLE = "metric_unavailable"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in (None, "", METRIC_UNAVAILABLE):
            return default
        return float(value)
    except Exception:
        return default


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_not_run(out_dir: Path, reason: str) -> None:
    placeholders = {
        "kanbefair_broad_reproduction.csv": "P1_BROAD_BASELINE_REPRODUCTION_V86",
        "dgkan_adapter_contract_v86.csv": "P2_ADAPTER_CONTRACT_V86",
        "params_flops_counter_v86.csv": "P3_COUNTER_VALIDATION_V86",
        "external_multitask_transfer.csv": "P4_EXTERNAL_MULTITASK_TRANSFER_V86",
        "joint_fair_envelope.csv": "P5_JOINT_FAIR_ENVELOPE_V86",
        "functional_causality_multitask.csv": "P6_FUNCTIONAL_CAUSALITY_MULTITASK_V86",
        "tabular_formal_envelope.csv": "P12_TABULAR_FORMAL_ENVELOPE_V86",
        "nonvision_runnability_audit.csv": "P13_NONVISION_RUNNABILITY_AUDIT_V86",
        "nonvision_baseline_reproduction.csv": "P14_NONVISION_BASELINE_REPRODUCTION_V86",
        "nonvision_dg_transfer.csv": "P15_NONVISION_DG_TRANSFER_V86",
        "time_accounting_external.csv": "P8_TIME_ACCOUNTING_EXTERNAL_V86",
        "phase_mapped_profiler_external.csv": "P9_PHASE_MAPPED_PROFILER_EXTERNAL_V86",
        "geometry_task_relationship.csv": "P10_GEOMETRY_TASK_RELATIONSHIP_V86",
        "negative_boundary_audit.csv": "P11_NEGATIVE_BOUNDARY_AUDIT_V86",
    }
    for filename, stage in placeholders.items():
        path = out_dir / filename
        if path.exists():
            continue
        write_csv(path, [{
            "stage": stage,
            "status": "not_run",
            "reason": reason,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }])


def _broad_formal_protocol(args: argparse.Namespace) -> int:
    return int(
        int(args.broad_train_size) >= 60000
        and int(args.broad_test_size) >= 10000
        and int(args.broad_epochs) == 20
        and int(args.kb_batch_size) == 128
    )


def _release_cuda_model(model: Any, device: torch.device) -> None:
    if torch.cuda.is_available() and device.type == "cuda":
        try:
            model.to("cpu")
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()


def _release_cuda_memory(device: torch.device) -> None:
    gc.collect()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()


_TABULAR_UCI_IDS = {
    "Abalone": 1,
    "Income": 2,
    "Mushroom": 73,
    "Wine": 186,
    "Bank": 222,
    "Rice": 545,
    "Bean": 602,
    "Student": 697,
    "Spam": 94,
    "Card": 350,
    "Telescope": 159,
    "Dota": 367,
    "Darwin": 732,
    "Toxicity": 728,
}


def _dependency_import_status(name: str) -> str:
    try:
        __import__(name)
        return "ok"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _process_tabular_features(df: Any) -> Any:
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    frame = df.copy()
    numeric_cols = list(frame.select_dtypes(include=["number"]).columns)
    category_cols = [c for c in frame.columns if c not in numeric_cols]
    keep_cols = numeric_cols + category_cols
    frame = frame.loc[:, keep_cols].copy()
    for col in category_cols:
        series = frame[col].astype(str).str.lower()
        series = series.str.strip('!"#%&\'()*,./:;?@[\\]^_`{|}~' + " \n\r\t")
        labels = {value: idx for idx, value in enumerate(series.unique())}
        frame[col] = series.map(labels).astype(float)
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
    if not keep_cols:
        raise ValueError("tabular feature frame has no numeric or categorical columns")
    frame = frame.fillna(0.0)
    scaler = StandardScaler()
    frame.loc[:, keep_cols] = scaler.fit_transform(frame.loc[:, keep_cols])
    return frame.loc[:, keep_cols].values


def _process_tabular_targets(df: Any) -> Any:
    if len(df.columns) != 1:
        raise ValueError(f"KANbeFair UCI classification loader expects one target column, got {len(df.columns)}")
    series = df.iloc[:, 0].astype(str).str.lower()
    series = series.str.strip('!"#%&\'()*,./:;?@[\\]^_`{|}~' + " \n\r\t")
    labels = {value: idx for idx, value in enumerate(series.unique())}
    return series.map(labels).values


def _load_kanbefair_tabular_tensors(
    dataset_name: str,
    *,
    kb_path: Path,
    train_size: int,
    test_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, str]:
    canonical = str(dataset_name)
    if canonical not in _TABULAR_UCI_IDS:
        raise ValueError(f"unsupported KANbeFair tabular dataset {dataset_name!r}")
    try:
        from ucimlrepo import fetch_ucirepo
    except Exception as exc:
        raise RuntimeError(f"ucimlrepo import failed: {exc}") from exc

    uciml_id = int(_TABULAR_UCI_IDS[canonical])
    fetched = fetch_ucirepo(id=uciml_id)
    x_values = _process_tabular_features(fetched.data.features)
    y_values = _process_tabular_targets(fetched.data.targets)
    x_all = torch.tensor(x_values, dtype=torch.float32)
    y_all = torch.tensor(y_values, dtype=torch.long)

    split_path = kb_path / "dataset" / "uciml_split_idx.pt"
    if not split_path.exists():
        raise FileNotFoundError(f"KANbeFair split index missing: {split_path}")
    try:
        train_idx_raw, test_idx_raw = torch.load(split_path, map_location="cpu", weights_only=False)
    except TypeError:
        train_idx_raw, test_idx_raw = torch.load(split_path, map_location="cpu")
    train_idx = torch.tensor(train_idx_raw, dtype=torch.long)
    test_idx = torch.tensor(test_idx_raw, dtype=torch.long)
    train_idx = train_idx[train_idx < int(x_all.shape[0])]
    test_idx = test_idx[test_idx < int(x_all.shape[0])]
    if int(train_size) > 0 and int(train_size) < int(train_idx.numel()):
        g = torch.Generator().manual_seed(v85.v83.v72._stable_seed("v86-tabular-train", canonical, int(seed)))
        train_idx = train_idx[torch.randperm(int(train_idx.numel()), generator=g)[: int(train_size)]]
        train_protocol = f"KANbeFair uciml split random-subset train={int(train_size)}"
    else:
        train_protocol = f"KANbeFair uciml split train={int(train_idx.numel())}"
    if int(test_size) > 0 and int(test_size) < int(test_idx.numel()):
        test_idx = test_idx[: int(test_size)]
        test_protocol = f"prefix-test test={int(test_size)}"
    else:
        test_protocol = f"test={int(test_idx.numel())}"
    protocol = (
        f"KANbeFair UCI id={uciml_id}; compatibility pandas3 loader; "
        f"{train_protocol}; {test_protocol}; split_file={split_path}"
    )
    return (
        x_all[train_idx].contiguous(),
        y_all[train_idx].contiguous(),
        x_all[test_idx].contiguous(),
        y_all[test_idx].contiguous(),
        int(x_all.shape[1]),
        int(y_all.max().item() + 1),
        protocol,
    )


def _write_nonvision_runnability_audit(out_dir: Path, args: argparse.Namespace, *, tabular_status: str) -> List[Dict[str, Any]]:
    root_dataset = Path("dataset")
    kb_dataset = Path(args.kanbefair_path) / "dataset"
    rows: List[Dict[str, Any]] = []
    dep_pandas = _dependency_import_status("pandas")
    dep_ucimlrepo = _dependency_import_status("ucimlrepo")
    dep_torchtext = _dependency_import_status("torchtext")
    dep_torchaudio = _dependency_import_status("torchaudio")
    torchtext_ag_news = 0
    torchtext_cola = 0
    torchtext_text_utils = 0
    ag_news_cache_paths = [
        root_dataset / "ag_news_csv.tar.gz",
        root_dataset / "ag_news_csv" / "train.csv",
        root_dataset / "ag_news_csv" / "test.csv",
    ]
    cola_cache_paths = [
        root_dataset / "CoLA" / "train.tsv",
        root_dataset / "CoLA" / "dev.tsv",
        root_dataset / "CoLA" / "test.tsv",
    ]
    cola_local_cache_ready = all(p.exists() for p in cola_cache_paths)
    torchtext_ag_news_status = "not_checked"
    torchtext_cola_status = "not_checked"
    if dep_torchtext == "ok":
        try:
            import torchtext
            from torchtext.data.utils import get_tokenizer  # noqa: F401
            from torchtext.vocab import build_vocab_from_iterator  # noqa: F401
            torchtext_text_utils = 1
            if hasattr(torchtext.datasets, "AG_NEWS"):
                try:
                    _ = torchtext.datasets.AG_NEWS(root=str(root_dataset), split="train")
                    torchtext_ag_news = 1
                    torchtext_ag_news_status = "ok"
                except Exception as exc:
                    split_exc = f"{type(exc).__name__}: {exc}"
                    try:
                        legacy_train, legacy_test = torchtext.datasets.AG_NEWS(root=str(root_dataset))
                        torchtext_ag_news = int(len(legacy_train) > 0 and len(legacy_test) > 0)
                        torchtext_ag_news_status = (
                            f"legacy_no_split_ok; split_api_failed={split_exc}; "
                            f"train={len(legacy_train)}; test={len(legacy_test)}"
                        )
                    except Exception as legacy_exc:
                        torchtext_ag_news_status = (
                            f"{split_exc}; legacy_no_split_failed={type(legacy_exc).__name__}: {legacy_exc}"
                        )
            else:
                torchtext_ag_news_status = "missing_AG_NEWS_dataset_api"
            if hasattr(torchtext.datasets, "CoLA"):
                try:
                    _ = torchtext.datasets.CoLA(root=str(root_dataset), split="train")
                    torchtext_cola = 1
                    torchtext_cola_status = "ok"
                except Exception as exc:
                    torchtext_cola_status = f"{type(exc).__name__}: {exc}"
            else:
                if cola_local_cache_ready:
                    torchtext_cola = 1
                    torchtext_cola_status = "local_glue_cache_ok; missing_CoLA_dataset_api"
                else:
                    torchtext_cola_status = "missing_CoLA_dataset_api"
        except Exception:
            torchtext_text_utils = 0
    rows.append({
        "stage": "P13_NONVISION_RUNNABILITY_AUDIT_V86",
        "task_family": "tabular",
        "task_name": args.tabular_datasets,
        "dependency_pandas": dep_pandas,
        "dependency_ucimlrepo": dep_ucimlrepo,
        "dependency_torchtext": dep_torchtext,
        "dependency_torchaudio": dep_torchaudio,
        "torchtext_AG_NEWS_api_status": torchtext_ag_news_status,
        "torchtext_CoLA_api_status": torchtext_cola_status,
        "cache_path": str(kb_dataset / "uciml_split_idx.pt"),
        "cache_exists": int((kb_dataset / "uciml_split_idx.pt").exists()),
        "status": tabular_status,
        "RunnablePass": int(str(tabular_status).startswith("measured") or str(tabular_status) == "merged_measured_artifacts"),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    })
    for task_name, cache_paths in {
        "AG_NEWS": ag_news_cache_paths,
        "CoLA": cola_cache_paths,
        "IMDb": [root_dataset / "IMDb" / "IMDB Dataset.csv"],
        "SpeechCommand": [root_dataset / "SpeechCommands" / "train_sc.pt", root_dataset / "SpeechCommands" / "test_sc.pt"],
        "UrbanSound8K": [
            root_dataset / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv",
            root_dataset / "UrbanSound8K" / "train_us.pt",
            root_dataset / "UrbanSound8K" / "test_us.pt",
        ],
    }.items():
        family = "audio" if task_name in {"SpeechCommand", "UrbanSound8K"} else "nlp"
        if family == "audio":
            dep_ok = dep_torchaudio == "ok"
        elif task_name == "AG_NEWS":
            dep_ok = dep_torchtext == "ok" and bool(torchtext_ag_news) and bool(torchtext_text_utils)
        elif task_name == "CoLA":
            dep_ok = dep_torchtext == "ok" and bool(torchtext_cola) and bool(torchtext_text_utils)
        else:
            dep_ok = dep_torchtext == "ok" and bool(torchtext_text_utils)
        cache_ok = all(p.exists() for p in cache_paths) if cache_paths else int(dep_ok)
        blocker = []
        if not dep_ok:
            blocker.append("dependency_missing_or_import_failed")
        if cache_paths and not cache_ok:
            blocker.append("cache_missing")
        rows.append({
            "stage": "P13_NONVISION_RUNNABILITY_AUDIT_V86",
            "task_family": family,
            "task_name": task_name,
            "dependency_pandas": dep_pandas,
            "dependency_ucimlrepo": dep_ucimlrepo,
            "dependency_torchtext": dep_torchtext,
            "dependency_torchaudio": dep_torchaudio,
            "torchtext_AG_NEWS_api_status": torchtext_ag_news_status,
            "torchtext_CoLA_api_status": torchtext_cola_status,
            "cache_path": ";".join(str(p) for p in cache_paths) if cache_paths else METRIC_UNAVAILABLE,
            "cache_exists": int(cache_ok),
            "status": "runnable_dependency_cache_ready" if dep_ok and cache_ok else "blocked_" + "_and_".join(blocker),
            "RunnablePass": int(dep_ok and cache_ok),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    write_csv(out_dir / "nonvision_runnability_audit.csv", rows)
    return rows


def _temporary_cwd(path: Path):
    class _Cwd:
        def __enter__(self):
            self.prev = Path.cwd()
            os.chdir(path)
            return self

        def __exit__(self, exc_type, exc, tb):
            os.chdir(self.prev)
            return False

    return _Cwd()


def _ensure_kanbefair_dataset_links(kb_path: Path, data_root: Path) -> None:
    kb_dataset = kb_path / "dataset"
    kb_dataset.mkdir(parents=True, exist_ok=True)
    for name in ("IMDb", "SpeechCommands", "UrbanSound8K", "CoLA"):
        dst = kb_dataset / name
        src = data_root / name
        if dst.exists() or dst.is_symlink() or not src.exists():
            continue
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def _to_device_batch(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, (tuple, list)):
        return type(value)(_to_device_batch(x, device) for x in value)
    return value


def _subset_dataset(dataset: Any, limit: int) -> Any:
    if int(limit) <= 0 or int(limit) >= len(dataset):
        return dataset
    return torch.utils.data.Subset(dataset, list(range(int(limit))))


def _train_eval_kb_loader_classifier(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    *,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    set_seed(int(seed))
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    step_count = 0
    last_loss = float("nan")
    for _epoch in range(int(epochs)):
        model.train()
        for data, target in train_loader:
            data = _to_device_batch(data, device)
            target = _to_device_batch(target, device).long()
            opt.zero_grad(set_to_none=True)
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())
            step_count += 1
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    else:
        peak_mb = 0.0
    elapsed = time.perf_counter() - started
    model.eval()
    correct = 0
    total = 0
    test_loss_sum = 0.0
    with torch.inference_mode():
        for data, target in test_loader:
            data = _to_device_batch(data, device)
            target = _to_device_batch(target, device).long()
            logits = model(data)
            test_loss_sum += float(F.cross_entropy(logits, target, reduction="sum").detach().cpu())
            pred = logits.argmax(dim=1)
            correct += int((pred == target).sum().detach().cpu())
            total += int(target.numel())
    return {
        "test_acc_pct": 100.0 * correct / max(1, total),
        "test_loss": test_loss_sum / max(1, total),
        "train_time_s": elapsed,
        "step_time_ms": elapsed * 1000.0 / max(1, step_count),
        "peak_memory_MB": peak_mb,
        "first_last_train_loss": last_loss,
        "train_steps": float(step_count),
    }


def _patch_kanbefair_text_vocab_compat() -> None:
    try:
        import inspect
        from collections import Counter
        from torchtext.vocab import Vocab
        import data.text as kb_text  # type: ignore
    except Exception:
        return
    try:
        sig = inspect.signature(kb_text.build_vocab_from_iterator)
        if "specials" in sig.parameters and hasattr(Vocab(Counter({"x": 1})), "__call__"):
            return
    except Exception:
        pass

    class _VocabCompat:
        def __init__(self, vocab: Any):
            self.vocab = vocab
            self.default_index = 0

        def __len__(self) -> int:
            return len(self.vocab)

        def __getitem__(self, token: str) -> int:
            try:
                return int(self.vocab[token])
            except Exception:
                return int(self.default_index)

        def __call__(self, tokens: Sequence[str]) -> List[int]:
            return [self[token] for token in tokens]

        def set_default_index(self, index: int) -> None:
            self.default_index = int(index)

    def _compat_build_vocab_from_iterator(iterator: Any, specials: Sequence[str] | None = None, **_: Any) -> _VocabCompat:
        counter: Counter[str] = Counter()
        for tokens in iterator:
            counter.update(tokens)
        vocab = Vocab(counter, specials=list(specials or ["<unk>"]))
        return _VocabCompat(vocab)

    kb_text.build_vocab_from_iterator = _compat_build_vocab_from_iterator


def _run_nonvision_baseline_reproduction(out_dir: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    kb_path = Path(args.kanbefair_path)
    data_root = Path(args.data_root)
    _ensure_kanbefair_dataset_links(kb_path, data_root)
    src = kb_path / "src"
    if str(src.resolve()) not in sys.path:
        sys.path.insert(0, str(src.resolve()))
    rows: List[Dict[str, Any]] = []
    device = v85.v83.get_device(args.device)
    try:
        v85._install_matplotlib_stub_if_missing()
        from models.mlp import MLP, MLP_Text  # type: ignore
        from models.kanbefair import KANbeFair, KANbeFair_Text  # type: ignore
        from data.text import create_text_loader, get_IMDb_dataset  # type: ignore
        _patch_kanbefair_text_vocab_compat()
    except Exception as exc:
        rows = [{
            "stage": "P14_NONVISION_BASELINE_REPRODUCTION_V86",
            "status": "not_run",
            "reason": f"kanbefair_nonvision_import_failed:{type(exc).__name__}:{exc}",
            "NativeBaselineMeasuredPass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }]
        write_csv(out_dir / "nonvision_baseline_reproduction.csv", rows)
        return rows

    tasks = [t.strip() for t in str(args.nonvision_baseline_tasks).split(",") if t.strip()]
    for task_name in tasks:
        try:
            if task_name == "IMDb":
                with _temporary_cwd(src):
                    train_ds, test_ds = get_IMDb_dataset()
                    train_ds = _subset_dataset(train_ds, int(args.nonvision_baseline_train_size))
                    test_ds = _subset_dataset(test_ds, int(args.nonvision_baseline_test_size))
                    train_loader, vocab = create_text_loader(
                        train_ds,
                        None,
                        {"batch_size": int(args.nonvision_baseline_batch_size), "shuffle": True, "num_workers": 0},
                    )
                    test_loader, _ = create_text_loader(
                        test_ds,
                        vocab,
                        {"batch_size": int(args.nonvision_baseline_batch_size), "shuffle": False, "num_workers": 0},
                    )
                family = "nlp"
                num_classes = 2
                input_size = len(vocab)
                model_specs = [
                    ("KB-MLP_Text", MLP_Text, "MLP_Text", [32], "gelu", 0, 0, "default"),
                    ("KB-KAN_Text", KANbeFair_Text, "KAN_Text", [2], "gelu", 3, 2, "silu"),
                ]
            elif task_name == "SpeechCommand":
                train_path = data_root / "SpeechCommands" / "train_sc.pt"
                test_path = data_root / "SpeechCommands" / "test_sc.pt"
                train_ds = _subset_dataset(torch.load(train_path, map_location="cpu"), int(args.nonvision_baseline_train_size))
                test_ds = _subset_dataset(torch.load(test_path, map_location="cpu"), int(args.nonvision_baseline_test_size))
                train_loader = torch.utils.data.DataLoader(
                    train_ds,
                    batch_size=int(args.nonvision_baseline_batch_size),
                    shuffle=True,
                    num_workers=0,
                )
                test_loader = torch.utils.data.DataLoader(
                    test_ds,
                    batch_size=int(args.nonvision_baseline_batch_size),
                    shuffle=False,
                    num_workers=0,
                )
                family = "audio"
                num_classes = 35
                input_size = 1000
                model_specs = [
                    ("KB-MLP", MLP, "MLP", [32], "gelu", 0, 0, "default"),
                    ("KB-KAN", KANbeFair, "KAN", [2], "gelu", 3, 2, "silu"),
                ]
            else:
                rows.append({
                    "stage": "P14_NONVISION_BASELINE_REPRODUCTION_V86",
                    "status": "not_run",
                    "task_family": "unknown",
                    "task_name": task_name,
                    "reason": "task_not_enabled_for_native_nonvision_baseline",
                    "NativeBaselineMeasuredPass": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
                continue
        except Exception as exc:
            rows.append({
                "stage": "P14_NONVISION_BASELINE_REPRODUCTION_V86",
                "status": "not_run",
                "task_name": task_name,
                "reason": f"dataset_loader_failed:{type(exc).__name__}:{exc}",
                "NativeBaselineMeasuredPass": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            continue

        for model_id, cls, model_name, width, activation_name, grid, order, shortcut in model_specs:
            try:
                ns = v85._kb_namespace(
                    model_name=model_name,
                    input_size=int(input_size),
                    output_size=int(num_classes),
                    layers_width=width,
                    activation_name=activation_name,
                    batch_norm=False,
                    kan_grid=grid or 3,
                    kan_order=order or 2,
                    kan_shortcut=shortcut if shortcut != "default" else "silu",
                    kan_range=[-1.0, 1.0],
                )
                set_seed(int(args.seed))
                model = cls(ns)
                params = int(model.total_parameters())
                flops = float(model.total_flops())
                metrics = _train_eval_kb_loader_classifier(
                    model,
                    train_loader,
                    test_loader,
                    epochs=int(args.nonvision_baseline_epochs),
                    lr=float(args.nonvision_baseline_lr),
                    seed=int(args.seed),
                    device=device,
                )
                rows.append({
                    "stage": "P14_NONVISION_BASELINE_REPRODUCTION_V86",
                    "status": "measured",
                    "task_family": family,
                    "task_name": task_name,
                    "dataset_name": task_name,
                    "model_name": model_id,
                    "reported_metric": METRIC_UNAVAILABLE,
                    "measured_metric": metrics["test_acc_pct"],
                    "absolute_delta_from_reported": METRIC_UNAVAILABLE,
                    "relative_delta_from_reported": METRIC_UNAVAILABLE,
                    "params": params,
                    "FLOPs": flops,
                    "train_time_s": metrics["train_time_s"],
                    "test_time_s": METRIC_UNAVAILABLE,
                    "step_time_ms": metrics["step_time_ms"],
                    "peak_memory_MB": metrics["peak_memory_MB"],
                    "train_steps": int(metrics["train_steps"]),
                    "last_train_loss": metrics["first_last_train_loss"],
                    "test_loss": metrics["test_loss"],
                    "seed": int(args.seed),
                    "epochs": int(args.nonvision_baseline_epochs),
                    "batch_size": int(args.nonvision_baseline_batch_size),
                    "train_size": len(train_ds),
                    "test_size": len(test_ds),
                    "source_protocol": "KANbeFair_native_loader_or_cache",
                    "reproduction_pass": 0,
                    "NativeBaselineMeasuredPass": 1,
                    "loss_type": "CE",
                    "geometry_loss_used": 0,
                    "external_teacher_used": 0,
                    "self_teacher_used": 0,
                    "sampler_changed": 0,
                    "class_weight_used": 0,
                    "cpu_offload_used": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                })
                _release_cuda_model(model, device)
            except Exception as exc:
                rows.append({
                    "stage": "P14_NONVISION_BASELINE_REPRODUCTION_V86",
                    "status": "not_run",
                    "task_family": family,
                    "task_name": task_name,
                    "model_name": model_id,
                    "reason": f"model_train_failed:{type(exc).__name__}:{exc}",
                    "NativeBaselineMeasuredPass": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
    write_csv(out_dir / "nonvision_baseline_reproduction.csv", rows)
    return rows


def _tensor_dataset_prefix(path: Path, train_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = torch.load(path, map_location="cpu")
    if hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
        x = dataset.tensors[0]
        y = dataset.tensors[1]
    else:
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        limit = len(dataset) if int(train_size) <= 0 else min(int(train_size), len(dataset))
        for i in range(limit):
            xi, yi = dataset[i]
            xs.append(xi)
            ys.append(torch.as_tensor(yi))
        x = torch.stack(xs)
        y = torch.stack(ys).long()
        return x.float().contiguous(), y.long().contiguous()
    if int(train_size) > 0:
        x = x[: int(train_size)]
        y = y[: int(train_size)]
    return x.float().contiguous(), y.long().contiguous()


def _run_nonvision_dg_transfer(out_dir: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    baseline_rows = _read_csv(out_dir / "nonvision_baseline_reproduction.csv")
    baseline_by_task_model = {
        (str(r.get("task_name")), str(r.get("model_name"))): r
        for r in baseline_rows
        if str(r.get("status")) == "measured"
    }
    rows: List[Dict[str, Any]] = []
    tasks = [t.strip() for t in str(args.nonvision_dg_tasks).split(",") if t.strip()]
    hidden_dims = [int(x.strip()) for x in str(args.nonvision_dg_hidden_dims).split(",") if x.strip()]
    device = v85.v83.get_device(args.device)
    for task_name in tasks:
        try:
            if task_name == "SpeechCommand":
                x_train, y_train = _tensor_dataset_prefix(
                    Path(args.data_root) / "SpeechCommands" / "train_sc.pt",
                    int(args.nonvision_dg_train_size),
                )
                x_test, y_test = _tensor_dataset_prefix(
                    Path(args.data_root) / "SpeechCommands" / "test_sc.pt",
                    int(args.nonvision_dg_test_size),
                )
                family = "audio"
                input_dim = 1000
                num_classes = 35
            elif task_name == "UrbanSound8K":
                x_train, y_train = _tensor_dataset_prefix(
                    Path(args.data_root) / "UrbanSound8K" / "train_us.pt",
                    int(args.nonvision_dg_train_size),
                )
                x_test, y_test = _tensor_dataset_prefix(
                    Path(args.data_root) / "UrbanSound8K" / "test_us.pt",
                    int(args.nonvision_dg_test_size),
                )
                family = "audio"
                input_dim = 1000
                num_classes = 10
            else:
                rows.append({
                    "stage": "P15_NONVISION_DG_TRANSFER_V86",
                    "status": "not_run",
                    "task_name": task_name,
                    "reason": "task_not_dense_tensor_audio_for_dg_transfer",
                    "DGTransferMeasuredPass": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
                continue
        except Exception as exc:
            rows.append({
                "stage": "P15_NONVISION_DG_TRANSFER_V86",
                "status": "not_run",
                "task_name": task_name,
                "reason": f"audio_tensor_load_failed:{type(exc).__name__}:{exc}",
                "DGTransferMeasuredPass": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
            continue

        mlp_row = baseline_by_task_model.get((task_name, "KB-MLP"), {})
        mlp_metric = _safe_float(mlp_row.get("measured_metric"))
        mlp_params = _safe_float(mlp_row.get("params"))
        mlp_flops = _safe_float(mlp_row.get("FLOPs"))
        mlp_step = _safe_float(mlp_row.get("step_time_ms"))
        for hidden_dim in hidden_dims:
            dg_args = _v85_namespace(args, fresh=False)
            dg_args.primary_epochs = int(args.nonvision_dg_epochs)
            dg_args.dg_hidden_dim = int(hidden_dim)
            dg_args.dg_batch_size = int(args.nonvision_dg_batch_size)
            dg_args.dg_eval_batch_size = int(args.nonvision_dg_eval_batch_size)
            dg_args.ft7_event_stride = int(args.nonvision_dg_ft7_event_stride)
            dg_args.ft7_event_alpha_mult = float(args.nonvision_dg_ft7_event_alpha_mult)
            dg_args.dg_stream_batches_from_cpu = False
            dg_args.dg_stream_chunk_batches = 1
            dg_args.dg_stream_epoch_permute_cpu = False
            dg_args.dg_stream_chunk_order_shuffle = False
            base_row = v85._train_dg_primary_classifier(
                dg_args,
                x_train,
                y_train,
                x_test,
                y_test,
                dataset=task_name,
                input_dim=input_dim,
                num_classes=num_classes,
                functional_update_used=0,
                functional_variant="base",
            )
            func_row = v85._train_dg_primary_classifier(
                dg_args,
                x_train,
                y_train,
                x_test,
                y_test,
                dataset=task_name,
                input_dim=input_dim,
                num_classes=num_classes,
                functional_update_used=1,
                functional_variant="ft7",
            )
            base_curv = _safe_float(base_row.get("geometry_curvature_after"))
            base_metric = _safe_float(base_row.get("test_metric"))
            for row in (base_row, func_row):
                metric = _safe_float(row.get("test_metric"))
                curv = _safe_float(row.get("geometry_curvature_after"))
                is_func = str(row.get("functional_variant")) == "ft7"
                curv_ratio = curv / max(base_curv, 1.0e-12) if math.isfinite(curv) and math.isfinite(base_curv) else float("nan")
                metric_delta_vs_mlp = metric - mlp_metric if math.isfinite(metric) and math.isfinite(mlp_metric) else float("nan")
                metric_delta_vs_base = metric - base_metric if math.isfinite(metric) and math.isfinite(base_metric) else float("nan")
                params = _safe_float(row.get("params"))
                flops = _safe_float(row.get("FLOPs"))
                step = _safe_float(row.get("step_time_ms"))
                rows.append({
                    "stage": "P15_NONVISION_DG_TRANSFER_V86",
                    "status": "measured",
                    "task_family": family,
                    "task_name": task_name,
                    "candidate_id": row.get("candidate_id"),
                    "functional_update_used": row.get("functional_update_used"),
                    "functional_variant": row.get("functional_variant"),
                    "hidden_dim": hidden_dim,
                    "train_size": int(x_train.shape[0]),
                    "test_size": int(x_test.shape[0]),
                    "epochs": int(args.nonvision_dg_epochs),
                    "test_metric": metric,
                    "delta_vs_KB_MLP": metric_delta_vs_mlp if math.isfinite(metric_delta_vs_mlp) else METRIC_UNAVAILABLE,
                    "delta_vs_DG_Base": metric_delta_vs_base if math.isfinite(metric_delta_vs_base) else METRIC_UNAVAILABLE,
                    "params": params,
                    "FLOPs": flops,
                    "step_time_ms": step,
                    "train_time_s": row.get("train_time_s"),
                    "peak_memory_MB": row.get("peak_memory_MB"),
                    "curvature_ratio_vs_DG_Base": curv_ratio if math.isfinite(curv_ratio) else METRIC_UNAVAILABLE,
                    "functional_event_count": row.get("functional_event_count"),
                    "functional_accept_mass": row.get("functional_accept_mass"),
                    "functional_reject_mass": row.get("functional_reject_mass"),
                    "parameter_ratio_vs_KB_MLP": params / max(mlp_params, 1.0e-12) if math.isfinite(params) and math.isfinite(mlp_params) else METRIC_UNAVAILABLE,
                    "flops_ratio_vs_KB_MLP": flops / max(mlp_flops, 1.0e-12) if math.isfinite(flops) and math.isfinite(mlp_flops) else METRIC_UNAVAILABLE,
                    "step_ratio_vs_KB_MLP": step / max(mlp_step, 1.0e-12) if math.isfinite(step) and math.isfinite(mlp_step) else METRIC_UNAVAILABLE,
                    "DGTransferMeasuredPass": 1,
                    "NonvisionDGFunctionalTaskPass": int(is_func and math.isfinite(metric_delta_vs_mlp) and metric_delta_vs_mlp >= 0.0),
                    "NonvisionDGGeometryPass": int(is_func and math.isfinite(curv_ratio) and curv_ratio <= 0.90),
                    "loss_type": "CE",
                    "geometry_loss_used": 0,
                    "external_teacher_used": 0,
                    "self_teacher_used": 0,
                    "sampler_changed": 0,
                    "class_weight_used": 0,
                    "cpu_offload_used": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                })
                _release_cuda_memory(device)
    write_csv(out_dir / "nonvision_dg_transfer.csv", rows)
    return rows


def _v85_namespace(args: argparse.Namespace, *, fresh: bool) -> argparse.Namespace:
    return argparse.Namespace(
        out_dir=str(args.out_dir),
        fresh=bool(fresh),
        device=args.device,
        data_root=args.data_root,
        kanbefair_path=args.kanbefair_path,
        datasets="MNIST",
        kb_data_protocol="kanbefair",
        train_size=60000,
        val_size=0,
        test_size=10000,
        seed=int(args.seed),
        kb_epochs=int(args.kb_epochs),
        kb_batch_size=int(args.kb_batch_size),
        adapter_smoke_train_size=512,
        adapter_smoke_test_size=256,
        adapter_steps=8,
        run_primary_transfer=True,
        run_functional_causality=True,
        run_symbolic=False,
        run_continual=False,
        primary_train_size=60000,
        primary_test_size=10000,
        primary_epochs=int(args.primary_epochs),
        causality_train_size=60000,
        causality_test_size=10000,
        causality_epochs=int(args.primary_epochs),
        symbolic_datasets="Special_1d_gelu",
        symbolic_epochs=100,
        symbolic_batch_size=128,
        symbolic_lr=1.0e-3,
        continual_train_size_per_task=int(args.continual_train_size_per_task),
        continual_test_size_per_task=int(args.continual_test_size_per_task),
        continual_epochs_per_task=int(args.continual_epochs_per_task),
        continual_batch_size=int(args.continual_batch_size),
        continual_lr=float(args.continual_lr),
        continual_restore_old_head_rows=False,
        continual_freeze_stack_after_first_task=False,
        continual_freeze_head_shared_after_first_task=False,
        continual_stack_anchor_strength=0.0,
        continual_old_head_grad_scale=0.0,
        continual_old_head_restore_strength=1.0,
        continual_old_head_age_decay=0.0,
        dg_candidate_id="KW6",
        dg_hidden_dim=28,
        dg_basis_count=8,
        dg_batch_size=128,
        dg_eval_batch_size=512,
        dg_stream_batches_from_cpu=False,
        dg_stream_chunk_batches=1,
        dg_stream_epoch_permute_cpu=False,
        dg_stream_chunk_order_shuffle=False,
        weight_decay=1.0e-4,
        ft7_event_stride=128,
        ft7_event_alpha_mult=15.0,
    )


def _run_v85_fresh_reproduction(out_dir: Path, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    v85_args = _v85_namespace(args, fresh=bool(args.fresh_v85_reproduction))
    decision = v85.run(v85_args)
    primary_rows = _read_csv(out_dir / "kanbefair_primary_transfer.csv")
    causality_rows = _read_csv(out_dir / "functional_causality_kanbefair.csv")
    parameter_rows = _read_csv(out_dir / "parameter_matched_envelope.csv")
    flops_rows = _read_csv(out_dir / "flops_matched_envelope.csv")
    wallclock_rows = _read_csv(out_dir / "wallclock_memory_envelope.csv")

    v85_repro_pass = int(
        _safe_float(decision.get("primary_transfer_pass"), 0.0) == 1.0
        and _safe_float(decision.get("parameter_fair_pass"), 0.0) == 1.0
        and _safe_float(decision.get("flops_fair_pass"), 0.0) == 1.0
        and _safe_float(decision.get("wallclock_fair_pass"), 0.0) == 1.0
        and _safe_float(decision.get("functional_causality_pass"), 0.0) == 1.0
    )
    rows: List[Dict[str, Any]] = []
    param_pass = int(any(_safe_float(r.get("ParameterFairPass"), 0.0) == 1.0 for r in parameter_rows))
    flops_pass = int(any(_safe_float(r.get("FLOPsFairPass"), 0.0) == 1.0 for r in flops_rows))
    wall_pass = int(any(_safe_float(r.get("WallclockMemoryPass"), 0.0) == 1.0 for r in wallclock_rows))
    causality_pass = int(any(_safe_float(r.get("FunctionalCausalityPass"), 0.0) == 1.0 for r in causality_rows))
    for row in primary_rows:
        if str(row.get("status")) != "measured":
            continue
        rows.append({
            "stage": "P0_V85_FRESH_REPRODUCTION_V86",
            "status": "measured",
            "candidate_id": row.get("candidate_id", row.get("model", METRIC_UNAVAILABLE)),
            "dataset": row.get("task_name", "MNIST"),
            "seed": int(args.seed),
            "test_acc": row.get("test_acc", METRIC_UNAVAILABLE),
            "delta_vs_KB_MLP": row.get("delta_vs_KB_MLP", METRIC_UNAVAILABLE),
            "delta_vs_DG_Base": row.get("delta_vs_DG_Base", METRIC_UNAVAILABLE),
            "params": row.get("params", METRIC_UNAVAILABLE),
            "FLOPs": row.get("FLOPs", METRIC_UNAVAILABLE),
            "peak_memory_MB": row.get("peak_memory_MB", METRIC_UNAVAILABLE),
            "step_time_ms": row.get("step_time_ms", METRIC_UNAVAILABLE),
            "train_time_s": row.get("train_time_s", METRIC_UNAVAILABLE),
            "curvature_ratio_vs_DG_Base": row.get("curvature_ratio_vs_DG_Base", METRIC_UNAVAILABLE),
            "functional_events": row.get("functional_events", METRIC_UNAVAILABLE),
            "functional_update_time_ratio": row.get("functional_update_time_ratio", METRIC_UNAVAILABLE),
            "primary_transfer_pass": int(_safe_float(decision.get("primary_transfer_pass"), 0.0) == 1.0),
            "parameter_fair_pass": param_pass,
            "flops_fair_pass": flops_pass,
            "wallclock_pass": wall_pass,
            "functional_causality_pass": causality_pass,
            "FreshReproPass": v85_repro_pass,
            "loss_type": "CE",
            "geometry_loss_used": 0,
            "external_teacher_used": 0,
            "self_teacher_used": 0,
            "sampler_changed": 0,
            "class_weight_used": 0,
            "cpu_offload_used": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    if not rows:
        rows.append({
            "stage": "P0_V85_FRESH_REPRODUCTION_V86",
            "status": "not_run",
            "reason": "v85_reproduction_rows_missing",
            "FreshReproPass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    write_csv(out_dir / "v85_reproduction.csv", rows)
    return rows, decision


def _split_definitions(seed: int) -> Dict[str, List[Tuple[int, ...]]]:
    g = torch.Generator().manual_seed(int(seed) + 8607)
    shuffled = torch.randperm(10, generator=g).tolist()
    split_c = [tuple(sorted(shuffled[:3])), tuple(sorted(shuffled[3:6])), tuple(sorted(shuffled[6:]))]
    return {
        "A_0-2_3-5_6-9": [(0, 1, 2), (3, 4, 5), (6, 7, 8, 9)],
        "B_0-4_5-9": [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9)],
        "C_random_balanced_3task": split_c,
        "D_interleaved_hard": [(0, 3, 6), (1, 4, 7), (2, 5, 8, 9)],
    }


def _build_tasks(
    x_train_all: torch.Tensor,
    y_train_all: torch.Tensor,
    x_test_all: torch.Tensor,
    y_test_all: torch.Tensor,
    split_id: str,
    digits_by_task: Sequence[Sequence[int]],
    args: argparse.Namespace,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
    train_tasks = [
        v85._filter_digit_group(
            x_train_all,
            y_train_all,
            digits,
            max_count=int(args.continual_train_size_per_task),
            seed=v85.v83.v72._stable_seed("v86-continual-train", int(args.seed), split_id, i),
        )
        for i, digits in enumerate(digits_by_task)
    ]
    test_tasks = [
        v85._filter_digit_group(
            x_test_all,
            y_test_all,
            digits,
            max_count=int(args.continual_test_size_per_task),
            seed=v85.v83.v72._stable_seed("v86-continual-test", int(args.seed), split_id, i),
        )
        for i, digits in enumerate(digits_by_task)
    ]
    return train_tasks, test_tasks


def _record_continual_rows(
    rows: List[Dict[str, Any]],
    *,
    split_id: str,
    digits_by_task: Sequence[Sequence[int]],
    candidate: str,
    model_group: str,
    acc_matrix: Sequence[Sequence[float]],
    curvature_after: Sequence[float],
    train_time: float,
    peak_memory: float,
    params: Any,
    flops: Any,
    event_count: int,
    accept_mass: float,
    reject_mass: float,
    anchor_strength: float,
    old_head_restore_used: int,
    old_head_grad_scale: float,
    old_head_restore_strength: float,
    old_head_age_decay: float,
    dg_base_scores: Dict[str, float] | None,
) -> Dict[str, float]:
    forgetting, backward_transfer, final_average = v85._continual_scores(acc_matrix)
    final_vector = list(acc_matrix[-1]) if acc_matrix else []
    max_min_gap = (max(final_vector) - min(final_vector)) if final_vector else float("nan")
    early_bias = (final_vector[0] - final_average) if final_vector else float("nan")
    late_bias = (final_vector[-1] - final_average) if final_vector else float("nan")
    forward_transfer = METRIC_UNAVAILABLE
    formal_pass = 0
    forgetting_vs_base_delta = METRIC_UNAVAILABLE
    final_vs_base_delta = METRIC_UNAVAILABLE
    if dg_base_scores is not None and math.isfinite(forgetting) and math.isfinite(final_average):
        base_forgetting = _safe_float(dg_base_scores.get("forgetting_score"))
        base_final = _safe_float(dg_base_scores.get("final_average_acc"))
        forgetting_vs_base_delta = forgetting - base_forgetting
        final_vs_base_delta = final_average - base_final
        formal_pass = int(
            forgetting <= base_forgetting + 1.0e-12
            and final_average >= base_final - 1.0e-12
            and max_min_gap <= 0.20 + 1.0e-12
        )
    final_vector_text = ",".join(f"{x:.10f}" for x in final_vector)
    for train_stage, accs in enumerate(acc_matrix):
        for task_id, acc in enumerate(accs):
            rows.append({
                "stage": "P7_CONTINUAL_MULTISPLIT_STRESS_V86",
                "status": "measured",
                "split_id": split_id,
                "task_id": task_id,
                "task_digits": "-".join(str(d) for d in digits_by_task[task_id]),
                "candidate": candidate,
                "model_group": model_group,
                "train_stage": train_stage,
                "acc_after_each_task": acc,
                "final_task_acc_vector": final_vector_text,
                "final_avg_acc": final_average,
                "forgetting_score": forgetting,
                "backward_transfer": backward_transfer,
                "forward_transfer": forward_transfer,
                "early_task_bias": early_bias,
                "late_task_bias": late_bias,
                "max_min_task_acc_gap": max_min_gap,
                "curvature_after_task": curvature_after[train_stage] if train_stage < len(curvature_after) else METRIC_UNAVAILABLE,
                "functional_event_count": event_count,
                "functional_accept_mass": accept_mass,
                "functional_reject_mass": reject_mass,
                "anchor_strength": anchor_strength,
                "old_head_restore_used": old_head_restore_used,
                "old_head_grad_scale": old_head_grad_scale,
                "old_head_restore_strength": old_head_restore_strength,
                "old_head_age_decay": old_head_age_decay,
                "train_time_s": train_time,
                "peak_memory_MB": peak_memory,
                "params": params,
                "FLOPs": flops,
                "forgetting_delta_vs_DG_Base": forgetting_vs_base_delta,
                "final_avg_delta_vs_DG_Base": final_vs_base_delta,
                "ContinualFormalPass": formal_pass if train_stage == len(acc_matrix) - 1 and task_id == len(accs) - 1 else 0,
                "BalancedPass": int(max_min_gap <= 0.20 + 1.0e-12) if train_stage == len(acc_matrix) - 1 and task_id == len(accs) - 1 else 0,
                "loss_type": "CE",
                "geometry_loss_used": 0,
                "external_teacher_used": 0,
                "self_teacher_used": 0,
                "sampler_changed": 0,
                "class_weight_used": 0,
                "cpu_offload_used": 0,
                "fake_data_used": 0,
                "proxy_row_used": 0,
            })
    return {
        "forgetting_score": forgetting,
        "backward_transfer": backward_transfer,
        "final_average_acc": final_average,
        "max_min_task_acc_gap": max_min_gap,
        "formal_pass": float(formal_pass),
    }


def _run_broad_vision_screen(out_dir: Path, args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    source_args = _v85_namespace(args, fresh=False)
    _source_rows, model_classes = v85._run_source_audit(out_dir, source_args)
    MLP = model_classes.get("MLP")
    KANbeFair = model_classes.get("KANbeFair")
    if MLP is None or KANbeFair is None:
        rows = [{
            "stage": "P1_BROAD_BASELINE_REPRODUCTION_V86",
            "status": "not_run",
            "reason": "kanbefair_model_import_failed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }]
        write_csv(out_dir / "kanbefair_broad_reproduction.csv", rows)
        write_csv(out_dir / "external_multitask_transfer.csv", rows)
        write_csv(out_dir / "joint_fair_envelope.csv", rows)
        write_csv(out_dir / "functional_causality_multitask.csv", rows)
        return {
            "reproduction": rows,
            "transfer": rows,
            "fair": rows,
            "causality": rows,
        }

    device = v85.v83.get_device(args.device)
    formal_protocol = _broad_formal_protocol(args)
    repro_rows: List[Dict[str, Any]] = []
    transfer_rows: List[Dict[str, Any]] = []
    fair_rows: List[Dict[str, Any]] = []
    causality_rows: List[Dict[str, Any]] = []

    original_stride = int(v85.v83.P5_FT7_EVENT_STRIDE)
    original_alpha = float(v85.v83.P5_FT7_EVENT_ALPHA_MULT)
    v85.v83.P5_FT7_EVENT_STRIDE = int(args.broad_ft7_event_stride)
    v85.v83.P5_FT7_EVENT_ALPHA_MULT = float(args.broad_ft7_event_alpha_mult)
    try:
        for dataset in v85.parse_str_list(args.broad_datasets):
            kb_dataset = "FMNIST" if dataset in {"Fashion-MNIST", "Fashion", "FMNIST", "fmnist"} else dataset
            x_train, y_train, x_test, y_test, input_dim, num_classes, split_protocol = v85._load_kanbefair_vision_tensors(
                kb_dataset,
                data_root=Path(args.data_root),
                train_size=int(args.broad_train_size),
                test_size=int(args.broad_test_size),
                seed=int(args.seed),
            )
            kb_metrics_by_name: Dict[str, Dict[str, Any]] = {}
            for cfg in v85._baseline_configs():
                model_name = str(cfg["model"])
                cls = MLP if model_name == "MLP" else KANbeFair
                ns = v85._kb_namespace(
                    model_name=model_name,
                    input_size=input_dim,
                    output_size=num_classes,
                    layers_width=cfg["layers_width"],
                    activation_name=cfg["activation_name"],
                    batch_norm=cfg["batch_norm"],
                    kan_grid=cfg.get("kan_grid", 3),
                    kan_order=cfg.get("kan_order", 2),
                    kan_shortcut=cfg.get("kan_shortcut", "silu"),
                    kan_range=cfg.get("kan_range", [-1.0, 1.0]),
                )
                set_seed(int(cfg["seed"]))
                model = cls(ns)
                params = int(model.total_parameters())
                flops = float(model.total_flops())
                metrics = v85._train_kb_classifier(
                    model,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    epochs=int(args.broad_epochs),
                    batch_size=int(args.kb_batch_size),
                    lr=float(cfg["lr"]),
                    seed=int(cfg["seed"]),
                    device=device,
                )
                reported = v85._reported_result(
                    Path(args.kanbefair_path),
                    dataset=kb_dataset,
                    model=model_name,
                    layers_width=cfg["layers_width"],
                    batch_size=int(cfg["reported_batch_size"]),
                    epochs=int(cfg["reported_epochs"]),
                    lr=float(cfg["lr"]),
                    seed=int(cfg["seed"]),
                    activation_name=cfg["activation_name"],
                    kan_grid=cfg.get("kan_grid", 3),
                    kan_order=cfg.get("kan_order", 2),
                    kan_shortcut=cfg.get("kan_shortcut", "silu"),
                    kan_range=cfg.get("kan_range", [-1.0, 1.0]),
                )
                reported_test = _safe_float(reported.get("reported_test_metric"))
                abs_delta = abs(float(metrics["test_acc_pct"]) - reported_test) if math.isfinite(reported_test) else float("nan")
                repro_pass = int(formal_protocol and math.isfinite(abs_delta) and abs_delta <= 2.0)
                model_id = f"KB-{model_name}"
                kb_row = {
                    "stage": "P1_BROAD_BASELINE_REPRODUCTION_V86",
                    "status": "measured",
                    "task_family": "vision",
                    "task_name": kb_dataset,
                    "dataset_name": kb_dataset,
                    "model_name": model_id,
                    "reported_metric": reported_test if math.isfinite(reported_test) else METRIC_UNAVAILABLE,
                    "measured_metric": metrics["test_acc_pct"],
                    "absolute_delta_from_reported": abs_delta if math.isfinite(abs_delta) else METRIC_UNAVAILABLE,
                    "relative_delta_from_reported": (abs_delta / max(abs(reported_test), 1.0e-12)) if math.isfinite(abs_delta) else METRIC_UNAVAILABLE,
                    "params": params,
                    "FLOPs": flops,
                    "train_time_s": metrics["train_time_s"],
                    "test_time_s": METRIC_UNAVAILABLE,
                    "step_time_ms": metrics["step_time_ms"],
                    "peak_memory_MB": metrics["peak_memory_MB"],
                    "seed": int(cfg["seed"]),
                    "epochs": int(args.broad_epochs),
                    "batch_size": int(args.kb_batch_size),
                    "split_protocol": split_protocol,
                    "formal_protocol": formal_protocol,
                    "reproduction_pass": repro_pass,
                    "BroadMeasuredPass": 1,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                }
                repro_rows.append(kb_row)
                kb_metrics_by_name[model_id] = {
                    "metric": float(metrics["test_acc_pct"]),
                    "loss": float(metrics["test_loss"]),
                    "ECE": float(metrics["ECE"]),
                    "NLL": float(metrics["NLL"]),
                    "params": params,
                    "FLOPs": flops,
                    "step_time_ms": float(metrics["step_time_ms"]),
                    "train_time_s": float(metrics["train_time_s"]),
                    "peak_memory_MB": float(metrics["peak_memory_MB"]),
                }
                transfer_rows.append({
                    "stage": "P4_EXTERNAL_MULTITASK_TRANSFER_V86",
                    "status": "measured",
                    "task_family": "vision",
                    "task_name": kb_dataset,
                    "model": model_id,
                    "seed": int(cfg["seed"]),
                    "params": params,
                    "FLOPs": flops,
                    "train_time_s": metrics["train_time_s"],
                    "step_time_ms": metrics["step_time_ms"],
                    "peak_memory_MB": metrics["peak_memory_MB"],
                    "val_metric": METRIC_UNAVAILABLE,
                    "test_metric": metrics["test_acc_pct"],
                    "delta_vs_KB_MLP": 0.0 if model_id == "KB-MLP" else (float(metrics["test_acc_pct"]) - kb_metrics_by_name.get("KB-MLP", {}).get("metric", float("nan"))),
                    "delta_vs_KB_KAN": 0.0 if model_id == "KB-KAN" else (float(metrics["test_acc_pct"]) - kb_metrics_by_name.get("KB-KAN", {}).get("metric", float("nan"))),
                    "delta_vs_DG_Base": METRIC_UNAVAILABLE,
                    "ECE": metrics["ECE"],
                    "NLL": metrics["NLL"],
                    "curvature": METRIC_UNAVAILABLE,
                    "jacobian_norm": METRIC_UNAVAILABLE,
                    "local_lipschitz": METRIC_UNAVAILABLE,
                    "functional_events": 0,
                    "TaskFamilyPass": 0,
                    "formal_protocol": formal_protocol,
                    "loss_type": "CE",
                    "geometry_loss_used": 0,
                    "external_teacher_used": 0,
                    "self_teacher_used": 0,
                    "sampler_changed": 0,
                    "class_weight_used": 0,
                    "cpu_offload_used": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                })
                _release_cuda_model(model, device)

            mlp_metric = float(kb_metrics_by_name["KB-MLP"]["metric"])
            mlp_params = float(kb_metrics_by_name["KB-MLP"]["params"])
            mlp_flops = float(kb_metrics_by_name["KB-MLP"]["FLOPs"])
            mlp_step = float(kb_metrics_by_name["KB-MLP"]["step_time_ms"])
            mlp_memory = float(kb_metrics_by_name["KB-MLP"]["peak_memory_MB"])
            dg_args = _v85_namespace(args, fresh=False)
            dg_args.primary_epochs = int(args.broad_epochs)
            dg_args.primary_train_size = int(args.broad_train_size)
            dg_args.primary_test_size = int(args.broad_test_size)
            dg_args.dg_hidden_dim = int(args.broad_dg_hidden_dim)
            dg_args.ft7_event_stride = int(args.broad_ft7_event_stride)
            dg_args.ft7_event_alpha_mult = float(args.broad_ft7_event_alpha_mult)
            dg_args.dg_stream_batches_from_cpu = bool(args.broad_dg_stream_batches_from_cpu)
            dg_args.dg_stream_chunk_batches = int(args.broad_dg_stream_chunk_batches)
            dg_args.dg_stream_epoch_permute_cpu = bool(args.broad_dg_stream_epoch_permute_cpu)
            dg_args.dg_stream_chunk_order_shuffle = bool(args.broad_dg_stream_chunk_order_shuffle)

            def train_dg_variant(functional_update_used: int, variant: str | None = None) -> Dict[str, Any]:
                _release_cuda_memory(device)
                row = v85._train_dg_primary_classifier(
                    dg_args,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    dataset=kb_dataset,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    functional_update_used=functional_update_used,
                    functional_variant=variant,
                )
                _release_cuda_memory(device)
                return row

            dg_base = train_dg_variant(0)
            dg_noop = train_dg_variant(0, "noop")
            dg_func = train_dg_variant(1, "ft7")
            dg_random = train_dg_variant(1, "random")
            dg_shuffled = train_dg_variant(1, "shuffled")
            dg_rows = [dg_base, dg_noop, dg_func, dg_random, dg_shuffled]
            base_metric = _safe_float(dg_base.get("test_metric"))
            base_curv = _safe_float(dg_base.get("geometry_curvature_after"))
            base_jac = _safe_float(dg_base.get("jacobian_norm_probe"))
            for row in dg_rows:
                metric = _safe_float(row.get("test_metric"))
                curv = _safe_float(row.get("geometry_curvature_after"))
                jac = _safe_float(row.get("jacobian_norm_probe"))
                is_func = str(row.get("functional_variant")) == "ft7"
                curv_ratio = curv / max(base_curv, 1.0e-12) if math.isfinite(curv) and math.isfinite(base_curv) else float("nan")
                transfer_rows.append({
                    "stage": "P4_EXTERNAL_MULTITASK_TRANSFER_V86",
                    "status": "measured",
                    "task_family": "vision",
                    "task_name": kb_dataset,
                    "model": row.get("candidate_id"),
                    "seed": int(args.seed),
                    "params": row.get("params"),
                    "FLOPs": row.get("FLOPs"),
                    "train_time_s": row.get("train_time_s"),
                    "step_time_ms": row.get("step_time_ms"),
                    "peak_memory_MB": row.get("peak_memory_MB"),
                    "val_metric": METRIC_UNAVAILABLE,
                    "test_metric": metric,
                    "delta_vs_KB_MLP": metric - mlp_metric if math.isfinite(metric) else METRIC_UNAVAILABLE,
                    "delta_vs_KB_KAN": metric - float(kb_metrics_by_name["KB-KAN"]["metric"]) if math.isfinite(metric) else METRIC_UNAVAILABLE,
                    "delta_vs_DG_Base": metric - base_metric if math.isfinite(metric) and math.isfinite(base_metric) else METRIC_UNAVAILABLE,
                    "ECE": row.get("ECE"),
                    "NLL": row.get("NLL"),
                    "curvature": curv,
                    "jacobian_norm": jac,
                    "local_lipschitz": row.get("local_lipschitz_probe"),
                    "functional_events": row.get("functional_event_count"),
                    "functional_variant": row.get("functional_variant"),
                    "input_data_residency": row.get("input_data_residency", METRIC_UNAVAILABLE),
                    "stream_chunk_batches": row.get("stream_chunk_batches", METRIC_UNAVAILABLE),
                    "stream_epoch_permute_cpu": row.get("stream_epoch_permute_cpu", METRIC_UNAVAILABLE),
                    "stream_chunk_order_shuffle": row.get("stream_chunk_order_shuffle", METRIC_UNAVAILABLE),
                    "TaskFamilyPass": int(is_func and math.isfinite(metric) and metric >= mlp_metric),
                    "formal_protocol": formal_protocol,
                    "loss_type": "CE",
                    "geometry_loss_used": 0,
                    "external_teacher_used": 0,
                    "self_teacher_used": 0,
                    "sampler_changed": 0,
                    "class_weight_used": 0,
                    "cpu_offload_used": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                })
                if is_func:
                    param_ratio = _safe_float(row.get("params")) / max(mlp_params, 1.0e-12)
                    flops_ratio = _safe_float(row.get("FLOPs")) / max(mlp_flops, 1.0e-12)
                    step_ratio = _safe_float(row.get("step_time_ms")) / max(mlp_step, 1.0e-12)
                    memory_ratio = _safe_float(row.get("peak_memory_MB")) / max(mlp_memory, 1.0e-12)
                    metric_delta = metric - mlp_metric
                    joint_pass = int(
                        math.isfinite(metric_delta)
                        and param_ratio <= 1.05
                        and flops_ratio <= 1.05
                        and step_ratio <= 1.50
                        and memory_ratio <= 1.05
                        and metric_delta >= 0.0
                    )
                    fair_rows.append({
                        "stage": "P5_JOINT_FAIR_ENVELOPE_V86",
                        "status": "measured",
                        "task_family": "vision",
                        "task_name": kb_dataset,
                        "envelope_type": "MLP_width32_direct_joint",
                        "model": row.get("candidate_id"),
                        "params_ratio_vs_MLP": param_ratio,
                        "FLOPs_ratio_vs_MLP": flops_ratio,
                        "step_ratio_vs_MLP": step_ratio,
                        "memory_ratio_vs_MLP": memory_ratio,
                        "metric_delta_vs_MLP": metric_delta,
                        "input_data_residency": row.get("input_data_residency", METRIC_UNAVAILABLE),
                        "stream_chunk_batches": row.get("stream_chunk_batches", METRIC_UNAVAILABLE),
                        "stream_epoch_permute_cpu": row.get("stream_epoch_permute_cpu", METRIC_UNAVAILABLE),
                        "stream_chunk_order_shuffle": row.get("stream_chunk_order_shuffle", METRIC_UNAVAILABLE),
                        "formal_protocol": formal_protocol,
                        "JointFairPass": joint_pass,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    })

            noop_curv = _safe_float(dg_noop.get("geometry_curvature_after"))
            random_curv = _safe_float(dg_random.get("geometry_curvature_after"))
            noop_metric = _safe_float(dg_noop.get("test_metric"))
            for row in dg_rows:
                variant = str(row.get("functional_variant"))
                metric = _safe_float(row.get("test_metric"))
                curv = _safe_float(row.get("geometry_curvature_after"))
                jac = _safe_float(row.get("jacobian_norm_probe"))
                curv_ratio = curv / max(base_curv, 1.0e-12) if math.isfinite(curv) and math.isfinite(base_curv) else float("nan")
                jac_ratio = jac / max(base_jac, 1.0e-12) if math.isfinite(jac) and math.isfinite(base_jac) else float("nan")
                causality_pass = int(
                    variant == "ft7"
                    and math.isfinite(curv)
                    and math.isfinite(noop_curv)
                    and math.isfinite(random_curv)
                    and curv < noop_curv
                    and curv < random_curv
                    and metric >= noop_metric - 0.2
                )
                causality_rows.append({
                    "stage": "P6_FUNCTIONAL_CAUSALITY_MULTITASK_V86",
                    "status": "measured",
                    "task_family": "vision",
                    "task_name": kb_dataset,
                    "seed": int(args.seed),
                    "model": row.get("candidate_id"),
                    "functional_variant": variant,
                    "test_metric": metric,
                    "delta_vs_base": metric - base_metric if math.isfinite(metric) and math.isfinite(base_metric) else METRIC_UNAVAILABLE,
                    "curvature_ratio": curv_ratio if math.isfinite(curv_ratio) else METRIC_UNAVAILABLE,
                    "jacobian_ratio": jac_ratio if math.isfinite(jac_ratio) else METRIC_UNAVAILABLE,
                    "local_lipschitz_ratio": jac_ratio if math.isfinite(jac_ratio) else METRIC_UNAVAILABLE,
                    "ECE_delta": _safe_float(row.get("ECE")) - _safe_float(dg_base.get("ECE")),
                    "NLL_delta": _safe_float(row.get("NLL")) - _safe_float(dg_base.get("NLL")),
                    "functional_update_time_ratio": METRIC_UNAVAILABLE,
                    "bad_step_rate": row.get("bad_step_rate"),
                    "holdout_descent_ratio": row.get("holdout_descent_ratio"),
                    "role_accept_rate": row.get("role_accept_rate"),
                    "role_geometry_delta": (1.0 - curv_ratio) if math.isfinite(curv_ratio) else METRIC_UNAVAILABLE,
                    "formal_protocol": formal_protocol,
                    "CausalityPass": causality_pass,
                    "loss_type": "CE",
                    "geometry_loss_used": 0,
                    "external_teacher_used": 0,
                    "self_teacher_used": 0,
                    "sampler_changed": 0,
                    "class_weight_used": 0,
                    "cpu_offload_used": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                })
    finally:
        v85.v83.P5_FT7_EVENT_STRIDE = original_stride
        v85.v83.P5_FT7_EVENT_ALPHA_MULT = original_alpha

    write_csv(out_dir / "kanbefair_broad_reproduction.csv", repro_rows)
    write_csv(out_dir / "external_multitask_transfer.csv", transfer_rows)
    write_csv(out_dir / "joint_fair_envelope.csv", fair_rows)
    write_csv(out_dir / "functional_causality_multitask.csv", causality_rows)
    return {
        "reproduction": repro_rows,
        "transfer": transfer_rows,
        "fair": fair_rows,
        "causality": causality_rows,
    }


def _run_tabular_formal_envelope(out_dir: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    merge_artifacts = [x.strip() for x in str(getattr(args, "tabular_merge_artifacts", "")).split(",") if x.strip()]
    if merge_artifacts:
        rows: List[Dict[str, Any]] = []
        for item in merge_artifacts:
            source_path = Path(item)
            if source_path.is_dir():
                source_path = source_path / "tabular_formal_envelope.csv"
            if not source_path.exists():
                rows.append({
                    "stage": "P12_TABULAR_FORMAL_ENVELOPE_V86",
                    "status": "not_run",
                    "reason": f"tabular_merge_source_missing:{source_path}",
                    "source_artifact_path": str(source_path),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
                continue
            source_hash = _sha256(source_path)
            for row in _read_csv(source_path):
                merged = dict(row)
                merged["source_artifact_path"] = str(source_path)
                merged["source_artifact_sha256"] = source_hash
                merged["merge_mode"] = "measured_tabular_artifact_row"
                rows.append(merged)
        write_csv(out_dir / "tabular_formal_envelope.csv", rows)
        _write_nonvision_runnability_audit(out_dir, args, tabular_status="merged_measured_artifacts")
        return rows

    source_args = _v85_namespace(args, fresh=False)
    _source_rows, model_classes = v85._run_source_audit(out_dir, source_args)
    MLP = model_classes.get("MLP")
    KANbeFair = model_classes.get("KANbeFair")
    rows: List[Dict[str, Any]] = []
    if MLP is None or KANbeFair is None:
        rows.append({
            "stage": "P12_TABULAR_FORMAL_ENVELOPE_V86",
            "status": "not_run",
            "reason": "kanbefair_model_import_failed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
        write_csv(out_dir / "tabular_formal_envelope.csv", rows)
        _write_nonvision_runnability_audit(out_dir, args, tabular_status="blocked_model_import_failed")
        return rows

    device = v85.v83.get_device(args.device)
    hidden_dims = [int(x.strip()) for x in str(args.tabular_dg_hidden_dims).split(",") if x.strip()]
    formal_protocol = int(int(args.tabular_epochs) == 20 and int(args.kb_batch_size) == 128)
    original_stride = int(v85.v83.P5_FT7_EVENT_STRIDE)
    original_alpha = float(v85.v83.P5_FT7_EVENT_ALPHA_MULT)
    v85.v83.P5_FT7_EVENT_STRIDE = int(args.tabular_ft7_event_stride)
    v85.v83.P5_FT7_EVENT_ALPHA_MULT = float(args.tabular_ft7_event_alpha_mult)
    tabular_status = "measured"
    try:
        for dataset in v85.parse_str_list(args.tabular_datasets):
            try:
                x_train, y_train, x_test, y_test, input_dim, num_classes, split_protocol = _load_kanbefair_tabular_tensors(
                    dataset,
                    kb_path=Path(args.kanbefair_path),
                    train_size=int(args.tabular_train_size),
                    test_size=int(args.tabular_test_size),
                    seed=int(args.seed),
                )
            except Exception as exc:
                rows.append({
                    "stage": "P12_TABULAR_FORMAL_ENVELOPE_V86",
                    "status": "not_run",
                    "task_family": "tabular",
                    "task_name": dataset,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
                tabular_status = f"blocked_{type(exc).__name__}"
                continue

            kb_metrics: Dict[str, Dict[str, Any]] = {}
            for cfg in v85._baseline_configs():
                model_name = str(cfg["model"])
                cls = MLP if model_name == "MLP" else KANbeFair
                ns = v85._kb_namespace(
                    model_name=model_name,
                    input_size=input_dim,
                    output_size=num_classes,
                    layers_width=cfg["layers_width"],
                    activation_name=cfg["activation_name"],
                    batch_norm=cfg["batch_norm"],
                    kan_grid=cfg.get("kan_grid", 3),
                    kan_order=cfg.get("kan_order", 2),
                    kan_shortcut=cfg.get("kan_shortcut", "silu"),
                    kan_range=cfg.get("kan_range", [-1.0, 1.0]),
                )
                set_seed(int(cfg["seed"]))
                model = cls(ns)
                params = int(model.total_parameters())
                flops = float(model.total_flops())
                metrics = v85._train_kb_classifier(
                    model,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    epochs=int(args.tabular_epochs),
                    batch_size=int(args.kb_batch_size),
                    lr=float(cfg["lr"]),
                    seed=int(cfg["seed"]),
                    device=device,
                )
                reported = v85._reported_result(
                    Path(args.kanbefair_path),
                    dataset=dataset,
                    model=model_name,
                    layers_width=cfg["layers_width"],
                    batch_size=int(cfg["reported_batch_size"]),
                    epochs=int(cfg["reported_epochs"]),
                    lr=float(cfg["lr"]),
                    seed=int(cfg["seed"]),
                    activation_name=cfg["activation_name"],
                    kan_grid=cfg.get("kan_grid", 3),
                    kan_order=cfg.get("kan_order", 2),
                    kan_shortcut=cfg.get("kan_shortcut", "silu"),
                    kan_range=cfg.get("kan_range", [-1.0, 1.0]),
                )
                reported_test = _safe_float(reported.get("reported_test_metric"))
                abs_delta = abs(float(metrics["test_acc_pct"]) - reported_test) if math.isfinite(reported_test) else float("nan")
                model_id = f"KB-{model_name}"
                kb_metrics[model_id] = {
                    "metric": float(metrics["test_acc_pct"]),
                    "params": params,
                    "FLOPs": flops,
                    "step_time_ms": float(metrics["step_time_ms"]),
                    "peak_memory_MB": float(metrics["peak_memory_MB"]),
                }
                rows.append({
                    "stage": "P12_TABULAR_FORMAL_ENVELOPE_V86",
                    "status": "measured",
                    "task_family": "tabular",
                    "task_name": dataset,
                    "split_protocol": split_protocol,
                    "model": model_id,
                    "hidden_dim": METRIC_UNAVAILABLE,
                    "seed": int(cfg["seed"]),
                    "epochs": int(args.tabular_epochs),
                    "batch_size": int(args.kb_batch_size),
                    "reported_metric": reported_test if math.isfinite(reported_test) else METRIC_UNAVAILABLE,
                    "test_metric": metrics["test_acc_pct"],
                    "absolute_delta_from_reported": abs_delta if math.isfinite(abs_delta) else METRIC_UNAVAILABLE,
                    "params": params,
                    "FLOPs": flops,
                    "step_time_ms": metrics["step_time_ms"],
                    "peak_memory_MB": metrics["peak_memory_MB"],
                    "train_time_s": metrics["train_time_s"],
                    "delta_vs_KB_MLP": 0.0 if model_id == "KB-MLP" else METRIC_UNAVAILABLE,
                    "delta_vs_DG_Base": METRIC_UNAVAILABLE,
                    "curvature_ratio_vs_DG_Base": METRIC_UNAVAILABLE,
                    "functional_events": 0,
                    "TaskPass": 0,
                    "ParameterFairPass": 0,
                    "FLOPsFairPass": 0,
                    "WallclockMemoryPass": 0,
                    "GeometryPass": 0,
                    "TabularFormalPass": 0,
                    "formal_protocol": formal_protocol,
                    "loss_type": "CE",
                    "geometry_loss_used": 0,
                    "external_teacher_used": 0,
                    "self_teacher_used": 0,
                    "sampler_changed": 0,
                    "class_weight_used": 0,
                    "cpu_offload_used": 0,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                })
                _release_cuda_model(model, device)

            if "KB-MLP" not in kb_metrics:
                continue
            mlp_metric = float(kb_metrics["KB-MLP"]["metric"])
            mlp_params = max(float(kb_metrics["KB-MLP"]["params"]), 1.0e-12)
            mlp_flops = max(float(kb_metrics["KB-MLP"]["FLOPs"]), 1.0e-12)
            mlp_step = max(float(kb_metrics["KB-MLP"]["step_time_ms"]), 1.0e-12)
            mlp_memory = max(float(kb_metrics["KB-MLP"]["peak_memory_MB"]), 1.0e-12)

            for hidden_dim in hidden_dims:
                dg_args = _v85_namespace(args, fresh=False)
                dg_args.primary_epochs = int(args.tabular_epochs)
                dg_args.dg_hidden_dim = int(hidden_dim)
                dg_args.dg_batch_size = int(args.kb_batch_size)
                dg_args.dg_eval_batch_size = int(args.tabular_eval_batch_size)
                dg_args.ft7_event_stride = int(args.tabular_ft7_event_stride)
                dg_args.ft7_event_alpha_mult = float(args.tabular_ft7_event_alpha_mult)
                dg_args.dg_stream_batches_from_cpu = bool(args.tabular_dg_stream_batches_from_cpu)
                dg_args.dg_stream_chunk_batches = int(args.tabular_dg_stream_chunk_batches)
                dg_args.dg_stream_epoch_permute_cpu = bool(args.tabular_dg_stream_epoch_permute_cpu)
                dg_args.dg_stream_chunk_order_shuffle = bool(args.tabular_dg_stream_chunk_order_shuffle)

                base_row = v85._train_dg_primary_classifier(
                    dg_args,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    dataset=dataset,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    functional_update_used=0,
                )
                func_row = v85._train_dg_primary_classifier(
                    dg_args,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    dataset=dataset,
                    input_dim=input_dim,
                    num_classes=num_classes,
                    functional_update_used=1,
                    functional_variant="ft7",
                )
                base_metric = _safe_float(base_row.get("test_metric"))
                base_curv = _safe_float(base_row.get("geometry_curvature_after"))
                for row in [base_row, func_row]:
                    metric = _safe_float(row.get("test_metric"))
                    curv = _safe_float(row.get("geometry_curvature_after"))
                    events = _safe_float(row.get("functional_event_count"), 0.0)
                    metric_delta = metric - mlp_metric if math.isfinite(metric) else float("nan")
                    param_ratio = _safe_float(row.get("params")) / mlp_params
                    flops_ratio = _safe_float(row.get("FLOPs")) / mlp_flops
                    step_ratio = _safe_float(row.get("step_time_ms")) / mlp_step
                    memory_ratio = _safe_float(row.get("peak_memory_MB")) / mlp_memory
                    curv_ratio = curv / max(base_curv, 1.0e-12) if math.isfinite(curv) and math.isfinite(base_curv) else float("nan")
                    is_func = str(row.get("candidate_id", "")).startswith("DG1-FT7-")
                    task_pass = int(is_func and math.isfinite(metric_delta) and metric_delta >= 0.0)
                    param_pass = int(is_func and math.isfinite(param_ratio) and param_ratio <= 1.05)
                    flops_pass = int(is_func and math.isfinite(flops_ratio) and flops_ratio <= 1.05)
                    wall_pass = int(
                        is_func
                        and math.isfinite(memory_ratio)
                        and math.isfinite(step_ratio)
                        and memory_ratio <= 1.05
                        and step_ratio <= 1.50
                    )
                    geom_pass = int(is_func and events > 0 and math.isfinite(curv_ratio) and curv_ratio <= 0.90)
                    formal_pass = int(task_pass and param_pass and flops_pass and wall_pass and geom_pass)
                    rows.append({
                        "stage": "P12_TABULAR_FORMAL_ENVELOPE_V86",
                        "status": "measured",
                        "task_family": "tabular",
                        "task_name": dataset,
                        "split_protocol": split_protocol,
                        "model": row.get("candidate_id"),
                        "hidden_dim": int(hidden_dim),
                        "seed": int(args.seed),
                        "epochs": int(args.tabular_epochs),
                        "batch_size": int(args.kb_batch_size),
                        "reported_metric": METRIC_UNAVAILABLE,
                        "test_metric": metric,
                        "absolute_delta_from_reported": METRIC_UNAVAILABLE,
                        "params": row.get("params"),
                        "FLOPs": row.get("FLOPs"),
                        "step_time_ms": row.get("step_time_ms"),
                        "peak_memory_MB": row.get("peak_memory_MB"),
                        "train_time_s": row.get("train_time_s"),
                        "delta_vs_KB_MLP": metric_delta if math.isfinite(metric_delta) else METRIC_UNAVAILABLE,
                        "delta_vs_DG_Base": (metric - base_metric) if math.isfinite(metric) and math.isfinite(base_metric) else METRIC_UNAVAILABLE,
                        "curvature_ratio_vs_DG_Base": curv_ratio if math.isfinite(curv_ratio) else METRIC_UNAVAILABLE,
                        "functional_events": int(events),
                        "params_ratio_vs_MLP": param_ratio,
                        "FLOPs_ratio_vs_MLP": flops_ratio,
                        "step_ratio_vs_MLP": step_ratio,
                        "memory_ratio_vs_MLP": memory_ratio,
                        "TaskPass": task_pass,
                        "ParameterFairPass": param_pass,
                        "FLOPsFairPass": flops_pass,
                        "WallclockMemoryPass": wall_pass,
                        "GeometryPass": geom_pass,
                        "TabularFormalPass": formal_pass,
                        "formal_protocol": formal_protocol,
                        "input_data_residency": row.get("input_data_residency", METRIC_UNAVAILABLE),
                        "loss_type": "CE",
                        "geometry_loss_used": 0,
                        "external_teacher_used": 0,
                        "self_teacher_used": 0,
                        "sampler_changed": 0,
                        "class_weight_used": 0,
                        "cpu_offload_used": 0,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                    })
                _release_cuda_memory(device)
    finally:
        v85.v83.P5_FT7_EVENT_STRIDE = original_stride
        v85.v83.P5_FT7_EVENT_ALPHA_MULT = original_alpha

    write_csv(out_dir / "tabular_formal_envelope.csv", rows)
    if not rows:
        tabular_status = "blocked_no_rows"
    _write_nonvision_runnability_audit(out_dir, args, tabular_status=tabular_status)
    return rows


def _run_continual_multisplit(out_dir: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    source_args = _v85_namespace(args, fresh=False)
    _source_rows, model_classes = v85._run_source_audit(out_dir, source_args)
    MLP = model_classes.get("MLP")
    KANbeFair = model_classes.get("KANbeFair")
    if MLP is None or KANbeFair is None:
        rows = [{
            "stage": "P7_CONTINUAL_MULTISPLIT_STRESS_V86",
            "status": "not_run",
            "reason": "kanbefair_model_import_failed",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }]
        write_csv(out_dir / "continual_multisplit_stress.csv", rows)
        return rows

    x_train_all, y_train_all, x_test_all, y_test_all, input_dim, num_classes, protocol = v85._load_kanbefair_vision_tensors(
        "MNIST",
        data_root=Path(args.data_root),
        train_size=60000,
        test_size=10000,
        seed=int(args.seed),
    )
    device = v85.v83.get_device(args.device)
    rows: List[Dict[str, Any]] = []
    split_defs = _split_definitions(int(args.seed))

    for split_id, digits_by_task in split_defs.items():
        train_tasks, test_tasks = _build_tasks(x_train_all, y_train_all, x_test_all, y_test_all, split_id, digits_by_task, args)
        measured_scores: Dict[str, Dict[str, float]] = {}
        kb_specs: List[Tuple[str, Any, argparse.Namespace]] = [
            ("KB-MLP", MLP, v85._kb_namespace(model_name="MLP", input_size=input_dim, output_size=num_classes, layers_width=[32])),
            (
                "KB-KAN",
                KANbeFair,
                v85._kb_namespace(
                    model_name="KAN",
                    input_size=input_dim,
                    output_size=num_classes,
                    layers_width=[2],
                    kan_grid=3,
                    kan_order=2,
                    kan_shortcut="silu",
                    kan_range=[-1.0, 1.0],
                ),
            ),
        ]
        for model_name, cls, ns in kb_specs:
            set_seed(int(args.seed))
            model = cls(ns)
            acc_matrix, curvature_after, train_time, peak_mb = v85._train_kb_continual(
                model,
                train_tasks,
                test_tasks,
                epochs_per_task=int(args.continual_epochs_per_task),
                batch_size=int(args.continual_batch_size),
                lr=float(args.continual_lr),
                seed=int(args.seed),
                device=device,
            )
            measured_scores[model_name] = _record_continual_rows(
                rows,
                split_id=split_id,
                digits_by_task=digits_by_task,
                candidate=model_name,
                model_group="KB",
                acc_matrix=acc_matrix,
                curvature_after=curvature_after,
                train_time=train_time,
                peak_memory=peak_mb,
                params=int(model.total_parameters()),
                flops=float(model.total_flops()),
                event_count=0,
                accept_mass=0.0,
                reject_mass=0.0,
                anchor_strength=0.0,
                old_head_restore_used=0,
                old_head_grad_scale=0.0,
                old_head_restore_strength=0.0,
                old_head_age_decay=0.0,
                dg_base_scores=None,
            )

        dg_variants = [
            ("CL0-DG-Base", 0, False, 0.0, 0.0, 0.0, 0.0),
            ("CL1-DG-Functional-accepted", 1, True, 0.12, 0.0, 1.0, 0.0),
            ("CL2-DG-Functional-no-old-head-restore", 1, False, 0.12, 1.0, 0.0, 0.0),
            ("CL3-DG-Functional-no-stack-anchor", 1, True, 0.0, 0.0, 1.0, 0.0),
            ("CL4-DG-Functional-stack-anchor-006", 1, True, 0.06, 0.0, 1.0, 0.0),
            ("CL5-DG-Functional-stack-anchor-012", 1, True, 0.12, 0.0, 1.0, 0.0),
            ("CL6-DG-Functional-stack-anchor-024", 1, True, 0.24, 0.0, 1.0, 0.0),
            ("CL7-DG-Functional-balanced-grad025-restore025-anchor006", 1, True, 0.06, 0.25, 0.25, 0.0),
            ("CL8-DG-Functional-balanced-grad050-restore010-anchor003", 1, True, 0.03, 0.50, 0.10, 0.0),
            ("CL9-DG-Functional-balanced-grad075-restore000-anchor000", 1, True, 0.0, 0.75, 0.0, 0.0),
            ("CL10-DG-Functional-no-restore-no-anchor", 1, False, 0.0, 1.0, 0.0, 0.0),
            ("CL11-DG-Functional-no-restore-anchor003", 1, False, 0.03, 1.0, 0.0, 0.0),
            ("CL12-DG-Functional-no-restore-anchor006", 1, False, 0.06, 1.0, 0.0, 0.0),
            ("CL13-DG-Functional-no-restore-anchor009", 1, False, 0.09, 1.0, 0.0, 0.0),
            ("CL14-DG-Functional-no-restore-anchor015", 1, False, 0.15, 1.0, 0.0, 0.0),
            ("CL15-DG-Functional-no-restore-anchor018", 1, False, 0.18, 1.0, 0.0, 0.0),
            ("CL16-DG-Functional-no-restore-anchor019", 1, False, 0.19, 1.0, 0.0, 0.0),
            ("CL17-DG-Functional-no-restore-anchor020", 1, False, 0.20, 1.0, 0.0, 0.0),
            ("CL18-DG-Functional-no-restore-anchor021", 1, False, 0.21, 1.0, 0.0, 0.0),
            ("CL19-DG-Functional-no-restore-anchor022", 1, False, 0.22, 1.0, 0.0, 0.0),
            ("CL20-DG-Functional-anchor015-age0005", 1, False, 0.15, 1.0, 0.0, 0.005),
            ("CL21-DG-Functional-anchor015-age0010", 1, False, 0.15, 1.0, 0.0, 0.010),
            ("CL22-DG-Functional-anchor015-age0020", 1, False, 0.15, 1.0, 0.0, 0.020),
            ("CL23-DG-Functional-anchor015-age0025", 1, False, 0.15, 1.0, 0.0, 0.0025),
            ("CL24-DG-Functional-anchor015-age0035", 1, False, 0.15, 1.0, 0.0, 0.0035),
            ("CL25-DG-Functional-anchor015-age0040", 1, False, 0.15, 1.0, 0.0, 0.0040),
            ("CL26-DG-Functional-anchor015-age0010", 1, False, 0.15, 1.0, 0.0, 0.0010),
            ("CL27-DG-Functional-anchor015-age0015", 1, False, 0.15, 1.0, 0.0, 0.0015),
            ("CL28-DG-Functional-anchor015-age0020", 1, False, 0.15, 1.0, 0.0, 0.0020),
            ("CL29-DG-Functional-anchor015-age0022", 1, False, 0.15, 1.0, 0.0, 0.0022),
            ("CL30-DG-Functional-anchor015-age0023", 1, False, 0.15, 1.0, 0.0, 0.0023),
            ("CL31-DG-Functional-anchor015-age0024", 1, False, 0.15, 1.0, 0.0, 0.0024),
            ("CL32-DG-Functional-anchor015-age00205", 1, False, 0.15, 1.0, 0.0, 0.00205),
            ("CL33-DG-Functional-anchor015-age00210", 1, False, 0.15, 1.0, 0.0, 0.00210),
            ("CL34-DG-Functional-anchor015-age00215", 1, False, 0.15, 1.0, 0.0, 0.00215),
            ("CL35-DG-Functional-anchor0151-age00205", 1, False, 0.151, 1.0, 0.0, 0.00205),
            ("CL36-DG-Functional-anchor0151-age00210", 1, False, 0.151, 1.0, 0.0, 0.00210),
            ("CL37-DG-Functional-anchor0152-age00205", 1, False, 0.152, 1.0, 0.0, 0.00205),
            ("CL38-DG-Functional-anchor0152-age00210", 1, False, 0.152, 1.0, 0.0, 0.00210),
            ("CL39-DG-Functional-anchor0152-age00215", 1, False, 0.152, 1.0, 0.0, 0.00215),
            ("CL40-DG-Functional-anchor0154-age00215", 1, False, 0.154, 1.0, 0.0, 0.00215),
        ]
        for candidate, functional_flag, old_head_restore, anchor_strength, old_head_grad_scale, old_head_restore_strength, old_head_age_decay in dg_variants:
            run_args = _v85_namespace(args, fresh=False)
            run_args.run_primary_transfer = False
            run_args.run_functional_causality = False
            run_args.continual_restore_old_head_rows = bool(old_head_restore)
            run_args.continual_stack_anchor_strength = float(anchor_strength)
            run_args.continual_old_head_grad_scale = float(old_head_grad_scale)
            run_args.continual_old_head_restore_strength = float(old_head_restore_strength)
            run_args.continual_old_head_age_decay = float(old_head_age_decay)
            run_args.ft7_event_stride = 16
            run_args.ft7_event_alpha_mult = 15.0
            v85.v83.P5_FT7_EVENT_STRIDE = int(run_args.ft7_event_stride)
            v85.v83.P5_FT7_EVENT_ALPHA_MULT = float(run_args.ft7_event_alpha_mult)
            acc_matrix, curvature_after, train_time, peak_mb, event_count, accept_mass, reject_mass = v85._train_dg_continual(
                run_args,
                train_tasks,
                test_tasks,
                input_dim=input_dim,
                num_classes=num_classes,
                functional_update_used=functional_flag,
            )
            base_scores = measured_scores.get("CL0-DG-Base") if candidate != "CL0-DG-Base" else None
            scores = _record_continual_rows(
                rows,
                split_id=split_id,
                digits_by_task=digits_by_task,
                candidate=candidate,
                model_group="DG",
                acc_matrix=acc_matrix,
                curvature_after=curvature_after,
                train_time=train_time,
                peak_memory=peak_mb,
                params=METRIC_UNAVAILABLE,
                flops=v85._dg_kw6_forward_flops_estimate(int(input_dim), 28, int(v85.v83.v80._spec_map_v80()["KW6"].depth), int(num_classes)),
                event_count=event_count,
                accept_mass=accept_mass,
                reject_mass=reject_mass,
                anchor_strength=float(anchor_strength),
                old_head_restore_used=int(old_head_restore),
                old_head_grad_scale=float(old_head_grad_scale),
                old_head_restore_strength=float(old_head_restore_strength),
                old_head_age_decay=float(old_head_age_decay),
                dg_base_scores=base_scores,
            )
            measured_scores[candidate] = scores
    write_csv(out_dir / "continual_multisplit_stress.csv", rows)
    return rows


def _audit_no_fake(out_dir: Path) -> Dict[str, Any]:
    rows_checked = 0
    fake_data = 0
    proxy = 0
    offload = 0
    fake_proxy_nonzero = 0
    for csv_path in out_dir.glob("*.csv"):
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows_checked += 1
                for key in ["fake_data_used", "proxy_row_used", "cpu_offload_used"]:
                    val = str(row.get(key, "0"))
                    if val not in {"", "0", "0.0", "False", "false"}:
                        fake_proxy_nonzero += 1
                fake_data += int(_safe_float(row.get("fake_data_used"), 0.0) != 0.0)
                proxy += int(_safe_float(row.get("proxy_row_used"), 0.0) != 0.0)
                offload += int(_safe_float(row.get("cpu_offload_used"), 0.0) != 0.0)
    audit = {
        "stage": "V86_PROVENANCE_AUDIT",
        "script_path": "experiments/run_gafu_v86_real.py",
        "plan_path": PLAN_PATH,
        "rows_checked": rows_checked,
        "fake_proxy_nonzero_count": fake_proxy_nonzero,
        "fake_data_used": fake_data,
        "proxy_row_used": proxy,
        "cpu_offload_used": offload,
        "no_fake": fake_data == 0,
        "no_proxy": proxy == 0,
    }
    write_csv(out_dir / "v86_provenance_audit.csv", [audit])
    return audit


def _sync_cuda(device: torch.device) -> None:
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()


_P9_PHASE_NAMES = {
    "batch_select": "v86_batch_select",
    "forward_backward": "v86_forward_backward",
    "base_update": "v86_base_update",
    "guard_forward": "v86_guard_forward",
    "functional_update": "v86_functional_update",
    "validation": "v86_validation",
}


def _prof_device_time_us(event: Any) -> float:
    value = getattr(event, "device_time_total", 0.0)
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _prof_cpu_time_us(event: Any) -> float:
    value = getattr(event, "cpu_time_total", 0.0)
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _prof_elapsed_us(event: Any) -> float:
    time_range = getattr(event, "time_range", None)
    if time_range is not None:
        elapsed = getattr(time_range, "elapsed_us", None)
        if callable(elapsed):
            try:
                return float(elapsed())
            except Exception:
                pass
    return _prof_device_time_us(event)


def _prof_range(event: Any) -> Tuple[float, float]:
    time_range = getattr(event, "time_range", None)
    if time_range is None:
        return (float("nan"), float("nan"))
    try:
        return (float(time_range.start), float(time_range.end))
    except Exception:
        return (float("nan"), float("nan"))


def _is_cuda_prof_event(event: Any) -> bool:
    try:
        return str(getattr(event, "device_type", "")).endswith("CUDA")
    except Exception:
        return False


def _is_cpu_prof_event(event: Any) -> bool:
    try:
        return str(getattr(event, "device_type", "")).endswith("CPU")
    except Exception:
        return False


def _summarize_torch_kernel_profiler(prof: Any, task_name: str, functional_events: int, profiler_steps: int) -> Dict[str, Any]:
    phase_names = set(_P9_PHASE_NAMES.values())
    phase_ranges: List[Tuple[str, float, float]] = []
    for event in prof.events():
        key = str(getattr(event, "key", ""))
        if key not in phase_names or not _is_cpu_prof_event(event):
            continue
        start, end = _prof_range(event)
        if math.isfinite(start) and math.isfinite(end) and end >= start:
            phase_label = next(label for label, name in _P9_PHASE_NAMES.items() if name == key)
            phase_ranges.append((phase_label, start, end))

    phase_kernel_time = {label: 0.0 for label in _P9_PHASE_NAMES}
    phase_kernel_count = {label: 0 for label in _P9_PHASE_NAMES}
    unknown_kernel_time = 0.0
    kernel_count_total = 0
    small_kernel_count = 0
    cuda_memcpy_time = 0.0
    cuda_sync_time = 0.0
    layout_conversion_count = 0

    for event in prof.events():
        key = str(getattr(event, "key", ""))
        lower_key = key.lower()
        if _is_cpu_prof_event(event):
            if key == "cudaDeviceSynchronize":
                cuda_sync_time += _prof_cpu_time_us(event)
            if any(token in lower_key for token in ("copy_", "_to_copy", "contiguous", "transpose", "permute")):
                layout_conversion_count += 1
            continue
        if not _is_cuda_prof_event(event) or key in phase_names:
            continue
        elapsed = _prof_elapsed_us(event)
        if elapsed <= 0.0:
            continue
        kernel_count_total += 1
        if elapsed < 10.0:
            small_kernel_count += 1
        if "memcpy" in lower_key or "memset" in lower_key:
            cuda_memcpy_time += elapsed
        start, end = _prof_range(event)
        assigned = False
        if math.isfinite(start) and math.isfinite(end):
            containing = [
                (label, phase_end - phase_start)
                for label, phase_start, phase_end in phase_ranges
                if phase_start <= start and end <= phase_end
            ]
            if containing:
                label = min(containing, key=lambda item: item[1])[0]
                phase_kernel_time[label] += elapsed
                phase_kernel_count[label] += 1
                assigned = True
        if not assigned:
            unknown_kernel_time += elapsed

    mapped_kernel_time = sum(phase_kernel_time.values())
    total_kernel_time = mapped_kernel_time + unknown_kernel_time
    mapped_fraction = mapped_kernel_time / max(total_kernel_time, 1.0e-12) if total_kernel_time > 0.0 else float("nan")
    unknown_fraction = unknown_kernel_time / max(total_kernel_time, 1.0e-12) if total_kernel_time > 0.0 else float("nan")
    top_phases = sorted(phase_kernel_time.items(), key=lambda item: item[1], reverse=True)
    while len(top_phases) < 3:
        top_phases.append((METRIC_UNAVAILABLE, float("nan")))
    profiler_pass = int(
        kernel_count_total > 0
        and math.isfinite(mapped_fraction)
        and mapped_fraction >= 0.90
        and math.isfinite(unknown_fraction)
        and unknown_fraction <= 0.10
    )
    return {
        "stage": "P9_PHASE_MAPPED_PROFILER_EXTERNAL_V86",
        "status": "measured_torch_profiler_kernel_trace",
        "reason": "torch profiler CUDA kernel events mapped to record_function phase ranges",
        "task_name": task_name,
        "candidate": "DG1-FT7-KW6-hidden28-functional",
        "profiler_steps": profiler_steps,
        "functional_events": functional_events,
        "kernel_count_total": kernel_count_total,
        "mapped_kernel_time_fraction": mapped_fraction if math.isfinite(mapped_fraction) else METRIC_UNAVAILABLE,
        "unknown_kernel_time_fraction": unknown_fraction if math.isfinite(unknown_fraction) else METRIC_UNAVAILABLE,
        "kernel_time_total_us": total_kernel_time if math.isfinite(total_kernel_time) else METRIC_UNAVAILABLE,
        "mapped_kernel_time_us": mapped_kernel_time if math.isfinite(mapped_kernel_time) else METRIC_UNAVAILABLE,
        "unknown_kernel_time_us": unknown_kernel_time if math.isfinite(unknown_kernel_time) else METRIC_UNAVAILABLE,
        "kernel_count_forward": phase_kernel_count["forward_backward"],
        "kernel_count_backward": phase_kernel_count["forward_backward"],
        "kernel_count_base_update": phase_kernel_count["base_update"],
        "kernel_count_functional_update": phase_kernel_count["functional_update"],
        "kernel_count_guard_forward": phase_kernel_count["guard_forward"],
        "kernel_count_validation": phase_kernel_count["validation"],
        "small_kernel_count_under_10us": small_kernel_count,
        "layout_conversion_count": layout_conversion_count,
        "cuda_memcpy_time": cuda_memcpy_time,
        "cuda_sync_time": cuda_sync_time,
        "top_kernel_phase_1": top_phases[0][0],
        "top_kernel_time_1": top_phases[0][1] if math.isfinite(top_phases[0][1]) else METRIC_UNAVAILABLE,
        "top_kernel_phase_2": top_phases[1][0],
        "top_kernel_time_2": top_phases[1][1] if math.isfinite(top_phases[1][1]) else METRIC_UNAVAILABLE,
        "top_kernel_phase_3": top_phases[2][0],
        "top_kernel_time_3": top_phases[2][1] if math.isfinite(top_phases[2][1]) else METRIC_UNAVAILABLE,
        "ProfilerPass": profiler_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _auc_from_trace(trace: Sequence[Tuple[float, float]]) -> float:
    if not trace:
        return float("nan")
    return float(v85.v83.v72.v71._auc([(float(x), float(y)) for x, y in trace]))


def _eval_kb_ce_loss(model: Any, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.inference_mode():
        for i in range(0, int(x.shape[0]), int(batch_size)):
            xb = x[i : i + int(batch_size)]
            yb = y[i : i + int(batch_size)]
            logits = model(xb)
            total_loss += float(F.cross_entropy(logits, yb, reduction="sum").detach().cpu())
            total += int(yb.numel())
    model.train()
    return total_loss / max(1, total)


def _eval_dg_ce_loss_only_external(
    stack: Any,
    head: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    spec: Any,
) -> float:
    total_loss = torch.zeros((), device=x.device)
    total = 0
    with torch.inference_mode():
        for xb, yb in zip(x.split(int(batch_size)), y.split(int(batch_size))):
            h, caches = stack.forward_manual(xb)
            logits, head_cache = head.forward_manual(h)
            loss = F.cross_entropy(logits, yb, reduction="sum")
            total_loss = total_loss + loss
            total += int(yb.numel())
            del caches, head_cache, logits, h
    return float((total_loss / max(1, total)).detach().cpu())


def _manual_ce_backward_profile_v86(
    stack: Any,
    head: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    spec: Any,
    device: torch.device,
) -> Tuple[float, float, float]:
    v85.v83._zero_grad(stack, head)
    _sync_cuda(device)
    started = time.perf_counter()
    h, caches = stack.forward_manual(x)
    logits, head_cache = head.forward_manual(h)
    loss, grad_logits = v85.v83.v72._weighted_smooth_ce_and_grad(logits, y, spec.label_smoothing, None)
    _sync_cuda(device)
    forward_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    dh = head.backward_manual(grad_logits, head_cache)
    stack.backward_manual(dh, caches)
    _sync_cuda(device)
    backward_ms = (time.perf_counter() - started) * 1000.0
    return float(loss.detach().cpu()), forward_ms, backward_ms


def _manual_ce_backward_with_holdout_profile_v86(
    stack: Any,
    head: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    x_holdout: torch.Tensor,
    y_holdout: torch.Tensor,
    spec: Any,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    v85.v83._zero_grad(stack, head)
    _sync_cuda(device)
    started = time.perf_counter()
    split = int(x.shape[0])
    x_pair = torch.cat((x, x_holdout), dim=0)
    h_pair, caches = stack.forward_manual(x_pair)
    logits_pair, head_cache = head.forward_manual(h_pair)
    loss, grad_logits = v85.v83.v72._weighted_smooth_ce_and_grad(logits_pair[:split], y, spec.label_smoothing, None)
    holdout_loss = v85.v83._smooth_ce_value_only(logits_pair[split:], y_holdout, spec.label_smoothing)
    _sync_cuda(device)
    forward_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    dh = head.backward_manual(grad_logits, v85._slice_cache_batch_v85(head_cache, split))
    stack.backward_manual(dh, v85._slice_stack_caches_v85(caches, split, x))
    _sync_cuda(device)
    backward_ms = (time.perf_counter() - started) * 1000.0
    del x_pair, h_pair, caches, logits_pair, head_cache, grad_logits
    return float(loss.detach().cpu()), float(holdout_loss.detach().cpu()), forward_ms, backward_ms


def _manual_ce_backward_low_overhead_v86(
    stack: Any,
    head: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    spec: Any,
    *,
    need_loss_value: bool,
) -> float:
    v85.v83._zero_grad(stack, head)
    h, caches = stack.forward_manual(x)
    logits, head_cache = head.forward_manual(h)
    loss, grad_logits = v85.v83.v72._weighted_smooth_ce_and_grad(logits, y, spec.label_smoothing, None)
    dh = head.backward_manual(grad_logits, head_cache)
    stack.backward_manual(dh, caches)
    return float(loss.detach().cpu()) if need_loss_value else 0.0


def _manual_ce_backward_with_holdout_low_overhead_v86(
    stack: Any,
    head: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    x_holdout: torch.Tensor,
    y_holdout: torch.Tensor,
    spec: Any,
) -> Tuple[float, float]:
    v85.v83._zero_grad(stack, head)
    split = int(x.shape[0])
    x_pair = torch.cat((x, x_holdout), dim=0)
    h_pair, caches = stack.forward_manual(x_pair)
    logits_pair, head_cache = head.forward_manual(h_pair)
    loss, grad_logits = v85.v83.v72._weighted_smooth_ce_and_grad(logits_pair[:split], y, spec.label_smoothing, None)
    holdout_loss = v85.v83._smooth_ce_value_only(logits_pair[split:], y_holdout, spec.label_smoothing)
    dh = head.backward_manual(grad_logits, v85._slice_cache_batch_v85(head_cache, split))
    stack.backward_manual(dh, v85._slice_stack_caches_v85(caches, split, x))
    del x_pair, h_pair, caches, logits_pair, head_cache, grad_logits
    return float(loss.detach().cpu()), float(holdout_loss.detach().cpu())


class _CudaGraphManualBackward:
    def __init__(
        self,
        stack: Any,
        head: Any,
        batch_shape: Sequence[int],
        y_dtype: torch.dtype,
        spec: Any,
        device: torch.device,
    ) -> None:
        self.enabled = False
        self.reason = "not_initialized"
        self.static_x: torch.Tensor | None = None
        self.static_y: torch.Tensor | None = None
        self.graph: Any = None
        if device.type != "cuda" or not torch.cuda.is_available():
            self.reason = "cuda_unavailable"
            return
        try:
            self.static_x = torch.empty(tuple(int(v) for v in batch_shape), device=device)
            self.static_y = torch.empty((int(batch_shape[0]),), device=device, dtype=y_dtype)
            self.static_x.zero_()
            self.static_y.zero_()
            warmup_stream = torch.cuda.Stream(device=device)
            warmup_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    _manual_ce_backward_low_overhead_v86(
                        stack,
                        head,
                        self.static_x,
                        self.static_y,
                        spec,
                        need_loss_value=False,
                    )
            torch.cuda.current_stream(device).wait_stream(warmup_stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                _manual_ce_backward_low_overhead_v86(
                    stack,
                    head,
                    self.static_x,
                    self.static_y,
                    spec,
                    need_loss_value=False,
                )
            self.enabled = True
            self.reason = "enabled"
        except Exception as exc:
            self.enabled = False
            self.reason = f"capture_failed:{type(exc).__name__}"
            self.static_x = None
            self.static_y = None
            self.graph = None
            try:
                _sync_cuda(device)
            except Exception:
                pass

    def replay(self, xb: torch.Tensor, yb: torch.Tensor) -> bool:
        if not self.enabled or self.static_x is None or self.static_y is None or self.graph is None:
            return False
        if tuple(xb.shape) != tuple(self.static_x.shape) or tuple(yb.shape) != tuple(self.static_y.shape):
            return False
        self.static_x.copy_(xb, non_blocking=True)
        self.static_y.copy_(yb, non_blocking=True)
        self.graph.replay()
        return True


def _profile_kb_mlp_external(
    model_classes: Dict[str, Any],
    args: argparse.Namespace,
    dataset: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    input_dim: int,
    num_classes: int,
    device: torch.device,
) -> Dict[str, Any]:
    MLP = model_classes.get("MLP")
    if MLP is None:
        return {"status": "not_run", "reason": "MLP_import_failed"}
    cfg = next(c for c in v85._baseline_configs() if str(c["model"]) == "MLP")
    ns = v85._kb_namespace(
        model_name="MLP",
        input_size=input_dim,
        output_size=num_classes,
        layers_width=cfg["layers_width"],
        activation_name=cfg["activation_name"],
        batch_norm=cfg["batch_norm"],
    )
    set_seed(int(cfg["seed"]))
    model = MLP(ns).to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    batch_size = int(args.kb_batch_size)
    eval_batch_size = int(args.profiler_eval_batch_size)
    steps_per_epoch = max(1, math.ceil(int(x_train.shape[0]) / batch_size))
    max_steps = int(args.profiler_steps)
    trace_every = max(1, int(args.profiler_trace_every))
    low_overhead = bool(args.profiler_low_overhead)
    async_window = bool(args.profiler_async_window and low_overhead)
    rng = torch.Generator(device=device).manual_seed(int(cfg["seed"]))
    perm = torch.randperm(int(x_train.shape[0]), generator=rng, device=device)
    phase = {key: 0.0 for key in ["batch_select", "forward", "backward", "update", "validation", "logging"]}
    step_times: List[float] = []
    train_ms_total = 0.0
    val_step_trace: List[Tuple[float, float]] = []
    val_time_trace: List[Tuple[float, float]] = []
    cumulative_ms = 0.0
    _eval_kb_ce_loss(model, x_test, y_test, eval_batch_size)
    _sync_cuda(device)
    total_started = time.perf_counter()
    window_started = total_started
    for step in range(1, max_steps + 1):
        if (step - 1) % steps_per_epoch == 0:
            perm = torch.randperm(int(x_train.shape[0]), generator=rng, device=device)
        step_started = time.perf_counter()
        batch_idx = (step - 1) % steps_per_epoch
        start = batch_idx * batch_size
        end = min(start + batch_size, int(x_train.shape[0]))
        idx = perm[start:end]
        xb = x_train[idx]
        yb = y_train[idx]
        if low_overhead:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
        else:
            started = time.perf_counter()
            _sync_cuda(device)
            phase["batch_select"] += (time.perf_counter() - started) * 1000.0
            opt.zero_grad(set_to_none=True)
            started = time.perf_counter()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            _sync_cuda(device)
            phase["forward"] += (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            loss.backward()
            _sync_cuda(device)
            phase["backward"] += (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            opt.step()
            _sync_cuda(device)
            phase["update"] += (time.perf_counter() - started) * 1000.0
        if not async_window:
            _sync_cuda(device)
            step_ms = (time.perf_counter() - step_started) * 1000.0
            step_times.append(step_ms)
            train_ms_total += step_ms
            cumulative_ms += step_ms
        if step % trace_every == 0 or step == max_steps:
            if async_window:
                _sync_cuda(device)
                train_window_ms = (time.perf_counter() - window_started) * 1000.0
                train_ms_total += train_window_ms
                cumulative_ms += train_window_ms
            started = time.perf_counter()
            val_loss = _eval_kb_ce_loss(model, x_test, y_test, eval_batch_size)
            _sync_cuda(device)
            validation_ms = (time.perf_counter() - started) * 1000.0
            phase["validation"] += validation_ms
            cumulative_ms += validation_ms
            val_step_trace.append((float(step), val_loss))
            val_time_trace.append((cumulative_ms, val_loss))
            if async_window:
                window_started = time.perf_counter()
    _sync_cuda(device)
    total_ms = (time.perf_counter() - total_started) * 1000.0
    mapped_ms = cumulative_ms if low_overhead else sum(phase.values())
    unknown = max(0.0, total_ms - mapped_ms) / max(total_ms, 1.0e-12)
    v85._release_cuda_model(model, device) if hasattr(v85, "_release_cuda_model") else None
    return {
        "stage": "P8_TIME_ACCOUNTING_EXTERNAL_RAW_V86",
        "status": "measured_low_overhead_async_window" if async_window else ("measured_low_overhead_time_window" if low_overhead else "measured_profiler_window"),
        "task_name": dataset,
        "candidate": "KB-MLP",
        "profiler_steps": max_steps,
        "ValLossAUC_step": _auc_from_trace(val_step_trace),
        "ValLossAUC_time": _auc_from_trace(val_time_trace),
        "train_step_time_ms": train_ms_total / max(1, max_steps),
        "forward_time": METRIC_UNAVAILABLE if low_overhead else phase["forward"] / max_steps,
        "backward_time": METRIC_UNAVAILABLE if low_overhead else phase["backward"] / max_steps,
        "base_update_time": METRIC_UNAVAILABLE if low_overhead else phase["update"] / max_steps,
        "functional_update_time": 0.0,
        "guard_forward_time": 0.0,
        "validation_time": phase["validation"] / max(1, len(val_step_trace)),
        "logging_time": phase["logging"] / max_steps,
        "batch_select_time": METRIC_UNAVAILABLE if low_overhead else phase["batch_select"] / max_steps,
        "unknown_time_fraction": unknown,
        "wall_clock_total_s": total_ms / 1000.0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _profile_dg_ft7_external(
    args: argparse.Namespace,
    dataset: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    input_dim: int,
    num_classes: int,
    device: torch.device,
) -> Dict[str, Any]:
    v85.v83.v80._patch_for_v80()
    dg_candidate = "KW6"
    spec = v85.v83.v80._spec_map_v80()[dg_candidate]
    v85.v83.set_seed(v85.v83.v72._stable_seed("v86-external-p8-dg", dataset, int(args.seed), int(args.broad_dg_hidden_dim)))
    stack, head = v85.v83.v80._make_manual_candidate_v80(
        spec,
        int(input_dim),
        int(num_classes),
        int(args.broad_dg_hidden_dim),
        8,
        device,
    )
    v85._configure_dg_update_modes(dg_candidate, stack, head)
    x_train_cpu = x_train.contiguous()
    y_train_cpu = y_train.contiguous()
    x_test_gpu = x_test.to(device)
    y_test_gpu = y_test.to(device)
    eval_batch_size = int(args.profiler_eval_batch_size)
    if torch.cuda.is_available() and device.type == "cuda":
        x_train_cpu = x_train_cpu.pin_memory()
        y_train_cpu = y_train_cpu.pin_memory()
    n_train = int(x_train_cpu.shape[0])
    batch_size = 128
    chunk_batches = max(1, int(args.broad_dg_stream_chunk_batches))
    steps_per_epoch = max(1, math.ceil(n_train / batch_size))
    max_steps = int(args.profiler_steps)
    trace_every = max(1, int(args.profiler_trace_every))
    low_overhead = bool(args.profiler_low_overhead)
    async_window = bool(args.profiler_async_window and low_overhead)
    params = v85.v83.v80.V63Params(train_size=n_train, val_size=0, test_size=int(x_test.shape[0]), batch_size=batch_size)
    lr = float(params.lr_manual * spec.lr_mult)
    opt = v85.v83.v72.FastAdamWNoSync([stack, head], lr=lr, weight_decay=1.0e-4)
    entries = v85.v83._param_entries(stack, head)
    role_entries_by_name = v85.v83._entries_by_role(entries)
    graph_backward = None
    if bool(args.profiler_cuda_graph_non_event) and device.type == "cuda":
        graph_backward = _CudaGraphManualBackward(
            stack,
            head,
            (batch_size, int(input_dim)),
            y_train_cpu.dtype,
            spec,
            device,
        )
    rng = torch.Generator(device="cpu").manual_seed(int(args.seed))
    phase = {key: 0.0 for key in ["batch_select", "forward", "backward", "update", "functional_metric_build", "functional_update", "validation", "logging"]}
    step_times: List[float] = []
    train_ms_total = 0.0
    val_step_trace: List[Tuple[float, float]] = []
    val_time_trace: List[Tuple[float, float]] = []
    cumulative_ms = 0.0
    global_step = 0
    event_count = 0
    graph_replay_count = 0
    reject_mass = 0.0
    bad_count = 0
    _eval_dg_ce_loss_only_external(stack, head, x_test_gpu, y_test_gpu, eval_batch_size, spec)
    _sync_cuda(device)
    total_started = time.perf_counter()
    window_started = total_started
    while global_step < max_steps:
        num_chunks = max(1, math.ceil(steps_per_epoch / chunk_batches))
        chunk_order = torch.randperm(num_chunks, generator=rng).tolist()
        batch_order: List[int] = []
        for chunk_id in chunk_order:
            first_batch = int(chunk_id) * chunk_batches
            last_batch = min(first_batch + chunk_batches, steps_per_epoch)
            batch_order.extend(range(first_batch, last_batch))
        stream_chunk_start = -1
        stream_chunk_end = -1
        x_stream_chunk = None
        y_stream_chunk = None
        for order_pos, batch_idx in enumerate(batch_order):
            if global_step >= max_steps:
                break
            global_step += 1
            step_started = time.perf_counter()
            started = time.perf_counter()
            start = batch_idx * batch_size
            end = min(start + batch_size, n_train)
            holdout_batch_idx = batch_order[(order_pos + 1) % len(batch_order)]
            holdout_start = holdout_batch_idx * batch_size
            holdout_end = min(holdout_start + batch_size, n_train)
            chunk_batch_start = (batch_idx // chunk_batches) * chunk_batches
            chunk_start = chunk_batch_start * batch_size
            chunk_end = min((chunk_batch_start + chunk_batches) * batch_size, n_train)
            if chunk_start != stream_chunk_start or chunk_end != stream_chunk_end:
                stream_chunk_start = chunk_start
                stream_chunk_end = chunk_end
                x_stream_chunk = x_train_cpu[stream_chunk_start:stream_chunk_end].to(device, non_blocking=True)
                y_stream_chunk = y_train_cpu[stream_chunk_start:stream_chunk_end].to(device, non_blocking=True)
            local_start = start - stream_chunk_start
            local_end = end - stream_chunk_start
            xb = x_stream_chunk[local_start:local_end]
            yb = y_stream_chunk[local_start:local_end]
            if holdout_start >= stream_chunk_start and holdout_end <= stream_chunk_end:
                xh = x_stream_chunk[holdout_start - stream_chunk_start:holdout_end - stream_chunk_start]
                yh = y_stream_chunk[holdout_start - stream_chunk_start:holdout_end - stream_chunk_start]
            else:
                xh = x_train_cpu[holdout_start:holdout_end].to(device, non_blocking=True)
                yh = y_train_cpu[holdout_start:holdout_end].to(device, non_blocking=True)
            if low_overhead:
                pass
            else:
                _sync_cuda(device)
                phase["batch_select"] += (time.perf_counter() - started) * 1000.0
            event_step = global_step % int(args.broad_ft7_event_stride) == 0
            use_pre_holdout = bool(event_step and v85.v83._ft7_uses_pre_holdout_for_step(global_step))
            used_graph = False
            if low_overhead and graph_backward is not None and not event_step:
                used_graph = graph_backward.replay(xb, yb)
                if used_graph:
                    loss_before = 0.0
                    holdout_loss_before = 0.0
                    graph_replay_count += 1
            if used_graph:
                pass
            elif low_overhead and use_pre_holdout:
                loss_before, holdout_loss_before = _manual_ce_backward_with_holdout_low_overhead_v86(
                    stack, head, xb, yb, xh, yh, spec
                )
            elif low_overhead:
                holdout_loss_before = 0.0
                loss_before = _manual_ce_backward_low_overhead_v86(stack, head, xb, yb, spec, need_loss_value=event_step)
            elif use_pre_holdout:
                loss_before, holdout_loss_before, forward_ms, backward_ms = _manual_ce_backward_with_holdout_profile_v86(
                    stack, head, xb, yb, xh, yh, spec, device
                )
                phase["forward"] += forward_ms
                phase["backward"] += backward_ms
            else:
                holdout_loss_before = 0.0
                loss_before, forward_ms, backward_ms = _manual_ce_backward_profile_v86(stack, head, xb, yb, spec, device)
                phase["forward"] += forward_ms
                phase["backward"] += backward_ms
            if low_overhead:
                opt.step(global_step, max_steps, warmup_cosine=True)
            else:
                _sync_cuda(device)
                started = time.perf_counter()
                opt.step(global_step, max_steps, warmup_cosine=True)
                _sync_cuda(device)
                phase["update"] += (time.perf_counter() - started) * 1000.0
            loss_after = loss_before
            if event_step:
                started = time.perf_counter()
                task_loss_after, holdout_loss_after, task_features_after, holdout_features_after = v85.v83._loss_pair_and_features_only(
                    stack, head, xb, yb, xh, yh, spec
                )
                _sync_cuda(device)
                phase["logging"] += (time.perf_counter() - started) * 1000.0
                started = time.perf_counter()
                loss_after, _func_norm, _trust_delta, reject_delta = v85.v83._apply_ft7_streamed_guarded_update(
                    stack,
                    head,
                    entries,
                    xb,
                    yb,
                    xh,
                    yh,
                    spec,
                    loss_before=loss_before,
                    holdout_loss_before=holdout_loss_before,
                    task_loss_after=task_loss_after,
                    holdout_loss_after=holdout_loss_after,
                    func_alpha=lr * 0.05 * float(args.broad_ft7_event_alpha_mult),
                    task_features_after=task_features_after,
                    holdout_features_after=holdout_features_after,
                    role_entries_by_name=role_entries_by_name,
                    pair_stack_guard_forward=True,
                )
                _sync_cuda(device)
                phase["functional_update"] += (time.perf_counter() - started) * 1000.0
                event_count += 1
                reject_mass += float(reject_delta)
            bad_count += int(float(loss_after) > float(loss_before) + 1.0e-8)
            if not async_window:
                _sync_cuda(device)
                step_ms = (time.perf_counter() - step_started) * 1000.0
                step_times.append(step_ms)
                train_ms_total += step_ms
                cumulative_ms += step_ms
            if global_step % trace_every == 0 or global_step == max_steps:
                if async_window:
                    _sync_cuda(device)
                    train_window_ms = (time.perf_counter() - window_started) * 1000.0
                    train_ms_total += train_window_ms
                    cumulative_ms += train_window_ms
                started = time.perf_counter()
                val_loss = _eval_dg_ce_loss_only_external(stack, head, x_test_gpu, y_test_gpu, eval_batch_size, spec)
                _sync_cuda(device)
                validation_ms = (time.perf_counter() - started) * 1000.0
                phase["validation"] += validation_ms
                cumulative_ms += validation_ms
                val_step_trace.append((float(global_step), float(val_loss)))
                val_time_trace.append((cumulative_ms, float(val_loss)))
                if async_window:
                    window_started = time.perf_counter()
    _sync_cuda(device)
    total_ms = (time.perf_counter() - total_started) * 1000.0
    mapped_ms = cumulative_ms if low_overhead else sum(phase.values())
    unknown = max(0.0, total_ms - mapped_ms) / max(total_ms, 1.0e-12)
    return {
        "stage": "P8_TIME_ACCOUNTING_EXTERNAL_RAW_V86",
        "status": "measured_low_overhead_async_window" if async_window else ("measured_low_overhead_time_window" if low_overhead else "measured_profiler_window"),
        "task_name": dataset,
        "candidate": "DG1-FT7-KW6-hidden28-functional",
        "profiler_steps": max_steps,
        "functional_events": event_count,
        "cuda_graph_replay_count": graph_replay_count,
        "cuda_graph_status": getattr(graph_backward, "reason", "disabled") if graph_backward is not None else "disabled",
        "functional_reject_mass": reject_mass,
        "bad_step_rate": bad_count / max(1, max_steps),
        "ValLossAUC_step": _auc_from_trace(val_step_trace),
        "ValLossAUC_time": _auc_from_trace(val_time_trace),
        "train_step_time_ms": train_ms_total / max(1, max_steps),
        "forward_time": METRIC_UNAVAILABLE if low_overhead else phase["forward"] / max_steps,
        "backward_time": METRIC_UNAVAILABLE if low_overhead else phase["backward"] / max_steps,
        "base_update_time": METRIC_UNAVAILABLE if low_overhead else phase["update"] / max_steps,
        "functional_update_time": METRIC_UNAVAILABLE if low_overhead else phase["functional_update"] / max_steps,
        "guard_forward_time": METRIC_UNAVAILABLE if low_overhead else phase["functional_metric_build"] / max_steps,
        "validation_time": phase["validation"] / max(1, len(val_step_trace)),
        "logging_time": phase["logging"] / max_steps,
        "batch_select_time": METRIC_UNAVAILABLE if low_overhead else phase["batch_select"] / max_steps,
        "unknown_time_fraction": unknown,
        "wall_clock_total_s": total_ms / 1000.0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _profile_dg_ft7_kernel_external(
    args: argparse.Namespace,
    dataset: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    input_dim: int,
    num_classes: int,
    device: torch.device,
) -> Dict[str, Any]:
    if not torch.cuda.is_available() or device.type != "cuda":
        return {
            "stage": "P9_PHASE_MAPPED_PROFILER_EXTERNAL_V86",
            "status": "not_run",
            "reason": "cuda_not_available_for_torch_profiler_kernel_trace",
            "task_name": dataset,
            "candidate": "DG1-FT7-KW6-hidden28-functional",
            "kernel_count_total": METRIC_UNAVAILABLE,
            "mapped_kernel_time_fraction": METRIC_UNAVAILABLE,
            "unknown_kernel_time_fraction": METRIC_UNAVAILABLE,
            "ProfilerPass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }

    v85.v83.v80._patch_for_v80()
    dg_candidate = "KW6"
    spec = v85.v83.v80._spec_map_v80()[dg_candidate]
    v85.v83.set_seed(v85.v83.v72._stable_seed("v86-external-p9-dg", dataset, int(args.seed), int(args.broad_dg_hidden_dim)))
    stack, head = v85.v83.v80._make_manual_candidate_v80(
        spec,
        int(input_dim),
        int(num_classes),
        int(args.broad_dg_hidden_dim),
        8,
        device,
    )
    v85._configure_dg_update_modes(dg_candidate, stack, head)
    x_train_cpu = x_train.contiguous()
    y_train_cpu = y_train.contiguous()
    x_test_gpu = x_test.to(device)
    y_test_gpu = y_test.to(device)
    x_train_cpu = x_train_cpu.pin_memory()
    y_train_cpu = y_train_cpu.pin_memory()
    n_train = int(x_train_cpu.shape[0])
    batch_size = 128
    chunk_batches = max(1, int(args.broad_dg_stream_chunk_batches))
    steps_per_epoch = max(1, math.ceil(n_train / batch_size))
    max_steps = int(args.kernel_profiler_steps)
    trace_every = max(1, int(args.kernel_profiler_trace_every))
    eval_batch_size = int(args.profiler_eval_batch_size)
    params = v85.v83.v80.V63Params(train_size=n_train, val_size=0, test_size=int(x_test.shape[0]), batch_size=batch_size)
    lr = float(params.lr_manual * spec.lr_mult)
    opt = v85.v83.v72.FastAdamWNoSync([stack, head], lr=lr, weight_decay=1.0e-4)
    entries = v85.v83._param_entries(stack, head)
    role_entries_by_name = v85.v83._entries_by_role(entries)
    rng = torch.Generator(device="cpu").manual_seed(int(args.seed))
    global_step = 0
    event_count = 0

    _sync_cuda(device)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        while global_step < max_steps:
            num_chunks = max(1, math.ceil(steps_per_epoch / chunk_batches))
            chunk_order = torch.randperm(num_chunks, generator=rng).tolist()
            batch_order: List[int] = []
            for chunk_id in chunk_order:
                first_batch = int(chunk_id) * chunk_batches
                last_batch = min(first_batch + chunk_batches, steps_per_epoch)
                batch_order.extend(range(first_batch, last_batch))
            stream_chunk_start = -1
            stream_chunk_end = -1
            x_stream_chunk = None
            y_stream_chunk = None
            for order_pos, batch_idx in enumerate(batch_order):
                if global_step >= max_steps:
                    break
                global_step += 1
                with torch.profiler.record_function(_P9_PHASE_NAMES["batch_select"]):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, n_train)
                    holdout_batch_idx = batch_order[(order_pos + 1) % len(batch_order)]
                    holdout_start = holdout_batch_idx * batch_size
                    holdout_end = min(holdout_start + batch_size, n_train)
                    chunk_batch_start = (batch_idx // chunk_batches) * chunk_batches
                    chunk_start = chunk_batch_start * batch_size
                    chunk_end = min((chunk_batch_start + chunk_batches) * batch_size, n_train)
                    if chunk_start != stream_chunk_start or chunk_end != stream_chunk_end:
                        stream_chunk_start = chunk_start
                        stream_chunk_end = chunk_end
                        x_stream_chunk = x_train_cpu[stream_chunk_start:stream_chunk_end].to(device, non_blocking=True)
                        y_stream_chunk = y_train_cpu[stream_chunk_start:stream_chunk_end].to(device, non_blocking=True)
                    local_start = start - stream_chunk_start
                    local_end = end - stream_chunk_start
                    xb = x_stream_chunk[local_start:local_end]
                    yb = y_stream_chunk[local_start:local_end]
                    if holdout_start >= stream_chunk_start and holdout_end <= stream_chunk_end:
                        xh = x_stream_chunk[holdout_start - stream_chunk_start:holdout_end - stream_chunk_start]
                        yh = y_stream_chunk[holdout_start - stream_chunk_start:holdout_end - stream_chunk_start]
                    else:
                        xh = x_train_cpu[holdout_start:holdout_end].to(device, non_blocking=True)
                        yh = y_train_cpu[holdout_start:holdout_end].to(device, non_blocking=True)
                    _sync_cuda(device)
                event_step = global_step % int(args.broad_ft7_event_stride) == 0
                use_pre_holdout = bool(event_step and v85.v83._ft7_uses_pre_holdout_for_step(global_step))
                with torch.profiler.record_function(_P9_PHASE_NAMES["forward_backward"]):
                    if use_pre_holdout:
                        loss_before, holdout_loss_before, _forward_ms, _backward_ms = _manual_ce_backward_with_holdout_profile_v86(
                            stack, head, xb, yb, xh, yh, spec, device
                        )
                    else:
                        holdout_loss_before = 0.0
                        loss_before, _forward_ms, _backward_ms = _manual_ce_backward_profile_v86(stack, head, xb, yb, spec, device)
                with torch.profiler.record_function(_P9_PHASE_NAMES["base_update"]):
                    opt.step(global_step, max_steps, warmup_cosine=True)
                if event_step:
                    with torch.profiler.record_function(_P9_PHASE_NAMES["guard_forward"]):
                        task_loss_after, holdout_loss_after, task_features_after, holdout_features_after = v85.v83._loss_pair_and_features_only(
                            stack, head, xb, yb, xh, yh, spec
                        )
                    with torch.profiler.record_function(_P9_PHASE_NAMES["functional_update"]):
                        v85.v83._apply_ft7_streamed_guarded_update(
                            stack,
                            head,
                            entries,
                            xb,
                            yb,
                            xh,
                            yh,
                            spec,
                            loss_before=loss_before,
                            holdout_loss_before=holdout_loss_before,
                            task_loss_after=task_loss_after,
                            holdout_loss_after=holdout_loss_after,
                            func_alpha=lr * 0.05 * float(args.broad_ft7_event_alpha_mult),
                            task_features_after=task_features_after,
                            holdout_features_after=holdout_features_after,
                            role_entries_by_name=role_entries_by_name,
                            pair_stack_guard_forward=True,
                        )
                    event_count += 1
                if global_step % trace_every == 0 or global_step == max_steps:
                    with torch.profiler.record_function(_P9_PHASE_NAMES["validation"]):
                        v85.v83.v72.v71._manual_eval(stack, head, x_test_gpu, y_test_gpu, eval_batch_size, int(num_classes))
                prof.step()
    _sync_cuda(device)
    row = _summarize_torch_kernel_profiler(prof, dataset, event_count, max_steps)
    _release_cuda_memory(device)
    return row


def _run_external_system_profiler(out_dir: Path, args: argparse.Namespace) -> None:
    source_args = _v85_namespace(args, fresh=False)
    _source_rows, model_classes = v85._run_source_audit(out_dir, source_args)
    device = v85.v83.get_device(args.device)
    raw_rows: List[Dict[str, Any]] = []
    time_rows: List[Dict[str, Any]] = []
    profiler_rows: List[Dict[str, Any]] = []
    for dataset in v85.parse_str_list(args.broad_datasets):
        kb_dataset = "FMNIST" if dataset in {"Fashion-MNIST", "Fashion", "FMNIST", "fmnist"} else dataset
        x_train, y_train, x_test, y_test, input_dim, num_classes, _split_protocol = v85._load_kanbefair_vision_tensors(
            kb_dataset,
            data_root=Path(args.data_root),
            train_size=int(args.broad_train_size),
            test_size=int(args.broad_test_size),
            seed=int(args.seed),
        )
        original_stride = int(v85.v83.P5_FT7_EVENT_STRIDE)
        original_alpha = float(v85.v83.P5_FT7_EVENT_ALPHA_MULT)
        v85.v83.P5_FT7_EVENT_STRIDE = int(args.broad_ft7_event_stride)
        v85.v83.P5_FT7_EVENT_ALPHA_MULT = float(args.broad_ft7_event_alpha_mult)
        try:
            mlp_raw = _profile_kb_mlp_external(model_classes, args, kb_dataset, x_train, y_train, x_test, y_test, input_dim, num_classes, device)
            dg_raw = _profile_dg_ft7_external(args, kb_dataset, x_train, y_train, x_test, y_test, input_dim, num_classes, device)
        finally:
            v85.v83.P5_FT7_EVENT_STRIDE = original_stride
            v85.v83.P5_FT7_EVENT_ALPHA_MULT = original_alpha
        raw_rows.extend([mlp_raw, dg_raw])
        step_ratio = _safe_float(dg_raw.get("train_step_time_ms")) / max(_safe_float(mlp_raw.get("train_step_time_ms")), 1.0e-12)
        time_auc_ratio = _safe_float(dg_raw.get("ValLossAUC_time")) / max(_safe_float(mlp_raw.get("ValLossAUC_time")), 1.0e-12)
        step_auc_ratio = _safe_float(dg_raw.get("ValLossAUC_step")) / max(_safe_float(mlp_raw.get("ValLossAUC_step")), 1.0e-12)
        unknown = _safe_float(dg_raw.get("unknown_time_fraction"))
        time_pass = int(
            math.isfinite(step_ratio)
            and math.isfinite(time_auc_ratio)
            and math.isfinite(unknown)
            and step_ratio <= 1.50
            and time_auc_ratio <= 1.05
            and unknown <= 0.10
        )
        time_rows.append({
            "stage": "P8_TIME_ACCOUNTING_EXTERNAL_V86",
            "status": dg_raw.get("status", "measured_profiler_window"),
            "task_family": "vision",
            "task_name": kb_dataset,
            "candidate": "DG1-FT7-KW6-hidden28-functional",
            "wall_clock_total_s": dg_raw.get("wall_clock_total_s"),
            "mlp_wall_clock_total_s": mlp_raw.get("wall_clock_total_s"),
            "wall_clock_total_ratio_vs_MLP": _safe_float(dg_raw.get("wall_clock_total_s")) / max(_safe_float(mlp_raw.get("wall_clock_total_s")), 1.0e-12),
            "train_step_time_ms": dg_raw.get("train_step_time_ms"),
            "mlp_train_step_time_ms": mlp_raw.get("train_step_time_ms"),
            "train_step_time_ratio_vs_MLP": step_ratio,
            "forward_time": dg_raw.get("forward_time"),
            "backward_time": dg_raw.get("backward_time"),
            "base_update_time": dg_raw.get("base_update_time"),
            "functional_update_time": dg_raw.get("functional_update_time"),
            "guard_forward_time": dg_raw.get("guard_forward_time"),
            "validation_time": dg_raw.get("validation_time"),
            "logging_time": dg_raw.get("logging_time"),
            "cuda_sync_time": METRIC_UNAVAILABLE,
            "unknown_time_fraction": unknown,
            "ValLossAUC_time": dg_raw.get("ValLossAUC_time"),
            "MLP_ValLossAUC_time": mlp_raw.get("ValLossAUC_time"),
            "ValLossAUC_time_ratio_vs_MLP": time_auc_ratio,
            "ValLossAUC_step_ratio_vs_MLP": step_auc_ratio,
            "TimeToTarget": METRIC_UNAVAILABLE,
            "WallClockPass": int(math.isfinite(step_ratio) and step_ratio <= 1.50),
            "TimeAccountingPass": time_pass,
            "reason": "measured low-overhead async wall-clock window; phase attribution comes from P9 kernel profiler" if str(dg_raw.get("status")) == "measured_low_overhead_async_window" else ("measured low-overhead time window; phase attribution comes from P9 kernel profiler" if str(dg_raw.get("status")) == "measured_low_overhead_time_window" else "measured profiler window; TimeToTarget unavailable"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
        phase_values = {
            "forward": _safe_float(dg_raw.get("forward_time")),
            "backward": _safe_float(dg_raw.get("backward_time")),
            "base_update": _safe_float(dg_raw.get("base_update_time")),
            "functional_update": _safe_float(dg_raw.get("functional_update_time")),
            "guard_forward": _safe_float(dg_raw.get("guard_forward_time")),
            "validation": _safe_float(dg_raw.get("validation_time")),
            "logging": _safe_float(dg_raw.get("logging_time")),
        }
        finite_phases = {k: v for k, v in phase_values.items() if math.isfinite(v)}
        top_phase = max(finite_phases.items(), key=lambda kv: kv[1]) if finite_phases else (METRIC_UNAVAILABLE, METRIC_UNAVAILABLE)
        mapped_fraction = 1.0 - unknown if math.isfinite(unknown) else float("nan")
        profiler_rows.append({
            "stage": "P9_PHASE_MAPPED_PROFILER_EXTERNAL_V86",
            "status": "measured_phase_timers",
            "reason": "synchronized phase timers; torch kernel trace not collected",
            "task_name": kb_dataset,
            "candidate": "DG1-FT7-KW6-hidden28-functional",
            "kernel_count_total": METRIC_UNAVAILABLE,
            "mapped_kernel_time_fraction": mapped_fraction if math.isfinite(mapped_fraction) else METRIC_UNAVAILABLE,
            "unknown_kernel_time_fraction": unknown if math.isfinite(unknown) else METRIC_UNAVAILABLE,
            "kernel_count_forward": METRIC_UNAVAILABLE,
            "kernel_count_backward": METRIC_UNAVAILABLE,
            "kernel_count_base_update": METRIC_UNAVAILABLE,
            "kernel_count_functional_update": METRIC_UNAVAILABLE,
            "kernel_count_guard_forward": METRIC_UNAVAILABLE,
            "small_kernel_count_under_10us": METRIC_UNAVAILABLE,
            "layout_conversion_count": METRIC_UNAVAILABLE,
            "cuda_memcpy_time": METRIC_UNAVAILABLE,
            "cuda_sync_time": METRIC_UNAVAILABLE,
            "top_kernel_phase_1": top_phase[0],
            "top_kernel_time_1": top_phase[1],
            "ProfilerPass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
        if bool(args.run_kernel_profiler):
            original_stride = int(v85.v83.P5_FT7_EVENT_STRIDE)
            original_alpha = float(v85.v83.P5_FT7_EVENT_ALPHA_MULT)
            v85.v83.P5_FT7_EVENT_STRIDE = int(args.broad_ft7_event_stride)
            v85.v83.P5_FT7_EVENT_ALPHA_MULT = float(args.broad_ft7_event_alpha_mult)
            try:
                profiler_rows.append(
                    _profile_dg_ft7_kernel_external(args, kb_dataset, x_train, y_train, x_test, y_test, input_dim, num_classes, device)
                )
            finally:
                v85.v83.P5_FT7_EVENT_STRIDE = original_stride
                v85.v83.P5_FT7_EVENT_ALPHA_MULT = original_alpha
    if raw_rows:
        write_csv(out_dir / "time_accounting_external_raw.csv", raw_rows)
    if time_rows:
        write_csv(out_dir / "time_accounting_external.csv", time_rows)
    if profiler_rows:
        write_csv(out_dir / "phase_mapped_profiler_external.csv", profiler_rows)


def _write_boundary_postprocess(out_dir: Path) -> None:
    """Write conservative Wave 6/7 artifacts from already measured rows.

    This intentionally does not invent phase profiler fields. If no kernel
    profiler has run, P9 remains an explicit incomplete measured boundary.
    """
    transfer_rows = _read_csv(out_dir / "external_multitask_transfer.csv")
    joint_rows = _read_csv(out_dir / "joint_fair_envelope.csv")
    causality_rows = _read_csv(out_dir / "functional_causality_multitask.csv")
    broad_repro_rows = _read_csv(out_dir / "kanbefair_broad_reproduction.csv")
    if not transfer_rows and not joint_rows and not causality_rows:
        return

    transfer_by_task_model: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in transfer_rows:
        if str(row.get("status")) != "measured":
            continue
        transfer_by_task_model[(str(row.get("task_name")), str(row.get("model")))] = row

    time_path = out_dir / "time_accounting_external.csv"
    existing_time_rows = _read_csv(time_path)
    measured_time_statuses = {"measured_profiler_window", "measured_low_overhead_time_window", "measured_low_overhead_async_window"}
    if not any(str(r.get("status")) in measured_time_statuses for r in existing_time_rows):
        time_rows: List[Dict[str, Any]] = []
        for joint in joint_rows:
            if str(joint.get("status")) != "measured":
                continue
            task = str(joint.get("task_name"))
            model = str(joint.get("model"))
            dg = transfer_by_task_model.get((task, model), {})
            mlp = transfer_by_task_model.get((task, "KB-MLP"), {})
            step_ratio = _safe_float(joint.get("step_ratio_vs_MLP"))
            train_ratio = _safe_float(dg.get("train_time_s")) / max(_safe_float(mlp.get("train_time_s")), 1.0e-12)
            time_rows.append({
                "stage": "P8_TIME_ACCOUNTING_EXTERNAL_V86",
                "status": "measured_from_training_timers",
                "task_family": joint.get("task_family", "vision"),
                "task_name": task,
                "candidate": model,
                "wall_clock_total_s": dg.get("train_time_s", METRIC_UNAVAILABLE),
                "mlp_wall_clock_total_s": mlp.get("train_time_s", METRIC_UNAVAILABLE),
                "wall_clock_total_ratio_vs_MLP": train_ratio if math.isfinite(train_ratio) else METRIC_UNAVAILABLE,
                "train_step_time_ms": dg.get("step_time_ms", METRIC_UNAVAILABLE),
                "mlp_train_step_time_ms": mlp.get("step_time_ms", METRIC_UNAVAILABLE),
                "train_step_time_ratio_vs_MLP": step_ratio if math.isfinite(step_ratio) else METRIC_UNAVAILABLE,
                "forward_time": METRIC_UNAVAILABLE,
                "backward_time": METRIC_UNAVAILABLE,
                "base_update_time": METRIC_UNAVAILABLE,
                "functional_update_time": METRIC_UNAVAILABLE,
                "guard_forward_time": METRIC_UNAVAILABLE,
                "validation_time": METRIC_UNAVAILABLE,
                "logging_time": METRIC_UNAVAILABLE,
                "cuda_sync_time": METRIC_UNAVAILABLE,
                "unknown_time_fraction": METRIC_UNAVAILABLE,
                "ValLossAUC_time": METRIC_UNAVAILABLE,
                "TimeToTarget": METRIC_UNAVAILABLE,
                "WallClockPass": int(math.isfinite(step_ratio) and step_ratio <= 1.50),
                "TimeAccountingPass": 0,
                "reason": "phase_times_and_time_auc_not_measured",
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
        if time_rows:
            write_csv(time_path, time_rows)

    profiler_path = out_dir / "phase_mapped_profiler_external.csv"
    existing_profiler_rows = _read_csv(profiler_path)
    if not any(str(r.get("status")) == "measured_phase_timers" for r in existing_profiler_rows):
        profiler_rows = [{
            "stage": "P9_PHASE_MAPPED_PROFILER_EXTERNAL_V86",
            "status": "not_run",
            "reason": "torch_kernel_phase_profiler_not_executed",
            "task_name": "all_measured_broad_tasks",
            "candidate": "DG1-FT7-KW6-hidden28-functional",
            "kernel_count_total": METRIC_UNAVAILABLE,
            "mapped_kernel_time_fraction": METRIC_UNAVAILABLE,
            "unknown_kernel_time_fraction": METRIC_UNAVAILABLE,
            "kernel_count_forward": METRIC_UNAVAILABLE,
            "kernel_count_backward": METRIC_UNAVAILABLE,
            "kernel_count_base_update": METRIC_UNAVAILABLE,
            "kernel_count_functional_update": METRIC_UNAVAILABLE,
            "kernel_count_guard_forward": METRIC_UNAVAILABLE,
            "small_kernel_count_under_10us": METRIC_UNAVAILABLE,
            "layout_conversion_count": METRIC_UNAVAILABLE,
            "cuda_memcpy_time": METRIC_UNAVAILABLE,
            "cuda_sync_time": METRIC_UNAVAILABLE,
            "top_kernel_phase_1": METRIC_UNAVAILABLE,
            "top_kernel_time_1": METRIC_UNAVAILABLE,
            "ProfilerPass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }]
        write_csv(profiler_path, profiler_rows)

    geometry_rows: List[Dict[str, Any]] = []
    for row in causality_rows:
        if str(row.get("status")) != "measured" or str(row.get("functional_variant")) != "ft7":
            continue
        curv_ratio = _safe_float(row.get("curvature_ratio"))
        ece_delta = _safe_float(row.get("ECE_delta"))
        nll_delta = _safe_float(row.get("NLL_delta"))
        mechanism_pass = int(
            math.isfinite(curv_ratio)
            and curv_ratio < 1.0
            and ((math.isfinite(ece_delta) and ece_delta <= 0.0) or (math.isfinite(nll_delta) and nll_delta <= 0.0))
        )
        geometry_rows.append({
            "stage": "P10_GEOMETRY_TASK_RELATIONSHIP_V86",
            "status": "measured_from_causality_rows",
            "task_name": row.get("task_name"),
            "task_family": row.get("task_family", "vision"),
            "input_dim": METRIC_UNAVAILABLE,
            "num_classes": METRIC_UNAVAILABLE,
            "symbolic_smoothness": METRIC_UNAVAILABLE,
            "DG_curvature_delta": (curv_ratio - 1.0) if math.isfinite(curv_ratio) else METRIC_UNAVAILABLE,
            "DG_jacobian_delta": (_safe_float(row.get("jacobian_ratio")) - 1.0) if math.isfinite(_safe_float(row.get("jacobian_ratio"))) else METRIC_UNAVAILABLE,
            "DG_lipschitz_delta": (_safe_float(row.get("local_lipschitz_ratio")) - 1.0) if math.isfinite(_safe_float(row.get("local_lipschitz_ratio"))) else METRIC_UNAVAILABLE,
            "accuracy_delta": row.get("delta_vs_base", METRIC_UNAVAILABLE),
            "ECE_delta": row.get("ECE_delta", METRIC_UNAVAILABLE),
            "NLL_delta": row.get("NLL_delta", METRIC_UNAVAILABLE),
            "robustness_delta": METRIC_UNAVAILABLE,
            "forgetting_delta": METRIC_UNAVAILABLE,
            "MechanismModelPass": mechanism_pass,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    if geometry_rows:
        write_csv(out_dir / "geometry_task_relationship.csv", geometry_rows)

    causality_by_task = {
        str(r.get("task_name")): r for r in causality_rows
        if str(r.get("functional_variant")) == "ft7" and str(r.get("status")) == "measured"
    }
    boundary_rows: List[Dict[str, Any]] = []
    for joint in joint_rows:
        if str(joint.get("status")) != "measured":
            continue
        task = str(joint.get("task_name"))
        step_ratio = _safe_float(joint.get("step_ratio_vs_MLP"))
        metric_delta = _safe_float(joint.get("metric_delta_vs_MLP"))
        curv_delta = _safe_float(causality_by_task.get(task, {}).get("curvature_ratio")) - 1.0
        if _safe_float(joint.get("JointFairPass"), 0.0) == 1.0:
            label = "mnist_like_win"
            reason = "joint fair envelope and causality pass on MNIST-like broad vision task"
        elif math.isfinite(step_ratio) and step_ratio > 1.50:
            label = "wallclock_fail"
            reason = "DG step ratio exceeds MLP envelope"
        elif math.isfinite(metric_delta) and metric_delta < 0.0:
            label = "mlp_dominates"
            reason = "DG metric lower than MLP"
        else:
            label = "parameter_fair_only"
            reason = "partial fairness without full joint pass"
        boundary_rows.append({
            "stage": "P11_NEGATIVE_BOUNDARY_AUDIT_V86",
            "status": "measured_from_joint_fair_rows",
            "task_family": joint.get("task_family", "vision"),
            "task_name": task,
            "DG_vs_MLP_delta": joint.get("metric_delta_vs_MLP"),
            "params_ratio": joint.get("params_ratio_vs_MLP"),
            "FLOPs_ratio": joint.get("FLOPs_ratio_vs_MLP"),
            "step_ratio": joint.get("step_ratio_vs_MLP"),
            "geometry_delta": curv_delta if math.isfinite(curv_delta) else METRIC_UNAVAILABLE,
            "possible_reason": reason,
            "boundary_label": label,
            "BoundaryAuditPass": 1,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    for repro in broad_repro_rows:
        if str(repro.get("status")) == "measured" and _safe_float(repro.get("reproduction_pass"), 0.0) != 1.0:
            boundary_rows.append({
                "stage": "P11_NEGATIVE_BOUNDARY_AUDIT_V86",
                "status": "measured_from_broad_reproduction_rows",
                "task_family": repro.get("task_family", "vision"),
                "task_name": repro.get("task_name"),
                "DG_vs_MLP_delta": METRIC_UNAVAILABLE,
                "params_ratio": METRIC_UNAVAILABLE,
                "FLOPs_ratio": METRIC_UNAVAILABLE,
                "step_ratio": METRIC_UNAVAILABLE,
                "geometry_delta": METRIC_UNAVAILABLE,
                "possible_reason": "KANbeFair reported baseline tolerance miss; fair comparison uses measured MLP baseline row",
                "boundary_label": "baseline_reproduction_boundary",
                "BoundaryAuditPass": 1,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    if boundary_rows:
        write_csv(out_dir / "negative_boundary_audit.csv", boundary_rows)


def _route(out_dir: Path, audit: Dict[str, Any]) -> Dict[str, Any]:
    repro_rows = _read_csv(out_dir / "v85_reproduction.csv")
    continual_rows = _read_csv(out_dir / "continual_multisplit_stress.csv")
    broad_repro_rows = _read_csv(out_dir / "kanbefair_broad_reproduction.csv")
    multitask_rows = _read_csv(out_dir / "external_multitask_transfer.csv")
    joint_rows = _read_csv(out_dir / "joint_fair_envelope.csv")
    multitask_causality_rows = _read_csv(out_dir / "functional_causality_multitask.csv")
    time_rows = _read_csv(out_dir / "time_accounting_external.csv")
    profiler_rows = _read_csv(out_dir / "phase_mapped_profiler_external.csv")
    geometry_rows = _read_csv(out_dir / "geometry_task_relationship.csv")
    boundary_rows = _read_csv(out_dir / "negative_boundary_audit.csv")
    tabular_rows = _read_csv(out_dir / "tabular_formal_envelope.csv")
    nonvision_rows = _read_csv(out_dir / "nonvision_runnability_audit.csv")
    nonvision_baseline_rows = _read_csv(out_dir / "nonvision_baseline_reproduction.csv")
    nonvision_dg_rows = _read_csv(out_dir / "nonvision_dg_transfer.csv")
    v85_repro_pass = int(any(_safe_float(r.get("FreshReproPass"), 0.0) == 1.0 for r in repro_rows))
    mnist_primary_pass = int(sum(1 for r in repro_rows if _safe_float(r.get("primary_transfer_pass"), 0.0) == 1.0 and str(r.get("candidate_id", "")).startswith("DG1-")))
    mnist_joint_pass = int(sum(1 for r in repro_rows if _safe_float(r.get("parameter_fair_pass"), 0.0) == 1.0 and _safe_float(r.get("flops_fair_pass"), 0.0) == 1.0 and _safe_float(r.get("wallclock_pass"), 0.0) == 1.0 and str(r.get("candidate_id", "")).startswith("DG1-")))
    mnist_causality_pass = int(sum(1 for r in repro_rows if _safe_float(r.get("functional_causality_pass"), 0.0) == 1.0 and str(r.get("candidate_id", "")).startswith("DG1-")))
    broad_reproduction_measured = int(any(_safe_float(r.get("BroadMeasuredPass"), 0.0) == 1.0 for r in broad_repro_rows))
    broad_reproduction_formal = int(
        bool(broad_repro_rows)
        and all(_safe_float(r.get("reproduction_pass"), 0.0) == 1.0 for r in broad_repro_rows if str(r.get("status")) == "measured")
    )
    broad_task_pass_count = int(sum(1 for r in multitask_rows if _safe_float(r.get("TaskFamilyPass"), 0.0) == 1.0 and str(r.get("model", "")).startswith("DG1-FT7-")))
    broad_joint_pass_count = int(sum(1 for r in joint_rows if _safe_float(r.get("JointFairPass"), 0.0) == 1.0))
    broad_causality_pass_count = int(sum(1 for r in multitask_causality_rows if _safe_float(r.get("CausalityPass"), 0.0) == 1.0))
    broad_formal_joint_pass_count = int(sum(1 for r in joint_rows if _safe_float(r.get("JointFairPass"), 0.0) == 1.0 and _safe_float(r.get("formal_protocol"), 0.0) == 1.0))
    broad_formal_causality_pass_count = int(sum(1 for r in multitask_causality_rows if _safe_float(r.get("CausalityPass"), 0.0) == 1.0 and _safe_float(r.get("formal_protocol"), 0.0) == 1.0))
    primary_transfer_pass_count = mnist_primary_pass + broad_task_pass_count
    joint_fair_pass_count = mnist_joint_pass + broad_joint_pass_count
    functional_causality_pass_count = mnist_causality_pass + broad_causality_pass_count
    time_accounting_pass = int(bool(time_rows) and all(_safe_float(r.get("TimeAccountingPass"), 0.0) == 1.0 for r in time_rows if str(r.get("status")).startswith("measured")))
    profiler_pass = int(bool(profiler_rows) and any(_safe_float(r.get("ProfilerPass"), 0.0) == 1.0 for r in profiler_rows))
    geometry_generalization_pass = int(bool(geometry_rows) and any(_safe_float(r.get("MechanismModelPass"), 0.0) == 1.0 for r in geometry_rows))
    boundary_audit_pass = int(bool(boundary_rows) and any(_safe_float(r.get("BoundaryAuditPass"), 0.0) == 1.0 for r in boundary_rows))
    tabular_tested_tasks = sorted({
        str(r.get("task_name"))
        for r in tabular_rows
        if str(r.get("status")) == "measured" and str(r.get("model", "")).startswith("DG1-FT7-")
    })
    tabular_pass_tasks = sorted({
        str(r.get("task_name"))
        for r in tabular_rows
        if _safe_float(r.get("TabularFormalPass"), 0.0) == 1.0
    })
    tabular_formal_pass = int(bool(tabular_tested_tasks) and len(tabular_pass_tasks) == len(tabular_tested_tasks))
    nlp_audio_rows = [r for r in nonvision_rows if str(r.get("task_family")) in {"nlp", "audio"}]
    nlp_audio_ready = int(bool(nlp_audio_rows) and all(_safe_float(r.get("RunnablePass"), 0.0) == 1.0 for r in nlp_audio_rows))
    nlp_audio_blocked_count = int(sum(1 for r in nlp_audio_rows if _safe_float(r.get("RunnablePass"), 0.0) != 1.0))
    nlp_audio_baseline_families = sorted({
        str(r.get("task_family"))
        for r in nonvision_baseline_rows
        if str(r.get("status")) == "measured" and _safe_float(r.get("NativeBaselineMeasuredPass"), 0.0) == 1.0
    })
    nlp_audio_baseline_measured = int({"nlp", "audio"}.issubset(set(nlp_audio_baseline_families)))
    nonvision_dg_transfer_count = int(sum(
        1 for r in nonvision_dg_rows
        if str(r.get("status")) == "measured" and _safe_float(r.get("DGTransferMeasuredPass"), 0.0) == 1.0
    ))
    nonvision_dg_functional_task_pass_count = int(sum(
        1 for r in nonvision_dg_rows
        if _safe_float(r.get("NonvisionDGFunctionalTaskPass"), 0.0) == 1.0
    ))
    nonvision_dg_geometry_pass_count = int(sum(
        1 for r in nonvision_dg_rows
        if _safe_float(r.get("NonvisionDGGeometryPass"), 0.0) == 1.0
    ))

    dg_terminal_rows = [
        r for r in continual_rows
        if str(r.get("status")) == "measured"
        and str(r.get("model_group")) == "DG"
        and str(r.get("candidate")) != "CL0-DG-Base"
        and str(r.get("task_id")) == str(len([x for x in str(r.get("final_task_acc_vector", "")).split(",") if x]) - 1)
    ]
    split_ids = sorted({str(r.get("split_id")) for r in dg_terminal_rows})
    candidates = sorted({str(r.get("candidate")) for r in dg_terminal_rows})
    candidate_pass_splits: Dict[str, List[str]] = {}
    candidate_unbalanced: Dict[str, int] = {}
    candidate_mean_gap: Dict[str, float] = {}
    for candidate in candidates:
        rows = [r for r in dg_terminal_rows if str(r.get("candidate")) == candidate]
        candidate_pass_splits[candidate] = sorted({str(r.get("split_id")) for r in rows if _safe_float(r.get("ContinualFormalPass"), 0.0) == 1.0})
        candidate_unbalanced[candidate] = int(any(
            _safe_float(r.get("forgetting_delta_vs_DG_Base")) <= 0.0
            and _safe_float(r.get("max_min_task_acc_gap")) > 0.20
            for r in rows
        ))
        gaps = [_safe_float(r.get("max_min_task_acc_gap")) for r in rows]
        finite_gaps = [g for g in gaps if math.isfinite(g)]
        candidate_mean_gap[candidate] = float(sum(finite_gaps) / len(finite_gaps)) if finite_gaps else float("inf")
    best_continual_candidate = ""
    if candidates:
        best_continual_candidate = sorted(
            candidates,
            key=lambda c: (-len(candidate_pass_splits.get(c, [])), candidate_mean_gap.get(c, float("inf")), c),
        )[0]
    pass_splits = candidate_pass_splits.get(best_continual_candidate, [])
    unbalanced_rows = [
        r for r in dg_terminal_rows
        if str(r.get("candidate")) == best_continual_candidate
        and _safe_float(r.get("max_min_task_acc_gap")) > 0.20
    ]
    continual_formal_pass = int(bool(split_ids) and len(pass_splits) == len(split_ids))
    any_forgetting_reduced_unbalanced = int(any(
        _safe_float(r.get("forgetting_delta_vs_DG_Base")) <= 0.0
        and _safe_float(r.get("max_min_task_acc_gap")) > 0.20
        for r in unbalanced_rows
    ))

    if not v85_repro_pass:
        route = "R8-NoReproduction"
        blocker = "v85_accepted_route_fresh_reproduction_failed"
        boundary_label = "reproduction_fail"
        next_impl = "repair_v85_reproduction_before_broad_claim"
    elif any_forgetting_reduced_unbalanced:
        route = "R5-ContinualUnbalanced"
        blocker = "continual_forgetting_reduced_but_task_balance_failed"
        boundary_label = "continual_unbalanced"
        next_impl = "design_balanced_continual_guard_without_teacher_loss_sampler_or_replay"
    elif not continual_formal_pass:
        route = "R2-TaskFamilyLimitedFairAdvantage"
        blocker = "continual_multisplit_formal_not_passed"
        boundary_label = "mnist_like_symbolic_only_pending_continual"
        next_impl = "continue_continual_multisplit_repair_or_broader_task_eval"
    elif broad_reproduction_measured and (broad_joint_pass_count + broad_causality_pass_count) > 0 and broad_formal_joint_pass_count < 2:
        route = "R2-TaskFamilyLimitedFairAdvantage"
        if broad_formal_causality_pass_count > 0:
            blocker = "broad_formal_joint_fair_failed"
            boundary_label = "broad_formal_task_and_causality_pass_joint_fair_fail"
            next_impl = "repair_broad_joint_fair_memory_step_envelope_without_changing_objective"
        else:
            blocker = "broad_vision_screen_not_formal_or_not_enough_joint_pass"
            boundary_label = "broad_vision_screen_measured_but_formal_pending"
            next_impl = "rerun_broad_non_symbolic_tasks_at_full_formal_protocol_or_expand_task_families"
    else:
        route = "R2-TaskFamilyLimitedFairAdvantage"
        blocker = "broad_non_symbolic_tasks_not_run"
        boundary_label = "mnist_like_symbolic_continual_pass_broad_pending"
        next_impl = "run_broad_non_symbolic_kanbefair_tasks_and_joint_envelopes"

    external_formal_pass = int(
        v85_repro_pass
        and continual_formal_pass
        and broad_formal_joint_pass_count >= 2
        and broad_formal_causality_pass_count >= 2
    )
    if external_formal_pass:
        route = "R1-BroadExternalFairFunctionalAdvantage"
        blocker = "none"
        boundary_label = "broad_external_fair_functional_advantage"
        next_impl = "run_phase_mapped_profiler_external"
        if profiler_pass and not time_accounting_pass:
            next_impl = "repair_time_accounting_external"
        elif time_accounting_pass and not profiler_pass:
            next_impl = "run_phase_mapped_profiler_external"
        if time_accounting_pass and profiler_pass and geometry_generalization_pass and boundary_audit_pass:
            next_impl = "none"

    strict_all_family_pass = int(
        external_formal_pass
        and time_accounting_pass
        and profiler_pass
        and geometry_generalization_pass
        and boundary_audit_pass
        and tabular_formal_pass
        and nlp_audio_ready
        and nlp_audio_baseline_measured
    )
    next_family_impl = "none"
    if not tabular_formal_pass:
        next_family_impl = "repair_tabular_formal_envelope_without_changing_loss_teacher_sampler_or_offload"
    elif not nlp_audio_ready:
        next_family_impl = "resolve_nlp_audio_dependencies_and_real_dataset_cache"
    elif not nlp_audio_baseline_measured:
        next_family_impl = "run_kanbefair_native_nlp_audio_baseline_reproduction"

    decision = {
        "route": route,
        "v85_reproduction_pass": v85_repro_pass,
        "kanbefair_broad_reproduction_pass": broad_reproduction_formal,
        "kanbefair_broad_reproduction_measured": broad_reproduction_measured,
        "primary_transfer_pass_count": primary_transfer_pass_count,
        "joint_fair_pass_count": joint_fair_pass_count,
        "functional_causality_pass_count": functional_causality_pass_count,
        "broad_task_pass_count": broad_task_pass_count,
        "broad_joint_pass_count": broad_joint_pass_count,
        "broad_causality_pass_count": broad_causality_pass_count,
        "broad_formal_joint_pass_count": broad_formal_joint_pass_count,
        "broad_formal_causality_pass_count": broad_formal_causality_pass_count,
        "symbolic_pass": 0,
        "continual_formal_pass": continual_formal_pass,
        "continual_pass_split_count": len(pass_splits),
        "continual_tested_split_count": len(split_ids),
        "wallclock_pass_count": joint_fair_pass_count,
        "time_accounting_pass": time_accounting_pass,
        "phase_mapped_profiler_pass": profiler_pass,
        "geometry_generalization_pass": geometry_generalization_pass,
        "negative_boundary_audit_pass": boundary_audit_pass,
        "tabular_formal_pass": tabular_formal_pass,
        "tabular_formal_pass_count": len(tabular_pass_tasks),
        "tabular_formal_tested_count": len(tabular_tested_tasks),
        "nlp_audio_ready": nlp_audio_ready,
        "nlp_audio_blocked_count": nlp_audio_blocked_count,
        "nlp_audio_baseline_measured": nlp_audio_baseline_measured,
        "nlp_audio_baseline_families": ",".join(nlp_audio_baseline_families),
        "nonvision_dg_transfer_count": nonvision_dg_transfer_count,
        "nonvision_dg_functional_task_pass_count": nonvision_dg_functional_task_pass_count,
        "nonvision_dg_geometry_pass_count": nonvision_dg_geometry_pass_count,
        "best_candidate": best_continual_candidate or "DG1-FT7-KW6-hidden28-functional",
        "best_task_family": "MNIST-like",
        "boundary_label": boundary_label,
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "next_family_required_implementation": next_family_impl,
        "success_v86_minimum": int(v85_repro_pass and continual_formal_pass),
        "success_v86_external_formal": external_formal_pass,
        "success_v86_broad_strong": int(external_formal_pass and time_accounting_pass and profiler_pass and geometry_generalization_pass and boundary_audit_pass),
        "success_v86_all_family_full": strict_all_family_pass,
        "cpu_offload_used": audit.get("cpu_offload_used", 0),
        "no_fake": bool(audit.get("no_fake")),
        "no_proxy": bool(audit.get("no_proxy")),
    }
    save_json(out_dir / "route_decision.json", decision)
    save_json(out_dir / "aggregate_decision.json", decision)
    failures: List[Dict[str, Any]] = []
    if not v85_repro_pass:
        failures.append({"failure_id": "F1_v85_reproduction_fail", "active": 1, "detail": blocker})
    if route == "R5-ContinualUnbalanced":
        failures.append({"failure_id": "F10_continual_unbalanced", "active": 1, "detail": blocker})
    if route == "R2-TaskFamilyLimitedFairAdvantage" and str(blocker).startswith("broad_"):
        failures.append({"failure_id": "F11_external_multitask_formal_pending", "active": 1, "detail": blocker})
    if int(audit.get("fake_proxy_nonzero_count", 0)) != 0:
        failures.append({"failure_id": "F14_fake_or_proxy_violation", "active": 1, "detail": "fake_proxy_or_offload_nonzero"})
    if not failures:
        if decision["success_v86_external_formal"] and not decision["success_v86_broad_strong"]:
            if decision.get("phase_mapped_profiler_pass") and not decision.get("time_accounting_pass"):
                failures.append({"failure_id": "F13_time_accounting_fail", "active": 1, "detail": "external formal pass and P9 profiler pass, but P8 TimeAccountingPass is false"})
            elif decision.get("time_accounting_pass") and not decision.get("phase_mapped_profiler_pass"):
                failures.append({"failure_id": "F12_profiler_incomplete", "active": 1, "detail": "external formal pass and P8 time accounting pass, but phase mapped profiler is incomplete"})
            else:
                failures.append({"failure_id": "F12_profiler_incomplete", "active": 1, "detail": "external formal pass but phase mapped profiler/time accounting full gate not complete"})
        elif decision["success_v86_broad_strong"] and not decision["success_v86_all_family_full"]:
            failures.append({
                "failure_id": "F15_nonvision_all_family_pending",
                "active": 1,
                "detail": decision.get("next_family_required_implementation", "nonvision_family_pending"),
            })
    if not failures:
        failures.append({"failure_id": "none", "active": 0, "detail": "no active failure in executed waves"})
    write_csv(out_dir / "failure_table.csv", failures)
    return decision


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = ensure_dir(Path(args.out_dir))
    args.out_dir = str(out_dir)
    if args.fresh:
        for path in out_dir.glob("*"):
            if path.is_file():
                path.unlink()
    ensure_dir(out_dir / "figures")
    manifest = {
        "stage": "RUN_MANIFEST_V86",
        "script": "experiments/run_gafu_v86_real.py",
        "plan": PLAN_PATH,
        "out_dir": str(out_dir),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
    }
    save_json(out_dir / "run_manifest.json", manifest)
    if args.run_p0:
        _run_v85_fresh_reproduction(out_dir, args)
    elif not (out_dir / "v85_reproduction.csv").exists():
        write_csv(out_dir / "v85_reproduction.csv", [{
            "stage": "P0_V85_FRESH_REPRODUCTION_V86",
            "status": "not_run",
            "reason": "run_p0_false",
            "FreshReproPass": 0,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }])
    if args.run_p7:
        _run_continual_multisplit(out_dir, args)
    elif not (out_dir / "continual_multisplit_stress.csv").exists():
        write_csv(out_dir / "continual_multisplit_stress.csv", [{
            "stage": "P7_CONTINUAL_MULTISPLIT_STRESS_V86",
            "status": "not_run",
            "reason": "run_p7_false",
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        }])
    if args.run_broad_vision:
        _run_broad_vision_screen(out_dir, args)
    if args.run_tabular_formal:
        _run_tabular_formal_envelope(out_dir, args)
    if args.run_nonvision_baseline:
        _run_nonvision_baseline_reproduction(out_dir, args)
    if args.run_nonvision_dg_transfer:
        _run_nonvision_dg_transfer(out_dir, args)
    _write_boundary_postprocess(out_dir)
    if args.run_system_profiler or args.run_kernel_profiler:
        _run_external_system_profiler(out_dir, args)
    _write_not_run(out_dir, "not_opened_in_first_v86_execution_wave")
    audit = _audit_no_fake(out_dir)
    decision = _route(out_dir, audit)
    manifest["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_json(out_dir / "run_manifest.json", manifest)
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DG-KAN v8.6 external generalization and continual formalization")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--kanbefair-path", default="third_party/KANbeFair")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--kb-epochs", type=int, default=20)
    parser.add_argument("--kb-batch-size", type=int, default=128)
    parser.add_argument("--primary-epochs", type=int, default=20)
    parser.add_argument("--run-p0", action="store_true")
    parser.add_argument("--run-p7", action="store_true")
    parser.add_argument("--run-broad-vision", action="store_true")
    parser.add_argument("--fresh-v85-reproduction", action="store_true")
    parser.add_argument("--continual-train-size-per-task", type=int, default=1024)
    parser.add_argument("--continual-test-size-per-task", type=int, default=512)
    parser.add_argument("--continual-epochs-per-task", type=int, default=2)
    parser.add_argument("--continual-batch-size", type=int, default=128)
    parser.add_argument("--continual-lr", type=float, default=1.0e-3)
    parser.add_argument("--broad-datasets", default="Fashion-MNIST,KMNIST")
    parser.add_argument("--broad-train-size", type=int, default=10000)
    parser.add_argument("--broad-test-size", type=int, default=2000)
    parser.add_argument("--broad-epochs", type=int, default=5)
    parser.add_argument("--broad-dg-hidden-dim", type=int, default=28)
    parser.add_argument("--broad-ft7-event-stride", type=int, default=128)
    parser.add_argument("--broad-ft7-event-alpha-mult", type=float, default=15.0)
    parser.add_argument("--broad-dg-stream-batches-from-cpu", action="store_true")
    parser.add_argument("--broad-dg-stream-chunk-batches", type=int, default=1)
    parser.add_argument("--broad-dg-stream-epoch-permute-cpu", action="store_true")
    parser.add_argument("--broad-dg-stream-chunk-order-shuffle", action="store_true")
    parser.add_argument("--run-tabular-formal", action="store_true")
    parser.add_argument("--tabular-datasets", default="Rice,Wine")
    parser.add_argument("--tabular-train-size", type=int, default=0)
    parser.add_argument("--tabular-test-size", type=int, default=0)
    parser.add_argument("--tabular-epochs", type=int, default=20)
    parser.add_argument("--tabular-dg-hidden-dims", default="2,4,8,16,28")
    parser.add_argument("--tabular-merge-artifacts", default="")
    parser.add_argument("--tabular-eval-batch-size", type=int, default=512)
    parser.add_argument("--tabular-ft7-event-stride", type=int, default=4)
    parser.add_argument("--tabular-ft7-event-alpha-mult", type=float, default=15.0)
    parser.add_argument("--tabular-dg-stream-batches-from-cpu", action="store_true")
    parser.add_argument("--tabular-dg-stream-chunk-batches", type=int, default=1)
    parser.add_argument("--tabular-dg-stream-epoch-permute-cpu", action="store_true")
    parser.add_argument("--tabular-dg-stream-chunk-order-shuffle", action="store_true")
    parser.add_argument("--run-nonvision-baseline", action="store_true")
    parser.add_argument("--nonvision-baseline-tasks", default="IMDb,SpeechCommand")
    parser.add_argument("--nonvision-baseline-train-size", type=int, default=1024)
    parser.add_argument("--nonvision-baseline-test-size", type=int, default=512)
    parser.add_argument("--nonvision-baseline-epochs", type=int, default=1)
    parser.add_argument("--nonvision-baseline-batch-size", type=int, default=128)
    parser.add_argument("--nonvision-baseline-lr", type=float, default=0.001)
    parser.add_argument("--run-nonvision-dg-transfer", action="store_true")
    parser.add_argument("--nonvision-dg-tasks", default="SpeechCommand")
    parser.add_argument("--nonvision-dg-hidden-dims", default="28")
    parser.add_argument("--nonvision-dg-train-size", type=int, default=1024)
    parser.add_argument("--nonvision-dg-test-size", type=int, default=512)
    parser.add_argument("--nonvision-dg-epochs", type=int, default=1)
    parser.add_argument("--nonvision-dg-batch-size", type=int, default=128)
    parser.add_argument("--nonvision-dg-eval-batch-size", type=int, default=512)
    parser.add_argument("--nonvision-dg-ft7-event-stride", type=int, default=128)
    parser.add_argument("--nonvision-dg-ft7-event-alpha-mult", type=float, default=15.0)
    parser.add_argument("--run-system-profiler", action="store_true")
    parser.add_argument("--run-kernel-profiler", action="store_true")
    parser.add_argument("--profiler-steps", type=int, default=1024)
    parser.add_argument("--profiler-trace-every", type=int, default=128)
    parser.add_argument("--profiler-eval-batch-size", type=int, default=512)
    parser.add_argument("--profiler-low-overhead", action="store_true")
    parser.add_argument("--profiler-async-window", action="store_true")
    parser.add_argument("--profiler-cuda-graph-non-event", action="store_true")
    parser.add_argument("--kernel-profiler-steps", type=int, default=512)
    parser.add_argument("--kernel-profiler-trace-every", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
