#!/usr/bin/env python3
"""KANbeFair import, model complexity, and dataset availability smoke tests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_17_common import (  # noqa: E402
    OUT_ROOT,
    PYTHON,
    RAW_ROOT,
    WORKTREE_ROOT,
    append_exec,
    ensure_out,
    write_rows,
)


@dataclass
class ArgsObj:
    model: str
    input_size: int = 16
    output_size: int = 3
    layers_width: list[int] | None = None
    batch_norm: bool = False
    activation_name: str = "gelu"
    kan_bspline_grid: int = 3
    kan_bspline_order: int = 2
    kan_shortcut_name: str = "silu"
    kan_grid_range: list[float] | None = None
    batch_size: int = 16
    test_batch_size: int = 16
    dataset: str = "MNIST"

    def __post_init__(self) -> None:
        if self.layers_width is None:
            self.layers_width = [4, 4]
        if self.kan_grid_range is None:
            self.kan_grid_range = [-2.0, 2.0]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="MNIST,FMNIST,KMNIST,Bean,Rice,Spam,Titanic,AG_NEWS,SpeechCommand")
    return p


def _subprocess_import(root: Path, label: str) -> dict[str, Any]:
    code = "import sys; sys.path.insert(0, 'src'); import utils; print('import_ok')"
    proc = subprocess.run(
        [PYTHON, "-c", code],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    return {
        "target": label,
        "root": str(root),
        "import_ok": int(proc.returncode == 0),
        "exit_code": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-500:].replace("\n", "\\n"),
        "stderr_tail": (proc.stderr or "")[-1000:].replace("\n", "\\n"),
    }


def _load_utils() -> Any:
    src = WORKTREE_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import utils  # type: ignore

    return utils


def _model_smoke() -> list[dict[str, Any]]:
    import torch

    utils = _load_utils()
    rows: list[dict[str, Any]] = []
    for model_name in ["MLP", "KAN", "BSpline_MLP", "BSpline_First_MLP"]:
        args = ArgsObj(model=model_name)
        try:
            args.activation = utils.get_activation(args)
            args.kan_shortcut_function = utils.get_shortcut_function(args)
            model = utils.get_model(args)
            x = torch.randn(5, args.input_size)
            y = model(x)
            loss = y.square().mean()
            loss.backward()
            grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
            num_parameters, flops = utils.get_model_complexity(model, None, args)
            rows.append(
                {
                    "model_name": model_name,
                    "input_size": args.input_size,
                    "output_size": args.output_size,
                    "layers_width": "4,4",
                    "param_count": num_parameters,
                    "flops": flops,
                    "forward_ok": int(tuple(y.shape) == (5, args.output_size)),
                    "backward_ok": int(grad_ok),
                    "complexity_ok": int(float(num_parameters) > 0 and float(flops) > 0),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model_name": model_name,
                    "input_size": args.input_size,
                    "output_size": args.output_size,
                    "layers_width": "4,4",
                    "param_count": "",
                    "flops": "",
                    "forward_ok": 0,
                    "backward_ok": 0,
                    "complexity_ok": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def _dataset_smoke(names: list[str]) -> list[dict[str, Any]]:
    utils = _load_utils()
    rows: list[dict[str, Any]] = []
    old_cwd = Path.cwd()
    os.chdir(WORKTREE_ROOT / "src")
    for name in names:
        args = ArgsObj(model="MLP", dataset=name)
        try:
            train_loader, test_loader, num_classes, input_size = utils.get_loader(args, use_cuda=False)
            train_batch = next(iter(train_loader))
            test_batch = next(iter(test_loader))
            rows.append(
                {
                    "dataset": name,
                    "availability": "available",
                    "train_len": len(train_loader.dataset),
                    "test_len": len(test_loader.dataset),
                    "num_classes": num_classes,
                    "input_size": input_size,
                    "first_train_batch_shape": str(tuple(train_batch[0].shape)),
                    "first_test_batch_shape": str(tuple(test_batch[0].shape)),
                    "deferred_reason": "",
                    "error": "",
                }
            )
        except Exception as exc:
            tier_reason = "dependency_or_cache_unavailable"
            if name in {"AG_NEWS", "CoLA", "IMDb"}:
                tier_reason = "text_dependency_or_dataset_deferred"
            elif name in {"SpeechCommand", "UrbanSound8K"}:
                tier_reason = "audio_cache_deferred"
            elif name in {"Bean", "Rice", "Spam", "Bank", "Income", "Wine", "Telescope"}:
                tier_reason = "ucimlrepo_or_network_deferred"
            elif name == "Titanic":
                tier_reason = "titanic_csv_missing_or_schema_deferred"
            rows.append(
                {
                    "dataset": name,
                    "availability": "unavailable",
                    "train_len": "",
                    "test_len": "",
                    "num_classes": "",
                    "input_size": "",
                    "first_train_batch_shape": "",
                    "first_test_batch_shape": "",
                    "deferred_reason": tier_reason,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    os.chdir(old_cwd)
    return rows


def _dependency_rows() -> list[dict[str, Any]]:
    rows = []
    for module in ["torchtext", "torchaudio", "ucimlrepo", "scipy"]:
        try:
            __import__(module)
            rows.append({"dependency": module, "available": 1, "error": ""})
        except Exception as exc:
            rows.append({"dependency": module, "available": 0, "error": f"{type(exc).__name__}: {exc}"})
    return rows


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    import_rows = []
    try:
        import_rows.append(_subprocess_import(RAW_ROOT / "KANbeFair-main", "raw_original"))
    except Exception as exc:
        import_rows.append({"target": "raw_original", "root": str(RAW_ROOT / "KANbeFair-main"), "import_ok": 0, "exit_code": "exception", "stdout_tail": "", "stderr_tail": str(exc)})
    try:
        import_rows.append(_subprocess_import(WORKTREE_ROOT, "worktree_patched"))
    except Exception as exc:
        import_rows.append({"target": "worktree_patched", "root": str(WORKTREE_ROOT), "import_ok": 0, "exit_code": "exception", "stdout_tail": "", "stderr_tail": str(exc)})
    write_rows(OUT_ROOT / "v22_17_kanbefair_import_smoke.csv", import_rows)

    try:
        model_rows = _model_smoke()
    except Exception:
        model_rows = [{"model_name": "ALL", "error": traceback.format_exc(), "forward_ok": 0, "backward_ok": 0, "complexity_ok": 0}]
    write_rows(OUT_ROOT / "v22_17_kanbefair_model_complexity_smoke.csv", model_rows)

    dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    dataset_rows = _dataset_smoke(dataset_names)
    write_rows(OUT_ROOT / "v22_17_kanbefair_dataset_availability_matrix.csv", dataset_rows)
    write_rows(OUT_ROOT / "v22_17_kanbefair_dataset_grid_availability.csv", dataset_rows)
    write_rows(OUT_ROOT / "v22_17_kanbefair_dependency_matrix.csv", _dependency_rows())
    bridge_import_ok = int(any(r.get("import_ok") == 1 and r.get("target") == "worktree_patched" for r in import_rows))
    write_rows(
        OUT_ROOT / "v22_17_kanbefair_bridge_import_tests.csv",
        [{"bridge_import": "worktree_utils", "pass": bridge_import_ok, "root": str(WORKTREE_ROOT)}],
    )
    append_exec(
        f"{sys.executable} experiments/run_v22_17_kanbefair_env_smoke.py --datasets {args.datasets}",
        task_id="A-G0-env-smoke",
        status="pass" if bridge_import_ok and all(int(r.get("complexity_ok", 0)) for r in model_rows) else "partial",
        gpu="0",
        exit_code=0,
        files=(
            "results/v22_17/v22_17_kanbefair_import_smoke.csv, "
            "results/v22_17/v22_17_kanbefair_model_complexity_smoke.csv, "
            "results/v22_17/v22_17_kanbefair_dataset_availability_matrix.csv"
        ),
        note="raw import and patched import are recorded separately",
    )


if __name__ == "__main__":
    main()
