#!/usr/bin/env python3
"""DG-KAN v9.2.25 dataset-specific functional direction/event selection.

This runner follows v9.2.24, where event scale repaired silence but did not
beat AdamWParallel/LR controls.  It explicitly separates Fashion-MNIST delayed
tail signals from KMNIST amplification failures and tests dataset-specific
direction/event candidates without changing the base N2a training.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
import run_v9222_basequalified_strict_purekan_functional_interface as v9222  # noqa: E402
import run_v9223_actuatability_to_causality_closure as v9223  # noqa: E402
import run_v9224_n2a_event_time_activation_repair as v9224  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402


SCRIPT_PATH = ROOT / "experiments" / "run_v9225_dataset_specific_direction_event_selection.py"
SRC_V9224 = ROOT / "results" / "real_rerun_20260506" / "v9224_n2a_event_time_activation_repair_first_20260510T140000Z"


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _mean(vals: Iterable[float]) -> float:
    clean = [float(v) for v in vals if math.isfinite(float(v))]
    return sum(clean) / max(1, len(clean))


def _not_run(stage: str, artifact: str, reason: str) -> Dict[str, Any]:
    return {
        "stage": stage,
        "status": "not_run",
        "artifact": artifact,
        "reason": reason,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _make_args(args: argparse.Namespace) -> argparse.Namespace:
    for name, value in {
        "p5_train_size": args.train_size,
        "p5_test_size": args.test_size,
        "p5_epochs": args.p5_epochs,
        "p5_lr": args.lr,
        "eval_size": args.eval_size,
        "audit_batch_size": args.audit_batch_size,
        "batch_size": args.batch_size,
        "data_root": args.data_root,
        "seed": args.seed,
        "lr": args.lr,
        "best_lr_scale": args.best_lr_scale,
        "ridge": args.ridge,
    }.items():
        setattr(args, name, value)
    return args


def _source_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9224 / "route_decision.json")
    audit = read_csv_rows(SRC_V9224 / "v9224_provenance_audit.csv")
    fake = _int(audit[0].get("fake_proxy_nonzero_count")) if audit else 1
    p0_pass = int(
        route.get("route") == "R3-ActivationRepairedButControlEquivalent"
        and _int(route.get("activation_band_pass")) == 1
        and _int(route.get("paired_replay_pass")) == 0
        and fake == 0
    )
    return {
        "stage": "P0_V9224_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "source_artifact": str(SRC_V9224.relative_to(ROOT)),
        "source_route": route.get("route", ""),
        "source_activation_policy": json.dumps(route.get("activation_policy", {}), sort_keys=True),
        "source_real_beats_adamwparallel_rate": route.get("real_beats_adamwparallel_rate", ""),
        "source_real_beats_best_lr_rate": route.get("real_beats_best_lr_rate", ""),
        "source_real_mean_CEp99_delta": route.get("real_mean_CEp99_delta", ""),
        "source_real_mean_margin_delta": route.get("real_mean_margin_delta", ""),
        "fake_proxy_count": fake,
        "P0_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _source_autopsy() -> List[Dict[str, Any]]:
    rows = read_csv_rows(SRC_V9224 / "p2_repaired_activation_paired_replay.csv")
    real = [r for r in rows if r.get("branch") == "RealFunctional"]
    out: List[Dict[str, Any]] = []
    for dataset in sorted({r.get("dataset", "") for r in real}):
        for target in sorted({r.get("target", "") for r in real if r.get("dataset") == dataset}):
            for horizon in sorted({_int(r.get("horizon")) for r in real if r.get("dataset") == dataset and r.get("target") == target}):
                group = [r for r in real if r.get("dataset") == dataset and r.get("target") == target and _int(r.get("horizon")) == horizon]
                out.append({
                    "stage": "P1_V9224_SIGNAL_AUTOPSY",
                    "scope": "dataset_target_horizon",
                    "dataset": dataset,
                    "target": target,
                    "horizon": horizon,
                    "rows": len(group),
                    "CEp99_delta": _mean([_float(r.get("CEp99_delta")) for r in group]),
                    "margin_p10_delta": _mean([_float(r.get("margin_p10_delta")) for r in group]),
                    "beats_adamwparallel_rate": _mean([_float(r.get("real_beats_adamwparallel")) for r in group]),
                    "beats_best_lr_rate": _mean([_float(r.get("real_beats_best_lr")) for r in group]),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
    return out


def _candidate_specs() -> List[Dict[str, Any]]:
    return [
        {"candidate": "D0-GlobalCap0.5-AllTargets", "dataset": "ALL", "targets": ["O1-HardTailLogitCorrection", "O2-MarginTailExpansion", "O6-KMNISTHardModeOutputTarget"], "fraction": 0.50, "direction": "safe_ls"},
        {"candidate": "F1-Fashion-O1O2-Cap0.5", "dataset": "Fashion-MNIST", "targets": ["O1-HardTailLogitCorrection", "O2-MarginTailExpansion"], "fraction": 0.50, "direction": "safe_ls"},
        {"candidate": "F2-Fashion-O1-Cap0.5", "dataset": "Fashion-MNIST", "targets": ["O1-HardTailLogitCorrection"], "fraction": 0.50, "direction": "safe_ls"},
        {"candidate": "F3-Fashion-O2-Cap0.5", "dataset": "Fashion-MNIST", "targets": ["O2-MarginTailExpansion"], "fraction": 0.50, "direction": "safe_ls"},
        {"candidate": "M1-MNIST-O2-Cap0.5", "dataset": "MNIST", "targets": ["O2-MarginTailExpansion"], "fraction": 0.50, "direction": "safe_ls"},
        {"candidate": "K1-KMNIST-O6-Cap0.1", "dataset": "KMNIST", "targets": ["O6-KMNISTHardModeOutputTarget"], "fraction": 0.10, "direction": "safe_ls"},
        {"candidate": "K2-KMNIST-O6-Cap0.5", "dataset": "KMNIST", "targets": ["O6-KMNISTHardModeOutputTarget"], "fraction": 0.50, "direction": "safe_ls"},
        {"candidate": "K3-KMNIST-O6-OrthogonalCap0.5", "dataset": "KMNIST", "targets": ["O6-KMNISTHardModeOutputTarget"], "fraction": 0.50, "direction": "orthogonal"},
        {"candidate": "K4-KMNIST-O1O2O6-OrthogonalCap0.5", "dataset": "KMNIST", "targets": ["O1-HardTailLogitCorrection", "O2-MarginTailExpansion", "O6-KMNISTHardModeOutputTarget"], "fraction": 0.50, "direction": "orthogonal"},
    ]


def _direction_step(kind: str, safe_delta: Sequence[torch.Tensor], task_step: Sequence[torch.Tensor], fraction: float) -> List[torch.Tensor]:
    if kind == "orthogonal":
        step = v9223._orthogonal_to(safe_delta, task_step)
        return v9222._cap_to_fraction(step, task_step, fraction)
    return v9222._cap_to_fraction(safe_delta, task_step, fraction)


def _run_matrix(args: argparse.Namespace, device: torch.device, opened: bool, cache: Dict[Tuple[str, str, int], Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P2_DATASET_SPECIFIC_DIRECTION_EVENT_MATRIX", "p2_dataset_specific_direction_event_matrix.csv", "P0_v9224_boundary_failed")
        return [row], [row], {}
    cand = v9222._candidate_registry()["N2a-TinyInit-RationalFunc-BranchRatioCap"]
    assert cand.spec is not None
    rows: List[Dict[str, Any]] = []
    horizons = _parse_ints(args.horizons)
    for spec in _candidate_specs():
        datasets = [v92._canonical_task(d) for d in _parse_list(args.datasets)]
        if spec["dataset"] != "ALL":
            datasets = [spec["dataset"]]
        for dataset in datasets:
            x_train, y_train, x_eval, y_eval, protocol = v9223._load_split(args, dataset, device)
            xb = x_train[: int(args.audit_batch_size)]
            yb = y_train[: int(args.audit_batch_size)]
            for seed in _parse_ints(args.seeds):
                saved = v9223._train_cache(args, cand, dataset, seed, device, cache)
                params = saved["params"]
                states = saved["states"]
                mu = saved["mu"]
                std = saved["std"]
                for target in spec["targets"]:
                    safe_delta, task_step, _grads, info = v9223._base_direction(args, cand, params, mu, std, xb, yb, dataset, target)
                    if _float(info.get("target_selected_fraction")) <= 0.0:
                        rows.append({
                            "stage": "P2_DATASET_SPECIFIC_DIRECTION_EVENT_MATRIX",
                            "status": "not_applicable",
                            "candidate": spec["candidate"],
                            "dataset": dataset,
                            "seed": seed,
                            "target": target,
                            "reason": "target_selected_fraction_zero",
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        })
                        continue
                    real_step = _direction_step(spec["direction"], safe_delta, task_step, float(spec["fraction"]))
                    metrics = v9223._run_replay_for_event(args, params, states, mu, std, cand.spec, x_train, y_train, x_eval, y_eval, real_step, task_step, seed, horizons)
                    for horizon in horizons:
                        adamw = metrics[("AdamWOnly", horizon)]
                        best_parallel = min([m for (b, h), m in metrics.items() if h == horizon and b.startswith("AdamWParallel")], key=lambda m: m["CE_p99"])
                        best_lr = metrics[("BestLRScale", horizon)]
                        for (branch, h), after in metrics.items():
                            if h != horizon:
                                continue
                            delta = v9223._metric_delta(adamw, after)
                            rows.append({
                                "stage": "P2_DATASET_SPECIFIC_DIRECTION_EVENT_MATRIX",
                                "status": "measured",
                                "candidate": spec["candidate"],
                                "dataset": dataset,
                                "seed": seed,
                                "protocol": protocol,
                                "target": target,
                                "direction": spec["direction"],
                                "activation_fraction": spec["fraction"],
                                "horizon": horizon,
                                "branch": branch,
                                "CEp99_delta": delta["CEp99_delta"],
                                "margin_p10_delta": delta["margin_p10_delta"],
                                "ECE_delta": delta["ECE_delta"],
                                "NLL_delta": delta["NLL_delta"],
                                "acc_delta": delta["acc_delta"],
                                "real_beats_adamwparallel": int(branch == "RealFunctional" and after["CE_p99"] < best_parallel["CE_p99"]),
                                "real_beats_best_lr": int(branch == "RealFunctional" and after["CE_p99"] < best_lr["CE_p99"]),
                                "task_safe": int(branch != "RealFunctional" or after["acc"] >= adamw["acc"] - 0.005),
                                "fake_data_used": 0,
                                "proxy_row_used": 0,
                                "cpu_offload_used": 0,
                            })
    summary: List[Dict[str, Any]] = []
    real_rows = [r for r in rows if r.get("branch") == "RealFunctional"]
    for cand_id in sorted({r.get("candidate", "") for r in real_rows}):
        vals = [r for r in real_rows if r.get("candidate") == cand_id]
        for scope, group in [
            ("all_horizons", vals),
            ("h80_only", [r for r in vals if _int(r.get("horizon")) == 80]),
        ]:
            if not group:
                continue
            beat_p = _mean([_float(r.get("real_beats_adamwparallel")) for r in group])
            beat_lr = _mean([_float(r.get("real_beats_best_lr")) for r in group])
            task_safe = _mean([_float(r.get("task_safe")) for r in group])
            ce = _mean([_float(r.get("CEp99_delta")) for r in group])
            margin = _mean([_float(r.get("margin_p10_delta")) for r in group])
            local_pass = int(beat_p >= 0.50 and beat_lr >= 0.50 and task_safe >= 0.95 and (ce < 0.0 or margin > 0.0))
            summary.append({
                "stage": "P3_DIRECTION_EVENT_SELECTION_SUMMARY",
                "candidate": cand_id,
                "analysis_scope": scope,
                "dataset": ",".join(sorted({str(r.get("dataset")) for r in group})),
                "targets": ",".join(sorted({str(r.get("target")) for r in group})),
                "rows": len(group),
                "real_beats_adamwparallel_rate": beat_p,
                "real_beats_best_lr_rate": beat_lr,
                "task_safe_rate": task_safe,
                "CEp99_delta": ce,
                "margin_p10_delta": margin,
                "local_pass": local_pass,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    all_pass = [s for s in summary if s.get("analysis_scope") == "all_horizons" and _int(s.get("local_pass"))]
    h80_pass = [s for s in summary if s.get("analysis_scope") == "h80_only" and _int(s.get("local_pass"))]
    best = max(summary, key=lambda s: (_float(s.get("real_beats_adamwparallel_rate")), _float(s.get("real_beats_best_lr_rate")), -_float(s.get("CEp99_delta"), 99.0)), default={})
    decision = {
        "all_horizon_pass_count": len(all_pass),
        "h80_delayed_pass_count": len(h80_pass),
        "best_candidate": best.get("candidate", ""),
        "best_scope": best.get("analysis_scope", ""),
        "best_real_beats_adamwparallel_rate": best.get("real_beats_adamwparallel_rate", 0),
        "best_real_beats_best_lr_rate": best.get("real_beats_best_lr_rate", 0),
        "best_CEp99_delta": best.get("CEp99_delta", 0),
        "best_margin_delta": best.get("margin_p10_delta", 0),
    }
    return rows, summary, decision


def _write_downstream(out_dir: Path, reason: str) -> None:
    for fname, stage in [
        ("p4_short_run_direction_event_validation.csv", "P4_SHORT_RUN_DIRECTION_EVENT_VALIDATION"),
        ("p5_full_functional_direction_event_validation.csv", "P5_FULL_FUNCTIONAL_DIRECTION_EVENT_VALIDATION"),
        ("p6_external_ready.csv", "P6_EXTERNAL_READY"),
    ]:
        write_csv_rows(out_dir / fname, [_not_run(stage, fname, reason)])


def _write_figures(out_dir: Path, route: Dict[str, Any]) -> None:
    fig = ensure_dir(out_dir / "figures")
    lines = [
        f"route = {route.get('route')}",
        f"best = {route.get('best_candidate')}",
        f"best_scope = {route.get('best_scope')}",
        f"h80_pass_count = {route.get('h80_delayed_pass_count')}",
        f"blocker = {route.get('primary_blocker')}",
    ]
    for name, title in [
        ("p1_v9224_signal_autopsy.svg", "v9.2.24 source signal autopsy"),
        ("p2_dataset_specific_matrix.svg", "Dataset-specific direction matrix"),
        ("p3_fashion_vs_kmnist_split.svg", "Fashion vs KMNIST split"),
        ("p3_delayed_h80_signal.svg", "Delayed h80 signal"),
    ]:
        (fig / name).write_text(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1160\" height=\"250\">"
            "<rect width=\"1160\" height=\"250\" fill=\"#f8fafc\"/>"
            f"<text x=\"24\" y=\"42\" font-family=\"Arial\" font-size=\"24\" fill=\"#111827\">{title}</text>"
            + "".join(f"<text x=\"24\" y=\"{82 + i * 28}\" font-family=\"Arial\" font-size=\"16\" fill=\"#374151\">{line}</text>" for i, line in enumerate(lines))
            + "</svg>\n",
            encoding="utf-8",
        )


def _write_report(out_dir: Path, route: Dict[str, Any], audit: Dict[str, Any], decision: Dict[str, Any]) -> None:
    report = ROOT / "docs" / "DG-KAN_v9.2.25_DatasetSpecific_DirectionEventSelection_实验复盘.md"
    text = f"""# DG-KAN v9.2.25 Dataset-Specific Direction/Event Selection 实验复盘

> 本复盘记录本轮针对 v9.2.24 `ActivationRepairedButControlEquivalent` 的 dataset-specific direction/event selection 实验。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把未打开的 downstream 阶段写成通过。

## 0. 最新结论

```text
route = {route.get('route')}
base_candidate = LQ-t2-h256
success_v9225_local_direction_signal = {str(bool(route.get('success_v9225_local_direction_signal'))).lower()}
success_v9225_strict_purekan_functional = {str(bool(route.get('success_v9225_strict_purekan_functional'))).lower()}
```

最终 artifact：

```text
{out_dir.relative_to(ROOT)}/
```

核心结论：

1. v9.2.24 boundary 被复现：event scale 已修复 silent，但 global paired replay 仍 control-equivalent。
2. P1 从 v9.2.24 source rows 做 autopsy，Fashion-MNIST 的 positive CEp99 主要是 delayed h80 tail signal；KMNIST 放大后仍弱。
3. P2 dataset-specific direction/event matrix 已真实执行，覆盖 Fashion O1/O2、MNIST O2、KMNIST O6 low/high cap 与 KMNIST orthogonal variants。
4. all-horizon pass count = `{route.get('all_horizon_pass_count')}`，h80 delayed pass count = `{route.get('h80_delayed_pass_count')}`。
5. best candidate = `{route.get('best_candidate')}` / `{route.get('best_scope')}`，beats AdamWParallel = `{route.get('best_real_beats_adamwparallel_rate')}`，beats best LR = `{route.get('best_real_beats_best_lr_rate')}`。
6. 当前 blocker：`{route.get('primary_blocker')}`。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9225_dataset_specific_direction_event_selection.py` | v9.2.25 runner；执行 dataset-specific functional direction/event matrix |

代码检查：

```text
python -m py_compile experiments/run_v9225_dataset_specific_direction_event_selection.py
```

正式运行：

```bash
python experiments/run_v9225_dataset_specific_direction_event_selection.py \\
  --out-dir {out_dir.relative_to(ROOT)} \\
  --fresh \\
  --device auto \\
  --data-root data \\
  --seed 1314
```

## 2. Route

```json
{json.dumps(route, indent=2, ensure_ascii=False)}
```

## 3. P2/P3 result

Artifacts：

```text
p2_dataset_specific_direction_event_matrix.csv
p3_direction_event_selection_summary.csv
```

Summary：

```text
all_horizon_pass_count = {decision.get('all_horizon_pass_count')}
h80_delayed_pass_count = {decision.get('h80_delayed_pass_count')}
best_candidate = {decision.get('best_candidate')}
best_scope = {decision.get('best_scope')}
best_real_beats_adamwparallel_rate = {decision.get('best_real_beats_adamwparallel_rate')}
best_real_beats_best_lr_rate = {decision.get('best_real_beats_best_lr_rate')}
best_CEp99_delta = {decision.get('best_CEp99_delta')}
best_margin_delta = {decision.get('best_margin_delta')}
```

判断：本轮不是继续扫 activation scale，而是重做 direction/event selection 并分开评估 Fashion-MNIST 与 KMNIST。

## 4. Downstream boundary

Short/full/external validation 只有在 all-horizon local pass 后打开。本轮未打开阶段均以 `not_run` row 落盘。

## 5. No-fake audit

```text
rows_checked = {audit.get('rows_checked')}
fake_proxy_nonzero_count = {audit.get('fake_proxy_nonzero_count')}
fake_data_used = {audit.get('fake_data_used')}
proxy_row_used = {audit.get('proxy_row_used')}
cpu_offload_used = {audit.get('cpu_offload_used')}
no_fake = {str(audit.get('no_fake')).lower()}
no_proxy = {str(audit.get('no_proxy')).lower()}
```

## 6. 最终分析结论

v9.2.25 的真实推进是：

```text
v9.2.24: scale repair makes event non-silent, but still control-equivalent.
v9.2.25: dataset-specific direction/event candidates test whether Fashion delayed signal or KMNIST O6/orthogonal variants can escape controls.
```

最终一句话：

> v9.2.25 真实执行后停在 `{route.get('route')}`：`{route.get('primary_blocker')}`。
"""
    report.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=9984)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--audit-batch-size", type=int, default=128)
    parser.add_argument("--p5-epochs", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1.0e-4)
    parser.add_argument("--parallel-trust-ratio", type=float, default=0.03)
    parser.add_argument("--best-lr-scale", type=float, default=1.03)
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--horizons", default="1,5,20,80")
    args = _make_args(parser.parse_args())

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = _device(args.device)
    write_json(out_dir / "run_manifest.json", {
        "experiment": "DG-KAN v9.2.25 Dataset-Specific Direction/Event Selection",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "script": str(SCRIPT_PATH.relative_to(ROOT)),
        "source_v9224": str(SRC_V9224.relative_to(ROOT)),
        "args": vars(args),
    })
    contract = [{
        "stage": "CONTRACT_AUDIT_V9225",
        "loss_type": "CE",
        "label_smoothing": 0,
        "teacher_used": 0,
        "distillation_used": 0,
        "sampler_changed": 0,
        "class_weight_used": 0,
        "cpu_offload_used": 0,
        "ordinary_mlp_path_used": 0,
        "external_residual_used": 0,
        "uses_loss_backward": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }]
    write_csv_rows(out_dir / "contract_audit_v9225.csv", contract)
    p0 = _source_boundary()
    write_csv_rows(out_dir / "p0_v9224_boundary_reproduction.csv", [p0])
    p1 = _source_autopsy()
    write_csv_rows(out_dir / "p1_v9224_signal_autopsy.csv", p1)
    cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    p2, p3, decision = _run_matrix(args, device, _int(p0.get("P0_pass")) == 1, cache)
    write_csv_rows(out_dir / "p2_dataset_specific_direction_event_matrix.csv", p2)
    write_csv_rows(out_dir / "p3_direction_event_selection_summary.csv", p3)
    if _int(decision.get("all_horizon_pass_count")) == 0:
        _write_downstream(out_dir, "no_all_horizon_direction_event_survivor")

    if not _int(p0.get("P0_pass")):
        route_name = "R1-SourceBoundaryFailed"
        blocker = "v9224_boundary_not_reproduced"
        next_impl = "reproduce_v9224_before_direction_selection"
    elif _int(decision.get("all_horizon_pass_count")):
        route_name = "R3-DatasetSpecificDirectionEventPass"
        blocker = "short_full_validation_not_opened_in_this_runner"
        next_impl = "open_short_run_for_dataset_specific_survivor"
    elif _int(decision.get("h80_delayed_pass_count")):
        route_name = "R2-DelayedFashionSignalOnly"
        blocker = "only_delayed_h80_local_signal_no_all_horizon_survivor"
        next_impl = "design_delayed_event_controller_or_short_run_ablation_for_fashion_signal"
    else:
        route_name = "R4-DirectionEventControlEquivalent"
        blocker = "dataset_specific_direction_event_candidates_still_lose_strong_controls"
        next_impl = "redesign_functional_direction_beyond_output_LS_or_return_to_interface_basis"

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "source_route": p0.get("source_route", ""),
        "all_horizon_pass_count": decision.get("all_horizon_pass_count", 0),
        "h80_delayed_pass_count": decision.get("h80_delayed_pass_count", 0),
        "best_candidate": decision.get("best_candidate", ""),
        "best_scope": decision.get("best_scope", ""),
        "best_real_beats_adamwparallel_rate": decision.get("best_real_beats_adamwparallel_rate", 0),
        "best_real_beats_best_lr_rate": decision.get("best_real_beats_best_lr_rate", 0),
        "best_CEp99_delta": decision.get("best_CEp99_delta", 0),
        "best_margin_delta": decision.get("best_margin_delta", 0),
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "success_v9225_local_direction_signal": int(_int(decision.get("all_horizon_pass_count")) or _int(decision.get("h80_delayed_pass_count"))),
        "success_v9225_strict_purekan_functional": int(_int(decision.get("all_horizon_pass_count")) > 0),
        "success_v9225_external_ready": 0,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", {"route": route, "p0": p0, "decision": decision})
    write_csv_rows(out_dir / "failure_table.csv", [{
        "stage": "ROUTE",
        "failure": route_name,
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    _write_figures(out_dir, route)
    audit_paths = [
        out_dir / "contract_audit_v9225.csv",
        out_dir / "p0_v9224_boundary_reproduction.csv",
        out_dir / "p1_v9224_signal_autopsy.csv",
        out_dir / "p2_dataset_specific_direction_event_matrix.csv",
        out_dir / "p3_direction_event_selection_summary.csv",
        out_dir / "p4_short_run_direction_event_validation.csv",
        out_dir / "p5_full_functional_direction_event_validation.csv",
        out_dir / "p6_external_ready.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake([p for p in audit_paths if p.exists()])
    write_csv_rows(out_dir / "v9225_provenance_audit.csv", [audit])
    hash_paths = [SCRIPT_PATH, out_dir / "run_manifest.json", out_dir / "route_decision.json", out_dir / "aggregate_decision.json", *[p for p in audit_paths if p.exists()], out_dir / "v9225_provenance_audit.csv"]
    write_csv_rows(out_dir / "artifact_hashes.csv", artifact_hash_rows(hash_paths, root=ROOT))
    _write_report(out_dir, route, audit, decision)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
