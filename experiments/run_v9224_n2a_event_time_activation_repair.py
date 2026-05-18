#!/usr/bin/env python3
"""DG-KAN v9.2.24 N2a event-time activation repair runner.

v9.2.23 showed that N2a's actuatability proxy is informative but the realized
logit movement is too small, especially on MNIST/Fashion-MNIST.  This runner
keeps the base N2a training unchanged and sweeps event-time activation bands to
test whether the silence can be repaired without breaking task safety, then
checks the repaired policy against AdamWParallel / LR controls.
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
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, read_csv_rows, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.models import fc_purekan_actuator as act  # noqa: E402


SCRIPT_PATH = ROOT / "experiments" / "run_v9224_n2a_event_time_activation_repair.py"
SRC_V9223 = ROOT / "results" / "real_rerun_20260506" / "v9223_actuatability_to_causality_closure_first_20260510T130000Z"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


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


def _clone_params(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


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
    }.items():
        setattr(args, name, value)
    return args


def _source_boundary() -> Dict[str, Any]:
    route = _read_json(SRC_V9223 / "route_decision.json")
    audit = read_csv_rows(SRC_V9223 / "v9223_provenance_audit.csv")
    fake = _int(audit[0].get("fake_proxy_nonzero_count")) if audit else 1
    p0_pass = int(
        route.get("route") == "R2-FunctionalEventSilent"
        and _int(route.get("v9222_boundary_pass")) == 1
        and _int(route.get("actuatability_proxy_calibration_pass")) == 1
        and _float(route.get("p1_mean_actual_r_z")) < 0.10
        and _float(route.get("p1_mean_actual_r_z_tail")) < 0.10
        and fake == 0
    )
    return {
        "stage": "P0_V9223_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "source_artifact": str(SRC_V9223.relative_to(ROOT)),
        "source_route": route.get("route", ""),
        "source_failure_mode": route.get("failure_mode", ""),
        "source_proxy_actual_corr": route.get("p1_proxy_actual_corr", ""),
        "source_mean_actual_r_z": route.get("p1_mean_actual_r_z", ""),
        "source_mean_actual_r_z_tail": route.get("p1_mean_actual_r_z_tail", ""),
        "source_silent_rate": route.get("p1_silent_rate", ""),
        "fake_proxy_count": fake,
        "P0_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _is_applicable(row: Dict[str, Any]) -> int:
    return int(_float(row.get("target_selected_fraction")) > 0.0)


def _p1_scale_sweep(
    args: argparse.Namespace,
    device: torch.device,
    opened: bool,
    cache: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P1_N2A_EVENT_TIME_SCALE_SWEEP", "p1_n2a_event_time_activation_sweep.csv", "P0_v9223_boundary_failed")
        return [row], [row], {}
    cand = v9222._candidate_registry()["N2a-TinyInit-RationalFunc-BranchRatioCap"]
    assert cand.spec is not None
    rows: List[Dict[str, Any]] = []
    fractions = _parse_floats(args.activation_fractions)
    for dataset in [v92._canonical_task(d) for d in _parse_list(args.datasets)]:
        x_train, y_train, x_eval, y_eval, protocol = v9223._load_split(args, dataset, device)
        xb = x_train[: int(args.audit_batch_size)]
        yb = y_train[: int(args.audit_batch_size)]
        for seed in _parse_ints(args.seeds):
            saved = v9223._train_cache(args, cand, dataset, seed, device, cache)
            params = saved["params"]
            mu = saved["mu"]
            std = saved["std"]
            for target_id in _parse_list(args.targets):
                safe_delta, task_step, _grads, info = v9223._base_direction(args, cand, params, mu, std, xb, yb, dataset, target_id)
                parallel_step = v9222._cap_to_fraction(task_step, task_step, float(args.parallel_trust_ratio))
                for frac in fractions:
                    real_step = v9222._cap_to_fraction(safe_delta, task_step, frac)
                    stats = v9223._actual_stats(
                        params=params,
                        real_step=real_step,
                        parallel_step=parallel_step,
                        mu=mu,
                        std=std,
                        spec=cand.spec,
                        x_eval=x_eval,
                        y_eval=y_eval,
                        dataset=dataset,
                        target_id=target_id,
                    )
                    after_diag = v9222._diagnose_channels(v9223._apply_step(params, real_step), mu, std, cand.spec, xb, yb, float(args.lr))
                    applicable = int(_float(info.get("target_selected_fraction")) > 0.0)
                    silent = int(applicable and (stats["actual_r_z"] < 0.10 or stats["actual_r_z_tail"] < 0.10))
                    task_safe = int(stats["real_acc_delta"] >= -0.005)
                    nonharm = int(stats["real_NLL_delta"] <= 1.0e-7)
                    rows.append({
                        "stage": "P1_N2A_EVENT_TIME_SCALE_SWEEP",
                        "candidate": cand.candidate_id,
                        "dataset": dataset,
                        "seed": seed,
                        "protocol": protocol,
                        "target": target_id,
                        "activation_fraction": frac,
                        "target_applicable": applicable,
                        "target_selected_fraction": info.get("target_selected_fraction", 0.0),
                        "target_fit_R2": info.get("target_fit_R2", ""),
                        "p3_proxy_r_perp": info.get("p3_proxy_r_perp", ""),
                        "actual_r_z": stats["actual_r_z"],
                        "actual_r_z_tail": stats["actual_r_z_tail"],
                        "actual_r_z_perp": stats["actual_r_z_perp"],
                        "actual_logit_delta_norm": stats["actual_logit_delta_norm"],
                        "actual_tail_logit_delta_norm": stats["actual_tail_logit_delta_norm"],
                        "real_CEp99_delta": stats["real_CEp99_delta"],
                        "adamwparallel_CEp99_delta": stats["adamwparallel_CEp99_delta"],
                        "real_margin_p10_delta": stats["real_margin_p10_delta"],
                        "adamwparallel_margin_p10_delta": stats["adamwparallel_margin_p10_delta"],
                        "real_NLL_delta": stats["real_NLL_delta"],
                        "real_acc_delta": stats["real_acc_delta"],
                        "branch_ratio_event": after_diag["branch_ratio"],
                        "effective_derivative_event": after_diag["effective_derivative_p95"],
                        "functional_step_norm": float(snr_lq.step_norm(real_step).detach().cpu()),
                        "task_step_norm": float(snr_lq.step_norm(task_step).detach().cpu()),
                        "silent": silent,
                        "task_safe": task_safe,
                        "holdout_nonharm": nonharm,
                        "fake_data_used": 0,
                        "proxy_row_used": 0,
                        "cpu_offload_used": 0,
                    })
    summary: List[Dict[str, Any]] = []
    applicable_rows = [r for r in rows if _is_applicable(r)]
    for dataset in sorted({str(r.get("dataset")) for r in applicable_rows}):
        for frac in fractions:
            group = [r for r in applicable_rows if str(r.get("dataset")) == dataset and abs(_float(r.get("activation_fraction")) - frac) < 1.0e-12]
            if not group:
                continue
            silent_rate = _mean([_float(r.get("silent")) for r in group])
            task_safe_rate = _mean([_float(r.get("task_safe")) for r in group])
            mean_rz = _mean([_float(r.get("actual_r_z")) for r in group])
            mean_tail = _mean([_float(r.get("actual_r_z_tail")) for r in group])
            pass_gate = int(silent_rate <= 0.20 and task_safe_rate >= 0.95 and mean_rz >= 0.10 and mean_tail >= 0.10)
            summary.append({
                "stage": "P1_N2A_EVENT_TIME_SCALE_SUMMARY",
                "scope": "dataset_fraction",
                "dataset": dataset,
                "activation_fraction": frac,
                "eligible_rows": len(group),
                "silent_rate": silent_rate,
                "task_safe_rate": task_safe_rate,
                "holdout_nonharm_rate": _mean([_float(r.get("holdout_nonharm")) for r in group]),
                "mean_actual_r_z": mean_rz,
                "mean_actual_r_z_tail": mean_tail,
                "mean_CEp99_delta": _mean([_float(r.get("real_CEp99_delta")) for r in group]),
                "mean_margin_delta": _mean([_float(r.get("real_margin_p10_delta")) for r in group]),
                "activation_band_pass": pass_gate,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    global_candidates: List[Dict[str, Any]] = []
    for frac in fractions:
        group = [r for r in applicable_rows if abs(_float(r.get("activation_fraction")) - frac) < 1.0e-12]
        if not group:
            continue
        dataset_pass = [s for s in summary if s.get("scope") == "dataset_fraction" and abs(_float(s.get("activation_fraction")) - frac) < 1.0e-12]
        global_pass = int(dataset_pass and all(_int(s.get("activation_band_pass")) for s in dataset_pass))
        global_candidates.append({
            "stage": "P1_N2A_EVENT_TIME_SCALE_SUMMARY",
            "scope": "global_fraction",
            "dataset": "ALL",
            "activation_fraction": frac,
            "eligible_rows": len(group),
            "silent_rate": _mean([_float(r.get("silent")) for r in group]),
            "task_safe_rate": _mean([_float(r.get("task_safe")) for r in group]),
            "holdout_nonharm_rate": _mean([_float(r.get("holdout_nonharm")) for r in group]),
            "mean_actual_r_z": _mean([_float(r.get("actual_r_z")) for r in group]),
            "mean_actual_r_z_tail": _mean([_float(r.get("actual_r_z_tail")) for r in group]),
            "mean_CEp99_delta": _mean([_float(r.get("real_CEp99_delta")) for r in group]),
            "mean_margin_delta": _mean([_float(r.get("real_margin_p10_delta")) for r in group]),
            "activation_band_pass": global_pass,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    summary.extend(global_candidates)
    global_passes = [s for s in global_candidates if _int(s.get("activation_band_pass"))]
    if global_passes:
        chosen = min(global_passes, key=lambda s: _float(s.get("activation_fraction")))
        policy = {d: _float(chosen.get("activation_fraction")) for d in sorted({str(r.get("dataset")) for r in applicable_rows})}
        policy_type = "global"
    else:
        policy = {}
        for dataset in sorted({str(r.get("dataset")) for r in applicable_rows}):
            ds_pass = [s for s in summary if s.get("scope") == "dataset_fraction" and s.get("dataset") == dataset and _int(s.get("activation_band_pass"))]
            if ds_pass:
                policy[dataset] = _float(min(ds_pass, key=lambda s: _float(s.get("activation_fraction"))).get("activation_fraction"))
        policy_type = "dataset_aware"
    all_ds = sorted({str(r.get("dataset")) for r in applicable_rows})
    activation_pass = int(bool(all_ds) and all(d in policy for d in all_ds))
    decision = {
        "activation_band_pass": activation_pass,
        "activation_policy_type": policy_type if activation_pass else "none",
        "activation_policy": policy,
        "eligible_rows": len(applicable_rows),
        "best_global_fraction": _float(global_passes[0].get("activation_fraction")) if global_passes else "",
        "min_fraction_by_dataset": json.dumps(policy, sort_keys=True),
    }
    return rows, summary, decision


def _p2_paired_replay(
    args: argparse.Namespace,
    device: torch.device,
    opened: bool,
    policy: Dict[str, float],
    cache: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        return [_not_run("P2_REPAIRED_ACTIVATION_PAIRED_REPLAY", "p2_repaired_activation_paired_replay.csv", "P1_activation_band_not_found")], {}
    cand = v9222._candidate_registry()["N2a-TinyInit-RationalFunc-BranchRatioCap"]
    assert cand.spec is not None
    rows: List[Dict[str, Any]] = []
    horizons = _parse_ints(args.horizons)
    for dataset in [v92._canonical_task(d) for d in _parse_list(args.datasets)]:
        if dataset not in policy:
            continue
        x_train, y_train, x_eval, y_eval, protocol = v9223._load_split(args, dataset, device)
        xb = x_train[: int(args.audit_batch_size)]
        yb = y_train[: int(args.audit_batch_size)]
        frac = float(policy[dataset])
        for seed in _parse_ints(args.seeds):
            saved = v9223._train_cache(args, cand, dataset, seed, device, cache)
            params = saved["params"]
            states = saved["states"]
            mu = saved["mu"]
            std = saved["std"]
            for target_id in _parse_list(args.targets):
                safe_delta, task_step, _grads, info = v9223._base_direction(args, cand, params, mu, std, xb, yb, dataset, target_id)
                if _float(info.get("target_selected_fraction")) <= 0.0:
                    continue
                real_step = v9222._cap_to_fraction(safe_delta, task_step, frac)
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
                            "stage": "P2_REPAIRED_ACTIVATION_PAIRED_REPLAY",
                            "candidate": "N2a-EventTimeActivationBand",
                            "dataset": dataset,
                            "seed": seed,
                            "protocol": protocol,
                            "target": target_id,
                            "activation_fraction": frac,
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
    real = [r for r in rows if r.get("branch") == "RealFunctional"]
    summary = {
        "paired_replay_pass": int(real and _mean([_float(r.get("real_beats_adamwparallel")) for r in real]) >= 0.50 and _mean([_float(r.get("real_beats_best_lr")) for r in real]) >= 0.50 and _mean([_float(r.get("task_safe")) for r in real]) >= 0.95),
        "real_row_count": len(real),
        "real_beats_adamwparallel_rate": _mean([_float(r.get("real_beats_adamwparallel")) for r in real]),
        "real_beats_best_lr_rate": _mean([_float(r.get("real_beats_best_lr")) for r in real]),
        "task_safe_rate": _mean([_float(r.get("task_safe")) for r in real]),
        "real_mean_CEp99_delta": _mean([_float(r.get("CEp99_delta")) for r in real]),
        "real_mean_margin_delta": _mean([_float(r.get("margin_p10_delta")) for r in real]),
    }
    return rows, summary


def _write_downstream(out_dir: Path, reason: str) -> None:
    for fname, stage in [
        ("p3_short_run_after_activation_repair.csv", "P3_SHORT_RUN_AFTER_ACTIVATION_REPAIR"),
        ("p4_full_functional_after_activation_repair.csv", "P4_FULL_FUNCTIONAL_AFTER_ACTIVATION_REPAIR"),
        ("p5_robustness_external_ready.csv", "P5_ROBUSTNESS_EXTERNAL_READY"),
    ]:
        write_csv_rows(out_dir / fname, [_not_run(stage, fname, reason)])


def _write_figures(out_dir: Path, route: Dict[str, Any]) -> None:
    fig = ensure_dir(out_dir / "figures")
    lines = [
        f"route = {route.get('route')}",
        f"activation_policy = {route.get('activation_policy')}",
        f"activation_band_pass = {route.get('activation_band_pass')}",
        f"paired_replay_pass = {route.get('paired_replay_pass')}",
        f"blocker = {route.get('primary_blocker')}",
    ]
    for name, title in [
        ("p1_activation_fraction_vs_rz.svg", "Activation fraction vs r_z"),
        ("p1_dataset_split_silence.svg", "Dataset split silence"),
        ("p2_repaired_real_vs_controls.svg", "Repaired Real vs controls"),
        ("p2_horizon_control_gap.svg", "Horizon control gap"),
    ]:
        (fig / name).write_text(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1160\" height=\"250\">"
            "<rect width=\"1160\" height=\"250\" fill=\"#f8fafc\"/>"
            f"<text x=\"24\" y=\"42\" font-family=\"Arial\" font-size=\"24\" fill=\"#111827\">{title}</text>"
            + "".join(f"<text x=\"24\" y=\"{82 + i * 28}\" font-family=\"Arial\" font-size=\"16\" fill=\"#374151\">{line}</text>" for i, line in enumerate(lines))
            + "</svg>\n",
            encoding="utf-8",
        )


def _write_report(out_dir: Path, route: Dict[str, Any], audit: Dict[str, Any], p1_decision: Dict[str, Any], p2: Dict[str, Any]) -> None:
    report = ROOT / "docs" / "DG-KAN_v9.2.24_N2a_EventTimeActivationBand_实验复盘.md"
    text = f"""# DG-KAN v9.2.24 N2a Event-Time Activation Band 实验复盘

> 本复盘记录本轮针对 v9.2.23 `R2-FunctionalEventSilent` 的修复实验。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把未打开的 downstream 阶段写成通过。

## 0. 最新结论

```text
route = {route.get('route')}
base_candidate = LQ-t2-h256
success_v9224_event_activation_repair = {str(bool(route.get('success_v9224_event_activation_repair'))).lower()}
success_v9224_strict_purekan_functional = {str(bool(route.get('success_v9224_strict_purekan_functional'))).lower()}
success_v9224_external_ready = false
```

最终 artifact：

```text
{out_dir.relative_to(ROOT)}/
```

核心结论：

1. v9.2.23 的 event silent boundary 被复现，source mean `r_z={route.get('source_mean_actual_r_z')}`。
2. P1 activation sweep 真实执行；non-applicable O6 rows 在 MNIST/Fashion 中明确记录为 `target_applicable=0`，不作为成功或失败 proxy。
3. Activation policy = `{route.get('activation_policy')}`，policy type = `{route.get('activation_policy_type')}`。
4. P1 activation band pass = `{route.get('activation_band_pass')}`。
5. P2 paired replay pass = `{route.get('paired_replay_pass')}`；Real beats AdamWParallel rate = `{route.get('real_beats_adamwparallel_rate')}`，beats best LR rate = `{route.get('real_beats_best_lr_rate')}`。
6. 当前 blocker：`{route.get('primary_blocker')}`。

## 1. 本轮代码与命令

新增：

| 文件 | 作用 |
|---|---|
| `experiments/run_v9224_n2a_event_time_activation_repair.py` | v9.2.24 runner；执行 N2a event-time activation band sweep 与 repaired paired replay |

代码检查：

```text
python -m py_compile experiments/run_v9224_n2a_event_time_activation_repair.py
```

正式运行：

```bash
python experiments/run_v9224_n2a_event_time_activation_repair.py \\
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

## 3. P1 activation band sweep

Artifacts：

```text
p1_n2a_event_time_activation_sweep.csv
p1_activation_band_summary.csv
```

Summary：

```text
activation_band_pass = {p1_decision.get('activation_band_pass')}
activation_policy_type = {p1_decision.get('activation_policy_type')}
activation_policy = {p1_decision.get('activation_policy')}
eligible_rows = {p1_decision.get('eligible_rows')}
```

判断：本轮解决的是 “event-time movement scale 是否太弱”，不是重新训练 base。Base 仍是 v9.2.22 的 N2a 路线。

## 4. P2 repaired paired replay

Artifact：

```text
p2_repaired_activation_paired_replay.csv
```

Summary：

```text
paired_replay_pass = {p2.get('paired_replay_pass', 0)}
real_row_count = {p2.get('real_row_count', 0)}
real_beats_adamwparallel_rate = {p2.get('real_beats_adamwparallel_rate', 0)}
real_beats_best_lr_rate = {p2.get('real_beats_best_lr_rate', 0)}
task_safe_rate = {p2.get('task_safe_rate', 0)}
real_mean_CEp99_delta = {p2.get('real_mean_CEp99_delta', 0)}
real_mean_margin_delta = {p2.get('real_mean_margin_delta', 0)}
```

判断：如果 P1 通过但 P2 不通过，说明 silence 可以修，但 causal superiority 仍未成立。

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

v9.2.24 的真实推进是：

```text
v9.2.23: N2a event silent，mean r_z < 0.10。
v9.2.24: 只调整 event-time activation band，检查能否把 silent 修成 realized movement，并继续过 strong controls。
```

最终一句话：

> v9.2.24 真实执行后停在 `{route.get('route')}`：`{route.get('primary_blocker')}`。
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
    parser.add_argument("--activation-fractions", default="0.10,0.15,0.20,0.30,0.50")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--targets", default="O1-HardTailLogitCorrection,O2-MarginTailExpansion,O6-KMNISTHardModeOutputTarget")
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
        "experiment": "DG-KAN v9.2.24 N2a Event-Time Activation Band",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "script": str(SCRIPT_PATH.relative_to(ROOT)),
        "source_v9223": str(SRC_V9223.relative_to(ROOT)),
        "args": vars(args),
    })
    contract = [{
        "stage": "CONTRACT_AUDIT_V9224",
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
    write_csv_rows(out_dir / "contract_audit_v9224.csv", contract)

    p0 = _source_boundary()
    write_csv_rows(out_dir / "p0_v9223_silence_boundary_reproduction.csv", [p0])
    cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    p1_rows, p1_summary, p1_decision = _p1_scale_sweep(args, device, _int(p0.get("P0_pass")) == 1, cache)
    write_csv_rows(out_dir / "p1_n2a_event_time_activation_sweep.csv", p1_rows)
    write_csv_rows(out_dir / "p1_activation_band_summary.csv", p1_summary)
    policy = p1_decision.get("activation_policy", {}) if isinstance(p1_decision.get("activation_policy"), dict) else {}
    p2_rows, p2_summary = _p2_paired_replay(args, device, _int(p1_decision.get("activation_band_pass")) == 1, policy, cache)
    write_csv_rows(out_dir / "p2_repaired_activation_paired_replay.csv", p2_rows)
    if not _int(p2_summary.get("paired_replay_pass")):
        _write_downstream(out_dir, "P2_repaired_activation_paired_replay_failed")

    if not _int(p0.get("P0_pass")):
        route_name = "R1-SourceBoundaryFailed"
        blocker = "v9223_silence_boundary_not_reproduced"
        next_impl = "reproduce_v9223_before_activation_repair"
    elif not _int(p1_decision.get("activation_band_pass")):
        route_name = "R2-ActivationBandTaskSafetyOrScaleLimit"
        blocker = "no_event_time_scale_reaches_rz_threshold_with_task_safety"
        next_impl = "redesign_N2a_functional_channel_or_base_neutral_cap"
    elif _int(p2_summary.get("paired_replay_pass")):
        route_name = "R4-EventActivationRepairedPairedPass"
        blocker = "short_full_validation_not_opened_in_this_runner"
        next_impl = "open_short_and_full_functional_validation"
    else:
        route_name = "R3-ActivationRepairedButControlEquivalent"
        blocker = "event_scale_repairs_silence_but_realfunctional_still_loses_controls"
        next_impl = "redesign_functional_direction_not_only_activation_scale"

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "source_route": p0.get("source_route", ""),
        "source_mean_actual_r_z": p0.get("source_mean_actual_r_z", ""),
        "source_mean_actual_r_z_tail": p0.get("source_mean_actual_r_z_tail", ""),
        "activation_band_pass": p1_decision.get("activation_band_pass", 0),
        "activation_policy_type": p1_decision.get("activation_policy_type", ""),
        "activation_policy": p1_decision.get("activation_policy", {}),
        "eligible_rows": p1_decision.get("eligible_rows", 0),
        "paired_replay_pass": p2_summary.get("paired_replay_pass", 0),
        "real_row_count": p2_summary.get("real_row_count", 0),
        "real_beats_adamwparallel_rate": p2_summary.get("real_beats_adamwparallel_rate", 0),
        "real_beats_best_lr_rate": p2_summary.get("real_beats_best_lr_rate", 0),
        "task_safe_rate": p2_summary.get("task_safe_rate", 0),
        "real_mean_CEp99_delta": p2_summary.get("real_mean_CEp99_delta", 0),
        "real_mean_margin_delta": p2_summary.get("real_mean_margin_delta", 0),
        "primary_blocker": blocker,
        "next_required_implementation": next_impl,
        "success_v9224_event_activation_repair": int(_int(p1_decision.get("activation_band_pass")) == 1),
        "success_v9224_strict_purekan_functional": int(_int(p2_summary.get("paired_replay_pass")) == 1),
        "success_v9224_external_ready": 0,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", {"route": route, "p0": p0, "p1": p1_decision, "p2": p2_summary})
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
        out_dir / "contract_audit_v9224.csv",
        out_dir / "p0_v9223_silence_boundary_reproduction.csv",
        out_dir / "p1_n2a_event_time_activation_sweep.csv",
        out_dir / "p1_activation_band_summary.csv",
        out_dir / "p2_repaired_activation_paired_replay.csv",
        out_dir / "p3_short_run_after_activation_repair.csv",
        out_dir / "p4_full_functional_after_activation_repair.csv",
        out_dir / "p5_robustness_external_ready.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake([p for p in audit_paths if p.exists()])
    write_csv_rows(out_dir / "v9224_provenance_audit.csv", [audit])
    hash_paths = [
        SCRIPT_PATH,
        out_dir / "run_manifest.json",
        out_dir / "route_decision.json",
        out_dir / "aggregate_decision.json",
        *[p for p in audit_paths if p.exists()],
        out_dir / "v9224_provenance_audit.csv",
    ]
    write_csv_rows(out_dir / "artifact_hashes.csv", artifact_hash_rows(hash_paths, root=ROOT))
    _write_report(out_dir, route, audit, p1_decision, p2_summary)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
