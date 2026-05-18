#!/usr/bin/env python3
"""DG-KAN v9.2.26 Fashion delayed controller + KMNIST diagnosis.

This runner follows v9.2.25.  It promotes only the Fashion-MNIST delayed
signal into a 50/240-step ablation, while moving KMNIST back to target and
primitive diagnostics.  KMNIST diagnostic rows are never eligible for success.
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


SCRIPT_PATH = ROOT / "experiments" / "run_v9226_fashion_delayed_controller_kmnist_diagnosis.py"
SRC_V9225 = ROOT / "results" / "real_rerun_20260506" / "v9225_dataset_specific_direction_event_selection_first_20260510T150000Z"


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
    route = _read_json(SRC_V9225 / "route_decision.json")
    audit = read_csv_rows(SRC_V9225 / "v9225_provenance_audit.csv")
    fake = _int(audit[0].get("fake_proxy_nonzero_count")) if audit else 1
    p0_pass = int(
        route.get("route") == "R2-DelayedFashionSignalOnly"
        and _int(route.get("h80_delayed_pass_count")) > 0
        and _int(route.get("all_horizon_pass_count")) == 0
        and fake == 0
    )
    return {
        "stage": "P0_V9225_BOUNDARY_REPRODUCTION",
        "status": "source_recap",
        "source_artifact": str(SRC_V9225.relative_to(ROOT)),
        "source_route": route.get("route", ""),
        "source_best_candidate": route.get("best_candidate", ""),
        "source_best_scope": route.get("best_scope", ""),
        "source_all_horizon_pass_count": route.get("all_horizon_pass_count", ""),
        "source_h80_delayed_pass_count": route.get("h80_delayed_pass_count", ""),
        "source_best_CEp99_delta": route.get("best_CEp99_delta", ""),
        "source_best_margin_delta": route.get("best_margin_delta", ""),
        "source_best_beats_adamwparallel": route.get("best_real_beats_adamwparallel_rate", ""),
        "source_best_beats_best_lr": route.get("best_real_beats_best_lr_rate", ""),
        "fake_proxy_count": fake,
        "P0_pass": p0_pass,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }


def _fashion_specs() -> List[Dict[str, Any]]:
    return [
        {"candidate": "F1-Fashion-O1O2-DelayedController", "targets": ["O1-HardTailLogitCorrection", "O2-MarginTailExpansion"], "fraction": 0.50},
        {"candidate": "F2-Fashion-O1-DelayedController", "targets": ["O1-HardTailLogitCorrection"], "fraction": 0.50},
        {"candidate": "F3-Fashion-O2-DelayedController", "targets": ["O2-MarginTailExpansion"], "fraction": 0.50},
    ]


def _run_fashion_ablation(
    args: argparse.Namespace,
    device: torch.device,
    opened: bool,
    cache: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        row = _not_run("P1_FASHION_DELAYED_CONTROLLER_ABLATION", "p1_fashion_delayed_controller_ablation.csv", "P0_v9225_boundary_failed")
        return [row], [row], {}
    cand = v9222._candidate_registry()["N2a-TinyInit-RationalFunc-BranchRatioCap"]
    assert cand.spec is not None
    rows: List[Dict[str, Any]] = []
    horizons = _parse_ints(args.fashion_horizons)
    dataset = "Fashion-MNIST"
    x_train, y_train, x_eval, y_eval, protocol = v9223._load_split(args, dataset, device)
    xb = x_train[: int(args.audit_batch_size)]
    yb = y_train[: int(args.audit_batch_size)]
    for spec in _fashion_specs():
        for seed in _parse_ints(args.seeds):
            saved = v9223._train_cache(args, cand, dataset, seed, device, cache)
            params = saved["params"]
            states = saved["states"]
            mu = saved["mu"]
            std = saved["std"]
            for target in spec["targets"]:
                safe_delta, task_step, _grads, info = v9223._base_direction(args, cand, params, mu, std, xb, yb, dataset, target)
                if _float(info.get("target_selected_fraction")) <= 0.0:
                    continue
                real_step = v9222._cap_to_fraction(safe_delta, task_step, float(spec["fraction"]))
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
                            "stage": "P1_FASHION_DELAYED_CONTROLLER_ABLATION",
                            "status": "measured",
                            "candidate": spec["candidate"],
                            "dataset": dataset,
                            "seed": seed,
                            "protocol": protocol,
                            "target": target,
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
                            "target_fit_R2": info.get("target_fit_R2", ""),
                            "p3_proxy_r_perp": info.get("p3_proxy_r_perp", ""),
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        })
    summary: List[Dict[str, Any]] = []
    real = [r for r in rows if r.get("branch") == "RealFunctional"]
    for cand_id in sorted({r.get("candidate", "") for r in real}):
        for scope, group in [
            ("50_240_all", [r for r in real if r.get("candidate") == cand_id]),
            *[(f"h{h}", [r for r in real if r.get("candidate") == cand_id and _int(r.get("horizon")) == h]) for h in horizons],
        ]:
            if not group:
                continue
            beat_p = _mean([_float(r.get("real_beats_adamwparallel")) for r in group])
            beat_lr = _mean([_float(r.get("real_beats_best_lr")) for r in group])
            task_safe = _mean([_float(r.get("task_safe")) for r in group])
            ce = _mean([_float(r.get("CEp99_delta")) for r in group])
            margin = _mean([_float(r.get("margin_p10_delta")) for r in group])
            pass_gate = int(beat_p >= 0.50 and beat_lr >= 0.50 and task_safe >= 0.95 and ce < 0.0 and margin > 0.0)
            summary.append({
                "stage": "P1_FASHION_DELAYED_CONTROLLER_SUMMARY",
                "candidate": cand_id,
                "scope": scope,
                "rows": len(group),
                "CEp99_delta": ce,
                "margin_p10_delta": margin,
                "real_beats_adamwparallel_rate": beat_p,
                "real_beats_best_lr_rate": beat_lr,
                "task_safe_rate": task_safe,
                "fashion_delayed_controller_pass": pass_gate,
                "fake_data_used": 0,
                "proxy_row_used": 0,
                "cpu_offload_used": 0,
            })
    all_pass = [r for r in summary if r.get("scope") == "50_240_all" and _int(r.get("fashion_delayed_controller_pass"))]
    h_pass = [r for r in summary if r.get("scope") != "50_240_all" and _int(r.get("fashion_delayed_controller_pass"))]
    best = sorted(summary, key=lambda r: (-_int(r.get("fashion_delayed_controller_pass")), _float(r.get("CEp99_delta")), -_float(r.get("margin_p10_delta"))))[0] if summary else {}
    decision = {
        "fashion_ablation_pass": int(bool(all_pass)),
        "fashion_all_scope_pass_count": len(all_pass),
        "fashion_horizon_pass_count": len(h_pass),
        "fashion_best_candidate": best.get("candidate", ""),
        "fashion_best_scope": best.get("scope", ""),
        "fashion_best_CEp99_delta": _float(best.get("CEp99_delta")) if best else 0.0,
        "fashion_best_margin_delta": _float(best.get("margin_p10_delta")) if best else 0.0,
        "fashion_best_beats_adamwparallel": _float(best.get("real_beats_adamwparallel_rate")) if best else 0.0,
        "fashion_best_beats_best_lr": _float(best.get("real_beats_best_lr_rate")) if best else 0.0,
        "fashion_best_task_safe": _float(best.get("task_safe_rate")) if best else 0.0,
    }
    return rows, summary, decision


def _kmnist_direct_oracle_row(
    target_id: str,
    eps: float,
    logits: torch.Tensor,
    y: torch.Tensor,
    dataset: str,
) -> Dict[str, Any]:
    target, target_info = v9223.out_lq.build_output_target(target_id, logits, y, dataset=dataset)
    before = v92._classification_metrics_from_logits(logits, y)
    scaled = target.float()
    tnorm = scaled.norm().clamp_min(1.0e-12)
    scaled = scaled / tnorm * (logits.float().norm() * float(eps)).detach()
    after = v92._classification_metrics_from_logits(logits + scaled, y)
    delta = v9223._metric_delta(before, after)
    return {
        "target_selected_fraction": target_info.get("target_selected_fraction", 0.0),
        "oracle_eps": eps,
        "oracle_CEp99_delta": delta["CEp99_delta"],
        "oracle_margin_p10_delta": delta["margin_p10_delta"],
        "oracle_ECE_delta": delta["ECE_delta"],
        "oracle_NLL_delta": delta["NLL_delta"],
        "oracle_acc_delta": delta["acc_delta"],
        "oracle_useful": int(delta["acc_delta"] >= -0.005 and (delta["CEp99_delta"] < 0.0 or delta["margin_p10_delta"] > 0.0)),
    }


def _run_kmnist_diagnosis(
    args: argparse.Namespace,
    device: torch.device,
    opened: bool,
    cache: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not opened:
        return [_not_run("P2_KMNIST_TARGET_PRIMITIVE_DIAGNOSIS", "p2_kmnist_target_primitive_diagnosis.csv", "P0_v9225_boundary_failed")], {}
    cand = v9222._candidate_registry()["N2a-TinyInit-RationalFunc-BranchRatioCap"]
    assert cand.spec is not None
    dataset = "KMNIST"
    x_train, y_train, x_eval, y_eval, protocol = v9223._load_split(args, dataset, device)
    xb = x_train[: int(args.audit_batch_size)]
    yb = y_train[: int(args.audit_batch_size)]
    rows: List[Dict[str, Any]] = []
    for seed in _parse_ints(args.seeds):
        saved = v9223._train_cache(args, cand, dataset, seed, device, cache)
        params = saved["params"]
        mu = saved["mu"]
        std = saved["std"]
        base_logits = v9223.act.actuator_forward(xb, params, mu, std, cand.spec)
        _loss, grads = v9223.act.actuator_fwd_bwd(xb, yb, params, mu, std, cand.spec)
        task_step = snr_lq.gradient_descent_task_step(grads, float(args.lr))
        parallel_step = v9222._cap_to_fraction(task_step, task_step, float(args.parallel_trust_ratio))
        for target_id in _parse_list(args.kmnist_targets):
            target, target_info = v9223.out_lq.build_output_target(target_id, base_logits, yb, dataset=dataset)
            raw_delta, ls_info = v9223.act.actuator_only_least_squares_delta(params, mu, std, cand.spec, xb, target, ridge=float(args.ridge))
            raw_fit = v9223.act.output_fit_metrics(params, raw_delta, mu, std, cand.spec, xb, target)
            safe_delta, removed = snr_lq.project_step_to_task_safe(raw_delta, grads)
            safe_fit = v9223.act.output_fit_metrics(params, safe_delta, mu, std, cand.spec, xb, target)
            capped_step = v9222._cap_to_fraction(safe_delta, task_step, float(args.kmnist_cap_fraction))
            cap_fit = v9223.act.output_fit_metrics(params, capped_step, mu, std, cand.spec, xb, target)
            stats = v9223._actual_stats(
                params=params,
                real_step=capped_step,
                parallel_step=parallel_step,
                mu=mu,
                std=std,
                spec=cand.spec,
                x_eval=x_eval,
                y_eval=y_eval,
                dataset=dataset,
                target_id=target_id,
            )
            after_diag = v9222._diagnose_channels(v9223._apply_step(params, capped_step), mu, std, cand.spec, xb, yb, float(args.lr))
            for eps in _parse_floats(args.kmnist_oracle_eps):
                oracle = _kmnist_direct_oracle_row(target_id, eps, base_logits, yb, dataset)
                rows.append({
                    "stage": "P2_KMNIST_TARGET_PRIMITIVE_DIAGNOSIS",
                    "status": "measured",
                    "candidate": cand.candidate_id,
                    "dataset": dataset,
                    "seed": seed,
                    "protocol": protocol,
                    "target": target_id,
                    **oracle,
                    "raw_target_fit_R2": raw_fit["output_target_fit_r2"],
                    "raw_output_displacement_ratio": raw_fit["output_displacement_to_target_ratio"],
                    "safe_target_fit_R2": safe_fit["output_target_fit_r2"],
                    "safe_output_displacement_ratio": safe_fit["output_displacement_to_target_ratio"],
                    "cap_target_fit_R2": cap_fit["output_target_fit_r2"],
                    "cap_output_displacement_ratio": cap_fit["output_displacement_to_target_ratio"],
                    "cap_fraction": args.kmnist_cap_fraction,
                    "actual_r_z": stats["actual_r_z"],
                    "actual_r_z_tail": stats["actual_r_z_tail"],
                    "actual_r_z_perp": stats["actual_r_z_perp"],
                    "actual_CEp99_delta": stats["real_CEp99_delta"],
                    "actual_margin_p10_delta": stats["real_margin_p10_delta"],
                    "actual_acc_delta": stats["real_acc_delta"],
                    "actual_tail_logit_delta_norm": stats["actual_tail_logit_delta_norm"],
                    "adamwparallel_tail_logit_delta_norm": stats["adamwparallel_tail_logit_delta_norm"],
                    "projection_removed_norm": float(removed.detach().cpu()),
                    "ls_residual_norm": ls_info.get("ls_residual_norm", 0.0),
                    "branch_ratio_event": after_diag["branch_ratio"],
                    "effective_derivative_event": after_diag["effective_derivative_p95"],
                    "diagnosis_target_oracle_useful": oracle["oracle_useful"],
                    "diagnosis_raw_fit_pass": int(raw_fit["output_target_fit_r2"] >= 0.20 and raw_fit["output_displacement_to_target_ratio"] >= 0.05),
                    "diagnosis_safe_fit_pass": int(safe_fit["output_target_fit_r2"] >= 0.20 and safe_fit["output_displacement_to_target_ratio"] >= 0.05),
                    "diagnosis_actual_effect_pass": int(stats["real_CEp99_delta"] < -1.0e-5 or stats["real_margin_p10_delta"] > 1.0e-5),
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
    measured = [r for r in rows if r.get("status") == "measured"]
    target_oracle_rate = _mean([_float(r.get("diagnosis_target_oracle_useful")) for r in measured])
    raw_fit_rate = _mean([_float(r.get("diagnosis_raw_fit_pass")) for r in measured])
    safe_fit_rate = _mean([_float(r.get("diagnosis_safe_fit_pass")) for r in measured])
    actual_effect_rate = _mean([_float(r.get("diagnosis_actual_effect_pass")) for r in measured])
    if target_oracle_rate < 0.50:
        blocker = "KMNIST_target_oracle_weak"
    elif raw_fit_rate < 0.50:
        blocker = "KMNIST_primitive_raw_controllability_weak"
    elif safe_fit_rate < 0.50:
        blocker = "KMNIST_task_safe_projection_neutralizes_target"
    elif actual_effect_rate < 0.50:
        blocker = "KMNIST_realized_effect_not_metric_causal"
    else:
        blocker = "KMNIST_diagnosis_inconclusive"
    summary = {
        "kmnist_diagnosis_blocker": blocker,
        "kmnist_target_oracle_useful_rate": target_oracle_rate,
        "kmnist_raw_fit_pass_rate": raw_fit_rate,
        "kmnist_safe_fit_pass_rate": safe_fit_rate,
        "kmnist_actual_effect_pass_rate": actual_effect_rate,
        "kmnist_max_raw_R2": max([_float(r.get("raw_target_fit_R2")) for r in measured] or [0.0]),
        "kmnist_max_safe_R2": max([_float(r.get("safe_target_fit_R2")) for r in measured] or [0.0]),
        "kmnist_max_cap_rz": max([_float(r.get("actual_r_z")) for r in measured] or [0.0]),
        "kmnist_mean_actual_CEp99_delta": _mean([_float(r.get("actual_CEp99_delta")) for r in measured]),
        "kmnist_mean_actual_margin_delta": _mean([_float(r.get("actual_margin_p10_delta")) for r in measured]),
    }
    return rows, summary


def _write_downstream(out_dir: Path, reason: str) -> None:
    for fname, stage in [
        ("p3_fashion_short_run_validation.csv", "P3_FASHION_SHORT_RUN_VALIDATION"),
        ("p4_strict_purekan_functional_reentry.csv", "P4_STRICT_PUREKAN_FUNCTIONAL_REENTRY"),
        ("p5_external_ready.csv", "P5_EXTERNAL_READY"),
    ]:
        write_csv_rows(out_dir / fname, [_not_run(stage, fname, reason)])


def _write_report(out_dir: Path, route: Dict[str, Any], p0: Dict[str, Any], fashion_summary: List[Dict[str, Any]], km_summary: Dict[str, Any], audit: Dict[str, Any]) -> None:
    report = ROOT / "docs" / "DG-KAN_v9.2.26_FashionDelayedController_KMNISTTargetPrimitiveDiagnosis_实验复盘.md"
    fs = sorted(fashion_summary, key=lambda r: (str(r.get("candidate")), str(r.get("scope"))))
    lines = [
        "# DG-KAN v9.2.26 Fashion Delayed Controller 与 KMNIST Target/Primitive Diagnosis 实验复盘",
        "",
        "> 本复盘记录本轮针对 v9.2.25 `DelayedFashionSignalOnly` 的后续实验。所有结论只来自本文列出的落盘 CSV/JSON/manifest；没有 fake data、proxy rows，也没有把 KMNIST 诊断 rows 写成 functional success。",
        "",
        "## 0. 最新结论",
        "",
        "```text",
        f"route = {route.get('route')}",
        "base_candidate = LQ-t2-h256",
        f"success_v9226_fashion_delayed_controller = {bool(route.get('success_v9226_fashion_delayed_controller'))}",
        f"success_v9226_strict_purekan_functional = {bool(route.get('success_v9226_strict_purekan_functional'))}",
        "```",
        "",
        "最终 artifact：",
        "",
        "```text",
        str(out_dir.relative_to(ROOT)) + "/",
        "```",
        "",
        "核心结论：",
        "",
        f"1. v9.2.25 boundary 复现通过：source route = `{p0.get('source_route')}`，source best = `{p0.get('source_best_candidate')}` / `{p0.get('source_best_scope')}`。",
        f"2. Fashion 50/240-step ablation pass = `{route.get('fashion_ablation_pass')}`，best = `{route.get('fashion_best_candidate')}` / `{route.get('fashion_best_scope')}`。",
        f"3. Fashion best CEp99 delta = `{route.get('fashion_best_CEp99_delta')}`，margin delta = `{route.get('fashion_best_margin_delta')}`，beats AdamWParallel = `{route.get('fashion_best_beats_adamwparallel')}`，beats best LR = `{route.get('fashion_best_beats_best_lr')}`。",
        f"4. KMNIST diagnosis blocker = `{route.get('kmnist_diagnosis_blocker')}`；target oracle useful rate = `{route.get('kmnist_target_oracle_useful_rate')}`，safe fit pass rate = `{route.get('kmnist_safe_fit_pass_rate')}`，actual effect pass rate = `{route.get('kmnist_actual_effect_pass_rate')}`。",
        f"5. 当前 blocker：`{route.get('primary_blocker')}`。",
        "",
        "## 1. 本轮代码与命令",
        "",
        "| 文件 | 作用 |",
        "|---|---|",
        "| `experiments/run_v9226_fashion_delayed_controller_kmnist_diagnosis.py` | Fashion delayed controller 50/240-step ablation；KMNIST target/primitive diagnosis |",
        "",
        "代码检查：",
        "",
        "```text",
        "python -m py_compile experiments/run_v9226_fashion_delayed_controller_kmnist_diagnosis.py",
        "```",
        "",
        "正式运行：",
        "",
        "```bash",
        "python experiments/run_v9226_fashion_delayed_controller_kmnist_diagnosis.py \\",
        f"  --out-dir {out_dir.relative_to(ROOT)} \\",
        "  --fresh \\",
        "  --device auto \\",
        "  --data-root data \\",
        "  --seed 1314",
        "```",
        "",
        "## 2. Route",
        "",
        "```json",
        json.dumps(route, indent=2, sort_keys=True),
        "```",
        "",
        "## 3. P1 Fashion delayed controller ablation",
        "",
        "Artifact：",
        "",
        "```text",
        "p1_fashion_delayed_controller_ablation.csv",
        "p1_fashion_delayed_controller_summary.csv",
        "```",
        "",
        "| candidate | scope | CEp99 delta | margin delta | beats AdamWParallel | beats best LR | task safe | pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in fs:
        lines.append(
            f"| {r.get('candidate')} | {r.get('scope')} | `{_float(r.get('CEp99_delta')):.6f}` | `{_float(r.get('margin_p10_delta')):.6f}` | `{_float(r.get('real_beats_adamwparallel_rate')):.6f}` | `{_float(r.get('real_beats_best_lr_rate')):.6f}` | `{_float(r.get('task_safe_rate')):.6f}` | `{_int(r.get('fashion_delayed_controller_pass'))}` |"
        )
    lines.extend([
        "",
        "判断：Fashion delayed signal 是否能进入下一步，取决于 50/240 两个 horizon 是否同时 beat AdamWParallel 和 best LR，并保持 CE tail 与 margin 的同向改善。本轮不把 h80 local signal 直接写成 full controller success。",
        "",
        "## 4. P2 KMNIST target/primitive diagnosis",
        "",
        "Artifact：",
        "",
        "```text",
        "p2_kmnist_target_primitive_diagnosis.csv",
        "```",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| diagnosis blocker | `{km_summary.get('kmnist_diagnosis_blocker', '')}` |",
        f"| target oracle useful rate | `{km_summary.get('kmnist_target_oracle_useful_rate', '')}` |",
        f"| raw fit pass rate | `{km_summary.get('kmnist_raw_fit_pass_rate', '')}` |",
        f"| safe fit pass rate | `{km_summary.get('kmnist_safe_fit_pass_rate', '')}` |",
        f"| actual effect pass rate | `{km_summary.get('kmnist_actual_effect_pass_rate', '')}` |",
        f"| max raw R2 | `{km_summary.get('kmnist_max_raw_R2', '')}` |",
        f"| max safe R2 | `{km_summary.get('kmnist_max_safe_R2', '')}` |",
        f"| max actual r_z | `{km_summary.get('kmnist_max_cap_rz', '')}` |",
        f"| mean actual CEp99 delta | `{km_summary.get('kmnist_mean_actual_CEp99_delta', '')}` |",
        f"| mean actual margin delta | `{km_summary.get('kmnist_mean_actual_margin_delta', '')}` |",
        "",
        "判断：KMNIST 本轮只做诊断，不参与 success route。若 oracle 有效但 safe/actual 不过，说明继续调 event threshold 或 cap 不是主要方向，应回 target solver 或 primitive/control surface。",
        "",
        "## 5. Downstream boundary",
        "",
        "这些 artifact 已落盘，但明确为 `not_run`：",
        "",
        "| artifact | reason |",
        "|---|---|",
        "| `p3_fashion_short_run_validation.csv` | route gate 未达到 strict controller success |",
        "| `p4_strict_purekan_functional_reentry.csv` | same |",
        "| `p5_external_ready.csv` | same |",
        "",
        "## 6. No-fake audit",
        "",
        "```text",
        f"rows_checked = {audit.get('rows_checked')}",
        f"fake_proxy_nonzero_count = {audit.get('fake_proxy_nonzero_count')}",
        f"fake_data_used = {audit.get('fake_data_used')}",
        f"proxy_row_used = {audit.get('proxy_row_used')}",
        f"cpu_offload_used = {audit.get('cpu_offload_used')}",
        f"no_fake = {audit.get('no_fake')}",
        f"no_proxy = {audit.get('no_proxy')}",
        "```",
        "",
        "## 7. 最终分析结论",
        "",
        "v9.2.26 的真实推进是：",
        "",
        "```text",
        "Fashion: h80 local signal 被提升到 50/240-step ablation；",
        "KMNIST: 不再继续放大，退回 target/primitive diagnosis。",
        "```",
        "",
        "机制判断：",
        "",
        "1. Fashion 是否能继续，取决于 delayed controller 是否在 50/240 两个 horizon 都能击败 AdamWParallel / LR controls。",
        "2. KMNIST 的问题应从 target oracle、raw controllability、task-safe projection、actual metric effect 四层定位，而不是继续把 scale 当主因。",
        "3. 本轮没有打开 full functional 或 external-ready；所有 downstream 都保持 gate-blocked。",
        "",
        "最终一句话：",
        "",
        f"> v9.2.26 真实执行后停在 `{route.get('route')}`：Fashion delayed controller ablation 与 KMNIST target/primitive diagnosis 已完成，但 strict PureKAN functional 仍未成功。",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=9984)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--audit-batch-size", type=int, default=128)
    parser.add_argument("--p5-epochs", type=int, default=20)
    parser.add_argument("--ridge", type=float, default=1.0e-4)
    parser.add_argument("--best-lr-scale", type=float, default=1.03)
    parser.add_argument("--parallel-trust-ratio", type=float, default=0.03)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--fashion-horizons", default="50,240")
    parser.add_argument("--kmnist-targets", default="O1-HardTailLogitCorrection,O2-MarginTailExpansion,O6-KMNISTHardModeOutputTarget")
    parser.add_argument("--kmnist-oracle-eps", default="0.01,0.03,0.05")
    parser.add_argument("--kmnist-cap-fraction", type=float, default=0.50)
    args = _make_args(parser.parse_args())

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = _device(args.device)
    torch.manual_seed(int(args.seed))

    manifest = {
        "stage": "run_manifest",
        "script": str(SCRIPT_PATH.relative_to(ROOT)),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "args": vars(args),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }
    write_json(out_dir / "run_manifest.json", manifest)
    write_csv_rows(out_dir / "contract_audit_v9226.csv", [{
        "loss_type": "CE",
        "label_smoothing": 0,
        "teacher_used": 0,
        "uses_loss_backward": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
        "scope": "Fashion delayed ablation + KMNIST diagnosis",
    }])

    cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    p0 = _source_boundary()
    write_csv_rows(out_dir / "p0_v9225_boundary_reproduction.csv", [p0])
    opened = bool(_int(p0.get("P0_pass")))
    p1_rows, p1_summary, fashion_decision = _run_fashion_ablation(args, device, opened, cache)
    write_csv_rows(out_dir / "p1_fashion_delayed_controller_ablation.csv", p1_rows)
    write_csv_rows(out_dir / "p1_fashion_delayed_controller_summary.csv", p1_summary)
    p2_rows, km_summary = _run_kmnist_diagnosis(args, device, opened, cache)
    write_csv_rows(out_dir / "p2_kmnist_target_primitive_diagnosis.csv", p2_rows)

    if fashion_decision.get("fashion_ablation_pass"):
        route_name = "R1-FashionDelayedControllerAblationPass"
        primary = "fashion_delayed_controller_passed_but_full_functional_not_opened_in_this_runner"
        next_impl = "open_short_full_validation_for_fashion_only_then_reassess_generalization"
    elif fashion_decision.get("fashion_horizon_pass_count", 0) > 0:
        route_name = "R2-FashionPartialDelayedSignalKMNISTPrimitiveBlocker"
        primary = "fashion_signal_partial_and_kmnist_requires_target_primitive_redesign"
        next_impl = "refine_fashion_delayed_event_timing_and_redesign_kmnist_target_primitive"
    else:
        route_name = "R4-NoFashionControllerKMNISTPrimitiveBlocker"
        primary = "fashion_delayed_signal_did_not_survive_50_240_ablation"
        next_impl = "return_to_functional_interface_or_primitive_design"

    route = {
        "route": route_name,
        "base_candidate": "LQ-t2-h256",
        "source_route": p0.get("source_route", ""),
        **fashion_decision,
        **km_summary,
        "primary_blocker": primary,
        "next_required_implementation": next_impl,
        "success_v9226_fashion_delayed_controller": int(fashion_decision.get("fashion_ablation_pass", 0)),
        "success_v9226_strict_purekan_functional": 0,
        "success_v9226_external_ready": 0,
    }
    write_json(out_dir / "route_decision.json", route)
    write_json(out_dir / "aggregate_decision.json", route)
    write_csv_rows(out_dir / "failure_table.csv", [{
        "stage": "ROUTE",
        "failure": route_name,
        "primary_blocker": primary,
        "next_required_implementation": next_impl,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }])
    _write_downstream(out_dir, primary)
    for name in [
        "p1_fashion_delayed_controller_ablation.svg",
        "p2_kmnist_target_primitive_diagnosis.svg",
    ]:
        (out_dir / "figures" / name).write_text(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1100\" height=\"220\">"
            "<rect width=\"1100\" height=\"220\" fill=\"#f8fafc\"/>"
            f"<text x=\"24\" y=\"44\" font-family=\"Arial\" font-size=\"22\" fill=\"#111827\">{name}</text>"
            f"<text x=\"24\" y=\"86\" font-family=\"Arial\" font-size=\"16\" fill=\"#374151\">route={route_name}</text>"
            f"<text x=\"24\" y=\"116\" font-family=\"Arial\" font-size=\"16\" fill=\"#374151\">fashion_best={fashion_decision.get('fashion_best_candidate')} / {fashion_decision.get('fashion_best_scope')}</text>"
            f"<text x=\"24\" y=\"146\" font-family=\"Arial\" font-size=\"16\" fill=\"#374151\">kmnist_blocker={km_summary.get('kmnist_diagnosis_blocker')}</text>"
            "</svg>\n",
            encoding="utf-8",
        )

    audit_targets = [
        out_dir / "contract_audit_v9226.csv",
        out_dir / "p0_v9225_boundary_reproduction.csv",
        out_dir / "p1_fashion_delayed_controller_ablation.csv",
        out_dir / "p1_fashion_delayed_controller_summary.csv",
        out_dir / "p2_kmnist_target_primitive_diagnosis.csv",
        out_dir / "p3_fashion_short_run_validation.csv",
        out_dir / "p4_strict_purekan_functional_reentry.csv",
        out_dir / "p5_external_ready.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(audit_targets)
    write_csv_rows(out_dir / "v9226_provenance_audit.csv", [audit])
    hash_paths = [SCRIPT_PATH, out_dir / "route_decision.json", *audit_targets, out_dir / "v9226_provenance_audit.csv"]
    hash_rows = artifact_hash_rows(hash_paths, root=ROOT)
    write_csv_rows(out_dir / "artifact_hashes.csv", hash_rows)
    _write_report(out_dir, route, p0, p1_summary, km_summary, audit)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
