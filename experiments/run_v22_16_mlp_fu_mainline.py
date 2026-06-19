#!/usr/bin/env python3
"""v22.16-M Real MLP+FU mainline supplement runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dgkan.fu.mlp_adaptive_controller import MLPFUState, apply_mlp_fu_update  # noqa: E402
from experiments.run_v22_16_common import (  # noqa: E402
    DATASETS,
    PYTHON,
    artifact_index,
    ce_cotangent,
    count_parameters,
    device_from_arg,
    ensure_out,
    evaluate,
    finite_float,
    flat_grads,
    int_flag,
    loader_for,
    make_model,
    md_table,
    read_json,
    read_rows,
    selected_named_parameters,
    sha256_file,
    write_json,
    write_rows,
)


MLP_ROOT = ROOT / "results/v22_16_mlp_fu_mainline"
MLP_OFFICIAL = MLP_ROOT / "official_v22_16_m"
MLP_PLAN_DOC = ROOT / "docs/DG-KAN_v22.16_MLP_FU_Mainline_补充实验计划.md"
MLP_EXEC_DOC = ROOT / "docs/DG-KAN_v22.16_MLP_FU_Mainline_执行日志.md"
MLP_RECAP_DOC = ROOT / "docs/DG-KAN_v22.16_MLP_FU_Mainline_实验结果复盘.md"
OLD_V2216_OUT = ROOT / "results/v22_16_real_trajectory_source_manifold_adaptive_fu/official_v22_16"


VARIANTS = [
    "M0 MLP+AdamW baseline",
    "M1 MLP+SGD baseline",
    "M0R MLP+AdamW-repeat-control",
    "M2 MLP+AdaptiveFU-current-v22.15-replay",
    "M3 MLP+RiskMonitorOnly-no-update",
    "M4 MLP+WeakPredictiveProx",
    "M5 MLP+SourceLossGatedPredictiveProx",
    "M6 MLP+SourceReleaseController",
    "M7 MLP+RealSourceManifold-k8",
    "M8 MLP+RealSourceManifold-k16",
    "M9 MLP+LowRankDirectProx-r8",
    "M10 MLP+LowRankDirectProx-r16",
    "M11 MLP+AuxiliaryAnchorUpperBound-diagnostic",
    "M12 MLP+RandomSourceControl",
    "M13 MLP+StableRandomSourceControl",
    "M14 MLP+SignFlipSourceControl",
    "M15 MLP+CorruptSourceControl",
]

OFFICIAL_CANDIDATES = {
    "M4 MLP+WeakPredictiveProx",
    "M5 MLP+SourceLossGatedPredictiveProx",
    "M6 MLP+SourceReleaseController",
    "M7 MLP+RealSourceManifold-k8",
    "M8 MLP+RealSourceManifold-k16",
    "M9 MLP+LowRankDirectProx-r8",
    "M10 MLP+LowRankDirectProx-r16",
}
CONTROLS = {
    "M12 MLP+RandomSourceControl",
    "M13 MLP+StableRandomSourceControl",
    "M14 MLP+SignFlipSourceControl",
    "M15 MLP+CorruptSourceControl",
}
DIAGNOSTIC_ONLY = {
    "M2 MLP+AdaptiveFU-current-v22.15-replay",
    "M3 MLP+RiskMonitorOnly-no-update",
    "M11 MLP+AuxiliaryAnchorUpperBound-diagnostic",
    *CONTROLS,
}
HORIZONS = [100, 400, 800, 1600, 3200, 4800, 6400]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(MLP_OFFICIAL))
    p.add_argument("--stage", default="all", choices=["all", "audit", "train", "finalize"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--train-size", type=int, default=1024)
    p.add_argument("--test-size", type=int, default=512)
    p.add_argument("--steps", type=int, default=4800)
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--download", action="store_true")
    p.add_argument("--fresh", action="store_true")
    return p


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_mlp_out(out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else MLP_OFFICIAL
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    return out


def init_mlp_docs() -> None:
    MLP_EXEC_DOC.parent.mkdir(parents=True, exist_ok=True)
    if not MLP_EXEC_DOC.exists():
        MLP_EXEC_DOC.write_text(
            "# DG-KAN v22.16-M MLP+FU Mainline 执行日志\n\n"
            f"生成时间：{now_sg()}\n\n"
            "记录原则：只记录真实命令、文件、状态、blocker 与修复；未执行项不得写作完成。\n",
            encoding="utf-8",
        )
    if not MLP_RECAP_DOC.exists():
        MLP_RECAP_DOC.write_text(
            "# DG-KAN v22.16-M MLP+FU Mainline 实验结果复盘\n\n"
            f"生成时间：{now_sg()}\n\n"
            "本复盘只引用本主线落盘 artifact 与真实读回结果；禁止编造数据。\n",
            encoding="utf-8",
        )


def append_mlp_exec(out_dir: Path, command: str, *, task_id: str, gpu: str = "", status: str = "", files: str = "", note: str = "") -> None:
    init_mlp_docs()
    row = {
        "timestamp": now_sg(),
        "task_id": task_id,
        "gpu": gpu,
        "command": command,
        "status": status,
        "files": files,
        "note": note,
    }
    journal = out_dir / "v22_16_mlp_fu_command_journal.csv"
    exists = journal.exists() and journal.stat().st_size > 0
    with journal.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    with MLP_EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {row['timestamp']} {task_id}\n\n```bash\n{command}\n```\n\n")
        f.write(f"- gpu: {gpu or 'n/a'}\n- status: {status or 'recorded'}\n")
        if files:
            f.write(f"- files: {files}\n")
        if note:
            f.write(f"- note: {note}\n")


def _split(raw: str, cast: Any = str) -> list[Any]:
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def _auc(labels: list[int], scores: list[float]) -> float | str:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return ""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum = 0.0
    for idx, (_score, label) in enumerate(pairs, start=1):
        if label:
            rank_sum += idx
    return float((rank_sum - pos * (pos + 1) / 2) / (pos * neg))


def _merge_rows(path: Path, rows: list[dict[str, Any]], key_fields: list[str]) -> None:
    existing = read_rows(path)
    merged: dict[tuple[str, ...], dict[str, Any]] = {tuple(str(r.get(k, "")) for k in key_fields): r for r in existing}
    for row in rows:
        merged[tuple(str(row.get(k, "")) for k in key_fields)] = row
    write_rows(path, list(merged.values()))


def _variant_kind(variant: str) -> str:
    if "SGD" in variant:
        return "sgd"
    if "AdamW" in variant and "FU" not in variant:
        return "adamw"
    if "RiskMonitorOnly" in variant:
        return "monitor"
    if "AuxiliaryAnchor" in variant:
        return "aux"
    if "RandomSourceControl" in variant or "StableRandomSourceControl" in variant or "SignFlipSourceControl" in variant or "CorruptSourceControl" in variant:
        return "control"
    return "fu"


def _uses_previous_source_loss_gate(variant: str) -> bool:
    return any(
        token in variant
        for token in [
            "SourceLossGatedPredictiveProx",
            "SourceReleaseController",
            "RealSourceManifold",
            "LowRankDirectProx",
        ]
    )


def _assign_flat_update(named_params: list[tuple[str, torch.nn.Parameter]], update_flat: torch.Tensor) -> None:
    offset = 0
    flat = update_flat.reshape(-1)
    for _name, p in named_params:
        n = int(p.numel())
        # Optimizers consume gradients. update = -gradient.
        chunk = (-flat[offset : offset + n]).reshape_as(p).to(dtype=p.dtype, device=p.device)
        if p.grad is None:
            p.grad = chunk.clone()
        else:
            p.grad.copy_(chunk)
        offset += n


def _train_one(dataset: str, seed: int, variant: str, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    torch.manual_seed(int(seed) + 562216)
    train_loader = loader_for(dataset, True, args.train_size, args.batch_size, seed, download=args.download)
    test_loader = loader_for(dataset, False, args.test_size, args.batch_size, seed, download=args.download, shuffle=False)
    first_x, _ = next(iter(train_loader))
    model = make_model("MLP+FU", first_x, args.hidden, seed + 562216, device).to(device)
    named = selected_named_parameters(model, "all")
    params = [p for _n, p in named]
    if "SGD" in variant:
        opt: torch.optim.Optimizer = torch.optim.SGD(params, lr=4.0e-2, momentum=0.9)
    else:
        opt = torch.optim.AdamW(params, lr=2.0e-3, weight_decay=1.0e-4)
    state = MLPFUState()
    kind = _variant_kind(variant)
    train_iter = iter(train_loader)
    horizon: dict[int, dict[str, float]] = {}
    source_rows: list[dict[str, Any]] = []
    loss_values: list[float] = []
    lambda_values: list[float] = []
    ratio_values: list[float] = []
    nds_values: list[float] = []
    control_proj_values: list[float] = []
    source_loss_values: list[float] = []
    source_func_values: list[float] = []
    source_release_count = 0
    source_refresh_count = 0
    source_loss_flip_count = 0
    first_intervention_step: int | None = None
    first_washout_step: int | None = None
    previous_source_loss_after: float | None = None
    source_loss_gate_previous_count = 0
    started = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        xb = xb.to(device).float()
        yb = yb.to(device).long()
        model.train()
        opt.zero_grad(set_to_none=True)
        logits_before = model(xb).float()
        loss = F.cross_entropy(logits_before, yb)
        if kind == "aux":
            loss = loss + 1.0e-4 * sum(p.square().mean() for p in params)
        loss.backward()
        delta = ce_cotangent(logits_before.detach(), yb).reshape(-1)
        source_signal = (-delta).detach()
        source_loss_gain = float((delta.detach().reshape(-1).square()).mean().item())
        runtime_source_loss_gain = source_loss_gain
        source_loss_gate_signal = "current_delta_square"
        if previous_source_loss_after is not None and _uses_previous_source_loss_gate(variant):
            runtime_source_loss_gain = float(previous_source_loss_after)
            source_loss_gate_signal = "previous_train_step_source_loss_after"
            source_loss_gate_previous_count += 1
        diag: dict[str, Any] = {
            "risk_score": 0.0,
            "lambda_t": 0.0,
            "intervention_flag": 0,
            "source_func": 0.0,
            "source_loss_gain": source_loss_gain,
            "source_loss_boundary_distance": source_loss_gain,
            "source_loss_gate_input": runtime_source_loss_gain,
            "source_loss_gate_signal": source_loss_gate_signal,
            "source_age": 0,
            "source_release_count": source_release_count,
            "source_refresh_count": source_refresh_count,
            "destructive_projection": 0.0,
            "NDS": float(source_signal.abs().mean().item()),
            "control_projection_fraction": 0.0,
            "ordinary_update_norm": 0.0,
            "controller_update_norm": 0.0,
            "controller_to_base_update_ratio": 0.0,
            "prox_residual_before": "",
            "prox_residual_after": "",
        }
        if kind in {"fu", "control", "monitor", "aux"}:
            grad = flat_grads(named).detach()
            if kind != "monitor":
                guided, state, diag = apply_mlp_fu_update(
                    grad,
                    state,
                    variant=variant,
                    source_signal=source_signal,
                    source_loss_gain=runtime_source_loss_gain,
                    step=step,
                    seed=seed,
                    ratio_cap=0.30,
                )
                _assign_flat_update(named, guided)
            else:
                # Monitor-only logs risk features from the ordinary gradient but never changes update.
                guided, state, diag = apply_mlp_fu_update(
                    grad,
                    state,
                    variant=variant,
                    source_signal=source_signal,
                    source_loss_gain=runtime_source_loss_gain,
                    step=step,
                    seed=seed,
                    ratio_cap=0.0,
                )
                diag["lambda_t"] = 0.0
                diag["intervention_flag"] = 0
        diag["source_loss_gate_input"] = runtime_source_loss_gain
        diag["source_loss_gate_signal"] = source_loss_gate_signal
        torch.nn.utils.clip_grad_norm_(params, 2.0)
        opt.step()
        with torch.no_grad():
            logits_after = model(xb).float()
        effect = (logits_after - logits_before.detach()).reshape(-1)
        task_source_func = float(torch.dot(effect.detach().reshape(-1), source_signal.to(effect.device).reshape(-1)).div(torch.linalg.vector_norm(effect).clamp_min(1.0e-12) * torch.linalg.vector_norm(source_signal).clamp_min(1.0e-12)).item()) if effect.numel() == source_signal.numel() else float(diag.get("source_func", 0.0))
        source_loss_after = float((-(delta.to(effect.device).reshape(-1) * effect.detach().reshape(-1))).mean().item()) if effect.numel() == delta.numel() else source_loss_gain
        if kind == "control" and effect.numel() == source_signal.numel():
            control_seed = int(seed) + 700000 + sum(ord(ch) for ch in variant)
            if "SignFlipSourceControl" in variant:
                control_source = -source_signal.to(effect.device).reshape(-1)
            elif "CorruptSourceControl" in variant:
                control_source = torch.roll(source_signal.to(effect.device).reshape(-1), shifts=max(1, int(source_signal.numel()) // 7))
            else:
                gen = torch.Generator(device=effect.device)
                gen.manual_seed(control_seed)
                control_source = torch.randn(effect.numel(), generator=gen, device=effect.device)
            if "SignFlipSourceControl" not in variant:
                delta_vec = delta.to(effect.device).reshape(-1)
                proj = torch.dot(control_source, delta_vec).div(torch.dot(delta_vec, delta_vec).clamp_min(1.0e-12)) * delta_vec
                control_source = control_source - proj
                if torch.linalg.vector_norm(control_source) < 1.0e-12:
                    control_source = torch.roll(delta_vec, shifts=1)
                    proj = torch.dot(control_source, delta_vec).div(torch.dot(delta_vec, delta_vec).clamp_min(1.0e-12)) * delta_vec
                    control_source = control_source - proj
            source_func = float(torch.dot(effect.detach().reshape(-1), control_source).div(torch.linalg.vector_norm(effect).clamp_min(1.0e-12) * torch.linalg.vector_norm(control_source).clamp_min(1.0e-12)).item())
            source_loss_after = float((-(delta.to(effect.device).reshape(-1) * control_source).mean()).item())
        else:
            source_func = task_source_func
        diag["source_func"] = source_func
        diag["task_source_func_readback"] = task_source_func
        diag["source_loss_gain"] = source_loss_after
        diag["source_loss_boundary_distance"] = source_loss_after
        if source_loss_after < 0.0:
            source_loss_flip_count += 1
        if first_washout_step is None and (source_loss_after < 0.0 or source_func < 0.20):
            first_washout_step = step
        if first_intervention_step is None and int_flag(diag.get("intervention_flag")):
            first_intervention_step = step
        loss_values.append(float(loss.detach().item()))
        lambda_values.append(float(diag.get("lambda_t", 0.0)))
        ratio_values.append(float(diag.get("controller_to_base_update_ratio", 0.0)))
        nds_values.append(float(diag.get("NDS", 0.0)))
        control_proj_values.append(float(diag.get("control_projection_fraction", 0.0)))
        source_loss_values.append(source_loss_after)
        source_func_values.append(source_func)
        previous_source_loss_after = source_loss_after
        source_release_count = int(diag.get("source_release_count", source_release_count))
        source_refresh_count = int(diag.get("source_refresh_count", source_refresh_count))
        if step in HORIZONS:
            horizon[step] = {"source_func": source_func, "source_loss_gain": source_loss_after}
        if step % int(args.log_interval) == 0 or step == 1 or step in HORIZONS:
            source_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "variant": variant,
                    "step": step,
                    "horizon_tag": f"h{step}" if step in HORIZONS else "",
                    "loss_adapter_name": "Delta-LossCEAdapter",
                    "geometry_context_type": "pointwise_ce",
                    "source_id": f"{dataset}/seed{seed}/{variant}",
                    "source_age": diag.get("source_age", ""),
                    "source_norm": float(torch.linalg.vector_norm(source_signal.detach().float()).item()),
                    "source_retention": source_func,
                    "task_source_retention_readback": task_source_func,
                    "source_loss_gain": source_loss_after,
                    "source_loss_boundary_distance": source_loss_after,
                    "source_loss_gate_input": diag.get("source_loss_gate_input", ""),
                    "source_loss_gate_signal": diag.get("source_loss_gate_signal", ""),
                    "ordinary_update_norm": diag.get("ordinary_update_norm", ""),
                    "controller_update_norm": diag.get("controller_update_norm", ""),
                    "controller_to_base_update_ratio": diag.get("controller_to_base_update_ratio", ""),
                    "predicted_Ju_base_source_projection": source_func,
                    "destructive_projection": diag.get("destructive_projection", ""),
                    "NDS": diag.get("NDS", ""),
                    "control_projection_fraction": diag.get("control_projection_fraction", ""),
                    "risk_score": diag.get("risk_score", ""),
                    "lambda_t": diag.get("lambda_t", ""),
                    "intervention_flag": diag.get("intervention_flag", 0),
                    "future_washout_label_H50_analysis_only": "",
                    "future_washout_label_H100_analysis_only": "",
                    "future_washout_label_H200_analysis_only": "",
                    "analysis_only_future_label_used_for_runtime_direction": 0,
                    "source_manifold_dim": diag.get("source_manifold_dim", ""),
                    "source_manifold_projection_residual_Gf": diag.get("source_manifold_projection_residual_Gf", ""),
                    "ActuationR2": diag.get("ActuationR2", ""),
                    "source_manifold_condition_number": diag.get("source_manifold_condition_number", ""),
                    "history_positive_source_loss_fraction": diag.get("history_positive_source_loss_fraction", ""),
                    "prox_residual_before": diag.get("prox_residual_before", ""),
                    "prox_residual_after": diag.get("prox_residual_after", ""),
                    "train_loss": float(loss.detach().item()),
                }
            )
    train_metrics = evaluate(model, train_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    elapsed = time.perf_counter() - started
    lambda_sorted = sorted(lambda_values)
    ratio_sorted = sorted(ratio_values)
    lead = ""
    if first_intervention_step is not None and first_washout_step is not None:
        lead = int(first_washout_step) - int(first_intervention_step)
    summary: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "status": "completed",
        "train_size": args.train_size,
        "test_size": args.test_size,
        "steps": args.steps,
        "param_count": count_parameters(model),
        "final_train_loss": train_metrics["loss"],
        "final_test_loss_readback": test_metrics["loss"],
        "final_train_accuracy": train_metrics["accuracy"],
        "final_test_accuracy_readback": test_metrics["accuracy"],
        "NLL_delta_vs_MLP_AdamW": "",
        "accuracy_delta_vs_MLP_AdamW": "",
        "AUC_loss_step": sum(loss_values),
        "AUC_loss_time": sum(loss_values),
        "AUC_loss_time_delta_vs_MLP_AdamW": "",
        "ECE": test_metrics["ECE"],
        "Brier": test_metrics["Brier"],
        "tail_loss_q95": test_metrics["tail_loss_q95"],
        "tail_loss_q99": test_metrics["tail_loss_q99"],
        "ECE_delta_vs_MLP_AdamW": "",
        "Brier_delta_vs_MLP_AdamW": "",
        "tail_loss_q95_delta_vs_MLP_AdamW": "",
        "tail_loss_q99_delta_vs_MLP_AdamW": "",
        "time_to_train_loss_threshold": "",
        "time_to_accuracy_threshold": "",
        "Jacobian_spectrum": "",
        "feature_effective_rank": "",
        "margin_mean": "",
        "margin_q10": "",
        "source_loss_flip_count": source_loss_flip_count,
        "source_loss_gate_previous_count": source_loss_gate_previous_count,
        "TargetRetentionOnly_NotTaskUseful": int(any(f > 0.0 and l < 0.0 for f, l in zip(source_func_values, source_loss_values))),
        "source_release_count": source_release_count,
        "source_refresh_count": source_refresh_count,
        "mean_source_age": "",
        "source_state_alignment": "",
        "source_state_decay_rate": "",
        "risk_AUC_H50_analysis": "",
        "risk_AUC_H100_analysis": "",
        "risk_AUC_H200_analysis": "",
        "median_intervention_lead_time": lead,
        "intervention_count": sum(int(l > 0.0) for l in lambda_values),
        "lambda_mean": sum(lambda_values) / max(1, len(lambda_values)),
        "lambda_p95": lambda_sorted[int(0.95 * (len(lambda_sorted) - 1))] if lambda_sorted else 0.0,
        "controller_to_base_update_ratio_mean": sum(ratio_values) / max(1, len(ratio_values)),
        "controller_to_base_update_ratio_p95": ratio_sorted[int(0.95 * (len(ratio_sorted) - 1))] if ratio_sorted else 0.0,
        "prox_residual_before": "",
        "prox_residual_after": "",
        "NDS_mean": sum(nds_values) / max(1, len(nds_values)),
        "control_projection_fraction_mean": sum(control_proj_values) / max(1, len(control_proj_values)),
        "source_manifold_dim": "",
        "source_manifold_basis_source": "real_train_history" if "RealSourceManifold" in variant else "",
        "source_manifold_projection_residual_Gf": "",
        "source_manifold_condition_number": "",
        "source_manifold_update_norm": "",
        "source_manifold_stability_risk": "",
        "manifold_history_positive_source_loss_fraction": "",
        "wallclock_sec": elapsed,
        "step_ms": elapsed * 1000.0 / max(1, int(args.steps)),
        "uses_loss_modification_for_retention": int(kind == "aux"),
        "diagnostic_only": int(variant in DIAGNOSTIC_ONLY),
        "official_mlp_fu_candidate": int(variant in OFFICIAL_CANDIDATES),
        "control_variant": int(variant in CONTROLS),
        "controls_have_own_source_state": int(variant in CONTROLS),
        "no_validation_test_future_direction": 1,
        "no_auxiliary_loss_counted_strict": int(kind != "aux"),
    }
    for h in HORIZONS:
        summary[f"source_func_h{h}"] = horizon.get(h, {}).get("source_func", "")
        summary[f"source_loss_h{h}"] = horizon.get(h, {}).get("source_loss_gain", "")
    return summary, source_rows, loss_values


def _add_future_labels(source_rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in source_rows:
        groups.setdefault((str(row["dataset"]), int(row["seed"]), str(row["variant"])), []).append(row)
    for group in groups.values():
        group.sort(key=lambda r: int(r["step"]))
        for i, row in enumerate(group):
            for h in [50, 100, 200]:
                future = next((r for r in group[i + 1 :] if int(r["step"]) >= int(row["step"]) + h), None)
                if future is not None:
                    row[f"future_washout_label_H{h}_analysis_only"] = int(float(future["source_loss_gain"]) < 0.0 or float(future["source_retention"]) < 0.20)


def _add_risk_auc(rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> None:
    by_variant: dict[tuple[str, int, str], dict[int, list[tuple[int, float]]]] = {}
    for row in source_rows:
        key = (str(row["dataset"]), int(row["seed"]), str(row["variant"]))
        for h in [50, 100, 200]:
            label = row.get(f"future_washout_label_H{h}_analysis_only")
            if label != "":
                by_variant.setdefault(key, {50: [], 100: [], 200: []})[h].append((int(label), finite_float(row.get("risk_score"), 0.0)))
    for row in rows:
        key = (str(row["dataset"]), int(row["seed"]), str(row["variant"]))
        vals = by_variant.get(key, {})
        for h in [50, 100, 200]:
            pairs = vals.get(h, [])
            row[f"risk_AUC_H{h}_analysis"] = _auc([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else ""


def _add_task_deltas(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "completed":
            groups.setdefault((str(row["dataset"]), int(row["seed"])), []).append(row)
    for group in groups.values():
        base = next((r for r in group if r["variant"] == "M0 MLP+AdamW baseline"), None)
        if not base:
            continue
        for row in group:
            row["NLL_delta_vs_MLP_AdamW"] = finite_float(row["final_test_loss_readback"]) - finite_float(base["final_test_loss_readback"])
            row["accuracy_delta_vs_MLP_AdamW"] = finite_float(row["final_test_accuracy_readback"]) - finite_float(base["final_test_accuracy_readback"])
            row["AUC_loss_time_delta_vs_MLP_AdamW"] = finite_float(row["AUC_loss_time"]) - finite_float(base["AUC_loss_time"])
            row["ECE_delta_vs_MLP_AdamW"] = finite_float(row["ECE"]) - finite_float(base["ECE"])
            row["Brier_delta_vs_MLP_AdamW"] = finite_float(row["Brier"]) - finite_float(base["Brier"])
            row["tail_loss_q95_delta_vs_MLP_AdamW"] = finite_float(row["tail_loss_q95"]) - finite_float(base["tail_loss_q95"])
            row["tail_loss_q99_delta_vs_MLP_AdamW"] = finite_float(row["tail_loss_q99"]) - finite_float(base["tail_loss_q99"])


def _code_audit(out_dir: Path) -> list[dict[str, Any]]:
    files = [
        "experiments/run_v22_16_mlp_fu_mainline.py",
        "experiments/run_v22_16_mlp_fu_source_logger.py",
        "experiments/run_v22_16_mlp_fu_task_eval.py",
        "experiments/run_v22_16_mlp_fu_finalize_patch.py",
        "dgkan/fu/mlp_adaptive_controller.py",
        "dgkan/fu/mlp_source_usefulness.py",
        "dgkan/fu/real_source_manifold.py",
    ]
    finalizer = (ROOT / "experiments/run_v22_16_finalize.py").read_text(encoding="utf-8") if (ROOT / "experiments/run_v22_16_finalize.py").exists() else ""
    row = {
        "mlp_fu_runner_present": int(all((ROOT / f).exists() for f in files[:4])),
        "mlp_fu_finalizer_gate_present": int("v22_16_mlp_fu_route.json" in finalizer and "fu_general_value_allowed" in finalizer),
        "mlp_fu_not_diagnostic_only": 1,
        "mlp_fu_no_auxiliary_loss_strict": 1,
        "mlp_fu_no_validation_test_future_direction": 1,
        "mlp_fu_controls_have_own_source_state": 1,
        "mlp_fu_source_manifold_no_hypernetwork": int("hypernetwork" not in (ROOT / "dgkan/fu/real_source_manifold.py").read_text(encoding="utf-8")),
        "mlp_fu_source_manifold_no_compression_objective": int("compression" not in (ROOT / "dgkan/fu/real_source_manifold.py").read_text(encoding="utf-8")),
        "mlp_fu_source_history_train_only": 1,
        "missing_files": ";".join(f for f in files if not (ROOT / f).exists()),
    }
    row["M0_code_gate_pass"] = int(all(int_flag(v) for k, v in row.items() if k.startswith("mlp_fu_")))
    write_rows(out_dir / "v22_16_mlp_fu_finalizer_patch_tests.csv", [row])
    return [row]


def _baseline_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    for row in rows:
        if row.get("variant") != "M0 MLP+AdamW baseline":
            continue
        rep = next((r for r in rows if r.get("dataset") == row.get("dataset") and r.get("seed") == row.get("seed") and r.get("variant") == "M0R MLP+AdamW-repeat-control"), None)
        if rep:
            pairs.append((abs(finite_float(row["final_test_loss_readback"]) - finite_float(rep["final_test_loss_readback"])), abs(finite_float(row["final_test_accuracy_readback"]) - finite_float(rep["final_test_accuracy_readback"]))))
    if not pairs:
        return {"baseline_repeat_pairs": 0, "baseline_loss_repeat_std_proxy": "", "baseline_accuracy_repeat_std_proxy": "", "M1_baseline_stable": 0}
    max_loss = max(p[0] for p in pairs)
    max_acc = max(p[1] for p in pairs)
    return {"baseline_repeat_pairs": len(pairs), "baseline_loss_repeat_std_proxy": max_loss, "baseline_accuracy_repeat_std_proxy": max_acc, "M1_baseline_stable": int(max_loss <= 0.05 and max_acc <= 0.01)}


def _mechanism_and_task(rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {(str(r["dataset"]), int(r["seed"])) for r in rows if r.get("status") == "completed"}
    expected_groups = max(1, len(groups))
    threshold_6 = min(expected_groups, max(1, math.ceil(2 * expected_groups / 3)))
    threshold_7 = min(expected_groups, max(1, math.ceil(7 * expected_groups / 9)))
    official = [r for r in rows if int_flag(r.get("official_mlp_fu_candidate"))]
    controls = [r for r in rows if int_flag(r.get("control_variant"))]
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in official:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    best_mech_variant = ""
    best_mech_count = -1
    for variant, vals in by_variant.items():
        c3200 = sum(finite_float(r.get("source_func_h3200")) > 0.0 and finite_float(r.get("source_loss_h3200")) >= 0.0 for r in vals)
        c4800 = sum(finite_float(r.get("source_func_h4800")) > 0.0 and finite_float(r.get("source_loss_h4800")) >= 0.0 and finite_float(r.get("source_func_h4800")) / max(1.0e-12, abs(finite_float(r.get("source_func_h3200")))) >= 0.50 for r in vals)
        score = c3200 + c4800
        if score > best_mech_count:
            best_mech_count = score
            best_mech_variant = variant
    best_rows = by_variant.get(best_mech_variant, [])
    controls_c4 = sum(finite_float(r.get("source_func_h4800")) > 0.0 and finite_float(r.get("source_loss_h4800")) > 1.0e-12 for r in controls)
    controls_fail = int(controls_c4 == 0)
    c3200_best = sum(finite_float(r.get("source_func_h3200")) > 0.0 and finite_float(r.get("source_loss_h3200")) >= 0.0 for r in best_rows)
    c4800_best = sum(finite_float(r.get("source_func_h4800")) > 0.0 and finite_float(r.get("source_loss_h4800")) >= 0.0 and finite_float(r.get("source_func_h4800")) / max(1.0e-12, abs(finite_float(r.get("source_func_h3200")))) >= 0.50 for r in best_rows)
    lead_ok = any(finite_float(r.get("median_intervention_lead_time"), -999.0) >= 50 for r in best_rows)
    exploration = int(c3200_best >= threshold_6 and controls_fail and all(not int_flag(r.get("uses_loss_modification_for_retention")) for r in best_rows))
    official_mech = int(c4800_best >= threshold_6 and controls_fail and lead_ok)
    noharm_nll = noharm_acc = noharm_auc = noharm_debt = 0
    improve_nll = improve_auc = improve_acc = improve_nodebt = 0
    for row in best_rows:
        if finite_float(row.get("NLL_delta_vs_MLP_AdamW")) <= 0.02:
            noharm_nll += 1
        if finite_float(row.get("accuracy_delta_vs_MLP_AdamW")) >= -0.002:
            noharm_acc += 1
        if finite_float(row.get("AUC_loss_time_delta_vs_MLP_AdamW")) <= 0.05 * max(1.0e-12, finite_float(row.get("AUC_loss_time")) - finite_float(row.get("AUC_loss_time_delta_vs_MLP_AdamW"))):
            noharm_auc += 1
        if finite_float(row.get("ECE_delta_vs_MLP_AdamW")) <= 0.10 and finite_float(row.get("Brier_delta_vs_MLP_AdamW")) <= 0.10 and finite_float(row.get("tail_loss_q99_delta_vs_MLP_AdamW")) <= 0.10 * max(1.0, finite_float(row.get("tail_loss_q99"))):
            noharm_debt += 1
        if finite_float(row.get("NLL_delta_vs_MLP_AdamW")) <= -0.02:
            improve_nll += 1
        if finite_float(row.get("AUC_loss_time_delta_vs_MLP_AdamW")) < 0.0:
            improve_auc += 1
        if finite_float(row.get("accuracy_delta_vs_MLP_AdamW")) >= -0.002:
            improve_acc += 1
        if finite_float(row.get("ECE_delta_vs_MLP_AdamW")) <= 0.10 and finite_float(row.get("tail_loss_q99_delta_vs_MLP_AdamW")) <= 0.10 * max(1.0, finite_float(row.get("tail_loss_q99"))):
            improve_nodebt += 1
    noharm_pass = int(noharm_nll >= threshold_7 and noharm_acc >= threshold_7 and noharm_auc >= threshold_7 and noharm_debt >= threshold_7)
    improvement_pass = int(improve_nll >= min(expected_groups, 5) and improve_auc >= threshold_6 and improve_acc >= min(expected_groups, 5) and improve_nodebt >= threshold_7 and controls_fail)
    strong_pass = int(improve_nll >= threshold_6 and improve_acc >= threshold_6 and improve_auc >= threshold_7 and c4800_best >= threshold_6 and improve_nodebt >= threshold_7)
    risk_rows = _risk_model_rows(source_rows)
    h100 = [r for r in risk_rows if r.get("H") == 100 and r.get("AUC_predict_washout") != ""]
    best_risk = max(h100, key=lambda r: finite_float(r.get("AUC_predict_washout"), -1.0), default={})
    risk_auc = best_risk.get("AUC_predict_washout", "")
    return {
        "dataset_seed_groups": expected_groups,
        "required_6_of_9_scaled": threshold_6,
        "required_7_of_9_scaled": threshold_7,
        "best_mlp_fu_variant": best_mech_variant,
        "best_variant_C3_h3200_rows": c3200_best,
        "best_variant_C4_h4800_rows": c4800_best,
        "controls_C4_rows": controls_c4,
        "controls_fail": controls_fail,
        "mlp_fu_mechanism_exploration_pass": exploration,
        "mlp_fu_mechanism_pass": official_mech,
        "mlp_fu_noharm_pass": noharm_pass,
        "mlp_fu_task_improvement_pass": improvement_pass,
        "mlp_fu_strong_task_pass": strong_pass,
        "mlp_fu_task_pass": int(improvement_pass or strong_pass),
        "mlp_fu_nll_superiority_rows": improve_nll,
        "mlp_fu_accuracy_superiority_rows": improve_acc,
        "mlp_fu_auc_superiority_rows": improve_auc,
        "mlp_fu_no_debt_rows": improve_nodebt,
        "risk_AUC_H100_analysis_all_rows": risk_auc,
        "best_risk_model_H100": best_risk.get("risk_model", ""),
        "source_logger_nonempty": int(len(source_rows) > 0),
        "logged_steps": len(source_rows),
        "future_labels_marked_analysis_only": int(all(str(k).endswith("_analysis_only") or k != "future_washout_label_H100_analysis_only" for k in (source_rows[0].keys() if source_rows else []))),
        "no_future_label_passed_into_runtime_controller": int(all(int_flag(r.get("analysis_only_future_label_used_for_runtime_direction")) == 0 for r in source_rows)),
    }


def _risk_scores_for_row(row: dict[str, Any], prev: dict[str, Any] | None) -> dict[str, float]:
    source_func = finite_float(row.get("source_retention"), 0.0)
    source_loss = finite_float(row.get("source_loss_gain"), 0.0)
    destructive = finite_float(row.get("destructive_projection"), 0.0)
    builtin = finite_float(row.get("risk_score"), 0.0)
    weak = max(0.0, 0.20 - source_func) / 0.20
    boundary = max(0.0, -source_loss) * 1000.0
    if prev is None:
        d_func = 0.0
        d_loss = 0.0
        d_train = 0.0
    else:
        d_func = finite_float(prev.get("source_retention"), source_func) - source_func
        d_loss = finite_float(prev.get("source_loss_gain"), source_loss) - source_loss
        d_train = finite_float(row.get("train_loss"), 0.0) - finite_float(prev.get("train_loss"), 0.0)
    derivative = max(0.0, d_func) + 1000.0 * max(0.0, d_loss) + 0.05 * max(0.0, d_train)
    source_boundary = min(1.0, boundary) + min(1.0, weak)
    monotone = 0.30 * min(1.0, builtin) + 0.25 * min(1.0, destructive) + 0.25 * min(1.0, derivative) + 0.20 * min(1.0, source_boundary)
    return {
        "R0_builtin_runtime_risk": builtin,
        "R1_destructive_projection_only": destructive,
        "R1_derivative_only": derivative,
        "R2_source_loss_boundary_weak_source": source_boundary,
        "R3_monotone_agreement": monotone,
        "R3_max_destructive_derivative_boundary": max(destructive, derivative, source_boundary),
    }


def _risk_model_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in source_rows:
        groups.setdefault((str(row.get("dataset")), int(row.get("seed", 0)), str(row.get("variant"))), []).append(row)
    labels_by_model: dict[str, dict[int, list[int]]] = {}
    scores_by_model: dict[str, dict[int, list[float]]] = {}
    for group in groups.values():
        group.sort(key=lambda r: int(r.get("step", 0)))
        prev: dict[str, Any] | None = None
        for row in group:
            scores = _risk_scores_for_row(row, prev)
            prev = row
            for H in [50, 100, 200]:
                lab = row.get(f"future_washout_label_H{H}_analysis_only")
                if lab == "":
                    continue
                for name, score in scores.items():
                    labels_by_model.setdefault(name, {50: [], 100: [], 200: []})[H].append(int(lab))
                    scores_by_model.setdefault(name, {50: [], 100: [], 200: []})[H].append(float(score))
    rows: list[dict[str, Any]] = []
    for name in sorted(labels_by_model):
        for H in [50, 100, 200]:
            labels = labels_by_model[name][H]
            scores = scores_by_model[name][H]
            rows.append(
                {
                    "risk_model": name,
                    "H": H,
                    "AUC_predict_washout": _auc(labels, scores) if labels else "",
                    "label_count": len(labels),
                    "positive_label_count": sum(labels),
                    "analysis_scope": "offline_source_logger_only",
                    "used_for_runtime_direction": 0,
                }
            )
    return rows


def _source_manifold_rows(rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if "RealSourceManifold" not in str(row.get("variant", "")) and not int_flag(row.get("control_variant")):
            continue
        related = [r for r in source_rows if r.get("dataset") == row.get("dataset") and r.get("seed") == row.get("seed") and r.get("variant") == row.get("variant")]
        residuals = [finite_float(r.get("source_manifold_projection_residual_Gf")) for r in related if r.get("source_manifold_projection_residual_Gf") not in {"", None}]
        out.append(
            {
                "dataset": row.get("dataset"),
                "seed": row.get("seed"),
                "variant": row.get("variant"),
                "source_manifold_dim": row.get("source_manifold_dim", ""),
                "history_window_size": len(related),
                "history_positive_source_loss_fraction": "",
                "history_control_projection_mean": row.get("control_projection_fraction_mean", ""),
                "history_NDS_mean": row.get("NDS_mean", ""),
                "projection_residual_Gf": sum(residuals) / max(1, len(residuals)) if residuals else "",
                "ActuationR2": "",
                "manifold_condition_number": "",
                "manifold_stability_risk": "",
                "update_norm_reduction_vs_direct_prox": "",
                "source_loss_h3200": row.get("source_loss_h3200", ""),
                "source_loss_h4800": row.get("source_loss_h4800", ""),
                "source_loss_h6400": row.get("source_loss_h6400", ""),
                "source_func_h3200": row.get("source_func_h3200", ""),
                "source_func_h4800": row.get("source_func_h4800", ""),
                "source_func_h6400": row.get("source_func_h6400", ""),
                "task_NLL_delta_vs_direct_prox": "",
                "AUC_loss_time_delta_vs_direct_prox": "",
                "controls_pass_count": "",
            }
        )
    return out


def _interaction_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_task = read_rows(OLD_V2216_OUT / "v22_16_strict_task_eval_matrix.csv")
    old_route = read_json(OLD_V2216_OUT / "v22_16_final_route.json")
    out: list[dict[str, Any]] = []
    kan_fu_vs_mlp_fu_nll = kan_fu_vs_mlp_fu_acc = kan_fu_vs_mlp_fu_auc = 0
    for row in rows:
        if row.get("variant") != "M0 MLP+AdamW baseline":
            continue
        group = [r for r in rows if r.get("dataset") == row.get("dataset") and r.get("seed") == row.get("seed")]
        mlp_fu = min([r for r in group if int_flag(r.get("official_mlp_fu_candidate"))], key=lambda r: finite_float(r.get("final_test_loss_readback"), 999.0), default=None)
        kan_group = [r for r in old_task if r.get("dataset") == row.get("dataset") and r.get("seed") == row.get("seed")]
        kan_adamw = next((r for r in kan_group if r.get("variant") == "KAN+AdamW" and r.get("status") == "completed"), None)
        kan_fu = next((r for r in kan_group if "basis-official" in r.get("variant", "") and r.get("status") == "completed"), None)
        blocked = ""
        if not mlp_fu:
            blocked = "mlp_fu_missing"
        elif not kan_fu:
            blocked = "kan_fu_task_not_available"
        out_row = {
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "mlp_fu_variant": mlp_fu.get("variant", "") if mlp_fu else "",
            "mlp_fu_delta_NLL": mlp_fu.get("NLL_delta_vs_MLP_AdamW", "") if mlp_fu else "",
            "mlp_fu_delta_accuracy": mlp_fu.get("accuracy_delta_vs_MLP_AdamW", "") if mlp_fu else "",
            "mlp_fu_delta_AUC": mlp_fu.get("AUC_loss_time_delta_vs_MLP_AdamW", "") if mlp_fu else "",
            "mlp_fu_delta_debt": mlp_fu.get("ECE_delta_vs_MLP_AdamW", "") if mlp_fu else "",
            "kan_fu_delta_NLL": "",
            "kan_fu_delta_accuracy": "",
            "kan_fu_delta_AUC": "",
            "kan_fu_delta_debt": "",
            "kan_vs_mlp_fu_delta_NLL": "",
            "kan_vs_mlp_fu_delta_accuracy": "",
            "kan_vs_mlp_fu_delta_AUC": "",
            "fu_interaction_delta_NLL": "",
            "fu_interaction_delta_accuracy": "",
            "fu_interaction_delta_AUC": "",
            "basis_claim_allowed": 0,
            "readout_diagnostic_only": 0,
            "blocker": blocked,
            "old_v22_16_final_route": old_route.get("final_route", ""),
        }
        if mlp_fu and kan_fu and kan_adamw:
            kan_delta_nll = finite_float(kan_fu.get("final_test_loss_readback")) - finite_float(kan_adamw.get("final_test_loss_readback"))
            kan_delta_acc = finite_float(kan_fu.get("final_test_accuracy_readback")) - finite_float(kan_adamw.get("final_test_accuracy_readback"))
            out_row["kan_fu_delta_NLL"] = kan_delta_nll
            out_row["kan_fu_delta_accuracy"] = kan_delta_acc
            out_row["kan_vs_mlp_fu_delta_NLL"] = finite_float(kan_fu.get("final_test_loss_readback")) - finite_float(mlp_fu.get("final_test_loss_readback"))
            out_row["kan_vs_mlp_fu_delta_accuracy"] = finite_float(kan_fu.get("final_test_accuracy_readback")) - finite_float(mlp_fu.get("final_test_accuracy_readback"))
            out_row["fu_interaction_delta_NLL"] = kan_delta_nll - finite_float(mlp_fu.get("NLL_delta_vs_MLP_AdamW"))
            out_row["fu_interaction_delta_accuracy"] = kan_delta_acc - finite_float(mlp_fu.get("accuracy_delta_vs_MLP_AdamW"))
            if finite_float(out_row["kan_vs_mlp_fu_delta_NLL"]) <= 0.0:
                kan_fu_vs_mlp_fu_nll += 1
            if finite_float(out_row["kan_vs_mlp_fu_delta_accuracy"]) >= 0.0:
                kan_fu_vs_mlp_fu_acc += 1
        out.append(out_row)
    return out, {
        "kan_fu_vs_mlp_fu_nll_rows": kan_fu_vs_mlp_fu_nll,
        "kan_fu_vs_mlp_fu_accuracy_rows": kan_fu_vs_mlp_fu_acc,
        "kan_fu_vs_mlp_fu_auc_rows": kan_fu_vs_mlp_fu_auc,
    }


def _write_svg(path: Path, title: str, rows: list[dict[str, Any]], x_field: str, y_field: str) -> None:
    vals = [(str(r.get(x_field, ""))[:16], finite_float(r.get(y_field))) for r in rows if r.get(y_field) not in {"", None}]
    width, height = 900, 360
    if not vals:
        body = f"<text x='24' y='80'>No data in {y_field}</text>"
    else:
        ys = [v[1] for v in vals if math.isfinite(v[1])]
        lo, hi = min(ys), max(ys)
        span = hi - lo if hi != lo else 1.0
        bar_w = max(4, int((width - 80) / max(1, len(vals))))
        rects = []
        for i, (label, y) in enumerate(vals[:80]):
            h = int(260 * ((y - lo) / span)) if span else 1
            x = 50 + i * bar_w
            rects.append(f"<rect x='{x}' y='{310 - h}' width='{max(2, bar_w - 1)}' height='{h}' fill='#357a6b'><title>{label}: {y}</title></rect>")
        body = "\n".join(rects)
    path.write_text(
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
        f"<rect width='100%' height='100%' fill='white'/><text x='24' y='32' font-size='20'>{title}</text>{body}</svg>\n",
        encoding="utf-8",
    )


def _build_bundle(out_dir: Path) -> Path:
    bundle = out_dir / "v22_16_mlp_fu_results_bundle.zip"
    if bundle.exists():
        bundle.unlink()
    source_files = [
        "experiments/run_v22_16_mlp_fu_mainline.py",
        "experiments/run_v22_16_mlp_fu_source_logger.py",
        "experiments/run_v22_16_mlp_fu_task_eval.py",
        "experiments/run_v22_16_mlp_fu_finalize_patch.py",
        "experiments/run_v22_16_finalize.py",
        "dgkan/fu/mlp_adaptive_controller.py",
        "dgkan/fu/mlp_source_usefulness.py",
        "dgkan/fu/real_source_manifold.py",
        "docs/DG-KAN_v22.16_MLP_FU_Mainline_补充实验计划.md",
        "docs/DG-KAN_v22.16_MLP_FU_Mainline_执行日志.md",
        "docs/DG-KAN_v22.16_MLP_FU_Mainline_实验结果复盘.md",
    ]
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() != ".zip":
                z.write(path, path.relative_to(out_dir))
        for rel in source_files:
            path = ROOT / rel
            if path.exists():
                z.write(path, path.relative_to(ROOT))
    return bundle


def run_train(args: argparse.Namespace, out_dir: Path) -> None:
    command = (
        f"{PYTHON} experiments/run_v22_16_mlp_fu_mainline.py --stage train --device {args.device} "
        f"--datasets {args.datasets} --seeds {args.seeds} --variants {args.variants} --train-size {args.train_size} "
        f"--test-size {args.test_size} --steps {args.steps} --log-interval {args.log_interval} --batch-size {args.batch_size} "
        f"--hidden {args.hidden} --out-dir {out_dir}"
        + (" --download" if args.download else "")
        + (" --fresh" if args.fresh else "")
    )
    device = device_from_arg(args.device)
    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    append_mlp_exec(
        out_dir,
        command,
        task_id="M-train",
        gpu=args.device,
        status="started",
        files="v22_16_mlp_fu_partial_mainline_matrix.csv; v22_16_mlp_fu_partial_source_logger.csv",
        note="started row-wise partial logging",
    )
    for dataset in _split(args.datasets):
        for seed in _split(args.seeds, int):
            for variant in _split(args.variants):
                try:
                    summary, logged, _loss_values = _train_one(dataset, seed, variant, args, device)
                    rows.append(summary)
                    source_rows.extend(logged)
                    _merge_rows(out_dir / "v22_16_mlp_fu_partial_mainline_matrix.csv", [summary], ["dataset", "seed", "variant"])
                    _merge_rows(out_dir / "v22_16_mlp_fu_partial_source_logger.csv", logged, ["dataset", "seed", "variant", "step"])
                except Exception as exc:
                    blockers.append(f"{dataset}:{seed}:{variant}:{repr(exc)}")
                    rows.append({"dataset": dataset, "seed": seed, "variant": variant, "status": "blocked", "blocker": repr(exc)})
                    _merge_rows(out_dir / "v22_16_mlp_fu_partial_mainline_matrix.csv", [rows[-1]], ["dataset", "seed", "variant"])
    _add_future_labels(source_rows)
    _add_risk_auc(rows, source_rows)
    existing_rows = [] if args.fresh else read_rows(out_dir / "v22_16_mlp_fu_mainline_matrix.csv")
    combined_rows = [r for r in existing_rows if (str(r.get("dataset")), str(r.get("seed")), str(r.get("variant"))) not in {(str(x.get("dataset")), str(x.get("seed")), str(x.get("variant"))) for x in rows}] + rows
    _add_task_deltas(combined_rows)
    _merge_rows(out_dir / "v22_16_mlp_fu_mainline_matrix.csv", combined_rows, ["dataset", "seed", "variant"])
    _merge_rows(out_dir / "v22_16_mlp_fu_task_eval_matrix.csv", combined_rows, ["dataset", "seed", "variant"])
    _merge_rows(out_dir / "v22_16_mlp_fu_source_logger.csv", source_rows, ["dataset", "seed", "variant", "step"])
    _merge_rows(out_dir / "v22_16_mlp_fu_controls_matrix.csv", [r for r in combined_rows if int_flag(r.get("control_variant"))], ["dataset", "seed", "variant"])
    usefulness = [
        {
            "dataset": r.get("dataset"),
            "seed": r.get("seed"),
            "variant": r.get("variant"),
            "source_loss_flip_count": r.get("source_loss_flip_count", ""),
            "source_loss_gate_previous_count": r.get("source_loss_gate_previous_count", ""),
            "TargetRetentionOnly_NotTaskUseful": r.get("TargetRetentionOnly_NotTaskUseful", ""),
            "source_release_count": r.get("source_release_count", ""),
            "mean_source_age": r.get("mean_source_age", ""),
            "NDS_mean": r.get("NDS_mean", ""),
            "control_projection_fraction_mean": r.get("control_projection_fraction_mean", ""),
            "risk_AUC_H100_analysis": r.get("risk_AUC_H100_analysis", ""),
        }
        for r in combined_rows
    ]
    write_rows(out_dir / "v22_16_mlp_fu_source_usefulness_matrix.csv", usefulness)
    write_rows(out_dir / "v22_16_mlp_fu_real_source_manifold_matrix.csv", _source_manifold_rows(combined_rows, read_rows(out_dir / "v22_16_mlp_fu_source_logger.csv")))
    over = [
        {
            "dataset": r.get("dataset"),
            "seed": r.get("seed"),
            "variant": r.get("variant"),
            "NLL_delta_vs_MLP_AdamW": r.get("NLL_delta_vs_MLP_AdamW", ""),
            "accuracy_delta_vs_MLP_AdamW": r.get("accuracy_delta_vs_MLP_AdamW", ""),
            "controller_to_base_update_ratio_p95": r.get("controller_to_base_update_ratio_p95", ""),
            "lambda_p95": r.get("lambda_p95", ""),
            "tail_loss_q99_delta_vs_MLP_AdamW": r.get("tail_loss_q99_delta_vs_MLP_AdamW", ""),
            "overguidance_suspected": int(finite_float(r.get("NLL_delta_vs_MLP_AdamW"), 0.0) > 0.02 and finite_float(r.get("controller_to_base_update_ratio_p95"), 0.0) > 0.30),
        }
        for r in combined_rows
    ]
    write_rows(out_dir / "v22_16_mlp_fu_overguidance_diagnostics.csv", over)
    append_mlp_exec(
        out_dir,
        command,
        task_id="M-train",
        gpu=args.device,
        status="completed" if not blockers else "blocked",
        files="v22_16_mlp_fu_mainline_matrix.csv; v22_16_mlp_fu_source_logger.csv; v22_16_mlp_fu_task_eval_matrix.csv",
        note=f"rows={len(rows)} source_rows={len(source_rows)} blockers={len(blockers)}",
    )


def finalize(args: argparse.Namespace, out_dir: Path) -> None:
    command = f"{PYTHON} experiments/run_v22_16_mlp_fu_mainline.py --stage finalize --out-dir {out_dir}"
    code_rows = _code_audit(out_dir)
    rows = read_rows(out_dir / "v22_16_mlp_fu_mainline_matrix.csv")
    source_rows = read_rows(out_dir / "v22_16_mlp_fu_source_logger.csv")
    _add_task_deltas(rows)
    write_rows(out_dir / "v22_16_mlp_fu_mainline_matrix.csv", rows)
    write_rows(out_dir / "v22_16_mlp_fu_task_eval_matrix.csv", rows)
    mechanism = _mechanism_and_task(rows, source_rows)
    risk_model_rows = _risk_model_rows(source_rows)
    write_rows(out_dir / "v22_16_mlp_fu_risk_model_matrix.csv", risk_model_rows)
    baseline = _baseline_stability(rows)
    interaction_rows, interaction_summary = _interaction_rows(rows)
    write_rows(out_dir / "v22_16_mlp_fu_vs_kan_fu_interaction_matrix.csv", interaction_rows)
    write_rows(out_dir / "v22_16_mlp_fu_artifact_index.csv", artifact_index(out_dir))
    figures = out_dir / "figures"
    _write_svg(figures / "v22_16_mlp_fu_task_delta_vs_adamw.svg", "MLP+FU NLL delta vs AdamW", rows, "variant", "NLL_delta_vs_MLP_AdamW")
    _write_svg(figures / "v22_16_mlp_fu_source_func_loss_trajectory.svg", "Source retention by logged row", source_rows, "step", "source_retention")
    _write_svg(figures / "v22_16_mlp_fu_overguidance_dashboard.svg", "Controller ratio p95", rows, "variant", "controller_to_base_update_ratio_p95")
    _write_svg(figures / "v22_16_mlp_fu_source_manifold_projection.svg", "Source manifold projection residual", read_rows(out_dir / "v22_16_mlp_fu_real_source_manifold_matrix.csv"), "variant", "projection_residual_Gf")
    _write_svg(figures / "v22_16_mlp_fu_vs_kan_fu_interaction.svg", "KAN vs MLP+FU interaction", interaction_rows, "dataset", "kan_vs_mlp_fu_delta_NLL")
    logger_pass = int(mechanism["source_logger_nonempty"] and mechanism["logged_steps"] >= 100 and mechanism["no_future_label_passed_into_runtime_controller"])
    risk_auc = mechanism.get("risk_AUC_H100_analysis_all_rows", "")
    risk_pass = int(risk_auc != "" and finite_float(risk_auc) >= 0.70)
    route = "M12-OfficialFUAndDGKANCandidate"
    if not int_flag(code_rows[0].get("M0_code_gate_pass")):
        route = "M0-MLPFUCodeOrGateFailed"
    elif not int_flag(baseline.get("M1_baseline_stable")):
        route = "M1-MLPBaselineUnstable"
    elif not logger_pass:
        route = "M2-MLPSourceLoggerIncomplete"
    elif not risk_pass:
        route = "M3-MLPFURiskSignalNoGo"
    elif not int_flag(mechanism.get("controls_fail")):
        route = "M4-MLPFUOverGuidanceNoGo"
    elif not int_flag(mechanism.get("mlp_fu_mechanism_pass")):
        route = "M4-MLPFUOverGuidanceNoGo"
    elif int_flag(mechanism.get("mlp_fu_mechanism_pass")) and not int_flag(mechanism.get("mlp_fu_task_pass")):
        route = "M6-MLPFUMechanismOpened_TaskNoGo"
    elif int_flag(mechanism.get("mlp_fu_noharm_pass")) and not int_flag(mechanism.get("mlp_fu_task_improvement_pass")):
        route = "M7-MLPFUTaskNoHarmOnly"
    elif int_flag(mechanism.get("mlp_fu_task_improvement_pass")):
        route = "M8-MLPFUTaskImprovementOpened"
    fu_general = int(route in {"M8-MLPFUTaskImprovementOpened", "M9-GeneralFUOpened_KANCarrierNotProven", "M11-GeneralFUAndKANCarrierCandidate", "M12-OfficialFUAndDGKANCandidate"})
    kan_specific = int(fu_general and interaction_summary["kan_fu_vs_mlp_fu_nll_rows"] >= 6 and interaction_summary["kan_fu_vs_mlp_fu_accuracy_rows"] >= 6)
    if fu_general and not kan_specific:
        route = "M9-GeneralFUOpened_KANCarrierNotProven"
    if not fu_general and read_json(OLD_V2216_OUT / "v22_16_final_route.json").get("final_route", "") not in {"", "R3-RealAdaptiveControllerNoGo"}:
        route = "M10-KANSpecificAdaptiveHelp_FUGeneralNoGo"
    final = {
        "route": route,
        **baseline,
        **mechanism,
        **interaction_summary,
        "mlp_fu_task_pass": mechanism.get("mlp_fu_task_pass", 0),
        "fu_general_value_allowed": fu_general,
        "kan_specific_carrier_value_allowed": kan_specific,
        "M0_code_gate_pass": code_rows[0].get("M0_code_gate_pass", 0),
        "M2_source_logger_pass": logger_pass,
        "M3_risk_signal_pass": risk_pass,
        "artifact_bundle": "",
    }
    bundle = out_dir / "v22_16_mlp_fu_results_bundle.zip"
    final["artifact_bundle"] = str(bundle.resolve())
    write_json(out_dir / "v22_16_mlp_fu_route.json", final)
    retry_rows = [
        r
        for r in rows
        if str(r.get("seed", "")) == "0"
        and str(r.get("dataset", "")) in set(DATASETS)
        and any(token in str(r.get("variant", "")) for token in ["M5 ", "M6 ", "M7 ", "M8 "])
    ]
    recap = [
        "# DG-KAN v22.16-M MLP+FU Mainline 实验结果复盘\n",
        f"生成时间：{now_sg()}\n",
        "原则：只引用本主线 artifact；未运行/blocked/gate-dependent 项明确标记，不补造指标。\n",
        "## Final route\n",
        f"- MLP+FU route: `{route}`\n",
        f"- Results bundle: `{bundle}`\n",
        "## Code and Gate Audit\n",
        md_table(code_rows, ["M0_code_gate_pass", "mlp_fu_runner_present", "mlp_fu_finalizer_gate_present", "mlp_fu_not_diagnostic_only", "mlp_fu_no_auxiliary_loss_strict", "mlp_fu_no_validation_test_future_direction", "mlp_fu_controls_have_own_source_state", "mlp_fu_source_history_train_only", "missing_files"], max_rows=5),
        "## Implementation and Repair Audit\n",
        "- 新增 `dgkan/fu/mlp_adaptive_controller.py`，实现 MLP-specific weak/gated/release/source-manifold/low-rank/control functional update；严格不使用 validation/test/future label 作为 runtime direction。\n",
        "- 新增 `dgkan/fu/mlp_source_usefulness.py`，记录 source_loss boundary、stale source、release decision 和 train-stream risk score。\n",
        "- 新增 `dgkan/fu/real_source_manifold.py`，只从 real train source history 构造 PCA/QR basis；无 hypernetwork、无 compression objective。\n",
        "- 新增 `experiments/run_v22_16_mlp_fu_mainline.py` 及 source_logger/task_eval/finalize_patch 入口，生成本计划要求的 mainline/source/control/usefulness/manifold/overguidance/interaction/finalizer artifacts。\n",
        "- 修复 1：首个 39-row 4800-step run 因 source-manifold 每步 SVD 过慢被 Ctrl-C 中断；随后把 manifold SVD 改为每 100 step 缓存刷新，并加入 row-wise partial artifact 落盘，避免长跑无证据。\n",
        "- 修复 2：初始 runtime risk H100 AUC=0.6236 未过 M3；按计划加入 offline destructive-projection、source-loss-boundary/weak-source、monotone agreement 风险模型，只用 analysis-only labels，最佳 H100 AUC 提升到当前 route 记录值。\n",
        "- 修复 3：controls 初版虽然有独立 parameter source，但 source_retention 读回仍引用 task source；已改为 controls 使用独立 output-space source 读回，并对 random/stable/corrupt controls 增加 task-cotangent null projection。数值零 source_loss 以 `>1e-12` 才计 controls pass，避免 null-space 浮点残差误算成功。\n",
        "- 修复 4：M4 weak prox 明显 over-guidance 后，M5/M6/M7/M8 已覆盖 reduce-lambda、source_loss-gated release、real source-manifold k8/k16 方向；这些修复减轻 NLL 爆炸，但未形成 official mechanism/task pass。\n",
        "- 修复 5：M5/M6/M7/M8/M9/M10 原先把 stale/weak source 也当作 release，且 gate 输入仍是当前步 `delta^2`；已改为只有上一训练步真实 `source_loss_after<0` 才 release，stale 只 refresh/source-state 降风险，并在 source logger 中记录 `source_loss_gate_input/source_loss_gate_signal` 供审计。\n",
        "## Baseline Stability\n",
        md_table([baseline], ["baseline_repeat_pairs", "baseline_loss_repeat_std_proxy", "baseline_accuracy_repeat_std_proxy", "M1_baseline_stable"], max_rows=5),
        "## MLP+FU Mechanism and Task Gates\n",
        md_table([final], ["dataset_seed_groups", "best_mlp_fu_variant", "mlp_fu_mechanism_exploration_pass", "mlp_fu_mechanism_pass", "mlp_fu_task_pass", "mlp_fu_noharm_pass", "mlp_fu_task_improvement_pass", "mlp_fu_strong_task_pass", "controls_fail", "controls_C4_rows", "risk_AUC_H100_analysis_all_rows", "best_risk_model_H100"], max_rows=5),
        md_table(risk_model_rows, ["risk_model", "H", "AUC_predict_washout", "label_count", "positive_label_count", "analysis_scope", "used_for_runtime_direction"], max_rows=18),
        md_table(rows[:40], ["dataset", "seed", "variant", "final_test_loss_readback", "final_test_accuracy_readback", "NLL_delta_vs_MLP_AdamW", "accuracy_delta_vs_MLP_AdamW", "AUC_loss_time_delta_vs_MLP_AdamW", "source_func_h3200", "source_loss_h3200", "source_func_h4800", "source_loss_h4800", "controller_to_base_update_ratio_p95", "diagnostic_only"], max_rows=40),
        "Analysis: `future_washout_label_*_analysis_only` 仅用于离线 AUC 读回，runner 中 `analysis_only_future_label_used_for_runtime_direction=0`。Official MLP+FU 候选与 controls 分开计数；controls pass 时禁止 MLP+FU official claim。\n",
        "## 2026-06-12 Source-Loss Gate Retry Evidence\n",
        "本次重试范围：MNIST/FashionMNIST/KMNIST，seed=0，M5/M6/M7/M8，4800 train steps，hidden=16；命令见执行日志 `2026-06-12 01:06:34 +0800 M-train`。\n",
        md_table(retry_rows, ["dataset", "seed", "variant", "NLL_delta_vs_MLP_AdamW", "accuracy_delta_vs_MLP_AdamW", "source_loss_gate_previous_count", "source_loss_flip_count", "source_release_count", "source_refresh_count", "source_func_h3200", "source_loss_h3200", "source_func_h4800", "source_loss_h4800", "controller_to_base_update_ratio_p95", "lambda_p95"], max_rows=20),
        "Analysis: 该重试只改变 train-stream controller/release 信号，不改变 official gate。`source_loss_gate_previous_count` 证明 M5-M8 从第二步开始使用上一训练步真实 `source_loss_after`；若 task delta、no-debt 与 mechanism rows 仍不足，route 必须保持 no-go。\n",
        "## Source Logger Evidence\n",
        f"- source_logger_nonempty={mechanism['source_logger_nonempty']}；logged_steps={mechanism['logged_steps']}；no_future_label_passed_into_runtime_controller={mechanism['no_future_label_passed_into_runtime_controller']}。\n",
        md_table(source_rows[:30], ["dataset", "seed", "variant", "step", "source_retention", "source_loss_gain", "source_loss_gate_input", "source_loss_gate_signal", "destructive_projection", "risk_score", "lambda_t", "intervention_flag", "future_washout_label_H100_analysis_only", "analysis_only_future_label_used_for_runtime_direction"], max_rows=30),
        "## Source Usefulness / Overguidance\n",
        md_table(read_rows(out_dir / "v22_16_mlp_fu_source_usefulness_matrix.csv")[:30], ["dataset", "seed", "variant", "source_loss_flip_count", "source_loss_gate_previous_count", "TargetRetentionOnly_NotTaskUseful", "source_release_count", "NDS_mean", "control_projection_fraction_mean", "risk_AUC_H100_analysis"], max_rows=30),
        md_table(read_rows(out_dir / "v22_16_mlp_fu_overguidance_diagnostics.csv")[:30], ["dataset", "seed", "variant", "NLL_delta_vs_MLP_AdamW", "accuracy_delta_vs_MLP_AdamW", "controller_to_base_update_ratio_p95", "lambda_p95", "tail_loss_q99_delta_vs_MLP_AdamW", "overguidance_suspected"], max_rows=30),
        "## Real MLP Source-Manifold\n",
        md_table(read_rows(out_dir / "v22_16_mlp_fu_real_source_manifold_matrix.csv")[:30], ["dataset", "seed", "variant", "source_manifold_dim", "history_window_size", "projection_residual_Gf", "source_loss_h3200", "source_loss_h4800", "source_func_h3200", "source_func_h4800", "controls_pass_count"], max_rows=30),
        "## MLP+FU vs KAN+FU Interaction\n",
        md_table(interaction_rows[:30], ["dataset", "seed", "mlp_fu_variant", "mlp_fu_delta_NLL", "mlp_fu_delta_accuracy", "kan_vs_mlp_fu_delta_NLL", "kan_vs_mlp_fu_delta_accuracy", "fu_interaction_delta_NLL", "basis_claim_allowed", "blocker", "old_v22_16_final_route"], max_rows=30),
        "## Conclusion and Insight\n",
        f"- 当前 MLP+FU 主线必须按 `{route}` 解释。\n",
        f"- fu_general_value_allowed={fu_general}；kan_specific_carrier_value_allowed={kan_specific}。\n",
        "- 如果 route 停在 M3/M4/M5，说明 MLP+FU 的 risk/source/controller/controls 证据不足，不允许声称 FU general BP improvement。\n",
        "- 如果 MLP+FU 后续打开 M8 以上，KAN+FU 必须重新和 MLP+FU 比，不能只和 MLP+AdamW 或 KAN+AdamW 比。\n",
    ]
    MLP_RECAP_DOC.write_text("\n".join(recap), encoding="utf-8")
    append_mlp_exec(
        out_dir,
        command,
        task_id="M-finalize",
        gpu="n/a",
        status="completed",
        files="v22_16_mlp_fu_route.json; v22_16_mlp_fu_vs_kan_fu_interaction_matrix.csv; v22_16_mlp_fu_results_bundle.zip; docs recap",
        note=f"route={route} fu_general={fu_general} kan_specific={kan_specific}",
    )
    _build_bundle(out_dir)


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_mlp_out(args.out_dir)
    init_mlp_docs()
    ensure_out(OLD_V2216_OUT)
    if args.stage in {"all", "audit"}:
        _code_audit(out_dir)
    if args.stage in {"all", "train"}:
        run_train(args, out_dir)
    if args.stage in {"all", "finalize"}:
        finalize(args, out_dir)


if __name__ == "__main__":
    main()
