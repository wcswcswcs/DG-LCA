#!/usr/bin/env python3
"""v22.13 adapter horizon protocol for role-blind operator FU."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dgkan.fu.core import flat_params, load_flat_params  # noqa: E402
from dgkan.fu.loss_interface import GenericUpstreamCotangent, PolicyPreferenceAdapter, stable_random_delta_like  # noqa: E402
from dgkan.fu.operator_commit import commit_operator_target  # noqa: E402
from dgkan.fu.operator_horizon import HORIZONS  # noqa: E402
from dgkan.fu.operator_core import apply_operator  # noqa: E402
from dgkan.fu.source_loss import source_gate_row  # noqa: E402
from experiments.run_v22_11_arbitrary_loss_horizon import _make_model  # noqa: E402
from experiments.run_v22_11_arbitrary_loss_horizon import _loss_adapters  # noqa: E402
from experiments.run_v22_13_common import PYTHON, append_exec, ensure_out, int_flag, read_json, simple_svg, write_json, write_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    p.add_argument("--seed", type=int, default=2213)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--norm-scale", type=float, default=0.16)
    p.add_argument("--attempts", default="")
    return p


def _load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _device(name: str) -> torch.device:
    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _move_task_data(data: Any, device: torch.device) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: _move_task_data(v, device) for k, v in data.items()}
    return data


def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.item()) <= 1.0e-8:
        return 0.0
    return float((x @ y / denom).clamp(-1.0, 1.0).item())


def _snapshot_gpu(model: torch.nn.Module, x: torch.Tensor, base_logits: torch.Tensor, target_delta: torch.Tensor, adapter: Any, task_data: Any) -> dict[str, float]:
    with torch.no_grad():
        logits = model(x).detach().float()
        displacement = logits - base_logits.detach().float()
        loss_value = float(adapter.value(logits, task_data).detach().item())
    target_norm = torch.linalg.vector_norm(target_delta).clamp_min(1.0e-8)
    projected_gain = float(((displacement * target_delta).sum() / target_delta.square().sum().clamp_min(1.0e-8)).item())
    residual_ratio = float(torch.linalg.vector_norm(displacement - target_delta).item() / target_norm.item())
    alignment = _safe_cos(displacement, target_delta)
    return {
        "target_alignment": alignment,
        "target_residual_ratio": residual_ratio,
        "target_retention_score": alignment - residual_ratio,
        "target_projected_gain": projected_gain,
        "loss_value": loss_value,
        "geometry_energy": float(displacement.square().mean().item()),
    }


def _make_optimizer(model: torch.nn.Module, optimizer_name: str, lr_scale: float) -> torch.optim.Optimizer | None:
    if optimizer_name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=2.0e-3 * float(lr_scale), weight_decay=1.0e-4)
    if optimizer_name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=1.0e-2 * float(lr_scale))
    if optimizer_name == "Momentum":
        return torch.optim.SGD(model.parameters(), lr=1.0e-2 * float(lr_scale), momentum=0.9)
    if optimizer_name == "NoOp":
        return None
    raise ValueError(f"unknown optimizer {optimizer_name}")


def _matched_random_like(vec: torch.Tensor, seed: int, offset: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + int(offset))
    raw = torch.randn(vec.shape, generator=gen, dtype=vec.detach().cpu().dtype).to(device=vec.device, dtype=vec.dtype)
    return raw * (torch.linalg.vector_norm(vec).clamp_min(1.0e-8) / torch.linalg.vector_norm(raw).clamp_min(1.0e-8))


def _stable_like(vec: torch.Tensor) -> torch.Tensor:
    raw = torch.sin(torch.arange(vec.numel(), device=vec.device, dtype=vec.dtype)).reshape_as(vec)
    return raw * (torch.linalg.vector_norm(vec).clamp_min(1.0e-8) / torch.linalg.vector_norm(raw).clamp_min(1.0e-8))


def _control_updates_gpu(payload: dict[str, Any], commit_update: torch.Tensor, seed: int, device: torch.device) -> list[tuple[str, torch.Tensor, str]]:
    zero = torch.zeros_like(commit_update, device=device)
    random_vec = _matched_random_like(commit_update, seed, 711)
    stable_vec = _stable_like(commit_update)
    sign_flip = -commit_update
    corrupt = torch.roll(commit_update, shifts=max(1, int(commit_update.numel() // 11)), dims=0)
    corrupt = corrupt * (torch.linalg.vector_norm(commit_update).clamp_min(1.0e-8) / torch.linalg.vector_norm(corrupt).clamp_min(1.0e-8))
    model = _make_model(payload).to(device)
    x = payload["x"].detach().float().to(device)
    target = payload["target_delta"].detach().float().to(device)
    random_target = _matched_random_like(target, seed, 907)
    solver_random_update, _ = commit_operator_target(
        model,
        x,
        random_target,
        solver_level="v22_13_gpu_control_same_solver_random_target",
        block_role="readout_only",
        damping=1.0e-3,
        fit_scope="all_train_stream",
        seed=seed,
    )
    return [
        ("AdamW", zero, "AdamW"),
        ("SGD", zero, "SGD"),
        ("Momentum", zero, "Momentum"),
        ("NoOpMatchedOverhead", zero, "NoOp"),
        ("RandomMatchedNorm", random_vec, "AdamW"),
        ("StableRandom", stable_vec, "AdamW"),
        ("SameSolverRandomTarget", solver_random_update.detach().float().to(device), "AdamW"),
        ("SignFlipTarget", sign_flip, "AdamW"),
        ("CorruptTarget", corrupt, "AdamW"),
    ]


def _train_variant_gpu(
    payload: dict[str, Any],
    *,
    update_vec: torch.Tensor,
    optimizer_name: str,
    adapter: Any,
    task_data: Any,
    seed: int,
    lr_scale: float,
    periodic_update_vec: torch.Tensor | None,
    periodic_interval: int,
    periodic_scale: float,
    periodic_stop_step: int,
    retention_weight: float,
    retention_start_step: int,
    retention_stop_step: int,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    torch.manual_seed(int(seed))
    model = _make_model(payload).to(device)
    x = payload["x"].detach().float().to(device)
    target_delta = payload["target_delta"].detach().float().to(device)
    update = update_vec.detach().float().to(device)
    task_data = _move_task_data(task_data, device)
    before = flat_params(model).detach()
    with torch.no_grad():
        base_logits = model(x).detach().float()
        load_flat_params(model, before - update.to(dtype=before.dtype, device=before.device))
        source_anchor = (model(x).detach().float() - base_logits).detach()
    opt = _make_optimizer(model, optimizer_name, lr_scale)
    snapshots: dict[int, dict[str, float]] = {}
    max_h = max(HORIZONS)
    for step in range(1, max_h + 1):
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            logits_now = model(x).float()
            loss = adapter.value(logits_now, task_data)
            if float(retention_weight) > 0.0:
                retention_active = step >= int(retention_start_step) and (int(retention_stop_step) <= 0 or step <= int(retention_stop_step))
                if retention_active:
                    displacement_now = logits_now.float() - base_logits.to(device=logits_now.device, dtype=logits_now.dtype)
                    loss = loss + float(retention_weight) * F.mse_loss(displacement_now, source_anchor.to(device=logits_now.device, dtype=logits_now.dtype))
            loss.backward()
            opt.step()
        if periodic_update_vec is not None and periodic_interval > 0 and periodic_scale > 0.0:
            active = periodic_stop_step <= 0 or step <= int(periodic_stop_step)
            if active and step % int(periodic_interval) == 0:
                with torch.no_grad():
                    current = flat_params(model).detach()
                    load_flat_params(model, current - float(periodic_scale) * periodic_update_vec.to(device=current.device, dtype=current.dtype))
        if step in HORIZONS:
            snapshots[step] = _snapshot_gpu(model, x, base_logits, target_delta, adapter, task_data)
    return snapshots


def _evaluate_attempt_gpu(payload: dict[str, Any], attempt: dict[str, Any], adapter_name: str, adapter: Any, task_data: Any, controls: list[tuple[str, torch.Tensor, str]], seed: int, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    commit_update = payload["update_vec"].detach().float().to(device)
    task_data = _move_task_data(task_data, device)
    fu_snaps = _train_variant_gpu(
        payload,
        update_vec=commit_update,
        optimizer_name="AdamW",
        adapter=adapter,
        task_data=task_data,
        seed=seed,
        lr_scale=float(attempt["lr_scale"]),
        periodic_update_vec=commit_update if int(attempt["periodic_interval"]) > 0 else None,
        periodic_interval=int(attempt["periodic_interval"]),
        periodic_scale=float(attempt["periodic_scale"]),
        periodic_stop_step=int(attempt.get("periodic_stop_step", 0)),
        retention_weight=float(attempt.get("retention_weight", 0.0)),
        retention_start_step=int(attempt.get("retention_start_step", 0)),
        retention_stop_step=int(attempt.get("retention_stop_step", 0)),
        device=device,
    )
    control_rows: list[dict[str, Any]] = []
    control_snaps: dict[str, dict[int, dict[str, float]]] = {}
    for idx, (control_name, update, optimizer_name) in enumerate(controls):
        periodic_vec = update.detach().float().to(device) if int(attempt["periodic_interval"]) > 0 and torch.linalg.vector_norm(update).item() > 0.0 else None
        snaps = _train_variant_gpu(
            payload,
            update_vec=update.detach().float().to(device),
            optimizer_name=optimizer_name,
            adapter=adapter,
            task_data=task_data,
            seed=seed + idx + 1,
            lr_scale=float(attempt["lr_scale"]),
            periodic_update_vec=periodic_vec,
            periodic_interval=int(attempt["periodic_interval"]),
            periodic_scale=float(attempt["periodic_scale"]),
            periodic_stop_step=int(attempt.get("periodic_stop_step", 0)),
            retention_weight=float(attempt.get("retention_weight", 0.0)),
            retention_start_step=int(attempt.get("retention_start_step", 0)),
            retention_stop_step=int(attempt.get("retention_stop_step", 0)),
            device=device,
        )
        control_snaps[control_name] = snaps
        row: dict[str, Any] = {
            "horizon_status": "gpu_control_v22_13",
            "loss_adapter_name": adapter_name,
            "attempt": attempt["attempt"],
            "variant": control_name,
            "optimizer": optimizer_name,
            "loss_agnostic_contract_pass": 1,
            "uses_labels_for_direction": 0,
            "uses_loss_for_direction": 0,
            "device": str(device),
        }
        for h in HORIZONS:
            row[f"target_retention_score_h{h}"] = snaps[h]["target_retention_score"]
            row[f"loss_value_h{h}"] = snaps[h]["loss_value"]
        control_rows.append(row)
    best_func = {h: max(snaps[h]["target_retention_score"] for snaps in control_snaps.values()) for h in HORIZONS}
    best_loss = {h: min(snaps[h]["loss_value"] for snaps in control_snaps.values()) for h in HORIZONS}
    source_func = {h: fu_snaps[h]["target_retention_score"] - best_func[h] for h in HORIZONS}
    source_loss = {h: best_loss[h] - fu_snaps[h]["loss_value"] for h in HORIZONS}
    positive_counts = {
        h: sum(int(fu_snaps[h]["target_retention_score"] > snaps[h]["target_retention_score"] + 0.005) for snaps in control_snaps.values())
        for h in HORIZONS
    }
    control_equiv = {
        h: sum(int(abs(fu_snaps[h]["target_retention_score"] - snaps[h]["target_retention_score"]) <= 0.005) for snaps in control_snaps.values()) / max(1, len(control_snaps))
        for h in HORIZONS
    }
    auc_fu = sum(fu_snaps[h]["loss_value"] for h in HORIZONS)
    auc_best = sum(best_loss[h] for h in HORIZONS)
    source_h3200 = source_func[3200]
    source_h4800 = source_func[4800]
    r4800 = source_h4800 / source_h3200 if abs(source_h3200) > 1.0e-12 else ""
    loss_neutral_c3 = int(all(source_loss[h] >= -1.0e-6 for h in [800, 1600, 3200]))
    loss_neutral_c4 = int(source_loss[4800] >= -1.0e-6 and auc_fu <= 1.05 * max(auc_best, 1.0e-8))
    c3 = int(all(source_func[h] >= 0.005 for h in [100, 400, 800, 1600, 3200]) and loss_neutral_c3 and positive_counts[3200] >= 6 and control_equiv[3200] < 0.50)
    c4 = int(c3 and source_func[4800] >= 0.005 and isinstance(r4800, float) and r4800 >= 0.50 and positive_counts[4800] >= 7 and loss_neutral_c4)
    row: dict[str, Any] = {
        "horizon_status": "v22_13_gpu_adapter_specific_operator_horizon",
        "loss_adapter_name": adapter_name,
        "attempt": attempt["attempt"],
        "variant": "FU",
        "reason": "same role-blind operator law is applied to this adapter cotangent; output direction is adapter-specific T(delta)",
        "control_count": len(control_snaps),
        "loss_agnostic_contract_pass": 1,
        "uses_labels_for_direction": 0,
        "uses_loss_for_direction": 0,
        "uses_loss_formula_specific_direction": 0,
        "periodic_interval": attempt["periodic_interval"],
        "periodic_scale": attempt["periodic_scale"],
        "periodic_stop_step": attempt.get("periodic_stop_step", 0),
        "retention_weight": attempt.get("retention_weight", 0.0),
        "retention_start_step": attempt.get("retention_start_step", 0),
        "retention_stop_step": attempt.get("retention_stop_step", 0),
        "lr_scale": attempt["lr_scale"],
        "R4800_over_3200_func": r4800,
        "R4800_over_3200_loss": source_loss[4800] / source_loss[3200] if abs(source_loss[3200]) > 1.0e-12 else "",
        "AUC_loss_step": auc_fu,
        "AUC_loss_time": auc_fu,
        "AUC_loss_time_ratio_vs_best_control": auc_fu / max(auc_best, 1.0e-8),
        "source_state_alignment": fu_snaps[3200]["target_alignment"],
        "source_state_decay_rate": 1.0 - (source_func[4800] / source_func[3200] if abs(source_func[3200]) > 1.0e-12 else 0.0),
        "optimizer_destructive_projection": max(0.0, 1.0 - (source_func[3200] / source_func[100] if abs(source_func[100]) > 1.0e-12 else 0.0)),
        "device": str(device),
        "C3_source_formation_pass": c3,
        "C4_terminal_retention_pass": c4,
        "TargetRetentionOnly_NotTaskUseful": int(any(source_func[h] >= 0.005 for h in [3200, 4800]) and not loss_neutral_c3),
        "blocker": "" if c3 else "source_func_or_source_loss_or_control_gate_failed",
    }
    for h in HORIZONS:
        row[f"source_func_h{h}"] = source_func[h]
        row[f"source_loss_h{h}"] = source_loss[h]
        row[f"row_positive_count_h{h}"] = positive_counts[h]
        row[f"control_equivalent_fraction_h{h}"] = control_equiv[h]
        row[f"FU_loss_value_h{h}"] = fu_snaps[h]["loss_value"]
        row[f"best_control_loss_value_h{h}"] = best_loss[h]
        row[f"FU_target_retention_score_h{h}"] = fu_snaps[h]["target_retention_score"]
        row[f"best_control_target_retention_score_h{h}"] = best_func[h]
    return row, control_rows


def _adapters(payload: dict[str, Any], seed: int) -> list[tuple[str, Any, Any, int]]:
    adapters: list[tuple[str, Any, Any, int]] = [(name, adapter, data, 1) for name, adapter, data in _loss_adapters(payload)]
    logits = payload["logits"].detach().float()
    target = payload["target_delta"].detach().float()
    labels = payload.get("labels_for_loss_adapter_only")
    if labels is None:
        labels = torch.arange(int(logits.shape[0])) % int(logits.shape[-1])
    half = max(1, int(logits.shape[0]) // 2)
    pref_data = {"chosen": torch.arange(0, half), "rejected": torch.arange(half, min(int(logits.shape[0]), 2 * half))}
    adapters.append(("Delta-PreferenceAdapter-smoke", PolicyPreferenceAdapter(beta=0.1), pref_data, 0))
    norm_target = -target / torch.linalg.vector_norm(target).clamp_min(1.0e-8) * (float(target.numel()) ** 0.5)
    adapters.append(("Delta-GenericSourceTarget-normalized-holdout", GenericUpstreamCotangent(norm_target, "GenericSourceTargetNormalized"), None, 0))
    stable = stable_random_delta_like(logits, seed=seed, kind="stable")
    random = stable_random_delta_like(logits, seed=seed + 17, kind="gaussian")
    adapters.append(("Delta-StableRandom-control", GenericUpstreamCotangent(stable, "StableRandomControl"), None, 0))
    adapters.append(("Delta-RandomMatched-control", GenericUpstreamCotangent(random, "RandomMatchedControl"), None, 0))
    return adapters


def _attempts() -> list[dict[str, Any]]:
    return [
        {"attempt": "role_blind_initial_commit_lr1", "lr_scale": 1.0, "periodic_interval": 0, "periodic_scale": 0.0, "periodic_stop_step": 0, "retention_weight": 0.0, "retention_start_step": 0, "retention_stop_step": 0, "source_state_attempt": 0},
        {"attempt": "low_lr_control_null_retention_lr0p2", "lr_scale": 0.2, "periodic_interval": 0, "periodic_scale": 0.0, "periodic_stop_step": 0, "retention_weight": 0.0, "retention_start_step": 0, "retention_stop_step": 0, "source_state_attempt": 1},
        {"attempt": "slow_source_state_fixed_800x0p006", "lr_scale": 0.08, "periodic_interval": 800, "periodic_scale": 0.006, "periodic_stop_step": 4800, "retention_weight": 0.0, "retention_start_step": 0, "retention_stop_step": 0, "source_state_attempt": 1},
        {"attempt": "slow_source_anchor_w0p02_800x0p006", "lr_scale": 0.08, "periodic_interval": 800, "periodic_scale": 0.006, "periodic_stop_step": 4800, "retention_weight": 0.02, "retention_start_step": 1, "retention_stop_step": 6400, "source_state_attempt": 1},
        {"attempt": "slow_source_anchor_w0p10_800x0p006", "lr_scale": 0.08, "periodic_interval": 800, "periodic_scale": 0.006, "periodic_stop_step": 4800, "retention_weight": 0.10, "retention_start_step": 1, "retention_stop_step": 6400, "source_state_attempt": 1},
        {"attempt": "low_lr_source_anchor_w0p10_lr0p02", "lr_scale": 0.02, "periodic_interval": 800, "periodic_scale": 0.006, "periodic_stop_step": 4800, "retention_weight": 0.10, "retention_start_step": 1, "retention_stop_step": 6400, "source_state_attempt": 1},
    ]


def _decorate(row: dict[str, Any], adapter_seen: int, is_control: int, operator_id: str) -> dict[str, Any]:
    row["run_id"] = "v22_13_operator_horizon"
    row["dataset"] = "synthetic_train_stream"
    row["carrier"] = "MLP"
    row["model_family"] = "TinyArbitraryLossMLP"
    row["operator_id"] = operator_id
    row["operator_family"] = "role_blind_loss_interface_operator"
    row["adapter_is_control"] = is_control
    row["adapter_seen_in_operator_tuning"] = adapter_seen
    row["same_operator_parameters_across_adapters"] = 1
    row["uses_adapter_name_for_direction"] = 0
    row["uses_loss_formula_for_direction"] = 0
    row["uses_labels_for_fu_core"] = 0
    row["uses_validation_test_future_query"] = 0
    row["uses_audit_metric_for_direction"] = 0
    src = {h: float(row.get(f"source_func_h{h}", -999.0)) for h in HORIZONS if row.get(f"source_func_h{h}", "") != ""}
    los = {h: float(row.get(f"source_loss_h{h}", -999.0)) for h in HORIZONS if row.get(f"source_loss_h{h}", "") != ""}
    row.update(source_gate_row(src, los))
    row["source_loss_boundary_active_fraction"] = int("source" in str(row.get("attempt", "")))
    row["boundary_term_energy"] = ""
    return row


def _payload_for_adapter(base_payload: dict[str, Any], operator_id: str, adapter: Any, task_data: Any, seed: int, device: torch.device, norm_scale: float = 0.16) -> tuple[dict[str, Any], dict[str, Any]]:
    model = _make_model(base_payload).to(device)
    x = base_payload["x"].detach().float().to(device)
    logits = base_payload["logits"].detach().float().to(device)
    task_data = _move_task_data(task_data, device)
    delta = adapter.cotangent(logits, task_data).detach().float()
    target, telem = apply_operator(operator_id=operator_id, logits=logits, cotangent=delta, seed=seed, norm_scale=float(norm_scale))
    update, diag = commit_operator_target(
        model,
        x,
        target,
        solver_level=f"v22_13_horizon_{operator_id}_adapter_cotangent_commit",
        block_role="readout_only",
        damping=1.0e-3,
        fit_scope="all_train_stream",
        seed=seed,
    )
    payload = dict(base_payload)
    payload["target_delta"] = target.detach().float()
    payload["update_vec"] = update.detach().float()
    return payload, {**telem.to_row(), **diag}


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    device = _device(args.device)
    commit_route = read_json(source_dir / "v22_13_operator_commit_route.json")
    rows: list[dict[str, Any]] = []
    if int_flag(commit_route.get("S4_operator_metric_commit_pass_rows")) <= 0:
        route = {
            "route": "S5-BlockedBeforeOperatorHorizon",
            "C3_adapter_pass_count": 0,
            "C4_adapter_pass_count": 0,
            "blocker": "S4_operator_metric_commit_gate_failed",
            "promotion_allowed": 0,
        }
    else:
        payload = _load(source_dir / "v22_13_operator_commit_payload.pt")
        operator_id = str(payload.get("operator_variational_route", {}).get("selected_operators", "")).split(";")[0] or "role_blind_operator"
        for adapter_idx, (display, adapter, task_data, seen) in enumerate(_adapters(payload, int(args.seed))):
            is_control = int(display.endswith("-control"))
            adapter_payload, adapter_commit_diag = _payload_for_adapter(payload, operator_id, adapter, task_data, int(args.seed) + adapter_idx * 1000, device, float(args.norm_scale))
            controls = _control_updates_gpu(adapter_payload, adapter_payload["update_vec"].detach().float().to(device), int(args.seed) + adapter_idx * 1000, device)
            attempt_filter = {x.strip() for x in str(args.attempts).split(",") if x.strip()}
            attempts = [a for a in _attempts() if not attempt_filter or str(a.get("attempt")) in attempt_filter]
            for attempt_idx, attempt in enumerate(attempts):
                row, control_rows = _evaluate_attempt_gpu(adapter_payload, attempt, display, adapter, task_data, controls, int(args.seed) + adapter_idx * 1000 + attempt_idx * 100, device)
                row = _decorate(row, seen, is_control, operator_id)
                row.update({f"adapter_commit_{k}": v for k, v in adapter_commit_diag.items() if k in {"projection_residual_Gf", "ActuationR2", "function_displacement_cos_with_target", "operator_family", "NDS_reduction", "control_projection_after"}})
                rows.append(row)
                for control in control_rows:
                    control = _decorate(control, seen, 1, operator_id)
                    rows.append(control)
        fu_rows = [r for r in rows if str(r.get("variant")) == "FU"]
        non_random = [r for r in fu_rows if not int_flag(r.get("adapter_is_control"))]
        c3_adapters = sorted({str(r.get("loss_adapter_name")) for r in non_random if int_flag(r.get("C3_source_formation_pass"))})
        c4_adapters = sorted({str(r.get("loss_adapter_name")) for r in non_random if int_flag(r.get("C4_terminal_retention_pass"))})
        stable_pass = any(int_flag(r.get("C3_source_formation_pass")) or int_flag(r.get("C4_terminal_retention_pass")) for r in fu_rows if int_flag(r.get("adapter_is_control")))
        route_name = "S5-RoleBlindOperatorHorizonPass" if len(c3_adapters) >= 2 else "S5-RoleBlindOperatorHorizonNoGo"
        if stable_pass:
            route_name = "S5-ControlEquivalentOrSourceDefinitionBroken"
        route = {
            "route": route_name,
            "FU_attempt_rows": len(fu_rows),
            "C3_source_formation_pass_rows": sum(int_flag(r.get("C3_source_formation_pass")) for r in fu_rows),
            "C4_terminal_retention_pass_rows": sum(int_flag(r.get("C4_terminal_retention_pass")) for r in fu_rows),
            "C3_adapter_pass_count": len(c3_adapters),
            "C4_adapter_pass_count": len(c4_adapters),
            "C3_adapter_pass_list": ";".join(c3_adapters),
            "C4_adapter_pass_list": ";".join(c4_adapters),
            "StableRandom_or_RandomMatched_pass": int(stable_pass),
            "adapter_holdout_pass": int(any("Preference" in a or "normalized" in a for a in c3_adapters)),
            "same_operator_parameters_across_adapters": 1,
            "uses_adapter_name_for_direction": 0,
            "uses_loss_formula_for_direction": 0,
            "norm_scale": float(args.norm_scale),
            "promotion_allowed": 0,
            "blocker": "" if route_name == "S5-RoleBlindOperatorHorizonPass" else ";".join(dict.fromkeys(str(r.get("blocker")) for r in fu_rows if r.get("blocker"))),
        }
    write_rows(out_dir / "v22_13_adapter_horizon_matrix.csv", rows)
    write_rows(out_dir / "v22_13_adapter_holdout_matrix.csv", [r for r in rows if "Preference" in str(r.get("loss_adapter_name")) or "normalized" in str(r.get("loss_adapter_name"))])
    write_rows(out_dir / "v22_13_source_loss_boundary_matrix.csv", [r for r in rows if int_flag(r.get("source_loss_boundary_active_fraction"))])
    write_rows(out_dir / "v22_13_source_state_dynamics.csv", [r for r in rows if int_flag(r.get("source_state_attempt")) or "source_state" in str(r.get("attempt"))])
    write_rows(out_dir / "v22_13_control_attribution_matrix.csv", [r for r in rows if int_flag(r.get("adapter_is_control")) or str(r.get("variant")) != "FU"])
    write_json(out_dir / "v22_13_operator_horizon_route.json", route)
    simple_svg(out_dir / "figures/v22_13_adapter_horizon_source_func.svg", "v22.13 adapter horizon source_func", rows, "source_func_h3200")
    simple_svg(out_dir / "figures/v22_13_adapter_horizon_source_loss.svg", "v22.13 adapter horizon source_loss", rows, "source_loss_h3200")
    simple_svg(out_dir / "figures/v22_13_control_projection_dashboard.svg", "v22.13 controls", rows, "source_func_h3200")
    append_exec(out_dir, f"{PYTHON} experiments/run_v22_13_operator_horizon.py --source-dir {source_dir} --out-dir {out_dir} --seed {int(args.seed)} --device {args.device} --norm-scale {float(args.norm_scale)} --attempts {args.attempts}", status="completed" if int_flag(route.get("C3_adapter_pass_count")) >= 2 else "blocked", note=f"route={route['route']} c3={route.get('C3_adapter_pass_count')} c4={route.get('C4_adapter_pass_count')} device={device} norm_scale={float(args.norm_scale)} attempts={args.attempts} blocker={route.get('blocker')}")


if __name__ == "__main__":
    main()
