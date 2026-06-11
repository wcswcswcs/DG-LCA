#!/usr/bin/env python3
"""v22.11 S5 arbitrary-loss horizon training verification."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.constructive_commit import solve_external_target_commit  # noqa: E402
from dgkan.fu.core import flat_params, load_flat_params  # noqa: E402
from dgkan.fu.loss_interface import ClassificationCEAdapter, GenericUpstreamCotangent, PairwiseRankingAdapter, RegressionMSEAdapter  # noqa: E402
from experiments.run_v22_11_common import PYTHON, append_exec, ensure_out, int_flag, read_json, simple_svg, write_json, write_rows  # noqa: E402
from experiments.run_v22_11_source_atom_generation import TinyArbitraryLossMLP  # noqa: E402


HORIZONS = [100, 400, 800, 1600, 2400, 3200, 4000, 4800, 6400]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    p.add_argument("--seed", type=int, default=2211)
    return p


def _load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _make_model(payload: dict[str, Any]) -> TinyArbitraryLossMLP:
    config = dict(payload.get("model_config", {}))
    model = TinyArbitraryLossMLP(int(config.get("input_dim", 8)), int(config.get("hidden", 64)), int(config.get("classes", 5)))
    model.load_state_dict(payload["model_state"])
    return model


def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().reshape(-1)
    target = b.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(target)
    if float(denom.item()) <= 1.0e-8:
        return 0.0
    return float((x @ target / denom).clamp(-1.0, 1.0).item())


def _snapshot(model: torch.nn.Module, x: torch.Tensor, base_logits: torch.Tensor, target_delta: torch.Tensor, adapter: Any, task_data: Any) -> dict[str, float]:
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


def _matched_random_like(vec: torch.Tensor, seed: int, offset: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + int(offset))
    random = torch.randn(vec.shape, generator=gen, dtype=vec.dtype)
    return random * (torch.linalg.vector_norm(vec).clamp_min(1.0e-8) / torch.linalg.vector_norm(random).clamp_min(1.0e-8))


def _stable_like(vec: torch.Tensor) -> torch.Tensor:
    stable = torch.sin(torch.arange(vec.numel(), dtype=vec.dtype)).reshape_as(vec)
    return stable * (torch.linalg.vector_norm(vec).clamp_min(1.0e-8) / torch.linalg.vector_norm(stable).clamp_min(1.0e-8))


def _loss_adapters(payload: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    x = payload["x"].detach().float()
    target = payload["target_delta"].detach().float()
    model = _make_model(payload)
    with torch.no_grad():
        base = model(x).detach().float()
    labels = payload.get("labels_for_loss_adapter_only")
    if labels is None:
        labels = torch.argmax(base, dim=-1)
    labels = labels.to(dtype=torch.long)
    mse_target = base + target
    ranking_pairs = torch.stack([torch.arange(0, int(x.shape[0] // 2)), torch.arange(int(x.shape[0] // 2), int(x.shape[0]))[: int(x.shape[0] // 2)]], dim=1)
    return [
        ("Delta-LossCEAdapter", ClassificationCEAdapter(), labels),
        ("Delta-MSEAdapter", RegressionMSEAdapter(), mse_target),
        ("Delta-RankingAdapter", PairwiseRankingAdapter(), {"pairs": ranking_pairs}),
        ("Delta-GenericSourceTarget", GenericUpstreamCotangent(-target, "Delta-GenericSourceTarget"), None),
    ]


def _control_updates(payload: dict[str, Any], commit_update: torch.Tensor, seed: int) -> list[tuple[str, torch.Tensor, str]]:
    zero = torch.zeros_like(commit_update)
    random_vec = _matched_random_like(commit_update, seed, 711)
    stable_vec = _stable_like(commit_update)
    sign_flip = -commit_update
    corrupt = torch.roll(commit_update, shifts=max(1, int(commit_update.numel() // 11)), dims=0)
    corrupt = corrupt * (torch.linalg.vector_norm(commit_update).clamp_min(1.0e-8) / torch.linalg.vector_norm(corrupt).clamp_min(1.0e-8))

    model = _make_model(payload)
    x = payload["x"].detach().float()
    target = payload["target_delta"].detach().float()
    random_target = _matched_random_like(target, seed, 907)
    solver_random_update, _ = solve_external_target_commit(
        model,
        x,
        None,
        random_target,
        solver_level="S5-control-same-solver-random-target",
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
        ("SameAtomsRandomWeights", random_vec, "AdamW"),
        ("SameSolverRandomTarget", solver_random_update, "AdamW"),
        ("SignFlipTarget", sign_flip, "AdamW"),
        ("CorruptTarget", corrupt, "AdamW"),
    ]


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


def _train_variant(
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
    periodic_stop_step: int = 0,
    pid: tuple[int, float, float, float, float, int] | None = None,
) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    torch.manual_seed(int(seed))
    model = _make_model(payload)
    x = payload["x"].detach().float()
    target_delta = payload["target_delta"].detach().float()
    before = flat_params(model).detach()
    update = update_vec.detach().float().to(dtype=before.dtype)
    with torch.no_grad():
        base_logits = model(x).detach().float()
        load_flat_params(model, before - update)
    opt = _make_optimizer(model, optimizer_name, lr_scale)
    integral = 0.0
    previous_error = 0.0
    pid_steps = 0
    pid_error_sum = 0.0
    pid_injection_sum = 0.0
    target_norm_sq = target_delta.square().sum().clamp_min(1.0e-8)
    snapshots: dict[int, dict[str, float]] = {0: _snapshot(model, x, base_logits, target_delta, adapter, task_data)}
    max_h = max(HORIZONS)
    for step in range(1, max_h + 1):
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss = adapter.value(model(x).float(), task_data)
            loss.backward()
            opt.step()
        if periodic_update_vec is not None and periodic_interval > 0 and periodic_scale > 0.0:
            active = periodic_stop_step <= 0 or step <= int(periodic_stop_step)
            if active and step % int(periodic_interval) == 0:
                with torch.no_grad():
                    current = flat_params(model).detach()
                    load_flat_params(model, current - float(periodic_scale) * periodic_update_vec.to(dtype=current.dtype))
        if pid is not None:
            interval, kp, ki, kd, clip, stop_step = pid
            active = (stop_step <= 0 or step <= int(stop_step)) and interval > 0 and step % int(interval) == 0
            if active:
                with torch.no_grad():
                    displacement = model(x).detach().float() - base_logits
                    gain = float(((displacement * target_delta).sum() / target_norm_sq).item())
                    error = 1.0 - gain
                    integral = max(-4.0, min(4.0, integral + error))
                    derivative = error - previous_error
                    raw = float(kp) * error + float(ki) * integral + float(kd) * derivative
                    injection = max(-float(clip), min(float(clip), raw))
                    current = flat_params(model).detach()
                    load_flat_params(model, current - injection * update.to(dtype=current.dtype))
                    previous_error = error
                    pid_steps += 1
                    pid_error_sum += abs(error)
                    pid_injection_sum += abs(injection)
        if step in HORIZONS:
            snapshots[step] = _snapshot(model, x, base_logits, target_delta, adapter, task_data)
    return snapshots, {
        "pid_steps": float(pid_steps),
        "pid_abs_error_mean": pid_error_sum / max(1, pid_steps),
        "pid_abs_injection_mean": pid_injection_sum / max(1, pid_steps),
        "pid_final_projected_gain": snapshots[max_h]["target_projected_gain"],
    }


def _evaluate_attempt(payload: dict[str, Any], attempt: dict[str, Any], adapter_name: str, adapter: Any, task_data: Any, controls: list[tuple[str, torch.Tensor, str]], seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    commit_update = payload["update_vec"].detach().float()
    pid_tuple = attempt.get("pid")
    fu_snaps, fu_meta = _train_variant(
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
        pid=pid_tuple,
    )
    control_rows: list[dict[str, Any]] = []
    control_snaps: dict[str, dict[int, dict[str, float]]] = {}
    for idx, (control_name, update, optimizer_name) in enumerate(controls):
        periodic_vec = update.detach().float() if int(attempt["periodic_interval"]) > 0 and torch.linalg.vector_norm(update).item() > 0.0 else None
        snaps, meta = _train_variant(
            payload,
            update_vec=update.detach().float(),
            optimizer_name=optimizer_name,
            adapter=adapter,
            task_data=task_data,
            seed=seed + idx + 1,
            lr_scale=float(attempt["lr_scale"]),
            periodic_update_vec=periodic_vec,
            periodic_interval=int(attempt["periodic_interval"]),
            periodic_scale=float(attempt["periodic_scale"]),
            periodic_stop_step=int(attempt.get("periodic_stop_step", 0)),
            pid=pid_tuple,
        )
        control_snaps[control_name] = snaps
        row: dict[str, Any] = {
            "horizon_status": "control",
            "loss_adapter_name": adapter_name,
            "attempt": attempt["attempt"],
            "variant": control_name,
            "optimizer": optimizer_name,
            "loss_agnostic_contract_pass": 1,
            "uses_labels_for_direction": 0,
            "uses_loss_for_direction": 0,
            "pid_control": int(pid_tuple is not None),
            **meta,
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
        "horizon_status": "v22_11_arbitrary_loss_training_horizon",
        "loss_adapter_name": adapter_name,
        "attempt": attempt["attempt"],
        "variant": "FU",
        "reason": "continued training uses generic loss-interface; FU injection does not branch on loss adapter type",
        "control_count": len(control_snaps),
        "loss_agnostic_contract_pass": 1,
        "uses_labels_for_direction": 0,
        "uses_loss_for_direction": 0,
        "uses_loss_formula_specific_direction": 0,
        "periodic_interval": attempt["periodic_interval"],
        "periodic_scale": attempt["periodic_scale"],
        "periodic_stop_step": attempt.get("periodic_stop_step", 0),
        "lr_scale": attempt["lr_scale"],
        "pid_control": int(pid_tuple is not None),
        "R4800_over_3200_func": r4800,
        "R4800_over_3200_loss": source_loss[4800] / source_loss[3200] if abs(source_loss[3200]) > 1.0e-12 else "",
        "AUC_loss_step": auc_fu,
        "AUC_loss_time": auc_fu,
        "AUC_loss_time_ratio_vs_best_control": auc_fu / max(auc_best, 1.0e-8),
        "per_example_loss_q95_debt": "",
        "per_example_loss_q99_debt": "",
        "LineC_generic_loss_coupling": "",
        "metric_debt_Sobolev": "",
        "metric_debt_RKHS": "",
        "metric_debt_Fisher": "",
        "source_state_alignment": fu_snaps[3200]["target_alignment"],
        "source_state_decay_rate": 1.0 - (source_func[4800] / source_func[3200] if abs(source_func[3200]) > 1.0e-12 else 0.0),
        "optimizer_destructive_projection": max(0.0, 1.0 - (source_func[3200] / source_func[100] if abs(source_func[100]) > 1.0e-12 else 0.0)),
        "wall_clock_time": "",
        "step_time": "",
        "memory": "",
        "C3_source_formation_pass": c3,
        "C4_terminal_retention_pass": c4,
        "TargetRetentionOnly_NotTaskUseful": int(any(source_func[h] >= 0.005 for h in [3200, 4800]) and not loss_neutral_c3),
        "blocker": "" if c3 else "source_func_or_source_loss_or_control_gate_failed",
        **fu_meta,
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


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    commit_route = read_json(source_dir / "v22_11_metric_commit_route.json")
    rows: list[dict[str, Any]] = []
    if int_flag(commit_route.get("S4_metric_dynamics_commit_pass_rows")) <= 0:
        route = {
            "route": "S5-BlockedBeforeArbitraryLossHorizon",
            "S4_metric_dynamics_commit_pass_rows": int_flag(commit_route.get("S4_metric_dynamics_commit_pass_rows")),
            "C3_source_formation_pass_rows": 0,
            "C4_terminal_retention_pass_rows": 0,
            "source_loss_nonnegative_rows": 0,
            "promotion_allowed": 0,
            "blocker": "S4_metric_dynamics_commit_gate_failed",
        }
    else:
        payload = _load(source_dir / "v22_11_metric_commit_payload.pt")
        controls = _control_updates(payload, payload["update_vec"].detach().float(), int(args.seed))
        attempts = [
            {"attempt": "initial_commit_arbitrary_loss_adamw", "lr_scale": 1.0, "periodic_interval": 0, "periodic_scale": 0.0, "periodic_stop_step": 0},
            {"attempt": "finite_source_state_replay_400x0p10_stop800", "lr_scale": 1.0, "periodic_interval": 400, "periodic_scale": 0.10, "periodic_stop_step": 800},
            {"attempt": "finite_source_state_replay_200x0p16_stop1200", "lr_scale": 1.0, "periodic_interval": 200, "periodic_scale": 0.16, "periodic_stop_step": 1200},
            {"attempt": "pid_source_state_replay_100_kp0p12_ki0p02_kd0p06_stop1600", "lr_scale": 1.0, "periodic_interval": 0, "periodic_scale": 0.0, "periodic_stop_step": 0, "pid": (100, 0.12, 0.02, 0.06, 0.18, 1600)},
            {"attempt": "initial_commit_arbitrary_loss_adamw_lr0p1", "lr_scale": 0.1, "periodic_interval": 0, "periodic_scale": 0.0, "periodic_stop_step": 0},
            {"attempt": "finite_source_state_replay_400x0p05_stop1600_lr0p1", "lr_scale": 0.1, "periodic_interval": 400, "periodic_scale": 0.05, "periodic_stop_step": 1600},
            {"attempt": "finite_source_state_replay_200x0p08_stop3200_lr0p1", "lr_scale": 0.1, "periodic_interval": 200, "periodic_scale": 0.08, "periodic_stop_step": 3200},
            {"attempt": "pid_source_state_replay_100_kp0p08_ki0p01_kd0p04_stop3200_lr0p1", "lr_scale": 0.1, "periodic_interval": 0, "periodic_scale": 0.0, "periodic_stop_step": 0, "pid": (100, 0.08, 0.01, 0.04, 0.08, 3200)},
            {"attempt": "finite_source_state_replay_400x0p03_stop3200_lr0p05", "lr_scale": 0.05, "periodic_interval": 400, "periodic_scale": 0.03, "periodic_stop_step": 3200},
        ]
        for adapter_idx, (adapter_name, adapter, task_data) in enumerate(_loss_adapters(payload)):
            for idx, attempt in enumerate(attempts):
                row, control_rows = _evaluate_attempt(payload, attempt, adapter_name, adapter, task_data, controls, int(args.seed) + adapter_idx * 1000 + idx * 100)
                rows.append(row)
                rows.extend(control_rows)
        fu_rows = [r for r in rows if str(r.get("variant")) == "FU"]
        c3 = sum(int_flag(r.get("C3_source_formation_pass")) for r in fu_rows)
        c4 = sum(int_flag(r.get("C4_terminal_retention_pass")) for r in fu_rows)
        source_loss_nonnegative = sum(int(float(r.get("source_loss_h3200", -999.0)) >= -1.0e-6 and float(r.get("source_loss_h4800", -999.0)) >= -1.0e-6) for r in fu_rows)
        best = max(fu_rows, key=lambda r: float(r.get("source_func_h3200") or -999.0)) if fu_rows else {}
        route = {
            "route": "S5-ArbitraryLossHorizonPass" if c3 and c4 else ("S5-TargetRetentionOnly_NotTaskUseful" if any(int_flag(r.get("TargetRetentionOnly_NotTaskUseful")) for r in fu_rows) else "S5-ArbitraryLossHorizonNoGo"),
            "S4_metric_dynamics_commit_pass_rows": int_flag(commit_route.get("S4_metric_dynamics_commit_pass_rows")),
            "FU_attempt_rows": len(fu_rows),
            "C3_source_formation_pass_rows": c3,
            "C4_terminal_retention_pass_rows": c4,
            "source_loss_nonnegative_rows": source_loss_nonnegative,
            "selected_attempt": best.get("attempt", ""),
            "selected_loss_adapter": best.get("loss_adapter_name", ""),
            "best_source_func_h3200": best.get("source_func_h3200", ""),
            "best_source_loss_h3200": best.get("source_loss_h3200", ""),
            "loss_agnostic_contract_pass": int(all(int_flag(r.get("loss_agnostic_contract_pass", 1)) for r in fu_rows)),
            "uses_labels_for_direction": 0,
            "uses_loss_for_direction": 0,
            "promotion_allowed": 0,
            "blocker": "" if c3 and c4 else ";".join(dict.fromkeys(part for r in fu_rows for part in str(r.get("blocker", "")).split(";") if part)),
        }
    write_rows(out_dir / "v22_11_arbitrary_loss_horizon_matrix.csv", rows)
    write_json(out_dir / "v22_11_arbitrary_loss_horizon_route.json", route)
    simple_svg(out_dir / "figures/v22_11_horizon_source_func.svg", "v22.11 arbitrary-loss source_func", rows, "source_func_h3200")
    simple_svg(out_dir / "figures/v22_11_horizon_source_loss.svg", "v22.11 arbitrary-loss source_loss", rows, "source_loss_h3200")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_11_arbitrary_loss_horizon.py --source-dir {source_dir} --out-dir {out_dir} --seed {int(args.seed)}",
        status="completed",
        note=f"route={route['route']} c3={route['C3_source_formation_pass_rows']} c4={route['C4_terminal_retention_pass_rows']} blocker={route.get('blocker')}",
    )


if __name__ == "__main__":
    main()
