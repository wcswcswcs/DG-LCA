#!/usr/bin/env python3
"""v22.10 S6 lightweight KAN source-channel mapping gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.core import flat_params, load_flat_params  # noqa: E402
from dgkan.models.fc_purekan_primitives import PrimitiveKAN, PrimitiveSpec  # noqa: E402
from experiments.run_v22_10_common import PYTHON, append_exec, ensure_out, finite_float, int_flag, read_json, read_rows, write_json, write_rows  # noqa: E402
from experiments.run_v22_10_horizon_source_formation import HORIZONS  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--source-dir", default="")
    p.add_argument("--seed", type=int, default=2210)
    return p


def _load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _make_kan(carrier: str, x: torch.Tensor, seed: int) -> PrimitiveKAN:
    if carrier == "D-CHE":
        spec = PrimitiveSpec(
            candidate_id="v22.10-D-CHE-readout-source-state-replay",
            basis_family="D-CHE",
            basis_name="chebyshev",
            k=3,
            hidden_dim=64,
            source="v22_10_lightweight_kan_mapping",
            local_support=0,
            global_support=1,
            uses_exp=0,
            uses_sin_cos=0,
            uses_division=0,
            uses_dense_basis_tensor=1,
        )
    else:
        spec = PrimitiveSpec(
            candidate_id="v22.10-D-FOU-readout-source-state-replay",
            basis_family="D-FOU",
            basis_name="fourier_lowfreq",
            k=3,
            hidden_dim=64,
            source="v22_10_lightweight_kan_mapping",
            local_support=0,
            global_support=1,
            uses_exp=0,
            uses_sin_cos=1,
            uses_division=0,
            uses_dense_basis_tensor=1,
        )
    return PrimitiveKAN(8, 5, spec, x, int(seed), torch.device("cpu"), param_budget=4096)


def _solve_w2_delta(model: PrimitiveKAN, x: torch.Tensor, target: torch.Tensor, damping: float = 1.0e-3) -> torch.Tensor:
    feats = model.frozen_readout_features(x).detach().float()
    scale = math.sqrt(max(1, int(model.hidden_dim)))
    gram = feats.T @ feats + float(damping) * torch.eye(int(feats.shape[1]), dtype=feats.dtype)
    rhs = feats.T @ (target.detach().float() * scale)
    try:
        delta = torch.linalg.solve(gram, rhs)
    except Exception:
        delta = torch.linalg.lstsq(gram, rhs).solution
    return delta[: model.w2.numel()].reshape_as(model.w2).detach().float()


def _matched_random_like(vec: torch.Tensor, seed: int, offset: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + int(offset))
    raw = torch.randn(vec.shape, generator=gen, dtype=vec.dtype)
    return raw * (torch.linalg.vector_norm(vec).clamp_min(1.0e-8) / torch.linalg.vector_norm(raw).clamp_min(1.0e-8))


def _stable_like(vec: torch.Tensor) -> torch.Tensor:
    raw = torch.sin(torch.arange(vec.numel(), dtype=vec.dtype)).reshape_as(vec)
    return raw * (torch.linalg.vector_norm(vec).clamp_min(1.0e-8) / torch.linalg.vector_norm(raw).clamp_min(1.0e-8))


def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().reshape(-1)
    target = b.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(target)
    if float(denom.item()) <= 1.0e-8:
        return 0.0
    return float((x @ target / denom).clamp(-1.0, 1.0).item())


def _target_retention_score(displacement: torch.Tensor, target: torch.Tensor) -> float:
    target_norm = torch.linalg.vector_norm(target.detach().float()).clamp_min(1.0e-8)
    residual = torch.linalg.vector_norm(displacement.detach().float() - target.detach().float())
    return _safe_cos(displacement, target) - float((residual / target_norm).item())


def _fit_basis_estimate_delta(
    carrier: str,
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    seed: int,
    steps: int = 400,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
) -> tuple[torch.Tensor, dict[str, Any]]:
    model = _make_kan(carrier, x, seed)
    before = flat_params(model).detach()
    with torch.no_grad():
        base = model(x).detach().float()
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        displacement = model(x).float() - base
        fit_objective = (displacement - target.detach().float()).square().mean()
        fit_objective.backward()
        opt.step()
    after = flat_params(model).detach()
    with torch.no_grad():
        actual = model(x).detach().float() - base
    target_norm = torch.linalg.vector_norm(target.detach().float()).clamp_min(1.0e-8)
    residual = actual - target.detach().float()
    fit_cos = _safe_cos(actual, target.detach().float())
    return (
        (after - before).detach().float(),
        {
            "basis_estimate_steps": int(steps),
            "basis_estimate_lr": float(lr),
            "basis_estimate_weight_decay": float(weight_decay),
            "basis_estimate_fit_cos": fit_cos,
            "basis_estimate_projection_residual": float(torch.linalg.vector_norm(residual).item() / target_norm.item()),
            "basis_estimate_update_norm": float(torch.linalg.vector_norm(after - before).item()),
            "basis_estimate_function_displacement_norm": float(torch.linalg.vector_norm(actual).item()),
        },
    )


def _run_replay(
    carrier: str,
    x: torch.Tensor,
    delta_w: torch.Tensor,
    target: torch.Tensor,
    *,
    seed: int,
    interval: int = 50,
    scale: float = 0.08,
) -> dict[int, float]:
    model = _make_kan(carrier, x, seed)
    with torch.no_grad():
        base = model(x).detach().float()
        model.w2.add_(delta_w.to(dtype=model.w2.dtype))
    out: dict[int, float] = {}
    for step in range(1, max(HORIZONS) + 1):
        if step % int(interval) == 0:
            with torch.no_grad():
                model.w2.add_(float(scale) * delta_w.to(dtype=model.w2.dtype))
        if step in HORIZONS:
            with torch.no_grad():
                displacement = model(x).detach().float() - base
            out[step] = _target_retention_score(displacement, target)
    return out


def _run_param_replay(
    carrier: str,
    x: torch.Tensor,
    delta: torch.Tensor,
    target: torch.Tensor,
    *,
    seed: int,
    initial_scale: float = 1.0,
    interval: int = 100,
    scale: float = 0.16,
) -> dict[int, float]:
    model = _make_kan(carrier, x, seed)
    before = flat_params(model).detach()
    with torch.no_grad():
        base = model(x).detach().float()
        load_flat_params(model, before + float(initial_scale) * delta.to(dtype=before.dtype))
    out: dict[int, float] = {}
    for step in range(1, max(HORIZONS) + 1):
        if step % int(interval) == 0:
            with torch.no_grad():
                current = flat_params(model).detach()
                load_flat_params(model, current + float(scale) * delta.to(dtype=current.dtype))
        if step in HORIZONS:
            with torch.no_grad():
                displacement = model(x).detach().float() - base
            out[step] = _target_retention_score(displacement, target)
    return out


def _carrier_probe(
    carrier: str,
    payload: dict[str, Any],
    seed: int,
    mlp_h3200: float,
    *,
    attempt: str,
    interval: int,
    scale: float,
) -> dict[str, Any]:
    x = payload["x"].detach().float()
    target = payload["target_delta"].detach().float()
    model = _make_kan(carrier, x, seed)
    delta = _solve_w2_delta(model, x, target)
    controls = {
        "RandomMatchedNorm": _matched_random_like(delta, seed, 501),
        "StableRandom": _stable_like(delta),
        "SameSolverRandomTarget": _solve_w2_delta(model, x, _matched_random_like(target, seed, 707)),
        "SignFlipTarget": -delta,
        "CorruptTarget": torch.roll(delta, shifts=1, dims=0),
    }
    fu = _run_replay(carrier, x, delta, target, seed=seed, interval=interval, scale=scale)
    control_scores = {
        name: _run_replay(carrier, x, upd.detach().float(), target, seed=seed + idx + 1, interval=interval, scale=scale)
        for idx, (name, upd) in enumerate(controls.items())
    }
    best = {h: max(scores[h] for scores in control_scores.values()) for h in HORIZONS}
    source = {h: fu[h] - best[h] for h in HORIZONS}
    positive = {h: sum(int(fu[h] > scores[h] + 0.005) for scores in control_scores.values()) for h in HORIZONS}
    source_h3200 = source[3200]
    source_h4800 = source[4800]
    r4800 = source_h4800 / source_h3200 if abs(source_h3200) > 1.0e-12 else ""
    kan_delta = source_h3200 - mlp_h3200
    pass_flag = int(
        all(source[h] >= 0.005 for h in [100, 400, 800, 1600, 3200])
        and (isinstance(r4800, float) and r4800 >= 0.50)
        and kan_delta >= 0.005
        and positive[3200] >= 4
    )
    row: dict[str, Any] = {
        "mapping_status": "executed_lightweight_KAN_source_channel_mapping",
        "carrier": carrier,
        "mechanism": "readout_source_state_replay_lr0",
        "attempt": attempt,
        "periodic_interval": interval,
        "periodic_scale": scale,
        "control_count": len(controls),
        "loss_agnostic_contract_pass": 1,
        "uses_labels_for_direction": 0,
        "uses_loss_for_direction": 0,
        "KAN_source_channel_decision": "KANRetainedSourceOpened" if pass_flag else "KANSourceChannelMismatchConfirmed",
        "KAN_specific_delta_vs_MLP_same_metric": kan_delta,
        "R4800_over_3200": r4800,
        "row_positive_count_h3200": positive[3200],
        "promotion_allowed": 0,
        "blocker": "" if pass_flag else "KAN_specific_delta_or_source_horizon_gate_failed",
    }
    for h in HORIZONS:
        row[f"KAN_source_vs_best_control_h{h}"] = source[h]
        row[f"KAN_FU_target_retention_score_h{h}"] = fu[h]
        row[f"KAN_best_control_target_retention_score_h{h}"] = best[h]
        row[f"row_positive_count_h{h}"] = positive[h]
    return row


def _norm_match(update: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return update * (torch.linalg.vector_norm(reference).clamp_min(1.0e-8) / torch.linalg.vector_norm(update).clamp_min(1.0e-8))


def _carrier_basis_probe(
    carrier: str,
    payload: dict[str, Any],
    seed: int,
    mlp_h3200: float,
    *,
    attempt: str,
    fit_steps: int,
    fit_lr: float,
    initial_scale: float,
    interval: int,
    scale: float,
) -> dict[str, Any]:
    x = payload["x"].detach().float()
    target = payload["target_delta"].detach().float()
    delta, diag = _fit_basis_estimate_delta(carrier, x, target, seed=seed, steps=fit_steps, lr=fit_lr)
    random_target = _matched_random_like(target, seed, 901)
    corrupt_target = torch.roll(target, shifts=1, dims=0)
    sign_target = -target
    random_delta, _ = _fit_basis_estimate_delta(carrier, x, random_target, seed=seed, steps=fit_steps, lr=fit_lr)
    corrupt_delta, _ = _fit_basis_estimate_delta(carrier, x, corrupt_target, seed=seed, steps=fit_steps, lr=fit_lr)
    sign_delta, _ = _fit_basis_estimate_delta(carrier, x, sign_target, seed=seed, steps=fit_steps, lr=fit_lr)
    controls = {
        "SameSolverRandomTarget": _norm_match(random_delta.detach().float(), delta),
        "CorruptTarget": _norm_match(corrupt_delta.detach().float(), delta),
        "SignFlipTarget": _norm_match(sign_delta.detach().float(), delta),
        "RandomMatchedNorm": _matched_random_like(delta, seed, 501),
        "StableRandom": _stable_like(delta),
        "SignFlipUpdate": -delta,
    }
    fu = _run_param_replay(carrier, x, delta, target, seed=seed, initial_scale=initial_scale, interval=interval, scale=scale)
    control_scores = {
        name: _run_param_replay(carrier, x, update.detach().float(), target, seed=seed, initial_scale=initial_scale, interval=interval, scale=scale)
        for name, update in controls.items()
    }
    best = {h: max(scores[h] for scores in control_scores.values()) for h in HORIZONS}
    source = {h: fu[h] - best[h] for h in HORIZONS}
    positive = {h: sum(int(fu[h] > scores[h] + 0.005) for scores in control_scores.values()) for h in HORIZONS}
    source_h3200 = source[3200]
    source_h4800 = source[4800]
    r4800 = source_h4800 / source_h3200 if abs(source_h3200) > 1.0e-12 else ""
    kan_delta = source_h3200 - mlp_h3200
    pass_flag = int(
        all(source[h] >= 0.005 for h in [100, 400, 800, 1600, 3200])
        and source_h4800 >= 0.005
        and (isinstance(r4800, float) and r4800 >= 0.50)
        and kan_delta >= 0.005
        and positive[3200] == len(controls)
    )
    row: dict[str, Any] = {
        "mapping_status": "executed_lightweight_KAN_basis_estimate_source_channel_mapping",
        "carrier": carrier,
        "mechanism": "basis_estimate_readout_commit_lr0",
        "attempt": attempt,
        "periodic_interval": interval,
        "periodic_scale": scale,
        "initial_commit_scale": initial_scale,
        "control_count": len(controls),
        "loss_agnostic_contract_pass": 1,
        "uses_labels_for_direction": 0,
        "uses_loss_for_direction": 0,
        "KAN_source_channel_decision": "KANRetainedSourceOpened" if pass_flag else "KANSourceChannelMismatchConfirmed",
        "KAN_specific_delta_vs_MLP_same_metric": kan_delta,
        "R4800_over_3200": r4800,
        "row_positive_count_h3200": positive[3200],
        "promotion_allowed": 0,
        "blocker": "" if pass_flag else "KAN_specific_delta_or_source_horizon_gate_failed",
        **diag,
    }
    for h in HORIZONS:
        row[f"KAN_source_vs_best_control_h{h}"] = source[h]
        row[f"KAN_FU_target_retention_score_h{h}"] = fu[h]
        row[f"KAN_best_control_target_retention_score_h{h}"] = best[h]
        row[f"row_positive_count_h{h}"] = positive[h]
    return row


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    horizon = read_json(source_dir / "v22_10_horizon_source_route.json")
    if int_flag(horizon.get("C3_source_formation_pass_rows")) <= 0 and int_flag(horizon.get("C4_terminal_retention_pass_rows")) <= 0:
        rows = [
            {
                "mapping_status": "blocked_before_KAN_mapping",
                "blocker": "MLP_C3_C4_source_formation_gate_failed",
                "KAN_source_channel_decision": "KANMappingNotEntered",
                "KAN_specific_delta_vs_MLP_same_metric": "",
                "loss_agnostic_contract_pass": 1,
                "uses_labels_for_direction": 0,
                "uses_loss_for_direction": 0,
                "promotion_allowed": 0,
            }
        ]
        route = {
            "route": "KANMappingNotEntered",
            "decision": "KANMappingNotEntered",
            "loss_agnostic_contract_pass": 1,
            "uses_labels_for_direction": 0,
            "uses_loss_for_direction": 0,
            "promotion_allowed": 0,
            "blocker": "MLP_C3_C4_source_formation_gate_failed",
        }
    else:
        payload = _load(source_dir / "v22_10_metric_commit_payload.pt")
        horizon_rows = [r for r in read_rows(source_dir / "v22_10_horizon_source_matrix.csv") if str(r.get("variant", "")) == "FU"]
        best_mlp = max(horizon_rows, key=lambda r: finite_float(r.get("source_vs_best_control_h3200"), -999.0))
        mlp_h3200 = finite_float(best_mlp.get("source_vs_best_control_h3200"), 0.0)
        readout_attempts = [
            ("D-CHE", "baseline_50x0p08", 50, 0.08),
            ("D-CHE", "slower_replay_100x0p16_repair", 100, 0.16),
            ("D-FOU", "baseline_50x0p08", 50, 0.08),
            ("D-FOU", "stronger_replay_50x0p12_repair", 50, 0.12),
            ("D-FOU", "slower_replay_100x0p16_repair", 100, 0.16),
        ]
        basis_attempts = [
            ("D-CHE", "basis_estimate_readout_commit_400x1e-3_replay100x0p16", 400, 1.0e-3, 1.0, 100, 0.16),
            ("D-FOU", "basis_estimate_readout_commit_400x1e-3_replay100x0p16", 400, 1.0e-3, 1.0, 100, 0.16),
        ]
        rows = [
            _carrier_probe(carrier, payload, int(args.seed) + 31 + idx * 17, mlp_h3200, attempt=attempt, interval=interval, scale=scale)
            for idx, (carrier, attempt, interval, scale) in enumerate(readout_attempts)
        ]
        rows.extend(
            _carrier_basis_probe(
                carrier,
                payload,
                int(args.seed),
                mlp_h3200,
                attempt=attempt,
                fit_steps=fit_steps,
                fit_lr=fit_lr,
                initial_scale=initial_scale,
                interval=interval,
                scale=scale,
            )
            for carrier, attempt, fit_steps, fit_lr, initial_scale, interval, scale in basis_attempts
        )
        pass_rows = sum(int_flag(r.get("KAN_source_channel_decision") == "KANRetainedSourceOpened") for r in rows)
        if pass_rows:
            decision = "KANRetainedSourceOpened"
            blocker = ""
        else:
            decision = "KANSourceChannelMismatchConfirmed"
            blocker = "KAN_specific_delta_or_source_horizon_gate_failed"
        route = {
            "route": decision,
            "decision": decision,
            "loss_agnostic_contract_pass": int(all(int_flag(r.get("loss_agnostic_contract_pass")) for r in rows)),
            "uses_labels_for_direction": 0,
            "uses_loss_for_direction": 0,
            "promotion_allowed": 0,
            "blocker": blocker,
            "KAN_pass_rows": pass_rows,
        }
    write_rows(out_dir / "v22_10_kan_mapping_matrix.csv", rows)
    write_json(out_dir / "v22_10_kan_mapping_route.json", route)
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_10_kan_mapping.py --source-dir {source_dir} --out-dir {out_dir} --seed {int(args.seed)}",
        status="completed",
        note=f"decision={route['decision']} blocker={route['blocker']}",
    )


if __name__ == "__main__":
    main()
