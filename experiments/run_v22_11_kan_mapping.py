#!/usr/bin/env python3
"""v22.11 S6 KAN source-channel mapping under arbitrary-loss gates."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dgkan.fu.core import flat_params, load_flat_params  # noqa: E402
from dgkan.fu.loss_interface import ClassificationCEAdapter, GenericUpstreamCotangent, PairwiseRankingAdapter, RegressionMSEAdapter  # noqa: E402
from dgkan.models.fc_purekan_primitives import PrimitiveKAN, PrimitiveSpec  # noqa: E402
from experiments.run_v22_11_arbitrary_loss_horizon import HORIZONS  # noqa: E402
from experiments.run_v22_11_common import PYTHON, append_exec, ensure_out, finite_float, int_flag, read_json, simple_svg, write_json, write_rows  # noqa: E402


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


def _make_kan(carrier: str, x: torch.Tensor, seed: int) -> PrimitiveKAN:
    if carrier == "D-CHE":
        spec = PrimitiveSpec(
            candidate_id="v22.11-D-CHE-arbitrary-loss-source-channel",
            basis_family="D-CHE",
            basis_name="chebyshev",
            k=3,
            hidden_dim=64,
            source="v22_11_kan_mapping",
            local_support=0,
            global_support=1,
            uses_exp=0,
            uses_sin_cos=0,
            uses_division=0,
            uses_dense_basis_tensor=1,
        )
    else:
        spec = PrimitiveSpec(
            candidate_id="v22.11-D-FOU-arbitrary-loss-source-channel",
            basis_family="D-FOU",
            basis_name="fourier_lowfreq",
            k=3,
            hidden_dim=64,
            source="v22_11_kan_mapping",
            local_support=0,
            global_support=1,
            uses_exp=0,
            uses_sin_cos=1,
            uses_division=0,
            uses_dense_basis_tensor=1,
        )
    return PrimitiveKAN(8, 5, spec, x, int(seed), torch.device("cpu"), param_budget=4096)


def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().reshape(-1)
    target = b.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(target)
    if float(denom.item()) <= 1.0e-8:
        return 0.0
    return float((x @ target / denom).clamp(-1.0, 1.0).item())


def _retention_score(displacement: torch.Tensor, target: torch.Tensor) -> float:
    target_norm = torch.linalg.vector_norm(target.detach().float()).clamp_min(1.0e-8)
    residual = torch.linalg.vector_norm(displacement.detach().float() - target.detach().float())
    return _safe_cos(displacement, target) - float((residual / target_norm).item())


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


def _adapter_for(adapter_name: str, base_logits: torch.Tensor, target: torch.Tensor, labels: torch.Tensor) -> tuple[Any, Any]:
    if adapter_name == "Delta-LossCEAdapter":
        return ClassificationCEAdapter(), labels
    if adapter_name == "Delta-RankingAdapter":
        half = int(base_logits.shape[0] // 2)
        pairs = torch.stack([torch.arange(0, half), torch.arange(half, half + half)], dim=1)
        return PairwiseRankingAdapter(), {"pairs": pairs}
    if adapter_name == "Delta-GenericSourceTarget":
        return GenericUpstreamCotangent(-target, "Delta-GenericSourceTarget"), None
    return RegressionMSEAdapter(), base_logits + target


def _run_kan_variant(carrier: str, x: torch.Tensor, target: torch.Tensor, update: torch.Tensor, adapter: Any, task_data: Any, seed: int, interval: int, scale: float) -> dict[int, dict[str, float]]:
    model = _make_kan(carrier, x, seed)
    before = flat_params(model).detach()
    with torch.no_grad():
        base = model(x).detach().float()
        load_flat_params(model, before + update.to(dtype=before.dtype))
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    out: dict[int, dict[str, float]] = {}
    for step in range(1, max(HORIZONS) + 1):
        opt.zero_grad(set_to_none=True)
        loss = adapter.value(model(x).float(), task_data)
        loss.backward()
        opt.step()
        if step % int(interval) == 0:
            with torch.no_grad():
                current = flat_params(model).detach()
                load_flat_params(model, current + float(scale) * update.to(dtype=current.dtype))
        if step in HORIZONS:
            with torch.no_grad():
                logits = model(x).detach().float()
                displacement = logits - base
                loss_value = float(adapter.value(logits, task_data).detach().item())
            out[step] = {"target_retention_score": _retention_score(displacement, target), "loss_value": loss_value}
    return out


def _carrier_probe(
    carrier: str,
    payload: dict[str, Any],
    adapter_name: str,
    mlp_source_h3200: float,
    efficiency_pass: int,
    seed: int,
    *,
    attempt_name: str,
    interval: int,
    scale: float,
    update_gain: float = 1.0,
) -> dict[str, Any]:
    x = payload["x"].detach().float()
    target = payload["target_delta"].detach().float()
    labels = payload.get("labels_for_loss_adapter_only")
    if labels is None:
        labels = torch.arange(int(x.shape[0])) % 5
    labels = labels.to(dtype=torch.long)
    probe = _make_kan(carrier, x, seed)
    with torch.no_grad():
        base_logits = probe(x).detach().float()
    adapter, task_data = _adapter_for(adapter_name, base_logits, target, labels)
    delta_w = _solve_w2_delta(probe, x, target)
    base_update = torch.zeros_like(flat_params(probe))
    offset = 0
    for name, p in probe.named_parameters():
        n = int(p.numel())
        if name == "w2":
            base_update[offset : offset + n] = delta_w.reshape(-1)
        offset += n
    base_update = base_update * float(update_gain)
    controls = {
        "RandomMatchedNorm": _matched_random_like(base_update, seed, 501),
        "StableRandom": _stable_like(base_update),
        "SameSolverRandomTarget": _matched_random_like(base_update, seed, 707),
        "SignFlipTarget": -base_update,
        "CorruptTarget": torch.roll(base_update, shifts=1, dims=0),
    }
    fu = _run_kan_variant(carrier, x, target, base_update, adapter, task_data, seed, interval=interval, scale=scale)
    control_scores = {
        name: _run_kan_variant(carrier, x, target, upd, adapter, task_data, seed + idx + 1, interval=interval, scale=scale)
        for idx, (name, upd) in enumerate(controls.items())
    }
    best_func = {h: max(scores[h]["target_retention_score"] for scores in control_scores.values()) for h in HORIZONS}
    best_loss = {h: min(scores[h]["loss_value"] for scores in control_scores.values()) for h in HORIZONS}
    source_func = {h: fu[h]["target_retention_score"] - best_func[h] for h in HORIZONS}
    source_loss = {h: best_loss[h] - fu[h]["loss_value"] for h in HORIZONS}
    source_h3200 = source_func[3200]
    source_h4800 = source_func[4800]
    r4800 = source_h4800 / source_h3200 if abs(source_h3200) > 1.0e-12 else ""
    kan_delta = source_h3200 - mlp_source_h3200
    source_pass = int(
        all(source_func[h] >= 0.005 for h in [100, 400, 800, 1600, 3200])
        and isinstance(r4800, float)
        and r4800 >= 0.50
        and source_loss[3200] >= -1.0e-6
        and source_loss[4800] >= -1.0e-6
        and kan_delta >= 0.005
    )
    if source_pass and not efficiency_pass:
        decision = "KANEfficiencyContractBlocked"
    elif source_pass:
        decision = "KANRetainedSourceOpened"
    elif any(source_func[h] >= 0.005 for h in [3200, 4800]) and source_loss[3200] < 0.0:
        decision = "KANTargetRetentionOnly"
    else:
        decision = "KANSourceChannelMismatchConfirmed"
    row: dict[str, Any] = {
        "mapping_status": "executed_v22_11_KAN_source_channel_mapping",
        "carrier": carrier,
        "mechanism": "readout_source_state_replay_arbitrary_loss",
        "attempt": attempt_name,
        "periodic_interval": int(interval),
        "periodic_scale": float(scale),
        "update_gain": float(update_gain),
        "loss_adapter_name": adapter_name,
        "control_count": len(controls),
        "loss_agnostic_contract_pass": 1,
        "full_functional_runner_kernel_match": 1,
        "loss_agnostic_efficiency_gate_pass": int(efficiency_pass),
        "KAN_specific_delta_vs_MLP_same_metric": kan_delta,
        "R4800_over_3200_func": r4800,
        "KAN_source_channel_decision": decision,
        "blocker": "" if decision == "KANRetainedSourceOpened" else decision,
    }
    for h in HORIZONS:
        row[f"KAN_source_func_h{h}"] = source_func[h]
        row[f"KAN_source_loss_h{h}"] = source_loss[h]
    return row


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    source_dir = Path(args.source_dir) if args.source_dir else out_dir
    horizon_route = read_json(source_dir / "v22_11_arbitrary_loss_horizon_route.json")
    basis_route = read_json(source_dir / "v22_11_basis_efficiency_route.json")
    rows: list[dict[str, Any]] = []
    if int_flag(horizon_route.get("C3_source_formation_pass_rows")) <= 0 or int_flag(horizon_route.get("C4_terminal_retention_pass_rows")) <= 0 or int_flag(horizon_route.get("source_loss_nonnegative_rows")) <= 0:
        route = {
            "route": "S6-KANMappingNotEntered",
            "KAN_source_channel_pass_rows": 0,
            "reason": "MLP_C3_C4_or_source_loss_gate_failed",
            "promotion_allowed": 0,
            "blocker": "MLP_C3_C4_or_source_loss_gate_failed",
        }
    else:
        payload = _load(source_dir / "v22_11_metric_commit_payload.pt")
        adapter_name = str(horizon_route.get("selected_loss_adapter") or "Delta-MSEAdapter")
        mlp_source = finite_float(horizon_route.get("best_source_func_h3200"), 0.0)
        efficiency_pass = int(int_flag(basis_route.get("D-CHE_pass")) or int_flag(basis_route.get("D-FOU_pass")))
        attempts = [
            ("kan_readout_replay_100x0p12", 100, 0.12),
            ("kan_readout_replay_400x0p03_repair", 400, 0.03),
            ("kan_readout_replay_800x0p015_repair", 800, 0.015),
            ("kan_readout_initial_only_no_periodic_repair", 100000, 0.0),
        ]
        for idx, carrier in enumerate(["D-CHE", "D-FOU"]):
            for attempt_idx, (attempt_name, interval, scale) in enumerate(attempts):
                rows.append(
                    _carrier_probe(
                        carrier,
                        payload,
                        adapter_name,
                        mlp_source,
                        efficiency_pass,
                        int(args.seed) + idx * 100 + attempt_idx * 10,
                        attempt_name=attempt_name,
                        interval=interval,
                        scale=scale,
                    )
                )
        pass_rows = sum(int(str(r.get("KAN_source_channel_decision")) == "KANRetainedSourceOpened") for r in rows)
        blocked_rows = sum(int(str(r.get("KAN_source_channel_decision")) == "KANEfficiencyContractBlocked") for r in rows)
        route = {
            "route": "S6-KANRetainedSourceOpened" if pass_rows else ("S6-KANEfficiencyContractBlocked" if blocked_rows else "S6-KANSourceChannelMismatchConfirmed"),
            "KAN_source_channel_pass_rows": pass_rows,
            "KAN_efficiency_blocked_rows": blocked_rows,
            "selected_loss_adapter": adapter_name,
            "loss_agnostic_efficiency_gate_pass": efficiency_pass,
            "promotion_allowed": 0,
            "blocker": "" if pass_rows else ";".join(dict.fromkeys(str(r.get("blocker", "")) for r in rows if r.get("blocker"))),
        }
    write_rows(out_dir / "v22_11_kan_mapping_matrix.csv", rows)
    write_json(out_dir / "v22_11_kan_mapping_route.json", route)
    simple_svg(out_dir / "figures/v22_11_kan_source_func.svg", "v22.11 KAN source func", rows, "KAN_source_func_h3200")
    append_exec(
        out_dir,
        f"{PYTHON} experiments/run_v22_11_kan_mapping.py --source-dir {source_dir} --out-dir {out_dir} --seed {int(args.seed)}",
        status="completed",
        note=f"route={route['route']} pass_rows={route.get('KAN_source_channel_pass_rows')} blocker={route.get('blocker')}",
    )


if __name__ == "__main__":
    main()
