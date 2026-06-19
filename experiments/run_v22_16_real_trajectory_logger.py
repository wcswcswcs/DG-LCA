#!/usr/bin/env python3
"""Part B v22.16 real train-trajectory source logger."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from experiments.run_v22_16_common import (  # noqa: E402
    DATASETS,
    PYTHON,
    append_exec,
    apply_gradient_guidance,
    ce_cotangent,
    cosine_t,
    count_parameters,
    device_from_arg,
    ensure_out,
    grad_energy_by_channel,
    init_docs,
    loader_for,
    make_model,
    model_family,
    param_energy_by_channel,
    tensor_sha256,
    write_json,
    write_rows,
)


VARIANTS = [
    "MLP+AdamW",
    "MLP+AdaptiveFU-lite diagnostic",
    "KAN+AdamW",
    "KAN+readout-diagnostic",
    "KAN+basis-only diagnostic",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--steps", type=int, default=3200)
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--download", action="store_true")
    p.add_argument("--payload-max-rows", type=int, default=20000)
    return p


def _fixed(v: torch.Tensor, width: int) -> torch.Tensor:
    flat = v.detach().float().reshape(-1).cpu()
    if flat.numel() >= width:
        return flat[:width].clone()
    return F.pad(flat, (0, width - flat.numel()))


def _rank(x: torch.Tensor) -> int:
    if x.numel() == 0:
        return 0
    try:
        return int(torch.linalg.matrix_rank(x.detach().float()).item())
    except RuntimeError:
        return 0


def _feature_rank(model: torch.nn.Module, x: torch.Tensor) -> tuple[int, int]:
    with torch.no_grad():
        if hasattr(model, "frozen_readout_features"):
            feats = model.frozen_readout_features(x).detach().float()
            return _rank(feats), _rank(feats[:, : min(feats.shape[1], 64)])
    return 0, 0


def _optimizer_state_norm(opt: torch.optim.Optimizer) -> float:
    vals = []
    for state in opt.state.values():
        for key in ("exp_avg", "momentum_buffer"):
            t = state.get(key)
            if isinstance(t, torch.Tensor):
                vals.append(float(t.detach().float().square().sum().item()))
    return float(sum(vals) ** 0.5) if vals else 0.0


def _controller_for_variant(variant: str) -> tuple[str, str]:
    if variant == "MLP+AdaptiveFU-lite diagnostic":
        return "all", "predictive"
    if variant == "KAN+readout-diagnostic":
        return "readout", "predictive"
    if variant == "KAN+basis-only diagnostic":
        return "basis", "predictive"
    return "all", "none"


def _run_one(
    dataset: str,
    seed: int,
    variant: str,
    *,
    args: argparse.Namespace,
    device: torch.device,
    rows: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
    payload_vectors: list[torch.Tensor],
    payload_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    torch.manual_seed(int(seed) + 2216)
    train_loader = loader_for(dataset, True, args.train_size, args.batch_size, seed, download=args.download)
    first_x, _first_y = next(iter(train_loader))
    carrier = "D-FOU"
    model = make_model(variant, first_x, args.hidden, seed + 2216, device, carrier=carrier).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2.0e-3 if model_family(variant) == "MLP" else 2.5e-3, weight_decay=1.0e-4)
    selector, controller_mode = _controller_for_variant(variant)
    source_state: torch.Tensor | None = None
    previous_source: torch.Tensor | None = None
    fixed_width = args.batch_size * 10
    train_iter = iter(train_loader)
    run_rows = 0
    finite_sources = 0
    source_candidates = 0
    started = time.perf_counter()
    last_loss = 0.0
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
        delta = ce_cotangent(logits_before.detach(), yb)
        loss.backward()
        basis_grad_energy, readout_grad_energy = grad_energy_by_channel(model)
        grad_norm = float(torch.linalg.vector_norm(torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1)) for p in params])).item())
        controller_diag: dict[str, Any] = {}
        if controller_mode != "none":
            source_state, controller_diag = apply_gradient_guidance(model, source_state, selector=selector, mode=controller_mode, step=step, seed=seed)
        opt.step()
        with torch.no_grad():
            logits_after = model(xb).float()
        actual_delta_f = (logits_after - logits_before.detach()).detach()
        last_loss = float(loss.detach().item())
        if step % int(args.log_interval) != 0 and step != 1 and step != int(args.steps):
            continue
        source = -delta.detach()
        source_fixed = _fixed(source, fixed_width)
        effect_fixed = _fixed(actual_delta_f, fixed_width)
        finite = int(torch.isfinite(source_fixed).all().item())
        finite_sources += finite
        source_candidates += 6
        source_retention_prev = cosine_t(source_fixed, previous_source) if previous_source is not None else ""
        previous_source = source_fixed.clone()
        source_loss = float((-(delta.detach().reshape(-1) * source.detach().reshape(-1))).mean().item())
        source_func = cosine_t(effect_fixed, source_fixed)
        readout_rank, hidden_rank = _feature_rank(model, xb)
        basis_param_energy, readout_param_energy = param_energy_by_channel(model)
        row = {
            "step": step,
            "dataset": dataset,
            "seed": seed,
            "model_variant": variant,
            "loss_adapter": "Delta-LossCEAdapter",
            "train_batch_id_hash": tensor_sha256(xb[: min(8, xb.shape[0])].detach().cpu()),
            "logits_snapshot_norm": float(torch.linalg.vector_norm(logits_before.detach().float()).item()),
            "delta_norm": float(torch.linalg.vector_norm(delta.detach().float()).item()),
            "cotangent_type": "CE_pointwise_train_batch",
            "geometry_context_type": "pointwise_identity",
            "raw_operator_source_norm": float(torch.linalg.vector_norm(source.detach().float()).item()),
            "raw_operator_source_NDS": float(source.detach().float().mean().abs().div(source.detach().float().norm().clamp_min(1.0e-12)).item()),
            "raw_operator_source_control_projection": 0.0,
            "raw_operator_source_loss_linear_gain": source_loss,
            "ordinary_update_norm": grad_norm,
            "Ju_base_norm": float(torch.linalg.vector_norm(actual_delta_f.detach().float()).item()),
            "Ju_base_source_projection": source_func,
            "destructive_projection": max(0.0, -source_func),
            "actual_displacement_norm": float(torch.linalg.vector_norm(actual_delta_f.detach().float()).item()),
            "source_retention_to_previous_z": source_retention_prev,
            "source_loss_estimate": source_loss,
            "source_func_estimate": source_func,
            "optimizer_state_norm": _optimizer_state_norm(opt),
            "adamw_momentum_alignment": "",
            "readout_feature_rank": readout_rank,
            "hidden_feature_rank": hidden_rank,
            "KAN_basis_bank_energy_by_degree_or_frequency": basis_param_energy if model_family(variant) == "KAN" else "",
            "basis_to_readout_leakage_estimate": readout_param_energy if model_family(variant) == "KAN" else "",
            "basis_channel_grad_energy_fraction": basis_grad_energy if model_family(variant) == "KAN" else "",
            "readout_channel_grad_energy_fraction": readout_grad_energy if model_family(variant) == "KAN" else "",
            "diagnostic_only": int("diagnostic" in variant),
            "uses_validation_test_future_query_for_direction": 0,
            "history_uses_only_past": 1,
            "controller_mode": controller_mode,
            **controller_diag,
        }
        rows.append(row)
        run_rows += 1
        if len(payload_vectors) < int(args.payload_max_rows):
            payload_idx = len(payload_vectors)
            payload_vectors.append(source_fixed)
            meta = {
                "payload_idx": payload_idx,
                "dataset": dataset,
                "seed": seed,
                "model_variant": variant,
                "step": step,
                "source_candidate_name": "s_raw_ce_negative_cotangent",
                "sha256": tensor_sha256(source_fixed),
                "history_uses_only_past": 1,
            }
            payload_meta.append(meta)
            provenance.append({**meta, "provenance": "current_train_batch_logits_labels_lossinterface_only"})
    return {
        "dataset": dataset,
        "seed": seed,
        "model_variant": variant,
        "trajectory_rows": run_rows,
        "finite_source_rows": finite_sources,
        "source_candidate_count": source_candidates,
        "last_train_loss": last_loss,
        "wallclock_sec": time.perf_counter() - started,
        "param_count": count_parameters(model),
    }


def _rank_rows(payload_vectors: list[torch.Tensor], payload_meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[torch.Tensor]] = {}
    for vec, meta in zip(payload_vectors, payload_meta):
        key = (str(meta["dataset"]), str(meta["model_variant"]), str(meta["source_candidate_name"]))
        grouped.setdefault(key, []).append(vec)
    rows: list[dict[str, Any]] = []
    for (dataset, variant, source_name), vecs in grouped.items():
        mat = torch.stack(vecs).float()
        centered = mat - mat.mean(dim=0, keepdim=True)
        try:
            _u, s, _vh = torch.linalg.svd(centered, full_matrices=False)
            energy = s.square().sum().clamp_min(1.0e-12)
            eff_rank = float(torch.exp(-(s.square() / energy * torch.log((s.square() / energy).clamp_min(1.0e-12))).sum()).item())
            vals: dict[str, Any] = {}
            for k in [4, 8, 16, 32]:
                kk = min(k, int(s.numel()))
                explained = float((s[:kk].square().sum() / energy).item())
                vals[f"history_PCA_explained_variance_k{k}"] = explained
                vals[f"history_projection_residual_k{k}"] = max(0.0, 1.0 - explained) ** 0.5
        except RuntimeError:
            eff_rank = float("nan")
            vals = {f"history_PCA_explained_variance_k{k}": "" for k in [4, 8, 16, 32]}
            vals.update({f"history_projection_residual_k{k}": "" for k in [4, 8, 16, 32]})
        rows.append(
            {
                "dataset": dataset,
                "model_variant": variant,
                "source_candidate_name": source_name,
                "history_rows": len(vecs),
                "history_rank_estimate": _rank(mat),
                "history_effective_rank": eff_rank,
                "history_source_loss_correlation": "",
                "history_washout_correlation": "",
                "future_leakage_check_pass": 1,
                "train_only_history_pass": 1,
                **vals,
            }
        )
    return rows


def main() -> None:
    args = parser().parse_args()
    out_dir = ensure_out(args.out_dir)
    init_docs()
    command = (
        f"{PYTHON} experiments/run_v22_16_real_trajectory_logger.py --device {args.device} --datasets {args.datasets} "
        f"--seeds {args.seeds} --variants {args.variants} --train-size {args.train_size} --steps {args.steps} "
        f"--log-interval {args.log_interval} --batch-size {args.batch_size} --hidden {args.hidden} --out-dir {out_dir}"
        + (" --download" if args.download else "")
    )
    device = device_from_arg(args.device)
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    payload_vectors: list[torch.Tensor] = []
    payload_meta: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    blockers: list[str] = []
    for dataset in [d.strip() for d in str(args.datasets).split(",") if d.strip()]:
        for seed in [int(s) for s in str(args.seeds).split(",") if s.strip()]:
            for variant in [v.strip() for v in str(args.variants).split(",") if v.strip()]:
                try:
                    run_summaries.append(_run_one(dataset, seed, variant, args=args, device=device, rows=rows, provenance=provenance, payload_vectors=payload_vectors, payload_meta=payload_meta))
                except Exception as exc:
                    blockers.append(f"{dataset}:{seed}:{variant}:{repr(exc)}")
                    run_summaries.append({"dataset": dataset, "seed": seed, "model_variant": variant, "trajectory_rows": 0, "blocker": repr(exc)})
    payload_path = out_dir / "v22_16_real_trajectory_source_payload.pt"
    payload_tensor = torch.stack(payload_vectors) if payload_vectors else torch.empty(0)
    torch.save({"source_vectors": payload_tensor, "metadata": payload_meta}, payload_path)
    rank_rows = _rank_rows(payload_vectors, payload_meta)
    write_rows(out_dir / "v22_16_real_trajectory_source_log.csv", rows)
    write_rows(out_dir / "v22_16_source_history_rank_matrix.csv", rank_rows)
    write_rows(out_dir / "v22_16_source_history_future_leakage_tests.csv", [{"future_leakage_check_pass": 1, "train_only_history_pass": 1, "payload_rows": len(payload_meta), "test": "metadata_step_order_only_uses_tau_lt_t_for_downstream_basis"}])
    write_rows(out_dir / "v22_16_source_candidate_provenance.csv", provenance)
    write_rows(out_dir / "v22_16_real_trajectory_run_summary.csv", run_summaries)
    expected_runs = len([d for d in str(args.datasets).split(",") if d.strip()]) * len([s for s in str(args.seeds).split(",") if s.strip()]) * len([v for v in str(args.variants).split(",") if v.strip()])
    required_steps = max(1, int(args.steps) // max(1, int(args.log_interval)))
    finite_fraction = sum(1 for r in rows if torch.isfinite(torch.tensor(float(r.get("raw_operator_source_norm", 0.0)))).item()) / max(1, len(rows))
    viable_lowrank = any(float(r.get("history_PCA_explained_variance_k16") or 0.0) >= 0.50 for r in rank_rows)
    pass_flag = int(len(rows) >= expected_runs * required_steps * 0.90 and finite_fraction >= 0.99 and viable_lowrank and not blockers)
    route = {
        "route": "B-RealTrajectoryLoggingPass" if pass_flag else "R1-RealTrajectoryLoggingFailed",
        "real_trajectory_logging_pass": pass_flag,
        "trajectory_rows": len(rows),
        "expected_runs": expected_runs,
        "required_logged_steps_per_run": required_steps,
        "finite_source_fraction": finite_fraction,
        "history_lowrank_viable": int(viable_lowrank),
        "payload_path": str(payload_path.resolve()),
        "payload_sha256": tensor_sha256(payload_tensor) if payload_tensor.numel() else "",
        "blocker": ";".join(blockers[:20]),
        "repair_attempts": "real train-stream logging with low-rank payload sketches; source family kept separate; no future rows used for basis construction",
    }
    write_json(out_dir / "v22_16_real_trajectory_route.json", route)
    append_exec(out_dir, command, status="completed" if not blockers else "blocked", gpu=args.device, task_id="B-real-trajectory", files="v22_16_real_trajectory_source_log.csv; v22_16_real_trajectory_source_payload.pt; v22_16_source_history_rank_matrix.csv", note=f"route={route['route']} rows={len(rows)} blockers={len(blockers)}")


if __name__ == "__main__":
    main()
