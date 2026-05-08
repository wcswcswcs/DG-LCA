#!/usr/bin/env python3
"""DG-KAN v4.5 runner: Accelerated Functional Optimizer probes."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dgkan_core import (
    RuntimeState,
    _get_sobolev_gram,
    coefficient_named_params,
    ensure_dir,
    evaluate,
    get_device,
    iter_minibatches,
    load_vision_bundle,
    parse_int_list,
    parse_str_list,
    rbf_basis_occupancy_audit,
    read_csv,
    set_seed,
    write_csv,
)
from run_gafu_v3 import add_args as add_v3_args, dataset_name
from run_gafu_v43 import (
    AdamState,
    V43Params,
    V43RunState,
    _adam_updates,
    _apply_updates,
    _feature_audit,
    _flat_updates,
    _mean,
    _method_updates as _v43_method_updates,
    _restore,
    _role_for_name,
    _role_group,
    _safe_cos,
    _snapshot,
)
from run_gafu_v44 import (
    FCAdamState,
    V44Params,
    V44RunState,
    _compute_loss_and_grad,
    _estimate_geometry_any,
    _fcadam_updates,
    _function_displacement,
    _make_model,
    _metric_matrix,
    _r2,
)


DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST"]
P2_METHODS = [
    "PureKAN-AdamW-one-step",
    "D0-allFullSobolev",
    "D6-allTaskAware",
    "F4-FNG-leftFullRight",
    "FCAdam-L2",
    "FCAdam-H1-low",
    "FCAdam-dataSob",
    "AB-RBF-FCAdam-H1-low",
    "FCAdam-dataSob-noGeometryGate",
    "FCAdam-dataSob-phasedGeometry",
]


@dataclass
class V45Params(V44Params):
    train_size: int = 1536
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 128
    p2_steps: int = 20
    fc_lr: float = 8e-4
    fc_data_sob_alpha: float = 0.02
    phase_phi_budget_early: float = 1.15


def _fc_metric_kind(method: str) -> str:
    key = method.lower().replace("-", "_")
    if "datasob" in key:
        return "dataSob"
    if "h1_low" in key:
        return "h1-low"
    if "h1" in key:
        return "h1"
    return "l2"


def _method_updates(
    model: torch.nn.Module,
    dataset: str,
    seed: int,
    method: str,
    params: V45Params,
    run_state: V44RunState,
    xb: torch.Tensor,
    yb: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    key = method.lower().replace("-", "_")
    if key in {"purekan_adamw_one_step", "purekan_adamw"}:
        updates, rms = _adam_updates(model, run_state.adam, lr=params.adam_lr, mutate=True)
        return updates, {"optimizer_family": "adamw", "adam_rms_mean": _mean([float(v.mean().cpu()) for v in rms.values()], float("nan"))}
    if key in {"d0_allfullsobolev", "d6_alltaskaware", "f4_fng_leftfullright"}:
        v43_params = V43Params(
            train_size=params.train_size,
            val_size=params.val_size,
            test_size=params.test_size,
            batch_size=params.batch_size,
            hidden_dim=params.hidden_dim,
            depth=params.depth,
            basis_count=params.basis_count,
        )
        updates, stats = _v43_method_updates(model, dataset, seed, method, v43_params, V43RunState())
        stats["optimizer_family"] = stats.get("optimizer_family", "baseline_functional")
        return updates, stats
    if key in {
        "fcadam_l2",
        "fcadam_h1_low",
        "fcadam_datasob",
        "fcadam_datasob_nogeometrygate",
        "fcadam_datasob_phasedgeometry",
        "ab_rbf_fcadam_h1_low",
    }:
        updates, stats = _fcadam_updates(model, run_state.fc, params, metric_kind=_fc_metric_kind(method))
        stats["optimizer_family"] = "fcadam_v45"
        if "phasedgeometry" in key:
            stats["geometry_policy"] = "phased_early_phi_budget_1.15"
        elif "nogeometrygate" in key:
            stats["geometry_policy"] = "none"
        else:
            stats["geometry_policy"] = "default"
        return updates, stats
    raise ValueError(f"unknown v4.5 method: {method}")


def _iter_steps(n: int, batch_size: int, seed: int, steps: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    epoch = 0
    while len(out) < steps:
        for idx in iter_minibatches(n, batch_size, seed + epoch * 1009):
            out.append(idx)
            if len(out) >= steps:
                break
        epoch += 1
    return out


def _role_norms(updates: Dict[str, torch.Tensor]) -> Dict[str, float]:
    by_role: Dict[str, List[float]] = {}
    for name, upd in updates.items():
        by_role.setdefault(_role_group(_role_for_name(name)), []).append(float(upd.detach().float().norm().cpu()))
    return {role: _mean(vals, 0.0) for role, vals in by_role.items()}


def _coef_norms(model: torch.nn.Module) -> Dict[str, float]:
    by_role: Dict[str, List[float]] = {}
    for name, p in coefficient_named_params(model):
        by_role.setdefault(_role_group(_role_for_name(name)), []).append(float(p.detach().float().norm().cpu()))
    return {role: _mean(vals, 0.0) for role, vals in by_role.items()}


def _metric_conditions(model: torch.nn.Module, params: V45Params, device: torch.device) -> Dict[str, float]:
    conds: Dict[str, List[float]] = {"l2": [], "h1": [], "dataSob": []}
    for layer in model.kan_layers():  # type: ignore[attr-defined]
        for kind in conds:
            try:
                _, stats = _metric_matrix(layer, kind, params, device)
                conds[kind].append(float(stats["metric_condition"]))
            except Exception:
                pass
    return {f"metric_condition_{k}": _mean(v, float("nan")) for k, v in conds.items()}


def _roundtrip_errors(model: torch.nn.Module, params: V45Params, device: torch.device) -> Dict[str, float]:
    recon: List[float] = []
    update_rt: List[float] = []
    grad_chain: List[float] = []
    for name, p in coefficient_named_params(model):
        layer = list(model.kan_layers())[list(n for n, _ in coefficient_named_params(model)).index(name)]  # type: ignore[attr-defined]
        mat, _ = _metric_matrix(layer, "h1-low", params, device)
        chol = torch.linalg.cholesky(mat.double())
        coeff = p.detach().reshape(-1, p.shape[-1]).double()
        u = coeff @ chol
        coeff_back = torch.linalg.solve_triangular(chol.T, u.T, upper=True).T
        recon.append(float(((coeff_back - coeff).norm() / coeff.norm().clamp_min(1e-12)).cpu()))
        du = torch.randn_like(u) * 1e-3
        da = torch.linalg.solve_triangular(chol.T, du.T, upper=True).T
        du_back = da @ chol
        update_rt.append(float(((du_back - du).norm() / du.norm().clamp_min(1e-12)).cpu()))
        if p.grad is not None:
            grad = p.grad.detach().reshape(-1, p.shape[-1]).double()
            gu = torch.linalg.solve_triangular(chol, grad.T, upper=False).T
            grad_back = gu @ chol.T
            grad_chain.append(float(((grad_back - grad).norm() / grad.norm().clamp_min(1e-12)).cpu()))
    return {
        "whiten_reconstruction_error": _mean(recon, float("nan")),
        "u_to_a_roundtrip_error": _mean(update_rt, float("nan")),
        "gradient_transform_error": _mean(grad_chain, float("nan")),
    }


def run_p0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = []
    device = get_device(args.device)
    configs = [(16, 64, 2), (16, 64, 4), (16, 96, 2), (16, 96, 4), (24, 64, 2), (24, 64, 4), (24, 96, 2), (24, 96, 4)]
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        for basis_count, hidden_dim, depth in configs:
            params = V45Params(train_size=512, val_size=128, test_size=128, basis_count=basis_count, hidden_dim=hidden_dim, depth=depth)
            bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=512, val_size=128, test_size=128, seed=0, download=not args.no_download, allow_fake_data=args.allow_fake_data)
            set_seed(0)
            model = _make_model(bundle, params, device, ab=False)
            idx = next(iter(iter_minibatches(len(bundle.x_train), params.batch_size, 0)))
            xb = bundle.x_train[idx].to(device)
            yb = bundle.y_train[idx].to(device)
            named = coefficient_named_params(model)
            snap = _snapshot(named)
            err = ""
            try:
                _compute_loss_and_grad(model, xb, yb)
                updates, stats = _fcadam_updates(model, FCAdamState(), params, metric_kind="dataSob")
                _apply_updates(named, updates)
            except Exception as exc:
                err = repr(exc)
                stats = {}
            _restore(named, snap)
            rollback = max(float((p.detach() - snap[n]).abs().max().cpu()) for n, p in named)
            coeff_total = sum(p.numel() for _, p in named)
            coeff_ids = {id(p) for _, p in named}
            nonkan = sum(p.numel() for p in model.parameters() if p.requires_grad and id(p) not in coeff_ids)
            row: Dict[str, Any] = {
                "stage": "P0",
                "dataset": dataset,
                "basis_count": basis_count,
                "hidden_dim": hidden_dim,
                "depth": depth,
                "learnable_nonKAN_params": nonkan,
                "functional_coverage": 1.0 if coeff_total else 0.0,
                "alpha_trainable": 0,
                "input_coeff_seen": 1.0,
                "block_coeff_seen": 1.0,
                "output_coeff_seen": 1.0,
                "adam_state_shape_match": int(all(name in updates for name, _ in named)) if not err else 0,
                "rollback_error": rollback,
                "no_nan_inf": int(all(torch.isfinite(p).all().item() for _, p in named)),
                "error": err,
            }
            row.update(_metric_conditions(model, params, device))
            row.update(_roundtrip_errors(model, params, device))
            rows.append(row)
            print(f"P0 {dataset} b={basis_count} h={hidden_dim} d={depth} nonKAN={nonkan} rt={row['whiten_reconstruction_error']:.2g} err={err}")
    write_csv(out_dir / "p0_coordinate_invariants.csv", rows)
    return rows


def run_p1(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows: List[Dict[str, Any]] = []
    device = get_device(args.device)
    params = V45Params(train_size=1536, val_size=512, test_size=512)
    for dataset in [dataset_name(d) for d in parse_str_list(args.datasets)]:
        bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=params.train_size, val_size=params.val_size, test_size=params.test_size, seed=0, download=not args.no_download, allow_fake_data=args.allow_fake_data)
        set_seed(0)
        model = _make_model(bundle, params, device, ab=False)
        state = AdamState()
        idxs = _iter_steps(len(bundle.x_train), params.batch_size, 123, params.p2_steps)
        prev_fun: torch.Tensor | None = None
        for step, idx in enumerate(idxs):
            xb = bundle.x_train[idx].to(device)
            yb = bundle.y_train[idx].to(device)
            hb = bundle.x_val[: params.batch_size].to(device)
            yh = bundle.y_val[: params.batch_size].to(device)
            train_loss = F.cross_entropy(model(xb), yb).item()
            hold_loss = F.cross_entropy(model(hb), yh).item()
            _compute_loss_and_grad(model, xb, yb)
            updates, rms = _adam_updates(model, state, lr=params.adam_lr, mutate=True)
            fun_delta, fun_drift = _function_displacement(model, updates, xb)
            role_updates = _role_norms(updates)
            coef = _coef_norms(model)
            _apply_updates(coefficient_named_params(model), updates)
            feat = _feature_audit(model, hb, yh)
            geom = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
            occ = rbf_basis_occupancy_audit(model)
            test = evaluate(model, bundle.x_test, bundle.y_test, device=device, batch_size=params.eval_batch_size) if step in {0, 4, 9, 19} else {"acc": float("nan"), "loss": float("nan"), "ece": float("nan")}
            row = {
                "stage": "P1",
                "dataset": dataset,
                "step": step,
                "train_loss": train_loss,
                "holdout_loss": hold_loss,
                "test_acc": test["acc"],
                "feature_effective_rank_input": feat["input_feature_effective_rank"],
                "feature_effective_rank_block": feat["block_feature_effective_rank"],
                "class_centroid_separation": feat["class_centroid_separation"],
                "margin_mean": feat["margin_mean"],
                "margin_p10": feat["margin_p10"],
                "phi_prime_p95": geom["phi_prime_p95"],
                "jacobian_condition": geom["max_jac_condition"],
                "basis_occupancy_entropy": occ.get("rbf_basis_mean", float("nan")),
                "dead_basis_fraction": occ.get("rbf_basis_dead_frac", float("nan")),
                "function_displacement_norm": float(fun_delta.norm().cpu()),
                "function_displacement_drift": fun_drift,
                "function_displacement_cos_prev": _safe_cos(fun_delta, prev_fun) if prev_fun is not None else float("nan"),
                "input_update_norm": role_updates.get("input", 0.0),
                "block_update_norm": role_updates.get("block", 0.0),
                "output_update_norm": role_updates.get("output", 0.0),
                "input_coeff_norm": coef.get("input", 0.0),
                "block_coeff_norm": coef.get("block", 0.0),
                "output_coeff_norm": coef.get("output", 0.0),
                "AdamW_m_norm": _mean([float(v.norm().cpu()) for v in state.m.values()], float("nan")),
                "AdamW_v_norm": _mean([float(v.norm().cpu()) for v in state.v.values()], float("nan")),
                "AdamW_update_over_param": float(_flat_updates(updates).norm().cpu()) / max(1e-12, _mean(list(coef.values()), 1.0)),
                "AdamW_rms_norm": _mean([float(v.norm().cpu()) for v in rms.values()], float("nan")),
            }
            rows.append(row)
            prev_fun = fun_delta.detach()
            print(f"P1 {dataset} step={step} hold={hold_loss:.4f} rank={row['feature_effective_rank_block']:.2f} phi={row['phi_prime_p95']:.3f}")
    write_csv(out_dir / "p1_adamw_trajectory_forensic.csv", rows)
    return rows


def _run_p2_one(args: argparse.Namespace, dataset: str, seed: int, method: str, params: V45Params) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    device = get_device(args.device)
    bundle = load_vision_bundle(dataset, data_root=args.data_root, train_size=params.train_size, val_size=params.val_size, test_size=params.test_size, seed=seed, download=not args.no_download, allow_fake_data=args.allow_fake_data)
    set_seed(seed)
    model = _make_model(bundle, params, device, ab=("AB-RBF" in method))
    state = V44RunState()
    idxs = _iter_steps(len(bundle.x_train), params.batch_size, seed + 45, params.p2_steps)
    hb = bundle.x_val[: params.batch_size].to(device)
    yh = bundle.y_val[: params.batch_size].to(device)
    train0 = F.cross_entropy(model(bundle.x_train[idxs[0]].to(device)), bundle.y_train[idxs[0]].to(device)).item()
    hold0 = F.cross_entropy(model(hb), yh).item()
    feat0 = _feature_audit(model, hb, yh)
    geom0 = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
    trace: List[Dict[str, Any]] = []
    bad = 0
    cos_vals: List[float] = []
    r2_vals: List[float] = []
    train_after: Dict[int, float] = {}
    hold_after: Dict[int, float] = {}
    accepted_lrs: List[float] = []
    start = time.perf_counter()
    for step, idx in enumerate(idxs, start=1):
        xb = bundle.x_train[idx].to(device)
        yb = bundle.y_train[idx].to(device)
        tb = F.cross_entropy(model(xb), yb).item()
        hb_loss = F.cross_entropy(model(hb), yh).item()
        _compute_loss_and_grad(model, xb, yb)
        adam_ref, _ = _adam_updates(model, AdamState(), lr=params.adam_lr, mutate=True)
        adam_fun, _ = _function_displacement(model, adam_ref, xb)
        updates, stats = _method_updates(model, dataset, seed, method, params, state, xb, yb)
        cand_fun, drift = _function_displacement(model, updates, xb)
        cos_f = _safe_cos(cand_fun, adam_fun)
        r2 = _r2(cand_fun, adam_fun)
        cos_vals.append(cos_f)
        r2_vals.append(r2)
        _apply_updates(coefficient_named_params(model), updates)
        ta = F.cross_entropy(model(xb), yb).item()
        ha = F.cross_entropy(model(hb), yh).item()
        if ta > tb:
            bad += 1
        if step in {1, 5, 20}:
            train_after[step] = ta
            hold_after[step] = ha
        accepted_lrs.append(params.adam_lr if "AdamW" in method else params.fc_lr)
        feat = _feature_audit(model, hb, yh)
        geom = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
        trace.append(
            {
                "stage": "P2",
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "step": step,
                "train_loss_before": tb,
                "train_loss_after": ta,
                "holdout_loss_before": hb_loss,
                "holdout_loss_after": ha,
                "cos_function_with_adam": cos_f,
                "function_R2_with_adam": r2,
                "logit_drift": drift,
                "feature_rank": feat["block_feature_effective_rank"],
                "margin_mean": feat["margin_mean"],
                "phi_prime_p95": geom["phi_prime_p95"],
                "jacobian_condition": geom["max_jac_condition"],
                "u_m_norm": _mean([float(v.norm().cpu()) for v in state.fc.m.values()], float("nan")),
                "u_v_norm": _mean([float(v.norm().cpu()) for v in state.fc.v.values()], float("nan")),
                "accepted_lr": accepted_lrs[-1],
                "restart_count": 0,
                "optimizer_family": stats.get("optimizer_family", ""),
                "geometry_policy": stats.get("geometry_policy", ""),
            }
        )
    wall = time.perf_counter() - start
    feat_end = _feature_audit(model, hb, yh)
    geom_end = _estimate_geometry_any(model, hb, device=device, batch_size=params.audit_batch_size)
    test = evaluate(model, bundle.x_test, bundle.y_test, device=device, batch_size=params.eval_batch_size)
    row = {
        "stage": "P2",
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "steps": params.p2_steps,
        "test_acc_after_20step": test["acc"],
        "ECE": test["ece"],
        "holdout_1step_descent": hold0 - hold_after.get(1, float("nan")),
        "holdout_5step_descent": hold0 - hold_after.get(5, float("nan")),
        "holdout_20step_descent": hold0 - hold_after.get(20, float("nan")),
        "train_20step_descent": train0 - train_after.get(20, float("nan")),
        "bad_step_rate": bad / max(1, params.p2_steps),
        "cos_function_with_adam": _mean(cos_vals, float("nan")),
        "function_R2_with_adam": _mean(r2_vals, float("nan")),
        "rank_final": feat_end["block_feature_effective_rank"],
        "rank_change": feat_end["block_feature_effective_rank"] - feat0["block_feature_effective_rank"],
        "margin_final": feat_end["margin_mean"],
        "margin_p10_final": feat_end["margin_p10"],
        "margin_change": feat_end["margin_mean"] - feat0["margin_mean"],
        "phi_prime_p95": geom_end["phi_prime_p95"],
        "phi_change": geom_end["phi_prime_p95"] - geom0["phi_prime_p95"],
        "jacobian_condition": geom_end["max_jac_condition"],
        "jac_change": geom_end["max_jac_condition"] - geom0["max_jac_condition"],
        "accepted_lr": _mean(accepted_lrs, float("nan")),
        "u_m_norm": _mean([float(v.norm().cpu()) for v in state.fc.m.values()], float("nan")),
        "u_v_norm": _mean([float(v.norm().cpu()) for v in state.fc.v.values()], float("nan")),
        "u_update_norm": float("nan"),
        "restart_count": 0,
        "step_time_ms": 1000.0 * wall / max(1, params.p2_steps),
        "error": "",
    }
    return row, trace


def run_p2(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(args.out_dir)
    rows = [] if args.fresh else read_csv(out_dir / "p2_fcadam_backbone_audit.csv")
    trace = [] if args.fresh else read_csv(out_dir / "optimizer_dynamics_trace.csv")
    params = V45Params()
    datasets = [dataset_name(d) for d in parse_str_list(args.datasets)]
    seeds = parse_int_list(args.seeds)
    if args.seeds == add_args().get_default("seeds"):
        seeds = [0, 1, 2]
    wanted = set(parse_str_list(args.methods))
    methods = [m for m in P2_METHODS if not wanted or m in wanted]
    done = {(r.get("dataset"), int(r.get("seed", -1)), r.get("method")) for r in rows if not r.get("error")}
    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                if (dataset, seed, method) in done:
                    continue
                try:
                    row, tr = _run_p2_one(args, dataset, seed, method, params)
                    rows.append(row)
                    trace.extend(tr)
                    print(f"P2 {dataset} {method} seed={seed} hold20={row['holdout_20step_descent']:.4g} cos={row['cos_function_with_adam']:.3f} r2={row['function_R2_with_adam']:.3f}")
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"stage": "P2", "dataset": dataset, "seed": seed, "method": method, "error": repr(exc)})
                    print(f"P2 ERROR {dataset} {method} seed={seed}: {exc!r}")
                write_csv(out_dir / "p2_fcadam_backbone_audit.csv", rows)
                write_csv(out_dir / "optimizer_dynamics_trace.csv", trace)
    return rows


def _blank_later_files(out_dir: Path, reason: str) -> None:
    for name in [
        "p3_temporal_dynamics.csv",
        "p4_lyapunov_restart.csv",
        "p5_rank_margin_ablation.csv",
        "p6_candidate_selection.csv",
        "p7_confirm5.csv",
        "p8_confirm10.csv",
        "p9_failure_diagnosis.csv",
    ]:
        write_csv(out_dir / name, [{"status": "not_run", "reason": reason}])


def add_args() -> argparse.ArgumentParser:
    p = add_v3_args()
    p.description = __doc__
    p.set_defaults(packages="V4_5_P0_SMOKE", datasets="MNIST,Fashion-MNIST,KMNIST", out_dir=Path("results/v4_5"), seeds="0,1,2")
    return p


def main() -> int:
    args = add_args().parse_args()
    all_rows: List[Dict[str, Any]] = []
    for package in parse_str_list(args.packages):
        key = package.strip().upper().replace("-", "_")
        if key == "V4_5_P0_SMOKE":
            all_rows = run_p0(args)
        elif key == "V4_5_P1_ADAMW_FORENSIC":
            all_rows = run_p1(args)
        elif key == "V4_5_P2_FCADAM_BACKBONE":
            all_rows = run_p2(args)
            _blank_later_files(ensure_dir(args.out_dir), "pending analyzer gate decision")
        else:
            raise ValueError(f"unknown v4.5 package: {package}")
    return 0 if all_rows is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
