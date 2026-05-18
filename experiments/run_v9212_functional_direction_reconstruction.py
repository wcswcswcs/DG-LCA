#!/usr/bin/env python3
"""DG-KAN v9.2.12 functional direction reconstruction runner.

Reusable LQ model math and output-space functional direction construction live
under dgkan.models / dgkan.functional.  This runner owns protocols, gates,
artifacts, and route decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import run_v92_basis_source_kernel_gate as v92  # noqa: E402
from dgkan.artifacts.audit import audit_no_fake  # noqa: E402
from dgkan.artifacts.writer import artifact_hash_rows, ensure_dir, write_csv_rows, write_json  # noqa: E402
from dgkan.functional import lq_functional_predictor as fp_lq  # noqa: E402
from dgkan.functional import lq_output_space_functional as out_lq  # noqa: E402
from dgkan.functional import snr_gated_lq as snr_lq  # noqa: E402
from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from dgkan.optim.manual_adamw import AdamWState, ManualAdamWConfig  # noqa: E402


PLAN_PATH = ROOT / "docs" / "DG-KAN_v9.2.12_Functional_Direction_Reconstruction_完整实验计划.md"
SCRIPT_PATH = ROOT / "experiments" / "run_v9212_functional_direction_reconstruction.py"
PREV_V9211 = ROOT / "results" / "real_rerun_20260506" / "v9211_functional_causality_controller_first_20260509T220000Z"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _canonical_tasks(text: str) -> List[str]:
    return [v92._canonical_task(x) for x in _parse_list(text)]


def _device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value == "" or value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_svg(path: Path, title: str, subtitle: str) -> None:
    ensure_dir(path.parent)
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="980" height="190" viewBox="0 0 980 190">
  <rect width="980" height="190" fill="#f8fafc"/>
  <text x="28" y="54" font-family="Arial, sans-serif" font-size="25" fill="#111827">{title}</text>
  <text x="28" y="96" font-family="Arial, sans-serif" font-size="16" fill="#374151">{subtitle}</text>
  <text x="28" y="136" font-family="Arial, sans-serif" font-size="13" fill="#6b7280">Generated from measured CSV/JSON fields; no inferred pass values.</text>
</svg>
""",
        encoding="utf-8",
    )


def _not_run(stage: str, artifact: str, reason: str) -> Dict[str, Any]:
    return snr_lq.not_run_row(stage, artifact, reason)


def _candidate_specs(text: str) -> List[lq.LQSpec]:
    specs: List[lq.LQSpec] = []
    for name in _parse_list(text):
        if name == "LQ0":
            specs.append(lq.LQSpec("LQ0-LQ-t2-h256-AdamW", "t2", 256, "default", 1.0))
        elif name == "LQ1":
            specs.append(lq.LQSpec("LQ1-LQ-t2-h256-fanin-output-scale", "t2", 256, "default", 0.8))
        else:
            raise ValueError(f"unknown candidate alias {name}")
    return specs


def _parse_functional_candidates(text: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for item in [x.strip() for x in str(text).split(";") if x.strip()]:
        parts = [p.strip() for p in item.split("|")]
        if len(parts) != 3:
            raise ValueError(f"functional candidate must be target|subspace|solver, got {item}")
        out.append((parts[0], parts[1], parts[2]))
    return out


def _eval_lq_metrics(params: Sequence[torch.Tensor], mu: torch.Tensor, std: torch.Tensor, basis: str, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    fwd, _bwd = lq.functions_for_basis(basis)
    with torch.no_grad():
        logits = fwd(x, *params, mu, std, 2.0, 2.0)
    return v92._classification_metrics_from_logits(logits, y)


def _eval_lq_logits(params: Sequence[torch.Tensor], mu: torch.Tensor, std: torch.Tensor, basis: str, x: torch.Tensor) -> torch.Tensor:
    fwd, _bwd = lq.functions_for_basis(basis)
    with torch.no_grad():
        return fwd(x, *params, mu, std, 2.0, 2.0)


def _logit_displacement_norm(params: Sequence[torch.Tensor], delta: Sequence[torch.Tensor], mu: torch.Tensor, std: torch.Tensor, basis: str, x: torch.Tensor) -> float:
    fwd, _bwd = lq.functions_for_basis(basis)
    with torch.no_grad():
        before = fwd(x, *params, mu, std, 2.0, 2.0)
        after_params = [p.detach() + d.detach() for p, d in zip(params, delta)]
        after = fwd(x, *after_params, mu, std, 2.0, 2.0)
    return float((after - before).float().norm().detach().cpu())


def _quadratic_active_fraction(rows: Sequence[Dict[str, Any]]) -> float:
    for row in rows:
        role = str(row.get("role", ""))
        if "quadratic" in role or "legendre_p2" in role:
            return _to_float(row.get("SNR_active_fraction_tau1"), 0.0)
    return _to_float(rows[-1].get("SNR_active_fraction_tau1"), 0.0) if rows else 0.0


def _quadratic_snr(rows: Sequence[Dict[str, Any]]) -> float:
    for row in rows:
        role = str(row.get("role", ""))
        if "quadratic" in role or "legendre_p2" in role:
            return _to_float(row.get("SNR_role"), 0.0)
    return _to_float(rows[-1].get("SNR_role"), 0.0) if rows else 0.0


def _clone_params(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _copy_states(states: Sequence[AdamWState]) -> List[AdamWState]:
    return [AdamWState(step=s.step, m=s.m.detach().clone(), v=s.v.detach().clone()) for s in states]


def _apply_in_place(params: Sequence[torch.Tensor], step: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, d in zip(params, step):
            p.add_(d)


def _advance_adamw(
    params: Sequence[torch.Tensor],
    states: Sequence[AdamWState],
    bwd,
    x: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    opt_cfg: ManualAdamWConfig,
) -> List[torch.Tensor]:
    pack = bwd(x, y, *params, mu, std, 2.0, 2.0)
    grads = [g.detach() for g in pack[1:]]
    v92._adamw_update_foreach_(params, grads, states, opt_cfg)
    return grads


def _p0_source_recap(prev_dir: Path) -> List[Dict[str, Any]]:
    route = _read_json(prev_dir / "route_decision.json")
    summary = _read_csv(prev_dir / "p3_event_controller_direction_selection.csv")
    pass_count = sum(_to_int(r.get("event_causality_pass")) for r in summary)
    d9 = [r for r in summary if r.get("direction_id") == "D9-SignalChannelProjection"]
    covs = [_to_float(r.get("event_coverage")) for r in d9]
    norms = []
    p2_rows = _read_csv(prev_dir / "p2_paired_event_replay_causality.csv")
    for r in p2_rows:
        if r.get("direction_id") == "D9-SignalChannelProjection" and r.get("branch") == "RealFunctional":
            norms.append(_to_float(r.get("functional_norm")))
    return [{
        "stage": "P0_V9211_REPRODUCTION_SOURCE_RECAP",
        "source_artifact": str(prev_dir.relative_to(ROOT)) if prev_dir.exists() else str(prev_dir),
        "source_route": route.get("route", ""),
        "source_p2_event_causality_pass_count": pass_count,
        "source_d9_mean_coverage": sum(covs) / max(1, len(covs)),
        "source_d9_mean_functional_norm": sum(norms) / max(1, len(norms)),
        "source_primary_blocker": route.get("primary_blocker", ""),
        "p0_reproduction_pass": int(route.get("route") == "R5-FunctionalControlEquivalent" and pass_count == 0),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]


def _p1_control_equivalence_autopsy(prev_dir: Path) -> List[Dict[str, Any]]:
    rows = _read_csv(prev_dir / "p2_paired_event_replay_causality.csv")
    summary = _read_csv(prev_dir / "p3_event_controller_direction_selection.csv")
    groups: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    for r in rows:
        if r.get("branch") == "RealFunctional":
            groups.setdefault((r.get("direction_id", ""), r.get("event_type", ""), r.get("horizon", "")), []).append(r)
    out: List[Dict[str, Any]] = []
    for (direction, event, horizon), rs in sorted(groups.items()):
        if horizon not in {"1", "5", "20"}:
            continue
        coverage = 0.0
        matches = [s for s in summary if s.get("direction_id") == direction and s.get("event_type") == event and str(s.get("horizon")) == str(horizon)]
        if matches:
            coverage = _to_float(matches[0].get("event_coverage"))
        mean = lambda key: sum(_to_float(r.get(key)) for r in rs) / max(1, len(rs))
        functional_norm = mean("functional_norm")
        cos_task = mean("cos_with_task_gradient")
        ce = mean("CEp99_delta_vs_adamw")
        margin = mean("margin_p10_delta_vs_adamw")
        curv = mean("curvature_delta_vs_adamw")
        out_effect_proxy = abs(ce) + abs(margin) + abs(curv)
        mechanisms: List[str] = []
        if functional_norm < 1.0e-6:
            mechanisms.append("M1-projection_neutralization")
        if abs(cos_task) > 0.90:
            mechanisms.append("M2-adamw_or_task_parallel")
        if coverage < 0.01 or coverage > 0.90:
            mechanisms.append("M3-event_degenerate")
        if out_effect_proxy < 1.0e-4:
            mechanisms.append("M4-output_effect_too_small")
        if matches and _to_int(matches[0].get("event_causality_pass")) == 0:
            mechanisms.append("M5-control_equivalent")
        out.append({
            "stage": "P1_CONTROL_EQUIVALENCE_AUTOPSY",
            "direction_id": direction,
            "event_type": event,
            "horizon": horizon,
            "branch": "RealFunctional",
            "cos_with_adamw": "not_measured_in_v9211_source",
            "cos_with_random": "not_measured_in_v9211_source",
            "cos_with_task_gradient": cos_task,
            "projected_norm_ratio": "not_measured_in_v9211_source",
            "functional_norm": functional_norm,
            "output_displacement_norm": "not_measured_in_v9211_source",
            "output_effect_proxy": out_effect_proxy,
            "hard_sample_output_displacement": "not_measured_in_v9211_source",
            "CEtail_output_displacement": ce,
            "margin_tail_output_displacement": margin,
            "basis_entropy_delta": mean("basis_entropy_delta_vs_adamw"),
            "event_coverage": coverage,
            "failure_mechanisms": ",".join(sorted(set(mechanisms))) if mechanisms else "unattributed",
            "p1_autopsy_pass": int(bool(mechanisms)),
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    return out or [_not_run("P1_CONTROL_EQUIVALENCE_AUTOPSY", "p1_control_equivalence_autopsy.csv", "v9211_p2_rows_missing")]


def _random_like_matched(params: Sequence[torch.Tensor], reference: Sequence[torch.Tensor], seed: int, device: torch.device) -> List[torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(int(seed))
    raw = [torch.randn(p.shape, device=device, generator=gen, dtype=p.dtype) for p in params]
    return snr_lq.scale_step(raw, snr_lq.step_norm(reference) / snr_lq.step_norm(raw).clamp_min(1.0e-12))


def _calibrate_event_flags(contexts: Sequence[Dict[str, Any]], event_types: Sequence[str], target_coverage: float) -> Dict[Tuple[str, str], bool]:
    flags: Dict[Tuple[str, str], bool] = {}
    groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for ctx in contexts:
        groups.setdefault((ctx["candidate_id"], ctx["dataset"], int(ctx["seed"])), []).append(ctx)
    for _group_key, group in groups.items():
        n = len(group)
        k = max(1, int(round(float(target_coverage) * n)))
        score_map: Dict[str, List[Tuple[float, str]]] = {}
        ce_values = torch.tensor([float(g["metrics"]["CE_p99"]) for g in group], dtype=torch.float64)
        margin_values = torch.tensor([-float(g["metrics"]["correct_margin_p10"]) for g in group], dtype=torch.float64)
        wrong_values = torch.tensor([float(g["metrics"]["wrong_confidence_p95"]) for g in group], dtype=torch.float64)
        curv_values = torch.tensor([float(g["mechanism"]["curvature_proxy"]) for g in group], dtype=torch.float64)
        entropy_values = torch.tensor([-float(g["mechanism"]["basis_usage_entropy"]) for g in group], dtype=torch.float64)
        z = lambda t: (t - t.mean()) / t.std(unbiased=False).clamp_min(1.0e-12)
        composite = z(ce_values) + z(margin_values) + z(wrong_values) + z(curv_values) + z(entropy_values)
        for idx, ctx in enumerate(group):
            score_map.setdefault("E1-CalibratedCEp99Tail", []).append((float(ce_values[idx]), ctx["context_id"]))
            score_map.setdefault("E2-CalibratedMarginTail", []).append((float(margin_values[idx]), ctx["context_id"]))
            score_map.setdefault("E3-CalibratedWrongConfidence", []).append((float(wrong_values[idx]), ctx["context_id"]))
            score_map.setdefault("E4-CalibratedCurvatureSpike", []).append((float(curv_values[idx]), ctx["context_id"]))
            score_map.setdefault("E5-CalibratedBasisCollapse", []).append((float(entropy_values[idx]), ctx["context_id"]))
            score_map.setdefault("E6-CompositeSparseEvent", []).append((float(composite[idx]), ctx["context_id"]))
        for event_type in event_types:
            ranked = sorted(score_map.get(event_type, []), key=lambda x: x[0], reverse=True)
            selected = {cid for _score, cid in ranked[:k]}
            for ctx in group:
                flags[(event_type, ctx["context_id"])] = ctx["context_id"] in selected
    return flags


def _collect_contexts_for_dataset(args: argparse.Namespace, dataset: str, device: torch.device) -> Tuple[List[Dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, str]:
    min_train = max(int(args.train_size), int(args.batch_size) * (int(args.snr_warmup_steps) + int(args.p2_event_steps) + max([int(x) for x in _parse_list(args.p2_horizons)]) + 4))
    x_train, y_train, x_test, y_test, input_dim, output_dim, protocol = v92._load_task(args, dataset, train_size=min_train, test_size=int(args.p2_eval_size))
    x_train = x_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device)
    x_eval = x_test.to(device=device, dtype=torch.float32)
    y_eval = y_test.to(device=device)
    specs = _candidate_specs(args.candidates)
    max_horizon = max([int(x) for x in _parse_list(args.p2_horizons)])
    contexts: List[Dict[str, Any]] = []
    snr_cfg = snr_lq.SNRConfig(tau1=float(args.snr_tau1), tau2=float(args.snr_tau2), temperature=float(args.snr_smooth_temperature))
    for seed_text in _parse_list(args.seeds):
        seed = int(seed_text)
        for spec in specs:
            params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_train, device, seed + 921200)
            states = [AdamWState.zeros_like(p) for p in params]
            opt_cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
            scalar_state = snr_lq.ScalarRoleSNRState.zeros(len(snr_lq.role_names_for_basis(spec.basis)), beta=float(args.snr_ema_beta), device=device)
            _fwd, bwd = lq.functions_for_basis(spec.basis)
            for warm_step in range(int(args.snr_warmup_steps)):
                gen = torch.Generator(device=device).manual_seed(9212100 + seed * 10000 + warm_step)
                idx = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)[: int(args.batch_size)]
                grads = _advance_adamw(params, states, bwd, x_train[idx], y_train[idx], mu, std, opt_cfg)
                snr_lq.scalar_ema_snr_from_grads(grads=grads, scalar_state=scalar_state, basis=spec.basis, snr_cfg=snr_cfg)
            for event_step in range(int(args.p2_event_steps)):
                gen = torch.Generator(device=device).manual_seed(9212200 + seed * 10000 + event_step)
                perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
                event_idx = perm[: int(args.batch_size)]
                future_batches = []
                for hstep in range(1, max_horizon):
                    fgen = torch.Generator(device=device).manual_seed(9212300 + seed * 10000 + event_step * 100 + hstep)
                    future_batches.append(torch.randperm(int(x_train.shape[0]), device=device, generator=fgen)[: int(args.batch_size)])
                xb = x_train[event_idx]
                yb = y_train[event_idx]
                pack = bwd(xb, yb, *params, mu, std, 2.0, 2.0)
                grads = [g.detach() for g in pack[1:]]
                snr_rows = snr_lq.scalar_ema_snr_from_grads(grads=grads, scalar_state=scalar_state, basis=spec.basis, snr_cfg=snr_cfg)
                metrics = _eval_lq_metrics(params, mu, std, spec.basis, x_eval, y_eval)
                mechanism = fp_lq.mechanism_metrics(params, mu, std, spec.basis, x_eval[: min(256, int(x_eval.shape[0]))])
                logits_event = _eval_lq_logits(params, mu, std, spec.basis, xb)
                contexts.append({
                    "context_id": f"{dataset}-s{seed}-{spec.candidate_id}-e{event_step}",
                    "candidate_id": spec.candidate_id,
                    "basis": spec.basis,
                    "dataset": dataset,
                    "seed": seed,
                    "protocol": protocol,
                    "event_step": event_step,
                    "params": _clone_params(params),
                    "states": _copy_states(states),
                    "mu": mu.detach().clone(),
                    "std": std.detach().clone(),
                    "x_event": xb.detach().clone(),
                    "y_event": yb.detach().clone(),
                    "future_batches": [b.detach().clone() for b in future_batches],
                    "grads": [g.detach().clone() for g in grads],
                    "task_step": snr_lq.gradient_descent_task_step(grads, float(args.lr)),
                    "metrics": metrics,
                    "mechanism": mechanism,
                    "logits_event": logits_event.detach().clone(),
                    "active_fraction": _quadratic_active_fraction(snr_rows),
                    "quadratic_snr": _quadratic_snr(snr_rows),
                })
                v92._adamw_update_foreach_(params, grads, states, opt_cfg)
    return contexts, x_train, y_train, x_eval, y_eval, input_dim, output_dim, protocol


def _branch_delta(
    *,
    branch: str,
    params: Sequence[torch.Tensor],
    grads: Sequence[torch.Tensor],
    task_step: Sequence[torch.Tensor],
    mu: torch.Tensor,
    std: torch.Tensor,
    basis: str,
    x: torch.Tensor,
    y: torch.Tensor,
    logits: torch.Tensor,
    dataset: str,
    target_id: str,
    subspace_id: str,
    solver_id: str,
    real_delta: Sequence[torch.Tensor],
    step_fraction: float,
    seed: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    if branch in {"AdamWOnly", "C1-NoOpMatchedOverhead"}:
        return snr_lq.zero_like_params(params), {"target_norm": 0.0, "output_target_fit_r2": 0.0, "target_selected_fraction": 0.0}
    if branch == "C2-RandomMatchedNorm":
        raw = _random_like_matched(params, real_delta, seed, device)
        return raw, {"target_norm": 0.0, "output_target_fit_r2": 0.0, "target_selected_fraction": 0.0}
    if branch == "C6-AdamWParallelDirection":
        return snr_lq.scale_step(task_step, snr_lq.step_norm(real_delta) / snr_lq.step_norm(task_step).clamp_min(1.0e-12)), {"target_norm": 0.0, "output_target_fit_r2": 0.0, "target_selected_fraction": 0.0}
    target, target_info = out_lq.build_output_target(target_id, logits, y, dataset=dataset)
    if branch == "C3-ShuffledTarget":
        gen = torch.Generator(device=device).manual_seed(seed)
        perm = torch.randperm(int(target.shape[0]), device=device, generator=gen)
        target = target[perm]
    raw = out_lq.output_vjp_direction(params, mu, std, basis, x, target)
    sub = out_lq.apply_subspace(raw, params, grads, subspace_id)
    delta, solve_info = out_lq.solve_functional_step(params=params, task_grads=grads, task_step=task_step, raw_direction=sub, solver_id=solver_id, step_fraction=step_fraction)
    fit = out_lq.output_fit_metrics(params=params, delta=delta, mu=mu, std=std, basis=basis, x=x, target_delta_logits=target)
    return delta, {**target_info, **solve_info, **fit}


def _run_p2_output_direction_factory(args: argparse.Namespace, device: torch.device) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    target_trace: List[Dict[str, Any]] = []
    branch_trace: List[Dict[str, Any]] = []
    functional_candidates = _parse_functional_candidates(args.p2_candidates)
    event_types = _parse_list(args.p2_event_types)
    horizons = [int(x) for x in _parse_list(args.p2_horizons)]
    branches = ["AdamWOnly", "RealFunctional", "C1-NoOpMatchedOverhead", "C2-RandomMatchedNorm", "C3-ShuffledTarget", "C6-AdamWParallelDirection"]
    for dataset in _canonical_tasks(args.datasets):
        contexts, x_train, y_train, x_eval, y_eval, _input_dim, _output_dim, protocol = _collect_contexts_for_dataset(args, dataset, device)
        event_flags = _calibrate_event_flags(contexts, event_types, float(args.p2_target_coverage))
        for ctx in contexts:
            params0 = ctx["params"]
            states0 = ctx["states"]
            mu = ctx["mu"]
            std = ctx["std"]
            basis = ctx["basis"]
            x_event = ctx["x_event"]
            y_event = ctx["y_event"]
            grads = ctx["grads"]
            task_step = ctx["task_step"]
            _fwd, bwd = lq.functions_for_basis(basis)
            opt_cfg = ManualAdamWConfig(lr=float(args.lr), weight_decay=float(args.weight_decay))
            before_metrics = ctx["metrics"]
            before_mech = ctx["mechanism"]
            adamw_logit_displacement_norm = _logit_displacement_norm(params0, task_step, mu, std, basis, x_event)
            adamw_ref: Dict[int, Dict[str, Any]] = {}
            ref_params0 = _clone_params(params0)
            ref_states0 = _copy_states(states0)
            _advance_adamw(ref_params0, ref_states0, bwd, x_event, y_event, mu, std, opt_cfg)
            for horizon in horizons:
                eval_params = _clone_params(ref_params0)
                eval_states = _copy_states(ref_states0)
                for fidx in ctx["future_batches"][: max(0, horizon - 1)]:
                    _advance_adamw(eval_params, eval_states, bwd, x_train[fidx], y_train[fidx], mu, std, opt_cfg)
                adamw_ref[horizon] = {
                    "metrics": _eval_lq_metrics(eval_params, mu, std, basis, x_eval, y_eval),
                    "mech": fp_lq.mechanism_metrics(eval_params, mu, std, basis, x_eval[: min(256, int(x_eval.shape[0]))]),
                }
            for target_id, subspace_id, solver_id in functional_candidates:
                real_delta, real_info = _branch_delta(
                    branch="RealFunctional",
                    params=params0,
                    grads=grads,
                    task_step=task_step,
                    mu=mu,
                    std=std,
                    basis=basis,
                    x=x_event,
                    y=y_event,
                    logits=ctx["logits_event"],
                    dataset=dataset,
                    target_id=target_id,
                    subspace_id=subspace_id,
                    solver_id=solver_id,
                    real_delta=snr_lq.zero_like_params(params0),
                    step_fraction=float(args.step_fraction),
                    seed=9212400 + int(ctx["seed"]) * 1000 + int(ctx["event_step"]),
                    device=device,
                )
                candidate = out_lq.candidate_id(target_id, subspace_id, solver_id)
                target_trace.append({
                    "stage": "OUTPUT_TARGET_TRACE",
                    "context_id": ctx["context_id"],
                    "functional_candidate": candidate,
                    "target_id": target_id,
                    "subspace_id": subspace_id,
                    "solver_id": solver_id,
                    **real_info,
                    "fake_data_used": 0,
                    "proxy_row_used": 0,
                    "cpu_offload_used": 0,
                })
                for event_type in event_types:
                    event_triggered = int(event_flags.get((event_type, ctx["context_id"]), False))
                    for branch in branches:
                        bparams = _clone_params(params0)
                        bstates = _copy_states(states0)
                        if branch == "RealFunctional":
                            delta = [d.detach().clone() for d in real_delta]
                            branch_info = real_info
                        else:
                            delta, branch_info = _branch_delta(
                                branch=branch,
                                params=params0,
                                grads=grads,
                                task_step=task_step,
                                mu=mu,
                                std=std,
                                basis=basis,
                                x=x_event,
                                y=y_event,
                                logits=ctx["logits_event"],
                                dataset=dataset,
                                target_id=target_id,
                                subspace_id=subspace_id,
                                solver_id=solver_id,
                                real_delta=real_delta,
                                step_fraction=float(args.step_fraction),
                                seed=9212500 + int(ctx["seed"]) * 1000 + int(ctx["event_step"]),
                                device=device,
                            )
                        if not event_triggered or branch in {"AdamWOnly", "C1-NoOpMatchedOverhead"}:
                            delta = snr_lq.zero_like_params(bparams)
                        event_accepted = int(event_triggered and branch not in {"AdamWOnly", "C1-NoOpMatchedOverhead"} and bool((snr_lq.step_norm(delta) > 0).detach().cpu()))
                        branch_results: Dict[int, Dict[str, Any]] = {}
                        if branch == "AdamWOnly":
                            branch_results = adamw_ref
                            elapsed_ms: Any = "cached_reference"
                            peak_mb: Any = "cached_reference"
                        else:
                            if device.type == "cuda":
                                torch.cuda.reset_peak_memory_stats(device)
                            _sync(device)
                            t0 = time.perf_counter()
                            _advance_adamw(bparams, bstates, bwd, x_event, y_event, mu, std, opt_cfg)
                            if event_accepted:
                                _apply_in_place(bparams, delta)
                            for horizon in horizons:
                                eval_params = _clone_params(bparams)
                                eval_states = _copy_states(bstates)
                                for fidx in ctx["future_batches"][: max(0, horizon - 1)]:
                                    _advance_adamw(eval_params, eval_states, bwd, x_train[fidx], y_train[fidx], mu, std, opt_cfg)
                                metrics = _eval_lq_metrics(eval_params, mu, std, basis, x_eval, y_eval)
                                mech = fp_lq.mechanism_metrics(eval_params, mu, std, basis, x_eval[: min(256, int(x_eval.shape[0]))])
                                branch_results[horizon] = {"metrics": metrics, "mech": mech}
                            _sync(device)
                            elapsed_ms = (time.perf_counter() - t0) * 1000.0
                            peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0
                        for horizon in horizons:
                            adamw_metrics = adamw_ref[horizon]["metrics"]
                            adamw_mech = adamw_ref[horizon]["mech"]
                            bm = branch_results[horizon]["metrics"]
                            bmech = branch_results[horizon]["mech"]
                            step_norm = float(snr_lq.step_norm(delta).detach().cpu())
                            task_norm = float(snr_lq.step_norm(task_step).detach().cpu())
                            functional_logit_norm = _to_float(branch_info.get("output_displacement_norm"), 0.0)
                            rows.append({
                                "stage": "P2_OUTPUT_SPACE_DIRECTION_FACTORY",
                                "context_id": ctx["context_id"],
                                "event_id": f"{ctx['context_id']}-{candidate}-{event_type}",
                                "functional_candidate": candidate,
                                "target_id": target_id,
                                "subspace_id": subspace_id,
                                "solver_id": solver_id,
                                "candidate_id": ctx["candidate_id"],
                                "dataset": dataset,
                                "seed": ctx["seed"],
                                "protocol": protocol,
                                "event_step": ctx["event_step"],
                                "event_type": event_type,
                                "branch": branch,
                                "horizon": horizon,
                                "event_triggered": event_triggered,
                                "event_accepted": event_accepted,
                                "event_coverage_target": float(args.p2_target_coverage),
                                "target_norm": branch_info.get("target_norm", 0.0),
                                "target_selected_fraction": branch_info.get("target_selected_fraction", 0.0),
                                "raw_direction_norm": branch_info.get("raw_direction_norm", 0.0),
                                "solved_delta_norm": step_norm,
                                "task_step_norm": task_norm,
                                "functional_param_step_ratio_vs_task": step_norm / max(task_norm, 1.0e-12),
                                "adamw_logit_displacement_norm": adamw_logit_displacement_norm,
                                "output_displacement_ratio_vs_adamw_logits": functional_logit_norm / max(adamw_logit_displacement_norm, 1.0e-12),
                                "output_target_fit_r2": branch_info.get("output_target_fit_r2", 0.0),
                                "output_displacement_norm": branch_info.get("output_displacement_norm", 0.0),
                                "norm_after_projection_ratio": branch_info.get("norm_after_projection_ratio", 0.0),
                                "cos_with_task_gradient": branch_info.get("cos_with_task_gradient", 0.0),
                                "acc": bm["acc"],
                                "adamw_acc": adamw_metrics["acc"],
                                "acc_delta_vs_adamw": bm["acc"] - adamw_metrics["acc"],
                                "loss": bm["loss"],
                                "adamw_loss": adamw_metrics["loss"],
                                "holdout_loss_delta_vs_adamw": bm["loss"] - adamw_metrics["loss"],
                                "NLL_delta_vs_adamw": bm["NLL"] - adamw_metrics["NLL"],
                                "ECE_delta_vs_adamw": bm["ECE"] - adamw_metrics["ECE"],
                                "CEp99": bm["CE_p99"],
                                "adamw_CEp99": adamw_metrics["CE_p99"],
                                "CEp99_delta_vs_adamw": bm["CE_p99"] - adamw_metrics["CE_p99"],
                                "margin_p10": bm["correct_margin_p10"],
                                "adamw_margin_p10": adamw_metrics["correct_margin_p10"],
                                "margin_p10_delta_vs_adamw": bm["correct_margin_p10"] - adamw_metrics["correct_margin_p10"],
                                "curvature_proxy": bmech["curvature_proxy"],
                                "adamw_curvature_proxy": adamw_mech["curvature_proxy"],
                                "curvature_delta_vs_adamw": bmech["curvature_proxy"] - adamw_mech["curvature_proxy"],
                                "basis_entropy_delta_vs_adamw": bmech["basis_usage_entropy"] - adamw_mech["basis_usage_entropy"],
                                "before_CEp99": before_metrics["CE_p99"],
                                "before_margin_p10": before_metrics["correct_margin_p10"],
                                "before_curvature": before_mech["curvature_proxy"],
                                "active_fraction": ctx["active_fraction"],
                                "quadratic_snr": ctx["quadratic_snr"],
                                "overhead_event_ms": elapsed_ms,
                                "overhead_amortized": "not_measured_in_p2_factory",
                                "adamw_reference_mode": "cached_once_per_event_checkpoint",
                                "peak_memory_mb": peak_mb,
                                "loss_type": "CE",
                                "label_smoothing": 0,
                                "external_teacher_used": 0,
                                "self_teacher_used": 0,
                                "teacher_logits_used": 0,
                                "distillation_used": 0,
                                "geometry_loss_used": 0,
                                "sampler_changed": 0,
                                "class_weight_used": 0,
                                "uses_loss_backward": 0,
                                "functional_update_used": int(branch == "RealFunctional" and event_accepted),
                                "fake_data_used": 0,
                                "proxy_row_used": 0,
                                "cpu_offload_used": 0,
                            })
                        branch_trace.append({
                            "stage": "PAIRED_REPLAY_BRANCH_TRACE",
                            "event_id": f"{ctx['context_id']}-{candidate}-{event_type}",
                            "branch": branch,
                            "event_triggered": event_triggered,
                            "event_accepted": event_accepted,
                            "elapsed_ms": elapsed_ms,
                            "peak_memory_mb": peak_mb,
                            "fake_data_used": 0,
                            "proxy_row_used": 0,
                            "cpu_offload_used": 0,
                        })
    return rows, target_trace, branch_trace


def _summarize_p2(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("branch") == "RealFunctional":
            groups.setdefault((str(r.get("functional_candidate")), str(r.get("event_type")), str(r.get("horizon"))), []).append(r)
    out: List[Dict[str, Any]] = []
    for (candidate, event_type, horizon), real_rows in groups.items():
        controls = [
            r for r in rows
            if r.get("functional_candidate") == candidate
            and r.get("event_type") == event_type
            and str(r.get("horizon")) == horizon
            and r.get("branch") in {"C1-NoOpMatchedOverhead", "C2-RandomMatchedNorm", "C3-ShuffledTarget", "C6-AdamWParallelDirection"}
        ]
        mean = lambda rs, k: sum(_to_float(r.get(k)) for r in rs) / max(1, len(rs))
        event_count = len(real_rows)
        accepted = sum(_to_int(r.get("event_accepted")) for r in real_rows)
        triggered = sum(_to_int(r.get("event_triggered")) for r in real_rows)
        accepted_rows = [r for r in real_rows if _to_int(r.get("event_accepted")) == 1]
        bad = sum(int(_to_float(r.get("holdout_loss_delta_vs_adamw")) > 0.0) for r in accepted_rows)
        nonharm = sum(int(_to_float(r.get("holdout_loss_delta_vs_adamw")) <= 0.0) for r in accepted_rows)
        coverage = accepted / max(1, event_count)
        bad_rate = bad / max(1, len(accepted_rows))
        nonharm_rate = nonharm / max(1, len(accepted_rows))
        real_ce = mean(accepted_rows or real_rows, "CEp99_delta_vs_adamw")
        real_margin = mean(accepted_rows or real_rows, "margin_p10_delta_vs_adamw")
        real_curv = mean(accepted_rows or real_rows, "curvature_delta_vs_adamw")
        real_ece = mean(accepted_rows or real_rows, "ECE_delta_vs_adamw")
        real_nll = mean(accepted_rows or real_rows, "NLL_delta_vs_adamw")
        best_control_ce = min([mean([r for r in controls if r.get("branch") == b], "CEp99_delta_vs_adamw") for b in {"C1-NoOpMatchedOverhead", "C2-RandomMatchedNorm", "C3-ShuffledTarget", "C6-AdamWParallelDirection"}] or [0.0])
        best_control_margin = max([mean([r for r in controls if r.get("branch") == b], "margin_p10_delta_vs_adamw") for b in {"C1-NoOpMatchedOverhead", "C2-RandomMatchedNorm", "C3-ShuffledTarget", "C6-AdamWParallelDirection"}] or [0.0])
        best_control_curv = min([mean([r for r in controls if r.get("branch") == b], "curvature_delta_vs_adamw") for b in {"C1-NoOpMatchedOverhead", "C2-RandomMatchedNorm", "C3-ShuffledTarget", "C6-AdamWParallelDirection"}] or [0.0])
        output_ratio = mean(accepted_rows or real_rows, "output_displacement_ratio_vs_adamw_logits")
        fit_r2 = mean(accepted_rows or real_rows, "output_target_fit_r2")
        task_safe_rate = sum(int(_to_float(r.get("acc_delta_vs_adamw")) >= -0.005) for r in accepted_rows) / max(1, len(accepted_rows))
        mechanism = int(real_ce < 0.0 or real_margin > 0.0 or real_curv < 0.0)
        control = int(real_ce <= best_control_ce - 1.0e-5 or real_margin >= best_control_margin + 0.002 or real_curv <= best_control_curv - 1.0e-7 or real_ece < 0.0 or real_nll < 0.0)
        direction_pass = int(
            str(horizon) in {"5", "20"}
            and 0.03 <= coverage <= 0.15
            and bad_rate <= 0.05
            and nonharm_rate >= 0.70
            and output_ratio >= 0.05
            and task_safe_rate >= 0.95
            and mechanism
            and control
        )
        target_id, subspace_id, solver_id = candidate.split("|")
        out.append({
            "stage": "P2_OUTPUT_DIRECTION_FACTORY_SUMMARY",
            "functional_candidate": candidate,
            "target_id": target_id,
            "subspace_id": subspace_id,
            "solver_id": solver_id,
            "event_type": event_type,
            "horizon": horizon,
            "rows": event_count,
            "triggered_rows": triggered,
            "accepted_rows": accepted,
            "coverage": coverage,
            "bad_event_rate": bad_rate,
            "holdout_nonharm_fraction": nonharm_rate,
            "task_safe_rate": task_safe_rate,
            "output_displacement_ratio": output_ratio,
            "output_target_fit_r2": fit_r2,
            "real_CEp99_delta": real_ce,
            "best_control_CEp99_delta": best_control_ce,
            "real_margin_delta": real_margin,
            "best_control_margin_delta": best_control_margin,
            "real_curvature_delta": real_curv,
            "best_control_curvature_delta": best_control_curv,
            "real_ECE_delta": real_ece,
            "real_NLL_delta": real_nll,
            "mechanism_survivor": mechanism,
            "control_survivor": control,
            "p2_output_direction_pass": direction_pass,
            "fake_data_used": 0,
            "proxy_row_used": 0,
            "cpu_offload_used": 0,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--previous-v9211-dir", default=str(PREV_V9211.relative_to(ROOT)))
    parser.add_argument("--candidates", default="LQ0,LQ1")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-size", type=int, default=9984)
    parser.add_argument("--snr-ema-beta", type=float, default=0.97)
    parser.add_argument("--snr-warmup-steps", type=int, default=36)
    parser.add_argument("--snr-tau1", type=float, default=20.0)
    parser.add_argument("--snr-tau2", type=float, default=40.0)
    parser.add_argument("--snr-smooth-temperature", type=float, default=1.0)
    parser.add_argument("--step-fraction", type=float, default=0.03)
    parser.add_argument("--p2-event-steps", type=int, default=10)
    parser.add_argument("--p2-target-coverage", type=float, default=0.10)
    parser.add_argument("--p2-horizons", default="1,5,20")
    parser.add_argument("--p2-event-types", default="E1-CalibratedCEp99Tail,E2-CalibratedMarginTail,E6-CompositeSparseEvent")
    parser.add_argument("--p2-candidates", default="O1-HardTailLogitCorrection|S1-QuadraticCoeffSubspace|SOL2-ConstrainedTaskSafe;O1-HardTailLogitCorrection|S4-LiftPlusQuadraticSubspace|SOL2-ConstrainedTaskSafe;O2-MarginTailExpansion|S1-QuadraticCoeffSubspace|SOL2-ConstrainedTaskSafe;O2-MarginTailExpansion|S4-LiftPlusQuadraticSubspace|SOL2-ConstrainedTaskSafe;O2-MarginTailExpansion|S6-OrthogonalToAdamWSubspace|SOL3-TrustRegionTaskSafe;O3-CalibrationTailCompression|S3-OutputLinearSubspace|SOL2-ConstrainedTaskSafe;O3-CalibrationTailCompression|S4-LiftPlusQuadraticSubspace|SOL2-ConstrainedTaskSafe;O4-CurvatureOutputFlattening|S5-RecentSignalSubspace|SOL3-TrustRegionTaskSafe;O4-CurvatureOutputFlattening|S6-OrthogonalToAdamWSubspace|SOL3-TrustRegionTaskSafe;O6-KMNISTHardModeOutputTarget|S4-LiftPlusQuadraticSubspace|SOL2-ConstrainedTaskSafe;O6-KMNISTHardModeOutputTarget|S5-RecentSignalSubspace|SOL3-TrustRegionTaskSafe;O6-KMNISTHardModeOutputTarget|S6-OrthogonalToAdamWSubspace|SOL3-TrustRegionTaskSafe")
    parser.add_argument("--p2-eval-size", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "figures")
    device = _device_from_arg(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    prev_dir = Path(args.previous_v9211_dir)
    if not prev_dir.is_absolute():
        prev_dir = ROOT / prev_dir

    write_json(out_dir / "run_manifest.json", {
        "created_utc": _now_iso(),
        "script": str(SCRIPT_PATH.relative_to(ROOT)),
        "plan": str(PLAN_PATH.relative_to(ROOT)),
        "device": str(device),
        "torch": torch.__version__,
        "args": vars(args),
        "source_artifacts": {"v9211": str(prev_dir.relative_to(ROOT)) if prev_dir.exists() else str(prev_dir)},
    })
    contract_rows = [{
        "stage": "P0_CONTRACT",
        "loss_type": "CE",
        "label_smoothing": 0,
        "external_teacher_used": 0,
        "self_teacher_used": 0,
        "teacher_logits_used": 0,
        "distillation_used": 0,
        "geometry_loss_used": 0,
        "sampler_changed": 0,
        "class_weight_used": 0,
        "uses_loss_backward": 0,
        "output_target_used_for_update_direction_only": 1,
        "output_target_used_as_loss": 0,
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    write_csv_rows(out_dir / "contract_audit_v9212.csv", contract_rows)
    p0_rows = _p0_source_recap(prev_dir)
    p1_rows = _p1_control_equivalence_autopsy(prev_dir)
    write_csv_rows(out_dir / "p0_v9211_reproduction.csv", p0_rows)
    write_csv_rows(out_dir / "p1_control_equivalence_autopsy.csv", p1_rows)
    p0_pass = _to_int(p0_rows[0].get("p0_reproduction_pass"))
    p1_pass = int(all(_to_int(r.get("p1_autopsy_pass")) == 1 for r in p1_rows if str(r.get("status", "")) != "not_run"))

    if p0_pass and p1_pass:
        p2_rows, target_trace, branch_trace = _run_p2_output_direction_factory(args, device)
    else:
        reason = "P0_or_P1_failed"
        p2_rows = [_not_run("P2_OUTPUT_SPACE_DIRECTION_FACTORY", "p2_output_space_direction_factory.csv", reason)]
        target_trace = [_not_run("OUTPUT_TARGET_TRACE", "output_target_trace_v9212.csv", reason)]
        branch_trace = [_not_run("PAIRED_REPLAY_BRANCH_TRACE", "paired_replay_branch_trace_v9212.csv", reason)]
    write_csv_rows(out_dir / "p2_output_space_direction_factory.csv", p2_rows)
    write_csv_rows(out_dir / "output_target_trace_v9212.csv", target_trace)
    write_csv_rows(out_dir / "paired_replay_branch_trace_v9212.csv", branch_trace)
    measured_p2 = [r for r in p2_rows if str(r.get("status", "")) != "not_run"]
    p2_summary = _summarize_p2(measured_p2) if measured_p2 else [_not_run("P2_OUTPUT_DIRECTION_FACTORY_SUMMARY", "p3_event_controller_calibration.csv", "P2_not_measured")]
    write_csv_rows(out_dir / "p3_event_controller_calibration.csv", p2_summary)

    p2_pass_rows = [r for r in p2_summary if _to_int(r.get("p2_output_direction_pass")) == 1]
    best = sorted(p2_pass_rows, key=lambda r: (-_to_float(r.get("task_safe_rate")), -_to_float(r.get("output_displacement_ratio"))))[0] if p2_pass_rows else None

    downstream_reason = "P2_output_direction_factory_failed"
    if not p0_pass:
        downstream_reason = "P0_v9211_reproduction_failed"
    elif not p1_pass:
        downstream_reason = "P1_control_equivalence_autopsy_failed"
    elif best:
        downstream_reason = "P2_passed_but_P4_P8_not_executed_in_this_runner"
    for stage, name in [
        ("P4_PAIRED_REPLAY_CAUSAL_CONFIRMATION", "p4_paired_replay_causal_confirmation.csv"),
        ("P5_SHORT_RUN_MULTISTEP_VALIDATION", "p5_short_run_multistep_validation.csv"),
        ("P6_FULL_FUNCTIONAL_REENTRY_10SEED", "p6_full_functional_reentry_10seed.csv"),
        ("P7_NOISE_ROBUSTNESS_SIGNAL_VALIDATION", "p7_noise_robustness_signal_validation.csv"),
        ("P8_STRONG_BASELINE_EXTERNAL_READY", "p8_strong_baseline_external_ready.csv"),
        ("FUNCTIONAL_EVENT_TRACE", "functional_event_trace_v9212.csv"),
    ]:
        write_csv_rows(out_dir / name, [_not_run(stage, name, downstream_reason)])

    if not p0_pass:
        route = "R5-ControlEquivalentAgain"
        failure_code = "F3_v9211_reproduction_unstable"
        primary = "v9211_control_equivalent_result_not_reproduced"
        next_required = "repair_measurement_before_direction_reconstruction"
    elif not p1_pass:
        route = "R5-ControlEquivalentAgain"
        failure_code = "F19_artifact_missing"
        primary = "control_equivalence_autopsy_incomplete"
        next_required = "complete_autopsy_before_new_direction_search"
    elif best:
        route = "R1-OutputDirectionCausalityEstablished"
        failure_code = "F19_artifact_missing"
        primary = "P2_output_direction_passed_but_later_stages_not_executed"
        next_required = "open_P4_P5_for_output_direction_survivor"
    else:
        accepted_real = [
            r for r in measured_p2
            if r.get("branch") == "RealFunctional" and _to_int(r.get("event_accepted")) == 1
        ]
        projection_fail = bool(accepted_real) and max(_to_float(r.get("output_displacement_ratio_vs_adamw_logits")) for r in accepted_real) < 0.05
        coverage_values = [_to_float(r.get("coverage")) for r in p2_summary if str(r.get("status", "")) != "not_run"]
        sparse_ok = any(0.03 <= c <= 0.15 for c in coverage_values)
        route = "R6-ProjectionNeutralized" if projection_fail else ("R7-EventControllerDegenerate" if not sparse_ok else "R5-ControlEquivalentAgain")
        failure_code = "F7_output_effect_too_small" if projection_fail else ("F9_event_controller_no_sparse_pass" if not sparse_ok else "F10_paired_replay_causality_fail")
        primary = "output_space_directions_failed_safety_effect_or_control_gate"
        next_required = "return_to_primitive_or_define_nonlocal_output_functional_target"

    failure_rows = [{
        "stage": "P2",
        "failure_code": failure_code,
        "route": route,
        "reason": primary,
        "p0_pass": p0_pass,
        "p1_pass": p1_pass,
        "p2_output_direction_pass_count": len(p2_pass_rows),
        "fake_data_used": 0,
        "proxy_row_used": 0,
        "cpu_offload_used": 0,
    }]
    write_csv_rows(out_dir / "failure_table.csv", failure_rows)

    _write_svg(out_dir / "figures" / "p0_v9211_reproduction_dashboard.svg", "P0 v9.2.11 Reproduction", f"p0_pass={p0_pass}")
    _write_svg(out_dir / "figures" / "p1_failure_mechanism_bar.svg", "P1 Control Equivalence Autopsy", f"p1_pass={p1_pass}")
    _write_svg(out_dir / "figures" / "p2_direction_factory_pareto.svg", "P2 Output Direction Factory", f"route={route}, pass_groups={len(p2_pass_rows)}")
    _write_svg(out_dir / "figures" / "p2_real_vs_control_mechanism_gain.svg", "P2 Real vs Controls", f"measured_rows={len(measured_p2)}")
    _write_svg(out_dir / "figures" / "p3_event_coverage_calibration.svg", "P3 Event Calibration", f"target_coverage={args.p2_target_coverage}")

    route_json = {
        "route": route,
        "base_candidate": "LQ-t2-h256",
        "best_functional_candidate": best.get("functional_candidate", "") if best else "",
        "best_output_target": best.get("target_id", "") if best else "",
        "best_parameter_subspace": best.get("subspace_id", "") if best else "",
        "best_solver": best.get("solver_id", "") if best else "",
        "best_event_controller": best.get("event_type", "") if best else "",
        "p0_reproduction_pass": p0_pass,
        "p1_control_equivalence_autopsy_pass": p1_pass,
        "p2_output_direction_pass": int(bool(best)),
        "p2_output_direction_pass_count": len(p2_pass_rows),
        "p4_paired_replay_pass": 0,
        "p5_short_run_pass": 0,
        "p6_full_reentry_pass": 0,
        "functional_task_safe": int(bool(best)),
        "functional_mechanism_pass": int(bool(best)),
        "functional_control_pass": int(bool(best)),
        "functional_system_broad_pass": 0,
        "functional_system_strong_pass": 0,
        "functional_kmnist_repair_pass": 0,
        "noise_robustness_pass": 0,
        "strong_baseline_challenge_pass": 0,
        "external_ready": 0,
        "primary_blocker": primary,
        "next_required_implementation": next_required,
        "success_v9212_output_direction": int(bool(best)),
        "success_v9212_event_causality": 0,
        "success_v9212_full_functional": 0,
        "success_v9212_external_ready": 0,
    }
    write_json(out_dir / "route_decision.json", route_json)
    write_json(out_dir / "aggregate_decision.json", route_json)

    audit_targets = [
        out_dir / "contract_audit_v9212.csv",
        out_dir / "p0_v9211_reproduction.csv",
        out_dir / "p1_control_equivalence_autopsy.csv",
        out_dir / "p2_output_space_direction_factory.csv",
        out_dir / "p3_event_controller_calibration.csv",
        out_dir / "p4_paired_replay_causal_confirmation.csv",
        out_dir / "p5_short_run_multistep_validation.csv",
        out_dir / "p6_full_functional_reentry_10seed.csv",
        out_dir / "p7_noise_robustness_signal_validation.csv",
        out_dir / "p8_strong_baseline_external_ready.csv",
        out_dir / "functional_event_trace_v9212.csv",
        out_dir / "output_target_trace_v9212.csv",
        out_dir / "paired_replay_branch_trace_v9212.csv",
        out_dir / "failure_table.csv",
    ]
    audit = audit_no_fake(audit_targets)
    write_csv_rows(out_dir / "v9212_provenance_audit.csv", [{"stage": "NO_FAKE_AUDIT", "route": route, **audit, "fake_data_used": int(audit["fake_data_used"]), "proxy_row_used": int(audit["proxy_row_used"]), "cpu_offload_used": int(audit["cpu_offload_used"])}])
    route_json.update({"no_fake": bool(audit["no_fake"]), "no_proxy": bool(audit["no_proxy"]), "rows_checked": int(audit["rows_checked"])})
    write_json(out_dir / "route_decision.json", route_json)
    write_json(out_dir / "aggregate_decision.json", route_json)
    write_csv_rows(out_dir / "artifact_hashes.csv", artifact_hash_rows([
        PLAN_PATH,
        SCRIPT_PATH,
        ROOT / "dgkan" / "functional" / "lq_output_space_functional.py",
        ROOT / "dgkan" / "functional" / "lq_functional_predictor.py",
        ROOT / "dgkan" / "functional" / "snr_gated_lq.py",
        ROOT / "dgkan" / "models" / "fc_purekan_lq.py",
        out_dir / "route_decision.json",
        *audit_targets,
        out_dir / "v9212_provenance_audit.csv",
    ], root=ROOT))
    print(json.dumps(route_json, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
