#!/usr/bin/env python
"""v12.34.2 basis-specific loss-agnostic functional repair P3 audit."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v1223_failclosed_explore_open2_functional_rebuild as v1223  # noqa: E402
from experiments.run_v1231_basis_kernel_workspace import linec_metrics, make_adamw, make_basis_model  # noqa: E402
from dgkan.diagnostics.basis_workspace import V12342_BASIS_CANDIDATES, family_telemetry_metrics, finite_mean, fnum, parse_csv, parse_ints  # noqa: E402
from dgkan.training.eval import classification_basic  # noqa: E402


FAMILY_FUNCTIONALS = {
    "D-RAT": [
        "B-RAT1-TangentTrustRegionNoCE",
        "B-RAT2-DenSlopeGuardNoCE",
        "B-RAT3-GroupDiversityTransport",
        "B-RAT4-ReadoutRationalDecouple",
        "B-RAT5-LineCStableTangentMix",
    ],
    "D-CHE": [
        "B-CHE1-DegreeEnergyDamping",
        "B-CHE2-HighDegreeLateEnable",
        "B-CHE3-DegreeTangentTrustRegion",
        "B-CHE4-RoleDegreeEnergyCap",
        "B-CHE5-HighDegreeNoiseLeakVeto",
    ],
    "D-FOU": [
        "B-FOU1-FrequencyBandDamping",
        "B-FOU2-PhaseStabilityCorrection",
        "B-FOU3-LowFreqSignalTransport",
        "B-FOU4-HighFreqNoiseLeakVeto",
        "B-FOU5-LateEnableHighFreqResidual",
    ],
    "D-RBF": [
        "B-RBF1-CenterOccupancyRebalance",
        "B-RBF2-WidthConditionGuard",
        "B-RBF3-OOGBoundaryRepair",
        "B-RBF4-LocalCurvatureSmoothCompensated",
        "B-RBF5-ActiveCenterDiversityTransport",
    ],
    "D-WAV": [
        "B-WAV1-ScaleEnergyBalance",
        "B-WAV2-LocalSupportOccupancyRepair",
        "B-WAV3-LocalTailCoverageGuard",
        "B-WAV4-SupportOverlapEntropyGuard",
        "B-WAV5-ScaleDiversityTransport",
    ],
}

CONTROL_IDS = [
    "C0-BasisAdamW",
    "C1-NoOpMatchedOverhead",
    "C2-RandomMatchedNorm",
    "C3-AdamWParallelDirection",
    "C4-SNROnlyAudit",
]


def centered_cov(x: torch.Tensor) -> torch.Tensor:
    z = x.float() - x.float().mean(dim=0, keepdim=True)
    return (z.T @ z) / max(1, int(z.shape[0]) - 1)


def offdiag_energy(cov: torch.Tensor) -> torch.Tensor:
    return (cov - torch.diag(torch.diagonal(cov))).square().mean()


def parameter_norm(model: torch.nn.Module) -> float:
    total = torch.zeros((), device=next(model.parameters()).device)
    for p in model.parameters():
        if p.requires_grad:
            total = total + p.detach().float().square().sum()
    return float(total.sqrt().item())


def grad_delta(model: torch.nn.Module, objective: torch.Tensor) -> list[torch.Tensor]:
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(objective, params, retain_graph=False, create_graph=False, allow_unused=True)
    return [torch.zeros_like(p) if g is None else -g.detach() for p, g in zip(params, grads)]


def delta_norm(delta: list[torch.Tensor]) -> float:
    if not delta:
        return 0.0
    total = torch.zeros((), device=delta[0].device)
    for d in delta:
        total = total + d.float().square().sum()
    return float(total.sqrt().item())


def scale_delta(delta: list[torch.Tensor], target_norm: float) -> list[torch.Tensor]:
    norm = delta_norm(delta)
    if not math.isfinite(norm) or norm <= 0.0:
        return [torch.zeros_like(d) for d in delta]
    return [d * (float(target_norm) / max(norm, 1.0e-12)) for d in delta]


def random_delta_like(delta: list[torch.Tensor], target_norm: float, seed: int) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for idx, d in enumerate(delta):
        gen = torch.Generator(device=d.device).manual_seed(int(seed) + 104729 * idx)
        out.append(torch.randn(d.shape, device=d.device, generator=gen, dtype=d.dtype))
    return scale_delta(out, target_norm)


def apply_delta(model: torch.nn.Module, delta: list[torch.Tensor]) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    with torch.no_grad():
        for p, d in zip(params, delta):
            p.add_(d)


def basis_functional_objective(model: torch.nn.Module, x: torch.Tensor, family: str, functional_id: str, seed: int) -> torch.Tensor:
    logits = model(x)
    gen = torch.Generator(device=x.device).manual_seed(int(seed) + 123)
    jitter = torch.randn(x.shape, device=x.device, generator=gen, dtype=x.dtype) * 0.015
    logits_jitter = model((x + jitter).clamp(0.0, 1.0))
    response = logits_jitter - logits.detach()
    cov = centered_cov(logits)
    response_cov = centered_cov(response)
    probs = logits.softmax(dim=1)
    entropy = -(probs.clamp_min(1.0e-8).log() * probs).sum(dim=1)
    logit_norm = logits.float().norm(dim=1)
    response_norm = response.float().norm(dim=1)
    cov_balance = offdiag_energy(cov) + 0.25 * offdiag_energy(response_cov)
    entropy_guard = entropy.var(unbiased=False) + 0.01 * (entropy.mean() - math.log(max(1, int(logits.shape[1])))).square()
    norm_guard = logit_norm.var(unbiased=False) + 0.25 * response_norm.var(unbiased=False)

    if functional_id.endswith("TangentTrustRegionNoCE") or "Tangent" in functional_id:
        return 0.04 * response_norm.square().mean() + 0.02 * cov_balance + 0.00025 * logits.float().square().mean()
    if "DenSlope" in functional_id or "Condition" in functional_id or "Width" in functional_id or "Scale" in functional_id:
        return 0.03 * norm_guard + 0.02 * cov_balance + 0.001 * entropy_guard
    if "Diversity" in functional_id or "Occupancy" in functional_id or "Energy" in functional_id:
        diag = torch.diagonal(cov).clamp_min(1.0e-8)
        return 0.03 * ((diag / diag.mean().clamp_min(1.0e-8) - 1.0).square().mean()) + 0.02 * offdiag_energy(cov)
    if "Readout" in functional_id or "SignalTransport" in functional_id or "LowFreq" in functional_id:
        return 0.02 * response_cov.square().mean() + 0.01 * cov_balance + 0.00025 * logits.float().square().mean()
    if "LeakVeto" in functional_id or "Tail" in functional_id or "Overlap" in functional_id or "Entropy" in functional_id:
        return 0.02 * entropy_guard + 0.02 * response_norm.square().mean() + 0.00025 * logits.float().square().mean()
    # Family fallback remains label-free and is recorded through functional_id.
    return 0.02 * cov_balance + 0.01 * entropy_guard + 0.00025 * logits.float().square().mean()


def workspace_pass_map(path: Path) -> dict[tuple[str, str, int], int]:
    rows = v1223.read_csv_rows(path) if path.exists() else []
    out: dict[tuple[str, str, int], int] = {}
    for row in rows:
        out[(str(row.get("candidate_id", "")), str(row.get("dataset", "")), int(float(row.get("seed", 0) or 0)))] = int(float(row.get("workspace_gate_pass", 0) or 0))
    return out


def select_base_candidates(args: argparse.Namespace) -> list[str]:
    requested = parse_csv(args.base_candidates)
    if requested:
        return requested
    rows = v1223.read_csv_rows(Path(args.auc_summary_csv)) if args.auc_summary_csv and Path(args.auc_summary_csv).exists() else []
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cid = str(row.get("candidate_id", ""))
        fam = str(row.get("family", "")) or V12342_BASIS_CANDIDATES.get(cid, V12342_BASIS_CANDIDATES[next(iter(V12342_BASIS_CANDIDATES))]).family
        if int(float(row.get("executed_rows", 0) or 0)) <= 0:
            continue
        if fnum(row.get("mean_delta_vs_MLP"), -999.0) < -0.05:
            continue
        if fnum(row.get("worst_delta_vs_MLP"), -999.0) < -0.10:
            continue
        if fnum(row.get("max_CEp99_delta_vs_MLP"), 999.0) > 5.0:
            continue
        by_family.setdefault(fam, []).append(row)
    selected: list[str] = []
    for fam, group in sorted(by_family.items()):
        group = sorted(
            group,
            key=lambda r: (
                fnum(r.get("mean_delta_vs_MLP"), -999.0),
                -fnum(r.get("max_AUC_time_ratio_vs_MLP"), 999.0),
                -fnum(r.get("max_CEp99_delta_vs_MLP"), 999.0),
            ),
            reverse=True,
        )
        selected.extend(str(r.get("candidate_id", "")) for r in group[: int(args.max_base_candidates_per_family)])
    return [s for s in selected if s in V12342_BASIS_CANDIDATES]


def train_branch(
    args: argparse.Namespace,
    cand_id: str,
    functional_id: str,
    branch_id: str,
    dataset: str,
    seed: int,
    input_dim: int,
    output_dim: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    cand = V12342_BASIS_CANDIDATES[cand_id]
    model, _spec = make_basis_model(cand.method_id, int(input_dim), int(output_dim), x_train, device, int(seed) + int(args.train_seed_base))
    opt = make_adamw(args, [p for p in model.parameters() if p.requires_grad])
    gen = torch.Generator(device=device).manual_seed(int(seed) + int(args.train_seed_base) + 43)
    times: list[float] = []
    func_times: list[float] = []
    functional_events = 0
    for epoch in range(int(args.window_epochs)):
        perm = torch.randperm(int(x_train.shape[0]), device=device, generator=gen)
        for off in range(0, int(x_train.shape[0]), int(args.batch_size)):
            idx = perm[off:off + int(args.batch_size)]
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000.0)
        if branch_id == "C0-BasisAdamW":
            continue
        xb = x_train[: min(int(args.functional_batch), int(x_train.shape[0]))]
        yb = y_train[: int(xb.shape[0])]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ft0 = time.perf_counter()
        objective = basis_functional_objective(model, xb, cand.family, functional_id, int(seed) + int(args.train_seed_base) + epoch)
        source_delta = grad_delta(model, objective)
        target_norm = max(1.0e-12, float(args.functional_norm_frac) * parameter_norm(model))
        source_delta = scale_delta(source_delta, target_norm)
        if branch_id == functional_id:
            delta = source_delta
        elif branch_id == "C1-NoOpMatchedOverhead":
            delta = [torch.zeros_like(d) for d in source_delta]
        elif branch_id == "C2-RandomMatchedNorm":
            delta = random_delta_like(source_delta, target_norm, int(seed) + epoch + 991)
        elif branch_id == "C3-AdamWParallelDirection":
            ce = F.cross_entropy(model(xb), yb)
            delta = scale_delta(grad_delta(model, ce), target_norm)
        elif branch_id == "C4-SNROnlyAudit":
            logits = model(xb)
            probs = logits.softmax(dim=1)
            snr_obj = -logits.float().var(dim=0).mean() + 0.05 * probs.float().var(dim=0).mean()
            delta = scale_delta(grad_delta(model, snr_obj), target_norm)
        else:
            raise KeyError(branch_id)
        apply_delta(model, delta)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        func_times.append((time.perf_counter() - ft0) * 1000.0)
        functional_events += 1
    ev = classification_basic(model, x_val, y_val)
    q90 = float(torch.tensor(times, device=device).quantile(0.90).item()) if times else float("nan")
    fq90 = float(torch.tensor(func_times, device=device).quantile(0.90).item()) if func_times else 0.0
    telemetry = family_telemetry_metrics(model, x_train[: min(int(args.functional_batch), int(x_train.shape[0]))], cand.family, int(seed) + 34200)
    return {
        "stage": "V12342_BASIS_FUNCTIONAL_BRANCH",
        "family": cand.family,
        "base_candidate_id": cand_id,
        "functional_candidate_id": functional_id,
        "branch_id": branch_id,
        "dataset": dataset,
        "seed": int(seed),
        "window": int(args.window_epochs),
        "lambda": float(args.functional_norm_frac),
        "adamw_foreach": str(getattr(args, "adamw_foreach", "auto")),
        "acc": ev["acc"],
        "NLL": ev["NLL"],
        "ECE": ev["ECE"],
        "CEp99": ev["CEp99"],
        "step_time_q90_ms": q90,
        "functional_time_q90_ms": fq90,
        "AUC_time_proxy": ev["NLL"] * (q90 + fq90),
        "functional_events": functional_events,
        "loss_agnostic_direction": int(branch_id != "C3-AdamWParallelDirection"),
        "label_used_for_direction": int(branch_id == "C3-AdamWParallelDirection"),
        "ce_vector_used_for_direction": int(branch_id == "C3-AdamWParallelDirection"),
        "validation_used_for_commit": 0,
        "test_used_for_commit": 0,
        "query_batch_used_for_direction": 0,
        "linec_hard_target_used_for_direction": 0,
        "dataset_name_branch": 0,
        "promotion_allowed": 0,
        "no_fake": 1,
        "_model": model,
        "_telemetry": telemetry,
    }


def run() -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--artifact-prefix", default="v12342_basis_functional")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--base-candidates", default="")
    ap.add_argument("--auc-summary-csv", default="")
    ap.add_argument("--workspace-csv", default="")
    ap.add_argument("--max-base-candidates-per-family", type=int, default=1)
    ap.add_argument("--functional-candidates", default="")
    ap.add_argument("--train-size", type=int, default=512)
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--window-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--functional-batch", type=int, default=128)
    ap.add_argument("--functional-norm-frac", type=float, default=5.0e-5)
    ap.add_argument("--lr", type=float, default=0.0015)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--adamw-foreach", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--train-seed-base", type=int, default=12342000)
    ap.add_argument("--linec-batch-size", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=8)
    ap.add_argument("--linec-seeds", default="12349500,12350600,12351600")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    v1223.ensure_dir(out_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("v12.34.2 basis functional repair requires CUDA")
    torch.cuda.set_device(device)

    datasets = [v1223.v120._canonical_dataset(d) for d in parse_csv(args.datasets)]
    seeds = parse_ints(args.seeds)
    base_candidates = select_base_candidates(args)
    wpass = workspace_pass_map(Path(args.workspace_csv)) if args.workspace_csv else {}

    p3_rows: list[dict[str, Any]] = []
    p4_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    linec_rows: list[dict[str, Any]] = []

    for base_id in base_candidates:
        cand = V12342_BASIS_CANDIDATES[base_id]
        funcs = parse_csv(args.functional_candidates) or FAMILY_FUNCTIONALS.get(cand.family, [])
        funcs = [f for f in funcs if f.startswith(f"B-{cand.family.split('-')[-1]}") or f in FAMILY_FUNCTIONALS.get(cand.family, [])]
        for dataset in datasets:
            for seed in seeds:
                if wpass and not int(wpass.get((base_id, dataset, int(seed)), 0)):
                    for fid in funcs:
                        p3_rows.append({
                            "stage": "V12342_BASIS_FUNCTIONAL_P3",
                            "family": cand.family,
                            "base_candidate_id": base_id,
                            "functional_candidate_id": fid,
                            "dataset": dataset,
                            "seed": int(seed),
                            "executed": 0,
                            "skip_reason": "workspace_gate_fail_for_dataset_seed",
                            "p3_pass": 0,
                            "promotion_allowed": 0,
                            "no_fake": 1,
                        })
                    continue
                load_args = argparse.Namespace(data_root=args.data_root, no_download=bool(args.no_download), seed=int(seed))
                data = v1223.v120._load_vision_split(load_args, dataset, train_size=int(args.train_size), val_size=int(args.val_size), test_size=int(args.val_size))
                x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, _xt, _yt, input_dim, output_dim, _protocol = data
                x_train = x_train_cpu.to(device=device, dtype=torch.float32)
                y_train = y_train_cpu.to(device=device)
                x_val = x_val_cpu.to(device=device, dtype=torch.float32)
                y_val = y_val_cpu.to(device=device)
                for fid in funcs:
                    branches = [fid, *CONTROL_IDS]
                    results: dict[str, dict[str, Any]] = {}
                    models: dict[str, torch.nn.Module] = {}
                    for branch in branches:
                        row = train_branch(args, base_id, fid, branch, dataset, int(seed), int(input_dim), int(output_dim), x_train, y_train, x_val, y_val, device)
                        model = row.pop("_model")
                        telemetry = row.pop("_telemetry")
                        row.update({f"telemetry_{k}": v for k, v in telemetry.items() if not isinstance(v, str)})
                        branch_rows.append(row)
                        results[branch] = row
                        if branch in {fid, "C1-NoOpMatchedOverhead"}:
                            models[branch] = model
                        else:
                            del model
                        torch.cuda.empty_cache()
                    b = min(int(args.linec_batch_size), int(x_train.shape[0]), int(x_val.shape[0]))
                    for branch, model in list(models.items()):
                        for ls in parse_ints(args.linec_seeds):
                            lm = linec_metrics(model, x_train[:b], y_train[:b], x_val[:b], y_val[:b], int(ls), int(args.linec_sketch_dim), float(args.lr), float(args.weight_decay))
                            linec_rows.append({
                                "stage": "V12342_BASIS_FUNCTIONAL_LINEC",
                                "family": cand.family,
                                "base_candidate_id": base_id,
                                "functional_candidate_id": fid,
                                "branch_id": branch,
                                "dataset": dataset,
                                "seed": int(seed),
                                "linec_seed": int(ls),
                                **lm,
                                "label_used_for_audit_only": 1,
                                "linec_hard_target_used_for_direction": 0,
                                "promotion_allowed": 0,
                                "no_fake": 1,
                            })
                        del model
                    source = results[fid]
                    noop = results["C1-NoOpMatchedOverhead"]
                    rand = results["C2-RandomMatchedNorm"]
                    adamw = results["C3-AdamWParallelDirection"]
                    snr = results["C4-SNROnlyAudit"]
                    control_group = [results[c] for c in CONTROL_IDS]
                    best_control = max(control_group, key=lambda r: fnum(r.get("acc"), -999.0))
                    src_lc = [r for r in linec_rows if r.get("base_candidate_id") == base_id and r.get("functional_candidate_id") == fid and r.get("dataset") == dataset and int(r.get("seed")) == int(seed) and r.get("branch_id") == fid]
                    noop_lc = [r for r in linec_rows if r.get("base_candidate_id") == base_id and r.get("functional_candidate_id") == fid and r.get("dataset") == dataset and int(r.get("seed")) == int(seed) and r.get("branch_id") == "C1-NoOpMatchedOverhead"]
                    coupling_delta = finite_mean([fnum(s.get("CouplingR2"), 0.0) - fnum(n.get("CouplingR2"), 0.0) for s, n in zip(src_lc, noop_lc)])
                    noise_delta = finite_mean([fnum(s.get("NoiseSignalLeak"), 0.0) - fnum(n.get("NoiseSignalLeak"), 0.0) for s, n in zip(src_lc, noop_lc)])
                    reservoir_delta = finite_mean([fnum(s.get("RealSignalReservoirRatio"), 0.0) - fnum(n.get("RealSignalReservoirRatio"), 0.0) for s, n in zip(src_lc, noop_lc)])
                    linec_pass = sum(
                        int(
                            fnum(s.get("CouplingR2"), -999.0) >= fnum(n.get("CouplingR2"), 0.0)
                            and fnum(s.get("NoiseSignalLeak"), 999.0) <= fnum(n.get("NoiseSignalLeak"), 0.0) + 0.02
                            and fnum(s.get("RealSignalReservoirRatio"), 999.0) <= fnum(n.get("RealSignalReservoirRatio"), 0.0) + 0.03
                        )
                        for s, n in zip(src_lc, noop_lc)
                    )
                    source_vs_noop = fnum(source.get("acc"), -999.0) - fnum(noop.get("acc"), -999.0)
                    source_vs_random = fnum(source.get("acc"), -999.0) - fnum(rand.get("acc"), -999.0)
                    source_vs_adamw = fnum(source.get("acc"), -999.0) - fnum(adamw.get("acc"), -999.0)
                    source_vs_snr = fnum(source.get("acc"), -999.0) - fnum(snr.get("acc"), -999.0)
                    source_vs_best = fnum(source.get("acc"), -999.0) - fnum(best_control.get("acc"), -999.0)
                    ce_delta = fnum(source.get("CEp99"), 999.0) - fnum(noop.get("CEp99"), 999.0)
                    nll_delta = fnum(source.get("NLL"), 999.0) - fnum(noop.get("NLL"), 999.0)
                    ece_delta = fnum(source.get("ECE"), 999.0) - fnum(noop.get("ECE"), 999.0)
                    auc_delta = fnum(source.get("AUC_time_proxy"), 999.0) - fnum(noop.get("AUC_time_proxy"), 999.0)
                    p3_pass = int(
                        source_vs_noop >= 0.0
                        and source_vs_best >= 0.0
                        and auc_delta <= 0.0
                        and ce_delta <= 0.05
                        and nll_delta <= 0.02
                        and ece_delta <= 0.02
                        and coupling_delta >= 0.02
                        and noise_delta <= -0.01
                        and reservoir_delta <= -0.01
                        and linec_pass >= math.ceil(len(src_lc) / 2.0)
                    )
                    fail_reasons = []
                    if source_vs_noop < 0.0:
                        fail_reasons.append("source_harms_noop")
                    if source_vs_best < 0.0:
                        fail_reasons.append("source_not_best_control")
                    if auc_delta > 0.0:
                        fail_reasons.append("auc_time_proxy_worse")
                    if ce_delta > 0.05:
                        fail_reasons.append("CEp99_worse")
                    if nll_delta > 0.02:
                        fail_reasons.append("NLL_worse")
                    if ece_delta > 0.02:
                        fail_reasons.append("ECE_worse")
                    if coupling_delta < 0.02:
                        fail_reasons.append("CouplingR2_no_gain")
                    if noise_delta > -0.01:
                        fail_reasons.append("NoiseSignalLeak_no_drop")
                    if reservoir_delta > -0.01:
                        fail_reasons.append("RealSignalReservoirRatio_no_drop")
                    if linec_pass < math.ceil(len(src_lc) / 2.0):
                        fail_reasons.append("LineC_not_preserved")
                    p3_rows.append({
                        "stage": "V12342_BASIS_FUNCTIONAL_P3",
                        "family": cand.family,
                        "base_candidate_id": base_id,
                        "functional_candidate_id": fid,
                        "dataset": dataset,
                        "seed": int(seed),
                        "window": int(args.window_epochs),
                        "lambda": float(args.functional_norm_frac),
                        "executed": 1,
                        "source_vs_noop": source_vs_noop,
                        "source_vs_random": source_vs_random,
                        "source_vs_adamwparallel": source_vs_adamw,
                        "source_vs_snr": source_vs_snr,
                        "source_vs_best_control": source_vs_best,
                        "best_control_id": best_control.get("branch_id", ""),
                        "CouplingR2_delta": coupling_delta,
                        "NoiseSignalLeak_delta": noise_delta,
                        "RealSignalReservoirRatio_delta": reservoir_delta,
                        "AUC_time_delta_proxy": auc_delta,
                        "CEp99_delta_audit": ce_delta,
                        "NLL_delta_audit": nll_delta,
                        "ECE_delta_audit": ece_delta,
                        "telemetry_delta_summary": "branch telemetry recorded in controls artifact",
                        "control_gap": source_vs_best,
                        "LineC_pass_count": linec_pass,
                        "LineC_total": len(src_lc),
                        "loss_agnostic_direction": 1,
                        "label_used_for_direction": 0,
                        "ce_vector_used_for_direction": 0,
                        "validation_used_for_commit": 0,
                        "query_batch_used_for_direction": 0,
                        "linec_hard_target_used_for_direction": 0,
                        "dataset_name_branch": 0,
                        "p3_pass": p3_pass,
                        "p3_fail_reason": "pass" if p3_pass else ";".join(fail_reasons),
                        "promotion_allowed": 0,
                        "no_fake": 1,
                    })
                    p4_rows.append({
                        "stage": "V12342_BASIS_FUNCTIONAL_P4",
                        "family": cand.family,
                        "base_candidate_id": base_id,
                        "functional_candidate_id": fid,
                        "dataset": dataset,
                        "seed": int(seed),
                        "executed": 0,
                        "skip_reason": "P3_failed" if not p3_pass else "P3_passed_but_P4_runner_not_invoked_in_this_artifact",
                        "p3_pass": p3_pass,
                        "p4_pass": 0,
                        "promotion_allowed": 0,
                        "no_fake": 1,
                    })
                    torch.cuda.empty_cache()

    p3_path = out_dir / f"{args.artifact_prefix}_p3.csv"
    p4_path = out_dir / f"{args.artifact_prefix}_p4.csv"
    branch_path = out_dir / f"{args.artifact_prefix}_branches.csv"
    linec_path = out_dir / f"{args.artifact_prefix}_linec.csv"
    v1223.write_csv_rows(p3_path, p3_rows)
    v1223.write_csv_rows(p4_path, p4_rows)
    v1223.write_csv_rows(branch_path, branch_rows)
    v1223.write_csv_rows(linec_path, linec_rows)
    result = {
        "stage": "V12342_BASIS_FUNCTIONAL_REPAIR_AGGREGATE",
        "base_candidates": base_candidates,
        "p3_csv": v1223.rel(p3_path),
        "p4_csv": v1223.rel(p4_path),
        "branch_csv": v1223.rel(branch_path),
        "linec_csv": v1223.rel(linec_path),
        "p3_rows": len(p3_rows),
        "p3_executed_rows": sum(int(float(r.get("executed", 0) or 0)) for r in p3_rows),
        "p3_pass_rows": sum(int(float(r.get("p3_pass", 0) or 0)) for r in p3_rows),
        "p4_rows": len(p4_rows),
        "p4_pass_rows": sum(int(float(r.get("p4_pass", 0) or 0)) for r in p4_rows),
        "promotion_allowed": 0,
        "no_fake": 1,
    }
    route_path = out_dir / f"{args.artifact_prefix}_aggregate.json"
    v1223.write_json(route_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
