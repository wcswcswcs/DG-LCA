#!/usr/bin/env python3
"""v12.18 B320 label-free init ablation continuation.

This runner performs a real small-budget CUDA smoke ablation for the Line A
blocker found by v12.18: current B320 uses label-informed trainprobe init while
comparable label-free B320 artifacts are missing.  The output is deliberately
marked as smoke-only, not as an official replacement for the locked v12.14
anchor budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "experiments"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

import run_v120_good_geometry_battery as v120  # noqa: E402
import run_v124_multibasis_functional_dual as v124  # noqa: E402
import run_v126_lowerlevel_fhq_functional_geometry as v126  # noqa: E402
import run_v1283_b109_classic_family_functional_geometry as v1283  # noqa: E402
import run_v1252_efficiency_functional_manifold as v1252  # noqa: E402
from dgkan.kernels import fused_hinge_quadratic as fhq  # noqa: E402
from dgkan.models import fc_purekan_primitives as prim  # noqa: E402


DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "results"
    / "v12_18_b320_codeaudit_lossagnostic_functional"
    / "official_from_v1217_v1216_artifacts"
)
B320_ID = (
    "B320b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-"
    "signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-"
    "absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-"
    "hingeamp025-temp075"
)
EPS = 1.0e-12
MLP_SAME_PARAM_ID = "MLP-same-param-AdamW"
MLP_SAME_STEP_FLOP_ID = "MLP-same-step-FLOP-AdamW"
MLP_CONTROL_IDS = {MLP_SAME_PARAM_ID, MLP_SAME_STEP_FLOP_ID}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(p)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        val = float(value)
    except Exception:
        return default
    if math.isnan(val) or math.isinf(val):
        return default
    return val


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except Exception:
        return default


def finite_float_or_none(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        val = float(value)
    except Exception:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def q(values: Sequence[float], quantile: float, default: float = 0.0) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return default
    pos = (len(vals) - 1) * float(quantile)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if not fields:
        fields = ["stage", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("v12.18 label-free ablation requires CUDA; CPU-offload is not allowed")
        return torch.device("cuda:0")
    device = torch.device(name)
    if device.type != "cuda":
        raise RuntimeError("v12.18 label-free ablation requires CUDA; CPU-offload is not allowed")
    return device


def base_b320_spec(input_dim: int, output_dim: int) -> prim.PrimitiveSpec:
    _, budget = v124._param_budget(input_dim, output_dim)
    specs = {spec.candidate_id: spec for spec in prim.primitive_specs(budget, input_dim, output_dim)}
    if B320_ID not in specs:
        raise KeyError(f"B320 spec not found: {B320_ID}")
    return specs[B320_ID]


def strip_trainprobe_tokens(variant: str) -> str:
    pieces = []
    for token in str(variant).split("_"):
        low = token.lower()
        if low == "trainprobep":
            continue
        if low.startswith("signalbroad") or low.startswith("signalblock"):
            continue
        if low.startswith("trainprobedirect"):
            continue
        pieces.append(token)
    return "_".join(pieces)


def replace_variant_token(variant: str, old: str, new: str) -> str:
    tokens = [new if token == old else token for token in str(variant).split("_")]
    return "_".join(tokens)


def build_y_stats_for_mode(
    y_train: torch.Tensor,
    output_dim: int,
    seed: int,
    mode: str,
    x_train: torch.Tensor | None = None,
) -> torch.Tensor:
    mode = str(mode or "actual")
    if mode == "actual":
        return y_train
    gen = torch.Generator(device=y_train.device).manual_seed(int(seed) + 90217)
    if mode == "shuffled":
        return y_train[torch.randperm(int(y_train.numel()), device=y_train.device, generator=gen)]
    if mode == "random_balanced":
        base = torch.arange(int(y_train.numel()), device=y_train.device, dtype=torch.long) % max(1, int(output_dim))
        return base[torch.randperm(int(base.numel()), device=y_train.device, generator=gen)]
    if mode == "permuted_class_ids":
        offset = 1 + (int(seed) % max(1, int(output_dim) - 1))
        return (y_train + offset) % max(1, int(output_dim))
    if mode == "unsupervised_kmeans":
        if x_train is None:
            raise ValueError("unsupervised_kmeans y_stats_mode requires x_train")
        with torch.no_grad():
            x = x_train.to(device=y_train.device, dtype=torch.float32)
            x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True).clamp_min(1.0e-6)
            n = int(x.shape[0])
            k = max(1, int(output_dim))
            perm = torch.randperm(n, device=x.device, generator=gen)
            centroids = x[perm[:k]].clone()
            if int(centroids.shape[0]) < k:
                repeats = k - int(centroids.shape[0])
                centroids = torch.cat([centroids, x[:repeats].clone()], dim=0)
            assign = torch.zeros(n, device=x.device, dtype=torch.long)
            for _ in range(6):
                dist = torch.cdist(x.float(), centroids.float(), p=2)
                assign = dist.argmin(dim=1).to(dtype=torch.long)
                for idx in range(k):
                    mask = assign == idx
                    if bool(mask.any()):
                        centroids[idx] = x[mask].mean(dim=0)
            return assign
    if mode == "small_label_fraction":
        y_stats = torch.full_like(y_train, -1)
        frac = 0.10
        for class_idx in range(max(1, int(output_dim))):
            idx = torch.nonzero(y_train == int(class_idx), as_tuple=False).flatten()
            if int(idx.numel()) == 0:
                continue
            take = max(1, int(round(float(idx.numel()) * frac)))
            idx_perm = idx[torch.randperm(int(idx.numel()), device=y_train.device, generator=gen)[:take]]
            y_stats[idx_perm] = int(class_idx)
        return y_stats
    raise ValueError(f"unknown y_stats_mode: {mode}")


def ablation_specs(input_dim: int, output_dim: int) -> dict[str, dict[str, Any]]:
    base = base_b320_spec(input_dim, output_dim)
    stripped = strip_trainprobe_tokens(base.init_variant)
    lower_quad = replace_variant_token(
        replace_variant_token(stripped, "quadreadinit125", "quadreadinit075"),
        "identitytailquad030",
        "identitytailquad020",
    )
    lower_quad_strong = replace_variant_token(lower_quad, "quadreadinit075", "quadreadinit050")
    v1226_id = "V1226LabelFreeOnly"
    v1228_id = "V1228LabelFreeBaseRecovery"
    v1230_id = "V1230BasisFirstMonitor"
    return {
        "A0-labelInit": {"spec": base, "uses_y_for_stats": 1, "seed_offset": 0, "description": "current B320 with y_stats label-informed trainprobe init"},
        "A1-noYForStats": {"spec": base, "uses_y_for_stats": 0, "seed_offset": 0, "description": "current B320 variant with y_stats=None; trainprobe buffers label-capable but not applied"},
        "A-LF0-BaseNoProbe": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF0-BaseNoProbe", init_variant=stripped),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 label-free-only baseline: B320-like FHQ primitive with all train-probe tokens removed and y_stats disabled",
        },
        "A-LF1-OrthoBank": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF1-OrthoBank", init_variant=f"{stripped}_orthop"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 label-free orthogonal low-coherence projection bank",
        },
        "A-LF2-CovFrame": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF2-CovFrame", init_variant=f"{stripped}_covwhitenp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 label-free covariance-whitened frame from train-stream x only",
        },
        "A-LF3-AugStable": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF3-AugStable", init_variant=f"{stripped}_augstablep"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 label-free augmentation-stable projection frame",
        },
        "A-LF4-ResidualLowRank": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF4-ResidualLowRank", init_variant=f"{stripped}_orthop_reslowrankr008p001_rmsq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 label-free residual-on-baseline low-rank frame with RMSQ stabilization",
        },
        "A-LF5-RoleEnergyBalance": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF5-RoleEnergyBalance", init_variant=f"{stripped}_rolebalancedp_directreadinit125"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 label-free role-energy balanced frame with direct/quad energy normalization",
        },
        "A-LF6-CouplingAware": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF6-CouplingAware", init_variant=f"{stripped}_orthop_reslowrankr008p001_boundq_rmsq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 label-free coupling-aware residual frame using only unlabeled geometry proxies",
        },
        "A-LF7-EMACovAdapt": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF7-EMACovAdapt", init_variant=f"{stripped}_covadaptp_rmsq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 label-free EMA covariance adaptation frame",
        },
        "A-LF8-LowQuadOrtho": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF8-LowQuadOrtho", init_variant=f"{lower_quad}_orthop"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-5 label-free low-quad orthogonal frame; reduces quadratic reservoir load without trainprobe signal",
        },
        "A-LF9-LowQuadBoundQ": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF9-LowQuadBoundQ", init_variant=f"{lower_quad}_orthop_boundq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-5 label-free low-quad plus bounded-Q repair for LineC noise/reservoir tearing",
        },
        "A-LF10-LowQuadDirectRead": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF10-LowQuadDirectRead", init_variant=f"{lower_quad}_orthop_directreadinit125"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-5 label-free low-quad direct-read balance repair",
        },
        "A-LF11-MultiFrameBank": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF11-MultiFrameBank", init_variant=f"{stripped}_multiframebankp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-5 label-free multi-frame bank using PCA/SRHT/augmentation/local/low-frequency frames",
        },
        "A-LF12-MultiFrameDirect": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF12-MultiFrameDirect", init_variant=f"{stripped}_multiframebankp_directreadinit125"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-5 label-free multi-frame bank with direct-read balancing",
        },
        "A-LF13-ConvexFrameMix": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF13-ConvexFrameMix", init_variant=f"{stripped}_convexmixp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-5 label-free convex mixture of unlabeled frame families",
        },
        "A-LF14-SelfCondStopGrad": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF14-SelfCondStopGrad", init_variant=f"{stripped}_selfcondstopgradp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-5 label-free stop-grad self-conditioned activation covariance frame",
        },
        "A-LF15-RoleCondDirect": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF15-RoleCondDirect", init_variant=f"{stripped}_rolecondp_directreadinit125"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-5 label-free role-conditioned frame with direct-read balancing",
        },
        "A-LF16-FrozenBranchGain": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF16-FrozenBranchGain", init_variant=f"{stripped}_orthop_fixedp_classbranch_classgain_gainramp050_quadramp010"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-5 label-free frozen projector with branch/gain-only residual motion",
        },
        "A-LF17-LowQuadDirectResidual": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF17-LowQuadDirectResidual", init_variant=f"{lower_quad}_orthop_directreadinit125_reslowrankr008p001_rmsq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-6 repair: A-LF10 task anchor plus low-rank residual/RMSQ LineC stabilizer",
        },
        "A-LF18-LowQuadDirectBoundQ": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF18-LowQuadDirectBoundQ", init_variant=f"{lower_quad}_orthop_directreadinit125_boundq_rmsq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-6 repair: A-LF10 plus bounded-Q/RMSQ noise-reservoir guard",
        },
        "A-LF19-LowQuadDirectCovAdapt": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF19-LowQuadDirectCovAdapt", init_variant=f"{lower_quad}_orthop_directreadinit125_covadaptp_rmsq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-6 repair: A-LF10 plus EMA covariance adaptation and RMSQ",
        },
        "A-LF20-LowQuadRoleCondDirect": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF20-LowQuadRoleCondDirect", init_variant=f"{lower_quad}_rolecondp_directreadinit125_rmsq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-6 repair: low-quad role-conditioned direct-read geometry balancing",
        },
        "A-LF21-LowQuadConvexDirect": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF21-LowQuadConvexDirect", init_variant=f"{lower_quad}_convexmixp_directreadinit125"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-6 repair: low-quad convex unlabeled frame mix with direct-read balancing",
        },
        "A-LF22-LowQuadSelfCondDirect": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF22-LowQuadSelfCondDirect", init_variant=f"{lower_quad}_selfcondstopgradp_directreadinit125"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-6 repair: low-quad stop-grad self-conditioned frame with direct-read balancing",
        },
        "A-LF23-PCAOrthoMix": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF23-PCAOrthoMix", init_variant=f"{stripped}_pcaorthomixp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-7 label-free signal-frame estimator: PCA top components plus orthogonal residual frame from train-stream x",
        },
        "A-LF24-RandomCotangentStable": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF24-RandomCotangentStable", init_variant=f"{stripped}_randomcotangentstablep"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-7 label-free random cotangent stable response frame",
        },
        "A-LF25-BlockLocalAugStable": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF25-BlockLocalAugStable", init_variant=f"{stripped}_blocklocalaugstablep"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-7 label-free block-local augmentation-stable signal frame",
        },
        "A-LF26-AugTangentFrame": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF26-AugTangentFrame", init_variant=f"{stripped}_augtangentp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-7 label-free augmentation tangent signal frame from input perturbation geometry",
        },
        "A-LF27-DriftCotangentBank": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF27-DriftCotangentBank", init_variant=f"{stripped}_driftcotbankp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-7 label-free cotangent-bank drift frame computed at initialization from train-stream x order only",
        },
        "A-LF28-LowQuadPCAOrthoDirect": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF28-LowQuadPCAOrthoDirect", init_variant=f"{lower_quad}_pcaorthomixp_directreadinit125"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-7 label-free PCA/orthogonal signal frame with A-LF10 low-quad direct-read balance",
        },
        "A-LF29-LowQuadAugTangentDirect": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF29-LowQuadAugTangentDirect", init_variant=f"{lower_quad}_augtangentp_directreadinit125"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-7 label-free augmentation tangent frame with low-quad direct-read balance",
        },
        "A-LF30-LowQuadCotangentDirect": {
            "spec": replace(base, candidate_id=f"{v1226_id}::A-LF30-LowQuadCotangentDirect", init_variant=f"{lower_quad}_randomcotangentstablep_directreadinit125"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.26.1 depth-7 label-free cotangent stable frame with low-quad direct-read balance",
        },
        "A-S1a-MultiViewStableFrame": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S1a-MultiViewStableFrame", init_variant=f"{stripped}_multiviewstablep"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 S1 unlabeled multi-view stable frame from identity/translation/noise/dropout/low-pass views; no labels or CE",
        },
        "A-S1b-MultiViewStablePlusResidual": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S1b-MultiViewStablePlusResidual", init_variant=f"{stripped}_multiviewstablep_multiviewresidualp_reslowrankr008p002"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 S1 stable multi-view frame plus bounded unstable residual frame; no labels or CE",
        },
        "A-S1c-MultiViewBlockLocalStable": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S1c-MultiViewBlockLocalStable", init_variant=f"{stripped}_multiviewstablep_multiviewblockp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 S1 multi-view stable frame with block-local stable basis quota; no labels or CE",
        },
        "A-S1d-MultiViewLowQuadDirect": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S1d-MultiViewLowQuadDirect", init_variant=f"{lower_quad}_multiviewstablep_directreadinit125"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 S1 multi-view stable frame with low-quad direct-read balance; no labels or CE",
        },
        "A-S2a-TemporalDriftStableFrame": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S2a-TemporalDriftStableFrame", init_variant=f"{stripped}_temporaldriftstablep"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 S2 label-free random-probe temporal drift stable frame; no CE warmup, labels, or optimizer update",
        },
        "A-S2b-TemporalDriftResidualFrame": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S2b-TemporalDriftResidualFrame", init_variant=f"{stripped}_temporaldriftstablep_temporaldriftresidualp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 S2 label-free random-probe temporal drift frame with residual response channels; no CE warmup",
        },
        "A-S2c-TemporalDriftLowRankDirect": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S2c-TemporalDriftLowRankDirect", init_variant=f"{lower_quad}_temporaldriftstablep_temporaldriftlowrankp_directreadinit125"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 S2 temporal drift low-rank direct-read frame from label-free random probes",
        },
        "A-S3a-BlockLocalCovFrame": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S3a-BlockLocalCovFrame", init_variant=f"{stripped}_blocklocalcovp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 S3 block-local covariance frame from fixed image blocks; no dataset-name branch",
        },
        "A-S3b-BlockLocalEdgeEnergyFrame": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S3b-BlockLocalEdgeEnergyFrame", init_variant=f"{stripped}_blocklocaledgep"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 S3 block-local edge-energy frame from unlabeled x geometry",
        },
        "A-S3c-BlockLocalStableAugFrame": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S3c-BlockLocalStableAugFrame", init_variant=f"{stripped}_blocklocalstableaugp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 S3 block-local stable augmentation frame from unlabeled x geometry",
        },
        "A-S3d-BlockLocalLowQuadDirect": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S3d-BlockLocalLowQuadDirect", init_variant=f"{lower_quad}_blocklocalcovp_directreadinit125"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 S3 block-local covariance frame with low-quad direct-read balance",
        },
        "A-S4a-CrossRandomProjectionFrame": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S4a-CrossRandomProjectionFrame", init_variant=f"{stripped}_crossrandprojp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 S4 cross-random-projection predictive covariance frame; no reconstruction loss",
        },
        "A-S4b-CrossProjectionStableResidual": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S4b-CrossProjectionStableResidual", init_variant=f"{stripped}_crossrandprojp_crossprojresidualp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 S4 cross-projection predictive frame plus bounded residual projection",
        },
        "A-S4c-CrossProjectionLowQuadDirect": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S4c-CrossProjectionLowQuadDirect", init_variant=f"{lower_quad}_crossrandprojp_directreadinit125"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 S4 cross-projection predictive frame with low-quad direct-read balance",
        },
        "A-S5a-ViewStableLineCResidual": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S5a-ViewStableLineCResidual", init_variant=f"{stripped}_multiviewstablep_reslowrankr016p002_boundq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 repair: view-stable frame with bounded low-rank LineC residual; label-free.",
        },
        "A-S5b-BlockStableReducedDirect": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S5b-BlockStableReducedDirect", init_variant=f"{lower_quad}_blocklocalstableaugp_directreadinit090_boundq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 repair: block-local stable augmentation with reduced direct-read dominance; label-free.",
        },
        "A-S5c-MultiViewBlockLowRank": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S5c-MultiViewBlockLowRank", init_variant=f"{stripped}_multiviewblockp_reslowrankr016p002_boundq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 repair: multi-view block-local frame with bounded low-rank residual; label-free.",
        },
        "A-S5d-CrossProjLineCResidual": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S5d-CrossProjLineCResidual", init_variant=f"{stripped}_crossrandprojp_crossprojresidualp_reslowrankr016p002_boundq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 repair: cross-projection stable residual plus bounded LineC residual; label-free.",
        },
        "A-S6a-PseudoPartitionSignalFrame": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S6a-PseudoPartitionSignalFrame", init_variant=f"{stripped}_pseudopartitionp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 depth-6: unlabeled pseudo-partition signal frame from train-stream x clustering; no label or CE.",
        },
        "A-S6b-PseudoPartitionLowQuadDirect": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S6b-PseudoPartitionLowQuadDirect", init_variant=f"{lower_quad}_pseudopartitionp_directreadinit110"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 depth-6: pseudo-partition frame with low-quad direct-read balance; label-free.",
        },
        "A-S6c-AffinityAnchorSignalFrame": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S6c-AffinityAnchorSignalFrame", init_variant=f"{stripped}_affinityanchorp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 depth-6: unlabeled affinity-anchor signal frame from train-stream landmarks; no label or CE.",
        },
        "A-S6d-RankConsensusSignalFrame": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S6d-RankConsensusSignalFrame", init_variant=f"{stripped}_rankconsensusp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 depth-6: random-projection rank-consensus frame; label-free.",
        },
        "A-S6e-AffinityPseudoResidual": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S6e-AffinityPseudoResidual", init_variant=f"{stripped}_affinityanchorp_pseudopartitionp_reslowrankr016p002_boundq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 depth-6: affinity-anchor plus pseudo-partition residual frame; label-free.",
        },
        "A-S7a-PseudoViewMixReducedDirect": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S7a-PseudoViewMixReducedDirect", init_variant=f"{lower_quad}_pseudopartitionp_pseudoviewmixp_directreadinit090_boundq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 depth-7 repair: pseudo-partition signal frame with view-stability mix and reduced direct-read dominance.",
        },
        "A-S7b-PseudoBlockGuardResidual": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S7b-PseudoBlockGuardResidual", init_variant=f"{stripped}_pseudopartitionp_pseudoblockguardp_reslowrankr016p002_boundq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 depth-7 repair: pseudo-partition frame with block-local geometry guard and bounded low-rank residual.",
        },
        "A-S7c-AffinityPseudoBlockReducedDirect": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S7c-AffinityPseudoBlockReducedDirect", init_variant=f"{lower_quad}_affinityanchorp_pseudopartitionp_pseudoblockguardp_directreadinit090_boundq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 depth-7 repair: affinity + pseudo-partition + block guard with reduced direct-read dominance.",
        },
        "A-S7d-AffinityRankViewMix": {
            "spec": replace(base, candidate_id="V1227LabelFreeSignalSource::A-S7d-AffinityRankViewMix", init_variant=f"{stripped}_affinityanchorp_rankconsensusp_pseudoviewmixp_boundq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.27 depth-7 repair: affinity-anchor plus rank-consensus and view-stability mix; label-free.",
        },
        "A-F1a-OrthoP-LearnableFrameWarmup": {
            "spec": replace(base, candidate_id=f"{v1228_id}::A-F1a-OrthoP-LearnableFrameWarmup", init_variant=f"{stripped}_orthop_warm2pupdateevery4_directreadinit090_directramp050_quadramp030"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 Family A: label-free orthogonal P with learnable-P warmup for two epochs then periodic P updates; direct/quad ramps probe frame formation without label init.",
        },
        "A-F1b-SRHTP-LearnableFrameWarmup": {
            "spec": replace(base, candidate_id=f"{v1228_id}::A-F1b-SRHTP-LearnableFrameWarmup", init_variant=f"{stripped}_srhtp_warm2pupdateevery4_directreadinit090_directramp050_quadramp050"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 Family A: label-free SRHT/Rademacher low-coherence P with early learnable frame formation and balanced direct/quad release.",
        },
        "A-F1c-BlockOrthoP-LearnableFrameWarmup": {
            "spec": replace(base, candidate_id=f"{v1228_id}::A-F1c-BlockOrthoP-LearnableFrameWarmup", init_variant=f"{stripped}_blockframep_warm2pupdateevery4_directreadinit090_directramp050_quadramp050"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 Family A: label-free block-local orthogonal frame with learnable-P warmup; no dataset-name branch or label statistics.",
        },
        "A-F1d-RandomLowCoherenceP-LearnableFrameWarmup": {
            "spec": replace(base, candidate_id=f"{v1228_id}::A-F1d-RandomLowCoherenceP-LearnableFrameWarmup", init_variant=f"{stripped}_lowcoherencep_warm2pupdateevery4_directreadinit090_directramp050_quadramp050"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 Family A: random low-coherence label-free P with trainable frame warmup and periodic release.",
        },
        "A-F2a-RoleBalancedFHQ-LF": {
            "spec": replace(base, candidate_id=f"{v1228_id}::A-F2a-RoleBalancedFHQ-LF", init_variant=f"{stripped}_rolebalancedp_directramp050_quadramp050"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 Family B: role-balanced label-free FHQ frame to prevent early direct dominance while preserving normal supervised training.",
        },
        "A-F2b-QuadDirectBalanceRamp-LF": {
            "spec": replace(base, candidate_id=f"{v1228_id}::A-F2b-QuadDirectBalanceRamp-LF", init_variant=f"{lower_quad}_orthop_directreadinit090_directramp050_quadramp080_boundq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 Family B: low-quad orthogonal base with delayed direct branch and stronger quad release for LineC co-location.",
        },
        "A-F2c-BranchGainFrozenEarly-LF": {
            "spec": replace(base, candidate_id=f"{v1228_id}::A-F2c-BranchGainFrozenEarly-LF", init_variant=f"{stripped}_orthop_classbranch_classgain_gainramp050_directramp050_quadramp050"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 Family B: class-branch/gain ramp using label-free P; only ordinary task loss trains parameters, no label init.",
        },
        "A-F2d-RoleEnergyEqualizedWarmup-LF": {
            "spec": replace(base, candidate_id=f"{v1228_id}::A-F2d-RoleEnergyEqualizedWarmup-LF", init_variant=f"{stripped}_rolebalancedp_classbranch_classgain_gainramp050_directramp050_quadramp050_rmsq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 Family B: role-energy equalized warmup with RMSQ guard; intended to test direct/quad/branch energy balance.",
        },
        "A-F3a-UpdateSpectrumFrame-LF": {
            "spec": replace(base, candidate_id=f"{v1228_id}::A-F3a-UpdateSpectrumFrame-LF", init_variant=f"{stripped}_orthop_optframep_warm1pupdateevery4"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 Family C: label-free optimizer-update aggregate frame refresh from quad projector deltas; no per-example label or CE vector.",
        },
        "A-F3b-MomentumCovFrame-LF": {
            "spec": replace(base, candidate_id=f"{v1228_id}::A-F3b-MomentumCovFrame-LF", init_variant=f"{stripped}_srhtp_optframep_reslowrankr008p001_warm1pupdateevery4"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 Family C: SRHT start plus optimizer-update spectrum residual refresh; uses aggregate update covariance only.",
        },
        "A-F3c-GradientNormOnlyFrame-LF": {
            "spec": replace(base, candidate_id=f"{v1228_id}::A-F3c-GradientNormOnlyFrame-LF", init_variant=f"{stripped}_lowcoherencep_optframep_boundq_warm1pupdateevery4"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 Family C: low-coherence frame with norm-limited optimizer-observable refresh and bounded quadratic response.",
        },
        "A-F3d-AdamSecondMomentFrame-LF": {
            "spec": replace(base, candidate_id=f"{v1228_id}::A-F3d-AdamSecondMomentFrame-LF", init_variant=f"{stripped}_rolecondp_optframep_reslowrankr008p001_rmsq_warm1pupdateevery4"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 Family C: role-conditioned optimizer-observable frame refresh, audited as aggregate-state only and not a functional direction.",
        },
        "A-F4a-OvercompleteSRHTP-h192-LF": {
            "spec": replace(base, hidden_dim=192, candidate_id=f"{v1228_id}::A-F4a-OvercompleteSRHTP-h192-LF", init_variant=f"{stripped}_srhtp_pupdateevery4_directreadinit090_quadramp050"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 Family D: h192 structured SRHT label-free cover with periodic P updates; tests capacity without label directions.",
        },
        "A-F4b-OvercompleteBlockP-h192-LF": {
            "spec": replace(base, hidden_dim=192, candidate_id=f"{v1228_id}::A-F4b-OvercompleteBlockP-h192-LF", init_variant=f"{stripped}_blockframep_pupdateevery4_directreadinit090_quadramp050"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 Family D: h192 block-local structured label-free cover with periodic P updates.",
        },
        "A-F4c-OvercompleteSparseP-h192-LF": {
            "spec": replace(base, hidden_dim=192, candidate_id=f"{v1228_id}::A-F4c-OvercompleteSparseP-h192-LF", init_variant=f"{stripped}_sparsek8p_fixedp_directreadinit090_quadramp050"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 Family D: h192 sparse label-free P cover to test capacity under structured sparse efficiency constraints.",
        },
        "A-F4d-OvercompleteThenPruneP-LF": {
            "spec": replace(base, hidden_dim=192, candidate_id=f"{v1228_id}::A-F4d-OvercompleteThenPruneP-LF", init_variant=f"{stripped}_srhtp_activep96_warm2pupdateevery6_directreadinit090_quadramp050"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 Family D: h192 overcomplete SRHT cover with active-column pruning mask and delayed P update cadence.",
        },
        "A-R1-TaskGoodLineCQuadWarm": {
            "spec": replace(base, candidate_id=f"{v1228_id}::A-R1-TaskGoodLineCQuadWarm", init_variant=f"{lower_quad}_srhtp_warm2pupdateevery4_directreadinit075_directramp040_quadramp100_boundq_rmsq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 repair: if task is near but LineC fails, lower direct dominance and release quad branch with bound/RMSQ guard.",
        },
        "A-R2-LineCGoodTaskDirectWarm": {
            "spec": replace(base, candidate_id=f"{v1228_id}::A-R2-LineCGoodTaskDirectWarm", init_variant=f"{lower_quad}_blockframep_warm1pupdateevery4_directreadinit125_directramp100_quadramp050"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 repair: if LineC improves but task fails, increase direct readout warmup while keeping label-free block frame.",
        },
        "A-R3-AUCTrajectoryRelease": {
            "spec": replace(base, candidate_id=f"{v1228_id}::A-R3-AUCTrajectoryRelease", init_variant=f"{stripped}_lowcoherencep_warm1pupdateevery8_directreadinit090_directramp075_quadramp050"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 repair: shorten learnable-P warmup and use slower periodic P release for AUC-time failure.",
        },
        "A-R4-UpdateSpectrumWeakRefresh": {
            "spec": replace(base, candidate_id=f"{v1228_id}::A-R4-UpdateSpectrumWeakRefresh", init_variant=f"{stripped}_orthop_optframep_reslowrankr008p001_boundq_warm1pupdateevery6"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 repair: weak optimizer-observable frame refresh with bounded residual for update-spectrum blocker.",
        },
        "A-R5-OvercompleteRoleBalance": {
            "spec": replace(base, hidden_dim=192, candidate_id=f"{v1228_id}::A-R5-OvercompleteRoleBalance", init_variant=f"{stripped}_srhtp_rolebalancedp_directramp050_quadramp080_pupdateevery4_rmsq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.28 repair: overcomplete structured cover combined once with role balance when LineC improves but task/AUC remains weak.",
        },
        "A-DYN1-LearnableSignalFrameWarmup": {
            "spec": replace(base, candidate_id=f"{v1230_id}::A-DYN1-LearnableSignalFrameWarmup", init_variant=f"{stripped}_srhtp_warm2pupdateevery4_directreadinit090_directramp050_quadramp050"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.30 monitor: label-free learnable signal-frame warmup; no y_stats or trainprobe token.",
        },
        "A-DYN2-EarlySelfPredictiveFrame": {
            "spec": replace(base, candidate_id=f"{v1230_id}::A-DYN2-EarlySelfPredictiveFrame", init_variant=f"{stripped}_selfcondstopgradp_lowcoherencep_warm2pupdateevery4_directreadinit090_quadramp050"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.30 monitor: self-conditioned stop-grad/input geometry frame, audited as label-free base formation.",
        },
        "A-DYN3-OptimizerObservableFrameRefresh": {
            "spec": replace(base, candidate_id=f"{v1230_id}::A-DYN3-OptimizerObservableFrameRefresh", init_variant=f"{stripped}_orthop_optframep_reslowrankr008p001_warm1pupdateevery6_boundq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.30 monitor: optimizer-observable aggregate frame refresh; exploration-only because task optimizer labels influence aggregate updates.",
        },
        "A-DYN4-OvercompleteRankGuardFrame": {
            "spec": replace(base, hidden_dim=192, candidate_id=f"{v1230_id}::A-DYN4-OvercompleteRankGuardFrame", init_variant=f"{stripped}_srhtp_activep96_warm2pupdateevery6_directreadinit090_quadramp050_rmsq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.30 monitor: overcomplete structured frame with active-rank guard; no label-informed initialization.",
        },
        "A-DYN5-RoleEnergyBalancedFHQMonitor": {
            "spec": replace(base, candidate_id=f"{v1230_id}::A-DYN5-RoleEnergyBalancedFHQMonitor", init_variant=f"{stripped}_rolebalancedp_directramp050_quadramp080_boundq_rmsq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.30 monitor: role-energy balanced FHQ base with LineC audit only; no LineC hard target direction.",
        },
        "A2-randomP-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A2-randomP-labelFree", init_variant=f"{stripped}_randomp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "trainprobe tokens removed; random P/direct label-free init",
        },
        "A3-PCA-P-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A3-PCA-P-labelFree", init_variant=f"{stripped}_pcap"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "trainprobe tokens removed; input PCA-P label-free init",
        },
        "A4-lowfreqP-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A4-lowfreqP-labelFree", init_variant=f"{stripped}_lowfreqp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "trainprobe tokens removed; low-frequency P label-free init",
        },
        "A5-orthogonalP-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A5-orthogonalP-labelFree", init_variant=f"{stripped}_orthop"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "trainprobe tokens removed; orthogonal P label-free init",
        },
        "A6-orthogonalP-active64-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A6-orthogonalP-active64-labelFree", init_variant=f"{stripped}_orthop_activep64"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "A5 plus activeP64 gradient mask; label-free repair for over-broad quadratic projection movement",
        },
        "A7-orthogonalP-lowQuad-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A7-orthogonalP-lowQuad-labelFree", init_variant=f"{lower_quad}_orthop"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "A5 plus quadreadinit075 and identitytailquad020; label-free reservoir-ratio repair",
        },
        "A8-orthogonalP-boundQ-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A8-orthogonalP-boundQ-labelFree", init_variant=f"{stripped}_orthop_boundq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "A5 plus bounded quadratic features; label-free noise-leak repair",
        },
        "A9-orthogonalP-lowQuad-boundQ-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A9-orthogonalP-lowQuad-boundQ-labelFree", init_variant=f"{lower_quad}_orthop_boundq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "A7 plus bounded quadratic features; combined label-free LineC repair",
        },
        "A10-orthogonalP-strongLowQuad-boundQ-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A10-orthogonalP-strongLowQuad-boundQ-labelFree", init_variant=f"{lower_quad_strong}_orthop_boundq_quadramp010"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "A9 plus quadreadinit050 and quadRamp010 warm start; aggressive label-free LineC reservoir repair",
        },
        "A11-orthogonalP-directRead125-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A11-orthogonalP-directRead125-labelFree", init_variant=f"{stripped}_orthop_directreadinit125"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "A5 plus label-free direct readout init scale 1.25; direct-branch repair for reservoir overuse",
        },
        "A12-orthogonalP-identityAmp150-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A12-orthogonalP-identityAmp150-labelFree", init_variant=f"{stripped}_orthop_identityamp150"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "A5 plus label-free identity direct readout scale 1.50; direct-branch repair for reservoir overuse",
        },
        "A13-orthogonalP-lowQuad-directRead125-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A13-orthogonalP-lowQuad-directRead125-labelFree", init_variant=f"{lower_quad}_orthop_directreadinit125"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "A7 plus label-free direct readout init scale 1.25; combined direct/low-quad LineC repair",
        },
        "A14-orthogonalP-lowQuad-identityAmp150-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A14-orthogonalP-lowQuad-identityAmp150-labelFree", init_variant=f"{lower_quad}_orthop_identityamp150"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "A7 plus label-free identity direct readout scale 1.50; combined direct/low-quad LineC repair",
        },
        "A15-PCAOrthoMix-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A15-PCAOrthoMix-labelFree", init_variant=f"{stripped}_pcaorthomixp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "PCA top components plus orthogonal residual P; label-free manifold-aligned projector",
        },
        "A16-AugStableP-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A16-AugStableP-labelFree", init_variant=f"{stripped}_augstablep"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "augmentation-stable input directions; label-free projector",
        },
        "A17-RandomCotangentStableP-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A17-RandomCotangentStableP-labelFree", init_variant=f"{stripped}_randomcotangentstablep"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "random cotangent stable directions from unlabeled train-stream geometry; label-free projector",
        },
        "A18-InputCovWhitenedOrthoP-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A18-InputCovWhitenedOrthoP-labelFree", init_variant=f"{stripped}_covwhitenp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "input-covariance whitened orthogonal P; label-free projector",
        },
        "A19-BlockLocalAugStableP-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A19-BlockLocalAugStableP-labelFree", init_variant=f"{stripped}_blocklocalaugstablep"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "block-local augmentation-stable directions; label-free projector",
        },
        "A20-shuffledLabelTrainProbe-diagnostic": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A20-shuffledLabelTrainProbe-diagnostic"),
            "uses_y_for_stats": 1,
            "y_stats_mode": "shuffled",
            "seed_offset": 0,
            "description": "diagnostic trainprobe init using shuffled training labels; not eligible for label-free claim",
        },
        "A21-randomClassCentroid-diagnostic": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A21-randomClassCentroid-diagnostic"),
            "uses_y_for_stats": 1,
            "y_stats_mode": "random_balanced",
            "seed_offset": 0,
            "description": "diagnostic trainprobe init using random balanced pseudo-class ids; not eligible for label-free claim",
        },
        "A22-permutedClassMeanP-diagnostic": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A22-permutedClassMeanP-diagnostic"),
            "uses_y_for_stats": 1,
            "y_stats_mode": "permuted_class_ids",
            "seed_offset": 0,
            "description": "diagnostic trainprobe init using fixed class-id permutation; not eligible for label-free claim",
        },
        "A23-MultiFrameBank-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A23-MultiFrameBank-labelFree", init_variant=f"{stripped}_multiframebankp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.20 A-F1 multi-frame projection bank: PCA/SRHT/augmentation/local/low-frequency frames; label-free",
        },
        "A24-MultiFrameBankDirect125-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A24-MultiFrameBankDirect125-labelFree", init_variant=f"{stripped}_multiframebankp_directreadinit125"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.20 A-F1 multi-frame projection bank plus label-free direct readout scaling",
        },
        "A25-SelfConditionResidualP-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A25-SelfConditionResidualP-labelFree", init_variant=f"{stripped}_selfcondresp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.20 A-F2 activation-covariance plus random-cotangent residual projector; label-free",
        },
        "A26-CovAdaptP-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A26-CovAdaptP-labelFree", init_variant=f"{stripped}_covadaptp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.20 A-F3 label-free covariance-initialized projector with unlabeled epoch adaptation",
        },
        "A27-MultiFrameLowFreqBias-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A27-MultiFrameLowFreqBias-labelFree", init_variant=f"{stripped}_multiframebankp_lowbias"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.20 A-F1 multi-frame bank with extra low-frequency image-frame quota; label-free",
        },
        "A28-unsupervisedClusterTrainProbe-diagnostic": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A28-unsupervisedClusterTrainProbe-diagnostic"),
            "uses_y_for_stats": 1,
            "y_stats_mode": "unsupervised_kmeans",
            "seed_offset": 0,
            "description": "v12.20 A-UB4 unsupervised clustering centroid diagnostic; pseudo-label diagnostic only, not promotion",
        },
        "A29-oracleSmallLabelTrainProbe-diagnostic": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A29-oracleSmallLabelTrainProbe-diagnostic"),
            "uses_y_for_stats": 1,
            "y_stats_mode": "small_label_fraction",
            "seed_offset": 0,
            "description": "v12.20 A-UB5 oracle small-label class-mean diagnostic; upper-bound only, not promotion",
        },
        "A30-OptimizerObservableFrame-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A30-OptimizerObservableFrame-labelFree", init_variant=f"{stripped}_optframep"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.21 A30 optimizer-observable opaque update frame; no label/CE vector read for frame construction",
        },
        "A31-AugConsistencyTangentFrame-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A31-AugConsistencyTangentFrame-labelFree", init_variant=f"{stripped}_augtangentp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.21 A31 augmentation-consistency tangent frame from unlabeled input perturbation geometry",
        },
        "A32-PersistentDriftFrame-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A32-PersistentDriftFrame-labelFree", init_variant=f"{stripped}_persistentdriftp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.21 A32 persistent train-stream logit drift frame; labels/CE not read for frame construction",
        },
        "A33-RoleBalancedPrimitiveEnergyFrame-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A33-RoleBalancedPrimitiveEnergyFrame-labelFree", init_variant=f"{stripped}_rolebalancedp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.21 A33 role-balanced primitive energy frame with equalized projector quotas",
        },
        "A34-HybridA1OptimizerFrame-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A34-HybridA1OptimizerFrame-labelFree", init_variant=f"{base.init_variant}_optframep"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.21 A34 hybrid A1 no-y-for-stats B320 plus optimizer-observable residual frame",
        },
        "A35-HybridA1AugDriftFrame-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A35-HybridA1AugDriftFrame-labelFree", init_variant=f"{base.init_variant}_augtangentp_persistentdriftp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.21 A35 hybrid A1 no-y-for-stats B320 plus augmentation tangent and persistent drift frames",
        },
        "A36-ResidualA1LowRankFrame-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A36-ResidualA1LowRankFrame-labelFree", init_variant=f"{base.init_variant}_reslowrankp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.22 A36 residual-on-A1 low-rank label-free frame from input covariance and random cotangent stability",
        },
        "A37-ConvexMultiFrameMixture-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A37-ConvexMultiFrameMixture-labelFree", init_variant=f"{stripped}_convexmixp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.22 A37 convex mixture of PCA, SRHT, augmentation-stable, persistent-drift, and local frames",
        },
        "A38-PersistentDriftCotangentBank-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A38-PersistentDriftCotangentBank-labelFree", init_variant=f"{stripped}_driftcotbankp_persistentdriftp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.22 A38 frozen random cotangent bank mapped through early unlabeled logit/hidden drift",
        },
        "A39-RoleConditionedFrame-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A39-RoleConditionedFrame-labelFree", init_variant=f"{stripped}_rolecondp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.22 A39 role-conditioned frame with separate direct/quad/branch quotas",
        },
        "A40-SelfConditionedStopGradFrame-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A40-SelfConditionedStopGradFrame-labelFree", init_variant=f"{stripped}_selfcondstopgradp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.22 A40 stop-grad self-conditioned activation covariance frame",
        },
        "A41-UnlabeledFrameAdaptSchedule-labelFree": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A41-UnlabeledFrameAdaptSchedule-labelFree", init_variant=f"{base.init_variant}_adaptframeschedp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.22 A41 A1 start followed by small unlabeled covariance residual frame adaptation after epoch 1",
        },
        "A42-SmallLabelOracleUpperBound-diagnostic": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A42-SmallLabelOracleUpperBound-diagnostic"),
            "uses_y_for_stats": 1,
            "y_stats_mode": "small_label_fraction",
            "seed_offset": 0,
            "description": "v12.22 A42 small-label oracle upper bound; diagnostic only and not promotable",
        },
        "A43-ConservativeResidualA1Frame-r005": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A43-ConservativeResidualA1Frame-r005", init_variant=f"{base.init_variant}_reslowrankp005"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 A43 conservative residual-on-A1 low-rank frame with residual strength 0.005",
        },
        "A44-ConservativeResidualA1Frame-r010": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A44-ConservativeResidualA1Frame-r010", init_variant=f"{base.init_variant}_reslowrankp010"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 A44 conservative residual-on-A1 low-rank frame with residual strength 0.010",
        },
        "A45-StagedUnlabeledAdapt-warm1-r005": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A45-StagedUnlabeledAdapt-warm1-r005", init_variant=f"{base.init_variant}_adaptframeschedp005"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 A45 A1 start with staged unlabeled projector adaptation strength 0.005 after warmup epoch 1",
        },
        "A46-RoleResidualNoReplace-directOnly": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A46-RoleResidualNoReplace-directOnly", init_variant=f"{stripped}_orthop_directreadinit125_quadramp010"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 A46 label-free direct-role residual emphasis without trainprobe replacement",
        },
        "A47-QuadProjLowRankResidualFrozenMain": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A47-QuadProjLowRankResidualFrozenMain", init_variant=f"{base.init_variant}_reslowrankp005_fixedp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 A47 low-rank residual frame with frozen main quadratic projection after initialization",
        },
        "A48-BranchGainOnlyLabelFreeResidual": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A48-BranchGainOnlyLabelFreeResidual", init_variant=f"{stripped}_orthop_fixedp_classbranch_classgain_gainramp050_quadramp010"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 A48 label-free frozen projector with branch/gain-only residual movement",
        },
        "A49-A1PlusT1BWeakSignalFrame-diagnostic": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A49-A1PlusT1BWeakSignalFrame-diagnostic", init_variant=f"{base.init_variant}_optframep_reslowrankp005"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 A49 diagnostic A1 plus optimizer-observable weak-signal residual frame; not promotion eligible",
        },
        "A50-SmallLabelOracleMatchedBudget-diagnostic": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A50-SmallLabelOracleMatchedBudget-diagnostic"),
            "uses_y_for_stats": 1,
            "y_stats_mode": "small_label_fraction",
            "seed_offset": 0,
            "description": "v12.23 A50 matched-budget small-label oracle diagnostic; not promotion eligible",
        },
        "A51-StagedUnlabeledAdapt-warm1-r002": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A51-StagedUnlabeledAdapt-warm1-r002", init_variant=f"{base.init_variant}_adaptframeschedp002"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 continuation repair: lower A45 staged unlabeled adaptation strength from 0.005 to 0.002 after warmup epoch 1",
        },
        "A52-StagedUnlabeledAdapt-warm1-r001": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A52-StagedUnlabeledAdapt-warm1-r001", init_variant=f"{base.init_variant}_adaptframeschedp001"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 continuation repair: minimal staged unlabeled adaptation strength 0.001 to test over-injection as the hardening blocker",
        },
        "A53-ResidualA1LowRankFrame-r002-fixedP": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A53-ResidualA1LowRankFrame-r002-fixedP", init_variant=f"{base.init_variant}_reslowrankp002_fixedp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 continuation repair: lower residual low-rank frame strength to 0.002 and freeze the main projector after initialization",
        },
        "A54-A1FixedPBranchLowQuad020": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A54-A1FixedPBranchLowQuad020",
                init_variant=f"{replace_variant_token(base.init_variant, 'identitytailquad030', 'identitytailquad020')}_fixedp",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 continuation repair: keep A1 label-free trainprobe path, freeze projector, and lower identity-tail quadratic branch from 0.030 to 0.020",
        },
        "A55-ResidualA1LowRankFrame-r001-fixedP": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A55-ResidualA1LowRankFrame-r001-fixedP", init_variant=f"{base.init_variant}_reslowrankp001_fixedp"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 all-line continuation: minimal residual low-rank frame strength 0.001 with frozen main projector after initialization",
        },
        "A56-A1FixedPBranchLowQuad010": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A56-A1FixedPBranchLowQuad010",
                init_variant=f"{replace_variant_token(base.init_variant, 'identitytailquad030', 'identitytailquad010')}_fixedp",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 all-line continuation: keep A1 label-free path, freeze projector, and lower identity-tail quadratic branch from 0.030 to 0.010",
        },
        "A57-A51BoundQLineCRepair": {
            "spec": replace(base, candidate_id=f"{B320_ID}::A57-A51BoundQLineCRepair", init_variant=f"{base.init_variant}_adaptframeschedp002_boundq"),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 target-open LineC repair: A51 staged unlabeled adaptation plus bounded quadratic features to reduce noise/reservoir tearing",
        },
        "A58-A51SignalBlock010LineCRepair": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A58-A51SignalBlock010LineCRepair",
                init_variant=f"{replace_variant_token(base.init_variant, 'signalblock015', 'signalblock010')}_adaptframeschedp002",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 target-open LineC repair: A51 staged adaptation with lower signal block fraction 0.010",
        },
        "A59-A51SignalBroad025Block010BoundQ": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A59-A51SignalBroad025Block010BoundQ",
                init_variant=f"{replace_variant_token(replace_variant_token(base.init_variant, 'signalbroad035', 'signalbroad025'), 'signalblock015', 'signalblock010')}_adaptframeschedp002_boundq",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 target-open LineC repair: A51 with lower signal broad/block fractions plus bounded quadratic features",
        },
        "A60-A51DirectRead125ReservoirRepair": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A60-A51DirectRead125ReservoirRepair",
                init_variant=f"{base.init_variant}_adaptframeschedp002_directreadinit125",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 target-open LineC repair: A51 plus label-free direct readout scale 1.25 to reduce reservoir dependence",
        },
        "A61-A51IdentityAmp150ReservoirRepair": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A61-A51IdentityAmp150ReservoirRepair",
                init_variant=f"{base.init_variant}_adaptframeschedp002_identityamp150",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 target-open LineC repair: A51 plus identity direct-path amplification 1.50 to rebalance signal/reservoir roles",
        },
        "A62-A58DirectRead125ReservoirRepair": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A62-A58DirectRead125ReservoirRepair",
                init_variant=f"{replace_variant_token(base.init_variant, 'signalblock015', 'signalblock010')}_adaptframeschedp002_directreadinit125",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 target-open LineC repair: A58 lower signal block plus direct readout scale 1.25",
        },
        "A63-A51RMSQReservoirRepair": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A63-A51RMSQReservoirRepair",
                init_variant=f"{base.init_variant}_adaptframeschedp002_rmsq",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 target-open LineC repair: A51 plus quadratic batch-RMS normalization to reduce reservoir domination",
        },
        "A64-A51RMSQBoundQReservoirRepair": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A64-A51RMSQBoundQReservoirRepair",
                init_variant=f"{base.init_variant}_adaptframeschedp002_rmsq_boundq",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 target-open LineC repair: A51 plus quadratic batch-RMS normalization and bounded quadratic response",
        },
        "A65-A58RMSQSignalBlockRepair": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A65-A58RMSQSignalBlockRepair",
                init_variant=f"{replace_variant_token(base.init_variant, 'signalblock015', 'signalblock010')}_adaptframeschedp002_rmsq",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.23 target-open LineC repair: A58 lower signal block plus quadratic batch-RMS normalization",
        },
        "A66-LineCAwareUnlabeledMultiSketchFrame": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A66-LineCAwareUnlabeledMultiSketchFrame",
                init_variant=f"{base.init_variant}_adaptframeschedp002_reslowrankp001_rmsq",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.24 Line A repair: A51 staged unlabeled adaptation plus very-low-rank unlabeled residual and RMSQ stabilization; no labels or CE vector used for frame construction",
        },
        "A67-ReservoirStabilizedResidualFrame": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A67-ReservoirStabilizedResidualFrame",
                init_variant=f"{base.init_variant}_reslowrankp001_rmsq_boundq",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.24 Line A repair: conservative residual low-rank frame with RMSQ and bounded quadratic response to test reservoir/noise stabilization",
        },
        "A68-A51TrainProbeCouplingPreservingFrame": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A68-A51TrainProbeCouplingPreservingFrame",
                init_variant=f"{replace_variant_token(base.init_variant, 'signalblock015', 'signalblock010')}_adaptframeschedp002_reslowrankp001_rmsq",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.24 Line A repair: A51 with lower signal block, tiny unlabeled residual frame, and RMSQ; intended to preserve task while probing LineC non-tearing",
        },
        "A69-A51MultiSketchRMSQBoundQNoReadoutRepair": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A69-A51MultiSketchRMSQBoundQNoReadoutRepair",
                init_variant=f"{replace_variant_token(replace_variant_token(base.init_variant, 'signalbroad035', 'signalbroad025'), 'signalblock015', 'signalblock010')}_adaptframeschedp002_rmsq_boundq",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.24 Line A repair: A51 with lower signal broad/block fractions plus RMSQ and bounded Q; label-free fallback after A63/A65 LineC fail",
        },
        "A70-A51ResidualLineCFrame-rank8": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A70-A51ResidualLineCFrame-rank8",
                init_variant=f"{base.init_variant}_adaptframeschedp002_reslowrankr008p001_rmsq",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.25 Line A repair: A51 staged unlabeled adaptation plus rank-8 low-strength residual frame and RMSQ; no labels or CE vector used",
        },
        "A71-A51ResidualLineCFrame-rank16": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A71-A51ResidualLineCFrame-rank16",
                init_variant=f"{base.init_variant}_adaptframeschedp002_reslowrankr016p001_rmsq",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.25 Line A repair: A51 staged unlabeled adaptation plus rank-16 low-strength residual frame and RMSQ; no labels or CE vector used",
        },
        "A72-A51ControlResidualFrame-rank8": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A72-A51ControlResidualFrame-rank8",
                init_variant=f"{base.init_variant}_adaptframeschedp002_reslowrankr008p001_boundq",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.25 Line A repair: rank-8 residual frame with bounded quadratic response to test control-residual non-tearing without labels",
        },
        "A73-A51ControlResidualFrame-rank16": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A73-A51ControlResidualFrame-rank16",
                init_variant=f"{base.init_variant}_adaptframeschedp002_reslowrankr016p001_boundq",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.25 Line A repair: rank-16 residual frame with bounded quadratic response to test control-residual non-tearing without labels",
        },
        "A74-A51DirectQuadBalance-noY": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A74-A51DirectQuadBalance-noY",
                init_variant=f"{base.init_variant}_adaptframeschedp002_directreadinit125_quadreadinit125",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.25 Line A repair: direct/quad readout balance on top of A51 staged adaptation; label-free",
        },
        "A75-A51RoleEnergyTailClamp-noY": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A75-A51RoleEnergyTailClamp-noY",
                init_variant=f"{base.init_variant}_adaptframeschedp002_rmsq_boundq_directreadinit090",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.25 Line A repair: label-free tail proxy clamp using RMSQ, bounded quadratic features, and conservative direct readout scale",
        },
        "A76-A51LineCEMAAdapt-noY": {
            "spec": replace(
                base,
                candidate_id=f"{B320_ID}::A76-A51LineCEMAAdapt-noY",
                init_variant=f"{base.init_variant}_adaptframeschedp001_reslowrankr008p001_rmsq",
            ),
            "uses_y_for_stats": 0,
            "seed_offset": 0,
            "description": "v12.25 Line A repair: lower-strength EMA-style adaptation with rank-8 residual and RMSQ; no labels or CE vector used",
        },
    }


def make_model(
    method_id: str,
    spec: prim.PrimitiveSpec | None,
    input_dim: int,
    output_dim: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    device: torch.device,
    seed: int,
    uses_y_for_stats: int,
    y_stats_mode: str = "actual",
) -> torch.nn.Module:
    _, budget = v124._param_budget(input_dim, output_dim)
    if method_id in MLP_CONTROL_IDS:
        hidden, _ = v124._param_budget(input_dim, output_dim)
        # v12.18 uses the v12.4/v12.14 hidden=256 control budget; under this
        # budget the same-param and same-step/FLOP MLP controls collapse.
        if method_id == MLP_SAME_STEP_FLOP_ID:
            hidden = 256
        return prim.MLPBaseline(input_dim, output_dim, hidden, seed, device).to(device)
    assert spec is not None
    y_stats = build_y_stats_for_mode(y_train, output_dim, seed, y_stats_mode, x_train=x_train) if int(uses_y_for_stats) else None
    return v124._make_model(spec.candidate_id, input_dim, output_dim, x_train, device, seed, spec, budget, y_stats)


def buffer_int(model: torch.nn.Module, name: str) -> int:
    value = getattr(model, name, None)
    if torch.is_tensor(value):
        return int(value.detach().flatten()[0].cpu().item())
    return 0


def signal_reservoir_metrics_detailed(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    sketch_dim: int,
    seed: int,
) -> dict[str, float]:
    b = min(int(x.shape[0]), int(y.shape[0]))
    x = x[:b]
    y = y[:b]
    grad_sketch = v1252._sample_grad_sketch(model, x, y, int(sketch_dim), int(seed))
    if not bool(torch.isfinite(grad_sketch).all()):
        nan = float("nan")
        return {
            "signal_effective_rank": nan,
            "signal_mass_topk": nan,
            "reservoir_fraction": nan,
            "top_eigen_share": nan,
            "dissipation_condition": nan,
            "RealSignalReservoirRatio": nan,
            "NoiseSignalLeak": nan,
            "SNR_positive_fraction": nan,
            "real_noise_gap": nan,
            "signal_top_count": 0.0,
            "real_residual_energy": nan,
            "real_total_energy": nan,
            "noise_signal_energy": nan,
            "noise_total_energy": nan,
        }
    k_mat = grad_sketch @ grad_sketch.T
    k_mat = 0.5 * (k_mat + k_mat.T)
    k_mat = torch.nan_to_num(k_mat.float(), nan=0.0, posinf=1.0e6, neginf=-1.0e6)
    evals = evecs = None
    eye = torch.eye(int(k_mat.shape[0]), device=k_mat.device, dtype=torch.float32)
    for jitter_scale in [0.0, 1.0e-6, 1.0e-4, 1.0e-2]:
        try:
            evals, evecs = torch.linalg.eigh(k_mat + float(jitter_scale) * eye)
            break
        except RuntimeError:
            evals = evecs = None
    if evals is None or evecs is None:
        try:
            cpu_mat = (k_mat + 1.0e-2 * eye).cpu()
            evals_cpu, evecs_cpu = torch.linalg.eigh(cpu_mat)
            evals = evals_cpu.to(device=x.device)
            evecs = evecs_cpu.to(device=x.device)
        except RuntimeError:
            nan = float("nan")
            return {
                "signal_effective_rank": nan,
                "signal_mass_topk": nan,
                "reservoir_fraction": nan,
                "top_eigen_share": nan,
                "dissipation_condition": nan,
                "RealSignalReservoirRatio": nan,
                "NoiseSignalLeak": nan,
                "SNR_positive_fraction": nan,
                "real_noise_gap": nan,
                "signal_top_count": 0.0,
                "real_residual_energy": nan,
                "real_total_energy": nan,
                "noise_signal_energy": nan,
                "noise_total_energy": nan,
            }
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp_min(0.0)
    evecs = evecs[:, order]
    total = evals.sum().clamp_min(EPS)
    cum = torch.cumsum(evals, dim=0)
    top_count = int(torch.searchsorted(cum, 0.80 * total).item()) + 1
    top_count = max(1, min(top_count, int(evals.numel())))
    p_sig = evecs[:, :top_count] @ evecs[:, :top_count].T
    p_res = torch.eye(b, device=x.device) - p_sig
    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(dim=1)
        ce_real = F.cross_entropy(logits, y, reduction="none")
        noise_gen = torch.Generator(device=x.device).manual_seed(int(seed) + 33)
        y_noise = y[torch.randperm(b, device=x.device, generator=noise_gen)]
        ce_noise = F.cross_entropy(logits, y_noise, reduction="none")
        r_real = ce_real.float() - ce_real.float().mean()
        r_noise = ce_noise.float() - ce_noise.float().mean()
        real_residual_energy = (p_res @ r_real).square().sum()
        real_total_energy = r_real.square().sum().clamp_min(EPS)
        noise_signal_energy = (p_sig @ r_noise).square().sum()
        noise_total_energy = r_noise.square().sum().clamp_min(EPS)
        real_res = float(real_residual_energy.div(real_total_energy).item())
        noise_sig = float(noise_signal_energy.div(noise_total_energy).item())
        snr_pos = float(((probs.gather(1, y.view(-1, 1)).squeeze(1) - probs.mean(dim=1)) > 0).float().mean().item())
        real_noise_gap = float(ce_noise.mean().sub(ce_real.mean()).item())
    eff_rank = float((evals.sum().square() / evals.square().sum().clamp_min(EPS)).item())
    signal_mass = float(evals[:top_count].sum().div(total).item())
    positive = evals[evals > 1.0e-8]
    return {
        "signal_effective_rank": eff_rank,
        "signal_mass_topk": signal_mass,
        "reservoir_fraction": float(1.0 - signal_mass),
        "top_eigen_share": float(evals[0].div(total).item()) if evals.numel() else 0.0,
        "dissipation_condition": float(evals[0].div(positive[-1].clamp_min(EPS)).item()) if bool(positive.numel()) else 0.0,
        "RealSignalReservoirRatio": real_res,
        "NoiseSignalLeak": noise_sig,
        "SNR_positive_fraction": snr_pos,
        "real_noise_gap": real_noise_gap,
        "signal_top_count": float(top_count),
        "real_residual_energy": float(real_residual_energy.item()),
        "real_total_energy": float(real_total_energy.item()),
        "noise_signal_energy": float(noise_signal_energy.item()),
        "noise_total_energy": float(noise_total_energy.item()),
    }


def projector_alignment_diagnostics(model: torch.nn.Module, x_train: torch.Tensor) -> dict[str, Any]:
    if not hasattr(model, "quad_proj") or not hasattr(model, "mu") or not hasattr(model, "std"):
        return {}
    quad_proj = getattr(model, "quad_proj")
    if not torch.is_tensor(quad_proj) or int(quad_proj.dim()) != 2:
        return {}
    with torch.no_grad():
        p = quad_proj.detach().float()
        total = p.square().sum().clamp_min(EPS)
        x = x_train[: min(1024, int(x_train.shape[0]))].float()
        z = ((x - getattr(model, "mu")) / getattr(model, "std")).clamp(-3.0, 3.0)
        z = z - z.mean(dim=0, keepdim=True)
        out: dict[str, Any] = {}
        try:
            _u, _s, vh = torch.linalg.svd(z, full_matrices=False)
            top = min(16, int(vh.shape[0]))
            if top > 0:
                basis = vh[:top].transpose(0, 1).contiguous()
                out["P_energy_on_top_pca"] = float((basis.transpose(0, 1) @ p).square().sum().div(total).item())
        except RuntimeError:
            out["P_energy_on_top_pca"] = ""
        try:
            z_img = z
            side = int(round(math.sqrt(float(z.shape[1]))))
            if side * side == int(z.shape[1]):
                img = z.reshape(int(z.shape[0]), side, side)
                stable = 0.5 * (
                    img
                    + 0.25
                    * (
                        torch.roll(img, shifts=1, dims=1)
                        + torch.roll(img, shifts=-1, dims=1)
                        + torch.roll(img, shifts=1, dims=2)
                        + torch.roll(img, shifts=-1, dims=2)
                    )
                )
                z_img = stable.reshape(int(stable.shape[0]), -1)
            _u_aug, _s_aug, vh_aug = torch.linalg.svd(z_img - z_img.mean(dim=0, keepdim=True), full_matrices=False)
            top_aug = min(16, int(vh_aug.shape[0]))
            if top_aug > 0:
                aug_basis = vh_aug[:top_aug].transpose(0, 1).contiguous()
                out["P_energy_on_aug_stable_subspace"] = float((aug_basis.transpose(0, 1) @ p).square().sum().div(total).item())
        except RuntimeError:
            out["P_energy_on_aug_stable_subspace"] = ""
        try:
            s = torch.linalg.svdvals(p)
            good = s[s > 1.0e-7]
            out["P_condition"] = float((s.max() / good.min()).item()) if bool(good.numel()) else ""
            out["P_singular_min"] = float(good.min().item()) if bool(good.numel()) else ""
            out["P_singular_max"] = float(s.max().item()) if bool(s.numel()) else ""
        except RuntimeError:
            out["P_condition"] = ""
            out["P_singular_min"] = ""
            out["P_singular_max"] = ""
        qstd = getattr(model, "quad_feature_std", None)
        if torch.is_tensor(qstd):
            qstd_f = qstd.detach().float()
            out["quad_feature_std_mean"] = float(qstd_f.mean().item())
            out["quad_feature_std_min"] = float(qstd_f.min().item())
            out["quad_feature_std_max"] = float(qstd_f.max().item())
        for name in ["direct_readout", "quad_readout", "branch_scale", "logit_gain"]:
            value = getattr(model, name, None)
            if torch.is_tensor(value):
                out[f"{name}_norm"] = float(value.detach().float().norm().item())
        out["quad_proj_norm"] = float(p.norm().item())
        return out


def apply_unlabeled_projector_adaptation(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    seed: int,
    epoch: int,
    strength: float = 0.08,
) -> dict[str, Any]:
    if not hasattr(model, "quad_proj") or not hasattr(model, "mu") or not hasattr(model, "std"):
        return {"covadapt_applied": 0}
    quad_proj = getattr(model, "quad_proj")
    if not torch.is_tensor(quad_proj) or int(quad_proj.dim()) != 2:
        return {"covadapt_applied": 0}
    with torch.no_grad():
        p = quad_proj.detach()
        x = x_train[: min(1024, int(x_train.shape[0]))].float()
        z = ((x - getattr(model, "mu")) / getattr(model, "std")).clamp(-3.0, 3.0)
        z = z - z.mean(dim=0, keepdim=True)
        try:
            _u, _s, vh = torch.linalg.svd(z, full_matrices=False)
            target = torch.zeros_like(p)
            take = min(int(vh.shape[0]), int(target.shape[1]))
            if take > 0:
                target[:, :take] = vh[:take].transpose(0, 1).contiguous().to(dtype=target.dtype)
            gen = torch.Generator(device=x_train.device).manual_seed(int(seed) + 1777 + int(epoch))
            col = take
            while col < int(target.shape[1]):
                vec = torch.randn(int(target.shape[0]), device=x_train.device, generator=gen, dtype=target.dtype)
                if col > 0:
                    comps = target[:, :col]
                    vec = vec - comps @ (comps.transpose(0, 1) @ vec)
                vec = vec - vec.mean()
                target[:, col] = vec / vec.norm().clamp_min(1.0e-6)
                col += 1
            target = target - target.mean(dim=0, keepdim=True)
            target = target / target.norm(dim=0, keepdim=True).clamp_min(1.0e-6)
            mixed = p.mul(1.0 - float(strength)).add(target, alpha=float(strength))
            mixed = mixed - mixed.mean(dim=0, keepdim=True)
            mixed = mixed / mixed.norm(dim=0, keepdim=True).clamp_min(1.0e-6)
            quad_proj.copy_(mixed)
            overlap = float((target.transpose(0, 1) @ mixed).diag().abs().mean().item())
            return {"covadapt_applied": 1, "covadapt_strength": float(strength), "covadapt_target_overlap": overlap}
        except RuntimeError:
            return {"covadapt_applied": 0, "covadapt_strength": float(strength), "covadapt_target_overlap": ""}


def _trainable_param_snapshot(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter, torch.Tensor]]:
    snap: list[tuple[str, torch.nn.Parameter, torch.Tensor]] = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            snap.append((name, param, param.detach().clone()))
    return snap


def _update_role(name: str) -> str:
    if "quad_proj" in name:
        return "quad_proj"
    if "quad_readout" in name:
        return "quad_readout"
    if "direct_readout" in name:
        return "direct_readout"
    if "branch_scale" in name:
        return "branch_scale"
    if "logit_gain" in name:
        return "logit_gain"
    return "other"


def optimizer_update_observable_stats(
    snapshot: Sequence[tuple[str, torch.nn.Parameter, torch.Tensor]],
    prev_flat: torch.Tensor | None,
) -> tuple[dict[str, Any], torch.Tensor | None, torch.Tensor | None]:
    role_energy = {"quad_proj": 0.0, "quad_readout": 0.0, "direct_readout": 0.0, "branch_scale": 0.0, "logit_gain": 0.0, "other": 0.0}
    flat_chunks: list[torch.Tensor] = []
    quad_delta: torch.Tensor | None = None
    max_abs = 0.0
    abs_means: list[float] = []
    for name, param, before in snapshot:
        delta = (param.detach() - before).float()
        if not bool(torch.isfinite(delta).all()):
            continue
        role = _update_role(name)
        energy = float(delta.square().sum().item())
        role_energy[role] = role_energy.get(role, 0.0) + energy
        max_abs = max(max_abs, float(delta.abs().max().item()) if delta.numel() else 0.0)
        abs_means.append(float(delta.abs().mean().item()) if delta.numel() else 0.0)
        flat_chunks.append(delta.flatten().cpu())
        if role == "quad_proj":
            quad_delta = delta.detach().clone()
    if not flat_chunks:
        return {"t1b_native_logged": 0}, prev_flat, quad_delta
    flat = torch.cat(flat_chunks)
    norm = float(flat.norm().item())
    cosine = ""
    if prev_flat is not None and int(prev_flat.numel()) == int(flat.numel()) and norm > 0.0 and float(prev_flat.norm().item()) > 0.0:
        cosine = float(torch.dot(flat, prev_flat).div(flat.norm().clamp_min(EPS) * prev_flat.norm().clamp_min(EPS)).item())
    total_energy = max(EPS, sum(role_energy.values()))
    stats: dict[str, Any] = {
        "t1b_native_logged": 1,
        "t1b_update_total_norm": norm,
        "t1b_update_abs_mean": mean(abs_means) if abs_means else 0.0,
        "t1b_update_abs_max": max_abs,
        "t1b_update_cosine_prev": cosine,
    }
    for role, energy in role_energy.items():
        stats[f"t1b_update_{role}_norm"] = math.sqrt(max(0.0, energy))
        stats[f"t1b_update_{role}_energy_frac"] = float(energy / total_energy)
    if quad_delta is not None and int(quad_delta.dim()) == 2 and float(quad_delta.norm().item()) > 0.0:
        try:
            svals = torch.linalg.svdvals(quad_delta.float())
            total = svals.sum().clamp_min(EPS)
            positive = svals[svals > 1.0e-10]
            prob = svals / total
            stats["t1b_update_spectrum_top_share"] = float(svals[0].div(total).item()) if int(svals.numel()) else ""
            stats["t1b_update_spectrum_condition"] = float(svals[0].div(positive[-1].clamp_min(EPS)).item()) if bool(positive.numel()) else ""
            stats["t1b_update_spectrum_rank90"] = int(torch.searchsorted(torch.cumsum(svals, dim=0), 0.90 * total).item()) + 1 if int(svals.numel()) else 0
            stats["t1b_update_spectrum_entropy"] = (
                float((-(prob * prob.clamp_min(EPS).log()).sum() / math.log(max(2, int(prob.numel())))).item())
                if int(prob.numel())
                else ""
            )
        except RuntimeError:
            stats["t1b_update_spectrum_top_share"] = ""
            stats["t1b_update_spectrum_condition"] = ""
            stats["t1b_update_spectrum_rank90"] = ""
            stats["t1b_update_spectrum_entropy"] = ""
    return stats, flat, quad_delta


def apply_optimizer_update_projector_adaptation(
    model: torch.nn.Module,
    update_frame: torch.Tensor | None,
    strength: float = 0.06,
) -> dict[str, Any]:
    if update_frame is None or not hasattr(model, "quad_proj"):
        return {"optframe_applied": 0}
    quad_proj = getattr(model, "quad_proj")
    if not torch.is_tensor(quad_proj) or int(quad_proj.numel()) == 0 or tuple(update_frame.shape) != tuple(quad_proj.shape):
        return {"optframe_applied": 0}
    with torch.no_grad():
        if float(update_frame.float().norm().item()) <= 1.0e-10:
            return {"optframe_applied": 0}
        target = update_frame.detach().to(device=quad_proj.device, dtype=quad_proj.dtype)
        target = target - target.mean(dim=0, keepdim=True)
        target = target / target.norm(dim=0, keepdim=True).clamp_min(1.0e-6)
        mixed = quad_proj.detach().mul(1.0 - float(strength)).add(target, alpha=float(strength))
        mixed = mixed - mixed.mean(dim=0, keepdim=True)
        mixed = mixed / mixed.norm(dim=0, keepdim=True).clamp_min(1.0e-6)
        quad_proj.copy_(mixed)
        if hasattr(model, "_calibrate_quadratic_feature_norm"):
            try:
                model._calibrate_quadratic_feature_norm(getattr(model, "_v1221_calibration_x"))
            except Exception:
                pass
        return {"optframe_applied": 1, "optframe_strength": float(strength), "optframe_update_norm": float(update_frame.float().norm().item())}


def apply_persistent_drift_projector_adaptation(
    model: torch.nn.Module,
    x_probe: torch.Tensor,
    prev_logits: torch.Tensor | None,
    strength: float = 0.04,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    if prev_logits is None or not hasattr(model, "quad_proj") or not hasattr(model, "mu") or not hasattr(model, "std"):
        with torch.no_grad():
            return {"persistentdrift_applied": 0}, model(x_probe).detach()
    quad_proj = getattr(model, "quad_proj")
    if not torch.is_tensor(quad_proj) or int(quad_proj.dim()) != 2:
        with torch.no_grad():
            return {"persistentdrift_applied": 0}, model(x_probe).detach()
    with torch.no_grad():
        logits = model(x_probe).detach()
        drift = logits - prev_logits.to(device=logits.device, dtype=logits.dtype)
        if float(drift.float().norm().item()) <= 1.0e-10:
            return {"persistentdrift_applied": 0}, logits
        z = ((x_probe - getattr(model, "mu")) / getattr(model, "std")).clamp(-3.0, 3.0)
        z = z - z.mean(dim=0, keepdim=True)
        response = z.transpose(0, 1) @ drift.float()
        try:
            u, _s, _vh = torch.linalg.svd(response.float(), full_matrices=False)
            target = torch.zeros_like(quad_proj)
            take = min(int(u.shape[1]), int(target.shape[1]))
            if take > 0:
                target[:, :take] = u[:, :take].to(device=target.device, dtype=target.dtype)
            col = take
            gen = torch.Generator(device=quad_proj.device).manual_seed(int(930021 + take + target.shape[0] + target.shape[1]))
            while col < int(target.shape[1]):
                vec = torch.randn(int(target.shape[0]), device=quad_proj.device, generator=gen, dtype=target.dtype)
                if col > 0:
                    comps = target[:, :col]
                    vec = vec - comps @ (comps.transpose(0, 1) @ vec)
                vec = vec - vec.mean()
                target[:, col] = vec / vec.norm().clamp_min(1.0e-6)
                col += 1
            target = target - target.mean(dim=0, keepdim=True)
            target = target / target.norm(dim=0, keepdim=True).clamp_min(1.0e-6)
            mixed = quad_proj.detach().mul(1.0 - float(strength)).add(target, alpha=float(strength))
            mixed = mixed - mixed.mean(dim=0, keepdim=True)
            mixed = mixed / mixed.norm(dim=0, keepdim=True).clamp_min(1.0e-6)
            quad_proj.copy_(mixed)
            if hasattr(model, "_calibrate_quadratic_feature_norm"):
                try:
                    model._calibrate_quadratic_feature_norm(getattr(model, "_v1221_calibration_x"))
                except Exception:
                    pass
            new_logits = model(x_probe).detach()
            return {
                "persistentdrift_applied": 1,
                "persistentdrift_strength": float(strength),
                "persistentdrift_logit_delta_norm": float(drift.float().norm().item()),
            }, new_logits
        except RuntimeError:
            return {"persistentdrift_applied": 0, "persistentdrift_strength": float(strength)}, logits


def step_model(
    model: torch.nn.Module,
    method_id: str,
    specs: Mapping[str, prim.PrimitiveSpec],
    xb: torch.Tensor,
    yb: torch.Tensor,
    opt: torch.optim.Optimizer,
    args: argparse.Namespace,
    manual_update: Any,
    epoch: int,
    step_id: int,
    workspace: Mapping[str, torch.Tensor] | None,
) -> str:
    if method_id in MLP_CONTROL_IDS:
        impl = "torch-autograd"
    else:
        impl = v1283._select_b109_step_impl(method_id, specs, epoch_idx=epoch, step_id=step_id)
    v1283._step_b109(model, xb, yb, impl, workspace, opt=opt, args=args, manual_update=manual_update)
    return impl


def train_one(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    method_id: str,
    spec: prim.PrimitiveSpec | None,
    uses_y_for_stats: int,
    device: torch.device,
    y_stats_mode: str = "actual",
) -> dict[str, Any]:
    load_args = argparse.Namespace(**vars(args))
    load_args.seed = int(seed)
    data = v120._load_vision_split(
        load_args,
        dataset,
        train_size=int(args.train_size),
        val_size=int(args.val_size),
        test_size=int(args.test_size),
    )
    x_train_cpu, y_train_cpu, x_val_cpu, y_val_cpu, x_test_cpu, y_test_cpu, input_dim, output_dim, _protocol = data
    x_train = x_train_cpu.to(device=device, dtype=torch.float32)
    y_train = y_train_cpu.to(device=device)
    x_val = x_val_cpu.to(device=device, dtype=torch.float32)
    y_val = y_val_cpu.to(device=device)
    x_test = x_test_cpu.to(device=device, dtype=torch.float32)
    y_test = y_test_cpu.to(device=device)
    specs = {method_id: spec} if spec is not None else {}
    model_seed = int(seed) + int(args.seed_base)
    model = make_model(method_id, spec, int(input_dim), int(output_dim), x_train, y_train, device, model_seed, uses_y_for_stats, y_stats_mode)
    if hasattr(model, "_calibrate_quadratic_feature_norm"):
        setattr(model, "_v1221_calibration_x", x_train[: min(2048, int(x_train.shape[0]))])
    opt = v1283._make_adamw(model, args)
    triton_update_params = v126._triton_adamw_params(args, model)
    manual_update = (
        v126._ManualForeachAdamW(opt.param_groups, args, triton_update_params=triton_update_params)
        if v126._variant_uses_manual_adamw(args, model)
        else None
    )
    warm_impl = "torch-autograd" if method_id in MLP_CONTROL_IDS else v1283._select_b109_step_impl(method_id, specs, 0, 0)
    workspace = fhq.make_workspace(model, int(args.batch_size), device) if warm_impl in {"F3-triton-learnableP-workspace", "F4-triton-fixedP-workspace"} else None
    if workspace is not None:
        warm = min(int(args.batch_size), int(x_train.shape[0]))
        for warm_idx in range(max(0, int(args.task_compile_warmup_steps))):
            start_idx = (warm_idx * warm) % int(x_train.shape[0])
            xb = x_train[start_idx : start_idx + warm]
            yb = y_train[start_idx : start_idx + warm]
            if int(xb.shape[0]) < warm:
                xb = x_train[:warm]
                yb = y_train[:warm]
            opt.zero_grad(set_to_none=True)
            v1283._step_b109(model, xb, yb, warm_impl, workspace, opt=opt, args=args, manual_update=manual_update, allow_fused_update=False)
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
    total_steps = max(1, int(args.epochs) * int(math.ceil(float(x_train.shape[0]) / float(max(1, int(args.batch_size))))))
    gen = torch.Generator(device=device).manual_seed(int(seed) + int(args.seed_base) + 33)
    step_id = 0
    step_times: list[float] = []
    val_losses: list[float] = []
    val_times_q90: list[float] = []
    output_cov_drifts: list[float] = []
    impls: list[str] = []
    covadapt_events = 0
    covadapt_last: dict[str, Any] = {}
    optframe_events = 0
    optframe_last: dict[str, Any] = {}
    persistentdrift_events = 0
    persistentdrift_last: dict[str, Any] = {}
    update_norms: list[float] = []
    update_abs_means: list[float] = []
    update_abs_maxes: list[float] = []
    update_cosines: list[float] = []
    update_role_energy_sum = {"quad_proj": 0.0, "quad_readout": 0.0, "direct_readout": 0.0, "branch_scale": 0.0, "logit_gain": 0.0, "other": 0.0}
    update_spectrum_top_share: list[float] = []
    update_spectrum_condition: list[float] = []
    update_spectrum_rank90: list[float] = []
    update_spectrum_entropy: list[float] = []
    update_role_energy_values = {role: [] for role in update_role_energy_sum.keys()}
    prev_update_flat: torch.Tensor | None = None
    optframe_accum: torch.Tensor | None = None
    drift_probe = x_train[: min(256, int(x_train.shape[0]))]
    with torch.no_grad():
        drift_prev_logits = model(drift_probe).detach() if "persistentdriftp" in (str(spec.init_variant).lower() if spec is not None else "") else None
        cov_probe = x_val[: min(256, int(x_val.shape[0]))]
        logits0 = model(cov_probe).detach().float()
        logits0 = logits0 - logits0.mean(dim=0, keepdim=True)
        prev_output_cov = logits0.transpose(0, 1) @ logits0 / float(max(1, int(logits0.shape[0]) - 1))
    spec_variant = str(spec.init_variant).lower() if spec is not None else ""
    for epoch in range(int(args.epochs)):
        if "covadaptp" in spec_variant:
            covadapt_last = apply_unlabeled_projector_adaptation(model, x_train, model_seed, epoch, float(args.covadapt_strength))
            covadapt_events += int(covadapt_last.get("covadapt_applied", 0))
        v1283._apply_logit_gain_ramp(model, method_id, specs, epoch, int(args.epochs))
        v1283._apply_quad_branch_ramp(model, method_id, specs, epoch, int(args.epochs))
        v1283._apply_direct_branch_ramp(model, method_id, specs, epoch, int(args.epochs))
        epoch_start = len(step_times)
        perm = torch.randperm(int(x_train.shape[0]), generator=gen, device=device)
        for off in range(0, int(x_train.shape[0]), int(args.batch_size)):
            idx = perm[off : off + int(args.batch_size)]
            xb = x_train[idx]
            yb = y_train[idx]
            lr_now = v126._task_lr_for(args, step_id + 1, total_steps)
            for group in opt.param_groups:
                lr_scale = float(group.get("lr_scale", 1.0))
                for switch_epoch, switch_scale in group.get("lr_scale_schedule", []):
                    if epoch >= int(switch_epoch):
                        lr_scale = float(switch_scale)
                if not group.get("lr_scale_schedule") and group.get("lr_scale_after", None) is not None and epoch >= int(group.get("lr_scale_switch_epoch", 0)):
                    lr_scale = float(group.get("lr_scale_after", lr_scale))
                group["lr"] = lr_now * lr_scale
            opt.zero_grad(set_to_none=True)
            update_snapshot = _trainable_param_snapshot(model)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            impl = step_model(model, method_id, specs, xb, yb, opt, args, manual_update, epoch, step_id, workspace)
            if manual_update is not None:
                manual_update.step()
            else:
                opt.step()
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            update_stats, prev_update_flat, quad_delta = optimizer_update_observable_stats(update_snapshot, prev_update_flat)
            if safe_int(update_stats.get("t1b_native_logged"), 0):
                update_norms.append(safe_float(update_stats.get("t1b_update_total_norm")))
                update_abs_means.append(safe_float(update_stats.get("t1b_update_abs_mean")))
                update_abs_maxes.append(safe_float(update_stats.get("t1b_update_abs_max")))
                if update_stats.get("t1b_update_cosine_prev") not in ("", None):
                    update_cosines.append(safe_float(update_stats.get("t1b_update_cosine_prev")))
                for role in update_role_energy_sum.keys():
                    frac = safe_float(update_stats.get(f"t1b_update_{role}_energy_frac"), 0.0)
                    update_role_energy_sum[role] += float(frac)
                    update_role_energy_values[role].append(float(frac))
                if update_stats.get("t1b_update_spectrum_top_share") not in ("", None):
                    update_spectrum_top_share.append(safe_float(update_stats.get("t1b_update_spectrum_top_share")))
                if update_stats.get("t1b_update_spectrum_condition") not in ("", None):
                    update_spectrum_condition.append(safe_float(update_stats.get("t1b_update_spectrum_condition")))
                if update_stats.get("t1b_update_spectrum_rank90") not in ("", None):
                    update_spectrum_rank90.append(safe_float(update_stats.get("t1b_update_spectrum_rank90")))
                if update_stats.get("t1b_update_spectrum_entropy") not in ("", None):
                    update_spectrum_entropy.append(safe_float(update_stats.get("t1b_update_spectrum_entropy")))
            if quad_delta is not None and ("optframep" in spec_variant):
                optframe_accum = quad_delta.detach().clone() if optframe_accum is None else optframe_accum.add(quad_delta.detach())
            impls.append(impl)
            step_times.append((t1 - t0) * 1000.0)
            step_id += 1
        val = v1252._classification_basic(model, x_val, y_val)
        val_losses.append(float(val["NLL"]))
        val_times_q90.append(q(step_times[epoch_start:], 0.90))
        with torch.no_grad():
            logits_cov = model(cov_probe).detach().float()
            logits_cov = logits_cov - logits_cov.mean(dim=0, keepdim=True)
            output_cov = logits_cov.transpose(0, 1) @ logits_cov / float(max(1, int(logits_cov.shape[0]) - 1))
            output_cov_drifts.append(float((output_cov - prev_output_cov).norm().item()))
            prev_output_cov = output_cov
        if "optframep" in spec_variant and epoch < int(args.epochs) - 1:
            optframe_last = apply_optimizer_update_projector_adaptation(model, optframe_accum, float(args.optframe_strength))
            optframe_events += int(optframe_last.get("optframe_applied", 0))
            optframe_accum = None
        if "persistentdriftp" in spec_variant and epoch < int(args.epochs) - 1:
            persistentdrift_last, drift_prev_logits = apply_persistent_drift_projector_adaptation(
                model,
                drift_probe,
                drift_prev_logits,
                float(args.persistentdrift_strength),
            )
            persistentdrift_events += int(persistentdrift_last.get("persistentdrift_applied", 0))
        if "adaptframeschedp" in spec_variant and epoch >= 1 and epoch < int(args.epochs) - 1:
            adapt_strength = float(args.adaptframesched_strength)
            adapt_match = re.search(r"adaptframeschedp(\d{3})", spec_variant)
            if adapt_match is not None:
                adapt_strength = float(int(adapt_match.group(1))) / 1000.0
            covadapt_last = apply_unlabeled_projector_adaptation(model, x_train, model_seed, epoch, adapt_strength)
            covadapt_events += int(covadapt_last.get("covadapt_applied", 0))
    final = v1252._classification_basic(model, x_val, y_val)
    test = v1252._classification_basic(model, x_test, y_test)
    steady_start = min(max(0, int(args.task_timing_warmup_epochs)), len(val_losses) - 1)
    steady_losses = val_losses[steady_start:]
    steady_times = val_times_q90[steady_start:]
    auc_step = sum(steady_losses) / max(1, len(steady_losses))
    auc_time = sum(loss * max(1.0, t) for loss, t in zip(steady_losses, steady_times)) / max(1, len(steady_losses))
    half_life_step = ""
    if update_norms and update_norms[0] > EPS:
        threshold = 0.5 * update_norms[0]
        half_life_step = next((idx for idx, value in enumerate(update_norms, start=1) if value <= threshold), len(update_norms))
    role_transition_l1 = ""
    if update_norms:
        first = [values[0] for values in update_role_energy_values.values() if values]
        last = [values[-1] for values in update_role_energy_values.values() if values]
        if first and len(first) == len(last):
            role_transition_l1 = sum(abs(a - b) for a, b in zip(first, last))
    cosine_transition = ""
    if len(update_cosines) >= 2:
        cosine_transition = mean([abs(b - a) for a, b in zip(update_cosines[:-1], update_cosines[1:])])
    entropy_delta = ""
    if len(update_spectrum_entropy) >= 2:
        entropy_delta = update_spectrum_entropy[-1] - update_spectrum_entropy[0]
    linec: dict[str, Any] = {}
    if int(args.measure_linec):
        b = min(int(args.linec_batch_size), int(x_train.shape[0]) // 2, int(x_val.shape[0]))
        s = min(int(args.linec_sketch_batch_size), b)
        xb = x_train[:b]
        yb = y_train[:b]
        xq = x_val[:b]
        yq = y_val[:b]
        model.eval()
        with torch.no_grad():
            before_b = model(xb).detach()
            before_q = model(xq).detach()
        updated = v1252._take_adamw_window(model, xb, yb, float(args.lr), float(args.weight_decay)).eval()
        with torch.no_grad():
            after_b = updated(xb).detach()
            after_q = updated(xq).detach()
        r2, corr, resid, pred_norm = v1252._ridge_coupling(after_b - before_b, after_q - before_q, float(args.linec_ridge_lambda))
        sig = signal_reservoir_metrics_detailed(
            updated,
            xb[:s],
            yb[:s],
            int(args.linec_sketch_dim),
            int(seed) + int(args.seed_base) + 402,
        )
        linec = {
            "linec_measured": 1,
            "linec_batch_size": b,
            "linec_sketch_batch_size": s,
            "linec_sketch_dim": int(args.linec_sketch_dim),
            "linec_CouplingR2": r2,
            "linec_CouplingCorr": corr,
            "linec_coupling_residual_norm": resid,
            "linec_coupling_prediction_norm": pred_norm,
            "linec_NoiseSignalLeak": sig["NoiseSignalLeak"],
            "linec_RealSignalReservoirRatio": sig["RealSignalReservoirRatio"],
            "linec_signal_effective_rank": sig["signal_effective_rank"],
            "linec_signal_mass_topk": sig["signal_mass_topk"],
            "linec_reservoir_fraction": sig["reservoir_fraction"],
            "linec_top_eigen_share": sig["top_eigen_share"],
            "linec_dissipation_condition": sig["dissipation_condition"],
            "linec_SNR_positive_fraction": sig["SNR_positive_fraction"],
            "linec_real_noise_gap": sig["real_noise_gap"],
            "linec_signal_top_count": sig["signal_top_count"],
            "linec_real_residual_energy": sig["real_residual_energy"],
            "linec_real_total_energy": sig["real_total_energy"],
            "linec_noise_signal_energy": sig["noise_signal_energy"],
            "linec_noise_total_energy": sig["noise_total_energy"],
            "linec_label_used_for_audit_only": 1,
            "linec_label_used_for_direction": 0,
            "linec_ce_vector_used_for_direction": 0,
        }
    return {
        "stage": str(args.result_stage),
        "result_scope": str(args.result_scope),
        "candidate_id": method_id,
        "dataset": dataset,
        "seed": seed,
        "train_size": int(args.train_size),
        "val_size": int(args.val_size),
        "test_size": int(args.test_size),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "uses_y_for_stats": int(uses_y_for_stats),
        "y_stats_mode": str(y_stats_mode),
        "uses_real_y_for_stats": int(int(uses_y_for_stats) == 1 and str(y_stats_mode) == "actual"),
        "trainprobe_signal_init_uses_labels_buffer": buffer_int(model, "trainprobe_signal_init_uses_labels"),
        "trainprobe_signal_init_applied_buffer": buffer_int(model, "trainprobe_signal_init_applied"),
        "strict_label_free_init": int(method_id.startswith("A") and method_id != "A0-labelInit" and int(uses_y_for_stats) == 0 and buffer_int(model, "trainprobe_signal_init_applied") == 0),
        "val_acc": final["acc"],
        "test_acc": test["acc"],
        "NLL": final["NLL"],
        "ECE": final["ECE"],
        "CEp99": final["CEp99"],
        "margin_p10": final["margin_p10"],
        "val_loss_auc_step": auc_step,
        "val_loss_auc_time": auc_time,
        "step_time_q90_ms": q(step_times, 0.90),
        "task_step_impl": "+".join(sorted(set(impls))),
        "task_update_impl": "manual_or_fused_adamw" if manual_update is not None else "adamw",
        "label_free_covadapt_events": covadapt_events,
        "label_free_covadapt_applied_last": covadapt_last.get("covadapt_applied", 0) if covadapt_last else 0,
        "label_free_covadapt_strength": covadapt_last.get("covadapt_strength", "") if covadapt_last else "",
        "label_free_covadapt_target_overlap": covadapt_last.get("covadapt_target_overlap", "") if covadapt_last else "",
        "label_free_optframe_events": optframe_events,
        "label_free_optframe_applied_last": optframe_last.get("optframe_applied", 0) if optframe_last else 0,
        "label_free_optframe_strength": optframe_last.get("optframe_strength", "") if optframe_last else "",
        "label_free_optframe_update_norm_last": optframe_last.get("optframe_update_norm", "") if optframe_last else "",
        "label_free_persistentdrift_events": persistentdrift_events,
        "label_free_persistentdrift_applied_last": persistentdrift_last.get("persistentdrift_applied", 0) if persistentdrift_last else 0,
        "label_free_persistentdrift_strength": persistentdrift_last.get("persistentdrift_strength", "") if persistentdrift_last else "",
        "label_free_persistentdrift_logit_delta_norm_last": persistentdrift_last.get("persistentdrift_logit_delta_norm", "") if persistentdrift_last else "",
        "t1b_optimizer_update_native_logged": int(bool(update_norms)),
        "t1b_update_total_norm_mean": mean(update_norms) if update_norms else "",
        "t1b_update_total_norm_q90": q(update_norms, 0.90, default=0.0) if update_norms else "",
        "t1b_update_total_norm_last": update_norms[-1] if update_norms else "",
        "t1b_update_abs_mean": mean(update_abs_means) if update_abs_means else "",
        "t1b_update_abs_max": max(update_abs_maxes) if update_abs_maxes else "",
        "t1b_update_cosine_mean": mean(update_cosines) if update_cosines else "",
        "t1b_update_spectrum_top_share_mean": mean(update_spectrum_top_share) if update_spectrum_top_share else "",
        "t1b_update_spectrum_condition_mean": mean(update_spectrum_condition) if update_spectrum_condition else "",
        "t1b_update_spectrum_rank90_mean": mean(update_spectrum_rank90) if update_spectrum_rank90 else "",
        "t1b_update_spectrum_entropy_mean": mean(update_spectrum_entropy) if update_spectrum_entropy else "",
        "t1b_update_spectrum_entropy_delta": entropy_delta,
        "t1b_update_half_life_step": half_life_step,
        "t1b_update_half_life_frac": (safe_float(half_life_step, 0.0) / max(1, len(update_norms)) if update_norms and half_life_step != "" else ""),
        "t1b_update_cosine_transition_abs_mean": cosine_transition,
        "t1b_update_role_energy_transition_l1": role_transition_l1,
        "t1b_update_quad_direct_energy_flow_mean": (
            (update_role_energy_sum["quad_proj"] + update_role_energy_sum["quad_readout"] - update_role_energy_sum["direct_readout"]) / max(1, len(update_norms))
            if update_norms
            else ""
        ),
        **{f"t1b_update_{role}_energy_frac_mean": (update_role_energy_sum[role] / max(1, len(update_norms)) if update_norms else "") for role in update_role_energy_sum.keys()},
        "g_cov_output_cov_drift_mean": mean(output_cov_drifts) if output_cov_drifts else "",
        "g_cov_output_cov_drift_max": max(output_cov_drifts) if output_cov_drifts else "",
        "smoke_not_official": int(args.smoke_not_official),
        "official_training_result_available": int(args.official_training_result_available),
        "protocol_note": str(args.protocol_note),
        **linec,
        "no_fake": 1,
        "no_proxy": 1,
        "cpu_offload_used": 0,
        **projector_alignment_diagnostics(model, x_train),
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["dataset"], str(row["seed"])): row for row in rows if row.get("candidate_id") == MLP_SAME_PARAM_ID}
    a0_key = {(row["dataset"], str(row["seed"])): row for row in rows if row.get("candidate_id") == "A0-labelInit"}
    enriched: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        base = by_key.get((row["dataset"], str(row["seed"])))
        a0 = a0_key.get((row["dataset"], str(row["seed"])))
        if base and row.get("candidate_id") != MLP_SAME_PARAM_ID:
            row_acc = finite_float_or_none(row.get("val_acc"))
            base_acc = finite_float_or_none(base.get("val_acc"))
            if row_acc is not None and base_acc is not None:
                out["val_acc_delta_vs_mlp"] = row_acc - base_acc
            row_auc_step = finite_float_or_none(row.get("val_loss_auc_step"))
            base_auc_step = finite_float_or_none(base.get("val_loss_auc_step"))
            if row_auc_step is not None and base_auc_step is not None and base_auc_step > EPS:
                out["AUC_step_ratio_vs_mlp"] = row_auc_step / base_auc_step
            row_auc_time = finite_float_or_none(row.get("val_loss_auc_time"))
            base_auc_time = finite_float_or_none(base.get("val_loss_auc_time"))
            if row_auc_time is not None and base_auc_time is not None and base_auc_time > EPS:
                out["AUC_time_ratio_vs_mlp"] = row_auc_time / base_auc_time
            row_ece = finite_float_or_none(row.get("ECE"))
            base_ece = finite_float_or_none(base.get("ECE"))
            if row_ece is not None and base_ece is not None:
                out["ECE_delta_vs_mlp"] = row_ece - base_ece
        if a0 and row.get("candidate_id") not in {MLP_SAME_PARAM_ID, MLP_SAME_STEP_FLOP_ID, "A0-labelInit"}:
            row_acc = finite_float_or_none(row.get("val_acc"))
            a0_acc = finite_float_or_none(a0.get("val_acc"))
            if row_acc is not None and a0_acc is not None:
                out["val_acc_delta_vs_A0_labelInit"] = row_acc - a0_acc
            row_nll = finite_float_or_none(row.get("NLL"))
            a0_nll = finite_float_or_none(a0.get("NLL"))
            if row_nll is not None and a0_nll is not None:
                out["NLL_delta_vs_A0_labelInit"] = row_nll - a0_nll
        if base and row.get("candidate_id") != MLP_SAME_PARAM_ID and int(safe_float(row.get("linec_measured"), 0)) == 1 and int(safe_float(base.get("linec_measured"), 0)) == 1:
            coupling_ok = safe_float(row.get("linec_CouplingR2"), -999.0) >= safe_float(base.get("linec_CouplingR2"), 0.0) - 0.02
            noise_ok = safe_float(row.get("linec_NoiseSignalLeak"), 999.0) <= safe_float(base.get("linec_NoiseSignalLeak"), 0.0) + 0.02
            reservoir_ok = safe_float(row.get("linec_RealSignalReservoirRatio"), 999.0) <= safe_float(base.get("linec_RealSignalReservoirRatio"), 0.0) + 0.02
            out["linec_nontearing_pass_vs_mlp"] = int(coupling_ok and noise_ok and reservoir_ok)
            out["linec_nontearing_fail_reason"] = ";".join(
                name
                for name, ok in [
                    ("CouplingR2_below_mlp_minus_0.02", coupling_ok),
                    ("NoiseSignalLeak_above_mlp_plus_0.02", noise_ok),
                    ("RealSignalReservoirRatio_above_mlp_plus_0.02", reservoir_ok),
                ]
                if not ok
            )
        enriched.append(out)
    summary_rows: list[dict[str, Any]] = []
    for method in sorted({str(row.get("candidate_id")) for row in enriched}):
        group = [row for row in enriched if row.get("candidate_id") == method]
        if not group:
            continue
        deltas = [safe_float(row.get("val_acc_delta_vs_mlp")) for row in group if row.get("val_acc_delta_vs_mlp") not in (None, "")]
        a0_deltas = [safe_float(row.get("val_acc_delta_vs_A0_labelInit")) for row in group if row.get("val_acc_delta_vs_A0_labelInit") not in (None, "")]
        auc_step = [safe_float(row.get("AUC_step_ratio_vs_mlp")) for row in group if row.get("AUC_step_ratio_vs_mlp") not in (None, "")]
        auc_time = [safe_float(row.get("AUC_time_ratio_vs_mlp")) for row in group if row.get("AUC_time_ratio_vs_mlp") not in (None, "")]
        ece_delta = [safe_float(row.get("ECE_delta_vs_mlp")) for row in group if row.get("ECE_delta_vs_mlp") not in (None, "")]
        linec_passes = [safe_float(row.get("linec_nontearing_pass_vs_mlp")) for row in group if row.get("linec_nontearing_pass_vs_mlp") not in (None, "")]
        summary_rows.append(
            {
                "stage": "V1218_B320_LABEL_FREE_SMOKE_SUMMARY",
                "result_scope": str(getattr(group[0], "result_scope", "")) if hasattr(group[0], "result_scope") else group[0].get("result_scope", ""),
                "candidate_id": method,
                "rows": len(group),
                "datasets": ",".join(sorted({str(row.get("dataset")) for row in group})),
                "seeds": ",".join(sorted({str(row.get("seed")) for row in group})),
                "mean_delta_vs_mlp": mean(deltas) if deltas else "",
                "worst_delta_vs_mlp": min(deltas) if deltas else "",
                "near_pass_rate_vs_mlp": mean([1.0 if d >= -0.010 else 0.0 for d in deltas]) if deltas else "",
                "max_AUC_step_ratio_vs_mlp": max(auc_step) if auc_step else "",
                "max_AUC_time_ratio_vs_mlp": max(auc_time) if auc_time else "",
                "max_ECE_delta_vs_mlp": max(ece_delta) if ece_delta else "",
                "mean_delta_vs_A0_labelInit": mean(a0_deltas) if a0_deltas else "",
                "worst_delta_vs_A0_labelInit": min(a0_deltas) if a0_deltas else "",
                "linec_nontearing_rows": len(linec_passes),
                "linec_nontearing_pass_rate": mean(linec_passes) if linec_passes else "",
                "linec_nontearing_all_pass": int(bool(linec_passes) and min(linec_passes) >= 1.0) if linec_passes else "",
                "strict_label_free_rows": sum(int(row.get("strict_label_free_init", 0)) for row in group),
                "smoke_not_official": group[0].get("smoke_not_official", 1),
                "official_training_result_available": group[0].get("official_training_result_available", 0),
                "protocol_note": group[0].get("protocol_note", ""),
                "no_fake": 1,
                "no_proxy": 1,
                "cpu_offload_used": 0,
            }
        )
    rows[:] = enriched  # type: ignore[index]
    return summary_rows


def run_main(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)
    device = device_from_arg(args.device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    datasets = [v120._canonical_dataset(name) for name in parse_list(args.datasets)]
    seeds = parse_ints(args.seeds)
    all_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        probe_args = argparse.Namespace(**vars(args))
        probe_args.dataset = dataset
        probe_args.seed = int(seeds[0]) if seeds else 0
        data = v120._load_vision_split(
            probe_args,
            dataset,
            train_size=int(args.train_size),
            val_size=int(args.val_size),
            test_size=int(args.test_size),
        )
        input_dim = int(data[6])
        output_dim = int(data[7])
        specs = ablation_specs(input_dim, output_dim)
        methods: list[tuple[str, prim.PrimitiveSpec | None, int, str]] = [(MLP_SAME_PARAM_ID, None, 0, "actual"), (MLP_SAME_STEP_FLOP_ID, None, 0, "actual")]
        for ablation_id in parse_list(args.ablation_ids):
            if ablation_id not in specs:
                raise KeyError(f"unknown ablation id: {ablation_id}")
            item = specs[ablation_id]
            methods.append((ablation_id, item["spec"], int(item["uses_y_for_stats"]), str(item.get("y_stats_mode", "actual"))))
        for seed in seeds:
            for method_id, spec, uses_y_for_stats, y_stats_mode in methods:
                all_rows.append(train_one(args, dataset, int(seed), method_id, spec, uses_y_for_stats, device, y_stats_mode))
    summary_rows = summarize(all_rows)
    for row in summary_rows:
        row["stage"] = str(args.summary_stage)
    prefix = str(args.artifact_prefix)
    write_csv_rows(out_dir / f"{prefix}_ablation.csv", all_rows)
    write_csv_rows(out_dir / f"{prefix}_summary.csv", summary_rows)
    label_free = [
        row
        for row in summary_rows
        if str(row.get("candidate_id", "")).startswith("A")
        and row.get("candidate_id") != "A0-labelInit"
        and safe_int(row.get("strict_label_free_rows"), 0) > 0
    ]
    best = max(label_free, key=lambda row: (safe_float(row.get("mean_delta_vs_A0_labelInit"), -999.0), safe_float(row.get("mean_delta_vs_mlp"), -999.0)), default={})
    payload = {
        "stage": str(args.route_stage),
        "generated_at": now_iso(),
        "run_id": args.run_id,
        "artifact_prefix": prefix,
        "result_scope": str(args.result_scope),
        "datasets": datasets,
        "seeds": seeds,
        "train_size": int(args.train_size),
        "val_size": int(args.val_size),
        "test_size": int(args.test_size),
        "epochs": int(args.epochs),
        "rows": len(all_rows),
        "summary_rows": len(summary_rows),
        "label_free_ablation_available_count": len(label_free),
        "best_label_free_candidate": best.get("candidate_id", ""),
        "best_label_free_mean_delta_vs_A0": best.get("mean_delta_vs_A0_labelInit", ""),
        "smoke_not_official": int(args.smoke_not_official),
        "official_training_result_available": int(args.official_training_result_available),
        "protocol_note": str(args.protocol_note),
        "route_impact": str(args.route_impact),
        "no_fake": 1,
        "no_proxy": 1,
        "cpu_offload_used": 0,
    }
    if "anchor_budget" in prefix:
        payload.update(
            {
                "label_free_anchor_budget_ablation_available_count": len(label_free),
                "best_label_free_anchor_budget_candidate": best.get("candidate_id", ""),
                "best_label_free_anchor_budget_mean_delta_vs_A0": best.get("mean_delta_vs_A0_labelInit", ""),
            }
        )
    else:
        payload.update(
            {
                "label_free_smoke_ablation_available_count": len(label_free),
                "best_label_free_smoke_candidate": best.get("candidate_id", ""),
                "best_label_free_smoke_mean_delta_vs_A0": best.get("mean_delta_vs_A0_labelInit", ""),
            }
        )
    write_json(out_dir / f"{prefix}_route.json", payload)
    route_path = out_dir / "v1218_route_decision.json"
    route = read_json(route_path)
    if route:
        if "anchor_budget" in prefix:
            route.update(
                {
                    "b320_label_free_anchor_budget_ablation_available_count": len(label_free),
                    "b320_label_free_anchor_budget_best_candidate": best.get("candidate_id", ""),
                    "b320_label_free_anchor_budget_best_mean_delta_vs_A0": best.get("mean_delta_vs_A0_labelInit", ""),
                    "b320_label_free_anchor_budget_official_training_result_available": int(args.official_training_result_available),
                    "b320_label_free_anchor_budget_protocol_note": str(args.protocol_note),
                }
            )
        else:
            route.update(
                {
                    "b320_label_free_smoke_ablation_available_count": len(label_free),
                    "b320_label_free_smoke_best_candidate": best.get("candidate_id", ""),
                    "b320_label_free_smoke_best_mean_delta_vs_A0": best.get("mean_delta_vs_A0_labelInit", ""),
                    "b320_label_free_smoke_not_official": int(args.smoke_not_official),
                    "b320_label_free_official_ablation_still_missing": 1,
                }
            )
        write_json(route_path, route)
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="label_free_smoke_seed0_3x3")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--artifact-prefix", default="v1218_b320_label_free_smoke")
    parser.add_argument("--result-stage", default="V1218_B320_LABEL_FREE_SMOKE_ABLATION")
    parser.add_argument("--summary-stage", default="V1218_B320_LABEL_FREE_SMOKE_SUMMARY")
    parser.add_argument("--route-stage", default="V1218_B320_LABEL_FREE_SMOKE_ROUTE")
    parser.add_argument("--result-scope", default="small_budget_smoke")
    parser.add_argument("--smoke-not-official", type=int, default=1)
    parser.add_argument("--official-training-result-available", type=int, default=0)
    parser.add_argument("--protocol-note", default="small-budget smoke; not official anchor protocol")
    parser.add_argument("--route-impact", default="does_not_close_R2; full comparable label-free B320 budget still missing")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--ablation-ids", default="A0-labelInit,A1-noYForStats,A2-randomP-labelFree,A3-PCA-P-labelFree,A4-lowfreqP-labelFree,A5-orthogonalP-labelFree,A6-orthogonalP-active64-labelFree,A7-orthogonalP-lowQuad-labelFree,A8-orthogonalP-boundQ-labelFree,A9-orthogonalP-lowQuad-boundQ-labelFree,A10-orthogonalP-strongLowQuad-boundQ-labelFree,A11-orthogonalP-directRead125-labelFree,A12-orthogonalP-identityAmp150-labelFree,A13-orthogonalP-lowQuad-directRead125-labelFree,A14-orthogonalP-lowQuad-identityAmp150-labelFree")
    parser.add_argument("--seed-base", type=int, default=1218600)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--task-compile-warmup-steps", type=int, default=2)
    parser.add_argument("--task-timing-warmup-epochs", type=int, default=1)
    parser.add_argument("--task-lr-schedule", default="linear_warmup10_cosine_final050")
    parser.add_argument("--optimizer-impl", default="adamw")
    parser.add_argument("--measure-linec", type=int, default=0)
    parser.add_argument("--linec-batch-size", type=int, default=32)
    parser.add_argument("--linec-sketch-batch-size", type=int, default=8)
    parser.add_argument("--linec-sketch-dim", type=int, default=8)
    parser.add_argument("--linec-ridge-lambda", type=float, default=1.0e-3)
    parser.add_argument("--covadapt-strength", type=float, default=0.08)
    parser.add_argument("--optframe-strength", type=float, default=0.06)
    parser.add_argument("--persistentdrift-strength", type=float, default=0.04)
    parser.add_argument("--adaptframesched-strength", type=float, default=0.05)
    return parser


if __name__ == "__main__":
    parsed = build_argparser().parse_args()
    result = run_main(parsed)
    print(json.dumps(result, indent=2, ensure_ascii=False))
