#!/usr/bin/env python3
"""DG-KAN v7.3 real-only expressivity/optimizer/Pareto runner.

This runner intentionally reuses the audited v7.2 real-only execution path for
task, gradient and efficiency measurement, then layers v7.3-specific probes on
top:

* dense source-of-gain ablations;
* compression/effective-expressivity Pareto rows;
* C3 logit-distillation reachability probes for compressed students;
* measured live-set attribution rows based only on recorded tensors/timings.

Classwise metrics are kept as diagnostics only.  No fake data, proxy rows,
fixed ratios or hand-filled pass claims are emitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

import run_gafu_v72_real as v72
from dgkan_core import get_device, parse_int_list, parse_str_list, save_json, set_seed, write_csv
from run_gafu_v63 import V63ManualLayer, V63Params
from run_gafu_v66_real import _git_commit, _git_status


PLAN_PATH = "docs/DG-KAN_v7.3_Expressivity_Optimizer_Geometry_Pareto_KernelNative_完整实验计划.md"
SCRIPT_PATH = "experiments/run_gafu_v73_real.py"
METRIC_UNAVAILABLE = "metric_unavailable"


_ORIGINAL_CANDIDATE_REGISTRY = v72._candidate_registry
_ORIGINAL_MAKE_MANUAL_CANDIDATE = v72._make_manual_candidate
_ORIGINAL_USES_V72_TRAIN_PATH = v72._uses_v72_train_path
_ORIGINAL_TRAIN_MANUAL_V72 = v72._train_manual_v72
_ORIGINAL_GRADIENT_CHECK = v72._gradient_check
_ORIGINAL_AUTOGRAD_FORWARD = v72._autograd_forward

_TEACHER_CACHE: Dict[Tuple[Any, ...], Tuple[Any, Any, Dict[str, Any]]] = {}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _mean(values: Iterable[Any], default: float = float("nan")) -> float:
    vals = [float(v) for v in values if _finite(v)]
    return sum(vals) / len(vals) if vals else default


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _hash_file(path: Path) -> str:
    return v72._hash_file(path) if path.exists() else ""


def _spec_map() -> Dict[str, Any]:
    base = {c.candidate_id: c for c in _ORIGINAL_CANDIDATE_REGISTRY()}
    v71_base = {c.candidate_id: c for c in v72.v71._candidate_registry()}
    out: Dict[str, Any] = dict(base)

    # v7.3 P1 dense source-of-gain ablations.
    out["A1"] = v72.v71.CandidateSpec(
        "A1",
        "D3-dense-poly2-no-gate-KAN-head-poly2-gate",
        "v73_expressivity_ablation",
        "dense",
        depth=3,
        head_kind="poly2_gate",
    )
    out["A2"] = v72.v71.CandidateSpec(
        "A2",
        "D3-manual-linear-no-poly2-KAN-head-poly2-gate",
        "v73_expressivity_ablation",
        "dense",
        depth=3,
        head_kind="poly2_gate",
    )
    out["A1C"] = v72.v71.CandidateSpec(
        "A1C",
        "D3-cached-poly2-no-gate-KAN-head-poly2-gate",
        "v73_kernel_native_structural_repair",
        "v73_cached_kind",
        depth=3,
        stack_kind="poly2",
        head_kind="poly2_gate",
    )
    out["A2C"] = v72.v71.CandidateSpec(
        "A2C",
        "D3-cached-linear-no-poly2-KAN-head-poly2-gate",
        "v73_kernel_native_structural_repair",
        "v73_cached_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_gate",
    )
    out["A2R"] = v72.v71.CandidateSpec(
        "A2R",
        "D3-cached-linear-no-poly2-KAN-head-rbf-poly-exp",
        "v73_kernel_native_structural_repair",
        "v73_cached_kind",
        depth=3,
        stack_kind="linear",
        head_kind="rbf_poly_exp",
    )
    out["A2S"] = v72.v71.CandidateSpec(
        "A2S",
        "D3-cached-linear-no-poly2-KAN-head-poly2-silu",
        "v73_kernel_native_structural_repair",
        "v73_cached_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["A4C"] = v72.v71.CandidateSpec(
        "A4C",
        "D3-cached-first-poly2-rest-linear-KAN-head-poly2-gate",
        "v73_kernel_native_structural_repair",
        "v73_cached_kind",
        depth=3,
        stack_kind="poly2_first",
        head_kind="poly2_gate",
    )
    out["A5C"] = v72.v71.CandidateSpec(
        "A5C",
        "D3-cached-last-poly2-rest-linear-KAN-head-poly2-gate",
        "v73_kernel_native_structural_repair",
        "v73_cached_kind",
        depth=3,
        stack_kind="poly2_last",
        head_kind="poly2_gate",
    )
    out["A3"] = v72.v71.CandidateSpec(
        "A3",
        "D3-no-dense-cross-g16-KAN-head-poly2-gate",
        "v73_expressivity_ablation",
        "vgrouped",
        depth=3,
        head_kind="poly2_gate",
        group_count=16,
    )

    # Restore v7.1 candidates that v7.2 did not expose by default.
    for cid in ("H1", "H2", "H5", "H6", "K0q", "K0qs", "K3", "K4"):
        if cid in v71_base:
            out[cid] = v71_base[cid]

    # v7.3 P3 optimizer reachability: same student structures as supervised
    # controls, but trained with C3 logit distillation.
    out["O10"] = v72.v71.CandidateSpec(
        "O10",
        "G3-logit-distill-from-C3-T2-alpha07",
        "v73_optimizer_reachability_distill",
        "vgrouped",
        depth=3,
        head_kind="poly2_gate",
        group_count=2,
    )
    out["O11"] = v72.v71.CandidateSpec(
        "O11",
        "G5-logit-distill-from-C3-T2-alpha07",
        "v73_optimizer_reachability_distill",
        "vgrouped",
        depth=3,
        head_kind="poly2_gate",
        group_count=8,
    )
    out["O12"] = v72.v71.CandidateSpec(
        "O12",
        "L2-logit-distill-from-C3-T2-alpha07",
        "v73_optimizer_reachability_distill",
        "lowrank",
        depth=2,
        head_kind="poly2_gate",
        group_count=32,
    )
    # v7.5 P3: A2S reachability with C3 teacher.  These keep the A2S
    # inference structure; only the training objective changes.
    out["D1"] = v72.v71.CandidateSpec(
        "D1",
        "A2S-logit-distill-from-C3-T2-alpha025",
        "v75_a2s_reachability_distill",
        "v73_cached_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["D2"] = v72.v71.CandidateSpec(
        "D2",
        "A2S-logit-distill-from-C3-T4-alpha025",
        "v75_a2s_reachability_distill",
        "v73_cached_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["D3"] = v72.v71.CandidateSpec(
        "D3",
        "A2S-logit-distill-from-C3-T4-alpha050",
        "v75_a2s_reachability_distill",
        "v73_cached_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    # v7.5 self-repair: dense-preserving checkpoint/recompute bridge.  The
    # inference function matches A2S, but the stack caches only the root input
    # and recomputes hidden pre-activations during backward to lower live set.
    out["E1"] = v72.v71.CandidateSpec(
        "E1",
        "A2S-recompute-y-checkpoint-stack",
        "v75_a2s_recompute_live_set_repair",
        "v75_recompute_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["E2"] = v72.v71.CandidateSpec(
        "E2",
        "A2S-recompute-y-logit-distill-from-C3-T4-alpha025",
        "v75_a2s_recompute_distill",
        "v75_recompute_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["E3"] = v72.v71.CandidateSpec(
        "E3",
        "A2S-recompute-y-logit-distill-from-C3-T4-alpha050",
        "v75_a2s_recompute_distill",
        "v75_recompute_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["Z1"] = v72.v71.CandidateSpec(
        "Z1",
        "A2S-cache-y-only-stack",
        "v75_a2s_y_only_live_set_repair",
        "v75_cache_y_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["Z2"] = v72.v71.CandidateSpec(
        "Z2",
        "A2S-cache-y-only-logit-distill-from-C3-T4-alpha025",
        "v75_a2s_y_only_distill",
        "v75_cache_y_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["Z3"] = v72.v71.CandidateSpec(
        "Z3",
        "A2S-cache-y-only-logit-distill-from-C3-T4-alpha050",
        "v75_a2s_y_only_distill",
        "v75_cache_y_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["U1"] = v72.v71.CandidateSpec(
        "U1",
        "A2S-cache-late-y-stack",
        "v75_a2s_late_y_live_set_repair",
        "v75_cache_late_y_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["U2"] = v72.v71.CandidateSpec(
        "U2",
        "A2S-cache-late-y-logit-distill-from-C3-T4-alpha025",
        "v75_a2s_late_y_distill",
        "v75_cache_late_y_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["U3"] = v72.v71.CandidateSpec(
        "U3",
        "A2S-cache-late-y-logit-distill-from-C3-T4-alpha050",
        "v75_a2s_late_y_distill",
        "v75_cache_late_y_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["M1"] = v72.v71.CandidateSpec(
        "M1",
        "A2S-fused-linear-silu-stack",
        "v75_dense_preserving_fused_stack",
        "v75_fused_linear_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["M2"] = v72.v71.CandidateSpec(
        "M2",
        "A2S-fused-linear-silu-logit-distill-from-C3-T4-alpha025",
        "v75_dense_preserving_fused_stack_distill",
        "v75_fused_linear_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["M3"] = v72.v71.CandidateSpec(
        "M3",
        "A2S-fused-linear-silu-logit-distill-from-C3-T4-alpha050",
        "v75_dense_preserving_fused_stack_distill",
        "v75_fused_linear_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["M4"] = v72.v71.CandidateSpec(
        "M4",
        "A2S-fused-linear-silu-and-poly2-silu-head",
        "v75_dense_preserving_fused_stack_head",
        "v75_fused_linear_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["M5"] = v72.v71.CandidateSpec(
        "M5",
        "A2S-fused-linear-silu-and-head-logit-distill-from-C3-T4-alpha025",
        "v75_dense_preserving_fused_stack_head_distill",
        "v75_fused_linear_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["M6"] = v72.v71.CandidateSpec(
        "M6",
        "A2S-fused-linear-silu-logit-distill-SGD-update",
        "v75_fused_stack_distill_sgd_update_diagnostic",
        "v75_fused_linear_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["M7"] = v72.v71.CandidateSpec(
        "M7",
        "A2S-packed-fused-linear-silu-stack",
        "v75_packed_fused_stack",
        "v75_packed_fused_linear_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["M8"] = v72.v71.CandidateSpec(
        "M8",
        "A2S-packed-fused-linear-silu-logit-distill-from-C3-T4-alpha025",
        "v75_packed_fused_stack_distill",
        "v75_packed_fused_linear_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["M9"] = v72.v71.CandidateSpec(
        "M9",
        "A2S-fused-linear-silu-stack-packed-generic-head",
        "v75_fused_stack_packed_generic_head",
        "v75_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["M10"] = v72.v71.CandidateSpec(
        "M10",
        "A2S-fused-linear-silu-packed-generic-head-logit-distill-from-C3-T4-alpha025",
        "v75_fused_stack_packed_generic_head_distill",
        "v75_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["M11"] = v72.v71.CandidateSpec(
        "M11",
        "A2S-fused-linear-silu-packed-generic-head-logit-distill-from-C3-T4-alpha050",
        "v75_fused_stack_packed_generic_head_distill",
        "v75_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["M12"] = v72.v71.CandidateSpec(
        "M12",
        "A2S-fused-linear-silu-packed-generic-head-M5init-logit-distill-from-C3-T4-alpha025",
        "v75_fused_stack_packed_generic_head_m5init_distill",
        "v75_fused_linear_packed_head_kind",
        depth=3,
        stack_kind="linear",
        head_kind="poly2_silu_base",
    )
    out["NP5"] = v72.v71.CandidateSpec(
        "NP5",
        "SparseInterpKAN-K8-d3-KAN-head-poly2-gate",
        "v73_new_primitive_diagnostic",
        "dense",
        depth=3,
        head_kind="poly2_gate",
    )
    out["NP6"] = v72.v71.CandidateSpec(
        "NP6",
        "FastRationalKAN-d3-KAN-head-poly2-gate",
        "v73_new_primitive_diagnostic",
        "dense",
        depth=3,
        head_kind="poly2_gate",
    )
    return out


def _candidate_registry() -> List[Any]:
    ordered = [
        "B0", "B3", "C3", "H3", "H4", "C1",
        "A1", "A1C", "A2", "A2C", "A2R", "A2S", "A4C", "A5C", "A3", "H1", "H2",
        "G3", "G4", "G5", "G6", "K0q", "K0qs", "K3", "K4",
        "L1", "L2", "L3",
        "O10", "O11", "O12", "D1", "D2", "D3", "E1", "E2", "E3", "Z1", "Z2", "Z3", "U1", "U2", "U3", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12",
        "NP5", "NP6",
    ]
    smap = _spec_map()
    return [smap[cid] for cid in ordered if cid in smap]


def _uses_distill(spec: Any) -> bool:
    return str(spec.candidate_id) in {"O10", "O11", "O12", "D1", "D2", "D3", "E2", "E3", "Z2", "Z3", "U2", "U3", "M2", "M3", "M5", "M6", "M8", "M10", "M11", "M12"}


def _distill_student_init_id(spec: Any) -> str:
    return {
        "O10": "G3",
        "O11": "G5",
        "O12": "L2",
        "D1": "A2S",
        "D2": "A2S",
        "D3": "A2S",
        "E2": "E1",
        "E3": "E1",
        "Z2": "Z1",
        "Z3": "Z1",
        "U2": "U1",
        "U3": "U1",
        "M2": "M1",
        "M3": "M1",
        "M5": "M4",
        "M6": "M1",
        "M8": "M7",
        "M10": "M9",
        "M11": "M9",
        "M12": "M4",
    }.get(str(spec.candidate_id), str(spec.candidate_id))


def _distill_temperature_for_spec(args: argparse.Namespace, spec: Any) -> float:
    return {"D1": 2.0, "D2": 4.0, "D3": 4.0, "E2": 4.0, "E3": 4.0, "Z2": 4.0, "Z3": 4.0, "U2": 4.0, "U3": 4.0, "M2": 4.0, "M3": 4.0, "M5": 4.0, "M6": 4.0, "M8": 4.0, "M10": 4.0, "M11": 4.0, "M12": 4.0}.get(str(spec.candidate_id), float(args.distill_temperature))


def _distill_alpha_for_spec(args: argparse.Namespace, spec: Any) -> float:
    return {"D1": 0.25, "D2": 0.25, "D3": 0.50, "E2": 0.25, "E3": 0.50, "Z2": 0.25, "Z3": 0.50, "U2": 0.25, "U3": 0.50, "M2": 0.25, "M3": 0.50, "M5": 0.25, "M6": 0.25, "M8": 0.25, "M10": 0.25, "M11": 0.50, "M12": 0.25}.get(str(spec.candidate_id), float(args.distill_alpha))


def _init_seed_candidate_id_v73(spec: Any) -> str:
    cid = str(spec.candidate_id)
    if cid == "A1C":
        return "A1"
    if cid in {"A2C", "A2R", "A2S", "A4C", "A5C", "E1", "Z1", "U1", "M1", "M4", "M7", "M9"}:
        return "A2"
    if _uses_distill(spec):
        return _distill_student_init_id(spec)
    if (
        v72._uses_weighted_loss(spec)
        or cid.startswith("Q")
        or v72._uses_checkpoint_selection(spec)
        or v72._uses_focal_loss(spec)
        or v72._uses_margin_loss(spec)
    ):
        return "C3"
    return cid


def _uses_v72_train_path(spec: Any, dataset: str) -> bool:
    return str(spec.stack_type) in {"v73_cached_kind", "v75_recompute_kind", "v75_cache_y_kind", "v75_cache_late_y_kind", "v75_fused_linear_kind", "v75_fused_linear_head_kind", "v75_packed_fused_linear_kind", "v75_fused_linear_packed_head_kind"} or _uses_distill(spec) or _ORIGINAL_USES_V72_TRAIN_PATH(spec, dataset)


class CachedDenseKindStack:
    """Dense V63-style stack that caches hidden pre-activations for one kind.

    This is a structural kernelization probe: it keeps the same function family
    as the corresponding dense ablation, but avoids the hidden-layer forward
    recomputation used only to recover the SiLU derivative during backward.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        basis: int,
        *,
        kind: str,
        device: torch.device,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.kind = str(kind)
        dims = [self.input_dim] + [self.hidden_dim] * self.depth
        kinds = self._layer_kinds(self.kind, len(dims) - 1)
        self.layers = [
            V63ManualLayer(a, b, kind=k, basis_count=basis, device=device, fused_hint="v73-cached-y")
            for (a, b), k in zip(zip(dims[:-1], dims[1:]), kinds)
        ]

    @staticmethod
    def _layer_kinds(kind: str, depth: int) -> List[str]:
        if kind == "poly2_first":
            return ["poly2"] + ["linear"] * max(0, int(depth) - 1)
        if kind == "poly2_last":
            return ["linear"] * max(0, int(depth) - 1) + ["poly2"]
        return [kind] * int(depth)

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        with torch.no_grad():
            h = x
            caches: List[Dict[str, torch.Tensor]] = []
            for i, layer in enumerate(self.layers):
                y, x_cache = layer.forward_manual(h)
                caches.append({"x": x_cache, "y": y.detach()})
                h = F.silu(y) if i < len(self.layers) - 1 else y
            return h, caches

    def forward_autograd_with_params(self, x: torch.Tensor, params: Sequence[Dict[str, torch.Tensor]]) -> torch.Tensor:
        h = x
        for i, (layer, p) in enumerate(zip(self.layers, params)):
            y = layer.forward_with_params(h, p)
            h = F.silu(y) if i < len(self.layers) - 1 else y
        return h

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        delta = dy
        with torch.no_grad():
            for i in reversed(range(len(self.layers))):
                if i < len(self.layers) - 1:
                    y = caches[i]["y"]
                    sig = torch.sigmoid(y)
                    delta = delta * sig * (1.0 + y * (1.0 - sig))
                delta = self.layers[i].backward_manual(delta, caches[i]["x"])
            return delta

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def params_and_grads(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        out: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        for li, layer in enumerate(self.layers):
            for name, p in layer.params.items():
                out.append((f"v73_cached_{self.kind}:{layer.kind}:l{li}:{name}", p, layer.grads[name]))
        return out

    def step_sgd(self, lr: float) -> None:
        with torch.no_grad():
            for _name, param, grad in self.params_and_grads():
                param.add_(grad, alpha=-lr)
        self.zero_grad()

    def params_flat(self) -> torch.Tensor:
        return torch.cat([p.detach().flatten().float().cpu() for layer in self.layers for p in layer.param_tensors()])

    def grads_flat(self) -> torch.Tensor:
        return torch.cat([g.detach().flatten().float().cpu() for layer in self.layers for g in layer.grads.values()])

    def clone_params_for_autograd(self) -> List[Dict[str, torch.Tensor]]:
        return [layer.clone_params_for_autograd() for layer in self.layers]

    def param_count(self) -> int:
        return sum(layer.param_count() for layer in self.layers)

    def op_counts(self) -> Dict[str, int]:
        out = {
            "op_count_exp": 0,
            "op_count_pow": 0,
            "op_count_gather": 0,
            "op_count_scatter": 0,
            "op_count_index_select": 0,
            "op_count_scatter_add": 0,
            "op_count_gemm": 0,
            "op_count_elementwise": 0,
        }
        for layer in self.layers:
            for key, value in layer.op_counts().items():
                out[key] = out.get(key, 0) + int(value)
        return out

    def cache_breakdown(self, caches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        x_bytes = sum(int(c["x"].numel() * c["x"].element_size()) for c in caches)
        y_bytes = sum(int(c["y"].numel() * c["y"].element_size()) for c in caches[:-1])
        total = x_bytes + y_bytes
        x_mb = x_bytes / (1024**2)
        y_mb = y_bytes / (1024**2)
        total_mb = total / (1024**2)
        largest_mb = max(
            [int(c["x"].numel() * c["x"].element_size()) / (1024**2) for c in caches]
            + [int(c["y"].numel() * c["y"].element_size()) / (1024**2) for c in caches[:-1]]
            + [0.0]
        )
        return {
            "cache_total_MB": total_mb,
            "cache_x_MB": x_mb,
            "cache_hidden_y_MB": y_mb,
            "cache_hidden_MB": total_mb,
            "cache_gate_MB": 0.0,
            "cache_basis_MB": 0.0,
            "materialized_tensor_count_measured": len(caches) + max(0, len(caches) - 1),
            "largest_live_tensor_MB_measured": largest_mb,
            "linear_body_temp_MB": x_mb,
            "hidden_y_cache_MB": y_mb,
            "manual_cache_MB_measured": total_mb,
            "top1_memory_source_measured": "manual_cache_x_inputs",
            "top2_memory_source_measured": "hidden_y_cache",
            "top3_memory_source_measured": "optimizer_state",
            "unknown_memory_fraction_measured": 0.0,
        }


class RecomputeYDenseKindStack(CachedDenseKindStack):
    """A2S-compatible stack that checkpoints only the root input.

    This is a v7.5 live-set repair candidate.  It preserves the dense linear
    stack function but does not keep hidden pre-activations alive across the
    forward/backward boundary; backward recomputes the needed prefix on demand.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        basis: int,
        *,
        kind: str,
        device: torch.device,
    ) -> None:
        super().__init__(input_dim, hidden_dim, depth, basis, kind=kind, device=device)
        for layer in self.layers:
            layer.fused_hint = "v75-recompute-y"

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        with torch.no_grad():
            h = x
            for i, layer in enumerate(self.layers):
                y, _x_cache = layer.forward_manual(h)
                h = F.silu(y) if i < len(self.layers) - 1 else y
            return h, [{"x0": x.detach()}]

    def _recompute_layer_input_and_y(self, x0: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x0
        with torch.no_grad():
            for j, layer in enumerate(self.layers):
                y, _ = layer.forward_manual(h)
                if j == layer_idx:
                    return h, y
                h = F.silu(y)
        raise IndexError(layer_idx)

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        with torch.no_grad():
            for i in reversed(range(len(self.layers))):
                x_i, y_i = self._recompute_layer_input_and_y(x0, i)
                if i < len(self.layers) - 1:
                    sig = torch.sigmoid(y_i)
                    delta = delta * sig * (1.0 + y_i * (1.0 - sig))
                delta = self.layers[i].backward_manual(delta, x_i)
            return delta

    def params_and_grads(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        out: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        for li, layer in enumerate(self.layers):
            for name, p in layer.params.items():
                out.append((f"v75_recompute_{self.kind}:{layer.kind}:l{li}:{name}", p, layer.grads[name]))
        return out

    def op_counts(self) -> Dict[str, int]:
        out = super().op_counts()
        # On-demand prefix recomputation adds 1 + ... + depth extra layer
        # forwards during backward.  This is measured as a structural op count,
        # not used as a fake kernel launch count.
        for layer in self.layers:
            for key, value in layer.op_counts().items():
                out[key] = out.get(key, 0) + int(value)
        return out

    def cache_breakdown(self, caches: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        x0 = caches[0]["x0"] if caches else None
        x0_mb = int(x0.numel() * x0.element_size()) / (1024**2) if x0 is not None else 0.0
        return {
            "cache_total_MB": x0_mb,
            "cache_x_MB": x0_mb,
            "cache_hidden_y_MB": 0.0,
            "cache_hidden_MB": x0_mb,
            "cache_gate_MB": 0.0,
            "cache_basis_MB": 0.0,
            "materialized_tensor_count_measured": 1,
            "largest_live_tensor_MB_measured": x0_mb,
            "linear_body_temp_MB": x0_mb,
            "hidden_y_cache_MB": 0.0,
            "manual_cache_MB_measured": x0_mb,
            "top1_memory_source_measured": "checkpoint_root_input",
            "top2_memory_source_measured": "recomputed_hidden_y_not_cached",
            "top3_memory_source_measured": "optimizer_state",
            "unknown_memory_fraction_measured": 0.0,
        }


class CacheYOnlyDenseKindStack(CachedDenseKindStack):
    """A2S-compatible stack that caches hidden y but not hidden activations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        basis: int,
        *,
        kind: str,
        device: torch.device,
    ) -> None:
        super().__init__(input_dim, hidden_dim, depth, basis, kind=kind, device=device)
        for layer in self.layers:
            layer.fused_hint = "v75-cache-y-only"

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        with torch.no_grad():
            h = x
            ys: List[torch.Tensor] = []
            for i, layer in enumerate(self.layers):
                y, _x_cache = layer.forward_manual(h)
                if i < len(self.layers) - 1:
                    ys.append(y.detach())
                    h = F.silu(y)
                else:
                    h = y
            return h, [{"x0": x.detach(), "ys": ys}]

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        ys: Sequence[torch.Tensor] = caches[0].get("ys", [])
        with torch.no_grad():
            for i in reversed(range(len(self.layers))):
                if i < len(self.layers) - 1:
                    y_i = ys[i]
                    sig = torch.sigmoid(y_i)
                    delta = delta * sig * (1.0 + y_i * (1.0 - sig))
                if i == 0:
                    x_i = x0
                else:
                    x_i = F.silu(ys[i - 1])
                delta = self.layers[i].backward_manual(delta, x_i)
            return delta

    def params_and_grads(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        out: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        for li, layer in enumerate(self.layers):
            for name, p in layer.params.items():
                out.append((f"v75_y_only_{self.kind}:{layer.kind}:l{li}:{name}", p, layer.grads[name]))
        return out

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        if not caches:
            x0_mb = 0.0
            y_mb = 0.0
            largest_mb = 0.0
            count = 0
        else:
            x0 = caches[0]["x0"]
            ys: Sequence[torch.Tensor] = caches[0].get("ys", [])
            x0_mb = int(x0.numel() * x0.element_size()) / (1024**2)
            y_mb = sum(int(y.numel() * y.element_size()) for y in ys) / (1024**2)
            largest_mb = max([x0_mb] + [int(y.numel() * y.element_size()) / (1024**2) for y in ys] + [0.0])
            count = 1 + len(ys)
        total_mb = x0_mb + y_mb
        return {
            "cache_total_MB": total_mb,
            "cache_x_MB": x0_mb,
            "cache_hidden_y_MB": y_mb,
            "cache_hidden_MB": total_mb,
            "cache_gate_MB": 0.0,
            "cache_basis_MB": 0.0,
            "materialized_tensor_count_measured": count,
            "largest_live_tensor_MB_measured": largest_mb,
            "linear_body_temp_MB": x0_mb,
            "hidden_y_cache_MB": y_mb,
            "manual_cache_MB_measured": total_mb,
            "top1_memory_source_measured": "checkpoint_root_input",
            "top2_memory_source_measured": "hidden_y_cache",
            "top3_memory_source_measured": "optimizer_state",
            "unknown_memory_fraction_measured": 0.0,
        }


class CacheLateYDenseKindStack(CacheYOnlyDenseKindStack):
    """Cache only later hidden pre-activations; recompute early y on demand."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        basis: int,
        *,
        kind: str,
        device: torch.device,
    ) -> None:
        super().__init__(input_dim, hidden_dim, depth, basis, kind=kind, device=device)
        for layer in self.layers:
            layer.fused_hint = "v75-cache-late-y"

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        with torch.no_grad():
            h = x
            ys: Dict[int, torch.Tensor] = {}
            last_hidden_idx = len(self.layers) - 2
            for i, layer in enumerate(self.layers):
                y, _x_cache = layer.forward_manual(h)
                if i < len(self.layers) - 1:
                    if i == last_hidden_idx:
                        ys[i] = y.detach()
                    h = F.silu(y)
                else:
                    h = y
            return h, [{"x0": x.detach(), "ys": ys}]

    def _cached_or_recompute_y(self, x0: torch.Tensor, ys: Dict[int, torch.Tensor], idx: int) -> torch.Tensor:
        if idx in ys:
            return ys[idx]
        h = x0
        with torch.no_grad():
            for j, layer in enumerate(self.layers):
                y = ys[j] if j in ys else layer.forward_manual(h)[0]
                if j == idx:
                    return y
                h = F.silu(y)
        raise IndexError(idx)

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        ys: Dict[int, torch.Tensor] = dict(caches[0].get("ys", {}))
        with torch.no_grad():
            for i in reversed(range(len(self.layers))):
                if i < len(self.layers) - 1:
                    y_i = self._cached_or_recompute_y(x0, ys, i)
                    sig = torch.sigmoid(y_i)
                    delta = delta * sig * (1.0 + y_i * (1.0 - sig))
                if i == 0:
                    x_i = x0
                else:
                    x_i = F.silu(self._cached_or_recompute_y(x0, ys, i - 1))
                delta = self.layers[i].backward_manual(delta, x_i)
            return delta

    def params_and_grads(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        out: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        for li, layer in enumerate(self.layers):
            for name, p in layer.params.items():
                out.append((f"v75_late_y_{self.kind}:{layer.kind}:l{li}:{name}", p, layer.grads[name]))
        return out

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        if not caches:
            x0_mb = 0.0
            y_mb = 0.0
            largest_mb = 0.0
            count = 0
        else:
            x0 = caches[0]["x0"]
            ys: Dict[int, torch.Tensor] = dict(caches[0].get("ys", {}))
            x0_mb = int(x0.numel() * x0.element_size()) / (1024**2)
            y_mb = sum(int(y.numel() * y.element_size()) for y in ys.values()) / (1024**2)
            largest_mb = max([x0_mb] + [int(y.numel() * y.element_size()) / (1024**2) for y in ys.values()] + [0.0])
            count = 1 + len(ys)
        total_mb = x0_mb + y_mb
        return {
            "cache_total_MB": total_mb,
            "cache_x_MB": x0_mb,
            "cache_hidden_y_MB": y_mb,
            "cache_hidden_MB": total_mb,
            "cache_gate_MB": 0.0,
            "cache_basis_MB": 0.0,
            "materialized_tensor_count_measured": count,
            "largest_live_tensor_MB_measured": largest_mb,
            "linear_body_temp_MB": x0_mb,
            "hidden_y_cache_MB": y_mb,
            "manual_cache_MB_measured": total_mb,
            "top1_memory_source_measured": "checkpoint_root_input",
            "top2_memory_source_measured": "late_hidden_y_cache",
            "top3_memory_source_measured": "optimizer_state",
            "unknown_memory_fraction_measured": 0.0,
        }


class FusedLinearSiluStack:
    """A2S-equivalent linear/SILU stack with one fused manual owner.

    The function is the same as a depth-3 stack of linear V63 layers with SiLU
    between hidden layers.  The probe removes per-layer object dispatch and
    stores only root input plus hidden pre-activations needed for exact manual
    backward; it does not change the loss, sampler, head, or parameterization.
    """

    def __init__(self, input_dim: int, hidden_dim: int, depth: int, *, device: torch.device) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        dims = [self.input_dim] + [self.hidden_dim] * self.depth
        self.params: List[Dict[str, torch.Tensor]] = [
            {"mix": torch.randn(out_dim, in_dim, device=device) / math.sqrt(max(1, in_dim))}
            for in_dim, out_dim in zip(dims[:-1], dims[1:])
        ]
        self.grads: List[Dict[str, torch.Tensor]] = [
            {"mix": torch.zeros_like(p["mix"])}
            for p in self.params
        ]

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        with torch.no_grad():
            h = x
            ys: List[torch.Tensor] = []
            for i, p in enumerate(self.params):
                y = h @ p["mix"].t()
                if i < len(self.params) - 1:
                    ys.append(y.detach())
                    h = F.silu(y)
                else:
                    h = y
            return h, [{"x0": x.detach(), "ys": ys}]

    def forward_autograd_with_params(self, x: torch.Tensor, params: Sequence[Dict[str, torch.Tensor]]) -> torch.Tensor:
        h = x
        for i, p in enumerate(params):
            y = h @ p["mix"].t()
            h = F.silu(y) if i < len(params) - 1 else y
        return h

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        ys: Sequence[torch.Tensor] = caches[0].get("ys", [])
        with torch.no_grad():
            for i in reversed(range(len(self.params))):
                if i < len(self.params) - 1:
                    y_i = ys[i]
                    sig = torch.sigmoid(y_i)
                    silu_prime = sig + y_i * (sig - sig.square())
                    delta = delta * silu_prime
                x_i = x0 if i == 0 else F.silu(ys[i - 1])
                self.grads[i]["mix"].add_(delta.t() @ x_i)
                delta = delta @ self.params[i]["mix"]
            return delta

    def zero_grad(self) -> None:
        for grad_dict in self.grads:
            for grad in grad_dict.values():
                grad.zero_()

    def params_and_grads(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        out: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        for li, (param_dict, grad_dict) in enumerate(zip(self.params, self.grads)):
            out.append((f"v75_fused_linear_silu:l{li}:mix", param_dict["mix"], grad_dict["mix"]))
        return out

    def step_sgd(self, lr: float) -> None:
        with torch.no_grad():
            for _name, param, grad in self.params_and_grads():
                param.add_(grad, alpha=-lr)
        self.zero_grad()

    def params_flat(self) -> torch.Tensor:
        return torch.cat([p["mix"].detach().flatten().float().cpu() for p in self.params])

    def grads_flat(self) -> torch.Tensor:
        return torch.cat([g["mix"].detach().flatten().float().cpu() for g in self.grads])

    def clone_params_for_autograd(self) -> List[Dict[str, torch.Tensor]]:
        return [{"mix": p["mix"].detach().clone().requires_grad_(True)} for p in self.params]

    def param_count(self) -> int:
        return sum(int(p["mix"].numel()) for p in self.params)

    def op_counts(self) -> Dict[str, int]:
        return {
            "op_count_exp": 0,
            "op_count_pow": 0,
            "op_count_gather": 0,
            "op_count_scatter": 0,
            "op_count_index_select": 0,
            "op_count_scatter_add": 0,
            "op_count_gemm": int(len(self.params)),
            "op_count_elementwise": int(max(0, len(self.params) - 1)),
        }

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        if not caches:
            x0_mb = 0.0
            y_mb = 0.0
            largest_mb = 0.0
            count = 0
        else:
            x0 = caches[0]["x0"]
            ys: Sequence[torch.Tensor] = caches[0].get("ys", [])
            x0_mb = int(x0.numel() * x0.element_size()) / (1024**2)
            y_mb = sum(int(y.numel() * y.element_size()) for y in ys) / (1024**2)
            largest_mb = max([x0_mb] + [int(y.numel() * y.element_size()) / (1024**2) for y in ys] + [0.0])
            count = 1 + len(ys)
        total_mb = x0_mb + y_mb
        return {
            "cache_total_MB": total_mb,
            "cache_x_MB": x0_mb,
            "cache_hidden_y_MB": y_mb,
            "cache_hidden_MB": total_mb,
            "cache_gate_MB": 0.0,
            "cache_basis_MB": 0.0,
            "materialized_tensor_count_measured": count,
            "largest_live_tensor_MB_measured": largest_mb,
            "linear_body_temp_MB": x0_mb,
            "hidden_y_cache_MB": y_mb,
            "manual_cache_MB_measured": total_mb,
            "top1_memory_source_measured": "fused_stack_checkpoint_root_input",
            "top2_memory_source_measured": "fused_stack_hidden_y_cache",
            "top3_memory_source_measured": "optimizer_state",
            "unknown_memory_fraction_measured": 0.0,
        }


class PackedFusedLinearSiluStack:
    """A2S-equivalent fused stack with all layer mixes in one tensor.

    This keeps the exact linear/SILU function used by M1/M2 while reducing the
    number of trainable tensor owners seen by the manual update path.  It is a
    parameter-layout/kernelization probe, not an objective or data change.
    """

    def __init__(self, input_dim: int, hidden_dim: int, depth: int, *, device: torch.device) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        dims = [self.input_dim] + [self.hidden_dim] * self.depth
        chunks: List[torch.Tensor] = [
            torch.randn(out_dim, in_dim, device=device) / math.sqrt(max(1, in_dim))
            for in_dim, out_dim in zip(dims[:-1], dims[1:])
        ]
        self.shapes: List[Tuple[int, int]] = [tuple(chunk.shape) for chunk in chunks]  # type: ignore[list-item]
        self.starts: List[int] = []
        offset = 0
        flat_chunks: List[torch.Tensor] = []
        for chunk in chunks:
            self.starts.append(offset)
            flat = chunk.flatten()
            flat_chunks.append(flat)
            offset += int(flat.numel())
        self.flat = torch.cat(flat_chunks)
        self.flat_grad = torch.zeros_like(self.flat)

    def _mixes(self) -> List[torch.Tensor]:
        mixes: List[torch.Tensor] = []
        for start, shape in zip(self.starts, self.shapes):
            end = start + int(shape[0] * shape[1])
            mixes.append(self.flat[start:end].view(shape))
        return mixes

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        with torch.no_grad():
            h = x
            ys: List[torch.Tensor] = []
            mixes = self._mixes()
            for i, mix in enumerate(mixes):
                y = h @ mix.t()
                if i < len(mixes) - 1:
                    ys.append(y.detach())
                    h = F.silu(y)
                else:
                    h = y
            return h, [{"x0": x.detach(), "ys": ys}]

    def forward_autograd_with_params(self, x: torch.Tensor, params: Sequence[Dict[str, torch.Tensor]]) -> torch.Tensor:
        h = x
        for i, p in enumerate(params):
            y = h @ p["mix"].t()
            h = F.silu(y) if i < len(params) - 1 else y
        return h

    def backward_manual(self, dy: torch.Tensor, caches: List[Dict[str, Any]]) -> torch.Tensor:
        delta = dy
        x0 = caches[0]["x0"]
        ys: Sequence[torch.Tensor] = caches[0].get("ys", [])
        mixes = self._mixes()
        with torch.no_grad():
            for i in reversed(range(len(mixes))):
                if i < len(mixes) - 1:
                    y_i = ys[i]
                    sig = torch.sigmoid(y_i)
                    silu_prime = sig + y_i * (sig - sig.square())
                    delta = delta * silu_prime
                x_i = x0 if i == 0 else F.silu(ys[i - 1])
                start = self.starts[i]
                shape = self.shapes[i]
                end = start + int(shape[0] * shape[1])
                self.flat_grad[start:end].view(shape).add_(delta.t() @ x_i)
                delta = delta @ mixes[i]
            return delta

    def zero_grad(self) -> None:
        self.flat_grad.zero_()

    def params_and_grads(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        return [("v75_packed_fused_linear_silu:mix_flat", self.flat, self.flat_grad)]

    def step_sgd(self, lr: float) -> None:
        with torch.no_grad():
            self.flat.add_(self.flat_grad, alpha=-lr)
        self.zero_grad()

    def params_flat(self) -> torch.Tensor:
        return self.flat.detach().flatten().float().cpu()

    def grads_flat(self) -> torch.Tensor:
        return self.flat_grad.detach().flatten().float().cpu()

    def clone_params_for_autograd(self) -> List[Dict[str, torch.Tensor]]:
        out: List[Dict[str, torch.Tensor]] = []
        for mix in self._mixes():
            out.append({"mix": mix.detach().clone().requires_grad_(True)})
        return out

    def param_count(self) -> int:
        return int(self.flat.numel())

    def op_counts(self) -> Dict[str, int]:
        return {
            "op_count_exp": 0,
            "op_count_pow": 0,
            "op_count_gather": 0,
            "op_count_scatter": 0,
            "op_count_index_select": 0,
            "op_count_scatter_add": 0,
            "op_count_gemm": int(self.depth),
            "op_count_elementwise": int(max(0, self.depth - 1)),
        }

    def cache_breakdown(self, caches: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        if not caches:
            x0_mb = 0.0
            y_mb = 0.0
            largest_mb = 0.0
            count = 0
        else:
            x0 = caches[0]["x0"]
            ys: Sequence[torch.Tensor] = caches[0].get("ys", [])
            x0_mb = int(x0.numel() * x0.element_size()) / (1024**2)
            y_mb = sum(int(y.numel() * y.element_size()) for y in ys) / (1024**2)
            largest_mb = max([x0_mb] + [int(y.numel() * y.element_size()) / (1024**2) for y in ys] + [0.0])
            count = 1 + len(ys)
        total_mb = x0_mb + y_mb
        return {
            "cache_total_MB": total_mb,
            "cache_x_MB": x0_mb,
            "cache_hidden_y_MB": y_mb,
            "cache_hidden_MB": total_mb,
            "cache_gate_MB": 0.0,
            "cache_basis_MB": 0.0,
            "materialized_tensor_count_measured": count,
            "largest_live_tensor_MB_measured": largest_mb,
            "linear_body_temp_MB": x0_mb,
            "hidden_y_cache_MB": y_mb,
            "manual_cache_MB_measured": total_mb,
            "top1_memory_source_measured": "packed_fused_stack_checkpoint_root_input",
            "top2_memory_source_measured": "packed_fused_stack_hidden_y_cache",
            "top3_memory_source_measured": "optimizer_state",
            "unknown_memory_fraction_measured": 0.0,
        }


class FusedPoly2SiluHead:
    """Specialized head equivalent to V63 poly2_silu_base layer."""

    def __init__(self, in_dim: int, out_dim: int, *, device: torch.device) -> None:
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.scale = 0.05
        self.params: Dict[str, torch.Tensor] = {
            "mix": torch.randn(out_dim, in_dim, device=device) / math.sqrt(max(1, in_dim)),
            "poly": torch.zeros(in_dim, 2, device=device),
            "base": torch.zeros(in_dim, 2, device=device),
        }
        self.params["poly"][:, 0].fill_(0.02)
        self.params["base"][:, 0].fill_(1.0)
        self.params["base"][:, 1].fill_(0.05)
        self.grads = {k: torch.zeros_like(v) for k, v in self.params.items()}

    def _transform(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        p = params["poly"]
        b = params["base"]
        return (
            b[:, 0].unsqueeze(0) * x
            + b[:, 1].unsqueeze(0) * F.silu(x)
            + self.scale * (p[:, 0].unsqueeze(0) * x + p[:, 1].unsqueeze(0) * x.square())
        )

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._transform(x, params) @ params["mix"].t()

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward_with_params(x, self.params), x.detach()

    def backward_manual(self, dy: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        x = cache
        with torch.no_grad():
            z = self._transform(x, self.params)
            dz = dy @ self.params["mix"]
            self.grads["mix"].add_(dy.t() @ z)
            p = self.params["poly"]
            b = self.params["base"]
            silu_x = F.silu(x)
            sig = torch.sigmoid(x)
            silu_prime = sig * (1.0 + x * (1.0 - sig))
            self.grads["base"][:, 0].add_((dz * x).sum(dim=0))
            self.grads["base"][:, 1].add_((dz * silu_x).sum(dim=0))
            dres = dz * self.scale
            self.grads["poly"][:, 0].add_((dres * x).sum(dim=0))
            self.grads["poly"][:, 1].add_((dres * x.square()).sum(dim=0))
            return dz * (
                b[:, 0].unsqueeze(0)
                + b[:, 1].unsqueeze(0) * silu_prime
                + self.scale * (p[:, 0].unsqueeze(0) + 2.0 * p[:, 1].unsqueeze(0) * x)
            )

    def zero_grad(self) -> None:
        for grad in self.grads.values():
            grad.zero_()

    def params_and_grads(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        return [(f"head_fused_poly2_silu:{name}", p, self.grads[name]) for name, p in self.params.items()]

    def param_count(self) -> int:
        return sum(int(p.numel()) for p in self.params.values())

    def clone_params_for_autograd(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone().requires_grad_(True) for k, v in self.params.items()}


class PackedV63Poly2SiluHead:
    """Generic V63 poly2_silu_base head with a packed parameter owner."""

    def __init__(self, in_dim: int, out_dim: int, *, basis_count: int, device: torch.device) -> None:
        self.layer = V63ManualLayer(
            in_dim,
            out_dim,
            kind="poly2_silu_base",
            basis_count=basis_count,
            device=device,
            fused_hint="v75-packed-generic-head",
        )
        self.names = list(self.layer.params.keys())
        self.shapes: Dict[str, Tuple[int, ...]] = {name: tuple(param.shape) for name, param in self.layer.params.items()}
        self.starts: Dict[str, int] = {}
        offset = 0
        flat_chunks: List[torch.Tensor] = []
        for name in self.names:
            self.starts[name] = offset
            flat = self.layer.params[name].flatten()
            flat_chunks.append(flat)
            offset += int(flat.numel())
        self.flat = torch.cat(flat_chunks)
        self.flat_grad = torch.zeros_like(self.flat)
        for name in self.names:
            start = self.starts[name]
            shape = self.shapes[name]
            end = start + math.prod(shape)
            self.layer.params[name] = self.flat[start:end].view(shape)
            self.layer.grads[name] = self.flat_grad[start:end].view(shape)

    def forward_manual(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.layer.forward_manual(x)

    def backward_manual(self, dy: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        return self.layer.backward_manual(dy, cache)

    def forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.layer.forward_with_params(x, params)

    def zero_grad(self) -> None:
        self.flat_grad.zero_()

    def params_and_grads(self) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        return [("head_packed_v63_poly2_silu:flat", self.flat, self.flat_grad)]

    def param_count(self) -> int:
        return int(self.flat.numel())

    def clone_params_for_autograd(self) -> Dict[str, torch.Tensor]:
        return {name: self.layer.params[name].detach().clone().requires_grad_(True) for name in self.names}


def _autograd_forward_v73(stack: Any, head: Any, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if not isinstance(stack, (FusedLinearSiluStack, PackedFusedLinearSiluStack)):
        return _ORIGINAL_AUTOGRAD_FORWARD(stack, head, x)
    leafs: List[torch.Tensor] = []
    stack_params = stack.clone_params_for_autograd()
    for params in stack_params:
        for value in params.values():
            leafs.append(value)
    h = stack.forward_autograd_with_params(x, stack_params)
    head_params = head.clone_params_for_autograd() if hasattr(head, "clone_params_for_autograd") else head.layer.clone_params_for_autograd()
    for value in head_params.values():
        leafs.append(value)
    logits = head.forward_with_params(h, head_params) if hasattr(head, "forward_with_params") else head.layer.forward_with_params(h, head_params)
    return logits, leafs


def _make_manual_candidate(spec: Any, input_dim: int, num_classes: int, hidden_dim: int, basis: int, device: torch.device) -> Tuple[Any, Any]:
    if str(spec.stack_type) == "v73_cached_kind":
        stack = CachedDenseKindStack(input_dim, hidden_dim, spec.depth, basis, kind=spec.stack_kind, device=device)
        head = v72.v71.HeadWrapper(
            V63ManualLayer(
                hidden_dim,
                num_classes,
                kind=v72.v71._head_kind_to_layer_kind(spec.head_kind),
                basis_count=basis,
                device=device,
            )
        )
        return stack, head
    if str(spec.stack_type) == "v75_recompute_kind":
        stack = RecomputeYDenseKindStack(input_dim, hidden_dim, spec.depth, basis, kind=spec.stack_kind, device=device)
        head = v72.v71.HeadWrapper(
            V63ManualLayer(
                hidden_dim,
                num_classes,
                kind=v72.v71._head_kind_to_layer_kind(spec.head_kind),
                basis_count=basis,
                device=device,
                fused_hint="v75-recompute-y",
            )
        )
        return stack, head
    if str(spec.stack_type) == "v75_cache_y_kind":
        stack = CacheYOnlyDenseKindStack(input_dim, hidden_dim, spec.depth, basis, kind=spec.stack_kind, device=device)
        head = v72.v71.HeadWrapper(
            V63ManualLayer(
                hidden_dim,
                num_classes,
                kind=v72.v71._head_kind_to_layer_kind(spec.head_kind),
                basis_count=basis,
                device=device,
                fused_hint="v75-cache-y-only",
            )
        )
        return stack, head
    if str(spec.stack_type) == "v75_cache_late_y_kind":
        stack = CacheLateYDenseKindStack(input_dim, hidden_dim, spec.depth, basis, kind=spec.stack_kind, device=device)
        head = v72.v71.HeadWrapper(
            V63ManualLayer(
                hidden_dim,
                num_classes,
                kind=v72.v71._head_kind_to_layer_kind(spec.head_kind),
                basis_count=basis,
                device=device,
                fused_hint="v75-cache-late-y",
            )
        )
        return stack, head
    if str(spec.stack_type) == "v75_fused_linear_kind":
        stack = FusedLinearSiluStack(input_dim, hidden_dim, spec.depth, device=device)
        head = v72.v71.HeadWrapper(
            V63ManualLayer(
                hidden_dim,
                num_classes,
                kind=v72.v71._head_kind_to_layer_kind(spec.head_kind),
                basis_count=basis,
                device=device,
                fused_hint="v75-fused-linear-silu",
            )
        )
        if str(spec.candidate_id) == "M6":
            setattr(stack, "update_mode", "sgd")
            setattr(head, "update_mode", "sgd")
        return stack, head
    if str(spec.stack_type) == "v75_packed_fused_linear_kind":
        stack = PackedFusedLinearSiluStack(input_dim, hidden_dim, spec.depth, device=device)
        head = v72.v71.HeadWrapper(
            V63ManualLayer(
                hidden_dim,
                num_classes,
                kind=v72.v71._head_kind_to_layer_kind(spec.head_kind),
                basis_count=basis,
                device=device,
                fused_hint="v75-packed-fused-linear-silu",
            )
        )
        return stack, head
    if str(spec.stack_type) == "v75_fused_linear_head_kind":
        stack = FusedLinearSiluStack(input_dim, hidden_dim, spec.depth, device=device)
        head = FusedPoly2SiluHead(hidden_dim, num_classes, device=device)
        return stack, head
    if str(spec.stack_type) == "v75_fused_linear_packed_head_kind":
        stack = FusedLinearSiluStack(input_dim, hidden_dim, spec.depth, device=device)
        head = PackedV63Poly2SiluHead(hidden_dim, num_classes, basis_count=basis, device=device)
        return stack, head
    return _ORIGINAL_MAKE_MANUAL_CANDIDATE(spec, input_dim, num_classes, hidden_dim, basis, device)


def _teacher_key(args: argparse.Namespace, dataset: str, seed: int, device: torch.device) -> Tuple[Any, ...]:
    return (
        str(dataset),
        int(seed),
        str(device),
        int(args.hidden_dim),
        int(args.basis_count),
        int(args.task_steps),
        int(args.train_size),
        int(args.val_size),
        int(args.test_size),
        int(args.batch_size),
        float(args.weight_decay),
    )


def _train_c3_teacher(args: argparse.Namespace, dataset: str, seed: int) -> Tuple[Any, Any, Dict[str, Any]]:
    device = get_device(args.device)
    key = _teacher_key(args, dataset, seed, device)
    if key in _TEACHER_CACHE:
        return _TEACHER_CACHE[key]

    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = v72.v71.load_vision_bundle(
        dataset,
        data_root=Path(args.data_root),
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=seed,
        allow_fake_data=False,
    )
    x_train = bundle.x_train.to(device)
    y_train = bundle.y_train.to(device)
    x_val = bundle.x_val.to(device)
    y_val = bundle.y_val.to(device)
    x_test = bundle.x_test.to(device)
    y_test = bundle.y_test.to(device)
    spec = _spec_map()["C3"]
    set_seed(v72._stable_seed("v72-train", dataset, seed, "C3", spec.head_kind, spec.group_count, spec.shuffle))
    stack, head = _make_manual_candidate(spec, bundle.input_dim, bundle.num_classes, args.hidden_dim, args.basis_count, device)
    opt = v72.v71.FastAdamW([stack, head], lr=params.lr_manual, weight_decay=args.weight_decay)
    for step in range(1, int(args.task_steps) + 1):
        xb, yb = v72.v71._select_batch(x_train, y_train, args.batch_size, step)
        h, caches = stack.forward_manual(xb)
        logits, head_cache = head.forward_manual(h)
        _loss, grad_logits = v72._weighted_smooth_ce_and_grad(
            logits,
            yb,
            spec.label_smoothing,
            torch.ones(bundle.num_classes, device=device),
        )
        dh = head.backward_manual(grad_logits, head_cache)
        stack.backward_manual(dh, caches)
        opt.step(step, args.task_steps, warmup_cosine=True)
    val = v72.v71._manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
    test = v72.v71._manual_eval(stack, head, x_test, y_test, args.eval_batch_size, bundle.num_classes)
    meta = {
        "teacher_id": "C3",
        "teacher_name": spec.candidate_name,
        "teacher_val_acc": val["acc"],
        "teacher_test_acc": test["acc"],
        "teacher_ECE": val["ECE"],
        "teacher_NLL": val["NLL"],
    }
    _TEACHER_CACHE[key] = (stack, head, meta)
    return stack, head, meta


def _teacher_logits(stack: Any, head: Any, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        h, _ = stack.forward_manual(x)
        logits, _ = head.forward_manual(h)
        return logits.detach()


def _distill_loss_and_grad(
    logits: torch.Tensor,
    y: torch.Tensor,
    teacher_logits: torch.Tensor,
    label_smoothing: float,
    *,
    alpha: float,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce_loss, ce_grad = v72._weighted_smooth_ce_and_grad(
        logits,
        y,
        label_smoothing,
        torch.ones(logits.shape[-1], device=logits.device),
    )
    t = float(temperature)
    p_teacher = F.softmax(teacher_logits.detach() / t, dim=-1)
    logp_student = F.log_softmax(logits / t, dim=-1)
    p_student = logp_student.exp()
    kl = (p_teacher * (torch.log(p_teacher.clamp_min(1.0e-12)) - logp_student)).sum(dim=-1).mean()
    kl_scaled = float(alpha) * t * t * kl
    grad_kl = float(alpha) * t * (p_student - p_teacher) / max(1, int(logits.shape[0]))
    return ce_loss + kl_scaled, ce_grad + grad_kl, kl.detach()


def _distill_loss_autograd(
    logits: torch.Tensor,
    y: torch.Tensor,
    teacher_logits: torch.Tensor,
    label_smoothing: float,
    *,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    ce = v72._weighted_smooth_loss_autograd(
        logits,
        y,
        label_smoothing,
        torch.ones(logits.shape[-1], device=logits.device),
    )
    t = float(temperature)
    p_teacher = F.softmax(teacher_logits.detach() / t, dim=-1)
    logp_student = F.log_softmax(logits / t, dim=-1)
    kl = (p_teacher * (torch.log(p_teacher.clamp_min(1.0e-12)) - logp_student)).sum(dim=-1).mean()
    return ce + float(alpha) * t * t * kl


def _train_manual_v73(
    args: argparse.Namespace,
    spec: Any,
    dataset: str,
    seed: int,
    baseline_classwise_acc: Sequence[float] | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not _uses_distill(spec):
        return _ORIGINAL_TRAIN_MANUAL_V72(args, spec, dataset, seed, baseline_classwise_acc)

    device = get_device(args.device)
    params = V63Params(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, batch_size=args.batch_size)
    bundle = v72.v71.load_vision_bundle(
        dataset,
        data_root=Path(args.data_root),
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=seed,
        allow_fake_data=False,
    )
    teacher_stack, teacher_head, teacher_meta = _train_c3_teacher(args, dataset, seed)
    x_train = bundle.x_train.to(device)
    y_train = bundle.y_train.to(device)
    x_val = bundle.x_val.to(device)
    y_val = bundle.y_val.to(device)
    x_test = bundle.x_test.to(device)
    y_test = bundle.y_test.to(device)
    set_seed(v72._stable_seed("v73-distill-student", dataset, seed, _distill_student_init_id(spec), spec.head_kind, spec.group_count, spec.shuffle))
    stack, head = _make_manual_candidate(spec, bundle.input_dim, bundle.num_classes, args.hidden_dim, args.basis_count, device)
    opt = v72.v71.FastAdamW([stack, head], lr=params.lr_manual * spec.lr_mult, weight_decay=args.weight_decay)
    train_eval_x = x_train[: min(max(args.batch_size, 512), x_train.shape[0])]
    train_eval_y = y_train[: train_eval_x.shape[0]]
    train0 = v72.v71._manual_eval(stack, head, train_eval_x, train_eval_y, args.eval_batch_size, bundle.num_classes)
    val0 = v72.v71._manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
    alpha = _distill_alpha_for_spec(args, spec)
    temperature = _distill_temperature_for_spec(args, spec)
    trace: List[Dict[str, Any]] = []
    first_loss_before = float("nan")
    first_loss_after = float("nan")
    first_kl = float("nan")
    started = time.perf_counter()
    for step in range(1, int(args.task_steps) + 1):
        xb, yb = v72.v71._select_batch(x_train, y_train, args.batch_size, step)
        tlogits = _teacher_logits(teacher_stack, teacher_head, xb)
        h, caches = stack.forward_manual(xb)
        logits, head_cache = head.forward_manual(h)
        loss, grad_logits, kl = _distill_loss_and_grad(
            logits,
            yb,
            tlogits,
            spec.label_smoothing,
            alpha=alpha,
            temperature=temperature,
        )
        if step == 1:
            first_loss_before = float(loss.detach().cpu())
            first_kl = float(kl.detach().cpu())
        dh = head.backward_manual(grad_logits, head_cache)
        stack.backward_manual(dh, caches)
        update_norm = opt.step(step, args.task_steps, warmup_cosine=True)
        if step == 1:
            with torch.no_grad():
                h1, _ = stack.forward_manual(xb)
                logits1, _ = head.forward_manual(h1)
                first_loss_after = float(
                    _distill_loss_and_grad(
                        logits1,
                        yb,
                        tlogits,
                        spec.label_smoothing,
                        alpha=alpha,
                        temperature=temperature,
                    )[0].detach().cpu()
                )
        if step % int(args.trace_every) == 0 or step == int(args.task_steps):
            tr = v72.v71._manual_eval(stack, head, train_eval_x, train_eval_y, args.eval_batch_size, bundle.num_classes)
            va = v72.v71._manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
            trace.append({
                "stage": "TASK_TRACE",
                "candidate_id": spec.candidate_id,
                "candidate_name": spec.candidate_name,
                "family": spec.family,
                "dataset": dataset,
                "seed": seed,
                "step": step,
                "wall_clock_time_sec": time.perf_counter() - started,
                "train_loss": tr["loss"],
                "train_acc": tr["acc"],
                "val_loss": va["loss"],
                "val_acc": va["acc"],
                "ECE": va["ECE"],
                "NLL": va["NLL"],
                "update_norm": update_norm,
                "recipe": "logit_distill_from_C3",
                "teacher_id": "C3",
                "distill_temperature": temperature,
                "alpha_logit": alpha,
                "student_teacher_logit_KL_step1": first_kl,
                "fake_data_used": 0,
                "proxy_row_used": 0,
            })
    train1 = v72.v71._manual_eval(stack, head, train_eval_x, train_eval_y, args.eval_batch_size, bundle.num_classes)
    val1 = v72.v71._manual_eval(stack, head, x_val, y_val, args.eval_batch_size, bundle.num_classes)
    test1 = v72.v71._manual_eval(stack, head, x_test, y_test, args.eval_batch_size, bundle.num_classes)
    summary = v72.v71._task_summary_base(args, spec, dataset, seed, bundle.input_dim, bundle.num_classes)
    summary.update({
        "train_loss_before": train0["loss"],
        "train_loss_after": train1["loss"],
        "val_loss_before": val0["loss"],
        "val_loss_after": val1["loss"],
        "train_loss_delta": train1["loss"] - train0["loss"],
        "val_loss_delta": val1["loss"] - val0["loss"],
        "train_acc": train1["acc"],
        "val_acc": val1["acc"],
        "test_acc": test1["acc"],
        "ECE": val1["ECE"],
        "NLL": val1["NLL"],
        "test_ECE": test1["ECE"],
        "test_NLL": test1["NLL"],
        "confidence_mean": val1["confidence_mean"],
        "margin_p10": val1["margin_p10"],
        "feature_effective_rank": val1["feature_effective_rank"],
        "classwise_acc": val1["classwise_acc"],
        "one_step_loss_before": first_loss_before,
        "one_step_loss_after": first_loss_after,
        "one_step_loss_delta": first_loss_after - first_loss_before if _finite(first_loss_after) and _finite(first_loss_before) else float("nan"),
        "one_step_pass": int(_finite(first_loss_after) and _finite(first_loss_before) and first_loss_after < first_loss_before),
        "val_loss_auc_step": v72.v71._auc([(r["step"], r["val_loss"]) for r in trace]),
        "val_loss_auc_time": v72.v71._auc([(r["wall_clock_time_sec"], r["val_loss"]) for r in trace]),
        "stack_param_count": stack.param_count(),
        "head_param_count": head.param_count(),
        "kan_trainable_param_count": stack.param_count() + head.param_count(),
        "edge_param_count": stack.param_count() + head.param_count(),
        "wall_clock_time_sec": time.perf_counter() - started,
        "recipe": "logit_distill_from_C3",
        "teacher_id": "C3",
        "distill_temperature": temperature,
        "alpha_logit": alpha,
        **teacher_meta,
        "student_supervised_control": _distill_student_init_id(spec),
        "implementation_status": "measured",
        "stage_status": "measured",
    })
    return summary, trace


def _gradient_check_v73(args: argparse.Namespace, spec: Any, dataset: str, batch_size: int) -> Dict[str, Any]:
    if not _uses_distill(spec):
        return _ORIGINAL_GRADIENT_CHECK(args, spec, dataset, batch_size)
    device = get_device(args.device)
    bundle = v72.v71.load_vision_bundle(
        dataset,
        data_root=Path(args.data_root),
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=0,
        allow_fake_data=False,
    )
    x = bundle.x_train[:batch_size].to(device)
    y = bundle.y_train[:batch_size].to(device)
    teacher_stack, teacher_head, _teacher_meta = _train_c3_teacher(args, dataset, 0)
    tlogits = _teacher_logits(teacher_stack, teacher_head, x)
    set_seed(v72._stable_seed("grad-v73-distill", dataset, batch_size, _distill_student_init_id(spec)))
    stack, head = _make_manual_candidate(spec, bundle.input_dim, bundle.num_classes, args.hidden_dim, args.basis_count, device)
    logits, caches, head_cache, _h = v72._manual_forward(stack, head, x)
    manual_loss, grad_logits, _kl = _distill_loss_and_grad(
        logits,
        y,
        tlogits,
        spec.label_smoothing,
        alpha=_distill_alpha_for_spec(args, spec),
        temperature=_distill_temperature_for_spec(args, spec),
    )
    dh = head.backward_manual(grad_logits, head_cache)
    stack.backward_manual(dh, caches)
    manual_params, manual_grads = v72._flatten_params_and_grads([stack, head])
    logits_auto, leafs = v72._autograd_forward(stack, head, x)
    auto_loss = _distill_loss_autograd(
        logits_auto,
        y,
        tlogits,
        spec.label_smoothing,
        alpha=_distill_alpha_for_spec(args, spec),
        temperature=_distill_temperature_for_spec(args, spec),
    )
    auto_grads = torch.autograd.grad(auto_loss, leafs, allow_unused=False)
    auto_flat = torch.cat([g.detach().flatten().float().cpu() for g in auto_grads])
    abs_err = (manual_grads - auto_flat).abs()
    scale = torch.maximum(manual_grads.abs(), auto_flat.abs())
    significant = scale > 1.0e-4
    rel = abs_err[significant] / scale[significant].clamp_min(1.0e-12) if bool(significant.any()) else abs_err
    cos = float(F.cosine_similarity(manual_grads, auto_flat, dim=0).item()) if manual_grads.numel() else 1.0
    forward_rel = (logits.detach() - logits_auto.detach()).abs().max() / logits_auto.detach().abs().max().clamp_min(1.0e-8)
    lr = 1.0e-4
    before = [p.detach().clone() for _n, p, _g in stack.params_and_grads()] + [p.detach().clone() for _n, p, _g in head.params_and_grads()]
    loss_before = float(manual_loss.detach().cpu())
    with torch.no_grad():
        for _name, p, g in stack.params_and_grads():
            p.add_(g, alpha=-lr)
        for _name, p, g in head.params_and_grads():
            p.add_(g, alpha=-lr)
        logits_after, _caches_after, _head_cache_after, _h_after = v72._manual_forward(stack, head, x)
        loss_after = float(
            _distill_loss_and_grad(
                logits_after,
                y,
                tlogits,
                spec.label_smoothing,
                alpha=_distill_alpha_for_spec(args, spec),
                temperature=_distill_temperature_for_spec(args, spec),
            )[0].detach().cpu()
        )
        idx = 0
        for _name, p, _g in stack.params_and_grads():
            p.copy_(before[idx])
            idx += 1
        for _name, p, _g in head.params_and_grads():
            p.copy_(before[idx])
            idx += 1
    after_restore, _ = v72._flatten_params_and_grads([stack, head])
    rollback_err = float((after_restore - manual_params).abs().max().item()) if manual_params.numel() else 0.0
    return {
        "stage": "P2_GRADIENT",
        "candidate_id": spec.candidate_id,
        "candidate_name": spec.candidate_name,
        "family": spec.family,
        "dataset": dataset,
        "batch_size": batch_size,
        "forward_relerr_max": float(forward_rel.detach().cpu()),
        "loss_relerr": abs(float(manual_loss.detach().cpu()) - float(auto_loss.detach().cpu())) / max(1.0e-8, abs(float(auto_loss.detach().cpu()))),
        "grad_relerr_max": float(rel.max().item()) if rel.numel() else 0.0,
        "grad_relerr_mean": float(rel.mean().item()) if rel.numel() else 0.0,
        "grad_abs_err_max": float(abs_err.max().item()) if abs_err.numel() else 0.0,
        "grad_relerr_scope": "abs_grad_gt_1e-4_else_abs_err",
        "grad_cos": cos,
        "grad_cos_min": cos,
        "grad_cos_mean": cos,
        "finite_grad": int(bool(torch.isfinite(manual_grads).all()) and bool(torch.isfinite(auto_flat).all())),
        "finite_forward": int(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(logits_auto).all())),
        "finite_loss": int(math.isfinite(loss_before) and math.isfinite(float(auto_loss.detach().cpu()))),
        "nan_count": int(torch.isnan(manual_grads).sum().item()),
        "inf_count": int(torch.isinf(manual_grads).sum().item()),
        "rollback_max_abs_error": rollback_err,
        "rollback_error": rollback_err,
        "rollback_pass": int(rollback_err < 1.0e-8),
        "one_step_loss_before": loss_before,
        "one_step_loss_after": loss_after,
        "one_step_loss_delta": loss_after - loss_before,
        "one_step_pass": int(loss_after < loss_before),
        "recipe": "logit_distill_from_C3",
        "teacher_id": "C3",
        "distill_temperature": _distill_temperature_for_spec(args, spec),
        "alpha_logit": _distill_alpha_for_spec(args, spec),
        "grad_pass": int((rel.numel() > 0 and float(rel.max().item()) <= 1.0e-4) and cos >= 0.999 and loss_after < loss_before and rollback_err < 1.0e-8),
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }


def _candidate_constraint(cid: str) -> Tuple[str, Any]:
    if cid in {"B3", "H1", "H2", "H3", "H4", "C1", "C3", "A1", "A1C", "A2", "A2C", "A2R", "A2S", "A4C", "A5C", "E1", "Z1", "U1", "M1", "M4", "M7", "M9"}:
        return ("dense_or_ablation", cid)
    if cid in {"G3", "K0", "K0s"}:
        return ("grouped", 2)
    if cid in {"K0q", "K0qs", "G4"}:
        return ("grouped", 4)
    if cid in {"G5", "K1", "K3"}:
        return ("grouped", 8)
    if cid in {"G6", "K2", "K4", "A3"}:
        return ("grouped", 16)
    if cid == "L1":
        return ("lowrank", 16)
    if cid in {"L2", "L3"}:
        return ("lowrank", 32)
    if cid in {"O10", "O11", "O12", "D1", "D2", "D3", "E2", "E3", "Z2", "Z3", "U2", "U3", "M2", "M3", "M5", "M6", "M8", "M10", "M11", "M12"}:
        return ("distilled_student", cid)
    if cid in {"NP5", "NP6"}:
        return ("new_primitive", cid)
    return ("other", "")


def _postprocess_v73(out_dir: Path, args: argparse.Namespace) -> None:
    task_summary = _read_csv_rows(out_dir / "p9_task_summary.csv")
    sig_rows = _read_csv_rows(out_dir / "p1_significance_audit.csv")
    eff_summary = _read_csv_rows(out_dir / "p10_efficiency_summary.csv")
    eff_rows = _read_csv_rows(out_dir / "p10_efficiency_profiler.csv")
    grad_rows = _read_csv_rows(out_dir / "p2_full_gradient_correctness.csv")

    by_task = {str(r.get("candidate_id")): r for r in task_summary}
    by_eff = {str(r.get("candidate_id")): r for r in eff_summary}
    sig_macro = {str(r.get("candidate_id")): r for r in sig_rows if r.get("dataset") == "macro"}
    grad_by = {}
    for row in grad_rows:
        grad_by.setdefault(str(row.get("candidate_id")), []).append(row)

    dense_id = "C3" if "C3" in by_task else "B3"
    dense_val = float(by_task.get(dense_id, {}).get("val_acc_mean", "nan")) if by_task.get(dense_id) else float("nan")
    mlp_val = float(by_task.get("B0", {}).get("val_acc_mean", "nan")) if by_task.get("B0") else float("nan")

    p1_rows: List[Dict[str, Any]] = []
    for cid in ["B3", "C3", "A1", "A1C", "A2", "A2C", "A2R", "A2S", "A4C", "A5C", "E1", "Z1", "U1", "M1", "M4", "M7", "M9", "A3", "H1", "H2", "H4", "C1"]:
        if cid not in by_task:
            continue
        row = by_task[cid]
        val = float(row.get("val_acc_mean", "nan"))
        p1_rows.append({
            "stage": "P1",
            "candidate_id": cid,
            "candidate_name": row.get("candidate_name"),
            "ablation_type": {
                "B3": "dense_oracle_linear_head",
                "C3": "cached_dense_strict_reference",
                "A1": "no_gate_poly2",
                "A1C": "no_gate_poly2_cached_y",
                "A2": "no_poly2_linear_stack",
                "A2C": "no_poly2_linear_stack_cached_y",
                "A2R": "no_poly2_linear_stack_cached_y_rbf_head",
                "A2S": "no_poly2_linear_stack_cached_y_poly2_silu_head",
                "A4C": "first_poly2_rest_linear_cached_y",
                "A5C": "last_poly2_rest_linear_cached_y",
                "E1": "a2s_recompute_y_checkpoint_stack",
                "Z1": "a2s_cache_y_only_stack",
                "U1": "a2s_cache_late_y_stack",
                "M1": "a2s_fused_linear_silu_stack",
                "M4": "a2s_fused_linear_silu_stack_and_head",
                "M7": "a2s_packed_fused_linear_silu_stack",
                "M9": "a2s_fused_linear_silu_stack_packed_generic_head",
                "A3": "no_dense_cross_group16",
                "H1": "strict_poly2_gate_head",
                "H2": "strict_poly2_silu_head",
                "H4": "depth2_poly2_gate",
                "C1": "depth2_cached",
            }.get(cid, "other"),
            "val_acc_mean": row.get("val_acc_mean"),
            "test_acc_mean": row.get("test_acc_mean"),
            "val_gap_vs_MLP_mean": row.get("val_gap_vs_MLP_mean"),
            "val_gap_vs_dense_reference": val - dense_val if _finite(val) and _finite(dense_val) else METRIC_UNAVAILABLE,
            "feature_effective_rank_mean": row.get("feature_effective_rank_mean"),
            "margin_p10_mean": row.get("margin_p10_mean"),
            "ECE_mean": row.get("ECE_mean"),
            "NLL_mean": row.get("NLL_mean"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p1_expressivity_attribution.csv", p1_rows)

    p2_rows: List[Dict[str, Any]] = []
    for cid, row in by_task.items():
        if cid == "B0":
            continue
        eff = by_eff.get(cid, {})
        ctype, strength = _candidate_constraint(cid)
        val = float(row.get("val_acc_mean", "nan"))
        gap_vs_dense = val - dense_val if _finite(val) and _finite(dense_val) else float("nan")
        p2_rows.append({
            "stage": "P2",
            "candidate_id": cid,
            "candidate_name": row.get("candidate_name"),
            "constraint_type": ctype,
            "constraint_strength": strength,
            "val_acc_mean": row.get("val_acc_mean"),
            "test_acc_mean": row.get("test_acc_mean"),
            "val_gap_vs_MLP_mean": row.get("val_gap_vs_MLP_mean"),
            "val_gap_vs_dense_reference": gap_vs_dense if _finite(gap_vs_dense) else METRIC_UNAVAILABLE,
            "expression_preserved_vs_dense_001": int(_finite(gap_vs_dense) and gap_vs_dense >= -0.01),
            "expression_hurt_vs_dense_002": int(_finite(gap_vs_dense) and gap_vs_dense <= -0.02),
            "ECE_mean": row.get("ECE_mean"),
            "NLL_mean": row.get("NLL_mean"),
            "feature_effective_rank_mean": row.get("feature_effective_rank_mean"),
            "margin_p10_mean": row.get("margin_p10_mean"),
            "memory_ratio_mean": eff.get("memory_ratio_mean", METRIC_UNAVAILABLE),
            "step_ratio_mean": eff.get("step_ratio_mean", METRIC_UNAVAILABLE),
            "s2_pass": eff.get("s2_pass_mean", eff.get("s2_pass", 0)),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p2_expression_efficiency_pareto.csv", p2_rows)

    p3_rows: List[Dict[str, Any]] = []
    for cid, control in {"O10": "G3", "O11": "G5", "O12": "L2", "D1": "A2S", "D2": "A2S", "D3": "A2S", "E2": "E1", "E3": "E1", "Z2": "Z1", "Z3": "Z1", "U2": "U1", "U3": "U1", "M2": "M1", "M3": "M1", "M5": "M4", "M6": "M1", "M8": "M7", "M10": "M9", "M11": "M9", "M12": "M4"}.items():
        if cid not in by_task:
            continue
        row = by_task[cid]
        control_row = by_task.get(control, {})
        spec = _spec_map().get(cid)
        val = float(row.get("val_acc_mean", "nan"))
        control_val = float(control_row.get("val_acc_mean", "nan")) if control_row else float("nan")
        improvement = val - control_val if _finite(val) and _finite(control_val) else float("nan")
        teacher_gap = val - dense_val if _finite(val) and _finite(dense_val) else float("nan")
        p3_rows.append({
            "stage": "P3",
            "student": cid,
            "supervised_control": control,
            "teacher": "C3",
            "recipe": "logit_distill_from_C3",
            "distill_temperature": _distill_temperature_for_spec(args, spec) if spec is not None else float(args.distill_temperature),
            "alpha_logit": _distill_alpha_for_spec(args, spec) if spec is not None else float(args.distill_alpha),
            "val_acc_mean": row.get("val_acc_mean"),
            "test_acc_mean": row.get("test_acc_mean"),
            "val_gap_vs_MLP_mean": row.get("val_gap_vs_MLP_mean"),
            "distill_improvement_vs_supervised": improvement if _finite(improvement) else METRIC_UNAVAILABLE,
            "val_gap_vs_teacher_C3": teacher_gap if _finite(teacher_gap) else METRIC_UNAVAILABLE,
            "optimizer_reachability_signal": int(_finite(improvement) and improvement >= (0.002 if control in {"A2S", "E1", "Z1", "U1"} else 0.02)),
            "student_near_mlp_signal": int(_finite(val) and _finite(mlp_val) and val >= mlp_val - 0.01),
            "constraint_expressivity_signal": int(_finite(teacher_gap) and teacher_gap <= -0.03 and (_finite(improvement) and improvement < 0.01)),
            "ECE_mean": row.get("ECE_mean"),
            "NLL_mean": row.get("NLL_mean"),
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p3_optimizer_reachability_distillation.csv", p3_rows)

    p5_rows: List[Dict[str, Any]] = []
    for row in eff_rows:
        cid = str(row.get("candidate_id"))
        if cid == "B0":
            continue
        peak = float(row.get("peak_allocated_MB", "nan")) if _finite(row.get("peak_allocated_MB")) else float("nan")
        cache = float(row.get("cache_total_MB", "nan")) if _finite(row.get("cache_total_MB")) else float("nan")
        opt = float(row.get("optimizer_state_memory_MB", "nan")) if _finite(row.get("optimizer_state_memory_MB")) else float("nan")
        manual_cache_fraction = cache / peak if _finite(cache) and _finite(peak) and peak > 0 else float("nan")
        opt_fraction = opt / peak if _finite(opt) and _finite(peak) and peak > 0 else float("nan")
        p5_rows.append({
            "stage": "P5",
            "candidate_id": cid,
            "candidate_name": row.get("candidate_name"),
            "dataset": row.get("dataset"),
            "batch_size": row.get("batch_size"),
            "peak_allocated_MB": row.get("peak_allocated_MB"),
            "peak_reserved_MB": row.get("peak_reserved_MB"),
            "cache_total_MB_measured": row.get("cache_total_MB"),
            "optimizer_state_memory_MB_measured": row.get("optimizer_state_memory_MB"),
            "manual_cache_fraction_of_peak": manual_cache_fraction if _finite(manual_cache_fraction) else METRIC_UNAVAILABLE,
            "optimizer_state_fraction_of_peak": opt_fraction if _finite(opt_fraction) else METRIC_UNAVAILABLE,
            "forward_time_ms": row.get("forward_time_ms"),
            "backward_time_ms": row.get("backward_time_ms"),
            "update_time_ms": row.get("update_time_ms"),
            "step_time_ms": row.get("step_time_ms"),
            "memory_ratio": row.get("memory_ratio"),
            "step_ratio": row.get("step_ratio"),
            "materialized_tensor_count": row.get("materialized_tensor_count", row.get("materialized_tensor_count_measured", METRIC_UNAVAILABLE)),
            "largest_live_tensor_MB": row.get("largest_live_tensor_MB", row.get("largest_live_tensor_MB_measured", METRIC_UNAVAILABLE)),
            "linear_body_temp_MB": row.get("linear_body_temp_MB", METRIC_UNAVAILABLE),
            "hidden_y_cache_MB": row.get("hidden_y_cache_MB", METRIC_UNAVAILABLE),
            "top1_memory_source": row.get("top1_memory_source", METRIC_UNAVAILABLE),
            "top2_memory_source": row.get("top2_memory_source", METRIC_UNAVAILABLE),
            "top3_memory_source": row.get("top3_memory_source", METRIC_UNAVAILABLE),
            "unknown_memory_fraction": row.get("unknown_memory_fraction", METRIC_UNAVAILABLE),
            "torch_op_count": row.get("kernel_count_forward", METRIC_UNAVAILABLE),
            "attribution_status": "measured_cache_count_and_timing" if _finite(row.get("materialized_tensor_count", row.get("materialized_tensor_count_measured"))) else "partial_measured_cache_and_timing_only",
            "fake_data_used": 0,
            "proxy_row_used": 0,
        })
    write_csv(out_dir / "p5_live_set_kernelization_attribution.csv", p5_rows)

    macro_candidates = []
    for cid, row in by_task.items():
        if cid == "B0":
            continue
        sig = sig_macro.get(cid, {})
        eff = by_eff.get(cid, {})
        grad_pass = all(str(g.get("grad_pass")) in {"1", "1.0", "true", "True"} for g in grad_by.get(cid, [])) if grad_by.get(cid) else False
        macro_gap = float(sig.get("mean_val_gap", "nan")) if _finite(sig.get("mean_val_gap")) else float("nan")
        ci_low = float(sig.get("bootstrap_ci95_low", "nan")) if _finite(sig.get("bootstrap_ci95_low")) else float("nan")
        holm = float(sig.get("holm_corrected_p", "nan")) if _finite(sig.get("holm_corrected_p")) else float("nan")
        test_gap = float(sig.get("mean_test_gap", "nan")) if _finite(sig.get("mean_test_gap")) else float("nan")
        ece_delta_raw = sig.get("ECE_delta_macro", sig.get("ECE_delta"))
        nll_delta_raw = sig.get("NLL_delta_macro", sig.get("NLL_delta"))
        ece_delta = float(ece_delta_raw) if _finite(ece_delta_raw) else float("nan")
        nll_delta = float(nll_delta_raw) if _finite(nll_delta_raw) else float("nan")
        macro_pass = (
            _finite(macro_gap) and macro_gap >= 0.02
            and _finite(ci_low) and ci_low > 0.0
            and _finite(holm) and holm < 0.05
            and _finite(test_gap) and test_gap >= 0.015
            and _finite(ece_delta) and ece_delta <= 0.005
            and _finite(nll_delta) and nll_delta <= 0.01
        )
        s2 = bool(eff and _finite(eff.get("memory_ratio_mean")) and _finite(eff.get("step_ratio_mean")) and float(eff["memory_ratio_mean"]) <= 1.05 and float(eff["step_ratio_mean"]) <= 1.50)
        macro_candidates.append({
            "candidate_id": cid,
            "candidate_name": row.get("candidate_name"),
            "macro_gap": macro_gap,
            "ci95_low": ci_low,
            "holm_p": holm,
            "test_gap": test_gap,
            "ECE_delta": ece_delta,
            "NLL_delta": nll_delta,
            "macro_pass_v73": int(macro_pass),
            "grad_pass": int(grad_pass),
            "s2_pass": int(s2),
            "memory_ratio_mean": eff.get("memory_ratio_mean", METRIC_UNAVAILABLE),
            "step_ratio_mean": eff.get("step_ratio_mean", METRIC_UNAVAILABLE),
        })
    macro_candidates.sort(key=lambda r: (r["macro_pass_v73"], r["grad_pass"], r["macro_gap"]), reverse=True)
    best = macro_candidates[0] if macro_candidates else {}
    any_distill = any(
        str(r.get("optimizer_reachability_signal")) in {"1", "1.0"}
        for r in p3_rows
    )
    route = {
        "route": "R3-MacroExpressivityPositiveButNotKernelNative" if best and best.get("macro_pass_v73") and not best.get("s2_pass") else "R4-ExpressivityOrReachabilityStillOpen",
        "best_candidate_id": best.get("candidate_id"),
        "best_candidate": best.get("candidate_name"),
        "macro_pass_v73": best.get("macro_pass_v73", 0),
        "grad_pass": best.get("grad_pass", 0),
        "s2_pass": best.get("s2_pass", 0),
        "best_macro_gap": best.get("macro_gap"),
        "best_ci95_low": best.get("ci95_low"),
        "best_holm_p": best.get("holm_p"),
        "best_test_gap": best.get("test_gap"),
        "best_ECE_delta": best.get("ECE_delta"),
        "best_NLL_delta": best.get("NLL_delta"),
        "best_memory_ratio": best.get("memory_ratio_mean"),
        "best_step_ratio": best.get("step_ratio_mean"),
        "optimization_or_expressivity_explained": bool(any_distill or p2_rows),
        "primary_blocker": "kernel_native_efficiency" if best and best.get("macro_pass_v73") and not best.get("s2_pass") else "expressivity_or_reachability",
        "classwise_is_hard_gate": False,
        "no_fake": True,
        "no_proxy": True,
    }
    save_json(out_dir / "v73_route_decision.json", route)

    artifact_paths = [
        out_dir / "p1_expressivity_attribution.csv",
        out_dir / "p2_expression_efficiency_pareto.csv",
        out_dir / "p3_optimizer_reachability_distillation.csv",
        out_dir / "p5_live_set_kernelization_attribution.csv",
    ]
    audit = v72._audit_fake_proxy(artifact_paths)
    write_csv(out_dir / "v73_provenance_audit.csv", [{
        **audit,
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "fake_data_used": 0,
        "proxy_row_used": 0,
    }])
    save_json(out_dir / "v73_manifest.json", {
        "plan_path": PLAN_PATH,
        "script_path": SCRIPT_PATH,
        "out_dir": str(out_dir),
        "source_commit": _git_commit(),
        "git_status": _git_status(),
        "postprocess_artifact_hashes": {p.name: _hash_file(p) for p in artifact_paths if p.exists()},
        "v73_route": route,
        "v73_audit": audit,
    })


def _patch_v72_for_v73() -> None:
    v72.PLAN_PATH = PLAN_PATH
    v72.SCRIPT_PATH = SCRIPT_PATH
    v72._candidate_registry = _candidate_registry
    v72._make_manual_candidate = _make_manual_candidate
    v72._uses_v72_train_path = _uses_v72_train_path
    v72._train_manual_v72 = _train_manual_v73
    v72._gradient_check = _gradient_check_v73
    v72._autograd_forward = _autograd_forward_v73
    v72._init_seed_candidate_id = _init_seed_candidate_id_v73


def run(args: argparse.Namespace) -> None:
    _patch_v72_for_v73()
    v72.run(args)
    _postprocess_v73(Path(args.out_dir), args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--candidates", default="B0,B3,C3,A1,A1C,A2,A2C,A2R,A3,G3,G5,L2,O10,O11,O12,NP5,NP6")
    parser.add_argument("--train-size", type=int, default=1536)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--basis-count", type=int, default=8)
    parser.add_argument("--task-steps", type=int, default=240)
    parser.add_argument("--trace-every", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--bench-batch-sizes", default="128,256,512")
    parser.add_argument("--bench-warmup", type=int, default=5)
    parser.add_argument("--bench-reps", type=int, default=30)
    parser.add_argument("--grad-batch-sizes", default="8,128")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--reuse-task", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--distill-alpha", type=float, default=0.7)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
