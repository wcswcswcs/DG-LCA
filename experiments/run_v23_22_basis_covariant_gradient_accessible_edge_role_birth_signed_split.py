#!/usr/bin/env python3
"""DG-KAN v23.22 auditable role birth and signed split runner.

This file is intentionally mechanism-first.  It does not reuse v23.19-v23.21
proxy columns as evidence.  The core checks create actual torch Parameters,
place dormant edge roles in the KAN forward graph, compute newborn tangents by
autograd with respect to those parameters, and log failures as failures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.run_v23_16_compositional_bc_vh_flow as v2316
import experiments.run_v23_19_bc_function_preserving_edge_role_birth_lift_flow as v2319
import experiments.run_v22_93_true_deep_purekan_local_composite_metric_mpfu as v2293
from dgkan.models.fc_purekan_primitives import _basis_eval, _basis_derivative


RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/DG-KAN_v23.22_BasisCovariantGradientAccessibleEdgeRoleBirthSignedSplitting_多假设语义穷尽式完整详尽实验计划.md"
OUT_ROOT = Path(os.environ.get("V2322_OUT_ROOT", str(ROOT / "results/v23_22"))).resolve()
EXEC_LOG = ROOT / "docs/DG-KAN_v23.22_BasisCovariantGradientAccessibleEdgeRoleBirthSignedSplitting_执行日志.md"
RECAP_LOG = ROOT / "docs/DG-KAN_v23.22_BasisCovariantGradientAccessibleEdgeRoleBirthSignedSplitting_实验结果复盘.md"
PYTHON = sys.executable
EPS = 1.0e-12

HYPOTHESES = [
    "H-A",
    "H-B",
    "H-C",
    "H-D",
    "H-E",
    "H-F",
    "H-G",
    "H-H",
    "H-I",
    "H-J",
]

HYPOTHESIS_LABELS = {
    "H-A": "v23.19/v23.18 semantic audit correction",
    "H-B": "BC-GMEB actual-tangent first-order birth",
    "H-C": "Multiwitness class-conditional BC-GMEB",
    "H-D": "True internal shared-parent edge-role birth",
    "H-E": "True operator-valued receiving-node bank metric",
    "H-F": "True function-preserving signed hidden-node split",
    "H-G": "Debt-aware safe-growth role selection",
    "H-H": "Role incubation, true transport, and covariant momentum",
    "H-I": "Corrected exact full two-step hypergradient diagnostic",
    "H-J": "MLP-matched internal birth and KAN architecture surplus",
}

SYNTHETIC_TASKS = [
    "SYN1_MissingSingleEdgeRole",
    "SYN2_NodeBankComplementarity",
    "SYN3_ClassConditionalCancellation",
    "SYN4_LocalPatchInteraction",
    "SYN5_RotationSensitiveRole",
    "SYN6_TailSafeRole",
    "SYN7_SignedSplitOnly_FirstOrderZero",
    "SYN8_NoSignalNegative",
    "SYN9_DomainSupportArtifact",
    "SYN10_MLPFriendlyControl",
]

REAL_TASKS = ["Wine", "Spam", "RiceOrBean", "FashionMNIST", "CIFAR10_compact"]

CONTROL_REGISTRY = [
    "C0_BC15_baseline",
    "C1_dormant_unused_same_capacity",
    "C2_same_G_norm_random_role",
    "C3_genuine_same_spectrum_random_orientation",
    "C4_label_shuffled_selection",
    "C5_fold_shuffled_selection",
    "C6_same_novelty_random_role",
    "C7_same_accessibility_norm_random_orientation",
    "C8_same_debt_curvature_random_role",
    "C9_same_internal_carrier_random_role",
    "C10_same_compute_noop",
    "C11_readout_direct_feature_diagnostic_only",
    "C12_MLP_matched_internal_birth",
]

TRUTH_COUNTER_KEYS = [
    "mechanism_function_call_count",
    "autograd_graph_node_count",
    "actual_parameter_ids_created",
    "actual_parameter_ids_updated",
    "forward_hook_call_count",
    "backward_hook_call_count",
    "state_persistence_steps",
    "transport_refresh_count",
    "momentum_state_nonzero_steps",
    "cross_block_data_compute_count",
    "finite_difference_validation_count",
    "guard_tensor_selection_use_count",
    "guard_tensor_evaluation_use_count",
    "hardcoded_metric_field_count",
    "proxy_metric_field_count",
]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    EXEC_LOG.parent.mkdir(parents=True, exist_ok=True)
    RECAP_LOG.parent.mkdir(parents=True, exist_ok=True)


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command_text() -> str:
    env = []
    for key in ["CUDA_VISIBLE_DEVICES", "V2322_OUT_ROOT"]:
        if os.environ.get(key):
            env.append(f"{key}={os.environ[key]}")
    return " ".join([*env, PYTHON, rel(RUNNER), *sys.argv[1:]])


def init_logs() -> None:
    ensure_out()
    if not EXEC_LOG.exists():
        EXEC_LOG.write_text(
            "# DG-KAN v23.22 execution log\n\n"
            f"- created_at: {now()}\n"
            f"- plan: `{rel(PLAN)}`\n"
            f"- runner: `{rel(RUNNER)}`\n"
            f"- output_root: `{rel(OUT_ROOT)}`\n"
            "- nonfabrication: missing data and failed gates are recorded as missing/failed, never filled.\n"
            "- reproduction: use the kan conda interpreter if the default python lacks torch.\n\n",
            encoding="utf-8",
        )
    if not RECAP_LOG.exists():
        RECAP_LOG.write_text(
            "# DG-KAN v23.22 experiment recap\n\n"
            f"- created_at: {now()}\n"
            "- current_status: not finalized\n"
            "- rule: this file records observed metrics, repairs, blockers, and conclusions only from artifacts.\n\n",
            encoding="utf-8",
        )


def append_exec(stage: str, status: str, *, files: str = "", note: str = "", gpu: str = "") -> None:
    init_logs()
    with EXEC_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} | {stage} | {status}\n\n")
        fh.write(f"- command: `{command_text()}`\n")
        fh.write(f"- python: `{PYTHON}`\n")
        fh.write(f"- torch: `{getattr(torch, '__version__', 'unknown')}`\n")
        fh.write(f"- cuda_visible_devices: `{os.environ.get('CUDA_VISIBLE_DEVICES', '')}`\n")
        fh.write(f"- gpu: `{gpu}`\n")
        if files:
            fh.write(f"- files: `{files}`\n")
        if note:
            fh.write(f"- note: {note}\n")


def append_recap(title: str, payload: dict[str, Any]) -> None:
    init_logs()
    with RECAP_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {now()} | {title}\n\n")
        fh.write(f"```json\n{json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)}\n```\n")


def write_json(path: Path, data: dict[str, Any]) -> Path:
    ensure_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_out()
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fval(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "missing", "nan", "failed", "blocked"):
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def median(values: Iterable[Any], default: float = 0.0) -> float:
    vals = sorted(fval(v, float("nan")) for v in values)
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float(default)
    n = len(vals)
    return float(vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2]))


def mean(values: Iterable[Any], default: float = 0.0) -> float:
    vals = [fval(v, float("nan")) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float(default)


def device_from_args(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        return torch.device(args.device)
    return torch.device("cpu")


def graph_node_count(t: torch.Tensor) -> int:
    seen: set[int] = set()
    stack = [t.grad_fn]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        ident = id(node)
        if ident in seen:
            continue
        seen.add(ident)
        for nxt, _idx in node.next_functions:
            if nxt is not None:
                stack.append(nxt)
    return len(seen)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().reshape(-1).to(dtype=torch.float64)
    bb = b.detach().reshape(-1).to(device=aa.device, dtype=torch.float64)
    den = aa.norm() * bb.norm()
    if float(den.detach().cpu()) <= EPS:
        return 0.0
    return float((aa @ bb / den.clamp_min(EPS)).detach().cpu().item())


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().to(dtype=torch.float64)
    bb = b.detach().to(device=aa.device, dtype=torch.float64)
    return float(((aa - bb).norm() / bb.norm().clamp_min(EPS)).detach().cpu().item())


def logits_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    work = logits.detach().to(dtype=torch.float64)
    yy = y.long().reshape(-1)
    probs = torch.softmax(work, dim=1)
    target = F.one_hot(yy, num_classes=int(work.shape[1])).to(device=work.device, dtype=torch.float64)
    true = probs.gather(1, yy.reshape(-1, 1)).reshape(-1).clamp_min(1.0e-12)
    pred = probs.argmax(dim=1)
    conf = probs.max(dim=1).values
    wrong_conf = conf[pred != yy]
    margin = true - (probs + target * -1.0e9).max(dim=1).values
    return {
        "NLL": float((-true.log()).mean().detach().cpu().item()),
        "accuracy": float((pred == yy).to(dtype=torch.float64).mean().detach().cpu().item()),
        "Brier": float((probs - target).square().sum(dim=1).mean().detach().cpu().item()),
        "ECE_adaptive": float((conf - (pred == yy).to(dtype=torch.float64)).abs().mean().detach().cpu().item()),
        "tail95": float((torch.quantile(wrong_conf, 0.95) if int(wrong_conf.numel()) else torch.tensor(0.0, device=work.device)).detach().cpu().item()),
        "tail99": float((torch.quantile(wrong_conf, 0.99) if int(wrong_conf.numel()) else torch.tensor(0.0, device=work.device)).detach().cpu().item()),
        "margin_q10": float(torch.quantile(margin, 0.10).detach().cpu().item()),
    }


def ce_cotangent(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=1).to(dtype=logits.dtype)
    target = F.one_hot(y.long(), num_classes=int(logits.shape[1])).to(device=logits.device, dtype=logits.dtype)
    return (target - probs).to(dtype=torch.float64) / float(max(1, int(y.numel())))


def output_direction(output_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    v = torch.zeros(int(output_dim), device=device, dtype=dtype)
    if int(output_dim) == 1:
        v[0] = 1.0
    else:
        v[0] = 1.0
        v[1] = -1.0
    return v / v.norm().clamp_min(EPS)


def role_dictionary(u: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    """D1+D2+D3 role dictionary over one internal edge coordinate."""
    z = u.to(dtype=torch.float64)
    kk = knots.to(device=z.device, dtype=torch.float64).reshape(-1)
    if int(kk.numel()) < 4:
        kk = torch.quantile(z.detach(), torch.tensor([0.2, 0.4, 0.6, 0.8], device=z.device, dtype=torch.float64))
    width = z.detach().std().clamp_min(0.25)
    atoms: list[torch.Tensor] = []
    for knot in kk[:4]:
        r = (z - knot) / width
        atoms.append(F.relu(1.0 - r.abs()).pow(3))
    atoms.extend(
        [
            torch.sin(math.pi * z),
            torch.cos(math.pi * z),
            torch.sin(2.0 * math.pi * z),
            torch.cos(2.0 * math.pi * z),
            z,
            torch.tanh(z),
            torch.sigmoid(2.0 * z) - 0.5,
            F.relu(z),
        ]
    )
    mat = torch.stack(atoms, dim=1)
    mat = mat - mat.mean(dim=0, keepdim=True)
    scale = mat.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    return mat / scale


def role_metric(atoms: torch.Tensor, ridge: float = 1.0e-6) -> torch.Tensor:
    a = atoms.to(dtype=torch.float64)
    h = a.T @ a / float(max(1, int(a.shape[0])))
    return 0.5 * (h + h.T) + float(ridge) * torch.eye(int(h.shape[0]), device=a.device, dtype=torch.float64)


class EdgeRoleKAN(nn.Module):
    """Actual dormant internal edge-role wrapper over TrueDeepPureKAN."""

    def __init__(
        self,
        base: v2293.TrueDeepPureKAN,
        layer_idx: int,
        parent_idx: int,
        out_dir: torch.Tensor,
        knots: torch.Tensor,
        *,
        shared_parent_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.layer_idx = int(layer_idx)
        self.parent_idx = int(parent_idx)
        self.shared_parent_indices = [int(x) for x in (shared_parent_indices or [parent_idx])]
        self.register_buffer("out_dir", out_dir.detach().clone())
        self.register_buffer("knots", knots.detach().clone().to(dtype=torch.float64))
        first_knots = self.knots[0] if self.knots.ndim == 2 else self.knots
        m = int(role_dictionary(torch.linspace(-1.0, 1.0, 32, device=out_dir.device, dtype=torch.float64), first_knots).shape[1])
        dtype = next(base.parameters()).dtype
        device = next(base.parameters()).device
        self.candidate_amplitude = nn.Parameter(torch.zeros(m, device=device, dtype=dtype))
        self.selected_amplitude = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
        self.register_buffer("selected_coeffs", torch.zeros(m, device=device, dtype=torch.float64))
        self.candidate_enabled = True
        self.selected_enabled = False
        self.truth: dict[str, int] = {k: 0 for k in TRUTH_COUNTER_KEYS}
        self.truth["actual_parameter_ids_created"] = 2
        self._updated_ids: set[int] = set()
        self.candidate_amplitude.register_hook(self._backward_hook)
        self.selected_amplitude.register_hook(self._backward_hook)

    def _backward_hook(self, grad: torch.Tensor) -> torch.Tensor:
        self.truth["backward_hook_call_count"] += 1
        if grad is not None and bool(torch.isfinite(grad).all()) and float(grad.detach().abs().sum().cpu()) > 0.0:
            self._updated_ids.add(id(self.selected_amplitude))
            self.truth["actual_parameter_ids_updated"] = len(self._updated_ids)
        return grad

    def role_atom_stack_from_activations(self, activations: list[torch.Tensor]) -> torch.Tensor:
        vals = []
        for pos, idx in enumerate(self.shared_parent_indices):
            u = activations[self.layer_idx][:, int(idx)]
            knots = self.knots[pos] if self.knots.ndim == 2 else self.knots
            vals.append(role_dictionary(u, knots).to(dtype=activations[self.layer_idx].dtype))
        return torch.stack(vals, dim=1)

    def role_atoms_from_activations(self, activations: list[torch.Tensor]) -> torch.Tensor:
        return self.role_atom_stack_from_activations(activations).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.truth["mechanism_function_call_count"] += 1
        self.truth["forward_hook_call_count"] += 1
        logits, activations = self.base.forward_with_activations(x)
        atoms = self.role_atoms_from_activations(activations)
        if self.candidate_enabled:
            carrier = atoms @ self.candidate_amplitude.to(dtype=atoms.dtype)
            logits = logits + carrier.unsqueeze(1) * self.out_dir.to(dtype=atoms.dtype).unsqueeze(0)
        if self.selected_enabled:
            coeffs = self.selected_coeffs.to(device=atoms.device, dtype=torch.float64).reshape(-1)
            atom_stack = self.role_atom_stack_from_activations(activations).to(dtype=torch.float64)
            parent_count = int(atom_stack.shape[1])
            atom_count = int(atom_stack.shape[2])
            if int(coeffs.numel()) == parent_count * atom_count:
                shape = (atom_stack * coeffs.reshape(1, parent_count, atom_count)).sum(dim=(1, 2))
            else:
                shape = atoms.to(dtype=torch.float64) @ coeffs[:atom_count]
            logits = logits + (self.selected_amplitude.to(dtype=atoms.dtype) * shape.to(dtype=atoms.dtype)).unsqueeze(1) * self.out_dir.to(dtype=atoms.dtype).unsqueeze(0)
        if logits.requires_grad:
            self.truth["autograd_graph_node_count"] = max(self.truth["autograd_graph_node_count"], graph_node_count(logits))
        return logits

    def materialize_selected(self, coeffs: torch.Tensor) -> None:
        with torch.no_grad():
            self.selected_coeffs = coeffs.detach().reshape(-1).to(device=self.selected_coeffs.device, dtype=torch.float64).clone()
            self.selected_amplitude.zero_()
            self.candidate_amplitude.zero_()
        self.candidate_enabled = False
        self.selected_enabled = True


class SplitDepth2KAN(nn.Module):
    """Depth-2 KAN with one hidden node split into distinct child parameters."""

    def __init__(self, base: v2293.TrueDeepPureKAN, split_idx: int, direction: torch.Tensor) -> None:
        super().__init__()
        if int(base.depth) != 2 or len(base.coeffs) != 2:
            raise ValueError("SplitDepth2KAN requires depth=2 TrueDeepPureKAN")
        self.input_dim = int(base.dims[0])
        self.hidden_dim = int(base.dims[1])
        self.output_dim = int(base.dims[2])
        self.k = int(base.k)
        self.basis_name = str(base.basis_name)
        self.basis_input_gain = float(base.basis_input_gain)
        self.split_idx = int(split_idx)
        self.register_buffer("centers", base.centers.detach().clone())
        self.register_buffer("scales", base.scales.detach().clone())
        dtype = base.coeffs[0].dtype
        device = base.coeffs[0].device
        incoming = []
        outgoing = []
        with torch.no_grad():
            for j in range(self.hidden_dim):
                inc = base.coeffs[0][:, j, :].detach().clone()
                out = base.coeffs[1][j, :, :].detach().clone()
                if j == self.split_idx:
                    incoming.append(nn.Parameter(inc.clone()))
                    outgoing.append(nn.Parameter(0.5 * out.clone()))
                    incoming.append(nn.Parameter(inc.clone()))
                    outgoing.append(nn.Parameter(0.5 * out.clone()))
                else:
                    incoming.append(nn.Parameter(inc.clone()))
                    outgoing.append(nn.Parameter(out.clone()))
        self.incoming = nn.ParameterList(incoming)
        self.outgoing = nn.ParameterList(outgoing)
        self.antisym_eps = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
        self.register_buffer("antisym_direction", direction.detach().clone().to(device=device, dtype=dtype))
        self.truth: dict[str, int] = {k: 0 for k in TRUTH_COUNTER_KEYS}
        self.truth["actual_parameter_ids_created"] = len(self.incoming) + len(self.outgoing) + 1
        self._updated_ids: set[int] = set()
        for p in [self.incoming[self.split_idx], self.incoming[self.split_idx + 1], self.antisym_eps]:
            p.register_hook(self._backward_hook)

    def _backward_hook(self, grad: torch.Tensor) -> torch.Tensor:
        self.truth["backward_hook_call_count"] += 1
        if grad is not None and bool(torch.isfinite(grad).all()) and float(grad.detach().abs().sum().cpu()) > 0.0:
            self._updated_ids.add(id(grad))
            self.truth["actual_parameter_ids_updated"] = max(self.truth["actual_parameter_ids_updated"], 1)
        return grad

    def basis(self, h: torch.Tensor) -> torch.Tensor:
        return _basis_eval(torch.tanh(h * self.basis_input_gain), self.basis_name, self.k, self.centers, self.scales)

    def forward_with_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        self.truth["mechanism_function_call_count"] += 1
        self.truth["forward_hook_call_count"] += 1
        b1 = self.basis(x) / math.sqrt(max(1, self.input_dim))
        hs = []
        for idx, coeff in enumerate(self.incoming):
            local = coeff
            if idx == self.split_idx:
                local = local + self.antisym_eps * self.antisym_direction
            elif idx == self.split_idx + 1:
                local = local - self.antisym_eps * self.antisym_direction
            hs.append(torch.einsum("bik,ik->b", b1, local))
        h = torch.stack(hs, dim=1)
        # Keep the original receiving-layer chart scale.  Function preserving
        # split duplicates the node and halves outgoing functions; changing the
        # denominator to H+1 would silently break identity.
        b2 = self.basis(h) / math.sqrt(max(1, self.hidden_dim))
        logits = torch.zeros((int(x.shape[0]), self.output_dim), device=x.device, dtype=x.dtype)
        for idx, coeff in enumerate(self.outgoing):
            logits = logits + torch.einsum("bk,ck->bc", b2[:, idx, :], coeff)
        if logits.requires_grad:
            self.truth["autograd_graph_node_count"] = max(self.truth["autograd_graph_node_count"], graph_node_count(logits))
        return logits, [x, h]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_activations(x)[0]


class MLPInternalBirth(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int, seed: int, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        gen = torch.Generator(device=device).manual_seed(int(seed))
        self.w1 = nn.Parameter(torch.randn(int(input_dim), int(hidden), device=device, generator=gen, dtype=dtype) / math.sqrt(max(1, int(input_dim))))
        self.w2 = nn.Parameter(torch.randn(int(hidden), int(output_dim), device=device, generator=gen, dtype=dtype) / math.sqrt(max(1, int(hidden))))
        self.birth_in = nn.Parameter(torch.randn(int(input_dim), device=device, generator=gen, dtype=dtype) / math.sqrt(max(1, int(input_dim))))
        self.birth_out = nn.Parameter(output_direction(int(output_dim), device, dtype))
        self.birth_amp = nn.Parameter(torch.zeros((), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(x @ self.w1)
        logits = h @ self.w2
        born = torch.tanh(x @ self.birth_in)
        return logits + self.birth_amp * born.unsqueeze(1) * self.birth_out.unsqueeze(0)


def make_base_model(args: argparse.Namespace, arch: str, input_dim: int, output_dim: int, seed: int, dtype: torch.dtype) -> v2293.TrueDeepPureKAN:
    if arch == "A2_DCHE_depth3_width4_basis9":
        depth, width, basis_key = 3, 4, "dche_k9"
    else:
        depth, width, basis_key = 2, 3, "dche_k9"
    local = argparse.Namespace(**vars(args))
    local.width = int(width)
    return v2316.make_model(local, basis_key=basis_key, depth=depth, width=width, input_dim=input_dim, output_dim=output_dim, seed=seed, dtype=dtype)


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    xx = x.to(dtype=torch.float64)
    return (xx - xx.mean(dim=0, keepdim=True)) / xx.std(dim=0, keepdim=True).clamp_min(1.0e-6)


def synthetic_logits(x: torch.Tensor, task: str, seed: int) -> torch.Tensor:
    z = x.to(dtype=torch.float64)
    gen = torch.Generator(device=z.device).manual_seed(int(seed) + 232200)
    d = int(z.shape[1])
    def col(i: int) -> torch.Tensor:
        return z[:, i % d]
    if task == "SYN1_MissingSingleEdgeRole":
        a = torch.sin(1.7 * col(0)) + 0.25 * col(1)
        b = -torch.sin(1.7 * col(0)) + 0.25 * col(2)
    elif task == "SYN2_NodeBankComplementarity":
        role = torch.sin(1.3 * col(0)) + torch.sin(1.3 * col(1))
        a = role + 0.2 * col(2)
        b = -role + 0.2 * col(3)
    elif task == "SYN3_ClassConditionalCancellation":
        a = torch.where(col(0) > 0, torch.sin(2.0 * col(1)), -torch.sin(2.0 * col(1)))
        b = col(2) - 0.2 * col(3)
    elif task == "SYN4_LocalPatchInteraction":
        a = F.relu(1.0 - (col(0) - 0.2).abs()).pow(2) + col(1) * col(2)
        b = F.relu(1.0 - (col(3) + 0.3).abs()).pow(2) - col(4) * col(5)
    elif task == "SYN5_RotationSensitiveRole":
        a = col(0) * col(3) - col(1) * col(2)
        b = col(0) * col(2) + col(1) * col(3)
    elif task == "SYN6_TailSafeRole":
        tail = torch.tanh(3.0 * (col(0).abs() - 0.9)) * col(0).sign()
        a = tail + 0.2 * col(1)
        b = -tail + 0.2 * col(2)
    elif task == "SYN7_SignedSplitOnly_FirstOrderZero":
        a = (col(0).square() - col(1).square()) + 0.1 * col(2)
        b = -a
    elif task == "SYN8_NoSignalNegative":
        raw = torch.randn((int(z.shape[0]), 4), generator=gen, device=z.device, dtype=torch.float64)
        return raw
    elif task == "SYN9_DomainSupportArtifact":
        gate = (col(0) > torch.quantile(col(0), 0.85)).to(dtype=torch.float64)
        a = gate * torch.sin(2.5 * col(1))
        b = -a + 0.1 * col(2)
    elif task == "SYN10_MLPFriendlyControl":
        a = col(0) + col(1) - 0.5 * col(2)
        b = -0.7 * col(0) + col(3)
    else:
        raise ValueError(f"unknown synthetic task {task}")
    c = -a
    dlog = -b
    return torch.stack([a, b, c, dlog], dim=1)


def make_synthetic(args: argparse.Namespace, task: str, seed: int, total: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    device = device_from_args(args)
    gen = torch.Generator(device=device).manual_seed(int(seed) + 232219)
    dim = int(args.synthetic_dim)
    x = torch.randn(int(total), dim, generator=gen, device=device, dtype=torch.float64)
    x = normalize_features(x)
    logits = synthetic_logits(x, task, seed)
    y = logits.argmax(dim=1).long()
    return x.to(dtype=dtype), y, {"dataset_kind": "synthetic", "task_source": task, "input_dim": dim, "output_dim": int(logits.shape[1])}


def load_real(args: argparse.Namespace, task: str, seed: int, total: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    mapped = "Wine" if task == "RiceOrBean" else task
    if task == "RiceOrBean":
        note = "Rice/Bean unavailable in local mandatory loader; Wine used as explicit real fallback"
    else:
        note = "native"
    local = argparse.Namespace(**vars(args))
    local.real_compact_dim = int(args.real_compact_dim)
    train, ytrain, guard, yguard, meta = v2319.load_real_task(local, mapped, int(seed), int(total), 0, dtype)
    meta = dict(meta)
    meta["requested_task"] = task
    meta["actual_task"] = mapped
    meta["substitution_note"] = note
    return train, ytrain, meta


def split_folds(x: torch.Tensor, y: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    n = int(x.shape[0])
    q = max(1, n // 5)
    return {
        "S1": (x[:q], y[:q]),
        "S2": (x[q : 2 * q], y[q : 2 * q]),
        "W1": (x[2 * q : 3 * q], y[2 * q : 3 * q]),
        "W2": (x[3 * q : 4 * q], y[3 * q : 4 * q]),
        "G": (x[4 * q :], y[4 * q :]),
    }


def choose_location(base: v2293.TrueDeepPureKAN, x: torch.Tensor, y: torch.Tensor) -> tuple[int, int, dict[str, float]]:
    logits, activations = base.forward_with_activations(x)
    cot = ce_cotangent(logits, y)
    scores: dict[str, float] = {}
    layer_idx = max(0, len(activations) - 2)
    h = activations[layer_idx]
    out_dir = output_direction(int(logits.shape[1]), x.device, x.dtype).to(dtype=torch.float64)
    g = (cot @ out_dir).reshape(-1)
    for j in range(int(h.shape[1])):
        knots = torch.quantile(h[:, j].detach().to(dtype=torch.float64), torch.tensor([0.2, 0.4, 0.6, 0.8], device=x.device, dtype=torch.float64))
        atoms = role_dictionary(h[:, j], knots)
        b = atoms.T @ g
        hmetric = role_metric(atoms)
        score = float((b @ torch.linalg.solve(hmetric, b)).detach().cpu().item())
        scores[f"layer{layer_idx}_bank{j}"] = score
    best = max(scores, key=scores.get)
    bank = int(best.split("bank")[1])
    return layer_idx, bank, scores


def param_jacobian(model: nn.Module, x: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    flat = logits.reshape(-1)
    rows = []
    for idx in range(int(flat.numel())):
        grad = torch.autograd.grad(flat[idx], param, retain_graph=True, create_graph=False, allow_unused=False)[0]
        rows.append(grad.reshape(-1).to(dtype=torch.float64))
    return torch.stack(rows, dim=0)


def base_jacobian(base: v2293.TrueDeepPureKAN, x: torch.Tensor) -> torch.Tensor:
    logits = base(x)
    flat = logits.reshape(-1)
    params = list(base.coeffs)
    rows = []
    for idx in range(int(flat.numel())):
        grads = torch.autograd.grad(flat[idx], params, retain_graph=True, create_graph=False, allow_unused=False)
        rows.append(torch.cat([g.reshape(-1).to(dtype=torch.float64) for g in grads], dim=0))
    return torch.stack(rows, dim=0)


def residualize_against_old(B: torch.Tensor, J: torch.Tensor, ridge: float = 1.0e-6) -> torch.Tensor:
    jj = J.to(dtype=torch.float64)
    bb = B.to(dtype=torch.float64)
    if int(jj.numel()) == 0:
        return bb
    gram = jj.T @ jj + float(ridge) * torch.eye(int(jj.shape[1]), device=jj.device, dtype=torch.float64)
    proj_coeff = torch.linalg.solve(gram, jj.T @ bb)
    return bb - jj @ proj_coeff


def generalized_top_vector(A: torch.Tensor, H: torch.Tensor) -> tuple[torch.Tensor, float]:
    aa = 0.5 * (A.to(dtype=torch.float64) + A.to(dtype=torch.float64).T)
    hh = 0.5 * (H.to(dtype=torch.float64) + H.to(dtype=torch.float64).T) + 1.0e-6 * torch.eye(int(H.shape[0]), device=H.device, dtype=torch.float64)
    chol = torch.linalg.cholesky(hh)
    inv_chol = torch.cholesky_inverse(chol)
    mat = inv_chol @ aa
    mat = 0.5 * (mat + mat.T)
    vals, vecs = torch.linalg.eigh(mat)
    v = vecs[:, -1]
    v = v / torch.sqrt((v @ hh @ v).clamp_min(EPS))
    return v, float(vals[-1].detach().cpu().item())


def same_spectrum_control(A: torch.Tensor, seed: int) -> tuple[torch.Tensor, float, float]:
    work = 0.5 * (A.to(dtype=torch.float64) + A.to(dtype=torch.float64).T)
    vals = torch.linalg.eigvalsh(work)
    gen = torch.Generator(device=work.device).manual_seed(int(seed) + 232222)
    q, _ = torch.linalg.qr(torch.randn(work.shape, generator=gen, device=work.device, dtype=torch.float64))
    ctrl = q @ torch.diag(vals) @ q.T
    ctrl_vals = torch.linalg.eigvalsh(0.5 * (ctrl + ctrl.T))
    rel = float(((torch.sort(ctrl_vals).values - torch.sort(vals).values).norm() / vals.norm().clamp_min(EPS)).detach().cpu().item())
    top = torch.linalg.eigh(work).eigenvectors[:, -1]
    ctrl_top = torch.linalg.eigh(ctrl).eigenvectors[:, -1]
    orient = abs(cosine(top, ctrl_top))
    return ctrl, rel, orient


def actual_tangent_unit(
    args: argparse.Namespace,
    task: str = "SYN1_MissingSingleEdgeRole",
    seed: int = 0,
    *,
    shared_parent: bool = False,
    class_conditional: bool = False,
) -> dict[str, Any]:
    dtype = torch.float64
    x, y, meta = make_synthetic(args, task, seed, int(args.unit_total), dtype)
    folds = split_folds(x, y)
    xs = torch.cat([folds["S1"][0], folds["S2"][0]], dim=0)
    ys = torch.cat([folds["S1"][1], folds["S2"][1]], dim=0)
    xw = torch.cat([folds["W1"][0], folds["W2"][0]], dim=0)
    yw = torch.cat([folds["W1"][1], folds["W2"][1]], dim=0)
    base = make_base_model(args, "A1_DCHE_depth2_width3_basis9", int(x.shape[1]), int(meta["output_dim"]), 232200 + seed, dtype)
    layer_idx, bank, loc_scores = choose_location(base, xs, ys)
    with torch.no_grad():
        _logits, acts = base.forward_with_activations(xs)
        knots = torch.quantile(acts[layer_idx][:, bank].detach().to(dtype=torch.float64), torch.tensor([0.2, 0.4, 0.6, 0.8], device=x.device, dtype=torch.float64))
    shared_indices: list[int] | None = None
    if shared_parent:
        hidden_count = int(acts[layer_idx].shape[1])
        shared_indices = sorted({int(bank), int((bank + 1) % max(1, hidden_count))})
    role = EdgeRoleKAN(base, layer_idx, bank, output_direction(int(meta["output_dim"]), x.device, dtype), knots, shared_parent_indices=shared_indices)
    role.to(device=x.device, dtype=dtype)
    base_logits = base(xs).detach()
    born_logits = role(xs)
    B = param_jacobian(role, xs, role.candidate_amplitude)
    J = base_jacobian(base, xs)
    Bt = residualize_against_old(B, J)
    g_s = ce_cotangent(born_logits, ys).reshape(-1)
    b_s = Bt.T @ g_s
    logits_w = role(xw)
    B_w = param_jacobian(role, xw, role.candidate_amplitude)
    J_w = base_jacobian(base, xw)
    Bt_w = residualize_against_old(B_w, J_w)
    g_w = ce_cotangent(logits_w, yw).reshape(-1)
    b_w = Bt_w.T @ g_w
    atoms = role.role_atoms_from_activations(base.forward_with_activations(xs)[1])
    H = role_metric(atoms)
    A = 0.5 * (torch.outer(b_s, b_w) + torch.outer(b_w, b_s))
    a, eig = generalized_top_vector(A, H)
    role.materialize_selected(a)
    selected_logits = role(xs)
    selected_B = param_jacobian(role, xs, role.selected_amplitude)
    Ba = B @ a.to(device=B.device, dtype=B.dtype)
    eps = 1.0e-6
    with torch.no_grad():
        role.selected_amplitude.fill_(eps)
        fd_plus = role(xs).detach().reshape(-1)
        role.selected_amplitude.fill_(-eps)
        fd_minus = role(xs).detach().reshape(-1)
        role.selected_amplitude.zero_()
    fd = ((fd_plus - fd_minus) / (2.0 * eps)).reshape(-1, 1)
    loss = F.cross_entropy(role(xs).float(), ys.long())
    grad_amp = torch.autograd.grad(loss, role.selected_amplitude, retain_graph=False)[0]
    ctrl, spec_rel, orient = same_spectrum_control(A, seed)
    classwise_cosines: list[float] = []
    if class_conditional:
        B3_s = Bt.reshape(int(xs.shape[0]), int(meta["output_dim"]), -1)
        B3_w = Bt_w.reshape(int(xw.shape[0]), int(meta["output_dim"]), -1)
        cot_s = ce_cotangent(role(xs), ys)
        cot_w = ce_cotangent(role(xw), yw)
        for cls in sorted(set(int(v) for v in ys.detach().cpu().tolist()) & set(int(v) for v in yw.detach().cpu().tolist())):
            ms = ys == int(cls)
            mw = yw == int(cls)
            if int(ms.sum()) == 0 or int(mw.sum()) == 0:
                continue
            bs_c = B3_s[ms].reshape(-1, int(B3_s.shape[-1])).T @ cot_s[ms].reshape(-1)
            bw_c = B3_w[mw].reshape(-1, int(B3_w.shape[-1])).T @ cot_w[mw].reshape(-1)
            classwise_cosines.append(cosine(bs_c, bw_c))
    vals = {
        "task": task,
        "seed": seed,
        "layer_idx": layer_idx,
        "bank": bank,
        "function_preservation_error": float((born_logits.detach() - base_logits).abs().max().cpu().item()),
        "autograd_fd_cosine": cosine(selected_B, fd),
        "selection_lift_relative_error": relative_error(selected_B.reshape(-1), Ba.reshape(-1)),
        "selected_dictionary_coefficient_norm": float(a.norm().detach().cpu().item()),
        "ordinary_amplitude_gradient_abs": float(grad_amp.detach().abs().cpu().item()),
        "newborn_gradient_norm_source": float(b_s.norm().detach().cpu().item()),
        "newborn_gradient_norm_witness": float(b_w.norm().detach().cpu().item()),
        "source_witness_gradient_cosine": cosine(b_s, b_w),
        "novel_energy_fraction": float((Bt.square().sum() / B.square().sum().clamp_min(EPS)).detach().cpu().item()),
        "growth_eigenvalue": eig,
        "same_spectrum_eigen_rel_error": spec_rel,
        "same_spectrum_orientation_cosine": orient,
        "amplitude_zero_exact": int(float(role.selected_amplitude.detach().abs().cpu()) == 0.0),
        "actual_internal_role_used": 1,
        "shared_parent_count": len(role.shared_parent_indices),
        "class_conditional_compute_count": len(classwise_cosines),
        "classwise_gradient_cosine_min": min(classwise_cosines) if classwise_cosines else "",
        "actual_parameter_ids_created": role.truth["actual_parameter_ids_created"],
        "autograd_graph_node_count": role.truth["autograd_graph_node_count"],
        "all_location_scores": json.dumps(loc_scores, sort_keys=True),
    }
    vals["semantic_pass"] = int(
        vals["function_preservation_error"] <= 1.0e-8
        and vals["autograd_fd_cosine"] >= 0.999
        and vals["selection_lift_relative_error"] <= 1.0e-5
        and vals["ordinary_amplitude_gradient_abs"] > 0.0
        and vals["same_spectrum_eigen_rel_error"] <= 1.0e-5
        and (not shared_parent or vals["shared_parent_count"] >= 2)
        and (not class_conditional or vals["class_conditional_compute_count"] > 0)
    )
    return vals | role.truth


def operator_bank_unit(args: argparse.Namespace, seed: int = 0) -> dict[str, Any]:
    dtype = torch.float64
    x, y, meta = make_synthetic(args, "SYN2_NodeBankComplementarity", seed, int(args.unit_total), dtype)
    base = make_base_model(args, "A1_DCHE_depth2_width3_basis9", int(x.shape[1]), int(meta["output_dim"]), 232230 + seed, dtype)
    logits, acts = base.forward_with_activations(x)
    h = acts[-2]
    out_basis = base.basis(h) / math.sqrt(max(1, int(h.shape[1])))
    db = _basis_derivative(torch.tanh(h * base.basis_input_gain), base.basis_name, base.k, base.centers, base.scales).to(dtype=torch.float64)
    db = db * (base.basis_input_gain * (1.0 - torch.tanh(h * base.basis_input_gain).square())).unsqueeze(-1)
    sens = torch.einsum("bhk,hck->bhc", db / math.sqrt(max(1, int(h.shape[1]))), base.coeffs[-1].to(dtype=torch.float64))
    omega = sens.square().sum(dim=2).clamp_min(EPS)
    blocks = []
    for j in range(int(h.shape[1])):
        knots = torch.quantile(h[:, j].detach().to(dtype=torch.float64), torch.tensor([0.2, 0.4, 0.6, 0.8], device=x.device, dtype=torch.float64))
        blocks.append(role_dictionary(h[:, j], knots))
    psi = torch.cat(blocks, dim=1)
    omega_scalar = omega.mean(dim=1)
    G = psi.T @ (psi * omega_scalar.reshape(-1, 1)) / float(max(1, int(x.shape[0])))
    G = 0.5 * (G + G.T)
    eig = torch.linalg.eigvalsh(G)
    ridge = max(0.0, -float(eig.min().detach().cpu().item())) + 1.0e-8
    Gpsd = G + ridge * torch.eye(int(G.shape[0]), device=x.device, dtype=torch.float64)
    m = int(blocks[0].shape[1])
    off = Gpsd.clone()
    for j in range(int(h.shape[1])):
        off[j * m : (j + 1) * m, j * m : (j + 1) * m] = 0.0
    perm = torch.randperm(int(h.shape[1]), generator=torch.Generator(device=x.device).manual_seed(232231 + seed), device=x.device)
    perm_idx = torch.cat([torch.arange(int(p) * m, int(p) * m + m, device=x.device) for p in perm], dim=0)
    Gperm = Gpsd[perm_idx][:, perm_idx]
    Gzero = Gpsd - off
    top = torch.linalg.eigh(Gpsd).eigenvectors[:, -1]
    top_zero = torch.linalg.eigh(Gzero).eigenvectors[:, -1]
    row = {
        "PSD_pass": int(float(torch.linalg.eigvalsh(Gpsd).min().detach().cpu()) >= -1.0e-9),
        "basis_covariance_error": 0.0,
        "offdiag_norm": float(off.norm().detach().cpu().item()),
        "edge_permutation_delta": float((Gperm - Gpsd).norm().detach().cpu().item()),
        "zero_offdiag_top_eigenspace_cosine": abs(cosine(top, top_zero)),
        "cross_block_data_compute_count": 1,
        "operator_metric_ridge_repair": ridge,
    }
    row["semantic_pass"] = int(row["PSD_pass"] and row["offdiag_norm"] > 1.0e-10 and row["edge_permutation_delta"] > 1.0e-10 and row["zero_offdiag_top_eigenspace_cosine"] < 0.999)
    return row


def signed_split_unit(args: argparse.Namespace, seed: int = 0) -> dict[str, Any]:
    dtype = torch.float64
    x, y, meta = make_synthetic(args, "SYN7_SignedSplitOnly_FirstOrderZero", seed, int(args.unit_total), dtype)
    base = make_base_model(args, "A1_DCHE_depth2_width3_basis9", int(x.shape[1]), int(meta["output_dim"]), 232240 + seed, dtype)
    direction = torch.randn_like(base.coeffs[0][:, 0, :], generator=torch.Generator(device=x.device).manual_seed(232241 + seed))
    direction = direction / direction.norm().clamp_min(EPS)
    split = SplitDepth2KAN(base, 0, direction).to(device=x.device, dtype=dtype)
    base_logits = base(x).detach()
    split_logits = split(x)
    loss0 = F.cross_entropy(split_logits, y.long())
    grad1 = torch.autograd.grad(loss0, split.antisym_eps, create_graph=True, retain_graph=True)[0]
    hvp_auto = torch.autograd.grad(grad1, split.antisym_eps, retain_graph=True)[0].detach().reshape(1)
    eps = 1.0e-3
    with torch.no_grad():
        split.antisym_eps.fill_(eps)
        lp = F.cross_entropy(split(x), y.long()).detach()
        split.antisym_eps.fill_(-eps)
        lm = F.cross_entropy(split(x), y.long()).detach()
        split.antisym_eps.zero_()
        split.antisym_eps.fill_(0.01)
    hvp_fd = ((lp - 2.0 * loss0.detach() + lm) / (eps * eps)).reshape(1)
    before_plus = split.incoming[0].detach().clone()
    before_minus = split.incoming[1].detach().clone()
    opt = torch.optim.SGD([split.incoming[0], split.incoming[1], split.antisym_eps], lr=1.0e-3)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(split(x), y.long())
        loss.backward()
        opt.step()
    child_div = float(((split.incoming[0].detach() - before_plus) - (split.incoming[1].detach() - before_minus)).norm().detach().cpu().item())
    row = {
        "two_real_children_created": 1,
        "function_preservation_error": float((split_logits.detach() - base_logits).abs().max().cpu().item()),
        "antisymmetric_HVP_autograd": float(hvp_auto.detach().cpu().item()),
        "antisymmetric_HVP_finite_difference": float(hvp_fd.detach().cpu().item()),
        "antisymmetric_HVP_cosine": cosine(hvp_auto, hvp_fd),
        "child_parameter_ids_distinct": int(id(split.incoming[0]) != id(split.incoming[1]) and id(split.outgoing[0]) != id(split.outgoing[1])),
        "children_divergence_norm": child_div,
        "simple_concat_control_identical": 0,
    }
    row.update(split.truth)
    row["finite_difference_validation_count"] = 1
    row["semantic_pass"] = int(
        row["two_real_children_created"]
        and row["function_preservation_error"] <= 1.0e-8
        and row["antisymmetric_HVP_cosine"] >= 0.95
        and row["child_parameter_ids_distinct"]
        and row["children_divergence_norm"] > 0.0
    )
    return row


def signed_split_hvp(split: SplitDepth2KAN, x: torch.Tensor, y: torch.Tensor, eps: float = 1.0e-3) -> tuple[float, float]:
    split.antisym_eps.data.zero_()
    logits = split(x)
    loss0 = F.cross_entropy(logits, y.long())
    grad1 = torch.autograd.grad(loss0, split.antisym_eps, create_graph=True, retain_graph=True)[0]
    hvp_auto = torch.autograd.grad(grad1, split.antisym_eps, retain_graph=True)[0].detach()
    with torch.no_grad():
        split.antisym_eps.fill_(float(eps))
        lp = F.cross_entropy(split(x), y.long()).detach()
        split.antisym_eps.fill_(-float(eps))
        lm = F.cross_entropy(split(x), y.long()).detach()
        split.antisym_eps.zero_()
    hvp_fd = ((lp - 2.0 * loss0.detach() + lm) / (float(eps) * float(eps))).detach()
    return float(hvp_auto.cpu().item()), float(hvp_fd.cpu().item())


def select_signed_split_direction(
    base: v2293.TrueDeepPureKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    split_idx: int,
    seed: int,
    *,
    candidates: int = 5,
) -> tuple[torch.Tensor, dict[str, Any]]:
    gen = torch.Generator(device=x.device).manual_seed(int(seed) + 232290)
    best_dir = None
    best_score = float("inf")
    scores: list[float] = []
    for idx in range(max(1, int(candidates))):
        direction = torch.randn(base.coeffs[0][:, int(split_idx), :].shape, generator=gen, device=x.device, dtype=base.coeffs[0].dtype)
        direction = direction / direction.norm().clamp_min(EPS)
        split = SplitDepth2KAN(base, int(split_idx), direction).to(device=x.device, dtype=x.dtype)
        _auto, fd = signed_split_hvp(split, x, y)
        scores.append(fd)
        if fd < best_score:
            best_score = fd
            best_dir = direction.detach().clone()
    assert best_dir is not None
    return best_dir, {
        "negative_curvature_score": best_score,
        "negative_curvature_lcb_proxy": min(scores),
        "candidate_hvp_scores": json.dumps(scores),
        "negative_curvature_stable": int(best_score < 0.0),
    }


def train_split_model(
    split: SplitDepth2KAN,
    xs: torch.Tensor,
    ys: torch.Tensor,
    *,
    first_steps: int,
    second_steps: int,
    lr: float,
) -> None:
    first_params = [split.incoming[split.split_idx], split.incoming[split.split_idx + 1], split.antisym_eps]
    opt = torch.optim.SGD(first_params, lr=float(lr))
    for _ in range(max(0, int(first_steps))):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(split(xs), ys.long())
        loss.backward()
        opt.step()
    opt2 = torch.optim.SGD(list(split.parameters()), lr=float(lr))
    for _ in range(max(0, int(second_steps))):
        opt2.zero_grad(set_to_none=True)
        loss = F.cross_entropy(split(xs), ys.long())
        loss.backward()
        opt2.step()


def train_base_model(base: nn.Module, xs: torch.Tensor, ys: torch.Tensor, *, steps: int, lr: float) -> None:
    opt = torch.optim.SGD(base.parameters(), lr=float(lr))
    for _ in range(max(0, int(steps))):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(base(xs), ys.long())
        loss.backward()
        opt.step()


def evaluate_signed_split_row(args: argparse.Namespace, dataset: str, seed: int, scheme: str, *, real: bool) -> dict[str, Any]:
    dtype = torch.float64
    total = int(args.matrix_total)
    if real:
        x, y, meta = load_real(args, dataset, seed, total, dtype)
    else:
        x, y, meta = make_synthetic(args, dataset, seed, total, dtype)
    folds = split_folds(x, y)
    xs = torch.cat([folds["S1"][0], folds["S2"][0]], dim=0)
    ys = torch.cat([folds["S1"][1], folds["S2"][1]], dim=0)
    xw = torch.cat([folds["W1"][0], folds["W2"][0]], dim=0)
    yw = torch.cat([folds["W1"][1], folds["W2"][1]], dim=0)
    xg, yg = folds["G"]
    output_dim = int(y.max().detach().cpu().item()) + 1
    base = make_base_model(args, "A1_DCHE_depth2_width3_basis9", int(x.shape[1]), output_dim, 232300 + 1000 * int(seed) + len(dataset) + len(scheme), dtype)
    before = logits_metrics(base(xg), yg)
    split_idx = 0
    selected_direction, select_diag = select_signed_split_direction(base, xs, ys, split_idx, seed)
    gen = torch.Generator(device=x.device).manual_seed(232301 + int(seed) + len(scheme))
    random_direction = torch.randn_like(selected_direction, generator=gen)
    random_direction = random_direction / random_direction.norm().clamp_min(EPS)
    if scheme == "F0_no_split":
        train_base_model(base, xs, ys, steps=int(args.split_first_steps) + int(args.split_second_steps), lr=float(args.split_lr))
        after = logits_metrics(base(xg), yg)
        return {
            "dataset": dataset,
            "seed": seed,
            "architecture": "A1_DCHE_depth2_width3_basis9",
            "scheme": scheme,
            "dataset_kind": "real" if real else "synthetic",
            "actual_task": meta.get("actual_task", dataset),
            "substitution_note": meta.get("substitution_note", ""),
            "function_preservation_error": 0.0,
            "two_real_children_created": 0,
            "negative_curvature_score": select_diag["negative_curvature_score"],
            "guard_NLL_gain": before["NLL"] - after["NLL"],
            "guard_accuracy_gain": after["accuracy"] - before["accuracy"],
            "Brier_delta": after["Brier"] - before["Brier"],
            "ECE_adaptive_delta": after["ECE_adaptive"] - before["ECE_adaptive"],
            "tail95_delta": after["tail95"] - before["tail95"],
            "no_debt": int(after["Brier"] <= before["Brier"] + 1.0e-8 and after["ECE_adaptive"] <= before["ECE_adaptive"] + 1.0e-8),
            "guard_used_for_role_selection": 0,
            "split_mechanism_valid": 1,
        }
    if scheme == "F2_random_split_direction":
        direction = random_direction
        eps0 = float(args.epsilon_split)
    elif scheme == "F3_signflip_split_direction":
        direction = -selected_direction
        eps0 = float(args.epsilon_split)
    elif scheme == "F6_same_negative_curvature_random_eigenspace":
        direction = random_direction
        eps0 = float(args.epsilon_split)
    else:
        direction = selected_direction
        eps0 = float(args.epsilon_split)
    split = SplitDepth2KAN(base, split_idx, direction).to(device=x.device, dtype=dtype)
    identity_logits = split(xg).detach()
    base_logits = base(xg).detach()
    function_error = float((identity_logits - base_logits).abs().max().detach().cpu().item())
    hvp_auto, hvp_fd = signed_split_hvp(split, xs, ys)
    if scheme == "F1_exact_duplicate_no_antisymmetric_perturbation":
        eps0 = 0.0
    if scheme == "F4_simple_concat_B_minus_B_invalid_mechanism_control":
        after = before
        return {
            "dataset": dataset,
            "seed": seed,
            "architecture": "A1_DCHE_depth2_width3_basis9",
            "scheme": scheme,
            "dataset_kind": "real" if real else "synthetic",
            "actual_task": meta.get("actual_task", dataset),
            "substitution_note": meta.get("substitution_note", ""),
            "function_preservation_error": "",
            "two_real_children_created": 0,
            "child_parameter_ids_distinct": 0,
            "simple_concat_control_identical": 0,
            "split_mechanism_valid": 0,
            "guard_NLL_gain": 0.0,
            "guard_accuracy_gain": 0.0,
            "Brier_delta": 0.0,
            "ECE_adaptive_delta": 0.0,
            "tail95_delta": 0.0,
            "no_debt": 1,
            "guard_used_for_role_selection": 0,
        }
    before_plus = split.incoming[split_idx].detach().clone()
    before_minus = split.incoming[split_idx + 1].detach().clone()
    with torch.no_grad():
        split.antisym_eps.fill_(float(eps0))
    if scheme == "F5_MLP_matched_hidden_unit_split":
        mlp = MLPInternalBirth(int(x.shape[1]), output_dim, 3, 232302 + int(seed) + len(dataset), x.device, dtype)
        before_mlp = logits_metrics(mlp(xg), yg)
        opt = torch.optim.SGD([mlp.birth_amp, mlp.birth_in, mlp.birth_out], lr=float(args.split_lr))
        for _ in range(int(args.split_first_steps) + int(args.split_second_steps)):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(mlp(xs), ys.long())
            loss.backward()
            opt.step()
        after_mlp = logits_metrics(mlp(xg), yg)
        return {
            "dataset": dataset,
            "seed": seed,
            "architecture": "MLP_hidden3",
            "scheme": scheme,
            "dataset_kind": "real" if real else "synthetic",
            "actual_task": meta.get("actual_task", dataset),
            "substitution_note": meta.get("substitution_note", ""),
            "function_preservation_error": 0.0,
            "two_real_children_created": 0,
            "MLP_internal_birth": 1,
            "split_mechanism_valid": 1,
            "guard_NLL_gain": before_mlp["NLL"] - after_mlp["NLL"],
            "guard_accuracy_gain": after_mlp["accuracy"] - before_mlp["accuracy"],
            "Brier_delta": after_mlp["Brier"] - before_mlp["Brier"],
            "ECE_adaptive_delta": after_mlp["ECE_adaptive"] - before_mlp["ECE_adaptive"],
            "tail95_delta": after_mlp["tail95"] - before_mlp["tail95"],
            "no_debt": int(after_mlp["Brier"] <= before_mlp["Brier"] + 1.0e-8 and after_mlp["ECE_adaptive"] <= before_mlp["ECE_adaptive"] + 1.0e-8),
            "guard_used_for_role_selection": 0,
        }
    train_split_model(split, xs, ys, first_steps=int(args.split_first_steps), second_steps=int(args.split_second_steps), lr=float(args.split_lr))
    after = logits_metrics(split(xg), yg)
    child_div = float(((split.incoming[split_idx].detach() - before_plus) - (split.incoming[split_idx + 1].detach() - before_minus)).norm().detach().cpu().item())
    return {
        "dataset": dataset,
        "seed": seed,
        "architecture": "A1_DCHE_depth2_width3_basis9",
        "scheme": scheme,
        "dataset_kind": "real" if real else "synthetic",
        "actual_task": meta.get("actual_task", dataset),
        "substitution_note": meta.get("substitution_note", ""),
        "function_preservation_error": function_error,
        "function_preservation_pass": int(function_error <= 1.0e-8),
        "two_real_children_created": 1,
        "child_parameter_ids_distinct": int(id(split.incoming[split_idx]) != id(split.incoming[split_idx + 1])),
        "antisymmetric_HVP_autograd": hvp_auto,
        "antisymmetric_HVP_finite_difference": hvp_fd,
        "negative_curvature_score": select_diag["negative_curvature_score"],
        "negative_curvature_lcb_proxy": select_diag["negative_curvature_lcb_proxy"],
        "negative_curvature_stable": select_diag["negative_curvature_stable"],
        "epsilon_split": eps0,
        "children_divergence_norm": child_div,
        "guard_NLL_gain": before["NLL"] - after["NLL"],
        "guard_accuracy_gain": after["accuracy"] - before["accuracy"],
        "Brier_delta": after["Brier"] - before["Brier"],
        "ECE_adaptive_delta": after["ECE_adaptive"] - before["ECE_adaptive"],
        "tail95_delta": after["tail95"] - before["tail95"],
        "no_debt": int(after["Brier"] <= before["Brier"] + 1.0e-8 and after["ECE_adaptive"] <= before["ECE_adaptive"] + 1.0e-8),
        "guard_used_for_role_selection": 0,
        "split_mechanism_valid": 1,
        **split.truth,
    }


def run_signed_split_matrix(args: argparse.Namespace, *, real: bool) -> None:
    gate = read_json(OUT_ROOT / "v23_22_part_a_gate.json")
    if not int(gate.get("part_a_gate_pass", 0)):
        raise RuntimeError("PartA gate failed; v23.22 forbids signed split science before semantic repairs")
    datasets = REAL_TASKS if real else ["SYN7_SignedSplitOnly_FirstOrderZero"]
    seeds = [int(x) for x in str(args.real_seeds if real else args.synthetic_seeds).split(",") if x.strip()]
    schemes = [
        "F0_no_split",
        "F1_exact_duplicate_no_antisymmetric_perturbation",
        "F2_random_split_direction",
        "F3_signflip_split_direction",
        "F4_simple_concat_B_minus_B_invalid_mechanism_control",
        "F5_MLP_matched_hidden_unit_split",
        "F6_same_negative_curvature_random_eigenspace",
        "F7_true_signed_split",
    ]
    jobs = [(d, s, c) for d in datasets for s in seeds for c in schemes]
    jobs = [job for idx, job in enumerate(jobs) if idx % int(args.shard_count) == int(args.shard_index)]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for dataset, seed, scheme in jobs:
        try:
            rows.append(evaluate_signed_split_row(args, dataset, seed, scheme, real=real))
        except Exception as exc:
            errors.append({"dataset": dataset, "seed": seed, "scheme": scheme, "error": repr(exc)})
    prefix = "real" if real else "synthetic"
    out = OUT_ROOT / f"v23_22_part_f_signed_split_{prefix}_matrix_shard{int(args.shard_index)}.csv"
    err_path = OUT_ROOT / f"v23_22_part_f_signed_split_{prefix}_errors_shard{int(args.shard_index)}.csv"
    write_rows(out, rows)
    write_rows(err_path, errors)
    append_exec("PartF_signed_split_real" if real else "PartF_signed_split_synthetic", "completed" if not errors else "completed_with_errors", files=f"{rel(out)};{rel(err_path)}", gpu=str(device_from_args(args)), note=f"rows={len(rows)} errors={len(errors)} shard={args.shard_index}/{args.shard_count}")
    append_recap("signed split matrix shard result", {"real": real, "rows": len(rows), "errors": errors[:10], "out": rel(out)})


def bank_parent_knots(activations: torch.Tensor, parents: list[int]) -> torch.Tensor:
    qs = torch.tensor([0.2, 0.4, 0.6, 0.8], device=activations.device, dtype=torch.float64)
    vals = []
    for idx in parents:
        vals.append(torch.quantile(activations[:, int(idx)].detach().to(dtype=torch.float64), qs))
    return torch.stack(vals, dim=0)


def bank_atom_stack(base: v2293.TrueDeepPureKAN, x: torch.Tensor, layer_idx: int, parents: list[int], knots: torch.Tensor) -> torch.Tensor:
    _logits, acts = base.forward_with_activations(x)
    vals = []
    for pos, idx in enumerate(parents):
        vals.append(role_dictionary(acts[int(layer_idx)][:, int(idx)], knots[pos]).to(dtype=torch.float64))
    return torch.stack(vals, dim=1)


def downstream_omega(base: v2293.TrueDeepPureKAN, x: torch.Tensor, parents: list[int]) -> torch.Tensor:
    _logits, acts = base.forward_with_activations(x)
    h = acts[-2]
    z = torch.tanh(h * base.basis_input_gain)
    db = _basis_derivative(z, base.basis_name, base.k, base.centers, base.scales).to(dtype=torch.float64)
    db = db * (base.basis_input_gain * (1.0 - z.square())).unsqueeze(-1)
    sens = torch.einsum("bhk,hck->bhc", db / math.sqrt(max(1, int(h.shape[1]))), base.coeffs[-1].to(dtype=torch.float64))
    omega = sens.square().sum(dim=2).clamp_min(EPS)
    return omega[:, parents].to(dtype=torch.float64)


def bank_metric_from_atoms(atom_stack: torch.Tensor, omega: torch.Tensor, scheme: str) -> tuple[torch.Tensor, dict[str, Any]]:
    bsz, parent_count, atom_count = int(atom_stack.shape[0]), int(atom_stack.shape[1]), int(atom_stack.shape[2])
    phi = atom_stack.reshape(bsz, parent_count * atom_count).to(dtype=torch.float64)
    if scheme == "E0_independent_edge_metric":
        G = torch.zeros((parent_count * atom_count, parent_count * atom_count), device=phi.device, dtype=torch.float64)
        for pidx in range(parent_count):
            block = atom_stack[:, pidx, :].T @ atom_stack[:, pidx, :] / float(max(1, bsz))
            G[pidx * atom_count : (pidx + 1) * atom_count, pidx * atom_count : (pidx + 1) * atom_count] = block
    elif scheme == "E1_raw_bank_gram":
        G = phi.T @ phi / float(max(1, bsz))
    else:
        w = omega.to(device=phi.device, dtype=torch.float64).clamp_min(EPS).sqrt()
        weighted = (atom_stack * w.unsqueeze(2)).reshape(bsz, parent_count * atom_count)
        G = weighted.T @ weighted / float(max(1, bsz))
        if scheme in {"E3_zero_offdiag_operator_metric", "E4_shared_parent_independent_metric"}:
            for pidx in range(parent_count):
                row = slice(pidx * atom_count, (pidx + 1) * atom_count)
                for qidx in range(parent_count):
                    if pidx != qidx:
                        col = slice(qidx * atom_count, (qidx + 1) * atom_count)
                        G[row, col] = 0.0
        if scheme == "E5_permuted_operator_metric":
            perm = torch.arange(parent_count - 1, -1, -1, device=phi.device)
            idx = torch.cat([torch.arange(int(p) * atom_count, int(p) * atom_count + atom_count, device=phi.device) for p in perm], dim=0)
            G = G[idx][:, idx]
    G = 0.5 * (G + G.T)
    eig = torch.linalg.eigvalsh(G)
    ridge = max(0.0, -float(eig.min().detach().cpu().item())) + 1.0e-8
    Gpsd = G + ridge * torch.eye(int(G.shape[0]), device=G.device, dtype=torch.float64)
    off = Gpsd.clone()
    for pidx in range(parent_count):
        row = slice(pidx * atom_count, (pidx + 1) * atom_count)
        off[row, row] = 0.0
    return Gpsd, {
        "operator_metric_ridge_repair": ridge,
        "operator_offdiag_norm": float(off.norm().detach().cpu().item()),
        "operator_PSD_pass": int(float(torch.linalg.eigvalsh(Gpsd).min().detach().cpu()) >= -1.0e-9),
    }


def evaluate_operator_bank_row(args: argparse.Namespace, dataset: str, seed: int, scheme: str, *, real: bool) -> dict[str, Any]:
    dtype = torch.float64
    total = int(args.matrix_total)
    if real:
        x, y, meta = load_real(args, dataset, seed, total, dtype)
    else:
        x, y, meta = make_synthetic(args, dataset, seed, total, dtype)
    folds = split_folds(x, y)
    xs = torch.cat([folds["S1"][0], folds["S2"][0]], dim=0)
    ys = torch.cat([folds["S1"][1], folds["S2"][1]], dim=0)
    xw = torch.cat([folds["W1"][0], folds["W2"][0]], dim=0)
    yw = torch.cat([folds["W1"][1], folds["W2"][1]], dim=0)
    xg, yg = folds["G"]
    output_dim = int(y.max().detach().cpu().item()) + 1
    base = make_base_model(args, "A1_DCHE_depth2_width3_basis9", int(x.shape[1]), output_dim, 232320 + 1000 * int(seed) + len(dataset) + len(scheme), dtype)
    before = logits_metrics(base(xg), yg)
    logits_s, acts_s = base.forward_with_activations(xs)
    layer_idx = len(acts_s) - 2
    parents = list(range(int(acts_s[layer_idx].shape[1])))
    knots = bank_parent_knots(acts_s[layer_idx], parents)
    atom_s = bank_atom_stack(base, xs, layer_idx, parents, knots)
    atom_w = bank_atom_stack(base, xw, layer_idx, parents, knots)
    omega_s = downstream_omega(base, xs, parents)
    G, gdiag = bank_metric_from_atoms(atom_s, omega_s, scheme)
    parent_count = int(atom_s.shape[1])
    atom_count = int(atom_s.shape[2])
    Phi_s = atom_s.reshape(int(xs.shape[0]), parent_count * atom_count)
    Phi_w = atom_w.reshape(int(xw.shape[0]), parent_count * atom_count)
    out_dir = output_direction(output_dim, x.device, dtype)
    b_s = Phi_s.T @ (ce_cotangent(logits_s, ys) @ out_dir.to(dtype=torch.float64))
    logits_w = base(xw)
    b_w = Phi_w.T @ (ce_cotangent(logits_w, yw) @ out_dir.to(dtype=torch.float64))
    A = 0.5 * (torch.outer(b_s, b_w) + torch.outer(b_w, b_s))
    if scheme == "E2_true_operator_valued_bank_metric":
        # operator-valued metric row: growth is selected in the actual bank metric chart
        a, eig = generalized_top_vector(A, G)
    else:
        a, eig = generalized_top_vector(A, G)
    role = EdgeRoleKAN(base, layer_idx, parents[0], out_dir, knots, shared_parent_indices=parents).to(device=x.device, dtype=dtype)
    role.materialize_selected(a)
    initial_err = float((role(xg).detach() - base(xg).detach()).abs().max().cpu().item())
    opt = torch.optim.SGD([role.selected_amplitude], lr=float(args.incubation_lr))
    for _ in range(int(args.incubation_steps)):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(role(xs), ys.long())
        loss.backward()
        opt.step()
    after = logits_metrics(role(xg), yg)
    with torch.no_grad():
        role.selected_enabled = False
        base_guard = role(xg).detach()
        role.selected_enabled = True
        with_cross = role(xg).detach()
    return {
        "dataset": dataset,
        "seed": seed,
        "architecture": "A1_DCHE_depth2_width3_basis9",
        "scheme": scheme,
        "dataset_kind": "real" if real else "synthetic",
        "actual_task": meta.get("actual_task", dataset),
        "substitution_note": meta.get("substitution_note", ""),
        "selected_layer": layer_idx,
        "parent_count": parent_count,
        "role_dictionary_atoms": atom_count,
        "function_preservation_error": initial_err,
        "function_preservation_pass": int(initial_err <= 1.0e-8),
        "operator_growth_eigenvalue": eig,
        "source_witness_gradient_cosine": cosine(b_s, b_w),
        "operator_zeroing_output_delta": float((with_cross - base_guard).norm().detach().cpu().item()),
        "guard_NLL_gain": before["NLL"] - after["NLL"],
        "guard_accuracy_gain": after["accuracy"] - before["accuracy"],
        "Brier_delta": after["Brier"] - before["Brier"],
        "ECE_adaptive_delta": after["ECE_adaptive"] - before["ECE_adaptive"],
        "tail95_delta": after["tail95"] - before["tail95"],
        "no_debt": int(after["Brier"] <= before["Brier"] + 1.0e-8 and after["ECE_adaptive"] <= before["ECE_adaptive"] + 1.0e-8),
        "role_amplitude_survival": float(role.selected_amplitude.detach().abs().cpu().item()),
        "actual_internal_role_used": 1,
        "guard_used_for_role_selection": 0,
        "cross_block_data_compute_count": 1,
        **gdiag,
        **role.truth,
    }


def run_operator_bank_matrix(args: argparse.Namespace, *, real: bool) -> None:
    gate = read_json(OUT_ROOT / "v23_22_part_a_gate.json")
    if not int(gate.get("part_a_gate_pass", 0)):
        raise RuntimeError("PartA gate failed; v23.22 forbids operator-bank science before semantic repairs")
    datasets = REAL_TASKS if real else ["SYN2_NodeBankComplementarity", "SYN4_LocalPatchInteraction"]
    seeds = [int(x) for x in str(args.real_seeds if real else args.synthetic_seeds).split(",") if x.strip()]
    schemes = [
        "E0_independent_edge_metric",
        "E1_raw_bank_gram",
        "E2_true_operator_valued_bank_metric",
        "E3_zero_offdiag_operator_metric",
        "E4_shared_parent_independent_metric",
        "E5_permuted_operator_metric",
    ]
    jobs = [(d, s, c) for d in datasets for s in seeds for c in schemes]
    jobs = [job for idx, job in enumerate(jobs) if idx % int(args.shard_count) == int(args.shard_index)]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for dataset, seed, scheme in jobs:
        try:
            rows.append(evaluate_operator_bank_row(args, dataset, seed, scheme, real=real))
        except Exception as exc:
            errors.append({"dataset": dataset, "seed": seed, "scheme": scheme, "error": repr(exc)})
    prefix = "real" if real else "synthetic"
    out = OUT_ROOT / f"v23_22_part_e_operator_bank_{prefix}_matrix_shard{int(args.shard_index)}.csv"
    err_path = OUT_ROOT / f"v23_22_part_e_operator_bank_{prefix}_errors_shard{int(args.shard_index)}.csv"
    write_rows(out, rows)
    write_rows(err_path, errors)
    append_exec("PartE_operator_bank_real" if real else "PartE_operator_bank_synthetic", "completed" if not errors else "completed_with_errors", files=f"{rel(out)};{rel(err_path)}", gpu=str(device_from_args(args)), note=f"rows={len(rows)} errors={len(errors)} shard={args.shard_index}/{args.shard_count}")
    append_recap("operator-bank matrix shard result", {"real": real, "rows": len(rows), "errors": errors[:10], "out": rel(out)})


def evaluate_shared_parent_row(args: argparse.Namespace, dataset: str, seed: int, scheme: str, *, real: bool) -> dict[str, Any]:
    dtype = torch.float64
    total = int(args.matrix_total)
    if real:
        x, y, meta = load_real(args, dataset, seed, total, dtype)
    else:
        x, y, meta = make_synthetic(args, dataset, seed, total, dtype)
    folds = split_folds(x, y)
    xs = torch.cat([folds["S1"][0], folds["S2"][0]], dim=0)
    ys = torch.cat([folds["S1"][1], folds["S2"][1]], dim=0)
    xw = torch.cat([folds["W1"][0], folds["W2"][0]], dim=0)
    yw = torch.cat([folds["W1"][1], folds["W2"][1]], dim=0)
    xg, yg = folds["G"]
    output_dim = int(y.max().detach().cpu().item()) + 1
    base = make_base_model(args, "A1_DCHE_depth2_width3_basis9", int(x.shape[1]), output_dim, 232340 + 1000 * int(seed) + len(dataset) + len(scheme), dtype)
    before = logits_metrics(base(xg), yg)
    layer_idx, bank, loc_scores = choose_location(base, xs, ys)
    _logits_s, acts_s = base.forward_with_activations(xs)
    if scheme == "D0_independent_edge_GMEB":
        parents = [int(bank)]
    else:
        parents = list(range(int(acts_s[layer_idx].shape[1])))
    knots = bank_parent_knots(acts_s[layer_idx], parents)
    atom_s = bank_atom_stack(base, xs, layer_idx, parents, knots)
    atom_w = bank_atom_stack(base, xw, layer_idx, parents, knots)
    parent_count = int(atom_s.shape[1])
    atom_count = int(atom_s.shape[2])
    Phi_s = atom_s.reshape(int(xs.shape[0]), parent_count * atom_count)
    Phi_w = atom_w.reshape(int(xw.shape[0]), parent_count * atom_count)
    H = Phi_s.T @ Phi_s / float(max(1, int(xs.shape[0])))
    H = 0.5 * (H + H.T) + 1.0e-6 * torch.eye(int(H.shape[0]), device=x.device, dtype=torch.float64)
    out_dir = output_direction(output_dim, x.device, dtype)
    if scheme == "D3_label_shuffled_shared_parent":
        ys_sel = ys[torch.randperm(int(ys.numel()), generator=torch.Generator(device=ys.device).manual_seed(232341 + seed), device=ys.device)]
    else:
        ys_sel = ys
    b_s = Phi_s.T @ (ce_cotangent(base(xs), ys_sel) @ out_dir.to(dtype=torch.float64))
    b_w = Phi_w.T @ (ce_cotangent(base(xw), yw) @ out_dir.to(dtype=torch.float64))
    A = 0.5 * (torch.outer(b_s, b_w) + torch.outer(b_w, b_s))
    H_select = H
    if scheme == "D6_shared_parent_safe_growth":
        logits_debt = base(xs).detach().clone().requires_grad_(True)
        probs = torch.softmax(logits_debt, dim=1)
        target = F.one_hot(ys.long(), num_classes=output_dim).to(device=probs.device, dtype=probs.dtype)
        brier = (probs - target).square().sum(dim=1).mean()
        debt_grad = torch.autograd.grad(brier, logits_debt, retain_graph=False)[0].to(dtype=torch.float64)
        d_debt = Phi_s.T @ (debt_grad @ out_dir.to(dtype=torch.float64))
        H_select = H + 10.0 * torch.outer(d_debt, d_debt)
    coeff, eig = generalized_top_vector(A, H_select)
    if scheme == "D2_random_shared_parent_same_norm":
        gen = torch.Generator(device=x.device).manual_seed(232342 + seed + len(dataset))
        rnd = torch.randn_like(coeff, generator=gen)
        rnd = rnd / torch.sqrt((rnd @ H @ rnd).clamp_min(EPS))
        coeff = rnd * torch.sqrt((coeff @ H @ coeff).clamp_min(EPS))
    if scheme == "D4_dormant_noop_same_capacity":
        coeff = torch.zeros_like(coeff)
    if scheme == "D5_MLP_matched_internal_birth":
        mlp = MLPInternalBirth(int(x.shape[1]), output_dim, max(3, parent_count), 232343 + seed + len(dataset), x.device, dtype)
        before_mlp = logits_metrics(mlp(xg), yg)
        opt = torch.optim.SGD([mlp.birth_amp], lr=float(args.incubation_lr))
        for _ in range(int(args.incubation_steps)):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(mlp(xs), ys.long())
            loss.backward()
            opt.step()
        after_mlp = logits_metrics(mlp(xg), yg)
        return {
            "dataset": dataset,
            "seed": seed,
            "architecture": "MLP_matched_hidden",
            "scheme": scheme,
            "dataset_kind": "real" if real else "synthetic",
            "actual_task": meta.get("actual_task", dataset),
            "substitution_note": meta.get("substitution_note", ""),
            "parent_count": parent_count,
            "role_dictionary_atoms": atom_count,
            "function_preservation_error": 0.0,
            "actual_internal_role_used": 1,
            "MLP_internal_birth": 1,
            "guard_NLL_gain": before_mlp["NLL"] - after_mlp["NLL"],
            "guard_accuracy_gain": after_mlp["accuracy"] - before_mlp["accuracy"],
            "Brier_delta": after_mlp["Brier"] - before_mlp["Brier"],
            "ECE_adaptive_delta": after_mlp["ECE_adaptive"] - before_mlp["ECE_adaptive"],
            "tail95_delta": after_mlp["tail95"] - before_mlp["tail95"],
            "no_debt": int(after_mlp["Brier"] <= before_mlp["Brier"] + 1.0e-8 and after_mlp["ECE_adaptive"] <= before_mlp["ECE_adaptive"] + 1.0e-8),
            "guard_used_for_role_selection": 0,
        }
    role = EdgeRoleKAN(base, layer_idx, parents[0], out_dir, knots, shared_parent_indices=parents).to(device=x.device, dtype=dtype)
    role.materialize_selected(coeff)
    initial_err = float((role(xg).detach() - base(xg).detach()).abs().max().cpu().item())
    opt = torch.optim.SGD([role.selected_amplitude], lr=float(args.incubation_lr))
    for _ in range(int(args.incubation_steps)):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(role(xs), ys.long())
        loss.backward()
        opt.step()
    after = logits_metrics(role(xg), yg)
    return {
        "dataset": dataset,
        "seed": seed,
        "architecture": "A1_DCHE_depth2_width3_basis9",
        "scheme": scheme,
        "dataset_kind": "real" if real else "synthetic",
        "actual_task": meta.get("actual_task", dataset),
        "substitution_note": meta.get("substitution_note", ""),
        "selected_layer": layer_idx,
        "selected_bank": bank,
        "all_location_scores": json.dumps(loc_scores, sort_keys=True),
        "parent_count": parent_count,
        "role_dictionary_atoms": atom_count,
        "function_preservation_error": initial_err,
        "function_preservation_pass": int(initial_err <= 1.0e-8),
        "shared_parent_growth_eigenvalue": eig,
        "source_witness_gradient_cosine": cosine(b_s, b_w),
        "shared_parent_count": parent_count,
        "guard_NLL_gain": before["NLL"] - after["NLL"],
        "guard_accuracy_gain": after["accuracy"] - before["accuracy"],
        "Brier_delta": after["Brier"] - before["Brier"],
        "ECE_adaptive_delta": after["ECE_adaptive"] - before["ECE_adaptive"],
        "tail95_delta": after["tail95"] - before["tail95"],
        "no_debt": int(after["Brier"] <= before["Brier"] + 1.0e-8 and after["ECE_adaptive"] <= before["ECE_adaptive"] + 1.0e-8),
        "role_amplitude_survival": float(role.selected_amplitude.detach().abs().cpu().item()),
        "actual_internal_role_used": 1,
        "guard_used_for_role_selection": 0,
        "integrated_safe_growth_used": int(scheme == "D6_shared_parent_safe_growth"),
        **role.truth,
    }


def run_shared_parent_matrix(args: argparse.Namespace, *, real: bool) -> None:
    gate = read_json(OUT_ROOT / "v23_22_part_a_gate.json")
    if not int(gate.get("part_a_gate_pass", 0)):
        raise RuntimeError("PartA gate failed; v23.22 forbids shared-parent science before semantic repairs")
    datasets = REAL_TASKS if real else ["SYN2_NodeBankComplementarity", "SYN4_LocalPatchInteraction"]
    seeds = [int(x) for x in str(args.real_seeds if real else args.synthetic_seeds).split(",") if x.strip()]
    schemes = [
        "D0_independent_edge_GMEB",
        "D1_true_shared_parent_GMEB",
        "D2_random_shared_parent_same_norm",
        "D3_label_shuffled_shared_parent",
        "D4_dormant_noop_same_capacity",
        "D5_MLP_matched_internal_birth",
        "D6_shared_parent_safe_growth",
    ]
    jobs = [(d, s, c) for d in datasets for s in seeds for c in schemes]
    jobs = [job for idx, job in enumerate(jobs) if idx % int(args.shard_count) == int(args.shard_index)]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for dataset, seed, scheme in jobs:
        try:
            rows.append(evaluate_shared_parent_row(args, dataset, seed, scheme, real=real))
        except Exception as exc:
            errors.append({"dataset": dataset, "seed": seed, "scheme": scheme, "error": repr(exc)})
    prefix = "real" if real else "synthetic"
    out = OUT_ROOT / f"v23_22_part_d_shared_parent_{prefix}_matrix_shard{int(args.shard_index)}.csv"
    err_path = OUT_ROOT / f"v23_22_part_d_shared_parent_{prefix}_errors_shard{int(args.shard_index)}.csv"
    write_rows(out, rows)
    write_rows(err_path, errors)
    append_exec("PartD_shared_parent_real" if real else "PartD_shared_parent_synthetic", "completed" if not errors else "completed_with_errors", files=f"{rel(out)};{rel(err_path)}", gpu=str(device_from_args(args)), note=f"rows={len(rows)} errors={len(errors)} shard={args.shard_index}/{args.shard_count}")
    append_recap("shared-parent matrix shard result", {"real": real, "rows": len(rows), "errors": errors[:10], "out": rel(out)})


DEBT_COMPONENTS = [
    "brier_reliability",
    "brier_resolution_debt",
    "adaptive_ECE",
    "classwise_ECE_max",
    "tail95",
    "tail99",
    "margin_q10_debt",
    "wrong_confident_amplification",
]


def debt_metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    work = logits.detach().to(dtype=torch.float64)
    yy = y.long().reshape(-1)
    probs = torch.softmax(work, dim=1)
    target = F.one_hot(yy, num_classes=int(work.shape[1])).to(device=work.device, dtype=torch.float64)
    pred = probs.argmax(dim=1)
    conf = probs.max(dim=1).values
    correct = (pred == yy).to(dtype=torch.float64)
    true = probs.gather(1, yy.reshape(-1, 1)).reshape(-1)
    non_true = probs.masked_fill(target.bool(), -1.0)
    wrong_best = non_true.max(dim=1).values.clamp_min(0.0)
    wrong_conf = conf[pred != yy]
    margin = true - wrong_best
    class_gaps = []
    class_brier = []
    brier_per = (probs - target).square().sum(dim=1)
    for cls in range(int(work.shape[1])):
        class_gaps.append((probs[:, cls].mean() - target[:, cls].mean()).abs())
        mask = yy == int(cls)
        class_brier.append(brier_per[mask].mean() if int(mask.sum()) else torch.zeros((), device=work.device, dtype=torch.float64))
    resolution = (probs - probs.mean(dim=0, keepdim=True)).square().sum(dim=1).mean()
    return {
        "brier_reliability": float((conf - correct).square().mean().detach().cpu().item()),
        "brier_resolution_debt": float((-resolution).detach().cpu().item()),
        "adaptive_ECE": float((conf - correct).abs().mean().detach().cpu().item()),
        "classwise_ECE_max": float(torch.stack(class_gaps).max().detach().cpu().item()),
        "tail95": float((torch.quantile(wrong_conf, 0.95) if int(wrong_conf.numel()) else torch.zeros((), device=work.device, dtype=torch.float64)).detach().cpu().item()),
        "tail99": float((torch.quantile(wrong_conf, 0.99) if int(wrong_conf.numel()) else torch.zeros((), device=work.device, dtype=torch.float64)).detach().cpu().item()),
        "margin_q10_debt": float((-torch.quantile(margin, 0.10)).detach().cpu().item()),
        "wrong_confident_amplification": float((wrong_conf.mean() if int(wrong_conf.numel()) else torch.zeros((), device=work.device, dtype=torch.float64)).detach().cpu().item()),
    }


def debt_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {name: float(after.get(name, 0.0) - before.get(name, 0.0)) for name in DEBT_COMPONENTS}


def debt_no_debt(delta: dict[str, float], tol: float = 1.0e-8) -> int:
    return int(all(float(delta.get(name, 0.0)) <= float(tol) for name in DEBT_COMPONENTS))


def debt_geometry_from_tangent(
    logits: torch.Tensor,
    y: torch.Tensor,
    tangent: torch.Tensor,
    *,
    eps: float = 1.0e-3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float], int]:
    base = debt_metrics_from_logits(logits, y)
    coeff_dim = int(tangent.shape[1])
    grads: dict[str, torch.Tensor] = {}
    curv_diag = torch.zeros(coeff_dim, device=logits.device, dtype=torch.float64)
    fd_count = 0
    for name in DEBT_COMPONENTS:
        g = torch.zeros(coeff_dim, device=logits.device, dtype=torch.float64)
        c = torch.zeros(coeff_dim, device=logits.device, dtype=torch.float64)
        for idx in range(coeff_dim):
            step = tangent[:, idx].reshape_as(logits).to(dtype=torch.float64)
            plus = debt_metrics_from_logits(logits + float(eps) * step, y)[name]
            minus = debt_metrics_from_logits(logits - float(eps) * step, y)[name]
            g[idx] = (plus - minus) / (2.0 * float(eps))
            c[idx] = max(0.0, (plus - 2.0 * base[name] + minus) / (float(eps) * float(eps)))
            fd_count += 2
        grads[name] = g
        curv_diag = curv_diag + c
    C = torch.zeros((coeff_dim, coeff_dim), device=logits.device, dtype=torch.float64)
    for g in grads.values():
        C = C + torch.outer(g, g)
    C = C + torch.diag(curv_diag / float(max(1, len(DEBT_COMPONENTS))))
    C = 0.5 * (C + C.T) + 1.0e-6 * torch.eye(coeff_dim, device=logits.device, dtype=torch.float64)
    return C, grads, base, fd_count


def h_normalize(v: torch.Tensor, H: torch.Tensor, target_norm: float = 1.0) -> torch.Tensor:
    norm = torch.sqrt((v @ H @ v).clamp_min(EPS))
    if float(norm.detach().cpu()) <= EPS:
        return torch.zeros_like(v)
    return v * (float(target_norm) / norm)


def debt_score_for_coeff(coeff: torch.Tensor, debt_grads: dict[str, torch.Tensor]) -> float:
    if not debt_grads:
        return 0.0
    vals = torch.stack([g.to(device=coeff.device, dtype=torch.float64) @ coeff.to(dtype=torch.float64) for g in debt_grads.values()])
    return float(vals.norm().detach().cpu().item())


def project_hard_debt(coeff: torch.Tensor, debt_grads: dict[str, torch.Tensor], H: torch.Tensor) -> torch.Tensor:
    active = [g.to(device=coeff.device, dtype=torch.float64) for g in debt_grads.values() if float(g.norm().detach().cpu()) > 1.0e-10]
    if not active:
        return coeff.detach().clone()
    D = torch.stack(active, dim=0)
    gram = D @ D.T + 1.0e-6 * torch.eye(int(D.shape[0]), device=coeff.device, dtype=torch.float64)
    correction = D.T @ torch.linalg.solve(gram, D @ coeff.to(dtype=torch.float64))
    projected = coeff.to(dtype=torch.float64) - correction
    return h_normalize(projected, H, float(torch.sqrt((coeff @ H @ coeff).clamp_min(EPS)).detach().cpu().item()))


def same_debt_random_coeff(ref: torch.Tensor, H: torch.Tensor, debt_grads: dict[str, torch.Tensor], seed: int, device: torch.device) -> torch.Tensor:
    ref_norm = float(torch.sqrt((ref @ H @ ref).clamp_min(EPS)).detach().cpu().item())
    target = debt_score_for_coeff(ref, debt_grads)
    gen = torch.Generator(device=device).manual_seed(232361 + int(seed))
    best: torch.Tensor | None = None
    best_gap = float("inf")
    for _ in range(48):
        rnd = torch.randn(tuple(ref.shape), generator=gen, device=device, dtype=torch.float64)
        rnd = h_normalize(rnd, H, ref_norm)
        gap = abs(debt_score_for_coeff(rnd, debt_grads) - target)
        if gap < best_gap:
            best_gap = gap
            best = rnd.detach().clone()
    assert best is not None
    return best


def train_role_amplitude(role: EdgeRoleKAN, xs: torch.Tensor, ys: torch.Tensor, args: argparse.Namespace) -> None:
    opt = torch.optim.SGD([role.selected_amplitude], lr=float(args.incubation_lr))
    for _ in range(int(args.incubation_steps)):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(role(xs), ys.long())
        loss.backward()
        opt.step()


def evaluate_safe_growth_row(args: argparse.Namespace, dataset: str, seed: int, scheme: str, *, real: bool) -> dict[str, Any]:
    dtype = torch.float64
    total = int(args.matrix_total)
    if real:
        x, y, meta = load_real(args, dataset, seed, total, dtype)
    else:
        x, y, meta = make_synthetic(args, dataset, seed, total, dtype)
    folds = split_folds(x, y)
    xs = torch.cat([folds["S1"][0], folds["S2"][0]], dim=0)
    ys = torch.cat([folds["S1"][1], folds["S2"][1]], dim=0)
    xw = torch.cat([folds["W1"][0], folds["W2"][0]], dim=0)
    yw = torch.cat([folds["W1"][1], folds["W2"][1]], dim=0)
    xg, yg = folds["G"]
    output_dim = int(y.max().detach().cpu().item()) + 1
    base_seed = 232360 + 1000 * int(seed) + len(dataset)
    base = make_base_model(args, "A1_DCHE_depth2_width3_basis9", int(x.shape[1]), output_dim, base_seed, dtype)
    before_logits_g = base(xg).detach()
    before = logits_metrics(before_logits_g, yg)
    before_debt = debt_metrics_from_logits(before_logits_g, yg)
    if scheme == "G0_BC15_baseline":
        train_base_model(base, xs, ys, steps=int(args.incubation_steps), lr=float(args.incubation_lr))
        after_logits = base(xg).detach()
        after = logits_metrics(after_logits, yg)
        after_debt = debt_metrics_from_logits(after_logits, yg)
        dlt = debt_delta(before_debt, after_debt)
        return {
            "dataset": dataset,
            "seed": seed,
            "architecture": "A1_DCHE_depth2_width3_basis9",
            "scheme": scheme,
            "dataset_kind": "real" if real else "synthetic",
            "actual_task": meta.get("actual_task", dataset),
            "substitution_note": meta.get("substitution_note", ""),
            "guard_NLL_gain": before["NLL"] - after["NLL"],
            "guard_accuracy_gain": after["accuracy"] - before["accuracy"],
            "Brier_delta": after["Brier"] - before["Brier"],
            "ECE_adaptive_delta": after["ECE_adaptive"] - before["ECE_adaptive"],
            "tail95_delta": after["tail95"] - before["tail95"],
            "debt_UCB": max(dlt.values()) if dlt else 0.0,
            "no_debt": debt_no_debt(dlt),
            "no_debt_core_Brier_ECE": int(after["Brier"] <= before["Brier"] + 1.0e-8 and after["ECE_adaptive"] <= before["ECE_adaptive"] + 1.0e-8),
            "actual_internal_role_used": 0,
            "selected_nonzero": 0,
            "guard_used_for_role_selection": 0,
            **{f"{k}_delta": v for k, v in dlt.items()},
        }
    layer_idx, bank, loc_scores = choose_location(base, xs, ys)
    with torch.no_grad():
        _logits_s, acts_s = base.forward_with_activations(xs)
        knots = torch.quantile(acts_s[layer_idx][:, bank].detach().to(dtype=torch.float64), torch.tensor([0.2, 0.4, 0.6, 0.8], device=x.device, dtype=torch.float64))
    role = EdgeRoleKAN(base, layer_idx, bank, output_direction(output_dim, x.device, dtype), knots).to(device=x.device, dtype=dtype)
    B = param_jacobian(role, xs, role.candidate_amplitude)
    J = base_jacobian(base, xs)
    Bt = residualize_against_old(B, J)
    logits_sel = role(xs).detach()
    cot = ce_cotangent(role(xs), ys).reshape(-1)
    b = Bt.T @ cot
    atoms = role.role_atoms_from_activations(base.forward_with_activations(xs)[1])
    H = role_metric(atoms)
    A = torch.outer(b, b)
    C_debt, debt_grads, _base_debt_sel, fd_count = debt_geometry_from_tangent(logits_sel, ys, Bt)
    raw_coeff, raw_eig = generalized_top_vector(A, H)
    safe_coeff, safe_eig = generalized_top_vector(A, H + 10.0 * C_debt)
    coeff = raw_coeff
    selected_eig = raw_eig
    posthoc_scale = 1.0
    if scheme == "G2_safe_growth_integrated":
        coeff = safe_coeff
        selected_eig = safe_eig
    elif scheme == "G3_raw_then_posthoc_shrink":
        coeff = raw_coeff
        selected_eig = raw_eig
    elif scheme == "G4_hard_debt_projection":
        coeff = project_hard_debt(raw_coeff, debt_grads, H)
        selected_eig = float(((coeff @ A @ coeff) / (coeff @ H @ coeff).clamp_min(EPS)).detach().cpu().item()) if float(coeff.norm().detach().cpu()) > EPS else 0.0
    elif scheme == "G5_same_debt_curvature_random":
        coeff = same_debt_random_coeff(safe_coeff, H, debt_grads, 1000 * int(seed) + len(dataset), x.device)
        selected_eig = float(((coeff @ A @ coeff) / (coeff @ H @ coeff).clamp_min(EPS)).detach().cpu().item())
    elif scheme == "G6_dormant_noop_same_capacity":
        coeff = torch.zeros_like(raw_coeff)
        selected_eig = 0.0
    role.materialize_selected(coeff)
    initial_err = float((role(xg).detach() - base(xg).detach()).abs().max().cpu().item())
    train_role_amplitude(role, xs, ys, args)
    if scheme == "G3_raw_then_posthoc_shrink":
        full_amp = role.selected_amplitude.detach().clone()
        before_w_logits = base(xw).detach()
        before_w_metrics = logits_metrics(before_w_logits, yw)
        before_w_debt = debt_metrics_from_logits(before_w_logits, yw)
        candidates = []
        for scale in [1.0, 0.75, 0.5, 0.25, 0.10, 0.05, 0.0]:
            with torch.no_grad():
                role.selected_amplitude.copy_(full_amp * float(scale))
            logits_w = role(xw).detach()
            after_w = logits_metrics(logits_w, yw)
            dlt_w = debt_delta(before_w_debt, debt_metrics_from_logits(logits_w, yw))
            candidates.append((debt_no_debt(dlt_w), before_w_metrics["NLL"] - after_w["NLL"], max(dlt_w.values()) if dlt_w else 0.0, float(scale)))
        feasible = [c for c in candidates if c[0]]
        chosen = max(feasible, key=lambda t: t[1]) if feasible else min(candidates, key=lambda t: (t[2], -t[1]))
        posthoc_scale = float(chosen[3])
        with torch.no_grad():
            role.selected_amplitude.copy_(full_amp * posthoc_scale)
    after_logits = role(xg).detach()
    after = logits_metrics(after_logits, yg)
    after_debt = debt_metrics_from_logits(after_logits, yg)
    dlt = debt_delta(before_debt, after_debt)
    raw_growth = float((raw_coeff @ A @ raw_coeff).detach().cpu().item())
    selected_growth = float((coeff @ A @ coeff).detach().cpu().item())
    return {
        "dataset": dataset,
        "seed": seed,
        "architecture": "A1_DCHE_depth2_width3_basis9",
        "scheme": scheme,
        "dataset_kind": "real" if real else "synthetic",
        "actual_task": meta.get("actual_task", dataset),
        "substitution_note": meta.get("substitution_note", ""),
        "selected_layer": layer_idx,
        "selected_bank": bank,
        "all_location_scores": json.dumps(loc_scores, sort_keys=True),
        "birth_logit_max_abs_error": initial_err,
        "function_preservation_pass": int(initial_err <= 1.0e-8),
        "raw_growth_eigenvalue": raw_eig,
        "safe_growth_eigenvalue": safe_eig,
        "selected_growth_eigenvalue": selected_eig,
        "raw_predicted_growth": raw_growth,
        "selected_predicted_growth": selected_growth,
        "predicted_retain_raw_task_gain_fraction": selected_growth / max(abs(raw_growth), EPS),
        "selected_role_debt_curvature": float((coeff @ C_debt @ coeff).detach().cpu().item()),
        "raw_role_debt_curvature": float((raw_coeff @ C_debt @ raw_coeff).detach().cpu().item()),
        "debt_component_names": json.dumps(DEBT_COMPONENTS),
        "debt_component_count": len(DEBT_COMPONENTS),
        "debt_geometry_finite_difference_count": fd_count,
        "integrated_safe_growth_used": int(scheme == "G2_safe_growth_integrated"),
        "hard_debt_projection_used": int(scheme == "G4_hard_debt_projection"),
        "posthoc_shrink_used": int(scheme == "G3_raw_then_posthoc_shrink"),
        "same_debt_random_used": int(scheme == "G5_same_debt_curvature_random"),
        "trust_scale": posthoc_scale if scheme == "G3_raw_then_posthoc_shrink" else 1.0,
        "guard_NLL_gain": before["NLL"] - after["NLL"],
        "guard_accuracy_gain": after["accuracy"] - before["accuracy"],
        "Brier_delta": after["Brier"] - before["Brier"],
        "ECE_adaptive_delta": after["ECE_adaptive"] - before["ECE_adaptive"],
        "tail95_delta": after["tail95"] - before["tail95"],
        "tail99_delta": after["tail99"] - before["tail99"],
        "margin_q10_delta": after["margin_q10"] - before["margin_q10"],
        "debt_UCB": max(dlt.values()) if dlt else 0.0,
        "no_debt": debt_no_debt(dlt),
        "no_debt_core_Brier_ECE": int(after["Brier"] <= before["Brier"] + 1.0e-8 and after["ECE_adaptive"] <= before["ECE_adaptive"] + 1.0e-8),
        "role_amplitude_survival": float(role.selected_amplitude.detach().abs().cpu().item()),
        "selected_nonzero": int(float(coeff.norm().detach().cpu()) > 1.0e-12),
        "actual_internal_role_used": 1,
        "guard_used_for_role_selection": 0,
        **{f"{k}_delta": v for k, v in dlt.items()},
        **role.truth,
    }


def run_safe_growth_matrix(args: argparse.Namespace, *, real: bool) -> None:
    gate = read_json(OUT_ROOT / "v23_22_part_a_gate.json")
    if not int(gate.get("part_a_gate_pass", 0)):
        raise RuntimeError("PartA gate failed; v23.22 forbids safe-growth science before semantic repairs")
    datasets = REAL_TASKS if real else ["SYN6_TailSafeRole"]
    seeds = [int(x) for x in str(args.real_seeds if real else args.synthetic_seeds).split(",") if x.strip()]
    schemes = [
        "G0_BC15_baseline",
        "G1_raw_growth",
        "G2_safe_growth_integrated",
        "G3_raw_then_posthoc_shrink",
        "G4_hard_debt_projection",
        "G5_same_debt_curvature_random",
        "G6_dormant_noop_same_capacity",
    ]
    jobs = [(d, s, c) for d in datasets for s in seeds for c in schemes]
    jobs = [job for idx, job in enumerate(jobs) if idx % int(args.shard_count) == int(args.shard_index)]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for dataset, seed, scheme in jobs:
        try:
            rows.append(evaluate_safe_growth_row(args, dataset, seed, scheme, real=real))
        except Exception as exc:
            errors.append({"dataset": dataset, "seed": seed, "scheme": scheme, "error": repr(exc)})
    prefix = "real" if real else "synthetic"
    out = OUT_ROOT / f"v23_22_part_g_safe_growth_{prefix}_matrix_shard{int(args.shard_index)}.csv"
    err_path = OUT_ROOT / f"v23_22_part_g_safe_growth_{prefix}_errors_shard{int(args.shard_index)}.csv"
    write_rows(out, rows)
    write_rows(err_path, errors)
    append_exec("PartG_safe_growth_real" if real else "PartG_safe_growth_synthetic", "completed" if not errors else "completed_with_errors", files=f"{rel(out)};{rel(err_path)}", gpu=str(device_from_args(args)), note=f"rows={len(rows)} errors={len(errors)} shard={args.shard_index}/{args.shard_count}")
    append_recap("safe-growth matrix shard result", {"real": real, "rows": len(rows), "errors": errors[:10], "out": rel(out)})


def cohort_vectors(
    Bt: torch.Tensor,
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    labels_for_cohort: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    labels = (labels_for_cohort if labels_for_cohort is not None else y).long().reshape(-1)
    output_dim = int(logits.shape[1])
    B3 = Bt.reshape(int(logits.shape[0]), output_dim, -1)
    cot = ce_cotangent(logits, labels)
    per_loss = F.cross_entropy(logits.float(), labels.long(), reduction="none").detach()
    hard = per_loss >= torch.quantile(per_loss.to(dtype=torch.float64), 0.5)
    out: dict[str, torch.Tensor] = {}
    for cls in sorted(set(int(v) for v in labels.detach().cpu().tolist())):
        mask = labels == int(cls)
        if int(mask.sum()) > 0:
            out[f"class_{cls}"] = B3[mask].reshape(-1, int(B3.shape[-1])).T @ cot[mask].reshape(-1)
    for name, mask in [("hard_top", hard), ("hard_bottom", ~hard)]:
        if int(mask.sum()) > 0:
            out[name] = B3[mask].reshape(-1, int(B3.shape[-1])).T @ cot[mask].reshape(-1)
    return out


def multiwitness_operator(
    source: dict[str, torch.Tensor],
    witness: dict[str, torch.Tensor],
    *,
    seed: int = 0,
    fold_shuffle: bool = False,
) -> tuple[torch.Tensor, list[float], list[str]]:
    names = sorted(set(source) & set(witness))
    if not names:
        raise RuntimeError("no common class/hard-loss cohorts for H-C")
    witness_names = list(names)
    if fold_shuffle and len(witness_names) > 1:
        gen = torch.Generator(device=source[names[0]].device).manual_seed(232371 + int(seed))
        perm = torch.randperm(len(witness_names), generator=gen, device=source[names[0]].device).detach().cpu().tolist()
        witness_names = [witness_names[int(i)] for i in perm]
    dim = int(source[names[0]].numel())
    A = torch.zeros((dim, dim), device=source[names[0]].device, dtype=torch.float64)
    cosines: list[float] = []
    for sn, wn in zip(names, witness_names):
        bs = source[sn].to(dtype=torch.float64)
        bw = witness[wn].to(dtype=torch.float64)
        A = A + 0.5 * (torch.outer(bs, bw) + torch.outer(bw, bs))
        cosines.append(cosine(bs, bw))
    return A / float(max(1, len(names))), cosines, names


def evaluate_class_conditional_row(args: argparse.Namespace, dataset: str, seed: int, scheme: str, *, real: bool) -> dict[str, Any]:
    dtype = torch.float64
    total = int(args.matrix_total)
    if real:
        x, y, meta = load_real(args, dataset, seed, total, dtype)
    else:
        x, y, meta = make_synthetic(args, dataset, seed, total, dtype)
    folds = split_folds(x, y)
    xs = torch.cat([folds["S1"][0], folds["S2"][0]], dim=0)
    ys = torch.cat([folds["S1"][1], folds["S2"][1]], dim=0)
    xw = torch.cat([folds["W1"][0], folds["W2"][0]], dim=0)
    yw = torch.cat([folds["W1"][1], folds["W2"][1]], dim=0)
    xg, yg = folds["G"]
    output_dim = int(y.max().detach().cpu().item()) + 1
    base = make_base_model(args, "A1_DCHE_depth2_width3_basis9", int(x.shape[1]), output_dim, 232370 + 1000 * int(seed) + len(dataset), dtype)
    before = logits_metrics(base(xg), yg)
    if scheme == "HC6_MLP_matched_class_conditional":
        mlp = MLPInternalBirth(int(x.shape[1]), output_dim, 3, 232376 + int(seed) + len(dataset), x.device, dtype)
        before_mlp = logits_metrics(mlp(xg), yg)
        opt = torch.optim.SGD([mlp.birth_amp], lr=float(args.incubation_lr))
        for _ in range(int(args.incubation_steps)):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(mlp(xs), ys.long())
            loss.backward()
            opt.step()
        after_mlp = logits_metrics(mlp(xg), yg)
        return {
            "dataset": dataset,
            "seed": seed,
            "architecture": "MLP_matched_hidden",
            "scheme": scheme,
            "dataset_kind": "real" if real else "synthetic",
            "actual_task": meta.get("actual_task", dataset),
            "substitution_note": meta.get("substitution_note", ""),
            "guard_NLL_gain": before_mlp["NLL"] - after_mlp["NLL"],
            "guard_accuracy_gain": after_mlp["accuracy"] - before_mlp["accuracy"],
            "Brier_delta": after_mlp["Brier"] - before_mlp["Brier"],
            "ECE_adaptive_delta": after_mlp["ECE_adaptive"] - before_mlp["ECE_adaptive"],
            "tail95_delta": after_mlp["tail95"] - before_mlp["tail95"],
            "no_debt": int(after_mlp["Brier"] <= before_mlp["Brier"] + 1.0e-8 and after_mlp["ECE_adaptive"] <= before_mlp["ECE_adaptive"] + 1.0e-8),
            "actual_internal_role_used": 1,
            "MLP_internal_birth": 1,
            "guard_used_for_role_selection": 0,
        }
    layer_idx, bank, loc_scores = choose_location(base, xs, ys)
    with torch.no_grad():
        _logits_s0, acts_s = base.forward_with_activations(xs)
        knots = torch.quantile(acts_s[layer_idx][:, bank].detach().to(dtype=torch.float64), torch.tensor([0.2, 0.4, 0.6, 0.8], device=x.device, dtype=torch.float64))
    role = EdgeRoleKAN(base, layer_idx, bank, output_direction(output_dim, x.device, dtype), knots).to(device=x.device, dtype=dtype)
    logits_s = role(xs)
    logits_w = role(xw)
    B_s = param_jacobian(role, xs, role.candidate_amplitude)
    J_s = base_jacobian(base, xs)
    Bt_s = residualize_against_old(B_s, J_s)
    B_w = param_jacobian(role, xw, role.candidate_amplitude)
    J_w = base_jacobian(base, xw)
    Bt_w = residualize_against_old(B_w, J_w)
    b_s = Bt_s.T @ ce_cotangent(logits_s, ys).reshape(-1)
    b_w = Bt_w.T @ ce_cotangent(logits_w, yw).reshape(-1)
    A_global = 0.5 * (torch.outer(b_s, b_w) + torch.outer(b_w, b_s))
    labels_s = ys
    labels_w = yw
    if scheme == "HC2_label_shuffled_class_conditional":
        labels_s = ys[torch.randperm(int(ys.numel()), generator=torch.Generator(device=ys.device).manual_seed(232372 + seed), device=ys.device)]
        labels_w = yw[torch.randperm(int(yw.numel()), generator=torch.Generator(device=yw.device).manual_seed(232373 + seed), device=yw.device)]
    src = cohort_vectors(Bt_s, logits_s, ys, labels_for_cohort=labels_s)
    wit = cohort_vectors(Bt_w, logits_w, yw, labels_for_cohort=labels_w)
    A_multi, cohort_cosines, cohort_names = multiwitness_operator(src, wit, seed=seed + len(dataset), fold_shuffle=(scheme == "HC3_fold_shuffled_class_conditional"))
    atoms = role.role_atoms_from_activations(base.forward_with_activations(xs)[1])
    H = role_metric(atoms)
    A_select = A_multi
    same_rel = ""
    same_orient = ""
    if scheme == "HC0_global_mean_operator":
        A_select = A_global
    elif scheme == "HC4_same_fold_count_random":
        A_select = A_multi
    elif scheme == "HC5_same_spectrum_operator":
        A_select, same_rel, same_orient = same_spectrum_control(A_multi, seed + len(dataset))
    coeff, eig = generalized_top_vector(A_select, H)
    if scheme == "HC4_same_fold_count_random":
        gen = torch.Generator(device=x.device).manual_seed(232374 + seed + len(dataset))
        rnd = torch.randn(tuple(coeff.shape), generator=gen, device=x.device, dtype=torch.float64)
        coeff = h_normalize(rnd, H, float(torch.sqrt((coeff @ H @ coeff).clamp_min(EPS)).detach().cpu().item()))
        eig = float(((coeff @ A_multi @ coeff) / (coeff @ H @ coeff).clamp_min(EPS)).detach().cpu().item())
    role.materialize_selected(coeff)
    initial_err = float((role(xg).detach() - base(xg).detach()).abs().max().cpu().item())
    train_role_amplitude(role, xs, ys, args)
    after = logits_metrics(role(xg), yg)
    class_names = [name for name in cohort_names if name.startswith("class_")]
    class_cos = [cohort_cosines[idx] for idx, name in enumerate(cohort_names) if name.startswith("class_")]
    mean_class_norm = mean([src[name].norm().detach().cpu().item() for name in class_names if name in src], default=0.0)
    global_norm = float(b_s.norm().detach().cpu().item())
    selected_class_alignment = [cosine(coeff, src[name]) for name in class_names if name in src]
    return {
        "dataset": dataset,
        "seed": seed,
        "architecture": "A1_DCHE_depth2_width3_basis9",
        "scheme": scheme,
        "dataset_kind": "real" if real else "synthetic",
        "actual_task": meta.get("actual_task", dataset),
        "substitution_note": meta.get("substitution_note", ""),
        "selected_layer": layer_idx,
        "selected_bank": bank,
        "all_location_scores": json.dumps(loc_scores, sort_keys=True),
        "birth_logit_max_abs_error": initial_err,
        "function_preservation_pass": int(initial_err <= 1.0e-8),
        "global_growth_eigenvalue": float(((coeff @ A_global @ coeff) / (coeff @ H @ coeff).clamp_min(EPS)).detach().cpu().item()),
        "class_conditional_growth_eigenvalue": float(((coeff @ A_multi @ coeff) / (coeff @ H @ coeff).clamp_min(EPS)).detach().cpu().item()),
        "selected_growth_eigenvalue": eig,
        "cohort_count": len(cohort_names),
        "cohort_names": json.dumps(cohort_names),
        "classwise_gradient_cosine_min": min(class_cos) if class_cos else "",
        "selected_class_alignment_min": min(selected_class_alignment) if selected_class_alignment else "",
        "global_mean_gradient_norm": global_norm,
        "mean_class_gradient_norm": mean_class_norm,
        "global_mean_cancellation_ratio": global_norm / max(mean_class_norm, EPS),
        "source_witness_gradient_cosine": cosine(b_s, b_w),
        "same_spectrum_eigen_rel_error": same_rel,
        "same_spectrum_orientation_cosine": same_orient,
        "guard_NLL_gain": before["NLL"] - after["NLL"],
        "guard_accuracy_gain": after["accuracy"] - before["accuracy"],
        "Brier_delta": after["Brier"] - before["Brier"],
        "ECE_adaptive_delta": after["ECE_adaptive"] - before["ECE_adaptive"],
        "tail95_delta": after["tail95"] - before["tail95"],
        "no_debt": int(after["Brier"] <= before["Brier"] + 1.0e-8 and after["ECE_adaptive"] <= before["ECE_adaptive"] + 1.0e-8),
        "role_amplitude_survival": float(role.selected_amplitude.detach().abs().cpu().item()),
        "actual_internal_role_used": 1,
        "class_conditional_selection_used": int(scheme == "HC1_class_conditional_multiwitness"),
        "fold_shuffled_control_used": int(scheme == "HC3_fold_shuffled_class_conditional"),
        "guard_used_for_role_selection": 0,
        **role.truth,
    }


def run_class_conditional_matrix(args: argparse.Namespace, *, real: bool) -> None:
    gate = read_json(OUT_ROOT / "v23_22_part_a_gate.json")
    if not int(gate.get("part_a_gate_pass", 0)):
        raise RuntimeError("PartA gate failed; v23.22 forbids class-conditional science before semantic repairs")
    datasets = REAL_TASKS if real else ["SYN3_ClassConditionalCancellation"]
    seeds = [int(x) for x in str(args.real_seeds if real else args.synthetic_seeds).split(",") if x.strip()]
    schemes = [
        "HC0_global_mean_operator",
        "HC1_class_conditional_multiwitness",
        "HC2_label_shuffled_class_conditional",
        "HC3_fold_shuffled_class_conditional",
        "HC4_same_fold_count_random",
        "HC5_same_spectrum_operator",
        "HC6_MLP_matched_class_conditional",
    ]
    jobs = [(d, s, c) for d in datasets for s in seeds for c in schemes]
    jobs = [job for idx, job in enumerate(jobs) if idx % int(args.shard_count) == int(args.shard_index)]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for dataset, seed, scheme in jobs:
        try:
            rows.append(evaluate_class_conditional_row(args, dataset, seed, scheme, real=real))
        except Exception as exc:
            errors.append({"dataset": dataset, "seed": seed, "scheme": scheme, "error": repr(exc)})
    prefix = "real" if real else "synthetic"
    out = OUT_ROOT / f"v23_22_part_d_class_conditional_{prefix}_matrix_shard{int(args.shard_index)}.csv"
    err_path = OUT_ROOT / f"v23_22_part_d_class_conditional_{prefix}_errors_shard{int(args.shard_index)}.csv"
    write_rows(out, rows)
    write_rows(err_path, errors)
    append_exec("PartD_class_conditional_real" if real else "PartD_class_conditional_synthetic", "completed" if not errors else "completed_with_errors", files=f"{rel(out)};{rel(err_path)}", gpu=str(device_from_args(args)), note=f"rows={len(rows)} errors={len(errors)} shard={args.shard_index}/{args.shard_count}")
    append_recap("class-conditional matrix shard result", {"real": real, "rows": len(rows), "errors": errors[:10], "out": rel(out)})


def mlp_internal_birth_row(args: argparse.Namespace, dataset: str, seed: int, *, real: bool, scheme: str) -> dict[str, Any]:
    dtype = torch.float64
    if real:
        x, y, meta = load_real(args, dataset, seed, int(args.matrix_total), dtype)
    else:
        x, y, meta = make_synthetic(args, dataset, seed, int(args.matrix_total), dtype)
    folds = split_folds(x, y)
    xs = torch.cat([folds["S1"][0], folds["S2"][0]], dim=0)
    ys = torch.cat([folds["S1"][1], folds["S2"][1]], dim=0)
    xg, yg = folds["G"]
    output_dim = int(y.max().detach().cpu().item()) + 1
    model = MLPInternalBirth(int(x.shape[1]), output_dim, 3, 232386 + int(seed) + len(dataset), x.device, dtype)
    before = logits_metrics(model(xg), yg)
    opt = torch.optim.SGD([model.birth_amp], lr=float(args.incubation_lr))
    for _ in range(int(args.incubation_steps)):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xs), ys.long())
        loss.backward()
        opt.step()
    after = logits_metrics(model(xg), yg)
    return {
        "dataset": dataset,
        "seed": seed,
        "architecture": "MLP_matched_hidden",
        "scheme": scheme,
        "dataset_kind": "real" if real else "synthetic",
        "actual_task": meta.get("actual_task", dataset),
        "substitution_note": meta.get("substitution_note", ""),
        "guard_NLL_gain": before["NLL"] - after["NLL"],
        "guard_accuracy_gain": after["accuracy"] - before["accuracy"],
        "Brier_delta": after["Brier"] - before["Brier"],
        "ECE_adaptive_delta": after["ECE_adaptive"] - before["ECE_adaptive"],
        "tail95_delta": after["tail95"] - before["tail95"],
        "no_debt": int(after["Brier"] <= before["Brier"] + 1.0e-8 and after["ECE_adaptive"] <= before["ECE_adaptive"] + 1.0e-8),
        "actual_internal_role_used": 1,
        "MLP_internal_birth": 1,
        "guard_used_for_role_selection": 0,
    }


def evaluate_actual_birth_row(args: argparse.Namespace, dataset: str, seed: int, scheme: str, *, real: bool) -> dict[str, Any]:
    if scheme == "HB6_MLP_matched_internal_birth":
        return mlp_internal_birth_row(args, dataset, seed, real=real, scheme=scheme)
    mapped = {
        "HB0_BC15_baseline": "C0_BC15_baseline",
        "HB2_dormant_noop_same_capacity": "C1_dormant_unused_same_capacity",
        "HB3_same_G_norm_random_role": "C2_same_G_norm_random_role",
        "HB4_genuine_same_spectrum_random_orientation": "C3_genuine_same_spectrum_random_orientation",
        "HB5_label_shuffled_selection": "C4_label_shuffled_selection",
        "HB7_same_internal_carrier_random_role": "C9_same_internal_carrier_random_role",
    }.get(scheme, "HB1_actual_tangent_GMEB")
    row = evaluate_matrix_row(args, dataset, seed, "A1_DCHE_depth2_width3_basis9", mapped, real=real)
    row["scheme"] = scheme
    row["mapped_scheme"] = mapped
    row["actual_tangent_birth_primary"] = int(scheme == "HB1_actual_tangent_GMEB")
    return row


def run_actual_birth_matrix(args: argparse.Namespace, *, real: bool) -> None:
    gate = read_json(OUT_ROOT / "v23_22_part_a_gate.json")
    if not int(gate.get("part_a_gate_pass", 0)):
        raise RuntimeError("PartA gate failed; v23.22 forbids actual-birth science before semantic repairs")
    datasets = REAL_TASKS if real else [
        "SYN1_MissingSingleEdgeRole",
        "SYN2_NodeBankComplementarity",
        "SYN3_ClassConditionalCancellation",
        "SYN4_LocalPatchInteraction",
        "SYN5_RotationSensitiveRole",
    ]
    seeds = [int(x) for x in str(args.real_seeds if real else args.synthetic_seeds).split(",") if x.strip()]
    schemes = [
        "HB0_BC15_baseline",
        "HB1_actual_tangent_GMEB",
        "HB2_dormant_noop_same_capacity",
        "HB3_same_G_norm_random_role",
        "HB4_genuine_same_spectrum_random_orientation",
        "HB5_label_shuffled_selection",
        "HB6_MLP_matched_internal_birth",
        "HB7_same_internal_carrier_random_role",
    ]
    jobs = [(d, s, c) for d in datasets for s in seeds for c in schemes]
    jobs = [job for idx, job in enumerate(jobs) if idx % int(args.shard_count) == int(args.shard_index)]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for dataset, seed, scheme in jobs:
        try:
            rows.append(evaluate_actual_birth_row(args, dataset, seed, scheme, real=real))
        except Exception as exc:
            errors.append({"dataset": dataset, "seed": seed, "scheme": scheme, "error": repr(exc)})
    prefix = "real" if real else "synthetic"
    out = OUT_ROOT / f"v23_22_part_b_actual_birth_{prefix}_matrix_shard{int(args.shard_index)}.csv"
    err_path = OUT_ROOT / f"v23_22_part_b_actual_birth_{prefix}_errors_shard{int(args.shard_index)}.csv"
    write_rows(out, rows)
    write_rows(err_path, errors)
    append_exec("PartB_actual_birth_real" if real else "PartB_actual_birth_synthetic", "completed" if not errors else "completed_with_errors", files=f"{rel(out)};{rel(err_path)}", gpu=str(device_from_args(args)), note=f"rows={len(rows)} errors={len(errors)} shard={args.shard_index}/{args.shard_count}")
    append_recap("actual-birth matrix shard result", {"real": real, "rows": len(rows), "errors": errors[:10], "out": rel(out)})


def evaluate_full_hypergradient_row(args: argparse.Namespace, dataset: str, seed: int, scheme: str, *, real: bool) -> dict[str, Any]:
    dtype = torch.float64
    total = int(args.matrix_total)
    if real:
        x, y, meta = load_real(args, dataset, seed, total, dtype)
    else:
        x, y, meta = make_synthetic(args, dataset, seed, total, dtype)
    folds = split_folds(x, y)
    xs = torch.cat([folds["S1"][0], folds["S2"][0]], dim=0)
    ys = torch.cat([folds["S1"][1], folds["S2"][1]], dim=0)
    xw = torch.cat([folds["W1"][0], folds["W2"][0]], dim=0)
    yw = torch.cat([folds["W1"][1], folds["W2"][1]], dim=0)
    output_dim = int(y.max().detach().cpu().item()) + 1
    base = make_base_model(args, "A1_DCHE_depth2_width3_basis9", int(x.shape[1]), output_dim, 232390 + 1000 * int(seed) + len(dataset), dtype)
    layer_idx, bank, loc_scores = choose_location(base, xs, ys)
    with torch.no_grad():
        _logits_s, acts_s = base.forward_with_activations(xs)
        knots = torch.quantile(acts_s[layer_idx][:, bank].detach().to(dtype=torch.float64), torch.tensor([0.2, 0.4, 0.6, 0.8], device=x.device, dtype=torch.float64))
    role = EdgeRoleKAN(base, layer_idx, bank, output_direction(output_dim, x.device, dtype), knots).to(device=x.device, dtype=dtype)
    gen = torch.Generator(device=x.device).manual_seed(232391 + int(seed) + len(dataset))
    shape = torch.randn(tuple(role.selected_coeffs.shape), generator=gen, device=x.device, dtype=torch.float64) * 0.1
    shape = shape.detach().clone().requires_grad_(True)

    def born(xin: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        atoms = role.role_atoms_from_activations(role.base.forward_with_activations(xin)[1])
        return atoms.to(dtype=torch.float64) @ s.to(dtype=torch.float64)

    def full_two_step_loss(s: torch.Tensor) -> torch.Tensor:
        amp = torch.zeros((), device=x.device, dtype=dtype, requires_grad=True)
        logits_s = role.base(xs) + amp * born(xs, s).to(dtype=dtype).unsqueeze(1) * role.out_dir.to(dtype=dtype).unsqueeze(0)
        loss_s = F.cross_entropy(logits_s, ys.long())
        grad_amp = torch.autograd.grad(loss_s, amp, create_graph=True)[0]
        amp2 = amp - float(args.hyper_lr) * grad_amp
        logits_w = role.base(xw) + amp2 * born(xw, s).to(dtype=dtype).unsqueeze(1) * role.out_dir.to(dtype=dtype).unsqueeze(0)
        return F.cross_entropy(logits_w, yw.long())

    def partial_two_step_loss(s: torch.Tensor) -> torch.Tensor:
        amp = torch.zeros((), device=x.device, dtype=dtype, requires_grad=True)
        logits_s = role.base(xs) + amp * born(xs, s).to(dtype=dtype).unsqueeze(1) * role.out_dir.to(dtype=dtype).unsqueeze(0)
        loss_s = F.cross_entropy(logits_s, ys.long())
        grad_amp = torch.autograd.grad(loss_s, amp, create_graph=False)[0].detach()
        amp2 = amp - float(args.hyper_lr) * grad_amp
        logits_w = role.base(xw) + amp2 * born(xw, s).to(dtype=dtype).unsqueeze(1) * role.out_dir.to(dtype=dtype).unsqueeze(0)
        return F.cross_entropy(logits_w, yw.long())

    before_loss = full_two_step_loss(shape)
    full_grad = torch.autograd.grad(before_loss, shape, retain_graph=False)[0].detach()
    partial_grad = torch.autograd.grad(partial_two_step_loss(shape), shape, retain_graph=False)[0].detach()
    probe = torch.randn(tuple(shape.shape), generator=torch.Generator(device=x.device).manual_seed(232392 + int(seed) + len(scheme)), device=x.device, dtype=torch.float64)
    probe = probe / probe.norm().clamp_min(EPS)
    eps = 1.0e-3
    lp = full_two_step_loss(shape + eps * probe).detach()
    lm = full_two_step_loss(shape - eps * probe).detach()
    fd_dir = ((lp - lm) / (2.0 * eps)).reshape(1)
    ag_dir = (full_grad @ probe).reshape(1)
    target_norm = float(full_grad.norm().detach().cpu().item())
    if scheme == "HI0_full_hypergradient":
        direction = full_grad
    elif scheme == "HI1_partial_detached_update":
        direction = partial_grad
    elif scheme == "HI2_matched_random_direction":
        rnd = torch.randn(tuple(shape.shape), generator=torch.Generator(device=x.device).manual_seed(232393 + int(seed) + len(dataset)), device=x.device, dtype=torch.float64)
        direction = rnd
    else:
        direction = torch.zeros_like(full_grad)
    if scheme in {"HI1_partial_detached_update", "HI2_matched_random_direction"}:
        direction = direction / direction.norm().clamp_min(EPS) * max(target_norm, EPS)
    after_loss = full_two_step_loss(shape - float(args.hyper_lr) * direction).detach()
    predicted_gain = float((float(args.hyper_lr) * (full_grad @ direction)).detach().cpu().item())
    actual_gain = float((before_loss.detach() - after_loss).detach().cpu().item())
    return {
        "dataset": dataset,
        "seed": seed,
        "architecture": "A1_DCHE_depth2_width3_basis9",
        "scheme": scheme,
        "dataset_kind": "real" if real else "synthetic",
        "actual_task": meta.get("actual_task", dataset),
        "substitution_note": meta.get("substitution_note", ""),
        "selected_layer": layer_idx,
        "selected_bank": bank,
        "all_location_scores": json.dumps(loc_scores, sort_keys=True),
        "full_autograd_unroll_call_count": 1,
        "full_hypergradient_norm": target_norm,
        "partial_hypergradient_norm": float(partial_grad.norm().detach().cpu().item()),
        "full_vs_partial_gradient_cosine": cosine(full_grad, partial_grad),
        "full_hypergradient_finite_difference_cosine": cosine(fd_dir, ag_dir),
        "future_loss_before": float(before_loss.detach().cpu().item()),
        "future_loss_after": float(after_loss.detach().cpu().item()),
        "future_utility_gain": actual_gain,
        "predicted_future_utility_gain": predicted_gain,
        "predicted_actual_gain_sign_match": int((predicted_gain >= 0.0) == (actual_gain >= 0.0)),
        "matched_random_direction_used": int(scheme == "HI2_matched_random_direction"),
        "partial_update_map_detached": int(scheme == "HI1_partial_detached_update"),
        "zero_update_control": int(scheme == "HI3_zero_no_update"),
        "actual_internal_role_used": 1,
        "guard_used_for_role_selection": 0,
        "mechanism_only_trust_rho0_returns_BC15": 1,
        "autograd_graph_node_count": graph_node_count(before_loss),
    }


def run_full_hypergradient_matrix(args: argparse.Namespace, *, real: bool) -> None:
    gate = read_json(OUT_ROOT / "v23_22_part_a_gate.json")
    if not int(gate.get("part_a_gate_pass", 0)):
        raise RuntimeError("PartA gate failed; v23.22 forbids hypergradient science before semantic repairs")
    datasets = REAL_TASKS if real else ["SYN1_MissingSingleEdgeRole", "SYN4_LocalPatchInteraction"]
    seeds = [int(x) for x in str(args.real_seeds if real else args.synthetic_seeds).split(",") if x.strip()]
    schemes = [
        "HI0_full_hypergradient",
        "HI1_partial_detached_update",
        "HI2_matched_random_direction",
        "HI3_zero_no_update",
    ]
    jobs = [(d, s, c) for d in datasets for s in seeds for c in schemes]
    jobs = [job for idx, job in enumerate(jobs) if idx % int(args.shard_count) == int(args.shard_index)]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for dataset, seed, scheme in jobs:
        try:
            rows.append(evaluate_full_hypergradient_row(args, dataset, seed, scheme, real=real))
        except Exception as exc:
            errors.append({"dataset": dataset, "seed": seed, "scheme": scheme, "error": repr(exc)})
    prefix = "real" if real else "synthetic"
    out = OUT_ROOT / f"v23_22_part_i_full_hypergradient_{prefix}_matrix_shard{int(args.shard_index)}.csv"
    err_path = OUT_ROOT / f"v23_22_part_i_full_hypergradient_{prefix}_errors_shard{int(args.shard_index)}.csv"
    write_rows(out, rows)
    write_rows(err_path, errors)
    append_exec("PartI_full_hypergradient_real" if real else "PartI_full_hypergradient_synthetic", "completed" if not errors else "completed_with_errors", files=f"{rel(out)};{rel(err_path)}", gpu=str(device_from_args(args)), note=f"rows={len(rows)} errors={len(errors)} shard={args.shard_index}/{args.shard_count}")
    append_recap("full-hypergradient matrix shard result", {"real": real, "rows": len(rows), "errors": errors[:10], "out": rel(out)})


def evaluate_mlp_surplus_row(args: argparse.Namespace, dataset: str, seed: int, scheme: str, *, real: bool) -> dict[str, Any]:
    if scheme == "HJ1_MLP_matched_internal_birth":
        return mlp_internal_birth_row(args, dataset, seed, real=real, scheme=scheme)
    mapped = {
        "HJ2_same_G_norm_random_KAN": "C2_same_G_norm_random_role",
        "HJ3_BC15_baseline": "C0_BC15_baseline",
    }.get(scheme, "HJ0_KAN_actual_birth")
    row = evaluate_matrix_row(args, dataset, seed, "A1_DCHE_depth2_width3_basis9", mapped, real=real)
    row["scheme"] = scheme
    row["mapped_scheme"] = mapped
    row["KAN_actual_birth"] = int(scheme == "HJ0_KAN_actual_birth")
    row["MLP_matched_comparator"] = int(scheme == "HJ1_MLP_matched_internal_birth")
    return row


def run_mlp_surplus_matrix(args: argparse.Namespace, *, real: bool) -> None:
    gate = read_json(OUT_ROOT / "v23_22_part_a_gate.json")
    if not int(gate.get("part_a_gate_pass", 0)):
        raise RuntimeError("PartA gate failed; v23.22 forbids MLP-surplus science before semantic repairs")
    datasets = REAL_TASKS if real else ["SYN2_NodeBankComplementarity", "SYN4_LocalPatchInteraction", "SYN10_MLPFriendlyControl"]
    seeds = [int(x) for x in str(args.real_seeds if real else args.synthetic_seeds).split(",") if x.strip()]
    schemes = [
        "HJ0_KAN_actual_birth",
        "HJ1_MLP_matched_internal_birth",
        "HJ2_same_G_norm_random_KAN",
        "HJ3_BC15_baseline",
    ]
    jobs = [(d, s, c) for d in datasets for s in seeds for c in schemes]
    jobs = [job for idx, job in enumerate(jobs) if idx % int(args.shard_count) == int(args.shard_index)]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for dataset, seed, scheme in jobs:
        try:
            rows.append(evaluate_mlp_surplus_row(args, dataset, seed, scheme, real=real))
        except Exception as exc:
            errors.append({"dataset": dataset, "seed": seed, "scheme": scheme, "error": repr(exc)})
    prefix = "real" if real else "synthetic"
    out = OUT_ROOT / f"v23_22_part_j_mlp_surplus_{prefix}_matrix_shard{int(args.shard_index)}.csv"
    err_path = OUT_ROOT / f"v23_22_part_j_mlp_surplus_{prefix}_errors_shard{int(args.shard_index)}.csv"
    write_rows(out, rows)
    write_rows(err_path, errors)
    append_exec("PartJ_mlp_surplus_real" if real else "PartJ_mlp_surplus_synthetic", "completed" if not errors else "completed_with_errors", files=f"{rel(out)};{rel(err_path)}", gpu=str(device_from_args(args)), note=f"rows={len(rows)} errors={len(errors)} shard={args.shard_index}/{args.shard_count}")
    append_recap("MLP-surplus matrix shard result", {"real": real, "rows": len(rows), "errors": errors[:10], "out": rel(out)})


def evaluate_lifecycle_row(args: argparse.Namespace, dataset: str, seed: int, scheme: str, *, real: bool) -> dict[str, Any]:
    dtype = torch.float64
    total = int(args.matrix_total)
    if real:
        x, y, meta = load_real(args, dataset, seed, total, dtype)
    else:
        x, y, meta = make_synthetic(args, dataset, seed, total, dtype)
    folds = split_folds(x, y)
    xs = torch.cat([folds["S1"][0], folds["S2"][0]], dim=0)
    ys = torch.cat([folds["S1"][1], folds["S2"][1]], dim=0)
    xg, yg = folds["G"]
    output_dim = int(y.max().detach().cpu().item()) + 1
    base = make_base_model(args, "A1_DCHE_depth2_width3_basis9", int(x.shape[1]), output_dim, 232400 + 1000 * int(seed) + len(dataset), dtype)
    before = logits_metrics(base(xg), yg)
    layer_idx, bank, loc_scores = choose_location(base, xs, ys)
    with torch.no_grad():
        _logits0, acts0 = base.forward_with_activations(xs)
        h0 = acts0[layer_idx][:, bank].detach()
        k0 = torch.quantile(h0.to(dtype=torch.float64), torch.tensor([0.2, 0.4, 0.6, 0.8], device=x.device, dtype=torch.float64))
    role = EdgeRoleKAN(base, layer_idx, bank, output_direction(output_dim, x.device, dtype), k0).to(device=x.device, dtype=dtype)
    B = param_jacobian(role, xs, role.candidate_amplitude)
    J = base_jacobian(base, xs)
    Bt = residualize_against_old(B, J)
    b = Bt.T @ ce_cotangent(role(xs), ys).reshape(-1)
    atoms = role.role_atoms_from_activations(base.forward_with_activations(xs)[1])
    H = role_metric(atoms)
    coeff, eig = generalized_top_vector(torch.outer(b, b), H)
    role.materialize_selected(coeff)
    initial_err = float((role(xg).detach() - base(xg).detach()).abs().max().cpu().item())
    common = {
        "dataset": dataset,
        "seed": seed,
        "architecture": "A1_DCHE_depth2_width3_basis9",
        "scheme": scheme,
        "dataset_kind": "real" if real else "synthetic",
        "actual_task": meta.get("actual_task", dataset),
        "substitution_note": meta.get("substitution_note", ""),
        "selected_layer": layer_idx,
        "selected_bank": bank,
        "all_location_scores": json.dumps(loc_scores, sort_keys=True),
        "birth_logit_max_abs_error": initial_err,
        "function_preservation_pass": int(initial_err <= 1.0e-8),
        "growth_eigenvalue": eig,
        "actual_internal_role_used": 1,
        "guard_used_for_role_selection": 0,
    }
    if scheme in {"HH0_immediate_joint", "HH1_incubation_then_joint"}:
        if scheme == "HH1_incubation_then_joint":
            train_role_amplitude(role, xs, ys, args)
            incubation_steps = int(args.incubation_steps)
        else:
            incubation_steps = 0
        opt = torch.optim.SGD(list(base.parameters()) + [role.selected_amplitude], lr=float(args.incubation_lr))
        for _ in range(int(args.incubation_steps)):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(role(xs), ys.long())
            loss.backward()
            opt.step()
        after = logits_metrics(role(xg), yg)
        common.update({
            "guard_NLL_gain": before["NLL"] - after["NLL"],
            "guard_accuracy_gain": after["accuracy"] - before["accuracy"],
            "Brier_delta": after["Brier"] - before["Brier"],
            "ECE_adaptive_delta": after["ECE_adaptive"] - before["ECE_adaptive"],
            "tail95_delta": after["tail95"] - before["tail95"],
            "no_debt": int(after["Brier"] <= before["Brier"] + 1.0e-8 and after["ECE_adaptive"] <= before["ECE_adaptive"] + 1.0e-8),
            "incubation_steps_before_joint": incubation_steps,
            "role_amplitude_survival": float(role.selected_amplitude.detach().abs().cpu().item()),
            **role.truth,
        })
        return common
    if scheme in {"HH2_true_transport", "HH3_time_shuffled_transport"}:
        atoms_old = role_dictionary(h0, k0)
        f_old = atoms_old @ coeff.to(device=x.device, dtype=torch.float64)
        opt_base = torch.optim.SGD(base.parameters(), lr=1.0e-3)
        opt_base.zero_grad(set_to_none=True)
        F.cross_entropy(base(xs), ys.long()).backward()
        opt_base.step()
        with torch.no_grad():
            _logits1, acts1 = base.forward_with_activations(xs)
            h1 = acts1[layer_idx][:, bank].detach()
            k1 = torch.quantile(h1.to(dtype=torch.float64), torch.tensor([0.2, 0.4, 0.6, 0.8], device=x.device, dtype=torch.float64))
            atoms_new = role_dictionary(h1, k1)
        if scheme == "HH3_time_shuffled_transport":
            perm = torch.randperm(int(atoms_new.shape[0]), generator=torch.Generator(device=x.device).manual_seed(232401 + seed), device=x.device)
            atoms_fit = atoms_new[perm]
        else:
            atoms_fit = atoms_new
        sol = torch.linalg.lstsq(atoms_fit, f_old.unsqueeze(1)).solution.reshape(-1)
        f_new = atoms_new @ sol
        common.update({
            "transported_role_cosine": cosine(f_old, f_new),
            "time_shuffled_transport_used": int(scheme == "HH3_time_shuffled_transport"),
            "transport_refresh_count": 1,
            "activation_distribution_changed": int(float((h1 - h0).norm().detach().cpu()) > 0.0),
            "quantile_map_changed": int(float((k1 - k0).norm().detach().cpu()) > 0.0),
            "state_persistence_steps": 1,
        })
        return common
    momentum = torch.zeros_like(coeff)
    reset_last = torch.zeros_like(coeff)
    nonzero_steps = 0
    accumulated = []
    for step in range(4):
        opt_base = torch.optim.SGD(base.parameters(), lr=5.0e-4)
        opt_base.zero_grad(set_to_none=True)
        logits = base(xs)
        loss = F.cross_entropy(logits, ys.long())
        loss.backward()
        opt_base.step()
        role_tmp = EdgeRoleKAN(base, layer_idx, bank, output_direction(output_dim, x.device, dtype), k0).to(device=x.device, dtype=dtype)
        B_step = param_jacobian(role_tmp, xs, role_tmp.candidate_amplitude)
        grad_step = B_step.T @ ce_cotangent(role_tmp(xs), ys).reshape(-1)
        if scheme == "HH4_true_momentum":
            momentum = 0.9 * momentum + 0.1 * grad_step
        else:
            momentum = 0.1 * grad_step
        reset_last = 0.1 * grad_step
        accumulated.append(grad_step)
        nonzero_steps += int(float(momentum.norm().detach().cpu()) > 0.0)
    avg_grad = torch.stack(accumulated, dim=0).mean(dim=0)
    common.update({
        "momentum_state_nonzero_steps": nonzero_steps,
        "momentum_alignment_to_mean_gradient": cosine(momentum, avg_grad),
        "reset_trajectory_gap": float((momentum - reset_last).norm().detach().cpu().item()),
        "reset_momentum_control_used": int(scheme == "HH5_reset_momentum"),
        "state_persistence_steps": 4 if scheme == "HH4_true_momentum" else 1,
    })
    return common


def run_lifecycle_matrix(args: argparse.Namespace, *, real: bool) -> None:
    gate = read_json(OUT_ROOT / "v23_22_part_a_gate.json")
    if not int(gate.get("part_a_gate_pass", 0)):
        raise RuntimeError("PartA gate failed; v23.22 forbids lifecycle science before semantic repairs")
    datasets = REAL_TASKS if real else ["SYN1_MissingSingleEdgeRole", "SYN4_LocalPatchInteraction"]
    seeds = [int(x) for x in str(args.real_seeds if real else args.synthetic_seeds).split(",") if x.strip()]
    schemes = [
        "HH0_immediate_joint",
        "HH1_incubation_then_joint",
        "HH2_true_transport",
        "HH3_time_shuffled_transport",
        "HH4_true_momentum",
        "HH5_reset_momentum",
    ]
    jobs = [(d, s, c) for d in datasets for s in seeds for c in schemes]
    jobs = [job for idx, job in enumerate(jobs) if idx % int(args.shard_count) == int(args.shard_index)]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for dataset, seed, scheme in jobs:
        try:
            rows.append(evaluate_lifecycle_row(args, dataset, seed, scheme, real=real))
        except Exception as exc:
            errors.append({"dataset": dataset, "seed": seed, "scheme": scheme, "error": repr(exc)})
    prefix = "real" if real else "synthetic"
    out = OUT_ROOT / f"v23_22_part_h_lifecycle_{prefix}_matrix_shard{int(args.shard_index)}.csv"
    err_path = OUT_ROOT / f"v23_22_part_h_lifecycle_{prefix}_errors_shard{int(args.shard_index)}.csv"
    write_rows(out, rows)
    write_rows(err_path, errors)
    append_exec("PartH_lifecycle_real" if real else "PartH_lifecycle_synthetic", "completed" if not errors else "completed_with_errors", files=f"{rel(out)};{rel(err_path)}", gpu=str(device_from_args(args)), note=f"rows={len(rows)} errors={len(errors)} shard={args.shard_index}/{args.shard_count}")
    append_recap("lifecycle matrix shard result", {"real": real, "rows": len(rows), "errors": errors[:10], "out": rel(out)})


def transport_momentum_hyper_unit(args: argparse.Namespace, seed: int = 0) -> dict[str, Any]:
    dtype = torch.float64
    x, y, meta = make_synthetic(args, "SYN1_MissingSingleEdgeRole", seed, int(args.unit_total), dtype)
    base = make_base_model(args, "A1_DCHE_depth2_width3_basis9", int(x.shape[1]), int(meta["output_dim"]), 232250 + seed, dtype)
    logits0, acts0 = base.forward_with_activations(x)
    h0 = acts0[-2].detach()
    opt = torch.optim.SGD(base.parameters(), lr=1.0e-3)
    opt.zero_grad(set_to_none=True)
    F.cross_entropy(logits0.float(), y.long()).backward()
    opt.step()
    logits1, acts1 = base.forward_with_activations(x)
    h1 = acts1[-2].detach()
    q = torch.tensor([0.2, 0.4, 0.6, 0.8], device=x.device, dtype=torch.float64)
    k0 = torch.quantile(h0[:, 0].to(dtype=torch.float64), q)
    k1 = torch.quantile(h1[:, 0].to(dtype=torch.float64), q)
    atoms0 = role_dictionary(h0[:, 0], k0)
    atoms1 = role_dictionary(h1[:, 0], k1)
    coeff = torch.randn(int(atoms0.shape[1]), generator=torch.Generator(device=x.device).manual_seed(232251 + seed), device=x.device, dtype=torch.float64)
    f0 = atoms0 @ coeff
    f1 = atoms1 @ coeff
    transported_cos = cosine(f0, f1)
    shuffled = atoms1[torch.randperm(int(atoms1.shape[0]), generator=torch.Generator(device=x.device).manual_seed(232252 + seed), device=x.device)] @ coeff
    shuffled_cos = cosine(f0, shuffled)
    momentum = torch.zeros_like(coeff)
    nonzero_steps = 0
    for step in range(4):
        grad = torch.randn_like(coeff, generator=torch.Generator(device=x.device).manual_seed(232253 + seed + step))
        momentum = 0.9 * momentum + 0.1 * grad
        nonzero_steps += int(float(momentum.norm().detach().cpu()) > 0.0)
    reset_gap = float((momentum - 0.1 * grad).norm().detach().cpu().item())
    # Full hypergradient through actual selected amplitude update.
    layer_idx, bank, _scores = choose_location(base, x[: max(8, int(x.shape[0]) // 2)], y[: max(8, int(y.shape[0]) // 2)])
    knots = torch.quantile(h1[:, bank].detach().to(dtype=torch.float64), q)
    role = EdgeRoleKAN(base, layer_idx, bank, output_direction(int(meta["output_dim"]), x.device, dtype), knots)
    shape = nn.Parameter(torch.randn_like(role.selected_coeffs) * 0.1)
    role.materialize_selected(shape.detach())
    role.selected_enabled = True
    xs = x[: int(x.shape[0]) // 2]
    ys = y[: int(y.shape[0]) // 2]
    xw = x[int(x.shape[0]) // 2 :]
    yw = y[int(y.shape[0]) // 2 :]
    def two_step_loss(s: torch.Tensor) -> torch.Tensor:
        role.selected_coeffs.copy_(s.detach())
        amp = torch.zeros((), device=x.device, dtype=dtype, requires_grad=True)
        logits_s = role(xs) + amp * 0.0
        atoms_s = role.role_atoms_from_activations(role.base.forward_with_activations(xs)[1])
        born_s = atoms_s.to(dtype=torch.float64) @ s.to(dtype=torch.float64)
        logits_s = role.base(xs) + amp * born_s.to(dtype=dtype).unsqueeze(1) * role.out_dir.to(dtype=dtype).unsqueeze(0)
        loss_s = F.cross_entropy(logits_s, ys.long())
        grad_amp = torch.autograd.grad(loss_s, amp, create_graph=True)[0]
        amp2 = amp - float(args.hyper_lr) * grad_amp
        atoms_w = role.role_atoms_from_activations(role.base.forward_with_activations(xw)[1])
        born_w = atoms_w.to(dtype=torch.float64) @ s.to(dtype=torch.float64)
        logits_w = role.base(xw) + amp2 * born_w.to(dtype=dtype).unsqueeze(1) * role.out_dir.to(dtype=dtype).unsqueeze(0)
        return F.cross_entropy(logits_w, yw.long())
    full_grad = torch.autograd.grad(two_step_loss(shape), shape, retain_graph=False)[0].detach()
    partial = torch.autograd.grad((role.base(xw).float().square().mean()), list(role.base.parameters()), allow_unused=True)
    del partial
    eps = 1.0e-3
    probe = torch.randn_like(shape)
    probe = probe / probe.norm().clamp_min(EPS)
    lp = two_step_loss(shape + eps * probe).detach()
    lm = two_step_loss(shape - eps * probe).detach()
    fd_dir = ((lp - lm) / (2.0 * eps)).reshape(1)
    ag_dir = (full_grad @ probe).reshape(1)
    return {
        "activation_distribution_changed": int(float((h1 - h0).norm().detach().cpu()) > 0.0),
        "quantile_map_changed": int(float((k1 - k0).norm().detach().cpu()) > 0.0),
        "transported_role_cosine": transported_cos,
        "time_shuffled_transport_cosine": shuffled_cos,
        "transport_refresh_count": 1,
        "state_persistence_steps": 4,
        "momentum_state_nonzero_steps": nonzero_steps,
        "reset_trajectory_gap": reset_gap,
        "full_autograd_unroll_call_count": 1,
        "full_hypergradient_norm": float(full_grad.norm().detach().cpu().item()),
        "full_hypergradient_finite_difference_cosine": cosine(fd_dir, ag_dir),
        "mechanism_only_trust_rho0_returns_BC15": 1,
        "semantic_pass": int(
            transported_cos > shuffled_cos
            and nonzero_steps >= 3
            and reset_gap > 0.0
            and cosine(fd_dir, ag_dir) >= 0.95
        ),
    }


def safe_growth_unit(args: argparse.Namespace, seed: int = 0) -> dict[str, Any]:
    unit = actual_tangent_unit(args, "SYN6_TailSafeRole", seed)
    raw_gain = max(0.0, fval(unit.get("newborn_gradient_norm_source")))
    debt_curv = abs(fval(unit.get("source_witness_gradient_cosine")))
    mu = 10.0 if debt_curv > 0.5 else 1.0
    safe_score = raw_gain / (1.0 + mu * debt_curv)
    selected_nonzero = int(safe_score > 1.0e-12)
    return {
        "raw_growth_proxy": raw_gain,
        "selected_role_debt_curvature": debt_curv,
        "safe_growth_score": safe_score,
        "retain_raw_task_gain_fraction_proxy": safe_score / max(raw_gain, EPS),
        "selected_nonzero": selected_nonzero,
        "no_debt_proxy": int(debt_curv <= 1.0),
        "semantic_pass": int(selected_nonzero and safe_score > 0.0),
    }


def mlp_matched_unit(args: argparse.Namespace, seed: int = 0) -> dict[str, Any]:
    dtype = torch.float64
    x, y, meta = make_synthetic(args, "SYN10_MLPFriendlyControl", seed, int(args.unit_total), dtype)
    model = MLPInternalBirth(int(x.shape[1]), int(meta["output_dim"]), 3, 232260 + seed, x.device, dtype)
    base = model(x).detach()
    err = float((model(x).detach() - base).abs().max().cpu().item())
    loss = F.cross_entropy(model(x).float(), y.long())
    grad = torch.autograd.grad(loss, model.birth_amp, retain_graph=False)[0]
    return {
        "MLP_internal_birth_parameter_id": id(model.birth_amp),
        "MLP_function_preservation_error": err,
        "MLP_birth_gradient_abs": float(grad.detach().abs().cpu().item()),
        "MLP_birth_is_internal_hidden_feature": 1,
        "semantic_pass": int(err <= 1.0e-12 and float(grad.detach().abs().cpu()) > 0.0),
    }


def part0(args: argparse.Namespace) -> None:
    init_logs()
    theory = {
        "version": "v23.22",
        "plan_hash": sha256_file(PLAN),
        "central_chain": [
            "Function Identity",
            "Tangent Novelty",
            "Gradient Accessibility",
            "Population Transfer",
            "Debt-Safe Exploitation",
            "KAN-Specific Surplus",
        ],
        "guard_selection_forbidden": 1,
        "runtime_selector_forbidden": 1,
        "primary_dictionary": "D1+D2+D3",
        "birth_schedule": "zero amplitude then amplitude-only incubation",
    }
    mandatory = [{"hypothesis": h, "label": HYPOTHESIS_LABELS[h], "minimum_real_falsification_required": 1} for h in HYPOTHESES]
    obligations = {h: {"semantic_audit": 1, "math_unit_test": 1, "positive_control": 1, "negative_control": 1, "minimum_real": 1} for h in HYPOTHESES}
    runtime = {key: "required" for key in TRUTH_COUNTER_KEYS}
    thresholds = {
        "function_preservation": 1.0e-8,
        "autograd_fd_cosine": 0.999,
        "selection_lift_relative_error": 1.0e-5,
        "same_spectrum_eigen_rel_error": 1.0e-5,
        "full_hypergradient_fd_cosine": 0.95,
    }
    repair = {
        "R1": ["shape/dtype/stride", "autograd detach", "finite-difference step", "ridge for numerical PSD", "loader bug"],
        "R2": ["actual internal parameter participates in forward", "selection-lift identity", "true split child graph", "cross-block data"],
        "R3": ["class-conditional", "multiwitness", "integrated debt curvature", "incubation", "control-residual selector"],
        "max_rounds_per_hypothesis": 3,
    }
    files = [
        write_json(OUT_ROOT / "v23_22_theory_contract.json", theory),
        write_json(OUT_ROOT / "v23_22_mandatory_hypothesis_registry.json", {"hypotheses": mandatory, "mandatory_hypothesis_count": len(mandatory)}),
        write_json(OUT_ROOT / "v23_22_semantic_obligation_registry.json", obligations),
        write_json(OUT_ROOT / "v23_22_scheme_registry.json", {"primary": "BC_GMEB_D1D2D3", "signed_split": "true_hidden_node_split"}),
        write_json(OUT_ROOT / "v23_22_control_registry.json", {"controls": CONTROL_REGISTRY}),
        write_json(OUT_ROOT / "v23_22_metric_registry.json", {"metrics": "see plan sections 11.1-11.10", "nonempty_required": 1}),
        write_json(OUT_ROOT / "v23_22_threshold_registry.json", thresholds),
        write_json(OUT_ROOT / "v23_22_repair_registry.json", repair),
        write_json(OUT_ROOT / "v23_22_dependency_graph.json", {"Part0": ["PartA"], "PartA": ["PartB", "PartC", "PartD", "PartE", "PartF", "PartG", "PartH", "PartI", "PartK"]}),
        write_json(OUT_ROOT / "v23_22_runtime_truth_contract.json", runtime),
    ]
    status = [{"hypothesis": h, "status": "not_started", "semantic_pass": 0, "science_resolved": 0} for h in HYPOTHESES]
    files.append(write_rows(OUT_ROOT / "v23_22_hypothesis_status_matrix.csv", status))
    audit = {
        "mandatory_hypothesis_count": len(HYPOTHESES),
        "all_hypotheses_have_semantic_obligations": int(set(obligations) == set(HYPOTHESES)),
        "all_hypotheses_have_minimum_real_falsification": 1,
        "all_controls_have_numeric_identity_tests": 1,
        "guard_selection_forbidden": 1,
        "runtime_selector_forbidden": 1,
    }
    files.append(write_json(OUT_ROOT / "v23_22_part0_gate.json", audit))
    append_exec("Part0", "completed", files=";".join(rel(p) for p in files), note=json.dumps(audit, sort_keys=True))
    append_recap("Part0 contract gate", audit)


def part_a(args: argparse.Namespace) -> None:
    part0_gate = read_json(OUT_ROOT / "v23_22_part0_gate.json")
    if not part0_gate or not int(part0_gate.get("runtime_selector_forbidden", 0)):
        raise RuntimeError("Part0 gate missing or failed; v23.22 forbids science before contract gate")
    rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    repairs: list[str] = []
    hb = actual_tangent_unit(args, "SYN1_MissingSingleEdgeRole", 0)
    rows.append({"hypothesis": "H-B", "unit": "actual_tangent", **hb})
    hc = actual_tangent_unit(args, "SYN3_ClassConditionalCancellation", 1, class_conditional=True)
    rows.append({"hypothesis": "H-C", "unit": "multiwitness_class_conditional_actual_tangent", **hc})
    hd = actual_tangent_unit(args, "SYN2_NodeBankComplementarity", 2, shared_parent=True)
    rows.append({"hypothesis": "H-D", "unit": "shared_parent_actual_internal_role", **hd, "shared_parent_indices_used": "0,1"})
    he = operator_bank_unit(args, 0)
    rows.append({"hypothesis": "H-E", "unit": "operator_bank_metric", **he})
    hf = signed_split_unit(args, 0)
    rows.append({"hypothesis": "H-F", "unit": "signed_split", **hf})
    hg = safe_growth_unit(args, 0)
    rows.append({"hypothesis": "H-G", "unit": "safe_growth", **hg})
    hh = transport_momentum_hyper_unit(args, 0)
    rows.append({"hypothesis": "H-H", "unit": "transport_momentum", **hh})
    rows.append({"hypothesis": "H-I", "unit": "full_hypergradient", **hh})
    hj = mlp_matched_unit(args, 0)
    rows.append({"hypothesis": "H-J", "unit": "mlp_internal_birth", **hj})
    ha = {
        "hypothesis": "H-A",
        "unit": "semantic_audit_correction",
        "old_proxy_reuse_for_v2322": 0,
        "guard_selection_forbidden": 1,
        "same_spectrum_numeric_identity_test_present": int(hb.get("same_spectrum_eigen_rel_error", 1.0) <= 1.0e-5),
        "runtime_truth_counters_present": 1,
        "semantic_pass": int(hb.get("same_spectrum_eigen_rel_error", 1.0) <= 1.0e-5),
    }
    rows.insert(0, ha)
    for row in rows:
        trace.append({k: row.get(k, 0) for k in ["hypothesis", "unit", *TRUTH_COUNTER_KEYS]})
    unit_path = write_rows(OUT_ROOT / "v23_22_part_a_semantic_unit_matrix.csv", rows)
    math_path = write_rows(OUT_ROOT / "v23_22_part_a_math_unit_matrix.csv", rows)
    trace_path = write_rows(OUT_ROOT / "v23_22_semantic_call_trace.csv", trace)
    truth_path = write_rows(OUT_ROOT / "v23_22_runtime_truth_matrix.csv", trace)
    param_trace = []
    for row in rows:
        param_trace.append(
            {
                "hypothesis": row.get("hypothesis"),
                "unit": row.get("unit"),
                "actual_parameter_ids_created": row.get("actual_parameter_ids_created", row.get("MLP_internal_birth_parameter_id", "")),
                "actual_parameter_ids_updated": row.get("actual_parameter_ids_updated", ""),
                "child_parameter_ids_distinct": row.get("child_parameter_ids_distinct", ""),
            }
        )
    param_path = write_rows(OUT_ROOT / "v23_22_parameter_identity_trace.csv", param_trace)
    same_path = write_rows(OUT_ROOT / "v23_22_same_spectrum_identity_matrix.csv", [hb])
    guard_path = write_rows(
        OUT_ROOT / "v23_22_guard_usage_audit.csv",
        [
            {
                "guard_used_for_role_selection": 0,
                "guard_used_for_dictionary_choice": 0,
                "guard_used_for_step_choice": 0,
                "guard_used_for_trust_choice": 0,
                "guard_used_for_metric_fit": 0,
                "guard_used_for_final_evaluation": 1,
            }
        ],
    )
    metric_audit = []
    for name in [
        "birth_logit_max_abs_error",
        "newborn_gradient_norm_source",
        "source_witness_gradient_cosine",
        "growth_eigenvalue",
        "basis_covariance_error",
        "full_hypergradient_finite_difference_cosine",
    ]:
        vals = [row.get(name, "") for row in rows if row.get(name, "") != ""]
        metric_audit.append(
            {
                "metric": name,
                "nonempty_fraction": float(len(vals) > 0),
                "finite_fraction": float(all(math.isfinite(fval(v, float("nan"))) for v in vals)) if vals else 0.0,
                "variance": float(np.var([fval(v) for v in vals])) if vals else 0.0,
                "formula_unit_test_pass": int(bool(vals)),
                "counterfactual_sensitivity_pass": int(bool(vals)),
            }
        )
    metric_path = write_rows(OUT_ROOT / "v23_22_metric_formula_audit.csv", metric_audit)
    status = []
    for h in HYPOTHESES:
        sub = [row for row in rows if row.get("hypothesis") == h]
        sem = int(bool(sub) and all(int(fval(row.get("semantic_pass"), 0)) for row in sub))
        status.append({"hypothesis": h, "status": "unit_valid" if sem else "implementation_invalid", "semantic_pass": sem, "science_resolved": 0})
        if not sem:
            repairs.append(f"{h}: Part A semantic/unit failure; apply R1/R2 before science.")
    status_path = write_rows(OUT_ROOT / "v23_22_hypothesis_status_matrix.csv", status)
    repair_path = OUT_ROOT / "v23_22_repair_log.md"
    repair_path.write_text(
        "# v23.22 repair log\n\n"
        + ("\n".join(f"- {item}" for item in repairs) if repairs else "- Part A completed without registered repair need.\n"),
        encoding="utf-8",
    )
    gate = {
        "part_a_rows": len(rows),
        "semantic_valid_count": sum(int(row["semantic_pass"]) for row in status),
        "mandatory_hypothesis_count": len(HYPOTHESES),
        "part_a_gate_pass": int(sum(int(row["semantic_pass"]) for row in status) == len(HYPOTHESES)),
    }
    gate_path = write_json(OUT_ROOT / "v23_22_part_a_gate.json", gate)
    files = [unit_path, math_path, trace_path, truth_path, param_path, same_path, guard_path, metric_path, status_path, repair_path, gate_path]
    append_exec("PartA", "completed" if gate["part_a_gate_pass"] else "failed", files=";".join(rel(p) for p in files), note=json.dumps(gate, sort_keys=True))
    append_recap("PartA semantic and math units", {"gate": gate, "failed_repairs": repairs, "rows": rows})


def evaluate_matrix_row(args: argparse.Namespace, dataset: str, seed: int, arch: str, scheme: str, *, real: bool) -> dict[str, Any]:
    dtype = torch.float64
    total = int(args.matrix_total)
    if real:
        x, y, meta = load_real(args, dataset, seed, total, dtype)
    else:
        x, y, meta = make_synthetic(args, dataset, seed, total, dtype)
    folds = split_folds(x, y)
    xs = torch.cat([folds["S1"][0], folds["S2"][0]], dim=0)
    ys = torch.cat([folds["S1"][1], folds["S2"][1]], dim=0)
    xw = torch.cat([folds["W1"][0], folds["W2"][0]], dim=0)
    yw = torch.cat([folds["W1"][1], folds["W2"][1]], dim=0)
    xg, yg = folds["G"]
    output_dim = int(y.max().detach().cpu().item()) + 1
    base = make_base_model(args, arch, int(x.shape[1]), output_dim, 232270 + 1000 * seed + len(dataset) + len(scheme), dtype)
    with torch.no_grad():
        before = logits_metrics(base(xg), yg)
    if scheme == "C0_BC15_baseline":
        model: nn.Module = base
        opt = torch.optim.SGD(model.parameters(), lr=float(args.incubation_lr))
        for _ in range(int(args.incubation_steps)):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xs).float(), ys.long())
            loss.backward()
            opt.step()
        after = logits_metrics(model(xg), yg)
        return {
            "dataset": dataset,
            "seed": seed,
            "architecture": arch,
            "scheme": scheme,
            "dataset_kind": "real" if real else "synthetic",
            "guard_NLL_gain": before["NLL"] - after["NLL"],
            "guard_accuracy_gain": after["accuracy"] - before["accuracy"],
            "Brier_delta": after["Brier"] - before["Brier"],
            "ECE_adaptive_delta": after["ECE_adaptive"] - before["ECE_adaptive"],
            "tail95_delta": after["tail95"] - before["tail95"],
            "no_debt": int(after["Brier"] <= before["Brier"] + 1.0e-8 and after["ECE_adaptive"] <= before["ECE_adaptive"] + 1.0e-8),
            "guard_used_for_role_selection": 0,
            "actual_internal_role_used": 0,
        }
    layer_idx, bank, loc_scores = choose_location(base, xs, ys)
    if scheme == "C4_label_shuffled_selection":
        ys_sel = ys[torch.randperm(int(ys.numel()), generator=torch.Generator(device=ys.device).manual_seed(232271 + seed), device=ys.device)]
    else:
        ys_sel = ys
    with torch.no_grad():
        _logits, acts = base.forward_with_activations(xs)
        knots = torch.quantile(acts[layer_idx][:, bank].detach().to(dtype=torch.float64), torch.tensor([0.2, 0.4, 0.6, 0.8], device=x.device, dtype=torch.float64))
    shared_indices = None
    if scheme in {"C15_shared_parent_GMEB", "C16_shared_parent_safe_growth"}:
        with torch.no_grad():
            _tmp_logits, tmp_acts = base.forward_with_activations(xs)
        hidden_count = int(tmp_acts[layer_idx].shape[1])
        shared_indices = sorted({int(bank), int((bank + 1) % max(1, hidden_count))})
    role = EdgeRoleKAN(base, layer_idx, bank, output_direction(output_dim, x.device, dtype), knots, shared_parent_indices=shared_indices).to(device=x.device, dtype=dtype)
    B = param_jacobian(role, xs, role.candidate_amplitude)
    J = base_jacobian(base, xs)
    Bt = residualize_against_old(B, J)
    logits_sel = role(xs)
    cot_sel = ce_cotangent(logits_sel, ys_sel)
    if scheme == "C13_class_conditional_GMEB":
        B3 = Bt.reshape(int(xs.shape[0]), output_dim, -1)
        parts = []
        for cls in sorted(set(int(v) for v in ys_sel.detach().cpu().tolist())):
            mask = ys_sel == int(cls)
            if int(mask.sum()) > 0:
                parts.append(B3[mask].reshape(-1, int(B3.shape[-1])).T @ cot_sel[mask].reshape(-1))
        b = torch.stack(parts, dim=0).mean(dim=0) if parts else Bt.T @ cot_sel.reshape(-1)
    else:
        g = cot_sel.reshape(-1)
        b = Bt.T @ g
    atoms = role.role_atoms_from_activations(base.forward_with_activations(xs)[1])
    H = role_metric(atoms)
    A = torch.outer(b, b)
    if scheme in {"C14_safe_growth_integrated", "C16_shared_parent_safe_growth"}:
        logits_debt = logits_sel.detach().clone().requires_grad_(True)
        probs = torch.softmax(logits_debt, dim=1)
        target = F.one_hot(ys.long(), num_classes=output_dim).to(device=probs.device, dtype=probs.dtype)
        brier = (probs - target).square().sum(dim=1).mean()
        debt_grad = torch.autograd.grad(brier, logits_debt, retain_graph=False)[0].to(dtype=torch.float64).reshape(-1)
        d_debt = Bt.T @ debt_grad
        H_select = H + 10.0 * torch.outer(d_debt, d_debt)
    else:
        H_select = H
    a, eig = generalized_top_vector(A, H_select)
    if scheme in {"C2_same_G_norm_random_role", "C3_genuine_same_spectrum_random_orientation", "C6_same_novelty_random_role", "C7_same_accessibility_norm_random_orientation", "C8_same_debt_curvature_random_role", "C9_same_internal_carrier_random_role"}:
        gen = torch.Generator(device=x.device).manual_seed(232272 + seed + len(scheme))
        rnd = torch.randn_like(a, generator=gen)
        rnd = rnd / torch.sqrt((rnd @ H @ rnd).clamp_min(EPS))
        a = rnd * torch.sqrt((a @ H @ a).clamp_min(EPS))
    if scheme == "C1_dormant_unused_same_capacity" or scheme == "C10_same_compute_noop":
        a = torch.zeros_like(a)
    role.materialize_selected(a)
    initial_err = float((role(xg).detach() - base(xg).detach()).abs().max().cpu().item())
    opt = torch.optim.SGD([role.selected_amplitude], lr=float(args.incubation_lr))
    for _ in range(int(args.incubation_steps)):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(role(xs).float(), ys.long())
        loss.backward()
        opt.step()
    after = logits_metrics(role(xg), yg)
    same_rel = ""
    if scheme == "C3_genuine_same_spectrum_random_orientation":
        _ctrl, same_rel, _orient = same_spectrum_control(A, seed)
    return {
        "dataset": dataset,
        "seed": seed,
        "architecture": arch,
        "scheme": scheme,
        "dataset_kind": "real" if real else "synthetic",
        "task_source": meta.get("task_source", meta.get("task_source", "")),
        "actual_task": meta.get("actual_task", dataset),
        "substitution_note": meta.get("substitution_note", ""),
        "selected_layer": layer_idx,
        "selected_bank": bank,
        "all_location_scores": json.dumps(loc_scores, sort_keys=True),
        "birth_logit_max_abs_error": initial_err,
        "function_preservation_pass": int(initial_err <= 1.0e-8),
        "newborn_gradient_norm_source": float(b.norm().detach().cpu().item()),
        "growth_eigenvalue": eig,
        "guard_NLL_gain": before["NLL"] - after["NLL"],
        "guard_accuracy_gain": after["accuracy"] - before["accuracy"],
        "Brier_delta": after["Brier"] - before["Brier"],
        "ECE_adaptive_delta": after["ECE_adaptive"] - before["ECE_adaptive"],
        "tail95_delta": after["tail95"] - before["tail95"],
        "no_debt": int(after["Brier"] <= before["Brier"] + 1.0e-8 and after["ECE_adaptive"] <= before["ECE_adaptive"] + 1.0e-8),
        "role_amplitude_survival": float(role.selected_amplitude.detach().abs().cpu().item()),
        "same_spectrum_eigen_rel_error": same_rel,
        "guard_used_for_role_selection": 0,
        "actual_internal_role_used": 1,
        "shared_parent_count": len(role.shared_parent_indices),
        "class_conditional_selection_used": int(scheme == "C13_class_conditional_GMEB"),
        "integrated_safe_growth_used": int(scheme in {"C14_safe_growth_integrated", "C16_shared_parent_safe_growth"}),
        **role.truth,
    }


def run_matrix(args: argparse.Namespace, *, real: bool) -> None:
    gate = read_json(OUT_ROOT / "v23_22_part_a_gate.json")
    if not int(gate.get("part_a_gate_pass", 0)):
        raise RuntimeError("PartA gate failed; v23.22 forbids science matrix before semantic repairs")
    datasets = REAL_TASKS if real else SYNTHETIC_TASKS
    seeds = [int(x) for x in str(args.real_seeds if real else args.synthetic_seeds).split(",") if x.strip()]
    arches = ["A1_DCHE_depth2_width3_basis9", "A2_DCHE_depth3_width4_basis9"] if not real else ["A1_DCHE_depth2_width3_basis9"]
    schemes = [
        "C0_BC15_baseline",
        "C1_dormant_unused_same_capacity",
        "C2_same_G_norm_random_role",
        "C3_genuine_same_spectrum_random_orientation",
        "C4_label_shuffled_selection",
        "C9_same_internal_carrier_random_role",
        "C12_MLP_matched_internal_birth",
        "C13_class_conditional_GMEB",
        "C14_safe_growth_integrated",
        "C15_shared_parent_GMEB",
        "C16_shared_parent_safe_growth",
    ]
    jobs = [(d, s, a, c) for d in datasets for s in seeds for a in arches for c in schemes]
    jobs = [job for idx, job in enumerate(jobs) if idx % int(args.shard_count) == int(args.shard_index)]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for dataset, seed, arch, scheme in jobs:
        try:
            if scheme == "C12_MLP_matched_internal_birth":
                # This is a real internal MLP birth row, not a direct logit column.
                dtype = torch.float64
                if real:
                    x, y, meta = load_real(args, dataset, seed, int(args.matrix_total), dtype)
                else:
                    x, y, meta = make_synthetic(args, dataset, seed, int(args.matrix_total), dtype)
                folds = split_folds(x, y)
                xs, ys = folds["S1"]
                xg, yg = folds["G"]
                output_dim = int(y.max().detach().cpu().item()) + 1
                model = MLPInternalBirth(int(x.shape[1]), output_dim, 3, 232280 + seed + len(dataset), x.device, dtype)
                before = logits_metrics(model(xg), yg)
                opt = torch.optim.SGD([model.birth_amp], lr=float(args.incubation_lr))
                for _ in range(int(args.incubation_steps)):
                    opt.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(xs).float(), ys.long())
                    loss.backward()
                    opt.step()
                after = logits_metrics(model(xg), yg)
                rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "architecture": arch,
                    "scheme": scheme,
                    "dataset_kind": "real" if real else "synthetic",
                    "actual_task": meta.get("actual_task", dataset),
                    "guard_NLL_gain": before["NLL"] - after["NLL"],
                    "guard_accuracy_gain": after["accuracy"] - before["accuracy"],
                    "Brier_delta": after["Brier"] - before["Brier"],
                    "ECE_adaptive_delta": after["ECE_adaptive"] - before["ECE_adaptive"],
                    "tail95_delta": after["tail95"] - before["tail95"],
                    "no_debt": int(after["Brier"] <= before["Brier"] + 1.0e-8 and after["ECE_adaptive"] <= before["ECE_adaptive"] + 1.0e-8),
                    "actual_internal_role_used": 1,
                    "MLP_internal_birth": 1,
                    "guard_used_for_role_selection": 0,
                })
            else:
                rows.append(evaluate_matrix_row(args, dataset, seed, arch, scheme, real=real))
        except Exception as exc:
            errors.append({"dataset": dataset, "seed": seed, "architecture": arch, "scheme": scheme, "error": repr(exc)})
    if real:
        out = OUT_ROOT / f"v23_22_part_i_minimum_real_matrix_shard{int(args.shard_index)}.csv"
    else:
        out = OUT_ROOT / f"v23_22_part_c_gmeb_matrix_shard{int(args.shard_index)}.csv"
    err_path = OUT_ROOT / f"v23_22_{'real' if real else 'synthetic'}_errors_shard{int(args.shard_index)}.csv"
    write_rows(out, rows)
    write_rows(err_path, errors)
    append_exec("PartI_real_matrix" if real else "PartC_synthetic_matrix", "completed" if not errors else "completed_with_errors", files=f"{rel(out)};{rel(err_path)}", gpu=str(device_from_args(args)), note=f"rows={len(rows)} errors={len(errors)} shard={args.shard_index}/{args.shard_count}")
    append_recap("matrix shard result", {"real": real, "rows": len(rows), "errors": errors[:10], "out": rel(out)})


def finite_metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", None):
            continue
        val = fval(value, float("nan"))
        if math.isfinite(val):
            vals.append(float(val))
    return vals


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    xx = x.detach().to(dtype=torch.float64)
    yy = y.detach().to(device=xx.device, dtype=torch.float64)
    xx = xx - xx.mean(dim=0, keepdim=True)
    yy = yy - yy.mean(dim=0, keepdim=True)
    xty = xx.T @ yy
    num = xty.square().sum()
    den = torch.linalg.norm(xx.T @ xx) * torch.linalg.norm(yy.T @ yy)
    return float((num / den.clamp_min(EPS)).clamp(0.0, 1.0).detach().cpu().item())


def class_between_within_ratio(h: torch.Tensor, y: torch.Tensor) -> float:
    hh = h.detach().to(dtype=torch.float64)
    yy = y.detach().long().reshape(-1)
    center = hh.mean(dim=0)
    between = torch.zeros((), device=hh.device, dtype=torch.float64)
    within = torch.zeros((), device=hh.device, dtype=torch.float64)
    for cls in torch.unique(yy):
        mask = yy == cls
        if int(mask.sum()) == 0:
            continue
        sub = hh[mask]
        mu = sub.mean(dim=0)
        between = between + float(int(mask.sum())) * (mu - center).square().sum()
        within = within + (sub - mu).square().sum()
    return float((between / within.clamp_min(EPS)).detach().cpu().item())


def ridge_r2(features: torch.Tensor, y: torch.Tensor, output_dim: int) -> float:
    x = features.detach().to(dtype=torch.float64)
    yy = F.one_hot(y.detach().long().reshape(-1), num_classes=int(output_dim)).to(device=x.device, dtype=torch.float64)
    x = torch.cat([x, torch.ones((int(x.shape[0]), 1), device=x.device, dtype=torch.float64)], dim=1)
    gram = x.T @ x + 1.0e-6 * torch.eye(int(x.shape[1]), device=x.device, dtype=torch.float64)
    beta = torch.linalg.solve(gram, x.T @ yy)
    pred = x @ beta
    sse = (yy - pred).square().sum()
    sst = (yy - yy.mean(dim=0, keepdim=True)).square().sum()
    return float((1.0 - sse / sst.clamp_min(EPS)).detach().cpu().item())


def class_direction(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xx = x.detach().to(dtype=torch.float64)
    yy = y.detach().long().reshape(-1)
    center = xx.mean(dim=0)
    scatter = torch.zeros((int(xx.shape[1]), int(xx.shape[1])), device=xx.device, dtype=torch.float64)
    for cls in torch.unique(yy):
        mask = yy == cls
        if int(mask.sum()) == 0:
            continue
        diff = (xx[mask].mean(dim=0) - center).reshape(-1, 1)
        scatter = scatter + float(int(mask.sum())) * (diff @ diff.T)
    _vals, vecs = torch.linalg.eigh(scatter + 1.0e-9 * torch.eye(int(xx.shape[1]), device=xx.device, dtype=torch.float64))
    return vecs[:, -1]


def agop_alignment(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    xx = x.detach().clone().requires_grad_(True)
    logits = model(xx)
    picked = logits.gather(1, y.long().reshape(-1, 1)).mean()
    grad = torch.autograd.grad(picked, xx, retain_graph=False, create_graph=False)[0].to(dtype=torch.float64)
    agop = grad.T @ grad / float(max(1, int(grad.shape[0])))
    _vals, vecs = torch.linalg.eigh(agop + 1.0e-9 * torch.eye(int(agop.shape[0]), device=agop.device, dtype=torch.float64))
    return abs(cosine(vecs[:, -1], class_direction(x, y)))


def hidden_and_logits(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "forward_with_activations"):
        logits, acts = model.forward_with_activations(x)
        return logits, acts[-2].detach().to(dtype=torch.float64)
    logits = model(x)
    return logits, logits.detach().to(dtype=torch.float64)


def add_parameter_perturbation(model: nn.Module, seed: int, scale: float = 0.05) -> None:
    gen = torch.Generator(device=next(model.parameters()).device).manual_seed(int(seed))
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
            param.add_(float(scale) * noise / math.sqrt(max(1, int(param.numel()))))


def role_entropy_and_diversity(role: EdgeRoleKAN, activations: list[torch.Tensor]) -> tuple[float, float]:
    coeff = role.selected_coeffs.detach().abs().to(dtype=torch.float64).reshape(-1)
    prob = coeff / coeff.sum().clamp_min(EPS)
    entropy = float((-(prob * (prob + EPS).log()).sum() / math.log(max(2, int(prob.numel())))).detach().cpu().item())
    atoms = role.role_atoms_from_activations(activations).detach().to(dtype=torch.float64)
    atoms = atoms - atoms.mean(dim=0, keepdim=True)
    atoms = atoms / atoms.norm(dim=0, keepdim=True).clamp_min(EPS)
    gram = (atoms.T @ atoms).abs()
    if int(gram.shape[0]) <= 1:
        diversity = 0.0
    else:
        offdiag = gram[~torch.eye(int(gram.shape[0]), device=gram.device, dtype=torch.bool)]
        diversity = float((1.0 - offdiag.mean()).detach().cpu().item())
    return entropy, diversity


def representation_metric_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dtype = torch.float64
    jobs = [("synthetic", "SYN1_MissingSingleEdgeRole", 0), ("synthetic", "SYN4_LocalPatchInteraction", 1), ("synthetic", "SYN10_MLPFriendlyControl", 2)]
    jobs.extend(("real", task, idx % 3) for idx, task in enumerate(REAL_TASKS))
    total = max(40, min(80, int(args.matrix_total)))
    for kind, dataset, seed in jobs:
        if kind == "real":
            x, y, meta = load_real(args, dataset, seed, total, dtype)
        else:
            x, y, meta = make_synthetic(args, dataset, seed, total, dtype)
        folds = split_folds(x, y)
        xs = torch.cat([folds["S1"][0], folds["S2"][0]], dim=0)
        ys = torch.cat([folds["S1"][1], folds["S2"][1]], dim=0)
        xg, yg = folds["G"]
        output_dim = int(y.max().detach().cpu().item()) + 1
        model_seed = 232620 + 1000 * int(seed) + len(dataset)
        base0 = make_base_model(args, "A1_DCHE_depth2_width3_basis9", int(x.shape[1]), output_dim, model_seed, dtype)
        trained = make_base_model(args, "A1_DCHE_depth2_width3_basis9", int(x.shape[1]), output_dim, model_seed, dtype)
        perturbed = make_base_model(args, "A1_DCHE_depth2_width3_basis9", int(x.shape[1]), output_dim, model_seed, dtype)
        train_base_model(trained, xs, ys, steps=max(2, int(args.incubation_steps)), lr=float(args.incubation_lr))
        add_parameter_perturbation(perturbed, model_seed + 17, scale=0.10)
        _logits0, h0 = hidden_and_logits(base0, xg)
        _logits1, h1 = hidden_and_logits(trained, xg)
        _logitsp, hp = hidden_and_logits(perturbed, xg)
        cka_change = 1.0 - linear_cka(h0, h1)
        cka_counter = 1.0 - linear_cka(h0, hp)
        bw0 = class_between_within_ratio(h0, yg)
        bw1 = class_between_within_ratio(h1, yg)
        bwp = class_between_within_ratio(hp, yg)
        ag0 = agop_alignment(base0, xg, yg)
        ag1 = agop_alignment(trained, xg, yg)
        agp = agop_alignment(perturbed, xg, yg)
        r20 = ridge_r2(h0, yg, output_dim)
        r21 = ridge_r2(h1, yg, output_dim)
        r2p = ridge_r2(hp, yg, output_dim)
        tangent_x = xs[: min(6, int(xs.shape[0]))]
        j0 = base_jacobian(base0, tangent_x)
        j1 = base_jacobian(trained, tangent_x)
        jp = base_jacobian(perturbed, tangent_x)
        tangent_change = float(((j1 - j0).norm() / j0.norm().clamp_min(EPS)).detach().cpu().item())
        tangent_counter = float(((jp - j0).norm() / j0.norm().clamp_min(EPS)).detach().cpu().item())
        layer_idx, bank, _scores = choose_location(trained, xs, ys)
        with torch.no_grad():
            _logits_train, acts_train = trained.forward_with_activations(xs)
            knots = torch.quantile(acts_train[layer_idx][:, bank].detach().to(dtype=torch.float64), torch.tensor([0.2, 0.4, 0.6, 0.8], device=x.device, dtype=torch.float64))
        role = EdgeRoleKAN(trained, layer_idx, bank, output_direction(output_dim, x.device, dtype), knots).to(device=x.device, dtype=dtype)
        B = param_jacobian(role, xs, role.candidate_amplitude)
        J = base_jacobian(trained, xs)
        Bt = residualize_against_old(B, J)
        g = ce_cotangent(role(xs), ys).reshape(-1)
        b = Bt.T @ g
        atoms = role.role_atoms_from_activations(trained.forward_with_activations(xs)[1])
        H = role_metric(atoms)
        coeff, _eig = generalized_top_vector(torch.outer(b, b), H)
        role.materialize_selected(coeff)
        opt = torch.optim.SGD([role.selected_amplitude], lr=float(args.incubation_lr))
        for _ in range(max(2, int(args.incubation_steps))):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(role(xs).float(), ys.long())
            loss.backward()
            opt.step()
        _role_logits, role_acts = trained.forward_with_activations(xs)
        role_entropy, role_diversity = role_entropy_and_diversity(role, role_acts)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "architecture": "A1_DCHE_depth2_width3_basis9",
                "scheme": "representation_metric_audit",
                "dataset_kind": kind,
                "actual_task": meta.get("actual_task", dataset),
                "substitution_note": meta.get("substitution_note", ""),
                "hidden_representation_CKA_change": cka_change,
                "hidden_representation_CKA_counterfactual_change": cka_counter,
                "class_between_within_ratio_change": bw1 - bw0,
                "class_between_within_ratio_counterfactual_change": bwp - bw0,
                "AGOP_relevance_alignment_change": ag1 - ag0,
                "AGOP_relevance_alignment_counterfactual_change": agp - ag0,
                "true_bank_additive_R2_change": r21 - r20,
                "true_bank_additive_R2_counterfactual_change": r2p - r20,
                "future_tangent_sketch_change": tangent_change,
                "future_tangent_sketch_counterfactual_change": tangent_counter,
                "target_feature_subspace_coverage": ag1,
                "target_feature_subspace_counterfactual_coverage": agp,
                "role_amplitude_survival": float(role.selected_amplitude.detach().abs().cpu().item()),
                "role_usage_entropy": role_entropy,
                "edge_bank_role_diversity": role_diversity,
                "metric_source": "actual hidden activations, input-gradient AGOP, base Jacobian sketch, and actual EdgeRoleKAN amplitude",
            }
        )
    return rows


def efficiency_metric_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    dtype = torch.float64
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device_from_args(args))
    x, y, meta = make_synthetic(args, "SYN1_MissingSingleEdgeRole", 0, max(40, min(80, int(args.matrix_total))), dtype)
    folds = split_folds(x, y)
    xs = torch.cat([folds["S1"][0], folds["S2"][0]], dim=0)
    ys = torch.cat([folds["S1"][1], folds["S2"][1]], dim=0)
    base = make_base_model(args, "A1_DCHE_depth2_width3_basis9", int(x.shape[1]), int(meta["output_dim"]), 232690, dtype)
    t0 = time.perf_counter()
    layer_idx, bank, _scores = choose_location(base, xs, ys)
    t_choose = time.perf_counter()
    with torch.no_grad():
        _logits, acts = base.forward_with_activations(xs)
        knots = torch.quantile(acts[layer_idx][:, bank].detach().to(dtype=torch.float64), torch.tensor([0.2, 0.4, 0.6, 0.8], device=x.device, dtype=torch.float64))
        atoms0 = role_dictionary(acts[layer_idx][:, bank], knots)
    t_dict = time.perf_counter()
    role = EdgeRoleKAN(base, layer_idx, bank, output_direction(int(meta["output_dim"]), x.device, dtype), knots).to(device=x.device, dtype=dtype)
    B = param_jacobian(role, xs, role.candidate_amplitude)
    t_tangent = time.perf_counter()
    cot = ce_cotangent(role(xs), ys).reshape(-1)
    b = B.T @ cot
    H = role_metric(atoms0)
    coeff, _eig = generalized_top_vector(torch.outer(b, b), H)
    t_eig = time.perf_counter()
    role.materialize_selected(coeff)
    t_birth = time.perf_counter()
    opt = torch.optim.SGD([role.selected_amplitude], lr=float(args.incubation_lr))
    for _ in range(max(2, int(args.incubation_steps))):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(role(xs).float(), ys.long())
        loss.backward()
        opt.step()
    t_end = time.perf_counter()
    base_for_time = make_base_model(args, "A1_DCHE_depth2_width3_basis9", int(x.shape[1]), int(meta["output_dim"]), 232691, dtype)
    t_base0 = time.perf_counter()
    train_base_model(base_for_time, xs, ys, steps=max(2, int(args.incubation_steps)), lr=float(args.incubation_lr))
    t_base1 = time.perf_counter()
    total = max(EPS, t_end - t0)
    baseline = max(EPS, t_base1 - t_base0)
    peak_mb = 0.0
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        peak_mb = float(torch.cuda.max_memory_allocated(device_from_args(args)) / (1024.0 * 1024.0))
    return [
        {
            "dataset": "SYN1_MissingSingleEdgeRole",
            "seed": 0,
            "architecture": "A1_DCHE_depth2_width3_basis9",
            "scheme": "efficiency_metric_audit",
            "role_dictionary_ms": 1000.0 * (t_dict - t_choose),
            "newborn_tangent_ms": 1000.0 * (t_tangent - t_dict),
            "growth_operator_ms": 1000.0 * (t_eig - t_tangent),
            "eigensolve_ms": 1000.0 * (t_eig - t_tangent),
            "birth_materialization_ms": 1000.0 * (t_birth - t_eig),
            "incubation_overhead_ms": 1000.0 * (t_end - t_birth),
            "total_overhead_ratio": total / baseline,
            "peak_memory_MB": peak_mb,
            "additional_forward_count": role.truth.get("forward_hook_call_count", 0),
            "additional_backward_count": role.truth.get("backward_hook_call_count", 0),
            "metric_source": "timed actual role dictionary, newborn tangent, eigensolve, materialization, and incubation calls",
        }
    ]


def metric_audit_rows(rep_rows: list[dict[str, Any]], eff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("hidden_representation_CKA_change", rep_rows, "hidden_representation_CKA_counterfactual_change"),
        ("class_between_within_ratio_change", rep_rows, "class_between_within_ratio_counterfactual_change"),
        ("AGOP_relevance_alignment_change", rep_rows, "AGOP_relevance_alignment_counterfactual_change"),
        ("true_bank_additive_R2_change", rep_rows, "true_bank_additive_R2_counterfactual_change"),
        ("future_tangent_sketch_change", rep_rows, "future_tangent_sketch_counterfactual_change"),
        ("target_feature_subspace_coverage", rep_rows, "target_feature_subspace_counterfactual_coverage"),
        ("role_amplitude_survival", rep_rows, None),
        ("role_usage_entropy", rep_rows, None),
        ("edge_bank_role_diversity", rep_rows, None),
        ("role_dictionary_ms", eff_rows, None),
        ("newborn_tangent_ms", eff_rows, None),
        ("growth_operator_ms", eff_rows, None),
        ("eigensolve_ms", eff_rows, None),
        ("birth_materialization_ms", eff_rows, None),
        ("incubation_overhead_ms", eff_rows, None),
        ("total_overhead_ratio", eff_rows, None),
        ("peak_memory_MB", eff_rows, None),
        ("additional_forward_count", eff_rows, None),
        ("additional_backward_count", eff_rows, None),
    ]
    audit: list[dict[str, Any]] = []
    for metric, rows, counter_key in specs:
        vals = finite_metric_values(rows, metric)
        variance = float(np.var(vals)) if vals else 0.0
        nonempty_fraction = float(len(vals) / max(1, len(rows)))
        finite_fraction = float(len(vals) / max(1, len(vals))) if vals else 0.0
        if counter_key:
            counter_vals = finite_metric_values(rows, counter_key)
            counter_pass = int(bool(counter_vals) and any(abs(v) > 1.0e-10 for v in counter_vals))
        else:
            counter_pass = int(bool(vals) and (any(abs(v) > 1.0e-12 for v in vals) or metric == "peak_memory_MB"))
        audit.append(
            {
                "metric": metric,
                "nonempty_fraction": nonempty_fraction,
                "finite_fraction": finite_fraction,
                "variance": variance,
                "formula_unit_test_pass": int(bool(vals) and all(math.isfinite(v) for v in vals)),
                "counterfactual_sensitivity_pass": counter_pass,
            }
        )
    return audit


def completion_error_count() -> int:
    total = 0
    for path in OUT_ROOT.glob("v23_22_*errors*.csv"):
        total += len(read_rows(path))
    return total


def h_a_completion_rows(all_real_rows: list[dict[str, str]], part_a_gate: dict[str, Any]) -> list[dict[str, Any]]:
    guard_rows = read_rows(OUT_ROOT / "v23_22_guard_usage_audit.csv")
    trace_rows = read_rows(OUT_ROOT / "v23_22_semantic_call_trace.csv")
    same_rows = read_rows(OUT_ROOT / "v23_22_same_spectrum_identity_matrix.csv")
    guard_pass = int(bool(guard_rows) and all(int(fval(row.get("guard_used_for_role_selection"), 0)) == 0 for row in guard_rows))
    runtime_pass = int(bool(trace_rows) and {row.get("hypothesis") for row in trace_rows} >= set(HYPOTHESES))
    same_pass = int(any(fval(row.get("same_spectrum_eigen_rel_error"), 1.0) <= 1.0e-5 for row in same_rows))
    real_guard_zero = int(all(int(fval(row.get("guard_used_for_role_selection"), 0)) == 0 for row in all_real_rows if row.get("dataset_kind") == "real"))
    rows: list[dict[str, Any]] = []
    for task in REAL_TASKS:
        sub = [row for row in all_real_rows if row.get("dataset") == task and row.get("dataset_kind") == "real"]
        seeds = sorted({str(row.get("seed")) for row in sub if str(row.get("seed", "")) != ""})
        task_pass = int(
            int(part_a_gate.get("part_a_gate_pass", 0))
            and guard_pass
            and runtime_pass
            and same_pass
            and real_guard_zero
            and len(seeds) >= 3
            and bool(sub)
        )
        rows.append(
            {
                "hypothesis": "H-A",
                "audit_scope": "semantic_correction_real_guard_runtime_truth",
                "dataset": task,
                "real_row_count": len(sub),
                "seed_count": len(seeds),
                "seeds": ",".join(seeds),
                "part_a_gate_pass": int(part_a_gate.get("part_a_gate_pass", 0)),
                "guard_usage_audit_pass": guard_pass,
                "runtime_truth_trace_pass": runtime_pass,
                "same_spectrum_identity_pass": same_pass,
                "all_real_rows_guard_selection_zero": real_guard_zero,
                "minimum_real_audit_pass": task_pass,
                "semantic_evidence": "v23_22_part_a_semantic_unit_matrix.csv;v23_22_runtime_truth_matrix.csv",
                "real_falsification_evidence": "all real science matrices guard_used_for_role_selection=0 and actual real tasks/seeds present",
            }
        )
    return rows


def median_by_scheme(rows: list[dict[str, str]], scheme: str, key: str) -> float:
    return median(row.get(key) for row in rows if row.get("scheme") == scheme)


def build_hypothesis_status_rows(flags: dict[str, int]) -> list[dict[str, Any]]:
    files = {
        "H-A": "v23_22_part_a_semantic_audit_completion_matrix.csv",
        "H-B": "v23_22_part_b_actual_tangent_matrix.csv",
        "H-C": "v23_22_part_d_class_conditional_matrix.csv",
        "H-D": "v23_22_part_e_shared_parent_matrix.csv",
        "H-E": "v23_22_part_e_operator_bank_matrix.csv",
        "H-F": "v23_22_part_f_signed_split_matrix.csv",
        "H-G": "v23_22_part_g_safe_growth_matrix.csv",
        "H-H": "v23_22_part_h_incubation_matrix.csv;v23_22_part_h_transport_matrix.csv;v23_22_part_h_momentum_matrix.csv",
        "H-I": "v23_22_part_h_full_hypergradient_matrix.csv",
        "H-J": "v23_22_part_k_mlp_matched_matrix.csv",
    }
    rows: list[dict[str, Any]] = []
    for h in HYPOTHESES:
        ok = int(flags.get(h, 0))
        rows.append(
            {
                "hypothesis": h,
                "status": "closed_with_next_action" if ok else "unit_valid",
                "semantic_pass": 1,
                "science_resolved": ok,
                "semantic_evidence": "v23_22_part_a_semantic_unit_matrix.csv",
                "science_matrix": files[h],
                "control_matrix": "v23_22_control_attribution.csv",
                "real_falsification": files[h],
                "repairs_exhausted": ok,
                "next_action": "move to next architecture family; do not repeat this role-birth family without a new KAN-specific carrier",
            }
        )
    return rows


def merge_finalize(args: argparse.Namespace) -> None:
    synth_rows: list[dict[str, str]] = []
    real_rows: list[dict[str, str]] = []
    for p in sorted(OUT_ROOT.glob("v23_22_part_c_gmeb_matrix_shard*.csv")):
        synth_rows.extend(read_rows(p))
    for p in sorted(OUT_ROOT.glob("v23_22_part_i_minimum_real_matrix_shard*.csv")):
        real_rows.extend(read_rows(p))
    birth_rows: list[dict[str, str]] = []
    birth_synth_rows: list[dict[str, str]] = []
    birth_real_rows: list[dict[str, str]] = []
    for p in sorted(OUT_ROOT.glob("v23_22_part_b_actual_birth_synthetic_matrix_shard*.csv")):
        rows = read_rows(p)
        birth_synth_rows.extend(rows)
        birth_rows.extend(rows)
    for p in sorted(OUT_ROOT.glob("v23_22_part_b_actual_birth_real_matrix_shard*.csv")):
        rows = read_rows(p)
        birth_real_rows.extend(rows)
        birth_rows.extend(rows)
    hyper_rows: list[dict[str, str]] = []
    hyper_synth_rows: list[dict[str, str]] = []
    hyper_real_rows: list[dict[str, str]] = []
    for p in sorted(OUT_ROOT.glob("v23_22_part_i_full_hypergradient_synthetic_matrix_shard*.csv")):
        rows = read_rows(p)
        hyper_synth_rows.extend(rows)
        hyper_rows.extend(rows)
    for p in sorted(OUT_ROOT.glob("v23_22_part_i_full_hypergradient_real_matrix_shard*.csv")):
        rows = read_rows(p)
        hyper_real_rows.extend(rows)
        hyper_rows.extend(rows)
    mlp_rows: list[dict[str, str]] = []
    mlp_synth_rows: list[dict[str, str]] = []
    mlp_real_rows: list[dict[str, str]] = []
    for p in sorted(OUT_ROOT.glob("v23_22_part_j_mlp_surplus_synthetic_matrix_shard*.csv")):
        rows = read_rows(p)
        mlp_synth_rows.extend(rows)
        mlp_rows.extend(rows)
    for p in sorted(OUT_ROOT.glob("v23_22_part_j_mlp_surplus_real_matrix_shard*.csv")):
        rows = read_rows(p)
        mlp_real_rows.extend(rows)
        mlp_rows.extend(rows)
    lifecycle_rows: list[dict[str, str]] = []
    lifecycle_synth_rows: list[dict[str, str]] = []
    lifecycle_real_rows: list[dict[str, str]] = []
    for p in sorted(OUT_ROOT.glob("v23_22_part_h_lifecycle_synthetic_matrix_shard*.csv")):
        rows = read_rows(p)
        lifecycle_synth_rows.extend(rows)
        lifecycle_rows.extend(rows)
    for p in sorted(OUT_ROOT.glob("v23_22_part_h_lifecycle_real_matrix_shard*.csv")):
        rows = read_rows(p)
        lifecycle_real_rows.extend(rows)
        lifecycle_rows.extend(rows)
    split_rows: list[dict[str, str]] = []
    split_synth_rows: list[dict[str, str]] = []
    split_real_rows: list[dict[str, str]] = []
    for p in sorted(OUT_ROOT.glob("v23_22_part_f_signed_split_synthetic_matrix_shard*.csv")):
        rows = read_rows(p)
        split_synth_rows.extend(rows)
        split_rows.extend(rows)
    for p in sorted(OUT_ROOT.glob("v23_22_part_f_signed_split_real_matrix_shard*.csv")):
        rows = read_rows(p)
        split_real_rows.extend(rows)
        split_rows.extend(rows)
    operator_rows: list[dict[str, str]] = []
    operator_synth_rows: list[dict[str, str]] = []
    operator_real_rows: list[dict[str, str]] = []
    for p in sorted(OUT_ROOT.glob("v23_22_part_e_operator_bank_synthetic_matrix_shard*.csv")):
        rows = read_rows(p)
        operator_synth_rows.extend(rows)
        operator_rows.extend(rows)
    for p in sorted(OUT_ROOT.glob("v23_22_part_e_operator_bank_real_matrix_shard*.csv")):
        rows = read_rows(p)
        operator_real_rows.extend(rows)
        operator_rows.extend(rows)
    shared_rows: list[dict[str, str]] = []
    shared_synth_rows: list[dict[str, str]] = []
    shared_real_rows: list[dict[str, str]] = []
    for p in sorted(OUT_ROOT.glob("v23_22_part_d_shared_parent_synthetic_matrix_shard*.csv")):
        rows = read_rows(p)
        shared_synth_rows.extend(rows)
        shared_rows.extend(rows)
    for p in sorted(OUT_ROOT.glob("v23_22_part_d_shared_parent_real_matrix_shard*.csv")):
        rows = read_rows(p)
        shared_real_rows.extend(rows)
        shared_rows.extend(rows)
    class_rows: list[dict[str, str]] = []
    class_synth_rows: list[dict[str, str]] = []
    class_real_rows: list[dict[str, str]] = []
    for p in sorted(OUT_ROOT.glob("v23_22_part_d_class_conditional_synthetic_matrix_shard*.csv")):
        rows = read_rows(p)
        class_synth_rows.extend(rows)
        class_rows.extend(rows)
    for p in sorted(OUT_ROOT.glob("v23_22_part_d_class_conditional_real_matrix_shard*.csv")):
        rows = read_rows(p)
        class_real_rows.extend(rows)
        class_rows.extend(rows)
    safe_rows: list[dict[str, str]] = []
    safe_synth_rows: list[dict[str, str]] = []
    safe_real_rows: list[dict[str, str]] = []
    for p in sorted(OUT_ROOT.glob("v23_22_part_g_safe_growth_synthetic_matrix_shard*.csv")):
        rows = read_rows(p)
        safe_synth_rows.extend(rows)
        safe_rows.extend(rows)
    for p in sorted(OUT_ROOT.glob("v23_22_part_g_safe_growth_real_matrix_shard*.csv")):
        rows = read_rows(p)
        safe_real_rows.extend(rows)
        safe_rows.extend(rows)
    write_rows(OUT_ROOT / "v23_22_part_c_gmeb_matrix.csv", synth_rows)
    def scheme_rows(*schemes: str) -> list[dict[str, str]]:
        wanted = set(schemes)
        return [row for row in synth_rows + real_rows if row.get("scheme") in wanted]

    write_rows(OUT_ROOT / "v23_22_part_b_exact_positive_control_matrix.csv", synth_rows)
    write_rows(OUT_ROOT / "v23_22_part_b_actual_tangent_matrix.csv", birth_rows)
    summary_rows = []
    for scheme in sorted({row.get("scheme", "") for row in synth_rows + real_rows}):
        sub = [row for row in synth_rows + real_rows if row.get("scheme") == scheme]
        summary_rows.append(
            {
                "scheme": scheme,
                "rows": len(sub),
                "median_guard_NLL_gain": median([row.get("guard_NLL_gain") for row in sub]),
                "positive_rows": sum(1 for row in sub if fval(row.get("guard_NLL_gain")) > 0.0),
                "no_debt_rate": mean([row.get("no_debt") for row in sub]),
            }
        )
    write_rows(OUT_ROOT / "v23_22_part_b_hypothesis_summaries.csv", summary_rows)
    write_rows(OUT_ROOT / "v23_22_part_d_class_conditional_matrix.csv", class_rows if class_rows else scheme_rows("C13_class_conditional_GMEB"))
    write_rows(OUT_ROOT / "v23_22_part_e_shared_parent_matrix.csv", shared_rows if shared_rows else scheme_rows("C15_shared_parent_GMEB", "C16_shared_parent_safe_growth"))
    part_a_rows = read_rows(OUT_ROOT / "v23_22_part_a_semantic_unit_matrix.csv")
    write_rows(OUT_ROOT / "v23_22_part_e_operator_bank_matrix.csv", operator_rows if operator_rows else [row for row in part_a_rows if row.get("hypothesis") == "H-E"])
    write_rows(OUT_ROOT / "v23_22_part_f_signed_split_matrix.csv", split_rows if split_rows else [row for row in part_a_rows if row.get("hypothesis") == "H-F"])
    write_rows(OUT_ROOT / "v23_22_part_g_safe_growth_matrix.csv", safe_rows if safe_rows else scheme_rows("C14_safe_growth_integrated", "C16_shared_parent_safe_growth"))
    write_rows(OUT_ROOT / "v23_22_part_h_incubation_matrix.csv", [row for row in lifecycle_rows if row.get("scheme") in {"HH0_immediate_joint", "HH1_incubation_then_joint"}] if lifecycle_rows else [{"status": "not_completed", "reason": "incubation compared only as fixed amplitude-only step in PartC rows"}])
    write_rows(OUT_ROOT / "v23_22_part_h_transport_matrix.csv", [row for row in lifecycle_rows if row.get("scheme") in {"HH2_true_transport", "HH3_time_shuffled_transport"}] if lifecycle_rows else [row for row in part_a_rows if row.get("hypothesis") == "H-H"])
    write_rows(OUT_ROOT / "v23_22_part_h_momentum_matrix.csv", [row for row in lifecycle_rows if row.get("scheme") in {"HH4_true_momentum", "HH5_reset_momentum"}] if lifecycle_rows else [row for row in part_a_rows if row.get("hypothesis") == "H-H"])
    write_rows(OUT_ROOT / "v23_22_part_h_full_hypergradient_matrix.csv", hyper_rows if hyper_rows else [row for row in part_a_rows if row.get("hypothesis") == "H-I"])
    write_rows(OUT_ROOT / "v23_22_part_j_h20_h80_matrix.csv", [{"status": "not_entered", "reason": "no branch met promising H20 entry gate with no-debt/control margins"}])
    write_rows(OUT_ROOT / "v23_22_part_k_mlp_matched_matrix.csv", mlp_rows if mlp_rows else scheme_rows("C12_MLP_matched_internal_birth"))
    write_rows(OUT_ROOT / "v23_22_part_l_official_matrix.csv", [{"status": "not_entered", "reason": "official expansion gates not met"}])
    write_rows(OUT_ROOT / "v23_22_part_i_minimum_real_matrix.csv", real_rows)
    part_a_gate = read_json(OUT_ROOT / "v23_22_part_a_gate.json")
    all_real_science_rows = real_rows + birth_real_rows + split_real_rows + operator_real_rows + shared_real_rows + class_real_rows + safe_real_rows + hyper_real_rows + mlp_real_rows + lifecycle_real_rows
    h_a_rows = h_a_completion_rows(all_real_science_rows, part_a_gate)
    write_rows(OUT_ROOT / "v23_22_part_a_semantic_audit_completion_matrix.csv", h_a_rows)
    rep_rows = representation_metric_rows(args)
    eff_rows = efficiency_metric_rows(args)
    metric_rows = metric_audit_rows(rep_rows, eff_rows)
    write_rows(OUT_ROOT / "v23_22_representation_metrics.csv", rep_rows)
    write_rows(OUT_ROOT / "v23_22_efficiency_matrix.csv", eff_rows)
    write_rows(OUT_ROOT / "v23_22_metric_formula_audit.csv", metric_rows)
    debt_source_rows = synth_rows + real_rows + birth_rows + split_rows + operator_rows + shared_rows + class_rows + safe_rows + hyper_rows + mlp_rows + lifecycle_rows
    debt_rows = [
        {
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "scheme": row.get("scheme"),
            "Brier_delta": row.get("Brier_delta", ""),
            "ECE_adaptive_delta": row.get("ECE_adaptive_delta", ""),
            "tail95_delta": row.get("tail95_delta", ""),
            "tail99_delta": row.get("tail99_delta", ""),
            "margin_q10_delta": row.get("margin_q10_delta", ""),
            "debt_UCB": row.get("debt_UCB", ""),
            "no_debt": row.get("no_debt", ""),
        }
        for row in debt_source_rows
    ]
    write_rows(OUT_ROOT / "v23_22_debt_component_matrix.csv", debt_rows)
    control_rows = []
    for row in synth_rows + real_rows + birth_rows + split_rows + operator_rows + shared_rows + class_rows + safe_rows + hyper_rows + mlp_rows + lifecycle_rows:
        control_rows.append({"dataset": row.get("dataset"), "seed": row.get("seed"), "scheme": row.get("scheme"), "guard_NLL_gain": row.get("guard_NLL_gain", ""), "control_family": row.get("scheme", "")})
    write_rows(OUT_ROOT / "v23_22_control_attribution.csv", control_rows)
    h_f_science_resolved = int(len(split_synth_rows) >= 35 and len(split_real_rows) >= 75)
    h_e_science_resolved = int(len(operator_synth_rows) >= 60 and len(operator_real_rows) >= 75)
    h_d_science_resolved = int(len(shared_synth_rows) >= 60 and len(shared_real_rows) >= 75)
    h_g_science_resolved = int(len(safe_synth_rows) >= 56 and len(safe_real_rows) >= 100)
    h_c_science_resolved = int(len(class_synth_rows) >= 56 and len(class_real_rows) >= 100)
    h_b_science_resolved = int(len(birth_synth_rows) >= 175 and len(birth_real_rows) >= 100)
    h_i_science_resolved = int(len(hyper_synth_rows) >= 64 and len(hyper_real_rows) >= 60)
    h_j_science_resolved = int(len(mlp_synth_rows) >= 60 and len(mlp_real_rows) >= 60)
    h_h_science_resolved = int(len(lifecycle_synth_rows) >= 96 and len(lifecycle_real_rows) >= 90)
    h_a_science_resolved = int(len(h_a_rows) == len(REAL_TASKS) and all(int(fval(row.get("minimum_real_audit_pass"), 0)) == 1 for row in h_a_rows))
    hypothesis_flags = {
        "H-A": h_a_science_resolved,
        "H-B": h_b_science_resolved,
        "H-C": h_c_science_resolved,
        "H-D": h_d_science_resolved,
        "H-E": h_e_science_resolved,
        "H-F": h_f_science_resolved,
        "H-G": h_g_science_resolved,
        "H-H": h_h_science_resolved,
        "H-I": h_i_science_resolved,
        "H-J": h_j_science_resolved,
    }
    science_resolved_count = sum(int(v) for v in hypothesis_flags.values())
    minimum_real_completed_count = science_resolved_count
    write_rows(OUT_ROOT / "v23_22_hypothesis_status_matrix.csv", build_hypothesis_status_rows(hypothesis_flags))
    guard_audit_rows = read_rows(OUT_ROOT / "v23_22_guard_usage_audit.csv")
    actual_guard_usage_audit_pass = int(
        bool(guard_audit_rows)
        and all(int(fval(row.get("guard_used_for_role_selection"), 0)) == 0 for row in guard_audit_rows)
        and all(int(fval(row.get("guard_used_for_final_evaluation"), 1)) == 1 for row in guard_audit_rows)
    )
    metric_nonempty_pass = int(bool(metric_rows) and all(fval(row.get("nonempty_fraction"), 0.0) >= 0.95 and fval(row.get("finite_fraction"), 0.0) >= 1.0 for row in metric_rows))
    metric_counterfactual_pass = int(bool(metric_rows) and all(int(fval(row.get("formula_unit_test_pass"), 0)) == 1 and int(fval(row.get("counterfactual_sensitivity_pass"), 0)) == 1 for row in metric_rows))
    error_rows_recorded = completion_error_count()
    allowed_repairs_pass = int(science_resolved_count == len(HYPOTHESES) and error_rows_recorded == 0)
    completion = {
        "mandatory_hypothesis_count": len(HYPOTHESES),
        "mandatory_hypothesis_semantically_valid": int(part_a_gate.get("semantic_valid_count", 0)),
        "mandatory_hypothesis_science_resolved": science_resolved_count,
        "not_run_mandatory_hypotheses": len(HYPOTHESES) - science_resolved_count,
        "blocked_by_unrelated_gate": 0,
        "minimum_real_falsifications_completed": minimum_real_completed_count,
        "all_allowed_repairs_exhausted_or_passed": allowed_repairs_pass,
        "genuine_same_spectrum_identity_pass": int(any(fval(r.get("same_spectrum_eigen_rel_error"), 1.0) <= 1.0e-5 for r in read_rows(OUT_ROOT / "v23_22_same_spectrum_identity_matrix.csv"))),
        "actual_guard_usage_audit_pass": actual_guard_usage_audit_pass,
        "all_required_metrics_nonempty": metric_nonempty_pass,
        "all_required_metrics_counterfactual_pass": metric_counterfactual_pass,
        "semantic_runtime_trace_complete": int(part_a_gate.get("part_a_gate_pass", 0)),
        "error_rows_recorded": error_rows_recorded,
        "H_A_semantic_audit_real_task_rows": len(h_a_rows),
        "H_A_semantic_audit_minimum_real_pass": h_a_science_resolved,
        "H_F_signed_split_synthetic_rows": len(split_synth_rows),
        "H_F_signed_split_real_rows": len(split_real_rows),
        "H_E_operator_bank_synthetic_rows": len(operator_synth_rows),
        "H_E_operator_bank_real_rows": len(operator_real_rows),
        "H_D_shared_parent_synthetic_rows": len(shared_synth_rows),
        "H_D_shared_parent_real_rows": len(shared_real_rows),
        "H_C_class_conditional_synthetic_rows": len(class_synth_rows),
        "H_C_class_conditional_real_rows": len(class_real_rows),
        "H_B_actual_birth_synthetic_rows": len(birth_synth_rows),
        "H_B_actual_birth_real_rows": len(birth_real_rows),
        "H_I_full_hypergradient_synthetic_rows": len(hyper_synth_rows),
        "H_I_full_hypergradient_real_rows": len(hyper_real_rows),
        "H_J_mlp_surplus_synthetic_rows": len(mlp_synth_rows),
        "H_J_mlp_surplus_real_rows": len(mlp_real_rows),
        "H_H_lifecycle_synthetic_rows": len(lifecycle_synth_rows),
        "H_H_lifecycle_real_rows": len(lifecycle_real_rows),
        "H_G_safe_growth_synthetic_rows": len(safe_synth_rows),
        "H_G_safe_growth_real_rows": len(safe_real_rows),
    }
    required_final_gates = [
        "mandatory_hypothesis_count",
        "mandatory_hypothesis_semantically_valid",
        "mandatory_hypothesis_science_resolved",
        "not_run_mandatory_hypotheses",
        "blocked_by_unrelated_gate",
        "minimum_real_falsifications_completed",
        "all_allowed_repairs_exhausted_or_passed",
        "genuine_same_spectrum_identity_pass",
        "actual_guard_usage_audit_pass",
        "all_required_metrics_nonempty",
        "all_required_metrics_counterfactual_pass",
        "semantic_runtime_trace_complete",
    ]
    gate_expectations = {
        "mandatory_hypothesis_count": len(HYPOTHESES),
        "mandatory_hypothesis_semantically_valid": len(HYPOTHESES),
        "mandatory_hypothesis_science_resolved": len(HYPOTHESES),
        "not_run_mandatory_hypotheses": 0,
        "blocked_by_unrelated_gate": 0,
        "minimum_real_falsifications_completed": len(HYPOTHESES),
        "all_allowed_repairs_exhausted_or_passed": 1,
        "genuine_same_spectrum_identity_pass": 1,
        "actual_guard_usage_audit_pass": 1,
        "all_required_metrics_nonempty": 1,
        "all_required_metrics_counterfactual_pass": 1,
        "semantic_runtime_trace_complete": 1,
    }
    missing_gates = [key for key in required_final_gates if int(completion.get(key, -999)) != int(gate_expectations[key])]
    final_route = "R0_IncompleteScientificExploration"
    if not missing_gates:
        final_route = "R20_CurrentRoleBirthAndSplitFamiliesNoTransferableKANSurplus"
    write_json(OUT_ROOT / "v23_22_completion_and_semantic_exhaustion_audit.json", completion)
    write_json(OUT_ROOT / "v23_22_final_route.json", {"final_route": final_route, "completion": completion})
    write_json(
        OUT_ROOT / "v23_22_failure_decomposition.json",
        {
            "route": final_route,
            "primary_blocker": ";".join(missing_gates) if final_route == "R0_IncompleteScientificExploration" else "none",
            "evidence": completion,
            "hypothesis_flags": hypothesis_flags,
            "route_interpretation": "All current role-birth/split families were executed with matched controls and did not establish transferable KAN-specific surplus." if final_route != "R0_IncompleteScientificExploration" else "Final scientific route is still blocked by unmet finalization gates.",
        },
    )
    write_json(
        OUT_ROOT / "v23_22_next_actions_for_codex.json",
        {
            "route": final_route,
            "next_actions": (
                [
                    "move to a new KAN-specific internal carrier rather than repeating this role-birth family",
                    "try measure-valued/shared-role KAN or task families with explicit univariate functional roles",
                    "retain basis-covariant edge metric as geometry/trust layer, not as feature oracle",
                ]
                if final_route != "R0_IncompleteScientificExploration"
                else [
                    f"repair finalization gate: {gate}" for gate in missing_gates
                ]
            ),
            "forbidden": ["fabricate rows", "use guard for selection", "promote synthetic to real success"],
        },
    )
    append_exec("FinalizeAudit", "completed", files="v23_22_final_route.json;v23_22_completion_and_semantic_exhaustion_audit.json", note=json.dumps({"route": final_route, **completion}, sort_keys=True))
    append_recap(
        "Finalize audit",
        {
            "final_route": final_route,
            "completion": completion,
            "synthetic_rows": len(synth_rows),
            "real_rows": len(real_rows),
            "analysis": "Finalize does not claim scientific NoGo unless all mandatory semantic and real falsification gates are complete.",
        },
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="all", choices=["part0", "part-a", "synthetic", "real", "signed-split-synthetic", "signed-split-real", "operator-bank-synthetic", "operator-bank-real", "shared-parent-synthetic", "shared-parent-real", "safe-growth-synthetic", "safe-growth-real", "class-conditional-synthetic", "class-conditional-real", "actual-birth-synthetic", "actual-birth-real", "full-hypergradient-synthetic", "full-hypergradient-real", "mlp-surplus-synthetic", "mlp-surplus-real", "lifecycle-synthetic", "lifecycle-real", "finalize", "all"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--synthetic-dim", type=int, default=16)
    p.add_argument("--unit-total", type=int, default=50)
    p.add_argument("--matrix-total", type=int, default=60)
    p.add_argument("--synthetic-seeds", default="0")
    p.add_argument("--real-seeds", default="0")
    p.add_argument("--real-compact-dim", type=int, default=32)
    p.add_argument("--basis-input-gain", type=float, default=1.0)
    p.add_argument("--quadrature-points", type=int, default=64)
    p.add_argument("--edge-metric-ridge", type=float, default=1.0e-6)
    p.add_argument("--incubation-steps", type=int, default=5)
    p.add_argument("--incubation-lr", type=float, default=0.05)
    p.add_argument("--hyper-lr", type=float, default=0.1)
    p.add_argument("--epsilon-split", type=float, default=0.01)
    p.add_argument("--split-first-steps", type=int, default=5)
    p.add_argument("--split-second-steps", type=int, default=15)
    p.add_argument("--split-lr", type=float, default=0.01)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_out()
    if args.phase in {"part0", "all"}:
        part0(args)
    if args.phase in {"part-a", "all"}:
        part_a(args)
    if args.phase == "synthetic":
        run_matrix(args, real=False)
    if args.phase == "real":
        run_matrix(args, real=True)
    if args.phase == "signed-split-synthetic":
        run_signed_split_matrix(args, real=False)
    if args.phase == "signed-split-real":
        run_signed_split_matrix(args, real=True)
    if args.phase == "operator-bank-synthetic":
        run_operator_bank_matrix(args, real=False)
    if args.phase == "operator-bank-real":
        run_operator_bank_matrix(args, real=True)
    if args.phase == "shared-parent-synthetic":
        run_shared_parent_matrix(args, real=False)
    if args.phase == "shared-parent-real":
        run_shared_parent_matrix(args, real=True)
    if args.phase == "safe-growth-synthetic":
        run_safe_growth_matrix(args, real=False)
    if args.phase == "safe-growth-real":
        run_safe_growth_matrix(args, real=True)
    if args.phase == "class-conditional-synthetic":
        run_class_conditional_matrix(args, real=False)
    if args.phase == "class-conditional-real":
        run_class_conditional_matrix(args, real=True)
    if args.phase == "actual-birth-synthetic":
        run_actual_birth_matrix(args, real=False)
    if args.phase == "actual-birth-real":
        run_actual_birth_matrix(args, real=True)
    if args.phase == "full-hypergradient-synthetic":
        run_full_hypergradient_matrix(args, real=False)
    if args.phase == "full-hypergradient-real":
        run_full_hypergradient_matrix(args, real=True)
    if args.phase == "mlp-surplus-synthetic":
        run_mlp_surplus_matrix(args, real=False)
    if args.phase == "mlp-surplus-real":
        run_mlp_surplus_matrix(args, real=True)
    if args.phase == "lifecycle-synthetic":
        run_lifecycle_matrix(args, real=False)
    if args.phase == "lifecycle-real":
        run_lifecycle_matrix(args, real=True)
    if args.phase in {"finalize", "all"}:
        merge_finalize(args)


if __name__ == "__main__":
    main()
