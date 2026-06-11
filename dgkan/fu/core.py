"""Core update tensor semantics for DG-KAN v17."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable
import math

import torch
import torch.nn.functional as F


@dataclass
class UpdateTensor:
    tensor: torch.Tensor
    kind: str
    sign_rule: str
    space: str
    source: str
    mechanism: str
    role: str = "all"
    one_step_descent_claim: int = 1
    diagnostics: dict[str, Any] | None = None

    def to_metadata(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("tensor", None)
        row["update_norm"] = float(torch.linalg.vector_norm(self.tensor.detach()).item()) if self.tensor.numel() else 0.0
        return row


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    chunks = [t.reshape(-1) for t in tensors]
    if not chunks:
        return torch.zeros(0)
    return torch.cat(chunks)


def flat_params(model: torch.nn.Module) -> torch.Tensor:
    params = trainable_parameters(model)
    if not params:
        return torch.zeros(0)
    return torch.cat([p.detach().reshape(-1) for p in params])


def flat_grad(model: torch.nn.Module, device: torch.device | None = None) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for p in trainable_parameters(model):
        if p.grad is None:
            chunks.append(torch.zeros_like(p).reshape(-1))
        else:
            chunks.append(p.grad.detach().reshape(-1))
    if not chunks:
        dev = device or torch.device("cpu")
        return torch.zeros(0, device=dev)
    return torch.cat(chunks)


def load_flat_params(model: torch.nn.Module, vector: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in trainable_parameters(model):
            n = int(p.numel())
            p.copy_(vector[offset : offset + n].view_as(p).to(device=p.device, dtype=p.dtype))
            offset += n


def apply_update(model: torch.nn.Module, update: UpdateTensor, lr: float = 1.0) -> None:
    sign = -1.0 if update.sign_rule == "subtract" else 1.0
    offset = 0
    with torch.no_grad():
        for p in trainable_parameters(model):
            n = int(p.numel())
            if n:
                p.add_(update.tensor[offset : offset + n].view_as(p).to(device=p.device, dtype=p.dtype), alpha=sign * float(lr))
            offset += n


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    n = min(int(a.numel()), int(b.numel()))
    x = a[:n].detach().float().reshape(-1)
    y = b[:n].detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom.item()) <= 1.0e-12:
        return 0.0
    return float((x @ y / denom).clamp(-1.0, 1.0).item())


def normalized_like(update: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    ref_norm = torch.linalg.vector_norm(reference.detach())
    upd_norm = torch.linalg.vector_norm(update.detach())
    if float(upd_norm.item()) <= 1.0e-12:
        return torch.zeros_like(update)
    return update * (ref_norm / upd_norm.clamp_min(1.0e-12))


def loss_value(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float(F.cross_entropy(model(x).float(), y).item())


def small_step_sanity(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, update: UpdateTensor, eps: float = 1.0e-3) -> dict[str, Any]:
    before = flat_params(model)
    base_loss = loss_value(model, x, y)
    apply_update(model, update, lr=float(eps))
    plus_loss = loss_value(model, x, y)
    load_flat_params(model, before)
    opposite = UpdateTensor(
        tensor=update.tensor,
        kind=update.kind,
        sign_rule="add" if update.sign_rule == "subtract" else "subtract",
        space=update.space,
        source=update.source,
        mechanism=update.mechanism,
        role=update.role,
        one_step_descent_claim=update.one_step_descent_claim,
    )
    apply_update(model, opposite, lr=float(eps))
    minus_loss = loss_value(model, x, y)
    load_flat_params(model, before)
    if not all(math.isfinite(v) for v in [base_loss, plus_loss, minus_loss]):
        status = "invalid_nonfinite_loss"
    elif int(update.one_step_descent_claim):
        status = "pass" if plus_loss <= minus_loss + 1.0e-9 else "fail"
    else:
        status = "not_descent_boundary"
    return {
        "mechanism": update.mechanism,
        "update_kind": update.kind,
        "sign_rule": update.sign_rule,
        "space": update.space,
        "source": update.source,
        "one_step_descent_claim": int(update.one_step_descent_claim),
        "base_loss": base_loss,
        "L_plus_eps": plus_loss,
        "L_minus_eps": minus_loss,
        "small_step_loss_sanity": int(status == "pass" or status == "not_descent_boundary"),
        "opposite_sign_loss_sanity": int((not int(update.one_step_descent_claim)) or plus_loss <= minus_loss + 1.0e-9),
        "status": status,
    }
