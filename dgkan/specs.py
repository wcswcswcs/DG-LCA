"""Typed specs for clean DG-KAN candidates and run contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional


ModelLevel = Literal["transitional", "strict_head", "full_edge"]


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    model_level: ModelLevel
    stack_type: str
    head_type: str
    edge_basis: str
    hidden_dim: int
    depth: int
    basis_count: int
    loss_type: str = "CE"
    label_smoothing: float = 0.0
    optimizer_type: str = "manual_adamw"
    functional_update: Optional[str] = None
    official_eligible: bool = True

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContractFlags:
    external_teacher_used: int = 0
    self_teacher_used: int = 0
    geometry_loss_used: int = 0
    sampler_changed: int = 0
    class_weight_used: int = 0
    cpu_offload_used: int = 0
    uses_loss_backward: int = 0
    fake_data_used: int = 0
    proxy_row_used: int = 0

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModuleAuditSpec:
    module_name: str
    source_file: str
    public_api_count: int
    unit_test_count: int
    import_pass: int

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)
