"""Small candidate registry used by v9 runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .specs import CandidateSpec


@dataclass
class CandidateRegistry:
    _items: Dict[str, CandidateSpec] = field(default_factory=dict)

    def register(self, spec: CandidateSpec) -> CandidateSpec:
        if spec.candidate_id in self._items:
            raise ValueError(f"duplicate candidate_id={spec.candidate_id}")
        self._items[spec.candidate_id] = spec
        return spec

    def get(self, candidate_id: str) -> CandidateSpec:
        return self._items[candidate_id]

    def rows(self) -> List[dict]:
        return [spec.to_row() for spec in self._items.values()]

    def __iter__(self) -> Iterable[CandidateSpec]:
        return iter(self._items.values())


def default_v90_registry() -> CandidateRegistry:
    registry = CandidateRegistry()
    registry.register(
        CandidateSpec(
            candidate_id="DG-Transitional-CleanCE-KW4-hidden28",
            model_level="transitional",
            stack_type="linear_silu_packed",
            head_type="poly2_silu_head",
            edge_basis="poly2_silu_head_only",
            hidden_dim=28,
            depth=2,
            basis_count=3,
            functional_update="Adaptive-FT-P",
        )
    )
    registry.register(
        CandidateSpec(
            candidate_id="DG-FullEdge-Poly2Silu-smoke",
            model_level="full_edge",
            stack_type="edge_function_layer",
            head_type="edge_function_head",
            edge_basis="poly2_silu",
            hidden_dim=16,
            depth=2,
            basis_count=5,
            functional_update=None,
        )
    )
    return registry
