"""KANbeFair bridge metadata for v22.17.

This module is deliberately small: it does not import the external KANbeFair
tree, so provenance and patching remain controlled by experiment runners.
"""

from __future__ import annotations

from dataclasses import dataclass


KANBEFAIR_BASELINE_MODELS = (
    "MLP",
    "KAN",
    "BSpline_MLP",
    "BSpline_First_MLP",
)

DG_BRIDGE_MODELS = (
    "DGMLP",
    "DGMLP_FU",
    "DGMLP_FU_NOOP",
    "DGMLP_FU_GATED",
    "DGMLP_FU_ACTIONABLE",
    "DGMLP_FU_RELEASE",
    "DGMLP_FU_SPLIT_ACTIONABLE",
    "DGMLP_FU_SPLIT_RELEASE",
    "DGMLP_FU_SPLIT_VIRTUAL",
    "DGKAN_DFOU",
    "DGKAN_DFOU_FU",
    "DGKAN_DFOU_FU_NOOP",
    "DGKAN_DFOU_FU_GATED",
    "DGKAN_DFOU_FU_ACTIONABLE",
    "DGKAN_DFOU_FU_RELEASE",
    "DGKAN_DFOU_FU_SPLIT_ACTIONABLE",
    "DGKAN_DFOU_FU_SPLIT_RELEASE",
    "DGKAN_DFOU_FU_SPLIT_VIRTUAL",
    "DGKAN_DCHE",
    "DGKAN_DCHE_FU",
    "DGKAN_DCHE_FU_NOOP",
    "DGKAN_DCHE_FU_GATED",
    "DGKAN_DCHE_FU_ACTIONABLE",
    "DGKAN_DCHE_FU_RELEASE",
    "DGKAN_DCHE_FU_SPLIT_ACTIONABLE",
    "DGKAN_DCHE_FU_SPLIT_RELEASE",
    "DGKAN_DCHE_FU_SPLIT_VIRTUAL",
)

DATASET_NAME_MAP = {
    "FashionMNIST": "FMNIST",
    "CIFAR10": "Cifar10",
    "CIFAR100": "Cifar100",
    "continual_MNIST": "Class_MNIST",
}


@dataclass(frozen=True)
class FirewallRow:
    model_name: str
    model_origin: str
    is_dgkan_strict_fc_purekan: int
    uses_kanbefair_loader: int
    uses_kanbefair_baseline_model: int
    uses_bspline_official_path: int
    uses_readout_diagnostic: int
    basis_native_claim_allowed: int

    def to_row(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "model_origin": self.model_origin,
            "is_dgkan_strict_fc_purekan": self.is_dgkan_strict_fc_purekan,
            "uses_kanbefair_loader": self.uses_kanbefair_loader,
            "uses_kanbefair_baseline_model": self.uses_kanbefair_baseline_model,
            "uses_bspline_official_path": self.uses_bspline_official_path,
            "uses_readout_diagnostic": self.uses_readout_diagnostic,
            "basis_native_claim_allowed": self.basis_native_claim_allowed,
        }


def firewall_rows() -> list[dict[str, object]]:
    rows: list[FirewallRow] = []
    for model in KANBEFAIR_BASELINE_MODELS:
        rows.append(
            FirewallRow(
                model_name=model,
                model_origin="KANbeFair_baseline",
                is_dgkan_strict_fc_purekan=0,
                uses_kanbefair_loader=1,
                uses_kanbefair_baseline_model=1,
                uses_bspline_official_path=int("BSpline" in model),
                uses_readout_diagnostic=0,
                basis_native_claim_allowed=0,
            )
        )
    for model in DG_BRIDGE_MODELS:
        is_kan = int(model.startswith("DGKAN"))
        rows.append(
            FirewallRow(
                model_name=model,
                model_origin="DGKAN_bridge",
                is_dgkan_strict_fc_purekan=is_kan,
                uses_kanbefair_loader=1,
                uses_kanbefair_baseline_model=0,
                uses_bspline_official_path=0,
                uses_readout_diagnostic=0,
                basis_native_claim_allowed=is_kan,
            )
        )
    return [row.to_row() for row in rows]


def canonical_dataset_name(name: str) -> str:
    return DATASET_NAME_MAP.get(name, name)
