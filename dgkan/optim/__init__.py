"""Optimizer primitives."""

from .manual_adamw import AdamWState, ManualAdamWConfig, adamw_update_
from .ccsg_optimizer_wrapper import CCSGOptimizerWrapper
from .meta_fu_optimizer_wrapper import MetaFUOptimizerWrapper
from .scfg_optimizer_wrapper import SCFGOptimizerWrapper
from .witnessed_optimizer_wrapper import WitnessedOptimizerWrapper
from .witnessed_preconditioner_wrapper import WitnessedPreconditionerWrapper
from .composite_metric_preserving_optimizer_wrapper import CompositeMetricPreservingOptimizerWrapper
from dgkan.fu.kan_edge_natural_residual import EdgeNaturalResidualOptimizer
from dgkan.fu.kan_distributional_edge_natural_residual import DistributionalEdgeNaturalResidualOptimizer
from dgkan.fu.kan_brier_natural_dynamic_edge_basis import BrierNaturalDynamicEdgeOptimizer

__all__ = [
    "AdamWState",
    "BrierNaturalDynamicEdgeOptimizer",
    "CCSGOptimizerWrapper",
    "CompositeMetricPreservingOptimizerWrapper",
    "DistributionalEdgeNaturalResidualOptimizer",
    "EdgeNaturalResidualOptimizer",
    "ManualAdamWConfig",
    "MetaFUOptimizerWrapper",
    "SCFGOptimizerWrapper",
    "WitnessedOptimizerWrapper",
    "WitnessedPreconditionerWrapper",
    "adamw_update_",
]
