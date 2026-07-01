"""Optimizer-owned wrapper for controlled composite metric tube flow."""

from __future__ import annotations

from dgkan.optim.composite_metric_preserving_optimizer_wrapper import CompositeMetricPreservingOptimizerWrapper


class CompositeMetricTubeOptimizerWrapper(CompositeMetricPreservingOptimizerWrapper):
    """Named wrapper for v22.90 tube experiments.

    The mechanics are identical to the v22.89 optimizer-owned wrapper: gradient
    transformation happens inside ``step()``, then the operator performs its
    retraction hook after the base optimizer mutates parameters.
    """


__all__ = ["CompositeMetricTubeOptimizerWrapper"]
