"""Compatibility shim for v17.1 manual optimizer imports."""

from __future__ import annotations

from dgkan.fu.optimizers import ManualOptimizer, OptimizerStepResult, make_manual_optimizer

__all__ = ["ManualOptimizer", "OptimizerStepResult", "make_manual_optimizer"]
