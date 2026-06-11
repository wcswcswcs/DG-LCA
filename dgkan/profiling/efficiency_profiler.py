"""Compatibility shim for v17.1 efficiency profiler imports."""

from __future__ import annotations

from dgkan.profiling.efficiency_v17 import profile_isolated

__all__ = ["profile_isolated"]
