#!/usr/bin/env python
"""v12.28 label-free base recovery functional re-entry runner.

The base construction is delegated to the v12.26 label-free harness, which
forces y_stats=None and rejects trainprobe/label-init tokens.  This wrapper only
renames and groups the loss-agnostic functional candidates for the v12.28
shadow/re-entry contract.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

import run_v1225_composite_functional_bridge as comp  # noqa: E402
import run_v1226_label_free_only_bridge as lfbridge  # noqa: E402


def _take(old_name: str, new_name: str, component_ids: str, family: str) -> tuple[str, dict[str, Any]]:
    spec = copy.deepcopy(comp.CANDIDATES[old_name])
    spec["component_ids"] = component_ids
    spec["v1228_source_template"] = old_name
    spec["v1228_mechanism_family"] = family
    spec["promotion_allowed"] = 0
    return new_name, spec


def v1228_candidates() -> dict[str, dict[str, Any]]:
    pairs = [
        _take(
            "F25-NG148-stableCenteredLowRankMixAlpha060PostCal125",
            "F28-S1-geometryGuardThenTask",
            "geometry_guard_then_task_shadow",
            "F1-task-control-positive-linec-guard",
        ),
        _take(
            "F25-NG138-stableCenteredDenoiseAlpha060PostCal125",
            "F28-S2-taskThenGeometryGuard",
            "task_then_geometry_guard_shadow",
            "F1-task-control-positive-linec-guard",
        ),
        _take(
            "F25-NG40-quarterResponseResidual-quadDirect-I27I26",
            "F28-S3-controlResidualizedDirectQuad",
            "control_residualized_direct_quad_shadow",
            "F2-control-residualized-direct-quad",
        ),
        _take(
            "F25-NG147-tailClippedLowRankCouplingAlpha060PostCal125",
            "F28-S4-quadReservoirReleaseShadow",
            "quad_reservoir_release_shadow",
            "F3-linec-reservoir-release",
        ),
        _take(
            "F25-NG158-entropyCrossRefCouplingAlpha060PostCal125",
            "F28-S5-lowRankLogitSubspaceShadow",
            "low_rank_logit_subspace_shadow",
            "F4-low-rank-logit-subspace",
        ),
        _take(
            "F25-NG44-featureCovDirectLift",
            "F28-S6-unlabeledCovTransportShadow",
            "unlabeled_cov_transport_shadow",
            "F5-unlabeled-covariance-transport",
        ),
    ]
    out = dict(pairs)
    for name, spec in out.items():
        if lfbridge.forbidden_token_present(name, spec.get("component_ids", ""), spec.get("v1228_source_template", "")):
            raise RuntimeError(f"v12.28 functional candidate has forbidden token: {name}")
    return out


def run() -> dict[str, Any]:
    lfbridge.v1226_candidates = v1228_candidates  # type: ignore[assignment]
    return lfbridge.run()


if __name__ == "__main__":
    run()
