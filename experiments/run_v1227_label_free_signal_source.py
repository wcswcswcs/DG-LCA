#!/usr/bin/env python
"""v12.27 label-free signal-source functional bridge runner.

This wrapper reuses the v12.26 label-free-only execution harness while
registering v12.27 F27 functional candidates.  Base construction is still
strictly label-free: ``y_stats=None`` is forced by the imported harness, and
candidate directions are inherited from train-stream / unlabeled v12.25
functional primitives.
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


def _take(old_name: str, new_name: str, component_ids: str | None = None, extra: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    spec = copy.deepcopy(comp.CANDIDATES[old_name])
    spec["component_ids"] = component_ids or str(spec.get("component_ids", old_name))
    spec["v1227_source_template"] = old_name
    if extra:
        spec.update(extra)
    return new_name, spec


def v1227_candidates() -> dict[str, dict[str, Any]]:
    pairs = [
        _take(
            "F25-NG148-stableCenteredLowRankMixAlpha060PostCal125",
            "F27-G1-GeometryGuardThenDirect",
            "geometry_guard_then_direct",
            {"v1227_mechanism_family": "F1-guard_then_task"},
        ),
        _take(
            "F25-NG144-lowRankCouplingAlpha060PostCal125",
            "F27-G2-GeometryGuardThenQuadDirect",
            "geometry_guard_then_quad_direct",
            {"v1227_mechanism_family": "F1-guard_then_task"},
        ),
        _take(
            "F25-NG156-crossRefCouplingAlpha060PostCal125",
            "F27-G3-GeometryGuardThenLowRank",
            "geometry_guard_then_low_rank",
            {"v1227_mechanism_family": "F1-guard_then_task"},
        ),
        _take(
            "F25-NG40-quarterResponseResidual-quadDirect-I27I26",
            "F27-R1-TaskEventLineCResidualized",
            "task_event_linec_residualized",
            {"v1227_mechanism_family": "F2-task_event_linec_residual"},
        ),
        _take(
            "F25-NG37-quarterResponseResidual-I27I26-8515",
            "F27-R2-ControlResidualizedLineCPreserved",
            "control_residualized_linec_preserved",
            {"v1227_mechanism_family": "F2-task_event_linec_residual"},
        ),
        _take(
            "F25-NG154-lowRankQuadDirectAlpha085PostCal125",
            "F27-R3-DirectGainLineCNullProjected",
            "direct_gain_linec_null_projected",
            {"v1227_mechanism_family": "F2-task_event_linec_residual"},
        ),
        _take(
            "F25-NG44-featureCovDirectLift",
            "F27-C1-LineCEventUnlabeledReadoutComp",
            "linec_event_unlabeled_readout_comp",
            {"v1227_mechanism_family": "F3-linec_event_task_comp"},
        ),
        _take(
            "F25-NG160-stableCenteredCrossRefMixAlpha060PostCal125",
            "F27-C2-LineCEventCovTransportComp",
            "linec_event_cov_transport_comp",
            {"v1227_mechanism_family": "F3-linec_event_task_comp"},
        ),
        _take(
            "F25-NG114-tailClippedDenoiseLowTailMix-7030",
            "F27-C3-LineCEventViewStableComp",
            "linec_event_view_stable_comp",
            {"v1227_mechanism_family": "F3-linec_event_task_comp"},
        ),
    ]
    out = dict(pairs)
    for name, spec in out.items():
        if lfbridge.forbidden_token_present(name, spec.get("component_ids", ""), spec.get("v1227_source_template", "")):
            raise RuntimeError(f"v12.27 functional candidate has forbidden token: {name}")
    return out


def run() -> dict[str, Any]:
    lfbridge.v1226_candidates = v1227_candidates  # type: ignore[assignment]
    return lfbridge.run()


if __name__ == "__main__":
    run()
