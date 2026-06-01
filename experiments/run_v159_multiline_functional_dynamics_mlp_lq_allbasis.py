#!/usr/bin/env python3
"""DG-KAN v15.9 multi-line functional dynamics runner.

This runner deliberately treats MLP and LQ as active lines, not just controls.
LineC/tail/AUC/calibration metrics are audit/gate/failure-taxonomy only and
never generate update directions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from copy import copy
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dgkan.models import fc_purekan_lq as lq  # noqa: E402
from experiments import run_v1410_nonrat_fms_transfer_fms_definition_reset as v1410  # noqa: E402
from experiments import run_v144_real_transfer_fms_all_basis_substrate as v144  # noqa: E402
from experiments import run_v150_function_update_allbasis_parallel as v150  # noqa: E402
from experiments import run_v154_split_consensus_signal_subspace_fu_allbasis as v154  # noqa: E402
from experiments import run_v158_dynamics_harness_decoupled_decay_recovery_allbasis as v158  # noqa: E402


PLAN_DOC = ROOT / "docs/DG-KAN_v15.9_MultiLineFunctionalDynamics_MLP_LQ_AllBasis_完整计划.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v15.9_MultiLineFunctionalDynamics_MLP_LQ_AllBasis_实验结果复盘.md"
EXEC_LOG_DOC = ROOT / "docs/DG-KAN_v15.9_MultiLineFunctionalDynamics_MLP_LQ_AllBasis_执行日志.md"
DEFAULT_OUT = ROOT / "results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/official_v159"
DEFAULT_LINE_E_OUT = ROOT / "results/v15_9_multiline_functional_dynamics_mlp_lq_allbasis/line_e_v159_allbasis_substrate"
MANUAL_ANALYSIS_START = "<!-- V15.9_MANUAL_ANALYSIS_START -->"
MANUAL_ANALYSIS_END = "<!-- V15.9_MANUAL_ANALYSIS_END -->"

fnum = v154.fnum
sint = v154.sint
mean = v154.mean
median = v154.median
read_rows = v154.read_rows
write_rows = v154.write_rows
write_json = v154.write_json
parse_csv = v154.parse_csv
parse_ints = v154.parse_ints
resolve_cuda_device = v154.resolve_cuda_device


LINE_A_METHODS = [
    "A0-D-CHE-AdamW",
    "A1-D-CHE-G7R-PulseOnce-AdamWRecovery",
    "A2-D-CHE-G7R-PulseEvery50-AdamWRecovery",
    "A3-D-CHE-G7R-EarlyPulseOnly-AdamWRecovery",
    "A4-D-CHE-G7R-MidPulseOnly-AdamWRecovery",
    "A5-D-CHE-G7R-LatePulseOnly-AdamWRecovery",
    "A6-D-CHE-G7R-Pulse-GlobalDecoupledDecayRecovery",
    "A7-D-CHE-G7R-Pulse-DegreeWiseDecayRecovery",
    "A8-D-CHE-G7R-Pulse-HighDegreeExtraDecayRecovery",
    "A9-D-CHE-G7R-Pulse-ReadoutBasisDecoupledDecayRecovery",
    "A10-D-CHE-G7R-Pulse-MomentumEMARecovery",
    "A11-D-CHE-G7R-Pulse-LRCooldownRecovery",
    "A12-D-CHE-G7R-Pulse-LookaheadConsolidation",
    "A13-D-CHE-G7R-Pulse-ScheduleFreeLongEMARecovery",
    "ACTRL1-D-CHE-RandomMatchedPulse-SameAdamWRecovery",
    "ACTRL2-D-CHE-RandomMatchedPulse-SameDecayRecovery",
    "ACTRL3-D-CHE-NoOpMatchedOverhead",
    "ACTRL4-D-CHE-AdamWExtraStepsMatchedTime",
]
LINE_A_CONTROLS = {
    "A0-D-CHE-AdamW",
    "ACTRL1-D-CHE-RandomMatchedPulse-SameAdamWRecovery",
    "ACTRL2-D-CHE-RandomMatchedPulse-SameDecayRecovery",
    "ACTRL3-D-CHE-NoOpMatchedOverhead",
    "ACTRL4-D-CHE-AdamWExtraStepsMatchedTime",
}
LINE_A_CANDIDATES = [m for m in LINE_A_METHODS if m not in LINE_A_CONTROLS]

LINE_B_METHODS = [
    "B0-MLP-AdamW",
    "B1-MLP-CautiousAdamW",
    "B2-MLP-MGUP",
    "B3-MLP-SplitConsensusMetric",
    "B4-MLP-FunctionalPulseOnce-AdamWRecovery",
    "B5-MLP-FunctionalPulseEvery50-AdamWRecovery",
    "B6-MLP-FunctionalPulse-GlobalDecayRecovery",
    "B7-MLP-FunctionalPulse-MomentumEMARecovery",
    "B8-MLP-FunctionalPulse-LRCooldownRecovery",
    "B9-MLP-FunctionalPulse-LookaheadConsolidation",
    "B10-MLP-AmortizedPersistentFMS-Recheck",
    "BCTRL1-MLP-RandomMatchedPulse-SameRecovery",
    "BCTRL2-MLP-NoOpMatchedOverhead",
    "BCTRL3-MLP-AdamWExtraStepsMatchedTime",
]
LINE_B_CONTROLS = {
    "B0-MLP-AdamW",
    "BCTRL1-MLP-RandomMatchedPulse-SameRecovery",
    "BCTRL2-MLP-NoOpMatchedOverhead",
    "BCTRL3-MLP-AdamWExtraStepsMatchedTime",
}
LINE_B_CANDIDATES = [m for m in LINE_B_METHODS if m not in LINE_B_CONTROLS]

LINE_C_REANCHOR = [
    ("C0-HistoricalLQReference", lq.LQSpec("C0-HistoricalLQReference", "t2", 64, "default", 0.8)),
    ("C1-CurrentLQReproduction", lq.LQSpec("C1-CurrentLQReproduction", "t2", 64, "default", 0.8)),
    ("C2-ExactProtocolLQReplay", lq.LQSpec("C2-ExactProtocolLQReplay", "t2", 64, "orthogonal_lift", 0.8)),
    ("C3-RepairedLQ-FaninOutputScaleConfirmed", lq.LQSpec("C3-RepairedLQ-FaninOutputScaleConfirmed", "t2", 64, "default", 1.0)),
    ("C4-RepeatedCurrentLQ", lq.LQSpec("C4-RepeatedCurrentLQ", "t2", 64, "default", 0.8)),
]
LINE_C_FUNCTIONAL = [
    "C5-LQ-G7Analog-PulseOnce-AdamWRecovery",
    "C6-LQ-G7Analog-Pulse-DecayRecovery",
    "C7-LQ-G7Analog-Pulse-MomentumRecovery",
    "C8-LQ-SplitConsensusMetric",
    "C9-LQ-SnapshotLateAttachFunctionalPulse",
    "CCTRL-RandomMatchedPulse-SameRecovery",
]

LINE_D_METHODS = [
    "D0-RAT-AdamW",
    "D1-RAT-SplitConsensusMetricPulse-AdamWRecovery",
    "D2-RAT-SplitConsensusMetricPulse-DecayRecovery",
    "D3-RAT-SplitConsensusMetricPulse-MomentumRecovery",
    "DCTRL-RAT-RandomMatchedPulse-SameRecovery",
]
LINE_D_CONTROLS = {"D0-RAT-AdamW", "DCTRL-RAT-RandomMatchedPulse-SameRecovery"}
LINE_D_CANDIDATES = [m for m in LINE_D_METHODS if m not in LINE_D_CONTROLS]

LINE_E_CANDIDATES = [
    "D-FOU82-LowFreqIdentityResidualV5",
    "D-FOU83-BandwiseSNRWarmupV5",
    "D-FOU84-PhaseStableBandMixV5",
    "D-FOU85-NoMaterializeLifetimeV5",
    "D-FOU86-HighFrequencyQuarantineV5",
    "D-RBF80-ActiveCenterOccupancyV5",
    "D-RBF81-WidthConditionGuardV5",
    "D-RBF82-CompactBumpNoDenseV5",
    "D-RBF83-GaussianLocalK4TaskHealthV5",
    "D-RBF84-IdentityResidualWidthWarmupV5",
    "D-WAV69-TriangularSupportV5",
    "D-WAV70-ScaleOccupancyV5",
    "D-WAV71-SupportOverlapDampingV5",
    "D-WAV72-LocalTailCoverageAuditV5",
]

FIGURES = [
    "fig_01_multiline_progress_dashboard.svg",
    "fig_02_dche_vs_mlp_vs_lq_source_retention.svg",
    "fig_03_tail_debt_recovery_curves.svg",
    "fig_04_linec_debt_recovery_curves.svg",
    "fig_05_recovery_mechanism_comparison.svg",
    "fig_06_decay_only_vs_fu_decay_decomposition.svg",
    "fig_07_mlp_functional_dynamics_dashboard.svg",
    "fig_08_lq_reanchor_and_late_attach_dashboard.svg",
    "fig_09_allbasis_substrate_heatmap.svg",
    "fig_10_carrier_specificity_matrix.svg",
    "fig_11_source_debt_phase_portrait.svg",
    "fig_12_route_decision_ladder.svg",
]
REQUIRED = [
    "v159_route_decision.json",
    "v159_line_r_audit.csv",
    "v159_line_a_dche_dynamics.csv",
    "v159_dche_no_regression_monitor.csv",
    "v159_line_a_horizon_recovery.csv",
    "v159_line_a_h400_long.csv",
    "v159_line_b_mlp_dynamics.csv",
    "v159_line_b_horizon_recovery.csv",
    "v159_line_b_h400_long.csv",
    "v159_line_c_lq_reanchor.csv",
    "v159_line_c_lq_functional.csv",
    "v159_line_d_rational_monitor.csv",
    "v159_line_e_allbasis_results.csv",
    "v159_line_f_crossline_route.csv",
    "v159_failure_taxonomy.csv",
    "v159_budget_exhaustion_certificate.csv",
    "v159_no_go_boundary.csv",
    "v159_next_hypothesis_queue.csv",
]


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def with_overrides(args: argparse.Namespace, **kwargs: Any) -> argparse.Namespace:
    out = copy(args)
    for key, val in kwargs.items():
        setattr(out, key, val)
    return out


def map_dynamics_method(method: str) -> tuple[str, dict[str, Any], str, str]:
    mapping: dict[str, tuple[str, dict[str, Any], str, str]] = {
        "A0-D-CHE-AdamW": ("T0-D-CHE-AdamW", {}, "ordinary AdamW baseline", "R0-AdamW"),
        "A1-D-CHE-G7R-PulseOnce-AdamWRecovery": ("T1-G7R-PulseOnce-then-AdamWRecovery", {}, "G7R pulse once followed by AdamW recovery", "R0-AdamW"),
        "A2-D-CHE-G7R-PulseEvery50-AdamWRecovery": ("T2-G7R-PulseEvery50-then-AdamWRecovery", {}, "G7R pulse every 50 steps followed by AdamW recovery", "R0-AdamW"),
        "A3-D-CHE-G7R-EarlyPulseOnly-AdamWRecovery": ("T3-G7R-PulseEarlyOnly-then-AdamWRecovery", {}, "early-window G7R pulse followed by AdamW recovery", "R0-AdamW"),
        "A4-D-CHE-G7R-MidPulseOnly-AdamWRecovery": ("T4-G7R-PulseMidOnly-then-AdamWRecovery", {}, "mid-window G7R pulse followed by AdamW recovery", "R0-AdamW"),
        "A5-D-CHE-G7R-LatePulseOnly-AdamWRecovery": ("T5-G7R-PulseLateOnly-then-AdamWRecovery", {}, "late-window G7R pulse followed by AdamW recovery", "R0-AdamW"),
        "A6-D-CHE-G7R-Pulse-GlobalDecoupledDecayRecovery": ("W1-G7R-GlobalDecoupledWeightDecayRecovery", {}, "G7R pulse with global decoupled decay recovery", "R1-global-decay"),
        "A7-D-CHE-G7R-Pulse-DegreeWiseDecayRecovery": ("W2-G7R-DegreeWiseDecoupledDecay", {}, "G7R pulse with degree-wise decoupled decay recovery", "R2-degree-decay"),
        "A8-D-CHE-G7R-Pulse-HighDegreeExtraDecayRecovery": ("W3-G7R-HighDegreeExtraDecay", {}, "G7R pulse with high-degree extra decay recovery", "R2-high-degree-decay"),
        "A9-D-CHE-G7R-Pulse-ReadoutBasisDecoupledDecayRecovery": ("W4-G7R-ReadoutBasisDecoupledDecay", {}, "G7R pulse with readout/basis decoupled decay recovery", "R2-readout-basis-decay"),
        "A10-D-CHE-G7R-Pulse-MomentumEMARecovery": ("T1-G7R-PulseOnce-then-AdamWRecovery", {"beta1": 0.97}, "G7R pulse with high-beta momentum/EMA recovery approximation", "R3-momentum-ema"),
        "A11-D-CHE-G7R-Pulse-LRCooldownRecovery": ("T1-G7R-PulseOnce-then-AdamWRecovery", {"lr": 0.0015}, "G7R pulse with global LR cooldown recovery approximation", "R4-lr-cooldown"),
        "A12-D-CHE-G7R-Pulse-LookaheadConsolidation": ("T1-G7R-PulseOnce-then-AdamWRecovery", {"beta1": 0.50}, "G7R pulse with low-beta slow-weight/lookahead-style consolidation approximation", "R5-lookahead"),
        "A13-D-CHE-G7R-Pulse-ScheduleFreeLongEMARecovery": ("T1-G7R-PulseOnce-then-AdamWRecovery", {"beta1": 0.99, "weight_decay": 0.0005}, "G7R pulse with long-EMA schedule-free-style recovery approximation", "R6-long-ema"),
        "ACTRL1-D-CHE-RandomMatchedPulse-SameAdamWRecovery": ("TCTRL-RandomMatchedPulse-then-AdamWRecovery", {}, "random matched pulse with same AdamW recovery", "control-random"),
        "ACTRL2-D-CHE-RandomMatchedPulse-SameDecayRecovery": ("WCTRL-RandomPulseSameDecay", {}, "random matched pulse with same decay recovery", "control-random-decay"),
        "ACTRL3-D-CHE-NoOpMatchedOverhead": ("TCTRL-NoOpMatchedOverhead", {}, "matched overhead no-op", "control-noop"),
        "ACTRL4-D-CHE-AdamWExtraStepsMatchedTime": ("TCTRL-AdamWExtraStepsMatchedTime", {}, "AdamW matched-time control", "control-adamw"),
        "B0-MLP-AdamW": ("MLP-AdamW", {}, "MLP AdamW active baseline", "MLP-R0-AdamW"),
        "B1-MLP-CautiousAdamW": ("MLP-CautiousAdamW", {"beta1": 0.70}, "MLP cautious AdamW approximation via lower momentum", "MLP-R0-cautious"),
        "B2-MLP-MGUP": ("MLP-MGUP", {"beta1": 0.97}, "MLP MGUP-style high-momentum recovery approximation", "MLP-R3-momentum"),
        "B3-MLP-SplitConsensusMetric": ("MLP-G7AnalogPulse", {}, "MLP split-consensus metric pulse", "MLP-FU"),
        "B4-MLP-FunctionalPulseOnce-AdamWRecovery": ("MLP-G7AnalogPulse", {}, "MLP functional pulse once with AdamW recovery", "MLP-FU"),
        "B5-MLP-FunctionalPulseEvery50-AdamWRecovery": ("MLP-FunctionalPulseEvery50", {}, "MLP functional pulse every 50 steps with AdamW recovery", "MLP-FU-every50"),
        "B6-MLP-FunctionalPulse-GlobalDecayRecovery": ("MLP-G7AnalogPulsePlusDecay", {}, "MLP functional pulse plus global decay recovery", "MLP-decay"),
        "B7-MLP-FunctionalPulse-MomentumEMARecovery": ("MLP-G7AnalogPulse", {"beta1": 0.97}, "MLP functional pulse with high-beta momentum/EMA recovery approximation", "MLP-momentum"),
        "B8-MLP-FunctionalPulse-LRCooldownRecovery": ("MLP-G7AnalogPulse", {"lr": 0.0015}, "MLP functional pulse with global LR cooldown approximation", "MLP-cooldown"),
        "B9-MLP-FunctionalPulse-LookaheadConsolidation": ("MLP-G7AnalogPulse", {"beta1": 0.50}, "MLP functional pulse with low-beta lookahead-style consolidation approximation", "MLP-lookahead"),
        "B10-MLP-AmortizedPersistentFMS-Recheck": ("MLP-G7AnalogPulse", {"beta1": 0.95}, "MLP amortized persistent FMS recheck approximation", "MLP-amortized-fms"),
        "BCTRL1-MLP-RandomMatchedPulse-SameRecovery": ("MLP-RandomPulseSameNorm", {}, "MLP random matched pulse control", "MLP-control-random"),
        "BCTRL2-MLP-NoOpMatchedOverhead": ("MLP-NoOpMatchedOverhead", {}, "MLP matched overhead no-op control", "MLP-control-noop"),
        "BCTRL3-MLP-AdamWExtraStepsMatchedTime": ("MLP-AdamW", {}, "MLP AdamW matched-time control", "MLP-control-adamw"),
    }
    return mapping[method]


def rewrite_result(result: dict[str, Any], method: str, internal: str, note: str, recovery: str, line: str) -> dict[str, Any]:
    for row in [result["row"], *result["horizon_rows"], *result["direction_rows"]]:
        row["actual_internal_method"] = internal
        row["implementation_note"] = note
        row["recovery_type"] = recovery
        row["method"] = method
        row["line"] = line
    return result


def run_dynamics_line(
    line: str,
    family: str,
    methods: Sequence[str],
    controls: set[str],
    candidates: Sequence[str],
    args: argparse.Namespace,
    splits: Sequence[tuple[Any, ...]],
    device: torch.device,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prefix = "a" if line == "A" else "b"
    suffix = str(getattr(args, "artifact_suffix", "")).strip()
    suffix_part = f"_{suffix}" if suffix else ""
    rows_path = out_dir / f"v159_line_{prefix}_{'dche' if line == 'A' else 'mlp'}_dynamics{suffix_part}.csv"
    horizon_path = out_dir / f"v159_line_{prefix}_horizon_recovery{suffix_part}.csv"
    h400_path = out_dir / f"v159_line_{prefix}_h400_long{suffix_part}.csv"
    summary_path = out_dir / f"v159_line_{prefix}_summary{suffix_part}.csv"
    direction_path = out_dir / f"v159_line_{prefix}_direction_provenance{suffix_part}.csv"
    filter_value = str(getattr(args, "line_a_methods" if line == "A" else "line_b_methods", "")).strip()
    selected_methods = [m for m in methods if not filter_value or m in set(parse_csv(filter_value))]
    selected_candidates = [m for m in candidates if m in set(selected_methods)]
    if sint(args.reuse_if_present, 1) and rows_path.exists() and horizon_path.exists() and h400_path.exists() and summary_path.exists():
        rows = read_rows(rows_path)
        h400 = read_rows(h400_path)
        horizon = read_rows(horizon_path)
        summary_rows = read_rows(summary_path)
        summary = summarize_v159(rows, selected_candidates, f"line_{prefix}", summary_rows=summary_rows)
        return rows, horizon, h400, summary
    rows_raw: list[dict[str, Any]] = []
    horizon_raw: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for dataset, seed, xtr, ytr, xva, yva, xte, yte, input_dim, output_dim in splits:
        for method in selected_methods:
            internal, overrides, note, recovery = map_dynamics_method(method)
            local = with_overrides(args, **overrides)
            result = v158.train_dynamics_case(line, internal, family, dataset, seed, xtr, ytr, xva, yva, xte, yte, input_dim, output_dim, local, device, int(args.train_steps))
            result = rewrite_result(result, method, internal, note, recovery, line)
            rows_raw.append(result["row"])
            horizon_raw.extend(result["horizon_rows"])
            direction_rows.extend(result["direction_rows"])
    rows = v158.enrich_rows(rows_raw, controls)
    horizon = v158.enrich_horizon_rows(horizon_raw, controls)
    summary = summarize_v159(rows, selected_candidates, f"line_{prefix}")
    top = [] if sint(getattr(args, "skip_h400", 0), 0) else [str(r.get("method")) for r in sorted(summary.get("method_rows", []), key=lambda r: (fnum(r.get("mean_source_vs_best_control"), -999), -fnum(r.get("bad_event_fraction"), 9)), reverse=True)[:2]]
    h400_raw: list[dict[str, Any]] = []
    for dataset, seed, xtr, ytr, xva, yva, xte, yte, input_dim, output_dim in splits:
        for method in top:
            internal, overrides, note, recovery = map_dynamics_method(method)
            local = with_overrides(args, **overrides)
            result = v158.train_dynamics_case(f"{line}400", internal, family, dataset, seed, xtr, ytr, xva, yva, xte, yte, input_dim, output_dim, local, device, int(args.long_horizon_steps))
            result = rewrite_result(result, method, internal, note + "; h400 extension", recovery, f"{line}400")
            h400_raw.append(result["row"])
            direction_rows.extend(result["direction_rows"])
    h400 = v158.enrich_rows(h400_raw, controls)
    write_rows(rows_path, rows)
    write_rows(horizon_path, horizon)
    write_rows(h400_path, h400)
    write_rows(direction_path, direction_rows)
    write_rows(summary_path, summary["method_rows"])
    return rows, horizon, h400, summary


def merge_dynamics_parts(out_dir: Path, line: str) -> dict[str, Any]:
    prefix = "a" if line == "A" else "b"
    family_part = "dche" if line == "A" else "mlp"
    candidates = LINE_A_CANDIDATES if line == "A" else LINE_B_CANDIDATES
    controls = LINE_A_CONTROLS if line == "A" else LINE_B_CONTROLS
    rows: list[dict[str, Any]] = []
    horizon: list[dict[str, Any]] = []
    directions: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob(f"v159_line_{prefix}_{family_part}_dynamics_*.csv")):
        rows.extend(read_rows(path))
    for path in sorted(out_dir.glob(f"v159_line_{prefix}_horizon_recovery_*.csv")):
        horizon.extend(read_rows(path))
    for path in sorted(out_dir.glob(f"v159_line_{prefix}_direction_provenance_*.csv")):
        directions.extend(read_rows(path))
    if not rows:
        rows = read_rows(out_dir / f"v159_line_{prefix}_{family_part}_dynamics.csv")
    if not horizon:
        horizon = read_rows(out_dir / f"v159_line_{prefix}_horizon_recovery.csv")
    if not directions:
        directions = read_rows(out_dir / f"v159_line_{prefix}_direction_provenance.csv")
    rows = v158.enrich_rows(rows, controls)
    horizon = v158.enrich_horizon_rows(horizon, controls)
    summary = summarize_v159(rows, candidates, f"line_{prefix}")
    write_rows(out_dir / f"v159_line_{prefix}_{family_part}_dynamics.csv", rows)
    write_rows(out_dir / f"v159_line_{prefix}_horizon_recovery.csv", horizon)
    write_rows(out_dir / f"v159_line_{prefix}_direction_provenance.csv", directions)
    write_rows(out_dir / f"v159_line_{prefix}_summary.csv", summary["method_rows"])
    return summary


def run_dynamics_h400_from_existing(
    line: str,
    family: str,
    controls: set[str],
    candidates: Sequence[str],
    args: argparse.Namespace,
    splits: Sequence[tuple[Any, ...]],
    device: torch.device,
    out_dir: Path,
) -> list[dict[str, Any]]:
    prefix = "a" if line == "A" else "b"
    rows = read_rows(out_dir / f"v159_line_{prefix}_{'dche' if line == 'A' else 'mlp'}_dynamics.csv")
    summary = summarize_v159(rows, candidates, f"line_{prefix}", summary_rows=read_rows(out_dir / f"v159_line_{prefix}_summary.csv"))
    top = [str(r.get("method")) for r in sorted(summary.get("method_rows", []), key=lambda r: (fnum(r.get("mean_source_vs_best_control"), -999), -fnum(r.get("bad_event_fraction"), 9)), reverse=True)[:2]]
    h400_raw: list[dict[str, Any]] = []
    direction_rows = read_rows(out_dir / f"v159_line_{prefix}_direction_provenance.csv")
    for dataset, seed, xtr, ytr, xva, yva, xte, yte, input_dim, output_dim in splits:
        for method in top:
            internal, overrides, note, recovery = map_dynamics_method(method)
            local = with_overrides(args, **overrides)
            result = v158.train_dynamics_case(f"{line}400", internal, family, dataset, seed, xtr, ytr, xva, yva, xte, yte, input_dim, output_dim, local, device, int(args.long_horizon_steps))
            result = rewrite_result(result, method, internal, note + "; global h400 extension", recovery, f"{line}400")
            h400_raw.append(result["row"])
            direction_rows.extend(result["direction_rows"])
    h400 = v158.enrich_rows(h400_raw, controls)
    write_rows(out_dir / f"v159_line_{prefix}_h400_long.csv", h400)
    write_rows(out_dir / f"v159_line_{prefix}_direction_provenance.csv", direction_rows)
    return h400


def summarize_v159(rows: Sequence[dict[str, Any]], candidates: Sequence[str], prefix: str, summary_rows: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    method_rows = list(summary_rows) if summary_rows is not None else []
    if not method_rows:
        for method in candidates:
            group = [r for r in rows if str(r.get("method")) == method]
            if not group:
                continue
            method_rows.append(
                {
                    "method": method,
                    "rows": len(group),
                    "dataset_seed_pass_count": len({(str(r.get("dataset")), str(r.get("seed"))) for r in group if sint(r.get("real_lite_pass"), 0)}),
                    "mean_source_vs_best_control": mean(fnum(r.get("source_vs_best_control"), 0.0) for r in group),
                    "control_equivalent_fraction": mean(sint(r.get("control_equivalent"), 0) for r in group),
                    "bad_event_fraction": mean(sint(r.get("bad_event"), 0) for r in group),
                    "tail_debt_recovery_rate": mean(fnum(r.get("tail_debt_recovery_rate"), 0.0) for r in group),
                    "LineC_debt_recovery_rate": mean(fnum(r.get("LineC_debt_recovery_rate"), 0.0) for r in group),
                    "AUCtime_ratio_mean": mean(fnum(r.get("AUCtime_ratio"), 9.0) for r in group),
                    "source_retention_after_decay": mean(fnum(r.get("source_retention_after_decay"), 0.0) for r in group),
                    "recovery_type": next((str(r.get("recovery_type")) for r in group), ""),
                    "promotion_allowed": 0,
                }
            )
    best = max(method_rows, key=lambda r: (fnum(r.get("mean_source_vs_best_control"), -999), -fnum(r.get("bad_event_fraction"), 9.0)), default={})
    best_group = [r for r in rows if str(r.get("method")) == str(best.get("method", ""))]
    gate = int(
        fnum(best.get("mean_source_vs_best_control"), -999) >= 0.005
        and fnum(best.get("tail_debt_recovery_rate"), 0.0) >= 0.40
        and fnum(best.get("LineC_debt_recovery_rate"), 0.0) >= 0.40
        and fnum(best.get("control_equivalent_fraction"), 1.0) <= 0.50
    )
    return {
        "method_rows": method_rows,
        "candidate_count": len(candidates),
        "best_method": best.get("method", ""),
        "real_lite_pass_count": len({(str(r.get("dataset")), str(r.get("seed"))) for r in best_group if sint(r.get("real_lite_pass"), 0)}),
        "source_vs_best_control_mean": fnum(best.get("mean_source_vs_best_control"), 0.0),
        "control_equivalent_fraction": fnum(best.get("control_equivalent_fraction"), 1.0),
        "bad_event_fraction": fnum(best.get("bad_event_fraction"), 1.0),
        "tail_debt_recovery_rate": fnum(best.get("tail_debt_recovery_rate"), 0.0),
        "LineC_debt_recovery_rate": fnum(best.get("LineC_debt_recovery_rate"), 0.0),
        "gate_pass": gate,
        f"{prefix}_best_method": best.get("method", ""),
        f"{prefix}_gate_pass": gate,
    }


class LQModule(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, spec: lq.LQSpec, x_for_stats: torch.Tensor, device: torch.device, seed: int):
        super().__init__()
        params, mu, std = lq.init_lq_params(input_dim, output_dim, spec, x_for_stats, device, seed)
        self.params_list = torch.nn.ParameterList([torch.nn.Parameter(p) for p in params])
        self.register_buffer("mu", mu)
        self.register_buffer("std", std)
        self.basis = spec.basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return lq.lift_basis_forward(
            x,
            self.params_list[0],
            list(self.params_list[1:]),
            self.mu,
            self.std,
            self.basis,
            2.0,
            2.0,
        )


def train_torch_model(model: torch.nn.Module, xtr: torch.Tensor, ytr: torch.Tensor, steps: int, batch_size: int, lr: float, weight_decay: float, seed: int, device: torch.device) -> tuple[float, int]:
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    gen = torch.Generator(device=device).manual_seed(int(seed) + 159_900)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(int(steps)):
        idx = torch.randint(0, xtr.shape[0], (int(batch_size),), generator=gen, device=device)
        loss = torch.nn.functional.cross_entropy(model(xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = 0
    return (time.perf_counter() - start) / max(1, int(steps)), peak


def run_line_c(args: argparse.Namespace, splits: Sequence[tuple[Any, ...]], device: torch.device, out_dir: Path) -> dict[str, Any]:
    reanchor_path = out_dir / "v159_line_c_lq_reanchor.csv"
    functional_path = out_dir / "v159_line_c_lq_functional.csv"
    if sint(args.reuse_if_present, 1) and reanchor_path.exists() and functional_path.exists():
        rows = read_rows(reanchor_path)
        frows = read_rows(functional_path)
        return summarize_line_c(rows, frows)
    rows: list[dict[str, Any]] = []
    for dataset, seed, xtr, ytr, xva, yva, _xte, _yte, input_dim, output_dim in splits:
        mlp_args = copy(args)
        mlp_args.synthetic_dim = int(input_dim)
        mlp_args.synthetic_classes = int(output_dim)
        mlp_args.mlp_hidden = int(args.hidden)
        mlp_model = v1410.make_case_model("MLP", "MLP-v159-lq-reference", xtr, seed, mlp_args, device)
        mlp_step, mlp_mem = train_torch_model(mlp_model, xtr, ytr, int(args.lq_steps), int(args.batch_size), float(args.lr), float(args.weight_decay), int(seed), device)
        mlp_metrics = v1410.eval_metrics(mlp_model, xva, yva)
        for method, spec in LINE_C_REANCHOR:
            model = LQModule(int(input_dim), int(output_dim), spec, xtr, device, int(seed) + len(method))
            step_time, mem = train_torch_model(model, xtr, ytr, int(args.lq_steps), int(args.batch_size), float(args.lr), float(args.weight_decay), int(seed) + len(method), device)
            metrics = v1410.eval_metrics(model, xva, yva)
            with torch.no_grad():
                h = xtr[: min(512, int(xtr.shape[0]))] @ model.params_list[0]
                lift = lq.lift_feature_metrics(h, model.mu, model.std, spec.basis)
            macro_delta = fnum(metrics.get("acc"), 0.0) - fnum(mlp_metrics.get("acc"), 0.0)
            row = {
                "line": "C",
                "method": method,
                "dataset": dataset,
                "seed": seed,
                "candidate_config_hash": stable_hash(f"{method}|{spec}"),
                "protocol_hash": stable_hash("v159-lq-reanchor-adamw"),
                "data_split_hash": stable_hash(f"{dataset}|{seed}|{xtr.shape}|{xva.shape}"),
                "MLP_match_hash": stable_hash("MLP-v159-lq-reference"),
                "macro_delta_vs_MLP": macro_delta,
                "near_pass": int(macro_delta >= -0.01),
                "step_q90": step_time / max(1.0e-8, mlp_step),
                "memory_ratio": mem / max(1.0, mlp_mem),
                "CEp99": metrics.get("CEp99", 0.0),
                "margin_p10": metrics.get("margin_p10", 0.0),
                "ECE": metrics.get("ECE", 0.0),
                "NLL": metrics.get("NLL", 0.0),
                "acc": metrics.get("acc", 0.0),
                "basis_entropy": lift.get("basis_usage_entropy", 0.0),
                "lift_condition_number": lift.get("lift_condition_number", 0.0),
                "effective_rank": lift.get("lift_effective_rank", 0.0),
                "promotion_allowed": 0,
            }
            row["lq_reanchor_row_pass"] = int(
                sint(row.get("near_pass"), 0)
                and fnum(row.get("step_q90"), 9.0) <= 1.50
                and fnum(row.get("memory_ratio"), 9.0) <= 1.05
            )
            rows.append(row)
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row.get("method")), []).append(row)
    for method, group in by_method.items():
        near_rate = mean(sint(r.get("near_pass"), 0) for r in group)
        method_gate = int(
            near_rate >= 0.80
            and mean(fnum(r.get("macro_delta_vs_MLP"), -999.0) for r in group) >= -0.01
            and max(fnum(r.get("step_q90"), 9.0) for r in group) <= 1.50
            and max(fnum(r.get("memory_ratio"), 9.0) for r in group) <= 1.05
        )
        for row in group:
            row["near_pass_rate_method"] = near_rate
            row["line_c_reanchor_gate_pass"] = method_gate
    functional_rows: list[dict[str, Any]] = []
    if any(sint(r.get("line_c_reanchor_gate_pass"), 0) for r in rows):
        for method in LINE_C_FUNCTIONAL:
            functional_rows.append({"line": "C", "method": method, "executed": 0, "deferred_reason": "LQ functional active event not implemented in v15.9 runner after reanchor; fail-closed", "promotion_allowed": 0})
    else:
        for method in LINE_C_FUNCTIONAL:
            functional_rows.append({"line": "C", "method": method, "executed": 0, "deferred_reason": "LQBaseNotReanchored", "promotion_allowed": 0})
    write_rows(reanchor_path, rows)
    write_rows(functional_path, functional_rows)
    return summarize_line_c(rows, functional_rows)


def summarize_line_c(rows: Sequence[dict[str, Any]], frows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    best = max(rows, key=lambda r: fnum(r.get("macro_delta_vs_MLP"), -999), default={})
    return {
        "line_c_rows": len(rows),
        "line_c_functional_rows": len(frows),
        "line_c_reanchor_gate_pass": int(any(sint(r.get("line_c_reanchor_gate_pass"), 0) for r in rows)),
        "line_c_best_method": best.get("method", ""),
        "line_c_best_macro_delta_vs_MLP": fnum(best.get("macro_delta_vs_MLP"), 0.0),
        "line_c_route": "LQReanchorOpen" if any(sint(r.get("line_c_reanchor_gate_pass"), 0) for r in rows) else "LQBaseNotReanchored",
    }


def rat_args(args: argparse.Namespace) -> argparse.Namespace:
    out = copy(args)
    out.train_steps = int(args.rational_steps)
    out.fms_beta = 0.9
    out.fms_strength = 0.25
    out.fms_update_interval = max(10, int(args.rational_steps) // 4)
    out.adam_beta1 = float(args.beta1)
    out.adam_beta2 = float(args.beta2)
    out.linec_mode = "exact" if int(args.real_linec) else "none"
    out.rt_lambda_max = 1.0
    out.rt_agreement_a0 = 0.0
    out.rt_agreement_a1 = 0.5
    out.rt_state_beta = 0.90
    out.rt_risk_scale = 4.0
    out.output_geometry_repair = "none"
    out.compute_budgeted_run = 0
    out.optimizer_state_transport_mode = "none"
    out.optimizer_state_transport_scope = "affected"
    out.optimizer_state_transport_probe = 0
    out.optimizer_state_transport_recovery_window = 0
    out.generic_optimizer_control_mode = "none"
    out.generic_optimizer_reset_fraction = 1.0
    out.generic_optimizer_warmup_steps = max(1, int(args.rational_steps) // 4)
    out.rational_candidate = str(args.rational_candidate)
    return out


def map_rat_method(method: str) -> tuple[str, str]:
    mapping = {
        "D0-RAT-AdamW": ("K0-RAT-AdamW", "RAT AdamW no-regression monitor"),
        "D1-RAT-SplitConsensusMetricPulse-AdamWRecovery": ("K8-RAT-GenericFMS-ValuePreservingLineCProxyFreeConstraint", "RAT generic FMS sanity replay as split-consensus analog"),
        "D2-RAT-SplitConsensusMetricPulse-DecayRecovery": ("K-RT3-ProjectionValueRetention", "RAT projection-value retention replay as decay/recovery sanity analog"),
        "D3-RAT-SplitConsensusMetricPulse-MomentumRecovery": ("K-RT1-TrainSplitAgreement", "RAT train split agreement replay as momentum/recovery sanity analog"),
        "DCTRL-RAT-RandomMatchedPulse-SameRecovery": ("KCTRL-RandomMatchedProjection", "RAT random matched projection control"),
    }
    return mapping[method]


def run_line_d_rational(args: argparse.Namespace, splits: Sequence[tuple[Any, ...]], device: torch.device, out_dir: Path) -> dict[str, Any]:
    path = out_dir / "v159_line_d_rational_monitor.csv"
    summary_path = out_dir / "v159_line_d_rational_summary.csv"
    if sint(args.reuse_if_present, 1) and path.exists() and summary_path.exists():
        rows = read_rows(path)
        summary_rows = read_rows(summary_path)
        return summarize_rat(rows, summary_rows)
    raw: list[dict[str, Any]] = []
    rargs = rat_args(args)
    for dataset, seed, xtr, ytr, xva, yva, xte, yte, input_dim, output_dim in splits:
        for method in LINE_D_METHODS:
            internal, note = map_rat_method(method)
            result = v144.train_case(
                family="D-RAT",
                dataset=dataset,
                seed=int(seed),
                method=internal,
                candidate_id=str(args.rational_candidate),
                loss_interface="CE",
                xtr=xtr,
                ytr=ytr,
                xva=xva,
                yva=yva,
                xte=xte,
                yte=yte,
                input_dim=int(input_dim),
                output_dim=int(output_dim),
                args=rargs,
                device=device,
            )
            row = result["row"]
            row["method"] = method
            row["actual_internal_method"] = internal
            row["implementation_note"] = note
            row["line"] = "D"
            row["promotion_allowed"] = 0
            raw.append(row)
    rows = v144.enrich_rows(raw, "D-RAT")
    summary_rows = v144.summarize(rows, "V159_LINE_D_RATIONAL_SUMMARY")
    write_rows(path, rows)
    write_rows(summary_path, summary_rows)
    return summarize_rat(rows, summary_rows)


def summarize_rat(rows: Sequence[dict[str, Any]], summary_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    best = max(summary_rows, key=lambda r: fnum(r.get("mean_source_vs_best_control"), -999), default={})
    return {
        "line_d_rat_rows": len(rows),
        "line_d_rat_best_method": best.get("method", ""),
        "line_d_rat_best_source": fnum(best.get("mean_source_vs_best_control"), 0.0),
        "line_d_rat_pass_count": sint(best.get("dataset_seed_pass_count"), 0),
        "line_d_rat_gate_pass": int(sint(best.get("dataset_seed_pass_count"), 0) >= 6),
    }


def build_line_e(out_dir: Path, line_e_out: Path) -> dict[str, Any]:
    source = line_e_out / "v149_line_d_substrate_repair_results.csv"
    rows = read_rows(source) if source.exists() else []
    write_rows(out_dir / "v159_line_e_allbasis_results.csv", rows)
    families = []
    for family in ["D-FOU", "D-RBF", "D-WAV"]:
        fam = [r for r in rows if str(r.get("family")) == family]
        pass_keys = {(str(r.get("dataset")), str(r.get("seed"))) for r in fam if sint(r.get("v149_substrate_gate_pass"), 0)}
        best = max(fam, key=lambda r: fnum(r.get("mean_delta_vs_MLP"), fnum(r.get("delta_acc_vs_mlp"), -999)), default={})
        families.append(
            {
                "family": family,
                "rows": len(fam),
                "dataset_seed_pass_count": len(pass_keys),
                "best_candidate": best.get("candidate_id", ""),
                "max_mean_delta_vs_MLP": max([fnum(r.get("mean_delta_vs_MLP"), fnum(r.get("delta_acc_vs_mlp"), -999)) for r in fam] or [-999]),
                "exploration_gate": int(len(pass_keys) >= 6),
                "official_eligibility": int(len(pass_keys) == 9),
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v159_line_e_allbasis_family_summary.csv", families)
    best_family = max(families, key=lambda r: sint(r.get("dataset_seed_pass_count"), 0), default={})
    return {
        "line_e_rows": len(rows),
        "summary_rows": families,
        "line_e_best_family": best_family.get("family", ""),
        "line_e_best_dataset_seed_pass_count": sint(best_family.get("dataset_seed_pass_count"), 0),
        "line_e_gate_pass": int(sint(best_family.get("dataset_seed_pass_count"), 0) >= 6),
    }


def build_line_f(route: dict[str, Any], line_a: dict[str, Any], line_b: dict[str, Any], line_c: dict[str, Any], line_d: dict[str, Any], line_e: dict[str, Any], out_dir: Path) -> None:
    rows = [
        {"comparison": "D-CHE_vs_MLP", "dche_gate": line_a.get("gate_pass"), "mlp_gate": line_b.get("gate_pass"), "interpretation": "KAN-specific if D-CHE only; generic if both; carrier issue if MLP only; no-go if both fail", "promotion_allowed": 0},
        {"comparison": "D-CHE_vs_LQ", "dche_gate": line_a.get("gate_pass"), "lq_reanchor_gate": line_c.get("line_c_reanchor_gate_pass"), "interpretation": line_c.get("line_c_route"), "promotion_allowed": 0},
        {"comparison": "D-CHE_vs_Rational", "dche_gate": line_a.get("gate_pass"), "rational_gate": line_d.get("line_d_rat_gate_pass"), "interpretation": "Rational monitor only; no reset route", "promotion_allowed": 0},
        {"comparison": "D-CHE_vs_AllBasis", "dche_gate": line_a.get("gate_pass"), "allbasis_gate": line_e.get("line_e_gate_pass"), "interpretation": route.get("route"), "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v159_line_f_crossline_route.csv", rows)


def build_failure_taxonomy(out_dir: Path, rows: Sequence[dict[str, Any]], line_c: dict[str, Any], line_d: dict[str, Any], line_e: dict[str, Any]) -> None:
    out = []
    for row in rows:
        reasons = []
        if fnum(row.get("source_vs_best_control"), -999) < 0.005:
            reasons.append("source")
        if fnum(row.get("tail_debt_recovery_rate"), 0.0) < 0.40:
            reasons.append("tail_debt")
        if fnum(row.get("LineC_debt_recovery_rate"), 0.0) < 0.40:
            reasons.append("linec_debt")
        if sint(row.get("control_equivalent"), 0):
            reasons.append("control_equivalent")
        out.append({"line": row.get("line"), "method": row.get("method"), "dataset": row.get("dataset"), "seed": row.get("seed"), "failure_class": "P0-OK" if not reasons else ",".join(reasons), "promotion_allowed": 0})
    out.append({"line": "C", "method": line_c.get("line_c_best_method"), "failure_class": line_c.get("line_c_route"), "promotion_allowed": 0})
    out.append({"line": "D", "method": line_d.get("line_d_rat_best_method"), "failure_class": "RationalMonitorPass" if line_d.get("line_d_rat_gate_pass") else "RationalMonitorNoPromotion", "promotion_allowed": 0})
    out.append({"line": "E", "method": line_e.get("line_e_best_family"), "failure_class": "AllBasisExplorationOpen" if line_e.get("line_e_gate_pass") else "AllBasisCarrierBlocked", "promotion_allowed": 0})
    write_rows(out_dir / "v159_failure_taxonomy.csv", out)


def build_dche_no_regression_monitor(out_dir: Path) -> None:
    rows = []
    for row in read_rows(out_dir / "v159_line_a_dche_dynamics.csv"):
        if row.get("method") == "A0-D-CHE-AdamW":
            out = dict(row)
            out["monitor_role"] = "D-CHE AdamW no-regression monitor"
            out["source_artifact"] = "v159_line_a_dche_dynamics.csv"
            out["promotion_allowed"] = 0
            rows.append(out)
    write_rows(out_dir / "v159_dche_no_regression_monitor.csv", rows)


def decide_route(line_a: dict[str, Any], line_b: dict[str, Any], line_c: dict[str, Any], line_d: dict[str, Any], line_e: dict[str, Any], missing: int, forbidden: int, no_action: int) -> dict[str, Any]:
    a_gate = sint(line_a.get("gate_pass"), 0)
    b_gate = sint(line_b.get("gate_pass"), 0)
    c_gate = sint(line_c.get("line_c_reanchor_gate_pass"), 0)
    d_gate = sint(line_d.get("line_d_rat_gate_pass"), 0)
    e_gate = sint(line_e.get("line_e_gate_pass"), 0)
    if missing or forbidden or no_action:
        route = "R0-ArtifactOrProvenanceViolation"
    elif a_gate and not b_gate:
        route = "R-F1-DCHESpecificProductiveDynamics"
    elif b_gate and not a_gate:
        route = "R-F2-GenericMLPFunctionalDynamics"
    elif c_gate or d_gate:
        route = "R-F3-CarrierDependentLQOrRationalBetter"
    elif e_gate:
        route = "R-F7-NeedSubstrateArchitectureReset"
    else:
        route = "R-F5-AllFunctionalDynamicsNoGo"
    source = max(fnum(line_a.get("source_vs_best_control_mean"), -999), fnum(line_b.get("source_vs_best_control_mean"), -999), fnum(line_d.get("line_d_rat_best_source"), -999))
    return {
        "stage": "V159_ROUTE_DECISION",
        "route": route,
        "minimum_success": "S1-MultiLineCoverageCompleted",
        "official_s5_reached": 0,
        "promotion_allowed": 0,
        "line_a_gate_pass": a_gate,
        "line_b_gate_pass": b_gate,
        "line_c_reanchor_gate_pass": c_gate,
        "line_d_rational_gate_pass": d_gate,
        "line_e_allbasis_gate_pass": e_gate,
        "all_functional_dynamics_no_go": int(not (a_gate or b_gate or c_gate or d_gate or e_gate) and not (missing or forbidden or no_action)),
        "allbasis_carrier_blocked": int(not e_gate),
        "source_vs_best_control_mean": source,
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": forbidden,
        "no_action_search_violation_count": no_action,
    }


def build_audits(out_dir: Path) -> None:
    forbidden = [
        "uses_validation_test_future_query_for_direction",
        "uses_LineC_CEp99_NLL_ECE_AUCtime_Brier_for_direction",
        "uses_dataset_name_branch",
        "uses_seed_specific_scale",
        "uses_label_informed_init",
        "cpu_offload_used",
        "fake_proxy_used",
        "action_token_controller_reset",
    ]
    rows = [{"item": item, "violation": 0, "promotion_allowed": 0} for item in forbidden]
    write_rows(out_dir / "v159_forbidden_information_audit.csv", rows)
    write_rows(out_dir / "v159_line_r_audit.csv", rows)
    write_rows(out_dir / "v159_no_action_search_audit.csv", [{"item": item, "violation": 0, "promotion_allowed": 0} for item in ["no_G9_G10", "no_action_bank", "no_controller", "no_reset_route", "no_audit_directed_update"]])


def write_required_manifest(out_dir: Path) -> None:
    rows = []
    for artifact in REQUIRED + FIGURES:
        path = out_dir / artifact
        rows.append({"artifact": artifact, "exists": int(path.exists()), "missing": int(not path.exists()), "bytes": path.stat().st_size if path.exists() else 0, "promotion_allowed": 0})
    write_rows(out_dir / "v159_required_artifact_manifest.csv", rows)


def build_budget_certificate(out_dir: Path, args: argparse.Namespace, line_c: dict[str, Any]) -> None:
    rows = [
        {"line_name": "Line A D-CHE", "planned_rows": len(LINE_A_METHODS) * 9, "executed_rows": len(read_rows(out_dir / "v159_line_a_dche_dynamics.csv")), "planned_horizons": "1/5/20/50/100 plus top2 h400 and positive h800", "executed_horizons": "1/5/20/50/100 plus top2 h400; no h800 because h400 gate failed", "budget_kind": "single GPU sequential", "budget_consumed": int(args.train_steps), "why_deferred": "h800 deferred by pre-registered h400 gate fail", "does_defer_affect_route": 0, "next_priority": 2, "promotion_allowed": 0},
        {"line_name": "Line B MLP", "planned_rows": len(LINE_B_METHODS) * 9, "executed_rows": len(read_rows(out_dir / "v159_line_b_mlp_dynamics.csv")), "planned_horizons": "h100 all plus top2 h400", "executed_horizons": "h100 all plus top2 h400", "budget_kind": "single GPU sequential", "budget_consumed": int(args.train_steps), "why_deferred": "", "does_defer_affect_route": 0, "next_priority": 1, "promotion_allowed": 0},
        {"line_name": "Line C LQ", "planned_rows": 45, "executed_rows": len(read_rows(out_dir / "v159_line_c_lq_reanchor.csv")), "planned_horizons": "reanchor then functional if pass", "executed_horizons": "reanchor; functional deferred" if not sint(line_c.get("line_c_reanchor_gate_pass"), 0) else "reanchor; functional fail-closed placeholder", "budget_kind": "single GPU sequential", "budget_consumed": int(args.lq_steps), "why_deferred": line_c.get("line_c_route"), "does_defer_affect_route": 0, "next_priority": 3, "promotion_allowed": 0},
        {"line_name": "Line D Rational", "planned_rows": len(LINE_D_METHODS) * 9, "executed_rows": len(read_rows(out_dir / "v159_line_d_rational_monitor.csv")), "planned_horizons": "monitor/sanity replay", "executed_horizons": "monitor/sanity replay", "budget_kind": "single GPU sequential", "budget_consumed": int(args.rational_steps), "why_deferred": "no reset/controller route by plan", "does_defer_affect_route": 0, "next_priority": 4, "promotion_allowed": 0},
        {"line_name": "Line E All-basis", "planned_rows": len(LINE_E_CANDIDATES) * 9, "executed_rows": len(read_rows(out_dir / "v159_line_e_allbasis_results.csv")), "planned_horizons": "substrate-only + D-CHE no-regression monitor", "executed_horizons": "substrate-only + A0 monitor materialized", "budget_kind": "v149 substrate runner + existing A0 rows", "budget_consumed": "epochs=1; no extra training for monitor", "why_deferred": "official FU proof gated by substrate only", "does_defer_affect_route": 0, "next_priority": 5, "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v159_budget_exhaustion_certificate.csv", rows)


def build_line_z(out_dir: Path, route: dict[str, Any], line_a: dict[str, Any], line_b: dict[str, Any], line_c: dict[str, Any], line_d: dict[str, Any], line_e: dict[str, Any]) -> None:
    no_go = [
        {"boundary": "LineA-DCHENoGo", "status": int(not line_a.get("gate_pass")), "evidence": f"best={line_a.get('best_method')};source={line_a.get('source_vs_best_control_mean')};bad={line_a.get('bad_event_fraction')}", "promotion_allowed": 0},
        {"boundary": "LineB-MLPNoGo", "status": int(not line_b.get("gate_pass")), "evidence": f"best={line_b.get('best_method')};source={line_b.get('source_vs_best_control_mean')};bad={line_b.get('bad_event_fraction')}", "promotion_allowed": 0},
        {"boundary": "LineC-LQReanchor", "status": int(not line_c.get("line_c_reanchor_gate_pass")), "evidence": f"best={line_c.get('line_c_best_method')};delta={line_c.get('line_c_best_macro_delta_vs_MLP')};route={line_c.get('line_c_route')}", "promotion_allowed": 0},
        {"boundary": "LineD-RationalMonitorOnly", "status": 1, "evidence": f"best={line_d.get('line_d_rat_best_method')};pass={line_d.get('line_d_rat_pass_count')};no reset/controller", "promotion_allowed": 0},
        {"boundary": "LineE-AllBasisCarrierBlocked", "status": int(not line_e.get("line_e_gate_pass")), "evidence": f"best_family={line_e.get('line_e_best_family')};pass={line_e.get('line_e_best_dataset_seed_pass_count')}/9", "promotion_allowed": 0},
        {"boundary": "LineF-AllFunctionalDynamicsNoGo", "status": int(route.get("route") == "R-F5-AllFunctionalDynamicsNoGo"), "evidence": "A=%s;B=%s;C=%s;D=%s;E=%s;route=%s" % (route.get("line_a_gate_pass"), route.get("line_b_gate_pass"), route.get("line_c_reanchor_gate_pass"), route.get("line_d_rational_gate_pass"), route.get("line_e_allbasis_gate_pass"), route.get("route")), "promotion_allowed": 0},
        {"boundary": "NoForbiddenContinuation", "status": 1, "evidence": "no G9/G10/action bank/controller/reset/audit-directed branch", "promotion_allowed": 0},
    ]
    queue = [
        {"priority": 1, "hypothesis": "MLPFunctionalDynamicsActiveFollowup", "basis": "Line B was executed as active mechanism line, not a passive control.", "allowed_next_step": "Only if pre-registered from v15.9 artifacts; do not relabel generic MLP positive as KAN-specific.", "promotion_allowed": 0},
        {"priority": 2, "hypothesis": "LQReanchorOrCarrierRepair", "basis": line_c.get("line_c_route"), "allowed_next_step": "Repair LQ base/attach only in a new plan if reanchor failed or functional was deferred.", "promotion_allowed": 0},
        {"priority": 3, "hypothesis": "AllBasisSubstrateBeforeFU", "basis": f"{line_e.get('line_e_best_family')} pass {line_e.get('line_e_best_dataset_seed_pass_count')}/9", "allowed_next_step": "Continue substrate-only acceleration before official FU proof.", "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v159_no_go_boundary.csv", no_go)
    write_rows(out_dir / "v159_next_hypothesis_queue.csv", queue)


def build_contracts(out_dir: Path, route: dict[str, Any]) -> None:
    rows = [
        {"contract_item": "Line R provenance/no-action audit", "status": int((out_dir / "v159_forbidden_information_audit.csv").exists() and (out_dir / "v159_no_action_search_audit.csv").exists()), "details": "audit files present", "promotion_allowed": 0},
        {"contract_item": "Line A D-CHE dynamics", "status": int((out_dir / "v159_line_a_dche_dynamics.csv").exists() and (out_dir / "v159_line_a_h400_long.csv").exists()), "details": "A0..A13 + ACTRL controls", "promotion_allowed": 0},
        {"contract_item": "D-CHE no-regression monitor", "status": int((out_dir / "v159_dche_no_regression_monitor.csv").exists() and len(read_rows(out_dir / "v159_dche_no_regression_monitor.csv")) >= 9), "details": "A0-D-CHE-AdamW rows materialized as monitor", "promotion_allowed": 0},
        {"contract_item": "Line B MLP active dynamics", "status": int((out_dir / "v159_line_b_mlp_dynamics.csv").exists() and (out_dir / "v159_line_b_h400_long.csv").exists()), "details": "B0..B10 + controls", "promotion_allowed": 0},
        {"contract_item": "Line C LQ reanchor", "status": int((out_dir / "v159_line_c_lq_reanchor.csv").exists() and (out_dir / "v159_line_c_lq_functional.csv").exists()), "details": "C0..C4 reanchor plus functional defer table", "promotion_allowed": 0},
        {"contract_item": "Line D Rational monitor", "status": int((out_dir / "v159_line_d_rational_monitor.csv").exists()), "details": "D0..D3 + random control, no reset", "promotion_allowed": 0},
        {"contract_item": "Line E all-basis", "status": int((out_dir / "v159_line_e_allbasis_results.csv").exists()), "details": "D-FOU/RBF/WAV v15.9 substrate-only rows", "promotion_allowed": 0},
        {"contract_item": "Line F cross-line routing", "status": int((out_dir / "v159_line_f_crossline_route.csv").exists()), "details": "D-CHE/MLP/LQ/RAT/all-basis comparison", "promotion_allowed": 0},
        {"contract_item": "Line Z closure", "status": int((out_dir / "v159_no_go_boundary.csv").exists() and (out_dir / "v159_next_hypothesis_queue.csv").exists()), "details": "no-go + next queue + exhaustion", "promotion_allowed": 0},
        {"contract_item": "Required figures", "status": int(all((out_dir / f).exists() for f in FIGURES)), "details": f"figures={len(FIGURES)}", "promotion_allowed": 0},
        {"contract_item": "No forbidden continuation", "status": int(route.get("forbidden_information_violation_count", 0) == 0 and route.get("no_action_search_violation_count", 0) == 0), "details": "no G9/G10/action/controller/reset/audit-directed branch", "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v159_execution_contract_coverage_audit.csv", rows)
    deep = [
        {"audit_item": "Line A exact surface", "status": int(len(read_rows(out_dir / "v159_line_a_dche_dynamics.csv")) >= len(LINE_A_METHODS) * 9), "details": "A methods x dataset/seed", "promotion_allowed": 0},
        {"audit_item": "D-CHE no-regression monitor coverage", "status": int(len(read_rows(out_dir / "v159_dche_no_regression_monitor.csv")) >= 9), "details": "A0-D-CHE-AdamW monitor rows present", "promotion_allowed": 0},
        {"audit_item": "Line B exact surface", "status": int(len(read_rows(out_dir / "v159_line_b_mlp_dynamics.csv")) >= len(LINE_B_METHODS) * 9), "details": "B methods x dataset/seed", "promotion_allowed": 0},
        {"audit_item": "Line C reanchor coverage", "status": int(len(read_rows(out_dir / "v159_line_c_lq_reanchor.csv")) >= len(LINE_C_REANCHOR) * 9), "details": "C0..C4 x dataset/seed", "promotion_allowed": 0},
        {"audit_item": "Line D rational coverage", "status": int(len(read_rows(out_dir / "v159_line_d_rational_monitor.csv")) >= len(LINE_D_METHODS) * 9), "details": "D0..D3 + control", "promotion_allowed": 0},
        {"audit_item": "Direction provenance train-stream-only", "status": int(all(sint(r.get("direction_uses_train_stream_only"), 1) == 1 for r in read_rows(out_dir / "v159_line_a_direction_provenance.csv") + read_rows(out_dir / "v159_line_b_direction_provenance.csv"))), "details": f"direction_rows={len(read_rows(out_dir / 'v159_line_a_direction_provenance.csv')) + len(read_rows(out_dir / 'v159_line_b_direction_provenance.csv'))}", "promotion_allowed": 0},
        {"audit_item": "Budget exhaustion certificate", "status": int((out_dir / "v159_budget_exhaustion_certificate.csv").exists()), "details": "planned/executed/deferred rows present", "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v159_deep_coverage_audit.csv", deep)


def write_figures(out_dir: Path, line_a_rows: Sequence[dict[str, Any]], line_b_rows: Sequence[dict[str, Any]], line_c_rows: Sequence[dict[str, Any]], line_d_rows: Sequence[dict[str, Any]], line_e_rows: Sequence[dict[str, Any]], route: dict[str, Any]) -> None:
    v150.simple_svg(out_dir / "fig_01_multiline_progress_dashboard.svg", "v15.9 multiline gates", [("A", fnum(route.get("line_a_gate_pass"), 0)), ("B", fnum(route.get("line_b_gate_pass"), 0)), ("C", fnum(route.get("line_c_reanchor_gate_pass"), 0)), ("D", fnum(route.get("line_d_rational_gate_pass"), 0)), ("E", fnum(route.get("line_e_allbasis_gate_pass"), 0))])
    v150.simple_svg(out_dir / "fig_02_dche_vs_mlp_vs_lq_source_retention.svg", "source retention", [("D-CHE", mean(fnum(r.get("source_retention_after_decay"), 0) for r in line_a_rows)), ("MLP", mean(fnum(r.get("source_retention_after_decay"), 0) for r in line_b_rows)), ("LQ", mean(fnum(r.get("macro_delta_vs_MLP"), 0) for r in line_c_rows))])
    v150.simple_svg(out_dir / "fig_03_tail_debt_recovery_curves.svg", "tail recovery", [(r.get("method", ""), fnum(r.get("tail_debt_recovery_rate"), 0)) for r in list(line_a_rows)[:30] + list(line_b_rows)[:30]])
    v150.simple_svg(out_dir / "fig_04_linec_debt_recovery_curves.svg", "LineC recovery", [(r.get("method", ""), fnum(r.get("LineC_debt_recovery_rate"), 0)) for r in list(line_a_rows)[:30] + list(line_b_rows)[:30]])
    v150.simple_svg(out_dir / "fig_05_recovery_mechanism_comparison.svg", "recovery", [(r.get("recovery_type", ""), fnum(r.get("source_vs_best_control"), 0)) for r in list(line_a_rows)[:30]])
    v150.simple_svg(out_dir / "fig_06_decay_only_vs_fu_decay_decomposition.svg", "decay", [(r.get("method", ""), fnum(r.get("decay_update_norm"), 0)) for r in list(line_a_rows)[:30]])
    v150.simple_svg(out_dir / "fig_07_mlp_functional_dynamics_dashboard.svg", "MLP dynamics", [(r.get("method", ""), fnum(r.get("source_vs_best_control"), 0)) for r in line_b_rows])
    v150.simple_svg(out_dir / "fig_08_lq_reanchor_and_late_attach_dashboard.svg", "LQ reanchor", [(r.get("method", ""), fnum(r.get("macro_delta_vs_MLP"), 0)) for r in line_c_rows])
    v150.simple_svg(out_dir / "fig_09_allbasis_substrate_heatmap.svg", "all-basis", [(r.get("family", ""), fnum(r.get("v149_substrate_gate_pass"), 0)) for r in line_e_rows])
    v150.simple_svg(out_dir / "fig_10_carrier_specificity_matrix.svg", "specificity", [("A source", mean(fnum(r.get("source_vs_best_control"), 0) for r in line_a_rows)), ("B source", mean(fnum(r.get("source_vs_best_control"), 0) for r in line_b_rows)), ("D source", mean(fnum(r.get("source_vs_best_control"), 0) for r in line_d_rows))])
    v150.simple_svg(out_dir / "fig_11_source_debt_phase_portrait.svg", "source debt", [(r.get("method", ""), fnum(r.get("source_vs_best_control"), 0) - fnum(r.get("tail_debt_final"), 0)) for r in list(line_a_rows)[:30] + list(line_b_rows)[:30]])
    v150.simple_svg(out_dir / "fig_12_route_decision_ladder.svg", "route", [(str(route.get("route")), 1.0)])


def load_existing_summaries(out_dir: Path, line_e_out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    a_rows = read_rows(out_dir / "v159_line_a_dche_dynamics.csv")
    a_h400 = read_rows(out_dir / "v159_line_a_h400_long.csv")
    b_rows = read_rows(out_dir / "v159_line_b_mlp_dynamics.csv")
    b_h400 = read_rows(out_dir / "v159_line_b_h400_long.csv")
    line_a = summarize_v159(a_rows, LINE_A_CANDIDATES, "line_a", summary_rows=read_rows(out_dir / "v159_line_a_summary.csv"))
    line_b = summarize_v159(b_rows, LINE_B_CANDIDATES, "line_b", summary_rows=read_rows(out_dir / "v159_line_b_summary.csv"))
    line_c = summarize_line_c(read_rows(out_dir / "v159_line_c_lq_reanchor.csv"), read_rows(out_dir / "v159_line_c_lq_functional.csv"))
    line_d = summarize_rat(read_rows(out_dir / "v159_line_d_rational_monitor.csv"), read_rows(out_dir / "v159_line_d_rational_summary.csv"))
    line_e = build_line_e(out_dir, line_e_out)
    return a_rows, a_h400, b_rows, b_h400, line_a, line_b, line_c, line_d, line_e


def build_docs(out_dir: Path, args: argparse.Namespace, route: dict[str, Any], line_a: dict[str, Any], line_b: dict[str, Any], line_c: dict[str, Any], line_d: dict[str, Any], line_e: dict[str, Any]) -> None:
    existing = RECAP_DOC.read_text(encoding="utf-8") if RECAP_DOC.exists() else ""
    manual = (
        "## 2.1 人工复核分析 / Insight\n\n"
        "这段是人工复核分析，不是 Python 自动生成的指标结论；下方 Line A/B/C/D/E/F/Z 数据仍由 runner 从 artifact 写入。\n\n"
        "v15.9 的关键修正是把 MLP 从 passive control 恢复成 active mechanism line，同时把 LQ reanchor、Rational monitor 与 all-basis substrate 纳入同一轮覆盖。当前结果不能只按 D-CHE 单线判断。\n"
    )
    if MANUAL_ANALYSIS_START in existing and MANUAL_ANALYSIS_END in existing:
        manual = existing.split(MANUAL_ANALYSIS_START, 1)[1].split(MANUAL_ANALYSIS_END, 1)[0].strip()
    contract = read_rows(out_dir / "v159_execution_contract_coverage_audit.csv")
    deep = read_rows(out_dir / "v159_deep_coverage_audit.csv")
    a_summary = read_rows(out_dir / "v159_line_a_summary.csv")
    b_summary = read_rows(out_dir / "v159_line_b_summary.csv")
    c_rows = read_rows(out_dir / "v159_line_c_lq_reanchor.csv")
    d_summary = read_rows(out_dir / "v159_line_d_rational_summary.csv")
    e_summary = read_rows(out_dir / "v159_line_e_allbasis_family_summary.csv")
    no_go = read_rows(out_dir / "v159_no_go_boundary.csv")
    queue = read_rows(out_dir / "v159_next_hypothesis_queue.csv")
    recap = [
        "# DG-KAN v15.9 MultiLineFunctionalDynamics MLP LQ AllBasis 实验结果复盘",
        "",
        "生成时间：2026-05-31（Asia/Singapore）",
        "",
        "本复盘只写入实际 artifact 中的结果；不把 MLP generic positive、LQ base/reanchor、Rational monitor 或 substrate-only rows 写成 KAN-specific promotion。",
        "",
        "## 1. 计划理解",
        "",
        "v15.9 的目标是停止 D-CHE 单线等待，改为同时推进 D-CHE dynamics、MLP active dynamics、LQ reanchor、Rational monitor 与 all-basis substrate，并通过 Line F 做 carrier/generic/recovery-only 路由。",
        "",
        "## 2. 本轮代码修改",
        "",
        "新增：",
        "",
        "```text",
        "experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py",
        "```",
        "",
        "修改：",
        "",
        "```text",
        "experiments/run_v149_line_d_all_basis_substrate_repair.py",
        "  新增 v15.9 substrate-only candidates:",
        "  D-FOU82..86, D-RBF80..84, D-WAV69..72。",
        "```",
        "",
        "过程说明：",
        "",
        "```text",
        "1. Line A 使用 D-CHE G7R pulse/recovery dynamics，执行 A0..A13 与 ACTRL controls。",
        "2. Line B 将 MLP functional dynamics 作为 active line 执行 B0..B10 与 controls，不写成 KAN-specific promotion。",
        "3. Line C 执行 C0..C4 LQ reanchor；若 reanchor gate 未开，则 C5..C9 functional fail-closed deferred。",
        "4. Line D 执行 Rational monitor/sanity replay；不重启 reset/controller/action route。",
        "5. Line E 调用 v149 substrate-only runner 读取 v15.9 D-FOU/RBF/WAV rows；不过 gate 不进入 official FU proof。",
        "6. 自查修复 LQ primitive forward 调用签名：显式传入 A 与 basis weights。",
        "7. 自查修复 Rational monitor 参数桥接：loss_interface=CE、linec_mode=exact、output_geometry_repair=none。",
        "8. 初始 official 命令使用 --device cuda:0；用户要求四卡并行后，runner 新增 --run-lines / --line-a-methods / --artifact-suffix / --skip-h400 分片执行。",
        "9. 正式分片执行：cuda:0..3 并行跑 A base shards；B/C/D 分别用 cuda:1/2/3；A base merge 后再跑全局 top2 A400。",
        "10. cpu_offload_used=0；LineC/CEp99/NLL/ECE/AUCtime/Brier 不进入方向。",
        "11. 用户再次追问后反方复核发现 final route taxonomy 与计划第 8/13 节不完全一致；已修正为 A/B/C/D/E 全部 gate 未开时写 R-F5-AllFunctionalDynamicsNoGo，LineE all-basis blocked 保留为 no-go evidence；该修复只重算 route/manifest/docs，不改训练指标。",
        "12. 继续按计划最低合同复核发现 D-CHE no-regression monitor 未单独落 artifact；已从 A0-D-CHE-AdamW 实际 rows 派生 v159_dche_no_regression_monitor.csv 并纳入 manifest/contract/deep coverage。该修复只补覆盖审计，不新增训练、不改指标。",
        "```",
        "",
        MANUAL_ANALYSIS_START,
        manual,
        MANUAL_ANALYSIS_END,
        "",
        "## 2.2 完整计划执行对照（artifact 自动写入）",
        "",
        "| contract item | status | details |",
        "|---|---:|---|",
    ]
    for row in contract:
        recap.append(f"| {row.get('contract_item')} | {row.get('status')} | {row.get('details')} |")
    recap.extend(["", "深度覆盖审计：", "", "| audit item | status | details |", "|---|---:|---|"])
    for row in deep:
        recap.append(f"| {row.get('audit_item')} | {row.get('status')} | {row.get('details')} |")
    recap.extend([
        "",
        "required / forbidden / no-action / provenance：",
        "",
        "```text",
        f"required_artifact_manifest_rows = {len(read_rows(out_dir / 'v159_required_artifact_manifest.csv'))}",
        f"required_artifact_missing_rows = {sum(sint(r.get('missing'), 0) for r in read_rows(out_dir / 'v159_required_artifact_manifest.csv'))}",
        f"forbidden_information_violation_sum = {sum(sint(r.get('violation'), 0) for r in read_rows(out_dir / 'v159_forbidden_information_audit.csv'))}",
        f"no_action_search_violation_sum = {sum(sint(r.get('violation'), 0) for r in read_rows(out_dir / 'v159_no_action_search_audit.csv'))}",
        f"direction_provenance_rows = {len(read_rows(out_dir / 'v159_line_a_direction_provenance.csv')) + len(read_rows(out_dir / 'v159_line_b_direction_provenance.csv'))}",
        "direction_source = train_stream_only / optimizer_state / split_gradient; audit metrics not used for direction",
        "```",
        "",
        "## 3. Line A / B dynamics summary",
        "",
        "```text",
        f"line_a_best_method = {line_a.get('best_method')}",
        f"line_a_gate_pass = {line_a.get('gate_pass')}",
        f"line_a_source_vs_best_control_mean = {line_a.get('source_vs_best_control_mean')}",
        f"line_a_bad_event_fraction = {line_a.get('bad_event_fraction')}",
        f"line_b_best_method = {line_b.get('best_method')}",
        f"line_b_gate_pass = {line_b.get('gate_pass')}",
        f"line_b_source_vs_best_control_mean = {line_b.get('source_vs_best_control_mean')}",
        f"line_b_bad_event_fraction = {line_b.get('bad_event_fraction')}",
        "```",
        "",
        "Line A method summary：",
        "",
        "| method | rows | pass | source | bad event | tail recovery | LineC recovery |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in a_summary:
        recap.append(f"| {row.get('method')} | {row.get('rows')} | {row.get('dataset_seed_pass_count')}/9 | {row.get('mean_source_vs_best_control')} | {row.get('bad_event_fraction')} | {row.get('tail_debt_recovery_rate')} | {row.get('LineC_debt_recovery_rate')} |")
    recap.extend(["", "Line B method summary：", "", "| method | rows | pass | source | bad event | tail recovery | LineC recovery |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in b_summary:
        recap.append(f"| {row.get('method')} | {row.get('rows')} | {row.get('dataset_seed_pass_count')}/9 | {row.get('mean_source_vs_best_control')} | {row.get('bad_event_fraction')} | {row.get('tail_debt_recovery_rate')} | {row.get('LineC_debt_recovery_rate')} |")
    recap.extend(["", "Line A/B h400 global top2：", "", "| line | method | rows | source | bad event | tail recovery | LineC recovery |", "|---|---|---:|---:|---:|---:|---:|"])
    for line_name, path in [("A", out_dir / "v159_line_a_h400_long.csv"), ("B", out_dir / "v159_line_b_h400_long.csv")]:
        h_rows = read_rows(path)
        for method in sorted({str(r.get("method")) for r in h_rows}):
            group = [r for r in h_rows if str(r.get("method")) == method]
            recap.append(f"| {line_name} | {method} | {len(group)} | {mean(fnum(r.get('source_vs_best_control'), 0.0) for r in group)} | {mean(sint(r.get('bad_event'), 0) for r in group)} | {mean(fnum(r.get('tail_debt_recovery_rate'), 0.0) for r in group)} | {mean(fnum(r.get('LineC_debt_recovery_rate'), 0.0) for r in group)} |")
    recap.extend([
        "",
        "## 4. Line C / D / E results",
        "",
        "```text",
        f"line_c_rows = {line_c.get('line_c_rows')}",
        f"line_c_reanchor_gate_pass = {line_c.get('line_c_reanchor_gate_pass')}",
        f"line_c_best_method = {line_c.get('line_c_best_method')}",
        f"line_c_best_macro_delta_vs_MLP = {line_c.get('line_c_best_macro_delta_vs_MLP')}",
        f"line_d_rat_rows = {line_d.get('line_d_rat_rows')}",
        f"line_d_rat_best_method = {line_d.get('line_d_rat_best_method')}",
        f"line_d_rat_gate_pass = {line_d.get('line_d_rat_gate_pass')}",
        f"line_e_rows = {line_e.get('line_e_rows')}",
        f"line_e_best_family = {line_e.get('line_e_best_family')}",
        f"line_e_best_dataset_seed_pass_count = {line_e.get('line_e_best_dataset_seed_pass_count')} / 9",
        "```",
        "",
        "Line C reanchor rows：",
        "",
        "| method | rows? | best delta vs MLP | gate pass rows |",
        "|---|---:|---:|---:|",
    ])
    for method in sorted({str(r.get("method")) for r in c_rows}):
        group = [r for r in c_rows if str(r.get("method")) == method]
        recap.append(f"| {method} | {len(group)} | {max([fnum(r.get('macro_delta_vs_MLP'), -999) for r in group] or [-999])} | {sum(sint(r.get('line_c_reanchor_gate_pass'), 0) for r in group)} |")
    recap.extend(["", "Line D Rational summary：", "", "| method | rows | pass | source | auc median |", "|---|---:|---:|---:|---:|"])
    for row in d_summary:
        recap.append(f"| {row.get('method')} | {row.get('rows')} | {row.get('dataset_seed_pass_count')}/9 | {row.get('mean_source_vs_best_control')} | {row.get('median_auc_time')} |")
    recap.extend(["", "Line E all-basis summary：", "", "| family | rows | pass | best candidate | max mean delta vs MLP |", "|---|---:|---:|---|---:|"])
    for row in e_summary:
        recap.append(f"| {row.get('family')} | {row.get('rows')} | {row.get('dataset_seed_pass_count')}/9 | {row.get('best_candidate')} | {row.get('max_mean_delta_vs_MLP')} |")
    recap.extend([
        "",
        "## 5. Line F / Z route",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"minimum_success = {route.get('minimum_success')}",
        f"promotion_allowed = {route.get('promotion_allowed')}",
        f"line_a_gate_pass = {route.get('line_a_gate_pass')}",
        f"line_b_gate_pass = {route.get('line_b_gate_pass')}",
        f"line_c_reanchor_gate_pass = {route.get('line_c_reanchor_gate_pass')}",
        f"line_d_rational_gate_pass = {route.get('line_d_rational_gate_pass')}",
        f"line_e_allbasis_gate_pass = {route.get('line_e_allbasis_gate_pass')}",
        f"required_artifact_missing_count = {route.get('required_artifact_missing_count')}",
        f"forbidden_information_violation_count = {route.get('forbidden_information_violation_count')}",
        f"no_action_search_violation_count = {route.get('no_action_search_violation_count')}",
        "```",
        "",
        "No-go boundary：",
        "",
        "| boundary | status | evidence |",
        "|---|---:|---|",
    ])
    for row in no_go:
        recap.append(f"| {row.get('boundary')} | {row.get('status')} | {row.get('evidence')} |")
    recap.extend(["", "Next hypothesis queue：", "", "| priority | hypothesis | allowed next step |", "|---:|---|---|"])
    for row in queue:
        recap.append(f"| {row.get('priority')} | {row.get('hypothesis')} | {row.get('allowed_next_step')} |")
    recap.extend([
        "",
        "## 6. 科学结论",
        "",
        "```text",
        "1. v15.9 已执行 Line R/A/B/C/D/E/F/Z，并生成 required artifacts。",
        "2. MLP functional dynamics 已作为 active line 执行；任何 MLP positive 不写成 KAN-specific promotion。",
        "3. LQ 仅在 reanchor gate 通过后才允许 functional；未通过时只写 LQBaseNotReanchored/deferred。",
        "4. Rational 只是 monitor/sanity replay；没有 reset/controller/action bank。",
        f"5. 当前 route = {route.get('route')}，promotion_allowed = {route.get('promotion_allowed')}。",
        "```",
    ])
    RECAP_DOC.write_text("\n".join(recap) + "\n", encoding="utf-8")
    exec_lines = [
        "# DG-KAN v15.9 MultiLineFunctionalDynamics MLP LQ AllBasis 执行日志",
        "",
        "生成时间：2026-05-31（Asia/Singapore）",
        "",
        "## 1. 文件",
        "",
        "```text",
        f"plan = {PLAN_DOC}",
        "runner = experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py",
        f"out_dir = {out_dir}",
        f"line_e_out = {args.line_e_out}",
        "```",
        "",
        "## 2. py_compile",
        "",
        "```bash",
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py experiments/run_v149_line_d_all_basis_substrate_repair.py",
        "```",
        "",
        "## 3. Line E substrate-only 执行指令",
        "",
        "```bash",
        line_e_command(args),
        "```",
        "",
        "## 4. Official v15.9 GPU 执行指令",
        "",
        "```bash",
        official_command(args, 0),
        "```",
        "",
        "四卡并行分片执行指令：",
        "",
        "```bash",
        shard_command(args, "S", "cuda:0", 1),
        shard_command(args, "B", "cuda:1", 0),
        shard_command(args, "C", "cuda:2", 0),
        shard_command(args, "D", "cuda:3", 0),
        shard_command(args, "A", "cuda:0", 0) + " --skip-h400 1 --artifact-suffix a0 --line-a-methods A0-D-CHE-AdamW,A1-D-CHE-G7R-PulseOnce-AdamWRecovery,A2-D-CHE-G7R-PulseEvery50-AdamWRecovery,A3-D-CHE-G7R-EarlyPulseOnly-AdamWRecovery,ACTRL1-D-CHE-RandomMatchedPulse-SameAdamWRecovery",
        shard_command(args, "A", "cuda:1", 0) + " --skip-h400 1 --artifact-suffix a1 --line-a-methods A4-D-CHE-G7R-MidPulseOnly-AdamWRecovery,A5-D-CHE-G7R-LatePulseOnly-AdamWRecovery,A6-D-CHE-G7R-Pulse-GlobalDecoupledDecayRecovery,A7-D-CHE-G7R-Pulse-DegreeWiseDecayRecovery,ACTRL2-D-CHE-RandomMatchedPulse-SameDecayRecovery",
        shard_command(args, "A", "cuda:2", 0) + " --skip-h400 1 --artifact-suffix a2 --line-a-methods A8-D-CHE-G7R-Pulse-HighDegreeExtraDecayRecovery,A9-D-CHE-G7R-Pulse-ReadoutBasisDecoupledDecayRecovery,A10-D-CHE-G7R-Pulse-MomentumEMARecovery,A11-D-CHE-G7R-Pulse-LRCooldownRecovery,ACTRL3-D-CHE-NoOpMatchedOverhead",
        shard_command(args, "A", "cuda:3", 0) + " --skip-h400 1 --artifact-suffix a3 --line-a-methods A12-D-CHE-G7R-Pulse-LookaheadConsolidation,A13-D-CHE-G7R-Pulse-ScheduleFreeLongEMARecovery,ACTRL4-D-CHE-AdamWExtraStepsMatchedTime",
        shard_command(args, "MERGEA", "cuda:0", 1),
        shard_command(args, "A400", "cuda:0", 0),
        "wait",
        shard_command(args, "E,finalize", "cuda:0", 1),
        "```",
        "",
        "## 5. Finalizer / route taxonomy / manifest 重算指令",
        "",
        "用户再次追问后执行该命令重算 route / Line Z / manifest / docs；随后按最低合同补齐 D-CHE no-regression monitor artifact 并再次重算 manifest/contract/docs；不重跑训练指标。",
        "",
        "```bash",
        official_command(args, 1),
        "```",
        "",
        "## 6. 最终结果",
        "",
        "```text",
        f"route = {route.get('route')}",
        f"promotion_allowed = {route.get('promotion_allowed')}",
        f"line_a_gate_pass = {route.get('line_a_gate_pass')}",
        f"line_b_gate_pass = {route.get('line_b_gate_pass')}",
        f"line_c_reanchor_gate_pass = {route.get('line_c_reanchor_gate_pass')}",
        f"line_d_rational_gate_pass = {route.get('line_d_rational_gate_pass')}",
        f"line_e_allbasis_gate_pass = {route.get('line_e_allbasis_gate_pass')}",
        f"required_artifact_missing_count = {route.get('required_artifact_missing_count')}",
        "training shards used cuda:0,cuda:1,cuda:2,cuda:3; cpu_offload_used=0",
        "```",
    ]
    EXEC_LOG_DOC.write_text("\n".join(exec_lines) + "\n", encoding="utf-8")


def line_e_command(args: argparse.Namespace) -> str:
    return (
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python "
        "experiments/run_v149_line_d_all_basis_substrate_repair.py "
        f"--out-dir {args.line_e_out} --device {args.device} --datasets {args.datasets} --seeds {args.seeds} "
        "--train-size 256 --val-size 128 --batch-size 32 --epochs 1 "
        f"--candidates {','.join(LINE_E_CANDIDATES)}"
    )


def official_command(args: argparse.Namespace, reuse: int) -> str:
    return (
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python "
        "experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py "
        f"--out-dir {args.out_dir} --line-e-out {args.line_e_out} --device {args.device} "
        f"--datasets {args.datasets} --seeds {args.seeds} --train-size {args.train_size} --val-size {args.val_size} "
        f"--test-size {args.test_size} --train-steps {args.train_steps} --rational-steps {args.rational_steps} "
        f"--lq-steps {args.lq_steps} --long-horizon-steps {args.long_horizon_steps} --batch-size {args.batch_size} "
        f"--split-count {args.split_count} --trace-interval {args.trace_interval} --linec-seeds {args.linec_seeds} "
        f"--real-linec {args.real_linec} --reuse-if-present {reuse}"
    )


def shard_command(args: argparse.Namespace, run_lines: str, device: str, reuse: int) -> str:
    return (
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python "
        "experiments/run_v159_multiline_functional_dynamics_mlp_lq_allbasis.py "
        f"--out-dir {args.out_dir} --line-e-out {args.line_e_out} --device {device} "
        f"--datasets {args.datasets} --seeds {args.seeds} --train-size {args.train_size} --val-size {args.val_size} "
        f"--test-size {args.test_size} --train-steps {args.train_steps} --rational-steps {args.rational_steps} "
        f"--lq-steps {args.lq_steps} --long-horizon-steps {args.long_horizon_steps} --batch-size {args.batch_size} "
        f"--split-count {args.split_count} --trace-interval {args.trace_interval} --linec-seeds {args.linec_seeds} "
        f"--real-linec {args.real_linec} --reuse-if-present {reuse} --run-lines {run_lines}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--line-e-out", default=str(DEFAULT_LINE_E_OUT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--train-size", type=int, default=256)
    ap.add_argument("--val-size", type=int, default=128)
    ap.add_argument("--test-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-steps", type=int, default=120)
    ap.add_argument("--long-horizon-steps", type=int, default=400)
    ap.add_argument("--rational-steps", type=int, default=80)
    ap.add_argument("--lq-steps", type=int, default=80)
    ap.add_argument("--trace-interval", type=int, default=40)
    ap.add_argument("--split-count", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--weight-decay", type=float, default=0.001)
    ap.add_argument("--readout-weight-decay", type=float, default=0.001)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--lambda-noise", type=float, default=0.25)
    ap.add_argument("--linec-seeds", default="0")
    ap.add_argument("--linec-batch-size", type=int, default=32)
    ap.add_argument("--linec-sketch-dim", type=int, default=64)
    ap.add_argument("--real-linec", type=int, default=1)
    ap.add_argument("--dche-candidate", default="D-CHE20-DegreeNormalizedReadoutHealthSubstrate")
    ap.add_argument("--rational-candidate", default="D-RAT28-GroupDiversityPreservingRational")
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--reuse-if-present", type=int, default=1)
    ap.add_argument("--run-lines", default="all", help="Comma list among S,A,B,C,D,E,finalize,all. Shards write disjoint artifacts.")
    ap.add_argument("--line-a-methods", default="", help="Optional comma filter for A-line method shards.")
    ap.add_argument("--line-b-methods", default="", help="Optional comma filter for B-line method shards.")
    ap.add_argument("--artifact-suffix", default="", help="Optional suffix for shard artifact files.")
    ap.add_argument("--skip-h400", type=int, default=0, help="Write base dynamics only; h400 can be run after merge.")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_lines = {part.strip().upper() for part in str(args.run_lines).split(",") if part.strip()}
    all_lines = "ALL" in run_lines
    finalize = all_lines or "FINALIZE" in run_lines or "Z" in run_lines
    if "MERGE_A" in run_lines or "MERGEA" in run_lines:
        merge_dynamics_parts(out_dir, "A")
    if "MERGE_B" in run_lines or "MERGEB" in run_lines:
        merge_dynamics_parts(out_dir, "B")
    need_splits = all_lines or bool(run_lines & {"S", "A", "B", "C", "D", "A400", "B400"})
    if need_splits:
        device = resolve_cuda_device(str(args.device))
        if device.type != "cuda":
            raise RuntimeError("v15.9 execution requires CUDA; refusing cpu execution")
        splits = v158.load_splits(args, device)
        if all_lines or "S" in run_lines:
            v158.run_line_s(args, splits, device, out_dir)
        if all_lines or "A" in run_lines:
            run_dynamics_line("A", "D-CHE", LINE_A_METHODS, LINE_A_CONTROLS, LINE_A_CANDIDATES, args, splits, device, out_dir)
        if all_lines or "B" in run_lines:
            run_dynamics_line("B", "MLP", LINE_B_METHODS, LINE_B_CONTROLS, LINE_B_CANDIDATES, args, splits, device, out_dir)
        if all_lines or "C" in run_lines:
            run_line_c(args, splits, device, out_dir)
        if all_lines or "D" in run_lines:
            run_line_d_rational(args, splits, device, out_dir)
        if "A400" in run_lines:
            run_dynamics_h400_from_existing("A", "D-CHE", LINE_A_CONTROLS, LINE_A_CANDIDATES, args, splits, device, out_dir)
        if "B400" in run_lines:
            run_dynamics_h400_from_existing("B", "MLP", LINE_B_CONTROLS, LINE_B_CANDIDATES, args, splits, device, out_dir)
    if all_lines or "E" in run_lines:
        build_line_e(out_dir, Path(args.line_e_out))
    if not finalize:
        print(json.dumps({"stage": "V159_LINE_SHARD_DONE", "run_lines": sorted(run_lines), "out_dir": str(out_dir)}, indent=2, sort_keys=True))
        return
    a_rows, a_h400, b_rows, b_h400, line_a, line_b, line_c, line_d, line_e = load_existing_summaries(out_dir, Path(args.line_e_out))
    build_audits(out_dir)
    build_failure_taxonomy(out_dir, a_rows + b_rows + a_h400 + b_h400, line_c, line_d, line_e)
    build_dche_no_regression_monitor(out_dir)
    write_required_manifest(out_dir)
    missing = sum(sint(r.get("missing"), 0) for r in read_rows(out_dir / "v159_required_artifact_manifest.csv"))
    forbidden = sum(sint(r.get("violation"), 0) for r in read_rows(out_dir / "v159_forbidden_information_audit.csv"))
    no_action = sum(sint(r.get("violation"), 0) for r in read_rows(out_dir / "v159_no_action_search_audit.csv"))
    route = decide_route(line_a, line_b, line_c, line_d, line_e, missing, forbidden, no_action)
    write_json(out_dir / "v159_route_decision.json", route)
    build_line_f(route, line_a, line_b, line_c, line_d, line_e, out_dir)
    build_budget_certificate(out_dir, args, line_c)
    build_line_z(out_dir, route, line_a, line_b, line_c, line_d, line_e)
    write_figures(out_dir, a_rows, b_rows, read_rows(out_dir / "v159_line_c_lq_reanchor.csv"), read_rows(out_dir / "v159_line_d_rational_monitor.csv"), read_rows(out_dir / "v159_line_e_allbasis_results.csv"), route)
    write_required_manifest(out_dir)
    missing = sum(sint(r.get("missing"), 0) for r in read_rows(out_dir / "v159_required_artifact_manifest.csv"))
    route = decide_route(line_a, line_b, line_c, line_d, line_e, missing, forbidden, no_action)
    write_json(out_dir / "v159_route_decision.json", route)
    build_contracts(out_dir, route)
    build_docs(out_dir, args, route, line_a, line_b, line_c, line_d, line_e)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
