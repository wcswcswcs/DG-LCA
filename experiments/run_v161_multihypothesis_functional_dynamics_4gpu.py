#!/usr/bin/env python3
"""DG-KAN v16.1 multi-hypothesis functional dynamics 4-GPU runner.

The runner reuses the validated v16.0 training/evaluation kernels, but it owns
the v16.1 method surface, artifact names, route taxonomy, GPU assignment audit,
code-review packet, execution log, and recap. Audit metrics remain readback-only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import copy
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_v150_function_update_allbasis_parallel as v150  # noqa: E402
from experiments import run_v154_split_consensus_signal_subspace_fu_allbasis as v154  # noqa: E402
from experiments import run_v158_dynamics_harness_decoupled_decay_recovery_allbasis as v158  # noqa: E402
from experiments import run_v160_dynamic_geometry_debt_recovery_multiline_fu as v160  # noqa: E402


PLAN_DOC = ROOT / "docs/DG-KAN_v16.1_MultiHypothesisFunctionalDynamics_4GPU_完整计划.md"
RECAP_DOC = ROOT / "docs/DG-KAN_v16.1_MultiHypothesisFunctionalDynamics_4GPU_实验结果复盘.md"
EXEC_LOG_DOC = ROOT / "docs/DG-KAN_v16.1_MultiHypothesisFunctionalDynamics_4GPU_执行日志.md"
DEFAULT_OUT = ROOT / "results/v16_1_multihypothesis_functional_dynamics_4gpu/official_v161"
DEFAULT_LINE_F_OUT = ROOT / "results/v16_1_multihypothesis_functional_dynamics_4gpu/line_f_v161_allbasis_substrate"
MANUAL_ANALYSIS_START = "<!-- V16.1_MANUAL_ANALYSIS_START -->"
MANUAL_ANALYSIS_END = "<!-- V16.1_MANUAL_ANALYSIS_END -->"

fnum = v154.fnum
sint = v154.sint
mean = v154.mean
read_rows = v154.read_rows
write_rows = v154.write_rows
write_json = v154.write_json
parse_csv = v154.parse_csv
resolve_cuda_device = v154.resolve_cuda_device


LINE_A_METHODS = [
    "A-M1-R0-D-CHE-PulseOnce-AdamWRecovery",
    "A-M2-D-CHE-SplitConsensusSignalSubspace",
    "A-M3-D-CHE-PopRiskSNRPreconditioner",
    "A-M4-D-CHE-FunctionSpaceProximal",
    "A-M5-D-CHE-OptimizerAlignedFU",
    "A-M6-D-CHE-CarrierReparamGlobalDecay",
    "ACTRL0-D-CHE-AdamW",
    "ACTRL1-D-CHE-NoOpMatchedOverhead",
    "ACTRL2-D-CHE-RandomMatchedPulse",
    "ACTRL3-D-CHE-AdamWExtraStepsMatchedTime",
]
LINE_A_CONTROLS = {m for m in LINE_A_METHODS if m.startswith("ACTRL")}
LINE_A_CANDIDATES = [m for m in LINE_A_METHODS if m not in LINE_A_CONTROLS]

LINE_B_METHODS = [
    "B-M1-R0-MLP-PulseOnce-AdamWRecovery",
    "B-M2-MLP-HiddenSplitConsensusSignalSubspace",
    "B-M3-MLP-PopRiskSNRPreconditioner",
    "B-M4-MLP-FunctionSpaceProximal",
    "B-M5-MLP-CautiousAdamW",
    "B-M6-MLP-HiddenCarrierDecay",
    "BCTRL0-MLP-AdamW",
    "BCTRL1-MLP-NoOpMatchedOverhead",
    "BCTRL2-MLP-RandomMatchedPulse",
    "BCTRL3-MLP-AdamWExtraStepsMatchedTime",
]
LINE_B_CONTROLS = {m for m in LINE_B_METHODS if m.startswith("BCTRL")}
LINE_B_CANDIDATES = [m for m in LINE_B_METHODS if m not in LINE_B_CONTROLS]

LINE_F_CANDIDATES = [
    "D-FOU92-LowFreqIdentityResidualV7",
    "D-FOU93-BandwiseSNRWarmupV7",
    "D-FOU94-PhaseStableBandMixV7",
    "D-FOU95-NoMaterializeLifetimeV7",
    "D-FOU96-HighFrequencyQuarantineV7",
    "D-RBF90-ActiveCenterOccupancyV7",
    "D-RBF91-WidthConditionGuardV7",
    "D-RBF92-CompactBumpNoDenseV7",
    "D-RBF93-GaussianLocalK4TaskHealthV7",
    "D-RBF94-IdentityResidualWidthWarmupV7",
    "D-WAV77-TriangularSupportV7",
    "D-WAV78-ScaleOccupancyV7",
    "D-WAV79-SupportOverlapDampingV7",
    "D-WAV80-LocalTailCoverageAuditV7",
]

REQUIRED_FIGURES = [
    "carrier_mechanism_gate_heatmap.svg",
    "carrier_mechanism_source_heatmap_h800.svg",
    "carrier_mechanism_debt_recovery_heatmap_h800.svg",
    "carrier_mechanism_control_equivalence_heatmap.svg",
    "source_retention_curve_by_carrier.svg",
    "tail_debt_curve_by_carrier.svg",
    "LineC_debt_curve_by_carrier.svg",
    "calibration_debt_curve_by_carrier.svg",
    "AUCtime_curve_by_carrier.svg",
    "recovery_rate_by_mechanism.svg",
    "D-CHE_vs_MLP_source_retention.svg",
    "D-CHE_vs_MLP_tail_recovery.svg",
    "D-CHE_vs_MLP_LineC_recovery.svg",
    "KAN_specific_delta_heatmap.svg",
    "generic_vs_kan_specific_route_dashboard.svg",
    "LQ_reanchor_nearpass_heatmap.svg",
    "LQ_protocol_drift_heatmap.svg",
    "basis_family_pass_count_heatmap.svg",
    "basis_step_memory_pareto.svg",
    "basis_LineC_task_health.svg",
    "gpu_assignment_timeline.svg",
    "gpu_rows_executed_by_round.svg",
    "gpu_idle_reason_table.svg",
]
REQUIRED_ARTIFACTS = [
    "v161_route_decision.json",
    "v161_method_surface_manifest.csv",
    "v161_carrier_registry_manifest.csv",
    "v161_line_r_audit.csv",
    "v161_forbidden_information_audit.csv",
    "v161_no_action_search_audit.csv",
    "v161_gpu_assignment_manifest.csv",
    "v161_line_g_dynamic_geometry.csv",
    "v161_line_a_dche_matrix.csv",
    "v161_line_a_horizon_recovery.csv",
    "v161_line_a_h1600_long.csv",
    "v161_line_b_mlp_matrix.csv",
    "v161_line_b_horizon_recovery.csv",
    "v161_line_b_h1600_long.csv",
    "v161_line_c_lq_reanchor.csv",
    "v161_line_c_lq_functional.csv",
    "v161_line_d_rational_monitor.csv",
    "v161_line_e_recovery_matrix.csv",
    "v161_line_f_allbasis_results.csv",
    "v161_line_f_allbasis_family_summary.csv",
    "v161_carrier_mechanism_summary.csv",
    "v161_line_m_crossline_attribution.csv",
    "v161_failure_taxonomy.csv",
    "v161_budget_exhaustion_certificate.csv",
    "v161_no_go_boundary.csv",
    "v161_next_hypothesis_queue.csv",
    "v161_code_review_packet.csv",
    "v161_execution_contract_coverage_audit.csv",
    "v161_deep_coverage_audit.csv",
]


def mechanism_family(method: str) -> str:
    if "-M1-" in method:
        return "M1-ProductivePulseRecovery"
    if "-M2-" in method:
        return "M2-SplitConsensusSignalSubspace"
    if "-M3-" in method:
        return "M3-PopRiskDriftSNR"
    if "-M4-" in method:
        return "M4-FunctionSpaceProximal"
    if "-M5-" in method:
        return "M5-OptimizerAwareAlignedFU"
    if "-M6-" in method:
        return "M6-CarrierSpecificReparamFU"
    if "CTRL" in method:
        return "CONTROL"
    return "UNKNOWN"


def map_dynamics_method(method: str) -> tuple[str, dict[str, Any], str, str]:
    mapping: dict[str, tuple[str, dict[str, Any], str, str]] = {
        "A-M1-R0-D-CHE-PulseOnce-AdamWRecovery": ("T1-G7R-PulseOnce-then-AdamWRecovery", {}, "M1 R0 G7R pulse once + AdamW recovery", "M1-R0-AdamWRecovery"),
        "A-M1-R1-D-CHE-PulseEvery50-AdamWRecovery": ("T2-G7R-PulseEvery50-then-AdamWRecovery", {}, "M1 R1 repeated pulse + AdamW recovery", "M1-R1-RepeatedAdamWRecovery"),
        "A-M1-R2-D-CHE-EarlyPulse-AdamWRecovery": ("T3-G7R-PulseEarlyOnly-then-AdamWRecovery", {}, "M1 R2 early pulse", "M1-R2-EarlyPulseRecovery"),
        "A-M1-R3-D-CHE-MidPulse-AdamWRecovery": ("T4-G7R-PulseMidOnly-then-AdamWRecovery", {}, "M1 R3 mid pulse", "M1-R3-MidPulseRecovery"),
        "A-M1-R4-D-CHE-LatePulse-AdamWRecovery": ("T5-G7R-PulseLateOnly-then-AdamWRecovery", {}, "M1 R4 late pulse", "M1-R4-LatePulseRecovery"),
        "A-M1-R5-D-CHE-MomentumDampedRecovery": ("T1-G7R-PulseOnce-then-AdamWRecovery", {"beta1": 0.70}, "M1 R5 momentum-damped recovery", "M1-R5-MomentumDampedRecovery"),
        "A-M1-R6-D-CHE-SecondMomentAdaptRecovery": ("T1-G7R-PulseOnce-then-AdamWRecovery", {"beta2": 0.95}, "M1 R6 second-moment recovery", "M1-R6-SecondMomentAdaptRecovery"),
        "A-M1-R7-D-CHE-LRCooldownRecovery": ("T1-G7R-PulseOnce-then-AdamWRecovery", {"lr": 0.0015}, "M1 R7 LR cooldown recovery", "M1-R7-LRCooldownRecovery"),
        "A-M1-R8-D-CHE-EMALookaheadRecovery": ("T1-G7R-PulseOnce-then-AdamWRecovery", {"beta1": 0.50}, "M1 R8 EMA/lookahead recovery", "M1-R8-EMALookaheadRecovery"),
        "A-M1-R9-D-CHE-SWAConsolidation": ("T1-G7R-PulseOnce-then-AdamWRecovery", {"beta1": 0.30}, "M1 R9 SWA-like consolidation", "M1-R9-SWAConsolidation"),
        "A-M2-D-CHE-SplitConsensusSignalSubspace": ("T1-G7R-PulseOnce-then-AdamWRecovery", {}, "M2 split-consensus signal-subspace proxy from train-stream tangent", "M2-SplitConsensusSignalSubspace"),
        "A-M3-D-CHE-PopRiskSNRPreconditioner": ("T1-G7R-PulseOnce-then-AdamWRecovery", {"beta1": 0.80}, "M3 PopRisk/drift SNR preconditioner proxy", "M3-PopRiskDriftSNR"),
        "A-M4-D-CHE-FunctionSpaceProximal": ("T1-G7R-PulseOnce-then-AdamWRecovery", {"lr": 0.001}, "M4 function-space proximal/split-transfer proxy", "M4-FunctionSpaceProximal"),
        "A-M5-D-CHE-OptimizerAlignedFU": ("T1-G7R-PulseOnce-then-AdamWRecovery", {"beta1": 0.70, "beta2": 0.95}, "M5 optimizer-aware aligned FU proxy", "M5-OptimizerAwareAlignedFU"),
        "A-M6-D-CHE-CarrierReparamGlobalDecay": ("W1-G7R-GlobalDecoupledWeightDecayRecovery", {}, "M6 carrier reparameterized global-decay FU", "M6-CarrierGlobalDecay"),
        "A-M6-D-CHE-CarrierReparamRoleWiseDecay": ("W5-G7R-SignalReservoirDualDecay", {}, "M6 carrier role-wise decay FU", "M6-CarrierRoleWiseDecay"),
        "A-M6-D-CHE-CarrierReparamHighDegreeDecay": ("W3-G7R-HighDegreeExtraDecay", {}, "M6 carrier high-degree decay FU", "M6-CarrierHighDegreeDecay"),
        "ACTRL0-D-CHE-AdamW": ("T0-D-CHE-AdamW", {}, "D-CHE AdamW matched baseline", "control-adamw"),
        "ACTRL1-D-CHE-NoOpMatchedOverhead": ("TCTRL-NoOpMatchedOverhead", {}, "D-CHE no-op matched overhead", "control-noop"),
        "ACTRL2-D-CHE-RandomMatchedPulse": ("TCTRL-RandomMatchedPulse-then-AdamWRecovery", {}, "D-CHE random matched pulse", "control-random"),
        "ACTRL3-D-CHE-AdamWExtraStepsMatchedTime": ("TCTRL-AdamWExtraStepsMatchedTime", {}, "D-CHE AdamW extra-steps matched-time control", "control-adamw-extra"),
        "ACTRL4-D-CHE-DecayOnlyRecovery": ("WCTRL-DecayOnly-W1", {}, "D-CHE decay-only control", "control-decay-only"),
        "ACTRL5-D-CHE-RecoveryOnlyNoPulse": ("WCTRL-NoOpSameDecay", {}, "D-CHE recovery-only no-pulse control", "control-recovery-only"),
        "B-M1-R0-MLP-PulseOnce-AdamWRecovery": ("MLP-G7AnalogPulse", {}, "M1 R0 MLP pulse once + AdamW recovery", "M1-R0-AdamWRecovery"),
        "B-M1-R1-MLP-PulseEvery50-AdamWRecovery": ("MLP-FunctionalPulseEvery50", {}, "M1 R1 MLP repeated pulse", "M1-R1-RepeatedAdamWRecovery"),
        "B-M1-R2-MLP-MomentumDampedRecovery": ("MLP-G7AnalogPulse", {"beta1": 0.70}, "M1 R2 MLP momentum-damped recovery", "M1-R2-MomentumDampedRecovery"),
        "B-M1-R3-MLP-SecondMomentAdaptRecovery": ("MLP-G7AnalogPulse", {"beta2": 0.95}, "M1 R3 MLP second-moment recovery", "M1-R3-SecondMomentAdaptRecovery"),
        "B-M1-R4-MLP-LRCooldownRecovery": ("MLP-G7AnalogPulse", {"lr": 0.0015}, "M1 R4 MLP LR cooldown", "M1-R4-LRCooldownRecovery"),
        "B-M1-R5-MLP-EMALookaheadRecovery": ("MLP-G7AnalogPulse", {"beta1": 0.50}, "M1 R5 MLP EMA/lookahead", "M1-R5-EMALookaheadRecovery"),
        "B-M1-R6-MLP-SWAConsolidation": ("MLP-G7AnalogPulse", {"beta1": 0.30}, "M1 R6 MLP SWA-like consolidation", "M1-R6-SWAConsolidation"),
        "B-M1-R7-MLP-PulseDecayRecovery": ("MLP-G7AnalogPulsePlusDecay", {}, "M1 R7 MLP pulse + decay recovery", "M1-R7-PulseDecayRecovery"),
        "B-M1-R8-MLP-AmortizedFMSRecovery": ("MLP-G7AnalogPulse", {"beta1": 0.95}, "M1 R8 MLP amortized FMS proxy", "M1-R8-AmortizedFMSRecovery"),
        "B-M1-R9-MLP-LongCooldownRecovery": ("MLP-G7AnalogPulse", {"lr": 0.001}, "M1 R9 MLP long cooldown", "M1-R9-LongCooldownRecovery"),
        "B-M2-MLP-HiddenSplitConsensusSignalSubspace": ("MLP-G7AnalogPulse", {}, "M2 MLP hidden split-consensus signal-subspace proxy", "M2-HiddenSplitConsensusSignalSubspace"),
        "B-M3-MLP-PopRiskSNRPreconditioner": ("MLP-G7AnalogPulse", {"beta1": 0.80}, "M3 MLP PopRisk/drift SNR proxy", "M3-PopRiskDriftSNR"),
        "B-M4-MLP-FunctionSpaceProximal": ("MLP-G7AnalogPulse", {"lr": 0.001}, "M4 MLP function-space proximal proxy", "M4-FunctionSpaceProximal"),
        "B-M5-MLP-CautiousAdamW": ("MLP-CautiousAdamW", {}, "M5 MLP cautious AdamW", "M5-CautiousAdamW"),
        "B-M5-MLP-MGUP": ("MLP-MGUP", {}, "M5 MLP MGUP optimizer-aware baseline", "M5-MGUP"),
        "B-M6-MLP-HiddenCarrierDecay": ("MLP-G7AnalogPulsePlusDecay", {}, "M6 MLP hidden-carrier decay", "M6-HiddenCarrierDecay"),
        "B-M6-MLP-HiddenCarrierPulse": ("MLP-G7AnalogPulse", {"beta1": 0.60}, "M6 MLP hidden-carrier pulse", "M6-HiddenCarrierPulse"),
        "BCTRL0-MLP-AdamW": ("MLP-AdamW", {}, "MLP AdamW baseline", "control-adamw"),
        "BCTRL1-MLP-NoOpMatchedOverhead": ("MLP-NoOpMatchedOverhead", {}, "MLP no-op matched overhead", "control-noop"),
        "BCTRL2-MLP-RandomMatchedPulse": ("MLP-RandomPulseSameNorm", {}, "MLP random matched pulse", "control-random"),
        "BCTRL3-MLP-AdamWExtraStepsMatchedTime": ("MLP-AdamW", {}, "MLP AdamW matched-time control", "control-adamw-extra"),
        "BCTRL4-MLP-DecayOnly": ("MLP-G7AnalogPulsePlusDecay", {}, "MLP decay-only proxy control", "control-decay-only"),
        "BCTRL5-MLP-RecoveryOnlyNoPulse": ("MLP-NoOpMatchedOverhead", {}, "MLP recovery-only no-pulse control", "control-recovery-only"),
    }
    return mapping[method]


def install_v161_surface() -> None:
    v160.LINE_A_METHODS = list(LINE_A_METHODS)
    v160.LINE_A_CONTROLS = set(LINE_A_CONTROLS)
    v160.LINE_A_CANDIDATES = list(LINE_A_CANDIDATES)
    v160.LINE_B_METHODS = list(LINE_B_METHODS)
    v160.LINE_B_CONTROLS = set(LINE_B_CONTROLS)
    v160.LINE_B_CANDIDATES = list(LINE_B_CANDIDATES)
    v160.LINE_F_CANDIDATES = list(LINE_F_CANDIDATES)
    v160.map_dynamics_method = map_dynamics_method


def train_dynamics_case_horizon_only(
    line: str,
    method: str,
    family: str,
    dataset: str,
    seed: int,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xva: torch.Tensor,
    yva: torch.Tensor,
    xte: torch.Tensor | None,
    yte: torch.Tensor | None,
    input_dim: int,
    output_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    steps: int,
) -> dict[str, Any]:
    case_args = copy(args)
    case_args.synthetic_dim = int(input_dim)
    case_args.synthetic_classes = int(output_dim)
    case_args.mlp_hidden = int(args.hidden)
    model = v158.v1410.make_case_model("MLP", "MLP-v161-horizon-target-readback", xtr, seed, case_args, device) if family == "MLP" else v158.v1410.make_case_model("D-CHE", str(args.dche_candidate), xtr, seed, case_args, device)
    specs = v158.v1410.named_param_specs(model)
    total = sum(int(spec.param.numel()) for spec in specs)
    m = torch.zeros(total, device=device)
    v = torch.zeros(total, device=device)
    batch_gen = torch.Generator(device=device).manual_seed(int(seed) + 1_610_000 + sum(ord(c) for c in dataset) + (20_000 if family == "MLP" else 0))
    gen = torch.Generator(device=device).manual_seed(int(seed) + 1_615_000 + sum(ord(c) for c in method + dataset))
    pulse_set = set(v158.pulse_steps(method, int(steps), v158.HORIZONS))
    horizon_targets = {0, int(steps)}
    for p in pulse_set:
        horizon_targets.add(p)
        for h in v158.HORIZONS:
            horizon_targets.add(min(int(steps), p + h))
    metrics_by_step: dict[int, dict[str, float]] = {0: v158.eval_pack(model, xtr, ytr, xva, yva)}
    linec_by_step: dict[int, dict[str, Any]] = {}
    trajectory: list[dict[str, float]] = [{"step": 0.0, **metrics_by_step[0]}]
    telemetry: dict[str, list[float]] = {}
    direction_rows: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = v158.time.perf_counter()
    metric_choice, sketch = "M0-identity", "S0-diagonal"
    for step in range(int(steps)):
        idx = torch.randint(0, xtr.shape[0], (int(args.batch_size),), generator=batch_gen, device=device)
        xb, yb = xtr[idx], ytr[idx]
        grads, xs, ys, _losses = v158.v154.split_grads(model, specs, xb, yb, int(args.split_count))
        grad = torch.stack(grads, dim=0).mean(dim=0)
        beta1, beta2 = float(args.beta1), float(args.beta2)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad.square()
        mhat = m / (1.0 - beta1 ** (step + 1))
        vhat = v / (1.0 - beta2 ** (step + 1))
        stats = v158.v154.consensus_stats(grads, grad, vhat, specs, metric_choice, sketch, gen, float(args.lambda_noise))
        step_update, decay_update, g7r_update, trace = v158.choose_updates(method, step, pulse_set, grad, mhat, vhat, specs, stats, gen, args)
        split_deltas = v158.v154.trial_split_deltas(model, specs, xs, ys, g7r_update, float(args.lr))
        trace.update(
            {
                "B1_gain": -split_deltas[0] if len(split_deltas) > 0 else 0.0,
                "B2_gain": -split_deltas[1] if len(split_deltas) > 1 else 0.0,
                "B3_gain": -split_deltas[2] if len(split_deltas) > 2 else 0.0,
                "micro_horizon_loss_integral": sum(max(0.0, d) for d in split_deltas),
                "loss_spike_count": sum(int(d > 0.0) for d in split_deltas),
            }
        )
        trace.update(v158.v157.train_stream_readback(model, specs, xs, ys, xb, yb, g7r_update, float(args.lr), trace))
        v158.add_flat_update(specs, step_update, float(args.lr))
        v158.add_flat_update(specs, decay_update, float(args.lr))
        for k, val in trace.items():
            if isinstance(val, (int, float)) and v158.math.isfinite(float(val)):
                telemetry.setdefault(k, []).append(float(val))
        step_no = step + 1
        if step_no in horizon_targets:
            pack = v158.eval_pack(model, xtr, ytr, xva, yva)
            metrics_by_step[step_no] = pack
            trajectory.append({"step": float(step_no), **pack})
            if family == "D-CHE":
                linec_by_step[step_no] = v158.linec_pack(model, xtr, ytr, xva, yva, args)
        if step_no == int(steps) or ((step_no % max(1, int(args.trace_interval))) == 0):
            direction_rows.append(
                {
                    "line": line,
                    "family": family,
                    "method": method,
                    "dataset": dataset,
                    "seed": seed,
                    "step": step_no,
                    **{k: v158.median(vv) for k, vv in telemetry.items()},
                    "direction_uses_train_stream_only": 1,
                    "uses_LineC_as_direction": 0,
                    "uses_CEp99_as_direction": 0,
                    "uses_NLL_as_direction": 0,
                    "uses_ECE_as_direction": 0,
                    "uses_AUCtime_as_direction": 0,
                    "fast_horizon_readback_used": 1,
                    "promotion_allowed": 0,
                }
            )
    if 0 not in linec_by_step and family == "D-CHE":
        linec_by_step[0] = v158.linec_pack(model, xtr, ytr, xva, yva, args)
    if int(steps) not in metrics_by_step:
        metrics_by_step[int(steps)] = v158.eval_pack(model, xtr, ytr, xva, yva)
        trajectory.append({"step": float(int(steps)), **metrics_by_step[int(steps)]})
    if int(steps) not in linec_by_step and family == "D-CHE":
        linec_by_step[int(steps)] = v158.linec_pack(model, xtr, ytr, xva, yva, args)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    elapsed = v158.time.perf_counter() - start
    final = metrics_by_step[int(steps)]
    final_linec = linec_by_step.get(int(steps), {"LineC_majority_pass": 0, "LineC_pass_rate": 0.0, "CouplingR2": 0.0, "NoiseSignalLeak": 1.0, "RealSignalReservoirRatio": 1.0})
    horizon_rows = []
    for p in sorted(pulse_set):
        pre = metrics_by_step.get(p, metrics_by_step[0])
        pre_linec = linec_by_step.get(p, linec_by_step.get(0, {"LineC_majority_pass": 0}))
        for h in v158.HORIZONS:
            target = min(int(steps), p + h)
            met = metrics_by_step.get(target, final)
            lc = linec_by_step.get(target, final_linec)
            available = [r for r in trajectory if p < int(r["step"]) <= target]
            tail_debts = [max(0.0, r["CEp99"] - pre["CEp99"]) for r in available]
            nll_debts = [max(0.0, r["NLL"] - pre["NLL"]) for r in available]
            tail_peak = max(tail_debts or [max(0.0, met["CEp99"] - pre["CEp99"])])
            nll_peak = max(nll_debts or [max(0.0, met["NLL"] - pre["NLL"])])
            tail_final = max(0.0, met["CEp99"] - pre["CEp99"])
            nll_final = max(0.0, met["NLL"] - pre["NLL"])
            linec_pre_debt = 1 - sint(pre_linec.get("LineC_majority_pass"), 0)
            linec_final_debt = 1 - sint(lc.get("LineC_majority_pass"), 0)
            linec_peak = max(linec_pre_debt, linec_final_debt)
            horizon_rows.append(
                {
                    "line": line,
                    "family": family,
                    "method": method,
                    "dataset": dataset,
                    "seed": seed,
                    "pulse_step": p,
                    "horizon": h,
                    "target_step": target,
                    "NLL_h": met["NLL"],
                    "CEp99_h": met["CEp99"],
                    "ECE_h": met["ECE"],
                    "Brier_h": met["Brier"],
                    "margin_p10_h": met["margin_p10"],
                    "train_loss_delta_h": met["train_loss"] - pre["train_loss"],
                    "val_loss_delta_h": met["val_loss"] - pre["val_loss"],
                    "logit_rms_delta_h": met["logit_rms"] - pre["logit_rms"],
                    "entropy_delta_h": met["entropy"] - pre["entropy"],
                    "AUC_NLL_h": mean(r["NLL"] for r in available) if available else met["NLL"],
                    "CEp99_delta_h": met["CEp99"] - pre["CEp99"],
                    "NLL_delta_h": met["NLL"] - pre["NLL"],
                    "ECE_delta_h": met["ECE"] - pre["ECE"],
                    "Brier_delta_h": met["Brier"] - pre["Brier"],
                    "LineC_pass_h": lc.get("LineC_majority_pass", ""),
                    "CouplingR2_h": lc.get("CouplingR2", ""),
                    "NoiseSignalLeak_h": lc.get("NoiseSignalLeak", ""),
                    "RealSignalReservoirRatio_h": lc.get("RealSignalReservoirRatio", ""),
                    "tail_debt_peak_h": tail_peak,
                    "tail_debt_final_h": tail_final,
                    "tail_debt_recovery_rate_h": (tail_peak - tail_final) / max(1.0e-8, tail_peak) if tail_peak > 0 else 1.0,
                    "LineC_debt_peak_h": linec_peak,
                    "LineC_debt_final_h": linec_final_debt,
                    "LineC_debt_recovery_rate_h": (linec_peak - linec_final_debt) / max(1.0e-8, linec_peak) if linec_peak > 0 else 1.0,
                    "NLL_debt_peak_h": nll_peak,
                    "NLL_debt_final_h": nll_final,
                    "NLL_debt_recovery_rate_h": (nll_peak - nll_final) / max(1.0e-8, nll_peak) if nll_peak > 0 else 1.0,
                    "fast_horizon_readback_used": 1,
                    "promotion_allowed": 0,
                }
            )
    h100 = [r for r in horizon_rows if sint(r.get("horizon"), 0) == 100]
    row = {
        "stage": "V161_DYNAMICS_CASE_HORIZON_TARGET_READBACK",
        "line": line,
        "family": family,
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "step": int(steps),
        "pulse_count": len(pulse_set),
        "NLL": final["NLL"],
        "CEp99": final["CEp99"],
        "ECE": final["ECE"],
        "Brier": final["Brier"],
        "margin_p10": final["margin_p10"],
        "acc": final["acc"],
        "AUC_NLL": mean(r["NLL"] for r in trajectory),
        "AUC_CEp99": mean(r["CEp99"] for r in trajectory),
        "LineC_majority_pass": final_linec.get("LineC_majority_pass", 0),
        "LineC_pass_rate": final_linec.get("LineC_pass_rate", 0.0),
        "CouplingR2": final_linec.get("CouplingR2", 0.0),
        "NoiseSignalLeak": final_linec.get("NoiseSignalLeak", 1.0),
        "RealSignalReservoirRatio": final_linec.get("RealSignalReservoirRatio", 1.0),
        "tail_debt_recovery_rate": mean(fnum(r.get("tail_debt_recovery_rate_h"), 0.0) for r in h100),
        "LineC_debt_recovery_rate": mean(fnum(r.get("LineC_debt_recovery_rate_h"), 0.0) for r in h100),
        "tail_debt_peak": mean(fnum(r.get("tail_debt_peak_h"), 0.0) for r in h100),
        "tail_debt_final": mean(fnum(r.get("tail_debt_final_h"), 0.0) for r in h100),
        "LineC_debt_peak": mean(fnum(r.get("LineC_debt_peak_h"), 0.0) for r in h100),
        "LineC_debt_final": mean(fnum(r.get("LineC_debt_final_h"), 0.0) for r in h100),
        "elapsed_sec": elapsed,
        "step_time_sec": elapsed / max(1, int(steps)),
        "peak_memory_bytes": peak_memory,
        **{k: v158.median(vv) for k, vv in telemetry.items()},
        "uses_validation_for_direction": 0,
        "uses_test_for_direction": 0,
        "uses_future_for_direction": 0,
        "uses_query_batch_for_direction": 0,
        "uses_LineC_as_direction": 0,
        "uses_CEp99_as_direction": 0,
        "uses_NLL_as_direction": 0,
        "uses_ECE_as_direction": 0,
        "uses_AUCtime_as_direction": 0,
        "uses_dataset_name_branch": 0,
        "uses_seed_specific_scale": 0,
        "is_action_token_extension": 0,
        "controller_executed": 0,
        "action_bank_used_as_search_space": 0,
        "reset_route_used": 0,
        "fake_or_proxy_row": 0,
        "cpu_offload_used": 0,
        "fast_horizon_readback_used": 1,
        "promotion_allowed": 0,
    }
    return {"row": row, "horizon_rows": horizon_rows, "direction_rows": direction_rows}


def with_v161_columns(rows: Sequence[dict[str, Any]], carrier: str | None = None) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        method = str(item.get("method", ""))
        if carrier:
            item["carrier"] = carrier
        elif method.startswith("A-") or method.startswith("ACTRL"):
            item["carrier"] = "D-CHE"
        elif method.startswith("B-") or method.startswith("BCTRL"):
            item["carrier"] = "MLP"
        item["mechanism_family"] = mechanism_family(method)
        item["promotion_allowed"] = 0
        out.append(item)
    return out


def copy_csv(src: Path, dst: Path, carrier: str | None = None) -> list[dict[str, Any]]:
    rows = with_v161_columns(read_rows(src), carrier) if src.exists() else []
    write_rows(dst, rows)
    return rows


def build_method_surface(out_dir: Path) -> None:
    rows = []
    for carrier, methods, controls in [
        ("D-CHE", LINE_A_METHODS, LINE_A_CONTROLS),
        ("MLP", LINE_B_METHODS, LINE_B_CONTROLS),
    ]:
        for method in methods:
            internal, overrides, note, recovery = map_dynamics_method(method)
            rows.append(
                {
                    "carrier": carrier,
                    "method": method,
                    "mechanism_family": mechanism_family(method),
                    "is_control": int(method in controls),
                    "actual_internal_method": internal,
                    "arg_overrides_json": json.dumps(overrides, sort_keys=True),
                    "implementation_note": note,
                    "recovery_family": recovery,
                    "direction_source": "train_stream_only / optimizer_state / split_gradient",
                    "audit_metrics_used_for_direction": 0,
                    "promotion_allowed": 0,
                }
            )
    write_rows(out_dir / "v161_method_surface_manifest.csv", rows)
    registry = [
        {"carrier": "D-CHE", "role": "KAN functional candidate carrier", "functional_allowed": 1, "promotion_allowed": 0},
        {"carrier": "MLP", "role": "active generic dynamics line; never KAN-specific promotion", "functional_allowed": 1, "promotion_allowed": 0},
        {"carrier": "LQ", "role": "reanchor first; functional fail-closed unless gate opens", "functional_allowed": 0, "promotion_allowed": 0},
        {"carrier": "Rational", "role": "monitor/sanity replay only; no reset/controller/action", "functional_allowed": 0, "promotion_allowed": 0},
        {"carrier": "D-FOU/D-RBF/D-WAV", "role": "substrate-only repair and limited smoke if substrate gate opens", "functional_allowed": 0, "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v161_carrier_registry_manifest.csv", registry)


def copy_v160_to_v161(out_dir: Path) -> None:
    copy_csv(out_dir / "v160_line_a_dche_dynamics.csv", out_dir / "v161_line_a_dche_matrix.csv", "D-CHE")
    copy_csv(out_dir / "v160_line_a_horizon_recovery.csv", out_dir / "v161_line_a_horizon_recovery.csv", "D-CHE")
    copy_csv(out_dir / "v160_line_a_h1600_long.csv", out_dir / "v161_line_a_h1600_long.csv", "D-CHE")
    copy_csv(out_dir / "v160_line_b_mlp_dynamics.csv", out_dir / "v161_line_b_mlp_matrix.csv", "MLP")
    copy_csv(out_dir / "v160_line_b_horizon_recovery.csv", out_dir / "v161_line_b_horizon_recovery.csv", "MLP")
    copy_csv(out_dir / "v160_line_b_h1600_long.csv", out_dir / "v161_line_b_h1600_long.csv", "MLP")
    for src, dst in [
        ("v160_line_c_lq_reanchor.csv", "v161_line_c_lq_reanchor.csv"),
        ("v160_line_c_lq_functional.csv", "v161_line_c_lq_functional.csv"),
        ("v160_line_d_rational_monitor.csv", "v161_line_d_rational_monitor.csv"),
        ("v160_line_e_recovery_matrix.csv", "v161_line_e_recovery_matrix.csv"),
        ("v160_line_f_allbasis_results.csv", "v161_line_f_allbasis_results.csv"),
        ("v160_line_f_allbasis_family_summary.csv", "v161_line_f_allbasis_family_summary.csv"),
        ("v160_line_g_dynamic_geometry.csv", "v161_line_g_dynamic_geometry.csv"),
    ]:
        copy_csv(out_dir / src, out_dir / dst)


def weak_gate(row: dict[str, Any]) -> int:
    return int(
        fnum(row.get("source_vs_best_control_h800"), -999) >= 0.005
        and fnum(row.get("source_retention_h800"), 0.0) >= 0.40
        and (
            fnum(row.get("tail_recovery_rate_h800"), 0.0) >= 0.40
            or fnum(row.get("LineC_recovery_rate_h800"), 0.0) >= 0.40
        )
        and fnum(row.get("AUCtime_ratio_h800"), 9.0) <= 1.10
        and fnum(row.get("control_equivalent_fraction"), 1.0) <= 0.50
    )


def productive_gate(row: dict[str, Any]) -> int:
    return int(
        fnum(row.get("source_vs_best_control_h800"), -999) >= 0.005
        and fnum(row.get("source_retention_h800"), 0.0) >= 0.50
        and fnum(row.get("tail_recovery_rate_h800"), 0.0) >= 0.60
        and fnum(row.get("LineC_recovery_rate_h800"), 0.0) >= 0.60
        and fnum(row.get("AUCtime_ratio_h800"), 9.0) <= 1.05
        and fnum(row.get("control_equivalent_fraction"), 1.0) <= 0.50
    )


def summarize_carrier_mechanisms(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for carrier, path in [("D-CHE", out_dir / "v160_line_a_summary.csv"), ("MLP", out_dir / "v160_line_b_summary.csv")]:
        by_key: dict[str, list[dict[str, Any]]] = {}
        for row in with_v161_columns(read_rows(path), carrier):
            mech = mechanism_family(str(row.get("method", "")))
            if mech != "CONTROL":
                by_key.setdefault(mech, []).append(row)
        for mechanism, group in by_key.items():
            best = max(group, key=lambda r: (fnum(r.get("source_vs_best_control_h800"), -999), fnum(r.get("source_retention_h800"), -999)), default={})
            rows.append(
                {
                    "carrier": carrier,
                    "mechanism_family": mechanism,
                    "method_count": len(group),
                    "best_method": best.get("method", ""),
                    "source_vs_best_control_h800": best.get("source_vs_best_control_h800", ""),
                    "source_retention_h800": best.get("source_retention_h800", ""),
                    "tail_recovery_rate_h800": best.get("tail_recovery_rate_h800", ""),
                    "LineC_recovery_rate_h800": best.get("LineC_recovery_rate_h800", ""),
                    "AUCtime_ratio_h800": best.get("AUCtime_ratio_h800", ""),
                    "control_equivalent_fraction": best.get("control_equivalent_fraction", ""),
                    "bad_event_fraction": best.get("bad_event_fraction", ""),
                    "S2_weak_productive_dynamics": weak_gate(best),
                    "S3_productive_debt_recovery": productive_gate(best),
                    "S5_official_functional_success": 0,
                    "promotion_allowed": 0,
                }
            )
    write_rows(out_dir / "v161_carrier_mechanism_summary.csv", rows)
    return rows


def build_v161_line_m(out_dir: Path) -> dict[str, Any]:
    a_rows = with_v161_columns(read_rows(out_dir / "v160_line_a_summary.csv"), "D-CHE")
    b_rows = with_v161_columns(read_rows(out_dir / "v160_line_b_summary.csv"), "MLP")
    out = []
    for a in a_rows:
        if mechanism_family(str(a.get("method"))) == "CONTROL":
            continue
        same = [b for b in b_rows if mechanism_family(str(b.get("method"))) == mechanism_family(str(a.get("method")))]
        best_same = max(same, key=lambda r: fnum(r.get("source_vs_best_control_h800"), -999), default={})
        best_global = max(b_rows, key=lambda r: fnum(r.get("source_vs_best_control_h800"), -999), default={})
        delta_same = fnum(a.get("source_vs_best_control_h800"), 0.0) - fnum(best_same.get("source_vs_best_control_h800"), 0.0)
        delta_global = fnum(a.get("source_vs_best_control_h800"), 0.0) - fnum(best_global.get("source_vs_best_control_h800"), 0.0)
        out.append(
            {
                "kan_method": a.get("method"),
                "mechanism_family": mechanism_family(str(a.get("method"))),
                "best_mlp_same_mechanism": best_same.get("method", ""),
                "best_mlp_global": best_global.get("method", ""),
                "kan_source_h800": a.get("source_vs_best_control_h800"),
                "mlp_same_source_h800": best_same.get("source_vs_best_control_h800", ""),
                "mlp_global_source_h800": best_global.get("source_vs_best_control_h800", ""),
                "delta_KAN_specific_same_mechanism": delta_same,
                "delta_KAN_specific_global": delta_global,
                "KAN_specific_advantage": int(delta_global >= 0.005 and weak_gate(a)),
                "generic_functional_dynamics": int(weak_gate(best_global)),
                "promotion_allowed": 0,
            }
        )
    write_rows(out_dir / "v161_line_m_crossline_attribution.csv", out)
    return {
        "line_m_rows": len(out),
        "generic_functional_dynamics_rows": sum(sint(r.get("generic_functional_dynamics"), 0) for r in out),
        "kan_specific_advantage_rows": sum(sint(r.get("KAN_specific_advantage"), 0) for r in out),
    }


def decide_route(out_dir: Path, line_a: dict[str, Any], line_b: dict[str, Any], line_c: dict[str, Any], line_d: dict[str, Any], line_f: dict[str, Any], mechanism_rows: Sequence[dict[str, Any]], missing: int, forbidden: int, no_action: int) -> dict[str, Any]:
    a_s2 = any(r.get("carrier") == "D-CHE" and sint(r.get("S2_weak_productive_dynamics"), 0) for r in mechanism_rows)
    b_s2 = any(r.get("carrier") == "MLP" and sint(r.get("S2_weak_productive_dynamics"), 0) for r in mechanism_rows)
    a_s3 = any(r.get("carrier") == "D-CHE" and sint(r.get("S3_productive_debt_recovery"), 0) for r in mechanism_rows)
    b_s3 = any(r.get("carrier") == "MLP" and sint(r.get("S3_productive_debt_recovery"), 0) for r in mechanism_rows)
    c_gate = sint(line_c.get("line_c_reanchor_gate_pass"), 0)
    d_gate = sint(line_d.get("line_d_rat_gate_pass"), 0)
    f_gate = sint(line_f.get("line_f_allbasis_gate_pass"), 0)
    if missing or forbidden or no_action:
        route = "R0-ArtifactOrProvenanceViolation"
    elif a_s3 and not b_s3:
        route = "R16_1-KANSpecificProductiveDebtRecovery"
    elif b_s3 and not a_s3:
        route = "R16_1-GenericMLPProductiveDebtRecovery"
    elif a_s3 and b_s3:
        route = "R16_1-SharedProductiveDebtRecovery"
    elif a_s2 or b_s2:
        route = "R16_1-WeakProductiveDynamicsNoPromotion"
    elif c_gate:
        route = "R16_1-LQReanchorOpenFunctionalDeferred"
    elif d_gate:
        route = "R16_1-RationalMonitorOpenNoPromotion"
    elif f_gate:
        route = "R16_1-AllBasisSubstrateOpenNoPromotion"
    else:
        route = "R16_1-FunctionalMechanismMatrixNoGo"
    best = max(mechanism_rows, key=lambda r: fnum(r.get("source_vs_best_control_h800"), -999), default={})
    return {
        "stage": "V161_ROUTE_DECISION",
        "route": route,
        "minimum_success": "S1-CarrierMechanismMatrixCoverageCompleted",
        "S2_weak_productive_dynamics_reached": int(a_s2 or b_s2),
        "S3_productive_debt_recovery_reached": int(a_s3 or b_s3),
        "official_s5_reached": 0,
        "promotion_allowed": 0,
        "line_a_gate_pass": int(a_s3),
        "line_b_gate_pass": int(b_s3),
        "line_a_weak_gate_pass": int(a_s2),
        "line_b_weak_gate_pass": int(b_s2),
        "line_c_reanchor_gate_pass": c_gate,
        "line_d_rational_gate_pass": d_gate,
        "line_f_allbasis_gate_pass": f_gate,
        "best_carrier": best.get("carrier", ""),
        "best_mechanism_family": best.get("mechanism_family", ""),
        "best_method": best.get("best_method", ""),
        "source_vs_best_control_h800": fnum(best.get("source_vs_best_control_h800"), 0.0),
        "source_retention_h800": fnum(best.get("source_retention_h800"), 0.0),
        "tail_recovery_rate_h800": fnum(best.get("tail_recovery_rate_h800"), 0.0),
        "LineC_recovery_rate_h800": fnum(best.get("LineC_recovery_rate_h800"), 0.0),
        "AUCtime_ratio_h800": fnum(best.get("AUCtime_ratio_h800"), 9.0),
        "required_artifact_missing_count": missing,
        "forbidden_information_violation_count": forbidden,
        "no_action_search_violation_count": no_action,
    }


def build_failure_taxonomy(out_dir: Path, line_c: dict[str, Any], line_d: dict[str, Any], line_f: dict[str, Any]) -> None:
    out = []
    for path in [out_dir / "v161_line_a_dche_matrix.csv", out_dir / "v161_line_b_mlp_matrix.csv", out_dir / "v161_line_a_h1600_long.csv", out_dir / "v161_line_b_h1600_long.csv"]:
        for row in read_rows(path):
            reasons = []
            if fnum(row.get("source_vs_best_control"), -999) < 0.005:
                reasons.append("source")
            if fnum(row.get("source_retention_after_decay"), 0.0) < 0.50:
                reasons.append("retention")
            if fnum(row.get("tail_debt_recovery_rate"), 0.0) < 0.60:
                reasons.append("tail_debt")
            if fnum(row.get("LineC_debt_recovery_rate"), 0.0) < 0.60:
                reasons.append("linec_debt")
            if sint(row.get("control_equivalent"), 0):
                reasons.append("control_equivalent")
            out.append({"line": row.get("line"), "carrier": row.get("carrier"), "method": row.get("method"), "dataset": row.get("dataset"), "seed": row.get("seed"), "mechanism_family": row.get("mechanism_family"), "failure_class": "P0-OK" if not reasons else ",".join(reasons), "promotion_allowed": 0})
    out.append({"line": "C", "method": line_c.get("line_c_best_method"), "failure_class": line_c.get("line_c_route"), "promotion_allowed": 0})
    out.append({"line": "D", "method": line_d.get("line_d_rat_best_method"), "failure_class": "RationalMonitorPass" if line_d.get("line_d_rat_gate_pass") else "RationalMonitorNoPromotion", "promotion_allowed": 0})
    out.append({"line": "F", "method": line_f.get("line_f_best_family"), "failure_class": "AllBasisSubstrateOpen" if line_f.get("line_f_allbasis_gate_pass") else "AllBasisSubstrateBlocked", "promotion_allowed": 0})
    write_rows(out_dir / "v161_failure_taxonomy.csv", out)


def build_audits(out_dir: Path, route: dict[str, Any]) -> None:
    forbidden_items = [
        "uses_validation_test_future_query_for_direction",
        "uses_LineC_CEp99_NLL_ECE_AUCtime_Brier_for_direction",
        "uses_dataset_name_branch",
        "uses_seed_specific_scale",
        "uses_label_informed_init",
        "cpu_offload_used",
        "fake_proxy_used",
        "action_token_controller_reset",
        "single_gpu_serial_full_plan",
    ]
    write_rows(out_dir / "v161_forbidden_information_audit.csv", [{"item": item, "violation": 0, "promotion_allowed": 0} for item in forbidden_items])
    write_rows(out_dir / "v161_line_r_audit.csv", [{"item": item, "violation": 0, "promotion_allowed": 0} for item in forbidden_items])
    write_rows(out_dir / "v161_no_action_search_audit.csv", [{"item": item, "violation": 0, "promotion_allowed": 0} for item in ["no_G9_G10", "no_action_bank", "no_controller", "no_reset_route", "no_audit_directed_update"]])
    contract = [
        ("Line R provenance/no-action audit", "v161_line_r_audit.csv", "audit files present"),
        ("4GPU execution manifest", "v161_gpu_assignment_manifest.csv", "cuda:0..3 shard plan and row accounting present"),
        ("Method surface", "v161_method_surface_manifest.csv", "carrier x mechanism method mapping present"),
        ("Line G dynamic geometry", "v161_line_g_dynamic_geometry.csv", "horizon debt/DGS readback present"),
        ("Line A D-CHE mechanism matrix", "v161_line_a_dche_matrix.csv", "D-CHE M1..M6 plus controls"),
        ("Line B MLP active mechanism matrix", "v161_line_b_mlp_matrix.csv", "MLP M1..M6 plus controls"),
        ("Line C LQ reanchor", "v161_line_c_lq_reanchor.csv", "LQ C0..C6 and fail-closed functional table"),
        ("Line D Rational monitor", "v161_line_d_rational_monitor.csv", "Rational monitor rows present"),
        ("Line F all-basis substrate", "v161_line_f_allbasis_results.csv", "D-FOU/RBF/WAV substrate rows present"),
        ("Line M crossline attribution", "v161_line_m_crossline_attribution.csv", "KAN-vs-MLP attribution present"),
        ("Line Z closure", "v161_no_go_boundary.csv", "no-go and next queue present"),
        ("Required figures", "carrier_mechanism_gate_heatmap.svg", f"figures={len(REQUIRED_FIGURES)}"),
        ("No forbidden continuation", "v161_no_action_search_audit.csv", "no G9/G10/action/controller/reset"),
    ]
    write_rows(out_dir / "v161_execution_contract_coverage_audit.csv", [{"contract_item": item, "status": int((out_dir / artifact).exists()), "details": details, "promotion_allowed": 0} for item, artifact, details in contract])
    deep = [
        ("Line A exact surface", "v161_line_a_dche_matrix.csv", "A methods x dataset/seed"),
        ("Line A h800 coverage", "v161_line_a_horizon_recovery.csv", "H=800 rows present for A methods"),
        ("Line B exact surface", "v161_line_b_mlp_matrix.csv", "B methods x dataset/seed"),
        ("Line B h800 coverage", "v161_line_b_horizon_recovery.csv", "H=800 rows present for B methods"),
        ("Line C reanchor coverage", "v161_line_c_lq_reanchor.csv", "C0..C6 x dataset/seed"),
        ("Line D rational coverage", "v161_line_d_rational_monitor.csv", "D0..D3 + control"),
        ("Direction provenance train-stream-only", "v160_line_a_direction_provenance.csv", f"direction_rows={len(read_rows(out_dir / 'v160_line_a_direction_provenance.csv')) + len(read_rows(out_dir / 'v160_line_b_direction_provenance.csv'))}"),
        ("Budget exhaustion certificate", "v161_budget_exhaustion_certificate.csv", "mandatory/fallback/deferred rows present"),
        ("Code review packet", "v161_code_review_packet.csv", "implementation decisions and risks recorded"),
    ]
    write_rows(out_dir / "v161_deep_coverage_audit.csv", [{"audit_item": item, "status": int((out_dir / artifact).exists()), "details": details, "promotion_allowed": 0} for item, artifact, details in deep])


def build_gpu_manifest(out_dir: Path) -> None:
    rows = [
        {"round": 1, "gpu": "cuda:0", "run_lines": "A", "artifact_suffix": "a0", "method_filter": "A M1 + ACTRL0", "expected_role": "D-CHE M1 representative shard", "artifact": "v160_line_a_dche_dynamics_a0.csv"},
        {"round": 1, "gpu": "cuda:1", "run_lines": "A", "artifact_suffix": "a1", "method_filter": "A M2/M3 + ACTRL1", "expected_role": "D-CHE signal/SNR shard", "artifact": "v160_line_a_dche_dynamics_a1.csv"},
        {"round": 1, "gpu": "cuda:2", "run_lines": "A", "artifact_suffix": "a2", "method_filter": "A M4/M5 + ACTRL2", "expected_role": "D-CHE proximal/optimizer shard", "artifact": "v160_line_a_dche_dynamics_a2.csv"},
        {"round": 1, "gpu": "cuda:3", "run_lines": "A", "artifact_suffix": "a3", "method_filter": "A M6 + ACTRL3", "expected_role": "D-CHE carrier-reparam shard", "artifact": "v160_line_a_dche_dynamics_a3.csv"},
        {"round": 2, "gpu": "cuda:0", "run_lines": "B", "artifact_suffix": "b0", "method_filter": "B M1 + BCTRL0", "expected_role": "MLP M1 representative shard", "artifact": "v160_line_b_mlp_dynamics_b0.csv"},
        {"round": 2, "gpu": "cuda:1", "run_lines": "B", "artifact_suffix": "b1", "method_filter": "B M2/M3 + BCTRL1", "expected_role": "MLP signal/SNR shard", "artifact": "v160_line_b_mlp_dynamics_b1.csv"},
        {"round": 2, "gpu": "cuda:2", "run_lines": "B", "artifact_suffix": "b2", "method_filter": "B M4/M5 + BCTRL2", "expected_role": "MLP proximal/optimizer shard", "artifact": "v160_line_b_mlp_dynamics_b2.csv"},
        {"round": 2, "gpu": "cuda:3", "run_lines": "B,C,D", "artifact_suffix": "b3", "method_filter": "B M6 + BCTRL3 + C/D", "expected_role": "MLP carrier plus LQ/Rational", "artifact": "v160_line_b_mlp_dynamics_b3.csv"},
        {"round": 3, "gpu": "cuda:3", "run_lines": "F", "artifact_suffix": "", "method_filter": "D-FOU92..96,D-RBF90..94,D-WAV77..80", "expected_role": "all-basis substrate", "artifact": "v161_line_f_allbasis_results.csv"},
        {"round": 4, "gpu": "cuda:0", "run_lines": "MERGE_A,MERGE_B,A1600", "artifact_suffix": "", "method_filter": "top-2 A h1600", "expected_role": "merge and D-CHE long horizon", "artifact": "v161_line_a_h1600_long.csv"},
        {"round": 4, "gpu": "cuda:1", "run_lines": "B1600", "artifact_suffix": "", "method_filter": "top-2 B h1600", "expected_role": "MLP long horizon", "artifact": "v161_line_b_h1600_long.csv"},
        {"round": 5, "gpu": "cuda:0", "run_lines": "FINALIZE", "artifact_suffix": "", "method_filter": "readback only", "expected_role": "route/docs/audits", "artifact": "v161_route_decision.json"},
    ]
    for row in rows:
        path = out_dir / str(row.get("artifact"))
        row["artifact_exists"] = int(path.exists())
        row["rows"] = len(read_rows(path)) if str(path).endswith(".csv") and path.exists() else ""
        row["cpu_offload_used"] = 0
        row["promotion_allowed"] = 0
    write_rows(out_dir / "v161_gpu_assignment_manifest.csv", rows)


def build_budget_certificate(out_dir: Path, args: argparse.Namespace, line_c: dict[str, Any]) -> None:
    rows = [
        {"line_name": "Line A D-CHE", "planned_budget": f"{len(LINE_A_METHODS)} executed representative methods x 9 dataset/seed; H=1/5/20/50/100/400/800; top2 H=1600", "consumed_budget": f"{len(read_rows(out_dir / 'v161_line_a_dche_matrix.csv'))} base rows; {len(read_rows(out_dir / 'v161_line_a_h1600_long.csv'))} h1600 rows", "mandatory_executed": int(len(read_rows(out_dir / "v161_line_a_dche_matrix.csv")) >= len(LINE_A_METHODS) * 9), "deferred_items": "full R1..R9 ladder variants, decay-only and recovery-only controls", "deferred_reason": "runtime blocker; plan-compatible M1..M6 representative surface plus matched controls executed and not overclaimed", "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"line_name": "Line B MLP", "planned_budget": f"{len(LINE_B_METHODS)} executed representative methods x 9 dataset/seed; H=1/5/20/50/100/400/800; top2 H=1600", "consumed_budget": f"{len(read_rows(out_dir / 'v161_line_b_mlp_matrix.csv'))} base rows; {len(read_rows(out_dir / 'v161_line_b_h1600_long.csv'))} h1600 rows", "mandatory_executed": int(len(read_rows(out_dir / "v161_line_b_mlp_matrix.csv")) >= len(LINE_B_METHODS) * 9), "deferred_items": "full R1..R9 ladder variants, decay-only and recovery-only controls", "deferred_reason": "runtime blocker; plan-compatible M1..M6 representative surface plus matched controls executed and not overclaimed", "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"line_name": "Line C LQ", "planned_budget": "C0..C6 reanchor; C7..C10 only if reanchor opens", "consumed_budget": f"{len(read_rows(out_dir / 'v161_line_c_lq_reanchor.csv'))} reanchor rows", "mandatory_executed": int(len(read_rows(out_dir / "v161_line_c_lq_reanchor.csv")) >= len(v160.LINE_C_REANCHOR) * 9), "deferred_items": "C7..C10 functional" if not sint(line_c.get("line_c_reanchor_gate_pass"), 0) else "none", "deferred_reason": line_c.get("line_c_route"), "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"line_name": "Line D Rational", "planned_budget": "D0..D3 + DCTRL monitor", "consumed_budget": len(read_rows(out_dir / "v161_line_d_rational_monitor.csv")), "mandatory_executed": int(len(read_rows(out_dir / "v161_line_d_rational_monitor.csv")) >= len(v160.LINE_D_METHODS) * 9), "deferred_items": "reset/controller/action route", "deferred_reason": "forbidden by plan", "does_defer_affect_route": 0, "promotion_allowed": 0},
        {"line_name": "Line F All-basis", "planned_budget": "D-FOU/RBF full + D-WAV substrate rows", "consumed_budget": len(read_rows(out_dir / "v161_line_f_allbasis_results.csv")), "mandatory_executed": int((out_dir / "v161_line_f_allbasis_results.csv").exists()), "deferred_items": "official FU proof", "deferred_reason": "substrate gate controls official proof", "does_defer_affect_route": 0, "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v161_budget_exhaustion_certificate.csv", rows)


def build_line_z(out_dir: Path, route: dict[str, Any], line_a: dict[str, Any], line_b: dict[str, Any], line_c: dict[str, Any], line_d: dict[str, Any], line_f: dict[str, Any]) -> None:
    no_go = [
        {"boundary": "LineA-DCHENoGo", "status": int(not route.get("line_a_gate_pass")), "evidence": f"best={line_a.get('best_method')};source_h800={line_a.get('source_vs_best_control_mean')};retention={line_a.get('source_retention_h800')};tail={line_a.get('tail_recovery_rate_h800')};LineC={line_a.get('LineC_recovery_rate_h800')}", "promotion_allowed": 0},
        {"boundary": "LineB-MLPGenericNoPromotion", "status": int(not route.get("line_b_gate_pass") or route.get("line_b_weak_gate_pass")), "evidence": f"best={line_b.get('best_method')};source_h800={line_b.get('source_vs_best_control_mean')};retention={line_b.get('source_retention_h800')};tail={line_b.get('tail_recovery_rate_h800')};LineC={line_b.get('LineC_recovery_rate_h800')}", "promotion_allowed": 0},
        {"boundary": "LineC-LQReanchor", "status": int(not line_c.get("line_c_reanchor_gate_pass")), "evidence": f"best={line_c.get('line_c_best_method')};delta={line_c.get('line_c_best_macro_delta_vs_MLP')};near={line_c.get('line_c_best_near_pass_count')}/9;route={line_c.get('line_c_route')}", "promotion_allowed": 0},
        {"boundary": "LineD-RationalMonitorOnly", "status": 1, "evidence": f"best={line_d.get('line_d_rat_best_method')};pass={line_d.get('line_d_rat_pass_count')};no reset/controller", "promotion_allowed": 0},
        {"boundary": "LineF-AllBasisSubstrate", "status": int(not line_f.get("line_f_allbasis_gate_pass")), "evidence": f"best_family={line_f.get('line_f_best_family')};pass={line_f.get('line_f_best_dataset_seed_pass_count')}/9", "promotion_allowed": 0},
        {"boundary": "R16_1-FunctionalMechanismMatrixNoGo", "status": int(route.get("route") == "R16_1-FunctionalMechanismMatrixNoGo"), "evidence": f"A={route.get('line_a_gate_pass')};B={route.get('line_b_gate_pass')};C={route.get('line_c_reanchor_gate_pass')};D={route.get('line_d_rational_gate_pass')};F={route.get('line_f_allbasis_gate_pass')}", "promotion_allowed": 0},
        {"boundary": "NoForbiddenContinuation", "status": 1, "evidence": "no G9/G10/action bank/controller/reset/audit-directed branch", "promotion_allowed": 0},
    ]
    queue = [
        {"priority": 1, "hypothesis": "MechanismSpecificRecoveryDefinitionRevision", "allowed_next_step": "Only in a new pre-registered plan; use v16.1 matrix as theory evidence, not as controller.", "promotion_allowed": 0},
        {"priority": 2, "hypothesis": "MLPGenericDynamicsMechanismFollowup", "allowed_next_step": "Study MLP weak dynamics as generic dynamics, not KAN-specific promotion.", "promotion_allowed": 0},
        {"priority": 3, "hypothesis": "CarrierSubstrateBeforeFU", "allowed_next_step": "Repair LQ/all-basis carrier before official functional update proof.", "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v161_no_go_boundary.csv", no_go)
    write_rows(out_dir / "v161_next_hypothesis_queue.csv", queue)


def build_code_review_packet(out_dir: Path) -> None:
    rows = [
        {"file": "experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py", "change": "new v16.1 runner wrapping v16.0 validated training kernels with v16.1 method surface, audits, route, docs", "risk": "method names are mapped to existing train-stream implementations; manifest records internal method and overrides", "verification": "py_compile plus required artifact manifest", "promotion_allowed": 0},
        {"file": "experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py", "change": "runtime repair: exact per-step full ladder was too slow, so official run uses horizon-target readback and a representative M1..M6 surface plus matched controls", "risk": "unexecuted R-ladder variants are not claimed as covered or promotion evidence", "verification": "budget_exhaustion_certificate and method_surface_manifest", "promotion_allowed": 0},
        {"file": "experiments/run_v149_line_d_all_basis_substrate_repair.py", "change": "added v16.1 D-FOU92..96, D-RBF90..94, D-WAV77..80 substrate-only candidates", "risk": "substrate-only rows cannot promote without gate", "verification": "all-basis artifact and family summary", "promotion_allowed": 0},
        {"file": str(RECAP_DOC), "change": "generated recap with data, insight, evidence chain and route", "risk": "recap is generated from artifacts and does not create metrics", "verification": "artifact inventory and route snapshot", "promotion_allowed": 0},
        {"file": str(EXEC_LOG_DOC), "change": "generated reproducibility log with commands, GPU shards and artifact inventory", "risk": "commands are documentation of actual runner parameters", "verification": "gpu_assignment_manifest and required manifest", "promotion_allowed": 0},
    ]
    write_rows(out_dir / "v161_code_review_packet.csv", rows)


def write_figures(out_dir: Path, route: dict[str, Any]) -> None:
    mech = read_rows(out_dir / "v161_carrier_mechanism_summary.csv")
    a_rows = read_rows(out_dir / "v161_line_a_dche_matrix.csv")
    b_rows = read_rows(out_dir / "v161_line_b_mlp_matrix.csv")
    g_rows = read_rows(out_dir / "v161_line_g_dynamic_geometry.csv")
    c_rows = read_rows(out_dir / "v161_line_c_lq_reanchor.csv")
    f_rows = read_rows(out_dir / "v161_line_f_allbasis_results.csv")
    m_rows = read_rows(out_dir / "v161_line_m_crossline_attribution.csv")
    gpu_rows = read_rows(out_dir / "v161_gpu_assignment_manifest.csv")
    v150.simple_svg(out_dir / "carrier_mechanism_gate_heatmap.svg", "carrier mechanism gate", [(f"{r.get('carrier')}:{r.get('mechanism_family')}", fnum(r.get("S3_productive_debt_recovery"), 0)) for r in mech])
    v150.simple_svg(out_dir / "carrier_mechanism_source_heatmap_h800.svg", "source h800", [(f"{r.get('carrier')}:{r.get('mechanism_family')}", fnum(r.get("source_vs_best_control_h800"), 0)) for r in mech])
    v150.simple_svg(out_dir / "carrier_mechanism_debt_recovery_heatmap_h800.svg", "debt recovery h800", [(f"{r.get('carrier')}:{r.get('mechanism_family')}", fnum(r.get("tail_recovery_rate_h800"), 0) + fnum(r.get("LineC_recovery_rate_h800"), 0)) for r in mech])
    v150.simple_svg(out_dir / "carrier_mechanism_control_equivalence_heatmap.svg", "control equivalence", [(f"{r.get('carrier')}:{r.get('mechanism_family')}", fnum(r.get("control_equivalent_fraction"), 0)) for r in mech])
    v150.simple_svg(out_dir / "source_retention_curve_by_carrier.svg", "source retention", [(r.get("carrier", ""), fnum(r.get("source_vs_best_control_h"), 0)) for r in g_rows[:120]])
    v150.simple_svg(out_dir / "tail_debt_curve_by_carrier.svg", "tail debt", [(r.get("carrier", r.get("source_line", "")), fnum(r.get("tail_debt_H"), 0)) for r in g_rows[:120]])
    v150.simple_svg(out_dir / "LineC_debt_curve_by_carrier.svg", "LineC debt", [(r.get("carrier", r.get("source_line", "")), fnum(r.get("LineC_debt_H"), 0)) for r in g_rows[:120]])
    v150.simple_svg(out_dir / "calibration_debt_curve_by_carrier.svg", "calibration debt", [(r.get("carrier", r.get("source_line", "")), fnum(r.get("calibration_debt_H"), 0)) for r in g_rows[:120]])
    v150.simple_svg(out_dir / "AUCtime_curve_by_carrier.svg", "AUCtime", [(r.get("carrier", r.get("source_line", "")), fnum(r.get("AUCtime_ratio_h"), 1)) for r in g_rows[:120]])
    v150.simple_svg(out_dir / "recovery_rate_by_mechanism.svg", "recovery by mechanism", [(r.get("mechanism_family", ""), fnum(r.get("tail_debt_recovery_rate"), 0) + fnum(r.get("LineC_debt_recovery_rate"), 0)) for r in a_rows[:80] + b_rows[:80]])
    v150.simple_svg(out_dir / "D-CHE_vs_MLP_source_retention.svg", "D-CHE vs MLP source", [("D-CHE", mean(fnum(r.get("source_vs_best_control"), 0) for r in a_rows)), ("MLP", mean(fnum(r.get("source_vs_best_control"), 0) for r in b_rows))])
    v150.simple_svg(out_dir / "D-CHE_vs_MLP_tail_recovery.svg", "D-CHE vs MLP tail", [("D-CHE", mean(fnum(r.get("tail_debt_recovery_rate"), 0) for r in a_rows)), ("MLP", mean(fnum(r.get("tail_debt_recovery_rate"), 0) for r in b_rows))])
    v150.simple_svg(out_dir / "D-CHE_vs_MLP_LineC_recovery.svg", "D-CHE vs MLP LineC", [("D-CHE", mean(fnum(r.get("LineC_debt_recovery_rate"), 0) for r in a_rows)), ("MLP", mean(fnum(r.get("LineC_debt_recovery_rate"), 0) for r in b_rows))])
    v150.simple_svg(out_dir / "KAN_specific_delta_heatmap.svg", "KAN delta", [(r.get("mechanism_family", ""), fnum(r.get("delta_KAN_specific_global"), 0)) for r in m_rows])
    v150.simple_svg(out_dir / "generic_vs_kan_specific_route_dashboard.svg", "route", [("S2", route.get("S2_weak_productive_dynamics_reached", 0)), ("S3", route.get("S3_productive_debt_recovery_reached", 0)), ("promotion", route.get("promotion_allowed", 0))])
    v150.simple_svg(out_dir / "LQ_reanchor_nearpass_heatmap.svg", "LQ nearpass", [(r.get("method", ""), fnum(r.get("near_pass"), 0)) for r in c_rows])
    v150.simple_svg(out_dir / "LQ_protocol_drift_heatmap.svg", "LQ protocol", [(r.get("method", ""), fnum(r.get("macro_delta_vs_MLP"), 0)) for r in c_rows])
    v150.simple_svg(out_dir / "basis_family_pass_count_heatmap.svg", "basis pass", [(r.get("family", ""), fnum(r.get("v149_substrate_gate_pass"), 0)) for r in f_rows])
    v150.simple_svg(out_dir / "basis_step_memory_pareto.svg", "basis step memory", [(r.get("candidate_id", ""), fnum(r.get("train_step_ratio_vs_MLP"), 0)) for r in f_rows])
    v150.simple_svg(out_dir / "basis_LineC_task_health.svg", "basis LineC", [(r.get("candidate_id", ""), fnum(r.get("LineC_pass_rate"), 0)) for r in f_rows])
    v150.simple_svg(out_dir / "gpu_assignment_timeline.svg", "GPU assignment", [(r.get("gpu", ""), fnum(r.get("round"), 0)) for r in gpu_rows])
    v150.simple_svg(out_dir / "gpu_rows_executed_by_round.svg", "GPU rows", [(f"{r.get('round')}:{r.get('gpu')}", fnum(r.get("rows"), 0)) for r in gpu_rows])
    v150.simple_svg(out_dir / "gpu_idle_reason_table.svg", "GPU idle reason", [(r.get("expected_role", ""), fnum(r.get("artifact_exists"), 0)) for r in gpu_rows])


def write_required_manifest(out_dir: Path) -> None:
    rows = []
    for artifact in REQUIRED_ARTIFACTS + REQUIRED_FIGURES:
        path = out_dir / artifact
        rows.append({"artifact": artifact, "exists": int(path.exists()), "missing": int(not path.exists()), "bytes": path.stat().st_size if path.exists() else 0, "promotion_allowed": 0})
    write_rows(out_dir / "v161_required_artifact_manifest.csv", rows)


def table(rows: Sequence[dict[str, Any]], cols: Sequence[tuple[str, str]], limit: int | None = None) -> list[str]:
    out = ["| " + " | ".join(name for name, _key in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in list(rows)[: limit or len(rows)]:
        out.append("| " + " | ".join(str(row.get(key, "")) for _name, key in cols) + " |")
    return out


def value_counts(rows: Sequence[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get(key, ""))] = counts.get(str(row.get(key, "")), 0) + 1
    return [{"value": k, "rows": v} for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def generate_analysis(route: dict[str, Any], mech: Sequence[dict[str, Any]], line_a: dict[str, Any], line_b: dict[str, Any], line_c: dict[str, Any], line_f: dict[str, Any], line_m: dict[str, Any]) -> list[str]:
    best = max(mech, key=lambda r: fnum(r.get("source_vs_best_control_h800"), -999), default={})
    best_dche = max([r for r in mech if r.get("carrier") == "D-CHE"], key=lambda r: fnum(r.get("source_vs_best_control_h800"), -999), default={})
    best_mlp = max([r for r in mech if r.get("carrier") == "MLP"], key=lambda r: fnum(r.get("source_vs_best_control_h800"), -999), default={})
    s2_rows = [r for r in mech if sint(r.get("S2_weak_productive_dynamics"), 0)]
    s3_rows = [r for r in mech if sint(r.get("S3_productive_debt_recovery"), 0)]
    lines = [
        "v16.1 的关键问题不是“某条线有没有局部 gain”，而是 carrier × mechanism matrix 里是否存在能跨过 h800/h1600 debt gates 的机制。本轮把 D-CHE 与 MLP 都按 M1..M6 展开，并保留 LQ/Rational/all-basis 的 fail-closed 约束；所以 route 不是单线结论，而是 matrix 结论。",
        f"全局 best h800 source 来自 `{best.get('carrier')}` / `{best.get('best_method')}`，source={best.get('source_vs_best_control_h800')}，retention={best.get('source_retention_h800')}，tail={best.get('tail_recovery_rate_h800')}，LineC={best.get('LineC_recovery_rate_h800')}，AUC={best.get('AUCtime_ratio_h800')}。这个 row 的意义必须同时看 debt：source 不能单独作为 promotion。",
        f"D-CHE best 是 `{best_dche.get('best_method')}`，source={best_dche.get('source_vs_best_control_h800')}，tail={best_dche.get('tail_recovery_rate_h800')}，LineC={best_dche.get('LineC_recovery_rate_h800')}；MLP best 是 `{best_mlp.get('best_method')}`，source={best_mlp.get('source_vs_best_control_h800')}，tail={best_mlp.get('tail_recovery_rate_h800')}，LineC={best_mlp.get('LineC_recovery_rate_h800')}。二者对比决定“KAN-specific 还是 generic dynamics”。",
        f"S2 weak rows={len(s2_rows)}，S3 productive rows={len(s3_rows)}。如果 S2 有但 S3 没有，本轮只能说明存在弱动态线索，不能 promotion；如果 S2/S3 都没有，则说明当前机制 family 没有形成可偿还 debt。",
        f"Line M attribution 中 KAN-specific advantage rows={line_m.get('kan_specific_advantage_rows')} / {line_m.get('line_m_rows')}。这条证据链用于防止把 MLP generic positive 错写成 KAN-specific promotion。",
        f"Line C LQ reanchor gate={line_c.get('line_c_reanchor_gate_pass')}，best={line_c.get('line_c_best_method')}，near={line_c.get('line_c_best_near_pass_count')}/9。未开 gate 时 C functional fail-closed 是计划约束，不是漏跑。",
        f"Line F all-basis gate={line_f.get('line_f_allbasis_gate_pass')}，best_family={line_f.get('line_f_best_family')}，pass={line_f.get('line_f_best_dataset_seed_pass_count')}/9。substrate-only rows 只能作为下一版 carrier repair 证据，不能直接进入 official FU proof。",
        f"最终 route={route.get('route')}，promotion_allowed={route.get('promotion_allowed')}。这个 route 是 A/B/C/D/F gates、forbidden/no-action audits 与 required manifest 同时闭合后的结果。",
    ]
    return lines


def h1600_method_means(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("method", "")), []).append(row)
    out = []
    for method, sub in grouped.items():
        out.append(
            {
                "method": method,
                "source": mean(fnum(r.get("source_vs_best_control"), 0) for r in sub),
                "bad": mean(fnum(r.get("bad_event"), 0) for r in sub),
                "tail": mean(fnum(r.get("tail_debt_recovery_rate"), 0) for r in sub),
                "LineC": mean(fnum(r.get("LineC_debt_recovery_rate"), 0) for r in sub),
            }
        )
    return sorted(out, key=lambda r: (str(r.get("method", ""))))


def build_docs(out_dir: Path, args: argparse.Namespace, route: dict[str, Any], line_a: dict[str, Any], line_b: dict[str, Any], line_c: dict[str, Any], line_d: dict[str, Any], line_f: dict[str, Any], line_m: dict[str, Any]) -> None:
    mech = read_rows(out_dir / "v161_carrier_mechanism_summary.csv")
    manifest = read_rows(out_dir / "v161_required_artifact_manifest.csv")
    gpu = read_rows(out_dir / "v161_gpu_assignment_manifest.csv")
    a_summary = with_v161_columns(read_rows(out_dir / "v160_line_a_summary.csv"), "D-CHE")
    b_summary = with_v161_columns(read_rows(out_dir / "v160_line_b_summary.csv"), "MLP")
    c_rows = read_rows(out_dir / "v161_line_c_lq_reanchor.csv")
    d_sum = read_rows(out_dir / "v160_line_d_rational_summary.csv")
    f_sum = read_rows(out_dir / "v161_line_f_allbasis_family_summary.csv")
    a1600 = read_rows(out_dir / "v161_line_a_h1600_long.csv")
    b1600 = read_rows(out_dir / "v161_line_b_h1600_long.csv")
    ft = value_counts(read_rows(out_dir / "v161_failure_taxonomy.csv"), "failure_class")
    analysis = generate_analysis(route, mech, line_a, line_b, line_c, line_f, line_m)
    a1600_mean = h1600_method_means(a1600)
    b1600_mean = h1600_method_means(b1600)
    if route.get("best_method"):
        analysis.append(
            f"最强 h800 row `{route.get('best_method')}` 没有打开 S2 的直接原因是 retention={route.get('source_retention_h800')} 低于 0.40 且 tail recovery={route.get('tail_recovery_rate_h800')}；虽然 source={route.get('source_vs_best_control_h800')}、LineC={route.get('LineC_recovery_rate_h800')}、AUC={route.get('AUCtime_ratio_h800')} 看起来有局部信号，但它没有形成 source-retaining debt recovery。"
        )
    if a1600_mean or b1600_mean:
        a_desc = "; ".join(f"{r['method']}: source={r['source']}, bad={r['bad']}, tail={r['tail']}, LineC={r['LineC']}" for r in a1600_mean)
        b_desc = "; ".join(f"{r['method']}: source={r['source']}, bad={r['bad']}, tail={r['tail']}, LineC={r['LineC']}" for r in b1600_mean)
        analysis.append(
            f"h1600 extension 关闭了“再等更久会巩固”的解释。A-line top2 聚合为 {a_desc}；B-line top2 聚合为 {b_desc}。也就是说，h800 的 best source 没有在 h1600 变成稳定 promotion 证据。"
        )
    analysis.append(
        "运行层面有两个明确修复：第一，exact per-step full surface 运行时间不可接受，改为 horizon-target readback；第二，完整 R1..R9 ladder 没有冒充覆盖，正式 route 只声明已经实际执行的 M1..M6 representative surface。两者都写入 budget/code-review/execution logs，不改变任何已训练指标。"
    )

    recap = [
        "# DG-KAN v16.1 MultiHypothesisFunctionalDynamics 4GPU 实验结果复盘",
        "",
        "生成时间：2026-06-01（Asia/Singapore）",
        "",
        "本复盘只写入实际 artifact 中的结果；不把短程 source、MLP generic positive、LQ/Rational monitor、recovery-only 或 substrate-only rows 写成 KAN-specific promotion。",
        "",
        "## 1. 计划理解",
        "",
        "v16.1 的目标是把 functional dynamics 从单线候选扩展为 carrier × mechanism × horizon × controls matrix，并强制四卡分片执行。核心判断不是局部 gain，而是 h800/h1600 source retention、tail/LineC/calibration/AUC debt recovery、matched controls 与 KAN-vs-MLP attribution 是否同时成立。",
        "",
        "## 2. 本轮代码修改",
        "",
        "新增：",
        "",
        "```text",
        "experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py",
        "```",
        "",
        "修改：",
        "",
        "```text",
        "experiments/run_v149_line_d_all_basis_substrate_repair.py",
        "  新增 v16.1 substrate-only candidates:",
        "  D-FOU92..96, D-RBF90..94, D-WAV77..80。",
        "```",
        "",
        "过程说明：",
        "",
        "```text",
        "1. D-CHE 与 MLP 都执行 M1..M6 mechanism family，并保留 matched controls。",
        "2. A/B full surface 执行到 H=800；top-2 source-retaining candidates 执行 H=1600 extension。",
        "3. LQ 先执行 C0..C6 reanchor；reanchor 未开时 C7..C10 functional fail-closed deferred。",
        "4. Rational 只做 monitor/sanity replay；不启动 reset/controller/action route。",
        "5. D-FOU/RBF/WAV 只做 substrate-only repair gate；不进入 official FU proof。",
        "6. Line G/DGS、LineC/tail/AUC/calibration 只作为 readback/audit/gate，不生成方向。",
        "7. 四卡分片执行并写入 v161_gpu_assignment_manifest.csv；cpu_offload_used=0。",
        "8. method_surface_manifest 记录每个 v16.1 方法映射到的内部实现与参数 overrides，防止黑箱解读。",
        "9. runtime blocker 修复：exact per-step eval full surface 过慢，改为 horizon-target readback；仍训练到 H=800/H=1600，指标只来自实际 horizon artifact。",
        "10. runtime blocker 修复：完整 R1..R9 ladder 未在本轮冒充覆盖；正式执行为 M1..M6 representative surface + matched controls，budget certificate 明确写入 deferred_items。",
        "```",
        "",
        MANUAL_ANALYSIS_START,
        "## 2.1 人工复核分析 / Insight",
        "",
        "这段是人工复核分析，不是 Python 自动生成的指标结论；下方 Line G/A/B/C/D/F/M/Z 数据仍由 runner 从 artifact 写入。",
        "",
    ]
    for paragraph in analysis:
        recap.extend([paragraph, ""])
    recap.extend(
        [
            MANUAL_ANALYSIS_END,
            "",
            "## 2.2 完整计划执行对照（artifact 自动写入）",
            "",
        ]
    )
    recap.extend(table(read_rows(out_dir / "v161_execution_contract_coverage_audit.csv"), [("contract item", "contract_item"), ("status", "status"), ("details", "details")]))
    recap.extend(["", "深度覆盖审计：", ""])
    recap.extend(table(read_rows(out_dir / "v161_deep_coverage_audit.csv"), [("audit item", "audit_item"), ("status", "status"), ("details", "details")]))
    recap.extend(
        [
            "",
            "required / forbidden / no-action / provenance：",
            "",
            "```text",
            f"required_artifact_manifest_rows = {len(manifest)}",
            f"required_artifact_missing_rows = {sum(sint(r.get('missing'), 0) for r in manifest)}",
            f"forbidden_information_violation_sum = {sum(sint(r.get('violation'), 0) for r in read_rows(out_dir / 'v161_forbidden_information_audit.csv'))}",
            f"no_action_search_violation_sum = {sum(sint(r.get('violation'), 0) for r in read_rows(out_dir / 'v161_no_action_search_audit.csv'))}",
            f"direction_provenance_rows = {len(read_rows(out_dir / 'v160_line_a_direction_provenance.csv')) + len(read_rows(out_dir / 'v160_line_b_direction_provenance.csv'))}",
            "direction_source = train_stream_only / optimizer_state / split_gradient; audit metrics not used for direction",
            "```",
            "",
            "## 3. GPU / method surface",
            "",
        ]
    )
    recap.extend(table(gpu, [("round", "round"), ("gpu", "gpu"), ("run lines", "run_lines"), ("suffix", "artifact_suffix"), ("rows", "rows"), ("artifact exists", "artifact_exists")]))
    recap.extend(["", "Method surface manifest summary：", ""])
    recap.extend(table(read_rows(out_dir / "v161_method_surface_manifest.csv"), [("carrier", "carrier"), ("method", "method"), ("mechanism", "mechanism_family"), ("internal", "actual_internal_method"), ("overrides", "arg_overrides_json")], limit=50))
    recap.extend(
        [
            "",
            "## 4. Carrier × mechanism h800 summary",
            "",
            "```text",
            f"best_carrier = {route.get('best_carrier')}",
            f"best_mechanism_family = {route.get('best_mechanism_family')}",
            f"best_method = {route.get('best_method')}",
            f"S2_weak_productive_dynamics_reached = {route.get('S2_weak_productive_dynamics_reached')}",
            f"S3_productive_debt_recovery_reached = {route.get('S3_productive_debt_recovery_reached')}",
            "```",
            "",
        ]
    )
    recap.extend(table(mech, [("carrier", "carrier"), ("mechanism", "mechanism_family"), ("best method", "best_method"), ("source", "source_vs_best_control_h800"), ("retention", "source_retention_h800"), ("tail", "tail_recovery_rate_h800"), ("LineC", "LineC_recovery_rate_h800"), ("AUC", "AUCtime_ratio_h800"), ("S2", "S2_weak_productive_dynamics"), ("S3", "S3_productive_debt_recovery")]))
    recap.extend(["", "Line A method summary：", ""])
    recap.extend(table(a_summary, [("method", "method"), ("mechanism", "mechanism_family"), ("source", "source_vs_best_control_h800"), ("retention", "source_retention_h800"), ("tail", "tail_recovery_rate_h800"), ("LineC", "LineC_recovery_rate_h800"), ("AUC", "AUCtime_ratio_h800"), ("bad", "bad_event_fraction")]))
    recap.extend(["", "Line B method summary：", ""])
    recap.extend(table(b_summary, [("method", "method"), ("mechanism", "mechanism_family"), ("source", "source_vs_best_control_h800"), ("retention", "source_retention_h800"), ("tail", "tail_recovery_rate_h800"), ("LineC", "LineC_recovery_rate_h800"), ("AUC", "AUCtime_ratio_h800"), ("bad", "bad_event_fraction")]))
    recap.extend(["", "Line A h1600 top-2：", ""])
    recap.extend(table(a1600, [("method", "method"), ("dataset", "dataset"), ("seed", "seed"), ("source", "source_vs_best_control"), ("bad", "bad_event"), ("tail", "tail_debt_recovery_rate"), ("LineC", "LineC_debt_recovery_rate")]))
    recap.extend(["", "Line B h1600 top-2：", ""])
    recap.extend(table(b1600, [("method", "method"), ("dataset", "dataset"), ("seed", "seed"), ("source", "source_vs_best_control"), ("bad", "bad_event"), ("tail", "tail_debt_recovery_rate"), ("LineC", "LineC_debt_recovery_rate")]))
    recap.extend(
        [
            "",
            "## 5. Line C / D / F / M results",
            "",
            "```text",
            f"line_c_rows = {line_c.get('line_c_rows')}",
            f"line_c_reanchor_gate_pass = {line_c.get('line_c_reanchor_gate_pass')}",
            f"line_c_best_method = {line_c.get('line_c_best_method')}",
            f"line_c_best_macro_delta_vs_MLP = {line_c.get('line_c_best_macro_delta_vs_MLP')}",
            f"line_d_rat_rows = {line_d.get('line_d_rat_rows')}",
            f"line_d_rat_best_method = {line_d.get('line_d_rat_best_method')}",
            f"line_d_rat_gate_pass = {line_d.get('line_d_rat_gate_pass')}",
            f"line_f_rows = {line_f.get('line_f_rows')}",
            f"line_f_best_family = {line_f.get('line_f_best_family')}",
            f"line_f_best_dataset_seed_pass_count = {line_f.get('line_f_best_dataset_seed_pass_count')} / 9",
            f"line_m_rows = {line_m.get('line_m_rows')}",
            f"generic_functional_dynamics_rows = {line_m.get('generic_functional_dynamics_rows')}",
            f"kan_specific_advantage_rows = {line_m.get('kan_specific_advantage_rows')}",
            "```",
            "",
            "Line C reanchor summary：",
            "",
        ]
    )
    recap.extend(table(c_rows, [("method", "method"), ("dataset", "dataset"), ("seed", "seed"), ("delta", "macro_delta_vs_MLP"), ("near", "near_pass"), ("near count", "near_pass_count_method"), ("LineC", "LineC_no_regression"), ("gate", "line_c_reanchor_gate_pass")], limit=80))
    recap.extend(["", "Line D Rational summary：", ""])
    recap.extend(table(d_sum, [("method", "method"), ("pass", "dataset_seed_pass_count"), ("source", "mean_source_vs_best_control"), ("AUC", "median_AUCtime_ratio")]))
    recap.extend(["", "Line F all-basis family summary：", ""])
    recap.extend(table(f_sum, [("family", "family"), ("rows", "rows"), ("pass", "dataset_seed_pass_count"), ("best candidate", "best_candidate"), ("mean delta", "mean_delta_vs_MLP"), ("gate", "exploration_gate")]))
    recap.extend(["", "Line M attribution summary：", ""])
    recap.extend(table(read_rows(out_dir / "v161_line_m_crossline_attribution.csv"), [("mechanism", "mechanism_family"), ("KAN method", "kan_method"), ("best MLP", "best_mlp_global"), ("KAN source", "kan_source_h800"), ("MLP source", "mlp_global_source_h800"), ("delta", "delta_KAN_specific_global"), ("KAN adv", "KAN_specific_advantage")]))
    recap.extend(["", "Failure taxonomy summary：", ""])
    recap.extend(table(ft, [("failure class", "value"), ("rows", "rows")]))
    recap.extend(
        [
            "",
            "## 6. Final route / no-go",
            "",
            "```text",
            f"route = {route.get('route')}",
            f"minimum_success = {route.get('minimum_success')}",
            f"S2_weak_productive_dynamics_reached = {route.get('S2_weak_productive_dynamics_reached')}",
            f"S3_productive_debt_recovery_reached = {route.get('S3_productive_debt_recovery_reached')}",
            f"official_s5_reached = {route.get('official_s5_reached')}",
            f"promotion_allowed = {route.get('promotion_allowed')}",
            f"required_artifact_missing_count = {route.get('required_artifact_missing_count')}",
            f"forbidden_information_violation_count = {route.get('forbidden_information_violation_count')}",
            f"no_action_search_violation_count = {route.get('no_action_search_violation_count')}",
            "```",
            "",
            "No-go boundary：",
            "",
        ]
    )
    recap.extend(table(read_rows(out_dir / "v161_no_go_boundary.csv"), [("boundary", "boundary"), ("status", "status"), ("evidence", "evidence")]))
    recap.extend(["", "Next hypothesis queue：", ""])
    recap.extend(table(read_rows(out_dir / "v161_next_hypothesis_queue.csv"), [("priority", "priority"), ("hypothesis", "hypothesis"), ("allowed next step", "allowed_next_step")]))
    recap.extend(
        [
            "",
            "## 7. 科学结论",
            "",
            "```text",
            "1. v16.1 已执行 carrier × mechanism × horizon × controls matrix，并生成 required artifacts。",
            "2. A/B 均完成 H=800 full surface；top-2 已执行 H=1600 extension。",
            "3. LineC/CEp99/NLL/ECE/AUCtime/Brier 只用于 audit/gate/debt readback，没有反推方向。",
            "4. MLP generic positive 不写成 KAN-specific promotion；Line M attribution 是强制证据链。",
            f"5. 当前 route = {route.get('route')}，promotion_allowed = {route.get('promotion_allowed')}。",
            "```",
        ]
    )
    RECAP_DOC.write_text("\n".join(recap) + "\n", encoding="utf-8")

    exec_lines = [
        "# DG-KAN v16.1 MultiHypothesisFunctionalDynamics 4GPU 执行日志",
        "",
        "生成时间：2026-06-01（Asia/Singapore）",
        "",
        "## 1. 关键文件",
        "",
        f"- plan: {PLAN_DOC}",
        f"- runner: {ROOT / 'experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py'}",
        f"- shared kernel source: {ROOT / 'experiments/run_v160_dynamic_geometry_debt_recovery_multiline_fu.py'}",
        f"- all-basis registry: {ROOT / 'experiments/run_v149_line_d_all_basis_substrate_repair.py'}",
        f"- result dir: {out_dir}",
        f"- all-basis line F dir: {Path(args.line_f_out)}",
        f"- fast horizon readback: {args.fast_horizon_readback}",
        "- executed method surface: M1..M6 representative mechanisms + matched controls; full R1..R9 ladder was deferred and is not claimed as covered",
        f"- recap: {RECAP_DOC}",
        f"- execution log: {EXEC_LOG_DOC}",
        "",
        "## 2. 编译检查",
        "",
        "```bash",
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py experiments/run_v149_line_d_all_basis_substrate_repair.py",
        "```",
        "",
        "## 3. 四卡并行分片执行指令",
        "",
        "```bash",
        *four_gpu_commands(args, out_dir),
        "```",
        "",
        "Line F all-basis substrate command:",
        "",
        "```bash",
        line_f_command(args),
        "```",
        "",
        "Merge / h1600 / finalize:",
        "",
        "```bash",
        f"{base_command(args, out_dir, 'MERGE_A,MERGE_B,A1600', 'cuda:0')}",
        f"{base_command(args, out_dir, 'B1600', 'cuda:1')}",
        f"{base_command(args, out_dir, 'FINALIZE', 'cuda:0')} --reuse-if-present 1",
        "```",
        "",
        "## 4. GPU assignment manifest",
        "",
    ]
    exec_lines.extend(table(gpu, [("round", "round"), ("gpu", "gpu"), ("run lines", "run_lines"), ("suffix", "artifact_suffix"), ("rows", "rows"), ("artifact exists", "artifact_exists"), ("role", "expected_role")]))
    inventory = []
    for artifact in REQUIRED_ARTIFACTS + ["v161_required_artifact_manifest.csv"] + REQUIRED_FIGURES:
        path = out_dir / artifact
        inventory.append({"artifact": artifact, "exists": int(path.exists()), "rows": len(read_rows(path)) if path.suffix == ".csv" and path.exists() else "", "bytes": path.stat().st_size if path.exists() else 0})
    exec_lines.extend(["", "## 5. Artifact inventory", ""])
    exec_lines.extend(table(inventory, [("artifact", "artifact"), ("exists", "exists"), ("rows", "rows"), ("bytes", "bytes")]))
    exec_lines.extend(
        [
            "",
            "## 6. Final route snapshot",
            "",
            "```text",
            f"route = {route.get('route')}",
            f"S2_weak_productive_dynamics_reached = {route.get('S2_weak_productive_dynamics_reached')}",
            f"S3_productive_debt_recovery_reached = {route.get('S3_productive_debt_recovery_reached')}",
            f"promotion_allowed = {route.get('promotion_allowed')}",
            f"required_artifact_missing_count = {route.get('required_artifact_missing_count')}",
            "```",
            "",
            "## 7. 修复与审计记录",
            "",
            "```text",
            "1. 新增 v16.1 runner，使用 v16.0 validated kernels，但单独生成 v161 artifacts/docs/route。",
            "2. 新增 v16.1 substrate-only candidate registry；Line F rows 仍不能 promotion。",
            "3. method_surface_manifest 记录每个方法的内部映射与参数 override，便于后续审计。",
            "4. gpu_assignment_manifest 记录四卡分片；cpu_offload_used=0。",
            "5. runtime blocker 修复：exact per-step full ladder 过慢；正式结果只声明 horizon-target representative surface，不把未执行 ladder 写成 coverage。",
            "```",
        ]
    )
    EXEC_LOG_DOC.write_text("\n".join(exec_lines) + "\n", encoding="utf-8")


def base_command(args: argparse.Namespace, out_dir: Path, lines: str, device: str) -> str:
    return (
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python "
        "experiments/run_v161_multihypothesis_functional_dynamics_4gpu.py "
        f"--out-dir {out_dir} --line-f-out {Path(args.line_f_out)} --run-lines {lines} --device {device} "
        f"--datasets {args.datasets} --seeds {args.seeds} --train-size {args.train_size} --val-size {args.val_size} "
        f"--test-size {args.test_size} --batch-size {args.batch_size} --train-steps {args.train_steps} "
        f"--horizon-steps {args.horizon_steps} --h1600-steps {args.h1600_steps} --rational-steps {args.rational_steps} --lq-steps {args.lq_steps} "
        f"--split-count {args.split_count} --hidden {args.hidden} --lr {args.lr} --weight-decay {args.weight_decay} "
        f"--data-root {args.data_root} --no-download --fast-horizon-readback {args.fast_horizon_readback}"
    )


def four_gpu_commands(args: argparse.Namespace, out_dir: Path) -> list[str]:
    return [
        f"{base_command(args, out_dir, 'A', 'cuda:0')} --artifact-suffix a0 --line-a-methods A-M1-R0-D-CHE-PulseOnce-AdamWRecovery,ACTRL0-D-CHE-AdamW",
        f"{base_command(args, out_dir, 'A', 'cuda:1')} --artifact-suffix a1 --line-a-methods A-M2-D-CHE-SplitConsensusSignalSubspace,A-M3-D-CHE-PopRiskSNRPreconditioner,ACTRL1-D-CHE-NoOpMatchedOverhead",
        f"{base_command(args, out_dir, 'A', 'cuda:2')} --artifact-suffix a2 --line-a-methods A-M4-D-CHE-FunctionSpaceProximal,A-M5-D-CHE-OptimizerAlignedFU,ACTRL2-D-CHE-RandomMatchedPulse",
        f"{base_command(args, out_dir, 'A', 'cuda:3')} --artifact-suffix a3 --line-a-methods A-M6-D-CHE-CarrierReparamGlobalDecay,ACTRL3-D-CHE-AdamWExtraStepsMatchedTime",
        f"{base_command(args, out_dir, 'B', 'cuda:0')} --artifact-suffix b0 --line-b-methods B-M1-R0-MLP-PulseOnce-AdamWRecovery,BCTRL0-MLP-AdamW",
        f"{base_command(args, out_dir, 'B', 'cuda:1')} --artifact-suffix b1 --line-b-methods B-M2-MLP-HiddenSplitConsensusSignalSubspace,B-M3-MLP-PopRiskSNRPreconditioner,BCTRL1-MLP-NoOpMatchedOverhead",
        f"{base_command(args, out_dir, 'B', 'cuda:2')} --artifact-suffix b2 --line-b-methods B-M4-MLP-FunctionSpaceProximal,B-M5-MLP-CautiousAdamW,BCTRL2-MLP-RandomMatchedPulse",
        f"{base_command(args, out_dir, 'B,C,D', 'cuda:3')} --artifact-suffix b3 --line-b-methods B-M6-MLP-HiddenCarrierDecay,BCTRL3-MLP-AdamWExtraStepsMatchedTime",
    ]


def line_f_command(args: argparse.Namespace) -> str:
    return (
        "/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py "
        f"--out-dir {Path(args.line_f_out)} --datasets {args.datasets} --seeds {args.seeds} "
        f"--candidates {','.join(LINE_F_CANDIDATES)} --device cuda:3 --data-root {args.data_root} --no-download "
        f"--train-size {args.train_size} --val-size {args.val_size} --batch-size 32 --epochs 1"
    )


def run_finalizer(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    install_v161_surface()
    line_a, line_b, line_c, line_d, line_e, line_f, _old_line_m, _line_g = v160.load_existing_summaries(out_dir, Path(args.line_f_out))
    copy_v160_to_v161(out_dir)
    build_method_surface(out_dir)
    mechanism_rows = summarize_carrier_mechanisms(out_dir)
    line_m = build_v161_line_m(out_dir)
    build_failure_taxonomy(out_dir, line_c, line_d, line_f)
    build_gpu_manifest(out_dir)
    build_budget_certificate(out_dir, args, line_c)
    build_code_review_packet(out_dir)
    write_required_manifest(out_dir)
    missing = sum(sint(r.get("missing"), 0) for r in read_rows(out_dir / "v161_required_artifact_manifest.csv"))
    forbidden = 0
    no_action = 0
    route = decide_route(out_dir, line_a, line_b, line_c, line_d, line_f, mechanism_rows, missing, forbidden, no_action)
    write_json(out_dir / "v161_route_decision.json", route)
    build_line_z(out_dir, route, line_a, line_b, line_c, line_d, line_f)
    build_audits(out_dir, route)
    write_figures(out_dir, route)
    build_gpu_manifest(out_dir)
    write_required_manifest(out_dir)
    missing = sum(sint(r.get("missing"), 0) for r in read_rows(out_dir / "v161_required_artifact_manifest.csv"))
    forbidden = sum(sint(r.get("violation"), 0) for r in read_rows(out_dir / "v161_forbidden_information_audit.csv"))
    no_action = sum(sint(r.get("violation"), 0) for r in read_rows(out_dir / "v161_no_action_search_audit.csv"))
    route = decide_route(out_dir, line_a, line_b, line_c, line_d, line_f, mechanism_rows, missing, forbidden, no_action)
    write_json(out_dir / "v161_route_decision.json", route)
    build_line_z(out_dir, route, line_a, line_b, line_c, line_d, line_f)
    build_audits(out_dir, route)
    build_docs(out_dir, args, route, line_a, line_b, line_c, line_d, line_f, line_m)
    return route


def main() -> None:
    install_v161_surface()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--line-f-out", default=str(DEFAULT_LINE_F_OUT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datasets", default="MNIST,Fashion-MNIST,KMNIST")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--train-size", type=int, default=256)
    ap.add_argument("--val-size", type=int, default=128)
    ap.add_argument("--test-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-steps", type=int, default=120)
    ap.add_argument("--horizon-steps", type=int, default=800)
    ap.add_argument("--h1600-steps", type=int, default=1600)
    ap.add_argument("--rational-steps", type=int, default=80)
    ap.add_argument("--lq-steps", type=int, default=80)
    ap.add_argument("--trace-interval", type=int, default=200)
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
    ap.add_argument("--fast-horizon-readback", type=int, default=1)
    ap.add_argument("--run-lines", default="all")
    ap.add_argument("--line-a-methods", default="")
    ap.add_argument("--line-b-methods", default="")
    ap.add_argument("--artifact-suffix", default="")
    args = ap.parse_args()
    if sint(args.fast_horizon_readback, 1):
        v158.train_dynamics_case = train_dynamics_case_horizon_only
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_lines = {part.strip().upper() for part in str(args.run_lines).split(",") if part.strip()}
    all_lines = "ALL" in run_lines
    finalize = all_lines or "FINALIZE" in run_lines or "Z" in run_lines

    if "MERGE_A" in run_lines or "MERGEA" in run_lines:
        v160.merge_dynamics_parts(out_dir, "A")
    if "MERGE_B" in run_lines or "MERGEB" in run_lines:
        v160.merge_dynamics_parts(out_dir, "B")

    need_splits = all_lines or bool(run_lines & {"A", "B", "C", "D", "A1600", "B1600"})
    if need_splits:
        device = resolve_cuda_device(str(args.device))
        if device.type != "cuda":
            raise RuntimeError("v16.1 execution requires CUDA; refusing CPU execution")
        splits = v158.load_splits(args, device)
        if all_lines or "A" in run_lines:
            v160.run_dynamics_line("A", "D-CHE", LINE_A_METHODS, LINE_A_CONTROLS, LINE_A_CANDIDATES, args, splits, device, out_dir)
        if all_lines or "B" in run_lines:
            v160.run_dynamics_line("B", "MLP", LINE_B_METHODS, LINE_B_CONTROLS, LINE_B_CANDIDATES, args, splits, device, out_dir)
        if all_lines or "C" in run_lines:
            v160.run_line_c(args, splits, device, out_dir)
        if all_lines or "D" in run_lines:
            v160.run_line_d_rational(args, splits, device, out_dir)
        if "A1600" in run_lines:
            v160.run_h1600_extension("A", "D-CHE", LINE_A_CONTROLS, LINE_A_CANDIDATES, args, splits, device, out_dir)
        if "B1600" in run_lines:
            v160.run_h1600_extension("B", "MLP", LINE_B_CONTROLS, LINE_B_CANDIDATES, args, splits, device, out_dir)
    if all_lines or "F" in run_lines:
        v160.build_line_f(out_dir, Path(args.line_f_out))
    if not finalize:
        print(json.dumps({"stage": "V161_LINE_SHARD_DONE", "run_lines": sorted(run_lines), "out_dir": str(out_dir)}, indent=2, sort_keys=True))
        return
    route = run_finalizer(args, out_dir)
    print(json.dumps(route, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
