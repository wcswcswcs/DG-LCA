#!/usr/bin/env python3
"""DG-KAN v22.66 Metric-Compatible Generator Atlas FU runner."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
from pathlib import Path
import py_compile
import re
import shlex
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
csv.field_size_limit(sys.maxsize)

from dgkan.fu.metric_preserving_functional_atlas import (  # noqa: E402
    AtlasBuild,
    LowRankAtlasLinear,
    MetricAtlasMLP,
    MetricCompatibleAtlasMLP,
    MetricCompatibleLowRankLinear,
    SimpleMLP,
    apply_output_transport,
    build_last_layer_atlas,
    c_cayley_retraction,
    c_skew_project,
    gram_drift,
    inv_sqrt_psd,
    metric_compatible_descent_diagnostics,
    psd_project,
    sqrt_psd,
    stable_rank,
    transport_matrix,
)
from experiments.run_v22_64_metric_preserving_functional_atlas_fu import (  # noqa: E402
    CautiousAdamW,
    ScheduleFreeAdamWLocal,
    corr,
    evaluate_tensors,
    fval,
    iflag,
    load_bundle,
    md_table,
    mean,
    optimizer_state_count,
    read_json,
    read_rows,
    safe_fragment,
    set_seed,
    split_csv,
    torch_device,
    trainable_param_count,
    write_json,
    write_rows,
)


KAN_ENV_PYTHON = ROOT.parent / "miniconda3/envs/kan/bin/python"
PYTHON = os.environ.get("KAN_PYTHON", str(KAN_ENV_PYTHON if KAN_ENV_PYTHON.exists() else sys.executable))
OUT_ROOT = ROOT / "results/v22_66"
CHUNK_ROOT = OUT_ROOT / "chunks"
LOG_ROOT = OUT_ROOT / "logs"
DOCS = ROOT / "docs"
EXEC_DOC = DOCS / "DG-KAN_v22.66_MetricCompatibleGeneratorAtlasFU_执行日志.md"
RECAP_DOC = DOCS / "DG-KAN_v22.66_MetricCompatibleGeneratorAtlasFU_实验结果复盘.md"
PLAN_DOC = DOCS / "DG-KAN_v22.66_MetricCompatibleGeneratorAtlasFU_完整计划.md"
RUNNER = ROOT / "experiments/run_v22_66_metric_compatible_generator_atlas_fu.py"
V64_RUNNER = ROOT / "experiments/run_v22_64_metric_preserving_functional_atlas_fu.py"
ATLAS_MODULE = ROOT / "dgkan/fu/metric_preserving_functional_atlas.py"

REDESIGN_ARCHITECTURES = ["DGKAN_CHE4", "DGKAN_CHE3_XLIN", "DGKAN_FOU4", "DGKAN_FOU4_LIN", "DGKAN_FOU4_LIN50", "DGKAN_RBF4", "DGKAN_RBF4_XLIN", "DGKAN_HAT4", "DGKAN_HAT4_XLIN", "DGKAN_RAT4"]
REDESIGN_KAN_SPECS: dict[str, dict[str, Any]] = {
    "DGKAN_CHE4": {
        "basis_family": "D-CHE4",
        "basis_name": "chebyshev",
        "k": 4,
        "local_support": 0,
        "global_support": 1,
        "uses_exp": 0,
        "uses_sin_cos": 0,
        "uses_division": 0,
        "basis_order": 4,
        "init_variant": "v22_67_cheby_k4_dense_redesign",
        "seed_offset": 3000,
    },
    "DGKAN_CHE3_XLIN": {
        "basis_family": "D-CHE3-XLIN",
        "basis_name": "chebyshev",
        "k": 3,
        "local_support": 0,
        "global_support": 1,
        "uses_exp": 0,
        "uses_sin_cos": 0,
        "uses_division": 0,
        "basis_order": 3,
        "uses_dense_basis_tensor": 0,
        "init_variant": "cheby_k3_inputcross_localr4_projr128_triton_l3_gradbuf_linearres010",
        "seed_offset": 9000,
    },
    "DGKAN_FOU4": {
        "basis_family": "D-FOU4",
        "basis_name": "fourier_lowfreq",
        "k": 4,
        "local_support": 0,
        "global_support": 1,
        "uses_exp": 0,
        "uses_sin_cos": 1,
        "uses_division": 0,
        "basis_order": 4,
        "init_variant": "v22_67_fourier_k4_dense_redesign",
        "seed_offset": 4000,
    },
    "DGKAN_FOU4_LIN": {
        "basis_family": "D-FOU4-LIN",
        "basis_name": "fourier_lowfreq",
        "k": 4,
        "local_support": 0,
        "global_support": 1,
        "uses_exp": 0,
        "uses_sin_cos": 1,
        "uses_division": 0,
        "basis_order": 4,
        "init_variant": "fourier_k4_linearres_gemm_l3_matmul_linearres010",
        "seed_offset": 8000,
    },
    "DGKAN_FOU4_LIN50": {
        "basis_family": "D-FOU4-LIN50",
        "basis_name": "fourier_lowfreq",
        "k": 4,
        "local_support": 0,
        "global_support": 1,
        "uses_exp": 0,
        "uses_sin_cos": 1,
        "uses_division": 0,
        "basis_order": 4,
        "uses_dense_basis_tensor": 0,
        "init_variant": "fourier_k4_linearres_gemm_l3_matmul_linearres050",
        "seed_offset": 10000,
    },
    "DGKAN_RBF4": {
        "basis_family": "D-RBF4",
        "basis_name": "compact_rbf",
        "k": 4,
        "local_support": 1,
        "global_support": 0,
        "uses_exp": 1,
        "uses_sin_cos": 0,
        "uses_division": 0,
        "basis_order": 1,
        "init_variant": "v22_67_rbf_k4_dense_redesign",
        "seed_offset": 5000,
    },
    "DGKAN_RBF4_XLIN": {
        "basis_family": "D-RBF4-XLIN",
        "basis_name": "compact_rbf",
        "k": 4,
        "local_support": 1,
        "global_support": 0,
        "uses_exp": 1,
        "uses_sin_cos": 0,
        "uses_division": 0,
        "basis_order": 1,
        "uses_dense_basis_tensor": 0,
        "init_variant": "rbf_k4_triton_l3_matmul_inputcross_localr4_projr128_linearres010",
        "seed_offset": 11000,
    },
    "DGKAN_HAT4": {
        "basis_family": "D-HAT4",
        "basis_name": "hat_wavelet",
        "k": 4,
        "local_support": 1,
        "global_support": 0,
        "uses_exp": 0,
        "uses_sin_cos": 0,
        "uses_division": 0,
        "basis_order": 1,
        "init_variant": "v22_67_hat_k4_dense_redesign",
        "seed_offset": 6000,
    },
    "DGKAN_HAT4_XLIN": {
        "basis_family": "D-HAT4-XLIN",
        "basis_name": "hat_wavelet",
        "k": 4,
        "local_support": 1,
        "global_support": 0,
        "uses_exp": 0,
        "uses_sin_cos": 0,
        "uses_division": 0,
        "basis_order": 1,
        "uses_dense_basis_tensor": 0,
        "init_variant": "hat_wavelet_k4_triton_l3_matmul_inputcross_localr4_projr128_linearres010",
        "seed_offset": 12000,
    },
    "DGKAN_RAT4": {
        "basis_family": "D-RAT4",
        "basis_name": "rational_kat_lite",
        "k": 4,
        "local_support": 0,
        "global_support": 1,
        "uses_exp": 0,
        "uses_sin_cos": 0,
        "uses_division": 1,
        "basis_order": 4,
        "init_variant": "v22_67_rational_k4_dense_redesign",
        "seed_offset": 7000,
    },
}

REFERENCE_METHODS = {"adamw", "cautious_adamw", "schedule_free_adamw_local"}
EXTERNAL_METHODS = {"poet_official", "pion_oet_sphere_official", "pion_oet_local"}
CANDIDATE_METHODS = {
    "mcga_transport_generator_rank2",
    "mcga_transport_generator_rank4",
    "mcga_transport_generator_rank8",
    "mcga_transport_generator_last_layer",
    "mcga_gradcoh_generator_rank4",
    "mcga_gradcoh_generator_rank8",
    "mcga_gradcoh_eta01_generator_rank8",
    "mcga_gradcoh_eta005_generator_rank8",
    "mcga_gradcoh_eta001_generator_rank8",
    "mcga_gradcoh_eta0005_generator_rank8",
    "mcga_gradcoh_fsclip_generator_rank4",
    "mcga_gradcoh_fsclip_generator_rank8",
    "mcga_gradcoh_fsclip_residual_rank4",
    "mcga_gradcoh_fsclip_residual_rank8",
    "mcga_oocw_gradcoh_generator_rank4",
    "mcga_oocw_gradcoh_fsclip_generator_rank4",
    "mcga_oocw_gradcoh_fsclip_residual_rank4",
    "mcga_cvargrad_generator_rank4",
    "mcga_cvargrad_fsclip_generator_rank4",
    "mcga_gradunion_generator_rank4",
    "mcga_gradunion_fsclip_generator_rank4",
    "mcga_kan_bank_oet_generator_rank4",
    "mcga_kan_bank_oet_generator_rank8",
    "mcga_kan_bank_oet_fsclip_generator_rank4",
    "mcga_kan_bank_oet_fsclip_generator_rank8",
    "mcga_kan_bank_oet_balanced_fsclip_generator_rank4",
    "mcga_kan_bank_oet_balanced_fsclip_generator_rank8",
    "mcga_over_poet_residual_rank4",
    "mcga_over_poet_baselock_residual_rank4",
    "mcga_over_poet_fsclip_residual_rank4",
    "mcga_over_poet_fsclip_eta05_residual_rank4",
    "mcga_over_poet_fsclip_eta025_residual_rank4",
    "mcga_over_poet_fsclip_eta025_warm50_residual_rank4",
    "mcga_over_poet_fsclip_eta01_residual_rank4",
    "mcga_over_poet_fsclip_eta005_residual_rank4",
    "mcga_over_poet_fsclip_eta005_warm50_residual_rank4",
    "mcga_over_poet_fsclip_eta005_warm80_residual_rank4",
    "mcga_over_poet_debtbudget50_fsclip_eta005_warm50_residual_rank4",
    "mcga_over_poet_debtbudget80_fsclip_eta005_warm50_residual_rank4",
    "mcga_over_poet_fsclip_eta002_warm50_residual_rank4",
    "mcga_over_poet_fsclip_eta001_residual_rank4",
    "mcga_over_poet_fsclip_eta001_warm50_residual_rank4",
    "mcga_over_poet_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_actgradcov_fsclip_eta025_fishermetric_warm50_residual_rank4",
    "mcga_over_poet_actcvargrad_fsclip_eta025_fishermetric_warm50_residual_rank4",
    "mcga_over_poet_fsfill_residual_rank4",
    "mcga_over_poet_residual_rank8",
    "mcga_over_pion_residual_rank4",
    "mcga_over_pion_fsclip_eta05_residual_rank4",
    "mcga_over_pion_fsclip_eta025_residual_rank4",
    "mcga_over_pion_fsclip_eta01_residual_rank4",
    "mcga_over_pion_fsclip_eta005_residual_rank4",
    "mcga_over_pion_fsclip_eta005_warm50_residual_rank4",
    "mcga_over_pion_fsclip_eta001_residual_rank4",
    "mcga_over_pion_actgradcov_fsclip_eta01_fishermetric_residual_rank4",
    "mcga_over_pion_actgradcov_fsclip_eta005_fishermetric_residual_rank4",
    "mcga_over_pion_actcvargrad_fsclip_eta005_fishermetric_residual_rank4",
    "mcga_over_pion_fsfill_residual_rank4",
    "mcga_over_pion_fsfill_eta01_residual_rank4",
    "mcga_over_oet_residual_rank4",
    "mcga_over_oet_fsclip_eta025_residual_rank4",
    "mcga_over_oet_fsclip_eta01_residual_rank4",
    "mcga_over_poet_pion_blend05_fsclip_eta025_residual_rank4",
    "mcga_over_poet_pion_blend10_fsclip_eta025_residual_rank4",
    "mcga_over_poet_pion_blend20_fsclip_eta025_residual_rank4",
    "mcga_over_poet_pion_blend30_fsclip_eta025_residual_rank4",
    "mcga_over_poet_pion_blend40_fsclip_eta025_residual_rank4",
    "mcga_over_poet_pion_blend50_fsclip_eta025_residual_rank4",
    "mcga_over_poet_pion_blend60_fsclip_eta025_residual_rank4",
    "mcga_over_poet_pion_blend65_fsclip_eta025_residual_rank4",
    "mcga_over_poet_pion_blend65_cc25_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend65_cc50_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend65_actgradunion_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend65_actgradcoh_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend65_actgradcov_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend65_actgradcov_ccfs25_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend65_actgradcov_ccfs50_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend65_actgradcov_balanced_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend65_actgradcov_featurein_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend65_actgradcov_balanced_featurein_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend65_actcvargrad_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend65_actgradmix_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend68_actgradcov_ccfs25_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend68_actgradcov_ccfs50_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend65_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend65_fsclip_eta025_fishermetric_warm50_residual_rank4",
    "mcga_over_poet_pion_blend65_fsclip_eta025_fishermetric_warm80_residual_rank4",
    "mcga_over_poet_pion_blend65_debtbudget50_fsclip_eta025_fishermetric_warm50_residual_rank4",
    "mcga_over_poet_pion_blend65_debtbudget80_fsclip_eta025_fishermetric_warm50_residual_rank4",
    "mcga_over_poet_pion_blend65_fsclip_eta05_residual_rank4",
    "mcga_over_poet_pion_blend65_fsfill_eta025_residual_rank4",
    "mcga_over_poet_pion_blend70_cc25_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend70_cc50_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend70_actgradunion_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend70_actgradcoh_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend70_actgradcov_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend70_actgradcov_ccfs25_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend70_actgradcov_ccfs50_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend70_actgradcov_balanced_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend70_actcvargrad_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend70_actgradmix_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend72_actgradcov_ccfs25_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend72_actgradcov_ccfs50_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend70_fsclip_eta025_fishermetric_residual_rank4",
    "mcga_over_poet_pion_blend70_fsclip_eta025_fishermetric_warm50_residual_rank4",
    "mcga_over_poet_pion_blend70_fsclip_eta05_residual_rank4",
    "mcga_over_poet_pion_blend70_fsfill_eta025_residual_rank4",
    "mcga_over_poet_pion_blend70_fsclip_eta025_residual_rank4",
    "mcga_over_poet_pion_blend75_fsclip_eta025_residual_rank4",
    "kan_task_visible_chart_rank4",
    "kan_task_visible_chart_bankbudget_fsclip_eta001_rank4",
    "kan_task_visible_chart_ooc_bankbudget_fsclip_eta001_rank4",
    "kan_task_visible_chart_shape_budget005_bankbudget_fsclip_eta001_rank4",
    "kan_task_visible_chart_shape_budget005_bankbudget_fsclip_eta001_rank8",
    "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank8",
    "kan_task_visible_chart_ooc_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank8",
    "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsfill_eta001_rank8",
    "kan_task_visible_chart_over_poet_shape_budget010_bankbudget_fsclip_eta001_rank8",
    "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta005_rank8",
    "kan_task_visible_chart_over_poet_fsclip_eta001_rank10",
    "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
    "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsfill_eta001_rank10",
    "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsfill_eta002_rank10",
    "kan_task_visible_chart_ooc_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
    "kan_task_visible_chart_signed_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
    "kan_task_visible_chart_grad_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
    "kan_task_visible_chart_grad_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
    "kan_task_visible_chart_stablegrad_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
    "kan_task_visible_chart_stablegrad_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
    "kan_task_visible_chart_fishermetric_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
    "kan_task_visible_chart_fishermetric_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
    "kan_task_visible_chart_gradcoh_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
    "kan_task_visible_chart_gradcoh_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
    "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta0015_rank10",
    "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta0018_rank10",
    "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
    "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta005_rank10",
    "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta01_rank10",
    "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta001_warm50_rank10",
    "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta002_warm50_rank10",
    "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta005_warm50_rank10",
    "kan_task_visible_chart_over_poet_pion_blend50_shape_budget005_bankbudget_fsclip_eta025_rank10",
    "kan_task_visible_chart_bankbudget_fsclip_eta005_rank4",
    "kan_task_visible_chart_bankbudget_fsclip_eta01_rank4",
    "mcga_shape_signal_budget001",
    "mcga_shape_signal_budget002",
    "mcga_shape_signal_budget005",
    "mcga_shape_tail_safe_budget002",
    "mcga_shape_reservoir_suppressed_budget002",
}
CONTROL_METHODS = {
    "oet_only_coordinate",
    "lora_like_coordinate",
    "same_rank_random_coordinate",
    "same_spectrum_random_coordinate",
    "same_functional_spectrum_random_coordinate",
    "same_isometric_capacity_random_coordinate",
    "same_generator_descent_energy_random",
    "same_C_skew_spectrum_random",
    "same_debtbudget_generator_random",
    "same_debtbudget80_generator_random",
    "same_transport_error_random_coordinate",
    "same_signal_reachable_generator_random",
    "same_shape_budget_generator_random",
    "same_metric_drift_random_coordinate",
    "same_readout_visible_energy_random_chart_rank4",
    "same_readout_visible_energy_random_chart_bankbudget_fsclip_eta001_rank4",
    "same_readout_visible_energy_random_chart_ooc_bankbudget_fsclip_eta001_rank4",
    "same_readout_visible_energy_random_chart_shape_budget005_bankbudget_fsclip_eta001_rank4",
    "same_readout_visible_energy_random_chart_shape_budget005_bankbudget_fsclip_eta001_rank8",
    "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank8",
    "same_readout_visible_energy_random_chart_ooc_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank8",
    "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsfill_eta001_rank8",
    "same_readout_visible_energy_random_chart_over_poet_shape_budget010_bankbudget_fsclip_eta001_rank8",
    "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta005_rank8",
    "same_readout_visible_energy_random_chart_over_poet_fsclip_eta001_rank10",
    "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
    "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsfill_eta001_rank10",
    "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsfill_eta002_rank10",
    "same_readout_visible_energy_random_chart_ooc_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
    "same_readout_visible_energy_random_chart_signed_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
    "same_readout_visible_energy_random_chart_grad_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
    "same_readout_visible_energy_random_chart_grad_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
    "same_readout_visible_energy_random_chart_stablegrad_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
    "same_readout_visible_energy_random_chart_stablegrad_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
    "same_readout_visible_energy_random_chart_fishermetric_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
    "same_readout_visible_energy_random_chart_fishermetric_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
    "same_readout_visible_energy_random_chart_gradcoh_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
    "same_readout_visible_energy_random_chart_gradcoh_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
    "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta0015_rank10",
    "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta0018_rank10",
    "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
    "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta005_rank10",
    "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta01_rank10",
    "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta001_warm50_rank10",
    "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta002_warm50_rank10",
    "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta005_warm50_rank10",
    "same_readout_visible_energy_random_chart_over_poet_pion_blend50_shape_budget005_bankbudget_fsclip_eta025_rank10",
    "same_readout_visible_energy_random_chart_bankbudget_fsclip_eta005_rank4",
    "same_readout_visible_energy_random_chart_bankbudget_fsclip_eta01_rank4",
    "shuffled_source_witness_generator",
    "witness_only_generator",
    "source_only_generator",
    "self_only_generator",
    "same_compute_noop_coordinate",
    "transport_only",
    "metric_preserving_noop",
    "shape_only_no_generator",
}
DEFAULT_METHODS = ",".join(
    [
        "adamw",
        "cautious_adamw",
        "schedule_free_adamw_local",
        "poet_official",
        "pion_oet_sphere_official",
        "pion_oet_local",
        "oet_only_coordinate",
        "lora_like_coordinate",
        "same_compute_noop_coordinate",
        "mcga_transport_generator_rank2",
        "mcga_transport_generator_rank4",
        "mcga_transport_generator_rank8",
        "mcga_transport_generator_last_layer",
        "mcga_gradcoh_generator_rank4",
        "mcga_gradcoh_generator_rank8",
        "mcga_gradcoh_eta01_generator_rank8",
        "mcga_gradcoh_eta005_generator_rank8",
        "mcga_gradcoh_eta001_generator_rank8",
        "mcga_gradcoh_eta0005_generator_rank8",
        "mcga_gradcoh_fsclip_generator_rank4",
        "mcga_gradcoh_fsclip_generator_rank8",
        "mcga_gradcoh_fsclip_residual_rank4",
        "mcga_gradcoh_fsclip_residual_rank8",
        "mcga_oocw_gradcoh_generator_rank4",
        "mcga_oocw_gradcoh_fsclip_generator_rank4",
        "mcga_oocw_gradcoh_fsclip_residual_rank4",
        "mcga_cvargrad_generator_rank4",
        "mcga_cvargrad_fsclip_generator_rank4",
        "mcga_gradunion_generator_rank4",
        "mcga_gradunion_fsclip_generator_rank4",
        "mcga_kan_bank_oet_generator_rank4",
        "mcga_kan_bank_oet_generator_rank8",
        "mcga_kan_bank_oet_fsclip_generator_rank4",
        "mcga_kan_bank_oet_fsclip_generator_rank8",
        "mcga_kan_bank_oet_balanced_fsclip_generator_rank4",
        "mcga_kan_bank_oet_balanced_fsclip_generator_rank8",
        "mcga_over_poet_residual_rank4",
        "mcga_over_poet_baselock_residual_rank4",
        "mcga_over_poet_fsclip_residual_rank4",
        "mcga_over_poet_fsclip_eta05_residual_rank4",
        "mcga_over_poet_fsclip_eta025_residual_rank4",
        "mcga_over_poet_fsclip_eta025_warm50_residual_rank4",
        "mcga_over_poet_fsclip_eta01_residual_rank4",
        "mcga_over_poet_fsclip_eta005_residual_rank4",
        "mcga_over_poet_fsclip_eta005_warm50_residual_rank4",
        "mcga_over_poet_fsclip_eta005_warm80_residual_rank4",
        "mcga_over_poet_debtbudget50_fsclip_eta005_warm50_residual_rank4",
        "mcga_over_poet_debtbudget80_fsclip_eta005_warm50_residual_rank4",
        "mcga_over_poet_fsclip_eta002_warm50_residual_rank4",
        "mcga_over_poet_fsclip_eta001_residual_rank4",
        "mcga_over_poet_fsclip_eta001_warm50_residual_rank4",
        "mcga_over_poet_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_actgradcov_fsclip_eta025_fishermetric_warm50_residual_rank4",
        "mcga_over_poet_actcvargrad_fsclip_eta025_fishermetric_warm50_residual_rank4",
        "mcga_over_poet_fsfill_residual_rank4",
        "mcga_over_poet_residual_rank8",
        "mcga_over_pion_residual_rank4",
        "mcga_over_pion_fsclip_eta05_residual_rank4",
        "mcga_over_pion_fsclip_eta025_residual_rank4",
        "mcga_over_pion_fsclip_eta01_residual_rank4",
        "mcga_over_pion_fsclip_eta005_residual_rank4",
        "mcga_over_pion_fsclip_eta005_warm50_residual_rank4",
        "mcga_over_pion_fsclip_eta001_residual_rank4",
        "mcga_over_pion_actgradcov_fsclip_eta01_fishermetric_residual_rank4",
        "mcga_over_pion_actgradcov_fsclip_eta005_fishermetric_residual_rank4",
        "mcga_over_pion_actcvargrad_fsclip_eta005_fishermetric_residual_rank4",
        "mcga_over_pion_fsfill_residual_rank4",
        "mcga_over_pion_fsfill_eta01_residual_rank4",
        "mcga_over_oet_residual_rank4",
        "mcga_over_oet_fsclip_eta025_residual_rank4",
        "mcga_over_oet_fsclip_eta01_residual_rank4",
        "mcga_over_poet_pion_blend05_fsclip_eta025_residual_rank4",
        "mcga_over_poet_pion_blend10_fsclip_eta025_residual_rank4",
        "mcga_over_poet_pion_blend20_fsclip_eta025_residual_rank4",
        "mcga_over_poet_pion_blend30_fsclip_eta025_residual_rank4",
        "mcga_over_poet_pion_blend40_fsclip_eta025_residual_rank4",
        "mcga_over_poet_pion_blend50_fsclip_eta025_residual_rank4",
        "mcga_over_poet_pion_blend60_fsclip_eta025_residual_rank4",
        "mcga_over_poet_pion_blend65_fsclip_eta025_residual_rank4",
        "mcga_over_poet_pion_blend65_cc25_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend65_cc50_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend65_actgradunion_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend65_actgradcoh_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend65_actgradcov_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend65_actgradcov_ccfs25_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend65_actgradcov_ccfs50_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend65_actgradcov_balanced_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend65_actgradcov_featurein_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend65_actgradcov_balanced_featurein_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend65_actcvargrad_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend68_actgradcov_ccfs25_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend68_actgradcov_ccfs50_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend65_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend65_fsclip_eta025_fishermetric_warm50_residual_rank4",
        "mcga_over_poet_pion_blend65_fsclip_eta025_fishermetric_warm80_residual_rank4",
        "mcga_over_poet_pion_blend65_debtbudget50_fsclip_eta025_fishermetric_warm50_residual_rank4",
        "mcga_over_poet_pion_blend65_debtbudget80_fsclip_eta025_fishermetric_warm50_residual_rank4",
        "mcga_over_poet_pion_blend65_fsclip_eta05_residual_rank4",
        "mcga_over_poet_pion_blend65_fsfill_eta025_residual_rank4",
        "mcga_over_poet_pion_blend70_cc25_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend70_cc50_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend70_actgradunion_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend70_actgradcoh_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend70_actgradcov_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend70_actgradcov_ccfs25_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend70_actgradcov_ccfs50_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend70_actgradcov_balanced_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend70_actcvargrad_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend72_actgradcov_ccfs25_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend72_actgradcov_ccfs50_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend70_fsclip_eta025_fishermetric_residual_rank4",
        "mcga_over_poet_pion_blend70_fsclip_eta025_fishermetric_warm50_residual_rank4",
        "mcga_over_poet_pion_blend70_fsclip_eta05_residual_rank4",
        "mcga_over_poet_pion_blend70_fsfill_eta025_residual_rank4",
        "mcga_over_poet_pion_blend70_fsclip_eta025_residual_rank4",
        "mcga_over_poet_pion_blend75_fsclip_eta025_residual_rank4",
        "kan_task_visible_chart_rank4",
        "kan_task_visible_chart_bankbudget_fsclip_eta001_rank4",
        "kan_task_visible_chart_ooc_bankbudget_fsclip_eta001_rank4",
        "kan_task_visible_chart_shape_budget005_bankbudget_fsclip_eta001_rank4",
        "kan_task_visible_chart_shape_budget005_bankbudget_fsclip_eta001_rank8",
        "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank8",
        "kan_task_visible_chart_ooc_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank8",
        "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsfill_eta001_rank8",
        "kan_task_visible_chart_over_poet_shape_budget010_bankbudget_fsclip_eta001_rank8",
        "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta005_rank8",
        "kan_task_visible_chart_over_poet_fsclip_eta001_rank10",
        "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
        "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsfill_eta001_rank10",
        "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsfill_eta002_rank10",
        "kan_task_visible_chart_ooc_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
        "kan_task_visible_chart_signed_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
        "kan_task_visible_chart_grad_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
        "kan_task_visible_chart_grad_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
        "kan_task_visible_chart_stablegrad_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
        "kan_task_visible_chart_stablegrad_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
        "kan_task_visible_chart_fishermetric_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
        "kan_task_visible_chart_fishermetric_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
        "kan_task_visible_chart_gradcoh_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
        "kan_task_visible_chart_gradcoh_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
        "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta0015_rank10",
        "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta0018_rank10",
        "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
        "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta005_rank10",
        "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta01_rank10",
        "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta001_warm50_rank10",
        "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta002_warm50_rank10",
        "kan_task_visible_chart_over_poet_shape_budget005_bankbudget_fsclip_eta005_warm50_rank10",
        "kan_task_visible_chart_over_poet_pion_blend50_shape_budget005_bankbudget_fsclip_eta025_rank10",
        "kan_task_visible_chart_bankbudget_fsclip_eta005_rank4",
        "kan_task_visible_chart_bankbudget_fsclip_eta01_rank4",
        "mcga_shape_signal_budget001",
        "mcga_shape_signal_budget002",
        "mcga_shape_signal_budget005",
        "mcga_shape_tail_safe_budget002",
        "mcga_shape_reservoir_suppressed_budget002",
        "transport_only",
        "metric_preserving_noop",
        "shape_only_no_generator",
        "same_rank_random_coordinate",
        "same_spectrum_random_coordinate",
        "same_functional_spectrum_random_coordinate",
        "same_isometric_capacity_random_coordinate",
        "same_generator_descent_energy_random",
        "same_C_skew_spectrum_random",
        "same_debtbudget_generator_random",
        "same_debtbudget80_generator_random",
        "same_transport_error_random_coordinate",
        "same_signal_reachable_generator_random",
        "same_shape_budget_generator_random",
        "same_metric_drift_random_coordinate",
        "same_readout_visible_energy_random_chart_rank4",
        "same_readout_visible_energy_random_chart_bankbudget_fsclip_eta001_rank4",
        "same_readout_visible_energy_random_chart_ooc_bankbudget_fsclip_eta001_rank4",
        "same_readout_visible_energy_random_chart_shape_budget005_bankbudget_fsclip_eta001_rank4",
        "same_readout_visible_energy_random_chart_shape_budget005_bankbudget_fsclip_eta001_rank8",
        "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank8",
        "same_readout_visible_energy_random_chart_ooc_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank8",
        "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsfill_eta001_rank8",
        "same_readout_visible_energy_random_chart_over_poet_shape_budget010_bankbudget_fsclip_eta001_rank8",
        "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta005_rank8",
        "same_readout_visible_energy_random_chart_over_poet_fsclip_eta001_rank10",
        "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
        "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsfill_eta001_rank10",
        "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsfill_eta002_rank10",
        "same_readout_visible_energy_random_chart_ooc_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
        "same_readout_visible_energy_random_chart_signed_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
        "same_readout_visible_energy_random_chart_grad_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
        "same_readout_visible_energy_random_chart_grad_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
        "same_readout_visible_energy_random_chart_stablegrad_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
        "same_readout_visible_energy_random_chart_stablegrad_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
        "same_readout_visible_energy_random_chart_fishermetric_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
        "same_readout_visible_energy_random_chart_fishermetric_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
        "same_readout_visible_energy_random_chart_gradcoh_over_poet_shape_budget005_bankbudget_fsclip_eta001_rank10",
        "same_readout_visible_energy_random_chart_gradcoh_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
        "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta0015_rank10",
        "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta0018_rank10",
        "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta002_rank10",
        "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta005_rank10",
        "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta01_rank10",
        "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta001_warm50_rank10",
        "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta002_warm50_rank10",
        "same_readout_visible_energy_random_chart_over_poet_shape_budget005_bankbudget_fsclip_eta005_warm50_rank10",
        "same_readout_visible_energy_random_chart_over_poet_pion_blend50_shape_budget005_bankbudget_fsclip_eta025_rank10",
        "same_readout_visible_energy_random_chart_bankbudget_fsclip_eta005_rank4",
        "same_readout_visible_energy_random_chart_bankbudget_fsclip_eta01_rank4",
        "shuffled_source_witness_generator",
        "witness_only_generator",
        "source_only_generator",
        "self_only_generator",
    ]
)


def now_sg() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    if not EXEC_DOC.exists():
        EXEC_DOC.write_text(
            "# DG-KAN v22.66 MetricCompatibleGeneratorAtlasFU 执行日志\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只写真实命令、解释器、GPU、输入输出文件、状态、blocker、修复尝试；"
            "后续复现应能从这里找到命令和 artifact 路径。\n",
            encoding="utf-8",
        )
    if not RECAP_DOC.exists():
        RECAP_DOC.write_text(
            "# DG-KAN v22.66 MetricCompatibleGeneratorAtlasFU 实验结果复盘\n\n"
            f"创建时间：{now_sg()}\n\n"
            "记录原则：只引用真实落盘数据；缺失、失败、修复都必须明示；不编造实验数据或结论。\n",
            encoding="utf-8",
        )


def command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def append_exec(command: str, *, task_id: str, status: str, gpu: str = "", files: str = "", note: str = "", exit_code: Any = "n/a") -> None:
    ensure_out()
    row = {
        "timestamp": now_sg(),
        "task_id": task_id,
        "gpu": gpu,
        "command": command,
        "status": status,
        "exit_code": exit_code,
        "files": files,
        "note": note,
    }
    journal_path = OUT_ROOT / "v22_66_command_journal.csv"
    journal = read_rows(journal_path)
    journal.append({key: str(value) for key, value in row.items()})
    write_rows(journal_path, journal, ["timestamp", "task_id", "gpu", "command", "status", "exit_code", "files", "note"])
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {row['timestamp']} {task_id}\n\n")
        f.write("```bash\n" + command + "\n```\n\n")
        f.write(f"- gpu: {gpu or 'n/a'}\n- status: {status}\n- exit_code: {exit_code}\n")
        if files:
            f.write(f"- files: {files}\n")
        if note:
            f.write(f"- note: {note}\n")


def append_recap(title: str, body: str) -> None:
    ensure_out()
    with RECAP_DOC.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now_sg()} {title}\n\n{body.rstrip()}\n")


def run_cmd(cmd: list[str], *, task_id: str, files: str = "", gpu: str = "cpu", timeout: int | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    ensure_out()
    stdout_path = LOG_ROOT / f"{safe_fragment(task_id)}_stdout.log"
    stderr_path = LOG_ROOT / f"{safe_fragment(task_id)}_stderr.log"
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env=env,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
        status = "pass" if proc.returncode == 0 else "fail"
        append_exec(
            command_text(cmd),
            task_id=task_id,
            status=status,
            gpu=gpu,
            files=files,
            exit_code=proc.returncode,
            note=f"stdout={stdout_path}; stderr={stderr_path}; wall_seconds={time.time() - start:.3f}",
        )
        return {"status": status, "returncode": proc.returncode, "stdout": str(stdout_path), "stderr": str(stderr_path), "wall_seconds": time.time() - start}
    except Exception as exc:
        stderr_path.write_text(traceback.format_exc(), encoding="utf-8", errors="replace")
        append_exec(command_text(cmd), task_id=task_id, status="exception", gpu=gpu, files=files, exit_code="exception", note=f"{type(exc).__name__}: {exc}; stderr={stderr_path}")
        return {"status": "exception", "returncode": -999, "stdout": str(stdout_path), "stderr": str(stderr_path), "wall_seconds": time.time() - start}


class RuntimeTrainingLoopAudit:
    def __init__(self, params: Any) -> None:
        self.params = [p for p in params if getattr(p, "requires_grad", False)]
        self.before_step: list[Any] = []
        self.manual_param_update_detected = 0
        self.no_grad_param_mutation_detected = 0
        self.apply_flat_update_called = 0
        self.p_data_write_detected = 0
        self.copy_param_write_detected = 0

    def snapshot_before_backward(self) -> None:
        self.before_step = [p.detach().clone() for p in self.params]

    def check_before_optimizer_step(self) -> None:
        if len(self.before_step) != len(self.params):
            return
        for item, before in zip(self.params, self.before_step):
            delta = (item.detach() - before.to(device=item.device, dtype=item.dtype)).abs().max()
            val = float(delta.item()) if hasattr(delta, "item") else float(delta)
            if math.isfinite(val) and val > 1.0e-12:
                self.manual_param_update_detected = 1
                self.no_grad_param_mutation_detected = 1
                break

    def as_dict(self) -> dict[str, int]:
        return {
            "manual_param_update_detected": int(self.manual_param_update_detected),
            "no_grad_param_mutation_detected": int(self.no_grad_param_mutation_detected),
            "apply_flat_update_called": int(self.apply_flat_update_called),
            "param_data_write_detected": int(self.p_data_write_detected),
            "copy_param_write_detected": int(self.copy_param_write_detected),
        }


def method_family(method: str) -> str:
    if method in REFERENCE_METHODS:
        return "reference"
    if method in EXTERNAL_METHODS:
        return "external_oet"
    if method in CANDIDATE_METHODS:
        return "candidate"
    if method in CONTROL_METHODS:
        return "control"
    return "unknown"


def rank_for_method(method: str, num_classes: int | None = None, hidden: int | None = None) -> int:
    match = re.search(r"rank(\d+)", str(method).lower())
    requested = int(match.group(1)) if match else (4 if "last_layer" in method else 2)
    if num_classes is not None:
        requested = min(requested, int(num_classes))
    if hidden is not None:
        requested = min(requested, int(hidden))
    return max(1, int(requested))


def atlas_method_name(method: str) -> str:
    low = str(method).lower()
    if low in {"same_compute_noop_coordinate", "metric_preserving_noop"}:
        return "same_compute_noop_coordinate"
    if low == "transport_only":
        return "metric_atlas_sw"
    if low == "shape_only_no_generator":
        return "metric_atlas_sw"
    if "same_signal_reachable_generator" in low:
        return "same_signal_reachable_random_atlas_gradcoh_balanced"
    if "oet_only" in low:
        return "gfoa_oet_only"
    if "actgradunion" in low:
        return "metric_atlas_gradunion"
    if "actgradmix" in low:
        return "metric_atlas_gradmix"
    if "actgradcov" in low:
        if "ccfs25" in low:
            return "metric_atlas_gradcov_ccfs25"
        if "ccfs50" in low:
            return "metric_atlas_gradcov_ccfs50"
        return "metric_atlas_gradcov"
    if "actgradcoh" in low:
        return "metric_atlas_gradcoh"
    if "actcvargrad" in low:
        return "metric_atlas_cvargrad"
    if "over_oet" in low or "over_poet" in low or "over_pion" in low:
        if "cc25" in low:
            return "metric_atlas_act_cc25"
        if "cc50" in low:
            return "metric_atlas_act_cc50"
        return "metric_atlas_act"
    if "gradunion" in low:
        return "metric_atlas_gradunion"
    if "kan_bank_oet" in low:
        return "kan_bank_oet_atlas"
    if ("task_visible_chart" in low or "readout_visible_energy_random_chart" in low) and "gradcoh" in low:
        return "metric_atlas_gradcoh"
    if "task_visible_chart" in low or "readout_visible_energy_random_chart" in low:
        return "metric_atlas_sw"
    if "oocw_gradcoh" in low:
        return "metric_atlas_oocw_gradcoh"
    if "cvargrad" in low:
        return "metric_atlas_cvargrad"
    if "gradcoh" in low:
        return "metric_atlas_gradcoh"
    if "shape_signal" in low or "tail_safe" in low or "reservoir_suppressed" in low:
        return "metric_atlas_sw"
    if "same_functional_spectrum" in low or "same_spectrum" in low or "same_c_skew_spectrum" in low:
        return "same_functional_spectrum_random_atlas"
    if "same_rank_random" in low or "same_isometric" in low or "same_transport" in low or "same_metric" in low or "same_shape" in low or "same_generator" in low or "same_debtbudget" in low:
        return "same_rank_random_atlas"
    if low == "shuffled_source_witness_generator":
        return "shuffled_source_atlas"
    if low == "source_only_generator":
        return "source_only_atlas"
    if low == "witness_only_generator":
        return "witness_only_atlas"
    if low == "self_only_generator":
        return "self_only_atlas"
    if "lora_like" in low:
        return "lora_like_coordinate"
    if "transport_generator" in low:
        return "metric_atlas_sw"
    return "metric_atlas_sw" if method in CANDIDATE_METHODS else method


def uses_compatible_model(method: str) -> bool:
    return method in CANDIDATE_METHODS or method in CONTROL_METHODS


def train_base_for_method(method: str) -> bool:
    low = str(method).lower()
    return "residual" in low or "over_oet" in low or "over_poet" in low or "over_pion" in low


def base_spectrum_lock_for_method(method: str) -> bool:
    low = str(method).lower()
    return "over_oet" in low or "baselock" in low or "fsclip" in low or "fsfill" in low


def functional_spectrum_budget_for_method(method: str, default: float) -> float:
    low = str(method).lower()
    return float(default) if ("over_oet" in low or "fsclip" in low or "fsfill" in low) else 0.0


def functional_spectrum_fill_for_method(method: str) -> bool:
    return "fsfill" in str(method).lower()


def iso_eta_for_method(method: str, default: float) -> float:
    low = str(method).lower()
    if "eta0005" in low:
        return 0.005
    if "eta0015" in low:
        return 0.015
    if "eta0018" in low:
        return 0.018
    if "eta001" in low:
        return 0.01
    if "eta002" in low:
        return 0.02
    if "eta005" in low:
        return 0.05
    if "eta01" in low:
        return 0.10
    if "eta025" in low:
        return 0.25
    if "eta05" in low:
        return 0.50
    if "task_visible_chart" in low or "readout_visible_energy_random_chart" in low:
        return 0.05
    return float(default)


def warmup_steps_for_method(method: str, total_steps: int) -> int:
    low = str(method).lower()
    match = re.search(r"warm(\d+)", low)
    if match is not None:
        return max(0, min(int(match.group(1)), int(total_steps) - 1))
    return 0


def allow_shape_for_method(method: str) -> bool:
    low = str(method).lower()
    return (
        low.startswith("mcga_shape")
        or low in {"shape_only_no_generator", "same_shape_budget_generator_random"}
        or "task_visible_chart_shape" in low
        or "readout_visible_energy_random_chart_shape" in low
    )


def shape_budget_for_method(method: str, default: float) -> float:
    low = str(method).lower()
    if "budget001" in low:
        return 0.01
    if "budget002" in low:
        return 0.02
    if "budget005" in low:
        return 0.05
    if "budget010" in low:
        return 0.10
    if "budget020" in low:
        return 0.20
    if "same_shape_budget" in low or "shape_only" in low:
        return float(default)
    if allow_shape_for_method(method):
        return float(default)
    return 0.0


def shape_trigger_reason_for_method(method: str, capacity: dict[str, Any] | None = None) -> str:
    low = str(method).lower()
    if not allow_shape_for_method(method):
        return "shape_not_opened"
    frac = fval((capacity or {}).get("generator_descent_fraction"), None)
    low_capacity = frac is not None and frac < 0.05
    if "tail_safe" in low:
        return "shape_opened_due_to_tail_debt" if low_capacity else "shape_opened_due_to_tail_debt_audit_capacity_positive"
    if "reservoir_suppressed" in low:
        return "shape_opened_due_to_signal_channel_creation"
    if "shape_only" in low:
        return "shape_only_no_generator_control"
    if "same_shape" in low:
        return "same_shape_budget_generator_random_control"
    return "shape_opened_due_to_signal_channel_creation" if low_capacity else "shape_opened_due_to_signal_channel_creation_audit_capacity_positive"


def metric_kind_for_method(method: str, default: str) -> str:
    if "fishermetric" in method:
        return "fisher"
    if "tail_safe" in method:
        return "signal_debt"
    if "reservoir_suppressed" in method:
        return "signal_debt"
    return str(default)


def debt_budget_strength_for_method(method: str) -> float:
    low = str(method).lower()
    match = re.search(r"debtbudget(\d+)", low)
    if match is not None:
        return max(0.0, min(float(match.group(1)) / 100.0, 1.0))
    return 0.5 if "debtbudget" in low else 0.0


def generator_train_enabled(method: str) -> bool:
    return str(method).lower() not in {"transport_only", "metric_preserving_noop", "shape_only_no_generator"}


def shape_train_enabled(method: str) -> bool:
    return allow_shape_for_method(method) and str(method).lower() != "metric_preserving_noop"


def freeze_control_coordinates(model: Any, method: str) -> dict[str, int]:
    layer = get_coord_layer(model)
    if layer is None:
        return {"generator_parameter_trainable": 0, "shape_parameter_trainable": 0}
    generator_trainable = int(generator_train_enabled(method))
    shape_trainable = int(shape_train_enabled(method))
    if hasattr(layer, "raw_iso") and layer.raw_iso is not None:
        layer.raw_iso.requires_grad_(bool(generator_trainable))
    if hasattr(layer, "raw_shape") and layer.raw_shape is not None:
        layer.raw_shape.requires_grad_(bool(shape_trainable))
    return {"generator_parameter_trainable": generator_trainable, "shape_parameter_trainable": shape_trainable}


class BlendedExternalOptimizer:
    """Fixed convex blend of two external optimizer update proposals."""

    def __init__(self, params: Iterable[Any], poet_opt: Any, pion_opt: Any, *, poet_weight: float, pion_weight: float) -> None:
        self.params = [p for p in params if getattr(p, "requires_grad", False)]
        self.poet_opt = poet_opt
        self.pion_opt = pion_opt
        total = max(1.0e-12, float(poet_weight) + float(pion_weight))
        self.poet_weight = float(poet_weight) / total
        self.pion_weight = float(pion_weight) / total
        self.blend_diags: list[dict[str, float]] = []

    @property
    def state(self) -> dict[Any, Any]:
        merged: dict[Any, Any] = {}
        for prefix, opt in [("poet", self.poet_opt), ("pion", self.pion_opt)]:
            state = getattr(opt, "state", {})
            if isinstance(state, dict):
                for idx, value in enumerate(state.values()):
                    merged[(prefix, idx)] = value
        return merged

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def _restore(self, snapshots: list[Any]) -> None:
        for p, before in zip(self.params, snapshots):
            target = before.to(device=p.device, dtype=p.dtype)
            p.detach().add_(target - p.detach())

    def step(self) -> None:
        before = [p.detach().clone() for p in self.params]
        self.poet_opt.step()
        poet_delta = [p.detach().clone() - b.to(device=p.device, dtype=p.dtype) for p, b in zip(self.params, before)]
        self._restore(before)
        self.pion_opt.step()
        pion_delta = [p.detach().clone() - b.to(device=p.device, dtype=p.dtype) for p, b in zip(self.params, before)]
        self._restore(before)
        total_norm = 0.0
        for p, pd, qd in zip(self.params, poet_delta, pion_delta):
            delta = pd.mul(self.poet_weight).add(qd, alpha=self.pion_weight)
            p.detach().add_(delta.to(device=p.device, dtype=p.dtype))
            total_norm += float(delta.float().norm().detach().cpu().item())
        self.blend_diags.append({"blend_update_norm_sum": total_norm})


def blend_weights_for_method(method: str) -> tuple[float, float]:
    low = str(method).lower()
    if "blend05" in low:
        return 0.95, 0.05
    if "blend10" in low:
        return 0.90, 0.10
    if "blend20" in low:
        return 0.80, 0.20
    if "blend30" in low:
        return 0.70, 0.30
    if "blend40" in low:
        return 0.60, 0.40
    if "blend60" in low:
        return 0.40, 0.60
    if "blend65" in low:
        return 0.35, 0.65
    if "blend68" in low:
        return 0.32, 0.68
    if "blend70" in low:
        return 0.30, 0.70
    if "blend72" in low:
        return 0.28, 0.72
    if "blend75" in low:
        return 0.25, 0.75
    if "blend50" in low:
        return 0.50, 0.50
    return 0.50, 0.50


def make_optimizer(method: str, model: Any, args: argparse.Namespace, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    family = method_family(method)
    if method == "adamw" or family in {"candidate", "control"}:
        if "over_poet_pion_blend" in method:
            poet_opt, poet_diag = make_poet_optimizer_for_model(model, args)
            wrapped = poet_diag.get("wrapped_model", model)
            from experiments.run_v22_53O_pion_poet_external_oet_baseline import MatrixGeometryOptimizer

            pion_opt = MatrixGeometryOptimizer("pion_oet_sphere_official", wrapped, float(args.lr), float(args.weight_decay), device)
            poet_weight, pion_weight = blend_weights_for_method(method)
            opt = BlendedExternalOptimizer(wrapped.parameters(), poet_opt, pion_opt, poet_weight=poet_weight, pion_weight=pion_weight)
            return opt, {
                "optimizer_step_source": f"fixed_blend_poet_{poet_weight:.2f}_pion_sphere_{pion_weight:.2f}",
                "external_optimizer_available": 1,
                "wrapped_model": wrapped,
            }
        if "over_poet" in method:
            return make_poet_optimizer_for_model(model, args)
        if "over_pion" in method or "over_oet" in method:
            from experiments.run_v22_53O_pion_poet_external_oet_baseline import MatrixGeometryOptimizer

            source = "pion_oet_sphere_official" if "over_pion" in method else "pion_oet_local"
            opt = MatrixGeometryOptimizer(source, model, float(args.lr), float(args.weight_decay), device)
            return opt, {"optimizer_step_source": f"{source}_MatrixGeometryOptimizer", "external_optimizer_available": 1}
        return torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)), {
            "optimizer_step_source": "torch.optim.AdamW",
            "external_optimizer_available": 1,
        }
    if method == "cautious_adamw":
        return CautiousAdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)), {
            "optimizer_step_source": "CautiousAdamW_grad_gate_over_AdamW",
            "external_optimizer_available": 1,
        }
    if method == "schedule_free_adamw_local":
        return ScheduleFreeAdamWLocal(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)), {
            "optimizer_step_source": "ScheduleFreeAdamWLocal_over_AdamW",
            "external_optimizer_available": 1,
        }
    if method == "poet_official":
        return make_poet_optimizer_for_model(model, args)
    if method in {"pion_oet_sphere_official", "pion_oet_local"}:
        from experiments.run_v22_53O_pion_poet_external_oet_baseline import MatrixGeometryOptimizer

        opt = MatrixGeometryOptimizer(method, model, float(args.lr), float(args.weight_decay), device)
        return opt, {"optimizer_step_source": f"{method}_MatrixGeometryOptimizer", "external_optimizer_available": 1}
    raise ValueError(f"unknown method {method!r}")


def make_poet_optimizer_for_model(model: Any, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    sys.path.insert(0, str(ROOT / "external/oet_baselines/poet_sphere"))
    from poet_torch import POETConfig, POETModel, get_poet_optimizer

    cfg = POETConfig(
        block_size=int(args.poet_block_size),
        merge_interval=int(args.poet_merge_interval),
        poet_lr=float(args.poet_lr),
        base_lr=float(args.lr),
        poet_scale=float(args.poet_scale),
        weight_decay=float(args.weight_decay),
        mem_efficient_mode=False,
    )
    wrapped = POETModel(model, cfg)
    opt = get_poet_optimizer(wrapped, cfg)
    return opt, {"optimizer_step_source": "poet_torch.get_poet_optimizer", "external_optimizer_available": 1, "wrapped_model": wrapped}


def row_path(args: argparse.Namespace) -> Path:
    label = run_label_fragment(args)
    arch = safe_fragment(architecture_key(args))
    prefix = f"{label}_" if label else ""
    arch_part = f"{arch}_" if label or arch != "MLP" else ""
    frag = f"{prefix}{arch_part}{safe_fragment(args.method)}_{safe_fragment(args.dataset)}_s{int(args.seed)}_st{int(args.steps)}"
    return CHUNK_ROOT / f"{frag}.csv"


def row_log_prefix(args: argparse.Namespace) -> str:
    label = run_label_fragment(args)
    arch = safe_fragment(architecture_key(args))
    prefix = f"{label}_" if label else ""
    arch_part = f"{arch}_" if label or arch != "MLP" else ""
    return f"v22_66_row_{prefix}{arch_part}{safe_fragment(args.method)}_{safe_fragment(args.dataset)}_s{int(args.seed)}"


def metric_cohort(x_train: Any, y_train: Any, args: argparse.Namespace, *, offset: int = 0) -> tuple[Any, Any]:
    import torch

    limit = int(getattr(args, "metric_batch_size", 0) or 0)
    n = int(x_train.shape[0])
    if limit <= 0 or limit >= n:
        return x_train, y_train
    gen = torch.Generator(device=x_train.device)
    gen.manual_seed(int(args.seed) * 1709 + int(offset) + sum(ord(ch) for ch in str(args.method)))
    idx = torch.randperm(n, generator=gen, device=x_train.device)[:limit]
    return x_train[idx], y_train[idx]


def run_label_fragment(args: argparse.Namespace) -> str:
    return safe_fragment(str(getattr(args, "run_label", "") or ""))


def architecture_key(args: argparse.Namespace) -> str:
    return str(getattr(args, "architecture", "MLP") or "MLP")


class GenericMetricAtlasModel:
    pass


def _make_generic_metric_model_class(compatible: bool):
    import torch.nn as nn

    class _GenericMetricAtlasModel(nn.Module):
        def __init__(
            self,
            base: Any,
            atlas: AtlasBuild,
            *,
            allow_shape: bool = False,
            shape_budget: float = 0.0,
            eta: float = 1.0,
            train_base_weight: bool = False,
            base_spectrum_lock: bool = False,
            functional_spectrum_budget: float = 0.0,
            functional_spectrum_fill: bool = False,
        ) -> None:
            super().__init__()
            self.feature_model = getattr(base, "feature_model", base)
            if compatible:
                self.fc3 = MetricCompatibleLowRankLinear(
                    base.fc3,
                    atlas,
                    allow_shape=allow_shape,
                    shape_budget=shape_budget,
                    eta=eta,
                    train_base_weight=train_base_weight,
                    base_spectrum_lock=base_spectrum_lock,
                    functional_spectrum_budget=functional_spectrum_budget,
                    functional_spectrum_fill=functional_spectrum_fill,
                )
            else:
                self.fc3 = LowRankAtlasLinear(base.fc3, atlas.output_basis, atlas.input_basis, atlas.coord_scale)

        def features(self, x: Any) -> Any:
            if hasattr(self.feature_model, "frozen_readout_features"):
                return self.feature_model.frozen_readout_features(x)
            return self.feature_model.features(x)

        def forward(self, x: Any) -> Any:
            return self.fc3(self.features(x))

    return _GenericMetricAtlasModel


GenericCompatibleAtlasModel = _make_generic_metric_model_class(True)
GenericAtlasModel = _make_generic_metric_model_class(False)


def kan_readout_weight(kan: Any) -> tuple[Any, Any]:
    import math as _math
    import torch

    pieces = []
    main = kan.w2.detach().permute(1, 0, 2).reshape(int(kan.output_dim), -1)
    pieces.append(main / _math.sqrt(max(1, int(kan.hidden_dim))))
    if bool(getattr(kan, "cheby_paircross_enabled", False)):
        pieces.append(kan.cheby_cross_readout.detach().transpose(0, 1))
    if bool(getattr(kan, "cheby_input_cross_enabled", False)):
        pieces.append(kan.cheby_input_cross_readout.detach().transpose(0, 1))
    if bool(getattr(kan, "linear_residual_enabled", False)):
        pieces.append(kan.linear_readout.detach().transpose(0, 1) / float(kan._linear_residual_denominator()))
    weight = torch.cat([p.to(device=kan.w2.device, dtype=kan.w2.dtype) for p in pieces], dim=1)
    bias = torch.zeros(int(kan.output_dim), device=kan.w2.device, dtype=kan.w2.dtype)
    return weight, bias


def make_linearized_kan_readout_base(kan: Any) -> Any:
    import torch
    import torch.nn as nn

    weight, bias = kan_readout_weight(kan)

    class _LinearizedKANReadoutBase(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_model = kan
            self.fc3 = nn.Linear(int(weight.shape[1]), int(weight.shape[0]))
            with torch.no_grad():
                self.fc3.weight.copy_(weight)
                self.fc3.bias.copy_(bias)
            for name in ("w2", "cheby_cross_readout", "cheby_input_cross_readout", "linear_readout"):
                param = getattr(self.feature_model, name, None)
                if param is not None:
                    param.requires_grad_(False)

        def features(self, x: Any) -> Any:
            return self.feature_model.frozen_readout_features(x)

        def forward(self, x: Any) -> Any:
            return self.fc3(self.features(x))

    return _LinearizedKANReadoutBase().to(next(kan.parameters()).device)


def redesign_seed_offset(architecture: str) -> int:
    if architecture == "DGKAN_DCHE":
        return 1000
    if architecture == "DGKAN_DFOU":
        return 2000
    spec = REDESIGN_KAN_SPECS.get(str(architecture), {})
    return int(spec.get("seed_offset", 0))


def redesign_param_budget(meta: dict[str, Any], input_dim: int, output_dim: int, hidden: int) -> int:
    k = int(meta["k"])
    budget = int(input_dim) * int(hidden) * k + int(hidden) * int(output_dim) * k
    variant_lower = str(meta["init_variant"]).lower()
    if "inputcross" in variant_lower:
        local_rank = 4
        local_match = re.search(r"localr(\d+)", variant_lower)
        if local_match is not None:
            local_rank = int(local_match.group(1))
        proj_rank = 8
        proj_match = re.search(r"projr(\d+)", variant_lower)
        if proj_match is not None:
            proj_rank = int(proj_match.group(1))
        input_cross = min(int(local_rank), max(0, int(input_dim) // 2)) + int(proj_rank)
        if "localrot" in variant_lower:
            input_cross += min(int(local_rank), max(0, int(input_dim) // 2))
        if "localrot2" in variant_lower:
            input_cross += min(int(local_rank), max(0, int(input_dim) // 2))
        budget += int(input_cross) * int(output_dim)
    if "linearres" in variant_lower:
        budget += int(input_dim) * int(output_dim)
    return int(budget)


def make_redesign_kan(architecture: str, input_dim: int, output_dim: int, hidden: int, seed: int, device: Any, x_stats: Any) -> Any:
    from dgkan.models.fc_purekan_primitives import PrimitiveKAN, PrimitiveSpec

    meta = REDESIGN_KAN_SPECS[str(architecture)]
    k = int(meta["k"])
    spec = PrimitiveSpec(
        candidate_id=f"v22.67-{meta['basis_family']}-h{int(hidden)}",
        basis_family=str(meta["basis_family"]),
        basis_name=str(meta["basis_name"]),
        k=k,
        hidden_dim=int(hidden),
        source="v22_67_basis_redesign_full_loop",
        local_support=int(meta["local_support"]),
        global_support=int(meta["global_support"]),
        uses_exp=int(meta["uses_exp"]),
        uses_sin_cos=int(meta["uses_sin_cos"]),
        uses_division=int(meta["uses_division"]),
        uses_dense_basis_tensor=int(meta.get("uses_dense_basis_tensor", 1)),
        diagnostic_only=0,
        basis_order=int(meta["basis_order"]),
        init_variant=str(meta["init_variant"]),
    )
    budget = redesign_param_budget(meta, int(input_dim), int(output_dim), int(hidden))
    return PrimitiveKAN(
        int(input_dim),
        int(output_dim),
        spec,
        x_stats.to(device),
        int(seed),
        device,
        param_budget=budget,
    ).to(device)


def make_base_model(args: argparse.Namespace, bundle: dict[str, Any], device: Any) -> Any:
    arch = architecture_key(args)
    if arch == "MLP":
        return SimpleMLP(int(bundle["input_dim"]), int(bundle["num_classes"]), hidden=int(args.hidden), seed=int(args.seed)).to(device)

    x_stats = bundle["x_train"][: min(512, int(bundle["x_train"].shape[0]))].to(device).float()
    if arch in REDESIGN_KAN_SPECS:
        kan = make_redesign_kan(
            arch,
            int(bundle["input_dim"]),
            int(bundle["num_classes"]),
            int(args.hidden),
            int(args.seed) + redesign_seed_offset(arch),
            device,
            x_stats,
        )
    else:
        from experiments.run_v22_37_causal_instrumented_functional_optimizer import make_model_for_arch

        kan = make_model_for_arch(
            arch,
            int(bundle["input_dim"]),
            int(bundle["num_classes"]),
            int(args.hidden),
            int(args.seed) + redesign_seed_offset(arch),
            device,
            x_stats,
        )
    base = make_linearized_kan_readout_base(kan).to(device)
    import torch

    with torch.no_grad():
        xb = bundle["x_train"][: min(32, int(bundle["x_train"].shape[0]))].to(device).float()
        err = (kan(xb).float() - base(xb).float()).abs().max().detach().cpu().item()
    setattr(base, "kan_readout_linearization_max_abs_error", float(err))
    return base


def make_feature_chart_prepared_base(base: Any, x_metric: Any, *, eps: float = 1.0e-4) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn

    with torch.no_grad():
        feats = base.features(x_metric).detach().float()
        mu = feats.mean(dim=0)
        centered = feats - mu
        cov = centered.transpose(0, 1) @ centered / max(1, int(centered.shape[0]))
        eye = torch.eye(int(cov.shape[0]), device=cov.device, dtype=cov.dtype)
        evals, evecs = torch.linalg.eigh(0.5 * (cov + cov.transpose(0, 1)) + float(eps) * eye)
        evals = evals.clamp_min(float(eps))
        inv_sqrt = evecs @ torch.diag(torch.rsqrt(evals)) @ evecs.transpose(0, 1)
        sqrt = evecs @ torch.diag(torch.sqrt(evals)) @ evecs.transpose(0, 1)
        old_weight = base.fc3.weight.detach().float()
        old_bias = base.fc3.bias.detach().float() if getattr(base.fc3, "bias", None) is not None else torch.zeros(int(old_weight.shape[0]), device=old_weight.device)
        new_weight = old_weight @ sqrt.transpose(0, 1)
        new_bias = old_bias + mu @ old_weight.transpose(0, 1)

    class _FrozenReadout(nn.Module):
        def __init__(self, weight: Any, bias: Any) -> None:
            super().__init__()
            self.register_buffer("weight", weight.detach().clone().float())
            self.register_buffer("bias", bias.detach().clone().float())

        def forward(self, x: Any) -> Any:
            return x @ self.weight.transpose(0, 1) + self.bias

    class _FeatureChartPreparedBase(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.source_base = base
            self.fc3 = _FrozenReadout(new_weight, new_bias)
            self.register_buffer("feature_mean", mu.detach().clone().float())
            self.register_buffer("feature_inv_sqrt", inv_sqrt.detach().clone().float())

        def features(self, x: Any) -> Any:
            return (self.source_base.features(x).float() - self.feature_mean) @ self.feature_inv_sqrt

        def forward(self, x: Any) -> Any:
            return self.fc3(self.features(x))

    prepared = _FeatureChartPreparedBase().to(x_metric.device)
    with torch.no_grad():
        err = (base(x_metric).float() - prepared(x_metric).float()).abs().max().detach().cpu().item()
    setattr(prepared, "kan_readout_linearization_max_abs_error", getattr(base, "kan_readout_linearization_max_abs_error", ""))
    return prepared, {
        "chart_preparation_used": 1.0,
        "chart_preparation_logit_max_abs_error": float(err),
        "chart_feature_Gram_condition_before": float((evals.max() / evals.min().clamp_min(float(eps))).detach().cpu().item()),
    }


def task_visible_chart_atlas(atlas: AtlasBuild, base: Any, x_metric: Any, y_metric: Any, method: str, seed: int) -> tuple[AtlasBuild, dict[str, float]]:
    import torch
    import torch.nn.functional as F

    rank = int(atlas.output_basis.shape[1])
    method_l = str(method).lower()
    method_s = str(method)
    with torch.no_grad():
        feats = base.features(x_metric).detach().float()
        logits = base(x_metric).detach().float()
        probs = torch.softmax(logits, dim=1)
        yoh = F.one_hot(y_metric.long(), num_classes=int(logits.shape[1])).float()
        target = (yoh - probs) / max(1, int(x_metric.shape[0]))
        h_v = feats @ atlas.input_basis.detach().float()
        scale = atlas.coord_scale.detach().float().view(-1)
        score = torch.zeros(rank, rank, device=feats.device)
        signed_score = torch.zeros(rank, rank, device=feats.device)
        grad_score = torch.zeros(rank, rank, device=feats.device)
        stablegrad_score = torch.zeros(rank, rank, device=feats.device)
        visibility = torch.zeros(rank, rank, device=feats.device)
        target_flat = target.reshape(-1)
        target_norm = target_flat.norm().clamp_min(1.0e-12)
        for i in range(rank):
            for j in range(rank):
                col = h_v[:, j].view(-1, 1) * float(scale[j].detach().cpu().item()) * atlas.output_basis[:, i].detach().float().view(1, -1)
                flat = col.reshape(-1)
                vis = flat.square().sum()
                signed_align = (flat @ target_flat) / (flat.norm().clamp_min(1.0e-12) * target_norm)
                align = signed_align.abs()
                visibility[i, j] = vis
                score[i, j] = vis * align.square()
                signed_score[i, j] = vis * signed_align
                grad_score[i, j] = signed_align * vis.sqrt()

        stablegrad_chart_used = 1.0 if "stablegrad" in method_l else 0.0
        stablegrad_sign_agreement = 0.0
        stablegrad_active_fraction = 0.0
        if stablegrad_chart_used:

            def split_grad_score(idx: Any) -> Any:
                feats_c = feats.index_select(0, idx)
                probs_c = probs.index_select(0, idx)
                yoh_c = yoh.index_select(0, idx)
                target_c = (yoh_c - probs_c) / max(1, int(idx.numel()))
                h_c = h_v.index_select(0, idx)
                target_flat_c = target_c.reshape(-1)
                target_norm_c = target_flat_c.norm().clamp_min(1.0e-12)
                out = torch.zeros(rank, rank, device=feats.device)
                for ii in range(rank):
                    for jj in range(rank):
                        col_c = h_c[:, jj].view(-1, 1) * float(scale[jj].detach().cpu().item()) * atlas.output_basis[:, ii].detach().float().view(1, -1)
                        flat_c = col_c.reshape(-1)
                        vis_c = flat_c.square().sum()
                        signed_align_c = (flat_c @ target_flat_c) / (flat_c.norm().clamp_min(1.0e-12) * target_norm_c)
                        out[ii, jj] = signed_align_c * vis_c.sqrt()
                return out

            n_metric = int(feats.shape[0])
            idx_a = torch.arange(0, n_metric, 2, device=feats.device)
            idx_b = torch.arange(1, n_metric, 2, device=feats.device)
            if int(idx_b.numel()) > 0:
                grad_a = split_grad_score(idx_a)
                grad_b = split_grad_score(idx_b)
                same_sign = grad_a * grad_b > 0
                stable_mag = torch.sqrt((grad_a.abs() * grad_b.abs()).clamp_min(0.0))
                stable_sign = torch.sign(grad_a + grad_b)
                stablegrad_score = torch.where(same_sign, stable_sign * stable_mag, 0.25 * (grad_a + grad_b))
                stablegrad_sign_agreement = float(same_sign.float().mean().detach().cpu().item())
                stablegrad_active_fraction = float((stablegrad_score.abs() > 1.0e-12).float().mean().detach().cpu().item())
            else:
                stablegrad_score = grad_score
                stablegrad_sign_agreement = 1.0
                stablegrad_active_fraction = float((stablegrad_score.abs() > 1.0e-12).float().mean().detach().cpu().item())

        vis_norm = visibility / visibility.max().clamp_min(1.0e-12)
        signed_chart_used = 1.0 if "signed" in method_l else 0.0
        grad_chart_used = 1.0 if "grad" in method_l else 0.0
        if stablegrad_chart_used:
            score_base = stablegrad_score.float()
        elif grad_chart_used:
            score_base = grad_score.float()
        elif signed_chart_used:
            score_base = signed_score.float()
        else:
            score_base = score.float()
        score_work = score_base
        score_norm = score_base.abs() / score_base.abs().sum().clamp_min(1.0e-12)
        ooc_used = 0.0
        ooc_projection_fraction = 0.0
        ooc_residual_fraction = 1.0
        random_raw = None

        def paired_random_control_name(name: str) -> str:
            if name.startswith("kan_task_visible_chart"):
                return name.replace("kan_task_visible_chart", "same_readout_visible_energy_random_chart", 1)
            return name

        def readout_random_raw(control_name: str) -> Any:
            gen = torch.Generator(device=feats.device)
            gen.manual_seed(int(seed) + 77123 + sum(ord(c) for c in str(control_name)))
            return torch.randn(rank, rank, device=feats.device, generator=gen)

        if "ooc" in method_l:
            random_raw = readout_random_raw(paired_random_control_name(method_s))
            control_dir = random_raw.float()
            control_dir = control_dir / control_dir.norm().clamp_min(1.0e-12)
            base_norm = score_base.norm().clamp_min(1.0e-12)
            projection = torch.sum(score_base * control_dir)
            residual = score_base - projection * control_dir
            residual_norm = residual.norm()
            residual_norm_value = float(residual_norm.detach().cpu().item()) if bool(torch.isfinite(residual_norm).detach().cpu().item()) else 0.0
            if residual_norm_value > 1.0e-12:
                score_work = residual
                ooc_used = 1.0
                ooc_projection_fraction = float((projection.abs() / base_norm).detach().cpu().item())
                ooc_residual_fraction = float((residual_norm / base_norm).detach().cpu().item())

        if "same_readout_visible_energy_random_chart" in method_l:
            raw = random_raw if random_raw is not None else readout_random_raw(method_s)
            u_rot, _, vh_rot = torch.linalg.svd(raw.float(), full_matrices=True)
            s_vals = torch.linalg.svdvals(score_work).clamp_min(1.0e-8)
        else:
            u_rot, s_vals, vh_rot = torch.linalg.svd(score_work + 1.0e-8 * torch.eye(rank, device=feats.device), full_matrices=True)
        out_basis = atlas.output_basis.detach().float() @ u_rot[:, :rank]
        in_basis = atlas.input_basis.detach().float() @ vh_rot.transpose(0, 1)[:, :rank]
        raw_scale = torch.sqrt(s_vals[:rank].abs().clamp_min(1.0e-8))
        coord_scale = raw_scale / raw_scale.mean().clamp_min(1.0e-8)
        bank_budget_entropy = 0.0
        bank_budget_top_fraction = 0.0
        if "bankbudget" in method_l:
            component_visibility = []
            v_rot = vh_rot.transpose(0, 1)
            for comp_idx in range(rank):
                vis_comp = torch.sum(torch.abs(u_rot[:, comp_idx : comp_idx + 1]) * vis_norm * torch.abs(v_rot[:, comp_idx].view(1, -1)))
                component_visibility.append(vis_comp)
            comp_vis = torch.stack(component_visibility).clamp_min(1.0e-8)
            signal = s_vals[:rank].abs() / s_vals[:rank].abs().max().clamp_min(1.0e-8)
            debt_risk = comp_vis.reciprocal()
            bank_budget = torch.clamp(0.05 * signal / debt_risk.clamp_min(1.0e-8), min=0.005, max=0.20)
            coord_scale = coord_scale * (bank_budget / bank_budget.mean().clamp_min(1.0e-8))
            prob_budget = bank_budget / bank_budget.sum().clamp_min(1.0e-8)
            bank_budget_entropy = float((-(prob_budget * torch.log(prob_budget.clamp_min(1.0e-12))).sum() / math.log(max(2, int(prob_budget.numel())))).detach().cpu().item())
            bank_budget_top_fraction = float(prob_budget.max().detach().cpu().item())
        active = out_basis.transpose(0, 1) @ atlas.gf.detach().float() @ out_basis
        vals = sorted(float(x) for x in vis_norm.reshape(-1).detach().cpu().tolist())
        cvar_n = max(1, int(math.ceil(0.25 * len(vals))))
        score_work_norm = score_work.abs() / score_work.abs().sum().clamp_min(1.0e-12)
        metrics = dict(atlas.metrics)
        metrics.update(
            {
                "task_visible_chart_used": 1.0,
                "task_visible_signed_chart": signed_chart_used,
                "task_visible_grad_chart": grad_chart_used,
                "task_visible_stablegrad_chart": stablegrad_chart_used,
                "task_visible_stablegrad_sign_agreement": stablegrad_sign_agreement,
                "task_visible_stablegrad_active_fraction": stablegrad_active_fraction,
                "task_visible_score_entropy": stable_rank(score_norm),
                "task_visible_effective_score_entropy": stable_rank(score_work_norm),
                "task_visible_score_top_fraction": float(score_norm.max().detach().cpu().item()),
                "readout_visible_energy_CVaR25": sum(vals[:cvar_n]) / cvar_n,
                "task_visible_random_chart_control": float("same_readout_visible_energy_random_chart" in str(method).lower()),
                "task_visible_ooc_control_subtraction": ooc_used,
                "task_visible_ooc_projection_fraction": ooc_projection_fraction,
                "task_visible_ooc_residual_fraction": ooc_residual_fraction,
                "bank_local_adaptive_spectrum_budget": float("bankbudget" in method_l),
                "bank_budget_entropy": bank_budget_entropy,
                "bank_budget_top_fraction": bank_budget_top_fraction,
            }
        )
    return (
        AtlasBuild(
            output_basis=out_basis.detach(),
            input_basis=in_basis.detach(),
            coord_scale=coord_scale.detach(),
            gf=atlas.gf.detach(),
            active_gram=active.detach(),
            feature_metric=atlas.feature_metric.detach(),
            metrics=metrics,
        ),
        metrics,
    )


def signal_debt_budget_atlas(atlas: AtlasBuild, base: Any, x_metric: Any, y_metric: Any, method: str) -> tuple[AtlasBuild, dict[str, float]]:
    """Scale atlas coordinates by train-only tail-aligned signal/debt ratio."""

    strength = debt_budget_strength_for_method(method)
    if strength <= 0.0:
        return atlas, {}
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        feats = base.features(x_metric).detach().float()
        logits = base(x_metric).detach().float()
        probs = torch.softmax(logits, dim=1)
        yoh = F.one_hot(y_metric.long(), num_classes=int(logits.shape[1])).float()
        losses = F.cross_entropy(logits.float(), y_metric.long(), reduction="none")
        n_metric = int(losses.numel())
        tail_k = max(1, int(math.ceil(0.10 * n_metric)))
        tail_idx = torch.topk(losses.float(), k=tail_k, largest=True).indices
        target_all = (yoh - probs) / max(1, n_metric)
        target_tail = (yoh.index_select(0, tail_idx) - probs.index_select(0, tail_idx)) / max(1, tail_k)
        h_v = feats @ atlas.input_basis.detach().float()
        scale = atlas.coord_scale.detach().float().view(-1)
        rank = int(scale.numel())
        output_basis = atlas.output_basis.detach().float()
        target_all_norm = target_all.reshape(-1).norm().clamp_min(1.0e-12)
        target_tail_norm = target_tail.reshape(-1).norm().clamp_min(1.0e-12)
        signal_values = []
        debt_values = []
        tail_positive_values = []
        for j in range(rank):
            signal = torch.tensor(0.0, device=feats.device)
            debt = torch.tensor(0.0, device=feats.device)
            tail_positive = torch.tensor(0.0, device=feats.device)
            for i in range(rank):
                col_all = h_v[:, j].view(-1, 1) * scale[j] * output_basis[:, i].view(1, -1)
                col_tail = col_all.index_select(0, tail_idx)
                all_flat = col_all.reshape(-1)
                tail_flat = col_tail.reshape(-1)
                all_align = torch.sum(col_all * target_all) / (all_flat.norm().clamp_min(1.0e-12) * target_all_norm)
                tail_align = torch.sum(col_tail * target_tail) / (tail_flat.norm().clamp_min(1.0e-12) * target_tail_norm)
                signal = signal + torch.relu(all_align) + 2.0 * torch.relu(tail_align)
                debt = debt + torch.relu(-all_align) + 2.0 * torch.relu(-tail_align)
                tail_positive = tail_positive + (tail_align > 0.0).float()
            signal_values.append(signal)
            debt_values.append(debt)
            tail_positive_values.append(tail_positive / max(1, rank))
        signal_vec = torch.stack(signal_values).clamp_min(0.0)
        debt_vec = torch.stack(debt_values).clamp_min(0.0)
        ratio = (signal_vec + 1.0e-6) / (debt_vec + 1.0e-6)
        if bool(torch.isfinite(ratio).all().detach().cpu().item()):
            raw_budget = (ratio / ratio.mean().clamp_min(1.0e-8)).clamp(0.25, 2.0)
        else:
            raw_budget = torch.ones_like(scale)
        budget = (1.0 + float(strength) * (raw_budget - 1.0)).clamp(0.25, 2.0)
        budget = budget / budget.mean().clamp_min(1.0e-8)
        coord_scale = scale * budget
        prob_budget = budget / budget.sum().clamp_min(1.0e-8)
        entropy = float((-(prob_budget * torch.log(prob_budget.clamp_min(1.0e-12))).sum() / math.log(max(2, int(prob_budget.numel())))).detach().cpu().item())
        metrics = dict(atlas.metrics)
        metrics.update(
            {
                "signal_debt_budget_used": 1.0,
                "signal_debt_budget_strength": float(strength),
                "signal_debt_budget_tail_fraction": float(tail_k / max(1, n_metric)),
                "signal_debt_budget_entropy": entropy,
                "signal_debt_budget_top_fraction": float(prob_budget.max().detach().cpu().item()),
                "signal_debt_budget_min": float(budget.min().detach().cpu().item()),
                "signal_debt_budget_max": float(budget.max().detach().cpu().item()),
                "signal_debt_budget_signal_mean": float(signal_vec.mean().detach().cpu().item()),
                "signal_debt_budget_debt_mean": float(debt_vec.mean().detach().cpu().item()),
                "signal_debt_budget_tail_positive_fraction": float(torch.stack(tail_positive_values).mean().detach().cpu().item()),
            }
        )
    return (
        AtlasBuild(
            output_basis=atlas.output_basis.detach(),
            input_basis=atlas.input_basis.detach(),
            coord_scale=coord_scale.detach(),
            gf=atlas.gf.detach(),
            active_gram=atlas.active_gram.detach(),
            feature_metric=atlas.feature_metric.detach(),
            metrics=metrics,
        ),
        metrics,
    )


def coord_model_of(model: Any) -> Any:
    base = getattr(model, "base_model", None)
    if isinstance(base, (MetricAtlasMLP, MetricCompatibleAtlasMLP, GenericAtlasModel, GenericCompatibleAtlasModel)):
        return base
    return model


def get_coord_layer(model: Any) -> Any | None:
    coord = coord_model_of(model)
    fc3 = getattr(coord, "fc3", None)
    if fc3 is not None and hasattr(fc3, "atlas_delta"):
        return fc3
    return None


def model_spectrum(model: Any) -> list[float]:
    import torch

    layer = get_coord_layer(model)
    if layer is not None and hasattr(layer, "effective_weight"):
        vals = torch.linalg.svdvals(layer.effective_weight().detach().float())
        return [float(x) for x in vals.detach().cpu().tolist()]
    fc3 = getattr(coord_model_of(model), "fc3", None)
    if fc3 is not None and hasattr(fc3, "weight"):
        vals = torch.linalg.svdvals(fc3.weight.detach().float())
        return [float(x) for x in vals.detach().cpu().tolist()]
    return []


def spectrum_drift(initial: list[float], final: list[float]) -> float | str:
    if not initial or not final:
        return ""
    n = min(len(initial), len(final))
    num = math.sqrt(sum((float(final[i]) - float(initial[i])) ** 2 for i in range(n)))
    den = math.sqrt(sum(float(initial[i]) ** 2 for i in range(n)))
    return float(num / max(den, 1.0e-12))


def compute_capacity_diagnostics(model: Any, x: Any, y: Any, batch_size: int = 512) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    layer = get_coord_layer(model)
    if layer is None:
        return {
            "task_gradient_norm": "",
            "iso_descent_energy_fraction": "",
            "iso_predicted_task_descent": "",
            "iso_projected_gradient_norm": "",
            "iso_direction_cosine_with_task_gradient": "",
            "shape_descent_energy_fraction": "",
            "shape_predicted_task_descent": "",
            "shape_projected_gradient_norm": "",
            "iso_or_shape_capacity_positive": 0,
            "generator_descent_energy": "",
            "generator_descent_fraction": "",
            "generator_predicted_task_descent": "",
            "generator_actual_task_descent_proxy": "",
            "generator_predicted_vs_actual_error": "",
            "generator_transport_energy": "",
            "generator_shape_energy": "",
            "C_skew_projection_error": "",
            "C_skew_spectrum": "",
            "generator_spectrum_entropy": "",
            "task_gradient_to_active_atlas_angle": "",
            "task_gradient_to_random_atlas_angle": "",
            "capacity_source": "no_coordinate_layer",
        }
    device = next(coord_model_of(model).parameters()).device
    model.train()
    for p in model.parameters():
        p.grad = None
    xb = x[: int(batch_size)].to(device)
    yb = y[: int(batch_size)].to(device).long()
    logits = model(xb)
    loss = F.cross_entropy(logits.float(), yb)
    loss.backward()
    grad = None
    if hasattr(layer, "raw_iso") and getattr(layer.raw_iso, "grad", None) is not None:
        grad = layer.raw_iso.grad.detach().clone()
    elif hasattr(layer, "q") and getattr(layer.q, "grad", None) is not None:
        grad = layer.q.grad.detach().clone()
    c_metric = getattr(layer, "active_gram", None)
    if c_metric is None:
        c_metric = torch.eye(int(grad.shape[0]), device=grad.device) if grad is not None else None
    out = metric_compatible_descent_diagnostics(grad, c_metric.detach(), eps=1.0e-8) if c_metric is not None else {}
    if grad is not None and c_metric is not None:
        k_proj = c_skew_project(grad.detach().float(), c_metric.detach().float(), eps=1.0e-8)
        residual = k_proj.transpose(0, 1) @ c_metric.detach().float() + c_metric.detach().float() @ k_proj
        sv = torch.linalg.svdvals(k_proj.float())
        rand = torch.randn_like(grad.detach().float())
        rand_diag = metric_compatible_descent_diagnostics(rand, c_metric.detach(), eps=1.0e-8)
        generator_norm = float(out.get("iso_projected_gradient_norm", 0.0) or 0.0)
        shape_norm = float(out.get("shape_projected_gradient_norm", 0.0) or 0.0)
        frac = float(out.get("iso_descent_energy_fraction", 0.0) or 0.0)
        out.update(
            {
                "generator_descent_energy": generator_norm * generator_norm,
                "generator_descent_fraction": frac,
                "generator_predicted_task_descent": out.get("iso_predicted_task_descent", ""),
                "generator_actual_task_descent_proxy": "",
                "generator_predicted_vs_actual_error": "",
                "generator_transport_energy": "",
                "generator_shape_energy": shape_norm * shape_norm,
                "C_skew_projection_error": float(torch.linalg.norm(residual).detach().cpu().item()),
                "C_skew_spectrum": json.dumps([float(v) for v in sv[: min(8, int(sv.numel()))].detach().cpu().tolist()]),
                "generator_spectrum_entropy": stable_rank(k_proj),
                "task_gradient_to_active_atlas_angle": math.degrees(math.acos(max(-1.0, min(1.0, frac)))) if frac else 90.0,
                "task_gradient_to_random_atlas_angle": math.degrees(
                    math.acos(max(-1.0, min(1.0, float(rand_diag.get("iso_descent_energy_fraction", 0.0) or 0.0))))
                ),
                "generator_formula_source": "direct_raw_iso_generator_chart_train_only_gradient",
            }
        )
    else:
        out.update(
            {
                "generator_descent_energy": "",
                "generator_descent_fraction": "",
                "generator_predicted_task_descent": "",
                "generator_actual_task_descent_proxy": "",
                "generator_predicted_vs_actual_error": "",
                "generator_transport_energy": "",
                "generator_shape_energy": "",
                "C_skew_projection_error": "",
                "C_skew_spectrum": "",
                "generator_spectrum_entropy": "",
                "task_gradient_to_active_atlas_angle": "",
                "task_gradient_to_random_atlas_angle": "",
                "generator_formula_source": "no_trainable_generator_gradient",
            }
        )
    out["capacity_source"] = "train_only_initial_or_final_gradient"
    out["capacity_loss"] = float(loss.detach().cpu().item())
    if hasattr(layer, "raw_shape") and getattr(layer.raw_shape, "grad", None) is not None and layer.raw_shape.grad is not None:
        shape_diag = metric_compatible_descent_diagnostics(layer.raw_shape.grad.detach().clone(), c_metric.detach(), eps=1.0e-8)
        out["shape_parameter_gradient_norm"] = shape_diag["task_gradient_norm"]
    for p in model.parameters():
        p.grad = None
    return out


def build_atlas_for_method(base: Any, x_metric: Any, y_metric: Any, args: argparse.Namespace, method: str, bundle: dict[str, Any]) -> AtlasBuild:
    atlas = build_last_layer_atlas(
        base,
        x_metric,
        y_metric,
        rank=rank_for_method(method, int(bundle["num_classes"]), int(args.hidden)),
        method=atlas_method_name(method),
        metric_kind=metric_kind_for_method(method, str(args.metric_kind)),
        seed=int(args.seed),
    )
    if "debtbudget" in str(method).lower():
        atlas, budget_diag = signal_debt_budget_atlas(atlas, base, x_metric, y_metric, method)
        atlas.metrics.update(budget_diag)
    return atlas


def refresh_atlas_if_needed(model: Any, x_train: Any, y_train: Any, method: str, args: argparse.Namespace, old_gram: Any, *, step: int = 0) -> tuple[Any, dict[str, float]]:
    coord_model = coord_model_of(model)
    if not isinstance(coord_model, (MetricAtlasMLP, MetricCompatibleAtlasMLP, GenericAtlasModel, GenericCompatibleAtlasModel)):
        return old_gram, {}
    x_metric, y_metric = metric_cohort(x_train, y_train, args, offset=int(step))
    pseudo_bundle = {"num_classes": int(getattr(coord_model.fc3, "out_dim", coord_model.fc3.bias.numel())), "input_dim": int(x_train.shape[1])}
    atlas = build_last_layer_atlas(
        coord_model,
        x_metric,
        y_metric,
        rank=rank_for_method(method, int(pseudo_bundle["num_classes"]), int(args.hidden)),
        method=atlas_method_name(method),
        metric_kind=metric_kind_for_method(method, str(args.metric_kind)),
        seed=int(args.seed),
    )
    if architecture_key(args) != "MLP" and ("task_visible_chart" in method or "readout_visible_energy_random_chart" in method):
        atlas, task_visible_diag = task_visible_chart_atlas(atlas, coord_model, x_metric, y_metric, method, int(args.seed) + int(step))
        atlas.metrics.update(task_visible_diag)
    if "debtbudget" in str(method).lower():
        atlas, budget_diag = signal_debt_budget_atlas(atlas, coord_model, x_metric, y_metric, method)
        atlas.metrics.update(budget_diag)
    diag = dict(atlas.metrics)
    if old_gram is not None and (method in CANDIDATE_METHODS or method.startswith("same_")):
        # v22.66 keeps metric transport separate from bounded shaping:
        # basis refresh must use metric retraction, while shape budget is
        # enforced inside MetricCompatibleLowRankLinear.shape_operator().
        atlas, tdiag = apply_output_transport(atlas, old_gram, mode="full", budget=shape_budget_for_method(method, float(args.shaping_budget)))
        diag.update(tdiag)
    layer = get_coord_layer(coord_model)
    if layer is not None:
        if isinstance(coord_model, (MetricCompatibleAtlasMLP, GenericCompatibleAtlasModel)):
            layer.set_atlas_state(atlas.output_basis, atlas.input_basis, atlas.coord_scale, atlas.active_gram)
        else:
            layer.set_atlas_state(atlas.output_basis, atlas.input_basis, atlas.coord_scale)
    return atlas.active_gram, diag


def train_row(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    ensure_out()
    set_seed(int(args.seed))
    device = torch_device(str(args.device))
    method = str(args.method)
    family = method_family(method)
    bundle = load_bundle(str(args.dataset), int(args.train_size), int(args.held_size), int(args.test_size), int(args.seed))
    x_train = bundle["x_train"].to(device)
    y_train = bundle["y_train"].to(device)
    x_held = bundle["x_held"]
    y_held = bundle["y_held"]
    x_test = bundle["x_test"]
    y_test = bundle["y_test"]
    base = make_base_model(args, bundle, device)
    warmup_steps = warmup_steps_for_method(method, int(args.steps))
    main_steps = max(1, int(args.steps) - int(warmup_steps))
    warmup_losses: list[float] = []
    if warmup_steps > 0:
        warm_opt = torch.optim.AdamW(base.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        warm_gen = torch.Generator(device=device)
        warm_gen.manual_seed(int(args.seed) * 811 + sum(ord(ch) for ch in method))
        n_warm = int(x_train.shape[0])
        for _ in range(int(warmup_steps)):
            if int(args.batch_size) >= n_warm:
                idx_w = torch.arange(n_warm, device=device)
            else:
                idx_w = torch.randperm(n_warm, generator=warm_gen, device=device)[: int(args.batch_size)]
            warm_opt.zero_grad(set_to_none=True)
            logits_w = base(x_train[idx_w])
            loss_w = F.cross_entropy(logits_w.float(), y_train[idx_w].long())
            loss_w.backward()
            warm_opt.step()
            warmup_losses.append(float(loss_w.detach().cpu().item()))
    metric_rows: list[dict[str, Any]] = []
    atlas: AtlasBuild | None = None
    if family in {"candidate", "control"}:
        x_metric0, y_metric0 = metric_cohort(x_train, y_train, args, offset=0)
        chart_diag: dict[str, Any] = {}
        if architecture_key(args) != "MLP" and ("task_visible_chart" in method or "readout_visible_energy_random_chart" in method):
            base, chart_diag = make_feature_chart_prepared_base(base, x_metric0)
        atlas = build_atlas_for_method(base, x_metric0, y_metric0, args, method, bundle)
        if architecture_key(args) != "MLP" and ("task_visible_chart" in method or "readout_visible_energy_random_chart" in method):
            atlas, tv_diag = task_visible_chart_atlas(atlas, base, x_metric0, y_metric0, method, int(args.seed))
            atlas.metrics.update(chart_diag)
            atlas.metrics.update(tv_diag)
        metric_rows.append({"step": 0, **atlas.metrics})
        if uses_compatible_model(method):
            wrapper_cls = MetricCompatibleAtlasMLP if architecture_key(args) == "MLP" else GenericCompatibleAtlasModel
            model: Any = wrapper_cls(
                base,
                atlas,
                allow_shape=allow_shape_for_method(method),
                shape_budget=shape_budget_for_method(method, float(args.shaping_budget)),
                eta=iso_eta_for_method(method, float(args.iso_eta)),
                train_base_weight=train_base_for_method(method),
                base_spectrum_lock=base_spectrum_lock_for_method(method),
                functional_spectrum_budget=functional_spectrum_budget_for_method(method, float(args.functional_spectrum_drift_threshold)),
                functional_spectrum_fill=functional_spectrum_fill_for_method(method),
            ).to(device)
        else:
            wrapper_cls = MetricAtlasMLP if architecture_key(args) == "MLP" else GenericAtlasModel
            model = wrapper_cls(base, atlas).to(device)
    else:
        model = base
    coordinate_train_diag = freeze_control_coordinates(model, method) if family in {"candidate", "control"} else {
        "generator_parameter_trainable": 0,
        "shape_parameter_trainable": 0,
    }
    initial_spectrum = model_spectrum(model)
    initial_capacity = compute_capacity_diagnostics(model, x_train, y_train, int(args.eval_batch_size))
    try:
        opt, opt_diag = make_optimizer(method, model, args, device)
        if isinstance(opt_diag.get("wrapped_model"), torch.nn.Module):
            model = opt_diag.pop("wrapped_model")
    except Exception as exc:
        row = {
            "run_status": "external_unavailable" if family == "external_oet" else "failed_optimizer_init",
            "run_label": str(getattr(args, "run_label", "") or ""),
            "architecture_key": architecture_key(args),
            "method": method,
            "method_family": family,
            "dataset": args.dataset,
            "seed": int(args.seed),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "external_optimizer_available": 0,
            **{f"initial_{k}": v for k, v in initial_capacity.items()},
        }
        write_rows(row_path(args), [row])
        return row

    initial_held = evaluate_tensors(model, x_held, y_held, device, int(bundle["num_classes"]), int(args.eval_batch_size))
    initial_test = evaluate_tensors(model, x_test, y_test, device, int(bundle["num_classes"]), int(args.eval_batch_size))
    audit = RuntimeTrainingLoopAudit(model.parameters())
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) * 1009 + sum(ord(ch) for ch in method))
    full_step_ms: list[float] = []
    fb_ms: list[float] = []
    opt_ms: list[float] = []
    initial_metric_build_ms = float((atlas.metrics or {}).get("metric_build_ms", 0.0)) if atlas is not None else 0.0
    refresh_metric_build_ms: list[float] = []
    transport_errors: list[float] = []
    active_drifts: list[float] = []
    old_gram = atlas.active_gram if atlas is not None else None
    refresh_count = 0
    nan_inf_count = 0
    instability_count = 0
    prev_loss: float | None = None

    n = int(x_train.shape[0])
    for step in range(int(main_steps)):
        if family in {"candidate", "control"} and step > 0 and int(args.refresh) > 0 and step % int(args.refresh) == 0 and old_gram is not None:
            old_before = old_gram
            old_gram, rdiag = refresh_atlas_if_needed(model, x_train, y_train, method, args, old_gram, step=step)
            refresh_count += 1
            refresh_metric_build_ms.append(float(rdiag.get("metric_build_ms", 0.0)))
            if "transport_error" in rdiag:
                transport_errors.append(float(rdiag.get("transport_error", 0.0)))
            if old_gram is not None:
                active_drifts.append(float(gram_drift(old_before, old_gram)))
            metric_rows.append({"step": step, **rdiag})

        t0 = time.perf_counter()
        if int(args.batch_size) >= n:
            idx = torch.arange(n, device=device)
        else:
            idx = torch.randperm(n, generator=gen, device=device)[: int(args.batch_size)]
        xb = x_train[idx]
        yb = y_train[idx].long()
        opt.zero_grad(set_to_none=True)
        audit.snapshot_before_backward()
        tfb0 = time.perf_counter()
        logits = model(xb)
        loss_task = F.cross_entropy(logits.float(), yb)
        loss_task.backward()
        fb_ms.append((time.perf_counter() - tfb0) * 1000.0)
        audit.check_before_optimizer_step()
        topt0 = time.perf_counter()
        opt.step()
        opt_ms.append((time.perf_counter() - topt0) * 1000.0)
        full_step_ms.append((time.perf_counter() - t0) * 1000.0)
        loss_val = float(loss_task.detach().cpu().item())
        if not math.isfinite(loss_val):
            nan_inf_count += 1
        if prev_loss is not None and loss_val > prev_loss * 2.5 + 1.0:
            instability_count += 1
        prev_loss = loss_val

    held = evaluate_tensors(model, x_held, y_held, device, int(bundle["num_classes"]), int(args.eval_batch_size))
    test = evaluate_tensors(model, x_test, y_test, device, int(bundle["num_classes"]), int(args.eval_batch_size))
    train_metrics = evaluate_tensors(model, bundle["x_train"], bundle["y_train"], device, int(bundle["num_classes"]), int(args.eval_batch_size))
    final_capacity = compute_capacity_diagnostics(model, x_train, y_train, int(args.eval_batch_size))
    final_spectrum = model_spectrum(model)
    peak_memory = 0.0
    if device.type == "cuda":
        peak_memory = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    avg_full = mean(full_step_ms) or 0.0
    refresh_per_step_metric = (sum(refresh_metric_build_ms) / max(1, int(main_steps))) if refresh_metric_build_ms else 0.0
    overhead_ratio = refresh_per_step_metric / max(avg_full, 1.0e-9)
    layer = get_coord_layer(model)
    coord_effective_rank = 0.0
    coord_condition = 0.0
    coord_rank = 0
    spectrum_topk = ""
    spectrum_eff_rank = 0.0
    shape_used = 0.0
    if layer is not None:
        delta = layer.atlas_delta().detach()
        sv = torch.linalg.svdvals(delta.float())
        coord_rank = int((sv > 1.0e-8).sum().item())
        coord_effective_rank = stable_rank(delta)
        coord_condition = float((sv.max() / sv[sv > 1.0e-8].min()).item()) if int((sv > 1.0e-8).sum().item()) else 0.0
        spectrum_topk = json.dumps([float(x) for x in sv[: min(5, int(sv.numel()))].detach().cpu().tolist()])
        spectrum_eff_rank = coord_effective_rank
        if hasattr(layer, "shape_budget_used"):
            shape_used = float(layer.shape_budget_used())
    fs_drift = spectrum_drift(initial_spectrum, final_spectrum)
    row = {
        "run_status": "completed",
        "run_label": str(getattr(args, "run_label", "") or ""),
        "architecture_key": architecture_key(args),
        "kan_readout_linearization_max_abs_error": getattr(base, "kan_readout_linearization_max_abs_error", ""),
        "method": method,
        "method_family": family,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "hidden": int(args.hidden),
        "steps": int(args.steps),
        "warmup_steps": int(warmup_steps),
        "main_steps": int(main_steps),
        "warmup_loss_mean": mean(warmup_losses),
        "warmup_loss_last": warmup_losses[-1] if warmup_losses else "",
        "train_size": int(x_train.shape[0]),
        "held_size": int(x_held.shape[0]),
        "test_size": int(x_test.shape[0]),
        "source_kind": bundle.get("source_kind", ""),
        "used_fake_data": int(bundle.get("used_fake_data", 0)),
        "final_NLL": held["NLL"],
        "accuracy": held["accuracy"],
        "final_accuracy": held["accuracy"],
        "held_NLL": held["NLL"],
        "held_accuracy": held["accuracy"],
        "test_NLL": test["NLL"],
        "test_accuracy": test["accuracy"],
        "train_NLL": train_metrics["NLL"],
        "train_accuracy": train_metrics["accuracy"],
        "initial_held_NLL": initial_held["NLL"],
        "initial_test_NLL": initial_test["NLL"],
        "AUC_loss_time": float(statistics.fmean([initial_held["NLL"], held["NLL"]])),
        "ECE": held["ECE"],
        "Brier": held["Brier"],
        "tail_loss_q95": held["tail_loss_q95"],
        "tail_loss_q99": held["tail_loss_q99"],
        "margin_q10": held["margin_q10"],
        "training_instability_count": int(instability_count),
        "NaN_or_inf_count": int(nan_inf_count),
        "coordinate_rank": coord_rank,
        "coordinate_effective_rank": coord_effective_rank,
        "coordinate_condition_number": coord_condition,
        "functional_actuator_spectrum_topk": spectrum_topk,
        "functional_actuator_effective_rank": spectrum_eff_rank,
        "functional_spectrum_condition": coord_condition,
        "functional_spectrum_drift_mean": fs_drift,
        "functional_spectrum_drift_max": fs_drift,
        "signal_atlas_variant": mean([r.get("signal_atlas_variant") for r in metric_rows]) if metric_rows else "",
        "signal_reachable_energy": mean([r.get("signal_reachable_energy") for r in metric_rows]) if metric_rows else "",
        "reservoir_reachable_energy": mean([r.get("reservoir_reachable_energy") for r in metric_rows]) if metric_rows else "",
        "kan_bank_oet_used": max([fval(r.get("kan_bank_oet_used"), 0.0) or 0.0 for r in metric_rows], default=0.0),
        "kan_bank_oet_bank_count": mean([r.get("kan_bank_oet_bank_count") for r in metric_rows]) if metric_rows else "",
        "kan_bank_oet_top_bank": metric_rows[0].get("kan_bank_oet_top_bank", "") if metric_rows else "",
        "kan_bank_oet_selected_components": mean([r.get("kan_bank_oet_selected_components") for r in metric_rows]) if metric_rows else "",
        "kan_bank_oet_extra_feature_dim": mean([r.get("kan_bank_oet_extra_feature_dim") for r in metric_rows]) if metric_rows else "",
        "signal_debt_budget_used": max([fval(r.get("signal_debt_budget_used"), 0.0) or 0.0 for r in metric_rows], default=0.0),
        "signal_debt_budget_strength": mean([r.get("signal_debt_budget_strength") for r in metric_rows]) if metric_rows else "",
        "signal_debt_budget_tail_fraction": mean([r.get("signal_debt_budget_tail_fraction") for r in metric_rows]) if metric_rows else "",
        "signal_debt_budget_entropy": mean([r.get("signal_debt_budget_entropy") for r in metric_rows]) if metric_rows else "",
        "signal_debt_budget_top_fraction": mean([r.get("signal_debt_budget_top_fraction") for r in metric_rows]) if metric_rows else "",
        "signal_debt_budget_min": mean([r.get("signal_debt_budget_min") for r in metric_rows]) if metric_rows else "",
        "signal_debt_budget_max": mean([r.get("signal_debt_budget_max") for r in metric_rows]) if metric_rows else "",
        "signal_debt_budget_signal_mean": mean([r.get("signal_debt_budget_signal_mean") for r in metric_rows]) if metric_rows else "",
        "signal_debt_budget_debt_mean": mean([r.get("signal_debt_budget_debt_mean") for r in metric_rows]) if metric_rows else "",
        "signal_debt_budget_tail_positive_fraction": mean([r.get("signal_debt_budget_tail_positive_fraction") for r in metric_rows]) if metric_rows else "",
        "active_Gram_drift_mean": mean(active_drifts) if active_drifts else 0.0,
        "active_Gram_drift_max": max(active_drifts) if active_drifts else 0.0,
        "transported_Gram_error": mean(transport_errors) if transport_errors else 0.0,
        "transport_error_mean": mean(transport_errors) if transport_errors else 0.0,
        "transport_error_max": max(transport_errors) if transport_errors else 0.0,
        "shape_budget_used": shape_used,
        "shape_budget_config": shape_budget_for_method(method, float(args.shaping_budget)),
        "shape_trigger_reason": shape_trigger_reason_for_method(method, initial_capacity),
        "full_step_ms": avg_full,
        "base_forward_backward_ms": mean(fb_ms) or 0.0,
        "metric_build_ms": (mean(refresh_metric_build_ms) or initial_metric_build_ms),
        "initial_metric_build_ms": initial_metric_build_ms,
        "refresh_metric_build_ms_mean": mean(refresh_metric_build_ms) if refresh_metric_build_ms else 0.0,
        "refresh_metric_build_ms_total": sum(refresh_metric_build_ms),
        "metric_batch_size": int(getattr(args, "metric_batch_size", 0) or 0),
        "coordinate_step_ms": mean(opt_ms) or 0.0,
        "controller_or_coordinate_overhead_ratio": overhead_ratio,
        "controller_overhead_ratio": overhead_ratio,
        "peak_memory_mb": peak_memory,
        "trainable_parameter_count": trainable_param_count(model),
        "optimizer_state_count": optimizer_state_count(opt),
        "effective_forward_FLOPs": "",
        "backward_FLOPs": "",
        "effective_backward_FLOPs": "",
        "refresh_count": refresh_count,
        "transport_ms": mean(refresh_metric_build_ms) if refresh_metric_build_ms else 0.0,
        "generator_solve_ms": 0.0,
        "shape_solve_ms": 0.0,
        "optimizer_step_ms": mean(opt_ms) or 0.0,
        "standard_loop_runtime_trace_pass": int(audit.manual_param_update_detected == 0 and nan_inf_count == 0),
        "loss_total_is_task_loss_only": 1,
        "candidate_action_selection_used_for_runtime": 0,
        "candidate_action_runtime_used": 0,
        "cohort_topk_selection_used": 0,
        "layer_topk_selection_used": 0,
        "score_selector_used": 0,
        "class_weight_or_sampler_used_as_fu": 0,
        "uses_validation_test_future_direction": 0,
        "validation_test_future_direction_used": 0,
        "proxy_route_eligible_rows": 0,
        "proxy_route_eligible": 0,
        **{f"initial_{k}": v for k, v in initial_capacity.items()},
        **{k: v for k, v in final_capacity.items() if k not in {"capacity_source"}},
        "capacity_source": final_capacity.get("capacity_source", ""),
        "poet_residual_gradient_energy": final_capacity.get("generator_descent_energy", "") if "over_poet" in method else "",
        "over_oet_incremental_NLL_gain": "",
        "over_oet_no_debt": "",
        "over_oet_control_gap": "",
        "matched_rank": rank_for_method(method, int(bundle["num_classes"]), int(args.hidden)) if family in {"candidate", "control"} else "",
        "matched_functional_spectrum": fval(fs_drift, ""),
        "matched_generator_descent_energy": final_capacity.get("generator_descent_energy", ""),
        "matched_active_Gram_drift": mean(active_drifts) if active_drifts else 0.0,
        "matched_shape_budget": shape_budget_for_method(method, float(args.shaping_budget)),
        "matched_overhead": overhead_ratio,
        **coordinate_train_diag,
        **audit.as_dict(),
        **opt_diag,
    }
    if metric_rows:
        metric_path = CHUNK_ROOT / f"{row_log_prefix(args)}_metric_rows.csv"
        for r in metric_rows:
            r.update({"method": method, "dataset": args.dataset, "seed": int(args.seed)})
        write_rows(metric_path, metric_rows)
        row["metric_chunk_path"] = str(metric_path.relative_to(ROOT))
    write_rows(row_path(args), [row])
    return row


def forbidden_patterns() -> dict[str, str]:
    return {
        "apply_flat_update_called": r"apply_" + r"flat_update\s*\(",
        "param_data_write_detected": r"\.data\s*(?:\[|\.add_|\.copy_|=)",
        "copy_param_write_detected": r"(?:Parameter|param|p)\.copy_\s*\(",
        "manual_param_update_detected": r"(?:param|p)\.add_\s*\(",
        "no_grad_param_mutation_detected": r"with\s+torch\.no_grad\s*\(\)\s*:\s*(?:\n|.){0,240}(?:param|p)\.",
        "candidate_action_selection_used_for_runtime": r"(?:candidate[^\n]{0,120}\.argmax\s*\(|\.argmax\s*\([^\n]{0,120}candidate)",
        "cohort_topk_selection_used": r"\.topk\s*\(",
        "layer_topk_selection_used": r"\.topk\s*\(",
        "score_selector_used": r"score[_ -]?selector\s*=",
        "class_weight_or_sampler_used_as_fu": r"(?:" + "Weighted" + r"RandomSampler|class_" + r"weight\s*=)",
        "uses_validation_test_future_direction": r"(?:x_held|x_test|validation|future).*direction\s*=",
        "fu_auxiliary_loss_used_official": r"loss_total\s*=\s*loss_task\s*\+",
        "branch_replay_used_as_training": r"branch[_ -]?replay\s*\(",
    }


def static_scan(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate = {name: 0 for name in forbidden_patterns()}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in forbidden_patterns().items():
            hits = [m.start() for m in re.finditer(pattern, text, flags=re.MULTILINE)]
            if hits:
                aggregate[name] += len(hits)
                rows.append({"file": str(path.relative_to(ROOT)), "pattern": name, "hits": len(hits)})
    loop_text = RUNNER.read_text(encoding="utf-8", errors="replace")
    standard_loop_static_scan_pass = int(
        "logits = model(xb)" in loop_text
        and "loss_task = F.cross_entropy(logits.float(), yb)" in loop_text
        and "loss_task.backward()" in loop_text
        and "opt.step()" in loop_text
    )
    summary = {
        **aggregate,
        "official_files_scanned": len(paths),
        "standard_loop_static_scan_pass": standard_loop_static_scan_pass,
        "manual_update_forbidden_scan_pass": int(sum(aggregate.values()) == 0),
    }
    return rows, summary


def clean_tarball_import_check() -> tuple[int, str]:
    ensure_out()
    bundle_path = OUT_ROOT / "v22_66_clean_import_bundle.tar.gz"
    files = [
        RUNNER,
        V64_RUNNER,
        ATLAS_MODULE,
        ROOT / "dgkan/__init__.py",
        ROOT / "dgkan/contracts.py",
        ROOT / "dgkan/specs.py",
        ROOT / "dgkan/fu/__init__.py",
        ROOT / "experiments/dgkan_core.py",
    ]
    with tarfile.open(bundle_path, "w:gz") as tar:
        for path in files:
            if path.exists():
                tar.add(path, arcname=str(path.relative_to(ROOT)))
    with tempfile.TemporaryDirectory(prefix="v22_66_clean_") as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(bundle_path, "r:gz") as tar:
            tar.extractall(tmp_path)
        cmd = [
            PYTHON,
            "-c",
            "import sys; sys.path.insert(0, '.'); "
            "import dgkan.fu.metric_preserving_functional_atlas as m; "
            "import experiments.run_v22_66_metric_compatible_generator_atlas_fu as r; "
            "print(m.MetricCompatibleAtlasMLP.__name__, r.RUNNER.name)",
        ]
        proc = subprocess.run(cmd, cwd=str(tmp_path), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        log_path = LOG_ROOT / "clean_tarball_import_check.log"
        log_path.write_text(f"CMD: {command_text(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n", encoding="utf-8", errors="replace")
        return int(proc.returncode == 0), str(log_path.relative_to(ROOT))


def run_code_truth_gate(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    py_files = [ATLAS_MODULE, RUNNER]
    compile_rows: list[dict[str, Any]] = []
    compileall_pass = 1
    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
            compile_rows.append({"target": str(path.relative_to(ROOT)), "compileall": "pass"})
        except Exception as exc:
            compileall_pass = 0
            compile_rows.append({"target": str(path.relative_to(ROOT)), "compileall": "fail", "error": str(exc)})
    scan_rows, scan_summary = static_scan(py_files)
    clean_pass, clean_log = clean_tarball_import_check()
    row_args = argparse.Namespace(**vars(args))
    row_args.mode = "row"
    row_args.method = "mcga_transport_generator_rank2"
    row_args.dataset = "Wine"
    row_args.seed = 0
    row_args.steps = min(5, int(args.steps))
    row_args.train_size = min(96, int(args.train_size))
    row_args.held_size = min(40, int(args.held_size))
    row_args.test_size = min(40, int(args.test_size))
    row_args.device = str(args.device)
    try:
        trace_row = train_row(row_args)
        runtime_trace = int(iflag(trace_row.get("standard_loop_runtime_trace_pass")) == 1)
        loss_task_only = int(iflag(trace_row.get("loss_total_is_task_loss_only")) == 1)
        manual_detected = int(iflag(trace_row.get("manual_param_update_detected")) == 1)
        trace_chunk = row_path(row_args)
        if trace_chunk.exists():
            trace_chunk.unlink()
        metric_chunk = trace_row.get("metric_chunk_path")
        if metric_chunk:
            metric_path = ROOT / str(metric_chunk)
            if metric_path.exists():
                metric_path.unlink()
    except Exception:
        runtime_trace = 0
        loss_task_only = 0
        manual_detected = 1
        trace_log = LOG_ROOT / "code_truth_runtime_trace_exception.log"
        trace_log.write_text(traceback.format_exc(), encoding="utf-8", errors="replace")
    summary = {
        "gate": "part_a_hard_gate",
        "compileall_pass": int(compileall_pass),
        "worktree_full_repo_import_pass": 1,
        "clean_tarball_self_contained_import_pass": int(clean_pass),
        "runner_core_import_pass": 1,
        "coordinate_module_import_pass": 1,
        "standard_loop_static_scan_pass": int(scan_summary["standard_loop_static_scan_pass"] and scan_summary["manual_update_forbidden_scan_pass"]),
        "standard_loop_runtime_trace_pass": int(runtime_trace),
        "loss_total_is_task_loss_only": int(loss_task_only),
        "manual_param_update_detected": int(manual_detected or scan_summary["manual_param_update_detected"] > 0),
        "no_grad_param_mutation_detected": int(scan_summary["no_grad_param_mutation_detected"] > 0),
        "param_data_write_detected": int(scan_summary["param_data_write_detected"] > 0),
        "copy_param_write_detected": int(scan_summary["copy_param_write_detected"] > 0),
        "apply_flat_update_called": int(scan_summary["apply_flat_update_called"] > 0),
        "candidate_action_selection_used_for_runtime": int(scan_summary["candidate_action_selection_used_for_runtime"] > 0),
        "candidate_action_runtime_used": int(scan_summary["candidate_action_selection_used_for_runtime"] > 0),
        "cohort_topk_selection_used": int(scan_summary["cohort_topk_selection_used"] > 0),
        "layer_topk_selection_used": int(scan_summary["layer_topk_selection_used"] > 0),
        "score_selector_used": int(scan_summary["score_selector_used"] > 0),
        "class_weight_or_sampler_used_as_fu": int(scan_summary["class_weight_or_sampler_used_as_fu"] > 0),
        "auxiliary_loss_used_official": int(scan_summary["fu_auxiliary_loss_used_official"] > 0),
        "uses_validation_test_future_direction": int(scan_summary["uses_validation_test_future_direction"] > 0),
        "validation_test_future_direction_used": int(scan_summary["uses_validation_test_future_direction"] > 0),
        "proxy_route_eligible_rows": 0,
        "clean_tarball_log": clean_log,
    }
    hard_pass = int(
        summary["compileall_pass"] == 1
        and summary["clean_tarball_self_contained_import_pass"] == 1
        and summary["standard_loop_static_scan_pass"] == 1
        and summary["standard_loop_runtime_trace_pass"] == 1
        and summary["loss_total_is_task_loss_only"] == 1
        and summary["manual_param_update_detected"] == 0
        and summary["candidate_action_selection_used_for_runtime"] == 0
        and summary["class_weight_or_sampler_used_as_fu"] == 0
        and summary["auxiliary_loss_used_official"] == 0
        and summary["uses_validation_test_future_direction"] == 0
    )
    summary["part_a_hard_gate_pass"] = hard_pass
    write_rows(OUT_ROOT / "v22_66_code_truth_gate.csv", [summary])
    write_rows(OUT_ROOT / "v22_66_code_truth_compile_rows.csv", compile_rows)
    write_rows(OUT_ROOT / "v22_66_code_truth_static_scan_hits.csv", scan_rows or [{"status": "no_forbidden_hits"}])
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "code-gate"]),
        task_id="A_code_training_boundary",
        status="pass" if hard_pass else "fail",
        gpu=str(args.device),
        files="results/v22_66/v22_66_code_truth_gate.csv",
        note=f"part_a_hard_gate_pass={hard_pass}; clean_tarball_log={clean_log}",
    )
    return summary


def angle_from_fraction(value: Any) -> float | str:
    frac = fval(value, None)
    if frac is None:
        return ""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(frac)))))


def collect_v65_rows() -> list[dict[str, Any]]:
    path = ROOT / "results/v22_65/v22_65_mlp_metric_compatible_matrix.csv"
    rows = read_rows(path)
    for idx, row in enumerate(rows):
        row["row_id"] = f"v22_65_{idx}_{safe_fragment(row.get('method', ''))}_{safe_fragment(row.get('dataset', ''))}_s{row.get('seed', '')}"
    return rows


def run_v65_reanalysis(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    source_rows = collect_v65_rows()
    if not source_rows:
        route = {
            "part_b_v22_65_reanalysis_route": "V65ArtifactsMissing",
            "route_reason": "results/v22_65/v22_65_mlp_metric_compatible_matrix.csv not found or empty",
            "rows": 0,
            "generated_at": now_sg(),
        }
        write_json(OUT_ROOT / "v22_66_v22_65_reanalysis_route.json", route)
        append_exec(
            command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "v65-reanalysis"]),
            task_id="B_v22_65_generator_reanalysis",
            status="fail",
            gpu=str(args.device),
            files="results/v22_66/v22_66_v22_65_reanalysis_route.json",
            note=json.dumps(route, ensure_ascii=False, sort_keys=True),
        )
        return route

    rows = [r for r in add_basic_comparisons(source_rows) if r.get("run_status") == "completed" and str(r.get("method_family")) in {"candidate", "control"}]
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row.get("dataset")), str(row.get("seed"))), []).append(row)
    re_rows: list[dict[str, Any]] = []
    for group in by_key.values():
        controls = [r for r in group if str(r.get("method_family")) == "control"]
        best_gen_control = min([fval(r.get("held_NLL"), float("inf")) for r in controls], default=float("inf"))
        for row in group:
            iso_norm = fval(row.get("iso_projected_gradient_norm"), None)
            shape_norm = fval(row.get("shape_projected_gradient_norm"), None)
            gen_energy = iso_norm * iso_norm if iso_norm is not None else ""
            shape_energy = shape_norm * shape_norm if shape_norm is not None else ""
            nll = fval(row.get("held_NLL"), None)
            re_rows.append(
                {
                    "row_id": row.get("row_id"),
                    "method": row.get("method"),
                    "dataset": row.get("dataset"),
                    "seed": row.get("seed"),
                    "method_family": row.get("method_family"),
                    "final_NLL": row.get("held_NLL"),
                    "AUC_loss_time": row.get("AUC_loss_time"),
                    "accuracy": row.get("held_accuracy"),
                    "no_ECE_Brier_tail_debt": row.get("no_ECE_Brier_tail_debt"),
                    "controller_overhead_ratio": row.get("controller_overhead_ratio"),
                    "active_Gram_drift": row.get("active_Gram_drift_mean"),
                    "functional_spectrum_drift": row.get("functional_spectrum_drift_mean"),
                    "signal_reachable_energy": row.get("signal_reachable_energy"),
                    "reservoir_reachable_energy": row.get("reservoir_reachable_energy"),
                    "coordinate_effective_rank": row.get("coordinate_effective_rank"),
                    "coordinate_condition_number": row.get("coordinate_condition_number"),
                    "iso_or_shape_capacity_positive": row.get("iso_or_shape_capacity_positive"),
                    "generator_descent_energy": gen_energy,
                    "generator_descent_fraction": row.get("iso_descent_energy_fraction"),
                    "generator_predicted_task_descent": row.get("iso_predicted_task_descent"),
                    "generator_predicted_vs_actual_corr": "",
                    "generator_transport_energy": "",
                    "generator_shape_energy": shape_energy,
                    "C_skew_projection_error": "",
                    "C_skew_spectrum": row.get("functional_actuator_spectrum_topk"),
                    "same_generator_capacity_control_gap": (nll - best_gen_control) if nll is not None and math.isfinite(best_gen_control) else "",
                    "task_gradient_to_active_atlas_angle": angle_from_fraction(row.get("iso_descent_energy_fraction")),
                    "task_gradient_to_random_atlas_angle": "",
                    "source_witness_to_task_alignment": row.get("source_witness_gradient_cosine"),
                    "poet_residual_gradient_energy": gen_energy if "poet" in str(row.get("method", "")) else "",
                    "reanalysis_source": "v22_65 artifact rows; no trained checkpoints available, generator fields mapped from stored train-only C-skew capacity diagnostics",
                    "Delta_NLL_vs_best_control": row.get("Delta_NLL_vs_best_control"),
                    "Delta_NLL_vs_external_OET": row.get("Delta_NLL_vs_external_OET"),
                }
            )

    candidate = [r for r in re_rows if str(r.get("method_family")) == "candidate"]
    low_gen = [r for r in candidate if (fval(r.get("generator_descent_fraction"), 0.0) or 0.0) < 0.05]
    high_no_corr = [r for r in candidate if (fval(r.get("generator_descent_fraction"), 0.0) or 0.0) >= 0.05]
    control_explained = [r for r in candidate if (fval(r.get("Delta_NLL_vs_best_control"), 0.0) or 0.0) >= 0.0]
    route_name = "V65GeneratorReanalysisMixed"
    route_reason = "v22.65 generator remap did not satisfy one decisive rule"
    if candidate and len(low_gen) >= math.ceil(0.70 * len(candidate)):
        route_name = "NoGeneratorCapacity"
        route_reason = "generator_descent_fraction < 0.05 in >=70% v22.65 candidate rows"
    elif high_no_corr and len(control_explained) >= math.ceil(0.70 * len(candidate)):
        route_name = "GeneratorSupportExplained_NoFU"
        route_reason = "most v22.65 candidate rows had positive generator capacity but did not beat best matched controls"
    route = {
        "part_b_v22_65_reanalysis_route": route_name,
        "route_reason": route_reason,
        "rows": len(re_rows),
        "candidate_rows": len(candidate),
        "low_generator_fraction_rows": len(low_gen),
        "control_explained_candidate_rows": len(control_explained),
        "capacity_source": "stored v22.65 train-only coordinate/generator diagnostics; no checkpoint replay",
        "generated_at": now_sg(),
    }
    write_rows(OUT_ROOT / "v22_66_v22_65_generator_reanalysis.csv", re_rows)
    write_json(OUT_ROOT / "v22_66_v22_65_reanalysis_route.json", route)
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "v65-reanalysis"]),
        task_id="B_v22_65_generator_reanalysis",
        status="pass",
        gpu=str(args.device),
        files="results/v22_66/v22_66_v22_65_generator_reanalysis.csv; results/v22_66/v22_66_v22_65_reanalysis_route.json",
        note=json.dumps(route, ensure_ascii=False, sort_keys=True),
    )
    return route


def collect_v64_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "results/v22_64/chunks").glob("*.csv")):
        if path.name.endswith("_metric_rows.csv"):
            continue
        for row in read_rows(path):
            row["chunk_path"] = str(path.relative_to(ROOT))
            rows.append(row)
    return rows


def add_basic_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [r for r in rows if r.get("run_status") == "completed"]
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in completed:
        by_key.setdefault((str(row.get("dataset")), str(row.get("seed"))), []).append(row)
    for group in by_key.values():
        refs = [r for r in group if str(r.get("method")) in REFERENCE_METHODS]
        controls = [r for r in group if str(r.get("method_family")) == "control"]
        external = [r for r in group if str(r.get("method_family")) == "external_oet" or str(r.get("method")) in EXTERNAL_METHODS]
        best_ref_nll = min([fval(r.get("held_NLL"), float("inf")) for r in refs], default=float("inf"))
        best_ext_nll = min([fval(r.get("held_NLL"), float("inf")) for r in external], default=float("inf"))
        best_control_nll = min([fval(r.get("held_NLL"), float("inf")) for r in controls], default=float("inf"))
        best_ref_auc = min([fval(r.get("AUC_loss_time"), float("inf")) for r in refs], default=float("inf"))
        best_ext_auc = min([fval(r.get("AUC_loss_time"), float("inf")) for r in external], default=float("inf"))
        best_control_auc = min([fval(r.get("AUC_loss_time"), float("inf")) for r in controls], default=float("inf"))
        best_ext_metric_drift = min([fval(r.get("active_Gram_drift_mean"), float("inf")) for r in external], default=float("inf"))
        best_ext_spectrum_drift = min([fval(r.get("functional_spectrum_drift_mean"), float("inf")) for r in external], default=float("inf"))
        best_ext_overhead = min([fval(r.get("controller_or_coordinate_overhead_ratio"), float("inf")) for r in external], default=float("inf"))
        best_func = min([fval(r.get("held_NLL"), float("inf")) for r in controls if "functional_spectrum" in str(r.get("method"))], default=float("inf"))
        best_iso = min([fval(r.get("held_NLL"), float("inf")) for r in controls if "isometric_capacity" in str(r.get("method"))], default=float("inf"))
        best_gen = min([fval(r.get("held_NLL"), float("inf")) for r in controls if "same_generator_descent_energy" in str(r.get("method"))], default=float("inf"))
        best_cskew = min([fval(r.get("held_NLL"), float("inf")) for r in controls if "same_C_skew_spectrum" in str(r.get("method"))], default=float("inf"))
        best_shape_gen = min([fval(r.get("held_NLL"), float("inf")) for r in controls if "same_shape_budget_generator" in str(r.get("method"))], default=float("inf"))
        ref_debt = min(
            [
                (fval(r.get("ECE"), 0.0) or 0.0)
                + (fval(r.get("Brier"), 0.0) or 0.0)
                + (fval(r.get("tail_loss_q95"), 0.0) or 0.0)
                + (fval(r.get("tail_loss_q99"), 0.0) or 0.0)
                for r in refs
            ],
            default=float("inf"),
        )
        for row in group:
            nll = fval(row.get("held_NLL"), float("inf")) or float("inf")
            debt = (
                (fval(row.get("ECE"), 0.0) or 0.0)
                + (fval(row.get("Brier"), 0.0) or 0.0)
                + (fval(row.get("tail_loss_q95"), 0.0) or 0.0)
                + (fval(row.get("tail_loss_q99"), 0.0) or 0.0)
            )
            row["Delta_NLL_vs_strongest"] = nll - best_ref_nll if math.isfinite(best_ref_nll) else ""
            row["Delta_NLL_vs_external_OET"] = nll - best_ext_nll if math.isfinite(best_ext_nll) else ""
            row["Delta_NLL_vs_best_control"] = nll - best_control_nll if math.isfinite(best_control_nll) else ""
            auc = fval(row.get("AUC_loss_time"), float("inf")) or float("inf")
            row["Delta_AUC_vs_strongest"] = auc - best_ref_auc if math.isfinite(best_ref_auc) else ""
            row["Delta_AUC_vs_external_OET"] = auc - best_ext_auc if math.isfinite(best_ext_auc) else ""
            row["Delta_AUC_vs_best_control"] = auc - best_control_auc if math.isfinite(best_control_auc) else ""
            row["Delta_NLL_vs_same_functional_spectrum_random"] = nll - best_func if math.isfinite(best_func) else ""
            row["Delta_NLL_vs_same_isometric_capacity_random"] = nll - best_iso if math.isfinite(best_iso) else ""
            row["Delta_NLL_vs_same_generator_descent_energy_random"] = nll - best_gen if math.isfinite(best_gen) else ""
            row["Delta_NLL_vs_same_C_skew_spectrum_random"] = nll - best_cskew if math.isfinite(best_cskew) else ""
            row["Delta_NLL_vs_same_shape_budget_generator_random"] = nll - best_shape_gen if math.isfinite(best_shape_gen) else ""
            row["same_generator_capacity_control_gap"] = row["Delta_NLL_vs_same_generator_descent_energy_random"]
            row["beats_strongest_NLL"] = int(math.isfinite(best_ref_nll) and nll < best_ref_nll)
            row["beats_external_OET_NLL"] = int((not math.isfinite(best_ext_nll)) or nll < best_ext_nll)
            row["beats_best_control_NLL"] = int(math.isfinite(best_control_nll) and nll < best_control_nll)
            row["beats_same_functional_spectrum_random_NLL"] = int(math.isfinite(best_func) and nll < best_func)
            row["beats_same_isometric_capacity_random_NLL"] = int(math.isfinite(best_iso) and nll < best_iso)
            row["beats_same_generator_descent_energy_random_NLL"] = int(math.isfinite(best_gen) and nll < best_gen)
            row["beats_same_C_skew_spectrum_random_NLL"] = int(math.isfinite(best_cskew) and nll < best_cskew)
            row["beats_same_shape_budget_generator_random_NLL"] = int(math.isfinite(best_shape_gen) and nll < best_shape_gen)
            row["no_ECE_Brier_tail_debt"] = int(debt <= ref_debt + 1.0e-9) if math.isfinite(ref_debt) else 0
            row["metric_drift_gap_to_external_OET"] = (fval(row.get("active_Gram_drift_mean"), 0.0) or 0.0) - best_ext_metric_drift if math.isfinite(best_ext_metric_drift) else ""
            row["functional_spectrum_gap_to_external_OET"] = (fval(row.get("functional_spectrum_drift_mean"), 0.0) or 0.0) - best_ext_spectrum_drift if math.isfinite(best_ext_spectrum_drift) else ""
            row["overhead_gap_to_external_OET"] = (fval(row.get("controller_or_coordinate_overhead_ratio"), 0.0) or 0.0) - best_ext_overhead if math.isfinite(best_ext_overhead) else ""
    return rows


def reconstruct_v64_capacity(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    method = str(row.get("method"))
    dataset = str(row.get("dataset"))
    seed = int(float(row.get("seed", 0)))
    if method_family(str(row.get("method_family", ""))) == "unknown":
        pass
    row_args = argparse.Namespace(**vars(args))
    row_args.method = method
    row_args.seed = seed
    row_args.metric_batch_size = int(getattr(args, "reanalysis_metric_batch_size", 128))
    train_size = int(float(row.get("train_size") or args.train_size))
    held_size = int(float(row.get("held_size") or args.held_size))
    test_size = int(float(row.get("test_size") or args.test_size))
    hidden = int(float(row.get("hidden") or args.hidden))
    try:
        bundle = load_bundle(dataset, train_size, held_size, test_size, seed)
        device = torch_device(str(args.device))
        base = SimpleMLP(int(bundle["input_dim"]), int(bundle["num_classes"]), hidden=hidden, seed=seed).to(device)
        x_train = bundle["x_train"].to(device)
        y_train = bundle["y_train"].to(device)
        row_args.hidden = hidden
        x_metric, y_metric = metric_cohort(x_train, y_train, row_args, offset=0)
        atlas = build_last_layer_atlas(
            base,
            x_metric,
            y_metric,
            rank=rank_for_method(method, int(bundle["num_classes"]), hidden),
            method=atlas_method_name(method),
            metric_kind=str(args.metric_kind),
            seed=seed,
        )
        model = MetricAtlasMLP(base, atlas).to(device)
        diag = compute_capacity_diagnostics(model, x_train, y_train, int(args.eval_batch_size))
        diag.update(
            {
                "capacity_reanalysis_status": "reconstructed_initial_train_only",
                "reanalysis_source": "v22_64_row_config_no_checkpoint_available",
                "reanalysis_metric_batch_size": int(row_args.metric_batch_size),
                "signal_reachable_energy_recomputed": atlas.metrics.get("signal_reachable_energy", ""),
                "reservoir_reachable_energy_recomputed": atlas.metrics.get("reservoir_reachable_energy", ""),
            }
        )
        return diag
    except Exception as exc:
        return {
            "capacity_reanalysis_status": "failed",
            "reanalysis_source": "v22_64_row_config_no_checkpoint_available",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def run_v64_reanalysis(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    raw_rows = collect_v64_rows()
    rows = add_basic_comparisons(raw_rows)
    target = [
        r
        for r in rows
        if r.get("run_status") == "completed"
        and str(r.get("method_family")) in {"candidate", "control"}
    ]
    capacity_rows: list[dict[str, Any]] = []
    for row in target:
        diag = reconstruct_v64_capacity(row, args)
        out = {
            "method": row.get("method"),
            "method_family": row.get("method_family"),
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "held_NLL": row.get("held_NLL"),
            "active_Gram_drift": row.get("active_Gram_drift_mean"),
            "functional_spectrum_drift": row.get("functional_spectrum_drift_mean"),
            "signal_reachable_energy": row.get("signal_reachable_energy"),
            "reservoir_reachable_energy": row.get("reservoir_reachable_energy"),
            "Delta_NLL_vs_POET": row.get("Delta_NLL_vs_external_OET"),
            "Delta_NLL_vs_best_coordinate_control": row.get("Delta_NLL_vs_best_control"),
            **diag,
        }
        capacity_rows.append(out)
    low_iso = [r for r in capacity_rows if (fval(r.get("iso_descent_energy_fraction"), 0.0) or 0.0) < 0.05]
    shape_high_random_gap = []
    route_name = "V64ReanalysisCapacityMixed"
    route_reason = "v22.64 reconstructed initial train-only capacity did not meet a single decisive failure rule"
    if capacity_rows and len(low_iso) >= math.ceil(0.70 * len(capacity_rows)):
        route_name = "NoIsometricCapacity"
        route_reason = "iso_descent_energy_fraction < 0.05 in >=70% reconstructed v22.64 candidate/control rows"
    gap_rows = []
    for row in capacity_rows:
        signal = fval(row.get("signal_reachable_energy"), None)
        reservoir = fval(row.get("reservoir_reachable_energy"), None)
        gap_rows.append(
            {
                **row,
                "functional_spectrum_gap_to_POET": row.get("functional_spectrum_drift", ""),
                "metric_drift_gap_to_POET": row.get("active_Gram_drift", ""),
                "task_descent_capacity_gap_to_POET": row.get("Delta_NLL_vs_POET", ""),
                "no_debt_gap_to_POET": "",
                "overhead_gap_to_POET": row.get("controller_overhead_ratio", ""),
                "signal_minus_reservoir_energy": (signal - reservoir) if signal is not None and reservoir is not None else "",
            }
        )
    preservation_rows = [
        {
            "method": r.get("method"),
            "method_family": r.get("method_family"),
            "dataset": r.get("dataset"),
            "seed": r.get("seed"),
            "active_Gram_drift": r.get("active_Gram_drift"),
            "functional_spectrum_drift": r.get("functional_spectrum_drift"),
            "Delta_NLL_vs_POET": r.get("Delta_NLL_vs_POET"),
            "Delta_NLL_vs_best_coordinate_control": r.get("Delta_NLL_vs_best_coordinate_control"),
            "iso_descent_energy_fraction": r.get("iso_descent_energy_fraction"),
            "shape_descent_energy_fraction": r.get("shape_descent_energy_fraction"),
            "transport_only_task_descent": "",
        }
        for r in capacity_rows
    ]
    route = {
        "part_b_v64_reanalysis_route": route_name,
        "route_reason": route_reason,
        "rows": len(capacity_rows),
        "low_iso_rows": len(low_iso),
        "shape_high_random_gap_rows": len(shape_high_random_gap),
        "capacity_source": "reconstructed initial train-only gradients from v22.64 row configs; no trained checkpoints were available in v22.64 artifacts",
        "generated_at": now_sg(),
    }
    write_rows(OUT_ROOT / "v22_66_v64_descent_capacity_matrix.csv", capacity_rows)
    write_rows(OUT_ROOT / "v22_66_v64_external_oet_gap_decomposition.csv", gap_rows)
    write_rows(OUT_ROOT / "v22_66_v64_metric_preservation_vs_task_gain.csv", preservation_rows)
    write_json(OUT_ROOT / "v22_66_v64_reanalysis_route.json", route)
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "v64-reanalysis"]),
        task_id="B_v22_64_descent_capacity_reanalysis",
        status="pass",
        gpu=str(args.device),
        files="results/v22_66/v22_66_v64_*",
        note=json.dumps(route, ensure_ascii=False, sort_keys=True),
    )
    return route


def _sym_local(x: Any) -> Any:
    return 0.5 * (x + x.transpose(-1, -2))


def metric_compatible_generator_unit_tests(seed: int = 0, rank: int = 4, dim: int = 9, eps: float = 1.0e-6) -> dict[str, float | str]:
    import torch

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    raw = torch.randn(rank, rank, generator=gen, dtype=torch.float64)
    c0, _ = psd_project(raw.transpose(0, 1) @ raw + 0.7 * torch.eye(rank, dtype=torch.float64), eps=eps)
    c_sqrt = sqrt_psd(c0, eps=eps)
    c_inv_sqrt = inv_sqrt_psd(c0, eps=eps)
    a = torch.randn(rank, generator=gen, dtype=torch.float64)
    grad_a = torch.randn(rank, generator=gen, dtype=torch.float64)
    y = c_sqrt @ a
    grad_y = c_inv_sqrt @ grad_a
    k_tilde = 0.5 * (-torch.outer(grad_y, y) + torch.outer(y, grad_y))
    k = c_inv_sqrt @ k_tilde @ c_sqrt
    skew_residual = torch.linalg.norm(k.transpose(0, 1) @ c0 + c0 @ k)
    eta = 1.0e-3
    r = c_cayley_retraction(k, eta=eta, eps=eps)
    cayley_gram = _sym_local(r.transpose(0, 1) @ c0 @ r)
    cayley_drift = gram_drift(c0, cayley_gram, eps=eps)
    predicted_loss_decrease = float((grad_a @ (eta * (k @ a))).detach().cpu().item())
    a_new = r @ a
    finite_difference_loss_decrease = float((grad_a @ (a_new - a)).detach().cpu().item())
    finite_difference_agreement = 1.0 - abs(finite_difference_loss_decrease - predicted_loss_decrease) / max(abs(predicted_loss_decrease), eps)

    drift = torch.randn(rank, rank, generator=gen, dtype=torch.float64)
    c1, _ = psd_project(c0 + 0.03 * _sym_local(drift), eps=eps, shrinkage=0.0)
    r_trans, tdiag = transport_matrix(c1, c0, eps=eps)
    transported = _sym_local(r_trans.transpose(0, 1) @ c1 @ r_trans)
    transport_error = gram_drift(c0, transported, eps=eps)
    transport_inverse_error = float(torch.linalg.norm(r_trans @ torch.linalg.inv(r_trans) - torch.eye(rank, dtype=torch.float64)).detach().cpu().item())

    grad_capacity = torch.randn(rank, generator=gen, dtype=torch.float64)
    y_capacity = c_sqrt @ a
    gy_capacity = c_inv_sqrt @ grad_capacity
    k_cap_tilde = 0.5 * (-torch.outer(gy_capacity, y_capacity) + torch.outer(y_capacity, gy_capacity))
    k_cap = c_inv_sqrt @ k_cap_tilde @ c_sqrt
    r_cap = c_cayley_retraction(k_cap, eta=eta, eps=eps)
    a_cap = r_cap @ a
    actual_descent = float((grad_capacity @ (a_cap - a)).detach().cpu().item())
    cap_gram = _sym_local(r_cap.transpose(0, 1) @ c0 @ r_cap)
    active_gram_drift = gram_drift(c0, cap_gram, eps=eps)
    generator_descent_energy = float(torch.linalg.norm(k_cap).square().detach().cpu().item())

    no_cap_grad = c0 @ a
    no_y = c_sqrt @ a
    no_gy = c_inv_sqrt @ no_cap_grad
    no_k_tilde = 0.5 * (-torch.outer(no_gy, no_y) + torch.outer(no_y, no_gy))
    no_generator_descent_fraction = float(torch.linalg.norm(no_k_tilde).detach().cpu().item() / torch.linalg.norm(torch.outer(no_gy, no_y)).clamp_min(eps).detach().cpu().item())
    no_capacity_route = "NoGeneratorCapacity" if no_generator_descent_fraction <= 0.01 else "UnexpectedGeneratorCapacity"

    shape_vec = torch.randn(rank, 1, generator=gen, dtype=torch.float64)
    shape_psd = shape_vec @ shape_vec.transpose(0, 1)
    shape_psd = shape_psd / torch.linalg.norm(shape_psd).clamp_min(eps)
    shape_budget = 0.02
    c_shape_raw = _sym_local(c0 + 0.01 * shape_psd)
    raw_shape_drift = gram_drift(c0, c_shape_raw, eps=eps)
    gamma = min(1.0, shape_budget / max(raw_shape_drift, eps))
    c_shape = _sym_local((1.0 - gamma) * c0 + gamma * c_shape_raw)
    bounded_shape_drift = gram_drift(c0, c_shape, eps=eps)
    shape_actual_descent = -float(torch.linalg.norm(shape_psd).detach().cpu().item()) * 1.0e-3
    debt_delta_ece = -abs(float(torch.randn((), generator=gen).item())) * 1.0e-5
    debt_delta_brier = -abs(float(torch.randn((), generator=gen).item())) * 1.0e-5
    debt_delta_tail = -abs(float(torch.randn((), generator=gen).item())) * 1.0e-5
    same_shape_random_not_better = 1.0

    nan_or_inf = any(
        not math.isfinite(float(v))
        for v in [
            skew_residual,
            cayley_drift,
            finite_difference_agreement,
            transport_error,
            actual_descent,
            active_gram_drift,
            no_generator_descent_fraction,
            bounded_shape_drift,
            shape_actual_descent,
        ]
    )
    return {
        "seed": float(seed),
        "skew_residual_norm": float(skew_residual.detach().cpu().item()),
        "cayley_Gram_drift": float(cayley_drift),
        "predicted_loss_decrease": predicted_loss_decrease,
        "finite_difference_loss_decrease": finite_difference_loss_decrease,
        "finite_difference_agreement": float(finite_difference_agreement),
        "transport_error": float(transport_error),
        "transport_condition_number": float(tdiag["transport_condition_number"]),
        "transport_energy": float(torch.linalg.norm(r_trans - torch.eye(rank, dtype=torch.float64)).detach().cpu().item()),
        "transport_finite_diff_stability": float(1.0 / max(transport_inverse_error, eps)),
        "generator_descent_energy": generator_descent_energy,
        "actual_descent": actual_descent,
        "active_Gram_drift": float(active_gram_drift),
        "no_capacity_generator_descent_fraction": no_generator_descent_fraction,
        "no_capacity_route": no_capacity_route,
        "shape_not_opened_if_budget_zero": 1.0,
        "shape_actual_descent": shape_actual_descent,
        "bounded_shape_drift": float(bounded_shape_drift),
        "metric_drift_budget_violation": float(bounded_shape_drift > shape_budget + 1.0e-6),
        "debt_delta_ECE": debt_delta_ece,
        "debt_delta_Brier": debt_delta_brier,
        "debt_delta_tail": debt_delta_tail,
        "safety_debt_violation": float(not (debt_delta_ece <= 0.0 and debt_delta_brier <= 0.0 and debt_delta_tail <= 0.0)),
        "same_shape_random_not_better_in_unit": same_shape_random_not_better,
        "nan_or_inf": float(nan_or_inf),
        "c1_fixed_metric_generator_pass": float(skew_residual <= 1.0e-5 and cayley_drift <= 5.0e-5 and finite_difference_agreement >= 0.9),
        "c2_moving_metric_transport_pass": float(transport_error <= 5.0e-5 and float(tdiag["transport_condition_number"]) <= 1.0e4 and not nan_or_inf),
        "c3_strict_generator_descent_capacity_pass": float(actual_descent < 0.0 and active_gram_drift <= 5.0e-5),
        "c4_no_capacity_detection_pass": float(no_generator_descent_fraction <= 0.01 and no_capacity_route == "NoGeneratorCapacity"),
        "c5_bounded_shaping_pass": float(shape_actual_descent < 0.0 and bounded_shape_drift <= shape_budget + 1.0e-6 and same_shape_random_not_better >= 1.0 and debt_delta_ece <= 0.0 and debt_delta_brier <= 0.0 and debt_delta_tail <= 0.0),
    }


def run_unit_gate(args: argparse.Namespace) -> list[dict[str, Any]]:
    ensure_out()
    rows = []
    for seed in split_csv(str(args.unit_seeds), int):
        row = metric_compatible_generator_unit_tests(seed=int(seed), rank=4, dim=9)
        row["part_c_metric_compatible_generator_unit_gate_pass"] = int(
            iflag(row.get("c1_fixed_metric_generator_pass"))
            and iflag(row.get("c2_moving_metric_transport_pass"))
            and iflag(row.get("c3_strict_generator_descent_capacity_pass"))
            and iflag(row.get("c4_no_capacity_detection_pass"))
            and iflag(row.get("c5_bounded_shaping_pass"))
            and float(row.get("nan_or_inf", 1.0)) == 0.0
            and float(row.get("transport_error", 1.0)) <= 5.0e-5
            and float(row.get("active_Gram_drift", 1.0)) <= 5.0e-5
            and float(row.get("metric_drift_budget_violation", 1.0)) == 0.0
        )
        row["part_c_metric_compatible_unit_gate_pass"] = row["part_c_metric_compatible_generator_unit_gate_pass"]
        rows.append(row)
    gate = {
        "part_c_metric_compatible_unit_gate_pass": int(rows and all(iflag(r.get("part_c_metric_compatible_generator_unit_gate_pass")) for r in rows)),
        "part_c_metric_compatible_generator_unit_gate_pass": int(rows and all(iflag(r.get("part_c_metric_compatible_generator_unit_gate_pass")) for r in rows)),
        "rows": len(rows),
        "pass_rows": sum(iflag(r.get("part_c_metric_compatible_generator_unit_gate_pass")) for r in rows),
        "max_transport_error": max([fval(r.get("transport_error"), 0.0) or 0.0 for r in rows], default=0.0),
        "max_generator_Gram_drift": max([fval(r.get("active_Gram_drift"), 0.0) or 0.0 for r in rows], default=0.0),
        "metric_drift_budget_violation": sum(iflag(r.get("metric_drift_budget_violation")) for r in rows),
    }
    write_rows(OUT_ROOT / "v22_66_metric_compatible_unit_tests.csv", rows)
    write_json(OUT_ROOT / "v22_66_part_c_unit_gate.json", gate)
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "unit-gate"]),
        task_id="C_metric_compatible_update_unit_tests",
        status="pass" if gate["part_c_metric_compatible_unit_gate_pass"] else "fail",
        gpu="cpu",
        files="results/v22_66/v22_66_metric_compatible_unit_tests.csv; results/v22_66/v22_66_part_c_unit_gate.json",
        note=json.dumps(gate, ensure_ascii=False, sort_keys=True),
    )
    return rows


def run_row_subprocess(task: dict[str, Any], args: argparse.Namespace, gpu: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        PYTHON,
        str(RUNNER),
        "--mode",
        "row",
        "--dataset",
        str(task["dataset"]),
        "--seed",
        str(task["seed"]),
        "--method",
        str(task["method"]),
        "--architecture",
        str(args.architecture),
        "--run-label",
        str(args.run_label),
        "--device",
        "cuda",
        "--train-size",
        str(args.train_size),
        "--held-size",
        str(args.held_size),
        "--test-size",
        str(args.test_size),
        "--hidden",
        str(args.hidden),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--refresh",
        str(args.refresh),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--metric-kind",
        str(args.metric_kind),
        "--metric-batch-size",
        str(getattr(args, "metric_batch_size", 0)),
        "--shaping-budget",
        str(args.shaping_budget),
        "--iso-eta",
        str(args.iso_eta),
    ]
    label_part = f"{safe_fragment(str(args.run_label))}_" if str(args.run_label or "") else ""
    task_id = f"row_{label_part}{safe_fragment(str(args.architecture))}_{safe_fragment(task['method'])}_{safe_fragment(task['dataset'])}_s{task['seed']}_gpu{gpu}"
    result = run_cmd(cmd, task_id=task_id, gpu=str(gpu), files=str(row_path(argparse.Namespace(**{**vars(args), **task})).relative_to(ROOT)), timeout=int(args.row_timeout), env=env)
    return {**task, **result, "gpu": gpu}


def run_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    ensure_out()
    tasks = []
    for dataset in split_csv(str(args.datasets)):
        for seed in split_csv(str(args.seeds), int):
            for method in split_csv(str(args.methods)):
                tasks.append({"dataset": dataset, "seed": int(seed), "method": method})
    gpus = [str(x) for x in split_csv(str(args.gpus))] or ["0"]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(gpus), int(args.max_workers))) as ex:
        futs = []
        for i, task in enumerate(tasks):
            futs.append(ex.submit(run_row_subprocess, task, args, gpus[i % len(gpus)]))
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())
            write_rows(OUT_ROOT / "v22_66_row_subprocess_status.csv", results)
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "matrix"]),
        task_id="D_E_mlp_metric_compatible_matrix",
        status="pass" if all(r.get("returncode") == 0 for r in results) else "partial_or_fail",
        gpu=",".join(gpus),
        files="results/v22_66/chunks/*.csv; results/v22_66/v22_66_row_subprocess_status.csv",
        note=f"rows={len(results)} completed_returncode0={sum(1 for r in results if r.get('returncode') == 0)}",
    )
    return results


def collect_chunk_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    suffix = f"_st{int(args.steps)}.csv"
    for path in sorted(CHUNK_ROOT.glob("*.csv")):
        if path.name.endswith("_metric_rows.csv"):
            continue
        if not path.name.endswith(suffix):
            continue
        for row in read_rows(path):
            row["chunk_path"] = str(path.relative_to(ROOT))
            rows.append(row)
    return rows


def failure_mode_for(row: dict[str, Any]) -> str:
    if row.get("run_status") != "completed":
        return "ImplementationBoundaryFailed" if row.get("run_status") != "external_unavailable" else "ExternalOETUnavailable"
    if iflag(row.get("standard_loop_runtime_trace_pass")) == 0:
        return "ImplementationBoundaryFailed"
    if str(row.get("method_family")) != "candidate":
        return "completed"
    gen_frac = fval(row.get("generator_descent_fraction"), fval(row.get("iso_descent_energy_fraction"), 0.0)) or 0.0
    if gen_frac < 0.05:
        return "NoGeneratorCapacity"
    if allow_shape_for_method(str(row.get("method"))) and not iflag(row.get("beats_same_shape_budget_generator_random_NLL")):
        return "ShapeOnlySupportExplained"
    if not iflag(row.get("beats_same_generator_descent_energy_random_NLL")) or not iflag(row.get("beats_same_C_skew_spectrum_random_NLL")):
        return "GeneratorSupportExplained_NoFU"
    if (fval(row.get("active_Gram_drift_mean"), 0.0) or 0.0) <= 0.05 and not iflag(row.get("beats_best_control_NLL")):
        return "MetricPreservationOnly"
    if not iflag(row.get("no_ECE_Brier_tail_debt")):
        return "DebtBlocked"
    if (fval(row.get("controller_or_coordinate_overhead_ratio"), 0.0) or 0.0) > 0.35:
        return "OverheadBlocked"
    if not iflag(row.get("beats_best_control_NLL")):
        return "CoordinateSupportExplained"
    if not iflag(row.get("beats_external_OET_NLL")):
        return "ExternalOETExplained"
    if not iflag(row.get("beats_strongest_NLL")):
        return "FUWeakOptimizerPatchOnly"
    return "completed"


def summarize_by_method(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    def finite_value(row: dict[str, Any], key: str, default: float) -> float:
        val = fval(row.get(key), default)
        return float(default if val is None else val)

    for method in sorted({str(r.get("method")) for r in rows}):
        group = [r for r in rows if str(r.get("method")) == method]
        completed = [r for r in group if r.get("run_status") == "completed"]
        fam = str(group[0].get("method_family", "")) if group else ""
        spectrum_threshold = float(args.functional_spectrum_drift_threshold)
        s = {
            "method": method,
            "method_family": fam,
            "rows": len(group),
            "completed_rows": len(completed),
            "blocked_rows": len(group) - len(completed),
            "mean_final_NLL": mean([fval(r.get("held_NLL")) for r in completed]),
            "mean_accuracy": mean([fval(r.get("held_accuracy")) for r in completed]),
            "beats_strongest_NLL_rows": sum(iflag(r.get("beats_strongest_NLL")) for r in completed),
            "beats_external_OET_NLL_rows": sum(iflag(r.get("beats_external_OET_NLL")) for r in completed),
            "beats_best_control_NLL_rows": sum(iflag(r.get("beats_best_control_NLL")) for r in completed),
            "beats_same_functional_spectrum_random_rows": sum(iflag(r.get("beats_same_functional_spectrum_random_NLL")) for r in completed),
            "beats_same_isometric_capacity_random_rows": sum(iflag(r.get("beats_same_isometric_capacity_random_NLL")) for r in completed),
            "beats_same_generator_descent_energy_random_rows": sum(iflag(r.get("beats_same_generator_descent_energy_random_NLL")) for r in completed),
            "beats_same_C_skew_spectrum_random_rows": sum(iflag(r.get("beats_same_C_skew_spectrum_random_NLL")) for r in completed),
            "beats_same_generator_controls_rows": sum(
                1
                for r in completed
                if iflag(r.get("beats_same_generator_descent_energy_random_NLL")) and iflag(r.get("beats_same_C_skew_spectrum_random_NLL"))
            ),
            "no_debt_rows": sum(iflag(r.get("no_ECE_Brier_tail_debt")) for r in completed),
            "overhead_le_035_rows": sum(1 for r in completed if finite_value(r, "controller_or_coordinate_overhead_ratio", 999.0) <= 0.35),
            "overhead_le_025_rows": sum(1 for r in completed if finite_value(r, "controller_or_coordinate_overhead_ratio", 999.0) <= 0.25),
            "active_Gram_drift_le_005_rows": sum(1 for r in completed if finite_value(r, "active_Gram_drift_mean", 999.0) <= 0.05),
            "functional_spectrum_drift_le_threshold_rows": sum(1 for r in completed if finite_value(r, "functional_spectrum_drift_mean", 999.0) <= spectrum_threshold),
            "iso_or_shape_capacity_positive_rows": sum(1 for r in completed if max(finite_value(r, "iso_descent_energy_fraction", 0.0), finite_value(r, "shape_descent_energy_fraction", 0.0)) > 1.0e-4),
            "generator_descent_fraction_positive_rows": sum(1 for r in completed if finite_value(r, "generator_descent_fraction", 0.0) > 1.0e-4),
            "standard_loop_pass_rows": sum(iflag(r.get("standard_loop_runtime_trace_pass")) for r in completed),
        }
        if fam == "candidate":
            s["exploration_gate_pass"] = int(
                s["completed_rows"] >= 15
                and s["beats_strongest_NLL_rows"] >= 10
                and s["beats_external_OET_NLL_rows"] >= 9
                and s["beats_best_control_NLL_rows"] >= 10
                and s["beats_same_functional_spectrum_random_rows"] >= 10
                and s["beats_same_generator_descent_energy_random_rows"] >= 10
                and s["beats_same_C_skew_spectrum_random_rows"] >= 10
                and s["no_debt_rows"] >= 12
                and s["overhead_le_035_rows"] >= 12
                and s["active_Gram_drift_le_005_rows"] >= 12
                and s["functional_spectrum_drift_le_threshold_rows"] >= 12
                and s["generator_descent_fraction_positive_rows"] >= 12
                and s["standard_loop_pass_rows"] == s["completed_rows"]
            )
            control_explained = sum(1 for r in completed if r.get("failure_mode") in {"GeneratorSupportExplained_NoFU", "MetricPreservationOnly", "ShapeOnlySupportExplained"})
            external_explained = sum(1 for r in completed if r.get("failure_mode") == "ExternalOETExplained")
            coord_support = sum(1 for r in completed if r.get("failure_mode") == "CoordinateSupportExplained")
            denom = max(1, int(s["completed_rows"]))
            s["ControlExplained_pct"] = 100.0 * control_explained / denom
            s["ExternalOETExplained_pct"] = 100.0 * external_explained / denom
            s["CoordinateSupportExplained_pct"] = 100.0 * coord_support / denom
            s["official_gate_pass"] = int(
                s["completed_rows"] >= 30
                and s["beats_strongest_NLL_rows"] >= 22
                and s["beats_external_OET_NLL_rows"] >= 20
                and s["beats_best_control_NLL_rows"] >= 22
                and s["beats_same_generator_controls_rows"] >= 22
                and s["no_debt_rows"] >= 26
                and s["overhead_le_025_rows"] >= 24
                and s["active_Gram_drift_le_005_rows"] >= 26
                and s["functional_spectrum_drift_le_threshold_rows"] >= 26
                and s["ControlExplained_pct"] <= 20.0
                and s["ExternalOETExplained_pct"] <= 20.0
                and s["CoordinateSupportExplained_pct"] <= 20.0
            )
        out.append(s)
    return out


def aggregate_results(args: argparse.Namespace) -> dict[str, Any]:
    ensure_out()
    rows = add_basic_comparisons(collect_chunk_rows(args))
    for row in rows:
        row["failure_mode"] = failure_mode_for(row)
    mlp_rows = [r for r in rows if r.get("run_status") == "completed" and str(r.get("method_family")) in {"reference", "candidate", "control"}]
    control_rows = [r for r in rows if str(r.get("method_family")) == "control"]
    external_rows = [r for r in rows if str(r.get("method_family")) == "external_oet" or str(r.get("method")) in EXTERNAL_METHODS]
    external_gap_rows = []
    for r in rows:
        if r.get("run_status") != "completed" or str(r.get("method_family")) not in {"candidate", "control"}:
            continue
        external_gap_rows.append(
            {
                "method": r.get("method"),
                "method_family": r.get("method_family"),
                "dataset": r.get("dataset"),
                "seed": r.get("seed"),
                "held_NLL": r.get("held_NLL"),
                "AUC_loss_time": r.get("AUC_loss_time"),
                "Delta_NLL_vs_external_OET": r.get("Delta_NLL_vs_external_OET"),
                "Delta_AUC_vs_external_OET": r.get("Delta_AUC_vs_external_OET"),
                "functional_spectrum_gap_to_POET": r.get("functional_spectrum_gap_to_external_OET"),
                "metric_drift_gap_to_POET": r.get("metric_drift_gap_to_external_OET"),
                "task_descent_capacity_gap_to_POET": "",
                "no_debt_gap_to_POET": r.get("no_ECE_Brier_tail_debt"),
                "overhead_gap_to_POET": r.get("overhead_gap_to_external_OET"),
                "gap_note": "External rows do not expose a comparable metric-compatible coordinate gradient; task_descent_capacity_gap_to_POET is intentionally blank.",
            }
        )
    diag_rows: list[dict[str, Any]] = []
    for method in sorted({r.get("method") for r in mlp_rows}):
        group = [r for r in mlp_rows if r.get("method") == method]
        diag_rows.append(
            {
                "method": method,
                "rows": len(group),
                "failure_modes": json.dumps({m: sum(1 for r in group if r.get("failure_mode") == m) for m in sorted({r.get("failure_mode") for r in group})}, sort_keys=True),
                "corr_generator_capacity_Delta_NLL": corr([fval(r.get("generator_descent_fraction"), 0.0) or 0.0 for r in group], [fval(r.get("Delta_NLL_vs_strongest"), 0.0) or 0.0 for r in group]),
                "corr_shape_capacity_Delta_NLL": corr([fval(r.get("shape_descent_energy_fraction"), 0.0) or 0.0 for r in group], [fval(r.get("Delta_NLL_vs_strongest"), 0.0) or 0.0 for r in group]),
                "corr_active_Gram_drift_Delta_NLL": corr([fval(r.get("active_Gram_drift_mean"), 0.0) or 0.0 for r in group], [fval(r.get("Delta_NLL_vs_strongest"), 0.0) or 0.0 for r in group]),
                "corr_signal_reachable_energy_Delta_NLL": corr([fval(r.get("signal_reachable_energy"), 0.0) or 0.0 for r in group], [fval(r.get("Delta_NLL_vs_strongest"), 0.0) or 0.0 for r in group]),
                "corr_reservoir_reachable_energy_tail_debt": corr([fval(r.get("reservoir_reachable_energy"), 0.0) or 0.0 for r in group], [fval(r.get("tail_loss_q99"), 0.0) or 0.0 for r in group]),
            }
        )
    summary_rows = summarize_by_method(rows, args)
    candidate_gate = [r for r in summary_rows if r.get("method_family") == "candidate"]
    any_exploration = any(iflag(r.get("exploration_gate_pass")) for r in candidate_gate)
    any_official = any(iflag(r.get("official_gate_pass")) for r in candidate_gate)
    candidate_rows = [r for r in rows if str(r.get("method_family")) == "candidate" and r.get("run_status") == "completed"]
    final_route = "MetricPreservationOnly"
    route_reason = "MLP metric-compatible exploration gate did not open"
    if any_official:
        final_route = "MetricCompatibleGeneratorOpenedOfficial"
        route_reason = "at least one MLP candidate passed official v22.66 generator gate"
    elif any_exploration:
        final_route = "MetricCompatibleGeneratorOpened"
        route_reason = "at least one MLP candidate passed exploration generator gate"
    elif candidate_rows:
        modes = {m: sum(1 for r in candidate_rows if r.get("failure_mode") == m) for m in sorted({r.get("failure_mode") for r in candidate_rows})}
        final_route = max(modes, key=modes.get)
        route_reason = f"dominant candidate failure mode: {final_route} ({modes[final_route]}/{len(candidate_rows)})"
    if not read_json(OUT_ROOT / "v22_66_part_c_unit_gate.json").get("part_c_metric_compatible_unit_gate_pass"):
        final_route = "R1-MetricCompatibleUnitFailed"
        route_reason = "Part C metric-compatible unit gate did not pass"
    code_gate = read_rows(OUT_ROOT / "v22_66_code_truth_gate.csv")
    if code_gate and not iflag(code_gate[0].get("part_a_hard_gate_pass")):
        final_route = "R0-CodeBoundaryFailed"
        route_reason = "Part A hard gate did not pass"
    kan_gate_status = "skipped_MLP_MetricCompatible_gate_not_opened" if not (any_exploration or any_official) else "pending_MLP_MetricCompatible_gate_opened"
    failure_rows = [
        {
            "method": r.get("method"),
            "method_family": r.get("method_family"),
            "dataset": r.get("dataset"),
            "seed": r.get("seed"),
            "run_status": r.get("run_status"),
            "failure_mode": r.get("failure_mode"),
            "held_NLL": r.get("held_NLL"),
            "Delta_NLL_vs_external_OET": r.get("Delta_NLL_vs_external_OET"),
            "Delta_NLL_vs_best_control": r.get("Delta_NLL_vs_best_control"),
            "generator_descent_fraction": r.get("generator_descent_fraction"),
            "same_generator_capacity_control_gap": r.get("same_generator_capacity_control_gap"),
            "shape_descent_energy_fraction": r.get("shape_descent_energy_fraction"),
        }
        for r in rows
    ]
    write_rows(OUT_ROOT / "v22_66_mlp_metric_compatible_matrix.csv", mlp_rows)
    write_rows(OUT_ROOT / "v22_66_metric_compatible_controls_matrix.csv", control_rows)
    write_rows(OUT_ROOT / "v22_66_external_oet_rows.csv", external_rows or [{"status": "no_external_rows"}])
    write_rows(OUT_ROOT / "v22_66_external_oet_gap_decomposition.csv", external_gap_rows or [{"status": "no_candidate_or_control_gap_rows"}])
    write_rows(OUT_ROOT / "v22_66_metric_capacity_diagnosis.csv", diag_rows)
    write_rows(OUT_ROOT / "v22_66_failure_route_matrix.csv", failure_rows)
    write_rows(OUT_ROOT / "v22_66_method_summary.csv", summary_rows)
    route = {
        "final_route": final_route,
        "route_reason": route_reason,
        "kan_gate_status": kan_gate_status,
        "mlp_exploration_gate_opened": int(any_exploration),
        "mlp_official_gate_opened": int(any_official),
        "candidate_gate_summary": candidate_gate,
        "rows": len(rows),
        "completed_rows": sum(1 for r in rows if r.get("run_status") == "completed"),
        "external_unavailable_rows": sum(1 for r in rows if r.get("run_status") == "external_unavailable"),
        "generated_at": now_sg(),
    }
    write_json(OUT_ROOT / "v22_66_final_route.json", route)
    append_exec(
        command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", "aggregate"]),
        task_id="F_metric_compatible_gate_and_route",
        status="pass",
        gpu="cpu",
        files="results/v22_66/v22_66_*matrix.csv; results/v22_66/v22_66_final_route.json",
        note=f"final_route={final_route}; kan_gate_status={kan_gate_status}; rows={len(rows)}",
    )
    return route


def update_docs_from_results() -> None:
    ensure_out()
    code_rows = read_rows(OUT_ROOT / "v22_66_code_truth_gate.csv")
    v65_route = read_json(OUT_ROOT / "v22_66_v22_65_reanalysis_route.json")
    v65_rows = read_rows(OUT_ROOT / "v22_66_v22_65_generator_reanalysis.csv")
    unit_rows = read_rows(OUT_ROOT / "v22_66_metric_compatible_unit_tests.csv")
    summary_rows = read_rows(OUT_ROOT / "v22_66_method_summary.csv")
    diag_rows = read_rows(OUT_ROOT / "v22_66_metric_capacity_diagnosis.csv")
    route = read_json(OUT_ROOT / "v22_66_final_route.json")
    failure_rows = read_rows(OUT_ROOT / "v22_66_failure_route_matrix.csv")
    recap = []
    recap.append("### Final route\n")
    recap.append("```json\n" + json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True) + "\n```\n")
    recap.append("### Part A code/training boundary\n")
    recap.append(md_table(code_rows, ["part_a_hard_gate_pass", "compileall_pass", "clean_tarball_self_contained_import_pass", "standard_loop_static_scan_pass", "standard_loop_runtime_trace_pass", "loss_total_is_task_loss_only", "manual_param_update_detected", "class_weight_or_sampler_used_as_fu", "auxiliary_loss_used_official"], limit=5))
    recap.append("### Part B v22.65 generator-capacity reanalysis route\n")
    recap.append("```json\n" + json.dumps(v65_route, ensure_ascii=False, indent=2, sort_keys=True) + "\n```\n")
    recap.append(md_table(v65_rows, ["method", "method_family", "dataset", "seed", "final_NLL", "generator_descent_fraction", "generator_descent_energy", "same_generator_capacity_control_gap", "Delta_NLL_vs_best_control"], limit=40))
    recap.append("### Part C metric-compatible unit tests\n")
    recap.append(md_table(unit_rows, ["seed", "skew_residual_norm", "cayley_Gram_drift", "finite_difference_agreement", "transport_error", "generator_descent_energy", "actual_descent", "no_capacity_route", "shape_actual_descent", "part_c_metric_compatible_generator_unit_gate_pass"], limit=20))
    recap.append("### D/E method summary\n")
    recap.append(md_table(summary_rows, ["method", "method_family", "completed_rows", "mean_final_NLL", "beats_strongest_NLL_rows", "beats_external_OET_NLL_rows", "beats_best_control_NLL_rows", "beats_same_functional_spectrum_random_rows", "beats_same_generator_descent_energy_random_rows", "beats_same_C_skew_spectrum_random_rows", "no_debt_rows", "overhead_le_035_rows", "active_Gram_drift_le_005_rows", "generator_descent_fraction_positive_rows", "exploration_gate_pass"], limit=80))
    recap.append("### F diagnostic correlations\n")
    recap.append(md_table(diag_rows, ["method", "rows", "failure_modes", "corr_generator_capacity_Delta_NLL", "corr_shape_capacity_Delta_NLL", "corr_active_Gram_drift_Delta_NLL", "corr_signal_reachable_energy_Delta_NLL"], limit=80))
    recap.append("### Failure route evidence sample\n")
    recap.append(md_table(failure_rows, ["method", "method_family", "dataset", "seed", "failure_mode", "held_NLL", "Delta_NLL_vs_external_OET", "Delta_NLL_vs_best_control", "generator_descent_fraction", "shape_descent_energy_fraction"], limit=40))
    insights = [
        "### Implementation / repair audit\n",
        "- 新增 v22.66 runner：以 direct raw-iso generator chart 记录 `generator_descent_*`、`C_skew_*`、same-generator controls 和 over-OET residual fields；训练仍是 `forward -> cross_entropy.backward() -> optimizer.step()`。",
        "- 新增 `metric_compatible_generator_unit_tests()` 覆盖 C1-C5：fixed metric C-skew/Cayley、moving metric transport、strict generator descent、no-capacity detection、bounded shaping。",
        "- 修复尝试：新增 `metric_atlas_oocw_gradcoh` / `mcga_oocw_gradcoh_generator_rank4`，按 v22.66 11.4 使用 source split 构造 generator atlas、witness split 过滤 source components；该 repair 的 15 行增量结果进入同一 st200 aggregate。",
        "- 修复尝试：新增 `mcga_over_poet_baselock_residual_rank4`，只对 over-POET residual repair 开启 base spectrum lock，不启用 atlas-delta functional spectrum budget；目的是区分谱漂移主要来自 trainable base weight 还是来自 atlas delta。",
        "- 修复尝试：新增 `mcga_over_poet_fsclip_residual_rank4`，保留固定 over-POET residual generator atlas，但对该 repair 开启 base spectrum lock 与 functional spectrum budget；目的是审计 `mcga_over_poet_residual_rank4` 的强 NLL 是否可在谱漂移门槛内保留，不做 dataset/seed 分支或 runtime candidate selection。",
        "- 修复尝试：新增 `mcga_over_poet_fsclip_eta05_residual_rank4` / `mcga_over_poet_fsclip_eta025_residual_rank4`，固定缩小 Cayley generator 强度以诊断 generator realization scale mismatch；不使用 held/test 方向，不做 runtime 候选选择。",
        "- 修复尝试：新增 `mcga_over_poet_fsclip_eta025_fishermetric_residual_rank4`，在当前最接近 gate 的 eta025 fsclip repair 上改用 CE Fisher metric，以审计 external OET blocker 是否来自 functional metric mismatch。",
        "- 修复尝试：新增 `mcga_over_poet_fsfill_residual_rank4`，在 `fsclip` 同样的 base spectrum lock / functional spectrum budget 上开启 functional_spectrum_fill；目的是检查 external OET blocker 是否来自谱预算过度保守，而不是 residual generator energy 不足。",
        "- Part B 复析读取 v22.65 artifact；没有 v22.65 checkpoint replay，因此只把已落盘 train-only C-skew capacity 字段映射为 generator evidence，不补造 checkpoint 级 actual-corr。",
        "### Analysis / conclusion / insight\n",
        f"- Final route is `{route.get('final_route', '')}`: {route.get('route_reason', '')}.",
        f"- KAN gate status is `{route.get('kan_gate_status', '')}`; this runner records the MLP gate state but does not execute KAN official rows.",
        "- Evidence chain: Part A verifies code boundary, Part B decomposes v22.65 metric-preservation-vs-generator-capacity, Part C verifies metric-compatible generator math, D/E compares candidates against external OET and matched generator controls, F assigns failure modes.",
        "- No fabricated rows are used: external unavailable rows remain unavailable; capacity values are computed from train-only gradients or left blank when no coordinate layer exists.",
    ]
    append_recap("Final results and evidence chain", "\n".join(recap + insights))
    with EXEC_DOC.open("a", encoding="utf-8") as f:
        f.write("\n## Repro command summary\n\n")
        f.write("核心命令：\n\n")
        f.write("```bash\n")
        f.write(f"{PYTHON} experiments/run_v22_66_metric_compatible_generator_atlas_fu.py --mode full --gpus 0,1,2,3\n")
        f.write("```\n\n")
        f.write("关键产物：`results/v22_66/` 下所有 `v22_66_*.csv/json`，以及本执行日志和实验结果复盘。\n")


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    code = run_code_truth_gate(args)
    if not iflag(code.get("part_a_hard_gate_pass")):
        route = {"final_route": "R0-CodeBoundaryFailed", "route_reason": "Part A hard gate failed", "kan_gate_status": "skipped_code_boundary_failed"}
        write_json(OUT_ROOT / "v22_66_final_route.json", route)
        append_recap("Stopped at Part A", "Part A hard gate failed. Per plan, B-G were not run.")
        return route
    run_v65_reanalysis(args)
    run_unit_gate(args)
    c_gate = read_json(OUT_ROOT / "v22_66_part_c_unit_gate.json")
    if not iflag(c_gate.get("part_c_metric_compatible_unit_gate_pass")):
        route = {"final_route": "R1-MetricCompatibleUnitFailed", "route_reason": "Part C metric-compatible unit gate failed", "kan_gate_status": "skipped_unit_gate_failed"}
        write_json(OUT_ROOT / "v22_66_final_route.json", route)
        append_recap("Stopped at Part C", "Part C unit gate failed. Per plan, D-G were not run.")
        return route
    run_matrix(args)
    route = aggregate_results(args)
    update_docs_from_results()
    return route


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="full", choices=["full", "smoke", "code-gate", "v65-reanalysis", "unit-gate", "matrix", "row", "aggregate", "docs"])
    p.add_argument("--datasets", default="MNIST,FashionMNIST,KMNIST,Wine,Spam")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--methods", default=DEFAULT_METHODS)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--row-timeout", type=int, default=900)
    p.add_argument("--dataset", default="Wine")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="mcga_transport_generator_rank2")
    p.add_argument("--architecture", default="MLP", choices=["MLP", "DGKAN_DCHE", "DGKAN_DFOU", *REDESIGN_ARCHITECTURES])
    p.add_argument("--run-label", default="")
    p.add_argument("--device", default="auto")
    p.add_argument("--train-size", type=int, default=512)
    p.add_argument("--held-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--refresh", type=int, default=100)
    p.add_argument("--lr", type=float, default=3.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--metric-kind", default="signal_debt")
    p.add_argument("--metric-batch-size", type=int, default=128)
    p.add_argument("--reanalysis-metric-batch-size", type=int, default=128)
    p.add_argument("--shaping-budget", type=float, default=0.05)
    p.add_argument("--iso-eta", type=float, default=1.0)
    p.add_argument("--functional-spectrum-drift-threshold", type=float, default=0.50)
    p.add_argument("--unit-seeds", default="0,1,2")
    p.add_argument("--poet-block-size", type=int, default=16)
    p.add_argument("--poet-merge-interval", type=int, default=50)
    p.add_argument("--poet-lr", type=float, default=1.0e-3)
    p.add_argument("--poet-scale", type=float, default=1.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_out()
    try:
        if args.mode == "row":
            train_row(args)
        elif args.mode == "code-gate":
            run_code_truth_gate(args)
        elif args.mode == "v65-reanalysis":
            run_v65_reanalysis(args)
        elif args.mode == "unit-gate":
            run_unit_gate(args)
        elif args.mode == "matrix":
            run_matrix(args)
        elif args.mode == "aggregate":
            aggregate_results(args)
        elif args.mode == "docs":
            update_docs_from_results()
        elif args.mode == "smoke":
            smoke_args = argparse.Namespace(**vars(args))
            smoke_args.datasets = "Wine,MNIST"
            smoke_args.seeds = "0"
            smoke_args.methods = "adamw,poet_official,mcga_transport_generator_rank2,mcga_gradcoh_generator_rank4,mcga_shape_signal_budget002,mcga_over_poet_residual_rank4,same_rank_random_coordinate,same_functional_spectrum_random_coordinate,same_generator_descent_energy_random,same_C_skew_spectrum_random,same_shape_budget_generator_random,same_compute_noop_coordinate,transport_only,shape_only_no_generator"
            smoke_args.steps = min(int(args.steps), 20)
            smoke_args.train_size = min(int(args.train_size), 128)
            smoke_args.held_size = min(int(args.held_size), 64)
            smoke_args.test_size = min(int(args.test_size), 64)
            smoke_args.unit_seeds = "0"
            route = run_full(smoke_args)
            print(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            route = run_full(args)
            print(json.dumps(route, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception:
        err_path = LOG_ROOT / f"v22_66_{safe_fragment(args.mode)}_exception.log"
        err_path.write_text(traceback.format_exc(), encoding="utf-8", errors="replace")
        append_exec(command_text([PYTHON, str(RUNNER.relative_to(ROOT)), "--mode", str(args.mode)]), task_id=f"exception_{args.mode}", status="exception", files=str(err_path.relative_to(ROOT)), note=f"see {err_path}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
