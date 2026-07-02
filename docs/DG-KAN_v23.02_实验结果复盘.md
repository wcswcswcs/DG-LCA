# DG-KAN v23.02 PredictiveTrustFunctionalEdgePopulationFlow 实验结果复盘

创建时间：2026-07-02 23:53:51 +0800

## 当前结论

尚未 final。


## 2026-07-02 23:53:51 +0800 Part B history boundary lock

```json
{
  "candidate_update_selection_used": 0,
  "dominant_blocker": "missing_history_or_v23_01_boundary_field",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-02 23:53:51 +0800",
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 18,
  "part": "B",
  "route": "B_HistoryBoundaryMissing",
  "row_count": 19,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0,
  "v22_94_F5_best": "missing",
  "v22_94_decomposition_pass": 1,
  "v22_94_ordinary_c2_opened": 0,
  "v22_94_tailsafe_kills_C2": "F_SkippedBecausePartEFailed",
  "v22_94_witness_pass": 1,
  "v23_01_E4_FunctionalGram_BlockSNR_s0_C2_coverage": 0.08936140247351432,
  "v23_01_E4_FunctionalGram_BlockSNR_s0_random_gap": 0.042837288723603706,
  "v23_01_E5_FunctionalGram_BlockSNR_degree_s0_C2_coverage": 0.08936140247351432,
  "v23_01_E5_FunctionalGram_BlockSNR_degree_s0_random_gap": 0.042837288723603706,
  "v23_01_F2_accept_rate": 0.05,
  "v23_01_F2_scale_mean": 0.00625,
  "v23_01_final_promotion_allowed": 0,
  "v23_01_local_patch_interaction_median": -0.10533262381431996,
  "v23_01_part_e_route": "E_C2FormationPass",
  "v23_01_part_f_route": "F_SafetyDiagnosticOnlyHeavyRejection",
  "v23_01_part_g_route": "G_PositiveControlFailed",
  "v23_01_repair_f_diag_guard_every2_f2_F5": 0,
  "v23_01_repair_f_diag_guard_every3_f2_F5": 0,
  "v23_01_rotation_sensitive_median": 0.23397090542130172
}
```

## 2026-07-02 23:53:52 +0800 Part A identity/control readiness

```json
{
  "candidate_update_selection_used": 0,
  "compile_errors": [],
  "dominant_blocker": "identity_control_or_rollback_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-02 23:53:52 +0800",
  "held_test_usage": 0,
  "import_errors": [],
  "metric_winner_selection_used": 0,
  "ok_rows": 1,
  "part": "A",
  "route": "A_IdentityOrControlReadinessFailed",
  "row_count": 1,
  "runtime_selector_used": 0,
  "static_scan_hits": [
    {
      "check": "static_scan",
      "file": "experiments/run_v23_02_predictive_trust_functional_population_flow.py",
      "match": "select_metric("
    },
    {
      "check": "static_scan",
      "file": "experiments/run_v23_02_predictive_trust_functional_population_flow.py",
      "match": "choose_metric("
    },
    {
      "check": "static_scan",
      "file": "experiments/run_v23_02_predictive_trust_functional_population_flow.py",
      "match": "select_update("
    },
    {
      "check": "static_scan",
      "file": "experiments/run_v23_02_predictive_trust_functional_population_flow.py",
      "match": "choose_update("
    },
    {
      "check": "static_scan",
      "file": "experiments/run_v23_02_predictive_trust_functional_population_flow.py",
      "match": "make_external_product_feature("
    }
  ],
  "used_fake_data_rows": 0
}
```

## 2026-07-02 23:56:24 +0800 Part B history boundary lock

```json
{
  "candidate_update_selection_used": 0,
  "dominant_blocker": "none",
  "error_rows": 0,
  "gate_pass": 1,
  "generated_at": "2026-07-02 23:56:24 +0800",
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 19,
  "part": "B",
  "route": "B_HistoryBoundaryLockPass",
  "row_count": 19,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0,
  "v22_94_F5_best": 0,
  "v22_94_decomposition_pass": 1,
  "v22_94_ordinary_c2_opened": 0,
  "v22_94_tailsafe_kills_C2": "F_SkippedBecausePartEFailed",
  "v22_94_witness_pass": 1,
  "v23_01_E4_FunctionalGram_BlockSNR_s0_C2_coverage": 0.08936140247351432,
  "v23_01_E4_FunctionalGram_BlockSNR_s0_random_gap": 0.042837288723603706,
  "v23_01_E5_FunctionalGram_BlockSNR_degree_s0_C2_coverage": 0.08936140247351432,
  "v23_01_E5_FunctionalGram_BlockSNR_degree_s0_random_gap": 0.042837288723603706,
  "v23_01_F2_accept_rate": 0.05,
  "v23_01_F2_scale_mean": 0.00625,
  "v23_01_final_promotion_allowed": 0,
  "v23_01_local_patch_interaction_median": -0.10533262381431996,
  "v23_01_part_e_route": "E_C2FormationPass",
  "v23_01_part_f_route": "F_SafetyDiagnosticOnlyHeavyRejection",
  "v23_01_part_g_route": "G_PositiveControlFailed",
  "v23_01_repair_f_diag_guard_every2_f2_F5": 0,
  "v23_01_repair_f_diag_guard_every3_f2_F5": 0,
  "v23_01_rotation_sensitive_median": 0.23397090542130172
}
```

## 2026-07-02 23:56:25 +0800 Part A identity/control readiness

```json
{
  "candidate_update_selection_used": 0,
  "compile_errors": [],
  "dominant_blocker": "none",
  "error_rows": 0,
  "gate_pass": 1,
  "generated_at": "2026-07-02 23:56:25 +0800",
  "held_test_usage": 0,
  "import_errors": [],
  "metric_winner_selection_used": 0,
  "ok_rows": 1,
  "part": "A",
  "route": "A_Pass",
  "row_count": 1,
  "runtime_selector_used": 0,
  "static_scan_hits": [],
  "used_fake_data_rows": 0
}
```

## 2026-07-02 23:58:02 +0800 Part C taskwise C2 formation

```json
{
  "candidate_update_selection_used": 0,
  "dominant_blocker": "taskwise_c2_or_random_gap_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-02 23:58:02 +0800",
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 4,
  "part": "C",
  "passing_scheme_groups": [],
  "route": "C_C2FormationTaskwiseFailed",
  "row_count": 4,
  "runtime_selector_used": 0,
  "scheme_groups": [
    {
      "C2_accuracy_improvement_median": 0.1171875,
      "C2_coverage_improvement_median": -0.1758493612287566,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7420436243216197,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": 0.0012253203894942999,
      "rows": 2,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.1171875,
      "C2_coverage_improvement_median": -0.1770746816182509,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7409230475624402,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": 0.0,
      "rows": 2,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "taskwise_all_pass": 0
    }
  ],
  "taskwise_groups": [
    {
      "C2_accuracy_improvement_median": 0.03125,
      "C2_coverage_improvement_median": -0.1903495555743575,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7368283907572428,
      "gate_floor": 0.3,
      "random_gap": -0.0020756127778440714,
      "rows": 1,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.203125,
      "C2_coverage_improvement_median": -0.1613491668831557,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7472588578859966,
      "gate_floor": 0.3,
      "random_gap": 0.004526253556832671,
      "rows": 1,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.0,
      "C2_coverage_improvement_median": -0.18827394279651344,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7369118561347324,
      "gate_floor": 0.3,
      "random_gap": 0.0,
      "rows": 1,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.234375,
      "C2_coverage_improvement_median": -0.16587542043998837,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.744934238990148,
      "gate_floor": 0.3,
      "random_gap": 0.0,
      "rows": 1,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    }
  ],
  "used_fake_data_rows": 0
}
```

## 2026-07-03 00:08:54 +0800 Part C taskwise C2 formation

```json
{
  "candidate_update_selection_used": 0,
  "dominant_blocker": "taskwise_c2_or_random_gap_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 00:08:54 +0800",
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 180,
  "part": "C",
  "passing_scheme_groups": [],
  "route": "C_C2FormationTaskwiseFailed",
  "row_count": 180,
  "runtime_selector_used": 0,
  "scheme_groups": [
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.013416306973340397,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": -0.2188375259293025,
      "rows": 10,
      "scheme": "C0_AdamW_control",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.013416306973340397,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_gap": -0.21797225170303136,
      "rows": 10,
      "scheme": "C0_AdamW_control",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.50390625,
      "C2_coverage_improvement_median": 0.09553816406014448,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": -0.10988305489581762,
      "rows": 10,
      "scheme": "C1_FunctionalGram_AdamW_s0_no_Qpop",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.50390625,
      "C2_coverage_improvement_median": 0.09553816406014448,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_gap": -0.10901778066954648,
      "rows": 10,
      "scheme": "C1_FunctionalGram_AdamW_s0_no_Qpop",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.53125,
      "C2_coverage_improvement_median": 0.2225333244759895,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5600951047986744,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": 0.0171121055200274,
      "rows": 10,
      "scheme": "C2_FunctionalGram_DiagonalSNR_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.51953125,
      "C2_coverage_improvement_median": 0.2023982469954717,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6175932097683352,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_gap": -0.002157697734219255,
      "rows": 10,
      "scheme": "C2_FunctionalGram_DiagonalSNR_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.52734375,
      "C2_coverage_improvement_median": 0.16312713510114918,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7388204566140972,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": -0.042294083854812925,
      "rows": 10,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": 0.016768410238000797,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7724817405641078,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_gap": -0.18778753449169017,
      "rows": 10,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.52734375,
      "C2_coverage_improvement_median": 0.16312713510114918,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7388204566140972,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": -0.042294083854812925,
      "rows": 10,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": 0.016768410238000797,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7724817405641078,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_gap": -0.18778753449169017,
      "rows": 10,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.53125,
      "C2_coverage_improvement_median": 0.19761323678631015,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.617393665711085,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": -0.00780798216965195,
      "rows": 10,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5234375,
      "C2_coverage_improvement_median": 0.20769645571090223,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6742886504530907,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_gap": 0.003140510981211264,
      "rows": 10,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5234375,
      "C2_coverage_improvement_median": 0.2054212189559621,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5742894908289116,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": 0.0,
      "rows": 10,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5234375,
      "C2_coverage_improvement_median": 0.20455594472969096,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6314434118072194,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_gap": 0.0,
      "rows": 10,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.0,
      "C2_coverage_improvement_median": 0.0,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.0,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": -0.2054212189559621,
      "rows": 10,
      "scheme": "C8_FunctionalGram_SameComputeNoOp_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.0,
      "C2_coverage_improvement_median": 0.0,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.0,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_gap": -0.20455594472969096,
      "rows": 10,
      "scheme": "C8_FunctionalGram_SameComputeNoOp_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.26171875,
      "C2_coverage_improvement_median": -0.06121773690756527,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": -0.26663895586352737,
      "rows": 10,
      "scheme": "C9_FunctionalGram_s1_derivative_diagnostic",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.26171875,
      "C2_coverage_improvement_median": -0.06121773690756527,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_gap": -0.26577368163725623,
      "rows": 10,
      "scheme": "C9_FunctionalGram_s1_derivative_diagnostic",
      "taskwise_all_pass": 0
    }
  ],
  "taskwise_groups": [
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.022159406968967232,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.3,
      "random_gap": 0.02144437548577116,
      "rows": 5,
      "scheme": "C0_AdamW_control",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5390625,
      "C2_coverage_improvement_median": 0.2505487341309163,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.3,
      "random_gap": -0.0485914552068607,
      "rows": 5,
      "scheme": "C0_AdamW_control",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.022159406968967232,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.4,
      "random_gap": 0.059216575086793455,
      "rows": 5,
      "scheme": "C0_AdamW_control",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5390625,
      "C2_coverage_improvement_median": 0.2505487341309163,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.4,
      "random_gap": -0.028206773373767646,
      "rows": 5,
      "scheme": "C0_AdamW_control",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.3125,
      "C2_coverage_improvement_median": -0.1720952183748068,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.3,
      "random_gap": -0.1284914359200684,
      "rows": 5,
      "scheme": "C1_FunctionalGram_AdamW_s0_no_Qpop",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5546875,
      "C2_coverage_improvement_median": 0.23184427102387417,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.3,
      "random_gap": -0.0672959183139028,
      "rows": 5,
      "scheme": "C1_FunctionalGram_AdamW_s0_no_Qpop",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.3125,
      "C2_coverage_improvement_median": -0.1720952183748068,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.4,
      "random_gap": -0.0907192363190461,
      "rows": 5,
      "scheme": "C1_FunctionalGram_AdamW_s0_no_Qpop",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5546875,
      "C2_coverage_improvement_median": 0.23184427102387417,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.4,
      "random_gap": -0.04691123648080975,
      "rows": 5,
      "scheme": "C1_FunctionalGram_AdamW_s0_no_Qpop",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.4921875,
      "C2_coverage_improvement_median": -0.02546280545357149,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.595149854570627,
      "gate_floor": 0.3,
      "random_gap": 0.018140977001166902,
      "rows": 5,
      "scheme": "C2_FunctionalGram_DiagonalSNR_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.578125,
      "C2_coverage_improvement_median": 0.3281784041028004,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5551764302452402,
      "gate_floor": 0.3,
      "random_gap": 0.02903821476502344,
      "rows": 5,
      "scheme": "C2_FunctionalGram_DiagonalSNR_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.006384089303537621,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6446822438140718,
      "gate_floor": 0.4,
      "random_gap": 0.07499189275222307,
      "rows": 5,
      "scheme": "C2_FunctionalGram_DiagonalSNR_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5625,
      "C2_coverage_improvement_median": 0.2932576275525207,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6154300943017005,
      "gate_floor": 0.4,
      "random_gap": 0.014502120047836797,
      "rows": 5,
      "scheme": "C2_FunctionalGram_DiagonalSNR_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.4921875,
      "C2_coverage_improvement_median": 0.04401013016104116,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7381580670674646,
      "gate_floor": 0.3,
      "random_gap": 0.08761391261577955,
      "rows": 5,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.5859375,
      "C2_coverage_improvement_median": 0.25256247026118217,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7401545042296257,
      "gate_floor": 0.3,
      "random_gap": -0.046577719076594803,
      "rows": 5,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.46875,
      "C2_coverage_improvement_median": -0.0710737795152454,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7715271794795991,
      "gate_floor": 0.4,
      "random_gap": 0.010302202540515282,
      "rows": 5,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5859375,
      "C2_coverage_improvement_median": 0.3768447985603416,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7755642925699554,
      "gate_floor": 0.4,
      "random_gap": 0.09808929105565767,
      "rows": 5,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.4921875,
      "C2_coverage_improvement_median": 0.04401013016104116,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7381580670674646,
      "gate_floor": 0.3,
      "random_gap": 0.08761391261577955,
      "rows": 5,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.5859375,
      "C2_coverage_improvement_median": 0.25256247026118217,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7401545042296257,
      "gate_floor": 0.3,
      "random_gap": -0.046577719076594803,
      "rows": 5,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.46875,
      "C2_coverage_improvement_median": -0.0710737795152454,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7715271794795991,
      "gate_floor": 0.4,
      "random_gap": 0.010302202540515282,
      "rows": 5,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5859375,
      "C2_coverage_improvement_median": 0.3768447985603416,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7755642925699554,
      "gate_floor": 0.4,
      "random_gap": 0.09808929105565767,
      "rows": 5,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.04121152313018683,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6388773581385613,
      "gate_floor": 0.3,
      "random_gap": 0.00239225932455156,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.59375,
      "C2_coverage_improvement_median": 0.4106036140528886,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6136607921620211,
      "gate_floor": 0.3,
      "random_gap": 0.11146342471511161,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.4921875,
      "C2_coverage_improvement_median": -0.009335888214991428,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6829593942066033,
      "gate_floor": 0.4,
      "random_gap": 0.07204009384076926,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.6015625,
      "C2_coverage_improvement_median": 0.3614147638436407,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6619330755869548,
      "gate_floor": 0.4,
      "random_gap": 0.08265925633895677,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.04360378245473839,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5943124534686405,
      "gate_floor": 0.3,
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5625,
      "C2_coverage_improvement_median": 0.299140189337777,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5715550301472347,
      "gate_floor": 0.3,
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.4609375,
      "C2_coverage_improvement_median": -0.08137598205576069,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6517065076529976,
      "gate_floor": 0.4,
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5859375,
      "C2_coverage_improvement_median": 0.2787555075046839,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6264173669616379,
      "gate_floor": 0.4,
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.0,
      "C2_coverage_improvement_median": 0.0,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.0,
      "gate_floor": 0.3,
      "random_gap": 0.04360378245473839,
      "rows": 5,
      "scheme": "C8_FunctionalGram_SameComputeNoOp_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.0,
      "C2_coverage_improvement_median": 0.0,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.0,
      "gate_floor": 0.3,
      "random_gap": -0.299140189337777,
      "rows": 5,
      "scheme": "C8_FunctionalGram_SameComputeNoOp_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.0,
      "C2_coverage_improvement_median": 0.0,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.0,
      "gate_floor": 0.4,
      "random_gap": 0.08137598205576069,
      "rows": 5,
      "scheme": "C8_FunctionalGram_SameComputeNoOp_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.0,
      "C2_coverage_improvement_median": 0.0,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.0,
      "gate_floor": 0.4,
      "random_gap": -0.2787555075046839,
      "rows": 5,
      "scheme": "C8_FunctionalGram_SameComputeNoOp_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.1484375,
      "C2_coverage_improvement_median": -0.10302948227035813,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.3,
      "random_gap": -0.05942569981561974,
      "rows": 5,
      "scheme": "C9_FunctionalGram_s1_derivative_diagnostic",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.3125,
      "C2_coverage_improvement_median": -0.005917941220104694,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.3,
      "random_gap": -0.30505813055788167,
      "rows": 5,
      "scheme": "C9_FunctionalGram_s1_derivative_diagnostic",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.1484375,
      "C2_coverage_improvement_median": -0.10302948227035813,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.4,
      "random_gap": -0.021653500214597443,
      "rows": 5,
      "scheme": "C9_FunctionalGram_s1_derivative_diagnostic",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.3125,
      "C2_coverage_improvement_median": -0.005917941220104694,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.4,
      "random_gap": -0.2846734487247886,
      "rows": 5,
      "scheme": "C9_FunctionalGram_s1_derivative_diagnostic",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    }
  ],
  "used_fake_data_rows": 0
}
```

## 2026-07-03 00:11:21 +0800 Part C taskwise C2 formation

```json
{
  "candidate_update_selection_used": 0,
  "dominant_blocker": "taskwise_c2_or_random_gap_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 00:11:21 +0800",
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 180,
  "part": "C",
  "passing_scheme_groups": [],
  "route": "C_C2FormationTaskwiseFailed",
  "row_count": 180,
  "runtime_selector_used": 0,
  "scheme_groups": [
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.013416306973340397,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": -0.2188375259293025,
      "rows": 10,
      "scheme": "C0_AdamW_control",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.013416306973340397,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_gap": -0.21797225170303136,
      "rows": 10,
      "scheme": "C0_AdamW_control",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.50390625,
      "C2_coverage_improvement_median": 0.09553816406014448,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": -0.10988305489581762,
      "rows": 10,
      "scheme": "C1_FunctionalGram_AdamW_s0_no_Qpop",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.50390625,
      "C2_coverage_improvement_median": 0.09553816406014448,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_gap": -0.10901778066954648,
      "rows": 10,
      "scheme": "C1_FunctionalGram_AdamW_s0_no_Qpop",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.53125,
      "C2_coverage_improvement_median": 0.2225333244759895,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5600951047986744,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": 0.0171121055200274,
      "rows": 10,
      "scheme": "C2_FunctionalGram_DiagonalSNR_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.51953125,
      "C2_coverage_improvement_median": 0.2023982469954717,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6175932097683352,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_gap": -0.002157697734219255,
      "rows": 10,
      "scheme": "C2_FunctionalGram_DiagonalSNR_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.52734375,
      "C2_coverage_improvement_median": 0.16312713510114918,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7388204566140972,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": -0.042294083854812925,
      "rows": 10,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": 0.016768410238000797,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7724817405641078,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_gap": -0.18778753449169017,
      "rows": 10,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.52734375,
      "C2_coverage_improvement_median": 0.16312713510114918,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7388204566140972,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": -0.042294083854812925,
      "rows": 10,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": 0.016768410238000797,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7724817405641078,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_gap": -0.18778753449169017,
      "rows": 10,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.53125,
      "C2_coverage_improvement_median": 0.19761323678631015,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.617393665711085,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": -0.00780798216965195,
      "rows": 10,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5234375,
      "C2_coverage_improvement_median": 0.20769645571090223,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6742886504530907,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_gap": 0.003140510981211264,
      "rows": 10,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5234375,
      "C2_coverage_improvement_median": 0.2054212189559621,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5742894908289116,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": 0.0,
      "rows": 10,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5234375,
      "C2_coverage_improvement_median": 0.20455594472969096,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6314434118072194,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_gap": 0.0,
      "rows": 10,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.0,
      "C2_coverage_improvement_median": 0.0,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.0,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": -0.2054212189559621,
      "rows": 10,
      "scheme": "C8_FunctionalGram_SameComputeNoOp_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.0,
      "C2_coverage_improvement_median": 0.0,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.0,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_gap": -0.20455594472969096,
      "rows": 10,
      "scheme": "C8_FunctionalGram_SameComputeNoOp_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.26171875,
      "C2_coverage_improvement_median": -0.06121773690756527,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": -0.26663895586352737,
      "rows": 10,
      "scheme": "C9_FunctionalGram_s1_derivative_diagnostic",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.26171875,
      "C2_coverage_improvement_median": -0.06121773690756527,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_gap": -0.26577368163725623,
      "rows": 10,
      "scheme": "C9_FunctionalGram_s1_derivative_diagnostic",
      "taskwise_all_pass": 0
    }
  ],
  "taskwise_groups": [
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.022159406968967232,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.3,
      "random_gap": 0.02144437548577116,
      "rows": 5,
      "scheme": "C0_AdamW_control",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5390625,
      "C2_coverage_improvement_median": 0.2505487341309163,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.3,
      "random_gap": -0.0485914552068607,
      "rows": 5,
      "scheme": "C0_AdamW_control",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.022159406968967232,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.4,
      "random_gap": 0.059216575086793455,
      "rows": 5,
      "scheme": "C0_AdamW_control",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5390625,
      "C2_coverage_improvement_median": 0.2505487341309163,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.4,
      "random_gap": -0.028206773373767646,
      "rows": 5,
      "scheme": "C0_AdamW_control",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.3125,
      "C2_coverage_improvement_median": -0.1720952183748068,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.3,
      "random_gap": -0.1284914359200684,
      "rows": 5,
      "scheme": "C1_FunctionalGram_AdamW_s0_no_Qpop",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5546875,
      "C2_coverage_improvement_median": 0.23184427102387417,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.3,
      "random_gap": -0.0672959183139028,
      "rows": 5,
      "scheme": "C1_FunctionalGram_AdamW_s0_no_Qpop",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.3125,
      "C2_coverage_improvement_median": -0.1720952183748068,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.4,
      "random_gap": -0.0907192363190461,
      "rows": 5,
      "scheme": "C1_FunctionalGram_AdamW_s0_no_Qpop",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5546875,
      "C2_coverage_improvement_median": 0.23184427102387417,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.4,
      "random_gap": -0.04691123648080975,
      "rows": 5,
      "scheme": "C1_FunctionalGram_AdamW_s0_no_Qpop",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.4921875,
      "C2_coverage_improvement_median": -0.02546280545357149,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.595149854570627,
      "gate_floor": 0.3,
      "random_gap": 0.018140977001166902,
      "rows": 5,
      "scheme": "C2_FunctionalGram_DiagonalSNR_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.578125,
      "C2_coverage_improvement_median": 0.3281784041028004,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5551764302452402,
      "gate_floor": 0.3,
      "random_gap": 0.02903821476502344,
      "rows": 5,
      "scheme": "C2_FunctionalGram_DiagonalSNR_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.006384089303537621,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6446822438140718,
      "gate_floor": 0.4,
      "random_gap": 0.07499189275222307,
      "rows": 5,
      "scheme": "C2_FunctionalGram_DiagonalSNR_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5625,
      "C2_coverage_improvement_median": 0.2932576275525207,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6154300943017005,
      "gate_floor": 0.4,
      "random_gap": 0.014502120047836797,
      "rows": 5,
      "scheme": "C2_FunctionalGram_DiagonalSNR_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.4921875,
      "C2_coverage_improvement_median": 0.04401013016104116,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7381580670674646,
      "gate_floor": 0.3,
      "random_gap": 0.08761391261577955,
      "rows": 5,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.5859375,
      "C2_coverage_improvement_median": 0.25256247026118217,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7401545042296257,
      "gate_floor": 0.3,
      "random_gap": -0.046577719076594803,
      "rows": 5,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.46875,
      "C2_coverage_improvement_median": -0.0710737795152454,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7715271794795991,
      "gate_floor": 0.4,
      "random_gap": 0.010302202540515282,
      "rows": 5,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5859375,
      "C2_coverage_improvement_median": 0.3768447985603416,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7755642925699554,
      "gate_floor": 0.4,
      "random_gap": 0.09808929105565767,
      "rows": 5,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.4921875,
      "C2_coverage_improvement_median": 0.04401013016104116,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7381580670674646,
      "gate_floor": 0.3,
      "random_gap": 0.08761391261577955,
      "rows": 5,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.5859375,
      "C2_coverage_improvement_median": 0.25256247026118217,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7401545042296257,
      "gate_floor": 0.3,
      "random_gap": -0.046577719076594803,
      "rows": 5,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.46875,
      "C2_coverage_improvement_median": -0.0710737795152454,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7715271794795991,
      "gate_floor": 0.4,
      "random_gap": 0.010302202540515282,
      "rows": 5,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5859375,
      "C2_coverage_improvement_median": 0.3768447985603416,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7755642925699554,
      "gate_floor": 0.4,
      "random_gap": 0.09808929105565767,
      "rows": 5,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.04121152313018683,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6388773581385613,
      "gate_floor": 0.3,
      "random_gap": 0.00239225932455156,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.59375,
      "C2_coverage_improvement_median": 0.4106036140528886,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6136607921620211,
      "gate_floor": 0.3,
      "random_gap": 0.11146342471511161,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.4921875,
      "C2_coverage_improvement_median": -0.009335888214991428,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6829593942066033,
      "gate_floor": 0.4,
      "random_gap": 0.07204009384076926,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.6015625,
      "C2_coverage_improvement_median": 0.3614147638436407,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6619330755869548,
      "gate_floor": 0.4,
      "random_gap": 0.08265925633895677,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.04360378245473839,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5943124534686405,
      "gate_floor": 0.3,
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5625,
      "C2_coverage_improvement_median": 0.299140189337777,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5715550301472347,
      "gate_floor": 0.3,
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.4609375,
      "C2_coverage_improvement_median": -0.08137598205576069,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6517065076529976,
      "gate_floor": 0.4,
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5859375,
      "C2_coverage_improvement_median": 0.2787555075046839,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6264173669616379,
      "gate_floor": 0.4,
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.0,
      "C2_coverage_improvement_median": 0.0,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.0,
      "gate_floor": 0.3,
      "random_gap": 0.04360378245473839,
      "rows": 5,
      "scheme": "C8_FunctionalGram_SameComputeNoOp_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.0,
      "C2_coverage_improvement_median": 0.0,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.0,
      "gate_floor": 0.3,
      "random_gap": -0.299140189337777,
      "rows": 5,
      "scheme": "C8_FunctionalGram_SameComputeNoOp_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.0,
      "C2_coverage_improvement_median": 0.0,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.0,
      "gate_floor": 0.4,
      "random_gap": 0.08137598205576069,
      "rows": 5,
      "scheme": "C8_FunctionalGram_SameComputeNoOp_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.0,
      "C2_coverage_improvement_median": 0.0,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.0,
      "gate_floor": 0.4,
      "random_gap": -0.2787555075046839,
      "rows": 5,
      "scheme": "C8_FunctionalGram_SameComputeNoOp_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.1484375,
      "C2_coverage_improvement_median": -0.10302948227035813,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.3,
      "random_gap": -0.05942569981561974,
      "rows": 5,
      "scheme": "C9_FunctionalGram_s1_derivative_diagnostic",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.3125,
      "C2_coverage_improvement_median": -0.005917941220104694,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.3,
      "random_gap": -0.30505813055788167,
      "rows": 5,
      "scheme": "C9_FunctionalGram_s1_derivative_diagnostic",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.1484375,
      "C2_coverage_improvement_median": -0.10302948227035813,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.4,
      "random_gap": -0.021653500214597443,
      "rows": 5,
      "scheme": "C9_FunctionalGram_s1_derivative_diagnostic",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.3125,
      "C2_coverage_improvement_median": -0.005917941220104694,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 1.0,
      "gate_floor": 0.4,
      "random_gap": -0.2846734487247886,
      "rows": 5,
      "scheme": "C9_FunctionalGram_s1_derivative_diagnostic",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    }
  ],
  "used_fake_data_rows": 0
}
```

## 2026-07-03 00:27:00 +0800 Part C taskwise C2 formation

```json
{
  "candidate_update_selection_used": 0,
  "dominant_blocker": "taskwise_c2_or_random_gap_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 00:27:00 +0800",
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 120,
  "part": "C",
  "passing_scheme_groups": [],
  "route": "C_C2FormationTaskwiseFailed",
  "row_count": 120,
  "runtime_selector_used": 0,
  "scheme_groups": [
    {
      "C2_accuracy_improvement_median": 0.515625,
      "C2_coverage_improvement_median": 0.15960431391658858,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.742208830929465,
      "gate_floor": 0.32,
      "official_pass": 0,
      "random_gap": -0.03453083946396873,
      "rows": 10,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.515625,
      "C2_coverage_improvement_median": 0.1542484033607252,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7520062084413233,
      "gate_floor": 0.35,
      "official_pass": 0,
      "random_gap": -0.03818176750337443,
      "rows": 10,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5078125,
      "C2_coverage_improvement_median": 0.09772646660985629,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7589993000858358,
      "gate_floor": 0.37,
      "official_pass": 0,
      "random_gap": -0.060812751291450695,
      "rows": 10,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.515625,
      "C2_coverage_improvement_median": 0.15960431391658858,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.742208830929465,
      "gate_floor": 0.32,
      "official_pass": 0,
      "random_gap": -0.03453083946396873,
      "rows": 10,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.515625,
      "C2_coverage_improvement_median": 0.1542484033607252,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7520062084413233,
      "gate_floor": 0.35,
      "official_pass": 0,
      "random_gap": -0.03818176750337443,
      "rows": 10,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5078125,
      "C2_coverage_improvement_median": 0.09772646660985629,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7589993000858358,
      "gate_floor": 0.37,
      "official_pass": 0,
      "random_gap": -0.060812751291450695,
      "rows": 10,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.52734375,
      "C2_coverage_improvement_median": 0.20545441875674442,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5987014381421938,
      "gate_floor": 0.32,
      "official_pass": 0,
      "random_gap": 0.011319265376187104,
      "rows": 10,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5234375,
      "C2_coverage_improvement_median": 0.23298060026445455,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6171186923773753,
      "gate_floor": 0.35,
      "official_pass": 0,
      "random_gap": 0.040550429400354915,
      "rows": 10,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.51171875,
      "C2_coverage_improvement_median": 0.20033139481074613,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6272512699166933,
      "gate_floor": 0.37,
      "official_pass": 0,
      "random_gap": 0.04179217690943915,
      "rows": 10,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.52734375,
      "C2_coverage_improvement_median": 0.1941351533805573,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5587228533087503,
      "gate_floor": 0.32,
      "official_pass": 0,
      "random_gap": 0.0,
      "rows": 10,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.53515625,
      "C2_coverage_improvement_median": 0.19243017086409964,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5757052010132209,
      "gate_floor": 0.35,
      "official_pass": 0,
      "random_gap": 0.0,
      "rows": 10,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.53515625,
      "C2_coverage_improvement_median": 0.15853921790130698,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5918575508726969,
      "gate_floor": 0.37,
      "official_pass": 0,
      "random_gap": 0.0,
      "rows": 10,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "taskwise_all_pass": 0
    }
  ],
  "taskwise_groups": [
    {
      "C2_accuracy_improvement_median": 0.4453125,
      "C2_coverage_improvement_median": -0.044726319814799353,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7406164024439126,
      "gate_floor": 0.32,
      "random_gap": 0.008363818888028618,
      "rows": 5,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5859375,
      "C2_coverage_improvement_median": 0.36002164553065086,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.742757442469398,
      "gate_floor": 0.32,
      "random_gap": 0.028178139102237765,
      "rows": 5,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.46875,
      "C2_coverage_improvement_median": -0.0022102527000242844,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7506580443018013,
      "gate_floor": 0.35,
      "random_gap": 0.01590482243045699,
      "rows": 5,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5546875,
      "C2_coverage_improvement_median": 0.3022923715179786,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.754577949560351,
      "gate_floor": 0.35,
      "random_gap": -0.0005887603183509782,
      "rows": 5,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": 0.00995896446693223,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7575208225184014,
      "gate_floor": 0.37,
      "random_gap": -0.055148612962511834,
      "rows": 5,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.6015625,
      "C2_coverage_improvement_median": 0.44086906337179244,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7622905483676331,
      "gate_floor": 0.37,
      "random_gap": 0.10356258857063949,
      "rows": 5,
      "scheme": "C3_FunctionalGram_BlockSNR_layer_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.4453125,
      "C2_coverage_improvement_median": -0.044726319814799353,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7406164024439126,
      "gate_floor": 0.32,
      "random_gap": 0.008363818888028618,
      "rows": 5,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5859375,
      "C2_coverage_improvement_median": 0.36002164553065086,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.742757442469398,
      "gate_floor": 0.32,
      "random_gap": 0.028178139102237765,
      "rows": 5,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.46875,
      "C2_coverage_improvement_median": -0.0022102527000242844,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7506580443018013,
      "gate_floor": 0.35,
      "random_gap": 0.01590482243045699,
      "rows": 5,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5546875,
      "C2_coverage_improvement_median": 0.3022923715179786,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.754577949560351,
      "gate_floor": 0.35,
      "random_gap": -0.0005887603183509782,
      "rows": 5,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": 0.00995896446693223,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7575208225184014,
      "gate_floor": 0.37,
      "random_gap": -0.055148612962511834,
      "rows": 5,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.6015625,
      "C2_coverage_improvement_median": 0.44086906337179244,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7622905483676331,
      "gate_floor": 0.37,
      "random_gap": 0.10356258857063949,
      "rows": 5,
      "scheme": "C4_FunctionalGram_BlockSNR_degree_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.020059306158145773,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6176398488796416,
      "gate_floor": 0.32,
      "random_gap": 0.0330308325446822,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.578125,
      "C2_coverage_improvement_median": 0.3157287054054905,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5957046620133849,
      "gate_floor": 0.32,
      "random_gap": -0.01611480102292262,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.4375,
      "C2_coverage_improvement_median": -0.03992684016338899,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6310253197534222,
      "gate_floor": 0.35,
      "random_gap": -0.021811765032907715,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.578125,
      "C2_coverage_improvement_median": 0.3640085997758433,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6112302595128618,
      "gate_floor": 0.35,
      "random_gap": 0.061127467939513735,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.007848862631362863,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6429530199617152,
      "gate_floor": 0.37,
      "random_gap": -0.07295644006080693,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.609375,
      "C2_coverage_improvement_median": 0.39572704915190116,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.622405069362786,
      "gate_floor": 0.37,
      "random_gap": 0.05842057435074821,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.453125,
      "C2_coverage_improvement_median": -0.05309013870282797,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5841724356429444,
      "gate_floor": 0.32,
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5546875,
      "C2_coverage_improvement_median": 0.3318435064284131,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5561934719069136,
      "gate_floor": 0.32,
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.4921875,
      "C2_coverage_improvement_median": -0.018115075130481273,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6063622005697755,
      "gate_floor": 0.35,
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5625,
      "C2_coverage_improvement_median": 0.3028811318363296,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5735111739900375,
      "gate_floor": 0.35,
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5,
      "C2_coverage_improvement_median": 0.06510757742944406,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5997221902840667,
      "gate_floor": 0.37,
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.578125,
      "C2_coverage_improvement_median": 0.33730647480115294,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5857084357490145,
      "gate_floor": 0.37,
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    }
  ],
  "used_fake_data_rows": 0
}
```

## 2026-07-03 00:29:43 +0800 Part C taskwise C2 formation

```json
{
  "candidate_update_selection_used": 0,
  "dominant_blocker": "taskwise_c2_or_random_gap_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 00:29:43 +0800",
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 4,
  "part": "C",
  "passing_scheme_groups": [],
  "route": "C_C2FormationTaskwiseFailed",
  "row_count": 4,
  "runtime_selector_used": 0,
  "scheme_groups": [
    {
      "C2_accuracy_improvement_median": 0.1171875,
      "C2_coverage_improvement_median": -0.1381147353677079,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7434714416662853,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": 0.03439881675876677,
      "rows": 2,
      "scheme": "C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.09375,
      "C2_coverage_improvement_median": -0.17251355212647468,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7408410400152207,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_gap": 0.0,
      "rows": 2,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "taskwise_all_pass": 0
    }
  ],
  "taskwise_groups": [
    {
      "C2_accuracy_improvement_median": 0.046875,
      "C2_coverage_improvement_median": -0.20149343204684556,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7404051899909974,
      "gate_floor": 0.3,
      "random_gap": -0.018963699461892247,
      "rows": 1,
      "scheme": "C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.1875,
      "C2_coverage_improvement_median": -0.07473603868857026,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7465376933415733,
      "gate_floor": 0.3,
      "random_gap": 0.08776133297942579,
      "rows": 1,
      "scheme": "C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.015625,
      "C2_coverage_improvement_median": -0.1825297325849533,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7376690546671549,
      "gate_floor": 0.3,
      "random_gap": 0.0,
      "rows": 1,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.171875,
      "C2_coverage_improvement_median": -0.16249737166799605,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.7440130253632864,
      "gate_floor": 0.3,
      "random_gap": 0.0,
      "rows": 1,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    }
  ],
  "used_fake_data_rows": 0
}
```

## 2026-07-03 01:03:34 +0800 Part C taskwise C2 formation

```json
{
  "candidate_update_selection_used": 0,
  "dominant_blocker": "taskwise_c2_or_random_gap_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 01:03:34 +0800",
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 180,
  "part": "C",
  "passing_scheme_groups": [],
  "route": "C_C2FormationTaskwiseFailed",
  "row_count": 180,
  "runtime_selector_used": 0,
  "scheme_groups": [
    {
      "C2_accuracy_improvement_median": 0.51171875,
      "C2_coverage_improvement_median": 0.057514577923939214,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6050186137151383,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_control_scheme": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
      "random_gap": 0.11297866600580164,
      "rows": 10,
      "scheme": "C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.47265625,
      "C2_coverage_improvement_median": 0.0025621782356211042,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6180154232308268,
      "gate_floor": 0.35,
      "official_pass": 0,
      "random_control_scheme": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
      "random_gap": 0.06914213063237185,
      "rows": 10,
      "scheme": "C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.4921875,
      "C2_coverage_improvement_median": -0.02547430050890398,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6393613741008773,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_control_scheme": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
      "random_gap": -0.13075031460448372,
      "rows": 10,
      "scheme": "C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5546875,
      "C2_coverage_improvement_median": 0.23740104381067795,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5855199228558279,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_control_scheme": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
      "random_gap": -0.09762595224310644,
      "rows": 10,
      "scheme": "C11_FunctionalSobolev_DegreeEdgebankSNR_s0p5",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.57421875,
      "C2_coverage_improvement_median": 0.24115184188485728,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6048496488688722,
      "gate_floor": 0.35,
      "official_pass": 0,
      "random_control_scheme": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
      "random_gap": 0.0666477116610622,
      "rows": 10,
      "scheme": "C11_FunctionalSobolev_DegreeEdgebankSNR_s0p5",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5390625,
      "C2_coverage_improvement_median": 0.22607380631598062,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6356918596145179,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_control_scheme": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
      "random_gap": 0.13273095046133676,
      "rows": 10,
      "scheme": "C11_FunctionalSobolev_DegreeEdgebankSNR_s0p5",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.390625,
      "C2_coverage_improvement_median": -0.06687771657016128,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5300801126700309,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_control_scheme": "C15_FunctionalSobolev_RandomMatchedGate_s1",
      "random_gap": 0.06025089272952755,
      "rows": 10,
      "scheme": "C12_FunctionalSobolev_DegreeEdgebankSNR_s1",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.265625,
      "C2_coverage_improvement_median": -0.1161280295564211,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5560370418139627,
      "gate_floor": 0.35,
      "official_pass": 0,
      "random_control_scheme": "C15_FunctionalSobolev_RandomMatchedGate_s1",
      "random_gap": 0.031281626641430194,
      "rows": 10,
      "scheme": "C12_FunctionalSobolev_DegreeEdgebankSNR_s1",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.4140625,
      "C2_coverage_improvement_median": -0.017912887189595494,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5949879554617733,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_control_scheme": "C15_FunctionalSobolev_RandomMatchedGate_s1",
      "random_gap": 0.1369524833178275,
      "rows": 10,
      "scheme": "C12_FunctionalSobolev_DegreeEdgebankSNR_s1",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.4453125,
      "C2_coverage_improvement_median": -0.05546408808186243,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5600085920136835,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.05546408808186243,
      "rows": 10,
      "scheme": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.44921875,
      "C2_coverage_improvement_median": -0.06657995239675074,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5931450143663419,
      "gate_floor": 0.35,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.06657995239675074,
      "rows": 10,
      "scheme": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.49609375,
      "C2_coverage_improvement_median": 0.10527601409557974,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6194175439576309,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.10527601409557974,
      "rows": 10,
      "scheme": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.57421875,
      "C2_coverage_improvement_median": 0.3350269960537844,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5600260704134901,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.3350269960537844,
      "rows": 10,
      "scheme": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.52734375,
      "C2_coverage_improvement_median": 0.17450413022379507,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.577767308284011,
      "gate_floor": 0.35,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.17450413022379507,
      "rows": 10,
      "scheme": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.4921875,
      "C2_coverage_improvement_median": 0.09334285585464386,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6131234028066199,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.09334285585464386,
      "rows": 10,
      "scheme": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.36328125,
      "C2_coverage_improvement_median": -0.12712860929968883,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.518939533850385,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.12712860929968883,
      "rows": 10,
      "scheme": "C15_FunctionalSobolev_RandomMatchedGate_s1",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.36328125,
      "C2_coverage_improvement_median": -0.1474096561978513,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5570511959700122,
      "gate_floor": 0.35,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.1474096561978513,
      "rows": 10,
      "scheme": "C15_FunctionalSobolev_RandomMatchedGate_s1",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.30859375,
      "C2_coverage_improvement_median": -0.154865370507423,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5933570970056788,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.154865370507423,
      "rows": 10,
      "scheme": "C15_FunctionalSobolev_RandomMatchedGate_s1",
      "taskwise_all_pass": 0
    }
  ],
  "taskwise_groups": [
    {
      "C2_accuracy_improvement_median": 0.3671875,
      "C2_coverage_improvement_median": -0.1821083694594563,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6037934131506411,
      "gate_floor": 0.3,
      "random_control_scheme": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
      "random_gap": -0.10903275215969188,
      "rows": 5,
      "scheme": "C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.65625,
      "C2_coverage_improvement_median": 0.4945798340049805,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6062438142796355,
      "gate_floor": 0.3,
      "random_control_scheme": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
      "random_gap": -0.08583364779769909,
      "rows": 5,
      "scheme": "C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.390625,
      "C2_coverage_improvement_median": -0.12648838751920266,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6235343474066919,
      "gate_floor": 0.35,
      "random_control_scheme": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
      "random_gap": 0.024437723179289605,
      "rows": 5,
      "scheme": "C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.6484375,
      "C2_coverage_improvement_median": 0.5690271660077997,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6124964990549616,
      "gate_floor": 0.35,
      "random_control_scheme": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
      "random_gap": 0.04346527096822683,
      "rows": 5,
      "scheme": "C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.34375,
      "C2_coverage_improvement_median": -0.16245085476111853,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6411614849335617,
      "gate_floor": 0.4,
      "random_control_scheme": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
      "random_gap": -0.046700534754563705,
      "rows": 5,
      "scheme": "C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.640625,
      "C2_coverage_improvement_median": 0.5505262564111035,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6359204081611505,
      "gate_floor": 0.4,
      "random_control_scheme": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
      "random_gap": -0.07625394806382246,
      "rows": 5,
      "scheme": "C10_FunctionalSobolev_DegreeEdgebankSNR_s0p25",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.3203125,
      "C2_coverage_improvement_median": -0.17813252921405365,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5855141288704343,
      "gate_floor": 0.3,
      "random_control_scheme": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
      "random_gap": -0.02550484933090047,
      "rows": 5,
      "scheme": "C11_FunctionalSobolev_DegreeEdgebankSNR_s0p5",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.59375,
      "C2_coverage_improvement_median": 0.24266555699432502,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5855257168412216,
      "gate_floor": 0.3,
      "random_control_scheme": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
      "random_gap": -0.15102150729944697,
      "rows": 5,
      "scheme": "C11_FunctionalSobolev_DegreeEdgebankSNR_s0p5",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.265625,
      "C2_coverage_improvement_median": -0.1449272356403526,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6053761880016992,
      "gate_floor": 0.35,
      "random_control_scheme": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
      "random_gap": 0.01198293858760735,
      "rows": 5,
      "scheme": "C11_FunctionalSobolev_DegreeEdgebankSNR_s0p5",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.6015625,
      "C2_coverage_improvement_median": 0.46292816969980777,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6043231097360451,
      "gate_floor": 0.35,
      "random_control_scheme": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
      "random_gap": 0.08862192533524649,
      "rows": 5,
      "scheme": "C11_FunctionalSobolev_DegreeEdgebankSNR_s0p5",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.34375,
      "C2_coverage_improvement_median": -0.17571369562574546,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6341921414352123,
      "gate_floor": 0.4,
      "random_control_scheme": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
      "random_gap": -0.02216278261403204,
      "rows": 5,
      "scheme": "C11_FunctionalSobolev_DegreeEdgebankSNR_s0p5",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.546875,
      "C2_coverage_improvement_median": 0.2428119194773899,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6414460454963975,
      "gate_floor": 0.4,
      "random_control_scheme": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
      "random_gap": 0.09106654693459859,
      "rows": 5,
      "scheme": "C11_FunctionalSobolev_DegreeEdgebankSNR_s0p5",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.234375,
      "C2_coverage_improvement_median": -0.18496060467441566,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5266886990517378,
      "gate_floor": 0.3,
      "random_control_scheme": "C15_FunctionalSobolev_RandomMatchedGate_s1",
      "random_gap": -0.044570503021532204,
      "rows": 5,
      "scheme": "C12_FunctionalSobolev_DegreeEdgebankSNR_s1",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.46875,
      "C2_coverage_improvement_median": 0.008991630347736645,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.533471526288324,
      "gate_floor": 0.3,
      "random_control_scheme": "C15_FunctionalSobolev_RandomMatchedGate_s1",
      "random_gap": 0.12285874729423085,
      "rows": 5,
      "scheme": "C12_FunctionalSobolev_DegreeEdgebankSNR_s1",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.1875,
      "C2_coverage_improvement_median": -0.1342895606667298,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5565760835177367,
      "gate_floor": 0.35,
      "random_control_scheme": "C15_FunctionalSobolev_RandomMatchedGate_s1",
      "random_gap": 0.11500318754224281,
      "rows": 5,
      "scheme": "C12_FunctionalSobolev_DegreeEdgebankSNR_s1",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.40625,
      "C2_coverage_improvement_median": -0.10887971489137271,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.54976377690004,
      "gate_floor": 0.35,
      "random_control_scheme": "C15_FunctionalSobolev_RandomMatchedGate_s1",
      "random_gap": -0.07737184948200593,
      "rows": 5,
      "scheme": "C12_FunctionalSobolev_DegreeEdgebankSNR_s1",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.359375,
      "C2_coverage_improvement_median": 0.007896749448264018,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5993867297967272,
      "gate_floor": 0.4,
      "random_control_scheme": "C15_FunctionalSobolev_RandomMatchedGate_s1",
      "random_gap": 0.2463842357537942,
      "rows": 5,
      "scheme": "C12_FunctionalSobolev_DegreeEdgebankSNR_s1",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.46875,
      "C2_coverage_improvement_median": -0.04372252382745501,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5905891811268194,
      "gate_floor": 0.4,
      "random_control_scheme": "C15_FunctionalSobolev_RandomMatchedGate_s1",
      "random_gap": -0.019192171486793086,
      "rows": 5,
      "scheme": "C12_FunctionalSobolev_DegreeEdgebankSNR_s1",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.4296875,
      "C2_coverage_improvement_median": -0.07307561729976442,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5657532379031184,
      "gate_floor": 0.3,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.07307561729976442,
      "rows": 5,
      "scheme": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.625,
      "C2_coverage_improvement_median": 0.5804134818026796,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5490831993106339,
      "gate_floor": 0.3,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.5804134818026796,
      "rows": 5,
      "scheme": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.3984375,
      "C2_coverage_improvement_median": -0.15092611069849227,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5987493323369159,
      "gate_floor": 0.35,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.15092611069849227,
      "rows": 5,
      "scheme": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.6328125,
      "C2_coverage_improvement_median": 0.5255618950395728,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5875406963957679,
      "gate_floor": 0.35,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.5255618950395728,
      "rows": 5,
      "scheme": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.4140625,
      "C2_coverage_improvement_median": -0.11575032000655483,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6236170914851955,
      "gate_floor": 0.4,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.11575032000655483,
      "rows": 5,
      "scheme": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.640625,
      "C2_coverage_improvement_median": 0.626780204474926,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6183848245690268,
      "gate_floor": 0.4,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.626780204474926,
      "rows": 5,
      "scheme": "C13_FunctionalSobolev_RandomMatchedGate_s0p25",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.2734375,
      "C2_coverage_improvement_median": -0.15262767988315318,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5627205110258527,
      "gate_floor": 0.3,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.15262767988315318,
      "rows": 5,
      "scheme": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.59375,
      "C2_coverage_improvement_median": 0.393687064293772,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5411680695911246,
      "gate_floor": 0.3,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.393687064293772,
      "rows": 5,
      "scheme": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.28125,
      "C2_coverage_improvement_median": -0.15691017422795994,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5846539518071548,
      "gate_floor": 0.35,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.15691017422795994,
      "rows": 5,
      "scheme": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5703125,
      "C2_coverage_improvement_median": 0.3743062443645613,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.570216222976645,
      "gate_floor": 0.35,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.3743062443645613,
      "rows": 5,
      "scheme": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.3359375,
      "C2_coverage_improvement_median": -0.15355091301171342,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6169343600670498,
      "gate_floor": 0.4,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.15355091301171342,
      "rows": 5,
      "scheme": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5078125,
      "C2_coverage_improvement_median": 0.1517453725427913,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6093124455461901,
      "gate_floor": 0.4,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.1517453725427913,
      "rows": 5,
      "scheme": "C14_FunctionalSobolev_RandomMatchedGate_s0p5",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.296875,
      "C2_coverage_improvement_median": -0.14039010165288346,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5322728600353,
      "gate_floor": 0.3,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.14039010165288346,
      "rows": 5,
      "scheme": "C15_FunctionalSobolev_RandomMatchedGate_s1",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.40625,
      "C2_coverage_improvement_median": -0.1138671169464942,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5188777773744532,
      "gate_floor": 0.3,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.1138671169464942,
      "rows": 5,
      "scheme": "C15_FunctionalSobolev_RandomMatchedGate_s1",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.2734375,
      "C2_coverage_improvement_median": -0.24929274820897263,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.556834323787027,
      "gate_floor": 0.35,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.24929274820897263,
      "rows": 5,
      "scheme": "C15_FunctionalSobolev_RandomMatchedGate_s1",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.4609375,
      "C2_coverage_improvement_median": -0.031507865409366786,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5572680681529973,
      "gate_floor": 0.35,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.031507865409366786,
      "rows": 5,
      "scheme": "C15_FunctionalSobolev_RandomMatchedGate_s1",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.265625,
      "C2_coverage_improvement_median": -0.23848748630553018,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5965218813882935,
      "gate_floor": 0.4,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.23848748630553018,
      "rows": 5,
      "scheme": "C15_FunctionalSobolev_RandomMatchedGate_s1",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.53125,
      "C2_coverage_improvement_median": -0.02453035234066192,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5901923126230642,
      "gate_floor": 0.4,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.02453035234066192,
      "rows": 5,
      "scheme": "C15_FunctionalSobolev_RandomMatchedGate_s1",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    }
  ],
  "used_fake_data_rows": 0
}
```

## 2026-07-03 01:10:41 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 01:10:41 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": -1.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 1.8793139457702637,
      "overhead_ratio_vs_F0": 0.7281094944527982,
      "random_veto_matched_gap": 0,
      "rows": 3,
      "safety_type": "F0_no_safety_control",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": -0.28406744028372616,
      "F5_no_debt_count": 1,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 1,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.15,
      "finite_step_scale_mean_median": 0.034375,
      "finite_step_skip_count_median": 17.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 6.968613386154175,
      "overhead_ratio_vs_F0": 2.699875441806559,
      "random_veto_matched_gap": 1,
      "rows": 3,
      "safety_type": "F2_finite_step_trust_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": -0.9997608988756166,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 10.60962462425232,
      "overhead_ratio_vs_F0": 4.110525779306196,
      "random_veto_matched_gap": 0,
      "rows": 3,
      "safety_type": "F6_random_veto_matched_control",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 9,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 9,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 02:09:26 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 02:09:26 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 1.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 20.438265085220337,
      "overhead_ratio_vs_F0": 0.9794985921299051,
      "random_veto_matched_gap": 0,
      "rows": 15,
      "safety_type": "F0_no_safety_control",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 1.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.05,
      "finite_step_scale_mean_median": 0.01171875,
      "finite_step_skip_count_median": 152.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 21.061222553253174,
      "overhead_ratio_vs_F0": 1.0093536683974216,
      "random_veto_matched_gap": 5,
      "rows": 15,
      "safety_type": "F2_finite_step_trust_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.9956442707001877,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 105.56189465522766,
      "overhead_ratio_vs_F0": 5.059026623161452,
      "random_veto_matched_gap": 0,
      "rows": 15,
      "safety_type": "F6_random_veto_matched_control",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 1.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.8,
      "finite_step_scale_mean_median": 0.8,
      "finite_step_skip_count_median": 32.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 20.669357299804688,
      "overhead_ratio_vs_F0": 0.9905736270163661,
      "random_veto_matched_gap": 0,
      "rows": 15,
      "safety_type": "F7_cadence_finite_step_trust",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 1.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 1,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 20.660175323486328,
      "overhead_ratio_vs_F0": 0.992313557094027,
      "random_veto_matched_gap": 0,
      "rows": 15,
      "safety_type": "F0_no_safety_control",
      "scheme": "E7_FunctionalGram_DegreeEdgebankSNR_s0p25"
    },
    {
      "C2_coverage_retention_vs_F0": 19.676455746524304,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.0375,
      "finite_step_scale_mean_median": 0.008203125,
      "finite_step_skip_count_median": 154.0,
      "group_idx": 1,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 21.127284049987793,
      "overhead_ratio_vs_F0": 1.014748909877186,
      "random_veto_matched_gap": 5,
      "rows": 15,
      "safety_type": "F2_finite_step_trust_all",
      "scheme": "E7_FunctionalGram_DegreeEdgebankSNR_s0p25"
    },
    {
      "C2_coverage_retention_vs_F0": -3.1916242771404897,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 1,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 105.69323015213013,
      "overhead_ratio_vs_F0": 5.076473143661574,
      "random_veto_matched_gap": 0,
      "rows": 15,
      "safety_type": "F6_random_veto_matched_control",
      "scheme": "E7_FunctionalGram_DegreeEdgebankSNR_s0p25"
    },
    {
      "C2_coverage_retention_vs_F0": 3.802741688023416,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.8,
      "finite_step_scale_mean_median": 0.8,
      "finite_step_skip_count_median": 32.0,
      "group_idx": 1,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 21.01823401451111,
      "overhead_ratio_vs_F0": 1.0095112085067584,
      "random_veto_matched_gap": 0,
      "rows": 15,
      "safety_type": "F7_cadence_finite_step_trust",
      "scheme": "E7_FunctionalGram_DegreeEdgebankSNR_s0p25"
    },
    {
      "C2_coverage_retention_vs_F0": 1.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 2,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 20.372085571289062,
      "overhead_ratio_vs_F0": 0.9892147374504205,
      "random_veto_matched_gap": 0,
      "rows": 15,
      "safety_type": "F0_no_safety_control",
      "scheme": "E7_FunctionalGram_DegreeEdgebankSNR_s0p5"
    },
    {
      "C2_coverage_retention_vs_F0": 0.7996650863854837,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.05,
      "finite_step_scale_mean_median": 0.01015625,
      "finite_step_skip_count_median": 152.0,
      "group_idx": 2,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 21.278844356536865,
      "overhead_ratio_vs_F0": 1.0332445521957456,
      "random_veto_matched_gap": 5,
      "rows": 15,
      "safety_type": "F2_finite_step_trust_all",
      "scheme": "E7_FunctionalGram_DegreeEdgebankSNR_s0p5"
    },
    {
      "C2_coverage_retention_vs_F0": 0.21284677275451297,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 2,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 105.73421263694763,
      "overhead_ratio_vs_F0": 5.13417445784697,
      "random_veto_matched_gap": 0,
      "rows": 15,
      "safety_type": "F6_random_veto_matched_control",
      "scheme": "E7_FunctionalGram_DegreeEdgebankSNR_s0p5"
    },
    {
      "C2_coverage_retention_vs_F0": 0.21772384270026743,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.8,
      "finite_step_scale_mean_median": 0.8,
      "finite_step_skip_count_median": 32.0,
      "group_idx": 2,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 20.549071311950684,
      "overhead_ratio_vs_F0": 0.9978086981604519,
      "random_veto_matched_gap": 0,
      "rows": 15,
      "safety_type": "F7_cadence_finite_step_trust",
      "scheme": "E7_FunctionalGram_DegreeEdgebankSNR_s0p5"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 180,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 180,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 02:25:19 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 02:25:19 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 1.0,
      "F5_no_debt_count": 1,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 1,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.06875,
      "finite_step_scale_mean_median": 0.016796875,
      "finite_step_skip_count_median": 149.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 21.437013149261475,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 1,
      "rows": 15,
      "safety_type": "F3_finite_step_trust_tail_only",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 1.0,
      "F5_no_debt_count": 3,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 3,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.04375,
      "finite_step_scale_mean_median": 0.005859375,
      "finite_step_skip_count_median": 153.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 21.13849925994873,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 3,
      "rows": 15,
      "safety_type": "F4_finite_step_trust_tail_margin",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 1.0,
      "F5_no_debt_count": 2,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 2,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.075,
      "finite_step_scale_mean_median": 0.02578125,
      "finite_step_skip_count_median": 148.0,
      "group_idx": 1,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 21.41947650909424,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 2,
      "rows": 15,
      "safety_type": "F3_finite_step_trust_tail_only",
      "scheme": "E7_FunctionalGram_DegreeEdgebankSNR_s0p5"
    },
    {
      "C2_coverage_retention_vs_F0": 1.0,
      "F5_no_debt_count": 2,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 2,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.06875,
      "finite_step_scale_mean_median": 0.019921875,
      "finite_step_skip_count_median": 149.0,
      "group_idx": 1,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 21.132699966430664,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 2,
      "rows": 15,
      "safety_type": "F4_finite_step_trust_tail_margin",
      "scheme": "E7_FunctionalGram_DegreeEdgebankSNR_s0p5"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 60,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 60,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 02:29:06 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 02:29:06 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 2.1408650875091553,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 1,
      "safety_type": "F8_predictive_scaled_cadence_scale0125",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 0.81211256980896,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 1,
      "safety_type": "F8_predictive_scaled_cadence_scale025",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 0.8697309494018555,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 1,
      "safety_type": "F8_predictive_scaled_cadence_scale05",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 3,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 3,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 02:30:53 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 02:30:53 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.8,
      "finite_step_scale_mean_median": 0.1,
      "finite_step_skip_count_median": 4.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 2.5900371074676514,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 1,
      "safety_type": "F8_predictive_scaled_cadence_scale0125",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.8,
      "finite_step_scale_mean_median": 0.2,
      "finite_step_skip_count_median": 4.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 1.3260655403137207,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 1,
      "safety_type": "F8_predictive_scaled_cadence_scale025",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.8,
      "finite_step_scale_mean_median": 0.4,
      "finite_step_skip_count_median": 4.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 1.263922929763794,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 1,
      "safety_type": "F8_predictive_scaled_cadence_scale05",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 3,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 3,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 02:31:39 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 02:31:39 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.5,
      "finite_step_scale_mean_median": 0.0625,
      "finite_step_skip_count_median": 20.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 5.660463094711304,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 1,
      "safety_type": "F8_predictive_scaled_cadence_scale0125",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.5,
      "finite_step_scale_mean_median": 0.125,
      "finite_step_skip_count_median": 20.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 4.2524073123931885,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 1,
      "safety_type": "F8_predictive_scaled_cadence_scale025",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 2,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 2,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 02:33:12 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 02:33:12 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.5,
      "finite_step_scale_mean_median": 0.0625,
      "finite_step_skip_count_median": 20.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 25.26580238342285,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 1,
      "safety_type": "F9_predictive_scaled_cadence_veto_scale0125",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.5,
      "finite_step_scale_mean_median": 0.125,
      "finite_step_skip_count_median": 20.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 24.00546932220459,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 1,
      "safety_type": "F9_predictive_scaled_cadence_veto_scale025",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 2,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 2,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 02:34:16 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 02:34:16 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 1,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 1,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.1625,
      "finite_step_scale_mean_median": 0.022674590349197388,
      "finite_step_skip_count_median": 67.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 30.203073978424072,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 1,
      "rows": 1,
      "safety_type": "F2_finite_step_trust_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 1,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 1,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 02:41:20 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 02:41:20 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.4125,
      "finite_step_scale_mean_median": 0.0036316803842782975,
      "finite_step_skip_count_median": 47.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 3.506927251815796,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 1,
      "safety_type": "F10_predictive_slack_scale005",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.075,
      "finite_step_scale_mean_median": 0.003347394336014986,
      "finite_step_skip_count_median": 74.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 1.2316715717315674,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 1,
      "safety_type": "F10_predictive_slack_scale00625",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.075,
      "finite_step_scale_mean_median": 0.003964299112558365,
      "finite_step_skip_count_median": 74.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 1.198685884475708,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 1,
      "safety_type": "F10_predictive_slack_scale0075",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.075,
      "finite_step_scale_mean_median": 0.0047286969423294065,
      "finite_step_skip_count_median": 74.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 1.2059738636016846,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 1,
      "safety_type": "F10_predictive_slack_scale01",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.075,
      "finite_step_scale_mean_median": 0.003964299112558365,
      "finite_step_skip_count_median": 74.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 5.778787136077881,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 1,
      "safety_type": "F11_predictive_slack_veto_scale0075",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.075,
      "finite_step_scale_mean_median": 0.0047286969423294065,
      "finite_step_skip_count_median": 74.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 5.744604587554932,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 1,
      "safety_type": "F11_predictive_slack_veto_scale01",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 6,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 6,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 02:44:27 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 02:44:27 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.0625,
      "finite_step_scale_mean_median": 0.04140486717224121,
      "finite_step_skip_count_median": 75.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 6.841021299362183,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F2_finite_step_trust_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 02:50:50 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 02:50:50 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 1,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 1,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.175,
      "finite_step_scale_mean_median": 0.06609113812446595,
      "finite_step_skip_count_median": 66.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 91.68823289871216,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 1,
      "rows": 1,
      "safety_type": "F12_proactive_corrected_veto_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 1,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 1,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.175,
      "finite_step_scale_mean_median": 0.06609113812446595,
      "finite_step_skip_count_median": 66.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 16.422637224197388,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 1,
      "rows": 1,
      "safety_type": "F2_finite_step_trust_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 2,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 2,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 02:55:15 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 02:55:15 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 1,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 1,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.1625,
      "finite_step_scale_mean_median": 0.06464837789535523,
      "finite_step_skip_count_median": 67.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 18.23909831047058,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 1,
      "rows": 1,
      "safety_type": "F13_debtreg0005_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 1,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 1,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.125,
      "finite_step_scale_mean_median": 0.04671963155269623,
      "finite_step_skip_count_median": 70.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 34.81179928779602,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 1,
      "rows": 1,
      "safety_type": "F13_debtreg001_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 1,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 1,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.2125,
      "finite_step_scale_mean_median": 0.10315083861351013,
      "finite_step_skip_count_median": 63.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 32.45853567123413,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 1,
      "rows": 1,
      "safety_type": "F13_debtreg001_margin_only_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 1,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 1,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.25,
      "finite_step_scale_mean_median": 0.10162189304828644,
      "finite_step_skip_count_median": 60.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 28.97312879562378,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 1,
      "rows": 1,
      "safety_type": "F13_debtreg002_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 1,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 1,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.1375,
      "finite_step_scale_mean_median": 0.06365490853786468,
      "finite_step_skip_count_median": 69.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 15.63518762588501,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 1,
      "rows": 1,
      "safety_type": "F13_debtreg005_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 1,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 1,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.0625,
      "finite_step_scale_mean_median": 0.032042074203491214,
      "finite_step_skip_count_median": 75.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 4.187385320663452,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 1,
      "rows": 1,
      "safety_type": "F13_debtreg005_margin_only_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 1,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 1,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.1,
      "finite_step_scale_mean_median": 0.0386018842458725,
      "finite_step_skip_count_median": 72.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 4.2113494873046875,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 1,
      "rows": 1,
      "safety_type": "F13_debtreg01_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 1,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 1,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.175,
      "finite_step_scale_mean_median": 0.06609113812446595,
      "finite_step_skip_count_median": 66.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 16.78425121307373,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 1,
      "rows": 1,
      "safety_type": "F2_finite_step_trust_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 8,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 8,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 02:56:56 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 02:56:56 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.05,
      "finite_step_scale_mean_median": 0.026092720031738282,
      "finite_step_skip_count_median": 76.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 4.778624534606934,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F13_debtreg001_margin_only_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.05,
      "finite_step_scale_mean_median": 0.03059633672237396,
      "finite_step_skip_count_median": 76.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 6.683332204818726,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F13_debtreg002_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.0625,
      "finite_step_scale_mean_median": 0.04140486717224121,
      "finite_step_skip_count_median": 75.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 4.906471490859985,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F2_finite_step_trust_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 15,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 15,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 02:58:11 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 02:58:11 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.0125,
      "finite_step_scale_mean_median": 0.0005279392004013062,
      "finite_step_skip_count_median": 79.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 4.712890148162842,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F14_cli_marginreg_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 02:58:11 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 02:58:11 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.0125,
      "finite_step_scale_mean_median": 0.0005279392004013062,
      "finite_step_skip_count_median": 79.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 4.7982048988342285,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F14_cli_marginreg_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 02:58:11 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 02:58:11 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.0125,
      "finite_step_scale_mean_median": 0.0005279392004013062,
      "finite_step_skip_count_median": 79.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 4.768645763397217,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F14_cli_marginreg_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 02:58:12 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 02:58:12 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.0125,
      "finite_step_scale_mean_median": 0.0007039189338684082,
      "finite_step_skip_count_median": 79.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 4.976008653640747,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F14_cli_marginreg_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:01:32 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:01:32 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 1,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 1,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.1,
      "finite_step_scale_mean_median": 0.02028627097606659,
      "finite_step_skip_count_median": 72.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 5.462536811828613,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 1,
      "rows": 1,
      "safety_type": "F2_finite_step_trust_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.8,
      "finite_step_scale_mean_median": 0.8,
      "finite_step_skip_count_median": 16.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 10.137159585952759,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 1,
      "safety_type": "F7_cadence_finite_step_trust",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 0,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 0,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.8,
      "finite_step_scale_mean_median": 0.2,
      "finite_step_skip_count_median": 16.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 10.081978797912598,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 1,
      "safety_type": "F8_predictive_scaled_cadence_scale025",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 3,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 3,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:04:11 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:04:11 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.1125,
      "finite_step_scale_mean_median": 0.09307479858398438,
      "finite_step_skip_count_median": 71.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 6.7118096351623535,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F14_cli_tailreg_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.1125,
      "finite_step_scale_mean_median": 0.09307479858398438,
      "finite_step_skip_count_median": 71.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 4.765873670578003,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F2_finite_step_trust_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 10,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 10,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:06:14 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:06:14 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.1125,
      "finite_step_scale_mean_median": 0.07533158957958222,
      "finite_step_skip_count_median": 71.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 5.944394826889038,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F15_cli_debtreg_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:06:15 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:06:15 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.1125,
      "finite_step_scale_mean_median": 0.07533158957958222,
      "finite_step_skip_count_median": 71.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 6.254960298538208,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F15_cli_debtreg_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:06:15 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:06:15 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.1125,
      "finite_step_scale_mean_median": 0.07533158957958222,
      "finite_step_skip_count_median": 71.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 6.486160039901733,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F15_cli_debtreg_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:06:16 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:06:16 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.1125,
      "finite_step_scale_mean_median": 0.07533158957958222,
      "finite_step_skip_count_median": 71.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 6.242003917694092,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F15_cli_debtreg_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:08:41 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:08:41 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.125,
      "finite_step_scale_mean_median": 0.09125898778438568,
      "finite_step_skip_count_median": 70.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 24.71307682991028,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F15_cli_reg_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:08:42 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:08:42 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.125,
      "finite_step_scale_mean_median": 0.08405094146728516,
      "finite_step_skip_count_median": 70.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 24.51910638809204,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F15_cli_reg_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:08:45 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:08:45 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.1375,
      "finite_step_scale_mean_median": 0.08555220067501068,
      "finite_step_skip_count_median": 69.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 25.2571222782135,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F15_cli_reg_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:08:56 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:08:56 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.15,
      "finite_step_scale_mean_median": 0.08789283037185669,
      "finite_step_skip_count_median": 68.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 25.546772718429565,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F15_cli_reg_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:11:02 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:11:02 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.5375,
      "finite_step_scale_mean_median": 0.5002368450164795,
      "finite_step_skip_count_median": 37.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 16.115572452545166,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F15_cli_reg_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:11:03 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:11:03 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.55,
      "finite_step_scale_mean_median": 0.5126288056373596,
      "finite_step_skip_count_median": 36.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 16.64361834526062,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F15_cli_reg_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:11:05 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:11:05 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.275,
      "finite_step_scale_mean_median": 0.21529955863952638,
      "finite_step_skip_count_median": 58.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 14.202312469482422,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F15_cli_reg_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:11:33 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:11:33 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.2625,
      "finite_step_scale_mean_median": 0.20595521628856658,
      "finite_step_skip_count_median": 59.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 28.055259466171265,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F15_cli_reg_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:18:53 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:18:53 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 1.0,
      "F5_no_debt_count": 1,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 1,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 9.632885932922363,
      "overhead_ratio_vs_F0": 0.9905512328189203,
      "random_veto_matched_gap": 0,
      "rows": 15,
      "safety_type": "F0_no_safety_control",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 1.7100568125961697,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.5375,
      "finite_step_scale_mean_median": 0.5002368450164795,
      "finite_step_skip_count_median": 37.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 9.935895204544067,
      "overhead_ratio_vs_F0": 1.0217097256787424,
      "random_veto_matched_gap": 4,
      "rows": 15,
      "safety_type": "F15_etailreg0p3_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 1.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.4875,
      "finite_step_scale_mean_median": 0.46944198608398435,
      "finite_step_skip_count_median": 41.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 9.75661325454712,
      "overhead_ratio_vs_F0": 1.0032741334970978,
      "random_veto_matched_gap": 4,
      "rows": 15,
      "safety_type": "F2_finite_step_trust_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 1.00137289533035,
      "F5_no_debt_count": 1,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 1,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 53.35070252418518,
      "overhead_ratio_vs_F0": 5.486061448778606,
      "random_veto_matched_gap": 0,
      "rows": 15,
      "safety_type": "F6_random_veto_matched_control",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 60,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 60,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:35:50 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [
    {
      "C2_coverage_retention_vs_F0": 2.117805482167732,
      "F5_no_debt_count": 15,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 15,
      "depth": "depth3",
      "diagnostic_pass": 1,
      "finite_step_accept_rate_median": 0.225,
      "finite_step_scale_mean_median": 0.1533581554889679,
      "finite_step_skip_count_median": 62.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 9.961692333221436,
      "overhead_ratio_vs_F0": 1.034682856398498,
      "random_veto_matched_gap": 13,
      "rows": 45,
      "safety_type": "F15_etailreg0p3_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:35:50 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": -1.0,
      "F5_no_debt_count": 2,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 2,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 9.573370933532715,
      "overhead_ratio_vs_F0": 0.9943493988301837,
      "random_veto_matched_gap": 0,
      "rows": 45,
      "safety_type": "F0_no_safety_control",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 2.117805482167732,
      "F5_no_debt_count": 15,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 15,
      "depth": "depth3",
      "diagnostic_pass": 1,
      "finite_step_accept_rate_median": 0.225,
      "finite_step_scale_mean_median": 0.1533581554889679,
      "finite_step_skip_count_median": 62.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 9.961692333221436,
      "overhead_ratio_vs_F0": 1.034682856398498,
      "random_veto_matched_gap": 13,
      "rows": 45,
      "safety_type": "F15_etailreg0p3_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": -1.0,
      "F5_no_debt_count": 15,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 15,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.15,
      "finite_step_scale_mean_median": 0.11692121922969818,
      "finite_step_skip_count_median": 68.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 9.717191219329834,
      "overhead_ratio_vs_F0": 1.0092874614744543,
      "random_veto_matched_gap": 13,
      "rows": 45,
      "safety_type": "F2_finite_step_trust_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": -0.9765493785199876,
      "F5_no_debt_count": 2,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 2,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 53.042489767074585,
      "overhead_ratio_vs_F0": 5.509320403287029,
      "random_veto_matched_gap": 0,
      "rows": 45,
      "safety_type": "F6_random_veto_matched_control",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 180,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectDiagnosticOnly",
  "row_count": 180,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:36:52 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:36:52 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 4.4530720710754395,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F15_etailreg0p3_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:37:06 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:37:06 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.9875,
      "finite_step_scale_mean_median": 0.9586737275123596,
      "finite_step_skip_count_median": 1.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 5.3817524909973145,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F15_etailreg0p3_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:37:18 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:37:18 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.9875,
      "finite_step_scale_mean_median": 0.9667177200317383,
      "finite_step_skip_count_median": 1.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 4.4617509841918945,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F15_etailreg0p3_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:37:31 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:37:31 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.8625,
      "finite_step_scale_mean_median": 0.7908974289894104,
      "finite_step_skip_count_median": 11.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 11.038059711456299,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 5,
      "rows": 5,
      "safety_type": "F15_etailreg0p3_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 5,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 5,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:50:59 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [
    {
      "C2_coverage_retention_vs_F0": 1.9296350710173653,
      "F5_no_debt_count": 15,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 15,
      "depth": "depth3",
      "diagnostic_pass": 1,
      "finite_step_accept_rate_median": 0.2125,
      "finite_step_scale_mean_median": 0.17161787450313568,
      "finite_step_skip_count_median": 63.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 9.938280820846558,
      "overhead_ratio_vs_F0": 1.0129393455117706,
      "random_veto_matched_gap": 13,
      "rows": 45,
      "safety_type": "F15_etailreg0p3_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:50:59 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 1.0,
      "F5_no_debt_count": 2,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 2,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 9.648782968521118,
      "overhead_ratio_vs_F0": 0.9834328573829098,
      "random_veto_matched_gap": 0,
      "rows": 45,
      "safety_type": "F0_no_safety_control",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 1.9296350710173653,
      "F5_no_debt_count": 15,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 15,
      "depth": "depth3",
      "diagnostic_pass": 1,
      "finite_step_accept_rate_median": 0.2125,
      "finite_step_scale_mean_median": 0.17161787450313568,
      "finite_step_skip_count_median": 63.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 9.938280820846558,
      "overhead_ratio_vs_F0": 1.0129393455117706,
      "random_veto_matched_gap": 13,
      "rows": 45,
      "safety_type": "F15_etailreg0p3_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 1.0017192822398517,
      "F5_no_debt_count": 2,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 2,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 53.273351192474365,
      "overhead_ratio_vs_F0": 5.429779502399597,
      "random_veto_matched_gap": 0,
      "rows": 45,
      "safety_type": "F6_random_veto_matched_control",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 135,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectDiagnosticOnly",
  "row_count": 135,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:57:20 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:57:20 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 4.281567811965942,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 15,
      "safety_type": "F0_no_safety_control",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 15,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 15,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.725,
      "finite_step_scale_mean_median": 0.7134385585784913,
      "finite_step_skip_count_median": 22.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 4.867719888687134,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 10,
      "rows": 15,
      "safety_type": "F15_etailreg0p3_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 24.13340163230896,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 15,
      "safety_type": "F6_random_veto_matched_control",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 45,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 45,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 03:57:21 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 03:57:21 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 9,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 9,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 4.281543970108032,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 15,
      "safety_type": "F0_no_safety_control",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 15,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 15,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 4.495959758758545,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 6,
      "rows": 15,
      "safety_type": "F15_etailreg0p3_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 9,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 9,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 23.911327838897705,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 0,
      "rows": 15,
      "safety_type": "F6_random_veto_matched_control",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 45,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 45,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 04:01:41 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "f5_efficiency_or_c2_retention_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 04:01:41 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 0.0,
      "F5_no_debt_count": 15,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 15,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.375,
      "finite_step_scale_mean_median": 0.2553262910323156,
      "finite_step_skip_count_median": 50.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 34.7570116519928,
      "overhead_ratio_vs_F0": 0.0,
      "random_veto_matched_gap": 15,
      "rows": 15,
      "safety_type": "F15_etailreg0p3_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 15,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [],
  "route": "F_DirectC2F5PositiveControlFailed",
  "row_count": 15,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 04:11:05 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "none",
  "error_rows": 0,
  "gate_pass": 1,
  "generated_at": "2026-07-03 04:11:05 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": 1.0,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 9.627933740615845,
      "overhead_ratio_vs_F0": 0.9880332014544189,
      "random_veto_matched_gap": 0,
      "rows": 45,
      "safety_type": "F0_no_safety_control",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 1.1504751932933055,
      "F5_no_debt_count": 15,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 15,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.725,
      "finite_step_scale_mean_median": 0.7134385585784913,
      "finite_step_skip_count_median": 22.0,
      "group_idx": 0,
      "official_pass": 1,
      "overhead_proxy_wall_time_median": 10.792680740356445,
      "overhead_ratio_vs_F0": 1.1075613097735901,
      "random_veto_matched_gap": 10,
      "rows": 45,
      "safety_type": "F15_etailreg0p3_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 1.0000481951980165,
      "F5_no_debt_count": 5,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 5,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 53.85291838645935,
      "overhead_ratio_vs_F0": 5.526468377796867,
      "random_veto_matched_gap": 0,
      "rows": 45,
      "safety_type": "F6_random_veto_matched_control",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 135,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [
    {
      "C2_coverage_retention_vs_F0": 1.1504751932933055,
      "F5_no_debt_count": 15,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 15,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.725,
      "finite_step_scale_mean_median": 0.7134385585784913,
      "finite_step_skip_count_median": 22.0,
      "group_idx": 0,
      "official_pass": 1,
      "overhead_proxy_wall_time_median": 10.792680740356445,
      "overhead_ratio_vs_F0": 1.1075613097735901,
      "random_veto_matched_gap": 10,
      "rows": 45,
      "safety_type": "F15_etailreg0p3_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "route": "F_DirectC2F5PositiveControlPass",
  "row_count": 135,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 04:11:37 +0800 Part G direct C2/F5 route

```json
{
  "candidate_update_selection_used": 0,
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "G_DirectFPassButPartCSanityFailed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 04:11:37 +0800",
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 0,
  "part": "G",
  "part_c_route": "C_C2FormationTaskwiseFailed",
  "part_f_route": "F_DirectC2F5PositiveControlPass",
  "passing_part_f_groups": [
    {
      "C2_coverage_retention_vs_F0": 1.1504751932933055,
      "F5_no_debt_count": 15,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 15,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.725,
      "finite_step_scale_mean_median": 0.7134385585784913,
      "finite_step_skip_count_median": 22.0,
      "group_idx": 0,
      "official_pass": 1,
      "overhead_proxy_wall_time_median": 10.792680740356445,
      "overhead_ratio_vs_F0": 1.1075613097735901,
      "random_veto_matched_gap": 10,
      "rows": 45,
      "safety_type": "F15_etailreg0p3_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "promotion_allowed": 0,
  "route": "G_DirectFPassButPartCSanityFailed",
  "row_count": 0,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 04:11:38 +0800 Final route

```json
{
  "candidate_update_selection_used": 0,
  "dominant_blocker": "taskwise_c2_or_random_gap_failed",
  "external_product_feature_used": 0,
  "generated_at": "2026-07-03 04:11:38 +0800",
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "mlp_readout_used": 0,
  "mlp_stem_used": 0,
  "new_edge_function_added": 0,
  "promotion_allowed": 0,
  "route": "DirectFPassButPartCSanityFailed",
  "runtime_selector_used": 0,
  "stale_artifact_count": 0,
  "stale_artifacts": [],
  "used_fake_data_rows": 0
}
```
## 2026-07-03 04:12:00 +0800 Final Direct-F Sign-Fix Recap

### Final status

- Selected root: `results/v23_02_direct_f_signfix_etailreg_lr00025_seed15`
- Final machine route: `DirectFPassButPartCSanityFailed`
- `promotion_allowed=0`
- Reason promotion remains blocked: Part C taskwise C2 official audit still has route `C_C2FormationTaskwiseFailed`.
- Important positive result: Direct Part F C2/F5 continuation pass was achieved after fixing the `margin10` sign bug and adding safety-specific ECE/tail regularized exact trust.

### Critical bug fix

`margin10` was audited against the historical metric definition and found to be a margin where larger is better. Pre-fix v23.00/v23.02 code treated all debt components as safe when `delta <= budget`, so positive `margin10_delta` was incorrectly rejected as debt. This made many pre-fix `debt_margin10` rejections invalid evidence for safety efficiency.

Implemented fix:

- `margin10` guard/final safety now uses `margin10_delta >= -no_debt_budget`.
- Brier/ECE/tail95/tail99 still use `delta <= no_debt_budget`.
- Merge-level component counts were updated to use the same signed rule.
- Result rows now include optimizer/safety audit fields (`edge_lr`, `no_debt_budget`, regularization fields).

### Final passing Direct F group

From `results/v23_02_direct_f_signfix_etailreg_lr00025_seed15/part_f_predictive_trust_positive_control_group_summary.csv`:

```json
{
  "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0",
  "basis_key": "dche_k9",
  "depth": "depth3",
  "safety_type": "F15_etailreg0p3_finite_step_all",
  "rows": 45,
  "F5_no_debt_count": 15,
  "component_non_positive_rows": 15,
  "C2_coverage_retention_vs_F0": 1.1504751932933055,
  "finite_step_accept_rate_median": 0.725,
  "finite_step_scale_mean_median": 0.7134385585784913,
  "finite_step_skip_count_median": 22.0,
  "random_veto_matched_gap": 10,
  "overhead_ratio_vs_F0": 1.1075613097735901,
  "official_pass": 1
}
```

This satisfies the Direct F gate thresholds used by the runner: F5 no-debt >=12/15, component rows >=12/15, accept >=0.20, scale >=0.05, skip <=56 for 80 steps, C2 retention >=0.70, random gap >=8/15, and overhead <=2.5.

### Evidence chain and insights

- Part C remains a real blocker. Earlier taskwise C2 formation did not pass both local_patch_interaction and rotation_sensitive simultaneously, even after floor/step repairs and true Functional Sobolev/Hilbert Gram + Qpop variants.
- Lowering Part C to sanity and directly validating `G_edge^{-1/2} Q_pop G_edge^{-1/2}` plus trust was productive: the final F15 Direct F group passes C2/F5 positive-control despite Part C official failure.
- The main safety-efficiency blocker was not only predictive trust design. A sign error in margin safety caused the exact oracle to reject beneficial margin improvements.
- After sign fix, the active safety blocker moved from margin10 to ECE/tail. A train-only regularizer on `ECE,tail95,tail99` with `edge_lr=0.00025` and exact finite-step guard produced efficient no-debt without using held/test data.
- Very low lr (`0.0001`) made no-safety/random controls too safe and reduced random gap; `0.00025` was the best observed compromise: efficient F15 safety with random gap 10/15.
- Pre-fix results that report heavy `debt_margin10` rejection are useful for debugging history but must not be used as final safety-efficiency conclusions.

### Remaining limitations

- This is not a full promotion because Part C official taskwise C2 formation failed.
- The successful Direct F run uses `train_steps=80`; it is a valid recorded positive-control configuration but should be distinguished from earlier default 160-step diagnostics.
- Real-task preflight remains prohibited by the plan because Part C/G official promotion is not achieved.

## 2026-07-03 04:19:42 +0800 Additional Tries20 Diagnostic Recap

### Result root

- `results/v23_02_etailreg0p3_tries20_seed15`

### What changed relative to the selected final Direct F run

- `finite_step_tries` increased from 12 to 20.
- `edge_lr` increased from 0.00025 to 0.001.
- The same final safety family was used: `F15_etailreg0p3_finite_step_all`.
- The same Part C failure artifact was used only as sanity context, not as an official pass.

### Observed data

From `results/v23_02_etailreg0p3_tries20_seed15/part_f_predictive_trust_positive_control_group_summary.csv`:

```json
{
  "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0",
  "basis_key": "dche_k9",
  "depth": "depth3",
  "safety_type": "F15_etailreg0p3_finite_step_all",
  "rows": 45,
  "F5_no_debt_count": 15,
  "component_non_positive_rows": 15,
  "C2_coverage_retention_vs_F0": 2.117805482167732,
  "finite_step_accept_rate_median": 0.375,
  "finite_step_scale_mean_median": 0.2553262910323156,
  "finite_step_skip_count_median": 50.0,
  "random_veto_matched_gap": 13,
  "overhead_ratio_vs_F0": 0.909128721336865,
  "official_pass": 1
}
```

### Analysis and insight

- This confirms the sign-fixed Direct F path is not a one-off artifact of `finite_step_tries=12`: the F15 group also passes with 20 tries and a higher edge learning rate.
- The higher-try setting improves the no-safety/random contrast (`random_veto_matched_gap=13` versus 10 in the selected main run).
- The cost is a more conservative accepted-update profile: accept rate falls to 0.375, mean accepted scale falls to 0.2553262910323156, and skip count rises to 50.0. It still passes the runner thresholds, but it is closer to the skip boundary than the selected main run.
- The selected main result remains `results/v23_02_direct_f_signfix_etailreg_lr00025_seed15` because it has better update efficiency (`accept_rate_median=0.725`, `scale_mean_median=0.7134385585784913`, `skip_count_median=22.0`) while still satisfying Direct F.
- Part G still fails with route `G_DirectFPassButPartCSanityFailed`, and final route remains `DirectFPassButPartCSanityFailed`; therefore this diagnostic strengthens the Direct F evidence chain but does not remove the Part C blocker.

## 2026-07-03 04:28:44 +0800 Code-Save Audit Recap

During final git-save audit, remaining core-code diffs were found in the edge optimizer and v23 runner support files. These were saved as support code, not as new experimental evidence.

### What was saved

- `dgkan/optim/edge_sobolev_population_flow.py`: added checker/local-patch block construction over first-layer KAN edge functions; added block-granular random-matched permutation for matched random controls; added `gate_input_side` and `gate_patch_size` parameters.
- `experiments/run_v23_00r_curve_geometry_population_flow.py`: passed visual-side and patch-size metadata into the optimizer; parsed checker/local-patch schemes.
- `experiments/run_v23_02_predictive_trust_functional_population_flow.py`: added C16/C17 checker-patch Part C scheme mappings and random-control pairing.

### Verification

- Command: `/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile dgkan/optim/edge_sobolev_population_flow.py experiments/run_v23_00r_curve_geometry_population_flow.py experiments/run_v23_02_predictive_trust_functional_population_flow.py`
- Result: exit code 0.

### Interpretation

- This implements an additional edge-function structured gate family aligned with the plan's repair direction, but no new numeric experiment result is claimed from this code-save step.
- The final recorded evidence remains the sign-fixed Direct F results, especially `results/v23_02_direct_f_signfix_etailreg_lr00025_seed15` and the additional `results/v23_02_etailreg0p3_tries20_seed15` diagnostic.
- The promotion conclusion is unchanged: Part C remains the official blocker and `promotion_allowed=0`.

## 2026-07-03 04:39:40 +0800 Checker-Patch Part C Diagnostic Recap

### Result root

- `results/v23_02_checker_patch_c_local_seed5`

### Scope

- Tested task: `local_patch_interaction`
- Schemes: `C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0`, `C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0`, `C5_FunctionalGram_BlockSNR_edgebank_s0`, `C7_FunctionalGram_RandomMatchedGate_s0`
- Basis/depth: `dche_k9`, `depth3`
- Seeds per group: 5
- Gate floors: 0.3, 0.35, 0.4
- This is a single-task diagnostic, not the full official two-task Part C gate.

### Data

From `results/v23_02_checker_patch_c_local_seed5/part_c_taskwise_c2_group_summary.csv`:

```json
{
  "route": "C_C2FormationTaskwiseFailed",
  "gate_pass": 0,
  "row_count": 60,
  "used_fake_data_rows": 0,
  "best_structured_checker_patch_row": {
    "scheme": "C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0",
    "gate_floor": 0.4,
    "rows": 5,
    "taskwise_all_pass": 1,
    "official_pass": 0,
    "C2_accuracy_improvement_median": 0.53125,
    "C2_coverage_improvement_median": 0.10959683824330568,
    "gate_density_median": 0.654383334683048,
    "random_gap": 0.034741508337901905,
    "random_control_scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0"
  },
  "checker_random_matched_control_rows": [
    {
      "scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "gate_floor": 0.3,
      "taskwise_all_pass": 1,
      "C2_coverage_improvement_median": 0.15899902285309508,
      "random_gap": 0.15758791645930614
    },
    {
      "scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "gate_floor": 0.35,
      "taskwise_all_pass": 1,
      "C2_coverage_improvement_median": 0.09787708072690293,
      "random_gap": 0.12942709519666096
    },
    {
      "scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "gate_floor": 0.4,
      "taskwise_all_pass": 1,
      "C2_coverage_improvement_median": 0.07485532990540378,
      "random_gap": 0.0814660030464438
    }
  ]
}
```

### Analysis

- There is real local-task signal in the new edge-function checker/local-patch block family: `C16` at `gate_floor=0.4` passes the local taskwise row with positive accuracy and coverage improvements.
- However, the random-control separation is not strong enough. The structured `C16` row has `random_gap=0.034741508337901905`, below the official threshold, and the checker random-matched control `C17` also passes local-task rows.
- This suggests the block family is capturing useful local-patch geometry, but the current random-matched baseline is too permissive or too structurally similar for an official C2 claim.
- Because this diagnostic only tested `local_patch_interaction`, it cannot repair the original full Part C failure across both `local_patch_interaction` and `rotation_sensitive`.

### Conclusion

- Official Part C remains failed.
- This is not a promotion result.
- Useful next repair direction: keep the edge-function checker/local-patch Hilbert/Sobolev block idea, but redesign the matched random control or add the missing rotation-sensitive structured block before attempting a full Part C rerun.

## 2026-07-03 04:44:08 +0800 Checker-Patch Full Two-Task Part C Recap

### Result root

- `results/v23_02_checker_patch_c_both_seed5_floor04`

### Scope

- Tasks: `local_patch_interaction`, `rotation_sensitive`
- Schemes: `C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0`, `C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0`
- Basis/depth: `dche_k9`, `depth3`
- Seeds per group: 5
- Gate floor: 0.4

### Data

From `results/v23_02_checker_patch_c_both_seed5_floor04/part_c_taskwise_c2_summary.json`:

```json
{
  "route": "C_C2FormationTaskwiseFailed",
  "gate_pass": 0,
  "row_count": 20,
  "used_fake_data_rows": 0,
  "C16_group": {
    "taskwise_all_pass": 0,
    "official_pass": 0,
    "C2_accuracy_improvement_median": 0.546875,
    "C2_coverage_improvement_median": 0.22123066616404685,
    "gate_density_median": 0.6496231400718293,
    "random_gap": 0.014359392538608517
  },
  "C16_tasks": {
    "local_patch_interaction": {
      "taskwise_pass": 1,
      "C2_coverage_improvement_median": 0.10959683824330568,
      "random_gap": 0.05742679873583256
    },
    "rotation_sensitive": {
      "taskwise_pass": 0,
      "C2_coverage_improvement_median": 0.32753474580385955,
      "random_gap": -0.1696422048189561
    }
  },
  "C17_random_matched_control": {
    "taskwise_all_pass": 1,
    "official_pass": 0,
    "C2_coverage_improvement_median": 0.20687127362543833,
    "random_gap": 0.20687127362543833
  }
}
```

### Analysis

- C16 has a positive aggregate C2 coverage improvement, and it passes `local_patch_interaction`, so the checker/local-patch edge-function block is not inert.
- The failure is concentrated in the official random-control separation, especially on `rotation_sensitive`: C16 has positive raw coverage on rotation but loses to the checker random-matched control (`random_gap=-0.1696422048189561`).
- C17, the checker random-matched control, passes both tasks against the older C7 random baseline. This means the new random-matched control retains strong useful structure and cannot be treated as a neutral baseline.
- Therefore the next Part C repair should not simply reuse C16 at more seeds/floors. The evidence points to either a stricter randomization that preserves compute but destroys checker-patch locality more cleanly, or a separate rotation-sensitive edge-function block that can beat the current C17 control.

### Conclusion

- Part C remains officially failed.
- This full two-task diagnostic strengthens the mechanistic insight but does not change final route or promotion status.
- Final promotion remains blocked: `promotion_allowed=0`.

## 2026-07-03 04:51:55 +0800 Visual-Pattern Part C Diagnostic Recap

### Result root

- `results/v23_02_visual_pattern_c_both_seed5_floor04`

### Scope

- Tasks: `local_patch_interaction`, `rotation_sensitive`
- Schemes: `C18_FunctionalGram_VisualPatternDegreeEdgebankSNR_s0`, `C19_FunctionalGram_VisualPatternDegreeRandomMatched_s0`
- Basis/depth: `dche_k9`, `depth3`
- Seeds per group: 5
- Gate floor: 0.4
- Sharding: 4 shards with explicit `CUDA_VISIBLE_DEVICES=0,1,2,3`

### Execution audit note

A conflicting automatically-started 2-shard C18/C19 run targeted the same result root while the intended 4-shard run was active. It was stopped before producing shard CSVs, to avoid artifact overwrite. The retained evidence is the 4-shard output files:

- `part_c_taskwise_c2_matrix_shard0_of_4.csv`
- `part_c_taskwise_c2_matrix_shard1_of_4.csv`
- `part_c_taskwise_c2_matrix_shard2_of_4.csv`
- `part_c_taskwise_c2_matrix_shard3_of_4.csv`

### Data

From `results/v23_02_visual_pattern_c_both_seed5_floor04/part_c_taskwise_c2_summary.json`:

```json
{
  "route": "C_C2FormationTaskwiseFailed",
  "gate_pass": 0,
  "row_count": 20,
  "used_fake_data_rows": 0,
  "C18_group": {
    "taskwise_all_pass": 0,
    "official_pass": 0,
    "C2_accuracy_improvement_median": 0.51953125,
    "C2_coverage_improvement_median": 0.2034493596147513,
    "gate_density_median": 0.6550553835307555,
    "random_gap": -0.0010013098562922096
  },
  "C18_tasks": {
    "local_patch_interaction": {
      "taskwise_pass": 0,
      "C2_coverage_improvement_median": -0.05120259671821259,
      "random_gap": -0.0539369899634039
    },
    "rotation_sensitive": {
      "taskwise_pass": 0,
      "C2_coverage_improvement_median": 0.3508070065290667,
      "random_gap": -0.017270728771109134
    }
  },
  "C19_random_matched_control": {
    "taskwise_all_pass": 0,
    "official_pass": 0,
    "C2_coverage_improvement_median": 0.2044506694710435,
    "random_gap": 0.2044506694710435
  }
}
```

### Analysis

- Adding diagonal/anti-diagonal visual-pattern edge blocks improved raw rotation coverage for C18, but it did not beat its matched random control. The C18 rotation random gap stayed negative.
- C18 hurt local patch coverage at this floor, so the visual-pattern block is not a drop-in replacement for checker/local-patch blocks.
- C19 still shows strong rotation signal as a random-matched control, reinforcing that the current random-control family is too informative for an official C2 claim.
- Compared with C16, C18 shifts signal toward rotation but loses local-patch behavior. The repair problem now looks like a composition/control issue: local checker blocks and rotation visual blocks each capture pieces, but matched random separation remains the bottleneck.

### Conclusion

- Visual-pattern repair did not pass official Part C.
- The final blocker remains Part C taskwise/random-gap separation.
- Direct F positive-control evidence remains valid but cannot be promoted while Part C stays failed.

## 2026-07-03 04:17:29 +0800 Part F direct C2/F5 positive-control

```json
{
  "candidate_update_selection_used": 0,
  "diagnostic_part_f_groups": [],
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "none",
  "error_rows": 0,
  "gate_pass": 1,
  "generated_at": "2026-07-03 04:17:29 +0800",
  "group_summaries": [
    {
      "C2_coverage_retention_vs_F0": -1.0,
      "F5_no_debt_count": 2,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 2,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 10.62653398513794,
      "overhead_ratio_vs_F0": 0.9809815847759358,
      "random_veto_matched_gap": 0,
      "rows": 45,
      "safety_type": "F0_no_safety_control",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": 2.117805482167732,
      "F5_no_debt_count": 15,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 15,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.375,
      "finite_step_scale_mean_median": 0.2553262910323156,
      "finite_step_skip_count_median": 50.0,
      "group_idx": 0,
      "official_pass": 1,
      "overhead_proxy_wall_time_median": 9.84818410873413,
      "overhead_ratio_vs_F0": 0.909128721336865,
      "random_veto_matched_gap": 13,
      "rows": 45,
      "safety_type": "F15_etailreg0p3_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    },
    {
      "C2_coverage_retention_vs_F0": -0.9789636627103427,
      "F5_no_debt_count": 2,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 2,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 1.0,
      "finite_step_scale_mean_median": 1.0,
      "finite_step_skip_count_median": 0.0,
      "group_idx": 0,
      "official_pass": 0,
      "overhead_proxy_wall_time_median": 55.300302505493164,
      "overhead_ratio_vs_F0": 5.105011517988681,
      "random_veto_matched_gap": 0,
      "rows": 45,
      "safety_type": "F6_random_veto_matched_control",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 135,
  "part": "F",
  "part_c_role": "sanity",
  "passing_part_f_groups": [
    {
      "C2_coverage_retention_vs_F0": 2.117805482167732,
      "F5_no_debt_count": 15,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 15,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.375,
      "finite_step_scale_mean_median": 0.2553262910323156,
      "finite_step_skip_count_median": 50.0,
      "group_idx": 0,
      "official_pass": 1,
      "overhead_proxy_wall_time_median": 9.84818410873413,
      "overhead_ratio_vs_F0": 0.909128721336865,
      "random_veto_matched_gap": 13,
      "rows": 45,
      "safety_type": "F15_etailreg0p3_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "route": "F_DirectC2F5PositiveControlPass",
  "row_count": 135,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 04:18:28 +0800 Part G direct C2/F5 route

```json
{
  "candidate_update_selection_used": 0,
  "direct_after_part_c_failure": 1,
  "dominant_blocker": "G_DirectFPassButPartCSanityFailed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 04:18:28 +0800",
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 0,
  "part": "G",
  "part_c_route": "C_C2FormationTaskwiseFailed",
  "part_f_route": "F_DirectC2F5PositiveControlPass",
  "passing_part_f_groups": [
    {
      "C2_coverage_retention_vs_F0": 2.117805482167732,
      "F5_no_debt_count": 15,
      "basis_key": "dche_k9",
      "component_non_positive_rows": 15,
      "depth": "depth3",
      "diagnostic_pass": 0,
      "finite_step_accept_rate_median": 0.375,
      "finite_step_scale_mean_median": 0.2553262910323156,
      "finite_step_skip_count_median": 50.0,
      "group_idx": 0,
      "official_pass": 1,
      "overhead_proxy_wall_time_median": 9.84818410873413,
      "overhead_ratio_vs_F0": 0.909128721336865,
      "random_veto_matched_gap": 13,
      "rows": 45,
      "safety_type": "F15_etailreg0p3_finite_step_all",
      "scheme": "E5_FunctionalGram_DegreeEdgebankSNR_s0"
    }
  ],
  "promotion_allowed": 0,
  "route": "G_DirectFPassButPartCSanityFailed",
  "row_count": 0,
  "runtime_selector_used": 0,
  "used_fake_data_rows": 0
}
```

## 2026-07-03 04:18:30 +0800 Final route

```json
{
  "candidate_update_selection_used": 0,
  "dominant_blocker": "taskwise_c2_or_random_gap_failed",
  "external_product_feature_used": 0,
  "generated_at": "2026-07-03 04:18:30 +0800",
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "mlp_readout_used": 0,
  "mlp_stem_used": 0,
  "new_edge_function_added": 0,
  "promotion_allowed": 0,
  "route": "DirectFPassButPartCSanityFailed",
  "runtime_selector_used": 0,
  "stale_artifact_count": 0,
  "stale_artifacts": [],
  "used_fake_data_rows": 0
}
```

## 2026-07-03 04:37:50 +0800 Part C taskwise C2 formation

```json
{
  "candidate_update_selection_used": 0,
  "dominant_blocker": "taskwise_c2_or_random_gap_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 04:37:50 +0800",
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 60,
  "part": "C",
  "passing_scheme_groups": [],
  "route": "C_C2FormationTaskwiseFailed",
  "row_count": 60,
  "runtime_selector_used": 0,
  "scheme_groups": [
    {
      "C2_accuracy_improvement_median": 0.5234375,
      "C2_coverage_improvement_median": 0.13065108249429613,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6027655059678686,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_control_scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "random_gap": -0.02834794035879895,
      "rows": 5,
      "scheme": "C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5078125,
      "C2_coverage_improvement_median": 0.06757369340630248,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6333055045869614,
      "gate_floor": 0.35,
      "official_pass": 0,
      "random_control_scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "random_gap": -0.03030338732060045,
      "rows": 5,
      "scheme": "C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.53125,
      "C2_coverage_improvement_median": 0.10959683824330568,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.654383334683048,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_control_scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "random_gap": 0.034741508337901905,
      "rows": 5,
      "scheme": "C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0",
      "taskwise_all_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.5078125,
      "C2_coverage_improvement_median": 0.15899902285309508,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.604441136204534,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.15758791645930614,
      "rows": 5,
      "scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "taskwise_all_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.5,
      "C2_coverage_improvement_median": 0.09787708072690293,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6308495502505042,
      "gate_floor": 0.35,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.12942709519666096,
      "rows": 5,
      "scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "taskwise_all_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.5,
      "C2_coverage_improvement_median": 0.07485532990540378,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6576555415987962,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.0814660030464438,
      "rows": 5,
      "scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "taskwise_all_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.4765625,
      "C2_coverage_improvement_median": -0.039555550461955136,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6085627183318137,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.04096665685574408,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.4375,
      "C2_coverage_improvement_median": -0.03992684016338899,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6310253197534222,
      "gate_floor": 0.35,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.008376825693630963,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.011710698643582873,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6607347577396362,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.005100025502542849,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5,
      "C2_coverage_improvement_median": 0.0014111063937889412,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5589441901693739,
      "gate_floor": 0.3,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.453125,
      "C2_coverage_improvement_median": -0.031550014469758025,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5958076555281876,
      "gate_floor": 0.35,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.4765625,
      "C2_coverage_improvement_median": -0.006610673141040024,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.619117463876804,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "taskwise_all_pass": 0
    }
  ],
  "taskwise_groups": [
    {
      "C2_accuracy_improvement_median": 0.5234375,
      "C2_coverage_improvement_median": 0.13065108249429613,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6027655059678686,
      "gate_floor": 0.3,
      "random_control_scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "random_gap": -0.02834794035879895,
      "rows": 5,
      "scheme": "C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5078125,
      "C2_coverage_improvement_median": 0.06757369340630248,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6333055045869614,
      "gate_floor": 0.35,
      "random_control_scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "random_gap": -0.03030338732060045,
      "rows": 5,
      "scheme": "C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.53125,
      "C2_coverage_improvement_median": 0.10959683824330568,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.654383334683048,
      "gate_floor": 0.4,
      "random_control_scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "random_gap": 0.034741508337901905,
      "rows": 5,
      "scheme": "C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.5078125,
      "C2_coverage_improvement_median": 0.15899902285309508,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.604441136204534,
      "gate_floor": 0.3,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.15758791645930614,
      "rows": 5,
      "scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.5,
      "C2_coverage_improvement_median": 0.09787708072690293,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6308495502505042,
      "gate_floor": 0.35,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.12942709519666096,
      "rows": 5,
      "scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.5,
      "C2_coverage_improvement_median": 0.07485532990540378,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6576555415987962,
      "gate_floor": 0.4,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.0814660030464438,
      "rows": 5,
      "scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.4765625,
      "C2_coverage_improvement_median": -0.039555550461955136,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6085627183318137,
      "gate_floor": 0.3,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.04096665685574408,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.4375,
      "C2_coverage_improvement_median": -0.03992684016338899,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6310253197534222,
      "gate_floor": 0.35,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.008376825693630963,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.011710698643582873,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6607347577396362,
      "gate_floor": 0.4,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": -0.005100025502542849,
      "rows": 5,
      "scheme": "C5_FunctionalGram_BlockSNR_edgebank_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5,
      "C2_coverage_improvement_median": 0.0014111063937889412,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5589441901693739,
      "gate_floor": 0.3,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.453125,
      "C2_coverage_improvement_median": -0.031550014469758025,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.5958076555281876,
      "gate_floor": 0.35,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.4765625,
      "C2_coverage_improvement_median": -0.006610673141040024,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.619117463876804,
      "gate_floor": 0.4,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.0,
      "rows": 5,
      "scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    }
  ],
  "used_fake_data_rows": 0
}
```

## 2026-07-03 04:43:10 +0800 Part C taskwise C2 formation

```json
{
  "candidate_update_selection_used": 0,
  "dominant_blocker": "taskwise_c2_or_random_gap_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 04:43:10 +0800",
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 20,
  "part": "C",
  "passing_scheme_groups": [],
  "route": "C_C2FormationTaskwiseFailed",
  "row_count": 20,
  "runtime_selector_used": 0,
  "scheme_groups": [
    {
      "C2_accuracy_improvement_median": 0.546875,
      "C2_coverage_improvement_median": 0.22123066616404685,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6496231400718293,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_control_scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "random_gap": 0.014359392538608517,
      "rows": 10,
      "scheme": "C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.515625,
      "C2_coverage_improvement_median": 0.20687127362543833,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6535577203250592,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.20687127362543833,
      "rows": 10,
      "scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "taskwise_all_pass": 1
    }
  ],
  "taskwise_groups": [
    {
      "C2_accuracy_improvement_median": 0.53125,
      "C2_coverage_improvement_median": 0.10959683824330568,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.654383334683048,
      "gate_floor": 0.4,
      "random_control_scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "random_gap": 0.05742679873583256,
      "rows": 5,
      "scheme": "C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.578125,
      "C2_coverage_improvement_median": 0.32753474580385955,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6494034336672896,
      "gate_floor": 0.4,
      "random_control_scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "random_gap": -0.1696422048189561,
      "rows": 5,
      "scheme": "C16_FunctionalGram_CheckerPatchDegreeEdgebankSNR_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5,
      "C2_coverage_improvement_median": 0.05217003950747312,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6672600912551088,
      "gate_floor": 0.4,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.05217003950747312,
      "rows": 5,
      "scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 1
    },
    {
      "C2_accuracy_improvement_median": 0.6171875,
      "C2_coverage_improvement_median": 0.49717695062281564,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6530371892369458,
      "gate_floor": 0.4,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.49717695062281564,
      "rows": 5,
      "scheme": "C17_FunctionalGram_CheckerPatchDegreeRandomMatched_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    }
  ],
  "used_fake_data_rows": 0
}
```

## 2026-07-03 04:51:12 +0800 Part C taskwise C2 formation

```json
{
  "candidate_update_selection_used": 0,
  "dominant_blocker": "taskwise_c2_or_random_gap_failed",
  "error_rows": 0,
  "gate_pass": 0,
  "generated_at": "2026-07-03 04:51:12 +0800",
  "held_test_usage": 0,
  "metric_winner_selection_used": 0,
  "ok_rows": 20,
  "part": "C",
  "passing_scheme_groups": [],
  "route": "C_C2FormationTaskwiseFailed",
  "row_count": 20,
  "runtime_selector_used": 0,
  "scheme_groups": [
    {
      "C2_accuracy_improvement_median": 0.51953125,
      "C2_coverage_improvement_median": 0.2034493596147513,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6550553835307555,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_control_scheme": "C19_FunctionalGram_VisualPatternDegreeRandomMatched_s0",
      "random_gap": -0.0010013098562922096,
      "rows": 10,
      "scheme": "C18_FunctionalGram_VisualPatternDegreeEdgebankSNR_s0",
      "taskwise_all_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.52734375,
      "C2_coverage_improvement_median": 0.2044506694710435,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6585125369330244,
      "gate_floor": 0.4,
      "official_pass": 0,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.2044506694710435,
      "rows": 10,
      "scheme": "C19_FunctionalGram_VisualPatternDegreeRandomMatched_s0",
      "taskwise_all_pass": 0
    }
  ],
  "taskwise_groups": [
    {
      "C2_accuracy_improvement_median": 0.484375,
      "C2_coverage_improvement_median": -0.05120259671821259,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6586613964703348,
      "gate_floor": 0.4,
      "random_control_scheme": "C19_FunctionalGram_VisualPatternDegreeRandomMatched_s0",
      "random_gap": -0.0539369899634039,
      "rows": 5,
      "scheme": "C18_FunctionalGram_VisualPatternDegreeEdgebankSNR_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.578125,
      "C2_coverage_improvement_median": 0.3508070065290667,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6548974348853028,
      "gate_floor": 0.4,
      "random_control_scheme": "C19_FunctionalGram_VisualPatternDegreeRandomMatched_s0",
      "random_gap": -0.017270728771109134,
      "rows": 5,
      "scheme": "C18_FunctionalGram_VisualPatternDegreeEdgebankSNR_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5078125,
      "C2_coverage_improvement_median": 0.0027343932451913133,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6601212263521216,
      "gate_floor": 0.4,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.0027343932451913133,
      "rows": 5,
      "scheme": "C19_FunctionalGram_VisualPatternDegreeRandomMatched_s0",
      "task": "local_patch_interaction",
      "taskwise_pass": 0
    },
    {
      "C2_accuracy_improvement_median": 0.5625,
      "C2_coverage_improvement_median": 0.36807773530017585,
      "basis_key": "dche_k9",
      "depth": "depth3",
      "gate_density_median": 0.6565408330824642,
      "gate_floor": 0.4,
      "random_control_scheme": "C7_FunctionalGram_RandomMatchedGate_s0",
      "random_gap": 0.36807773530017585,
      "rows": 5,
      "scheme": "C19_FunctionalGram_VisualPatternDegreeRandomMatched_s0",
      "task": "rotation_sensitive",
      "taskwise_pass": 1
    }
  ],
  "used_fake_data_rows": 0
}
```
