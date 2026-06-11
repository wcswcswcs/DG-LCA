# DG-KAN v22.11 ArbitraryLoss ConstructiveFU BasisEfficiency 实验结果复盘

生成时间：2026-06-07 22:08:33 +0800

## Route

- route: `R9-KANArbitraryLossSourceOpened`
- promotion_allowed: `1`
- blocking_metric: ``
- minimum_effective_progress: `A-CodeClosure;B-ArbitraryCotangentRunnerImplemented;C-SourceConstructionProgress;D-VariationalSolveProgress;E-CommitProgress;F-ArbitraryLossFunctionalProgress;G-KANSourceProgress`
- next_codex_action: `implement kernel-native arbitrary-cotangent efficiency path or repair source/horizon blocker according to failure taxonomy`
- S0.18 pass: 1
- arbitrary upstream-cotangent runner exists / CE-targeted trainpath used: 1 / 0
- D-CHE/D-FOU/D-RAT/D-RBF S1 pass: 0 / 1 / 0 / 1
- S2/S3/S4 pass rows: 3 / 4 / 2
- arbitrary-loss horizon C3/C4/source-loss-nonnegative rows: 3 / 1 / 9
- KAN route / pass rows: `S6-KANRetainedSourceOpened` / 2
- results bundle: `results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_results_bundle.zip`
- code review packet: `results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/v22_11_code_review_packet.zip`

## Executive Summary

- v22.11 已实现并执行 arbitrary-loss/upstream-cotangent 接口与 S1 runner；当前 final route=`R9-KANArbitraryLossSourceOpened`，promotion_allowed=1。
- S1 不再走 CE-targeted trainpath，`optimization_loss_agnostic_contract_pass=1`；本轮改为 manual upstream-VJP runner，fallback rows=0，manual_upstream_vjp_rows=120。`official_fused_kernel_complete_rows=0`，因此不声称 fused official kernel 完成。
- best arbitrary-loss horizon row: loss_adapter=`Delta-MSEAdapter`, attempt=`initial_commit_arbitrary_loss_adamw_lr0p1`, source_func_h3200=0.006307865885714858, source_loss_h3200=1.2584553488181882e-07, C3=1, C4=0。
- 所有结论来自本轮落盘 CSV/JSON；未执行或 gate 阻塞的阶段以 blocker 记录，没有写成通过。

## Exploration Process

| stage | question | result | decision |
| --- | --- | --- | --- |
| S0.18 | 代码/loss-interface/packet 是否自洽 | S0.18-CodeLossInterfaceTruthGatePass |  |
| S1 | basis efficiency 是否支持 arbitrary upstream cotangent | S1-ArbitraryCotangentEfficiencyPass |  |
| S2 | real train-stream source atoms 是否跨 cotangent readback | S2-SourceAtomProgress |  |
| S3 | variational solve 是否 retained-source | S3-VariationalSourceSolvePass |  |
| S4 | metric commit 是否可 actuation | S4-MetricDynamicsCommitPass |  |
| S5 | arbitrary-loss horizon 是否 C3/C4 | S5-ArbitraryLossHorizonPass |  |
| S6 | KAN source channel 是否打开 | S6-KANRetainedSourceOpened |  |

## Part A Code / Loss Interface Truth Gate

| check | pass | metric | value | blocker |
| --- | --- | --- | --- | --- |
| required_source_files | 1 | exists | 29/29 |  |
| compileall_repo | 1 | py_compile | 0 |  |
| import_closure_repo | 1 | import | 0 |  |
| loss_interface_tests | 1 | tests | 6/6 |  |
| arbitrary_upstream_cotangent_tests | 1 | tests | 1/1 |  |
| linec_fast_golden | 1 | tests | 9 |  |
| linec_channel_golden | 1 | tests | 8 |  |
| source_chain_tests | 1 | tests | 7/7 |  |
| terminal_retention_tests | 1 | tests | 4/4 |  |
| metric_solver_tests | 1 | tests | 27/27 |  |
| source_atom_generator_tests | 1 | tests | 2/2 |  |
| variational_solver_tests | 1 | tests | 2/2 |  |
| constructive_commit_tests | 1 | tests | 1/1 |  |
| profiler_phase_tests | 1 | tests | 1/1 |  |
| kernel_status_consistency | 1 | official_vs_gradcheck | 1 |  |
| CE_specific_formula_in_core_FU | 1 | semantic | 0 |  |
| LineC_ECE_Brier_AUCtime_used_for_direction | 1 | semantic | 0 |  |
| semantic_alias_undeclared_pairs | 1 | semantic | 0 |  |
| csv_claimed_exists_but_zip_missing_count | 1 | zip_required_compare | 0 |  |
| self_contained_compileall | 1 | clean_unzip_compile | 0 |  |
| self_contained_import_check | 1 | clean_unzip_import | 0 |  |

## Part B Arbitrary-Cotangent Basis Efficiency

| carrier | profile_rows | batch_pass_rows | cotangent_pass_rows | near_E1_rows | fallback_rows | manual_upstream_vjp_rows | official_fused_kernel_complete_rows | best_forward_ratio | best_step_ratio | best_memory_ratio | decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | 24 | 1 | 1 | 0 | 0 | 24 | 0 | 0.8607284052016727 | 0.6815196406224499 | 0.9746170244410292 | ArbitraryCotangentEfficiencyBlocked | ratio_or_fallback_kernel_gate_failed |
| D-FOU | 24 | 4 | 6 | 0 | 0 | 24 | 0 | 0.30947956549393263 | 0.29291213963595564 | 0.9746170244410292 | ArbitraryCotangentEfficiencyPass |  |
| D-RAT | 24 | 1 | 1 | 21 | 0 | 24 | 0 | 0.6101658527546808 | 0.5310789709704629 | 1.0037420335613636 | ArbitraryCotangentEfficiencyBlocked | ratio_or_fallback_kernel_gate_failed |
| D-RBF | 24 | 4 | 6 | 24 | 0 | 24 | 0 | 0.3690263878520741 | 0.3220491810719138 | 1.0 | ArbitraryCotangentEfficiencyPass |  |

Insight: S1 runner 已经从 CE trainpath 改为 arbitrary upstream cotangent manual VJP/update。D-FOU/D-RBF 可以在当前 ratio gate 下 pass；D-RAT 仍只达到 near-E1；D-CHE 低阶 repair 有局部 ratio 改善但未 robust pass。所有 `official_fused_kernel_complete_rows=0`，不写 fused kernel 完成。

## Part C Source Atom Generation

| attempt | fallback_step | norm_scale | split_count | block_role | S2_source_atom_pass_rows | route |
| --- | --- | --- | --- | --- | --- | --- |
| initial_norm_0p08_split4_all | initial | 0.08 | 4 | all | 0 | S2-SourceAtomAttemptBlocked |
| fallback_lower_norm_0p04_split4_all | lower_source_atom_norm | 0.04 | 4 | all | 0 | S2-SourceAtomAttemptBlocked |
| fallback_lower_norm_0p02_split4_all | lower_source_atom_norm | 0.02 | 4 | all | 0 | S2-SourceAtomAttemptBlocked |
| fallback_more_splits_0p02_split6_all | increase_micro_batch_split_count | 0.02 | 6 | all | 0 | S2-SourceAtomAttemptBlocked |
| fallback_block_restricted_0p02_split6_readout | block_restricted_atom | 0.02 | 6 | readout_only | 0 | S2-SourceAtomAttemptBlocked |
| fallback_control_null_residual_0p02_split6_all | control_null_residual_atom | 0.02 | 6 | control_null_only | 0 | S2-SourceAtomAttemptBlocked |
| fallback_arbitrary_cotangent_invariant_0p04_split4_all | arbitrary_cotangent_invariant_atom | 0.04 | 4 | all | 3 | S2-SourceAtomPass |

| attempt | atom_id | atom_family | B2_transfer_gain | random_gap | control_projection_fraction | NDS | loss_adapter_invariance_score | cotangent_same_direction_count | S2_v22_11_source_atom_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initial_norm_0p08_split4_all | A1_split_transfer_atom | split_transfer | -0.05891246348619461 | -0.03293064795434475 | 0.0009692574385553598 | 4.62351655960083 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;loss_agnostic_NDS_gate |
| initial_norm_0p08_split4_all | A2_drift_diffusion_atom | drift_diffusion | -0.058912452310323715 | -0.032930636778473854 | 0.0009692572057247162 | 4.62351655960083 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;loss_agnostic_NDS_gate |
| initial_norm_0p08_split4_all | A3_control_null_residual_atom | control_null_residual | -0.05923762917518616 | -0.033255813643336296 | 0.0007482753717340529 | 4.621641635894775 | 0.6666666666666666 | 4 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;loss_agnostic_NDS_gate |
| initial_norm_0p08_split4_all | A4_low_NDS_atom | low_NDS | -0.11489854007959366 | -0.0889167245477438 | 0.006238869857043028 | 4.413143634796143 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;loss_agnostic_B3_safety_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate |
| initial_norm_0p08_split4_all | A5_row_orthogonal_atom | row_orthogonal | -0.05891246348619461 | -0.03293064795434475 | 0.000969256623648107 | 4.62351655960083 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;loss_agnostic_NDS_gate |
| initial_norm_0p08_split4_all | A6_fast_slow_source_state_atom | fast_slow_source_state | -0.06258795037865639 | -0.036606134846806526 | 0.0010169055312871933 | 4.6379170417785645 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;commutator_norm_ratio_gate;loss_agnostic_NDS_gate |
| initial_norm_0p08_split4_all | A7_train_flow_commutator_atom | train_flow_commutator | -0.09724065661430359 | -0.07125884108245373 | 2.5773257220862433e-05 | 4.759941577911377 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;loss_agnostic_B3_safety_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| initial_norm_0p08_split4_all | A8_low_degree_frequency_atom | low_degree_frequency | -0.07727174460887909 | -0.05128992907702923 | 0.008782981894910336 | 3.4505717754364014 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate |
| initial_norm_0p08_split4_all | A9_control_balanced_drift_atom | control_balanced_drift | -0.13020290434360504 | -0.10422108881175518 | 0.006437549367547035 | 4.337165832519531 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;loss_agnostic_B3_safety_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate |
| initial_norm_0p08_split4_all | A10_split_consensus_atom | split_consensus | 0.3370145112276077 | 0.3629963267594576 | 0.6723052859306335 | 5.17899751663208 | 0.6666666666666666 | 4 | 0 | loss_agnostic_NDS_gate |
| initial_norm_0p08_split4_all | A11_causal_split_transport_atom | causal_split_transport | 0.36271653324365616 | 0.388698348775506 | 0.6050073504447937 | 5.514195919036865 | 0.16666666666666666 | 1 | 0 | loss_agnostic_NDS_gate;cotangent_same_direction_gate;loss_adapter_invariance_gate |
| fallback_lower_norm_0p04_split4_all | A1_split_transfer_atom | split_transfer | -0.05888683348894119 | -0.03651375137269497 | 0.0009692574385553598 | 4.62351655960083 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p04_split4_all | A2_drift_diffusion_atom | drift_diffusion | -0.0588868074119091 | -0.03651372529566288 | 0.0009692574385553598 | 4.62351655960083 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p04_split4_all | A3_control_null_residual_atom | control_null_residual | -0.059174180030822754 | -0.03680109791457653 | 0.0007482744986191392 | 4.621641635894775 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p04_split4_all | A4_low_NDS_atom | low_NDS | -0.1072903648018837 | -0.08491728268563747 | 0.006238869857043028 | 4.413143634796143 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;loss_agnostic_B3_safety_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate |
| fallback_lower_norm_0p04_split4_all | A5_row_orthogonal_atom | row_orthogonal | -0.05888683348894119 | -0.03651375137269497 | 0.000969256623648107 | 4.62351655960083 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p04_split4_all | A6_fast_slow_source_state_atom | fast_slow_source_state | -0.06213798001408577 | -0.039764897897839546 | 0.0010169055312871933 | 4.6379170417785645 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;commutator_norm_ratio_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p04_split4_all | A7_train_flow_commutator_atom | train_flow_commutator | -0.09074397385120392 | -0.0683708917349577 | 2.5773257220862433e-05 | 4.759941577911377 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;loss_agnostic_B3_safety_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p04_split4_all | A8_low_degree_frequency_atom | low_degree_frequency | -0.07388781011104584 | -0.051514727994799614 | 0.008782981894910336 | 3.4505717754364014 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate |
| fallback_lower_norm_0p04_split4_all | A9_control_balanced_drift_atom | control_balanced_drift | -0.12011778354644775 | -0.09774470143020153 | 0.00643755029886961 | 4.337165832519531 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;loss_agnostic_B3_safety_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate |
| fallback_lower_norm_0p04_split4_all | A10_split_consensus_atom | split_consensus | 0.29259929060935974 | 0.31497237272560596 | 0.6723052859306335 | 5.17899751663208 | 0.6666666666666666 | 4 | 0 | loss_agnostic_NDS_gate |
| fallback_lower_norm_0p04_split4_all | A11_causal_split_transport_atom | causal_split_transport | 0.30513031780719757 | 0.3275033999234438 | 0.6050073504447937 | 5.514195919036865 | 0.16666666666666666 | 1 | 0 | loss_agnostic_NDS_gate;cotangent_same_direction_gate;loss_adapter_invariance_gate |
| fallback_lower_norm_0p02_split4_all | A1_split_transfer_atom | split_transfer | -0.0588727667927742 | -0.03852767311036587 | 0.0009692574385553598 | 4.62351655960083 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p02_split4_all | A2_drift_diffusion_atom | drift_diffusion | -0.05887274071574211 | -0.03852764703333378 | 0.0009692572057247162 | 4.62351655960083 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p02_split4_all | A3_control_null_residual_atom | control_null_residual | -0.05913996696472168 | -0.03879487328231335 | 0.0007482744986191392 | 4.621641635894775 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p02_split4_all | A4_low_NDS_atom | low_NDS | -0.10325319319963455 | -0.08290809951722622 | 0.006238869857043028 | 4.413143634796143 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;loss_agnostic_B3_safety_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate |
| fallback_lower_norm_0p02_split4_all | A5_row_orthogonal_atom | row_orthogonal | -0.0588727667927742 | -0.03852767311036587 | 0.000969256623648107 | 4.62351655960083 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p02_split4_all | A6_fast_slow_source_state_atom | fast_slow_source_state | -0.06189832463860512 | -0.041553230956196785 | 0.0010169055312871933 | 4.6379170417785645 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;commutator_norm_ratio_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p02_split4_all | A7_train_flow_commutator_atom | train_flow_commutator | -0.08726227283477783 | -0.0669171791523695 | 2.5773257220862433e-05 | 4.759941577911377 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;loss_agnostic_B3_safety_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate;loss_agnostic_NDS_gate |
| fallback_lower_norm_0p02_split4_all | A8_low_degree_frequency_atom | low_degree_frequency | -0.07207083702087402 | -0.05172574333846569 | 0.008782981894910336 | 3.4505717754364014 | 0.5 | 3 | 0 | loss_agnostic_B2_transfer_gain_gate;random_gap_gate;sign_flip_gap_gate;corrupt_target_gap_gate |

_仅显示前 30 / 70 rows；完整 CSV 见 artifact。_

Insight: S2 先完整保留 label-free geometry attempts 的负结果；随后新增 generic cotangent descent consensus/control-null repair。该 repair 使用 loss-interface output cotangent 生成方向，`uses_loss_interface_cotangent_for_direction=1`，但 `uses_loss_formula_specific_direction=0`，不针对 CE adapter 写特化公式。

## Part D Variational Solve / Commit

| attempt | source_combo_atom_count | DDR | control_projection_fraction | random_gap | NDS | selected_atoms | S3_variational_source_solve_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default | 3 | 1.4947019815444946 | 0.11968756467103958 | 0.1027525682002306 | 4.3088250160217285 | A12_arbitrary_cotangent_invariant_atom;A14_cotangent_low_NDS_smooth_atom;A13_cotangent_control_null_residual_atom | 1 |  |
| control_balanced_l1 | 3 | 1.203437089920044 | 0.034143127501010895 | 0.097033167257905 | 4.553361892700195 | A12_arbitrary_cotangent_invariant_atom;A13_cotangent_control_null_residual_atom | 1 |  |
| sparse_l1 | 2 | 1.7878772020339966 | 0.2279433161020279 | 0.09502170421183109 | 3.9417264461517334 | A12_arbitrary_cotangent_invariant_atom;A14_cotangent_low_NDS_smooth_atom | 1 |  |
| low_nds_only | 2 | 1.7873659133911133 | 0.2277521938085556 | 0.09540222026407719 | 3.940692663192749 | A12_arbitrary_cotangent_invariant_atom;A14_cotangent_low_NDS_smooth_atom | 1 |  |
| control_null_only | 1 | 1.0788404941558838 | 3.2768731643624018e-15 | 0.10826134122908115 | 4.644715785980225 | A13_cotangent_control_null_residual_atom | 0 | loss_agnostic_NDS_gate;Sobolev_or_RKHS_gate |
| block_restricted | 0 |  |  |  |  |  | 0 | no_S2_pass_atoms_for_attempt |

| solver_level | block_role | projection_residual_Gf | ActuationR2 | B2_transfer_gain | random_matched_B2_transfer_gain | function_displacement_cos_with_target | optimizer_state_write_fraction | S4_metric_dynamics_commit_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S4.1-readout-exact | readout_only | 0.825053657747062 | 0.31928640604019165 | -0.7407282888889313 | -1.5704880207777023 | 0.7107675075531006 | 0.0 | 0 | projection_residual_Gf_gate |
| S4.2-readout-all-train-fit | readout_only | 0.14827299314195064 | 0.978015124797821 | 0.8352429568767548 | -1.581134058535099 | 0.9890711307525635 | 0.0 | 1 |  |
| S4.3-hidden-readout-block | hidden_readout | 0.825053657747062 | 0.31928640604019165 | -0.7407282888889313 | -1.5704880207777023 | 0.7107675075531006 | 0.0 | 0 | projection_residual_Gf_gate |
| S4.4-optimizer-state-only-write | optimizer_state_only | 1.0 | 0.0 | -1.0 | -1.0 | 0.0 | 1.0 | 0 | projection_residual_Gf_gate;ActuationR2_gate;B2_transfer_gain_vs_random_gate;function_displacement_cos_gate |
| S4.5-source-state-integrated-all-train | readout_only | 0.12550980823153218 | 0.9842472672462463 | 0.8614766001701355 | -1.5947434604167938 | 0.9921687841415405 | 0.35 | 1 |  |

## Part E Arbitrary-Loss Horizon

| loss_adapter_name | attempt | source_func_h100 | source_func_h800 | source_func_h3200 | source_func_h4800 | source_loss_h800 | source_loss_h3200 | source_loss_h4800 | row_positive_count_h3200 | C3_source_formation_pass | C4_terminal_retention_pass | TargetRetentionOnly_NotTaskUseful | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Delta-LossCEAdapter | initial_commit_arbitrary_loss_adamw | -58.67281426165495 | -160.06545885939863 | -216.93245936607346 | -236.4214819885589 | -0.0010753665119409561 | -3.8709447835572064e-05 | -1.2337965017650276e-05 | 5 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-LossCEAdapter | finite_source_state_replay_400x0p10_stop800 | -58.67281426165495 | -164.21590105497557 | -218.4980809187708 | -237.73052750443878 | -0.0008161766454577446 | -3.7955076550133526e-05 | -1.2451513612177223e-05 | 5 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-LossCEAdapter | finite_source_state_replay_200x0p16_stop1200 | -58.67281426165495 | -171.29162929886417 | -225.8757740824036 | -243.74189011865175 | -0.0006846664473414421 | -3.0322320526465774e-05 | -1.1402844393160194e-05 | 2 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-LossCEAdapter | pid_source_state_replay_100_kp0p12_ki0p02_kd0p06_stop1600 | -62.04269737427153 | -184.42516263380824 | -255.1947159377902 | -268.18152570628297 | -0.000969326589256525 | -5.485626752488315e-05 | -1.723309833323583e-05 | 2 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-LossCEAdapter | initial_commit_arbitrary_loss_adamw_lr0p1 | -8.628061304929583 | -48.2155136661566 | -123.05300808673134 | -153.43230665351987 | -0.04708114266395569 | -0.005223626270890236 | -0.0015734047628939152 | 5 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-LossCEAdapter | finite_source_state_replay_400x0p05_stop1600_lr0p1 | -8.628061304929583 | -49.532791996087155 | -124.85949792474827 | -154.65625707992652 | -0.051276594400405884 | -0.0042576175183057785 | -0.0012706979177892208 | 5 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-LossCEAdapter | finite_source_state_replay_200x0p08_stop3200_lr0p1 | -8.628061304929583 | -52.22617766882283 | -142.2457010594247 | -166.4015215745046 | -0.058622971177101135 | -0.005280336365103722 | -0.0014282395131886005 | 1 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-LossCEAdapter | pid_source_state_replay_100_kp0p08_ki0p01_kd0p04_stop3200_lr0p1 | -8.916195512470695 | -56.10004713873464 | -163.3807343534097 | -183.4526612097593 | -0.06556965410709381 | -0.004915384575724602 | -0.0015286202542483807 | 1 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-LossCEAdapter | finite_source_state_replay_400x0p03_stop3200_lr0p05 | -4.197624166068599 | -29.10239863542648 | -87.01255198707446 | -116.52283374930512 | -0.09633257985115051 | -0.018061399459838867 | -0.005840230733156204 | 5 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-MSEAdapter | initial_commit_arbitrary_loss_adamw | 0.23157300883786325 | -0.015903501848103074 | -0.01311279145414801 | -0.013402326504429518 | -1.0937982040104544e-06 | -6.09121386219158e-07 | -6.338790683457773e-07 | 8 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-MSEAdapter | finite_source_state_replay_400x0p10_stop800 | 0.23157300883786325 | -0.13141984165121678 | 0.002522053187434148 | -0.029312366735981144 | -2.9455943234779625e-05 | 6.419955411729461e-08 | -1.6884535973815673e-06 | 9 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-MSEAdapter | finite_source_state_replay_200x0p16_stop1200 | 0.23157300883786325 | -0.14801497340944214 | -0.008589271252375075 | -0.008552789303667319 | -3.7135878386607146e-05 | -5.087460692720924e-07 | -2.2133018973136132e-07 | 6 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-MSEAdapter | pid_source_state_replay_100_kp0p12_ki0p02_kd0p06_stop1600 | 0.23174044319226594 | -0.009768720927807117 | -0.037035297368896325 | -0.00356693805530528 | -5.927366544256074e-07 | -3.4122351166843146e-06 | -7.454723771616045e-08 | 4 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-MSEAdapter | initial_commit_arbitrary_loss_adamw_lr0p1 | 0.5362245620913209 | 0.1913832338152468 | 0.006307865885714858 | -0.006334005590352798 | 5.4913427561587014e-05 | 1.2584553488181882e-07 | -8.230650316853882e-08 | 10 | 1 | 0 | 0 |  |
| Delta-MSEAdapter | finite_source_state_replay_400x0p05_stop1600_lr0p1 | 0.5362245620913209 | 0.15249630441917916 | 0.0045612314876577464 | -0.003991469301156125 | 4.996093957743142e-05 | 1.0211089573886056e-07 | -3.873395137965474e-08 | 9 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-MSEAdapter | finite_source_state_replay_200x0p08_stop3200_lr0p1 | 0.5362245620913209 | 0.12651362300949376 | -0.06792686199249942 | -0.00686479732368217 | 4.4348917981551494e-05 | -8.262293846428292e-06 | -9.424868729901448e-08 | 9 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-MSEAdapter | pid_source_state_replay_100_kp0p08_ki0p01_kd0p04_stop3200_lr0p1 | 0.5363672480026725 | 0.1902341058072673 | 0.005593352280429675 | -0.006652136500866934 | 5.4828713132337725e-05 | 1.171643120301269e-07 | -8.937684392584799e-08 | 10 | 1 | 0 | 0 |  |
| Delta-MSEAdapter | finite_source_state_replay_400x0p03_stop3200_lr0p05 | 0.5817943817108548 | 0.32973075215195513 | 0.005847697959483589 | 0.006522737404454415 | 0.00014712000552208337 | 5.851825335412286e-07 | 1.4025191852340413e-07 | 10 | 1 | 1 | 0 |  |
| Delta-RankingAdapter | initial_commit_arbitrary_loss_adamw | -29.527153442890295 | -129.08552960940924 | -179.0292087195243 | -197.04763940188272 | -1.7997808754444122e-07 | -1.1328666005283594e-08 | -4.51291271019727e-09 | 2 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-RankingAdapter | finite_source_state_replay_400x0p10_stop800 | -29.527153442890295 | -129.64202942151496 | -179.48516655639432 | -197.47693317236786 | -3.061606548726559e-05 | -1.1008523870259523e-08 | -4.136381903663278e-09 | 3 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-RankingAdapter | finite_source_state_replay_200x0p16_stop1200 | -29.527153442890295 | -131.21495551962465 | -181.97101217129241 | -199.82062127422694 | -0.0002544029848650098 | -4.728521162178367e-06 | -1.167045411420986e-06 | 4 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-RankingAdapter | pid_source_state_replay_100_kp0p12_ki0p02_kd0p06_stop1600 | -29.139004090390465 | -128.27810543291795 | -178.36262352897833 | -196.4211604950541 | -0.000529856828507036 | -2.7921640139538795e-05 | -5.7135566748911515e-06 | 4 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-RankingAdapter | initial_commit_arbitrary_loss_adamw_lr0p1 | -3.7694387997473084 | -28.157433277511434 | -95.36798385103859 | -118.8512072828076 | -0.015888065099716187 | 4.6566128730773926e-08 | -6.705522537231445e-08 | 3 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-RankingAdapter | finite_source_state_replay_400x0p05_stop1600_lr0p1 | -3.7694387997473084 | -28.455014369684466 | -95.98962057815717 | -119.41474832832145 | -0.023333996534347534 | -0.00014668703079223633 | -6.728805601596832e-08 | 3 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-RankingAdapter | finite_source_state_replay_200x0p08_stop3200_lr0p1 | -3.7694387997473084 | -29.204929771485602 | -101.22211612595981 | -124.1997686992991 | -0.04032912850379944 | -0.0032133120112121105 | -0.000491653336212039 | 4 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-RankingAdapter | pid_source_state_replay_100_kp0p08_ki0p01_kd0p04_stop3200_lr0p1 | -3.7626692098315955 | -27.183728555030505 | -94.12129577657122 | -117.72607094895262 | -0.06362253427505493 | -0.005404786206781864 | -0.0009646097896620631 | 6 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-RankingAdapter | finite_source_state_replay_400x0p03_stop3200_lr0p05 | -1.1057740282541793 | -17.533530018593158 | -56.45524075599805 | -82.07492113173207 | -0.02271294593811035 | -0.012000396847724915 | -0.0027596428990364075 | 4 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-GenericSourceTarget | initial_commit_arbitrary_loss_adamw | -127.47291685960045 | -1678.0109956017204 | -6936.255845557994 | -10303.09595178847 | -27337.823181152344 | -114899.248046875 | -173376.49682617188 | 8 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-GenericSourceTarget | finite_source_state_replay_400x0p10_stop800 | -127.47291685960045 | -1690.1051852581581 | -6956.636216522409 | -10320.139515876059 | -27333.401489257812 | -114894.6923828125 | -173372.15063476562 | 8 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-GenericSourceTarget | finite_source_state_replay_200x0p16_stop1200 | -127.47291685960045 | -1714.55585608543 | -6960.09554795955 | -10297.068812075291 | -27323.552001953125 | -114879.38330078125 | -173358.80834960938 | 8 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-GenericSourceTarget | pid_source_state_replay_100_kp0p12_ki0p02_kd0p06_stop1600 | -121.15845418221849 | -1580.354399081266 | -6583.815479716246 | -9880.161869259917 | -27369.83233642578 | -114977.02783203125 | -173460.10766601562 | 8 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-GenericSourceTarget | initial_commit_arbitrary_loss_adamw_lr0p1 | -8.189864362526238 | -104.56141556488993 | -577.1617388147087 | -889.5323530683307 | -2145.222686767578 | -10817.257568359375 | -16716.140655517578 | 5 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-GenericSourceTarget | finite_source_state_replay_400x0p05_stop1600_lr0p1 | -8.189864362526238 | -107.91829280249048 | -588.4913091131059 | -904.3221791580811 | -2144.0244674682617 | -10812.993606567383 | -16711.40460205078 | 5 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-GenericSourceTarget | finite_source_state_replay_200x0p08_stop3200_lr0p1 | -8.189864362526238 | -115.85112253502035 | -659.2088177582674 | -988.5027172687727 | -2141.212070465088 | -10788.483612060547 | -16685.271850585938 | 4 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-GenericSourceTarget | pid_source_state_replay_100_kp0p08_ki0p01_kd0p04_stop3200_lr0p1 | -7.744427023772369 | -85.69727308223861 | -490.4675076166021 | -778.513017476299 | -2152.1589698791504 | -10847.404373168945 | -16749.196990966797 | 8 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-GenericSourceTarget | finite_source_state_replay_400x0p03_stop3200_lr0p05 | -3.168709992663114 | -44.21943609913951 | -255.06866243799885 | -407.5723936315653 | -853.1079044342041 | -5032.743026733398 | -7942.021347045898 | 3 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |

Evidence chain: S5 同时记录 function-retention source 和 loss-interface source。C3/C4 必须同时看 source_func、source_loss、matched controls 与 AUC_loss_time；如果 source_func 过但 source_loss 负，会被写成 TargetRetentionOnly_NotTaskUseful。

## Part F KAN Mapping

| carrier | attempt | periodic_interval | periodic_scale | loss_adapter_name | KAN_source_func_h3200 | KAN_source_loss_h3200 | KAN_source_loss_h4800 | KAN_specific_delta_vs_MLP_same_metric | R4800_over_3200_func | loss_agnostic_efficiency_gate_pass | KAN_source_channel_decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | kan_readout_replay_100x0p12 | 100 | 0.12 | Delta-MSEAdapter | 1.35784512758255 | -0.0008478505478706211 | -0.00032978825038298965 | 1.3515372616968353 | 1.1412155361467682 | 1 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-CHE | kan_readout_replay_400x0p03_repair | 400 | 0.03 | Delta-MSEAdapter | 3.34799000620842 | -3.2290946364810225e-05 | -3.0984950171841774e-05 | 3.341682140322705 | 1.002384375389749 | 1 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-CHE | kan_readout_replay_800x0p015_repair | 800 | 0.015 | Delta-MSEAdapter | 3.2093923538923264 | -2.3728211317575187e-05 | -2.3459125827685057e-05 | 3.2030844880066116 | 0.9994561810727083 | 1 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-CHE | kan_readout_initial_only_no_periodic_repair | 100000 | 0.0 | Delta-MSEAdapter | 3.594503864645958 | -1.184850034263718e-05 | -2.0067516256216944e-06 | 3.588195998760243 | 1.0159727953633424 | 1 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-FOU | kan_readout_replay_100x0p12 | 100 | 0.12 | Delta-MSEAdapter | 2.1501737385988235 | -2.638707974256249e-05 | -1.89076372407726e-05 | 2.143865872713109 | 1.0111633666853743 | 1 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-FOU | kan_readout_replay_400x0p03_repair | 400 | 0.03 | Delta-MSEAdapter | 1.2071230709552765 | -3.1645599847252015e-06 | -2.951610440504737e-06 | 1.2008152050695617 | 1.00027969541231 | 1 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-FOU | kan_readout_replay_800x0p015_repair | 800 | 0.015 | Delta-MSEAdapter | 1.3939154483377934 | -4.892771130471374e-07 | -4.78223824984525e-07 | 1.3876075824520786 | 1.0000878810268086 | 1 | KANRetainedSourceOpened |  |
| D-FOU | kan_readout_initial_only_no_periodic_repair | 100000 | 0.0 | Delta-MSEAdapter | 2.2672564671956934 | -9.121969990483073e-11 | -8.482103208087182e-09 | 2.2609486013099787 | 0.9990332494384428 | 1 | KANRetainedSourceOpened |  |

## 4GPU Queue

| task_id | line | gpu | command | status | returncode | start_time | end_time | duration_sec | log_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S2_source_atom_generation | S2 | 1 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_source_atom_generation.py --seed 2211 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 | completed | 0 | 2026-06-07 21:52:37 +0800 | 2026-06-07 21:52:40 +0800 | 3.0017385482788086 | /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/logs/S2_source_atom_generation.log |
| S3_variational_source_solve | S3 | 1 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_variational_source_solve.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 | completed | 0 | 2026-06-07 21:52:40 +0800 | 2026-06-07 21:52:42 +0800 | 2.0019776821136475 | /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/logs/S3_variational_source_solve.log |
| S0_18_code_loss_interface_truth_gate | S0.18 | 0 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_s018_truth_gate.py --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 | completed | 0 | 2026-06-07 21:52:37 +0800 | 2026-06-07 21:52:44 +0800 | 7.006624698638916 | /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/logs/S0_18_code_loss_interface_truth_gate.log |
| S4_metric_commit | S4 | 3 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_metric_commit.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 | completed | 0 | 2026-06-07 21:52:42 +0800 | 2026-06-07 21:52:44 +0800 | 2.0019690990448 | /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/logs/S4_metric_commit.log |
| S1_arbitrary_cotangent_efficiency | S1 | 0 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_basis_efficiency.py --device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1 --seed 2211 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 | completed | 0 | 2026-06-07 21:52:44 +0800 | 2026-06-07 21:52:48 +0800 | 4.003909349441528 | /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/logs/S1_arbitrary_cotangent_efficiency.log |
| S5_arbitrary_loss_horizon | S5 | 3 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_arbitrary_loss_horizon.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --seed 2211 | completed | 0 | 2026-06-07 21:52:44 +0800 | 2026-06-07 22:01:08 +0800 | 504.0323419570923 | /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/logs/S5_arbitrary_loss_horizon.log |
| S6_kan_mapping | S6 | 2 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 --seed 2211 | completed | 0 | 2026-06-07 22:01:08 +0800 | 2026-06-07 22:03:46 +0800 | 158.01152062416077 | /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/logs/S6_kan_mapping.log |
| S7_finalize_packet_recap | S7 | 0 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_11_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11 | completed | 0 | 2026-06-07 22:03:46 +0800 | 2026-06-07 22:03:47 +0800 | 1.0017311573028564 | /home/chengshun.wang/DG-LCA/results/v22_11_arbitrary_loss_constructive_fu_basis_efficiency_4gpu/official_v22_11/logs/S7_finalize_packet_recap.log |

| execution_contract_violation | reason | max_idle_gap_sec |
| --- | --- | --- |
| 0 | no_dependency_free_runnable_task_waited_on_idle_gpu_over_10min | 0 |

## Failure Taxonomy

| blocker | repair_or_next_direction |
| --- | --- |
| arbitrary_loss_efficiency_gate | replace PyTorch fallback with kernel-native arbitrary-cotangent forward/backward/update path; keep same cotangent suite |
|  | if S2 failed, continue lower norm / split_count / block-restricted / control-null source atom repairs |
|  | if source_loss failed, redefine source atoms for arbitrary loss dynamics instead of tuning PID/replay only |
|  | if KAN mismatch, focus basis-channel mapping; if efficiency blocked, return to Line B |

## 修改记录

- 扩展 `dgkan/fu/loss_interface.py`：新增 `LossInterface`、`GenericUpstreamCotangent`、CE/MSE/ranking/policy-preference adapters 与 unit tests；旧 CE/Brier helper 保留供历史脚本使用。
- 新增 `dgkan/fu/upstream_cotangent.py`：统一生成 Delta-Gaussian/StableRandom/SourceTarget/CE/MSE/Ranking/Policy smoke cotangent suite，并记录 cotangent contract。
- 新增 `dgkan/profiling/efficiency_v22_11.py`：用 arbitrary upstream cotangent 真实测 forward/VJP/update/full-step/memory ratio；早期版本显式暴露 autograd fallback blocker。
- 修复 `dgkan/profiling/efficiency_v22_11.py`：新增 manual upstream-VJP/update 路径，D-CHE/D-FOU 低阶/低频 repair variant，真实落盘 `fallback_kernel_used=0`、`manual_upstream_vjp_used=1`、`official_fused_kernel_complete=0`。
- 修复 `experiments/run_v22_11_source_atom_generation.py`：新增 `fallback_arbitrary_cotangent_invariant_0p04_split4_all`，构造 A12-A15 generic-cotangent source atoms，并记录 `uses_loss_interface_cotangent_for_direction=1` 与 cotangent invariance readback。
- 修复 `experiments/run_v22_11_arbitrary_loss_horizon.py`：新增低 lr arbitrary-loss continuation 与 `finite_source_state_replay_400x0p03_stop3200_lr0p05` repair，解决 MSE adapter 下 h3200/h4800 terminal retention blocker。
- 修复 `experiments/run_v22_11_kan_mapping.py`：新增 KAN replay scale/interval repair scan，D-FOU 在低 replay/no-periodic attempts 下通过 source_func 与 source_loss gate；强 replay rows 保留为 `KANTargetRetentionOnly` 负例。
- 新增 `experiments/run_v22_11_*` runner：S0.18 truth gate、S1 efficiency、S2 source atom、S3 variational、S4 commit、S5 arbitrary-loss horizon、S6 KAN mapping、4GPU queue 与 finalizer。
- S5 horizon 从 v22.10 target-retention replay 升级为 generic loss-interface continuation；FU/controls 使用同一 loss adapter，FU 注入不读取 adapter 类型。
- Finalizer 生成 `v22_11_results_bundle.zip`、`v22_11_code_review_packet.zip` 和 artifact index；所有 route/blocker 来自落盘 artifact。

## 补充分析 / Evidence Chain

- v22.11 已清除 v22.10 的 S1 CE-targeted formal blocker：runner 接收任意 output cotangent，并在 CE/MSE/ranking/source-target/random cotangent 下测 VJP/update。
- 当前 efficiency 证据已不再是 PyTorch autograd fallback；但 fused official kernel 仍未完成，S1 结论限定为 manual upstream-VJP efficiency closure。
- Source-side 证据必须分两层读：S2/S3/S4 证明 label-free target displacement 可构造并写回；S5 才裁决它是否在任意 loss training dynamics 中对 loss neutral-positive。
- KAN mapping 被设计成条件阶段，且被 S1 efficiency gate 约束；即便 KAN source_func 为正，也不能越过 efficiency blocker 写 promotion。
- 完整 artifact 清单在 `v22_11_artifact_index.csv`，正文只列关键表。
