# DG-KAN v22.12 ArbitraryLossOperatorFU BasisEfficiency 4GPU 实验结果复盘

生成时间：2026-06-08 11:30:26 +0800

## Route

- route: `R7-KANSourceExistsEfficiencyBlocked`
- exploration_promotion_allowed: `1`
- official_promotion_allowed: `0`
- blocking_metric: `D-CHE_arbitrary_cotangent_efficiency_blocked_after_source_exists;D-CHE_source_loss_h4800_gate_after_source_exists`
- minimum_effective_progress: `A-CodeTruth;B-Efficiency;C-Functional;D-KAN`
- next_codex_action: `continue kernel-native arbitrary-cotangent officialization or repair adapter/KAN blockers according to v22.12 failure taxonomy`
- S0.19 pass: 1
- D-CHE/D-FOU/D-RAT/D-RBF S1 pass: 0 / 1 / 0 / 0
- official_fused_kernel_complete_rows: 0
- S2/S3/S4 pass rows: 2 / 3 / 3
- S5 C3/C4 adapter pass count: 2 / 1
- S5 C3/C4 adapter list: `Delta-MSEAdapter;Delta-RankingAdapter` / `Delta-MSEAdapter`
- KAN route / pass rows: `S6-DFOUSourceOpened_DCHESourceExistsEfficiencyBlocked` / 2
- results bundle: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_results_bundle.zip`
- code review packet: `/home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/v22_12_code_review_packet.zip`

## Executive Summary

- v22.12 route=`R7-KANSourceExistsEfficiencyBlocked`；exploration_promotion_allowed=1，official_promotion_allowed=0。
- S1 执行 arbitrary upstream cotangent repair variants，manual_upstream_vjp_rows=360，official_fused_kernel_complete_rows=0；因此不写 official fused success。
- best S5 FU row: loss_adapter=`Delta-MSEAdapter`, attempt=`source_loss_pid20_lr0p0005_clip0p03`, source_func_h3200=1.36857182165343, source_loss_h3200=0.021983585043926723, C3=1, C4=1。
- 所有关键结论来自本轮落盘 CSV/JSON/PT/log；未执行或 gate 阻塞阶段以 blocker 记录，没有写成通过。

## 本轮修复过程

- 初始 v22.12 不是 official success：D-FOU 已经通过 K13 fan-scale source-loss boundary rows 打开 source channel，但 D-CHE 在 K1/K8/K10/K11 中仍表现为 `KANSourceChannelMismatchConfirmed` 或 `KANTargetRetentionOnly`。这意味着不能把 D-FOU 的 carrier-specific efficiency pass 借给 D-CHE。
- 这轮重点检查 D-CHE readout commit。诊断发现 `frozen_readout_features` 的自然顺序是 `(hidden,basis)`，而旧 readout solve 直接 reshape 到 `w2(hidden,class,basis)`，会把 class/basis layout 混在一起。修复没有回写历史 K1-K13 row，而是新增 K14 corrected-readout-layout row，保留前序负例。
- K14 先做缩短 horizon diagnostic grid；完整 grid 和 full S6 重跑都因为重复计算过多被 kill 并写入执行日志，随后改成 K14-only incremental runner：复用已落盘 S6 matrix，只重新计算新增 K14 row。
- 最终 K14 把 route 从 `S6-DFOUReadoutSourceOpened_DCHEMismatch` 推进到 `S6-DFOUSourceOpened_DCHESourceExistsEfficiencyBlocked`：D-CHE source-exists 有证据，但仍被 D-CHE efficiency 与 h4800 source-loss gate 阻塞。

### K14 执行尝试摘要

| timestamp | status | note |
| --- | --- | --- |
| 2026-06-08 05:54:01 +0800 | killed | first K14 official S6 run exceeded useful time budget with identity/inputcross bracket rows; killed pid 390432 and reduced official K14 matrix to fan_scale_25x0p08 row, leaving brackets as diagnostics only |
| 2026-06-08 05:54:01 +0800 | completed | post K14 single-row route repair compile check passed |
| 2026-06-08 06:02:00 +0800 | killed | second full S6 rerun with single K14 row still exceeded useful repeated-row recomputation budget; killed pid 395686 and moved K14 officialization to incremental K14-only runner that reuses existing S6 rows and computes only the new row |
| 2026-06-08 06:02:00 +0800 | completed | post K14-only repair runner compile check passed |
| 2026-06-08 06:02:39 +0800 | completed | route=S6-DFOUReadoutSourceOpened_DCHEMismatch K14_decision=KANSourceChannelMismatchConfirmed K14_source_func_h3200=8.544725932180882 K14_source_loss_h3200=0.00013344238095669425 blocker=D-CHE_source_channel_mismatch_after_DFOU_open |
| 2026-06-08 06:04:55 +0800 | completed | post K14 source-exploration/terminal-loss split compile check passed |
| 2026-06-08 06:05:32 +0800 | completed | route=S6-DFOUSourceOpened_DCHESourceExistsEfficiencyBlocked K14_decision=KANEfficiencyContractBlocked K14_source_func_h3200=8.544725932180882 K14_source_loss_h3200=0.00013344238095669425 blocker=D-CHE_arbitrary_cotangent_efficiency_blocked_after_source_exists;D-CHE_source_loss_h4800_gate_after_source_exists |

### K14 关键证据

| carrier | attempt | mapping_strategy | basis_estimate_fit_cosine | basis_commit_projection_residual | KAN_source_func_h3200 | KAN_source_func_h4800 | KAN_source_loss_h3200 | KAN_source_loss_h4800 | KAN_specific_delta_vs_MLP_same_metric | K14_source_exploration_pass | K14_terminal_source_loss_pass | carrier_specific_efficiency_pass | KAN_source_channel_decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | K14_dche_corrected_readout_layout_fan_scale_25x0p08 | K14_dche_corrected_readout_layout_low_degree_commit | 0.9972032308578491 | 0.08182405680418015 | 8.544725932180882 | 8.541018530726433 | 0.00013344238095669425 | -8.785344834905118e-05 | 7.176154110527452 | 1 | 0 | 0 | KANEfficiencyContractBlocked | KANEfficiencyContractBlocked;source_loss_h4800_gate |

Interpretation: K14 的 `basis_estimate_fit_cosine=0.9972032308578491` 和 `basis_commit_projection_residual=0.08182405680418015` 说明 corrected layout 后 readout/basis target 能被拟合；`KAN_specific_delta_vs_MLP_same_metric=7.176154110527452` 与 h3200 正 source_loss 说明 D-CHE source channel 已经不是原来的纯 mismatch。但 h4800 source_loss 为负，且 D-CHE S1 carrier efficiency pass 仍为 0，所以只能写成 source-exists / efficiency-blocked。

### K13/K14 前后对照

| carrier | attempt | mapping_strategy | KAN_source_func_h3200 | KAN_source_loss_h3200 | KAN_source_loss_h4800 | KAN_specific_delta_vs_MLP_same_metric | carrier_specific_efficiency_pass | KAN_source_channel_decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-FOU | K13_dfou_fan_scale_loss_guard_25x0p10 | K13_fan_scale_source_loss_boundary_repair | 3.3744351658970118 | 6.205737008713186e-07 | 1.2992959455004893e-06 | 2.005863344243582 | 1 | KANRetainedSourceOpened |  |
| D-FOU | K13_dfou_fan_scale_loss_guard_25x0p08 | K13_fan_scale_source_loss_boundary_repair | 3.5989733152091503 | 3.060299604840111e-06 | 1.0244675650028512e-06 | 2.2304014935557204 | 1 | KANRetainedSourceOpened |  |
| D-CHE | K1_readout_replay_100x0p08 | K1_readout_source_channel | 0.5010721981525421 | 0.0002690380497369915 | 6.451907393056899e-05 | -0.8674996235008878 | 0 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-CHE | K8_readout_replay_25x0p12_repair | K8_readout_replay_scale_interval_repair | 1.1868340224027634 | 0.00039954425301402807 | -0.08108080388046801 | -0.18173779925066658 | 0 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-CHE | K10_readout_gain1p0_25x0p12_repair | K1_readout_source_channel | 1.108336627483368 | 0.0020577211398631334 | 0.00023859716020524502 | -0.260235194170062 | 0 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-CHE | K11_basis_linearized_rank12_cap4p0_initial_only | K11_basis_linearized_commit | 0.5665958322824736 | -6.309381836433481e-11 | -3.892423482502743e-06 | -0.8019759893709564 | 0 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-CHE | K14_dche_corrected_readout_layout_fan_scale_25x0p08 | K14_dche_corrected_readout_layout_low_degree_commit | 8.544725932180882 | 0.00013344238095669425 | -8.785344834905118e-05 | 7.176154110527452 | 0 | KANEfficiencyContractBlocked | KANEfficiencyContractBlocked;source_loss_h4800_gate |

Interpretation: D-FOU K13 是 clean pass，因为 carrier_specific_efficiency_pass=1 且 source_loss_h3200/h4800 非负；D-CHE K1/K8/K10/K11 即便有局部 positive source_loss，也低于 MLP same-metric 或触发 target-only/mismatch。K14 第一次把 D-CHE 的 `delta_vs_MLP` 推到正数，但官方 gate 仍要求 D-CHE 自己的 efficiency pass 与 terminal source-loss 同时成立。

## 为什么仍不允许 Promotion

| gate | required | observed | decision |
| --- | --- | --- | --- |
| S0.19 code/semantic truth | pass=1 | S0.19=1 | pass |
| arbitrary-cotangent official kernel | official_fused_kernel_complete_rows > 0 | official_fused_kernel_complete_rows=0; manual_upstream_vjp_rows=360 | blocked |
| operator FU arbitrary-loss robustness | C3 adapters >= 2; C4 adapters >= 2 for official | C3=2 (Delta-MSEAdapter;Delta-RankingAdapter); C4=1 (Delta-MSEAdapter) | exploration_pass_official_blocked |
| KAN carrier-specific source | no carrier borrows another carrier's efficiency pass | S6=S6-DFOUSourceOpened_DCHESourceExistsEfficiencyBlocked; KAN_pass_rows=2 | exploration_pass_with_D-CHE_blocker |
| D-CHE K14 terminal/effectiveness | carrier_specific_efficiency_pass=1 and source_loss_h4800 >= 0 | carrier_specific_efficiency_pass=0; K14_terminal_source_loss_pass=0; source_loss_h4800=-8.785344834905118e-05 | blocked |

- `exploration_promotion_allowed=1` 的含义只是：代码 truth、manual arbitrary-cotangent runner、operator source、至少两个 C3 adapter、至少一个 C4 adapter、以及 KAN source evidence 都存在。
- `official_promotion_allowed=0` 有三个硬原因：第一，S1 没有任何 `official_fused_kernel_complete` row；第二，S5 的 C4 只有 Delta-MSEAdapter，Ranking 只到 C3 未到 terminal retention；第三，K14 虽然打开 D-CHE source-exists，但 D-CHE `carrier_specific_efficiency_pass=0` 且 h4800 source_loss 为负。
- 因此本轮有效进展应写作 `A-CodeTruth;B-Efficiency;C-Functional;D-KAN` 的 exploration progress，而不是 official promotion。下一步应优先做 D-CHE kernel-native arbitrary-cotangent officialization，并同时修 K14 h4800 source-loss retention。

## Part A Code / Semantic Truth Gate

| check | pass | metric | value | blocker |
| --- | --- | --- | --- | --- |
| required_source_files | 1 | exists | 16/16 |  |
| compileall_repo | 1 | py_compile | 0 |  |
| import_closure_repo | 1 | import | 0 |  |
| loss_interface_tests | 1 | tests | 6/6 |  |
| arbitrary_upstream_cotangent_tests | 1 | tests | 1/1 |  |
| v22_12_efficiency_profiler_tests | 1 | tests | 1/1 |  |
| linec_fast_golden | 1 | tests | 9 |  |
| linec_channel_golden | 1 | tests | 8 |  |
| source_atom_generator_tests | 1 | tests | 2/2 |  |
| variational_solver_tests | 1 | tests | 2/2 |  |
| constructive_commit_tests | 1 | tests | 1/1 |  |
| wrong_global_efficiency_gate_detected | 1 | source_semantic | 0 |  |
| carrier_specific_efficiency_pass_consistency | 1 | source_semantic | 1 |  |
| promotion_semantics_split | 1 | source_semantic | 1 |  |
| CE_specific_formula_in_v22_12_direction_path | 1 | semantic | 0 |  |
| CE_adapter_only_used_as_LossInterface | 1 | semantic | 1 |  |
| legacy_CE_helper_not_on_official_path | 1 | semantic | 1 |  |
| csv_claimed_exists_but_zip_missing_count | 1 | zip_required_compare | 0 |  |
| self_contained_compileall | 1 | clean_unzip_compile | 0 |  |
| self_contained_import_check | 1 | clean_unzip_import | 0 |  |

## Part B Arbitrary-Cotangent Efficiency

| carrier | profile_rows | batch_pass_rows | cotangent_pass_rows | near_E1_rows | fallback_rows | manual_upstream_vjp_rows | official_fused_kernel_complete_rows | best_forward_ratio | best_step_ratio | best_memory_ratio | decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | 96 | 0 | 0 | 48 | 0 | 96 | 0 | 1.3586531094428782 | 0.8295109437563136 | 0.947102275544098 | ArbitraryCotangentNearE1Only | ratio_or_fallback_kernel_gate_failed |
| D-FOU | 24 | 4 | 6 | 24 | 0 | 24 | 0 | 0.8020237704330234 | 0.7779178473989167 | 0.9746170244410292 | ArbitraryCotangentEfficiencyPass |  |
| D-RAT | 96 | 0 | 0 | 8 | 0 | 96 | 0 | 1.773714762352284 | 1.351422890869797 | 1.0037420335613636 | ArbitraryCotangentNearE1Only | ratio_or_fallback_kernel_gate_failed |
| D-RBF | 120 | 2 | 4 | 50 | 0 | 120 | 0 | 1.1030228230880428 | 0.9085159971938921 | 1.0 | ArbitraryCotangentNearE1Only | ratio_or_fallback_kernel_gate_failed |

| carrier | repair_attempt | attempt_rows | pass_rows | near_E1_rows | best_step_ratio | official_fused_kernel_complete_rows | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | CHE22.12-R1-k3-low-degree-upstream-vjp | 24 | 0 | 24 | 1.0348789431956065 | 0 | repair_variant_ratio_gate_failed |
| D-CHE | CHE22.12-R2-k5-gradbuf-no-materialize | 24 | 0 | 0 | 1.4704751028659724 | 0 | repair_variant_ratio_gate_failed |
| D-CHE | CHE22.12-R3-readout-only-fast-path | 24 | 0 | 24 | 0.8295109437563136 | 0 | repair_variant_ratio_gate_failed |
| D-CHE | CHE22.12-R4-dense-degree-debug-telemetry | 24 | 0 | 0 | 1.6818301944663607 | 0 | repair_variant_ratio_gate_failed |
| D-FOU | FOU22.12-R3-tablelookup-bandreadout-reconfirm | 24 | 24 | 24 | 0.7779178473989167 | 0 |  |
| D-RAT | RAT22.12-R1-num-den-fused-vjp | 24 | 0 | 2 | 1.3875335250797496 | 0 | repair_variant_ratio_gate_failed |
| D-RAT | RAT22.12-R2-reciprocal-approx-safety-clamp | 24 | 0 | 0 | 1.6434105012992073 | 0 | repair_variant_ratio_gate_failed |
| D-RAT | RAT22.12-R3-telemetry-free-train-path | 24 | 0 | 3 | 1.351422890869797 | 0 | repair_variant_ratio_gate_failed |
| D-RAT | RAT22.12-R4-num-den-blockwise-shared-cotangent | 24 | 0 | 3 | 1.367525217354401 | 0 | repair_variant_ratio_gate_failed |
| D-RBF | RBF22.12-R1-local-K4-no-dense-train-path | 24 | 3 | 24 | 0.9836195704803111 | 0 |  |
| D-RBF | RBF22.12-R2-local-K2-compact-support-path | 24 | 0 | 1 | 1.3084410219016591 | 0 | repair_variant_ratio_gate_failed |
| D-RBF | RBF22.12-R3-active-center-mask-hard-sparse | 24 | 0 | 1 | 1.3446792941519916 | 0 | repair_variant_ratio_gate_failed |
| D-RBF | RBF22.12-R4-exp-approximation-path | 24 | 0 | 0 | 1.5964480616057701 | 0 | repair_variant_ratio_gate_failed |
| D-RBF | RBF22.12-R5-local-backward-fused-scatter | 24 | 4 | 24 | 0.9085159971938921 | 0 |  |

Insight: v22.12 的 S1 不再使用 CE trainpath；repair variants 实际计时 forward/manual VJP/update/component telemetry。`official_fused_kernel_complete_rows=0` 是保留 blocker，不因 manual path ratio pass 而改写。

## Part C Operator Atoms

| operator_id | cotangent_types_supported | expected_loss_linear_gain | random_matched_expected_gain | adapter_invariance_score | control_projection_fraction | NDS | metric_energy_Sobolev | S2_operator_atom_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O1_UpstreamCotangentDrift | Delta-Gaussian;Delta-StableRandom;Delta-SourceTarget;Delta-LossCEAdapter;Delta-MSEAdapter;Delta-RankingAdapter | 0.08024626860746291 | 0.002790295503887208 | 1.0 | 0.16057375073432922 | 3.381873607635498 | 0.0651661604642868 | 0 | NDS_gate |
| O2_SignalReservoirProjection | Delta-Gaussian;Delta-StableRandom;Delta-SourceTarget;Delta-LossCEAdapter;Delta-MSEAdapter;Delta-RankingAdapter | 0.08009739427380207 | 0.002790295503887208 | 1.0 | 0.15459196269512177 | 3.383666515350342 | 0.06605156511068344 | 0 | NDS_gate |
| O3_ControlNullOperator | Delta-Gaussian;Delta-StableRandom;Delta-SourceTarget;Delta-LossCEAdapter;Delta-MSEAdapter;Delta-RankingAdapter | 0.05993201889195158 | 0.002790295503887208 | 1.0 | 1.1700818265680812e-12 | 3.2026185989379883 | 0.0653250515460968 | 0 | NDS_gate |
| O4_LowNDSOperator | Delta-Gaussian;Delta-StableRandom;Delta-SourceTarget;Delta-LossCEAdapter;Delta-MSEAdapter;Delta-RankingAdapter | 0.05785776260747374 | 0.002790295503887208 | 1.0 | 0.16320472955703735 | 0.3346346616744995 | 0.04152429848909378 | 1 |  |
| O5_TrainFlowCommutatorOperator | Delta-Gaussian;Delta-StableRandom;Delta-SourceTarget;Delta-LossCEAdapter;Delta-MSEAdapter;Delta-RankingAdapter | 0.08024626860746291 | 0.002790295503887208 | 1.0 | 0.16057375073432922 | 3.381873607635498 | 0.0651661604642868 | 0 | NDS_gate |
| O6_FastSlowSourceOperator | Delta-Gaussian;Delta-StableRandom;Delta-SourceTarget;Delta-LossCEAdapter;Delta-MSEAdapter;Delta-RankingAdapter | 0.07956698796126223 | 0.002790295503887208 | 1.0 | 0.1611330360174179 | 2.538123846054077 | 0.060979776084423065 | 0 | NDS_gate |
| O7_RowOrthogonalMatrixOperator | Delta-Gaussian;Delta-StableRandom;Delta-SourceTarget;Delta-LossCEAdapter;Delta-MSEAdapter;Delta-RankingAdapter | 0.07603954930595098 | 0.002790295503887208 | 0.8333333333333334 | 0.21118216216564178 | 4.795553684234619 | 0.07502222061157227 | 0 | NDS_gate |
| O9_AdapterInvariantOperator | Delta-Gaussian;Delta-StableRandom;Delta-SourceTarget;Delta-LossCEAdapter;Delta-MSEAdapter;Delta-RankingAdapter | -0.01598137651270311 | 0.002790295503887208 | 0.3333333333333333 | 0.20576590299606323 | 3.216639757156372 | 0.06300657242536545 | 0 | adapter_invariance_gate;expected_loss_linear_gain_gate;NDS_gate |
| O10_NonRandomSourceLossBalancedOperator | Delta-Gaussian;Delta-StableRandom;Delta-SourceTarget;Delta-LossCEAdapter;Delta-MSEAdapter;Delta-RankingAdapter | 0.006577638895388797 | 0.002790295503887208 | 0.8333333333333334 | 8.378063648706302e-05 | 1.1243727207183838 | 0.04247770085930824 | 1 |  |

Insight: operator atoms 直接消费 LossInterface output cotangent，记录 linearity/homogeneity、adapter variance、control projection 与 metric/NDS energy；labels 只允许在 supervised adapter 内部产生 cotangent，不进入 FU core。

## Part D Variational Solve / Commit

| attempt | operator_combo_atom_count | selected_operators | adapter_invariance_score | expected_loss_linear_gain | random_matched_expected_gain | control_projection_fraction | NDS | operator_variational_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| source_loss_balanced_nonrandom_consensus | 1 | O10_NonRandomSourceLossBalancedOperator | 0.8333333333333334 | 0.006577638895388797 | 0.002790295503887208 | 8.378063648706302e-05 | 1.1243728399276733 | 1 |  |
| default_adapter_balanced | 2 | O4_LowNDSOperator;O10_NonRandomSourceLossBalancedOperator | 0.9166666666666667 | 0.032217700751431266 | 0.002790295503887208 | 0.08164425509676221 | 0.4158661365509033 | 1 |  |
| cotangent_norm_normalized_repair | 4 | O2_SignalReservoirProjection;O3_ControlNullOperator;O4_LowNDSOperator;O9_AdapterInvariantOperator | 0.8333333333333334 | 0.04547644981513107 | 0.002790295503887208 | 0.1308906488123481 | 1.8312294483184814 | 0 | NDS_gate |
| source_loss_aware_low_NDS_repair | 4 | O3_ControlNullOperator;O4_LowNDSOperator;O6_FastSlowSourceOperator;O10_NonRandomSourceLossBalancedOperator | 0.9583333333333334 | 0.05098360208901908 | 0.002790295503887208 | 0.0811053865530281 | 1.4289066791534424 | 1 |  |

| solver_level | block_role | projection_residual_Gf | ActuationR2 | B2_transfer_gain | random_matched_B2_transfer_gain | function_displacement_cos_with_target | optimizer_state_write_fraction | S4_operator_metric_commit_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S4.1-readout-exact | readout_only | 1.2673271137090354 | -0.6067163944244385 | -2.0978871285915375 | -1.382032960653305 | 0.1464727520942688 | 0.0 | 0 | projection_residual_Gf_gate;ActuationR2_gate;B2_transfer_gain_vs_random_gate;function_displacement_cos_gate |
| S4.2-readout-all-train-fit | readout_only | 0.1187492700835116 | 0.9858933687210083 | 0.8838283270597458 | -1.203414410352707 | 0.9931895136833191 | 0.0 | 1 |  |
| S4.3-hidden-readout-block | hidden_readout | 1.2673271137090354 | -0.6067163944244385 | -2.0978871285915375 | -1.382032960653305 | 0.1464727520942688 | 0.0 | 0 | projection_residual_Gf_gate;ActuationR2_gate;B2_transfer_gain_vs_random_gate;function_displacement_cos_gate |
| S4.4-optimizer-state-only-write | optimizer_state_only | 1.0 | -0.0003724098205566406 | -1.0 | -1.0 | 0.0 | 1.0 | 0 | projection_residual_Gf_gate;ActuationR2_gate;B2_transfer_gain_vs_random_gate;function_displacement_cos_gate |
| S4.5-source-state-integrated-all-train | readout_only | 0.09193171138739693 | 0.9915454387664795 | 0.9089783206582069 | -1.2217497676610947 | 0.9958831071853638 | 0.35 | 1 |  |
| S4.6-operator-readout-block-solve | readout_only | 0.07266748157694948 | 0.9947174787521362 | 0.9280222952365875 | -1.2353231273591518 | 0.9974051117897034 | 0.25 | 1 |  |

## Part E Arbitrary-Loss Horizon

| loss_adapter_name | attempt | source_func_h100 | source_func_h800 | source_func_h3200 | source_func_h4800 | source_loss_h800 | source_loss_h3200 | source_loss_h4800 | row_positive_count_h3200 | C3_source_formation_pass | C4_terminal_retention_pass | TargetRetentionOnly_NotTaskUseful | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Delta-LossCEAdapter | operator_initial_commit_adamw | -20.028896887215947 | -43.01610614269106 | -58.77103262206927 | -64.11886162737082 | -0.001083737937733531 | -2.1494022803381085e-05 | -5.923749995417893e-06 | 1 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-LossCEAdapter | adapter_balanced_replay_400x0p04_stop1600 | -13.102852378741968 | -36.46792641941863 | -54.20484550788483 | -59.52494757110113 | -0.00553753599524498 | -0.0001155759091489017 | -2.6674591936171055e-05 | 1 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-LossCEAdapter | source_loss_aware_replay_400x0p02_stop3200 | -6.658367035037047 | -24.51115318635183 | -45.06292153533914 | -50.91323229309448 | -0.02820054441690445 | -0.0013303628657013178 | -0.00034083210630342364 | 1 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-LossCEAdapter | cotangent_norm_normalized_lr0p1 | -3.46253938442121 | -16.615655123921666 | -34.123106444707105 | -41.22774030104558 | -0.06305333971977234 | -0.006622405722737312 | -0.0018185640219599009 | 2 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-LossCEAdapter | terminal_preservation_800x0p015_stop4800_lr0p05 | -1.2797149234094212 | -10.824654368954084 | -26.50887335265397 | -33.90136204151463 | -0.09642815589904785 | -0.024166762828826904 | -0.009405440650880337 | 2 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-LossCEAdapter | terminal_preservation_400x0p025_stop6400_lr0p03 | -0.14400345606700427 | -7.664610247007333 | -22.324563972183814 | -30.275606268789918 | -0.10605445504188538 | -0.04507491737604141 | -0.021732443943619728 | 2 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-LossCEAdapter | terminal_preservation_200x0p018_stop6400_lr0p02 | 0.5448853809534722 | -5.604942213257913 | -18.64469338376371 | -26.53981067416664 | -0.10256731510162354 | -0.06558319181203842 | -0.03611099347472191 | 2 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-LossCEAdapter | terminal_preservation_100x0p010_stop6400_lr0p01 | 1.31113435334829 | -2.7952518968072018 | -11.992742966915742 | -18.065652910235993 | -0.07792699337005615 | -0.10115620493888855 | -0.07485026121139526 | 5 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-LossCEAdapter | source_loss_pid20_lr0p0005_clip0p03 | 1.8726387822548118 | 1.7233060815758412 | 0.6459863083948023 | -0.2730656404737577 | 0.0438610315322876 | -0.2499856948852539 | -0.3920425772666931 | 10 | 0 | 0 | 1 | source_func_or_source_loss_or_control_gate_failed |
| Delta-MSEAdapter | operator_initial_commit_adamw | 0.35732032874724795 | -0.018708517668904023 | -0.005927903025040648 | -0.07752752414821018 | -1.03573105718624e-05 | -1.4714697513795727e-06 | -0.00014373909612413333 | 6 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-MSEAdapter | adapter_balanced_replay_400x0p04_stop1600 | 0.6443211983367604 | -0.032233929136759154 | -0.01090058915468528 | 0.001143011039854036 | -4.5127047997084446e-05 | -3.0538568488724938e-06 | 1.058404563991644e-07 | 8 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-MSEAdapter | source_loss_aware_replay_400x0p02_stop3200 | 1.0705024154010432 | 0.08753949512265458 | -0.023086094269275992 | -0.0034362926431102947 | 0.00028521320200525224 | -1.3703217375593546e-05 | -3.176385528769199e-07 | 9 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-MSEAdapter | cotangent_norm_normalized_lr0p1 | 1.2150067146183634 | 0.36306811090145297 | 0.006228595183454311 | -0.008139233775547927 | 0.00277020771682146 | 1.5465901626043888e-06 | -1.7955400397617893e-06 | 10 | 1 | 0 | 0 |  |
| Delta-MSEAdapter | terminal_preservation_800x0p015_stop4800_lr0p05 | 1.276158354138654 | 0.6428308676410357 | 0.07124155077049932 | 0.0001439540574669218 | 0.006947546611627331 | 0.0001723752093312214 | 3.472814569249749e-08 | 10 | 1 | 0 | 0 |  |
| Delta-MSEAdapter | terminal_preservation_400x0p025_stop6400_lr0p03 | 1.3208221712564372 | 0.9116227550939155 | 0.20128015451153736 | 0.0617719768718169 | 0.012006887853203807 | 0.0010522138545638882 | 0.00015519035332545172 | 10 | 1 | 0 | 0 |  |
| Delta-MSEAdapter | terminal_preservation_200x0p018_stop6400_lr0p02 | 1.3527034171296548 | 1.1634463078703514 | 0.3419553166015712 | 0.14038767726883006 | 0.016639006720652105 | 0.002456119581438543 | 0.0005384573487390298 | 10 | 1 | 0 | 0 |  |
| Delta-MSEAdapter | terminal_preservation_100x0p010_stop6400_lr0p01 | 1.393254084799068 | 1.2586808009474928 | 0.6952386831150943 | 0.4177751083569292 | 0.018288127554114908 | 0.007750606212539424 | 0.003367133163010294 | 10 | 1 | 1 | 0 |  |
| Delta-MSEAdapter | source_loss_pid20_lr0p0005_clip0p03 | 1.4240315146336402 | 1.4220943162049868 | 1.36857182165343 | 1.3440459533706364 | 0.024253861571196467 | 0.021983585043926723 | 0.02092853040085174 | 10 | 1 | 1 | 0 |  |
| Delta-RankingAdapter | operator_initial_commit_adamw | -14.296990110395118 | -34.21102382829784 | -46.85328314227103 | -51.33861502353322 | 0.00030102382879704237 | 1.2306336429901421e-05 | 3.7201298255240545e-06 | 2 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-RankingAdapter | adapter_balanced_replay_400x0p04_stop1600 | -8.722551137847088 | -28.966236336016948 | -42.402206595677946 | -46.852095721936145 | 0.0013585023116320372 | 4.571178578771651e-05 | 1.4253826520871371e-05 | 2 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-RankingAdapter | source_loss_aware_replay_400x0p02_stop3200 | -5.626111527842535 | -19.418746597607356 | -35.60393224803174 | -40.12475070341657 | 0.01628025807440281 | 0.00028330908389762044 | 8.479187090415508e-05 | 2 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-RankingAdapter | cotangent_norm_normalized_lr0p1 | -3.806306604246463 | -12.209000733612728 | -27.468900113437424 | -32.89723087084774 | 0.07098928838968277 | 0.0018747358117252588 | 0.00047373061534017324 | 2 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-RankingAdapter | terminal_preservation_800x0p015_stop4800_lr0p05 | -1.8927646848180144 | -7.7189631392847025 | -20.681370127363504 | -26.69032211361246 | 0.10556343197822571 | 0.012690888717770576 | 0.003002010751515627 | 4 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-RankingAdapter | terminal_preservation_400x0p025_stop6400_lr0p03 | -0.7647923876156346 | -6.048762197945975 | -16.250433717099224 | -22.528616923129018 | 0.10988792777061462 | 0.042905982583761215 | 0.011696781031787395 | 5 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-RankingAdapter | terminal_preservation_200x0p018_stop6400_lr0p02 | -0.05774128866840922 | -5.012988003910624 | -12.778457497599307 | -18.52401252119478 | 0.11138468980789185 | 0.08280553668737411 | 0.03299504891037941 | 5 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-RankingAdapter | terminal_preservation_100x0p010_stop6400_lr0p01 | 0.7619114921078368 | -3.2084903925782475 | -8.234151687541024 | -11.815852590054499 | 0.11874598264694214 | 0.12184613943099976 | 0.10316630452871323 | 6 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-RankingAdapter | source_loss_pid20_lr0p0005_clip0p03 | 1.5349862884317256 | 1.3109692659935823 | 0.5758680369312964 | 0.222869803228533 | 0.10966694355010986 | 0.005508840084075928 | -0.02250492572784424 | 10 | 1 | 0 | 0 |  |
| Delta-GenericSourceTarget | operator_initial_commit_adamw | -84.18228991388084 | -550.2600458032274 | -1938.8465265205346 | -2808.1586666813687 | -429289.6208496094 | -1771682.251953125 | -2666956.84765625 | 4 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-GenericSourceTarget | adapter_balanced_replay_400x0p04_stop1600 | -40.96607456626155 | -321.7554491324362 | -1008.1230553833276 | -1454.9262670521412 | -217223.0167236328 | -910209.9565429688 | -1372270.0380859375 | 3 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-GenericSourceTarget | source_loss_aware_replay_400x0p02_stop3200 | -15.392515360778741 | -147.1842119957778 | -456.98846935763584 | -619.8562118412699 | -83964.96203613281 | -366862.38623046875 | -555487.9057617188 | 3 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-GenericSourceTarget | cotangent_norm_normalized_lr0p1 | -7.048228129632719 | -69.82526652270796 | -253.95505876675134 | -342.2141333710812 | -38283.986587524414 | -178095.42822265625 | -271417.22521972656 | 3 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-GenericSourceTarget | terminal_preservation_800x0p015_stop4800_lr0p05 | -3.1388864238914413 | -33.334630207004444 | -145.26936693662938 | -207.0483885248992 | -16116.429000854492 | -85666.80630493164 | -132307.79150390625 | 3 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-GenericSourceTarget | terminal_preservation_400x0p025_stop6400_lr0p03 | -1.4041504837964662 | -19.94863420040781 | -97.77322568425423 | -151.91977860417416 | -7798.72087097168 | -48735.55389404297 | -76495.34948730469 | 3 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-GenericSourceTarget | terminal_preservation_200x0p018_stop6400_lr0p02 | -0.4727851201230938 | -13.1150790771764 | -69.32255688130215 | -113.33023143839492 | -4146.249900817871 | -30452.22817993164 | -48974.93606567383 | 3 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-GenericSourceTarget | terminal_preservation_100x0p010_stop6400_lr0p01 | 0.45415737822388647 | -6.155109329608189 | -34.76139337617176 | -58.881487355936066 | -1237.9830627441406 | -12391.05989074707 | -21448.0789642334 | 2 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-GenericSourceTarget | source_loss_pid20_lr0p0005_clip0p03 | 1.3184600492618017 | 1.0836176476295707 | 0.46942621050484135 | 0.16758560338153905 | -1.23134183883667 | -50.37758731842041 | -115.47407293319702 | 10 | 0 | 0 | 1 | source_func_or_source_loss_or_control_gate_failed |
| Delta-StableRandom | operator_initial_commit_adamw | -68.59297797051057 | -518.5744427058082 | -1772.0584853149207 | -2551.9034626665443 | -6865790.0107421875 | -27926722.62109375 | -42047914.8984375 | 3 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-StableRandom | adapter_balanced_replay_400x0p04_stop1600 | -33.70187397954443 | -281.88661866992015 | -927.5203786094241 | -1318.0489670126626 | -5689049.2578125 | -22985670.841796875 | -34180978.6484375 | 4 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-StableRandom | source_loss_aware_replay_400x0p02_stop3200 | -12.77860046598263 | -118.71922478141451 | -406.52826719064876 | -565.688184513469 | -2465498.649291992 | -9906357.001953125 | -14923630.559570312 | 4 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-StableRandom | cotangent_norm_normalized_lr0p1 | -5.769182664333076 | -57.031263454654756 | -213.60659387475008 | -298.27479555479107 | -1209498.5539550781 | -5004918.291015625 | -7536774.876953125 | 4 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-StableRandom | terminal_preservation_800x0p015_stop4800_lr0p05 | -2.176169795034862 | -27.509355333558233 | -115.40368045036317 | -166.63586914483162 | -588948.5827941895 | -2525021.16796875 | -3822379.797607422 | 4 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-StableRandom | terminal_preservation_400x0p025_stop6400_lr0p03 | -0.6441630377623779 | -16.5436618754702 | -78.27374131703993 | -119.31141447278604 | -342133.05628967285 | -1512000.157836914 | -2295525.8610839844 | 3 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-StableRandom | terminal_preservation_200x0p018_stop6400_lr0p02 | 0.17287026919074666 | -10.895992347059915 | -56.065416977879956 | -90.1387657925852 | -216593.60066223145 | -992842.7914428711 | -1513868.578125 | 3 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-StableRandom | terminal_preservation_100x0p010_stop6400_lr0p01 | 1.068825602628762 | -4.844937733890874 | -28.353251130760583 | -47.66115281107301 | -93504.35606384277 | -484130.90618896484 | -745499.7790527344 | 4 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-StableRandom | source_loss_pid20_lr0p0005_clip0p03 | 1.8250564242459404 | 1.598803180299024 | 0.48741192214893236 | -0.21449990676281794 | -229.75036692619324 | -6359.873881340027 | -15183.271532058716 | 10 | 0 | 0 | 1 | source_func_or_source_loss_or_control_gate_failed |

Evidence chain: S5 按 adapter 计数 C3/C4；Delta-StableRandom 只作为 control 读回，不计入非随机 robustness。若 source_func 过但 source_loss 负，route 不允许提升为 arbitrary-loss success。

## Repair Diagnostics

| diagnostic_id | operator_id | norm_scale | attempt | loss_adapter_name | S2_operator_atom_pass | S4_operator_metric_commit_pass_rows | source_func_h800 | source_func_h1600 | source_func_h3200 | source_loss_h800 | source_loss_h1600 | source_loss_h3200 | row_positive_count_h3200 | C3_source_formation_pass | C4_terminal_retention_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O10_scale0p16_nonrandom_source_loss_balanced_pid20 | O10_NonRandomSourceLossBalancedOperator | 0.16 | source_loss_pid20_lr0p0005_clip0p03 | Delta-LossCEAdapter | 1 | 3 | 1.723307092890777 | 1.4121471176515432 | 0.64600591142448 | 0.04386091232299805 | -0.025439143180847168 | -0.24998295307159424 | 10 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| O10_scale0p16_nonrandom_source_loss_balanced_pid20 | O10_NonRandomSourceLossBalancedOperator | 0.16 | source_loss_pid20_lr0p0005_clip0p03 | Delta-MSEAdapter | 1 | 3 | 1.4220945973155117 | 1.3989894056728518 | 1.368571988396306 | 0.024253870884422213 | 0.02337199338944629 | 0.021983589889714494 | 10 | 1 | 1 |  |
| O10_scale0p16_nonrandom_source_loss_balanced_pid20 | O10_NonRandomSourceLossBalancedOperator | 0.16 | source_loss_pid20_lr0p0005_clip0p03 | Delta-RankingAdapter | 1 | 3 | 1.3109681637700552 | 1.0486473483737748 | 0.5758654428160732 | 0.10966712236404419 | 0.07068198919296265 | 0.005508780479431152 | 10 | 1 | 0 |  |
| O4_scale0p08_formal_rowmean_pid_boundary | O4_LowNDSOperator | 0.08 | pid10_kp0p035_ki0p003_kd0p012_clip0p05_lr0p001 | Delta-MSEAdapter | 1 | 3 | 1.2719592835834557 | 1.2497362642526089 | 1.2364594019508683 | 0.0053288865892682225 | 0.004950767892296426 | 0.004677283097407781 | 10 | 1 | 1 |  |
| O4_scale0p08_formal_rowmean_pid_boundary | O4_LowNDSOperator | 0.08 | pid10_kp0p035_ki0p003_kd0p012_clip0p05_lr0p001 | Delta-RankingAdapter | 1 | 3 | 1.0623325530242795 | 0.47699153636099256 | -0.3908181678994358 | 0.003959953784942627 | -0.046400606632232666 | -0.04489398002624512 | 6 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| O4_scale0p16_loss_positive_retention_shortfall | O4_LowNDSOperator | 0.16 | pid5_kp0p040_ki0p003_kd0p012_clip0p060_lr0p001 | Delta-MSEAdapter | 1 | 3 | 1.3189438996387919 | 1.2994792242632507 | 1.2815407988837673 | 0.022819325735326856 | 0.021496724686585367 | 0.019921564642572775 | 10 | 1 | 1 |  |
| O4_scale0p16_loss_positive_retention_shortfall | O4_LowNDSOperator | 0.16 | pid5_kp0p040_ki0p003_kd0p012_clip0p060_lr0p001 | Delta-RankingAdapter | 1 | 3 | 1.132627606486891 | 0.6070539535214879 | -0.1971157667423188 | -0.03256118297576904 | -0.04702800512313843 | -0.05130094289779663 | 7 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| O1_scale0p16_retention_positive_loss_shortfall | O1_UpstreamCotangentDrift | 0.16 | pid10_kp0p035_ki0p003_kd0p012_clip0p05_lr0p001 | Delta-RankingAdapter | 0 | 3 | 1.2880031454052667 | 0.8829975951071706 | 0.22608493418023667 | -0.01731741428375244 | -0.1204524040222168 | -0.2618222236633301 | 10 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |

Insight: repair diagnostics 是 gate 外的缩短 horizon 边界复算，用来审计已尝试修复方向。O10 non-random/source-loss-balanced consensus 用 random/stable 作为 controls、把 SourceTarget/CE/MSE/Ranking 的 raw consensus 与 CE/MSE/Ranking smooth consensus 混合；若同一 official seed 下第二个 adapter 打开，会在 S5 official matrix 中再确认。O4 scale/PID 与 O1 high-gain rows 保留为边界/负例，用于说明单纯放大或只追 retention 会触发 source_loss blocker。

## Part F KAN Mapping

| carrier | attempt | mapping_strategy | loss_adapter_name | KAN_source_func_h3200 | KAN_source_loss_h3200 | KAN_source_loss_h4800 | KAN_specific_delta_vs_MLP_same_metric | carrier_specific_efficiency_pass | wrong_global_efficiency_gate_detected | KAN_source_channel_decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | K1_readout_replay_100x0p08 | K1_readout_source_channel | Delta-MSEAdapter | 0.5010721981525421 | 0.0002690380497369915 | 6.451907393056899e-05 | -0.8674996235008878 | 0 | 0 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-CHE | K1_readout_replay_400x0p025_repair | K1_readout_source_channel | Delta-MSEAdapter | 0.6099423468112946 | -0.0006028676871210337 | 0.00015938072465360165 | -0.7586294748421354 | 0 | 0 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-CHE | K2_basis_estimate_readout_commit_800x0p015 | K2_basis_estimate_readout_commit | Delta-MSEAdapter | 0.19240379333496094 | -0.0017103266436606646 | -0.0006440427387133241 | -1.176168028318469 | 0 | 0 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-CHE | K7_optimizer_state_integrated_initial_only | K7_optimizer_state_integrated_source_replay | Delta-MSEAdapter | 0.6377114057540894 | -0.0016370617922802921 | -0.00045772704548820847 | -0.7308604158993406 | 0 | 0 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-CHE | K8_readout_replay_50x0p12_repair | K8_readout_replay_scale_interval_repair | Delta-MSEAdapter | 0.3208370804786682 | -0.022624326404184103 | -0.022664300980977714 | -1.0477347411747617 | 0 | 0 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-CHE | K8_readout_replay_50x0p20_repair | K8_readout_replay_scale_interval_repair | Delta-MSEAdapter | 0.39612460136413574 | 0.006017860025167465 | 0.010552571155130863 | -0.9724472202892942 | 0 | 0 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-CHE | K8_readout_replay_25x0p12_repair | K8_readout_replay_scale_interval_repair | Delta-MSEAdapter | 1.1868340224027634 | 0.00039954425301402807 | -0.08108080388046801 | -0.18173779925066658 | 0 | 0 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-CHE | K8_readout_replay_100x0p16_repair | K8_readout_replay_scale_interval_repair | Delta-MSEAdapter | 0.11484074592590332 | 0.00010492093861103058 | 0.00033899396657943726 | -1.2537310757275266 | 0 | 0 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-CHE | K8_readout_replay_200x0p08_repair | K8_readout_replay_scale_interval_repair | Delta-MSEAdapter | 0.9124283790588379 | 0.0013543227687478065 | 0.001178257167339325 | -0.45614344259459205 | 0 | 0 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-CHE | K10_readout_gain1p0_25x0p12_repair | K1_readout_source_channel | Delta-MSEAdapter | 1.108336627483368 | 0.0020577211398631334 | 0.00023859716020524502 | -0.260235194170062 | 0 | 0 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-CHE | K10_readout_gain1p5_25x0p12_repair | K1_readout_source_channel | Delta-MSEAdapter | 0.26638543605804443 | 0.015255101025104523 | 0.0053477901965379715 | -1.1021863855953855 | 0 | 0 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-CHE | K10_readout_gain1p5_200x0p08_repair | K1_readout_source_channel | Delta-MSEAdapter | 0.6047378480434418 | 0.00039201416075229645 | 0.00030534242978319526 | -0.7638339736099882 | 0 | 0 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-CHE | K10_readout_gain2p0_100x0p16_repair | K1_readout_source_channel | Delta-MSEAdapter | 0.5841592252254486 | -0.011305354535579681 | -0.005314784124493599 | -0.7844125964279813 | 0 | 0 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-CHE | K11_basis_linearized_rank12_cap1p0_25x0p12 | K11_basis_linearized_commit | Delta-MSEAdapter | 0.2740531265735626 | -0.0013429432292468846 | -0.0020735264697577804 | -1.0945186950798673 | 0 | 0 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-CHE | K11_basis_linearized_rank12_cap2p0_50x0p08 | K11_basis_linearized_commit | Delta-MSEAdapter | 0.26413320004940033 | -0.0001827927044359967 | -0.00023006050469120964 | -1.1044386216040296 | 0 | 0 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-CHE | K11_basis_linearized_rank12_cap4p0_initial_only | K11_basis_linearized_commit | Delta-MSEAdapter | 0.5665958322824736 | -6.309381836433481e-11 | -3.892423482502743e-06 | -0.8019759893709564 | 0 | 0 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-FOU | K1_readout_replay_100x0p08 | K1_readout_source_channel | Delta-MSEAdapter | 0.31330588832497597 | 3.122026100754738e-06 | 1.7634847608860582e-06 | -1.055265933328454 | 1 | 0 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-FOU | K1_readout_replay_400x0p025_repair | K1_readout_source_channel | Delta-MSEAdapter | 0.2075478360056877 | -2.1308624127414078e-05 | -2.1570675016846508e-05 | -1.1610239856477422 | 1 | 0 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-FOU | K2_basis_estimate_readout_commit_800x0p015 | K2_basis_estimate_readout_commit | Delta-MSEAdapter | 0.22444985434412956 | -2.4542332539567724e-05 | -2.7981972380075604e-05 | -1.1441219673093004 | 1 | 0 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-FOU | K7_optimizer_state_integrated_initial_only | K7_optimizer_state_integrated_source_replay | Delta-MSEAdapter | 0.20226521603763103 | -6.752587554596833e-06 | -7.967050968710372e-09 | -1.166306605615799 | 1 | 0 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-FOU | K8_readout_replay_50x0p12_repair | K8_readout_replay_scale_interval_repair | Delta-MSEAdapter | 0.26769401878118515 | -2.1940533770248294e-05 | 3.743999695871025e-05 | -1.1008778028722448 | 1 | 0 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-FOU | K8_readout_replay_50x0p20_repair | K8_readout_replay_scale_interval_repair | Delta-MSEAdapter | 0.26819541305303574 | 3.133117570541799e-05 | 3.170844865962863e-05 | -1.1003764086003942 | 1 | 0 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-FOU | K8_readout_replay_25x0p12_repair | K8_readout_replay_scale_interval_repair | Delta-MSEAdapter | 0.4999959245324135 | 2.1201747586019337e-05 | 6.254718755371869e-06 | -0.8685758971210165 | 1 | 0 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-FOU | K8_readout_replay_100x0p16_repair | K8_readout_replay_scale_interval_repair | Delta-MSEAdapter | 0.24123278260231018 | 5.271635018289089e-06 | 9.918876457959414e-05 | -1.1273390390511198 | 1 | 0 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-FOU | K8_readout_replay_200x0p08_repair | K8_readout_replay_scale_interval_repair | Delta-MSEAdapter | 0.2155883014202118 | -0.00010298762936145067 | -5.440917448140681e-05 | -1.1529835202332182 | 1 | 0 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-FOU | K10_readout_gain1p0_25x0p12_repair | K1_readout_source_channel | Delta-MSEAdapter | 0.489858727902174 | -2.481439514667727e-05 | 0.0002625755369081162 | -0.878713093751256 | 1 | 0 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-FOU | K10_readout_gain1p5_25x0p12_repair | K1_readout_source_channel | Delta-MSEAdapter | 0.44210848212242126 | -0.0003001656150445342 | -0.0008504395082127303 | -0.9264633395310087 | 1 | 0 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-FOU | K10_readout_gain1p5_200x0p08_repair | K1_readout_source_channel | Delta-MSEAdapter | 0.26853301376104355 | 8.608253847341985e-05 | 6.898229185026139e-05 | -1.1000388078923864 | 1 | 0 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-FOU | K10_readout_gain2p0_100x0p16_repair | K1_readout_source_channel | Delta-MSEAdapter | 0.042680129408836365 | -0.0001467724796384573 | -0.00011305557563900948 | -1.3258916922445936 | 1 | 0 | KANTargetRetentionOnly | KANTargetRetentionOnly |
| D-FOU | K11_basis_linearized_rank12_cap1p0_25x0p12 | K11_basis_linearized_commit | Delta-MSEAdapter | 0.23721939325332642 | -8.110720955301076e-06 | -1.3249060430098325e-05 | -1.1313524284001035 | 1 | 0 | KANTargetRetentionOnly | KANTargetRetentionOnly |

_仅显示前 30 / 43 rows；完整 CSV 见 artifact。_

Insight: S6 每一行使用自己的 carrier_specific_efficiency_pass。D-CHE 不会借 D-FOU pass 放行；readout-only source 会保留为 basis-channel still open，不写成 KAN-general success。本轮新增 K8 readout replay scale/interval repair scan、K10 readout update-gain repair rows、K11 finite-difference basis-linearized commit、K12 D-FOU low-frequency init-variant rows、K13 fan-scale source_loss boundary rows 与 K14 D-CHE corrected-readout-layout low-degree rows。K13 已打开 D-FOU source-channel；K14 显示 D-CHE source 可被构造，但 carrier_specific_efficiency_pass=0，因此 route 写成 source-exists / efficiency-blocked，而不是 KAN-general success。K9 all/basis/readout target-gradient commit 作为执行日志中的边界诊断保留，未进入 official pass。

## 4GPU Queue

| task_id | line | gpu | command | status | returncode | start_time | end_time | duration_sec | log_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S2_S4_operator_fu_construct_solve_commit | S2-S4 | 1 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_operator_fu.py --seed 2212 --norm-scale 0.16 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 | completed | 0 | 2026-06-08 03:30:19 +0800 | 2026-06-08 03:30:21 +0800 | 2.001427173614502 | /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/logs/S2_S4_operator_fu_construct_solve_commit.log |
| S0_19_code_semantic_truth_gate | S0.19 | 0 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_s019_truth_gate.py --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 | completed | 0 | 2026-06-08 03:30:19 +0800 | 2026-06-08 03:30:25 +0800 | 6.004892826080322 | /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/logs/S0_19_code_semantic_truth_gate.log |
| S1_arbitrary_cotangent_operator_efficiency | S1 | 0 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_basis_efficiency.py --device cuda:0 --batch-sizes 128,256,512,1024 --hidden 64 --repeats 3 --warmup 1 --seed 2212 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 | completed | 0 | 2026-06-08 03:30:25 +0800 | 2026-06-08 03:30:30 +0800 | 5.000988483428955 | /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/logs/S1_arbitrary_cotangent_operator_efficiency.log |
| S5_arbitrary_loss_operator_horizon | S5 | 1 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_arbitrary_loss_horizon.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212 | completed | 0 | 2026-06-08 03:30:21 +0800 | 2026-06-08 03:40:07 +0800 | 586.0378081798553 | /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/logs/S5_arbitrary_loss_operator_horizon.log |
| S6_kan_mapping_carrier_specific | S6 | 2 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 --seed 2212 | completed | 0 | 2026-06-08 03:40:07 +0800 | 2026-06-08 03:42:39 +0800 | 152.01137256622314 | /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/logs/S6_kan_mapping_carrier_specific.log |
| S7_finalize_packet_recap | S7 | 3 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_12_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12 | completed | 0 | 2026-06-08 03:42:39 +0800 | 2026-06-08 03:42:40 +0800 | 1.0021371841430664 | /home/chengshun.wang/DG-LCA/results/v22_12_arbitrary_loss_operator_fu_basis_efficiency_4gpu/official_v22_12/logs/S7_finalize_packet_recap.log |

| execution_contract_violation | reason | max_idle_gap_sec |
| --- | --- | --- |
| 0 | no_dependency_free_runnable_task_waited_on_idle_gpu_over_10min | 0 |

## Failure Taxonomy / Next Repair

| blocker | repair_or_next_direction |
| --- | --- |
|  | if S0 fails, repair code/semantic packet first and rerun S0 only |
|  | if efficiency blocked or fused rows remain zero, implement kernel-native arbitrary-cotangent forward/VJP/update path |
|  | if MSE-only or source_loss negative, continue adapter-balanced/source-loss-aware operator objective before replay tuning |
| D-CHE_arbitrary_cotangent_efficiency_blocked_after_source_exists;D-CHE_source_loss_h4800_gate_after_source_exists | if D-CHE source exists but carrier efficiency is blocked, continue D-CHE kernel-native arbitrary-cotangent officialization; if source is still missing, continue low-degree/basis-estimate commit |

## 修改记录

- 新增 `dgkan/profiling/efficiency_v22_12.py`：实现 arbitrary-cotangent repair variant profiler，记录 manual VJP、component telemetry、carrier-specific pass 与 fused-complete blocker。
- 新增 `experiments/run_v22_12_s019_truth_gate.py`：检查 v22.12 source closure、semantic firewall、carrier-specific gate 与 promotion split。
- 新增并修复 `experiments/run_v22_12_operator_fu.py`：把固定 source displacement 升级为 LossInterface operator atoms；首轮 operator atom 被 NDS/adapter gate 阻塞后，新增 class 维 LowNDS 平滑；随后修复 operator atom/combo 归一化，改为保留 arbitrary cotangent row-mean 方向，避免 ranking/source-target cotangent 被 CE/logit-gauge 去均值抹掉；本轮新增 `O10_NonRandomSourceLossBalancedOperator`，把 random/stable cotangent 留作 controls，并优先尝试 non-random raw + loss-smooth consensus，final queue norm_scale 调为 0.16 以修复 ranking source_loss blocker。
- 修复 `experiments/run_v22_12_arbitrary_loss_horizon.py`：按 CE/MSE/ranking/source-target/stable-random adapter 运行 horizon，并按不同非随机 adapter 计数 C3/C4；本轮新增 `source_loss_pid20_lr0p0005_clip0p03` 低 lr/PID source-loss-aware attempt。
- 修复 `experiments/run_v22_12_repair_diagnostics.py`：将 row-mean/PID/norm-scale/O1-high-gain/O10-nonrandom-source-loss-balanced 边界修复尝试落盘为 diagnostic CSV；这些行不单独改 official route，只用于审计 blocker。
- 修复 `experiments/run_v22_12_kan_mapping.py`：修复 v22.11 的全局 OR gate，所有 KAN row 使用 carrier-specific efficiency pass；本轮新增 K8 readout replay scale/interval repair scan、K10 readout update-gain repair rows、K11 finite-difference basis-linearized commit rows、K12 D-FOU low-frequency init-variant rows、K13 fan-scale source_loss boundary rows 与 K14 D-CHE corrected-readout-layout rows。K11 直接在 `w1/w2` basis/readout tensor 上解 train-stream target_delta 的低秩线性化 update，并落盘 operator_rank/basis_channel_energy/projection_residual 等诊断；K12/K13 只在 carrier-specific efficient 的 D-FOU 上测试 strict PrimitiveKAN `signed_pair_linear`/`signed_pair_random_linear`/`fan_scale_repair` 初始化与 replay scale/interval，不读取 adapter 公式；K14 修复 D-CHE frozen-readout feature `(hidden,basis)` 到 `w2(hidden,class,basis)` 的 layout 映射，仅作为新 row 落盘，不改写历史 K1-K13；若 source pass 但 D-CHE efficiency pass=0，则写为 `KANEfficiencyContractBlocked`。
- 新增 `experiments/run_v22_12_full.py` 与 finalizer：生成 4GPU queue artifacts、results bundle、code review packet、执行日志和复盘日志。

## 补充分析 / Evidence Chain

- Code truth 与 S6 runtime row 双重记录 `wrong_global_efficiency_gate_detected=0` 和 `carrier_specific_efficiency_pass_consistency=1`，这是 v22.12 的 A-CodeTruth 最小进展。
- S1 repair rows 显示具体尝试过 CHE/RAT/RBF/FOU variants；若 ratio pass 仍不能越过 fused blocker，结论必须限定为 manual-path exploration。
- Operator source 证据链从 cotangent suite -> operator atom -> variational combo -> metric commit -> horizon adapter robustness 逐层落盘；O10 若打开第二个 adapter，仍需同一 official payload 在 S5 horizon matrix 中过 C3/C4 adapter-count gate，finalizer 不读取 diagnostic stdout 来放行。
- KAN mapping 只证明 readout/basis-estimate/linearized-basis runner 中实际读回的 source channel；若 K11/K14 低秩或 corrected-layout commit 仍低于 MLP same-metric source 或触发 source_loss/efficiency gate，会作为 KANSourceChannelMismatch/TargetRetentionOnly/KANEfficiencyContractBlocked 写入 artifact。D-RAT/D-RBF KAN primitive 若未实现，会作为 deferred/blocker 写入 artifact。
