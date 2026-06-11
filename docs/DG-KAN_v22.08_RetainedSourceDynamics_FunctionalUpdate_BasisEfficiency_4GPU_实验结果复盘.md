# DG-KAN v22.08 RetainedSourceDynamics FunctionalUpdate BasisEfficiency 4GPU 实验结果复盘

生成时间：2026-06-07 04:02:18 +0800

## Route

- route: `R-PostNoGoTrainFlowCommutatorRepeatVerifyBlocked`
- promotion_allowed: 0
- blocking_metric: `D-RAT_D-RBF_multibatch_robust_or_telemetry;retained_source_observer_gate;C2_true_block_solver_or_observer_gate;C3_source_formation_gate;terminal_preservation_not_entered_C3_source_gate;KAN_mapping_not_entered_C3_source_gate;post_nogo_C-O6_source_state_certificate_gate;post_nogo_C-O7_counterfactual_washout_gate;post_nogo_C-O8_ntk_jacobian_channel_gate;post_nogo_C-O9_aug_tangent_gate;post_nogo_C-O10_bootstrap_basin_gate;post_nogo_C-O11_crossfit_influence_gate;post_nogo_C-O12_train_flow_commutator_gate;post_nogo_C-O12_repeat_verify_official_gate`
- S0.15 pass: 1
- D-CHE/D-FOU S1 pass: 1 / 1
- D-RAT/D-RBF multibatch closed: 0
- C0 retained-source observer pass rows: 0
- official early-chain positive rows: 0
- retained h800+h3200 positive rows: 13
- C2 true block solver pass rows: 0
- C3 source formation pass rows: 0
- post-no-go C-O6 route: `PostNoGoSourceStateCertificateBlocked`
- post-no-go C-O6 train-only/fresh/official pass rows: 144/0/0
- post-no-go C-O7 route: `PostNoGoCounterfactualWashoutBlocked`
- post-no-go C-O7 train-only/fresh/official pass rows: 65/0/0
- post-no-go C-O8 route: `PostNoGoNTKChannelBlocked`
- post-no-go C-O8 train-only/fresh/official pass rows: 0/0/0
- post-no-go C-O9 route: `PostNoGoAugTangentBlocked`
- post-no-go C-O9 train-only/fresh/official pass rows: 13/0/0
- post-no-go C-O10 route: `PostNoGoBootstrapBasinBlocked`
- post-no-go C-O10 train-only/fresh/official pass rows: 144/0/0
- post-no-go C-O11 route: `PostNoGoCrossFitInfluenceBlocked`
- post-no-go C-O11 train-only/fresh/official pass rows: 144/0/0
- post-no-go C-O12 route: `PostNoGoTrainFlowCommutatorFreshC3Opened`
- post-no-go C-O12 train-only/fresh/official pass rows: 48/1/0
- post-no-go C-O12 verify route: `TrainFlowCommutatorRepeatVerifyBlocked`
- post-no-go C-O12 verify robust-fresh/official pass rows: 0/0
- terminal_preservation_decision: `D0D1BlockedBeforeTerminalPreservation`
- KAN_mapping_decision: `KANMappingNotEntered`
- execution_contract_violation: 0
- next_codex_action: C-O12 fresh smoke failed repeat verification; continue only with a genuinely new source-observability principle outside C-O1..C-O12, or record the local boundary

## Part A S0.15 Code / Metric / Solver Gate

| check | pass | metric | value | blocker |
| --- | --- | --- | --- | --- |
| required_source_files | 1 | exists | 39/39 |  |
| compileall | 1 | py_compile | 0 |  |
| import_closure | 1 | self_contained_import | 0 |  |
| linec_fast_golden | 1 | tests | 9 |  |
| linec_channel_golden | 1 | tests | 8 |  |
| source_chain_tests | 1 | tests | 7/7 |  |
| terminal_retention_tests | 1 | tests | 14/14 |  |
| metric_solver_tests | 1 | tests | 33/33 |  |
| c2_recompute_tests | 1 | scale+block recompute | 4/4 |  |
| mechanism_contracts | 1 | v22.08 semantic+forbidden | 12/12;alias_undeclared=0 |  |
| kernel_gradcheck | 1 | kernels | 4 |  |
| kernel_status_consistency | 1 | official_vs_gradcheck | 1 |  |
| profiler_phase_tests | 1 | tests | 2/2 |  |


### C2 Repair Recompute Correctness

- audit note: `projection_residual_Gf` is a ratio and can remain scale-invariant for an exact linear readout solve; v22.08 therefore also records target_norm/update_norm scaling and block-restricted masks. This is written as evidence, not as promotion.
- recompute summary: rows=4; pass_rows=4; projection_ratio_identical_rows=0
| case | target_scale | block_role | projection_residual_Gf | ActuationR2 | B2_transfer_gain | parameter_update_norm | target_norm_L2 | block_restricted_solver | C2_recompute_test_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_x1 | 1.0 | all | 0.5807860176284673 | 0.6582547426223755 | 0.14279025793075562 | 2.58319354057312 | 1.6395914554595947 | 0 | 1 |  |
| all_x64 | 64.0 | all | 0.5676424269788402 | 0.673547625541687 | -1.5597349405288696 | 72.55255889892578 | 104.93385314941406 | 0 | 1 |  |
| hidden_only_x64 | 64.0 | hidden_only | 0.9974963446704895 | -0.008074641227722168 | 0.02849721908569336 | 57.91011428833008 | 104.93385314941406 | 1 | 1 |  |
| readout_only_x64 | 64.0 | readout_only | 1.0819951301050104 | -0.18609833717346191 | -2.3969502449035645 | 43.706905364990234 | 104.93385314941406 | 1 | 1 |  |


## Part B Basis Efficiency / Officialization

| carrier | variant_family | forward_ratio_vs_mlp | backward_ratio_vs_mlp | step_ratio_vs_mlp | memory_ratio_vs_mlp | functional_direction_ms | metric_solver_ms | LineC_audit_ms | same_kernel_functional_runner_proof | fallback_kernel_used | official_fused_kernel_complete | v22_08_S1_pass | v22_08_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | CHE21-R2/R4 | 0.9305108277155166 |  | 0.46682782835473463 | 0.9913236539978382 |  |  | 0.4715628068273266 | 1 | 0 | 1 | 1 |  |
| D-FOU | FOU21-R3 | 1.0087315984225318 |  | 0.5089279836996342 | 0.9888469358117921 |  |  | 0.4722052059757213 | 1 | 0 | 1 | 1 |  |


### D-RAT / D-RBF Multibatch

| carrier | profile_rows | robust_production_pass_rows | near_E1_rows | component_telemetry_complete_rows | best_forward_ratio | best_step_ratio | best_memory_ratio | decision | blocker | repair_attempt_rows | repair_micro_near_E1_rows | repair_best_variant | repair_attempt_status | repair_blocker | official_closure_claimed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 4 | 0 | 4 | 0 | 1.6740033989141296 | 0.5052953483076714 | 0.8889482858803022 | NearE1LimitedSmokeAllowed | forward_ratio;component_telemetry_incomplete | 28 | 19 | RAT22.03-R5-numden-fused-backward | MicroNearE1RunnerBlocked | official_fused_missing;functional_runner_kernel_mismatch | 0 |
| D-RBF | 4 | 0 | 4 | 0 | 1.7652721315974544 | 0.6421926655220669 | 0.8889482858803022 | NearE1LimitedSmokeAllowed | forward_ratio;component_telemetry_incomplete | 28 | 4 | RBF22.03-R4-exp-approx-trainpath | MicroNearE1RunnerBlocked | official_fused_missing;functional_runner_kernel_mismatch | 0 |


### D-RAT / D-RBF Repair Attempt

| carrier | repair_attempt_rows | repair_micro_near_E1_rows | repair_nonreference_micro_near_E1_rows | repair_component_telemetry_complete_rows | repair_best_forward_ratio | repair_best_step_ratio | repair_best_memory_ratio | repair_best_variant | repair_best_batch_size | repair_attempt_status | repair_blocker | official_closure_claimed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 28 | 19 | 16 | 28 | 2.241468024262574 | 1.5434419200788174 | 1.0625 | RAT22.03-R5-numden-fused-backward | 512 | MicroNearE1RunnerBlocked | official_fused_missing;functional_runner_kernel_mismatch | 0 |
| D-RBF | 28 | 4 | 4 | 28 | 1.8989079161216158 | 1.4484736351713348 | 1.0338325500488281 | RBF22.03-R4-exp-approx-trainpath | 512 | MicroNearE1RunnerBlocked | official_fused_missing;functional_runner_kernel_mismatch | 0 |


## Part C Retained-Source Observer Theory

- best retained-label observer: `C-O5_control_nullspace` AUC_retained=0.7080144985904149 precision@20=0.0
- best official-early observer: `` AUC_official_early= precision@20=
- best top candidate by retained score: `MLP-V2206-C4-T7G0-EarlyWarmCarry-stop800-lr50` dataset=KMNIST h100=-0.3380619287490845 h400=0.05838441848754883 h800=0.10895073413848877 h3200=-0.0006132125854492188

| observer_family | rows | positive_official_early_chain_rows | positive_h3200_rows | positive_retained_h800_h3200_rows | AUC_predict_official_early_chain | AUC_predict_h3200_positive | AUC_predict_retained_h800_h3200_positive | precision_at_top20 | heldout_dataset_seed_precision_at_top20 | control_equivalent_fraction | C0_retained_source_observer_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-O1_drift_diffusion_retention | 204 | 0 | 45 | 13 |  | 0.6029350104821803 | 0.5533628674989931 | 0.0 | 0.0 | 0.0 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate;heldout_precision_gate |
| C-O2_micro_trajectory_invariance | 204 | 0 | 45 | 13 |  | 0.6146750524109015 | 0.6657269432138542 | 0.0 | 0.0 | 0.0 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate;heldout_precision_gate |
| C-O3_low_NDS_low_curvature | 204 | 0 | 45 | 13 |  | 0.5972047519217331 | 0.6568666935159082 | 0.0 | 0.0 | 0.0 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate;heldout_precision_gate |
| C-O4_info_volume_no_fold | 204 | 0 | 45 | 13 |  | 0.3772187281621244 | 0.33588401127668144 | 0.0 | 0.0 | 1.0 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate;control_equivalent_fraction_gate;heldout_precision_gate;random_sign_corrupt_gap_gate |
| C-O5_control_nullspace | 204 | 0 | 45 | 13 |  | 0.6941998602375961 | 0.7080144985904149 | 0.0 | 0.0 | 0.0 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate;heldout_precision_gate |


### Top Candidates

| job_order | v21_id | dataset | future_audit_source_h100 | future_audit_source_h400 | future_audit_source_h800 | future_audit_source_h1600 | future_audit_source_h3200 | official_early_chain_h100_h400_h800_positive | retained_h800_h3200_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 65 | MLP-V2206-C4-T7G0-EarlyWarmCarry-stop800-lr50 | KMNIST | -0.3380619287490845 | 0.05838441848754883 | 0.10895073413848877 | 0.1308000087738037 | -0.0006132125854492188 | 0 | 0 |
| 110 | MLP-V2206-C4-T10G0-EarlyWarmCarry-stop1200-lr50 | Fashion-MNIST | -1.0158993005752563 | -0.07163035869598389 | 0.02028369903564453 | -0.07717788219451904 | -0.11038780212402344 | 0 | 0 |
| 19 | MLP-V2206-S2-T7G0-HiddenBlockSolver | KMNIST | -0.5140684843063354 | -0.20567989349365234 | -0.29926395416259766 | -0.5583347082138062 | -0.7201063632965088 | 0 | 0 |
| 72 | MLP-V2206-C4-T7G0-EarlyWarmCarry-stop1200-lr50 | MNIST | -1.1719499826431274 | -0.7876436710357666 | -0.36689043045043945 | 0.1483926773071289 | 0.1322997808456421 | 0 | 0 |
| 168 | MLP-V2206-C4-T10G0-OptTransport-lr50 | MNIST | -1.1534048318862915 | -0.781975507736206 | -0.42433226108551025 | 0.03953385353088379 | 0.24899780750274658 | 0 | 0 |
| 129 | MLP-V2206-C4-T10G0-PeriodicCarry-alt200-lr50 | Fashion-MNIST | -0.9508649110794067 | -0.10181307792663574 | 0.04776310920715332 | -0.0018540620803833008 | -0.2137467861175537 | 0 | 0 |
| 150 | MLP-V2206-C4-T10G0-OptTransport-lr150 | MNIST | -1.0936940908432007 | -0.7010617256164551 | -0.35333728790283203 | 0.03934669494628906 | 0.09051167964935303 | 0 | 0 |
| 193 | MLP-V2206-C4-T10G0-EarlyWarm400ThenOptTransport-lr150 | KMNIST | -0.3280905485153198 | 0.10294008255004883 | 0.13653504848480225 | 0.004935026168823242 | -0.2212611436843872 | 0 | 0 |
| 202 | MLP-V2206-C4-T12G0-SourceStateCarry | Fashion-MNIST | -1.0627917051315308 | -0.14960205554962158 | 0.04750847816467285 | 0.09232807159423828 | -0.05565071105957031 | 0 | 0 |
| 200 | MLP-V2206-C4-T12G0-OptTransport-lr150 | MNIST | -1.153650164604187 | -0.7657277584075928 | -0.39424872398376465 | 0.053174734115600586 | 0.042084455490112305 | 0 | 0 |
| 35 | MLP-V2206-C4-T7G0-EarlyWarmCarry-stop1200-lr50 | Fashion-MNIST | -0.889654278755188 | 0.043212890625 | 0.1704336404800415 | 0.0709371566772461 | -0.009943842887878418 | 0 | 0 |
| 197 | MLP-V2206-C4-T10G0-EarlyWarm800ThenOptTransport-lr150 | Fashion-MNIST | -0.9307035207748413 | 0.023661375045776367 | 0.17415201663970947 | -0.009909510612487793 | -0.12717044353485107 | 0 | 0 |

_仅显示前 12 / 20 rows；完整 CSV 见 artifact。_


### Post-No-Go C-O6 Source-State Persistence

- trigger: v22.08 reached `R-ObserverLocalNoGoBoundary`; per plan Case C, this run stops C-O1..C-O5 scale/cap/floor variants and tries a new train-only pathwise source-state persistence certificate.
- route: `PostNoGoSourceStateCertificateBlocked` previous_observer_route=`RetainedSourceObserverLocalNoGo_v22.08` blocker=`official_observer_gate_or_fresh_C3_source_horizon_failed`
- probe rows / measured rows: 144/144
- train-only pass / fresh C3 pass / official C3 pass: 144/0/0

| certificate_family | rows | positive_official_early_chain_rows | positive_h3200_rows | positive_retained_h800_h3200_rows | AUC_predict_official_early_chain | AUC_predict_h3200_positive | AUC_predict_retained_h800_h3200_positive | precision_at_top20 | PSSC_train_only_certificate_pass_rows | PSSC_official_observer_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-O6_pathwise_source_state_persistence | 144 | 0 | 45 | 13 |  | 0.4664421997755331 | 0.3857897827363476 | 0.0 | 144 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate |


#### C-O6 Selected Candidates

| job_order | v21_id | mechanism | dataset | seed | PSSC_score | source_state_coherence | source_state_pair_cosine_mean | max_control_cosine | B2_transfer_gain | B3_safety_gain | future_audit_source_h800 | future_audit_source_h3200 | PSSC_train_only_certificate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 46 | MLP-V2206-S1-T0G0-ReadoutSolver | M259-V2206MetricSolverT0G0ReadoutFU | Fashion-MNIST | 0 | 2.1666929831633808 | 0.6171263456344604 | 0.17445981595665216 | 0.0426487997174263 | 0.00010171804024139419 | 9.743733971845359e-05 | -0.506301999092102 | -1.0011218786239624 | 1 |
| 76 | MLP-V2206-C4-T7G0-EarlyWarmCarry-stop1200-lr50 | M265-V2206MetricSolverT7G0HiddenBlockFU | KMNIST | 0 | 2.156710130033316 | 0.6276095509529114 | 0.1918583574394385 | 0.11856582015752792 | 6.504492193926126e-05 | 8.367408736376092e-05 | -0.03418850898742676 | -0.18448269367218018 | 1 |


#### C-O6 Fresh Source-State C3

| v21_id | dataset | seed | PSSC_score | source_vs_best_control_h100 | source_vs_best_control_h400 | source_vs_best_control_h800 | source_vs_best_control_h1600 | source_vs_best_control_h2400 | source_vs_best_control_h3200 | source_state_alignment_mean | debt_not_exploded | PSSC_fresh_C3_source_state_pass | official_C3_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-S1-T0G0-ReadoutSolver | Fashion-MNIST | 0 | 2.1666929831633808 | 0.03356146812438965 | -0.3270375728607178 | -0.9113070964813232 | -1.4689940214157104 | -1.7483670711517334 | -1.8880099058151245 | 0.5630784527835203 | 0 | 0 | 0 | source_horizon_or_debt_gate_failed |
| MLP-V2206-C4-T7G0-EarlyWarmCarry-stop1200-lr50 | KMNIST | 0 | 2.156710130033316 | -0.12969160079956055 | -0.4130885601043701 | -0.8057007789611816 | -1.3517229557037354 | -1.6678894758224487 | -1.8447577953338623 | 0.5582782698905794 | 1 | 0 | 0 | source_horizon_or_debt_gate_failed |


### Post-No-Go C-O7 Counterfactual Washout

- trigger: C-O6 source-state persistence failed fresh C3; this follow-up tests a distinct train-only principle: whether a candidate direction leaves persistent functional displacement after a short AdamW washout compared with sign-flip, corrupt-label and random matched controls.
- route: `PostNoGoCounterfactualWashoutBlocked` previous_post_nogo_route=`PostNoGoSourceStateCertificateBlocked` blocker=`official_observer_gate_or_fresh_C3_source_horizon_failed`
- probe rows / measured rows: 144/144
- train-only pass / fresh C3 pass / official C3 pass: 65/0/0

| certificate_family | rows | positive_official_early_chain_rows | positive_h3200_rows | positive_retained_h800_h3200_rows | AUC_predict_official_early_chain | AUC_predict_h3200_positive | AUC_predict_retained_h800_h3200_positive | precision_at_top20 | COW_train_only_certificate_pass_rows | COW_official_observer_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-O7_counterfactual_washout_persistence | 144 | 0 | 45 | 13 |  | 0.48282828282828283 | 0.5361127422196125 | 0.0 | 65 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate |


#### C-O7 Selected Candidates

| job_order | v21_id | mechanism | dataset | seed | COW_score | source_washout_retention_ratio | source_retention_control_gap | source_B2_control_gap_after_washout | source_B3_control_gap_after_washout | future_audit_source_h800 | future_audit_source_h3200 | COW_train_only_certificate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 131 | MLP-V2206-C4-T11G0-AdamWControlGate-stop800-lr50 | M269-V2206MetricSolverT11G0EarlyObservableFU | Fashion-MNIST | 0 | 3.7986927489437523 | 35.03042105649809 | 19.113946876758583 | 5.805492401123047e-05 | 0.00011742115020751953 | -0.18797147274017334 | -0.7266116142272949 | 1 |
| 134 | MLP-V2206-C4-T10G0-EarlyWarm800ThenOptTransport-lr150 | M268-V2206MetricSolverT10G0CompensatedHiddenBlockFU | MNIST | 0 | 2.9828251043252894 | 8.499298449327807 | 6.579352605879548 | 9.334087371826172e-05 | 8.761882781982422e-05 | -0.30213451385498047 | 0.08555686473846436 | 1 |


#### C-O7 Fresh Counterfactual-Washout C3

| v21_id | dataset | seed | COW_score | source_vs_best_control_h100 | source_vs_best_control_h400 | source_vs_best_control_h800 | source_vs_best_control_h1600 | source_vs_best_control_h2400 | source_vs_best_control_h3200 | injection_count | injection_norm_sum | debt_not_exploded | COW_fresh_C3_source_washout_pass | official_C3_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-C4-T11G0-AdamWControlGate-stop800-lr50 | Fashion-MNIST | 0 | 3.7986927489437523 | -0.045703768730163574 | -0.10435402393341064 | -0.2961195707321167 | -1.019582748413086 | -1.4787722826004028 | -1.700298547744751 | 400 | 48.68383268913112 | 0 | 0 | 0 | source_horizon_or_debt_gate_failed |
| MLP-V2206-C4-T10G0-EarlyWarm800ThenOptTransport-lr150 | MNIST | 0 | 2.9828251043252894 | 0.041124701499938965 | 0.004461407661437988 | -0.03865814208984375 | -0.5794615745544434 | -0.7776210308074951 | -0.8718024492263794 | 400 | 303.7348715029657 | 1 | 0 | 0 | source_horizon_or_debt_gate_failed |


### Post-No-Go C-O8 NTK / Jacobian Eigen-Channel

- trigger: C-O7 counterfactual washout failed fresh C3; this run tests a distinct train-only operator principle: candidate directions should live in a controllable train-margin Jacobian mid-spectrum channel rather than a near-null or control-equivalent channel.
- route: `PostNoGoNTKChannelBlocked` previous_post_nogo_route=`PostNoGoCounterfactualWashoutBlocked` blocker=`official_observer_gate_or_fresh_C3_source_horizon_failed`
- probe rows / measured rows: 144/144
- train-only pass / fresh C3 pass / official C3 pass: 0/0/0

| certificate_family | rows | positive_official_early_chain_rows | positive_h3200_rows | positive_retained_h800_h3200_rows | AUC_predict_official_early_chain | AUC_predict_h3200_positive | AUC_predict_retained_h800_h3200_positive | precision_at_top20 | NTKC_train_only_certificate_pass_rows | NTKC_official_observer_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-O8_NTK_Jacobian_eigen_channel_certificate | 144 | 0 | 45 | 13 |  | 0.48484848484848486 | 0.40634174985320026 | 0.0 | 0 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate |


#### C-O8 Selected Candidates

| job_order | v21_id | mechanism | dataset | seed | NTKC_score | source_mid_eigen_channel_fraction | source_jacobian_null_energy_fraction | source_train_margin_gain_mean | mid_channel_control_gap | margin_control_gap | future_audit_source_h800 | future_audit_source_h3200 | NTKC_train_only_certificate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | MLP-V2206-C4-T4G3-SourceStateCarry-lr50 | M262-V2206MetricSolverT4G3SmoothManifoldFU | MNIST | 0 | 2.343481731271855 | 0.003170182423846553 | 0.9942476140836386 | 0.5568775534629822 | 0.0 | 0.3659524619579315 | -0.3284416198730469 | 0.1418299674987793 | 0 |
| 91 | MLP-V2206-C4-T10G0-PeriodicCarry-alt100-lr50 | M268-V2206MetricSolverT10G0CompensatedHiddenBlockFU | KMNIST | 0 | 1.8360876924243725 | 0.0017100011024040864 | 0.9981031728515252 | 0.7426657676696777 | -0.0020743451004087378 | 0.7488074135035276 | 0.07238292694091797 | -0.04722321033477783 | 0 |


#### C-O8 Fresh Eigen-Channel C3

| v21_id | dataset | seed | NTKC_score | source_vs_best_control_h100 | source_vs_best_control_h400 | source_vs_best_control_h800 | source_vs_best_control_h1600 | source_vs_best_control_h2400 | source_vs_best_control_h3200 | projected_mid_channel_fraction_mean | injection_count | injection_norm_sum | debt_not_exploded | NTKC_fresh_C3_eigen_channel_pass | official_C3_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-C4-T4G3-SourceStateCarry-lr50 | MNIST | 0 | 2.343481731271855 | -0.008188724517822266 | -0.04385805130004883 | -0.054555535316467285 | -0.05919361114501953 | -0.2153317928314209 | -0.3362762928009033 | 0.999999533419562 | 400 | 42.61266922110917 | 1 | 0 | 0 | source_horizon_or_debt_gate_failed |
| MLP-V2206-C4-T10G0-PeriodicCarry-alt100-lr50 | KMNIST | 0 | 1.8360876924243725 | 0.0023080110549926758 | -0.022842884063720703 | -0.28974246978759766 | -0.7335479259490967 | -0.9801496267318726 | -1.1315706968307495 | 0.9999994965971141 | 400 | 166.1762551870197 | 1 | 0 | 0 | source_horizon_or_debt_gate_failed |


### Post-No-Go C-O9 Augmentation Tangent Consistency

- trigger: C-O8 NTK/Jacobian channel failed fresh C3; this run tests a distinct train-only input-neighborhood principle: candidate directions should improve local augmentation consistency and train margin relative to random/sign-flip/corrupt controls.
- route: `PostNoGoAugTangentBlocked` previous_post_nogo_route=`PostNoGoNTKChannelBlocked` blocker=`official_observer_gate_or_fresh_C3_source_horizon_failed`
- probe rows / measured rows: 144/144
- train-only pass / fresh C3 pass / official C3 pass: 13/0/0

| certificate_family | rows | positive_official_early_chain_rows | positive_h3200_rows | positive_retained_h800_h3200_rows | AUC_predict_official_early_chain | AUC_predict_h3200_positive | AUC_predict_retained_h800_h3200_positive | precision_at_top20 | ATC_train_only_certificate_pass_rows | ATC_official_observer_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-O9_Augmentation_Tangent_Consistency_Certificate | 144 | 0 | 45 | 13 |  | 0.5012345679012346 | 0.5695830886670581 | 0.0 | 13 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate |


#### C-O9 Selected Candidates

| job_order | v21_id | mechanism | dataset | seed | ATC_score | source_aug_tangent_consistency_gain | consistency_control_gap | source_aug_tangent_margin_gain | margin_control_gap | B2_control_gap | B3_control_gap | future_audit_source_h800 | future_audit_source_h3200 | ATC_train_only_certificate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | MLP-V2206-S2-T7G0-HiddenBlockSolver | M265-V2206MetricSolverT7G0HiddenBlockFU | Fashion-MNIST | 0 | 2.088795072906115 | 7.818639278411865e-05 | 7.842481136322021e-05 | 0.0002721548080444336 | 0.0002924799919128418 | -7.033348083496094e-05 | -7.033348083496094e-05 | -0.5142961740493774 | -0.871238112449646 | 1 |
| 99 | MLP-V2206-C4-T10G0-SGDBootstrap800Carry-lr50 | M268-V2206MetricSolverT10G0CompensatedHiddenBlockFU | KMNIST | 0 | 1.8097775116720256 | 2.3748725652694702e-05 | 2.0995736122131348e-05 | -0.00011989474296569824 | -0.0001347064971923828 | 8.416175842285156e-05 | 8.416175842285156e-05 | 0.0805048942565918 | 0.041371941566467285 | 1 |


#### C-O9 Fresh Aug-Tangent C3

| v21_id | dataset | seed | ATC_score | source_vs_best_control_h100 | source_vs_best_control_h400 | source_vs_best_control_h800 | source_vs_best_control_h1600 | source_vs_best_control_h2400 | source_vs_best_control_h3200 | aug_consistency_gain_mean | injection_count | injection_norm_sum | debt_not_exploded | ATC_fresh_C3_aug_tangent_pass | official_C3_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-S2-T7G0-HiddenBlockSolver | Fashion-MNIST | 0 | 2.088795072906115 | -0.03802764415740967 | -0.17008936405181885 | -0.8065956830978394 | -1.447131872177124 | -1.736900806427002 | -1.8960530757904053 | 5.078024230897427e-05 | 400 | 153.44768126308918 | 0 | 0 | 0 | source_horizon_or_debt_gate_failed |
| MLP-V2206-C4-T10G0-SGDBootstrap800Carry-lr50 | KMNIST | 0 | 1.8097775116720256 | 0.04266023635864258 | 0.0049037933349609375 | -0.01742994785308838 | -0.20737802982330322 | -0.44907867908477783 | -0.623552680015564 | 4.763592965900898e-05 | 400 | 190.8354609720409 | 1 | 0 | 0 | source_horizon_or_debt_gate_failed |


### Post-No-Go C-O10 Bootstrap Basin Consensus

- trigger: C-O9 augmentation tangent consistency failed fresh C3; this run tests a distinct train-only basin principle: candidate directions should align with short-run bootstrap sub-training consensus directions and remain separated from random/sign-flip/corrupt controls.
- route: `PostNoGoBootstrapBasinBlocked` previous_post_nogo_route=`PostNoGoAugTangentBlocked` blocker=`official_observer_gate_or_fresh_C3_source_horizon_failed`
- probe rows / measured rows: 144/144
- train-only pass / fresh C3 pass / official C3 pass: 144/0/0

| certificate_family | rows | positive_official_early_chain_rows | positive_h3200_rows | positive_retained_h800_h3200_rows | AUC_predict_official_early_chain | AUC_predict_h3200_positive | AUC_predict_retained_h800_h3200_positive | precision_at_top20 | BBC_train_only_certificate_pass_rows | BBC_official_observer_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-O10_Bootstrap_Basin_Consensus_Certificate | 144 | 0 | 45 | 13 |  | 0.4637485970819304 | 0.4762184380504991 | 0.0 | 144 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate |


#### C-O10 Selected Candidates

| job_order | v21_id | mechanism | dataset | seed | BBC_score | source_basin_cosine | basin_alignment_control_gap | bootstrap_pair_cosine_mean | bootstrap_consensus_norm_ratio | train_loss_control_gap | B2_control_gap | B3_control_gap | future_audit_source_h800 | future_audit_source_h3200 | BBC_train_only_certificate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 46 | MLP-V2206-S1-T0G0-ReadoutSolver | M259-V2206MetricSolverT0G0ReadoutFU | Fashion-MNIST | 0 | 2.936608055346315 | 0.020668325945734978 | 0.01641678623855114 | 0.9384878079096476 | 0.9792934418745947 | 0.0014560222625732422 | 0.0012612342834472656 | 0.0012612342834472656 | -0.506301999092102 | -1.0011218786239624 | 1 |
| 65 | MLP-V2206-S1-T1G0-SplitTransferSolver | M260-V2206MetricSolverT1G0SplitTransferFU | Fashion-MNIST | 0 | 2.552469588272518 | 0.05050605535507202 | 0.06116668041795492 | 0.9488097429275513 | 0.9828188944512138 | 0.0007913112640380859 | 0.0007903575897216797 | 0.0007903575897216797 | -0.5329755544662476 | -1.0545121431350708 | 1 |


#### C-O10 Fresh Bootstrap-Basin C3

| v21_id | dataset | seed | BBC_score | source_vs_best_control_h100 | source_vs_best_control_h400 | source_vs_best_control_h800 | source_vs_best_control_h1600 | source_vs_best_control_h2400 | source_vs_best_control_h3200 | bootstrap_pair_cosine_mean | bootstrap_consensus_norm_ratio | raw_basin_cosine_mean | train_loss_gain_mean | injection_count | injection_norm_sum | debt_not_exploded | BBC_fresh_C3_bootstrap_basin_pass | official_C3_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-S1-T0G0-ReadoutSolver | Fashion-MNIST | 0 | 2.936608055346315 | -0.007747292518615723 | -0.25688159465789795 | -0.8270628452301025 | -1.3957195281982422 | -1.6828564405441284 | -1.8229573965072632 | 0.9384878079096476 | 0.9792934418745947 | 0.00435277959180894 | 3.531350230332464e-06 | 400 | 94.03323360845025 | 0 | 0 | 0 | source_horizon_or_debt_gate_failed |
| MLP-V2206-S1-T1G0-SplitTransferSolver | Fashion-MNIST | 0 | 2.552469588272518 | 0.02365291118621826 | 0.060553550720214844 | -0.36342108249664307 | -0.9024083614349365 | -1.200688123703003 | -1.4018386602401733 | 0.9488097429275513 | 0.9828188944512138 | 0.00577669204290487 | 2.7250288985669613e-06 | 400 | 73.38235025667564 | 0 | 0 | 0 | source_horizon_or_debt_gate_failed |


### Post-No-Go C-O11 Cross-Fit Influence Transport

- trigger: C-O10 bootstrap basin consensus failed fresh C3; this run tests a distinct train-only cross-fit principle: a source direction built on one train fold should transport through an influence direction from a disjoint train fold and remain safe on a third train fold relative to random/sign-flip/corrupt controls.
- route: `PostNoGoCrossFitInfluenceBlocked` previous_post_nogo_route=`PostNoGoBootstrapBasinBlocked` blocker=`official_observer_gate_or_fresh_C3_source_horizon_failed`
- probe rows / measured rows: 144/144
- train-only pass / fresh C3 pass / official C3 pass: 144/0/0

| certificate_family | rows | positive_official_early_chain_rows | positive_h3200_rows | positive_retained_h800_h3200_rows | AUC_predict_official_early_chain | AUC_predict_h3200_positive | AUC_predict_retained_h800_h3200_positive | precision_at_top20 | CFI_train_only_certificate_pass_rows | CFI_official_observer_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-O11_CrossFit_Influence_Transport_Certificate | 144 | 0 | 45 | 13 |  | 0.4994388327721661 | 0.5684086905460951 | 0.0 | 144 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate |


#### C-O11 Selected Candidates

| job_order | v21_id | mechanism | dataset | seed | CFI_score | source_crossfit_B_loss_gain | B_crossfit_control_gap | source_crossfit_C_safety_loss_gain | C_safety_control_gap | source_influence_cosine | crossfit_alignment_control_gap | future_audit_source_h800 | future_audit_source_h3200 | CFI_train_only_certificate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 46 | MLP-V2206-S1-T0G0-ReadoutSolver | M259-V2206MetricSolverT0G0ReadoutFU | Fashion-MNIST | 0 | 4.308916446953833 | 0.001970052719116211 | 0.0019736289978027344 | 0.0015382766723632812 | 0.0015344619750976562 | 0.017987782135605812 | 0.019217738416045904 | -0.506301999092102 | -1.0011218786239624 | 1 |
| 28 | MLP-V2206-C4-T0G0-SourceStateCarry-lr50 | M259-V2206MetricSolverT0G0ReadoutFU | Fashion-MNIST | 0 | 3.025079247313755 | 0.0016016960144042969 | 0.001611471176147461 | 0.0010879039764404297 | 0.0011196136474609375 | 0.019669359549880028 | 0.024016439449042082 | -0.025346994400024414 | -0.032465338706970215 | 1 |


#### C-O11 Fresh Cross-Fit Influence C3

| v21_id | dataset | seed | CFI_score | source_vs_best_control_h100 | source_vs_best_control_h400 | source_vs_best_control_h800 | source_vs_best_control_h1600 | source_vs_best_control_h2400 | source_vs_best_control_h3200 | raw_crossfit_cosine_mean | crossfit_loss_gain_mean | injection_count | injection_norm_sum | debt_not_exploded | CFI_fresh_C3_crossfit_influence_pass | official_C3_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-S1-T0G0-ReadoutSolver | Fashion-MNIST | 0 | 4.308916446953833 | -0.042677879333496094 | -0.5708853006362915 | -1.2298486232757568 | -1.8189079761505127 | -2.2288674116134644 | -2.5476841926574707 | 0.001205461551189728 | 1.7220631198142654e-05 | 400 | 143.80819461809006 | 0 | 0 | 0 | source_horizon_or_debt_gate_failed |
| MLP-V2206-C4-T0G0-SourceStateCarry-lr50 | Fashion-MNIST | 0 | 3.025079247313755 | -0.008188247680664062 | -0.2582845687866211 | -0.9494001865386963 | -1.6622402667999268 | -2.036086082458496 | -2.3336918354034424 | -0.0030434321144639396 | 1.2513867040979676e-05 | 400 | 111.74412430459779 | 0 | 0 | 0 | source_horizon_or_debt_gate_failed |


### Post-No-Go C-O12 Train-Flow Commutator

- trigger: C-O11 cross-fit influence transport failed fresh C3; this run tests a distinct train-only operator principle: a candidate source step should have a useful finite commutator with local train loss flow, distinguishable from random/sign-flip/corrupt-label commutators.
- route: `PostNoGoTrainFlowCommutatorFreshC3Opened` previous_post_nogo_route=`PostNoGoCrossFitInfluenceBlocked` blocker=`official_observer_gate_failed;official_C3_gate_not_claimed`
- probe rows / measured rows: 144/144
- train-only pass / fresh C3 pass / official C3 pass: 48/1/0

| certificate_family | rows | positive_official_early_chain_rows | positive_h3200_rows | positive_retained_h800_h3200_rows | AUC_predict_official_early_chain | AUC_predict_h3200_positive | AUC_predict_retained_h800_h3200_positive | precision_at_top20 | TFC_train_only_certificate_pass_rows | TFC_official_observer_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-O12_Train_Flow_Commutator_Certificate | 144 | 0 | 45 | 13 |  | 0.42312008978675647 | 0.40458015267175573 | 0.0 | 48 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate |


#### C-O12 Selected Candidates

| job_order | v21_id | mechanism | dataset | seed | TFC_score | source_commutator_B_loss_gain | B_commutator_control_gap | source_commutator_norm_ratio | source_commutator_cosine | commutator_alignment_control_gap | future_audit_source_h800 | future_audit_source_h3200 | TFC_train_only_certificate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 92 | MLP-V2206-C4-T10G0-PeriodicCarry-alt200-lr50 | M268-V2206MetricSolverT10G0CompensatedHiddenBlockFU | MNIST | 0 | 2.005540666480767 | 9.703636169433594e-05 | -2.3603439331054688e-05 | 2.3598996677246517e-08 | -0.011906036175787449 | 0.008325261063873768 | -0.32414495944976807 | 0.20451831817626953 | 1 |
| 39 | MLP-V2206-C4-T9G0-EarlyWarmCarry-stop800-lr50 | M267-V2206MetricSolverT9G0C3GatedHiddenBlockFU | KMNIST | 0 | 1.7005666000652648 | 0.0001125335693359375 | 0.00011229515075683594 | 2.4206813618654273e-08 | -0.023342695087194443 | -0.016630960162729025 | -0.03698325157165527 | -0.10113656520843506 | 1 |


#### C-O12 Fresh Train-Flow Commutator C3

| v21_id | dataset | seed | TFC_score | source_vs_best_control_h100 | source_vs_best_control_h400 | source_vs_best_control_h800 | source_vs_best_control_h1600 | source_vs_best_control_h2400 | source_vs_best_control_h3200 | raw_commutator_cosine_mean | commutator_norm_ratio_mean | commutator_loss_gain_mean | injection_count | injection_norm_sum | debt_not_exploded | TFC_fresh_C3_train_flow_commutator_pass | official_C3_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-C4-T10G0-PeriodicCarry-alt200-lr50 | MNIST | 0 | 2.005540666480767 | 0.039599597454071045 | 0.02066981792449951 | 0.038215458393096924 | 0.05214285850524902 | 0.05790621042251587 | 0.0667877197265625 | -0.00276164268408877 | 4.41449486253554e-09 | 7.777075734338723e-07 | 400 | 417.5873824432492 | 1 | 1 | 0 |  |
| MLP-V2206-C4-T9G0-EarlyWarmCarry-stop800-lr50 | KMNIST | 0 | 1.7005666000652648 | 0.09901082515716553 | 0.17162775993347168 | 0.08425068855285645 | -0.45452046394348145 | -0.8927386999130249 | -1.1488535404205322 | -0.0004474684371380277 | 6.4109440707322115e-09 | 6.025046786817256e-08 | 400 | 235.28929041698575 | 0 | 0 | 0 | source_horizon_or_debt_gate_failed |


### C-O12 Repeat Verification

- trigger: C-O12 opened fresh smoke; this verification reruns the opened train-flow commutator candidate under repeated train seeds with the same AdamW/SGD/NoOp/random controls. Passing here is still fresh evidence only, not official C2/C3.
- route: `TrainFlowCommutatorRepeatVerifyBlocked` blocker=`repeat_fresh_C3_horizon_or_debt_gate_failed;official_observer_gate_not_open;official_C2_solver_not_claimed;official_C3_gate_not_claimed`
- verify candidates / repeat rows / robust fresh rows / official C3 rows: 1/3/0/0

| v21_id | dataset | seed | repeat_rows | required_repeats | fresh_C3_repeat_pass_rows | debt_not_exploded_rows | source_vs_best_control_h100_min | source_vs_best_control_h400_min | source_vs_best_control_h800_min | source_vs_best_control_h1600_min | source_vs_best_control_h3200_min | TFC_repeat_robust_fresh_C3_verified | official_observer_pass | official_C2_solver_claimed | official_C3_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-C4-T10G0-PeriodicCarry-alt200-lr50 | MNIST | 0 | 3 | 3 | 1 | 3 | -0.04353064298629761 | -0.12971240282058716 | -0.15336978435516357 | -0.1816556453704834 | -0.366357684135437 | 0 | 0 | 0 | 0 | repeat_fresh_C3_horizon_or_debt_gate_failed;official_observer_gate_not_open;official_C2_solver_not_claimed;official_C3_gate_not_claimed |


#### C-O12 Verify Fresh Rows

| v21_id | dataset | seed | TFC_score | source_vs_best_control_h100 | source_vs_best_control_h400 | source_vs_best_control_h800 | source_vs_best_control_h1600 | source_vs_best_control_h2400 | source_vs_best_control_h3200 | raw_commutator_cosine_mean | commutator_norm_ratio_mean | commutator_loss_gain_mean | debt_not_exploded | TFC_fresh_C3_train_flow_commutator_pass | official_C3_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-C4-T10G0-PeriodicCarry-alt200-lr50 | MNIST | 0 | 2.005540666480767 | 0.039599597454071045 | 0.02066981792449951 | 0.038215458393096924 | 0.05214285850524902 | 0.05790621042251587 | 0.0667877197265625 | -0.00276164268408877 | 4.41449486253554e-09 | 7.777075734338723e-07 | 1 | 1 | 0 |  |
| MLP-V2206-C4-T10G0-PeriodicCarry-alt200-lr50 | MNIST | 0 | 2.005540666480767 | 0.014820396900177002 | 0.006572842597961426 | -0.014001131057739258 | -0.022987961769104004 | -0.1399608850479126 | -0.3507455587387085 | -0.0018351055877087674 | 4.267246330470164e-09 | 4.703770218839054e-07 | 1 | 0 | 0 | source_horizon_or_debt_gate_failed |
| MLP-V2206-C4-T10G0-PeriodicCarry-alt200-lr50 | MNIST | 0 | 2.005540666480767 | -0.04353064298629761 | -0.12971240282058716 | -0.15336978435516357 | -0.1816556453704834 | -0.1969752311706543 | -0.366357684135437 | -0.0027682180311675553 | 6.267858747203102e-09 | 2.444664232825744e-07 | 1 | 0 | 0 | source_horizon_or_debt_gate_failed |


## Part D Metric-As-Dynamics / Terminal / KAN

| block_role | rows | C2_solver_gate_pass_rows | projection_residual_Gf_min | ActuationR2_max | B2_transfer_gain_max | blocker |
| --- | --- | --- | --- | --- | --- | --- |
| all | 2 | 0 | 0.5676424269788402 | 0.673547625541687 | 0.14279025793075562 | projection_residual_or_actuation_or_transfer_gate_failed |
| hidden_only | 1 | 0 | 0.9974963446704895 | -0.008074641227722168 | 0.02849721908569336 | projection_residual_or_actuation_or_transfer_gate_failed |
| readout_only | 1 | 0 | 1.0819951301050104 | -0.18609833717346191 | -2.3969502449035645 | projection_residual_or_actuation_or_transfer_gate_failed |


| source_h3200 | source_h4800 | R4800_over_3200 | C5_productive_terminal_source | preservation_status | blocker |
| --- | --- | --- | --- | --- | --- |
|  |  |  | 0 | blocked_before_D0_D1 | retained_source_observer_gate_failed |


| mapping_status | blocker | best_metric_v21_id | KAN_source_channel_decision | KAN_specific_delta_vs_MLP_same_metric |
| --- | --- | --- | --- | --- |
| blocked_before_KAN_mapping | retained_source_observer_gate_failed |  | KANMappingNotEntered |  |


## 4GPU Queue

| task_id | line | gpu | command | status | returncode | start_time | end_time | duration_sec | log_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lineB_dche_dfou_reconfirm | B | 2 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_efficiency_reconfirm.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08 | completed | 0 | 2026-06-07 02:18:16 +0800 | 2026-06-07 02:18:18 +0800 | 2.002610921859741 | /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/logs/lineB_dche_dfou_reconfirm.log |
| lineC_retained_source_observer | C | 0 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_retained_source_observer.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08 --top-k 20 | completed | 0 | 2026-06-07 02:18:16 +0800 | 2026-06-07 02:18:18 +0800 | 2.0010178089141846 | /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/logs/lineC_retained_source_observer.log |
| lineA_s015_truth_gate | A | 0 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_s015_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-import-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08 | completed | 0 | 2026-06-07 02:18:16 +0800 | 2026-06-07 02:18:20 +0800 | 4.003793239593506 | /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/logs/lineA_s015_truth_gate.log |
| lineB_drat_drbf_multibatch | B | 3 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_drat_drbf_multibatch.py --device cuda:3 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08 | completed | 0 | 2026-06-07 02:18:16 +0800 | 2026-06-07 02:18:24 +0800 | 8.002783298492432 | /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/logs/lineB_drat_drbf_multibatch.log |
| lineD_metric_dynamics_solver_gate | D | 1 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_metric_dynamics_solver.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08 | completed | 0 | 2026-06-07 02:18:24 +0800 | 2026-06-07 02:18:24 +0800 | 0.04171633720397949 | /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/logs/lineD_metric_dynamics_solver_gate.log |
| lineD_terminal_preservation_gate | D | 1 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_terminal_preservation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08 | completed | 0 | 2026-06-07 02:18:24 +0800 | 2026-06-07 02:18:24 +0800 | 0.04175209999084473 | /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/logs/lineD_terminal_preservation_gate.log |
| lineE_kan_mapping_gate | E | 2 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_kan_source_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08 --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08 | completed | 0 | 2026-06-07 02:18:24 +0800 | 2026-06-07 02:18:24 +0800 | 0.03459310531616211 | /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/logs/lineE_kan_mapping_gate.log |
| finalize_packet_recap | Final | 0 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_08_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08 | completed | 0 | 2026-06-07 02:18:24 +0800 | 2026-06-07 02:18:24 +0800 | 0.053728342056274414 | /home/chengshun.wang/DG-LCA/results/v22_08_retained_source_dynamics_functional_update_basis_efficiency_4gpu/official_v22_08/logs/finalize_packet_recap.log |


| execution_contract_violation | reason | max_idle_gap_sec |
| --- | --- | --- |
| 0 | no_dependency_free_runnable_task_waited_on_idle_gpu_over_10min | 0 |


## 修改记录

- 新增 `experiments/run_v22_08_common.py`：独立 v22.08 结果目录、执行日志、复盘、artifact index 与 bundle helper。
- 更新 `dgkan/fu/metric_solver.py`：给 `solve_metric_readout_update` 增加默认不改变旧行为的 `target_scale` / `block_role` 参数，并输出 block-restricted diagnostics；用于 v22.08 C2 recompute hard gate。
- 新增 `experiments/run_v22_08_s015_truth_gate.py`：S0.15 code/import/LineC/source-chain/solver/semantic/kernel/profiler gate，并新增 C2 repair recompute tests。
- 新增 `experiments/run_v22_08_efficiency_reconfirm.py`：按 v22.08 字段读回 D-CHE/D-FOU measured full-loop artifact，不补造缺失 timing 字段。
- 新增 `experiments/run_v22_08_drat_drbf_multibatch.py`：复用底层 profiler/repair 函数重新跑 D-RAT/D-RBF multibatch 与 component repair waterfall，输出 v22.08 artifact。
- 新增 `experiments/run_v22_08_retained_source_observer.py`：把 v22.08 C-O1..C-O5 retained-source observer 作为 no-commit audit 评分；future source 只作为 label，不生成方向。
- 新增 `experiments/run_v22_08_metric_dynamics_solver.py`、`run_v22_08_terminal_preservation.py`、`run_v22_08_kan_source_mapping.py`：严格按 observer/C3 gate fail-closed，不强行进入 h4800/KAN。
- 新增 `experiments/run_v22_08_post_nogo_source_state_certificate.py`：在 local no-go 后尝试 C-O6 pathwise source-state persistence certificate；方向只来自 train-stream micro-batch source-state consistency/control-null evidence，future source 只作 audit label，并另跑 fresh source-state integration vs AdamW/SGD/NoOp/random controls。
- 新增 `experiments/run_v22_08_post_nogo_counterfactual_washout.py`：在 C-O6 blocked 后继续尝试 C-O7 counterfactual washout persistence；短程 AdamW washout 后比较 source/sign-flip/corrupt/random matched 方向的训练流功能残留，并跑 periodic persistent FU fresh h3200 对照。
- 新增 `experiments/run_v22_08_post_nogo_ntk_channel_certificate.py`：在 C-O7 blocked 后继续尝试 C-O8 NTK/Jacobian eigen-channel certificate；只用 train-margin Jacobian 局部谱通道评估方向可控性，并跑 mid-spectrum projected FU fresh h3200 对照。
- 新增 `experiments/run_v22_08_post_nogo_aug_tangent_certificate.py`：在 C-O8 blocked 后继续尝试 C-O9 augmentation tangent consistency certificate；只用 train-only 输入邻域增强一致性/训练 margin/control-gap 构造证据，并跑 augmentation-tangent FU fresh h3200 对照。
- 新增 `experiments/run_v22_08_post_nogo_bootstrap_basin_certificate.py`：在 C-O9 blocked 后继续尝试 C-O10 bootstrap basin consensus certificate；只用 train-only bootstrap 子训练流形成短程盆地方向，并跑 bootstrap-basin FU fresh h3200 对照。
- 新增 `experiments/run_v22_08_post_nogo_crossfit_influence_certificate.py`：在 C-O10 blocked 后继续尝试 C-O11 cross-fit influence transport certificate；用互斥 train folds 的 source/influence/safety 证据检验方向跨训练子集迁移，并跑 cross-fit influence FU fresh h3200 对照。
- 新增 `experiments/run_v22_08_post_nogo_train_flow_commutator_certificate.py`：在 C-O11 blocked 后继续尝试 C-O12 train-flow commutator certificate；用 source step 与局部 train loss flow 的有限非交换子检验方向是否携带可用训练流结构，并跑 train-flow-commutator FU fresh h3200 对照。
- 新增 `experiments/run_v22_08_train_flow_commutator_fresh_verify.py`：当 C-O12 fresh smoke 打开后，按 repeat train seeds 重新验证同一候选对 AdamW/SGD/NoOp/random controls 的 h100-h3200 source-vs-control；通过也只写成 fresh verification，不声明 official C2/C3。
- 新增 `experiments/run_v22_08_full.py` 与 `experiments/run_v22_08_finalize.py`：4GPU queue、日志、复盘、final route 和 bundle。
- 更新 `experiments/run_v22_08_common.py` 与 `experiments/run_v22_08_finalize.py`：新增 `v22_08_code_review_packet.zip` 打包和 C-O6/C-O7/C-O8/C-O9/C-O10/C-O11/C-O12/verify post-no-go 复盘汇总；results bundle 与 code review packet 均重新生成。

## 分析 / Insight / 证据链

- S0.15 不是只跑旧 S0.14：本轮新增 C2 recompute rows，分别覆盖 target_scale=1/64 与 all/hidden/readout block role，且记录 JVP/VJP/CG、target_norm、update_norm 与 block-restricted mask。
- C2 recompute 中 `projection_residual_Gf` 是 ratio，因此 scale=1 与 scale=64 可以出现相同比值；这不被改写成不同数值。本轮用 target_norm/update_norm scaling 与 block_role mask 作为是否重算的直接证据。
- D-CHE/D-FOU 仍来自 measured full-loop artifact readback；缺失的 functional_direction_ms/metric_solver_ms/LineC_audit_ms 字段保留为空，没有估计填补。
- D-RAT/D-RBF 若 robust multibatch 未闭合，只允许 component repair / limited smoke；repair rows 即使 micro-near-E1，也因 `official_closure_claimed=0` 不提升为 official carrier。
- Retained-source observer 的 official early-chain label 是 h100+h400+h800 同时为正；本轮读回矩阵中该 label 正例为 0，因此所有 C-O families fail-closed。这个结论只针对 v22.08 已尝试的 C-O 评分定义，不是 universal no-go。
- retained h800+h3200 label 仍可有 AUC/precision 读数，但它不能替代 official early-chain gate；因此不能用 late retained/late rebound 进入 Line D。
- Line D 按计划依赖 C-O observer gate。observer fail 后，C2 recompute 只作为 code/solver integrity evidence，不作为 C3 source-formation promotion。
- Terminal preservation 与 KAN mapping 均因 C3 未打开而未进入；这避免了把 blocked-before-gate 的 h4800/KAN 表写成实验成功。
- C-O6 不是 C-O1..C-O5 的同族 scale/cap/floor 调参：它从同一机制在多个合法 train micro-batch 上的 signed displacement 一致性构造 slow source-state probe，并显式记录 max_control_cosine/radial_fraction，试图区分普通 CE/corrupt 控制流。
- C-O6 的 future source 仍只用于 audit label；fresh C3 部分只比较 `PSSC-AdamWPlusSlowSourceState` 对 AdamW/SGD/NoOp/random matched controls 的 horizon delta。即使 fresh smoke 有局部正值，也不会被写成 official C3，除非 h100/h400/h800/h1600/h3200 与 debt gate 同时打开且后续 official gate 补齐。
- C-O7 进一步检查“被短程 AdamW 洗出后仍保留的功能位移”是否可作为 retained-source certificate。该 probe 仍只用 train stream 构造方向和对照；fresh C3 使用 `COW-AdamWPlusPeriodicPersistentFU` 与 AdamW/SGD/NoOp/random periodic controls 比较 horizon delta。
- C-O8 换成 operator-level train-margin Jacobian eigen-channel 证书：候选方向必须在局部 NTK/Jacobian 中谱可控通道里产生 margin effect，并与 random/sign-flip/corrupt controls 区分。fresh C3 使用 frozen mid-spectrum projection，而不是继续调前面 observer 的 scale/cap/floor。
- C-O9 再换成 train-only 输入邻域/augmentation tangent 证书：候选方向必须在 train augmentation 一致性与 margin 上优于 random/sign-flip/corrupt controls。fresh C3 使用 `ATC-AdamWPlusAugTangentFU` 与 AdamW/SGD/NoOp/random matched controls 比较 horizon delta。
- C-O10 再换成 train-only bootstrap basin-consensus 证书：候选方向必须与多个 bootstrap 子训练流形成的短程 basin consensus direction 对齐，并在 train/B2/B3 上不输给 random/sign-flip/corrupt controls。fresh C3 使用 `BBC-AdamWPlusBootstrapBasinFU` 与 AdamW/SGD/NoOp/random matched bootstrap controls 比较 horizon delta。
- C-O10 的 train-only certificate rows 打开不等价于 source formation：本轮 C-O10 official observer 仍为 0，fresh C3 pass rows 仍为 0，top fresh rows 到 h3200 仍为负。因此不能把 bootstrap basin 共识写成 promotion 或 official C3。
- C-O11 再换成 train-only cross-fit influence transport 证书：候选方向必须在 A 折生成后，经 B 折 influence direction 外推，并在 C 折安全检查上不输给 random/sign-flip/corrupt controls。fresh C3 使用 `CFI-AdamWPlusCrossFitInfluenceFU` 与 AdamW/SGD/NoOp/random matched cross-fit controls 比较 horizon delta。
- C-O11 的 train-only certificate rows 若打开仍不等价于 source formation：只有 official observer 与 fresh h100/h400/h800/h1600/h3200/debt gate 同时打开，才允许后续 official C2/C3/terminal 论证；否则必须保持 blocked。
- C-O12 再换成 train-only train-flow commutator 证书：候选方向必须与局部 train loss flow 产生有限非交换残差，并且该残差在 B/C train folds 上不输给 random/sign-flip/corrupt commutator controls。fresh C3 使用 `TFC-AdamWPlusTrainFlowCommutatorFU` 与 AdamW/SGD/NoOp/random matched train-flow-commutator controls 比较 horizon delta。
- C-O12 的 train-only certificate rows 若打开仍不等价于 source formation：只有 official observer 与 fresh h100/h400/h800/h1600/h3200/debt gate 同时打开，才允许后续 official C2/C3/terminal 论证；否则必须保持 blocked。
- C-O12 repeat verification 若 robust fresh 通过，仍然只是重复 fresh smoke：本轮没有 official observer positive label、没有 official C2 solver claim、没有 official C3 claim，因此不能进入 terminal/KAN promotion。
- 最新结论：v22.08 把 v22.07 的 C2 repair 语义漏洞修成可审计重算；C-O6/C-O7/C-O8/C-O9/C-O10/C-O11 之后，C-O12 route=`PostNoGoTrainFlowCommutatorFreshC3Opened`，fresh_C3_pass_rows=1，official_C3_pass_rows=0；repeat route=`TrainFlowCommutatorRepeatVerifyBlocked`，robust_fresh_rows=0，official_C3_pass_rows=0。本轮 C-O12 single-run fresh smoke 打开，但 repeat verification 未稳住，因此 promotion 仍 blocked；继续推进需要 C-O1..C-O12 之外的新 source-observability principle，不能把这条 unstable smoke 写成成功。

_完整 artifact 清单保留在 `v22_08_artifact_index.csv`，不放入复盘正文。_
