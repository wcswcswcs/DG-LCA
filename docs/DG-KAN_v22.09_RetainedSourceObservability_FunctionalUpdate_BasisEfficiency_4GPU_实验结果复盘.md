# DG-KAN v22.09 RetainedSourceObservability FunctionalUpdate BasisEfficiency 4GPU 实验结果复盘

生成时间：2026-06-07 10:50:08 +0800

## Route

- route: `RetainedSourceObserverLocalNoGo_v22.09`
- promotion_allowed: 0
- blocking_metric: `D-RAT_D-RBF_multibatch_robust_or_telemetry;retained_source_certificate_gate;post_boundary_BVFR_gate;post_state_holonomy_gate;post_row_orthogonal_ROST_gate;post_info_volume_IVSC_gate;post_fast_slow_FSMR_gate;post_fast_slow_FSMR_damped_repair_gate;post_time_reversal_TRAC_gate;post_time_reversal_TRAC_damped_repair_gate;C2_true_block_solver_or_observer_gate;C3_source_formation_gate;terminal_preservation_not_entered_or_failed_C3_gate;KAN_mapping_not_entered_C3_or_C4_gate`
- S0.16 pass / CodeRoute: 1 / `R0-CodePacketSelfContained`
- D-CHE/D-FOU S1 pass: 1 / 1
- D-RAT/D-RBF robust pass: 0 / 0
- D-RAT/D-RBF telemetry bridge robust rows / telemetry rows / blocker: 0 / 8 / `forward_ratio`
- D-RAT/D-RBF shape/local-K scan route / candidate groups / rows: `D-RAT_D-RBFShapeLocalKRepairBlocked` / 0 / 36
- C0 retained-source certificate pass rows: 0
- official early-chain positive rows: 0
- retained h800+h3200 positive rows: 13
- post-boundary BVFR route: `PostBoundaryBVFRBlocked`
- post-boundary BVFR train-only/fresh/official pass rows: 126 / 0 / 0
- post-state OSH route: `PostStateHolonomyBlocked`
- post-state OSH train-only/fresh/official pass rows: 9 / 0 / 0
- post-row ROST route: `PostRowOrthogonalBlocked`
- post-row ROST train-only/fresh/official pass rows: 120 / 0 / 0
- post-info IVSC route: `PostInfoVolumeBlocked`
- post-info IVSC train-only/fresh/official pass rows: 126 / 0 / 0
- post-fast-slow FSMR route: `PostFastSlowMemoryBlocked`
- post-fast-slow FSMR train-only/fresh/official pass rows: 143 / 0 / 0
- post-fast-slow FSMR damped repair route: `PostFastSlowMemoryDampedRepairBlocked`
- post-fast-slow FSMR damped repair fresh/official pass rows: 0 / 0
- post-time-reversal TRAC route: `PostTimeReversalAdjointBlocked`
- post-time-reversal TRAC train-only/fresh/official pass rows: 124 / 0 / 0
- post-time-reversal TRAC damped repair route: `PostTimeReversalAdjointDampedRepairBlocked`
- post-time-reversal TRAC damped repair fresh/official pass rows: 0 / 0
- C2/C3/C4 pass rows: 0 / 0 / 0
- KAN mapping decision: `KANMappingNotEntered`
- execution_contract_violation: 0
- minimum_effective_progress: `A-CodeClosure`
- next_codex_action: C-O13/C-O14/C-O15, BVFR, OSH, ROST, IVSC, FSMR, FSMR damped repair, TRAC, and TRAC damped repair did not open official source observability; record boundary or change first principle again

## Part A Code / Packet Truth Gate

| check | pass | metric | value | blocker |
| --- | --- | --- | --- | --- |
| required_source_files | 1 | exists | 43/43 |  |
| compileall_repo | 1 | py_compile | 0 |  |
| import_closure_repo | 1 | import | 0 |  |
| linec_fast_golden | 1 | tests | 9 |  |
| linec_channel_golden | 1 | tests | 8 |  |
| source_chain_tests | 1 | tests | 7/7 |  |
| terminal_retention_tests | 1 | tests | 4/4 |  |
| metric_solver_tests | 1 | tests | 33/33 |  |
| retained_source_observability_tests | 1 | tests | 12/12 |  |
| profiler_phase_tests | 1 | tests | 5/5 |  |
| semantic_contract | 1 | forbidden+alias | forbidden_pass=1;undeclared_alias_pairs=0 |  |
| csv_claimed_exists_but_zip_missing_count | 1 | zip_required_compare | 0 |  |
| self_contained_compileall | 1 | clean_unzip_compile | 0 |  |
| self_contained_import_check | 1 | clean_unzip_import | 0 |  |


### Clean Unzip Self Test

| packet | unzip_root | compileall_returncode | import_returncode | self_contained_import_check | pass |
| --- | --- | --- | --- | --- | --- |
| /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/v22_09_code_review_packet.zip | /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/_code_packet_unzip | 0 | 0 | 1 | 1 |


### Required CSV vs Zip

| path | csv_exists | zip_contains | csv_claimed_exists_but_zip_missing |
| --- | --- | --- | --- |
| dgkan/fu/core.py | 1 | 1 | 0 |
| dgkan/fu/source_chain.py | 1 | 1 | 0 |
| dgkan/fu/terminal_retention.py | 1 | 1 | 0 |
| dgkan/fu/debt_accounting.py | 1 | 1 | 0 |
| dgkan/fu/function_space_metrics.py | 1 | 1 | 0 |
| dgkan/fu/metric_projection.py | 1 | 1 | 0 |
| dgkan/fu/metric_solver.py | 1 | 1 | 0 |
| dgkan/fu/jacobian_sketch.py | 1 | 1 | 0 |
| dgkan/fu/sobolev_metric.py | 1 | 1 | 0 |
| dgkan/fu/rkhs_metric.py | 1 | 1 | 0 |
| dgkan/fu/fisher_metric.py | 1 | 1 | 0 |
| dgkan/fu/basis_channel_metric.py | 1 | 1 | 0 |
| dgkan/fu/retained_source_certificate.py | 1 | 1 | 0 |
| dgkan/fu/train_flow_commutator.py | 1 | 1 | 0 |
| dgkan/fu/source_state_dynamics.py | 1 | 1 | 0 |
| dgkan/fu/optimizer_state_integration.py | 1 | 1 | 0 |
| dgkan/metrics/linec.py | 1 | 1 | 0 |
| dgkan/kernels/fused_chebyshev_k3.py | 1 | 1 | 0 |
| dgkan/kernels/fused_fourier_k2.py | 1 | 1 | 0 |
| dgkan/kernels/fused_rational_k4.py | 1 | 1 | 0 |

_仅显示前 20 / 43 rows；完整 CSV 见 artifact。_


## Part B Basis Efficiency

| carrier | variant_family | forward_ratio_vs_mlp | backward_ratio_vs_mlp | step_ratio_vs_mlp | memory_ratio_vs_mlp | functional_runner_same_kernel | fallback_kernel_used | official_fused_kernel_complete | v22_09_S1_pass | v22_09_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | CHE21-R2/R4 | 0.9305108277155166 |  | 0.46682782835473463 | 0.9913236539978382 | 1 | 0 | 1 | 1 |  |
| D-FOU | FOU21-R3 | 1.0087315984225318 |  | 0.5089279836996342 | 0.9888469358117921 | 1 | 0 | 1 | 1 |  |


### D-RAT / D-RBF Multibatch

| carrier | profile_rows | robust_production_pass_rows | near_E1_rows | component_telemetry_complete_rows | best_forward_ratio | best_step_ratio | best_memory_ratio | decision | blocker | repair_attempt_rows | repair_micro_near_E1_rows | repair_best_variant | repair_attempt_status | repair_blocker | official_closure_claimed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 4 | 0 | 4 | 0 | 1.6893629833541228 | 0.5776855607530009 | 0.8889482858803022 | NearE1LimitedSmokeOnly | forward_ratio;component_telemetry_incomplete | 28 | 22 | RAT22.03-R5-numden-fused-backward | MicroNearE1RunnerBlocked | official_fused_missing;functional_runner_kernel_mismatch | 0 |
| D-RBF | 4 | 0 | 4 | 0 | 1.8086909080398306 | 0.6097079764107836 | 0.8889482858803022 | NearE1LimitedSmokeOnly | forward_ratio;component_telemetry_incomplete | 28 | 4 | RBF22.03-R4-exp-approx-trainpath | MicroNearE1RunnerBlocked | official_fused_missing;functional_runner_kernel_mismatch | 0 |


### D-RAT / D-RBF Telemetry Bridge

- bridge purpose: combine official fused runner timing with component-complete repair telemetry by carrier/batch to test whether telemetry incompleteness is still the decisive blocker. This is audit-only and does not claim official closure.
| carrier | profile_rows | robust_production_pass_rows | near_E1_rows | component_telemetry_complete_rows | best_forward_ratio | best_backward_ratio | best_step_ratio | best_memory_ratio | decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 4 | 0 | 4 | 4 | 1.6893629833541228 | 0.4885228047703645 | 0.5776855607530009 | 0.8889482858803022 | NearE1LimitedSmokeOnly | forward_ratio |
| D-RBF | 4 | 0 | 4 | 4 | 1.8086909080398306 | 0.5431276780288211 | 0.6097079764107836 | 0.8889482858803022 | NearE1LimitedSmokeOnly | forward_ratio |


### D-RAT / D-RBF Tile64 Repair Attempt

- repair result: changing official Triton forward batch tile to 64 for B>=128 was tried, measured, and then reverted because it worsened forward ratio. The failed run is retained as artifact evidence.
| carrier | profile_rows | robust_production_pass_rows | near_E1_rows | component_telemetry_complete_rows | best_forward_ratio | best_backward_ratio | best_step_ratio | best_memory_ratio | decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 4 | 0 | 2 | 0 | 1.826345650276404 | 0.46715369786448774 | 0.6125192150696943 | 0.8889482858803022 | OfficialFusedBlocked | forward_ratio;component_telemetry_incomplete |
| D-RBF | 4 | 0 | 4 | 0 | 2.0251449831267068 | 0.53686617473395 | 0.6213620916355306 | 0.8889482858803022 | NearE1LimitedSmokeOnly | forward_ratio;component_telemetry_incomplete |


### D-RBF Exp2 Repair Attempts

- repair result: replacing Gaussian `exp(-0.5*r*r)` with an equivalent `exp2` path was tried as both full forward/backward and forward-only variants. Neither opened robust production; the full path improved one forward readout but introduced step/backward blocker, and the forward-only path remained worse than the stable direct-`exp` kernel. Both failed runs are retained as audit artifacts and the final kernel was restored.
#### Full Path Exp2

| carrier | profile_rows | robust_production_pass_rows | near_E1_rows | component_telemetry_complete_rows | best_forward_ratio | best_backward_ratio | best_step_ratio | best_memory_ratio | decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 4 | 0 | 4 | 0 | 1.62879816820383 | 0.427362739187783 | 0.503600533302005 | 0.8889482858803022 | NearE1LimitedSmokeOnly | forward_ratio;component_telemetry_incomplete |
| D-RBF | 4 | 0 | 4 | 0 | 1.4630903747199637 | 1.3683722739482203 | 1.3797999673892463 | 0.8889482858803022 | NearE1LimitedSmokeOnly | forward_ratio;step_ratio;component_telemetry_incomplete |


#### Forward-Only Exp2

| carrier | profile_rows | robust_production_pass_rows | near_E1_rows | component_telemetry_complete_rows | best_forward_ratio | best_backward_ratio | best_step_ratio | best_memory_ratio | decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 4 | 0 | 4 | 0 | 1.7765925045227533 | 0.41393562459500177 | 0.5017955568047384 | 0.8889482858803022 | NearE1LimitedSmokeOnly | forward_ratio;component_telemetry_incomplete |
| D-RBF | 4 | 0 | 2 | 0 | 1.8583730847065625 | 1.5185359238677376 | 1.5421757084925156 | 0.8889482858803022 | OfficialFusedBlocked | forward_ratio;step_ratio;component_telemetry_incomplete |


### D-RAT / D-RBF Shape Local-K Repair Scan

- repair result: h64/h96/h128 and D-RBF K2/K4 were scanned under the same v22.09 ratio gates. These rows are diagnostic only; they do not replace the official h128/K4 main gate.
| carrier | repair_variant | scan_hidden | profile_rows | robust_production_pass_rows | near_E1_rows | best_forward_ratio | best_step_ratio | best_memory_ratio | shape_localk_candidate | official_closure_claimed | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | RAT22.05-official-rational-k4-triton | 128 | 4 | 0 | 4 | 1.6646557302846312 | 0.6277875403856997 | 0.8889482858803022 | 0 | 0 | forward_ratio;component_telemetry_incomplete |
| D-RAT | RAT22.05-official-rational-k4-triton | 64 | 4 | 0 | 2 | 1.5660793928055257 | 0.8961523190915319 | 0.938978109527759 | 0 | 0 | forward_ratio;step_ratio;component_telemetry_incomplete |
| D-RAT | RAT22.05-official-rational-k4-triton | 96 | 4 | 0 | 3 | 1.8986183732885502 | 0.929503544985782 | 0.9107663398299164 | 0 | 0 | forward_ratio;step_ratio;component_telemetry_incomplete |
| D-RBF | RBF22.03-R1-compact-local-k4-no-dense | 128 | 4 | 0 | 4 | 1.815193837464525 | 0.4703204646108033 | 0.8889482858803022 | 0 | 0 | forward_ratio;component_telemetry_incomplete |
| D-RBF | RBF22.03-R1-compact-local-k4-no-dense | 64 | 4 | 0 | 3 | 1.9743385457784988 | 1.508496900163891 | 0.938978109527759 | 0 | 0 | forward_ratio;step_ratio;component_telemetry_incomplete |
| D-RBF | RBF22.03-R1-compact-local-k4-no-dense | 96 | 4 | 0 | 3 | 1.941055349687285 | 1.4180523220538137 | 0.9107663398299164 | 0 | 0 | forward_ratio;step_ratio;component_telemetry_incomplete |
| D-RBF | RBF22.03-R1-compact-local-low-k2-no-dense | 128 | 4 | 0 | 3 | 2.006834442493199 | 0.9955277097250671 | 0.938978109527759 | 0 | 0 | forward_ratio;step_ratio;component_telemetry_incomplete |
| D-RBF | RBF22.03-R1-compact-local-low-k2-no-dense | 64 | 4 | 0 | 4 | 1.7301432739800082 | 1.4420154624156714 | 0.9790230189880131 | 0 | 0 | forward_ratio;step_ratio;component_telemetry_incomplete |
| D-RBF | RBF22.03-R1-compact-local-low-k2-no-dense | 96 | 4 | 0 | 3 | 1.1771323328825194 | 1.4563459780727996 | 0.9570405112874961 | 0 | 0 | forward_ratio;step_ratio;component_telemetry_incomplete |


### Repair / Limited Smoke

| carrier | repair_attempt_rows | repair_micro_near_E1_rows | repair_component_telemetry_complete_rows | repair_best_forward_ratio | repair_best_step_ratio | repair_best_memory_ratio | repair_best_variant | repair_attempt_status | repair_blocker | official_closure_claimed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 28 | 22 | 28 | 2.2813103434824162 | 1.63281395318779 | 1.0625 | RAT22.03-R5-numden-fused-backward | MicroNearE1RunnerBlocked | official_fused_missing;functional_runner_kernel_mismatch | 0 |
| D-RBF | 28 | 4 | 28 | 1.9824271508435165 | 1.3455729263771141 | 1.0338325500488281 | RBF22.03-R4-exp-approx-trainpath | MicroNearE1RunnerBlocked | official_fused_missing;functional_runner_kernel_mismatch | 0 |

| carrier | scope | promotion_allowed | limited_smoke_executed | status | reason |
| --- | --- | --- | --- | --- | --- |
| D-RAT | MNIST seed0 h800/h1600 only | 0 | 0 | deferred_no_official_fused_functional_runner | near_E1 allows smoke, but official fused kernel/full FU runner gate remains closed; no success is claimed |
| D-RBF | MNIST seed0 h800/h1600 only | 0 | 0 | deferred_no_official_fused_functional_runner | near_E1 allows smoke, but official fused kernel/full FU runner gate remains closed; no success is claimed |


## Part C Retained-Source Observability

- best heldout observer: `C-O13_Train_Flow_Algebra_Certificate` heldout_precision=0.0 precision@20=0.0
| certificate_family | rows | positive_official_early_chain_rows | positive_h3200_rows | positive_retained_h800_h3200_rows | AUC_predict_official_early_chain | AUC_predict_h3200_positive | AUC_predict_retained_h800_h3200_positive | precision_at_top20 | heldout_precision_at_top20 | control_equivalent_fraction | C0_retained_source_certificate_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-O13_Train_Flow_Algebra_Certificate | 204 | 0 | 45 | 13 |  | 0.6788259958071279 | 0.6910994764397905 | 0.0 | 0.0 | 0.0 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate;heldout_precision_gate |
| C-O14_Drift_Diffusion_Signal_Channel_Certificate | 204 | 0 | 45 | 13 |  | 0.30328441649196364 | 0.47603705195328233 | 0.0 | 0.0 | 1.0 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate;heldout_precision_gate;control_equivalent_fraction_gate;control_null_residual_source_gate |
| C-O15_Block_Coordinate_Source_State_Certificate | 204 | 0 | 45 | 13 |  | 0.6666666666666666 | 0.6600886024969794 | 0.0 | 0.0 | 0.0 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate;heldout_precision_gate;block_source_age_h800_gate |


### Top C-O13 Candidates

| job_order | v21_id | dataset | seed | C-O13_score | commutator_norm_ratio | flow_order_gap | source_direction_reproducibility | future_audit_source_h100 | future_audit_source_h400 | future_audit_source_h800 | future_audit_source_h3200 | official_early_chain_h100_h400_h800_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 150 | MLP-V2206-C4-T10G0-OptTransport-lr150 | MNIST | 0 | 0.28386377798699164 | 0.017427384430934 | 0.00012420523853506893 | 1.0 | -1.0936940908432007 | -0.7010617256164551 | -0.35333728790283203 | 0.09051167964935303 | 0 |
| 202 | MLP-V2206-C4-T12G0-SourceStateCarry | Fashion-MNIST | 0 | 0.2829859764238478 | 0.01837802903869059 | 9.752405458129942e-05 | 1.0 | -1.0627917051315308 | -0.14960205554962158 | 0.04750847816467285 | -0.05565071105957031 | 0 |
| 168 | MLP-V2206-C4-T10G0-OptTransport-lr50 | MNIST | 0 | 0.27999506982814415 | 0.021789070916464762 | 2.357090124860406e-05 | 1.0 | -1.1534048318862915 | -0.781975507736206 | -0.42433226108551025 | 0.24899780750274658 | 0 |
| 193 | MLP-V2206-C4-T10G0-EarlyWarm400ThenOptTransport-lr150 | KMNIST | 0 | 0.27906809297518465 | 0.02265686534624863 | 0.00012598256580531597 | 1.0 | -0.3280905485153198 | 0.10294008255004883 | 0.13653504848480225 | -0.2212611436843872 | 0 |
| 110 | MLP-V2206-C4-T10G0-EarlyWarmCarry-stop1200-lr50 | Fashion-MNIST | 0 | 0.2779631757008313 | 0.02395719670539121 | 0.0001013062137644738 | 1.0 | -1.0158993005752563 | -0.07163035869598389 | 0.02028369903564453 | -0.11038780212402344 | 0 |
| 132 | MLP-V2206-C4-T10G0-SGDBootstrap400Carry-lr50 | MNIST | 0 | 0.2773007630684553 | 0.010240066897408341 | 1.4229241060093045e-05 | 0.9557713149109759 | -1.0643600225448608 | -0.659675121307373 | -0.30107808113098145 | -0.06018543243408203 | 0 |
| 100 | MLP-V2206-C4-T10G0-SourceStateCarry | KMNIST | 0 | 0.27728228903504154 | 0.024513399923168696 | 0.00019359589350642636 | 1.0 | -0.40669143199920654 | -0.0477142333984375 | -0.0548710823059082 | -0.10781145095825195 | 0 |
| 186 | MLP-V2206-C4-T10G0-AdamWBootstrap800OptTransport-lr150 | MNIST | 0 | 0.27476055806531363 | 0.016353755742760955 | 0.00011730194091796875 | 0.9661847813981997 | -0.36696505546569824 | -0.595177173614502 | -0.6795488595962524 | -0.6536520719528198 | 0 |
| 197 | MLP-V2206-C4-T10G0-EarlyWarm800ThenOptTransport-lr150 | Fashion-MNIST | 0 | 0.2730517624826861 | 0.029338730627412962 | 2.600922016426921e-06 | 1.0 | -0.9307035207748413 | 0.023661375045776367 | 0.17415201663970947 | -0.12717044353485107 | 0 |
| 200 | MLP-V2206-C4-T12G0-OptTransport-lr150 | MNIST | 0 | 0.2712019377751841 | 0.031343642015069006 | 0.00014217333227861673 | 1.0 | -1.153650164604187 | -0.7657277584075928 | -0.39424872398376465 | 0.042084455490112305 | 0 |
| 129 | MLP-V2206-C4-T10G0-PeriodicCarry-alt200-lr50 | Fashion-MNIST | 0 | 0.2692496780091023 | 0.033489109811433805 | 9.89220425253734e-05 | 1.0 | -0.9508649110794067 | -0.10181307792663574 | 0.04776310920715332 | -0.2137467861175537 | 0 |
| 106 | MLP-V2206-C4-T10G0-EarlyWarmCarry-stop800-lr50 | KMNIST | 0 | 0.25072297714482267 | 0.020546946911946452 | 2.165275509469211e-05 | 0.8991427028277715 | -0.41435062885284424 | -3.075599670410156e-05 | 0.0403752326965332 | 0.0033316612243652344 | 0 |

_仅显示前 12 / 20 rows；完整 CSV 见 artifact。_


### Post-Boundary BVFR Certificate

- trigger: C-O13/C-O14/C-O15 observer gate stayed closed; this follow-up tests a different first-principles boundary-value flow replay certificate. The source step is treated as a train-only boundary condition, replayed through short local optimizer flow, and compared with random/sign/corrupt boundary controls. Future source columns remain audit labels only.
| certificate_family | rows | positive_official_early_chain_rows | positive_h3200_rows | positive_retained_h800_h3200_rows | AUC_predict_official_early_chain | AUC_predict_h3200_positive | AUC_predict_retained_h800_h3200_positive | precision_at_top20 | BVFR_train_only_certificate_pass_rows | BVFR_official_observer_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BVFR_Boundary_Value_Flow_Replay_Certificate | 144 | 0 | 45 | 13 |  | 0.48641975308641977 | 0.5085143863769818 | 0.0 | 126 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate |


#### BVFR Selected Candidates

| job_order | v21_id | mechanism | dataset | seed | BVFR_score | source_boundary_replay_B_loss_gain | B_boundary_replay_control_gap | boundary_replay_norm_ratio | boundary_replay_source_cosine | future_audit_source_h800 | future_audit_source_h3200 | BVFR_train_only_certificate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 28 | MLP-V2206-C4-T0G0-SourceStateCarry-lr50 | M259-V2206MetricSolverT0G0ReadoutFU | Fashion-MNIST | 0 | 2.64152790170952 | 4.3392181396484375e-05 | 5.245208740234375e-05 | 0.00010054816583453759 | 0.9943137764930725 | -0.025346994400024414 | -0.032465338706970215 | 1 |
| 56 | MLP-V2206-C4-T0G0-EarlyWarmCarry-stop800-lr50 | M259-V2206MetricSolverT0G0ReadoutFU | KMNIST | 0 | 2.1023814559834504 | 4.2438507080078125e-05 | 6.532669067382812e-05 | 0.00010071091214854604 | 0.9934178590774536 | 0.08240294456481934 | -0.1705937385559082 | 1 |


#### BVFR Fresh C3

| v21_id | dataset | seed | BVFR_score | source_vs_best_control_h100 | source_vs_best_control_h400 | source_vs_best_control_h800 | source_vs_best_control_h1600 | source_vs_best_control_h2400 | source_vs_best_control_h3200 | boundary_replay_source_cosine_mean | boundary_replay_norm_ratio_mean | debt_not_exploded | BVFR_fresh_C3_boundary_replay_pass | official_C3_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-C4-T0G0-SourceStateCarry-lr50 | Fashion-MNIST | 0 | 2.64152790170952 | 0.06614243984222412 | -0.12874078750610352 | -0.773114800453186 | -1.4547141790390015 | -1.8017951250076294 | -2.0736583471298218 | 0.466200089699596 | 0.004253595289344768 | 0 | 0 | 0 | source_horizon_or_debt_gate_failed |
| MLP-V2206-C4-T0G0-EarlyWarmCarry-stop800-lr50 | KMNIST | 0 | 2.1023814559834504 | 0.06460213661193848 | 0.1612081527709961 | -0.15549898147583008 | -0.6045325994491577 | -0.8546193838119507 | -0.9986504316329956 | 0.29131971944105317 | 0.000772077424751333 | 1 | 0 | 0 | source_horizon_or_debt_gate_failed |


### Post-State OSH Certificate

- trigger: BVFR failed fresh/official gate; this follow-up implements the plan's train-flow algebra recommendation with optimizer-state holonomy. A source boundary is embedded into short AdamW state cycles over split orders, and fresh runs inject a live source/momentum mixed direction against AdamW/SGD/NoOp/random matched controls.
| certificate_family | rows | positive_official_early_chain_rows | positive_h3200_rows | positive_retained_h800_h3200_rows | AUC_predict_official_early_chain | AUC_predict_h3200_positive | AUC_predict_retained_h800_h3200_positive | precision_at_top20 | OSH_train_only_certificate_pass_rows | OSH_official_observer_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OSH_Optimizer_State_Holonomy_Certificate | 144 | 0 | 45 | 13 |  | 0.46487093153759823 | 0.4339401056958309 | 0.0 | 9 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate |


#### OSH Selected Candidates

| job_order | v21_id | mechanism | dataset | seed | OSH_score | source_stateful_B_loss_gain | B_stateful_control_gap | stateful_holonomy_ratio | stateful_holonomy_control_gap | source_momentum_cosine | future_audit_source_h800 | future_audit_source_h3200 | OSH_train_only_certificate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 56 | MLP-V2206-C4-T0G0-EarlyWarmCarry-stop800-lr50 | M259-V2206MetricSolverT0G0ReadoutFU | KMNIST | 0 | 1.8301667376196518 | 3.641843795776367e-05 | 3.039836883544922e-05 | 0.009599574972446808 | -6.871022083930478e-06 | 0.04528120905160904 | 0.08240294456481934 | -0.1705937385559082 | 1 |
| 46 | MLP-V2206-S1-T0G0-ReadoutSolver | M259-V2206MetricSolverT0G0ReadoutFU | Fashion-MNIST | 0 | 1.6371542702722892 | 6.705522537231445e-05 | 3.49879264831543e-05 | 0.007685371904742827 | -2.7029745224421697e-06 | 0.026964807882905006 | -0.506301999092102 | -1.0011218786239624 | 1 |


#### OSH Fresh C3

| v21_id | dataset | seed | OSH_score | source_vs_best_control_h100 | source_vs_best_control_h400 | source_vs_best_control_h800 | source_vs_best_control_h1600 | source_vs_best_control_h2400 | source_vs_best_control_h3200 | source_momentum_cosine_mean | momentum_norm_ratio_mean | debt_not_exploded | OSH_fresh_C3_state_holonomy_pass | official_C3_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-C4-T0G0-EarlyWarmCarry-stop800-lr50 | KMNIST | 0 | 1.8301667376196518 | -0.024121642112731934 | -0.12482380867004395 | -0.46796107292175293 | -0.9356253147125244 | -1.2001675367355347 | -1.3732030391693115 | 0.039936632365861445 | 0.6551469042518654 | 1 | 0 | 0 | source_horizon_or_debt_gate_failed |
| MLP-V2206-S1-T0G0-ReadoutSolver | Fashion-MNIST | 0 | 1.6371542702722892 | 0.10859560966491699 | -0.7291704416275024 | -1.4951826333999634 | -2.1735843420028687 | -2.64144766330719 | -2.988242030143738 | 0.033016495355605004 | 0.4323521141875863 | 0 | 0 | 0 | source_horizon_or_debt_gate_failed |


### Post-Row ROST Certificate

- trigger: OSH failed fresh/official gate; this follow-up implements the plan's row-wise angular stability direction. Candidate source updates are projected into each weight row's tangent space to suppress row-norm jitter, then compared with random/sign/corrupt row-orthogonal controls and fresh AdamW/SGD/NoOp/random matched controls.
| certificate_family | rows | positive_official_early_chain_rows | positive_h3200_rows | positive_retained_h800_h3200_rows | AUC_predict_official_early_chain | AUC_predict_h3200_positive | AUC_predict_retained_h800_h3200_positive | precision_at_top20 | ROST_train_only_certificate_pass_rows | ROST_official_observer_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ROST_Row_Orthogonal_Source_Transport_Certificate | 144 | 0 | 45 | 13 |  | 0.4334455667789001 | 0.4227833235466823 | 0.0 | 120 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate |


#### ROST Selected Candidates

| job_order | v21_id | mechanism | dataset | seed | ROST_score | source_row_orthogonal_B_loss_gain | B_row_orthogonal_control_gap | row_norm_drift | row_tangential_energy_fraction | row_tangent_source_cosine | future_audit_source_h800 | future_audit_source_h3200 | ROST_train_only_certificate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 126 | MLP-V2206-C4-T10G0-AdamWBootstrap800OptTransport-lr150 | M268-V2206MetricSolverT10G0CompensatedHiddenBlockFU | MNIST | 0 | 3.4128508521386864 | 7.915496826171875e-05 | 9.632110595703125e-05 | 4.1422467234042415e-08 | 0.9490016461455677 | 0.949001669883728 | -0.6795488595962524 | -0.6536520719528198 | 1 |
| 66 | MLP-V2206-C4-T0G0-SourceStateCarry | M259-V2206MetricSolverT0G0ReadoutFU | Fashion-MNIST | 0 | 2.573898329146653 | 4.57763671875e-05 | 3.814697265625e-05 | 6.152856002472618e-08 | 0.9457839810902879 | 0.9457839131355286 | 0.1727752685546875 | -0.05905318260192871 | 1 |


#### ROST Fresh C3

| v21_id | dataset | seed | ROST_score | source_vs_best_control_h100 | source_vs_best_control_h400 | source_vs_best_control_h800 | source_vs_best_control_h1600 | source_vs_best_control_h2400 | source_vs_best_control_h3200 | row_tangential_energy_fraction_mean | row_tangent_source_cosine_mean | debt_not_exploded | ROST_fresh_C3_row_orthogonal_pass | official_C3_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-C4-T10G0-AdamWBootstrap800OptTransport-lr150 | MNIST | 0 | 3.4128508521386864 | -0.04421710968017578 | -0.009959876537322998 | -0.022787511348724365 | -0.03442978858947754 | -0.048040807247161865 | -0.0522347092628479 | 0.9304551199245994 | 0.930455117970705 | 1 | 0 | 0 | source_horizon_or_debt_gate_failed |
| MLP-V2206-C4-T0G0-SourceStateCarry | Fashion-MNIST | 0 | 2.573898329146653 | -0.05193626880645752 | -0.22431695461273193 | -0.9031153917312622 | -1.5990866422653198 | -1.9986388683319092 | -2.213092803955078 | 0.9473392503060143 | 0.9473392491042614 | 0 | 0 | 0 | source_horizon_or_debt_gate_failed |


### Post-Info IVSC Certificate

- trigger: ROST failed fresh/official gate; this follow-up implements the plan's signal-channel / information-volume direction. It measures per-example train-logit displacement drift, diffusion, effective rank, and information volume, then runs fresh source updates against AdamW/SGD/NoOp/random matched controls.
| certificate_family | rows | positive_official_early_chain_rows | positive_h3200_rows | positive_retained_h800_h3200_rows | AUC_predict_official_early_chain | AUC_predict_h3200_positive | AUC_predict_retained_h800_h3200_positive | precision_at_top20 | IVSC_train_only_certificate_pass_rows | IVSC_official_observer_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IVSC_Information_Volume_Signal_Channel_Certificate | 144 | 0 | 45 | 13 |  | 0.4599326599326599 | 0.4703464474456841 | 0.0 | 126 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate |


#### IVSC Selected Candidates

| job_order | v21_id | mechanism | dataset | seed | IVSC_score | source_IVSC_B_loss_gain | B_IVSC_control_gap | DDR | DDR_control_gap | effective_rank | info_volume | future_audit_source_h800 | future_audit_source_h3200 | IVSC_train_only_certificate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 28 | MLP-V2206-C4-T0G0-SourceStateCarry-lr50 | M259-V2206MetricSolverT0G0ReadoutFU | Fashion-MNIST | 0 | 2.4287779904777254 | 4.410743713378906e-05 | 5.3882598876953125e-05 | 0.15416772222102082 | 2.114885956111534e-06 | 1.6391221284866333 | 0.0038136036600917578 | -0.025346994400024414 | -0.032465338706970215 | 1 |
| 21 | MLP-V2206-C4-T0G0-SourceStateCarry | M259-V2206MetricSolverT0G0ReadoutFU | MNIST | 0 | 2.2878464963148426 | 2.5510787963867188e-05 | -2.2172927856445312e-05 | 0.411916705368586 | -4.301523645500183e-06 | 2.9589884281158447 | 0.004153393674641848 | -0.2500298023223877 | 0.1464231014251709 | 1 |


#### IVSC Fresh C3

| v21_id | dataset | seed | IVSC_score | source_vs_best_control_h100 | source_vs_best_control_h400 | source_vs_best_control_h800 | source_vs_best_control_h1600 | source_vs_best_control_h2400 | source_vs_best_control_h3200 | DDR_mean | signal_energy_fraction_mean | effective_rank_mean | debt_not_exploded | IVSC_fresh_C3_info_volume_pass | official_C3_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-C4-T0G0-SourceStateCarry-lr50 | Fashion-MNIST | 0 | 2.4287779904777254 | -0.08521616458892822 | -0.2116074562072754 | -0.8943390846252441 | -1.564297080039978 | -1.8920036554336548 | -2.1677695512771606 | 0.043118710808869525 | 0.039003720305623324 | 3.6371585670113564 | 0 | 0 | 0 | source_horizon_or_debt_gate_failed |
| MLP-V2206-C4-T0G0-SourceStateCarry | MNIST | 0 | 2.2878464963148426 | -0.057463109493255615 | -0.07273262739181519 | -0.07888883352279663 | -0.08538573980331421 | -0.08912551403045654 | -0.09830081462860107 | 0.029619623032492975 | 0.026298781860481898 | 3.1852087885141374 | 1 | 0 | 0 | source_horizon_or_debt_gate_failed |


### Post-Fast-Slow FSMR Certificate

- trigger: IVSC failed fresh/official gate; this follow-up implements the plan's fast/slow source-memory and source-state age direction. It builds short train-only source-memory traces, checks slow-memory alignment/overwrite against random/sign/corrupt controls, then runs fresh AdamW+slow-source-memory updates against AdamW/SGD/NoOp/random matched controls.
| certificate_family | rows | positive_official_early_chain_rows | positive_h3200_rows | positive_retained_h800_h3200_rows | AUC_predict_official_early_chain | AUC_predict_h3200_positive | AUC_predict_retained_h800_h3200_positive | precision_at_top20 | FSMR_train_only_certificate_pass_rows | FSMR_official_observer_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FSMR_Fast_Slow_Source_Memory_Resonance_Certificate | 144 | 0 | 45 | 13 |  | 0.5147025813692481 | 0.537874339401057 | 0.0 | 143 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate |


#### FSMR Selected Candidates

| job_order | v21_id | mechanism | dataset | seed | FSMR_score | source_FSMR_B_loss_gain | B_FSMR_control_gap | slow_fast_alignment_mean | source_state_age_fraction | source_state_overwrite_fraction | slow_memory_source_cosine | future_audit_source_h800 | future_audit_source_h3200 | FSMR_train_only_certificate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 37 | MLP-V2206-C4-T7G0-SourceStateCarry | M265-V2206MetricSolverT7G0HiddenBlockFU | KMNIST | 0 | 2.2076953965391195 | 0.00010371208190917969 | 5.078315734863281e-05 | 0.5281694205477834 | 1.0 | 0.22795617493526285 | 0.8829489350318909 | 0.20036637783050537 | 0.2813851833343506 | 1 |
| 17 | MLP-V2206-S2-T7G0-HiddenBlockSolver | M265-V2206MetricSolverT7G0HiddenBlockFU | KMNIST | 0 | 1.905559325724111 | 0.0001049041748046875 | 4.6253204345703125e-05 | 0.4450955593492836 | 1.0 | 0.20415876519508958 | 0.9555367827415466 | -0.29926395416259766 | -0.7201063632965088 | 1 |


#### FSMR Fresh C3

| v21_id | dataset | seed | FSMR_score | source_vs_best_control_h100 | source_vs_best_control_h400 | source_vs_best_control_h800 | source_vs_best_control_h1600 | source_vs_best_control_h2400 | source_vs_best_control_h3200 | slow_fast_alignment_mean | source_state_overwrite_fraction_mean | memory_loss_gain_mean | debt_not_exploded | FSMR_fresh_C3_fast_slow_memory_pass | official_C3_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-C4-T7G0-SourceStateCarry | KMNIST | 0 | 2.2076953965391195 | 0.029252171516418457 | -0.1579608917236328 | -0.5420629978179932 | -1.043696641921997 | -1.2928184270858765 | -1.4095040559768677 | 0.3150993192824535 | 0.24874715210968062 | 7.773771130814567e-07 | 0 | 0 | 0 | source_horizon_or_debt_gate_failed |
| MLP-V2206-S2-T7G0-HiddenBlockSolver | KMNIST | 0 | 1.905559325724111 | -0.1303011178970337 | 0.0026102066040039062 | -0.2809426784515381 | -0.8120005130767822 | -1.186484932899475 | -1.4180058240890503 | 0.3210444548574742 | 0.24778722624035276 | 6.042563336450257e-07 | 1 | 0 | 0 | source_horizon_or_debt_gate_failed |


#### FSMR Damped Repair Fresh C3

- repair trigger: FSMR fresh rows showed early local positives followed by h800-h3200 washout and one debt failure; this repair lowers FU injection, raises slow-memory beta, and refreshes less often. It is an integration-stability test only, not a new official observer.
| v21_id | dataset | seed | FSMR_score | source_vs_best_control_h100 | source_vs_best_control_h400 | source_vs_best_control_h800 | source_vs_best_control_h1600 | source_vs_best_control_h2400 | source_vs_best_control_h3200 | slow_fast_alignment_mean | source_state_overwrite_fraction_mean | memory_loss_gain_mean | debt_not_exploded | FSMR_fresh_C3_fast_slow_memory_pass | official_C3_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-C4-T7G0-SourceStateCarry | KMNIST | 0 | 2.2076953965391195 | 0.029488205909729004 | -0.15733003616333008 | -0.5411465167999268 | -1.0423901081085205 | -1.291139006614685 | -1.4075895547866821 | 0.1094192453371943 | 0.0613081163424599 | 1.4215547707863153e-07 | 0 | 0 | 0 | source_horizon_or_debt_gate_failed |
| MLP-V2206-S2-T7G0-HiddenBlockSolver | KMNIST | 0 | 1.905559325724111 | -0.02161550521850586 | 0.006903648376464844 | -0.2800612449645996 | -0.810826301574707 | -1.1849795579910278 | -1.416190266609192 | 0.10907351089845178 | 0.06288738246060516 | 1.147692819358781e-07 | 1 | 0 | 0 | source_horizon_or_debt_gate_failed |


### Post-Time-Reversal TRAC Certificate

- trigger: FSMR damped repair failed fresh/official gate; this follow-up changes principle again. TRAC treats a candidate update as a local train-flow coordinate: after a source step and short AdamW transport, an adjoint inverse update should partially close a train-only round trip and remain separated from random/sign/corrupt controls. Future source columns remain audit labels only.
| certificate_family | rows | positive_official_early_chain_rows | positive_h3200_rows | positive_retained_h800_h3200_rows | AUC_predict_official_early_chain | AUC_predict_h3200_positive | AUC_predict_retained_h800_h3200_positive | precision_at_top20 | TRAC_train_only_certificate_pass_rows | TRAC_official_observer_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TRAC_Time_Reversal_Adjoint_Consistency_Certificate | 144 | 0 | 45 | 13 |  | 0.4271604938271605 | 0.4198473282442748 | 0.0 | 124 | 0 | no_positive_official_early_chain_label_rows;auc_official_early_chain_gate;precision_top20_gate |


#### TRAC Selected Candidates

| job_order | v21_id | mechanism | dataset | seed | TRAC_score | source_TRAC_B_loss_gain | B_TRAC_control_gap | source_transport_cosine | source_adjoint_closure_ratio | adjoint_closure_control_gap | source_roundtrip_loss_gap | future_audit_source_h800 | future_audit_source_h3200 | TRAC_train_only_certificate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 28 | MLP-V2206-C4-T0G0-SourceStateCarry-lr50 | M259-V2206MetricSolverT0G0ReadoutFU | Fashion-MNIST | 0 | 1.8100835154729529 | 4.410743713378906e-05 | 5.3882598876953125e-05 | 0.9446508288383484 | 0.997098881793534 | -0.003317651733295679 | 0.06290030479431152 | -0.025346994400024414 | -0.032465338706970215 | 1 |
| 46 | MLP-V2206-S1-T0G0-ReadoutSolver | M259-V2206MetricSolverT0G0ReadoutFU | Fashion-MNIST | 0 | 1.7774927087826087 | 4.9591064453125e-05 | 4.863739013671875e-05 | 0.9744712710380554 | 0.9988635675379983 | -0.003922532558194192 | 0.08468484878540039 | -0.506301999092102 | -1.0011218786239624 | 1 |


#### TRAC Fresh C3

| v21_id | dataset | seed | TRAC_score | source_vs_best_control_h100 | source_vs_best_control_h400 | source_vs_best_control_h800 | source_vs_best_control_h1600 | source_vs_best_control_h2400 | source_vs_best_control_h3200 | adjoint_transport_cosine_mean | adjoint_transport_residual_ratio_mean | adjoint_loss_gain_mean | debt_not_exploded | TRAC_fresh_C3_time_reversal_adjoint_pass | official_C3_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-C4-T0G0-SourceStateCarry-lr50 | Fashion-MNIST | 0 | 1.8100835154729529 | -0.08915936946868896 | -0.08373880386352539 | -0.7554031610488892 | -1.432206630706787 | -1.7737400531768799 | -2.036725401878357 | 0.030107801160738745 | 1.9061657088035702 | 2.1536194253712892e-07 | 0 | 0 | 0 | source_horizon_or_debt_gate_failed |
| MLP-V2206-S1-T0G0-ReadoutSolver | Fashion-MNIST | 0 | 1.7774927087826087 | -0.17062664031982422 | -0.7499784231185913 | -1.4610490798950195 | -2.1291533708572388 | -2.5747663974761963 | -2.9035520553588867 | 0.0286791276691838 | 1.8272615202439568 | 2.2090942366048695e-07 | 0 | 0 | 0 | source_horizon_or_debt_gate_failed |


#### TRAC Damped Repair Fresh C3

- repair trigger: TRAC fresh rows lost to controls from h100 and both selected rows failed debt; this repair lowers FU injection, reduces transported-source mixing, and refreshes less often. It is an integration-stability test only, not a new official observer.
| v21_id | dataset | seed | TRAC_score | source_vs_best_control_h100 | source_vs_best_control_h400 | source_vs_best_control_h800 | source_vs_best_control_h1600 | source_vs_best_control_h2400 | source_vs_best_control_h3200 | adjoint_transport_cosine_mean | adjoint_transport_residual_ratio_mean | adjoint_loss_gain_mean | debt_not_exploded | TRAC_fresh_C3_time_reversal_adjoint_pass | official_C3_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-C4-T0G0-SourceStateCarry-lr50 | Fashion-MNIST | 0 | 1.8100835154729529 | -0.04628098011016846 | -0.08462762832641602 | -0.7566975355148315 | -1.4339954853057861 | -1.775855302810669 | -2.039037346839905 | 0.03025134149870986 | 1.9061798346093834 | 5.2852556109428406e-08 | 0 | 0 | 0 | source_horizon_or_debt_gate_failed |
| MLP-V2206-S1-T0G0-ReadoutSolver | Fashion-MNIST | 0 | 1.7774927087826087 | -0.0549851655960083 | -0.7502120733261108 | -1.4610884189605713 | -2.129093289375305 | -2.574763059616089 | -2.9035511016845703 | 0.03255881508068301 | 1.83767748485597 | 1.1412202751726137e-08 | 0 | 0 | 0 | source_horizon_or_debt_gate_failed |


## Part D Metric Solver / Source Formation / Terminal / KAN

| block_role | rows | C2_solver_gate_pass_rows | projection_residual_Gf_min | ActuationR2_max | B2_transfer_gain_max | blocker |
| --- | --- | --- | --- | --- | --- | --- |
| all | 2 | 0 | 0.6117080369613683 | 0.6256974935531616 | 0.22854465246200562 | projection_residual_or_actuation_or_transfer_gate_failed |
| hidden_only | 1 | 0 | 0.994962120022437 | 0.00974416732788086 | 0.019373059272766113 | projection_residual_or_actuation_or_transfer_gate_failed |
| readout_only | 1 | 0 | 1.2460874747538198 | -0.5532140731811523 | -1.2502224445343018 | projection_residual_or_actuation_or_transfer_gate_failed |

| case | target_family | block_role | projection_residual_Gf | ActuationR2 | B2_transfer_gain | solve_time_ms | NDS | C2_solver_gate_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0_loss_cotangent_all_x1 | T0-LossCotangent | all | 0.6819851688923696 | 0.5348961353302002 | 0.22854465246200562 | 3.3565391786396503 |  | 0 | projection_residual_or_actuation_or_transfer_or_solve_time_gate_failed |
| T3_train_flow_all_x8 | T10-CompensatedBlockSourceChannel | all | 0.6117080369613683 | 0.6256974935531616 | 0.13690853118896484 | 4.676891956478357 |  | 0 | projection_residual_or_actuation_or_transfer_or_solve_time_gate_failed |
| T4_block_source_hidden_x8 | T10-CompensatedBlockSourceChannel | hidden_only | 0.994962120022437 | 0.00974416732788086 | 0.019373059272766113 | 4.0474082343280315 |  | 0 | projection_residual_or_actuation_or_transfer_or_solve_time_gate_failed |
| T4_block_source_readout_x8 | T10-CompensatedBlockSourceChannel | readout_only | 1.2460874747538198 | -0.5532140731811523 | -1.2502224445343018 | 4.0587917901575565 |  | 0 | projection_residual_or_actuation_or_transfer_or_solve_time_gate_failed |

| source_formation_status | reason | C3_source_formation_pass | official_C3_pass |
| --- | --- | --- | --- |
| blocked_before_C3 | retained_source_certificate_gate_failed | 0 | 0 |

| terminal_preservation_status | source_h3200 | source_h4800 | R4800_over_3200 | C4_terminal_preservation_pass | blocker |
| --- | --- | --- | --- | --- | --- |
| blocked_before_C4 |  |  |  | 0 | C3_source_formation_gate_failed |

| mapping_status | blocker | KAN_source_channel_decision | KAN_specific_delta_vs_MLP_same_metric |
| --- | --- | --- | --- |
| blocked_before_KAN_mapping | MLP_C3_source_formation_gate_failed | KANMappingNotEntered |  |


## 4GPU Queue

| task_id | line | gpu | command | status | returncode | start_time | end_time | duration_sec | log_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lineC_retained_source_observer_reset | C | 1 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_retained_source_observer.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_07_metric_dynamics_functional_update_basis_efficiency_4gpu/official_v22_07 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 --top-k 20 | completed | 0 | 2026-06-07 08:52:19 +0800 | 2026-06-07 08:52:21 +0800 | 2.0011444091796875 | /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/logs/lineC_retained_source_observer_reset.log |
| lineA_s016_truth_gate | A | 0 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_s016_truth_gate.py --mode all --source-root /home/chengshun.wang/DG-LCA --self-contained-check 1 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 | completed | 0 | 2026-06-07 08:52:19 +0800 | 2026-06-07 08:52:27 +0800 | 8.003578662872314 | /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/logs/lineA_s016_truth_gate.log |
| lineB_basis_efficiency_closure | B | 2 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_basis_efficiency_closure.py --device cuda:2 --official-transition-batch-sizes 128,256,512,1024 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 | completed | 0 | 2026-06-07 08:52:19 +0800 | 2026-06-07 08:52:27 +0800 | 8.002775430679321 | /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/logs/lineB_basis_efficiency_closure.log |
| lineD_metric_dynamics_solver_gate | D | 1 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_metric_dynamics_solver.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 | completed | 0 | 2026-06-07 08:52:27 +0800 | 2026-06-07 08:52:29 +0800 | 2.0374197959899902 | /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/logs/lineD_metric_dynamics_solver_gate.log |
| lineD_terminal_preservation_gate | D | 1 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_terminal_preservation.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 | completed | 0 | 2026-06-07 08:52:29 +0800 | 2026-06-07 08:52:29 +0800 | 0.04157662391662598 | /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/logs/lineD_terminal_preservation_gate.log |
| lineE_kan_mapping_gate | E | 3 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_kan_mapping.py --source-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 | completed | 0 | 2026-06-07 08:52:29 +0800 | 2026-06-07 08:52:29 +0800 | 0.04083418846130371 | /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/logs/lineE_kan_mapping_gate.log |
| finalize_packet_recap | Final | 0 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_09_finalize.py --out-dir /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09 | completed | 0 | 2026-06-07 08:52:29 +0800 | 2026-06-07 08:52:30 +0800 | 1.3790843486785889 | /home/chengshun.wang/DG-LCA/results/v22_09_retained_source_observability_functional_update_basis_efficiency_4gpu/official_v22_09/logs/finalize_packet_recap.log |

| execution_contract_violation | reason | max_idle_gap_sec |
| --- | --- | --- |
| 0 | no_dependency_free_runnable_task_waited_on_idle_gpu_over_10min | 0 |


## Failure Taxonomy

| blocker | route | repair_or_next_direction |
| --- | --- | --- |
| D-RAT_D-RBF_multibatch_robust_or_telemetry | RetainedSourceObserverLocalNoGo_v22.09 | component telemetry completion; remove telemetry from train path; fused rational num/den reciprocal or RBF active local backward |
| retained_source_certificate_gate | RetainedSourceObserverLocalNoGo_v22.09 | stop C-O13/C-O15 variants; define a new first-principles retained-source certificate |
| post_boundary_BVFR_gate | RetainedSourceObserverLocalNoGo_v22.09 | BVFR boundary replay failed fresh/official gate; change principle, not thresholds |
| post_state_holonomy_gate | RetainedSourceObserverLocalNoGo_v22.09 | OSH optimizer-state holonomy failed fresh/official gate; record boundary or change principle again |
| post_row_orthogonal_ROST_gate | RetainedSourceObserverLocalNoGo_v22.09 | ROST row-orthogonal source transport failed fresh/official gate; record row-stability boundary or change principle |
| post_info_volume_IVSC_gate | RetainedSourceObserverLocalNoGo_v22.09 | IVSC information-volume signal-channel failed fresh/official gate; record signal-channel boundary or change principle |
| post_fast_slow_FSMR_gate | RetainedSourceObserverLocalNoGo_v22.09 | FSMR fast-slow source memory failed fresh/official gate; record source-memory boundary or change principle |
| post_fast_slow_FSMR_damped_repair_gate | RetainedSourceObserverLocalNoGo_v22.09 | FSMR damped integration repair failed fresh/official gate; record integration-stability boundary or change principle |
| post_time_reversal_TRAC_gate | RetainedSourceObserverLocalNoGo_v22.09 | TRAC time-reversal adjoint consistency failed fresh/official gate; record adjoint-transport boundary or change principle |
| post_time_reversal_TRAC_damped_repair_gate | RetainedSourceObserverLocalNoGo_v22.09 | TRAC damped integration repair failed fresh/official gate; record integration-stability boundary or change principle |
| C2_true_block_solver_or_observer_gate | RetainedSourceObserverLocalNoGo_v22.09 | repair only after C0 opens; reduce rank/damping/block CG |
| C3_source_formation_gate | RetainedSourceObserverLocalNoGo_v22.09 | optimizer-state and slow source-state integration comparison |
| terminal_preservation_not_entered_or_failed_C3_gate | RetainedSourceObserverLocalNoGo_v22.09 | C-O13/C-O14/C-O15, BVFR, OSH, ROST, IVSC, FSMR, FSMR damped repair, TRAC, and TRAC damped repair did not open official source observability; record boundary or change first principle again |
| KAN_mapping_not_entered_C3_or_C4_gate | RetainedSourceObserverLocalNoGo_v22.09 | C-O13/C-O14/C-O15, BVFR, OSH, ROST, IVSC, FSMR, FSMR damped repair, TRAC, and TRAC damped repair did not open official source observability; record boundary or change first principle again |


## 修改记录

- 新增 `dgkan/fu/retained_source_certificate.py`、`train_flow_commutator.py`、`source_state_dynamics.py`、`optimizer_state_integration.py`：把 v22.09 C-O13/C-O14/C-O15 与 optimizer-state evidence 写成纯函数和 unit tests。
- 新增 `dgkan/profiling/efficiency_v22_09.py`：记录 D-RAT/D-RBF component telemetry keys 与 unit tests。
- 新增 `experiments/run_v22_09_common.py`：独立 v22.09 结果目录、日志、required source list、SHA manifest、code packet、bundle helper。
- 新增 `experiments/run_v22_09_s016_truth_gate.py`：基于 final code packet 解压目录执行 compile/import self-contained gate，并核对 required CSV 与 zip entries。
- 新增并更新 `experiments/run_v22_09_basis_efficiency_closure.py`：D-CHE/D-FOU measured artifact readback，D-RAT/D-RBF multibatch officialization、repair waterfall、limited smoke 与 telemetry bridge；bridge 只作 audit，不声明 official closure。
- 尝试并撤回 D-RAT/D-RBF official Triton forward tile64 repair：把 `dgkan/kernels/fused_rational_k4.py` 与 `dgkan/kernels/fused_rbf.py` 的 B>=128 forward tile 改成 64 后重测，forward ratio 变差，因此恢复原策略；失败结果保留为 `v22_09_drat_drbf_tile64_repair_*` artifact。
- 尝试并撤回 D-RBF `exp2` Gaussian repair：分别测试 full forward/backward 与 forward-only 两种等价指数路径；均未打开 robust production，最终恢复稳定 direct-`exp` kernel，失败结果保留为 `v22_09_drat_drbf_rbf_exp2_*` artifact。
- 新增 `experiments/run_v22_09_drat_drbf_shape_localk_repair_scan.py`：扫描 D-RAT/D-RBF hidden=64/96/128 与 D-RBF K2/K4 local-K 形态，作为 forward blocker 的诊断 evidence；不替代 official h128/K4 closure gate。
- 新增 `experiments/run_v22_09_retained_source_observer.py`：执行 C-O13 train-flow algebra、C-O14 drift-diffusion signal-channel、C-O15 block-coordinate source-state 三类 train-only certificate scoring；future source 只作 audit label。
- 新增 `experiments/run_v22_09_post_boundary_flow_replay_certificate.py`：在 C-O13/C-O14/C-O15 blocked 后尝试 BVFR boundary-value flow replay certificate；用 train-only boundary condition + local optimizer replay residual 构造方向，并跑 fresh h100-h3200 vs AdamW/SGD/NoOp/random matched controls。
- 新增 `experiments/run_v22_09_post_optimizer_state_holonomy_certificate.py`：在 BVFR blocked 后尝试 OSH optimizer-state holonomy certificate；用多 split 短 AdamW state cycle 的 holonomy / momentum alignment 构造 train-only 证据，并跑 source+momentum fresh h100-h3200 对照。
- 新增 `experiments/run_v22_09_post_row_orthogonal_source_transport_certificate.py`：在 OSH blocked 后尝试 ROST row-orthogonal source transport certificate；把 source/control update 投影到 row-tangent 子空间，记录 row_norm_drift / tangential energy / train transfer，并跑 row-orthogonal fresh h100-h3200 对照。
- 新增 `experiments/run_v22_09_post_info_volume_signal_channel_certificate.py`：在 ROST blocked 后尝试 IVSC information-volume signal-channel certificate；用 train-only per-example logit displacement 记录 drift/diffusion/DDR/effective_rank/info_volume，并跑 info-volume fresh h100-h3200 对照。
- 新增 `experiments/run_v22_09_post_fast_slow_memory_resonance_certificate.py`：在 IVSC blocked 后尝试 FSMR fast/slow source-memory resonance certificate；用 train-only 短轨迹 slow source memory 的 alignment/age/overwrite 构造证据，并跑 AdamW+slow-memory fresh h100-h3200 对照。
- 新增 `experiments/run_v22_09_post_fast_slow_memory_damped_repair.py`：针对 FSMR fresh 的 h800-h3200 washout/debt blocker，重跑低 FU 强度、高 beta、低刷新频率的 damped integration repair；仍只作为 fresh repair evidence，不声明 official。
- 新增 `experiments/run_v22_09_post_time_reversal_adjoint_certificate.py`：在 FSMR damped repair blocked 后尝试 TRAC time-reversal adjoint consistency certificate；用 train-only source-step + AdamW transport + adjoint inverse round-trip closure 构造证据，并跑 time-reversal-adjoint fresh h100-h3200 对照。
- 新增 `experiments/run_v22_09_post_time_reversal_adjoint_damped_repair.py`：针对 TRAC fresh 的 h100 起点失败与 debt blocker，重跑低 FU 强度、低 transport mix、低刷新频率的 damped integration repair；仍只作为 fresh repair evidence，不声明 official。
- 修复 BVFR/OSH fresh selection：fresh C3 优先选择 train-only certificate pass rows；只有没有 pass row 时才退回按 score 选取，避免把未过证书硬门的高分候选误作 fresh source candidate。
- 新增 `experiments/run_v22_09_metric_dynamics_solver.py`、`run_v22_09_terminal_preservation.py`、`run_v22_09_kan_mapping.py`：按 C0/C2/C3/C4 gate fail-closed，不强行进入 terminal/KAN。
- 新增 `experiments/run_v22_09_full.py` 与 `run_v22_09_finalize.py`：4GPU queue、final route、复盘、failure taxonomy、artifact index、code review packet 与 results bundle。

## 分析 / Insight / 证据链

- v22.09 修复了 v22.08 code packet 的审计漏洞：truth gate 不只检查 repo 文件存在，还构建 `v22_09_code_review_packet.zip`，解压后执行 `compileall` 与 required import closure，并记录 `csv_claimed_exists_but_zip_missing_count`。
- D-CHE/D-FOU 仍按 measured full-loop readback 判定，不补造缺失 timing 字段；它们是否 S1 pass 由 forward/step/memory/same-kernel/fallback/official fused gate 决定。
- D-RAT/D-RBF 使用 v22.09 更严格的 robust gate：>=3/4 batch sizes 需要 forward<=1.25、step<=1.25、memory<=1.05、gradcheck、official fused、same-kernel、component telemetry 同时成立。repair rows 不会被提升成 official closure。
- D-RAT/D-RBF telemetry bridge 将 official fused runner timing 与 component-complete repair telemetry 对齐后，component_telemetry_complete_rows 变为 4/4，但 robust rows 仍为 0，blocker 收敛到 `forward_ratio`。因此不能把 telemetry 缺口修成 efficiency closure。
- Tile64 kernel repair 是真实负结果：B>=128 使用 64 batch tile 后 D-RAT/D-RBF forward ratio 变差，说明当前 forward blocker 不是简单 grid launch/tile overhead；该改动已撤回，失败数据保留供审计。
- D-RBF exp2 repair 是真实负结果：base-2 equivalent exponential 没有稳定改善 official robust gate；full path 引入 step/backward blocker，forward-only path 仍未优于稳定 direct-`exp` 路径。因此不能把 exp approximation 写成 D-RBF closure。
- Shape/local-K scan 只检验当前 blocker 是否来自 hidden/K 形态；即便某个低 hidden 或 K2 组出现 candidate，也只说明下一轮可设计新的 official carrier contract，不能 retroactively 关闭本轮 h128/K4 D-RAT/D-RBF gate。
- C-O13/C-O14/C-O15 均只用 v22.07 train-only C0/C1/C2 artifact 构造证书分数，future horizon 只作为 audit label。若 official early-chain 正例为 0，AUC/precision official gate 必须 fail-closed。
- BVFR 是 C-O13/C-O14/C-O15 之后的 post-boundary 尝试，不是同族 threshold/cap/scale 调参：它把候选 source update 作为边界条件，在短局部 AdamW replay 后取 residual direction，再用 disjoint train folds 和 random/sign/corrupt boundary controls 检查是否仍有可用训练流结构。
- BVFR fresh C3 仍只允许作为 fresh evidence；如果 h100/h400/h800/h1600/h3200/debt gate 或 official observer/C2/C3 未同时打开，final route 必须保持 promotion blocked。
- OSH 进一步把计划中的 optimizer-state/vector-field consistency 显式化：source boundary 必须在 AdamW stateful split cycle 中呈现低 holonomy、正 transfer、momentum alignment，并在 fresh run 中通过 live AdamW moment mixed update 战胜 AdamW/SGD/NoOp/random controls。
- OSH 初次跑出 9 行 train-only pass，但 selected fresh top-2 未过 train-only gate；因此先修正 selection 语义再重跑，不能把选择逻辑瑕疵直接写成科学结论。
- OSH 若只有 train-only rows 打开，仍不等价于 retained-source formation；必须以 fresh horizon/debt 与 official observer/C2/C3 共同开门为准。
- ROST 是按计划中 Nora / row-wise angular stability 推荐新增的第一性尝试：若 retained source 被 row norm jitter 擦除，row-tangent source transport 应降低 row_norm_drift，同时保持 B/C train transfer 并在 fresh horizon 打败 AdamW/SGD/NoOp/random controls。
- ROST 若只有 train-only rows 或局部 h100 正值，仍不等价于 source formation；必须 h100/h400/h800/h1600/h3200/debt 与 official observer/C2/C3 同时打开才可 promotion。
- IVSC 是按计划中 signal-channel / reservoir / information-volume 推荐新增的第一性尝试：source update 的 per-example train-logit 位移应呈现 coherent drift > diffusion、非退化 effective rank，并且这些信号不能被 random/sign/corrupt controls 解释。
- IVSC 若只在 train-only DDR/effective-rank 上有读数，仍不等价于 retained source；fresh h100-h3200/debt 和 official observer/C2/C3 gate 未开时必须保持 blocked。
- FSMR 是按计划中 fast/slow source memory、AdEMAMix 式慢记忆与 source-state age 推荐新增的第一性尝试：若 source 真在训练流中形成可保留慢态，短轨迹 slow memory 应与 fast source update 保持正 alignment、低 overwrite，并相对 random/sign/corrupt memory controls 有可用 train transfer。
- FSMR fresh C3 仍只比较 `FSMR-AdamWPlusFastSlowMemoryFU` 对 AdamW/SGD/NoOp/random matched controls 的真实 h100-h3200 horizon；即使 train-only memory certificate rows 打开，没有 fresh/debt 与 official observer/C2/C3 同开也不能进入 Line D。
- FSMR damped repair 是 blocker-directed integration test：若失败来自注入过强或 slow-memory overwrite 过大，降低 `fu_lr`、提高 `memory_beta`、降低 refresh frequency 应改善 h800-h3200。若仍失败，就支持 source-memory principle 在当前候选/runner 下未形成稳健 retained source。
- TRAC 是 FSMR-damped 后再次更换的第一性尝试，不是 slow-memory 阈值调参：若 source update 真是局部训练流坐标，source step 经短 AdamW transport 后，其 adjoint inverse 应能在 train-only round trip 中降低 closure residual，并相对 random/sign/corrupt controls 保持 transfer 和 transport consistency。
- TRAC fresh C3 仍只比较 `TRAC-AdamWPlusTimeReversalAdjointFU` 对 AdamW/SGD/NoOp/random matched controls 的真实 h100-h3200 horizon；没有 fresh/debt 与 official observer/C2/C3 同开时不能进入 Line D。
- TRAC damped repair 是 blocker-directed integration test：如果 TRAC 失败主要来自注入过强、transport mixing 过大或刷新过频，降低 `fu_lr`、降低 `transport_mix`、提高 refresh interval 应改善 h100-h3200。若仍失败，就支持 adjoint-transport principle 在当前候选/runner 下未形成稳健 retained source。
- retained h800+h3200 positive rows 仍可作为旁证，但不能替代 official early-chain h100+h400+h800 label，也不能打开 Line D。
- C2 solver matrix 是真实 tiny-model recompute evidence；但当 C0 retained-source certificate 未打开时，C2 只说明 solver integrity，不声明 C1 target 或 C3 source formation。
- Terminal preservation 与 KAN mapping 均严格依赖 MLP C3/C4。blocked-before-gate 的 rows 是状态证据，不是实验成功或失败 horizon。
- 最新结论：route=`RetainedSourceObserverLocalNoGo_v22.09`，minimum_effective_progress=`A-CodeClosure`。若 BVFR/OSH/ROST/IVSC/FSMR/FSMR-damped/TRAC/TRAC-damped 仍未打开 official gate，本轮应写成实现/审计进展与 post-boundary negative evidence；不能把 train-only/fresh smoke 包装成 promotion。

_完整 artifact 清单保留在 `v22_09_artifact_index.csv`，不放入复盘正文。_
