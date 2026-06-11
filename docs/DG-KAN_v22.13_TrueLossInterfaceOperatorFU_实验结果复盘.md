# DG-KAN v22.13 TrueLossInterfaceOperatorFU 实验结果复盘

生成时间：2026-06-08 23:01:53 +0800

## Route

- route: `R9-TrueLossInterfaceOperatorExplorationSuccess_KANPending`
- exploration_promotion_allowed: `1`
- official_promotion_allowed: `1`
- official_kan_carrier_promotion_allowed: `0`
- scientific_claim_allowed: `0`
- blocking_metric: `KANSourceChannelMismatchConfirmed`
- minimum_effective_progress: `A-CodeTruth;B-Operator;C-Adapter;D-Efficiency;E-KAN`
- next_codex_action: `repair KAN basis-channel carrier mismatch with a new basis-state objective or carrier architecture`
- results bundle: `results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_results_bundle.zip`
- code review packet: `results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_code_review_packet.zip`

## 0. Repair and GPU audit

- S0 fixes: narrowed semantic false-positive matching, added clean-packet source dependencies, and marked norm-capped operator laws as nonlinear instead of claiming linearity.
- Operator core fixes: made control-null/random helpers row-permutation equivariant and disabled row smoothing where it broke the law test.
- CUDA bug fix: changed `dgkan/fu/constructive_commit.py` random target generation to create CPU RNG samples then move them to the target device, fixing the MLP+FU CUDA mismatch in task readback.
- Horizon protocol fix: replaced the fixed single-update diagnostic with per-adapter `T(delta)` commits, then aborted the CPU run and reran the official matrix on `cuda:1`; CPU horizon artifacts are diagnostic only.
- KAN GPU fix: replaced the inherited CPU-only v22.12 corrected-layout probe with a v22.13 local GPU probe and ran K15/K16 on `cuda:2`.
- Norm-scale plumbing fix: made `--norm-scale` affect S3 selected target and per-adapter horizon `T(delta)` instead of silently falling back to `0.16`.
- Repair sweep fix: added `diagnostic_only=1` GPU preview runner for selected horizons; the first attempt failed because it reused the full h3200/h4800 evaluator, and the failure is recorded in the execution log.
- Row-axis control-null fix: removed `row_axis_control` from the default control span because it erased PairwiseRanking/Preference row-wise cotangents; row-axis remains available only for diagnostics.
- LIO7 diagnostic operator: added `LIO7_LowNDSControlNull` to test the planned LowNDS + control-null repair direction; it is tracked in S0/S2 law tests but not promoted unless full horizon gates pass.
- Source-anchor retention fix: added a role-blind per-variant displacement anchor loss in the horizon training loop (`retention_weight`, `retention_start_step`, `retention_stop_step`); FU and controls each retain their own initial source anchor, so controls do not inherit the FU target.
- Finalizer promotion semantics fix: split official loss-interface operator promotion from KAN carrier promotion, matching the plan's separate 14.2 and 14.3 gates.
- Native fused cotangent fix: split D-FOU/D-CHE fused CE backward into generic `backward_from_grad_logits` paths, routed v22.13 efficiency through PrimitiveKAN fused forward caches, and replaced hardcoded gradcheck with native-vs-autograd gradient relative-error checks.
- KAN basis-channel repair: added `basis_linearized_w1` KAN commits and gain sweep (`1/4/16`) to test whether source retention can be carried by basis parameters instead of corrected readout replay.
- Honesty guard: no task/eval/readback metric is used for direction; no autograd-native row is promoted as official fused kernel when arbitrary-cotangent fused backward is missing.
- Latest repair sweep route: `RepairSweepPreviewPassFound`; preview_pass_rows=18; best_candidate=`LIO7_LowNDSControlNull;5.12;Delta-RankingAdapter;low_lr_source_anchor_w0p10_lr0p02`; diagnostic_only=1。
| evidence_id | scope | gpu | pid | used_memory_mib | evidence_source | status | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| live_nvidia_smi_horizon | operator_horizon | cuda:1 | 749994 | 750 | observed via nvidia-smi --query-compute-apps while process was running | blocked | route=S5-RoleBlindOperatorHorizonNoGo c3=0 c4=0 device=cuda:1 |
| live_nvidia_smi_kan | KAN_mapping | cuda:2 | 762411 | 750 | observed via nvidia-smi --query-compute-apps while process was running | completed | route=R7-DFOUCorrectedSourceOpened_DCHEBlocked pass_rows=0 device=cuda:2 |
| declared_device_efficiency | native_efficiency | cuda:0 |  |  | command journal --device argument | blocked | route=R3-KernelNativeEfficiencyBlocked |
| declared_device_task_eval | task_eval | cuda:3 |  |  | command journal --device argument | completed | route=TaskReadbackCompleted |
| live_nvidia_smi_repair_sweep_rowaxis | operator_repair_sweep | cuda:1 | 785223 | 750 | observed via nvidia-smi --query-compute-apps while process was running | completed | route=RepairSweepPreviewPassFound preview_pass_rows=10 diagnostic_only=1 |
| live_nvidia_smi_horizon_rowaxis | operator_horizon | cuda:1 | 788869 | 750 | observed via nvidia-smi --query-compute-apps while process was running | blocked | route=S5-RoleBlindOperatorHorizonNoGo c3=1 c4=0 device=cuda:1 norm_scale=5.12 |
| live_nvidia_smi_repair_sweep_source_anchor | operator_repair_sweep | cuda:1 | 797727 | 750 | observed via nvidia-smi --query-compute-apps while process was running | completed | route=RepairSweepPreviewPassFound preview_pass_rows=18 diagnostic_only=1 |
| live_nvidia_smi_horizon_source_anchor | operator_horizon | cuda:1 | 801790 | 750 | observed via nvidia-smi --query-compute-apps while process was running | completed | route=S5-RoleBlindOperatorHorizonPass c3=4 c4=3 device=cuda:1 norm_scale=5.12 |
| live_nvidia_smi_kan_after_operator_pass | KAN_mapping | cuda:2 | 807211 | 750 | observed via nvidia-smi --query-compute-apps while process was running | completed | route=S6-KANCorrectedCarrierNoGo pass_rows=0 device=cuda:2 blocker=KANSourceChannelMismatchConfirmed |
| live_nvidia_smi_efficiency_native_fused | native_efficiency | cuda:0 | 817418 | 684 | observed via nvidia-smi --query-compute-apps while process was running | completed | route=S1-NativeOfficialRowsPresent fused_complete=84 blocker= |
| live_nvidia_smi_kan_after_efficiency_pass | KAN_mapping | cuda:2 | 818258 | 750 | observed via nvidia-smi --query-compute-apps while process was running | completed | route=S6-KANCorrectedCarrierNoGo pass_rows=0 device=cuda:2 blocker=KANSourceChannelMismatchConfirmed |
| live_nvidia_smi_kan_basis_linearized | KAN_mapping | cuda:2 | 821114 | 792 | observed via nvidia-smi --query-compute-apps while process was running | completed | route=S6-KANCorrectedCarrierNoGo basis_linearized_w1_gain1 device=cuda:2 |
| live_nvidia_smi_kan_basis_gain_sweep | KAN_mapping | cuda:2 | 824256 | 792 | observed via nvidia-smi --query-compute-apps while process was running | completed | route=S6-KANCorrectedCarrierNoGo basis_linearized_w1 gains=1,4,16 device=cuda:2 |

| operator_id | norm_scale | loss_adapter_name | attempt | device | source_func_h100 | source_func_h800 | source_func_h1600 | source_loss_h1600 | repair_sweep_preview_pass | diagnostic_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LIO2_SplitCoherentControlNull | 5.12 | Delta-LossCEAdapter | slow_source_anchor_w0p02_800x0p006 | cuda:1 | 1.2089239452955964 | 1.0872773203295893 | 1.0654161242061635 | 0.043880729470402 | 1 | 1 |
| LIO2_SplitCoherentControlNull | 5.12 | Delta-LossCEAdapter | slow_source_anchor_w0p10_800x0p006 | cuda:1 | 1.246760780725026 | 1.1466855463957568 | 1.1197853126142523 | 0.17565765231847763 | 1 | 1 |
| LIO2_SplitCoherentControlNull | 5.12 | Delta-LossCEAdapter | low_lr_source_anchor_w0p10_lr0p02 | cuda:1 | 1.277092053360855 | 1.211723026741908 | 1.1719354999141962 | 0.3243587911128998 | 1 | 1 |
| LIO2_SplitCoherentControlNull | 5.12 | Delta-MSEAdapter | slow_source_anchor_w0p02_800x0p006 | cuda:1 | 1.263144115434007 | 0.3710634651990141 | 0.13290099127006005 | 1.2056831941008568 | 1 | 1 |
| LIO2_SplitCoherentControlNull | 5.12 | Delta-MSEAdapter | slow_source_anchor_w0p10_800x0p006 | cuda:1 | 1.2700169806810835 | 0.4079159031919914 | 0.17732963113035072 | 1.5904121026396751 | 1 | 1 |
| LIO2_SplitCoherentControlNull | 5.12 | Delta-MSEAdapter | low_lr_source_anchor_w0p10_lr0p02 | cuda:1 | 1.335854640216752 | 1.1949821589901224 | 0.7403311446790456 | 10.330980256199837 | 1 | 1 |
| LIO2_SplitCoherentControlNull | 5.12 | Delta-RankingAdapter | slow_source_anchor_w0p02_800x0p006 | cuda:1 | 1.3985676805963203 | 1.140700016270538 | 0.830539835882905 | 0.04018136896775104 | 1 | 1 |
| LIO2_SplitCoherentControlNull | 5.12 | Delta-RankingAdapter | slow_source_anchor_w0p10_800x0p006 | cuda:1 | 1.393188563755444 | 1.1016474018264704 | 0.7419777496042497 | 0.14124297356465831 | 1 | 1 |
| LIO2_SplitCoherentControlNull | 5.12 | Delta-RankingAdapter | low_lr_source_anchor_w0p10_lr0p02 | cuda:1 | 1.4194235181301638 | 1.3536286364554342 | 1.2944732064938969 | 0.16628136817598715 | 1 | 1 |
| LIO7_LowNDSControlNull | 5.12 | Delta-LossCEAdapter | slow_source_anchor_w0p02_800x0p006 | cuda:1 | 1.16982460678379 | 1.0717281422440466 | 1.0531352136378496 | 0.038239460438489914 | 1 | 1 |
| LIO7_LowNDSControlNull | 5.12 | Delta-LossCEAdapter | slow_source_anchor_w0p10_800x0p006 | cuda:1 | 1.2477747873127552 | 1.1549497557740114 | 1.1364058593656747 | 0.15758934617042542 | 1 | 1 |
| LIO7_LowNDSControlNull | 5.12 | Delta-LossCEAdapter | low_lr_source_anchor_w0p10_lr0p02 | cuda:1 | 1.2745034699616244 | 1.2153031682213387 | 1.1784181513473948 | 0.3017714023590088 | 1 | 1 |
| LIO7_LowNDSControlNull | 5.12 | Delta-MSEAdapter | slow_source_anchor_w0p02_800x0p006 | cuda:1 | 1.2550102251895927 | 0.3655084176058957 | 0.12909605047852413 | 1.1917253732681274 | 1 | 1 |
| LIO7_LowNDSControlNull | 5.12 | Delta-MSEAdapter | slow_source_anchor_w0p10_800x0p006 | cuda:1 | 1.2627315869131315 | 0.4031363473731874 | 0.1738773170124318 | 1.5756179094314575 | 1 | 1 |
| LIO7_LowNDSControlNull | 5.12 | Delta-MSEAdapter | low_lr_source_anchor_w0p10_lr0p02 | cuda:1 | 1.3318083138693972 | 1.18820859323586 | 0.7341495601887755 | 10.291054219007492 | 1 | 1 |
| LIO7_LowNDSControlNull | 5.12 | Delta-RankingAdapter | slow_source_anchor_w0p02_800x0p006 | cuda:1 | 1.3985680497056499 | 1.1407001326094623 | 0.8305397928274267 | 0.04018042382085696 | 1 | 1 |
| LIO7_LowNDSControlNull | 5.12 | Delta-RankingAdapter | slow_source_anchor_w0p10_800x0p006 | cuda:1 | 1.393188688705464 | 1.1016697587147721 | 0.7419714294928289 | 0.14124323532450944 | 1 | 1 |
| LIO7_LowNDSControlNull | 5.12 | Delta-RankingAdapter | low_lr_source_anchor_w0p10_lr0p02 | cuda:1 | 1.4194237443348352 | 1.353628758535089 | 1.2944743913811683 | 0.16628133883932605 | 1 | 1 |



## 1. Code truth

- S0 route: `S0-CodeSemanticLayoutTruthGatePass`；S0_pass=1。
- role-blind static/runtime: static=1, semantic=1。
- corrected layout pass: 1。
| check | pass | metric | value | blocker |
| --- | --- | --- | --- | --- |
| required_source_files_present | 1 | exists | 23/23 |  |
| compileall_ok | 1 | py_compile | 0 |  |
| core_import_pass | 1 | import | 0 |  |
| import_error_count | 1 | import_errors | 0 |  |
| official_operator_role_blind_static_pass | 1 | semantic | 1 |  |
| adapter_renaming_pass | 1 | operator | 6/6 |  |
| operator_law_pass | 1 | operator | 6/6 |  |
| corrected_layout_tests | 1 | layout | 2/2 |  |
| promotion_semantics_split | 1 | promotion | 1 |  |
| efficiency_profiler_unit_tests | 1 | tests | 1 |  |
| csv_claimed_exists_but_zip_missing_count | 1 | zip_required_compare | 0 |  |
| self_contained_compileall | 1 | clean_unzip_compile | 0 |  |
| self_contained_import_check | 1 | clean_unzip_import | 0 |  |

| check | adapter_name_branch_count | official_core_loss_formula_branch_count | O10_diagnostic_only_pass | official_operator_role_blind_static_pass | blocker |
| --- | --- | --- | --- | --- | --- |
| official_operator_role_blind_static_pass | 0 | 0 | 1 | 1 |  |

| operator_id | adapter_renaming_output_cosine | adapter_renaming_output_rel_error | adapter_renaming_pass |
| --- | --- | --- | --- |
| LIO1_LowNDSGreen | 1.0 | 0.0 | 1 |
| LIO2_SplitCoherentControlNull | 1.0 | 0.0 | 1 |
| LIO3_KANLowBankSpectral | 1.0 | 0.0 | 1 |
| LIO5_DualMemoryRetained | 1.0 | 0.0 | 1 |
| LIO6_SourceLossBoundary | 0.9999997615814209 | 0.0 | 1 |
| LIO7_LowNDSControlNull | 0.9999997615814209 | 0.0 | 1 |

| operator_id | operator_class | operator_linearity_error | operator_homogeneity_error | operator_lipschitz_ratio | cotangent_permutation_equivariance_error | operator_law_pass |
| --- | --- | --- | --- | --- | --- | --- |
| LIO1_LowNDSGreen | nonlinear_norm_capped_operator | 0.3011358678340912 | 0.5 | 0.15987612810812873 | 0.0 | 1 |
| LIO2_SplitCoherentControlNull | nonlinear_norm_capped_operator | 0.30819573998451233 | 0.5 | 0.15818647076037032 | 4.2517658727092567e-08 | 1 |
| LIO3_KANLowBankSpectral | nonlinear_norm_capped_operator | 0.3076913356781006 | 0.5 | 0.15815001134469642 | 3.2716993558778995e-08 | 1 |
| LIO5_DualMemoryRetained | nonlinear_norm_capped_operator | 0.3085578382015228 | 0.5 | 0.15800294113904992 | 6.380044936804552e-08 | 1 |
| LIO6_SourceLossBoundary | nonlinear_norm_capped_operator | 0.30211636424064636 | 0.5 | 0.16017946620552773 | 1.6597327601175493e-07 | 1 |
| LIO7_LowNDSControlNull | nonlinear_norm_capped_operator | 0.3059522807598114 | 0.5 | 0.15885554278540923 | 4.730453539991686e-08 | 1 |

| carrier | legacy_layout_fit_cosine | legacy_layout_projection_residual | corrected_layout_fit_cosine | corrected_layout_projection_residual | legacy_vs_corrected_gap_detected | layout_unit_test_pass |
| --- | --- | --- | --- | --- | --- | --- |
| D-CHE | -0.07537172734737396 | 1.342240571975708 | 0.9999571442604065 | 0.009266252629458904 | 1 | 1 |
| D-FOU | 0.5290412902832031 | 0.8828994035720825 | 0.9996626377105713 | 0.02620095945894718 | 1 | 1 |

Analysis: S0 只确认代码/语义/layout 是否可进入实验；所有 task/loss adapter formula 只在 LossInterface 或实验矩阵内使用，official operator core 不读取 adapter 名称来生成方向。

## 2. Efficiency

- official_fused_kernel_complete_rows: 84; manual_upstream_vjp_rows: 0。
| carrier | profile_rows | native_E1_pass_rows | official_fused_kernel_complete_rows | manual_upstream_vjp_rows | best_operator_step_ratio | best_full_step_ratio | best_memory_ratio | decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-FOU | 36 | 35 | 36 | 0 | 0.8722564572434481 | 0.6763482970673955 | 0.9886977599953273 | OfficialNativeRowsPresent |  |
| D-CHE | 48 | 48 | 48 | 0 | 0.8564060077519949 | 0.6999745425407639 | 0.9891066265588038 | OfficialNativeRowsPresent |  |
| D-RBF | 12 | 12 | 0 | 0 | 1.1069202859038658 | 1.0009462169054537 | 1.0038174768863704 | OfficialNativeBlocked | arbitrary_cotangent_fused_backward_contract_missing |
| D-RAT | 12 | 12 | 0 | 0 | 1.2347245441920782 | 1.0567154032012036 | 1.0114524306591113 | OfficialNativeBlocked | arbitrary_cotangent_fused_backward_contract_missing |

| carrier | variant | cotangent_type | batch_size | forward_ratio_vs_mlp | vjp_ratio_vs_mlp | operator_step_ratio_vs_mlp | full_step_ratio_vs_mlp | official_fused_kernel_complete | native_status_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | MLP-same-param-tanh-reference | Delta-Gaussian | 128 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |  |
| MLP | MLP-same-param-tanh-reference | Delta-StableRandom | 128 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |  |
| MLP | MLP-same-param-tanh-reference | Delta-LossCEAdapter | 128 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |  |
| MLP | MLP-same-param-tanh-reference | Delta-MSEAdapter | 128 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |  |
| MLP | MLP-same-param-tanh-reference | Delta-Gaussian | 256 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |  |
| MLP | MLP-same-param-tanh-reference | Delta-StableRandom | 256 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |  |
| MLP | MLP-same-param-tanh-reference | Delta-LossCEAdapter | 256 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |  |
| MLP | MLP-same-param-tanh-reference | Delta-MSEAdapter | 256 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |  |
| MLP | MLP-same-param-tanh-reference | Delta-Gaussian | 512 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |  |
| MLP | MLP-same-param-tanh-reference | Delta-StableRandom | 512 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |  |
| MLP | MLP-same-param-tanh-reference | Delta-LossCEAdapter | 512 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |  |
| MLP | MLP-same-param-tanh-reference | Delta-MSEAdapter | 512 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |  |
| D-FOU | FOU22.13-R1-lowfreq-bandreadout-native-vjp | Delta-Gaussian | 128 | 3.1720538065955957 | 1.9566338205792537 | 1.2478676523116123 | 0.9467571066738245 | 1 |  |
| D-FOU | FOU22.13-R1-lowfreq-bandreadout-native-vjp | Delta-StableRandom | 128 | 1.2853980427144998 | 0.4930769725006425 | 0.9700596483817059 | 0.9780107615378665 | 1 |  |
| D-FOU | FOU22.13-R1-lowfreq-bandreadout-native-vjp | Delta-LossCEAdapter | 128 | 1.3161563421054934 | 0.5300023300363024 | 1.0783160612029676 | 0.987094524450881 | 1 |  |
| D-FOU | FOU22.13-R1-lowfreq-bandreadout-native-vjp | Delta-MSEAdapter | 128 | 1.3839606493787204 | 0.5143942011399462 | 1.0478736050388997 | 0.8565092178626933 | 1 |  |
| D-FOU | FOU22.13-R1-lowfreq-bandreadout-native-vjp | Delta-Gaussian | 256 | 5.533110613539135 | 2.5101172116521915 | 3.0675140684740496 | 0.6959448889428038 | 1 |  |
| D-FOU | FOU22.13-R1-lowfreq-bandreadout-native-vjp | Delta-StableRandom | 256 | 1.1504001398918569 | 0.5146954076766538 | 1.0349567852788597 | 0.7650074913011183 | 1 |  |
| D-FOU | FOU22.13-R1-lowfreq-bandreadout-native-vjp | Delta-LossCEAdapter | 256 | 1.2083844806979964 | 0.5305081827870426 | 1.0988530818197118 | 0.7645034661586735 | 1 |  |
| D-FOU | FOU22.13-R1-lowfreq-bandreadout-native-vjp | Delta-MSEAdapter | 256 | 1.1264164572391686 | 0.5196228282702966 | 1.1063294815804032 | 0.7588808460983464 | 1 |  |
| D-FOU | FOU22.13-R1-lowfreq-bandreadout-native-vjp | Delta-Gaussian | 512 | 5.029113027935654 | 2.5890905609715515 | 1.3425968277377511 | 0.6763482970673955 | 1 |  |
| D-FOU | FOU22.13-R1-lowfreq-bandreadout-native-vjp | Delta-StableRandom | 512 | 1.1954440634766514 | 0.5256035454898165 | 1.0441701456474086 | 0.7752289456657849 | 1 |  |
| D-FOU | FOU22.13-R1-lowfreq-bandreadout-native-vjp | Delta-LossCEAdapter | 512 | 1.2231175812727468 | 0.20204549219548995 | 0.8776365572967781 | 0.768960505700449 | 1 |  |
| D-FOU | FOU22.13-R1-lowfreq-bandreadout-native-vjp | Delta-MSEAdapter | 512 | 1.184196534471241 | 0.527848808619071 | 1.1092514642341076 | 0.7676957153900393 | 1 |  |
| D-FOU | FOU22.13-R2-tablelookup-bandreadout-native-operator-step | Delta-Gaussian | 128 | 1.204235498398106 | 0.3791832041271399 | 0.9238724093829747 | 0.9499658478813445 | 1 |  |
| D-FOU | FOU22.13-R2-tablelookup-bandreadout-native-operator-step | Delta-StableRandom | 128 | 1.3381238445970218 | 0.5135491119492661 | 1.008285654878759 | 0.9845589123097617 | 1 |  |
| D-FOU | FOU22.13-R2-tablelookup-bandreadout-native-operator-step | Delta-LossCEAdapter | 128 | 1.2910351382001475 | 0.5187096579115404 | 1.072228500052142 | 0.9860978371969589 | 1 |  |
| D-FOU | FOU22.13-R2-tablelookup-bandreadout-native-operator-step | Delta-MSEAdapter | 128 | 1.2989210125073753 | 0.505123962443153 | 1.0513794310628906 | 0.8484714275083363 | 1 |  |
| D-FOU | FOU22.13-R2-tablelookup-bandreadout-native-operator-step | Delta-Gaussian | 256 | 1.2252639887306342 | 0.5184434814744907 | 1.1002613312433753 | 0.7722060269629256 | 1 |  |
| D-FOU | FOU22.13-R2-tablelookup-bandreadout-native-operator-step | Delta-StableRandom | 256 | 1.2549091932447534 | 0.5534841476474328 | 1.1007420877870648 | 0.7525116104583507 | 1 |  |
| D-FOU | FOU22.13-R2-tablelookup-bandreadout-native-operator-step | Delta-LossCEAdapter | 256 | 1.2117001788696409 | 0.5186101151105459 | 1.113191356100269 | 0.7600497036643741 | 1 |  |
| D-FOU | FOU22.13-R2-tablelookup-bandreadout-native-operator-step | Delta-MSEAdapter | 256 | 1.1243171660321138 | 0.5360121885965754 | 1.1062551344380174 | 0.7581083368714169 | 1 |  |
| D-FOU | FOU22.13-R2-tablelookup-bandreadout-native-operator-step | Delta-Gaussian | 512 | 1.114717430160866 | 0.5315988977049195 | 1.1122660245190101 | 0.7689172993647672 | 1 |  |
| D-FOU | FOU22.13-R2-tablelookup-bandreadout-native-operator-step | Delta-StableRandom | 512 | 1.2147014797058793 | 0.5426769572169827 | 1.1081163640452796 | 0.7683918295532948 | 1 |  |
| D-FOU | FOU22.13-R2-tablelookup-bandreadout-native-operator-step | Delta-LossCEAdapter | 512 | 1.2349437947242783 | 0.2047840188606141 | 0.8722564572434481 | 0.7670801669575298 | 1 |  |
| D-FOU | FOU22.13-R2-tablelookup-bandreadout-native-operator-step | Delta-MSEAdapter | 512 | 1.238156262354978 | 0.5248334748982967 | 1.1003879700525523 | 0.7688564100851216 | 1 |  |
| D-FOU | FOU22.13-R3-lowfreq-source-state-fused-commit | Delta-Gaussian | 128 | 1.1965549291198505 | 0.3719018482121745 | 0.9215875119559308 | 0.9539658316741397 | 1 |  |
| D-FOU | FOU22.13-R3-lowfreq-source-state-fused-commit | Delta-StableRandom | 128 | 1.211869876435766 | 0.4811801489230171 | 1.0031353126945008 | 0.9511651651042619 | 1 |  |
| D-FOU | FOU22.13-R3-lowfreq-source-state-fused-commit | Delta-LossCEAdapter | 128 | 1.2238052341429142 | 0.5394140759093812 | 1.1068776249584276 | 0.9954415134232348 | 1 |  |
| D-FOU | FOU22.13-R3-lowfreq-source-state-fused-commit | Delta-MSEAdapter | 128 | 1.2326190712645826 | 0.45809229585916755 | 1.060059316060935 | 0.8496265335060743 | 1 |  |

_仅显示前 40 / 120 rows；完整 CSV 见 artifact。_

Analysis: D-FOU/D-CHE 已从 label-only fused backward 修复为 generic output-cotangent fused VJP；official_fused rows 只在实际 fused path 与 native-vs-autograd gradcheck 通过后计数。D-RBF/D-RAT 仍保持 blocked，不借 D-FOU/D-CHE 的成功。

## 3. Operator

- S2/S3/S4 pass rows: 5 / 6 / 3。
| operator_id | operator_family | gain_positive_fraction | control_projection_after | NDS_reduction | operator_linearity_error | operator_homogeneity_error | S2_operator_atom_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LIO1_LowNDSGreen | LowNDSGreen | 1.0 | 0.08519414812326431 | 0.0691955039037072 | 0.32662373781204224 | 0.5 | 1 |  |
| LIO2_SplitCoherentControlNull | SplitCoherentControlNull | 1.0 | 1.8098113230280433e-08 | 0.08452886904114111 | 0.3264230787754059 | 0.5 | 1 |  |
| LIO3_KANLowBankSpectral | KANLowBankSpectral | 1.0 | 2.689061595617659e-08 | 0.11294957755642081 | 0.32539060711860657 | 0.5 | 0 | operator_lipschitz |
| LIO5_DualMemoryRetained | DualMemoryRetained | 1.0 | 2.7083292053475816e-08 | 0.09948680095311828 | 0.3258828818798065 | 0.5 | 1 |  |
| LIO6_SourceLossBoundary | SourceLossBoundary | 1.0 | 1.0098347047460265e-08 | 0.07180865659601832 | 0.33072376251220703 | 0.5 | 1 |  |
| LIO7_LowNDSControlNull | LowNDSControlNull | 1.0 | 4.642412676503227e-08 | 0.07183369854726641 | 0.3269997835159302 | 0.5 | 1 |  |

| solver_level | block_role | projection_residual_Gf | ActuationR2 | B2_transfer_gain | function_displacement_cos_with_target | S4_operator_metric_commit_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S4.1-v2213-readout-all-train | readout_only | 0.15092749574565972 | 0.977220892906189 | 0.858426108956337 | 0.9888907074928284 | 1 |  |
| S4.2-v2213-readout-damped-lowcurv | readout_only | 0.19398540117164811 | 0.9623696804046631 | 0.8058266341686249 | 0.9819192290306091 | 1 |  |
| S4.3-v2213-hidden-readout-audit | hidden_readout | 0.15092749574565972 | 0.977220892906189 | 0.858426108956337 | 0.9888907074928284 | 1 |  |

Analysis: operator rows 是 true `cotangent -> Delta_f` role-blind atoms；没有把 O10 diagnostic consensus 算作 official success。

## 4. Adapter robustness

- C3/C4 adapter pass count: 4 / 3。
- C3 list: `Delta-LossCEAdapter;Delta-MSEAdapter;Delta-PreferenceAdapter-smoke;Delta-RankingAdapter`；C4 list: `Delta-LossCEAdapter;Delta-PreferenceAdapter-smoke;Delta-RankingAdapter`。
- Latest official horizon used norm_scale=5.12; route=`S5-RoleBlindOperatorHorizonPass`; blocker=``。
| loss_adapter_name | attempt | device | source_func_h100 | source_func_h800 | source_func_h3200 | source_func_h4800 | source_loss_h3200 | source_loss_h4800 | C3_source_formation_pass | C4_terminal_retention_pass | TargetRetentionOnly_NotTaskUseful | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Delta-LossCEAdapter | low_lr_source_anchor_w0p10_lr0p02 | cuda:1 | 1.2770918867608008 | 1.2117231771887074 | 1.1470507108111558 | 1.1372898534141154 | 0.23886382952332497 | 0.21234815567731857 | 1 | 1 | 0 |  |
| Delta-MSEAdapter | low_lr_source_anchor_w0p10_lr0p02 | cuda:1 | 1.335854385644918 | 1.2159116645014505 | 0.4277930930894471 | 0.2043746988065751 | 5.128191985189915 | 2.0548082180321217 | 1 | 0 | 0 |  |
| Delta-RankingAdapter | low_lr_source_anchor_w0p10_lr0p02 | cuda:1 | 1.4194235181301638 | 1.3536286364554342 | 1.1096996867744093 | 0.8892649843069008 | 0.12147705705137923 | 0.1188801503740251 | 1 | 1 | 0 |  |
| Delta-GenericSourceTarget | low_lr_source_anchor_w0p10_lr0p02 | cuda:1 | 0.0409221027587332 | -9.562340223014832 | -39.81906046174876 | -56.589669316634065 | -6432700.21875 | -6990317.0 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-PreferenceAdapter-smoke | low_lr_source_anchor_w0p10_lr0p02 | cuda:1 | 1.4206091228350313 | 1.3385743599307864 | 0.9029643084547543 | 0.7735650690132883 | 0.3390205502510071 | 0.33896833658218384 | 1 | 1 | 0 |  |
| Delta-GenericSourceTarget-normalized-holdout | low_lr_source_anchor_w0p10_lr0p02 | cuda:1 | -0.08838481949771637 | -9.19578147211838 | -35.996490591122885 | -49.684763330342875 | -233430.876953125 | -246272.796875 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-StableRandom-control | low_lr_source_anchor_w0p10_lr0p02 | cuda:1 | -0.7100163782170664 | -13.447374145757788 | -44.96460503722377 | -58.85512483862556 | -203538.42578125 | -213565.60546875 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| Delta-RandomMatched-control | low_lr_source_anchor_w0p10_lr0p02 | cuda:1 | -0.6668589946013934 | -12.678549134634533 | -37.98479506563762 | -47.228037116592695 | -152604.61328125 | -171296.953125 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |

Analysis: source-anchor retention 后，正式 GPU horizon 从 NoGo 推进到 `S5-RoleBlindOperatorHorizonPass`：CE/MSE/Ranking/Preference-smoke 均 C3，CE/Ranking/Preference-smoke 均 C4；MSE source_loss 到 h6400 仍为正但 C4 终端保留阈值未过。StableRandom/RandomMatched 未通过，说明 source gate 未被 random controls 污染。

## 5. KAN carrier

- KAN route: `S6-KANCorrectedCarrierNoGo`；DFOU source=0, DCHE source=0, DCHE efficiency=1。
| carrier | attempt | device | basis_estimate_fit_cosine | basis_commit_projection_residual | KAN_source_func_h3200 | KAN_source_func_h4800 | KAN_source_loss_h3200 | KAN_source_loss_h4800 | carrier_specific_efficiency_pass | KAN_source_channel_decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-FOU | K15_DFOU_corrected_K13_revalidation_readout_corrected | cuda:2 | 0.9675318598747253 | 0.28125596046447754 | 0.01113634230569005 | 0.006274208426475525 | 0.005247701425105333 | 0.0010294776875525713 | 1 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-FOU | K15_DFOU_corrected_K13_revalidation_basis_linearized_w1_gain1 | cuda:2 | 0.0732899084687233 | 0.9987722635269165 | -0.0026349425315856934 | -0.02008315920829773 | 0.008267402648925781 | -0.2184305191040039 | 1 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-FOU | K15_DFOU_corrected_K13_revalidation_basis_linearized_w1_gain4 | cuda:2 | 0.06330905109643936 | 0.9989525675773621 | 0.007681131362915039 | 0.007912158966064453 | 0.08522796630859375 | 0.11306953430175781 | 1 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-FOU | K15_DFOU_corrected_K13_revalidation_basis_linearized_w1_gain16 | cuda:2 | 0.06695825606584549 | 0.9988787770271301 | -0.002820611000061035 | -0.00633925199508667 | -0.05929088592529297 | -0.1266183853149414 | 1 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-CHE | K16_DCHE_corrected_source_loss_repair_slow_state_readout_corrected | cuda:2 | 0.9376921653747559 | 0.3690424859523773 | 0.027288314420729876 | 0.024146966636180878 | 0.00807100161910057 | 0.0026639638817869127 | 1 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-CHE | K16_DCHE_corrected_source_loss_repair_slow_state_basis_linearized_w1_gain1 | cuda:2 | 0.08513602614402771 | 0.997999370098114 | 0.013971269130706787 | 0.007326185703277588 | 0.24326419830322266 | 0.08289194107055664 | 1 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-CHE | K16_DCHE_corrected_source_loss_repair_slow_state_basis_linearized_w1_gain4 | cuda:2 | 0.0805286318063736 | 0.9981114268302917 | 0.006399273872375488 | 0.0039778947830200195 | 0.021241188049316406 | 0.013300895690917969 | 1 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |
| D-CHE | K16_DCHE_corrected_source_loss_repair_slow_state_basis_linearized_w1_gain16 | cuda:2 | 0.0818919837474823 | 0.9980655312538147 | -0.026010572910308838 | -0.025905728340148926 | -0.44808006286621094 | -0.4704551696777344 | 1 | KANSourceChannelMismatchConfirmed | KANSourceChannelMismatchConfirmed |

| carrier | commit_channel | readout_channel_energy | basis_channel_energy | basis_to_readout_energy_ratio | K17_basis_channel_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- |
| D-FOU | corrected_readout_layout_low_degree_source_replay | 1.0 | 0.0 | 0.0 | 0 | KANReadoutSourceOpened_BasisChannelBlocked |
| D-FOU | basis_linearized_w1 | 0.0 | 1.0 | 1000000000000.0 | 0 | KANBasisChannelSourceNotRetained |
| D-FOU | basis_linearized_w1 | 0.0 | 1.0 | 1000000000000.0 | 0 | KANBasisChannelSourceNotRetained |
| D-FOU | basis_linearized_w1 | 0.0 | 1.0 | 1000000000000.0 | 0 | KANBasisChannelSourceNotRetained |
| D-CHE | corrected_readout_layout_low_degree_source_replay | 1.0 | 0.0 | 0.0 | 0 | KANReadoutSourceOpened_BasisChannelBlocked |
| D-CHE | basis_linearized_w1 | 0.0 | 1.0 | 1000000000000.0 | 0 | KANBasisChannelSourceNotRetained |
| D-CHE | basis_linearized_w1 | 0.0 | 1.0 | 1000000000000.0 | 0 | KANBasisChannelSourceNotRetained |
| D-CHE | basis_linearized_w1 | 0.0 | 1.0 | 1000000000000.0 | 0 | KANBasisChannelSourceNotRetained |

Analysis: corrected layout 与 basis-linearized gain sweep 均为 operator/efficiency pass 后的 GPU 重跑。Readout-only rows 有弱 positive source 但 basis_channel_energy=0；basis_linearized_w1 rows 有 basis_channel_energy=1，但 source_func 仍远低于 MLP same-adapter baseline，gain16 还出现 source_loss collapse。因此不能写成 KAN carrier success，当前 no-go 精确定位为 KAN basis-channel source mismatch。

## 6. Task-level value

- task route: `TaskReadbackCompleted`；rows=45; KAN_FU_accuracy_ge_MLP_rows=1。
| dataset | seed | variant | final_train_loss | final_test_loss_readback | final_train_accuracy | final_test_accuracy_readback | NLL_delta_vs_MLP | AUC_loss_time_ratio_vs_best_control | ECE_delta | Brier_delta | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST | 0 | MLP+AdamW | 0.3228437267243862 | 0.48396050930023193 | 0.9404296875 | 0.876953125 | 0.0 | 1.1079551511288837 | 0.11519027635222301 | 0.20687860250473022 |  |
| MNIST | 0 | MLP+SGD | 0.29138699918985367 | 0.44899167865514755 | 0.9345703125 | 0.875 | -0.03496883064508438 | 1.0 | 0.062370297499001026 | 0.19482028484344482 |  |
| MNIST | 0 | MLP+FU+AdamW | 0.24560638330876827 | 0.5080105662345886 | 0.9521484375 | 0.865234375 | 0.02405005693435669 | 0.842887239278452 | 0.06887405342422426 | 0.21050582826137543 | projection_residual_Gf_gate |
| MNIST | 0 | KAN+AdamW | 1.6863840967416763 | 1.7318183183670044 | 0.6845703125 | 0.646484375 | 1.2478578090667725 | 5.787437673713473 | 0.44601667736424133 | 0.7514122128486633 |  |
| MNIST | 0 | KAN+FU+AdamW | 0.8116625398397446 | 0.955645427107811 | 0.8349609375 | 0.771484375 | 0.47168491780757904 | 2.785513911383893 | 0.25975963214295916 | 0.42003828287124634 |  |
| MNIST | 1 | MLP+AdamW | 0.34627945721149445 | 0.4289395809173584 | 0.927734375 | 0.875 | 0.0 | 1.1109188356860793 | 0.09817234007641673 | 0.18600940704345703 |  |
| MNIST | 1 | MLP+SGD | 0.31170545145869255 | 0.40996307134628296 | 0.9248046875 | 0.8828125 | -0.01897650957107544 | 1.0 | 0.07076766295358539 | 0.1794867366552353 |  |
| MNIST | 1 | MLP+FU+AdamW | 0.30337703227996826 | 0.4446879103779793 | 0.9365234375 | 0.86328125 | 0.01574832946062088 | 0.9732811244084771 | 0.07519517262699082 | 0.19568383693695068 | projection_residual_Gf_gate |
| MNIST | 1 | KAN+AdamW | 1.719664826989174 | 1.7183066606521606 | 0.6552734375 | 0.646484375 | 1.2893670797348022 | 5.516954608723181 | 0.44363515014993027 | 0.7457297444343567 |  |
| MNIST | 1 | KAN+FU+AdamW | 0.8746089786291122 | 0.9287257790565491 | 0.81640625 | 0.80078125 | 0.4997861981391907 | 2.8058828439996537 | 0.28889493420138024 | 0.406839519739151 |  |
| MNIST | 2 | MLP+AdamW | 0.31412906385958195 | 0.4417181611061096 | 0.9384765625 | 0.88671875 | 0.0 | 1.1554260425829237 | 0.10716305021196604 | 0.18992231786251068 |  |
| MNIST | 2 | MLP+SGD | 0.2718729302287102 | 0.43239589780569077 | 0.939453125 | 0.890625 | -0.009322263300418854 | 1.0 | 0.05882811170886271 | 0.1889062374830246 |  |
| MNIST | 2 | MLP+FU+AdamW | 0.23348661698400974 | 0.5016113817691803 | 0.958984375 | 0.861328125 | 0.05989322066307068 | 0.8588078878893594 | 0.04618405341170728 | 0.2154110074043274 | projection_residual_Gf_gate |
| MNIST | 2 | KAN+AdamW | 1.696116328239441 | 1.750687301158905 | 0.671875 | 0.609375 | 1.3089691400527954 | 6.238636287962179 | 0.40973116279928945 | 0.7533087730407715 |  |
| MNIST | 2 | KAN+FU+AdamW | 0.8225182965397835 | 0.9653562158346176 | 0.8271484375 | 0.7578125 | 0.523638054728508 | 3.0253776859941475 | 0.2468163312296383 | 0.4260888695716858 |  |
| FashionMNIST | 0 | MLP+AdamW | 0.5851891934871674 | 0.7361513376235962 | 0.8095703125 | 0.740234375 | 0.0 | 1.181133270574365 | 0.07072478550253436 | 0.35941997170448303 |  |
| FashionMNIST | 0 | MLP+SGD | 0.49544721841812134 | 0.6617356240749359 | 0.818359375 | 0.755859375 | -0.07441571354866028 | 1.0 | 0.055416835471987724 | 0.3341016471385956 |  |
| FashionMNIST | 0 | MLP+FU+AdamW | 0.476649709045887 | 0.7008531093597412 | 0.8525390625 | 0.76171875 | -0.03529822826385498 | 0.9620595117432457 | 0.053764852142194286 | 0.33908650279045105 | projection_residual_Gf_gate |
| FashionMNIST | 0 | KAN+AdamW | 1.4383433014154434 | 1.4974991381168365 | 0.6572265625 | 0.611328125 | 0.7613478004932404 | 2.903121155887864 | 0.3514300094975624 | 0.6848530769348145 |  |
| FashionMNIST | 0 | KAN+FU+AdamW | 0.6452863663434982 | 0.8885620683431625 | 0.8359375 | 0.73046875 | 0.15241073071956635 | 1.3024321105359877 | 0.13624855561647564 | 0.404183954000473 |  |
| FashionMNIST | 1 | MLP+AdamW | 0.5805817022919655 | 0.7689249366521835 | 0.8095703125 | 0.712890625 | 0.0 | 1.0917999410332095 | 0.07031447219196707 | 0.39572086930274963 |  |
| FashionMNIST | 1 | MLP+SGD | 0.5317656472325325 | 0.7284308969974518 | 0.8046875 | 0.720703125 | -0.04049403965473175 | 1.0 | 0.06274457031395286 | 0.3813680410385132 |  |
| FashionMNIST | 1 | MLP+FU+AdamW | 0.45321017503738403 | 0.7261873036623001 | 0.859375 | 0.740234375 | -0.04273763298988342 | 0.8522742629126672 | 0.03033248847350478 | 0.35524117946624756 | projection_residual_Gf_gate |
| FashionMNIST | 1 | KAN+AdamW | 1.4562621265649796 | 1.5035547316074371 | 0.697265625 | 0.60546875 | 0.7346297949552536 | 2.738541186599404 | 0.35148365882923827 | 0.693809449672699 |  |
| FashionMNIST | 1 | KAN+FU+AdamW | 0.6849484071135521 | 0.9780692905187607 | 0.8212890625 | 0.689453125 | 0.20914435386657715 | 1.2880644146123927 | 0.11162556891213171 | 0.4490724503993988 |  |
| FashionMNIST | 2 | MLP+AdamW | 0.5849178098142147 | 0.672571673989296 | 0.8056640625 | 0.75 | 0.0 | 1.1206268126013201 | 0.07781872031046078 | 0.3467608094215393 |  |
| FashionMNIST | 2 | MLP+SGD | 0.5219559296965599 | 0.6117997616529465 | 0.8076171875 | 0.76171875 | -0.06077191233634949 | 1.0 | 0.040611756965518 | 0.32730162143707275 |  |
| FashionMNIST | 2 | MLP+FU+AdamW | 0.4839170463383198 | 0.6481611877679825 | 0.8408203125 | 0.75390625 | -0.024410486221313477 | 0.9271224231893407 | 0.030517953680828214 | 0.33485156297683716 | projection_residual_Gf_gate |
| FashionMNIST | 2 | KAN+AdamW | 1.4657369703054428 | 1.484780341386795 | 0.671875 | 0.65234375 | 0.8122086673974991 | 2.808162312012725 | 0.3991692241106648 | 0.6833850145339966 |  |
| FashionMNIST | 2 | KAN+FU+AdamW | 0.6360604837536812 | 0.7797758728265762 | 0.8427734375 | 0.75390625 | 0.10720419883728027 | 1.2186095560279508 | 0.13614900090033188 | 0.36393871903419495 |  |
| KMNIST | 0 | MLP+AdamW | 0.4192407615482807 | 1.2811090648174286 | 0.91015625 | 0.607421875 | 0.0 | 1.1108400341514526 | 0.05603179018362425 | 0.5351961851119995 |  |
| KMNIST | 0 | MLP+SGD | 0.3774087615311146 | 1.3407984673976898 | 0.90234375 | 0.603515625 | 0.05968940258026123 | 1.0 | 0.061424801708199084 | 0.5356713533401489 |  |
| KMNIST | 0 | MLP+FU+AdamW | 0.3545861169695854 | 1.369480937719345 | 0.923828125 | 0.59375 | 0.0883718729019165 | 0.9395280478679412 | 0.042153201677137986 | 0.5664109587669373 | projection_residual_Gf_gate |
| KMNIST | 0 | KAN+AdamW | 1.6863786578178406 | 1.9508927762508392 | 0.685546875 | 0.40625 | 0.6697837114334106 | 4.468308183880917 | 0.2163307554437779 | 0.812343955039978 |  |
| KMNIST | 0 | KAN+FU+AdamW | 0.76046422123909 | 1.5294268131256104 | 0.8193359375 | 0.474609375 | 0.24831774830818176 | 2.0149617569924785 | 0.08985863046837039 | 0.650860071182251 |  |
| KMNIST | 1 | MLP+AdamW | 0.44169826805591583 | 1.270053744316101 | 0.9052734375 | 0.619140625 | 0.0 | 1.0945223377268243 | 0.03730097730294801 | 0.5250023603439331 |  |
| KMNIST | 1 | MLP+SGD | 0.40355345234274864 | 1.3845194280147552 | 0.90234375 | 0.611328125 | 0.11446568369865417 | 1.0 | 0.07277532115404028 | 0.5488320589065552 |  |
| KMNIST | 1 | MLP+FU+AdamW | 0.39231767505407333 | 1.3989314436912537 | 0.9228515625 | 0.59765625 | 0.1288776993751526 | 0.9721578957547055 | 0.05782304331660271 | 0.5655450820922852 | projection_residual_Gf_gate |
| KMNIST | 1 | KAN+AdamW | 1.7187798917293549 | 1.978567659854889 | 0.673828125 | 0.453125 | 0.7085139155387878 | 4.259113338645284 | 0.2671792677429039 | 0.8195012807846069 |  |
| KMNIST | 1 | KAN+FU+AdamW | 0.8533529415726662 | 1.511796772480011 | 0.791015625 | 0.51953125 | 0.2417430281639099 | 2.1145970542903223 | 0.09904682825435884 | 0.6390188336372375 |  |
| KMNIST | 2 | MLP+AdamW | 0.4084533080458641 | 1.2329094260931015 | 0.9140625 | 0.619140625 | 0.0 | 1.1247848494099235 | 0.05420608620624989 | 0.509240984916687 |  |
| KMNIST | 2 | MLP+SGD | 0.3631390556693077 | 1.2540477812290192 | 0.90625 | 0.609375 | 0.021138355135917664 | 1.0 | 0.09132453246274963 | 0.5205239057540894 |  |
| KMNIST | 2 | MLP+FU+AdamW | 0.3565902151167393 | 1.3813719153404236 | 0.927734375 | 0.595703125 | 0.14846248924732208 | 0.981966025272335 | 0.07256228951155208 | 0.5541315078735352 | projection_residual_Gf_gate |
| KMNIST | 2 | KAN+AdamW | 1.6927913278341293 | 1.9674163162708282 | 0.6865234375 | 0.447265625 | 0.7345068901777267 | 4.661551274660108 | 0.2523387508117594 | 0.8133058547973633 |  |
| KMNIST | 2 | KAN+FU+AdamW | 0.8086192011833191 | 1.517326831817627 | 0.8125 | 0.51171875 | 0.28441740572452545 | 2.226748097069701 | 0.07543375110253692 | 0.6369938254356384 |  |

Analysis: task metrics 是 readback/gate，不参与 direction。由于本轮 official operator/KAN/fused gates 未全部成立，task improvement 即便出现也不能写成 FU mechanism proof。

## 7. Execution and artifacts

| timestamp | command | status | note |
| --- | --- | --- | --- |
| 2026-06-08 19:43:21 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_s0_truth_gate.py --check all --source-root /home/chengshun.wang/DG-LCA --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 | blocked | route=R0-CodeTruthFailed S0=0 blocker=dgkan/fu/operator_atoms_v22_13.py:if not renaming["adapter;clean_unzip_import_failed |
| 2026-06-08 19:48:12 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_s0_truth_gate.py --check all --source-root /home/chengshun.wang/DG-LCA --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 | completed | route=S0-CodeSemanticLayoutTruthGatePass S0=1 blocker= |
| 2026-06-08 19:48:50 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_semantic_tests.py --seed 2213 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 | completed | route=OperatorSemanticTestsPass semantic_pass=1 |
| 2026-06-08 19:49:09 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_construct.py --seed 2213 --norm-scale 0.16 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 | completed | S2=S2-RoleBlindOperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricCommitPass blocker= |
| 2026-06-08 19:50:02 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_efficiency_native.py --device cuda:0 --batch-sizes 128,256,512 --hidden 64 --repeats 3 --warmup 1 --seed 2213 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 | blocked | route=R3-KernelNativeEfficiencyBlocked fused_complete=0 blocker=arbitrary_cotangent_fused_backward_contract_missing |
| 2026-06-08 19:50:19 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_task_eval.py --device cuda:3 --datasets MNIST,FashionMNIST,KMNIST --seeds 0,1,2 --train-size 1024 --test-size 512 --steps 80 --batch-size 128 --hidden 64 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 | completed | route=TaskReadbackPartialOrBlocked rows=45 blocker=MNIST:0:MLP+FU+AdamW:Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cpu!;MNIST:1:MLP+FU+AdamW:Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cpu!;MNIST:2:MLP+FU+AdamW:Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cpu!;FashionMNIST:0:MLP+FU+AdamW:Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cpu!;FashionMNIST:1:MLP+FU+AdamW:Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cpu!;FashionMNIST:2:MLP+FU+AdamW:Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cpu!;KMNIST:0:MLP+FU+AdamW:Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cpu!;KMNIST:1:MLP+FU+AdamW:Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cpu!;KMNIST:2:MLP+FU+AdamW:Expected all tensors to be on the same device, but found at least two devices, cuda:3 and cpu! |
| 2026-06-08 19:52:27 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_task_eval.py --device cuda:3 --datasets MNIST,FashionMNIST,KMNIST --seeds 0,1,2 --train-size 1024 --test-size 512 --steps 80 --batch-size 128 --hidden 64 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 | completed | route=TaskReadbackCompleted rows=45 blocker= |
| 2026-06-08 19:55:33 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_horizon.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 | blocked | route=S5-RoleBlindOperatorHorizonNoGo c3=1 c4=0 blocker=source_func_or_source_loss_or_control_gate_failed |
| 2026-06-08 19:58:16 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_construct.py --seed 2213 --norm-scale 0.16 --prefer-attempt source_loss_boundary_only --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 | completed | S2=S2-RoleBlindOperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricCommitPass blocker= |
| 2026-06-08 20:04:13 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_horizon.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 | blocked | route=S5-RoleBlindOperatorHorizonNoGo c3=1 c4=0 blocker=source_func_or_source_loss_or_control_gate_failed |
| 2026-06-08 20:12:14 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_horizon.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 | blocked | route=S5-RoleBlindOperatorHorizonNoGo c3=0 c4=0 blocker=source_func_or_source_loss_or_control_gate_failed |
| 2026-06-08 20:13:18 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_construct.py --seed 2213 --norm-scale 0.16 --prefer-attempt split_control_null_only --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 | completed | S2=S2-RoleBlindOperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricCommitPass blocker= |
| 2026-06-08 20:19:06 +0800 | kill 745263  # aborted CPU run_v22_13_operator_horizon.py after user required GPU execution | blocked | CPU operator horizon aborted; its partial/fixed CPU artifacts are diagnostic only and must not count toward final gate |
| 2026-06-08 20:47:44 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_horizon.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1 | blocked | route=S5-RoleBlindOperatorHorizonNoGo c3=0 c4=0 device=cuda:1 blocker=source_func_or_source_loss_or_control_gate_failed |
| 2026-06-08 20:50:50 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_kan_corrected_mapping.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:2 | completed | route=R7-DFOUCorrectedSourceOpened_DCHEBlocked pass_rows=0 device=cuda:2 blocker=KANEfficiencyContractBlocked |
| 2026-06-08 20:55:11 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_finalize.py --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 | completed | route=R8-DCHESourceExists_EfficiencyOrSourceLossBlocked exploration=0 official=0 blocker=KANEfficiencyContractBlocked |
| 2026-06-08 20:57:54 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_finalize.py --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 | completed | route=R8-DCHESourceExists_EfficiencyOrSourceLossBlocked exploration=0 official=0 blocker=KANEfficiencyContractBlocked |
| 2026-06-08 21:12:02 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_repair_sweep.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1 --operator-ids LIO2_SplitCoherentControlNull --norm-scales 0.08,0.16,0.32,0.64,1.28,2.56 --adapters Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter --attempts role_blind_initial_commit_lr1,slow_source_state_fixed_800x0p006 --horizons 100,400,800 | completed | route=RepairSweepPreviewPassFound preview_pass_rows=2 best=LIO2_SplitCoherentControlNull;0.16;Delta-MSEAdapter;slow_source_state_fixed_800x0p006 device=cuda:1 diagnostic_only=1 |
| 2026-06-08 21:19:27 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_repair_sweep.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1 --operator-ids LIO1_LowNDSGreen,LIO3_KANLowBankSpectral,LIO6_SourceLossBoundary --norm-scales 0.32,0.64,1.28,2.56,5.12 --adapters Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter --attempts slow_source_state_fixed_800x0p006 --horizons 100,400,800 | completed | route=RepairSweepPreviewPassFound preview_pass_rows=7 best=LIO3_KANLowBankSpectral;0.32;Delta-MSEAdapter;slow_source_state_fixed_800x0p006 device=cuda:1 diagnostic_only=1 |
| 2026-06-08 21:26:52 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_repair_sweep.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1 --operator-ids LIO2_SplitCoherentControlNull,LIO3_KANLowBankSpectral,LIO6_SourceLossBoundary --norm-scales 3.84,5.12,6.40,7.68,10.24 --adapters Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter --attempts slow_source_state_fixed_800x0p006 --horizons 100,400,800 | completed | route=RepairSweepPreviewPassFound preview_pass_rows=19 best=LIO2_SplitCoherentControlNull;10.24;Delta-RankingAdapter;slow_source_state_fixed_800x0p006 device=cuda:1 diagnostic_only=1 |
| 2026-06-08 21:32:31 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_repair_sweep.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1 --operator-ids LIO7_LowNDSControlNull --norm-scales 4.50,5.00,5.12 --adapters Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter --attempts slow_source_state_fixed_800x0p006 --horizons 100,400,800 | completed | route=RepairSweepPreviewPassFound preview_pass_rows=2 best=LIO7_LowNDSControlNull;5.12;Delta-LossCEAdapter;slow_source_state_fixed_800x0p006 device=cuda:1 diagnostic_only=1 |
| 2026-06-08 21:37:17 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_repair_sweep.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1 --operator-ids LIO2_SplitCoherentControlNull,LIO7_LowNDSControlNull --norm-scales 4.50,5.00,5.12 --adapters Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter --attempts slow_source_state_fixed_800x0p006 --horizons 100,400,800 | completed | route=RepairSweepPreviewPassFound preview_pass_rows=10 best=LIO7_LowNDSControlNull;5.12;Delta-RankingAdapter;slow_source_state_fixed_800x0p006 device=cuda:1 diagnostic_only=1 |
| 2026-06-08 21:38:58 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_s0_truth_gate.py --check all --source-root /home/chengshun.wang/DG-LCA --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 | completed | route=S0-CodeSemanticLayoutTruthGatePass S0=1 blocker= |
| 2026-06-08 21:39:17 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_construct.py --seed 2213 --norm-scale 5.12 --prefer-attempt split_control_null_only --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 | completed | S2=S2-RoleBlindOperatorAtomProgress S3=S3-OperatorVariationalSolvePass S4=S4-OperatorMetricCommitPass blocker= |
| 2026-06-08 21:48:40 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_horizon.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1 --norm-scale 5.12 --attempts slow_source_state_fixed_800x0p006 | blocked | route=S5-RoleBlindOperatorHorizonNoGo c3=1 c4=0 device=cuda:1 norm_scale=5.12 attempts=slow_source_state_fixed_800x0p006 blocker=source_func_or_source_loss_or_control_gate_failed |
| 2026-06-08 21:50:44 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_finalize.py --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 | completed | route=R8-DCHESourceExists_EfficiencyOrSourceLossBlocked exploration=0 official=0 blocker=KANEfficiencyContractBlocked |
| 2026-06-08 22:03:36 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_repair_sweep.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1 --operator-ids LIO2_SplitCoherentControlNull,LIO7_LowNDSControlNull --norm-scales 5.12 --adapters Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter --attempts slow_source_anchor_w0p02_800x0p006,slow_source_anchor_w0p10_800x0p006,low_lr_source_anchor_w0p10_lr0p02 --horizons 100,400,800,1600 | completed | route=RepairSweepPreviewPassFound preview_pass_rows=18 best=LIO7_LowNDSControlNull;5.12;Delta-RankingAdapter;low_lr_source_anchor_w0p10_lr0p02 device=cuda:1 diagnostic_only=1 |
| 2026-06-08 22:15:19 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_operator_horizon.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:1 --norm-scale 5.12 --attempts low_lr_source_anchor_w0p10_lr0p02 | completed | route=S5-RoleBlindOperatorHorizonPass c3=4 c4=3 device=cuda:1 norm_scale=5.12 attempts=low_lr_source_anchor_w0p10_lr0p02 blocker= |
| 2026-06-08 22:19:34 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_kan_corrected_mapping.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:2 | completed | route=S6-KANCorrectedCarrierNoGo pass_rows=0 device=cuda:2 blocker=KANSourceChannelMismatchConfirmed |
| 2026-06-08 22:23:34 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_finalize.py --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 | completed | route=R9-TrueLossInterfaceOperatorExplorationSuccess_KANPending exploration=1 official=1 blocker=KANSourceChannelMismatchConfirmed;arbitrary_cotangent_fused_backward_contract_missing |
| 2026-06-08 22:24:57 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_finalize.py --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 | completed | route=R9-TrueLossInterfaceOperatorExplorationSuccess_KANPending exploration=1 official=1 blocker=KANSourceChannelMismatchConfirmed;arbitrary_cotangent_fused_backward_contract_missing |
| 2026-06-08 22:37:27 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_efficiency_native.py --device cuda:0 --batch-sizes 128,256,512 --hidden 64 --repeats 3 --warmup 1 --seed 2213 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 | completed | route=S1-NativeOfficialRowsPresent fused_complete=84 blocker= |
| 2026-06-08 22:40:26 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_kan_corrected_mapping.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:2 | completed | route=S6-KANCorrectedCarrierNoGo pass_rows=0 device=cuda:2 blocker=KANSourceChannelMismatchConfirmed |
| 2026-06-08 22:48:29 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_kan_corrected_mapping.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:2 | completed | route=S6-KANCorrectedCarrierNoGo pass_rows=0 device=cuda:2 blocker=KANSourceChannelMismatchConfirmed |
| 2026-06-08 22:59:02 +0800 | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_13_kan_corrected_mapping.py --source-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --out-dir results/v22_13_true_loss_interface_operator_fu/official_v22_13 --seed 2213 --device cuda:2 | completed | route=S6-KANCorrectedCarrierNoGo pass_rows=0 device=cuda:2 blocker=KANSourceChannelMismatchConfirmed |

| artifact | exists | size_bytes | sha256 |
| --- | --- | --- | --- |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/figures/v22_13_DCHE_corrected_layout_efficiency.svg | 1 | 323 | 27e02c2e1e414e909c609fb9f5dc39f9da106417c4a3b664f23d09a3c76d7c0e |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/figures/v22_13_DFOU_native_efficiency.svg | 1 | 322 | d488f47f0f017c3042d300edc3e4166631cc4cef64e63b0731b29edef66cde01 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/figures/v22_13_KAN_source_channel_dashboard.svg | 1 | 324 | ff4790a0b2eac658a2dfa7fcc259764fcf15f84978595f6ec3459bfad73ac33d |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/figures/v22_13_adapter_horizon_source_func.svg | 1 | 327 | bd7fd8a3b4469a5648bc9409fe7bac85b18f3fed73d572fc17be152b894604f9 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/figures/v22_13_adapter_horizon_source_loss.svg | 1 | 336 | 9b0ad6e60bbe129d2f126667051f41c85c5d9d7454aaa4f7d414bdc54a82bcd2 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/figures/v22_13_control_projection_dashboard.svg | 1 | 308 | ddb06be157927893c271ffdbbb705e1ec3c44e25a6c6fb11ac34ffb832309233 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/figures/v22_13_convergence_forgetting_expression_dashboard.svg | 1 | 315 | 71ba99491ce19c9cffbf4103a979de7b4e33e6b935e83e17caaea65b259feb7c |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/figures/v22_13_gpu_utilization_dashboard.svg | 1 | 285 | 6175d938a5db8696313caa111dc2781419f5d0d1026e8c588a0b96f4dd704a11 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/figures/v22_13_metric_as_operator_pareto.svg | 1 | 324 | da9ea450da0bb9187711e7498000d6d9e5da89b7dc8ce4f15ec53dde8960b983 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/figures/v22_13_operator_law_dashboard.svg | 1 | 311 | 20caa553252eea7a93e283b7a271bd4970c61905edb730684f5dbe271bb8f1a4 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_DCHE_officialization_matrix.csv | 1 | 20002 | a17b93546ea405865fa2374af520138d3e65bdab0582725c6533c5f31fc4889e |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_DFOU_officialization_matrix.csv | 1 | 15411 | e3aa7c28d14d5576cf17b7e5761114cd3761f96d9f55aa549cf03023924e71c0 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_K15_DFOU_corrected_K13.csv | 1 | 4310 | 256fa090eefcde03705175f37f4bd68924e90840e51ec358bb0a068d7633dd80 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_K16_DCHE_corrected_source_loss_repair.csv | 1 | 4320 | e3caea69f11d0923eb7e3d2d2421e73f7aede50196f7d67a7cd633516d5a5147 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_K17_basis_channel_commit.csv | 1 | 1602 | 3f27877ac68e350f7deda15c4d5aa90c5d7e0d279100d466f8d1fccd4d72abb0 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_KAN_mapping_matrix.csv | 1 | 7391 | 49822cf8a0d1f25f371eca6018fb9be2d7576cd46e22dc55f74d931ef2c6817a |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_KAN_vs_MLP_same_operator.csv | 1 | 7391 | 49822cf8a0d1f25f371eca6018fb9be2d7576cd46e22dc55f74d931ef2c6817a |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_adapter_holdout_matrix.csv | 1 | 19943 | 8eaa87cd347430980de6e333da14bfc6021f7ceaf050d2e77e7123b51ac6ce5e |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_adapter_horizon_matrix.csv | 1 | 68237 | 7e56056c247ece19ccdeb350370ea72f3b24d8b24771a3a1b0d2d9048d84c019 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_adapter_horizon_matrix_fixed_update_diagnostic.csv | 1 | 213534 | 6f3703ee52b118fc256ae4303393c24c070175075146109877b70e83be9fbf33 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_adapter_horizon_matrix_initial_lio2.csv | 1 | 215370 | cbe1dc137612c2711d9b12bdf915a9d06df21e9472bfc8a61b575f1a9068effe |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_adapter_horizon_matrix_true_per_adapter_lio6.csv | 1 | 218250 | 156dd451717ee2c5fbc854383142ff7cc2cd0c2bad006136b0e3b9853c0f526f |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_adapter_renaming_tests.csv | 1 | 396 | 0ea14833398b2761c7fc694b5d5cbb6fdaa370227c9ac7529d7e0416476ccd52 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_artifact_index.csv | 1 | 14883 | 75cd06995e9c315586a99fd873196c9633e10299ab5ea7799acf3d710731fd07 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_calibration_debt_matrix.csv | 1 | 26383 | 35d72d2ed75ccb2760307516f962459fdcdd76d111541fe82e3edb8f17e48c82 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_clean_unzip_compileall.log | 1 | 127 | 1d6627f087a8c62e558e1b2f1df78d91ac11fd0827e8407237dfba9c0c5e54cd |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_clean_unzip_import_closure.log | 1 | 971 | 1824d583ee9834cfb27e8d3a130f11d3dffb2369cf21959fb0d2425db68c22e6 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_code_review_packet.zip | 1 | 897411 | b64ca6bce63181a9b57d7644f02cbb1a0440ed2dd884941550f409d4436f757a |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_code_review_packet_manifest.csv | 1 | 53797 | 9a182bc0e400b1e3a223662c5eda39651629f35d0dc9b342c40e25fb18bc75c5 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_code_truth_gate.csv | 1 | 606 | d6f504c63908acd7d4ad5b77a60a8a542c9327f8a4f0c22e2b917cabcd20ddcb |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_code_truth_route.json | 1 | 452 | 6ae605c2b848e6d02190dc0f516e70b66cc81e403490aec0981f47296b7782c3 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_command_journal.csv | 1 | 16613 | 18fd0895b1789d537d8bd470c88f3430f0f3f0d5638da027ddb7d89fe158a3a8 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_compatibility_manifest.csv | 1 | 654 | 13a5ca17b84a23c8f8c2150cc47cd46a04b8fd4801cbfe5aeecb9727240ee3a3 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_compileall.log | 1 | 127 | 1d6627f087a8c62e558e1b2f1df78d91ac11fd0827e8407237dfba9c0c5e54cd |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_component_timing_waterfall.csv | 1 | 47646 | 8af7e2887f70b5701331fd2250d5aeed6d5f73982e5d508a00a544e0327b4a75 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_control_attribution_matrix.csv | 1 | 57376 | 80e37bea8bc8c440f7ccc6464998c231e22389fef081b0b3c42db024344d2158 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_convergence_speed_matrix.csv | 1 | 26383 | 35d72d2ed75ccb2760307516f962459fdcdd76d111541fe82e3edb8f17e48c82 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_corrected_layout_tests.csv | 1 | 535 | 8ba5dfb247aff1ce1bda348b9f03232812435bb0e994504f432a329672bdfbcf |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_cotangent_suite_readback.csv | 1 | 579 | e057bcc1fa32225111616947908d3cc01a4c45fa13167ee69906b27f6d62e0c0 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_deferred_items.csv | 1 | 178 | b6ebc4ceaede17c3c8f1d6121586dcc1c6078cb05402521a32ab9b456cb1d1c2 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_efficiency_native_route.json | 1 | 274 | b7890240ce1e451965e8f05d7258f00dae57d7f85587ccbcaf765d4e06e03a42 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_expression_metrics_matrix.csv | 1 | 26383 | 35d72d2ed75ccb2760307516f962459fdcdd76d111541fe82e3edb8f17e48c82 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_final_decision.json | 1 | 498 | 7a47010572dc6a731be3ea378a52532d07821acf00f15c0446a5428d2993bca4 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_forgetting_readback_matrix.csv | 1 | 26383 | 35d72d2ed75ccb2760307516f962459fdcdd76d111541fe82e3edb8f17e48c82 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_gpu_assignment_manifest.csv | 1 | 16258 | a2ae19ef67255ff956d8677f8cca11be69139ef4f8bb394bfff609535e14d395 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_gpu_runtime_evidence.csv | 1 | 7142 | 1a49094e9cb6a6ccbe3ace803bf888db2f8c5171a63b43f38468ba173463c344 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_gpu_utilization_timeline.csv | 1 | 2122 | 8dc7cd27f1bd45e81d6e965bc74abfeb691ba9fd6759502d045459fd5bac02e1 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_idle_violation.csv | 1 | 165 | 1db2fe2be16581e535bd8e554df1996b43fb27df8a4f8c3a3599198d31691320 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_import_closure.log | 1 | 971 | 1824d583ee9834cfb27e8d3a130f11d3dffb2369cf21959fb0d2425db68c22e6 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_kan_mapping_route.json | 1 | 403 | 0e91563a8db0341fa8634a2e6cda9b03babd8b5156746a61daecf239a230de4e |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_manual_vs_native_vjp_comparison.csv | 1 | 47646 | 8af7e2887f70b5701331fd2250d5aeed6d5f73982e5d508a00a544e0327b4a75 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_metric_geometry_matrix.csv | 1 | 4748 | f7bcf12200d1ff13b8e44d845c101026eedc6744a2537f11556f5c66567c7fa7 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_native_efficiency_truth_table.csv | 1 | 47646 | 8af7e2887f70b5701331fd2250d5aeed6d5f73982e5d508a00a544e0327b4a75 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_native_kernel_gradcheck.csv | 1 | 228 | 04f4c087d87f87223feaad8a7c37e1457673a457acb21197fdfa7bbbba8fa01f |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_official_fused_status_matrix.csv | 1 | 820 | 627c3da1c4c1049b0a1aaef99f4c0f1f5fde6a4bc0a5a5b02db088ae249b2a0c |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_operator_atom_matrix.csv | 1 | 4748 | f7bcf12200d1ff13b8e44d845c101026eedc6744a2537f11556f5c66567c7fa7 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_operator_atom_route.json | 1 | 193 | d6fdc8495e8a009df3062c8f30dc077bda84bed47cd341b61d4977ca81a339e2 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_operator_commit_matrix.csv | 1 | 1928 | 259c0f95727508e5022a18b0015b1a30b12e22732c258340d506662f1fbe899c |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_operator_commit_payload.pt | 1 | 21391 | f7ee6532cf9c7f019ce91440fb3ef2d65da52650ac2a2eddc72f59467b8c7172 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_operator_commit_route.json | 1 | 191 | 2a417f90d59aad6c1447582b718ec8f3bfbc5c9ccacc05c8a41de2345837f132 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_operator_horizon_route.json | 1 | 694 | f296615cd9f93aa157d9721757746d10df5dd9d7d54cf4d105b6c3bffa50c4e2 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_operator_horizon_route_fixed_update_diagnostic.json | 1 | 581 | 9037636baa053230f5ecd13ac8ca59615c8818c12e12c80b3d89c214d93d0c94 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_operator_horizon_route_initial_lio2.json | 1 | 581 | 9037636baa053230f5ecd13ac8ca59615c8818c12e12c80b3d89c214d93d0c94 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_operator_horizon_route_true_per_adapter_lio6.json | 1 | 565 | 5fff27aa35c0cd662b44190a1376d71b9cb498e3201ce9ed336f05e63296396a |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_operator_law_tests.csv | 1 | 1047 | da6e3d59e52f782cb1b4276fbb61d312fa2b5170d39bc01d2bd87b5d231a2816 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_operator_repair_sweep.csv | 1 | 18640 | 30a6517279ce91ce6c0972ef4ae42fbd3943fa10fb6b74e2fc944e485777c4c0 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_operator_repair_sweep_route.json | 1 | 386 | b559c38b8f037708100b430d1db8d03472bf6067c098a4d772cd09df52b15cd4 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_operator_role_blind_tests.csv | 1 | 351 | b94ccd9653052cf00bb770bd8c70e9c3186e54f511d38089591308ab82d8d9e6 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_operator_semantic_route.json | 1 | 180 | 79cfc0e5de98987c72ebfa33b82f258d7be46d20c50f5c632e4d3de475a40cdb |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_operator_step_efficiency.csv | 1 | 47646 | 8af7e2887f70b5701331fd2250d5aeed6d5f73982e5d508a00a544e0327b4a75 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_operator_variational_route.json | 1 | 262 | fba0867bb5c6472682481dbd333d0b569ba6058f9c181181b1571ad6f22bde18 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_packet_required_compare.csv | 1 | 1030 | 5398894a0a2e06291d5ec0e96363316a6be8a62327d28fef994814650c2de80c |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_profiler_unit_tests.csv | 1 | 85 | c3a85575d0cfa3111bf73af781e6ac085d8cb5d2f83227a445afdef5ac155492 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_promotion_semantics_tests.csv | 1 | 277 | 30364c2a638a6525eceff53f5990df0716397c630cff58f27fb232e9a49c86d8 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_queue_drain_report.json | 1 | 124 | ff640e487bb921b3b26cfc0b952f5b1beb218045ff56847c5ec64702c210ee32 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_readout_vs_basis_ablation.csv | 1 | 1602 | 3f27877ac68e350f7deda15c4d5aa90c5d7e0d279100d466f8d1fccd4d72abb0 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_required_source_files.csv | 1 | 1018 | 3a09ce238d4d009283f3ee62379608081a6650b3ff443a0845b1868f8fd643bd |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_results_bundle.zip | 1 | 895734 | 098c49f6a1b9ebb53a0ad24a235348c18d7aa721c96130438cdd2ed025fb3845 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_runnable_queue.csv | 1 | 10805 | 9e4e1f79b43efeffa35f6fb6dc3bbbce11b8b13aa9a3bd3b8195c01d266170b6 |
| results/v22_13_true_loss_interface_operator_fu/official_v22_13/v22_13_runtime_provenance.csv | 1 | 2563 | 7b78238b7ab604222588dce3530aef37580c6f7ac214290c803e6b306e628661 |

_仅显示前 80 / 86 rows；完整 CSV 见 artifact。_

Next action: FU operator 与 D-FOU/D-CHE native fused arbitrary-cotangent efficiency 已过；下一 blocker 是 KAN basis-channel carrier geometry，需要新的 basis-state objective/architecture，而不是继续 readout replay 或无界 gain sweep。

