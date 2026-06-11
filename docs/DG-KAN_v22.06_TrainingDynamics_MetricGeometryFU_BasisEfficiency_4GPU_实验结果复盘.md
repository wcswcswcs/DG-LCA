# DG-KAN v22.06 Training-Dynamics MetricGeometryFU BasisEfficiency 4GPU 实验结果复盘

生成时间：2026-06-06 20:38:35 +0800

## Route

- route: `C3-ActuationOnly`
- promotion_allowed: 0
- blocking_metric: `functional_metric_solver_source_chain;h3200_source_missing;MLP_h3200_source_missing`
- S0.13 pass: 1
- D-CHE/D-FOU S1 pass: 1 / 1
- D-RAT/D-RBF production official any pass: 1
- functional_route: `C3-ActuationOnly`
- terminal_preservation_decision: `C5BlockedBeforeTerminalPreservation`
- KAN_mapping_decision: `KANMappingNotEntered`
- next_codex_action: repair C1/C3/C4 blocker before h4800/KAN mapping

## Part A S0.13 Code / Metric / Mechanism Gate

| check | pass | metric | value | blocker |
| --- | --- | --- | --- | --- |
| required_source_files | 1 | exists | 28/28 |  |
| compileall | 1 | py_compile | 0 |  |
| import_closure | 1 | self_contained_import | 0 |  |
| linec_fast_golden | 1 | tests | 9 |  |
| linec_channel_golden | 1 | tests | 8 |  |
| source_chain_tests | 1 | tests | 7/7 |  |
| terminal_retention_tests | 1 | tests | 14/14 |  |
| metric_solver_tests | 1 | tests | 33/33 |  |
| mechanism_contracts | 1 | v22.06 semantic+solver | 12/12;alias_undeclared=0 |  |
| kernel_gradcheck | 1 | kernels | 4 |  |
| profiler_phase_tests | 1 | tests | 2/2 |  |


### Functional Semantic Contract

| mechanism_id | target_constructor_type | metric_operator_type | solver_type | uses_future_or_validation | uses_audit_metric_for_direction | is_proxy |
| --- | --- | --- | --- | --- | --- | --- |
| M259-V2206MetricSolverT0G0ReadoutFU | T0-LossCotangent | G0-L2 | S1-ReadoutExactSolve | 0 | 0 | 0 |
| M260-V2206MetricSolverT1G0SplitTransferFU | T1-SplitTransfer | G0-L2 | S1-ReadoutExactSolve | 0 | 0 | 0 |
| M261-V2206MetricSolverT3G3SignalSobolevFU | T3-SignalChannel | G3-SobolevH1 | S1-ReadoutExactSolve | 0 | 0 | 0 |
| M262-V2206MetricSolverT4G3SmoothManifoldFU | T4-SmoothManifold | G3-SobolevH1 | S1-ReadoutExactSolve | 0 | 0 | 0 |
| M263-V2206MetricSolverT5G6LowNDSFU | T5-LowNDS | G6-LowNDS | S1-ReadoutExactSolve | 0 | 0 | 0 |
| M264-V2206MetricSolverT6G0DualMemoryFU | T6-DualMemorySource | G0-L2 | S1-ReadoutExactSolve | 0 | 0 | 0 |
| M265-V2206MetricSolverT7G0HiddenBlockFU | T7-BlockSourceChannel | G0-L2 | S2-ReadoutExactHiddenResidualBlock | 0 | 0 | 0 |
| M266-V2206MetricSolverT8G0AdaptiveHiddenBlockFU | T8-AdaptiveBlockSourceChannel | G0-L2 | S2-ReadoutExactAdaptiveHiddenResidualBlock | 0 | 0 | 0 |
| M267-V2206MetricSolverT9G0C3GatedHiddenBlockFU | T9-C3GatedBlockSourceChannel | G0-L2 | S2-ReadoutExactC3GatedHiddenResidualBlock | 0 | 0 | 0 |
| M268-V2206MetricSolverT10G0CompensatedHiddenBlockFU | T10-CompensatedBlockSourceChannel | G0-L2 | S2-ReadoutExactCompensatedHiddenResidualBlock | 0 | 0 | 0 |
| M269-V2206MetricSolverT11G0EarlyObservableFU | T11-EarlyObservableSourceChannel | G0-L2 | S2-ReadoutExactCompensatedHiddenResidualBlock | 0 | 0 | 0 |
| M270-V2206MetricSolverT12G0SoftCompensatedHiddenBlockFU | T12-SoftCompensatedBlockSourceChannel | G0-L2 | S2-ReadoutExactSoftCompensatedHiddenResidualBlock | 0 | 0 | 0 |


### Semantic Alias Summary

| pairs | semantic_alias_pairs | undeclared_alias_pairs | pass | gradient_norm |
| --- | --- | --- | --- | --- |
| 66 | 2 | 0 | 1 | 0.9039162397384644 |


## Part B Basis Efficiency / Officialization

| carrier | forward_ratio_vs_mlp | step_ratio_vs_mlp | memory_ratio_vs_mlp | v22_06_S1_pass | v22_06_decision | v22_06_blocker |
| --- | --- | --- | --- | --- | --- | --- |
| D-CHE | 0.9305108277155166 | 0.46682782835473463 | 0.9913236539978382 | 1 | OfficialEfficientCarrier |  |
| D-FOU | 1.0087315984225318 | 0.5089279836996342 | 0.9888469358117921 | 1 | OfficialEfficientCarrier |  |


| carrier | profile_rows | production_fused_official_rows | near_E1_rows | best_forward_ratio | best_step_ratio | best_memory_ratio | decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 1 | 1 | 1 | 1.286250435712521 | 0.4656189118282311 | 0.9618257261410789 | ProductionFusedPass |  |
| D-RBF | 1 | 1 | 1 | 1.2300505670534045 | 0.4242078290119725 | 0.9618257261410789 | ProductionFusedPass |  |


## Part C Metric-As-Geometry Solver

- best_metric_v21_id: `MLP-V2206-C4-T10G0-OptTransport-lr150`
- best_metric_solver_name: `C4-I2-T10-G0-optimizer-state-transport-lr150`
- best source_h800: -0.01733402411142985
- best source_h3200: 0.10594364007314046
- best C1/C3/C4: 1/1/0

| v21_id | metric_solver_name | ActuationR2_mean | projection_residual_Gf_mean | B2_transfer_gain_mean | source_to_reservoir_leakage_mean | source_h100_mean | source_h400_mean | source_h800_mean | source_h1600_mean | source_h3200_mean | C1_observability_pass | C3_actuation_pass | C4_early_source_pass | C4_h3200_source_pass | failure_taxonomy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-C4-T0G0-EarlyWarmCarry-stop800-lr50 | C4-I1-T0-G0-early-warm-carry-stop800-lr50 | 0.99272487560908 | 0.07225360355367132 | 0.8054320414861044 | 0.0 | -0.8436988592147827 | -0.2790458599726359 | -0.08068915208180745 | 0.02560742696126302 | -0.024547497431437176 | 0 | 1 | 0 | 0 | SourceObservableMissing |
| MLP-V2206-C4-T0G0-SourceStateCarry | C4-I2-T0-G0-source-state-carry | 0.9973375002543131 | 0.045386538611308334 | 0.7947990496953329 | 0.0 | -0.8496090173721313 | -0.2616690794626872 | -0.03692921002705892 | 0.015709320704142254 | -0.012500127156575521 | 0 | 1 | 0 | 0 | SourceObservableMissing |
| MLP-V2206-C4-T0G0-SourceStateCarry-lr50 | C4-I2-T0-G0-source-state-carry-lr50 | 0.998833179473877 | 0.02918781325718307 | 0.8156107266743978 | 0.0 | -0.8193617264429728 | -0.2612844705581665 | -0.06766160329182942 | 0.04786324501037598 | 0.025296489397684734 | 1 | 1 | 0 | 0 | SourceFormationFailed |
| MLP-V2206-C4-T4G3-EarlyWarmCarry-stop800-lr50 | C4-I1-T4-G3-early-warm-carry-stop800-lr50 | 0.9493783513704935 | 0.2242478830891259 | 0.5370466311772665 | 0.0 | -0.8256907860438029 | -0.24316024780273438 | -0.05088643232981364 | 0.012374957402547201 | 0.025008996327718098 | 0 | 1 | 0 | 0 | SourceObservableMissing |
| MLP-V2206-C4-T4G3-SourceStateCarry | C4-I2-T4-G3-source-state-carry | 0.9602707624435425 | 0.19882547707043807 | 0.46993152300516766 | 0.0 | -0.8079226414362589 | -0.2080939213434855 | -0.0245439608891805 | 0.05801637967427572 | -0.027982711791992188 | 1 | 1 | 0 | 0 | SourceFormationFailed |
| MLP-V2206-C4-T4G3-SourceStateCarry-lr100 | C4-I2-T4-G3-source-state-carry-lr100 | 0.9595637321472168 | 0.19843675834820437 | 0.49695467948913574 | 0.0 | -0.8304964303970337 | -0.2887642780939738 | -0.13448822498321533 | -0.08899132410685222 | -0.10860745112101237 | 0 | 1 | 0 | 0 | SourceObservableMissing |
| MLP-V2206-C4-T4G3-SourceStateCarry-lr50 | C4-I2-T4-G3-source-state-carry-lr50 | 0.9644299944241842 | 0.18503772217720157 | 0.5260411103566488 | 0.0 | -0.8422542015711466 | -0.27304601669311523 | -0.0700670878092448 | -0.008948922157287598 | -0.11255943775177002 | 1 | 1 | 0 | 0 | SourceFormationFailed |
| MLP-V2206-C4-T6G0-EarlyWarmCarry-stop1200-lr50 | C4-I1-T6-G0-early-warm-carry-stop1200-lr50 | 0.9852416117986044 | 0.11921276407258152 | 0.4654558499654134 | 0.0 | -0.8612551291783651 | -0.34471197923024494 | -0.17973379294077554 | -0.0692359209060669 | 0.08123322327931722 | 1 | 1 | 0 | 0 | SourceFormationFailed |
| MLP-V2206-C4-T6G0-EarlyWarmCarry-stop800-lr50 | C4-I1-T6-G0-early-warm-carry-stop800-lr50 | 0.9970434308052063 | 0.054059968776061194 | 0.4429957866668701 | 0.0 | -0.8809543053309122 | -0.3212350209554036 | -0.14276675383249918 | -0.010830005009969076 | 0.05094432830810547 | 0 | 1 | 0 | 0 | SourceObservableMissing |
| MLP-V2206-C4-T6G0-SourceStateCarry | C4-I2-T6-G0-source-state-carry | 0.9947919050852457 | 0.07200418068756309 | 0.40363264083862305 | 0.0 | -0.8706784645716349 | -0.32406993707021076 | -0.14671850204467773 | -0.0699915885925293 | -0.10445356369018555 | 1 | 1 | 0 | 0 | SourceFormationFailed |
| MLP-V2206-C4-T7G0-EarlyWarmCarry-stop1200-lr50 | C4-I1-T7-G0-hidden-block-early-warm-carry-stop1200-lr50 | -104.67362467447917 | 10.191904855566861 | -0.14645318190256754 | 0.0 | -0.8326851924260458 | -0.26878825823465985 | -0.0768817663192749 | 0.04568298657735189 | -0.020708918571472168 | 0 | 0 | 0 | 0 | SourceObservableMissing |
| MLP-V2206-C4-T7G0-EarlyWarmCarry-stop800-lr50 | C4-I1-T7-G0-hidden-block-early-warm-carry-stop800-lr50 | -205.03264172871908 | 12.8825433228911 | -1.3160916964213054 | 0.0 | -0.8211052020390829 | -0.25990716616312665 | -0.06610695521036784 | 0.014756401379903158 | -0.061401049296061196 | 0 | 0 | 0 | 0 | SourceObservableMissing |
| MLP-V2206-C4-T7G0-SourceStateCarry | C4-I2-T7-G0-hidden-block-source-state-carry | -376.8154748280843 | 15.61417212503987 | -0.2973654270172119 | 0.0 | -0.7898726065953573 | -0.19072425365447998 | 0.05185246467590332 | 0.17102781931559244 | 0.14964707692464194 | 0 | 0 | 0 | 0 | SourceObservableMissing |
| MLP-V2206-C4-T8G0-EarlyWarmCarry-stop800-lr50 | C4-I1-T8-G0-adaptive-hidden-block-early-warm-carry-stop800-lr50 | 0.7525013089179993 | 0.49425418835418594 | 0.5156490008036295 | 0.0 | -0.8170397679011027 | -0.2820080518722534 | -0.12115935484568278 | -0.02786719799041748 | -0.022089838981628418 | 1 | 1 | 0 | 0 | SourceFormationFailed |
| MLP-V2206-C4-T8G0-SourceStateCarry | C4-I2-T8-G0-adaptive-hidden-block-source-state-carry | 0.4323558410008748 | 0.6701708698169795 | 0.5701238711675009 | 0.0 | -0.8376067876815796 | -0.30284615357716876 | -0.13018997510274252 | 0.0029562711715698242 | -0.030415852864583332 | 1 | 0 | 0 | 0 | SolverBlocked |
| MLP-V2206-C4-T9G0-EarlyWarmCarry-stop800-lr50 | C4-I1-T9-G0-c3-gated-hidden-block-early-warm-carry-stop800-lr50 | 0.9844643274943033 | 0.11792926266428494 | 0.2618499994277954 | 0.0 | -0.8242918252944946 | -0.26072196165720624 | -0.0816118319829305 | -0.005133310953776042 | -0.07096739610036214 | 1 | 1 | 0 | 0 | SourceFormationFailed |
| MLP-V2206-C4-T9G0-SourceStateCarry | C4-I2-T9-G0-c3-gated-hidden-block-source-state-carry | 0.9533759554227194 | 0.21199685568490753 | 0.33714954058329266 | 0.0 | -0.8333876530329386 | -0.28026143709818524 | -0.10595667362213135 | -0.006269057591756185 | 0.006149133046468099 | 1 | 1 | 0 | 0 | SourceFormationFailed |
| MLP-V2206-S1-T0G0-ReadoutSolver | S1-T0-G0-readout-exact | 0.9997114141782125 | 0.016663477535324073 | 0.8370923598607382 | 0.0 | -0.8387937943140665 | -0.4592361052831014 | -0.4860517978668213 | -0.6799108187357584 | -0.7891813913981119 | 1 | 1 | 0 | 0 | SourceFormationFailed |
| MLP-V2206-S1-T1G0-SplitTransferSolver | S1-T1-G0-split-transfer-readout-exact | 0.9712942043940226 | 0.14055998289485463 | 0.22672414779663086 | 0.0 | -0.9130697647730509 | -0.5361276865005493 | -0.5664152304331461 | -0.7671961784362793 | -0.8891070683797201 | 1 | 1 | 0 | 0 | SourceFormationFailed |
| MLP-V2206-S1-T3G3-SignalSobolevSolver | S1-T3-G3-signal-sobolev-readout-exact | 0.9985658725102743 | 0.037510115377168494 | 0.7544203201929728 | 0.0 | -0.9217824538548788 | -0.5437065362930298 | -0.5726637045542399 | -0.7703961531321207 | -0.8871746063232422 | 1 | 1 | 0 | 0 | SourceFormationFailed |

_仅显示前 20 / 48 rows；完整 CSV 见 artifact。_


### Continuation Artifact Readback

| artifact | command_journal | matrix_rows | unique_v21_ids | measured_rows | decision |
| --- | --- | --- | --- | --- | --- |
| metric_solver_source_smoke_v6_full | metric_solver_source_smoke_v6_full/v21_01_command_journal.csv | 42 | 14 | 42 | registration_blocker: new M264 specs were absent from mlp allowlist; not used as final M264 evidence |
| metric_solver_source_smoke_v6_full_fresh | metric_solver_source_smoke_v6_full_fresh/v21_01_command_journal.csv | 60 | 20 | 60 | fresh T6 dual-memory / early-warm-carry readback after allowlist repair |
| metric_solver_source_smoke_v7_hidden_block | metric_solver_source_smoke_v7_hidden_block/v21_01_command_journal.csv | 72 | 24 | 72 | fresh T7 hidden-block solver readback |
| metric_solver_source_smoke_v8_adaptive_hidden_block | metric_solver_source_smoke_v8_adaptive_hidden_block/v21_01_command_journal.csv | 81 | 27 | 81 | fresh T8 adaptive hidden-block solver readback |
| metric_solver_source_smoke_v9_c3_gated_hidden_block | metric_solver_source_smoke_v9_c3_gated_hidden_block/v21_01_command_journal.csv | 90 | 30 | 90 | fresh T9 C3-gated hidden-block final readback |
| metric_solver_source_smoke_v10_compensated_hidden_block | metric_solver_source_smoke_v10_compensated_hidden_block/v21_01_command_journal.csv | 24 | 8 | 24 | initial T10 compensated hidden-block readback before trace compensation whitelist repair |
| metric_solver_source_smoke_v10_compensated_hidden_block_fresh | metric_solver_source_smoke_v10_compensated_hidden_block_fresh/v21_01_command_journal.csv | 24 | 8 | 24 | fresh T10 compensated hidden-block readback after trace compensation whitelist repair |
| metric_solver_source_smoke_v11_periodic_compensated_hidden_block | metric_solver_source_smoke_v11_periodic_compensated_hidden_block/v21_01_command_journal.csv | 18 | 6 | 18 | fresh T10 I3 periodic boundary carry readback |
| metric_solver_source_smoke_v12_sgd_bootstrap_compensated_hidden_block | metric_solver_source_smoke_v12_sgd_bootstrap_compensated_hidden_block/v21_01_command_journal.csv | 18 | 6 | 18 | fresh T10 SGD-bootstrap then compensated hidden carry readback |
| metric_solver_source_smoke_combined_v9_to_v12_readback |  | 150 | 38 | 150 | combined official readback for T9/T10 hidden-block, compensation, periodic, and bootstrap smokes |
| metric_solver_source_smoke_v13_optimizer_transport | metric_solver_source_smoke_v13_optimizer_transport/v21_01_command_journal.csv | 18 | 6 | 18 | fresh T10 optimizer-state transport smoke from plan 11.5 |
| metric_solver_source_smoke_v14_optimizer_transport_strength_sanity | metric_solver_source_smoke_v14_optimizer_transport_strength_sanity/v21_01_command_journal.csv | 18 | 6 | 18 | fresh T10 optimizer-state transport low-strength sanity smoke |
| metric_solver_source_smoke_combined_v9_to_v14_readback |  | 186 | 42 | 186 | combined official readback for T9/T10 hidden-block, compensation, periodic, bootstrap, optimizer transport, and low-strength sanity smokes |
| metric_solver_source_smoke_v15_early_observable_source_channel | metric_solver_source_smoke_v15_early_observable_source_channel/v21_01_command_journal.csv | 21 | 7 | 21 | diagnostic whitelist blocker: T11 target construction diagnostics were absent from raw trace; not used as final T11 evidence |
| metric_solver_source_smoke_v15_early_observable_source_channel_fresh | metric_solver_source_smoke_v15_early_observable_source_channel_fresh/v21_01_command_journal.csv | 21 | 7 | 21 | fresh T11 train-only early-observable source-channel smoke after trace whitelist repair |
| metric_solver_source_smoke_combined_v9_to_v15_readback |  | 0 | 0 | 0 | combined official readback including fresh T11 early-observable source-channel smoke |
| metric_solver_source_smoke_v16_train_split_control_gate | metric_solver_source_smoke_v16_train_split_control_gate/v21_01_command_journal.csv | 15 | 5 | 15 | fresh T11 train-split control-relative gate smoke after T11 source formation failed |
| metric_solver_source_smoke_combined_v9_to_v16_readback |  | 222 | 46 | 222 | combined official readback including T11 control-relative gate smoke |
| metric_solver_source_smoke_v17_adamw_compatible_c4 | metric_solver_source_smoke_v17_adamw_compatible_c4/v21_01_command_journal.csv | 6 | 2 | 6 | control-missing blocker: ran only new FU specs without matched controls; source_h* empty and not used as final AdamW-compatible C4 evidence |
| metric_solver_source_smoke_v17_adamw_compatible_c4_fresh | metric_solver_source_smoke_v17_adamw_compatible_c4_fresh/v21_01_command_journal.csv | 18 | 6 | 18 | fresh AdamW-compatible C4 smoke with matched controls: T10 AdamW bootstrap+transport and T11 AdamW-relative gate |

_仅显示前 20 / 26 rows；完整 CSV 见 artifact。_


### M264-M269 Repair Readback

| attempt | v21_id | source_h100 | source_h400 | source_h800 | source_h1600 | source_h3200 | C1 | C3 | C4 | ActuationR2 | projection_residual | B2_transfer | hidden_fraction | hidden_cap_ratio | compensation_scale | compensation_mix | compensation_residual | transport_active | transport_strength | transport_before | transport_after | control_gate_active | control_gate_accept | control_gate_margin | adamw_bootstrap_active | adamw_gate_active | adamw_gate_accept | adamw_gate_margin | failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T6 dual-memory early warm | MLP-V2206-C4-T6G0-EarlyWarmCarry-stop1200-lr50 | -0.8612551291783651 | -0.34471197923024494 | -0.17973379294077554 | -0.0692359209060669 | 0.08123322327931722 | 1 | 1 | 0 | 0.9852416117986044 | 0.11921276407258152 | 0.4654558499654134 | 0.0 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | SourceFormationFailed |
| T7 hidden block carry | MLP-V2206-C4-T7G0-SourceStateCarry | -0.7898726065953573 | -0.19072425365447998 | 0.05185246467590332 | 0.17102781931559244 | 0.14964707692464194 | 0 | 0 | 0 | -376.8154748280843 | 15.61417212503987 | -0.2973654270172119 | 0.3303504121762613 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | SourceObservableMissing |
| T8 adaptive hidden early warm | MLP-V2206-C4-T8G0-EarlyWarmCarry-stop800-lr50 | -0.8170397679011027 | -0.2820080518722534 | -0.12115935484568278 | -0.02786719799041748 | -0.022089838981628418 | 1 | 1 | 0 | 0.7525013089179993 | 0.49425418835418594 | 0.5156490008036295 | 0.0294184317523908 | 0.0841224839289983 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | SourceFormationFailed |
| T9 C3-gated hidden carry | MLP-V2206-C4-T9G0-SourceStateCarry | -0.8333876530329386 | -0.28026143709818524 | -0.10595667362213135 | -0.006269057591756185 | 0.006149133046468099 | 1 | 1 | 0 | 0.9533759554227194 | 0.21199685568490753 | 0.33714954058329266 | 0.0 | 0.0 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | SourceFormationFailed |
| T9 C3-gated hidden early warm | MLP-V2206-C4-T9G0-EarlyWarmCarry-stop800-lr50 | -0.8242918252944946 | -0.26072196165720624 | -0.0816118319829305 | -0.005133310953776042 | -0.07096739610036214 | 1 | 1 | 0 | 0.9844643274943033 | 0.11792926266428494 | 0.2618499994277954 | 0.0 | 0.0 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | SourceFormationFailed |
| T10 compensated hidden carry | MLP-V2206-C4-T10G0-SourceStateCarry | -0.8150824308395386 | -0.2511293888092041 | -0.051925857861836754 | -0.013471206029256185 | -0.0655744473139445 | 1 | 1 | 0 | 0.991801122824351 | 0.08641707506192091 | 0.40393821398417157 | 0.4994814325820182 | 1.0 | 1.0 | nan | 0.08641707400480907 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | SourceFormationFailed |
| T10 compensated hidden early warm | MLP-V2206-C4-T10G0-EarlyWarmCarry-stop800-lr50 | -0.8213712771733602 | -0.2901890277862549 | -0.08838526407877605 | 0.09818780422210693 | 0.04999268054962158 | 1 | 1 | 0 | 0.9864400227864584 | 0.10381262274160992 | 0.49437109629313153 | 0.29879223084026574 | 1.0 | 1.0 | nan | 0.103812621285518 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | SourceFormationFailed |
| T10 compensated hidden early warm 1200 | MLP-V2206-C4-T10G0-EarlyWarmCarry-stop1200-lr50 | -0.8541640043258667 | -0.27850425243377686 | -0.1156003475189209 | -0.055393099784851074 | -0.05905457337697347 | 1 | 1 | 0 | 0.9785407781600952 | 0.13916354678369833 | 0.5025749206542969 | 0.435843495379953 | 1.0 | 1.0 | nan | 0.13916354378064474 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | SourceFormationFailed |
| T10 periodic boundary alt100 | MLP-V2206-C4-T10G0-PeriodicCarry-alt100-lr50 | -0.8030741612116495 | -0.26765533288319904 | -0.08940935134887695 | -0.011136293411254883 | -0.07513431708017985 | 1 | 1 | 0 | 0.9949227968851725 | 0.060931013381421345 | 0.3734227816263835 | 0.3795884484650571 | 1.0 | 1.0 | nan | 0.06093101451794306 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | SourceFormationFailed |
| T10 periodic boundary alt200 | MLP-V2206-C4-T10G0-PeriodicCarry-alt200-lr50 | -0.8157867987950643 | -0.28404712677001953 | -0.09765390555063884 | 0.03194785118103027 | -0.010401129722595215 | 1 | 1 | 0 | 0.9954753319422404 | 0.06405369658109304 | 0.5348544915517172 | 0.388018057669943 | 1.0 | 1.0 | nan | 0.06405369689067204 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | SourceFormationFailed |
| T10 SGD bootstrap 400 | MLP-V2206-C4-T10G0-SGDBootstrap400Carry-lr50 | -0.8117222388585409 | -0.22788619995117188 | -0.017781813939412434 | 0.03737719853719076 | -0.0061052242914835615 | 0 | 1 | 0 | 0.987810214360555 | 0.09905393750675477 | 0.5123787720998129 | 0.41924507106788456 | 1.0 | 1.0 | nan | 0.09905393421649933 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | SourceObservableMissing |
| T10 SGD bootstrap 800 | MLP-V2206-C4-T10G0-SGDBootstrap800Carry-lr50 | -0.8211808602015177 | -0.27470799287160236 | -0.10890098412831624 | -0.04057816664377848 | 0.02890467643737793 | 1 | 1 | 0 | 0.9970758358637491 | 0.04508499088714533 | 0.35887567202250165 | 0.37413572555657915 | 1.0 | 1.0 | nan | 0.04508499056100845 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | SourceFormationFailed |
| T10 optimizer transport lr50 | MLP-V2206-C4-T10G0-OptTransport-lr50 | -0.8442986408869425 | -0.2821107308069865 | -0.08200136820475261 | 0.015452822049458822 | 0.049054245154062905 | 1 | 1 | 0 | 0.9713107744852701 | 0.16912504020863647 | 0.4394036531448364 | 0.39932173598156867 | 1.0 | 1.0 | nan | 0.16912504037221274 | 1.0 | 0.016666666666666666 | 0.1584947407245636 | 0.2053684244553248 | nan | nan | nan | nan | nan | nan | nan | SourceFormationFailed |
| T10 optimizer transport lr100 | MLP-V2206-C4-T10G0-OptTransport-lr100 | -0.8135633865992228 | -0.2716893752415975 | -0.09880586465199788 | 0.041435440381368004 | 0.029567519823710125 | 0 | 1 | 0 | 0.9934599002202352 | 0.07991415518624269 | 0.4631234407424927 | 0.42097741070790606 | 1.0 | 1.0 | nan | 0.07991415510574977 | 1.0 | 0.03333333333333333 | 0.1733075181643168 | 0.273345152537028 | nan | nan | nan | nan | nan | nan | nan | SourceObservableMissing |
| T10 optimizer transport lr150 | MLP-V2206-C4-T10G0-OptTransport-lr150 | -0.8047353029251099 | -0.24364701906840006 | -0.01733402411142985 | 0.08956396579742432 | 0.10594364007314046 | 1 | 1 | 0 | 0.995230476061503 | 0.06252188327806966 | 0.4587490161259969 | 0.4348008939512716 | 1.0 | 1.0 | nan | 0.06252188235521317 | 1.0 | 0.049999999999999996 | 0.17456597089767456 | 0.30884091059366864 | nan | nan | nan | nan | nan | nan | nan | SourceFormationFailed |
| T10 optimizer transport lr300 | MLP-V2206-C4-T10G0-OptTransport-lr300 | -0.8305602471033732 | -0.2678850491841634 | -0.02946778138478597 | 0.05090717474619547 | -0.015208919843037924 | 0 | 1 | 0 | 0.9864683945973715 | 0.09760163687741191 | 0.44353270530700684 | 0.4279762298953352 | 1.0 | 1.0 | nan | 0.09760163569202025 | 1.0 | 0.09999999999999999 | 0.15667145947615305 | 0.4246240059534709 | nan | nan | nan | nan | nan | nan | nan | SourceObservableMissing |
| T10 early warm400 then transport | MLP-V2206-C4-T10G0-EarlyWarm400ThenOptTransport-lr150 | -0.7822021245956421 | -0.21223068237304688 | -0.012645920117696127 | 0.056976874669392906 | -0.05749619007110596 | 1 | 1 | 0 | 0.976615826288859 | 0.14748027853414117 | 0.42245248953501385 | 0.2970512175969064 | 0.8333333333333334 | 0.8333333333333334 | nan | 0.14748027672370276 | 1.0 | 0.049999999999999996 | 0.15041036903858185 | 0.28391047318776447 | 0.0 | nan | nan | 0.0 | 0.0 | nan | nan | SourceFormationFailed |
| T10 early warm800 then transport | MLP-V2206-C4-T10G0-EarlyWarm800ThenOptTransport-lr150 | -0.7872394323348999 | -0.19613893826802573 | -0.022436102231343586 | -0.02192374070485433 | -0.09068103631337483 | 1 | 1 | 0 | 0.9891584912935892 | 0.09356643001376504 | 0.4132461945215861 | 0.35855322146546104 | 1.0 | 1.0 | nan | 0.09356643135348956 | 0.0 | 0.0 | 0.029193018873532612 | 0.07758076985677083 | 0.0 | nan | nan | 0.0 | 0.0 | nan | nan | SourceFormationFailed |
| T11 early observable solver | MLP-V2206-S2-T11G0-EarlyObservableSolver | -0.8701054652531942 | -0.49638672669728595 | -0.5309991041819254 | -0.7399615446726481 | -0.8780443668365479 | 1 | 1 | 0 | 0.9806051452954611 | 0.12664296840648911 | 0.0822907288869222 | 0.25123379590446343 | 1.0 | 1.0 | nan | 0.1266429697473844 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | SourceFormationFailed |
| T11 early observable warm carry | MLP-V2206-C4-T11G0-EarlyWarmCarry-stop800-lr50 | -0.8811734120051066 | -0.3169331153233846 | -0.12356845537821452 | -0.056880950927734375 | -0.022106210390726726 | 1 | 1 | 0 | 0.9710202217102051 | 0.14142618807247528 | 0.2116004228591919 | 0.24280001528603945 | 1.0 | 1.0 | nan | 0.14142618576685587 | 0.0 | 0.0 | nan | nan | nan | nan | nan | nan | nan | nan | nan | SourceFormationFailed |
| T11 early observable optimizer transport | MLP-V2206-C4-T11G0-OptTransport-lr150 | -0.8546330531438192 | -0.30620535214742023 | -0.1135173241297404 | -0.01245113213857015 | -0.0009657939275105795 | 1 | 1 | 0 | 0.9957298636436462 | 0.0584388924954947 | 0.13559802373250326 | 0.2462769997432921 | 1.0 | 1.0 | nan | 0.0584388921658198 | 1.0 | 0.049999999999999996 | 0.107305941482385 | 0.17463627457618713 | nan | nan | nan | nan | nan | nan | nan | SourceFormationFailed |
| T11 train-split control gate | MLP-V2206-C4-T11G0-ControlRelativeGate-stop800-lr50 | -0.8356713851292928 | -0.2914532820383708 | -0.07091979185740153 | 0.056077798207600914 | 0.09086314837137859 | 1 | 1 | 0 | 0.9796734849611918 | 0.10177733718824951 | 0.16056223710378012 | 0.2690184219946268 | 1.0 | 1.0 | nan | 0.10177733500798543 | 0.0 | 0.0 | nan | nan | 1.0 | 0.0 | -0.0008140788413584232 | nan | nan | nan | nan | SourceFormationFailed |
| T10 AdamW bootstrap transport | MLP-V2206-C4-T10G0-AdamWBootstrap800OptTransport-lr150 | -0.07926638921101888 | -0.2470987637837728 | -0.5363105138142904 | -0.7374041875203451 | -0.8600086371103922 | 1 | 1 | 0 | 0.959965705871582 | 0.18890827379968447 | 0.010280445218086243 | 0.27049391819764096 | 1.0 | 1.0 | nan | 0.1889082690080007 | 0.0 | 0.0 | 0.00017067718727048486 | 0.005942043848335743 | 0.0 | nan | nan | 1.0 | 0.0 | nan | nan | SourceFormationFailed |
| T11 AdamW-relative gate | MLP-V2206-C4-T11G0-AdamWControlGate-stop800-lr50 | -0.06274338563283284 | -0.17491801579793295 | -0.4344786802927653 | -0.635701576868693 | -0.75897745291392 | 1 | 1 | 0 | 0.9981906215349833 | 0.03724563138607965 | 0.022084130595127743 | 0.24752980986731407 | 1.0 | 1.0 | nan | 0.03724563059707483 | 0.0 | 0.0 | nan | nan | 0.0 | nan | nan | 0.0 | 1.0 | 0.0 | 5.4598586984866415e-08 | SourceFormationFailed |
| T12 soft-compensated solver | MLP-V2206-S2-T12G0-SoftCompensatedHiddenBlockSolver |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| T12 soft-compensated carry | MLP-V2206-C4-T12G0-SourceStateCarry | -0.8690359989802042 | -0.313898762067159 | -0.11981745560963948 | 0.02599922815958659 | 0.05252563953399658 | 1 | 1 | 0 | 0.991019626458486 | 0.08990498649177898 | 0.4888954162597656 | 0.4717701820269107 | 1.0 | 1.0 | 1.0 | 0.08990498756368955 | 0.0 | 0.0 | nan | nan | 0.0 | nan | nan | 0.0 | 0.0 | nan | nan | SourceFormationFailed |
| T12 soft-compensated transport | MLP-V2206-C4-T12G0-OptTransport-lr150 | -0.8527073462804159 | -0.27093275388081867 | -0.07342727979024251 | 0.02912139892578125 | -0.005533814430236816 | 1 | 1 | 0 | 0.9813480575879415 | 0.11606706024495705 | 0.3772574265797933 | 0.37030162651532256 | 1.0 | 1.0 | 1.0 | 0.11606705964853366 | 1.0 | 0.049999999999999996 | 0.1799267679452896 | 0.3190169930458069 | 0.0 | nan | nan | 0.0 | 0.0 | nan | nan | SourceFormationFailed |


### Dataset-Level Source Readback

| v21_id | dataset | source_h100 | source_h400 | source_h800 | source_h1600 | source_h3200 | R3200/1600 | retained_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-C4-T10G0-AdamWBootstrap800OptTransport-lr150 | Fashion-MNIST | -0.03741264343261719 | -0.060205936431884766 | -0.471227765083313 | -0.8785429000854492 | -1.0106698274612427 |  | 0 |
| MLP-V2206-C4-T10G0-AdamWBootstrap800OptTransport-lr150 | KMNIST | 0.1665785312652588 | -0.08591318130493164 | -0.45815491676330566 | -0.7302879095077515 | -0.9157040119171143 |  | 0 |
| MLP-V2206-C4-T10G0-AdamWBootstrap800OptTransport-lr150 | MNIST | -0.36696505546569824 | -0.595177173614502 | -0.6795488595962524 | -0.6033817529678345 | -0.6536520719528198 |  | 0 |
| MLP-V2206-C4-T10G0-EarlyWarm400ThenOptTransport-lr150 | Fashion-MNIST | -0.9271005392074585 | -0.03145718574523926 | 0.17733120918273926 | 0.13943791389465332 | -0.11545848846435547 | 0.0 | 0 |
| MLP-V2206-C4-T10G0-EarlyWarm400ThenOptTransport-lr150 | KMNIST | -0.3280905485153198 | 0.10294008255004883 | 0.13653504848480225 | 0.004935026168823242 | -0.2212611436843872 | 0.0 | 0 |
| MLP-V2206-C4-T10G0-EarlyWarm400ThenOptTransport-lr150 | MNIST | -1.091415286064148 | -0.7081749439239502 | -0.3518040180206299 | 0.02655768394470215 | 0.1642310619354248 | 6.183937661031861 | 0 |
| MLP-V2206-C4-T10G0-EarlyWarm800ThenOptTransport-lr150 | Fashion-MNIST | -0.9307035207748413 | 0.023661375045776367 | 0.17415201663970947 | -0.009909510612487793 | -0.12717044353485107 |  | 0 |
| MLP-V2206-C4-T10G0-EarlyWarm800ThenOptTransport-lr150 | KMNIST | -0.3200794458389282 | 0.07100272178649902 | 0.060674190521240234 | -0.15137195587158203 | -0.2304295301437378 |  | 0 |
| MLP-V2206-C4-T10G0-EarlyWarm800ThenOptTransport-lr150 | MNIST | -1.1109353303909302 | -0.6830809116363525 | -0.30213451385498047 | 0.09551024436950684 | 0.08555686473846436 | 0.895787308505513 | 0 |
| MLP-V2206-C4-T10G0-EarlyWarmCarry-stop1200-lr50 | Fashion-MNIST | -1.0158993005752563 | -0.07163035869598389 | 0.02028369903564453 | -0.07717788219451904 | -0.11038780212402344 |  | 0 |
| MLP-V2206-C4-T10G0-EarlyWarmCarry-stop1200-lr50 | KMNIST | -0.4167715311050415 | -0.012398958206176758 | 0.026126861572265625 | -0.05759584903717041 | -0.1931222677230835 |  | 0 |
| MLP-V2206-C4-T10G0-EarlyWarmCarry-stop1200-lr50 | MNIST | -1.1298211812973022 | -0.7514834403991699 | -0.39321160316467285 | -0.03140556812286377 | 0.12634634971618652 |  | 0 |
| MLP-V2206-C4-T10G0-EarlyWarmCarry-stop800-lr50 | Fashion-MNIST | -0.9524070024490356 | -0.12234187126159668 | 0.04810500144958496 | 0.08758676052093506 | -0.03667581081390381 | 0.0 | 0 |
| MLP-V2206-C4-T10G0-EarlyWarmCarry-stop800-lr50 | KMNIST | -0.41435062885284424 | -3.075599670410156e-05 | 0.0403752326965332 | 0.06948566436767578 | 0.0033316612243652344 | 0.04794746160497385 | 0 |
| MLP-V2206-C4-T10G0-EarlyWarmCarry-stop800-lr50 | MNIST | -1.0973562002182007 | -0.7481944561004639 | -0.3536360263824463 | 0.13749098777770996 | 0.18332219123840332 | 1.3333396915788507 | 0 |
| MLP-V2206-C4-T10G0-OptTransport-lr100 | Fashion-MNIST | -1.0166095495224 | -0.164831280708313 | -0.0505298376083374 | -0.13313210010528564 | -0.1746518611907959 |  | 0 |
| MLP-V2206-C4-T10G0-OptTransport-lr100 | KMNIST | -0.3627365827560425 | 0.06470537185668945 | 0.0982670783996582 | 0.12476921081542969 | 0.14990222454071045 | 1.2014360238477413 | 1 |
| MLP-V2206-C4-T10G0-OptTransport-lr100 | MNIST | -1.061344027519226 | -0.714942216873169 | -0.34415483474731445 | 0.13266921043395996 | 0.11345219612121582 | 0.8551509106756162 | 0 |
| MLP-V2206-C4-T10G0-OptTransport-lr150 | Fashion-MNIST | -0.9571815729141235 | -0.07740044593811035 | 0.1994706392288208 | 0.11802589893341064 | 0.07869386672973633 | 0.6667508355444498 | 1 |
| MLP-V2206-C4-T10G0-OptTransport-lr150 | KMNIST | -0.36333024501800537 | 0.047521114349365234 | 0.10186457633972168 | 0.11131930351257324 | 0.14862537384033203 | 1.3351266954661207 | 1 |

_仅显示前 20 / 69 rows；完整 CSV 见 artifact。_


## Part D Terminal Preservation / KAN Mapping

| v21_id | source_h3200 | source_h4800 | R4800_over_3200 | C5_productive_terminal_source | preservation_status | blocker |
| --- | --- | --- | --- | --- | --- | --- |
| MLP-V2206-C4-T10G0-OptTransport-lr150 | 0.10594364007314046 |  |  | 0 | blocked_before_C5 | h3200_source_missing |


| mapping_status | blocker | best_metric_v21_id | KAN_source_channel_decision |
| --- | --- | --- | --- |
| blocked_before_KAN_mapping | MLP_h3200_source_missing | MLP-V2206-C4-T10G0-OptTransport-lr150 | KANMappingNotEntered |


## 修改记录

- 新增 `dgkan/fu/metric_solver.py`：实现 v22.06 S1 readout exact metric solver，输出 ActuationR2、projection residual、B2 transfer、metric/source-channel diagnostics。
- 新增 `dgkan/fu/jacobian_sketch.py` 与 `dgkan/fu/basis_channel_metric.py`：用于函数位移 alias readback 和 hidden/readout/reservoir source energy audit。
- 扩展 `dgkan/fu/mechanisms.py`：注册 M259-M268 v22.06 metric-as-geometry solver mechanisms，并声明 semantic contract。
- 新增 M262/M263：按 C4 blocker 修复方向补 T4 SmoothManifold / T5 LowNDS readout exact solver；用于验证 low-Sobolev/low-NDS target 是否能打开 source formation。
- 新增 M264/T6 DualMemorySource：从 train split short loss-cotangent 与 B1/B2 class-axis long memory agreement 生成 train-only target，测试 cross-split source memory 是否能补 C4。
- 新增 M265/T7 HiddenBlockSource：在 readout exact solve 后加入 train-gradient hidden residual block，并诚实记录 `hidden_residual_is_exact_solve=0` 与 hidden fraction，用来测试 source 是否被 readout-only 限制卡住。
- 新增 M266/T8 AdaptiveHiddenBlock：用 function displacement cap 限制 hidden residual，测试保住 C3 solver evidence 后是否仍能打开 source。
- 新增 M267/T9 C3GatedHiddenBlock：对 hidden residual 做 train-only C3 gate/backtracking；tiny semantic audit 中与 M264 fallback alias 已声明，不伪装 non-collapse。
- 新增 M268/T10 CompensatedHiddenBlock：先写入 hidden residual，再用 readout exact compensation 把当前 train split 的函数位移拉回 metric target，测试 hidden source 与 C3 solver evidence 是否能同时成立。
- 按计划 11.5 新增 T10 optimizer-state transport integration：不新增同义 mechanism，而是在 runner 中把 T10 source carry 注入当前 SGD flat gradient，并移除反 source 的负投影；trace 落盘 transport before/after projection。
- 按计划 11.6 新增 M269/T11 EarlyObservableSourceChannel：从当前 train batch 的两个 class-conditional split 生成 class-wise source axis，并用 mid-debt window 加权；方向不读取 validation/test/future，也不使用 source audit horizon。
- 按计划 11.7 新增 T11 I4 train-split control-relative gate：每次 FU 提交前用同 batch 模拟 FU candidate 与 SGD candidate 的 B1/B2/corrupt gain，只有 FU 在 train-only signal margin 上胜出才写入。
- 新增 v17 AdamW-compatible C4 integration：`v2206_adamw_bootstrap_then_optimizer_transport` 用 AdamW 走早期 800 步后切换 T10 transport，`v2206_adamw_control_relative_gate` 要求 T11 FU+AdamW 在 train split 上胜过 AdamW candidate 才写入；用于审计早期 h100/h400 负值是否来自 optimizer baseline mismatch。
- 新增 v18 phase-composed C4 integration：`v2206_early_warm_then_optimizer_transport` 在 h400/h800 前用 direct warm writer 形成 source，随后切换同一 T10 carry direction 的 optimizer-state transport；这是相位组合检验，不是继续提高 transport 强度。
- 新增 M270/T12 SoftCompensatedHiddenBlock：在 T10 的 hidden residual + readout compensation 内加入 train-only partial compensation mix 选择；目标是在 C3 过线前提下减少 readout compensation 对 hidden source 的抵消。
- 扩展 `experiments/run_v17_common.py`：补 v22.06 solver trace whitelist，新增 I1 early-source-warm-carry、I2 source-state-carry、I3 periodic boundary carry、SGD-bootstrap-then-carry 与 optimizer-state transport 分支，并写入 warm writer / source carry / hidden residual / compensation / transport diagnostics。
- 扩展 `experiments/run_v21_01_source_retention.py`：注册 `MLP-V2206-*` source-retention specs、C4 source-state carry specs、early warm carry specs、hidden-block specs、T10 optimizer transport specs 与 T11 early-observable specs，并修复 `--scope mlp` allowlist 缺口；第一次 v6 full allowlist miss 只作为 registration blocker 记录。
- 扩展 `experiments/run_v22_06_metric_solver_fu.py`：纳入 M264-M269 IDs、T10 optimizer transport 与 T11 early-observable spec IDs，并汇总 early-warm/source-carry/hidden-residual/compensation/transport/T11 class-axis diagnostics；summary 只读 runner artifact，不生成实验数值。
- 调整 `dgkan/kernels/fused_rbf.py` low-D launch block sizing；D-RAT/D-RBF officialization 使用 production manual CE trainpath probe 复核，不采用 micro benchmark 替代。
- 新增 v22.06 runner：S0.13 truth gate、metric solver summary、terminal preservation gate、KAN mapping gate、D-RAT/D-RBF officialization、efficiency reconfirm、finalize。

## 分析 / Insight / 证据链

- S0.13 的核心证据来自 clean unzip packet 内运行；若 clean unzip/import/contract 任一失败，route 会停在 `R0-CodeMetricInvalid`。
- M259-M261 的 semantic contract 均标记 `is_proxy=0`，因为 update 由 readout exact solve 构造，并通过函数位移 alias matrix 读回；它们不是 v22.05 flat-gradient metric projection。
- C1/C3/C4 分开判定：C1 看 train-only B2 transfer 和 source/reservoir leakage，C3 看 ActuationR2/projection residual，C4 才看 matched-control source retention。任何一段失败都不进入 promotion。
- 本轮最佳 functional row 是 `MLP-V2206-C4-T10G0-OptTransport-lr150`：source_h3200_mean=0.10594364007314046，但 source_h800_mean=-0.01733402411142985，C4_h3200_source_pass=0；因此只能写作 C3/C4 进展，不能写成 retained source promotion。
- 第一次 v6 full 只产生 42 个 matrix rows、缺少新 M264 specs；这是 allowlist registration blocker，不作为 M264 科学负结果。修复 allowlist 后 fresh v6/v7/v8/v9 均重新落盘。
- T6 dual-memory + I1 early warm carry 能把 best h3200 推到正数，但 h100/h400/h800 仍为负，说明它打开的是 late/local recovery，不是 C4 要求的连续 early source chain。
- T7 hidden block 是本轮 source 读数最强的方向：source_h800_mean 与 source_h3200_mean 转正，但 ActuationR2/projection residual/B2 破坏 C1/C3，因此不能把 hidden residual 写成 metric solver success。
- T8 adaptive cap 与 T9 C3 gate 尝试修复 T7 的 solver evidence；它们能恢复 C1/C3，但 hidden residual 被压低或归零，source 又回到负值。
- T10 readout compensation 保住了 C1/C3 solver evidence，并在 trace 中落盘 nonzero compensation norm / scale / residual ratio；但 early source chain 仍未打开，说明“hidden residual + exact readout compensation”只解决了 solver-evidence 张力，没有解决 pre-h800 source formation。
- T10 periodic boundary carry 与 SGD-bootstrap-then-carry 均为 fresh 3-dataset smoke：它们产生了若干 dataset-local late positives，但没有让 h100/h400/h800 与 h3200 同时满足 dataset-invariant C4 gate。
- T10 optimizer-state transport 是 plan 11.5 的 optimizer-conflict fallback：它只改写当前 optimizer gradient，不额外直接 apply FU；若 transport projection after 明显上升但 C4 仍失败，则说明 source washout 不只是当前 step 的反 source 投影。
- T11 early-observable source-channel 是对 C1 的直接修复：它先测试当前 train split 是否存在可重复 class-axis source，再进入 warm carry / optimizer transport；若 T11 仍不能让 h100/h400/h800 同时转正，则 blocker 会继续定位在 early source observability，而不是 terminal preservation。
- T11 I4 control-relative gate 是对 T11 的 follow-up：若它拒绝大多数 FU 或通过后仍 h100/h400 为负，则说明当前 train split B1/B2/corrupt proxy 仍不足以预测 matched-control source formation。
- AdamW-compatible C4 是对当前 blocker 的直接检验：若 AdamW bootstrap 打开早期 source 但 AdamW-relative gate 没有 FU margin，则只能说明 optimizer baseline 解释了 early source，不能说明 FunctionalUpdate 本身达成 promotion；若二者都失败，则排除简单 optimizer mismatch。
- v18 phase-composed C4 检验的是“先形成早期 source，再写入 optimizer state”是否比纯 transport 更接近 C4；若它仍不能让 h100/h400/h800 连续过线，则 blocker 不是单纯相位切换，而是 target/source observer 本身尚未产生早期 dataset-invariant signal。
- T12 soft compensation 检验的是 T7/T10 之间的张力：若 partial compensation 能保住 C1/C3 但仍无法把 h100/h400/h800 同时转正，则早期 source blocker 不是 full readout compensation 单独造成。
- Dataset-level evidence 仍不稳定：T7 让 KMNIST 与部分 Fashion/MNIST horizon 转正，但 MNIST h100/h400 与 Fashion h3200 仍负；T9 C3-gated rows 在 MNIST h3200 局部转正，但 KMNIST/Fashion h3200 仍负。
- low-NDS/SmoothManifold target 修复没有解决 source formation：T4/T5 仍表现为 ActuationR2 高、B2 transfer 正，但 matched-control source 读数为负。
- 因 C4 h3200 source gate 未过，本轮没有启动 h4800/full 4GPU continuation；这是 fail-closed，而不是漏跑。
- Terminal preservation 与 KAN mapping 没有在 h3200 source gate 之前强行启动；如果复盘显示 `blocked_before_C5` 或 `KANMappingNotEntered`，这是按计划 fail-closed。
- D-RAT/D-RBF officialization 只读取/重跑 official manual CE trainpath probe；micro-near rows 不会被写成 production fused pass。

## 人工复盘笔记 / Observation / Insight

本节是实验后的手写复盘，不由 finalizer 自动生成。它记录我在推进 v22.06 时的观察、判断和失败转向，目的是让后续继续实验时能看到“为什么这么做、为什么停、为什么不 promotion”，而不是只看到 CSV 指针。

### 总体判断

v22.06 的主线其实已经从“代码/official basis blocker”转成了很窄的 functional blocker：S0.13 通过，D-CHE/D-FOU 通过，D-RAT/D-RBF production official 也通过；剩下的问题不是工程路径没接上，而是 metric solver 生成的 train-only update 没能形成 C4 要求的连续 early source chain。

最终最佳 row `MLP-V2206-C4-T10G0-OptTransport-lr150` 给了一个很典型的假希望：`source_h3200_mean=0.10594364007314046` 是正的，但 `source_h800_mean=-0.01733402411142985`，所以它不是 retained source，而是 late/local recovery。这个差别很关键。若只看 h3200，会误以为 route 接近成功；但 C4 要的是 h100/h400/h800 到 h3200 的连续、matched-control source chain。这里 early source 没打开，因此 C5、h4800、KAN mapping 都不能进入。

我现在认为 v22.06 的失败点不是“没有足够强的 functional update”，而是“当前 train-only target/source observer 不会在足够早的 horizon 上稳定地产生 dataset-invariant source”。继续放大 lr、延长 window、或者反复 transport 同一条 source direction，很可能只会制造局部 h3200 positives，而不会解决 early chain。

### 分阶段观察

**1. Readout exact solver M259-M263：solver 不是 source。**

S1 readout exact solve 的意义是把 v22.05 那种 flat-gradient proxy 升级为函数位移可审计的 metric solver。它确实让 C1/C3 这类 solver evidence 可读，也避免了“伪称 JTGJ 精确”的问题。但实验很快说明：函数位移 solve 只是把 update 写进目标函数空间，不等价于 matched-control source formation。很多 row 的 ActuationR2、B2 transfer 都能看起来合理，但 h100/h400/h800/h3200 source 仍然是负的。

这个阶段排除的是一个早期误区：不是只要 solver exact、metric 名字正确，就会自然打开 source channel。source channel 是额外的动力学问题。

**2. T4/T5 low-Sobolev/low-NDS：平滑/低 NDS 不是缺失钥匙。**

T4 SmoothManifold 与 T5 LowNDS 是按 C4 blocker 做的自然补充：如果 source washout 来自高频或 unstable displacement，那么 Sobolev/low-NDS target 应该改善 early source。结果没有。它们仍是高 C3、低 source 的形态，说明当前 failure 不是简单的 “target 太不平滑” 或 “NDS 太高”。

这条线我会停止继续扩展。再加一个 G-family metric，只要 target/source observer 不变，大概率还是把同一个负结果换名字。

**3. T6 dual-memory：能造 late recovery，但不是 early chain。**

T6 dual-memory 的出发点是让 train split short loss-cotangent 和 B1/B2 class-axis long memory 做 agreement，试图引入跨 split 的 source memory。它比早期 S1/SmoothManifold 更像一个真正的 source repair，因为它能把部分 h3200 推到正数，例如 T6 early warm 行出现 positive h3200。

但它的 h100/h400/h800 仍为负。这是一个重要观察：T6 的信息不是完全无效，它能在后段形成某种 local recovery；只是这个 recovery 来得太晚，且不像一个稳定的 early source channel。后续如果要复用 T6，我不会再把它当主 target，而会把它当一个 late diagnostic，帮助判断“source axis 是否存在但启动太晚”。

**4. T7 hidden block：source 可以被打开，但会破坏 solver evidence。**

T7 是 v22.06 里最有价值的负结果之一。它把 readout-only 的限制打破，在 readout exact solve 后加 hidden residual block，结果确实能把 source 读数推强，`MLP-V2206-C4-T7G0-SourceStateCarry` 的 h800/h3200 都能转正。但代价很硬：ActuationR2/projection residual/B2 transfer 崩掉，C1/C3 不成立。

这说明“没有 source”不是绝对事实。模型里存在能打开 source 的 hidden direction，只是那个 direction 没有被当前 metric solver 约束住，或者说它对函数位移/transfer evidence 太粗暴。这个观察改变了后续路线：问题不是找不到 source，而是找不到同时满足 C1/C3 和 C4 的 source。

**5. T8/T9：把 hidden residual 管住以后，source 又消失。**

T8 adaptive cap 和 T9 C3 gate 是为了修 T7 的副作用。它们做到了部分目标：C1/C3 恢复，ActuationR2/projection residual 不再像 T7 那样坏。但 hidden residual 被 cap/backtracking 压低后，source 也一起回落。这个阶段暴露了核心张力：当前 source-carrying hidden component 和 current metric actuation evidence 是相互冲突的。

我的理解是，T8/T9 不是失败得“没信息”，而是证明了 current C3 gate 太局部：它保护了当前 train split 的函数位移，却没有奖励早期 source trajectory。它只知道“别坏掉”，不知道“朝 source chain 方向活下来”。

**6. T10 compensation：C1/C3 和 hidden source 的折中仍然不够。**

T10 的思路是先写 hidden residual，再用 readout exact compensation 把函数位移拉回 metric target。这个设计比 T7/T8/T9 更接近目标，因为它试图同时满足 hidden source 和 readout-level C3。结果确实更稳，trace 里也有 nonzero compensation norm / residual ratio。

但 T10 仍然没打开 early source chain。我的直觉是：readout compensation 在保护 C3 的同时，也抵消了 hidden source 的一部分有效输出；而剩下那部分 source 在 matched-control 比较里不够早、不够稳。也就是说，T10 解决的是“hidden update 太破坏函数位移”的问题，不是“source observer 太晚/太弱”的问题。

**7. Periodic carry、SGD bootstrap、optimizer transport：optimizer 不是唯一解释。**

T10 后我尝试了三类 integration 修复：periodic boundary carry、SGD-bootstrap-then-carry、optimizer-state transport。它们都在不同程度上解决“怎么把 update 注入训练动力学”的问题，但共同失败点一样：产生 dataset-local late positives，没产生 dataset-invariant early positives。

optimizer-state transport 是最接近成功的一条，因为最终 best row 就是 `MLP-V2206-C4-T10G0-OptTransport-lr150`。它能让 Fashion-MNIST/KMNIST 等局部 horizon 转正，也能给 positive h3200 mean。但它的 h100/h400/h800 mean 仍然没有连起来。这个结果排除了一个简单解释：source washout 不只是普通 optimizer 当前 step 里有反 source 投影。即使把 optimizer state 往 source direction transport，early source 还是没有稳定出现。

**8. T11 early observable / control-relative gate：当前 train split proxy 还不够预测 source。**

T11 尝试更早地定义 source axis：从当前 train batch 的两个 class-conditional split 生成 class-wise source axis，再用 mid-debt window 加权。T11 I4 又进一步要求 FU candidate 在同 batch 的 B1/B2/corrupt gain 上胜过 control candidate。

这条线的意义是把问题从“怎么 transport”往前推到“source 是否能在 train-only proxy 里被观测到”。结果还是没有打开 C4，说明当前 B1/B2/corrupt proxy 对 matched-control source formation 的预测力不足。它能解释局部更新是否看起来安全，但不能保证 h100/h400/h800 的 matched source 变正。

这个负结果很重要：后续不能只把 gate 写得更严格。严格 gate 只会减少写入次数，不会凭空创造更好的 source observable。

**9. AdamW-compatible C4：baseline mismatch 被部分排除。**

因为 h100/h400 常常负，我怀疑过是不是 SGD/AdamW baseline mismatch 导致早期 source 比较不公平。于是做了 AdamW bootstrap + transport、AdamW-relative gate。结果没有带来 C4 pass。

这个阶段的结论是：optimizer baseline 确实会影响局部 source 读数，但不是主 blocker。若主因只是 early optimizer mismatch，AdamW bootstrap 应该至少稳定改善 h100/h400；实际没有。因此下一步不应继续换 optimizer 外壳，而应回到 source target/observer 本身。

**10. v18 phase-composed warm then transport：相位切换也不是解。**

v18 的想法是很朴素的：先用 direct warm writer 在 h400/h800 前形成 source，再切换到 T10 optimizer-state transport，看看是不是 “formation phase” 和 “retention phase” 需要分工。fresh run 后最好 row 是 `MLP-V2206-C4-T10G0-EarlyWarm400ThenOptTransport-lr150`，`source_h800=-0.012645920117696127`、`source_h3200=-0.05749619007110596`，仍然失败。

这里我排除了“只是切换时机不对”的解释。phase composition 没有把 h800 拉正，说明 early writer 本身的 target/source axis 仍不够强，而不是 transport 接得太晚。

**11. v19 M270/T12 soft compensation：full compensation 不是唯一坏因。**

T12 是为了测试 T7/T10 之间的核心张力：是不是 full readout compensation 抵消 hidden source 太多？所以我加入 partial compensation mix 选择，希望在 C3 过线前提下保留更多 hidden residual。结果 T12 carry 的 `source_h3200=0.05252563953399658`，但 `source_h800=-0.11981745560963948`；T12 transport 的 `source_h3200=-0.005533814430236816`。更关键的是，train-only selection 仍选择 `compensation_mix=1.0`。

这个结果说明 partial compensation 没有在当前 objective 下胜出。换句话说，若不改变 source objective/observer，只在 compensation mix 上做连续调参，route 仍会回到 full compensation 或 C3/C4 二选一的困境。

### 我现在确信的几件事

- 不能把 h3200 positive 单独写成 source success。v22.06 最容易误判的地方就是 late positive。只要 h100/h400/h800 没形成连续链，h3200 正数只能叫 local recovery。
- 当前最强 source direction 来自 hidden residual，但它天然会破坏 C1/C3；当前最稳 solver direction 来自 readout exact compensation，但它不能形成 early source。v22.06 真正的难点就是这两者没有被同一个 train-only objective 对齐。
- Dataset-level hard negative 不是固定一个数据集。不同 mechanism 下 MNIST、Fashion-MNIST、KMNIST 会轮流成为断点；因此不能做 Fashion-only 或 MNIST-only schedule tuning，否则很容易过拟合 smoke。
- T10 optimizer transport 的价值是证明 “optimizer state 可以承载 source”，不是证明 route 成功。它应该作为后续机制的 integration substrate，而不是继续调 strength。
- T11/I4 的价值是证明当前 train-only gate 不够。下一步需要一个更早、更直接的 source observable，最好在 h100/h400 前就能从 train batch 中读出，而不是依赖 h800/h3200 audit 之后回看。
- Terminal preservation 和 KAN mapping 现在不该碰。C4 没过时进入 C5/KAN，只会制造不可审计的 downstream noise。

### 下一步我会怎么修

我不会继续做 T10/T12 的 scalar sweep。下一条值得尝试的路线应该是重写 early source observer，而不是调已有 observer 的注入方式。

更具体地说，需要一个 train-only pre-h400 source objective：它必须在当前 batch 或短 memory 上构造一个 source axis，并且这个 axis 要同时满足三个条件：第一，能在 hidden space 中激活 source-carrying direction；第二，能被 readout compensation 后保留足够投影；第三，在 MNIST/Fashion-MNIST/KMNIST 三个 dataset 上不要互相换 hard negative。当前 B1/B2/corrupt gain 还不够，可能需要把 class-axis source 与 loss-cotangent 的局部 curvature、calibration debt slope 或 margin transition 绑定，而不是只看 instantaneous split agreement。

一个可审计的下一步候选是 “pre-h400 source identity test”：在每个 train batch 内构造两种独立 source probes，一种来自 class-conditional margin/cotangent，另一种来自 calibration/debt transition；只有两者在 hidden residual direction 上同向，才允许写入，并且 write 后立即用 readout compensation 的 residual-source projection 做 veto。这个机制仍然不能读取 validation/test/future，也不能使用 source audit horizon 作为方向；它只能使用当前 train batch 与短 EMA state。若这个 pre-h400 identity test 仍不能让 h100/h400/h800 同时转正，就应考虑 v22.06 的 metric-solver route 暂时 blocked，而不是继续堆 target family。

### 本轮不应该写成什么

- 不应该写成 “T10 已经打开 h3200 source”。正确写法是 “T10 optimizer transport 产生 late/local h3200 positive，但 C4 early chain failed”。
- 不应该写成 “T7 hidden block 是成功机制”。正确写法是 “T7 暴露了 source-carrying hidden direction，但破坏 C1/C3，不能 promotion”。
- 不应该写成 “T12 证明 compensation 无关”。正确写法是 “在当前 train-only selection 下 partial compensation 没胜出，full compensation 仍被选择；这只能排除简单 compensation-mix sweep”。
- 不应该启动 h4800/full/KAN 来碰运气。C4 gate 失败时继续下游实验，计算成本高而且科学解释差。

## Artifact Index

| artifact | exists | size_bytes | sha256 |
| --- | --- | --- | --- |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/00_README.md | 1 | 104 | eb854a50795d97af9b21319b646f19229148d3a6d42287a71b49a4935f7196a9 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/__init__.py | 1 | 447 | de3d83266f16ee40cc6e8994331904155cdd3c48e593c525a1831b042aae2386 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/__pycache__/__init__.cpython-311.pyc | 1 | 766 | 479436449c7a2b4ca948379fde5fb94e65d2946167a4b712254818ffd6e9b76f |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/__pycache__/config.cpython-311.pyc | 1 | 1171 | 65d1ea3730019276f94812ae1fb6aff7f89dd38cd3b1f30a613b713ff3970e23 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/__pycache__/contracts.cpython-311.pyc | 1 | 6199 | d85d08e0f03b2afa509245a2eb26654fae326b5ab7a615c4639950527456de0a |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/__pycache__/registry.cpython-311.pyc | 1 | 3204 | 76410938aec98572aa44b86ad75b658ba0fc2c27ef00d28dc0e3637cb2cb1021 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/__pycache__/specs.cpython-311.pyc | 1 | 3449 | abec01216ca9219a9dceb7a897f3cede16f7df9baa0f31a07c9c15648597785a |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/artifacts/__init__.py | 1 | 251 | 1d57257123b41115b756dbad3679ed70fd0f0233b91bf9ffbd0fa74745cfeb3b |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/artifacts/__pycache__/__init__.cpython-311.pyc | 1 | 396 | cbe546cd462029badad3551feb6f29bb1c0664e02500d0a609c0ae27988b9048 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/artifacts/__pycache__/audit.cpython-311.pyc | 1 | 1980 | 99a73b942d43a2e654f6ffa2e9e870a167963dd1ddc1ea3b11073fd7900b494d |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/artifacts/__pycache__/route.cpython-311.pyc | 1 | 1358 | d3b0710dbb3cefcb1526aa980c0428d1141e8c8f5ad097fe2b5e17130af33eea |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/artifacts/__pycache__/writer.cpython-311.pyc | 1 | 6185 | bad6776383f09d7d2bf7b337b1a60479e4dd46633c64ba74ed9e858c2c2b9783 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/artifacts/audit.py | 1 | 1047 | bc6790ae6347f99be618fd493f7aef99e6905cd0e3471120b1235b193e2e8989 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/artifacts/route.py | 1 | 1008 | 02319fa81e36de7d5a9a64b5e074fa1e6d183b6d2993435b05c165d5c6629d08 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/artifacts/writer.py | 1 | 2060 | 48a630b11379ad68af48f398e96ad58aca04232a80849bf1e42bac583e96efdf |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/config.py | 1 | 1060 | 87e5efff2492f53df84c3f5a7c6cb94ca1f9d7029be39feaa855d5d8eb377e30 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/contracts.py | 1 | 3255 | 03d0ce0f729d59505df8713e4b54d168e387946e7cc8ce036f39c644e133c465 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/diagnostics/__init__.py | 1 | 403 | 4a43eb55dbae11ca2a35739a12a8e4b17c0d0a61ac4445320463aba676c87dc1 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/diagnostics/__pycache__/__init__.cpython-311.pyc | 1 | 536 | 27c21745f99d1444be81936cf023e61d3e738248d2378dbea7cfe74b82135809 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/diagnostics/__pycache__/basis_workspace.cpython-311.pyc | 1 | 92894 | fed87fdc0bc54b0fbf7ec564f8695ec8bd2e5a1847ae4772ed00c7730c6d23e7 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/diagnostics/__pycache__/classic_basis.cpython-311.pyc | 1 | 14781 | 9a50a850d2656208de3b0a4533a9d09b1907ffa0f7ef434ba2b5abd51b501310 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/diagnostics/__pycache__/linec.cpython-311.pyc | 1 | 1922 | fcafd868812c961683935f9b762e373f280848dedf868751d1b8c99bd562dbc6 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/diagnostics/basis_workspace.py | 1 | 96053 | 89446acd9d467edf42fa8716986cfc56a5059ee925cac2dc790f98402e3b7437 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/diagnostics/classic_basis.py | 1 | 12451 | 3310364893cf4e9513406d0f0787c8993b06c3f1d6bcc41d801e45b3c0a36f84 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/diagnostics/linec.py | 1 | 1310 | af17f41dc3b33cca01cc4eaefef75e3b471730911d8784a0783579a6f8f845b8 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/__init__.py | 1 | 41 | 881010633493337f5531dde4c43ae77072d9b1d2c73973eae4f15cbaa9ec67c4 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/__pycache__/__init__.cpython-311.pyc | 1 | 184 | 190ab66200ff7c92b46fab6b8c358fbe1c163afd9058fcfdaaa66b836cd293ab |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/__pycache__/kernel_census.cpython-311.pyc | 1 | 371 | 1e61b2a00bddfb07810c092bb613f60b6f38c520a347881e7e26853223c703e1 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/__pycache__/memory_timer.cpython-311.pyc | 1 | 433 | db88478a286e7f82958738e7f983ced8ae2057e60df06484bf2a6cd0dd00bb5c |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/__pycache__/phase_timer.cpython-311.pyc | 1 | 467 | d3a8df6489dc954591e13d9699454322ece4b7f2a1beecc602d67143e9498595 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/__pycache__/profiler.cpython-311.pyc | 1 | 389 | c5a460ffec2f9139aee697071c0a2fc6a7b8104ec40056d0d9cc75f038f8b695 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/__pycache__/repair_registry.cpython-311.pyc | 1 | 1015 | d88cd3c126916d91e813879842688b3433b2c9f44cada1e366ca8bed24b79a04 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/__pycache__/same_param_mlp.cpython-311.pyc | 1 | 905 | b4f05a1f41aa3163b8e99a4903ea0d02d95662cd45b39fc5d69eaf00214eb2ac |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/kernel_census.py | 1 | 177 | 0e515e1a21839430904b943a6ba17f7cfa9024d456f4feda32a34b98a4f151d6 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/memory_timer.py | 1 | 305 | e9d0bb55d2d39ef0e9a22aab1e6f11826189cfeb3738be357fd103a1351cf8af |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/phase_timer.py | 1 | 373 | 81aca9f04546c482b888bc0aeea595eac1983f48d1bd5dea4cc70bc3c1493669 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/profiler.py | 1 | 194 | c2f9813b310d81cadeedd7ed23c5e11bda1f3e3227968ec815e8d579b3cb8a6b |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/repair_registry.py | 1 | 615 | 33f4293f5f1ad4d5900992d6ceb1771e0c9187b5f7d2427bccef840c58a26658 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/same_param_mlp.py | 1 | 464 | 2f34dbe22368505145397ed95fd0b1c6016ddd937b86ff4418a409c72f339f86 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/external/__init__.py | 1 | 36 | f95b9474abcdda646d518e002c522cac53eada80e7dd740d0fd60af5861a5451 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/external/__pycache__/__init__.cpython-311.pyc | 1 | 178 | cba86e4540d5902e441447005f1f16a9129a384aebdec72577316b1b7e49c615 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/external/__pycache__/baselines.cpython-311.pyc | 1 | 324 | 64f40a967b1200d7afdecf3f47fed3e5911adad19ec93f6841272bd93b1e1f3c |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/external/__pycache__/counters.cpython-311.pyc | 1 | 535 | bcc64e8398b6f072b4c96f02d1f1c0adb538161dfc966dcf0bbd01c1e796be19 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/external/__pycache__/kanbefair_adapter.cpython-311.pyc | 1 | 2088 | 2c29ddb05bc7caa57e51ae2cd72b6591534d5c913706471bf6550506930e06e7 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/external/baselines.py | 1 | 151 | 741ea1e43859720ad9ca0e3ac79fe720082379db5da32c57ef2e610f6e828f07 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/external/counters.py | 1 | 222 | 074275dff2e16b7029299ea2ce182c5f250330a117edeeed8fdc66e62b88ee0f |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/external/kanbefair_adapter.py | 1 | 1143 | f6de772edee26e57c7b41c4ae5556212cced228415adfaadf767ff2b8f2df906 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__init__.py | 1 | 51 | 4b1457bbb0b8007de6ab30711924ce6c968bf453ac797ff63b2a77b45c846162 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/__init__.cpython-311.pyc | 1 | 341 | 423ed3203ceff2739f0d63464e15e359b3f7d269d1f319cf5c4e433ab9ef8136 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/audit.cpython-311.pyc | 1 | 1452 | 463eec73958d081a2aed87971efff20c2423a244632e0ca8223005a6ee9763ef |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/basis_channel_metric.cpython-311.pyc | 1 | 8683 | f0d943a56f85292a71b51144f75d79e2373864c59f613408a14f2f75296684fc |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/carriers.cpython-311.pyc | 1 | 1835 | 7e2148842214ff9596b73cd08089bfe0fab2c73b3671233e63e4aa7098947477 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/controls.cpython-311.pyc | 1 | 3233 | 2d1356d113d1d4f9dff371fd5ebe8bb55c7787b117a4b84aa3ec6530234e13dd |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/core.cpython-311.pyc | 1 | 11749 | 20b59cf84862462b052dead29f677008fbb1edf869348cff429617d4d4cc4c2a |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/debt_accounting.cpython-311.pyc | 1 | 5983 | 425aeaf6d00aa4abb64fa7ef669373559da3f90ddf9b323cfcdc75685a2f1c4a |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/diffeomorphic_target.cpython-311.pyc | 1 | 7793 | a9c5be5e33665b14f60b2905f0a76bdeefba57de014fa5731b94e238b1fc0660 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/fisher_metric.cpython-311.pyc | 1 | 1659 | 87ad817367dc405a2b17e0b2cf000de83ace922df16ca3e84cf5a6844b5ba659 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/function_space_actuation.cpython-311.pyc | 1 | 7169 | a0fd0d4e06dd50d24df44029e75444d9e0a86a2457b1dde26e4fadd7255e1754 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/function_space_metrics.cpython-311.pyc | 1 | 19088 | 1e1e3d1e5dd94bed414ef753b1833bca48ac7fadae214adbceb05900daba8b66 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/jacobian_sketch.cpython-311.pyc | 1 | 9831 | c600d473bbe09b92848a4d9ff1a24cca0a56ee065d320811984b3be1739b6ca7 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/kan_source_bank.cpython-311.pyc | 1 | 3473 | f7b95cff808486557afe65e7ac7a4512101b5db1efb84dcbd08d18f32071bb2e |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/linec_readback.cpython-311.pyc | 1 | 398 | e915eba533d46d7942410732e60b5997c53c9e54a7aa101b3be90cfe867f6b94 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/loss_interface.cpython-311.pyc | 1 | 4703 | f85b667ae6d56ceb89c19ba4c130d4da2f1d47bc975f31362cd976c0d68ea2ec |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/matrix_block.cpython-311.pyc | 1 | 1316 | 186dcca8a97206079b749d6fc8c4ee83f44c9f8996010afdb5feba837bb7d858 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/mechanisms.cpython-311.pyc | 1 | 253378 | 9288990bf5f88650039f44d855abeddf4d40bca99b7d0547e24e5d1fc151e839 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/metric_projection.cpython-311.pyc | 1 | 1996 | 99f7dfd35c22fb102af799a5e262d890a1ec44b4410783a9cf12b65384e6bbbc |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/metric_solver.cpython-311.pyc | 1 | 54409 | ac774b1e309eb38363234ea697e65fbef54d1ef7c7b86165583ea6da80591ea0 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/optimizers.cpython-311.pyc | 1 | 5372 | 27d3b31f492c5ee5b5aa7a8a95ac811237dd0e5ae08d0a87919590fbe2a39389 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/poprisk_snr.cpython-311.pyc | 1 | 1231 | 48848fcb729138840a6c73bd8a3257fed1a7d0b6c1321d9979952c706041d76e |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/poprisk_source.cpython-311.pyc | 1 | 333 | 7f037697e553b24203d2b39da8f51cd28f5dc017d6c2f5daa35ebae34e02da85 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/precommit_selector.cpython-311.pyc | 1 | 7539 | 3f64baf917f86a2f3f918039e9aa25e16cfb1b4eff2879e1115a6321beb1ee50 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/rkhs_metric.cpython-311.pyc | 1 | 1260 | c8f2a628a126d6ab1bf9657769632ecf2a153585c831a10535bd69828da047a9 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/slow_state.cpython-311.pyc | 1 | 1024 | 24f3ce03b98d639408304533307dbb1c34942070246f0f3fb5f790c2d89e34a5 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/sobolev_metric.cpython-311.pyc | 1 | 1391 | 6252147d659775f692b6dada828f10e02a8b6a1e27300e7083dff8d7baa07c56 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/source_chain.cpython-311.pyc | 1 | 8786 | d86a7a249c7167c41a3750984a7b3fa6c934f992cb0a28225d20052fbc49d292 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/source_channel.cpython-311.pyc | 1 | 12626 | e9617351dc62cb810c05a1f0ee1ffce93fa5a18b07970b7aedb2a4677fd4e1d1 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/source_preservation.cpython-311.pyc | 1 | 2659 | f32ef205c772b84dd46ed1c3ebfcb00cd24a25cd8e8a377377ee3765e2c54402 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/source_state.cpython-311.pyc | 1 | 1639 | d1b96687060d4d865ea55602886e8626838c166f6725766c93d08f68be311d71 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/terminal_collapse.cpython-311.pyc | 1 | 7865 | e1b0ec7ed602d2458666c1fc6c5da5d91dee0b06a3cff4e18adac4bab4c7c852 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/terminal_erosion.cpython-311.pyc | 1 | 7066 | 4946726ead1a5f22ebcaccc75c5e364a5530d5e74e4936eb580441c2c0533c7b |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/terminal_retention.cpython-311.pyc | 1 | 10265 | c93893ab8cf9f743908c9b2a5804583725b2841db476bfa6e81fced3e95c080d |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/types.cpython-311.pyc | 1 | 1829 | e447b571e6340d88e69f80fabc54cb95eea8dcfcf8492e8ccfa773d4ffbd593a |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/update_semantics.cpython-311.pyc | 1 | 1969 | 1a7f0ea075810593229c60b00d270a7f6fb51e79c07aa02409e3e012a6c5f46e |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/audit.py | 1 | 956 | 1cd77be8d23994bcda66709d9637a8f2ad7b3baf2f330671421e8f52529f2c31 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/basis_channel_metric.py | 1 | 3814 | 3d8be1a4ee373448dfc89bd2a7d83671ba271bdf4385679fbe2675c428009d99 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/carriers.py | 1 | 956 | b02a55d3765a7f7b3220b421bd4e28d458a327ee26f7148acb1a234b2f4272ac |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/controls.py | 1 | 1776 | 112c6785ce87359c69fe7b194c8e86781ae57c05d2bb60fc05a21817d0127547 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/core.py | 1 | 5138 | 41e7dec2fdda1948caddb9ef1f930f1d26f6df8d76bed89899187f826cbee378 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/debt_accounting.py | 1 | 2727 | efa1948dde33255625609f7d71641649a3eea00b63b54bcc6c1a36ef54aab265 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/diffeomorphic_target.py | 1 | 4558 | 9400bb4b956727808ed8ae18831f1c054c36dae7ceb6d7e39c79bd5b93b771b3 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/fisher_metric.py | 1 | 717 | fa5c2ce8ba853aa5dac55bfe6299be4e2c5ee27d90bea8175f0330aba3f44f84 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/function_space_actuation.py | 1 | 3811 | 5b3ffc343f312b07dbcddf450fe1bc2ee51dd1ece430132af6221ff9b3f28ac2 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/function_space_metrics.py | 1 | 8791 | f420385b2025c37cd55581773c2e04138b7fef69b4c2f186f5ae0be604c34460 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/jacobian_sketch.py | 1 | 4186 | f498d5337cb8b7a980c98ab21915abf82fc6669515dae1d1a8a8dd9813faa6bc |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/kan_source_bank.py | 1 | 2525 | 7e8855bb0fddddc2b6fac0c757471b2610583053567c714e9a3e86d4b1fc3d7e |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/linec_readback.py | 1 | 193 | e59a29285e5b4f9b5313866fde6b29f88264674e0875c506e9e051df1a9a0bda |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/loss_interface.py | 1 | 2169 | 049074e61dca605ae373c3e4e04b469dc991a0b6dda958de82e69b6b4e45bb6d |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/matrix_block.py | 1 | 447 | 1d737d81d863a637f35a8af9f6ea5666af7925a70db22bfcf4c8c49f2b72e011 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/mechanisms.py | 1 | 232415 | 2fe529b5ebab50460d3327470ee8a07a31b2b05ac3d03e93ba56a1409180646a |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/metric_projection.py | 1 | 955 | 3ca37b93f912a3e65df384762471a5f03297d435956ce52a4ad5ffd8acd0184c |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/metric_solver.py | 1 | 37922 | c792cf51e8bcf7cb6fc726415eb3d2a0fe4b767ab93c974da11c57841c273be6 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/optimizers.py | 1 | 2597 | 6937c978a20a82491a2cfd2a5cae602e8903612a157efa4026d8c9a478d63d9d |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/poprisk_snr.py | 1 | 502 | e9018a4c58f04b8ea4eff85f143a5fe66db23bf945b4d9525f299a3fa9bf3ff1 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/poprisk_source.py | 1 | 169 | 54d19c4034eda008d1e3f0acc11a4c4ae78ff7807708218b8fb7f9101ac19e48 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/precommit_selector.py | 1 | 3920 | 3a1ad79b1f5e14b4d02a4512816d2fa7d4c5e9bf61dd6ca8ffd9501edc89bd46 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/rkhs_metric.py | 1 | 476 | 64936156e3017f72a07d18ed9d3e52e8958208531ef1d553d3c4874f625e282b |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/slow_state.py | 1 | 390 | c24ae16a02777eb4aa9281573666b5a94e32f169673eec6c3347ab2ca1c3c7e4 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/sobolev_metric.py | 1 | 530 | 73deffe772b9050415cd9b79e89087b42fd040611aa36c483594c18d7956e7b3 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/source_chain.py | 1 | 6982 | 8f78f54d28e195235e3319340eada9a821a97afc4e6bf7fbe386db80cd39d7ac |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/source_channel.py | 1 | 5671 | 189dc7ee1eaf9bf1796a92f3d4535838d00c439a5848677f95bfd5ccc4c7552a |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/source_preservation.py | 1 | 1325 | cdbc175914887e4b22f2eddb66561c70588c020e71d05862d36e2c596c09f20a |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/source_state.py | 1 | 725 | aa882d5c755920fde5961d17beec6eae913a5998c7fc18144d4f912371ff4140 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/terminal_collapse.py | 1 | 5961 | ace1528b37855807de9925ebebcf032d9488dc0a5f72338d6f9cffdedcab34aa |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/terminal_erosion.py | 1 | 4888 | a6646ea37c4cbfe313a18f08b85478867dea524c1b448e7c7a69bf13813f1348 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/terminal_retention.py | 1 | 7349 | 081da818eba371bf10b3788e6b6389352ab80c60938d233197314de234e036cf |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/types.py | 1 | 1343 | e9ebb7ebf88259bc8d7c175a1b23dcbaabfd01309a5abfb05d322b27903ead83 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/update_semantics.py | 1 | 1049 | 5939b1e2ed25a91670f32825bf787ea2569dfd3b4d36baa7f10b0b19dd80b562 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__init__.py | 1 | 206 | 03c11645f21d6f8d21b9648bd9d61964792a660280e33f7b8bde4c5d8457f0b7 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/__init__.cpython-311.pyc | 1 | 354 | 48c788a98f89415d877f25c0184721551707798d6f4d671a348e5cf86d656f4d |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/adamw_coupling_audit.cpython-311.pyc | 1 | 1290 | 7c39910ae23b7cb4147d2735f3a8d3eba91e1f1bc3daecfa7d105213d494a8ce |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/controller.cpython-311.pyc | 1 | 2772 | 8f3da12d7a4a50b08cd61702356772a990301bec62d079062122eb8ea683155d |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/functional_mechanisms.cpython-311.pyc | 1 | 529 | ca9ec0b1739e0e6c1fa9205266da998aa20f8765a219a81b1f671eb3b4b5b70a |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/geometry.cpython-311.pyc | 1 | 960 | 32298abc89e140bb1a744e10902782fa08fb010b109c1fa1d5d2ffe4b7039878 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/geometry_certificate.cpython-311.pyc | 1 | 13845 | 3cc98fbee35af46f3bf30b493337c8840a1a83da7d1401adab5b6d77b2720952 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/guards.cpython-311.pyc | 1 | 638 | d5da978e665cd8628498ade0b61d1e96f8c2e4463d2d8744e3afbac6e6642017 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/lq_functional_predictor.cpython-311.pyc | 1 | 12941 | 500850f8adfd2f7d00939db7884410a25aec47cbb315c345cd72b1a75b446198 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/lq_output_space_functional.cpython-311.pyc | 1 | 18500 | 45b6b056e0d4f841098b92e9861b7e2f6090dbc44a1766531b7ba0ad5873a4a3 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/manifold_channel_geometry.cpython-311.pyc | 1 | 28600 | 9d067b333bb92b7d7e500cb7d13e3ba7b24a8f45566b30669908a73f6d1199f6 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/manual_optimizers.cpython-311.pyc | 1 | 456 | cb4a0af2562995ef651bed8974af7542314f87cfb3eabee242c87aa075732755 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/mlp_functional.cpython-311.pyc | 1 | 47641 | 9f064c9466ae2f4950bc70ee1491239ad629e77265d65138c8db704d3742cc9d |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/role_budget.cpython-311.pyc | 1 | 705 | 8e56622712b90ca8b55d17312e0bbe790cc6f68a1b935f547ac49a9ea5566fbe |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/second_diff.cpython-311.pyc | 1 | 293 | ce97d2495076f4d8ace60e86c4b985801ae0aa404af210ab46469f1d5436b42f |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/snr_gated_lq.cpython-311.pyc | 1 | 42002 | 4f39d13004ff5a414cc1f45ead199e410c919f7f172b97e877cd3730fcfc7c2d |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/update_tensor.cpython-311.pyc | 1 | 593 | 3f95cd3eaa79133028a24a2ab4a09b057f0dc6cf86fac88534470a27629baae8 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/adamw_coupling_audit.py | 1 | 866 | a8c46bbebb951f05a117ffbc536001be7a992cc2a8362c4b14dceb1ffc00c8cf |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/controller.py | 1 | 1090 | d091b1ed2385b235304594cb9bf288cacf8fd3fbf87e88dd03d1b5bdae001abc |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/functional_mechanisms.py | 1 | 358 | 712cfd392064975c72c849a17a1abf16584585362d724c7a5caee89e6dccc07e |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/geometry.py | 1 | 398 | 144c5a1d0af8172947c605838652c9c646920d9f3937c654263c0f19ee7a7025 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/geometry_certificate.py | 1 | 9654 | a901b2d974bb71d8625a0b952c9643d500f173e9d2cf9a7a6cb51343a7c61d3a |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/guards.py | 1 | 220 | 52da0d02414931cec617143f5bf07e7e6576879b52358ebee388980d3c3be572 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/lq_functional_predictor.py | 1 | 7988 | caf7b9d72f1969a26654605cce91786d549dce7affc1c4ae3b837312fc06a799 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/lq_output_space_functional.py | 1 | 9950 | 0c55e598e7465dda65e4233d955a3756bb0adbea71de3b9e6295f86b54cd7d08 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/manifold_channel_geometry.py | 1 | 13490 | 6eac1d96efc4e83e190d688680d5a9952981f1c4ace6f589b0cdce560dd04a98 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/manual_optimizers.py | 1 | 269 | caaf1c0beda8c2399b0cd2d8fb21ba789811b5d2018a2081be250bf02e8e52c8 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/mlp_functional.py | 1 | 23475 | 4d975294813e8ac90ee80a02e23af74d21338209ef52ac9b48265a669ef52735 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/role_budget.py | 1 | 204 | b32459cec81d80c2767a3d7e140bbe1a68ba7a40c4348d8c09bbbde9180d2f16 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/second_diff.py | 1 | 127 | 71def41caa2d938ebb74eb85dc7280278641d6b64aaacd787749006c823a081d |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/snr_gated_lq.py | 1 | 22499 | cd04043742c9e82e5e6a0bbb9d71010133bcd4c7753707cbf28e202087295c5e |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/update_tensor.py | 1 | 401 | d9000d7e39324a645c89465429851f89974dbebf4cfb0494ae7a3e1ac8f7aeda |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__init__.py | 1 | 41 | f0abab685f0c2ac70fcd6b168064f0461407d55cb26a9729686f6019882b7832 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/__init__.cpython-311.pyc | 1 | 337 | 2140aec2a4b926401a1773f0d6d6180d099eba71d5cb794c22c024b3fd1e7fe4 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/che_official.cpython-311.pyc | 1 | 327 | 62d6af8570cf268e9dcdfffe4dff5c745e56d718eca7322b8c9429c89bb7adfc |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/cheby_fused.cpython-311.pyc | 1 | 1139 | f39650352d922bcebba5dca4cb73b02dab409b512a1f8a018baa5f6de825436d |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/compiled_head.cpython-311.pyc | 1 | 222 | 714979a492922469d1f96774bc3e0ce6bba0339a0b07e0d89241bc8334f45868 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/fou_official.cpython-311.pyc | 1 | 329 | 7d50c17384115f10ae129d3dba01275f3cac2c238a387af649807956736ae654 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/fourier_fused.cpython-311.pyc | 1 | 1063 | 8a7bf2f9d5ab70cf9fad488aeee4f6533abe368bcf248cb3da2ffb75c364672d |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/fused_chebyshev_k3.cpython-311.pyc | 1 | 50893 | cda5318f447273b9162b8638b3713aa5f891a53646bc179cb2aa1cfeddf4032f |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/fused_fourier_k2.cpython-311.pyc | 1 | 71805 | 121e7b477995dcaff49e371fe164f96ede04e2b6ba28ce2e41dd8195f69f96c0 |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/fused_hat_wavelet.cpython-311.pyc | 1 | 18285 | 2208c23aad6637ad71ad59eee85e08e8765cb119b9646f4e7f980aab7d3e52fe |
| results/v22_06_training_dynamics_metric_geometry_fu_basis_efficiency_4gpu/official_v22_06/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/fused_hinge_quadratic.cpython-311.pyc | 1 | 138228 | 03e5b9174a6f494790a28016ea589e57d428b6f82f53cbffff0b72f263b8d460 |

_仅显示前 160 / 2511 rows；完整 CSV 见 artifact。_

你要理解这个文档，然后根据这个计划执行直到完成，提供两个日志，执行日志和实验结果复盘日志。不允许虚假允许不允许编造数据，结束记录关键实验数据和分析结论到实验结果复盘日志 (也不允许编造数据).遇到blocker必须按文档规划的修复方向尝试解决，不允许轻易放弃，但同样不能造假。修复的结果写入也要写入实验结果复盘日志， 并要注明你做了什么修改 方便审计你做的是否合理。为了防止后人花太多时间读代码复现，请在实验的执行日志写清楚指令/文件等信息，让以后复现更容易。复盘文件里除了必须有的实验数据/结果（不能只依赖py文件），也要有实验分析，结论，insight 和更细的证据链。
、

达成目标了吗 没有请继续 按计划里的推荐思路修改 如果还是不行尝试思考如何解决然后尝试修复，结果都要添加进实验结果复盘日志，运行指令添加进执行日志。 深入思考，但不允许虚假允许不允许编造数据。除非你已经不确定该怎么做，否则继续推进。新代码和结果要保存到压缩包v22_06_results_bundle.zip 