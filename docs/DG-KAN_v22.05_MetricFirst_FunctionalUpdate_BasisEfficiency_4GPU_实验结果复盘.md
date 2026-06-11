# DG-KAN v22.05 MetricFirst FunctionalUpdate BasisEfficiency 4GPU 实验结果复盘

生成时间：2026-06-06 05:59:59 +0800

## Route

- route: `R3-DRATDRBFRepairBlocked`
- promotion_allowed: 0
- blocking_metric: `official_fused_missing`
- next_codex_action: finish production official fused D-RAT/D-RBF train runner wiring
- S0 pass: 1
- D-CHE/D-FOU S1 pass: 1 / 1
- D-RAT/D-RBF official fused blocked: 1
- functional_route: `F0-MetricNoEffect`
- KAN mapping decision: `KANSourceChannelMismatch`

## Part A S0.12 Code / Metric / Mechanism Gate

| check | pass | metric | value | blocker |
| --- | --- | --- | --- | --- |
| required_source_files | 1 | exists | 31/31 |  |
| compileall | 1 | py_compile | 0 |  |
| import_closure | 1 | self_contained_import | 0 |  |
| linec_fast_golden | 1 | tests | 9 |  |
| linec_channel_golden | 1 | tests | 8 |  |
| source_chain_tests | 1 | tests | 7/7 |  |
| terminal_retention_tests | 1 | tests | 14/14 |  |
| metric_tests | 1 | tests | 19/19 |  |
| mechanism_contracts | 1 | v22.05 schema+semantic | 18/18;alias_undeclared=0 |  |
| kernel_gradcheck | 1 | kernels | 4 |  |
| profiler_phase_tests | 1 | tests | 3/3 |  |


### Mechanism Schema And Semantic Alias

| pairs | semantic_alias_pairs | undeclared_alias_pairs | pass | gradient_norm |
| --- | --- | --- | --- | --- |
| 153 | 1 | 0 | 1 | 0.7430999279022217 |


| mechanism | mechanism_family | source_pattern | uses_optimizer_primary | uses_function_space_metric | uses_sobolev_metric | uses_rkhs_metric | uses_fisher_metric | semantic_noncollapse_group | semantic_alias_representative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M181-EarlySourceSlowEMATerminalNoraOrthogonalSourceFU | slow_state | early100_h800_slow_ema_terminal_nora_row_orthogonal_source | 0 | 0 | 0 | 0 | 0 | v22_04_terminal_momentum_like_alias | 1 |
| M218-EarlySourceSlowEMATerminalProjectedOptimizerLambda050FU | slow_state | early100_h800_slow_ema_terminal_projected_optimizer_lambda050 | 0 | 0 | 0 | 0 | 0 | v22_04_terminal_momentum_like_alias | 1 |
| M220-MetricFirstG0L2FU | function_metric | metric_first_G0_L2_parameter_pullback | 0 | 1 | 0 | 0 | 0 | v22_05_metric_first_G0-L2 | 1 |
| M221-MetricFirstDiagFisherFU | function_metric | metric_first_G1_diag_fisher_parameter_pullback | 0 | 1 | 0 | 0 | 1 | v22_05_metric_first_G1-DiagFisher | 1 |
| M222-MetricFirstPopRiskDiagFU | function_metric | metric_first_G2_poprisk_diag_parameter_pullback | 0 | 1 | 0 | 0 | 0 | v22_05_metric_first_G2-PopRiskDiag | 1 |
| M223-MetricFirstSobolevH1FU | function_metric | metric_first_G3_sobolev_h1_parameter_pullback | 0 | 1 | 1 | 0 | 0 | v22_05_metric_first_G3-SobolevH1-hidden | 1 |
| M224-MetricFirstRKHSKNNFU | function_metric | metric_first_G4_rkhs_knn_parameter_pullback | 0 | 1 | 0 | 1 | 0 | v22_05_metric_first_G4-RKHS-KNN | 1 |
| M225-MetricFirstFisherRKHSFU | function_metric | metric_first_G5_fisher_rkhs_parameter_pullback | 0 | 1 | 0 | 1 | 1 | v22_05_metric_first_G5-Fisher-RKHS | 1 |
| M226-MetricFirstLowNDSFU | function_metric | metric_first_G6_low_nds_parameter_pullback | 0 | 1 | 0 | 0 | 0 | v22_05_metric_first_G6-LowNDS | 1 |
| M227-MetricFirstEnsembleFU | function_metric | metric_first_G8_metric_ensemble_parameter_pullback | 0 | 1 | 0 | 0 | 0 | v22_05_metric_first_G8-MetricEnsemble | 1 |
| M228-MetricSourceMemoryG0L2FU | slow_state+function_metric | metric_source_memory_G0_L2_parameter_pullback | 0 | 1 | 0 | 0 | 0 | v22_05_metric_first_G0-L2 | 1 |
| M229-MetricSourceMemoryDiagFisherFU | slow_state+function_metric | metric_source_memory_G1_diag_fisher_parameter_pullback | 0 | 1 | 0 | 0 | 1 | v22_05_metric_first_G1-DiagFisher | 1 |
| M230-MetricSourceMemoryPopRiskDiagFU | slow_state+function_metric | metric_source_memory_G2_poprisk_diag_parameter_pullback | 0 | 1 | 0 | 0 | 0 | v22_05_metric_first_G2-PopRiskDiag | 1 |
| M231-MetricSourceMemorySobolevH1FU | slow_state+function_metric | metric_source_memory_G3_sobolev_h1_parameter_pullback | 0 | 1 | 1 | 0 | 0 | v22_05_metric_first_G3-SobolevH1-hidden | 1 |
| M232-MetricSourceMemoryRKHSKNNFU | slow_state+function_metric | metric_source_memory_G4_rkhs_knn_parameter_pullback | 0 | 1 | 0 | 1 | 0 | v22_05_metric_first_G4-RKHS-KNN | 1 |
| M233-MetricSourceMemoryFisherRKHSFU | slow_state+function_metric | metric_source_memory_G5_fisher_rkhs_parameter_pullback | 0 | 1 | 0 | 1 | 1 | v22_05_metric_first_G5-Fisher-RKHS | 1 |
| M234-MetricSourceMemoryLowNDSFU | slow_state+function_metric | metric_source_memory_G6_low_nds_parameter_pullback | 0 | 1 | 0 | 0 | 0 | v22_05_metric_first_G6-LowNDS | 1 |
| M235-MetricSourceMemoryEnsembleFU | slow_state+function_metric | metric_source_memory_G8_metric_ensemble_parameter_pullback | 0 | 1 | 0 | 0 | 0 | v22_05_metric_first_G8-MetricEnsemble | 1 |


## Part B Basis Efficiency / D-RAT / D-RBF

| carrier | forward_ratio_vs_mlp | step_ratio_vs_mlp | memory_ratio_vs_mlp | functional_overhead_ratio | v22_05_S1_pass | v22_05_decision | v22_05_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | 0.9305108277155166 | 0.46682782835473463 | 0.9913236539978382 | 1.0 | 1 | OfficialEfficientCarrier |  |
| D-FOU | 1.0087315984225318 | 0.5089279836996342 | 0.9888469358117921 | 1.0 | 1 | OfficialEfficientCarrier |  |


| carrier | profile_rows | micro_near_E1_rows | E1_official_rows | best_forward_ratio | best_step_ratio | best_memory_ratio | decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 7 | 7 | 0 | 1.092105636399739 | 1.1609796781106947 | 1.03125 | MicroNearE1OfficialFusedBlocked | official_fused_missing |
| D-RBF | 7 | 2 | 0 | 1.71900474871389 | 1.2603177294150847 | 1.03125 | MicroNearE1OfficialFusedBlocked | official_fused_missing |


## Part C Metric-First MLP FU

- best_metric_v21_id: `MLP-MFSM-G0-L2`
- best_metric_name: `G0-L2-source-memory`
- best R4800/3200: 
- best h4800: -0.8992380036248101
- best row_h4800_positive_count: 0

| v21_id | metric_name | rows | source_h800_mean | source_h3200_mean | source_h4800_mean | R4800_over_3200 | row_h4800_positive_count | R4800_improvement_vs_v2204_best | H_C1_exploration_S2_pass | failure_taxonomy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-MFSM-G0-L2 | G0-L2-source-memory | 9 | -0.5356423854827881 | -0.9386075470182631 | -0.8992380036248101 |  | 0 | nan | 0 | ControlEquivalent |
| MLP-MFSM-G1-DiagFisher | G1-DiagFisher-source-memory | 9 | -0.5342104964786105 | -0.945281015502082 | -0.9136777453952365 |  | 0 | nan | 0 | ControlEquivalent |
| MLP-MFSM-G2-PopRiskDiag | G2-PopRiskDiag-source-memory | 9 | -0.5216396649678549 | -0.9250186151928372 | -0.8861064116160074 |  | 0 | nan | 0 | ControlEquivalent |
| MLP-MFSM-G3-SobolevH1 | G3-SobolevH1-source-memory | 9 | -0.4827902317047119 | -0.9091992510689629 | -0.8829476303524442 |  | 0 | nan | 0 | ControlEquivalent |
| MLP-MFSM-G4-RKHSKNN | G4-RKHS-KNN-source-memory | 9 | -0.5200352403852675 | -0.9404021501541138 | -0.9094484912024604 |  | 0 | nan | 0 | ControlEquivalent |
| MLP-MFSM-G5-FisherRKHS | G5-Fisher-RKHS-source-memory | 9 | -0.47008440229627824 | -0.8795197937223647 | -0.8430123329162598 |  | 0 | nan | 0 | ControlEquivalent |
| MLP-MFSM-G6-LowNDS | G6-LowNDS-source-memory | 9 | -0.5493911107381185 | -0.9644929435518053 | -0.9328309694925944 |  | 0 | nan | 0 | ControlEquivalent |
| MLP-MFSM-G8-Ensemble | G8-MetricEnsemble-source-memory | 9 | -0.4964752462175157 | -0.9025347630182902 | -0.863989512125651 |  | 0 | nan | 0 | ControlEquivalent |


### Metric Failure Autopsy

| metric_families_tried | metric_rows | source_formation_failed_rows | terminal_retention_evaluable_rows | control_equivalent_rows | productive_h4800_rows | best_attempt_label | best_v21_id | best_source_h100_mean | best_source_h3200_mean | best_source_h4800_mean | dominant_failure_mode | strongest_logged_energy_correlation_to_h4800 | strongest_logged_energy_correlation_pearson |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G0-L2;G1-DiagFisher;G2-PopRiskDiag;G3-SobolevH1;G4-RKHSKNN;G5-FisherRKHS;G6-LowNDS;G8-Ensemble | 16 | 16 | 0 | 16 | 0 | source_memory_repair | MLP-MFSM-G5-FisherRKHS | -0.8142153157128228 | -0.8795197937223647 | -0.8430123329162598 | EarlySourceFormationFailed;ControlEquivalent;MetricNoEffect | Brier_debt_final | -0.992020385387869 |


| attempt_label | v21_id | source_h100_mean | source_h800_mean | source_h3200_mean | source_h4800_mean | source_h6400_mean | R4800_over_3200 | early_source_chain_open | terminal_retention_evaluable | control_equivalent_fraction | metric_energy_mean | NDS | LineC_debt_final | hidden_source_fraction | readout_source_fraction | autopsy_failure_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metric_projection | MLP-MF-G0-L2 | -0.8825863467322456 | -0.5414169894324409 | -0.9600433111190796 | -0.9297758473290337 | -0.8336693180931939 |  | 0 | 0 | 1 | 0.15253325088156597 | 0.5404279265138838 | 0.9749689405294506 | 0.8213902728965093 | 0.17860972710349043 | EarlySourceFormationFailed;ControlEquivalent |
| metric_projection | MLP-MF-G1-DiagFisher | -0.8818994363149008 | -0.541116820441352 | -0.9669252104229398 | -0.9438836044735379 | -0.8555247783660889 |  | 0 | 0 | 1 | 0.15433175739728744 | 0.5233993298477597 | 0.9939288519162152 | 0.8740598905282295 | 0.1259401094717705 | EarlySourceFormationFailed;ControlEquivalent |
| metric_projection | MLP-MF-G2-PopRiskDiag | -0.869112491607666 | -0.5269252724117703 | -0.9452477693557739 | -0.9155759811401367 | -0.8196761608123779 |  | 0 | 0 | 1 | 0.13111812026343414 | 0.5328835215833452 | 0.9986520957680156 | 0.7740227826540473 | 0.22597721734595266 | EarlySourceFormationFailed;ControlEquivalent |
| metric_projection | MLP-MF-G3-SobolevH1 | -0.8218606842888726 | -0.4867965380350749 | -0.9234625630908542 | -0.9037005636427138 | -0.8170236216651069 |  | 0 | 0 | 1 | 0.11511472485338649 | 0.07733514201309946 | 0.9795053025098903 | 0.8851330093725392 | 0.11486699062746071 | EarlySourceFormationFailed;ControlEquivalent |
| metric_projection | MLP-MF-G4-RKHSKNN | -0.8607344097561307 | -0.5240310298071967 | -0.9555016491148207 | -0.9319777488708496 | -0.8406267960866293 |  | 0 | 0 | 1 | 0.12023803531709645 | 0.23419038289123112 | 0.9918676244419038 | 0.8241310992056597 | 0.17586890079434023 | EarlySourceFormationFailed;ControlEquivalent |
| metric_projection | MLP-MF-G5-FisherRKHS | -0.8149139616224501 | -0.47669000095791286 | -0.9034753243128458 | -0.8777479065789117 | -0.7855728467305502 |  | 0 | 0 | 1 | 0.16524519775476723 | 0.5099819170104133 | 0.9742769377005241 | 0.8816720603866066 | 0.11832793961339332 | EarlySourceFormationFailed;ControlEquivalent |
| metric_projection | MLP-MF-G6-LowNDS | -0.8932327429453532 | -0.5542649957868788 | -0.9813124206331041 | -0.9569069279564751 | -0.8660722573598226 |  | 0 | 0 | 1 | 0.13086532556141417 | 0.17023818029297721 | 0.9828687808577994 | 0.8354713885794796 | 0.16452861142052033 | EarlySourceFormationFailed;ControlEquivalent |
| metric_projection | MLP-MF-G8-Ensemble | -0.8423829343583848 | -0.5018103387620714 | -0.9229681359397041 | -0.8939592043558756 | -0.7982528209686279 |  | 0 | 0 | 1 | 0.1454764338417186 | 0.3190096384949154 | 0.9926397565781586 | 0.8589723926432645 | 0.1410276073567356 | EarlySourceFormationFailed;ControlEquivalent |
| source_memory_repair | MLP-MFSM-G0-L2 | -0.8820917341444228 | -0.5356423854827881 | -0.9386075470182631 | -0.8992380036248101 | -0.7943518691592746 |  | 0 | 0 | 1 | 0.1561635309002466 | 0.5510742796791924 | 0.989770250739768 | 0.7460880719786531 | 0.25391192802134677 | EarlySourceFormationFailed;ControlEquivalent |
| source_memory_repair | MLP-MFSM-G1-DiagFisher | -0.8811051845550537 | -0.5342104964786105 | -0.945281015502082 | -0.9136777453952365 | -0.8161097102695041 |  | 0 | 0 | 1 | 0.15683151340732976 | 0.5233430200152926 | 0.9872890644142278 | 0.8676728807507338 | 0.13232711924926627 | EarlySourceFormationFailed;ControlEquivalent |
| source_memory_repair | MLP-MFSM-G2-PopRiskDiag | -0.8687173260582818 | -0.5216396649678549 | -0.9250186151928372 | -0.8861064116160074 | -0.781456364525689 |  | 0 | 0 | 1 | 0.14156877248444494 | 0.5477706988652548 | 0.9647882381768186 | 0.6879836138863447 | 0.31201638611365523 | EarlySourceFormationFailed;ControlEquivalent |
| source_memory_repair | MLP-MFSM-G3-SobolevH1 | -0.8213804033067491 | -0.4827902317047119 | -0.9091992510689629 | -0.8829476303524442 | -0.7897394763098823 |  | 0 | 0 | 1 | 0.14051640938139628 | 0.07779041967458195 | 0.9830019088684155 | 0.8329118962482375 | 0.16708810375176253 | EarlySourceFormationFailed;ControlEquivalent |
| source_memory_repair | MLP-MFSM-G4-RKHSKNN | -0.8604100280337863 | -0.5200352403852675 | -0.9404021501541138 | -0.9094484912024604 | -0.8104226854112413 |  | 0 | 0 | 1 | 0.1410521056710018 | 0.2224353618092007 | 0.9934023985223379 | 0.7494241590854891 | 0.25057584091451096 | EarlySourceFormationFailed;ControlEquivalent |
| source_memory_repair | MLP-MFSM-G5-FisherRKHS | -0.8142153157128228 | -0.47008440229627824 | -0.8795197937223647 | -0.8430123329162598 | -0.7395434909396701 |  | 0 | 0 | 1 | 0.15944737885147334 | 0.5077908072206709 | 0.9851398904242331 | 0.8750346971847144 | 0.12496530281528556 | EarlySourceFormationFailed;ControlEquivalent |
| source_memory_repair | MLP-MFSM-G6-LowNDS | -0.8927095731099447 | -0.5493911107381185 | -0.9644929435518053 | -0.9328309694925944 | -0.8346467547946506 |  | 0 | 0 | 1 | 0.13957197941425775 | 0.1823765370580885 | 0.9837792012358805 | 0.7667997836827617 | 0.23320021631723828 | EarlySourceFormationFailed;ControlEquivalent |
| source_memory_repair | MLP-MFSM-G8-Ensemble | -0.8419986036088731 | -0.4964752462175157 | -0.9025347630182902 | -0.863989512125651 | -0.7585986587736342 |  | 0 | 0 | 1 | 0.14442938953224155 | 0.3100516034497155 | 0.9896151163035829 | 0.8127854161778052 | 0.18721458382219483 | EarlySourceFormationFailed;ControlEquivalent |


| attempt_label | x_metric | y_metric | n | pearson | spearman |
| --- | --- | --- | --- | --- | --- |
| metric_projection | metric_energy_L2 | source_h4800_mean | 8 | 0.2909160268214763 | 0.23809523809523808 |
| metric_projection | metric_energy_L2 | source_h3200_mean | 8 | 0.23909234254035047 | 0.14285714285714285 |
| metric_projection | metric_energy_L2 | source_h100_mean | 8 | 0.06908607071986116 | 0.023809523809523808 |
| metric_projection | metric_energy_L2 | source_decay_h3200_to_h4800 | 8 | 0.38863714096274576 | 0.40476190476190477 |
| metric_projection | metric_energy_L2 | R4800_over_3200 | 0 |  |  |
| metric_projection | metric_energy_Fisher | source_h4800_mean | 8 | 0.30876791704424933 | 0.35714285714285715 |
| metric_projection | metric_energy_Fisher | source_h3200_mean | 8 | 0.23820568807411213 | 0.21428571428571427 |
| metric_projection | metric_energy_Fisher | source_h100_mean | 8 | 0.0436866999337045 | 0.047619047619047616 |
| metric_projection | metric_energy_Fisher | source_decay_h3200_to_h4800 | 8 | 0.5224244914665522 | 0.5 |
| metric_projection | metric_energy_Fisher | R4800_over_3200 | 0 |  |  |
| metric_projection | metric_energy_PopRisk | source_h4800_mean | 8 | -0.1753078772738753 | -0.16666666666666666 |
| metric_projection | metric_energy_PopRisk | source_h3200_mean | 8 | -0.28207591046184005 | -0.2857142857142857 |
| metric_projection | metric_energy_PopRisk | source_h100_mean | 8 | -0.5118061258284126 | -0.5476190476190477 |
| metric_projection | metric_energy_PopRisk | source_decay_h3200_to_h4800 | 8 | 0.7409213268205002 | 0.7142857142857143 |
| metric_projection | metric_energy_PopRisk | R4800_over_3200 | 0 |  |  |
| metric_projection | metric_energy_Sobolev | source_h4800_mean | 8 | 0.24922757621045205 | 0.23809523809523808 |
| metric_projection | metric_energy_Sobolev | source_h3200_mean | 8 | 0.19035148132521312 | 0.14285714285714285 |
| metric_projection | metric_energy_Sobolev | source_h100_mean | 8 | 0.008516301056764139 | 0.023809523809523808 |
| metric_projection | metric_energy_Sobolev | source_decay_h3200_to_h4800 | 8 | 0.4352551203922985 | 0.40476190476190477 |
| metric_projection | metric_energy_Sobolev | R4800_over_3200 | 0 |  |  |
| metric_projection | metric_energy_RKHS | source_h4800_mean | 8 | 0.27184189231653794 | 0.30952380952380953 |
| metric_projection | metric_energy_RKHS | source_h3200_mean | 8 | 0.20341371350651644 | 0.19047619047619047 |
| metric_projection | metric_energy_RKHS | source_h100_mean | 8 | 0.033905519813743304 | 0.0 |
| metric_projection | metric_energy_RKHS | source_decay_h3200_to_h4800 | 8 | 0.5044964710812361 | 0.5714285714285714 |
| metric_projection | metric_energy_RKHS | R4800_over_3200 | 0 |  |  |
| metric_projection | metric_energy_mean | source_h4800_mean | 8 | 0.2669307341088869 | 0.23809523809523808 |
| metric_projection | metric_energy_mean | source_h3200_mean | 8 | 0.20675468390526688 | 0.14285714285714285 |
| metric_projection | metric_energy_mean | source_h100_mean | 8 | 0.02655770130763397 | 0.023809523809523808 |
| metric_projection | metric_energy_mean | source_decay_h3200_to_h4800 | 8 | 0.4458064485047137 | 0.40476190476190477 |
| metric_projection | metric_energy_mean | R4800_over_3200 | 0 |  |  |
| metric_projection | NDS | source_h4800_mean | 8 | 0.12245620386867384 | 0.047619047619047616 |
| metric_projection | NDS | source_h3200_mean | 8 | 0.0322115694318468 | -0.07142857142857142 |
| metric_projection | NDS | source_h100_mean | 8 | -0.19944667553337336 | -0.2857142857142857 |
| metric_projection | NDS | source_decay_h3200_to_h4800 | 8 | 0.6471325962903302 | 0.7142857142857143 |
| metric_projection | NDS | R4800_over_3200 | 0 |  |  |
| metric_projection | raw_gradient_NDS | source_h4800_mean | 8 | 0.5480217131238813 | 0.3333333333333333 |
| metric_projection | raw_gradient_NDS | source_h3200_mean | 8 | 0.5692933538162465 | 0.30952380952380953 |
| metric_projection | raw_gradient_NDS | source_h100_mean | 8 | 0.5388574337540705 | 0.4523809523809524 |
| metric_projection | raw_gradient_NDS | source_decay_h3200_to_h4800 | 8 | -0.10803640355641579 | -0.4523809523809524 |
| metric_projection | raw_gradient_NDS | R4800_over_3200 | 0 |  |  |

_仅显示前 40 / 225 rows；完整 CSV 见 artifact。_


## Part D KAN Source Mapping

| carrier | variant | v22_id | h800 | h3200 | h4800 | h4800_retention_ratio | KAN_h100_h400_h800_h3200_chain_open | KAN_productive_terminal_source | v22_05_source_mapping_decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-AdamW | 0.0 | 0.0 | 0.0 |  | 0 | 0 | KANSourceChannelMismatch |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-NoOpMatchedOverhead | -1.6334137121836345 | -1.5455449488427904 | -1.4494818581475153 |  | 0 | 0 | KANSourceChannelMismatch |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-RandomMatchedNorm | -1.6256221400366888 | -1.5377527674039204 | -1.4416825506422255 |  | 0 | 0 | KANSourceChannelMismatch |
| D-CHE | CHE-R4-k3-gradbuf-triton | CTRL-SGD | -1.6285207801394992 | -1.53294352028105 | -1.4355009661780462 |  | 0 | 0 | KANSourceChannelMismatch |
| D-CHE | CHE-R4-k3-gradbuf-triton | KSW1-basis-estimate-readout-commit | -1.623139566845364 | -1.5316170387797885 | -1.434997320175171 |  | 0 | 0 | KANSourceChannelMismatch |
| D-CHE | CHE-R4-k3-gradbuf-triton | KSW10-h800-dual-memory-source-bank | -0.18122186263402304 | -0.09334645668665568 | 0.0027211440934075248 |  | 0 | 0 | KANSourceChannelMismatch |
| D-CHE | CHE-R4-k3-gradbuf-triton | KSW2-lowdegree-lowfreq-source-bank | -0.18363622162077162 | -0.0202968782848782 | 0.004610690805647109 |  | 0 | 0 | KANSourceChannelMismatch |
| D-CHE | CHE-R4-k3-gradbuf-triton | KSW9-h800-source-slow-ema-bank | -0.17689558532502916 | -0.08902051051457723 | 0.007046798865000407 |  | 0 | 0 | KANSourceChannelMismatch |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-AdamW | 0.0 | 0.0 | 0.0 |  | 0 | 0 | KANSourceChannelMismatch |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-NoOpMatchedOverhead | -1.5411422385109796 | -1.4102725783983867 | -1.3356308473481073 |  | 0 | 0 | KANSourceChannelMismatch |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-RandomMatchedNorm | -1.5430146985583835 | -1.4121408263842266 | -1.3374980092048645 |  | 0 | 0 | KANSourceChannelMismatch |
| D-FOU | FOU-R4-k4-triton-no-materialize | CTRL-SGD | -1.5400735669665866 | -1.400064554479387 | -1.3185323410563998 |  | 0 | 0 | KANSourceChannelMismatch |
| D-FOU | FOU-R4-k4-triton-no-materialize | KSW1-basis-estimate-readout-commit | -1.5382218228446112 | -1.3981423444218106 | -1.3163674473762512 |  | 0 | 0 | KANSourceChannelMismatch |
| D-FOU | FOU-R4-k4-triton-no-materialize | KSW10-h800-dual-memory-source-bank | -0.09634670284059313 | 0.03452601035435995 | 0.10916988054911296 | 3.1619604880100773 | 0 | 0 | KANSourceChannelMismatch |
| D-FOU | FOU-R4-k4-triton-no-materialize | KSW2-lowdegree-lowfreq-source-bank | -0.16873362329271105 | -0.07953935199313694 | -0.09196770191192627 |  | 0 | 0 | KANSourceChannelMismatch |
| D-FOU | FOU-R4-k4-triton-no-materialize | KSW9-h800-source-slow-ema-bank | -0.10394989119635688 | 0.02692327234480116 | 0.1015673279762268 | 3.7724733708248244 | 0 | 0 | KANSourceChannelMismatch |


## 修改记录

- 新增 v22.05 runner：common、S0.12 truth gate、efficiency reconfirm、D-RAT/D-RBF repair wrapper、metric-first FU summarizer、metric failure autopsy、KAN mapping readback、queue sampler、finalize。
- 新增 `dgkan/fu/function_space_metrics.py`、`sobolev_metric.py`、`rkhs_metric.py`、`fisher_metric.py`、`metric_projection.py`；这些 helper 只用 train-batch/current-gradient 生成 metric-projected update 和 audit energy，不读取 validation/test/future。
- 扩展 `dgkan/fu/mechanisms.py`：新增 M220-M227 metric-first mechanisms；在 metric-only early-source failure 后追加 M228-M235 source-memory repair mechanisms；修正 contract schema，把 `source_pattern` 与 bool columns 分离，并给 v22.04 terminal momentum-like aliases 标注 `semantic_noncollapse_group`。
- 扩展 `experiments/run_v21_01_source_retention.py` 与 `experiments/run_v17_common.py`：注册 MLP-MF-G0..G8 与 MLP-MFSM-G0..G8 specs，并把 metric diagnostics 写入 horizon trace/enriched matrix，方便审计。

## 分析 / Insight / 结论

- S0.12 以 clean unzip self-contained import 为 hard gate；若 `self_contained_import_check=0` 或 import/compile/schema/alias 任一失败，不允许继续把 science rows 写成 promotion。
- v22.05 的 metric-first mechanisms 是 parameter-pullback proxy，而不是完整显式 Jacobian solver；因此复盘只把它们作为 metric-first 实验推进，不把它们伪称为精确 $J^T G_f J$ 求逆。
- D-CHE/D-FOU efficiency route 仍只证明 carrier efficiency；KAN source-channel 是否打开必须看 KAN source chain rows，不能用 efficiency 替代 functional evidence。
- D-RAT/D-RBF 若仍为 `official_fused_missing`，只能写 active repair progress/blocker，不能进入 functional promotion。
- Metric-first FU 若未过 R4800/3200 gate，failure taxonomy 以落盘 `NDS`、Sobolev/RKHS energy、control-equivalent 和 h4800 source 读数裁决，避免继续扩大旧 terminal-preservation 名字池。
- 本轮若 autopsy 显示 `terminal_retention_evaluable_rows=0`，则失败点不是 h3200 后保留，而是 metric update 没能形成 early/h3200 source；后续应先做 train-only loss-cotangent/source-channel metric target，再谈 terminal projection。

## Artifact Index

| artifact | exists | size_bytes | sha256 |
| --- | --- | --- | --- |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/00_README.md | 1 | 98 | 72e0346906475447794717d92feda885573f107374c229f25f53b88c801a103e |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/__init__.py | 1 | 447 | de3d83266f16ee40cc6e8994331904155cdd3c48e593c525a1831b042aae2386 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/__pycache__/__init__.cpython-311.pyc | 1 | 760 | 8d7c4275b8b5d16d10cf48aa62a1f1a1473259e0dcfdf72bea0bcf371feb62be |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/__pycache__/config.cpython-311.pyc | 1 | 1171 | fc498f680ee1eb37b86bb5dfa8de20684956108f3f95c8f52a95d2dc3c730160 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/__pycache__/contracts.cpython-311.pyc | 1 | 6193 | 4ae4b98d83b3a36e9ac3c14e07eebd2be19277bfc18f843c337135cb9e5d0797 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/__pycache__/registry.cpython-311.pyc | 1 | 3204 | 75a66ae8e1e699eb29b194d148de6502700fb6988d9d73ef18c20e1363e33588 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/__pycache__/specs.cpython-311.pyc | 1 | 3443 | 4dd7ef9f2f4b590274d5d4ad5f1a52cee00b32d70d259d992795acfb728df849 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/artifacts/__init__.py | 1 | 251 | 1d57257123b41115b756dbad3679ed70fd0f0233b91bf9ffbd0fa74745cfeb3b |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/artifacts/__pycache__/__init__.cpython-311.pyc | 1 | 396 | a106bb06d10a3bf2776df0f6b6e29721bd3be6d5e52e36154d946febf65ddc5b |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/artifacts/__pycache__/audit.cpython-311.pyc | 1 | 1980 | cd3a3b7dce60b66bee1ab5c2fd9f85e5016355995d72242694cf658c40f2b8b0 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/artifacts/__pycache__/route.cpython-311.pyc | 1 | 1358 | 78b9c941bcee56f25f731eac630eacdc4c5c984d016d658f58f317f1fcd3ed8b |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/artifacts/__pycache__/writer.cpython-311.pyc | 1 | 6185 | 171781e3c8ed96cc2e354839e6742fdcfaffb42aeeea78bd7aaac9aadb88546d |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/artifacts/audit.py | 1 | 1047 | bc6790ae6347f99be618fd493f7aef99e6905cd0e3471120b1235b193e2e8989 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/artifacts/route.py | 1 | 1008 | 02319fa81e36de7d5a9a64b5e074fa1e6d183b6d2993435b05c165d5c6629d08 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/artifacts/writer.py | 1 | 2060 | 48a630b11379ad68af48f398e96ad58aca04232a80849bf1e42bac583e96efdf |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/config.py | 1 | 1060 | 87e5efff2492f53df84c3f5a7c6cb94ca1f9d7029be39feaa855d5d8eb377e30 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/contracts.py | 1 | 3255 | 03d0ce0f729d59505df8713e4b54d168e387946e7cc8ce036f39c644e133c465 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/diagnostics/__init__.py | 1 | 403 | 4a43eb55dbae11ca2a35739a12a8e4b17c0d0a61ac4445320463aba676c87dc1 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/diagnostics/__pycache__/__init__.cpython-311.pyc | 1 | 536 | 00b913b279c103eae700d27fc688c694636a4d66a5d37036f46faa76063cdcb4 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/diagnostics/__pycache__/basis_workspace.cpython-311.pyc | 1 | 92894 | a6b752950ef08a8b40b9d1b5c2c2162b244efef5a02030a3f0496b9f32dd700f |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/diagnostics/__pycache__/classic_basis.cpython-311.pyc | 1 | 14781 | d013cd70514ccab0949fdc08a5c59aecd24317ce43bc748a2f89156ba9b5b404 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/diagnostics/__pycache__/linec.cpython-311.pyc | 1 | 1922 | 6051d0295158976f988df223758d740c7261209030f27dbd0ead9edd1aed0c36 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/diagnostics/basis_workspace.py | 1 | 96053 | 89446acd9d467edf42fa8716986cfc56a5059ee925cac2dc790f98402e3b7437 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/diagnostics/classic_basis.py | 1 | 12451 | 3310364893cf4e9513406d0f0787c8993b06c3f1d6bcc41d801e45b3c0a36f84 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/diagnostics/linec.py | 1 | 1310 | af17f41dc3b33cca01cc4eaefef75e3b471730911d8784a0783579a6f8f845b8 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/__init__.py | 1 | 41 | 881010633493337f5531dde4c43ae77072d9b1d2c73973eae4f15cbaa9ec67c4 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/__pycache__/__init__.cpython-311.pyc | 1 | 184 | 982c906ba884b2965ec031af629124043454ae3ee627be00869e8d98d5c02e02 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/__pycache__/kernel_census.cpython-311.pyc | 1 | 371 | ebc997930180175d3ad668f5d689c9c3e7c381433196611cf842324922aaa09d |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/__pycache__/memory_timer.cpython-311.pyc | 1 | 433 | af1f712ad6d1ba1b08edc4825a410996573589f10d0f516965f4de296777ab57 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/__pycache__/phase_timer.cpython-311.pyc | 1 | 467 | 9ac402bc5b2bbbfcd2c910f0c8736bca08344089c983e6537b2892e330c67d14 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/__pycache__/profiler.cpython-311.pyc | 1 | 389 | d700ec2b70308b935d92647786ccdcd28705678dc407795250754089b3af6198 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/__pycache__/repair_registry.cpython-311.pyc | 1 | 1015 | e674e21f35308cdd024c5948cc073388831a9053f410b3de91510df287e35d3c |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/__pycache__/same_param_mlp.cpython-311.pyc | 1 | 905 | e71cbbfee7a7972fbe2121145d6cd4890fe2d05de5a4aeb223a7c5fc4330ef46 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/kernel_census.py | 1 | 177 | 0e515e1a21839430904b943a6ba17f7cfa9024d456f4feda32a34b98a4f151d6 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/memory_timer.py | 1 | 305 | e9d0bb55d2d39ef0e9a22aab1e6f11826189cfeb3738be357fd103a1351cf8af |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/phase_timer.py | 1 | 373 | 81aca9f04546c482b888bc0aeea595eac1983f48d1bd5dea4cc70bc3c1493669 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/profiler.py | 1 | 194 | c2f9813b310d81cadeedd7ed23c5e11bda1f3e3227968ec815e8d579b3cb8a6b |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/repair_registry.py | 1 | 615 | 33f4293f5f1ad4d5900992d6ceb1771e0c9187b5f7d2427bccef840c58a26658 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/efficiency/same_param_mlp.py | 1 | 464 | 2f34dbe22368505145397ed95fd0b1c6016ddd937b86ff4418a409c72f339f86 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/external/__init__.py | 1 | 36 | f95b9474abcdda646d518e002c522cac53eada80e7dd740d0fd60af5861a5451 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/external/__pycache__/__init__.cpython-311.pyc | 1 | 178 | 568ae1c178b3d9cec5169e220ddb2d050a2a387c8c2eb9cb3f7be1cb846b2d24 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/external/__pycache__/baselines.cpython-311.pyc | 1 | 324 | dbfb55c4860e5b2b899c85cc4ad332e08a0b5368a9988afb7cdb374a1bfcb6ef |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/external/__pycache__/counters.cpython-311.pyc | 1 | 535 | f290cbdd3014a57bc3b46c2a6354169828d8566aa203f28477fa3ccb5a38d04f |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/external/__pycache__/kanbefair_adapter.cpython-311.pyc | 1 | 2088 | 0e0bcc15fa0e37b4ac5ca725c8dcd703dcf508bfa9d7f4dac8c0a1dc24b6b8d5 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/external/baselines.py | 1 | 151 | 741ea1e43859720ad9ca0e3ac79fe720082379db5da32c57ef2e610f6e828f07 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/external/counters.py | 1 | 222 | 074275dff2e16b7029299ea2ce182c5f250330a117edeeed8fdc66e62b88ee0f |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/external/kanbefair_adapter.py | 1 | 1143 | f6de772edee26e57c7b41c4ae5556212cced228415adfaadf767ff2b8f2df906 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__init__.py | 1 | 51 | 4b1457bbb0b8007de6ab30711924ce6c968bf453ac797ff63b2a77b45c846162 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/__init__.cpython-311.pyc | 1 | 335 | 4492b1807c9cd264cf32d33e31ac97542880df3a870fcf574d25794e4a94f965 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/audit.cpython-311.pyc | 1 | 1452 | d4049d15e2563bb151e5f14e1199adb805181b5436d279df0de9b7180fafc11c |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/carriers.cpython-311.pyc | 1 | 1835 | 526e5386df3f30df913f376a862b15255e32f1571ceb858271a3e90736be6648 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/controls.cpython-311.pyc | 1 | 3233 | 5fb8917616ee436114e1d88a48927ea8f236000d7ccf807401aacb645a4e08ec |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/core.cpython-311.pyc | 1 | 11743 | 9932291b05a3c17fc3ad21c4db2a30fa9186fee3a4cdb2b10d1da75820b49ea7 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/debt_accounting.cpython-311.pyc | 1 | 5977 | 97ccdd01d72d2349148c4227fba3970b15e10c7d7c19665bc8014db96f5b9961 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/diffeomorphic_target.cpython-311.pyc | 1 | 7787 | 456ac14763d88d0198702719cfd6161f24c4aef2d30207b00d3eedc9a56c7dbc |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/fisher_metric.cpython-311.pyc | 1 | 1653 | 4fe0055bd0f91f369786a9b7de7b80301730504f148c16363abc5ab6f4a4fdc6 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/function_space_actuation.cpython-311.pyc | 1 | 7169 | 9d78d423dd1a214f9563b8b5b519b44d0068a25c70d2472c5f345d3fcfe207d5 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/function_space_metrics.cpython-311.pyc | 1 | 19082 | 1886a0cece4380070039de25c4bf1de5768c069692607752466db168fbec8e21 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/kan_source_bank.cpython-311.pyc | 1 | 3473 | 7443d1df08eaffc7e1ab45dbf5e37db9b4aaff6eda0b4ad9f3b8f06c2f34a224 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/linec_readback.cpython-311.pyc | 1 | 398 | 148109d5b359008579b9d8f0486b5bb67613b8bca136081a1a50aca65113ff89 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/loss_interface.cpython-311.pyc | 1 | 4703 | 1ae31be4395ed251c87daae1c1ed215e210c8efbaa5c6ec27cec3bf219a78a6f |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/matrix_block.cpython-311.pyc | 1 | 1316 | ed4790e9082f02cb9b4dde873a030e5152d8dce4ce4816636f5bb58a290cc445 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/mechanisms.cpython-311.pyc | 1 | 215496 | d3998710bbe8c17aac2892baf292b6f8875c8fa1b8bd8c6dd463886988d8d328 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/metric_projection.cpython-311.pyc | 1 | 1990 | 10e007769ffdf21677706f89d15ac4d0ee7d1bcc066968c5e168fbf7d9ca07cf |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/optimizers.cpython-311.pyc | 1 | 5372 | e1703c6fe5b7de6f3f7818afcdb035ed5a749b137ef44c5643ae8889941652bb |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/poprisk_snr.cpython-311.pyc | 1 | 1231 | 4b5c09a64c104db6790ef6466c2ada0a10715a594cc4f7e0c5e0bbe03827aa52 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/poprisk_source.cpython-311.pyc | 1 | 333 | 3b20654e8bb3b94a5ca65f2c368a037856a93f34fb68b6a8eca1e4ae498346cf |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/precommit_selector.cpython-311.pyc | 1 | 7539 | 5bbef6308822fcf79a000d1da5abfc15fc37e47b86c489d417b128536ee93da2 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/rkhs_metric.cpython-311.pyc | 1 | 1254 | b7a58e448d0027ff4cc259d09188dde68b6cc6a6989e20dfae3be6a3389708a3 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/slow_state.cpython-311.pyc | 1 | 1024 | e9801b0255e21442772bee13bb9d6236229610efb55daabc9193dc1f564a3a0f |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/sobolev_metric.cpython-311.pyc | 1 | 1385 | 20f1db3a3fa89e2fd04862789907f2d512a7190beaef5005739c9172581c4a2c |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/source_chain.cpython-311.pyc | 1 | 8780 | b354651a6eae4052ce885baa3d8917079f8e7d8432a0d09e30b9283c5d4c8069 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/source_channel.cpython-311.pyc | 1 | 12626 | 9f5a215c64fb8379ca2b9ab48b84b9aee6c97fd2851953bba08d6f0311b162f6 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/source_preservation.cpython-311.pyc | 1 | 2653 | b1eed6cf2cf8217eee1c04b602a094982d6064ba2bb6485af186559226efea51 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/source_state.cpython-311.pyc | 1 | 1639 | 68f20d65092f62f26eb6b9c7ce74821c9e9488b3bed0be2725325712099c4834 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/terminal_collapse.cpython-311.pyc | 1 | 7865 | 0d45441a4271ff908d76196f8b27b2e804a93ba51b051831cc48a0850c8b6bb9 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/terminal_erosion.cpython-311.pyc | 1 | 7060 | b37006f38a2702b2cec8646f41e066b6e4c7b50036f3b3eb128cc0a9070a3595 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/terminal_retention.cpython-311.pyc | 1 | 10259 | 472fd9908a15c0cc17aa5244acf4240f972b0fc5c19dc13483d57c129e885f00 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/types.cpython-311.pyc | 1 | 1829 | 7884fbeb875e7b0cbe9f3e6684c6a7f2b519eb9298575359062345b05a0a6cf1 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/__pycache__/update_semantics.cpython-311.pyc | 1 | 1969 | c48ba3e9498f626f98b72b6dc2dc3c17ee30d598701ce18490388269d07af9b4 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/audit.py | 1 | 956 | 1cd77be8d23994bcda66709d9637a8f2ad7b3baf2f330671421e8f52529f2c31 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/carriers.py | 1 | 956 | b02a55d3765a7f7b3220b421bd4e28d458a327ee26f7148acb1a234b2f4272ac |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/controls.py | 1 | 1776 | 112c6785ce87359c69fe7b194c8e86781ae57c05d2bb60fc05a21817d0127547 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/core.py | 1 | 5138 | 41e7dec2fdda1948caddb9ef1f930f1d26f6df8d76bed89899187f826cbee378 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/debt_accounting.py | 1 | 2727 | efa1948dde33255625609f7d71641649a3eea00b63b54bcc6c1a36ef54aab265 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/diffeomorphic_target.py | 1 | 4558 | 9400bb4b956727808ed8ae18831f1c054c36dae7ceb6d7e39c79bd5b93b771b3 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/fisher_metric.py | 1 | 717 | fa5c2ce8ba853aa5dac55bfe6299be4e2c5ee27d90bea8175f0330aba3f44f84 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/function_space_actuation.py | 1 | 3811 | 5b3ffc343f312b07dbcddf450fe1bc2ee51dd1ece430132af6221ff9b3f28ac2 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/function_space_metrics.py | 1 | 8791 | f420385b2025c37cd55581773c2e04138b7fef69b4c2f186f5ae0be604c34460 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/kan_source_bank.py | 1 | 2525 | 7e8855bb0fddddc2b6fac0c757471b2610583053567c714e9a3e86d4b1fc3d7e |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/linec_readback.py | 1 | 193 | e59a29285e5b4f9b5313866fde6b29f88264674e0875c506e9e051df1a9a0bda |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/loss_interface.py | 1 | 2169 | 049074e61dca605ae373c3e4e04b469dc991a0b6dda958de82e69b6b4e45bb6d |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/matrix_block.py | 1 | 447 | 1d737d81d863a637f35a8af9f6ea5666af7925a70db22bfcf4c8c49f2b72e011 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/mechanisms.py | 1 | 196385 | fde2b6222e91e58aea7c8e1383514f58dd5890b660ab16bdef9ad9a85bb68e58 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/metric_projection.py | 1 | 955 | 3ca37b93f912a3e65df384762471a5f03297d435956ce52a4ad5ffd8acd0184c |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/optimizers.py | 1 | 2597 | 6937c978a20a82491a2cfd2a5cae602e8903612a157efa4026d8c9a478d63d9d |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/poprisk_snr.py | 1 | 502 | e9018a4c58f04b8ea4eff85f143a5fe66db23bf945b4d9525f299a3fa9bf3ff1 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/poprisk_source.py | 1 | 169 | 54d19c4034eda008d1e3f0acc11a4c4ae78ff7807708218b8fb7f9101ac19e48 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/precommit_selector.py | 1 | 3920 | 3a1ad79b1f5e14b4d02a4512816d2fa7d4c5e9bf61dd6ca8ffd9501edc89bd46 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/rkhs_metric.py | 1 | 476 | 64936156e3017f72a07d18ed9d3e52e8958208531ef1d553d3c4874f625e282b |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/slow_state.py | 1 | 390 | c24ae16a02777eb4aa9281573666b5a94e32f169673eec6c3347ab2ca1c3c7e4 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/sobolev_metric.py | 1 | 530 | 73deffe772b9050415cd9b79e89087b42fd040611aa36c483594c18d7956e7b3 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/source_chain.py | 1 | 6982 | 8f78f54d28e195235e3319340eada9a821a97afc4e6bf7fbe386db80cd39d7ac |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/source_channel.py | 1 | 5671 | 189dc7ee1eaf9bf1796a92f3d4535838d00c439a5848677f95bfd5ccc4c7552a |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/source_preservation.py | 1 | 1325 | cdbc175914887e4b22f2eddb66561c70588c020e71d05862d36e2c596c09f20a |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/source_state.py | 1 | 725 | aa882d5c755920fde5961d17beec6eae913a5998c7fc18144d4f912371ff4140 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/terminal_collapse.py | 1 | 5961 | ace1528b37855807de9925ebebcf032d9488dc0a5f72338d6f9cffdedcab34aa |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/terminal_erosion.py | 1 | 4888 | a6646ea37c4cbfe313a18f08b85478867dea524c1b448e7c7a69bf13813f1348 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/terminal_retention.py | 1 | 7349 | 081da818eba371bf10b3788e6b6389352ab80c60938d233197314de234e036cf |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/types.py | 1 | 1343 | e9ebb7ebf88259bc8d7c175a1b23dcbaabfd01309a5abfb05d322b27903ead83 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/fu/update_semantics.py | 1 | 1049 | 5939b1e2ed25a91670f32825bf787ea2569dfd3b4d36baa7f10b0b19dd80b562 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__init__.py | 1 | 206 | 03c11645f21d6f8d21b9648bd9d61964792a660280e33f7b8bde4c5d8457f0b7 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/__init__.cpython-311.pyc | 1 | 354 | 922b53fc945b06daf963817676efeb6f8f622074ed50ed97baad6f79bf84c869 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/adamw_coupling_audit.cpython-311.pyc | 1 | 1290 | c27adebe7f835c521448deb34eab578546b8248f5e9c33fdf76a2acc7191231d |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/controller.cpython-311.pyc | 1 | 2772 | 9d244d54b89a4275f3432e346ecc70962e172a39e33b368981bf2029444f63db |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/functional_mechanisms.cpython-311.pyc | 1 | 529 | 746129bcb636f3fc931bcaba84f46288c206def8693921d5d234c7e1b7b477ee |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/geometry.cpython-311.pyc | 1 | 960 | e145884d4a7ed4d5a55058bcbd5f00a8ebe6a782392534d74684e105e42156d4 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/geometry_certificate.cpython-311.pyc | 1 | 13845 | 7d0663866b7c02e1dbd223129f2b8b100d7aa1fd441736a16e78dab774d925e2 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/guards.cpython-311.pyc | 1 | 638 | 0dcc399c2478b349bccfdaa3a7b7b278b880fea1a5c3bdc9a7c0ea1511d9c204 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/lq_functional_predictor.cpython-311.pyc | 1 | 12941 | e328224b8acf1af1b131ad95cc6cc5c2ff2156c69cfacabf69adc0f1f44be63d |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/lq_output_space_functional.cpython-311.pyc | 1 | 18500 | 5dd5c0a9bf08d316cfe808bed53588f8563043395894741127526d3ae6136620 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/manifold_channel_geometry.cpython-311.pyc | 1 | 28600 | ba9022b94210f7c39953b59ae249257a75e5631bc82a02599c1387427f0def74 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/manual_optimizers.cpython-311.pyc | 1 | 456 | c69565f7349afc379af31487e117c71a478ab5a16202f8e937ce6c6e8d86cac0 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/mlp_functional.cpython-311.pyc | 1 | 47641 | 7e9d246d16bd7f397c4328f03cfc7cde65cac2b349add89da290444330ee43ad |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/role_budget.cpython-311.pyc | 1 | 705 | e0210b4d156ac95353339d6b987fbd771ecddcd1efbcc9e837e029a052a5d128 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/second_diff.cpython-311.pyc | 1 | 293 | 4eea27dd387ca24342b5fc38c06eff0f2e477f5c1395ce7a90450fab7d38496e |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/snr_gated_lq.cpython-311.pyc | 1 | 42002 | d1b038df52f275d3127554b9c50527c03132996fd4d8d840538b74bb56ff2076 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/__pycache__/update_tensor.cpython-311.pyc | 1 | 593 | 6a440a5076d9e28d1e875712128c767316ecdd5137f2cb434738cb31e00d62b4 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/adamw_coupling_audit.py | 1 | 866 | a8c46bbebb951f05a117ffbc536001be7a992cc2a8362c4b14dceb1ffc00c8cf |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/controller.py | 1 | 1090 | d091b1ed2385b235304594cb9bf288cacf8fd3fbf87e88dd03d1b5bdae001abc |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/functional_mechanisms.py | 1 | 358 | 712cfd392064975c72c849a17a1abf16584585362d724c7a5caee89e6dccc07e |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/geometry.py | 1 | 398 | 144c5a1d0af8172947c605838652c9c646920d9f3937c654263c0f19ee7a7025 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/geometry_certificate.py | 1 | 9654 | a901b2d974bb71d8625a0b952c9643d500f173e9d2cf9a7a6cb51343a7c61d3a |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/guards.py | 1 | 220 | 52da0d02414931cec617143f5bf07e7e6576879b52358ebee388980d3c3be572 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/lq_functional_predictor.py | 1 | 7988 | caf7b9d72f1969a26654605cce91786d549dce7affc1c4ae3b837312fc06a799 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/lq_output_space_functional.py | 1 | 9950 | 0c55e598e7465dda65e4233d955a3756bb0adbea71de3b9e6295f86b54cd7d08 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/manifold_channel_geometry.py | 1 | 13490 | 6eac1d96efc4e83e190d688680d5a9952981f1c4ace6f589b0cdce560dd04a98 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/manual_optimizers.py | 1 | 269 | caaf1c0beda8c2399b0cd2d8fb21ba789811b5d2018a2081be250bf02e8e52c8 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/mlp_functional.py | 1 | 23475 | 4d975294813e8ac90ee80a02e23af74d21338209ef52ac9b48265a669ef52735 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/role_budget.py | 1 | 204 | b32459cec81d80c2767a3d7e140bbe1a68ba7a40c4348d8c09bbbde9180d2f16 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/second_diff.py | 1 | 127 | 71def41caa2d938ebb74eb85dc7280278641d6b64aaacd787749006c823a081d |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/snr_gated_lq.py | 1 | 22499 | cd04043742c9e82e5e6a0bbb9d71010133bcd4c7753707cbf28e202087295c5e |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/functional/update_tensor.py | 1 | 401 | d9000d7e39324a645c89465429851f89974dbebf4cfb0494ae7a3e1ac8f7aeda |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__init__.py | 1 | 41 | f0abab685f0c2ac70fcd6b168064f0461407d55cb26a9729686f6019882b7832 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/__init__.cpython-311.pyc | 1 | 331 | 8fe394a17350611118a27e267aec64f025e19e00c1051931d81574407ec743d4 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/che_official.cpython-311.pyc | 1 | 327 | 3b8acbd468c18367995525628cf34ab6b707170f6bb31ce74b429a8e0cde7a75 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/cheby_fused.cpython-311.pyc | 1 | 1139 | ec6ead91bd50e60a9081d7cdcd88cb56ebc373f8d82ea5c915064c92cd411426 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/compiled_head.cpython-311.pyc | 1 | 222 | 4cfdc0535a3a175784fe31f8fedfb401aff6da4426d2cc2baa75f764bae69049 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/fou_official.cpython-311.pyc | 1 | 329 | 0c3f5356008654b6e32b33f090532d720b2694ad5da9e3149d7e61687d8e1539 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/fourier_fused.cpython-311.pyc | 1 | 1063 | f1dd627921c77df797256cc8e3710f00f7171970e281e849d876bbee58216353 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/fused_chebyshev_k3.cpython-311.pyc | 1 | 50893 | 12f17eef551c2a34fea518ac328471bebb8068e228366bf582521fdef0967524 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/fused_fourier_k2.cpython-311.pyc | 1 | 71805 | 7dee5781930b2c881976f5fe6e7fad70d575b28e41a394b32fd331d583275c74 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/fused_hat_wavelet.cpython-311.pyc | 1 | 18285 | 69be0a0b6558b28b8c0ff112ef6bc66a44136a2bdac38d53aa7e75710adc41ea |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/fused_hinge_quadratic.cpython-311.pyc | 1 | 138228 | 2b06d2d29b55148e604b02bdb240f37587be1bd2ca01f42b2c1ec82a09300bc9 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/fused_rational.cpython-311.pyc | 1 | 620 | 35009c9f89fadc025cf80ce848e326e0f288a77e437bb9864ac654672026256c |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/fused_rbf.cpython-311.pyc | 1 | 17701 | 90d9f594db649e9d5ee27f58dfc68f5c477cd7711e1132b64f7f4d7903147b1d |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/fused_rbf_local.cpython-311.pyc | 1 | 429 | 3f8c2940a6224d8b6195bdb8e0b7d6bb3b933fb33bdff977962c737b08b69ee5 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/head_backward.cpython-311.pyc | 1 | 222 | f09488c768c7a1ea99511c822c1a8b1d5cec624b1e96e5910a66f0e4c20e0cd6 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/lq_fused.cpython-311.pyc | 1 | 1020 | adcfa92da1c91505bfdfaeb561cb2d330906ec7e644cae93e606026bfba716a3 |
| results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/clean_unzip_self_test/02_SOURCE_TREE/dgkan/kernels/__pycache__/rational_fused.cpython-311.pyc | 1 | 891 | 7ecc6106e2780218d376391f9c19d59c2a29cd53bad5bb7ead66c1ad0a51d0d0 |

_仅显示前 160 / 1850 rows；完整 CSV 见 artifact。_

## 继续推进复盘：Metric Target / Window / Composite 修复

更新时间：2026-06-06 06:38:35 +0800

### 结论

- v22.05 目标仍未达成。
- functional route 仍为 `F0-MetricNoEffect`；三条 continuation 修复线的 `productive_metric_rows=0`。
- D-RAT/D-RBF 的 official fused blocker 在本轮未解除；因此即使 functional 修复有进展，也不能 promotion。
- 本轮没有把 smoke positive、h800 positive 或 row-positive 伪称为 promotion。

### 本轮做了什么修改

| file | change | audit note |
| --- | --- | --- |
| `dgkan/fu/mechanisms.py` | 新增 M236-M239 metric-target 机制，复用 existing exact-readout function-space solver；新增 `_metric_target_readout_update`。 | 机制只构造 train-only update 和 diagnostics，不生成实验数值。 |
| `experiments/run_v17_common.py` | M236-M239 接入 alt-period target runner；新增 `source_stop_steps` 和 `post_stop_mechanism/post_stop_fu_lr` 调度。 | 用于 early-window 和 composite 修复；旧 spec 默认 0/空，不改变旧路径。 |
| `experiments/run_v21_01_source_retention.py` | 注册 `MLP-MFT-*`、`MLP-MFTW-*`、`MLP-MFTC-*` continuation spec，并传递 stop/post-stop 参数。 | spec 名称直接写出 hypothesis，便于复现。 |
| `experiments/run_v22_05_metric_first_fu.py` | 扩展 metric ID 白名单。 | 防止 continuation summary 被过滤。 |
| `experiments/run_v22_05_metric_autopsy.py` | 扩展 metric ID 白名单。 | 防止 continuation autopsy 被过滤。 |
| `experiments/run_v22_05_s012_truth_gate.py` | S0 semantic audit tiny model 改为带 `w2` 和 `frozen_readout_features`。 | 避免 exact-readout 类机制 fallback 到普通梯度后造成假性 semantic alias。 |

### Code Audit / Truth Gate

| check | result |
| --- | --- |
| `py_compile` | pass |
| `run_v22_05_s012_truth_gate.py --mode mechanism_contracts` | pass=1；mechanism_contracts=22/22；alias_undeclared=0 |
| `run_v22_05_s012_truth_gate.py --mode all` | pass=1；required_source_files=31/31；metric_tests=19/19；kernel_gradcheck=4；profiler_phase_tests=3/3 |

### Full Result Summary

#### A. Metric Target Repair Full

Artifacts:

- `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_repair_full_summary/v22_05_metric_first_mlp_summary.csv`
- `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_repair_full_summary/v22_05_metric_failure_verdict.json`

Route:

| metric_rows | best_metric_v21_id | best_h4800 | best_row_h4800_positive_count | productive_metric_rows | functional_route |
| --- | --- | --- | --- | --- | --- |
| 4 | `MLP-MFT-T0-LossCotangent` | -1.2270795239342585 | 0 | 0 | `F0-MetricNoEffect` |

Rows:

| v21_id | h800 | h1600 | h3200 | h4800 | row_h4800_positive_count | productive_h4800_candidate |
| --- | --- | --- | --- | --- | --- | --- |
| `MLP-MFT-T0-LossCotangent` | -0.4429604742262099 | -0.8725063535902235 | -1.15487715933058 | -1.2270795239342585 | 0 | 0 |
| `MLP-MFT-T3-LowDegreeReadout` | 0.06114065647125244 | -0.12762358453538683 | -0.4208578136232164 | -0.5672601461410522 | 0 | 0 |
| `MLP-MFT-T4-B1Transfer` | 0.11328754160139295 | -0.026189664999643963 | -0.3898531993230184 | -0.518005026711358 | 1 | 0 |
| `MLP-MFT-T5-SourceProjectedB3Null` | 0.10569682386186388 | 0.04686853620741102 | -0.21551487843195596 | -0.3288894295692444 | 1 | 0 |

Autopsy:

| dominant_failure_mode | best_v21_id | best_source_h3200_mean | best_source_h4800_mean | productive_h4800_rows |
| --- | --- | --- | --- | --- |
| `EarlySourceFormationFailed;ControlEquivalent;MetricNoEffect` | `MLP-MFT-T5-SourceProjectedB3Null` | -0.21551487843195596 | -0.3288894295692444 | 0 |

Evidence: T3/T4/T5 can form weak h800 source in full matrix, and T5 remains positive at h1600, but all rows are negative by h3200/h4800 against the best matched control. Therefore source is not continuously retained.

#### B. Metric Target Early Window Full

Artifacts:

- `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_window_full_summary/v22_05_metric_first_mlp_summary.csv`
- `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_window_full_summary/v22_05_metric_failure_verdict.json`

Route:

| metric_rows | best_metric_v21_id | best_h4800 | best_row_h4800_positive_count | productive_metric_rows | functional_route |
| --- | --- | --- | --- | --- | --- |
| 4 | `MLP-MFTW-T3-LowDegreeReadout-stop800` | -0.2558178702990214 | 1 | 0 | `F0-MetricNoEffect` |

Rows:

| v21_id | h800 | h1600 | h3200 | h4800 | row_h4800_positive_count | productive_h4800_candidate |
| --- | --- | --- | --- | --- | --- | --- |
| `MLP-MFTW-T3-LowDegreeReadout-stop800` | 0.14687261316511366 | -0.0054810312059190534 | -0.17028290033340454 | -0.2558178702990214 | 1 | 0 |
| `MLP-MFTW-T4-B1Transfer-stop800` | -0.04326503806644016 | -0.19005702601538765 | -0.494251012802124 | -0.6383128431108263 | 0 | 0 |
| `MLP-MFTW-T5-SourceProjectedB3Null-stop1600` | 0.10569682386186388 | 0.04686853620741102 | -0.23033533493677774 | -0.33473769823710126 | 1 | 0 |
| `MLP-MFTW-T5-SourceProjectedB3Null-stop800` | 0.0373003880182902 | 0.06395339303546482 | -0.09545522928237915 | -0.20449439022276136 | 2 | 0 |

Autopsy:

| dominant_failure_mode | best_v21_id | best_source_h3200_mean | best_source_h4800_mean | productive_h4800_rows |
| --- | --- | --- | --- | --- |
| `EarlySourceFormationFailed;ControlEquivalent;MetricNoEffect` | `MLP-MFTW-T5-SourceProjectedB3Null-stop800` | -0.09545522928237915 | -0.20449439022276136 | 0 |

Evidence: early-window reduced terminal damage versus always-on target (`best h4800 -0.204494` vs `-0.328889` for non-window T5), but still did not beat SGD control at h3200/h4800. It is an improvement in failure degree, not a pass.

#### C. Metric Target + Post-Stop Terminal Anchor Full

Artifacts:

- `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_composite_full_summary/v22_05_metric_first_mlp_summary.csv`
- `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_composite_full_summary/v22_05_metric_failure_verdict.json`

Route:

| metric_rows | best_metric_v21_id | best_h4800 | best_row_h4800_positive_count | productive_metric_rows | functional_route |
| --- | --- | --- | --- | --- | --- |
| 4 | `MLP-MFTC-T3-stop800-postM181` | -0.25056517124176025 | 1 | 0 | `F0-MetricNoEffect` |

Rows:

| v21_id | h800 | h1600 | h3200 | h4800 | row_h4800_positive_count | productive_h4800_candidate |
| --- | --- | --- | --- | --- | --- | --- |
| `MLP-MFTC-T3-stop800-postM181` | 0.14687261316511366 | -0.00575819280412462 | -0.16781231429841784 | -0.25056517124176025 | 1 | 0 |
| `MLP-MFTC-T5-stop1600-postM181` | 0.10569682386186388 | 0.04686853620741102 | -0.22743061516020033 | -0.3314040568139818 | 1 | 0 |
| `MLP-MFTC-T5-stop800-postM181` | 0.046346147855122886 | -0.061058759689331055 | -0.34360651175181073 | -0.49021852016448975 | 0 | 0 |
| `MLP-MFTC-T5-stop800-postM218` | 0.0373003880182902 | 0.06306965483559503 | -0.09217674864663018 | -0.19945857259962294 | 2 | 0 |

Autopsy:

| dominant_failure_mode | best_v21_id | best_source_h3200_mean | best_source_h4800_mean | productive_h4800_rows |
| --- | --- | --- | --- | --- |
| `EarlySourceFormationFailed;ControlEquivalent;MetricNoEffect` | `MLP-MFTC-T5-stop800-postM218` | -0.09217674864663018 | -0.19945857259962294 | 0 |

Evidence: post-stop M218 slightly improved best h4800 versus window-only (`-0.199459` vs `-0.204494`) and h3200 (`-0.092177` vs `-0.095455`), but the values remain below the matched SGD control (`control_max_h4800=0.0`), so no productive group exists.

### Insight / Evidence Chain

- The original metric-first/source-memory family M220-M235 failed because parameter-space metric projection did not create retained source. M236-M239 moved the update into train-only readout/function-space target writing, which produced real early signal in T3/T4/T5, so the wiring was not inert.
- The failure shifted from “no early source at all” to “early source not continuously retained”: h800 positive appears in several candidates, but h3200/h4800 revert below control.
- Early-windowing is directionally better than always-on target, suggesting late application of metric target contributes to washout. However, stopping target updates alone does not preserve the source against SGD/optimizer erosion.
- Post-stop terminal anchor M181/M218 did not solve the retained-source problem in this narrow composite runner. It only marginally improved T5 stop800 terminal values and remained control-equivalent.
- The best continuation terminal value is `MLP-MFTC-T5-stop800-postM218` with h4800=-0.19945857259962294, row-positive=2/9, productive=0. This is strictly not promotion-worthy.
- Because h3200 is negative for all continuation rows, `R4800_over_3200` is not evaluable as a positive retained-ratio gate. The blocker is therefore earlier than ratio: continuous source chain fails before terminal retention can be assessed.

### Updated Decision

| item | decision |
| --- | --- |
| promotion_allowed | 0 |
| functional_route | `F0-MetricNoEffect` |
| continuation_best | `MLP-MFTC-T5-stop800-postM218` by least-negative h4800 in autopsy |
| blocking_metric | `official_fused_missing;metric_target_continuous_h3200_missing;h4800_below_control` |
| exhausted_lines | M236-M239 always-on metric target; MFTW early-window; MFTC metric-target plus post-stop M181/M218 anchor |
| next_codex_action | do not keep increasing metric-target strength; next viable repair likely needs source-observability/optimizer-conflict gating that directly preserves h3200 before h4800, plus separate official D-RAT/D-RBF fused wiring |

### Reproducibility Notes

- Full shard command journals:
  - `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_repair_full/v21_01_command_journal.csv`
  - `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_window_full/v21_01_command_journal.csv`
  - `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/metric_target_composite_full/v21_01_command_journal.csv`
- Execution details were appended to `docs/DG-KAN_v22.05_MetricFirst_FunctionalUpdate_BasisEfficiency_4GPU_执行日志.md`.

## 2026-06-06 07:56:20 +0800 Continuation: H-C2 Optimizer Projection / Stable Seed Audit

### 是否达成目标

未达成。v22.05 functional route 仍为 `F0-MetricNoEffect`，promotion_allowed 仍为 0。

本轮解决了一个重要复现 blocker：`run_v21_01_source_retention.py` 原先用 `job_index` 生成 train seed，导致同一 spec 放入不同 `--spec-ids` 队列时初始化会改变。这个问题会污染 v22.04 D1b baseline 与 v22.05 H-C2 reference 的对比。本轮已改为 stable semantic seed，并用 smoke 验证 D1b 在不同队列中 `train_seed=1186690` 且 h100 完全一致。

### 本轮做了什么修改

| file | change | audit note |
| --- | --- | --- |
| `experiments/run_v21_01_source_retention.py` | 新增 `stable_train_seed(job)`，基于 `scope/carrier/basis_repair_variant/dataset/seed/v21_id/mechanism` 的 sha256 派生 train seed；matrix/trace 写入 `train_seed`。 | 修复队列组成影响初始化的复现漏洞；旧 artifact 不改写，新实验使用 stable seed。 |
| `dgkan/fu/mechanisms.py` | 新增 M240-M244 H-C2 optimizer projection 机制集合；M243/M244 为 h4000 recompute anchor。 | `make_update` stateless fallback 与 M220-G0-L2 同向，已声明 semantic alias，不伪装 non-collapse；科学差异只在 runner stateful path。 |
| `experiments/run_v17_common.py` | 接入 H-C2 stateful branch：h3200 或 h4000 捕获 source anchor，统计 negative projection fraction / cumulative projection / removed norm，并对 optimizer gradient 去除 destructive component。 | H-C2 投影实际执行，projection norm 非零；不生成实验数值。 |
| `experiments/run_v22_05_s012_truth_gate.py` | 新增 M240-M244 expected mechanisms；修正 route JSON 只基于当前 invocation rows，CSV 保留历史失败行。 | 避免历史 failed row 让已修复 gate 永久显示 fail。 |
| `experiments/run_v22_05_metric_first_fu.py` / `experiments/run_v22_05_metric_autopsy.py` | 增加 H-C2 IDs；诊断字段从 h4800 raw trace 聚合，缺失时回退 matrix 宽表/final。 | 修复 H-C2 projection diagnostics 被读成 `nan` 的汇总问题。 |

### Code / Gate

| check | result |
| --- | --- |
| py_compile | pass |
| mechanism_contracts current invocation | pass=1; `27/27;alias_undeclared=0` |
| route JSON | `mode=mechanism_contracts`, `pass=1` |
| note | `v22_05_code_truth_gate.csv` 保留历史 failed rows，用于审计；当前 route 只看本次 invocation。 |

### Seed-invariance Evidence

| run | v21_id | train_seed | h100 | note |
| --- | --- | --- | --- | --- |
| `seed_invariance_d1b_only_smoke` | `MLP-D1b-roworth-hidden-readout` | 1186690 | -0.5388312339782715 | D1b + controls |
| `seed_invariance_d1b_hc2_context_smoke` | `MLP-D1b-roworth-hidden-readout` | 1186690 | -0.5388312339782715 | H-C2 + D1b + controls |

### H-C2 h3200 Projection Stable-Seed Full

Artifacts:

- `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_stableseed_full/`
- `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h3200_projection_stableseed_analysis/`

Route:

| metric_rows | best_metric_v21_id | best_h4800 | best_row_h4800_positive_count | productive_metric_rows | functional_route |
| --- | --- | --- | --- | --- | --- |
| 3 | `MLP-HC2-HalfProjection` | -0.39466531409157646 | 1 | 0 | `F0-MetricNoEffect` |

Rows:

| v21_id | h100 | h400 | h800 | h1600 | h3200 | h4000 | h4800 | R4800/3200 | pos4800 | neg_frac_h4800 | projection_norm_h4800 | productive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `MLP-HC2-HalfProjection` | -0.09781867927975124 | 0.27351975440979004 | 0.1781515810224745 | -0.07149864567650689 | -0.32287028100755477 | -0.37588562568028766 | -0.39466531409157646 |  | 1 | 0.08952737872163231 | 3.7790145218325253 | 0 |
| `MLP-HC2-L2Projection` | 0.01494614283243815 | 0.38784174124399823 | 0.297189368142022 | 0.060526384247673884 | -0.17306888103485107 | -0.20965660280651516 | -0.21680427259869045 |  | 2 | 0.03032826705531265 | 1.740065481919828 | 0 |
| `MLP-HC2-NoProjection` | 0.0325371159447564 | 0.4201142191886902 | 0.3337708314259847 | 0.09626168674892849 | -0.13286163409550986 | -0.1765130360921224 | -0.18816321425967747 |  | 3 | 0.04719272676799223 | 0.0 | 0 |

Autopsy:

| dominant_failure_mode | best_v21_id | best_source_h3200_mean | best_source_h4800_mean | terminal_retention_evaluable_rows | productive_h4800_rows |
| --- | --- | --- | --- | --- | --- |
| `MetricNoEffect` | `MLP-HC2-NoProjection` | -0.13286163409550986 | -0.18816321425967747 | 0 | 0 |

Evidence: H-C2 projection branch is active because L2/Half have nonzero h4800 projection norm, but both projected variants are worse than NoProjection at h4800. More importantly, all rows have h3200 < 0.005, so R4800/3200 is not evaluable as a retained-ratio gate.

### H-C2 h4000 Recompute Projection Stable-Seed Full

Artifacts:

- `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h4000_recompute_projection_stableseed_full/`
- `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc2_h4000_recompute_projection_stableseed_analysis/`

Route:

| metric_rows | best_metric_v21_id | best_h4800 | best_row_h4800_positive_count | productive_metric_rows | functional_route |
| --- | --- | --- | --- | --- | --- |
| 3 | `MLP-HC2-H4000HalfProjection` | -0.25310056077109444 | 1 | 0 | `F0-MetricNoEffect` |

Rows:

| v21_id | h100 | h400 | h800 | h1600 | h3200 | h4000 | h4800 | R4800/3200 | pos4800 | neg_frac_h4800 | projection_norm_h4800 | productive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `MLP-HC2-H4000HalfProjection` | 0.04652809434466892 | 0.40487202008565265 | 0.31058239936828613 | 0.058252308103773326 | -0.19116275840335423 | -0.23387962579727173 | -0.25310056077109444 |  | 1 | 0.08836177000971007 | 1.9766875295942252 | 0 |
| `MLP-HC2-H4000L2Projection` | -0.0054019490877787275 | 0.36469639672173393 | 0.27337649795744157 | 0.04078931940926446 | -0.1924941341082255 | -0.2239998843934801 | -0.24344299899207222 |  | 3 | 0.09071993341656263 | 4.36217813786061 | 0 |
| `MLP-HC2-NoProjection` | 0.0325371159447564 | 0.4201142191886902 | 0.3337708314259847 | 0.09626168674892849 | -0.13286163409550986 | -0.1765130360921224 | -0.18816321425967747 |  | 3 | 0.04719272676799223 | 0.0 | 0 |

Autopsy:

| dominant_failure_mode | best_v21_id | best_source_h3200_mean | best_source_h4800_mean | terminal_retention_evaluable_rows | productive_h4800_rows |
| --- | --- | --- | --- | --- | --- |
| `MetricNoEffect` | `MLP-HC2-NoProjection` | -0.13286163409550986 | -0.18816321425967747 | 0 | 0 |

Evidence: 按计划 9.2 的 fallback 重新在 h4000 捕获 source anchor 后，projection 仍比 NoProjection 差：h4000-L2 h4800=-0.24344299899207222，h4000-Half h4800=-0.25310056077109444，而 NoProjection h4800=-0.18816321425967747。

### D1b Stable-Seed Reference

| run | v21_id | h100 | h800 | h3200 | h4800 |
| --- | --- | --- | --- | --- | --- |
| H-C2 stable-seed full | `MLP-D1b-roworth-hidden-readout` | -0.14007573657565647 | -0.11334170235527886 | -0.483604629834493 | -0.46642043193181354 |
| control | `CTRL-SGD` | -0.8265806701448228 | -0.11783427662319607 | 0.0 | 0.0 |
| control | `CTRL-AdamW` | 0.0 | -0.595277746518453 | -1.5224400228924222 | -1.7113646335071988 |

Important: 这说明本轮 stable-seed protocol 下，旧 v22.04 D1b near-miss 不再复现。旧 v22.04 artifact 仍是真实历史结果，但它使用 job-index seed，不能直接和修复后的 stable-seed H-C2 队列混作同一 seed protocol 的结论。

### Insight / Evidence Chain

- H-C2 的核心假设是 h3200 后 ordinary optimizer update 含有破坏 source 的负投影。实验显示 negative_projection_fraction 不为 0，投影也确实移除了 component；因此 H-C2 不是 wiring no-op。
- 但 source_h3200 在 stable-seed full 中已经为负，terminal_retention_evaluable_rows=0。也就是说，h4800 gate 的 blocker 更早：continuous source chain 在 h1600->h3200 之间断掉。
- h3200-anchor projection 与 h4000-recompute projection 都比 NoProjection 差；这支持 “source direction not stable / projection overconstrains already-eroded source”，而不是 “只要去掉 post-h3200 destructive component 就能保留 source”。
- NoProjection 是两个 H-C2 stable full 的 autopsy best by h4800（-0.18816321425967747），但仍低于 control max h4800=0.0，且 row-positive 只有 3/9，所以仍为 control-equivalent / metric-no-effect。
- 本轮修复了一个会导致后续误判的 seed protocol bug。后续所有 v22.05 scientific rows 应以 stable semantic seed 作为可比协议；旧 job-index seed artifact 只能作为历史证据，不能和 stable-seed rows 混成同一 independent rerun。
- D-RAT/D-RBF official fused blocker 未在本轮解决；promotion 仍同时受 `official_fused_missing` 与 functional source-chain failure 阻塞。

### Updated Decision

| item | decision |
| --- | --- |
| promotion_allowed | 0 |
| functional_route | `F0-MetricNoEffect` |
| H-C2 decision | h3200 projection failed; h4000 recompute fallback failed |
| best stable H-C2 row | `MLP-HC2-NoProjection`, h4800=-0.18816321425967747, pos4800=3/9, productive=0 |
| blocking_metric | `official_fused_missing;stable_seed_continuous_h3200_missing;projection_worse_than_no_projection;h4800_below_control` |
| next_codex_action | stop increasing H-C2 projection strength; next repair must target pre-h3200 source-chain formation under stable seed, then re-test terminal projection only if h3200 becomes positive/evaluable |

## 2026-06-06 08:17:41 +0800 Continuation: H-C3 Supplemental Metric Targets

### 是否达成目标

未达成。补齐 H-C3 的 T1/T2/T6/T7/T8 metric-target 分支后，functional route 仍为 `F0-MetricNoEffect`，promotion_allowed 仍为 0。

本轮 H-C3 supplemental 只解决“计划覆盖不足 / summary 漏读”的问题，没有产生可 promotion 的 h4800 source。所有 supplemental rows 均低于 matched control max h4800=0.0，且没有任何 row 满足 h3200 positive / R4800/3200 evaluable gate。

### 本轮做了什么修改

| file | change | audit note |
| --- | --- | --- |
| `dgkan/fu/mechanisms.py` | 新增 M245-M249：`MetricTargetSplitConsensusFU`、`MetricTargetPopRiskSNRFU`、`MetricTargetDiffeomorphicNoFoldFU`、`MetricTargetRandomMatchedFU`、`MetricTargetSignFlippedFU`。 | 这些机制走 v22.05 metric-target readout solver；T7/T8 是 control，不计 promotion。 |
| `dgkan/fu/mechanisms.py` | 为 `_exact_readout_function_space_actuation_update` 增加 `target_kind="poprisk_snr"`，用 split ref 的 class-wise SNR 权重形成 T2 target，并记录 `PopRisk_SNR_score` 等诊断。 | 不是改名冒充 PopRisk；target 权重来自 train split 的 mean/variance。 |
| `experiments/run_v21_01_source_retention.py` | 注册 `MLP-HC3-T1/T2/T6/T7/T8` specs，并加入默认 MLP id list。 | 后续复现可直接用 spec id，不需要读代码找 mechanism。 |
| `experiments/run_v22_05_metric_first_fu.py` / `experiments/run_v22_05_metric_autopsy.py` | 纳入 H-C3 supplemental IDs；autopsy 的 metric_families_tried 补充 `T0/T1/T2/T3/T4/T5/T6/T7/T8 metric-target`。 | 防止 full run 数据落盘但 summary/autopsy 忽略。 |
| `experiments/run_v22_05_metric_first_fu.py` | 修正 best-row 选择：有 R4800/3200 时按 R；R 不可评估时按 h4800/h3200。 | 修复所有 R 为空时误选第一行的问题；不改变 promotion 判据。 |
| `experiments/run_v22_05_s012_truth_gate.py` | expected mechanisms 加入 M245-M249。 | 当前 mechanism_contracts route pass=1，`32/32;alias_undeclared=0`。 |

### Code / Gate

| check | result |
| --- | --- |
| py_compile | pass |
| mechanism_contracts current invocation | pass=1; `32/32;alias_undeclared=0` |
| route JSON | `mode=mechanism_contracts`, `pass=1` |

### H-C3 Supplemental 4GPU Full

Artifacts:

- `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_full/`
- `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/hc3_metric_target_supplemental_full_summary/`

Route:

| metric_rows | best_metric_v21_id | best_h4800 | best_row_h4800_positive_count | productive_metric_rows | functional_route |
| --- | --- | --- | --- | --- | --- |
| 5 | `MLP-HC3-T6-DiffeomorphicNoFold` | -0.3521238896581862 | 1 | 0 | `F0-MetricNoEffect` |

Rows:

| v21_id | metric_name | h100 | h400 | h800 | h1600 | h3200 | h4800 | R4800/3200 | pos4800 | control_equivalent | productive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `MLP-HC3-T1-SplitConsensus` | T1-SplitConsensus-metric-target | -0.867578837606642 | -0.16956350538465711 | -0.2524916993247138 | -0.5875975953208076 | -0.9005264573627048 | -0.9487915767563714 |  | 0 | 1 | 0 |
| `MLP-HC3-T2-PopRiskSNR` | T2-PopRiskSNR-metric-target | -0.822874903678894 | -0.05491258038414849 | 0.12300920486450195 | -0.0325201948483785 | -0.41105299525790745 | -0.5272373954455057 |  | 2 | 1 | 0 |
| `MLP-HC3-T6-DiffeomorphicNoFold` | T6-DiffeomorphicNoFold-metric-target | -0.7652776108847724 | -0.044696304533216685 | 0.14760091569688585 | 0.035605271657307945 | -0.25317005316416424 | -0.3521238896581862 |  | 1 | 1 | 0 |
| `MLP-HC3-T7-RandomMatched` | T7-RandomMatchedActuation-control | -0.8621733850902982 | -0.12975627846188015 | -0.15962915950351292 | -0.5851904153823853 | -1.032810754246182 | -1.1266324188974168 |  | 0 | 1 | 0 |
| `MLP-HC3-T8-SignFlipped` | T8-SignFlippedTarget-control | -0.8199358913633559 | -0.2975842422909207 | -0.21050731341044107 | -0.19581104649437797 | -0.5843649307886759 | -0.8378342456287808 |  | 1 | 1 | 0 |

Autopsy:

| dominant_failure_mode | best_v21_id | best_source_h100_mean | best_source_h3200_mean | best_source_h4800_mean | terminal_retention_evaluable_rows | productive_h4800_rows |
| --- | --- | --- | --- | --- | --- | --- |
| `EarlySourceFormationFailed;ControlEquivalent;MetricNoEffect` | `MLP-HC3-T6-DiffeomorphicNoFold` | -0.7652776108847724 | -0.25317005316416424 | -0.3521238896581862 | 0 | 0 |

Debt / energy evidence:

| v21_id | metric_energy_L2 | metric_energy_Sobolev | metric_energy_RKHS | LineC_debt_final | ECE_debt_final | Brier_debt_final |
| --- | --- | --- | --- | --- | --- | --- |
| `MLP-HC3-T1-SplitConsensus` | 8.990432787767431e-05 | 0.00026158134642173536 | 5.716332255663777e-05 | 0.996723903614812 | 0.29891761144002277 | 0.696660061677297 |
| `MLP-HC3-T2-PopRiskSNR` | 3.591221679420819e-05 | 0.00011560454842967576 | 2.1861955468062257e-05 | 0.9963584472565632 | 0.2532132781214184 | 0.64517832464642 |
| `MLP-HC3-T6-DiffeomorphicNoFold` | 0.0020597710562368673 | 0.0063420856208217125 | 0.000976399455390088 | 0.9937951761010978 | 0.24690434998936123 | 0.6244537532329559 |
| `MLP-HC3-T7-RandomMatched` | 2.8076423468014076e-05 | 8.468077102305769e-05 | 1.4056171753256624e-05 | 0.9915676150989354 | 0.28317708936002517 | 0.6940571467081705 |
| `MLP-HC3-T8-SignFlipped` | 3.058331694560934e-07 | 9.324173557700988e-07 | 1.7448558543699102e-07 | 0.9757442937061411 | 0.25895583381255466 | 0.7066738274362352 |

Strongest logged correlation to h4800 in this 5-row supplemental set: `Brier_debt_final`, pearson=-0.8989232882734184, n=5. 这只是小样本相关，不作为因果结论；它支持继续把 debt / calibration transition 放进后续 metric，而不是继续增加 target amplitude。

### Corrected H-C2 Route Readback

`run_v22_05_metric_first_fu.py` best-selection 修复后，两个 H-C2 stable summaries 的 route best 都与 autopsy 一致：

| attempt | best_metric_v21_id | best_h4800 | pos4800 | productive_metric_rows | functional_route |
| --- | --- | --- | --- | --- | --- |
| H-C2 h3200 projection stable-seed | `MLP-HC2-NoProjection` | -0.18816321425967747 | 3 | 0 | `F0-MetricNoEffect` |
| H-C2 h4000 recompute projection stable-seed | `MLP-HC2-NoProjection` | -0.18816321425967747 | 3 | 0 | `F0-MetricNoEffect` |

旧复盘上方 H-C2 route 表中 `best_metric_v21_id` 来自修复前 summary best-selection bug；科学判定以本节 corrected route 与 autopsy 为准。该 bug 只影响 best row display，不影响 productive gate。

### Insight / Evidence Chain

- H-C3 supplemental 覆盖了计划列出的剩余 target 类：T1 split-consensus、T2 PopRisk-SNR、T6 no-fold/diffeomorphic proxy、T7 random matched control、T8 sign-flipped control。T0/T3/T4/T5 已在前序 metric-target / window / composite runs 中覆盖。
- T2/T6 能形成 weak h800 source（0.1230 / 0.1476），但 h3200 都变负（-0.4111 / -0.2532），所以 blocker 仍是 continuous h3200 chain，而不是 h3200 后 ratio。
- T6 是 supplemental best by h4800，但 h4800=-0.3521，低于 control max h4800=0.0；T7/T8 controls 也失败，说明没有出现“bad control 也能打开 source”的假阳性。
- H-C2 projection 和 H-C3 target-family 都失败后，v22.05 已经把 functional failure 从泛泛的 source erosion 细化为：metric target 可产生 early weak source，但 source-channel/optimizer/debt coupling 在 h1600->h3200 之间把 chain 打断；terminal projection 对已 eroded source 只会更差。
- D-RAT/D-RBF official fused blocker 仍未解除；即使 H-C3 有 functional 改善，也不能绕过 official fused gate。

### Updated Decision

| item | decision |
| --- | --- |
| promotion_allowed | 0 |
| functional_route | `F0-MetricNoEffect` |
| H-C3 supplemental decision | T1/T2/T6/T7/T8 all failed; no productive h4800 group |
| best H-C3 supplemental row | `MLP-HC3-T6-DiffeomorphicNoFold`, h4800=-0.3521238896581862, pos4800=1/9, productive=0 |
| best stable H-C2 row after route fix | `MLP-HC2-NoProjection`, h4800=-0.18816321425967747, pos4800=3/9, productive=0 |
| blocking_metric | `official_fused_missing;stable_seed_continuous_h3200_missing;h4800_below_control;metric_target_control_equivalent` |
| next_codex_action | stop expanding H-C3 target variants; next viable metric must explicitly model train-only debt/calibration/source-channel coupling before h3200, or separately complete D-RAT/D-RBF official fused wiring |

## 2026-06-06 08:36:59 +0800 Continuation: D-RBF Official Fused Transition

### 是否达成目标

总体 v22.05 目标仍未达成：`promotion_allowed=0`，functional route 仍为 `F0-MetricNoEffect`，final route 仍为 `R3-DRATDRBFRepairBlocked`。

但本轮把 D-RAT/D-RBF blocker 缩小了：D-RBF 已从 `official_fused_missing` 推进到 official trainpath E1 opened；D-RAT 仍是 `MicroNearE1OfficialFusedBlocked` / `official_fused_missing`。这不是 functional promotion，只是 basis-kernel official repair 的真实进展。

### 本轮做了什么修改

| file | change | audit note |
| --- | --- | --- |
| `experiments/run_v21_common.py` | 将 `RBF21-compact-local-k4-smoke` 映射到 `RBF22.03-R1-compact-local-k4-no-dense`。 | 修复历史 runner 只走 `compact_rbf_stream_recompute`、没有打到 `rbf_k4_triton_l3_matmul` 的 wiring 缺口。 |
| `dgkan/kernels/fused_rbf.py` | `forward_matmul` block sizing 改为 shape-adaptive：batch>=1024 用 `BLOCK_B=64`，否则 32；input_dim<=16 用 `BLOCK_D=16`，<=32 用 32，否则 64。 | 该修改减少 D=8 场景的小 program 数和无效 dot width；不改变数学表达式。 |
| `experiments/run_v22_05_drat_drbf_repair.py` | 增加 D-RBF official transition probe：用 `PrimitiveKAN.manual_gradient_audit` + `profile_isolated(use_manual_ce=True)` 生成 official trainpath rows，并输出 `v22_05_drbf_official_transition.csv`。 | 不再把 v22.03 micro basis benchmark 伪称为 official；official rows 必须同时有 manual correctness、same manual kernel profiled、no-materialize 和 ratio。 |

### Code / Gate

| check | result |
| --- | --- |
| py_compile after D-RBF changes | pass |
| mapping self-check | `RBF21-compact-local-k4-smoke -> RBF22.03-R1-compact-local-k4-no-dense -> (4, 0, rbf_k4_triton_l3_matmul)` |
| S0.12 all gate after repair | pass=1; required files 31/31; mechanism contracts `32/32;alias_undeclared=0` |
| final route after repair | `R3-DRATDRBFRepairBlocked`; `promotion_allowed=0` |

### D-RBF Official Transition Evidence

Pre-wiring negative smoke:

| artifact | manual_kernel_variant | official_fused_kernel_complete | forward_ratio | blocker |
| --- | --- | --- | --- | --- |
| `drbf_official_transition_smoke` | `compact_rbf_stream_recompute` | 0 | 6.923751939295577 | mapping did not reach Triton official path |

After mapping, before block tune:

| artifact | batch | manual_kernel_variant | official | gradcheck | forward_ratio | step_ratio | decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `drbf_official_transition_smoke2` | 32 | `rbf_k4_triton_l3_matmul` | 1 | 1 | 3.8689944788355364 | 0.8192781292127397 | correctness opened, small-batch forward blocked |
| `drbf_official_transition_full` | 512 | `rbf_k4_triton_l3_matmul` | 1 | 1 | 4.366619876405829 | 1.799284986948501 | forward/backward/step blocked |
| `drbf_official_transition_full` | 2048 | `rbf_k4_triton_l3_matmul` | 1 | 1 | 2.8508475673038873 | 1.2031227290084638 | forward blocked |

After block tune and v22.05 route rerun:

| batch | manual_kernel_variant | manual_grad_relerr_max | manual_output_max_abs_error | official_fused_kernel_complete | functional_runner_kernel_match | no_materialize_complete | forward_ratio | backward_ratio | step_ratio | memory_ratio | E1_official |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | `rbf_k4_triton_l3_matmul` | 2.434284454011504e-07 | 3.725290298461914e-08 | 1 | 1 | 1 | 1.6466984844917827 | 0.41104739663217904 | 0.4825560429855775 | 0.9618257261410789 | 1 |
| 2048 | `rbf_k4_triton_l3_matmul` | 2.46920876634249e-07 | 3.3527612686157227e-08 | 1 | 1 | 1 | 1.5982390876322794 | 0.9396441248702825 | 0.9589440548168887 | 0.7969118982742961 | 1 |

### D-RAT / D-RBF Route After Repair

| carrier | profile_rows | micro_near_E1_rows | E1_official_rows | best_forward_ratio | best_step_ratio | best_memory_ratio | decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 7 | 6 | 0 | 1.0759703278296242 | 1.197399870964781 | 1.03125 | `MicroNearE1OfficialFusedBlocked` | `official_fused_missing` |
| D-RBF | 9 | 4 | 2 | 1.5982390876322794 | 0.4825560429855775 | 0.7969118982742961 | `OfficialRepairOpened` |  |

### Analysis / Insight / Evidence Chain

- 历史 `RBF21-compact-local-k4-smoke` artifact 使用 `compact_rbf_stream_recompute`，不是 official Triton trainpath；因此旧 `official_fused_missing` 对 D-RBF 是合理 blocker，不是数据错误。
- 当前代码已有 `PrimitiveKAN.manual_ce_forward_cache/backward_from_cache` 到 `fused_rbf.forward_matmul/backward` 的路径；缺口主要是 efficiency officialization runner 没把 D-RBF variant 映射到该 path。
- D-RBF official rows 的 correctness 证据来自 manual-vs-autograd：grad relerr 约 2.4e-7，output max error 约 3e-8，且 profiler row 显示同一 `rbf_k4_triton_l3_matmul` 被 train stream 使用。
- block tune 只改 launch shape，不改 basis math。standalone artifact 中 batch2048 forward ratio 从 2.8508475673038873 降到 1.6483218841013845；route rerun中 batch512/2048 都过 v22.05 D-RBF E1 official threshold。
- D-RAT 没有等价的 official fused trainpath row；当前仍只有 v22.03-style micro numerator/denominator benchmark，不能把 micro-near-E1 写成 official fused success。
- 即使 D-RBF official opened，v22.05 仍不能 promotion：functional route 是 `F0-MetricNoEffect`，KAN mapping 是 `KANSourceChannelMismatch`，且 D-RAT official fused 仍缺。

### Updated Decision

| item | decision |
| --- | --- |
| promotion_allowed | 0 |
| final_route | `R3-DRATDRBFRepairBlocked` |
| functional_route | `F0-MetricNoEffect` |
| D-RBF official fused | opened; 2 E1 official rows |
| D-RAT official fused | still blocked; no production fused trainpath row |
| remaining blocking_metric | `official_fused_missing` for D-RAT; plus functional metric/source-chain failure from H-C2/H-C3 |
| next_codex_action | either implement true D-RAT official fused trainpath, or pivot functional work to pre-h3200 train-only debt/calibration/source-channel coupling; do not expand old metric target variants further |

## 2026-06-06 10:24:22 +0800 Continuation: D-RAT/D-RBF Official Repair Closed, T9/T10/M252 Functional Route Still F0

### 是否达成目标

未达成 v22.05 promotion 目标。当前官方 root route 为：

| item | value |
| --- | --- |
| route | `F0-MetricNoEffect` |
| promotion_allowed | 0 |
| S0_pass | 1 |
| D-CHE_S1_pass / D-FOU_S1_pass | 1 / 1 |
| DRAT_DRBF_official_fused_blocked | 0 |
| functional_route | `F0-MetricNoEffect` |
| KAN_mapping_decision | `KANSourceChannelMismatch` |
| next_codex_action | `run terminal erosion autopsy and propose one new metric family` |

本节 supersede 上方 08:36 的 D-RAT blocker 描述：后续已完成 D-RAT rational-k4 official fused trainpath，D-RAT/D-RBF official fused blocker 现在为 0。functional metric route 仍未打开，不能 promotion。

### 本轮做了什么修改

| file | change | audit note |
| --- | --- | --- |
| `dgkan/kernels/fused_rational_k4.py` | 新增 rational k4 Triton/manual CE trainpath，用于 D-RAT official fused transition。 | 新 kernel 只替换 D-RAT official trainpath，不生成实验数值。 |
| `dgkan/models/fc_purekan_primitives.py` | 接入 rational k4 manual CE forward/backward 与 MLP `frozen_readout_features`。 | 支持 official transition probe 的 same-kernel/correctness readback。 |
| `dgkan/kernels/fused_rbf.py` | 调整 low-D launch block sizing。 | 只改 launch shape，不改 RBF 数学定义。 |
| `experiments/run_v22_05_drat_drbf_repair.py` | 增加 D-RAT/D-RBF official transition rows、`official_fused_rows`、official blocker 判定。 | 避免把 v22.03 micro benchmark 伪称 official trainpath。 |
| `experiments/run_v17_common.py` | 修复 post-stop slow-state wiring：传入 `slow_state`、commit 后更新、把 slow-state diagnostics 写回 trace；新增 M252 preheated-M181 bridge 分支。 | 此前 post-stop M181/M218 telemetry/slow-state 不闭合，修复后用 fresh smoke 重跑。 |
| `dgkan/fu/mechanisms.py` | 新增 M250/T9 DebtCalibratedSource、M251/T10 ObservableMidDebtSource、M252 T10+M181Bridge；把 M181/M218/M219 纳入 terminal slow-state update/read path。 | helper 只基于 train batch/current state；没有读取 validation/test/future horizon。 |
| `experiments/run_v21_01_source_retention.py` | 注册 T9/T10/M252 smoke/full specs，包括 stop-only、post M181/M218、preheated bridge、低 post-lr/late handoff。 | spec 命名与 artifact/run-label 对齐，便于复现。 |
| `experiments/run_v22_05_metric_first_fu.py` / `run_v22_05_s012_truth_gate.py` | 注册新 metric ids 与 M252 contract expected list。 | 最新 mechanism contract gate：`35/35;alias_undeclared=0`。 |

### Code / Contract Gate

| check | result |
| --- | --- |
| py_compile after M250/M251/M252 edits | pass |
| mechanism_contracts | pass=1; `35/35;alias_undeclared=0` |
| M252 contract row | `slow_state+function_metric_target`, source pattern=`metric_target_T10_observable_mid_debt_M181_preheated_bridge` |
| no-op command note | `--mode mechanism-contracts` was a mode typo; correct mode is `mechanism_contracts`; no scientific result was taken from the typo run |

### D-RAT / D-RBF Official Fused Repair

| carrier | profile_rows | micro_near_E1_rows | E1_official_rows | official_fused_rows | best_forward_ratio | best_step_ratio | best_memory_ratio | decision | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 9 | 8 | 1 | 2 | 1.0722116882873074 | 0.5003993472647733 | 0.7969118982742961 | `OfficialRepairOpened` |  |
| D-RBF | 9 | 2 | 1 | 2 | 1.6494557429799643 | 1.161386488585517 | 0.7969118982742961 | `OfficialRepairOpened` |  |

Official transition evidence:

| carrier | batch | manual_kernel_variant | grad_relerr_max | output_max_abs_error | official_fused_kernel_complete | functional_runner_kernel_match | no_materialize_complete | forward_ratio | backward_ratio | step_ratio | memory_ratio | E1_official |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-RAT | 512 | `rational_k4_triton_l3_matmul` | 1.6263002180494368e-06 | 6.05359673500061e-09 | 1 | 1 | 1 | 1.5397747690986563 | 0.43204657494223414 | 0.5003993472647733 | 0.9618257261410789 | 1 |
| D-RAT | 2048 | `rational_k4_triton_l3_matmul` | 1.4592519619327504e-06 | 5.3551048040390015e-09 | 1 | 1 | 1 | 1.7017878166612537 | 3.6629052012471686 | 3.3691496318388503 | 0.7969118982742961 | 0 |
| D-RBF | 512 | `rbf_k4_triton_l3_matmul` | 2.5691667815408437e-07 | 5.21540641784668e-08 | 1 | 1 | 1 | 1.6494557429799643 | 1.1336262775881143 | 1.161386488585517 | 0.9618257261410789 | 1 |
| D-RBF | 2048 | `rbf_k4_triton_l3_matmul` | 1.9326577671563427e-07 | 3.725290298461914e-08 | 1 | 1 | 1 | 1.7374609545697315 | 2.7918013970157607 | 2.6239888493263375 | 0.7969118982742961 | 0 |

Interpretation: D-RAT/D-RBF 最小 official deliverable 已达成，因为每个 carrier 都有 at least one batch 的 official fused trainpath E1 row，并且 summary blocker 为空。这仍不是 functional promotion。

### Functional Continuation Results

T9/T10/M252 只跑到 smoke/full 能支撑的范围；3-dataset smoke 没有跨数据集 h3200 chain，因此没有启动新的 4GPU full。

| attempt | best / row | rows | h800_mean | h1600_mean | h2400_mean | h3200_mean | h4800_mean | h3200_pass | h4800_pos | route / decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T9 post-stop slow-state wired 3ds | `MLP-HC4-T9-earlyburst-alt10-stop400-postM181` | 3 | 0.22444434960683188 | 0.03490797678629557 | -0.11522046724955241 | -0.16349144776662192 |  | 1/3 | 0 | `F0-MetricNoEffect` |
| T10 observable-mid-debt 3ds | `MLP-HC5-T10-earlyburst-alt10-stop400-postM218` | 3 | 0.23929325739542642 | 0.12054705619812012 | 0.0014020999272664387 | -0.04746800661087036 |  | 1/3 | 0 | `F0-MetricNoEffect` |
| T10 stop-only 3ds | `MLP-HC5-T10-earlyburst-alt10-stop800` | 3 | 0.125081737836202 | -0.0502169132232666 | -0.19274465243021646 | -0.2303001880645752 |  | 1/3 | 0 | `F0-MetricNoEffect` |
| M252 T10+preheated-M181 bridge 3ds | `MLP-HC6-T10M181Bridge-stop400` | 3 | 0.25827570756276447 | 0.12926808993021646 | -0.005457361539204915 | -0.049795826276143394 |  | 1/3 | 0 | `F0-MetricNoEffect` |
| M252 low post-lr / late handoff 3ds | `MLP-HC6-T10M181Bridge-stop1600-lr025` | 3 | 0.24121463298797607 | 0.02553872267405192 | -0.1430355707804362 | -0.20907604694366455 |  | 1/3 | 0 | `F0-MetricNoEffect` |

Dataset-level evidence for M252 bridge:

| variant | dataset | h800 | h1600 | h2400 | h3200 | R1600/800 | R3200/1600 | retained_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop400 | MNIST | 0.04595828056335449 | 0.22127556800842285 | 0.16074109077453613 | 0.20855450630187988 | 4.8147051041953075 | 0.9425103195032416 | 1 |
| stop400 | Fashion-MNIST | 0.36162590980529785 | -0.010954618453979492 | -0.19832658767700195 | -0.3054523468017578 | 0.0 |  | 0 |
| stop400 | KMNIST | 0.3672429323196411 | 0.17748332023620605 | 0.021213412284851074 | -0.052489638328552246 | 0.48328587051398453 | 0.0 | 0 |
| stop1600-lr025 | MNIST | 0.0520627498626709 | 0.053994178771972656 | -0.10354793071746826 | -0.12072896957397461 | 1.0370980963240783 | 0.0 | 0 |
| stop1600-lr025 | Fashion-MNIST | 0.16128063201904297 | -0.31153857707977295 | -0.5453387498855591 | -0.6665925979614258 | 0.0 |  | 0 |
| stop1600-lr025 | KMNIST | 0.5103005170822144 | 0.33416056632995605 | 0.21977996826171875 | 0.16009342670440674 | 0.6548309381315394 | 0.4790913196691427 | 0 |

Trace diagnostics:

| finding | evidence |
| --- | --- |
| M252 branch really used preheated slow state | trace rows show `post_stop_slow_state_active=1`, `terminal_source_slow_state_used=1`, `operator_status=t10_observable_mid_debt_m181_bridge_*`. |
| T10 target was not a no-op | Fashion rows had nonzero `target_source_projected_density`: e.g. stop800-lr025 step800 = 0.699999988079071; stop1200-lr025 step800 = 0.30000001192092896; stop1600-lr025 step1600 = 0.10000000149011612. |
| writer density alone does not explain failure | Despite nonzero target density and positive h800, Fashion-MNIST h1600/h3200 collapsed across T10 stop-only, post-anchor, M252 bridge, and low post-lr variants. |
| h3200 recovery is dataset-local, not route-level | M252 stop400 retains MNIST h3200; lowlr stop1600 retains KMNIST h3200 partially; no variant retains MNIST/Fashion/KMNIST jointly. |

### Analysis / Insight / Evidence Chain

- D-RAT/D-RBF official fused blocker is closed, so the remaining blocker is no longer `official_fused_missing`; it is functional source-chain failure plus KAN source-channel mismatch.
- T9/T10/M252 can create early source (`h800_mean` often positive, best M252 h800=0.25827570756276447), so the failure is not “metric target does nothing”.
- The chain repeatedly breaks before or at h3200: best T10/M252 rows have h1600 positive but h3200 mean negative, and only 1/3 datasets positive at h3200.
- Stop-only variants show that post-stop M181/M218 anchor is not the sole cause; M252 preheated bridge shows that cold-start slow-state is also not the sole cause.
- Lower post-stop lr and later handoff shift which dataset survives (MNIST vs KMNIST) but do not produce a dataset-invariant route; Fashion-MNIST remains the dominant failure case.
- Because 3-dataset smoke already fails the continuous h3200 source-chain criterion, launching a new 4GPU full would be compute-expensive but not promotion-informative.
- The current safe conclusion is: v22.05 has resolved the basis-efficiency officialization blocker but still lacks an audit-safe train-only functional mechanism that preserves source across dataset-specific optimizer/debt dynamics through h3200/h4800.

### Updated Decision

| item | decision |
| --- | --- |
| promotion_allowed | 0 |
| official fused D-RAT/D-RBF | opened; blocker cleared |
| functional_route | `F0-MetricNoEffect` |
| best continuation family after 08:39 | M252 T10+M181Bridge smoke, but only dataset-local h3200 positives |
| best continuation h3200 mean | `MLP-HC6-T10M181Bridge-stop400`, h3200_mean=-0.049795826276143394, h3200_pass=1/3 |
| full 4GPU continuation | not launched; smoke gate failed |
| remaining blocker | dataset-invariant h1600->h3200 source washout; Fashion-MNIST is the hard negative in all T10/M252 variants |
| next_codex_action | do not keep tuning stop schedule/post-lr; next viable idea must add a train-only cross-dataset/source-observability mechanism that directly penalizes h1600->h3200 washout proxies without validation/test/future leakage |

## 2026-06-06 10:57:29 +0800 Continuation: HC7/T11 GradientObservableMidDebt

### 是否达成目标

未达成 v22.05 promotion 目标。本轮按计划中 `F0-MetricNoEffect` 后的推荐方向，新增并测试一个 train-only source-observability metric target，但 3-dataset smoke 仍未打开 h3200/productive chain，因此没有启动新的 4GPU full，也没有任何 promotion。

| item | value |
| --- | --- |
| final evaluated route | `F0-MetricNoEffect` |
| promotion_allowed | 0 |
| best new metric row | `MLP-HC7-T11-GradObservableMidDebt-stop800` |
| productive_metric_rows | 0 |
| metric_improved_ge_0p03 | 0 |
| h4800 evidence | none; smoke stopped at 3200 because h3200 gate failed |
| full 4GPU continuation | not launched |
| still open blocker | dataset-invariant source retention through h3200; KAN source-channel mismatch remains from previous official route |

Route artifact:

```json
{
  "metric_rows": 3,
  "baseline_best_v22_id": "MLP-D1b-roworth-hidden-readout",
  "baseline_best_R4800_over_3200": 0.5268808201517082,
  "best_metric_v21_id": "MLP-HC7-T11-GradObservableMidDebt-stop800",
  "best_metric_name": "T11-GradientObservableMidDebtSource-stop800",
  "best_R4800_over_3200": "",
  "best_h4800": "",
  "best_row_h4800_positive_count": "0",
  "productive_metric_rows": 0,
  "metric_improved_ge_0p03": 0,
  "functional_route": "F0-MetricNoEffect"
}
```

### 本轮做了什么修改

| file | change | audit note |
| --- | --- | --- |
| `dgkan/fu/mechanisms.py` | 新增 `M253-MetricTargetGradientObservableMidDebtFU` / `T11-GradientObservableMidDebtSource`。 | 从 T10 observable-mid-debt source target 出发，只用当前 train batch 的 loss gradient，把 target update 投到非负 train-loss halfspace；不读取 validation/test/future。 |
| `experiments/run_v21_01_source_retention.py` | 注册 HC7/T11 stop400/stop800/stop1600 specs，并补入 `--scope mlp` allowlist。 | 第一次 smoke 暴露 allowlist 缺口，只生成 header；修复后才使用 fresh smoke 结果。 |
| `experiments/run_v21_01_source_retention.py` | 把 T11 诊断列加入 source-retention enrich keys。 | 便于 matrix 汇总保留 `target_gradient_conflict_*` 和 `source_observability_gate_accept`。 |
| `experiments/run_v22_05_metric_first_fu.py` | 注册 T11 metric names 和 summary diagnostic fields。 | summary 只汇总 artifact 里的 runner 输出，不生成实验数值。 |
| `experiments/run_v22_05_s012_truth_gate.py` | 将 M253 加入 v22.05 expected mechanism contracts。 | 最新 mechanism contract gate 变为 `36/36;alias_undeclared=0`。 |
| `experiments/run_v17_common.py` | 将 T11 halfspace diagnostics 写入 trace row whitelist。 | 修复后重跑 final smoke；此前无诊断列的 smoke 不作为最终诊断证据。 |

### Code / Contract Gate

| check | result |
| --- | --- |
| py_compile | pass; no stdout |
| CUDA availability | `cuda_available=True`, `device_count=4` |
| mechanism_contracts | pass=1; `36/36;alias_undeclared=0` |
| M253 contract | `function_metric_target`, source pattern=`metric_target_T11_gradient_observable_mid_debt_source`, leakage-free=1 |

### HC7/T11 Smoke Summary

Artifact:

- source: `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_smoke`
- analysis: `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_analysis`
- datasets/seeds/steps: MNIST,Fashion-MNIST,KMNIST x seed 0 x 3200 steps
- matched controls included: `CTRL-SGD`, `CTRL-AdamW`, `CTRL-RandomMatchedNorm`, `CTRL-NoOpMatchedOverhead`

| v21_id | rows | source_h800_mean | source_h1600_mean | source_h2400_mean | source_h3200_mean | R1600/800 | R3200/1600 | h3200_pass | productive_h3200 | failure_taxonomy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-HC7-T11-GradObservableMidDebt-stop400 | 3 | 0.18684804439544678 | 0.008451382319132486 | -0.20809821287790933 | -0.2925022045771281 | 0.04523131267697889 | 0.0 | 0/3 | 0 | `ControlEquivalent` |
| MLP-HC7-T11-GradObservableMidDebt-stop800 | 3 | 0.24417519569396973 | 0.021933436393737793 | -0.16677228609720865 | -0.23480300108591715 | 0.08982663587675574 | 0.0 | 1/3 | 0 | `ControlEquivalent` |
| MLP-HC7-T11-GradObservableMidDebt-stop1600 | 3 | 0.07729800542195638 | -0.16814486185709634 | -0.36500970522562665 | -0.4474201202392578 | 0.0 |  | 1/3 | 0 | `ControlEquivalent` |

Interpretation: T11 can still create local early/weak source, but it does not preserve a dataset-invariant h1600->h3200 chain. The best mean h3200 remains negative, and only one of three datasets passes h3200 in the best row.

### Dataset-Level Evidence

| variant | dataset | source_h800 | source_h1600 | source_h2400 | source_h3200 | R3200/1600 | retained_flag | washout_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop400 | MNIST | -0.0590057373046875 | 0.0775299072265625 | -0.029533743858337402 | -0.015619516372680664 | 0.0 | 0 | 1 |
| stop400 | Fashion-MNIST | 0.4353950023651123 | 0.009731292724609375 | -0.29605841636657715 | -0.42271459102630615 | 0.0 | 0 | 1 |
| stop400 | KMNIST | 0.18415486812591553 | -0.061907052993774414 | -0.2987024784088135 | -0.43917250633239746 |  | 0 | 1 |
| stop800 | MNIST | 0.09105479717254639 | 0.26405251026153564 | 0.15209126472473145 | 0.1478438377380371 | 0.5599031707428287 | 1 | 0 |
| stop800 | Fashion-MNIST | 0.18240833282470703 | -0.41066646575927734 | -0.7018990516662598 | -0.8181295394897461 |  | 0 | 1 |
| stop800 | KMNIST | 0.45906245708465576 | 0.21241426467895508 | 0.049490928649902344 | -0.03412330150604248 | 0.0 | 0 | 1 |
| stop1600 | MNIST | 0.09429359436035156 | 0.1836719512939453 | 0.07122802734375 | 0.08576476573944092 | 0.4669453617454334 | 0 | 0 |
| stop1600 | Fashion-MNIST | -0.05958914756774902 | -0.5317192077636719 | -0.7970863580703735 | -0.9548715353012085 |  | 0 | 0 |
| stop1600 | KMNIST | 0.1971895694732666 | -0.1563873291015625 | -0.36917078495025635 | -0.47315359115600586 |  | 0 | 1 |

Evidence chain:

- stop800 succeeds on MNIST through h3200 (`retained_flag=1`, R3200/1600=0.5599031707428287).
- The same stop800 row fails route-level because Fashion-MNIST collapses hard (`source_h3200=-0.8181295394897461`) and KMNIST ends below zero (`source_h3200=-0.03412330150604248`).
- stop1600 leaves MNIST positive but below retained ratio gate (R3200/1600=0.4669453617454334), while Fashion-MNIST and KMNIST stay negative.
- Therefore the new mechanism produces dataset-local signal but not cross-dataset source retention.

### T11 Diagnostic Evidence

Active T11 trace diagnostics:

| v21_id | active_trace_rows | projected_rows | gate_accept_rows | cos_before_min | cos_before_mean | cos_before_max | removed_fraction_max | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP-HC7-T11-GradObservableMidDebt-stop400 | 9 | 0 | 9 | 0.07323914766311646 | 0.14656170209248862 | 0.2487860769033432 | 0.0 | `gradient_observable_halfspace_already_safe` |
| MLP-HC7-T11-GradObservableMidDebt-stop800 | 12 | 0 | 12 | 0.031422607600688934 | 0.13721636154999337 | 0.3886694014072418 | 0.0 | `gradient_observable_halfspace_already_safe` |
| MLP-HC7-T11-GradObservableMidDebt-stop1600 | 15 | 0 | 15 | 0.043422944843769073 | 0.11727110172311465 | 0.29383379220962524 | 0.0 | `gradient_observable_halfspace_already_safe` |

Interpretation:

- The halfspace projection almost never had anything to remove: all active T11 rows were already non-conflicting with the instantaneous train-loss cotangent on the target support.
- This is a useful negative result. The h1600->h3200 washout is not explained by a simple current-batch train-loss gradient anti-alignment check.
- Because `removed_fraction_max=0.0` across all three stop schedules, the failure is likely in longer-horizon optimizer/source-channel dynamics, not in the immediate sign of the metric-target update.

### Artifact Index

| artifact | exists | size_bytes | lines | sha256 |
| --- | --- | --- | --- | --- |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_smoke/v21_01_source_retention_matrix.csv` | 1 | 92801 | 22 | `46b23945404af80aa64f89d4698682d9413bae82529588e83ebfced0e14f8b76` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_smoke/v21_01_source_retention_raw_traces.csv` | 1 | 186583 | 190 | `4a3a7be845e7cc40ddcddfb0fa237e3669c52ec05c4c902328c555adcd3cecce` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_analysis/v22_05_metric_first_mlp_summary.csv` | 1 | 3505 | 4 | `23a1433a451731f36a803ce69ab23436f88951e3ae46ece0ff83c221dd0fef65` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc7_t11_gradient_observable_analysis/v22_05_metric_first_route.json` | 1 | 472 | 13 | `3e367922e8776cd0dd8d7c78fe4b4ae5a2931dc403609a9bee7dc0ae28564dbd` |

### Analysis / Insight / Conclusion

- The v22.05 goal is still not achieved. It is not blocked by D-RAT/D-RBF official fused anymore; it is blocked by functional source retention and KAN source-channel mismatch.
- M253/T11 was a legitimate plan-following repair attempt because it directly targeted the `F0-MetricNoEffect` diagnosis with a train-only source-observability constraint.
- The new evidence rules out one tempting hypothesis: immediate train-loss gradient conflict is not the dominant cause of terminal washout, because all active T11 updates were already halfspace-safe and projection removed nothing.
- Fashion-MNIST remains the hard negative: it turns positive h800 into strongly negative h3200 under stop400/stop800, and is already negative by h800 under stop1600.
- MNIST retained under stop800, but promotion requires dataset-invariant chain evidence, not one-dataset success.
- It would be scientifically weak to launch a 4GPU full from this state because the 3-dataset smoke already fails the h3200 gate.

### Updated Decision

| item | decision |
| --- | --- |
| promotion_allowed | 0 |
| functional_route | `F0-MetricNoEffect` |
| best HC7/T11 row | `MLP-HC7-T11-GradObservableMidDebt-stop800` |
| best HC7/T11 h3200 mean | -0.23480300108591715 |
| best HC7/T11 h3200 pass | 1/3 |
| full 4GPU continuation | not launched |
| next_codex_action | do not keep tuning instantaneous gradient halfspace projection; next repair should target longer-horizon train-only source-channel/optimizer dynamics, especially Fashion-MNIST h800->h1600 collapse, with an auditable proxy that is present before h3200 and does not use validation/test/future horizons |

## 2026-06-06 11:45:31 +0800 Continuation: HC8/T12 Pre-H3200 Source-Channel Anti-Washout

### 是否达成目标

未达成 v22.05 promotion 目标。本轮按 T11 的负结果继续推进，新增并测试 longer-horizon train-only source-channel/optimizer dynamics 机制；三组 fresh 3-dataset smoke 均仍为 `F0-MetricNoEffect`，没有打开 dataset-invariant h3200 chain，因此没有启动新的 4GPU full，也没有任何 promotion。

| item | value |
| --- | --- |
| final evaluated route | `F0-MetricNoEffect` |
| promotion_allowed | 0 |
| best new T12 row | `MLP-HC8-T12-PreH3200SourceAntiWashout-stop2400-lr200` |
| best T12 source_h3200_mean | -0.12555956840515137 |
| best T12 h3200 pass | 1/3 |
| productive_metric_rows | 0 |
| metric_improved_ge_0p03 | 0 |
| h4800 evidence | none; all T12 runs were 3200-step smoke because h3200 gate failed |
| full 4GPU continuation | not launched |
| remaining blocker | dataset-invariant h1600->h3200 source washout; post-stop residual/window tuning did not solve Fashion-MNIST + KMNIST jointly |

Final long-window route artifact:

```json
{
  "metric_rows": 3,
  "baseline_best_v22_id": "MLP-D1b-roworth-hidden-readout",
  "baseline_best_R4800_over_3200": 0.5268808201517082,
  "best_metric_v21_id": "MLP-HC8-T12-PreH3200SourceAntiWashout-stop2400-lr200",
  "best_metric_name": "T12-PreH3200SourceChannelAntiWashout-stop2400-lr200",
  "best_R4800_over_3200": "",
  "best_h4800": "",
  "best_row_h4800_positive_count": "0",
  "productive_metric_rows": 0,
  "metric_improved_ge_0p03": 0,
  "functional_route": "F0-MetricNoEffect"
}
```

### 本轮做了什么修改

| file | change | audit note |
| --- | --- | --- |
| `dgkan/fu/mechanisms.py` | 新增 `M254-HC8PreH3200SourceChannelAntiWashoutFU` / `T12-PreH3200SourceChannelAntiWashout` 注册、fallback target、semantic contract。 | fallback 只生成 T10 同源 train-only target；真正 anti-washout 行为由 runner trace 记录。 |
| `experiments/run_v17_common.py` | 新增 M254 stateful runner branch。 | stop 前用 T10 observable-mid-debt source 维护短/长 EMA source axis；stop 后用 train channel A/B + corrupt-label gate 判定 projected optimizer + residual，不读取 validation/test/future。 |
| `experiments/run_v21_01_source_retention.py` | 注册 HC8/T12 low residual、residual-strength、long-window specs，并加入 MLP allowlist。 | 各分支用 fresh artifact 区分，避免覆盖负结果。 |
| `experiments/run_v22_05_metric_first_fu.py` | 注册 T12 metric names，并汇总 source-state gate/projection diagnostics。 | summary 只读 runner artifact，不生成实验数值。 |
| `experiments/run_v22_05_s012_truth_gate.py` | 将 M254 加入 v22.05 expected mechanism contracts。 | self-contained S0.12 gate 最新为 pass=1。 |

### Code / Contract Gate

| check | result |
| --- | --- |
| py_compile | pass; `dgkan/fu/mechanisms.py`, `experiments/run_v17_common.py`, `experiments/run_v21_01_source_retention.py`, `experiments/run_v22_05_metric_first_fu.py`, `experiments/run_v22_05_s012_truth_gate.py` |
| S0.12 self-contained truth gate | pass=1 |
| import_closure | pass=1; import returncode=0 |
| mechanism_contracts | pass=1; `37/37;alias_undeclared=0` |
| M254 contract | `slow_state+function_metric_target+optimizer_projection`, source pattern=`metric_target_T12_pre_h3200_source_channel_antiwashout`, leakage-free=1 |

### HC8/T12 Smoke Summary

Artifacts:

- low residual source: `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_smoke`
- low residual analysis: `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_analysis`
- residual strength source: `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_residual_strength_smoke`
- residual strength analysis: `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_residual_strength_analysis`
- long-window source: `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_longwindow_smoke`
- long-window analysis: `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_longwindow_analysis`
- datasets/seeds/steps: MNIST,Fashion-MNIST,KMNIST x seed 0 x 3200 steps
- matched controls included in every smoke: `CTRL-SGD`, `CTRL-AdamW`, `CTRL-RandomMatchedNorm`, `CTRL-NoOpMatchedOverhead`

| smoke | best row | rows | source_h800_mean | source_h1600_mean | source_h2400_mean | source_h3200_mean | R1600/800 | R3200/1600 | h3200_pass | productive_h3200 | route |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low residual | `MLP-HC8-T12-PreH3200SourceAntiWashout-stop1200` | 3 | 0.17877582708994547 | -0.0008862813313802084 | -0.1532947619756063 | -0.20169496536254883 | 0.0 |  | 1/3 | 0 | `F0-MetricNoEffect` |
| residual strength | `MLP-HC8-T12-PreH3200SourceAntiWashout-stop1600-lr200` | 3 | 0.25372469425201416 | 0.10267305374145508 | -0.03887017567952474 | -0.09980360666910808 | 0.40466322777188657 | 0.0 | 1/3 | 0 | `F0-MetricNoEffect` |
| long-window | `MLP-HC8-T12-PreH3200SourceAntiWashout-stop2400-lr200` | 3 | 0.25652043024698895 | 0.0555347204208374 | -0.08116328716278076 | -0.12555956840515137 | 0.21649238763308706 | 0.0 | 1/3 | 0 | `F0-MetricNoEffect` |

### Dataset-Level Evidence

| variant | dataset | source_h800 | source_h1600 | source_h2400 | source_h3200 | R1600/800 | R3200/1600 | retained_flag | washout_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low stop1200 | MNIST | 0.08840346336364746 | 0.19275927543640137 | 0.09471333026885986 | 0.11766481399536133 | 2.180449363657694 | 0.6104236163420496 | 1 | 0 |
| low stop1200 | Fashion-MNIST | 0.04937422275543213 | -0.3810945749282837 | -0.5787816047668457 | -0.6766637563705444 | 0.0 |  | 0 | 1 |
| low stop1200 | KMNIST | 0.39854979515075684 | 0.1856764554977417 | 0.024183988571166992 | -0.04608595371246338 | 0.46588019303210804 | 0.0 | 0 | 0 |
| residual stop1600-lr200 | MNIST | 0.10723495483398438 | 0.29325222969055176 | 0.2096186876296997 | 0.23232746124267578 | 2.7346701469175767 | 0.7922444834872507 | 1 | 0 |
| residual stop1600-lr200 | Fashion-MNIST | 0.4671132564544678 | 0.12862229347229004 | -0.001443028450012207 | -0.07696413993835449 | 0.2753556909272336 | 0.0 | 0 | 0 |
| residual stop1600-lr200 | KMNIST | 0.18682587146759033 | -0.11385536193847656 | -0.3247861862182617 | -0.4547741413116455 | 0.0 |  | 0 | 1 |
| long stop2400-lr200 | MNIST | 0.21485209465026855 | 0.38008344173431396 | 0.3282967805862427 | 0.37888944149017334 | 1.7690469453090756 | 0.9968585838975452 | 1 | 0 |
| long stop2400-lr200 | Fashion-MNIST | 0.23295021057128906 | -0.23433399200439453 | -0.41655027866363525 | -0.5259661674499512 | 0.0 |  | 0 | 1 |
| long stop2400-lr200 | KMNIST | 0.3217589855194092 | 0.020854711532592773 | -0.1552363634109497 | -0.22960197925567627 | 0.0648146981782884 | 0.0 | 0 | 1 |

Evidence chain:

- T12 low residual had high source-state consensus but projected optimizer removed essentially nothing (`source_state_antiwashout_removed_norm=0` in the low-residual trace summary); this rules out "instantaneous destructive projection is the missing piece".
- Increasing residual strength improved the best mean h1600 from near zero to 0.10267305374145508, but h3200 remained negative and only MNIST retained.
- Long-window source target kept MNIST strongly retained through h3200, but Fashion-MNIST and KMNIST still washed out. This argues against "stop too early" as the sole cause.
- The route failed in three fresh smokes with matched controls; launching full 4GPU from these h3200 states would not be promotion-informative.

### Artifact Index

| artifact | exists | size_bytes | lines | sha256 |
| --- | --- | --- | --- | --- |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/s012_t12_contract_check_selfcontained/v22_05_code_truth_gate.csv` | 1 | 447 | 12 | `503bef0d7642c1813d93e6d4d18ecb2b3a5d473e03f37c52efb306bcd5803fcf` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_analysis/v22_05_metric_first_mlp_summary.csv` | 1 | 4190 | 4 | `211b1175561790a92f2d006db8084868a783173a91fd483e85f47c3df268931b` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_analysis/v22_05_metric_first_route.json` | 1 | 479 | 13 | `ddfbb1d2778bc8470870f5d7e32601e1c9f033a7d5cbc602b414512cea0a04d3` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_residual_strength_analysis/v22_05_metric_first_mlp_summary.csv` | 1 | 5189 | 5 | `191e5162f7bae9b58e9480e65504b5409ddadfada1479be85118a17cd35e6b2d` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_residual_strength_analysis/v22_05_metric_first_route.json` | 1 | 491 | 13 | `572a2f3d568341a1334aae42efb2499d658b6812a171577405dcab9c5e3ac5c4` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_longwindow_analysis/v22_05_metric_first_mlp_summary.csv` | 1 | 4221 | 4 | `f0926737cdb9fdf8cb10f3f2661264f313e645d8fa9db5bbed77530f233e7aab` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc8_t12_pre_h3200_source_antiwashout_longwindow_analysis/v22_05_metric_first_route.json` | 1 | 491 | 13 | `19ff0f26f8e3de620da0207521528351dd6c0243a122b93d685d060fb1d7c02a` |

### Analysis / Insight / Conclusion

- The v22.05 goal remains unmet. The official fused basis-efficiency blocker is closed, but the functional mechanism still cannot produce cross-dataset source retention through h3200.
- T12 was a direct follow-up to T11: it moved from instantaneous halfspace projection to longer-horizon source-channel EMA + train-only projected optimizer/residual gate.
- The T12 evidence narrows the failure: neither stronger post-stop residual nor longer source-target window is sufficient. MNIST can be retained, but Fashion-MNIST and KMNIST alternate as the hard negatives.
- Continuing to tune `source_stop_steps` or `post_stop_fu_lr` alone is now likely retrospective overfitting. A next principled attempt needs a different train-only proxy for dataset-invariant source retention before h3200, not another scalar schedule sweep.

### Updated Decision

| item | decision |
| --- | --- |
| promotion_allowed | 0 |
| functional_route | `F0-MetricNoEffect` |
| best HC8/T12 row | `MLP-HC8-T12-PreH3200SourceAntiWashout-stop2400-lr200` |
| best HC8/T12 h3200 mean | -0.12555956840515137 |
| best HC8/T12 h3200 pass | 1/3 |
| full 4GPU continuation | not launched |
| next_codex_action | stop scalar T12 residual/window tuning; next repair must introduce a new train-only dataset-invariance/source-observability proxy, or explicitly mark the current v22.05 route blocked until such a proxy is specified |

## 2026-06-06 12:56:32 +0800 Continuation: HC9/T13 -> HC12/T16 Metric-First Source Observability

### 是否达成目标

未达成 v22.05 promotion 目标。本轮继续按上节推荐方向推进：不再只调 T12 标量，而是依次加入新的 train-only source-observability / dataset-invariance proxy，并保留所有负结果。T13/T14/T15/T16 均为 fresh 3-dataset smoke，route 仍为 `F0-MetricNoEffect`，没有 dataset-invariant h3200 chain，因此没有启动新的 4800/6400 full，也没有 promotion。

| item | value |
| --- | --- |
| final evaluated route | `F0-MetricNoEffect` |
| promotion_allowed | 0 |
| productive_metric_rows | 0 |
| metric_improved_ge_0p03 | 0 |
| h4800 evidence | none; all new runs were 3200-step smoke because h3200 gate failed |
| full 4GPU continuation | not launched |
| best new row by h3200 | `MLP-HC11-T15-ContinuousSourceCarry-stop3200-lr300` |
| best new source_h3200_mean | -0.12753812472025552 |
| best new h3200 pass | 1/3 |
| remaining blocker | Fashion-MNIST hard negative; cross-dataset h1600->h3200 source chain still breaks |

Final T16 route artifact:

```json
{
  "metric_rows": 2,
  "baseline_best_v22_id": "MLP-D1b-roworth-hidden-readout",
  "baseline_best_R4800_over_3200": 0.5268808201517082,
  "best_metric_v21_id": "MLP-HC12-T16-NoiseOrthogonalCarry-stop2400-lr500",
  "best_metric_name": "T16-NoiseOrthogonalSourceCarry-stop2400-lr500",
  "best_R4800_over_3200": "",
  "best_h4800": "",
  "best_row_h4800_positive_count": "0",
  "productive_metric_rows": 0,
  "metric_improved_ge_0p03": 0,
  "functional_route": "F0-MetricNoEffect"
}
```

### 本轮做了什么修改

| file | change | audit note |
| --- | --- | --- |
| `dgkan/fu/mechanisms.py` | 新增 `M255/M256/M257/M258`，分别对应 T13 risk-weighted split consensus、T14 view-consistent source carry、T15 continuous source carry anti-washout、T16 noise-orthogonal source carry。 | 机制只构造 train-only source target / slow state / projection，不生成实验数值。 |
| `experiments/run_v17_common.py` | 将 M255-M258 接入 T12 shared stateful runner；T15/T16 post-stop 使用 continuous carry。 | T15/T16 仍受 train channel A/B + corrupt-label gate 约束；不读取 validation/test/future horizon。 |
| `experiments/run_v21_01_source_retention.py` | 注册 HC9-HC12 smoke specs，并加入 MLP allowlist。 | T15 strength/window 作为 follow-up smoke 独立落盘，不覆盖初跑负结果。 |
| `experiments/run_v22_05_metric_first_fu.py` | 注册 T13-T16 metric names，并汇总 PopRisk / source-state diagnostics。 | summary 只读 runner artifact。 |
| `experiments/run_v22_05_s012_truth_gate.py` | 将 M255-M258 加入 v22.05 expected mechanism contracts。 | 最新 self-contained S0.12 gate 为 pass=1。 |

### Code / Contract Gate

| check | result |
| --- | --- |
| py_compile | pass; touched Python files compile |
| S0.12 T14 | pass=1; `mechanism_contracts=39/39;alias_undeclared=0` |
| S0.12 T15 | pass=1; `mechanism_contracts=40/40;alias_undeclared=0` |
| S0.12 T16 | pass=1; `mechanism_contracts=41/41;alias_undeclared=0` |
| linec / source-chain / terminal-retention unit gates | pass in each S0.12 truth gate |

### Smoke Summary

All rows below are 3-dataset smokes on MNIST,Fashion-MNIST,KMNIST x seed 0 x 3200 steps with matched controls.

| attempt | best row in attempt | source_h800_mean | source_h1600_mean | source_h2400_mean | source_h3200_mean | h3200_pass | R1600/800 | R3200/1600 | productive_h3200 | route |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HC9/T13 risk-weighted split-consensus | `MLP-HC9-T13-RiskWeightedSplitConsensus-stop2400-lr100` | 0.2805052200953166 | 0.05216574668884277 | -0.14002458254496256 | -0.2624085346857707 | 1/3 | 0.185970680585255 | 0.0 | 0 | `F0-MetricNoEffect` |
| HC10/T14 view-consistent source-carry | `MLP-HC10-T14-ViewConsistentSourceCarry-stop1600-lr100` | 0.23324986298878989 | -0.13014745712280273 | -0.4178553819656372 | -0.5328860282897949 | 0/3 | 0.0 |  | 0 | `F0-MetricNoEffect` |
| HC11/T15 continuous carry initial | `MLP-HC11-T15-ContinuousSourceCarry-stop2400-lr300` | 0.16040043036142984 | -0.016282955805460613 | -0.16506568590799967 | -0.22702674070994058 | 0/3 | 0.0 |  | 0 | `F0-MetricNoEffect` |
| HC11/T15 strength/window | `MLP-HC11-T15-ContinuousSourceCarry-stop3200-lr300` | 0.15483474731445312 | 0.05064964294433594 | -0.07236131032307942 | -0.12753812472025552 | 1/3 | 0.3271206484515509 | 0.0 | 0 | `F0-MetricNoEffect` |
| HC12/T16 noise-orthogonal carry | `MLP-HC12-T16-NoiseOrthogonalCarry-stop2400-lr500` | -0.27404971917470294 | -0.6525308688481649 | -0.8845804135004679 | -0.9975163141886393 | 0/3 |  |  | 0 | `F0-MetricNoEffect` |

### Dataset-Level Evidence

T13 best row:

| dataset | source_h800 | source_h1600 | source_h2400 | source_h3200 | retained_flag |
| --- | --- | --- | --- | --- | --- |
| MNIST | 0.0248951912 | 0.0877770185 | -0.1077365875 | -0.1881933212 | 0 |
| Fashion-MNIST | 0.2640881538 | -0.3144279718 | -0.5358501673 | -0.7002754211 | 0 |
| KMNIST | 0.5525323153 | 0.3831481934 | 0.2235130072 | 0.1012431383 | 0 |

T15 strength/window rows:

| variant | dataset | source_h800 | source_h1600 | source_h2400 | source_h3200 | R3200/1600 | retained_flag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stop2400-lr500 | MNIST | 0.07433199882507324 | 0.20830702781677246 | 0.11039233207702637 | 0.012359976768493652 | 0.059335380558381966 | 0 |
| stop2400-lr500 | Fashion-MNIST | 0.1615285873413086 | -0.19978606700897217 | -0.33219146728515625 | -0.5263265371322632 |  | 0 |
| stop2400-lr500 | KMNIST | 0.4012340307235718 | 0.15942144393920898 | 0.06993424892425537 | 0.0997546911239624 | 0.6257294417807502 | 1 |
| stop3200-lr300 | MNIST | 0.04903721809387207 | 0.2101154327392578 | 0.11802291870117188 | 0.13902604579925537 | 0.6616650856473707 | 1 |
| stop3200-lr300 | Fashion-MNIST | 0.01328277587890625 | -0.25135135650634766 | -0.31411242485046387 | -0.38701963424682617 |  | 0 |
| stop3200-lr300 | KMNIST | 0.40218424797058105 | 0.19318485260009766 | -0.02099442481994629 | -0.1346207857131958 | 0.0 | 0 |

T16 noise-orthogonal rows:

| variant | dataset | target_consensus_density | source_state_gate_accept | source_h800 | source_h3200 | note |
| --- | --- | --- | --- | --- | --- | --- |
| stop2400-lr500 | MNIST | 0.800000011920929 | 0 | -0.3657628297805786 | -0.6244202852249146 | gate rejected; corrupt gain positive |
| stop2400-lr500 | Fashion-MNIST | 0.30000001192092896 | 0 | -0.17363381385803223 | -1.1970549821853638 | worse than T15 |
| stop2400-lr500 | KMNIST | 0.30000001192092896 | 0 | -0.28275251388549805 | -1.1710736751556396 | worse than T15 |
| stop3200-lr300 | MNIST | 0.5 | 1 | -0.44394874572753906 | -0.7433885335922241 | active but negative from h800 |
| stop3200-lr300 | Fashion-MNIST | 0.9000000357627869 | 1 | -0.28129398822784424 | -1.5481666326522827 | active but destructive |
| stop3200-lr300 | KMNIST | 0.699999988079071 | 1 | -0.057518959045410156 | -0.7213952541351318 | active but destructive |

Evidence chain:

- T13 PopRisk/SNR weighting produced high source-state consensus but failed after h1600; it did not fix Fashion-MNIST.
- T14 view-consistency produced stable view alignment near 1 in the raw traces, yet source was worse than T13; stable view target is not sufficient.
- T15 continuous carry is the only new attempt that improved the h3200 mean relative to T13/T14 and produced 2/3 positive rows under stop2400-lr500, but Fashion-MNIST remained strongly negative, so the route-level gate still failed.
- Extending active source writing to h3200 helped MNIST but not Fashion/KMNIST, which means the blocker is not merely "writer stopped too early".
- T16 noise-orthogonalization was actively harmful in this setup: source was already negative at h800 and all h3200 rows were negative. This retires that proxy for this route.

### Artifact Index

| artifact | exists | size_bytes | lines | sha256 |
| --- | --- | --- | --- | --- |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/s012_t13_contract_check_selfcontained/v22_05_code_truth_gate.csv` | 1 | 447 | 12 | `79d8766de11d49b877546052bb605f361a5f690a6b10ecd4447337c4b8a2b666` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc9_t13_risk_weighted_split_consensus_analysis/v22_05_metric_first_mlp_summary.csv` | 1 | 5606 | 5 | `375f4602445569225a1a2378f4c5ec80bf91a3219de9fb4e067a12dc99346d51` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc9_t13_risk_weighted_split_consensus_analysis/v22_05_metric_first_route.json` | 1 | 492 | 13 | `6a60fc5396adc4b93be48958754a486deadc1ee6c516bbcc7cf5a09f8036aa09` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/s012_t14_contract_check_selfcontained/v22_05_code_truth_gate.csv` | 1 | 447 | 12 | `da4171803c07d437f39843be9083fd05ac9a3a9b4d1b8753de086bfc2811abc5` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc10_t14_view_consistent_source_carry_analysis/v22_05_metric_first_mlp_summary.csv` | 1 | 4284 | 4 | `ca694319ad047b03c46f0b595ba5960247bbf7abaca8fcc8216a4c3afd367397` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc10_t14_view_consistent_source_carry_analysis/v22_05_metric_first_route.json` | 1 | 485 | 13 | `6475449d8d60a4d90b7ea84900f4e2de4718ab4a7fdcac962f0f695836e94e8d` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/s012_t15_contract_check_selfcontained/v22_05_code_truth_gate.csv` | 1 | 447 | 12 | `8563cbb9b29a0614f32faef2a8f30315c118d83e557e188ebbd307c05986a480` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_continuous_source_carry_analysis/v22_05_metric_first_mlp_summary.csv` | 1 | 4345 | 4 | `03466782cbdce9c33ca658a2745b1a84c88cc00a04082a9a68f5151dfeafb2af` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_continuous_source_carry_analysis/v22_05_metric_first_route.json` | 1 | 488 | 13 | `9e1941518a97ec357a0321df50af3b0e59b18196b83785b50dc74e860f9506eb` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_strength_window_analysis/v22_05_metric_first_mlp_summary.csv` | 1 | 3430 | 3 | `8c59576e8b919a6b9a4f22dc32c7f2d4235ac428eba005dd7a6901c9cdda9b1b` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc11_t15_strength_window_analysis/v22_05_metric_first_route.json` | 1 | 488 | 13 | `3ced1827c040492255288767278b5b0bf603ead8425c888865d64e95034ff168` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/s012_t16_contract_check_selfcontained/v22_05_code_truth_gate.csv` | 1 | 447 | 12 | `30059416d62a45300fe69c956b8e65145cb3d17e88b31127d52c8898ba9e58e8` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc12_t16_noise_orthogonal_source_carry_analysis/v22_05_metric_first_mlp_summary.csv` | 1 | 3422 | 3 | `9eb0f14733e11be95b6c9d775fbb1d9bd6197ae040ec156d9c8bdd895972e3ec` |
| `results/v22_05_metric_first_functional_update_basis_efficiency_4gpu/official_v22_05/continuation_hc12_t16_noise_orthogonal_source_carry_analysis/v22_05_metric_first_route.json` | 1 | 481 | 13 | `f95ad45f32104647c09dc008fe97139a27ad4cab106026a4acdb9b6213fb70e0` |

### Analysis / Insight / Conclusion

- The v22.05 goal remains unmet. None of T13-T16 produced cross-dataset productive h3200, and no h4800 evidence exists for these new rows.
- T15 is the best new direction because it improves the mean h3200 and shows MNIST/KMNIST positive under one setting, but Fashion-MNIST remains a reliable hard negative.
- T13/T14/T16 are retired for this route unless a new independent reason appears: T13 and T14 fail retention, while T16 destroys early source.
- The next repair should not be another generic source carry scalar sweep. It needs a Fashion-MNIST-aware but train-only invariance signal, preferably one that can be audited before h1600 and proves it does not turn into dataset-local overfitting.

### Updated Decision

| item | decision |
| --- | --- |
| promotion_allowed | 0 |
| functional_route | `F0-MetricNoEffect` |
| best new candidate | `MLP-HC11-T15-ContinuousSourceCarry-stop3200-lr300` |
| best new h3200 mean | -0.12753812472025552 |
| best new h3200 pass | 1/3 |
| full 4GPU continuation | not launched |
| next_codex_action | stop T13/T14/T16; keep T15 only as evidence that active carry helps partly, then design a genuinely train-only hard-dataset invariance/source-observability proxy before any full run |
