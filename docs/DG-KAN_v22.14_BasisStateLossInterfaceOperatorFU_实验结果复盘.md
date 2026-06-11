# DG-KAN v22.14 Basis-State Loss-Interface Operator FU 实验结果复盘

生成时间：2026-06-09 02:10:00 +0800

## Route

- route: `R3-EfficiencyVariantRobustnessBlocked`
- exploration_promotion_allowed: `1`
- official_operator_promotion_allowed_strict: `0`
- official_kan_carrier_promotion_allowed: `0`
- scientific_claim_allowed: `0`
- execution_contract_violation: `0`
- artifact_nonempty_gate_pass: `1`
- blocking_metric: `planned_grid_or_ratio_gate_not_closed;SourceRetentionDynamicsNotSolved;R7-KANBasisTangentCoverageNoGo;KAN_basis_state_gate_not_passed`
- minimum_effective_progress: `A-CodeTruth;B-Efficiency;C-Operator;D-KAN;E-TaskGate`
- next_codex_action: `repair strict source retention/efficiency/KAN basis-state blocker shown by v22.14 route; do not promote auxiliary loss or readout-only rows`
- results bundle: `results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_results_bundle.zip`
- code review packet: `results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_code_review_packet.zip`

## 0. 本轮修改

- 在 `dgkan/fu/operator_core.py` 新增 LIO8/LIO9/LIO10/LIO11 四个 role-blind metric-as-operator 变体；这些算子只读 `cotangent/logits/source_state`，不读 adapter 名称、loss 公式、dataset、test/validation/future/query。
- 新增 `experiments/run_v22_14_common.py` 与 `experiments/run_v22_14_basis_state_operator_fu.py`，负责 v22.14 分阶段运行、queue artifact、strict finalizer、执行日志和复盘日志。
- F1 明确拆分 `operator_only / periodic_source_state / auxiliary_loss_anchor / optimizer_prox_anchor / source_state_projector`；`uses_loss_modification_for_retention=1` 的 row 只允许 exploration。
- K18/K19/K20 从 readout replay 改为 basis tangent coverage、direct basis-state operator 和 readout-to-basis transfer 证据链。

## 1. Code Truth

| check | pass | metric | value | blocker |
| --- | --- | --- | --- | --- |
| required_source_files_present | 1 | exists | 20/20 |  |
| compileall_ok | 1 | py_compile | 0 |  |
| core_import_pass | 1 | import | 0 |  |
| official_operator_role_blind_static_pass | 1 | semantic | 1 |  |
| adapter_renaming_pass | 1 | operator | 10/10 |  |
| operator_law_pass | 1 | operator | 10/10 |  |
| corrected_layout_tests | 1 | layout | 2/2 |  |

Analysis: S0 只说明本轮代码、导入、layout 与 role-blind firewall 可以审计；它不自动构成 FU/KAN success。

## 2. Operator Atoms

| operator_id | operator_family | gain_positive_fraction | control_projection_after | NDS_reduction | operator_lipschitz_ratio | S2_operator_atom_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LIO1_LowNDSGreen | LowNDSGreen | 1.0 | 0.08519414812326431 | 0.0691955039037072 | 4.987429780755332 | 1 |  |
| LIO2_SplitCoherentControlNull | SplitCoherentControlNull | 1.0 | 1.8098113230280433e-08 | 0.08452886904114111 | 4.993961390260886 | 1 |  |
| LIO3_KANLowBankSpectral | KANLowBankSpectral | 1.0 | 2.689061595617659e-08 | 0.11294957755642081 | 5.004201507647522 | 0 | operator_lipschitz |
| LIO5_DualMemoryRetained | DualMemoryRetained | 1.0 | 2.7083292053475816e-08 | 0.09948680095311828 | 4.999843701844928 | 1 |  |
| LIO6_SourceLossBoundary | SourceLossBoundary | 1.0 | 1.0098347047460265e-08 | 0.07180865659601832 | 4.9665502633001966 | 1 |  |
| LIO7_LowNDSControlNull | LowNDSControlNull | 1.0 | 4.642412676503227e-08 | 0.07183369854726641 | 4.9853765035585065 | 1 |  |
| LIO8_FisherSobolevResolvent | FisherSobolevResolvent | 0.8333333333333334 | 5.458047169781821e-08 | 0.08116851765941203 | 4.858598380262826 | 1 |  |
| LIO9_AdapterWhitenedCotangent | AdapterWhitenedCotangent | 0.8333333333333334 | 2.0610094964013115e-08 | 0.06602527398026928 | 5.031611089396596 | 0 | operator_lipschitz |
| LIO10_SignalReservoirSplit | SignalReservoirSplit | 1.0 | 8.184472655159425e-09 | 0.06704209882208331 | 4.985802054837861 | 1 |  |
| LIO11_SourceLossBoundaryNoAdapter | SourceLossBoundaryNoAdapter | 1.0 | 2.076074601120581e-08 | 0.08587310351716429 | 4.957538589149169 | 1 |  |

Analysis: v22.14 不只复用 LIO2/LIO7，也把 LIO8-LIO11 放进同一个 role-blind law/gain/control gate。没有通过 S2 的新算子不参与后续 claim。

## 3. F1 Anchor Split

| seed | loss_adapter_name | anchor_mechanism_type | uses_loss_modification_for_retention | source_func_h3200 | source_func_h4800 | source_loss_h3200 | source_loss_h4800 | C3_source_formation_pass | C4_terminal_retention_pass | C5_h6400_retention_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2213 | Delta-LossCEAdapter | operator_only | 0 | -0.26665961383806036 | -0.47536099559132394 | 0.0008524360018782318 | 0.0003065419150516391 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-LossCEAdapter | periodic_source_state | 0 | -0.2884740260845533 | -0.5093542787859903 | 0.0007940804352983832 | 0.00027142808539792895 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-LossCEAdapter | auxiliary_loss_anchor | 1 | 1.2543525882670454 | 1.2388449139348943 | 0.2080983929336071 | 0.18359456956386566 | 1 | 1 | 1 |  |
| 2213 | Delta-LossCEAdapter | optimizer_prox_anchor | 0 | -0.2715540566311725 | -0.4807040124448664 | -0.0008184493790395209 | -0.00032946659485233454 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-LossCEAdapter | source_state_projector | 0 | -0.26671279199349973 | -0.47541828207278025 | 0.0008473453344777226 | 0.0003044198383577168 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-MSEAdapter | operator_only | 0 | 0.2712046176543539 | 0.08657493080465012 | 0.7733504772186279 | -1.8268157243728638 | 1 | 0 | 0 |  |
| 2213 | Delta-MSEAdapter | periodic_source_state | 0 | 0.26823201453485224 | 0.08285101381612109 | 0.7201817035675049 | -1.8837696313858032 | 1 | 0 | 0 |  |
| 2213 | Delta-MSEAdapter | auxiliary_loss_anchor | 1 | 0.3807950021552453 | 0.20460488565185064 | 1.0773673057556152 | -1.4864380359649658 | 1 | 0 | 0 |  |
| 2213 | Delta-MSEAdapter | optimizer_prox_anchor | 0 | 0.17351648942315923 | 0.11175809048107882 | -1.1684119701385498 | -2.6342948973178864 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-MSEAdapter | source_state_projector | 0 | 0.270311395275064 | 0.0858025385445923 | 0.7616825103759766 | -1.8356362581253052 | 1 | 0 | 0 |  |
| 2213 | Delta-RankingAdapter | operator_only | 0 | -1.3558432256979942 | -1.4518086594746547 | -0.0012437743716873229 | -0.00044407379755284637 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-RankingAdapter | periodic_source_state | 0 | -1.3909946514735911 | -1.5037285766597135 | -0.001153753837570548 | -0.00039404493873007596 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-RankingAdapter | auxiliary_loss_anchor | 1 | 0.9418059873446999 | 0.9367982862894791 | -0.893524132668972 | -0.8773990571498871 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-RankingAdapter | optimizer_prox_anchor | 0 | -21.616710384812468 | -32.418747884840876 | 0.0005140892013877352 | 0.00020315730936601865 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-RankingAdapter | source_state_projector | 0 | -1.362920496050131 | -1.4587572186231692 | -0.0012250684667378664 | -0.00043716892832890153 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-PreferenceAdapter-smoke | operator_only | 0 | -12.262308072688464 | -17.616356164618367 | -0.02848856896162033 | -0.011237401515245438 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-PreferenceAdapter-smoke | periodic_source_state | 0 | -12.510755894758132 | -18.09229026002333 | -0.027882441878318787 | -0.010606169700622559 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-PreferenceAdapter-smoke | auxiliary_loss_anchor | 1 | 1.6739115817795678 | 1.673198963746951 | -0.15379440784454346 | -0.15367984771728516 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-PreferenceAdapter-smoke | optimizer_prox_anchor | 0 | -52.30351528461499 | -77.40178808395403 | 0.05573307257145643 | 0.01817975508811287 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-PreferenceAdapter-smoke | source_state_projector | 0 | -12.398795492534497 | -17.840250053642666 | -0.025793030858039856 | -0.009408120065927505 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-StableRandom-control | operator_only | 0 | -62.617304268337286 | -86.70516706913395 | -1230625.673828125 | -1885909.3203125 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-StableRandom-control | periodic_source_state | 0 | -64.34156463408173 | -90.17657038171733 | -1229805.294921875 | -1884230.5859375 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-StableRandom-control | auxiliary_loss_anchor | 1 | -43.984412026825176 | -55.88583689365835 | -207662.208984375 | -219650.52734375 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-StableRandom-control | optimizer_prox_anchor | 0 | -62.617304268337286 | -86.70516706913395 | -1230625.673828125 | -1885909.3203125 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-StableRandom-control | source_state_projector | 0 | -62.617304268337286 | -86.70516706913395 | -1230625.673828125 | -1885909.3203125 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-RandomMatched-control | operator_only | 0 | -78.33679394694316 | -108.6486606695875 | -532624.65625 | -817553.2890625 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-RandomMatched-control | periodic_source_state | 0 | -80.48993695317745 | -112.98461712479337 | -531985.513671875 | -816236.1328125 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-RandomMatched-control | auxiliary_loss_anchor | 1 | -39.90791020538161 | -47.42264474876433 | -154648.28515625 | -174793.00390625 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-RandomMatched-control | optimizer_prox_anchor | 0 | -78.33679394694316 | -108.6486606695875 | -532624.65625 | -817553.2890625 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-RandomMatched-control | source_state_projector | 0 | -78.33679394694316 | -108.6486606695875 | -532624.65625 | -817553.2890625 | 0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |

Analysis: 严格 official 只允许 `uses_loss_modification_for_retention=0`。如果 auxiliary anchor 过而 optimizer-prox/projector 不过，本轮结论必须写成 auxiliary upper bound，而不是 no-loss-modification FU 成功。

## 4. Efficiency

| carrier | variant | profile_rows | ratio_pass_rows | planned_cotangent_suite_complete | planned_hidden_grid_complete | planned_batch_1024_complete | variant_robust_pass | exploration_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | CHE22.13-R1-k3-corrected-layout-native-vjp | 12 | 12 | 0 | 0 | 0 | 0 | 1 |  |
| D-CHE | CHE22.13-R2-k5-gradbuf-no-materialize-native-vjp | 12 | 12 | 0 | 0 | 0 | 0 | 1 |  |
| D-CHE | CHE22.13-R3-lowdegree-readout-operator-step | 12 | 12 | 0 | 0 | 0 | 0 | 1 |  |
| D-CHE | CHE22.13-R4-corrected-layout-source-state-fused-commit | 12 | 11 | 0 | 0 | 0 | 0 | 1 | ratio_outlier_or_missing_fused_row |
| D-FOU | FOU22.13-R1-lowfreq-bandreadout-native-vjp | 12 | 12 | 0 | 0 | 0 | 0 | 1 |  |
| D-FOU | FOU22.13-R2-tablelookup-bandreadout-native-operator-step | 12 | 12 | 0 | 0 | 0 | 0 | 1 |  |
| D-FOU | FOU22.13-R3-lowfreq-source-state-fused-commit | 12 | 12 | 0 | 0 | 0 | 0 | 1 |  |

Analysis: D-FOU/D-CHE native fused rows 可作为 efficiency progress；variant robust official 还要求完整 batch/hidden/cotangent grid 和 ratio gate，未满足时不能用 carrier OR 宣称 official robust。

## 5. KAN Basis Carrier

| carrier | readout_projection_residual | basis_projection_residual | basis_plus_readout_projection_residual | readout_projection_cosine | basis_projection_cosine | K18_readout_dominated_target |
| --- | --- | --- | --- | --- | --- | --- |
| D-FOU | 0.25773178228708515 | 0.9985869818739171 | 1.8191896574703452 | 0.9726719856262207 | 0.08300703763961792 | 1 |
| D-CHE | 0.3620723793815807 | 0.9982884346261496 | 12.553547914167666 | 0.9388182163238525 | 0.0757177323102951 | 1 |

| carrier | basis_operator_id | basis_channel_energy | readout_channel_energy | basis_operator_solve_ms | basis_operator_residual | KAN_source_func_h3200 | KAN_source_func_h4800 | KAN_source_loss_h3200 | KAN_source_loss_h4800 | K19_basis_state_operator_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-FOU | K19-A-basis_w1_diagonal_green | 1.0 | 0.0 | 3.546224907040596 | 3.9393808282026286e-07 | -0.011055730283260345 | 0.0030081812292337418 | -0.049164533615112305 | 0.00018419045954942703 | 0 | KAN_basis_state_source_or_retention_gate_failed |
| D-CHE | K19-A-basis_w1_diagonal_green | 1.0 | 0.0 | 3.334767185151577 | 5.696916759402988e-07 | 0.005498044192790985 | 0.01320935133844614 | -0.01204034686088562 | 0.00042693130671977997 | 0 | KAN_basis_state_source_or_retention_gate_failed |

Analysis: K18 解释 v22.13 mismatch 是否是 readout-dominated target；K19 则直接用 basis tangent 解 `delta -> Delta_b -> Delta_f_B`，不再把 readout target 硬塞给 basis。

## 6. Task Gate

| dataset | seed | variant | status | blocker |
| --- | --- | --- | --- | --- |
| MNIST | 0 | KAN+FU_basis_state_operator | not_run | KAN_basis_state_gate_not_passed; task metrics are readback only and cannot repair direction |
| MNIST | 1 | KAN+FU_basis_state_operator | not_run | KAN_basis_state_gate_not_passed; task metrics are readback only and cannot repair direction |
| MNIST | 2 | KAN+FU_basis_state_operator | not_run | KAN_basis_state_gate_not_passed; task metrics are readback only and cannot repair direction |
| FashionMNIST | 0 | KAN+FU_basis_state_operator | not_run | KAN_basis_state_gate_not_passed; task metrics are readback only and cannot repair direction |
| FashionMNIST | 1 | KAN+FU_basis_state_operator | not_run | KAN_basis_state_gate_not_passed; task metrics are readback only and cannot repair direction |
| FashionMNIST | 2 | KAN+FU_basis_state_operator | not_run | KAN_basis_state_gate_not_passed; task metrics are readback only and cannot repair direction |
| KMNIST | 0 | KAN+FU_basis_state_operator | not_run | KAN_basis_state_gate_not_passed; task metrics are readback only and cannot repair direction |
| KMNIST | 1 | KAN+FU_basis_state_operator | not_run | KAN_basis_state_gate_not_passed; task metrics are readback only and cannot repair direction |
| KMNIST | 2 | KAN+FU_basis_state_operator | not_run | KAN_basis_state_gate_not_passed; task metrics are readback only and cannot repair direction |

Analysis: task metrics 只能 readback/gate。若 KAN basis-state gate 未过，本轮不会用 task 结果反向修 direction，也不伪造 task medium/full rows。

## 7. Execution

| timestamp | task_id | gpu | command | status | note |
| --- | --- | --- | --- | --- | --- |
| 2026-06-09 01:19:51 +0800 | s0_construct |  | /home/chengshun.wang/miniconda3/envs/kan/bin/python /home/chengshun.wang/DG-LCA/experiments/run_v22_14_basis_state_operator_fu.py --stage s0_construct --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --norm-scale 5.12 | completed | exit=0 |
| 2026-06-09 01:19:51 +0800 | efficiency | cuda:0 | /home/chengshun.wang/miniconda3/envs/kan/bin/python /home/chengshun.wang/DG-LCA/experiments/run_v22_14_basis_state_operator_fu.py --stage efficiency --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --norm-scale 5.12 --device cuda:0 --batch-sizes 128,256,512 --hidden 64 --repeats 2 --warmup 1 | running | launched by v22.14 dynamic queue runner |
| 2026-06-09 01:19:51 +0800 | f1_anchor_split | cuda:1 | /home/chengshun.wang/miniconda3/envs/kan/bin/python /home/chengshun.wang/DG-LCA/experiments/run_v22_14_basis_state_operator_fu.py --stage f1 --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --norm-scale 5.12 --device cuda:1 --f1-seeds 2213 --f1-adapters Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter,Delta-PreferenceAdapter-smoke,Delta-StableRandom-control,Delta-RandomMatched-control --f1-modes operator_only,periodic_source_state,auxiliary_loss_anchor,optimizer_prox_anchor,source_state_projector | running | launched by v22.14 dynamic queue runner |
| 2026-06-09 01:19:51 +0800 | kan_basis | cuda:2 | /home/chengshun.wang/miniconda3/envs/kan/bin/python /home/chengshun.wang/DG-LCA/experiments/run_v22_14_basis_state_operator_fu.py --stage kan --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --norm-scale 5.12 --device cuda:2 | running | launched by v22.14 dynamic queue runner |
| 2026-06-09 01:19:57 +0800 | efficiency | cuda:0 | /home/chengshun.wang/miniconda3/envs/kan/bin/python /home/chengshun.wang/DG-LCA/experiments/run_v22_14_basis_state_operator_fu.py --stage efficiency --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --norm-scale 5.12 --device cuda:0 --batch-sizes 128,256,512 --hidden 64 --repeats 2 --warmup 1 | completed | exit=0 |
| 2026-06-09 02:05:38 +0800 | f1_anchor_split | cuda:1 | /home/chengshun.wang/miniconda3/envs/kan/bin/python /home/chengshun.wang/DG-LCA/experiments/run_v22_14_basis_state_operator_fu.py --stage f1 --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --norm-scale 5.12 --device cuda:1 --f1-seeds 2213 --f1-adapters Delta-LossCEAdapter,Delta-MSEAdapter,Delta-RankingAdapter,Delta-PreferenceAdapter-smoke,Delta-StableRandom-control,Delta-RandomMatched-control --f1-modes operator_only,periodic_source_state,auxiliary_loss_anchor,optimizer_prox_anchor,source_state_projector | completed | exit=0 |
| 2026-06-09 02:05:38 +0800 | kan_basis | cuda:2 | /home/chengshun.wang/miniconda3/envs/kan/bin/python /home/chengshun.wang/DG-LCA/experiments/run_v22_14_basis_state_operator_fu.py --stage kan --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --norm-scale 5.12 --device cuda:2 | completed | exit=0 |
| 2026-06-09 02:05:40 +0800 | task_gate | cuda:3 | /home/chengshun.wang/miniconda3/envs/kan/bin/python /home/chengshun.wang/DG-LCA/experiments/run_v22_14_basis_state_operator_fu.py --stage task_gate --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --norm-scale 5.12 --device cuda:3 | completed | exit=0 |
| 2026-06-09 02:07:55 +0800 | finalize_rerun | n/a | /home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v22_14_basis_state_operator_fu.py --stage finalize --out-dir results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14 --seed 2213 --norm-scale 5.12 | completed | manual rerun after fixing artifact_index relative path bug; no experiment rows rerun |

## 8. Artifacts

| artifact | exists | size_bytes | sha256 |
| --- | --- | --- | --- |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/figures/v22_14_anchor_split_source_func.svg | 1 | 325 | 9925ef1b4903e13de838e098ea05a5a10f7cea66d3e86e0143e2830ae79bb44c |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/figures/v22_14_efficiency_variant_dashboard.svg | 1 | 321 | d4616367f1babc7a349da9a93a1b0e575cd71ec20c0205d4b217224b7f9c4a5f |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/figures/v22_14_kan_basis_coverage.svg | 1 | 319 | 0412c957b976fe1091ba2b73bff0baf9393ddb59878270055e51d55465773dc9 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/figures/v22_14_operator_atom_dashboard.svg | 1 | 311 | 54fa60ff57ad18ef44c4d40f058443613c730cf5cdc784037ce703acd0361550 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/logs/v22_14_compileall.log | 1 | 127 | 1d6627f087a8c62e558e1b2f1df78d91ac11fd0827e8407237dfba9c0c5e54cd |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/logs/v22_14_efficiency.log | 1 | 16 | d8760763be63067f00789fcf42648cb7e89259b37690e733415676ec0b8b6596 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/logs/v22_14_f1_anchor_split.log | 1 | 16 | d8760763be63067f00789fcf42648cb7e89259b37690e733415676ec0b8b6596 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/logs/v22_14_import_closure.log | 1 | 471 | b4012f48c992f2a3d8fce06a537c7da96d6f81472d2e1e673a5cfa1d3d9fc80f |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/logs/v22_14_kan_basis.log | 1 | 16 | d8760763be63067f00789fcf42648cb7e89259b37690e733415676ec0b8b6596 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/logs/v22_14_s0_construct.log | 1 | 16 | d8760763be63067f00789fcf42648cb7e89259b37690e733415676ec0b8b6596 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/logs/v22_14_task_gate.log | 1 | 16 | d8760763be63067f00789fcf42648cb7e89259b37690e733415676ec0b8b6596 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_DCHE_officialization_matrix.csv | 1 | 20073 | 80d38fee827dca3ad2f41a1aa971efa2482300b684ee24773cf1137227270b58 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_DFOU_officialization_matrix.csv | 1 | 15503 | 54f3e21c50adbf3c5554d26625b79f98b2016135e59aeffb2b3898538217af1b |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_DRBF_DRAT_generic_backward_matrix.csv | 1 | 210 | 4de7e6d951eee470a8922cbcc088d085ac87b9e0fef0c92453d599ea34eed40c |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_K18_basis_tangent_coverage.csv | 1 | 1021 | 7859d8640722eabfc89c004e2aea63c6eed2da8f191b13ea39e6748e24b95059 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_K19_basis_state_operator.csv | 1 | 1945 | 4b742a44ba0b83f40b5c21cac7ce08b8c976688738e7810a44441777c6313677 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_K20_readout_to_basis_transfer.csv | 1 | 1041 | 19522342fc08cb43ac3e5b9cf034ea74b2c698f4048c6344bcf8139a89a34709 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_KAN_mapping_matrix.csv | 1 | 1945 | 4b742a44ba0b83f40b5c21cac7ce08b8c976688738e7810a44441777c6313677 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_KAN_vs_MLP_same_operator.csv | 1 | 1945 | 4b742a44ba0b83f40b5c21cac7ce08b8c976688738e7810a44441777c6313677 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_adapter_holdout_matrix.csv | 1 | 13000 | c2b5583dbb10fb12e76f269fbc7ed18da63bb3659402c0096cf510192ed571fc |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_adapter_horizon_matrix.csv | 1 | 53428 | a98d359b77d8d7447306e9e931367ab661a43060687aebed4d1e29459bd275c2 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_adapter_renaming_tests.csv | 1 | 620 | 0355f281942883aab2a2b96e8f2017b25ffd2966d31a63c9c6653f030be99418 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_artifact_index.csv | 1 | 13839 | d28704c7acb53e6849f131274a44f1708dbb8023d437dc108344720c0f8b18fe |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_artifact_nonempty_gate.csv | 1 | 392 | 875e366eed040d0b9bdc3a811edfa23741b7d51738dde09ba38904d7ea129c2e |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_basis_state_operator_step_efficiency.csv | 1 | 1945 | 4b742a44ba0b83f40b5c21cac7ce08b8c976688738e7810a44441777c6313677 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_basis_state_source_dynamics.csv | 1 | 1041 | 19522342fc08cb43ac3e5b9cf034ea74b2c698f4048c6344bcf8139a89a34709 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_basis_vs_readout_ablation.csv | 1 | 1021 | 7859d8640722eabfc89c004e2aea63c6eed2da8f191b13ea39e6748e24b95059 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_calibration_debt_matrix.csv | 1 | 1294 | d699e5b88869f975599f04ebc889ab68d4cbecf8dba8f31c6e233a84868b86fa |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_code_review_packet.zip | 1 | 508036 | 35be22c1e0e356ed7012396862e6b8df6ec37f8bd10255c1df21438fdea951c4 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_code_review_packet_manifest.csv | 1 | 35197 | 38f715d9cee46cad82937717bf2c1daaefd755918cf1d26b98ba4da429d8426d |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_code_truth_gate.csv | 1 | 313 | 2282be327adc327a5cb26bf1b39258f1b725f8c761aebf3ff714b8c7702c4ca3 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_code_truth_route.json | 1 | 85 | e1aa9c113e269004b1854c420552f0caef67e2ce500b424623ebbcbde3206401 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_command_journal.csv | 1 | 3864 | fe8da569594ad0fae6b2f0e6566549cc5dfc660cd68e9b2d116352e216eccf75 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_compileall_report.csv | 1 | 222 | 5f0dd4ab88e60231875e5faf0650cf577d68710065f30015bd9a10812526576e |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_component_timing_waterfall.csv | 1 | 47829 | 0a3d4434d40aac6273b05ffcf929df1009273cda4529c24e2914d2853621324c |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_control_attribution_matrix.csv | 1 | 65495 | 0a452034e162ca7ea06e77af70c70a609b85ca10edaf36fe480c4eb7b4f7a8f8 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_convergence_speed_matrix.csv | 1 | 1294 | d699e5b88869f975599f04ebc889ab68d4cbecf8dba8f31c6e233a84868b86fa |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_corrected_layout_tests.csv | 1 | 535 | 8ba5dfb247aff1ce1bda348b9f03232812435bb0e994504f432a329672bdfbcf |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_cotangent_suite_readback.csv | 1 | 579 | e057bcc1fa32225111616947908d3cc01a4c45fa13167ee69906b27f6d62e0c0 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_deferred_items.csv | 1 | 218 | 3fc40c4094505aad880b50e2f6be53697fdd9aa5cae45feb389517fe49cf7f1d |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_efficiency_route.json | 1 | 240 | 787eecf69b7d1b63182e1b64fd58d18f8b6aed048b03bf024c53e488c83b1ee2 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_execution_contract_tests.csv | 1 | 202 | 3e4438212db38c0c70d0a327ab82a21389503e46119e290fa06fa4f2a40ae558 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_expression_metrics_matrix.csv | 1 | 1294 | d699e5b88869f975599f04ebc889ab68d4cbecf8dba8f31c6e233a84868b86fa |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_final_decision.json | 1 | 709 | 791123c6a0639c1dabdeb53935931767cc2eb736560034f5806631c924f133c6 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_forgetting_readback_matrix.csv | 1 | 1294 | d699e5b88869f975599f04ebc889ab68d4cbecf8dba8f31c6e233a84868b86fa |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_gpu_assignment_manifest.csv | 1 | 391 | 23d2a496747101c74e9f5d9efcd322a3278cc7b3164a7fc5e39e46d0a316ee5a |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_gpu_utilization_timeline.csv | 1 | 660 | 492ad4d7c148744e6207eb19160b3522db34189d24f13cc919561489d0e5919e |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_idle_violation.csv | 1 | 202 | 3e4438212db38c0c70d0a327ab82a21389503e46119e290fa06fa4f2a40ae558 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_import_closure.csv | 1 | 166 | 0d763c8a92c7c748b612fc2e93c1be4319e95f946547b85e1c57c7f87112860a |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_independent_replication_matrix.csv | 1 | 271 | a5e562126e3bef368e975a878a03adabe4a25e8e0c58347458423a9512a2252d |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_kan_basis_route.json | 1 | 269 | 8dd82de8207eb3c8d413ce21902abc509fa8ce1bb9886db9e23fe8db85f64e66 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_manual_vs_native_vjp_comparison.csv | 1 | 47829 | 0a3d4434d40aac6273b05ffcf929df1009273cda4529c24e2914d2853621324c |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_metric_geometry_matrix.csv | 1 | 7784 | d6ccb844e509942d11ad22bf5aac21aa608e73430ade43688cd53d9840480fda |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_native_efficiency_truth_table.csv | 1 | 47829 | 0a3d4434d40aac6273b05ffcf929df1009273cda4529c24e2914d2853621324c |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_native_kernel_gradcheck.csv | 1 | 228 | 04f4c087d87f87223feaad8a7c37e1457673a457acb21197fdfa7bbbba8fa01f |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_official_fused_status_matrix.csv | 1 | 816 | a588b9e518d404d0434d0247722ec2647c1f1ab79c41cf444812529ab9502899 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_operator_atom_matrix.csv | 1 | 7784 | d6ccb844e509942d11ad22bf5aac21aa608e73430ade43688cd53d9840480fda |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_operator_atom_route.json | 1 | 165 | 793b27a05d677679a220bbedf4c05ecd5f4577d4343bc464ec7bc6573e99632c |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_operator_commit_matrix.csv | 1 | 1590 | ded603e65099a0ed70f839d7aadb8d541830e70cea0b8800984707ea6016cc7e |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_operator_commit_payload.pt | 1 | 24079 | 99cff8306e8981caa7c69acf1fd5a71c3530e54a2dc99092d86ae4210fbb6a90 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_operator_commit_route.json | 1 | 165 | 6247aeb50dd2ce1e80d198e656c9ba5729c674418e5b9e5a3db6ffa9d50c6571 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_operator_horizon_route.json | 1 | 567 | 0399750f697ba5de3b2960f9caccb46d6d7120680c5915cefd70f25caf91aa05 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_operator_law_tests.csv | 1 | 1547 | a97df978627ee47735082a1bd7e87b56736fd077d6a4ea2208cfd9001d64a167 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_operator_only_vs_anchor_matrix.csv | 1 | 53428 | a98d359b77d8d7447306e9e931367ab661a43060687aebed4d1e29459bd275c2 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_operator_role_blind_tests.csv | 1 | 171 | 32df7df38393a43b32d3615053bd2c2f930bd6d6a0816a0072b1c77d98b414dc |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_operator_step_efficiency.csv | 1 | 47829 | 0a3d4434d40aac6273b05ffcf929df1009273cda4529c24e2914d2853621324c |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_operator_variational_route.json | 1 | 295 | e9e40b98ad5bf2565ea8b40c0148d5044590ffd687b9a80926d48840d4ed2717 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_promotion_semantics_tests.csv | 1 | 164 | f0743d66ae5b6b0528174c7205490f12ac49793457da97bae8956c9e48ec9183 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_queue_drain_report.json | 1 | 156 | 30c683e8f1dc4dbf5cd70765af66161814bf5ff72d4270a695684a6c176f307b |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_required_source_files.csv | 1 | 881 | 7cfbafeeeba3d8af908c8888f4409d49b7dd1b01b4e5c94c214259d0d46ba618 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_results_bundle.zip | 1 | 506509 | 7827f582ff50ebfbd8df05c59df9733cc1c3e65915ecd4337fa76b81410e5a70 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_runnable_queue.csv | 1 | 266 | 59a43c2a2bbc9a436ef0ce0b6aa3af616cd3febf70614ec058dd7e9f885ac16a |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_semantic_firewall.csv | 1 | 171 | 32df7df38393a43b32d3615053bd2c2f930bd6d6a0816a0072b1c77d98b414dc |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_source_loss_boundary_matrix.csv | 1 | 53428 | a98d359b77d8d7447306e9e931367ab661a43060687aebed4d1e29459bd275c2 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_source_state_dynamics.csv | 1 | 352297 | 289a6a142681a062c05949ffd7a96d8565c84c73940db1ebc5bf1124065dd233 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_task_eval_matrix.csv | 1 | 1294 | d699e5b88869f975599f04ebc889ab68d4cbecf8dba8f31c6e233a84868b86fa |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_task_eval_route.json | 1 | 152 | d9bd94268f547f38b59a25448f8c155538c56c978dceb6d378b0a8c5dcf70132 |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_variant_robust_efficiency_matrix.csv | 1 | 720 | 915685518cd8b504c87715395f573e1b56f289f482ef38b0663442a61625699a |
| results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_variational_operator_solve.csv | 1 | 785 | 329faefe6be9ac70efcd4fbb3f8349e172360ea2ba96ce760d3f4469ead26069 |


## 9. Repair Continuation v22.14-R1

生成时间：2026-06-09 03:57:34 +0800

### 9.1 Repair Route

- route: `R5-OptimizerProxAnchorNoGo`
- official_operator_promotion_allowed_strict: `0`
- official_kan_carrier_promotion_allowed: `0`
- scientific_claim_allowed: `0`
- blocking_metric: `JacobianSourceStateNoGo;R7-KANBasisTangentCoverageNoGo`
- next_codex_action: `continue from remaining repair blocker; do not promote diagnostic K21 or loss-modified auxiliary rows`
- refreshed results bundle: `results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_results_bundle.zip`
- refreshed code review packet: `results/v22_14_basis_state_loss_interface_operator_fu/official_v22_14/v22_14_code_review_packet.zip`

### 9.2 修改说明

- 修复 efficiency profiler：不再截断 cotangent suite 前 4 项，纳入 `Delta-SourceTarget` 与 `Delta-RankingAdapter` 的 non-smoke robust grid。
- 新增 `experiments/run_v22_14_repair_continuation.py`，按计划尝试 Jacobian source-prox/projector、K19 dual/coupled basis operator、K20 real tau transfer 和 K21 boundary diagnostic。
- 所有 repair row 均写入独立 `*_repair*.csv`，旧 v22.14 原始 no-go 矩阵未覆盖；K21 明确 `diagnostic_only=1`，不计 official promotion。

### 9.3 Source-State Repair

| seed | loss_adapter_name | anchor_mechanism_type | jacobian_rank_cap | source_func_h3200 | source_func_h4800 | source_loss_h3200 | source_loss_h4800 | C3_source_formation_pass | C4_terminal_retention_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2213 | Delta-LossCEAdapter | optimizer_prox_jacobian_rank16 | 16 | 0.43951537148779396 | 0.46430970195807797 | 0.3020707815885544 | 0.23175398260354996 | 1 | 1 |  |
| 2213 | Delta-LossCEAdapter | optimizer_prox_jacobian_rank32 | 32 | 0.47482885640164585 | 0.3328821844568234 | 0.04604986682534218 | 0.03620307147502899 | 1 | 1 |  |
| 2213 | Delta-LossCEAdapter | source_state_projector_jacobian_rank32 | 32 | -0.013995163274436062 | -0.0064762709333000545 | 0.0 | 0.0 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-MSEAdapter | optimizer_prox_jacobian_rank16 | 16 | 0.4992191947485439 | 0.5931413111066122 | 5.632363319396973 | 6.821812629699707 | 1 | 1 |  |
| 2213 | Delta-MSEAdapter | optimizer_prox_jacobian_rank32 | 32 | 0.7454577944189051 | 0.7280821122914266 | 8.23750925064087 | 8.469660758972168 | 1 | 1 |  |
| 2213 | Delta-MSEAdapter | source_state_projector_jacobian_rank32 | 32 | -0.2273693117568769 | -0.35703916817217585 | -5.842146873474121 | -8.291645050048828 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-RankingAdapter | optimizer_prox_jacobian_rank16 | 16 | 0.02798130308950264 | -0.057826292091273546 | -0.6104768067598343 | -0.5730050317943096 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-RankingAdapter | optimizer_prox_jacobian_rank32 | 32 | -0.21259029469447288 | -0.1534809425497501 | -0.20059216395020485 | -0.3200293742120266 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-RankingAdapter | source_state_projector_jacobian_rank32 | 32 | -0.3700487070590661 | -0.3863992523855917 | -4.4018030166625977e-05 | -7.230043411254883e-05 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-PreferenceAdapter-smoke | optimizer_prox_jacobian_rank16 | 16 | -0.2824048460677979 | -0.2872942548791382 | -0.29064029455184937 | -0.3599128723144531 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-PreferenceAdapter-smoke | optimizer_prox_jacobian_rank32 | 32 | -1.1096295622603547 | -0.9611890554847713 | -0.36209386587142944 | -0.42892061173915863 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-PreferenceAdapter-smoke | source_state_projector_jacobian_rank32 | 32 | -1.4367065786703646 | -2.223848409186257 | -7.545948028564453e-05 | -9.72747802734375e-05 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-StableRandom-control | optimizer_prox_jacobian_rank16 | 16 | -0.9508858892219204 | -2.1186695884621227 | -434057.69470214844 | -464045.8791503906 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-StableRandom-control | optimizer_prox_jacobian_rank32 | 32 | -3.6304501411047325 | -5.7677711184473 | -354144.85009765625 | -367987.25927734375 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-StableRandom-control | source_state_projector_jacobian_rank32 | 32 | -0.0002423907502375755 | -0.0025612848008707 | -1260777.9838256836 | -1927876.0711669922 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-RandomMatched-control | optimizer_prox_jacobian_rank16 | 16 | -0.9264057702483072 | -2.004841540825775 | -194117.2783203125 | -208041.87438964844 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-RandomMatched-control | optimizer_prox_jacobian_rank32 | 32 | -3.542569540326204 | -5.608074107850676 | -156477.87890625 | -161989.390625 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |
| 2213 | Delta-RandomMatched-control | source_state_projector_jacobian_rank32 | 32 | 0.0 | 0.0 | -556097.2952880859 | -850438.2979736328 | 0 | 0 | source_func_or_source_loss_or_control_gate_failed |

Analysis: 这些 row 是 no-loss-modification optimizer dynamics；如果仍不过，说明 v22.13 auxiliary loss anchor 不能简单转写为 readout-Jacobian source-state correction。

### 9.4 Efficiency Repair

| carrier | variant | profile_rows | ratio_pass_rows | cotangent_types | planned_cotangent_suite_complete | planned_batch_grid_complete | planned_hidden_grid_complete | variant_robust_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | CHE22.13-R1-k3-corrected-layout-native-vjp | 24 | 20 | Delta-Gaussian;Delta-LossCEAdapter;Delta-MSEAdapter;Delta-RankingAdapter;Delta-SourceTarget;Delta-StableRandom | 1 | 1 | 1 | 0 | ratio_outlier_or_planned_grid_not_closed |
| D-CHE | CHE22.13-R2-k5-gradbuf-no-materialize-native-vjp | 24 | 24 | Delta-Gaussian;Delta-LossCEAdapter;Delta-MSEAdapter;Delta-RankingAdapter;Delta-SourceTarget;Delta-StableRandom | 1 | 1 | 1 | 1 |  |
| D-CHE | CHE22.13-R3-lowdegree-readout-operator-step | 24 | 23 | Delta-Gaussian;Delta-LossCEAdapter;Delta-MSEAdapter;Delta-RankingAdapter;Delta-SourceTarget;Delta-StableRandom | 1 | 1 | 1 | 0 | ratio_outlier_or_planned_grid_not_closed |
| D-CHE | CHE22.13-R4-corrected-layout-source-state-fused-commit | 24 | 24 | Delta-Gaussian;Delta-LossCEAdapter;Delta-MSEAdapter;Delta-RankingAdapter;Delta-SourceTarget;Delta-StableRandom | 1 | 1 | 1 | 1 |  |
| D-FOU | FOU22.13-R1-lowfreq-bandreadout-native-vjp | 24 | 20 | Delta-Gaussian;Delta-LossCEAdapter;Delta-MSEAdapter;Delta-RankingAdapter;Delta-SourceTarget;Delta-StableRandom | 1 | 1 | 1 | 0 | ratio_outlier_or_planned_grid_not_closed |
| D-FOU | FOU22.13-R2-tablelookup-bandreadout-native-operator-step | 24 | 24 | Delta-Gaussian;Delta-LossCEAdapter;Delta-MSEAdapter;Delta-RankingAdapter;Delta-SourceTarget;Delta-StableRandom | 1 | 1 | 1 | 1 |  |
| D-FOU | FOU22.13-R3-lowfreq-source-state-fused-commit | 24 | 24 | Delta-Gaussian;Delta-LossCEAdapter;Delta-MSEAdapter;Delta-RankingAdapter;Delta-SourceTarget;Delta-StableRandom | 1 | 1 | 1 | 1 |  |

Analysis: robust gate 只按真实 profile rows 计数；ratio 或 grid 任一未闭合仍保持 blocked。

### 9.5 KAN Repair


_无落盘 rows。_


_无落盘 rows。_


_无落盘 rows。_

Analysis: K19/K20 repair 仍需 basis channel 同时满足 source_func/source_loss；readout-coupled 或 K21 diagnostic 不会被当成 KAN basis official success。
