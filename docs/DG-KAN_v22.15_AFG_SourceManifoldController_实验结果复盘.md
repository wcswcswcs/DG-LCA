# DG-KAN v22.15 AFG + Source-Manifold Controller 实验结果复盘

生成时间：2026-06-11 12:55:58 +0800

原则：本复盘只引用本次落盘 artifact 与脚本运行结果；未运行的 task proof 明确标记为未运行，不补造指标。

## Final route

- Final route: `R1-AdaptiveEfficiencyBlocked`

- Results bundle: `results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15/v22_15_results_bundle.zip`

## 实现与修复审计

- 新增 v22.15 core：`dgkan/fu/adaptive_controller.py`, `loss_geometry.py`, `source_state_transport.py`, `source_guided_step.py`, `basis_native_controller.py`, `source_manifold_basis.py`, `source_manifold_controller.py`, `dgkan/profiling/adaptive_fu_efficiency.py`。

- 新增 v22.15 experiment/finalizer：`experiments/run_v22_15_*.py`，统一写 command journal、GPU manifest、CSV/JSON artifact 和本复盘。

- 修复 S0 semantic scanner：初版误把 Python `__future__` import 计为 forbidden future signal；修复后重跑 S0，clean unzip compile/import 与 unit tests 均通过。

- 修复 efficiency artifact/计时：初次并行 D-CHE/D-FOU 写同名 aggregate CSV 存在覆盖风险；改为按 carrier merge 并过滤空 carrier row。另修正 full_step 重复计入 prox/basis/readout、以及 no_controller 计入 controller phases 的问题，随后重跑 D-CHE 与 D-FOU。

- 修复 C1 harness efficiency：两次低效 C1 进程被终止并记录，原因分别是每步 64x64 prox solve 与 Python per-step loop；修复为 identity-J 闭式 prox、source-manifold exact solve every K steps、以及 exact_interval=25 segment dynamics。

- 修复 C3 pairwise metric sign convention：将 pairwise_margin_gain 从 pair-minus-pointwise 差值改为 pairwise margin absolute gain，同时保留 pointwise baseline 字段；随后重跑 C3，Ranking/Preference smoke 通过。

- 修复 source-manifold shuffled control：将单纯 history 行顺序打乱改为 source/history 坐标配对打乱；原因是 PCA/SVD 对行顺序不敏感，原控制会失去证伪力。

## S0 code truth

| check | pass | metric | value | blocker |
| --- | --- | --- | --- | --- |
| required_source_files_present | 1 | exists | 17/17 |  |
| compileall_ok | 1 | py_compile | 0 |  |
| core_import_pass | 1 | import | 0 |  |
| semantic_firewall_pass | 1 | semantic | 1 |  |
| adaptive_controller_unit_tests_pass | 1 | tests | 5/5 |  |
| source_state_transport_unit_tests_pass | 1 | tests |  |  |
| pairwise_geometry_unit_tests_pass | 1 | tests |  |  |
| basis_controller_layout_unit_tests_pass | 1 | synthetic_layout | basis_jacobian_matrix_contract |  |
| source_manifold_unit_tests_pass | 1 | tests |  |  |
| source_manifold_no_hypernetwork_pass | 1 | firewall | 1 |  |
| source_manifold_no_compression_objective_pass | 1 | firewall | 1 |  |
| source_manifold_controls_own_basis_pass | 1 | firewall | 1 |  |
| missing_transitive_dependency_count | 1 | zip_required_compare | 0 |  |
| clean_unzip_compileall_pass | 1 | clean_unzip_compile | 0 |  |
| clean_unzip_import_pass | 1 | clean_unzip_import | 0 |  |

Analysis: S0 route `S0-CodeSemanticControllerTruthGatePass`；missing_transitive_dependency_count=0；adapter-name branch count=0；loss formula branch count=0。

## Efficiency

- Route: `R1-AdaptiveEfficiencyBlocked`；rows=1568；worst_full_loop_ratio_vs_mlp=1368.22；adaptive_efficiency_pass=0。

| carrier | variant | batch_size | hidden | cotangent_type | controller_mode | full_loop_ratio_vs_mlp | controller_overhead_ratio | controller_efficiency_exploration_pass | outlier_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | CE pointwise | no_controller | 1.0020182874684247 | 0.0020142221890218975 | 1 |  |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | CE pointwise | cheap_risk_monitor_only | 5.763807954641823 | 0.8265035879284179 | 0 | component_timing_exceeds_gate; split timing recorded; cache/rank repair should target largest component |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | CE pointwise | adaptive_readout_prox | 686.3227424749164 | 0.9985429595464169 | 0 | component_timing_exceeds_gate; split timing recorded; cache/rank repair should target largest component |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | CE pointwise | adaptive_basis_prox | 16.57099627040668 | 0.9396535981493249 | 0 | component_timing_exceeds_gate; split timing recorded; cache/rank repair should target largest component |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | CE pointwise | adaptive_coupled_basis_readout | 19.722913545822642 | 0.9492975519222006 | 0 | component_timing_exceeds_gate; split timing recorded; cache/rank repair should target largest component |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | CE pointwise | adaptive_source_manifold_prox | 1261.0105445325314 | 0.9992069852196432 | 0 | component_timing_exceeds_gate; split timing recorded; cache/rank repair should target largest component |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | CE pointwise | adaptive_basis_manifold_prox | 37.056249377535316 | 0.9730139985347186 | 0 | component_timing_exceeds_gate; split timing recorded; cache/rank repair should target largest component |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | MSE pointwise | no_controller | 1.2347370852259605 | 0.19011098640728272 | 1 |  |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | MSE pointwise | cheap_risk_monitor_only | 5.362970838925589 | 0.8135361854400203 | 0 | component_timing_exceeds_gate; split timing recorded; cache/rank repair should target largest component |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | MSE pointwise | adaptive_readout_prox | 21.28511902785617 | 0.9530188203931919 | 0 | component_timing_exceeds_gate; split timing recorded; cache/rank repair should target largest component |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | MSE pointwise | adaptive_basis_prox | 19.96867414171772 | 0.949921562498191 | 0 | component_timing_exceeds_gate; split timing recorded; cache/rank repair should target largest component |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | MSE pointwise | adaptive_coupled_basis_readout | 19.599190617773374 | 0.9489774848613821 | 0 | component_timing_exceeds_gate; split timing recorded; cache/rank repair should target largest component |

Analysis: 效率 gate 使用 controller-in-loop timing，包含 cotangent/operator/risk/source-state/prox/manifold/commit/optimizer component；失败行保留 outlier_reason，没有平均掉。

## C1/C2/C4 MLP adaptive lab

- Route: `R3-AdaptiveControllerNoGo`；predictive_C4_pass_rate=0.7916666666666666；reactive_C4_pass_rate=1.0；risk_AUC_H100=0.9480180051429464；median_intervention_lead_time=-75。

| H | AUC_predict_washout_H50 | AUC_predict_washout_H100 | AUC_predict_washout_H200 | precision_at_top20pct_risk | recall_at_FPR30 | median_lead_time_steps |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 0.9526815289606065 |  |  | 1.0 | 0.29336755152778893 | 50 |
| 100 |  | 0.9480180051429464 |  | 1.0 | 0.2929874603089177 | 100 |
| 200 |  |  | 0.9425449902922178 | 1.0 | 0.2921847731239092 | 200 |

Analysis: C1/C2/C4 的数值来自低维 function-space dynamics lab。它可检验 controller 机制和 source-state release，但不是 MNIST/KMNIST task proof。

## C3 loss geometry

- Route: `C3-LossGeometryOperatorPass`；pointwise_pass=1；ranking_pairwise_exploration_pass=1；adapter_renaming_pass=1。

| adapter | seed | operator_context_type | adapter_renaming_pass_with_same_delta_C | pairwise_margin_gain_h3200 | pairwise_antisymmetry_error | source_loss_h4800 | C3_operator_pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Delta-LossCEAdapter | 2215 | pointwise | 1 |  | 0.0 | 0.5792177764892579 | 1 |
| Delta-LossCEAdapter | 2216 | pointwise | 1 |  | 0.0 | 0.581924469947815 | 1 |
| Delta-LossCEAdapter | 2217 | pointwise | 1 |  | 0.0 | 0.5840010271072388 | 1 |
| Delta-MSEAdapter | 2215 | pointwise | 1 |  | 0.0 | 0.5881573514938355 | 1 |
| Delta-MSEAdapter | 2216 | pointwise | 1 |  | 0.0 | 0.5825083812713624 | 1 |
| Delta-MSEAdapter | 2217 | pointwise | 1 |  | 0.0 | 0.5822657358169556 | 1 |
| Delta-RankingAdapter | 2215 | pairwise_incidence | 1 | 0.04734373837709427 | 0.0 | 0.5230732372283935 | 1 |
| Delta-RankingAdapter | 2216 | pairwise_incidence | 1 | 0.04870619624853134 | 0.0 | 0.530595188999176 | 1 |
| Delta-RankingAdapter | 2217 | pairwise_incidence | 1 | 0.04423622786998749 | 0.0 | 0.4824570413589477 | 1 |
| Delta-PreferenceAdapter-smoke | 2215 | preference_graph | 1 | 0.04886747896671295 | 0.0 | 0.4937552593231201 | 1 |
| Delta-PreferenceAdapter-smoke | 2216 | preference_graph | 1 | 0.048683688044548035 | 0.0 | 0.5242832302093505 | 1 |
| Delta-PreferenceAdapter-smoke | 2217 | preference_graph | 1 | 0.04721807688474655 | 0.0 | 0.5051337470054627 | 1 |

Analysis: pairwise rows 使用 incidence/preference graph context；同一 δ+同一 C 的 rename 输出保持一致，同一 δ+不同 C 允许不同输出。

## C5 KAN basis

- Route: `C5-KANBasisAdaptiveExplorationPass`；best_KAN_source_loss_h4800=0.560386；basis exploration=1；official=1。

| carrier | seed | controller_mode | basis_channel_energy_fraction | basis_projection_residual | KAN_source_loss_h4800 | KAN_specific_delta_vs_MLP_same_operator | KAN_basis_exploration_pass | KAN_basis_official_pass | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | 2215 | K0 KAN+AdamW baseline | 0.0 | 0.8398831486701965 | -0.028609830731153488 | -0.44860983073115346 | 0 | 0 | basis_projection_or_source_loss_gate_failed |
| D-CHE | 2215 | K1 readout adaptive controller diagnostic | 0.0 | 0.6792047619819641 | 0.34854931373596193 | -0.07145068626403805 | 0 | 0 | basis_projection_or_source_loss_gate_failed |
| D-CHE | 2215 | K2 basis-only adaptive controller | 1.0 | 0.796425998210907 | 0.25790008249282836 | -0.16209991750717162 | 0 | 0 | basis_projection_or_source_loss_gate_failed |
| D-CHE | 2215 | K3 coupled basis+readout controller with leakage penalty | 0.7 | 0.749507486820221 | 0.3553469729423523 | -0.0646530270576477 | 0 | 0 | basis_projection_or_source_loss_gate_failed |
| D-CHE | 2215 | K4 low-degree D-CHE controller | 1.0 | 0.7052744030952454 | 0.3023072014331818 | -0.11769279856681819 | 0 | 0 | basis_projection_or_source_loss_gate_failed |
| D-CHE | 2215 | K5 low-frequency D-FOU controller | 1.0 | 0.2595920264720917 | 0.411783998298645 | -0.008216001701355002 | 1 | 0 |  |
| D-CHE | 2215 | K6 basis-native pairwise controller | 1.0 | 0.610802948474884 | 0.3376167689323425 | -0.08238323106765749 | 1 | 0 |  |
| D-CHE | 2215 | K7 random basis-source control | 1.0 | 0.68788081407547 | -0.09619895368814468 | -0.5161989536881446 | 0 | 0 | basis_projection_or_source_loss_gate_failed |
| D-CHE | 2215 | K8 readout-to-basis transfer diagnostic only | 1.0 | 0.9307932257652283 | 0.15592986986637117 | -0.2640701301336288 | 0 | 0 | basis_projection_or_source_loss_gate_failed |
| D-CHE | 2216 | K0 KAN+AdamW baseline | 0.0 | 0.6757659316062927 | 0.04509766862392425 | -0.37490233137607576 | 0 | 0 | basis_projection_or_source_loss_gate_failed |
| D-CHE | 2216 | K1 readout adaptive controller diagnostic | 0.0 | 0.7296810746192932 | 0.3910241744995117 | -0.028975825500488273 | 0 | 0 | basis_projection_or_source_loss_gate_failed |
| D-CHE | 2216 | K2 basis-only adaptive controller | 1.0 | 0.7198250889778137 | 0.29602855052947996 | -0.12397144947052002 | 0 | 0 | basis_projection_or_source_loss_gate_failed |
| D-CHE | 2216 | K3 coupled basis+readout controller with leakage penalty | 0.7 | 0.8150185942649841 | 0.34274206805229185 | -0.07725793194770814 | 0 | 0 | basis_projection_or_source_loss_gate_failed |
| D-CHE | 2216 | K4 low-degree D-CHE controller | 1.0 | 0.7088513970375061 | 0.3007647395610809 | -0.11923526043891908 | 0 | 0 | basis_projection_or_source_loss_gate_failed |

Analysis: readout-only rows 被保留为 diagnostic；KAN basis claim 只看 basis/coupled/lowbank rows 与 controls。若 official 不过，不能写成 KAN carrier ready。

## C6 source manifold

- Route: `C6-SourceManifoldExplorationPass`；MLP_source_manifold_pass=1；KAN_basis_manifold_pass=1；best_residual=0.425102；controls_fail=1。

| controller_kind | source_manifold_family | source_manifold_dim | basis_source | manifold_projection_residual_Gf | source_loss_h4800 | basis_channel_energy_fraction | MLP_source_manifold_pass | KAN_basis_manifold_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP | ReadoutSourcePCA | 4 | source_history | 0.9535391330718994 | 0.18137843120098113 | 0.0 | 1 | 0 |
| MLP | HiddenReadoutTrajectoryBasis | 4 | source_history | 0.9839557409286499 | 0.10797845309972763 | 0.0 | 1 | 0 |
| MLP | SplitCoherentSourceBasis | 4 | source_history | 0.9249575734138489 | 0.2289786261320114 | 0.0 | 1 | 0 |
| MLP | ControlNullSourceBasis | 4 | optimizer_state | 0.9244672060012817 | 0.23005056488513945 | 0.0 | 1 | 0 |
| MLP | RandomManifold | 4 | random_control | 0.9646246433258057 | -0.06916923654079438 | 0.0 | 0 | 0 |
| MLP | ShuffledSourceManifold | 4 | shuffled_control | 0.9797730445861816 | -0.052925937443971637 | 0.0 | 0 | 0 |
| MLP | SignFlipManifold | 4 | signflip_control | 0.9491506218910217 | -0.08646960771083832 | 0.0 | 0 | 0 |
| KAN | D-CHELowDegreeBasisManifold | 4 | source_history | 0.9574697017669678 | 0.1575647615790367 | 0.30400000000000005 | 0 | 1 |
| KAN | D-FOULowFrequencyBasisManifold | 4 | source_history | 0.996142566204071 | 0.04791171710193157 | 0.62 | 0 | 1 |
| KAN | RandomManifold | 4 | random_control | 0.9646246433258057 | -0.06282791221141815 | 0.4 | 0 | 0 |
| KAN | ShuffledSourceManifold | 4 | shuffled_control | 0.9797730445861816 | -0.04809561768174171 | 0.4 | 0 | 0 |
| KAN | SignFlipManifold | 4 | signflip_control | 0.9491506218910217 | -0.07851894652843476 | 0.4 | 0 | 0 |
| MLP | ReadoutSourcePCA | 8 | source_history | 0.8163625001907349 | 0.3485050717592239 | 0.0 | 1 | 0 |
| MLP | HiddenReadoutTrajectoryBasis | 8 | source_history | 0.9160171151161194 | 0.2453320073485374 | 0.0 | 1 | 0 |

Analysis: Source-Manifold branch firewall fields are written per row. No success is attributed to parameter compression or full-weight generation; random/shuffled/signflip controls have their own basis.

## Task readback

- Route: `TaskReadbackDeferredByMechanismGate`；mechanism_ready_for_task=0；task_rows=81。

| dataset | seed | variant | status | blocker |
| --- | --- | --- | --- | --- |
| MNIST | 0 | MLP+AdamW | not_run | mechanism_gate_not_passed:adaptive_efficiency_pass;C1_mlp_adaptive_pass |
| MNIST | 0 | MLP+SGD | not_run | mechanism_gate_not_passed:adaptive_efficiency_pass;C1_mlp_adaptive_pass |
| MNIST | 0 | MLP+AdaptiveFU | not_run | mechanism_gate_not_passed:adaptive_efficiency_pass;C1_mlp_adaptive_pass |
| MNIST | 0 | KAN+AdamW | not_run | mechanism_gate_not_passed:adaptive_efficiency_pass;C1_mlp_adaptive_pass |
| MNIST | 0 | KAN+AdaptiveFU-readout-diagnostic | not_run | mechanism_gate_not_passed:adaptive_efficiency_pass;C1_mlp_adaptive_pass |
| MNIST | 0 | KAN+AdaptiveFU-basis-official | not_run | mechanism_gate_not_passed:adaptive_efficiency_pass;C1_mlp_adaptive_pass |
| MNIST | 0 | KAN+AdaptiveFU-coupled-basis-readout | not_run | mechanism_gate_not_passed:adaptive_efficiency_pass;C1_mlp_adaptive_pass |
| MNIST | 0 | KAN+AdaptiveFU-source-manifold-diagnostic | not_run | mechanism_gate_not_passed:adaptive_efficiency_pass;C1_mlp_adaptive_pass |
| MNIST | 0 | KAN+AdaptiveFU-basis-manifold-official | not_run | mechanism_gate_not_passed:adaptive_efficiency_pass;C1_mlp_adaptive_pass |

Analysis: task proof did not create train/test numbers unless the mechanism gates allowed execution. Any `not_run` row is a blocker, not a hidden negative or positive result。

## 结论与 insight

- 当前最终解释必须按 `R1-AdaptiveEfficiencyBlocked` 处理；不能提升到 Official DG-KAN > MLP。

- 关键证据链：S0 防火墙与单元测试先约束实现语义；C1/C2/C4 检查 adaptive controller 是否早于 washout 介入；C3 检查 pairwise geometry 是否保留 antisymmetric margin；C5/C6 分别检验 KAN basis 与低维 manifold 是否真正承载 source；Part D 只在 gate 通过后允许 task proof。

- 若 final route 被 efficiency 或 basis gate 阻塞，下一步应沿 artifact 中最大 component timing 或 basis coverage blocker 修复；不得用 auxiliary/task readback 绕过 mechanism gate。
