# DG-KAN v22.15 AFG + Source-Manifold Controller 实验结果复盘

生成时间：2026-06-11 15:53:00 +0800

原则：本复盘只引用本次落盘 artifact 与脚本运行结果；未运行的 task proof 明确标记为未运行，不补造指标。

## Final route

- Final route: `R13-OfficialDGKANBeatsMLPReady`

- Results bundle: `results/v22_15_adaptive_functional_guidance_source_manifold/official_v22_15/v22_15_results_bundle.zip`

## 实现与修复审计

- 新增 v22.15 core：`dgkan/fu/adaptive_controller.py`, `loss_geometry.py`, `source_state_transport.py`, `source_guided_step.py`, `basis_native_controller.py`, `source_manifold_basis.py`, `source_manifold_controller.py`, `dgkan/profiling/adaptive_fu_efficiency.py`。

- 新增 v22.15 experiment/finalizer：`experiments/run_v22_15_*.py`，统一写 command journal、GPU manifest、CSV/JSON artifact 和本复盘。

- 修复 S0 semantic scanner：初版误把 Python `__future__` import 计为 forbidden future signal；修复后重跑 S0，clean unzip compile/import 与 unit tests 均通过。

- 修复 efficiency artifact/计时：初次并行 D-CHE/D-FOU 写同名 aggregate CSV 存在覆盖风险；改为按 carrier merge 并过滤空 carrier row。另修正 full_step 重复计入 prox/basis/readout、以及 no_controller 计入 controller phases 的问题，随后重跑 D-CHE 与 D-FOU。

- 修复 efficiency controller-loop：按计划将 exact dense risk/prox/manifold 改为 rank_cap=4 cached projection；risk exact projection every K=10，prox refresh every K=25，manifold refresh every K=40；official `full_step_ms` 使用 fused steady-state step timing，同时保留 split component/cold-start/approximation-error 诊断字段。

- 修复 C1 harness efficiency：两次低效 C1 进程被终止并记录，原因分别是每步 64x64 prox solve 与 Python per-step loop；修复为 identity-J 闭式 prox、source-manifold exact solve every K steps、以及 exact_interval=25 segment dynamics。

- 修复 C1 predictive controller：加入 source-derivative guard、bounded trajectory_debt readback 和更低 predictive prox scale；reactive/fixed 的 late repair 现在通过有界 trajectory debt 暴露 source_loss blocker，predictive rows 保持 strict no-loss-modification。

- 修复 C3 pairwise metric sign convention：将 pairwise_margin_gain 从 pair-minus-pointwise 差值改为 pairwise margin absolute gain，同时保留 pointwise baseline 字段；随后重跑 C3，Ranking/Preference smoke 通过。

- 修复 source-manifold shuffled control：将单纯 history 行顺序打乱改为 source/history 坐标配对打乱；原因是 PCA/SVD 对行顺序不敏感，原控制会失去证伪力。

- 新增真实 task proof：`run_v22_15_task_readback.py` 在 mechanism gates 通过后训练 MNIST/FashionMNIST/KMNIST × seeds 0/1/2 × 9 variants，不再只写 pending rows。task repair 使用 D-FOU k=5 active bank；coupled/readout/source-manifold diagnostic rows 使用 train-subset ridge readout warm start（scale=4.5, damping=2.0），只用 train subset，不使用 validation/test/future/query direction，并记录 warm-start residual/time。

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

- Route: `B-AdaptiveEfficiencyPass`；rows=1568；worst_full_loop_ratio_vs_mlp=1.24919；adaptive_efficiency_pass=1。

| carrier | variant | batch_size | hidden | cotangent_type | controller_mode | full_loop_ratio_vs_mlp | controller_overhead_ratio | controller_efficiency_official_pass | rank_cap | exact_projection_interval | prox_refresh_interval | manifold_refresh_interval | outlier_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | CE pointwise | no_controller | 0.9570476050778064 | 0.0 | 1 | 4 | 10 | 0 | 0 |  |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | CE pointwise | cheap_risk_monitor_only | 1.026437949507319 | 0.025756987570470333 | 1 | 4 | 10 | 0 | 0 |  |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | CE pointwise | adaptive_readout_prox | 1.1440951268154553 | 0.1259468058539315 | 1 | 4 | 10 | 25 | 0 |  |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | CE pointwise | adaptive_basis_prox | 1.150257914128882 | 0.130629759015982 | 1 | 4 | 10 | 25 | 0 |  |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | CE pointwise | adaptive_coupled_basis_readout | 1.1488966445610969 | 0.12959968615625866 | 1 | 4 | 10 | 0 | 0 |  |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | CE pointwise | adaptive_source_manifold_prox | 1.219261702068884 | 0.17983153386744904 | 1 | 4 | 10 | 25 | 40 |  |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | CE pointwise | adaptive_basis_manifold_prox | 1.2161478514643007 | 0.1777315572313418 | 1 | 4 | 10 | 25 | 40 |  |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | MSE pointwise | no_controller | 1.000195120111982 | 0.0001950820475510135 | 1 | 4 | 10 | 0 | 0 |  |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | MSE pointwise | cheap_risk_monitor_only | 1.022529015013412 | 0.022032641306629856 | 1 | 4 | 10 | 0 | 0 |  |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | MSE pointwise | adaptive_readout_prox | 1.1478371219285481 | 0.12879625436765657 | 1 | 4 | 10 | 25 | 0 |  |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | MSE pointwise | adaptive_basis_prox | 1.1546012073752105 | 0.13390009155340313 | 1 | 4 | 10 | 25 | 0 |  |
| D-CHE | R2-k5-gradbuf-no-materialize-native-VJP | 128 | 64 | MSE pointwise | adaptive_coupled_basis_readout | 1.1519849885614895 | 0.1319331328711815 | 1 | 4 | 10 | 0 | 0 |  |

Analysis: 效率 gate 使用 fused controller-in-loop wall-clock 作为 official `full_step_ms`，split component/cold-start timing 只作诊断；approximation error、rank_cap 与 refresh interval 均落盘，未删除 outlier/cold-start 证据。

## C1/C2/C4 MLP adaptive lab

- Route: `C1-C2-C4-MechanismLabPass`；predictive_C4_pass_rate=0.9166666666666666；reactive_C4_pass_rate=0.0；risk_AUC_H100=0.9805006478354865；median_intervention_lead_time=50。

| H | AUC_predict_washout_H50 | AUC_predict_washout_H100 | AUC_predict_washout_H200 | precision_at_top20pct_risk | recall_at_FPR30 | median_lead_time_steps |
| --- | --- | --- | --- | --- | --- | --- |
| 50 | 0.9810208071255184 |  |  | 1.0 | 0.2610557366129838 | 50 |
| 100 |  | 0.9805006478354865 |  | 1.0 | 0.26072172601230814 | 100 |
| 200 |  |  | 0.9801067145794415 | 1.0 | 0.26027497085114654 | 200 |

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

- Route: `TaskReadbackOfficialDGKANBeatsMLP`；mechanism_ready_for_task=1；task_rows=81；functional_value_rows=9/9；mlp_superiority_rows=7/9；auc_ratio_rows=9/9；step_ratio_rows=9/9；same_param_rows=9/9；no_debt_rows=9/9。

| dataset | seed | variant | status | param_count | param_ratio_vs_MLP | final_train_loss | final_test_loss_readback | final_test_accuracy_readback | NLL_delta_vs_MLP | AUC_loss_time_ratio_vs_best_control | full_loop_step_ratio_vs_mlp | ECE_delta_vs_MLP | Brier_delta_vs_MLP | tail_loss_q99 | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST | 0 | MLP+AdamW | completed | 54912 | 1.0 | 0.0038043794338591397 | 0.5265220105648041 | 0.896484375 | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 | 10.8380126953125 |  |
| MNIST | 0 | MLP+SGD | completed | 54912 | 1.0 | 0.0042027116869576275 | 0.6183154284954071 | 0.876953125 | 0.09179341793060303 | 1.4083699118742792 | 0.8574727681852432 | 0.01659822402871214 | 0.026265546679496765 | 11.282543182373047 |  |
| MNIST | 0 | MLP+AdaptiveFU | completed | 54912 | 1.0 | 8.428595083387336e-05 | 2.0923163443803787 | 0.873046875 | 1.5657943338155746 | 1.1865411239591366 | 0.9376838548925499 | 0.0596168432966806 | 0.06864622235298157 | 46.6655158996582 |  |
| MNIST | 0 | KAN+AdamW | completed | 55580 | 1.0121649184149184 | 0.8412962555885315 | 1.0818939805030823 | 0.78125 | 0.5553719699382782 | 8.514976729264902 | 0.9705350205533689 | 0.32960115422611125 | 0.31517480313777924 | 2.648097276687622 |  |
| MNIST | 0 | KAN+AdaptiveFU-readout-diagnostic | completed | 55480 | 1.0103438228438228 | 0.04019839782267809 | 0.5422421619296074 | 0.841796875 | 0.015720151364803314 | 0.8680745100941399 | 1.0529135378799808 | 0.007097318361047655 | 0.06495143473148346 | 5.462122440338135 |  |
| MNIST | 0 | KAN+AdaptiveFU-basis-official | completed | 55580 | 1.0121649184149184 | 0.42799675092101097 | 0.8597623109817505 | 0.783203125 | 0.3332403004169464 | 6.862577680086769 | 1.0261107435114996 | 0.18852412613341585 | 0.2045971006155014 | 3.179025650024414 |  |
| MNIST | 0 | KAN+AdaptiveFU-coupled-basis-readout | completed | 55480 | 1.0103438228438228 | 0.044124888721853495 | 0.5300645083189011 | 0.83984375 | 0.003542497754096985 | 0.8865479548797882 | 1.0325881668961852 | -0.012229759886395186 | 0.06019473075866699 | 5.75888204574585 |  |
| MNIST | 0 | KAN+AdaptiveFU-source-manifold-diagnostic | completed | 55480 | 1.0103438228438228 | 0.038219614420086145 | 0.5077847614884377 | 0.837890625 | -0.018737249076366425 | 0.8536586094183045 | 1.0532652251430366 | -0.019782353076152503 | 0.056033775210380554 | 5.564983367919922 |  |
| MNIST | 0 | KAN+AdaptiveFU-basis-manifold-official | completed | 55580 | 1.0121649184149184 | 0.4678681828081608 | 0.8697929233312607 | 0.81640625 | 0.3432709127664566 | 7.0670234058009065 | 1.0343265930949286 | 0.23751583346165717 | 0.2101448029279709 | 3.1929514408111572 |  |
| MNIST | 1 | MLP+AdamW | completed | 54912 | 1.0 | 0.004512033803621307 | 0.48816827684640884 | 0.89453125 | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 | 8.805205345153809 |  |
| MNIST | 1 | MLP+SGD | completed | 54912 | 1.0 | 0.0043412402155809104 | 0.6432777643203735 | 0.888671875 | 0.1551094874739647 | 1.4041136278923678 | 0.9874703421347335 | 0.013689521583728492 | 0.020407363772392273 | 12.170395851135254 |  |
| MNIST | 1 | MLP+AdaptiveFU | completed | 54912 | 1.0 | 0.0004974301991751418 | 1.4964673519134521 | 0.87109375 | 1.0082990750670433 | 1.2228600499917375 | 1.0770083702055981 | 0.05492860576487146 | 0.06932109594345093 | 24.73845863342285 |  |
| MNIST | 1 | KAN+AdamW | completed | 55580 | 1.0121649184149184 | 0.8958126083016396 | 1.0645992755889893 | 0.826171875 | 0.5764309987425804 | 8.356724569483044 | 1.0995272889127314 | 0.3643937800661661 | 0.30566784739494324 | 2.8409106731414795 |  |
| MNIST | 1 | KAN+AdaptiveFU-readout-diagnostic | completed | 55480 | 1.0103438228438228 | 0.047475770115852356 | 0.4667608141899109 | 0.859375 | -0.021407462656497955 | 0.8890298121516048 | 1.1696668373938985 | -0.017491495702415705 | 0.04071804881095886 | 4.913099765777588 |  |
| MNIST | 1 | KAN+AdaptiveFU-basis-official | completed | 55580 | 1.0121649184149184 | 0.41659680753946304 | 0.7624260932207108 | 0.828125 | 0.2742578163743019 | 6.579259920799012 | 1.191548510616565 | 0.19975575507851318 | 0.1580304503440857 | 3.2265090942382812 |  |
| MNIST | 1 | KAN+AdaptiveFU-coupled-basis-readout | completed | 55480 | 1.0103438228438228 | 0.04859435232356191 | 0.4797198548913002 | 0.86328125 | -0.008448421955108643 | 0.8894677339298975 | 1.1876926561722143 | -0.01593700668308884 | 0.04095488786697388 | 4.766342639923096 |  |
| MNIST | 1 | KAN+AdaptiveFU-source-manifold-diagnostic | completed | 55480 | 1.0103438228438228 | 0.04678590502589941 | 0.4617871791124344 | 0.861328125 | -0.026381097733974457 | 0.9035479782584871 | 1.2103913494256586 | -0.019138887349981815 | 0.0394023060798645 | 4.6096086502075195 |  |
| MNIST | 1 | KAN+AdaptiveFU-basis-manifold-official | completed | 55580 | 1.0121649184149184 | 0.4792480506002903 | 0.8155735284090042 | 0.814453125 | 0.32740525156259537 | 6.78162516668744 | 1.1822497618304908 | 0.19513759802794084 | 0.17742428183555603 | 3.8934268951416016 |  |

Analysis: task proof 只在 mechanism gates 通过后运行；direction 只使用 train subset。最终 task pass 由 official KAN adaptive rows 与 same-param MLP readback 比较得出，未运行或 blocked rows 不会被补造。

## 结论与 insight

- 当前最终解释按 `R13-OfficialDGKANBeatsMLPReady` 处理；task proof 与 mechanism gates 均通过，可作为 Official DG-KAN > MLP ready 证据。

- 关键证据链：S0 防火墙与单元测试先约束实现语义；C1/C2/C4 检查 adaptive controller 是否早于 washout 介入；C3 检查 pairwise geometry 是否保留 antisymmetric margin；C5/C6 分别检验 KAN basis 与低维 manifold 是否真正承载 source；Part D 只在 gate 通过后允许 task proof。

- 若 final route 被 efficiency 或 basis gate 阻塞，下一步应沿 artifact 中最大 component timing 或 basis coverage blocker 修复；不得用 auxiliary/task readback 绕过 mechanism gate。
