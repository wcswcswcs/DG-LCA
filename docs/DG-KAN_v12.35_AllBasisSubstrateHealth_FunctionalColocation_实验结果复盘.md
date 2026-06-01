# DG-KAN v12.35 AllBasisSubstrateHealth FunctionalColocation 实验结果复盘

生成时间：2026-05-27（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 diagnostic/near-pass/skipped row 写成 promotion。

## 1. 计划理解

v12.35 的目标不是继续扩 B-RAT/B-FOU/M-J token 网格，而是重审：

```text
1. 每个 active basis 是否先成为 substrate-health pass。
2. functional 失败是否来自缺少 basis response dictionary / channel co-location。
3. Non-RAT lifetime 打开后是否仍保持 task/AUC/LineC health。
4. MLP generic functional 是否仍为 monitor-only no-go。
```

硬约束：

```text
1. strict FC-PureKAN / classic no-BSpline active basis。
2. no teacher / distillation / loss modification / sampler / class weight / dataset-name branch。
3. 不使用 label-informed initialization。
4. functional direction 不使用 label、CE vector、permuted-label CE、validation/test、future outcome、query batch 或 LineC hard target。
5. CE/NLL/ECE/CEp99 只能作为审计和坏化约束。
6. blocker 后必须执行计划 fallback；smoke/diagnostic/skipped row 不能写成 promotion。
```

## 2. 当前状态

实验正在执行中。本文件后续只记录已经生成 artifact 的真实结果。

## 3. 本轮代码修改

修改/新增文件：

```text
dgkan/diagnostics/basis_workspace.py
experiments/run_v1231_basis_kernel_workspace.py
experiments/run_v1231_rational_auc_hardening.py
experiments/run_v1235_response_dictionary.py
experiments/run_v1235_finalize_substrate_health_colocation.py
```

已完成内容：

```text
1. 新增 V1235_BASIS_CANDIDATES，在 v12.34.2 active substrate map 基础上加入 D-RAT40、D-CHE20、D-FOU20、D-RBF17、D-WAV16。
2. 新增 V1235_WORKSPACE_FIELDS、workspace_gate_v1235、workspace_strong_gate_v1235。
3. workspace/AUC runner 支持 --candidate-registry v1235。
4. 新增 response dictionary runner：从真实 v12.34.2 P3/LineC artifacts 生成 v1235_basis_response_dictionary / controls / linec / task_tail / colocation summary。
5. 新增 v12.35 finalizer：按 v12.35 Gate S/H 重新审计 substrate-health，生成 route、required manifest、provenance、Non-RAT task-health repair、MLP closure monitor 与 figures。
```

审计说明：

```text
1. 新增 D-RAT40/D-CHE20/D-FOU20/D-RBF17/D-WAV16 均映射到已有 PrimitiveSpec/内核路径，没有声明新的 exact fused kernel 成功。
2. response dictionary 只读取既有 loss-agnostic P3 probe artifact、LineC audit 与 telemetry，不用 label/CE 构造方向。
3. AdamWParallelDirection 仅作为 non-promotable control，不进入 promotion。
4. v12.35 finalizer 不把 v12.34.2 P3 伪装成新训练成功；相关行标记为 reevaluated / recomputed_from_prior_probe。
```

## 4. Provenance / smoke

已执行：

```text
py_compile pass
```

环境说明：

```text
默认 python 环境缺少 torch/numpy，已切换到 conda env kan。
torch = 2.11.0+cu128
numpy = 2.4.4
cuda_available = True
device_count = 4
```

Registry smoke：

```text
basis_candidates = 49
families = D-CHE,D-FOU,D-RAT,D-RBF,D-WAV
new_candidates = D-RAT40,D-CHE20,D-FOU20,D-RBF17,D-WAV16
```

Runner smoke：

```text
workspace smoke rows = 2
workspace_gate_pass_rows = 1
workspace_strong_gate_pass_rows = 1
hardening_executed_rows = 1
AUC smoke candidate_rows = 2
AUC smoke summary_rows = 2
AUC smoke linec_rows = 3
AUC smoke near/official pass = 0
```

解释：smoke 只证明 v1235 registry / workspace / AUC runner 入口可运行，不作为 official evidence。

## 5. Line S substrate-health map v2

v12.35 finalizer 使用 v12.35 Gate S/H 对 v12.34.2 official all-basis substrate artifact 重新审计。输入为真实 artifact：

```text
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342/v12342_family_substrate_summary.csv
```

总体结果：

```text
basis_substrate_health_rows = 44
substrate_health_gate_pass_count = 11
substrate_health_near_pass_count = 16
healthy_base_gate_pass_count = 0
nonrat_substrate_health_pass_count = 0
```

按 family：

| family | rows | substrate health pass | near pass | healthy base pass | best mean delta | best worst delta | best LineC pass rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| D-RAT | 16 | 11 | 16 | 0 | 0.01318359375 | -0.01171875 | 0.9259259259259259 |
| D-CHE | 8 | 0 | 0 | 0 | nan | nan | 0.0 |
| D-FOU | 8 | 0 | 0 | 0 | -0.314453125 | -0.50390625 | 0.0 |
| D-RBF | 6 | 0 | 0 | 0 | -0.4535590277777778 | -0.6796875 | 0.5185185185185185 |
| D-WAV | 6 | 0 | 0 | 0 | nan | nan | 0.0 |

Substrate pass candidates 全部来自 Rational：

```text
D-RAT25,D-RAT26,D-RAT27,D-RAT28,D-RAT29,D-RAT30,D-RAT31,D-RAT35,D-RAT36,D-RAT37,D-RAT39
```

解释：

```text
1. v12.35 比 v12.34.2 更严格地把 workspace + task-health + AUC + LineC 合成 Gate S。
2. 结果仍是 Rational-only substrate；Non-RAT 没有任何 substrate-health pass。
3. healthy base 仍为 0，因此不能进入 official promotion。
```

## 6. Line Q basis response dictionary

执行对象：

```text
source artifacts =
  v12342_basis_functional_p3.csv
  v12342_basis_functional_nonrat_foreachoff_p3.csv
  v12342_basis_functional_linec.csv
  v12342_basis_functional_nonrat_foreachoff_linec.csv
```

结果：

```text
basis_response_dictionary_rows = 270
basis_response_dictionary_executed_rows = 195
response_colocation_summary_rows = 30
response_dictionary_pass_count = 0
p3_gate_recomputed_pass_rows = 0
```

按 family 的 response co-location：

| family | rows | response pass | max CouplingR2 delta | min NoiseSignalLeak delta | min RealSignalReservoirRatio delta | max mean control gap |
|---|---:|---:|---:|---:|---:|---:|
| D-RAT | 135 | 0 | 0.0015274248668856183 | -0.0007153823971748352 | -0.0015760064125061035 | 0.0 |
| D-FOU | 135 | 0 | 0.0000226801741939342 | -0.000036442031462987266 | -0.00007547934850056966 | 0.0 |

最接近但不能 pass 的 rows：

| family | base | functional | dataset | seed | source_vs_best | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | CEp99 delta | pass |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| D-RAT | D-RAT34 | B-RAT3 | KMNIST | 0 | 0.0 | 0.0015274248668856183 | 0.0003024737040201823 | -0.0003036955992380778 | -0.0022988319396972656 | 0 |
| D-RAT | D-RAT26 | B-RAT3 | KMNIST | 0 | 0.0 | 0.0015274248668856183 | 0.0003024737040201823 | -0.0003036955992380778 | -0.0022988319396972656 | 0 |
| D-RAT | D-RAT28 | B-RAT2 | KMNIST | 0 | 0.0 | 0.0012133150420687855 | 0.0005535384019215902 | -0.0005321701367696127 | -0.0025086402893066406 | 0 |

解释：

```text
1. CEp99 在部分 rows 上不坏，control_gap 也不是负数。
2. 但 CouplingR2 delta 最高只有约 0.00153，远低于 response dictionary gate 的 0.01。
3. NoiseSignalLeak 也没有稳定下降，最接近 Rational rows 甚至为正。
4. 因此 Line B 不能构造 official channel co-location repair；不能 promotion。
```

## 7. Line B basis-functional P3/P4 复核

v12.35 没有把 v12.34.2 的 P3 当作新训练成功，而是重算 v12.35 P3 gate：

```text
basis_functional_p3_rows = 270
basis_functional_p3_pass_count = 0
basis_functional_p4_rows = 195
basis_functional_p4_pass_count = 0
```

解释：

```text
response dictionary 没有 pass，因此没有合法基础触发新的 channel co-location functional construction。
既有 B-RAT/B-FOU probes 复核后 P3/P4 仍为 0。
```

## 8. Line N Non-RAT task-health substrate repair

Non-RAT task-health repair 复核 rows：

```text
nonrat_task_health_repair_rows = 40
nonrat_task_health_repair_pass_count = 0
```

按 family：

| family | rows | pass | best mean delta | best worst delta | min AUC time ratio | max LineC pass rate |
|---|---:|---:|---:|---:|---:|---:|
| D-CHE | 11 | 0 | nan | nan | nan | 0.0 |
| D-FOU | 13 | 0 | -0.314453125 | -0.50390625 | 4.303501406591466 | 0.0 |
| D-RBF | 8 | 0 | -0.4535590277777778 | -0.6796875 | 7.525988321545862 | 0.5185185185185185 |
| D-WAV | 8 | 0 | -0.546875 | -0.671875 | 10.238309934885777 | 0.0 |

解释：

```text
1. foreach-off/lifetime-open 是真实系统进展，但没有形成 task-health substrate。
2. Fourier/RBF/Wavelet 的 task/AUC 坏化仍很大；Chebyshev 仍缺少有效 task-health row。
3. route_detail 中保留 R3-NonRATLifetimeTaskHealthSplit。
```

## 9. Line M MLP closure monitor

结果：

| candidate | rows | pass | mean source_vs_noop | mean source_vs_control | mean CouplingR2 delta |
|---|---:|---:|---:|---:|---:|
| M-J1-CloneProbeCovarianceTransportV2 | 27 | 0 | 0.0 | -0.0014467592592592592 | -0.00030882551828906253 |
| M-J2-UnlabeledOptimizerObservableTransportV2 | 27 | 0 | 0.00043402777777777775 | -0.0010127314814814814 | -0.00009409804848446465 |
| M-J3-ArchitectureNeutralSNRTransportV2 | 27 | 0 | 0.00014467592592592592 | -0.0013020833333333333 | 0.00041898112804797404 |

结论：

```text
mlp_functional_closure_rows = 81
mlp_functional_mj_v3_pass_rows = 0
MLPFunctionalNoGo_CurrentLossAgnosticObservableFamily_v3 = 1
```

## 10. Final route

最终 route：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R4-RationalOnlySubstrate also applies; R3-NonRATLifetimeTaskHealthSplit also applies; R5-MLPFunctionalNoGo also applies; ResponseDictionaryNoGo
minimum_success = S1-SubstrateHealthPass
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 5
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_substrate_health_rows = 44
substrate_health_gate_pass_count = 11
substrate_health_near_pass_count = 16
healthy_base_gate_pass_count = 0
nonrat_substrate_health_pass_count = 0
family_telemetry_rows = 900
basis_response_dictionary_rows = 270
basis_response_dictionary_executed_rows = 195
response_dictionary_pass_count = 0
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
nonrat_task_health_repair_pass_count = 0
mlp_functional_mj_v3_pass_rows = 0
provenance_violation_count = 0
code_review_surface_incomplete_count = 0
code_review_packet_entries = 35
code_review_packet_sha256 = 以最终 v1235_route_decision.json 为准
```

## 11. Required artifacts

主要产物：

```text
results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_route_decision.json
results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_required_artifact_manifest.csv
results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_progress_by_line.csv
results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_code_provenance_audit.csv
results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_basis_substrate_health.csv
results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_family_substrate_summary.csv
results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_family_telemetry.csv
results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_basis_response_dictionary.csv
results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_response_colocation_summary.csv
results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_basis_functional_p3.csv
results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_basis_functional_p4.csv
results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_nonrat_task_health_repair.csv
results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_mlp_functional_closure.csv
results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_failure_table.csv
results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_next_hypothesis_queue.md
results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_code_review_packet.zip
```

Required manifest：

```text
missing = 0 / 33
```

## 12. 最终科学结论

v12.35 没有达成 S5，也没有达成 S2/S3/S4。

已闭合事实：

```text
1. v12.35 Gate S 下仍只有 Rational 形成 substrate-health pass：11/44。
2. healthy base gate 仍为 0，因此不能 official promotion。
3. Non-RAT 没有 substrate-health pass；foreach-off/lifetime-open 后 task/AUC/LineC 仍崩坏。
4. response dictionary 已建立，但 270 rows 中 0 pass；核心 blocker 是 CouplingR2 gain 太小且 NoiseSignalLeak 不下降。
5. 既有 basis-specific P3/P4 复核后仍为 0 pass。
6. MLP M-J closure monitor 仍为 0 pass，当前 MLP loss-agnostic observable family v3 no-go。
7. Provenance audit 通过，implementation readback 完整，required artifacts 缺失为 0。
```

新增 no-go boundary：

```text
1. v12.35 回答了计划中的核心问题：当前缺的是 basis-channel co-location value source，而不只是更多 substrate alias。
2. Rational-only substrate 仍存在，但 response dictionary 不支持构造 official functional repair。
3. Non-RAT 的下一步不能直接 functional P3，必须先解决 task-health substrate。
4. 继续扩 B-RAT/B-FOU/M-J 同类 token 已经是低价值网格搜索，应进入机制级 channel co-location 或 Non-RAT task-health path。
```

最终合法状态：

```text
R2-SubstrateButNoFunctionalRepair
minimum_success = S1-SubstrateHealthPass
```

当前已经按计划执行 substrate-health map v2、response dictionary、basis-functional P3/P4 复核、Non-RAT task-health repair 审计、MLP closure monitor、finalizer/provenance/required artifact 审计。`final_stop_allowed=1`、`fallback_all_executed=1`、`required_artifact_missing_count=0` 均成立。我现在不确定继续在同一 B-RAT/B-FOU/M-J token family 上排列局部变体能形成有效机制；继续执行会变成低价值网格搜索。

## 13. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v12.35 是否达成目标，若未达成则继续。再次读取最终 `v1235_route_decision.json` 与 required manifest 后，结论如下：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R4-RationalOnlySubstrate also applies; R3-NonRATLifetimeTaskHealthSplit also applies; R5-MLPFunctionalNoGo also applies; ResponseDictionaryNoGo
minimum_success = S1-SubstrateHealthPass
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 5
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_substrate_health_rows = 44
substrate_health_gate_pass_count = 11
substrate_health_near_pass_count = 16
healthy_base_gate_pass_count = 0
nonrat_substrate_health_pass_count = 0
response_dictionary_pass_count = 0
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
nonrat_task_health_repair_pass_count = 0
mlp_functional_mj_v3_pass_rows = 0
provenance_violation_count = 0
code_review_surface_incomplete_count = 0
manifest_rows = 33
missing_rows = 0
```

最终判断仍是：

```text
v12.35 没有达成 S5；
只达到 S1-SubstrateHealthPass；
没有达成 S2/S3/S4，即 response dictionary、basis-specific P3/P4 仍为 0；
Non-RAT task-health repair 仍为 0 pass；
MLP M-J v3 closure 仍为 no-go；
不允许 promotion；
合法 route 仍是 R2-SubstrateButNoFunctionalRepair；
允许 final stop。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改代码或 gate。原因是计划文件要求的 blocker fallback 已执行完：P3/CouplingR2 blocker 后已建立 response dictionary；Non-RAT lifetime/task collapse 后已做 task-health repair audit；MLP closure fail 已标记 no-go 并降为 monitor。当前 `final_stop_allowed=1` 且我已经不确定继续在同一 B-RAT/B-FOU/M-J token family 上排列局部变体能形成有效机制；继续执行会变成低价值网格搜索。

## 14. 用户再次追问后的 stop-contract 复核 2

用户再次要求确认 v12.35 是否达成目标，若未达成则继续。再次读取最终 `v1235_route_decision.json` 与 required manifest 后，结论没有变化：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R4-RationalOnlySubstrate also applies; R3-NonRATLifetimeTaskHealthSplit also applies; R5-MLPFunctionalNoGo also applies; ResponseDictionaryNoGo
minimum_success = S1-SubstrateHealthPass
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 5
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_substrate_health_rows = 44
substrate_health_gate_pass_count = 11
substrate_health_near_pass_count = 16
healthy_base_gate_pass_count = 0
nonrat_substrate_health_pass_count = 0
response_dictionary_pass_count = 0
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
nonrat_task_health_repair_pass_count = 0
mlp_functional_mj_v3_pass_rows = 0
provenance_violation_count = 0
code_review_surface_incomplete_count = 0
manifest_rows = 33
missing_rows = 0
```

最终判断仍是：

```text
v12.35 没有达成 S5；
只达到 S1-SubstrateHealthPass；
没有达成 S2/S3/S4；
response dictionary、basis-specific P3/P4、Non-RAT task-health repair、MLP M-J v3 closure 均为 0 pass；
不允许 promotion；
合法 route 仍是 R2-SubstrateButNoFunctionalRepair；
允许 final stop。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改代码或 gate。原因仍是：v12.35 计划中的 blocker fallback 已全部执行，且 `final_stop_allowed=1`、`fallback_all_executed=1`、`required_artifact_missing_count=0` 均成立。我已经不确定继续在同一 B-RAT/B-FOU/M-J token family 上扩局部变体能形成有效机制；继续执行会变成低价值网格搜索。

## 15. 用户再次追问后的 stop-contract 复核 3

用户再次要求确认 v12.35 是否达成目标，若未达成则继续。再次读取最终 `v1235_route_decision.json` 与 required manifest 后，结论仍未变化：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R4-RationalOnlySubstrate also applies; R3-NonRATLifetimeTaskHealthSplit also applies; R5-MLPFunctionalNoGo also applies; ResponseDictionaryNoGo
minimum_success = S1-SubstrateHealthPass
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 5
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_substrate_health_rows = 44
substrate_health_gate_pass_count = 11
substrate_health_near_pass_count = 16
healthy_base_gate_pass_count = 0
nonrat_substrate_health_pass_count = 0
basis_response_dictionary_rows = 270
basis_response_dictionary_executed_rows = 195
response_dictionary_pass_count = 0
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
nonrat_task_health_repair_pass_count = 0
mlp_functional_mj_v3_pass_rows = 0
provenance_violation_count = 0
code_review_surface_incomplete_count = 0
manifest_rows = 33
missing_rows = 0
```

计划 failure-handling 对照：

```text
1. P3/CouplingR2 blocker 后已建立 response dictionary；结果仍 0 pass。
2. Non-RAT lifetime/task collapse 后已执行 task-health repair audit；结果仍 0 pass。
3. MLP closure fail 后已标记 MLPFunctionalNoGo v3，不继续 M-J grid。
4. required artifacts 与 implementation readback 均完整。
```

最终判断仍是：

```text
v12.35 没有达成 S5；
只达到 S1-SubstrateHealthPass；
没有达成 S2/S3/S4；
不允许 promotion；
合法 route 仍是 R2-SubstrateButNoFunctionalRepair；
允许 final stop。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改代码或 gate。原因仍是：计划内 fallback 已全部执行，当前我已经不确定继续在同一 B-RAT/B-FOU/M-J token family 上排列局部变体能形成有效机制；继续执行会变成低价值网格搜索。

## 16. 用户再次追问后的 stop-contract 复核 4

用户再次要求确认 v12.35 是否达成目标，若未达成则继续。本次再次读取最终 `v1235_route_decision.json`，结论仍未变化：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R4-RationalOnlySubstrate also applies; R3-NonRATLifetimeTaskHealthSplit also applies; R5-MLPFunctionalNoGo also applies; ResponseDictionaryNoGo
minimum_success = S1-SubstrateHealthPass
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 5
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_substrate_health_rows = 44
substrate_health_gate_pass_count = 11
substrate_health_near_pass_count = 16
healthy_base_gate_pass_count = 0
nonrat_substrate_health_pass_count = 0
basis_response_dictionary_rows = 270
basis_response_dictionary_executed_rows = 195
response_dictionary_pass_count = 0
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
nonrat_task_health_repair_rows = 40
nonrat_task_health_repair_pass_count = 0
mlp_functional_mj_v3_pass_rows = 0
mlp_functional_no_go_current_family_v3 = 1
provenance_violation_count = 0
code_review_surface_incomplete_count = 0
code_review_packet_entries = 35
```

最终判断仍是：

```text
v12.35 没有达成 S5；
只达到 S1-SubstrateHealthPass；
没有达成 S2/S3/S4；
response dictionary、basis-specific P3/P4、Non-RAT task-health repair、MLP M-J v3 closure 均为 0 pass；
不允许 promotion；
合法 route 仍是 R2-SubstrateButNoFunctionalRepair；
允许 final stop。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改代码或 gate。原因仍是：计划内 SubstrateHealth、response dictionary、Non-RAT task-health fallback、basis-specific P3/P4、MLP closure、required artifact 与 code review surface 审计都已执行，且 `final_stop_allowed=1`、`fallback_all_executed=1`、`required_artifact_missing_count=0` 均成立。我已经不确定继续在同一 B-RAT/B-FOU/M-J token family 上扩局部变体能形成有效机制；继续执行会变成低价值网格搜索。
