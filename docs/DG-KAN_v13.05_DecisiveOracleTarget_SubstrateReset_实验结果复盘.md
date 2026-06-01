# DG-KAN v13.5 DecisiveOracleTarget SubstrateReset 实验结果复盘

生成时间：2026-05-28（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 oracle diagnostic / smoke / fallback 写成 promotion。

## 1. 计划理解

v13.5 是 decisive oracle target / legal observability / substrate reset。它不是继续修 v13.4 的 O7/O8 target，也不是回到 v13.2/v13.3 的 BN/BM metric grid，而是要回答：

```text
1. 当前 basis-channel substrate 是否已有 S1C。
2. 是否存在 oracle upper-bound target 能稳定带来 future advantage。
3. 如果 oracle target 存在，legal precommit feature 是否能提前看见。
4. Non-RAT 是否能形成可进入 functional proof 的 S1C vertical slice。
5. MLP analog 是否解释同一 oracle 机制。
```

硬约束：

```text
1. strict FC-PureKAN / no active B-spline budget。
2. 不使用 label-informed initialization。
3. 不使用 teacher / distillation / sampler / class weight / dataset-name branch。
4. O-OR1/O-OR2/O-OR4 使用 future outcome 构造 oracle diagnostic target，因此只能作为 upper-bound diagnostic。
5. O-OR3 使用 label/LineC-release target，也只能作为 diagnostic；不能作为 legal direction。
6. CEp99/NLL/ECE/LineC hard target 只能作为 audit/gate。
7. 不允许把 S1C / actuation pass / oracle diagnostic row 写成 promotion。
```

## 2. 本轮代码修改

新增文件：

```text
experiments/run_v135_decisive_oracle_target_substrate_reset.py
```

主要修改：

```text
1. 从 v13.4 runner 派生，复用 live basis-channel Z、J_theta->Z projection、future probe。
2. 新增 O-OR1/O-OR2/O-OR3/O-OR4 oracle targets。
3. 新增 P7/P8 projection fallback 语义：
   - P7-ReachableSubspaceProjection
   - P8-TrustRegionReachableProjection
4. 新增 legal precommit visibility audit：
   current logits / unlabeled basis telemetry / jacobian sketch only。
5. 新增 sequential two-event oracle diagnostic。
6. 新增 MLP oracle closure controls。
7. 新增 v13.5-specific Non-RAT vertical slices：
   FOU-S1C1/2/3 与 CHE-S1C1/2/3。
8. build_substrate_status() 使用 oracle projection gate 判定 Rational S1C，不把 current baseline row 当成 S1C。
9. 输出 required manifest、forbidden audit、target provenance、writeback trace、failure table、no-go boundary、next hypothesis queue、figures、code packet。
```

合法性说明：

```text
1. oracle rows promotion_allowed=0。
2. projection 写回真实 basis parameters；不是 readout-feature proxy。
3. writeback trace 记录 full_basis_param_update_rows=56。
4. readout_feature_proxy_only=0。
5. feature_table_proxy_only=0。
6. legal precommit audit 不使用 future outcome / validation / test / query batch 作为 feature。
7. 不改变 promotion gate。
```

## 3. Smoke

执行规模：

```text
task = X1
seed = 0
oracle = O-OR1-FutureTrainingDeltaZ
operator_batch_size = 8
max_channel_dim = 8
checkpoint_steps = 4
oracle_future_steps = 4
future_steps = 4
```

结果：

```text
route = R1-NoS1C
minimum_success = S1-EfficientSubstrate
required_artifact_missing_count = 0
oracle_rows = 1
oracle_pass_rows = 0
oracle_task_family_pass_count = 0
full_basis_param_update_rows = 1
provenance_violation_count = 0
```

解释：

```text
smoke 只证明 runner 和 artifact surface 可执行。
该 smoke 的 future-training DeltaZ norm = 0，不能写成 oracle success。
```

## 4. Official v13.5 结果

执行规模：

```text
synthetic_tasks = X1..X7
seeds = 0,1
oracle_types = O-OR1,O-OR2,O-OR3,O-OR4
projection_solver = P4-ConjugateGradientJtJProjection
operator_batch_size = 16
max_channel_dim = 16
max_jacobian_rows = 256
oracle_delta_z_scale = 0.01
checkpoint_steps = 20
oracle_future_steps = 20
future_steps = 30
```

最终 route：

```text
route = R4-OracleTargetNoUpperBound
minimum_success = S1C-ChannelControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
substrate_s1_count = 11
substrate_s1c_count = 1
nonrat_s1c_count = 0
oracle_rows = 56
oracle_pass_rows = 0
oracle_task_family_pass_count = 0
oracle_upper_bound_pass = 0
precommit_visibility_rows = 1
precommit_visibility_pass = 0
sequential_oracle_rows = 7
sequential_oracle_pass_count = 0
mlp_oracle_rows = 3
mlp_oracle_pass_count = 0
full_basis_param_update_rows = 56
provenance_violation_count = 0
forbidden_information_violation_count = 0
```

最接近 oracle row：

| task | seed | oracle | source_vs_best | actuation_error | actuation_cosine | NoiseSignalLeak delta | Reservoir delta | CEp99 delta | synthetic |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| X4 | 0 | O-OR2-BestControlResidual | 0.0017166185519627633 | 0.0 | 0.0 | -0.012942790985107422 | -1.6300039291381836 | -4.755213737487793 | 0 |
| X3 | 0 | O-OR4-TaskFamilySpecific | 0.001645626181047341 | 0.0 | 0.0 | 0.04412287473678589 | -0.6566669940948486 | -1.291684627532959 | 0 |
| X4 | 0 | O-OR1-FutureTrainingDeltaZ | 0.0015573118013524553 | 0.0 | 0.0 | -0.008160650730133057 | -1.6219964027404785 | -5.236171722412109 | 0 |

判断：

```text
1. S1C 仍存在，但只说明 channel actuation/projection 可执行。
2. oracle upper-bound 没有打开：oracle_pass_rows=0，task_family_pass=0/7。
3. 最接近 row 的 source_vs_best 仍低于 0.005 gate。
4. 因 R4，legal visibility 即使存在也不能进入 real short-run；本轮实际 precommit visibility 也没有通过。
5. 最终逐行数值以 official_v135/v135_oracle_synthetic_proof.csv 为准。
```

## 5. Legal precommit visibility

artifact：

```text
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_precommit_feature_audit.csv
```

结果：

```text
allowed_features_only = 1
forbidden_feature_count = 0
AUC_success = 0.5
precision_at_k = 0.0
recall_at_k = 0.0
leave_family_out_AUC = 0.5
oracle_positive_rows = 0
precommit_allowed = 0
```

解释：

```text
oracle_positive_rows=0，因此 visibility 没有可学习的 positive target；
这不是 legal observability success。
```

## 6. Sequential oracle diagnostic

结果：

```text
sequential_oracle_rows = 7
sequential_oracle_pass_count = 0
```

最接近 row：

| task | seed | source_vs_best | NoiseSignalLeak delta | Reservoir delta | CEp99 delta | sequential success |
|---|---:|---:|---:|---:|---:|---:|
| X7 | 0 | 0.0038971479048562283 | 0.015684008598327637 | -1.2906699180603027 | -1.6853594779968262 | 0 |
| X2 | 0 | 0.002647459005923773 | 0.018940269947052002 | -2.767468214035034 | -8.912230491638184 | 0 |

解释：

```text
X7 sequential row 仍低于 0.005 source gate，且 NoiseSignalLeak 坏化；
two-event oracle diagnostic 没有证明 sequential target 是缺失机制。
```

## 7. MLP oracle closure

artifact：

```text
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_mlp_oracle_control.csv
```

结果：

```text
M-OR1 source_vs_best = -0.0010140590571622414
M-OR1 CouplingR2_delta = -2.556486129760742
M-OR1 NoiseSignalLeak_delta = 0.027679622173309326
M-OR1 Reservoir_delta = 0.319486141204834
M-OR1 CEp99_delta = 0.4694552421569824
M-OR1 mlp_oracle_pass = 0
M-OR2 skipped because MLP oracle pass required before legal precommit proxy can be meaningful
M-OR3 random reachable control available = 1
```

解释：

```text
MLP analog 没有解释 v13.5 oracle mechanism；
但 KAN oracle 本身也没有 upper-bound pass，因此不能主张 KAN-specific success。
```

## 8. Non-RAT vertical slice

执行内容：

```text
FOU-S1C1/2/3
CHE-S1C1/2/3
```

结果：

| family | slice | actuation_error | cosine | workspace_incremental_ratio | step_ratio | LineC pass | S1C |
|---|---|---:|---:|---:|---:|---:|---:|
| D-FOU | FOU-S1C1 | 0.058193475008010864 | 0.9986006021499634 | 4.2583056478405314 | 1.4176488760262813 | 0.0 | 0 |
| D-FOU | FOU-S1C2 | 0.05084142088890076 | 0.9990485906600952 | 4.2583056478405314 | 1.4176488760262813 | 0.0 | 0 |
| D-FOU | FOU-S1C3 | 0.06653273850679398 | 0.9981427192687988 | 4.2583056478405314 | 1.4176488760262813 | 0.0 | 0 |
| D-CHE | CHE-S1C1 | 0.2500919997692108 | 0.9684257507324219 | 7.000830564784053 | 7.186207862791935 | 0.0 | 0 |
| D-CHE | CHE-S1C2 | 0.05191929265856743 | 0.9989519715309143 | 7.000830564784053 | 7.186207862791935 | 0.0 | 0 |
| D-CHE | CHE-S1C3 | 0.02824893593788147 | 0.9997985363006592 | 7.000830564784053 | 7.186207862791935 | 0.0 | 0 |

解释：

```text
Non-RAT random actuation gate 可通过，但 workspace / LineC gate fail；
计划要求不能把 actuation-only 写成 S1C，因此 nonrat_s1c_count=0。
```

## 9. Blocker 修复：P8 trust-region / scale fallback

执行：

```text
projection_solver = P8-TrustRegionReachableProjection
oracle_delta_z_scale = 0.03
```

结果：

```text
route = R4-OracleTargetNoUpperBound
oracle_rows = 56
oracle_pass_rows = 0
oracle_task_family_pass_count = 0
oracle_upper_bound_pass = 0
precommit_visibility_pass = 0
sequential_oracle_pass_count = 0
mlp_oracle_pass_count = 0
nonrat_s1c_count = 0
full_basis_param_update_rows = 56
required_artifact_missing_count = 0
```

最接近 P8 row：

```text
task = X7
seed = 0
oracle_type = O-OR3-LineCReleaseOracle
source_vs_best = 0.0044884306696907
synthetic_success = 0
```

判断：

```text
P8 / larger target scale 没有打开 oracle upper-bound。
这排除了“只是 projection trust-region 或 target scale 太小”的直接解释。
```

## 10. Required artifacts

official_v135 required manifest：

```text
manifest_rows = 30
missing_required_rows = 0
```

主要 artifact：

```text
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_route_decision.json
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_code_review_manifest.csv
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_target_provenance_manifest.csv
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_oracle_target_audit.csv
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_precommit_feature_audit.csv
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_projection_writeback_trace.csv
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_forbidden_information_audit.csv
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_substrate_s1c_status.csv
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_nonrat_s1c_vertical_slice.csv
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_current_target_baseline.csv
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_oracle_synthetic_proof.csv
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_oracle_family_summary.csv
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_sequential_oracle_diagnostic.csv
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_mlp_oracle_control.csv
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_failure_table.csv
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_no_go_boundary.md
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_next_hypothesis_queue.md
results/v13_5_decisive_oracle_target_substrate_reset/official_v135/v135_code_review_packet.zip
```

## 11. 最终科学结论

v13.5 没有达成 S2/S3/S5；最终合法 route：

```text
R4-OracleTargetNoUpperBound
minimum_success = S1C-ChannelControllableSubstrate
promotion_allowed = 0
```

已闭合事实：

```text
1. Rational S1C 仍成立：substrate_s1c_count=1。
2. 56 个 oracle rows 全部没有通过 synthetic success。
3. oracle task-family pass = 0/7。
4. legal precommit visibility 没有 positive oracle target，因此 precommit_allowed=0。
5. sequential two-event oracle diagnostic 没有通过。
6. P8 trust-region / larger scale fallback 没有打开 upper-bound。
7. Non-RAT vertical slice 只有 actuation pass，没有 workspace/LineC/S1C pass。
8. MLP oracle pass = 0。
9. required artifacts 缺失为 0，provenance / forbidden audit 为 0。
10. full_basis_param_update_rows=56，不是 readout-feature proxy。
```

no-go boundary：

```text
1. 当前失败已经不是 O/P actuation fail；v13.4/v13.5 均证明 Rational S1C 可执行。
2. 当前失败是 oracle value target upper-bound fail：即使用 future/label/LineC diagnostic oracle，也没有稳定打过 controls。
3. Legal visibility 不能先于 oracle upper-bound 成立；本轮 oracle_positive_rows=0。
4. Non-RAT 仍卡在 workspace/LineC substrate gate，不能作为 real short-run substrate。
5. 继续 O7/O8/O9、BN/BM、target scale / trust-region 小网格已经不符合 v13.5 计划；下一步需要 substrate/base architecture reset 或重新定义 synthetic proof 的可价值目标。
```

最终判断：

```text
v13.5 未达成目标；
不允许 promotion；
不允许 real short-run；
允许 final stop，原因是计划内 oracle upper-bound / legal visibility / Non-RAT S1C / MLP closure 均已闭合为 no-go。
```

## 12. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v13.5 是否达成目标，若未达成则继续。本次重新读取 v13.5 计划 stop rule、official route、no-go boundary 与 next hypothesis queue。

最终 artifact：

```text
route = R4-OracleTargetNoUpperBound
minimum_success = S1C-ChannelControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
substrate_s1_count = 11
substrate_s1c_count = 1
nonrat_s1c_count = 0
oracle_rows = 56
oracle_pass_rows = 0
oracle_task_family_pass_count = 0
oracle_upper_bound_pass = 0
precommit_visibility_rows = 1
precommit_visibility_pass = 0
sequential_oracle_rows = 7
sequential_oracle_pass_count = 0
mlp_oracle_rows = 3
mlp_oracle_pass_count = 0
full_basis_param_update_rows = 56
provenance_violation_count = 0
forbidden_information_violation_count = 0
readout_feature_proxy_only = 0
feature_table_proxy_only = 0
```

计划 stop rule 对照：

```text
1. 第 8.4 节写明：If oracle fails -> R4-OracleTargetNoUpperBound；recommendation = stop functional on current substrate; return to substrate/base architecture。
2. 第 17.1 节写明：如果 v13.5 没有 oracle upper-bound，禁止继续 O7/O8/O9 token 小修、BM/BN metric 回退、real short-run、LineC hard target direction、CEp99/NLL/ECE direction、readout-feature proxy、frozen feature-table transport。
3. v135_no_go_boundary.md 写明：oracle task-family pass count = 0/7，Non-RAT S1C count = 0，MLP oracle pass count = 0。
4. v135_next_hypothesis_queue.md 写明：If R4, return to substrate/base architecture; do not continue O7/O8/O9 or BM/BN token search。
```

最终判断仍是：

```text
v13.5 没有达成 S2/S3/S5；
只达到 S1C-ChannelControllableSubstrate；
不允许 promotion；
合法 route = R4-OracleTargetNoUpperBound；
允许 final stop。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改 gate。原因不是轻易放弃，而是计划内 no-go 已闭合：oracle upper-bound fail 后，计划明确禁止继续当前 functional target 小修。下一步需要新的 substrate/base architecture reset 计划；在当前 v13.5 runner 内继续排列 O-token、BM/BN、scale/trust-region 变体会违反计划并增加把 diagnostic 编造成 success 的风险。

## 13. 用户再次追问后的 substrate/base architecture 边界复核

用户再次要求“未达成则继续”。本次额外检查 v13.5 与当前打开的 v12.36 substrate-health 计划之间是否存在可直接执行的 plan-authorized fallback。

复核事实：

```text
1. v13.5 第 7.3 节要求的 Non-RAT S1C vertical slice 已覆盖：
   FOU-S1C1/2/3 与 CHE-S1C1/2/3。
2. v13.5 第 8.4 节写明 oracle fails -> R4-OracleTargetNoUpperBound。
3. v13.5 第 17.1 节写明没有 oracle upper-bound 时禁止继续：
   O7/O8/O9 token 小修、BM/BN metric 回退、real short-run、LineC hard target direction、
   CEp99/NLL/ECE direction、readout-feature proxy、frozen feature-table transport。
4. v13.5 第 17.2 节写明 oracle upper-bound fail 后：
   return to substrate/base architecture。
5. v12.36 确实包含新的 substrate-health redesign 候选，例如 D-CHE23/24/25、D-FOU23/24/25、D-RBF20/21/22。
```

边界判断：

```text
v12.36 里的候选属于新的 substrate/base architecture redesign；
它们不是 v13.5 oracle-target runner 的 plan-internal fallback。
如果在当前 v13.5 代码路径内硬塞这些新 substrate，并继续沿用 v13.5 oracle route，
会有把 substrate diagnostic / proxy 误写成 oracle target success 的风险。
```

最终判断仍是：

```text
v13.5 没有达成目标；
合法 route = R4-OracleTargetNoUpperBound；
promotion_allowed = 0；
允许 final stop。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改 gate。原因是我现在不确定如何在当前 v13.5 代码路径内安全实现新的 substrate/base architecture reset，同时保证不混淆 v13.5 oracle upper-bound 结论。合法下一步应是新开 substrate/base architecture reset 计划，而不是继续扩 v13.5 小修。

## 14. 用户再次追问后的可执行 runner 复核

用户再次要求“未达成则继续”。本次进一步检查 v12.36 的 substrate/base architecture 候选是否已有可直接接入 v13.5 的 runner。

复核事实：

```text
1. v12.36 计划列出了新的 Non-RAT task-health substrate candidates：
   D-CHE23/24/25、D-FOU23/24/25、D-RBF20/21/22 等。
2. 这些候选在当前检索中主要存在于计划文档；未发现已经实现成可直接接入 v13.5 oracle route 的 runner。
3. v1235 finalizer 只有既有 Non-RAT task-health repair audit，不是 v13.5 oracle-target fallback。
4. v13.5 runner 已经完成计划内 FOU-S1C1/2/3 与 CHE-S1C1/2/3 vertical slice。
```

最终判断仍是：

```text
v13.5 没有达成目标；
合法 route = R4-OracleTargetNoUpperBound；
promotion_allowed = 0；
允许 final stop。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改 gate。原因是我现在不确定如何在当前 v13.5 代码路径内安全实现 v12.36 substrate/base architecture reset，同时保证不混淆 v13.5 的 oracle upper-bound 结论。继续推进需要新的 substrate/base architecture reset 计划/runner，而不是在 v13.5 内补一个未经计划定义的 fallback。

## 15. 用户再次追问后的最终 stop-contract 复核

用户再次要求“未达成则继续”。本次再次对照 v13.5 route 与计划 stop rule。

最终 artifact：

```text
route = R4-OracleTargetNoUpperBound
minimum_success = S1C-ChannelControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
oracle_upper_bound_pass = 0
oracle_task_family_pass_count = 0
precommit_visibility_pass = 0
sequential_oracle_pass_count = 0
nonrat_s1c_count = 0
mlp_oracle_pass_count = 0
full_basis_param_update_rows = 56
readout_feature_proxy_only = 0
feature_table_proxy_only = 0
```

计划 stop rule：

```text
1. v13.5 第 8.4 节：oracle fails -> R4-OracleTargetNoUpperBound，并 stop functional on current substrate。
2. v13.5 第 17.1 节：没有 oracle upper-bound 后，禁止继续 O7/O8/O9、BM/BN、response dictionary vN、real short-run、LineC/CEp99/NLL/ECE direction、readout-feature proxy、frozen feature-table transport。
3. v13.5 第 17.2 节：oracle upper-bound fail 后 return to substrate/base architecture。
```

最终判断仍是：

```text
v13.5 没有达成目标；
合法 route = R4-OracleTargetNoUpperBound；
promotion_allowed = 0；
允许 final stop。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改 gate。原因不是轻易放弃，而是当前计划明确禁止继续 v13.5 内部小修；我也已经不确定如何在当前 v13.5 代码路径内安全实现新的 substrate/base architecture reset，同时保证不混淆 oracle upper-bound 结论。下一步需要新的 substrate/base architecture reset 计划/runner。
