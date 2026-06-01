# DG-KAN v13.4 OperatorLevelBasisChannelFunctional 实验结果复盘

生成时间：2026-05-27（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 smoke/diagnostic/skipped row 写成 promotion。

## 1. 计划理解

v13.4 是 strategic reset：不再继续参数空间 diagonal / low-rank / block metric 小修，而是验证：

```text
operator-level basis-channel target DeltaZ
-> true basis-parameter projection DeltaTheta
-> actuation fidelity
-> synthetic task-family proof
```

硬约束：

```text
1. strict FC-PureKAN / no active B-spline budget。
2. 不使用 teacher / distillation / loss modification / sampler / class weight / dataset-name branch。
3. 不使用 label-informed initialization。
4. functional direction 可接收 generic loss-interface cotangent，但不能 hardcode CE-specific formula。
5. functional direction 不使用 validation/test/future outcome/query batch。
6. CEp99/NLL/ECE/LineC hard target 只能作为 audit/gate，不能生成方向。
7. S1C / O / P 未通过时不能进入 synthetic proof 或 real short-run promotion。
```

本轮新增 runner：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
```

审计边界：

```text
1. PrimitiveKAN 的 Z 定义为 live layer2_basis(hidden)。
2. GroupedRational 的 Z 定义为 live hidden activation。
3. MLP analog 的 Z 定义为 hidden activation。
4. J_theta->Z 用真实参数 autograd Jacobian sketch。
5. 参数 writeback 实际执行并测量 actual DeltaZ，然后回滚以隔离候选。
6. 当前实现不使用 frozen_readout_features 作为 operator target。
```

## 2. 初始 smoke blocker 与修复

首次 smoke 在 `J_theta->Z` autograd sketch 阶段失败：

```text
RuntimeError: One of the differentiated Tensors does not require grad
```

原因：

```text
v12.35/v13.3 继承来的部分 D-RAT candidate 中，basis-channel 所需参数可能处于 frozen 状态；
v13.4 要测的是 operator-level parameter actuation，而不是普通 optimizer trainability；
因此 torch.autograd.grad 不能直接对 requires_grad=False 的 selected params 求导。
```

修复：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  jacobian_theta_to_z() 对 selected basis-channel params 临时 requires_grad_(True)；
  J_theta->Z sketch 完成后恢复原 requires_grad 状态；
  projection/writeback 仍只做真实参数 add_ 和回滚；
  jacobian/projection/writeback CSV 记录 temporarily_enabled_frozen_channel_params，方便审计。
```

同时修复 artifact route 生成顺序：

```text
先写 preliminary route/no-go/figures；
再生成 code packet 和 final manifest；
最终 route 使用 final manifest 的 missing count，避免中间 artifact 尚未写出时误判 R0。
```

审计判断：

```text
1. 这是 operator actuation Jacobian 的可执行性修复。
2. 不引入 validation/test/future/query batch。
3. 不使用 CEp99/NLL/ECE/LineC 生成方向。
4. 不把 frozen readout-feature proxy 写成 basis-channel success。
```

## 3. 修复后 smoke

执行规模：

```text
tasks = X1,X7
seed = 0
selected S1 substrate = 1
operator = O1-LossCotangentChannelNewtonDiag
projection = P1-DiagonalProjection,P3-LowRankWoodburyProjection
loss = CE
```

结果：

```text
route = R1-NoS1CSubstrate
minimum_success = S1-EfficientSubstrate
required_artifact_missing_count = 0
operator_channel_target_rows = 4
operator_gate_pass_count = 0
parameter_projection_rows = 4
parameter_projection_pass_count = 0
full_basis_param_update_rows = 4
synthetic_task_success_count = 0
```

最近 rows：

| task | operator | projection | predicted logit drift | actuation error | cosine | P |
|---|---|---|---:|---:|---:|---:|
| X1 | O1 | P3 | 0.9540233612060547 | 0.9144269824028015 | 0.9990381598472595 | 0 |
| X7 | O1 | P3 | 1.3303909301757812 | 0.8702125549316406 | 0.9991661310195923 | 0 |

解释：

```text
1. smoke 证明 v13.4 runner、真实参数 writeback、actuation measurement 与 required manifest 可运行。
2. O1 target 的 predicted logit drift 超过 0.35，因此 operator gate 未开。
3. projection 对 DeltaZ 的方向 cosine 高，但幅度 actuation error 很大，说明当前参数通道只能执行很小的 DeltaZ。
4. smoke 不允许 promotion。
```

## 4. 初始 official 与 blocker

初始 official 使用 hidden activation channel，完整 X1..X7 / CE,Brier / O1,O2,O7 / P1,P3,P6。

结果：

```text
route = R1-NoS1CSubstrate
minimum_success = S1-EfficientSubstrate
operator_channel_target_rows = 126
operator_gate_pass_count = 0
parameter_projection_pass_count = 0
synthetic_task_success_count = 0
full_basis_param_update_rows = 126
required_artifact_missing_count = 0
```

核心 blocker：

```text
basis_channel_rank_ratio = 0.1137884184718132 .. 0.1527169644832611
delta_z_effective_rank max = 1.9335395097732544
best projection actuation_error = 0.4365822970867157
```

解释：

```text
hidden activation channel 太低秩，O gate 没开；
projection fidelity 最接近也没有过 0.35。
```

## 5. live rational channel / O3-O4 / P4 fallback

修改：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  GroupedRational 的 basis_channel 改成 live rational basis output r = rational(norm_input(x))。
  新增 O3-PopRiskOffdiagChannel。
  新增 O4-TrainProbeCouplingPreservingChannel。
  新增 P4-ConjugateGradientJtJProjection。
```

fallback 结果：

```text
route = R1-NoS1CSubstrate
operator_gate_pass_count = 96 / 392
parameter_projection_pass_count = 0 / 392
basis_channel_rank_ratio = 0.4564659595489502 .. 0.5421897172927856
```

O pass：

| operator | pass |
|---|---:|
| O3-PopRiskOffdiagChannel | 52/56 |
| O4-TrainProbeCouplingPreservingChannel | 44/56 |
| O1/O2/O5/O6/O7 | 0/56 |

解释：

```text
live rational channel 修复了 O target rank/rank-ratio blocker；
但 parameter projection 仍不能执行 target，best actuation_error = 0.8251322507858276。
```

## 6. full-row Jacobian projection fallback

执行 full 256-row `J_theta->Z`：

```text
route = R1-NoS1CSubstrate
operator_gate_pass_count = 72 / 84
parameter_projection_pass_count = 0 / 84
best actuation_error = 0.6389598846435547
best cosine = 0.7697218656539917
```

解释：

```text
full-row projection 有改善，但 target 仍大多不在可执行参数子空间；
P gate 仍未打开。
```

## 7. O7 actuation-aware reachable target fallback

修改：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  O7-BalancedChannelSolve 在 full target rows 可用时，
  将 generic loss-interface DeltaZ 投影到当前 J_theta->Z 可执行子空间，
  再重新计算 O gate 与 P gate。
```

合法性：

```text
1. 使用 train-stream J_theta->Z 和 generic loss-interface DeltaZ。
2. 不读取 validation/test/future/query batch。
3. 不使用 CEp99/NLL/ECE/LineC 生成方向。
4. 仍对真实 basis parameters 做 writeback。
```

结果：

```text
route = R3-SyntheticProofFail
minimum_success = S1C-ChannelControllableSubstrate
operator_gate_pass_count = 42 / 42
parameter_projection_pass_count = 28 / 42
substrate_s1c_count = 1
synthetic_task_success_count = 0
```

解释：

```text
这是 v13.4 的真实进展：首次从 S1 推进到 S1C，证明 operator-level reachable channel target 可被真实参数执行。
```

## 8. Line X synthetic future probe

修改：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  新增 Functional / TaskOnlyAdamW / NoOpMatchedOverhead / RandomMatchedNorm controls。
  对 O/P pass row 执行 future probe。
  写出 source_vs_best、CouplingR2_delta、NoiseSignalLeak_delta、Reservoir_delta、CEp99/NLL/ECE delta。
  task-family gate 按计划改成 >=2 substrates 或 >=2 seeds。
```

两 seed synthetic fallback：

```text
route = R3-SyntheticProofFail
operator_gate_pass_count = 54 / 56
parameter_projection_pass_count = 54 / 56
synthetic_rows = 14
synthetic_task_success_count = 0
source_vs_best max = 0.0023105812220099928
```

最接近 rows：

| task | seed | source_vs_best | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | success |
|---|---:|---:|---:|---:|---:|---:|
| X4 | 0 | 0.0023105812220099928 | 11.007625579833984 | -0.02386528253555298 | -1.197321891784668 | 0 |
| X3 | 1 | 0.00026520338861887177 | 2.741302013397217 | -0.06517153978347778 | -0.5284132957458496 | 0 |
| X6 | 0 | 0.00013844595150508399 | 15.46923828125 | 0.009056508541107178 | -0.8682193756103516 | 0 |

解释：

```text
O/P 已经打开；
但 future source_vs_best 未达到 0.005，部分 row 还存在 NoiseSignalLeak 正值；
因此 synthetic 5/7 不能通过。
```

## 9. synthetic fallback sensitivity

更大 target scale：

```text
delta_z_eta = 0.03
max_update_norm_ratio = 1.00
route = R3-SyntheticProofFail
parameter_projection_pass_count = 50 / 56
synthetic_task_success_count = 0
source_vs_best max = 0.002224695740159814
```

短 horizon：

```text
future_steps = 10
route = R3-SyntheticProofFail
parameter_projection_pass_count = 54 / 56
synthetic_task_success_count = 0
source_vs_best max = 0.0004274057040334796
```

optimizer-state transport：

```text
optimizer_state_transport = 1
optimizer_state_scale = 0.20
route = R3-SyntheticProofFail
parameter_projection_pass_count = 54 / 56
synthetic_task_success_count = 0
source_vs_best max = 0.0010513481007466546
```

判断：

```text
scale / horizon / optimizer-state transport 都没有把 synthetic source gap 推过 0.005。
```

## 10. final official_v134

最终 official artifact：

```text
results/v13_4_operator_level_basis_channel_functional/official_v134
```

最终 route：

```text
route = R3-SyntheticProofFail
minimum_success = S1C-ChannelControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
substrate_s1_count = 11
substrate_s1c_count = 1
operator_channel_target_rows = 56
operator_gate_pass_count = 54
parameter_projection_rows = 56
parameter_projection_pass_count = 54
synthetic_rows = 14
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
nonrat_s1c_count = 0
mlp_analog_pass_count = 0
full_basis_param_update_rows = 56
provenance_violation_count = 0
```

Best projection rows：

| task | seed | solver | loss | actuation error | cosine | actual logit drift | P |
|---|---:|---|---|---:|---:|---:|---:|
| X1 | 0 | P4 | Brier | 0.1050683856010437 | 0.995138943195343 | 1.9088451153947972e-05 | 1 |
| X6 | 0 | P4 | Brier | 0.10709031671285629 | 0.994645357131958 | 1.3215068065619562e-05 | 1 |
| X2 | 0 | P3 | Brier | 0.11729145050048828 | 0.9947513341903687 | 1.0858337191166356e-05 | 1 |
| X7 | 1 | P3 | CE | 0.12505373358726501 | 0.9924115538597107 | 2.2886719307280146e-05 | 1 |

Synthetic blocker：

```text
source_vs_best max = 0.0010664487124175462
source_vs_best min = -0.006387620423195739
```

Non-RAT S1C design rows：

| family | candidate | workspace ratio | step ratio | S1C |
|---|---|---:|---:|---:|
| D-CHE | D-CHE16-DegreeEnergyDampingSubstrate | 7.000830564784053 | 7.186207862791935 | 0 |
| D-FOU | D-FOU14-SincosSharedWorkspace-K2 | 4.2583056478405314 | 1.4176488760262813 | 0 |
| D-RBF | D-RBF13-WidthConditionGuardSubstrate | 3.5456810631229234 | 1.8206010168168194 | 0 |
| D-WAV | D-WAV13-LocalTailCoverageGuardSubstrate | 4.189368770764119 | 1.977352831438976 | 0 |

Required manifest：

```text
manifest_rows = 30
missing_required = 0
code_review_packet_sha256 = 以 results/v13_4_operator_level_basis_channel_functional/official_v134/v134_route_decision.json 为准
```

说明：

```text
code packet 包含本执行日志与复盘日志；
任何后续日志编辑都会改变 zip bytes，最终 route 以 v134_route_decision.json 为准。
```

## 11. 最终科学结论

v13.4 没有达成 S3/S5，但相对 v13.3 有明确进展：

```text
v13.3: R4-DiagonalAndLowRankMetricNoGo，P3=0，S1C 未建立。
v13.4: R3-SyntheticProofFail，S1C=1，O/P actuation pass 大量成立。
```

已闭合事实：

```text
1. live rational basis-channel 比 hidden activation 更符合 operator-level basis channel，并打开 rank/rank-ratio blocker。
2. O7 actuation-aware reachable target 能把 DeltaZ 放入 J_theta->Z 可执行子空间。
3. 真实 basis parameter writeback 可稳定执行：54/56 projection pass。
4. Synthetic future probe 没有通过：0/14 synthetic rows pass，0/7 task family pass。
5. 主要 blocker 已从 actuation fidelity 转为 source_vs_best 不足。
6. Non-RAT S1C 仍为 0，MLP analog 也未 pass。
7. Provenance / forbidden audit 通过，required artifacts 缺失为 0。
```

最终合法状态：

```text
R3-SyntheticProofFail
minimum_success = S1C-ChannelControllableSubstrate
promotion_allowed = 0
```

当前 no-go boundary：

```text
1. 不能把 S1C/O/P actuation pass 写成 synthetic proof 或 promotion。
2. 继续 BM/BN metric 网格与计划冲突，不能回去扩。
3. 继续 real short-run 与 synthetic 5/7 fail 冲突。
4. 下一步应聚焦 operator target 如何带来 future source advantage，或重新设计 Non-RAT/S1C substrate，而不是把 reachable DeltaZ actuation 本身写成 functional success。
```

## 12. 用户再次追问后的 O8 reachable-logit fallback 与 stop-contract 复核

用户再次要求确认 v13.4 是否达成目标，若未达成则继续。本次对照计划 Stop/Go 后，只继续计划允许的 operator-level basis-channel target / parameter projection 方向；不回退 BM/BN metric 网格，不进入 real short-run。

代码修改：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  新增 O8-ReachableLogitCotangentChannel。
  新增 reachable_logit_cotangent_target()。
  O8 使用 train-stream J_theta->Z 和 ridge J_Z->Y，
  将 generic output cotangent 直接投到可执行 J_theta->Z->Y 子空间。
```

合法性说明：

```text
1. O8 不使用 validation/test/future outcome/query batch 生成方向。
2. O8 不使用 CEp99/NLL/ECE/LineC 生成方向。
3. O8 仍写回真实 basis parameters，不是 readout-feature proxy。
4. O8 不改变 promotion gate。
```

O8 fallback 执行结果：

```text
artifact = results/v13_4_operator_level_basis_channel_functional/fallback_o8_reachable_logit_v134
route = R3-SyntheticProofFail
minimum_success = S1C-ChannelControllableSubstrate
operator_gate_pass_count = 46 / 56
parameter_projection_pass_count = 46 / 56
synthetic_rows = 14
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
required_artifact_missing_count = 0
full_basis_param_update_rows = 56
provenance_violation_count = 0
```

最近但不能 pass 的 synthetic rows：

| task | seed | source_vs_best | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | CEp99 delta | success |
|---|---:|---:|---:|---:|---:|---:|---:|
| X3 | 1 | 0.001302997653710218 | 2.74135160446167 | -0.0651710033416748 | -0.5284199714660645 | -1.516082525253296 | 0 |
| X4 | 0 | 0.0009605891775466072 | 11.007604598999023 | -0.023865878582000732 | -1.1973199844360352 | -1.3616313934326172 | 0 |
| X6 | 0 | entered_synthetic_proof=0 | n/a | n/a | n/a | n/a | 0 |

判断：

```text
O8 没有超过 O7。
source_vs_best 仍低于 0.005，synthetic_task_success_count 仍为 0。
因此不能把 O8 写成 synthetic proof 或 promotion。
```

stop-contract 复核：

```text
计划第 14 节规定：
如果 v13.4 仍然不能产生 S1C 或 synthetic 5/7，
应记录 current PureKAN basis-channel functional update formulation no-go，
回到 substrate/base architecture，
而不是继续 functional metric 搜索。
```

当前事实：

```text
1. S1C 已产生：substrate_s1c_count = 1。
2. synthetic 5/7 未产生：synthetic_task_success_count = 0。
3. O7/O8 operator target、full-Jacobian projection、scale、horizon、optimizer-state transport 均未打开 synthetic。
4. real short-run 被计划禁止。
5. BM/BN metric grid 被计划禁止。
```

因此 final official 仍使用当前最强的 O7 actuation-aware reachable target 刷新 artifact / code packet。最终判断仍是：

```text
v13.4 没有达成 S3/S5；
最低成功为 S1C-ChannelControllableSubstrate；
不允许 promotion；
合法 route = R3-SyntheticProofFail。
```

final official 刷新结果：

```text
route = R3-SyntheticProofFail
minimum_success = S1C-ChannelControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
operator_gate_pass_count = 54 / 56
parameter_projection_pass_count = 54 / 56
synthetic_rows = 14
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
required_artifact_missing_count = 0
provenance_violation_count = 0
```

## 13. 用户再次追问后的 top-k O/P future probe 修复

用户再次要求未达成则继续。本次重新审计发现一个真实实现边界：

```text
synthetic future probe 每个 task/seed 只取 actuation_error 最小的一个 O/P 组合。
计划要求 O/P 通过后进入 synthetic proof，并未要求只测试最低 actuation error。
这可能漏掉 actuation 略差但 future source/control 更好的预注册 operator。
```

代码修改：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  新增 --future-probe-topk，默认 1 保持旧 official 语义。
  topk > 1 时，按 actuation_error 选取前 K 个已通过 O/P gate 的预注册候选执行 future controls。
  synthetic row 新增 future_probe_candidates_evaluated / future_probe_selection_rule。
```

合法性：

```text
1. direction 仍由 train-stream operator target 生成。
2. future outcome 不参与 direction construction。
3. top-k 只扩大 proof audit 覆盖，不改变 promotion gate。
4. 该 fallback 不能写成 official promotion。
```

### 13.1 top-k O/P probe

artifact：

```text
results/v13_4_operator_level_basis_channel_functional/fallback_topk_op_probe_v134
```

结果：

```text
route = R3-SyntheticProofFail
operator_gate_pass_count = 170 / 224
parameter_projection_pass_count = 74 / 224
synthetic_rows = 14
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
required_artifact_missing_count = 0
```

最近但不能 pass 的 row：

| task | seed | operator | solver | loss | source_vs_best | NoiseSignalLeak delta | synthetic pass |
|---|---:|---|---|---|---:|---:|---:|
| X2 | 0 | O7 | P3 | CE | 0.004211165376835094 | 0.0077700018882751465 | 0 |

解释：

```text
top-k 修复没有打开 synthetic；
但它证明单候选 actuation selection 偏保守，source gap 已接近 0.005。
剩余 blocker 是 source 不足与 NoiseSignalLeak 轻微坏化。
```

### 13.2 top-k eta sensitivity

artifact：

```text
results/v13_4_operator_level_basis_channel_functional/fallback_topk_eta02_v134
```

结果：

```text
route = R3-SyntheticProofFail
operator_gate_pass_count = 190 / 224
parameter_projection_pass_count = 90 / 224
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
```

最近但不能 pass 的 row：

| task | seed | operator | solver | loss | source_vs_best | NoiseSignalLeak delta | synthetic pass |
|---|---:|---|---|---|---:|---:|---:|
| X2 | 0 | O7 | P3 | Brier | 0.004867172003492869 | 0.007770240306854248 | 0 |

解释：

```text
更大 DeltaZ 把 source_vs_best 推到 0.004867，但仍低于 0.005；
同一 row 的 NoiseSignalLeak_delta 仍为正，因此不能 pass。
```

### 13.3 O9 reachable noise-quarantine fallback

代码修改：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  新增 O9-ReachableNoiseQuarantineChannel。
  O9 结合 O5 的 train-stream unlabeled energy mask 与 O7 的 reachable actuation projection。
```

合法性：

```text
1. O9 只使用 train-stream feature energy 与 generic loss-interface cotangent。
2. 不使用 validation/test/future/query batch 生成方向。
3. 不使用 CEp99/NLL/ECE/LineC 生成方向。
4. 不改变 gate。
```

artifact：

```text
results/v13_4_operator_level_basis_channel_functional/fallback_o9_noise_quarantine_v134
```

结果：

```text
route = R3-SyntheticProofFail
operator_gate_pass_count = 144 / 168
parameter_projection_pass_count = 138 / 168
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
required_artifact_missing_count = 0
```

最近但不能 pass 的 row：

| task | seed | operator | solver | loss | source_vs_best | NoiseSignalLeak delta | synthetic pass |
|---|---:|---|---|---|---:|---:|---:|
| X2 | 0 | O7 | P3 | Brier | 0.004425096191128762 | 0.007770240306854248 | 0 |

判断：

```text
O9 没有解决 source/noise blocker；
top-k eta=0.02 仍是最接近但不能 pass 的 fallback row。
```

最终判断仍是：

```text
v13.4 没有达成 S3/S5；
最低成功为 S1C-ChannelControllableSubstrate；
synthetic_task_success_count = 0；
不允许 promotion；
合法 route = R3-SyntheticProofFail。
```

top-k / O9 后 final official 刷新：

```text
route = R3-SyntheticProofFail
minimum_success = S1C-ChannelControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
operator_gate_pass_count = 54 / 56
parameter_projection_pass_count = 54 / 56
synthetic_rows = 14
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
required_artifact_missing_count = 0
provenance_violation_count = 0
```

最终 stop-contract：

```text
1. S1C 已经打开，但 synthetic 5/7 未打开。
2. O7/O8/O9、top-k O/P future probe、eta sensitivity、horizon / optimizer-state transport 均未打开 synthetic。
3. real short-run 被计划禁止，因为 synthetic_5of7_pass=0。
4. BM/BN metric grid 被计划禁止。
5. 因此当前 PureKAN basis-channel functional update formulation 记录为 no-go；
   下一步应回到 substrate/base architecture 或新 operator target 机制，而不是继续局部 functional metric 搜索。
```

## 14. Non-RAT S1C random actuation autopsy

用户再次要求未达成则继续。本次回到计划第 10.3 节复核 Non-RAT S1C gate，发现此前 `v134_nonrat_s1c_design.csv` 只记录 workspace / LineC blocker，`actuation_error_random` 为空。为闭合 substrate/base architecture 方向，补充 Non-RAT random actuation autopsy。

代码修改：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  新增 --nonrat-actuation-autopsy。
  新增 nonrat_random_actuation_autopsy()。
  对每个 Non-RAT best candidate 构造 train-stream random DeltaZ target，
  通过 J_theta->Z / P4 投影测量 actuation_error_random。
```

合法性：

```text
1. random actuation target 不使用 label、validation/test/future/query batch。
2. 不使用 CEp99/NLL/ECE/LineC 生成方向。
3. workspace 或 LineC fail 后仍不允许进入 functional P3。
4. 该项是 S1C substrate controllability autopsy，不是 promotion。
```

final official 刷新后 Non-RAT S1C autopsy：

| family | candidate | workspace ratio | step ratio | LineC | random actuation error | random cosine | S1C |
|---|---|---:|---:|---:|---:|---:|---:|
| D-CHE | D-CHE16-DegreeEnergyDampingSubstrate | 7.000830564784053 | 7.186207862791935 | 0.0 | 0.2500919997692108 | 0.9684257507324219 | 0 |
| D-FOU | D-FOU14-SincosSharedWorkspace-K2 | 4.2583056478405314 | 1.4176488760262813 | 0.0 | 0.058193475008010864 | 0.9986006021499634 | 0 |
| D-RBF | D-RBF13-WidthConditionGuardSubstrate | 3.5456810631229234 | 1.8206010168168194 | 0.0 | 0.6774501800537109 | 0.736657977104187 | 0 |
| D-WAV | D-WAV13-LocalTailCoverageGuardSubstrate | 4.189368770764119 | 1.977352831438976 | 0.0 | 0.7210556864738464 | 0.6931837797164917 | 0 |

解释：

```text
1. D-CHE/D-FOU 的 random actuation error 低于 0.45，但 workspace ratio 和 LineC 均 fail。
2. D-RBF/D-WAV 同时存在 random actuation fail。
3. Non-RAT S1C 仍为 0。
4. 因此 Non-RAT 仍不能进入 basis-functional proof。
```

最终判断仍是：

```text
v13.4 没有达成 S3/S5；
最低成功为 S1C-ChannelControllableSubstrate；
synthetic_task_success_count = 0；
nonrat_s1c_count = 0；
promotion_allowed = 0；
合法 route = R3-SyntheticProofFail。
```

## 15. no-go boundary / route audit 字段修复

用户再次要求未达成则继续。本次复核发现 `v134_no_go_boundary.md` 对 R3 的 stop-contract 解释不完整：它没有明确写出“当前 PureKAN basis-channel functional update formulation no-go”，且 next hypothesis 仍停留在 debug source/control metrics。但截至本轮，O7/O8/O9、top-k O/P future probe、eta sensitivity、horizon / optimizer-state transport 已经执行。

代码修改：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  route JSON 新增 operator_formulation_no_go。
  route JSON 新增 return_to_substrate_base_architecture。
  write_no_go() 明确写出：
    S1C + O/P pass 但 synthetic 5/7 fail 后，
    current PureKAN basis-channel functional update formulation no-go。
  next hypothesis 改为 substrate/base architecture 或新 operator target mechanism。
```

final official 刷新结果：

```text
route = R3-SyntheticProofFail
minimum_success = S1C-ChannelControllableSubstrate
operator_formulation_no_go = 1
return_to_substrate_base_architecture = 1
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
nonrat_s1c_count = 0
required_artifact_missing_count = 0
```

最终判断仍是：

```text
v13.4 没有达成 S3/S5；
不允许 promotion；
不允许 real short-run；
不应回退 BM/BN metric grid；
当前 PureKAN basis-channel functional update formulation no-go 已正式记录；
下一步应进入 substrate/base architecture 或新 operator target mechanism。
```

## 16. 用户再次追问后的 MLP analog M1-M4 closure 补齐

用户再次要求未达成则继续。本次复核 v13.4 计划第 11 节发现当前 MLP analog artifact 只有 M1 row，没有覆盖 M2/M3/M4。为闭合计划中的 analog closure，本次补齐 M1-M4。

代码修改：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  run_mlp_analog() 从单一 M1 row 改为 M1-M4 analog_specs loop。
  M1: O1-LossCotangentChannelNewtonDiag + P3-LowRankWoodburyProjection + CE。
  M2: O7-BalancedChannelSolve + P4-ConjugateGradientJtJProjection + Brier。
  M3: O7-BalancedChannelSolve + P3-LowRankWoodburyProjection + CE。
  M4: O8-ReachableLogitCotangentChannel + P4-ConjugateGradientJtJProjection + CE。
  每个 analog 使用 deepcopy(model) 单独执行 projection_case。
```

合理性：

```text
1. 这是 MLP analog closure，不是 KAN promotion。
2. 不读取 validation/test/future outcome/query batch 生成方向。
3. 不使用 CEp99/NLL/ECE/LineC 作为方向源。
4. 不改变 synthetic / real short-run / promotion gate。
5. MLP analog 即使 pass，也只能作为 generic effect 审计，不允许写成 PureKAN functional success。
```

official_v134 刷新结果：

```text
route = R3-SyntheticProofFail
minimum_success = S1C-ChannelControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
operator_gate_pass_count = 54
parameter_projection_pass_count = 54
synthetic_rows = 14
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
nonrat_s1c_count = 0
mlp_analog_rows = 4
mlp_analog_pass_count = 0
required_artifact_missing_count = 0
operator_formulation_no_go = 1
return_to_substrate_base_architecture = 1
```

MLP analog rows：

| analog | operator | solver | loss | operator gate | projection gate | actuation error | logit drift | cosine | pass |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| M1-hidden-activation-channel-solve | O1 | P3 | CE | 0 | 0 | 0.08087919652462006 | 0.00019470768165774643 | 0.9968292117118835 | 0 |
| M2-hidden-whitening-inverse-readout-compensation | O7 | P4 | Brier | 0 | 0 | 0.04233492538332939 | 0.00013271461648400873 | 0.9991492033004761 | 0 |
| M3-layerwise-balanced-coordinate-transport | O7 | P3 | CE | 0 | 0 | 0.04356534406542778 | 0.00021344720153138041 | 0.9990956783294678 | 0 |
| M4-lora-like-hidden-subspace-channel-solve | O8 | P4 | CE | 0 | 0 | 0.0027934873942285776 | 0.000034381828299956396 | 0.9999964237213135 | 0 |

解释：

```text
1. MLP analog closure 已从 1 row 补齐为 4 rows。
2. M1-M4 都没有通过 operator/projection gate，mlp_analog_pass_count = 0。
3. 四个 analog 的 actuation error 低但 gate 仍为 0；这说明本次补齐是 closure/audit，不构成 generic MLP positive，也不能支持 KAN promotion。
4. synthetic_task_success_count 仍为 0，因此 real short-run 仍不允许打开。
```

最终判断仍是：

```text
v13.4 没有达成 S3/S5；
最低成功为 S1C-ChannelControllableSubstrate；
synthetic_task_success_count = 0；
nonrat_s1c_count = 0；
mlp_analog_pass_count = 0；
promotion_allowed = 0；
real_short_run 不允许执行；
合法 route = R3-SyntheticProofFail。
```

更新后的 stop-contract：

```text
1. S1C 已打开，但 synthetic 5/7 未打开。
2. O7/O8/O9、top-k future probe、eta sensitivity、horizon / optimizer-state transport、Non-RAT random actuation autopsy 均已执行。
3. MLP analog M1-M4 closure 已补齐且 pass=0。
4. BM/BN metric grid、readout-feature proxy、tail/LineC direction、synthetic fail 后 real short-run 均被计划禁止。
5. 当前 PureKAN basis-channel functional update formulation no-go 仍成立；下一步应回到 substrate/base architecture 或新 operator target mechanism。
```
