# DG-KAN v13.4 OperatorLevelBasisChannelFunctional 执行日志

生成时间：2026-05-27（Asia/Singapore）

本日志只记录实际执行过的命令、修改过的文件和 artifact 路径；不把 smoke/diagnostic/skipped row 写成 promotion。

## 1. 计划读取与实现入口

读取计划文件：

```bash
sed -n '1,220p' docs/DG-KAN_v13.4_OperatorLevelBasisChannelFunctional_StrategicPlan.md
sed -n '220,520p' docs/DG-KAN_v13.4_OperatorLevelBasisChannelFunctional_StrategicPlan.md
sed -n '520,760p' docs/DG-KAN_v13.4_OperatorLevelBasisChannelFunctional_StrategicPlan.md
sed -n '760,900p' docs/DG-KAN_v13.4_OperatorLevelBasisChannelFunctional_StrategicPlan.md
```

关键理解：

```text
v13.4 不继续 BM8-BM15 / BN4-BN7 metric 网格；
必须实现 operator-level basis-channel target DeltaZ；
再通过 J_theta->Z 投影到真实 basis 参数；
以 actuation fidelity 作为进入 synthetic proof 的前置 gate。
```

新增 runner：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
```

主要实现面：

```text
1. basis_channel(model,x)：
   PrimitiveKAN 使用 live layer2_basis(hidden)；
   GroupedRational 使用 live hidden activation；
   MLP analog 使用 hidden activation。
2. DeltaZ target：
   从 generic loss-interface output cotangent 出发；
   用 train-stream live channel -> current logits 的 ridge sketch 估计 J_Z->Y；
   不使用 CEp99/NLL/ECE/LineC 生成方向。
3. J_theta->Z：
   对 live channel sketch 用 torch.autograd.grad 构造真实参数 Jacobian 行。
4. parameter projection：
   P1 diagonal / P2 block / P3 ridge / P6 trust-region projection。
5. writeback：
   对真实参数执行 add_，测量 actual DeltaZ 后回滚；
   writeback trace 记录 before/after sha。
```

代码修改合法性：

```text
1. 不使用 frozen_readout_features 作为 v13.4 channel。
2. 不使用 validation/test/future outcome/query batch 生成方向。
3. CE/Brier 只作为 generic loss interface；CE tail 指标不参与方向。
4. real short-run 若未过 S1C/O/P/synthetic gate，不执行。
```

## 2. 初始检查与 smoke blocker

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v134_operator_level_basis_channel_functional.py
```

结果：

```text
py_compile pass
```

初始 smoke：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/smoke_v134 \
  --synthetic-tasks X1,X7 --synthetic-seeds 0 \
  --synthetic-train-size 32 --synthetic-val-size 16 \
  --operator-batch-size 4 --max-channel-dim 8 --max-jacobian-rows 32 \
  --max-s1-per-family 1 \
  --operator-candidates O1-LossCotangentChannelNewtonDiag \
  --projection-solvers P1-DiagonalProjection,P3-LowRankWoodburyProjection \
  --loss-interfaces CE --checkpoint-steps 2 --batch-size 16 --device cuda:0
```

失败：

```text
RuntimeError: One of the differentiated Tensors does not require grad
```

修复：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  jacobian_theta_to_z() 对 selected basis-channel params 临时 requires_grad_(True)；
  autograd sketch 完成后恢复原 requires_grad 状态；
  jacobian/projection/writeback rows 增加 temporarily_enabled_frozen_channel_params；
  final route/manifest 生成顺序修复，避免用中间 missing manifest 错判 R0。
```

修复后检查：

```bash
conda run -n kan python -m py_compile experiments/run_v134_operator_level_basis_channel_functional.py
```

结果：

```text
py_compile pass
```

## 3. 修复后 smoke

执行：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/smoke_v134 \
  --synthetic-tasks X1,X7 --synthetic-seeds 0 \
  --synthetic-train-size 32 --synthetic-val-size 16 \
  --operator-batch-size 4 --max-channel-dim 8 --max-jacobian-rows 32 \
  --max-s1-per-family 1 \
  --operator-candidates O1-LossCotangentChannelNewtonDiag \
  --projection-solvers P1-DiagonalProjection,P3-LowRankWoodburyProjection \
  --loss-interfaces CE --checkpoint-steps 2 --batch-size 16 --device cuda:0
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
synthetic_rows = 2
synthetic_task_success_count = 0
nonrat_s1c_count = 0
mlp_analog_pass_count = 0
```

smoke 最近 rows：

```text
X1 O1/P3 actuation_error = 0.9144269824028015
X1 O1/P3 cosine = 0.9990381598472595
X1 O1 delta_z_predicted_logit_drift = 0.9540233612060547
X7 O1/P3 actuation_error = 0.8702125549316406
X7 O1/P3 cosine = 0.9991661310195923
X7 O1 delta_z_predicted_logit_drift = 1.3303909301757812
```

解释：

```text
runner / true parameter writeback / artifact path 已可执行；
但 smoke 的 O1 target 预测 logit drift 超预算，projection actuation error 也远高于 0.35；
smoke 不允许 promotion。
```

## 4. 初始 official_v134

执行：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/official_v134 \
  --device cuda:0
```

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

诊断：

```text
basis_channel_surface = kan_hidden_activation_channel
basis_channel_rank_ratio range = 0.1137884184718132 .. 0.1527169644832611
delta_z_effective_rank max = 1.9335395097732544
best projection actuation_error = 0.4365822970867157
```

## 5. fallback：live rational basis channel + O3/O4/P4

代码修改：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  GroupedRational basis_channel 改为 live rational basis output:
    r = _rational_forward(_norm_input(x))
  新增 O3-PopRiskOffdiagChannel。
  新增 O4-TrainProbeCouplingPreservingChannel。
  新增 P4-ConjugateGradientJtJProjection。
```

执行：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/fallback_live_rational_channel_v134 \
  --device cuda:0 \
  --operator-candidates O1-LossCotangentChannelNewtonDiag,O2-LossCotangentChannelSNR,O3-PopRiskOffdiagChannel,O4-TrainProbeCouplingPreservingChannel,O5-NoiseQuarantineUnlabeledProxy,O6-ReservoirReleaseUnlabeledProxy,O7-BalancedChannelSolve \
  --projection-solvers P1-DiagonalProjection,P3-LowRankWoodburyProjection,P4-ConjugateGradientJtJProjection,P6-TrustRegionProjection \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 128 \
  --max-update-norm-ratio 0.10 --delta-z-eta 0.01
```

结果：

```text
route = R1-NoS1CSubstrate
operator_gate_pass_count = 96 / 392
parameter_projection_pass_count = 0 / 392
basis_channel_rank_ratio range = 0.4564659595489502 .. 0.5421897172927856
```

O pass 分布：

```text
O3-PopRiskOffdiagChannel = 52/56
O4-TrainProbeCouplingPreservingChannel = 44/56
其他 O = 0/56
```

最近 projection：

```text
best actuation_error = 0.8251322507858276
best cosine = 0.5747919082641602
```

## 6. fallback：full-row Jacobian projection

执行：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/fallback_full_jacobian_projection_v134 \
  --device cuda:0 \
  --operator-candidates O3-PopRiskOffdiagChannel,O4-TrainProbeCouplingPreservingChannel \
  --projection-solvers P3-LowRankWoodburyProjection,P4-ConjugateGradientJtJProjection,P6-TrustRegionProjection \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --delta-z-eta 0.01
```

结果：

```text
route = R1-NoS1CSubstrate
operator_gate_pass_count = 72 / 84
parameter_projection_pass_count = 0 / 84
best actuation_error = 0.6389598846435547
best cosine = 0.7697218656539917
```

解释：

```text
full-row J_theta->Z 改善了 actuation fidelity，但仍高于 P gate 的 0.35。
```

## 7. fallback：O7 actuation-aware reachable target

代码修改：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  新增 delta_z_stats()。
  O7-BalancedChannelSolve 在 full target rows 可用时：
    先生成 generic loss-interface DeltaZ；
    再用 J_theta->Z / P4 投到当前真实参数可执行子空间；
    重新计算 O gate 指标；
    CSV 记录 actuation_aware_reachable_target=1。
```

执行：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/fallback_o7_actuation_aware_v134 \
  --device cuda:0 \
  --operator-candidates O7-BalancedChannelSolve \
  --projection-solvers P3-LowRankWoodburyProjection,P4-ConjugateGradientJtJProjection,P6-TrustRegionProjection \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --delta-z-eta 0.01
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

## 8. Line X future-probe / controls 实现

代码修改：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  新增 compute_projection_vector()。
  新增 run_future_operator_case()。
  synthetic proof 对每个 O/P pass task 执行：
    Functional；
    TaskOnlyAdamW；
    NoOpMatchedOverhead；
    RandomMatchedNorm。
  v134_synthetic_task_family_proof.csv 写入 source_vs_best、CouplingR2/Noise/Reservoir/tail delta。
  task-family summary 改为计划要求的 >=2 substrates 或 >=2 seeds。
  CLI 新增 --future-steps / --future-lr / --future-weight-decay。
```

执行：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/fallback_o7_actuation_aware_synthetic_v134 \
  --device cuda:0 --synthetic-seeds 0,1 \
  --operator-candidates O7-BalancedChannelSolve \
  --projection-solvers P3-LowRankWoodburyProjection,P4-ConjugateGradientJtJProjection \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --delta-z-eta 0.01 \
  --future-steps 30 --future-lr 0.003 --future-weight-decay 0.001
```

## 13. top-k/O9/Non-RAT autopsy 后 official 刷新记录

前续补跑已经在实验复盘中记录，本执行日志补充最终 official 命令形态。为了覆盖 Non-RAT S1C autopsy，本轮 official 命令加入 `--nonrat-actuation-autopsy`：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/official_v134 \
  --device cuda:0 --synthetic-seeds 0,1 \
  --operator-candidates O7-BalancedChannelSolve \
  --projection-solvers P3-LowRankWoodburyProjection,P4-ConjugateGradientJtJProjection \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --delta-z-eta 0.01 \
  --future-steps 30 --future-lr 0.003 --future-weight-decay 0.001 \
  --nonrat-actuation-autopsy
```

结果以 `results/v13_4_operator_level_basis_channel_functional/official_v134/v134_route_decision.json` 为准。

## 14. 用户再次追问后的 MLP analog M1-M4 closure 补齐

用户再次要求未达成则继续。本次复核 v13.4 计划第 11 节发现 MLP analog closure 在 artifact 中只输出了 M1：

```text
v134_mlp_analog_channel_control.csv rows = 1
analog = M1-hidden-activation-channel-solve
```

计划要求覆盖：

```text
M1 hidden activation channel solve
M2 hidden whitening + inverse readout compensation
M3 layerwise balanced coordinate transport
M4 LoRA-like hidden subspace channel solve
```

因此补齐 M2-M4，不改变 promotion gate，不把 MLP analog 写成 KAN success。

代码修改：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  run_mlp_analog() 从单一 M1 row 改为 M1-M4 analog_specs loop。
  M1 使用 O1 + P3 + CE。
  M2 使用 O7 + P4 + Brier。
  M3 使用 O7 + P3 + CE。
  M4 使用 O8 + P4 + CE。
  每个 analog 使用 deepcopy(model) 单独执行 projection_case，避免异常污染其他 analog。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v134_operator_level_basis_channel_functional.py
```

结果：

```text
py_compile pass
```

official_v134 刷新命令：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/official_v134 \
  --device cuda:0 --synthetic-seeds 0,1 \
  --operator-candidates O7-BalancedChannelSolve \
  --projection-solvers P3-LowRankWoodburyProjection,P4-ConjugateGradientJtJProjection \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --delta-z-eta 0.01 \
  --future-steps 30 --future-lr 0.003 --future-weight-decay 0.001 \
  --nonrat-actuation-autopsy
```

写日志前刷新结果：

```text
route = R3-SyntheticProofFail
minimum_success = S1C-ChannelControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
operator_gate_pass_count = 54 / 56
parameter_projection_pass_count = 54 / 56
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
nonrat_s1c_count = 0
mlp_analog_rows = 4
mlp_analog_pass_count = 0
required_artifact_missing_count = 0
operator_formulation_no_go = 1
return_to_substrate_base_architecture = 1
```

MLP analog closure rows：

| analog | operator | solver | loss | operator gate | projection gate | actuation error | cosine | pass |
|---|---|---|---|---:|---:|---:|---:|---:|
| M1-hidden-activation-channel-solve | O1 | P3 | CE | 0 | 0 | 0.08087919652462006 | 0.9968292117118835 | 0 |
| M2-hidden-whitening-inverse-readout-compensation | O7 | P4 | Brier | 0 | 0 | 0.04233492538332939 | 0.9991492033004761 | 0 |
| M3-layerwise-balanced-coordinate-transport | O7 | P3 | CE | 0 | 0 | 0.04356534406542778 | 0.9990956783294678 | 0 |
| M4-lora-like-hidden-subspace-channel-solve | O8 | P4 | CE | 0 | 0 | 0.0027934873942285776 | 0.9999964237213135 | 0 |

判断：

```text
1. MLP analog M1-M4 closure 已补齐，mlp_analog_rows = 4。
2. 四个 analog 的 actuation error 都低，但 operator/projection gate 均未 pass，因此 mlp_analog_pass_count = 0。
3. 该补齐没有改变 v13.4 route：synthetic_task_success_count 仍为 0。
4. v13.4 仍未达成 S3/S5，不允许 promotion。
```

结果：

```text
route = R3-SyntheticProofFail
operator_gate_pass_count = 54 / 56
parameter_projection_pass_count = 54 / 56
synthetic_rows = 14
synthetic_task_success_count = 0
source_vs_best max = 0.0023105812220099928
```

## 9. synthetic fallback sensitivity

更大 target scale：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/fallback_o7_eta003_synthetic_v134 \
  --device cuda:0 --synthetic-seeds 0,1 \
  --operator-candidates O7-BalancedChannelSolve \
  --projection-solvers P3-LowRankWoodburyProjection,P4-ConjugateGradientJtJProjection \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 1.00 --delta-z-eta 0.03 \
  --future-steps 30 --future-lr 0.003 --future-weight-decay 0.001
```

结果：

```text
route = R3-SyntheticProofFail
parameter_projection_pass_count = 50 / 56
synthetic_task_success_count = 0
source_vs_best max = 0.002224695740159814
```

短 horizon：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/fallback_o7_future10_v134 \
  --device cuda:0 --synthetic-seeds 0,1 \
  --operator-candidates O7-BalancedChannelSolve \
  --projection-solvers P3-LowRankWoodburyProjection,P4-ConjugateGradientJtJProjection \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --delta-z-eta 0.01 \
  --future-steps 10 --future-lr 0.003 --future-weight-decay 0.001
```

结果：

```text
route = R3-SyntheticProofFail
parameter_projection_pass_count = 54 / 56
synthetic_task_success_count = 0
source_vs_best max = 0.0004274057040334796
```

optimizer-state transport：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/fallback_o7_optimizer_state_v134 \
  --device cuda:0 --synthetic-seeds 0,1 \
  --operator-candidates O7-BalancedChannelSolve \
  --projection-solvers P3-LowRankWoodburyProjection,P4-ConjugateGradientJtJProjection \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --delta-z-eta 0.01 \
  --future-steps 30 --future-lr 0.003 --future-weight-decay 0.001 \
  --optimizer-state-transport --optimizer-state-scale 0.20
```

结果：

```text
route = R3-SyntheticProofFail
parameter_projection_pass_count = 54 / 56
synthetic_task_success_count = 0
source_vs_best max = 0.0010513481007466546
```

## 10. final official_v134 刷新

执行：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/official_v134 \
  --device cuda:0 --synthetic-seeds 0,1 \
  --operator-candidates O7-BalancedChannelSolve \
  --projection-solvers P3-LowRankWoodburyProjection,P4-ConjugateGradientJtJProjection \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --delta-z-eta 0.01 \
  --future-steps 30 --future-lr 0.003 --future-weight-decay 0.001
```

结果：

```text
route = R3-SyntheticProofFail
minimum_success = S1C-ChannelControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
operator_gate_pass_count = 54 / 56
parameter_projection_pass_count = 54 / 56
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
required_artifact_missing_count = 0
code_review_packet_sha256 = 以 results/v13_4_operator_level_basis_channel_functional/official_v134/v134_route_decision.json 为准
```

## 16. Non-RAT S1C random actuation autopsy

复核计划 10.3 后发现：

```text
Non-RAT S1C gate 包含 actuation_error_random <= 0.45。
此前 v134_nonrat_s1c_design.csv 只记录 workspace / LineC blocker，actuation_error_random 为空。
```

代码修改：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  新增 --nonrat-actuation-autopsy。
  新增 nonrat_random_actuation_autopsy()。
  对每个 Non-RAT best candidate 构造 train-stream random DeltaZ target，
  通过 J_theta->Z / P4 投影测量 actuation_error_random。
  workspace 或 LineC fail 后仍不允许进入 functional P3。
```

执行：

```bash
conda run -n kan python -m py_compile experiments/run_v134_operator_level_basis_channel_functional.py
```

结果：

```text
py_compile pass
```

执行：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/official_v134 \
  --device cuda:0 --synthetic-seeds 0,1 \
  --operator-candidates O7-BalancedChannelSolve \
  --projection-solvers P3-LowRankWoodburyProjection,P4-ConjugateGradientJtJProjection \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --delta-z-eta 0.01 \
  --future-steps 30 --future-lr 0.003 --future-weight-decay 0.001 \
  --nonrat-actuation-autopsy
```

结果：

```text
route = R3-SyntheticProofFail
nonrat_s1c_count = 0
synthetic_task_success_count = 0
required_artifact_missing_count = 0
```

Non-RAT S1C autopsy：

| family | candidate | workspace ratio | LineC | random actuation error | S1C |
|---|---|---:|---:|---:|---:|
| D-CHE | D-CHE16 | 7.000830564784053 | 0.0 | 0.2500919997692108 | 0 |
| D-FOU | D-FOU14 | 4.2583056478405314 | 0.0 | 0.058193475008010864 | 0 |
| D-RBF | D-RBF13 | 3.5456810631229234 | 0.0 | 0.6774501800537109 | 0 |
| D-WAV | D-WAV13 | 4.189368770764119 | 0.0 | 0.7210556864738464 | 0 |

判断：

```text
D-CHE/D-FOU random actuation 可控，但 workspace/LineC fail；
D-RBF/D-WAV 同时存在 random actuation fail；
Non-RAT S1C 仍为 0，不能进入 functional P3。
```

## 17. no-go boundary / route audit 字段修复

复核发现：

```text
v134_no_go_boundary.md 对 R3-SyntheticProofFail 的 stop-contract 解释不完整；
v134_next_hypothesis_queue.md 仍提示 debug operator target source/control metrics，
但 top-k / O8 / O9 / eta / horizon / optimizer-state fallback 已经执行。
```

代码修改：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  route JSON 新增 operator_formulation_no_go。
  route JSON 新增 return_to_substrate_base_architecture。
  write_no_go() 明确写出：
    S1C + O/P pass 但 synthetic 5/7 fail 后，
    current PureKAN basis-channel functional update formulation no-go。
  next hypothesis 改为 substrate/base architecture 或新 operator target mechanism，
  不再提示继续 BM/BN metric grid 或 real short-run。
```

执行：

```bash
conda run -n kan python -m py_compile experiments/run_v134_operator_level_basis_channel_functional.py
```

结果：

```text
py_compile pass
```

执行：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/official_v134 \
  --device cuda:0 --synthetic-seeds 0,1 \
  --operator-candidates O7-BalancedChannelSolve \
  --projection-solvers P3-LowRankWoodburyProjection,P4-ConjugateGradientJtJProjection \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --delta-z-eta 0.01 \
  --future-steps 30 --future-lr 0.003 --future-weight-decay 0.001 \
  --nonrat-actuation-autopsy
```

结果：

```text
route = R3-SyntheticProofFail
operator_formulation_no_go = 1
return_to_substrate_base_architecture = 1
synthetic_task_success_count = 0
nonrat_s1c_count = 0
required_artifact_missing_count = 0
code_review_packet_sha256 = 以 results/v13_4_operator_level_basis_channel_functional/official_v134/v134_route_decision.json 为准
```

## 13. 用户再次追问后的 top-k O/P future probe 修复

复核发现一个真实实现边界：

```text
当前 synthetic future probe 每个 task/seed 只用 actuation_error 最小的 1 个 O/P 组合。
计划要求 O/P 通过后进入 synthetic proof，并未要求只测试最低 actuation error。
这可能漏掉 actuation 略差但 future source/control 更好的预注册 operator。
```

代码修改：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  新增 --future-probe-topk，默认 1 保持旧 official 语义。
  当 topk > 1 时，按 actuation_error 选取前 K 个已通过 O/P gate 的预注册候选执行 future controls。
  synthetic row 新增 future_probe_candidates_evaluated / future_probe_selection_rule。
```

合法性：

```text
1. O/P direction 仍由 train-stream operator target 生成。
2. future outcome 不参与 direction construction。
3. top-k 只扩大 proof audit 的候选覆盖，不改变 promotion gate。
```

执行：

```bash
conda run -n kan python -m py_compile experiments/run_v134_operator_level_basis_channel_functional.py
```

结果：

```text
py_compile pass
```

执行：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/fallback_topk_op_probe_v134 \
  --device cuda:0 --synthetic-seeds 0,1 \
  --operator-candidates O3-PopRiskOffdiagChannel,O4-TrainProbeCouplingPreservingChannel,O7-BalancedChannelSolve,O8-ReachableLogitCotangentChannel \
  --projection-solvers P3-LowRankWoodburyProjection,P4-ConjugateGradientJtJProjection \
  --future-probe-topk 4 \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --delta-z-eta 0.01 \
  --future-steps 30 --future-lr 0.003 --future-weight-decay 0.001
```

结果：

```text
route = R3-SyntheticProofFail
operator_gate_pass_count = 170 / 224
parameter_projection_pass_count = 74 / 224
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
required_artifact_missing_count = 0
```

最近 row：

```text
X2 seed=0 source_vs_best = 0.004211165376835094
NoiseSignalLeak_delta = 0.0077700018882751465
synthetic_task_success = 0
```

判断：

```text
top-k 修复没有打开 synthetic，但证明单候选 actuation selection 确实偏保守；
source gap 接近 0.005，剩余 blocker 是 source 不足与 NoiseSignalLeak 轻微坏化。
```

## 14. top-k eta sensitivity 与 O9 noise-quarantine fallback

执行 top-k + 更大 DeltaZ：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/fallback_topk_eta02_v134 \
  --device cuda:0 --synthetic-seeds 0,1 \
  --operator-candidates O3-PopRiskOffdiagChannel,O4-TrainProbeCouplingPreservingChannel,O7-BalancedChannelSolve,O8-ReachableLogitCotangentChannel \
  --projection-solvers P3-LowRankWoodburyProjection,P4-ConjugateGradientJtJProjection \
  --future-probe-topk 4 \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --delta-z-eta 0.02 \
  --future-steps 30 --future-lr 0.003 --future-weight-decay 0.001
```

结果：

```text
route = R3-SyntheticProofFail
operator_gate_pass_count = 190 / 224
parameter_projection_pass_count = 90 / 224
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
```

最近 row：

```text
X2 seed=0 source_vs_best = 0.004867172003492869
NoiseSignalLeak_delta = 0.007770240306854248
synthetic_task_success = 0
```

代码修改：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  新增 O9-ReachableNoiseQuarantineChannel。
  O9 结合 O5 的 train-stream unlabeled energy mask 与 O7 的 reachable actuation projection。
```

执行：

```bash
conda run -n kan python -m py_compile experiments/run_v134_operator_level_basis_channel_functional.py
```

结果：

```text
py_compile pass
```

执行：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/fallback_o9_noise_quarantine_v134 \
  --device cuda:0 --synthetic-seeds 0,1 \
  --operator-candidates O7-BalancedChannelSolve,O8-ReachableLogitCotangentChannel,O9-ReachableNoiseQuarantineChannel \
  --projection-solvers P3-LowRankWoodburyProjection,P4-ConjugateGradientJtJProjection \
  --future-probe-topk 4 \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --delta-z-eta 0.02 \
  --future-steps 30 --future-lr 0.003 --future-weight-decay 0.001
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

最近 row：

```text
X2 seed=0 source_vs_best = 0.004425096191128762
NoiseSignalLeak_delta = 0.007770240306854248
synthetic_task_success = 0
```

判断：

```text
O9 没有解决 source/noise blocker；
top-k eta=0.02 仍是最接近但不能 pass 的 fallback row。
```

## 15. top-k / O9 后 final official 刷新

执行：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/official_v134 \
  --device cuda:0 --synthetic-seeds 0,1 \
  --operator-candidates O7-BalancedChannelSolve \
  --projection-solvers P3-LowRankWoodburyProjection,P4-ConjugateGradientJtJProjection \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --delta-z-eta 0.01 \
  --future-steps 30 --future-lr 0.003 --future-weight-decay 0.001
```

结果：

```text
route = R3-SyntheticProofFail
minimum_success = S1C-ChannelControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
operator_gate_pass_count = 54 / 56
parameter_projection_pass_count = 54 / 56
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
required_artifact_missing_count = 0
code_review_packet_sha256 = 以 results/v13_4_operator_level_basis_channel_functional/official_v134/v134_route_decision.json 为准
```

最终 artifact：

```text
results/v13_4_operator_level_basis_channel_functional/official_v134
```

最终 route：

```text
route = R3-SyntheticProofFail
minimum_success = S1C-ChannelControllableSubstrate
operator_gate_pass_count = 54 / 56
parameter_projection_pass_count = 54 / 56
synthetic_task_success_count = 0
nonrat_s1c_count = 0
mlp_analog_pass_count = 0
required_artifact_missing_count = 0
code_review_packet_sha256 = 以 results/v13_4_operator_level_basis_channel_functional/official_v134/v134_route_decision.json 为准
```

说明：

```text
code packet 包含执行日志与复盘日志；后续文档编辑会改变 zip bytes。
最终 route 以 v134_route_decision.json 为准。
```

## 11. 用户再次追问后的 O8 reachable-logit fallback

用户再次要求未达成则继续。本次对照计划 Stop/Go：

```text
禁止：BM/BN metric grid、readout-feature proxy、tail/LineC direction、synthetic fail 后 real short-run。
允许：operator-level basis-channel target、parameter projection actuation fidelity、true writeback、S1C、Non-RAT S1C design、MLP analog。
```

因此本次只继续 operator-level target，不回退到 BM/BN metric。

代码修改：

```text
experiments/run_v134_operator_level_basis_channel_functional.py
  新增 O8-ReachableLogitCotangentChannel。
  新增 reachable_logit_cotangent_target()。
  O8 使用 J_theta->Z 与 ridge J_Z->Y，把 generic output cotangent 直接投影到可执行 J_theta->Z->Y 子空间。
  不使用 validation/test/future/query batch。
  不使用 CEp99/NLL/ECE/LineC 生成方向。
```

语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v134_operator_level_basis_channel_functional.py
```

结果：

```text
py_compile pass
```

执行：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/fallback_o8_reachable_logit_v134 \
  --device cuda:0 --synthetic-seeds 0,1 \
  --operator-candidates O8-ReachableLogitCotangentChannel \
  --projection-solvers P3-LowRankWoodburyProjection,P4-ConjugateGradientJtJProjection \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --delta-z-eta 0.03 \
  --future-steps 30 --future-lr 0.003 --future-weight-decay 0.001
```

结果：

```text
route = R3-SyntheticProofFail
minimum_success = S1C-ChannelControllableSubstrate
operator_gate_pass_count = 46 / 56
parameter_projection_pass_count = 46 / 56
synthetic_rows = 14
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
required_artifact_missing_count = 0
```

最近 row：

```text
X3 seed=1 source_vs_best = 0.001302997653710218
X4 seed=0 source_vs_best = 0.0009605891775466072
X6 seed=0 entered_synthetic_proof = 0
```

判断：

```text
O8 没有超过 O7；source_vs_best 仍低于 0.005。
```

## 12. stop-contract 复核与 final official 刷新

计划第 14 节规定：

```text
如果 v13.4 仍然不能产生 S1C 或 synthetic 5/7，
应记录 current PureKAN basis-channel functional update formulation no-go，
回到 substrate/base architecture，
而不是继续 functional metric 搜索。
```

当前事实：

```text
S1C 已产生：substrate_s1c_count = 1。
synthetic 5/7 未产生：synthetic_task_success_count = 0。
O7/O8 operator target、full Jacobian projection、scale、horizon、optimizer-state transport 均未打开 synthetic。
real short-run 被计划禁止。
BM/BN metric grid 被计划禁止。
```

因此 final official 仍使用最强的 O7 actuation-aware reachable target，并刷新 code packet：

```bash
conda run -n kan python experiments/run_v134_operator_level_basis_channel_functional.py \
  --out-dir results/v13_4_operator_level_basis_channel_functional/official_v134 \
  --device cuda:0 --synthetic-seeds 0,1 \
  --operator-candidates O7-BalancedChannelSolve \
  --projection-solvers P3-LowRankWoodburyProjection,P4-ConjugateGradientJtJProjection \
  --operator-batch-size 16 --max-channel-dim 16 --max-jacobian-rows 256 \
  --max-update-norm-ratio 0.50 --delta-z-eta 0.01 \
  --future-steps 30 --future-lr 0.003 --future-weight-decay 0.001
```
