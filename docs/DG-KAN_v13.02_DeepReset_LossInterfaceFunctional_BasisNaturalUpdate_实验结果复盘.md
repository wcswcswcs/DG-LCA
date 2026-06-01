# DG-KAN v13.2 DeepReset LossInterfaceFunctional BasisNaturalUpdate 实验结果复盘

生成时间：2026-05-27（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 smoke/diagnostic/fallback row 写成 promotion。

## 1. 计划理解

v13.2 是 deep reset，不再继续 readout-feature proxy 或 token 网格，而是要求：

```text
loss-interface generic cotangent -> real basis-parameter natural update -> future probe / controls
```

硬约束：

```text
1. strict FC-PureKAN / no active B-spline budget。
2. 不使用 label-informed initialization。
3. functional direction 不使用 validation/test/future outcome/query batch/dataset-name branch。
4. CE/NLL/ECE/CEp99 只能作为审计和坏化约束。
5. 必须记录 code surface、basis param manifest、writeback trace、loss-interface audit、forbidden audit。
6. blocker 后必须按计划执行 metric/norm/safety/candidate breadth/MLP analog 等修复方向。
```

本轮实现明确区分：

```text
1. output cotangent 来自 generic loss interface。
2. update 写回真实 model basis 参数。
3. 不使用 frozen readout-feature transport proxy。
4. real short-run 只有在 S1 + P3 + synthetic 5/7 后才允许执行；未触发时写 explicit skip row。
```

## 2. 本轮代码修改

新增文件：

```text
experiments/run_v132_loss_interface_basis_natural_update.py
```

主要功能：

```text
1. 读取 v12.35 substrate artifact，生成 v132_substrate_map.csv。
2. 在真实 basis 参数上构造 BN1/BN2/BN3 basis-natural update。
3. 写出 v132_basis_param_manifest.csv 与 v132_functional_writeback_trace.csv。
4. 输出 v132_loss_interface_audit.csv 与 v132_forbidden_information_audit.csv。
5. 执行 synthetic X1..X7 future probe、controls、MLP analog、LineC audit。
6. 输出 required manifest、figures、route decision、code review packet。
```

追加修复：

```text
1. synthetic_dim 默认改为 16，以匹配 D-RAT paircrossR136 substrate。
2. CE interface audit 改为 loss_interface_is_ce=1、uses_ce_specific_formula=0，避免把合法 CE loss-interface 误标为 CE-specific tail formula。
3. ShuffledCotangent control 改为真正 shuffle output cotangent，而不是随机近似。
4. 新增 BN4-LogitEnergyOrthogonalNatural 与 BN5-LogitEnergyOrthogonalSNR。
```

BN4/BN5 合法性：

```text
1. 只用 train-stream current logits 计算 logit energy gradient。
2. 只做一阶安全投影，投掉会增加 logit energy 的 update 分量。
3. 不读取 validation/test/future outcome。
4. 不使用 CEp99/NLL/ECE/LineC 作为方向源。
5. 不改变 P3 gate。
```

## 3. smoke 与 blocker

首次 smoke 失败：

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x36 and 136x3)
```

原因：

```text
v12.35 S1 D-RAT substrate 使用 paircrossR136；synthetic_dim=8 时 pair feature 只有 36。
```

修复后 smoke：

```text
route = R2-BasisNaturalP3Failed
minimum_success = S1-EfficientControllableSubstrate
required_artifact_missing_count = 0
basis_natural_p3_rows = 1
basis_natural_p3_pass_count = 0
full_basis_param_update_rows = 1
```

解释：smoke 只证明 runner / writeback / manifest 可运行，不允许 promotion。

## 4. Substrate map

来源：

```text
results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_basis_substrate_health.csv
```

v13.2 S1/S2 结果：

```text
substrate_rows = 44
substrate_s1_count = 11
substrate_s2_healthy_base_count = 0
```

解释：

```text
1. S1 efficient controllable substrate 仍只来自 D-RAT。
2. 没有 S2 healthy base。
3. Non-RAT family 仍没有进入本轮 real functional promotion path。
```

## 5. 初始 official：BN1/BN2/BN3

执行规模：

```text
selected S1 family = D-RAT
base candidate = D-RAT26-TangentTrustRegionNoCE
synthetic tasks = X1..X7
seed = 0
loss interfaces = CE,Brier
updates = BN1-NaturalDiag,BN2-SNROnly,BN3-DampedNatural
p3_rows = 42
```

结果：

```text
route = R2-BasisNaturalP3Failed
basis_natural_p3_pass_count = 0
synthetic_task_success_count = 0
full_basis_param_update_rows = 42
required_artifact_missing_count = 0
```

最接近但不能 pass 的 rows：

| candidate | task | update | loss | source_vs_best | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | CEp99 delta | P3 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| D-RAT26 | X7 | BN2 | Brier | 0.0136790924427454 | 0.20059967041015625 | 0.02785104513168335 | -0.025494098663330078 | -2.628113269805908 | 0 |
| D-RAT26 | X7 | BN2 | CE | 0.013294316267763395 | 10.668424606323242 | 0.0223463773727417 | -0.9846115112304688 | -2.6807427406311035 | 0 |
| D-RAT26 | X4 | BN3 | CE | 0.0033973719756446585 | 1.2711772918701172 | 0.034008026123046875 | 0.07265615463256836 | 0.35059404373168945 | 0 |

主要 blocker：

```text
source_vs_best_control 不稳定；
即使 future proxy 改善，NoiseSignalLeak 或 Reservoir 常坏化；
CEp99/ECE 在部分 row 坏化。
```

## 6. fallback 1：metric/norm + future optimizer sensitivity

执行内容：

```text
functional_eta = 0.005
metric_rho = 0.01
max_update_norm_ratio = 0.005
future_lr = 0.003
```

结果：

```text
route = R2-BasisNaturalP3Failed
basis_natural_p3_pass_count = 0
synthetic_task_success_count = 0
```

解释：

```text
更保守 step/damping 没有打开 P3；
主要失败变成 source_vs_best_control < 0.005 与 NoiseSignalLeak 轻微坏化。
```

## 7. fallback 2：S1 candidate breadth

执行内容：

```text
top-3 S1 candidates = D-RAT26,D-RAT27,D-RAT28
update = BN2-SNROnly
loss interfaces = CE,Brier
p3_rows = 42
synthetic_rows = 21
```

结果：

```text
route = R2-BasisNaturalP3Failed
basis_natural_p3_pass_count = 0
synthetic_task_success_count = 0
```

解释：

```text
P3 no-go 不是 D-RAT26 单一 substrate 局部问题。
多个 X7 row 有 source_vs_best_control > 0.005，但仍被 NoiseSignalLeak / Reservoir blocker 拒绝。
```

## 8. fallback 3：logit-energy safety projection

执行内容：

```text
top-3 S1 candidates = D-RAT26,D-RAT27,D-RAT28
updates = BN4-LogitEnergyOrthogonalNatural,BN5-LogitEnergyOrthogonalSNR
loss interfaces = CE,Brier
p3_rows = 84
```

final official route：

```text
route = R3-P3OnlySyntheticFailed
minimum_success = S2-BasisNaturalP3
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
substrate_s1_count = 11
substrate_s2_healthy_base_count = 0
basis_natural_p3_rows = 84
basis_natural_p3_pass_count = 2
synthetic_rows = 21
synthetic_task_success_count = 2
synthetic_5of7_pass = 0
real_short_run_pass_count = 0
mlp_analog_pass_count = 0
generic_reparameterization_effect = 0
KAN_specific_claim_allowed = 0
full_basis_param_update_rows = 84
provenance_violation_count = 0
```

P3 pass rows：

| candidate | task | update | loss | source_vs_best | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | CEp99 delta | NLL delta | ECE delta |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| D-RAT26 | X7 | BN5 | Brier | 0.010142281714579426 | 1.9736881256103516 | -0.004589974880218506 | -0.13720369338989258 | -2.2259368896484375 | -0.24558186531066895 | -0.05060720443725586 |
| D-RAT27 | X7 | BN5 | Brier | 0.01019153260218647 | 4.897928237915039 | -0.003060281276702881 | -0.41165590286254883 | -2.7505455017089844 | -0.23843955993652344 | -0.028306961059570312 |

Safety projection audit：

```text
safety_rows = 84
accepted = 52
logit_energy_orthogonalized = 32
```

解释：

```text
1. BN5 是真实进展，首次打开了 basis-natural P3。
2. 但 P3 只集中在 X7，synthetic task success = 2/21，不满足 5/7。
3. 因 synthetic_5of7_pass=0，real short-run repair 按 gate 写 skipped row，没有执行 promotion path。
```

## 9. MLP analog

结果：

```text
mlp_analog_rows = 1
mlp_analog_pass_count = 0
generic_reparameterization_effect = 0
KAN_specific_claim_allowed = 0
```

解释：本轮没有出现 MLP analog 解释全部收益的情况；但 real short-run 没打开，因此也不能主张 KAN-specific official success。

## 10. Required artifacts

主要产物：

```text
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_route_decision.json
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_required_artifact_manifest.csv
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_code_surface.csv
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_basis_param_manifest.csv
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_functional_writeback_trace.csv
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_loss_interface_audit.csv
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_forbidden_information_audit.csv
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_substrate_map.csv
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_gradient_snr_histogram.csv
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_basis_metric_condition.csv
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_safety_projection.csv
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_basis_natural_p3.csv
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_synthetic_mechanism_proof.csv
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_real_short_run_repair.csv
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_mlp_analog_control.csv
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_linec_audit.csv
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_failure_table.csv
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_next_hypothesis_queue.md
results/v13_2_deep_reset_loss_interface_basis_natural_update/official_v132/v132_code_review_packet.zip
```

Required manifest：

```text
manifest_rows = 30
missing_required_rows = 0
```

## 11. 最终科学结论

v13.2 没有达成 S5，但相对初始 BN1/BN2/BN3 已有真实进展。

最终合法状态：

```text
route = R3-P3OnlySyntheticFailed
minimum_success = S2-BasisNaturalP3
promotion_allowed = 0
```

已闭合事实：

```text
1. v13.2 从 v12.35 artifact 中筛出 11 个 S1 substrate，全部来自 D-RAT；S2 healthy base 为 0。
2. runner 写回真实 basis 参数，不是 readout-feature proxy；writeback trace rows = 84。
3. 初始 BN1/BN2/BN3 没有 P3 pass。
4. metric/norm/future-lr fallback 没有 P3 pass。
5. S1 candidate breadth fallback 没有 P3 pass。
6. label-free logit-energy safety projection 打开 P3：2/84 rows pass。
7. synthetic success 只有 2/21，未达到 5/7，因此 real short-run gate 未打开。
8. MLP analog 未 pass；没有 generic reparameterization effect。
9. Provenance audit 通过，required artifacts 缺失为 0。
```

新增 no-go boundary：

```text
1. Loss-interface basis-natural update 本身可在 X7 上形成合法 P3，但目前泛化不到 X1..X6。
2. 主要 blocker 从“真实参数写回不可执行”转为“synthetic 5/7 不成立”。
3. BN5 safety projection 可压住部分 NoiseSignalLeak/Reservoir 坏化，但仍不是通用机制。
4. 继续推进应进入更强的 task-family robust basis metric / optimizer-state transport / Non-RAT S1 substrate repair，而不是把 X7-only P3 写成 official success。
```

## 12. 用户再次追问后的 Case C 复核与继续执行

用户再次要求确认 v13.2 是否达成目标，若未达成则继续。本次重新对照计划文件 Case C：

```text
Case C：P3 pass，但 synthetic fail。
Codex 必须：
1. 检查是否仍是 feature-table proxy；
2. 检查 optimizer-state transport；
3. 检查 update norm / clipping；
4. 检查 future probe 是否被 task optimizer 覆盖。
```

对照当前 final official：

```text
route = R3-P3OnlySyntheticFailed
minimum_success = S2-BasisNaturalP3
basis_natural_p3_pass_count = 2
synthetic_task_success_count = 2
synthetic_5of7_pass = 0
real_short_run_pass_count = 0
```

已完成项：

```text
1. feature-table proxy 检查：writeback trace 记录 full_basis_param_update_rows=84，readout_feature_proxy_only=0。
2. update norm / clipping：metric/norm fallback 已执行，P3=0。
3. future probe 覆盖检查：BN4/BN5 low future LR fallback 已执行，P3=0。
4. optimizer-state transport：本次新增并执行。
```

### 12.1 future optimizer sensitivity after BN4/BN5

执行：

```text
fallback_energy_projection_low_future_lr_v132
future_lr = 0.003
future_weight_decay = 0.001
```

结果：

```text
route = R2-BasisNaturalP3Failed
basis_natural_p3_pass_count = 0
synthetic_task_success_count = 0
```

解释：

```text
低 future LR 没有证明 task optimizer 覆盖了 functional signal；
相反，它让原先 X7 P3 消失。
```

### 12.2 optimizer-state transport fallback

新增代码修改：

```text
experiments/run_v132_loss_interface_basis_natural_update.py
  train_model 新增 optimizer_state_update / optimizer_state_scale。
  run_future_case 新增 optimizer_state_transport / optimizer_state_scale。
  CLI 新增 --optimizer-state-transport 与 --optimizer-state-scale。
  writeback trace 新增 optimizer_state_transport / optimizer_state_scale。
```

审计说明：

```text
1. 不改变 loss，不修改 sampler/class weight/dataset branch。
2. 不使用 validation/test/future outcome 选择 direction。
3. 只把 functional update 派生方向作为 AdamW exp_avg warm-start。
4. exp_avg_sq 使用 unit second moment，使它是谨慎 warm-start。
```

Smoke blocker 与修复：

```text
初次 smoke 因 `p in params` 触发 Tensor equality broadcast 报错。
修复为 `id(p) in param_ids`。
```

optimizer-state scale=0.20：

```text
route = R3-P3OnlySyntheticFailed
basis_natural_p3_pass_count = 2
synthetic_task_success_count = 2
optimizer_state_transport_rows = 42
```

optimizer-state scale=1.0：

```text
route = R3-P3OnlySyntheticFailed
basis_natural_p3_pass_count = 2
synthetic_task_success_count = 2
optimizer_state_transport_rows = 42
```

P3 pass rows 仍然只在 X7：

| run | candidate | task | update | loss | source_vs_best | NoiseSignalLeak delta | Reservoir delta |
|---|---|---|---|---|---:|---:|---:|
| state scale 0.20 | D-RAT26 | X7 | BN5 | Brier | 0.008902432409263972 | -0.004589974880218506 | -0.13720369338989258 |
| state scale 0.20 | D-RAT27 | X7 | BN5 | Brier | 0.008427021471569374 | -0.003060281276702881 | -0.41165590286254883 |
| state scale 1.0 | D-RAT26 | X7 | BN5 | Brier | 0.010440577932848971 | -0.004589974880218506 | -0.13720369338989258 |
| state scale 1.0 | D-RAT27 | X7 | BN5 | Brier | 0.008614227586040135 | -0.003060281276702881 | -0.41165590286254883 |

结论：

```text
optimizer-state transport 没有把 synthetic coverage 从 2/21 推进到 5/7。
Case C 要求的四项 fallback 均已执行。
```

## 13. 用户再次追问后的最终判断

再次复核后，最终判断仍是：

```text
v13.2 没有达成 S5；
已达到 S2-BasisNaturalP3；
没有达成 synthetic 5/7；
不允许进入 real short-run promotion path；
不允许 promotion；
合法 route 仍为 R3-P3OnlySyntheticFailed。
```

当前 no-go boundary 更清楚：

```text
1. 真实 basis-param writeback 已执行，不是 feature-table proxy。
2. BN5 + logit-energy projection 可在 X7 形成 P3，但不跨 task family。
3. 更低 future LR、optimizer-state transport scale 0.20/1.0 都不能扩大 synthetic coverage。
4. 继续推进需要新的 task-family robust basis metric 或回到 Non-RAT/Rational substrate design；继续调 BN5 局部尺度已经接近低价值搜索。
```

## 14. 用户再次追问后的 BN6 robust metric fallback

用户再次要求未达成则继续。本次按计划的 family-specific metric repair 方向新增并执行：

```text
BN6-ConsensusClippedEnergyNatural
```

代码修改：

```text
experiments/run_v132_loss_interface_basis_natural_update.py
  在 natural_update 中新增 BN6。
  BN6 使用 train-stream per-example gradient sign consensus mask。
  BN6 使用 sqrt(var)+rho diagonal metric。
  BN6 对 raw natural direction 做 75% quantile clip。
  BN6 继续使用 logit-energy orthogonal safety projection。
  metric_repair_attempted 覆盖 BN6。
```

合理性：

```text
1. 这是参数级 basis-natural metric repair，不是 readout proxy。
2. 方向来自 loss-interface-generic cotangent 与 train-stream per-example gradient。
3. 不使用 validation/test/future/query batch。
4. 不使用 CEp99/NLL/ECE/LineC 或 dataset name 生成方向。
5. 不改变任何 promotion gate。
```

执行 artifact：

```text
results/v13_2_deep_reset_loss_interface_basis_natural_update/fallback_consensus_metric_v132
```

结果：

```text
route = R2-BasisNaturalP3Failed
minimum_success = S1-EfficientControllableSubstrate
basis_natural_p3_rows = 42
basis_natural_p3_pass_count = 0
synthetic_rows = 21
synthetic_task_success_count = 0
full_basis_param_update_rows = 42
required_artifact_missing_count = 0
provenance_violation_count = 0
```

最接近但不能 pass：

| candidate | task | loss | source_vs_best | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | CEp99 delta | blocker |
|---|---|---|---:|---:|---:|---:|---:|---|
| D-RAT27 | X7 | Brier | 0.010846398201265095 | 1.4404830932617188 | 0.002047896385192871 | 0.031162261962890625 | -2.790827751159668 | NoiseSignalLeak / Reservoir |
| D-RAT26 | X7 | Brier | 0.0073690773573587 | 0.8988113403320312 | -0.0005670785903930664 | 0.061533451080322266 | -2.483287811279297 | Reservoir |
| D-RAT27 | X4 | CE | 0.0028140443622677314 | 2.827362060546875 | 0.03775596618652344 | -0.13692188262939453 | -0.16125726699829102 | source / NoiseSignalLeak |

主要失败类型：

| count | failure pattern |
|---:|---|
| 8 | source;coupling;noise;reservoir;cep99 |
| 5 | source;coupling;noise;reservoir;cep99;nll;ece |
| 4 | source;noise |
| 4 | source;coupling;noise;reservoir |

判断：

```text
BN6 没有达成 P3，也没有扩大 synthetic coverage。
它比 official BN5 更差：official BN4/BN5 仍有 2 个 P3 pass，而 BN6 为 0。
因此 official_v132 不被替换，最终可信 route 仍为 R3-P3OnlySyntheticFailed。
```

再次最终判断：

```text
v13.2 没有达成 S5；
已达到 S2-BasisNaturalP3；
没有达成 synthetic 5/7；
不允许进入 real short-run promotion path；
不允许 promotion；
合法 route 仍为 R3-P3OnlySyntheticFailed。
```

当前 no-go boundary：

```text
1. Case C 要求的 feature-proxy、norm/clipping、future optimizer sensitivity、optimizer-state transport 均已执行。
2. 进一步的 consensus/clipped robust basis metric 也未改善，说明问题不是单纯梯度符号不稳定或 update outlier。
3. 继续在 BN4/BN5/BN6 的局部 metric 变体上扩网格已经接近低价值搜索。
4. 下一步需要新的机制级方向：更强 task-family basis metric、真实 S2/Non-RAT substrate design，或重定义 synthetic objective；不能把 X7-only P3 写成 promotion。
```

## 15. 用户再次追问后的 future-case hyperparameter propagation 修复

用户再次要求未达成则继续。本次重新审计 runner 发现一个真实实现 blocker：

```text
run_future_case() 在 copied model 上重算 update 时硬编码：
eta = 0.02
rho = 1e-3
max_norm_ratio = 0.02
```

这意味着此前 metric/norm fallback 的 CLI 参数会进入初始 audit，但不会真实进入 future probe 的 copied-model writeback。该问题会影响 Case C 的 “update norm / clipping” 复核可信度。

本次代码修复：

```text
experiments/run_v132_loss_interface_basis_natural_update.py
  NaturalUpdate 新增 eta / rho / max_norm_ratio 字段。
  natural_update() 返回这些字段。
  run_future_case() 重算 copied-model update 时使用 update.eta / update.rho / update.max_norm_ratio。
  SNROnlyControl 使用 matched eta/rho/max_norm_ratio。
```

合法性：

```text
1. 不改变 gate。
2. 不引入 validation/test/future/query batch 作为方向源。
3. 不使用 CEp99/NLL/ECE/LineC 作为方向源。
4. 这是执行语义修复，使 CLI fallback 参数真实生效。
```

### 15.1 修复后 metric/norm fallback

artifact：

```text
results/v13_2_deep_reset_loss_interface_basis_natural_update/fallback_metric_norm_repaired_v132
```

结果：

```text
route = R3-P3OnlySyntheticFailed
minimum_success = S2-BasisNaturalP3
basis_natural_p3_rows = 84
basis_natural_p3_pass_count = 2
synthetic_rows = 21
synthetic_task_success_count = 2
full_basis_param_update_rows = 84
required_artifact_missing_count = 0
provenance_violation_count = 0
```

pass rows：

| candidate | task | update | loss | source_vs_best | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | CEp99 delta |
|---|---|---|---|---:|---:|---:|---:|---:|
| D-RAT26 | X7 | BN5 | Brier | 0.007548595096722038 | 1.9736995697021484 | -0.0045899152755737305 | -0.13720464706420898 | -2.2259368896484375 |
| D-RAT27 | X7 | BN5 | Brier | 0.008000795437745678 | 4.89793586730957 | -0.0030603408813476562 | -0.41165685653686523 | -2.750546455383301 |

解释：

```text
修复后 metric/norm fallback 仍只在 X7 打开 2 个 P3 pass；
没有推进 synthetic 5/7。
```

### 15.2 修复后 official_v132 重跑

结果：

```text
route = R3-P3OnlySyntheticFailed
minimum_success = S2-BasisNaturalP3
official_success_reached = 0
promotion_allowed = 0
basis_natural_p3_rows = 84
basis_natural_p3_pass_count = 2
synthetic_rows = 21
synthetic_task_success_count = 2
synthetic_5of7_pass = 0
real_short_run_pass_count = 0
mlp_analog_pass_count = 0
full_basis_param_update_rows = 84
required_artifact_missing_count = 0
provenance_violation_count = 0
```

official pass rows：

| candidate | task | update | loss | source_vs_best | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | CEp99 delta |
|---|---|---|---|---:|---:|---:|---:|---:|
| D-RAT26 | X7 | BN5 | Brier | 0.011487269300395009 | 1.9736881256103516 | -0.004589974880218506 | -0.13720369338989258 | -2.2259368896484375 |
| D-RAT27 | X7 | BN5 | Brier | 0.009255758784707857 | 4.897928237915039 | -0.003060281276702881 | -0.41165590286254883 | -2.7505455017089844 |

最终判断仍是：

```text
v13.2 没有达成 S5；
已达到 S2-BasisNaturalP3；
没有达成 synthetic 5/7；
不允许进入 real short-run promotion path；
不允许 promotion；
合法 route = R3-P3OnlySyntheticFailed。
```

更新后的 no-go boundary：

```text
1. hyperparameter propagation blocker 已修复，metric/norm fallback 已可信重跑。
2. BN5 的 P3 仍集中在 X7；X1..X6 仍无法同时满足 source/CouplingR2/NoiseSignalLeak/Reservoir/tail gate。
3. BN6 consensus clipped metric 没有改善，optimizer-state transport 也没有改善。
4. 当前继续推进需要新机制，而不是 BN4/BN5/BN6 局部网格：例如 task-family aware basis metric、非 Rational S1 substrate rescue，或重新设计 synthetic proof 的真实参数目标。
```

## 16. 用户再次追问后的 BN7 output-Jacobian basis metric fallback

用户再次要求未达成则继续。本次不再继续调 BN4/BN5/BN6 局部尺度，而是实现 v13.2 公式中更正统的 basis tangent metric：

```text
M_basis = E[J_theta(x)^T J_theta(x)] + rho I
```

代码修改：

```text
experiments/run_v132_loss_interface_basis_natural_update.py
  新增 BN7-JacobianDiagEnergyNatural。
  对 train-stream 每个样本/输出维度计算真实参数 output-Jacobian diagonal。
  BN7 的 metric 使用 label-free output-Jacobian diagonal。
  BN7 的 mu 仍来自 loss-interface-generic cotangent。
  BN7 继续使用 SNR gate 与 logit-energy orthogonal safety projection。
```

合理性：

```text
1. 这是计划第 7 节 M_basis 的更直接实现。
2. metric 不读取 label、CE tail、validation/test/future/query batch。
3. 不改变 gate。
4. 仍是 full basis parameter update，不是 feature-table proxy。
```

执行 artifact：

```text
results/v13_2_deep_reset_loss_interface_basis_natural_update/fallback_jacobian_diag_metric_v132
```

结果：

```text
route = R2-BasisNaturalP3Failed
minimum_success = S1-EfficientControllableSubstrate
basis_natural_p3_rows = 42
basis_natural_p3_pass_count = 0
synthetic_rows = 21
synthetic_task_success_count = 0
full_basis_param_update_rows = 42
required_artifact_missing_count = 0
provenance_violation_count = 0
metric_condition_min = 1.3862051963806152
metric_condition_max = 3.5969626903533936
metric_condition_mean = 2.0917059864316667
```

最接近但不能 pass：

| candidate | task | loss | source_vs_best | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | CEp99 delta | blocker |
|---|---|---|---:|---:|---:|---:|---:|---|
| D-RAT26 | X7 | Brier | 0.0018138602472285992 | 4.893610000610352 | 0.028290987014770508 | -0.36788415908813477 | -2.9503769874572754 | source / NoiseSignalLeak |
| D-RAT28 | X6 | Brier | 0.0006062911758760239 | -152.47772979736328 | 0.03697788715362549 | 5.895464897155762 | 1.275080680847168 | source / coupling / noise / reservoir / tail |
| D-RAT28 | X2 | Brier | 0.00014374580432294226 | -3.7359256744384766 | -0.043040692806243896 | 0.5299921035766602 | 0.745448112487793 | source / coupling / reservoir / tail |

主要失败类型：

| count | failure pattern |
|---:|---|
| 12 | source;coupling;noise;reservoir;cep99 |
| 6 | source;noise |
| 5 | source;coupling;noise;reservoir |
| 4 | source;coupling;noise;reservoir;cep99;nll;ece |

判断：

```text
BN7 没有达成 P3，也没有扩大 synthetic coverage。
它证明问题不是单纯 metric condition；即使用更接近计划公式的 output-Jacobian metric，source_vs_best 仍不足或 LineC/Reservoir/tail 坏化。
official_v132 不被替换，最终可信 route 仍为 R3-P3OnlySyntheticFailed。
```

再次最终判断：

```text
v13.2 没有达成 S5；
已达到 S2-BasisNaturalP3；
没有达成 synthetic 5/7；
不允许进入 real short-run promotion path；
不允许 promotion；
合法 route 仍为 R3-P3OnlySyntheticFailed。
```

当前 no-go boundary：

```text
1. 当前失败不再能归因于 readout proxy、未写真实参数、future hyperparameter 未传播、缺少 optimizer-state transport、缺少 robust sign consensus、或缺少 label-free output-Jacobian metric。
2. BN4/BN5/BN6/BN7 都不能把 coverage 推过 2/21。
3. 继续在同一 basis-natural diagonal/local metric family 上扩变体已经是低价值搜索。
4. 下一步需要新机制级计划：非对角/低秩 task-family metric、真实 Non-RAT/S2 substrate rescue，或重新定义 synthetic proof 的参数目标。
```

## 17. 用户再次追问后的 stop-contract 复核

用户再次要求“未达成则继续”。本次重新对照 v13.2 计划 stop rule、`v132_next_hypothesis_queue.md` 与最终 route。

最终 artifact：

```text
route = R3-P3OnlySyntheticFailed
minimum_success = S2-BasisNaturalP3
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
substrate_s1_count = 11
substrate_s2_healthy_base_count = 0
basis_natural_p3_pass_count = 2
synthetic_task_success_count = 2
synthetic_5of7_pass = 0
real_short_run_pass_count = 0
mlp_analog_pass_count = 0
full_basis_param_update_rows = 84
provenance_violation_count = 0
manifest_rows = 30
missing_rows = 0
```

计划要求的 fallback 覆盖情况：

```text
Case B：
1. per-example gradient SNR histogram 已输出。
2. basis metric condition 已输出。
3. safety projection rejection reason 已输出。
4. family-specific metric repair 已尝试：BN4/BN5/BN6/BN7。
5. MLP analog 已执行，mlp_analog_pass_count=0。

Case C：
1. feature-table proxy 检查已完成：full_basis_param_update_rows=84, readout_feature_proxy_only=0。
2. optimizer-state transport 已执行，scale=0.20/1.0 均未扩大 coverage。
3. update norm / clipping 已修复 hyperparameter propagation blocker 后重跑。
4. future probe optimizer sensitivity 已执行，低 LR 未扩大 coverage。
```

`v132_next_hypothesis_queue.md` 对照：

```text
1. inspect per-example gradient SNR and basis metric conditioning: 已完成。
2. add optimizer-state transport and tighter norm clipping: 已完成。
3. no S1 for non-RAT -> return to kernel/lifetime/task-health substrate design: 这是新的 substrate design 工作，不是当前 v13.2 basis-natural functional runner 内的局部修复。
```

最终判断仍是：

```text
v13.2 没有达成 S5；
已达到 S2-BasisNaturalP3；
没有达成 synthetic 5/7；
不允许进入 real short-run promotion path；
不允许 promotion；
合法 route = R3-P3OnlySyntheticFailed。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改 gate。原因不是轻易放弃，而是当前计划内 functional fallback 已闭合：继续在 BN4/BN5/BN6/BN7 的 diagonal/local basis-natural family 上排列变体已经是低价值搜索。我现在不确定如何在现有 v13.2 代码路径内安全实现新的 Non-RAT substrate design 或非对角 task-family metric，并保证不把 proxy 写成 success。
