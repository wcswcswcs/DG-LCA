# DG-KAN v13.6 PopRiskSNR BasisCoverBoundary 实验结果复盘

生成时间：2026-05-28（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 smoke / diagnostic / fallback 写成 promotion。

## 1. 计划理解

v13.6 的目标是把 v13.5 已经失败的 oracle target route 替换为：

```text
population-risk signal selection
+
basis-cover boundary stabilization
```

核心 functional source 不再是 future/oracle DeltaZ，而是 train-stream current batch 的 per-example gradient statistics：

```text
g_i = J_theta(x_i)^T delta_i
mu_k = mean_i g_i,k
sigma_k^2 = var_i g_i,k
SNR_k = mu_k^2 / (sigma_k^2/(b-1) + eps)
```

硬约束：

```text
1. strict FC-PureKAN / no active B-spline budget。
2. 不使用 label-informed initialization。
3. 不使用 teacher / distillation / sampler / class weight / dataset-name branch。
4. functional direction 不使用 validation/test/future outcome/query batch。
5. CEp99/NLL/ECE/LineC hard target 只能作为 audit/gate，不能生成方向。
6. 必须比较 MLP parameter-SNR、Rational parameter-SNR、Rational basis-channel SNR、basis-cover boundary、Non-RAT substrate-SNR scouts 与 MLP analog。
```

## 2. 本轮代码修改

新增文件：

```text
experiments/run_v136_poprisk_snr_basis_cover_boundary.py
```

关键实现：

```text
1. loss_interface_cotangent_per_example(): 通过 generic loss interface 生成 per-example output cotangent。
2. collect_per_example_gradients(): 逐样本计算 J_theta(x_i)^T delta_i。
3. make_snr_update(): 实现 Hard / Soft / EMA3RoleNorm SNR gate、parameter/basis-channel update、cover guard。
4. run_future_case(): 执行 Functional 与 matched controls。
5. run_substrate_scouts(): 执行 Rational + Fourier + Chebyshev + RBF substrate-SNR scouts。
6. build_route(): 输出 R4/R5/R6/R7/R8 no-go route 与 gate。
```

合法性说明：

```text
1. update 写回真实 named parameters，记录 writeback sha。
2. readout_feature_proxy_only=0。
3. feature_table_proxy_only=0。
4. CEp99/NLL/ECE/LineC 未用于方向，只用于 audit/gate。
5. validation/future outcome 不参与方向生成。
6. Brier/CE 都走同一个 loss-interface cotangent surface；不是 CE-tail 专用公式。
```

代码审计 line range 以 artifact 为准：

```text
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_code_review_manifest.csv
```

## 3. Smoke

artifact：

```text
results/v13_6_poprisk_snr_basis_cover_boundary/smoke_v136
```

结果：

```text
route = R4-PopRiskSNRNoGo
required_artifact_missing_count = 0
synthetic_rows = 2
control_rows = 8
full_parameter_update_rows = 2
forbidden_information_violation_count = 0
```

解释：

```text
smoke 只证明 v13.6 runner、per-example gradient、writeback、controls、manifest 与 code packet 可执行；
不能 promotion。
```

## 4. Official v13.6

执行规模：

```text
synthetic_tasks = X1..X7
seeds = 0,1
loss_interfaces = CE,Brier
methods =
  MLP-ParameterSNR
  MLP-HiddenChannelSNR
  KAN-ParameterSNR
  KAN-BasisChannelSNR
  KAN-BasisCoverBoundary
controls =
  Functional
  AdamW
  NoOpMatchedOverhead
  RandomMatchedNorm
  AdamWParallelDirection
  ShuffledPerExampleGradient
  SameMaskRandomSign
  SameActiveFractionRandomMask
```

final official route：

```text
route = R4-PopRiskSNRNoGo
minimum_success = S1-EfficientSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
substrate_s1_count = 11
substrate_snr_gate_pass_count = 1
nonrat_substrate_snr_gate_pass_count = 0
synthetic_rows = 140
control_rows = 1120
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
mlp_snr_pass_rows = 0
kan_parameter_snr_pass_rows = 0
kan_basis_snr_pass_rows = 0
cover_boundary_rows = 28
cover_boundary_accept_count = 22
full_parameter_update_rows = 134
forbidden_information_violation_count = 0
provenance_violation_count = 0
```

最接近但不能 pass 的 rows：

| method | task | seed | loss | source_vs_best | active fraction | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | CEp99 delta |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| KAN-BasisCoverBoundary | X4 | 0 | Brier | 0.0007192629328190044 | 0.6193181872367859 | 2.0438804626464844 | -0.01887744665145874 | -0.34347105026245117 | -0.9552578926086426 |
| KAN-BasisCoverBoundary | X4 | 0 | CE | 0.0001898873080178505 | 0.46590909361839294 | 2.0438709259033203 | -0.018877089023590088 | -0.34347105026245117 | -0.95526123046875 |
| MLP-ParameterSNR | X1 | 0 | Brier | 0.00000020563967318299597 | 0.39123600721359253 | -3.0977935791015625 | 0.043044090270996094 | 0.1884288787841797 | 0.7360701560974121 |

解释：

```text
1. 没有任何 official synthetic pass。
2. 没有任何 exploration pass。
3. 最接近 row 的 LineC/tail 很干净，但 source_vs_best_control 只有 0.000719 < 0.002 exploration gate，更低于 0.005 official gate。
4. 因 synthetic_5of7_pass=0，不允许 real short-run，不允许 promotion。
```

## 5. Method-level summary

official_v136：

| method | rows | best source_vs_best | mean active fraction | official pass | exploration pass |
|---|---:|---:|---:|---:|---:|
| KAN-BasisCoverBoundary | 28 | 0.0007192629328190044 | 0.5848214392151151 | 0 | 0 |
| MLP-ParameterSNR | 28 | 0.00000020563967318299597 | 0.39800976536103655 | 0 | 0 |
| MLP-HiddenChannelSNR | 28 | -0.00010800396597619472 | 0.8143262724791255 | 0 | 0 |
| KAN-BasisChannelSNR | 28 | -0.00023018903925793067 | 0.5286120335970607 | 0 | 0 |
| KAN-ParameterSNR | 28 | -0.0013982748115428344 | 0.3643937717591013 | 0 | 0 |

主要 failure pattern：

| count | failure pattern |
|---:|---|
| 20 | source;noise |
| 16 | source;coupling;noise;reservoir;cep99;nll;ece |
| 14 | source |
| 9 | source;coupling;reservoir;cep99;nll |
| 7 | source;noise;ece |
| 6 | source;coupling;noise;reservoir;nll;ece |
| 6 | source;cep99 |

判断：

```text
主要 blocker 是 source_vs_best_control 不足；
有些 rows 的 CouplingR2 / Noise / Reservoir / CE tail 同时坏化。
```

## 6. Basis-cover boundary

artifact：

```text
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_basis_cover_boundary.csv
```

结果：

```text
cover_boundary_rows = 28
cover_boundary_accept_count = 22
cover_boundary_reject_count = 6
```

解释：

```text
1. cover guard 没有把所有 update 变成 no-op，因此 route 不是 R7-BasisCoverOverConstrained。
2. cover accepted 的 rows 仍没有 synthetic success。
3. 失败不能简单归因于 basis-cover boundary 过严。
```

## 7. Substrate-SNR scouts

artifact：

```text
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_substrate_snr_gate.csv
```

结果：

| family | candidate | workspace pass | channel_snr_entropy | active fraction | substrate SNR gate | status |
|---|---|---:|---:|---:|---:|---|
| D-RAT | D-RAT26-TangentTrustRegionNoCE | 1 | 0.8318917155265808 | 0.5852273106575012 | 1 | SubstrateSNRGatePass |
| D-FOU | D-FOU14-SincosSharedWorkspace-K2 | 0 | 0.9164406657218933 | 0.875 | 0 | WorkspaceOnly_NotFunctionalSubstrate |
| D-CHE | D-CHE16-DegreeEnergyDampingSubstrate | 0 | 0.9597955942153931 | 1.0 | 0 | WorkspaceOnly_NotFunctionalSubstrate |
| D-RBF | D-RBF11-CompactExpressionRepair-Monitor | 0 | 0.8962802886962891 | 0.8125 | 0 | WorkspaceOnly_NotFunctionalSubstrate |

解释：

```text
1. Rational substrate-SNR gate 通过 1 个。
2. Non-RAT Fourier / Chebyshev / RBF 均未通过 substrate-SNR gate。
3. Non-RAT 失败主要来自 workspace/task-health gate，不是 SNR telemetry 完全不可计算。
```

## 8. Fallback 1：Soft / EMA3RoleNorm

artifact：

```text
results/v13_6_poprisk_snr_basis_cover_boundary/fallback_soft_ema_role_v136
```

执行内容：

```text
snr_variants = Soft,EMA3RoleNorm
loss_interface = CE
future_steps = 8
```

结果：

```text
route = R4-PopRiskSNRNoGo
synthetic_rows = 140
control_rows = 1120
official pass = 0
exploration pass = 0
required_artifact_missing_count = 0
best source_vs_best_control = 0.0004920390084638543
best row = KAN-BasisChannelSNR / EMA3RoleNorm / X2 / seed 0 / CE
```

解释：

```text
soft q、EMA mu/sigma、role-wise threshold normalization 没有打开 exploration pass。
```

## 9. Fallback 2：High tau + active cap

artifact：

```text
results/v13_6_poprisk_snr_basis_cover_boundary/fallback_tau_cap_v136
```

执行内容：

```text
snr_tau = 4.0
active_fraction_cap = 0.25
loss_interface = CE
```

结果：

```text
route = R4-PopRiskSNRNoGo
synthetic_rows = 70
control_rows = 560
official pass = 0
exploration pass = 0
required_artifact_missing_count = 0
best source_vs_best_control = 0.0002900220635735917
best row = KAN-BasisCoverBoundary / Hard / X4 / seed 0 / CE
```

解释：

```text
更高 tau 与 active cap 没有打开 exploration pass；
active fraction 降低后 source 反而没有改善。
```

## 10. Fallback 3：eta / norm sensitivity

artifact：

```text
results/v13_6_poprisk_snr_basis_cover_boundary/fallback_eta_norm_v136
```

执行内容：

```text
functional_eta = 0.05
max_update_norm_ratio = 0.05
loss_interface = CE
```

结果：

```text
route = R4-PopRiskSNRNoGo
synthetic_rows = 56
control_rows = 448
official pass = 0
exploration pass = 0
required_artifact_missing_count = 0
best source_vs_best_control = 0.00021401959926648298
best row = MLP-ParameterSNR / Hard / X4 / seed 0 / CE
```

解释：

```text
更大 functional step / norm cap 没有打开 exploration pass；
失败不能简单归因于 update 太弱。
```

## 11. Required artifacts

official_v136 required manifest：

```text
manifest_rows = 35
required_artifact_missing_count = 0
```

主要产物：

```text
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_route_decision.json
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_progress_table.csv
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_code_review_manifest.csv
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_loss_interface_audit.csv
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_per_example_gradient_stats.csv
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_snr_parameter_update.csv
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_snr_basis_channel_update.csv
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_basis_cover_boundary.csv
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_substrate_snr_gate.csv
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_synthetic_family_results.csv
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_mlp_analog_results.csv
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_linec_audit.csv
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_controls.csv
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_failure_table.csv
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_no_go_boundary.md
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_next_hypothesis_queue.md
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_required_manifest.csv
results/v13_6_poprisk_snr_basis_cover_boundary/official_v136/v136_code_review_packet.zip
```

追加审计修复：

```text
experiments/run_v136_poprisk_snr_basis_cover_boundary.py
  finalize_only() 读取 v136_controls.csv，并把 control_rows 写回 route。

原因：
  初次 finalize-only 刷新后 route 中 control_rows 为空字符串；
  CSV artifact 本体存在 1120 rows，但 route summary 不够清晰。

修复后：
  control_rows = 1120
  code_review_packet_entries = 7
  code_review_packet_sha256 以 v136_route_decision.json 为准
```

## 12. 最终科学结论

v13.6 没有达成 S3/S5；最终合法 route：

```text
R4-PopRiskSNRNoGo
minimum_success = S1-EfficientSubstrate
promotion_allowed = 0
```

已闭合事实：

```text
1. per-example gradient SNR implementation 可执行，required artifacts 缺失为 0。
2. MLP-ParameterSNR / MLP-HiddenChannelSNR 没有 official 或 exploration pass。
3. Rational KAN-ParameterSNR / KAN-BasisChannelSNR / KAN-BasisCoverBoundary 没有 official 或 exploration pass。
4. cover boundary 没有过度拒绝所有 update：22/28 accepted，但仍无 pass。
5. Soft / EMA3RoleNorm / role-wise normalization 没有改善到 exploration gate。
6. high tau + active cap 没有改善到 exploration gate。
7. eta / norm sensitivity 没有改善到 exploration gate。
8. Non-RAT Fourier / Chebyshev / RBF substrate-SNR scouts 均未打开 substrate-SNR gate。
9. forbidden audit 为 0，readout_feature_proxy_only=0，feature_table_proxy_only=0。
```

no-go boundary：

```text
1. 当前失败不是 v13.5 oracle target 的遗留问题；v13.6 已换成 train-stream population-risk SNR。
2. 当前失败也不是单纯 SNR active fraction 过低/过高、soft gate 缺失、EMA/role-wise 缺失、cover boundary 过严或 step 太弱。
3. MLP 与 KAN 都没有 pass，因此本轮不能主张 KAN-specific basis-coordinate advantage，也不能主张 generic optimizer positive。
4. Rational substrate-SNR gate 仍可通过，但 functional value 没有转化为 synthetic success。
5. Non-RAT 仍是 WorkspaceOnly_NotFunctionalSubstrate。
6. 合法下一步应进入新的 substrate/base architecture 或重新定义可价值 functional source；不能把本轮 SNR diagnostic 写成 promotion。
```

最终判断：

```text
v13.6 未达成目标；
不允许 promotion；
不允许 real short-run；
允许 final stop，原因是计划内 MLP/KAN SNR baseline、basis-channel SNR、cover boundary、Non-RAT scouts、MLP analog 与 fallback 均已闭合为 no-go。
```

## 13. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v13.6 是否达成目标，若未达成则继续。本次重新读取 v13.6 计划 `5.7`、`14`、`15` 的 fallback / stop rule，并复核 official route 与三个 fallback artifact。

最终 artifact：

```text
route = R4-PopRiskSNRNoGo
minimum_success = S1-EfficientSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
substrate_s1_count = 11
substrate_snr_gate_pass_count = 1
nonrat_substrate_snr_gate_pass_count = 0
synthetic_rows = 140
control_rows = 1120
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
mlp_snr_pass_rows = 0
kan_parameter_snr_pass_rows = 0
kan_basis_snr_pass_rows = 0
cover_boundary_rows = 28
cover_boundary_accept_count = 22
full_parameter_update_rows = 134
forbidden_information_violation_count = 0
provenance_violation_count = 0
readout_feature_proxy_only = 0
feature_table_proxy_only = 0
```

计划 stop rule 对照：

```text
1. active fraction 太低时的 soft q / EMA3 / role-wise threshold normalization 已执行：fallback_soft_ema_role_v136，仍为 R4，pass=0。
2. active fraction 太高时的 tau increase / active cap / SameActiveFractionRandomMask 已执行：fallback_tau_cap_v136，仍为 R4，pass=0。
3. step/norm sensitivity 已执行：fallback_eta_norm_v136，仍为 R4，pass=0。
4. MLP parameter-SNR、MLP hidden-channel SNR、KAN parameter-SNR、KAN basis-channel SNR、KAN basis-cover boundary 均没有 official / exploration pass。
5. Non-RAT Fourier / Chebyshev / RBF substrate-SNR scouts 均未打开 substrate-SNR gate。
6. v13.6 第 14.2 节写明 parameter-SNR 与 basis-SNR 在 MLP/KAN 上都 fail 时，合法 route = R4-PopRiskSNRNoGo。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改 gate。原因不是轻易放弃，而是当前计划内继续条件已经闭合：v13.6 已完成 MLP/KAN SNR baseline、basis-channel SNR、basis-cover boundary、Non-RAT scouts、matched controls、MLP analog 与计划指定 fallback。继续在同一 SNR mask / tau / EMA / norm / cover-boundary 局部网格上扩展，已经不是计划推荐方向，并会增加把 diagnostic 编造成 success 的风险。

最终判断仍是：

```text
v13.6 没有达成 S3/S5；
只达到 S1-EfficientSubstrate；
不允许 promotion；
不允许 real short-run；
合法 route = R4-PopRiskSNRNoGo；
允许 final stop。
```

## 14. 用户再次追问后的 Line S substrate/base architecture scout 补跑

用户再次要求“未达成则继续”。本次重新审计 v13.6 计划后发现：第 8 节 `Line S：Substrate / base architecture reset` 列出了 RAT/CHE/FOU/RBF/WAV-SNR1/2/3 candidate families。此前 official 的 `v136_substrate_snr_gate.csv` 只覆盖：

```text
D-RAT26-TangentTrustRegionNoCE
D-FOU14-SincosSharedWorkspace-K2
D-CHE16-DegreeEnergyDampingSubstrate
D-RBF11-CompactExpressionRepair-Monitor
```

这意味着 v13.6 Line S candidate 名称覆盖不完整。本次按计划补齐 substrate/base architecture scout，但不降低任何 gate，不把 scout row 写成 promotion。

代码修改：

```text
experiments/run_v136_poprisk_snr_basis_cover_boundary.py
  新增 LINE_S_CANDIDATE_SPECS：
    RAT-SNR1/2/3
    CHE-SNR1/2/3
    FOU-SNR1/2/3
    RBF-SNR1/2/3
    WAV-SNR1/2/3

  新增 make_line_s_model()：
    用 mapped_candidate_id 构造已有 audited primitive，不声明新 kernel。

  新增 line_s_candidate_rows()：
    把 Line S 设计名映射到 v12.35/v1235 已有 candidate，并保留 workspace/task metrics。

  修改 run_substrate_scouts()：
    追加 Line S candidate rows；
    输出 mapped_candidate_id / line_s_candidate / line_s_mapping_no_new_kernel_claim。

  修改 code_review_manifest：
    增加 Line S mapping / model builder / substrate-SNR gate surface。
```

合法性说明：

```text
1. Line S candidate 是 substrate scout，不是 functional promotion。
2. mapped_candidate_id 显式写入 artifact，避免把设计名误读成新实现成功。
3. 仍按 v13.6 Substrate-SNR gate 判定：step / memory / task / AUC-time / entropy / cover rejection。
4. 不使用 validation/test/future outcome 生成方向。
5. 不使用 CEp99/NLL/ECE/LineC 生成方向。
6. 不修改 synthetic 5/7、LineC、tail metric 或 real short-run gate。
```

执行 artifact：

```text
results/v13_6_poprisk_snr_basis_cover_boundary/fallback_line_s_substrate_scout_v136
```

执行规模：

```text
synthetic_task = X1
seed = 0
method = KAN-BasisChannelSNR
loss = CE
operator_batch_size = 8
checkpoint_steps = 4
future_steps = 4
controls = Functional, SameActiveFractionRandomMask
```

结果：

```text
route = R4-PopRiskSNRNoGo
required_artifact_missing_count = 0
synthetic_rows = 1
synthetic_task_success_count = 0
full_parameter_update_rows = 1
substrate_snr_gate_rows = 19
line_s_candidate_rows = 15
substrate_snr_gate_pass_count = 4
nonrat_substrate_snr_gate_pass_count = 0
```

Family-level substrate-SNR scout summary：

| family | rows | pass | status |
|---|---:|---:|---|
| D-RAT | 4 | 4 | SubstrateSNRGatePass |
| D-CHE | 4 | 0 | WorkspaceOnly_NotFunctionalSubstrate |
| D-FOU | 4 | 0 | WorkspaceOnly_NotFunctionalSubstrate |
| D-RBF | 4 | 0 | WorkspaceOnly_NotFunctionalSubstrate |
| D-WAV | 3 | 0 | WorkspaceOnly_NotFunctionalSubstrate |

通过的 rows 全部是 Rational：

| candidate | mapped_candidate_id | substrate-SNR pass |
|---|---|---:|
| D-RAT26-TangentTrustRegionNoCE | D-RAT26-TangentTrustRegionNoCE | 1 |
| RAT-SNR1-denSlopeTelemetrySubstrate | D-RAT27-DenSlopeGuardNoCE | 1 |
| RAT-SNR2-groupDiversityFloorSubstrate | D-RAT28-GroupDiversityPreservingRational | 1 |
| RAT-SNR3-readoutRationalDecoupledSubstrate | D-RAT35-ReadoutRationalDecoupleNoCE | 1 |

Non-RAT 失败边界：

```text
1. CHE/FOU/RBF/WAV 的 channel_snr_entropy 通常不为 0，说明 telemetry 不是全退化。
2. 但它们仍被 workspace / efficiency / task-health gate 拒绝。
3. 所有 Non-RAT Line S rows 的 substrate_snr_gate_pass = 0。
4. 因此 v13.6 的 substrate/base architecture scout 没有找到可进入 functional proof 的 Non-RAT substrate。
```

最终判断仍是：

```text
v13.6 没有达成 S3/S5；
Rational substrate-SNR 可见，但 functional synthetic success 仍为 0；
Non-RAT Line S substrate-SNR pass = 0；
不允许 promotion；
不允许 real short-run；
合法 route 仍为 R4-PopRiskSNRNoGo。
```

## 15. Line S 合并后的 official_v136 重跑

为了让 official_v136 本体也直接包含 Line S substrate/base architecture scout，本次在代码补丁后用原 official 矩阵重跑。

执行规模：

```text
synthetic_tasks = X1..X7
seeds = 0,1
methods = MLP-ParameterSNR,MLP-HiddenChannelSNR,KAN-ParameterSNR,KAN-BasisChannelSNR,KAN-BasisCoverBoundary
snr_variant = Hard
loss_interfaces = CE,Brier
controls = Functional,AdamW,NoOpMatchedOverhead,RandomMatchedNorm,AdamWParallelDirection,ShuffledPerExampleGradient,SameMaskRandomSign,SameActiveFractionRandomMask
checkpoint_steps = 8
future_steps = 10
operator_batch_size = 16
```

重跑后的 official route：

```text
route = R4-PopRiskSNRNoGo
minimum_success = S1-EfficientSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
substrate_snr_gate_pass_count = 4
nonrat_substrate_snr_gate_pass_count = 0
synthetic_rows = 140
control_rows = 1120
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
mlp_snr_pass_rows = 0
kan_parameter_snr_pass_rows = 0
kan_basis_snr_pass_rows = 0
full_parameter_update_rows = 134
```

official_v136 substrate-SNR gate coverage：

```text
substrate_snr_gate_rows = 19
line_s_candidate_rows = 15
```

通过的 rows 全部是 Rational：

| candidate | mapped_candidate_id | line_s_candidate | substrate-SNR pass |
|---|---|---:|---:|
| D-RAT26-TangentTrustRegionNoCE | D-RAT26-TangentTrustRegionNoCE | 0 | 1 |
| RAT-SNR1-denSlopeTelemetrySubstrate | D-RAT27-DenSlopeGuardNoCE | 1 | 1 |
| RAT-SNR2-groupDiversityFloorSubstrate | D-RAT28-GroupDiversityPreservingRational | 1 | 1 |
| RAT-SNR3-readoutRationalDecoupledSubstrate | D-RAT35-ReadoutRationalDecoupleNoCE | 1 | 1 |

Non-RAT official Line S 结果：

| family | rows | pass | status |
|---|---:|---:|---|
| D-CHE | 4 | 0 | WorkspaceOnly_NotFunctionalSubstrate |
| D-FOU | 4 | 0 | WorkspaceOnly_NotFunctionalSubstrate |
| D-RBF | 4 | 0 | WorkspaceOnly_NotFunctionalSubstrate |
| D-WAV | 3 | 0 | WorkspaceOnly_NotFunctionalSubstrate |

重跑后最接近但不能 pass 的 row：

| method | task | seed | loss | source_vs_best | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | CEp99 delta | pass |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| KAN-BasisCoverBoundary | X4 | 0 | CE | 0.0003762373062765681 | 2.0438709259033203 | -0.018877089023590088 | -0.34347105026245117 | -0.95526123046875 | 0 |

主要失败类型：

| count | failure pattern |
|---:|---|
| 20 | source;noise |
| 16 | source;coupling;noise;reservoir;cep99;nll;ece |
| 14 | source |
| 9 | source;coupling;reservoir;cep99;nll |
| 7 | source;noise;ece |

判断：

```text
1. official_v136 已经闭合 v13.6 第 8 节 Line S coverage，不再只依赖 fallback 目录。
2. Rational Line S candidate 可过 substrate-SNR gate，但 source_vs_best 仍远低于 0.005 official gate。
3. Non-RAT CHE/FOU/RBF/WAV 仍被 workspace / efficiency / task-health gate 拒绝。
4. synthetic task success 仍为 0/7，不允许 real short-run。
5. 最终合法 route 不变：R4-PopRiskSNRNoGo。
```

## 16. 用户再次追问后的 Soft / EMA3RoleNorm CE+Brier 补跑

用户再次要求“未达成则继续”。本次继续复核计划 5.7 的 soft q / EMA3 / role-wise threshold normalization fallback。此前 `fallback_soft_ema_role_v136` 只使用 CE；official_v136 使用 CE/Brier。因此本次补跑 CE+Brier，排除 fallback loss-interface 覆盖不完整的解释。

执行 artifact：

```text
results/v13_6_poprisk_snr_basis_cover_boundary/fallback_soft_ema_role_ce_brier_v136
```

执行规模：

```text
synthetic_tasks = X1..X7
seeds = 0,1
methods = MLP-ParameterSNR,MLP-HiddenChannelSNR,KAN-ParameterSNR,KAN-BasisChannelSNR,KAN-BasisCoverBoundary
snr_variants = Soft,EMA3RoleNorm
loss_interfaces = CE,Brier
controls = Functional,AdamW,NoOpMatchedOverhead,RandomMatchedNorm,AdamWParallelDirection,ShuffledPerExampleGradient,SameMaskRandomSign,SameActiveFractionRandomMask
checkpoint_steps = 8
future_steps = 8
operator_batch_size = 16
```

结果：

```text
route = R4-PopRiskSNRNoGo
promotion_allowed = 0
synthetic_rows = 280
control_rows = 2240
synthetic_task_success_count = 0
mlp_snr_pass_rows = 0
kan_parameter_snr_pass_rows = 0
kan_basis_snr_pass_rows = 0
cover_boundary_accept_count = 49
cover_boundary_rows = 56
substrate_snr_gate_pass_count = 4
nonrat_substrate_snr_gate_pass_count = 0
required_artifact_missing_count = 0
```

最接近但不能 pass 的 rows：

| method | task | seed | loss | variant | source_vs_best | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | CEp99 delta | pass |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| KAN-BasisChannelSNR | X2 | 0 | CE | EMA3RoleNorm | 0.0005674217174649332 | 11.909942626953125 | -0.013969182968139648 | -0.8789176940917969 | -0.9672317504882812 | 0 |
| KAN-BasisCoverBoundary | X3 | 1 | Brier | EMA3RoleNorm | 0.0005254489736433374 | 6.925106048583984 | -0.0006207823753356934 | -0.6136655807495117 | -0.9633407592773438 | 0 |
| KAN-BasisCoverBoundary | X4 | 0 | CE | EMA3RoleNorm | 0.00046121535849860673 | 1.7122459411621094 | -0.01597738265991211 | -0.306429386138916 | -0.8939476013183594 | 0 |

主要 failure pattern：

| count | failure pattern |
|---:|---|
| 49 | source;noise |
| 28 | source |
| 19 | source;coupling;noise;reservoir;cep99;nll;ece |
| 19 | source;cep99 |
| 18 | source;coupling;noise;reservoir;cep99;nll |

判断：

```text
1. CE+Brier 下 Soft / EMA3RoleNorm 仍没有打开 exploration pass 或 official pass。
2. 最接近 row 的 source_vs_best_control 只有 0.000567，仍低于 exploration gate 0.002 与 official gate 0.005。
3. failure pattern 仍由 source 不足主导，不是 CE-only fallback 覆盖不足造成的假 no-go。
4. v13.6 合法 route 仍为 R4-PopRiskSNRNoGo。
```

## 17. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v13.6 是否达成目标，若未达成则继续。本次重新读取 official route、计划 5.7 fallback、Line S、stop/go rules、no-go boundary、next hypothesis queue 与 required manifest。

最终 artifact：

```text
route = R4-PopRiskSNRNoGo
minimum_success = S1-EfficientSubstrate
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
synthetic_rows = 140
control_rows = 1120
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
mlp_snr_pass_rows = 0
kan_parameter_snr_pass_rows = 0
kan_basis_snr_pass_rows = 0
cover_boundary_accept_count = 22
cover_boundary_rows = 28
substrate_snr_gate_pass_count = 4
nonrat_substrate_snr_gate_pass_count = 0
required_artifact_missing_count = 0
provenance_violation_count = 0
forbidden_information_violation_count = 0
readout_feature_proxy_only = 0
feature_table_proxy_only = 0
code_review_packet_sha256 = 84b4aeb1b3ed76537ac6a9b4c70b0a0155284fd5f667778075d3bf39cea10d12
manifest_rows = 35
missing_required_rows = 0
```

计划 stop/go 对照：

```text
1. 计划 5.7：SNR active fraction 低时尝试 soft q / EMA3 / role-wise threshold normalization；已执行，CE+Brier 补跑仍为 R4。
2. 计划 5.7：SNR active fraction 高时尝试 increase tau / cap active fraction / SameActiveFractionRandomMask；已执行，仍为 R4。
3. 计划 5.7：如果 MLP-SNR 与 KAN-SNR 都 fail，记录 PopRiskSNRNoGo_CurrentImplementation 并转 substrate/base architecture；official 中 mlp_snr_pass_rows=0, kan_parameter_snr_pass_rows=0, kan_basis_snr_pass_rows=0。
4. 计划 Line S：RAT/CHE/FOU/RBF/WAV candidate families 已补跑并合入 official_v136；Non-RAT substrate-SNR pass 仍为 0。
5. 计划 14.2：parameter-SNR 与 basis-SNR across MLP/KAN 都 fail 时 route = R4-PopRiskSNRNoGo；当前 official route 与该 stop rule 一致。
6. 计划 15：MLP baseline、Rational parameter/basis/cover、Non-RAT scouts、X1-X7、MLP analog、final no-go queue 均已覆盖。
```

最终判断仍是：

```text
v13.6 没有达成目标；
不允许 promotion；
不允许 real short-run；
合法 route = R4-PopRiskSNRNoGo；
允许 final stop。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改 gate。原因不是轻易放弃，而是计划内继续条件已经闭合：MLP/KAN SNR 均为 0 pass，soft/EMA/role、tau/cap/norm、Line S substrate/base scout 都没有打开 synthetic task-family pass。当前我不确定如何在 v13.6 runner 内安全实现新的 substrate/base architecture 或重定义 value source，同时保证不把 diagnostic/scout row 编造成 success。继续推进应新开 substrate/base architecture 或 value-source reset 计划，而不是在 v13.6 内继续排列局部超参。

## 18. 用户再次追问后的 Line G / Line X 分支复核

用户再次要求“未达成则继续”。本次检查是否还有计划内未闭合的 cover-boundary loosening 或 synthetic exploration 分支。

复核结果：

```text
v136_basis_cover_boundary.csv rows = 28
v136_synthetic_family_results.csv rows = 140
v136_snr_parameter_update.csv rows = 56
v136_snr_basis_channel_update.csv rows = 84
v136_controls.csv rows = 1120
v136_substrate_snr_gate.csv rows = 19
boundary_accept = 22
boundary_reject = 6
official_synthetic_pass = 0/140
exploration_pass = 0/140
```

最接近但不能 pass 的 official row：

| method | task | seed | loss | variant | source_vs_best | CouplingR2 delta | NoiseSignalLeak delta | Reservoir delta | CEp99 delta | exploration | official |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| KAN-BasisCoverBoundary | X4 | 0 | CE | Hard | 0.0003762373062765681 | 2.0438709259033203 | -0.018877089023590088 | -0.34347105026245117 | -0.95526123046875 | 0 | 0 |

判断：

```text
1. cover-boundary 不是全拒绝 no-op，boundary_accept = 22/28。
2. 因此计划 7.5 的 phase-specific loosening 条件没有触发；强行 loosening 会变成未授权 gate/guard 小修。
3. Line X exploration_pass = 0/140，official_synthetic_pass = 0/140，无法进入 real short-run。
4. 当前 blocker 仍是 source_vs_best_control 不足，不是 cover 全拒绝或 artifact 缺失。
5. v13.6 合法 route 仍为 R4-PopRiskSNRNoGo。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改 gate。当前计划内可执行修复线已经闭合；我仍不确定如何在 v13.6 runner 内安全发明新的 substrate/base architecture 或 value-source reset，同时不混淆当前 R4 no-go 结论。
