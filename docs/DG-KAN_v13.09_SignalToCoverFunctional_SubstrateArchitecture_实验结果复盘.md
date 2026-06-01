# DG-KAN v13.09 SignalToCoverFunctional SubstrateArchitecture 实验结果复盘

生成时间：2026-05-28（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不虚构成功、不补填未执行数据，不把 smoke / focused repair / Non-RAT substrate audit 写成 promotion。

## 1. 计划理解

v13.09 的问题不是继续扩大 K1-K7 或 MLP G6-G9 小修，而是验证：

```text
population-risk SNR signal 是否能转移成 stable basis cover，
再由 Rational KAN substrate 打开 KAN-specific S3/S4。
```

硬约束：

```text
1. strict FC-PureKAN / no active B-spline budget。
2. 不使用 teacher / distillation / sampler / class weight / dataset-name branch。
3. 不使用 label-informed initialization。
4. functional direction 只能使用 train-stream gradient / parameter / group / basis telemetry。
5. LineC / CEp99 / NLL / ECE 只能作为 audit / gate，不能生成方向。
6. MLP row 只能是 generic optimizer control，不能写成 KAN promotion。
7. Non-RAT 必须先过 exact no-materialize substrate-health gate，不能以 scout / proxy alias 进入 functional proof。
```

计划 route 的关键含义：

```text
R2-SignalRetentionPassCoverFormationFail =
K0 signal retention / cosine 通过，但 cover purity / churn 未形成稳定 cover。
```

## 2. 本轮代码修改

修改文件：

```text
experiments/run_v137_boundary_conditioned_poprisk_training.py
experiments/run_v139_signal_to_cover_functional_substrate_architecture.py
```

主要修改：

```text
1. 在 SNR update 中加入 train-stream per-example gradient cover telemetry：
   cover_purity_mean / cover_purity_p10 / cover_churn / cover_load_gini /
   cover_specialization_entropy / signal_to_cover_score / split/merge/freeze /
   false_drop_fraction / false_keep_fraction。
2. 实现 K8-K12 方法语义：
   K8 GradientClusterCover k4/k8；
   K9 CoverSplitMerge lite/noMerge；
   K10 ParamSNRThenCover 3phase/slowConsolidate；
   K11 ReadoutBasisDecoupledSNR / ReadoutFirstBasisConsolidate；
   K12 VarianceReservoirProxyCoverGrowth / LowVarianceSignalCoverGrowth。
3. 新增 v13.09 official runner 与 required artifact surface。
4. Non-RAT exact substrate 表不把 scout / materialized proxy 写成 exact no-materialize success。
5. route decision 明确区分 K0 retention、cover formation、KAN S3/S4、Non-RAT substrate 与 MLP control。
```

合法性说明：

```text
1. K8-K12 direction 不使用 validation/test/future/query batch。
2. K8-K12 direction 不使用 LineC / CEp99 / NLL / ECE。
3. MLP 只作为 control monitor。
4. Non-RAT exact audit promotion_allowed=0。
5. 不改变 promotion gate。
```

## 3. Smoke 结果

执行规模：

```text
datasets = MNIST
synthetic_tasks = X1
synthetic_seeds = 0
k_methods = RAT-AdamW,K8-RAT-GradientClusterCover-k4,K10-RAT-ParamSNRThenCover-3phase
train_steps = 4
batch_size = 16
```

结果：

```text
route = R2-SignalRetentionPassCoverFormationFail
minimum_success = S1C-SignalRetentionPositive
promotion_allowed = 0
official_success_reached = 0
required_artifact_missing_count = 0
snr_transfer_gate_pass = 1
snr_transfer_median_retention_group = 1.089290976524353
snr_transfer_median_cos_group_vs_param = 0.9345244765281677
cover_purity_median = 0.055969491600990295
cover_churn_mean = 0.0
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
```

解释：

```text
smoke 只证明 runner、artifact、K8/K10 surface 可执行；
不能 promotion。
```

## 4. Official v13.09 结果

执行规模：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
mlp_seeds = 0,1,2,3,4
mlp_methods = MLP-AdamW, G6/SNRBlend, G7/LogitNormTrust
synthetic_tasks = X1..X7
synthetic_seeds = 0,1,2
loss_interfaces = CE,Brier
k_methods = RAT-AdamW, old K4/K7, K8..K12
train_steps = 200
batch_size = 32
```

最终 route：

```text
route = R2-SignalRetentionPassCoverFormationFail
minimum_success = S1C-SignalRetentionPositive
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
failure_rows = 0
compute_budgeted_run = 0
```

K0 / cover：

```text
snr_transfer_gate_pass = 1
snr_transfer_median_retention_group = 0.7663904428482056
snr_transfer_median_cos_group_vs_param = 0.6665693521499634
cover_purity_median = 0.030393941327929497
cover_churn_mean = 0.0
cover_formation_gate_pass = 0
```

KAN synthetic：

```text
v139_rat_signal_to_cover_summary rows = 546
pass_s3 rows = 55
pass_s4 rows = 0
kan_s3_task_pass_count = 4 / 7
kan_s4_task_pass_count = 0 / 7
kan_real_short_run_open_allowed = 0
```

逐 task S3 row count：

| task | pass_s3 rows |
|---|---:|
| X1 | 14 |
| X2 | 15 |
| X3 | 8 |
| X4 | 9 |
| X5 | 0 |
| X6 | 1 |
| X7 | 8 |

说明：

```text
虽然 X1/X2/X3/X4/X6/X7 有若干逐行 pass_s3，
official family gate 需要每个 family 内 >=2 seeds 或 >=2 losses；
最终只得到 4/7，不达到 KAN S3 >=5/7。
S4 没有任何 row 通过，主要因为 cover purity 太低且 LineC/all-pass 不稳定。
```

最接近但不能写成 S3/S4 family success 的 rows：

| task | seed | loss | method | source_vs_adamw | AUC_time_ratio | LineC majority | cover_purity | Noise delta | Reservoir delta | pass_s3 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| X1 | 0 | Brier | K9-noMerge | 0.366386 | 0.668597 | 0 | 0.024950 | 0.028929 | -6.144982 | 0 |
| X4 | 0 | Brier | K9-noMerge | 0.272646 | 0.771586 | 1 | 0.020179 | -0.013780 | -5.168757 | 1 |
| X3 | 0 | Brier | K8-k8 | 0.239457 | 0.772755 | 0 | 0.056231 | 0.107049 | -4.454212 | 0 |
| X5 | 0 | CE | K9-noMerge | 0.184970 | 0.888682 | 1 | 0.041099 | -0.016985 | -6.053541 | 0 |
| X6 | 1 | Brier | K10-3phase | 0.099839 | 0.834062 | 1 | 0.031200 | -0.023474 | -2.170931 | 1 |
| X7 | 1 | Brier | K10-slowConsolidate | 0.091669 | 0.904011 | 1 | 0.039662 | -0.023346 | -0.842398 | 1 |

主要 blocker：

```text
1. K0 signal retention / cosine 已经通过。
2. cover_purity_median = 0.0304，远低于计划 gate 0.20。
3. X5 source/AUC/LineC majority 看起来好，但 NLL_delta / ECE_delta tail harm 超 gate。
4. X6/X7 仍只有局部 seed/loss pass，不满足 task-family coverage。
5. S4 需要 cover / LineC all-pass / non-harm 同时成立，本轮 0 row。
```

MLP control：

```text
mlp_generic_10seed_dataset_pass_count = 2
mlp_generic_10seed_dataset_count = 3
mlp_generic_10seed_confirmed = 0
```

解释：

```text
MLP control 不能写成 KAN success；
且本轮 MLP 也没有达到 3/3 dataset confirmed。
```

## 5. Non-RAT exact substrate repair

执行内容：

```text
D1 Fourier: D-FOU20-LowFreqIdentityResidualHealthSubstrate
D2 RBF: D-RBF17-CompactCapacityK4HealthSubstrate
D3 Wavelet: D-WAV16-SupportStableHatHealthSubstrate
D4 Chebyshev: D-CHE20-DegreeNormalizedReadoutHealthSubstrate
dataset = MNIST
seed = 0
train_size = 64
val_size = 32
workspace_warmup_steps = 2
workspace_profile_steps = 5
```

结果：

```text
workspace_rows = 4
workspace_gate_pass_rows = 0
workspace_strong_gate_pass_rows = 0
hardening_rows = 4
hardening_executed_rows = 0
family_near_pass_rows = 0
promotion_allowed = 0
```

逐候选结果：

| candidate | exact_kernel | no_materialize_hard_gate | raw ratio | incremental ratio | step ratio | workspace pass |
|---|---:|---:|---:|---:|---:|---:|
| D-FOU20 | 1 | 1 | 1.105324 | 2.964912 | 1.426523 | 0 |
| D-RBF17 | 0 | 0 | 1.220914 | 2.525480 | 1.226483 | 0 |
| D-WAV16 | 0 | 0 | 1.184995 | 2.525480 | 1.358163 | 0 |
| D-CHE20 | 0 | 0 | 1.228871 | 2.811195 | 1.151777 | 0 |

解释：

```text
1. D-FOU20 已有 exact/no-materialize audit pass，但 incremental memory ratio = 2.964912 > 2.00，workspace gate fail。
2. D-RBF17 / D-WAV16 / D-CHE20 仍是 proxy/materialized path，不是 exact no-materialize success。
3. hardening/LineC 按 workspace fail 被 skip，不能写成 substrate-health pass。
4. Non-RAT 仍不能进入 functional proof。
```

## 6. Focused Case B/C repair

触发原因：

```text
official 后仍未达成 S3/S4；
X5 是 source 好但 tail fail；
X6/X7 是 source/LineC/seed coverage 不稳定。
```

修复方向：

```text
按计划 Case B/C：
1. 延后 cover consolidation；
2. readout follows AdamW/SNR, basis follows cover-SNR；
3. 使用 reservoir / variance proxy；
4. 不使用 LineC target 生成方向；
5. 不降低 source / tail / LineC gate。
```

执行规模：

```text
synthetic_tasks = X5,X6,X7
synthetic_seeds = 0,1,2
loss_interfaces = CE,Brier
k_methods = RAT-AdamW,K9,K10,K11,K12 selected
train_steps = 300
batch_size = 32
compute_budgeted_run = 1
```

结果：

```text
route = R2-SignalRetentionPassCoverFormationFail
promotion_allowed = 0
official_success_reached = 0
required_artifact_missing_count = 0
snr_transfer_gate_pass = 1
snr_transfer_median_retention_group = 0.7194734811782837
snr_transfer_median_cos_group_vs_param = 0.6489424705505371
cover_purity_median = 0.03219109773635864
cover_churn_mean = 0.0
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
```

focused repair pass rows：

```text
pass_s3 rows = 5
X5 pass_s3 rows = 0
X6 pass_s3 rows = 1
X7 pass_s3 rows = 4
```

解释：

```text
1. X5 仍被 NLL/ECE tail harm 拒绝。
2. X6 仍只有单 row pass。
3. X7 增加 source，但 LineC / family coverage 仍不足。
4. 因 compute_budgeted_run=1 且只覆盖 X5/X6/X7，不能替代 official。
```

## 7. Required artifacts

official_v139 manifest：

```text
manifest_rows = 30
missing_required_rows = 0
```

主要产物：

```text
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_route_decision.json
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_required_manifest.csv
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_code_provenance_audit.csv
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_forbidden_information_audit.csv
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_signal_to_cover_audit.csv
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_rat_signal_to_cover_training.csv
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_rat_signal_to_cover_summary.csv
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_rat_signal_to_cover_linec.csv
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_nonrat_exact_substrate.csv
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_mlp_control_monitor.csv
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_failure_table.csv
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_no_go_boundary.md
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_next_hypothesis_queue.md
results/v13_09_signal_to_cover_functional_substrate_architecture/official_v139/v139_code_review_packet.zip
```

repair artifacts：

```text
results/v13_09_signal_to_cover_functional_substrate_architecture/nonrat_exact_v139/
results/v13_09_signal_to_cover_functional_substrate_architecture/repair_v139_x5x6x7_casebc/
```

## 8. 最终科学结论

v13.09 没有达成 S3/S4/S5；最终合法 route：

```text
R2-SignalRetentionPassCoverFormationFail
minimum_success = S1C-SignalRetentionPositive
promotion_allowed = 0
official_success_reached = 0
kan_real_short_run_open_allowed = 0
```

已闭合事实：

```text
1. K0 signal retention 成立：retention median = 0.7664，cos median = 0.6666。
2. cover formation 没有成立：cover_purity_median = 0.0304 < 0.20。
3. K8-K12 official 共 546 summary rows，S3 family pass = 4/7，未达到 5/7。
4. S4 rows = 0，说明 cover / LineC / non-harm 组合没有打开。
5. Focused Case B/C repair 对 X5/X6/X7 仍为 R2，不能补成 official success。
6. Non-RAT exact substrate repair 4 candidates 全部 workspace fail；RBF/WAV/CHE 仍不是 exact no-materialize success。
7. MLP control 只过 2/3 datasets，且不能作为 KAN promotion。
8. required artifacts 缺失为 0，forbidden information violation 为 0。
```

no-go boundary：

```text
1. v13.09 证明当前 blocker 不是 signal retention 丢失，而是 signal 没有形成稳定 basis cover。
2. Rational substrate 可产生强 source/AUC 局部行，但 cover purity 极低，family coverage 不足。
3. X5 是 tail harm blocker；X6/X7 是 seed/loss/LineC 稳定性 blocker。
4. Non-RAT 仍卡在 exact/no-materialize 和 workspace gate，不能进入 functional proof。
5. 按 v13.09 计划，继续小幅排列 K8-K12 或 MLP G-token 价值低；下一步需要更强 substrate/base architecture reset。
```

最终判断：

```text
v13.09 未达成目标；
不允许 promotion；
不允许 real short-run；
允许 final stop，原因是 official K8-K12、focused Case B/C repair、Non-RAT exact substrate repair 均未打开计划 gate。
```

## 9. 用户再次追问后的 K13 cover-debt guard 修复

用户再次要求确认未达成则继续。本次重新对照计划第 14-16 节后，结论是：

```text
1. v13.09 official route 仍是 R2。
2. Case A 的 K8/K9/K10/K11 已在 official 覆盖。
3. Case B/C 的 focused repair 已覆盖 X5/X6/X7，但仍为 R2。
4. Non-RAT Case E exact substrate repair 也没有打开 workspace gate。
5. 计划第 16 节建议：若 Rational 仍不能 synthetic >=5/7 S3，则需要 stronger substrate/base architecture reset。
```

但为了排除一个实现层边界，本次继续做了一次安全修复：

```text
底层已有 cover_policy() / would_increase_cover_debt() train-batch cover-debt guard；
K8-K12 alias 没有显式接上 CoverPhaseSchedule。
因此新增 K13-style aliases，把 Case B/C 的 delayed consolidation / low-purity update reduction / Phase3 consolidation 接到现有合法 train-batch cover-debt guard。
```

代码修改：

```text
experiments/run_v139_signal_to_cover_functional_substrate_architecture.py
  新增：
  K13-RAT-CoverDebtGuardedSplitMerge-lite
  K13-RAT-ParamSNRSlowCoverPhaseGuard
  K13-RAT-LowVariancePhase3CoverGuard
  K13-RAT-ReadoutFirstCoverGuard
```

合法性说明：

```text
1. K13 只使用 train-stream gradient 与 train-batch basis cover-debt telemetry。
2. 不使用 LineC / CEp99 / NLL / ECE 生成方向。
3. 不降低 source / AUC / tail / LineC / cover gate。
4. focused repair compute_budgeted_run=1，不能替代 official。
```

执行规模：

```text
synthetic_tasks = X5,X6,X7
synthetic_seeds = 0,1,2
loss_interfaces = CE,Brier
k_methods = RAT-AdamW,K13-RAT-CoverDebtGuardedSplitMerge-lite,K13-RAT-ParamSNRSlowCoverPhaseGuard,K13-RAT-LowVariancePhase3CoverGuard,K13-RAT-ReadoutFirstCoverGuard
train_steps = 300
batch_size = 32
```

结果：

```text
route = R2-SignalRetentionPassCoverFormationFail
compute_budgeted_run = 1
promotion_allowed = 0
official_success_reached = 0
required_artifact_missing_count = 0
snr_transfer_gate_pass = 1
snr_transfer_median_retention_group = 0.7194734811782837
snr_transfer_median_cos_group_vs_param = 0.6489424705505371
cover_purity_median = 0.03219109773635864
cover_churn_mean = 0.0
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
```

逐行结果：

```text
v139_rat_signal_to_cover_summary rows = 90
pass_s3 rows = 3
pass_s4 rows = 0
X5 pass_s3 rows = 1
X6 pass_s3 rows = 0
X7 pass_s3 rows = 2
```

最接近 rows：

| task | seed | loss | method | source_vs_adamw | AUC_time_ratio | NLL delta | ECE delta | LineC majority | cover purity | pass_s3 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| X7 | 2 | Brier | K13-LowVariancePhase3CoverGuard | 0.283445 | 0.732842 | -3.577312 | -0.296999 | 0 | 0.084654 | 0 |
| X7 | 1 | Brier | K13-LowVariancePhase3CoverGuard | 0.101607 | 0.899954 | -3.032255 | -0.249470 | 1 | 0.016306 | 1 |
| X5 | 0 | Brier | K13-ParamSNRSlowCoverPhaseGuard | 0.023444 | 0.969900 | -0.139494 | -0.022553 | 1 | 0.066040 | 1 |
| X6 | 2 | Brier | K13-ParamSNRSlowCoverPhaseGuard | 0.083134 | 0.841373 | -0.518049 | -0.048963 | 0 | 0.015432 | 0 |

判断：

```text
K13 cover-debt guard 没有打开 targeted family gate；
X5/X6/X7 仍不能形成 >=2 seeds 或 >=2 losses 的 family pass；
S4 仍为 0；
不能写成 v13.09 official success。
```

最终 stop-contract 更新：

```text
v13.09 仍未达成目标；
合法 route 仍为 R2-SignalRetentionPassCoverFormationFail；
promotion_allowed = 0；
real_short_run_open_allowed = 0；
当前 v13.09 内已执行：K8-K12 official、Non-RAT exact substrate repair、Case B/C focused repair、K13 cover-debt guard focused repair。
```

## 10. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v13.09 是否达成目标，若未达成则继续。本次重新读取计划第 14-16 节、official route 与 K13 repair route。

最终 artifact：

```text
official_v139:
route = R2-SignalRetentionPassCoverFormationFail
minimum_success = S1C-SignalRetentionPositive
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
snr_transfer_gate_pass = 1
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 4
kan_s4_task_pass_count = 0
nonrat_substrate_health_pass_count = 0

repair_v139_k13_coverguard_x5x6x7:
route = R2-SignalRetentionPassCoverFormationFail
official_success_reached = 0
promotion_allowed = 0
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
compute_budgeted_run = 1
```

计划 stop rule 对照：

```text
1. Case A 的 K8/K9/K10/K11 已在 official 覆盖。
2. Case B/C 的 focused repair 与 K13 cover-debt guard repair 已执行，仍未打开 X5/X6/X7 family gate。
3. Case E 的 Non-RAT exact substrate repair 已执行，workspace gate 仍为 0/4。
4. 第 16 节写明：若 v13.9 仍不能让 Rational synthetic >=5/7 达到 S3，
   需要 stronger substrate/base architecture reset，而不是继续 functional update 搜索。
```

最终判断仍是：

```text
v13.09 没有达成 S3/S4/S5；
只达到 S1C-SignalRetentionPositive；
不允许 promotion；
不允许 real short-run；
合法 route = R2-SignalRetentionPassCoverFormationFail。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改 gate。原因不是轻易放弃，而是计划内已执行的 functional / repair fallback 均未打开 gate；继续在当前 v13.09 runner 中排列 K-token 小修会违反第 16 节的边界。我现在不确定如何在当前 v13.09 代码路径内安全实现 stronger substrate/base architecture reset，同时保证不把新的 substrate diagnostic / proxy 误写成 v13.09 functional success。下一步应新开 substrate/base architecture reset runner/计划，而不是继续扩当前 v13.09 functional update 搜索。

当前边界：

```text
我现在不确定如何在当前 v13.09 functional-update runner 内安全实现 stronger substrate/base architecture reset，
同时不混淆 v13.09 的 signal-to-cover no-go 结论。
继续增加 K8-K13 变体会变成 functional token 搜索，已经偏离计划第 16 节。
下一步应新开 substrate/base architecture reset 计划/runner。
```

## 11. 用户再次追问后的 Line S cross-plan substrate/base scout

用户再次要求未达成则继续。本次重新搜索仓库后发现 v13.6 已实现过 `Line S：Substrate / base architecture reset` 的 audited scout mapping。为继续尝试 v13.09 第 16 节推荐的 substrate/base architecture reset，本次在 v13.09 result 目录下补跑 v13.6 Line S scout。

执行内容：

```text
runner = experiments/run_v136_poprisk_snr_basis_cover_boundary.py
out_dir = results/v13_09_signal_to_cover_functional_substrate_architecture/line_s_crossplan_substrate_scout_v139
synthetic_task = X1
seed = 0
method = KAN-BasisChannelSNR
loss = CE
operator_batch_size = 8
checkpoint_steps = 4
future_steps = 4
controls = Functional,SameActiveFractionRandomMask
```

审计边界：

```text
1. 这是 cross-plan substrate/base scout，不是 v13.09 official functional promotion。
2. Line S 设计名映射到已有 v12.35/v1235 audited primitive。
3. artifact 显式记录 mapped_candidate_id / line_s_candidate / line_s_mapping_no_new_kernel_claim。
4. 仍按 substrate-SNR / workspace gate 判定，不降低任何 S3/S4/LineC/tail gate。
5. 不把 Rational substrate-SNR scout row 写成 KAN functional success。
```

结果：

```text
route = R4-PopRiskSNRNoGo
required_artifact_missing_count = 0
synthetic_rows = 1
synthetic_task_success_count = 0
promotion_allowed = 0
substrate_snr_gate_pass_count = 4
nonrat_substrate_snr_gate_pass_count = 0
full_parameter_update_rows = 1
control_rows = 2
```

Family summary：

| family | rows | substrate-SNR pass | status |
|---|---:|---:|---|
| D-RAT | 4 | 4 | SubstrateSNRGatePass |
| D-CHE | 4 | 0 | WorkspaceOnly_NotFunctionalSubstrate |
| D-FOU | 4 | 0 | WorkspaceOnly_NotFunctionalSubstrate |
| D-RBF | 4 | 0 | WorkspaceOnly_NotFunctionalSubstrate |
| D-WAV | 3 | 0 | WorkspaceOnly_NotFunctionalSubstrate |

通过的 Rational rows：

| candidate | mapped_candidate_id | incremental ratio | step ratio | channel_snr_entropy | pass |
|---|---|---:|---:|---:|---:|
| D-RAT26-TangentTrustRegionNoCE | D-RAT26-TangentTrustRegionNoCE | 1.603821 | 1.291973 | 0.893051 | 1 |
| RAT-SNR1-denSlopeTelemetrySubstrate | D-RAT27-DenSlopeGuardNoCE | 1.593854 | 1.232687 | 0.893685 | 1 |
| RAT-SNR2-groupDiversityFloorSubstrate | D-RAT28-GroupDiversityPreservingRational | 1.603821 | 1.372908 | 0.893051 | 1 |
| RAT-SNR3-readoutRationalDecoupledSubstrate | D-RAT35-ReadoutRationalDecoupleNoCE | 1.349668 | 1.218606 | 0.893752 | 1 |

判断：

```text
Line S cross-plan substrate/base scout 没有改变 v13.09 结论。
Rational substrate-SNR scout 仍可见，但这不是 S3/S4 functional success。
Non-RAT Line S rows 的 channel_snr_entropy 非零，但 workspace_pass 全部为 0；
nonrat_substrate_snr_gate_pass_count = 0。
synthetic_task_success_count = 0，不能打开 real short-run。
```

最终判断更新：

```text
v13.09 仍未达成 S3/S4/S5；
合法 route 仍为 R2-SignalRetentionPassCoverFormationFail；
cross-plan Line S scout 自身 route = R4-PopRiskSNRNoGo；
promotion_allowed = 0；
real_short_run_open_allowed = 0。
```

本次继续推进后，计划内与近邻计划可安全借用的 fallback 已更闭合：

```text
1. K8-K12 official 未达成。
2. Non-RAT exact substrate repair 未达成。
3. Case B/C focused repair 未达成。
4. K13 cover-debt guard focused repair 未达成。
5. v13.6 Line S cross-plan substrate/base scout 未找到 Non-RAT functional substrate，也没有 synthetic task success。
```

我现在仍不确定如何在当前 v13.09 代码路径内继续安全推进 stronger substrate/base architecture reset，同时不把 substrate scout / diagnostic / mapped primitive row 混写成 v13.09 functional success。继续推进需要新开 substrate/base architecture reset runner，而不是在 v13.09 functional-update runner 内继续补 K-token 或 scout alias。

## 12. 用户再次追问后的 Rational Line S substrate-candidate focused repair

用户再次要求未达成则继续。本次进一步检查 v13.09 runner 后发现它支持 `--rational-candidate`，而 official_v139 默认只用 `choose_primary_rational()`。既然 Line S cross-plan scout 中 RAT-SNR1/2/3 mapped substrates 通过 substrate-SNR gate，本次尝试把它们直接接入 v13.09 signal-to-cover training。

候选：

```text
D-RAT27-DenSlopeGuardNoCE
D-RAT28-GroupDiversityPreservingRational
D-RAT35-ReadoutRationalDecoupleNoCE
```

先尝试较大的 D-RAT35 focused matrix：

```text
candidate = D-RAT35-ReadoutRationalDecoupleNoCE
tasks = X5,X6,X7
seeds = 0,1,2
losses = CE,Brier
k_methods = RAT-AdamW,K9,K10,K13 variants
train_steps = 200
```

结果：

```text
运行约 11 分钟仍未结束；
为避免整轮卡在一个过大的 focused run 上，手动终止；
该 run 未产生可用 route artifact，不写入成功或失败指标。
```

随后改跑 micro focused probes：

```text
tasks = X5,X6,X7
seeds = 0,1
losses = CE,Brier
k_methods = RAT-AdamW,K13-RAT-LowVariancePhase3CoverGuard,K13-RAT-ReadoutFirstCoverGuard
train_steps = 80
synthetic_train_size = 64
synthetic_val_size = 32
compute_budgeted_run = 1
```

结果汇总：

| rational candidate | route | transfer gate | retention | cosine | cover purity | S3 task pass | S3 rows | S4 rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D-RAT27-DenSlopeGuardNoCE | R1-GenericSNROptimizerNoGo | 0 | 0.592759 | 0.532397 | 0.031367 | 0 | 3 | 0 |
| D-RAT28-GroupDiversityPreservingRational | R1-GenericSNROptimizerNoGo | 0 | 0.592886 | 0.532733 | 0.031374 | 0 | 2 | 0 |
| D-RAT35-ReadoutRationalDecoupleNoCE | R1-GenericSNROptimizerNoGo | 0 | 0.592714 | 0.532039 | 0.031379 | 0 | 3 | 0 |

Task-level S3 row count：

```text
D-RAT27: X5=1, X6=1, X7=1
D-RAT28: X5=0, X6=1, X7=1
D-RAT35: X5=1, X6=1, X7=1
```

最接近 rows 的共同模式：

```text
1. K13-LowVariancePhase3CoverGuard 在 X5/X6/X7 上能产生局部 source_vs_adamw positive。
2. X7 CE rows source 很强，但 NoiseSignalLeak_delta > 0，因此被 gate 拒绝。
3. cover_purity 仍只有约 0.031，远低于 0.20。
4. micro run 的 K0 transfer retention/cos 低于 official gate，因此 route 退到 R1。
5. compute_budgeted_run=1，不能替代 official。
```

判断：

```text
把 Line S Rational mapped candidates 作为 v13.09 substrate/base focused repair，没有打开 S3/S4。
逐行局部 positive 不能写成 family success；
Rational substrate-SNR 可见仍没有转化成 stable basis cover；
不允许 promotion。
```

最终 stop-contract 更新：

```text
v13.09 仍未达成目标；
official route 仍为 R2-SignalRetentionPassCoverFormationFail；
Line S cross-plan scout route = R4-PopRiskSNRNoGo；
Rational Line S substrate-candidate focused probes route = R1-GenericSNROptimizerNoGo；
promotion_allowed = 0；
real_short_run_open_allowed = 0。
```

本次继续推进后，当前计划内和相邻已定义 substrate/base scout 均没有打开 gate：

```text
1. K8-K12 official 未达成。
2. Non-RAT exact substrate repair 未达成。
3. Case B/C focused repair 未达成。
4. K13 cover-debt guard focused repair 未达成。
5. v13.6 Line S cross-plan substrate/base scout 未达成。
6. Line S Rational mapped candidate direct signal-to-cover micro probes 未达成。
```

我现在已经不确定如何继续在当前 v13.09 代码路径内安全推进，而不把 compute-budgeted focused probe / substrate scout / mapped primitive diagnostic 编造成 official success。继续推进需要新的 substrate/base architecture reset 计划/runner。

## 13. 用户再次追问后的最终复核

用户再次要求未达成则继续。本次只做 stop-contract 与进程复核，不新增训练实验。

最终 artifact：

```text
official_v139:
route = R2-SignalRetentionPassCoverFormationFail
minimum_success = S1C-SignalRetentionPositive
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
snr_transfer_gate_pass = 1
cover_formation_gate_pass = 0
kan_s3_task_pass_count = 4
kan_s4_task_pass_count = 0
nonrat_substrate_health_pass_count = 0

k13_coverguard:
route = R2-SignalRetentionPassCoverFormationFail
promotion_allowed = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0

line_s_crossplan:
route = R4-PopRiskSNRNoGo
promotion_allowed = 0
synthetic_task_success_count = 0
nonrat_substrate_snr_gate_pass_count = 0

drat27_micro / drat28_micro / drat35_micro:
route = R1-GenericSNROptimizerNoGo
promotion_allowed = 0
kan_s3_task_pass_count = 0
kan_s4_task_pass_count = 0
compute_budgeted_run = 1
```

复核到当前没有 v13.09 / v13.6 实验进程仍在运行。

计划第 16 节对照：

```text
如果 v13.9 仍然不能让 Rational 在 synthetic >=5/7 上达到 S3，
那么我们需要更强的 substrate/base architecture reset，
而不是继续 functional update 搜索。
```

最终判断仍是：

```text
v13.09 没有达成 S3/S4/S5；
不允许 promotion；
不允许 real short-run；
合法 official route = R2-SignalRetentionPassCoverFormationFail。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改 gate。原因是可安全执行的当前计划内与相邻 substrate/base scout fallback 都已经闭合：K8-K12 official、Non-RAT exact repair、Case B/C repair、K13 cover-debt repair、Line S cross-plan scout、Rational Line S direct micro probes 均未打开 gate。我现在已经不确定如何在当前 v13.09 runner 内继续推进而不混淆 official success 边界。下一步需要新的 substrate/base architecture reset 计划/runner。
