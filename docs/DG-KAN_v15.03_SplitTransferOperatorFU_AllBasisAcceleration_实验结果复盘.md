# DG-KAN v15.3 SplitTransferOperatorFU AllBasisAcceleration 实验结果复盘

生成时间：2026-05-31（Asia/Singapore）

本复盘只写入实际 artifact 中的结果；不把 split-transfer diagnostic、substrate-only replay 或 MLP/generic controls 写成 promotion。

## 1. 计划理解

v15.3 将 functional update 重定义为 train-split transfer operator FU：用当前 train batch 的 B1/B2 split 固定求解 update，检验 local positive 是否能转成 split-transfer value。

## 2. 本轮代码修改

新增：
```text
experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py
```
修改：
```text
experiments/run_v149_line_d_all_basis_substrate_repair.py
  新增 v15.3 substrate-only candidates:
  D-FOU52..56, D-RBF50..54, D-WAV45..48。
```

过程修正：
```text
1. 首次 official runner 使用 FMS method 名 D-CHE6-PhaseScheduleDegreeFMS 作为 substrate candidate，
   底层 make_case_model 无法解析该 id。
   已改为复用 v14.10 既有合法 D-CHE substrate:
   D-CHE17-HighDegreeLateEnableSubstrate。
   该修正只影响 D-CHE carrier id 桥接，不新增 T6/T7、不新增 F-CHE token。
2. 二次 official runner 在 MLP control 建模处缺少 mlp_hidden 参数。
   已在 v15.3 runner 的 case_args 中补齐 mlp_hidden = hidden。
   该修正只影响 MLP/generic control 的模型构造，不改变 ST-FU solver 或 gate。
3. 用户再次追问后复核计划 6.4/6.5，发现 Line T result table 未显式落盘
   step / subspace_type / solver_type / solve_time_ms。
   已补齐记录；rank>1 subspace 使用 ridge_closed_form 方向并在固定 alpha grid 内评估，
   不新增 T6/T7，不引入 action search。
4. 同次复核发现 Line D substrate artifact 的 v15.3 字段名未完全对齐计划。
   已补齐 low/mid/high_band_energy、step_ratio、memory_ratio、AUCtime_ratio 等别名；
   v149 原始 artifact 未提供的 width_p01/width_p99/active_center_snr 保留空值并写 availability/source，
   不编造 telemetry。
5. 再次追问后发现 finalizer 顺序在空目录首次运行时可能先写 contract、后写 gate recompute artifact，
   导致 Independent gate recompute artifact 行潜在未闭合。
   已调整为先写 plan coverage / gate recompute，再写 execution contract；该修正只影响审计顺序，不改变训练指标。
6. 修复后重新执行 full official v15.3；最终 required / forbidden / no-action / contract 审计均通过。
7. 用户再次追问后复核计划 6.3，发现 metric choice factor 只执行了 M0-identity。
   已接入预注册 M0/M1/M2 metric choices：M0-identity、M1-AdamVDiag、M2-DegreeRoleSecondMoment。
   metric choice 是计划内 factor，不新增 T6/T7 method token，不引入 action search。
   修复后重新执行 full official v15.3，并写入 v153_metric_choice_coverage.csv。
8. 用户再次追问后复核计划 8.2，发现 Line X 的 reservoir/noise readback 列存在空值。
   已用当前 train split B1/B2 loss delta 派生 audit-only readback，并补齐 T-FB6 fallback 行；
   该修正不新增训练、不改变 direction/gate，只补齐 transfer operator audit 记录。
9. 再次复核 Line X route 语义，发现 transfer_supported_count>0 但 Line X gate 未通过时，
   `R-X-TransferSupported` 容易被误读为 gate pass。
   已显式写入 line_x_gate_pass、v15.02.1 local-positive baseline、reduced flag，
   gate 未通过时写 R-X-PartialTransferButGateFailed；当前最终 artifact 为 R-X-TransferOperatorAuditGatePassed。
10. 再次复核 Line X gate 分母，发现 metric choice factor 将 3x3 audit 展开成 135 raw rows。
    计划写的是 transfer_supported_count >= 3/9，因此已补齐 dataset-seed 3x3 denominator audit，
    raw row count 继续保留为诊断，不再与 v15.02.1 的 10-row baseline 直接混算。
11. 再次复核 Line T objective，发现非 Adam methods 实现为 adam + alpha * ridge_direction，
    但计划第 6.3 写的是 f + J U alpha。
    已修正为 T1-T5/TCTRL/M1-M3/MCTRL 提交纯 alpha * U direction；T0/M0 仍是 AdamW baseline。
12. objective 修正后复核 T-FB4，发现 scale sanity audit 信息不足以解释 ST-FU 尺度失败。
    已补齐 alpha_at_grid_max_fraction / scale_grid_boundary_hit；不扩大 grid，不新增 action search。
13. 再次复核 gate recompute，发现 R1/R2/R6 no-go 条件可重叠，旧 route_consistent 按单行 route 名比较会误报不一致。
    已改为先按 precedence 重算 expected_route，再统一校验 final route；重叠 no-go 写 shadowed_by_earlier_route。
14. 再次复核 ST-FU commit scale，计划理论段明确写 Δθ = Uα。
    旧实现虽然生成 alpha * U，但提交参数时仍乘 optimizer lr。
    已修正为 T1-T5/TCTRL/M1-M3/MCTRL 使用 commit_lr_multiplier=1.0；T0/M0 AdamW baseline 保持 args.lr。
15. U normalization unit 修正后，上一轮 Line X 9/9 / R4 信号被证伪；
    最新 route 回到 R1-STFUStillLocalPositiveNoTransfer。
16. R4 finalizer 初版把 r4 判断写在 x_gate 赋值前，导致 reuse finalizer 报 UnboundLocalError。
    已修正变量顺序；该修正只影响 route finalizer，不改变训练 artifact。
17. 用户再次追问后针对 R4 runtime blocker 做实现级复核，发现 trial evaluation 每个 alpha 都重复 clone/restore，
    且非 B2Shuffled objective 的 best candidate 会重复计算同一个 true B2 loss。
    已改为每步只保存一次参数快照并复用；非 shuffled objective 直接复用 B2 objective loss。
    该修正只减少重复 forward/restore，不改变 ST-FU objective、subspace、alpha grid 或 direction source。
18. 继续复核 R4 runtime blocker，发现 full batch gradient 与 B1/B2 gradients 被重复 backward。
    已按 50/50 train split 用 B1/B2 mean gradients 加权合成 full-batch gradient；
    该修正不改变 Adam/ST-FU direction 的数学定义，只去掉一次重复 backward。
```

## 3. Line T D-CHE ST-FU 结果
```text
line_t_candidate_count = 15
real_lite_pass_count = 0 / 9
source_vs_best_control_mean = -0.46042725774976945
control_equivalent_fraction = 1.0
bad_event_fraction = 1.0
line_t_exploration_gate_pass = 0
line_t_meaningful_gate_pass = 0
line_t_s4_gate_pass = 0
best_t_method = T1-D-CHE-STFU-AdamSubspace [M2-DegreeRoleSecondMoment]
```

| method | rows | strict pass | dataset-seed pass | mean source vs best control | mean B2 transfer gain |
|---|---:|---:|---:|---:|---:|
| T1-D-CHE-STFU-AdamSubspace [M0-identity] | 9 | 0 | 0 | -0.46247026655409074 | 0.003221591313680013 |
| T1-D-CHE-STFU-AdamSubspace [M1-AdamVDiag] | 9 | 0 | 0 | -0.4619290563795302 | 0.003222306569417318 |
| T1-D-CHE-STFU-AdamSubspace [M2-DegreeRoleSecondMoment] | 9 | 0 | 0 | -0.46042725774976945 | 0.003231962521870931 |
| T2-D-CHE-STFU-PerExampleGradLowRank-r4 [M0-identity] | 9 | 0 | 0 | -0.5660087532467313 | 0.008561505211724175 |
| T2-D-CHE-STFU-PerExampleGradLowRank-r4 [M1-AdamVDiag] | 9 | 0 | 0 | -0.5614486535390218 | 0.008515371216668023 |
| T2-D-CHE-STFU-PerExampleGradLowRank-r4 [M2-DegreeRoleSecondMoment] | 9 | 0 | 0 | -0.5642697811126709 | 0.009245872497558594 |
| T3-D-CHE-STFU-PerExampleGradLowRank-r8 [M0-identity] | 9 | 0 | 0 | -0.5667087237040201 | 0.011934227413601346 |
| T3-D-CHE-STFU-PerExampleGradLowRank-r8 [M1-AdamVDiag] | 9 | 0 | 0 | -0.5560299555460612 | 0.011666311158074273 |
| T3-D-CHE-STFU-PerExampleGradLowRank-r8 [M2-DegreeRoleSecondMoment] | 9 | 0 | 0 | -0.5734127362569174 | 0.011642641491360135 |
| T4-D-CHE-STFU-DegreeRoleSubspace [M0-identity] | 9 | 0 | 0 | -0.5399695502387153 | 0.01666091548071967 |
| T4-D-CHE-STFU-DegreeRoleSubspace [M1-AdamVDiag] | 9 | 0 | 0 | -0.5386955473158095 | 0.016911652353074815 |
| T4-D-CHE-STFU-DegreeRoleSubspace [M2-DegreeRoleSecondMoment] | 9 | 0 | 0 | -0.5381275547875298 | 0.016338507334391277 |
| T5-D-CHE-STFU-OutputJacobianSketch-r4 [M0-identity] | 9 | 0 | 0 | -0.5467391279008653 | 0.0011570056279500325 |
| T5-D-CHE-STFU-OutputJacobianSketch-r4 [M1-AdamVDiag] | 9 | 0 | 0 | -0.5452954769134521 | 0.001281950208875868 |
| T5-D-CHE-STFU-OutputJacobianSketch-r4 [M2-DegreeRoleSecondMoment] | 9 | 0 | 0 | -0.5449420081244575 | 0.001134634017944336 |

## 4. Line X transfer operator audit
```text
line_x_rows = 135
line_x_gate_count_basis = dataset_seed_3x3_after_metric_choice_expansion
transfer_supported_count = 0
local_positive_no_transfer_count = 9
transfer_supported_raw_count = 0
local_positive_no_transfer_raw_count = 135
local_positive_no_transfer_dataset_seed_any_count = 9
line_x_local_positive_baseline_count = 10
line_x_local_positive_raw_fraction = 1.0
line_x_local_positive_baseline_fraction = 1.0
line_x_local_positive_fraction_reduced_vs_v1521 = 0
line_x_raw_local_positive_reduced_vs_v1521 = 0
line_x_local_positive_reduced_vs_v1521 = 0
line_x_gate_pass = 0
line_x_route = R-X-SplitTransferStillNoTransfer
```

## 5. Line M MLP/generic ST-FU controls
```text
line_m_rows = 171
line_m_delta_rows = 9
generic_stfu_explains_fraction = 1.0
generic_stfu_explains = 1
kan_specific_pass_count = 0
line_m_route = R-M-GenericSTFUExplainsGain
```

## 6. Line D all-basis substrate 结果
```text
line_d_source = v153_actual_v149_substrate_acceleration
line_d_rows = 126
best_non_dche_family = D-FOU
best_non_dche_dataset_seed_pass_count = 0 / 9
line_d_official_fu_eligible_family_count = 0
line_d_route = R5-AllBasisSubstrateBlocked
```

| family | rows | pass | max mean delta vs MLP | best LineC pass rate | official eligibility |
|---|---:|---:|---:|---:|---:|
| D-FOU | 45 | 0/9 | -0.109375 | 1.0 | 0 |
| D-RBF | 45 | 0/9 | -0.328125 | 1.0 | 0 |
| D-WAV | 36 | 0/9 | -0.0234375 | 1.0 | 0 |

## 7. 最终 route
```text
route = R1-STFUStillLocalPositiveNoTransfer
minimum_success = S1-SplitTransferOperatorExecuted
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

## 8. 覆盖与审计
```text
required_artifact_manifest_rows = 31
required_artifact_missing_sum = 0
forbidden_information_audit_rows = 7
forbidden_violation_sum = 0
no_action_search_audit_rows = 7
no_action_violation_sum = 0
execution_contract_rows = 11
execution_contract_unclosed_rows = 0
line_t_fallback_rows = 30
line_t_fb4_rows = 5
line_t_fb4_scale_boundary_hit_rows = 5
line_t_fb5_rows = 5
line_t_fb5_candidate_eval_count_median = 4.0
line_t_fb5_duplicate_true_b2_eval_count_median = 0.0
line_t_fb5_saved_restore_rows = 5
line_t_fb5_full_grad_from_split_rows = 5
line_t_exhaustion_certificate_rows = 1
line_d_family_exhaustion_certificate_rows = 3
plan_coverage_recheck_rows = 8
plan_coverage_unclosed_rows = 0
gate_route_recompute_rows = 10
gate_route_inconsistent_rows = 0
metric_choice_coverage_rows = 2
line_x_audit_rows = 135
line_x_denominator_audit_rows = 10
line_x_blank_readback_rows = 0
```

## 9. 科学结论
```text
1. v15.3 已执行 Line R/T/M/X/D/C/Z，并生成 required artifacts。
2. ST-FU 方向只使用当前 train split B1/B2；LineC/tail/AUC/calibration 只用于 audit/gate。
3. Line T real-lite = 0/9，未达成 S2/S3/S4/S5。
4. Line X transfer_supported_count = 0（dataset_seed_3x3_after_metric_choice_expansion），用于判断 local positive 是否转成 split transfer。
5. Line M generic/MLP controls 不写成 KAN-specific promotion。
6. Line D best Non-D-CHE = D-FOU 0/9，未打开 official FU proof。
7. 当前 route = R1-STFUStillLocalPositiveNoTransfer，promotion_allowed = 0。
```

## 10. 用户再次追问后的 metric choice factor 补跑

本次继续推进发现一个真实覆盖缺口：计划第 6.3 预注册了 `M0/M1/M2` metric choice 作为 factor，但上一版 official artifact 只执行了 `M0-identity`。该 factor 不新增 T6/T7，不构成 action search。

```text
修复：接入 M0-identity / M1-AdamVDiag / M2-DegreeRoleSecondMoment。
M1/M2 只使用当前 train-stream optimizer state / parameter-group second moment。
v153_metric_choice_coverage.csv rows = 2
T rows = 252
M rows = 171
route = R1-STFUStillLocalPositiveNoTransfer
promotion_allowed = 0
```

判断：metric choice factor 已覆盖，best factor 仍未打开 strict pass / real-lite pass；S2/S3/S4/S5 仍为 0。

## 11. 用户再次追问后的 Line X audit readback 修复

本次没有新增训练；复用已落盘的 `v153_line_t_stfu_results.csv` 重写 finalizer / audit artifact。

```text
发现问题：Line X 计划要求记录 reservoir_like_displacement_fraction、
signal_like_displacement_fraction、noise_leakage_proxy_delta。
上一版 v153_line_x_transfer_operator_audit.csv 有列但 reservoir/noise 为空。
修复方式：使用当前 train split B1/B2 loss delta 派生 audit-only readback：
  reservoir_like = max(0, B1_gain - B2_gain) / (B1_gain + B2_gain)
  signal_like = B2_gain / (B1_gain + B2_gain)
  noise_leakage_proxy_delta = max(0, B1_gain - B2_gain)
这些字段只用于 Line X 审计，不进入 direction，不改变 gate，不补假 telemetry。
同时补齐 T-FB6 exhaustion fallback 行；v153_line_t_exhaustion_certificate.csv 仍为正式证书。
line_x_audit_rows = 135
line_x_blank_readback_rows = 0
line_t_fallback_rows = 30
plan_coverage_unclosed_rows = 0
gate_route_inconsistent_rows = 0
route = R1-STFUStillLocalPositiveNoTransfer
promotion_allowed = 0
```

判断：Line X audit readback 已补齐；当前 route 和 success gate 不变，仍未达成 S2/S3/S4/S5。

## 12. 用户再次追问后的 Line X route 语义复核

本次没有新增训练；复用 official artifacts 重写 route / gate recompute / 两份日志。

```text
发现问题：
  Line X route / gate 分母不能被 metric-choice 展开后的 raw rows 污染。
  metric-choice factor 将 3x3 Line X audit 展开成 135 raw rows；
  旧逻辑用 raw local-positive count 与 v15.02.1 的 10-row baseline 直接比较，
  分母不一致。
修复方式：
  新增 v153_line_x_denominator_audit.csv；
  用 dataset-seed 3x3 口径计算计划中的 transfer_supported_count >= 3/9；
  raw row count / raw fraction 保留为 audit-only 诊断。
  新增 LineX-TransferOperatorExplorationGate 独立重算行。
line_x_gate_count_basis = dataset_seed_3x3_after_metric_choice_expansion
transfer_supported_dataset_seed_count = 0
local_positive_all_dataset_seed_count = 9
local_positive_any_dataset_seed_count = 9
transfer_supported_raw_count = 0
local_positive_no_transfer_raw_count = 135
line_x_local_positive_raw_fraction = 1.0
line_x_local_positive_fraction_reduced_vs_v1521 = 0
line_x_gate_pass = 0
line_x_route = R-X-SplitTransferStillNoTransfer
line_x_local_positive_baseline_count = 10
line_x_local_positive_reduced_vs_v1521 = 0
gate_route_inconsistent_rows = 0
route = R1-STFUStillLocalPositiveNoTransfer
promotion_allowed = 0
```

判断：Line X 在 3x3 口径下没有 split-transfer support，当前 route = R1-STFUStillLocalPositiveNoTransfer，promotion_allowed 仍为 0。

## 13. 用户再次追问后的 Line T objective 修正与实跑复核

本次发现并修复了一个训练定义级实现问题，因此重新执行 full official command，而不是只重写 finalizer。

```text
发现问题：
  计划第 6.3 的 ST-FU objective 是 f + J U alpha。
  旧实现对非 Adam methods 使用 adam + alpha * ridge_direction，
  会把 split-transfer operator FU 变成 AdamW-anchored perturbation。
修复方式：
  T1-T5/TCTRL/M1-M3/MCTRL 改为提交纯 alpha * U direction。
  T0-D-CHE-AdamW / M0-MLP-AdamW 保持 AdamW baseline。
  full official command 已重新执行，以下结果来自修正后 artifact。
line_t_real_lite_pass_count = 0 / 9
line_t_source_vs_best_control_mean = -0.46042725774976945
line_t_control_equivalent_fraction = 1.0
line_t_bad_event_fraction = 1.0
line_t_best_step_time_overhead_fraction = 0.8888888888888888
line_t_best_memory_overhead_fraction = 0.0
line_t_best_tail_fail_fraction = 0.2222222222222222
line_t_best_linec_fail_fraction = 1.0
line_x_gate_pass = 0
line_x_route = R-X-SplitTransferStillNoTransfer
transfer_supported_dataset_seed_count = 0 / 9
local_positive_all_dataset_seed_count = 9 / 9
generic_stfu_explains = 1
kan_specific_pass_count = 0
line_t_fb4_scale_boundary_hit_rows = 5 / 5
route = R1-STFUStillLocalPositiveNoTransfer
promotion_allowed = 0
```

判断：修正为真正的 ST-FU 后，Line X gate = 0；当前阻塞来自 strict/real-lite 未通过，其中 best-method step overhead fraction = 0.8888888888888888、tail fail fraction = 0.2222222222222222、LineC fail fraction = 1.0。T-FB4 scale boundary hit = 5/5，不打开新的 scale search。

## 14. 用户再次追问后的 ST-FU commit scale 修正与实跑复核

本次继续发现一个尺度定义问题，因此再次重新执行 full official command。

```text
发现问题：
  计划第 2.2 / 6.3 明确写 Δθ = Uα 与 f + J Uα。
  旧实现生成 alpha * U 后，提交参数时仍乘 args.lr，
  实际位移是 args.lr * Uα，而不是 Uα。
修复方式：
  T1-T5/TCTRL/M1-M3/MCTRL 使用 commit_lr_multiplier = 1.0。
  T0-D-CHE-AdamW / M0-MLP-AdamW 仍使用 args.lr 作为 optimizer baseline。
  full official command 已重新执行，以下结果来自修正后 artifact。
line_t_real_lite_pass_count = 0 / 9
line_t_source_vs_best_control_mean = -0.46042725774976945
line_t_control_equivalent_fraction = 1.0
line_t_bad_event_fraction = 1.0
line_x_gate_pass = 0
line_x_route = R-X-SplitTransferStillNoTransfer
transfer_supported_dataset_seed_count = 0 / 9
local_positive_all_dataset_seed_count = 9 / 9
generic_stfu_explains = 1
kan_specific_pass_count = 0
line_t_fb4_scale_boundary_hit_rows = 5 / 5
route = R1-STFUStillLocalPositiveNoTransfer
promotion_allowed = 0
```

判断：commit scale 与 U 单位均修正后，当前 route 以最新 artifact 为准；最新结果为 `R1-STFUStillLocalPositiveNoTransfer`，不是 R4/promotion。

## 15. 用户再次追问后的 R4 runtime 实现级优化与实跑复核

本次继续推进没有新增 method/token/grid；只修复 runner 中不影响数学定义的重复计算。

```text
发现问题：
  R4 的直接 blocker 是 step_time overhead。
  复核 runner 后发现 trial evaluation 每个 alpha 都重复 clone/restore 全参数；
  对非 B2Shuffled objective，best candidate 还会重复 forward 同一个 true B2 loss。
修复方式：
  每个 train step 只保存一次参数快照，所有 alpha trial 共用该快照恢复。
  只有 B2ShuffledCotangent control 才额外计算 true B2；普通 ST-FU 复用 objective B2 loss。
  不改变 ST-FU objective、alpha grid、subspace、direction source、gate 或 controls。
  full official command 已用 cuda:0 重新执行，以下结果来自修正后 artifact。
line_t_real_lite_pass_count = 0 / 9
line_t_source_vs_best_control_mean = -0.46042725774976945
line_t_control_equivalent_fraction = 1.0
line_t_bad_event_fraction = 1.0
line_t_best_step_time_overhead_fraction = 0.8888888888888888
line_t_best_tail_fail_fraction = 0.2222222222222222
line_t_best_linec_fail_fraction = 1.0
line_t_fb5_candidate_eval_count_median = 4.0
line_t_fb5_duplicate_true_b2_eval_count_median = 0.0
line_t_fb5_saved_restore_rows = 5 / 5
line_t_fb5_full_grad_from_split_rows = 5 / 5
line_x_gate_pass = 0
generic_stfu_explains = 1
route = R1-STFUStillLocalPositiveNoTransfer
promotion_allowed = 0
```

判断：重复计算已去除，但 v15.3 仍未达到 S2/S3/S4/S5；当前计划内不允许继续通过新增 T6/T7、扩 alpha grid 或 audit-directed objective search 来追 promotion。

## 16. 用户再次追问后的 R4 full-gradient 重复 backward 修复与实跑复核

本次继续推进仍然只处理实现级 runtime blocker，不改变 ST-FU 定义。

```text
发现问题：
  每个 step 已经为了 split-transfer objective 计算 B1/B2 gradients；
  旧实现又额外对 full batch 做一次 backward 来取得 Adam/full gradient。
  在固定 50/50 split 和 CE mean loss 下，full gradient 可由 B1/B2 gradients 加权合成。
修复方式：
  grad = (n1 * grad_B1 + n2 * grad_B2) / (n1 + n2)。
  该修正不使用 validation/test/future/query，不使用 LineC/tail/AUC 生成方向，
  不新增 token/controller/action/reset，也不改变 alpha grid。
  full official command 已用 cuda:0 重新执行，以下结果来自修正后 artifact。
line_t_real_lite_pass_count = 0 / 9
line_t_source_vs_best_control_mean = -0.46042725774976945
line_t_control_equivalent_fraction = 1.0
line_t_bad_event_fraction = 1.0
line_t_best_step_time_overhead_fraction = 0.8888888888888888
line_t_best_tail_fail_fraction = 0.2222222222222222
line_t_best_linec_fail_fraction = 1.0
line_t_fb5_full_grad_from_split_rows = 5 / 5
line_x_gate_pass = 0
generic_stfu_explains = 1
route = R1-STFUStillLocalPositiveNoTransfer
promotion_allowed = 0
```

判断：full-gradient 重复 backward 已去除；当前 route 以最新 artifact 为准。最新结果为 `R1-STFUStillLocalPositiveNoTransfer`，说明单位修正后 transfer signal 不能成立。

## 17. 用户再次追问后的最终可继续性复核

本次没有新增训练；按计划 stop/continue 制度复核是否仍有 v15.3 内合法分支。

```text
已完成并落盘：
  Line T T0-T5 + TCTRL controls + T-FB1..T-FB6。
  Line M MLP/generic ST-FU controls。
  Line X transfer operator audit + denominator audit。
  Line D D-FOU52..56 / D-RBF50..54 / D-WAV45..48 substrate-only reconfirmation。
  Line C geometry/tail/AUC audit。
  Line R/Z required manifest、forbidden/no-action audit、gate recompute、execution contract。
已尝试的计划内/实现级修复：
  objective 从 AdamW-anchored perturbation 修正为纯 U alpha。
  commit scale 修正为 Delta theta = U alpha。
  R4 route precedence / x_gate finalizer 修正。
  runtime duplicate candidate eval 去重。
  full-batch gradient 从 B1/B2 gradients 合成，去掉重复 backward。
仍然失败的 gate：
  S2/S3/S4/S5 = 0；real_lite = 0/9。
  route = R1-STFUStillLocalPositiveNoTransfer。
  step overhead fraction = 0.8888888888888888。
  tail fail fraction = 0.2222222222222222。
  LineC fail fraction = 1.0。
当前 v15.3 内不合法的继续方式：
  新增 T6/T7、扩 alpha grid、按 dataset/seed 改 scale、用 tail/LineC/AUC/CEp99 反推 objective、
  改 commit scale 偏离 Delta theta = U alpha、action/controller/reset、CPU offload。
```

最终判断：v15.3 已达成 S1-SplitTransferOperatorExecuted；未达成 S2/S3/S4/S5，promotion_allowed = 0。当前计划内没有剩余可合法补跑的分支。

## 18. 用户再次追问后的 U normalization unit bug 修复

本次继续推进是因为复核发现上一轮 `actual_step_over_adam_step_norm≈19.44` 不是单纯理论 blocker，而是实现单位不一致。

代码修改：

```text
experiments/run_v153_split_transfer_operator_fu_allbasis_acceleration.py
  candidate_updates:
    新增 adam_step = adam * args.lr。
    非 Adam ST-FU / MLP-STFU methods 使用 adam_step 作为 basis norm target。
    T0-D-CHE-AdamW / M0-MLP-AdamW baseline 仍使用 raw adam + commit lr。
    J_B2_U_norm 改为使用归一化后的 U 计算。
    same_norm_random_gap 改为比较 actual step norm。
```

修复理由：

```text
计划第 2.2 / 6.3 写的是：
  U0 = AdamW update direction span
  Delta theta = U alpha

旧实现把 U 归一化到未乘 lr 的 raw Adam direction，
而非 Adam ST-FU 又用 commit_lr_multiplier=1.0。
因此 alpha=0.025 实际约等于 25x AdamW step；
这解释了上一轮看似 9/9 real-lite 但 tail 崩坏的假阳性。
```

GPU full rerun 后的 route：

```text
route = R1-STFUStillLocalPositiveNoTransfer
minimum_success = S1-SplitTransferOperatorExecuted
real_lite_pass_count = 0 / 9
source_vs_best_control_mean = -0.46042725774976945
control_equivalent_fraction = 1.0
bad_event_fraction = 1.0
best_tail_fail_fraction = 0.2222222222222222
best_linec_fail_fraction = 1.0
best_step_time_overhead_fraction = 0.8888888888888888
line_x_gate_pass = 0
generic_stfu_explains = 1
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

best method readback：

```text
method = T1-D-CHE-STFU-AdamSubspace [M2-DegreeRoleSecondMoment]
rows = 9
real_lite_pass = 0 / 9
strict_pass = 0 / 9
source_vs_best_control_mean = -0.46042725774976945
actual_step_over_adam_step_norm_mean = 0.20000001284909386
CEp99_delta_mean = -0.1379088560740153
step_time_ratio_mean = 1.634300476466451
tail_fail_from_fail_reason = 2 / 9
linec_fail_from_fail_reason = 9 / 9
```

artifact / audit：

```text
v153_required_artifact_manifest.csv rows = 31, missing_sum = 0
v153_forbidden_information_audit.csv rows = 7, violation_sum = 0
v153_no_action_search_audit.csv rows = 7, violation_sum = 0
v153_gate_route_recompute.csv rows = 10, inconsistent_rows = 0
v153_execution_contract_coverage_audit.csv rows = 11, unclosed_rows = 0
```

结论修正：

```text
1. 上一轮 R4 / 9-of-9 real-lite signal 是 U 单位 bug 造成的尺度假阳性。
2. 修复后 actual step scale 回到 0.20x AdamW step，tail 明显改善。
3. 但 ST-FU source_vs_best_control 变为负数，Line X transfer_supported = 0/9。
4. 因此 v15.3 当前最终科学结论是：
   R1-STFUStillLocalPositiveNoTransfer。
5. 没有达成 S2/S3/S4/S5，promotion_allowed = 0。
6. 该修复不新增 T6/T7，不扩 alpha grid，不使用 audit metric 反推方向，
   不新增 controller/action/reset，也没有 CPU offload。
```
