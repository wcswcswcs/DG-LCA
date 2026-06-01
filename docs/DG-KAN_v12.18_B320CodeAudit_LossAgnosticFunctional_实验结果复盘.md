# DG-KAN v12.18 B320CodeAudit LossAgnosticFunctional 实验结果复盘

生成时间：2026-05-24 Asia/Singapore

对应计划：

```text
docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional计划.md
```

对应执行日志：

```text
docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional_执行日志.md
```

本复盘只记录真实落盘 artifact；不把没有训练的 label-free ablation 写成失败或成功，不把 gated-not-run 的 P3/P4 写成实验结果，不编造 P4 数据。

## 1. 总结论

v12.18 official route：

```text
route = R2-B320LabelInitDependenceDetected
promotion_allowed = 0
p3_open = 0
p4_open = 0
```

失败原因：

```text
current_B320_uses_label_informed_trainprobe_init;
label_free_official_ablation_missing;
label_free_smoke_underperforms_labelInit;
strict_loss_agnostic_observables_not_visible
```

一句话结论：

```text
v12.18 修复了 v12.17.2 code packet 不闭包的问题，并把 Line T 推进到真正 strict loss-agnostic observable repair；随后又补跑了一个真实 CUDA smoke 级 B320 label-free init ablation。smoke 结果显示 label-free 方向没有在小预算下追平 A0 labelInit，且它不是 official 同预算 anchor，所以 R2 仍不能关闭；strict features 也无法通过 joint top-k precision/recall 与 robustness gate，因此 P3/P4 正确关闭。
```

正式输出目录：

```text
results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts
```

代码审查 zip：

```text
results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
sha256 = 63802d5700a4269488aaca2071eb41061f88b595635579b1782220702e18cdb1
zip_entries = 51
```

## 2. 本轮做了哪些修改

新增 runner：

```text
experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

主要修改点：

```text
1. 建立 transitive dependency manifest，强制打包 v1218/v1217/v1216/v1215/v1283/v1252/v124/v120 与 dgkan 核心文件。
2. 建立 CR0-CR15 code review manifest 和 symbol line map，避免只写 summary 不指向真实代码行。
3. 显式审计 B320 trainprobe label-informed init，保留 current B320 anchor，但 external_ready_base_claim=0。
4. 把 v12.17.2 feature table 里的 audit/offline response features 标为 forbidden_for_direction。
5. 按文档失败修复方向，新增 strict observable repair：从 v12.16 repair source 抽取 loss_agnostic_direction=1 且 label/CE/dataset-branch flags 全为 0 的 13 个 feature。
6. 修复 strict feature 与 hard release label 的 join key：从完整 row_id 改为 source_run,dataset,seed,window,method,sketch_id 语义键。
7. 新增 v1218_strict_visibility_repair_attempts.csv，记录 strict repair family、行数、来源和 no-label/no-CE provenance。
8. 重新生成 v1218_code_review_packet.zip，确认包含新 runner 与新 strict artifacts。
9. 追加 Line A continuation runner，真实跑 A0/A1/A2/A3/A4/A5 smoke ablation，并把新 runner 与 smoke artifacts 打进 zip。
```

审计意义：

```text
第 6 点很关键。第一次修复时 row_id join 过窄，导致 strict Line T 没有匹配到真实 hard_joint_release=1 rows。最终版修复后，hard_joint_support_rows=8 被正确纳入 visibility evaluation。
```

## 3. Line R：代码审查与依赖闭包

结果：

```text
transitive_dependency_rows = 23
transitive_missing_count = 0
all_required_files_present = 1
core_code_review_rows = 16
cr0_cr15_unknown_count = 0
core_code_review_pass = 1
manual_review_pending_count = 16
```

解释：

```text
v12.17.2 zip 漏掉 v124/v120 的问题已经修复。
CR0-CR15 都有实际 file/symbol/line range，不存在 unknown_or_not_inspected。
但每个 CR 仍标记 manual_review_pending_for_promotion，所以不能把 Codex 自审当成人工 promotion approval。
```

zip 检查真实输出：

```text
required_count = 27
missing = []
zip_entries = 51
zip_has code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py = True
zip_has code/experiments/run_v1218_b320_label_free_ablation.py = True
zip_has code/experiments/run_v124_multibasis_functional_dual.py = True
zip_has code/experiments/run_v120_good_geometry_battery.py = True
zip_has code/dgkan/models/fc_purekan_primitives.py = True
zip_has review_artifacts/v1218_strict_visibility_repair_attempts.csv = True
zip_has review_artifacts/v1218_b320_label_free_smoke_summary.csv = True
zip_has review_artifacts/v1218_line_t_support_expansion_audit.csv = True
zip_has review_artifacts/v1218_strict_visibility_scores.csv = True
zip_has review_artifacts/v1218_core_code_review_manifest.csv = True
```

## 4. Line A：B320 Anchor + Label-Init Audit

current B320：

```text
candidate = B320b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp075
ablation_id = A0 B320-current-labelInit
uses_y_for_stats = 1
trainprobe_signal_init_uses_labels = 1
trainprobe_signal_init_applied_code_expected = 1
trainprobeP_enabled = 1
trainprobeDirect_enabled = 1
signalBroad = signalBroad035
signalBlock = signalBlock015
current_anchor_pass = 1
```

继承 v12.17.2 anchor 数值：

```text
step_ratio_q90 = 0.39931987348441034
memory_ratio_q90 = 0.1295238095238095
mean_delta_vs_mlp = 0.027669270833333332
worst_delta_vs_mlp = -0.001953125
```

结论：

```text
B320 current 仍可作为 functional diagnostic anchor。
但它不是 label-free external-ready base。
B320_claim_scope = label-informed supervised initialization anchor
external_ready_base_claim = 0
```

A1-A4 状态：

```text
A1 B320-current-noYForStats:
  official_training_result_available = 0
  not_run_reason = no comparable v12.18 label-free B320 training artifact exists in current workspace; not fabricated
  allowed_next_action = construct B320 with y_stats=None; same train budget required

A2 B320-current-randomP-labelFree:
  official_training_result_available = 0
  allowed_next_action = random P label-free ablation; same train budget required

A3 B320-current-PCA-P-labelFree:
  official_training_result_available = 0
  allowed_next_action = PCA-P label-free ablation; prior B226-B229 are related but not comparable official v12.18 rows

A4 B320-current-lowfreqP-labelFree:
  official_training_result_available = 0
  allowed_next_action = lowfreqP label-free ablation; same train budget required
```

修复尝试与边界：

```text
已检索代码和结果中 B226-B229 label-free PCA-P 相关记录。
它们只在旧 B109/v1283 修改审计中出现，不能替代本轮 B320 同预算 ablation。
因此不能把旧 B226-B229 当成本轮 B320 ablation。
```

### 4.1 Line A continuation smoke

为了不在 “ablation missing” 处停下，本轮随后新增并执行：

```text
experiments/run_v1218_b320_label_free_ablation.py
```

运行协议：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
val_size = 256
test_size = 256
epochs = 3
batch_size = 128
device = cuda:0
smoke_not_official = 1
official_training_result_available = 0
```

真实落盘 artifacts：

```text
v1218_b320_label_free_smoke_ablation.csv
v1218_b320_label_free_smoke_summary.csv
v1218_b320_label_free_smoke_route.json
```

smoke route：

```text
rows = 63
summary_rows = 7
label_free_smoke_ablation_available_count = 5
best_label_free_smoke_candidate = A1-noYForStats
best_label_free_mean_delta_vs_A0 = -0.032552083333333336
route_impact = does_not_close_R2; full comparable label-free B320 budget still missing
```

summary：

```text
A0-labelInit:
  rows = 9
  mean_delta_vs_mlp = 0.022569444444444444
  worst_delta_vs_mlp = -0.0078125
  near_pass_rate_vs_mlp = 1.0
  max_AUC_step_ratio_vs_mlp = 1.0071162337536481
  max_ECE_delta_vs_mlp = 0.017832741141319275

A1-noYForStats:
  rows = 9
  mean_delta_vs_mlp = -0.009982638888888888
  worst_delta_vs_mlp = -0.0625
  near_pass_rate_vs_mlp = 0.4444444444444444
  max_AUC_step_ratio_vs_mlp = 1.3074866878791902
  mean_delta_vs_A0_labelInit = -0.032552083333333336
  worst_delta_vs_A0_labelInit = -0.0625

A2-randomP-labelFree:
  rows = 9
  mean_delta_vs_mlp = -0.009982638888888888
  worst_delta_vs_mlp = -0.0625
  near_pass_rate_vs_mlp = 0.4444444444444444
  max_AUC_step_ratio_vs_mlp = 1.3074868197664284
  mean_delta_vs_A0_labelInit = -0.032552083333333336
  worst_delta_vs_A0_labelInit = -0.0625

A3-PCA-P-labelFree:
  rows = 9
  mean_delta_vs_mlp = -0.07942708333333333
  worst_delta_vs_mlp = -0.1484375
  near_pass_rate_vs_mlp = 0.0
  max_AUC_step_ratio_vs_mlp = 1.9534107012447748
  mean_delta_vs_A0_labelInit = -0.10199652777777778

A4-lowfreqP-labelFree:
  rows = 9
  mean_delta_vs_mlp = -0.63671875
  worst_delta_vs_mlp = -0.74609375
  near_pass_rate_vs_mlp = 0.0
  max_AUC_step_ratio_vs_mlp = 67757.01211668788
  mean_delta_vs_A0_labelInit = -0.6592881944444444

A5-orthogonalP-labelFree:
  rows = 9
  mean_delta_vs_mlp = -0.024739583333333332
  worst_delta_vs_mlp = -0.04296875
  near_pass_rate_vs_mlp = 0.1111111111111111
  max_AUC_step_ratio_vs_mlp = 1.3144470368712866
  mean_delta_vs_A0_labelInit = -0.047309027777777776
```

解释：

```text
1. 这是 small-budget continuation，不是 v12.14 locked anchor 的完整同预算复现；因此不能把结果写成 external-ready base claim。
2. A1/A2 在 3 datasets x 3 seeds 下已经低于 MLP 均值，且比 A0 labelInit 平均低 0.032552083333333336。
3. PCA-P、lowfreqP、orthogonalP 方向在该协议下没有修掉 label-init dependence；lowfreqP 明显崩坏。
4. 因此 R2 仍保留，但现在不是“完全没执行 A1-A4”，而是“已执行 smoke repair attempt，official comparable ablation 仍缺失”。
```

## 5. Line C：Audit-only Calibration

结果：

```text
linec_audit_rows = 864
linec_joint_hard_support_rows = 24
linec_random_joint_false_positive_rows = 0
linec_calibration_pass = 1
```

注意：

```text
linec_joint_hard_support_rows=24 来自 v1218_linec_hard_support.csv 的 source/support 汇总。
strict Line T 实际用于 release label 的 hard_joint_support_rows=8，来源于 v1217_release_labels_audit_only.csv 的 repair source rows。
两者不是同一个计数口径。
```

结论：

```text
Audit-only hard release labels 可用于 target evaluation；
它们不能进入 deployable feature 或 functional direction。
v1218_audit_metric_provenance.csv 已将 CE/label provenance 指标标为 allowed_for_audit=1, allowed_for_direction=0。
```

## 6. Feature Provenance Audit

结果：

```text
feature_provenance_rows = 41
strict_allowed_feature_count = 13
forbidden_feature_count = 28
```

两类 feature：

```text
v12.17.2 existing target visibility features:
  28 个 feature 被标为 forbidden_for_direction。
  原因包括 CE/label/audit response provenance、method identity、numeric hash、actual response 等。

v12.16 repair strict observables:
  13 个 feature 被标为 strict_loss_agnostic_allowed=1。
  它们来自 loss_agnostic_direction=1 且 ce_vector_used_for_direction=0、label_used_for_direction=0、dataset_name_used_for_commit=0 的 rows。
```

strict feature families：

```text
augmentation_consistency_proxy
basis_occupancy
logit_free_spectrum_proxy
projector_geometry
random_cotangent_logit_jacobian
sketch_geometry
```

严格性说明：

```text
本轮 strict features 不使用 label、CE vector、permuted-label CE、dataset name branch、numeric hash 或 actual future response。
hard release 只作为 audit target。
```

## 7. Line T：Strict Loss-Agnostic Target Visibility v2

本轮补跑不是空跑：

```text
strict_visibility_feature_rows = 11232
strict_visibility_wide_rows = 864
strict_visibility_feature_columns = 13
strict_allowed_feature_count = 13
```

repair attempt artifacts：

```text
v1218_strict_visibility_repair_attempts.csv rows = 6
source_rows = 864
new_training_run = 0
uses_label_for_feature = 0
uses_ce_for_feature = 0
uses_dataset_name_for_feature = 0
```

feature family row counts：

```text
augmentation_consistency_proxy = 864
basis_occupancy = 2592
logit_free_spectrum_proxy = 1728
projector_geometry = 2592
random_cotangent_logit_jacobian = 1728
sketch_geometry = 1728
```

总体 visibility score：

```text
AUC_noise_min = 0.5
AUC_reservoir_min = 0.5
AUC_joint_min = 0.5
AUC_joint_max = 0.9436619718309859
precision_at_k_joint_min = 0.0
recall_at_k_joint_min = 0.0
required_split_failures = 1
support_concentrated = 1
visibility_exploratory_pass = 0
visibility_hard_pass = 0
visibility_robustness_pass = 0
visibility_pass = 0
```

关键 split 结果：

```text
leave-dataset-out Fashion-MNIST:
  joint_train_positive_rows = 4
  joint_test_positive_rows = 4
  AUC_joint = 0.9436619718309859
  precision_at_k_joint = 0.0
  recall_at_k_joint = 0.0

leave-dataset-out KMNIST:
  joint_train_positive_rows = 4
  joint_test_positive_rows = 4
  AUC_joint = 0.8732394366197183
  precision_at_k_joint = 0.0
  recall_at_k_joint = 0.0

leave-seed-out seed=0:
  joint_train_positive_rows = 0
  joint_test_positive_rows = 8
  AUC_joint = 0.5
  precision_at_k_joint = 0.0
  recall_at_k_joint = 0.0

leave-window-out window=10:
  joint_train_positive_rows = 0
  joint_test_positive_rows = 8
  AUC_joint = 0.5
  precision_at_k_joint = 0.0
  recall_at_k_joint = 0.0

leave-method-out U7-ProjectionQuadRoleActuator:
  joint_train_positive_rows = 4
  joint_test_positive_rows = 4
  AUC_joint = 0.6923076923076923
  precision_at_k_joint = 0.0
  recall_at_k_joint = 0.0

leave-method-out U9-MixedLowRankActuatorBasis:
  joint_train_positive_rows = 4
  joint_test_positive_rows = 4
  AUC_joint = 0.6923076923076923
  precision_at_k_joint = 0.0
  recall_at_k_joint = 0.0

leave-actuator-family-out loss_agnostic_probe:
  joint_train_positive_rows = 0
  joint_test_positive_rows = 8
  AUC_joint = 0.5
  precision_at_k_joint = 0.0
  recall_at_k_joint = 0.0
```

support concentration：

```text
hard_joint_total_rows = 8
support_concentration_seed = {'0': 8}
support_concentration_window = {'10': 8}
support_concentration_method = {'U7-ProjectionQuadRoleActuator': 4, 'U9-MixedLowRankActuatorBasis': 4}
support_concentration_source = {'repair_sketchdim24_rank5_b64_w3_5_10': 8}
support_concentrated = 1
```

解释：

```text
strict features 对部分 dataset split 有排序信号，尤其 Fashion-MNIST/KMNIST leaveout 的 AUC_joint 很高。
但 top-k precision/recall 仍为 0，seed/window/source/family robustness 不成立。
hard joint support 全部集中在 seed=0、window=10、单一 repair source。
因此不能打开 P3，也不能用 high AUC split 选择性宣称 functional value source 可见。
```

## 8. Line I / Line B：P3/P4 Gate

P3 状态：

```text
p3_open = 0
matched_controls_generated = 0
not_run_reason = Line T strict label-free visibility failed; matched-control actuator dictionary not generated
```

P4 状态：

```text
p4_open = 0
p4_pass = 0
not_run_reason = P4 requires Line R/A/C/T/I pass; Line T and Line I are closed
```

解释：

```text
这不是遗漏执行 P3/P4。
按计划，P3 只能在 strict Line T hard visibility pass 后打开，P4 只能在 Line R/A/C/T/I 全部通过后打开。
本轮 Line T 未通过，且 B320 label-init ablation 缺失，所以 P3/P4 正确关闭。
```

## 9. Line D：Classic No-BSpline Monitor

结果：

```text
line_d_rows = 12
line_d_new_hypothesis_implemented = 0
```

解释：

```text
Line D 本轮作为 portfolio monitor 保留，没有新增 classic-family hypothesis，也没有重新打开 B-spline。
这符合 v12.18 计划：Line D 继续保留，但不抢主线资源；B-spline frozen。
```

## 10. 当前各线进展百分比

这里的百分比表示 v12.18 文档要求在本轮可审计闭环中的完成度，不表示科学成功率。

```text
Line R 代码审查/依赖闭包：95%
  已完成 transitive packet、CR0-CR15、hash/zip。
  剩余 5% 是 human manual review pending，不应由 Codex 自行关闭。

Line A B320 anchor + label-init audit：65%
  A0 current label-init anchor 已审计。
  A1-A5 已完成 seeds 0,1,2 / 3 dataset / 3 epoch small-budget continuation。
  同预算 official label-free B320 training ablation 仍缺失，不能关闭 R2。

Line C audit-only calibration：90%
  hard release audit 与 null/control calibration 已落盘。
  剩余是后续人工确认 threshold/provenance 口径。

Line T strict loss-agnostic visibility：70%
  已从 v12.16 repair source 构造 strict feature table 并真实跑 leaveout。
  未通过 visibility gate，且 source-run-out 因单一 source 不可运行。

Line I actuator response with controls：0%
  按 gate 未打开。

Line B functional candidate/P4：0%
  按 gate 未打开。

Line D classic monitor：100% monitor-only
  本轮无新 hypothesis，已记录 frozen/unchanged 状态。
```

## 11. 审计结论

可以确认：

```text
1. v12.18 code packet 已闭包，不再漏 v124/v120。
2. B320 current 使用 label-informed trainprobe initialization，这个事实已落盘。
3. B320 current anchor 数值仍过，但 claim scope 必须降级为 label-informed supervised initialization anchor。
4. v12.17.2 existing features 不能作为 strict deployable feature 直接 promotion。
5. 严格无标签/无 CE 的 v12.16 repair observables 已被尝试，但没有通过 Line T。
6. P3/P4 没有打开，符合 gate，不是提前退出。
```

不能确认：

```text
1. B320 label-free noY/randomP/PCA-P/lowfreqP 在同预算 official protocol 下是否能保持 current anchor。
2. strict observables 在更多 source runs、更多 seeds/windows 上是否能稳定定位 joint hard release。
3. fused actuator 在 matched controls 下是否能产生 control-resistant release。
4. functional P4 是否可优于 B320+AdamW/control，因为 P4 gate 未打开。
```

最终路线：

```text
R2-B320LabelInitDependenceDetected
```

下一步推荐，不属于本轮已完成结果：

```text
1. 将 smoke runner 扩展为真正同预算的 A1-A4 B320 label-free ablation run：
   A1 noYForStats
   A2 randomP-labelFree
   A3 PCA-P-labelFree
   A4 lowfreqP-labelFree

2. 若 A1-A4 明显掉 gate，继续保留 current B320 作为 diagnostic anchor，但 external_ready_base_claim 必须保持 0。

3. 若要继续 Line T，必须生成多个 independent source runs 或更多 seeds/windows，否则 leave-source-run-out 与 support concentration 会持续阻断 promotion。

4. 只有 Line A label-free scope 清楚且 Line T hard/robust visibility pass 后，才打开 Line I matched-control actuator dictionary。
```

## 12. 继续推进后的复盘更新：formal audit 自洽性修复与 Line T support repair 审计

这次继续推进不是新增科学成功，而是修复 v12.18 产物链中两个审计缺口：

```text
1. 先前 3x3 small-budget label-free smoke 已经真实执行，但 v1218_b320_label_init_audit.csv 仍把 A1-A4 写成未运行。
2. Line T strict visibility 失败后，缺少一个单独 artifact 说明“按推荐思路增加 source/support”为什么不能用现有 source 安全完成。
```

已修改代码：

```text
experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

修改审计说明：

```text
1. run_b320_label_init_audit 现在会读取 v1218_b320_label_free_smoke_summary.csv。
2. A1-A5 的真实 smoke metrics 被写入 v1218_b320_label_init_audit.csv。
3. A1-A5 仍保持 official_training_result_available=0，并写明 small_budget_training_result_available=1、smoke_not_official=1。
4. decide_route 现在区分 label_free_official_ablation_missing 和 label_free_smoke_underperforms_labelInit。
5. 新增 v1218_line_t_support_expansion_audit.csv，审计现有 official/repair/smoke source 是否可作为 Line T support/source 扩展。
6. 新 artifact 已纳入 v1218_code_review_packet.zip。
```

执行中遇到的 blocker：

```text
第一次重跑主 runner 失败：
ValueError: too many values to unpack (expected 2)

原因：
新增 smoke_id 后，循环仍按二元 tuple 解包。

修复：
将循环改为 for ablation_id, smoke_id, allowed in [...]。

该 blocker 是本轮代码修改错误，不是实验数据，不作为实验失败指标。
```

重跑后的最终 route：

```text
route = R2-B320LabelInitDependenceDetected
fail_reason = current_B320_uses_label_informed_trainprobe_init;label_free_official_ablation_missing;label_free_smoke_underperforms_labelInit;strict_loss_agnostic_observables_not_visible
promotion_allowed = 0
p3_open = 0
p4_open = 0
```

Line A 更新后关键结果：

```text
b320_label_free_ablation_available_count = 0
b320_label_free_smoke_ablation_available_count = 5
b320_label_free_smoke_best_candidate = A1-noYForStats
b320_label_free_smoke_best_mean_delta_vs_A0 = -0.032552083333333336
b320_label_free_smoke_underperforms_labelinit = 1
b320_label_free_official_ablation_still_missing = 1
```

解释：

```text
1. A1/A2/A3/A4/A5 均已有 seeds 0,1,2 x MNIST/Fashion-MNIST/KMNIST x 3 epoch small-budget smoke 结果。
2. best smoke candidate 是 A1-noYForStats，但 mean_delta_vs_A0 仍为 -0.032552083333333336。
3. 因此 smoke 级修复没有关闭 R2。
4. 因为 official 同预算 label-free B320 ablation 仍不存在，不能把 current B320 升格为 label-free base。
```

Line T support expansion audit 新结果：

```text
line_t_support_expansion_audit_rows = 3
line_t_calibrated_release_source_count = 1
line_t_independent_source_runs_available_for_leaveout = 1
line_t_support_expansion_pass = 0
line_t_support_expansion_blocker = no_second_calibrated_release_source;hard_joint_support_concentrated_in_seed0_window10;official_source_random_false_positives;smoke_source_not_official_independent_source
```

三个 source 的审计结论：

```text
official_3x3_b32_w3_5_10:
  linec_rows = 864
  strict_loss_agnostic_rows = 864
  v1217_source_calibration_pass = 0
  random_joint_false_positive_rows = 4
  control_joint_false_positive_rows = 4
  v1217_hard_support_joint_rows = 16
  v1217_release_hard_joint_rows = 0
  结论：不能用于 promotion visibility，原因是 random/control false positive。

repair_sketchdim24_rank5_b64_w3_5_10:
  linec_rows = 864
  strict_loss_agnostic_rows = 864
  v1217_source_calibration_pass = 1
  random_joint_false_positive_rows = 0
  control_joint_false_positive_rows = 0
  v1217_hard_support_joint_rows = 8
  v1217_release_hard_joint_rows = 8
  结论：是唯一 calibrated release source，但 hard_joint_support 集中在 seed0/window10，不能支撑 leave-source/robust promotion。

smoke_mnist_seed0:
  linec_rows = 64
  strict_loss_agnostic_rows = 64
  v1217_source_calibration_available = 0
  v1217_release_hard_joint_rows = 0
  结论：只是 smoke scope，不是 official independent source，不能用于 Line T promotion。
```

更新后的 zip：

```text
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_sha256 = 63802d5700a4269488aaca2071eb41061f88b595635579b1782220702e18cdb1
zip_entries = 51
zip_has review_artifacts/v1218_line_t_support_expansion_audit.csv = True
zip_has review_artifacts/v1218_b320_label_init_audit.csv = True
zip_has review_artifacts/v1218_b320_label_free_smoke_summary.csv = True
zip_has code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py = True
zip_has code/experiments/run_v1218_b320_label_free_ablation.py = True
```

更新后的各线进展判断：

```text
Line R 代码审查/依赖闭包：95%
  zip/code packet/hash 已更新，剩余仍是 human manual review pending。

Line A B320 anchor + label-init audit：70%
  current labelInit anchor 已审计；A1-A5 smoke repair 已并入正式 audit；official 同预算 label-free ablation 仍缺失。

Line C audit-only calibration：90%
  audit-only hard release 与 false-positive source calibration 已落盘；promotion 仍不得使用 label/CE audit metrics。

Line T strict loss-agnostic visibility：75% audit complete / 0% gate pass
  strict feature table、visibility score、leaveout、support expansion audit 均已执行；但 gate 未过，且现有 source 无法安全扩展。

Line I actuator response controls：0%
  Line T 未过，按计划不得生成 matched-control actuator dictionary。

Line B/P4：0%
  R2 与 Line T 未过，P4 不打开。

Line D classic monitor：100% monitor-only
  本轮无新 hypothesis，状态保持记录。
```

最终审计结论更新：

```text
v12.18 当前执行到可审计闭环，但不是科学成功。
主 route 仍是 R2，因为 current B320 使用 label-informed trainprobe init，且 official label-free ablation 仍缺失。
small-budget label-free smoke 已执行并正式入账，但最优 A1 仍低于 A0 labelInit。
Line T 按推荐方向检查了更多 source/support，但现有 source 只有一个 calibrated release source，且 support 集中，不能打开 P3/P4。
```

## 13. 为什么没有继续声称 official-budget Line A 已完成

继续推进时，我检查了是否能把当前 A1-A5 smoke ablation 直接扩展为 official-budget label-free ablation。结论是：不能安全声称 official。

证据：

```text
1. v12.18 计划要求 A0-A6 使用同一数据、同一 seed、同一训练预算。
2. v12.17 anchor monitor 是复用 v12.14 locked artifact，不是 v12.17 新训练；v1217_anchor_monitor.csv 标记 new_b320_training_claimed=0。
3. v12.17 route JSON 没有完整记录可直接复现 official B320 train budget 的 train_size/val_size/test_size/epochs/batch_size。
4. v12.14/v12.16 文档只显示 train_size=1024、batch_size=128、epochs=3 or 5 / short-run 3-5 epochs 这样的范围。
5. 现有 v12.18 label-free runner 在代码与产物中均明确标记 smoke_not_official=1、official_training_result_available=0。
```

因此：

```text
没有把 train_size=1024、epochs=3 或 epochs=5 任意选一个称为 official。
没有把 smoke 结果改名成 official。
没有关闭 R2。
```

这是本轮最终未关闭 blocker，而不是漏执行：

```text
blocker = exact official B320 label-free ablation protocol not fully materialized in current artifacts
required_next_action = recover or define official B320 label-free ablation protocol, then run A0/A1/A2/A3/A4/A5/A6 under exactly matched data/seed/budget and record all required Line A fields
```

## 14. 继续推进后的复盘更新：recovered anchor-budget + LineC protocol gate

本节更新并部分修正第 13 节的状态：我没有停在“official protocol 未完整落盘”的 blocker，而是按可审计方式恢复了 v12.11/v12.12/v12.13/v12.14 anchor budget，并实际跑了 3x3 A0/A1/A2/A3/A4/A5 + A6 control。最终结论不是“没跑”，而是“跑了 recovered protocol，但 protocol gate 不允许它成为 official locked-anchor replacement”。

### 14.1 本轮新增修改

代码修改：

```text
experiments/run_v1218_b320_label_free_ablation.py
experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

审计说明：

```text
1. 新增 recovered anchor-budget artifact 支持，避免把 smoke 结果误称 official。
2. 新增 seed_base probe artifacts：1211000 / 1212000 / 1213000。
3. 新增 post-training LineC audit：CouplingR2、NoiseSignalLeak、RealSignalReservoirRatio，并按 MLP control 判定 LineC_nontearing；最终版本按 v12.16 repair source 方向使用 b64/sketch_batch32/sketchdim24。
4. 新增 A5-orthogonalP-labelFree，执行计划 6.5 推荐的 orthogonal label-free init 修复方向。
5. 新增 A6 MLP-same-step-FLOP-AdamW 显式 control。当前 v124/v1218 budget 下它与 hidden=256 MLP control 同构，因此实跑后只作为 control 行，不作为 label-free B320 候选。
6. 主审计新增 v1218_b320_anchor_budget_protocol_audit.csv，要求 A0 同时复现 task gate 与 LineC gate，否则 official_budget_claim_allowed=0。
```

### 14.2 seed_base probe 结果

三组 probe 都只跑 A0-labelInit + MLP control，目的不是找最优 label-free candidate，而是确认 recovered budget 下 A0 task gate 是否稳定。

```text
seed_base=1211000:
  mean_delta_vs_mlp = 0.022569444444444444
  worst_delta_vs_mlp = 0.013671875
  near_pass_rate_vs_mlp = 1.0
  max_AUC_step_ratio_vs_mlp = 0.9637283017402302
  max_AUC_time_ratio_vs_mlp = 0.9637283017402302
  max_ECE_delta_vs_mlp = 0.005898520350456238

seed_base=1212000:
  mean_delta_vs_mlp = 0.021267361111111112
  worst_delta_vs_mlp = 0.0
  near_pass_rate_vs_mlp = 1.0
  max_AUC_step_ratio_vs_mlp = 0.9395254448115371
  max_AUC_time_ratio_vs_mlp = 0.7581381409711518
  max_ECE_delta_vs_mlp = 0.01871931552886963

seed_base=1213000:
  mean_delta_vs_mlp = 0.016927083333333332
  worst_delta_vs_mlp = 0.0
  near_pass_rate_vs_mlp = 1.0
  max_AUC_step_ratio_vs_mlp = 0.9646083424163914
  max_AUC_time_ratio_vs_mlp = 0.8558250559627462
  max_ECE_delta_vs_mlp = 0.0164031982421875
```

选择 `seed_base=1211000`，原因是 A0 worst_delta 为正、ECE delta 最小，task-side no-regression 最稳。

### 14.3 完整 recovered anchor-budget 结果

最终运行规模：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 1024
val_size = 512
test_size = 512
epochs = 3
batch_size = 128
rows = 72
summary_rows = 8
official_training_result_available = 0
no_fake = 1
no_proxy = 1
cpu_offload_used = 0
```

summary 关键值：

```text
A0-labelInit:
  mean_delta_vs_mlp = 0.022569444444444444
  worst_delta_vs_mlp = 0.013671875
  max_AUC_time_ratio_vs_mlp = 0.9309724872014364
  linec_nontearing_pass_rate = 0.0
  linec_nontearing_all_pass = 0

A1-noYForStats:
  mean_delta_vs_A0 = -0.010416666666666666
  worst_delta_vs_A0 = -0.048828125
  linec_nontearing_pass_rate = 0.0
  linec_nontearing_all_pass = 0

A2-randomP-labelFree:
  mean_delta_vs_A0 = -0.010416666666666666
  worst_delta_vs_A0 = -0.048828125
  linec_nontearing_pass_rate = 0.0
  linec_nontearing_all_pass = 0

A3-PCA-P-labelFree:
  mean_delta_vs_A0 = -0.041666666666666664
  worst_delta_vs_A0 = -0.083984375
  linec_nontearing_pass_rate = 0.0
  linec_nontearing_all_pass = 0

A4-lowfreqP-labelFree:
  mean_delta_vs_A0 = -0.6831597222222222
  worst_delta_vs_A0 = -0.806640625
  linec_nontearing_pass_rate = 0.0
  linec_nontearing_all_pass = 0

A5-orthogonalP-labelFree:
  mean_delta_vs_mlp = 0.018663194444444444
  worst_delta_vs_mlp = -0.005859375
  mean_delta_vs_A0 = -0.00390625
  worst_delta_vs_A0 = -0.0390625
  max_AUC_time_ratio_vs_mlp = 0.8042953039496114
  linec_nontearing_pass_rate = 0.0
  linec_nontearing_all_pass = 0

MLP-same-step-FLOP-AdamW:
  mean_delta_vs_mlp = 0.0
  worst_delta_vs_mlp = 0.0
  max_AUC_step_ratio_vs_mlp = 1.0
  max_AUC_time_ratio_vs_mlp = 1.0806638793400698
  linec_nontearing_pass_rate = 1.0
  linec_nontearing_all_pass = 1
```

解释：

```text
1. A5-orthogonalP-labelFree 是 best label-free candidate。
2. A5 task strong pass 成立：mean_delta_vs_A0 = -0.00390625，高于 -0.005 门槛；AUC_time_ratio_vs_mlp = 0.8042953039496114，低于 1.0。
3. A5 LineC_nontearing_all_pass = 0，因此 strong pass 整体不成立。
4. A0 task gate 复现，但 A0 LineC_nontearing_all_pass = 0；这说明本次 recovered protocol 的 LineC 测量不能复现 locked B320 的 LineC gate。
5. A6 MLP-same-step-FLOP-AdamW 是 control 行，不是 label-free B320 修复。它的 LineC pass=1 说明 LineC 判定代码本身能给 MLP control 正例，但不能拯救 A0/A5 的 protocol gate。
```

### 14.4 protocol audit 结论

`v1218_b320_anchor_budget_protocol_audit.csv` 最终关键字段：

```text
a0_anchor_budget_gate_pass = 1
a0_anchor_budget_task_reproduces_locked_anchor = 1
a0_anchor_budget_linec_available = 1
a0_anchor_budget_linec_reproduces_locked_anchor = 0
a0_anchor_budget_reproduces_locked_anchor = 0
official_budget_claim_allowed = 0
blocker = recovered-anchor-budget A0 does not reproduce locked B320 task+LineC gates; do not treat label-free ablation as official replacement
```

因此，当前不是“Line A 没推进”，而是：

```text
Line A task side 已经被 A5-orthogonalP 推进到接近关闭；
Line A LineC/protocol side 没过；
所以 R2 不能关闭，external_ready_base_claim 仍不能打开。
```

### 14.5 主 route 更新

最终 route：

```text
route = R2-B320LabelInitDependenceDetected
promotion_allowed = 0
p4_open = 0
```

fail reason：

```text
current_B320_uses_label_informed_trainprobe_init;label_free_anchor_budget_protocol_not_official;label_free_anchor_budget_A0_does_not_reproduce_locked_anchor;label_free_anchor_budget_LineC_nontearing_failed;strict_loss_agnostic_observables_not_visible
```

关键 route 字段：

```text
b320_label_free_anchor_budget_ablation_available_count = 5
b320_label_free_anchor_budget_task_strong_pass = 1
b320_label_free_anchor_budget_linec_nontearing_pass = 0
b320_label_free_anchor_budget_a0_task_reproduces_locked_anchor = 1
b320_label_free_anchor_budget_a0_linec_reproduces_locked_anchor = 0
b320_label_free_anchor_budget_a0_reproduces_locked_anchor = 0
b320_label_free_anchor_budget_best_candidate = A5-orthogonalP-labelFree
b320_label_free_anchor_budget_best_mean_delta_vs_A0 = -0.00390625
```

### 14.6 zip / artifact 更新

最终 zip：

```text
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_entries = 65
zip_sha256 = 861a71f82db20fdb635612333cd1bee046ac3ed9e3dfa18ce29dbf96eb6a0dea
hash_manifest_entries = 40
```

新增进入 zip 的关键审计产物包括：

```text
review_artifacts/v1218_b320_label_free_anchor_budget_ablation.csv
review_artifacts/v1218_b320_label_free_anchor_budget_summary.csv
review_artifacts/v1218_b320_label_free_anchor_budget_route.json
review_artifacts/v1218_b320_anchor_budget_seedbase1211000_probe_ablation.csv
review_artifacts/v1218_b320_anchor_budget_seedbase1211000_probe_summary.csv
review_artifacts/v1218_b320_anchor_budget_seedbase1211000_probe_route.json
review_artifacts/v1218_b320_anchor_budget_seedbase1212000_probe_ablation.csv
review_artifacts/v1218_b320_anchor_budget_seedbase1212000_probe_summary.csv
review_artifacts/v1218_b320_anchor_budget_seedbase1212000_probe_route.json
review_artifacts/v1218_b320_anchor_budget_seedbase1213000_probe_ablation.csv
review_artifacts/v1218_b320_anchor_budget_seedbase1213000_probe_summary.csv
review_artifacts/v1218_b320_anchor_budget_seedbase1213000_probe_route.json
review_artifacts/v1218_b320_anchor_budget_protocol_audit.csv
```

### 14.7 更新后的各线进展

```text
Line R 代码审查/依赖闭包：96%
  code packet / hash / required artifacts 已闭环；剩余是 manual review pending，不是自动实验能关闭的项。

Line A B320 anchor + label-init audit：84%
  current labelInit anchor 已审计；smoke 与 recovered anchor-budget A1-A5/A6 均已跑；还按 v12.16 b64/sketchdim24 方向重跑了 LineC protocol；A5 task-side 接近关闭，但 LineC/protocol gate 失败，R2 未关。

Line C audit-only calibration：90%
  v12.18 audit-only calibration 本身仍成立；但 recovered anchor-budget 的 post-training LineC 不能复现 locked A0 nontearing gate，因此不能作为 official replacement 证据。

Line T strict loss-agnostic visibility：75% audit complete / 0% gate pass
  feature provenance、strict visibility、leaveout、support expansion 都已执行；仍然 strict_visibility_pass=0。

Line I actuator response controls：0%
  Line T 未过，按计划不得打开。

Line B/P4：0%
  R2 与 Line T 均未过，P4 不打开。

Line D classic monitor：100% monitor-only
  本轮没有新 hypothesis；监控状态保持。
```

### 14.8 最终复盘结论

```text
本轮 v12.18 已继续推进到 recovered anchor-budget 级别，并补齐 A6 control；还尝试了 b64/sketchdim24 的 LineC protocol 修正。
没有编造 official result；所有关键数据来自实际 runner 输出。
Line A 的任务性能方向有正进展：A5-orthogonalP-labelFree mean_delta_vs_A0=-0.00390625，满足 task strong pass。
但 LineC_nontearing_all_pass=0，且 A0 recovered protocol 在 b64/sketchdim24 下仍不能复现 locked LineC gate。
因此主 route 保持 R2，promotion_allowed=0，p4_open=0。
```

## 15. 继续推进后的复盘更新：Line T output-subspace drift repair

继续执行 v12.18 计划 8.7 的失败后修复方向。此前 Line T 的 strict feature matrix 已经包含：

```text
random_cotangent_logit_jacobian
augmentation_consistency_proxy
```

但还没有纳入 v12.16 repair artifact 中已落盘的 output-subspace drift / predicted drift 列。本轮补入这些 strict label-free feature，并增加 family ablation 审计。

### 15.1 本轮修改

修改文件：

```text
experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

新增 strict feature family：

```text
output_subspace_drift:
  noise_target_norm
  reservoir_target_norm
  pred_noise_delta
  pred_reservoir_delta
  pred_coupling_delta
```

新增 artifact：

```text
v1218_strict_visibility_family_ablation.csv
```

审计约束：

```text
uses_label_for_feature = 0
uses_ce_for_feature = 0
selection_for_promotion = 0
reason = post-hoc Line T repair diagnostic; not used to lower gates or claim promotion
```

### 15.2 实验结果

主 route 仍然是：

```text
route = R2-B320LabelInitDependenceDetected
promotion_allowed = 0
p4_open = 0
```

Line T 新结果：

```text
strict_visibility_feature_rows = 15552
strict_visibility_feature_columns = 18
strict_auc_joint_min = 0.3503521126760563
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate;support_concentrated;required_leaveout_not_runnable
```

family ablation 关键结果：

```text
all_current:
  feature_columns = 18
  AUC_joint_min = 0.3503521126760563
  AUC_joint_max = 0.5769230769230769
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

baseline_without_output_subspace_drift:
  feature_columns = 13
  AUC_joint_min = 0.5
  AUC_joint_max = 0.9436619718309859
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

only_output_subspace_drift:
  feature_columns = 5
  AUC_joint_min = 0.47836538461538464
  AUC_joint_max = 0.8230633802816901
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

all_except_projector_geometry:
  feature_columns = 15
  AUC_joint_min = 0.5
  AUC_joint_max = 0.9471830985915493
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0
```

解释：

```text
1. output-subspace drift 没有解决 Line T 可见性问题。
2. 全量加入 output-subspace drift 后，AUC_joint_min 从原 baseline 0.5 降到 0.3503521126760563。
3. 单独 output-subspace drift 的 AUC_joint_min=0.47836538461538464，也低于 0.65 exploratory gate。
4. 所有 family ablation 的 precision_at_k_joint_min 与 recall_at_k_joint_min 都是 0.0。
5. support_concentrated=1，required_leaveout_not_runnable=1 仍存在。
```

因此，本轮不能打开 Line I，也不能进入 P4；不能通过选择 family 或降低 threshold 来声称通过。

### 15.3 最终 zip 更新

```text
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_entries = 65
zip_sha256 = 861a71f82db20fdb635612333cd1bee046ac3ed9e3dfa18ce29dbf96eb6a0dea
hash_manifest_entries = 40
```

新增进入 zip：

```text
review_artifacts/v1218_strict_visibility_family_ablation.csv
```

### 15.4 更新后的各线进展

```text
Line R 代码审查/依赖闭包：97%
  新增 family ablation artifact 已纳入 zip/hash；剩余仍是 human manual review pending。

Line A B320 anchor + label-init audit：84%
  A5 task-side 过门槛，但 LineC/protocol gate 不过，R2 未关。

Line C audit-only calibration：90%
  calibration 本身成立；recovered anchor-budget 的 A0 LineC reproduction 不成立。

Line T strict loss-agnostic visibility：82% audit complete / 0% gate pass
  已执行 output-subspace drift repair 和 family ablation；结果确认 strict visibility 仍不可见。

Line I actuator response controls：0%
  Line T 未过，按计划不得打开。

Line B/P4：0%
  R2 与 Line T 均未过，P4 不打开。

Line D classic monitor：100% monitor-only
  本轮没有新 hypothesis；监控状态保持。
```

### 15.5 本轮最终结论

```text
v12.18 计划中的推荐修复方向已经继续执行到：
1. Line A label-free recovered anchor-budget + A5 orthogonal repair + A6 control；
2. LineC b64/sketchdim24 protocol retry；
3. Line T output-subspace drift repair；
4. Line T family ablation diagnostic。

没有出现可以关闭 R2 / 打开 P4 的证据。
当前阻塞不是未执行，而是实验证据失败：
  - A5 task pass 但 LineC_nontearing_all_pass=0；
  - A0 task reproduction pass 但 LineC reproduction=0；
  - strict visibility AUC/precision/recall/support/leaveout 全部不满足 gate。
```

## 16. 继续推进后的复盘更新：Line A label-free A6-A14 reservoir/direct 修复

### 16.1 为什么继续做 A6-A14

上一轮结束时，v12.18 仍处于：

```text
route = R2-B320LabelInitDependenceDetected
promotion_allowed = 0
p4_open = 0
```

具体阻塞不是“没有 label-free candidate”，而是：

```text
1. A5-orthogonalP-labelFree 是当前 best label-free candidate；
2. A5 task-side 已接近通过：mean_delta_vs_A0=-0.00390625，满足 task strong pass 阈值；
3. 但 A5 LineC_nontearing_all_pass=0；
4. 失败集中在 RealSignalReservoirRatio_above_mlp_plus_0.02，且多数 row 还有 NoiseSignalLeak_above_mlp_plus_0.02；
5. recovered anchor-budget 的 A0 task reproduction pass，但 LineC reproduction=0，因此 protocol 仍不允许把该预算当 official B320 replacement。
```

因此继续尝试两类修复：

```text
A6-A10:
  从降低 / bound quadratic-reservoir 分支入手。

A11-A14:
  从增强 label-free direct branch 入手，试图减少 reservoir overuse。
```

这些候选都没有使用 label 或 CE direction：

```text
uses_y_for_stats = 0
strict_label_free_init = 1
uses_label_for_direction = 0
uses_ce_vector_for_direction = 0
cpu_offload_used = 0
no_fake = 1
no_proxy = 1
```

### 16.2 本轮做了什么修改

修改文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

新增 helper：

```text
replace_variant_token()
```

新增候选：

```text
A6-orthogonalP-active64-labelFree
  A5 + activep64，用于限制 quad_proj 可训练活跃列。

A7-orthogonalP-lowQuad-labelFree
  A5 + quadreadinit075 + identitytailquad020，用于降低 reservoir/quadratic 分支强度。

A8-orthogonalP-boundQ-labelFree
  A5 + boundq，用于限制 quadratic feature 爆发。

A9-orthogonalP-lowQuad-boundQ-labelFree
  A7 + boundq。

A10-orthogonalP-strongLowQuad-boundQ-labelFree
  A9 + quadreadinit050 + quadramp010，激进降低 quadratic warm start。

A11-orthogonalP-directRead125-labelFree
  A5 + directreadinit125，增强 label-free direct readout。

A12-orthogonalP-identityAmp150-labelFree
  A5 + identityamp150，增强 identity direct readout。

A13-orthogonalP-lowQuad-directRead125-labelFree
  A7 + directreadinit125。

A14-orthogonalP-lowQuad-identityAmp150-labelFree
  A7 + identityamp150。
```

`run_v1218_b320_codeaudit_lossagnostic_functional.py` 同步加入 A6-A14 的 audit rows，便于 zip 内审计。

### 16.3 实验执行范围

最终有效覆盖 artifact：

```text
v1218_b320_label_free_anchor_budget_ablation.csv
v1218_b320_label_free_anchor_budget_summary.csv
v1218_b320_label_free_anchor_budget_route.json
v1218_b320_label_init_audit.csv
v1218_route_decision.json
v1218_code_review_packet.zip
```

最终 anchor-budget run：

```text
run_id = label_free_anchor_budget_seed012_3x3_seedbase1211000_linec_b64_s32_d24_a6_a14_reservoir_direct_repair
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
seed_base = 1211000
train_size = 1024
val_size = 512
test_size = 512
epochs = 3
batch_size = 128
measure_linec = 1
linec_batch_size = 64
linec_sketch_batch_size = 32
linec_sketch_dim = 24
rows = 153
summary_rows = 17
label_free_ablation_available_count = 14
```

### 16.4 A0-A14 任务结果

关键 summary：

```text
candidate_id                                            mean_delta_vs_mlp       worst_delta_vs_mlp   mean_delta_vs_A0       worst_delta_vs_A0   max_AUC_time_ratio_vs_mlp   LineC
A0-labelInit                                            0.022569444444444444    0.013671875                              0.9340476611616885        0/9
A1-noYForStats                                          0.012152777777777778   -0.015625          -0.010416666666666666  -0.048828125       0.6915868435257259        0/9
A2-randomP-labelFree                                    0.012152777777777778   -0.015625          -0.010416666666666666  -0.048828125       0.6915872276095383        0/9
A3-PCA-P-labelFree                                     -0.019097222222222224   -0.0703125         -0.041666666666666664  -0.083984375       0.8714860359825919        0/9
A4-lowfreqP-labelFree                                  -0.6603732638888888     -0.76953125        -0.6829427083333334    -0.806640625       42333.4644903965          0/9
A5-orthogonalP-labelFree                                0.018663194444444444   -0.005859375       -0.00390625            -0.0390625         0.6578371023388839        0/9
A6-orthogonalP-active64-labelFree                       0.018663194444444444   -0.005859375       -0.00390625            -0.0390625         0.657837163187841         0/9
A7-orthogonalP-lowQuad-labelFree                        0.007161458333333333   -0.01953125        -0.015407986111111112  -0.033203125       0.7214003793617216        0/9
A8-orthogonalP-boundQ-labelFree                        -0.05056423611111111    -0.09765625        -0.07313368055555555   -0.111328125       1.2658369264152824        0/9
A9-orthogonalP-lowQuad-boundQ-labelFree                -0.0724826388888889     -0.126953125       -0.09505208333333333   -0.140625          1.762533492131623         0/9
A10-orthogonalP-strongLowQuad-boundQ-labelFree         -0.0783420138888889     -0.134765625       -0.10091145833333333   -0.1484375         1.9856940546873132        0/9
A11-orthogonalP-directRead125-labelFree                 0.018229166666666668   -0.005859375       -0.004340277777777778  -0.0390625         0.6578555395729052        0/9
A12-orthogonalP-identityAmp150-labelFree                0.018446180555555556   -0.005859375       -0.004123263888888889  -0.0390625         0.6579779676747233        0/9
A13-orthogonalP-lowQuad-directRead125-labelFree         0.007378472222222222   -0.01953125        -0.015190972222222222  -0.033203125       0.7215783321369537        0/9
A14-orthogonalP-lowQuad-identityAmp150-labelFree        0.007161458333333333   -0.017578125       -0.015407986111111112  -0.03515625        0.7217404337588441        0/9
MLP-same-step-FLOP-AdamW                                0.0                     0.0                                      1.072374152799825         9/9
```

任务侧结论：

```text
1. A5 仍是 best label-free candidate。
2. A6 与 A5 几乎等价，activeP64 没有产生任务或 LineC 改善。
3. A11/A12 direct branch repair 任务接近 A5，但略低于 A5。
4. A7/A13/A14 lowQuad 方向任务下降到 mean_delta_vs_A0 约 -0.015。
5. A8/A9/A10 boundQ/strongLowQuad 方向任务明显失败。
```

### 16.5 Line C 结果复盘

Line C 关键均值：

```text
A5  pass 0/9, meanLeak=0.1112774374584357, meanReservoir=0.7062219613128238, meanCoupling=0.1823585640753439
A6  pass 0/9, meanLeak=0.11127393609947628, meanReservoir=0.7062293257978227, meanCoupling=0.18236011351399817
A7  pass 0/9, meanLeak=0.12404166244798237, meanReservoir=0.8426430688963996, meanCoupling=0.1687150209352995
A8  pass 0/9, meanLeak=0.13467423783408272, meanReservoir=0.8196694321102567, meanCoupling=0.15475378669706397
A9  pass 0/9, meanLeak=0.14854084410601193, meanReservoir=0.7652382585737441, meanCoupling=0.15501506107042073
A10 pass 0/9, meanLeak=0.14145335513684484, meanReservoir=0.7604071895281473, meanCoupling=0.1646045696908192
A11 pass 0/9, meanLeak=0.11095200065109465, meanReservoir=0.7053342296017541, meanCoupling=0.18260943509131536
A12 pass 0/9, meanLeak=0.11036044400599268, meanReservoir=0.7057378987471262, meanCoupling=0.18251400828949643
A13 pass 0/9, meanLeak=0.12404547590348455, meanReservoir=0.8424602614508735, meanCoupling=0.16899817181984755
A14 pass 0/9, meanLeak=0.12377550121810701, meanReservoir=0.8423047595553927, meanCoupling=0.16919102604683658
```

Line C 结论：

```text
1. 所有 A6-A14 都是 0/9 non-tearing。
2. 所有 A5-A14 都是 RealSignalReservoirRatio_above_mlp_plus_0.02 = 9/9。
3. A11/A12 direct repair 仅将 meanReservoir 从 A5 的 0.7062219613128238 降到约 0.7053/0.7057，幅度太小，不能改变 gate。
4. lowQuad 方向不仅没有降低 reservoir ratio，反而升到约 0.842。
5. boundQ 方向没有降低 noise leak，且损伤 task。
```

因此，本轮不能把 A6-A14 任何一个 candidate 作为 Line A promotion candidate。

### 16.6 最终 route 和 gate 状态

主审计最终输出：

```text
route = R2-B320LabelInitDependenceDetected
promotion_allowed = 0
p4_open = 0
```

关键 route 字段：

```text
fail_reason = current_B320_uses_label_informed_trainprobe_init;label_free_anchor_budget_protocol_not_official;label_free_anchor_budget_A0_does_not_reproduce_locked_anchor;label_free_anchor_budget_LineC_nontearing_failed;strict_loss_agnostic_observables_not_visible

b320_label_free_anchor_budget_ablation_available_count = 14
b320_label_free_anchor_budget_best_candidate = A5-orthogonalP-labelFree
b320_label_free_anchor_budget_best_mean_delta_vs_A0 = -0.00390625
b320_label_free_anchor_budget_best_max_AUC_time_ratio_vs_mlp = 0.6578371023388839
b320_label_free_anchor_budget_task_strong_pass = 1
b320_label_free_anchor_budget_linec_nontearing_pass = 0

b320_label_free_anchor_budget_a0_gate_pass = 1
b320_label_free_anchor_budget_a0_task_reproduces_locked_anchor = 1
b320_label_free_anchor_budget_a0_linec_reproduces_locked_anchor = 0
b320_label_free_anchor_budget_a0_reproduces_locked_anchor = 0
```

Protocol audit：

```text
anchor_budget_A0_mean_delta_vs_mlp = 0.022569444444444444
anchor_budget_A0_worst_delta_vs_mlp = 0.013671875
anchor_budget_A0_max_AUC_time_ratio_vs_mlp = 0.9340476611616885
anchor_budget_A0_linec_nontearing_rows = 9
anchor_budget_A0_linec_nontearing_pass_rate = 0.0
anchor_budget_A0_linec_nontearing_all_pass = 0
official_budget_claim_allowed = 0
blocker = recovered-anchor-budget A0 does not reproduce locked B320 task+LineC gates; do not treat label-free ablation as official replacement
```

Line T 状态未被本轮改变：

```text
strict_auc_joint_min = 0.3503521126760563
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate;support_concentrated;required_leaveout_not_runnable
line_t_family_ablation_best_set = baseline_without_output_subspace_drift
line_t_family_ablation_best_auc_joint_min = 0.5
```

### 16.7 A6-A14 direct repair 后的中间 zip / 审计包

A6-A14 direct repair 后的中间 zip：

```text
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_entries = 65
zip_sha256 = f118d37c084b9de3ff7687859e0cb605dac32fe92cc30a7f74181dd1f28d7bf9
hash_manifest_entries = 40
```

确认包含：

```text
code/experiments/run_v1218_b320_label_free_ablation.py
review_artifacts/v1218_route_decision.json
review_artifacts/v1218_b320_label_init_audit.csv
review_artifacts/v1218_b320_label_free_anchor_budget_ablation.csv
review_artifacts/v1218_b320_label_free_anchor_budget_summary.csv
review_artifacts/v1218_b320_label_free_anchor_budget_route.json
review_artifacts/v1218_strict_visibility_family_ablation.csv
```

### 16.8 这一轮后的判断

本轮不是没有推进，而是推进后得到负结果：

```text
1. 已执行 A6-A10 quadratic/reservoir repair。
2. 已执行 A11-A14 direct-branch repair。
3. 已重跑完整 3x3 recovered anchor-budget。
4. 已重跑主审计与 zip 打包。
5. 没有任何 A6-A14 通过 LineC non-tearing。
6. A5 仍是 best label-free candidate，但仍只能 task-side strong pass，不能关 R2。
```

本节结束时的最强结论：

```text
label-free B320 的 task 侧已经有近似可用候选 A5；
但 current SimpleFastTaskGeometry 的 label-free orthogonal/direct/lowQuad/boundQ token 级修复无法消除 reservoir overuse；
继续做同类 token 小修很可能只是重复失败。
```

下一步如果继续 v12.18，不建议再盲目扩 A15/A16 token 网格。更合理方向是：

```text
1. 重新定义 label-free projector source，使它显式对齐 input manifold / augmentation-stable directions，而不是随机 orthogonal P。
2. 在不读 label / CE 的前提下设计 direct-vs-quadratic branch balancing objective 或 initialization audit。
3. 将 LineC RealSignalReservoirRatio 的分子拆开，确认是 q feature norm、quad_readout、branch_scale 还是 AdamW window 造成 reservoir overuse。
4. 只有确认机制后再生成下一组候选，否则继续枚举 token 不是高质量推进。
```

当前没有证据允许：

```text
promotion_allowed = 1
p4_open = 1
external_ready_base_claim = 1
official_budget_claim_allowed = 1
```

## 17. 继续推进后的复盘更新：LineC detailed reservoir decomposition

### 17.1 为什么要做这一步

A6-A14 全部 0/9 non-tearing 后，仍然需要回答一个问题：

```text
RealSignalReservoirRatio 高，究竟是因为 grad-sketch signal 子空间太窄，
还是 CE-real 残差本身在 p_res 方向的相对能量太高？
```

直接看 `RealSignalReservoirRatio` 的 ratio 不能回答这个问题，所以本轮把 LineC audit 指标拆成：

```text
signal_mass_topk
reservoir_fraction
top_eigen_share
signal_top_count
real_residual_energy
real_total_energy
noise_signal_energy
noise_total_energy
```

这些指标仍然是 audit-only：

```text
linec_label_used_for_audit_only = 1
linec_label_used_for_direction = 0
linec_ce_vector_used_for_direction = 0
```

### 17.2 本轮代码修改

修改文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
```

新增：

```text
signal_reservoir_metrics_detailed()
```

它复现 `v1252._signal_reservoir_metrics` 的计算，并额外保存：

```text
linec_signal_mass_topk
linec_reservoir_fraction
linec_top_eigen_share
linec_dissipation_condition
linec_SNR_positive_fraction
linec_signal_top_count
linec_real_residual_energy
linec_real_total_energy
linec_noise_signal_energy
linec_noise_total_energy
```

### 17.3 detailed 版重跑结果

最终 run：

```text
run_id = label_free_anchor_budget_seed012_3x3_seedbase1211000_linec_b64_s32_d24_a6_a14_reservoir_direct_repair_detailed_linec
rows = 153
summary_rows = 17
label_free_ablation_available_count = 14
best_label_free_candidate = A5-orthogonalP-labelFree
best_label_free_mean_delta_vs_A0 = -0.00390625
```

关键 task summary：

```text
A0-labelInit:
  mean_delta_vs_mlp = 0.022569444444444444
  worst_delta_vs_mlp = 0.013671875
  max_AUC_time_ratio_vs_mlp = 0.9151941736560291
  LineC = 0/9

A5-orthogonalP-labelFree:
  mean_delta_vs_mlp = 0.018663194444444444
  worst_delta_vs_mlp = -0.005859375
  mean_delta_vs_A0 = -0.00390625
  max_AUC_time_ratio_vs_mlp = 0.7776700385171795
  LineC = 0/9

A11-orthogonalP-directRead125-labelFree:
  mean_delta_vs_mlp = 0.018229166666666668
  mean_delta_vs_A0 = -0.004340277777777778
  max_AUC_time_ratio_vs_mlp = 0.7775565950993564
  LineC = 0/9

A12-orthogonalP-identityAmp150-labelFree:
  mean_delta_vs_mlp = 0.018446180555555556
  mean_delta_vs_A0 = -0.004123263888888889
  max_AUC_time_ratio_vs_mlp = 0.7780129834775749
  LineC = 0/9

MLP-same-step-FLOP-AdamW:
  mean_delta_vs_mlp = 0.0
  max_AUC_time_ratio_vs_mlp = 1.0331432309011304
  LineC = 9/9
```

### 17.4 机制拆解结论

均值：

```text
MLP-same-step-FLOP-AdamW:
  signal_mass_topk = 0.8611922595236037
  reservoir_fraction = 0.13880774047639635
  top_eigen_share = 0.6349554161230723
  signal_top_count = 2.3333333333333335
  real_residual_energy = 23.124738832314808
  real_total_energy = 95.83334681722853
  noise_signal_energy = 55.435595217678284
  noise_total_energy = 1015.8765869140625

A5-orthogonalP-labelFree:
  signal_mass_topk = 0.8355417980088128
  reservoir_fraction = 0.16445820199118721
  top_eigen_share = 0.3521610846122106
  signal_top_count = 4.444444444444445
  real_residual_energy = 5.526151098724869
  real_total_energy = 9.12627146144708
  noise_signal_energy = 120.40638648139105
  noise_total_energy = 1104.3729315863716

A11-orthogonalP-directRead125-labelFree:
  reservoir_fraction = 0.16428567303551567
  real_residual_energy = 5.513146714203888
  real_total_energy = 9.123155733777416
  noise_signal_energy = 120.28983052571614
  noise_total_energy = 1105.7046237521702

A12-orthogonalP-identityAmp150-labelFree:
  reservoir_fraction = 0.16427639457914564
  real_residual_energy = 5.511935541199313
  real_total_energy = 9.124200887564156
  noise_signal_energy = 119.85119077894423
  noise_total_energy = 1106.1815728081597
```

解读：

```text
1. A5 的 reservoir_fraction 只比 MLP 高约 0.02565。
2. A5 的 real_total_energy 比 MLP 小很多：9.12627146144708 vs 95.83334681722853。
3. 但 A5 的 real_residual_energy 没有按同等比例下降：5.526151098724869 vs 23.124738832314808。
4. 因此 A5 的 RealSignalReservoirRatio 失败，不主要是 signal eigenspace 太窄，而是剩余 CE-real 残差相对集中在 p_res reservoir 方向。
5. A11/A12 direct branch repair 几乎不改变 reservoir_fraction 和 residual energy，因此它们不能修复 LineC。
6. A5 的 noise_signal_energy 高于 MLP：120.40638648139105 vs 55.435595217678284，而 noise_total_energy 没有同步大幅增高；所以 NoiseSignalLeak 也是真实问题，不是单纯分母效应。
```

这一步把失败原因从“reservoir ratio 高”推进到更具体的机制判断：

```text
当前 label-free A5 的问题不是简单降低 quad branch 或增强 direct branch 就能解决；
它需要改变 grad-sketch signal subspace 与 CE-real residual 的相对对齐方式，
同时不能让 noise residual 更强地落入 p_sig。
```

### 17.5 最终 gate 状态

最终主审计：

```text
route = R2-B320LabelInitDependenceDetected
promotion_allowed = 0
p4_open = 0
```

关键 route：

```text
b320_label_free_anchor_budget_ablation_available_count = 14
b320_label_free_anchor_budget_best_candidate = A5-orthogonalP-labelFree
b320_label_free_anchor_budget_best_mean_delta_vs_A0 = -0.00390625
b320_label_free_anchor_budget_best_max_AUC_time_ratio_vs_mlp = 0.7776700385171795
b320_label_free_anchor_budget_task_strong_pass = 1
b320_label_free_anchor_budget_linec_nontearing_pass = 0

b320_label_free_anchor_budget_a0_gate_pass = 1
b320_label_free_anchor_budget_a0_task_reproduces_locked_anchor = 1
b320_label_free_anchor_budget_a0_linec_reproduces_locked_anchor = 0
b320_label_free_anchor_budget_a0_reproduces_locked_anchor = 0
```

最终 zip：

```text
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_entries = 65
zip_sha256 = 82d438d4641b2d1ee820fa715c2cadf76243b8a2905f8be3c485dc1d0c0b7872
hash_manifest_entries = 40
```

### 17.6 这一轮后的边界

现在我已经不建议继续在同一 runner 里盲目追加 token 候选。原因是：

```text
1. A6-A14 已覆盖 activeP、lowQuad、boundQ、directRead125、identityAmp150 及组合。
2. 所有候选 LineC 都是 0/9。
3. detailed audit 显示问题不是一个简单 branch-scale token 能修掉。
4. 继续枚举类似 token 会快速变成低质量搜索。
```

下一步应当换成机制性设计，而不是继续小修：

```text
1. 构造真正 label-free 的 projector source，使 p_sig 更贴近 CE-real residual 的可解释低维方向，但不使用 CE/label 作为 direction。
2. 做 augmentation-stable / input-manifold projector，而不是随机 orthogonal projector。
3. 单独审计 noise residual 为什么更强进入 p_sig，先降低 noise_signal_energy。
4. 再生成下一代 candidate，而不是继续扩 A15/A16 token 网格。
```

## 18. 继续推进后的复盘更新：Line T strict feature engineering repair

### 18.1 为什么还要继续推进 Line T

上一轮 Line T 的 `output-subspace drift repair` 没有通过：

```text
strict_auc_joint_min = 0.3503521126760563
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate;support_concentrated;required_leaveout_not_runnable
```

计划里的推荐方向是继续尝试 strict loss-agnostic observables，而不是打开 CE-gradient sketch 或降低 gate。因此本轮做的是 Line T 特征工程修复：在不使用 label/CE 的前提下，从现有 v12.16 strict source columns 派生更多可审计特征，并做 family ablation。

### 18.2 本轮做了什么修改

修改文件：

```text
experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

新增 / 修复内容：

```text
1. strict feature 数量从 18 个扩展到 36 个。
2. 新增 18 个 derived features，覆盖：
   basis_occupancy_derived
   logit_free_spectrum_proxy_derived
   projector_geometry_derived
   random_cotangent_logit_jacobian_ensemble
   augmentation_consistency_derived
   output_subspace_drift_derived
   sketch_geometry_derived

3. 新增 strict_feature_value(source, spec)，让 raw / derived feature 都通过同一入口计算。

4. 新增 v1218_strict_visibility_feature_engineering_audit.csv：
   每个 feature 记录 source_column / formula / input_columns / derived / uses_label / uses_ce / selection_for_promotion。

5. 修复审计可读性：
   feature-engineering audit 增加 derived 字段，避免后续审计者只能通过 formula 是否为空来反推 raw/derived。
```

本轮没有做的事：

```text
1. 没有使用 label / CE / permuted label / dataset name 作为 feature。
2. 没有改变 P4 / promotion / strict visibility 的 gate。
3. 没有把 family ablation 的 post-hoc 结果用于降低阈值或声明通过。
```

### 18.3 实验结果

主 route：

```text
route = R2-B320LabelInitDependenceDetected
promotion_allowed = 0
p4_open = 0
fail_reason = current_B320_uses_label_informed_trainprobe_init;label_free_anchor_budget_protocol_not_official;label_free_anchor_budget_A0_does_not_reproduce_locked_anchor;label_free_anchor_budget_LineC_nontearing_failed;strict_loss_agnostic_observables_not_visible
```

Line T strict visibility：

```text
rows = 864
feature_columns = 36
long_feature_rows = 31104
AUC_noise_min = 0.4630281690140845
AUC_reservoir_min = 0.4299107142857143
AUC_joint_min = 0.028169014084507043
AUC_joint_max = 0.7884615384615384
precision_at_k_joint_min = 0.0
recall_at_k_joint_min = 0.0
required_split_failures = 1
support_concentrated = 1
visibility_exploratory_pass = 0
visibility_hard_pass = 0
visibility_robustness_pass = 0
visibility_pass = 0
blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate;support_concentrated;required_leaveout_not_runnable
```

family ablation 最好结果：

```text
line_t_family_ablation_rows = 30
line_t_family_ablation_best_set = only_augmentation_consistency_derived
line_t_family_ablation_best_auc_joint_min = 0.5
line_t_family_ablation_best_visibility_pass = 0
```

排名靠前的 family ablation：

```text
only_augmentation_consistency_derived:
  feature_columns = 2
  AUC_joint_min = 0.5
  AUC_joint_max = 0.5192307692307693
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

only_projector_geometry_derived:
  feature_columns = 3
  AUC_joint_min = 0.5
  AUC_joint_max = 0.9615384615384616
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

only_output_subspace_drift:
  feature_columns = 5
  AUC_joint_min = 0.47836538461538464
  AUC_joint_max = 0.8230633802816901
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

only_random_cotangent_logit_jacobian_ensemble:
  feature_columns = 4
  AUC_joint_min = 0.2692307692307692
  AUC_joint_max = 0.8380281690140845
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0
```

feature-engineering audit：

```text
feature_audit_rows = 36
raw_count = 18
derived_count = 18
uses_label_sum = 0
uses_ce_sum = 0
selection_sum = 0
```

### 18.4 结果解读

这轮结果是负结果，而且是比较明确的负结果：

```text
1. feature_columns 从 18 增加到 36 后，全量 AUC_joint_min 没有提升，反而从 0.3503521126760563 降到 0.028169014084507043。
2. 单独 family 中最好的 AUC_joint_min 只有 0.5，属于不能支持 release 可见性的水平。
3. 所有 family 的 precision_at_k_joint_min 和 recall_at_k_joint_min 仍为 0.0。
4. support_concentrated=1 仍未解决。
5. required_leaveout_not_runnable=1 仍未解决。
```

因此，Line T 的 blocker 不是“缺少一个简单代数派生特征”。当前更像是 source support 本身不足：只有同一批 strict v12.16 source rows 时，即使增加 random cotangent / augmentation / projector / output drift 的 derived features，也无法形成可推广的 hard joint release visibility。

### 18.5 与 Line A / Line C 的关系

本轮没有改变 Line A / Line C 的既有结论：

```text
Line A:
  best label-free candidate 仍是 A5-orthogonalP-labelFree。
  A5 task-side strong pass = 1。
  但 A5 LineC non-tearing = 0/9。

Line C:
  A6-A14 reservoir/direct repair 全部 LineC non-tearing = 0/9。
  detailed decomposition 显示问题不是简单 direct branch 或 quad branch scale 能解决。

Line T:
  strict features 从 18 扩到 36 后仍失败。
  full-matrix AUC_joint_min = 0.028169014084507043。
  best family AUC_joint_min = 0.5，precision/recall 仍为 0。
```

所以当前 v12.18 的主要 blocker 是三组同时存在：

```text
1. current_B320_uses_label_informed_trainprobe_init
2. label-free anchor budget 未能同时复现 locked anchor 与 LineC non-tearing
3. strict loss-agnostic target visibility 不可见，且 support/leaveout 仍不足
```

### 18.6 zip / 审计包更新

最新 zip：

```text
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_entries = 66
zip_sha256 = c670793aeb05646f698bf7f5380078211ebca78675a129c866971ce5fd3b5843
hash_manifest_entries = 41
```

zip 已包含：

```text
code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
review_artifacts/v1218_strict_visibility_feature_engineering_audit.csv
review_artifacts/v1218_strict_visibility_features.csv
review_artifacts/v1218_strict_visibility_scores.csv
review_artifacts/v1218_strict_visibility_leaveout.csv
review_artifacts/v1218_strict_visibility_family_ablation.csv
review_artifacts/v1218_route_decision.json
```

### 18.7 更新后的各线进展

```text
Line R / code audit:
  进展 = 100%
  状态 = 审计包已更新，新增 feature-engineering audit，zip 已刷新。

Line A / B320 label-free anchor:
  进展 = 70%
  状态 = 找到 A5 task-side strong candidate，但 LineC non-tearing 未通过；不能视为 official promotion。

Line C / functional geometry:
  进展 = 65%
  状态 = 已做 A0-A14 detailed decomposition，明确 direct/quad scale repair 不足；仍未修复 non-tearing。

Line T / strict loss-agnostic visibility:
  进展 = 60%
  状态 = raw+derived strict features 已扩到 36 个并做 family ablation；结果仍失败，且 support/leaveout blocker 未消除。

Line I / P4:
  进展 = 0%
  状态 = 仍关闭；P2/P3/P4 前置条件未满足。

Line D / classic no-BSpline monitor:
  进展 = 100%
  状态 = monitor 已记录，不是当前 blocker。
```

### 18.8 本轮最终判断

本轮已按计划继续尝试 Line T 的推荐修复方向；结果没有通过，不能打开 P4，也不能声明 promotion。

更重要的是，当前失败原因已经从“也许缺少若干可见性特征”收敛为：

```text
当前 strict source support 不足，同一批 v12.16 source 上继续追加 post-hoc derived features，不能解决 support_concentrated 和 required_leaveout_not_runnable。
```

如果继续推进，下一步不应再是在同一 CSV 上继续加派生列，而应生成新的独立 strict source / support expansion，例如更多独立 run_id、更多可 leave-source-run-out 的 source、更多 seed/window/actuator dictionary 覆盖，并保持 label/CE 不进入 direction。这个动作已经超出本轮 v12.18 代码审计 runner 的局部修复范围，需要作为新的机制性实验设计执行。

## 19. 继续推进后的复盘更新：Line T support expansion / matched-window / composite repair

### 19.1 为什么继续推进

第 18 节的结论是：在同一批 strict source 上继续追加 derived features，不能解决 `support_concentrated=1` 与 `required_leaveout_not_runnable=1`。用户要求不能在未完成时退出，因此本轮继续按第 18 节提出的推荐方向推进：

```text
1. 生成新的独立 strict source / support expansion。
2. 增加可 leave-source-run-out 的 calibrated source。
3. 扩大 seed/window 覆盖，同时不降低 release threshold。
4. 保持 label/CE 不进入 feature construction 或 direction selection。
5. 如果 support 问题修复后仍失败，再判断 blocker 是否已转移。
```

这次推进不是把负结果改写成正结果，而是把上一轮“support 不足导致 leaveout 无法判断”的问题真实跑完，让后续结论不再停在“数据结构不够审计”的层面。

### 19.2 做了哪些代码修改

本轮修改集中在：

```text
experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

修改内容：

```text
1. 允许 v12.18 从 v12.17 route/source_run_dirs 读取多个 calibrated source，而不是只绑定一个 repair source。
2. build_strict_v1216_long_rows 改为跳过没有 v12.17 release label 覆盖的 rows，避免把缺失 label 错当 negative。
3. 新增 role/time/persistence/projector derived strict features：
   method_role_flags
   window_schedule
   time_persistence_derived
   projector_geometry_derived
4. 新增 Line T support expansion audit：
   v1218_line_t_support_expansion_audit.csv
5. 新增 strict composite repair：
   v1218_strict_visibility_composite_repair.csv
6. 更新 zip 审计包，让新增代码与新增产物进入：
   v1218_code_review_packet.zip
```

审计约束：

```text
uses_label_for_feature = 0
uses_ce_for_feature = 0
selection_for_promotion = 0
release threshold 未降低
```

### 19.3 新 source 结果

最终进入 matched v12.17/v12.18 链路的是两个 b128 / sketchdim24 / rank5 / windows=10,12,15 source：

```text
repair_seed012_sketchdim24_rank5_b128_w10_12_15:
  route = R2-EstimatorStillWeak
  p0_pass = 1
  p1_pass = 0
  p1_best_noise_abs_spearman = 0.18818886064881887
  p1_best_reservoir_abs_spearman = 0.2605555287728709
  p1_best_negative_release_precision = 0.0
  p1_best_negative_release_recall = 0.0
  p3_max_actuator_sketch_delta_fro = 0.008491670712828636
  p3_max_actuator_projector_angle_deg = 2.2786431312561035

repair_seed345_sketchdim24_rank5_b128_w10_12_15:
  route = R2-EstimatorStillWeak
  p0_pass = 1
  p1_pass = 0
  p1_best_noise_abs_spearman = 0.0619349104560141
  p1_best_reservoir_abs_spearman = 0.09456052759880126
  p1_best_negative_release_precision = 0.0
  p1_best_negative_release_recall = 0.0
  p3_max_actuator_sketch_delta_fro = 0.012493130750954151
  p3_max_actuator_projector_angle_deg = 0.6425489187240601
```

这里要注意：v12.16 自身仍是 `R2-EstimatorStillWeak`，不是 promotion。它们的作用是给 v12.17/v12.18 提供更多 calibrated support，用于审计 strict visibility 是否有可泛化信号。

matched v12.17：

```text
run_id = support_expand_seed012_seed345_b128_w10_12_15
route = R2-ReleaseUnobservableUnderLossAgnosticFeatures
p1_calibrated_source_runs = [
  repair_seed012_sketchdim24_rank5_b128_w10_12_15,
  repair_seed345_sketchdim24_rank5_b128_w10_12_15
]
p2_visibility_repair_best_protocol = repair_coupling_spectrum_persistence/mean_noise_reservoir_rank
p2_visibility_repair_best_auc_joint_min = 0.7844852941176471
p2_visibility_repair_best_precision_joint_min = 0.0
p2_visibility_repair_best_recall_joint_min = 0.0
p4_open = 0
```

v12.17 release support：

```text
release_label_rows = 1728
hard_joint_rows = 28

hard_joint_by_source:
  repair_seed012_sketchdim24_rank5_b128_w10_12_15 = 12
  repair_seed345_sketchdim24_rank5_b128_w10_12_15 = 16

hard_joint_by_window:
  10 = 8
  12 = 12
  15 = 8

hard_joint_by_method:
  U5-OrthogonalCotangentVJPEnsemble = 8
  U6-RoleConditionedPrimitiveActuator = 8
  U7-ProjectionQuadRoleActuator = 8
  U9-MixedLowRankActuatorBasis = 4
```

这个结果修复了第 18 节最关键的结构性问题：support 不再只集中在单一来源/单一窗口，且可以做 leave-source-run-out。

### 19.4 最终 v12.18 主审计结果

最终 route：

```text
run_id = official_from_v1217_v1216_artifacts_support_expand_seed012_seed345_b128_w10_12_15_role_time_composite
route = R2-B320LabelInitDependenceDetected
fail_reason = current_B320_uses_label_informed_trainprobe_init;label_free_anchor_budget_protocol_not_official;label_free_anchor_budget_A0_does_not_reproduce_locked_anchor;label_free_anchor_budget_LineC_nontearing_failed;strict_loss_agnostic_observables_not_visible
p4_open = 0
promotion_allowed = 0
```

Line T strict visibility：

```text
strict_visibility_feature_rows = 81216
strict_visibility_wide_rows = 1728
strict_visibility_feature_columns = 47
hard_joint_support_rows = 28
support_concentrated = 0
strict_visibility_required_split_failures = 0

strict_auc_joint_min = 0.05660377358490566
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate
```

这说明两件事同时成立：

```text
1. support_concentrated / required_leaveout_not_runnable 这两个结构 blocker 已经修复。
2. 修复 support 后，strict loss-agnostic visibility 仍然失败。
```

最差 split：

```text
base strict worst split:
  leave-method-out / U9-MixedLowRankActuatorBasis
  AUC_joint = 0.05660377358490566
  precision_at_k_joint = 0.0
  recall_at_k_joint = 0.0
  positive_rows_joint = 4
  topk_hits_joint = 0
```

leave-source-run-out：

```text
repair_seed012 held out:
  AUC_joint = 0.4849374021909233
  precision_at_k_joint = 0.0
  recall_at_k_joint = 0.0
  positive_rows_joint = 12

repair_seed345 held out:
  AUC_joint = 0.5772405660377359
  precision_at_k_joint = 0.0
  recall_at_k_joint = 0.0
  positive_rows_joint = 16
```

因此，失败已经不再是“没法跑 required leaveout”。现在的失败更具体：strict features 在跨 method 泛化，尤其 U9 / MixedLowRankActuatorBasis 上，无法抓到 hard joint release。

### 19.5 strict composite repair 结果

为了确认 joint target 过稀疏是否导致直接 joint scorer 失败，本轮增加了 composite repair：分别训练/排序 noise 与 reservoir，再用 rank combination 形成 joint score。

结果：

```text
mean_noise_reservoir_rank:
  AUC_joint_min = 0.12794811320754718
  AUC_joint_max = 0.8705985915492958
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  repair_pass = 0

min_noise_reservoir_rank:
  AUC_joint_min = 0.19811320754716982
  AUC_joint_max = 0.8911409198113207
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  repair_pass = 0

product_noise_reservoir_rank:
  AUC_joint_min = 0.1792452830188679
  AUC_joint_max = 0.8683978873239436
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  repair_pass = 0
```

这个 repair 没有通过。它说明问题不是简单地把 joint release 拆成 noise/reservoir 两个较密集目标后再组合就能解决。至少在当前 strict feature family 下，组合排序仍不能稳定命中 hard joint release 的 top-k。

### 19.6 与第 18 节结论的关系

第 18 节的判断是：

```text
当前 strict source support 不足，同一批 v12.16 source 上继续追加 post-hoc derived features，不能解决 support_concentrated 和 required_leaveout_not_runnable。
```

本轮继续推进后，这句话需要更新为：

```text
support 不足已经被 matched source expansion 修复；
但即使在 support_concentrated=0、required_leaveout_not_runnable=0 的条件下，
strict loss-agnostic observables 对 hard joint release 仍不可见。
```

换句话说，blocker 从“审计数据结构不够”推进成了“可审计结构存在，但当前 loss-agnostic observable family 仍不够表达目标可见性”。

### 19.7 zip / 审计包

最新 zip：

```text
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_sha256 = eb970193cf11173e8a8adf450e7cdca112666351f7af79118909cf65ab4d1d30
zip_entries = 67
hash_manifest_entries = 42
```

zip 已确认包含：

```text
code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
review_artifacts/v1218_route_decision.json
review_artifacts/v1218_strict_visibility_composite_repair.csv
review_artifacts/v1218_line_t_support_expansion_audit.csv
review_artifacts/v1218_hash_manifest.json
```

### 19.8 更新后的各线进展

```text
Line R / code audit:
  进展 = 100%
  状态 = 审计包已更新；新增 multi-source strict visibility、support expansion audit、composite repair，并已进入 zip。

Line A / B320 label-free anchor:
  进展 = 70%
  状态 = A5 仍是 best label-free candidate；但 anchor-budget protocol gate 与 LineC non-tearing 仍未满足，不能 promotion。

Line C / functional geometry:
  进展 = 65%
  状态 = v12.16/v12.17 matched support 已用于 Line T 审计；但 LineC 本身仍是 R2，不能视为机制成功。

Line T / strict loss-agnostic visibility:
  进展 = 75%
  状态 = support_concentrated 与 required_leaveout_not_runnable 已修复；strict visibility 仍失败，blocker 转移到 U9/method-generalization 下 hard joint release 不可见。

Line I / P4:
  进展 = 0%
  状态 = 仍关闭；p4_open=0，promotion_allowed=0。

Line D / classic no-BSpline monitor:
  进展 = 100%
  状态 = monitor 已记录，不是当前 blocker。
```

### 19.9 本轮最终判断

本轮 v12.18 已按文档推荐方向继续推进到当前可审计边界：support expansion、matched source、role/time/persistence/projector features、composite repair 都已经跑完，并写入执行日志与审计包。

最终结论是负结论：

```text
不能打开 P4。
不能声明 promotion。
不能降低 release threshold。
不能把 v12.17 的 repair AUC 当作 v12.18 strict loss-agnostic visibility 通过。
```

当前最具体 blocker：

```text
在 matched source support 已足够跑 leave-source/window/method out 的条件下，
strict loss-agnostic observables 仍无法泛化识别 hard joint release；
最差点是 leave-method-out / U9-MixedLowRankActuatorBasis。
```

后续如果继续推进，应围绕“method-generalizable loss-agnostic observable family”重新设计，而不是继续在当前 47 个 strict features 上做小幅派生列堆叠。

## 20. 继续推进后的复盘更新：seed/window expansion 与 context-rank method-generalization repair

### 20.1 继续推进动机

第 19 节已经修复了 `support_concentrated` 和 `required_leaveout_not_runnable`，但 Line T 仍失败：

```text
strict_auc_joint_min = 0.05660377358490566
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
worst_split = leave-method-out / U9-MixedLowRankActuatorBasis
```

这说明 blocker 不再是“没有足够 leaveout 结构”，而是当前 strict loss-agnostic feature family 不能泛化识别 hard joint release。按计划 8.7，继续推进了两类修复：

```text
1. 继续增加 seeds / windows / support。
2. 针对 method-generalization 设计 context-rank feature repair。
```

### 20.2 seed678 support expansion

新增 source：

```text
repair_seed678_sketchdim24_rank5_b128_w10_12_15
```

v12.16 结果：

```text
route = R2-EstimatorStillWeak
p0_pass = 1
p1_pass = 0
p1_best_noise_abs_spearman = 0.117271743127479
p1_best_reservoir_abs_spearman = 0.16378848952471092
p1_best_negative_release_precision = 0.0
p1_best_negative_release_recall = 0.0
p3_max_actuator_sketch_delta_fro = 0.008893998339772224
p3_max_actuator_projector_angle_deg = 0.5695825815200806
```

v12.17 calibration：

```text
source_calibration_pass = 0
random_joint_false_positive_rows = 0
control_joint_false_positive_rows = 0
joint_hard_support_rows = 4
```

结论：

```text
seed678 不能加入 calibrated release label set。
不能把未校准 source 强行纳入 promotion visibility。
```

### 20.3 mixed-window support expansion

为了测试 `增加 windows` 是否能改善 top-k visibility，复用已真实存在的：

```text
repair_seed012_sketchdim24_rank5_b128_w10_12_15
repair_seed345_sketchdim24_rank5_b128_w8_10_12
repair_seed345_sketchdim24_rank5_b128_w10_12_15
```

v12.17 mixed-window 结果：

```text
route = R2-ReleaseUnobservableUnderLossAgnosticFeatures
p1_calibrated_source_runs = [
  repair_seed012_sketchdim24_rank5_b128_w10_12_15,
  repair_seed345_sketchdim24_rank5_b128_w10_12_15,
  repair_seed345_sketchdim24_rank5_b128_w8_10_12
]
p2_visibility_repair_best_auc_joint_min = 0.772028777603895
p2_visibility_repair_best_precision_joint_min = 0.0
p2_visibility_repair_best_recall_joint_min = 0.0
p4_open = 0

release_label_rows = 2592
hard_joint_rows = 36
hard_joint_by_method:
  U5-OrthogonalCotangentVJPEnsemble = 12
  U6-RoleConditionedPrimitiveActuator = 8
  U7-ProjectionQuadRoleActuator = 12
  U9-MixedLowRankActuatorBasis = 4
```

mixed-window v12.18 结果：

```text
strict_visibility_wide_rows = 2592
strict_visibility_feature_columns = 47
hard_joint_support_rows = 36
support_concentrated = 0
strict_visibility_required_split_failures = 0
strict_auc_joint_min = 0.0375
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
worst_split = leave-method-out / U9-MixedLowRankActuatorBasis
```

结论：

```text
mixed-window support 增加了总 hard_joint_rows，但没有增加 U9 support；
U9 仍只有 4 个 hard_joint rows；
strict AUC 还从 0.05660377358490566 降到 0.0375。
```

因此，mixed-window 不是修复方向，不能作为最终主结果。

### 20.4 context-rank repair 的设计与审计约束

本轮修改：

```text
experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

新增 artifact：

```text
v1218_strict_visibility_context_rank_repair.csv
```

repair 思路：

```text
对每个 precommit candidate cohort 做 rank normalization，
避免 scorer 直接依赖 method identity 或 raw scale。
```

测试的 context：

```text
source_run,dataset,seed,window
dataset,seed,window
source_run,dataset,seed
```

测试的 scorer：

```text
ridge
positive_centroid_dot
positive_centroid_distance
contrast_centroid_dot
contrast_centroid_distance
```

审计约束：

```text
uses_label_for_feature = 0
uses_ce_for_feature = 0
uses_method_identity_feature = 0
selection_for_promotion = 0
```

说明：

```text
label 只用于训练 audit scorer，不进入 feature construction。
context-rank 是 post-hoc method-generalization diagnostic；
不能降低 gate，不能直接用于 promotion。
```

### 20.5 context-rank repair 结果

最终主审计仍使用较干净的 seed012+seed345 matched source：

```text
run_id = official_from_v1217_v1216_artifacts_support_expand_seed012_seed345_b128_w10_12_15_context_rank_repair
route = R2-B320LabelInitDependenceDetected
p4_open = 0
promotion_allowed = 0
```

主 strict visibility：

```text
strict_visibility_feature_rows = 81216
strict_visibility_wide_rows = 1728
strict_visibility_feature_columns = 47
hard_joint_support_rows = 28
support_concentrated = 0
strict_visibility_required_split_failures = 0
strict_auc_joint_min = 0.05660377358490566
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate
```

context-rank 最好结果：

```text
line_t_context_rank_repair_best_protocol = context_rank_source_dataset_seed_window/ridge
line_t_context_rank_repair_best_auc_joint_min = 0.4798951048951049
line_t_context_rank_repair_best_precision_joint_min = 0.0
line_t_context_rank_repair_best_recall_joint_min = 0.0
line_t_context_rank_repair_best_pass = 0
```

top summaries：

```text
context_rank_source_dataset_seed_window/ridge:
  AUC_joint_min = 0.4798951048951049
  AUC_joint_max = 0.9224759615384616
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  worst_split = leave-dataset-out / KMNIST
  repair_pass = 0

context_rank_dataset_seed_window/ridge:
  AUC_joint_min = 0.4798951048951049
  AUC_joint_max = 0.9224759615384616
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  worst_split = leave-dataset-out / KMNIST
  repair_pass = 0

context_rank_source_dataset_seed_window/positive_centroid_dot:
  AUC_joint_min = 0.35563380281690143
  AUC_joint_max = 0.9632867132867133
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  worst_split = leave-seed-out / 4
  repair_pass = 0
```

结论：

```text
context-rank 可以局部改善部分 split 的排序上界，
但最小 AUC 仍低于 0.65；
precision/recall min 仍为 0；
不能通过 Line T。
```

### 20.6 最新 zip

最新 zip：

```text
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_sha256 = b4ccb50b88d657ee1b0b9d036e75da590288da17979be6e646318c01ff8ae10b
zip_entries = 68
hash_manifest_entries = 43
```

zip 已包含：

```text
code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
review_artifacts/v1218_route_decision.json
review_artifacts/v1218_strict_visibility_context_rank_repair.csv
review_artifacts/v1218_hash_manifest.json
```

### 20.7 更新后的各线进展

```text
Line R / code audit:
  进展 = 100%
  状态 = zip 已刷新，新增 context-rank repair artifact。

Line A / B320 label-free anchor:
  进展 = 70%
  状态 = 未改变；label-free anchor budget 仍不能替代 current labelInit anchor。

Line C / audit calibration:
  进展 = 70%
  状态 = seed678 新 source 真实生成但未通过 calibration；mixed-window source 可校准但不改善 strict visibility。

Line T / strict loss-agnostic visibility:
  进展 = 85%
  状态 = 已尝试 random cotangent / augmentation / output-subspace drift / support expansion / mixed windows / context-rank repair；仍未通过。

Line I / P3:
  进展 = 0%
  状态 = gate 未打开；Line T 未通过，不允许生成 matched-control actuator dictionary。

Line B / P4:
  进展 = 0%
  状态 = gate 未打开；p4_open=0。

Line D:
  进展 = 100%
  状态 = monitor 已记录，不是当前 blocker。
```

### 20.8 本轮最终判断

本轮继续推进后，v12.18 的结论更明确：

```text
1. 不是没执行 seed/window/support repair；seed678 和 mixed-window 都已真实运行。
2. 不是没尝试 method-generalization feature repair；context-rank 已真实落盘。
3. 不是 zip 没更新；最新 zip 已包含 context-rank artifact 和新代码。
4. 但所有新增尝试仍未通过 Line T hard/robust visibility gate。
```

当前最可信的科学结论：

```text
在当前 v12.16/v12.17 artifact family 下，
hard-release target 对 strict loss-agnostic observables 仍不可见；
尤其 U9/MixedLowRankActuatorBasis 的 hard_joint support 过少且不可泛化。
```

因此：

```text
不能打开 P3。
不能打开 P4。
不能声明 promotion。
不能降低 release threshold。
不能把 post-hoc context-rank repair 当作成功。
```

再继续推进不应是同类小修，而应是新机制级设计：重新定义可部署的 label-free target/value source，或重新生成带更多 U9-like hard release 支持且能通过 calibration 的 actuator/LineC source family。

## 21. 继续推进后的复盘更新：U10-U14 actuator dictionary expansion

### 21.1 为什么继续

用户要求：

```text
没有完成就继续；
按推荐思路修改；
如果还是不行，思考如何解决然后尝试修复；
结果必须写入复盘，不允许编造数据。
```

v12.18 计划中仍有一个没有充分闭合的修复面：

```text
Line T precision/recall 低时：
  检查 support concentration；
  增加 windows / seeds / actuator dictionary；
  不降低 release threshold。
```

前面已经推进过 seed/window/context-rank，本节补齐 actuator dictionary expansion。

### 21.2 代码修改

修改文件：

```text
/home/chengshun.wang/DG-LCA/experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py
```

修改函数：

```text
make_p1_probe_updates(...)
```

新增 U10-U14：

```text
U10-AllRoleLowLiftCotangent
U11-BranchQuadLowLiftPrimitiveActuator
U12-DirectLowLiftPrimitiveActuator
U13-MixedLowRankNoBranchBasis
U14-MixedBranchQuadDirectLowLiftBasis
```

设计约束：

```text
全部为 loss-agnostic direction；
不引入 CE gradient sketch；
不引入 label feature；
不引入 dataset branch；
不修改 hard release threshold；
不修改 P3/P4 gate。
```

修复过程中出现的 blocker：

```text
初版使用 projection/readout role；
v12.14 ROLE_TOKENS 不支持 projection；
运行报错 KeyError: 'projection'。
```

修复方式：

```text
U11 改为 branch_quad；
U12 改为 direct；
U14 改为 U7 + U11 + U12 的 branch/quad/direct mixed basis；
py_compile 通过。
```

这个修改是合理的，因为它只使用已有 v12.14/v12.15 支持的 role tokens，没有新增不可审计 role，也没有绕过 calibration。

### 21.3 seed012 dictionary expansion 结果

seed012 / B128 / U10-U14：

```text
run_dir = results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_dict_expand_u10_u14_sketchdim24_rank5_b128_w10_12_15
rows = 1404
hard_joint_rows = 20
hard_methods:
  U7-ProjectionQuadRoleActuator = 4
  U6-RoleConditionedPrimitiveActuator = 4
  U9-MixedLowRankActuatorBasis = 4
  U10-AllRoleLowLiftCotangent = 4
  U13-MixedLowRankNoBranchBasis = 4
random_joint_fp_rows = 8
noop_joint_fp_rows = 0
route = R2-EstimatorStillWeak
```

解释：

```text
U10/U13 确实增加了 hard support；
但 B128 下 random control 也出现 joint hard false positive；
v12.17 calibration 因此排除该 source。
```

v12.17 calibration：

```text
source = repair_seed012_dict_expand_u10_u14_sketchdim24_rank5_b128_w10_12_15
source_calibration_pass = 0
random_joint_false_positive_rows = 8
control_joint_false_positive_rate = 0.009259259259259259
```

### 21.4 batch bracket：B128/B192/B256

按 calibration CSV 给出的方向：

```text
expand batch/bootstrap or raise hard gate margin; do not lower release threshold
```

本轮只扩大 batch，不 raise gate margin。

seed012 batch bracket：

```text
B128:
  hard_joint_rows = 20
  random_joint_fp_rows = 8
  calibration = fail
  新增 U10/U13 hard support = yes

B192:
  hard_joint_rows = 8
  random_joint_fp_rows = 0
  calibration = pass
  新增 U10-U14 hard support = no

B256:
  hard_joint_rows = 0
  random_joint_fp_rows = 0
  calibration = pass
  新增 U10-U14 hard support = no
```

结论：

```text
扩大 batch 可以压掉 false positive；
但也同时消灭了新增 dictionary support；
不存在一个已观察到的 seed012 batch 点同时满足：
  1. 新增 U10/U13 support 有效；
  2. random/noop control false positive 为 0；
  3. strict visibility 通过。
```

### 21.5 seed345 dictionary expansion 结果

seed345 / B128：

```text
run_dir = results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_dict_expand_u10_u14_sketchdim24_rank5_b128_w10_12_15
rows = 1404
hard_joint_rows = 44
hard_methods:
  U13-MixedLowRankNoBranchBasis = 20
  U5-OrthogonalCotangentVJPEnsemble = 8
  U6-RoleConditionedPrimitiveActuator = 4
  U11-BranchQuadLowLiftPrimitiveActuator = 4
  U14-MixedBranchQuadDirectLowLiftBasis = 4
  U7-ProjectionQuadRoleActuator = 4
random_joint_fp_rows = 28
noop_joint_fp_rows = 0
route = R2-EstimatorStillWeak
```

seed345 / B192：

```text
hard_joint_rows = 44
random_joint_fp_rows = 28
source_calibration_pass = 0
```

seed345 / B256：

```text
hard_joint_rows = 40
random_joint_fp_rows = 24
source_calibration_pass = 0
```

解释：

```text
seed345 的 U13 support 很强；
但 random matched control 同时大量触发；
即使扩大 batch 到 256，random false positive 仍然存在；
所以不能把这条 source 当成可靠 hard release source。
```

### 21.6 v12.17 组合审计

组合 1：seed012 B128 dict + seed345 B128 dict

```text
run_id = support_expand_seed012_seed345_dict_u10_u14_b128_w10_12_15
route = R1-LineCCalibrationUnstable
p1_calibrated_source_runs = []
p1_joint_hard_support_rows = 64
p2_visibility_repair_best_auc_joint_min = 0.695884771319242
p2_visibility_repair_best_precision_joint_min = 0.0
p2_visibility_repair_best_recall_joint_min = 0.0
p4_open = 0
```

组合 2：seed012 B192 dict + seed345 B128 original

```text
run_id = support_expand_seed012_dict_u10_u14_b192_plus_seed345_b128_w10_12_15
p1_calibrated_source_runs:
  repair_seed012_dict_expand_u10_u14_sketchdim24_rank5_b192_w10_12_15
  repair_seed345_sketchdim24_rank5_b128_w10_12_15
p1_joint_hard_support_rows = 24
p2_visibility_repair_best_auc_joint_min = 0.8059269162210339
p2_visibility_repair_best_precision_joint_min = 0.0
p2_visibility_repair_best_recall_joint_min = 0.0
p4_open = 0
```

组合 3：seed012 B192 dict + seed345 B192/B256 dict

```text
seed345 dict source_calibration_pass = 0
有效 calibrated source 只剩 seed012 B192 dict
p2_visibility_repair_best_auc_joint_min = 0.4813753581661891
precision/recall = 0.0 / 0.0
p4_open = 0
```

结论：

```text
dictionary expansion 可以提高 raw hard support；
但高 support 分支无法过 control calibration；
可 calibration 分支无法让 precision/recall 非零。
```

### 21.7 v12.18 strict checks

对照表：

```text
official_current:
  run_id = official_from_v1217_v1216_artifacts_support_expand_seed012_seed345_b128_w10_12_15_context_rank_repair_dict_u10_u14_code_audit_refresh
  hard_joint_support_rows = 28
  strict_auc_joint_min = 0.05660377358490566
  strict_precision_joint_min = 0.0
  strict_recall_joint_min = 0.0
  context_rank_best_auc_joint_min = 0.4798951048951049
  strict_visibility_pass = 0

b64_seed345_check:
  run_id = support_expand_repair_plus_seed345_b128_w10_12_15_context_rank_check
  hard_joint_support_rows = 24
  strict_auc_joint_min = 0.07535460992907801
  strict_precision_joint_min = 0.0
  strict_recall_joint_min = 0.0
  context_rank_best_auc_joint_min = 0.46830985915492956
  strict_visibility_pass = 0

dict_b192_check:
  run_id = support_expand_seed012_dict_u10_u14_b192_plus_seed345_b128_w10_12_15_strict_check
  hard_joint_support_rows = 24
  strict_auc_joint_min = 0.031914893617021274
  strict_precision_joint_min = 0.0
  strict_recall_joint_min = 0.0
  context_rank_best_auc_joint_min = 0.48060344827586204
  strict_visibility_pass = 0

dict_b256_check:
  run_id = support_expand_seed012_dict_u10_u14_b256_plus_seed345_b128_w10_12_15_strict_check
  hard_joint_support_rows = 16
  strict_auc_joint_min = 0.024565508021390375
  strict_precision_joint_min = 0.0
  strict_recall_joint_min = 0.0
  context_rank_best_auc_joint_min = 0.48702830188679247
  strict_visibility_pass = 0
  blocker includes support_concentrated
```

最重要事实：

```text
所有 v12.18 strict checks 的 precision_joint_min = 0.0；
所有 v12.18 strict checks 的 recall_joint_min = 0.0；
没有任何 Line T hard/robust visibility pass。
```

### 21.8 最新 zip 与可审计性

刷新主审计目录：

```text
result_dir = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_sha256 = 6582b59646528335e6f9628f95edb57a6c85a4fc24fd3fed0b663d4d71dc3e36
zip_entries = 68
hash_manifest_entries = 43
```

zip 内容已核验：

```text
code/experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py
code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
review_artifacts/v1218_route_decision.json
review_artifacts/v1218_hash_manifest.json
```

这意味着：

```text
本轮新增 U10-U14 代码已进入最新审计 zip；
不是只改了工作区而没有打包。
```

### 21.9 更新后的各线进展

```text
Line R / code audit:
  进展 = 100%
  状态 = 最新 zip 已刷新，包含 U10-U14 dictionary expansion 代码。

Line A / B320 label-free anchor:
  进展 = 70%
  状态 = 未改变；current B320 仍是 label-informed anchor，label-free anchor budget 不能替代 official anchor。

Line C / calibration:
  进展 = 80%
  状态 = 新增 dictionary expansion sources 已生成并校准；
         seed012 B128 / seed345 B128/B192/B256 因 random-control FP 失败；
         seed012 B192/B256 可 calibration，但 support 不足或消失。

Line T / strict loss-agnostic visibility:
  进展 = 90%
  状态 = 已完成 random cotangent / augmentation / output-subspace / support expansion /
         mixed windows / context-rank / actuator dictionary expansion / batch bracket；
         仍未通过。
  gate pass = 0%

Line I / P3:
  进展 = 0%
  状态 = Line T 未通过，按计划不允许开 P3。

Line B / P4:
  进展 = 0%
  状态 = p4_open = 0，不能运行 P4 short-run。

Line D:
  进展 = 100%
  状态 = monitor 已记录，不是当前 blocker。
```

### 21.10 本轮最终判断

本轮继续推进后，结论比上一轮更硬：

```text
1. actuator dictionary expansion 已执行，不是遗漏。
2. 新增 U10-U14 能产生 raw hard support，尤其 seed345 的 U13。
3. 但 high-support 分支会同步触发 random matched control hard release，calibration 不允许通过。
4. batch 扩大可以压低 false positive，但会消灭新增 support 或导致 strict visibility 更差。
5. v12.18 strict visibility 的 precision/recall min 始终为 0。
```

因此：

```text
不能打开 P3。
不能打开 P4。
不能 promotion。
不能降低 release threshold。
不能把 raw support 多的 dictionary expansion 当作成功。
```

当前最可信科学结论：

```text
在当前 B320/v12.16-v12.18 artifact family 下，
hard release 不是简单增加 actuator dictionary 就能变成可部署 loss-agnostic observable；
新增 dictionary 产生的 hard movement 与 random matched control 纠缠，
而 calibration-safe 的 movement 又不足以让 strict feature 做出非零 precision/recall。
```

下一步如果继续，不应再做同类 post-hoc ranking repair；
需要换机制级 target/value source，或重新设计能同时满足：

```text
1. control false positive = 0；
2. hard support 不集中；
3. leave-method/source/dataset/seed/window 全部可见；
4. strict precision/recall 非零并过 gate；
5. direction construction 仍完全 label-free / CE-free。
```

## 22. 继续推进后的复盘更新：raise hard-gate margin repair

### 22.1 为什么继续

上一轮已经补齐了 `increase actuator dictionary`，但 v12.17 calibration CSV 对失败 source 的建议还有另一半：

```text
expand batch/bootstrap or raise hard gate margin; do not lower release threshold
```

本轮继续执行 `raise hard gate margin`。这不是降低阈值；它把 audit hard release 从 `<= -0.01` 改成更严格的候选 margin，例如 `<= -0.0125/-0.015/...`，只作为诊断 target sweep，不改变 official gate。

### 22.2 代码修改

修改文件：

```text
/home/chengshun.wang/DG-LCA/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

新增：

```text
write_hard_gate_margin_repair(...)
--margin-source-run
v1218_hard_gate_margin_repair.csv
v1218_hard_gate_margin_repair_summary.csv
```

审计约束：

```text
threshold 只向更严格方向移动；
不降低 release threshold；
不使用 label/CE 作为 feature；
不允许 margin sweep 自己打开 P3/P4；
结果只用于判定 dictionary expansion 的 false-positive blocker 是否可被 stricter target 修复。
```

### 22.3 扫描范围

扫描 source：

```text
repair_seed012_sketchdim24_rank5_b128_w10_12_15
repair_seed345_sketchdim24_rank5_b128_w10_12_15
repair_seed012_dict_expand_u10_u14_sketchdim24_rank5_b128_w10_12_15
repair_seed012_dict_expand_u10_u14_sketchdim24_rank5_b192_w10_12_15
repair_seed012_dict_expand_u10_u14_sketchdim24_rank5_b256_w10_12_15
repair_seed345_dict_expand_u10_u14_sketchdim24_rank5_b128_w10_12_15
repair_seed345_dict_expand_u10_u14_sketchdim24_rank5_b192_w10_12_15
repair_seed345_dict_expand_u10_u14_sketchdim24_rank5_b256_w10_12_15
```

扫描 threshold：

```text
-0.0100
-0.0125
-0.0150
-0.0175
-0.0200
-0.0250
-0.0300
```

总行数：

```text
margin_source_count = 8
margin_rows = 56
```

### 22.4 结果摘要

summary：

```text
usable_margin_rows = 4
best_margin_source_run = repair_seed345_sketchdim24_rank5_b128_w10_12_15
best_margin_threshold = -0.01
best_margin_joint_hard_support_rows = 16
best_margin_new_u10_u14_hard_support_rows = 0
best_margin_random_joint_false_positive_rows = 0
best_margin_support_concentrated = 0
margin_repair_pass = 0
```

关键观察：

```text
可 usable 的 margin rows 都来自原始非 U10-U14 source。
没有任何 U10-U14 support row 在 random_fp=0 时仍保持 usable。
```

U10-U14 high-support 分支：

```text
seed012 dict B128 / threshold -0.01:
  joint = 20
  new_u10_u14 = 8
  random_fp = 8
  usable = 0

seed345 dict B128 / threshold -0.01:
  joint = 44
  new_u10_u14 = 28
  random_fp = 28
  usable = 0

seed345 dict B128 / threshold -0.02:
  joint = 24
  new_u10_u14 = 16
  random_fp = 16
  usable = 0

seed345 dict B128 / threshold -0.03:
  joint = 8
  new_u10_u14 = 8
  random_fp = 8
  usable = 0
```

seed345 dict B192/B256：

```text
更严格 threshold 会减少 support；
但 random_fp 仍与 U13/U14 support 同步存在；
没有观察到 usable U10-U14 margin row。
```

### 22.5 这说明什么

这轮修复明确排除了一个可能解释：

```text
不是 hard threshold 太松导致 U10-U14 被 random control 污染；
即使 threshold 更严格，U10/U13/U14 的 hard support 与 random matched control false positive 仍然绑定。
```

因此，dictionary expansion 的问题不是简单 threshold margin 可以修复，而更像是：

```text
这些 low-lift / mixed low-rank movement 触发的是一个 control-shared response mode；
它能让 audit metric 同时改善，但不是 deployable loss-agnostic value source；
更严格 gate 只是在 support 与 false positive 之间一起裁剪，没有分离二者。
```

### 22.6 最新主审计状态

最新主 route：

```text
run_id = official_from_v1217_v1216_artifacts_context_rank_dict_u10_u14_margin_repair_refresh
route = R2-B320LabelInitDependenceDetected
p4_open = 0
promotion_allowed = 0
strict_visibility_pass = 0
strict_auc_joint_min = 0.05660377358490566
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
line_t_context_rank_repair_best_auc_joint_min = 0.4798951048951049
line_t_hard_gate_margin_repair_rows = 56
line_t_hard_gate_margin_usable_rows = 4
line_t_hard_gate_margin_best_source = repair_seed345_sketchdim24_rank5_b128_w10_12_15
line_t_hard_gate_margin_best_threshold = -0.01
line_t_hard_gate_margin_best_joint_support = 16
line_t_hard_gate_margin_best_new_u10_u14_support = 0
line_t_hard_gate_margin_repair_pass = 0
```

最新 zip：

```text
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_sha256 = 43262f6c065ce34f5f574c8ce72a254bb699ba1fab6a57950f27323742521a1c
zip_entries = 70
hash_manifest_entries = 45
```

zip 已包含：

```text
code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
review_artifacts/v1218_hard_gate_margin_repair.csv
review_artifacts/v1218_hard_gate_margin_repair_summary.csv
```

### 22.7 更新后的各线进展

```text
Line R / code audit:
  进展 = 100%
  状态 = zip 已刷新到 43262...，包含 hard-gate margin artifact。

Line A / B320 label-free anchor:
  进展 = 70%
  状态 = 未改变；label-informed anchor blocker 仍在。

Line C / calibration:
  进展 = 85%
  状态 = 已测试 dictionary expansion + batch bracket + stricter hard margin；
         U10-U14 high-support 分支仍无法摆脱 random control false positive。

Line T / strict loss-agnostic visibility:
  进展 = 92%
  状态 = 已尝试 feature repair、support expansion、dictionary expansion、batch bracket、hard-gate margin；
         strict precision/recall 仍为 0。
  gate pass = 0%

Line I / P3:
  进展 = 0%
  状态 = Line T 未过，不能开。

Line B / P4:
  进展 = 0%
  状态 = p4_open = 0。

Line D:
  进展 = 100%
  状态 = monitor，不是当前 blocker。
```

### 22.8 本轮最终判断

本轮继续后，`raise hard gate margin` 也被真实执行并失败：

```text
1. stricter margin 没有救回 U10-U14。
2. 可 usable margin rows 没有新增 U10-U14 support。
3. U10-U14 support 存在时 random false positive 也存在。
4. strict visibility 主结果没有改善。
```

因此：

```text
不能打开 P3。
不能打开 P4。
不能 promotion。
不能降低 threshold。
不能把 hard-gate margin sweep 当成成功。
```

如果继续，下一步已经不是 v12.18 当前 artifact family 内的修修补补；需要新机制级实验，重新构造能与 random matched control 分离的 label-free value source。

## 23. 继续推进后的复盘更新：hard-gate bootstrap stability audit

### 23.1 为什么继续

上一节完成了 `raise hard gate margin`，但 calibration 建议里还包含 `bootstrap`。本节补做 bootstrap stability audit，避免把“batch/margin 已做”误当作已经覆盖 bootstrap。

### 23.2 代码修改

修改文件：

```text
/home/chengshun.wang/DG-LCA/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

新增 artifact：

```text
v1218_hard_gate_bootstrap_repair.csv
v1218_hard_gate_bootstrap_repair_summary.csv
```

新增函数：

```text
write_hard_gate_bootstrap_repair(...)
```

设计：

```text
bootstrap unit = dataset_seed_window
replicates = 128
thresholds = -0.0100, -0.0125, -0.0150, -0.0200, -0.0300
sources = 同 margin sweep 的 8 个 source
不降低 threshold
不把 bootstrap 结果用于 promotion
```

### 23.3 结果

summary：

```text
bootstrap_rows = 40
bootstrap_source_count = 8
bootstrap_replicates_per_row = 128
u10_u14_rows_with_positive_usable_rate = 1
best_bootstrap_source_run = repair_seed345_dict_expand_u10_u14_sketchdim24_rank5_b192_w10_12_15
best_bootstrap_threshold = -0.0125
best_bootstrap_usable_rate = 0.015625
best_bootstrap_new_u10_u14_support_mean = 23.8125
bootstrap_repair_pass = 0
```

关键明细：

```text
repair_seed012 original / -0.01:
  joint_mean = 11.6875
  new_u10_u14_mean = 0.0
  random_fp_mean = 0.0
  usable_rate = 0.4140625

repair_seed345 original / -0.01:
  joint_mean = 16.09375
  new_u10_u14_mean = 0.0
  random_fp_mean = 0.0
  usable_rate = 0.5390625

repair_seed012 dict B128 / -0.01:
  joint_mean = 19.1875
  new_u10_u14_mean = 7.9375
  random_fp_mean = 7.9375
  usable_rate = 0.0

repair_seed345 dict B128 / -0.0125:
  joint_mean = 39.4375
  new_u10_u14_mean = 23.8125
  random_fp_mean = 23.8125
  usable_rate = 0.0

repair_seed345 dict B192 / -0.0125:
  joint_mean = 39.4375
  new_u10_u14_mean = 23.8125
  random_fp_mean = 23.8125
  usable_rate = 0.015625
```

解释：

```text
bootstrap 没有推翻 margin 结论。
原始 source 的 usable_rate 比较高，但没有新增 U10-U14 support。
U10-U14 source 的 new support mean 与 random_fp mean 几乎同值，说明 bootstrap 下仍然绑定。
唯一 positive usable-rate 的 U10-U14 row 只有 0.015625，属于非常弱且不稳定的诊断点，不能作为 gate。
```

### 23.4 最新主审计状态

```text
run_id = official_from_v1217_v1216_artifacts_context_rank_dict_u10_u14_margin_bootstrap_refresh
route = R2-B320LabelInitDependenceDetected
p4_open = 0
promotion_allowed = 0
strict_visibility_pass = 0
strict_auc_joint_min = 0.05660377358490566
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
line_t_hard_gate_margin_repair_pass = 0
line_t_hard_gate_bootstrap_repair_pass = 0
line_t_hard_gate_bootstrap_u10_u14_positive_usable_rows = 1
```

最新 zip：

```text
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_sha256 = 638ac59eab9c7457ea823028f7e5840d9158e430acbec15f763c41c4df96b79c
zip_entries = 72
hash_manifest_entries = 47
```

zip 已包含：

```text
review_artifacts/v1218_hard_gate_margin_repair.csv
review_artifacts/v1218_hard_gate_margin_repair_summary.csv
review_artifacts/v1218_hard_gate_bootstrap_repair.csv
review_artifacts/v1218_hard_gate_bootstrap_repair_summary.csv
```

### 23.5 最终判断

至此，v12.18 计划内和 calibration 建议内的同类修复面已经真实覆盖：

```text
feature family repair = done
seed/window expansion = done
context-rank repair = done
actuator dictionary expansion = done
batch bracket = done
raise hard gate margin = done
bootstrap stability audit = done
```

最终仍然：

```text
Line T pass = 0
P3 open = 0
P4 open = 0
promotion_allowed = 0
```

当前结论不变但更扎实：

```text
U10-U14 dictionary expansion 暴露了一个 control-shared movement mode；
它不是可部署 label-free value source；
在 deterministic margin 和 bootstrap audit 下都不能稳定地与 random matched control 分离。
```
