# DG-KAN v12.19 实验结果复盘

> 结论先行：v12.19 完成了计划要求的代码语义审计、A15-A22 去混淆实验、Line C protocol autopsy、T1/T2/T3 feature tier cleanup、T1-only visibility v3、P3/P4 gate 审计和 zip 打包。  
> 最终 route：`R2-B320LabelInitDependenceDetected`。  
> 没有 functional promotion，没有 P3/P4 open，没有伪造通过。

---

## 1. Route

最终 route：

```text
R2-B320LabelInitDependenceDetected
```

原因：

```text
no label-free B320-like candidate passed task+LineC;
current B320 remains label-informed diagnostic anchor
```

这不是代码审查失败：

```text
line_r_code_semantics_pass = 1
implementation_readback_missing_refs = 0
cr0_cr15_missing_or_unknown = 0
strict_feature_blocklist_active = 1
```

也不是 P4 被遗漏：

```text
T1_only_visibility_pass = 0
line_i_open = 0
p4_open = 0
```

---

## 2. Implementation Readback / Code Rationale

本轮新增 `v1219_core_semantics_audit.md`，逐项审计 R1-R9。关键 readback 如下。

### R1. B320 construction path

实际路径：

```text
experiments/run_v1218_b320_label_free_ablation.py::train_one
-> make_model
-> experiments/run_v124_multibasis_functional_dual.py::_make_model
-> dgkan/models/fc_purekan_primitives.py::SimpleFastTaskGeometryKAN.__init__
```

核心 line refs：

```text
run_v1218_b320_label_free_ablation.py::make_model lines 349-371
run_v124_multibasis_functional_dual.py::_make_model lines 104-119
fc_purekan_primitives.py::SimpleFastTaskGeometryKAN lines 4050-5420
B320 spec line 6868
```

解释：

`uses_y_for_stats=1` 时，`make_model` 把 `y_train` 或 diagnostic y_stats 传给 `v124._make_model`，再进入 `SimpleFastTaskGeometryKAN(..., y_for_stats=...)`。`uses_y_for_stats=0` 时传入 `None`。

### R2. trainprobe path

核心 line refs：

```text
fc_purekan_primitives.py::probe_dirs line 4097
fc_purekan_primitives.py::trainprobe_signal_init_applied line 4119
fc_purekan_primitives.py::direct_readout[: self.input_dim line 4189
fc_purekan_primitives.py::quad_proj[:, col] = probe_dirs line 4354
```

解释：

当 variant 含 `trainprobe` 且 `y_for_stats is not None` 时，代码用 class mean direction 构造 `probe_dirs`。随后：

```text
trainprobedirect -> direct_readout
trainprobeP      -> quad_proj
signalBroad      -> broad signal mixing
signalBlock      -> block-local signal mixing
```

所以 B320 current 的初始化确实是 label-informed。

### R3. Label-free ablation path

核心 line refs：

```text
strip_trainprobe_tokens lines 170-181
ablation_specs lines 205-346
strict_label_free_init line 697
build_y_stats_for_mode lines 189-202
```

解释：

A2-A19 删除 trainprobe tokens。A1 是特殊 deconfounding：保留 trainprobe-capable variant，但 `y_stats=None`，因此可显示“只去掉 y_for_stats”的影响。A20-A22 是 diagnostic controls，不计入 official label-free claim。

### R4. Manual / fused update path

核心 line refs：

```text
step_model lines 509-527
run_v1283::_step_b109 lines 2009-2055
run_v126::_ManualForeachAdamW lines 147-188
fused_hinge_quadratic backward fused AdamW lines 1374-1396
```

解释：

B320 task training 走 B109/FHQ selected step implementation，可能使用 manual/fused AdamW；MLP control 走普通 torch AdamW。v12.19 没有改 loss、sampler、class weight 或 dataset branch。

### R5. Line C computation

核心 line refs：

```text
signal_reservoir_metrics_detailed lines 381-438
linec_measured line 655
v1252::_ridge_coupling lines 883-896
v1252::_sample_grad_sketch lines 899-918
```

解释：

`CouplingR2` 来自 before/after logits delta 的 ridge coupling。`NoiseSignalLeak` / `RealSignalReservoirRatio` 使用 CE residual 与 label/permuted label，明确是 audit-only metric，不允许作为 direction source。

### R6-R9

v12.19 新增：

```text
tier_for_feature lines 343-356
write_visibility_tiers lines 426-549
score_visibility_tier lines 580-623
write_gate_status lines 873-899
write_classic_monitor lines 902-911
```

解释：

Line T 被重分层为：

```text
T1 deployable precommit
T2 clone-probe diagnostic
T3 audit-only targets
BLOCKED forbidden feature families
```

P3/P4 只看 T1-only pass；T2 只能诊断，不 promotion。Classic line 本轮 monitor-only，没有新 hypothesis，不冒充进展。

---

## 3. 代码修改审计

本轮代码修改是为执行 v12.19 计划，而不是调 CE。

### 3.1 新增 label-free projector tokens

文件：

```text
dgkan/models/fc_purekan_primitives.py
```

新增 token：

```text
pcaorthomixp:
  PCA top components + orthogonal residual fill。

augstablep:
  unlabeled augmentation-stable input directions。

randomcotangentstablep:
  random cotangent response stability from unlabeled train-stream geometry。

covwhitenp:
  input covariance whitened projection frame。

blocklocalaugstablep:
  image block-local augmentation-stable basis。
```

这些 token 不读取 label，不使用 CE vector，不使用 validation/test/future outcome。

### 3.2 新增 A15-A22

文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
```

新增 candidates：

```text
A15-PCAOrthoMix-labelFree
A16-AugStableP-labelFree
A17-RandomCotangentStableP-labelFree
A18-InputCovWhitenedOrthoP-labelFree
A19-BlockLocalAugStableP-labelFree
A20-shuffledLabelTrainProbe-diagnostic
A21-randomClassCentroid-diagnostic
A22-permutedClassMeanP-diagnostic
```

新增 diagnostics：

```text
P_energy_on_top_pca
P_energy_on_aug_stable_subspace
P_condition
quad_feature_std_mean/min/max
direct_readout_norm
quad_proj_norm
quad_readout_norm
branch_scale_norm
```

### 3.3 新增 v12.19 runner

文件：

```text
experiments/run_v1219_b320_deconfounding_true_lossagnostic_visibility.py
```

职责：

```text
Line R semantic audit
Line A gate summary
Line C protocol autopsy
Line T feature tier cleanup
T1/T2 visibility scorer
P3/P4 gate closure
figures
zip/hash manifest
```

---

## 4. Line A：B320 去混淆结果

正式协议：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
train_size = 1024
val_size = 512
test_size = 512
epochs = 3
batch_size = 128
Line C batch = 64
Line C sketch_dim = 24
```

输出：

```text
v1219_b320_deconfounding_linea_ablation.csv
v1219_b320_deconfounding_linea_summary.csv
v1219_linea_candidate_gate_summary.csv
v1219_P_basis_alignment_diagnostics.csv
v1219_linea_reservoir_decomposition.csv
```

### 4.1 Gate summary

Task-side gate：

```text
mean_delta_vs_A0 >= -0.005
worst_delta_vs_A0 >= -0.015
AUC_time_ratio_vs_mlp <= 1.0
```

Line C gate：

```text
LineC_nontearing_all_pass = 1
```

实际结果：

| candidate | mean_delta_vs_A0 | worst_delta_vs_A0 | max_AUC_time_ratio_vs_mlp | LineC pass rate | official pass |
|---|---:|---:|---:|---:|---:|
| A1-noYForStats | -0.011284722222222222 | -0.037109375 | 0.8449234491133157 | 0.0 | 0 |
| A5-orthogonalP-labelFree | -0.011935763888888888 | -0.0390625 | 0.7512128806815016 | 0.0 | 0 |
| A11-directRead125-labelFree | -0.011935763888888888 | -0.0390625 | 0.7517288200466036 | 0.0 | 0 |
| A12-identityAmp150-labelFree | -0.012152777777777778 | -0.0390625 | 0.7524022413181619 | 0.0 | 0 |
| A15-PCAOrthoMix-labelFree | -0.028428819444444444 | -0.083984375 | 1.0296983958544903 | 0.0 | 0 |
| A16-AugStableP-labelFree | -0.037109375 | -0.060546875 | 0.9532077817377353 | 0.0 | 0 |
| A17-RandomCotangentStableP-labelFree | -0.016493055555555556 | -0.052734375 | 0.9088650297671436 | 0.0 | 0 |
| A18-InputCovWhitenedOrthoP-labelFree | -0.091796875 | -0.13671875 | 181.95322703972 | 0.0 | 0 |
| A19-BlockLocalAugStableP-labelFree | -0.021267361111111112 | -0.04296875 | 0.8499729153353995 | 0.0 | 0 |

结论：

```text
linea_official_label_free_pass_count = 0
linea_pass = 0
best_candidate_by_task_delta = A1-noYForStats
best_candidate_mean_delta_vs_A0 = -0.011284722222222222
```

A1 是 best，但仍没有达到 `-0.005` task gate，且 Line C pass rate 为 0。

### 4.2 Group B 结果解读

计划要求如果 A15-A19 task fail，输出 P basis alignment diagnostics。本轮已输出。

关键观察：

```text
A15-PCAOrthoMix:
  P_energy_on_top_pca_mean = 0.14202035466829935
  P_condition_mean = 4.54478022787306
  task 失败，Line C 失败。

A17-RandomCotangentStableP:
  task 比 A15/A16/A18/A19 好，但 mean_delta_vs_A0 = -0.01649，仍未过。

A18-InputCovWhitenedOrthoP:
  task 明显崩，AUC_time_ratio_vs_mlp = 181.95322703972。

A19-BlockLocalAugStableP:
  task 中等，Line C 仍不过。
```

这说明本轮新增的 manifold-aligned projector family 没有解决 label-init confound，也没有修复 Line C。

### 4.3 Group C diagnostic controls

| candidate | mean_delta_vs_A0 | max_AUC_time_ratio_vs_mlp |
|---|---:|---:|
| A20-shuffledLabelTrainProbe-diagnostic | -0.04796006944444445 | 1.0808232233215522 |
| A21-randomClassCentroid-diagnostic | -0.04665798611111111 | 0.9754105844224822 |
| A22-permutedClassMeanP-diagnostic | -0.06705729166666667 | 1.2532740681763106 |

诊断意义：

乱标签 / 随机 centroid / permuted class mean 都显著弱于 A0。这支持 H-A1：B320 current 的 trainprobe signal 不是任意 centroid 都能替代，而是依赖真实 label-informed initialization。

---

## 5. Line C：protocol autopsy

输出：

```text
v1219_linec_protocol_autopsy.csv
```

实际设置：

```text
checkpoint_id = v1219_A0_autopsy_retrained_checkpoint
dataset = MNIST
seed = 0
train_size = 1024
val_size = 512
epochs = 3
batch_hash = 613a061015b6bce710de4f1e1ae84ecfc0731ee275157a0741f7b7ad783966dc
```

比较：

```text
C0 locked-source shared v1252
C1 recovered b64/sketch32/d24
C2 v12.17 repair-source shared v1252
C3 no-update baseline
C4 AdamW-window
C5 fixed-batch deterministic repeat
```

结果：

```text
max_update_impl_metric_absdiff = 0.0
protocol_impl_consistency_pass = 1
```

限制：

```text
does_not_close_locked_vs_recovered_artifact_mismatch = 1
```

解释：

同一 retrained A0 checkpoint、同一 batch/hash、同一 v1252 primitive calculator 下，C0/C1/C2/C4/C5 一致。C3 是 no-update negative control，不计入 implementation-diff gate。

这说明本轮没有发现当前 shared calculator 内部实现不一致。但它不能证明 locked artifact 与 recovered-budget artifact 的历史差异已经被完全关闭。因此 Line C 可作为本轮内部一致性检查，但不能单独解除 B320 label-init / Line C blocker。

---

## 6. Line T：True T1-only visibility v3

本轮按计划修正 v12.18 的两个 feature provenance 问题：

```text
method_role_flags -> BLOCKED
output_subspace_drift / pred_* / target_norm* -> T2 clone-probe diagnostic only
hard release / label / CE target -> T3 audit-only
```

新增 artifact：

```text
v1219_visibility_feature_tier_manifest.csv
v1219_feature_provenance_strict_v2.csv
v1219_forbidden_feature_blocklist.csv
v1219_T1_deployable_features.csv
v1219_T2_clone_probe_diagnostic_features.csv
v1219_T3_audit_targets.csv
v1219_visibility_scores_T1_only.csv
v1219_visibility_scores_T1_plus_T2_diagnostic.csv
v1219_visibility_leaveout_T1_only.csv
v1219_visibility_support_audit.csv
```

Feature rows：

```text
feature_provenance_strict_v2_rows = 51
T1_deployable_feature_rows = 53568
T2_clone_probe_feature_rows = 17280
blocked_visibility_feature_rows = 17280
T3_audit_target_rows = 1728
```

Support audit：

```text
support_concentrated = 0
control_false_positive_rate = 0.0
```

T1-only visibility：

```text
T1_only_auc_joint_min = 0.0
T1_only_precision_joint_min = 0.0
T1_only_recall_joint_min = 0.0
T1_only_visibility_pass = 0
```

T1+T2 diagnostic：

```text
T1_plus_T2_diagnostic_auc_joint_min = 0.03773584905660377
diagnostic_pass_not_promotable = 0
```

解释：

清理掉 method identity 和 clone-probe drift 后，T1-only signal 更弱，joint hard release 在 heldout split 上完全不可见。即使加回 T2 clone-probe diagnostic，也没有形成可用可见性。

所以 Line T 失败不是因为 support concentrated 或 control false positive，而是当前 T1 feature family 对 joint hard release 没有稳定可见性。

---

## 7. Line I / P4 / Line D

Line I：

```text
line_i_open = 0
matched_controls_generated = 0
not_run_reason = Line I is gated by T1-only visibility; v12.19 T1-only visibility did not pass
```

P4：

```text
p4_open = 0
line_t_t1_visibility_pass = 0
line_i_response_pass = 0
not_run_reason = P4 requires Line R/A/C/T/I pass; Line A or Line T/I is closed
```

Line D：

```text
line_d_new_hypothesis_implemented = 0
```

本轮没有新 classic No-BSpline hypothesis，因此只做 monitor，不冒充推进。

---

## 8. 最终关键数据

```text
route = R2-B320LabelInitDependenceDetected
promotion_allowed = 0
line_r_code_semantics_pass = 1
linea_pass = 0
linea_official_label_free_pass_count = 0
linec_protocol_impl_consistency_pass = 1
T1_only_visibility_pass = 0
line_i_open = 0
p4_open = 0
figure_count = 19
hash_manifest_entries = 47
zip_entries = 59
```

zip：

```text
results/v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry/official_from_v1218_v1217_v1216_artifacts/v1219_code_review_packet.zip
sha256 = e551e83cd531af0bf1cf74cae48fd5ec0fed38fb2ce0292e22b3e9cd4aba7e97
```

---

## 9. 本轮科学结论

### 9.1 B320 claim scope

B320-current-labelInit 仍是强 diagnostic anchor，但本轮进一步支持：

```text
B320_claim_scope = label-informed supervised initialization anchor
external_ready_base_claim = 0
```

不能把它写成 label-free external-ready PureKAN base。

### 9.2 Label-free B320

A15-A19 没有找到能同时保留 task trajectory 和修复 Line C 的 candidate。A1/A5/A11/A12 比新增 Group B 更接近，但也没过 task gate，Line C 全失败。

所以 v12.19 没有完成 Success A。

### 9.3 Line C

同一 retrained checkpoint 的 calculator consistency pass，但这只是本轮内部实现一致性；它不等于历史 locked/recovered mismatch 已经完全解决。

### 9.4 Line T

T1-only strict visibility v3 失败：

```text
AUC_joint_min = 0.0
precision_at_k_joint_min = 0.0
recall_at_k_joint_min = 0.0
```

T2 diagnostic 也没有救回来。因此当前没有合法 functional value source。

### 9.5 P3/P4

P3/P4 正确关闭。任何在本轮打开 P4 的做法都会绕过 Line T 和 Line I gate。

---

## 10. 下一步建议

不建议继续在 B320 init token 上小网格搜索。v12.19 的结果显示：

```text
1. 去掉 y_for_stats 后，task 已经明显掉出 gate。
2. PCA/augmentation/cotangent/cov-whiten/block-local projector 不能修复。
3. Line C 的核心问题是 label-free projector 没有把真实残差/噪声正确分配到 signal/reservoir channel。
4. T1-only feature 对 hard joint release 不可见。
```

下一步更合理方向：

```text
1. 重新设计 label-free base architecture，而不是继续给 B320 current 打补丁。
2. Line C target 需要一个真正 precommit 的 signal/reservoir construction，而不是 CE residual audit target 的后验可见性。
3. 如果继续 Line T，应先产生新一代 T1 physical observable，而不是调 scorer 或降低 hard gate。
4. P3/P4 继续关闭，直到 T1-only visibility 真通过。
```

---

## 11. 2026-05-24 23:43 +08 完成性复核结论

用户再次追问计划是否完成后，我按原计划逐项回读并核对了 artifacts。复核结论是：v12.19 已按计划执行到合法终止点，合法终止点不是 functional promotion，而是 `Success B`。

本轮完成了：

```text
Line R code semantics + strict feature blocklist fix：完成
Line C protocol mismatch autopsy：完成
Line A Group A/B/C 去混淆实验：完成
Line T T1-only visibility v3：完成
Line I matched-control actuator dictionary：按 gate 关闭，未运行是正确行为
Line B P4 short-run：按 gate 关闭，未运行是正确行为
Line D classic monitor：完成 monitor-only 记录，无新 hypothesis 不重跑
code review packet zip：完成并复核
```

最终 route 仍为：

```text
route = R2-B320LabelInitDependenceDetected
fail_reason = no label-free B320-like candidate passed task+LineC; current B320 remains label-informed diagnostic anchor
```

这对应 v12.19 文档中的 `Success B`：

```text
All label-free attempts fail Line C;
current B320 is formally bounded as supervised-initialized diagnostic anchor;
external-ready base claim is explicitly closed until new architecture.
```

为什么没有继续扩到 seeds 0..9：

```text
文档 6.2 写的是 seeds 0,1,2 initially; pass 后扩到 0..9。
本轮没有任何 strict label-free candidate 同时满足 task-side pass 与 Line C pass。
因此扩 10-seed 的前置条件没有成立。
```

为什么没有继续 Line I/P4：

```text
Line T T1-only visibility did not pass。
T1_only_visibility_pass = 0
line_i_open = 0
p4_open = 0
```

这不是遗漏，而是 gate 生效。若强行继续 Line I/P4，会违反计划中的 gated-only 约束，并把 diagnostic/postprocess 写成 official success。

zip 复核：

```text
path = results/v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry/official_from_v1218_v1217_v1216_artifacts/v1219_code_review_packet.zip
sha256 = e551e83cd531af0bf1cf74cae48fd5ec0fed38fb2ce0292e22b3e9cd4aba7e97
entries = 59
contains code/experiments/run_v1219_b320_deconfounding_true_lossagnostic_visibility.py = True
contains code/experiments/run_v1218_b320_label_free_ablation.py = True
contains code/dgkan/models/fc_purekan_primitives.py = True
contains review_artifacts/v1219_feature_provenance_strict_v2.csv = True
```

本复核没有新增实验结论，也没有修改实验数据。它只确认：当前产物满足 v12.19 计划的执行闭环，并且本轮未通过的部分已经以 gate-closed 形式记录，而不是被跳过。

---

## 12. 2026-05-24 23:47 +08 二次硬核验结论

用户再次追问后，进一步从 CSV/JSON 内部字段检查是否真的满足计划闭环，而不是仅检查文件存在。

二次核验补充了三个要点。

第一，Line C autopsy 确认包含计划要求的 C0-C5：

```text
C0-locked-source-shared-v1252
C1-recovered-b64-sketch32-d24
C2-v1217-repair-source-shared-v1252
C3-no-update-baseline
C4-AdamW-window
C5-fixed-batch-deterministic-repeat
```

其中 C0/C1/C2/C4/C5 在同一 batch hash 下的三项核心 metric 一致，summary row 记录：

```text
max_update_impl_metric_absdiff = 0.0
protocol_impl_consistency_pass = 1
```

C3 是 no-update negative control，不用于 update-impl diff gate。这个结果说明本轮 Line C 实现一致性 blocker 已关闭，但仍不宣称历史 locked artifact mismatch 被完全解释。

第二，Line A 的 strict label-free 候选全未满足继续条件。二次核验的 gate summary 显示：

```text
A1-noYForStats:
  strict_label_free_rows = 9
  task_side_label_free_pass = 0
  linec_pass = 0
  official_label_free_b320_pass = 0
  mean_delta_vs_A0 = -0.011284722222222222
  worst_delta_vs_A0 = -0.037109375

A5-orthogonalP-labelFree:
  strict_label_free_rows = 9
  task_side_label_free_pass = 0
  linec_pass = 0
  official_label_free_b320_pass = 0
  mean_delta_vs_A0 = -0.011935763888888888
  worst_delta_vs_A0 = -0.0390625

A15-PCAOrthoMix-labelFree:
  strict_label_free_rows = 9
  task_side_label_free_pass = 0
  linec_pass = 0
  official_label_free_b320_pass = 0
  mean_delta_vs_A0 = -0.028428819444444444
  worst_delta_vs_A0 = -0.083984375

A16-AugStableP-labelFree:
  strict_label_free_rows = 9
  task_side_label_free_pass = 0
  linec_pass = 0
  official_label_free_b320_pass = 0
  mean_delta_vs_A0 = -0.037109375
  worst_delta_vs_A0 = -0.060546875

A17-RandomCotangentStableP-labelFree:
  strict_label_free_rows = 9
  task_side_label_free_pass = 0
  linec_pass = 0
  official_label_free_b320_pass = 0
  mean_delta_vs_A0 = -0.016493055555555556
  worst_delta_vs_A0 = -0.052734375

A18-InputCovWhitenedOrthoP-labelFree:
  strict_label_free_rows = 9
  task_side_label_free_pass = 0
  linec_pass = 0
  official_label_free_b320_pass = 0
  mean_delta_vs_A0 = -0.091796875
  worst_delta_vs_A0 = -0.13671875

A19-BlockLocalAugStableP-labelFree:
  strict_label_free_rows = 9
  task_side_label_free_pass = 0
  linec_pass = 0
  official_label_free_b320_pass = 0
  mean_delta_vs_A0 = -0.021267361111111112
  worst_delta_vs_A0 = -0.04296875
```

因此没有触发 seeds 0..9 扩展条件。

第三，Line T/I/P4 关闭条件被结果文件直接支持：

```text
T1 visibility rows = 1728
T1 feature_columns = 31
T1 AUC_joint_min = 0.0
T1 precision_at_k_joint_min = 0.0
T1 recall_at_k_joint_min = 0.0

T1+T2 diagnostic rows = 1728
T1+T2 feature_columns = 41
T1+T2 AUC_joint_min = 0.03773584905660377
T1+T2 precision_at_k_joint_min = 0.0
T1+T2 recall_at_k_joint_min = 0.0

support_concentrated = 0
control_false_positive_rate = 0.0
line_i_open = 0
p4_open = 0
```

所以 Line T 失败不是 support concentration 或 control false positive 导致的假失败，而是 T1 feature family 对 hard joint release 的可见性不足。

二次核验后的最终状态不变：

```text
v12.19_status = completed_as_Success_B
route = R2-B320LabelInitDependenceDetected
no_10_seed_extension_reason = no strict label-free candidate passed task + Line C gates
line_i_not_run_reason = T1-only visibility did not pass
p4_not_run_reason = Line A/T/I gates closed
```

这不是放弃，也不是漏跑；这是按 v12.19 文档 gate 逻辑执行后的合法完成。
