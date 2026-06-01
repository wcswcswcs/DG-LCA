# DG-KAN v12.19：B320 去混淆、真实 Loss-Agnostic 可见性、Functional Re-entry 完整计划

> 版本：v12.19 execution plan  
> 基于：v12.18 B320 code audit / label-init audit / strict visibility repair 结果、本地解压 `v1218_code_review_packet.zip` 后的静态代码审查。  
> 总原则：不再做 B320 小修、不按数据集调参、不针对 CE 设计 functional direction、不把 diagnostic/postprocess 写成 official success。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。

---

# 0. 一句话结论

v12.18 不是没有进展。它的关键进展不是 functional 成功，而是把项目中两个长期混淆的问题暴露出来：

$$
\boxed{
\text{B320 current anchor 很强，但它依赖 label-informed trainprobe initialization；}
\text{strict loss-agnostic visibility 仍没有成立。}
}
$$

因此，当前不能继续把问题简化成：

```text
B320 已经 fully solved，只差 functional update。
```

更准确的状态是：

```text
1. B320-current-labelInit 可以继续作为 functional diagnostic anchor。
2. B320 还不能作为 label-free / external-ready PureKAN base claim。
3. Label-free A5-orthogonalP 在 task side 已经接近 current B320，但 Line C non-tearing 失败。
4. Functional update 的 loss-agnostic value source 仍不可见；P3/P4 正确关闭。
5. 代码审查发现 strict feature provenance 仍需要更硬的人工审查，不能完全相信 Codex 自审。
```

所以 v12.19 的核心任务不是小修模型，也不是继续扫 lambda，而是：

$$
\boxed{
\text{先把 B320 base 的 label-init confound 与 Line C protocol mismatch 解决，}
\text{再重建真正 pre-commit、loss-agnostic、control-resistant 的 functional visibility。}
}
$$

---

# 1. 当前结果的独立解读

## 1.1 Line R：代码审查闭包有进展，但还不能完全信任

v12.18 修复了 v12.17.2 code packet 不闭包的问题。当前 packet 已经包含 v1218 / v1217 / v1216 / v1215 / v1283 / v1252 / v124 / v120 和 `dgkan` 核心文件，CR0-CR15 都有 file / symbol / line range。

这是真进展，因为之前很多核心实现没有被审查，Codex 可以只交 CSV 和 route，而不解释实际代码路径。现在至少有了：

```text
v1218_core_code_review_manifest.csv
v1218_symbol_line_map.json
v1218_manual_review_packet.md
v1218_review_blocker_table.csv
v1218_transitive_dependency_manifest.csv
```

但当前仍不能 promotion，因为：

```text
manual_review_pending_count = 16
```

并且我本地静态审查看到两个必须修正的问题：

```text
1. B320 的 trainprobe 初始化确实可能使用 y_for_stats 标签。
2. strict feature list 中仍混入 method_role_flags、clone-probe / output-subspace drift 这类不应直接作为 deployable pre-commit functional feature 的字段。
```

因此 v12.19 需要把代码审查从“Codex 自报有 line range”升级为“人工可读的核心语义审计 + negative controls”。

## 1.2 Line A：B320 current 仍强，但它不是 label-free external-ready base

当前 B320-current-labelInit：

```text
step_ratio_q90 = 0.39931987348441034
memory_ratio_q90 = 0.1295238095238095
mean_delta_vs_mlp = +0.027669270833333332
worst_delta_vs_mlp = -0.001953125
```

这说明它仍是非常强的 diagnostic anchor。

但代码层面已经确认，在 `SimpleFastTaskGeometryKAN.__init__` 中，当 variant 包含 `trainprobe` 且 `y_for_stats is not None` 时，会计算 class mean direction：

```text
y_stats = y_for_stats[:xs.shape[0]]
for class_idx:
    vec = mean(z_stats[class]) - global_center
```

然后这些 `probe_dirs` 会进入：

```text
trainprobedirect -> direct_readout
trainprobeP      -> quad_proj
signalBroad      -> broad signal mixing
signalBlock      -> block-local signal mixing
```

这不是 functional update 的 label leakage，但它是 base initialization 的 label dependence。

因此 B320 当前 claim 必须限定为：

```text
B320_claim_scope = label-informed supervised initialization anchor
external_ready_base_claim = 0
```

## 1.3 Label-free ablation 的真实状态：task side 接近，但 geometry side 不过

v12.18 不只是发现 ablation missing，后来已经跑了 recovered anchor-budget 级 A0-A14。

最重要的结果是：

```text
A5-orthogonalP-labelFree:
  mean_delta_vs_mlp = +0.018663194444444444
  worst_delta_vs_mlp = -0.005859375
  mean_delta_vs_A0 = -0.00390625
  max_AUC_time_ratio_vs_mlp = 0.7776700385171795
  LineC_nontearing = 0/9
```

这说明：

```text
1. label-free 不是完全没希望。
2. A5 的 task/AUC 已经接近 A0-labelInit。
3. 但 A5 的 Line C non-tearing 完全没过。
```

A6-A14 的修复方向也已经被排除：

```text
A6 activeP64：几乎等价于 A5，无 Line C 改善。
A7/A13/A14 lowQuad：task 下降，reservoir ratio 反而升高。
A8/A9/A10 boundQ：task 明显失败，noise leak 不修。
A11/A12 direct branch：task 接近 A5，但 reservoir ratio 只微小下降。
```

所以 v12.19 不应继续 A15/A16 token grid。必须换问题层级。

## 1.4 Line C detailed decomposition 给出了更本质的失败机制

A5 的 Line C 失败不是单纯 signal eigenspace 太窄。

当前 detailed Line C 均值显示：

```text
MLP:
  signal_mass_topk = 0.8611922595236037
  reservoir_fraction = 0.13880774047639635
  real_residual_energy = 23.124738832314808
  real_total_energy = 95.83334681722853
  noise_signal_energy = 55.435595217678284
  noise_total_energy = 1015.8765869140625

A5-orthogonalP-labelFree:
  signal_mass_topk = 0.8355417980088128
  reservoir_fraction = 0.16445820199118721
  real_residual_energy = 5.526151098724869
  real_total_energy = 9.12627146144708
  noise_signal_energy = 120.40638648139105
  noise_total_energy = 1104.3729315863716
```

解释：

```text
1. A5 的 reservoir_fraction 只比 MLP 高约 0.026，不是灾难级子空间窄化。
2. 但 A5 的 real_total_energy 很小，real_residual_energy 没有同比例下降。
3. 因此 RealSignalReservoirRatio 高，主要来自真实残差信号相对集中在 reservoir 方向。
4. A5 的 noise_signal_energy 又比 MLP 高很多，说明 NoiseSignalLeak 也是真问题。
```

所以 A5 的失败本质是：

$$
\boxed{
\text{label-free projector 让最终 task 可学，但没有把真实残差信号和噪声正确分配到 signal/reservoir channel。}
}
$$

这不能靠降低 quadratic branch 或增加 direct readout 解决。

## 1.5 Line T：strict loss-agnostic visibility 仍然没有成立

v12.18 做了多轮修复：

```text
1. 修复 strict feature 与 hard release 的 semantic join key。
2. 加入 strict v12.16 loss-agnostic features。
3. 扩展 source/support。
4. 加入 output-subspace drift repair。
5. 做 family ablation。
6. 做 composite / context-rank repair。
7. 做 hard-gate margin / bootstrap repair。
```

最终 route 中：

```text
strict_visibility_feature_rows = 81216
strict_visibility_wide_rows = 1728
strict_visibility_feature_columns = 47
strict_auc_joint_min = 0.05660377358490566
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
```

这说明现在不是 source 不够的问题。即使 support_concentrated 已经被消掉，strict feature 仍不能在 heldout setting 下找到 joint hard release。

最关键的判断是：

$$
\boxed{
\text{当前 loss-agnostic feature family 仍不能作为 functional value source。}
}
$$

## 1.6 我对 v12.18 的独立结论

v12.18 的真实贡献是：

```text
1. 代码包闭包了。
2. B320 label-informed init 被明确证实。
3. Label-free A5 证明 task side 有希望。
4. Line C 证明 label-free geometry 不健康。
5. Strict visibility 多轮修复后仍失败。
6. P3/P4 没有被错误打开。
```

它没有完成：

```text
1. 没有关闭 B320 label-init confound。
2. 没有得到 label-free geometry-healthy base。
3. 没有得到 loss-agnostic functional value source。
4. 没有打开 P3/P4。
5. 没有产生 official functional success。
```

所以这次不是“慢到没进度”，而是：

$$
\boxed{
\text{项目从 base-performance 关卡进入 base-causality 与 functional-observability 关卡。}
}
$$

---

# 2. 当前各线进度估计

这里的百分比表示“离该线当前阶段目标的完成度”，不是最终论文完成度。

| Line | 完成度 | 状态 |
|---|---:|---|
| Line R 代码审查 / 依赖闭包 | 75% | packet 闭包，但 strict feature provenance 和 B320 construction 仍需人工确认 |
| Line A B320 current anchor | 85% | labelInit anchor 很强，但 external-ready claim 被 label init 限制 |
| Line A label-free B320 | 55% | A5 task side 接近，Line C 完全不过 |
| Line C audit-only metrics | 75% | 有 detailed decomposition，但 protocol mismatch 仍要修 |
| Line T strict visibility | 35% | feature table / leaveout / support 已跑，但 gate 完全不过 |
| Line I actuator dictionary | 0% | Line T 未过，不应打开 |
| Line B functional P4 | 0% | P3/P4 closed，official functional success 为 0 |
| Line D classic No-BSpline monitor | 100% monitor-only | 本轮无新 hypothesis，不抢主线 |

总体判断：

$$
\boxed{
\text{Base engineering 约 80% 以上；functional mechanism 仍约 15%-20%。}
}
$$

---

# 3. v12.19 总目标

v12.19 不再把“再跑一个 functional candidate”作为目标。总目标是建立一个可审计的因果路径：

$$
\boxed{
\text{B320 的优势来源必须被去混淆；}
\text{functional 的 value source 必须是真正 pre-commit、loss-agnostic、control-resistant。}
}
$$

具体目标：

```text
G1. 明确 B320 current 的 label-init 贡献边界。
G2. 找到一个 label-free B320-like candidate，至少 task side 不输 current anchor，且 Line C 不撕裂。
G3. 修复 Line C protocol mismatch，确保 recovered-budget A0 与 locked B320 的 Line C 可比较。
G4. 清理 strict feature set，剔除 method identity、clone-probe response、post-response drift 等不合规字段。
G5. 重新验证 loss-agnostic visibility；若仍失败，停止 P3/P4，不继续调 selector。
G6. 只有 Line T pass 后，才打开 matched-control actuator dictionary 和 P4 short-run。
```

---

# 4. 硬约束

所有 v12.19 实验必须满足：

```text
strict FC-PureKAN
no teacher
no distillation
no loss modification
no sampler / class weight
no dataset-name branch
no validation/test/future outcome for commit
no CE vector / label / permuted-label CE for functional direction
CE / ECE / CEp99 / Brier / hard release 只能作为 audit metric
functional direction 必须 loss-agnostic
P3/P4 必须 gated
Codex 必须写 implementation readback + file/symbol/line range
```

特别强调：

$$
\boxed{
\text{Functional update 不许针对 CE 设计。}
}
$$

允许：

```text
CE / label 用于 audit-only：评价 hard release、tail、ECE、CEp99。
```

禁止：

```text
CE / label 用于 direction source、feature source、candidate selector、commit gate。
```

---

# 5. Line R：核心代码审查与实现语义闭环

## 5.1 目标

v12.19 第一关不是跑实验，而是保证我们知道代码实际在做什么。Codex 不能只给 route 和 CSV。

## 5.2 Codex 必须在复盘中写清楚的核心代码路径

Codex 必须在结果复盘中新增章节：

```text
Implementation Readback / Code Rationale
```

并且逐项说明：

```text
R1. B320 construction path:
    actual files, function/class, line ranges, call chain, y_for_stats 如何传入。

R2. SimpleFastTaskGeometryKAN trainprobe path:
    probe_dirs 如何从 y_for_stats 构造；如何写入 direct_readout / quad_proj。

R3. Label-free ablation path:
    A1-A14 如何 strip trainprobe tokens；哪些仍可能保留 trainprobe-capable buffer；strict_label_free_init 如何计算。

R4. Manual / fused update path:
    B320 task training 是否用 loss.backward；manual AdamW / fusedProjGradAdamW 如何实现。

R5. Line C computation:
    CouplingR2, NoiseSignalLeak, RealSignalReservoirRatio 如何计算；哪些量用 label / CE；哪些只 audit。

R6. Line T feature provenance:
    每个 feature family 的 source column、是否 pre-commit、是否 clone-probe、是否 method identity。

R7. Strict visibility scorer:
    train/test split、leaveout keys、precision@k、support concentration 如何计算。

R8. P3/P4 gate:
    什么条件下打开；本轮为什么关闭。

R9. Classic-family monitor:
    是否新跑；没有新 hypothesis 时不得冒充进展。
```

## 5.3 新增 artifacts

```text
v1219_code_review_delta.csv
v1219_core_semantics_audit.md
v1219_feature_provenance_strict_v2.csv
v1219_forbidden_feature_blocklist.csv
v1219_b320_construction_trace.json
v1219_linec_metric_trace.json
```

## 5.4 Gate

如果以下任一条件不满足，route 必须为：

```text
R0-CodeSemanticsAuditIncomplete
```

条件：

```text
CR0-CR15 全部有 file/symbol/line range。
B320 y_for_stats path 有明确 line range。
Line C audit-only 与 deployable feature 明确分离。
Strict feature blocklist 生效。
P3/P4 gate 没有被 bypass。
```

---

# 6. Line A：B320 去混淆与 label-free anchor 修复

## 6.1 假设

### H-A1

B320 current 的强性能部分来自 label-informed trainprobe initialization。

### H-A2

存在 label-free 初始化，可以保留 B320 的 task trajectory，同时降低 Line C reservoir/noise 问题。

### H-A3

如果所有 label-free 初始化都无法过 Line C，则 B320 只能作为 supervised-initialized diagnostic anchor，而不是 external-ready base。

## 6.2 实验设计

固定同一协议：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2 initially; pass 后扩到 0..9
train_size = 1024
val_size = 512
test_size = 512
epochs = recovered locked anchor protocol + exact reproduced protocol
batch_size = 128
same MLP control
same Line C calculator
same seed_base
```

候选分三组。

### Group A：当前已有候选复验

```text
A0-labelInit
A1-noYForStats
A5-orthogonalP-labelFree
A11-directRead125-labelFree
A12-identityAmp150-labelFree
MLP-same-step-FLOP-AdamW
```

目的：只复验最有信息量候选，不再跑 A6-A10/A13/A14 大网格。

### Group B：manifold-aligned label-free projector

不使用 label，只使用 train-stream input geometry 和 augmentation consistency。

```text
A15-PCAOrthoMix-labelFree
A16-AugStableP-labelFree
A17-RandomCotangentStableP-labelFree
A18-InputCovWhitenedOrthoP-labelFree
A19-BlockLocalAugStableP-labelFree
```

设计思路：

```text
PCAOrthoMix:
  PCA top components + orthogonal residual basis。

AugStableP:
  选择对轻微 augmentation / noise 稳定的 input directions。

RandomCotangentStableP:
  用随机 cotangent 的 label-free logit-Jacobian 稳定方向，而不是 class mean。

InputCovWhitenedOrthoP:
  在 input covariance whitened space 中采样 orthogonal P，避免随机 orthogonal 与 data manifold 错配。

BlockLocalAugStableP:
  复用 signalBlock 思想，但 block selection 基于 unlabeled augmentation stability。
```

### Group C：label-init deconfounding controls

```text
A20-shuffledLabelTrainProbe-diagnostic
A21-randomClassCentroid-diagnostic
A22-permutedClassMeanP-diagnostic
```

这些不用于 official label-free claim，只用于判断 B320 current 的 label-informed signal 是否只是标签泄漏式捷径。

## 6.3 记录指标

```text
candidate_id
dataset
seed
uses_y_for_stats
trainprobe_signal_init_uses_labels
trainprobe_signal_init_applied
strict_label_free_init
val_acc
test_acc
mean_delta_vs_mlp
worst_delta_vs_mlp
mean_delta_vs_A0
val_loss_auc_step
val_loss_auc_time
AUC_step_ratio_vs_mlp
AUC_time_ratio_vs_mlp
ECE
NLL
CEp99
margin_p10
step_ratio_q90
memory_ratio_q90
LineC_CouplingR2
LineC_NoiseSignalLeak
LineC_RealSignalReservoirRatio
LineC_signal_mass_topk
LineC_reservoir_fraction
LineC_real_residual_energy
LineC_real_total_energy
LineC_noise_signal_energy
LineC_noise_total_energy
LineC_nontearing_pass
```

## 6.4 判断标准

Task-side label-free pass：

$$
\Delta Acc_{mean}(A_i - A0) \ge -0.005
$$

$$
\Delta Acc_{worst}(A_i - A0) \ge -0.015
$$

$$
AUCtime_{A_i}/AUCtime_{MLP} \le 1.0
$$

Line C pass：

$$
CouplingR^2_{A_i} \ge CouplingR^2_{MLP} - 0.02
$$

$$
NoiseSignalLeak_{A_i} \le NoiseSignalLeak_{MLP} + 0.02
$$

$$
RealSignalReservoirRatio_{A_i} \le RealSignalReservoirRatio_{MLP} + 0.02
$$

Official label-free B320 pass：

```text
task-side pass = 1
Line C pass = 1
strict_label_free_init = 1
no CE / label used for direction = 1
```

## 6.5 不满足条件时 Codex 先尝试什么

如果 A15-A19 task 失败：

```text
不要调 dataset-specific threshold。
先输出 input/P basis alignment diagnostics：
  P_energy_on_top_pca
  P_energy_on_aug_stable_subspace
  P_condition
  quad_feature_std
  branch_norms
再调整 global family，而不是按数据集调参。
```

如果 task pass 但 Line C fail：

```text
不要继续 direct/quad scalar grid。
先拆 RealSignalReservoirRatio：
  real_residual_energy / real_total_energy
  signal_mass_topk
  reservoir_fraction
  p_sig rank
判断是 residual concentrated in reservoir 还是 signal subspace too narrow。
```

如果 A0 locked Line C 仍不能复现：

```text
停止 Line A promotion。
先进入 Line C protocol autopsy。
```

## 6.6 可视化

```text
fig_v1219_label_free_task_delta.svg
fig_v1219_label_free_auc_time.svg
fig_v1219_label_free_linec_scatter.svg
fig_v1219_real_residual_energy_ratio.svg
fig_v1219_noise_signal_energy_ratio.svg
fig_v1219_P_basis_alignment.svg
fig_v1219_A0_locked_vs_recovered_protocol.svg
```

---

# 7. Line C：Line C protocol mismatch autopsy

## 7.1 目标

当前 recovered-budget A0 任务复现 locked anchor，但 Line C 不能复现。这是硬 blocker。

v12.19 必须回答：

$$
\boxed{
\text{Line C mismatch 来自模型状态、batch/window/sketch protocol、metric implementation，还是 locked artifact 与 recovered budget 不同？}
}
$$

## 7.2 实验设计

对同一组 checkpoint / model state，运行多个 Line C calculators：

```text
C0 locked-source LineC calculator
C1 v12.18 recovered b64/sketch32/d24 calculator
C2 v12.17 repair-source calculator
C3 no-update baseline calculator
C4 AdamW-window calculator
C5 fixed-batch deterministic calculator
```

关键是：同一模型、同一 batch、同一 seed，比较不同 calculator 是否一致。

## 7.3 记录字段

```text
linec_impl_id
checkpoint_id
candidate_id
dataset
seed
batch_hash
window
sketch_dim
rank
update_window_type
uses_ce_for_audit
uses_label_for_audit
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
signal_mass_topk
reservoir_fraction
top_eigen_share
signal_top_count
real_residual_energy
real_total_energy
noise_signal_energy
noise_total_energy
```

## 7.4 判断标准

同一 checkpoint 的 metric 差异必须满足：

$$
|CouplingR^2_{implA} - CouplingR^2_{implB}| \le 0.02
$$

$$
|NoiseSignalLeak_{implA} - NoiseSignalLeak_{implB}| \le 0.02
$$

$$
|RealSignalReservoirRatio_{implA} - RealSignalReservoirRatio_{implB}| \le 0.02
$$

如果不同实现差异超过阈值，则 Line C 不能作为 promotion gate。

## 7.5 不满足条件时 Codex 先尝试什么

```text
1. 固定 exact batch/hash，排除 batch sampling mismatch。
2. 固定 random cotangent seed，排除 sketch randomness。
3. 输出 p_sig / p_res projector overlap。
4. 分别检查 CE residual、noise residual、grad sketch matrix。
5. 如果只有 CE residual 不一致，检查 label / logits / update window。
6. 如果 grad sketch 不一致，检查 _sample_grad_sketch 调用和 model.eval/train 状态。
```

---

# 8. Line T：True Pre-commit Loss-Agnostic Visibility v3

## 8.1 当前问题

v12.18 的 strict feature table 仍存在两个问题：

```text
1. method_role_flags 被标成 strict allowed，但 method identity 不是物理 pre-commit observable。
2. output_subspace_drift / pred_* 需要 clone_probe，不能直接作为 deployable direction feature。
```

即使这些 feature 没有让 visibility pass，也必须清理。否则未来一旦 pass，会产生假阳性。

## 8.2 Feature 分层

### T1 deployable pre-commit features

允许：

```text
input covariance / PCA spectrum
augmentation consistency
basis occupancy / branch energy
label-free logit covariance spectrum
random cotangent Jacobian norms
projector stability computed before candidate selection
window stability computed from prior unlabeled states
```

禁止：

```text
method identity flags
numeric hash
actual candidate response
actual CouplingR2_delta
pred_* from post-response probe
CE / label / permuted-label CE
hard release label
validation/test/future outcome
dataset name
```

### T2 clone-probe diagnostic features

允许用于 offline diagnosis，不可用于 promotion：

```text
output_subspace_drift
pred_noise_delta
pred_reservoir_delta
pred_coupling_delta
clone-probe response summary
```

### T3 audit-only target labels

只用于评估：

```text
hard_noise_release
hard_reservoir_release
hard_joint_release
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
ECE_delta
```

## 8.3 新增 artifacts

```text
v1219_visibility_feature_tier_manifest.csv
v1219_T1_deployable_features.csv
v1219_T2_clone_probe_diagnostic_features.csv
v1219_T3_audit_targets.csv
v1219_visibility_scores_T1_only.csv
v1219_visibility_scores_T1_plus_T2_diagnostic.csv
v1219_visibility_leaveout_T1_only.csv
v1219_visibility_support_audit.csv
```

## 8.4 Visibility gate

T1-only gate：

$$
AUC_{joint,min} \ge 0.70
$$

$$
Precision@k_{joint,min} \ge 0.25
$$

$$
Recall@k_{joint,min} \ge 0.20
$$

Robustness gate：

```text
leave-dataset-out pass
leave-seed-out pass
leave-window-out pass
leave-source-run-out pass
support_concentrated = 0
control_false_positive_rate = 0
```

只有 T1-only pass，才能进入 Line I。

T1+T2 diagnostic pass 不能 promotion，只能说明 clone-probe 信息有解释力。

## 8.5 不满足条件时 Codex 先尝试什么

如果 AUC 很低：

```text
不要继续调 ridge/scorer。
先扩展 T1 feature family：
  augmentation-stable P alignment
  branch energy temporal drift
  unlabeled logit covariance curvature
  random cotangent persistence
```

如果 AUC 高但 precision/recall 为 0：

```text
检查 support imbalance。
增加 independent source runs；不要降低 hard threshold。
```

如果 support concentrated：

```text
生成新 source：seeds 0..5 / windows 5,10,15 / two actuator dictionaries。
不允许用单一 seed/window/family promotion。
```

如果 T2 pass 但 T1 fail：

```text
说明可见性需要 clone-probe response。
可以设计 cheap legal microprobe，但必须计入 overhead，并且不能用 audit label/CE。
```

---

# 9. Line I：Actuator dictionary v3, gated only after Line T pass

## 9.1 目标

只有 T1-only visibility pass 后才打开 Line I。Line I 不是寻找 value source，它只是 executor。

## 9.2 必须包含 matched controls

```text
NoOp
RandomMatchedNorm
AdamWParallel
SNR-only
Shuffled actuator basis
Matched role-energy random actuator
Matched branch-energy random actuator
T1-score shuffled candidate
```

## 9.3 记录指标

```text
actuator_id
target_score_T1
actuator_norm
matched_norm
sketch_delta_fro
projector_angle
logit_max_abs_drift
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
ECE_delta
control_gap
```

## 9.4 Gate

$$
\Delta CouplingR^2 \ge 0.02
$$

$$
\Delta NoiseSignalLeak \le -0.01
$$

$$
\Delta RealSignalReservoirRatio \le -0.01
$$

$$
control\_gap \ge 0.005
$$

$$
logit\_max\_abs\_drift \le 0.05
$$

---

# 10. Line B：Functional P4 short-run

## 10.1 开启条件

P4 只在以下全部满足后开启：

```text
Line R code semantics pass
Line A claim scope resolved
Line C protocol reproducible
Line T T1-only visibility pass
Line I matched-control actuator pass
```

## 10.2 P4 训练形式

Functional update 不替代 task optimizer，只作为低频 maintenance：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{AdamW}+\lambda_t\Delta\theta_{func}
$$

触发条件只能来自 T1 features：

```text
unlabeled geometry debt
T1 visibility score
branch/primitive energy imbalance
projector stability drift
```

不能来自：

```text
CE deterioration
validation loss
label-defined hard release
dataset name
future outcome
```

## 10.3 P4 gate

Functional must beat controls：

```text
B320 + AdamW
B320 + NoOp matched overhead
B320 + RandomMatchedNorm
B320 + AdamWParallel
B320 + SNR-only
B320 + shuffled T1-score functional
```

Hard gates：

$$
Acc_{func} \ge Acc_{base} - 0.003
$$

$$
AUCtime_{func} \le AUCtime_{base}
$$

$$
ECE_{func} \le ECE_{base}+0.01
$$

$$
CEp99_{func} \le CEp99_{base}+0.05
$$

$$
NoiseSignalLeak_{func} \le NoiseSignalLeak_{base}-0.01
$$

$$
RealSignalReservoirRatio_{func} \le RealSignalReservoirRatio_{base}-0.01
$$

$$
ControlGap \ge 0.005
$$

## 10.4 不满足条件时 Codex 先尝试什么

如果 task 受伤：

```text
降低 event frequency，而不是调 CE。
检查 logit drift / branch norm / update norm。
```

如果 controls 也赢：

```text
functional 方向没有独立价值。
回到 Line T / Line I，不调 lambda。
```

如果 only CouplingR2 提高：

```text
不 promotion。
必须同时看 noise/reservoir release。
```

---

# 11. Line D：Classic No-BSpline portfolio

## 11.1 当前状态

本轮不把 classic line 当主 blocker。B-spline 继续 frozen。

Active monitor：

```text
Rational
Chebyshev
Wavelet
RBF/FastKAN
Fourier
```

## 11.2 规则

只有出现新 hypothesis 时才跑：

```text
Rational: 如果提出 loss-agnostic geometry/coupling repair，而不是 CE tuning。
Chebyshev/Wavelet: 如果提出 task-stable coordinate repair。
RBF/Fourier: 如果提出 A4 expression repair 且不破坏 L3 efficiency。
```

否则只记录 status，不占主预算。

---

# 12. 必须生成的图

```text
fig_v1219_route_dashboard.svg
fig_v1219_b320_label_init_ablation_task.svg
fig_v1219_b320_label_init_ablation_linec.svg
fig_v1219_linec_protocol_mismatch.svg
fig_v1219_reservoir_decomposition_a5_vs_mlp.svg
fig_v1219_noise_signal_energy_a5_vs_mlp.svg
fig_v1219_visibility_pr_curve_T1_only.svg
fig_v1219_visibility_auc_leaveout_T1_only.svg
fig_v1219_visibility_support_heatmap.svg
fig_v1219_feature_tier_ablation.svg
fig_v1219_actuator_control_gap.svg
fig_v1219_p4_short_run_if_opened.svg
```

---

# 13. Route definitions

```text
R0-CodeSemanticsAuditIncomplete:
  code review / feature provenance / line range missing.

R1-B320AnchorRegression:
  current B320 task or efficiency no longer reproduces.

R2-B320LabelInitDependenceDetected:
  current B320 uses label-informed init and no label-free LineC-clean replacement exists.

R3-LineCProtocolMismatch:
  recovered protocol cannot reproduce locked B320 Line C.

R4-StrictVisibilityFailed:
  T1-only loss-agnostic visibility fails.

R5-ActuatorResponseFailed:
  T1 visibility passes, but matched-control actuator response fails.

R6-P4FunctionalFailed:
  P4 opens but functional does not beat controls.

R7-FunctionalOfficialCandidate:
  B320 claim scope clean or bounded, T1 visibility passes, actuator pass, P4 functional beats controls.
```

---

# 14. Final success criteria for v12.19

v12.19 success does not require final 10-seed paper claim. It requires at least one of the following high-quality outcomes:

## Success A：B320 label-free base cleaned

```text
A label-free B320-like candidate passes task + Line C under reproducible protocol.
```

## Success B：B320 claim scope finalized

```text
All label-free attempts fail Line C;
current B320 is formally bounded as supervised-initialized diagnostic anchor;
external-ready base claim is explicitly closed until new architecture.
```

This is a valid scientific outcome because it prevents false claims.

## Success C：True T1 visibility found

```text
T1-only precommit loss-agnostic features predict hard joint release robustly.
```

## Success D：Functional P3/P4 opens legitimately

```text
Line T and Line I pass; P4 short-run opens with strong controls.
```

Any result that only improves CouplingR2, only improves task, or only passes a single seed/window/dataset is not success.

---

# 15. 最终执行优先级

```text
Priority 1:
  Line R code semantics + strict feature blocklist fix.

Priority 2:
  Line C protocol mismatch autopsy.

Priority 3:
  Line A label-free B320 manifold-aligned projector group A15-A19.

Priority 4:
  Line T T1-only visibility v3.

Priority 5:
  Line I matched-control actuator dictionary, only if T1 pass.

Priority 6:
  Line B P4 short-run, only if Line I pass.

Priority 7:
  Classic No-BSpline monitor only if new hypothesis exists.
```

一句话：

$$
\boxed{
\text{v12.19 不再追求“多跑一个候选”，}
\text{而是清理 B320 label-init confound、修复 Line C protocol、重建真正可部署的 loss-agnostic visibility。}
}
$$

