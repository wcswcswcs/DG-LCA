# DG-KAN v12.20：Fail-Closed Continue、Label-Free Base 重建与 Loss-Agnostic Functional Update 下一步计划

> 版本：v12.20 execution plan  
> 基于：v12.19 实验结果复盘、v12.18/v12.17/v12.16 代码审查与 Line C / Line T 结果  
> 核心修正：上一轮 gate 本身没有放水，但实验设置过于 **fail-fast**。之后 Codex 不允许在一个 official gate fail 后直接停工；必须进入预注册 continuation queue，继续完成机制诊断、上界测试、替代路径验证和失败归因。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no fake / proxy / CPU offload；functional direction 不使用 CE vector、label、permuted-label CE、validation/test/future outcome 或 dataset 名称；CE / ECE / CEp99 / Brier 只能作为 audit / non-harm 约束，不能作为 functional direction 的设计目标。

---

# 0. 一句话结论

v12.19 不是没进展。它做实了三件事：

```text
1. B320-current 的强表现确实依赖 label-informed trainprobe initialization；
2. 当前 label-free B320 patch 没有同时保住 task 与 Line C；
3. 清理掉 method identity、clone-probe drift、hard-release target 后，T1-only loss-agnostic visibility 完全看不到 joint hard release。
```

但 v12.19 的实验设置有一个严重问题：

$$
\boxed{
\text{official gate fail 后直接停止，导致 Codex 没有继续做机制层面的 continuation diagnostics。}
}
$$

这在科学上避免了 false promotion，但在研究推进上太慢。v12.20 必须改成：

```text
Fail closed for promotion;
Continue open for diagnosis.
```

也就是：

```text
official P3/P4 仍必须 gated；
但 Line A / Line T / Line I / Line B 的后续诊断不能因为某个 gate fail 就停。
```

---

# 1. 当前结果的独立分析

## 1.1 B320-current 仍是强 diagnostic anchor，但不能作为 label-free base claim

v12.19 的代码审计确认了 B320 construction path：

```text
run_v1218_b320_label_free_ablation.py::train_one
-> make_model
-> run_v124_multibasis_functional_dual.py::_make_model
-> SimpleFastTaskGeometryKAN.__init__
```

当 `variant` 含 `trainprobe` 且 `y_for_stats is not None` 时，代码根据 class mean direction 构造 `probe_dirs`，并写入：

```text
trainprobedirect -> direct_readout
trainprobeP      -> quad_proj
signalBroad      -> broad signal mixing
signalBlock      -> block-local signal mixing
```

因此，B320-current 的强表现不能解释为纯 label-free architecture capability。它现在的正确身份是：

$$
\boxed{
\text{label-informed supervised-initialization diagnostic anchor}
}
$$

它可以继续用于 functional diagnostic anchor，但不能写成 external-ready label-free PureKAN base。

## 1.2 Label-free B320 patch 没有关闭 R2

v12.19 跑了 A1、A5、A11、A12，以及新增的 A15-A19 label-free projector family。结果显示没有任何候选同时满足 task-side gate 和 Line C gate。

关键结果：

```text
A1-noYForStats:
  mean_delta_vs_A0 = -0.011284722222222222
  worst_delta_vs_A0 = -0.037109375
  max_AUC_time_ratio_vs_mlp = 0.8449234491133157
  LineC pass rate = 0.0
  official pass = 0

A5-orthogonalP-labelFree:
  mean_delta_vs_A0 = -0.011935763888888888
  worst_delta_vs_A0 = -0.0390625
  max_AUC_time_ratio_vs_mlp = 0.7512128806815016
  LineC pass rate = 0.0
  official pass = 0

A15-PCAOrthoMix-labelFree:
  mean_delta_vs_A0 = -0.028428819444444444
  worst_delta_vs_A0 = -0.083984375
  max_AUC_time_ratio_vs_mlp = 1.0296983958544903
  LineC pass rate = 0.0

A18-InputCovWhitenedOrthoP-labelFree:
  mean_delta_vs_A0 = -0.091796875
  worst_delta_vs_A0 = -0.13671875
  max_AUC_time_ratio_vs_mlp = 181.95322703972
  LineC pass rate = 0.0
```

A1 是当前 best label-free removal test，但它也没有过 `mean_delta_vs_A0 >= -0.005` 和 `worst_delta_vs_A0 >= -0.015`，Line C 更是全 fail。

这说明问题不是 “PCA / augmentation / random cotangent / cov-whiten / block-local projector 再调一下就行”。更深的问题是：

$$
\boxed{
\text{B320 的 label-informed trainprobe 提供了 class-separating direction；}
\text{当前 label-free projector family 没有替代这种 signal injection。}
}
$$

## 1.3 Group C diagnostic 支持：真实 label signal 不是任意 centroid 可替代

v12.19 的 A20-A22 是 diagnostic control，不进入 official label-free claim。

```text
A20-shuffledLabelTrainProbe-diagnostic:
  mean_delta_vs_A0 = -0.04796006944444445
  max_AUC_time_ratio_vs_mlp = 1.0808232233215522

A21-randomClassCentroid-diagnostic:
  mean_delta_vs_A0 = -0.04665798611111111
  max_AUC_time_ratio_vs_mlp = 0.9754105844224822

A22-permutedClassMeanP-diagnostic:
  mean_delta_vs_A0 = -0.06705729166666667
  max_AUC_time_ratio_vs_mlp = 1.2532740681763106
```

这支持一个重要判断：

$$
\boxed{
\text{B320 的 trainprobe 不是普通初始化增益，}
\text{而是强 supervised signal initialization。}
}
$$

所以后续不能把 B320-current 的表现当作 label-free architecture success。

## 1.4 Line C protocol autopsy 只证明本轮 calculator 一致，不证明历史 mismatch 全关

v12.19 同一 retrained A0 checkpoint、同一 batch hash、同一 v1252 primitive calculator 下得到：

```text
max_update_impl_metric_absdiff = 0.0
protocol_impl_consistency_pass = 1
```

这是好事，说明当前 calculator 内部一致。

但报告也明确写了：

```text
does_not_close_locked_vs_recovered_artifact_mismatch = 1
```

因此不能说 Line C 已完全干净。更准确地说：

```text
本轮内部实现一致；
历史 locked/recovered protocol mismatch 仍需要单独关闭。
```

## 1.5 Line T 失败不是 support concentration，也不是 false positive，而是 T1 feature 真看不见

v12.19 把 features 分成：

```text
T1 deployable precommit
T2 clone-probe diagnostic
T3 audit-only targets
BLOCKED forbidden feature families
```

清理后：

```text
method_role_flags -> BLOCKED
output_subspace_drift / pred_* / target_norm* -> T2 diagnostic only
hard release / label / CE target -> T3 audit-only
```

结果：

```text
T1_only_auc_joint_min = 0.0
T1_only_precision_joint_min = 0.0
T1_only_recall_joint_min = 0.0
T1_only_visibility_pass = 0

T1_plus_T2_diagnostic_auc_joint_min = 0.03773584905660377
diagnostic_pass_not_promotable = 0

support_concentrated = 0
control_false_positive_rate = 0.0
```

这很关键。失败不是因为 support 全在单一 seed/window，也不是 NoOp/Random 造成假阳性，而是：

$$
\boxed{
\text{当前 T1 deployable features 对 joint hard release 没有可见性。}
}
$$

这也说明 functional update 不能继续靠 scorer 小修。

## 1.6 这次 Codex 为什么“一下就停”？

Codex 不是伪造成功，也不是漏掉了 P4。它确实按上一版 gate 做了合法关闭：

```text
T1_only_visibility_pass = 0
line_i_open = 0
p4_open = 0
```

但问题是上一版计划本身太 fail-fast。它只规定：

```text
T1 fail -> P3/P4 closed。
```

却没有强制：

```text
T1 fail 后必须继续做哪些 mechanistic continuation tests。
```

这导致实验“合法停止”，但研究没有推进到下一层。因此 v12.20 的核心修正是：

$$
\boxed{
\text{promotion 继续 gated，diagnosis 不允许 gated stop。}
}
$$

---

# 2. 当前各线进展

## 2.1 Line A：B320 base / deconfounding

完成度估计：`70%`。

```text
已完成：
  current B320 label-init path 审计；
  A1/A5/A11/A12/A15-A19 label-free patch；
  A20-A22 label diagnostic controls；
  A0/A1 task 与 Line C 的主要失败表。

未完成：
  真正 label-free architecture redesign；
  locked/recovered Line C mismatch 完全关闭；
  label-free B320-like candidate 的 task + LineC 同时通过。
```

Line A 当前不应继续小修 `trainprobe` token，而应进入新的 label-free base architecture 设计。

## 2.2 Line C：audit metric / calculator

完成度估计：`80% as audit`，`35% as deployable construction`。

```text
已完成：
  CE / label / permuted label provenance 明确为 audit-only；
  v12.19 同 checkpoint calculator consistency pass；
  support / false-positive audit 已落盘。

未完成：
  locked/recovered mismatch closure；
  deployable signal/reservoir construction；
  label-free Line C surrogate that can replace CE-residual audit target。
```

Line C 现在可用于审计，但不能直接作为 functional direction source。

## 2.3 Line T：strict loss-agnostic visibility

完成度估计：`85% audit complete / 0% gate pass`。

```text
已完成：
  T1/T2/T3/BLOCKED feature tier cleanup；
  method identity 和 clone-probe drift 被移出 promotion feature；
  T1-only visibility v3 正式跑完。

失败：
  AUC_joint_min = 0.0；
  precision_at_k_joint_min = 0.0；
  recall_at_k_joint_min = 0.0。
```

这条线现在不是缺 scorer，而是缺更好的 T1 physical observable。

## 2.4 Line I / Line B：actuator / functional update

完成度估计：`0% official / 20% tooling`。

```text
Line I 未打开；
P4 未打开；
matched controls 未生成；
functional official success 仍为 0。
```

这不是坏事。坏事是如果在 T1 fail 时强开 P4。v12.20 要做的是：不允许 official P4，但必须继续做 non-promotional actuator capacity / upper-bound diagnostics。

## 2.5 Line D：classic no-BSpline portfolio

完成度估计：`monitor complete / active progress paused`。

v12.19 没有新 classic hypothesis，只做 monitor。这个决定可以接受，因为本轮主 blocker 在 Line A/T/B，而不是 classic family。

后续原则：

```text
有新 hypothesis 才跑；
不要为了“看起来继续”重复旧 family candidate。
```

---

# 3. 当前本质问题

现在真正的问题不是：

```text
B320 是否快；
FHQ kernel 是否可用；
Line C calculator 是否完全坏；
functional 是否只差 lambda。
```

真正的问题是：

$$
\boxed{
\text{我们还没有一个 label-free、precommit、loss-agnostic 的 value source，}
\text{能预测或驱动 audit-defined signal/reservoir/noise improvement。}
}
$$

更具体地说：

```text
B320-current 的性能来自 supervised trainprobe signal；
去掉 y_for_stats 后 task 和 Line C 都掉；
当前 label-free projector family 没有恢复这个 signal；
当前 T1 deployable features 对 hard joint release 完全不可见；
因此 functional update 没有合法 value source。
```

所以 v12.20 不能继续：

```text
继续 A15-A19 的 projector 小网格；
继续 T1 scorer / threshold 小修；
继续用 T2 clone-probe diagnostic promotion；
继续在 T1 fail 时直接停工；
继续把 B320-current 写成 label-free base。
```

---

# 4. v12.20 总体目标

v12.20 的总目标不是马上证明 functional success，而是关闭当前最大不确定性：

$$
\boxed{
\text{是否存在一个 label-free B320-like base + loss-agnostic precommit observable，}
\text{足以支持 functional update 进入 P3/P4？}
}
$$

这个目标拆成四个问题：

```text
Q1. B320-current 的 supervised trainprobe signal 能否被 label-free architecture 替代？
Q2. Line C 的 audit-only target 能否被一个 loss-agnostic surrogate 近似？
Q3. T1 physical observables 是否能在 precommit 阶段预测 hard joint release？
Q4. 即使 Q1/Q2/Q3 失败，是否能得到一个明确 no-go 证据，指导 functional update 改目标？
```

---

# 5. v12.20 关键实验思想：Fail-Closed Continue

## 5.1 两种 gate 必须分开

以后所有实验必须区分：

```text
Promotion gate:
  是否允许进入 official P3/P4 / claim。

Continuation gate:
  gate fail 后是否还必须继续诊断。
```

v12.20 规定：

```text
Promotion gate fail -> 不 promotion；
Continuation gate fail -> 继续执行预注册 fallback branches；
只有 fallback branches 全部完成，route 才允许 final stop。
```

## 5.2 Codex 不允许的行为

Codex 不允许：

```text
1. 只输出 route，然后停止；
2. P4 gated 后不做 upper-bound / capacity / no-go diagnostic；
3. 只说“按计划关闭”，不说明下一层机制原因；
4. 没有 continuation artifact 就写 completed；
5. 用 old artifact monitor 代替新 hypothesis；
6. 把 T2 / audit / response feature 写成 T1 promotion feature。
```

## 5.3 Codex 必须继续执行的最低诊断包

如果某条线 fail，必须执行：

```text
Line A fail:
  继续 A-UB upper-bound: label-free task upper-bound, LineC upper-bound, oracle projection diagnostic。

Line T fail:
  继续 T-UB observability upper-bound, feature family no-go, target-reset diagnostic。

Line I gated:
  继续 I-EXP exploratory actuator capacity with matched controls, not promotable。

Line B/P4 gated:
  继续 B-SIM offline response simulation / persistence diagnostic, not promotable。
```

---

# 6. Line R：代码审查与实验执行完整性

## 6.1 目标

防止 Codex 在 runner / wrapper 层给出“完成”但核心代码没有审查，或在 gate fail 后过早停止。

## 6.2 必须记录 artifact

```text
v1220_code_review_manifest.csv
v1220_continuation_execution_manifest.csv
v1220_stop_reason_audit.csv
v1220_diff_intent_table.csv
v1220_core_symbol_map.json
v1220_required_artifact_manifest.csv
```

## 6.3 必须审查代码路径

```text
R0 runner / route decision / continuation queue
R1 B320-current construction and y_for_stats wiring
R2 label-free architecture construction
R3 FHQ fused forward/backward/update path
R4 manual optimizer semantics
R5 Line C audit-only metric construction
R6 T1/T2/T3 feature tiering
R7 strict visibility scoring and leaveout
R8 actuator dictionary and matched controls
R9 P4 event schedule / short-run gating
R10 classic family monitor
```

每项必须写：

```text
actual_file_path
actual_line_start / actual_line_end
main_symbols
called_by / calls_into
artifact_fields_written
uses_label
uses_ce_vector
uses_validation_or_test_for_commit
uses_dataset_name_branch
is_promotion_feature
is_diagnostic_only
unknown_or_not_inspected
requires_manual_review
```

## 6.4 Gate

如果 CR0-CR9 任一项没有真实 file / symbol / line range：

```text
route = R0-CodeReviewSurfaceIncomplete
promotion_allowed = 0
```

如果 continuation queue 没有执行：

```text
route = R0-FailFastIncomplete
promotion_allowed = 0
```

---

# 7. Line A：Label-Free Base Architecture Redesign

## 7.1 目标

不再给 B320-current 的 `trainprobe` 做 token 小修，而是设计真正 label-free 的 B320-like base。

核心假设：

$$
\boxed{
\text{B320-current 需要的是 class-separating signal frame；}
\text{label-free replacement 必须提供稳定、多尺度、task-adaptable 的 signal frame。}
}
$$

## 7.2 Candidate families

### A-F1：Multi-frame label-free projection bank

构造多组固定或轻量可学习的 projection frame：

```text
PCA frame
Hadamard / SRHT frame
random orthogonal frame
augmentation-stable frame
local block frame
low-frequency image frame
```

不是选一个 frame，而是并联 frame bank：

$$
P = [P_{pca}, P_{srht}, P_{aug}, P_{local}, P_{freq}].
$$

再用 trainable but label-free gate $g$ 混合：

$$
q = xP \operatorname{diag}(g).
$$

约束：

```text
g 初始化均衡；
不使用 y_for_stats；
不使用 pseudo-label；
不使用 CE vector；
不使用 validation/test。
```

### A-F2：Self-conditioning residual projector

使用 activation covariance 与 random cotangent stability 构造 projector，但不直接复制 label-informed class direction。

记录：

```text
P_condition
P_frame_coherence
P_energy_distribution
quad_feature_std
direct_feature_std
branch_balance
```

### A-F3：B320-noY with train-time projector adaptation

允许 `P` 在训练中根据 unlabeled activation covariance 做低频更新，但不读 label / CE：

$$
P_{t+1} = \operatorname{orth}(P_t + \eta \Delta P_{cov}).
$$

这个不是 functional update official claim，只是 label-free base architecture repair。

### A-F4：Upper-bound diagnostics

为了避免再“fail 就停”，必须跑 upper-bound：

```text
A-UB1 label-informed B320-current 作为 upper bound；
A-UB2 shuffled-label trainprobe negative control；
A-UB3 random centroid negative control；
A-UB4 unsupervised clustering centroid diagnostic，不 promotion；
A-UB5 oracle class-mean with small label fraction diagnostic，不 promotion。
```

A-UB4/A-UB5 只用于判断是否存在 signal-frame upper bound，不进入 official base claim。

## 7.3 运行设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2 initially
train_size = 1024
val_size = 512
test_size = 512
epochs = 3 and 8
batch_size = 128
LineC batch = 64
sketch_dim = 24
```

如果某 candidate 同时接近 task gate 和 Line C gate，自动扩：

```text
seeds = 0..9
```

但如果没有 pass，不允许只停；必须执行 A-UB diagnostics。

## 7.4 必须记录字段

`v1220_label_free_base_candidates.csv`

```text
candidate_id
family
uses_y_for_stats
uses_label
uses_ce_vector
uses_pseudo_label
uses_validation_for_init
P_frame_count
P_frame_types
P_condition
P_coherence_mean
P_energy_entropy
quad_feature_std_mean
quad_feature_std_min
quad_feature_std_max
direct_readout_norm
quad_proj_norm
quad_readout_norm
branch_scale_norm
step_ratio_q90
memory_ratio_q90
mean_delta_vs_A0
worst_delta_vs_A0
AUC_step_ratio_vs_mlp
AUC_time_ratio_vs_mlp
ECE_delta_vs_mlp
LineC_nontearing_all_pass
NoiseSignalLeak_delta_vs_mlp
RealSignalReservoirRatio_delta_vs_mlp
official_label_free_candidate_pass
failure_reason
```

`v1220_label_free_upper_bound.csv`

```text
candidate_id
upper_bound_type
promotion_allowed
mean_delta_vs_A0
LineC_pass_rate
what_this_bound_tests
```

## 7.5 Official pass

A label-free base candidate passes only if:

$$
\text{uses\_label}=0,
$$

$$
\text{uses\_y\_for\_stats}=0,
$$

$$
\Delta Acc_{mean\ vs\ A0}\ge -0.005,
$$

$$
\Delta Acc_{worst\ vs\ A0}\ge -0.015,
$$

$$
AUC_{time}/AUC_{time,MLP}\le 1.00,
$$

$$
LineC_{nontearing\ all}=1.
$$

## 7.6 不满足条件时 Codex 先尝试

```text
如果 task fail 但 Line C 接近：
  增加 frame count，而不是加标签；
  调整 frame balance / gate entropy；
  不按数据集调参。

如果 task pass 但 Line C fail：
  检查 NoiseSignalLeak vs ReservoirRatio 哪个坏；
  若 NoiseSignalLeak 高，减少 high-frequency / random frame energy；
  若 ReservoirRatio 高，增加 persistent low-rank signal frame；
  继续 A-UB diagnostics。

如果所有 label-free frame fail：
  输出 R2-LabelFreeSignalFrameMissing；
  不再继续 B320 patch 小网格；
  进入 new architecture family design。
```

## 7.7 可视化

```text
fig_A_label_free_task_vs_linec_pareto.svg
fig_A_projection_frame_energy.svg
fig_A_P_condition_vs_task.svg
fig_A_noise_reservoir_by_candidate.svg
fig_A_upper_bound_gap.svg
fig_A_AUC_time_matrix.svg
```

---

# 8. Line C：Audit Metric 与 Deployable Geometry 分离

## 8.1 目标

Line C 要继续用 CE / label 构造 audit metric，但这些不能进入 direction source。

需要显式区分：

```text
C-Audit:
  可以用 label / CE / permuted label；只用于评价。

C-Deployable:
  不用 label / CE；可作为 functional observable。

C-Forbidden:
  validation/test/future/dataset name/post-hoc selected threshold。
```

## 8.2 新增任务：Line C target reset diagnostic

当前 T1 看不到 CE-residual-defined hard release。v12.20 必须判断：

```text
是 T1 feature 不够；
还是 CE-residual hard release 本身不适合作为 loss-agnostic target。
```

因此新增两个 target：

### C-T0：Current audit target

```text
NoiseSignalLeak / RealSignalReservoirRatio using CE residual and permuted label.
```

只保留为 audit。

### C-T1：Loss-agnostic geometric target

只基于 unlabeled quantities：

```text
train-probe coupling stability
logit covariance spectral stability
random-cotangent Jacobian spectrum
activation occupancy / entropy
primitive-role projector stability
augmentation consistency drift
kernel-sketch condition
```

定义：

$$
G_{LA}=w_1\Delta CouplingStability
-w_2\Delta ProjectorInstability
-w_3\Delta LogitDrift
-w_4\Delta OccupancyCollapse
-w_5\Delta TailDriftProxy.
$$

C-T1 不能替代 final audit，但可以作为 functional value source。

## 8.3 必须记录字段

`v1220_linec_target_reset.csv`

```text
candidate_id
method
uses_label_for_target
uses_ce_for_target
target_type
CouplingR2_delta
CouplingStability_delta
ProjectorInstability_delta
LogitCovarianceDrift_delta
RandomCotangentSpectrum_delta
OccupancyEntropy_delta
TailDriftProxy_delta
NoiseSignalLeak_delta_audit
RealSignalReservoirRatio_delta_audit
corr_with_audit_noise_release
corr_with_audit_reservoir_release
promotion_allowed
```

## 8.4 Gate

C-T1 can be used as a functional value source only if:

$$
\operatorname{Spearman}(G_{LA}, \Delta NoiseSignalLeak_{audit}) \le -0.30
$$

and:

$$
\operatorname{Spearman}(G_{LA}, \Delta RealSignalReservoirRatio_{audit}) \le -0.30
$$

or, if these correlations remain impossible after the full continuation budget:

```text
route = R2-AuditTargetNotVisibleFromLossAgnosticGeometry
next_action = use audit metrics only as non-harm constraints, not value source。
```

## 8.5 不满足条件时 Codex 先尝试

```text
如果 C-T1 与 audit target 无相关：
  分别按 dataset/seed/window 诊断，但不做 dataset-specific rule；
  检查是否某个 audit target 本身方差太大；
  检查 hard threshold 是否使 positive support 太稀疏；
  运行 soft-release continuous target，但只做 audit，不降低 hard gate。

如果 C-T1 只预测 CouplingR2：
  不允许 promotion；
  增加 occupancy / spectrum / tail proxy。
```

---

# 9. Line T：True Loss-Agnostic Visibility v4

## 9.1 目标

Line T 不再只问：

```text
T1 能不能预测 old hard joint release？
```

而是并行问：

```text
T1 是否能预测 audit hard release；
T1 是否能预测 loss-agnostic geometric target；
T1 是否存在 no-go upper bound。
```

## 9.2 Feature tiers

### T1A：Pure precommit state features

```text
logit covariance eigenvalues
logit entropy distribution
activation occupancy entropy
primitive role energy
projector condition / coherence
random cotangent Jacobian spectrum
augmentation consistency without labels
kernel sketch condition
```

### T1B：Optimizer-observable but loss-agnostic API features

Functional update may observe the base optimizer update as an abstract vector $d_{task}$, without reading the loss or labels. This makes the method loss-agnostic across any optimizer/loss interface.

```text
role-wise norm of d_task
cosine between d_task and random cotangent VJPs
projected update spectrum
update-induced logit drift from first-order JVP sketch
AdamW moment norms by primitive role
```

Constraint:

```text
read optimizer update vector = allowed;
read CE vector / label / per-example CE = forbidden.
```

### T2：Clone-probe diagnostic only

```text
actual response after cloned candidate
output_subspace_drift
predicted release from candidate response
```

T2 cannot be used for promotion.

### T3：Audit target only

```text
NoiseSignalLeak
RealSignalReservoirRatio
hard release labels
CEp99 / ECE / Brier audit
```

## 9.3 Required experiments

```text
T-v4a: T1A only
T-v4b: T1B only
T-v4c: T1A + T1B
T-v4d: T1A/T1B leave-dataset-out
T-v4e: T1A/T1B leave-seed-out
T-v4f: T1A/T1B leave-window-out
T-v4g: soft release regression
T-v4h: hard release ranking
T-v4i: no-go upper-bound using T2 diagnostic
```

## 9.4 Must record

`v1220_visibility_scores.csv`

```text
feature_tier
feature_set
rows
feature_columns
long_feature_rows
AUC_noise_min
AUC_reservoir_min
AUC_joint_min
precision_at_k_joint_min
recall_at_k_joint_min
soft_target_r2_noise_min
soft_target_r2_reservoir_min
leave_dataset_out_min
leave_seed_out_min
leave_window_out_min
support_concentrated
control_false_positive_rate
visibility_pass
not_promotable_reason
```

`v1220_visibility_no_go_upper_bound.csv`

```text
upper_bound_type
feature_set
uses_T2_response
AUC_joint_min
precision_at_k_joint_min
recall_at_k_joint_min
interpretation
```

## 9.5 Promotion gate

T1 visibility pass requires:

$$
AUC_{joint,min}\ge 0.65,
$$

$$
precision@k_{joint,min}\ge 0.20,
$$

$$
recall@k_{joint,min}\ge 0.20,
$$

and no concentration:

$$
support\_concentrated=0.
$$

## 9.6 Continuation rule

If T1 visibility fails, Codex must still run:

```text
1. T1A/T1B ablation;
2. soft-release regression;
3. T2 upper-bound diagnostic;
4. target reset correlation with C-T1;
5. report whether failure is feature weakness or target unobservability.
```

Only after these five are complete may route stop.

---

# 10. Line I：Actuator Capacity and Matched Controls

## 10.1 目标

Line I official remains gated by T1. But if T1 fails, v12.20 still runs a **non-promotional actuator capacity diagnostic**.

This avoids Codex stopping with:

```text
line_i_open = 0
```

without knowing whether actuator is a future blocker.

## 10.2 Two modes

### I-Official

Runs only if T1 passes. Can lead to P4.

### I-Exploratory

Runs even if T1 fails. Cannot lead to P4. Its purpose is to measure actuator capacity and matched control baseline.

## 10.3 Actuator families

```text
I0 NoOp
I1 RandomMatchedNorm
I2 AdamWParallel
I3 SNR-only
I4 RoleEnergyMatchedRandom
I5 DirectReadoutActuator
I6 QuadProjActuator
I7 BranchScaleActuator
I8 MixedPrimitiveActuator
I9 Orthogonal-to-AdamW actuator
```

## 10.4 Must record

`v1220_actuator_capacity.csv`

```text
candidate_id
mode_official_or_exploratory
actuator_id
matched_control_id
dataset
seed
window
functional_norm_ratio
logit_max_abs_drift
sketch_delta_fro
projector_angle_deg
CouplingR2_delta
C_T1_score_delta
NoiseSignalLeak_delta_audit
RealSignalReservoirRatio_delta_audit
control_gap_vs_best
safe_movement_pass
release_audit_pass
promotion_allowed
```

## 10.5 Gate

Official actuator pass requires:

$$
sketch\_delta\_fro\ge 0.01,
$$

$$
projector\_angle\ge 1^\circ,
$$

$$
logit\_max\_abs\_drift\le 0.05,
$$

$$
control\_gap\ge 0.005.
$$

Exploratory pass only records capacity; it cannot open P4.

## 10.6 Failure actions

```text
If all actuators weak:
  increase primitive-level actuator basis, not scorer.

If actuators move but controls match:
  enforce AdamW-orthogonal residual and role-energy matched controls.

If actuators improve C-T1 but not audit target:
  keep C-T1 as possible loss-agnostic geometry target, audit remains non-harm only.
```

---

# 11. Line B：Functional Candidate Construction

## 11.1 目标

Functional update must be loss-agnostic. It cannot target CE. It can target deployable geometry.

Candidate form:

$$
\theta_{t+1}=\theta_t+\Delta\theta_{task}+\lambda\Delta\theta_{func}.
$$

Where:

```text
Delta theta_task:
  produced by base optimizer, may come from any loss but is treated as opaque optimizer update.

Delta theta_func:
  constructed only from T1A/T1B deployable geometry and primitive state.
```

## 11.2 Candidate families

```text
B1 C-T1 geometric stabilizer
B2 AdamW-orthogonal projector stabilizer
B3 signal-frame occupancy rebalance
B4 primitive role-energy transport
B5 low-frequency event geometry maintenance
B6 T1B update-spectrum preconditioner
```

No CE vector, no label, no dataset name.

## 11.3 P3 one-step gate

P3 candidate must satisfy:

$$
\Delta C\_T1\_score \ge 0.02,
$$

$$
\Delta NoiseSignalLeak_{audit} \le 0.005,
$$

$$
\Delta RealSignalReservoirRatio_{audit} \le 0.005,
$$

where audit metrics are **non-harm constraints**, not direction targets.

If the old hard audit target is still used as promotion target, also require:

$$
\Delta NoiseSignalLeak_{audit} \le -0.01,
$$

$$
\Delta RealSignalReservoirRatio_{audit} \le -0.01.
$$

But v12.20 allows a separate route:

```text
R3-LossAgnosticGeometryTargetPassAuditNonHarm
```

if C-T1 is strong and audit non-harm holds, even if CE-residual release does not improve by -0.01. This route is **not** final functional success; it only opens a low-frequency P4 exploratory run.

## 11.4 P4 short-run gate

P4 only opens if:

```text
Line R pass;
Line A has either label-free base pass OR current B320 diagnostic-anchor mode explicitly selected;
Line C target reset complete;
Line T visibility pass OR C-T1 target pass;
Line I official actuator pass.
```

P4 runs:

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 1024
val_size = 512
test_size = 512
epochs = 3 / 8
functional_event_interval = 24 / 48 / 96
```

## 11.5 Must record

`v1220_functional_p3.csv`

```text
candidate_id
uses_label
uses_ce_vector
uses_dataset_name_branch
uses_validation_for_commit
feature_tier_used
C_T1_score_delta
CouplingR2_delta
NoiseSignalLeak_delta_audit
RealSignalReservoirRatio_delta_audit
CEp99_delta
ECE_delta
Brier_delta
holdout_loss_ratio_CE_audit
holdout_loss_ratio_Brier_audit
control_gap_vs_best
promotion_allowed
failure_reason
```

`v1220_functional_p4_short.csv`

```text
candidate_id
control_id
dataset
seed
epoch
step
functional_event_count
accepted_event_count
rejected_event_count
event_interval
amortized_overhead
val_loss_auc_step_ratio
val_loss_auc_time_ratio
mean_delta_vs_base
worst_delta_vs_base
ECE_delta_vs_base
CEp99_delta_vs_base
C_T1_score_trajectory
NoiseSignalLeak_audit_trajectory
RealSignalReservoirRatio_audit_trajectory
control_gap_vs_best
P4_pass
failure_reason
```

## 11.6 P4 pass

Official functional success still requires:

$$
AUC_{time,func}\le AUC_{time,base},
$$

$$
Acc_{func}\ge Acc_{base}-0.003,
$$

$$
ECE_{func}\le ECE_{base}+0.01,
$$

$$
amortized\_overhead\le 1.05,
$$

and either:

$$
NoiseSignalLeak_{func}\le NoiseSignalLeak_{base}-0.01
$$

and:

$$
RealSignalReservoirRatio_{func}\le RealSignalReservoirRatio_{base}-0.01,
$$

or v12.20 exploratory route:

```text
C_T1_geometry improves;
audit CE/noise/reservoir non-harm;
controls beaten;
requires future v12.21 confirm before any scientific claim。
```

---

# 12. Line D：Classic No-BSpline Portfolio

## 12.1 目标

Line D 不抢 v12.20 主预算，但不能消失。

当前 status:

```text
Rational: TaskBlocked / GeometryBlocked
Chebyshev: TaskBlocked
Wavelet: TaskBlocked
RBF: ExpressionBlocked
Fourier: ExpressionBlocked
BSpline: Frozen / RejectedForThisVersion
```

## 12.2 v12.20 只允许新 hypothesis

```text
Rational:
  only if new geometry hypothesis targets coupling_collapse without CE tuning.

Chebyshev:
  only if task trajectory/AUC hypothesis, not degree increase only.

Wavelet:
  only if task-stable local support / scale diversity hypothesis.

RBF/FastKAN:
  only if A4 expression repair preserves L3 efficiency.

Fourier:
  only if low-frequency expression repair without high-frequency noise leak.
```

## 12.3 Must record

```text
v1220_classic_family_status.csv
v1220_classic_family_new_hypothesis.csv
v1220_classic_family_linec.csv
```

No new hypothesis means:

```text
Line D monitor-only
```

not failure and not success.

---

# 13. Parallel execution schedule

To avoid slow serial progress, v12.20 must run in parallel batches.

## Batch 0：P0 / code / continuation queue

```text
Run Line R.
Verify continuation queue.
If continuation queue missing, stop with R0-FailFastIncomplete.
```

## Batch 1：Line A label-free architecture and Line C target reset

Parallel jobs:

```text
A-F1 multi-frame bank
A-F2 self-conditioning residual projector
A-F3 train-time unlabeled projector adaptation
C-T0 audit target reproduction
C-T1 loss-agnostic target construction
```

## Batch 2：Line T visibility v4

Parallel jobs:

```text
T1A only
T1B only
T1A + T1B
soft-release regression
hard-release ranking
T2 upper-bound diagnostic
```

## Batch 3：Line I exploratory capacity

Runs even if T1 fails, but not promotable.

```text
I exploratory actuator capacity
matched controls
AdamW-orthogonal response
role-energy matched random controls
```

## Batch 4：Line B functional P3 / P4

```text
Only if gates allow official or exploratory route.
Otherwise output functional_not_opened_with_completed_continuation.
```

## Batch 5：Classic No-BSpline monitor or new hypothesis

Runs if new hypothesis is implemented.

---

# 14. Route definitions

```text
R0-CodeReviewSurfaceIncomplete:
  missing file/symbol/line references.

R0-FailFastIncomplete:
  official gate fail occurred but continuation diagnostics were not executed.

R1-B320CurrentLabelInitAnchorOnly:
  current B320 remains label-informed diagnostic anchor; no label-free candidate.

R2-LabelFreeSignalFrameMissing:
  all label-free architecture attempts fail task+LineC; upper-bound diagnostics completed.

R2-AuditTargetNotVisibleFromLossAgnosticGeometry:
  CE-residual hard release cannot be predicted by T1 even after upper-bound diagnostics.

R3-LossAgnosticGeometryTargetFound:
  C-T1 target is deployable and correlated with audit non-harm; not final functional success.

R4-FunctionalP3Survivor:
  P3 one-step survivor beats controls and passes non-harm.

R5-P4ShortRunCandidate:
  P4 opens and records short-run.

R6-FunctionalOfficialSuccess:
  P4 short-run beats controls, non-harm, and efficiency gate.
```

---

# 15. 判断成功的标准

## 15.1 v12.20 minimum success

v12.20 minimum success is not functional success. It is one of:

```text
Success A:
  A label-free B320-like base passes task + Line C.

Success B:
  A deployable loss-agnostic C-T1 target is found and correlates with audit non-harm.

Success C:
  T1A/T1B visibility passes hard joint release gate.

Success D:
  Strong no-go result: hard CE-residual release is not visible from legal T1 even with upper-bound diagnostics;
  then target reset is justified.
```

## 15.2 What is not success

```text
T2 diagnostic AUC improving;
method identity feature working;
clone-probe response working;
one dataset/seed/window pass;
CouplingR2-only improvement;
label-informed B320-current remaining strong;
P4 gated-not-run without continuation diagnostics.
```

---

# 16. Required visualizations

```text
fig_v1220_route_waterfall.svg
fig_v1220_failclosed_continue_matrix.svg
fig_v1220_label_free_task_linec_pareto.svg
fig_v1220_projection_frame_diagnostics.svg
fig_v1220_linec_target_reset_correlation.svg
fig_v1220_visibility_T1A_T1B_ablation.svg
fig_v1220_visibility_leaveout_heatmap.svg
fig_v1220_upper_bound_gap.svg
fig_v1220_actuator_capacity_vs_controls.svg
fig_v1220_functional_p3_control_gap.svg
fig_v1220_p4_if_open_loss_vs_step.svg
fig_v1220_p4_if_open_linec_trajectory.svg
fig_v1220_classic_status_dashboard.svg
```

---

# 17. 给 Codex 的执行纪律

Codex 必须在复盘中写出：

```text
1. 这轮哪些 gate fail；
2. fail 后哪些 continuation diagnostics 被执行；
3. 哪些 continuation diagnostics 没执行，为什么；
4. 每条线是否可以停；
5. 如果停，是否是 R0-FailFastIncomplete；
6. 核心代码路径与 line references；
7. 是否使用 label / CE / validation / dataset name；
8. 哪些结果可 promotion，哪些只能 diagnostic；
9. 下一轮具体不满足条件时应先尝试什么。
```

Codex 不能只写：

```text
T1 failed, so P4 closed。
```

必须写：

```text
T1 failed；
T1A/T1B ablation completed；
soft-release regression completed；
T2 upper bound completed；
C-T1 target reset completed；
actuator exploratory capacity completed；
therefore route = ...
```

---

# 18. 最终判断

v12.19 的失败不是“没有继续跑 P4”。P4 关闭是对的。真正不合理的是：计划没有要求 Codex 在 P4 关闭后继续执行机制层面的 continuation diagnostics。

v12.20 的核心是：

$$
\boxed{
\text{不降低 scientific gate，但扩大 diagnostic continuation。}
}
$$

项目现在不应该继续小修 B320-current，也不应该继续 scorer 小网格。下一步应该围绕三个核心问题并行推进：

```text
1. label-free signal frame 是否能替代 supervised trainprobe；
2. CE-residual hard release 是否能由 legal T1 features 可见；
3. 如果不可见，是否应该把 functional value source 重置为 loss-agnostic geometry target，只把 CE/noise/reservoir 保留为 audit non-harm。
```

这才是从“跑不动 / 不继续”变成“系统性推进”的下一步。
