# DG-KAN v12.24：S4-to-S5 Functional Bridge、Label-Free Signal Frame、Classic No-BSpline 并行推进计划

> 版本：v12.24 execution plan  
> 基于：v12.23 `FailClosedExploreOpen2 FunctionalRebuild` 完整复盘、v12.22 code packet 静态检查、本轮长期补充 continuation 结果。  
> 目标：不降低 gate，不按数据集调参，不针对 CE 设计 functional direction；但也不允许 Codex 在单层 fallback 后停止。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no fake / proxy / CPU offload；official functional direction 不使用 CE vector、label、permuted-label CE、validation/test/future outcome 或 dataset 名称。CE / NLL / ECE / CEp99 / Brier 只能作为审计与坏化约束。

---

# 0. 一句话判断

v12.23 不是没有进展。它已经从原来的

```text
R4-FunctionalMechanismNoGoAfterAllFallbacks
```

推进到至少出现过

```text
S4-FunctionalP3Opened
```

并且在 `I24/I25 direct-logit compensated` family 中看到了真实的 P3 几何信号和 audit-only P4 task gain。但 v12.23 仍然不是 S5 official functional success，因为最强信号都卡在三件事上：

```text
1. provenance 不干净：query-reference compensation 不能作为 official precommit train-stream source；
2. robustness 不够：train-shuffle seed、LineC sketch seed、multi-sketch all-gate 不稳定；
3. P3/P4 解耦：P3 pass 的点常常 P4 fail，P4 pass 的点又不满足 P3 geometry gate。
```

所以 v12.24 的核心不再是继续盲目扩大 I24/I25 budget grid，而是：

$$
\boxed{
\text{把 audit-only P3/P4 信号转成 precommit-safe、loss-agnostic、control-resistant、seed-robust 的 official functional path。}
}
$$

---

# 1. v12.23 结果独立分析

## 1.1 进展不是 functional success，而是 functional 信号第一次进入 S4

v12.23 最初完整 fallback 后仍是 no-go：Line A、Line T/C、Line I、Line B、Line D、Line R 都有 artifact，required artifacts 不缺失，但没有 promotion success。之后继续补跑 Level-3 label-free repair、T2 upper-bound diagnostic、Line I controls-win fallback、非 I18 drift bisection、全线 continuation、dense actuator scan、Line D hardening、I24/I25 targeted repair、P4 compensation modes、P4 official row scan、blend、P4 trainable-role scan、trajectory verifier、robustness replay、train-stream ensemble compensation、multi-sketch aggregation、precommit proxy 等。

这些补跑不是白做。它们把 functional 线推进出一个更精确的状态：

```text
Functional P3 can be opened by I24/I25 direct-logit compensated actuator.
P4 task gain exists under quad_only post-P3 update in audit-only settings.
But official S5 is still blocked by provenance and robustness.
```

这比早期“完全没有 actuator survivor”更进一步。

## 1.2 Label-free base 不是完全学不了任务，但 LineC non-tearing 仍为 0

v12.23 的 Line A 从 scout 到 hardening，再到 A51-A65 继续推进。最有价值的现象是：

```text
A51 / A1 / A63 类 label-free candidate 的 task mean 可以接近甚至超过 A0；
但 LineC pass rate 始终为 0。
```

例如后续 e12 与 rmsq/boundq/direct/readout 修复显示：

```text
A51 best mean delta vs A0 ≈ +0.0031
A63/A65 在 Fashion-MNIST 上能改善 worst/AUC
但所有这些 candidate 的 LineC pass rate = 0.0
```

这说明 label-free 方向的 blocker 不再只是 task accuracy。更准确地说：

$$
\boxed{
\text{label-free base 能学任务，但学出来的 signal/reservoir/noise 几何仍不健康。}
}
$$

因此下一步不能继续只调 readout、RMSQ、direct scale、residual strength。它们已经证明对 LineC non-tearing 没有本质帮助。Label-free base 的下一步必须围绕 **LineC-aware signal frame**，而不是更强 task head。

## 1.3 T2/T3 说明“信号不是不存在”，但它不是 precommit source

v12.23 的 T1A/T1B 仍然不过：

```text
T1A_auc_joint ≈ 0.41
T1B_auc_joint ≈ 0.50
```

这说明静态 precommit / optimizer-observable features 目前不能稳定预测 hard joint release。

但 T2/T3 很有信息：

```text
T2_clone_probe_auc_joint ≈ 0.74 - 0.78
T2 precision / recall ≈ 0.32
T3 support stress AUC min ≈ 0.66 - 0.74+
```

这不是 promotion，因为 T2/T3 使用 completed response / LineC observables，`promotion_allowed=0`。但它证明：

$$
\boxed{
\text{response-level upper-bound signal 存在；真正缺的是把它提前到 precommit / train-stream 可部署阶段。}
}
$$

所以 v12.24 不应再说“loss-agnostic value source 完全不存在”，而应更精确地问：

```text
怎样把 T2/T3 response-level signal 蒸馏成 T1B-style online micro-probe / precommit feature？
```

## 1.4 Line I 的真实 blocker：安全方向和 release 方向分离

v12.23 修复了一个关键 bug：Line I 的 base/trial LineC delta 之前用了不同 sketch seed，导致 NoOp 也会产生非零 delta。修复后：

```text
NoOp max abs delta = 0.0
actuator_release_audit_rows 从 309 降到 48
```

之后 dense scan 结果很清楚：

```text
I11/I12/I14: drift-safe / role-safe，但 release rows = 0，control gap 为负；
I13/I15/I17/I18: 有 release / high gap，但 drift 远超 0.05，role-safe = 0。
```

这把 blocker 精确成：

$$
\boxed{
\text{当前 actuator family 中，drift-safe movement 和 release/control-gap 不在同一个方向上。}
}
$$

后续 I24/I25 用 direct-logit compensation 创造了一个突破：能把某些高 release 方向压进安全 drift，并打开 P3。但这个补偿目前依赖 query-reference，所以仍然是 audit-only / diagnostic-only。

## 1.5 P3/P4 解耦是本轮最重要的科学发现

v12.23 找到了两个相互矛盾的点：

```text
KMNIST seed1 / I24 / norm0.027 / sign+:
  P3 pass, P4 fail。

KMNIST seed1 / I24 / norm0.03 / sign-:
  P4 pass, P3 fail，因为 CouplingR2_delta negative。
```

之后简单 blend 也没解决：25 个 p3_budget / shadow_budget 组合中，有 P3 pass，但没有 P4 pass，更没有 both pass。

这说明：

$$
\boxed{
\text{当前 P3 geometry gate 与短训 task-useful direction 没有充分对齐。}
}
$$

不是“P3 没有信号”，也不是“P4 没有 task gain”，而是 **P3 和 P4 的成功区域不重合**。

## 1.6 quad_only P4 audit pass 是真实信号，但不能 official

后续 quad_only post-P3 online update 在 e8/e12 出现了 audit pass。主线 exact replay 中甚至有一行通过 strict trajectory gate：

```text
source_acc > noop_acc >? control_acc
source_NLL <= noop_NLL
source_CEp99 <= noop_CEp99
step_time_ratio < 1
AUC_error_time_ratio < 1
CouplingR2 source > noop
NoiseSignalLeak source < noop
RealSignalReservoirRatio source < noop
strict_trajectory_gate_pass = 1
promotion_allowed = 0
```

这证明：

```text
P4 task gain 不是幻觉；
P3 source 在合适 role policy 下可以产生 NoOp-resistant task/geometry gain。
```

但它仍不能 official，因为：

```text
1. source 仍来自 I24 query-reference compensation；
2. train-shuffle replay 0/3；
3. train-stream-only I26/I27 不能复制 gain；
4. train-stream ensemble compensation 2/4/8 也不能复制；
5. multi-sketch all-pass 不成立；
6. precommit proxy 只能给 majority 或不稳定结果。
```

所以正确结论是：

$$
\boxed{
\text{有 audit-only functional task gain，但还没有 official precommit-safe functional update。}
}
$$

## 1.7 Classic No-BSpline 支线：Rational/Fourier 有继续价值，但不是本轮 success

v12.23 Line D 从 smoke 补到 Rational/Fourier hardening：

```text
Rational 在 MNIST/Fashion-MNIST task 上强于 Fourier；
Fourier 有少数 LineC CouplingR2 较高的行；
Chebyshev/Wavelet/RBF 仍较弱；
B-spline 冻结。
```

这意味着 Line D 不再是“没执行”。但它仍只是 hardening signal，不是 official FamilyPass。

下一步 Line D 应作为独立并行线推进 Rational/Fourier，不应再占用 functional 主线的 P3/P4 预算。

## 1.8 代码审查边界

我本地解压了上传的 `v1222_code_review_packet.zip`。这个包主要包含 v12.22 runner、v12.22 artifacts 和 v12.22 代码，不包含 v12.23 后续新增的完整 I24/I25/P4/multisketch/precommit-proxy 脚本。也就是说，当前 v12.23 的后续代码审查只能依据复盘中记录的 package zip sha 与脚本清单，不能视为我已经本地审查了完整 v12.23 最终包。

v12.24 必须要求 Codex 打包最终 v12.23 / v12.24 transitive code packet，至少包含：

```text
run_v1223_failclosed_explore_open2_functional_rebuild.py
run_v1223_p4_compensation_modes.py
run_v1223_p4_official_row_scan.py
run_v1223_shadowp4_coupling_preserving_blend.py
run_v1223_p4_trainable_role_scan.py
run_v1223_p4_trajectory_gate_verifier.py
run_v1223_p4_reservoir_veto_checkpoint_scan.py
run_v1223_p4_train_ensemble_compensation_scan.py
run_v1223_p4_multisketch_aggregate_gate.py
run_v1223_p4_precommit_proxy_checkpoint.py
run_v1223_line_d_hardening.py
summarize_v1223_line_d_hardening.py
fc_purekan_primitives.py
fused_hinge_quadratic.py
```

缺少这些，route 必须降为：

```text
R0-CodePacketIncomplete
```

---

# 2. v12.24 总目标

v12.24 不是继续做“多跑一点”的 continuation，而是针对 v12.23 已定位的矛盾做结构性闭环。总目标分三层。

## 2.1 Minimum success

v12.24 至少要完成：

```text
1. 用完整 final code packet 审计 v12.23/v12.24 的核心路径；
2. 把 I24/I25 的 query-reference compensation 替换为 train-stream / precommit-safe compensation family；
3. 对 P3/P4 解耦做 policy-aware analysis；
4. 对 Line D Rational/Fourier 做正式 hardening，不再只 smoke；
5. 若失败，必须输出 executable next-generation fallback，而不是只写 no-go。
```

## 2.2 Exploration success

至少满足其中之一：

```text
S4a: precommit-safe / train-stream compensation 打开 P3；
S4b: P3-pass source 在 strict P4 short-run 中赢 NoOp 和 matched control，并通过 multi-sketch majority；
S4c: Rational/Fourier 在 Line D hardening 中通过 task + LineC exploration gate；
S4d: 找到 T1B online micro-probe，使 T2/T3 upper-bound signal 能在 commit 前被预测。
```

## 2.3 Official success

functional official success 必须满足：

$$
\Delta CouplingR^2 \ge 0.02,
$$

$$
\Delta NoiseSignalLeak \le -0.01,
$$

$$
\Delta RealSignalReservoirRatio \le -0.01,
$$

$$
control\_gap \ge 0.005,
$$

并且 P4 short-run 同时满足：

$$
Acc_{source} \ge Acc_{noop} + 0.005,
$$

$$
Acc_{source} \ge Acc_{control} + 0.005,
$$

$$
NLL_{source} \le NLL_{noop},
$$

$$
CEp99_{source} \le CEp99_{noop} + 0.05,
$$

$$
AUCtime_{source} \le AUCtime_{noop},
$$

$$
step\_time\_ratio \le 1.05,
$$

并通过 robustness：

```text
multi-sketch majority pass >= 4/5；
train-shuffle pass >= 2/3；
不能集中在单一 dataset/seed/window；
source provenance precommit-safe；
loss_agnostic_direction = 1。
```

---

# 3. v12.24 执行总原则：Fail-Closed, Explore-Open, Depth-2 Mandatory

v12.23 的执行比 v12.21 好很多，但还是在多次用户推动后才继续。v12.24 必须内置强制探索深度。

## 3.1 不允许的停止条件

以下情况不允许 final stop：

```text
1. 写了 no-go boundary 但没有执行下一层 fallback；
2. 写了 next hypothesis generator 但没有执行 top hypotheses；
3. P3 fail 后没有执行 P3/P4 decoupling diagnostic；
4. P4 fail 后没有执行 compensation provenance repair；
5. Line D 只写 NotExecuted 或 smoke 后没有 hardening；
6. Line I 有 safe/release/control 三角冲突但没有做 role-specific repair；
7. 代码包不包含本轮新增脚本。
```

## 3.2 合法停止条件

`final_stop_allowed=1` 只能来自：

```text
1. official_success_reached = 1；
2. hard_budget_exhausted = 1 且 fallback_depth >= 2；
3. user_stop_flag = 1；
4. route = R-MechanismNoGo 且所有 fallback levels、Line D hardening、code packet 均完整。
```

## 3.3 Codex 必须执行的探索深度

每条主线必须至少有两层 fallback：

```text
Level 0: official plan candidate；
Level 1: failure-specific repair；
Level 2: mechanism-reset candidate。
```

如果 Level 1 fail，不能直接停，必须执行 Level 2。若 Level 2 仍 fail，才允许写 no-go。

---

# 4. Line R：代码与 provenance 审计

## 4.1 目标

确认 v12.23/v12.24 的 functional signal 不是由代码路径、query leakage、metric seed mismatch、control scope mismatch、artifact reuse 或 package 缺失造成。

## 4.2 必须审查的代码路径

Codex 必须在复盘中逐项写出真实文件、class/function、line range、调用链、shape、artifact 字段和 gate 关系。至少包括：

```text
R0 route decision / final stop / fallback execution
R1 B320-current / label-free model construction
R2 I24/I25/I26/I27 direct compensation actuator
R3 direct-logit compensation reference construction
R4 quad_only post-P3 training policy
R5 P4 trainable-role scan
R6 P4 trajectory verifier
R7 multi-sketch aggregation gate
R8 precommit proxy checkpoint selector
R9 LineC metric seed handling
R10 matched control scope
R11 Line D Rational/Fourier hardening runner
R12 code packet generation
```

## 4.3 审计字段

新增：`v1224_core_code_review_manifest.csv`

```text
review_id
file_path
symbol_name
line_start
line_end
called_by
calls_into
mathematical_object
expected_tensor_shapes
uses_label
uses_ce_vector
uses_query_batch
uses_train_batch
uses_validation_or_test
uses_future_outcome
uses_dataset_name_branch
precommit_available
promotion_allowed
loss_agnostic_direction
matched_control_scope
metric_seed_shared
manual_review_required
blocker_if_wrong
```

## 4.4 gate

如果以下任一成立，不能打开 P3/P4 official：

```text
code_packet_missing_final_scripts = 1
query_reference_used_in_official_source = 1
metric_seed_mismatch = 1
matched_control_scope_global_when_local_required = 1
uses_label_or_ce_for_direction = 1
```

---

# 5. Line A：B320-current monitor 与 label-free LineC-aware signal frame

## 5.1 目标

B320-current 继续作为 diagnostic anchor，但它有 label-informed trainprobe claim 限制。Label-free line 不再追“更强 task head”，而追：

$$
\boxed{\text{task 接近 A0，同时 LineC non-tearing 打开。}}
$$

## 5.2 baseline

必须保留：

```text
A0 B320-current label-informed diagnostic anchor
A1 noYForStats
A51 staged unlabeled adapt warm1 r002
A63/A65 rmsq variants
MLP same-param / same-step reference
```

## 5.3 新候选 family

### A66：LineC-aware unlabeled multi-sketch frame

不读 label/CE。使用 train-stream random cotangent sketch 的多 seed 稳定子空间作为 projector frame。

记录：

```text
cotangent_sketch_seed_count
frame_stability_cos_mean
signal_mass_stability
projector_condition
LineC pass rate
```

### A67：Reservoir-stabilized residual frame

目标不是降低 CE，而是减少 unlabeled sketch 中的 unstable reservoir fraction proxy。

可用 proxy：

```text
output covariance tail mass
train-probe drift variance
activation branch energy imbalance
random-cotangent projector instability
```

### A68：A51 + train-probe coupling-preserving frame

保留 A51 的 task trajectory，只增加一个小 residual frame，使 train-probe CouplingR2 和 multi-sketch stability 不低于 A0。

### A69：A51 + multi-sketch RMSQ/BoundQ no-readout repair

只允许改 projection / quad path，不允许继续增强 direct/readout。

## 5.4 记录指标

`v1224_label_free_signal_frame.csv`

```text
candidate_id
dataset
seed
epochs
train_size
mean_delta_vs_A0
worst_delta_vs_A0
AUC_step_ratio_vs_A0
AUC_time_ratio_vs_A0
ECE_delta_vs_A0
CEp99_delta_vs_A0
LineC_pass
LineC_pass_rate
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
multi_sketch_pass_count
train_shuffle_pass_count
step_ratio
memory_ratio
uses_label
uses_ce_vector
```

## 5.5 gate

Exploration：

```text
mean_delta_vs_A0 >= -0.005
worst_delta_vs_A0 >= -0.025
AUC_time_ratio_vs_A0 <= 1.10
LineC_pass_rate >= 3/9
```

Official：

```text
mean_delta_vs_A0 >= -0.003
worst_delta_vs_A0 >= -0.010
AUC_time_ratio_vs_A0 <= 1.00
LineC_pass_rate = 9/9
```

## 5.6 failure actions

```text
If task ok but LineC pass = 0:
  Do not tune direct/readout/RMSQ again.
  Move to multi-sketch frame and train-probe coupling-preserving frame.

If LineC improves but task drops:
  Use small residual frame on A51 instead of replacing projector.

If A66/A67/A68 all fail:
  Declare label-free B320-like architecture not recovered in v12.24 and keep B320-current as diagnostic anchor only.
```

---

# 6. Line C/T：从 response-level visibility 到 online micro-probe visibility

## 6.1 目标

v12.23 已证明 T2/T3 response-level signal 存在，但 T1A/T1B precommit signal 不够。v12.24 的目标是构造介于二者之间的 **online precommit micro-probe**：

```text
在 commit 前，可以对候选在 train-stream clone 上做无标签、无 CE 的几何 response probe；
不能使用 validation/test/future outcome；
不能使用 label/CE residual；
可以使用 unlabeled logits、activation、random-cotangent sketch、train-probe drift。
```

## 6.2 feature tiers

```text
T1A static precommit:
  不执行 candidate，只读当前 state。

T1B online micro-probe:
  在 clone 上执行 candidate，但只读无标签几何 response。
  允许成本较高，但必须在 commit 前可计算。

T2 response upper-bound diagnostic:
  可读 completed response / LineC audit outcome。
  promotion_allowed = 0。

T3 audit-only target:
  可用 label/CE/noise residual。
  promotion_allowed = 0。
```

## 6.3 T1B online micro-probe features

新增：`v1224_t1b_online_microprobe_features.csv`

```text
candidate_id
actuator_id
dataset
seed
budget
sign
probe_batch_source
uses_label
uses_ce_vector
uses_validation_or_test
precommit_available
clone_probe_used
unlabeled_logit_drift_l2
unlabeled_logit_drift_max
train_probe_CouplingR2_delta_proxy
random_cotangent_sketch_delta_fro
projector_angle_proxy
branch_energy_transport_l1
role_entropy_delta
output_cov_tail_delta
activation_cov_tail_delta
multi_sketch_stability_score
predicted_release_score
```

## 6.4 visibility gate

T1B exploration：

$$
AUC_{joint} \ge 0.65,
$$

$$
Precision@K_{joint} \ge 0.25,
$$

$$
Recall@K_{joint} \ge 0.20.
$$

T1B official：

$$
AUC_{joint} \ge 0.75,
$$

$$
Precision@K_{joint} \ge 0.40,
$$

$$
Recall@K_{joint} \ge 0.35.
$$

Support constraint：

```text
positive support must cover >= 2 datasets and >= 2 seeds;
no single dataset/seed/window concentration;
leave-one-dataset AUC_joint >= 0.60 for exploration.
```

## 6.5 failure actions

```text
If T1A fail but T1B pass:
  Use T1B online micro-probe as event selector, with cost accounting.

If T1B has AUC but precision low:
  Run variable-k threshold and abstention policy; do not lower hard gate.

If T2/T3 pass but T1B fail:
  Do not stop; distill T2 response features into T1B using only unlabeled response proxies.

If T1B still fail after Level 2:
  Declare current functional value source unobservable under allowed features; switch to Line D or new functional target.
```

---

# 7. Line I/B：Functional S4-to-S5 bridge

## 7.1 目标

当前最明确的 functional signal 来自 I24/I25 direct-logit compensated actuator，但它是 query-reference / audit-only。v12.24 的任务是把这个信号转化为 official path。

## 7.2 候选机制

### I28：Train-stream EMA compensation

替代单 train batch或query batch，使用 train-stream EMA 的 unlabeled logit / feature covariance 做 direct compensation reference。

```text
reference = EMA over K train microbatches
K = 4, 8, 16
no label / no CE
```

### I29：Train-probe ridge ensemble compensation

使用多个 train microbatch 的 ridge compensation，采用 median / trimmed-mean，而不是简单 average。

### I30：T1B-guided direct compensation

用 T1B online micro-probe score 选择 compensation strength 与 sign，不能使用 T2/T3 audit target。

### I31：Drift-safe release via null-logit compensation

对 I13/I15/I17 这种 high-release high-drift 方向，求一个 label-free logit-null correction：

$$
\Delta \theta = \Delta \theta_{release} + \lambda \Delta \theta_{comp}
$$

使：

$$
\|\Delta z\|_{\infty} \le 0.05
$$

同时保持 release proxy 不塌。

### I32：Policy-aware P3 candidate

P3 不只测一阶 actuator event，而测 “event + planned quad_only policy” 的无标签几何响应。它不是 P4 task short-run，而是一个 commit 前的 policy compatibility probe。

记录：

```text
policy = quad_only / direct_only / branch_gain / mixed
probe_steps = 1, 3, 5
uses_label = 0
uses_ce_vector = 0
```

## 7.3 P3 gate

P3 source 必须满足：

```text
role_safe_movement_pass = 1
release_audit_pass = 1
matched_control_gap >= 0.005
loss_agnostic_direction = 1
precommit_or_online_microprobe_available = 1
query_reference_used = 0
```

并且：

$$
\Delta CouplingR^2 \ge 0.02,
$$

$$
\Delta NoiseSignalLeak \le -0.01,
$$

$$
\Delta RealSignalReservoirRatio \le -0.01.
$$

## 7.4 P4 short-run gate

P4 必须比较：

```text
source
NoOpMatchedOverhead
RandomMatchedNorm
RoleMatchedRandomControl
AdamWParallelDirection
SNR-only
same-compensation-substrate control
```

记录：`v1224_p4_short_run.csv`

```text
source_id
control_id
dataset
seed
train_shuffle_seed
linec_sketch_seed_set
epochs
policy
lr
source_acc
noop_acc
control_acc
source_NLL
noop_NLL
source_CEp99
noop_CEp99
source_ECE
noop_ECE
source_AUC_error_time
noop_AUC_error_time
source_step_time_ratio
source_CouplingR2
noop_CouplingR2
source_NoiseSignalLeak
noop_NoiseSignalLeak
source_RealSignalReservoirRatio
noop_RealSignalReservoirRatio
multi_sketch_pass_count
train_shuffle_pass
p4_pass
promotion_allowed
```

## 7.5 robustness gate

Official P4 requires：

```text
exact replay pass = 1
train-shuffle pass >= 2/3
multi-sketch pass >= 4/5
at least 2 datasets or 3 dataset-seed rows pass exploration
not concentrated in one seed/window
```

## 7.6 failure actions

```text
If query-reference source passes but train-stream source fails:
  Do not repeat query source; improve EMA / train-probe compensation reference.

If P3 pass but P4 fail:
  Run policy-aware P3 probe and P4 role-policy scan.

If P4 pass but P3 fail:
  Run coupling-preserving projection, but require loss-agnostic provenance.

If exact replay passes but train-shuffle fails:
  Treat as audit-only; run stochastic robustness ensemble or stop this mechanism.

If majority-sketch passes but all-sketch fails:
  Keep as exploration, not official; test multi-sketch aggregated event selector.
```

---

# 8. Line D：Classic No-BSpline hardening

## 8.1 目标

v12.23 已证明 Rational/Fourier 不是 kernel-blocked，并且 Rational task smoke/hardening 最强。Line D 不能再只是 smoke；v12.24 必须把 Rational/Fourier 作为独立 classic-family hardening line。

## 8.2 active families

```text
Primary:
  Rational
  Fourier

Secondary:
  Chebyshev
  Wavelet
  RBF/FastKAN

Frozen:
  B-spline
```

## 8.3 Rational hardening

候选：

```text
B7me/B7lp/B7lz/B7ma/B7me-like output-scale geometry
Rational group flashgroup G16 / G32
pairNorm + stop-gradient batch cap
low linear residual gain
reservoir-vetoed output scale
```

目标不是再修 kernel，而是：

```text
task trajectory + LineC CouplingR2 + reservoir ratio。
```

## 8.4 Fourier hardening

候选：

```text
lowfreq K2 / K4
lowfreq + linear residual
phase-stable fixed frequency
high-frequency damped residual
```

主要风险：high frequency noise leak。

## 8.5 Line D 记录字段

`v1224_classic_hardening.csv`

```text
family
candidate_id
dataset
seed
epochs
step_ratio
memory_ratio
A4_expression_pass
val_acc
mean_delta_vs_MLP
AUC_time_ratio
ECE
NLL
CEp99
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
LineC_pass
status
blocker
```

## 8.6 gate

Exploration：

```text
step_ratio <= 1.25
memory_ratio <= 1.0
mean_delta_vs_MLP >= -0.03
CouplingR2 >= 0.15
NoiseSignalLeak <= 0.20
RealSignalReservoirRatio <= 0.70
```

Official family pass：

```text
step_ratio <= 1.10
memory_ratio <= 0.80
mean_delta_vs_MLP >= -0.005
AUC_time_ratio <= 1.00
LineC_pass_rate >= 7/9
```

## 8.7 failure actions

```text
If Rational task good but reservoir high:
  Try reservoir-vetoed output scale and group diversity regularization, no CE-specific tuning.

If Fourier CouplingR2 good but task weak:
  Add lowfreq+linear residual, not high-frequency expansion.

If Cheby/Wavelet/RBF remain task weak:
  Keep as secondary; do not consume P4 functional budget.
```

---

# 9. Required visualizations

v12.24 必须生成：

```text
fig_linea_task_linec_pareto.svg
fig_linea_linec_pass_by_candidate.svg
fig_t1_t2_visibility_gap.svg
fig_t1b_microprobe_auc_precision_recall.svg
fig_actuator_safe_release_control_triangle.svg
fig_p3_p4_decoupling_scatter.svg
fig_p4_source_noop_control_acc.svg
fig_multisketch_pass_heatmap.svg
fig_train_shuffle_robustness.svg
fig_query_vs_train_compensation.svg
fig_classic_hardening_pareto.svg
fig_rational_fourier_linec_task.svg
fig_stop_rule_execution_depth.svg
```

关键图定义：

## 9.1 P3/P4 decoupling scatter

横轴：

$$
P3Score = \Delta CouplingR^2 - \Delta NoiseSignalLeak - \Delta RealSignalReservoirRatio
$$

纵轴：

$$
P4Gain = Acc_{source} - \max(Acc_{noop}, Acc_{control})
$$

颜色：role / actuator family。

目标：判断 P3 与 P4 是否仍然错位。

## 9.2 Safe-release-control triangle

三轴：

```text
logit drift safety
release audit
matched control gap
```

目标：确认 v12.24 是否打破 safe/no-release 与 release/unsafe 分离。

---

# 10. v12.24 route definitions

```text
S0-CodeAuditPass:
  final code packet complete; no provenance blockers.

S1-LabelFreeLineCExploration:
  label-free candidate reaches task near-gate and LineC pass rate >= 3/9.

S2-PrecommitVisibilityExploration:
  T1B online micro-probe reaches exploration AUC/precision/recall.

S3-RoleSafeActuatorSurvivor:
  precommit-safe actuator has role-safe movement + release + control gap.

S4-FunctionalP3Opened:
  cloned P3 gate opens with source provenance clean.

S4b-P4AuditPass:
  P4 task/geometry gain exists, but one of provenance/robustness/multisketch gates fails.

S5-OfficialFunctionalSuccess:
  P3/P4/provenance/robustness all pass.

R4-MechanismNoGoAfterDepth2:
  all mandatory fallback levels executed; no official or exploration path.

R0-IncompleteExecution:
  any required fallback, code packet, or artifact missing.
```

---

# 11. v12.24 并行执行建议

为了避免 Codex 顺序执行太慢，建议并行：

```text
GPU 0:
  Line A label-free A66-A69 scout/hardening.

GPU 1:
  Line T1B online micro-probe feature generation and visibility scoring.

GPU 2:
  Line I/B I28-I32 actuator / P3 / P4 bridge.

GPU 3:
  Line D Rational/Fourier hardening.
```

执行顺序：

```text
Phase 0:
  Code packet and route contract audit.

Phase 1 parallel:
  Line A, T1B, I28-I32 scout, Line D scout.

Phase 2 auto fallback:
  If Line A LineC=0 -> A66/A67 mechanism reset.
  If T1B AUC<0.60 -> response-distillation feature generation.
  If I28/I29 fail -> I30/T1B-guided compensation.
  If safe/release split persists -> I31 null-logit compensation.
  If P3/P4 decouple -> I32 policy-aware P3.

Phase 3:
  P4 short-run only for legal P3 source.

Phase 4:
  robustness replay and multi-sketch aggregation.
```

---

# 12. 最终判断

v12.23 不能被解读成“没有任何 functional 进展”。它真正给出的结论是：

```text
1. B320-current 仍是强 diagnostic anchor，但 label-free/external-ready base 还没恢复。
2. Label-free task 已经能接近 A0，但 LineC non-tearing 为 0。
3. T2/T3 response-level signal 存在，但 T1 precommit source 不足。
4. I24/I25 可以打开 P3，甚至在 audit-only quad_only P4 中产生 task gain。
5. official S5 被 provenance、train-shuffle robustness、multi-sketch stability、P3/P4 decoupling 阻断。
6. Rational/Fourier classic line 有继续 hardening 价值。
```

v12.24 的任务不是继续同一网格堆数量，而是把这些 audit-only 信号变成 official path。核心问题是：

$$
\boxed{
\text{能否构造一个 train-stream / precommit-safe 的 compensation 与 value source，}
\text{使 P3 几何成功和 P4 task gain 落在同一个 candidate 上？}
}
$$

如果 v12.24 仍然失败，但完整执行了 T1B online micro-probe、train-stream compensation、policy-aware P3、multi-sketch robustness、Line D hardening，那么可以形成一个真正清楚的 no-go；否则不能再允许 Codex 只写 no-go 后停止。
