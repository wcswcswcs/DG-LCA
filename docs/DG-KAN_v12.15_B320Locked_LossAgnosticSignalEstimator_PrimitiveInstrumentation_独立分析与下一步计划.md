# DG-KAN v12.15：B320 Locked + Loss-Agnostic Signal Estimator / Primitive Instrumentation 重建计划

> 版本：v12.15 execution plan  
> 基于：v12.14 `B320Locked LossAgnosticFunctionalMechanism` 真实结果复盘  
> 目标：在 B320 已锁定为 strict efficient PureKAN anchor 的前提下，重新设计 **loss-agnostic functional update**。本轮不再做 CE-specific projector，不再继续 BM1/BM2/BM3/BM4 小网格，不按数据集调参。重点转向：**更强的 loss-agnostic signal-channel estimator + primitive-level instrumentation / actuator truth audit**。  
> 公式格式：Typora 友好，仅使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；B320 anchor locked；no teacher；no distillation；no loss modification；no sampler/class weight；no dataset-name branch；no validation/test/future outcome at commit time；no fake/proxy/CPU offload；B-spline frozen；functional direction generator 不使用 CE vector / label-loss VJP / permuted-label CE surrogate。CE、ECE、CEp99、AUC 只允许作为审计指标和坏化约束，不能作为 functional direction 的 value source。

---

# 0. 当前独立结论

v12.14 不是“没有进展”。它的真实进展是：**第一次严格只统计 loss-agnostic official candidates，并且确认当前 loss-agnostic functional 机制没有 P3 survivor。** 这比 v12.13 更干净，因为 v12.13 仍混有 CE-specific projector 的诊断信号；v12.14 明确把它们排除在 official 聚合之外。

当前 route 是：

```text
route = R3-FunctionalMechanismPartialOnly
P4_open = 0
bm2_full_3x3_pass_candidate_count = 0
bm2_partial_pass_rows = 0
```

这表示：

```text
1. B320 base 不是当前 blocker；
2. functional update 仍未 official success；
3. v12.14 所有 loss-agnostic B1/B2/B3/B4 候选均未过 P3；
4. 没有 P3 survivor，因此 P4 short-run 正确保持关闭。
```

最重要的新判断是：

$$
\boxed{
\text{当前 blocker 不是 CouplingR2 打不开，而是 loss-agnostic update 不能稳定释放 noise / reservoir。}
}
$$

v12.14 中大量候选可以把 `CouplingR2_delta_mean` 拉到 $0.1$ 到 $0.2$ 附近，但 `NoiseSignalLeak_delta_mean` 和 `RealSignalReservoirRatio_delta_mean` 通常只有 $10^{-4}$ 到 $10^{-3}$ 量级，距离当前 P3 hard gate 的 $-0.01$ 仍差一个数量级。

因此，下一步不能继续做：

```text
BM1/BM2/BM3/BM4 strength sweep；
BM2 coefficient grid；
window=3/5/10 小网格；
branch damping / role moment transport 小修；
把 CouplingR2 当成 promotion score；
把 CE-specific projector 包装为 loss-agnostic success；
按 MNIST / Fashion-MNIST / KMNIST 单独调 controller。
```

下一步必须回答一个更本质的问题：

$$
\boxed{
\text{loss-agnostic 的 signal-channel estimator 到底能不能预测并驱动 noise/reservoir release？}
}
$$

如果不能，仅仅“functional update 会动 logits / 会提高 CouplingR2”没有研究意义。

---

# 1. 这次结果说明了什么

## 1.1 B320 已经从“候选”变成当前实验地基

从 v12.12 / v12.13 起，B320 已经具备当前项目最需要的 base 条件：

```text
step_ratio_q90 = 0.39931987348441034
memory_ratio_q90 = 0.1295238095238095
mean_delta = +0.027669270833333332
worst_delta = -0.001953125
near_pass_rate = 1.0
AUC_step_ratio = 0.9336784156141542
AUC_time_ratio = 0.7424718100091173
ECE_delta = 0.01767905056476593
LineC_nontearing_pass = 1
```

这说明 B320 目前已经足够作为 functional update 的 anchor。v12.15 不再把“找 base”作为主目标，只做 B320 no-regression monitor。

更精确地说：

```text
过去：
  项目还卡在 PureKAN base 是否能同时快、稳、表达不差。

现在：
  B320 已经给出一个可用 anchor。
  项目真正进入 functional update 因果验证阶段。
```

## 1.2 v12.14 的 loss-agnostic audit 是必要进展

v12.14 新增了独立 runner：

```text
experiments/run_v1214_b320locked_lossagnostic_functional_mechanism.py
```

它实现了：

```text
B1 multi-cotangent spectral ensemble；
B2 role-conditioned cotangent shaping；
B3 primitive/role sketch low-eigen or isotropy shaping；
B4 loss-agnostic role moment transport；
loss-agnostic audit；
P3 controls；
P3 failure table；
P4 gate；
hash manifest；
复盘日志。
```

并且所有 official functional candidate 的方向生成字段都必须满足：

```text
loss_agnostic_direction = 1
ce_vector_used_for_direction = 0
label_used_for_direction = 0
permuted_label_used_for_direction = 0
validation_used_for_commit = 0
dataset_name_used_for_commit = 0
```

这是对的。因为你的目标不是在 CE 上设计一个补丁 optimizer，而是要让 functional update 具备 **loss-agnostic geometry maintenance** 的意义。

## 1.3 当前 metric 不是主要假阳性问题

v12.14 的 Line C calibration 仍然可信：

```text
NoOp false positive = 0
RandomMatchedNorm false positive = 0
```

这说明当前失败不是因为 Line C gate 被 NoOp / random 误触发，也不是因为所有指标都是噪声。真正问题是：**当前 functional directions 没有把 Line C 中最关键的两个量推过门槛**。

主失败计数为：

```text
NoiseSignalLeak_delta > -0.01 = 2186
RealSignalReservoirRatio_delta > -0.01 = 2178
control_gap < 0.005 = 1940
CouplingR2_delta < 0.02 = 926
```

这说明：

```text
CouplingR2 已经不是最难的门；
NoiseSignalLeak 和 RealSignalReservoirRatio 是真正 hard blocker；
control gap 仍然不足，说明很多方向仍会被 AdamW / TaskOnly / random controls 解释。
```

## 1.4 B1/B2/B3 的失败不是同一种失败

v12.14 的候选大致分成四类。

### B1：multi-cotangent spectral ensemble

代表：

```text
BM2aa-ClassMeanFreeCotangentK16
BM2ac-LogitWhitenedCotangentK16
BM2ae-OrthogonalRademacherCotangentK32
```

它们能产生可观 CouplingR2，但 noise/reservoir 改变很小。例如：

```text
BM2aa window=10:
  CouplingR2_delta_mean = 0.193464
  NoiseSignalLeak_delta_mean = +0.000291
  RealSignalReservoirRatio_delta_mean = +0.000613

BM2ac window=10:
  CouplingR2_delta_mean = 0.167980
  NoiseSignalLeak_delta_mean = +0.000175
  RealSignalReservoirRatio_delta_mean = -0.000574
```

解释：这类方向在输出位移上有效，但没有对准 signal/reservoir/noise 的真实分解。

### B2：role-conditioned cotangent shaping

代表：

```text
BM2ba-DirectRoleCotangentSpectral
BM2bb-QuadRoleCotangentSpectral
BM2bc-BranchRoleCotangentSpectral
BM2bd-DirectQuadRoleMixed
BM2bg-RoleAdaptiveCotangentNoLabelOrth
```

其中 direct role 很接近 no-op；quad role 能打开 CouplingR2；branch role 有一点 reservoir 倾向，但都远不到 $-0.01$。

解释：role information 是有用的，但当前 role-level estimator 只改变了输出耦合，不足以稳定改变梯度 sketch 的谱结构。

### B3：primitive / role sketch whitening or isotropy

代表：

```text
BM2ca-DirectPrimitiveSketchWhiten
BM2cb-QuadPrimitiveSketchWhiten
BM2cc-BranchPrimitiveSketchWhiten
BM2cd-DirectQuadSketchIsotropy
BM2ce-BranchQuadSketchIsotropy
BM2cf-SignalReservoirSketchSpread
BM2cg-NoiseNullSketchSpread
```

这些是本轮最接近“正确问题”的方向，因为它们开始直接碰 per-example gradient sketch / primitive role。但效果仍不足：大多能给 $0.06$ 到 $0.18$ 的 CouplingR2，但 noise/reservoir 仍只有 $10^{-4}$ 到 $10^{-3}$。

解释：primitive instrumentation 的方向是对的，但当前实现还太浅。它没有真的改变 B320 的 active gradient sketch eigenspace，只是在已有子空间里做了轻微重标定。

### B4：moment transport

代表：

```text
BM4a-BranchMomentTransport
BM4b-QuadMomentTransport
BM4c-DirectQuadCovTransport
BM4d-RoleCovarianceEqualization
BM4e-LogitJacobianMomentTransport
BM4f-TailSafeMomentTransport
```

它们大多接近 no-op。比如 BM4a 三个 window 都是 0，BM4b/BM4d/BM4e 的 CouplingR2_delta 只有 $0.001$ 到 $0.016$。

解释：当前 role moment transport 不是可用 value source。继续加权或调强度意义不大。

## 1.5 当前最本质的问题

我认为 v12.14 的本质结论是：

$$
\boxed{
\text{B320 的训练几何已经足够好，}
\text{但当前 loss-agnostic functional directions 只会制造输出位移，}
\text{没有足够控制 per-example gradient sketch 的 signal/reservoir eigenspace。}
}
$$

更直白地说：

```text
现在的 functional update 不是“完全没作用”；
它能提高 CouplingR2；
但它没有改变真正决定泛化几何的两个核心通道：
  noise 是否被推入 signal channel；
  real signal 是否被困在 reservoir。
```

所以慢不是因为实验没做，而是因为问题已经从工程 gate 变成机制问题。继续跑更多同类 rows 不会自然突破。

---

# 2. 是否在正确道路上

结论：**大方向是对的，但 v12.15 必须换推进层级。**

正确的部分：

```text
1. B320 locked，不再反复修 base。
2. CE-specific projector 被剔除 official 聚合。
3. Line C calibration 继续作为 hard gate。
4. P3 不过就不开 P4。
5. 不按 dataset 调参。
6. 不把 partial row 写成 survivor。
```

需要改变的部分：

```text
1. 不再把 CouplingR2 当主要 promotion signal。
2. 不再做 BM2 系数 / window / role 的平面小网格。
3. 不再用 moment transport 作为主 value source。
4. 不再把 loss-agnostic 理解成“随机 cotangent 多采样”。
5. 必须建立 primitive-level actuator truth：
   update 是否真的改变了 gradient sketch / projector / spectrum。
```

正确路线应变成：

```text
B320 anchor locked
-> Line C response identifiability
-> primitive-level actuator truth audit
-> stronger loss-agnostic signal-channel estimator
-> P3 3x3 Pareto survivor
-> P4 short-run
-> P5 confirmation
```

---

# 3. 离目标还差多远

## 3.1 Base 目标

B320 已经基本达成当前 base anchor 目标。v12.15 只需要做 no-regression monitor，不应继续把主预算花在 base repair。

状态：

```text
Base efficiency: close / pass
Base task trajectory: close / pass
Base Line C nontearing: pass
Base role: anchor, not blocker
```

## 3.2 Functional update 目标

还没达成，而且离 official 仍有距离。

当前缺口不是一两个阈值：

```text
NoiseSignalLeak_delta target <= -0.01;
current typical effect ≈ -0.0001 to -0.001.

RealSignalReservoirRatio_delta target <= -0.01;
current typical effect ≈ -0.0001 to -0.003.
```

也就是说，很多方向的 effect size 差一个数量级。

状态：

```text
Coupling movement: 已能打开；
Noise release: 未打开；
Reservoir release: 未打开；
Control-resistant value: 不足；
P4: 关闭；
Official functional success: 0。
```

## 3.3 Next-gen MLP 目标

还不能 claim。

现在最多能说：

```text
我们已有一个 strong B320 PureKAN base anchor；
但 functional update 尚未证明在该 base 上有独立、loss-agnostic、control-resistant 的几何收益。
```

最终 claim 还需要：

```text
B320 + functional
  beats B320 + ordinary update
  under same CE task training, same budget, same wall-clock envelope,
  while functional direction itself remains loss-agnostic.
```

---

# 4. v12.15 总目标

v12.15 的整体目标不是“继续找 functional candidate”，而是回答一个更精确的问题：

$$
\boxed{
\text{在 B320 anchor 上，是否存在 loss-agnostic、可观测、可部署的 functional direction，}
\text{能稳定改善 signal/reservoir/noise geometry，并击败 strong controls？}
}
$$

v12.15 不追求：

```text
继续提高 final accuracy；
调 CE；
修 ECE；
调 dataset-specific event；
继续扩大 BM2 coefficient grid；
用 CE-vector projector 追 reservoir release。
```

v12.15 追求：

```text
1. 证明 current failure 是 estimator weak、actuator weak，还是 objective misaligned；
2. 设计更强的 loss-agnostic signal-channel estimator；
3. 证明 functional update 真正改变 primitive-level gradient sketch eigenspace；
4. 找到至少一个 3x3 P3 Pareto survivor；
5. 只有 P3 survivor 出现后才打开 P4 short-run。
```

---

# 5. 核心假设

## H1：当前失败主要来自 estimator 弱，而不是 Line C gate 错

v12.14 已经显示 NoOp / RandomMatchedNorm false positive 为 0。因此假设：

$$
\boxed{
\text{Line C gate 有效；当前 loss-agnostic estimator 没有足够识别 noise/reservoir release direction。}
}
$$

验证方式：

```text
测 estimator 对真实 Line C response 的 correlation；
测不同 cotangent ensemble / sketch dim / batch size 下的 effect-size scaling；
测 estimator 选出的方向是否比 controls 更能预测 NoiseSignalLeak / RealSignalReservoirRatio。
```

如果 estimator correlation 仍接近 0，则不能继续采样更多同类 cotangent。

## H2：当前 actuator 可能太弱，没有真正改变 gradient sketch projector

很多 BM4 近似 no-op，B1/B2/B3 也只改变输出位移。因此假设：

$$
\boxed{
\text{functional direction 对 logits 有位移，但对 } \hat W_B, P_{sig}, P_{res} \text{ 的改变太弱。}
}
$$

验证方式：

```text
记录 delta_W_norm；
记录 signal projector angle；
记录 reservoir projector angle；
记录 sketch eigenvalue shift；
记录 role-wise parameter delta norm；
记录 primitive-level gradient-sketch drift。
```

如果 logit drift 大但 projector drift 小，说明是 coordinate/logit-only move。  
如果 projector drift 大但 noise/reservoir 不改善，说明 objective misaligned。  
如果二者都小，说明 actuator weak。

## H3：真正有效方向需要 primitive-level response matrix，而不是 output-only cotangent

B320 的 functional geometry 可能由 direct / quad / branch / projection / readout 不同 role 共同决定。当前 role-conditioned 只做浅层 gating，可能不够。因此假设：

$$
\boxed{
\text{需要构造 role-conditioned response matrix } R_{m,j}=\frac{\partial G_m}{\partial a_j},
\text{而不是直接从随机 cotangent 生成参数更新。}
}
$$

其中 $G_m$ 是 Line C 指标，$a_j$ 是一组 loss-agnostic primitive actuator basis 的系数。

## H4：SNR gate 是 safety signal，不是 value source

BM1/SNR 有局部 positive，但覆盖太低。因此：

```text
SNR 可以用于 veto / precondition / downweight；
但 v12.15 不把 SNR-only 当 functional value source。
```

## H5：functional update 必须 loss-agnostic，但 task training 仍可用 CE

这个 distinction 必须写死：

```text
主训练 optimizer 可以是 CE + AdamW-equivalent / manual optimizer；
functional direction generator 不可使用 CE vector、label-loss VJP、permuted-label CE；
CE / ECE / CEp99 / AUC 只用于审计、坏化约束、离线分析；
commit-time functional decision 不能使用 validation/test/future outcome，也不能使用 dataset_name。
```

---

# 6. 总体实验结构

v12.15 分成五条并行线。

```text
Line A：B320 Anchor Monitor
  只做 no-regression，不继续 base 小修。

Line C：Line C Response Identifiability
  估计当前 Line C 指标是否可被 loss-agnostic observables 预测。

Line I：Primitive-Level Instrumentation / Actuator Truth
  判断 candidate 是否真正改变 gradient sketch eigenspace。

Line B：Loss-Agnostic Functional Mechanism Rebuild
  基于 Line C/I 的结果构造新 functional candidates。

Line D：Classic No-BSpline Portfolio Monitor
  Rational / Chebyshev / Wavelet / RBF / Fourier 继续并行，但不抢 functional 主预算。
```

依赖关系：

```text
Line A 先跑一次，然后作为所有实验固定 base。
Line C/I 与 Line B 并行开发，但 P3 promotion 必须依赖 C/I 的 response evidence。
Line D 独立并行；若出现 FamilyPass 或 near-pass，再接入 Line C/I。
```

---

# 7. Line A：B320 Anchor Monitor

## 7.1 目标

确认 B320 没有因为代码改动或 instrumentation 发生回归。v12.15 不再优化 B320。

## 7.2 实验设置

```text
candidate = B320b locked config
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2 or reuse exact hardening artifact if hash matches
train_size = 1024
val_size = 512
test_size = 512
batch_size = 128
protocol = v12.14 exact
```

## 7.3 记录文件

`v1215_anchor_monitor.csv`

字段：

```text
run_id
candidate_id
dataset
seed
step_ratio_q90
memory_ratio_q90
forward_ratio_q90
backward_ratio_q90
update_ratio_q90
mean_delta_vs_mlp
worst_delta_vs_mlp
near_pass
AUC_step_ratio
AUC_time_ratio
ECE_delta
CEp99_delta
LineC_nontearing_pass
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
hash_match_v1214
regression_flag
```

## 7.4 通过标准

$$
step\_ratio_{q90}\le 0.60
$$

$$
memory\_ratio_{q90}\le 0.20
$$

$$
mean\_delta\ge 0
$$

$$
worst\_delta\ge -0.003
$$

$$
AUC_{step}\le 1.00
$$

$$
AUC_{time}\le 1.00
$$

$$
LineC\_nontearing\_pass=1
$$

如果失败：

```text
Codex 不要修 functional。
先定位是否 hash mismatch、kernel path regression、timing warmup change、Line C instrumentation mixed into timed path。
只允许修复 regression，不允许改变 B320 architecture 超参。
```

---

# 8. Line C：Response Identifiability

## 8.1 目标

回答：当前 loss-agnostic observables 是否能预测 Line C hard metrics 的真实变化。

不是直接找 candidate，而是先建立 response science：

$$
\Delta G = (\Delta CouplingR^2, \Delta NoiseSignalLeak, \Delta RealSignalReservoirRatio).
$$

我们要判断：

```text
当前 observable features 是否能预测 ΔNoiseSignalLeak 与 ΔReservoirRatio？
如果不能，继续用它们做 functional selection 没意义。
```

## 8.2 实验设计

构造小幅 loss-agnostic probe updates：

```text
U0 NoOp
U1 RandomMatchedNorm
U2 TaskOnlyAdamW direction, audit only
U3 SNR-gated AdamW residual, audit only
U4 random cotangent VJP ensemble
U5 orthogonal cotangent VJP ensemble
U6 role-conditioned primitive actuator
U7 projection/quad role actuator
U8 branch/readout role actuator
U9 mixed low-rank actuator basis
```

每个 update 都必须满足：

```text
loss_agnostic_direction = 1, except audit controls explicitly marked;
ce_vector_used_for_direction = 0;
label_used_for_direction = 0;
validation_used_for_commit = 0;
dataset_name_used_for_commit = 0.
```

对每个 update，计算实际 response：

$$
\Delta G_j = G(\theta+\epsilon u_j)-G(\theta)
$$

并记录 loss-agnostic observable：

```text
cotangent_spectrum
role_energy_direct
role_energy_quad
role_energy_branch
sketch_top_eigen_shift_pred
sketch_condition_pred
projector_angle_pred
logit_drift_pred
parameter_role_norm
SNR_positive_fraction
```

## 8.3 记录文件

`v1215_linec_response_identifiability.csv`

字段：

```text
run_id
dataset
seed
split_id
window
batch_size
probe_update_id
probe_update_family
loss_agnostic_direction
ce_vector_used_for_direction
label_used_for_direction
functional_norm_ratio
role_direct_norm
role_quad_norm
role_branch_norm
role_projection_norm
logit_drift_l2
logit_max_abs_drift
sketch_delta_fro
sketch_top_eigen_delta
signal_projector_angle
reservoir_projector_angle
CouplingR2_before
CouplingR2_after
CouplingR2_delta
NoiseSignalLeak_before
NoiseSignalLeak_after
NoiseSignalLeak_delta
RealSignalReservoirRatio_before
RealSignalReservoirRatio_after
RealSignalReservoirRatio_delta
CEp99_delta_audit
ECE_delta_audit
holdout_loss_ratio_audit
control_id
```

`v1215_linec_response_model.csv`

字段：

```text
model_id
feature_set
metric_target
spearman_corr
pearson_corr
kendall_corr
r2_oos
sign_accuracy
negative_release_precision
negative_release_recall
bootstrap_ci_low
bootstrap_ci_high
```

## 8.4 判断标准

Estimator 可以进入 Line B 的条件：

$$
Spearman(\hat{\Delta NoiseLeak}, \Delta NoiseLeak) \ge 0.30
$$

或：

$$
Precision(\Delta NoiseLeak\le -0.01)\ge 0.50
$$

并且：

$$
Spearman(\hat{\Delta Reservoir}, \Delta Reservoir) \ge 0.30
$$

或：

$$
Precision(\Delta Reservoir\le -0.01)\ge 0.50.
$$

如果所有 estimator 都低于这些标准：

```text
Route = R2-LossAgnosticEstimatorNotPredictive
Codex 动作：停止 BM2/B3 小网格；转入更深 primitive instrumentation 或重新定义 Line C sketch construction。
```

## 8.5 可视化

```text
fig_v1215_response_pred_vs_actual_noise.svg
fig_v1215_response_pred_vs_actual_reservoir.svg
fig_v1215_response_pred_vs_actual_coupling.svg
fig_v1215_projector_angle_vs_noise_release.svg
fig_v1215_role_energy_vs_response.svg
fig_v1215_response_precision_recall.svg
```

---

# 9. Line I：Primitive-Level Instrumentation / Actuator Truth

## 9.1 目标

当前 B1/B2/B3/B4 的失败可能是 actuator 没有真正改变 gradient sketch。Line I 要回答：

$$
\boxed{
\text{functional update 是否真正改变了 } \hat W_B, P_{sig}, P_{res} \text{？}
}
$$

## 9.2 三种失败模式

### ActuatorWeak

```text
parameter_delta_norm > 0
logit_drift > 0
but sketch_delta_fro ≈ 0
and projector_angle ≈ 0
```

说明 update 只是 readout/logit 层小扰动，没有改变训练几何。

### ObjectiveMisaligned

```text
sketch_delta_fro high
projector_angle high
but NoiseSignalLeak / ReservoirRatio not improved
```

说明 actuator 有力，但 objective 错。

### SafetyRejected

```text
Line C response good
but logit drift / CEp99 / ECE / holdout audit bad
```

说明需要 safety projection，不是 value source 失败。

## 9.3 实验设计

对 B320 角色拆分 actuator：

```text
I1 Direct role actuator
I2 Quad role actuator
I3 Branch role actuator
I4 Projection-P role actuator
I5 Readout role actuator
I6 Direct+Quad coupled actuator
I7 Quad+Branch coupled actuator
I8 Projection+Readout coupled actuator
I9 All-role low-rank actuator
```

每个 actuator 使用 label-free cotangent / random sketch / covariance target 生成方向，不使用 CE / label。

## 9.4 记录文件

`v1215_actuator_truth.csv`

字段：

```text
run_id
actuator_id
role
candidate_id
dataset
seed
window
batch_size
loss_agnostic_direction
param_delta_norm_total
param_delta_norm_direct
param_delta_norm_quad
param_delta_norm_branch
param_delta_norm_projection
param_delta_norm_readout
logit_drift_l2
logit_max_abs_drift
sketch_delta_fro
sketch_delta_op
signal_projector_angle_deg
reservoir_projector_angle_deg
signal_eigen_mass_delta
reservoir_fraction_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CouplingR2_delta
CEp99_delta_audit
holdout_loss_ratio_audit
classification_failure_mode
```

## 9.5 判断标准

Actuator 可以进入 functional candidate 生成的条件：

$$
sketch\_delta\_fro \ge 0.01
$$

$$
\max(signal\_projector\_angle, reservoir\_projector\_angle) \ge 1^\circ
$$

同时：

$$
logit\_max\_abs\_drift \le 0.05
$$

如果 actuator weak：

```text
Codex 先尝试：
1. 增大 role-specific norm budget，但保持 logit drift <= 0.05；
2. 改为 projection-P / quad role，因为 direct/branch 可能只改 readout；
3. 实现 primitive-level fused apply，使 update 作用到真实 active parameters；
4. 若仍 weak，标记该 role 不再作为 value source。
```

如果 objective misaligned：

```text
Codex 不调强度。
改 objective：从 coupling / isotropy 转向 explicit noise-reservoir spectral target。
```

如果 safety rejected：

```text
Codex 先做 loss-agnostic safety projection：
  max logit drift cap；
  sketch op-norm cap；
  tail proxy cap；
不允许用 CE gradient 做 safety direction。
```

## 9.6 可视化

```text
fig_v1215_actuator_role_response_heatmap.svg
fig_v1215_param_delta_vs_sketch_delta.svg
fig_v1215_projector_angle_by_role.svg
fig_v1215_actuator_failure_taxonomy.svg
```

---

# 10. Line B：Loss-Agnostic Functional Mechanism Rebuild

Line B 只允许使用通过 Line C/I 审计的 estimator + actuator。没有 C/I evidence 的 candidate 不进入 P3。

## 10.1 Candidate family B15-A：Response-Matrix Functional QP

### 目标

用 loss-agnostic probe 得到局部 response matrix：

$$
R_{m,j}=\frac{G_m(\theta+\epsilon u_j)-G_m(\theta)}{\epsilon},
$$

其中：

$$
G_m\in\{CouplingR^2, -NoiseSignalLeak, -RealSignalReservoirRatio, -TailProxy\}.
$$

再求一个低维系数 $a$：

$$
\max_a
\quad
\alpha \Delta CouplingR^2
-\beta \Delta NoiseSignalLeak
-\gamma \Delta RealSignalReservoirRatio
-\eta \Delta TailProxy
-\rho \|a\|^2.
$$

约束：

$$
\|\Delta logits\|_\infty \le 0.05
$$

$$
\|\Delta \hat W_B\|_{op} \le \tau_W
$$

$$
\Delta NoiseSignalLeak \le -0.01
$$

$$
\Delta RealSignalReservoirRatio \le -0.01
$$

注意：这里的 $G_m$ response 来自 loss-agnostic Line C probes，不使用 CE vector 作为方向生成。

### 记录字段

```text
response_matrix_rank
response_matrix_condition
basis_update_count
selected_basis_ids
qp_status
constraint_active_count
predicted_CouplingR2_delta
predicted_NoiseSignalLeak_delta
predicted_Reservoir_delta
actual_CouplingR2_delta
actual_NoiseSignalLeak_delta
actual_Reservoir_delta
prediction_error
```

### 失败后 Codex 动作

```text
if response_matrix_condition too high:
  prune nearly collinear basis;
  add ridge;
  split by role;
  reduce rank.

if predicted good but actual bad:
  Line C response is nonlinear or noisy;
  reduce lambda / epsilon;
  use two-point symmetric finite response;
  increase probe batch only once.

if no feasible solution:
  current actuator basis lacks noise/reservoir direction;
  go back to Line I and add primitive-level actuator, not coefficient grid.
```

## 10.2 Candidate family B15-B：Sketch-Eigenspace Targeting

### 目标

直接改变 gradient sketch 的谱，而不是追 CouplingR2。

令当前 sketch 累积为：

$$
\hat W = \sum_t \Phi_t\Phi_t^T.
$$

构造 loss-agnostic target：

```text
1. 降低 noise-like cotangent 在 top signal eigenspace 的投影；
2. 提高 low-energy but stable cotangent 在 mid eigenspace 的可动性；
3. 避免 top eigen share 过度集中。
```

指标：

$$
TopShare=\frac{\lambda_1}{\sum_i\lambda_i+\epsilon}
$$

$$
EffRank=\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2+\epsilon}
$$

目标：

$$
\Delta EffRank>0,
$$

$$
\Delta TopShare\le 0,
$$

并在 audit 中观察：

$$
\Delta NoiseSignalLeak<0,
$$

$$
\Delta RealSignalReservoirRatio<0.
$$

### 失败后 Codex 动作

```text
if EffRank improves but NoiseSignalLeak worsens:
  spectrum target too generic;
  add noise-null cotangent constraint.

if TopShare drops but CouplingR2 drops:
  over-flattened signal channel;
  add minimum coupling constraint.

if effect near no-op:
  actuator weak; return to Line I.
```

## 10.3 Candidate family B15-C：Noise-Null / Reservoir-Release Split Mechanism

### 目标

把 noise 和 reservoir 两个目标拆开，不再用单一 score。

先构造两个 loss-agnostic sub-directions：

```text
u_noise: 只尝试降低 NoiseSignalLeak；
u_res:   只尝试降低 RealSignalReservoirRatio；
```

然后求组合：

$$
u = a u_{noise}+b u_{res},
$$

并要求：

$$
\Delta NoiseSignalLeak(u)\le -0.01,
$$

$$
\Delta ReservoirRatio(u)\le -0.01.
$$

### 失败后 Codex 动作

```text
if u_noise works but u_res fails:
  mark reservoir release as primary blocker;
  add mid-spectrum cotangent basis.

if u_res works but u_noise worsens:
  add noise-null orthogonal projection;
  do not use CE-specific projector.

if both fail:
  estimator/actuator insufficient; return to Line C/I.
```

## 10.4 Candidate family B15-D：Unlabeled Temporal Consistency Functional Update

### 目标

完全不使用 label / CE，通过同一样本在 micro-augmentation 或 dropout-free perturbation 下的 logit-Jacobian consistency 来构造 functional direction。

对同一输入的小扰动 $x$ 与 $x+\xi$，定义：

$$
D_{cons}=\|J_f(x)-J_f(x+\xi)\|_F^2.
$$

functional direction 尝试降低这个 loss-agnostic consistency debt，同时不降低 Line C：

$$
\Delta D_{cons}<0,
$$

$$
\Delta NoiseSignalLeak\le 0,
$$

$$
\Delta RealSignalReservoirRatio\le 0.
$$

这不是任务 loss，也不使用 label。它测试 functional update 是否能做真正的 geometry maintenance。

### 失败后 Codex 动作

```text
if consistency improves but task audit worsens:
  consistency objective over-smooths;
  lower norm budget and add signal minimum.

if consistency does not affect Line C:
  it is irrelevant geometry; deprioritize.
```

---

# 11. P3 Gate：Loss-Agnostic Functional Survivor 标准

P3 row-level pass 必须同时满足：

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

$$
CEp99\_delta_{audit} \le 0.05
$$

并且：

```text
loss_agnostic_direction = 1
ce_vector_used_for_direction = 0
label_used_for_direction = 0
permuted_label_used_for_direction = 0
validation_used_for_commit = 0
dataset_name_used_for_commit = 0
```

Promotion 规则：

```text
Strong promotion:
  same candidate passes all 3 datasets x 3 seeds = 9/9 rows.

Weak promotion:
  >=8/9 pass, bootstrap CI lower for control_gap >= 0,
  and failure not concentrated in one dataset or one seed.

Forbidden promotion:
  single row pass；
  only MNIST pass；
  only one window pass；
  CE-specific direction；
  CouplingR2-only pass；
  control gap <= 0；
  direction indistinguishable from AdamWParallel / SNR-only。
```

---

# 12. P4 Short-Run：只在 P3 survivor 后打开

## 12.1 目标

验证 P3 的 one-step / window-level gain 是否能变成 short-run training gain。

## 12.2 设置

```text
base = B320 locked
training = normal CE task optimizer + functional maintenance event
functional_event_interval = 24 or 48 steps
functional_direction = P3 survivor only
loss used for task = CE
loss used for functional direction = none / loss-agnostic Line C estimator only
```

controls：

```text
C0 TaskOnlyAdamW-equivalent
C1 NoOpMatchedOverhead
C2 RandomMatchedNorm
C3 AdamWParallelDirection
C4 SNR-only
C5 GeometryOnlyNoSNR
C6 ShuffledPayload
C7 MLPAnalogMaintenance
C8 SameLineCScoreButRandomRole
```

## 12.3 记录文件

`v1215_p4_short_run.csv`

字段：

```text
run_id
method
control_id
dataset
seed
step
event_id
functional_event_applied
functional_norm_ratio
amortized_overhead_ratio
train_loss
val_loss
val_acc
AUC_step_ratio
AUC_time_ratio
ECE
CEp99
margin_p10
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
logit_max_abs_drift
bad_event_count
reject_event_count
accept_event_count
loss_agnostic_direction
ce_vector_used_for_direction
```

## 12.4 通过标准

P4 pass：

$$
AUC_{time,func}\le AUC_{time,base}
$$

$$
AUC_{step,func}\le AUC_{step,base}
$$

$$
Acc_{func}\ge Acc_{base}-0.003
$$

$$
ECE_{func}\le ECE_{base}+0.01
$$

$$
CEp99_{func}\le CEp99_{base}+0.05
$$

$$
NoiseSignalLeak_{func}\le NoiseSignalLeak_{base}-0.01
$$

$$
RealSignalReservoirRatio_{func}\le RealSignalReservoirRatio_{base}-0.01
$$

$$
control\_gap\ge 0.005
$$

$$
amortized\_overhead\le 1.05.
$$

如果 P4 fail：

```text
if P3 predicted effect disappears:
  response estimator not stable across training trajectory;
  add online response recalibration or reduce event interval.

if task metrics worsen:
  safety projection insufficient;
  add logit drift / sketch op-norm cap, not CE direction.

if controls match functional:
  value not unique;
  redesign estimator.
```

---

# 13. Line D：Classic No-BSpline Portfolio Monitor

B-spline 已冻结，不再占用 active budget。

Active families：

```text
Rational
Chebyshev
Wavelet
RBF / FastKAN
Fourier
```

v12.15 不把它们作为主 blocker，但要并行维护状态。

## 13.1 每个 family 的当前定位

```text
Rational:
  当前多次 L3/A4 可行，但 A5/AUC/Line C 不稳定。
  状态：TaskBlocked / GeometryBlocked。

Chebyshev:
  已有 L3/A4 接近或通过路径，但 task trajectory 仍失败。
  状态：TaskBlocked。

Wavelet:
  fused hat wavelet 已打开效率与部分 expression，但 task 仍失败。
  状态：TaskBlocked or Expression-to-TaskBlocked transition。

RBF/FastKAN:
  fused RBF L3 efficiency 已有进展，但 A4 expression 仍不足。
  状态：ExpressionBlocked。

Fourier:
  L3 efficiency 可以，A4 expression 不足。
  状态：ExpressionBlocked。
```

## 13.2 v12.15 对 Line D 的限制

```text
不跑 B-spline；
不做全 family 大扫；
每个 active family 最多两个 focused repair；
只要没有 A1/A4/A5 + Line C 近通过，就不接入 functional official；
所有 family 仍必须 loss-agnostic functional diagnostic。
```

## 13.3 记录文件

`v1215_classic_family_status.csv`

字段：

```text
family
candidate_id
status
L3_efficiency_pass
A4_expression_pass
A5_task_pass
LineC_nontearing_pass
best_step_ratio
best_memory_ratio
mean_delta
worst_delta
AUC_step_ratio
AUC_time_ratio
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
primary_blocker
next_action
active_budget_remaining
```

---

# 14. 必须生成的可视化

```text
fig_v1215_summary_route.svg
fig_v1215_noise_reservoir_effect_size_by_method.svg
fig_v1215_coupling_vs_noise_reservoir_scatter.svg
fig_v1215_loss_agnostic_audit_matrix.svg
fig_v1215_fail_reason_heatmap.svg
fig_v1215_response_pred_vs_actual_noise.svg
fig_v1215_response_pred_vs_actual_reservoir.svg
fig_v1215_actuator_role_response_heatmap.svg
fig_v1215_projector_angle_by_role.svg
fig_v1215_p3_survivor_pareto.svg
fig_v1215_p4_short_run_curves.svg
fig_v1215_controls_gap_matrix.svg
fig_v1215_classic_family_status.svg
```

---

# 15. Codex 执行顺序

为了加速，不要串行跑完整长流程。按下面并行：

```text
Batch 0:
  A0 B320 anchor monitor + artifact hash check。

Batch 1:
  C0 Line C response identifiability probes。
  I0 primitive actuator truth probes。

Batch 2:
  根据 C0/I0 结果生成 B15-A/B/C/D candidates。
  同时跑 controls。

Batch 3:
  P3 3x3 official candidate aggregation。
  没有 P3 survivor 则停止，不开 P4。

Batch 4:
  只有 P3 survivor 后才跑 P4 short-run。

Batch D:
  Classic No-BSpline focused repair 并行执行，不阻塞 Batch 1-4。
```

每个 Batch 都必须写：

```text
provenance_audit.csv
loss_agnostic_audit.csv
hash_manifest.json
route_decision.json
failure_table.csv
```

---

# 16. 停止条件

## 16.1 成功停止

进入 P4 的最低条件：

```text
至少一个 loss-agnostic P3 survivor；
P3 survivor 不是单 dataset / single seed；
NoOp / Random / AdamWParallel / SNR-only controls 被击败；
noise 和 reservoir 同时过门槛。
```

Functional official success：

```text
P4 short-run pass；
task non-harm；
Line C gain；
control-resistant；
amortized overhead <= 1.05；
loss-agnostic audit pass。
```

## 16.2 失败停止

如果出现以下情况，停止当前 mechanism family：

```text
C0 response estimator Spearman < 0.10 across all targets；
I0 actuator sketch_delta_fro < 0.01 for all roles；
B15-A/B/C/D 全部 P3 pass_rows = 0；
NoiseSignalLeak_delta 仍只有 1e-4 到 1e-3；
RealSignalReservoirRatio_delta 仍只有 1e-4 到 1e-3；
control_gap 仍大面积 < 0；
任何 direction 使用 CE vector / label-loss VJP / validation/test / dataset-name branch。
```

此时 route 应写为：

```text
R4-LossAgnosticFunctionalMechanismNotFound
```

然后进入更高层理论重建，不再继续该类 cotangent / moment / role shaping。

---

# 17. 最终判断

v12.14 的正确信息不是“functional update 没希望”，而是：

$$
\boxed{
\text{B320 base 已经把系统地基打稳；}
\text{functional update 的旧价值定义被证伪；}
\text{当前必须重建 loss-agnostic signal-channel estimator 与 primitive-level actuator。}
}
$$

v12.15 因此不是小修版本。它的核心是从 **output-level cotangent movement** 升级到 **primitive-level gradient-sketch response control**。如果这个仍失败，我们就可以非常清楚地说：当前 B320 上还没有找到可部署的 loss-agnostic functional update 机制，而不是继续在 CE 或 CouplingR2 上做局部优化。

