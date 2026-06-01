# DG-KAN v12.16：B320 Locked + Explicit Signal/Reservoir Functional Mechanism 实验结果分析与下一步计划

> 版本：v12.16 planning draft  
> 基于：v12.15 `B320Locked LossAgnosticSignalEstimator PrimitiveInstrumentation` 真实结果复盘与 continuation 结果  
> 目标：在 B320 locked base 上，停止 cotangent / role / moment / shallow actuator 小网格，重建 **loss-agnostic explicit signal/reservoir/noise functional mechanism**。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；B320 base 不再小修；functional direction 必须 loss-agnostic；不允许 CE-vector / label-loss VJP / validation/test/future outcome / dataset-name branch；CE、ECE、CEp99、Brier 只作为审计指标和坏化约束，不作为 functional direction 的目标。

---

# 0. 一句话结论

v12.15 **不是没有进展**，但也**不是 functional update 成功**。

最准确的判断是：

$$
\boxed{
\text{B320 base 已经不是 blocker；}
\text{v12.15 把 functional update 的失败从“动不了”推进到“能动但目标错位”。}
}
$$

v12.15 原始 P3 中，B15-A/B/C/D 全部没有 survivor：

```text
route = R4-LossAgnosticFunctionalMechanismNotFound
p4_open = 0
p3_survivor_count = 0
linec_rows = 90
actuator_rows = 81
p3_rows = 36
```

v12.15 continuation 进一步证明：fused primitive actuator 可以在极少数行上打开 actuator gate，说明 actuator 并非绝对不可行；但是这个 movement 仍不能稳定转化成 `NoiseSignalLeak <= -0.01` 和 `RealSignalReservoirRatio <= -0.01` 的 control-resistant release。因此，当前 blocker 不是 B320，不是 safety 过严，不是 Line D 漏跑，也不是 actuator 完全无效，而是：

$$
\boxed{
\text{functional value objective / selector 没有显式对准 noise/reservoir release。}
}
$$

下一步不能再做：

```text
cotangent count sweep；
role coefficient grid；
window / lambda 小网格；
CouplingR2-only promotion；
CE-specific projector 包装成 official；
B320 architecture 小修；
因为 K6 单行 actuator gate-open 就进入 P4。
```

下一步必须做：

```text
1. 重新定义 explicit noise/reservoir spectral target；
2. 重建或扩展 Line C sketch，使 noise/reservoir basis 在 direction generation 时可见；
3. 把 fused primitive actuator 当执行器，不把它本身当 value source；
4. 用强 controls 证明 functional update 的独立机制收益。
```

---

# 1. 当前实验结果独立分析

## 1.1 B320 base 已经锁住，不应该再小修

v12.15 没有重新优化 B320，也没有用 functional update 掩盖 base regression。Anchor monitor 复用 v12.14 locked artifact，hash match 为 1，regression flag 为 0。

B320 locked 指标：

```text
step_ratio_q90 = 0.39931987348441034
memory_ratio_q90 = 0.1295238095238095
mean_delta_vs_mlp = 0.027669270833333332
worst_delta_vs_mlp = -0.001953125
near_pass = 1.0
AUC_step_ratio = 0.9336784156141542
AUC_time_ratio = 0.7424718100091173
ECE_delta = 0.01767905056476593
LineC_nontearing_pass = 1
```

这说明：

$$
\boxed{
\text{当前失败不是 base regression，不是效率问题，不是 B320 architecture 还要调。}
}
$$

后续 B320 只做 no-regression monitor，不再做 temperature、gain、branch、epoch、optimizer 小修。继续修 B320 会把主线从 functional update 拉回 base 调参，属于跑偏。

## 1.2 Line C 不是纯噪声，但它现在只能预测 CouplingR2

v12.15 的 Line C response identifiability 说明：

```text
CouplingR2_delta 的 best feature 是 sketch_delta_fro：
  Spearman = 0.8325876488279158
  Pearson = 0.7450220117897491
  sign_accuracy = 0.9

NoiseSignalLeak_delta 的 best Spearman 只有 0.1776506664644746。
RealSignalReservoirRatio_delta 的 Spearman 约 -0.30037490215465745，但 negative_release precision / recall 仍为 0。
```

这不是坏消息，也不是好消息。它说明：

```text
Line C 测量不是随机噪声；
CouplingR2 的位移确实能被 sketch_delta_fro 解释；
但我们真正关心的 NoiseSignalLeak 和 RealSignalReservoirRatio 尚不能被当前 loss-agnostic observables 稳定预测。
```

换句话说：

$$
\boxed{
\text{我们已经能测到“模型动了”，但还不能预测“动得是否有益”。}
}
$$

这就是为什么继续提高 CouplingR2 没意义。CouplingR2 已经不是 hard blocker；NoiseSignalLeak、Reservoir release 和 control gap 才是。

## 1.3 原始 primitive actuator 确实太弱，但不是 safety 卡住

v12.15 I0 里 81 个 role actuator 全部被分类为 `ActuatorWeak`：

```text
ActuatorWeak = 81
ObjectiveMisaligned = 0
SafetyRejected = 0
AcceptedForCandidateBasis = 0
```

关键数值：

```text
max_sketch_delta_fro = 0.0038212935905903578
max_projector_angle_deg = 0.7409852743148804
```

计划里的 actuator gate 是：

$$
sketch\_delta\_fro \ge 0.01,
$$

$$
projector\_angle \ge 1^\circ.
$$

因此 I0 的结论是：

$$
\boxed{
\text{不是 logit drift / CEp99 safety 拒绝，而是 actuator 改变 active sketch/projector 的幅度不够。}
}
$$

所以继续降低 safety、加入 CE safety direction 或做 coefficient grid 都不是解决办法。

## 1.4 B15-A/B/C/D 全部失败，原因不是 coupling，而是 noise/reservoir/control

B15-A/B/C/D 都是 loss-agnostic 构造，且没有使用 CE vector、label-loss VJP、validation/test 或 dataset-name branch。它们的 mean CouplingR2 都能打开，但没有任何 P3 row 通过：

```text
B15-A/B/C/D: p3_pass_rows = 0
```

P3 failure table：

```text
NoiseSignalLeak_delta > -0.01 = 36 / 36
RealSignalReservoirRatio_delta > -0.01 = 36 / 36
control_gap < 0.005 = 36 / 36
CouplingR2_delta < 0.02 = 2 / 36
```

这排除了两个误解：

```text
误解 1：functional 失败是因为 CouplingR2 不够。
事实：只有 2/36 rows 失败 CouplingR2，CouplingR2 已不是主 blocker。

误解 2：functional 失败是因为 safety 太严。
事实：CEp99 / logit drift 不是主失败；主失败是 noise/reservoir release 和 control gap。
```

当前 functional update 的真实状态是：

$$
\boxed{
\text{能制造 output / coupling movement，但不能制造 control-resistant 的 noise/reservoir release。}
}
$$

## 1.5 continuation 的真实意义：从“动不了”推进到“目标错位”

v12.15 continuation 做了更深的 primitive instrumentation。它有两个层次的结果。

### Budget-only actuator repair

```text
route = R4-ContinuationActuatorBudgetRepairStillWeak
repair_rows = 216
actuator_gate_pass_rows = 0
max_repair_sketch_delta_fro = 0.008185695856809616
max_repair_projector_angle_deg = 1.8827358484268188
best_repair_noise_delta = -0.006306260824203491
best_repair_reservoir_delta = -0.014425188302993774
```

解释：budget-only 可以把 projector angle 推过 1 度，但 sketch_delta_fro 仍低于 0.01；单行 reservoir 有强信号，但不是可部署 estimator，不能接 P3。

### Fused primitive actuator

```text
route = R4-ContinuationActuatorGateOpenedButP3NotClaimed
repair_rows = 378
actuator_gate_pass_rows = 1
max_repair_sketch_delta_fro = 0.013528571464121342
max_repair_projector_angle_deg = 4.587743282318115
best_repair_noise_delta = -0.006748616695404053
best_repair_reservoir_delta = -0.014425188302993774
```

gate-open 行：

```text
repair_id = K6-FusedQuadSignBudgetCap
dataset = KMNIST
seed = 2
budget_multiplier = 8.0
logit_max_abs_drift = 0.040654659271240234
sketch_delta_fro = 0.010541788302361965
signal_projector_angle_deg = 4.587743282318115
reservoir_projector_angle_deg = 4.587273597717285
NoiseSignalLeak_delta = -0.0023810267448425293
RealSignalReservoirRatio_delta = -0.0028071701526641846
CouplingR2_delta = 0.2872958679372558
classification_failure_mode = ActuatorGateOpenButObjectiveMisaligned
```

这非常重要。它说明 fused primitive actuator **可以改变 active sketch/projector**，但改变方向没有对准我们的几何目标。它把问题从：

```text
functional actuator 完全太弱
```

推进成：

```text
actuator 有执行能力，但 objective / selector 没有显式构造 noise/reservoir release。
```

## 1.6 Fixed K6 / Bidirectional selector 没有解决问题

Fixed K6 P3：

```text
candidate_rows = 9
p3_pass_rows = 0
max_sketch_delta_fro = 0.014910714700818062
max_projector_angle_deg = 1.2201300859451294
best_noise_delta = -0.008338093757629395
best_reservoir_delta = -0.007685840129852295
mean_control_gap = -0.05098838506361029
```

Bidirectional selector：

```text
candidate_rows = 9
p3_pass_rows = 0
max_sketch_delta_fro = 0.010474889539182186
max_projector_angle_deg = 0.9291486144065857
best_noise_delta = -0.0035085678100585938
best_reservoir_delta = -0.002069234848022461
mean_control_gap = -0.091467112952939
```

Fixed K6 说明 effect size 能上来，但 noise/reservoir 仍不过，且部分 logit drift 超标。Bidirectional selector 解决了一些 drift，但 effect size 明显变小，control gap 仍不行。

因此：

$$
\boxed{
\text{问题不是方向符号，也不是预算倍率，而是 target 本身没有捕捉 noise/reservoir release basis。}
}
$$

## 1.7 Classic No-BSpline Line D 的状态：已补跑，但不能接入主线

v12.15 补跑 Line D focused repairs，约束是：不跑 B-spline，不做全 family 大扫，每个 active family 跑两个 focused repair，no-fake/no-proxy/no-cpu 均通过。

结果：

```text
lineD_completed = 1
lineD_focused_candidate_count = 10
lineD_family_pass_count = 0
lineD_family_linec_measured_count = 10
```

family 状态：

```text
Rational: ExpressionBlocked
RBF: ExpressionBlocked
Chebyshev: TaskBlocked
Fourier: ExpressionBlocked
Wavelet: TaskBlocked
BSpline: frozen / not run
```

更关键的是，所有 10 个 Line C family summaries 都是 `coupling_collapse`。因此 Line D 不能作为 B15/P3/P4 的接入口。

Line D 后续可以保留，但必须有新的 family-specific hypothesis；不能重复 B7lu/B7lv、B2aa/B2ab、B3an/B3ao、B4v/B4w、B5p/B5q。

---

# 2. 这次实验到底有没有进展？

有，而且是非常关键的机制定位进展。

但它不是那种“指标突然过了”的进展，而是把问题空间显著缩小：

```text
已排除：
  B320 base regression；
  Line D 未跑；
  Line C 全是假阳性；
  safety 过严；
  actuator 完全不能动；
  方向符号错误；
  CouplingR2 不够。

仍成立的 blocker：
  当前 functional objective / selector 无法把 actuator movement 转化为
  stable, loss-agnostic, control-resistant noise/reservoir release。
```

因此，本轮的价值是从“functional update 不 work”推进到更具体的判断：

$$
\boxed{
\text{functional update 的失败来自 value definition / sketch construction / target visibility，}
\text{而不是 base 或 kernel。}
}
$$

这就是为什么看起来慢：现在我们已经不在刷模型指标，而是在验证一个新的训练机制是否真实存在。这个阶段不可能靠一个超参或一条曲线快速结束。

---

# 3. 是否在正确道路上？

方向仍然正确，但 v12.15 明确要求路线升级。

正确的是：

```text
1. 没有降低 gate；
2. 没有把 coupling-only 写成 survivor；
3. 没有把 CE-specific surrogate 写成 official；
4. 没有把 B320 继续小修；
5. 没有按 MNIST / Fashion / KMNIST 调参；
6. 没有因为 K6 单行 actuator gate-open 就打开 P4。
```

需要停止的是：

```text
1. 继续 cotangent count / role grid / lambda grid；
2. 继续 B15-A/B/C/D 当前 objective；
3. 继续 K6 fixed sign / bidirectional selector；
4. 继续用 CouplingR2 作为主要 promotion score；
5. 继续把 fused actuator 当 value source；
6. 继续用 CE-vector 或 per-example CE projector 生成 functional direction。
```

接下来的正确路线应当是：

```text
B320 locked anchor
→ Line C sketch / target reconstruction
→ explicit noise/reservoir spectral objective
→ fused primitive actuator as executor
→ P3 strong-control survivor
→ P4 short-run
→ 10-seed confirmation
```

---

# 4. 离目标还差多远？

## 4.1 离合格 PureKAN base

很近，甚至当前主线可以视为已经有 locked anchor。B320 的 step/memory/task/AUC/Line C nontearing 都已经足够作为 functional update 的实验地基。

## 4.2 离 functional update 成功

仍然远。当前：

```text
official_functional_success = 0
p3_survivor_count = 0
p4_open = 0
B15-A/B/C/D = 0/9
Fixed K6 P3 = 0/9
Bidirectional selector P3 = 0/9
```

functional update 的最小成功还差一个 P3 survivor。更强成功还要 P4 short-run 与 10-seed confirm。

## 4.3 离 next-gen MLP claim

还不能 claim。现在只能说：

```text
B320 作为 efficient PureKAN base 已经站住；
functional update 的机制还没有证明独立收益；
经典 family No-BSpline portfolio 还没有 FamilyPass；
外部 fair / 更大任务还未打开。
```

最终 claim 仍必须满足：

$$
\text{B320 + functional} > \text{B320 + AdamW}
$$

并且击败：

```text
NoOpMatchedOverhead
RandomMatchedNorm
AdamWParallelDirection
SNR-only
ShuffledTarget / ShuffledPayload
MLPAnalogFunctional
```

---

# 5. v12.16 总目标

v12.16 的总目标不是继续找 base，也不是继续刷 CouplingR2，而是：

$$
\boxed{
\text{在 locked B320 上，构建 loss-agnostic explicit signal/reservoir/noise functional mechanism。}
}
$$

更具体地说：

```text
Line A：B320 Anchor Monitor
  只做 no-regression，不小修 base。

Line C：Line C Sketch / Target Reconstruction
  重建 signal/reservoir/noise 的可见 target，让 direction generation 能看到真正目标。

Line I：Fused Primitive Actuator Response Matrix
  把 actuator 当 executor，测它对 explicit target 的响应，不再把 actuator 本身当 value source。

Line B：Functional Candidate Construction
  用 explicit target + response matrix 解 constrained functional update。

Line D：Classic No-BSpline Portfolio Monitor
  只保留有新 hypothesis 的 family repair，不抢 functional 主预算。
```

---

# 6. 核心假设

## H1：当前 Line C sketch 对 CouplingR2 可见，但对 noise/reservoir release 不可见

v12.15 的响应模型已经显示：

```text
CouplingR2_delta 可由 sketch_delta_fro 高相关预测；
NoiseSignalLeak_delta 与 RealSignalReservoirRatio_delta 的 deployable release selector 不成立。
```

H1 认为，当前 sketch construction 把 noise/reservoir basis 放在 direction generator 看不清的位置。解决办法不是增大 update，而是重建 sketch：

```text
multi-window sketch；
role-conditioned sketch；
persistent projector basis；
augmentation / cross-batch consistency sketch；
label-free stable-vs-unstable output subspace；
primitive-role response sketch。
```

H1 成立标准：

$$
\operatorname{Spearman}(\hat s_{noise}, \Delta NoiseSignalLeak) \ge 0.35,
$$

$$
\operatorname{Spearman}(\hat s_{res}, \Delta RealSignalReservoirRatio) \ge 0.35,
$$

并且：

```text
negative_release_precision >= 0.25
negative_release_recall >= 0.25
NoOp / Random false positive = 0
```

这仍是 exploratory；official P3 仍看 hard Pareto，不用 learned black-box selector promotion。

## H2：fused primitive actuator 已具备执行潜力，但需要 response matrix 对齐目标

Continuation 证明：K6 fused primitive actuator 能把 `sketch_delta_fro` 推过 0.01，把 projector angle 推到 4.58 度，但 noise/reservoir release 只有约 -0.002 到 -0.003。

H2 认为，actuator 本身可以作为 executor，但需要先建立：

$$
R_{a\rightarrow g}
=
\frac{\partial g_{LineC}}{\partial a},
$$

其中 $a$ 是 actuator coefficient，$g_{LineC}$ 是 explicit signal/reservoir/noise target 的低维响应。

H2 成立标准：

```text
response_matrix_rank >= 4
condition_number <= 1e4
至少 6/9 rows 有 actuator direction 满足：
  sketch_delta_fro >= 0.01
  projector_angle >= 1 degree
  logit_drift <= 0.05
```

## H3：functional value 应直接约束 noise/reservoir，不应奖励 CouplingR2 单指标

v12.15 已经说明 CouplingR2 很容易打开。v12.16 的 functional objective 不再写成：

$$
\max \Delta CouplingR^2.
$$

而是写成 constrained objective：

$$
\min_a
\quad
\beta \widehat{\Delta NoiseSignalLeak}(a)
+
\gamma \widehat{\Delta RealSignalReservoirRatio}(a)
+
\eta \widehat{TailRisk}(a)
-
\alpha \widehat{\Delta CouplingR^2}(a),
$$

但 promotion 不能只靠该 objective 分数；必须 hard Pareto 过门。

默认权重优先级：

```text
noise / reservoir > control_gap > task safety > coupling
```

如果 coupling 提高但 noise/reservoir 不降，则该 update 仍判失败。

## H4：loss-agnostic 必须是 direction-level 硬审计

v12.16 任何 official functional direction 都必须满足：

```text
loss_agnostic_direction = 1
ce_vector_used_for_direction = 0
label_used_for_direction = 0
permuted_label_used_for_direction = 0
validation_used_for_commit = 0
test_used_for_commit = 0
future_outcome_used_for_commit = 0
dataset_name_used_for_commit = 0
```

允许使用：

```text
当前 train batch 输入 x；
model(x) 的 logits；
label-free random / orthogonal / PCA cotangent；
augmentation or cross-batch consistency without labels；
primitive internal activations / Jacobian sketch；
train-stream unlabeled statistics。
```

CE、Brier、ECE、CEp99、margin 只能作为 audit / no-harm 约束，不能作为 update 方向来源。

---

# 7. 实验总流程

v12.16 分为八个阶段。各阶段可以并行，但 gate 关系必须严格：没有 P3 survivor 不打开 P4；没有 P4 short-run success 不打开 10-seed functional confirm。

```text
P0  B320 anchor no-regression and provenance audit
P1  Line C v2 sketch / target reconstruction
P2  Explicit noise/reservoir spectral target calibration
P3  Fused primitive actuator response matrix
P4  Loss-agnostic functional candidate construction
P5  P3 3x3 strong-control promotion
P6  P4 short-run online event validation
P7  Classic No-BSpline portfolio targeted repair, low budget
P8  v12.16 decision and next route
```

---

# 8. P0：B320 Anchor Monitor

## 8.1 目标

确认当前失败不是 base regression。P0 不允许优化 B320，只允许复用 locked artifact 或做 exact no-regression check。

## 8.2 必须记录

```text
run_id
source_anchor_artifact
source_sha256
hash_match_previous
candidate_id
step_ratio_q90
memory_ratio_q90
mean_delta_vs_mlp
worst_delta_vs_mlp
near_pass_rate
AUC_step_ratio
AUC_time_ratio
ECE_delta
LineC_nontearing_pass
regression_flag
provenance_no_fake
provenance_no_proxy
provenance_no_cpu_offload
```

## 8.3 Gate

P0 pass：

$$
step\_ratio\_{q90} \le 0.50,
$$

$$
memory\_ratio\_{q90} \le 0.20,
$$

$$
mean\_delta \ge 0,
$$

$$
worst\_delta \ge -0.003,
$$

$$
AUC\_step \le 1.00,
\quad
AUC\_time \le 1.00,
$$

$$
LineC\_nontearing\_pass = 1.
$$

如果 P0 fail，停止 functional，先排查 artifact mismatch / runner regression；不调 functional。

---

# 9. P1：Line C v2 Sketch / Target Reconstruction

## 9.1 目标

解决 v12.15 的核心问题：当前 Line C observables 能预测 CouplingR2，但不能预测 noise/reservoir release。P1 要构造更直接可见的 signal/reservoir/noise target。

## 9.2 Sketch family

每个 sketch 都必须 loss-agnostic。

### C2-1：Multi-window gradient sketch

窗口：

```text
window = 3, 5, 10
```

构造：

$$
\hat W_B^{(\Delta)} = \sum_{\tau=t}^{t+\Delta} \Phi_\tau \Phi_\tau^T.
$$

记录短窗与长窗 projector 的稳定性：

$$
Angle(P_{sig}^{(3)}, P_{sig}^{(10)}),
$$

$$
Angle(P_{res}^{(3)}, P_{res}^{(10)}).
$$

### C2-2：Role-conditioned sketch

按 B320 primitive role 分解：

```text
direct
quad
branch
projection-P
readout
mixed-lowrank
```

每个 role 单独构造 sketch，再做 block-diagonal / weighted merge。

目标：找出 noise/reservoir release 是否只在某个 role 中可见。

### C2-3：Stable-vs-unstable output cotangent sketch

用 label-free output covariance：

$$
\Sigma_z = \operatorname{Cov}(f_\theta(x)).
$$

以及 augmentation / bootstrap consistency：

$$
C_{stab} = \operatorname{Cov}(f_\theta(x), f_\theta(\operatorname{aug}(x))).
$$

定义 stable cotangent basis：

$$
V_{stable} = \operatorname{eig}_{top}(C_{stab}),
$$

unstable cotangent basis：

$$
V_{unstable} = \operatorname{eig}_{top}(\Sigma_z - C_{stab}).
$$

不使用标签；只用模型输出和无标签增强。

### C2-4：Persistent projector basis

对多个 batch / split / window 得到的 $P_{sig},P_{res}$ 做 Procrustes alignment，得到 persistent basis：

$$
P_{sig}^{persist},
\quad
P_{res}^{persist}.
$$

如果 $P_{res}$ 在不同 batch 上随机旋转太大，说明当前 reservoir 指标不可作为 direction target，需要先稳定 sketch，而不是继续 update。

## 9.3 必须记录 CSV 字段

文件：`v1216_linec_v2_sketch_targets.csv`

```text
run_id
candidate_id
dataset
seed
split
window
batch_size
sketch_id
sketch_family
sketch_dim
role
cotangent_family
augmentation_used
label_used_for_direction
ce_vector_used_for_direction
signal_rank
reservoir_rank
signal_effective_rank
reservoir_fraction
top_eigen_share
dissipation_condition
signal_projector_stability
reservoir_projector_stability
noise_target_norm
reservoir_target_norm
stable_cotangent_energy
unstable_cotangent_energy
CouplingR2_delta_pred_corr
NoiseSignalLeak_delta_pred_corr
Reservoir_delta_pred_corr
negative_release_precision
negative_release_recall
noop_false_positive
random_false_positive
```

## 9.4 Gate

P1 exploratory pass：

$$
|\rho_{Spearman}(\hat s_{noise}, \Delta NoiseSignalLeak)| \ge 0.30,
$$

$$
|\rho_{Spearman}(\hat s_{res}, \Delta RealSignalReservoirRatio)| \ge 0.30,
$$

并且：

```text
NoOp false positive rows = 0
RandomMatchedNorm false positive rows = 0
negative_release_precision >= 0.20
negative_release_recall >= 0.20
```

P1 strong pass：

$$
|\rho_{Spearman}| \ge 0.40,
$$

```text
negative_release_precision >= 0.30
negative_release_recall >= 0.30
```

如果 P1 fail：

```text
Codex 先尝试：
1. 增加 sketch_dim；
2. 增加 window ensemble；
3. 增加 role-conditioned sketch；
4. 用 persistent projector alignment 降低 projector rotation；
5. 改变 eigen threshold / rank cutoff；
6. 检查 noise/reservoir 指标是否在当前 batch_size 下方差过大。

Codex 禁止：
1. 用 CE vector 构造方向；
2. 按 dataset 调 threshold；
3. 放宽 -0.01 gate；
4. 用 CouplingR2 代替 noise/reservoir。
```

## 9.5 必须可视化

```text
fig_C1_noise_estimator_oos_spearman.svg
fig_C1_reservoir_estimator_oos_spearman.svg
fig_C1_projector_stability_heatmap.svg
fig_C1_signal_reservoir_spectrum.svg
fig_C1_negative_release_precision_recall.svg
fig_C1_noise_reservoir_metric_null_distribution.svg
fig_C1_role_conditioned_release_visibility.svg
```

---

# 10. P2：Explicit Noise / Reservoir Spectral Target Calibration

## 10.1 目标

把 Line C v2 的测量结果转成 functional direction generation 可见的显式 target。

注意：这里 target 是 loss-agnostic proxy target，不是 CE target。

## 10.2 Target definitions

### T1：Noise-null target

定义 unstable / noise-prone cotangent subspace：

$$
P_{unstable}.
$$

目标是减少 update 对该 subspace 的 signal-channel投影：

$$
\Delta_{noise}(a)
=
\|P_{sig}(a)P_{unstable}\|_F^2
-
\|P_{sig}(0)P_{unstable}\|_F^2.
$$

希望：

$$
\Delta_{noise}(a) < 0.
$$

### T2：Reservoir-release target

定义 stable but low-dissipation subspace：

$$
P_{stable-lowW}.
$$

它代表模型输出中稳定但当前训练动力学没有有效移动的方向。目标是让这部分方向进入 signal channel：

$$
\Delta_{res}(a)
=
\|P_{sig}(a)P_{stable-lowW}\|_F^2
-
\|P_{sig}(0)P_{stable-lowW}\|_F^2.
$$

希望：

$$
\Delta_{res}(a) > 0.
$$

注意：official audit 仍用 `RealSignalReservoirRatio_delta <= -0.01` 验证。direction generation 不使用真实标签。

### T3：Coupling support target

Coupling 只作为支持项，而不是主项：

$$
\Delta_{coup}(a) = CouplingR^2(a)-CouplingR^2(0).
$$

要求：

$$
\Delta_{coup}(a) \ge 0.02,
$$

但不能用它单独 promotion。

### T4：Tail / drift safety target

这些是约束，不是收益源：

$$
\max |\Delta logits| \le 0.05,
$$

$$
\Delta CEp99 \le 0.05,
$$

$$
holdout\_loss\_ratio_{CE} \le 1.002,
$$

$$
holdout\_loss\_ratio_{Brier} \le 1.002.
$$

CE / Brier 只作为 safety audit，不参与方向目标构造。

## 10.3 Calibration experiments

对每个 target family 做 finite-difference response：

```text
T1 noise-null only
T2 reservoir-release only
T1+T2 joint
T1+T2+T3 support
T1+T2+T3 with tail constraints
```

每个 target 都使用同一组 actuator basis，避免 confound。

## 10.4 必须记录

文件：`v1216_explicit_target_calibration.csv`

```text
target_id
target_family
dataset
seed
split
window
actuator_basis_id
pred_noise_delta
actual_NoiseSignalLeak_delta
pred_reservoir_delta
actual_RealSignalReservoirRatio_delta
pred_coupling_delta
actual_CouplingR2_delta
pred_tail_delta
actual_CEp99_delta
logit_max_abs_drift
control_gap_vs_best
sign_accuracy_noise
sign_accuracy_reservoir
spearman_noise
spearman_reservoir
```

## 10.5 Gate

P2 pass：

```text
noise sign_accuracy >= 0.65
reservoir sign_accuracy >= 0.65
noise Spearman >= 0.35
reservoir Spearman >= 0.35
至少 3/9 rows 同时有：
  actual_NoiseSignalLeak_delta < -0.005
  actual_RealSignalReservoirRatio_delta < -0.005
  logit drift <= 0.05
```

P2 strong pass：

```text
至少 6/9 rows 同时满足上述 -0.005 exploratory target；
至少 3/9 rows 接近 official -0.01 target。
```

若 P2 fail：

```text
Codex 先尝试：
1. 将 stable-lowW target 从 single-window 改成 persistent basis；
2. 把 target 从 output logits 改成 primitive-role sketch；
3. 增加 target rank，但限制 condition number；
4. 对 noise target 加 orthogonality constraint；
5. 对 reservoir target 加 minimum-effect constraint。

Codex 禁止：
1. 增大 alpha 只追 CouplingR2；
2. 直接使用 CE/per-example label residual；
3. 因单行接近 -0.01 就 promotion。
```

---

# 11. P3：Fused Primitive Actuator Response Matrix

## 11.1 目标

把 actuator 从“候选 value source”改成“执行器”。P3 不问 actuator 自己是不是好，而问它能否执行 explicit target。

## 11.2 Actuator basis

基于 B320 active primitive roles：

```text
A1 DirectRole
A2 QuadRole
A3 BranchRole
A4 ProjectionP
A5 Readout
A6 DirectQuadCoupled
A7 QuadBranchCoupled
A8 ProjectionReadoutCoupled
A9 FusedQuadSign
A10 MixedLowRankPrimitive
A11 PersistentProjectorAlignedPrimitive
```

保留 K6 fused primitive actuator，但不把它作为唯一方向。

## 11.3 Response matrix

对每个 actuator basis $a_j$，测：

$$
R_{ij}=\frac{g_i(\theta+\epsilon a_j)-g_i(\theta)}{\epsilon},
$$

其中 $g_i$ 包括：

```text
pred_noise_target
pred_reservoir_target
CouplingR2
sketch_delta_fro
signal_projector_angle
reservoir_projector_angle
logit_drift
CEp99
Brier
```

## 11.4 必须记录

文件：`v1216_actuator_response_matrix.csv`

```text
run_id
dataset
seed
split
window
actuator_id
role_recipe
basis_dim
epsilon
sketch_delta_fro
signal_projector_angle_deg
reservoir_projector_angle_deg
pred_noise_response
actual_NoiseSignalLeak_delta
pred_reservoir_response
actual_RealSignalReservoirRatio_delta
CouplingR2_delta
logit_max_abs_drift
CEp99_delta
Brier_delta
response_matrix_rank
response_matrix_condition
classification_failure_mode
```

## 11.5 Gate

Actuator response pass：

```text
>= 6/9 rows satisfy:
  response_matrix_rank >= 4
  response_matrix_condition <= 1e4
  exists safe actuator combo with:
    sketch_delta_fro >= 0.01
    projector_angle >= 1 degree
    logit_drift <= 0.05
```

如果 P3 actuator response fail：

```text
Codex 先尝试：
1. 增加 primitive role basis depth，而不是增加 scalar budget；
2. 用 low-rank combinations 替代 single role；
3. 对 response matrix 做 orthogonalization / whitening；
4. 检查 actuator 是否只影响 logits，不影响 gradient sketch；
5. 检查 finite-difference epsilon 是否过小或过大。

Codex 禁止：
1. 仅重复 budget_multiplier；
2. 仅重复 +/- sign selector；
3. 降低 logit drift gate；
4. 用 single gate-open row 打开 P4。
```

## 11.6 可视化

```text
fig_I_response_matrix_heatmap.svg
fig_I_response_rank_condition.svg
fig_I_projector_angle_vs_noise_release.svg
fig_I_sketch_delta_vs_reservoir_release.svg
fig_I_actuator_role_pareto.svg
fig_I_safety_vs_effect_size.svg
```

---

# 12. P4：Loss-Agnostic Functional Candidate Construction

## 12.1 目标

用 explicit target + response matrix 构造真正的 functional candidates。

## 12.2 Candidate families

### B16-A：ExplicitNoiseReservoirQP

在 actuator basis $A$ 中解：

$$
\min_a
\quad
\beta \widehat{\Delta Noise}(a)
+
\gamma \widehat{\Delta Reservoir}(a)
-
\alpha \widehat{\Delta Coupling}(a)
+
\lambda\|a\|^2,
$$

约束：

$$
\widehat{\Delta Noise}(a) \le -\delta_n,
$$

$$
\widehat{\Delta Reservoir}(a) \le -\delta_r,
$$

$$
\widehat{\Delta Coupling}(a) \ge \delta_c,
$$

$$
\widehat{LogitDrift}(a) \le 0.05.
$$

### B16-B：NoiseNullThenReservoirRelease

两阶段：

```text
Stage 1: project update into noise-null subspace；
Stage 2: within remaining subspace maximize reservoir-release proxy；
Stage 3: remove AdamW-parallel component。
```

### B16-C：PersistentStableLowWRelease

只针对 persistent stable-low-dissipation subspace，避免 batch-specific reservoir 旋转。

### B16-D：ActuatorResponseOrthogonalResidual

从 AdamW / SNR control 中去除平行成分，只保留 response-matrix 显示对 noise/reservoir 有独立作用的 residual：

$$
a_{res}=a - \operatorname{Proj}_{\{AdamW,SNR\}}(a).
$$

### B16-E：TargetShuffleNegativeControl

把 explicit target 打乱，但保持 actuator norm，确认收益不是 norm / movement artifact。

## 12.3 Controls

必须同跑：

```text
C0 TaskOnlyAdamW audit control
C1 NoOpMatchedOverhead
C2 RandomMatchedNorm
C3 AdamWParallelDirection
C4 SNR-only
C5 ShuffledTarget
C6 ShuffledActuatorBasis
C7 SignFlippedTarget
C8 MLPAnalogFunctional
C9 SameNormK6FixedSign
```

## 12.4 P3 row gate

每个 dataset-seed row 必须满足：

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

$$
logit\_max\_abs\_drift \le 0.05,
$$

$$
CEp99\_delta \le 0.05,
$$

$$
holdout\_loss\_ratio_{CE} \le 1.002,
$$

$$
holdout\_loss\_ratio_{Brier} \le 1.002.
$$

注意：CE/Brier 仅为 audit，不参与 direction objective。

## 12.5 Promotion gate

允许进入 P4 short-run 的条件：

```text
Strong promotion:
  9/9 rows pass。

Weak promotion:
  >= 8/9 rows pass；
  bootstrap CI lower for control_gap >= 0；
  failure 不集中在同一 dataset；
  no single-seed / single-dataset promotion。
```

## 12.6 必须记录

文件：`v1216_functional_p3_candidates.csv`

```text
candidate_id
candidate_family
dataset
seed
split
window
basis_dim
solver_status
constraint_active_count
functional_norm_ratio
cos_to_adamw
cos_to_snr
adamw_parallel_component_removed
CouplingR2_before
CouplingR2_after
CouplingR2_delta
NoiseSignalLeak_before
NoiseSignalLeak_after
NoiseSignalLeak_delta
RealSignalReservoirRatio_before
RealSignalReservoirRatio_after
RealSignalReservoirRatio_delta
control_gap_vs_best
best_control_id
logit_max_abs_drift
CEp99_delta
Brier_delta
holdout_loss_ratio_CE
holdout_loss_ratio_Brier
loss_agnostic_direction
ce_vector_used_for_direction
label_used_for_direction
validation_used_for_commit
dataset_name_used_for_commit
p3_row_pass
fail_reason
```

## 12.7 可视化

```text
fig_B_p3_pareto_noise_vs_reservoir.svg
fig_B_p3_coupling_vs_noise.svg
fig_B_control_gap_by_candidate.svg
fig_B_fail_reason_waterfall.svg
fig_B_candidate_row_pass_matrix.svg
fig_B_adamw_cos_vs_control_gap.svg
fig_B_target_shuffle_control.svg
```

## 12.8 Failure handling

若所有 candidate 仍失败：

```text
Case 1: CouplingR2 fail 多：
  检查 actuator response matrix effect size，不先调 target 权重。

Case 2: Noise fail 多：
  重新构造 noise-null target，加入 unstable cotangent persistent basis。

Case 3: Reservoir fail 多：
  检查 stable-lowW target 是否与 audit RealSignalReservoirRatio 对齐。

Case 4: control_gap fail 多：
  remove AdamW/SNR parallel component；
  加 target-shuffle / basis-shuffle negative controls；
  如果仍 fail，说明 functional 只是 AdamW 等价运动。

Case 5: logit drift / CEp99 fail 多：
  先修 actuator norm / trust region；
  不允许用 CE vector 造方向。

Case 6: only one dataset/seed pass：
  作为 failure slice 诊断；
  不按 dataset 调参，不 promotion。
```

---

# 13. P5：P4 Short-Run Online Event Validation

P5 只有 P4 有 P3 survivor 才运行。

## 13.1 目标

验证 one-step / cloned P3 survivor 是否能转化为真实连续训练收益。v12.10-v12.15 多次显示 P3 positive 可能不转化成 P4，因此这一阶段必须严格。

## 13.2 训练设置

```text
base = B320 locked
functional_event_interval = 24 steps initially
train_size = 1024
val_size = 512
test_size = 512
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
budget = short-run 3-5 epochs
```

禁止：

```text
runtime validation commit；
dataset-specific threshold；
CE-vector direction；
teacher / distillation；
loss modification；
sampler / class weight。
```

## 13.3 Methods

```text
M0 B320-AdamW baseline
M1 B320-NoOpMatchedOverhead
M2 B320-RandomMatchedNorm
M3 B320-AdamWParallelDirection
M4 B320-SNR-only
M5 B320-B16-best-functional
M6 B320-B16-target-shuffle-control
M7 MLP-analog-functional
```

## 13.4 必须记录

文件：`v1216_p4_short_run.csv`

```text
method
dataset
seed
step
epoch
event_id
event_accepted
candidate_id
functional_norm_ratio
amortized_overhead
train_loss
val_loss
val_acc
test_acc
AUC_step_ratio
AUC_time_ratio
ECE
Brier
CEp99
margin_p10
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
control_gap_online
logit_drift_event
no_fake
no_proxy
```

## 13.5 Gate

P4 short-run pass：

$$
mean\_delta_{acc} \ge -0.003,
$$

$$
AUC\_time_{func} \le AUC\_time_{base},
$$

$$
ECE_{func} \le ECE_{base} + 0.01,
$$

$$
CEp99_{func} \le CEp99_{base} + 0.05,
$$

$$
NoiseSignalLeak_{func} \le NoiseSignalLeak_{base} - 0.01,
$$

$$
RealSignalReservoirRatio_{func} \le RealSignalReservoirRatio_{base} - 0.01,
$$

$$
amortized\_overhead \le 1.05,
$$

and beats all controls on Line C Pareto.

If P4 fail:

```text
1. 若 P3 predicted pass 但 P4 first events fail：做 event-time response drift autopsy。
2. 若 early events pass late events fail：加 cooldown / periodic recalc，不调 dataset。
3. 若 overhead fail：改 event amortization / precomputed response matrix，不改变 functional target。
4. 若 task fail：先看 logit drift / tail，不能用 CE-specific repair。
5. 若 controls match：说明 functional 没独立性，回 P4 objective orthogonalization。
```

## 13.6 可视化

```text
fig_P4_val_loss_step_time.svg
fig_P4_linec_trajectory.svg
fig_P4_event_accept_timeline.svg
fig_P4_noise_reservoir_delta_by_event.svg
fig_P4_control_gap_trajectory.svg
fig_P4_overhead_breakdown.svg
fig_P4_task_tail_noharm.svg
```

---

# 14. Line D：Classic No-BSpline Portfolio Targeted Repair

Line D 不作为 v12.16 主预算，但不能完全删除。B-spline 继续 frozen；其他 family 只有在有明确新 hypothesis 时才跑 targeted repair。

## 14.1 当前状态

```text
BSpline: RejectedForThisVersion / frozen
Rational: focused repair L3 efficient，但 v12.15 focused B7lu/B7lv A4 fail，Line C coupling_collapse
RBF: L3 efficient，但 A4 expression fail，coupling_collapse
Chebyshev: L3/A4 可过，但 A5 task fail，coupling_collapse
Fourier: L3 efficient，但 A4 expression fail，coupling_collapse
Wavelet: L3/A4 可过，但 A5 task fail，coupling_collapse
```

## 14.2 v12.16 只允许的新 hypothesis

### Rational

不再重复 B7lu/B7lv。新 hypothesis：Rational 的 coupling_collapse 来自 output-scale / reservoir trapping，而非 kernel。只允许测试：

```text
Rational-LineCAlignedOutputScale
Rational-StableLowWReadout
Rational-DenSafeTangentButNoCE
```

成功条件：

```text
L3 pass
A4 pass
A5 near >= 0.8
CouplingR2 >= MLP - 0.02
RealSignalReservoirRatio 不高于 B320 + 0.02
```

### RBF / Fourier

当前是 ExpressionBlocked。只允许测试能直接修 A4 的 hypothesis：

```text
RBF compact-local + expression coverage sidecar
Fourier lowfreq + local residual sidecar
```

如果 A4 仍 fail，停止，不进入 A5。

### Chebyshev / Wavelet

当前是 TaskBlocked。只允许测试 task-geometry hypothesis：

```text
Chebyshev degree-energy damping without CE target
Wavelet scale-energy rebalance without label branch
```

如果 A5 fail 仍伴随 coupling_collapse，停止。

## 14.3 Line D Gate

每个 active family 输出：

```text
FamilyPass
FamilyNearPass
ExpressionBlocked
TaskBlocked
GeometryBlocked
RejectedForThisVersion
```

禁止：

```text
Focused MNIST seed0 success promotion；
把 Line D focused candidate 接入 B16/P3，除非它 full 3x3 过 FamilyNearPass；
重新开启 B-spline。
```

---

# 15. 统一 artifact contract

v12.16 必须输出：

```text
v1216_route_decision.json
v1216_anchor_monitor.csv
v1216_linec_v2_sketch_targets.csv
v1216_explicit_target_calibration.csv
v1216_actuator_response_matrix.csv
v1216_functional_p3_candidates.csv
v1216_functional_p3_summary.csv
v1216_p4_short_run.csv, if opened
v1216_failure_table.csv
v1216_loss_agnostic_audit.csv
v1216_classic_family_status.csv
v1216_provenance_audit.csv
v1216_hash_manifest.json
```

每个 CSV 必须含：

```text
run_id
candidate_id
dataset
seed
split
window
stage
method
control_id
loss_agnostic_direction
ce_vector_used_for_direction
label_used_for_direction
validation_used_for_commit
dataset_name_used_for_commit
no_fake
no_proxy
cpu_offload_used
fail_reason
```

---

# 16. 必须生成的可视化总表

```text
fig_A_b320_anchor_monitor.svg
fig_C_sketch_target_estimator_oos.svg
fig_C_projector_stability_heatmap.svg
fig_C_noise_reservoir_null_distribution.svg
fig_C_role_conditioned_visibility.svg
fig_T_explicit_target_pred_actual.svg
fig_I_actuator_response_matrix.svg
fig_I_projector_angle_vs_release.svg
fig_B_p3_pareto_noise_reservoir.svg
fig_B_control_gap_by_candidate.svg
fig_B_fail_reason_waterfall.svg
fig_P4_linec_trajectory.svg, if P4 opened
fig_P4_event_accept_timeline.svg, if P4 opened
fig_D_classic_family_status.svg
```

---

# 17. v12.16 route decision tree

## Route R1：Functional P3 survivor found

条件：

```text
P3 strong/weak promotion pass。
```

下一步：打开 P4 short-run。

## Route R2：Estimator still weak

条件：

```text
Noise / Reservoir estimator OOS Spearman < 0.30
negative release precision / recall = 0
```

下一步：重建 Line C sketch；不跑 P3 candidate。

## Route R3：Actuator response weak

条件：

```text
response_matrix_rank < 4
或者 sketch_delta/projector_angle gate 不足。
```

下一步：更深 primitive instrumentation；不调 objective。

## Route R4：Objective misaligned

条件：

```text
actuator can move sketch/projector；
但 P3 noise/reservoir/control fail。
```

下一步：重定义 explicit target；不再重复 K6 sign / budget。

## Route R5：P3 passes but P4 fails

条件：

```text
P3 pass；P4 short-run fail。
```

下一步：P3-to-P4 event drift autopsy；不直接改 CE / dataset threshold。

## Route R6：Classic family independent progress

条件：

```text
某 family 过 FamilyNearPass 或 FamilyPass。
```

下一步：作为 independent base / diagnostic，不自动接入 functional，除非 Line C nontearing pass。

---

# 18. Codex 执行优先级

为了加速实验，Codex 按以下顺序并行执行：

```text
Wave 1:
  P0 B320 monitor；
  P1 Line C v2 sketch reconstruction；
  P3 actuator response matrix smoke。

Wave 2:
  P2 explicit target calibration；
  P3 full 3x3 response matrix；
  Line D low-budget targeted repair。

Wave 3:
  P4 functional candidates B16-A/B/C/D/E；
  strong controls；
  failure table + visualization。

Wave 4:
  Only if P3 survivor exists：P4 short-run。
```

如果 Wave 1 的 estimator 或 actuator gate 不过，Wave 3 不应盲跑大网格。

---

# 19. 最终说明

v12.15 的结果并不意味着 functional update 方向失败，而是说明旧的 functional update 构造不够本质。当前真正问题是：

$$
\boxed{
\text{怎样在 loss-agnostic 条件下，让 functional update 直接作用于 signal/reservoir/noise 的训练几何，}
\text{而不是只制造 output coupling movement。}
}
$$

v12.16 的计划正是围绕这个问题设计的。它不再小修 B320，不再重复 cotangent/role/lambda 网格，不再把 CouplingR2 当成 promotion score，而是要求 explicit noise/reservoir target、actuator response matrix、strong controls 和 P4 short-run 全链路闭合。
