# DG-KAN v9.2.12 Functional Direction Reconstruction：从 Control-Equivalent 到可证明因果收益的完整实验计划

> 本计划基于 v9.2.11 `Functional Causality Controller` 的真实 terminal route 制定。  
> v9.2.11 的结论非常清楚：**我们已经不能再把问题归因于 LQ base、AdamW、SNR 低成本实现或单步安全性；真正的问题是当前 functional direction 在 paired replay 中与 matched controls 等价，无法产生独特因果收益。**  
> 因此 v9.2.12 不继续微调 SNR 阈值、event stride、step fraction，也不直接打开 full functional training。  
> v9.2.12 的核心目标是重建 functional update 的“方向本身”：让 functional update 不只是安全，而是能在输出空间、hard-mode、CE tail、margin、curvature 或 basis-usage 上产生可测、可重复、超过 controls 的因果效应。

---

## 0. 最新状态与独立判断

### 0.1 v9.2.11 的真实结果

v9.2.11 的 terminal route 是：

```text
route = R5-FunctionalControlEquivalent
base_candidate = LQ-t2-h256
success_v9211_event_causality = false
success_v9211_short_run = false
success_v9211_full_functional = false
success_v9211_external_ready = false
```

本轮 P2 paired event replay 是真实测量，不是占位：

```text
rows = 22680
base candidates = LQ0,LQ1
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
directions = D9,D10,D11,D13,D14,D15,D16
event types = E0,E2,E5
branches = AdamW, real functional, NoOp, Random, Geometry controls
horizons = 1,5,20
```

核心结果：

```text
p2_event_causality_pass_count = 0
P3/P4/P5/P6/P7 = not_run
```

更重要的是，real functional branches 多数 task-safe，但不能按预注册阈值击败 matched controls。D9 在 task-safe projection 后 functional norm 极小，表现为：

```text
safe but mostly neutered
control-equivalent
```

事件控制器也没有形成有效稀疏触发：

```text
E5 high-confidence-SNR coverage = 0
E0 uniform coverage = 1.0
E2 margin-tail coverage = 1.0
```

所以 v9.2.11 的真正结论是：

$$
\boxed{
\text{当前 functional update 安全但无独特因果效应；问题在 direction 与 event definition，而不是 base 或 AdamW。}
}
$$

### 0.2 目标是否达成

没有。

当前已经达成的是：

```text
1. LQ-t2-h256 是 strict FC-PureKAN near-pass base；
2. LQ P4 system gate 已通过；
3. LQ P5 robust near-pass 已通过；
4. low-cost role-level SNR estimator 已经可用；
5. D9 SignalChannelProjection 在 one-step audit 中安全；
6. paired replay infrastructure 已经真实跑通。
```

当前没有达成的是：

```text
1. functional event causality；
2. functional short-run advantage；
3. functional full 10-seed re-entry；
4. robustness / noisy-signal advantage；
5. strong baseline / external-ready；
6. significant Beyond-MLP advantage。
```

因此不能声明：

```text
functional update 成功；
functional update 是 DG-KAN 的核心贡献；
PureKAN 已经显著超过 MLP；
可以进入 external fair 或 Conv/Former extension。
```

### 0.3 现在卡在哪里

当前不是系统实现 blocker。LQ base 的 forward/backward/step/memory 已经进入 P4 gate，functional P1/P2 的低成本 estimator 也可运行。当前也不是 optimizer 小参数 blocker，AdamW-only base 已经 near-pass。当前也不是 “functional 会伤 task” 这个早期问题，因为 D9 在 one-step audit 里已经 task-safe。

真正 blocker 是：

$$
\boxed{
\text{functional direction 没有产生独特的、可重复的函数空间变化。}
}
$$

D9 可能过于保守，task-safe projection 把 functional norm 削得太小；D10-D16 虽然尝试了更多方向，但 paired replay 仍然没有任何 group 通过 causality gate。更深层的问题是：我们现在的 functional direction 大多还是从参数空间启发式出发，然后再用 gate 去过滤。v9.2.12 必须反过来：先定义希望在输出空间产生什么安全变化，再求一个最小参数更新来实现这个变化。

---

## 1. v9.2.12 总体目标

v9.2.12 的整体目标是：

$$
\boxed{
\text{从 parameter-space heuristic functional direction 转向 output-space causal functional direction。}
}
$$

具体来说，本轮要让 functional update 满足五个条件。

### 1.1 Non-neutrality

Functional update 不能只是安全，它必须在 paired replay 中产生非零且可测的输出变化：

$$
\|\Delta z_{\text{functional}}-\Delta z_{\text{AdamW}}\|_2
>
\epsilon_z.
$$

其中 $z$ 是 logits。若 functional 方向经过 projection 后几乎归零，就不能进入 short-run。

### 1.2 Causal superiority over controls

Real functional 必须在同一 checkpoint、同一后续 batch sequence 下优于 matched controls：

```text
NoOpMatchedOverhead
RandomMatchedNorm
ShuffledSNRMask
GeometryD1Control
AdamW-only
```

至少在一个机制指标上优于 best control：

$$
CEp99_{\text{real}}\leq CEp99_{\text{best-control}}-0.05|CEp99_{\text{AdamW}}|,
$$

或：

$$
MarginP10_{\text{real}}\geq MarginP10_{\text{best-control}}+0.02,
$$

或：

$$
Curvature_{\text{real}}\leq0.90Curvature_{\text{best-control}}.
$$

### 1.3 Task safety

Functional update 不得伤害 task：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

One-step / event-level non-harm 要求：

$$
BadEventRate\leq0.05,
$$

$$
HoldoutNonharmFraction\geq0.70.
$$

### 1.4 System affordability

Functional update 要维持 broad system gate：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

强系统目标是：

$$
StepRatio_{q90}\leq1.20.
$$

如果机制有效但只满足 broad gate，仍可作为 functional diagnostic survivor，但不能 external-ready。

### 1.5 Hard-mode relevance

Functional update 的收益必须优先解释 KMNIST miss rows、CE tail、margin tail 或 curvature spike，而不是只在 easy MNIST 上产生微小变化：

$$
\Delta Acc_{\text{KMNIST,functional}}
-
\Delta Acc_{\text{KMNIST,AdamW}}
\geq0.005,
$$

或：

$$
CEp99_{\text{KMNIST,functional}}<CEp99_{\text{KMNIST,AdamW}},
$$

或：

$$
MarginP10_{\text{KMNIST,functional}}>MarginP10_{\text{KMNIST,AdamW}}.
$$

---

## 2. 与总计划的关系

DG-KAN 总计划的终极目标是：

$$
\text{Clean FullEdge PureKAN}
+
\text{Graph-Free Manual Training}
+
\text{Kernel-Native Efficiency}
+
\text{Task-Safe Functional Update}
+
\text{External Fair Advantage}
+
\text{Scalable Extension}.
$$

截至当前，前几项已经进入可用状态：

```text
Clean FC-PureKAN:
  LQ-t2-h256 equivalence audit 已通过。

Graph-free manual training:
  LQ path 使用 manual forward/backward/update。

Kernel-native efficiency:
  LQ P4 已通过。

Task trainability:
  LQ P5 robust near-pass 已通过，但 full pass 未达成。

Functional update:
  当前处于 direction / causality 未闭合阶段。

External fair:
  未打开。
```

因此 v9.2.12 是总计划中的 **functional update 方向重构阶段**。它不是系统优化阶段，也不是 basis 扩展阶段，也不是 external validation 阶段。

---

## 3. 关键理论判断

### 3.1 从 SNR gate 到 signal-causal direction

论文 *A Theory of Generalization in Deep Learning* 给出的 SNR gate 形式是：

$$
\mu_k^2>\frac{\sigma_k^2}{b-1}.
$$

这个条件告诉我们“哪些方向可能是 signal-dominant”，但没有告诉我们“functional 应该朝哪个方向走”。v9.2.9-v9.2.11 已经证明，只靠 role-level SNR 与启发式 geometry direction 不够。

v9.2.12 因此把 functional update 写成：

$$
\Delta\theta_{\text{functional}}
=
J^\dagger
\Delta z_{\text{target}},
$$

或其低秩 / sketch 近似，而不是先在参数空间猜一个 $d_{\text{geometry}}$。

这里：

```text
J:
  当前 batch / holdout / hard-mode 上的 parameter-to-logit Jacobian sketch。

Delta z_target:
  期望的安全输出空间位移。
```

这样 functional update 先有输出空间意义，再通过 PureKAN manual backward / JVP-VJP 求解近似参数更新。

### 3.2 Functional update 不能等同于 task gradient

如果 $\Delta z_{\text{target}}$ 只是 CE gradient，那么 functional update 会退化成 AdamW 的另一个版本。v9.2.11 中 D9 的一个可能问题就是过于接近 AdamW 或经过 projection 后被中和。

因此 v9.2.12 要显式分解：

$$
\Delta z_{\text{target}}
=
\Delta z_{\text{task-safe}}
+
\Delta z_{\text{geometry}}
+
\Delta z_{\text{tail}}
+
\Delta z_{\text{basis}}.
$$

其中 task-safe 不是 task gradient 本身，而是一个约束：

$$
\Delta CE_{\text{holdout}}\leq0.
$$

### 3.3 Functional update 必须做 paired causal proof

只看 final accuracy 太粗，也容易被 random/control 解释掉。v9.2.12 继续以 paired replay 为核心：

$$
\theta_t
\rightarrow
\theta_t+\Delta\theta_{\text{branch}}
\rightarrow
\theta_{t+h}
$$

并在相同 batch sequence 下比较所有 branch。只有 paired replay 有优势，才能进入 short-run。

---

## 4. 核心假设

### H1：当前 functional control-equivalent 的主因是 direction 过弱或被 projection 中和

v9.2.11 中 D9 方向经过 task-safe projection 后 functional norm 很小。H1 假设它不是“不应该 functional”，而是当前 direction 没有足够 output-space effect。

H1 成立标准：

若新的 output-space functional direction 满足：

$$
\frac{\|\Delta z_{\text{functional}}\|}{\|\Delta z_{\text{AdamW}}\|}
\geq0.05,
$$

同时：

$$
BadEventRate\leq0.05,
$$

则说明可以在安全前提下产生非中性功能变化。

### H2：hard-tail output-space target 比 parameter-space curvature direction 更可能产生 causal gain

当前 D10-D16 参数方向没有通过 paired replay。H2 假设直接针对 CE tail / margin tail / logit confidence 的输出空间 target 更有效。

H2 成立标准：

Hard-tail functional 在 paired replay 中满足：

$$
CEp99_{\text{real}}<CEp99_{\text{best-control}},
$$

或：

$$
MarginP10_{\text{real}}>MarginP10_{\text{best-control}},
$$

并保持 task-safe。

### H3：functional direction 需要低维子空间求解，而不是全参数更新

Full parameter direction 容易被 projection 消掉，也容易和 AdamW 重叠。H3 假设 functional update 应该限制在 PureKAN role/basis 子空间：

```text
quadratic coefficient subspace
lift conditioning subspace
output-scale subspace
basis entropy subspace
recent signal subspace
```

H3 成立标准：

子空间 functional 相比 full-parameter functional 满足：

$$
MechanismGain_{\text{subspace}}\geq MechanismGain_{\text{full}},
$$

并且：

$$
Overhead_{\text{subspace}}\leq Overhead_{\text{full}}.
$$

### H4：event controller 需要从 dense coverage 变成 calibrated sparse coverage

v9.2.11 中 E0/E2 coverage = 1，E5 coverage = 0。H4 假设有效 functional event coverage 应在：

$$
0.03\leq Coverage\leq0.15.
$$

H4 成立标准：

calibrated event controller 满足 coverage 区间，并且 paired replay gain 为正。

### H5：如果 output-space directions 仍无法击败 controls，则当前 LQ base 可能没有可提取 functional advantage

这是重要停止条件。如果 v9.2.12 的 output-space functional 仍 control-equivalent，则问题不是 SNR 或 projection，而是当前 LQ-T2 base 的 functional advantage 不明显。此时下一步应回到 basis/primitive design，而不是继续调 functional controller。

---

## 5. Candidate 设计

### 5.1 Base candidates

```text
A0-LQ0-AdamW:
  LQ-t2-h256 AdamW-only reference。

A1-LQ1-FaninScale-AdamW:
  fan-in output scale reference。

A2-QuadraticFeatureMLP:
  diagnostic baseline，防止 LQ family gain 被普通 quadratic feature 解释。
```

### 5.2 Output-space functional directions

```text
O1-HardTailLogitCorrection:
  针对 CEp99 / wrong-confidence tail 的输出空间位移。
  不改变 loss，只产生 one-step update target。

O2-MarginTailExpansion:
  对 margin_p10 低的样本构造安全 margin expansion target。

O3-CalibrationTailCompression:
  针对过高 confidence wrong samples，压缩 logit norm / wrong confidence。

O4-CurvatureOutputFlattening:
  在输出空间降低局部 curvature / local Lipschitz proxy。

O5-BasisEntropyOutputCorrection:
  让输出对不同 lifted basis 更均匀敏感，间接修 basis collapse。

O6-KMNISTHardModeOutputTarget:
  针对 KMNIST hard-mode rows 的 output-space target，不使用 class weight / sampler。
```

### 5.3 Parameter subspace solvers

```text
S1-QuadraticCoeffSubspace:
  只在 T2 coefficient 子空间求解 update。

S2-LiftSubspace:
  只在 identity lift coefficient 子空间求解 update。

S3-OutputLinearSubspace:
  只在 output linear coefficient 子空间求解 update。

S4-LiftPlusQuadraticSubspace:
  lift + T2 coefficient 联合低维子空间。

S5-RecentSignalSubspace:
  最近 coherent gradient top-k 子空间。

S6-OrthogonalToAdamWSubspace:
  去除 AdamW 平行成分，只保留 task-safe 正交 correction。
```

### 5.4 Solvers

```text
SOL0-ProjectedGradient:
  用 J^T Delta_z_target 得到参数方向，再 task-safe projection。

SOL1-LeastSquaresSketch:
  用小规模 sketch 解 min ||J delta - Delta_z_target||^2。

SOL2-ConstrainedLeastSquares:
  加 CE holdout non-increase constraint。

SOL3-TrustRegionQP:
  限制 ||delta|| <= rho ||AdamW step||，并最小化 output target mismatch。

SOL4-AbstainingSolver:
  若 predicted non-harm confidence 低，直接 skip。
```

### 5.5 Event controllers

```text
E1-CalibratedCEp99Tail:
  CEp99 高于 running q75/q90 时触发。

E2-CalibratedMarginTail:
  margin_p10 低于 running q25/q10 时触发。

E3-CalibratedWrongConfidence:
  wrong_confidence_p95 高时触发。

E4-CalibratedCurvatureSpike:
  curvature / local Lipschitz 高时触发。

E5-CalibratedBasisCollapse:
  basis entropy 低或 dominant basis fraction 高时触发。

E6-CompositeSparseEvent:
  多指标加权，目标 coverage 0.03-0.15。
```

### 5.6 Controls

```text
C0-AdamWOnly:
  no functional event。

C1-NoOpMatchedOverhead:
  相同 event 和测量，不改参数。

C2-RandomMatchedNorm:
  同 norm / 同 subspace / 同 event coverage 的随机方向。

C3-ShuffledTarget:
  打乱 output target 和样本对应关系。

C4-ShuffledSNRMask:
  同 active fraction，随机打乱 mask。

C5-InvertedEventController:
  在低风险事件触发，应该更差。

C6-AdamWParallelDirection:
  与 AdamW 平行的同 norm direction，检验是否只是 extra step。
```

---

## 6. 实验阶段

## P0：v9.2.11 结果复现与问题定位

### 目标

复现 v9.2.11 的 R5-ControlEquivalent 结论，确保后续不是在不稳定结果上继续。

### 必须记录

```text
candidate
dataset
seed
direction
event_type
branch
horizon
holdout_loss_delta
CEp99_delta
margin_p10_delta
curvature_delta
functional_norm
projected_norm
norm_after_projection_ratio
cos_with_adamw
cos_with_task_gradient
event_coverage
```

### 判断标准

复现通过：

```text
p2_event_causality_pass_count = 0
D9 norm_after_projection_ratio 很低
E5 coverage = 0 or near 0
E0/E2 coverage high
```

允许数值有随机波动，但 route 应仍为 control-equivalent。

### 可视化

```text
p0_v9211_reproduction_dashboard.svg
p0_functional_norm_after_projection.svg
p0_event_coverage_degeneracy.svg
p0_control_equivalence_matrix.svg
```

---

## P1：Control-equivalence autopsy

### 目标

明确 functional 为什么等价于 controls。P1 不新增新 functional，只分析旧方向。

### 必须记录

```text
direction
event_type
branch
cos_with_adamw
cos_with_random
cos_with_task_gradient
projected_norm_ratio
output_displacement_norm
output_displacement_rank
hard_sample_output_displacement
CEtail_output_displacement
margin_tail_output_displacement
basis_entropy_delta
```

### 判断标准

P1 必须把失败归入至少一种机制：

```text
M1-projection_neutralization:
  norm_after_projection_ratio < 0.10.

M2-adamw_parallel:
  cos_with_adamw > 0.90.

M3-event_degenerate:
  coverage < 0.01 or coverage > 0.90.

M4-output_effect_too_small:
  ||Delta z_functional|| / ||Delta z_AdamW|| < 0.05.

M5-control_equivalent:
  real branch no better than best matched control.
```

### 可视化

```text
p1_failure_mechanism_bar.svg
p1_output_displacement_spectrum.svg
p1_cosine_with_adamw_task.svg
p1_hard_sample_displacement.svg
```

---

## P2：Output-space functional direction factory

### 目标

构造新的 functional directions，并在 one-event / paired replay 中筛选。P2 是本轮核心，不能跳过。

### 设计

每个 candidate 是三元组：

```text
output target Oi
parameter subspace Sj
solver SOLk
```

优先测试：

```text
O1 x S1/S4 x SOL0/SOL2
O2 x S1/S4/S6 x SOL0/SOL2/SOL3
O3 x S3/S4 x SOL2
O4 x S5/S6 x SOL3
O6 x S4/S5 x SOL2/SOL3
```

### 必须记录

```text
target_id
subspace_id
solver_id
dataset
seed
event_type
coverage
target_norm
solved_delta_norm
output_target_fit_r2
output_displacement_norm
bad_event_rate
holdout_nonharm_fraction
CEp99_delta
margin_p10_delta
curvature_delta
ECE_delta
NLL_delta
overhead_event
overhead_amortized
```

### 判断标准

Direction survivor：

$$
BadEventRate\leq0.05,
$$

$$
HoldoutNonharmFraction\geq0.70,
$$

$$
\frac{\|\Delta z_{\text{functional}}\|}{\|\Delta z_{\text{AdamW}}\|}\geq0.05,
$$

$$
Coverage\in[0.03,0.15].
$$

Mechanism survivor：

至少一个：

$$
CEp99_{\text{after}}<CEp99_{\text{before}},
$$

$$
MarginP10_{\text{after}}>MarginP10_{\text{before}},
$$

$$
Curvature_{\text{after}}<Curvature_{\text{before}}.
$$

Control survivor：

real functional 在 paired branch 中优于 best control 至少一个机制指标。

### 可视化

```text
p2_direction_factory_pareto.svg
p2_output_target_fit_vs_gain.svg
p2_safety_coverage_gain_scatter.svg
p2_target_subspace_solver_heatmap.svg
p2_real_vs_control_mechanism_gain.svg
```

---

## P3：Event controller calibration

### 目标

把事件覆盖从 0 或 1 调整到 0.03-0.15，并保持 high precision。

### 方法

对每类 event 计算 rolling quantiles：

```text
CEp99_q75/q90
margin_p10_q25/q10
wrong_conf_p95_q75/q90
curvature_q75/q90
basis_entropy_q25/q10
```

然后选择阈值使训练前半段 coverage 约 0.05、0.10、0.15 三档。

### 必须记录

```text
event_controller_id
threshold_policy
target_coverage
actual_coverage
bad_event_rate
holdout_nonharm
mechanism_gain
overhead_amortized
dataset_coverage
kmnist_coverage
```

### 判断标准

Controller pass：

$$
0.03\leq Coverage\leq0.15,
$$

$$
BadEventRate\leq0.05,
$$

$$
HoldoutNonharm\geq0.70.
$$

若没有 controller 通过，停止，不进入 P4。

### 可视化

```text
p3_event_coverage_calibration.svg
p3_precision_coverage_curve.svg
p3_event_type_by_dataset.svg
p3_kmnist_event_density.svg
```

---

## P4：Paired replay causal confirmation

### 目标

对 P2/P3 survivor 做严格 paired replay，确认因果收益不是 control artifact。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
horizons = 1,5,20,80
branches = AdamW, RealFunctional, NoOp, Random, ShuffledTarget, AdamWParallel
```

### 必须记录

```text
event_id
branch
horizon
dataset
seed
holdout_loss_delta
val_proxy_acc_delta
CEp99_delta
margin_p10_delta
curvature_delta
local_lipschitz_delta
ECE_delta
NLL_delta
basis_entropy_delta
step_time
memory_ratio
```

### 判断标准

Paired causality pass：

RealFunctional 在 horizon 5 或 20 满足：

$$
MetricGain_{\text{real}}-MetricGain_{\text{best-control}}\geq\delta_{\text{mechanism}}.
$$

其中：

```text
CEp99 relative improvement threshold = 0.05
MarginP10 absolute improvement threshold = 0.02
Curvature ratio threshold = 0.90
ECE relative improvement threshold = 0.02
```

并且：

$$
Acc_{\text{real}}\geq Acc_{\text{AdamW}}-0.005.
$$

### 可视化

```text
p4_paired_replay_branch_curves.svg
p4_event_level_causal_effect_forest.svg
p4_horizon_effect_heatmap.svg
p4_real_vs_best_control.svg
```

---

## P5：Short-run multi-step validation

### 目标

验证 paired replay 的 event-level causality 能否转化为 50/240-step 连续训练收益。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
steps = 50,240
base = LQ0,LQ1
functional = top P4 survivor
controls = AdamW, NoOp, Random, ShuffledTarget, AdamWParallel
```

### 必须记录

```text
candidate
dataset
seed
steps
train_loss
holdout_loss
val_acc_proxy
CEp99
margin_p10
ECE_proxy
NLL_proxy
curvature
local_lipschitz
basis_usage_entropy
event_count
event_coverage
bad_event_rate
step_ratio_q50
step_ratio_q90
memory_ratio
```

### 判断标准

Short-run pass：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005,
$$

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

Mechanism pass：

至少一个：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

Strong system pass：

$$
StepRatio_{q90}\leq1.20.
$$

### 可视化

```text
p5_short_run_task_mechanism_pareto.svg
p5_event_timeline.svg
p5_step_ratio_distribution.svg
p5_controls_comparison.svg
```

---

## P6：Full 10-seed functional re-entry

### 目标

只有 P5 broad pass 后打开。验证 functional update 是否能在完整训练中带来 task-safe advantage。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
epochs = 20
base = LQ0/LQ1
functional = P5 survivor
controls = AdamW, NoOp, Random, ShuffledTarget
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
```

### 必须记录

```text
candidate
dataset
seed
val_acc
test_acc
train_acc
val_loss
test_loss
delta_vs_MLP
delta_vs_AdamW_LQ
delta_vs_QuadraticFeatureMLP
near_pass
full_pass
CEp50
CEp90
CEp99
margin_p10
wrong_confidence_p95
ECE
NLL
curvature_ratio
jacobian_norm_ratio
local_lipschitz_ratio
basis_usage_entropy
lift_condition_number
event_count
event_coverage
bad_event_rate
step_ratio_q90
memory_ratio
```

### 判断标准

Task safety：

$$
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005.
$$

Macro improvement：

$$
\Delta Acc_{\text{macro,functional}}
-
\Delta Acc_{\text{macro,AdamW}}
\geq0.003.
$$

KMNIST repair：

$$
\Delta Acc_{\text{KMNIST,functional}}
-
\Delta Acc_{\text{KMNIST,AdamW}}
\geq0.005.
$$

Geometry pass：

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

Tail / calibration pass：

至少一个：

$$
ECE_{\text{functional}}\leq ECE_{\text{AdamW}},
$$

$$
NLL_{\text{functional}}\leq NLL_{\text{AdamW}},
$$

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}}.
$$

System pass：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p6_macro_delta_vs_adamw.svg
p6_kmnist_repair_matrix.svg
p6_seedwise_win_matrix.svg
p6_task_geometry_pareto.svg
p6_ce_tail_margin_panel.svg
p6_ece_nll_panel.svg
p6_controls_comparison.svg
```

---

## P7：Noise / robustness / signal-channel validation

### 目标

验证 functional 不是在 clean setting 上偶然有效，而是符合 signal/noise 分离预期。

### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
candidates = AdamW-only, best functional, NoOp, Random
```

### 必须记录

```text
noise_type
noise_level
dataset
seed
candidate
clean_acc
noisy_acc
acc_drop
CEp99
margin_p10
ECE
NLL
curvature
event_coverage
SNR_active_fraction
bad_event_rate
```

### 判断标准

Robustness pass：

$$
AccDrop_{\text{functional}}\leq AccDrop_{\text{AdamW}}.
$$

Noise-tail pass：

$$
CEp99_{\text{functional}}\leq CEp99_{\text{AdamW}}.
$$

Signal/noise relevance：

$$
EventCoverage_{\text{noise}=0.20}
<
EventCoverage_{\text{noise}=0.05}
$$

或 low-SNR event 被更多 abstain。

### 可视化

```text
p7_noise_robustness_curve.svg
p7_noise_ce_tail.svg
p7_event_coverage_vs_noise.svg
p7_noise_task_geometry_pareto.svg
```

---

## P8：Strong baseline / external-ready gate

### 目标

判断 functional 是否足以进入 external fair validation。

### Baselines

```text
MLP-match
same-shape MLP
hidden-bracket MLP
QuadraticFeatureMLP
LQ AdamW-only
LQ functional
```

### 必须记录

```text
baseline_id
params
flops
forward_ratio
backward_ratio
step_ratio
memory_ratio
dataset
seed
acc
ECE
NLL
CEp99
margin_p10
curvature
local_lipschitz
```

### 判断标准

External-ready if：

```text
functional task-safe = 1
functional mechanism pass = 1
functional control pass = 1
functional system pass = 1
strong baseline challenge pass = 1
```

且至少一个成立：

$$
\Delta Acc_{\text{macro,functional}}\geq0,
$$

或：

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{MLP}},
$$

with no task drop.

### 可视化

```text
p8_strong_baseline_pareto.svg
p8_external_ready_scorecard.svg
p8_lq_functional_vs_quadratic_mlp.svg
```

---

## 7. Required artifacts

```text
run_manifest.json
contract_audit_v9212.csv
p0_v9211_reproduction.csv
p1_control_equivalence_autopsy.csv
p2_output_space_direction_factory.csv
p3_event_controller_calibration.csv
p4_paired_replay_causal_confirmation.csv
p5_short_run_multistep_validation.csv
p6_full_functional_reentry_10seed.csv
p7_noise_robustness_signal_validation.csv
p8_strong_baseline_external_ready.csv
functional_event_trace_v9212.csv
output_target_trace_v9212.csv
paired_replay_branch_trace_v9212.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_base_gate_regression
F3_v9211_reproduction_unstable
F4_projection_neutralization
F5_adamw_parallel_direction
F6_event_degenerate
F7_output_effect_too_small
F8_output_direction_safety_fail
F9_event_controller_no_sparse_pass
F10_paired_replay_causality_fail
F11_short_run_task_drop
F12_short_run_system_fail
F13_full_run_no_macro_gain
F14_kmnist_repair_fail
F15_control_equivalent
F16_quadratic_baseline_explains_gain
F17_noise_robustness_fail
F18_fake_or_proxy_violation
F19_artifact_missing
```

---

## 8. Route decision

### Route cases

```text
R1-OutputDirectionCausalityEstablished:
  output-space functional direction beats matched controls in paired replay.

R2-ShortRunFunctionalMechanismPass:
  short-run shows task-safe mechanism gain.

R3-FullFunctionalAdvantage:
  full 10-seed run improves macro/KMNIST or geometry/tail without task/system harm.

R4-FunctionalMechanismOnly:
  geometry/tail improves but task does not improve.

R5-ControlEquivalentAgain:
  new output-space directions still cannot beat controls.

R6-ProjectionNeutralized:
  all directions become too small after task-safe projection.

R7-EventControllerDegenerate:
  no sparse high-value events found.

R8-SystemOverheadFail:
  direction works but broad system gate fails.

R9-ExternalReady:
  functional passes task/geometry/system/control and strong baseline gates.

R10-ReturnToPrimitiveDesign:
  no functional advantage extractable from current LQ base.
```

### route_decision.json 必须记录

```text
route
base_candidate
best_functional_candidate
best_output_target
best_parameter_subspace
best_solver
best_event_controller
p2_output_direction_pass
p4_paired_replay_pass
p5_short_run_pass
p6_full_reentry_pass
functional_task_safe
functional_mechanism_pass
functional_control_pass
functional_system_broad_pass
functional_system_strong_pass
functional_kmnist_repair_pass
noise_robustness_pass
strong_baseline_challenge_pass
external_ready
primary_blocker
next_required_implementation
success_v9212_output_direction
success_v9212_event_causality
success_v9212_full_functional
success_v9212_external_ready
```

---

## 9. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.11 的 control-equivalent terminal route。

Step 2:
  P1 做 control-equivalence autopsy。
  必须明确是 projection neutralization、AdamW-parallel、event degeneracy、output effect too small 还是 control equivalent。

Step 3:
  P2 构造 output-space functional direction factory。
  这是 v9.2.12 的核心，不允许跳过。

Step 4:
  P3 校准 sparse event controller。
  coverage 必须进入 0.03-0.15。

Step 5:
  P4 做 paired replay causal confirmation。
  只有 P4 过，才进入 short-run。

Step 6:
  P5 short-run multistep validation。

Step 7:
  P6 full 10-seed functional re-entry。

Step 8:
  P7 noise / robustness validation。

Step 9:
  P8 strong baseline / external-ready gate。
```

---

## 10. 停止条件

### Minimum success

```text
P2 output direction safety pass
P3 sparse event controller pass
P4 paired replay causality pass
no fake/proxy
```

### Functional advantage success

```text
Minimum success
+
P5 short-run pass
+
P6 full 10-seed task-safe mechanism gain
+
control pass
```

### External-ready success

```text
Functional advantage success
+
P7 robustness pass
+
P8 strong baseline pass
```

### 失败停止

```text
1. v9.2.11 control-equivalent result cannot be reproduced；
2. all functional directions are projection-neutralized；
3. all directions are AdamW-parallel；
4. no event controller achieves sparse high-precision coverage；
5. output-space directions fail task safety；
6. paired replay cannot beat controls；
7. short-run harms task；
8. full run does not improve macro/KMNIST/geometry/tail；
9. gains are explained by QuadraticFeatureMLP or controls；
10. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 11. 最终解释规则

### Case A：Output-space functional beats controls

可以声明：

```text
Functional update has local causal mechanism under paired replay.
```

但还不能声明 full advantage，除非 P5/P6 通过。

### Case B：Short-run passes but full run fails

必须声明：

```text
Functional has local multi-step mechanism but is not robust across seeds/tasks.
```

### Case C：Full run improves KMNIST / macro safely

可以声明：

```text
Functional update gives task-safe advantage on LQ near-pass base.
```

### Case D：All output-space directions remain control-equivalent

必须声明：

```text
No functional advantage is extractable from current LQ base under tested controllers; next step should return to primitive/basis design.
```

### Case E：functional improves geometry but not task

必须声明：

```text
Functional provides geometry/calibration evidence but not task advantage.
```

---

## 12. 最终建议

v9.2.12 的一句话策略是：

$$
\boxed{
\text{不要再给旧 functional direction 加 gate；先从输出空间定义 functional 目标，再求 task-safe PureKAN 参数更新。}
}
$$

当前最重要的问题不是系统、AdamW 或 LQ 表达力，而是：

```text
1. functional direction 为什么被 projection 中和？
2. functional 是否太接近 AdamW？
3. 是否存在非中性 output-space correction？
4. 是否能找到 coverage 0.03-0.15 的高价值事件？
5. output-space correction 能否在 paired replay 中超过 controls？
6. 这种 local causality 能否扩展到 short-run / full-run？
```

只有这些问题回答完，functional update 才能成为 DG-KAN 的真正核心贡献。
