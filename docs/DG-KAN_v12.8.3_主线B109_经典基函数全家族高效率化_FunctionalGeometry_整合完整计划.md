# DG-KAN v12.8.3：B109/FHQ 主线 + 经典基函数全家族高效率化 + Functional Geometry 整合完整计划

> 版本：v12.8.3 integrated execution plan  
> 目的：把 v12.8 的“B109/FHQ + 经典基函数 + Functional + Manifold-Channel Geometry”总体计划，与 v12.8.2 的“经典基函数 family-specific 高效率化技术路线”整合成一个完整可执行文档。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；当前 task benchmark 继续沿用 CE protocol 以便和历史 artifact 可比，但新 primitive / VJP 不能写死 CE，必须接收一般上游 `dL/dlogits`；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no fake / proxy / CPU offload；official KAN path 不依赖 PyTorch `loss.backward()`；base 未 official qualified 前 functional update 只能 diagnostic。  

---

## 0. 这版计划为什么要重写

上一版计划里有一句话写成了：

```text
在 strict FC-PureKAN 下，找到至少一个 MLP-like efficient base candidate。
```

这个表述不准确。当前项目不是从零开始“找一个 base”，也不是哪个 basis 过了就停。我们现在已有一个接近可用的 anchor：B109/FHQ。它还没有 official base success，但已经足够成为主线 anchor。与此同时，Fourier、Wavelet、B-spline、Chebyshev、RBF/FastKAN、Rational 这些经典 KAN basis 不能因为 B109/FHQ 有进展就被放弃。它们应该作为全家族高效率化支线被系统推进。

因此本版重新锚定层次：

```text
项目总目标：
  在 strict FC-PureKAN base 上，通过 functional update 获得
  表达力不打折、前馈/反传/step 接近 MLP、训练轨迹健康、几何更好的 next-gen MLP candidate。

主线 A：
  继续推进 B109/FHQ anchor。
  目标是把它从 near-qualified anchor 推到 official base，尤其修 AUC-step/AUC-time。

支线 D：
  经典基函数全家族高效率化。
  Fourier / Wavelet / B-spline / Chebyshev / RBF-FastKAN / Rational 全部保留，全部做 family-specific high-efficiency implementation attempt。
  目标不是“至少一个过”，而是每个 family 都要被推进到可审计状态：FamilyPass / FamilyNearPass / KernelBlocked / ExpressionBlocked / TaskBlocked / GeometryBlocked / RejectedForThisVersion。

诊断线 C：
  Manifold-Channel Geometry Diagnostics。
  测 train-probe coupling、signal channel、reservoir、noise leakage、tail stability。

Functional 线 B：
  basis-aware functional update diagnostic。
  base 未过前不 official；base 过后必须击败 strong controls。
```

这不是把思路弄乱，而是把目标和支线的职责区分清楚：

```text
B109/FHQ 是当前最近的工程主线；
经典 basis 是不能放弃的正统 KAN basis portfolio；
Line C 是“几何好”的可计算定义；
Functional update 是最终差异化贡献，不能在 base 未合格时冒充成功。
```

---

## 1. 当前事实锚点

### 1.1 B109/FHQ 当前状态

当前 best anchor 是：

```text
B109b-SimpleFastTaskGeometry-h160-learnableP-absdiag050-classbranch-classgain-identitytailquad030-hingeamp025-temp075, protocolfix final075
```

它已经做到：

```text
F3 step ratio = 0.9659355274876386
memory ratio = 0.5966666666666667
A4 expression pass = true
task mean delta = +0.016927083333333332
worst delta = +0.00390625
near pass = 1.0
ECE ok = true
AUC-step ok = false
AUC-time ok = false
A5 pass = false
```

这说明 B109 不是“效率、表达、最终准确率”单点失败。它真正卡在训练轨迹：

$$
\boxed{
\text{B109 已经过了 per-step efficiency、A4 expression、final task/ECE，}
\text{但没有过 AUC-step / AUC-time。}
}
$$

AUC-step 和 AUC-time 同时失败，说明它不是简单 wall-clock timing 污染，而是 step-indexed validation-loss trajectory 本身不够健康。

### 1.2 B131/F4 当前状态

B131 的价值不是 base success，而是证明低成本 workspace 机制真实存在：

```text
F4 fixed-P workspace step = 0.685358064760707
memory = 0.13
backward ratio = 0.4557250926585637
update ratio = 0.7320725649235357
A4 expression fail = true
```

因此：

$$
\boxed{
\text{B131 不能作为 base，但它提供了必须迁移到 B109-like trajectory 的 cost-reduction mechanism。}
}
$$

### 1.3 B133/B135 排除 dense pair trajectory

B133/B135 证明 localdensepairtraj 即使过 A1/A4，也会 task collapse：

```text
B133b mean delta = -0.10026041666666667
B133b worst delta = -0.333984375
B133b near pass = 0.3333333333333333
B133b ECE fail

B135b mean delta = -0.08550347222222222
B135b worst delta = -0.26171875
B135b near pass = 0.3333333333333333
B135b ECE fail
```

这不是说 pair interaction 本身错，而是说明 dense/random pair trajectory 是 task-hostile。下一步如果继续 interaction，只能做 sparse/local/structured/task-stable interaction，不能回到 dense random pair 小修。

### 1.4 v12.4 经典基函数扫描的真实含义

v12.4 做过一轮多基函数效率优先扫描，包括：

```text
ReLU / RSWAF hinge
RBF / FastKAN
Chebyshev / Legendre
Fourier
Wavelet
B-spline
Rational
```

但这轮只能证明：

$$
\boxed{
\text{在当时实现和 gate 下，这些 basis 没有形成 A1+A4+A5 联合 survivor。}
}
$$

不能推断：

$$
\text{这些 basis 数学上或系统上永远失败。}
$$

尤其 B-spline、Rational、RBF/FastKAN 已有外部高效率化思路；Fourier、Wavelet、Chebyshev 也不能只用 naive PyTorch implementation 判死刑。经典基函数支线必须重启，但必须以 family-specific lower-level kernel / analytic backward / memory planning 的形式重启。

---

## 2. 本版总目标

本版总目标不是“找到至少一个 base”，而是：

$$
\boxed{
\text{以 B109/FHQ 为当前 anchor 主线，}
\text{同时推进所有经典 basis family 的高效率化，}
\text{并在合格或 near-qualified base 上验证 functional update 的独立几何收益。}
}
$$

更具体地说，本轮要完成四件事：

1. **主线 A：B109/FHQ official base closure**  
   把 B109 从 near-qualified 推向 official base。核心不是继续 temperature、epoch、optimizer 小修，而是解决 AUC-step/AUC-time，以及把 B131 的 F4 workspace cost reduction 安全迁移到 B109-like task-stable trajectory。

2. **支线 D：经典 basis 全家族高效率化**  
   Fourier、Wavelet、B-spline、Chebyshev、RBF/FastKAN、Rational 全部保留。每个 family 必须完成 L0 reference、L1 vectorized/GEMM-native、L2 fused forward、L3 fused backward/update 中至少 L2/L3 attempt。不能因为 naive implementation 慢就判死刑。

3. **诊断线 C：Manifold-Channel Geometry Diagnostics**  
   对每个 base candidate 与 functional candidate 测 train-probe coupling、signal/reservoir、noise leakage、hard-tail、ECE/NLL。好几何不是 kernel drift 小，而是训练运动能传到 probe/test-visible signal channel，噪声不进入 signal channel，tail 不坏。

4. **Functional 线 B：basis-aware functional update diagnostic / official re-entry**  
   Functional update 不再做 generic output perturbation，而要按 basis family 设计几何维护：B-spline 的 knot/occupancy，Rational 的 denominator/tangent metric，RBF 的 center-width occupancy，Chebyshev 的 degree-energy，Fourier 的 high-frequency leakage，Wavelet 的 scale/local tail，B109/FHQ 的 signal-projected geometry repair。base 未过前只能 diagnostic；base 过后必须击败 strong controls。

---

## 3. 总体假设

### H-A：B109 的 AUC failure 可以通过 task-stable F4 workspace / semi-fixed P / sparse-local interaction 修复

B109 已经过 final task 和 per-step efficiency，但 AUC-step/time 失败。B131 证明 fixed-P workspace 可以降 cost，但 A4 fail。因此假设：

$$
\boxed{
\text{存在 semi-fixed / block-sparse / low-frequency P update 或 proj-grad fusion，}
\text{能保住 B109 的 A4/task，同时吸收 B131 的 cost reduction。}
}
$$

成立标准：

```text
A1 pass；
A4 pass；
A5 mean/worst/near/ECE 不低于 B109；
AUC-step max < B109 AUC-step max；
step ratio <= 0.85；
memory ratio <= 0.30 或至少 <= 0.60；
Line C nontearing pass。
```

不成立时 Codex 先尝试：

```text
1. fixedP -> semiFixedP-scaleOnly；
2. semiFixedP -> blockSparseP；
3. blockSparseP -> lowFreqPUpdate-K16/K32；
4. learnableP cost 高时只 fusion projGrad，不改变 P trajectory；
5. 不许回到 dense random pair trajectory；
6. 不许用 Fashion-specific rule 修 Fashion failure。
```

### H-D：经典 basis 的旧失败来自实现路径，而不是 basis 本身已经被证伪

经典 basis 的 naive edge-wise expansion 往往构造：

$$
\Phi\in\mathbb{R}^{B\times d_{in}\times K}
$$

甚至隐式扩展到：

$$
\mathbb{R}^{B\times d_{in}\times d_{out}\times K}.
$$

这会导致巨大临时张量、small kernel launch、scatter/gather、backward saved basis、难以和 MLP GEMM 路径竞争。假设：

$$
\boxed{
\text{对每个经典 basis 使用 family-specific fused forward + analytic backward + workspace planning 后，}
\text{至少能判断它真实卡在 kernel、expression、task 还是 geometry，而不是被 naive implementation 误杀。}
}
$$

本轮不是“哪一个过就停”，而是：

$$
\forall f\in\mathcal{F}_{classic},\quad Status(f)\in
\{FamilyPass, FamilyNearPass, KernelBlocked, ExpressionBlocked, TaskBlocked, GeometryBlocked, RejectedForThisVersion\}.
$$

其中：

```text
FamilyPass:
  该 family 至少一个 candidate official 过 A1/A4/A5/C。

FamilyNearPass:
  该 family 有明确 near survivor，且失败项单一、可继续修。

KernelBlocked:
  L2/L3 fused attempt 后仍不能接近 MLP-like efficiency。

ExpressionBlocked:
  efficiency 过，但 A4 expression 不足。

TaskBlocked:
  efficiency + expression 过，但 A5 task/AUC 不足。

GeometryBlocked:
  task/efficiency near pass，但 Line C 显示 coupling collapse / noise leak / tail tearing。

RejectedForThisVersion:
  经过 family-specific implementation + failure repair 后，本版本不再投入，但不代表数学上永久失败。
```

### H-C：AUC-step failure 与 signal-channel geometry 有关

B109 的 final task 好但 AUC-step/time fail，可能说明训练过程中真实任务信号进入 signal channel 太慢，或者 hard-tail 方向在早中期没有被稳定覆盖。假设：

$$
\boxed{
\text{AUC-step failure 可通过 train-probe coupling、RealSignalReservoirRatio、NoiseSignalLeak、CEp99/margin 解释。}
}
$$

如果成立，functional update 的目标应该从“降 loss”改成“把真实信号从 reservoir 推进 signal channel，同时不让噪声进 signal channel”。

### H-B：Functional update 必须 basis-aware 才可能有独立价值

旧 functional update 多次被 strong controls 解释。因此本轮假设：

$$
\boxed{
\text{Functional update 的可行方向不是 generic output perturbation，}
\text{而是 basis-aware, signal-projected, task-safe geometry maintenance。}
}
$$

成立标准不是“one-step loss 下降”，而是：

```text
base qualified 或 near-qualified；
functional task non-harm；
Line C 指标改善；
beats NoOp / RandomMatchedNorm / AdamWParallel / SNR-only / MLP analog；
amortized overhead <= 1.05。
```

---

## 4. 统一 gate 定义

### 4.1 Strict FC-PureKAN invariants

所有 official candidate 必须满足：

```text
nonKAN_param_count = 0
edge_param_coverage = 1.0
no ordinary MLP stem/head/LN learnable params
no teacher / distillation / loss modification
no sampler / class weight
no dataset-name branch
no fake / proxy / CPU offload
manual/fused official path does not rely on PyTorch loss.backward()
```

### 4.2 A1：Efficiency gate

Exploratory：

$$
T_{forward}/T_{MLP}\le 1.35,
$$

$$
T_{backward}/T_{MLP}\le 1.35,
$$

$$
T_{step}/T_{MLP}\le 1.35,
$$

$$
M_{peak}/M_{MLP}\le 1.00.
$$

Official：

$$
T_{forward}/T_{MLP}\le 1.10,
$$

$$
T_{backward}/T_{MLP}\le 1.10,
$$

$$
T_{step}/T_{MLP}\le 1.10,
$$

$$
M_{peak}/M_{MLP}\le 0.80.
$$

B109/FHQ special target：

$$
T_{step,q90}\le 0.85,
$$

$$
M_{q90}\le 0.60,
$$

strong target：

$$
M_{q90}\le 0.30.
$$

### 4.3 A4：Expression gate

所有 base candidate 必须跑 expression battery：

```text
E0 additive
E1 pairwise product
E2 composition
E3 local XOR
E4 high-frequency
E5 noise-stress
E6 rotated pairwise / rotated interaction
E8 random quadratic / global matrix span
```

Official 条件：

```text
A4_expression_pass = 1；
key target delta vs MLP >= -0.005 或按 family-specific threshold；
frozen/global coverage 不出现 collapse；
dead basis / dead channel 不超过 family threshold。
```

### 4.4 A5：Task / AUC / calibration gate

Official 条件：

$$
\Delta Acc_{mean}\ge 0,
$$

$$
\Delta Acc_{worst}\ge -0.010,
$$

$$
near\_pass\_rate\ge 1.0,
$$

$$
AUCstep_{KAN}\le AUCstep_{MLP},
$$

$$
AUCtime_{KAN}\le AUCtime_{MLP},
$$

$$
ECE_{KAN}\le ECE_{MLP}+0.02.
$$

Exploratory 条件可以放宽：

```text
mean delta >= -0.005；
worst delta >= -0.020；
near pass >= 0.80；
AUC-step/time <= 1.05；
ECE <= MLP + 0.03。
```

但 exploratory pass 不能写成 official success。

### 4.5 C：Manifold-Channel nontearing gate

Base candidate 被动几何不能明显坏：

$$
CouplingR^2_{KAN}\ge CouplingR^2_{MLP}-0.02,
$$

$$
NoiseSignalLeak_{KAN}\le NoiseSignalLeak_{MLP}+0.02,
$$

$$
CEp99_{KAN}\le CEp99_{MLP}+\epsilon_{tail},
$$

$$
RealSignalReservoirRatio_{KAN}\le RealSignalReservoirRatio_{MLP}+0.02.
$$

如果 task/efficiency near pass，但 C-line 显示 coupling collapse 或 noise leak 高，则该 candidate 不能称为 good-geometry base。

### 4.6 Functional official gate

Functional update official 成功必须满足：

$$
Acc_{func}\ge Acc_{base}-0.003,
$$

$$
AUCtime_{func}\le AUCtime_{base},
$$

$$
CouplingR^2_{func}\ge CouplingR^2_{base}+0.02,
$$

$$
RealSignalReservoirRatio_{func}\le RealSignalReservoirRatio_{base}-0.02,
$$

$$
NoiseSignalLeak_{func}\le NoiseSignalLeak_{base}-0.02,
$$

$$
T_{amortized,func}/T_{base}\le 1.05.
$$

并且必须击败：

```text
NoOpMatchedOverhead
RandomMatchedNorm
AdamWParallelDirection
SNR-only
GeometryOnlyNoSNR
ShuffledPayload/Event
MLPAnalogGeometryMaintenance
QuadraticFeatureMLPAnalog
```

---

## 5. 四线并行总结构

```text
Line A: B109/FHQ 主线
  A0 AUC autopsy
  A1 F4 workspace transfer
  A2 semi-fixed / block-sparse / low-frequency P
  A3 sparse/local task-stable primitive repair
  A4 B109 official base confirm

Line D: 经典基函数全家族高效率化支线
  D1 B-spline
  D2 Rational
  D3 RBF / FastKAN
  D4 Chebyshev
  D5 Fourier
  D6 Wavelet

Line C: Manifold-Channel Geometry Diagnostics
  C0 train-probe displacement
  C1 signal/reservoir sketch
  C2 real signal reservoir ratio
  C3 noise signal leakage
  C4 kernel drift + coupling interpretation

Line B: Functional update diagnostic / official re-entry
  B0 cloned one-step/five-step audit
  B1 basis-aware functional geometry maintenance
  B2 control matrix
  B3 short-run functional training only after base official/near-official
```

这四条线可以并行执行，但 promotion 必须按 gate 控制。

---

# Line A：B109/FHQ official base closure

## 6. A0：B109 AUC autopsy

### 6.1 目标

B109 当前最大 blocker 是 AUC-step/AUC-time。第一步不是改模型，而是把 AUC failure 分解清楚：是 early / mid / late 失败？是 NLL / CEp99 / margin / calibration / coupling 问题？还是 system timing 污染？

### 6.2 必须记录 CSV

`v1283_b109_auc_autopsy_trace.csv`

```text
run_id
candidate_id
dataset
seed
epoch
step
time_sec
train_loss
val_loss
val_NLL
val_ECE
CEp95
CEp99
margin_mean
margin_p10
wrong_confidence_p95
val_acc
classwise_acc_json
classwise_NLL_json
step_time_ms
forward_time_ms
backward_time_ms
update_time_ms
AUC_step_partial
AUC_time_partial
CouplingR2
RealSignalReservoirRatio
NoiseSignalLeak
```

`v1283_b109_auc_attribution.csv`

```text
run_id
candidate_id
dataset
seed
AUC_step_ratio
AUC_time_ratio
early_AUC_step_ratio
mid_AUC_step_ratio
late_AUC_step_ratio
early_contribution_fraction
mid_contribution_fraction
late_contribution_fraction
step_time_ratio_mean
step_time_ratio_q90
loss_trajectory_failure
system_timing_failure
hard_tail_failure
calibration_failure
margin_failure
coupling_failure
reservoir_failure
noise_leak_failure
```

### 6.3 判断标准

AUC trajectory failure：

```text
AUC_step_ratio > 1.05
and
AUC_time_ratio / AUC_step_ratio <= 1.03
```

System timing failure：

```text
AUC_step_ratio <= 1.05
and
AUC_time_ratio > 1.05
```

Hard-tail failure：

```text
CEp99 or margin_p10 worsens vs MLP by tolerance
and contributes to late_AUC_step_ratio > 1.05
```

Coupling failure：

```text
CouplingR2 < MLP - 0.02
or RealSignalReservoirRatio > MLP + 0.02
```

### 6.4 可视化

```text
fig_v1283_b109_val_loss_vs_step.svg
fig_v1283_b109_val_loss_vs_time.svg
fig_v1283_auc_phase_decomposition.svg
fig_v1283_auc_step_vs_time_scatter.svg
fig_v1283_CEp99_margin_trace.svg
fig_v1283_classwise_NLL_heatmap.svg
fig_v1283_coupling_vs_auc.svg
fig_v1283_real_signal_reservoir_vs_auc.svg
fig_v1283_noise_leak_vs_ECE.svg
```

### 6.5 不满足条件时 Codex 先尝试

如果 AUC-step 是主因：

```text
1. 进入 A3 task-geometry primitive redesign。
2. 优先检查 Line C coupling / signal-reservoir 指标。
3. 不先调 optimizer，不先调 epoch，不按 dataset 分支。
```

如果 AUC-time 是主因：

```text
1. 进入 A1/A2 kernel timing repair。
2. 拆 forward/backward/update/proj_grad。
3. 检查 warmup 与 AUC measurement 是否仍混合。
```

如果 hard-tail 是主因：

```text
1. 设计 task-stable local/sparse primitive。
2. 测 CEp99 / margin_p10 / wrong_confidence。
3. Functional 只做 cloned diagnostic，不 official。
```

---

## 7. A1：F4 workspace transfer to B109-like trajectory

### 7.1 目标

B131 的价值不是 pairtraj，而是 F4 fixed-P workspace cost reduction。A1 要验证这个 cost reduction 是否能迁移到 B109/B124-like task-stable trajectory。

### 7.2 候选

```text
K0-B109-F3-learnableP-reference
K1-B109-F4-fixedP-workspace
K2-B109-F4-semiFixedP-scaleOnly
K3-B109-F4-blockSparseP-fixedDirection-learnScale
K4-B109-F4-lowFreqPUpdate-K16
K5-B109-F3-projGradFused
K6-B109-F3-projGradLowRank
K7-B124-F4-fixedbranch-fixedgain-workspace
```

解释：

```text
fixedP:
  完全固定 P，测试 cost lower bound，但可能影响 expression/task。

semiFixedP-scaleOnly:
  P 的方向固定，只学习每个 projection/channel 的 scale。

blockSparseP:
  P 是 block-sparse/local-structured，减少 proj_grad。

lowFreqPUpdate-K16:
  P 每 16 step 更新一次，其余 step 使用 cached workspace。

projGradFused:
  不改变数学结构，只把 proj_grad 下沉 fused kernel。

projGradLowRank:
  对 proj_grad 使用 rank-r approximation，测试是否能降低 update cost 且不伤 task。
```

### 7.3 必须记录

```text
candidate_id
P_mode
P_trainable_fraction
P_update_period
proj_grad_enabled
proj_grad_time_ms
proj_grad_memory_MB
workspace_reuse_enabled
forward_ratio
backward_ratio
update_ratio
step_ratio_q50
step_ratio_q90
memory_ratio_q90
A1_pass
A4_pass
A5_opened
```

### 7.4 通过标准

Efficiency transfer pass：

$$
T_{step,q90}\le 0.85,
$$

$$
M_{q90}\le 0.30.
$$

Task-safe transfer pass：

```text
A4 pass；
A5 mean/worst/near/ECE 不低于 B109 tolerance；
AUC-step 不比 B109 更差；
Line C nontearing。
```

### 7.5 不满足条件时 Codex 先尝试

```text
fixedP A4 fail:
  尝试 semiFixedP-scaleOnly，不要直接回 dense learnableP。

semiFixedP task fail:
  尝试 blockSparseP 或 lowFreqPUpdate，不调数据集特定阈值。

projGradFused 没降成本:
  拆 proj_grad 维度，做 low-rank 或 chunked reduction。

成本降了但 AUC-step 更差:
  说明 P trajectory 对 signal learning 有作用；保留 learnableP，但只 fusion update/proj_grad。
```

---

## 8. A2：sparse/local task-stable primitive redesign

### 8.1 目标

B133/B135 证明 dense pair trajectory task-hostile。A2 只测试 sparse/local/structured interaction，不再测 dense random pair。

### 8.2 候选

```text
S1 patch-local-pairtraj-fixedP
S2 block-sparse-pairtraj-scaleOnly
S3 low-coherence-sparse-pairtraj
S4 hinge-pair-local-cross
S5 diagonal-plus-local-pair
S6 B109-main + sparse-pair residual small init
S7 B109-main + local Fourier lowfreq residual diagnostic
S8 B109-main + local spline two-bin residual diagnostic
```

### 8.3 必须记录

```text
candidate_id
interaction_type
pair_count
pair_density
locality_type
fixed_or_learnable
init_scale
forward_ratio
backward_ratio
step_ratio
memory_ratio
A4_E1_pairwise
A4_E6_rotated
A4_E8_random_quadratic
A5_mean_delta
A5_worst_delta
A5_near
AUC_step_max
AUC_time_max
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
CEp99
margin_p10
```

### 8.4 判断标准

```text
A1 pass；
A4 pass；
A5 mean/worst/near/ECE 不低于 B109 tolerance；
AUC-step max < B109 AUC-step max；
Line C 不出现 coupling collapse 或 noise leak 上升。
```

### 8.5 不满足条件时 Codex 先尝试

```text
A4 不够:
  增加 structured local coverage，而不是增加 dense random pair。

A4 过但 task collapse:
  降 interaction density；检查 CouplingR2 和 NoiseSignalLeak；保留为 geometry diagnostic。

AUC-step 不改善:
  检查 early/mid/late attribution；若 early fail，考虑 signal-init；若 late fail，考虑 tail-local repair。

效率不过:
  先做 kernel fusion / workspace reuse，不先减 gate。
```

---

## 9. A3：B109 official base confirm

只有 A1/A2 至少一个 candidate 同时达到 A1/A4/A5 exploratory pass，才进入 confirm。

### 9.1 配置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2 initially; 0..4 for confirm; 0..9 only if 5-seed pass
train_size / val_size / test_size = official protocol
profile_steps = official clean architecture timing
Line C = enabled but not mixed into timing
Functional = diagnostic only unless base official
```

### 9.2 通过标准

3-seed official pass：

```text
A1 official pass；
A4 official pass；
A5 official pass；
C nontearing pass；
no fake/proxy/cpu；
strict FC-PureKAN contract pass。
```

5-seed confirm：

```text
paired mean delta >= 0 或 CI 下界 >= -0.002；
AUC-time paired not worse；
ECE not worse；
step q90 <= 1.10；
memory q90 <= 0.80。
```

---

# Line D：经典基函数全家族高效率化支线

## 10. Line D 的共同实现框架

### 10.1 统一层形式

对输入 $x\in\mathbb{R}^{B\times d}$，每个 family 生成 edge basis feature：

$$
\phi_k(x_i),\quad k=1,\dots,K.
$$

输出：

$$
y_o=\sum_{i=1}^{d}\sum_{k=1}^{K}c_{oik}\phi_k(x_i).
$$

但 implementation 不能显式构造 $B\times d\times o\times K$。每个 family 必须选择低成本实现：

```text
GEMM-native flatten path
local active basis tiled kernel
group-shared basis + edge readout
fixed-P workspace
custom Triton/CUDA fused forward/backward
```

### 10.2 实现等级

每个 family 至少推进到 L2/L3 attempt：

```text
L0 reference:
  naive PyTorch/autograd，只用于 correctness，不用于 efficiency claim。

L1 vectorized/GEMM-native:
  避免 [B,out,in,K]，尽量转换成 basis activation + matmul 或 tiled reduction。

L2 fused forward:
  Triton/CUDA forward kernel，消除 basis materialization 和多数 small kernels。

L3 fused backward/update:
  analytic backward，减少 saved tensors，必要时 fused CE delta + parameter gradient + update。
```

Family 不能只停在 L0/L1 就被判死刑。只有 L2/L3 attempted，才能裁决 KernelBlocked 或 RejectedForThisVersion。

### 10.3 共同 artifact contract

每个 family 必须输出：

```text
v1283_family_manifest.csv
v1283_family_gradcheck.csv
v1283_family_efficiency.csv
v1283_family_component_profile.csv
v1283_family_expression.csv
v1283_family_task_triage.csv
v1283_family_geometry_lineC.csv
v1283_family_functional_diagnostic.csv
v1283_family_failure_table.csv
v1283_family_status.json
```

### 10.4 共同 efficiency / task gate

Exploratory：

$$
T_{step}/T_{MLP}\le 1.35,
$$

$$
M_{step}/M_{MLP}\le 1.00.
$$

Official：

$$
T_{step}/T_{MLP}\le 1.10,
$$

$$
M_{step}/M_{MLP}\le 0.80.
$$

Task：

$$
\Delta Acc_{mean}\ge -0.005 \quad \text{exploratory},
$$

$$
\Delta Acc_{mean}\ge 0 \quad \text{official},
$$

$$
\Delta Acc_{worst}\ge -0.010,
$$

$$
AUCtime\le 1.05 \quad \text{exploratory},
$$

$$
AUCtime\le 1.00 \quad \text{official}.
$$

---

## 11. D1：B-spline 高效率化方案

### 11.1 技术难点

B-spline 是原始 KAN 的核心 basis，但 naive 实现存在系统问题：

```text
1. Cox-De Boor recursion 多层递归，不自然映射到 GPU。
2. dense basis materialization 会生成 [B,d,K] 或更大 tensor。
3. coefficient gradient 需要对 knot/bin scatter-add。
4. learnable knots 会带来排序、边界和梯度稳定性问题。
5. cubic B-spline 比 linear spline 更平滑，但 active basis 从 2 个变成 4 个，backward 更重。
```

### 11.2 候选设计

#### D1-A：UniformLinearSplineKAN / LinearKAN-style

先做 uniform linear $C^0$ B-spline。每个输入只激活两个 control points：

$$
q=\left\lfloor\frac{x-x_{min}}{\Delta}\right\rfloor,
$$

$$
t=\frac{x-(x_{min}+q\Delta)}{\Delta},
$$

$$
s(x)=(1-t)c_q+t c_{q+1}.
$$

输出：

$$
y_o=\sum_i\left[(1-t_i)c_{oiq_i}+t_ic_{oi,q_i+1}\right].
$$

实现：

```text
L1:
  torch.gather 两个 knot lookup，作为 correctness reference。

L2:
  Triton tiled kernel：block over B x O，loop over input i，计算 q/t 并累加 y_o。
  不构造 [B,d,K]。

L3:
  backward 分三部分：
    dC: 对 q/q+1 两个 knot 做 block-level reduction，再写回；
    dx: 用 (c_{q+1}-c_q)/Delta 乘上上游 delta；
    optional update: 对 c 做 fused AdamW 或 manual update。
```

优先 fixed uniform knots，不先 learn knots。

#### D1-B：UniformCubicLocalSplineKAN

只有 D1-A 过 A1/A4 后才做 cubic。每个输入激活 4 个 basis。使用 closed-form cardinal cubic，不做 runtime Cox-De Boor recursion。

```text
active_basis = 4；
no recursive dynamic program；
precomputed polynomial coefficients；
tiled forward/backward。
```

#### D1-C：MatrixSplineKAN diagnostic

高阶 B-spline 用 matrix representation / precomputed recursion matrix 做 diagnostic。它不是第一主线，因为如果仍构造 dense basis 会撞 memory；但可以验证 higher-degree spline 是否比 linear spline 有明显 A4/A5 收益。

### 11.3 指标

```text
family = bspline
variant = linear / cubic / matrix
degree
num_knots
active_basis_per_input
out_of_grid_fraction
bin_occupancy_entropy
empty_bin_fraction
knot_grad_collision_rate
second_difference_energy
slope_p95
forward_ratio
backward_ratio
step_ratio
memory_ratio
kernel_count
scatter_atomic_count
A4_expression_pass
A5_task_pass
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
```

### 11.4 判断标准

FamilyPass：

```text
A1 official pass；
A4 pass；
A5 pass；
C nontearing pass。
```

FamilyNearPass：

```text
A1 exploratory pass；
A4 pass；
A5 只差 AUC 或 worst row；
Line C 无严重 tearing。
```

### 11.5 失败后 Codex 先尝试

```text
step 慢:
  degree 3 -> degree 1；
  learnable knots -> fixed uniform knots；
  dense basis -> two-bin local active；
  PyTorch scatter -> Triton block reduction。

expression 不够:
  linear -> cubic；
  增加 knots；
  增加 residual linear base；
  不先改 loss。

task/AUC 不够:
  检查 bin occupancy、empty bins、CouplingR2；
  用 unlabeled train-stream quantile grid，但不按 dataset name 分支。

noise leak 高:
  降 degree 或做 finite-difference functional smoothing diagnostic。
```

---

## 12. D2：Rational / KAT 高效率化方案

### 12.1 技术难点

Rational 的难点不在 basis 数量，而在：

```text
1. denominator safety；
2. division 和 derivative kernel；
3. per-edge rational 太贵；
4. group rational 容易被质疑只是 shared activation + Linear；
5. functional metric 不能用普通 coefficient Sobolev。
```

### 12.2 候选设计

#### D2-A：GroupRationalKAN / GR-KAT-style

使用 group-shared rational function + edge readout。必须标注为 shared-basis FC-PureKAN，不混成 full-edge KAN claim。

Rational 形式：

$$
r_g(x)=\frac{P_g(x)}{Q_g(x)}.
$$

安全 denominator：

$$
Q_g(x)=1+\operatorname{softplus}(\tilde Q_g(x)).
$$

#### D2-B：FullEdgeSmallRational diagnostic

小规模 full-edge rational，只验证 full-edge 表达/几何是否值得，不作为效率主 claim。

#### D2-C：Rational tangent metric functional diagnostic

Rational 的 functional metric 用 tangent metric，而不是普通 coefficient smoothing：

$$
M_g=\sum_qJ_g(x_q)^TJ_g(x_q)+\alpha\sum_qJ'_g(x_q)^TJ'_g(x_q)+\rho I.
$$

#### D2-D：B7hx 后续 hidden-tail narrow update / amplitude repair

B7hx/B7hy 以后，Rational 线不再继续 CE 特化、post-hoc calibration 或 dataset-name branch。允许的下一步是 loss-agnostic hidden-tail repair：

```text
B7hx = B7hn hidden rational tail + generic hidden-tail Triton VJP + Triton AdamW only for hidden-tail readout。
B7hy = B7ho hidden abs tail + generic hidden-tail Triton VJP + Triton AdamW only for hidden-tail readout。
B7hz/B7ia = 在 B7hx 的窄 update 路径上调低 hidden rational tail scale，修复 KMNIST/AUC 轨迹；仍使用通用 dL/dlogits，不绑定 CE 公式。
B7ib/B7ic = 若 hidden tail scale 被 AdamW 自适应步长抵消，则改为 hidden-tail readout optimizer LR 分组；这是通用更新幅度修复，不是 CE-only 调参。
B7id/B7ie = 若降低 hidden-tail readout LR 使 A5 更差，则测试 1.5/2.0 上调 bracket，判断 blocker 是更新不足还是 tail 形状本身不适配。
B7if/B7ig = 若 1.5 改善 accuracy 但恶化 AUC，则测试 1.25/1.35 中间点，寻找 accuracy/AUC 折中。
B7ih/B7ii = 若固定中间点 L3 不稳定，则测试 hidden-tail readout LR 的通用 schedule：先用 1.00 保持早期 AUC 轨迹，再在 epoch 1/2 切到 1.50 获取后期 accuracy；仍只作用于 logits 前参数更新，不绑定 CE 公式。
B7ij/B7ik = 若 LR/schedule 不能越过 ECE/AUC blocker，则把 bounded hidden-rational tail 特征做 per-sample centering / partial centering，并用通用 Triton VJP 支持它；这是 logits 前 feature geometry repair，不是 CE calibration。
B7il/B7im = 若精确 centered VJP 因 mean-coupled hidden gradient 太慢，则测试 stop-gradient centering / partial stop-gradient centering；前向仍是 logits 前 feature geometry repair，反向刻意移除跨 hidden 均值耦合以降低 fused backward cost，不绑定 CE 公式。
B7in/B7io = 若 hidden-tail readout 新参数/新 update state 本身带来 L3 与 A5 blocker，则退回 B7co/B7cp 的已有 w2 readout，把 bounded rational hidden residual `h/(1+abs(h))` 直接折进 hidden state：`h_eff = h + 0.50*h/(1+abs(h))`。该路线不新增 hidden-tail readout 参数，不新增对应 optimizer state，不写 CE 公式，只让 forward/frozen/readout-only manual cache 使用同一 logits 前特征。
B7ip/B7iq = 若 B7in/B7io 的低 linear-residual 版本仍出现 task collapse，则把同一个 hidden residual 移到 B7ej-shaped high linear-residual + hidden-bias 路径；需要在 generic VJP 中缓存 residual 前的 tanh hidden state，并用 `1 + scale/(1+abs(h))^2` 链式导数修正 hidden-bias 梯度。B7ip 用 scale 0.50，B7iq 用 scale 0.25。
B7ir/B7is = 若 B7iq 梯度正确但 L3 仅略高于 1.25，则继续降低同一 hidden residual scale 到 0.15 / 0.10，目标是保持 generic hidden-bias VJP 正确的同时恢复 L3 official margin，再合法进入 A4/A5。
B7it/B7iu = 若 B7ir 打开 L3/A4 但 A5 仍卡在 worst/near/AUC，而 B7iq 只因 L3 略超 gate 被挡，则测试 0.18 / 0.20 midpoint；这是同一 logits 前 hidden trajectory primitive 的幅度 bracket，不改 loss、sampler、class weight、teacher 或 dataset branch。
B7iv/B7iw = 若 B7ir/B7is 已证明 no-new-readout hidden residual 可打开 L3/A4 但 A5 仍失败，则不继续 amplitude/epoch sweep，而是把已有 w2 readout-gradient 与 hidden-residual VJP 合进一个 generic Triton VJP 模式。该 kernel 接收任意上游 `dL/dlogits`，不是 CE-specific backward；B7iv 对应 scale 0.15，B7iw 对应 scale 0.10。
B7ix/B7iy = 若 B7iv/B7iw 证明 generic residual VJP 合法但没有修好 A5，则不继续 CE-specific backward 或同一 rational-residual amplitude sweep，改测新的 single-kernel-friendly smooth trajectory primitive：`h_eff = h + scale*tanh(h)`。该 primitive 仍复用同一个 `w2` readout、无新增 readout/update state，VJP 接收任意上游 `dL/dlogits`；B7ix 对应 scale 0.15，B7iy 对应 scale 0.10。
B7iz/B7ja = 若 B7ix/B7iy 仍复现旧的 KMNIST/Fashion AUC/worst blocker，则将 residual 改为 centered smooth trajectory：`h_eff = h + scale*(tanh(h) - mean_hidden(tanh(h)))`。这显式去除 hidden 维度上的整体偏移，仍无新增 readout/update state，VJP 接收任意上游 `dL/dlogits`，不是 CE-specific backward；B7iz 对应 scale 0.15，B7ja 对应 scale 0.10。
B7jb/B7jc = 若 B7iz/B7ja 说明 hidden-residual 形状本身不是解，则停止 hidden-residual scale/shape sweep，转到非 hidden-residual 的 pair-logit trajectory primitive：在 pair readout logits 写入最终 logits 前做 fused bounded tanh cap，cap 分别为 1.50 / 2.00。该变换不依赖 label 或 CE 公式，VJP 接收任意上游 `dL/dlogits`，用于检验是否能改变 KMNIST/Fashion AUC 形状而不增加 hidden readout/update state。
B7jd/B7je = 若 B7jb/B7jc 合法但仍复现 A5 blocker，则进一步去掉 pair 分支在类别维度上的共同偏移：`pair_centered = pair_logits - mean_class(pair_logits)`，再做 tanh cap。VJP 是 `g_centered - mean_class(g_centered)` 的通用投影链式法则，仍接收任意上游 `dL/dlogits`，不是 CE-specific backward；B7jd/B7je cap 分别为 1.50 / 2.00。
B7jf/B7jg = 若 B7jd/B7je 证明 class-common centering 对 softmax task 几乎等价，则改为 batch-level class centering：`pair_centered = pair_logits - mean_batch(pair_logits)`，再做 tanh cap。这个变换不看 label、不使用 CE 公式，但不是每个 sample 的共同平移，因此可真实改变 logits trajectory；B7jf/B7jg cap 分别为 1.50 / 2.00。
B7jh/B7ji = 若 B7jf 已经打开 L3 efficiency 与 A4，但 A5 task 仍失败，则不改 loss、不改 CE、不做数据集分支，只在同一个 batch-centered pair-logit cap 上补低 cap bracket：1.00 / 1.25。目的是确认 B7jf 的 task blocker 是否来自 pair-logit cap 太松导致 AUC/NLL trajectory 不稳；VJP 仍接收任意 `dL/dlogits`。
B7jj/B7jk = 若低 cap bracket 回到 L3 efficiency fail，则保留 B7jf/B7jg 的 batch-centered forward trajectory，但把 batch mean 视为 stop-gradient 常量：`pair_centered = pair_logits - stopgrad(mean_batch(pair_logits))`。这仍不看 label、不写 CE 公式，VJP 接收任意 `dL/dlogits`；目标是移除跨 batch mean-gradient coupling，降低 backward/update cost，同时验证 batch-centered trajectory 是否仍有 task 修复价值。
B7jl/B7jm = 若 B7jj/B7jk 修回 L3/A4 但 A5 仍失败，则停止 cap-only tuning，把 stop-gradient batch-centered pair-logit trajectory 与一个低成本 hidden residual trajectory 组合：B7jl 用 bounded rational hidden residual `h/(1+|h|)` scale `0.15`，B7jm 用 smooth `tanh(h)` residual scale `0.10`。两者都复用 generic hidden-residual VJP，不读取 label，不写 CE-specific backward，不改变 loss/sampler/class weight/dataset branch。
B7jn/B7jo = 若 B7jl/B7jm 的 hidden residual 组合在 gradcheck 通过但 L3 step 超 gate，则不要继续调 CE 或 cap-only；保留 B7jm 的 smooth hidden residual + stop-gradient batch cap primitive，只把 paircross rank 从 R136 降到 R96 / R80，检查是否能实质降低 fused backward/update cost 并重新打开 A4/A5。该 bracket 仍接收任意 `dL/dlogits`，不读取 label，不改 loss/sampler/class weight/teacher/dataset branch。
B7jp/B7jq = 若 B7jn 修回 L3 efficiency 但 A4 expression 失败，说明 R96 rank 过低；继续在 R96 与 R136 之间做 R112 / R120 midpoint bracket，目标是在不回到 B7jm L3 cost 的前提下恢复 E6/E8 expression capacity。仍保留同一个 loss-agnostic hidden residual + stop-gradient batch cap VJP，不读取 label、不改 CE/loss/sampler/class weight/teacher/dataset branch。
B7jr = 若 B7jq R120 仍过 L3 且 E6/E8 expression 明显恢复但未达 A4，则继续测试 R128 high-rank bracket。若 R128 过 L3 并打开 A4，则按 gate 进入 A5；若 R128 回到 L3 fail 或仍 A4 fail，则停止 rank-only 修复，转向更轻的 expression booster 或 kernel fusion repair。仍不读取 label、不改 CE/loss/sampler/class weight/teacher/dataset branch。
B7js = 若 B7jr R128 仍过 L3 且只剩 E8 frozen R2 未达 A4，则补一个 R132 narrow bracket。R132 需要在 parser 中显式加入 `paircrossr132`；若 R132 不开 A4 或回到 L3 fail，则 rank-only 修复关闭。仍不读取 label、不改 CE/loss/sampler/class weight/teacher/dataset branch。
B7jt = 若 B7js 打开 L3/A4 但 A5 task 失败，尤其出现 KMNIST accuracy/ECE/AUC 伤害，则保留 R132 与 stop-gradient batch cap，只把 smooth hidden tanh residual 从 `0.10` 降到 `0.05`。这是 loss-agnostic trajectory damping，不读 label，不改 CE/loss/sampler/class weight/teacher/dataset branch；若 B7jt 仍 A5 fail 或 A4 回落，则停止该 hidden residual amplitude 小修，转向新的非 CE 任务几何 primitive 或 kernel-level repair。
B7ju = 若 B7jt 降幅后回到 L3 fail 或无法进入 A5，则恢复 hidden tanh residual `0.10`，但改成 per-sample centered hidden tanh residual：`tanh(h) - mean_h(tanh(h))`。目标是保留 B7js 的 expression capacity，同时减少非 class-specific hidden drift 对 task/ECE/AUC 的伤害。仍使用 generic `dL/dlogits` VJP，不读取 label、不改 CE/loss/sampler/class weight/teacher/dataset branch。
B7jv = 若 B7ju 也回到 L3 fail，则回到 B7js 的普通 hidden tanh residual `0.10`，但把 stop-gradient batch pair-logit cap 从 `1.50` 降到 `1.25`。目标是减少 pair-logit 幅度和校准压力，同时保持 R132 expression structure。仍使用 generic `dL/dlogits` VJP，不读取 label、不改 CE/loss/sampler/class weight/teacher/dataset branch。
B7jw = 若 B7jv 仍回到 L3 fail，则移除 hidden residual，只保留 R132 + `crossBatchSGCap150`。这是 B7jj 的 R132 no-hidden-residual bracket，用于隔离 B7js 的 A5 伤害是否来自 hidden residual；仍使用 generic `dL/dlogits` VJP，不读取 label、不改 CE/loss/sampler/class weight/teacher/dataset branch。
B7jx = 若 B7jw 打开 L3/A4 但 A5 与 B7js 同型失败，则保持 no-hidden R132 路径，只把 stop-gradient batch pair-logit cap 从 `1.50` 降到 `1.25`。目标是检查 task/ECE/AUC 伤害是否来自 pair-logit 幅度，而不是 hidden residual；仍使用 generic `dL/dlogits` VJP，不读取 label、不改 CE/loss/sampler/class weight/teacher/dataset branch。
B7jy/B7jz = 若 B7jw/B7jx 都打开 L3/A4 但 A5 仍失败，则停止 cap-only 与 CE-adjacent 小修，改测试一个新的 single-kernel-friendly hidden trajectory primitive：`h_eff = h + s * h*abs(h)/(1+h^2)`，其中 B7jy 用 `s=0.10`，B7jz 用 `s=0.15`。这个 residual 是有界、保号、可在 hidden readout VJP Triton kernel 内计算导数的通用 logits 几何，不读取 label，不写 CE-specific backward，不改 loss/sampler/class weight/teacher/dataset branch；目标是在不增加新 readout/update 参数面的情况下修复 A5 task/ECE/AUC。
B7ka/B7kb = 若 B7jy/B7jz analytic gradcheck 通过但回到 L3 efficiency fail，则说明该新 hidden trajectory 的导数成本仍需要 rank-cost bracket。保持同一 `h*abs(h)/(1+h^2)` residual 与 generic hidden VJP，只把 paircross rank 从 R132 降到 R128 / R120，先重新打开 L3，再按 gate 合法进入 A4/A5；仍不读取 label、不改 CE/loss/sampler/class weight/teacher/dataset branch。
B7kc = 若 B7ka/B7kb 仍 L3 fail，则关闭 `h*abs(h)/(1+h^2)` 分支，回到已经证明 L3/A4 可打开的 bounded rational hidden residual 路线，把 B7iw 的 residual amplitude 从 `0.10` 降到 `0.05`。这是 loss-agnostic hidden trajectory damping，用于检查 A5/ECE/AUC 伤害是否来自 residual 幅度；仍不读取 label、不改 CE/loss/sampler/class weight/teacher/dataset branch。

B314/B315 = 若 B300-B313 的短程 2 seed 出现 A5 pass 但 10 seed confirmation 仍被 Fashion-MNIST AUC-time 卡住，则不要把短程结果写成完成；继续在已经过 F3 efficiency 的 B309/B310 direct-branch ramp 区间做窄 bracket。B314 使用 `directRamp105`，B315 使用 `directRamp110`，都只是 label-free branch-scale epoch schedule：不改 CE、loss、sampler、class weight、teacher、distillation 或 dataset branch，backward/update 仍走已有通用 `dL/dlogits` 路径。

Functional-delta-score-repair = 若 B314/B315 打开 B109 base 后，原 cloned functional strong-control audit 仍显示 no-op/control 占优，则先审计 functional metric 本身是否存在零更新退化。具体做法是：对 delta norm 为 0 的 update 显式记 `CouplingR2=0`，并用 delta 变化量比较 `CouplingR2 + RealSignalReservoirRatio_gain - NoiseSignalLeak_increase - NLL_increase`。新增 `task_delta + signal/orthogonal geometry residual` 的 loss-agnostic diagnostic update，只组合 logits 前方向和已有上游梯度，不写 CE 专用 VJP，不改变 loss/sampler/class weight/teacher/distillation/dataset branch。该分支只可作为 corrected metric diagnostic；除非 corrected metric 被正式纳入 official gate，并且 beats strong controls，否则不能写成 Functional official success。
```

这条线的判断顺序不变：先 L3 full-step + analytic gradcheck，再 A4 expression，再 A5 task；任何 task 正信号都不能绕过前置 gate。

### 12.3 指标

```text
family = rational
variant = group / full_edge_small / tangent_metric
num_groups
num_degree_num
num_degree_den
den_min
den_p01
den_condition
r_prime_p95
r_double_prime_p95
num_update_norm
den_update_norm
num_den_update_ratio
group_function_diversity
group_dead_fraction
forward_ratio
backward_ratio
step_ratio
memory_ratio
A4_expression_pass
A5_task_pass
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
```

### 12.4 判断标准

Safety：

$$
denominator_{p01}>0.25 \quad \text{exploratory},
$$

$$
denominator_{p01}>0.50 \quad \text{official},
$$

$$
denominator_{min}>0.05.
$$

Efficiency：

$$
T_{step}/T_{MLP}\le1.25 \quad \text{exploratory},
$$

$$
T_{step}/T_{MLP}\le1.10 \quad \text{official}.
$$

### 12.5 失败后 Codex 先尝试

```text
NaN / denominator unsafe:
  softplus denominator；
  add denominator floor；
  lower denominator degree；
  variance-preserving init。

step 慢:
  Horner fused eval；
  group rational instead of per-edge rational；
  fused division/derivative；
  use rational_kat_cu/Triton-style extension path if available。

expression 不够:
  increase group count；
  full-edge small diagnostic；
  add identity residual；
  check group function diversity。

geometry bad:
  use tangent metric diagnostic；
  control r_prime_p95 / r_double_prime_p95；
  inspect NoiseSignalLeak。
```

---

## 13. D3：RBF / FastKAN 高效率化方案

### 13.1 技术难点

RBF/FastKAN 的难点是：

```text
1. exp 成本；
2. dense centers；
3. width/center gradient；
4. active center collapse；
5. out-of-grid coverage。
```

Dense RBF 不再作为 final efficiency route，只保留为 mechanism/reference。Line D 只测 compact/local/fused RBF。

### 13.2 候选设计

```text
D3-A FastKAN-fixed-center-RBF
  fixed uniform centers / fixed width，只训练 coeff。

D3-B CompactLocalRBF
  只计算最近 K_active=2 或 4 个 centers。

D3-C RBF-as-spline approximation
  比较 exact exp、exp2 approximation、polynomial exp approximation。

D3-D QuantileCenterRBF diagnostic
  使用 unlabeled train-stream quantile centers，不按 dataset name。
```

RBF：

$$
\phi_k(x)=\exp\left(-\frac{(x-c_k)^2}{2\sigma^2}\right).
$$

### 13.3 指标

```text
family = rbf_fastkan
K_total
K_active
center_type
width_type
exp_count_per_sample
active_center_entropy
dead_center_fraction
width_p01
width_p99
out_of_grid_fraction
approx_exp_error
forward_ratio
backward_ratio
step_ratio
memory_ratio
A4_expression_pass
A5_task_pass
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
```

### 13.4 失败后 Codex 先尝试

```text
exp 慢:
  K_active 降到 2/4；
  exp2 approximation；
  polynomial exp approximation；
  precompute fixed grid。

dead centers 高:
  train-stream quantile centers；
  center occupancy rebalancing diagnostic；
  not dataset-specific。

AUC-step fail:
  检查 RealSignalReservoirRatio 和 active_center_entropy；
  若真实信号困在 reservoir，尝试 center/width init repair。

noise leak 高:
  增大 width；
  降 K；
  functional occupancy smoothing diagnostic。
```

---

## 14. D4：Chebyshev 高效率化方案

### 14.1 技术难点

Chebyshev recurrence 很便宜，主要风险不是速度，而是：

```text
1. global support；
2. 高阶 degree energy 爆；
3. input 必须稳定落在 [-1,1]；
4. 噪声可能直接进入 signal channel；
5. 高阶项可能改善 expression 但伤 AUC/ECE。
```

### 14.2 候选设计

```text
D4-A Cheby-K3-fused
D4-B Cheby-K4-fused
D4-C Cheby-K6-fused
D4-D Cheby-K4-degree-damped
D4-E Cheby-K4-local-normalized
```

recurrence：

$$
T_0(x)=1,
$$

$$
T_1(x)=x,
$$

$$
T_{k+1}(x)=2xT_k(x)-T_{k-1}(x).
$$

### 14.3 指标

```text
family = chebyshev
degree_K
input_norm_type
recurrence_max_abs
recurrence_overflow_count
degree_energy_k
high_degree_energy_ratio
degree_dead_fraction
forward_ratio
backward_ratio
step_ratio
memory_ratio
A4_expression_pass
A5_task_pass
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
CEp99
ECE
```

### 14.4 失败后 Codex 先尝试

```text
high_degree_energy_ratio 高:
  K 降低；
  degree damping；
  spectral whitening；
  input clipping to [-1,1] with fixed normalizer。

expression 不够:
  K3 -> K4 -> K6；
  add linear residual；
  not add dense quadratic first。

task/ECE fail:
  检查 NoiseSignalLeak；
  如果 noise leak 高，Cheby global support 可能不适合该 base form，转 GeometryBlocked。
```

---

## 15. D5：Fourier 高效率化方案

### 15.1 技术难点

Fourier 的难点是：

```text
1. sin/cos 成本；
2. global support；
3. high-frequency noise leakage；
4. learnable frequency/phase 后 backward 复杂；
5. 频谱强可能导致 A4 好看但 task/tail 坏。
```

### 15.2 候选设计

第一版只做 fixed low-frequency，不先学 frequency/phase：

```text
D5-A Fourier-K2-fixedfreq
D5-B Fourier-K4-fixedfreq
D5-C Fourier-K4-lowfreq-plus-linear
D5-D Fourier-K4-fused-sincos
D5-E Fourier-K4-late-enable-diagnostic
```

basis：

$$
\phi_{2k-1}(x)=\sin(k\omega x),
$$

$$
\phi_{2k}(x)=\cos(k\omega x).
$$

### 15.3 指标

```text
family = fourier
K_freq
frequency_type
phase_learnable
sincos_fused
high_freq_energy_ratio
phase_drift
spectral_entropy
forward_ratio
backward_ratio
step_ratio
memory_ratio
A4_expression_pass
A5_task_pass
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
CEp99
ECE
```

### 15.4 失败后 Codex 先尝试

```text
sin/cos 慢:
  K 降到 2；
  fused sincos；
  lowfreq only；
  late-enable high-frequency residual。

NoiseSignalLeak 高:
  high-frequency energy damping diagnostic；
  fixed low freq only；
  Fourier 降级为 high-frequency expression diagnostic。

A4 不够:
  K2 -> K4；
  add linear residual；
  不先学 frequency/phase。
```

---

## 16. D6：Wavelet 高效率化方案

### 16.1 技术难点

Wavelet 容易理论上漂亮但 kernel 很慢。Morlet/MexicanHat 会引入 exp/sin/cos，scale/shift learnable 后 backward 重。因此第一版不从 Morlet 开始。

### 16.2 候选设计

```text
D6-A Haar / Step Wavelet diagnostic
D6-B Triangle / Hat Wavelet
D6-C Compact B-spline-like wavelet
D6-D MexicanHat diagnostic
D6-E Morlet diagnostic only
```

优先 cheap local wavelet：

$$
\psi(x)=\max(1-|x|,0).
$$

这可以复用 B-spline local support kernel。

### 16.3 指标

```text
family = wavelet
wavelet_type
num_scales
num_shifts
scale_learnable
shift_learnable
active_support_count
scale_energy
scale_dead_fraction
local_tail_coverage
forward_ratio
backward_ratio
step_ratio
memory_ratio
A4_expression_pass
A5_task_pass
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
CEp99
margin_p10
ECE
```

### 16.4 失败后 Codex 先尝试

```text
Morlet 慢:
  回退 triangle/hat wavelet；
  不判 Wavelet family 失败。

scale/shift 不稳:
  fixed scale/shift grid；
  only learn coeff；
  quantile-local grid diagnostic。

expression 不够:
  增加 scales；
  增加 local shifts；
  仍不使用 dense full grid。

tail 好但 ECE/AUC 坏:
  检查 NoiseSignalLeak；
  late-enable local wavelet residual diagnostic。
```

---

# Line C：Manifold-Channel Geometry Diagnostics

## 17. Line C 的目标

Line C 不是 optional appendix，而是第三条主线。它回答：

```text
C-Q1: 一个 basis primitive 是否让真实任务信号更容易进入 test-visible signal channel？
C-Q2: 一个 functional update 是否让 train motion 更稳定地预测 probe motion？
C-Q3: 噪声是否被 functional update 推进了 signal channel，从而增加过拟合风险？
C-Q4: AUC-step failure 是 signal reservoir 问题、noise leakage 问题，还是 hard-tail 问题？
```

## 18. 核心定义

从 train stream 中拆两个 batch：

```text
B = update batch
Q = probe batch
```

在窗口 $[t,t+\Delta]$ 内记录 logits 位移：

$$
\Delta U_B=f_{\theta_{t+\Delta}}(B)-f_{\theta_t}(B),
$$

$$
\Delta U_Q=f_{\theta_{t+\Delta}}(Q)-f_{\theta_t}(Q).
$$

拟合 ridge predictor：

$$
A_t=\arg\min_A \|\Delta U_Q-A\Delta U_B\|_F^2+\lambda\|A\|_F^2.
$$

Train-probe coupling：

$$
CouplingR^2=
1-
\frac{\|\Delta U_Q-A_t\Delta U_B\|_F^2}
{\|\Delta U_Q\|_F^2+\epsilon}.
$$

$$
CouplingCorr=
\operatorname{corr}(\operatorname{vec}(A_t\Delta U_B),\operatorname{vec}(\Delta U_Q)).
$$

对 batch 样本构造 projected gradient / logit-Jacobian sketch：

$$
\hat K_{BB}=\Phi\Phi^\top.
$$

窗口累积：

$$
\hat W_B=\sum_{\tau=t}^{t+\Delta}\hat K_{BB}(\tau).
$$

由 $\hat W_B$ 的谱分解得到 signal projector $P_{sig}$ 与 reservoir projector $P_{res}$。

真实信号困在 reservoir 的比例：

$$
RealSignalReservoirRatio=
\frac{\|P_{res}r_{real}\|^2}{\|r_{real}\|^2+\epsilon}.
$$

噪声泄漏进 signal channel 的比例：

$$
NoiseSignalLeak=
\frac{\|P_{sig}r_{noise}\|^2}{\|r_{noise}\|^2+\epsilon}.
$$

Kernel drift 只作为配套指标，不作为越小越好的目标：

$$
KernelDrift=
\frac{\|\hat K(t)-\hat K(0)\|_F}{\|\hat K(0)\|_F+\epsilon}.
$$

解释规则：

```text
good feature learning:
  KernelDrift 高，但 CouplingR2 高、NoiseSignalLeak 低、tail 不坏。

bad tearing:
  KernelDrift 高，CouplingR2 低、NoiseSignalLeak 高、CEp99 / ECE 变坏。

lazy underfit:
  KernelDrift 低，但 RealSignalReservoirRatio 高、task 不下降。
```

## 19. Line C CSV 字段

### 19.1 `v1283_train_probe_coupling.csv`

```text
run_id
candidate_id
basis_family
method
update_type
control_id
dataset
seed
step
window_size
batch_size_B
batch_size_Q
ridge_lambda
train_logit_drift_l2
probe_logit_drift_l2
CouplingR2
CouplingCorr
coupling_residual_norm
coupling_stability_across_splits
KernelDrift
CEp99_delta
ECE_delta
margin_p10_delta
official_gate_open
```

### 19.2 `v1283_signal_reservoir_sketch.csv`

```text
run_id
candidate_id
basis_family
method
update_type
control_id
dataset
seed
step
window_size
sketch_dim
signal_effective_rank
signal_mass_topk
reservoir_fraction
top_eigen_share
dissipation_condition
RealSignalReservoirRatio
NoiseSignalLeak
SNR_positive_fraction
real_noise_gap
official_gate_open
```

### 19.3 `v1283_manifold_channel_diagnostics.csv`

```text
run_id
candidate_id
basis_family
method
dataset
seed
step
base_qualified
functional_official_open
CouplingR2
CouplingCorr
RealSignalReservoirRatio
NoiseSignalLeak
KernelDrift
effective_rank_hidden
basis_occupancy_entropy
basis_dead_fraction
lift_condition_proxy
perturb_logit_drift_p95
CEp99
ECE
NLL
step_time_ratio
memory_ratio
```

## 20. Line C 可视化

```text
fig_v1283_coupling_predicted_vs_actual.svg
fig_v1283_coupling_r2_by_candidate.svg
fig_v1283_coupling_r2_vs_task_delta.svg
fig_v1283_signal_spectrum.svg
fig_v1283_real_signal_reservoir_ratio.svg
fig_v1283_noise_signal_leak.svg
fig_v1283_kernel_drift_vs_coupling.svg
fig_v1283_noise_leak_vs_ECE.svg
fig_v1283_real_signal_reservoir_vs_auc_time.svg
fig_v1283_functional_control_gap_manifold_channel.svg
```

## 21. Line C 失败后 Codex 先尝试

如果 `CouplingR2` 数值不稳定：

```text
1. 增大 probe batch Q；
2. 对 logits 做 classwise centering；
3. 增加 ridge_lambda；
4. 使用 PCA sketch 后再拟合 A_t；
5. 缩短窗口 Delta，避免非线性漂移太大。
```

如果 `NoiseSignalLeak` 过高：

```text
1. 检查 shuffled-label residual 是否构造正确；
2. 检查 P_sig 阈值是否过宽；
3. 改用 top-energy signal projector；
4. 将 functional candidate 改为 AdamW-orthogonal residual direction；
5. 禁止直接降低 gate。
```

如果 `RealSignalReservoirRatio` 过高：

```text
1. 检查 basis / lift effective rank；
2. 检查 signal spectrum 是否 top-eigen collapse；
3. 对 base primitive 尝试 condition-preserving init；
4. 对 functional update 尝试 signal-projected geometry repair；
5. 不按 dataset name 调参。
```

如果 Line C 指标改善但 task 变差：

```text
1. 该 candidate 不 promotion；
2. 检查 logit drift / CEp99 / margin_p10；
3. 降低 lambda 或做 task-safe backtracking；
4. 保留为 diagnostic，不写 official success。
```

---

# Line B：Functional update diagnostic / official re-entry

## 22. Functional update 当前定位

Functional update 必须继续做，但不能抢 base gate。当前状态应写成：

```text
base 未 official qualified：
  functional_status = diagnostic_base_not_qualified

base official qualified 或 near-qualified：
  functional 可以进入 cloned one-step / five-step / short-run audit

functional official：
  必须 task-safe + Line C 改善 + beats strong controls
```

Functional update 不再是 generic task descent，而是：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{task}+\lambda_t\Delta\theta_{geo}.
$$

其中：

```text
Δθ_task:
  AdamW / ManualAdamW / stable task optimizer。

Δθ_geo:
  basis-aware geometry maintenance。

λ_t:
  train-stream probe + backtracking + signal-channel gate 决定。
```

## 23. basis-aware functional update 候选

### 23.1 B109/FHQ functional

```text
FHQ-SignalProjectedGeometry:
  用 Line C signal projector，只允许 geometry correction 进入 signal-safe subspace。

FHQ-AUCStepRepair:
  对 AUC-autopsy 中 early/mid/late 失败的 signal slice 做小步 correction。

FHQ-PConditionRepair:
  对 learnableP / semiFixedP 做 condition-preserving correction。

FHQ-TailStabilityCorrection:
  只在 CEp99 / margin_p10 恶化时触发，必须 task-safe backtracking。
```

### 23.2 B-spline functional

```text
Spline-KnotOccupancyRepair:
  修 empty bins / dead knots。

Spline-FiniteDifferenceSmoothing:
  二阶差分 smoothing，但必须 function-preserving compensation。

Spline-SlopeCap:
  控制 slope_p95，防止 local tearing。
```

### 23.3 Rational functional

```text
Rational-DenominatorSafety:
  增大 denominator_p01，防止 den 接近 0。

Rational-TangentMetricUpdate:
  用 J^T J + alpha J'_x^T J'_x + rho I 的 tangent metric。

Rational-rPrimeControl:
  控制 r_prime_p95 / r_double_prime_p95。
```

### 23.4 RBF/FastKAN functional

```text
RBF-OccupancyRebalance:
  修 active_center_entropy / dead_center_fraction。

RBF-WidthConditionRepair:
  防止 width 太小导致 noise leak。

RBF-CurvatureSmoothCompensated:
  平滑 curvature，但保 task output。
```

### 23.5 Chebyshev functional

```text
Cheby-DegreeEnergyDamping:
  降 high_degree_energy_ratio。

Cheby-SpectralWhitening:
  防止 top degree 垄断 signal channel。

Cheby-NoiseLeakProjection:
  如果 global support 把噪声送进 signal channel，则投影修正。
```

### 23.6 Fourier functional

```text
Fourier-HighFrequencyDamping:
  控制 high_freq_energy_ratio。

Fourier-PhaseStability:
  固定或小步校正 phase drift。

Fourier-NoiseLeakVeto:
  如果 NoiseSignalLeak 高，禁止 high-frequency update。
```

### 23.7 Wavelet functional

```text
Wavelet-ScaleEnergyBalance:
  修 scale_energy / scale_dead_fraction。

Wavelet-LocalTailRepair:
  对 hard-tail local region 做 small correction。

Wavelet-CoverageBalance:
  修 local support coverage 不均。
```

## 24. Functional audit matrix

每个 candidate 都要与以下 controls 同步：

```text
C0 TaskOnlyAdamW
C1 NoOpMatchedOverhead
C2 RandomMatchedNorm
C3 AdamWParallelDirection
C4 SNR-only
C5 GeometryOnlyNoSNR
C6 ShuffledPayload
C7 ShuffledEvent
C8 MLPAnalogGeometryMaintenance
C9 QuadraticFeatureMLPAnalog
C10 BasisFamilyMatchedRandom
```

## 25. Functional one-step / five-step 指标

```text
candidate_id
basis_family
base_candidate
base_qualified
functional_official_open
update_norm
update_over_param_norm
cos_with_adam
cos_with_control_best
train_descent
holdout_descent
holdout_descent_ratio
bad_step_rate
lambda_selected
lambda_backtrack_count
CouplingR2_delta
RealSignalReservoirRatio_delta
NoiseSignalLeak_delta
CEp99_delta
ECE_delta
margin_p10_delta
step_overhead
memory_overhead
beats_NoOp
beats_RandomMatched
beats_AdamWParallel
beats_SNRonly
beats_MLPAnalog
```

## 26. Functional official 判定

Functional official 必须满足：

```text
base_qualified = true；
Acc_func >= Acc_base - 0.003；
AUCtime_func <= AUCtime_base；
ECE_func <= ECE_base + 0.01；
CEp99_func <= CEp99_base + tolerance；
CouplingR2 improves by >= 0.02；
NoiseSignalLeak decreases by >= 0.02；
RealSignalReservoirRatio decreases by >= 0.02；
beats all strong controls；
amortized overhead <= 1.05。
```

如果 base 未过但 functional diagnostic 很好：

```text
status = diagnostic_promising_base_not_qualified
不能写成 official functional success。
```

---

# 27. 全局可视化面板

必须生成以下总图：

### 27.1 Efficiency dashboard

```text
fig_v1283_efficiency_pareto_step_memory.svg
fig_v1283_forward_backward_update_stacked.svg
fig_v1283_kernel_count_by_family.svg
fig_v1283_component_runtime_waterfall.svg
fig_v1283_memory_decomposition_by_family.svg
```

### 27.2 Expression dashboard

```text
fig_v1283_expression_radar_by_family.svg
fig_v1283_E1_pairwise_R2_by_family.svg
fig_v1283_E6_rotated_R2_by_family.svg
fig_v1283_E8_random_quadratic_R2_by_family.svg
fig_v1283_basis_dead_fraction_by_family.svg
```

### 27.3 Task / AUC dashboard

```text
fig_v1283_val_loss_vs_step_by_candidate.svg
fig_v1283_val_loss_vs_time_by_candidate.svg
fig_v1283_auc_step_time_scatter.svg
fig_v1283_mean_worst_near_pass.svg
fig_v1283_ECE_CEp99_margin.svg
```

### 27.4 Geometry dashboard

```text
fig_v1283_coupling_r2_by_family.svg
fig_v1283_signal_spectrum_by_family.svg
fig_v1283_real_signal_reservoir_by_family.svg
fig_v1283_noise_signal_leak_by_family.svg
fig_v1283_kernel_drift_vs_coupling.svg
```

### 27.5 Functional dashboard

```text
fig_v1283_functional_control_gap.svg
fig_v1283_functional_coupling_delta.svg
fig_v1283_functional_noise_leak_delta.svg
fig_v1283_functional_bad_step_rate.svg
fig_v1283_functional_overhead.svg
```

### 27.6 Family status dashboard

```text
fig_v1283_family_status_matrix.svg
fig_v1283_family_failure_taxonomy.svg
fig_v1283_family_pass_nearpass_kernelblocked.svg
```

---

# 28. 统一 artifact contract

每轮必须写：

```text
v1283_run_manifest.csv
v1283_strict_purekan_contract.csv
v1283_efficiency_profile.csv
v1283_component_profile.csv
v1283_grad_correctness.csv
v1283_expression_battery.csv
v1283_task_trace.csv
v1283_task_summary.csv
v1283_auc_autopsy.csv
v1283_lineC_train_probe_coupling.csv
v1283_lineC_signal_reservoir.csv
v1283_lineC_noise_leak.csv
v1283_functional_one_step.csv
v1283_functional_five_step.csv
v1283_functional_controls.csv
v1283_family_status.csv
v1283_failure_table.csv
v1283_route_decision.json
```

`v1283_route_decision.json` 至少包含：

```text
route
base_qualified
functional_open
best_anchor_candidate
best_classic_family_candidates
family_status_summary
lineC_pass_summary
functional_control_resistant_pass
next_recommended_action
no_fake
fake_data_used
proxy_row_used
cpu_offload_used
```

---

# 29. 并行执行计划

为了加快实验，Codex 不要串行等全部完成。按下面并行批次执行。

## Batch 0：contract / smoke / profiler scaffold

并行执行：

```text
A0-B109 AUC autopsy scaffold
D-common family manifest + L0/L1 correctness scaffold
C-line coupling recorder scaffold
B-line functional control matrix scaffold
```

通过条件：所有 CSV 能写，no fake/proxy/cpu，no dataset branch。

## Batch 1：B109 主线推进

并行跑：

```text
K0-B109 reference
K1-B109-F4-fixedP
K2-B109-semiFixedP-scaleOnly
K3-B109-blockSparseP
K4-B109-lowFreqPUpdate-K16
K5-B109-projGradFused
```

同时跑 Line C on B109。

## Batch 2：经典 basis microbench

并行跑：

```text
D1-A UniformLinearSplineKAN
D2-A GroupRationalKAN
D3-A FastKAN-fixed-center-RBF
D4-A Cheby-K3/K4
D5-A Fourier-K2/K4 fixedfreq
D6-A HatWavelet
```

只做 correctness + efficiency + component profile。通过 exploratory efficiency 才进入 Batch 3。

## Batch 3：classic basis expression + task triage

对 Batch 2 survivor / near survivor 跑：

```text
A4 expression battery
A5 3 datasets x seeds 0,1,2 short triage
Line C passive diagnostics
```

## Batch 4：family-specific repair

按 family failure 自动触发 repair：

```text
B-spline: linear -> cubic / Triton scatter reduction
Rational: denominator safety / Horner fused / group count
RBF: active centers / exp approximation / quantile center
Chebyshev: degree damping / K adjust / input norm
Fourier: lower K / fused sincos / highfreq damping
Wavelet: hat -> compact wavelet / fixed scale grid
```

## Batch 5：Functional diagnostic

只在以下 candidate 上并行做 cloned diagnostic：

```text
B109 reference and B109 repair candidates；
classic family FamilyPass / FamilyNearPass candidates；
MLP analog controls；
QuadraticFeatureMLP analog controls。
```

## Batch 6：confirm

只有 official candidate 才跑：

```text
5-seed confirm；
10-seed confirm only if 5-seed pass；
external fair later，不在本轮打开。
```

---

# 30. 全局失败决策树

## Case A：B109 AUC-step 仍失败，但 Line C 找到原因

```text
如果 RealSignalReservoirRatio 高：
  设计 signal-projected primitive / functional diagnostic。

如果 NoiseSignalLeak 高：
  限制 high-frequency/global support update；做 safety projection。

如果 CEp99 / margin tail 失败：
  sparse/local tail primitive；不按 dataset 调参。
```

## Case B：B109 F4 transfer 伤 A4/task

```text
说明 P trajectory 对 expression/task 必要。
保留 learnableP，但 fusion projGrad/update。
转 lowFreqPUpdate，而不是完全 fixedP。
```

## Case C：经典 family efficiency 失败

```text
确认已完成 L2/L3 attempt。
如果只 L0/L1 失败，不能判 family 失败。
若 L2/L3 仍 step > 1.35，则标 KernelBlocked，并写具体 bottleneck。
```

## Case D：经典 family expression 失败

```text
增加 minimal capacity，而不是先加 dense channel。
B-spline: degree/knots；
Rational: group count / degree；
RBF: active centers；
Cheby: K；
Fourier: K lowfreq；
Wavelet: scale/shift coverage。
```

## Case E：经典 family task 失败

```text
先查 AUC-step/time、CEp99、ECE、Line C。
不能直接调 dataset-specific schedule。
若 global support family noise leak 高，标 GeometryBlocked。
```

## Case F：Functional 改善 Line C 但 task 变差

```text
不 promotion。
降低 lambda / task-safe backtracking。
检查 logit drift / CEp99 / margin。
```

## Case G：Functional task-safe 但不 beat controls

```text
不能 official。
记录为 diagnostic。
转 basis-aware direction 或 signal-projected residual，不调低 controls。
```

---

# 31. v12.8.3 的成功定义

本轮完成时，不要求所有路线都 full success，但必须有清晰状态。

## 31.1 B109 主线成功

```text
B109-like candidate official base qualified：
  A1=1, A4=1, A5=1, C=1。
```

或至少：

```text
B109-like candidate near pass，且唯一 blocker 可定位为 AUC-step / Line C 具体机制。
```

## 31.2 经典 basis 支线成功

对全部六个 family：

```text
B-spline
Rational
RBF/FastKAN
Chebyshev
Fourier
Wavelet
```

必须输出：

```text
FamilyPass / FamilyNearPass / KernelBlocked / ExpressionBlocked / TaskBlocked / GeometryBlocked / RejectedForThisVersion
```

并且每个 blocked family 有：

```text
technical bottleneck；
attempted L2/L3 implementation；
metrics evidence；
next repair suggestion。
```

理想目标是六个 family 全部达到 FamilyPass 或 FamilyNearPass；但如果未达成，不能写成“basis 失败”，必须写成可审计 blocker。

## 31.3 Line C 成功

```text
Line C 能稳定输出 CouplingR2 / RealSignalReservoirRatio / NoiseSignalLeak；
能解释至少一个 AUC-step / task failure；
能作为 base promotion 的 nontearing gate。
```

## 31.4 Functional 成功

短期成功：

```text
在 B109-like 或 classic FamilyNearPass base 上，functional diagnostic task-safe，并在 Line C 至少一项指标改善。
```

Official 成功：

```text
base qualified；
functional task non-harm；
Line C 指标改善；
beats strong controls；
amortized overhead <= 1.05。
```

---

# 32. 不允许做的事情

```text
1. 不降低 A1/A4/A5/Line C/Functional gate。
2. 不按 MNIST/Fashion-MNIST/KMNIST 名称设计 controller 或 threshold。
3. 不用 teacher、distillation、modified loss、sampler、class weight。
4. 不把 diagnostic 写成 official success。
5. 不把 L0/L1 naive implementation 失败写成 family 失败。
6. 不让 functional update 拯救未合格 base 并写成成功。
7. 不继续 temperature / epoch / optimizer wrapper 小修作为主线。
8. 不继续 dense random pair trajectory。
9. 不把 “一个 family pass” 当作 classic basis portfolio 完成。
10. 不用 kernel drift 小作为好几何唯一标准。
```

---

# 33. 给 Codex 的执行摘要

Codex 下一步按以下优先级执行。

```text
P0:
  建 v1283 unified runner scaffold。
  输出 run_manifest、strict contract、family manifest、Line C recorder、functional controls scaffold。

P1:
  B109 AUC autopsy + F4 transfer candidates 并行。
  重点看 AUC-step vs AUC-time、Line C、CEp99/margin。

P2:
  经典 basis microbench 六家族并行：
    UniformLinearSplineKAN
    GroupRationalKAN
    FastKAN-fixed-center-RBF
    Cheby-K3/K4
    Fourier-K2/K4-fixedfreq
    HatWavelet
  每家必须至少 L0 correctness + L1/L2 efficiency attempt。

P3:
  对 efficiency survivor 跑 A4 expression 和 A5 task triage。

P4:
  family-specific repair：
    B-spline: local support / scatter reduction / cubic only if needed。
    Rational: safe denominator / Horner fused / group count；B7hx 后只允许 hidden-tail narrow update、true lower-level update fusion 或 loss-agnostic amplitude/trajectory primitive，不允许 CE-only calibration。
    RBF: active centers / exp approximation / quantile centers。
    Cheby: degree damping / input norm。
    Fourier: fused sincos / high-frequency damping。
    Wavelet: hat/local wavelet before Morlet。

P5:
  Line C diagnostics 全候选并行，不作为 optional。

P6:
  Functional cloned one-step/five-step diagnostic；base 未过时 official_gate_open=0。

P7:
  只有 A1/A4/A5/C 都过的 candidate 进入 5-seed confirm。
```

最终路线判断用一句话：

$$
\boxed{
\text{B109/FHQ 继续做 official base closure；}
\text{经典 basis 全家族继续做 high-efficiency implementation；}
\text{Line C 定义几何好；}
\text{Functional update 在合格 base 上证明独立贡献。}
}
$$
