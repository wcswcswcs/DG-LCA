# DG-KAN v12.14：B320 Locked Loss-Agnostic Functional Mechanism Rebuild + No-BSpline 支线计划

> 版本：v12.14 execution plan  
> 生成目的：基于 v12.13 的真实结果，重新判断当前进展、核心 blocker 和下一步实验路线。  
> 关键修正：**functional update 必须是 loss-agnostic，不允许针对 CE 设计方向。CE / ECE / CEp99 只能作为审计指标或坏化约束，不能作为 functional direction 的目标或上游 surrogate。**  
> 公式格式：Typora 友好，仅使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；B320 anchor locked；no teacher；no distillation；no loss modification；no sampler/class weight；no dataset-name branch；no validation/test/future outcome at commit time；no fake/proxy/CPU offload；B-spline frozen；functional direction generator 不使用 CE vector / label-loss VJP / permuted-label CE surrogate。

---

## 0. 一句话判断

v12.13 **不是没进展**。它完成了三件重要的事情：

```text
1. B320 base 已经锁定，不再是当前 blocker。
2. Line C calibration 可信：NoOp / RandomMatchedNorm 没有 false positive。
3. 旧 functional 机制被更清楚地证伪，并且发现了一个新的、但很弱的 loss-agnostic 局部信号。
```

但 v12.13 **没有 functional success**。当前 route 仍是：

```text
R3-FunctionalMechanismPartialOnly
official_functional_success = 0
P4_short_run_open = 0
```

最重要的结论不是“functional 没希望”，而是：

$$
\boxed{
\text{functional update 的旧方向可以打开 coupling，}
\text{但不能稳定改变 signal/reservoir/noise 的谱结构。}
}
$$

因此下一步不是继续调 BM1/BM2/BM3/BM5 的强度、系数、窗口或事件频率，而是重新设计 **loss-agnostic spectral / cotangent / primitive-level gradient-sketch shaping mechanism**。

---

## 1. 当前事实锚点

### 1.1 B320 已经是 locked anchor

v12.13 的 B320 指标为：

```text
B320_anchor_locked = 1
step_ratio_q90 = 0.39931987348441034
memory_ratio_q90 = 0.1295238095238095
mean_delta = 0.027669270833333332
worst_delta = -0.001953125
near_pass_rate = 1.0
AUC_step_ratio = 0.9336784156141542
AUC_time_ratio = 0.7424718100091173
ECE_delta = 0.01767905056476593
LineC_nontearing_pass = 1
```

解释：

```text
B320 已经满足当前主线 base anchor 要求。
v12.14 不再把“找 base”当主目标。
v12.14 只做 B320 no-regression monitor，不继续 B320 小修。
```

### 1.2 Line C calibration 可信

v12.13 记录：

```text
linec_calibration_pass = 1
noop_false_positive_rows = 0
random_false_positive_rows = 0
```

解释：

```text
当前 blocker 不是 Line C metric 自己容易被 NoOp / random 误触发。
当前 blocker 是 functional direction 没有稳定产生 noise / reservoir release。
```

### 1.3 BM1 / BM3 / BM5 / BM2 的真实结果

#### BM1-ext

BM1b / SNR 类方法有局部 positive，但没有 3x3 survivor。

```text
BM1b-SoftSNRGate 可以在局部 row 同时改善 coupling/noise/reservoir；
但 full_3x3_pass_candidate_count = 0。
```

解释：

```text
SNR gate 是有诊断价值的 safety / preconditioning signal，
但不是当前 functional value source。
```

#### BM3 Branch Rebalance

BM3a 是典型 coupling-only move：

```text
CouplingR2_delta_mean 很高；
NoiseSignalLeak_delta_mean > 0；
RealSignalReservoirRatio_delta_mean > 0。
```

解释：

```text
BM3 能改变坐标/branch readout，
但它不是 signal-reservoir functional value source。
```

#### BM5 Reservoir VJP

BM5a / BM5b 没有真正释放 reservoir：

```text
CouplingR2_delta_mean ≈ 0.21
NoiseSignalLeak_delta_mean > 0
RealSignalReservoirRatio_delta_mean > 0
pass_rows = 0
```

解释：

```text
简单 reservoir VJP 不是足够的 reservoir-release basis。
```

#### BM2 constrained low-rank solve

BM2 比 BM3 更合理，但仍不够：

```text
BM2e/BM2g/BM2h:
  CouplingR2_delta_mean ≈ 0.18 - 0.21
  NoiseSignalLeak_delta_mean ≈ 0
  RealSignalReservoirRatio_delta_mean ≈ -0.002 到 -0.003
  pass_rows = 0
```

解释：

```text
低秩组合能维持 task-safe coupling，
但 basis 本身不能提供足够强的 noise-release / reservoir-release。
```

### 1.4 CE-specific projector 给了诊断，但不能进入 official

Batch 6-8 的 CE-specific projector VJP 发现：

```text
BM5e-LineCProjectorCEVJP:
  RealSignalReservoirRatio_delta_mean = -0.018313
  NoiseSignalLeak_delta_mean = +0.006051

BM2m-LineCGrid-projectorCE:
  RealSignalReservoirRatio_delta_mean = -0.016844
  NoiseSignalLeak_delta_mean = +0.007772
```

这说明：

$$
\boxed{
\text{reservoir release 是可达的，}
\text{但 CE-specific projector 牺牲了 noise safety。}
}
$$

更重要的是，这条路线 **不合规**：

```text
直接使用 CE(real label) / CE(permuted label) / per-example CE vector；
违反 loss-agnostic functional update 约束；
只能作为诊断，不允许计入 official functional success。
```

### 1.5 loss-agnostic cotangent / spectral shaping 是唯一新的可追信号

Batch 9-11 改成 label-free cotangent ensemble：

```text
只使用：
  current train batch input x
  model(x) logits
  label-free random cotangent ensemble
  cotangent VJP / gradient sketch / spectral objective

不使用：
  CE(real label)
  CE(permuted label)
  per-example CE vector
  validation/test/future outcome
  dataset-name branch
```

结果：

```text
BM2y-LogitCotangentSpectralIso:
  window=5 rows=45
  pass_rows=1

BM2z-LogitCotangentSpectralIsoOrth:
  window=5 rows=45
  pass_rows=1

bm2_partial_pass_rows = 2
bm2_full_3x3_pass_candidate_count = 0
route = R3-FunctionalMechanismPartialOnly
```

更具体地，pass 行只出现在：

```text
MNIST seed=2 split=2 window=5
```

因此结论是：

$$
\boxed{
\text{loss-agnostic direction 第一次出现局部 Line C pass，}
\text{但它没有跨 dataset/seed 稳定复现。}
}
$$

这就是 v12.14 应该追的方向，但必须从“弱 cotangent ensemble”升级为“更强、更可控的 loss-agnostic gradient-sketch spectral shaping”。

---

## 2. 独立判断：为什么这轮看起来慢

你感觉慢是合理的，因为 v12.13 没有把 functional update 推到 P4，也没有 official success。但这不是原地空转，而是进入了更硬的问题：

```text
以前：
  主要在找一个可用 PureKAN base。

现在：
  已有 B320 locked base，问题变成：
  functional update 是否能在合格 base 上产生独立、可预测、control-resistant、loss-agnostic 的几何收益？
```

这比 base search 更难，因为现在不能靠：

```text
final accuracy 好看；
CouplingR2 好看；
CE-specific VJP；
dataset-specific threshold；
posthoc replay；
validation/test feature；
小网格调 lambda；
随机匹配范数。
```

v12.13 的慢，本质上是因为我们终于开始触碰 functional update 的第一性问题：

$$
\boxed{
\text{如何在不依赖特定 loss 的情况下，改变模型的 train-probe signal/reservoir/noise 几何。}
}
$$

---

## 3. 当前真正卡在哪里

当前最精确的 blocker 是：

$$
\boxed{
\text{loss-agnostic functional direction 与 B320 当前训练轨迹中的 signal/reservoir projector 耦合太弱。}
}
$$

分解为四个子问题。

### 3.1 CouplingR2 已经不是 hard blocker

BM1 / BM2 / BM3 / BM5 / cotangent methods 都能提高 CouplingR2。尤其 BM3 可以把 CouplingR2 提得很高。

所以不能再把目标写成：

$$
\Delta CouplingR^2 > 0.
$$

这太弱，会奖励 coordinate move。

### 3.2 NoiseSignalLeak 是第一 hard blocker

多数方法的失败理由里，`NoiseSignalLeak_delta>-0.01` 是主导项。BM2 / noise-contrast 虽然能把均值推到很小的负数，但远不到 `-0.01`。

这说明 functional 方向没有真正把 noise 从 signal channel 推出去。

### 3.3 RealSignalReservoirRatio 是第二 hard blocker

CE-specific projector 能释放 reservoir，但非合规；loss-agnostic cotangent 方法基本没有稳定释放 reservoir。说明当前 label-free cotangent basis 没有覆盖到真实任务 hard signal 的 reservoir subspace。

### 3.4 当前 basis 改的是 readout/update direction，不是 gradient-sketch eigenspace

现有方法的本质更像：

```text
在已有 AdamW / branch / quad / reservoir VJP 方向里调组合。
```

但 Line C 的核心对象是：

$$
K = G G^T,
$$

其中 $G$ 是 per-example gradient sketch。要真正影响 signal/reservoir，需要让 functional update 改变：

$$
\operatorname{eig}(G G^T)
$$

而不是只微调 readout 或让 logits 有小位移。

---

## 4. 路线是否正确

路线仍然正确，但必须重排优先级。

正确保留：

```text
B320 locked anchor；
Line C calibration；
NoOp / RandomMatchedNorm null；
strong controls；
No-BSpline family policy；
no dataset-specific tuning；
P3 survivor before P4；
functional update as update rule, not loss。
```

必须停止：

```text
继续 F14/F15 / BM1/BM3/BM5 小网格；
继续只用 CouplingR2 promotion；
继续 CE-vector projector VJP；
继续把 per-example CE vector 当 functional objective；
继续用 dataset/split 局部 pass 进入 P4；
继续任意 coefficient grid without new basis。
```

v12.14 的核心应是：

$$
\boxed{
\text{loss-agnostic gradient-sketch spectral shaping。}
}
$$

---

## 5. 离目标还差多远

| 模块 | 当前状态 | 距离目标 | 判断 |
|---|---|---:|---|
| B320 base | locked anchor | 近 | 当前不是 blocker |
| efficiency | step ratio 0.399，memory 0.129 | 近 | 已进入强优势区 |
| expression / task | mean delta +0.0277，near 1.0，AUC < 1 | 近 | B320 足够承载 functional |
| Line C calibration | pass，无 NoOp/Random false positive | 近 | 指标可信 |
| functional update | no P3 survivor，no P4 | 远 | 当前主 blocker |
| loss-agnostic mechanism | 有 2 个局部 pass row | 中远 | 有信号，但很弱 |
| classic family | No-BSpline active families 无 pass | 中远 | 支线，不是主 blocker |
| next-gen MLP claim | 不能 claim | 远 | 缺 functional 独立收益 |

一句话：

```text
base 已经很近甚至可锁；
functional update 还远；
但 v12.13 第一次把 functional 的正确问题定位到 loss-agnostic signal/reservoir spectral shaping。
```

---

## 6. v12.14 总目标

v12.14 的总目标不是重新找 base，也不是把 CE-specific projector 修成 official，而是：

$$
\boxed{
\text{在 B320 locked base 上，构造一个 loss-agnostic functional update direction，}
\text{它能稳定改善 signal/reservoir/noise geometry，}
\text{并击败 strong controls。}
}
$$

Functional update 的 official direction generator 必须满足：

$$
d_{func}
=
F(\theta, x, f_\theta(x), \mathcal{C}_{label-free}, \text{branch/cache/sketch stats})
$$

其中不得使用：

$$
y,\quad CE(y,f_\theta(x)),\quad \nabla_\theta CE,\quad CE(y_{\pi},f_\theta(x)),
$$

也不得使用：

```text
validation/test metrics；
future outcome；
dataset name；
posthoc replay label；
per-example CE vector；
permuted-label CE vector。
```

允许使用：

```text
logits；
hidden/branch/cache activations；
label-free cotangent ensemble；
random / orthogonal / class-mean-free cotangents；
unlabeled train-stream statistics；
branch-role Jacobian sketches；
direct/quad/branch primitive Jacobian sketches；
base optimizer update only as an abstract loss-gradient vector for safety projection,
  not as a CE-specific target.
```

---

## 7. v12.14 四条线

```text
Line A：B320 Anchor Monitor
  只做 no-regression，不继续 base 小修。

Line C：Loss-Agnostic Manifold-Channel Diagnostics
  继续校准 CouplingR2 / NoiseSignalLeak / RealSignalReservoirRatio，
  并新增 cotangent-basis coverage diagnostics。

Line B：Loss-Agnostic Functional Mechanism Rebuild
  主线。构造能改变 gradient-sketch spectrum 的 label-free functional basis。

Line D：Classic No-BSpline Portfolio
  保持并行，但不抢主线资源。Rational/Cheby/Wavelet/RBF/Fourier 按 blocker 推进；B-spline frozen。
```

---

# Line A：B320 Anchor Monitor

## A0. 目标

确认 B320 没有 regression，不再优化 B320。

## A1. 实验设计

每次 v12.14 run 前后，固定跑一组 B320 monitor：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
train_size = 1024
val_size = 512
test_size = 512
batch_size = 128
same protocol as v12.13
```

## A2. 记录指标

```text
candidate_id
step_ratio_q90
memory_ratio_q90
mean_delta_vs_MLP
worst_delta_vs_MLP
near_pass_rate
AUC_step_ratio
AUC_time_ratio
ECE_delta
CEp99
LineC_nontearing_pass
no_fake_proxy_cpu
```

## A3. 判断标准

B320 monitor 必须满足：

$$
step\_ratio_{q90}\le 0.60,
$$

$$
memory\_ratio_{q90}\le 0.20,
$$

$$
mean\_delta\ge 0,
$$

$$
worst\_delta\ge -0.003,
$$

$$
AUC\_step\le 1.00,
$$

$$
AUC\_time\le 1.00,
$$

$$
LineC\_nontearing=1.
$$

如果 B320 monitor fail：

```text
停止 Line B P4；
只允许修复 regression / protocol mismatch；
不能用 functional update 掩盖 base regression。
```

---

# Line C：Loss-Agnostic Manifold-Channel Diagnostics

## C0. 目标

v12.14 的 Line C 不只做 pass/fail，还要诊断为什么 loss-agnostic cotangent 方法只在 MNIST 单一 seed/split 上有效。

核心问题：

```text
C-Q1: loss-agnostic cotangent basis 覆盖了多少 Line C signal/reservoir eigenspace？
C-Q2: 为什么 CE-specific projector 能释放 reservoir，而 cotangent basis 不能？
C-Q3: 哪些 branch / primitive role 对 NoiseSignalLeak 和 RealSignalReservoirRatio 最敏感？
```

## C1. 新增诊断：cotangent-basis coverage

对每个 candidate 的 label-free cotangent ensemble $\mathcal{C}$，计算其 VJP gradient sketch：

$$
G_{\mathcal{C}}\in\mathbb{R}^{m\times p}.
$$

Line C 原始 audit sketch 为：

$$
G_{audit}\in\mathbb{R}^{n\times p}.
$$

构造两个子空间投影：

$$
P_{\mathcal{C}}=\operatorname{proj}(\operatorname{row}(G_{\mathcal{C}})),
$$

$$
P_{sig},P_{res}
=
\operatorname{eigspace}(G_{audit}G_{audit}^T).
$$

记录：

$$
SignalCoverage=
\frac{\operatorname{tr}(P_{\mathcal{C}}P_{sig})}
{\operatorname{rank}(P_{sig})+\epsilon},
$$

$$
ReservoirCoverage=
\frac{\operatorname{tr}(P_{\mathcal{C}}P_{res})}
{\operatorname{rank}(P_{res})+\epsilon}.
$$

再定义：

$$
CoverageGap=
SignalCoverage-ReservoirCoverage.
$$

解释：

```text
SignalCoverage 高但 ReservoirCoverage 低：
  cotangent ensemble 只覆盖 easy signal，难以释放 hard reservoir。

ReservoirCoverage 高但 NoiseSignalLeak 不降：
  方向进入 reservoir，但没有正确区分 real/noise。

Coverage 都低：
  cotangent ensemble 太弱，接近 random readout perturbation。
```

## C2. 新增诊断：role-conditioned sketch

把 B320 参数按 role 分组：

```text
direct
quad
branch
gain
projection
tail
```

对每个 role 记录：

```text
role_signal_coverage
role_reservoir_coverage
role_noise_leak_sensitivity
role_coupling_gain
role_logit_drift_p95
```

目标是回答：

```text
是否存在某个 role 能 loss-agnostically 改变 signal/reservoir spectrum？
```

## C3. 新增诊断：window stability

所有 candidate 统一记录：

```text
window = 3, 5, 10
```

不能按 window 调参。window 只是诊断变量。P3 survivor 必须满足：

```text
primary window=5 pass；
window=3 or window=10 不出现 hard harm；
pass row 不集中在单一 dataset/seed。
```

## C4. Line C 必须输出 artifact

```text
v1214_linec_calibration.csv
v1214_linec_null_distribution.csv
v1214_cotangent_coverage.csv
v1214_role_conditioned_sketch.csv
v1214_window_stability.csv
v1214_linec_pareto.csv
v1214_linec_failure_table.csv
```

## C5. 必须可视化

```text
fig_C_null_distribution_noise_reservoir.svg
fig_C_cotangent_signal_reservoir_coverage.svg
fig_C_role_coverage_heatmap.svg
fig_C_noise_leak_vs_reservoir_release.svg
fig_C_window_stability.svg
fig_C_partial_pass_by_dataset_seed.svg
fig_C_pareto_front_coupling_noise_reservoir.svg
```

---

# Line B：Loss-Agnostic Functional Mechanism Rebuild

## B0. Functional 合规定义

### B0.1 Official 方向生成器签名

允许的 official API：

```python
build_functional_direction(
    model,
    x_batch,
    logits,
    hidden_cache,
    branch_cache,
    label_free_cotangents,
    unlabeled_sketch_stats,
    task_update=None,        # optional; only for generic safety projection
    rng=None,
) -> delta_theta
```

禁止：

```python
build_direction(..., y)
build_direction(..., ce_loss)
build_direction(..., per_example_ce_vector)
build_direction(..., permuted_label_ce)
build_direction(..., validation_metric)
build_direction(..., dataset_name)
```

CSV 必须记录：

```text
loss_agnostic_direction
ce_vector_used_for_direction
label_used_for_direction
permuted_label_used_for_direction
validation_used_for_commit
dataset_name_used_for_commit
task_update_used
task_update_role
```

official candidate 必须满足：

```text
loss_agnostic_direction = 1
ce_vector_used_for_direction = 0
label_used_for_direction = 0
permuted_label_used_for_direction = 0
validation_used_for_commit = 0
dataset_name_used_for_commit = 0
```

### B0.2 CE 的允许位置

允许 CE 出现的位置：

```text
1. 训练基线 B320 + ordinary task optimizer；
2. task non-harm 审计；
3. ECE / CEp99 / AUC / accuracy 等坏化约束；
4. 与其他 audit loss 并列的评估指标。
```

不允许 CE 出现的位置：

```text
1. functional direction objective；
2. functional direction VJP cotangent；
3. signal/reservoir release surrogate 的上游 residual；
4. candidate selection score；
5. commit-time controller feature。
```

换句话说：

$$
\boxed{
\text{CE 是 audit，不是 functional objective。}
}
$$

## B1. 机制族一：multi-cotangent spectral ensemble

### 假设 H-B1

BM2y/BM2z 失败不是因为 loss-agnostic 方向不可能，而是当前 cotangent ensemble 太弱、太少、覆盖不到 relevant reservoir/noise subspace。

### 候选

```text
BM2aa-ClassMeanFreeCotangentK16
BM2ab-ClassMeanFreeCotangentK32
BM2ac-LogitWhitenedCotangentK16
BM2ad-LogitWhitenedCotangentK32
BM2ae-OrthogonalRademacherCotangentK32
BM2af-MixedCotangentEnsembleK48
BM2ag-MixedCotangentEnsembleAdamWOrth
```

### 构造

给 logits $z\in\mathbb{R}^{B\times C}$，构造 label-free cotangent：

$$
c_j\in\mathbb{R}^{B\times C}.
$$

要求：

$$
\sum_{k=1}^{C} c_{bkj}=0
$$

即 class-mean-free，避免只改变 logit offset。

logit-whitened cotangent：

$$
c = \hat{\Sigma}_{z}^{-1/2} \epsilon,
$$

其中 $\epsilon$ 为 Rademacher / Gaussian noise，$\hat{\Sigma}_{z}$ 来自当前 train batch logits，不使用 label。

### 记录指标

```text
cotangent_count
cotangent_type
cotangent_rank
cotangent_condition
class_mean_free_error
logit_cov_condition
SignalCoverage
ReservoirCoverage
CoverageGap
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
control_gap
holdout_loss_ratio
logit_max_abs_drift
```

### 不满足条件时 Codex 先尝试

```text
if SignalCoverage and ReservoirCoverage both low:
  increase cotangent_count K: 16 -> 32 -> 64;
  switch Rademacher -> orthogonal Rademacher;
  add logit-whitening.

if CouplingR2 opens but noise/reservoir not:
  add role-conditioned cotangents rather than increasing alpha.

if logit drift > 0.05:
  reduce norm budget;
  project out logit offset and top confidence direction.

if control_gap < 0.005:
  remove AdamW-parallel component;
  compare with RandomMatchedNorm and ShuffledCotangent controls.
```

## B2. 机制族二：role-conditioned cotangent shaping

### 假设 H-B2

B320 的 direct / quad / branch roles 对 signal/reservoir/noise 的贡献不同。loss-agnostic cotangent 必须针对 role 的 Jacobian sketch，而不是只对全 logits 做 random cotangent。

### 候选

```text
BM2ba-DirectRoleCotangentSpectral
BM2bb-QuadRoleCotangentSpectral
BM2bc-BranchRoleCotangentSpectral
BM2bd-DirectQuadRoleMixed
BM2be-BranchQuadRoleMixed
BM2bf-RoleAdaptiveCotangentNoLabel
BM2bg-RoleAdaptiveCotangentNoLabelOrth
```

### 公式

对 role $r$，记该 role 的参数为 $\theta_r$，cotangent VJP 为：

$$
g_{r,j}
=
J_{\theta_r}(z)^T c_j.
$$

构造 role sketch：

$$
G_r=[g_{r,1},...,g_{r,K}]^T.
$$

选择 update 时只允许用 label-free $G_r$ 的谱统计：

$$
S_r=G_rG_r^T.
$$

### 记录指标

```text
role
role_param_count
role_update_norm
role_update_norm_ratio
role_signal_coverage
role_reservoir_coverage
role_noise_sensitivity
role_logit_drift
role_tail_drift
role_control_gap
```

### 不满足条件时 Codex 先尝试

```text
if branch role coupling high but reservoir/noise worse:
  branch role cannot be value source; demote to control only.

if quad role improves reservoir but harms noise:
  add noise-veto projection in label-free cotangent space.

if direct role near no-op:
  don't promote direct-only; use as stabilizer in mixed role.

if role-adaptive selects different roles per dataset:
  do not use dataset-specific policy; aggregate by role metrics only.
```

## B3. 机制族三：primitive-level gradient-sketch spectral shaping

### 假设 H-B3

v12.13 的 basis 主要移动 readout / update direction，不能改变 $G G^T$。要影响 Line C hard metrics，必须直接 reshape primitive-level per-example gradient sketch。

### 候选

```text
BM2ca-DirectPrimitiveSketchWhiten
BM2cb-QuadPrimitiveSketchWhiten
BM2cc-BranchPrimitiveSketchWhiten
BM2cd-DirectQuadSketchIsotropy
BM2ce-BranchQuadSketchIsotropy
BM2cf-SignalReservoirSketchSpread
BM2cg-NoiseNullSketchSpread
```

### 机制

对 role gradient sketch $G_r$，设计小更新 $d_r$，目标是在 label-free cotangent ensemble 下改变谱：

$$
K_r=G_rG_r^T.
$$

一个 loss-agnostic spectral objective：

$$
J_{spectral}
=
-\lambda_{flat}\cdot H(\lambda(K_r))
+
\lambda_{top}\cdot \operatorname{TopShare}(K_r)
+
\lambda_{drift}\cdot \|J_z d_r\|^2.
$$

其中 $H$ 是谱熵。该 objective 不使用 label 或 CE。

### 记录指标

```text
spectrum_entropy_before
spectrum_entropy_after
top_eigen_share_before
top_eigen_share_after
condition_before
condition_after
SignalCoverage_delta
ReservoirCoverage_delta
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
```

### 不满足条件时 Codex 先尝试

```text
if spectrum entropy improves but Line C not:
  inspect role coverage; switch role.

if noise leak worsens:
  add cotangent noise-null projection using label-free random cotangent high-variance directions.

if reservoir not released:
  add low-eigen cotangent lifting rather than top-eigen flattening.

if task non-harm fails:
  reduce update norm; add generic task_update orthogonalization without using CE-specific construction.
```

## B4. 机制族四：loss-agnostic moment transport

### 假设 H-B4

Functional update 不一定要做 VJP direction。它可以通过 transporting primitive moments 改变 future gradient sketch 的 geometry，例如 branch norm、quad readout scale、direct/quad covariance。只要 moment target 不依赖 label/loss，它仍然是 loss-agnostic。

### 候选

```text
BM4a-BranchMomentTransport
BM4b-QuadMomentTransport
BM4c-DirectQuadCovTransport
BM4d-RoleCovarianceEqualization
BM4e-LogitJacobianMomentTransport
BM4f-TailSafeMomentTransport
```

### 机制

定义 unlabeled moment：

$$
M_r=\mathbb{E}_{x\in B}[\phi_r(x)\phi_r(x)^T].
$$

Functional direction 目标：

$$
M_r' \approx \operatorname{Iso}(M_r)
$$

或

$$
\operatorname{TopShare}(M_r')<\operatorname{TopShare}(M_r).
$$

### 记录指标

```text
role_moment_condition_before
role_moment_condition_after
role_moment_top_share_before
role_moment_top_share_after
feature_cov_drift
logit_cov_drift
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
ECE_delta
```

### 不满足条件时 Codex 先尝试

```text
if moment transport improves conditioning but not Line C:
  combine with cotangent spectral ensemble.

if logit covariance drift too high:
  add tail-safe norm budget.

if all moment transport acts like no-op:
  increase event interval window and inspect feature moment sensitivity.
```

---

# Line B P3：cloned one-step / five-step official diagnostic

## P3.1 实验矩阵

对每个 candidate：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
splits = 0,1,2
windows = 3,5,10
functional_batch_size = 32
probe_batch_size = 32
```

Controls：

```text
C0-NoOpMatchedOverhead
C1-RandomMatchedNorm
C2-AdamWParallelDirection
C3-SNR-only
C4-ShuffledCotangent
C5-ShuffledRole
C6-MLPAnalogCotangent
C7-TaskUpdateOnly
```

## P3.2 记录字段

```text
run_id
candidate_id
method_family
dataset
seed
split
window
loss_agnostic_direction
ce_vector_used_for_direction
label_used_for_direction
permuted_label_used_for_direction
validation_used_for_commit
dataset_name_used_for_commit

cotangent_count
cotangent_type
cotangent_rank
role
role_update_norm_ratio
functional_norm_ratio
task_orthogonal_fraction
control_id
best_control_id

CouplingR2_before
CouplingR2_after
CouplingR2_delta
NoiseSignalLeak_before
NoiseSignalLeak_after
NoiseSignalLeak_delta
RealSignalReservoirRatio_before
RealSignalReservoirRatio_after
RealSignalReservoirRatio_delta

SignalCoverage
ReservoirCoverage
CoverageGap
spectrum_entropy_delta
top_eigen_share_delta

holdout_loss_ratio_CE
holdout_loss_ratio_Brier
holdout_logit_mse_ratio
CEp99_delta
ECE_delta
margin_p10_delta
logit_max_abs_drift
control_gap_vs_best
amortized_overhead_estimate

pass_coupling
pass_noise
pass_reservoir
pass_control_gap
pass_task_safety
pass_loss_agnostic
pareto_pass
fail_reason
```

## P3.3 判断标准

单 row pass：

$$
\Delta CouplingR^2\ge 0.02,
$$

$$
\Delta NoiseSignalLeak\le -0.01,
$$

$$
\Delta RealSignalReservoirRatio\le -0.01,
$$

$$
control\_gap\ge 0.005,
$$

$$
logit\_max\_abs\_drift\le 0.05,
$$

$$
holdout\_loss\_ratio\_{CE}\le 1.002,
$$

$$
holdout\_loss\_ratio\_{Brier}\le 1.002,
$$

$$
CEp99\_delta\le 0.05.
$$

同时：

```text
loss_agnostic_direction = 1
ce_vector_used_for_direction = 0
label_used_for_direction = 0
permuted_label_used_for_direction = 0
```

Candidate promotion：

```text
primary:
  3x3 all-row pass on datasets x seeds for window=5.

secondary:
  >=8/9 pass;
  bootstrap CI lower for control_gap >= 0;
  failure not concentrated in one dataset;
  window=3/10 no hard harm.

禁止：
  单一 dataset/seed/split pass 进入 P4。
```

## P3.4 P3 failure 后 Codex 自动尝试

```text
if coupling fails:
  increase cotangent rank or use logit-whitened cotangents.

if coupling passes but noise fails:
  add label-free high-variance cotangent noise-null projection;
  do not use permuted-label CE.

if reservoir fails:
  add low-eigen cotangent lifting / primitive role low-eigen shaping;
  do not use CE projector VJP.

if control_gap fails:
  remove AdamW-parallel component;
  compare against RandomMatchedNorm and ShuffledCotangent.

if logit drift fails:
  reduce functional_norm_ratio;
  add tail-safe moment budget.

if task safety fails only under CE but not Brier:
  still fail; CE is an audit constraint.
  Do not tune direction for CE.
  Reduce norm or improve generic stability.
```

---

# Line B P4：short-run official re-entry

## P4.0 开启条件

只有 P3 有 promoted candidate 时，P4 才能运行。

```text
P4_open = P3_promoted_candidate_count > 0
```

如果 P3 没有 promoted candidate：

```text
P4 = not_run
reason = no_loss_agnostic_P3_survivor
```

## P4.1 实验设计

Methods：

```text
B320-TaskOnly
B320-NoOpMatchedOverhead
B320-RandomMatchedNorm
B320-AdamWParallelDirection
B320-SNR-only
B320-bestLossAgnosticFunctional
B320-MLPAnalogFunctional
```

设置：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
train_size = 1024
val_size = 512
test_size = 512
epochs = 3 or 5
functional_event_interval = 24, 48
fixed_lambda first; runtime backtracking only diagnostic due cost
```

## P4.2 记录字段

```text
method
dataset
seed
event_count
accepted_event_count
rejected_event_count
mean_functional_norm_ratio
amortized_overhead_ratio
acc_delta_vs_B320
AUC_step_delta_vs_B320
AUC_time_delta_vs_B320
ECE_delta_vs_B320
CEp99_delta_vs_B320
margin_p10_delta_vs_B320
CouplingR2_delta_vs_B320
NoiseSignalLeak_delta_vs_B320
RealSignalReservoirRatio_delta_vs_B320
control_gap_vs_best
loss_agnostic_audit_pass
strict_pass
fail_reason
```

## P4.3 判断标准

P4 strict pass：

$$
acc\_delta\_{vsB320}\ge -0.003,
$$

$$
AUC\_time\_delta\_{vsB320}\le 0,
$$

$$
ECE\_delta\_{vsB320}\le 0.01,
$$

$$
CEp99\_delta\_{vsB320}\le 0.05,
$$

$$
\Delta CouplingR^2\ge 0.02,
$$

$$
\Delta NoiseSignalLeak\le -0.01,
$$

$$
\Delta RealSignalReservoirRatio\le -0.01,
$$

$$
control\_gap\ge 0.005,
$$

$$
amortized\_overhead\le 1.05.
$$

P4 必须打过：

```text
NoOpMatchedOverhead
RandomMatchedNorm
AdamWParallelDirection
SNR-only
MLPAnalogFunctional
```

---

# Line D：Classic No-BSpline Portfolio

## D0. 当前状态

```text
Rational: TaskBlocked
Chebyshev: TaskBlocked
Wavelet: TaskBlocked
RBF: ExpressionBlocked
Fourier: ExpressionBlocked
BSpline: RejectedForThisVersion / frozen
```

Line D 不替代 B320 functional 主线。v12.14 只做低预算并行推进。

## D1. Rational

当前 blocker：

```text
L3/A4 pass exists, A5 task gate fail.
```

下一步：

```text
继续 loss-agnostic trajectory primitive；
不做 CE-specific calibration；
继续记录 coupling_collapse / reservoir_trapping。
```

指标：

```text
den_p01
r_prime_p95
r_double_prime_p95
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
A5 mean/worst/AUC/ECE
```

## D2. Chebyshev / Wavelet

当前 blocker：

```text
L3/A4 pass exists, A5 task gate fail.
```

下一步：

```text
Chebyshev:
  degree-energy damping / input-scale stability / loss-agnostic spectral audit。

Wavelet:
  local support coverage / scale-energy balance / tail stability。
```

## D3. RBF / Fourier

当前 blocker：

```text
L3 pass exists, A4 expression fail.
```

下一步：

```text
RBF:
  compact active center K4/K8;
  expression battery repair before task.

Fourier:
  low-frequency + local direct path;
  avoid high-frequency noise leak;
  expression repair before task.
```

## D4. B-spline

```text
BSpline.status = RejectedForThisVersion
active_followup = 0
codex_budget = 0
```

---

# 8. v12.14 artifact contract

必须输出：

```text
v1214_route_decision.json
v1214_b320_anchor_monitor.csv
v1214_linec_calibration.csv
v1214_linec_null_distribution.csv
v1214_cotangent_coverage.csv
v1214_role_conditioned_sketch.csv
v1214_functional_p3_lossagnostic.csv
v1214_functional_p3_controls.csv
v1214_functional_p3_failure_table.csv
v1214_functional_p4_short_run.csv
v1214_loss_agnostic_audit.csv
v1214_classic_family_status.json
v1214_provenance_audit.csv
v1214_hash_manifest.json
```

Loss-agnostic audit 必须包含：

```text
method
loss_agnostic_direction
ce_vector_used_for_direction
label_used_for_direction
permuted_label_used_for_direction
validation_used_for_commit
dataset_name_used_for_commit
task_update_used
task_update_role
official_eligible
violation_reason
```

---

# 9. 必须生成的图

```text
fig_A_b320_anchor_monitor.svg

fig_C_null_distribution_noise_reservoir.svg
fig_C_cotangent_signal_reservoir_coverage.svg
fig_C_role_coverage_heatmap.svg
fig_C_window_stability.svg
fig_C_partial_pass_by_dataset_seed.svg
fig_C_noise_leak_vs_reservoir_release.svg

fig_B_p3_lossagnostic_pareto.svg
fig_B_p3_control_gap.svg
fig_B_p3_fail_reason_heatmap.svg
fig_B_coupling_vs_noise_reservoir.svg
fig_B_cotangent_rank_vs_linec_gain.svg
fig_B_role_functional_effect.svg
fig_B_p4_short_run_if_open.svg

fig_D_classic_family_status.svg
```

---

# 10. 总体判断标准

## Minimum progress

满足：

```text
B320 anchor monitor pass；
Line C calibration pass；
loss-agnostic audit implemented；
at least one loss-agnostic candidate improves over v12.13 BM2y/BM2z mean result；
no CE-vector violation；
no P4 without P3 survivor。
```

## Functional P3 success

满足：

```text
loss_agnostic_direction = 1；
3x3 all-row pass or 8/9 + CI valid；
CouplingR2 / NoiseSignalLeak / RealSignalReservoirRatio all pass；
control_gap >= 0.005；
not concentrated in one dataset/seed。
```

## Functional P4 success

满足：

```text
P3 survivor enters short-run；
task non-harm；
AUC-time non-harm；
Line C improves；
beats controls；
amortized overhead <= 1.05；
loss-agnostic audit pass。
```

## Project success for current phase

满足：

```text
B320 + loss-agnostic functional update
beats B320 + ordinary task optimizer
on geometry and/or AUC-time
without task/cost regression
and beats strong controls.
```

---

# 11. 本轮禁止事项

v12.14 特别禁止：

```text
1. 不许再用 CE(real label) VJP / CE(permuted label) VJP 作为 functional direction；
2. 不许把 CE-specific projector surrogate 写成 official；
3. 不许只优化 CE / CEp99 / ECE 来生成 functional update；
4. 不许用 dataset name 调 threshold / role / window；
5. 不许单一 MNIST seed/split pass 进入 P4；
6. 不许继续 BM1/BM3/BM5 强度小网格；
7. 不许只报告 CouplingR2，而忽略 NoiseSignalLeak / RealSignalReservoirRatio；
8. 不许 functional update 在 base monitor fail 时打开；
9. 不许把 classic family 结果抢主线，B320 functional 是当前主线；
10. 不许把 diagnostic 写成 official success。
```

---

# 12. 推荐执行顺序

## Batch 1：audit and calibration

```text
A0 B320 monitor
C0 Line C null calibration
B0 loss-agnostic audit enforcement
```

Stop if：

```text
B320 monitor fail；
NoOp / Random false positive rows > 0；
loss-agnostic audit cannot distinguish CE-specific methods。
```

## Batch 2：cotangent coverage autopsy

```text
C1 cotangent-basis coverage
C2 role-conditioned sketch
C3 window stability
```

Goal：

```text
解释 BM2y/BM2z 为什么只在 MNIST seed=2 split=2 pass。
```

## Batch 3：mechanism family B1/B2

```text
B1 multi-cotangent spectral ensemble
B2 role-conditioned cotangent shaping
```

Gate：

```text
no CE vectors；
no label；
no validation/test；
P3 row-level hard gate。
```

## Batch 4：mechanism family B3/B4

```text
B3 primitive-level gradient-sketch spectral shaping
B4 moment transport
```

Gate same as Batch 3.

## Batch 5：P3 aggregation

```text
aggregate all candidates;
compute 3x3 pass;
compute bootstrap CI;
compare controls;
write route decision.
```

## Batch 6：P4 short-run only if P3 survivor

```text
run short-run with best loss-agnostic candidate;
compare strong controls;
write functional official decision.
```

---

# 13. 最终解释规则

### Case A：loss-agnostic P3 + P4 pass

可以声明：

```text
在 B320 strict FC-PureKAN base 上，loss-agnostic functional update 产生了 control-resistant 的 signal/reservoir/noise-safe geometry improvement。
```

### Case B：loss-agnostic P3 pass, P4 fail

声明：

```text
one-step/five-step loss-agnostic geometry signal exists, but does not survive continuous training.
```

下一步做 event scheduling / amortization，不改 functional objective。

### Case C：only CE-specific directions work

声明：

```text
Reservoir release is reachable in diagnostic CE-specific projector space, but current loss-agnostic functional mechanism is insufficient.
```

不能 claim official functional success。

### Case D：only coupling improves

声明：

```text
Functional direction is a coordinate/coupling move, not a good-geometry update.
```

停止 coupling-only promotion。

### Case E：no loss-agnostic signal beyond random

声明：

```text
Current functional mechanism family lacks population-safe value source; return to primitive-level geometry or new signal-channel estimator.
```

---

# 14. 最终总结

v12.13 的价值是把错误路线排除得很清楚：

```text
B320 不再是问题；
Line C measurement 本轮可信；
BM3 coupling-only 不是价值源；
CE-specific projector 能释放 reservoir，但不合规且 noise-unsafe；
loss-agnostic cotangent 方法出现局部信号，但覆盖太低。
```

v12.14 的核心不是小修，而是：

$$
\boxed{
\text{从 CE-specific projector 转向 loss-agnostic gradient-sketch spectral shaping。}
}
$$

只有当 functional update 不依赖 CE vector、label、dataset、validation/test/future outcome，并且仍能稳定降低 NoiseSignalLeak 和 RealSignalReservoirRatio，项目才真正接近原始目标：

$$
\boxed{
\text{PureKAN base}+\text{functional update}
>
\text{PureKAN base}+\text{ordinary backprop}.
}
$$
