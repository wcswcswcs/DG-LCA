# DG-KAN v12.2：v12.1 Good Geometry Battery 独立复盘与下一步实验计划

> 日期：2026-05-19  
> 目标：独立分析 v12.1 Good Geometry Battery 结果，不被报告原结论牵着走；明确当前进展、真正卡点、是否仍在正确道路上、离目标还差多远，并给出可直接交给 Codex 并行执行的下一步实验计划。  
> 公式格式：Typora 友好，仅使用 `$...$` 与 `$$...$$`。

---

## 0. 总判断

这次实验有进展，但不是科学结论上的进展，而是 **诊断系统闭环上的进展**。

更准确地说：

```text
工程进展：明显。
  P0 contract pass；
  P2 Good Geometry Battery 完整落盘；
  P3 GeometryCertificateV0 能生成；
  P4 即使 official gate 关闭，也能真实跑 diagnostic controls / one-step / five-step；
  no-fake audit 干净。

科学进展：有限，甚至暴露出更根本的 base 问题。
  repaired LQ 当前没有通过 base qualification；
  expression battery 失败；
  timing/step ratio 严重失败；
  functional candidates 没有击败 strong controls；
  P5 short-run 不应打开。

路线判断：仍在正确道路上，但必须停止“小修小补式 functional 方向搜索”。
  现在最重要的不是继续调 D8/D9/D10/D11，
  而是先把 repaired LQ base 的表达力、效率、条件数和 measurement truth 重新打穿。
```

所以本轮不能说“没有进度”。它真正推进了三件事：

1. 证明 v12 pipeline 现在能做真实 metric / gate / control audit，而不是只写计划。
2. 证明当前 repaired LQ base 在这个 runner 和这个 budget 下不合格。
3. 证明当前 functional geometry directions 即使多数 task-safe，也没有 control-resistant advantage。

但从最终目标看，确实仍然非常慢，因为我们还没有跨过第一道硬门：

$$
\boxed{
\text{合格 PureKAN base 尚未稳定成立。}
}
$$

如果 base 不稳定，functional update 的 official success 就不能打开。继续在 functional direction 上小修，会重复 v9.x 的旧问题：能造动作、能移动几何、但不能证明它相对 strong controls 有独立价值。

---

## 1. 本轮结果到底说明了什么

### 1.1 P1 task mean 不差，但稳定性没有过

本轮 repaired LQ 的 mean validation accuracy 是：

```text
R2 repaired LQ mean val acc = 0.8435329861111112
same-param MLP mean val acc = 0.8415798611111112
macro delta = +0.001953125
```

表面上看，R2 repaired LQ 均值略高于 same-param MLP。这不是坏信号，说明它不是完全不能学任务。

但 paired rows 显示它并不稳定：

```text
MNIST:
  seed0 +0.009765625
  seed1 -0.001953125
  seed2 -0.013671875

Fashion-MNIST:
  seed0 +0.0078125
  seed1 +0.01953125
  seed2 +0.015625

KMNIST:
  seed0 +0.00390625
  seed1 -0.009765625
  seed2 -0.013671875
```

这说明一个很关键的现象：

```text
Fashion 上 repaired LQ 有稳定优势；
MNIST / KMNIST 上 seed1/seed2 有稳定劣势；
所以问题不是简单“KAN 不行”，而是当前 repaired LQ 的训练形状或条件数在某些数据分布/seed 上不稳定。
```

但我们不能针对 Fashion / KMNIST 分别调参。正确做法是把它们当成 failure slices，用来诊断统一机制：

```text
为什么 Fashion 稳定正？
为什么 MNIST/KMNIST 后两个 seed 稳定负？
这个差异是否来自 lift conditioning、tail margin、signal consistency、basis usage 或 optimization trajectory？
```

### 1.2 P1 efficiency 失败非常严重，不能轻轻带过

本轮 P1 failure table 中：

```text
step_q90 = 6.872180773276933 > 1.10
compact_memory_ratio_q90 = 1.100739543844385 > 1.00
```

这不是 near miss，而是 hard fail。尤其 step ratio 接近 `6.87x`，如果这个数字是真实训练路径的 steady-state step ratio，那么 repaired LQ 距离 next-generation MLP 目标非常远。

但是这里有一个必须独立审视的问题：

```text
这和之前 LQ / repaired LQ 的 timing 结果不一致。
```

历史上 LQ / repaired LQ 有过接近 MLP envelope 的记录，而本轮 v12.1 runner 得到 `6.87x` step q90。这个矛盾不能靠一句“base not qualified”盖过去。它至少有两种可能：

```text
可能 A：repaired LQ 当前实现或当前 runner 的 profiling path 真的退化了。
可能 B：v12.1 profiling 混入了 geometry instrumentation、cold start、compile overhead、probe overhead、small profile_steps、或非 compact/recompute path。
```

因此当前第一优先级不是调功能方向，而是做 **Timing Truth Audit**。如果 timing 测量不可信，整个 P1 base gate 就不可信；如果 timing 测量可信，当前 repaired LQ 不能作为 official base。

### 1.3 Expression battery 失败比 task mean 更危险

本轮 expression battery 中，R2 repaired LQ 明显低于 MLP：

```text
E0-additive:
  R2 repaired LQ = 0.9409700433
  MLP = 0.9866019686
  delta = -0.0456319253

E1-pairwise-product:
  R2 repaired LQ = 0.4191594323
  MLP = 0.7380732695
  delta = -0.3189138373

E2-composition:
  R2 repaired LQ = 0.8255044023
  MLP = 0.8799602787
  delta = -0.0544558764

E4-high-frequency:
  R2 repaired LQ = -0.2573012908
  MLP = -0.0987630288
  delta = -0.1585382620
```

这比 task accuracy 小幅波动更严重。因为我们的核心假设是：

$$
\boxed{
\text{KAN base 至少不能牺牲表达力，尤其不能丢 interaction。}
}
$$

而 LQ / quadratic edge basis 本来应该在 pairwise interaction 上有优势或至少不弱。如果 pairwise-product R2 只有 `0.419`，说明当前结果出现了三种可能之一：

```text
可能 A：expression battery 的训练 budget 太短，60 steps 不足以训练 repaired LQ。
可能 B：repaired LQ 的 fanin-output-scale repair 破坏了原始 LQ 的 interaction channel。
可能 C：实现或参数映射有 bug，quadratic edge basis 没有真正参与表达。
```

这件事必须用 **representational truth test** 拆开：

```text
1. 如果 least-squares / closed-form edge coefficient fit 能高 R2，但 AdamW 60 steps 低 R2：
   问题是优化 / 初始化 / training recipe。

2. 如果 least-squares 也低 R2：
   问题是架构实现或 repaired LQ 表达通道被破坏。

3. 如果 old LQ-t2 高 R2、R2 repaired LQ 低 R2：
   问题是 repaired scaling 改坏了表达通道。
```

不能在这件事没查清前继续写“LQ 表达力更强”。

### 1.4 P2 Good Geometry Battery 有真实信号，但不是全好

P2 final checkpoint 上，R2 repaired LQ 相比 MLP 有几个好信号：

```text
effective_rank_hidden:
  MLP = 57.59919654
  R2 repaired LQ = 71.53436152

basis_usage_entropy:
  MLP = 0.0
  R2 repaired LQ = 0.71194866

noise_leak:
  MLP = 0.86209342
  R2 repaired LQ = 0.49669691

CE_p99:
  MLP = 6.32777479
  R2 repaired LQ = 5.86348444

ECE:
  MLP = 0.09202864
  R2 repaired LQ = 0.08345589
```

这些不是幻觉。它们说明 LQ 类结构确实可能带来：

```text
更高 hidden effective rank；
非零 basis usage；
更低 noise leak；
更好的 final checkpoint ECE / CE tail。
```

这正是我们希望 KAN base 能提供的“几何潜力”。

但同一张表里也有严重坏信号：

```text
lift_condition_proxy:
  R2 repaired LQ = 119167112154680.89

perturb_logit_drift_p95:
  MLP = 0.01389825
  R2 repaired LQ = 0.01662022

signal_consistency_real:
  MLP = 0.45425731
  R2 repaired LQ = 0.38097754
```

这说明当前 LQ 不是“几何好”，而是 **几何分裂**：

```text
rank / basis usage / noise leak / final ECE 有潜力；
lift conditioning / perturbation stability / real signal consistency 很差。
```

这非常符合我们对问题本质的判断：KAN 的表达坐标可能更丰富，但训练坐标系统不稳定；functional update 想保几何，首先必须面对的是 **coordinate conditioning**，不是单纯 smoothing。

因此下一步 base repair 的核心不是“再加一个平滑项”，而是：

$$
\boxed{
\text{修复 LQ 的 lift conditioning 与 signal alignment，同时保持 interaction expressivity。}
}
$$

### 1.5 P3 certificate 的失败说明：被动 Pareto 信号还不够

P3 结果是：

```text
certificate_pass = false
hard_gate_pass = false
pareto_pass = true
failure_reasons = val_loss_auc_time; step_time_ratio; CEp99; margin_p10
```

这说明 repaired LQ 在部分几何指标上确实能形成 Pareto 信号，但无法通过 hard gates。核心问题是：

```text
几何信号没有转成训练速度、tail margin、效率三者同时成立。
```

这里要避免一个误读：P2 final CEp99 看起来 R2 比 MLP 好，但 P3 aggregate hard gate 仍然失败。这不矛盾，因为它们口径不同：

```text
P2 是 final checkpoint val split；
P3 是 certificate aggregate / trajectory gate。
```

这反而提示我们：LQ 可能最终点上 tail/ECE 好，但训练过程中 tail/margin 或 AUC 不稳定。下一轮必须画 trajectory，而不是只看 final checkpoint。

### 1.6 P4 diagnostic 是有价值的，但 functional 没有进展到 official success

P4 这次修复了一个重要工程问题：以前 P1/P3 gate fail 后 P4 全部 not_run；现在 P4 diagnostic 能继续跑，并明确写 `official_gate_open=0`。这是正确的，因为它让我们在不冒充 official success 的前提下收集方向质量数据。

P4 结果：

```text
D8-SNRProjectedGeometry:
  best geometry score = 0.3039031239
  control best = 0.3162052612
  beats strong controls = 0

D9-BasisEntropyRebalance:
  best geometry score = 0.3031435287
  control best = 0.3162052612
  beats strong controls = 0

D10-LiftConditionRepair:
  best geometry score = 0.3037566523
  control best = 0.3162052612
  beats strong controls = 0

D11-TailStabilityCorrection:
  best geometry score = 0.3033008926
  control best = 0.3162052612
  beats strong controls = 0
```

独立看，这组结果有两层含义：

```text
1. D8-D11 不是完全乱动。它们多数 task-safe，且 geometry score 接近 control best。
2. 但它们没有独立价值。strong controls 仍更好，所以不能 claim functional advantage。
```

真正的问题不是“lambda 不够”或“阈值太严”。如果我们调低 gate，最多会得到一个被 AdamWParallel / RandomMatchedNorm 解释掉的假阳性。

当前 functional blocker 是：

$$
\boxed{
\text{functional direction 还没有形成相对 AdamW/task dynamics 的互补收益。}
}
$$

因此下一步 functional 方向必须从“直接做几何 score”改成：

```text
control-resistant complementarity：
  functional direction 必须改善 AdamW controls 改不了的几何；
  或者在同等 task slack 下比 AdamWParallel / RandomMatchedNorm 更有效；
  或者在 AdamW-orthogonal residual 子空间里提供稳定收益。
```

---

## 2. 当前卡在哪里

### 卡点一：base qualification 不是一个单点失败，而是三重失败

当前 P1 base fail 不是只差一点 near pass rate。它同时失败在：

```text
1. stable near-pass:
   near_pass_rate = 0.7777777778 < 0.80

2. efficiency:
   step_q90 = 6.8721807733 > 1.10
   compact_memory_ratio_q90 = 1.1007395438 > 1.00

3. expression:
   pairwise_product_R2 = 0.4191594323 < 0.85
   composition_R2_delta_vs_MLP = -0.0544558764 < -0.05
```

其中最根本的是 expression 与 efficiency。Accuracy 均值略高可以给我们一点信心，但不能覆盖 expression 和 step 的失败。

### 卡点二：LQ 的几何潜力和训练病灶同时存在

P2 显示 LQ 的 hidden rank、basis usage、noise leak、final ECE 有正信号；但 lift condition、perturb drift、real signal consistency 有负信号。

这说明我们不是走错了路，而是还没有掌握 LQ 训练坐标的稳定化方法。KAN 不是天然“几何好”；它只是提供了更丰富的 edge/lift 坐标。这个坐标如果 condition 不好，反而更容易产生训练不稳定。

### 卡点三：functional update 现在没有独立因果性

P4 diagnostic 的 D8-D11 没有打过 strong controls。这意味着当前 functional candidates 还只是：

```text
task-safe geometry-like perturbations
```

而不是：

```text
control-resistant functional geometry maintenance。
```

这和我们最终目标差得很远。最终要证明的是：

$$
\boxed{
\text{在同一 PureKAN base 上，functional update 带来 ordinary backprop 没有的几何/收敛优势。}
}
$$

本轮没有证明这一点。

### 卡点四：我们仍然可能被 measurement path 混淆

本轮 step ratio `6.87x` 和历史 repaired LQ / LQ-t2 timing 有明显张力。这里如果不做 timing truth audit，后续会陷入两种错误：

```text
错误 A：把 profiling artifact 当 base 失败。
错误 B：把真实 efficiency 失败当成 runner 问题。
```

必须先拆开。

---

## 3. 是否仍在正确道路上

我的判断是：**方向仍然正确，但当前执行顺序必须收紧。**

正确的地方：

```text
1. 不再直接打开 functional short-run；
2. Good Geometry Battery 的设计方向是对的；
3. P4 diagnostic 即使 gate closed 也继续落盘，这是对的；
4. strong controls 保留得很对，避免 functional 假阳性；
5. no-fake / no-proxy 约束是对的。
```

跑偏的风险：

```text
1. 把 P2 的部分 geometry 好信号包装成 base 成功；
2. 忽略 expression battery 失败；
3. 忽略 step ratio 6.87 的严重性；
4. 继续在 D8-D11 上调 lambda / threshold；
5. 按数据集修 Fashion/MNIST/KMNIST；
6. 继续跑 P5 short-run。
```

正确路线应该是：

$$
\boxed{
\text{先做 Base Truth 与 Conditioning Repair，再做 Functional Complementarity。}
}
$$

不是：

$$
\boxed{
\text{继续小修 functional directions。}
}
$$

---

## 4. 离最终目标还差多远

按照最终目标拆分，目前状态如下。

### 4.1 表达力

目标：

$$
\text{Expression}_{KAN}\geq\text{Expression}_{MLP}
$$

当前：

```text
task mean acc 略高，但 expression battery 明显低于 MLP。
pairwise-product R2 差距尤其大：
  0.419 vs 0.738
```

距离目标：**远**。  
不是因为完全无希望，而是需要先判断 expression fail 是 undertraining、实现问题还是 repaired scaling 破坏 interaction。

### 4.2 效率

目标：

$$
\frac{T_{step,KAN}}{T_{step,MLP}}\leq1.10
$$

当前：

$$
\frac{T_{step,KAN}}{T_{step,MLP}}=6.872
$$

距离目标：**非常远，除非这是 profiling artifact**。  
所以必须立即做 timing truth audit。

### 4.3 任务性能

目标：

$$
Acc_{KAN}\geq Acc_{MLP}
$$

当前 mean delta：

$$
+0.001953125
$$

但 near pass rate 未过，MNIST/KMNIST seed1/2 不稳。

距离目标：**近但不稳定**。  
这部分是最有希望的。

### 4.4 几何

目标：

```text
geometry 更稳定；
tail 更好；
noise leak 更低；
signal channel 更一致；
perturb drift 更低；
conditioning 不坏。
```

当前：

```text
rank / basis / noise leak / ECE 有正信号；
lift condition / perturb drift / signal consistency 有负信号；
P3 hard gate fail。
```

距离目标：**中等偏远**。  
不是没有几何信号，而是几何还没有变成 hard-gate-safe 的训练优势。

### 4.5 functional update

目标：

$$
C = \text{LQ}+\text{Functional}
$$

必须满足：

$$
C > \text{LQ}+\text{AdamW}
$$

并且打过：

```text
NoOp；
RandomMatchedNorm；
AdamWParallelDirection；
ShuffledPayload；
MLP analog；
QuadraticFeatureMLP analog。
```

当前：

```text
D8-D11 没有击败 strong controls；
official gate closed；
P5 not_run。
```

距离目标：**远**。  
本轮没有 functional scientific progress，只证明了现有 functional candidates 不够。

### 4.6 next-generation MLP claim

目标是：

```text
same-param / same-FLOPs / same-wall-clock 不输 MLP；
表达力不打折；
几何/收敛/calibration 至少一项有优势；
functional contribution 可独立区分。
```

当前还没有到 claim 阶段。  
准确说，我们离 final claim 不是“再调一两个参数”的距离，而是至少需要完成两个硬闭环：

```text
闭环 1：
  repaired LQ base expression + timing + task stability 通过。

闭环 2：
  functional direction 在合格 base 上 control-resistant 地改善 geometry / convergence。
```

---

## 5. v12.2 总体实验目标

v12.2 不再以“跑 functional short-run”为目标。它的目标是回答四个更根本的问题：

```text
Q1. 本轮 repaired LQ base fail 是真实架构失败，还是 measurement / runner / budget 造成的假失败？

Q2. expression battery 中 pairwise-product 失败，是训练不足、实现 bug，还是 repaired LQ 真的破坏了 interaction channel？

Q3. LQ 的几何潜力能否通过 dataset-agnostic conditioning repair 转成 hard-gate-safe base？

Q4. functional update 能否从“接近 controls 的 task-safe perturbation”变成“control-resistant complementary geometry maintenance”？
```

v12.2 的整体路线是：

$$
\boxed{
\text{Base Truth Audit}
\rightarrow
\text{Interaction / Conditioning Repair}
\rightarrow
\text{Passive Geometry Re-test}
\rightarrow
\text{Functional Complementarity Audit}
}
$$

只有前两步过，才允许打开 official functional training。

---

## 6. v12.2 核心假设

### H1：当前 step ratio 6.87 可能是 measurement path artifact

本轮 step ratio 严重偏离此前 LQ compact timing 预期。因此首先要验证：

```text
H1a: v12 runner 的 profiling 混入了 geometry instrumentation 或 probe overhead。
H1b: profile_steps=4 太短，cold/compile/sync noise 污染 q90。
H1c: repaired LQ 当前 path 没有使用 compact/recompute 或 compiled fast path。
H1d: 当前 R2 implementation 真的退化。
```

H1 成立标准：

```text
clean timing audit 中 R2-LQ step q90 <= 1.25；
或至少 clean timing 与 v12.1 timing 差异超过 3x，证明 v12.1 timing 不可作为 architecture gate。
```

H1 不成立标准：

```text
clean timing 仍然 step q90 > 2.0；
则 repaired LQ 当前实现不具备 base efficiency 资格。
```

### H2：expression fail 是 trainability / budget fail，而不是 representational capacity fail

如果 LQ 结构理论上能表达 pairwise interaction，则在充分训练或 closed-form coefficient fit 下应该高 R2。

H2 成立标准：

```text
least-squares / closed-form edge coefficient fit:
  pairwise_product_R2 >= 0.90

longer AdamW expression training:
  pairwise_product_R2 >= 0.85

and composition delta vs MLP >= -0.02
```

H2 不成立标准：

```text
least-squares 也 pairwise_product_R2 < 0.85；
说明 architecture / implementation 没有 interaction capacity。
```

### H3：current repaired LQ 的核心病灶是 lift conditioning，而不是 basis usage 不足

P2 中 basis usage 有正信号，但 lift condition 极端高。因此假设：

$$
\text{Training instability} \approx \text{bad lift conditioning} + \text{weak signal consistency}.
$$

H3 成立标准：

```text
condition repair 后：
  lift_condition_proxy 至少下降 100x；
  pairwise_product_R2 不下降；
  mean val acc 不下降；
  perturb_logit_drift_p95 下降；
  signal_consistency_real 上升；
  margin_p10 / CEp99 hard gate 改善。
```

### H4：functional candidates 失败是因为它们没有相对 AdamW controls 的互补性

D8-D11 与 controls 的 best geometry score 非常接近，但没有超过 control best。假设当前 functional direction 与 AdamW task direction/controls 太接近，或只是在已有 task direction 上做微小重标定。

H4 成立标准：

```text
AdamW-orthogonal / control-residual functional directions
在 one-step/five-step audit 中满足：

  beats AdamWParallelDirection；
  beats RandomMatchedNorm；
  beats SNR-only；
  beats GeometryOnlyNoSNR；

并且 paired geometry delta beyond control >= 10%。
```

### H5：不同数据集只能作为 failure slice，不能作为 controller input

Fashion 正、MNIST/KMNIST 部分负，说明数据分布差异有诊断价值。但 official method 不能按 dataset name branch。

H5 成立标准：

```text
任何 repair / functional rule 都只使用 train-stream features：
  lift_condition
  basis_entropy
  margin_p10
  CEp99
  signal_consistency
  perturb_drift
  SNR/offdiag
  step/cost metrics

不能使用 dataset_name、class-specific threshold、seed-specific rule。
```

---

## 7. v12.2 并行实验总览

为了加快进度，v12.2 分成五条并行线。每条线都有自己的 stop rule。

```text
Track A: Timing Truth Audit
  判断 step ratio 6.87 是否可信。

Track B: Expression Truth Audit
  判断 pairwise/product 失败是否真实表达力失败。

Track C: Global LQ Conditioning Repair
  修 lift condition / signal consistency，不按数据集调参。

Track D: Passive Geometry Trajectory Re-test
  重新确认 geometry 信号是否随训练稳定，而不只看 final checkpoint。

Track E: Functional Complementarity Audit
  只做 diagnostic，证明 functional 能否打过 controls。
```

执行顺序可以并行：

```text
A 和 B 立即并行；
C 等 B 的初步结果，但可以先实现候选；
D 在 A/B/C survivor 上跑；
E 在当前 R2 与 C survivor 上都可 diagnostic，但 official promotion 依赖 A/B/C。
```

---

# 8. Track A：Timing Truth Audit

## 8.1 目标

确定本轮 step ratio `6.872` 是否真实反映 repaired LQ base 效率。

## 8.2 方法

比较以下方法：

```text
A0-MLP-same-param-AdamW
A1-MLP-same-step-or-same-FLOPs-AdamW
A2-LQ-t2-h256-current
A3-R2-LQ-fanin-output-scale-confirmed-current
A4-R2-LQ-clean-fastpath-no-geometry-hooks
A5-R2-LQ-compiled-warm
A6-R2-LQ-compact-recompute
A7-R2-LQ-v12-runner-profile-path
```

A4/A5/A6 的目的不是改模型，而是隔离 timing path：

```text
no geometry snapshot；
no P2 battery；
no P4 probe；
no extra logging；
warmup >= 50 steps；
measure >= 200 steps；
separate cold/compile/warm；
profile same batch/hidden shape；
CUDA synchronize around measured regions。
```

## 8.3 必须记录指标

```text
method_id
dataset
seed
batch_size
hidden_dim
forward_ms_mean
forward_ms_p50
forward_ms_p90
forward_ms_q90
backward_ms_mean
backward_ms_p50
backward_ms_p90
step_ms_mean
step_ms_p50
step_ms_q90
optimizer_ms_mean
geometry_hook_ms
probe_hook_ms
logging_ms
compile_time_ms
cold_step_ms
warm_step_ms
step_ratio_vs_MLP
forward_ratio_vs_MLP
backward_ratio_vs_MLP
compact_memory_ratio
conservative_memory_ratio
peak_allocated_MB
peak_reserved_MB
kernel_count_forward
kernel_count_backward
graph_break_count
recompile_count
```

## 8.4 可视化

必须生成：

```text
A_timing_ratio_bar.svg
A_cold_vs_warm_step.svg
A_step_time_waterfall.svg
A_geometry_hook_overhead.svg
A_memory_ratio_bar.svg
A_kernel_count_vs_step_ratio.svg
A_clean_vs_v12_runner_scatter.svg
```

## 8.5 判断标准

Timing clean pass：

$$
\operatorname{step\_q90}_{clean}/\operatorname{step\_q90}_{MLP}\leq1.25.
$$

Official timing pass：

$$
\operatorname{step\_q90}_{official}/\operatorname{step\_q90}_{MLP}\leq1.10.
$$

If clean pass but v12 runner fail:

```text
结论：v12.1 P1 efficiency fail 主要是 measurement/instrumentation artifact。
Codex 下一步：修 runner，把 official profiling 与 diagnostic profiling 分离。
```

If clean fail:

```text
结论：repaired LQ 当前实现不满足 MLP-like efficiency。
Codex 下一步：暂停 functional official route，做 kernel/implementation repair。
```

---

# 9. Track B：Expression Truth Audit

## 9.1 目标

判断 expression battery 失败的根因。

## 9.2 方法

对每个 target 做四种训练/拟合协议：

```text
Protocol B0:
  v12.1 original setting
  expression_steps = 60
  train = 512
  val = 256

Protocol B1:
  longer optimization
  expression_steps = 1000
  same train/val

Protocol B2:
  high-budget optimization
  expression_steps = 5000
  train = 4096
  val = 2048

Protocol B3:
  frozen lift + ridge / least-squares coefficient fit
  用 closed-form 或 LBFGS 只拟合最后 edge coefficients

Protocol B4:
  oracle capacity probe
  random target-independent features fixed，solve linear/quadratic readout
```

比较方法：

```text
M0-MLP-same-param
M1-QuadraticFeatureMLP
M2-LQ-t2-h256
M3-R2-LQ-current
M4-R2-LQ-clean-fastpath
M5-R2-LQ-no-fanin-output-scale
M6-R2-LQ-identity-lift-only
M7-R2-LQ-quadratic-edge-only
```

## 9.3 Targets

```text
E0-additive
E1-pairwise-product
E2-composition
E3-local-XOR
E4-high-frequency
E5-noise-stress
E6-rotated-pairwise-product
E7-low-rank-composition
E8-random-quadratic-form
```

新增 E6/E8 是为了避免 pairwise target 只测到某个实现特例。它们不用于调参，只用于表达诊断。

## 9.4 必须记录指标

```text
target_id
method_id
protocol_id
train_R2
val_R2
test_R2
R2_delta_vs_MLP
R2_delta_vs_QuadraticFeatureMLP
steps_to_R2_0_80
steps_to_R2_0_90
final_train_loss
final_val_loss
optimization_gap = LS_R2 - AdamW_R2
representation_gap = MLP_R2 - LS_R2
gradient_norm_mean
gradient_norm_p95
basis_usage_entropy
dead_basis_fraction
lift_effective_rank
lift_condition_proxy
quadratic_channel_norm
linear_channel_norm
residual_over_linear
```

## 9.5 可视化

```text
B_expression_R2_bar_by_protocol.svg
B_pairwise_learning_curve.svg
B_LS_vs_AdamW_R2_scatter.svg
B_optimization_gap_heatmap.svg
B_representation_gap_heatmap.svg
B_quadratic_channel_usage.svg
B_lift_condition_vs_R2.svg
```

## 9.6 判断标准

Expression representational pass：

$$
R^2_{LS,pairwise}\geq0.90.
$$

Expression trainability pass：

$$
R^2_{AdamW,pairwise}\geq0.85.
$$

Composition pass：

$$
R^2_{composition,KAN}-R^2_{composition,MLP}\geq-0.02.
$$

Failure routes:

```text
Case B1: LS high, AdamW low
  根因：optimization / initialization / conditioning。
  Codex 尝试：conditioning repair、quadratic gate init、LR schedule、role-wise but dataset-agnostic optimizer。

Case B2: LS low
  根因：architecture or implementation capacity failure。
  Codex 尝试：检查 quadratic edge formula、lift-output wiring、fanin-output-scale、edge parameter coverage、manual equivalence。

Case B3: old LQ high, repaired R2 low
  根因：repair broke interaction。
  Codex 尝试：恢复 old LQ interaction path，只保留必要 scaling repair。

Case B4: MLP and LQ both low
  根因：target/budget issue。
  Codex 尝试：增加 expression steps / target normalization，但不能用它包装成功。
```

---

# 10. Track C：Global LQ Conditioning Repair

## 10.1 目标

在不牺牲表达力和效率的前提下，修复 lift conditioning、signal consistency 和 perturb stability。

当前主要病灶：

```text
lift_condition_proxy = 1.1916711215468089e14
signal_consistency_real 低于 MLP
perturb_logit_drift_p95 高于 MLP
P3 hard gates: CEp99 / margin_p10 fail
```

## 10.2 Repair candidates

所有 repair 都必须 dataset-agnostic。

```text
C0-R2-LQ-current

C1-centered-quadratic-basis:
  使用 centered quadratic features，例如 $x^2-\mathbb{E}[x^2]$ 或 batch-independent fixed scale。
  目标是降低 quadratic channel 与 linear channel 共线性。

C2-orthogonal-identity-lift-init:
  lift 初始化为近正交 / row-normalized identity-like map。
  不引入 learnable non-KAN normalization。

C3-fixed-scale-lift-standardization:
  使用 fixed non-learnable scale，来自 train-set global statistics 或 synthetic-independent constant。
  禁止 per-dataset controller。

C4-residual-gated-quadratic:
  quadratic edge 初始小 gate $g_q$，训练中统一 schedule。
  schedule 只依赖 step，不依赖 dataset。

C5-fanin-output-scale-decoupled:
  拆开 fanin scale 与 output scale，避免一个 scale 同时控制 condition 与 logit amplitude。

C6-low-condition-quadratic-parameterization:
  用 normalized polynomial basis 或 Legendre-like quadratic basis，保持 quadratic capacity。

C7-condition-preserving-weight-decay-path:
  只改参数化和 update scaling，不改 loss。
```

## 10.3 实验设置

先跑 synthetic + small task：

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

train_size:
  1024 for fast triage
  4096 for survivor confirm

epochs:
  5 for triage
  20 for survivor
```

## 10.4 必须记录指标

Base task：

```text
val_acc
test_acc_sanity
val_loss
val_loss_auc_time
val_loss_auc_step
ECE
NLL
CEp99
margin_p10
classwise_acc
```

Expression：

```text
pairwise_product_R2
composition_R2
high_frequency_R2
local_XOR_R2
LS_R2_pairwise
optimization_gap_pairwise
```

Condition / geometry：

```text
lift_condition_proxy
lift_effective_rank
basis_usage_entropy
dead_basis_fraction
curvature_debt
perturb_logit_drift_p95
local_jacobian_norm_p95
signal_consistency_real
noise_leak
real_noise_consistency_gap
```

Efficiency：

```text
step_q90_ratio_clean
step_q90_ratio_official
forward_q90_ratio
backward_q90_ratio
compact_memory_ratio
conservative_memory_ratio
```

## 10.5 可视化

```text
C_condition_vs_pairwise_R2.svg
C_condition_vs_val_acc_delta.svg
C_signal_consistency_vs_NLL.svg
C_noise_leak_vs_effective_rank.svg
C_margin_CEp99_trajectory.svg
C_task_expression_efficiency_pareto.svg
C_repair_gate_dashboard.svg
```

## 10.6 判断标准

A repair candidate survives if:

$$
R^2_{pairwise}\geq0.85,
$$

$$
\Delta Acc_{val}(KAN-MLP)\geq-0.005,
$$

$$
\operatorname{step\_q90\_ratio}\leq1.25,
$$

$$
\operatorname{lift\_condition}_{repair}\leq0.01\cdot\operatorname{lift\_condition}_{current},
$$

and at least two of:

$$
\operatorname{signal\_consistency}_{repair}>\operatorname{signal\_consistency}_{current},
$$

$$
\operatorname{perturb\_drift}_{repair}<\operatorname{perturb\_drift}_{current},
$$

$$
\operatorname{CEp99}_{repair}<\operatorname{CEp99}_{current},
$$

$$
\operatorname{margin}_{p10,repair}>\operatorname{margin}_{p10,current}.
$$

If no repair survives:

```text
Codex route:
  stop functional promotion；
  return to base architecture design；
  compare old LQ-t2, R2 repaired, and D2 compositional FullEdge as fallback diagnostics。
```

---

# 11. Track D：Passive Geometry Trajectory Re-test

## 11.1 目标

区分 final checkpoint geometry 好信号与 trajectory hard gate 失败。

## 11.2 方法

对 Track C survivors 与 current R2 进行 longer trajectory audit：

```text
epochs:
  20
  30 optional

checkpoints:
  epoch 0
  every 1 epoch for first 5 epochs
  every 5 epochs after
```

## 11.3 必须记录 trajectory metrics

```text
checkpoint_step
checkpoint_time_sec
val_loss
val_acc
ECE
NLL
CEp99
margin_p10
effective_rank_hidden
basis_usage_entropy
lift_condition_proxy
curvature_debt
perturb_logit_drift_p95
signal_consistency_real
noise_leak
local_jacobian_norm_p95
val_loss_auc_time_so_far
geometry_debt_auc_time
tail_risk_auc_time
```

Define geometry debt:

$$
D_G(t)=z(\operatorname{lift\_condition})+z(\operatorname{perturb\_drift})+z(\operatorname{CEp99})-z(\operatorname{basis\_entropy})-z(\operatorname{effective\_rank}).
$$

This is diagnostic only, not official score.

## 11.4 可视化

```text
D_geometry_trajectory_panel.svg
D_tail_metrics_trajectory.svg
D_condition_trajectory_logscale.svg
D_rank_entropy_trajectory.svg
D_val_loss_vs_geometry_debt.svg
D_AUC_time_vs_final_checkpoint_scatter.svg
```

## 11.5 判断标准

Trajectory pass:

```text
P3 hard gates pass on aggregate；
not only final checkpoint pass。
```

If final checkpoint good but trajectory bad:

```text
Codex 尝试：
  schedule-independent conditioning repair；
  early stabilization；
  warmup quadratic gate；
  optimizer recipe；
  不做 dataset-specific thresholds。
```

---

# 12. Track E：Functional Complementarity Audit

## 12.1 目标

不要继续问“functional candidate 是否有 geometry score”，而要问：

$$
\boxed{
\text{functional 是否提供 controls 无法提供的互补几何改进？}
}
$$

## 12.2 候选方向

Baseline controls：

```text
E0-TaskOnlyAdamW
E1-NoOpMatchedOverhead
E2-RandomMatchedNorm
E3-AdamWParallelDirection
E4-ShuffledPayload
E5-ShuffledEvent
E6-SNROnlyGate
E7-GeometryOnlyNoSNR
```

Existing candidates：

```text
E8-SNRProjectedGeometry
E9-BasisEntropyRebalance
E10-LiftConditionRepair
E11-TailStabilityCorrection
```

New complementarity candidates：

```text
E12-AdamWOrthogonalLiftRepair:
  先构造 lift repair direction $d_{lift}$，
  再去掉 task direction 投影：
  $d^\perp_{lift}=d_{lift}-\frac{\langle d_{lift},d_{task}\rangle}{\|d_{task}\|^2+\epsilon}d_{task}$。

E13-FunctionPreservingConditionRepair:
  在 logit drift bound 下最小化 lift condition proxy。
  目标不是 task descent，而是坐标修复。

E14-TailMarginResidualRepair:
  只针对 CEp99 / margin_p10 tail，
  但必须 old-batch no-harm。

E15-SignalReservoirSeparationRepair:
  基于 real/noise split consistency，
  保留 real signal，减少 noise leak。

E16-ControlResidualGeometry:
  从 functional direction 中扣除 AdamWParallel / RandomMatchedNorm 的可解释部分，
  只保留 residual functional component。
```

## 12.3 Direction formula examples

AdamW-orthogonal repair:

$$
d_{geo}^{\perp}
=
d_{geo}
-
\frac{\langle d_{geo},d_{task}\rangle}{\|d_{task}\|^2+\epsilon}
d_{task}.
$$

Task-safe combined direction:

$$
d_{new}=d_{task}+\lambda d_{geo}^{\perp}.
$$

Backtracking condition:

$$
L_{probe}(\theta+d_{new})\leq L_{probe}(\theta+d_{task})+\epsilon_L.
$$

Function-preserving condition repair:

$$
\min_{\Delta\theta}
\operatorname{CondProxy}(\theta+\Delta\theta)
$$

subject to:

$$
\|f_{\theta+\Delta\theta}(X)-f_{\theta}(X)\|_{p95}\leq \tau_{logit},
$$

$$
L_{probe}(\theta+\Delta\theta)\leq L_{probe}(\theta)+\tau_L.
$$

## 12.4 设置

Run on:

```text
Base set 1:
  current R2 repaired LQ

Base set 2:
  best Track C repair survivor, if available

datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

checkpoints:
  early
  middle
  late
```

## 12.5 必须记录指标

Direction metrics：

```text
candidate_id
base_id
dataset
seed
checkpoint_id
norm_candidate
norm_task
norm_ratio
cos_with_task
cos_with_adamw_parallel
cos_with_random_control
orthogonal_component_ratio
accepted_lambda
lambda_zero_rate
backtrack_count
```

Task safety：

```text
train_descent
probe_descent
holdout_descent
holdout_descent_ratio_vs_task
bad_step_rate
val_loss_delta_one_step
val_loss_delta_five_step
```

Geometry deltas：

```text
delta_lift_condition_proxy
delta_basis_usage_entropy
delta_effective_rank
delta_curvature_debt
delta_perturb_logit_drift_p95
delta_signal_consistency_real
delta_noise_leak
delta_CEp99
delta_margin_p10
delta_ECE
```

Control resistance：

```text
paired_delta_vs_NoOp
paired_delta_vs_RandomMatchedNorm
paired_delta_vs_AdamWParallelDirection
paired_delta_vs_SNROnly
paired_delta_vs_GeometryOnlyNoSNR
beats_all_strong_controls
control_gap_mean
control_gap_ci_low
```

Cost：

```text
direction_compute_ms
probe_compute_ms
amortized_overhead_estimate
peak_memory_delta
```

## 12.6 可视化

```text
E_functional_vs_controls_paired_violin.svg
E_control_gap_by_candidate.svg
E_cosine_with_task_heatmap.svg
E_orthogonal_component_vs_gain.svg
E_lambda_backtracking_hist.svg
E_one_step_vs_five_step_delta.svg
E_geometry_delta_radar.svg
E_cost_vs_control_gap_pareto.svg
```

## 12.7 判断标准

Diagnostic candidate can be promoted only if:

$$
\operatorname{bad\_step\_rate}\leq0.02,
$$

$$
\operatorname{holdout\_descent\_ratio}\geq0.95,
$$

$$
\operatorname{control\_gap\_mean}\geq0.10\cdot|\operatorname{control\_best}| \quad \text{or} \quad \operatorname{control\_gap\_mean}\geq0.02,
$$

$$
\operatorname{control\_gap\_ci\_low}>0,
$$

and:

$$
\operatorname{amortized\_overhead}\leq0.05.
$$

If candidate is task-safe but not control-resistant:

```text
do not tune threshold；
do not open P5；
record as diagnostic only。
```

---

# 13. Official promotion rules

v12.2 必须有严格 promotion tree。

## 13.1 Base promotion

Base is qualified only if:

```text
Timing:
  clean step_q90 <= 1.25
  official step_q90 <= 1.10 after runner fix

Expression:
  pairwise R2 >= 0.85
  composition delta vs MLP >= -0.02

Task:
  near_pass_rate >= 0.80 for triage
  10-seed near-pass for confirm

Geometry:
  P3 hard gates pass or only one minor fail with documented trajectory improvement
```

If base not qualified:

```text
functional remains diagnostic；
P5 short-run not opened。
```

## 13.2 Functional promotion

Functional candidate is official only if base is qualified and candidate satisfies:

```text
task-safe；
control-resistant；
amortized overhead <= 5%；
no rank/basis collapse；
no tail/margin degradation；
not explainable by MLP analog or QuadraticFeatureMLP analog。
```

## 13.3 External-ready promotion

External-ready only after:

```text
same-param MLP；
same-FLOPs MLP；
same-wall-clock MLP；
QuadraticFeatureMLP；
MLP analog functional maintenance；
CE-only；
no teacher；
no loss modification；
no dataset-specific rule。
```

---

# 14. Required artifacts

Codex must write the following directory:

```text
results/v12_2_base_truth_conditioning_functional_complementarity/<timestamp>/
```

Required CSV / JSON:

```text
a_timing_truth.csv
a_timing_breakdown.csv
a_runner_overhead_audit.csv

b_expression_truth.csv
b_expression_learning_curves.csv
b_ls_fit_results.csv
b_expression_failure_taxonomy.csv

c_conditioning_repair.csv
c_repair_task_trace.csv
c_repair_expression_trace.csv
c_repair_geometry_snapshot.csv

d_geometry_trajectory.csv
d_geometry_auc.csv

e_functional_complementarity_direction.csv
e_one_step_probe.csv
e_five_step_probe.csv
e_control_matrix.csv
e_lambda_backtracking.csv
e_cost_profile.csv

gate_dashboard.csv
failure_table.csv
route_decision.json
provenance_audit.csv
hash_manifest.json
```

Required figures:

```text
figures/A_timing_ratio_bar.svg
figures/A_clean_vs_v12_runner_scatter.svg
figures/B_expression_R2_bar_by_protocol.svg
figures/B_LS_vs_AdamW_R2_scatter.svg
figures/C_condition_vs_pairwise_R2.svg
figures/C_task_expression_efficiency_pareto.svg
figures/D_geometry_trajectory_panel.svg
figures/D_val_loss_vs_geometry_debt.svg
figures/E_functional_vs_controls_paired_violin.svg
figures/E_control_gap_by_candidate.svg
figures/E_cost_vs_control_gap_pareto.svg
figures/gate_dashboard.svg
figures/failure_taxonomy_heatmap.svg
```

---

# 15. Codex failure handling rules

为了加速实验，Codex 在 gate 不满足时不要停在“失败”两个字，而要按下面规则继续做第一层定位。

## 15.1 If timing fails

If:

```text
step_q90_ratio > 2.0
```

Codex must automatically run:

```text
1. no-geometry-hook profile；
2. no-P4-probe profile；
3. warmup 50 / measure 200 profile；
4. compile-warm profile；
5. compact/recompute profile；
6. kernel count profile；
7. logging overhead profile。
```

If clean profile passes:

```text
fix official runner accounting。
```

If clean profile fails:

```text
open implementation repair：
  remove Python loops；
  fuse quadratic/lift kernels；
  reduce temporary tensors；
  compare old LQ-t2 path。
```

## 15.2 If pairwise expression fails

If:

```text
pairwise_product_R2 < 0.85
```

Codex must automatically run:

```text
1. longer optimization 1000/5000 steps；
2. least-squares coefficient fit；
3. old LQ-t2 vs repaired R2 comparison；
4. no-fanin-output-scale ablation；
5. identity-lift-only / quadratic-only ablation；
6. parameter coverage audit；
7. manual equivalence audit for quadratic edge formula。
```

If LS high but Adam low:

```text
try conditioning repair and optimizer/init repair。
```

If LS low:

```text
treat as architecture/implementation failure。
```

## 15.3 If lift condition remains huge

If:

```text
lift_condition_proxy > 1e8
```

Codex must try:

```text
1. centered quadratic basis；
2. orthogonal lift init；
3. fixed scale standardization；
4. decoupled fanin-output scale；
5. Legendre-like quadratic basis；
6. small quadratic residual gate。
```

Do not add a geometry loss unless explicitly separated as a diagnostic, because the official route is CE-only.

## 15.4 If task unstable by dataset

Codex may diagnose by dataset but must not tune by dataset. It should run:

```text
classwise margin；
CEp99；
hard class pairs；
lift condition per checkpoint；
signal consistency per checkpoint；
basis entropy per checkpoint。
```

Allowed repair:

```text
global initialization；
global scale；
global schedule；
global optimizer recipe；
global parameterization。
```

Forbidden repair:

```text
if dataset == KMNIST then threshold = ...
if dataset == Fashion then lambda = ...
dataset-specific controller。
```

## 15.5 If functional fails controls

If:

```text
beats_all_strong_controls = 0
```

Codex must not lower threshold. It should run:

```text
1. AdamW-orthogonal functional residual；
2. function-preserving condition repair；
3. control-residual geometry direction；
4. paired delta distribution, not best score only；
5. leave-dataset-out control-gap audit。
```

If still fails:

```text
functional route remains diagnostic；
return to base repair。
```

---

# 16. What not to do in v12.2

Do not do:

```text
1. Do not open P5 short-run while base gate fails.
2. Do not lower P4 control gate.
3. Do not tune lambda by dataset.
4. Do not claim geometry success from final checkpoint only.
5. Do not ignore expression battery.
6. Do not compare functional candidate only to task-only AdamW; always include strong controls.
7. Do not use future labels / replay outcomes as training-time controller targets.
8. Do not introduce teacher / distillation / loss modification for official route.
9. Do not expand to Conv / Former before FC PureKAN base is qualified.
```

---

# 17. Expected route decisions

At the end of v12.2, route_decision.json must choose one:

```text
R0-MeasurementArtifact:
  v12.1 timing/base failure mostly due runner instrumentation.
  Next: fix runner and rerun base gate.

R1-BaseRepairSuccess:
  timing, expression, task, conditioning all pass.
  Next: official functional audit may open.

R2-ExpressionImplementationFailure:
  pairwise LS / equivalence fails.
  Next: repair LQ formula / interaction channel.

R3-TrainabilityConditioningFailure:
  LS expression passes but AdamW expression/task fails due condition.
  Next: conditioning repair / init / optimizer.

R4-EfficiencyFailure:
  clean timing still fails.
  Next: kernel path repair before any functional work.

R5-FunctionalComplementarityFailure:
  base passes diagnostic enough, but functional fails controls.
  Next: redesign functional directions; no P5.

R6-FunctionalDiagnosticSuccess:
  base not yet official, but functional complementarity shows control-resistant signal.
  Next: rerun after base qualification.

R7-OfficialFunctionalOpen:
  base qualified and functional diagnostic passes.
  Next: P5 short-run controlled training.
```

---

## 18. Final conclusion

这次实验最大的价值不是“证明 repaired LQ 失败”，而是把问题定位得更硬了：

$$
\boxed{
\text{当前不是 functional update 缺一个小技巧，而是 base truth、interaction expression、conditioning、control-resistant functional complementarity 四件事还没同时闭合。}
}
$$

后续应该把实验速度提上去，但不是靠乱跑更多方法，而是靠并行拆解：

```text
A: timing truth；
B: expression truth；
C: conditioning repair；
D: geometry trajectory；
E: functional complementarity。
```

只要 A/B/C 不过，P5 就不该开。  
只要 E 不打过 strong controls，functional 就不能 claim。  
只有 repaired LQ 同时满足表达、效率、任务稳定和几何 hard gates 后，functional update 才有资格进入 official training route。
