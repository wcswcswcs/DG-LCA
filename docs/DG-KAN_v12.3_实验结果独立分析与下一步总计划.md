# DG-KAN v12.3 实验结果独立分析与下一步总计划

> 目标：独立分析 v12.2 Base Truth / Conditioning / Functional Complementarity 及其后续 `signed_pair_cover`、`PCA-whiten` 修复结果，判断是否有进展、卡在哪里、是否仍在正确道路上，并给出可直接交给 Codex 并行执行的下一步实验计划。
>
> 公式格式：Typora 友好，只使用 `$...$` 或 `$$...$$`。
>
> 核心原则：不针对 MNIST / Fashion-MNIST / KMNIST 调参；可以按数据集诊断 failure slice，但 official rule、candidate design、gate、threshold 不允许使用 dataset name 分支。所有实验保持 CE-only、no teacher、no distillation、no loss modification、no sampler/class-weight、no fake/proxy/cpu offload。Functional update 仍是 update rule，不是 loss。

---

## 0. 一句话判断

这次实验**有进展，但不是能力进展，而是问题本质定位进展**。

当前最重要的结论不是“LQ 不行”，也不是“functional update 不行”，而是：

$$
\boxed{
\text{LQ quadratic edge formula 本身能表达 pairwise；真正 blocker 是 lift / interaction frame 同时满足 coverage、conditioning、task stability。}
}
$$

更具体地说：

```text
clean timing 已经基本不是主 blocker；
quadratic formula 不是主 blocker；
局部 pairwise coverage 可以被 signed_pair_cover 修复；
conditioning 可以被 PCA-whiten 局部修复；
但 coverage、conditioning、同参 vision task stability 还不能同时满足；
functional update 继续不能打开，因为 base 不合格，而且 diagnostic functional 仍打不过 controls。
```

所以这轮不是原地踏步。它把问题从模糊的：

```text
R2-LQ base not qualified
```

推进到了更精确的：

```text
需要设计一个 dataset-agnostic、low-condition、global quadratic coverage-preserving、task-stable 的 LQ lift / interaction frame。
```

但从终极目标看，进度仍然慢，而且离目标还远。原因是我们现在还没有一个合格 base，更没有 functional update 的独立优势。因此下一步不能继续小修 functional direction，也不能调低 gate，而应该集中火力解决 **Balanced Interaction-Cover Lift**。

---

## 1. 当前总目标重新固定

本项目目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，而是验证一个更根本的假设：

$$
\boxed{
\text{PureKAN base 的表达坐标} + \text{functional update 的几何维护}
\Rightarrow
\text{next-generation MLP alternative}
}
$$

这个目标必须拆成三层因果链。

### 1.1 Base 层目标

首先必须有一个合格 PureKAN base：

$$
\boxed{
\text{Strict FC-PureKAN LQ base} \approx \text{same-param / same-time MLP envelope}
}
$$

它至少要满足：

```text
1. 表达力不打折：pairwise / composition / rotated quadratic / random quadratic battery 不能明显输 MLP。
2. 任务能力不打折：same-param MLP 对照下 near-pass 或 full-pass。
3. 系统效率不打折：forward/backward/step 不能明显慢于 MLP。
4. 几何不塌：lift condition、basis usage、rank、tail、ECE、perturb drift 不坏。
5. Strict PureKAN：无 ordinary MLP path、无 external residual、无 teacher/loss trick、KAN path 不依赖 loss.backward graph。
```

如果 base 不合格，functional update 的成功没有系统意义。

### 1.2 Functional 层目标

在同一个合格 base 上验证：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{task}}
+
\lambda_t\Delta\theta_{\text{functional-geo}}.
$$

其中：

```text
Delta theta_task: AdamW / stable manual optimizer 的 task descent。
Delta theta_functional-geo: 低频、task-safe、geometry-aware 的 functional correction。
```

Functional update 不能只是另一个 task optimizer，不能只在 output space 制造扰动，不能被 AdamWParallel / RandomMatchedNorm / SNR-only controls 解释。

### 1.3 Beyond-MLP 层目标

最终才比较：

```text
MLP + AdamW
LQ + AdamW
LQ + AdamW + functional geometry maintenance
MLP + analogous geometry maintenance
QuadraticFeatureMLP + analogous maintenance
Random / NoOp / AdamWParallel controls
```

只有当 `LQ + functional` 同时赢过 `LQ + AdamW` 和 strong controls，并且不输 same-param / same-time MLP，才可以谈 next-generation MLP alternative。

---

## 2. 对本轮结果的独立分析

## 2.1 有进展吗？有，但不是最终能力进展

本轮至少完成了五件有价值的事。

### 2.1.1 Timing artifact 被基本剥离

v12.1 中出现过 step ratio 极高的问题，容易让人误以为 R2-LQ 架构本身慢。v12.2 的 timing truth audit 说明 clean path 下：

```text
R2 clean fastpath step q90 ratio = 1.2043
compiled-warm step ratio = 0.8278
后续 PCA-whiten run clean step ratio = 1.0306
```

而 official diagnostic path 的 `45x-48x` 主要来自 geometry hook / probe hook 混入 timed path：

```text
geometry hook mean 约 22-23 ms
probe hook 约 0.56 ms
```

独立判断：

$$
\boxed{
\text{architecture clean step timing 目前不是首要 blocker；diagnostic hook accounting 才是 official timing path 的问题。}
}
$$

这不是 base pass，因为 expression / task / condition gate 仍失败。但它极大缩小了问题范围。

### 2.1.2 Pairwise 表达能力不是简单失败

v12.2 中有一个看似矛盾但非常重要的结果：

```text
R2 frozen-lift LS R2 = 0.3787
R2 high-budget AdamW B2 pairwise R2 = 0.999279
quadratic-edge-only B2 pairwise R2 = 0.999443
oracle quadratic readout = 1.0
```

这说明不能说“LQ 不会表达 pairwise”。更准确是：

$$
\boxed{
\text{trainable LQ 有 pairwise capacity，但当前 frozen lift / coefficient truth audit 不可用。}
}
$$

也就是说，问题不在 quadratic edge formula 的理论容量，而在 lift 初始化、quadratic span coverage、coefficient fit/equivalence 与 trainability 的关系。

### 2.1.3 Quadratic edge 公式被证明不是错的

后续 interaction equivalence repair 里，manual equivalence audit 通过：

```text
manual equivalence val/test R2 = 1.0
max abs error = 1.19e-7
```

这说明 LQ 的 plus/minus quadratic construction 可以精确表达 pairwise product。根因进一步从：

```text
公式可能错了
```

收窄成：

```text
当前随机/frozen lift 的 interaction coverage 不合格。
```

### 2.1.4 signed_pair_cover 证明“局部 coverage 可修”

`signed_pair_cover` 对 E1 pairwise-product 很有效：

```text
E1 frozen-lift LS val R2 = 0.992139
E1 B2 high-budget R2 = 0.999311
E2 composition B1 delta vs MLP = -0.010846
```

这说明结构性初始化确实能修一个局部 interaction coverage 缺口。这个结果很重要，因为它证明下一步不是盲目换 primitive，而是可以继续做 lift / interaction frame design。

但它也失败得很清楚：

```text
rotated-pairwise frozen-LS R2 = 0.200855
random-quadratic frozen-LS R2 = 0.317812
vision mean val acc delta = -0.019314
lift condition mean = 3.139e14
survivor rows = 0
```

独立判断：

$$
\boxed{
\text{signed_pair_cover 是局部 pair cover，不是全局 quadratic cover；它修了 E1，但破坏了 task stability / conditioning。}
}
$$

### 2.1.5 PCA-whiten 证明“conditioning 可局部修”，但也不够

PCA-whiten / rank-limited PCA-whiten 做了另一类修复：

```text
C10-data-pca-whiten-h64 clean step ratio = 1.0353
C10 在部分 MNIST/Fashion split 上把 condition 从 1e14 降到 1e8 量级
```

这说明 conditioning 不是不可修。但 C10 的整体结果仍不合格：

```text
mean val acc delta = -0.018229
pairwise R2 = 0.893985，低于 C0/C8
KMNIST condition 仍高
参数量不等价，不能当 same-param success
survivor rows = 0
```

独立判断：

$$
\boxed{
\text{PCA-whiten 是有价值 diagnostic，不是 base survivor；它说明低 condition 与 interaction coverage 之间存在 tradeoff。}
}
$$

---

## 2.2 进度如何？工程推进快于科学推进

从 artifact 和 pipeline 角度，v12.2 进展不慢：

```text
A timing truth 完成；
B expression truth 完成；
C conditioning repair 完成；
D passive geometry trajectory 完成；
E functional complementarity diagnostic 完成；
signed_pair_cover follow-up 完成；
PCA-whiten follow-up 完成；
no-fake/proxy/cpu audit 干净。
```

但是从科学能力角度，进展确实慢：

```text
base_qualified = false
functional_complementarity_pass = false
official_functional_open = false
external_ready = false
```

这两者并不矛盾。现在的慢不是 runner 没跑，而是每次跑完都在排除一种错误解释：

```text
不是 timing 真慢；
不是 quadratic formula 错；
不是 pairwise 完全不可表达；
不是局部 pair cover 不可修；
不是 condition 完全不可降；
而是 coverage + condition + task stability 三者尚未合一。
```

这类进展不立刻涨 accuracy，但它能避免继续把时间浪费在错误方向上。

---

## 2.3 目前卡在哪里？卡在 lift / interaction frame 的三目标不可兼容

现在的主 blocker 不是 functional update，而是 base 的内部坐标系统。

LQ base 的抽象形式可以写成：

$$
h = xA,
$$

$$
\psi(h) = [h, h^2, \text{possibly signed / scaled quadratic channels}],
$$

$$
y = \psi(h)W.
$$

它能表达哪些二次交互，本质上取决于 $A$ 诱导出的 quadratic form span。

一个 quadratic channel 形如：

$$
(x^Ta)^2 = x^T(aa^T)x.
$$

多个 channel 的线性组合是：

$$
\sum_k w_k(x^Ta_k)^2
= x^T\left(\sum_k w_k a_ka_k^T\right)x.
$$

所以 LQ 的 interaction capacity 不是单纯“有没有平方”，而是：

$$
\mathcal{Q}_A = \operatorname{span}\{a_ka_k^T\}_{k=1}^m
$$

能否覆盖任务需要的 quadratic directions，同时 $A$ 的数值条件不能爆。

当前不同 candidate 只满足其中一部分：

| candidate | coverage | condition | task stability | independent conclusion |
|---|---:|---:|---:|---|
| C0 R2 current | pairwise trainable 有能力，但 frozen coverage 弱 | 极差，约 $2.47e14$ | 近但不够，delta $-0.0061$ | 当前最好 baseline，但不合格 |
| C8 signed_pair_cover | E1 coverage 强 | 更差，约 $3.14e14$ | 明显退化，delta $-0.0193$ | 局部 cover 修复，不是 base |
| C9 PCA-whiten h256 | condition 略降 | 仍很差，约 $2.32e14$ | 明显退化，delta $-0.0258$ | full-size whiten 失败 |
| C10 PCA-whiten h64 | condition 局部大幅降低 | mean 仍高，约 $4.56e13$ | 仍退化，delta $-0.0182$ | 低秩 diagnostic，不是 same-param base |

因此真正卡点是：

$$
\boxed{
\text{如何构造一个既覆盖全局 quadratic interaction、又低 condition、又不破坏 task trainability 的 lift frame。}
}
$$

这比“小修某个 threshold / lambda / functional direction”更本质。

---

## 2.4 发现了什么问题？

### 问题 1：之前的 expression battery 太容易被单一 E1 误导

E1-pairwise-product 是必要但不充分的。`signed_pair_cover` 把 E1 frozen-LS 修到 0.99，但 rotated / random quadratic 仍很低，vision task 也下降。这说明下一步 expression battery 必须至少包含：

```text
E1 pairwise-product
E2 composition
E6 rotated-pairwise-product
E8 random-quadratic-form
global matrix span R2
basis condition / coverage entropy
```

如果只看 E1，就会把局部 pair cover 误判为 full interaction cover。

### 问题 2：低 condition 不能通过牺牲 coverage 获得

PCA-whiten 证明 condition 可以降，但同时 pairwise R2 和 task 掉了。以后不能把 condition drop 单独作为成功。必须要求：

$$
\text{ConditionDrop} \land \text{CoverageRetain} \land \text{TaskSafe}.
$$

### 问题 3：task failure 不是单数据集问题

C8 在 MNIST、Fashion、KMNIST 都不同程度掉，尤其 Fashion/KMNIST 更明显；C10 虽在部分 MNIST/Fashion split 降 condition，但 KMNIST 仍高且 task 差。这些可以作为 failure slice 诊断，但不能因此为 KMNIST 单独调结构。正确做法是找 dataset-agnostic 的 frame 设计，让所有数据分布下的 coverage/condition 更稳。

### 问题 4：functional update 现在继续推进会制造假进展

E8-E16 functional diagnostic 的 control gap mean 全部为负；而且 base gate 本来关闭。即使强行打开 P5，也只能得到 diagnostic 行，不可能变成 official success。现在调低 gate 或继续换 functional target，本质是在一个不合格坐标系统上调 correction。

### 问题 5：official timing gate 的设计需要拆分

当前 official diagnostic path 会把 geometry hook 计入 architecture timing。这会导致一个错误现象：base clean step 约 $1.03x$，但 official path 显示 $48x$。下一步必须把：

```text
architecture_clean_step_time
required_online_metric_step_time
optional_offline_diagnostic_time
```

分开计入。否则会把 research diagnostic cost 误判成 model architecture cost。

---

## 2.5 是否在正确道路上？

我的判断是：**方向是正确的，但当前执行仍偏慢，需要更激进地并行化 base design，而不是串行修一个 candidate。**

正确的地方：

```text
1. 没有调低 gate。
2. 没有把 clean timing 当 base success。
3. 没有把 E1 success 当 global success。
4. 没有打开 functional/P5。
5. 没有按 dataset name 调参。
6. 继续做了真实 artifact / no-fake audit。
```

需要改变的地方：

```text
1. 不要再单线试一个 lift repair；下一轮要并行试 6-10 个 dataset-agnostic frame family。
2. 不要再把 expression、condition、task 分开串行判断；要在同一个 scorecard 中同时看三者。
3. 不要再让 functional diagnostic 消耗主资源；它只在 base survivor 出现后恢复。
4. 不要把 5-epoch triage 结果当最终结论；它只用于淘汰与选择，top candidate 必须 20/30 epoch confirm。
```

---

## 2.6 离目标还差多远？

按模块拆：

### Base timing

距离目标：近。

```text
clean step ratio 已经可到 1.03x；compiled-warm 曾到 0.83x；
需要修 accounting，不是主要 architecture blocker。
```

### Base expression

距离目标：中等偏远。

```text
pairwise trainable capacity 有；
formula 正确；
但 frozen/global coverage 不足；
rotated/random quadratic 仍暴露 coverage hole。
```

### Base conditioning

距离目标：远。

```text
current lift condition 约 1e14；
C10 可局部降到 1e8，但 mean 仍高，且 task/coverage 受伤；
需要新的 balanced frame，不是简单 whiten。
```

### Base task stability

距离目标：中等。

```text
C0 delta 约 -0.0061，接近但没过；
C8/C9/C10 修复后反而更差；
说明 task-stable repair 尚未找到。
```

### Functional update

距离目标：远。

```text
base 未过；
diagnostic functional 没打过 controls；
functional official route 必须继续关闭。
```

### Next-gen MLP claim

距离目标：很远。

```text
现在不能 claim beyond MLP；
最多能说：LQ route 已收窄到 lift-frame design problem。
```

---

# 3. 下一步总计划：v12.3 Balanced Interaction-Cover Lift

## 3.1 v12.3 核心目标

v12.3 不再做小修小补。核心目标是验证一个新假设：

$$
\boxed{
\text{存在一个 dataset-agnostic balanced quadratic lift frame，}
\text{能同时满足 global coverage、low condition、task stability、MLP-like timing。}
}
$$

这不是单一 candidate，而是一个并行 frame design 实验。下一步应该把资源集中到：

```text
1. interaction frame 的数学覆盖；
2. lift condition 的数值稳定；
3. task trainability 的同参稳定；
4. clean timing / diagnostic timing accounting；
5. base survivor 出现后再开 functional complementarity。
```

Functional update 在 v12.3 中只保留 frozen diagnostic，不进入 main compute budget。只有 base gate 通过后才打开。

---

## 3.2 v12.3 总假设

### H1：当前 failure 是 frame design failure，而不是 LQ formula failure

已知 manual equivalence 能让 E1 R2 = 1.0，B2 high-budget AdamW 也能让 pairwise 接近 1.0。因此假设：

$$
H1:
\text{formula 可用，失败来自 } A \text{ 的 quadratic frame coverage / condition。}
$$

成立标准：新的 frame family 可以在不改 formula 的情况下提升：

```text
global matrix span R2；
rotated / random quadratic frozen-LS R2；
lift condition；
vision task delta。
```

### H2：单一 coverage repair 或单一 condition repair 都不足

`signed_pair_cover` 修 E1 但 task/condition 坏；PCA-whiten 降部分 condition 但 coverage/task 坏。因此假设：

$$
H2:
\text{需要 coverage 与 conditioning 联合设计，而不是先修其中一个。}
$$

成立标准：candidate 必须同时满足：

$$
CoveragePass \land ConditionPass \land TaskSafePass.
$$

### H3：task collapse 来自 lift spectrum / scale，而不是必须针对 dataset 调参

C8/C10 的 task failure 分布在多个 dataset 上，说明它不是单个数据集上的偶然。假设：

$$
H3:
\text{task failure 来自 frame spectrum、channel scale、optimizer-visible coordinate conditioning。}
$$

成立标准：用同一 dataset-agnostic rule 改善所有 datasets 的 mean/worst delta，不允许 dataset-name branch。

### H4：如果 balanced frame 仍失败，LQ-t2 可能需要结构层级升级

如果多个 balanced frame 都不能同时过 gate，则说明 simple LQ lift/quadratic edge base 可能不足，需要进入下一层 primitive design，例如：

```text
multi-frame LQ；
low-rank + signed-pair hybrid；
depth-2 compositional LQ；
strict FullEdge compositional diagnostic；
```

但这必须在 v12.3 base-frame matrix 失败后才打开，不能现在跳到新路线。

---

# 4. v12.3 实验设计总览

v12.3 分为 7 条并行 track。

```text
Track A: Timing Accounting Repair
Track B: Quadratic Frame Capacity Audit
Track C: Balanced Lift Candidate Factory
Track D: 5-epoch Base Gate Triage
Track E: Geometry / Signal / Tail Passive Battery
Track F: 20/30-epoch Survivor Confirmation
Track G: Functional Complementarity Re-entry Gate, only if base passes
```

执行顺序不再完全串行。建议按 batch 并行：

```text
Batch 1:
  Track A + Track B + Track C cheap expression/capacity audit。

Batch 2:
  Track D 对 top candidates 跑 5-epoch triage。

Batch 3:
  Track E 对 top 2-3 candidates 跑 passive geometry trajectory。

Batch 4:
  Track F 对真正 near-survivor 跑 20/30 epoch confirm。

Batch 5:
  只有 Track F pass 后，Track G 打开 functional diagnostic。
```

---

# 5. Track A：Timing Accounting Repair

## 5.1 目标

把 architecture step timing 与 diagnostic hook timing 分离，避免 timing gate 继续误导 base 判断。

## 5.2 必跑方法

```text
A0-MLP-same-param-AdamW
A1-MLP-same-time/manual diagnostic, optional
A2-R2-LQ-current-clean
A3-R2-LQ-current-compiled-warm
A4-best-new-frame-clean
A5-best-new-frame-compiled-warm
A6-official-runner-no-hooks
A7-official-runner-with-hooks-diagnostic
```

## 5.3 必须记录 CSV 字段

```text
method
candidate_id
uses_geometry_hook
uses_probe_hook
uses_offline_diagnostic
compiled
warmup_steps
measure_steps
forward_q50_ms
forward_q90_ms
backward_q50_ms
backward_q90_ms
optimizer_q50_ms
optimizer_q90_ms
step_q50_ms
step_q90_ms
architecture_step_q90_ratio
online_metric_step_q90_ratio
offline_diagnostic_step_q90_ratio
geometry_hook_ms_mean
geometry_hook_ms_q90
probe_hook_ms_mean
probe_hook_ms_q90
memory_compact_ratio
memory_conservative_ratio
kernel_count
sync_count
allocation_count
```

## 5.4 判断标准

Architecture timing pass：

$$
StepRatio_{arch,q90}\leq1.10
$$

或 exploratory pass：

$$
StepRatio_{arch,q90}\leq1.25.
$$

Diagnostic hook accounting pass：

```text
architecture_step_time_excludes_optional_offline_hooks = 1
online_metric_step_time is separately reported
```

如果 architecture clean pass 而 official hook path fail，则 route 不应再写 timing blocker，而应写：

```text
R-TimingAccountingSplitNeeded
```

## 5.5 必须可视化

```text
v123_timing_clean_vs_official_bar.svg
v123_timing_waterfall_arch_online_offline.svg
v123_hook_cost_distribution.svg
v123_step_ratio_by_candidate.svg
v123_memory_compact_conservative_bar.svg
```

## 5.6 不满足条件时 Codex 自动尝试

如果 clean step ratio > 1.25：

```text
1. 自动重跑 compiled-warm path。
2. 自动关闭 all diagnostic hooks 后重测。
3. 自动增加 warmup_steps 到 200，measure_steps 到 500。
4. 自动记录 kernel_count / sync_count / allocation_count。
5. 如果 compiled-warm 仍 > 1.25，则标记 candidate timing fail，不进入 Track D。
```

如果 official path > 3x 但 clean path <= 1.25：

```text
1. 不判 candidate fail。
2. 输出 hook accounting failure。
3. 将 geometry hook 改为 offline snapshot 或 amortized low-frequency accounting。
```

---

# 6. Track B：Quadratic Frame Capacity Audit

## 6.1 目标

在不跑 vision task 前，先判断 candidate 的 quadratic frame 是否覆盖任务所需的 interaction space。

核心对象是 lift matrix $A$ 诱导的 quadratic span：

$$
\mathcal{Q}_A = \operatorname{span}\{a_ka_k^T\}_{k=1}^m.
$$

对 signed pair channel，如果使用 plus/minus construction：

$$
(x^T(a+b))^2 - (x^T(a-b))^2 = 4(x^Ta)(x^Tb),
$$

也要把对应 effective quadratic matrix 纳入 span audit。

## 6.2 必测 target

```text
E0-additive
E1-pairwise-product
E2-composition
E3-local-XOR
E4-high-frequency
E6-rotated-pairwise-product
E8-random-quadratic-form
E9-random-lowrank-quadratic
E10-block-local-quadratic
```

新增 target 的目的：

```text
E6/E8 检查 signed_pair_cover 是否只是修固定坐标 pair。
E9 检查低秩 quadratic coverage。
E10 检查局部 block interaction coverage。
```

## 6.3 必须记录指标

```text
candidate_id
target
protocol
train_steps
val_R2
test_R2
delta_vs_MLP
matrix_span_R2
frozen_lift_LS_val_R2
frozen_lift_LS_test_R2
best_trainable_R2
oracle_quadratic_R2
quadratic_frame_rank
quadratic_frame_effective_rank
quadratic_frame_condition
quadratic_frame_coherence
max_pair_coverage
min_pair_coverage
pair_coverage_entropy
rotated_coverage_R2
random_quadratic_coverage_R2
composition_delta_vs_MLP
```

## 6.4 通过标准

Expression triage pass：

$$
R^2_{B2,E1}\geq0.995.
$$

Global frozen coverage pass：

$$
\frac{1}{|T|}\sum_{t\in\{E1,E6,E8,E9,E10\}}R^2_{LS,t}\geq0.85.
$$

Worst-case coverage pass：

$$
\min_{t\in\{E1,E6,E8\}} R^2_{LS,t}\geq0.70.
$$

Composition pass：

$$
\Delta R^2_{E2,\text{B1 vs MLP}}\geq-0.02.
$$

Frame condition exploratory pass：

$$
\kappa_{frame}\leq10^{12}.
$$

Strong pass：

$$
\kappa_{frame}\leq10^{10}.
$$

## 6.5 必须可视化

```text
v123_expression_battery_heatmap.svg
v123_matrix_span_r2_by_target.svg
v123_quadratic_frame_spectrum.svg
v123_pair_coverage_entropy_bar.svg
v123_coverage_vs_condition_pareto.svg
v123_frozenLS_vs_trainableR2_scatter.svg
```

## 6.6 不满足条件时 Codex 自动尝试

如果 E1 pass 但 E6/E8 fail：

```text
说明 frame 只覆盖 fixed coordinate pair，不覆盖 rotated/global quadratic。
Codex 自动生成 SRHT / random orthogonal / block-orthogonal pair cover variants。
```

如果 E6/E8 pass 但 condition fail：

```text
说明 coverage 有但 frame ill-conditioned。
Codex 自动尝试 Gram-Schmidt / QR orthogonalization / covariance-whitened rescale。
```

如果 condition pass 但 E1/E6/E8 coverage fail：

```text
说明 whiten 过度损失 interaction span。
Codex 自动加入 signed-pair residual subframe，但保持 global orthogonal scaling。
```

如果 composition fail 而 pairwise pass：

```text
说明 single-step pairwise 不足以承载 composed nonlinearity。
Codex 自动测试 two-frame LQ 或 depth-2 compositional LQ diagnostic，但不得打开 functional。
```

---

# 7. Track C：Balanced Lift Candidate Factory

## 7.1 目标

一次性并行生成多个 dataset-agnostic lift frame family，不再串行试一个修复。

## 7.2 Candidate families

### C11：Balanced Signed Pair + Whitening

目标：把 `signed_pair_cover` 的 E1 coverage 与 PCA-whiten 的 condition repair 结合。

构造：

```text
1. 用 train-stream covariance 估计 input whitening transform W_x。
2. 在 whitened coordinates 上构造 signed pair cover。
3. 对 pair channels 做 row/column norm equalization。
4. 输出 scale 使用 fanin-output-scale。
```

公式：

$$
\tilde{x}=xW_x,
$$

$$
h_{ij}^{+}=\frac{\tilde{x}_i+\tilde{x}_j}{\sqrt{2}},
\quad
h_{ij}^{-}=\frac{\tilde{x}_i-\tilde{x}_j}{\sqrt{2}}.
$$

### C12：SRHT / Hadamard Quadratic Frame

目标：用结构化随机正交投影降低 coherence，改善 rotated/random quadratic coverage。

构造：

$$
A = PHD,
$$

其中 $D$ 是随机 sign diagonal，$H$ 是 Hadamard / orthogonal transform，$P$ 是行采样。

要求：

```text
随机种子固定，不按 dataset name 分支；
可以用 train-stream covariance 做全局 scale，但不做 label-dependent tuning。
```

### C13：Block-Orthogonal Pair Cover

目标：在局部 block 内保证 pair coverage，同时用 block orthogonality 控制 condition。

构造：

```text
1. 将 input channels 按 fixed hash 分 block。
2. 每个 block 内做 signed pair cover。
3. block 间用 orthogonal mixing。
4. hidden budget 固定，不能增加参数量超过 same-param envelope。
```

### C14：Low-Coherence Random Feature Quadratic Frame

目标：用 random low-coherence vectors $a_k$ 覆盖 quadratic space。

判据：

$$
\mu(A)=\max_{i\ne j}|a_i^Ta_j|.
$$

要求：

$$
\mu(A)\leq\mu_{max},
$$

并且 pair coverage entropy 高。

### C15：Two-Subframe LQ

目标：避免单一 frame 同时承担 identity、pairwise、composition。

结构：

```text
subframe 1: identity / low-condition whiten frame
subframe 2: signed / SRHT interaction frame
shared quadratic edge basis
single PureKAN output mixing
```

注意：这不是 ordinary residual 或 MLP path。两个 subframe 都属于 edge-owned KAN lift。

### C16：Condition-Normalized Learnable Lift Init

目标：初始化后仍允许 lift 训练，但对训练前 condition 做 normalization。

只允许 initialization / parameterization，不允许增加 loss regularizer。

参数化：

$$
A = \operatorname{normalize}(\hat{A}),
$$

其中 normalize 是 forward parameterization 或 init-time transform，不是 loss penalty。

### C17：Depth-2 Compositional LQ Diagnostic

只作为 fallback diagnostic，不作为第一批 official base。

目标：如果 all single-frame LQ fail composition，但 pairwise coverage pass，则测试 depth-2 composition 是否是必要结构。

## 7.3 必须记录字段

```text
candidate_id
family
hidden_dim
param_count
param_delta_vs_mlp
uses_dataset_name_branch
uses_label_info
uses_teacher
uses_loss_modification
strict_purekan_contract
lift_init_type
whitening_used
whitening_source
random_seed_fixed
frame_rank
frame_effective_rank
frame_condition
frame_coherence
pair_coverage_entropy
basis_usage_entropy_init
output_scale_rule
```

## 7.4 Candidate promotion rule

一个 candidate 进入 Track D 必须满足：

```text
strict_purekan_contract = 1
uses_dataset_name_branch = 0
uses_label_info = 0
param_count within matched envelope
Track B expression triage pass
Track A clean timing exploratory pass
```

## 7.5 不满足条件时 Codex 自动尝试

如果 candidate param_count 超出：

```text
降低 frame budget；
用 shared pair channels；
用 low-rank block cover；
记录 param-fair 和 time-fair 两种版本，但 official 只保留 fair 版本。
```

如果 candidate 使用 train-stream covariance 但不同 seed 方差大：

```text
增加 covariance shrinkage；
使用 fixed random projection + shrinkage whitening；
记录 covariance condition 与 frame condition。
```

如果 candidate expression 好但 task 初始 loss 极差：

```text
检查 output_scale_rule；
检查 logit norm / CEp99 / margin_p10；
自动尝试 global output scale，不得按 dataset 单独调。
```

---

# 8. Track D：5-epoch Base Gate Triage

## 8.1 目标

快速淘汰明显不可能的 candidate，并选出 2-3 个进入长程 confirmation。

## 8.2 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train/val/test = 1024/512/512
epochs = 5
batch_size = 128
eval_batch_size = 512
optimizer = AdamW same schedule as baseline
functional_update = off
loss = CE only
```

## 8.3 对照方法

```text
B0-MLP-same-param-AdamW
B1-MLP-same-time-or-FLOPs-AdamW
B2-QuadraticFeatureMLP-diagnostic
C0-R2-LQ-current
C8-signed-pair-cover-lift
C10-data-pca-whiten-h64 diagnostic
C11-C17 new candidates
```

## 8.4 必须记录指标

### Task

```text
dataset
seed
method
train_loss
val_loss
val_acc
test_acc
val_loss_auc_step
val_loss_auc_time
ECE
NLL
CEp99
margin_p10
wrong_confidence_p95
classwise_acc
```

### System

```text
clean_forward_q90
clean_backward_q90
clean_step_q90
compiled_step_q90
memory_compact
memory_conservative
architecture_step_ratio
online_metric_step_ratio
offline_diagnostic_time
```

### Expression / frame

```text
E1_B2_R2
E1_B3_LS_R2
E2_B1_delta_vs_MLP
E6_B3_LS_R2
E8_B3_LS_R2
global_matrix_span_R2
frame_condition
frame_effective_rank
frame_coherence
pair_coverage_entropy
```

### Geometry

```text
lift_condition_proxy
basis_usage_entropy
dead_basis_fraction
effective_rank_hidden
curvature_debt
perturb_logit_drift_p95
local_jacobian_norm_p95
signal_consistency_real
noise_leak
```

## 8.5 Base triage gate

Task near gate：

$$
\Delta Acc_{mean}\geq -0.005.
$$

Worst-row safety：

$$
\min_{dataset,seed}\Delta Acc\geq -0.025.
$$

Expression gate：

$$
\Delta R^2_{E2}\geq-0.02,
$$

$$
\min(R^2_{LS,E1},R^2_{LS,E6},R^2_{LS,E8})\geq0.70.
$$

Condition gate exploratory：

$$
\kappa_{lift,mean}\leq0.1\cdot\kappa_{C0}.
$$

Strong condition gate：

$$
\kappa_{lift,mean}\leq0.01\cdot\kappa_{C0}.
$$

System gate：

$$
StepRatio_{clean,q90}\leq1.25.
$$

Promotion to Track F requires at least:

```text
Task near gate pass
Expression gate pass
System gate pass
Condition exploratory pass
No fake/proxy/cpu
No dataset-specific branch
```

## 8.6 可视化

```text
v123_base_task_delta_matrix.svg
v123_task_expression_condition_pareto.svg
v123_lift_condition_by_dataset_seed.svg
v123_expression_gate_heatmap.svg
v123_clean_timing_gate.svg
v123_ce_tail_margin_panel.svg
v123_candidate_failure_heatmap.svg
```

## 8.7 不满足条件时 Codex 自动尝试

如果 task fail 但 expression/condition pass：

```text
1. 检查 logit scale / CEp99 / margin_p10。
2. 尝试 global output scale correction：fanin、fanin-output、unit-variance-logit。
3. 尝试 global warmup，不按 dataset 分支。
4. 若仍 fail，标记 trainability failure，不进入 functional。
```

如果 expression fail 但 task pass：

```text
1. 不立即淘汰，但标记 weak expression。
2. 补跑 rotated/random quadratic LS。
3. 若 global coverage 仍 fail，不进入 final base claim。
```

如果 condition fail 但 task/expression pass：

```text
1. 自动生成 normalized variant。
2. 对 lift columns 做 norm equalization / QR / shrinkage whitening。
3. 若 condition 仍 > current 0.1x，不进入 functional，因为 geometry maintenance 会建立在病态坐标上。
```

如果 system fail：

```text
1. 分离 clean vs hook path。
2. compiled-warm 重测。
3. 若 clean still > 1.25，candidate stop。
```

---

# 9. Track E：Passive Geometry / Signal / Tail Battery

## 9.1 目标

对 Track D 的 top candidates 做被动几何测量，不改变训练，不加入 functional update。目标是回答：

```text
balanced frame 是否真的改善 geometry，而不是只改善 expression 表格？
```

## 9.2 测量频率

```text
checkpoints: init, epoch1, epoch3, epoch5, final
splits: train_probe, val, noise_probe, perturb_probe
```

## 9.3 指标

### Manifold / cover stability

```text
lift_condition_proxy
basis_usage_entropy
dead_basis_fraction
basis_occupancy_entropy
frame_effective_rank
hidden_effective_rank
perturb_logit_drift_p95
local_jacobian_norm_p95
curvature_debt
augmentation_logit_drift
```

### Signal / noise separation

```text
signal_consistency_real
signal_consistency_noise
real_noise_consistency_gap
noise_leak
SNR_active_fraction
population_risk_offdiag_proxy
```

### Tail / calibration

```text
CEp99
margin_p10
wrong_confidence_p95
ECE
NLL
Brier
classwise_tail_fail_rate
```

## 9.4 判断标准

Geometry candidate 不要求所有指标都优，但必须满足：

```text
1. task gate 不坏；
2. lift condition 相比 C0 至少 10x 改善；
3. perturb drift / CEp99 / ECE 至少两个不坏；
4. basis / frame effective rank 不塌；
5. noise_leak 不上升。
```

形式化：

$$
TaskSafe=1,
$$

$$
\kappa_{lift,new}\leq0.1\kappa_{lift,C0},
$$

$$
\sum_{m\in\{PerturbDrift,CEp99,ECE,NoiseLeak\}}\mathbf{1}[m_{new}\leq m_{C0}+\epsilon_m]\geq3.
$$

## 9.5 可视化

```text
v123_geometry_trajectory_panel.svg
v123_signal_noise_gap_by_method.svg
v123_tail_calibration_panel.svg
v123_rank_condition_trajectory.svg
v123_geometry_task_pareto.svg
v123_dataset_slice_diagnostic.svg
```

## 9.6 不满足条件时 Codex 自动尝试

如果 condition 改善但 perturb drift 变坏：

```text
尝试 smoother output scale / smaller initial quadratic scale。
不加入 geometry loss。
```

如果 rank 塌：

```text
尝试 frame diversity increase、orthogonal subframe、basis entropy preserving init。
```

如果 noise_leak 上升：

```text
标记 signal/reservoir risk，不能进入 functional。
```

---

# 10. Track F：20/30-epoch Survivor Confirmation

## 10.1 打开条件

只有 Track D/E 都过的 candidate 才进入。

## 10.2 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2 initially
budget = 20 or 30 epochs equivalent
train/val/test >= previous official setting
functional_update = off
loss = CE only
```

## 10.3 对照

```text
MLP-same-param-AdamW
MLP-same-time/FLOPs-AdamW
QuadraticFeatureMLP-diagnostic
C0-R2-LQ-current
best v12.3 candidate
```

## 10.4 成功标准

Base qualified near-pass：

$$
NearPassRate\geq0.80,
$$

$$
\Delta Acc_{macro}\geq-0.005.
$$

Base strong pass：

$$
\Delta Acc_{macro}\geq0.
$$

Convergence gate：

$$
ValLossAUC_{time,KAN}\leq ValLossAUC_{time,MLP}+\epsilon.
$$

System gate：

$$
StepRatio_{clean,q90}\leq1.25,
$$

$$
MemoryRatio\leq1.05.
$$

Geometry gate：

```text
condition improves vs C0;
ECE/NLL not worse;
CEp99/margin_p10 not worse;
rank/basis not collapsed.
```

If candidate passes Track F, only then route becomes:

```text
R1-BaseQualifiedFunctionalCanReenter
```

## 10.5 可视化

```text
v123_confirm_loss_vs_time.svg
v123_confirm_acc_by_seed_dataset.svg
v123_confirm_paired_delta_ci.svg
v123_confirm_ece_nll_tail.svg
v123_confirm_system_pareto.svg
v123_confirm_geometry_certificate.svg
```

---

# 11. Track G：Functional Complementarity Re-entry Gate

## 11.1 打开条件

Functional re-entry 只在 base qualified 后打开：

```text
base_qualified = true
Track F near-pass or strong-pass = true
system gate pass = true
expression gate pass = true
geometry hard gate pass = true
```

## 11.2 功能定位

Functional update 只做：

```text
task-safe geometry maintenance
low-frequency correction
control-resistant complementarity
```

不做：

```text
主 optimizer；
future selector；
dataset-specific controller；
loss modification；
teacher/distill；
```

## 11.3 对照矩阵

```text
G0 TaskOnlyAdamW
G1 NoOpMatchedOverhead
G2 RandomMatchedNorm
G3 AdamWParallelDirection
G4 ShuffledPayload
G5 SNROnlyGate
G6 GeometryOnlyNoSNR
G7 SNRProjectedGeometry
G8 BasisEntropyRebalance
G9 LiftConditionRepair
G10 TailStabilityCorrection
G11 ControlResidualGeometry
G12 MLPAnalogGeometryMaintenance
G13 QuadraticFeatureMLPAnalogMaintenance
```

## 11.4 必须记录指标

```text
candidate
dataset
seed
step/event_id
base_candidate_id
functional_candidate_id
train_descent
holdout_descent
holdout_descent_ratio
bad_step_rate
cos_with_task_step
cos_with_adamwparallel
functional_norm_ratio
lambda_selected
lambda_zero_rate
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
lift_condition_delta
basis_entropy_delta
rank_delta
control_gap_vs_best_control
beats_strong_controls
event_overhead_ms
amortized_overhead_ratio
```

## 11.5 通过标准

Task safety：

$$
HoldoutDescentRatio\geq0.95,
$$

$$
BadStepRate\leq0.02.
$$

Control resistance：

$$
Gain_{functional} > Gain_{AdamWParallel},
$$

$$
Gain_{functional} > Gain_{RandomMatchedNorm},
$$

$$
Gain_{functional} > Gain_{SNROnly}.
$$

Geometry improvement：

至少一个核心 geometry 指标改善，且无 hard harm：

$$
\Delta GeoDebt\leq-0.10|GeoDebt_{base}|,
$$

or:

$$
\Delta ECE<0,
$$

or:

$$
\Delta CEp99<0.
$$

System overhead：

$$
Overhead_{amortized}\leq0.05.
$$

## 11.6 不满足条件时 Codex 自动尝试

如果 functional 不打 controls：

```text
不要调低 gate；
不要打开 short-run；
做 control-gap audit：找出它和 AdamWParallel/Random/SNR-only 的方向差异；
只保留能改善 controls 不能改善的 geometry component。
```

如果 lambda_zero_rate 高：

```text
说明 correction 与 task descent 冲突；
尝试 projection onto task-safe null/tangent subspace；
仍高则停止该 functional family。
```

如果 task safe 但 geometry 不改：

```text
说明 correction 是无效 perturbation；
停止，不进入 training。
```

如果 geometry 改但 task 伤：

```text
说明 smoothing/repair 牺牲表达；
不允许进入 official。
```

---

# 12. 失败路由与决策树

## R1：BaseQualifiedFunctionalCanReenter

条件：

```text
Track F base near-pass / strong pass
system clean timing pass
expression global coverage pass
condition pass
geometry hard gate pass
```

动作：打开 Track G functional complementarity。

## R2：CoverageConditionConflict

表现：

```text
coverage pass but condition fail
or condition pass but coverage fail
```

动作：继续 frame design，但不进入 task/functional。

## R3：CoveragePassTaskFail

表现：

```text
expression pass
condition pass
vision task delta < gate
```

动作：检查 scale / optimization / CE tail；若全局修复无效，说明 frame not trainable。

## R4：TimingAccountingFailOnly

表现：

```text
clean timing pass
official diagnostic path fail
```

动作：修 accounting，不判 base fail。

## R5：FunctionalStillControlEquivalent

表现：

```text
base pass 后 functional diagnostic 仍不打 controls
```

动作：停止 functional family，回到 target/geometry mechanism，不开 short-run。

## R6：LQFrameFamilyExhausted

表现：

```text
C11-C17 全部不能同时满足 coverage + condition + task
```

动作：进入 primitive-level pivot：depth-2 compositional LQ / strict FullEdge compositional / alternative PureKAN primitive。

---

# 13. 最小可执行并行实验包给 Codex

## Batch 1：cheap frame audit

命令目标：快速生成 C11-C17，并跑 Track A/B，不跑 vision full train。

必须输出：

```text
v123_frame_manifest.csv
v123_timing_accounting.csv
v123_expression_capacity.csv
v123_matrix_span_audit.csv
v123_frame_failure_table.csv
```

Batch 1 promote top 6 candidates 到 Batch 2。

## Batch 2：5-epoch triage

设置：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
epochs = 5
functional_update = off
```

必须输出：

```text
v123_base_triage.csv
v123_base_task_trace.csv
v123_base_geometry_snapshot.csv
v123_system_profile.csv
v123_triage_scorecard.csv
```

Batch 2 promote top 2-3 candidates。

## Batch 3：passive geometry trajectory

只跑 top candidates，不开 functional。

必须输出：

```text
v123_geometry_trajectory.csv
v123_signal_noise_battery.csv
v123_tail_calibration.csv
v123_geometry_certificate.csv
```

Batch 3 选择最多 2 个 candidate 到 Batch 4。

## Batch 4：20/30-epoch base confirm

必须输出：

```text
v123_base_confirm.csv
v123_confirm_trace.csv
v123_paired_delta_ci.csv
v123_final_base_route.json
```

若无 base survivor，停止 functional。

## Batch 5：functional re-entry diagnostic

仅在 Batch 4 route = R1 时运行。

必须输出：

```text
v123_functional_direction.csv
v123_one_step_probe.csv
v123_five_step_probe.csv
v123_control_matrix.csv
v123_lambda_backtracking.csv
v123_functional_route.json
```

---

# 14. 本轮最重要的判断标准

v12.3 不是必须立刻产生 final success，但必须回答下面四个问题：

```text
1. 是否存在一个 frame 同时满足 global quadratic coverage 与 low condition？
2. 如果存在，它是否保持 vision task stability？
3. 如果 task stability 失败，是 scale/optimization 问题，还是 LQ frame family 结构性不足？
4. 如果 base qualified，functional update 是否能打过 strong controls？
```

如果前三个问题没有正答案，functional 继续关闭。

---

# 15. 当前结论压缩版

当前最准确的判断是：

$$
\boxed{
\text{v12.2 不是没进展，而是把 blocker 精确定位到 LQ lift frame 的 coverage-condition-task 三目标冲突。}
}
$$

下一步不应该继续调 functional update，也不应该围绕某个数据集调参，而应该并行设计与验证 balanced interaction-cover lift。只有当 base 同时满足：

```text
global quadratic coverage；
low lift condition；
same-param task stability；
clean MLP-like timing；
```

functional geometry maintenance 才有资格重新进入 official audit。
