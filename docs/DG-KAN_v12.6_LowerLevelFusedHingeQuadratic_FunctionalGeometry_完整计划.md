# DG-KAN v12.6：Lower-Level Fused Simple Hinge/Quadratic Kernel + Functional Geometry 双线计划

> 版本：v12.6 execution draft
> 执行状态：已继续推进到 lower-level FHQ classgain / h160 / quad030 系列 B106-B111，B112/B115-B118 `normabs`，B119-B121 `meanstat`，B122 `groupabs4` + gradopt，并追加完成 A5 protocol repair、B124/B125 `fixedbranch+fixedgain`、B126 `logitnorm150`、B127 `absquad`、B128 `absmixsq`、B129 `sqdiag025`、B130 `cubicdiag025`、B131 `pairtraj`、B133/B135 `localdensepairtraj`。当前 best 仍为 B109 protocolfix final075；B131 的 F4 fixed-P workspace 实质降低 full-step/backward/update cost 但 A4 失败，B133/B135 A1/A4 打开但 task 坍塌。base 未合格，Functional official 关闭。真实结果与 hash/provenance 审计见 `DG-KAN_v12.6_LowerLevelFusedHingeQuadratic_FunctionalGeometry_结果复盘.md`。
> 目标：基于 v12.5.2 真实结果，停止 temperature / epoch / optimizer wrapper 等小修，转入真正 lower-level fused simple hinge/quadratic forward+backward kernel，同时并行保持 functional update 与流形-信号通道几何诊断。
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。
> 原则：不降低 gate，不按数据集调参，不使用 teacher / distillation / loss modification / sampler / class weight，不把 diagnostic 写成 official success。

---

## 0. 一句话结论

v12.5.2 不是失败到没有方向，也不是已经成功。它最真实的结果是：

$$
\boxed{
\text{B48 SimpleFastTaskGeometry 第一次同时打开了 efficiency 与 expression，}
\text{但没有打开 A5 task/base。}
}
$$

这说明我们已经越过了 v12.4 的一个关键坎：不是所有 strict PureKAN primitive 都必然慢，也不是所有低成本 primitive 都必然表达力不够。B48a 证明了一个低成本 hinge/direct/quadratic task-geometry primitive 可以做到：

```text
A1/A3 efficiency pass；
A4 expression pass；
accuracy / ECE 接近或部分通过；
```

但它也暴露了当前 hard blocker：

```text
AUC-time 失败；
official full-step 稳定性仍弱；
task trajectory 不够健康；
functional official route 仍关闭。
```

所以 v12.6 不再做：

```text
temperature 小网格；
epoch 数小网格；
optimizer impl wrapper 小修；
directskip scale 小修；
继续增加 quadratic branch 后再看能不能侥幸过。
```

v12.6 要做的是：

$$
\boxed{
\text{真正 lower-level fused simple hinge/quadratic forward+backward kernel，}
\text{并重新设计更低成本且不伤 A4 的 task-geometry primitive。}
}
$$

同时 functional update 不能停，但它必须保持：

```text
base 未合格前：diagnostic；
base 合格后：official re-entry；
任何时候都必须击败 strong controls。
```

## 0.1 当前执行更新

本计划已经被执行到 B135；最新真实状态不是完成态：

```text
route = R2-AUCStepAndTaskFail
latest_targeted_route = R2-AUCStepAndTaskFail
base_qualified = False
functional_open = False
best_current_candidate = B109b h160 absdiag050 classbranch classgain identitytailquad030 hingeamp025 protocolfix final075
latest_targeted_candidate = B135b h160 localdensepairtraj absdiag050 classbranch classgain fixedbranch fixedgain identitytailquad020 hingeamp025
```

已新增并验证的方向：

```text
B119 normabs050 + meanstat050:
  F3 full-step pass, A4 pass;
  task mean = 0.0008680555555555, worst = -0.009765625, near = 0.8888888888888888;
  AUC-step/time fail, A5 fail.

B120/B121 meanstat-only:
  full-step official fail, A4/A5 合法关闭。

B122 groupabs4:
  forward/grad correctness pass, gradopt 仍未过 full-step official gate;
  A4/A5 合法关闭。

A5 protocol repair:
  FHQ task compile warmup 与 AUC measurement 分离;
  AUC 同时记录 raw / steady;
  A5 同时要求 AUC-step 与 AUC-time，不再允许 timing-only pass。

B109 protocolfix final075:
  F3 full-step pass, A4 pass;
  task mean = 0.016927083333333332, worst = 0.00390625, near = 1.0, ECE ok;
  Fashion seed1/seed2 strict steady AUC-step fail, A5 fail.

B124/B125 fixedbranch+fixedgain:
  A4 pass, no fake/proxy/cpu pass;
  fixed branch/gain 与低 gain 不能修 Fashion late NLL/AUC。

B126 logitnorm150:
  FHQ analytic gradient correctness pass, F2 full-step pass;
  CEp99 降低但 ECE/AUC 明显失败。

B127 absquad050025:
  FHQ correctness/smoke pass;
  F3 step = 1.980083680412736, fixed-P F1 step = 1.5321638803678295;
  full-step official fail, A4/A5 合法关闭。

B128 absmixsq025:
  FHQ correctness/smoke pass;
  F3 step = 1.3378124510102267, fixed-P F1 step = 1.762266117059046;
  full-step official fail, A4/A5 合法关闭。

B129 sqdiag025:
  F3 full-step pass, A4 pass;
  task mean = 0.008680555555555556, worst = -0.0078125, near = 0.7777777777777778, ECE ok;
  Fashion seed0/1/2 AUC-step = 1.1682432188964107 / 1.216143071215535 / 1.1999663195537906;
  A5 fail, 不优于 B109。

B130 cubicdiag025:
  FHQ correctness/smoke pass;
  F3 step = 1.2481101999833282, fixed-P F1 step = 1.1436047527375663;
  full-step official fail, A4/A5 合法关闭。

B131 pairtraj:
  F4 fixed-P workspace step = 0.685358064760707, memory = 0.13;
  backward ratio = 0.4557250926585637, update ratio = 0.7320725649235357;
  真实降低 fused backward/update cost，但 A4 expression fail，A5 合法关闭。

B133 localdensepairtraj:
  F4 fixed-P workspace step = 0.7942844867303934, memory = 0.13;
  B133b/B133a A4 pass;
  B133b task mean = -0.10026041666666667, worst = -0.333984375, near = 0.3333333333333333, ECE fail;
  A5 fail。

B135 localdensepairtraj + fixedbranch/fixedgain + quad020:
  F4 fixed-P workspace step = 0.9283274262455713, memory = 0.12904761904761905;
  A4 pass;
  B135b task mean = -0.08550347222222222, worst = -0.26171875, near = 0.3333333333333333, ECE fail;
  A5 fail，说明 B124 稳定 branch/gain 也救不回 dense pair trajectory。
```

因此当前执行结论是：

```text
1. lower-level FHQ 主线有真实进展，但 v12.6 仍未完成 base。
2. B109 protocolfix final075 仍是 best：accuracy/worst/near/ECE 均过，只剩 strict steady AUC-step/time hard gate。
3. normabs、meanstat、groupabs4、fixed gain、logitnorm、direct-tail abs/square、mixed-tail、sqdiag energy-tail、signed cubic tail 这类 scalar / pooled / calibration / simple tail micro-primitive 已不值得继续小网格。
4. B131 证明 F4 fixed-P workspace 是真实 cost-reduction 路径；B133/B135 证明 dense random pair trajectory 会伤 task，不能继续小修。
5. 下一步应保留 F4 fixed-P workspace 的 cost reduction，但回到 B109/B124 这类 task-stable trajectory，或设计更稀疏/局部且 task-stable 的 single-kernel-friendly primitive。
6. Functional official 仍关闭；base 未过 A5 前，Functional/Line C 只能 diagnostic。
```

本计划下面的设计目标仍保留为 v12.6 的历史执行依据；最新落盘结果以复盘文件为准。

---

## 1. 对 v12.5.2 结果的独立分析

## 1.1 有进展吗？

有，而且是比 v12.4 更实质的进展。

v12.4 的 B47 证明 manual backward / manual CE step 可行，但 forward ratio 仍约 $6.8x$，step ratio 约 $2.62x$，说明 Python-level manual backward 已经不能解决问题。v12.5.2 的 B48a 则把系统推进到一个新状态：

```text
B48a compiled full-step step_ratio_q90 = 0.8748996687637653
B48a memory_ratio_q90 = 0.4033043955666884
B48a A4 expression pass
```

这不是小进展。它说明低成本 task-geometry primitive 的方向比继续在 GatedLegendre+Quadratic 大模型上补手写 backward 更接近 next-gen MLP envelope。

但是它不是 base success。B48a 5 epoch 时：

```text
mean delta = -0.00021701388888888888
worst = -0.021484375
near pass = 0.5555555555555556
ECE ok
AUC-time fail
```

B48a epochs8 是当前最好 task 结果：

```text
mean delta = 0.003472222222222222
worst = -0.015625
near = 0.8888888888888888
ECE ok
AUC-time fail
```

独立判断：

$$
\boxed{
\text{B48a 已经接近 base，但 AUC-time 失败说明它还不是合格 base。}
}
$$

这里不能只看 final accuracy。你的目标不是“最终多训几个 epoch 接近 MLP”，而是：

```text
表达力不打折；
前馈 / 反传 / step 代价接近 MLP；
收敛不比 MLP 慢；
几何更健康。
```

B48a 的 final accuracy 接近只是必要条件，不是充分条件。

---

## 1.2 为什么感觉进度还是慢？

因为此前进展多是 **排错型进展**，而 v12.5.2 才第一次出现 **接近 base 的能力型进展**。但它仍然没有跨过 official base gate，所以体感上仍然像“没成功”。

从历史看，路线经历了三次层级提升：

```text
v12.1-v12.3:
  LQ / quadratic frame family 失败，问题是 coverage-condition-task 无法同时成立。

v12.4:
  多基函数扫描收敛到 Gated Legendre + Quadratic；
  B47 证明 manual backward 可行，但 forward/kernel launch 是主 blocker。

v12.5.2:
  B48 SimpleFastTaskGeometry 打开 efficiency + expression；
  但 task AUC-time 仍失败。
```

这说明慢的根本不是 Codex 没跑实验，而是我们之前一直在找：

$$
\text{某个 basis family 是否能在 Torch/Python 层自然接近 MLP。}
$$

现在数据已经说明：

$$
\boxed{
\text{不能继续依赖 Torch/Python 层自然组合；}
\text{必须把有希望的简单 primitive 写成 fused layer。}
}
$$

---

## 1.3 当前真正卡在哪里？

我认为当前 blocker 不是单点问题，而是下面这个三目标闭环仍未成立：

$$
\boxed{
\text{low-cost primitive} + \text{task trajectory} + \text{A4 expression preservation}
}
$$

具体说：

### 1.3.1 B48a 说明低成本可行

B48a 的 step / memory 非常强，说明 simple hinge/direct/quadratic 结构可以落进 MLP-like 系统 envelope。它不是 dense basis、不是 huge Legendre recurrence、不是 full LQ frame，系统复杂度明显下降。

### 1.3.2 B48b / B49a 说明“加表达力”会立刻撞效率

B48b h192/quad050 与 B49a sqdiag direct channel 都停在 R1 efficiency。也就是说，当前不能通过“再加一点 quadratic / diagonal channel / hidden capacity”来解决 task trajectory，因为它会立即破坏效率门。

这暴露了一个核心事实：

$$
\boxed{
\text{task geometry 必须更聪明，而不是更大。}
}
$$

### 1.3.3 B50b 说明“减成本”会伤 A4

B50b two-hinge h96 非常快，compiled step ratio 到 $0.7423$，但 A4 expression 失败。B50c/B50d 加容量后又回到 R1。这个结果非常关键，因为它说明：

```text
只追求低成本会把表达力打折；
只补表达力会把效率打爆。
```

下一步必须围绕“不伤 A4 的低成本 primitive”设计，而不是继续把 B48 做温度 / LR 小修。

### 1.3.4 AUC-time 是硬问题，不是报告噪声

B48a epochs8 的 accuracy / ECE 已经过或近门，但 AUC-time 失败。这意味着当前模型不是完全学不会，而是训练轨迹不够好：

```text
早期 loss 下降慢；
某些 seed / dataset slice 的 learning path 延迟；
官方 full-step timing 可能在训练态仍有抖动；
logit / margin trajectory 不够稳定；
feature drift 可能没有形成足够好的 train-probe coupling。
```

所以 AUC-time 必须拆开看：

$$
AUC_{time}
=
\sum_t L_{val}(t)\Delta T_t.
$$

它同时受两个因素影响：

```text
loss trajectory；
step-time trajectory。
```

如果：

```text
AUC-step 过，但 AUC-time 不过：
  主要是 system/kernel timing 问题。

AUC-step 也不过：
  主要是 primitive / coordinate / task geometry 问题。
```

v12.6 必须把这两者分开落盘，不能只看一个 aggregate。

---

## 1.4 Functional update 有进展吗？

没有 official 进展，但它不能停。

当前状态是：

```text
functional_open = false
Line C = diagnostic only
```

这是正确的。因为 base 还没过 A5，functional 不能 official。过去多次经验已经说明：在 base 未合格时打开 functional，会把问题混成：

```text
base 不稳；
functional direction 不稳；
control gap 不稳；
timing 不稳。
```

但 functional diagnostic 仍然有价值。它应该在 B48a / B50 / fused kernel candidate 的 cloned setting 中并行测：

```text
train-probe coupling；
signal channel / reservoir；
NoiseSignalLeak；
RealSignalReservoirRatio；
functional-control gap。
```

Functional update 现在的角色不是“救一个不合格 base”，而是：

$$
\boxed{
\text{在合格或近合格 base 上，证明它能独立改善 signal-channel geometry。}
}
$$

---

## 1.5 是否在正确道路上？

方向是对的，但推进层级必须升级。

正确的地方：

```text
1. 不再继续 LQ frame 小修；
2. 没有把 B48a near result 写成 base success；
3. 没有在 base 未过时打开 functional official；
4. 没有用 dataset-specific tuning；
5. 已经把 candidate 收敛到简单 task-geometry primitive；
6. 已经发现 AUC-time 是 hard blocker；
7. 已经确认 lower-level fused kernel 是必要路线。
```

需要停止的地方：

```text
1. 不再做 temperature 小网格；
2. 不再做 epoch 数小网格；
3. 不再做 optimizer impl wrapper 小修；
4. 不再继续直接增加 quadratic branch；
5. 不再为了过某个 dataset slice 调参数；
6. 不再把 functional positive diagnostic 包装成 success。
```

v12.6 的正确主线是：

$$
\boxed{
\text{Fused Kernel First + Low-Cost Task-Geometry Redesign + Functional Diagnostic Parallel。}
}
$$

---

## 1.6 离目标还差多远？

分模块判断：

| 模块 | 当前状态 | 距离目标 |
|---|---|---|
| Efficiency micro / steady | B48a 已很强，部分 run step ratio < 1 | 近 |
| Official training efficiency | 多个 targeted run 又回 R1，仍不稳 | 中等 |
| A4 expression | B48a/B48d/B48e 已过 | 近 |
| A5 accuracy/ECE | B48a epochs8 已近门或部分过 | 近 |
| AUC-time | 当前 hard blocker | 中等偏远 |
| Low-cost primitive design | B50b 快但伤 A4，B48b/B49 加表达伤效率 | 中等偏远 |
| Functional official | base 未过，仍关闭 | 远 |
| Next-gen MLP claim | 还不能 claim | 远 |

一句话：

$$
\boxed{
\text{我们离第一个合格 FC-PureKAN base 比 v12.4 更近，}
\text{但离 functional update 成功和 next-gen MLP claim 仍远。}
}
$$

---

# 2. v12.6 总目标

v12.6 不以“多跑几个 epoch 看能不能过”为目标。它的目标是打穿两个硬闭环：

## 2.1 Base 闭环

构建至少一个 strict FC-PureKAN simple task-geometry primitive，使其同时满足：

```text
A1/A3:
  forward / backward / step / memory 接近 MLP；

A4:
  expression 不打折；

A5:
  task accuracy / ECE / AUC-time 合格；

Line C:
  train-probe coupling / signal-reservoir / noise leakage 不显著坏于 MLP。
```

正式写成：

$$
BaseQualified =
A1 \land A4 \land A5 \land C_{nontearing}.
$$

## 2.2 Functional 闭环

在 base qualified 或 near-qualified cloned setting 中，验证 functional update 是否能独立改善 signal-channel geometry：

$$
FunctionalAdvantage =
TaskSafe
\land
GeometryImprove
\land
ControlResistant
\land
CostBounded.
$$

其中：

$$
TaskSafe:
Acc_{func}\ge Acc_{base}-0.003.
$$

$$
CostBounded:
T_{amortized,func}/T_{base}\le 1.05.
$$

$$
ControlResistant:
G_{func} > G_{AdamWParallel},\quad
G_{func} > G_{RandomMatched},\quad
G_{func} > G_{SNRonly}.
$$

---

# 3. v12.6 核心假设

## H1：B48a 的主要系统问题来自 unfused operation graph，而不是 primitive 理论复杂度

B48a steady run 可以过 efficiency，但多个 targeted run 回到 R1。这提示：

```text
当前实现存在 compile/warmup/timing instability；
训练态可能仍有多 kernel launch / temporary materialization；
optimizer wrapper 不是根因；
需要 single-call fused forward+backward。
```

验证标准：

$$
T_{step,q90}^{fused}
\le
0.75T_{step,q90}^{B48a\_torch}
$$

或至少：

$$
T_{step,q90}^{fused}/T_{MLP,q90}\le 1.10.
$$

如果 fused 后仍然 AUC-time fail，但 AUC-step pass，则说明仍是 system timing 问题；如果 AUC-step 也 fail，则是 task trajectory 问题。

---

## H2：AUC-time 失败来自 early trajectory / coupling，而不是 final capacity

B48a epochs8 final accuracy / ECE 接近通过，但 AUC-time fail。假设：

```text
模型最终能学到；
但前期 signal channel 形成慢；
某些 hard slice 的 margin / CEp99 下降滞后；
train motion 到 probe motion 的 coupling 不稳定。
```

验证标准：

```text
若 AUC-step fail：
  primitive trajectory / coordinate geometry 是 blocker。

若 AUC-step pass 但 AUC-time fail：
  fused kernel / timing 是 blocker。
```

Line C 要记录：

$$
CouplingR^2,\quad
NoiseSignalLeak,\quad
RealSignalReservoirRatio,\quad
KernelDrift.
$$

---

## H3：低成本 task-geometry primitive 存在，但必须使用“结构性表达”而不是堆容量

B48b/B49a 说明直接加容量会失败在 R1；B50b 说明减容量会伤 A4。假设存在一个更结构化的 primitive：

```text
one-hinge / two-hinge with shared threshold；
minimal quadratic cross sketch；
fixed low-coherence projection；
direct identity edge path；
optional centered square channel；
all fused and no dense basis materialization。
```

它可以同时满足：

$$
A4_{new}\ge A4_{B48a}-\epsilon_{expr}
$$

且：

$$
T_{step,new}/T_{MLP}\le 1.10.
$$

---

## H4：Functional update 不应该救 base，而应该修 signal-channel geometry

Functional update 应并行测，但其 official success 必须等 base 合格。当前假设：

```text
functional update 的价值不是降低 train loss；
而是提高 train-probe coupling；
减少真实信号困在 reservoir；
减少噪声进入 signal channel；
不破坏 tail / cover / A4。
```

Functional gate：

$$
CouplingR^2_{func}\ge CouplingR^2_{base}+0.02.
$$

$$
NoiseSignalLeak_{func}\le NoiseSignalLeak_{base}-0.02.
$$

$$
RealSignalReservoirRatio_{func}\le RealSignalReservoirRatio_{base}-0.02.
$$

并且击败 strong controls。

---

# 4. v12.6 四线并行架构

v12.6 必须并行，不再串行等一个 run 完全失败才试下一个。

```text
Line A:
  lower-level fused simple hinge/quadratic kernel。

Line P:
  low-cost task-geometry primitive redesign，不伤 A4。

Line C:
  Manifold-Channel Geometry Diagnostics。

Line B:
  functional update diagnostic / official gated re-entry。
```

四线之间的关系：

```text
Line A 解决 B48a 的系统实现瓶颈；
Line P 解决 B48a / B50 的 task-geometry trade-off；
Line C 判断 base/functional 是否有好几何；
Line B 判断 functional update 是否有独立贡献。
```

---

# 5. Line A：Lower-Level Fused Simple Hinge/Quadratic Kernel

## A0. 目标

实现真正 lower-level fused simple hinge/quadratic forward+backward kernel，而不是继续依赖 PyTorch op 组合。

目标不是先创造新模型，而是先复现 B48a 的数学，同时消除：

```text
basis materialization；
多次 norm / hinge / quadratic temporary tensor；
Python/Torch op 边界；
kernel launch fragmentation；
optimizer wrapper 抖动；
training-state timing instability。
```

核心候选：

```text
FHQ0:
  B48a exact math fused implementation。

FHQ1:
  B48a exact math + fused CE backward + fused AdamW update。

FHQ2:
  B48a exact math + two-phase gradient reduction + fused update。

FHQ3:
  minimal primitive variant from Line P，如果 FHQ0 仍 AUC-step fail。
```

---

## A1. Simple hinge/quadratic primitive 的公式

以一个 batch $X\in\mathbb R^{B\times d}$ 为输入。固定无标签统计量 $\mu,s$：

$$
Z=\operatorname{clip}\left(\frac{X-\mu}{s+\epsilon}, -c, c\right).
$$

定义 direct edge channel：

$$
\Psi_0(Z)=Z.
$$

定义 hinge channel：

$$
\Psi_+(Z)=\operatorname{ReLU}(Z-\tau),
$$

$$
\Psi_-(Z)=\operatorname{ReLU}(-Z-\tau).
$$

定义 minimal quadratic sketch。给定 projection $P\in\mathbb R^{d\times r}$：

$$
U=ZP.
$$

$$
\Psi_q(Z)=U^2-\operatorname{stopgrad}\left(\mathbb E_B[U^2]\right).
$$

输出 logits：

$$
Y=
\Psi_0(Z)W_0
+
\Psi_+(Z)W_+
+
\Psi_-(Z)W_-
+
\Psi_q(Z)W_q
+
b_c.
$$

其中 $b_c$ 如果使用，必须作为 constant-basis edge coefficient，不作为 ordinary MLP head bias。所有 learnable tensor 必须被标记为 KAN edge / basis / lift 参数。

禁止：

```text
ordinary hidden Linear path；
learnable LayerNorm；
MLP stem/head；
dataset-name branch；
label-dependent normalization；
teacher/distillation。
```

---

## A2. Fused forward kernel

### 目标

一个 kernel 或极少数 kernels 完成：

```text
fixed norm；
hinge basis；
quadratic sketch；
readout accumulation；
logits output。
```

不允许 materialize：

```text
[B, d, K] dense basis tensor；
[B, d, number_of_basis] full temporary；
大量中间 U / hinge temp 被长期保存。
```

### 必须记录

`v126_fused_forward_profile.csv`：

```text
run_id
candidate_id
implementation_id
device
dtype
batch_size
input_dim
rank_q
hidden_dim_or_channels
kernel_count_forward
forward_time_ms_q50
forward_time_ms_q90
forward_ratio_vs_mlp_q90
temporary_allocated_mb
workspace_mb
basis_materialized
logits_max_abs_err_vs_reference
logits_relerr_vs_reference
official_forward_pass
```

### 正确性 gate

$$
\frac{\|Y_{fused}-Y_{ref}\|_\infty}{\|Y_{ref}\|_\infty+\epsilon}<10^{-5}.
$$

或者：

$$
\max |Y_{fused}-Y_{ref}|<10^{-5}.
$$

### 效率 gate

探索：

$$
T_{forward,fused}/T_{forward,MLP}\le 1.25.
$$

正式：

$$
T_{forward,fused}/T_{forward,MLP}\le 1.10.
$$

---

## A3. Fused backward kernel

### 目标

不再只做 manual backward Python path，而是 fused CE backward + coefficient gradient reduction。

对 CE loss，先计算：

$$
\Delta
=
\operatorname{softmax}(Y)-\operatorname{onehot}(y).
$$

梯度：

$$
\nabla W_0=\Psi_0(Z)^\top\Delta.
$$

$$
\nabla W_+=\Psi_+(Z)^\top\Delta.
$$

$$
\nabla W_-=\Psi_-(Z)^\top\Delta.
$$

$$
\nabla W_q=\Psi_q(Z)^\top\Delta.
$$

如果 $P$ 是 learnable，则：

$$
\nabla P
=
Z^\top
\left(
2U\odot(\Delta W_q^\top)
\right).
$$

第一阶段可固定 $P$，只更新 $W_q$，以降低 backward 成本。若固定 $P$ 伤 A4，再打开 learnable $P$。

### 必须记录

`v126_fused_backward_correctness.csv`：

```text
run_id
candidate_id
implementation_id
grad_role
grad_relerr
grad_cos
grad_max_abs_err
finite_grad_rate
uses_autograd_graph
uses_loss_backward
backward_kernel_count
backward_time_ms_q90
backward_ratio_vs_mlp_q90
memory_ratio_vs_mlp_q90
official_backward_pass
```

### 正确性 gate

$$
grad\_relerr < 10^{-4}
$$

或：

$$
grad\_cos > 0.999.
$$

### 效率 gate

探索：

$$
T_{backward,fused}/T_{backward,MLP}\le 1.40.
$$

正式：

$$
T_{backward,fused}/T_{backward,MLP}\le 1.20.
$$

显存：

$$
M_{step,fused}/M_{MLP}\le 0.80.
$$

---

## A4. Fused full-step kernel / optimizer update

### 目标

如果 backward 已经正确，测试：

```text
fused CE delta；
fused gradient reduction；
fused AdamW / SGD update；
workspace reuse；
CUDA graph fixed shape capture。
```

先实现三档：

```text
F0:
  fused forward only。

F1:
  fused forward + fused backward gradients，optimizer torch-side。

F2:
  fused forward + fused backward + fused AdamW update。

F3:
  F2 + CUDA graph capture / fixed workspace。
```

### 必须记录

`v126_fullstep_profile.csv`：

```text
candidate_id
implementation_id
optimizer_impl
forward_q90_ms
backward_q90_ms
update_q90_ms
step_q90_ms
forward_ratio_q90
backward_ratio_q90
update_ratio_q90
step_ratio_q90
memory_ratio_q90
kernel_count_total
workspace_reuse
cuda_graph_enabled
compile_warmup_excluded
official_efficiency_pass
```

### Gate

探索：

$$
T_{step}/T_{MLP}\le 1.25.
$$

正式：

$$
T_{step}/T_{MLP}\le 1.10.
$$

强目标：

$$
T_{step}/T_{MLP}\le 1.05.
$$

显存：

$$
M_{step}/M_{MLP}\le 0.80.
$$

---

## A5. 失败时 Codex 应该先尝试什么

如果 fused forward correctness 失败：

```text
1. 固定 dtype = fp32；
2. 关闭 centering / clipping，只比较 direct + hinge；
3. 分别打开 quadratic sketch；
4. 对每个 component 输出 partial logits；
5. 检查 row-major / col-major stride；
6. 检查 P / W 的 layout 与 reference 一致；
7. 禁止跳到 task 训练。
```

如果 fused forward 没有提速：

```text
1. 统计 kernel_count_forward 是否真的下降；
2. 检查是否仍 materialize hinge / q temp；
3. 用 tile-by-output 直接 accumulate logits；
4. 减少 global memory write；
5. 使用 persistent workspace；
6. 对 small batch 使用 block-level accumulation；
7. 若 improvement < 15%，该 implementation 降级，不做 task。
```

如果 backward correctness 失败：

```text
1. 先只测 W0 / W+ / W-；
2. 再测 Wq；
3. 若 P learnable，最后测 dP；
4. 对 CE delta 与 reference softmax delta 比较；
5. 检查 reduction 维度；
6. 检查 hinge mask derivative；
7. 检查 quadratic centering stopgrad。
```

如果 backward 慢：

```text
1. 固定 P，避免 dP；
2. 两阶段 reduction：block partial -> final reduce；
3. 合并 W0/W+/W- gradient；
4. bf16 forward + fp32 accumulate；
5. recompute basis，不保存 basis；
6. 不计算 input dx for first layer；
7. fuse update，避免 gradient tensor materialization。
```

如果 memory 高：

```text
1. reset peak after warmup；
2. 用 persistent workspace；
3. 减少 partial gradient buffer；
4. chunk output classes；
5. 避免保存 logits 以外的 activation；
6. 使用 bitpacked hinge mask 或 recompute mask；
7. 如果仍 > 0.8 MLP，记录 memory waterfall，不进入 A5。
```

---

# 6. Line P：低成本且不伤 A4 的 Task-Geometry Primitive Redesign

## P0. 目标

B48a 是当前 anchor，但不是唯一候选。Line P 的目标是设计更低成本、更稳定的 task-geometry primitive，但它必须保护 A4 expression。

不允许：

```text
通过增大 hidden / rank 粗暴补 task；
通过降低 A4 gate 换效率；
通过 dataset-specific branch 修某个数据集；
通过温度/epoch/optimizer 小网格冒充机制。
```

允许：

```text
结构性增加 interaction coverage；
改 fixed projection frame；
改 hinge basis 形状；
改 direct/quadratic 的信息流；
减少 redundancy；
保持 fused-kernel-friendly。
```

---

## P1. 候选族

### P1.1 `FHQ-B48a-exact`

目的：作为 anchor，先证明 fused implementation 不改变 B48a 数学。

```text
same formula as B48a；
same h/rank/temperature；
same A4 gate；
same A5 protocol。
```

判断：

```text
若 exact fused B48a AUC-time 过：
  当前主要是 implementation/timing blocker。

若 exact fused B48a AUC-time 仍不过：
  当前主要是 primitive trajectory blocker。
```

### P1.2 `FHQ-1HingeQ`

目的：比 B48a 更低成本，但保留 direct + one-sided hinge + quadratic sketch。

```text
direct channel；
positive hinge；
minimal q-rank；
fixed P；
centered q；
fused required。
```

风险：可能像 B50b 一样伤 A4。

通过条件：

$$
A4_{FHQ1}\ge A4_{B48a}-0.005.
$$

### P1.3 `FHQ-2HingeSharedQ`

目的：修复 one-hinge 的表达损失，但比 B48a 更少 redundancy。

```text
positive hinge + negative hinge；
shared threshold；
shared projection；
q rank 不增加；
W+ / W- 可以共享部分 layout；
```

通过条件：

```text
step ratio <= 1.10；
A4 pass；
AUC-step 不差于 B48a。
```

### P1.4 `FHQ-SignedPairLite`

目的：保留 LQ / signed-pair 的局部 pairwise 强项，但只作为少量 fused q channel。

```text
fixed signed-pair projection；
small r；
only q readout trainable；
no dense pair materialization。
```

防止重走 C8/C13 的路：必须测 E6/E8 rotated/random quadratic，不只测 E1。

### P1.5 `FHQ-IdentityTail`

目的：改善 AUC-time / early trajectory，而不是加表达容量。

```text
identity edge path 强保留；
hinge/quadratic 初始 scale 小；
训练早期 direct channel 保证快速下降；
中期 q channel 接管 interaction。
```

注意：这不是温度小修，而是结构性 early signal path 设计。

---

## P2. Expression gate A4

A4 不能只看一个 pairwise target。必须使用：

```text
E0 additive；
E1 pairwise product；
E2 composition；
E6 rotated pairwise；
E8 random quadratic；
local XOR / high-frequency optional diagnostic。
```

`v126_expression_audit.csv` 字段：

```text
candidate_id
implementation_id
target_id
protocol
seed
trainable_R2
frozen_readout_R2
delta_vs_MLP
delta_vs_B48a
dead_basis_fraction
basis_energy_entropy
A4_pass
```

A4 hard gate：

$$
\Delta R^2_{key} \ge -0.005
$$

for key targets relative to MLP, or at minimum:

$$
R^2_{candidate} \ge R^2_{B48a}-0.005.
$$

若 candidate 比 MLP 略差但优于 B48a，可以保留为 exploratory；若伤 B48a A4，则不能进入 task。

---

## P3. Task gate A5

A5 不只看 final accuracy。必须同时记录：

```text
accuracy；
worst-row；
near pass；
ECE；
NLL；
CEp99；
margin_p10；
AUC-step；
AUC-time；
time-to-target；
step-time distribution。
```

`v126_task_triage.csv` 字段：

```text
candidate_id
implementation_id
dataset
seed
epoch
val_acc
test_acc
val_loss
NLL
ECE
CEp99
margin_p10
val_loss_auc_step
val_loss_auc_time
time_to_target_loss
time_to_target_acc
step_time_q50_ms
step_time_q90_ms
step_ratio_q90
near_pass
A5_subgate_failure
```

A5 triage gate：

$$
\Delta Acc_{mean}\ge 0.
$$

$$
\Delta Acc_{worst}\ge -0.015.
$$

$$
NearPassRate\ge 0.80.
$$

$$
ECE_{candidate}\le ECE_{MLP}+0.02.
$$

$$
AUCtime_{mean}\le AUCtime_{MLP}.
$$

$$
AUCtime_{max}\le 1.05AUCtime_{MLP}.
$$

Promotion gate：

$$
NearPassRate\ge 0.8889
$$

for 3 datasets x 3 seeds, and:

$$
\Delta Acc_{worst}\ge -0.010.
$$

---

## P4. AUC-time 归因

每个 task run 必须输出：

```text
AUC-step ratio；
AUC-time ratio；
step-time q90；
loss slope early；
loss slope mid；
logit norm trajectory；
margin_p10 trajectory；
CEp99 trajectory。
```

分类：

```text
Case P4-A:
  AUC-step pass, AUC-time fail。
  说明 kernel/time 是 blocker，回 Line A。

Case P4-B:
  AUC-step fail, AUC-time fail。
  说明 primitive trajectory 是 blocker，回 Line P/C。

Case P4-C:
  AUC-step pass, AUC-time pass, final acc fail。
  说明 late capacity/tail 是 blocker，不做 temperature 小修，先看 CEp99/margin。

Case P4-D:
  accuracy pass, ECE fail。
  说明 calibration/tail path 问题，functional diagnostic 可重点看 tail-stability。
```

---

# 7. Line C：Manifold-Channel Geometry Diagnostics

## C0. 目标

Line C 不是附属指标，而是第三条并行主线。它回答：

```text
B48a / fused candidate 是不是用坏几何换来 task？
AUC-time 失败是否来自 train-probe coupling 不稳定？
functional update 是否真的改善 signal-channel geometry？
```

---

## C1. Train-probe coupling

从 train stream 拆：

```text
B = update batch
Q = probe batch
```

窗口 $[t,t+\Delta]$ 内：

$$
\Delta U_B=f_{\theta_{t+\Delta}}(B)-f_{\theta_t}(B).
$$

$$
\Delta U_Q=f_{\theta_{t+\Delta}}(Q)-f_{\theta_t}(Q).
$$

拟合：

$$
A_t=\arg\min_A \|\Delta U_Q-A\Delta U_B\|_F^2+\lambda\|A\|_F^2.
$$

指标：

$$
CouplingR^2=
1-
\frac{\|\Delta U_Q-A_t\Delta U_B\|_F^2}
{\|\Delta U_Q\|_F^2+\epsilon}.
$$

$$
CouplingCorr=
corr(vec(A_t\Delta U_B),vec(\Delta U_Q)).
$$

`v126_train_probe_coupling.csv` 字段：

```text
candidate_id
method
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
KernelDrift
CEp99_delta
ECE_delta
margin_p10_delta
official_gate_open
```

Base non-tearing gate：

$$
CouplingR^2_{KAN}\ge CouplingR^2_{MLP}-0.02.
$$

---

## C2. Signal / reservoir sketch

构造 projected logit-Jacobian sketch：

$$
\hat K_{BB}=\Phi\Phi^\top.
$$

窗口累积：

$$
\hat W_B=\sum_{\tau=t}^{t+\Delta}\hat K_{BB}(\tau).
$$

由 $\hat W_B$ 得到 $P_{sig}$ 与 $P_{res}$。

真实信号困在 reservoir：

$$
RealSignalReservoirRatio=
\frac{\|P_{res}r_{real}\|^2}{\|r_{real}\|^2+\epsilon}.
$$

噪声泄漏进 signal channel：

$$
NoiseSignalLeak=
\frac{\|P_{sig}r_{noise}\|^2}{\|r_{noise}\|^2+\epsilon}.
$$

`v126_signal_reservoir_sketch.csv` 字段：

```text
candidate_id
method
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

Base gate：

$$
NoiseSignalLeak_{KAN}\le NoiseSignalLeak_{MLP}+0.02.
$$

$$
RealSignalReservoirRatio_{KAN}\le RealSignalReservoirRatio_{MLP}+0.02.
$$

---

## C3. Kernel drift paired with coupling

不要把 kernel drift 当成越小越好。记录：

$$
KernelDrift=
\frac{\|\hat K(t)-\hat K(0)\|_F}
{\|\hat K(0)\|_F+\epsilon}.
$$

解释：

```text
good feature learning:
  KernelDrift 高，但 CouplingR2 高，NoiseSignalLeak 低，tail 不坏。

bad tearing:
  KernelDrift 高，CouplingR2 低，NoiseSignalLeak 高，CEp99/ECE 变坏。

lazy underfit:
  KernelDrift 低，但 RealSignalReservoirRatio 高，AUC-step 不下降。
```

---

## C4. 可视化

必须生成：

```text
fig_v126_coupling_predicted_vs_actual.svg
fig_v126_coupling_r2_by_candidate.svg
fig_v126_coupling_r2_vs_auc_step.svg
fig_v126_coupling_r2_vs_auc_time.svg
fig_v126_signal_spectrum.svg
fig_v126_real_signal_reservoir_ratio.svg
fig_v126_noise_signal_leak.svg
fig_v126_kernel_drift_vs_coupling.svg
fig_v126_noise_leak_vs_ECE.svg
fig_v126_real_signal_reservoir_vs_auc_time.svg
```

---

## C5. 不满足条件时 Codex 先尝试什么

如果 `CouplingR2` 低：

```text
1. 检查 AUC-step 是否也低；
2. 检查 early margin_p10；
3. 检查 direct identity channel 是否过弱；
4. 尝试 FHQ-IdentityTail；
5. 缩短窗口 Delta，排除非线性过大；
6. 使用 classwise centered logits；
7. 不按 dataset name 调参。
```

如果 `NoiseSignalLeak` 高：

```text
1. 检查 shuffled-label residual 构造；
2. 调整 P_sig 阈值而不是降低 gate；
3. 对 functional candidate 做 signal-projected update；
4. 对 primitive 降低 high-frequency / quadratic initial scale；
5. 检查 CEp99/ECE 是否同步变坏。
```

如果 `RealSignalReservoirRatio` 高：

```text
1. 检查 A4 expression 是否对 hard targets 低；
2. 检查 signal spectrum top eigen collapse；
3. 尝试增加结构性 interaction，而不是增大 hidden；
4. 检查 direct path 是否无法覆盖 real label residual；
5. 记录为 primitive trajectory blocker。
```

---

# 8. Line B：Functional Update Diagnostic / Official Re-entry

## B0. 目标

Functional update 双线继续，但不能抢 base gate。

Base 未合格时：

```text
functional_status = diagnostic_base_not_qualified
official_gate_open = 0
```

Base 合格后：

```text
functional_status = official_candidate
official_gate_open = 1
```

---

## B1. Functional candidates

只保留和当前 primitive 机制相关的 functional directions：

```text
B1-SignalProjectedGeometry:
  signal-channel projection 后的 geometry repair。

B2-HingeOccupancyRebalance:
  修 hinge active-region entropy，不改变 task logits 太多。

B3-QuadraticBranchConditionRepair:
  修 q sketch condition / scale，不加 branch capacity。

B4-AdamWOrthogonalTailRepair:
  与 AdamW direction 正交的 tail margin residual repair。

B5-FunctionPreservingConditionRepair:
  低 drift 的 coordinate condition maintenance。

B6-CouplingImprovementStep:
  以 train-probe coupling improvement 为 acceptance signal。
```

禁止：

```text
generic output perturbation；
future selector；
dataset-specific event；
teacher target；
loss modification；
direct class calibration loss。
```

---

## B2. Controls

每个 functional candidate 必须和以下 controls 同 batch / 同 checkpoint / 同 norm / 同 overhead 比较：

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
```

---

## B3. One-step / five-step cloned audit

`v126_functional_one_step.csv` 字段：

```text
candidate_id
base_candidate_id
checkpoint_id
dataset
seed
step
official_gate_open
control_id
update_norm
cos_with_adamw
cos_with_control
train_descent
probe_descent
holdout_descent_ratio
bad_step
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
ECE_delta
margin_p10_delta
rank_delta
A4_proxy_delta
amortized_time_ratio
beats_strong_controls
```

Gate：

$$
holdout\_descent\_ratio\ge 0.95.
$$

$$
bad\_step\_rate\le 0.02.
$$

$$
CouplingR^2_{func}\ge CouplingR^2_{base}+0.02.
$$

$$
NoiseSignalLeak_{func}\le NoiseSignalLeak_{base}-0.02.
$$

$$
RealSignalReservoirRatio_{func}\le RealSignalReservoirRatio_{base}-0.02.
$$

同时：

$$
CEp99_{func}\le CEp99_{base}+\epsilon.
$$

$$
A4Proxy_{func}\ge A4Proxy_{base}-0.003.
$$

且击败 controls。

---

## B4. Functional official re-entry

只有当：

$$
BaseQualified=1
$$

并且 one-step / five-step cloned audit 通过，才进入 short-run functional training。

Short-run methods：

```text
Base + AdamW
Base + NoOp overhead
Base + RandomMatchedNorm
Base + AdamWParallelDirection
Base + best functional candidate
MLP + analogous functional
```

Success：

$$
Acc_{func}\ge Acc_{base}-0.003.
$$

$$
AUCtime_{func}\le AUCtime_{base}.
$$

$$
ECE_{func}\le ECE_{base}.
$$

$$
CouplingR^2_{func}\ge CouplingR^2_{base}+0.02.
$$

$$
NoiseSignalLeak_{func}\le NoiseSignalLeak_{base}-0.02.
$$

$$
T_{amortized,func}/T_{base}\le 1.05.
$$

---

## B5. 不满足条件时 Codex 先尝试什么

如果 functional 输给 AdamWParallel：

```text
1. 计算 AdamW-orthogonal residual；
2. 限制 functional 只修 AdamW 不覆盖的 Line C debt；
3. 降低 update norm，但不降低 gate；
4. 若仍输，停止该 direction。
```

如果 functional 改善 coupling 但 task 变差：

```text
1. 加 task-safe backtracking；
2. 限制 logit drift；
3. 检查 margin_p10；
4. 不 promotion，只保留 diagnostic。
```

如果 functional task-safe 但几何不改善：

```text
1. 该 direction 不是 functional geometry；
2. 检查是否只是 NoOp/Random 等价；
3. 停止调 lambda。
```

如果 functional 改善某 dataset 但不是全部：

```text
1. 记录 dataset failure slice；
2. 做 leave-dataset-out；
3. 不能加入 dataset_name branch。
```

---

# 9. 并行执行计划

## 9.1 Batch 1：Fused kernel truth

并行运行：

```text
A1-FHQ0-forward-smoke
A2-FHQ0-backward-smoke
A3-FHQ0-fullstep-smoke
P1-FHQ1/FHQ2 primitive reference implementation
C1-B48a epochs8 Line C diagnostic
B1-B48a cloned functional diagnostic
```

Batch 1 的目的不是 task success，而是确定：

```text
B48a exact fused 能否正确且明显提速；
Line C 是否显示 AUC-time failure 的几何原因；
functional 是否有任何 control-resistant signal。
```

## 9.2 Batch 2：A4-preserving primitive triage

只让通过 A1 correctness / micro-efficiency 的 candidate 进 A4：

```text
FHQ0 exact；
FHQ1 one-hinge；
FHQ2 two-hinge shared；
FHQ-SignedPairLite；
FHQ-IdentityTail。
```

并行记录 A4 expression 与 C-line quick sketch。

## 9.3 Batch 3：A5 task triage

只让 A4 通过的 candidate 进 3 dataset x 3 seed：

```text
epochs = 5 and 8；
train/val/test same as v12.5.2；
no temperature/optimizer grid；
only one default recipe；
```

AUC-step 与 AUC-time 必须同时记录。

## 9.4 Batch 4：Fused official confirmation

对最好的 1-2 个 candidate 做：

```text
longer timing measurement；
official full-step profile；
CUDA graph fixed-shape path；
memory waterfall；
20/30 epoch optional confirmation。
```

## 9.5 Batch 5：Functional official re-entry

只有 base qualified 后运行。

---

# 10. 必须落盘的 artifacts

```text
v126_candidate_manifest.csv
v126_fused_forward_profile.csv
v126_fused_backward_correctness.csv
v126_fullstep_profile.csv
v126_expression_audit.csv
v126_task_triage.csv
v126_task_trace.csv
v126_auc_attribution.csv
v126_train_probe_coupling.csv
v126_signal_reservoir_sketch.csv
v126_manifold_channel_diagnostics.csv
v126_functional_one_step.csv
v126_functional_five_step.csv
v126_functional_control_matrix.csv
v126_failure_table.csv
v126_route_decision.json
v126_provenance_audit.csv
```

No-fake fields：

```text
fake_data_used
proxy_row_used
cpu_offload_used
uses_loss_backward
uses_torch_autograd_graph
dataset_name_branch_used
teacher_used
loss_modified
sampler_or_class_weight_used
```

---

# 11. 必须生成的图

## Efficiency

```text
fig_v126_forward_backward_step_ratio.svg
fig_v126_fullstep_time_breakdown.svg
fig_v126_memory_waterfall.svg
fig_v126_kernel_count_by_candidate.svg
fig_v126_step_ratio_distribution.svg
fig_v126_compile_warmup_vs_steady.svg
```

## Expression

```text
fig_v126_expression_delta_by_target.svg
fig_v126_A4_pass_dashboard.svg
fig_v126_dead_basis_fraction.svg
fig_v126_basis_energy_entropy.svg
```

## Task

```text
fig_v126_val_loss_vs_step.svg
fig_v126_val_loss_vs_time.svg
fig_v126_auc_step_vs_auc_time.svg
fig_v126_accuracy_delta_by_dataset_seed.svg
fig_v126_ECE_CEp99_margin_trace.svg
fig_v126_time_to_target.svg
```

## Manifold-channel geometry

```text
fig_v126_coupling_r2_by_candidate.svg
fig_v126_coupling_predicted_vs_actual.svg
fig_v126_signal_spectrum.svg
fig_v126_noise_signal_leak.svg
fig_v126_real_signal_reservoir_ratio.svg
fig_v126_kernel_drift_vs_coupling.svg
```

## Functional

```text
fig_v126_functional_control_gap.svg
fig_v126_functional_coupling_delta.svg
fig_v126_functional_noise_leak_delta.svg
fig_v126_functional_task_nonharm.svg
fig_v126_functional_norm_cosine.svg
```

---

# 12. Route decision

## R1：Fused exact B48a success

条件：

```text
FHQ0 exact fused passes A1/A4/A5/Line C。
```

结论：

```text
B48a math is valid；
previous blocker was implementation/timing；
open functional official cloned audit；
prepare 10-seed base confirm。
```

## R2：Fused B48a efficiency pass but AUC-step fail

条件：

```text
A1 pass；
A4 pass；
AUC-time fail；
AUC-step fail。
```

结论：

```text
implementation is not the main blocker；
primitive trajectory is the blocker；
go Line P and Line C；
do not tune optimizer/epoch。
```

## R3：Fused B48a AUC-step pass but AUC-time fail

条件：

```text
AUC-step pass；
AUC-time fail。
```

结论：

```text
system timing still blocker；
continue kernel fusion / CUDA graph / workspace；
do not redesign primitive yet。
```

## R4：Low-cost primitive efficiency pass but A4 fail

条件：

```text
FHQ1 or B50-like candidate very fast；
A4 expression fail。
```

结论：

```text
too cheap；
restore structural interaction；
do not enter task。
```

## R5：Primitive A4 pass but task fail

条件：

```text
A4 pass；
A5 fail。
```

结论：

```text
task trajectory / signal channel blocker；
use Line C to diagnose；
try IdentityTail / signal-preserving direct path；
no dataset tuning。
```

## R6：Base qualified but functional fails controls

条件：

```text
BaseQualified=1；
functional does not beat strong controls。
```

结论：

```text
base route can continue；
functional update not proven；
do not claim functional advantage。
```

## R7：Base + functional pass

条件：

```text
BaseQualified=1；
functional task-safe；
Line C improves；
beats controls；
amortized cost bounded。
```

结论：

```text
open 3-seed short-run functional training；
then 10-seed confirm。
```

---

# 13. 最终成功标准

## 13.1 Base success

$$
A1=1,\quad A4=1,\quad A5=1,\quad C_{nontearing}=1.
$$

具体：

$$
T_{step}/T_{MLP}\le 1.10.
$$

$$
M_{step}/M_{MLP}\le 0.80.
$$

$$
\Delta Acc_{mean}\ge 0.
$$

$$
\Delta Acc_{worst}\ge -0.010.
$$

$$
AUCtime_{mean}\le AUCtime_{MLP}.
$$

$$
ECE_{KAN}\le ECE_{MLP}+0.02.
$$

$$
A4_{KAN}\ge A4_{B48a}-0.005.
$$

$$
CouplingR^2_{KAN}\ge CouplingR^2_{MLP}-0.02.
$$

$$
NoiseSignalLeak_{KAN}\le NoiseSignalLeak_{MLP}+0.02.
$$

## 13.2 Functional success

$$
Acc_{func}\ge Acc_{base}-0.003.
$$

$$
AUCtime_{func}\le AUCtime_{base}.
$$

$$
ECE_{func}\le ECE_{base}.
$$

$$
CouplingR^2_{func}\ge CouplingR^2_{base}+0.02.
$$

$$
NoiseSignalLeak_{func}\le NoiseSignalLeak_{base}-0.02.
$$

$$
RealSignalReservoirRatio_{func}\le RealSignalReservoirRatio_{base}-0.02.
$$

$$
T_{amortized,func}/T_{base}\le 1.05.
$$

并且：

```text
beats NoOp；
beats RandomMatchedNorm；
beats AdamWParallel；
beats SNR-only；
beats MLP analog。
```

---

# 14. 给 Codex 的执行摘要

Codex 下一步不要做：

```text
temperature；
epoch；
optimizer impl；
directskip scale；
fixed gain / lower gain；
logitnorm / post-hoc calibration；
scalar normabs / meanstat 小网格；
direct-tail abs/square / mixed abs-square / sqdiag energy-tail 小网格；
signed cubic / signed polynomial tail 小网格；
dense random pair trajectory 小修；
dataset-specific threshold；
functional full training。
```

已完成并落盘：

```text
1. 实现 FHQ0 = B48a exact math fused forward。
2. 对 forward 做 correctness + micro timing。
3. 实现 fused CE backward，先固定 P，不算 dP。
4. 做 gradient correctness。
5. 实现 F1/F2 full-step profile。
6. 如果 exact fused 过 efficiency，跑 A4。
7. 如果 A4 过，跑 A5 5/8 epoch。
8. 同步跑 Line C coupling / signal-reservoir。
9. Functional 只做 cloned diagnostic，official_gate_open=0，直到 base qualified。
10. 修复 A5 protocol：warmup 与 AUC 分离，同时要求 AUC-step 与 AUC-time。
11. 排查 B109 final025、B110 quad020、B124/B125 fixedbranch+fixedgain、B126 logitnorm150。
12. 排查 B127 absquad、B128 absmixsq、B129 sqdiag025；B127/B128 停在 R0，B129 过 A1/A4 但 A5 task/AUC 失败。
13. 排查 B130 cubicdiag025；signed cubic tail correctness pass，但 full-step official fail，A4/A5 合法关闭。
14. 实现 B131 pairtraj 与 F4 fixed-P workspace path；F4 step = 0.685358064760707、backward = 0.4557250926585637、update = 0.7320725649235357，证明 cost-reduction path 真实存在，但 A4 fail。
15. 实现 B133/B135 localdensepairtraj；A1/A4 可打开，但 task 坍塌，B135 fixedbranch/fixedgain 也未救回，dense random pair trajectory 方向被排除。
```

Codex 下一步要先做：

```text
1. 不再继续 LR/gain/logitnorm/scalar calibration。
2. 不再继续 direct-tail arithmetic 或 square-energy tail 小网格。
3. 不再继续 signed polynomial tail 小网格。
4. 保留 F4 fixed-P workspace 这种真实 cost-reduction path，但不要继续 dense random pair trajectory。
5. 设计新的 sparse/local/task-stable single-kernel-friendly trajectory primitive，目标是降低 Fashion steady NLL AUC。
6. 若继续 B109 h160，必须实质降低 learnable-P backward/update cost，例如减少 `proj_grad` / 参数更新负担，而不是 optimizer wrapper 或 gate 小修。
7. 新 primitive 必须先证明 A1/A4，再进入 A5。
8. Functional 仍只做 diagnostic，直到 base qualified。
```

最重要的判断纪律：

$$
\boxed{
\text{v12.6 的成功不靠调小 gate，}
\text{而靠 lower-level fused implementation 和更结构化的低成本 task-geometry primitive。}
}
$$
