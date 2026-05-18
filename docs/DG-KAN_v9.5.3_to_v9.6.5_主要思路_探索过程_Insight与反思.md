# DG-KAN v9.5.3 到 v9.6.5：主要思路、探索过程、Insight 与反思

> 这份文档面向**不熟悉项目背景**的读者。  
> 我会尽量不用抽象词；必须出现的项目名、指标名、实验名都会先解释。  
> 这不是论文结论，也不是成功报告，而是一份“我们这十几轮到底在查什么、为什么这么慢、现在卡在哪”的复盘。

---

# 0. 先用人话说我们在做什么

普通神经网络训练时，每一步都会根据 loss 算梯度，然后用 AdamW 之类的优化器改一次参数。

我们现在想做的事是：

```text
在普通 AdamW 改参数之外，偶尔再给 KAN 加一小块额外参数改动。
```

这块额外改动，我们叫 **functional update action**，下面简称 **动作**。

一个动作可以理解成：

```text
“在这一训练步，除了 AdamW，还要不要额外把模型参数往这个方向推一下？”
```

我们最终不是想在 MNIST / Fashion-MNIST / KMNIST 上调榜，而是想证明：

```text
KAN 能不能通过这种额外动作，变成一种比普通 MLP 更值得用的训练系统？
```

这个系统必须满足很多条件：

```text
1. 仍然用标准分类 loss，不偷偷改 loss；
2. 不用 teacher，不用蒸馏，不加 auxiliary loss；
3. KAN 的 forward / backward / update 是手写的，不靠 PyTorch loss.backward 图；
4. 额外动作必须是 update rule，不是 loss trick；
5. 运行时间不能比 MLP 慢太多；
6. 不能针对某个数据集调规则；
7. 选中的动作必须真的比 AdamW / bestLR / NoOp / Random 这些对照好；
8. 还要能泛化、抗噪声、不遗忘旧知识。
```

所以我们现在最核心的问题不是：

```text
这个模型 acc 高不高？
```

而是：

```text
训练当下，能不能合法、稳定、便宜地认出或生成一小批“额外改参数后真的有用且不伤模型”的动作？
```

---

# 1. 本文会用到的词，先解释清楚

## 1.1 KAN、MLP、LQ-t2-h256

**MLP** 是普通多层感知机，可以理解成最常见的全连接神经网络。

**KAN** 是我们要研究的模型族。它的核心想法不是普通线性层加激活，而是让边上带有可学习函数。项目最终想做的是 Clean FullEdge PureKAN，也就是结构上尽量保持 KAN 的本质。

**LQ-t2-h256** 是当前主要 KAN 底座。你可以把它理解成：

```text
一个当前比较稳定、能作为实验平台的 KAN base 模型。
```

它不是最终 functional success，只是底座。

---

## 1.2 动作 action

**动作**就是一次额外的参数改动。

普通训练是：

$$
\theta_{new}=\theta_{old}+\Delta\theta_{AdamW}
$$

我们想做的是：

$$
\theta_{new}=\theta_{old}+\Delta\theta_{AdamW}+\Delta\theta_{extra}
$$

这里 $\Delta\theta_{extra}$ 就是一个动作。

一个动作必须能被真实加到模型上，不能只是纸面记录。因此每个动作都要检查：

```text
payload hash 是否一致；
action apply error 是否接近 0；
重新 replay 是否得到相同参数变化；
no-transform 重新跑是否一致。
```

---

## 1.3 horizon：h20 / h80 / h240

我们不会只看动作加完后下一秒怎么样，而是看三个距离：

```text
h20：加完动作后，往后看 20 个训练单位；
h80：往后看 80 个训练单位；
h240：往后看 240 个训练单位。
```

一个动作如果 h20 好，但 h240 造成很大风险，就不能算好。

---

## 1.4 RealFunctional、AdamWParallel、bestLR、NoOp、Random

为了判断一个动作是不是真的有用，我们会把同一个训练状态分成多个分支来跑：

| 名字 | 直观含义 |
|---|---|
| RealFunctional | 真的加这个额外动作 |
| AdamWParallel | 同样条件下只按 AdamW 走 |
| bestLR | 用强学习率对照，看是不是普通学习率调好就能做到 |
| NoOp | 什么都不额外加 |
| Random | 加随机动作，防止“随便加点东西也有效”的假象 |
| Shuffled payload | 把动作内容打乱，做负控 |

一个动作要好，至少应该比这些对照好。

---

## 1.5 V、long-risk、bad、null

我们用几个直观指标看动作好坏：

| 名字 | 解释 |
|---|---|
| V | 动作带来的收益。RealFunctional 比最强对照好多少。|
| long-risk | 长期风险。通常看 h240 有没有伤模型。|
| bad | 明确坏事件。比如让 loss tail、margin、稳定性明显变差。|
| null | 没明显用。动作不一定坏，但也没带来收益。|

一个动作如果只是“不坏但没用”，也不应该加，因为它浪费时间。

---

## 1.6 好训练形状

以前我们只看动作 outcome，好不好、坏不坏。后来发现不够。

我们现在关心的是：这个动作有没有让网络训练过程的“形状”变好。

这里“训练形状”不是抽象概念，它至少包括：

```text
1. 短期是否有收益；
2. 中期是否不坏；
3. 长期是否不出风险；
4. 是否不是噪声方向；
5. 是否没有破坏旧知识；
6. 是否没有让局部覆盖塌缩；
7. 是否没有让 hard tail 变坏；
8. 是否没有引入 memory/offdiag 风险；
9. 是否运行成本低。
```

其中：

- **memory** 指旧 family、旧 stratum、训练记忆区有没有被动作伤害。简单说就是“不忘旧知识”。
- **offdiag** 指动作有没有造成跨样本、跨局部区域之间的错误耦合。简单说就是“不要让一个局部修正伤到别的区域”。
- **cover** 指 KAN basis / 局部函数覆盖是否还健康。简单说就是“模型局部小片区不要塌”。

---

## 1.7 legal field、diagnostic field、red field

这是 v9.5.6 以后最重要的边界。

我们把字段分三类。

### Green / legal field：训练当下可以看到

比如：

```text
当前 batch 的梯度；
当前 batch 的 logit / CE / margin；
动作和 AdamW 的夹角；
payload norm；
当前 train-memory buffer 上的变化；
计算成本。
```

这些可以用于线上选动作。

### Amber / diagnostic field：可以离线分析，但不能线上直接用

比如：

```text
calibration split 上跑出来的 outcome label；
某个动作事后属于 GradeA/B/C；
某个 replay 后的评价。
```

这些可以用来训练或审计，但不能当作训练当下的输入。

### Red field：禁止用于 official 选择动作

比如：

```text
V_integrated；
risk_score；
未来 h20/h80/h240 outcome；
GradeAB label 本身；
validation/test metric；
旧表里的 outcome。
```

因为这些是“答案”或未来结果。用了它们，就等于作弊。

---

## 1.8 ranker、certificate、controller

这几个词容易抽象，我直接翻译成人话。

| 名字 | 人话解释 |
|---|---|
| ranker | 给动作打分排序的东西。它说“这些动作看起来更值得试”。|
| certificate | 动作附带的一张收据。它要证明这个动作为什么安全、为什么有用。|
| controller | 真正在线上训练时决定“这个动作要不要加”的规则。|

关系是：

```text
ranker 可以只是诊断；
certificate 需要更可信；
controller 必须能上线，不能用未来结果，不能按数据集调参，还要过运行时间。
```

---

## 1.9 TopK、accepted region

**TopK** 就是排序后取前 K 个动作。

比如 TopK87：

```text
把动作按分数排，从高到低取前 87 个。
```

为什么常见 87？因为 canonical action 总数是 2876，coverage 下限 0.03 对应：

$$
\lceil 0.03 \times 2876 \rceil = 87
$$

**accepted region** 是 controller 真正接受的那批动作。

TopK 高不一定等于 accepted region 能用。因为 TopK 可能只在某个局部 group 里好，冻结成线上规则后会崩。

---

## 1.10 LDO、LSO、LTO

这几个是 leave-out 检查。

| 名字 | 解释 |
|---|---|
| LDO | leave-dataset-out。留掉一个数据集，看规则在没见过的数据集上是否还行。|
| LSO | leave-stratum-out。留掉某类训练状态或样本分层，看是否仍然行。|
| LTO | leave-template-out。留掉某类动作模板，看规则是不是只依赖一个模板。|

我们不能按数据集调参，所以 LDO 很重要。

---

## 1.11 group pocket

**group pocket** 可以理解成：

```text
某个局部小口袋里动作特别好，但换个口袋就不行。
```

比如 v9.5.9 到 v9.6.2 一直看到 payload0 这个区域很强。后来 v9.6.1 做 intervention，发现它不是原因，只是混杂现象。也就是说：

```text
payload0 里有好动作，不代表 payload0 本身就是好动作机制。
```

---

## 1.12 candidate template / lineage

**candidate template** 是动作来源模板。

如果 87 个好动作看起来很多，但其实都是同一个模板扩出来的，那就不能说规则通用。

**lineage** 是动作来源链路。我们要看：

```text
这批动作是不是来自很多不同来源？
还是一个来源复制扩张出来？
```

v9.6.4 一度发现旧 template 口径下只有 1 个 template，v9.6.5 重建后变成 87 个 template，这个修正很重要。

---

# 2. 从 v9.5.3 到 v9.6.5 的主线变化

这段时间的主线可以分成四步。

## 第一步：v9.5.3 到 v9.5.5，从“动作好不好”转成“训练形状好不好”

早期我们只问：动作加了以后 outcome 好不好。

v9.5.3 开始，我们改问：

```text
这个动作有没有让 KAN 的训练状态变得更健康？
```

于是我们开始记录：

```text
SNR；
cover；
curvature；
memory；
hard-tail；
long-risk；
V；
bad/null；
cost。
```

这一步的主要收获是：

```text
好训练形状确实存在，但数量少；
生成器能生成合法动作，但生成不出好训练形状；
单个 AUC 好看不够，TopK/accepted region 才关键。
```

## 第二步：v9.5.6 到 v9.5.7，把“看起来很强的证书”拆成合法和非法

v9.5.5 看到 GCERT18 TopK 很强，像是快成功了。

但 v9.5.6 查出 GCERT18 用了结果字段，比如 $V_{integrated}$ 和 risk_score。这些字段训练当下看不到，不能 official。

于是我们把字段分成：

```text
legal：训练当下能用；
diagnostic：只能离线分析；
red：不能用于 official。
```

v9.5.7 又发现：完全合法的 rank 不是没有信号。legal rank 能在 TopK 里找到好动作，但 LDO / LSO 不稳。

这一步的主要收获是：

```text
事后答案很容易选出好动作；
训练当下合法信息也有一点希望；
但它还不是跨 group 稳定规则。
```

## 第三步：v9.5.8 到 v9.6.2，把强 legal rank 拆成 group pocket

v9.5.8 的 R2 ranker TopK87 很强：precision 约 0.8506，V 是正的，longrisk 为 0。但 leave-out drop 高达 0.885。

这说明：

```text
ranker 在整体池子里能找好动作；
但换一个 dataset / stratum / group 就大幅掉。
```

v9.5.9 发现这批好动作高度集中在 payload_norm_bucket / payload0。

v9.6.1 做 intervention，结果 payload0 normalize 后没有 lift，反而 V 和 GradeAB lift 都是负的。于是 payload0 被判定为混杂口袋，不能作为机制。

v9.6.2 正式禁止 payload_norm_bucket 作为 selector，只能作为诊断分层。

这一步的主要收获是：

```text
payload0 是高分口袋，但不是可用机制；
existing-action legal rank 仍然强；
当前真正问题变成 group-stable accepted region。
```

## 第四步：v9.6.3 到 v9.6.5，把 group pocket 继续拆到 template / memory / offdiag

v9.6.3 发现 accepted region 很强，但能被 hidden group pocket 解释。

v9.6.4 继续查，发现旧 candidate_template 口径下 87 个 accepted actions 像是来自 1 个模板。

v9.6.5 重建 lineage hierarchy 后，发现这不是最终真相：新的 candidate_template_id 下，accepted region 有 87 个 unique template，LTO 也基本稳定。

同时 v9.6.5 的 memory/offdiag matched-pair causal gate 首次过了：

```text
GradeAB lift = 0.1914
V lift = 0.2904
longrisk drop = 0.9067
```

这说明 memory/offdiag safe 不是随机噪声，而是很强的风险控制线索。

但 v9.6.5 仍然失败，因为：

```text
coverage 够的目标 bad/null 太高；
干净目标数量不够；
ranker/certificate 被 LDO drop 否决；
generated primitive APGT 仍然 value-negative / high-longrisk。
```

---

# 3. 逐版本详细复盘

下面按版本讲清楚：每轮问了什么、做了什么、看到什么、学到什么。

---

## 3.1 v9.5.3：第一次认真问“动作的训练形状好吗？”

### 当时的问题

v9.5.2 已经说明：按短中长期 outcome 扫 8000 个 target，也找不到足够干净、数量足够的目标。

于是 v9.5.3 改问：

```text
好动作是不是不应该只由 h20/h80/h240 outcome 定义？
是不是应该看它有没有改善训练形状？
```

### 做了什么

v9.5.3 建了 **GeometryCard**。这张卡给每个动作记录训练形状指标。

它真实落盘了：

```text
GeometryCard rows = 3388
canonical AP0 rows = 2876
generated APY rows = 512
missing / NaN / Inf = 0
```

同时实现了 APG1-APG8 这类 geometry-aware primitive，生成了 512 个新动作，并跑完 branch-horizon。

### 结果

最干净的 GoodGeomAction 只有 5 个：

```text
GGA count = 5
coverage = 0.00174
V_integrated LCB = 0.1308
h240 longrisk = 0
bad/null = 0
```

但 coverage 下限一般要 0.03，需要约 87 个动作，现在只有 5 个。

APG 生成器也失败：

```text
best APG = APG1
GGA precision = 0.03125
V_integrated LCB = -0.4062
h240 longrisk = 0.6875
```

### 学到什么

```text
1. 好训练形状不是 0 个，但太稀疏；
2. SNR 单独不够；
3. APG 能生成合法动作，但不是好动作；
4. 不能只看 action apply / payload replay 是否成功，必须看 outcome 和几何质量。
```

---

## 3.2 v9.5.4：把 GeometryCard 扩成 Geometry Outcome Ledger

### 当时的问题

v9.5.3 只有 5 个 GGA 太少。我们想知道：

```text
是不是定义太严？
有没有稍微放松但仍干净的目标？
```

### 做了什么

v9.5.4 建了 **Geometry Outcome Ledger**。它把 AP0、APY、APG 的 outcome 和训练形状指标放到一张表。

落盘情况：

```text
rows = 3900
canonical AP0 = 2876
generated APY = 512
generated APG = 512
identity join pass = 1
```

### 结果

找到一个干净但太薄的目标：`T5-RelaxedV80NonNegative`。

它有：

```text
action count = 57
coverage = 0.019819
V_integrated LCB = 0.1960
h240 longrisk = 0
bad/null = 0
```

这说明不是没有好动作，但数量仍不够 87。

GeoScore 也有 weak signal：

```text
GS1 accepted = 91
coverage = 0.06594
V_integrated LCB = 0.04369
longrisk = 0
```

但 bad event 超线，不能 controller。

APGS 新动作失败：

```text
best APGS7 OfficialGeo precision = 0.015625
V_integrated LCB = -0.9219
h240 longrisk = 0.765625
```

### 学到什么

```text
1. 干净目标 T5 存在，但 action count 不够；
2. 多指标 score 比单指标更接近可用，但仍有 bad event；
3. 生成器还是合法 payload builder，不是好训练形状 builder；
4. AUC 高不等于 TopK / accepted region 好。
```

---

## 3.3 v9.5.5：引入多等级标签，发现 GCERT18 这个强线索

### 当时的问题

T5 只有 57 个，还不够。于是我们把动作分成等级，不再只看一个二元标签。

### 做了什么

v9.5.5 建了 **Trainable Geometry Ledger v2**，包含：

```text
canonical AP0 = 2876
generated APY/APG/APGS = 512 / 512 / 512
ledger rows = 4412
```

并把动作分成 Grade A/B/C/D/E。这里 Grade 的意思是：

```text
Grade A：最干净，最接近可以用；
Grade B：也比较好，但可能需要风险过滤；
Grade C：有信号，但需要谨慎；
Grade D/E：风险或无效。
```

### 结果

Grade 计数是：

```text
A/B/C/D/E = 40 / 39 / 18 / 89 / 2690
```

这很重要：

```text
A+B = 79，离 87 只差 8；
A+B+C = 97，已经超过 87。
```

但 A/B/C 混起来可能带风险，不能直接用。

v9.5.5 还发现 GCERT18 TopK 很强：

```text
AUC_GradeB = 0.9927
TopK64 GradeB precision = 0.890625
TopK64 V_integrated LCB = 0.1650
TopK64 longrisk/bad/null = 0
```

但 ECE 很差：

```text
ECE = 0.6218
```

APGR 生成器仍失败：

```text
best APGR5 GradeB precision = 0.046875
V_integrated LCB = -0.8658
h240 longrisk = 0.78125
```

### 学到什么

```text
1. 好动作不是 0 个，Grade A/B/C 加起来开始接近 density 下限；
2. GCERT18 的 TopK 很强，值得查；
3. 但它可能只是排序强，不是概率校准强；
4. APGR 继续说明生成器目标错了；
5. cover 和 memory 是硬问题：cover collapse、old-family forgetting 都出现了。
```

---

## 3.4 v9.5.6：GCERT18 被降级，原因是它用了未来结果字段

### 当时的问题

GCERT18 TopK 很强。我们必须查：

```text
它是真的训练当下可用，还是偷看了结果？
```

### 做了什么

v9.5.6 做了字段合法性审计。

### 结果

发现 GCERT18 用了 outcome-derived fields：

```text
V_integrated
risk_score
```

这两个是结果字段，训练当下不能用。

legacy GCERT18 仍很强：

```text
TopK64 GradeAB precision = 0.890625
V_integrated LCB = 0.1650
h240 longrisk = 0
```

但 legal surrogate 完全复现不了：

```text
legal TopK64 GradeAB precision = 0.046875
V_integrated LCB = -0.5603
h240 longrisk UCB = 0.6991
```

APGC 生成器仍失败，甚至 negative control 成了 best diagnostic。

### 学到什么

```text
1. 事后知道答案很容易选好动作；
2. 训练当下可用字段没有复现这个强信号；
3. 必须严格区分 legal / diagnostic / red fields；
4. 不能因为 TopK 很强就 official。
```

这轮是一次关键排雷。

---

## 3.5 v9.5.7：legal rank 第一次出现希望，但 LDO/LSO 崩

### 当时的问题

既然 GCERT18 不能 official，那合法字段到底有没有希望？

### 做了什么

v9.5.7 做了 field legality ledger、legal causal geometry features、rank upper-bound、distillation sandbox、memory anatomy、APGM primitive。

### 结果

legacy rank 强但非法：

```text
legacy TopK64 GradeAB precision = 0.890625
best legal variant TopK64 precision = 0.09375
```

但更复杂的 legal rank 开始有强诊断：

```text
UB-M1 small-MLP diagnostic ranker:
TopK87 GradeAB precision = 0.8276
V_integrated LCB = 0.1613
h240 longrisk UCB = 0.0545
```

可是：

```text
LDO / LSO drop = 0.8385
```

这说明它在整体池子里能找好动作，但换 dataset / stratum 就掉。

memory blocker 也很强：

```text
high-longrisk count = 3993
memory explained fraction = 0.7638
memory-safe value-positive count = 121
```

APGM 生成器失败：

```text
best APGM2 GradeAB precision = 0.015625
V_integrated LCB = -0.3176
h240 longrisk UCB = 0.7726
```

### 学到什么

```text
1. legal rank 不是完全看不见；
2. 但它不跨 group 稳；
3. memory 是长期风险的重要来源；
4. 生成器仍不是能力进展。
```

---

## 3.6 v9.5.8：TopK 很强，但 group stability 失败

### 当时的问题

我们要把 legal rank 从诊断变成稳定选择规则。

### 结果

R2 ranker TopK87 很强：

```text
TopK87 GradeAB precision = 0.8506
V_integrated LCB = 0.1342
h240 longrisk UCB = 0
```

但 group 稳定性崩：

```text
LDO/LSO drop = 0.8854 / 0.8854
```

APGA 工程过，但 outcome 失败：

```text
best APGA7 GradeAB precision = 0.03125
V_integrated LCB = -0.3581
h240 longrisk UCB = 0.8151
```

### 学到什么

```text
1. strong pooled TopK 不是 success；
2. ranker 可能只在某些 group 里好；
3. 必须做 group-invariant / group-stable 检查；
4. APGA 仍只是合法动作生成器，不是好动作生成器。
```

---

## 3.7 v9.5.9：发现 group-specific pocket，payload0 成为可疑口袋

### 当时的问题

为什么 v9.5.8 TopK 很强但 leave-out 崩？

### 结果

v9.5.9 发现：好动作高度集中在 `payload_norm_bucket=payload0`。

```text
dominant axis = payload_norm_bucket
dominant group = payload0
max topK group share = 1.0
max drop = 0.8506
```

GIR9 ranker仍强：

```text
TopK87 precision = 0.8506
V LCB = 0.1563
h240 longrisk UCB = 0.0339
LDO/LSO drop = 0.3563
```

比 v9.5.8 的 drop 0.885 明显好，但还不够。

APGA OOD/damage 指向 longrisk veto ineffective，memory/offdiag 与 longrisk 强相关。

### 学到什么

```text
1. legal rank 有强信号；
2. 但信号集中在局部 pocket；
3. payload0 可能是原因，也可能只是混杂；
4. 需要 intervention，而不是继续调 threshold。
```

---

## 3.8 v9.6.0：payload pocket 仍未解释，rank 信号强但不是通用规则

### 结果

payload0 drop 很强：

```text
payload0 removal precision drop = 0.8506
```

但解释力度不够：

```text
payload_norm bucket explained fraction = 0.1818
matched pair count = 0
```

VR4 和 GIR9 仍强：

```text
VR4 TopK87 precision = 0.8391, V LCB = 0.1348, longrisk = 0
GIR9 TopK87 precision = 0.8506, V LCB = 0.1563, longrisk = 0.0339
```

但 leaveout drop 仍在 0.34 到 0.36。

APGD 失败：

```text
best APGD2 GradeAB precision = 0.0625
V LCB = -0.8752
longrisk = 0.8561
```

### 学到什么

```text
payload0 很像强口袋，但还没有因果证据；
existing-action rank 是最有希望路线；
generated route 仍持续失败。
```

---

## 3.9 v9.6.1：payload0 被 intervention 证伪

### 当时的问题

payload0 到底是不是好动作原因？

### 做了什么

v9.6.1 做了 intervention clone replay：把动作改成 payload0 相关形式，看是否产生正向 lift。

### 结果

```text
source actions = 104
intervention clones = 520
branch-horizon rows = 12480 / 12480
```

但结果是负的：

```text
payload0 normalized GradeAB precision = 0.0096
memory projection longrisk = 0.8558
mean V lift = -0.5199
mean GradeAB lift = -0.6135
payload0 causal lift supported = 0
payload0 confounded supported = 1
```

APGE 继续失败：

```text
best APGE6 GradeAB precision = 0.03125
V LCB = -0.8694
h240 longrisk UCB = 0.8825
```

### 学到什么

```text
payload0 不是可用机制；
它只是一个混杂口袋；
不能继续调 payload_norm_bucket；
必须做去混杂 rank 和 memory/offdiag-safe generator。
```

---

## 3.10 v9.6.2：payload pocket 正式停用，existing-action rank 仍然强

### 结果

payload0 stop audit 继续成立：

```text
matched-pair V lift = -0.5199
GradeAB lift = -0.6135
payload0 intervention precision = 0.0096
payload_norm_bucket_allowed_as_selector = 0
```

但 existing-action rank 仍强：

```text
RANK5 TopK87 GradeAB precision = 0.8736
V LCB = 0.1411
longrisk UCB = 0
```

CERT10 accepted 87 个动作：

```text
coverage = 0.03025
precision = 0.8736
V LCB = 0.1411
longrisk = 0
```

但仍失败，因为：

```text
LDO drop = 0.3793
max group share = 1.0
```

APGF 失败：

```text
best APGF5 GradeAB precision = 0.03125
V LCB = -0.3353
h240 longrisk = 0.8289
```

### 学到什么

```text
payload pocket 已停；
existing-action rank 很接近，但 group-stable 不过；
generated primitive 继续制造 longrisk；
下一步要查 accepted region 为什么 group share = 1。
```

---

## 3.11 v9.6.3：accepted region 质量是真的，但被 hidden pocket 解释

### 做了什么

审计 v9.6.2 那个强 accepted region 是否只是统计口径错误。

### 结果

scope 没错：

```text
accepted action / unique action / unique event = 87 / 87 / 87
duplicate action id = 0
scope mismatch = 0
universe mismatch = 0
```

accepted region 本身很强：

```text
GradeAB precision = 0.8736
V LCB = 0.1411
h240 longrisk UCB = 0
```

但 P2 认为它被 hidden multi-axis pocket 解释，其中 payload_norm_bucket=payload0 是主轴。不过 v9.6.3 里也暴露出：某些轴 base share 本来就是 1.0，不能作为解释。

APGH 失败：

```text
best APGH7 GradeAB precision = 0.046875
V LCB = -0.4881
h240 longrisk = 0.7869
```

### 学到什么

```text
accepted region 不是 duplicate 假象；
这批动作质量真的不错；
但它可能来自 hidden pocket；
需要排除 degenerate axis，继续看 lineage / memory / offdiag。
```

---

## 3.12 v9.6.4：发现旧 candidate-template 口径下塌缩，offdiag 成为重要线索

### 做了什么

v9.6.4 把 degenerate pocket 拆开：如果某个 group 在整个 universe 里 base share = 1.0，那它不能解释 accepted region。

### 结果

识别出 degenerate axes：

```text
payload_norm_bucket
action_norm_bucket
candidate_origin
```

去掉这些后，最强非退化口袋是：

```text
offdiag_bucket = offdiag_fail_0
base share = 0.1380
adjusted drop = 0.8736
```

这说明 offdiag / memory 可能是更真实的几何信号。

但是 v9.6.4 发现旧 `candidate_template_id` 下：

```text
candidate_template_unique_count = 1
candidate_template_max_share = 1.0
candidate-to-action expansion ratio = 87
candidate_id_collision_count = 86
```

也就是说，那 87 个动作可能像是一个模板扩张出来的。

APGL 失败：

```text
best APGL1 GradeAB precision = 0.03125
V LCB = -0.5827
h240 longrisk = 0.7869
```

### 学到什么

```text
1. 一些 group axis 是退化轴，不能拿来解释；
2. offdiag safe 很值得继续查；
3. accepted region 可能有 template 口径问题；
4. 需要重建 lineage hierarchy。
```

---

## 3.13 v9.6.5：template collapse 被修正，memory/offdiag causal gate 首次过

### 做了什么

v9.6.5 重新定义 candidate template。

### 结果 1：template collapse 不是最终真相

v9.6.4 旧口径下看起来只有 1 个 template。

v9.6.5 新口径下：

```text
unique candidate template count = 87
max candidate template share = 0.01149
template-to-action expansion ratio = 1.0
```

这说明 accepted actions 不是单模板假多样性。

### 结果 2：leave-template-out 基本过

```text
GradeAB precision = 0.8736
V_integrated LCB = 0.1411
longrisk UCB = 0
LTO precision drop = 0.01149
```

模板稳定性不再是主要 blocker。

### 结果 3：memory/offdiag matched-pair causal gate 过

```text
matched joint pairs = 397
intervention clones = 512
clone replay rows = 12288
GradeAB lift = 0.1914
V lift = 0.2904
longrisk drop = 0.9067
memory_offdiag_causal_pass = 1
```

这很重要。它说明：

```text
memory/offdiag safe 不是普通相关性；
它确实能解释一部分好动作区域；
但它更像风险过滤器，不是直接生成好动作的配方。
```

### 结果 4：仍没有 official target / controller

最好的 target T4.1：

```text
count = 97
coverage = 0.0337
V LCB = 0.1662
longrisk = 0
```

但：

```text
bad UCB = 0.1505
null UCB = 0.1892
LDO drop = 0.3333
max group share = 0.9278
```

更干净的 T4.2/T4.3：

```text
T4.2 count = 79, coverage = 0.02747
T4.3 count = 77, coverage = 0.02677
```

干净但数量不够。

R5B ranker很强：

```text
TopK87 precision = 0.8736
V LCB = 0.1411
longrisk = 0
candidate_template_count = 87
LTO drop = 0.01149
```

但 CERT13 仍因为 LDO drop = 0.3793 失败。

APGT 继续失败：

```text
best APGT4 GradeAB precision = 0.03125
V LCB = -0.6095
h240 longrisk = 0.7869
```

### 学到什么

```text
1. template collapse 被修正；
2. memory/offdiag 是强风险控制线索；
3. existing-action rank 是当前最有希望路线；
4. 但 dataset leave-out 仍不稳；
5. generated-action route 多轮工程闭合但科学失败。
```

---

# 4. 这段探索带来的主要 insight

## Insight 1：不能只用 acc 判断动作好不好

Accuracy 太粗。一个动作可能短期让 acc 或 loss 好一点，但：

```text
h80 坏；
h240 出 longrisk；
旧 family 被忘掉；
cover 塌缩；
只在某个 group 里好；
不能打过 bestLR；
运行太慢。
```

所以动作质量必须是一个向量，而不是一个 acc 数字。

推荐的动作质量表：

```text
V20, V80, V240
bad, null, longrisk
memory fail
offdiag fail
cover collapse
hard-tail change
cost
beats AdamWParallel / bestLR / NoOp / Random
LDO / LSO / LTO
```

---

## Insight 2：好动作存在，但数量和稳定性都不够

v9.5.3 的 GGA 只有 5 个，v9.5.4 的 T5 有 57 个，v9.5.5 的 Grade A+B 有 79 个，v9.6.5 的 T4.1 有 97 个。

这条线说明：

```text
不是完全没有好动作；
但干净、数量够、跨 group 稳定的动作集合还没同时满足。
```

典型冲突是：

```text
数量够的目标 bad/null 或 LDO 不过；
干净的目标数量不够。
```

---

## Insight 3：AUC 经常会骗我们，TopK 和 accepted region 更重要

很多次 AUC 很好看，但 TopK 很差。

例如：

```text
GCERT8 AUC_GGA 很高，但 TopK64 GGA precision 只有 0.125；
GCERT16 AUC 很高，但 TopK64 longrisk 仍高；
GCERT18 TopK 很强，但用了 outcome-derived fields。
```

真正 controller 不是问：

```text
全局排序曲线好看吗？
```

而是问：

```text
我真正要接受的那 87 个动作，够不够好，换 group 后还稳不稳？
```

---

## Insight 4：合法字段边界非常重要

v9.5.6 的最大教训是：看起来很强的 GCERT18，用了未来结果字段。

这提醒我们：

```text
只要 score 用了 V_integrated、risk_score、GradeAB、未来 outcome，就不能 official。
```

以后每个 score 都必须有 field legality ledger。

---

## Insight 5：existing-action rank 是当前最接近成功的路线

虽然它一直没 official，但它多次给出很强数字：

```text
TopK87 precision 约 0.85 - 0.87；
V LCB 为正；
longrisk UCB 为 0；
accepted count 可以到 87；
LTO 已经基本过。
```

尤其 v9.6.5 的 R5B：

```text
precision = 0.8736
V LCB = 0.1411
longrisk = 0
candidate_template_count = 87
LTO drop = 0.01149
```

这说明 existing-action route 没死。现在主要卡在 LDO。

---

## Insight 6：generated-action route 连续失败，不应再盲目加小变体

从 APG、APGS、APGR、APGC、APGM、APGA、APGF、APGH、APGL、APGT，多轮共同模式是：

```text
工程链路过；
payload hash 过；
action apply 过；
branch-horizon rows 过；
但 outcome value-negative / high-longrisk。
```

这说明不是实现没跑起来，而是生成目标错。

生成器不能再只是：

```text
合法 payload builder
```

它必须变成：

```text
value-preserving + memory-safe + offdiag-safe + longrisk-veto-effective 的动作生成器。
```

如果做不到，就应该暂时停止 blind generator variants。

---

## Insight 7：payload0 是混杂口袋，不是机制

v9.5.9 到 v9.6.2 一度看到 payload0 很强。

但 v9.6.1 intervention 显示：

```text
payload0 normalized precision = 0.0096
mean V lift = -0.5199
GradeAB lift = -0.6135
```

所以 payload0 不能用。

这给了一个通用教训：

```text
某个 group 里好动作多，不代表这个 group 是好动作原因。
```

必须做 intervention / matched pair，而不是只看 drop。

---

## Insight 8：memory/offdiag 是真正值得继续追的线索

v9.6.5 的 memory/offdiag matched-pair causal gate 首次过。

关键数字：

```text
GradeAB lift = 0.1914
V lift = 0.2904
longrisk drop = 0.9067
```

这说明：

```text
memory/offdiag safe 能显著降低 longrisk，并提高好动作概率。
```

但它还不是直接 recipe，因为 intervention clones 本身没有 positive lift。

更像是：

```text
一个很强的风险过滤器 / veto 条件。
```

---

## Insight 9：LDO 是当前最关键的硬门

v9.6.5 后 template 问题被修正，LTO 也基本过。

现在最接近成功的 route 被 LDO 否决：

```text
CERT13 LDO drop = 0.3793
```

这说明：

```text
同样的动作选择规则，换 dataset 后掉得太多。
```

注意：这不代表我们要按 dataset 调参。相反，下一步要诊断 dataset 差异，但 official controller 不能用 dataset branch。

---

# 5. 为什么感觉很慢，但不是完全没进度

你觉得慢是合理的。

因为从 v9.5.3 到 v9.6.5，我们一直没有进入：

```text
selected controller
-> selected runtime
-> official paired replay
-> short/full training
```

这条正反馈链。

但这段时间确实做了很多关键排雷：

```text
1. 把“好动作”从 acc/outcome 改成训练形状；
2. 建了 GeometryCard / Geometry Outcome Ledger / Trainable Geometry Ledger；
3. 发现 GCERT18 的强信号用了未来结果字段；
4. 把 legal rank 从完全看不见推进到 TopK 很强；
5. 发现 TopK 强但 group unstable；
6. payload0 口袋被 intervention 证伪；
7. accepted region scope/duplicate 问题被排除；
8. candidate-template collapse 被重新定义修正；
9. memory/offdiag 变成强机制线索；
10. generated-action 多轮失败被确认为目标问题，而不是工程问题。
```

这些不直接涨 acc，但它们让我们避免了很多错误成功。

如果没有这些检查，我们可能早就把下面这些误写成成功：

```text
GCERT18 strong TopK；
payload0 pocket；
v9.6.2 CERT10 accepted region；
v9.6.4 single-template region；
APG/APGR/APGT implementation pass。
```

但这些都不能代表真正 functional success。

---

# 6. 当前最准确状态：v9.6.5 后我们站在哪里

## 已经比较可信的东西

```text
1. canonical outcome truth base 已经可信；
2. old table / no-transform replay / fake-proxy 问题已经清理；
3. action payload / apply / branch-horizon materializer 基本可靠；
4. LQ-t2-h256 base 仍健康；
5. 现有 AP0 动作里有一批局部好动作；
6. existing-action legal rank 能强力找到部分好动作；
7. template collapse 已被新 lineage 口径修正；
8. memory/offdiag safe 是强风险控制线索；
9. generated-action 工程链路可靠，但科学结果持续失败。
```

## 仍未完成的东西

```text
1. 没有 official minimal controller；
2. 没有 selected runtime；
3. 没有 official LDO / LSO / LTO 全部过；
4. 没有 paired replay；
5. 没有 short/full functional training；
6. 没有证明 functional update 赢 MLP / bestLR / NoOp / Random；
7. 没有 external-ready Beyond-MLP 证据。
```

## 当前主 blocker

我会把 v9.6.5 后的主 blocker 写成三条：

```text
B1. Existing-action route:
    好动作 rank 很强，但 LDO 不稳。

B2. Target route:
    数量够的目标 bad/null 或 group 稳定性不够；
    干净目标数量不够。

B3. Generated-action route:
    多轮 primitive 都生成 value-negative / high-longrisk 动作，
    需要 stop-rule 和新生成目标。
```

一句话：

$$
\boxed{
\text{现在最有希望的是 existing-action rank；真正卡点是 LDO 稳定性和 bad/null/coverage 冲突。}
}
$$

---

# 7. 对“好几何结构”的重新理解

从 v9.5.3 到 v9.6.5，我们对“好几何结构”的理解已经比一开始清楚很多。

一个好的 functional update 不是：

```text
让某个 batch loss 降；
让 h20 好；
让 acc 短期涨；
TopK AUC 高；
certificate 分数高；
某个 group pocket 里 precision 高。
```

一个好的 functional update 应该是：

```text
1. 短期有正收益；
2. 中期不明显伤；
3. 长期不出 longrisk；
4. 不制造 bad/null；
5. 不忘 old family / old stratum；
6. 不进入 offdiag risk；
7. 不让 cover 塌；
8. 不只是某个 dataset / group / template 里的局部口袋；
9. 能用训练当下合法字段识别；
10. 运行成本可控。
```

更具体地说，每个动作应该有一张表：

```text
value:
  V20, V80, V240, V_integrated

risk:
  bad, null, h240 longrisk

memory:
  old-family fail, old-stratum fail, memory buffer delta

offdiag:
  population-risk offdiag fail, cross-region coupling risk

cover:
  basis effective rank delta, cover entropy delta, hard-tail cover change

signal:
  gradient mean/variance, SNR, AdamW alignment, control transfer improvement

cost:
  feature cost, certificate cost, payload apply cost, step ratio

support:
  dataset / stratum / template / family diversity
```

这个定义比单纯 acc 更符合终极目标。

---

# 8. 对相关工作的吸收

## 8.1 Deep Manifold 给我们的启发

Deep Manifold 的核心启发是：神经网络不是在固定坐标里拟合曲线，而是在训练中不断移动局部覆盖、改变坐标、积累或释放曲率，最后形成较稳定的区域。

换成 DG-KAN 的语言：

```text
KAN 的 edge/basis 像很多局部小片区。
一个动作如果只让短期 value 好，
但把局部覆盖拉塌、把旧 family 忘掉、把 offdiag risk 放大，
那它不是好动作。
```

这解释了为什么我们后来加入：

```text
cover；
curvature；
memory；
offdiag；
longrisk。
```

## 8.2 Generalization 相关工作给我们的启发

Generalization 那篇的启发是：训练方向里有 signal channel，也有 reservoir / noise。一个方向能降低训练 loss，不代表能转到测试或长期稳定。

它给了一个很实用的思想：

$$
\mu_k^2 > \frac{\sigma_k^2}{b-1}
$$

意思是：

```text
一个更新方向的平均信号，要强过 minibatch 方差噪声，才值得更新。
```

这启发我们：

```text
functional update 不能服务少数样本噪声；
它应该由多个样本、多个 group、多个 horizon 共同支持。
```

这也是我们为什么越来越重视：

```text
SNR；
leave-out；
memory/offdiag；
rank 的 group stability。
```

---

# 9. 对我们探索路线的反思

## 9.1 最大进步：我们没有把假成功写成成功

这很重要。

很多轮实验都出现过“看起来像成功”的东西：

```text
GCERT18 TopK 很强；
payload0 pocket 很强；
RANK5 / CERT10 accepted region 很强；
v9.6.4 的 accepted region 看起来很强；
APG/APGT 工程都能跑。
```

但我们逐个查清：

```text
GCERT18 用了未来字段；
payload0 是混杂口袋；
accepted region group unstable；
旧 template 口径可能错；
新 template 口径修正后仍 LDO 不稳；
generator 工程成功不代表动作好。
```

这说明项目科学约束是对的。

---

## 9.2 最大问题：执行还是偏慢

慢的原因是每轮经常只清一个坑：

```text
发现强信号 -> 查是否合法；
合法后 -> 查 group；
group 后 -> 查 payload；
payload 后 -> 查 template；
template 后 -> 查 LDO。
```

以后应该更多并行：

```text
同时查 LDO、bad/null、memory/offdiag、template、runtime；
同时跑 existing-action rank 和 generator stop-rule；
不等一个 runner 结束才发现下一层问题。
```

---

## 9.3 生成器路线要加 stop-rule

从 APG 到 APGT，生成器多轮共同失败。

以后不能继续：

```text
APGT5 不行就 APGT6；
APGT6 不行就 APGT7；
换个名字继续跑 512 个动作。
```

应该设 stop-rule：

```text
如果连续 N 轮 generated actions 的 best V LCB < 0 且 longrisk UCB > 0.5，
则停止小变体，必须重建生成目标。
```

生成目标必须直接包含：

```text
value-preserving；
memory-safe；
offdiag-safe；
cover-stable；
longrisk-veto-effective；
not OOD relative to existing good actions。
```

---

## 9.4 现在不该回去调数据集 acc

Base-Acc Sentinel 一直健康：LQ mean test acc 约 0.6538，AdamWStrongLRGridMLP 约 0.6285。

这说明底座没死。

但这不是 functional update success。

我们不能因为 LQ base 比 MLP sentinel 高，就绕开 functional controller 的困难。终极目标不是在这几个小数据集上打榜，而是证明：

```text
functional update 作为 update rule，能在严格对照和 leave-out 下带来稳定优势。
```

---

# 10. v9.6.5 后建议怎么继续

v9.6.5 后最自然的下一步不是小修 APGT4，也不是调 CERT13 threshold，而是 v9.6.6 这种方向。

## 10.1 第一件事：解释 LDO 为什么掉

目标：查清 R5B / CERT13 为什么 template 稳，但 dataset leave-out 不稳。

要记录：

```text
per-dataset target count；
per-dataset GradeAB precision；
per-dataset V LCB；
per-dataset longrisk / bad / null；
per-dataset memory/offdiag safe rate；
score distribution shift；
accepted actions 在每个 dataset 的来源。
```

判断标准：

```text
如果某个 dataset target density 本身不足：说明目标定义需要 core + expansion；
如果 score shift 明显：说明 ranker 需要 scale-invariant；
如果 memory/offdiag safe 分布不同：说明 veto 需要更稳；
但 official controller 仍不能用 dataset_name branch。
```

可视化：

```text
每个 dataset 的 score histogram；
每个 dataset 的 precision/longrisk bar chart；
accepted action 在 dataset 上的分布；
score vs V / longrisk scatter。
```

---

## 10.2 第二件事：做 core + expansion target

现在的问题是：

```text
T4.1 数量够但 bad/null 高；
T4.2/T4.3 干净但数量不够。
```

所以要把 target 分两层：

```text
core：最干净，数量少，但可信；
expansion：接近 core，用 memory/offdiag/bad/null veto 控住风险。
```

要记录：

```text
core count / coverage；
expansion count / coverage；
core+expansion 的 bad/null/longrisk；
LDO/LSO/LTO drop；
模板覆盖；
per-dataset density。
```

通过标准：

```text
count >= 87；
precision >= 0.75；
V LCB > 0；
longrisk UCB <= 0.05；
bad UCB <= 0.05；
null UCB <= 0.15；
LDO/LSO/LTO drop <= 0.10；
max template share <= 0.25。
```

---

## 10.3 第三件事：把 memory/offdiag 从“线索”变成 veto

v9.6.5 说明 memory/offdiag safe 有 causal signal，但不是直接 generator recipe。

下一步应该把它作为风险 veto：

```text
Accept(action) = high value rank
                 AND not longrisk predicted
                 AND memory safe
                 AND offdiag safe
                 AND bad/null safe
                 AND cost ok
```

要记录：

```text
只加 value rank 的结果；
value rank + longrisk veto；
再加 memory veto；
再加 offdiag veto；
再加 bad/null veto；
每加一个 veto 后的 count / precision / V / risk。
```

这样可以看清：

```text
到底哪个 veto 真有用；
哪个 veto 只是删掉 coverage；
哪个 veto 让 LDO 变稳。
```

---

## 10.4 第四件事：existing-action controller 优先于 generated-action

v9.6.5 后，existing-action route 明显比 generated-action route 更接近成功。

所以优先级应该是：

```text
1. existing-action minimal controller；
2. selected runtime；
3. official leaveout；
4. paired replay；
5. only then short/full。
```

Generated-action route 只作为并行支线，且必须加 stop-rule。

---

## 10.5 第五件事：selected runtime 只能在 controller 选中后测

不能再测没有 controller 的 runtime smoke 然后说 runtime 过。

selected runtime 要记录：

```text
accepted action count；
feature compute time；
certificate compute time；
payload apply time；
extra kernel launch；
step_ratio_q90；
memory_ratio；
zero-candidate skip；
active-step launch count。
```

通过标准：

```text
step_ratio_q90 <= 1.50；
memory_ratio <= 1.05；
no fake/proxy/cpu offload；
selected controller same as decision pass controller。
```

---

# 11. 最终一句话

从 v9.5.3 到 v9.6.5，我们做的事不是简单“调模型”，而是逐步回答：

```text
什么样的额外参数改动，才算真正帮助 KAN 训练？
训练当下能不能合法地认出这种动作？
如果认不出，能不能生成这种动作？
这种动作是否跨数据集、跨分层、跨模板仍然稳定？
它是否不会忘旧知识、不会制造长期风险、不会太慢？
```

现在最准确的状态是：

$$
\boxed{
\text{我们已经有可信 truth base，也能在 existing actions 中找到高质量局部区域；但还没有把它变成 LDO 稳定、bad/null 可控、runtime 可过的 official controller。}
}
$$

因此下一步不该小修阈值，也不该盲目加新生成器，而应该集中做：

```text
1. LDO failure autopsy；
2. core + expansion target；
3. value rank + longrisk/memory/offdiag/bad/null veto；
4. existing-action minimal controller；
5. selected runtime；
6. generated route stop-rule。
```

只有这几步过了，才应该进入 paired replay 和 short/full training。

