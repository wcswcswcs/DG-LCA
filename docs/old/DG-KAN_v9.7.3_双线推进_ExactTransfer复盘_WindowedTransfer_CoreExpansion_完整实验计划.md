# DG-KAN v9.7.3 双线推进：Exact Transfer 复盘、Windowed Transfer、Core Expansion 与 Direct Update 实验计划

> 本计划基于 v9.7.2 的真实执行结果制定。  
> v9.7.2 已经把 exact transfer materializer 真正落地，但 exact transfer 作为单一排序规则没有选出好动作。  
> v9.7.3 不继续小修 `R8A threshold`、`TransferLCB threshold`、`Core+10 expansion` 或 `APGU`。  
> 本轮采用双线验证：一条线继续推进已有动作的选择规则，另一条线验证更少人工设计的 transfer 原理是否能成为新的数学核心。

---

# 0. 先用普通话解释几个名字

为了让没有背景的读者也能看懂，这里先解释本计划里的几个词。

## 0.1 action / 动作

在一次普通 AdamW 训练步之外，我们考虑额外给模型参数加一小块变化。这个额外变化叫一个 action。

也就是：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{AdamW}+\Delta\theta_{func}.
$$

其中 $\Delta\theta_{func}$ 就是我们要不要加的 functional update action。

## 0.2 good action / 好动作

一个好动作不是“让某个 batch 的 acc 短暂上升”。一个好动作至少要满足：

```text
1. 它比 AdamWParallel / bestLR / NoOp / Random 这些对照更好；
2. 它短期 h20 有收益；
3. 它中期 h80 不明显变坏；
4. 它长期 h240 不产生 long-risk；
5. 它不制造 bad / null；
6. 它不破坏 old family / old stratum 这类旧知识；
7. 它不能只在某个数据集、某个模板、某个局部口袋里有效；
8. 它要能在训练当下被识别出来；
9. 它的运行成本不能让系统比 MLP 慢太多。
```

## 0.3 existing-action route / 已有动作路线

我们已经有一个 canonical AP0 action universe，总共 2876 个动作。已有动作路线就是：

```text
不先造新动作；
先问这 2876 个动作里，能不能用训练当下可见的信息挑出好动作。
```

## 0.4 generated-action route / 新生成动作路线

新生成动作路线就是：

```text
不只从旧动作里挑；
直接设计或求解一个新的参数改动。
```

过去很多 APG / APGH / APGL / APGT 类生成器都工程跑通了，但结果长期是 value-negative / high-longrisk，所以目前不能继续盲目加小变体。

## 0.5 transfer / 跨样本转移

一个动作如果只对生成它的样本有帮助，可能是在记噪声。更有价值的是：

```text
用一部分样本提出动作；
用另一部分没有参与生成的样本检查这个动作是否也有帮助。
```

如果对没参与生成的样本也有帮助，这个动作更像走在真实信号方向上。

对检查样本 $i$，一个 action $\Delta$ 的线性化收益可以写成：

$$
r_i(\Delta)=-g_i^\top\Delta.
$$

如果 $r_i(\Delta)>0$，表示这个动作预计让样本 $i$ 的 loss 降低。

我们把检查样本上的平均收益做保守下界：

$$
LCB_{transfer}(\Delta)
=
\mu_B(\Delta)-z\frac{\sigma_B(\Delta)}{\sqrt{|B|}}.
$$

这里 $B$ 是检查样本集合，$\mu_B$ 是平均收益，$\sigma_B$ 是样本间波动，$z$ 是保守系数。

---

# 1. v9.7.2 的一句话判断

v9.7.2 有真实进展，但不是 functional success。

它真正推进的是：

$$
\boxed{
\text{exact transfer 的测量系统已经落地，但 one-step exact transfer 作为单一选动作规则失败。}
}
$$

换成更直白的话：

```text
我们终于能精确测量每个动作对检查样本的即时影响；
而且这个线性测量和真实 apply 非常一致；
但用这个即时影响去挑动作，挑不出我们想要的长期好动作。
```

这不是“小失败”。它说明：

```text
一个动作是否会让检查样本当前 loss 下降，
不等于它就是一个短期有用、中期不坏、长期安全的 functional update。
```

---

# 2. v9.7.2 的关键数据

## 2.1 Exact transfer materializer 已经真正落地

v9.7.1 的问题是 exact artifact 缺失：

```text
exact_linear_transfer_row_count = 0
exact_per_sample_gradient_available = 0
exact_apply_checkpoint_available = 0
```

v9.7.2 把这个问题推进掉了：

```text
AP0 actions = 2876
samples per action = 64
exact linear transfer rows = 184064 / 184064
completion = 1.0
missing gradient / payload = 0 / 0
NaN / Inf / duplicate = 0 / 0 / 0
```

P2 exact apply audit 也很强：

```text
64 actions x 64 samples = 4096 exact apply rows
linear/apply Pearson = 0.9997818368443041
linear/apply Spearman = 0.9978941645960946
sign match = 0.961181640625
MAE = 6.0028745734372246e-05
```

这说明当前 one-step 线性化不是胡乱估计。它和真实 apply 很一致。

## 2.2 但 exact transfer score 没有选出好动作

P3 best exact score 是：

```text
best = T4-core-safe-transfer
TopK87 precision = 0.20689655172413793
V LCB = -0.22226101681921628
longrisk UCB = 0.0
LDO drop = 0.08045977011494251
```

这个结果说明：

```text
exact transfer 能压住 longrisk；
也能让 LDO drop 比旧 rank 小；
但 precision 太低，V LCB 还是负的。
```

也就是说，它选出来的动作更“稳定”，但不是更“有用”。

## 2.3 Core + exact transfer expansion 仍然卡在 LDO

已有 core 77 个动作很干净，但不够 coverage。目标是补到 87 个左右。

v9.7.2 的 best core expansion 是：

```text
best = T2-SNR-transfer
accepted = 87
GradeAB precision = 0.8850574712643678
V LCB = 0.0933837988474282
risk/bad/null/memory/offdiag UCB = 0
LDO drop = 0.39080459770114945
```

这组数很有意思：

```text
动作质量很好；
风险也干净；
但跨数据集不稳。
```

所以 exact transfer 没有解决 v9.6.8 / v9.7.0 以来反复出现的核心问题：LDO，也就是 leave-dataset-out 后掉得太多。

## 2.4 旧 rank 仍然是高质量 diagnostic，但不能 official

旧 R8A / R5B TopK87 仍然很强：

```text
precision = 0.8735632183908046
V LCB = 0.14111334880346277
LDO drop = 0.37931034482758624
```

这说明已有动作里确实有好动作。但旧 rank 的跨数据集稳定性仍然不过。

## 2.5 Proxy transfer 低质量，exact transfer略好但仍不可用

v9.7.0 proxy transfer 是：

```text
TopK87 precision = 0.09195402298850575
V LCB = -0.24289252733948727
LDO drop = 0.04597701149425287
```

v9.7.2 exact best 是：

```text
TopK87 precision = 0.20689655172413793
V LCB = -0.22226101681921628
LDO drop = 0.08045977011494251
```

exact 比 proxy 好一点，但仍远远不够。

---

# 3. 我对 v9.7.2 route 的独立判断

v9.7.2 的 route 是：

```text
R3-ExactTransferNotBetterThanProxy
```

我觉得这个 route 的字面说法有点过强。更准确应该是：

$$
\boxed{
\text{exact one-step transfer materializer 成功；但 exact one-step transfer 不能单独当 controller。}
}
$$

也就是说，v9.7.2 没有证明：

```text
transfer 思路彻底失败。
```

它证明的是：

```text
one-step exact transfer 这个最简单版本失败。
```

这个区别很重要。因为理论上，一个动作是否泛化，不一定由当前一步 $-g_i^\top\Delta$ 完全决定。它可能取决于未来几步的训练轨迹、旧知识区域、局部 cover、长期风险、以及 action 和 AdamW 的后续相互作用。

所以 transfer 不能再作为单一 score，但它可以继续作为一个更大数学框架里的组件。

---

# 4. 为什么 one-step exact transfer 会失败？

我认为至少有五个原因。

## 4.1 它只看当前一步，不看未来训练

one-step exact transfer 只问：

```text
这个动作现在对检查样本有没有降低 loss？
```

但我们的目标问的是：

```text
这个动作在 h20 / h80 / h240 后是否仍然有用且安全？
```

一个 action 可能现在降低 loss，但后面破坏训练轨迹。也可能现在变化不大，但长期保住了几何结构。

## 4.2 它只看 loss，不看 bad / null / memory / offdiag

v9.6.x 的结果反复说明，很多动作失败不是因为即时 loss 估计错，而是因为：

```text
longrisk 被制造出来；
memory / old family 被破坏；
offdiag risk 出现；
null 动作太多；
bad/null 和 coverage 冲突。
```

这些不一定能从 one-step loss transfer 里看出来。

## 4.3 它可能偏向“稳定但没用”的动作

exact transfer best 的 longrisk UCB 是 0，LDO drop 也低，但 precision 和 V 都差。这说明它可能倾向于选：

```text
不太冒险、也不太有收益的动作。
```

这类动作不能作为 functional update，因为 functional update 不是为了“不出事”，而是为了在安全前提下带来真实训练收益。

## 4.4 它没有解决 dataset shift

Core + exact transfer expansion 仍然有 LDO drop `0.3908`。这说明 exact transfer 没有把 Fashion-MNIST / KMNIST / MNIST 之间的 score/support 差异消掉。

## 4.5 它和旧 high-quality rank 信息没有自然合并

旧 rank 质量高但 LDO 差，exact transfer LDO 好但质量差。现在最核心的问题是：

```text
能不能把旧 rank 的 value 信息，和 exact transfer 的稳定性信息结合起来，
而不是让二者互相抵消？
```

这不能靠简单加权。下一轮要做分层实验。

---

# 5. 当前项目状态

## 5.1 已经可信的部分

```text
1. canonical AP0 outcome universe 已经可信；
2. action apply / payload replay / no-transform replay 已经可信；
3. exact per-sample transfer materializer 已经落地；
4. linear transfer 与 exact apply 高度一致；
5. 已有动作中存在一批高质量动作；
6. generated-action blind variants 应继续停止；
7. Base-Acc Sentinel 继续健康；
8. no fake / no proxy / no dataset-specific controller 约束仍守住。
```

## 5.2 仍然没有完成的部分

```text
1. 没有 selected controller；
2. 没有 selected runtime；
3. 没有 system legal controller pass；
4. 没有 official paired replay；
5. 没有 short/full functional training；
6. 没有证明 RealFunctional beats AdamWParallel / bestLR / NoOp / Random；
7. 没有证明 functional update 比 MLP training system 有外部优势。
```

## 5.3 当前真正 blocker

当前 blocker 不是：

```text
exact transfer materializer 缺失；
旧表不可信；
payload replay 不可信；
KAN base 崩；
generated route 需要继续盲跑。
```

当前真正 blocker 是：

$$
\boxed{
\text{已有好动作，但还没有一个跨数据集稳定、训练当下可验证、runtime 可承受的选择规则。}
}
$$

---

# 6. v9.7.3 总体目标

v9.7.3 不再只问：

```text
exact transfer score 能不能过？
```

而是问三个更本质的问题。

## 6.1 问题 A：exact transfer 为什么失败？

我们要把失败拆开：

```text
是因为 one-step 太短？
是因为只看 CE 不够？
是因为 check samples 选错？
是因为 dataset score shift？
是因为 exact transfer 只适合做 veto，不适合做 rank？
是因为 good action 的价值来自 memory/offdiag/cumulative effect，而不是 immediate transfer？
```

## 6.2 问题 B：已有动作路线还能不能过？

已有动作路线目前最接近成功：

```text
core 77 很干净；
旧 rank TopK87 质量高；
exact expansion 能补到 87，但 LDO 不稳。
```

v9.7.3 要专门问：

```text
能不能用 exact transfer / memory/offdiag / dataset-invariant normalization，
只补足那缺的 10 个动作，
同时让 LDO 降下来？
```

## 6.3 问题 C：更少人工设计的新路线是否可行？

我们不再把新路线写成复杂手工 score。先验证一个更简单原则：

```text
一个动作必须能从生成样本 transfer 到检查样本；
如果 one-step 不够，就看 windowed transfer；
如果 existing-action 里能验证，再考虑 direct solved update。
```

---

# 7. v9.7.3 的硬约束

继续遵守：

```text
no teacher
no self-teacher
no distillation as training trick
no loss modification
no label smoothing / focal / margin auxiliary loss
no sampler / class weight
no CPU offload
no fake / proxy rows
KAN path 不使用 PyTorch loss.backward graph
official controller 不使用 dataset_name 分支
official controller 不使用 validation/test/future outcome at commit
old table 不进入 official
proxy transfer 不得冒充 exact transfer
generated route 不得在没有新 objective evidence 时盲跑
```

允许：

```text
exact transfer materializer；
per-example gradient / per-example response；
train-memory buffer；
dataset / stratum / family 作为 diagnostics 和 leaveout axes；
calibration split 训练 legal ranker；
heldout split 冻结评估；
windowed transfer 作为 diagnostic；
如果 windowed transfer 过 gate，再作为 commit-time estimator 的训练目标。
```

---

# 8. v9.7.3 核心假设

## H1：one-step exact transfer 不是无用，而是只能当辅助信号

### 直觉

one-step exact transfer 可能不能单独挑动作，但可以帮助判断：

```text
哪些动作虽然旧 rank 高，但没有跨样本支持；
哪些 expansion 动作虽然 value rank 中等，但更可能跨 dataset 稳定。
```

### 成立标准

在 `core77 + expansion10` 的问题中，exact transfer 能让扩展动作满足：

$$
AcceptedCount \ge 87,
$$

$$
Precision_{GradeAB} \ge 0.85,
$$

$$
LCB(V)>0,
$$

$$
UCB(LongRisk)=0,
$$

$$
LDO\_drop \le 0.15.
$$

### 失败标准

只要所有 exact-assisted expansion 都仍然出现：

```text
LDO drop >= 0.30；
或 V LCB <= 0；
或 GradeAB precision <= 0.75；
```

则 H1 失败。

---

## H2：windowed transfer 比 one-step transfer 更接近真实目标

### 直觉

one-step transfer 只看当前 $-g^\top\Delta$。但好动作可能要经过几步训练后才体现价值。Windowed transfer 看的是：

```text
加动作后，在短窗口内对检查样本的累计影响。
```

### 定义

给定 action $\Delta$，检查样本集合 $B$，窗口长度 $m$，定义：

$$
WT_m(\Delta)
=
\frac{1}{|B|}\sum_{i\in B}
\left[
CE_i(\theta_t)-CE_i(\theta_{t+m}^{+\Delta})
\right].
$$

保守下界：

$$
LCB(WT_m)=\mu(WT_m)-z\frac{\sigma(WT_m)}{\sqrt{|B|}}.
$$

### 成立标准

至少一个窗口 $m\in\{1,5,20\}$ 满足：

```text
TopK87 GradeAB precision >= 0.75；
V LCB > 0；
longrisk UCB <= 0.05；
LDO / LSO / LTO drop <= 0.15。
```

### 失败标准

所有窗口都和 one-step 类似：

```text
precision 低；
V LCB 负；
或只在单一 dataset / group 有效。
```

---

## H3：旧 rank 的 value 信息和 transfer 的稳定性信息必须分层使用

### 直觉

旧 rank 的问题是：

```text
quality 高，LDO 差。
```

exact transfer 的问题是：

```text
LDO 好一点，quality 差。
```

所以不要把它们线性相加。应该用：

```text
旧 rank 负责找 value；
transfer / memory / offdiag 负责过滤不稳定或风险动作。
```

### 成立标准

分层规则能形成：

```text
accepted_count >= 87；
precision >= 0.85；
V LCB > 0.10；
longrisk/bad/null/memory/offdiag UCB 都过；
LDO drop <= 0.15。
```

---

## H4：direct transfer-solved update 只有在 H2 或 H3 成立后才允许打开

### 直觉

如果 transfer 对已有动作都解释不了，就不应该直接用 transfer 求新动作。

### Gate

Direct update 只在下面条件满足时打开：

```text
H2 pass 或 H3 pass；
exact transfer/apply audit pass；
memory/offdiag veto 可用；
runtime preflight 不超过预算。
```

---

# 9. 实验阶段设计

---

# P0. v9.7.2 boundary 复核

## 目标

确认本轮不是回退，也不是读错 artifact。

## 要记录

```text
source_route_v9720
exact_linear_transfer_rows
exact_apply_rows
linear_apply_pearson
linear_apply_spearman
best_exact_score_id
best_exact_precision
best_exact_V_LCB
best_exact_LDO_drop
best_core_expansion_id
best_core_expansion_precision
best_core_expansion_LDO_drop
generated_route_status
Base-Acc Sentinel rows / LQ acc / StrongMLP acc
```

## 通过标准

```text
exact_linear_transfer_rows = 184064；
linear_apply_pearson >= 0.99；
generated_route_status = stopped_no_new_objective；
field legality pass = 1。
```

## 可视化

```text
1. linear transfer vs exact apply scatter；
2. per-dataset linear/apply correlation bar；
3. v9.7.0 proxy vs v9.7.2 exact performance comparison bar。
```

---

# P1. Exact transfer failure autopsy

## 目标

解释为什么 exact transfer 线性测量准确，但选动作失败。

## 分组

把 2876 个 AP0 actions 分成以下集合：

```text
A. Core77：最干净但 coverage 不够的 77 个动作；
B. OldRankTop87：旧 R8A/R5B 选中的 87 个动作；
C. ExactTop87：exact transfer 选中的 87 个动作；
D. ProxyTop87：v9.7.0 proxy transfer 选中的 87 个动作；
E. OldRankTop87 ∩ ExactTop87；
F. OldRankTop87 - ExactTop87；
G. ExactTop87 - OldRankTop87；
H. Random87 negative control。
```

## 要记录

每个集合记录：

```text
count
GradeAB precision
V_integrated LCB
h240 longrisk UCB
bad UCB
null UCB
memory fail UCB
offdiag fail UCB
LDO drop
LSO drop
LTO drop
per-dataset accepted count
per-dataset precision
per-dataset score mean / std
candidate template count
max template share
```

## 判断标准

如果 `OldRankTop87 ∩ ExactTop87` 质量显著高于两者单独集合，说明 exact transfer 可作为辅助过滤。  
如果 `ExactTop87 - OldRankTop87` 质量差，说明 exact transfer 不能独立补 expansion。  
如果 `OldRankTop87 - ExactTop87` 质量仍高，说明 transfer 会误杀好动作。

## 可视化

```text
1. V_integrated vs exact transfer scatter；
2. exact transfer 分位数上的 GradeAB / longrisk 曲线；
3. OldRankTop87 和 ExactTop87 的 Venn 图；
4. per-dataset precision heatmap；
5. failure mode stacked bar。
```

---

# P2. Core77 + expansion10 的最小修复实验

## 目标

不要重新设计所有东西。先回答一个最小问题：

$$
\boxed{
\text{能否在 77 个干净 core 之外，只补 10 个稳定 expansion 动作？}
}
$$

## Core 定义

Core77 使用当前最干净集合：

```text
GradeAB = 1.0；
V LCB > 0；
longrisk/bad/null/memory/offdiag UCB = 0；
count = 77；
coverage = 0.02677。
```

## Expansion 候选来源

候选从以下集合取：

```text
1. OldRankTop150 - Core77；
2. ExactTop150 - Core77；
3. WindowedTransferTop150 - Core77；
4. memory/offdiag safe pool；
5. dataset-balanced calibration pool。
```

## 选择规则

不使用 dataset name 作为 feature。允许在 calibration 中用 dataset 做评估和 reweight，但 final rule 不含 dataset branch。

尝试四种 expansion 策略：

```text
E1: old rank 最高的 10 个；
E2: exact transfer LCB 最高且 risk clean 的 10 个；
E3: old rank 高 + exact transfer 非负的 10 个；
E4: old rank 高 + exact transfer 非负 + memory/offdiag clean 的 10 个。
```

注意：E3/E4 不是线性加权，而是分层过滤。

## 要记录

```text
accepted_count
coverage
GradeAB precision
V LCB
longrisk/bad/null/memory/offdiag UCB
LDO / LSO / LTO drop
per-dataset accepted count
per-dataset precision
expansion10 中每个 action 的 source rank / exact rank / dataset / template / failure reason
```

## 通过标准

强通过：

$$
accepted\_count \ge 87,
$$

$$
Precision_{GradeAB}\ge 0.85,
$$

$$
LCB(V)>0.10,
$$

$$
UCB(LongRisk)=0,
$$

$$
UCB(Bad)=0,
$$

$$
UCB(Null)\le0.15,
$$

$$
LDO\_drop\le0.10.
$$

弱通过：

```text
accepted_count >= 87；
precision >= 0.80；
V LCB > 0；
longrisk = 0；
LDO drop <= 0.15。
```

## 可视化

```text
1. Core77 -> Expansion87 waterfall；
2. 每个 expansion action 的风险雷达图；
3. per-dataset precision before/after expansion；
4. expansion action 的 exact transfer vs old rank scatter。
```

---

# P3. Windowed transfer materializer

## 目标

验证 one-step transfer 失败是否因为窗口太短。

## 实验对象

不先 full universe，先做 fail-fast subset：

```text
Core77：77 actions；
OldRankTop87：87 actions；
ExactTop87：87 actions；
Expansion candidates：约 200 actions；
Random control：87 actions。
```

总数约 400-500 actions。

## 窗口

```text
m = 1, 5, 20
```

其中：

```text
m=1：接近 one-step transfer；
m=5：短窗口；
m=20：接近 h20 早期效应。
```

## 计算方式

对 action $\Delta$，检查样本集合 $B$：

$$
WT_m(\Delta)=\frac{1}{|B|}\sum_{i\in B}
\left[
CE_i(\theta_t)-CE_i(\theta_{t+m}^{+\Delta})
\right].
$$

记录 conservative lower bound：

$$
LCB(WT_m)=\mu_m-z\frac{\sigma_m}{\sqrt{|B|}}.
$$

## 要记录

```text
WT1_mean / LCB / std
WT5_mean / LCB / std
WT20_mean / LCB / std
per-dataset WT
per-family WT
memory-buffer WT
hard-tail WT
actual GradeAB
actual V_integrated
actual longrisk/bad/null
```

## 通过标准

至少一个窗口 score 满足：

```text
TopK87 precision >= 0.75；
V LCB > 0；
longrisk UCB <= 0.05；
bad UCB <= 0.05；
null UCB <= 0.15；
LDO/LSO/LTO <= 0.15。
```

如果 subset 通过，再扩展到 full 2876 actions。  
如果 subset 明确失败，不做 full materializer。

## 可视化

```text
1. one-step transfer vs WT20 scatter；
2. WTm 与 V_integrated 的 correlation by m；
3. WTm TopK precision curve；
4. WTm per-dataset score distribution；
5. WTm 与 longrisk 的 2D density 图。
```

---

# P4. Dataset shift 复盘：不是按数据集调参，而是找无数据集分支的稳定化

## 目标

确认 LDO drop 到底来自：

```text
1. score scale shift；
2. target density shift；
3. support count shift；
4. memory/offdiag safe rate shift；
5. hard-tail distribution shift；
6. expansion action 分布偏移。
```

## 要记录

按 dataset 只做诊断，不做 controller branch：

```text
score mean / std / quantile
GradeAB density
Core77 density
Expansion candidate density
memory/offdiag clean rate
longrisk rate
bad/null rate
old rank score PSI
exact transfer PSI
windowed transfer PSI
```

## 方法

尝试三种不含 dataset name 的稳定化方式：

```text
S1: rank-based threshold，不用 raw score；
S2: calibration split 上的 global conformal cutoff；
S3: group-adversarial training，但 final score 不含 group id。
```

## 通过标准

```text
LDO drop <= 0.10；
且 precision / V / risk 不崩。
```

若 LDO 降低但 precision < 0.75 或 V LCB <= 0，则不是 pass。

## 可视化

```text
1. score distribution by dataset；
2. target density by dataset；
3. score transport before/after；
4. accepted action distribution by dataset；
5. LDO failure waterfall。
```

---

# P5. Existing-action minimal controller

## 目标

如果 P2/P3/P4 中至少有一个 accepted region 过 weak gate，就冻结一个 minimal controller。

## Controller 形式

不使用复杂手工加权，使用分层判断：

```text
1. Value rank 先选候选；
2. longrisk veto；
3. bad/null veto；
4. memory/offdiag veto；
5. transfer non-negative check；
6. cost check。
```

形式：

$$
Accept(a)=1
$$

当且仅当：

$$
Rank_{value}(a)\le K,
$$

$$
RiskVeto(a)=0,
$$

$$
BadNullVeto(a)=0,
$$

$$
MemoryOffdiagVeto(a)=0,
$$

$$
TransferCheck(a)\ge0,
$$

$$
Cost(a)\le C_{max}.
$$

注意：这里没有人工加权和。

## 要记录

```text
controller_id
calibration threshold / K
heldout accepted count
coverage
precision
V LCB
longrisk/bad/null/memory/offdiag UCB
LDO/LSO/LTO drop
feature cost
certificate cost
payload apply expected cost
```

## 通过标准

```text
accepted_count >= 87；
coverage >= 0.03；
precision >= 0.75；
V LCB > 0；
longrisk UCB <= 0.05；
bad UCB <= 0.05；
null UCB <= 0.15；
LDO/LSO/LTO <= 0.10；
no dataset branch；
no outcome-derived fields。
```

---

# P6. Direct transfer-solved update preflight

## Gate

只有当 P3 windowed transfer 或 P5 controller 至少 weak pass 时，才打开。

## 目标

不再盲目 APGU/APGV。直接求一个小更新：

$$
\Delta^*=
\arg\max_{\Delta\in\mathcal{S}}
LCB_{windowed\ transfer}(\Delta)
$$

约束：

$$
\|\Delta\|\le\epsilon,
$$

$$
MemoryHarm(\Delta)\le\tau_M,
$$

$$
OffdiagRisk(\Delta)\le\tau_O,
$$

$$
Cost(\Delta)\le C_{max}.
$$

其中 $\mathcal{S}$ 是一个小的 KAN 参数子空间，例如：

```text
1. 最后一层 edge coefficient；
2. 少数 high-SNR basis block；
3. low-rank edge residual；
4. AdamW-orthogonal small residual。
```

## 阶段

```text
P6a: single-action direct solve preflight；
P6b: 16-action smoke；
P6c: 64-action branch-horizon replay；
P6d: 如果仍 value-negative / high-longrisk，则 direct route stop。
```

## 判断标准

```text
GradeAB precision >= 0.50 at 64-action smoke；
V LCB > 0；
longrisk UCB <= 0.10；
new positive created rate >= 0.20；
longrisk created rate <= 0.10。
```

低于这些标准，不扩展到 full generated route。

---

# P7. Generated route stop-rule enforcement

## 目标

防止继续盲目造新 primitive。

## Stop-rule

如果满足任一条件，generated route 继续停止：

```text
1. exact/windowed transfer 不能解释 existing good actions；
2. direct transfer-solved update P6 不过；
3. 64-action smoke 中 V LCB <= 0；
4. longrisk created rate >= 0.50；
5. negative control 成为 best；
6. no-transform equivalence 不过。
```

## 要记录

```text
generated_route_status
reason
last_three_generated_family_summary
mean GradeAB precision
mean V LCB
mean longrisk created rate
new objective evidence present / absent
```

---

# P8. Selected runtime boundary

## Gate

只有 P5 minimal controller 或 P6 direct update 过 weak pass，才测 selected runtime。

## 要记录

```text
feature compute q50/q90/q99
transfer compute q50/q90/q99
certificate compute q50/q90/q99
payload apply q50/q90/q99
step time q90
step_ratio_q90
memory ratio
kernel launch count
sync count
active event count
```

## 通过标准

$$
step\_ratio_{q90}\le1.50,
$$

$$
memory\_ratio\le1.05.
$$

---

# P9. Official paired replay boundary

## Gate

只有 P5/P6 和 P8 都过，才打开。

## 目标

证明 RealFunctional 不是偶然，而是真的打过对照。

## Branches

```text
RealFunctional
AdamWParallel
bestLR
NoOp
RandomPayload
ShuffledFunctionalPayload
```

## Horizons

```text
h20
h80
h240
```

## 要记录

```text
CE delta
accuracy delta
NLL delta
ECE delta
CEp99 delta
margin_p10 delta
memory buffer CE delta
old family CE delta
RealFunctional beats each control
shuffled payload fail rate
```

## 通过标准

```text
RealFunctional beats AdamWParallel / bestLR / NoOp / Random；
Shuffled payload 不通过；
h20/h80/h240 不出现 longrisk；
LDO/LSO/LTO 保持稳定。
```

---

# P10. Short/full training boundary

## Gate

只有 P9 过线才打开。

## 目标

这一步才看训练系统最终是否有用。

## 记录

```text
train acc / val acc / test acc
train CE / val CE / test CE
NLL
ECE
CEp99
hard-tail accuracy
time-to-target
steps-to-target
sample efficiency
continual retained accuracy
forgetting rate
runtime
memory
```

## 对照

```text
MatchedMLP
AdamWStrongLRGridMLP
QuadraticFeatureMLP
LQ base AdamW-only
No functional update
```

## 注意

这里仍然不能按 dataset 调 controller。dataset 只能作为 final reporting axis。

---

# 10. 并行执行计划

为了避免再一轮只发现一个 blocker，v9.7.3 必须并行。

## Batch A：Existing-action analysis

```text
P1 exact failure autopsy；
P2 core77 + expansion10；
P4 dataset shift diagnostics。
```

## Batch B：Windowed transfer

```text
P3 subset materializer；
P3 transfer window comparison；
P3 full materializer only if subset passes。
```

## Batch C：Controller / runtime boundary

```text
P5 minimal controller；
P8 runtime preflight；
只在 P5 weak pass 后真正运行 P8。
```

## Batch D：Direct update gate

```text
P6 single-action preflight；
P6 16-action smoke；
P6 64-action smoke；
没有 P3/P5 weak pass 则不运行。
```

## Batch E：Base health sentinel

```text
继续复用或轻量重跑 Base-Acc Sentinel；
不用于 controller；
只检查 LQ base 没有 catastrophic fail。
```

---

# 11. Route decision

v9.7.3 最终 route 按下面规则。

## R1: ExactTransferAuxiliaryUseful

条件：

```text
P2 core+expansion 或 P5 controller 通过 weak gate；
exact transfer 作为辅助过滤有正贡献。
```

下一步：进入 selected runtime。

## R2: WindowedTransferUseful

条件：

```text
one-step 失败；
windowed transfer pass。
```

下一步：把 windowed transfer 蒸馏成低成本 commit-time estimator。

## R3: ExistingActionStillLDOBlocked

条件：

```text
quality 高；
LDO 仍然 > 0.15。
```

下一步：专门解决 dataset-invariant rank，而不是继续 expansion。

## R4: TransferPrincipleInsufficient

条件：

```text
exact one-step / windowed transfer 都不能产生可用 accepted region；
也不能辅助 core expansion。
```

下一步：停止 transfer-as-controller 主线，只保留为 diagnostic。

## R5: DirectSolvedUpdatePromising

条件：

```text
P6 direct update smoke 过。
```

下一步：生成路线重开，但只围绕 direct objective，不再盲目 APG variants。

## R6: GeneratedRouteStopped

条件：

```text
P6 不过或 gate 未打开。
```

下一步：继续停止 generated route。

---

# 12. 本轮最终希望得到的判断

v9.7.3 不追求直接 full functional success。它要给出一个明确判断：

```text
1. exact transfer 是否有用？
   是单独有用，辅助有用，还是完全无用？

2. 77 个 core 动作能否补到 87？
   如果能，是否跨数据集稳定？

3. windowed transfer 是否比 one-step 更接近真实好动作？

4. existing-action controller 是否可以打开 runtime？

5. generated route 是否仍应停止？

6. 如果 direct update 可行，是否值得进入 v9.7.4 的真实 branch-horizon scale-up？
```

如果这些问题回答清楚，即使 system 仍不过，也算有效推进。

---

# 13. 本轮最重要的反思

v9.7.2 说明：

```text
测得更精确，不等于规则更正确。
```

exact transfer 的线性估计很准，但它只解释即时 loss，不足以解释长期 functional update 的价值。这要求我们把路线从：

```text
找一个更准的单步分数
```

改成：

```text
验证一个动作是否有跨样本、跨窗口、跨数据集、跨模板的稳定收益。
```

v9.7.3 的双线推进就是为了解决这个问题。

