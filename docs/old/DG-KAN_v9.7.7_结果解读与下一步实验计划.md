# DG-KAN v9.7.6 结果解读与 v9.7.7 下一步实验计划

> 目标读者：不假设读者熟悉 DG-KAN 的所有历史实验。本文会先用直白语言解释我们在做什么，再解释 v9.7.6 到底说明了什么，最后给出下一轮 v9.7.7 的完整实验计划。  
> 公式使用 `$...$` 或 `$$...$$`，Typora 友好。

---

# 1. 先用最简单的话说：我们现在到底在做什么？

DG-KAN 的最终目标不是在 MNIST、Fashion-MNIST、KMNIST 这些小数据集上刷分。我们要做的是一种新的 KAN 训练系统：

```text
普通训练：
  每一步用 AdamW 改参数。

我们想做的训练：
  每一步仍然用 AdamW，
  但在少数合适的时刻，额外加一个小的 functional update。
```

这个额外动作可以写成：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{AdamW}
+
\Delta\theta_{func}.
$$

其中：

```text
θ_t：当前模型参数；
Δθ_AdamW：普通 AdamW 更新；
Δθ_func：我们额外加的 functional update 动作。
```

现在最难的问题不是“能不能生成一个额外动作”，而是：

```text
什么时候加？
加哪个？
怎么保证它不是短期看起来好、长期伤模型？
怎么保证它不是只在某个数据集 / 某个模板 / 某个局部口袋里好？
怎么保证它的运行成本不比 MLP 高太多？
```

所以我们现在的实验一直在问：

$$
\boxed{
\text{能否在训练当下，用合法信息选出一批真正有用、长期安全、跨数据分布稳定的额外参数动作？}
}
$$

这里“合法信息”是指：训练当下能看到的信息，比如当前 batch 的 loss、margin、梯度、动作大小、动作和 AdamW 的关系、memory/offdiag 风险 proxy 等。不能用未来 h20/h80/h240 replay 后才知道的结果，也不能用数据集名字写规则。

---

# 2. v9.7.6 这轮结果一句话总结

v9.7.6 的最准确结论是：

$$
\boxed{
\text{Core77 这批核心好动作本身质量是稳定的；过去看到的 raw LDO 失败，很大一部分是支持度 / 计数惩罚造成的。}
}
$$

但是：

$$
\boxed{
\text{我们仍然没有一个可部署 controller，因为动作数量不够，OldRank 机制解释失败，自然 AP0 动作池扩展也没有落地。}
}
$$

更直白地说：

```text
好消息：
  Core77 不是假好动作。
  三个数据集内部 precision 都是 1.0，V LCB 都为正。
  raw LDO 看起来很差，但 support-adjusted LDO 明显变小。

坏消息：
  Core77 只有 77 个动作，不够 coverage 下限所需约 87 个。
  用 OldOnly 补动作虽然能到 87 个，但 OldRank 的机制仍解释不出来。
  自然 AP0 stream 扩展没跑，所以不知道自然动作池扩大后好动作密度会不会变够。
  controller、runtime、paired replay、short/full training 都没打开。
```

v9.7.6 的 route 是：

```text
R4-OldOnlyMechanismAbsent_OldRankDiagnosticOnly
```

我的解读是：这个 route 对“OldOnly 机制没有解释出来”是对的，但它没有充分表达本轮最大的好消息：**Core77 raw LDO 可能不是质量不稳定，而是 leaveout 评估方式把支持度不足和质量下降混在一起惩罚了。**

---

# 3. v9.7.6 的关键数据怎么读？

## 3.1 Core77 是什么？

`Core77` 指的是当前 AP0 既有动作池里最干净的一批动作，共 77 个。它们满足：

```text
GradeAB precision = 1.0；
V LCB > 0；
longrisk = 0；
bad = 0；
null = 0；
memory/offdiag fail = 0。
```

这里：

```text
GradeAB：我们定义的高质量动作等级；
V LCB：动作真实收益的保守下界；
longrisk：长程风险，尤其 h240 后是否出问题；
bad/null：明确伤害 / 没有效果；
memory/offdiag：旧知识、旧 family、旧 stratum 或非对角风险是否被破坏。
```

v9.7.6 中 Core77 的数据是：

```text
Core-like count = 77
rate = 0.026773296244784424
precision = 1.0
V LCB = 0.16402807605399972
raw LDO = 0.4415584415584416
support-adjusted LDO = 0.09260529551331587
equal-count LDO = 0.0773703535816084
precision-only LDO = 0.0
value-only LDO = 0.09260529551331587
SupportStabilityPass = 1
Core77_LDO_support_artifact = 1
Core77_true_LDO_instability = 0
```

这组数字非常重要。

以前我们看到 raw LDO = 0.4416，会以为 Core77 换数据集后崩了。但 v9.7.6 把 LDO 拆开后发现：

```text
precision-only LDO = 0.0
value-only LDO = 0.0926
support-adjusted LDO = 0.0926
equal-count LDO = 0.0774
```

也就是说：

$$
\boxed{
\text{Core77 在各数据集内部的动作质量并没有明显崩；主要问题是支持度 / 数量分布不均。}
}
$$

这和 v9.7.5 的结果也一致。v9.7.5 里 Core77 三个数据集内部 precision 都是 1.0，但 Fashion-MNIST / KMNIST / MNIST 的 count 是 34 / 28 / 15，V LCB 分别是 0.2186 / 0.0778 / 0.0714。换句话说，三边都好，但数量和收益强度不均。

## 3.2 raw LDO 到底是什么？为什么它会误导？

LDO 是 leave-dataset-out，也就是：留出一个数据集，看规则在剩下 / 留出的数据分布上是否稳定。

raw LDO 问的是：

```text
如果离开某个数据集后，accepted region 的整体 precision / value 掉多少？
```

这个指标有用，但它会把三件事混在一起：

```text
1. 动作质量真的下降；
2. 某个数据集里好动作数量少，导致支持度不足；
3. 为了补满数量，外部低质动作被回填，拉低结果。
```

v9.7.6 证明 Core77 的 raw LDO 被第 2 / 第 3 类因素放大。也就是说，它不是简单的“Core77 质量不稳定”。数学上可以写成：

$$
RawLDO
\approx
QualityDrop
+
SupportPenalty
+
BackfillPenalty.
$$

v9.7.6 的结果显示：

```text
QualityDrop 很小；
SupportPenalty / BackfillPenalty 更大。
```

所以现在不能再简单说：

```text
Core77 LDO fail，所以 Core77 不稳定。
```

更准确应该说：

```text
Core77 质量稳定，但数量不够；当前 raw LDO gate 会把数量不足惩罚成质量失败。
```

## 3.3 Core77 为什么还不能 official？

因为它只有 77 个动作。

目前常见 coverage 下限是：

$$
Coverage \ge 0.03.
$$

当前 AP0 action count 是 2876，所以至少需要：

$$
N_{min}=\lceil 0.03 \times 2876\rceil=87.
$$

Core77 缺口是：

$$
87-77=10.
$$

这个缺口看起来只有 10 个，但过去多轮实验都说明：补这 10 个动作并不容易。因为只要补错，就会引入：

```text
LDO 不稳；
null 过高；
offdiag 风险；
V LCB 降低；
longrisk 或 bad 风险。
```

v9.7.6 中 `S4-Core77PlusOldOnlySupportBalanced` 能接受 87 个动作：

```text
accepted = 87
precision = 0.8850574712643678
V LCB = 0.14120469391749882
support-adjusted LDO = 0.08877827643026047
raw LDO = 0.39080459770114945
LFO = 0.16091954022988508
```

这说明它质量很高，但 raw LDO 和 LFO 仍不过。这里 LFO 是 leave-family-out，也就是离开某些 family 后是否稳定。

因此它仍不能 official。

## 3.4 OldOnly 是什么？为什么重要？

`OldOnly` 是指：旧的 OldRank 规则能选中，但 exact transfer 没选中的动作。

这批动作很重要，因为 v9.7.4 / v9.7.5 显示：

```text
OldOnly precision ≈ 0.855；
V LCB > 0；
不是单一 group pocket；
覆盖多个 dataset / template / family。
```

这说明 OldRank 确实找到了很多好动作。

但 v9.7.6 的 P3 做了 matched contrast 后发现：

```text
matched pair count = 294
matched success rate = 0.8521739130434782
mechanism pass count = 0
best matched feature = WT80_LCB
TopK87 precision = 0.47126436781609193
V LCB = -0.10386482729963209
LDO = 0.13793103448275862
```

也就是说：

```text
我们知道 OldOnly 结果好；
但还不知道训练当下哪个合法特征能解释它为什么好。
```

即便按 effect size 看，`memory_score_inverse / old_family_margin_delta / old_family_probe_loss_inverse` 约有 1.022 的 effect size，但 TopK87 precision 只有 0.4597，V LCB 仍为负。

所以：

$$
\boxed{
\text{OldOnly 仍是强经验现象，不是可部署机制。}
}
$$

## 3.5 OldRank 现在是什么地位？

OldRank 仍然是一个很强的 diagnostic。v9.7.6 P4 中：

```text
best ranker = R3-ValueProxy-HardRiskVeto
precision = 0.8850574712643678
V LCB = 0.14163282358223747
risk/bad/null/memory/offdiag UCB = 0
LDO drop = 0.39080459770114945
```

原始 OldRank 也仍然是：

```text
precision = 0.8735632183908046
V LCB = 0.14111334880346277
LDO = 0.37931034482758624
```

所以 OldRank 的情况是：

```text
它能选出好动作；
但 LDO 不过；
机制解释不出来；
不能 official。
```

因此 v9.7.6 的 route 写成 `OldRankDiagnosticOnly` 是合理的。

---

# 4. 这轮到底有没有进展？

有，而且不是小进展。但它不是“能力进展”，而是“问题分解进展”。

## 4.1 真正的进展 1：Core77 LDO 被重新解释

过去我们以为 Core77 LDO 高说明它跨数据集不稳。v9.7.6 证明：

```text
Core77 的质量跨 dataset 是稳的；
raw LDO 主要被支持度和回填机制放大。
```

这改变了下一步方向。以前我们可能会继续调 ranker；现在更应该问：

```text
如何补足 Core77 的数量？
如何定义一个不把支持度不足误当质量下降的 leaveout gate？
是否需要扩大自然 AP0 action stream？
```

## 4.2 真正的进展 2：OldOnly 机制解释失败被确认

v9.7.6 没有找到能解释 OldOnly 的合法机制。这个负结果也很重要，因为它阻止我们把 OldRank 当成“虽然不懂但能用”的黑箱。

现在 OldRank 只能是：

```text
diagnostic rank;
不能 official;
不能作为新动作生成目标。
```

## 4.3 真正的进展 3：自然 AP0 stream extension 成为核心 blocker

P2 中 natural AP0 stream extension 没跑：

```text
reason = no_landed_natural_AP0_stream_extension_materializer_for_v9760
```

这说明我们还不知道：

```text
Core-like action rate = 0.02677 是当前小池子偶然，
还是自然 AP0 action distribution 的真实密度。
```

如果自然扩展后 Core-like rate 仍低于 0.03，那么 existing-action route 可能天然 coverage 不足。  
如果自然扩展后 Core-like rate 或可补动作密度上升，那么我们应该转向 high-throughput natural AP0 harvesting，而不是继续造人工 APG 变体。

## 4.4 真正的进展 4：generated route 停止仍是正确的

v9.7.6 继续没有新 objective evidence，所以 generated route 保持 stopped。这个判断是对的。

之前 APGH/APGL/APGT/APGF 等生成路线多轮都表现为：

```text
工程闭合；
branch-horizon replay 完整；
但 value 负；
longrisk 高；
new positive 很少。
```

所以现在继续 blind generated variants 只是在堆人工设计。

---

# 5. 为什么你会感觉“非常慢、没进度”？

你的感觉是合理的。因为从用户视角看，系统还没进入真正的能力链路：

```text
selected controller
-> selected runtime
-> paired replay
-> short/full training
```

v9.7.6 中这些都还是：

```text
not_run
```

所以不会看到最终模型 acc、sample efficiency、robustness、continual learning 的提升。

但是从科学诊断角度，v9.7.6 确实有推进。它把问题从：

```text
Core77 LDO 高，是不是质量不稳？
```

推进到：

```text
Core77 质量稳定，但数量不足；raw LDO 混入了支持度惩罚。
```

这很重要，因为下一步不是继续小修 OldRank，而是要重新处理：

```text
support-aware official gate；
natural AP0 action density；
Core77 + expansion 的机制；
OldOnly 机制是否存在。
```

---

# 6. 我对报告结论的独立判断

报告的 route 是：

```text
R4-OldOnlyMechanismAbsent_OldRankDiagnosticOnly
```

我认为它对 OldRank 这条线是对的，但没有把本轮最大的积极发现放到前面。

我会改写成：

$$
\boxed{
\text{Core77 真质量跨数据集稳定，但密度不足；OldOnly 仍不可解释；下一步必须做自然动作密度与 support-aware controller。}
}
$$

也就是说，现在主 blocker 不是以前那种：

```text
truth base 不可信；
old table 有问题；
action apply 不可信；
generator 没实现；
LDO 全面崩；
transfer artifact 缺失。
```

这些都不是当前主因。

当前主因是：

```text
1. Core77 数量不足；
2. raw LDO 混合了质量和支持度；
3. OldOnly 机制解释失败；
4. 自然 AP0 stream extension 没落地；
5. generated route 没有新目标。
```

---

# 7. 数学上的 insight

## 7.1 这是“动作质量”和“动作密度”的分离问题

以前我们把一个规则是否好看成：

$$
RuleGood \approx Precision + V + LongRisk + LDO.
$$

v9.7.6 告诉我们：这个写法太粗。应该拆成：

$$
RuleGood
=
Quality
\times
Support
\times
Stability.
$$

其中：

```text
Quality：选中的动作本身是不是好；
Support：每个数据分布里有没有足够多这种动作；
Stability：换数据集、换 family、换 template 后，规则是否仍然成立。
```

Core77 的情况是：

```text
Quality 高；
Stability 通过 support-adjusted / precision-only 检查；
Support 不足。
```

所以它不是一个坏规则，而是一个**稀疏好规则**。

## 7.2 raw LDO 不应该独自决定路线死亡

raw LDO 可以写成：

$$
LDO_{raw}
=
LDO_{quality}
+
LDO_{support}
+
LDO_{backfill}.
$$

v9.7.6 显示：

```text
LDO_quality 很小；
LDO_support / LDO_backfill 很大。
```

这意味着：

$$
\boxed{
\text{如果只看 raw LDO，我们会误杀一个质量稳定但数量不足的规则。}
}
$$

但反过来，也不能直接说 support-adjusted LDO 过了就 official。因为最终训练系统还是需要足够 coverage。  
所以正确态度是：

```text
raw LDO 用来提醒 coverage/support 问题；
support-adjusted LDO 用来判断质量是否真的跨分布不稳；
两者都要记录，但含义不同。
```

## 7.3 当前问题可能不是“选不出好动作”，而是“当前动作分布里好动作密度略低”

当前 Core-like rate 是：

$$
p_{core}=\frac{77}{2876}=0.02677.
$$

coverage 下限是：

$$
0.03.
$$

差距是：

$$
0.03 - 0.02677 = 0.00323.
$$

这不是巨大差距，但它很关键。它提示：

```text
如果自然动作池稍微更丰富，也许就能到 0.03；
如果自然动作池扩大后 rate 仍固定在 0.0267，existing-action route 可能天然过不了 coverage gate；
如果能找到少量安全 expansion，existing-action route 也可能过。
```

所以 v9.7.7 必须做自然 AP0 stream extension，而不是继续只在 2876 个动作里挤。

## 7.4 OldRank 仍然是现象，不是理论

OldRank / OldOnly 很强，但机制解释失败。数学上，这说明我们还没有找到一个训练当下的充分统计量 $Z$，使得：

$$
Y \perp D \mid Z.
$$

直白说：如果 $Z$ 真能解释动作为什么好，那么知道 dataset $D$ 就不该再明显改变这个动作是否好。  
现在 OldRank 仍然高质量但 LDO 过不了，说明 $Z$ 还不够。

---

# 8. 当前是否还在正确道路上？

高层方向仍然对，因为我们没有：

```text
按 dataset 调规则；
把 diagnostic 写成 official；
把 OldRank 黑箱直接上线；
把 support-adjusted LDO 直接当 official pass；
盲目恢复 APG/APGU generated route；
在 controller 不过时打开 runtime / paired replay / short-full。
```

但是具体执行必须加速，并且要停止低效循环。

下一步不应该继续：

```text
调 OldRank threshold；
调 WT80_LCB；
调 Core77 + OldOnly 的 topK 数量；
调 support-aware LDO 定义直到 pass；
新增 APG/APGU 小变体；
只看 raw LDO；
只看 support-adjusted LDO；
继续没有 natural stream extension 的完整 runner。
```

正确路线应该是：

```text
1. 先把 LDO 指标拆干净；
2. 再测自然 AP0 action density；
3. 再判断 Core77 是低密度好规则，还是当前动作池太小；
4. 再尝试最小 expansion，而不是大规模规则堆叠；
5. 只有 accepted region 真的过线，才打开 runtime。
```

---

# 9. 离终极目标还差多远？

离 **system-legal local functional controller** 还差三道门：

```text
1. accepted_count >= 87，且 precision / V / longrisk / bad / null / memory / offdiag 同时过线；
2. LDO / LSO / LTO / LFO 稳定；
3. selected runtime step_ratio_q90 <= 1.50。
```

离 **strict PureKAN functional causal evidence** 还要：

```text
official paired replay；
RealFunctional beats AdamWParallel / bestLR / NoOp / Random；
shuffled controls fail；
leave-dataset-out / leave-stratum-out / leave-template-out / leave-family-out。
```

离 **external-ready Beyond-MLP** 还要：

```text
short/full training；
sample efficiency；
calibration；
robustness；
continual / anti-forgetting；
strong baseline 排除；
external reproducibility。
```

所以现在不是快成功了。最准确状态是：

$$
\boxed{
\text{已有高质量动作；Core77 质量跨数据集稳定；但动作密度不足，OldRank 机制不明，controller 尚未可部署。}
}
$$

---

# 10. v9.7.7 下一步实验计划

v9.7.7 不再继续小修 OldRank、WT80、Core77+OldOnly 阈值，也不恢复 blind generated route。

v9.7.7 的核心目标是：

$$
\boxed{
\text{判断 Core77 是一个可扩展的稀疏好规则，还是当前 AP0 动作分布下不可部署的低密度现象。}
}
$$

并且并行回答：

```text
1. raw LDO 到底应该如何拆成质量、支持度、回填惩罚？
2. 自然 AP0 action stream 扩大后，Core-like 好动作密度是否上升或稳定？
3. Core77 是否能用少量合法 expansion 补到 87 并保持 LDO / LFO / risk 过线？
4. OldOnly 是否有可解释机制？如果没有，是否必须停止 OldRank 黑箱路线？
5. generated route 是否仍应停止？
```

---

# 11. v9.7.7 实验总结构

v9.7.7 分为 8 个阶段。

```text
P0：复现 v9.7.6 边界与 artifact audit
P1：LDO 指标分解与 official gate 风险审计
P2：自然 AP0 action stream extension materializer
P3：Core-like / GradeAB / ValuePositiveNoLongRisk 密度曲线
P4：Core77 + expansion 最小补齐实验
P5：OldOnly 机制二次诊断，不再黑箱 promotion
P6：support-aware accepted region 和 raw official gate 对照
P7：controller/runtime boundary，只在 P6 过线后打开
P8：generated route stop / reopen decision
```

---

# P0. 复现 v9.7.6 边界

## 目标

确认 v9.7.6 的关键结论可以复现：

```text
Core77 raw LDO 高；
Core77 support-adjusted LDO 低；
Core77 precision-only LDO = 0；
OldOnly mechanism pass count = 0；
OldRank remains diagnostic；
generated route stopped。
```

## 必须记录

```text
core77_count
core77_precision
core77_V_LCB
core77_raw_LDO
core77_support_adjusted_LDO
core77_equal_count_LDO
core77_precision_only_LDO
core77_value_only_LDO
support_stability_pass
oldonly_matched_pair_count
oldonly_mechanism_pass_count
best_oldonly_feature
best_oldonly_feature_precision
best_oldonly_feature_V_LCB
generated_route_status
base_acc_sentinel_rows
base_acc_LQ_mean_test_acc
base_acc_StrongMLP_mean_test_acc
```

## 成立标准

P0 必须满足：

$$
Core77Precision=1.0,
$$

$$
LCB(V_{Core77})>0,
$$

$$
LDO_{support-adjusted}\le0.10,
$$

$$
OldOnlyMechanismPassCount=0.
$$

如果 P0 不满足，说明 v9.7.6 边界不稳定，v9.7.7 不能继续推进。

## 可视化

```text
1. Core77 per-dataset count / precision / V_LCB bar chart；
2. raw LDO vs support-adjusted LDO vs equal-count LDO 对比图；
3. OldOnly vs ExactOnly 的 matched feature effect size plot；
4. generated-route stop-rule 历史趋势图。
```

---

# P1. LDO 指标分解与 gate 审计

## 背景

v9.7.6 说明 raw LDO 把质量下降和支持度不足混在一起。P1 要把这件事正式写清楚。

## 假设

$$
H1:
LDO_{raw}
\text{ 主要由 support/backfill 造成，而不是 Core77 动作质量不稳定。}
$$

## 方法

对每个 leaveout split，分别计算：

```text
1. raw LDO：原始 official-style drop；
2. precision-only LDO：只看各 dataset 内已接受动作的 precision；
3. value-only LDO：只看各 dataset 内已接受动作的 V_LCB；
4. support-adjusted LDO：固定动作质量，调整支持度惩罚；
5. equal-count LDO：每个 dataset 抽相同数量 accepted 动作比较；
6. backfill LDO：为了补足 accepted_count 引入外部动作后的质量下降。
```

可写成：

$$
LDO_{raw}
=
LDO_{quality}
+
LDO_{support}
+
LDO_{backfill}.
$$

## 必须记录

```text
ldo_raw
ldo_quality
ldo_support
ldo_backfill
ldo_precision_only
ldo_value_only
ldo_support_adjusted
ldo_equal_count
per_dataset_accepted_count
per_dataset_precision
per_dataset_V_LCB
per_dataset_support_rate
per_dataset_backfill_count
```

## 判断标准

H1 成立要求：

$$
LDO_{precision-only}\le0.05,
$$

$$
LDO_{value-only}\le0.10,
$$

$$
LDO_{support-adjusted}\le0.10,
$$

且：

$$
LDO_{support}+LDO_{backfill}
>
LDO_{quality}.
$$

若成立，则下一步应把 raw LDO 视为“support/density blocker”，不是“quality blocker”。

若不成立，说明 Core77 确实跨数据集质量不稳，existing-action route 要降级。

## 可视化

```text
1. LDO decomposition stacked bar；
2. per-dataset support vs V_LCB scatter；
3. backfill_count vs precision drop scatter；
4. Core77 quality confidence interval forest plot。
```

---

# P2. 自然 AP0 action stream extension materializer

## 背景

v9.7.6 最大缺口之一是：自然 AP0 stream extension 没跑。

现在 Core-like rate 是：

$$
p_{core}=77/2876=0.02677.
$$

这离 $0.03$ 很近。必须判断这是当前动作池太小，还是自然动作分布本身密度低。

## 目标

不造新 APG primitive，不做人工生成器，只扩大自然 AP0 candidate/action stream。

## 实验设计

以同样规则、同样数据集、同样 no-dataset-tuning 约束，扩展自然 AP0 action stream：

```text
Panel A：当前 2876 actions，作为 baseline；
Panel B：约 5000 natural AP0 actions；
Panel C：约 10000 natural AP0 actions；
Panel D：约 20000 natural AP0 actions，若算力允许。
```

如果完整 branch-horizon replay 成本太高，采用分阶段：

```text
Stage 1：只 materialize commit-time legal fields + action identity；
Stage 2：对候选 core-like proxy subset 做 branch-horizon replay；
Stage 3：对完整 random stratified subset 做 outcome 校准；
Stage 4：只有通过 preflight 才扩大到全量 outcome。
```

## 必须记录

```text
action_count
event_count
candidate_count
candidate_per_event
action_per_candidate
core_like_count
core_like_rate
GradeAB_count
GradeAB_rate
ValuePositiveNoLongRisk_count
ValuePositiveNoLongRisk_rate
MemoryOffdiagCore_count
MemoryOffdiagCore_rate
per_dataset_core_like_rate
per_template_core_like_rate
per_family_core_like_rate
per_step_bucket_core_like_rate
branch_horizon_rows_materialized
rows_per_sec
unresolved_exception_count
no_fake_no_proxy_pass
```

## 判断标准

### 情况 A：自然扩展支持 existing-action route

若：

$$
p_{core}\ge0.03
$$

且 per-dataset rate 的最小值不明显低：

$$
\min_d p_{core,d}\ge0.02,
$$

则 existing-action route 继续推进。

### 情况 B：自然扩展说明密度不足

若：

$$
p_{core}<0.03
$$

并且随着 action count 增长，置信区间收敛到低于 0.03，则说明当前自然 AP0 source 本身密度不足。

这时不能继续只修 selector，要回到 action source / update rule 设计。

### 情况 C：密度总够但某些 dataset 低

若整体：

$$
p_{core}\ge0.03
$$

但某些 dataset：

$$
p_{core,d}\ll0.03,
$$

则要做 dataset-blind support balancing，不能按 dataset 调参。

## 可视化

```text
1. core-like rate vs action_count 曲线；
2. GradeAB rate vs action_count 曲线；
3. per-dataset density forest plot；
4. per-family density heatmap；
5. action_per_candidate / candidate_per_event 分布图；
6. materializer throughput and memory timeline。
```

---

# P3. Core-like 密度曲线与 coverage 下界

## 目标

用统计方法判断：当前 77/2876 是否只是有限样本偏差。

## 方法

对每个 panel 做 Wilson interval / bootstrap interval：

$$
\hat p_{core}=\frac{N_{core}}{N_{action}}.
$$

记录：

$$
LCB(p_{core}),\quad UCB(p_{core}).
$$

## 必须记录

```text
p_core_mean
p_core_lcb
p_core_ucb
p_gradeab_mean/lcb/ucb
p_value_no_longrisk_mean/lcb/ucb
p_clean_expansion_mean/lcb/ucb
estimated_actions_needed_for_87_core
estimated_actions_needed_for_0.03_density
```

## 判断标准

如果：

$$
LCB(p_{core})\ge0.03,
$$

则 core density 过线。

如果：

$$
UCB(p_{core})<0.03,
$$

则 core density 不足。

如果 interval 跨 0.03，则需要更多 natural stream 或分层分析。

## 可视化

```text
1. p_core with CI over panel size；
2. cumulative core count curve；
3. expected vs observed core count；
4. density gap to 0.03 threshold plot。
```

---

# P4. Core77 + expansion 最小补齐实验

## 背景

当前 Core77 差 10 个动作。P4 只做最小补齐，不再扫大量复杂阈值。

## 候选 expansion 来源

```text
E0：OldOnly top10；
E1：support-balanced OldOnly top10；
E2：natural stream 新发现 core-like top10；
E3：memory/offdiag-safe value-positive top10；
E4：dataset-blind score-normalized top10；
E5：horizon-safe low-null top10。
```

所有 expansion 都不能使用 dataset_name 作为规则。

## 必须记录

```text
accepted_count
core_count
expansion_count
expansion_source
GradeAB_precision
V_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
raw_LDO
support_adjusted_LDO
equal_count_LDO
LFO
LSO
LTO
template_count
family_count
per_dataset_count
per_dataset_precision
per_dataset_V_LCB
```

## 判断标准

P4 strong pass：

$$
N_{accept}\ge87,
$$

$$
Precision_{GradeAB}\ge0.75,
$$

$$
LCB(V)>0,
$$

$$
UCB(LongRisk)\le0.05,
$$

$$
UCB(Bad)\le0.05,
$$

$$
UCB(Null)\le0.15,
$$

$$
UCB(MemoryFail)\le0.05,
$$

$$
UCB(OffdiagFail)\le0.05,
$$

$$
LDO_{support-adjusted}\le0.10,
$$

$$
LFO\le0.10,
$$

$$
LSO\le0.10,
$$

$$
LTO\le0.10.
$$

如果 raw LDO 仍高，但 support-adjusted LDO 过线，必须 route 到：

```text
R_support_gate_definition_conflict
```

不能直接写 official pass。

## 可视化

```text
1. Core vs expansion 的 risk/value 分布；
2. 每个 expansion source 的 pass/fail 雷达图；
3. expansion 动作在 dataset / family / template 上的覆盖图；
4. raw vs support-adjusted LDO 对比图。
```

---

# P5. OldOnly 机制二次诊断

## 背景

v9.7.6 中 OldOnly mechanism pass count = 0。P5 不再试图把 OldRank 直接 official，而是做一个更小、更硬的机制诊断。

## 目标

判断 OldOnly 好动作是否可以由少数合法 commit-time 字段解释。

## 对照组

```text
Positive：OldOnly good actions；
Negative 1：ExactOnly bad actions；
Negative 2：OldRank near-miss actions；
Negative 3：same dataset / same family / same template matched bad actions。
```

## 只允许使用的字段

```text
current batch CE / margin / NLL；
action norm / linf / sparsity；
action-AdamW cosine；
old-family probe loss；
memory score；
offdiag proxy；
hard-tail fraction；
cover entropy proxy；
WT20 / WT80 diagnostic，若为 commit-time measurable；
exact transfer / horizon response 仅作 diagnostic，不直接 official。
```

## 必须记录

```text
matched_pair_count
matched_success_rate
feature_effect_size
feature_auc
feature_topK87_precision
feature_topK87_V_LCB
feature_LDO
feature_LFO
feature_LSO
minimal_feature_set_size
sparse_model_coefficients
mechanism_pass_count
```

## 判断标准

机制成立要求至少一个 legal feature set 满足：

$$
TopK87Precision\ge0.75,
$$

$$
LCB(V)>0,
$$

$$
UCB(LongRisk)\le0.05,
$$

$$
LDO_{support-adjusted}\le0.10,
$$

并且 feature set size 不超过 5。

如果没有任何 feature set 过，则 OldRank 继续保持 diagnostic-only，不允许 controller promotion。

## 可视化

```text
1. OldOnly vs ExactOnly matched feature violin plot；
2. feature effect size bar chart；
3. sparse model coefficients plot；
4. OldOnly mechanism pass/fail table。
```

---

# P6. Support-aware accepted region 与 raw official gate 对照

## 目标

不是强行改 official gate，而是同时报告两种评估：

```text
Legacy raw gate：原始 gate；
Support-aware diagnostic gate：拆掉支持度惩罚后的质量稳定性 gate。
```

## 必须记录

```text
legacy_raw_official_pass
support_aware_diagnostic_pass
raw_LDO
support_adjusted_LDO
equal_count_LDO
precision_only_LDO
value_only_LDO
accepted_count
coverage
risk metrics
family/template/dataset support
```

## 判断标准

如果：

```text
support-aware pass = 1
raw official pass = 0
```

则 route 不能写 system pass，必须写：

```text
R_support_density_gate_conflict
```

然后由 P2/P3 决定是否通过扩大自然 stream 解决 density。

---

# P7. Existing-action minimal controller boundary

只有当 P4/P6 满足以下条件时才运行：

```text
accepted_count >= 87；
precision / V / risk / bad / null / memory / offdiag 过线；
support-adjusted LDO / LFO / LSO / LTO 过线；
raw gate either passes or route explicitly marks support-gate conflict。
```

## Controller 形式

最小形式：

```text
CoreRule(action) OR ExpansionRule(action)
```

其中：

```text
CoreRule：严格干净规则；
ExpansionRule：只允许从 P4 通过的 expansion source 中补足。
```

不允许：

```text
dataset_name branch；
future outcome field；
old table field；
validation/test field；
outcome-derived score；
manual per-dataset threshold。
```

## Runtime 指标

```text
feature_compute_q90_ms
certificate_compute_q90_ms
payload_apply_q90_ms
controller_step_ratio_q90
memory_ratio
kernel_count
sync_count
selected_step_count
active_step_count
```

## Runtime pass 标准

$$
StepRatio_{q90}\le1.50,
$$

$$
MemoryRatio\le1.05.
$$

---

# P8. Generated route stop / reopen decision

## 当前默认状态

```text
generated route = stopped
```

原因：多轮 APG / APGH / APGL / APGT 等都工程闭合但科学失败。

## 重新打开条件

只有满足以下任一条件，才允许新 generated primitive：

```text
1. P2/P3 证明自然 AP0 clean density < 0.03，且 existing-action route 确认动作源不足；
2. P5 找到 OldOnly 的明确 legal mechanism，可转成生成目标；
3. P4 找到 expansion 成功机制，可转成生成目标；
4. Horizon-propagated effect 出现可计算、可验证的新目标。
```

否则：

```text
APG/APGU/APGV/APGW/APGX/APGY/APGZ 全部继续不跑。
```

---

# 12. v9.7.7 预期 route

可能 route 包括：

```text
R1-Core77QualityStableSupportLimited
  Core77 质量稳定，但 coverage/support 不足。

R2-NaturalStreamDensitySufficient
  自然 AP0 stream 扩展后 clean density 达到 0.03，可继续 existing-action controller。

R3-NaturalStreamDensityInsufficient
  自然 AP0 stream 扩展后 clean density 仍不足，需要换 action source。

R4-CoreExpansionPassButRawGateConflict
  Core+Expansion 质量过线，但 raw LDO 与 support-aware LDO 冲突。

R5-OldOnlyMechanismFound
  OldOnly 可由少数合法字段解释。

R6-OldOnlyMechanismAbsent
  OldOnly 仍只是 diagnostic，不能 official。

R7-ExistingActionControllerPass
  existing-action controller 真正过线，可打开 selected runtime。

R8-GeneratedRouteStillStopped
  没有新 objective evidence，generated route 继续停止。
```

---

# 13. 最终判断

v9.7.6 之后，我们不应该把项目判断成“没有进展”。更准确是：

$$
\boxed{
\text{我们已经知道 Core77 是高质量且质量稳定的动作集合；现在真正缺的是数量、机制解释和可部署 gate。}
}
$$

下一步必须避免两个错误：

```text
错误 1：因为 raw LDO 高，就杀掉 Core77 route。
错误 2：因为 support-adjusted LDO 低，就直接把 Core77 route 写成 official pass。
```

正确做法是：

```text
先拆 LDO；
再测自然 AP0 action density；
再做最小 Core+Expansion；
再决定是否能 controller；
generated route 没有新目标前继续停止。
```

如果 v9.7.7 能证明自然 AP0 stream 中 clean density 足够，或者能找到稳定的 +10 expansion，那么 existing-action route 仍然有希望。  
如果自然 stream 密度仍低、OldOnly 机制仍找不到、Core+Expansion 仍不能过线，那就应该停止 existing-action route，回到更根本的 functional update rule 设计。
