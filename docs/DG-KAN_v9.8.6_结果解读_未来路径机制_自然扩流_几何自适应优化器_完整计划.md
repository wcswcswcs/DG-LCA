# DG-KAN v9.8.5 结果解读与 v9.8.6 四线并行实验计划

> 目标读者：不假设已经熟悉前面所有版本。本文尽量把每个名字都解释清楚。  
> 公式使用 `$...$` 或 `$$...$$`，Typora 友好。  
> 本计划不针对 MNIST / Fashion-MNIST / KMNIST 调参。数据集只用于诊断、压力测试和 leave-out，不作为 controller 分支条件。

---

## 0. 一句话结论

v9.8.5 不是能力成功，但不是空转。它把四条线的状态压得很清楚：

```text
A 线：未来训练路径现象仍然成立；好动作确实能让后续训练走到更好路径。
B 线：当前训练当下可用的 virtual probe 失败；它能压风险，但选不出正收益动作，而且成本极高。
C 线：自然 AP0 动作扩流仍未落地；因此无法判断好动作密度到底够不够。
D 线：生成新动作仍不应该重开；因为没有合法机制，也没有自然扩流密度证据。
```

现在最准确状态是：

$$
\boxed{
\text{我们看得到好路径，但还不会在训练当下合法地识别或生成这种路径。}
}
$$

这不是“再调一个分数”能解决的问题。下一步必须同时解决两个硬问题：

```text
1. 好路径的机制：为什么 Core77 / CoreExpansion10 这些动作会让未来训练变好？
2. 好动作的密度：自然动作流扩大后，足够多的好动作是否自然出现？
```

如果这两个问题都不能解决，当前 existing-action route 就不应该继续被包装成即将成功的 controller。

---

## 1. 先解释四条线是什么意思

### A 线：未来训练路径

这里不是看动作加上去之后当前一步 loss 是否下降，而是看它加上去后，后续训练 $1,5,20,80,240$ 步是否走到更好的状态。

一个动作 $a$ 的未来路径可以看成：

$$
Path(a)=\{V_1(a),V_5(a),V_{20}(a),V_{80}(a),V_{240}(a)\}.
$$

其中 $V_h(a)$ 表示：加上这个 functional update 后，经过 $h$ 步训练，相比 AdamW / bestLR / NoOp / Random 等对照是否更好。

如果一个动作 $V_1$ 不强，但 $V_{20},V_{80},V_{240}$ 强，那么它不是当前一步降 loss 的动作，而是改变后续训练路线的动作。

### B 线：训练当下能不能合法识别好路径

A 线可以事后告诉我们哪些动作好，但训练时不能偷看未来结果。B 线问的是：训练当下能不能用合法信息提前判断哪个动作会走出好路径。

合法信息包括：

```text
当前 batch 的输出、梯度、margin、loss；
train-memory buffer 上的当前响应；
hard-tail 当前响应；
action 与 AdamW 的关系；
动作大小、cost、是否伤 memory / offdiag 的当前 proxy。
```

不合法信息包括：

```text
未来 h20/h80/h240 outcome；
V_integrated；
GradeAB label；
old table outcome-derived rank；
dataset_name 分支规则；
validation/test metric at commit time。
```

### C 线：自然动作扩流

现在 canonical AP0 里只有 `2876` 个动作。Core-like 好动作只有 `77` 个，rate 是 `0.02677`，而常用 coverage 下限 $0.03$ 对应大约 `87` 个动作。

所以 C 线问：如果我们不是造新 primitive，而是继续自然采更多 AP0 actions，比如 `5000 / 10000 / 20000` 个，干净好动作比例会不会稳定超过 $0.03$？

如果会，那么 current action source 还有继续价值。  
如果不会，那么靠 existing-action selection 挤来挤去没有意义，必须换 update generation 方式。

### D 线：生成新动作

D 线不是继续盲目造 APG/APGU/APGX 小变体。它只在 A/B/C 给出明确机制后才允许重开。

生成新动作必须有清楚目标，例如：

```text
生成一个能改善未来路径、保 memory/offdiag、不产生 long-risk、runtime 成本可控的小更新。
```

没有这个机制，继续 generated route 只会重复过去的失败：工程能跑，payload 合法，但 value-negative / high-longrisk。

---

## 2. v9.8.5 的独立判断

v9.8.5 的 route 是：

```text
Case4-LegalProxyFailNaturalStreamMissingGeneratedStopped
primary_blocker = legal_virtual_probe_failed
secondary_blocker = natural_stream_materializer_missing
system_legal_controller_pass = 0
generated_route_status = stopped_B_C_D_conditions_not_met
```

这个 route 基本合理，但我认为还应该更深一层表达为：

$$
\boxed{
\text{未来路径现象可信；失败的是训练当下合法识别和自然动作扩流。}
}
$$

也就是说，v9.8.5 不能被解读成“未来训练轨迹思路失败”。相反，它继续支持 A 线。真正失败的是 B/C/D：

```text
B：legal virtual probe 质量不够，且成本极高；
C：natural stream materializer 仍缺；
D：因此 generated route 不能重开。
```

---

## 3. 四线详细进展

## 3.1 A 线：未来路径现象仍然强，但还不是 controller

v9.8.5 读取了真实 landed future rows：

```text
realfunctional rows = 1695
actions = 339
A mechanism / strong / controller = 1 / 0 / 0
```

各组结果如下。

| group | 含义 | actions | AUV LCB | RiskAdjustedAUV LCB | DelayedGain LCB | V80 LCB | V240 LCB | RiskPath UCB | mem/off UCB | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Core77 | 过去最干净的核心好动作 | 77 | 2.8297 | 2.8166 | 4.5339 | 3.1200 | 4.7509 | 0.0383 | 0.0 | 很强，但数量只有 77，raw LDO 高 |
| CoreExpansion10 | 为补 coverage 加的 10 个动作 | 10 | 0.6225 | 0.6225 | 1.6428 | 0.5669 | 1.6236 | 0.0 | 0.0 | 小但干净，值得和 Core77 合并分析 |
| ExactOnly | exact transfer 喜欢但旧 rank 不喜欢的动作 | 68 | 1.5250 | 1.5250 | 2.7311 | 1.5824 | 2.9221 | 0.0 | 0.0 | 路径指标不差，但过去 outcome 质量解释冲突，需要重审定义 |
| OldOnly | 旧 rank 喜欢但 exact transfer 不喜欢的动作 | 10 | 1.0287 | 1.0287 | 2.1502 | 1.1094 | 2.0678 | 0.0 | 0.0 | 继续支持“非 immediate descent” |
| RandomMatched | 匹配随机动作 | 87 | 1.8285 | 0.1763 | 3.3821 | 1.8306 | 3.5923 | 0.8485 | 1.9003 | 大 AUV 但高风险，证明 AUV 不能单独用 |
| RiskCleanButLowValue | 风险干净但过去 value 不高的一组 | 87 | 2.1603 | 2.1343 | 4.0891 | 2.2688 | 4.2696 | 0.0545 | 0.0 | 很值得研究，说明“低风险路径”可能被低估 |

### A 线 insight

第一，Core77 仍然是强证据。它的 $V_{80}$ 和 $V_{240}$ 都很强，risk 很低。这说明好动作确实可能是未来训练路径改善，而不是当前一步 loss 下降。

第二，RandomMatched 的 AUV 也不低，但风险极高。这说明不能只看未来路径面积，也要同时看 risk path、memory/offdiag、bad/null。

第三，RiskCleanButLowValue 值得重审。它过去名字叫 low value，但这轮 AUV 和 delayed gain 都很强，longrisk 只略高于最严格门槛。这可能说明以前的 value 定义漏掉了一类“慢热但安全”的动作。

A 线失败原因不是没有 future-path 信号，而是没有任何单一 path type 同时满足：

```text
count 足够；
value 足够；
risk 足够低；
LDO/LSO/LTO 稳定；
可由训练当下合法机制识别。
```

## 3.2 B 线：legal virtual probe 失败，而且失败很硬

v9.8.5 真正评估了 legal virtual probes，不再只是空缺。结果如下。

| probe | precision | V LCB | AUV LCB | longrisk UCB | memory/offdiag UCB | cost q90 ms | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| B1 current response | 0.2874 | -0.5579 | 3.8598 | 0.3183 | 0.6626 | 154.58 | 质量低、风险高、成本高 |
| B2 virtual 1-step | 0.2989 | -0.5618 | 3.8630 | 0.3442 | 0.7141 | 153.80 | 质量低、风险高、成本高 |
| B3 virtual 5-step | 0.3103 | -0.5833 | 4.2975 | 0.3442 | 0.7649 | 154.58 | AUV 高但 value 负、风险高 |
| B4 hard-tail memory veto | 0.4253 | -0.2362 | 3.5253 | 0.0 | 0.0 | 155.13 | 风险压住了，但 value 仍负 |
| B5 cost-amortized virtual | 0.3103 | -0.5833 | 4.2975 | 0.3442 | 0.7649 | 154.58 | 没有真正 cost-amortize 成可用 |

### B 线 insight

第一，B4 说明 memory/hard-tail veto 可以压风险，但不能创造收益。它更像安全门，不是主选择器。

第二，B1/B2/B3/B5 说明虚拟路径响应会选到高 AUV 但负 V / 高风险动作，重复了 earlier exact transfer 的教训：response 大不等于好 update。

第三，probe cost q90 约 154-155 ms，这在 runtime 上也不可部署。就算质量过了，也要先降到可接受范围。

所以 B 线不是“阈值没调好”，而是当前 probe 目标错位：

$$
\boxed{
\text{当前 virtual probe 更像在测函数扰动强度，不是在测未来训练路径是否进入好 basin。}
}
$$

## 3.3 C 线：自然动作扩流仍是硬工程 blocker

当前 PanelA：

```text
labeled = 2876
CoreLike rate = 0.026773296244784424
LCB/UCB = 0.021475240123524635 / 0.033333885489495195
```

计划中的扩展 panel：

| panel | target | actual rows | missing labels | status |
|---|---:|---:|---:|---|
| Panel-5000 | 5000 | 0 | 2124 | not_run |
| Panel-10000 | 10000 | 0 | 7124 | not_run |
| Panel-20000 | 20000 | 0 | 17124 | not_run |

C 线真正的问题不是统计，而是 materializer 没有落地。没有 5000/10000/20000 的真实 labels，我们不能判断：

$$
p_{core}=P(CoreLikeAction)
$$

到底是低于、等于还是高于 $0.03$。

这件事必须优先解决，因为它决定 existing-action route 是否还有意义：

```text
如果自然扩流后 CoreLike LCB >= 0.03，existing-action route 还有希望；
如果扩流后 CoreLike LCB 仍 < 0.03，靠 AP0 action stream 本身不够，必须换 update generation。
```

## 3.4 D 线：generated route 不允许重开，判断正确

v9.8.5 的 D 线状态：

```text
generated_route_reopen_allowed = 0
generated_route_status = stopped_B_C_D_conditions_not_met
```

这个判断正确。因为：

```text
B 线没有合法机制；
C 线没有自然密度证据；
A 线虽然有 future-path 现象，但不能生成目标；
```

此时重开 generated route，只会回到过去 APG/APGT/APGU 的问题：生成合法 payload，但 outcome value-negative / high-longrisk。

---

## 4. 当前本质问题

现在不应该再把问题写成：

```text
缺一个更好的 feature；
缺一个更好的 threshold；
缺一个 APG 变体；
缺一个虚拟 probe 版本。
```

更准确的本质问题是：

$$
\boxed{
\text{未来路径中的好区域已经被观测到，但训练当下还没有可计算、低成本、合法、跨数据分布稳定的近似算子。}
}
$$

这里的“算子”不是一个复杂分数，而是一个能回答这个问题的过程：

```text
给定当前模型状态和一个小参数改动，
它是否会让后续训练路径进入更稳定、更有收益、更不遗忘、更低风险的区域？
```

当前失败说明：

```text
1. 大函数响应不是答案；
2. 当前一步 response 不是答案；
3. memory/offdiag veto 只能排坏，不能选好；
4. old diagnostic 能看见好动作，但不能 official；
5. natural action density 还没被判定；
6. generated update 没有可生成目标。
```

---

## 5. 对进度的判断

从系统能力看，进度确实很慢，因为还没有：

```text
controller pass；
selected runtime pass；
official paired replay；
short/full training；
external-ready evidence。
```

从科学定位看，v9.8.5 有价值，因为它让四线边界很清楚：

```text
A 线：现象成立；
B 线：当前合法 probe 失败；
C 线：工程 blocker；
D 线：不能重开。
```

这意味着下一轮不应该继续“调 B probe / 调 AUV / 调 CoreExpansion10”。应该做两件硬事：

```text
1. C 线工程必须落地；
2. B 线必须从 virtual response 改成 future-operator approximation，而不是继续测 response magnitude。
```

---

## 6. v9.8.6 总体目标

v9.8.6 的目标不是继续小修 v9.8.5 的 B4 或 AUV 阈值，而是并行解决两个根问题：

$$
\boxed{
\text{自然 AP0 动作密度是否足够？}
}
$$

和：

$$
\boxed{
\text{训练当下是否存在低成本、合法的未来路径近似算子？}
}
$$

如果两者都失败，则必须明确 pivot：停止 existing-action selection 主线，转向更基础的 optimizer-level update rule 研究。

---

# v9.8.6 完整实验计划

## P0. v9.8.5 边界复现

### 目标

确认 v9.8.6 没有跳过 v9.8.5 的失败边界。

### 假设

v9.8.5 的真实边界是：

```text
A future path exists；
B legal virtual probe failed；
C natural materializer missing；
D generated stopped；
controller/runtime/replay/short-full not_run。
```

### 必须记录

```text
source_route_v9850
A_line_mechanism_pass_v9850
A_line_strong_pass_v9850
B_weak_pass_v9850
B_strong_pass_v9850
C_materializer_entrypoint_found_v9850
D_generated_reopen_allowed_v9850
system_legal_controller_pass_v9850
fake/proxy/cpu audit
```

### 判断标准

通过条件：完全复现 v9.8.5 边界，并确认没有 fake/proxy/cpu rows。

失败条件：任何 v9.8.5 not_run 阶段被误写成 pass。

### 可视化

不需要复杂图，只写 boundary table。

---

## P1. A 线：未来路径类型重审，不只看单一路径类型

### 目标

v9.8.5 按单一 group 判断，没有任何 path type strong pass。但 Core77 + CoreExpansion10 正好是 $77+10=87$，应该作为组合路径类型重新评估。

### 核心问题

```text
Core77 不够 coverage；
CoreExpansion10 单独太少；
但二者组合是否形成 count=87、risk clean、value positive 的 accepted region？
```

### 假设

$$
H_{A1}: Core77 \cup CoreExpansion10
$$

可能是第一个真正接近 accepted_count gate 的 future-path accepted set。

### 必须记录

对以下集合分别记录：

```text
Core77
CoreExpansion10
Core77+CoreExpansion10
OldOnly
ExactOnly
RandomMatched
RiskCleanButLowValue
Core77+RiskCleanButLowValue top10 subset
```

每个集合记录：

```text
action_count
AUV_LCB
RiskAdjustedAUV_LCB
DelayedGain_LCB
V1/V5/V20/V80/V240 LCB
RiskPath_UCB
LongRisk_UCB
Bad_UCB
Null_UCB
MemoryFail_UCB
OffdiagFail_UCB
LDO_raw
LDO_quality
LDO_support
LDO_backfill
LDO_support_adjusted
LSO
LTO
LFO
candidate_template_count
family_count
per_dataset_count
per_dataset_V_LCB
```

### 判断标准

强通过：

$$
N_{accept}\ge 87
$$

$$
AUV^{LCB}>0
$$

$$
V_{80}^{LCB}>0
$$

$$
V_{240}^{LCB}>0
$$

$$
LongRisk^{UCB}\le 0.05
$$

$$
Bad^{UCB}\le 0.05
$$

$$
Null^{UCB}\le 0.15
$$

$$
MemoryFail^{UCB}=0
$$

$$
OffdiagFail^{UCB}=0
$$

并且：

$$
LDO_{support\_adjusted}\le 0.10,
\quad
LSO\le 0.10,
\quad
LTO\le 0.10.
$$

若 raw LDO 仍高，但 backfill share 解释超过 70%，则不能 official，但可以进入 density-gate dispute route。

### 可视化

```text
fig_A1_path_combo_V_curve.svg
fig_A2_path_combo_risk_curve.svg
fig_A3_core77_plus_expansion_pareto.svg
fig_A4_raw_LDO_vs_support_adjusted_LDO.svg
fig_A5_per_dataset_path_support.svg
```

---

## P2. B 线：停止 current virtual probe，改做 future-operator proxy

### 目标

v9.8.5 的 B probe 成本高、V LCB 负。下一轮不再修 B1-B5，而是验证一种更接近未来路径的训练当下 proxy。

### 为什么不能继续修 B1-B5

B1-B5 的共同失败是：

```text
AUV 可能高；
但 V LCB 负；
precision 低；
或者 longrisk/memory/offdiag 高；
cost q90 约 154-155 ms，完全不可部署。
```

这说明它们测的是 response magnitude，不是 future trajectory value。

### 新思路

构造一个低成本 future-operator proxy：

$$
\widehat{F}_{h}(\Delta)
$$

它不直接模拟完整 h 步，而是用训练当下的局部信息估计：

```text
这个 action 是否会让后续 h 步的 AdamW 更容易走；
是否会保住 memory/offdiag；
是否会让 hard-tail 降低；
是否会避免 risk path。
```

候选 proxy 不用 dataset name，不用未来 outcome：

```text
FO1: AdamW-aligned future residual proxy
FO2: memory-buffer response stability proxy
FO3: hard-tail contraction proxy
FO4: old-family margin preservation proxy
FO5: multi-batch agreement proxy
FO6: low-cost two-sample virtual path proxy
FO7: risk-veto-only proxy
FO8: FO1+FO2+FO3 with hard veto, no weighted score
```

### 必须记录

每个 proxy 记录：

```text
TopK87 precision
V_LCB
AUV_LCB
RiskAdjustedAUV_LCB
LongRisk_UCB
Bad_UCB
Null_UCB
MemoryFail_UCB
OffdiagFail_UCB
LDO/LSO/LTO/LFO
cost_q50/q90/q99_ms
feature_kernel_count
extra_memory_mb
correlation_with_Core77_label
correlation_with_future_AUV
correlation_with_RiskAdjustedAUV
```

### 判断标准

最低弱通过：

$$
Precision_{TopK87}\ge 0.70
$$

$$
V^{LCB}>0
$$

$$
LongRisk^{UCB}\le 0.05
$$

$$
MemoryFail^{UCB}\le 0.05
$$

$$
OffdiagFail^{UCB}\le 0.05
$$

$$
cost_{q90}\le 1.5\text{ ms}
$$

强通过：

$$
Precision_{TopK87}\ge 0.80,
\quad
LDO/LSO/LTO\le 0.10,
\quad
cost_{q90}\le 0.5\text{ ms}.
$$

如果所有 proxy 仍然低 precision 或 V LCB 负，B 线应该停止“训练当下识别”路线，转向 C/D 的动作生成或 optimizer-level rule。

### 可视化

```text
fig_B1_proxy_precision_vs_cost.svg
fig_B2_proxy_V_vs_longrisk.svg
fig_B3_proxy_AUV_vs_RiskAdjustedAUV.svg
fig_B4_proxy_score_vs_future_AUV_scatter.svg
fig_B5_proxy_score_distribution_by_group.svg
```

---

## P3. C 线：自然 AP0 动作扩流 materializer 必须落地

### 目标

解决拖了多轮的工程 blocker：5000/10000/20000 natural AP0 labeled stream extension 必须真实落盘。

### 核心假设

$$
H_C:
\text{如果自然 AP0 stream 扩大，CoreLike rate 的 LCB 会稳定超过 }0.03.
$$

如果 $H_C$ 成立，existing-action route 继续。  
如果 $H_C$ 不成立，现有 AP0 action source 密度不足，必须换生成方式。

### 实验设计

分阶段，不允许 full run 后才发现 row=0：

```text
C0: single-action preflight
C1: 16-action preflight
C2: 128-action preflight
C3: 5000 panel
C4: 10000 panel
C5: 20000 panel
```

每个阶段必须真实生成 labels：

```text
AP0 action identity
payload hash
branch/horizon rows
controls
GradeAB/CoreLike/ValuePositiveNoLongRisk labels
memory/offdiag/risk/bad/null labels
no-fake audit
```

### 必须记录

```text
panel_target
panel_actual_actions
expected_rows
actual_rows
rows_per_sec
failed_action_count
unresolved_exception_type
CoreLike_count/rate/LCB/UCB
GradeAB_count/rate/LCB/UCB
ValuePositiveNoLongRisk_count/rate/LCB/UCB
RiskCleanButLowValue_count/rate/LCB/UCB
per_dataset_rate
per_template_rate
per_family_rate
per_step_bucket_rate
quality_audit_pass
fake/proxy/cpu counts
```

### 判断标准

C 线强通过：

$$
CoreLike^{LCB}_{20000}\ge 0.03
$$

并且每个 dataset / major family 都不是 0 support。

C 线弱通过：

$$
CoreLike^{LCB}_{10000}\ge 0.028
$$

且趋势随 panel size 不下降。

C 线失败：

$$
CoreLike^{UCB}_{20000}<0.03.
$$

这种情况说明自然 AP0 动作池密度不足，必须停止 existing-action harvesting 主线。

### 可视化

```text
fig_C1_corelike_density_vs_panel_size.svg
fig_C2_gradeab_density_vs_panel_size.svg
fig_C3_density_CI_by_dataset.svg
fig_C4_density_CI_by_family.svg
fig_C5_rows_per_sec_and_failures.svg
fig_C6_quality_audit_heatmap.svg
```

---

## P4. D 线：generated route 只允许 conditional sandbox，不允许 official reopen

### 目标

防止继续盲目生成 APG 变体，同时为可能的 future-operator mechanism 准备最小 sandbox。

### Reopen 条件

只有当以下至少一个成立，才允许生成新动作：

```text
B 线找到 legal future-operator proxy weak pass；
C 线证明 natural AP0 density 不足但 A 线机制明确；
A 线证明某个 path-combo 是 clean future target，且 B 线能给出低成本近似；
```

否则：

```text
generated_route_status = stopped
```

### 如果允许 sandbox

只跑 64 actions，不直接 full branch-horizon：

```text
D0: action generation preflight
D1: no-transform sanity
D2: h1/h5/h20 lightweight path
D3: if D2 pass, h80/h240 branch-horizon
```

### 必须记录

```text
generated_action_count
payload_hash_missing
apply_error_linf
future_path_rows
V1/V5/V20/V80/V240
AUV_LCB
RiskAdjustedAUV_LCB
longrisk/bad/null/memory/offdiag
new_positive_created_rate
longrisk_created_rate
cost_q90
negative_control_gap
```

### 判断标准

Sandbox weak pass：

$$
new\_positive\_created\_rate\ge 0.10
$$

$$
RiskAdjustedAUV^{LCB}>0
$$

$$
LongRisk^{UCB}\le 0.10
$$

Sandbox strong pass：

$$
GradeABPrecision\ge 0.50
$$

$$
RiskAdjustedAUV^{LCB}>0
$$

$$
LongRisk^{UCB}\le 0.05
$$

如果 negative control 与 real primitive 差不多，立刻停止。

---

## P5. Minimal Controller Boundary

### 目标

只有 A/B/C 至少一条形成可用 accepted region，才允许 controller boundary。

### 候选 controller

```text
CTRL-A: Core77 + CoreExpansion10 path-combo diagnostic controller
CTRL-B: FO-proxy TopK87 controller
CTRL-C: Natural stream density controller
CTRL-D: Hybrid Core + FO-veto controller
```

### 必须记录

```text
accepted_count
coverage
precision
V_LCB
AUV_LCB
RiskAdjustedAUV_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO/LSO/LTO/LFO
runtime_feature_cost_q90
payload_apply_cost_q90
field legality
```

### 判断标准

$$
accepted\_count\ge 87
$$

$$
Precision\ge 0.75
$$

$$
V^{LCB}>0
$$

$$
LongRisk^{UCB}\le 0.05
$$

$$
Bad^{UCB}\le 0.05
$$

$$
Null^{UCB}\le 0.15
$$

$$
LDO/LSO/LTO\le 0.10
$$

如果不满足，不允许 runtime / paired replay / short-full。

---

## P6. Selected Runtime Boundary

### 目标

controller 过线后，才测 selected runtime。

### 必须记录

```text
step_ratio_q50/q90/q99
feature_compute_ms
probe_compute_ms
certificate_compute_ms
payload_apply_ms
kernel_launch_count
sync_count
memory_ratio
active_step_count
empty_step_count
```

### 判断标准

$$
step\_ratio_{q90}\le 1.50
$$

$$
memory\_ratio\le 1.05
$$

如果 B 线 proxy cost q90 仍然约 150 ms，则 runtime 必然失败，不能打开 controller runtime。

---

## P7. Paired Replay Boundary

### 目标

只有 controller + runtime 同时过线，才打开 paired replay。

### 必须比较

```text
RealFunctional
AdamWParallel
bestLR
NoOp
Random
ShuffledPayload
```

### 必须记录

```text
paired_action_count
real_beats_adamwparallel
real_beats_bestLR
real_beats_noop
real_beats_random
shuffle_control_fail_rate
V20/V80/V240
AUV
longrisk/bad/null/memory/offdiag
```

### 判断标准

RealFunctional 必须打过所有 controls，shuffled payload 不能通过。

---

## P8. Short/Full Boundary

### 目标

只有 paired replay 通过，才允许 short/full training。

### 指标

```text
test acc
NLL
ECE
CEp99
hard-tail acc
time-to-target
steps-to-target
sample efficiency
continual retained acc
forgetting
runtime
memory
```

### 说明

这些指标用于最终系统证明，不用于调 controller，也不允许按数据集调参。

---

## 7. 预期 route 决策

### R1-FuturePathComboCandidate

A 线 Core77+CoreExpansion10 或其他组合达到 accepted_count/value/risk gate，但 legal mechanism 仍缺。

下一步：只继续 B/C，不打开 controller。

### R2-LegalFutureOperatorProxyFound

B 线找到 low-cost legal proxy，TopK87 同时满足 precision/V/risk。

下一步：进入 P5 controller boundary。

### R3-NaturalStreamDensitySufficient

C 线 10000/20000 panel 显示 CoreLike LCB >= 0.03。

下一步：existing-action harvesting route 继续，重新定义 density-aware official gate。

### R4-NaturalStreamDensityInsufficient

C 线扩流后 CoreLike UCB < 0.03。

下一步：停止 existing-action harvesting，转向 optimizer-level generated update。

### R5-LegalProxyFailNaturalStreamMissing

B 失败且 C 仍 missing。

下一步：优先工程化 C materializer，不再调 B probe。

### R6-GeneratedSandboxAllowed

A/B/C 给出明确机制，D 可以跑 64-action sandbox。

### R7-GeneratedRouteStopped

B/C/D 条件都不满足，generated route 继续停止。

---

## 8. v9.8.6 的 Stop Rules

### Stop B current virtual probe

如果 B1-B5 类 probe 仍满足：

```text
V LCB <= 0
或 precision < 0.60
或 cost_q90 > 5 ms
```

则停止 current-response / virtual-step probe 路线。

### Stop existing-action route

如果 C 线 20000 panel 显示：

$$
CoreLike^{UCB}<0.03,
$$

且 B 线没有合法机制，则停止 existing-action route。

### Stop generated route

如果没有 B/C 机制证据，generated route 继续停止。

### Open controller

只有 P5 过线才打开 P6 runtime。

---

## 9. v9.8.6 最重要的反思

当前最危险的误区是继续这样做：

```text
发现 AUV 高 -> 加一个 AUV score；
发现 risk 高 -> 加一个 risk veto；
发现 memory/offdiag 高 -> 加一个 memory veto；
发现 B probe 失败 -> 再加一个 B probe；
发现 C missing -> 不解决 materializer，继续调 A/B；
```

这会重新进入人工规则堆叠。

v9.8.6 必须坚持一个更高层原则：

$$
\boxed{
\text{functional update 是对未来训练路径的可控扰动，不是当前 loss response 最大化。}
}
$$

因此实验重点必须是：

```text
1. 未来路径是否真实好；
2. 训练当下是否有合法低成本近似；
3. 自然动作池好动作密度是否足够；
4. 只有机制明确后才生成新动作。
```

