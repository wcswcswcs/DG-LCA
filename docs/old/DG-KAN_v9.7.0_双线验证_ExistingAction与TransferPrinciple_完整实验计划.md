# DG-KAN v9.7.0 双线验证：Existing-Action Core-Expansion 与 Cross-Sample Transfer Principle 完整实验计划

> 本计划基于 v9.6.8 的真实结果制定。  
> v9.6.8 的结论是：`route = R6-GeneratedRouteStoppedNoNewObjective`，strict PureKAN functional、full functional、external ready 都仍为 false。  
> 本计划不继续小修 `R8A / C16G / T3.1 / T3.2 / APGU`。  
> 本计划采用双线验证：一条线继续推进当前最接近成功的 existing-action 路线；另一条线验证一个更少人工设计的新原则：**一个动作必须能从生成它的样本迁移到没参与生成它的样本**。

---

# 0. 先解释几个词

## 0.1 什么是 action？

这里的 action 指一次额外参数改动。正常训练每一步都会有 AdamW 更新：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{AdamW}.
$$

我们想在某些训练步额外加一小块更新：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{AdamW}+\Delta\theta_{func}.
$$

其中 $\Delta\theta_{func}$ 就是本计划说的 action。

## 0.2 什么是 existing-action？

existing-action 指 canonical AP0 表里已经存在的 `2876` 个历史 action。  
这些 action 已经有完整 branch-horizon outcome，也就是它们在多个对照分支和多个 horizon 下的真实结果已经测过。

## 0.3 什么是 generated-action？

generated-action 指我们新造出来的 action，比如 APG / APGA / APGH / APGT / APGU 这类生成器产生的动作。  
过去多轮 generated-action 工程链路都能跑通，但大多数结果是 value 为负、long-risk 高，所以 v9.6.8 继续触发 generated route stop-rule。

## 0.4 什么是 GradeAB？

GradeAB 是我们用历史 branch-horizon outcome 给 action 打出的较高质量标签。它不是训练当下能直接看到的东西。  
它只用来评估某个选择规则最后选出的动作好不好，不能直接作为 online controller 的输入。

## 0.5 什么是 LDO / LSO / LTO？

这三个是 leave-out 检查。

```text
LDO = leave-dataset-out，留下一个数据集不参与调规则，看规则在这个数据集上是否还成立。
LSO = leave-stratum-out，留下一个样本分层不参与调规则，看规则在这个分层上是否还成立。
LTO = leave-template-out，留下一个 action 模板不参与调规则，看规则在这个模板上是否还成立。
```

这些检查很重要，因为我们不是在 MNIST / Fashion-MNIST / KMNIST 上调榜，而是在找一个跨数据、跨样本分层、跨 action 模板都成立的训练规则。

## 0.6 什么是 Cross-Sample Transfer？

Cross-Sample Transfer 的意思是：  
**一个 action 不能只让生成它的样本变好，还必须让没参与生成它的样本也变好。**

简单说：

```text
用 A 组样本提出一个 action；
用 B 组样本检查这个 action；
如果 B 组也变好，说明这个 action 更可能是真 signal；
如果只有 A 组变好，说明它可能只是记住了局部噪声。
```

这条原则比手写很多 score 项更简单，也更接近我们要的泛化目标。

---

# 1. v9.6.8 的独立判断

## 1.1 v9.6.8 有进展，但不是能力成功

v9.6.8 没有 selected controller，没有 selected runtime，没有 paired replay，也没有 short/full training。它不是 functional success。

但是它有真实推进：

```text
1. score transport 被系统检查过；
2. core + expansion target 又扫了一轮；
3. veto 顺序被检查过；
4. dataset-invariant ranker 又做了一轮；
5. high-precision certificate 也被检查过；
6. generated route stop-rule 继续执行，没有盲目跑 APGU。
```

所以这轮不是空跑，而是把当前路线的边界进一步确认了。

## 1.2 当前路线最核心的问题

v9.6.8 的 best transport 是 `S0-raw-score`：

```text
macro precision = 0.8671023965141612
macro V LCB = 0.10955705927483939
longrisk UCB = 0.0
LDO drop = 0.37931034482758624
```

这说明原始分数能找到高质量动作，但跨数据集不稳。  
另一方面，能降低 LDO 的 transport，比如 `S5-memory-offdiag-safe-subset-percentile`，会让质量崩掉：

```text
LDO drop = 0.06896551724137934
pooled precision = 0.39080459770114945
V LCB = -0.01929919347292527
```

所以当前不是“分数完全没用”，而是：

$$
\boxed{\text{高质量分数不稳；稳定分数质量差。}}
$$

## 1.3 Core + Expansion 仍没有解决

v9.6.8 的 best target 是 `T3.1-CoreOnly`：

```text
accepted = 77
coverage = 0.026773296244784424
GradeAB precision = 1.0
V LCB = 0.16402807605399972
longrisk/bad/null/memory/offdiag UCB = 0
LDO drop = 0.39080459770114945
```

它很干净，但数量不够。如果 coverage 下限是 $0.03$，在 2876 个 action 中至少需要：

$$
N_{min}=\lceil 0.03\times 2876\rceil=87.
$$

所以 core 还差：

$$
87-77=10
$$

个 action。

v9.6.8 试了扩展到 87 个 action，例如 `T3.2-CorePlusNearest10TransportedScore`：

```text
accepted = 87
coverage = 0.030250347705146036
GradeAB = 0.8850574712643678
V LCB = 0.1597545290247637
null UCB = 0.1674432324061406
offdiag UCB = 0.18196532683597333
LDO drop = 0.39080459770114945
```

它数量够了，但 null/offdiag/LDO 不过。  
所以当前 core-expansion 的本质不是“再补 10 个就好”，而是：

$$
\boxed{\text{最难的是找 10 个既干净、又跨数据集稳、又不触发 null/offdiag 的扩展动作。}}
$$

## 1.4 Veto 顺序没有解决根因

v9.6.8 的 best veto strategy 是 `V7-two-stage-core-first-expansion-second`：

```text
accepted = 87
GradeAB = 0.8850574712643678
V LCB = 0.13618096164286234
longrisk/bad/null/memory/offdiag UCB = 0
LDO drop = 0.39080459770114945
```

它质量可以，但 LDO 仍旧很高。  
风险优先 veto 可以降低 LDO，例如 `V1-risk-veto-before-value-rank`：

```text
precision = 0.27586206896551724
V LCB = -0.08151817674709344
LDO drop = 0.06896551724137931
```

这说明 risk-first 会把好动作也一起删掉。  
所以不应继续简单调 veto 顺序。

## 1.5 Ranker 和 certificate 仍是 diagnostic，不是 controller

v9.6.8 的 best ranker `R8A-transported-value-rank`：

```text
TopK87 precision = 0.8735632183908046
V LCB = 0.14111334880346277
longrisk/bad/null/memory/offdiag UCB = 0
LDO drop = 0.37931034482758624
```

质量强，但 LDO 不稳。

best certificate `C16G-topK64-high-precision-diagnostic` 更漂亮：

```text
accepted = 64
coverage = 0.022253129346314324
precision = 1.0
V LCB = 0.19924925130425136
risk/bad/null/memory/offdiag UCB = 0
LDO drop = 0.328125
```

它证明存在一小批极干净动作，但 coverage 不够，而且 LDO 仍不过。  
这不能 promotion 成 controller。

## 1.6 Generated route 停止是正确的

v9.6.8 继续触发 generated stop-rule：

```text
APGH/APGL/APGT consecutive failure family count = 3
new objective evidence = 0
APGU_run = 0
```

P9 damage notebook 显示：

```text
dominant damage mode = longrisk_created
assigned fraction = 1.0
recommendation = stop_blind_generated_variants
```

所以继续盲跑 APGU 是不科学的。generated route 只有在出现新目标、新数学判据或新动作空间时才应该重新打开。

---

# 2. 当前总体结论

v9.6.8 后，最准确的状态是：

$$
\boxed{
\text{existing-action 路线有强局部信号，但跨数据集不稳；generated-action 路线没有新目标，应继续停止。}
}
$$

更直白地说：

```text
我们已经能在历史 action 里找到一小批非常干净的动作；
也能找到 87 个左右看起来不错的动作；
但这些动作换数据集后不稳。

我们也尝试过很多新动作生成器；
它们工程上能跑，但生成动作长期风险高、value 负；
所以不能继续盲目造新变体。
```

---

# 3. 为什么感觉非常慢？

你的感觉是对的。现在慢有两个原因。

第一，我们一直在排除假成功：

```text
GCERT18 很强，但用了结果字段；
payload0 口袋很强，但 intervention 后不是因果机制；
accepted region 很强，但曾经被 hidden pocket / lineage / dataset shift 否掉；
R5B / R8A 分数很强，但 LDO 仍高；
CoreOnly 很干净，但数量不够；
APG/APGH/APGT 工程闭合，但生成动作反复 high-longrisk。
```

第二，我们在当前路线里继续尝试的东西越来越像人工规则：

```text
score transport；
core + expansion；
veto order；
ranker；
certificate；
new primitive family；
stop-rule；
```

这些都不是没用，但如果继续只沿这个方向走，会越来越像手工拼装。  
所以 v9.7.0 必须双线验证：

```text
A 线：把当前 existing-action route 推到它能到的边界；
B 线：验证一个更简单的数学原则，即 cross-sample transfer。
```

---

# 4. v9.7.0 的总目标

v9.7.0 的总目标是：

$$
\boxed{
\text{判断当前 existing-action 路线是否还能过线；同时验证 cross-sample transfer 是否能更少人工地选出好动作。}
}
$$

它不是在数据集上调榜，也不是按 dataset 写规则。

它要回答两个问题。

## 4.1 问题 A：当前路线还能不能继续？

具体问：

```text
能不能从 77 个干净 core 动作扩展到至少 87 个，
同时保持：
  GradeAB 高；
  V LCB > 0；
  longrisk/bad/null/memory/offdiag 低；
  LDO/LSO/LTO 稳定。
```

## 4.2 问题 B：新原则有没有用？

具体问：

```text
一个 action 如果能让没参与生成它的样本也受益，
它是否更可能跨数据集稳定？
```

也就是验证：

$$
\boxed{
LCB_{transfer}(\Delta)>0
}
$$

是否比复杂的人工 score 更接近真正好动作。

---

# 5. 核心假设

## H1：现有 R8A / C16G 不是无用，而是缺少跨数据集稳定条件

现象：

```text
R8A TopK87 precision 高，但 LDO 高；
C16G TopK64 precision = 1.0，但 coverage 不够且 LDO 高。
```

假设：

$$
\boxed{
\text{如果把 expansion action 按 transfer 规则选择，而不是按 raw score 选择，LDO 会下降且 value 不会崩。}
}
$$

成立标准：

```text
TopK87 / accepted87:
  GradeAB precision >= 0.75
  V LCB > 0
  longrisk UCB <= 0.05
  bad UCB <= 0.05
  null UCB <= 0.15
  memory/offdiag UCB <= 0.05
  LDO / LSO / LTO drop <= 0.10
```

## H2：Cross-sample transfer 是比人工 geometry score 更简单的主判据

定义：

给定 action $\Delta$，在当前训练数据里取一个检查集合 $B$，计算每个样本的线性化收益：

$$
r_i(\Delta)=-g_i^\top \Delta.
$$

其中 $g_i$ 是第 $i$ 个样本对参数的梯度。  
如果 $r_i(\Delta)>0$，表示这个 action 预计会让这个样本的 CE loss 下降。

定义：

$$
\mu_B(\Delta)=\frac{1}{|B|}\sum_{i\in B}r_i(\Delta),
$$

$$
\sigma_B^2(\Delta)=\frac{1}{|B|-1}\sum_{i\in B}(r_i(\Delta)-\mu_B(\Delta))^2,
$$

$$
LCB_{transfer}(\Delta)=\mu_B(\Delta)-z\frac{\sigma_B(\Delta)}{\sqrt{|B|}}.
$$

接受的核心条件是：

$$
LCB_{transfer}(\Delta)>0.
$$

这不是手工加权 score。它只问：

```text
这个动作是否能帮助没参与生成它的样本？
```

## H3：SNR gate 可以压住噪声动作

定义：

$$
SNR(\Delta)=\frac{\mu_B(\Delta)^2}{\sigma_B^2(\Delta)/(b-1)+\epsilon}.
$$

如果一个 action 的均值收益很小、样本间方差很大，它就更像噪声方向。  
如果均值收益大且方差小，它更像多个样本共同支持的方向。

成立标准：

```text
TransferLCB + SNR gate 相比 raw score：
  LDO drop 降低；
  precision 不明显下降；
  longrisk 不上升；
  V LCB 仍为正。
```

## H4：Generated route 不能在没有新 objective 时重开

v9.6.8 证明 APGH/APGL/APGT 连续失败，APGU 正确未运行。  
所以 generated route 只有在 H2/H3 给出新 objective 后才允许打开。

成立标准：

```text
direct transfer-solved update 至少在 smoke 中达到：
  GradeAB precision >= 0.25
  V LCB > 0
  longrisk UCB <= 0.25
  new positive created rate >= 0.10
```

如果达不到，generated route 继续 stop，不再盲目新增 primitive。

---

# 6. 总体实验结构

v9.7.0 分成两条线。

```text
A 线：Existing-action continuation
  用现有 canonical AP0 actions，继续寻找可部署 accepted region。

B 线：Cross-sample transfer principle
  不再靠复杂手工 score，而是用“跨样本收益”判断 action。
```

两条线并行跑，互不等待。  
如果 A 线先过，则进入 selected runtime。  
如果 B 线先过，则用 B 线的规则替代人工 score。  
如果都不过，则 existing-action route 和 blind generated route 都必须暂停，进入更底层的 action-space redesign。

---

# 7. P0：边界复现与字段合法性

## 目标

确认 v9.6.8 的数据边界被正确复现，不能跳过失败条件。

## 记录

```text
source_v9680_route
system_legal_controller_pass_v9680
generated_route_stop_triggered_v9680
APGU_run_v9680
field_legality_pass
green/yellow/red field counts
outcome_derived_field_used_count
dataset_name_commit_feature_count
fake/proxy/cpu_offload counts
Base-Acc Sentinel pass
```

## 判断标准

```text
P0 pass iff:
  source_v9680_route = R6-GeneratedRouteStoppedNoNewObjective
  field_legality_pass = 1
  outcome_derived_field_used_count = 0
  dataset_name_commit_feature_count = 0
  fake/proxy rows = 0
```

## 可视化

```text
fig_p0_boundary_gate_status.svg
fig_p0_field_legality_bar.svg
```

---

# 8. P1：Cross-Sample Transfer Ledger

## 目标

给每个 canonical AP0 action 计算训练当下可见的 transfer 分数。  
这一步不使用 future outcome，不使用 GradeAB，不使用 branch-horizon result。

## 方法

对每个 action $\Delta$，计算：

$$
r_i(\Delta)=-g_i^\top\Delta.
$$

然后在多个集合上分别计算：

```text
current batch heldout subset B
old-family memory subset M_old
hard-tail subset H
stratum-balanced subset S_bal
```

记录：

$$
\mu_B,\quad \sigma_B,\quad LCB_B,\quad SNR_B.
$$

同时记录 memory/hard-tail：

$$
LCB_{memory}(\Delta),
$$

$$
LCB_{hardtail}(\Delta).
$$

## 记录字段

```text
action_id
event_id
candidate_template_id
dataset_id_diagnostic_only
stratum_id_diagnostic_only
family_id
step_bucket
payload_norm
payload_linf
action_adamw_cosine
r_mean_batch
r_std_batch
transfer_lcb_batch
snr_batch
r_mean_memory
r_std_memory
transfer_lcb_memory
snr_memory
r_mean_hardtail
r_std_hardtail
transfer_lcb_hardtail
snr_hardtail
feature_compute_ms
payload_apply_ms_estimate
actual_GradeAB_label_for_evaluation_only
actual_V_integrated_for_evaluation_only
actual_longrisk_for_evaluation_only
actual_bad_null_for_evaluation_only
```

## 硬约束

```text
dataset_id 只能用于诊断与 leaveout，不能进入 score；
actual outcome 只用于评估，不进入 score；
所有 transfer 字段必须来自 commit-time train batch / train memory；
```

## 判断标准

P1 不直接要求 pass。它只要求 ledger 完整：

```text
all 2876 canonical AP0 actions have transfer rows;
missing required fields = 0;
NaN/Inf = 0;
feature_compute_ms_q90 recorded;
```

## 可视化

```text
fig_p1_transfer_lcb_hist_by_dataset.svg
fig_p1_transfer_lcb_vs_V_integrated.svg
fig_p1_transfer_lcb_vs_longrisk.svg
fig_p1_snr_vs_gradeab.svg
fig_p1_memory_transfer_vs_longrisk.svg
```

---

# 9. P2：Existing-action rank 对照实验

## 目标

比较旧路线和新 transfer 原则谁更能稳定选动作。

## 候选 ranking rules

```text
R0: raw R8A transported-value-rank
R1: C16G diagnostic score without outcome fields
R2: TransferLCB only
R3: TransferLCB + SNR gate
R4: TransferLCB + memory hard gate
R5: TransferLCB + hardtail hard gate
R6: TransferLCB + memory + hardtail gates
R7: TransferLCB with global conformal threshold
R8: TransferLCB with no dataset-specific quantile transport
```

注意：这里不允许手写很多权重。  
最多允许以下形式：

$$
Rank(\Delta)=-LCB_{transfer}(\Delta)
$$

并加少数硬门：

$$
LCB_{memory}(\Delta)>-\tau_M,
$$

$$
LCB_{hardtail}(\Delta)>-\tau_H,
$$

$$
Cost(\Delta)\le C_{max}.
$$

## 评估方式

对每个 rule，评估：

```text
TopK64
TopK77
TopK87
TopK97
frozen accepted region at calibration threshold
```

## 必须记录

```text
accepted_count
coverage
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
max dataset share
max stratum share
max template share
feature cost q90
```

## Pass 标准

Strong pass：

$$
N_{accept}\ge87,
$$

$$
Precision_{GradeAB}\ge0.75,
$$

$$
LCB(V_{integrated})>0,
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
LDO\_drop\le0.10,
$$

$$
LSO\_drop\le0.10,
$$

$$
LTO\_drop\le0.10.
$$

Weak pass：

```text
accepted_count >= 87
precision >= 0.70
V LCB > 0
longrisk UCB <= 0.10
LDO/LSO/LTO drop <= 0.20
```

Weak pass 只能决定是否继续，不允许直接 official paired replay。

## 可视化

```text
fig_p2_rule_precision_value_risk_bar.svg
fig_p2_rule_ldo_lso_lto_heatmap.svg
fig_p2_topk_stability_waterfall.svg
fig_p2_transfer_vs_old_rank_scatter.svg
```

---

# 10. P3：Core + Expansion with Transfer

## 目标

检查能不能以 T3.1 的 77 个 clean core 为起点，只用 transfer 原则补 10 个 expansion action。

## 方法

Core 固定为 v9.6.8 的 `T3.1-CoreOnly`：

```text
core_count = 77
core_quality = clean
```

Expansion pool 不从 core 中取。  
按照下面顺序挑 10 个：

```text
1. transfer_lcb_batch > 0
2. transfer_lcb_memory > -tau_M
3. hardtail transfer not harmful
4. no longrisk legal proxy alarm
5. lowest feature cost if tie
```

不得使用 dataset name，不得使用 outcome label。

## 需要比较的 baselines

```text
B0: T3.1-CoreOnly
B1: T3.2-CorePlusNearest10TransportedScore
B2: T3.4-CorePlusNearest10ConformalLowRisk
B3: CorePlus10TransferLCB
B4: CorePlus10TransferLCBMemoryGate
B5: CorePlus10TransferSNROnly
```

## Pass 标准

```text
accepted_count = 87
coverage >= 0.03
GradeAB precision >= 0.80
V LCB > 0
longrisk/bad UCB <= 0.05
null UCB <= 0.15
memory/offdiag UCB <= 0.05
LDO/LSO/LTO drop <= 0.10
```

## 可视化

```text
fig_p3_core_expansion_sankey.svg
fig_p3_expansion_candidates_transfer_scatter.svg
fig_p3_core_vs_expansion_quality_table.svg
fig_p3_core_expansion_dataset_support_heatmap.svg
```

---

# 11. P4：Dataset shift 不调参诊断

## 目标

诊断不同 dataset 上分数为什么会漂移，但不按 dataset 写规则。

## 要比较的 score

```text
raw score
transported-value-rank score
TransferLCB
TransferLCB + SNR gate
TransferLCB + memory gate
```

## 记录

```text
per dataset score mean
per dataset score std
per dataset PSI
per dataset TopK count
per dataset TopK precision
per dataset V LCB
per dataset longrisk UCB
per dataset core action count
per dataset expansion action count
```

## 判断

如果 transfer score 的 dataset PSI 明显低于 raw score，且 precision/value 不崩，则说明新原则能缓解 dataset shift。

Strong diagnostic：

```text
TransferLCB PSI <= 0.5 * raw_score PSI
and TopK87 precision drop <= 0.10
and V LCB remains > 0
```

如果做不到，则说明 dataset shift 不是单纯 score scale，而是 action support / target density 本身不同。

## 可视化

```text
fig_p4_score_distribution_by_dataset.svg
fig_p4_dataset_shift_waterfall.svg
fig_p4_dataset_target_density_map.svg
fig_p4_transfer_vs_raw_psi_bar.svg
```

---

# 12. P5：Exact microprobe 校验线性 transfer

## 目标

检查线性近似 $-g_i^\top\Delta$ 是否可信。  
如果线性 transfer 和真实小步 apply 差别很大，说明当前 transfer 估计不能直接作为 controller。

## 方法

对以下 action 子集做小规模 exact microprobe：

```text
TopK64 by TransferLCB
TopK64 by raw score
TopK64 by R8A
random64
high-risk64
```

对每个 action，在当前 batch holdout subset 上真实 apply 一个小比例：

$$
\theta' = \theta + \alpha\Delta,
$$

记录真实 CE delta：

$$
\Delta CE_i^{exact}=CE_i(\theta)-CE_i(\theta').
$$

比较：

$$
\Delta CE_i^{linear}\approx -g_i^\top\Delta.
$$

## 记录

```text
linear_exact_corr
linear_exact_mae
linear_exact_sign_match
exact_transfer_lcb
exact_memory_lcb
exact_hardtail_lcb
microprobe_cost_ms_q90
```

## Pass 标准

```text
sign_match >= 0.70
corr >= 0.50
microprobe_cost_ms_q90 <= budget
```

Microprobe 不直接进入 official controller，除非成本满足 runtime gate。

## 可视化

```text
fig_p5_linear_vs_exact_scatter.svg
fig_p5_sign_match_by_action_group.svg
fig_p5_microprobe_cost_hist.svg
```

---

# 13. P6：Direct Transfer-Solved Update，不再盲目 APG 变体

## 目标

如果 cross-sample transfer 是有效原则，就不应只用它给旧 action 排名，还应该直接求一个 action。

## 形式

在一个小 KAN 参数子空间 $\mathcal{A}$ 中求：

$$
\Delta^*=\arg\max_{\Delta\in\mathcal{A}}LCB_{transfer}(\Delta)
$$

并满足：

$$
\|\Delta\|\le\epsilon,
$$

$$
LCB_{memory}(\Delta)>-\tau_M,
$$

$$
Cost(\Delta)\le C_{max}.
$$

这里不再设计很多人工 primitive，只测试少量清楚的参数子空间。

## 子空间

```text
D1: last-edge coefficients only
D2: final KAN basis block only
D3: low-rank edge residual direction
D4: AdamW-orthogonal residual direction
D5: memory-gradient-orthogonal residual direction
```

每个子空间最多生成 64 个 action，总量不超过 320 个。  
如果 D1-D5 都失败，不再继续新增 D6-D20。

## 必须记录

```text
generated_action_count
payload/certificate hash missing
action_apply_linf_max
branch_horizon_rows expected/actual
GradeAB precision
V_integrated LCB
longrisk/bad/null UCB
memory/offdiag UCB
new positive created rate
longrisk created rate
runtime cost estimate
```

## Weak pass

```text
GradeAB precision >= 0.25
V LCB > 0
longrisk UCB <= 0.25
new positive created rate >= 0.10
```

## Strong pass

```text
GradeAB precision >= 0.50
V LCB > 0
longrisk UCB <= 0.10
bad/null UCB within gate
```

## Stop rule

如果所有 D1-D5 满足：

```text
GradeAB precision < 0.10
or V LCB < 0
or longrisk UCB > 0.50
or longrisk created rate > 0.50
```

则 generated route 继续停止，不允许新增 blind primitive。

## 可视化

```text
fig_p6_direct_solved_frontier.svg
fig_p6_new_positive_vs_longrisk.svg
fig_p6_subspace_damage_matrix.svg
fig_p6_action_norm_vs_transfer_lcb.svg
```

---

# 14. P7：双线合流判断

## 目标

决定 v9.7.0 后的路线。

## 路线判断

### Case A：existing-action 过线

条件：P2 或 P3 strong pass。

动作：

```text
进入 P8 minimal controller；
进入 P9 selected runtime；
不再优先生成新 action。
```

### Case B：transfer principle 过线但 existing-action 未过

条件：P2/P3 weak pass，P6 direct-solved weak pass。

动作：

```text
继续 transfer-solved primitive；
暂停人工 APG/APGU；
下一轮扩大 D1-D5 子空间。
```

### Case C：transfer principle 也失败

条件：P2/P3/P6 都未过。

动作：

```text
停止 existing-action rank patch；
停止 generated blind variant；
转入 action-space redesign：KAN basis / edge parameterization / source of functional update 要重审。
```

---

# 15. P8：Minimal Controller

## 目标

只有 P2/P3/P6 至少 weak pass，才允许打开 minimal controller。

## Controller 形式

Existing-action route：

$$
Accept(a)=1
$$

当且仅当：

$$
Rank_{transfer}(a)\le K
$$

并且：

$$
MemoryGate(a)=1,
$$

$$
HardTailGate(a)=1,
$$

$$
Cost(a)\le C_{max}.
$$

Direct-solved route：

$$
Accept(\Delta^*)=1
$$

当且仅当：

$$
LCB_{transfer}(\Delta^*)>0,
$$

$$
LCB_{memory}(\Delta^*)>-\tau_M,
$$

$$
Cost(\Delta^*)\le C_{max}.
$$

## Calibration / heldout

```text
calibration split: choose K / tau_M / tau_cost
heldout split: frozen evaluation
leave-dataset-out: no threshold change
leave-stratum-out: no threshold change
leave-template-out: no threshold change
```

## Pass 标准

```text
accepted_count >= 87
coverage >= 0.03
GradeAB precision >= 0.75
V LCB > 0
longrisk UCB <= 0.05
bad UCB <= 0.05
null UCB <= 0.15
memory/offdiag UCB <= 0.05
LDO/LSO/LTO drop <= 0.10
```

## 可视化

```text
fig_p8_calibration_vs_heldout.svg
fig_p8_accepted_region_by_dataset.svg
fig_p8_accepted_region_by_template.svg
fig_p8_controller_gate_waterfall.svg
```

---

# 16. P9：Selected Runtime

## 目标

controller 过线后，测真实 selected runtime。  
不能用 diagnostic estimate，也不能把 feature cost 漏掉。

## 记录

```text
selected controller id
feature compute q50/q90/q99
microprobe compute q50/q90/q99 if used
payload apply q50/q90/q99
kernel launch count
sync count
step_ratio_q90
peak memory ratio
zero-candidate step cost
active-step cost
```

## Pass 标准

$$
StepRatio_{q90}\le1.50,
$$

$$
MemoryRatio\le1.05.
$$

## 可视化

```text
fig_p9_runtime_component_stack.svg
fig_p9_step_ratio_distribution.svg
fig_p9_active_vs_zero_step_cost.svg
```

---

# 17. P10：System Boundary

## System pass 条件

```text
P8 controller pass = 1
P9 runtime pass = 1
field legality pass = 1
no fake/proxy/cpu offload = 1
no dataset-specific controller = 1
manual training contract pass = 1
```

如果不过，不允许打开 paired replay / short-full。

---

# 18. P11：Paired Replay Boundary

## 开启条件

```text
System pass = 1
```

## 记录

```text
RealFunctional vs AdamWParallel
RealFunctional vs bestLR
RealFunctional vs NoOp
RealFunctional vs Random
RealFunctional vs shuffled payload
CE / NLL / ECE / margin / CEp99
h20 / h80 / h240
leave-dataset-out
leave-stratum-out
leave-template-out
```

## Pass 标准

```text
RealFunctional beats all controls;
shuffled payload fails;
LDO/LSO/LTO stable;
no runtime regression;
```

---

# 19. P12：Base-Acc Sentinel

## 目标

继续健康检查，不参与 controller。

## 记录

```text
MNIST / Fashion-MNIST / KMNIST
seeds 0..9
LQ mean test acc
MatchedMLP mean test acc
AdamWStrongLRGridMLP mean test acc
LQ minus MLP
LQ minus StrongLRGridMLP
catastrophic fail
base_acc_used_for_controller = 0
```

## 判断

```text
Base-Acc Sentinel pass iff:
  catastrophic fail = 0
  base_acc_used_for_controller = 0
```

Base-Acc 不能用于：

```text
selector
controller
certificate
threshold
route promotion
```

---

# 20. 必须可视化清单

```text
fig_p1_transfer_lcb_hist_by_dataset.svg
fig_p1_transfer_lcb_vs_V_integrated.svg
fig_p1_snr_vs_gradeab.svg
fig_p2_rule_precision_value_risk_bar.svg
fig_p2_rule_ldo_lso_lto_heatmap.svg
fig_p3_core_expansion_sankey.svg
fig_p3_expansion_candidates_transfer_scatter.svg
fig_p4_score_distribution_by_dataset.svg
fig_p4_dataset_shift_waterfall.svg
fig_p5_linear_vs_exact_scatter.svg
fig_p6_direct_solved_frontier.svg
fig_p6_subspace_damage_matrix.svg
fig_p8_accepted_region_by_dataset.svg
fig_p8_controller_gate_waterfall.svg
fig_p9_runtime_component_stack.svg
```

---

# 21. 并行执行安排

## Batch A：existing-action，不需要生成新动作

```text
P0 boundary reproduction
P1 transfer ledger for canonical AP0
P2 rank comparison
P3 core + expansion with transfer
P4 dataset shift diagnostics
```

这些都基于 existing AP0，可以并行。

## Batch B：exact microprobe

```text
P5 exact microprobe
```

只跑 TopK / random / high-risk subset，不等待 P3 完全结束。

## Batch C：direct-solved update

```text
P6 direct transfer-solved update D1-D5
```

D1-D5 可以并行，但必须遵守总 action 数上限。

## Batch D：controller/runtime

```text
P8 controller
P9 runtime
P10 system
```

只有 A/B/C 出现 pass 才打开。

---

# 22. Stop Rules

## 停止 old rank patch

如果满足：

```text
P2 all transfer / old rank rules fail strong and weak;
P3 core+expansion fail;
P4 confirms dataset shift is target density not score scale;
```

则停止 R8A/C16G/T3.x 小修。

## 停止 generated route

如果 P6 D1-D5 全部满足：

```text
V LCB < 0
or longrisk UCB > 0.50
or GradeAB precision < 0.10
```

则 generated route 继续停止，不允许 blind APGU / APGV。

## 进入 system

只有在：

```text
P8 controller pass
P9 runtime pass
```

后进入 paired replay。

---

# 23. 最终 route 决策

```text
R0-BoundaryOrLegalityFail
  P0 fails.

R1-ExistingActionTransferControllerPass
  P2/P3/P8/P9 pass.

R2-TransferPrincipleWeakButNotSystem
  transfer rule improves LDO/quality but controller/runtime not pass.

R3-DirectSolvedTransferPrimitivePromising
  P6 weak/strong pass but controller not yet built.

R4-TransferPrincipleFail
  transfer/SNR does not improve old rank.

R5-CoreExpansionDensityStillAbsent
  core remains clean but expansion cannot reach 87 safely.

R6-GeneratedRouteStoppedNoNewObjective
  P6 fails and no new objective evidence.

R7-SystemPassReadyForPairedReplay
  P8/P9/P10 all pass.
```

---

# 24. 最终判断标准

v9.7.0 不要求一次性 full functional success。  
v9.7.0 的最低有效进展是回答：

```text
1. cross-sample transfer 是否比当前手工 score 更稳；
2. clean core 的 10 个缺口是否能用 transfer 补上；
3. generated route 是否有新的数学目标可以重开；
4. 如果没有，是否应该停止 existing-action rank patch 和 blind generated variants；
5. 如果有，是否可以进入 selected runtime / paired replay。
```

一句话：

$$
\boxed{
\text{v9.7.0 要把路线从“更多人工规则”压缩成“跨样本 transfer 是否成立”这个更小的数学问题。}
}
$$
