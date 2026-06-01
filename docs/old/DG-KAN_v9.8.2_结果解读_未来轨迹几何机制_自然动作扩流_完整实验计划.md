# DG-KAN v9.8.1 结果解读与 v9.8.2 下一步实验计划

> 本文面向没有背景知识的读者，解释 v9.8.1 这轮实验到底说明了什么，为什么看起来还是慢，当前真正卡在哪里，以及下一步应该怎样不靠小修小补，而是围绕“未来训练轨迹”和“网络几何自适应”继续并行验证。
>
> 本文中的公式全部使用 `$...$` 或 `$$...$$`，Typora 友好。

---

## 1. 一句话结论

v9.8.1 有进展，但不是能力成功。

这轮真正说明的是：

$$
\boxed{
\text{我们已经看到 Core77 / OldOnly 这类动作在 h20/h80/h240 端点上很强，}
\text{但还没有完整的 h1/h5/h20/h80/h240 训练路径，也没有一个合法、通用、低风险的几何选择规则。}
}
$$

更直白地说：

```text
已有好动作；
已有未来端点上的强信号；
但我们还不知道这些好动作为什么好，
也不能在训练当下稳定识别或生成它们。
```

所以 v9.8.1 不能打开 controller、runtime、paired replay、short/full training。它不是系统成功。

---

## 2. 这轮 v9.8.1 到底做了什么

v9.8.1 的 route 是：

```text
R4-ArchitectureAgnosticGeometryFail
```

主要 blocker 是：

```text
primary_blocker = geometry_score_low_precision_high_risk
secondary_blocker = natural_labeled_stream_extension_missing
system_legal_controller_pass = 0
generated_route_status = stopped_no_future_trajectory_mechanism_and_no_official_branch_horizon
```

也就是说，本轮没有进入 official controller，也没有 selected runtime、paired replay、short/full training。

但它不是空跑。它真实做了几件事：

```text
1. 复现 v9.8.0 boundary，没有跳过 no-system / generated-stop / no-fake 边界；
2. 再次确认自然 AP0 扩流 materializer 没有落地；
3. 把 future trajectory 从 endpoint 诊断推进到 full-path 审计，但 h1/h5 仍缺失；
4. 做了 mechanism decomposition，比较 OldRank、ExactTransfer、ImmediateValue、CleanCoreLike、function response、memory-safe 等机制；
5. 测了 architecture-agnostic geometry score，结果失败得很严重；
6. 因上游没有 official pass，controller、runtime、paired replay、short/full 全部 gate-blocked。
```

---

## 3. v9.8.1 的关键数据怎么读

### 3.1 自然动作扩流仍然没落地

当前仍然只有已有 AP0 的 `2876` 个动作。

```text
CoreLike count = 77
CoreLike rate = 0.026773296244784424
LCB = 0.021475240123524635
UCB = 0.033333885489495195
```

`5000 / 10000 / 20000` panel 都没有跑，原因是：

```text
no_landed_labeled_natural_AP0_stream_extension_materializer_for_v9810
```

这意味着我们仍然不知道：

```text
如果自然 AP0 action stream 扩大，Core-like 好动作密度会不会稳定超过 0.03。
```

这是一个根本问题。如果自然动作流里好动作密度本来就低于 $0.03$，那 existing-action route 再怎么挑也很难过 coverage；如果扩流后密度稳定超过 $0.03$，那 existing-action route 还有继续价值。

---

### 3.2 Future trajectory 端点信号强，但路径不完整

本轮要求的 horizons 是：

```text
h = 1, 5, 20, 80, 240
```

但已有 materialized horizons 只有：

```text
h = 20, 80, 240
```

也就是说，h1/h5 没有落地，所以不能声称完整 future path 已经闭合。

已观察到的 h20/h80/h240 端点 AUV LCB 是：

```text
Core77       = 2.7546595085012537
OldOnly      = 2.36612274912119
ExactOnly    = 1.4622620508289148
RandomMatched= 1.4778647567964251
```

V20 LCB 是：

```text
Core77    = 1.573549156031694
OldOnly   = 1.2768640608784525
ExactOnly = 0.49058574242981423
```

这其实是一个很有价值的信号。它说明 Core77 / OldOnly 不是只在标签上好，它们在未来端点上也明显优于 ExactOnly / RandomMatched。

但问题是：

```text
h1/h5 缺失；
每一步路径形状缺失；
h80/h240 的详细安全、memory、cover、hard-tail 曲线还没有完整成为机制证据。
```

所以我不同意把它读成“future trajectory 思路失败”。更准确是：

$$
\boxed{
\text{future endpoint evidence 支持这条路线，但完整 future path 还没落地。}
}
$$

---

### 3.3 当前的 architecture-agnostic geometry score 选错了

P4 的结果非常差：

```text
TopK87 precision = 0.06896551724137931
V LCB = -0.9348850937038008
longrisk UCB = 0.8485412092119363
LDO = 0.011494252873563218
```

这说明它不是“稍微差一点”。它选出来的动作大部分不是好动作，value 为负，longrisk 很高。

这轮最重要的反思是：

$$
\boxed{
\text{不能把“函数响应大”或“几何移动大”误认为“好几何”。}
}
$$

一个动作可能让输出变化很大、让某种 response 很强，但这可能正是危险信号：它可能把模型推到不稳定的训练路径上。

---

### 3.4 Mechanism decomposition 暴露了几个错误方向

P3 的机制表非常重要。

`M1-OldRankScore`：

```text
TopK87 precision = 0.8735632183908046
AUV LCB = 2.7378917778324734
longrisk UCB = 0.0
legal = 0
```

它很强，但不能 official，因为它不是我们已经证明可训练当下合法部署的机制。

`M2-ExactTransferScore`：

```text
TopK87 precision = 0.20689655172413793
AUV LCB = 2.0181662996752396
```

它弱，继续支持 v9.7.x 的结论：one-step exact transfer 不是好动作的充分条件。

`M3-ImmediateValue`：

```text
TopK87 precision = 0.4942528735632184
AUV LCB = 3.3202712196354365
longrisk UCB = 0.5053405200342863
```

它 value 很强，但 longrisk 很高。这说明“即时价值”会把风险动作也选上。

`M4-CleanCoreLikeLabel`：

```text
TopK87 precision = 0.8850574712643678
AUV LCB = 2.6816104392609805
longrisk UCB = 0.1674432324061406
legal = 0
```

它是很强的结果标签，但不能作为训练当下的规则。

`M5-v9800FunctionResponseDiagnostic`：

```text
effect size = 2.9632252555420653
corr/AUC proxy = 0.8031619087650855
TopK87 precision = 0.06896551724137931
AUV LCB = 7.443207887129655
longrisk UCB = 0.8485412092119363
```

这个最值得反思：AUV 很大，但 precision 极低、longrisk 极高。它说明大 function response 可能是危险的，不是好几何。

`M6-MemorySafeInverse`：

```text
legal = 1
TopK87 precision = 0.10344827586206896
AUV LCB = 1.6920562547123403
longrisk UCB = 0.0
```

它能压风险，但选不出足够 value-positive 的动作。memory safe 更像 veto，不像主 value 生成器。

---

## 4. 我对报告结论的独立判断

报告 route 写的是：

```text
R4-ArchitectureAgnosticGeometryFail
primary_blocker = geometry_score_low_precision_high_risk
```

这个对当前 P4 geometry score 是准确的。

但如果把它理解成：

```text
未来轨迹 / 几何自适应 optimizer 这条路线失败了。
```

那就过度了。

我更准确的判断是：

$$
\boxed{
\text{当前这版 geometry score 失败；但 Core77/OldOnly 的未来端点强信号仍然支持继续研究未来训练轨迹。}
}
$$

现在真正的 blocker 不是单个 score，而是三件事：

```text
1. future path 不完整：h1/h5 没有 landed rows；
2. geometry mechanism 不清楚：OldRank/Core77 为什么好仍没解释；
3. natural action density 不知道：扩流 materializer 没落地。
```

---

## 5. 为什么感觉还是非常慢

你的感觉是对的。

这几轮一直没有进入：

```text
selected controller
selected runtime
official paired replay
short/full training
```

所以从“系统能力”角度看，确实非常慢。

但从科学定位角度看，v9.8.1 不是空转。它把几个可能误导我们的方向压下去了：

```text
1. one-step exact transfer 不够；
2. windowed transfer 当前形式不够；
3. support-aware gate 不能直接改 official gate；
4. function response 大不等于好几何；
5. generated route 不能靠 h20 positive smoke 重开；
6. architecture-agnostic geometry score 不能只靠手工拼接。
```

慢的真正问题是：

$$
\boxed{
\text{我们仍然在用诊断表反推机制，而不是已经拥有一个能生成好训练轨迹的优化原则。}
}
$$

---

## 6. 当前真正卡在哪里

### 6.1 卡在完整未来路径

我们现在只看到部分端点：h20/h80/h240。

但真正要回答的是：

```text
一个动作加进去后，h1/h5/h20/h80/h240 的训练路径如何变化？
```

如果 Core77 / OldOnly 在 h1 并不强，但在 h20/h80/h240 变强，就说明它们不是 immediate loss descent，而是改变了后续训练路径。

如果它们 h1 就很强，那也许只是更好的短期 response。

现在 h1/h5 缺失，所以这个问题还没有答案。

---

### 6.2 卡在自然动作密度

CoreLike 当前是 77 个，coverage 约 `0.02677`。

coverage 下限如果按 $0.03$，在 2876 个动作中需要约：

$$
\lceil 0.03 \times 2876 \rceil = 87.
$$

现在差 10 个。

但 Wilson interval 跨过 0.03，说明还不能判断自然动作密度到底够不够。

必须跑：

```text
5000 actions
10000 actions
20000 actions
```

而不是继续在 2876 个动作里硬挤。

---

### 6.3 卡在机制解释

Core77 / OldOnly 结果好，但我们还不知道它们为什么好。

我们现在知道什么不是原因：

```text
不是 one-step exact transfer；
不是简单 WT80；
不是大 function response；
不是单纯 memory-safe；
不是当前 geometry score。
```

但正向机制仍不清楚。

现在应该把问题改成：

```text
OldOnly / Core77 相比 ExactOnly / RandomMatched，
在未来路径、memory、hard-tail、function displacement、Jacobian response、训练后续可优化性上有什么稳定差异？
```

---

### 6.4 卡在 generated route 是否能重新打开

v9.8.0 有 generated h20 positive smoke，但 v9.8.1 没有官方长程证据，因此 P6/P7 generated route 没打开。

这是对的。

不能因为 h20 LCB 为正，就重开 generated route。要重开，至少需要：

```text
h20 value positive；
h80 不坏；
h240 longrisk 低；
bad/null 低；
memory/offdiag 不伤；
和 controls 比较不输；
runtime 可承受。
```

---

## 7. 更高层的数学理解

现在最重要的数学结论不是某个指标，而是：

$$
\boxed{
\text{functional update 应该被建模为对未来训练动力学的可控扰动，而不是当前一步的 loss descent。}
}
$$

一个动作 $\Delta$ 不是好在：

$$
-g_t^\top \Delta > 0.
$$

而是好在它改变了未来训练轨迹：

$$
\Phi_h(\theta + \Delta) \quad \text{比} \quad \Phi_h(\theta) \quad \text{更好}.
$$

这里 $\Phi_h$ 表示从当前参数出发继续训练 $h$ 步后的状态。

真正的动作价值应该接近：

$$
V_h(\Delta)
=
Metric(\Phi_h(\theta))
-
Metric(\Phi_h(\theta+\Delta)).
$$

而不是只看当前一步：

$$
Metric(\theta)-Metric(\theta+\Delta).
$$

这也是为什么 ExactTransfer 测得很准，但不能挑出好动作：它近似的是当前一步的响应，不是完整训练动力学。

---

## 8. v9.8.2 下一步总体目标

v9.8.2 不再小修 P4 geometry score，也不继续盲目 generated route。

v9.8.2 的目标是：

$$
\boxed{
\text{把“未来训练轨迹变好”从解释口号，变成可落盘、可比较、可判定、可生成的实验对象。}
}
$$

本轮采用四条并行线：

```text
A 线：完整未来路径
  补 h1/h5，形成 h1/h5/h20/h80/h240 的全路径曲线。

B 线：机制对照
  比较 Core77 / OldOnly / ExactOnly / RandomMatched，查好动作到底改变了什么。

C 线：自然动作扩流
  落地 5000/10000/20000 labeled AP0 action panel，判断好动作密度是否足够。

D 线：几何自适应生成
  只有 A/B 给出清晰机制后，才允许有限重开 generated update。
```

---

# v9.8.2 完整实验计划

## P0. 复现 v9.8.1 边界

### 目标

确保 v9.8.2 没有跳过 v9.8.1 的失败边界。

### 必须记录

```text
source_route_v9810
source_primary_blocker_v9810
system_legal_controller_pass_v9810
generated_route_status_v9810
field_green_count
field_yellow_count
field_red_count
fake_data_used
proxy_row_used
cpu_offload_used
```

### 判断标准

P0 pass 当且仅当：

```text
source_route_v9810 = R4-ArchitectureAgnosticGeometryFail
system_legal_controller_pass_v9810 = 0
field_red_count = 0
fake_data_used = 0
proxy_row_used = 0
cpu_offload_used = 0
```

### 可视化

```text
p0_boundary_waterfall.svg
```

显示从 v9.8.1 route 到 v9.8.2 可运行阶段的 gate 继承关系。

---

## P1. Full Future Path Materializer：补齐 h1/h5

### 目标

v9.8.1 最大问题之一是 h1/h5 缺失。P1 要先落地完整未来路径：

```text
h = 1, 5, 20, 80, 240
```

不要先全量 2876 个动作。先跑四组动作：

```text
Core77
OldOnly
ExactOnly
RandomMatched
```

### 核心假设

$$
H_{P1}:
\text{好动作不是 current-step descent，而是在 h1/h5/h20/h80/h240 路径上表现出更稳定的后续改善。}
$$

### 每个 action / horizon 要记录

```text
action_id
group_id = Core77 / OldOnly / ExactOnly / RandomMatched
dataset_id 仅用于诊断，不进入 controller
seed
family_id
template_id
step_bucket
horizon

CE_delta
NLL_delta
margin_delta
CEp99_delta
hard_tail_loss_delta
memory_buffer_loss_delta
old_family_loss_delta
old_family_margin_delta
old_stratum_loss_delta
longrisk
bad_event
null_event
memory_fail
offdiag_fail
cover_entropy_delta
basis_effective_rank_delta
curvature_proxy_delta
jacobian_proxy_delta
AdamW_alignment
action_norm
payload_norm
OldRank_score
ExactTransfer_score
WT80_score
```

### 判断标准

P1 weak pass：

```text
h1/h5/h20/h80/h240 rows all materialized for the four groups
missing_horizon_count = 0
NaN/Inf = 0
no-transform path sanity pass = 1
```

P1 mechanism pass：

```text
Core77 and OldOnly:
  V20_LCB > 0
  V80_LCB >= 0 or not significantly negative
  V240_LCB >= 0 or longrisk_UCB <= 0.05
  h240 longrisk_UCB <= 0.05

ExactOnly:
  V20/V80/V240 weaker than OldOnly
  or longrisk/memory/offdiag fail higher than OldOnly
```

### 可视化

```text
p1_future_path_V_by_group.svg
p1_future_path_longrisk_by_group.svg
p1_future_path_memory_offdiag_by_group.svg
p1_horizon_metric_heatmap.svg
p1_group_trajectory_spaghetti.svg
```

---

## P2. Future Path Mechanism Contrast：解释 Core77 / OldOnly 为什么好

### 目标

P1 只回答“路径是不是好”。P2 要回答“为什么好”。

### 对比组

```text
Core77 vs RandomMatched
OldOnly vs ExactOnly
Core77 vs ExactOnly
OldOnly vs RandomMatched
```

### 机制候选

不要再设计大杂烩 score。每个机制单独测。

```text
M_value_future:
  future V20/V80/V240 shape

M_memory:
  old-family loss / old-stratum margin / memory buffer response

M_hardtail:
  hard-tail CEp99 / hard-tail margin

M_cover:
  cover entropy / basis effective rank / active region spread

M_curvature:
  curvature proxy / local Jacobian proxy / CEp99 tail movement

M_signal:
  cross-sample agreement / per-example response mean-variance / old-family consistency

M_adamw_path:
  whether action makes following AdamW steps easier, measured by post-action AdamW loss slope
```

### 记录指标

```text
mechanism_id
effect_size_Core_vs_Random
effect_size_OldOnly_vs_ExactOnly
TopK87_precision_if_used_alone
V_LCB_if_used_alone
longrisk_UCB_if_used_alone
memory_fail_UCB_if_used_alone
offdiag_fail_UCB_if_used_alone
LDO_drop
LSO_drop
LTO_drop
legal_at_commit
cost_ms_q90
```

### 判断标准

P2 pass 不是要求某个机制直接 controller pass，而是要求找到至少一个解释机制满足：

```text
effect_size_OldOnly_vs_ExactOnly >= 0.5
Core77_vs_Random effect size >= 0.5
TopK87 precision >= 0.60
V_LCB > 0
longrisk_UCB <= 0.10
legal_at_commit = 1
```

如果没有任何机制满足，则 route 进入：

```text
R-P2-NoMechanismExplainsGoodActions
```

这意味着 OldRank/Core77 仍只是 diagnostic，不能 controller，也不能 generator。

### 可视化

```text
p2_mechanism_effect_forest.svg
p2_oldonly_vs_exactonly_radar.svg
p2_mechanism_topk_quality_scatter.svg
p2_mechanism_lag_correlation.svg
```

---

## P3. Natural AP0 Labeled Stream Extension：动作密度是否够

### 目标

回答：当前 Core-like 动作只有 77 个，是因为 action universe 太小，还是因为自然 AP0 动作分布本来就太稀？

### 实验设计

新建 materializer，生成并标注：

```text
Panel A: 2876 existing actions, reuse canonical labels
Panel B: 5000 natural AP0 actions
Panel C: 10000 natural AP0 actions
Panel D: 20000 natural AP0 actions
```

要求：

```text
不能伪造 labels；
不能用 old table；
不能用 proxy rows；
不能只写 unlabeled actions；
每个新 action 必须有 branch-horizon outcome。
```

### 每个 panel 记录

```text
panel_id
action_count
labeled_action_count
CoreLike_count
CoreLike_rate
GradeAB_count
GradeAB_rate
ValuePositiveNoLongRisk_count
ValuePositiveNoLongRisk_rate
MemoryOffdiagCore_count
MemoryOffdiagCore_rate
per_dataset_rate
per_template_rate
per_family_rate
Wilson_LCB
Wilson_UCB
rows_per_sec
wallclock
failed_action_count
```

### 判断标准

P3 strong pass：

```text
Panel C or D labeled completion >= 0.95
CoreLike_rate_LCB >= 0.03
per_dataset CoreLike_rate_LCB >= 0.02
per_family coverage no catastrophic collapse
```

P3 weak pass：

```text
CoreLike_rate_mean >= 0.03
CoreLike_rate_UCB >= 0.03
but LCB still below 0.03
```

P3 fail：

```text
CoreLike_rate_UCB < 0.03
or labeled materializer not landed
```

### 可视化

```text
p3_density_curve_corelike.svg
p3_density_curve_gradeab.svg
p3_density_by_dataset.svg
p3_density_by_family_heatmap.svg
p3_materializer_throughput.svg
```

---

## P4. Gate Semantics：raw LDO、support-aware LDO、density gate 不能再混在一起

### 目标

v9.7.7 / v9.8.0 / v9.8.1 都显示 raw LDO 和 support-aware 解释冲突。P4 要把 official gate 的语义拆清楚。

### 三种 gate 同时记录

```text
G_raw:
  按 legacy raw leaveout 计算。

G_support:
  固定已接受动作集合，评估质量，不让外部低质动作回填污染。

G_density:
  要求动作密度足够，不允许靠低质回填补 coverage。
```

### 记录指标

```text
accepted_count
coverage
precision
V_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO_raw
LDO_quality
LDO_support
LDO_backfill
LFO
LSO
LTO
density_required_pass
```

### 判断标准

不允许单独修改 official gate。只有当 P3 density strong pass 后，才允许把 G_density 作为 official candidate gate 进入 proposal。

P4 pass：

```text
G_raw fail reason decomposed into quality / support / backfill;
if density strong pass, propose density-aware official gate;
if density fail, keep raw gate and block controller.
```

### 可视化

```text
p4_gate_decomposition_waterfall.svg
p4_raw_vs_support_lfo_lso_lto.svg
p4_backfill_quality_by_dataset.svg
```

---

## P5. Architecture-Agnostic Geometry Rebuild：不要再做一个大 score

### 目标

v9.8.1 的 geometry score 失败，因为它很可能选到了大 function response / high movement / high-risk 动作。P5 不再手写一个总分，而是先做分解。

### 几何读数分成 5 类

```text
G_func:
  function displacement on current batch / memory / hard-tail

G_path:
  future path effect h1/h5/h20/h80/h240

G_memory:
  old-family / old-stratum / memory buffer harm

G_stability:
  hard-tail CEp99 / margin tail / curvature proxy / Jacobian proxy

G_cost:
  feature cost / payload apply cost / memory delta
```

### 重要原则

```text
这些读数不要马上加权成一个 score。
先判断每类读数和 Core77 / OldOnly / ExactOnly / RandomMatched 的关系。
```

### 判断标准

P5 weak pass：

```text
at least one geometry family has:
  TopK87 precision >= 0.60
  V_LCB > 0
  longrisk_UCB <= 0.10
  legal_at_commit = 1
```

P5 strong pass：

```text
combined rule using at most 3 geometry families:
  accepted_count >= 87
  precision >= 0.75
  V_LCB > 0
  longrisk_UCB <= 0.05
  bad_UCB <= 0.05
  null_UCB <= 0.15
  LDO/LSO/LTO <= 0.10
```

### 可视化

```text
p5_geometry_family_quality_scatter.svg
p5_geometry_family_correlation_matrix.svg
p5_function_displacement_vs_longrisk.svg
p5_memory_harm_vs_future_value.svg
```

---

## P6. Existing-Action Minimal Controller Boundary

### 目标

只有 P1/P2/P3/P4/P5 给出足够证据后，才尝试 minimal controller。

### Controller 形式

不能超过 3 个主要条件：

```text
1. value/path positive condition；
2. memory/offdiag/longrisk veto；
3. cost gate。
```

形式上：

$$
Accept(a)=1
$$

当且仅当：

$$
PathValue(a)>q_v,
$$

$$
RiskVeto(a)=0,
$$

$$
Cost(a)\le C_{max}.
$$

### 记录指标

```text
accepted_count
coverage
precision
V20/V80/V240 LCB
AUV LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO/LSO/LTO/LFO
feature_cost_ms_q90
payload_apply_ms_q90
```

### 判断标准

Controller weak pass：

```text
accepted_count >= 87
precision >= 0.75
AUV_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
LDO/LSO/LTO <= 0.15
```

Controller strong pass：

```text
same as weak, but LDO/LSO/LTO <= 0.10
feature + payload cost within runtime budget
```

### 可视化

```text
p6_controller_accept_region.svg
p6_controller_metric_table.svg
p6_controller_leaveout_forest.svg
```

---

## P7. Geometry-Adaptive Generated Update：只在机制成立后重开

### 目标

Generated route 不能盲跑。只有 P1/P2/P5 找到机制后才运行。

### 允许的 generated update 形式

不拘泥于 KAN basis。动作在函数空间读数上定义，再投回参数空间。

候选子空间包括：

```text
last-layer function-space update
edge-block low-rank update
memory-safe residual update
hard-tail correction update
AdamW-compatible small residual update
```

但每个 generated action 必须满足：

```text
真实 payload；
真实 certificate；
真实 branch-horizon replay；
no fake / no proxy；
not dataset-specific。
```

### 记录指标

```text
generated_action_count
payload_hash_missing
certificate_hash_missing
action_apply_error_linf
branch_horizon_rows
h1/h5/h20/h80/h240 outcomes
V_AUV_LCB
longrisk_UCB
bad/null
memory/offdiag
source-to-generated damage
runtime cost
```

### 判断标准

P7 weak pass：

```text
generated_action_count >= 256
branch-horizon completion = 1
precision >= 0.50
V_AUV_LCB > 0
longrisk_UCB <= 0.10
```

P7 strong pass：

```text
precision >= 0.75
V_AUV_LCB > 0
longrisk_UCB <= 0.05
bad/null/memory/offdiag pass
```

### 可视化

```text
p7_generated_value_risk_frontier.svg
p7_source_to_generated_damage_sankey.svg
p7_generated_path_curves.svg
```

---

## P8. Selected Runtime Boundary

### 目标

只有 P6 或 P7 过线后才测 selected runtime。

### 记录指标

```text
selected_controller_id
selected_action_count_per_step
feature_cost_ms_q90
certificate_cost_ms_q90
payload_apply_ms_q90
kernel_launch_count
sync_count
step_ratio_q50/q90/q99
peak_memory_ratio
empty_step_launch_count
```

### 判断标准

Runtime pass：

```text
step_ratio_q90 <= 1.50
peak_memory_ratio <= 1.05
no CPU offload
no fake/proxy rows
```

### 可视化

```text
p8_runtime_breakdown.svg
p8_step_ratio_distribution.svg
p8_memory_trace.svg
```

---

## P9. Official Paired Replay Boundary

### 目标

只有 controller + runtime 同时过，才打开 paired replay。

### 分支

```text
RealFunctional
AdamWOnly
AdamWParallel
BestLR
NoOp
RandomPayload
ShuffledFunctionalPayload
```

### 记录指标

```text
final train/val/test acc
CE/NLL/ECE
CEp99
margin_p10
hard-tail acc
memory retention
forgetting
runtime
paired win rate
```

### 判断标准

Paired replay pass：

```text
RealFunctional beats AdamWParallel and BestLR on primary metric;
shuffled payload fails;
NoOp/Random do not reproduce improvement;
calibration / robustness / memory do not regress;
runtime remains within gate.
```

---

## P10. Short/Full Training Boundary

### 目标

只有 P9 过，才进入 short/full training。

### 不允许

```text
不按 dataset 调 controller；
不根据 test set 修改参数；
不把 Base-Acc Sentinel 当 functional success。
```

### 记录指标

```text
train/val/test acc
loss curve
time_to_target
steps_to_target
ECE/NLL/CEp99
hard-tail performance
memory retention
forgetting
runtime
memory
```

### 判断标准

Short/full pass：

```text
functional KAN beats AdamW KAN;
functional KAN beats matched MLP / strong LR MLP under fair budget;
robustness/calibration/memory not worse;
runtime remains legal.
```

---

# 9. 并行执行安排

v9.8.2 不能再串行等一个 runner 结束才发现 blocker。建议并行：

```text
Track A:
  P1 full future path materializer。

Track B:
  P3 natural stream extension materializer。

Track C:
  P2/P5 mechanism and geometry decomposition on existing rows。

Track D:
  P4 gate semantics and density gate analysis。
```

优先级：

```text
最高优先级：P1 h1/h5 materializer + P3 natural labeled extension。
中优先级：P2/P5 mechanism decomposition。
低优先级：P7 generated route，除非 P1/P2/P5 给出机制。
```

---

# 10. Stop / Pivot 条件

## 10.1 若 P1 无法补 h1/h5

停止 future trajectory theory promotion。只保留 h20/h80/h240 endpoint diagnostic。

## 10.2 若 P3 natural stream extension 仍无法落地

停止 density conclusion。不能再在 2876 action 上反复挤 10 个 expansion 作为主线。

## 10.3 若 P2/P5 找不到机制

OldRank/Core77 继续 diagnostic-only，不进入 controller。

## 10.4 若 P5 geometry score 继续 low precision / high risk

停止 architecture-agnostic geometry score 方向，回到机制分解，不继续调 threshold。

## 10.5 若 P7 generated route 仍 high-longrisk / value-negative

继续停止 generated route，直到出现新机制。

---

# 11. 最终判断

v9.8.1 不是失败回退。它的重要性在于：

```text
它证明我们不能把 geometry-adaptive optimizer 简化成一个手写几何分数；
也不能只看 h20 endpoint；
也不能在 natural action density 未闭合时改 official gate；
更不能在没有未来路径机制时重开 generated route。
```

下一步必须把问题提升到更硬的层面：

$$
\boxed{
\text{一个好的 functional update，必须在完整未来训练路径上改善模型，且这个改善机制要能被训练当下的合法几何读数解释。}
}
$$

这才是真正的 v9.8.2 主线。
