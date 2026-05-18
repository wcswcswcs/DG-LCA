# DG-KAN v9.8.9 自然扩流 / FuturePathOperator / 四线并行完整实验计划

> 本文件基于 v9.8.8 实验结果制定。目标不是继续小修某个分数、阈值或 APG 变体，而是把当前四线真正推到可以裁决的边界：自然动作池到底够不够、好动作的未来路径到底能不能被训练当下合法识别、以及是否应该继续停止 generated route。
>
> 本文件公式均使用 `$...$` 或 `$$...$$`，Typora 友好。

---

# 1. 对 v9.8.8 的独立判断

v9.8.8 的路线是：

```text
route = R7-EngineeringBlocked
primary_blocker = natural_extension_generator_missing
secondary_blocker = low_cost_proxy_failed
system_legal_controller_pass = 0
generated_route_status = stopped_no_legal_mechanism_or_density_evidence
```

这轮有进展，但不是能力成功。它没有打开 controller、selected runtime、paired replay、short/full training。它真正做的是把四线的边界压得更清楚：

```text
A 线：未来路径现象继续存在，但 strong/controller 仍未过。
B 线：低成本合法 proxy v2 仍失败。
C 线：自然 AP0 扩流仍没有 generator entrypoint。
D 线：generated route 没有重开条件，继续停止。
```

我不完全接受把 v9.8.8 只读成 “EngineeringBlocked”。更准确地说：

$$
\boxed{
\text{未来好路径现象仍可信；真正阻断系统的是自然扩流入口缺失和训练当下合法低成本机制缺失。}
}
$$

也就是说，v9.8.8 没有证明未来路径思路失败。它证明的是：**我们还没有足够动作密度证据，也没有低成本合法代理能把好路径在训练当下认出来。**

---

# 2. 四线进展复盘

## 2.1 A 线：未来训练路径

A 线仍然是目前最有价值的科学线索。

v9.8.8 里：

```text
P4 future-path mechanism weak / strong = 1 / 0
Core77 RAUV LCB = 2.8166481132099395
```

关键组合结果包括：

```text
Core77:
  actions = 77
  RAUV LCB = 2.8166481132099395
  V1 LCB = 0.08815910625793319
  V240 LCB = 4.750885696264312
  longrisk UCB = 0.03827572901862327

Core77 + CoreExpansion10:
  actions = 87
  RAUV LCB = 2.710737520242875
  V1 LCB = 0.11702769816449304
  V240 LCB = 4.588476154696239
  longrisk UCB = 0.03389313880385762

Core77 + RiskCleanButLowValueTop10:
  actions = 87
  RAUV LCB = 3.2442709696296745
  V1 LCB = 0.16186559086365848
  V240 LCB = 5.462997655266469
  longrisk UCB = 0.03389313880385762

OldOnly:
  actions = 10
  RAUV LCB = 1.0287178879851673
  V1 LCB = -0.174770564086397
  V240 LCB = 2.0678046281088522
  longrisk UCB = 0.0

ExactOnly:
  actions = 68
  RAUV LCB = 1.5250480927868013
  V1 LCB = 0.13699866747303657
  V240 LCB = 2.922090918869295
  longrisk UCB = 0.0

RandomMatched:
  actions = 87
  RAUV LCB = 0.17625930908187082
  V1 LCB = 0.09405836253137967
  V240 LCB = 3.5922551893613908
  longrisk UCB = 0.8485412092119363
```

这里有两个很重要的结论。

第一，好路径不是 current-step loss descent。`OldOnly` 的 `V1 LCB` 是负的，但 `RAUV LCB` 和 `V240 LCB` 是正的。这继续支持我们的核心观点：

$$
\boxed{
\text{好 functional update 不是当前一步降 loss 最多，而是能把后续训练带到更好轨迹的小参数改动。}
}
$$

第二，`AUV / RAUV` 不能单独做 controller。因为有些动作未来输出变化也很大，但风险很高。`RandomMatched` 就是典型反例：它的 `V240 LCB` 不低，但 `longrisk UCB` 高达 `0.8485`。所以未来路径必须同时看：

```text
收益路径；
延迟收益；
longrisk；
bad/null；
memory/offdiag；
跨 dataset/family/template 稳定性；
runtime cost。
```

A 线当前状态是：

```text
future-path 现象可信；
但还没有形成训练当下可用、低成本、跨分布稳定的机制。
```

## 2.2 B 线：训练当下合法低成本代理

B 线这轮继续失败。

v9.8.8 的 best proxy 是：

```text
LC8_RAUV_surrogate_sketch:
  precision = 0.3218390804597701
  V LCB = -0.2633631421716069
  cost q90 = 0.0010509975254535675 ms
  weak / strong = 0 / 0
```

这个结果很清楚：

```text
成本已经足够低；
但选出来的动作质量不够；
V 是负的；
不能进 controller。
```

这不是“阈值差一点”。低成本 proxy 的本质问题是：它能测到一些函数响应、memory/hard-tail 响应或风险清理能力，但它没有测到真正的未来路径收益。

因此 B 线不能继续做：

```text
调 LC8 threshold；
调 LC2 / LC3 / LC4 排名；
只追 cost q90；
只追 AUV surrogate；
只看 proxy scatter 的相关性。
```

B 线下一步必须改成 **FuturePathOperator Sketch**，也就是用训练当下可算的低成本近似去估计：

$$
\Delta \mapsto \left( V_{20}, V_{80}, V_{240}, LongRisk, Memory, Offdiag \right)
$$

而不是只估计一次函数响应或一次 memory response。

## 2.3 C 线：自然 AP0 动作扩流

C 线是当前最硬 blocker。

v9.8.8 的结果是：

```text
natural_extension_action_generator_found = 0
entrypoint_count = 0
P2 1/16/256 new-natural-action preflight = not_run
P3 5000/10000/20000 density panel = not_run
PanelA CoreLike LCB/UCB = 0.021475240123524635 / 0.033333885489495195
```

这意味着我们仍然不知道：

```text
CoreLike 好动作真实密度是否 >= 0.03；
现有 2876 个 action 是不是样本量太小；
自然动作池扩大后，是否会稳定出现足够多好动作；
还是 AP0 自然动作源本身已经到上限。
```

这是当前最应该优先解决的问题。因为只要 C 线不落地，我们就一直会在 2876 个旧动作里反复讨论：

```text
Core77 是否够；
Core77 + 10 怎么补；
raw LDO 是否 backfill artifact；
should support-aware gate replace raw gate。
```

这些讨论都不能最终裁决。

## 2.4 D 线：生成新动作

D 线继续停止是正确的。

v9.8.8 显示：

```text
generated sandbox allowed = 0
generated route status = stopped_no_legal_mechanism_or_density_evidence
```

原因很简单：

```text
A 线只有事后 future path 现象；
B 线没有合法低成本机制；
C 线没有自然动作密度证据；
因此 D 线没有目标函数。
```

如果现在重开 generated route，就会回到过去 APG/APGA/APGF/APGT 的老问题：

```text
payload 合法；
branch-horizon 能跑；
但动作 value-negative / high-longrisk；
最后只是增加人工 primitive 名字。
```

---

# 3. 当前项目进度判断

从系统能力角度看，进度确实慢：

```text
controller = not_run
selected runtime = not_run
paired replay = not_run
short/full training = not_run
system_legal_controller_pass = 0
```

从科学定位角度看，v9.8.8 仍然有价值，因为它把四线优先级排清楚了：

```text
C 线必须先解决 natural extension generator；
B 线不能继续小修 cheap proxy；
A 线继续证明好路径存在，但不能替代机制；
D 线继续停止，不允许盲造动作。
```

我会把当前阶段判断为：

$$
\boxed{
\text{truth base 可信，未来好路径可信；但自然动作密度和训练当下可用机制都未闭合。}
}
$$

---

# 4. 当前真正卡在哪里

## 4.1 第一 blocker：自然扩流入口缺失

这不是普通工程小问题，而是科学裁决的前置条件。

我们需要知道：

$$
p_{core} = P(\text{CoreLike action under natural AP0 stream})
$$

到底是不是大于 `0.03`。

现在只有：

$$
\hat p_{core}=0.02677
$$

Wilson interval：

$$
[0.02148, 0.03333]
$$

这个区间跨过 `0.03`，所以不能判定 sufficient，也不能判定 insufficient。

如果 10000 / 20000 panel 后：

$$
LCB(p_{core}) \ge 0.03,
$$

existing-action harvesting 还有希望。

如果 20000 panel 后：

$$
UCB(p_{core}) < 0.03,
$$

那就说明 AP0 自然动作源不足，必须换动作生成方式。

## 4.2 第二 blocker：合法低成本机制失败

当前 B 线最强结果是 `LC8`，但它的 precision 只有 `0.3218`，V LCB 为负。这个结果告诉我们：

```text
便宜不等于有用；
风险响应不等于收益；
AUV surrogate 不等于好路径；
低成本 proxy 如果不理解未来路径，只会选错动作。
```

## 4.3 第三 blocker：好路径定义还没压缩成优化器规则

A 线显示好路径存在，但还没有回答：

```text
为什么 Core77 好？
为什么 OldOnly V1 负但后面好？
为什么 RandomMatched V240 不低却风险高？
为什么 RiskCleanButLowValue 可能被旧 label 低估？
```

这些不是描述性问题，而是 optimizer design 的核心。

我们需要把好路径压缩成类似：

$$
Accept(\Delta)=1
$$

当且仅当：

$$
FutureValue(\Delta)>0,
$$

$$
LongRisk(\Delta)\le \tau_L,
$$

$$
MemoryHarm(\Delta)\le \tau_M,
$$

$$
OffdiagRisk(\Delta)\le \tau_O,
$$

$$
Cost(\Delta)\le C_{max}.
$$

但现在最大问题是 `FutureValue` 还没有合法低成本估计。

---

# 5. 下一步 v9.8.9 的总目标

v9.8.9 的总体目标不是小修 `LC8` 或重开 APG。它要直接裁决三件事：

```text
1. 自然 AP0 动作池扩大后，好动作密度是否够；
2. 是否存在训练当下合法、低成本的 future-path proxy；
3. 如果前两者都失败，是否应该正式停止 AP0 existing-action route，转向新的 optimizer-level update generation。
```

一句话目标：

$$
\boxed{
\text{v9.8.9 要把“好路径存在”推进到“好路径能否被自然扩流、合法机制或新生成规则利用”。}
}
$$

---

# 6. v9.8.9 实验总览

v9.8.9 仍然四线并行，但优先级不同。

```text
C 线最高优先级：自然 AP0 扩流 materializer。
A 线继续：未来路径类型分解。
B 线重写：低成本 future-operator sketch，而不是 LC threshold。
D 线严格 gate：只有 A/B/C 给出证据，才允许 generated sandbox。
```

---

# 7. P0：复现 v9.8.8 边界

## 目标

确认本轮输入边界一致，不在旧 artifact 混乱上继续推进。

## 要记录

```text
source_route_v9880
natural_extension_action_generator_found_v9880
entrypoint_count_v9880
P3_density_inconclusive_v9880
P4_future_path_weak_pass_v9880
P5_low_cost_proxy_weak_pass_v9880
system_legal_controller_pass_v9880
generated_sandbox_allowed_v9880
no_fake/proxy/cpu_v9880
```

## 通过标准

```text
P0 pass iff:
  source_route_v9880 = R7-EngineeringBlocked
  natural_extension_action_generator_found_v9880 = 0
  P4_future_path_weak_pass_v9880 = 1
  P5_low_cost_proxy_weak_pass_v9880 = 0
  system_legal_controller_pass_v9880 = 0
  no_fake/proxy/cpu = 0/0/0
```

## 可视化

```text
fig_p0_v980_to_v988_route_timeline.svg
fig_p0_four_line_status_matrix.svg
```

---

# 8. P1：自然 AP0 extension action generator 落地

## 目标

把 C 线从“找不到入口”推进到“至少能真实生成新自然 AP0 action 并落 branch-horizon labels”。

注意，这里不是设计新的 APG primitive。这里要生成的是 **自然 AP0 stream action**：来自同一类训练流、同一类 AP0 action lifecycle、同一套合法 payload/action apply/outcome materializer，而不是人工新动作 family。

## 假设

$$
H_{P1}:
\text{repo 中可以建立一个 natural AP0 extension action generator，使新增 action 满足同一 lifecycle。}
$$

## 实验阶段

### P1a：entrypoint search

查找并登记：

```text
candidate generation function
train stream event source
action payload builder
action_id allocator
candidate_id allocator
payload hash writer
action apply replay hook
branch-horizon materializer hook
```

必须输出：

```text
entrypoint_count
generator_module
generator_function
required_inputs
required_outputs
missing_dependency_list
```

### P1b：single-action preflight

生成 1 个新 natural AP0 action，跑：

```text
payload write
payload hash
action apply replay
no-transform sanity
branch-horizon h1/h5/h20/h80/h240 if cheap; otherwise h20/h80/h240 first
```

### P1c：16-action preflight

生成 16 个动作，要求：

```text
unique action_id = 16
unique payload_hash >= 16
branch-horizon completion = 1.0
unresolved_exception = 0
label exclusivity violation = 0
```

### P1d：256-action smoke

生成 256 个动作，要求：

```text
branch-horizon rows expected = 256 * branch_count * horizon_count
actual = expected
rows/sec recorded
memory peak recorded
no fake/proxy rows
```

### P1e：5000 panel

只有 P1d 过后才跑。

## 要记录的指标

```text
natural_action_count
new_action_count
reused_action_count
duplicate_action_id_count
duplicate_payload_hash_count
payload_hash_missing_count
action_apply_linf_max
action_apply_cosine_min
branch_horizon_expected_rows
branch_horizon_actual_rows
rows_per_sec
wallclock_sec
peak_gpu_memory
unresolved_exception_count
label_exclusivity_violation_count
fake/proxy/cpu count
```

## 通过标准

```text
P1 weak pass iff:
  P1a entrypoint_count >= 1
  P1b single-action pass = 1
  P1c 16-action pass = 1
  no fake/proxy/cpu = 0

P1 strong pass iff:
  P1d 256-action smoke pass = 1
  branch_horizon_completion = 1.0
  action_apply_linf_max <= 1e-7
  unresolved_exception_count = 0
  duplicate_action_id_count = 0
```

## fail-fast 规则

如果 P1a 仍然找不到 entrypoint：

```text
route = R-C1-NaturalGeneratorEntryMissing
停止 P2/P3/P5/P6/P7/P8 的大实验；
只允许 engineering-only materializer task；
不再跑完整 science runner。
```

---

# 9. P2：自然动作密度曲线

## 目标

回答：自然 AP0 action stream 扩大后，CoreLike / PathGood / SlowBurnGood 的密度是否足够支撑 controller。

## Panel 设计

```text
PanelA-existing-2876
PanelB-natural-5000
PanelC-natural-10000
PanelD-natural-20000
```

每个 panel 都记录：

```text
action_count
new_action_count
CoreLike_count
PathGood_count
SlowBurnGood_count
RiskyHighAUV_count
SafeLowValue_count
GradeAB_count
ValuePositiveNoLongRisk_count
```

## 标签定义

### CoreLike

```text
CoreLike = 1 iff:
  precision label positive / GradeAB positive
  V_LCB > 0
  LongRisk_UCB <= 0.05
  Bad_UCB <= 0.05
  Null_UCB <= 0.15
  Memory_UCB <= 0.05
  Offdiag_UCB <= 0.05
```

### PathGood

```text
PathGood = 1 iff:
  RAUV_LCB > 0
  V240_LCB > 0
  LongRisk_UCB <= 0.05
  Memory_UCB <= 0.05
  Offdiag_UCB <= 0.05
```

### SlowBurnGood

```text
SlowBurnGood = 1 iff:
  V1_LCB <= 0
  RAUV_LCB > 0
  V80_LCB > 0
  V240_LCB > 0
  LongRisk_UCB <= 0.05
```

### RiskyHighAUV

```text
RiskyHighAUV = 1 iff:
  RAUV_LCB > 0
  LongRisk_UCB > 0.20 or Memory_UCB > 0.20 or Offdiag_UCB > 0.20
```

## 判断标准

### 密度足够

$$
LCB(p_{CoreLike}) \ge 0.03
$$

或：

$$
LCB(p_{PathGood}) \ge 0.03
$$

并且：

```text
min_dataset_rate >= 0.02
min_family_rate >= 0.01
LDO/LSO/LTO/LFO <= 0.10 under density-aware evaluation
```

### 密度不足

如果 PanelD 20000 后：

$$
UCB(p_{CoreLike}) < 0.03
$$

且：

$$
UCB(p_{PathGood}) < 0.03,
$$

则判定：

```text
natural AP0 action source density insufficient。
```

这时 existing-action harvesting 不能继续当主线。

## 可视化

```text
fig_p2_corelike_density_vs_panel_size.svg
fig_p2_pathgood_density_vs_panel_size.svg
fig_p2_slowburn_density_vs_panel_size.svg
fig_p2_density_wilson_ci.svg
fig_p2_dataset_family_rate_heatmap.svg
fig_p2_corelike_count_stacked_by_dataset.svg
fig_p2_path_type_pareto.svg
```

---

# 10. P3：未来路径类型机制分解

## 目标

不要只看 AUV。拆清不同动作类型为什么好或坏。

## 组别

```text
Core77
CoreExpansion10
Core77+CoreExpansion10
Core77+RiskCleanButLowValueTop10
OldOnly
ExactOnly
RandomMatched
NaturalNewCoreLike
NaturalNewPathGood
NaturalNewSlowBurnGood
NaturalNewRiskyHighAUV
```

## 每个 action 记录

```text
action_id
dataset
seed
family
template_id
step_bucket
path_type
horizon = 1,5,20,80,240
V_h
CE_delta_h
NLL_delta_h
Margin_delta_h
HardTail_CEp99_delta_h
Memory_loss_delta_h
Old_family_loss_delta_h
Old_stratum_margin_delta_h
LongRisk_h
Bad_h
Null_h
MemoryFail_h
OffdiagFail_h
CoverEntropy_delta_h
BasisRank_delta_h
CurvatureProxy_delta_h
Cost
```

## 核心统计

```text
AUV
RiskAdjustedAUV
DelayedGain = AUV - V1
TailRiskAUC
MemoryDamageAUC
OffdiagDamageAUC
PathMonotonicity
PathRecoveryIndex
```

定义：

$$
AUV(a)=\sum_{h\in\{1,5,20,80,240\}} w_h V_h(a)
$$

$$
RiskAdjustedAUV(a)=AUV(a)-\lambda_L LongRisk(a)-\lambda_M MemoryFail(a)-\lambda_O OffdiagFail(a)
$$

这里 `lambda` 不作为 controller 手工分数使用，只用于诊断不同路径类型。

## 判断标准

A 线 weak pass：

```text
Core77 RAUV_LCB > 0
Core77 LongRisk_UCB <= 0.05
Core77 Memory/Offdiag UCB <= 0.05
```

A 线 strong pass：

```text
至少一个 87-action 组合满足：
  RAUV_LCB > 0
  V240_LCB > 0
  LongRisk_UCB <= 0.05
  Bad_UCB <= 0.05
  Null_UCB <= 0.15
  Memory/Offdiag UCB <= 0.05
  LDO/LSO/LTO/LFO <= 0.10
```

## 可视化

```text
fig_p3_future_path_V_curves_by_group.svg
fig_p3_future_path_risk_curves_by_group.svg
fig_p3_delayed_gain_vs_longrisk.svg
fig_p3_RiskAdjustedAUV_vs_AUV.svg
fig_p3_slowburn_examples.svg
fig_p3_random_highAUV_failure_cases.svg
fig_p3_memory_offdiag_over_horizon.svg
```

---

# 11. P4：低成本 FuturePathOperator Sketch v3

## 目标

B 线不再调 LC1-LC8 阈值，而是重写为训练当下合法的未来路径近似。

核心问题：

$$
\boxed{
\text{能否用低成本训练当下信号预测一个动作的 RiskAdjustedAUV / PathGood，而不是预测一次函数响应？}
}
$$

## 候选 sketch

### FPS1：tiny virtual AdamW sketch

对候选 action $\Delta$ 做极少步虚拟更新，不进入 official training state：

```text
apply Δ to shadow weights
run 1-2 tiny AdamW-like virtual steps on micro-batch
measure memory/hard-tail response
restore weights
```

记录：

```text
virtual_step_count
virtual_batch_size
shadow_apply_ms
sketch_response
sketch_memory_harm
sketch_hardtail_harm
cost_q90
```

### FPS2：JVP/VJP path sketch

用 Jacobian-vector 或 vector-Jacobian product 估计 action 对未来 gradient field 的影响：

$$
J(\theta)\Delta
$$

和：

$$
\nabla_\theta L_{memory}^\top \Delta
$$

但不使用 future outcome label。

### FPS3：memory-hardtail coupled response

不再只看 memory 或 hard-tail 单独响应，而看二者是否同时过：

```text
current batch response positive
memory response non-negative
hard-tail response non-negative
old-family response non-negative
```

### FPS4：delayed-gain proxy

专门针对 SlowBurnGood：

```text
允许 V1 proxy 不强；
要求 h20/h80 的 virtual path proxy 增强；
要求 memory/offdiag 不坏。
```

### FPS5：two-stage accept sketch

第一阶段只选安全：

```text
LongRiskProxy <= threshold
MemoryProxy <= threshold
OffdiagProxy <= threshold
```

第二阶段选收益：

```text
FutureValueProxy positive
DelayedGainProxy positive
```

避免一个分数同时承担 value 和 risk。

## 成本门

```text
weak cost gate: cost_q90 <= 0.50 ms
strong cost gate: cost_q90 <= 0.10 ms
```

## 判断标准

B weak pass：

```text
TopK87 precision >= 0.60
V_LCB > 0
LongRisk_UCB <= 0.10
Memory/Offdiag UCB <= 0.10
cost_q90 <= 0.50 ms
```

B strong pass：

```text
TopK87 precision >= 0.75
V_LCB > 0
LongRisk_UCB <= 0.05
Bad_UCB <= 0.05
Null_UCB <= 0.15
Memory/Offdiag UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
cost_q90 <= 0.10 ms
```

## 可视化

```text
fig_p4_proxy_precision_vs_cost.svg
fig_p4_proxy_V_vs_longrisk.svg
fig_p4_proxy_score_vs_RiskAdjustedAUV.svg
fig_p4_proxy_slowburn_detection.svg
fig_p4_proxy_failure_case_waterfall.svg
fig_p4_value_proxy_and_risk_veto_pareto.svg
```

---

# 12. P5：controller gate

## 触发条件

只有满足以下任一条件才运行 controller：

```text
C 线 density sufficient 且 B 线 weak pass；
或 A 线 strong pass 且 B 线 strong pass；
或 C 线 density sufficient 且 existing-action accepted region 已经有合法 proxy 解释。
```

## Controller 形式

不允许使用：

```text
dataset_name
future outcome
old table label
CoreLike label
PathGood label
AUV label
red diagnostic score
```

允许使用：

```text
legal future-operator sketch fields
memory/hard-tail/current-batch response
cost fields
action norm / payload norm / AdamW alignment
pre-commit train-memory probes
```

## 通过标准

```text
accepted_count >= 87
coverage >= 0.03
precision >= 0.75
V_LCB > 0
LongRisk_UCB <= 0.05
Bad_UCB <= 0.05
Null_UCB <= 0.15
Memory/Offdiag UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
no fake/proxy/cpu = 0
```

---

# 13. P6：generated sandbox reopen gate

## 触发条件

只有以下情况允许 64-action generated sandbox：

```text
Case D1:
  C 线证明 natural AP0 density insufficient。

Case D2:
  B 线找到 strong future-operator proxy，可作为生成目标。

Case D3:
  A 线明确找到某种 path type 的机制，并且 B 线有合法近似。
```

否则：

```text
generated_sandbox_allowed = 0
```

## 生成目标

如果重开，不能再做 APG 小变体。必须直接优化：

$$
\Delta^* = \arg\max_{\Delta \in \mathcal{A}} FutureOperatorSketch(\Delta)
$$

subject to：

$$
MemoryHarm(\Delta)\le \tau_M
$$

$$
OffdiagRisk(\Delta)\le \tau_O
$$

$$
LongRiskProxy(\Delta)\le \tau_L
$$

$$
Cost(\Delta)\le C_{max}
$$

这里 $\mathcal{A}$ 不应该限定为 KAN 当前基函数名，而应该尽量是架构无关或弱架构相关的函数空间更新子空间，例如：

```text
last-layer function displacement subspace
low-rank output-response subspace
memory-safe residual subspace
hard-tail repair subspace
AdamW-compatible small residual subspace
```

## sandbox 通过标准

```text
generated_actions = 64
branch_horizon_completion = 1.0
GradeAB precision >= 0.50
V_LCB > 0
LongRisk_UCB <= 0.10
Memory/Offdiag UCB <= 0.10
```

如果 64-action sandbox 未过，不允许放大到 512-action。

---

# 14. P7：runtime boundary

只有 controller 过 P5 后运行。

记录：

```text
feature_compute_ms_q90
future_operator_proxy_ms_q90
controller_decision_ms_q90
payload_apply_ms_q90
step_ratio_q90
memory_ratio
kernel_count
sync_count
```

通过标准：

$$
step\_ratio_{q90}\le 1.50
$$

$$
memory\_ratio\le 1.05
$$

如果 future-operator sketch 成本高，必须回到 B 线，不允许用 runtime 例外通过。

---

# 15. P8：paired replay boundary

只有 P5 controller 和 P7 runtime 同时过才运行。

对照：

```text
RealFunctional
AdamWParallel
bestLR
NoOp
RandomPayload
ShuffledFunctionalPayload
```

通过标准：

```text
RealFunctional beats AdamWParallel / bestLR / NoOp / Random
Shuffled payload fail
V_LCB > 0
LongRisk_UCB <= 0.05
Bad_UCB <= 0.05
Null_UCB <= 0.15
LDO/LSO/LTO/LFO <= 0.10
```

---

# 16. v9.8.9 stop / pivot 规则

## 16.1 C 线 fail-fast

如果 P1a 仍找不到 natural extension generator：

```text
停止 science full run；
route = R-C1-NaturalGeneratorEntryMissing；
只允许 engineering-only materializer task。
```

## 16.2 密度不足 pivot

如果 PanelD 20000 后：

```text
CoreLike UCB < 0.03
PathGood UCB < 0.03
SlowBurnGood UCB < 0.03
```

则：

```text
existing-action harvesting 降级；
D 线 generated route 必须重置为 geometry-adaptive update generation。
```

## 16.3 B 线停止规则

如果 FPS1-FPS5 都满足：

```text
precision < 0.50 or V_LCB <= 0
```

则停止 low-cost proxy family，不再继续换阈值。

## 16.4 A 线停止规则

如果自然扩流后发现：

```text
future path 好动作密度不足；
且 Core77 / OldOnly 机制仍无法被任何 legal sketch 解释；
```

则：

```text
A 线只保留为 diagnostic；
不能再作为 controller 主线。
```

---

# 17. 必须产出的 artifact

```text
route_decision_v9890.json
run_manifest_v9890.json
p0_boundary_reproduction_v9890.csv
p1_natural_generator_entrypoint_audit_v9890.csv
p1_single_action_preflight_v9890.csv
p1_16_action_preflight_v9890.csv
p1_256_action_smoke_v9890.csv
p2_natural_density_panel_v9890.csv
p2_density_by_dataset_family_template_v9890.csv
p3_future_path_type_decomposition_v9890.csv
p4_future_operator_sketch_v9890.csv
p5_controller_gate_v9890.csv
p6_generated_sandbox_gate_v9890.csv
p7_runtime_boundary_v9890.csv
p8_paired_replay_boundary_v9890.csv
no_fake_audit_v9890.csv
field_legality_ledger_v9890.csv
failure_taxonomy_v9890.csv
```

---

# 18. 必须可视化清单

```text
fig_v9890_four_line_gate_matrix.svg
fig_p1_generator_lifecycle_flow.svg
fig_p1_branch_horizon_completion.svg
fig_p2_corelike_density_curve.svg
fig_p2_pathgood_density_curve.svg
fig_p2_density_CI_vs_panel_size.svg
fig_p2_dataset_family_density_heatmap.svg
fig_p3_future_path_V_curves.svg
fig_p3_future_path_risk_curves.svg
fig_p3_delayed_gain_vs_longrisk.svg
fig_p3_AUV_vs_RiskAdjustedAUV.svg
fig_p4_proxy_precision_cost_pareto.svg
fig_p4_proxy_score_vs_RAUV.svg
fig_p4_proxy_failure_cases.svg
fig_p5_controller_accepted_region_pareto.svg
fig_p6_generated_sandbox_gate.svg
fig_p7_runtime_component_stack.svg
fig_stop_pivot_matrix.svg
```

---

# 19. 本轮最终成功条件

v9.8.9 不要求 full functional success，但至少必须完成一个硬裁决。

## 成功路径 1：自然动作密度足够

```text
P1 strong pass = 1
P2 density sufficient = 1
P4 legal proxy weak/strong pass 至少 weak
P5 controller weak pass
```

## 成功路径 2：自然动作密度不足但生成目标出现

```text
P1 strong pass = 1
P2 density insufficient = 1
P4 future-operator sketch strong pass = 1
P6 generated sandbox weak pass = 1
```

## 成功路径 3：工程 blocker 被明确隔离

如果 P1a 找不到 generator，则不能再跑一轮完整 science runner，而要输出：

```text
engineering-only materializer task list
missing entrypoint list
required implementation contract
single-action smoke test spec
```

这也是有效推进，因为它防止继续在没有 C 线的情况下讨论 density 和 controller。

---

# 20. 最终判断

v9.8.9 的核心不是再证明 Core77 好，也不是再证明 LC proxy 差。我们已经知道这些。

v9.8.9 必须回答：

$$
\boxed{
\text{自然动作池是否足够？训练当下是否有合法低成本 future operator？如果两者都没有，是否应停止 existing-action route？}
}
$$

这才是当前项目继续前进所需的硬裁决。
