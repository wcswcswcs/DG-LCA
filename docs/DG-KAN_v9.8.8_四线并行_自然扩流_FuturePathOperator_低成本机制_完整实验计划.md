# DG-KAN v9.8.8 四线并行实验计划：自然动作扩流、未来路径算子、低成本合法机制、生成路线重开门槛

> 本文件基于 v9.8.7 实验复盘制定。v9.8.8 不继续小修 `LC1`、`FO8`、`AUV threshold` 或某个 APG 变体。v9.8.8 的目标是把四条线变成硬实验：
>
> - A 线：未来训练路径到底是不是好动作的核心现象；
> - B 线：训练当下能不能用低成本合法信号预测这种路径；
> - C 线：自然 AP0 动作流扩大以后，好动作密度到底够不够；
> - D 线：只有在 A/B/C 给出明确证据后，才允许重开生成新动作。
>
> 本计划所有公式使用 `$` 或 `$$`，Typora 友好。

---

## 0. v9.8.7 的独立判断

v9.8.7 的 route 是：

```text
route = CaseE-MaterializerExtensionGeneratorMissing
primary_blocker = natural_stream_extension_generator_missing
secondary_blocker = legal_low_cost_proxy_failed
route_recommendation = prioritize_C_materializer_engineering
system_legal_controller_pass = 0
generated_route_status = stopped_no_B_or_C_or_A_reopen_condition
```

我对这轮的判断是：**v9.8.7 有真实进展，但不是能力成功。**

它推进了两件硬事：

```text
1. existing AP0 natural preflight branch-horizon materializer 真实跑通：
   selected actions = 256
   expected rows = 7680
   actual rows = 7680
   P1a/P1b/P1c = 1/1/1

2. 低成本 proxy 真实评估了：
   best = LC1_CheapMemoryHardTailResponse
   cost q90 = 0.001122 ms
   precision = 0.160920
   V LCB = -0.289129
```

但它没有解决两个核心问题：

```text
1. natural extension action generator found = 0，
   因此 5000/10000/20000 panel 仍然不能跑。

2. low-cost legal proxy 质量失败，
   cheap 不等于 useful。
```

所以 v9.8.7 的真实结论不是：

```text
future path 方向错了。
```

而是：

$$
\boxed{
\text{已有动作的未来路径现象仍可信；但自然扩流和低成本合法识别都没有闭合。}
}
$$

---

## 1. 现在为什么感觉非常慢

这种慢是真实的。因为项目还没有进入下面这条链：

```text
selected controller
-> selected runtime
-> official paired replay
-> short/full training
```

最近几轮主要在做：

```text
1. 排除 one-step exact transfer；
2. 排除 windowed transfer 当前形式；
3. 排除 architecture-agnostic geometry score 当前形式；
4. 排除高成本 FO proxy；
5. 排除低成本 LC1 proxy；
6. 证明 future path 现象存在；
7. 但自然动作扩流还没真正跑起来。
```

这不是系统能力进展，但它是科学边界进展。现在最危险的不是慢，而是继续用一个没有机制的 proxy 或生成器往前冲。

---

## 2. v9.8.8 的一句话目标

v9.8.8 的总目标是：

$$
\boxed{
\text{判断好 functional update 是否可以通过自然动作扩流获得，或者必须转向几何自适应生成。}
}
$$

更直白地说，本轮必须回答：

```text
1. 自然 AP0 动作流扩大后，好动作够不够多？
2. 好动作的未来训练路径到底有什么共同形状？
3. 训练当下有没有低成本、合法的信号能提前预测这种形状？
4. 如果自然动作流不够，是否有足够机制证据重开生成新动作？
```

---

## 3. 关键概念解释

### 3.1 functional update action 是什么

一次普通训练步可以写成：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{AdamW}.
$$

functional update 是在普通 AdamW 更新之外，额外加一个小参数改动：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{AdamW}+\Delta\theta_{func}.
$$

这里的 $\Delta\theta_{func}$ 就是一个 action。

我们现在不是问：

```text
哪个 action 让当前 batch loss 立刻下降最多？
```

而是问：

```text
哪个 action 能把后续训练路径带到更好的位置？
```

---

### 3.2 未来训练路径是什么

对一个 action $a$，我们比较两条训练路径：

```text
路径 1：只用 AdamW 继续训练；
路径 2：先加 action a，再继续用 AdamW 训练。
```

在不同 horizon 上记录差异：

$$
h\in\{1,5,20,80,240\}.
$$

如果 action $a$ 是好动作，它不一定在 $h=1$ 立刻最好，但应该在 $h=20,80,240$ 让模型更稳、更有收益、更少 long-risk。

---

### 3.3 AUV 是什么

AUV 是把多个 horizon 的 value 合成一个路径面积。可以理解成：

```text
一个 action 在未来训练路径上的总收益。
```

记为：

$$
AUV(a)=\sum_{h\in H}w_h V_h(a).
$$

但 AUV 不能单独决定好坏。因为 RandomMatched 也可能有较高 AUV，但 long-risk、memory fail、offdiag fail 很高。

所以本轮会使用 risk-adjusted AUV：

$$
RAUV(a)=AUV(a)-\lambda_L LongRisk(a)-\lambda_M MemoryFail(a)-\lambda_O OffdiagFail(a).
$$

但注意：$RAUV$ 只作为诊断，不直接作为 official controller，除非它能通过合法性、LDO/LSO/LTO 和 runtime gates。

---

### 3.4 自然 AP0 动作扩流是什么

当前 canonical AP0 action universe 只有：

```text
2876 actions
```

其中 CoreLike count 是：

```text
77 actions
```

rate 是：

$$
77/2876=0.02677.
$$

coverage gate 大约需要：

$$
0.03\times2876\approx87
$$

个好动作。

现在的问题是：

```text
是动作池太小，所以只看到 77 个？
还是自然 AP0 动作本身好动作密度就是低于 0.03？
```

必须通过 5000 / 10000 / 20000 action panel 真实扩流来回答。

---

## 4. 四线总体设计

### A 线：未来路径机制分解

A 线回答：

$$
\boxed{
\text{好动作到底让后续训练路径哪里变好了？}
}
$$

不是只看 AUV，而是看完整路径：

```text
V1, V5, V20, V80, V240
AUV
RAUV
DelayedGain
LongRisk
Bad
Null
MemoryFail
OffdiagFail
HardTail
CoverEntropy
BasisRank
CurvatureProxy
```

A 线要比较至少六组动作：

```text
Core77：当前最干净的 77 个动作；
CoreExpansion10：用于补足到 87 的扩展动作；
Core77+RiskCleanButLowValueTop10：v9.8.6 的强组合路径；
OldOnly：OldRank 选中、ExactTransfer 没选中，但真实 outcome 好；
ExactOnly：ExactTransfer 选中、OldRank 没选中，但真实 outcome 差；
RandomMatched：按 dataset/family/template/step/norm 匹配的随机动作。
```

A 线成功不是说某个组 AUV 高，而是要看到：

```text
1. Core77 / OldOnly / 合理扩展组在 h20/h80/h240 稳定正收益；
2. longrisk / bad / null / memory / offdiag 同时低；
3. ExactOnly 或 RandomMatched 即使 AUV 高，也在风险维度明显失败；
4. 能找到好路径和坏路径的可解释差异。
```

---

### B 线：低成本合法未来路径代理

B 线回答：

$$
\boxed{
\text{训练当下能不能低成本预测 A 线的好路径？}
}
$$

v9.8.6 的 FO1-FO8 成本太高，v9.8.7 的 LC1 成本很低但质量差。所以 v9.8.8 不再继续调 LC1，而是重写低成本 proxy。

B 线必须满足三个约束：

```text
1. legal-at-commit：不能用未来 outcome，不能用 old table diagnostic，不能用 dataset name；
2. low cost：q90 成本必须足够低，目标是 <= 0.20 ms，强目标 <= 0.05 ms；
3. predictive：TopK87 不能只清风险，还要有正 value。
```

本轮测试四类低成本 proxy：

```text
B1. memory/hard-tail directional response：
    看 action 对 memory buffer 和 hard-tail samples 的一阶响应。

B2. function-displacement stability sketch：
    看 action 在多个样本上的输出变化是否稳定，不追求变化大。

B3. AdamW compatibility sketch：
    看 action 是否帮助后续 AdamW，而不是和 AdamW 冲突。

B4. mini virtual path sketch：
    用极小样本和 1-3 个虚拟更新步，近似未来路径趋势。
```

B 线通过标准：

```text
TopK87 precision >= 0.75
V LCB > 0
LongRisk UCB <= 0.05
Bad UCB <= 0.05
Null UCB <= 0.15
MemoryFail UCB <= 0.05
OffdiagFail UCB <= 0.05
LDO / LSO / LTO / LFO drop <= 0.10
cost q90 <= 0.20 ms
```

如果 B 线只能做到低风险但 V 为负，则它只能作为 veto，不是 controller。

---

### C 线：自然 AP0 动作扩流

C 线回答：

$$
\boxed{
\text{自然 AP0 动作池扩大后，好动作密度是否足够？}
}
$$

这是 v9.8.8 的最高优先级。v9.8.7 已经证明 existing AP0 preflight 能跑 256 actions / 7680 rows，但新自然扩展 action generator 缺失。

C 线不允许直接跑 5000 panel。必须 fail-fast ladder：

```text
C0: locate generator entrypoint
C1: 1 action preflight
C2: 16 action preflight
C3: 256 action smoke
C4: 5000 action panel
C5: 10000 action panel
C6: 20000 action panel
```

每一级都必须记录：

```text
generated natural action count
unique event count
unique candidate template count
payload hash missing
action apply error
expected branch-horizon rows
actual branch-horizon rows
rows/sec
wallclock
NaN/Inf count
duplicate row id count
label exclusivity violation
fake/proxy/cpu audit
CoreLike count/rate/LCB/UCB
PathGood count/rate/LCB/UCB
SlowBurnGood count/rate/LCB/UCB
per-dataset density
per-family density
per-template density
```

C 线密度判定：

```text
Density sufficient:
  CoreLike LCB >= 0.03
  or PathGood LCB >= 0.03
  and per-dataset support not degenerate

Density insufficient:
  CoreLike UCB < 0.03
  and PathGood UCB < 0.03
  after 20000 panel

Density inconclusive:
  CI crosses 0.03
```

如果 C 线 sufficient，优先推进 existing-action harvesting / controller。

如果 C 线 insufficient，当前自然 AP0 action source 不够，需要转 D 线生成新动作。

---

### D 线：生成新动作重开门槛

D 线回答：

$$
\boxed{
\text{什么时候才允许重新生成新 functional update？}
}
$$

v9.8.8 继续默认停止 generated route。

只在以下情况之一成立时，允许 64-action sandbox：

```text
Condition D1:
  B 线发现 legal low-cost proxy 过 weak gate，
  可以把它写成生成目标。

Condition D2:
  C 线证明自然动作密度不足，
  且 A 线已经给出明确 PathGood 机制。

Condition D3:
  A 线发现一个明确可生成的路径机制，
  例如 delayed gain + memory safe + offdiag safe 的组合能稳定解释 Core77 / OldOnly。
```

D 线初始只允许：

```text
64 generated actions
horizons = 1,5,20,80,240
branches = 6
expected rows = 64 * 6 * 5 = 1920
```

D 线通过标准：

```text
PathGood precision >= 0.30 for smoke
V240 LCB > 0
RAUV LCB > 0
LongRisk UCB <= 0.10
MemoryFail UCB <= 0.10
OffdiagFail UCB <= 0.10
new positive created rate >= 0.10
longrisk created rate <= 0.20
```

如果 64-action sandbox 失败，不允许扩展到 512。

---

## 5. v9.8.8 详细实验阶段

---

## P0：复现 v9.8.7 boundary

### 目标

确认 v9.8.7 的核心边界仍成立，不在已经失败的分数上继续小修。

### 必须记录

```text
source_route_v9870
P1a_single_preflight_pass
P1b_16_preflight_pass
P1c_256_smoke_pass
P1d_5000_panel_weak_pass
P2_density_inconclusive
P3_A_line_weak_pass
P3_A_line_strong_pass
P4_low_cost_proxy_weak_pass
P4_low_cost_proxy_strong_pass
system_legal_controller_pass
generated_sandbox_allowed
no_fake_rows
no_proxy_rows
no_cpu_offload_rows
```

### 判断标准

P0 pass：

```text
source_route_v9870 = CaseE-MaterializerExtensionGeneratorMissing
P1c_256_smoke_pass = 1
P2_density_inconclusive = 1
P4_low_cost_proxy_strong_pass = 0
system_legal_controller_pass = 0
```

---

## P1：自然动作扩流入口定位

### 目标

找到能产生新自然 AP0 actions 的 generator entrypoint。不是 replay 旧 AP0 actions，而是在同一训练逻辑下继续产生新的自然 candidate/action rows。

### 假设

$$
H_{C0}: \text{repo 中存在可审计的 natural AP0 extension generator。}
$$

### 必须记录

```text
entrypoint_file
entrypoint_function
entrypoint_config
source_train_stream_id
action_generation_mode
old_action_reuse_flag
new_action_generation_flag
new_event_id_count
new_candidate_id_count
new_action_id_count
payload_hash_missing
candidate_template_id_available
```

### 判断标准

P1 pass：

```text
natural_extension_action_generator_found = 1
new_action_generation_flag = 1
old_action_reuse_flag = 0
payload_hash_missing = 0
```

P1 fail-fast：

```text
natural_extension_action_generator_found = 0
```

如果 P1 fail，本轮不能继续讨论 5000/10000/20000 density，只能进入工程修复任务。

---

## P2：自然扩流 1/16/256 action preflight

### 目标

确认新自然 actions 可以真实落盘、真实 replay、真实生成 branch-horizon rows。

### 必须记录

```text
panel_size
new_action_count
unique_event_count
unique_candidate_template_count
action_apply_error_linf_max
payload_hash_missing
expected_rows
actual_rows
branch_completion
horizon_completion
rows_per_sec
wallclock
NaN_count
Inf_count
duplicate_row_count
label_exclusivity_violation_count
no_fake_rows
no_proxy_rows
no_cpu_offload_rows
```

### 判断标准

对 $N\in\{1,16,256\}$：

$$
ActualRows = N\times 6\times 5.
$$

P2 pass：

```text
actual_rows = expected_rows
branch_completion = 1.0
horizon_completion = 1.0
payload_hash_missing = 0
action_apply_error_linf_max <= 1e-6
NaN/Inf/duplicate/label violation = 0
no_fake/proxy/cpu = 1
```

---

## P3：自然扩流 5000 / 10000 / 20000 density panel

### 目标

判断自然动作池扩大后，好动作密度是否足够。

### 必须记录

```text
panel_size
action_count
rows_expected
rows_actual
CoreLike_count
CoreLike_rate
CoreLike_LCB
CoreLike_UCB
PathGood_count
PathGood_rate
PathGood_LCB
PathGood_UCB
SlowBurnGood_count
SlowBurnGood_rate
SlowBurnGood_LCB
SlowBurnGood_UCB
per_dataset_rate
per_family_rate
per_template_rate
max_dataset_share
max_family_share
max_template_share
rows_per_sec
wallclock
```

### 标签定义

CoreLike：

```text
GradeAB = 1
V LCB > 0
LongRisk UCB <= 0.05
Bad UCB <= 0.05
Null UCB <= 0.15
MemoryFail UCB <= 0.05
OffdiagFail UCB <= 0.05
```

PathGood：

```text
RAUV LCB > 0
V240 LCB > 0
LongRisk UCB <= 0.05
Bad UCB <= 0.05
Null UCB <= 0.15
MemoryFail UCB <= 0.05
OffdiagFail UCB <= 0.05
```

SlowBurnGood：

```text
V1 LCB <= 0 allowed
AUV LCB > 0
V80 LCB > 0
V240 LCB > 0
LongRisk UCB <= 0.05
MemoryFail UCB <= 0.05
OffdiagFail UCB <= 0.05
```

### 判断标准

Density sufficient：

$$
LCB(CoreLikeRate)\ge0.03
$$

or

$$
LCB(PathGoodRate)\ge0.03.
$$

Density insufficient：

$$
UCB(CoreLikeRate)<0.03
$$

and

$$
UCB(PathGoodRate)<0.03.
$$

Inconclusive：

```text
confidence interval crosses 0.03
```

---

## P4：未来路径类型机制分解

### 目标

解释 Core77、OldOnly、Expansion、RiskCleanButLowValue 为什么有不同未来路径。

### 必须记录

```text
action_group
action_count
V1_LCB
V5_LCB
V20_LCB
V80_LCB
V240_LCB
AUV_LCB
RAUV_LCB
DelayedGain_LCB
LongRisk_UCB
Bad_UCB
Null_UCB
MemoryFail_UCB
OffdiagFail_UCB
HardTailDelta_UCB
CoverEntropyDelta
BasisRankDelta
CurvatureProxyDelta
```

### 关键指标

DelayedGain：

$$
DelayedGain(a)=V_{240}(a)-V_1(a).
$$

RiskAdjustedAUV：

$$
RAUV(a)=AUV(a)-\lambda_L LongRisk(a)-\lambda_M MemoryFail(a)-\lambda_O OffdiagFail(a).
$$

### 判断标准

P4 weak pass：

```text
Core77 RAUV_LCB > 0
Core77 LongRisk_UCB <= 0.05
Core77 Memory/Offdiag UCB <= 0.05
OldOnly immediate-not-dominant evidence remains present
RandomMatched risk is clearly higher than Core77
```

P4 strong pass：

```text
At least one 87-action group has:
  RAUV_LCB > 0
  V240_LCB > 0
  LongRisk/Bad/Null/Memory/Offdiag all pass
  LDO/LSO/LTO/LFO drop <= 0.10
```

---

## P5：低成本 future-operator proxy v2

### 目标

寻找训练当下合法、低成本、能预测未来路径的 proxy。

### 测试 proxy family

```text
LC2_memory_hardtail_directional_response
LC3_function_displacement_stability
LC4_AdamW_compatibility_sketch
LC5_cross_sample_sign_consistency
LC6_mini_virtual_path_1step
LC7_mini_virtual_path_3step
LC8_RAUV_surrogate_sketch
```

### 必须记录

```text
proxy_id
legal_status
feature_count
cost_mean_ms
cost_q90_ms
TopK87_precision
TopK87_V_LCB
TopK87_AUV_LCB
TopK87_RAUV_LCB
TopK87_LongRisk_UCB
TopK87_Bad_UCB
TopK87_Null_UCB
TopK87_MemoryFail_UCB
TopK87_OffdiagFail_UCB
LDO_drop
LSO_drop
LTO_drop
LFO_drop
```

### 判断标准

P5 weak pass：

```text
TopK87_precision >= 0.65
V_LCB > 0
LongRisk_UCB <= 0.10
cost_q90_ms <= 0.20
```

P5 strong pass：

```text
TopK87_precision >= 0.75
V_LCB > 0
RAUV_LCB > 0
LongRisk_UCB <= 0.05
Bad_UCB <= 0.05
Null_UCB <= 0.15
Memory/Offdiag_UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
cost_q90_ms <= 0.20
```

If all proxies fail with precision < 0.40 or V_LCB < 0, stop low-cost proxy patching.

---

## P6：existing-action controller gate

### 目标

如果 C 线 density sufficient 或 B 线 strong pass，则尝试最小 controller。

### Controller 输入

```text
legal proxy score
risk veto
memory veto
offdiag veto
cost veto
candidate template diversity check
```

### 必须记录

```text
accepted_count
coverage
precision
V_LCB
AUV_LCB
RAUV_LCB
LongRisk_UCB
Bad_UCB
Null_UCB
MemoryFail_UCB
OffdiagFail_UCB
LDO_drop
LSO_drop
LTO_drop
LFO_drop
max_dataset_share
max_family_share
max_template_share
feature_cost_q90
payload_apply_cost_q90
```

### 判断标准

P6 pass：

```text
accepted_count >= 87
precision >= 0.75
V_LCB > 0
LongRisk_UCB <= 0.05
Bad_UCB <= 0.05
Null_UCB <= 0.15
Memory/Offdiag_UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
feature_cost_q90 <= 0.20 ms
```

---

## P7：generated route reopen gate

### 目标

决定是否允许 D 线生成新动作。

### 判断标准

Generated sandbox allowed if and only if：

```text
(P5 weak pass = 1 and P4 weak pass = 1)
OR
(P3 density insufficient = 1 and P4 strong mechanism evidence = 1)
```

否则：

```text
generated_sandbox_allowed = 0
reason = no_legal_mechanism_or_density_evidence
```

---

## P8：64-action generated sandbox only if allowed

### 目标

如果 P7 允许，做最小生成测试，不直接扩到 512。

### 必须记录

```text
generated_action_count
payload_hash_missing
certificate_hash_missing
action_apply_error_linf
expected_rows
actual_rows
PathGood_precision
V240_LCB
AUV_LCB
RAUV_LCB
LongRisk_UCB
Bad_UCB
Null_UCB
MemoryFail_UCB
OffdiagFail_UCB
new_positive_created_rate
longrisk_created_rate
rows_per_sec
```

### 判断标准

P8 weak pass：

```text
PathGood_precision >= 0.30
V240_LCB > 0
LongRisk_UCB <= 0.10
longrisk_created_rate <= 0.20
```

P8 strong pass：

```text
PathGood_precision >= 0.50
V240_LCB > 0
RAUV_LCB > 0
LongRisk/Bad/Null/Memory/Offdiag pass
new_positive_created_rate >= 0.10
longrisk_created_rate <= 0.10
```

If P8 fails, do not expand generated route.

---

## P9：selected runtime only if controller exists

### 目标

只有 P6 或 P8 过线后，才测 selected runtime。

### 必须记录

```text
selected_controller_id
accepted_count_online
feature_cost_q90
certificate_cost_q90
payload_apply_cost_q90
step_time_q50
step_time_q90
step_ratio_q90
memory_ratio
extra_kernel_count
extra_sync_count
```

### 判断标准

Runtime pass：

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
no CPU offload
no fake/proxy rows
```

---

## P10：paired replay / short-full boundary

### 目标

只有 P6/P8 + P9 都过线后才打开。

### 必须记录

```text
RealFunctional
AdamWParallel
BestLR
NoOp
Random
ShuffledPayload
paired_delta_CE
paired_delta_NLL
paired_delta_margin
paired_delta_acc
paired_runtime
```

### 判断标准

Paired replay pass：

```text
RealFunctional beats AdamWParallel / BestLR / NoOp / Random
ShuffledPayload fails
paired V_LCB > 0
longrisk/bad/null pass
```

---

## 6. 可视化清单

必须生成：

```text
1. route waterfall: v9.8.2 -> v9.8.8
2. C-line preflight ladder: 1/16/256/5000/10000/20000 rows expected vs actual
3. CoreLike / PathGood / SlowBurnGood density curve with Wilson CI
4. per-dataset / per-family / per-template density heatmap
5. A-line future path curves: V1/V5/V20/V80/V240 by group
6. AUV vs longrisk scatter
7. RAUV vs GradeAB scatter
8. DelayedGain distribution by group
9. B-line cost-quality Pareto
10. proxy precision vs V_LCB plot
11. proxy cost_q90 vs step budget plot
12. B-line legal proxy failure taxonomy
13. D-line generated sandbox outcome chart if run
14. no-fake/no-proxy/cpu audit table
15. controller gate dashboard if P6 opens
```

---

## 7. 并行执行策略

### Batch 0：立即执行

```text
P0 boundary reproduction
P1 generator entrypoint search
P4 future path mechanism decomposition using landed rows
```

### Batch C：最高优先级工程线

```text
P2 1/16/256 natural extension preflight
P3 5000/10000/20000 density panels
```

### Batch B：低成本机制线

```text
P5 LC2-LC8 proxy evaluation
cost-quality Pareto
failure taxonomy
```

### Batch D：条件触发

```text
P7 generated reopen decision
P8 generated 64-action sandbox only if allowed
```

### Batch System：条件触发

```text
P6 controller
P9 runtime
P10 paired replay / short-full boundary
```

---

## 8. Stop / Pivot 规则

### Stop C-line discussion if generator missing

```text
natural_extension_action_generator_found = 0
```

then：

```text
route = R-C0-NaturalExtensionGeneratorMissing
next = engineering-only materializer task
```

### Stop existing-action route if density insufficient

```text
CoreLike_UCB < 0.03
and PathGood_UCB < 0.03
after 20000 panel
```

then：

```text
route = R-C2-NaturalAP0DensityInsufficient
next = generated route requires new objective
```

### Stop low-cost proxy patching

```text
best legal low-cost proxy precision < 0.40
or V_LCB < 0
or cost_q90 > 0.20 ms for all useful proxies
```

then：

```text
route = R-B2-LowCostProxyNoSignal
next = do not tune LC threshold; seek new mechanism
```

### Stop generated sandbox

```text
PathGood_precision < 0.10
or V240_LCB < 0
or LongRisk_UCB > 0.50
or longrisk_created_rate > 0.50
```

then：

```text
route = R-D2-GeneratedStillBad
next = stop generated primitive variants
```

---

## 9. 预期 route 格式

```text
R1-CNaturalDensitySufficient:
  natural stream density LCB >= 0.03, existing-action route continues.

R2-CNaturalDensityInsufficient:
  natural stream density UCB < 0.03 at 20000, existing AP0 source insufficient.

R3-BLegalProxyStrong:
  low-cost legal proxy passes precision/value/risk/cost/leaveout gates.

R4-AFuturePathMechanismOnly:
  future path signal is strong but no legal proxy or density evidence.

R5-GeneratedSandboxAllowed:
  B or C/A evidence allows 64-action generated sandbox.

R6-SystemControllerCandidateReady:
  accepted region and runtime gates ready for paired replay.

R7-EngineeringBlocked:
  natural extension generator missing; no more density claims allowed.
```

---

## 10. 本轮最重要的一句话

v9.8.8 不是为了继续证明 “future path 看起来不错”。这个已经有多轮证据。

v9.8.8 必须回答：

$$
\boxed{
\text{自然动作池是否足够大到能提供好路径动作？如果不能，训练当下是否有合法低成本机制来生成或选择它们？}
}
$$

如果 C 线仍然不能落地，那么项目会继续非常慢，因为所有 density / official gate 讨论都会停在猜测。

如果 B 线仍然失败，则不能继续堆低成本 proxy。

如果 A 线继续强但 B/C 都失败，那么未来路径理论仍有科学价值，但还不是优化器。

