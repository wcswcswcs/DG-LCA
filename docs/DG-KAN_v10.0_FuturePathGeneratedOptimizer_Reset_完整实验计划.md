# DG-KAN v10.0 FuturePath-Generated Optimizer Reset 完整实验计划

> 本计划基于 v9.9.8 `Multi-Marginal Natural Sampling / FPO Generated Hard Gate` 的真实结果制定。v9.9.8 第一次把自然 AP0 扩流从“生成器不可用 / tail fidelity 不可信”推进到“G40 通过 major+tail fidelity，并在真实 5000 panel 上给出 natural density insufficient”。这意味着下一阶段不能继续把主资源花在从自然 AP0 action stream 里挤出少数好动作，而要把主线转成：用未来训练路径定义好 update，并尝试生成这种 update。

本计划继续遵守：不按 dataset 调参；不使用 outcome-derived field 作为 commit-time feature；不把 diagnostic 写成 official；不使用 fake/proxy/cpu offload；functional update 只能是 update rule，不修改 CE loss。

---

## 0. v9.9.8 的独立结论

v9.9.8 的 route 是：

```text
route = CaseC-NaturalDensityInsufficientGeneratedSandboxFail
primary_blocker = natural_density_insufficient
secondary_blocker = generated_64_sandbox_failed
system_legal_controller_pass = 0
generated_route_status = generated_64_sandbox_failed
```

我对它的独立解读是：

$$
\boxed{\text{自然 AP0 harvesting 这条线首次被真实密度证据压到边界；但生成路线也没有准备好。}}
$$

这不是 functional success。controller、selected runtime、paired replay、short/full training 仍然没有打开。但它是一个战略分叉点。

关键事实如下：

```text
P1 TK3H coarseness pass = 1
P1 tail groups = 89
P1 slow+risky mixed tails = 22
P2 G40-G48 candidate count = 9
P2 official / weak pass count = 4 / 4
best generator = G40-IPF-major-tail-raking-sampler
P4 largest completed panel = 5000
P4 density sufficient / insufficient / inconclusive = 0 / 1 / 0
CoreLike+SlowBurn UCB = 0.007370975315718785
P6 FPO v10 weak / strong = 0 / 0
best FPO = FPO10C-signal-reservoir-drift-diffusion-v2
best FPO precision = 0.011494252873563218
best FPO V LCB = -0.7536804607446157
P8 generated sandbox executed / pass = 1 / 0
generated actions = 64
branch rows = 1920 / 1920
Fast / Slow precision = 0.0 / 0.0
V / V240 LCB = -1.1452801937561405 / -1.0240962102829358
longrisk UCB = 0.9972365292495987
new-positive / longrisk-created rate = 0.0 / 0.015625
No-fake audit rows checked = 211098
fake / proxy / cpu = 0 / 0 / 0
```

这些数字说明三件事。

第一，C 线不再是“generator missing”或“tail fidelity 未闭合”。G40 已经让 natural panel 足够可信，至少可以对当前 natural AP0 extension source 给出密度判断。

第二，5000 panel 上 `CoreLike+SlowBurn UCB = 0.00737`，远低于 `0.03`。这不是“差一点”，而是强烈说明：在当前自然 AP0 action source 里，好路径动作太稀疏，无法支撑一个 coverage 足够的 official controller。

第三，D 线 64-action sandbox 被正确打开，但结果很差。它没有产生 FastGood 或 SlowBurnGood，value 和 V240 都显著为负，longrisk UCB 接近 1。这说明当前生成器并没有学会产生 future-path-improving update。

---

## 1. 四线当前状态

## 1.1 A 线：未来训练路径

A 线仍然是科学主线。过去几轮已经证明 Core77、OldOnly、Core+RiskClean 这类动作不是简单 immediate loss descent，而更像能把后续训练路径带到更好区域的小参数扰动。

v9.9.8 中 A 线不是主执行对象，但 route 里仍记录 `P5_future_path_weak_pass = 1`、`P5_future_path_strong_pass = 0`。这说明 future-path 现象仍然存在，但还没变成 controller。

A 线下一步不能继续只看 AUV 或 RAUV。必须把路径类型固定为可操作定义：

```text
FastGood:
  h1 已经明显好，h20/h80/h240 继续好，risk 低。

SlowBurnGood:
  h1 不强，甚至负，但 h20/h80/h240 逐步变好，risk 低。

RiskyHighAUV:
  AUV 或 V240 高，但 longrisk / memory / offdiag / bad/null 高。

SafeLowValue:
  risk 低，但 value 不够；可能是保守动作，也可能是长 horizon 后才有价值。

BadPath:
  value 低、risk 高、memory/offdiag 坏。
```

其中 `SlowBurnGood` 是最重要的对象。它最接近我们要找的 functional update：不是当前一步降 loss，而是改变未来训练轨迹。

A 线的核心假设是：

$$
H_A: \text{真正好的 functional update 是 future-path improving action，尤其是 SlowBurnGood / FastGood，而不是 immediate-loss descent。}
$$

---

## 1.2 B 线：训练当下 FuturePathOperator

B 线当前失败得非常明确。v9.9.8 的 FPO10C precision 只有 `0.01149`，V LCB 为 `-0.75368`。这不是“调阈值”能解决的问题。

当前 B 线失败原因不是成本，而是目标错位。过去 LC / FOS / FPO 系列大多在测：

```text
当前响应大小；
当前 risk hard gate；
SNR / drift / diffusion 的局部近似；
memory / hard-tail 的局部变化。
```

但它们没有测到：

```text
这个动作会不会把后续训练带到更好路径。
```

所以 B 线不能继续调 `FPO10C`。下一步必须改成 path-type predictor：训练当下预测一个动作像不像 `FastGood / SlowBurnGood / RiskyHighAUV / SafeLowValue / BadPath`。

B 线的核心假设是：

$$
H_B: \text{存在低成本 commit-time sketch，可以预测 action 的 future path type。}
$$

如果 $H_B$ 不成立，existing-action controller 就无法部署；generated route 也缺少可用目标。

---

## 1.3 C 线：自然 AP0 动作扩流

C 线在 v9.9.8 第一次给出强结论。G40 通过 major+tail fidelity 后打开真实 1024/5000 branch-horizon panel；5000 panel 给出 natural density insufficient。`CoreLike+SlowBurn UCB = 0.00737` 远低于 coverage gate 需要的 `0.03`。

这意味着：

$$
\boxed{\text{当前自然 AP0 action source 不能提供足够密度的好路径动作。}}
$$

这个结论比过去更强，因为过去很多轮都被 generator fidelity 或 tail fidelity 卡住，不能裁决 density；v9.9.8 已经把这个上游 blocker 基本推掉。

但 C 线还需要做两个确认，避免过早关闭：

```text
1. seed repeat：确认 5000 density insufficient 不是 seed / shard 偶然；
2. 10000 confirmation：如果 10000 panel UCB 仍远低于 0.03，则关闭 natural harvesting 主线。
```

C 线不应该继续大规模探索 20000 / 50000，除非 10000 出现反转。现在不需要再花大量资源证明一件已经高度倾向成立的事。

C 线的核心假设已经基本转为：

$$
H_C: \text{当前 natural AP0 source 的 good-path density 低于 official coverage 需求。}
$$

v9.9.8 强烈支持 $H_C$。

---

## 1.4 D 线：生成 future-path update

D 线在 v9.9.8 第一次被合法打开：因为 C 线显示自然密度不足，所以允许 64-action generated sandbox。这个 gate 是正确的。

但 D 线结果很差：

```text
Fast precision = 0.0
Slow precision = 0.0
V LCB = -1.1452801937561405
V240 LCB = -1.0240962102829358
longrisk UCB = 0.9972365292495987
new-positive rate = 0.0
longrisk-created rate = 0.015625
```

这说明当前 generated sandbox 的目标没有对准 future-path improvement。它不是简单“多生成一点就会好”。如果 64 个里 Fast/Slow 都是 0，value 又大幅负，那么扩大到 512/1024 只会浪费算力。

D 线下一步必须重置成“future-path-targeted generation”，而不是继续 old primitive family。

D 线的核心假设应该改成：

$$
H_D: \text{如果 natural AP0 source 密度不足，必须直接生成 future-path-improving update；生成目标必须来自 A 线 path type，而不是旧 APG primitive。}
$$

---

## 2. 当前真正卡在哪里

## 2.1 已经不主要卡在自然密度是否可裁决

v9.9.8 之前，我们一直不能判断自然 AP0 source 是否足够，因为 generator distribution fidelity 或 tail fidelity 不闭合。v9.9.8 之后，这个问题已经大幅推进：G40 过了 fidelity，5000 panel 也真实跑了。

现在不应该继续把大部分精力放在 C 线 tail sampler 小修上。C 线只需要做 seed repeat 与 10000 confirmation。

## 2.2 卡在“未来好路径如何训练当下识别”

B 线没有解决这个问题。FPO10C 几乎完全不能区分好路径动作。当前 FPO 的问题是目标错了：它在预测局部响应，不是在预测 path type。

## 2.3 卡在“未来好路径如何生成”

D 线 64-action sandbox 没产生任何 FastGood / SlowBurnGood。说明我们还没有 generation objective。自然 source 稀缺后，项目不能停在“自然 source 不够”，而要转向主动生成。

## 2.4 卡在“official gate 和 discovery gate 混在一起”

v9.9.8 以后必须区分：

```text
official route:
  极严，必须 controller/runtime/paired replay 全部过。

discovery route:
  可以小规模生成和探索，但不能写成 pass。
```

如果继续所有实验都被 official gate 串行控制，项目会继续慢。

---

## 3. v10.0 总体目标

v10.0 的目标不是继续修 v9.9.x 的单一 gate，而是重置为：

$$
\boxed{\text{从自然 harvesting 转向 future-path-targeted generated optimizer。}}
$$

最低有效推进是：

```text
1. C 线用 5000 repeat + 10000 confirmation 确认 natural density insufficient；
2. A 线固定 future path type label；
3. B 线训练 / 评估 path-type predictor，而不是 scalar FPO；
4. D 线做 8 -> 32 -> 64 的 future-path-targeted generated discovery sandbox；
5. 如果 D 线出现至少少量 SlowBurnGood / FastGood，再进入 256-action generated panel；
6. 如果 D 线仍然 0 Fast/Slow 且 V 显著负，则停止当前生成目标，回到 update rule 理论设计。
```

---

## 4. v10.0 实验结构总览

v10.0 分成三类 runner，并行执行。

### Runner 1：Official Gate Runner

只在满足严格条件时打开 controller / runtime / paired replay。

### Runner 2：Science Discovery Runner

允许非 official 小规模探索，所有结果必须标为 diagnostic，不允许写成 success。

### Runner 3：Engineering Fail-Fast Runner

只解决 materializer、payload、branch-horizon、throughput、collision、hash 等工程问题，不写长科学报告。

这三类 runner 必须分开。否则每轮都会因为 official gate 不过而卡住机制探索。

---

## 5. P0：v9.9.8 boundary 复现

### 目标

确认 v9.9.8 的核心结果可复现，避免基于错误输入继续。

### 必须记录

```text
P0_route
P0_G40_official_pass
P0_G40_major_psi
P0_G40_tail_psi
P0_largest_completed_panel
P0_CoreLike_count/LCB/UCB
P0_SlowBurn_count/LCB/UCB
P0_CoreLikePlusSlowBurn_UCB
P0_FPO10C_precision
P0_generated64_Fast_precision
P0_generated64_Slow_precision
P0_generated64_V_LCB
P0_generated64_V240_LCB
P0_generated64_longrisk_UCB
fake/proxy/cpu
```

### 判断标准

P0 pass：

```text
route = CaseC-NaturalDensityInsufficientGeneratedSandboxFail
G40 official pass = 1
largest_completed_panel >= 5000
CoreLike+SlowBurn UCB < 0.03
generated64 pass = 0
fake/proxy/cpu = 0/0/0
```

### 不满足时 Codex 先尝试

```text
如果 G40 不复现：
  检查 reference payload norm 对齐、G40 quota state、random seed、tail key TK3H definition。

如果 5000 panel 缺失：
  检查 chunk_actions、branch-horizon materializer resume、GPU OOM、row sink。

如果 fake/proxy/cpu 非 0：
  立即停止，不允许继续。
```

---

## 6. A 线：Future Path Type 固化

### 目标

把好路径从 AUV 单分数改成 path type 标签，作为 B/D 线的训练目标和评估目标。

### 假设

$$
H_A: \text{FastGood 和 SlowBurnGood 是比 CoreLike 更接近 functional update 本质的目标。}
$$

### 路径标签定义

对每个 action 记录：

```text
V1, V5, V20, V80, V240
AUV, RAUV
LongRisk240
Bad, Null
MemoryFail
OffdiagFail
HardTailDelta
LDO/LSO/LTO/LFO group tags
```

定义：

```text
FastGood:
  V1_LCB > 0
  V20_LCB > 0
  V80_LCB > 0
  V240_LCB > 0
  LongRisk_UCB <= 0.05
  Memory/Offdiag_UCB <= 0.05
  Bad_UCB <= 0.05

SlowBurnGood:
  V1_LCB <= 0 or weak
  V20_LCB > 0
  V80_LCB > 0
  V240_LCB > 0
  LongRisk_UCB <= 0.05
  Memory/Offdiag_UCB <= 0.05
  Bad_UCB <= 0.05

RiskyHighAUV:
  AUV_LCB > 0
  LongRisk_UCB > 0.20 or Memory/Offdiag_UCB > 0.20

SafeLowValue:
  LongRisk_UCB <= 0.05
  Memory/Offdiag_UCB <= 0.05
  V240_LCB <= 0

BadPath:
  V240_LCB < 0 or LongRisk_UCB > 0.50
```

### 必须记录

```text
action_id
source_group: Core77 / OldOnly / ExactOnly / RandomMatched / NaturalNew / Generated
path_type
dataset, seed, family, template, tail_group
V1/V5/V20/V80/V240
AUV/RAUV
LongRisk/Bad/Null/Memory/Offdiag
payload_norm, action_norm, adamw_cosine
branch-horizon hashes
```

### 判断标准

A 线 pass：

```text
FastGood_count + SlowBurnGood_count >= 87 in reference universe
or
FastGood/SlowBurnGood_count >= 30 for discovery target

FastGood/SlowBurnGood longrisk UCB <= 0.05
Memory/Offdiag UCB <= 0.05
not concentrated in single dataset/family/template/tail group
```

A 线强 pass：

```text
PathGood accepted_count >= 87
PathGood precision >= 0.75
V240 LCB > 0
LDO/LSO/LTO/LFO <= 0.10
```

### 可视化

```text
A_path_type_counts_bar.svg
A_V_curve_by_path_type.svg
A_Risk_curve_by_path_type.svg
A_dataset_family_template_heatmap.svg
A_AUV_vs_LongRisk_scatter.svg
A_SlowBurnGood_trajectory_examples.svg
```

### 不满足时 Codex 先尝试

```text
如果 SlowBurnGood 太少：
  放宽 V1 条件，不放宽 h240 risk；
  用 V80/V240 positive 替代 V20 strict；
  分别在 Core77 / OldOnly / RiskClean 子集里找。

如果 RiskyHighAUV 太多：
  强化 longrisk/memory/offdiag hard gate；
  不调 AUV threshold。

如果 SafeLowValue 多：
  追加 h480 diagnostic subset，判断是否更慢热；
  不把 h480 写成 official，只做 discovery。

如果 path type 集中在单一 group：
  输出 group concentration report；
  在 B/D 线中加入 group-balanced split。
```

---

## 7. B 线：FuturePathOperator v11，改成 path-type predictor

### 目标

训练当下预测 action 的 path type，尤其区分 SlowBurnGood / FastGood / RiskyHighAUV。

### 不再做什么

```text
不继续调 FPO10C threshold；
不继续只看 AUC；
不继续只预测 AUV / RAUV；
不继续用单一 response magnitude 当 score。
```

### 候选 sketch

```text
FPO11A-tiny-virtual-adamw-path-sketch:
  对 action-applied state 做 1-3 步 tiny virtual AdamW，只用 mini memory/hard-tail buffer。

FPO11B-jvp-vjp-future-gradient-alignment:
  估计 action 对下一步梯度方向的影响，判断是否让后续 AdamW 更容易优化。

FPO11C-signal-reservoir-path-type-snr:
  用 per-example response mean/variance + old-family consistency 预测 path type。

FPO11D-memory-offdiag-hard-gate-plus-delayed-gain:
  memory/offdiag 只做 hard gate，value 用 delayed gain proxy。

FPO11E-path-type-calibrated-ranker:
  多头预测 FastGood / SlowBurnGood / RiskyHighAUV / BadPath，而不是二分类 Good。
```

### 必须记录

```text
action_id
commit_time_features_used
feature_legality: green/yellow/red
feature_cost_ms_q50/q90
predicted_path_type_scores
predicted_fast_prob
predicted_slow_prob
predicted_risky_prob
predicted_bad_prob
accepted_by_rule
actual_path_type
actual_V240_LCB
actual_LongRisk_UCB
actual_memory/offdiag
```

### 判断标准

B discovery pass：

```text
TopK87 FastGood+SlowBurnGood precision >= 0.50
V240 LCB > 0
LongRisk UCB <= 0.10
cost q90 <= 5 ms
```

B weak official candidate：

```text
TopK87 FastGood+SlowBurnGood precision >= 0.75
V LCB > 0
V240 LCB > 0
LongRisk UCB <= 0.05
Bad UCB <= 0.05
Null UCB <= 0.15
Memory/Offdiag UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.15
cost q90 <= 1.5 ms
```

B strong pass：

```text
accepted_count >= 87
same metrics as weak
LDO/LSO/LTO/LFO <= 0.10
controller step_ratio_q90 <= 1.50 after P7 runtime
```

### 可视化

```text
B_path_type_confusion_matrix.svg
B_predicted_vs_actual_path_type_heatmap.svg
B_cost_vs_precision_scatter.svg
B_false_positive_trajectory_grid.svg
B_false_negative_slowburn_grid.svg
B_feature_ablation_waterfall.svg
```

### 不满足时 Codex 先尝试

```text
如果 precision 低但 risk 低：
  当前 sketch 只是 veto，不是 value selector；加入 delayed-gain proxy。

如果 value 正但 longrisk 高：
  加 memory/offdiag hard gate；不要调 value threshold。

如果 SlowBurn false negative 多：
  训练目标从 immediate gain 改为 V80/V240 delayed gain；
  增加 virtual path horizon 到 3 steps；
  加 gradient-alignment drift feature。

如果 RiskyHighAUV false positive 多：
  添加 longrisk/memory/offdiag hard gate；
  增加 hard-tail CEp99 delta。

如果 cost 高：
  降低 virtual samples；
  使用 cached JVP/VJP；
  使用 stratified mini-buffer；
  标记为 discovery-only，不允许 official controller。
```

---

## 8. C 线：自然密度确认与收尾

### 目标

确认 v9.9.8 的 natural density insufficient 是否稳定，并决定是否关闭 natural harvesting 主线。

### 假设

$$
H_C: \text{当前 natural AP0 source 的 FastGood/SlowBurnGood density 低于 official coverage 需求。}
$$

### 实验设计

```text
C1: repeat G40 5000 panel with seed 1314;
C2: repeat G40 5000 panel with seed 2027;
C3: if both UCB < 0.03, run one 10000 confirmation;
C4: if 10000 UCB < 0.03, close natural harvesting as primary route;
C5: only if contradiction appears, run 20000.
```

### 必须记录

```text
panel_id
seed
action_count
branch_rows_expected/actual
major_psi/js
tail_psi/js
missing_tail
CoreLike_count/LCB/UCB
FastGood_count/LCB/UCB
SlowBurnGood_count/LCB/UCB
PathGood_count/LCB/UCB
CoreLike+SlowBurn UCB
RiskyHighAUV_count
BadPath_count
throughput rows/sec
GPU memory peak
```

### 判断标准

C sufficient：

```text
PathGood or CoreLike+SlowBurn LCB >= 0.03
```

C insufficient：

```text
two 5000 repeats UCB < 0.03
and one 10000 confirmation UCB < 0.03
```

C inconclusive：

```text
LCB < 0.03 <= UCB
or fidelity not pass
or repeat seeds disagree strongly
```

### 可视化

```text
C_density_curve_2876_5000_10000.svg
C_seed_repeat_density_bar.svg
C_tail_group_density_heatmap.svg
C_path_type_density_stack.svg
C_major_tail_fidelity_dashboard.svg
```

### 不满足时 Codex 先尝试

```text
如果 G40 fidelity 不复现：
  检查 IPF convergence、quota raking residual、tail key mapping、payload norm alignment。

如果 branch rows 缺失：
  检查 chunk resume、OOM、row sink、duplicate row id。

如果 seeds 差异大：
  增加 2 个 2000 pilot；
  检查 generator state 是否周期性塌缩。

如果 density 回升到接近 0.03：
  暂缓关闭 natural route；
  追加 10000/20000。
```

---

## 9. D 线：Future-path-targeted generated discovery sandbox

### 目标

在 natural density 不足后，尝试主动生成 future-path-improving action。D 线分为 discovery 和 official 两层：

```text
D-discovery:
  可运行，但不得写成 controller/system pass。

D-official:
  必须等 B 或 D-discovery 过线，再打开 controller/runtime/paired replay。
```

### 生成策略

```text
GEN11A-GoodPathPrototypeImitation:
  从 Core77 / OldOnly / SlowBurnGood 提取 function-space displacement prototype；生成相似但不复制的 action。

GEN11B-ConstrainedDelayedGainUpdate:
  最大化 delayed-gain proxy，加入 memory/offdiag/longrisk hard gate。

GEN11C-JVPFutureGradientAlignmentUpdate:
  生成让下一步 AdamW gradient 更接近 future-good direction 的小更新。

GEN11D-SignalReservoirDenoisedUpdate:
  用 signal/reservoir split 过滤噪声方向，只保留 drift-supported direction。

GEN11E-NegativeControlShuffledPrototype:
  打乱 prototype 作为负控。
```

### 运行阶梯

```text
D0: 8-action preflight
D1: 32-action discovery sandbox
D2: 64-action discovery sandbox
D3: only if D2 weak pass, 256-action extended sandbox
```

### 必须记录

```text
generated_action_id
generator_family
source_prototype_id
payload_hash
certificate_hash
action_apply_linf
payload_norm
payload_cosine_to_source
payload_cosine_to_adamw
predicted_path_type
actual_path_type
FastGood / SlowBurnGood / RiskyHighAUV / BadPath label
V1/V5/V20/V80/V240
AUV/RAUV
longrisk/bad/null/memory/offdiag
new_positive_created
longrisk_created
source_to_generated_damage_V
branch_rows_expected/actual
```

### 判断标准

D0 preflight pass：

```text
action_apply_linf = 0
payload/certificate hash missing = 0
branch rows complete
fake/proxy/cpu = 0
```

D discovery weak pass：

```text
generated actions >= 64
FastGood+SlowBurnGood precision >= 0.10
at least 5 FastGood/SlowBurnGood actions
V240 LCB > 0 or not significantly negative
LongRisk UCB <= 0.20
longrisk_created_rate <= 0.20
new_positive_created_rate >= 0.05
```

D discovery strong pass：

```text
FastGood+SlowBurnGood precision >= 0.25
V LCB > 0
V240 LCB > 0
LongRisk UCB <= 0.10
memory/offdiag UCB <= 0.10
```

D official candidate：

```text
accepted_count >= 87
precision >= 0.75
V LCB > 0
V240 LCB > 0
LongRisk UCB <= 0.05
Bad UCB <= 0.05
Null UCB <= 0.15
Memory/Offdiag UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
```

### 可视化

```text
D_generated_path_type_distribution.svg
D_generated_V_curve_by_family.svg
D_payload_cosine_vs_path_type.svg
D_source_to_generated_damage_waterfall.svg
D_negative_control_comparison.svg
D_generated_longrisk_created_by_family.svg
```

### 不满足时 Codex 先尝试

```text
如果 8-action preflight fail：
  修 payload hash / certificate hash / action apply，不继续 science。

如果 Fast/Slow precision = 0：
  不扩大到 256；
  分析 false target：payload norm、cosine、gradient alignment、memory/offdiag damage；
  回到 generator objective。

如果 high AUV high risk：
  加 longrisk/memory/offdiag hard gate，不调 AUV threshold。

如果 low risk low value：
  检查是否 SafeLowValue；
  追加 h480 discovery subset，不写 official。

如果 all value negative：
  说明 generation target 错；
  停止当前 family，不继续 blend-ratio 调参。
```

---

## 10. P7：Existing / Generated Controller Boundary

### 目标

只有 B 或 D 满足 weak gate，才允许 controller boundary。否则保持 `not_run`。

### Controller 输入限制

```text
允许：commit-time legal features、FPO path-type sketch、payload/action norm、memory/offdiag hard gate、cost estimate。
禁止：future outcome、V/AUV/RAUV actual、dataset_name branch、path_type label at commit、old table score、generated branch-horizon outcome。
```

### 必须记录

```text
controller_id
source: existing / generated / hybrid
accepted_count
coverage
precision
V_LCB
V240_LCB
longrisk_UCB
bad_UCB
null_UCB
memory/offdiag_UCB
LDO/LSO/LTO/LFO
cost_q90
feature_legality_counts
```

### 判断标准

```text
accepted_count >= 87
precision >= 0.75
V_LCB > 0
V240_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory/offdiag_UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
```

### 不满足时 Codex 先尝试

```text
如果 accepted_count < 87：
  不降低 risk gates；
  检查 B/D source density；
  如果 natural density insufficient 且 generated density insufficient，停止 controller。

如果 precision 高但 LDO 高：
  做 group-balanced calibration，不允许 dataset-specific branch。

如果 risk 高：
  加 hard gate，不调 value threshold。

如果 value 负：
  controller 不打开 runtime。
```

---

## 11. P8：Selected Runtime Boundary

### 目标

只有 P7 controller 通过才测 runtime。否则 runtime not_run。

### 必须记录

```text
step_ratio_q50/q90/q99
feature_compute_ms_q90
FPO_compute_ms_q90
payload_apply_ms_q90
kernel_count_per_active_step
sync_count_per_active_step
memory_peak_MB
empty_step_kernel_count
active_step_count
```

### 判断标准

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
no CPU offload
no fake/proxy
```

### 不满足时 Codex 先尝试

```text
如果 feature/FPO compute 太慢：
  降低 virtual samples；
  使用 cached JVP/VJP；
  batch-major grouping；
  禁止把高成本 proxy 写入 official controller。

如果 payload apply 太慢：
  fuse payload apply；
  event-driven sparse apply；
  避免 empty-step launches。

如果 kernel/sync 过多：
  active-step compaction；
  batch-major scheduler；
  persistent workspace。
```

---

## 12. P9：Paired Replay Boundary

### 目标

只有 P7/P8 都通过，才打开 paired replay。

### 必须比较

```text
RealFunctional
AdamWParallel
bestLR
NoOp
RandomPayload
ShuffledPayload
GeneratedNegativeControl
```

### 必须记录

```text
paired_seed
paired_dataset
branch
h20/h80/h240 V
h20/h80/h240 accuracy
NLL/ECE/CEp99
longrisk
bad/null
memory/offdiag
runtime
```

### 判断标准

```text
RealFunctional beats AdamWParallel / bestLR / NoOp / Random
ShuffledPayload fail
GeneratedNegativeControl fail
V LCB > 0
longrisk UCB <= 0.05
```

---

## 13. 总体 route 决策

### Case A：Natural harvesting unexpectedly sufficient

```text
C repeat/10000 显示 PathGood LCB >= 0.03。
```

下一步：回到 existing-action controller，但必须用 path-type target，不用旧 OldRank。

### Case B：Natural density insufficient，FPO path-type predictor works

```text
C insufficient；B weak/strong pass。
```

下一步：用 B 线做 generated target 和 controller。

### Case C：Natural density insufficient，D-discovery works

```text
C insufficient；B 不过；D 64-action sandbox 出现 >=5 Fast/Slow actions。
```

下一步：D 扩到 256，学习 generated path mechanism。

### Case D：Natural density insufficient，B/D 都失败

```text
C insufficient；B 不过；D Fast/Slow precision = 0 or V strongly negative。
```

结论：当前 update action space / generation target 仍错误。停止 APG-style primitive，回到更高层 update-rule theory：future-path-constrained optimizer / natural-gradient-like function-space update / compositional base architecture。

### Case E：Runtime fails

```text
P7 controller pass, P8 runtime fail。
```

结论：科学 signal 存在，但系统不合格。转入 kernel/runtime engineering，不得写 functional success。

---

## 14. 本轮可视化总清单

```text
fig_A_path_type_trajectory_grid.svg
fig_A_slowburn_examples.svg
fig_B_path_type_confusion_matrix.svg
fig_B_cost_precision_scatter.svg
fig_C_density_repeat_curve.svg
fig_C_major_tail_fidelity_dashboard.svg
fig_D_generated_path_type_distribution.svg
fig_D_generated_damage_waterfall.svg
fig_D_negative_control_comparison.svg
fig_controller_gate_waterfall.svg
fig_runtime_step_timeline.svg
```

---

## 15. 加速执行策略

```text
Parallel Batch 1:
  P0 boundary reproduction
  C1/C2 5000 repeat
  A path-type labeling on landed rows

Parallel Batch 2:
  B FPO11A-E sketch on existing AP0 + landed path labels
  D0 8-action generated preflight

Parallel Batch 3:
  If C repeats confirm insufficient:
    C3 10000 confirmation
  If D0 passes:
    D1 32-action sandbox

Parallel Batch 4:
  Only if B or D passes:
    P7 controller
    P8 runtime
    P9 paired replay
```

禁止再做“一轮只推进一个 blocker”的 full-gated runner。Engineering、Science Discovery、Official Gate 必须分离。

---

## 16. 最终判断

v10.0 的核心不是继续证明 natural AP0 source 稀疏，也不是继续调 FPO10C。v9.9.8 已经给出战略转向信号：

$$
\boxed{\text{natural harvesting 不足；generated sandbox 初版失败；下一步必须做 future-path-targeted optimizer reset。}}
$$

如果 v10.0 仍然只是小修 sampler、FPO threshold 或 APG primitive，那么项目会继续慢。真正的推进是：

```text
用 A 线定义 future-path target；
用 B 线寻找训练当下 path-type predictor；
用 C 线快速确认 natural density insufficiency；
用 D 线小规模生成 future-path-targeted updates；
把 official gate 和 discovery gate 分开。
```
