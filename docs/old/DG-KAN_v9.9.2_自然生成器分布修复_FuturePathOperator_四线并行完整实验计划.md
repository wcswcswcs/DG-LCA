# DG-KAN v9.9.2 自然生成器分布修复 / FuturePathOperator / 四线并行完整实验计划

> 本文件基于 v9.9.1 `Natural Density Decision / FuturePathOperator` 的真实执行结果制定。v9.9.2 不继续小修 `FOS4`、不直接打开 5000/10000/20000 panel、不重开 generated sandbox，也不把 1024-action pilot 写成 density closure。v9.9.2 的核心任务是先修复自然 AP0 扩流生成器的分布保真度，再做密度裁决，同时继续在低成本、训练当下合法的 FuturePathOperator 上做机制验证。

---

## 0. 当前一句话判断

v9.9.1 的真实状态是：

```text
route = R-P1-NaturalGeneratorDistributionMismatch
primary_blocker = natural_generator_distribution_fidelity_failed
secondary_blocker = full_density_panels_blocked
system_legal_controller_pass = 0
generated_route_status = stopped_P1_distribution_fidelity_failed
```

v9.9.1 比 v9.9.0 前进了一步：v9.9.0 只证明 natural AP0 extension materializer 能通过 single / 16 / 256 action smoke；v9.9.1 进一步真实跑了 1024-action pilot，并且落盘了 `30720 / 30720` branch rows。

但是 v9.9.1 没有成功，因为 1024-action pilot 的分布严重不像 canonical AP0 reference：

```text
P1 distribution fidelity weak / strong = 0 / 0
PSI max = 13.018683555518681
max group share = 1.0
missing major groups = 17
```

同时，pilot 的好动作密度非常低：

```text
CoreLike count / LCB / UCB = 2 / 0.0005357696110014586 / 0.007093421501714742
PathGood count / LCB / UCB = 2 / 0.0005357696110014586 / 0.007093421501714742
```

但这个低密度不能直接用来判定 AP0 自然动作源不足，因为上游生成分布已经失败。更准确的判断是：

$$
\boxed{
\text{当前 natural generator 已能生成动作并 materialize outcome，但它采样的不是我们要评估的自然 AP0 分布。}
}
$$

所以 v9.9.2 的第一优先级不是扩大到 5000，而是修复分布。

---

## 1. v9.9.1 独立复盘

### 1.1 有进展，但不是能力进展

v9.9.1 的进展是真实的：

```text
1. v9.9.0 的 256-action smoke 被推进到 1024-action pilot；
2. 1024 pilot 的 branch-horizon rows 完整落盘；
3. P1 distribution fidelity gate 真正执行；
4. P2 density pilot 真实计算 CoreLike / PathGood；
5. P4 FuturePathOperator sketch 真实执行；
6. controller / generated / runtime / paired replay 全部正确 gate-blocked；
7. no fake / no proxy / no CPU offload 守住。
```

这说明当前不是 action apply、branch-horizon materializer、payload lifecycle、row sink 或 fake/proxy 的问题。

### 1.2 但它暴露了更硬的问题：生成器分布错了

v9.9.0 的问题是：

```text
natural extension generator 终于落地，但只到 256 smoke。
```

v9.9.1 的问题变成：

```text
natural extension generator 能跑到 1024，但生成分布严重不保真。
```

这比 “density 低” 更上游。因为如果生成器只覆盖一个 group 或少数 group，那么：

```text
1. CoreLike 低，不代表自然 AP0 源真的低；
2. PathGood 低，不代表未来路径好动作不存在；
3. 5000 / 10000 / 20000 panel 继续跑，只会更高成本地重复错误分布；
4. controller / generated sandbox 都没有科学意义。
```

因此 v9.9.1 的最大 insight 是：

$$
\boxed{
\text{自然扩流已经从“入口缺失”推进为“入口存在但分布失真”。}
}
$$

这是一种进展，但它解释了为什么用户会感觉慢：我们又发现了一个更前置的 gate。

---

## 2. 四线进展判断

### A 线：未来训练路径

A 线在 v9.9.1 没有新增科学证据，因为 P3 被 P1 distribution fidelity fail 阻断。

但是 A 线没有被证伪。前几轮已经显示 Core77、OldOnly、Core77+RiskClean 这些动作的未来路径很强。当前仍应保留判断：

```text
好 functional update 更像是把后续训练带到更好路径，而不是当前一步 loss descent。
```

v9.9.1 对 A 线的影响是：未来路径类型不能只在旧 PanelA 上分析，也必须在分布保真的 natural panel 上重新验证。否则我们不知道未来路径现象是 canonical AP0 旧池里的局部现象，还是自然扩流后仍然存在。

A 线当前状态：

```text
现象可信；本轮未推进；等待 C 线分布保真后重新评估。
```

### B 线：训练当下合法 FuturePathOperator

B 线在 v9.9.1 明确失败。

```text
P4 FuturePathOperator sketch weak / strong = 0 / 0
best = FOS4_RiskAdjustedHardGate
precision = 0.0
```

这个结果非常差，不是 threshold 小问题。它说明当前 FOS4 类 sketch 要么过度风险过滤，要么对未来收益完全没有捕捉能力。

B 线当前状态：

```text
当前 sketch 失败；不能继续调 FOS4；需要重新设计 future-path 代理。
```

B 线要从 “cheap response / risk hard gate” 转为 “低成本未来路径算子”。它要估计的是：

$$
\Delta\theta \rightarrow \text{后续训练路径变化}
$$

而不是：

$$
\Delta\theta \rightarrow \text{当前 batch 响应大小}
$$

### C 线：自然 AP0 动作扩流

C 线是 v9.9.1 的主线，也是当前最大 blocker。

v9.9.1 已经完成：

```text
1024-action pilot branch rows = 30720 / 30720
completion = 1
peak GPU MB = 5679.9873046875
```

但 C 线没过，原因是：

```text
P1 distribution fidelity weak / strong = 0 / 0
PSI max = 13.018683555518681
max group share = 1.0
missing major groups = 17
```

这意味着当前 natural generator 的输出很可能存在以下问题之一：

```text
1. cursor / sampler collapse，只从一个 group 取动作；
2. stratification 没有按 canonical AP0 reference 分布走；
3. 生成器复用了某个窄 provenance；
4. new action id 虽然唯一，但 action source family 不完整；
5. 某些 major groups 完全没有被采到；
6. natural extension recipe 与 canonical AP0 recipe 不是同一分布。
```

C 线当前状态：

```text
工程能跑；分布不可信；不能打开 5000/10000/20000。
```

### D 线：generated route

D 线继续停止是正确的。

v9.9.1 的 generated sandbox allowed = `0`。这不是保守过头，而是必要 gate：

```text
B 线没有合法机制；
C 线没有分布保真；
A 线没有在 natural panel 上重新验证；
因此 generated route 没有新目标。
```

D 线当前状态：

```text
继续停止；只允许在 A/B/C 之一给出机制或密度证据后，用 64-action sandbox 试验。
```

---

## 3. 当前真正卡在哪里

### 3.1 第一 blocker：自然生成器分布保真失败

这是 v9.9.1 的核心。1024-action pilot 已经够大，足以暴露分布问题。`PSI max = 13.0`、`missing major groups = 17`、`max group share = 1.0` 不是轻微漂移，而是生成器采样机制很可能塌缩。

现在不能继续说：

```text
自然 AP0 好动作密度低。
```

只能说：

```text
当前 natural generator 生成的不是有效自然 AP0 扩流分布。
```

### 3.2 第二 blocker：好动作密度仍未被裁决

P2 pilot 中 CoreLike 和 PathGood 都只有 2 个，UCB 都约 `0.00709`，远低于 `0.03`。如果这个分布是可信的，我们可以很快判定自然 AP0 源不足。

但它不可信，因为 P1 已经失败。

所以 density 仍然是：

```text
inconclusive
```

但注意，这个 inconclusive 已经不是 “样本太小”；而是 “样本分布不对”。

### 3.3 第三 blocker：FuturePathOperator sketch 选不出动作

FOS4 precision = `0.0`，意味着当前 sketch 不是可用线索。它也说明不能把 B 线当作替代 C 线的捷径。

如果 B 线想继续，必须换目标：不再预测 RAUV 或 risk-clean 单项，而是预测未来路径类型。

### 3.4 第四 blocker：四线还没有汇合成 controller

现在四条线都没有达到 controller 前置条件：

```text
A 线：future path 旧证据存在，但本轮未在 natural panel 更新；
B 线：sketch 失败；
C 线：generator distribution fail；
D 线：generated route 无条件重开风险太高。
```

因此 system controller pass = 0 是正确结果。

---

## 4. 进度如何

从系统能力看，进度仍然很慢：

```text
controller = not_run
selected runtime = not_run
paired replay = not_run
short/full training = not_run
generated sandbox = 0
```

从科学定位看，v9.9.1 是有效推进：

```text
v9.8.9：没有 natural extension generator；
v9.9.0：generator 落地，256 smoke 过；
v9.9.1：1024 pilot 过，但发现 generator distribution mismatch。
```

所以当前不是“完全没进度”，而是：

$$
\boxed{
\text{每一步都在推上游工程门；还没有推到 functional controller 门。}
}
$$

真正需要提速的是执行方式。v9.9.2 不应该再次完整跑一个大 runner 才发现分布没修好。应该把 P1 distribution fidelity 做成 fail-fast 工程门：

```text
1 action -> 16 actions -> 256 actions -> 1024 actions
```

每一级都必须检查分布，不只是检查 rows completion。

---

## 5. 对路线的独立判断

我同意 v9.9.1 route 停在 `R-P1-NaturalGeneratorDistributionMismatch`，但我会更明确地说：

$$
\boxed{
\text{v9.9.1 证明“当前 natural generator 不能用于密度裁决”，不是证明“自然 AP0 源密度不足”。}
}
$$

这个区分非常重要。

如果误读成 “density 低”，下一步可能会错误地重开 generated route；但如果正确理解为 “distribution mismatch”，下一步应该先修 generator。

---

## 6. v9.9.2 总体目标

v9.9.2 的目标是：

$$
\boxed{
\text{把 natural AP0 extension generator 从“能生成动作”修到“能生成分布保真的动作”，再裁决自然动作密度。}
}
$$

同时继续两条低成本科学线：

```text
A 线：用旧 landed future rows 和新 natural samples 做 future path 类型分解；
B 线：重写低成本 FuturePathOperator sketch，但不允许它绕过 C 线；
D 线：继续严格 gate。
```

---

# 7. v9.9.2 详细实验计划

## P0：复现 v9.9.1 边界

### 目标

确认 v9.9.1 的失败不是 artifact 读取错误。

### 输入

```text
v9.9.1 route_decision_v9910.json
p1_distribution_fidelity_v9910.csv
p2_pilot_density_v9910.csv
natural pilot action table
natural pilot branch-horizon outcome table
```

### 必须记录

```text
route
primary_blocker
secondary_blocker
P1_PSI_major_max
P1_max_group_share
P1_missing_major_groups
P2_pilot_action_count
P2_rows_expected
P2_rows_actual
CoreLike_count / LCB / UCB
PathGood_count / LCB / UCB
FOS4_precision
fake/proxy/cpu counts
```

### 通过标准

```text
P0_boundary_reproduced = 1
if:
  route == R-P1-NaturalGeneratorDistributionMismatch
  P1_PSI_major_max >= 10
  missing_major_groups >= 1
  natural pilot rows complete = 1
  fake/proxy/cpu = 0/0/0
```

### 如果失败，Codex 先尝试

```text
1. 检查 v9.9.1 artifact path 是否指向最新 run；
2. 检查 route_decision schema 是否变更；
3. 检查 p1_distribution_fidelity 表是否被重命名；
4. 修复 loader，不允许重跑 science runner；
5. 只在 P0 loader 通过后进入 P1。
```

---

## P1：natural generator 分布保真错误定位

### 目标

不是直接修 generator，而是先知道它为什么分布错。

### 假设

$$
H_{P1}:\text{v9.9.1 distribution mismatch 来自 generator 的 source/cursor/stratification collapse，而不是 natural AP0 源真实分布。}
$$

### 必须比较的分布轴

以 canonical AP0 PanelA / canonical AP0 full ledger 作为 reference，比较新 natural pilot：

```text
dataset_id
seed
family_id
stratum_id
step_bucket
event_family
candidate_template_id
candidate_origin
source_recipe_id
payload_norm_bucket
action_norm_bucket
action_linf_bucket
AdamW_alignment_bucket
memory_bucket
offdiag_bucket
hard_tail_bucket
horizon_source
branch_source
```

### 必须记录的指标

```text
axis
reference_group_count
pilot_group_count
missing_major_group_count
new_extra_group_count
max_group_share_reference
max_group_share_pilot
entropy_reference
entropy_pilot
entropy_ratio
PSI
KL
JS_distance
TV_distance
Wasserstein_for_continuous_axis
major_group_coverage
worst_missing_group
worst_overrepresented_group
```

### 分布保真弱通过标准

```text
P1_distribution_weak_pass = 1
if:
  missing_major_group_count == 0
  PSI_major_max <= 0.50
  JS_major_max <= 0.15
  max_group_share_pilot <= 0.35
  entropy_ratio_min >= 0.70
  old_action_collision == 0
  old_payload_collision == 0
```

### 分布保真强通过标准

```text
P1_distribution_strong_pass = 1
if:
  missing_major_group_count == 0
  PSI_major_max <= 0.20
  JS_major_max <= 0.08
  max_group_share_pilot <= 0.20
  entropy_ratio_min >= 0.85
```

### 失败解释规则

```text
F1-cursor-collapse:
  max_group_share_pilot = 1.0 或 dominant source_recipe/candidate_template 占比 > 0.80

F2-missing-major-groups:
  missing_major_group_count > 0

F3-wrong-reference:
  reference axis 口径错误，导致 base share 本身异常

F4-natural-recipe-narrow:
  generator 只覆盖少数 AP0 recipe，不是 full natural AP0 extension

F5-action-id-only-new-payload-not-natural:
  action_id 新，但 payload/provenance 分布不对
```

### 如果不满足条件，Codex 先尝试

```text
如果 F1-cursor-collapse:
  - 检查 cursor 是否每次从同一 event/source 起点开始；
  - 加入 cursor advancement trace；
  - 实现 round-robin source cursor；
  - 生成 16-action anti-collapse smoke。

如果 F2-missing-major-groups:
  - 输出 missing_major_groups.csv；
  - 为每个 missing group 生成 one-action targeted smoke；
  - 如果 targeted smoke 能生成，说明 sampler 没覆盖；
  - 如果 targeted smoke 不能生成，说明 recipe 缺失。

如果 F3-wrong-reference:
  - 重新生成 canonical reference distribution；
  - 确认 reference 不是旧 source panel 或 smoke panel；
  - 禁止用 v9.9.1 pilot 自己当 reference。

如果 F4-natural-recipe-narrow:
  - 增加 source_recipe_id；
  - 从 canonical AP0 action provenance 反推出 recipe classes；
  - 每个 recipe class 至少跑 1/16 action smoke。

如果 F5-action-id-only-new-payload-not-natural:
  - 检查 payload hash distribution；
  - 检查 payload norm / linf / cosine；
  - 检查 action apply replay；
  - 修复 payload construction，不允许只改 metadata。
```

### 可视化

```text
1. reference vs pilot group share bar chart；
2. PSI by axis waterfall；
3. entropy ratio by axis bar chart；
4. missing major groups heatmap；
5. source_recipe_id × event_family heatmap；
6. payload_norm_bucket × action_norm_bucket heatmap；
7. generator cursor trace timeline。
```

---

## P2：并行生成器修复候选

### 目标

不要只修一个 generator。并行产出多个候选 generator，并用同一 P1 fidelity gate 评估。

### 候选生成器

```text
G0-current-v9910-generator
  当前 generator 原样复现，作为 negative baseline。

G1-round-robin-source-cursor
  修复 cursor collapse，按 source_recipe / event_family / step_bucket 轮转。

G2-stratified-reference-quota-generator
  按 canonical AP0 reference 的 group quota 生成，但 action payload 必须新生成，不能复制旧 action row。

G3-recipe-complete-generator
  从 canonical AP0 provenance 中重建所有 recipe class，每类至少生成若干新 actions。

G4-randomized-trainstream-window-generator
  不用固定 source cursor，而从 train stream 不同 windows 随机采样事件生成 AP0-style action。

G5-hybrid-quota-random-generator
  70% quota-based + 30% random-window，防止过拟合 reference distribution。
```

### 每个候选必须运行的最小测试

```text
1-action smoke
16-action smoke
256-action smoke
1024-action fidelity pilot
```

### 每个候选必须记录

```text
generator_id
action_count
branch_rows_expected / actual
old_action_collision
old_payload_collision
action_apply_linf_max
rows_per_sec
wallclock
peak_gpu_mb
PSI_major_max
missing_major_groups
max_group_share
CoreLike_count / LCB / UCB
PathGood_count / LCB / UCB
fake/proxy/cpu
```

### 选择进入 P3 的规则

```text
进入 P3 的 generator 必须满足：
  distribution_weak_pass = 1
  action_apply_linf_max <= 1e-7
  branch completion = 1.0
  old_action_collision = 0
  old_payload_collision = 0
  fake/proxy/cpu = 0/0/0
```

如果多个 generator 通过，优先选择：

```text
1. PSI_major_max 最低；
2. missing_major_groups = 0；
3. max_group_share 最低；
4. entropy_ratio_min 最高；
5. rows/sec 高且 peak GPU 可控。
```

### 如果没有候选通过，Codex 先尝试

```text
1. 不跑 P3 full density；
2. 自动输出 generator_failure_matrix.csv；
3. 按失败类生成 repair task list；
4. 优先修 missing groups 与 cursor collapse；
5. 若 recipe-complete 仍失败，回到 AP0 action provenance schema，查 canonical action recipe 是否不可复现。
```

---

## P3：序贯 natural density panel

### 目标

在分布保真通过后，裁决自然 AP0 action stream 中好动作密度是否足够。

### 假设

$$
H_{P3}:\text{如果 generator 分布保真，CoreLike / PathGood 的真实密度可以通过 1024/5000/10000/20000 panel 裁决。}
$$

### Panel 顺序

```text
1024 -> 5000 -> 10000 -> 20000
```

必须严格序贯：

```text
1024 distribution fail -> stop and return P1/P2
1024 density inconclusive -> run 5000
5000 density inconclusive -> run 10000
10000 density inconclusive -> run 20000
20000 still inconclusive -> route to R-density-unresolved-high-cost
```

### 标签定义

不要只用 CoreLike。至少同时记录：

```text
CoreLike:
  strict clean path / clean outcome group。

PathGood:
  future path good，risk/bad/null/memory/offdiag clean。

SlowBurnGood:
  h1 或 h5 不强，但 h20/h80/h240 变好。

RiskyHighAUV:
  AUV 高但 longrisk / memory / offdiag 高。

SafeLowValue:
  风险干净但 value 不足。

BadPath:
  value 负、risk 高、bad/null 高或 memory/offdiag fail。
```

### 必须记录的指标

```text
panel_size
panel_id
generator_id
distribution_weak_pass
distribution_strong_pass
CoreLike_count / rate / Wilson_LCB / Wilson_UCB
PathGood_count / rate / Wilson_LCB / Wilson_UCB
SlowBurnGood_count / rate / Wilson_LCB / Wilson_UCB
RiskyHighAUV_count / rate
SafeLowValue_count / rate
BadPath_count / rate
per_dataset_rate
per_family_rate
per_template_rate
per_recipe_rate
LDO/LSO/LTO/LFO estimates
rows/sec
wallclock
peak_gpu_mb
```

### Density sufficient 标准

```text
density_sufficient = 1
if:
  distribution_weak_pass = 1
  CoreLike_LCB >= 0.03 or PathGood_LCB >= 0.03
  per-major-group lower support not collapsed
  longrisk_UCB <= 0.05 for accepted type
  bad_UCB <= 0.05
  null_UCB <= 0.15
  memory_UCB <= 0.05
  offdiag_UCB <= 0.05
```

### Density insufficient 标准

```text
density_insufficient = 1
if:
  distribution_weak_pass = 1
  CoreLike_UCB < 0.03
  PathGood_UCB < 0.03
  SlowBurnGood_UCB < 0.03
  panel_size >= 10000
```

### Density inconclusive 标准

```text
density_inconclusive = 1
if:
  confidence intervals cross 0.03
  or distribution weak pass but per-group support too small
```

### 如果不满足条件，Codex 先尝试

```text
如果 density_insufficient:
  - 不继续扩大到 20000 之外；
  - 输出 natural_density_failure_report；
  - 标记 AP0 natural source may be too sparse；
  - 进入 D 线 generated route reopen consideration，但必须先有 A/B 机制。

如果 density_inconclusive:
  - 先检查分布 fidelity 是否 degraded；
  - 若 fidelity 良好，扩大 panel；
  - 若 fidelity 变坏，回 P1/P2。

如果 per-group support collapsed:
  - 不是 density fail；
  - 回 P2 修 generator coverage。
```

### 可视化

```text
1. CoreLike / PathGood density vs panel size with Wilson CI；
2. SlowBurnGood density vs panel size；
3. density by dataset / family / recipe heatmap；
4. distribution fidelity vs density scatter；
5. PathGood vs RiskyHighAUV stacked bar；
6. rows/sec and GPU memory scaling chart。
```

---

## P4：A 线 future path 类型分解

### 目标

不再只看 AUV，而是把 future path 分成可解释类型。

### 输入动作组

```text
Core77
OldOnly
ExactOnly
RandomMatched
CoreExpansion10
RiskCleanButLowValue
NaturalCoreLike
NaturalPathGood
NaturalRiskyHighAUV
NaturalRandomMatched
```

其中 Natural* 只在 P3 panel 可用时加入。

### 必须记录的 horizon

```text
h1, h5, h20, h80, h240
```

### 必须记录的指标

```text
action_id
group_id
generator_id
dataset_id diagnostic only
family_id
template_id
recipe_id
V_h1 / V_h5 / V_h20 / V_h80 / V_h240
RAUV
DelayedGain = V_h240 - V_h1
RiskPath_UCB
LongRisk_h240
Bad_h
Null_h
MemoryFail_h
OffdiagFail_h
HardTail_delta_h
NLL_delta_h
CEp99_delta_h
MarginP10_delta_h
```

### Path type 定义

```text
FastGood:
  V_h1 > 0, V_h20 > 0, V_h240 > 0, risk clean。

SlowBurnGood:
  V_h1 <= 0 or V_h5 <= 0,
  but V_h20 > 0 and V_h240 > 0,
  risk/memory/offdiag clean。

RiskyHighAUV:
  RAUV high,
  but longrisk or memory/offdiag high。

SafeLowValue:
  risk/memory/offdiag clean,
  but V_h20/V_h240 not positive enough。

BadPath:
  V negative or bad/null/longrisk/memory/offdiag fail。
```

### A 线 weak pass

```text
future_path_weak_pass = 1
if:
  Core77 or NaturalPathGood has RAUV_LCB > 0
  and V240_LCB > 0
  and LongRisk_UCB <= 0.05
```

### A 线 strong pass

```text
future_path_strong_pass = 1
if:
  at least one accepted path type has accepted_count >= 87
  precision >= 0.75
  V240_LCB > 0
  RAUV_LCB > 0
  LongRisk_UCB <= 0.05
  Bad_UCB <= 0.05
  Null_UCB <= 0.15
  Memory_UCB <= 0.05
  Offdiag_UCB <= 0.05
  LDO/LSO/LTO/LFO <= 0.10
```

### 如果不满足条件，Codex 先尝试

```text
如果 RAUV 高但 risk 高:
  - 将该组标为 RiskyHighAUV；
  - 不允许用 AUV 做 controller score；
  - 进入 B 线 risk-adjusted sketch。

如果 SlowBurnGood 强:
  - 设计 B 线 delayed-gain proxy；
  - 不是继续 immediate transfer。

如果 NaturalPathGood 不存在但 Core77 存在:
  - 回 C 线看 density / generator distribution；
  - 不直接说 future path 理论失败。
```

### 可视化

```text
1. horizon curve: V_h by group；
2. RAUV vs LongRisk scatter；
3. DelayedGain histogram；
4. path type Sankey: action group -> path type；
5. memory/offdiag over horizon curves；
6. Natural vs canonical path type comparison。
```

---

## P5：B 线 FuturePathOperator Sketch v4

### 目标

用训练当下可计算的低成本 sketch 预测 future path type，而不是预测当前响应大小。

### 候选 sketch

```text
FPO1_tiny_virtual_adamw_2step
  在 train-memory / hard-tail microbatch 上做 2-step virtual update sketch。

FPO2_jvp_vjp_path_alignment
  用 JVP/VJP 估计 action 对未来 value direction 的投影。

FPO3_delayed_gain_proxy
  专门预测 SlowBurnGood：immediate response 不强但 delayed gain 正。

FPO4_memory_offdiag_hard_veto
  只作为 safety gate，不作为 value score。

FPO5_risk_adjusted_path_score
  value sketch + hard risk veto，不用加权大拼盘。

FPO6_two_stage_acceptor
  stage1 value sketch，stage2 memory/offdiag/longrisk hard veto。
```

### 必须满足的 legality

```text
no dataset_name branch
no future outcome field
no old table
no validation/test metric at commit time
no outcome-derived label
uses only current train stream / train-memory / legal payload/action fields
```

### 必须记录

```text
sketch_id
feature_groups_used
legality_green/yellow/red
compute_ms_q50/q90
memory_mb_delta
TopK87_precision
V_LCB
RAUV_LCB
V240_LCB
LongRisk_UCB
Bad_UCB
Null_UCB
Memory_UCB
Offdiag_UCB
LDO/LSO/LTO/LFO
path_type_precision_by_type
```

### B 线 weak pass

```text
B_weak_pass = 1
if:
  TopK87 precision >= 0.65
  V_LCB > 0
  LongRisk_UCB <= 0.10
  compute_ms_q90 <= 0.10
```

### B 线 strong pass

```text
B_strong_pass = 1
if:
  TopK87 precision >= 0.75
  V_LCB > 0
  RAUV_LCB > 0
  LongRisk_UCB <= 0.05
  Bad_UCB <= 0.05
  Null_UCB <= 0.15
  Memory_UCB <= 0.05
  Offdiag_UCB <= 0.05
  LDO/LSO/LTO/LFO <= 0.10
  compute_ms_q90 <= 0.05
```

### 如果不满足条件，Codex 先尝试

```text
如果 precision 低但 risk 低:
  - 该 sketch 只能作为 veto；
  - 不再调 value threshold。

如果 value 正但 risk 高:
  - 强化 hard veto；
  - 不允许 weighted sum 把 risk 抵消掉。

如果 cost q90 > 0.10 ms:
  - 缩小 microbatch；
  - 降低 virtual step count；
  - 改成 cached JVP/VJP；
  - 如果仍慢，标记为 diagnostic-only。

如果 LDO/LSO fail:
  - 做 group-blind score normalization；
  - 不能用 dataset-specific threshold。
```

### 可视化

```text
1. sketch score distribution by path type；
2. precision-coverage curve；
3. V_LCB vs coverage curve；
4. risk vs coverage curve；
5. compute cost histogram；
6. LDO/LSO drop waterfall；
7. value sketch vs veto scatter。
```

---

## P6：existing-action controller gate

### 打开条件

只允许以下两种情况打开 controller：

```text
Case C-pass:
  P3 density sufficient = 1

Case B-pass:
  B_strong_pass = 1
```

### Controller 形式

```text
Accept(a)=1
if:
  PathScore(a) >= threshold
  RiskVeto(a)=0
  MemoryVeto(a)=0
  OffdiagVeto(a)=0
  Cost(a)<=Cmax
```

不能使用：

```text
dataset_name
future outcome
old table
CoreLike label
PathGood label
AUV label
RAUV label
validation/test metric
```

### 必须记录

```text
controller_id
feature_names
thresholds
calibration split hash
heldout split hash
accepted_count
coverage
precision
V_LCB
RAUV_LCB
V240_LCB
LongRisk_UCB
Bad_UCB
Null_UCB
Memory_UCB
Offdiag_UCB
LDO/LSO/LTO/LFO
feature_cost_q90
certificate_cost_q90
```

### Pass

```text
controller_pass = 1
if:
  accepted_count >= 87
  precision >= 0.75
  V_LCB > 0
  LongRisk_UCB <= 0.05
  Bad_UCB <= 0.05
  Null_UCB <= 0.15
  Memory_UCB <= 0.05
  Offdiag_UCB <= 0.05
  LDO/LSO/LTO/LFO <= 0.10
```

### 如果不满足条件，Codex 先尝试

```text
如果 accepted_count < 87:
  - 不降质量硬凑；
  - 回 C 线扩大自然 panel；
  - 或回 B 线改 sketch。

如果 precision < 0.75:
  - 回 P5，不调 controller threshold。

如果 LDO fail:
  - 做 score shift autopsy；
  - 禁止 dataset-specific threshold。

如果 risk fail:
  - 把 risk/memory/offdiag 放硬 veto；
  - 不做 weighted compensation。
```

---

## P7：generated sandbox gate

### 打开条件

只允许以下情况打开 64-action sandbox：

```text
1. C 线 density insufficient 且 B/A 给出明确生成机制；
2. 或 B_strong_pass = 1，且 sketch 可以转成优化目标；
3. 或 A 线发现清晰 SlowBurnGood 机制，能定义生成目标。
```

否则：

```text
generated_sandbox_allowed = 0
```

### Sandbox 最小规模

```text
generated_action_count = 64
branch_horizon_rows = 64 * branch_count * horizon_count
horizons = 1,5,20,80,240
```

### Pass

```text
generated_weak_pass = 1
if:
  GradeAB/PathGood precision >= 0.25
  V_LCB > 0
  LongRisk_UCB <= 0.10
  new_positive_rate >= 0.10
  longrisk_created_rate <= 0.10
```

```text
generated_strong_pass = 1
if:
  precision >= 0.75
  V_LCB > 0
  LongRisk_UCB <= 0.05
  Bad/Null/Memory/Offdiag gates pass
```

### 如果不满足条件，Codex 先尝试

```text
如果 generated value negative:
  - stop generated family；
  - output damage decomposition；
  - 不继续加小变体。

如果 longrisk created high:
  - mark veto ineffective；
  - generated route stop。

如果 implementation fail:
  - only fix payload/apply/materializer；
  - do not interpret science result。
```

---

## P8：selected runtime and paired replay boundary

只有 P6 controller pass 或 P7 generated strong pass 后才打开。

### Runtime 必须记录

```text
feature_compute_ms_q90
sketch_compute_ms_q90
certificate_compute_ms_q90
payload_apply_ms_q90
controller_launches_per_active_step_q90
step_ratio_q90
memory_ratio
kernel_count
sync_count
empty_step_kernel_count
selected_action_count_per_step
```

### Runtime pass

```text
selected_runtime_pass = 1
if:
  step_ratio_q90 <= 1.50
  memory_ratio <= 1.05
  empty_step_kernel_count = 0
```

### Paired replay pass

```text
paired_replay_pass = 1
if:
  RealFunctional beats AdamWParallel
  RealFunctional beats bestLR
  RealFunctional beats NoOp
  RealFunctional beats Random
  shuffled payload fails
  LDO/LSO/LTO pass
```

---

## 8. v9.9.2 并行执行安排

### Batch 1：必须先跑，失败即停止大 panel

```text
P0 boundary reproduction
P1 distribution error localization
P2 generator repair candidates 1/16/256/1024 smoke
```

### Batch 2：只有至少一个 generator weak pass 后跑

```text
P3 1024 -> 5000 -> 10000 -> 20000 sequential density
P4 future path type on natural samples
```

### Batch 3：可并行但不能替代 C 线

```text
P5 FuturePathOperator sketch v4
```

### Batch 4：严格 gate

```text
P6 controller
P7 generated sandbox
P8 runtime / paired replay
```

---

## 9. v9.9.2 Route decision

```text
R0-BoundaryRegression
  if P0 fail

R1-DistributionMismatchUnresolved
  if no generator candidate passes P1 weak fidelity

R2-DistributionFidelityPassDensityPending
  if P1 pass but P3 full density not completed

R3-NaturalDensitySufficientControllerPending
  if P3 density sufficient and P6 not run

R4-NaturalDensityInsufficientGeneratorNeeded
  if P3 density insufficient with distribution pass

R5-FutureOperatorProxyPassControllerPending
  if B_strong_pass and P6 not run

R6-ControllerPassRuntimePending
  if P6 pass but P8 runtime not run

R7-GeneratedSandboxPassControllerPending
  if P7 generated strong pass but controller/runtime not run

R8-SystemLegalPassPairedReplayPending
  if runtime pass but paired replay not run

R9-FuturePathHypothesisWeakened
  if A line shows no valid path type in distribution-fidelity natural panel
```

---

## 10. v9.9.2 必须产出的 artifacts

```text
p0_v9910_boundary_reproduction.csv
p1_distribution_fidelity_axes.csv
p1_missing_major_groups.csv
p1_generator_cursor_trace.csv
p2_generator_candidate_matrix.csv
p2_generator_repair_failure_matrix.csv
p3_density_panel_1024.csv
p3_density_panel_5000.csv
p3_density_panel_10000.csv
p3_density_panel_20000.csv
p3_density_decision.json
p4_future_path_types.csv
p4_future_path_curves_by_group.csv
p5_future_operator_sketch_v4.csv
p6_existing_action_controller_gate.csv
p7_generated_sandbox_gate.csv
p8_selected_runtime_boundary.csv
p8_paired_replay_boundary.csv
contract_audit_v9920.csv
no_fake_audit_v9920.csv
failure_taxonomy_v9920.csv
route_decision_v9920.json
run_manifest_v9920.json
```

---

## 11. 最终判断

v9.9.2 的关键不是让系统马上成功，而是裁决三件事：

```text
1. natural generator 能不能修到分布保真；
2. 分布保真后自然 AP0 源是否有足够好动作密度；
3. 如果密度不足，是否存在可以转成 generated update 的 future-path 机制。
```

当前最准确的路线判断是：

$$
\boxed{
\text{先修分布，再裁密度，再谈 controller 或 generated route。}
}
$$

