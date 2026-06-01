# DG-KAN v9.9.4 Tail Fidelity / Natural Density / FuturePathOperator 四线并行完整实验计划

> 本文件基于 v9.9.3 `Natural Density / Tail Fidelity / FuturePathOperator` 的真实执行结果制定。v9.9.4 不继续小修 FOS、LC、AUV、CoreExpansion，也不在 tail fidelity 未通过时打开 density panel、controller、generated sandbox 或 paired replay。v9.9.4 的核心目标是先修好 natural AP0 extension generator 的分布，特别是好动作所在的稀有尾部，再裁决自然动作池密度，并在保真自然 panel 上继续验证 future path 与训练当下低成本机制。

---

## 0. v9.9.3 的一句话判断

v9.9.3 的 route 是：

```text
route = R9-MaterializerOrFidelityEngineeringBlocked
primary_blocker = major_and_tail_fidelity_failed_after_repair_attempt
secondary_blocker = density_panels_blocked
system_legal_controller_pass = 0
generated_route_status = stopped_fidelity_failed
```

这轮不是能力成功，也不是 controller 成功。它真正推进的是：在 v9.9.2 已经找到 `G5-hybrid-quota-random-generator` 并通过粗分布 gate 后，v9.9.3 增加了更严格的 tail fidelity gate，然后发现 G5 并没有保住好动作所在的稀有尾部分布。

v9.9.3 的关键数据是：

```text
selected_generator = G5-hybrid-quota-random-generator
P1_major_fidelity_pass = 0
P1_tail_fidelity_pass = 0
PSI_major_max = 0.016903746143043054
PSI_tail_max = 0.2983309215401979
JS_tail_max = 0.1431685310441598
missing_tail_group_count = 13
candidate_template_coverage = 1.0
Core77 / OldOnly / RiskClean coverage = 1.0 / 1.0 / 1.0
```

随后尝试 `G6-tail-aware-quota-generator`，但 G5/G6 的 tail pass 仍然都是 0，最终原因是：

```text
G6_tail_repair_did_not_pass_full_major_tail_gate
```

因为 P1 major+tail fidelity 没过，P2 sequential density panel 没有打开：

```text
largest_completed_panel_size = 0
CoreLike count / LCB / UCB = 0 / 0 / 0
PathGood count / LCB / UCB = 0 / 0 / 0
CoreLike + SlowBurn UCB = 0
density sufficient / insufficient / inconclusive = 0 / 0 / 1
```

所以这轮不能说自然动作密度低，也不能说 natural AP0 source 已经不够。只能说：

$$
\boxed{\text{当前自然扩流生成器的分布仍不可信，不能用它裁决好动作密度。}}
$$

---

## 1. 当前四线状态

### A 线：未来训练路径

A 线本轮没有实质推进，因为 P1 fidelity gate 失败后，P4 future-path revalidation 被 gate-block。不能因此说 future path 方向失败。

前几轮已经看到的现象仍然成立：Core77、OldOnly、Core+RiskClean 这些动作不是当前一步 loss descent 最强，而是能在 h20/h80/h240 后表现出更好的训练路径。v9.9.3 只是说明：如果没有一个保真的 natural panel，就不能把这种现象推广到自然扩流动作池。

A 线下一步要做的是：在 tail-fidelity 通过的自然 panel 上重新验证以下路径类型：

```text
FastGood：h1/h5 就好，h20/h80/h240 继续好；
SlowBurnGood：h1/h5 不明显甚至略差，但 h20/h80/h240 变好；
RiskyHighAUV：AUV 很高，但 longrisk / memory / offdiag 高；
SafeLowValue：risk 很低，但 value 不够；
BadPath：value 低且 risk 高。
```

判断好路径时不能只看 AUV 或 RAUV。必须同时看：

```text
V1 / V5 / V20 / V80 / V240；
AUV / RAUV；
longrisk；
bad/null；
memory/offdiag；
hard-tail；
leaveout stability；
runtime cost。
```

### B 线：训练当下低成本 FuturePathOperator

B 线本轮也没有推进。P5 FPO v5 weak/strong 都是 0，best 和 precision 都是 `None`，因为上游 P1/P2 没过。

这不是 B 线失败的新证据，而是正确 gate。前几轮已经证明 cheap proxy 和 FO proxy 不够：它们要么便宜但选错动作，要么太贵且 value 仍为负。v9.9.4 不能继续调 LC8/FOS4 阈值，而要重写 FuturePathOperator sketch。

B 线现在的问题不是：

```text
怎么把某个 cheap response feature 阈值调好？
```

而是：

```text
训练当下有没有一种低成本算子，可以预测 action 对未来训练路径的影响？
```

这个算子不应该预测当前函数变化大不大，而应该预测：

$$
\Delta \mathcal{P}_{t:t+h}(a)
$$

也就是 action $a$ 对后续训练路径 $t:t+h$ 的影响。

### C 线：自然 AP0 动作扩流

C 线是 v9.9.3 的主线，也是当前最高优先级。

v9.9.0 把 natural AP0 extension materializer 落地；v9.9.1 发现原 generator 分布严重失真；v9.9.2 找到 G5，粗分布看似通过；v9.9.3 增加 tail fidelity 后发现 G5/G6 都不能同时满足 major 和 tail fidelity。

这说明：

$$
\boxed{\text{粗分布保真不够，好动作在尾部；自然扩流必须先保住尾部分布。}}
$$

如果 C 线不解决，后面所有 density、coverage、controller、generated route 都没有可靠依据。

### D 线：generated update

D 线继续停止是正确的。

原因很简单：

```text
B 线没有合法低成本机制；
C 线没有保真自然密度结论；
A 线仍是事后 future-path diagnostic；
```

在这种情况下重开 APG/APGU/APGX 只会重复过去的问题：payload 合法，branch-horizon 能跑，但 outcome value-negative / high-longrisk。

D 线只能在满足以下任一条件后重开 64-action sandbox：

```text
1. C 线证明自然 AP0 好动作密度不足，需要新动作源；
2. B 线找到可用 FuturePathOperator，可作为生成目标；
3. A 线找到明确 future-path 机制，可转成 generated objective。
```

---

## 2. v9.9.3 的核心 insight

### 2.1 现在不是“没有好动作”，而是“采样不到正确尾部”

v9.8.x 到 v9.9.0 已经表明 Core77、OldOnly、Core+RiskClean 这类动作有强 future path 信号。v9.9.2 的 G5 在 major axes 上看起来还不错，但 v9.9.3 的 tail fidelity 暴露出：如果不保住稀有尾部，density panel 的低密度结果就会误导我们。

用概率语言说：

$$
p_{natural}(a)
$$

是我们想估计的自然 AP0 action 分布，而当前 generator 生成的是：

$$
\hat p_G(a).
$$

v9.9.2 主要检查的是粗粒度 marginal：

$$
\hat p_G(z_{major}) \approx p_{natural}(z_{major}).
$$

v9.9.3 说明这不够。好动作位于稀有区域 $z_{tail}$，我们还需要：

$$
\hat p_G(z_{tail}) \approx p_{natural}(z_{tail}).
$$

否则：

$$
\hat P_G(GoodAction)
$$

不能代表：

$$
P_{natural}(GoodAction).
$$

### 2.2 G5 不是完全没用，但它不是 density 裁决工具

G5 的 major PSI 很低，说明它能覆盖粗分布；candidate_template coverage 和 Core77 / OldOnly / RiskClean coverage 都是 1.0，说明它不是完全漏掉所有关键来源。

但 tail PSI `0.2983`、JS `0.1432`、missing tail group `13` 说明它对关键尾部比例和细分组支持不可信。它可以作为 repair starting point，但不能作为 density estimator。

### 2.3 G6 tail-aware repair 的失败说明“按 precursor 分组轮转”仍不够

G6 不使用 outcome label，只按 canonical AP0 precursor 分组轮转。这是正确方向，因为不能用未来 outcome 直接控制生成。但它仍没通过 full major+tail gate，说明问题可能不是简单的 precursor quota，而是：

```text
1. tail group key 与 generator precursor key 不一致；
2. tail group 是多轴组合，单轴 quota 无法恢复；
3. cursor / quota sampler 仍有 collapse；
4. action recipe coverage 不全；
5. payload/action collision 或 filters 导致尾部被删；
6. major fidelity 与 tail fidelity 的采样权重冲突。
```

### 2.4 v9.9.3 是进展，但不是能力进展

它的价值是防止错误结论：

```text
错误结论 A：G5 major pass，所以可以跑 full density。
错误结论 B：1024 pilot 低密度，所以自然 AP0 source 不够。
错误结论 C：density 不够，所以重开 generated route。
```

v9.9.3 阻止了这些错误结论。这是科学进展，但不会让系统能力立刻上涨。

---

## 3. 当前真正卡在哪里

### 3.1 卡在自然生成器的 tail fidelity

这是第一 blocker。只要 tail fidelity 不过，不能打开 5000 / 10000 / 20000 density panel。

### 3.2 卡在 density 裁决

因为 P2 largest completed panel = 0，我们现在仍然不知道：

```text
自然 AP0 action stream 里的 CoreLike / PathGood / SlowBurnGood 密度是否足够；
Core77 只有 77 个到底是样本量问题，还是自然源真实密度低；
existing-action harvesting 是否还有战略价值。
```

### 3.3 卡在训练当下机制

即便自然密度够，controller 也需要合法、低成本、跨分布稳定的识别规则。B 线目前仍没有。

### 3.4 卡在 generated route 的目标

如果自然密度不足，需要新动作生成。但生成器不能再是手工 primitive 变体。它必须以 future path 机制或 FuturePathOperator 为目标。现在这个目标还没出现。

---

## 4. v9.9.4 总体目标

v9.9.4 的总体目标是：

$$
\boxed{\text{修复自然 AP0 extension generator 的 major+tail fidelity，并在保真 panel 上裁决好动作密度。}}
$$

同时继续维护四线：

```text
A 线：future path 类型与机制分解；
B 线：低成本 FuturePathOperator sketch；
C 线：major+tail fidelity 和 density panel；
D 线：严格 gate 的 generated sandbox。
```

但优先级是：

```text
C 线最高优先级；
A 线低成本继续；
B 线重写但不大规模扫；
D 线只在 A/B/C 给出证据后打开。
```

---

## 5. v9.9.4 实验计划

## P0. 复现 v9.9.3 边界

### 目标

确认本轮没有回退，并复现：

```text
route = R9-MaterializerOrFidelityEngineeringBlocked
G5 major/tail pass = 0 / 0
G6 tail repair pass = 0
P2 density not_run
controller/generated/runtime not_run
no fake/proxy/cpu = 1
```

### 要记录的指标

```text
P0_boundary_pass
selected_generator_id
G5 major PSI / JS / max-share / entropy
G5 tail PSI / JS / missing-tail / tail coverage
G6 major PSI / JS / max-share / entropy
G6 tail PSI / JS / missing-tail / tail coverage
P2 panel status
fake/proxy/cpu audit
```

### 判断标准

如果 P0 不能复现，停止本轮，Codex 先检查：

```text
1. v9.9.3 artifact 路径是否正确；
2. route_decision_v9930.json 是否读取错误；
3. tail group reference artifact 是否丢失；
4. G5/G6 generator id 是否和 runner 内部 id 不一致；
5. no-fake audit 是否误扫了旧 artifact。
```

---

## P1. Tail group reference audit

### 假设

v9.9.3 的 tail fidelity 失败可能有两种原因：

```text
H1a: generator 真的没有保住 tail groups；
H1b: tail group reference / key construction 本身有口径错误。
```

### 实验设计

把 tail group 拆成两类：

```text
1. diagnostic outcome tail：Core77 / OldOnly / RiskClean / SlowBurnGood 等事后好动作类型。
2. commit-time precursor tail：不使用未来 outcome 的 action family、step bucket、template、payload bucket、memory/offdiag precursor、candidate lineage 等。
```

注意：diagnostic outcome tail 只能用于 fidelity audit，不能用于 generator selector 或 controller。generator 修复只能使用 commit-time precursor tail 或 canonical precursor coverage。

### 要记录的指标

```text
tail_group_count_total
tail_group_count_diagnostic_outcome
tail_group_count_commit_time_precursor
tail_group_key_missing_rate
tail_group_singleton_count
tail_group_min_support
tail_group_max_support
tail_group_entropy
tail_group_overlap_matrix
Core77_tail_group_coverage
OldOnly_tail_group_coverage
RiskClean_tail_group_coverage
SlowBurn_tail_group_coverage
reference_tail_group_has_future_label_flag
```

### 判断标准

通过条件：

```text
1. tail group reference 中未来 outcome group 与 commit-time precursor group 明确分离；
2. generator 不使用 outcome tail 作为采样规则；
3. tail group key missing rate = 0；
4. singleton / ultra-rare groups 被标记为 diagnostic-only 或 merged-tail；
5. 每个用于 generator quota 的 group 都有明确 canonical support。
```

如果不通过，Codex 先尝试：

```text
1. 修复 tail key construction；
2. 对 support < 3 的 group 做 merge，不允许单点 group 直接作为 quota；
3. 把 outcome-derived tail groups 从 generator config 中移除；
4. 重建 precursor-tail reference；
5. 加入 artifact：tail_group_lineage_debug.csv。
```

---

## P2. Generator repair matrix v2

### 假设

G5/G6 失败不是因为 natural extension 不可能，而是因为 sampler 没有同时满足 major fidelity 和 tail fidelity。

### 候选 generator

本轮不只修一个 G6，而是并行实现并比较：

```text
G5: 保留 v9.9.2 hybrid-quota-random baseline；
G6: 保留 v9.9.3 tail-aware quota baseline；
G7: multi-axis precursor quota generator；
G8: tail-reservoir capped generator；
G9: two-stage major-then-tail fill generator；
G10: alias-table proportional sampler with tail minimum support；
G11: stratified mixture generator，major axes 与 precursor-tail axes 分别控制；
G12: diagnostic-only upper-bound sampler，只用于确认 tail groups 是否可 materialize，不能 official density。
```

G12 可以使用更强的 tail balancing，但必须标记 `official_density_eligible = 0`，只用于确认 materializer 能不能生产缺失 tail groups。

### 要记录的指标

每个 generator 在 1024 pilot 上记录：

```text
action_count
payload_collision_count
action_collision_count
branch_horizon_rows_expected / actual
major_PSI_max
major_JS_max
major_max_group_share
major_entropy_ratio_min
tail_PSI_max
tail_JS_max
missing_tail_group_count
tail_group_coverage_min
tail_group_coverage_mean
Core77_tail_coverage
OldOnly_tail_coverage
RiskClean_tail_coverage
SlowBurn_tail_coverage
rows_per_sec
peak_gpu_mb
cpu_offload
fake/proxy
```

### 判断标准

official density generator 必须满足：

$$
PSI_{major} \le 0.05
$$

$$
JS_{major} \le 0.08
$$

$$
MaxShare_{major} \le 0.35
$$

$$
EntropyRatio_{major} \ge 0.90
$$

$$
PSI_{tail} \le 0.10
$$

$$
JS_{tail} \le 0.08
$$

$$
MissingTailGroup = 0
$$

$$
TailCoverage_{min} \ge 0.80
$$

并且：

```text
payload_collision_count = 0
action_collision_count = 0
branch_horizon_completion = 1
fake/proxy/cpu = 0
```

如果没有 candidate 通过，Codex 先尝试：

```text
1. 对 missing tail groups 输出 top-20 missing group keys 和 canonical support；
2. 检查 cursor 是否只在前几个 group 上循环；
3. 检查 quota normalize 是否把小尾部 group rounding 到 0；
4. 改成 stochastic rounding / residual quota carry；
5. 对 missing groups 做 forced-first-pass coverage；
6. 检查 filters 是否在 payload generation 后删除 tail groups；
7. 检查 tail group key 是否和 generated action metadata 对不上；
8. 如果 major pass 被 tail repair 破坏，调 mixture weight，但必须用 heldout generator validation，不按 outcome 调。
```

---

## P3. Sequential natural density panel

### 前置条件

只有 P2 至少一个 official density generator 通过 major+tail gate，才允许打开 P3。

### 假设

在保真 natural AP0 extension 分布下，好动作密度可能满足或不满足 official coverage gate。

### Panel 设计

按序运行：

```text
1024 -> 5000 -> 10000 -> 20000
```

每个 panel 必须真实 materialize branch-horizon rows，不允许 fake / proxy。

### 标签定义

P3 不只看 CoreLike，还要同时记录：

```text
CoreLike：类似 Core77 的高质量严格动作；
PathGood：future path 全路径好；
SlowBurnGood：h1/h5 不强但 h20/h80/h240 好；
RiskClean：longrisk/bad/null/memory/offdiag 干净；
CorePlusSlow：CoreLike 或 SlowBurnGood；
RiskyHighAUV：AUV 高但 longrisk/memory/offdiag 高；
SafeLowValue：risk 干净但 value 不够。
```

### 要记录的指标

```text
panel_size
branch_horizon_rows_expected / actual
CoreLike_count / rate / Wilson_LCB / Wilson_UCB
PathGood_count / rate / Wilson_LCB / Wilson_UCB
SlowBurnGood_count / rate / Wilson_LCB / Wilson_UCB
CorePlusSlow_count / rate / Wilson_LCB / Wilson_UCB
RiskyHighAUV_rate
SafeLowValue_rate
longrisk_rate
bad_rate
null_rate
memory_fail_rate
offdiag_fail_rate
per_dataset_rate
per_family_rate
per_template_rate
per_tail_group_rate
effective_sample_size
throughput_rows_per_sec
peak_gpu_mb
```

### 判断标准

自然源充足：

$$
LCB(CoreLike \lor SlowBurnGood) \ge 0.03
$$

并且：

$$
UCB(LongRisk) \le 0.05
$$

或在 accepted candidate subset 上满足 controller gate。

自然源不足：

$$
UCB(CoreLike \lor SlowBurnGood) < 0.03
$$

在 10000 或 20000 panel 上成立，并且 generator fidelity 仍通过。

仍不确定：

```text
CI 跨过 0.03；
或 per-group support 不均导致有效样本量不足；
或 panel throughput / memory 不稳定。
```

如果 P3 因 throughput / memory 失败，Codex 先尝试：

```text
1. chunk-actions 从 512 降到 256 / 128；
2. 增加 resume manifest，不重跑已完成 action；
3. 每 action 清理 CUDA cache；
4. 分 dataset / family shard 执行；
5. 保证 row_id/action_id deterministic；
6. 检查 label exclusivity / duplicate row；
7. 只在 throughput 修复后继续 density panel。
```

---

## P4. Future path type decomposition on fidelity-pass natural panel

### 前置条件

P3 至少完成 1024 或 5000 fidelity-pass panel。

### 假设

自然 panel 中的好动作不是 immediate loss descent，而是 future path improvement。

### 要记录的指标

对 CoreLike、SlowBurnGood、RiskyHighAUV、SafeLowValue、RandomMatched、CanonicalCore77 分组记录：

```text
V1_LCB
V5_LCB
V20_LCB
V80_LCB
V240_LCB
AUV_LCB
RAUV_LCB
DelayedGain = V240 - V1
RiskPath_UCB
LongRisk_UCB
Bad_UCB
Null_UCB
MemoryFail_UCB
OffdiagFail_UCB
HardTailDelta
LDO / LSO / LTO / LFO
```

### 判断标准

future path mechanism weak pass：

```text
存在至少一个 path type，count >= 87 或 density LCB >= 0.03；
RAUV_LCB > 0；
V240_LCB > 0；
LongRisk_UCB <= 0.05；
Memory/Offdiag_UCB <= 0.05。
```

strong pass：

```text
上述条件同时在 LDO / LSO / LTO / LFO 中稳定；
且 RandomMatched / RiskyHighAUV 不误通过。
```

如果 P4 不通过，Codex 先尝试：

```text
1. 分离 immediate-good 与 slow-burn-good；
2. 检查 AUV 是否被 risky high-response 动作污染；
3. 使用 RAUV 而不是 AUV；
4. 增加 delayed-gain 指标；
5. 如果 SafeLowValue 很多，检查旧 value label 是否过严；
6. 不允许直接调标签到过线，必须写 label change audit。
```

---

## P5. FuturePathOperator sketch v6

### 前置条件

P3 或 P4 至少给出一种可分析的 path type。

### 假设

训练当下可以用低成本算子近似 future path，不必直接 replay 未来路径。

### 候选 sketch

```text
FPO6A: tiny virtual AdamW 1-step sketch；
FPO6B: two-step low-rank JVP/VJP sketch；
FPO6C: memory-buffer delayed-gain sketch；
FPO6D: hard-tail response + memory veto sketch；
FPO6E: risk-adjusted path response sketch；
FPO6F: CoreLike/SlowBurn contrast-trained legal ranker；
FPO6G: no-score hard-gate baseline；
FPO6H: negative-control shuffled action sketch。
```

所有 FPO sketch 只能使用 commit-time fields。不能使用 future outcome、GradeAB、AUV、RAUV、CoreLike label 作为 input。它们只能用于训练后的 heldout evaluation 或 calibration split label。

### 要记录的指标

```text
FPO_id
input_field_green/yellow/red count
cost_q50/q90/q99_ms
TopK87 precision
V_LCB
RAUV_LCB
LongRisk_UCB
Bad_UCB
Null_UCB
MemoryFail_UCB
OffdiagFail_UCB
LDO/LSO/LTO/LFO
calibration_split_id
heldout_split_id
negative_control_precision
```

### 判断标准

weak pass：

$$
Precision_{TopK87} \ge 0.75
$$

$$
V_{LCB} > 0
$$

$$
LongRisk_{UCB} \le 0.05
$$

$$
Cost_{q90} \le 2.0\text{ ms}
$$

strong pass：

```text
weak pass + LDO/LSO/LTO/LFO <= 0.10 + negative control fail。
```

如果 P5 不通过，Codex 先尝试：

```text
1. 输出 false positive / false negative contrast table；
2. 检查 FPO 是否只在清 risk 而没有 value signal；
3. 检查 cost 是否来自 per-sample loop，可改为 batched JVP；
4. 检查是否使用了 accidental yellow/red field；
5. 分开 value operator 和 risk veto，不要加权成一个复杂 score；
6. 若所有 low-cost sketch precision < 0.5，停止 B 线阈值微调，等待 A/C 新机制。
```

---

## P6. Existing-action controller gate

### 前置条件

满足以下任一条件：

```text
1. P3 density sufficient；
2. P5 FPO strong pass；
3. P4 future path strong pass 且有 commit-time safe rule。
```

### 目标

构造 system-legal accepted region，不按 dataset 调参，不使用 future outcome at commit。

### 记录指标

```text
accepted_count
coverage
precision
V_LCB
RAUV_LCB
LongRisk_UCB
Bad_UCB
Null_UCB
MemoryFail_UCB
OffdiagFail_UCB
LDO/LSO/LTO/LFO
per_dataset accepted count / precision / V
per_family accepted count / precision / V
feature cost q90
payload apply cost q90
step ratio q90 estimate
```

### 判断标准

$$
accepted\_count \ge 87
$$

$$
precision \ge 0.75
$$

$$
V_{LCB}>0
$$

$$
LongRisk_{UCB}\le0.05
$$

$$
Bad_{UCB}\le0.05
$$

$$
Null_{UCB}\le0.15
$$

$$
LDO,LSO,LTO,LFO\le0.10
$$

如果不通过，Codex 先尝试：

```text
1. 区分 quality fail、support fail、backfill fail；
2. 检查是否某个 split support 过低；
3. 不允许 dataset-specific threshold；
4. 如果只差 count，回到 C 线扩大自然 panel；
5. 如果 risk 高，回到 B 线 risk veto；
6. 如果 value 低，回到 A 线 path type 定义；
7. 如果 leaveout 高，输出 group-specific failure table。
```

---

## P7. Generated sandbox reopen gate

### 前置条件

只有以下情况允许打开 64-action generated sandbox：

```text
Case 1: P3 fidelity-pass density panel 证明 natural source insufficient；
Case 2: P5 FPO strong pass，可以作为 generated objective；
Case 3: P4 future path mechanism strong pass，并且能写成 commit-time objective。
```

### 禁止条件

```text
1. P1/P2 fidelity 不过；
2. density 仍 inconclusive；
3. B 线没有机制；
4. 只是为了“试试看”重开 APG/APGU/APGX。
```

### 记录指标

```text
generated_action_count
payload/certificate hash missing
action apply linf
branch_horizon_rows expected/actual
future path metrics
CoreLike / PathGood / SlowBurnGood precision
longrisk/bad/null/memory/offdiag
source-to-generated damage
runtime cost
negative control
```

### 判断标准

64-action sandbox weak pass：

```text
PathGood precision >= 0.25；
V_LCB > 0；
LongRisk_UCB <= 0.10；
new-positive-created-rate > 0.10；
negative control fail。
```

如果不通过，Codex 先尝试：

```text
1. 不新增 blind primitive；
2. 做 source-to-generated damage decomposition；
3. 查是 value direction lost、risk veto ineffective、memory/offdiag fail，还是 payload OOD；
4. 只有明确 failure mode 后才能设计下一类 primitive。
```

---

## P8. Runtime / paired replay boundary

### 前置条件

P6 controller pass 或 P7 generated sandbox strong pass。

### 记录指标

```text
feature_compute_ms_q90
FPO_compute_ms_q90
payload_apply_ms_q90
controller_kernel_count
sync_count
step_ratio_q50/q90/q99
peak_memory_ratio
paired replay RealFunctional vs AdamWParallel / bestLR / NoOp / Random
shuffled payload negative control
```

### 判断标准

$$
step\_ratio_{q90}\le1.50
$$

$$
memory\_ratio\le1.05
$$

paired replay 必须满足：

```text
RealFunctional beats AdamWParallel；
RealFunctional beats bestLR；
RealFunctional beats NoOp；
RealFunctional beats Random；
shuffled payload fail。
```

---

## 6. 可视化要求

必须输出：

```text
1. Major fidelity heatmap；
2. Tail fidelity heatmap；
3. Missing tail group bar chart；
4. Generator repair matrix table；
5. Density curve: panel size vs Wilson LCB/UCB；
6. CoreLike / PathGood / SlowBurnGood per-dataset density；
7. Future path curves V1/V5/V20/V80/V240；
8. Risk-adjusted future path scatter；
9. FPO false positive / false negative contrast plot；
10. Controller leaveout waterfall；
11. Runtime stacked cost chart。
```

---

## 7. Stop / pivot 规则

### 如果 P1/P2 仍找不到 major+tail fidelity generator

停止 density panel。Codex 优先做 engineering repair，不跑 full science runner。

### 如果 fidelity pass 但 5000/10000/20000 density UCB 低于 0.03

判定自然 AP0 source 不足，existing-action harvesting 不再作为主线。转向 generated route，但必须基于 A/B 机制。

### 如果 density LCB 高于 0.03，但 B 线仍失败

说明好动作足够多但训练当下不可识别。继续做 FuturePathOperator，不开 generated route。

### 如果 density LCB 高于 0.03 且 B 线过

打开 existing-action controller / runtime / paired replay。

### 如果 density 不足但 A/B 找到机制

打开 64-action generated sandbox。

### 如果 A/B/C 都失败

停止在当前 AP0 action family 上推进，回到更高层 geometry-adaptive update formulation。

---

## 8. 最终交付物

v9.9.4 必须交付：

```text
1. route_decision_v9940.json
2. p1_tail_group_reference_audit.csv
3. p2_generator_repair_matrix.csv
4. p2_generator_major_tail_fidelity.csv
5. p3_sequential_density_panel.csv
6. p4_future_path_type_decomposition.csv
7. p5_future_path_operator_sketch.csv
8. p6_controller_gate.csv
9. p7_generated_sandbox_gate.csv
10. p8_runtime_paired_boundary.csv
11. no_fake_audit.csv
12. failure_taxonomy.csv
13. codex_retry_suggestions.csv
14. figures/
```

---

## 9. v9.9.4 最终目标

v9.9.4 不要求直接 functional success。它要求裁决以下问题：

```text
1. natural generator 能否 major + tail fidelity pass？
2. 如果能，自然动作好动作密度到底够不够？
3. 好动作密度够时，是否有低成本合法 FuturePathOperator 能识别？
4. 好动作密度不够时，是否有足够机制打开 generated sandbox？
```

最终判断必须落到以下 case 之一：

```text
Case A: natural density sufficient，existing-action route 继续；
Case B: natural density insufficient，转向 generated route；
Case C: generator fidelity 仍失败，继续 engineering repair；
Case D: density sufficient but no legal mechanism，继续 FuturePathOperator；
Case E: density insufficient and no mechanism，回到高层 geometry-adaptive update formulation。
```
