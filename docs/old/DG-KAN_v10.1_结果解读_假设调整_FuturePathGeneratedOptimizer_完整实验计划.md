# DG-KAN v10.1 结果解读、假设调整与 FuturePath Generated Optimizer 完整实验计划

> 本文件基于 v10.0 `FuturePath Generated Optimizer Reset` 的真实复盘结果制定。公式使用 Typora 友好的 `$...$` 与 `$$...$$`。本计划不把任何 diagnostic 写成 official pass，不按 dataset 调参，不使用 outcome-derived commit feature，不恢复盲目 APG 变体。

---

## 0. 当前一句话判断

v10.0 的核心结果是：

$$
\boxed{
\text{自然 AP0 harvesting 已被基本削弱；训练当下 FPO 识别失败；generated discovery 初版失败。}
}
$$

这不是能力成功，但它是战略分叉点。下一步不应该继续在自然 AP0 动作池里挖 10 个补齐动作，也不应该继续调 `FPO11C` 或 AUV/RAUV 阈值。下一步的核心应该是：

$$
\boxed{
\text{直接学习或生成能改善未来训练路径的 functional update。}
}
$$

---

## 1. v10.0 结果复盘

v10.0 route：

```text
route = CaseD-NaturalInsufficientBDDiscoveryFail
primary_blocker = natural_density_insufficient
secondary_blocker = FPO11_and_generated_discovery_failed
system_legal_controller_pass = 0
generated_route_status = stopped_future_path_generated_optimizer_reset
```

四线核心结果：

```text
A 线 Fast / Slow / Risky / SafeLow / Bad = 3 / 11 / 121 / 9 / 4920
A discovery target pass = 0

B 线 FPO11 weak / discovery = 0 / 0
best = FPO11C-signal-reservoir-path-type-snr
precision = 0.011494252873563218
V240 LCB = -0.6468261963476977

C 线 completed panels = 3
largest panel = 10000
natural density sufficient / insufficient / inconclusive = 0 / 1 / 0

D 线 D0 / D1 / D2 = 1 / 1 / 1
D2 weak = 0
FastSlow precision = 0.0

P7 controller = not_run
P8 runtime = not_run
P9 paired replay = not_run
No-fake rows checked = 504261
fake / proxy / cpu = 0 / 0 / 0
```

### 1.1 A 线的解释

A 线不是失败得没有意义。它告诉我们：在当前自然 panel / discovery corpus 里，真正符合 FastGood 或 SlowBurnGood 的动作极少，分别只有 `3` 和 `11`。同时 Risky 和 Bad 数量很大。这说明未来路径标签是有区分力的，但当前动作来源没有足够多好路径动作。

最重要的是 SlowBurnGood 仍然应该作为核心目标。它对应的不是当前一步 loss descent，而是：

$$
V_1(a) \text{ 不强或为负，但 } V_{20}(a), V_{80}(a), V_{240}(a) \text{ 逐步转好。}
$$

这正是 functional update 应该追求的“未来轨迹修正”。

### 1.2 B 线的解释

FPO11C 的 precision 只有 `0.01149`，V240 LCB 为负。这不是阈值没调好，而是当前 FPO11 的目标与真正好路径错配。它仍然更像是在测当前响应、SNR 或局部风险，而不是在预测未来路径类型。

因此 B 线不能继续做单一分数：

$$
Score(a) \rightarrow Good(a)
$$

而要改成路径类型预测：

$$
FPO(a) \rightarrow \{FastGood, SlowBurnGood, RiskyHighAUV, SafeLowValue, BadPath\}.
$$

### 1.3 C 线的解释

C 线已经完成关键战略判断：自然 AP0 source 在保真 sampler 下给出 natural density insufficient。largest panel 达到 `10000`，并且 density sufficient / insufficient / inconclusive 为 `0 / 1 / 0`。

这意味着自然 harvesting 不能再作为主线。下一步最多做有限 confirmation，不应该再无限扩大 20000 / 50000 来拖慢路线。

### 1.4 D 线的解释

D 线第一次按 gate 做 generated discovery staircase，但 D2 weak 仍为 0，FastSlow precision 为 0。这说明当前 generated objective 没有对准未来好路径。它不是多跑就能变好。

D 线的问题不是 materializer，也不是 action apply，而是生成目标。

---

## 2. 假设需要如何调整

### 2.1 降级假设 H_old：自然 AP0 动作池足够

原假设：

$$
P_{natural}(FastGood \cup SlowBurnGood) \ge 0.03.
$$

v10.0 后，这个假设应降级。C 线已经给出 natural density insufficient。下一步只做 confirmation，不再把自然 harvesting 作为主路线。

### 2.2 降级假设 H_fpo_scalar：单一 FPO 分数能识别好动作

v10.0 的 FPO11C 明确失败，说明：

$$
FPO_{scalar}(a) \not\Rightarrow FuturePathGood(a).
$$

B 线必须改为 path-type prediction 和 false-positive/false-negative 对照，而不是继续调 scalar score。

### 2.3 新主假设 H_path_generator

新的主假设是：

$$
\boxed{
\text{好 functional update 需要直接以 future path type 为生成目标。}
}
$$

更具体：

$$
Generate(\Delta)
\quad \text{such that} \quad
\Delta \in FastGood \cup SlowBurnGood,
$$

同时满足：

$$
LongRisk_{240}(\Delta) \le \tau_L,
$$

$$
Bad(\Delta) \le \tau_B,
$$

$$
Null(\Delta) \le \tau_N,
$$

$$
MemoryFail(\Delta) \le \tau_M,
$$

$$
OffdiagFail(\Delta) \le \tau_O,
$$

$$
Cost(\Delta) \le C_{max}.
$$

### 2.4 新组织假设 H_runner_split

实验系统必须拆成三类 runner：

```text
engineering gate runner：只验证入口、materializer、payload、runtime。
science discovery runner：允许非 official 小规模探索机制，但不能写成 pass。
official gate runner：只有 discovery 达到硬条件后才运行。
```

否则每轮都会被上游 official gate 串行拖慢。

---

## 3. v10.1 总体目标

v10.1 的总体目标不是立即 system success，而是裁决三个问题：

```text
Q1. C 线 natural density insufficient 是否稳定？
Q2. B 线能否从 scalar FPO 改成 path-type predictor，并至少产生 discovery signal？
Q3. D 线能否生成哪怕少量 FastGood / SlowBurnGood-like action？
```

如果 Q2/Q3 都失败，就应停止当前 generated objective，回到更高层的 update-rule theory，而不是继续造 APG 名字。

---

## 4. 实验总览

v10.1 分为 9 个阶段：

```text
P0: 复现 v10.0 boundary；
P1: A 线 path type label 固化与 false-positive/false-negative 对照；
P2: C 线 natural density confirmation；
P3: B 线 path-type FPO v12；
P4: D 线 generated discovery staircase v2；
P5: generated failure autopsy；
P6: 条件性 existing/generated controller boundary；
P7: 条件性 selected runtime；
P8: 条件性 paired replay / short-full boundary；
P9: route decision 与 stop/pivot。 
```

---

## 5. P0：复现 v10.0 boundary

### 目标

确认 v10.0 的关键事实没有漂移：natural density insufficient、FPO11 failed、generated discovery failed。

### 必须记录

```text
source_route
P0_boundary_pass
A_fast_count
A_slow_count
A_risky_count
A_safelow_count
A_bad_count
B_best_fpo
B_best_precision
B_best_V240_LCB
C_largest_panel
C_density_status
D_generated_action_count
D_fastslow_precision
D_V240_LCB
D_longrisk_UCB
controller_status
runtime_status
paired_replay_status
fake/proxy/cpu
```

### 判断标准

P0 pass：

```text
source_route = CaseD-NaturalInsufficientBDDiscoveryFail
C_natural_density_insufficient = 1
B_FPO11_weak_pass = 0
D_generated_discovery_pass = 0
fake/proxy/cpu = 0/0/0
```

### 不满足时 Codex 先尝试

```text
如果 route 不一致：检查读取的 artifact 是否是 v10.0 final out-dir。
如果 C density 状态漂移：检查 panel seed / repeat seed / confirm seed 是否一致。
如果 D metrics 漂移：检查 generated supplement 是否重复写入或漏写。
如果 fake/proxy 非 0：停止全部 downstream，先修 audit。
```

---

## 6. P1：A 线 path type 固化

### 目标

把好路径定义从单一 AUV / RAUV 固化为路径类型。路径类型是 v10.1 的监督目标和生成目标。

### 路径类型定义

一个动作 $a$ 的路径向量：

$$
T(a)=(V_1,V_5,V_{20},V_{80},V_{240},RAUV,LongRisk,Bad,Null,MemoryFail,OffdiagFail).
$$

定义：

```text
FastGood:
  V1_LCB > 0
  V20_LCB > 0
  V80_LCB > 0
  V240_LCB > 0
  longrisk_UCB <= 0.05
  memory/offdiag_UCB <= 0.05

SlowBurnGood:
  V1_LCB <= 0.05
  V20_LCB > 0
  V80_LCB > 0
  V240_LCB > 0
  RAUV_LCB > 0
  longrisk_UCB <= 0.05
  memory/offdiag_UCB <= 0.05

RiskyHighAUV:
  RAUV_LCB > 0
  longrisk_UCB > 0.25 or memory/offdiag_UCB > 0.25

SafeLowValue:
  longrisk_UCB <= 0.05
  memory/offdiag_UCB <= 0.05
  V240_LCB <= 0

BadPath:
  V240_LCB < 0 or longrisk_UCB > 0.50 or Bad_UCB > 0.05
```

### 必须记录

```text
action_id
source_type = canonical/natural/generated
path_type
V1/V5/V20/V80/V240_LCB
RAUV_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
dataset_id_for_diagnostic_only
family_id
template_id
tail_group_id
payload_norm_bucket
action_norm_bucket
```

### 判断标准

A 线 discovery pass：

```text
FastGood_count + SlowBurnGood_count >= 30
SlowBurnGood_count >= 10
RiskyHighAUV_count >= 30  # 用作反例
BadPath_count >= 100      # 用作反例
path_type labels no NaN/Inf/missing
```

A 线 strong pass：

```text
FastGood + SlowBurnGood accepted_count >= 87
V240_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory/offdiag_UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
```

### 可视化

```text
p1_path_type_counts.svg
p1_v1_to_v240_trajectory_by_type.svg
p1_rauv_vs_longrisk_by_type.svg
p1_slowburn_good_examples.svg
p1_tail_group_path_type_heatmap.svg
p1_path_type_dataset_family_template_balance.svg
```

### 不满足时 Codex 先尝试

```text
如果 Fast/Slow 太少：
  不放宽 longrisk / memory / offdiag；只检查 V1 门槛是否过严。
  将 SlowBurnGood 分为 strict / relaxed 两版，但 official 仍用 strict。

如果 RiskyHighAUV 太多：
  保留为负样本，不调 AUV；强化 risk-path contrast。

如果 SafeLowValue 多：
  检查是否是 delayed-gain，需要 h480 diagnostic；但 h480 不得直接 official。

如果 path label 缺字段：
  先修 future path row schema，不跑 B/D。
```

---

## 7. P2：C 线 natural density confirmation

### 目标

确认自然 AP0 harvesting 是否正式降级为非主线。

v10.0 已经显示 largest panel = 10000 且 density insufficient。v10.1 不再无限扩大自然 panel，只做有限确认。

### 实验设计

```text
C1: repeat 5000 panel with new seed
C2: confirm 10000 panel with new seed
C3: optional 20000 only if C1/C2 contradictory
```

### 必须记录

```text
panel_id
panel_size
seed
generator_id
major_psi
major_js
tail_psi
tail_js
missing_tail
CoreLike_count/LCB/UCB
FastGood_count/LCB/UCB
SlowBurnGood_count/LCB/UCB
CoreLikePlusSlowBurn_count/LCB/UCB
RiskyHighAUV_rate
BadPath_rate
throughput_rows_sec
wallclock
peak_gpu_mb
fake/proxy/cpu
```

### 判断标准

Natural harvesting sufficient：

```text
CoreLikePlusSlowBurn_LCB >= 0.03
or PathGood_LCB >= 0.03
```

Natural harvesting insufficient：

```text
CoreLikePlusSlowBurn_UCB < 0.03
and PathGood_UCB < 0.03
in two independent panels including one >=10000.
```

Natural harvesting inconclusive：

```text
LCB < 0.03 <= UCB
or generator fidelity fails
or panels disagree strongly.
```

### 可视化

```text
p2_density_ci_by_panel_size.svg
p2_path_type_density_by_seed.svg
p2_tail_group_good_density_heatmap.svg
p2_natural_vs_canonical_path_type_distribution.svg
```

### 不满足时 Codex 先尝试

```text
如果 generator fidelity fail：
  回到 G40 sampler audit，不要继续 panel。

如果 panel throughput 太慢：
  降 chunk-actions，保留 exact rows；不允许 proxy density。

如果 5000 sufficient 但 10000 insufficient：
  检查 seed / tail group collapse / duplicate action。

如果 density insufficient 成立：
  将 C 线降为 monitoring，不再主导路线。

如果 density sufficient 成立：
  打开 existing-action controller candidate，但必须仍过 B/A risk gates。
```

---

## 8. P3：B 线 Path-Type FPO v12

### 目标

把 FPO 从单一 score 改成路径类型预测器，尤其识别 SlowBurnGood 和排除 RiskyHighAUV。

### 候选 FPO

```text
FPO12A-tiny-virtual-adamw-path-1step
FPO12B-tiny-virtual-adamw-path-3step
FPO12C-jvp-vjp-future-gradient-alignment
FPO12D-signal-reservoir-path-type-classifier
FPO12E-memory-offdiag-veto-plus-delayed-gain
FPO12F-path-type-contrastive-ranker
FPO12G-slowburn-specialist
FPO12H-negative-control-shuffled-features
```

### 输入字段限制

允许：

```text
current train batch response
train-memory buffer response
hard-tail response
per-example gradient mean/variance
JVP/VJP sketch
old-family / old-stratum response
memory/offdiag legal proxy
payload/action norm
AdamW alignment
cost estimate
```

禁止：

```text
future V labels as commit feature
GradeAB / FastGood / SlowBurnGood labels as commit feature
dataset name branch
validation/test metric
outcome-derived score
old table score
```

### 必须记录

```text
fpo_id
feature_set_id
cost_ms_q50/q90/q99
TopK87_FastGood_precision
TopK87_SlowBurnGood_precision
TopK87_FastOrSlow_precision
TopK87_RiskyHighAUV_rate
TopK87_BadPath_rate
V1/V20/V80/V240_LCB
RAUV_LCB
longrisk_UCB
bad_UCB
null_UCB
memory/offdiag_UCB
LDO/LSO/LTO/LFO
false_positive_type_distribution
false_negative_type_distribution
```

### Discovery pass

```text
FastOrSlow_precision >= 0.10
V240_LCB > -0.10
longrisk_UCB <= 0.25
BadPath_rate <= 0.50
cost_q90 <= 2.0 ms
```

### Weak pass

```text
FastOrSlow_precision >= 0.25
V240_LCB > 0
longrisk_UCB <= 0.10
bad_UCB <= 0.05
null_UCB <= 0.15
cost_q90 <= 1.5 ms
```

### Strong / controller candidate pass

```text
accepted_count >= 87
FastOrSlow_precision >= 0.75
V240_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory/offdiag_UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
cost_q90 <= 1.0 ms
```

### 可视化

```text
p3_fpo_path_type_confusion_matrix.svg
p3_fpo_false_positive_breakdown.svg
p3_fpo_false_negative_breakdown.svg
p3_fpo_cost_vs_quality_pareto.svg
p3_fpo_v240_vs_longrisk_scatter.svg
p3_slowburn_recall_by_feature_group.svg
```

### 不满足时 Codex 先尝试

```text
如果 precision 很低但 longrisk 低：
  该 FPO 是 veto，不是 selector；拆成 risk veto，另加 delayed-gain selector。

如果 Fast/Slow precision 高但 longrisk 高：
  加 memory/offdiag hard gate；不要调 value threshold。

如果 SlowBurn recall = 0：
  加 V1-negative / delayed-gain contrastive target；不要只用 current response。

如果 cost 高：
  降低 virtual samples；缓存 JVP/VJP；只保留 top feature group。

如果 LDO 高：
  检查是否 path type 只集中在某 tail/dataset/family；不按 dataset 调参。
```

---

## 9. P4：D 线 FuturePath-targeted generated discovery v2

### 目标

不再生成“合法 payload”，而是生成候选 future-path-improving update。D 线仍是 discovery，不写 official pass。

### 生成 family

```text
DGEN1-slowburn-delayed-gain-residual
DGEN2-risk-clean-future-value-correction
DGEN3-memory-offdiag-preserving-update
DGEN4-jvp-aligned-future-gradient-update
DGEN5-trust-region-path-type-anchor
DGEN6-negative-control-shuffled-path-target
```

### 阶梯执行

```text
Stage 8:
  每个 family 8 actions，总量不超过 48。
  如果 Fast/Slow-like = 0 且 V240_LCB < 0，停止该 family。

Stage 32:
  只有 Stage 8 出现 >=1 Fast/Slow-like 或 V240_LCB > -0.10 才打开。

Stage 64:
  只有 Stage 32 discovery pass 才打开。

禁止直接 512/1024 大跑。
```

### 必须记录

```text
generator_id
family_id
action_id
source_anchor_id
payload_norm
payload_linf
payload_cosine_to_adamw
payload_cosine_to_fastgood_anchor
payload_cosine_to_slowburn_anchor
action_apply_linf
FastGood_precision
SlowBurnGood_precision
FastOrSlow_precision
V1/V20/V80/V240_LCB
RAUV_LCB
longrisk_UCB
bad_UCB
null_UCB
memory/offdiag_UCB
new_positive_created_rate
longrisk_created_rate
Damage_V240_LCB
failure_type
```

### Stage 8 pass

```text
FastOrSlow_count >= 1
or V240_LCB > -0.10 and longrisk_UCB <= 0.25
```

### Stage 32 pass

```text
FastOrSlow_precision >= 0.05
V240_LCB > -0.05
longrisk_UCB <= 0.20
BadPath_rate <= 0.70
```

### Stage 64 weak pass

```text
FastOrSlow_precision >= 0.10
V240_LCB > 0
longrisk_UCB <= 0.15
bad_UCB <= 0.10
memory/offdiag_UCB <= 0.10
new_positive_created_rate > 0
```

### 可视化

```text
p4_generated_stage_waterfall.svg
p4_generated_path_type_distribution.svg
p4_generated_v240_vs_longrisk.svg
p4_generated_payload_to_anchor_cosine.svg
p4_generated_damage_modes.svg
p4_stage8_stop_reason_by_family.svg
```

### 不满足时 Codex 先尝试

```text
如果 Stage 8 全部 BadPath：
  停止该 family；检查 payload norm / AdamW conflict / memory-offdiag fail。

如果 high AUV high risk：
  增加 risk hard gate；不要调 AUV。

如果 low risk low value：
  检查是否 SafeLowValue；加 delayed-gain target。

如果 action apply error 非 0：
  修 payload materializer，不评估 outcome。

如果 generated 与 anchor cosine 极低：
  加 trust-region / anchor projection。

如果 longrisk UCB 高：
  检查 memory/offdiag/hardtail；不扩大 stage。
```

---

## 10. P5：Failure autopsy and theory update

### 目标

如果 B/D 仍失败，不能只写“失败”。必须明确 failure mechanism。

### Failure categories

```text
F1 path-type target too sparse
F2 FPO target mismatch
F3 generated direction OOD
F4 memory/offdiag veto insufficient
F5 AdamW conflict
F6 no delayed gain mechanism
F7 natural density source insufficient
F8 runtime infeasible
F9 branch-horizon/materializer issue
```

### 必须记录

```text
failure_category
assigned_count
assigned_fraction
evidence_metric
first_failed_gate
recommended_next_attempt
stop_or_continue
```

### 不满足时 Codex 先尝试

```text
如果 assigned_fraction < 0.80：
  增加 diagnostic fields，而不是新增 generator。

如果 F2 为主：
  重写 FPO target，不调 threshold。

如果 F3 为主：
  引入 anchor / trust region / natural-tail projection。

如果 F6 为主：
  改造 generated objective 为 delayed-gain residual。
```

---

## 11. P6：条件性 controller boundary

### 打开条件

只有以下任一成立：

```text
C 线 natural density sufficient；
B 线 FPO strong pass；
D 线 generated Stage64 weak pass；
```

否则 controller 必须 `not_run`。

### Controller pass 标准

```text
accepted_count >= 87
FastOrSlow_precision >= 0.75
V240_LCB > 0
RAUV_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory/offdiag_UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
feature_cost_q90 <= 1.0 ms
payload_apply_q90 <= runtime budget
```

### 不满足时 Codex 先尝试

```text
如果 accepted_count < 87：
  不放宽 risk；回到 density/generation source。

如果 precision 低：
  查 false positive path types，不调 dataset threshold。

如果 longrisk 高：
  加 memory/offdiag veto，重新评估。

如果 LDO 高：
  查 tail/family/template concentration，不按 dataset 调参。
```

---

## 12. P7：条件性 selected runtime

### 打开条件

```text
P6 controller pass = 1
```

### 必须记录

```text
controller_id
accepted_per_step
active_step_count
zero_candidate_step_count
kernel_count
sync_count
feature_cost_ms_q90
payload_apply_ms_q90
step_ratio_q50/q90/q99
memory_peak_mb
```

### Runtime pass

```text
step_ratio_q90 <= 1.50
no CPU offload
no fake/proxy runtime estimate
```

### 不满足时 Codex 先尝试

```text
如果 feature cost 高：缓存 FPO sketch，裁剪 feature set。
如果 payload apply 高：batch-major grouping，fuse payload apply。
如果 empty-step overhead 高：active-step compaction。
如果 kernel count 高：persistent workspace / grouped kernel。
```

---

## 13. P8：paired replay / short-full boundary

### 打开条件

```text
P6 controller pass = 1
P7 runtime pass = 1
```

### 必须比较

```text
RealFunctional
AdamWParallel
bestLR
NoOp
RandomAction
ShuffledPayload
GeneratedNegativeControl
```

### Pass 标准

```text
RealFunctional beats AdamWParallel / bestLR / NoOp / Random in V20/V80/V240;
shuffled payload fails;
longrisk remains <= 0.05;
no memory/offdiag fail;
short-run sample efficiency improves;
full-run not worse than matched MLP baseline.
```

---

## 14. v10.1 route decision

最终必须落入一个 route：

```text
R1-NaturalDensityConfirmedSufficient_OpenController
R2-NaturalDensityInsufficient_FPOPathTypeWeak
R3-NaturalDensityInsufficient_GeneratedDiscoveryWeak
R4-FPOAndGeneratedBothFail_UpdateRuleTheoryReset
R5-GeneratedFamilyPartiallyWorks_OpenStage64
R6-ControllerPass_OpenRuntime
R7-SystemPass_OpenPairedReplay
```

---

## 15. v10.1 成功与失败的真正含义

### 如果 B 线过而 D 线不过

说明好动作可以被识别但难生成。路线转为 controller-first。

### 如果 D 线过而 B 线不过

说明可以生成但不能识别。路线转为 generated update + lightweight certificate。

### 如果 B/D 都不过

说明当前 future-path target 仍未被压缩成可操作机制。此时应停止 APG-style generation，回到更高层的 update-rule theory。

### 如果 C 线重新 sufficient

说明自然 harvesting 仍可做，但必须解释为什么 v10.0 低密度。否则只作为备选。

---

## 16. 本轮最重要的原则

v10.1 不再追求“多跑一点可能出好动作”。

它只追求三个硬问题：

```text
1. natural density 是否真的不足；
2. future path type 能否训练当下识别；
3. future path type 能否主动生成。
```

如果三个问题都给出否定答案，就要承认：当前 functional update primitive 仍没有抓住网络几何本质，需要重新定义 update rule，而不是继续在当前 primitive family 内修补。
