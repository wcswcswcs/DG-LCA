# DG-KAN v9.9.5 结果解读与 v9.9.6 下一步实验计划

> 本文面向没有项目背景的读者，解释 v9.9.5 到底做了什么、有什么进展、为什么仍然很慢、四条线分别卡在哪里，以及 v9.9.6 应该如何并行推进。  
> 公式均使用 Typora 友好的 `$...$` 或 `$$...$$` 格式。

---

## 0. 一句话判断

v9.9.5 有进展，但不是能力成功。它把最关键的 C 线从“tail key 太碎，没法采样”推进到“tail key 已经可采样”，但随后发现新的 generator 仍然无法同时满足普通分布和稀有尾部分布。因此，密度裁决、controller、generated sandbox、runtime、paired replay 仍然都不能打开。

最准确的一句话是：

$$
\boxed{
\text{tail key 已经修到可采样；但自然动作生成器仍不能保住好动作可能所在的尾部。}
}
$$

这不是“小阈值没调好”。它说明当前问题已经从“有没有自然动作生成入口”变成了更深的采样问题：

```text
我们想估计自然 AP0 动作流中好动作的密度；
但如果生成器没有采到好动作可能所在的 tail，
那么任何 5000 / 10000 / 20000 panel 的密度结论都会是错的。
```

---

## 1. v9.9.5 的真实结果

v9.9.5 的 route 是：

```text
route = R2-TailKeyPassGeneratorFidelityFail
primary_blocker = no_major_tail_fidelity_generator_passed
secondary_blocker = density_panels_blocked
system_legal_controller_pass = 0
generated_route_status = stopped_generator_fidelity_failed
```

这说明本轮不是 functional success，也不是 controller success。

本轮的主要事实是：

```text
P0 boundary pass = 1
P1 tail key weak/strong pass = 1 / 1
P2 G13-G19 official/weak pass count = 0 / 0
P3 density panel largest completed size = 0
P4 future path weak/strong = 0 / 0
P5 FPO weak/strong = 0 / 0
P6 controller = not_run
P7 generated sandbox allowed = 0
P8 runtime / paired replay = 0 / 0
fake/proxy/cpu = 0 / 0 / 0
```

### 1.1 最大进展：tail key 修好了第一层

v9.9.4 的 tail key 太碎：原始 tail key 有 `2191` 个 group，其中大量 group support 很低。v9.9.5 先审计了 TK0-TK4，并选中：

```text
TK2-hierarchical-major-tail-key
number of groups = 337
support<3 fraction = 0.12166172106824925
max share = 0.01773296244784423
entropy ratio = 0.9291338663309526
weak / strong = 1 / 1
```

这是真进展。它说明我们不再卡在“tail key 过碎，根本不可采样”。

但是 tail key 可采样，不等于 generator 能采样好。

### 1.2 最大失败：G13-G19 没有一个通过 major + tail fidelity

v9.9.5 尝试了 7 个 generator 候选：G13-G19。结果没有 official pass，也没有 weak pass。最接近的是：

```text
G18-sequential-rejection-generator
major PSI = 0.03448359529300191
major JS = 0.06492298915120376
major max share = 0.0234375

tail PSI = 0.2473464173705414
tail JS = 0.13073471127743141
missing tail group = 23
tail coverage = 0.9317507418397626
```

G18 的 major 分布已经接近可接受，但 tail 仍然严重不对。这个结果说明：

$$
\boxed{
\text{当前 generator 主要问题不是粗分布，而是无法覆盖稀有 tail。}
}
$$

### 1.3 P5 FPO v7 明确失败

v9.9.5 还评估了 FPO v7，但结果很差。最好的 `FPO7D-signal-reservoir-snr-gate` 只有：

```text
precision = 0.011494252873563218
V LCB = -0.2744455578755013
longrisk UCB = 0.9446155167180331
cost q90 = 0.0009918585419654846 ms
```

这说明 FPO v7 的问题不是太慢，而是便宜但选错。它并没有预测 future path benefit，反而选到高 longrisk、负价值动作。

---

## 2. 四线分别怎么看

## 2.1 A 线：未来训练路径

A 线这轮没有真正推进，因为 P2 generator fidelity 不过，P3/P4 被 gate-block。不能因此说 A 线失败。

从 v9.8.2 到 v9.9.4，A 线已经有一条稳定线索：好动作不一定是当前一步降 loss 最多，而可能是后续训练路径变好。特别是 SlowBurnGood 这类动作：

```text
V1 可能为负；
V20 / V80 / V240 逐渐转正；
longrisk 低；
memory / offdiag 低。
```

这说明我们真正要找的不是：

$$
\Delta L_1 < 0
$$

而是：

$$
\Delta Path_{1,5,20,80,240}(a) \text{ 是好路径。}
$$

更直白地说：

```text
不是找当前一步 loss 降最多的动作；
而是找把后续训练带到更好轨迹的小参数改动。
```

v9.9.6 中 A 线不能只看 AUV，要把路径分成：

```text
FastGood：h1 就好，后面也好；
SlowBurnGood：h1 不好或一般，但 h20/h80/h240 变好；
RiskyHighAUV：AUV 高，但 longrisk/memory/offdiag 高；
SafeLowValue：风险低，但 value 不够；
BadPath：value 不好且风险高。
```

### A 线当前状态

```text
现象：可信；
训练当下机制：没有；
controller：没有；
下一步：在保真 natural panel 上重验 path type density 和稳定性。
```

---

## 2.2 B 线：训练当下 FuturePathOperator

B 线这轮明确失败。FPO7A-H 全部没有 weak/strong pass，其中 FPO7D 虽然成本很低，但 precision 接近 0、V 为负、longrisk 很高。

这说明当前 FPO 还没有测到我们真正关心的东西。它可能测到的是：

```text
当前函数响应；
局部 SNR；
风险过滤；
硬尾样本响应；
某种 immediate response。
```

但它没有测到：

```text
这个动作是否会让未来训练路径变好。
```

所以 B 线不能继续做这些事情：

```text
调 FPO7D threshold；
调 SNR gate；
调 hard-tail / memory veto；
继续扫 cheap response feature。
```

B 线下一步要改成真正的 future-path operator sketch：

```text
低成本估计：
  action 加进去后，下一步或后几步训练动力学是否更容易优化；
  是否进入 signal channel；
  是否降低 old-family / hard-tail / memory risk；
  是否避免 longrisk。
```

### B 线当前状态

```text
当前 cheap FPO：失败；
失败原因：选错动作，不是成本问题；
下一步：重写为 future operator sketch，不再调旧阈值。
```

---

## 2.3 C 线：自然 AP0 动作扩流

C 线是本轮主线，也是当前最高优先级。

最近几轮 C 线进展是：

```text
v9.8.9：找不到 natural extension generator entrypoint；
v9.9.0：entrypoint 落地，single/16/256 smoke 通过；
v9.9.1：1024 pilot 跑通，但 major distribution 失真；
v9.9.2：G5 让 major distribution 看起来过线；
v9.9.3：加入 tail fidelity 后发现 G5/G6 不可信；
v9.9.4：G7-G12 全部不能同时保 major + tail；
v9.9.5：tail key 修成 TK2，但 G13-G19 仍不能通过 major + tail fidelity。
```

这说明 C 线不是原地踏步，而是在一步步逼近真实 blocker。

当前最重要的数学判断是：

$$
P_{natural}(GoodAction)
=
\sum_z P(GoodAction\mid z)P_{natural}(z).
$$

如果好动作集中在某些稀有 tail group $z$，那么只让 major distribution 像是不够的。必须让：

$$
P_G(z_{tail}) \approx P_{natural}(z_{tail}).
$$

否则：

$$
P_G(GoodAction) \neq P_{natural}(GoodAction).
$$

也就是说：

$$
\boxed{
\text{好动作密度是 tail-conditional density 问题，不是 global density 问题。}
}
$$

### C 线当前状态

```text
tail key：已从不可采样推进到可采样；
generator：仍无法同时满足 major + tail；
density panel：不能打开；
controller：不能打开；
下一步：做 tail coverage root-cause + generator repair。
```

---

## 2.4 D 线：生成新动作

D 线继续停止是正确的。

原因很简单：

```text
A 线只有事后 future path 现象；
B 线没有合法低成本 operator；
C 线没有保真 natural density；
```

在这种情况下打开 generated route，等于没有目标函数地生成动作。过去 APG/APGA/APGF/APGT 多轮都已经证明：

```text
payload 合法；
action apply 闭合；
branch-horizon 能跑；
但 value-negative / high-longrisk。
```

所以 D 线现在不是主攻线。它只能在以下任一条件满足时重开：

```text
1. C 线证明 natural density 不足，需要新生成动作源；
2. B 线找到可用 FuturePathOperator，可以作为生成目标；
3. A 线给出明确 path type 机制，可以转成生成目标。
```

---

## 3. 为什么感觉还是非常慢

你觉得非常慢是合理的，因为系统能力链路还没打开：

```text
controller -> selected runtime -> paired replay -> short/full training
```

但慢的本质不是“没有进展”，而是每轮都在阻止一个错误结论：

```text
v9.9.0 阻止把 256 smoke 当 density closure；
v9.9.1 阻止把失真 generator 的低密度当 natural source density；
v9.9.2 阻止把 major distribution pass 当 density closure；
v9.9.3 阻止忽略 tail fidelity；
v9.9.4 阻止用 tail-failed G7-G12 打开 5000 panel；
v9.9.5 阻止把 tail key pass 当 generator fidelity pass。
```

这类进展不涨 acc，也不打开 controller，但它非常重要。因为一旦误判 density，就会把整个路线导向错误方向。

---

## 4. 当前真正卡在哪里

## 4.1 卡在 tail coverage root cause

v9.9.5 已经把 tail key 修到可采样，但 G18 仍 missing 23 个 tail group。现在必须回答：这些 missing tail group 到底为什么消失？

可能原因包括：

```text
1. quota rounding 到 0；
2. tail group 虽然定义存在，但可生成 precursor 不存在；
3. filter 删除了这些 tail；
4. cursor 或 refill 逻辑偏向少数 group；
5. major quota 和 tail quota 冲突；
6. payload collision / action dedup 删除了 tail；
7. tail key 仍然过细或语义不对；
8. materializer 只能 replay 某些 recipe，无法生成某些 tail recipe。
```

下一轮必须逐个 missing tail group 记录：

```text
tail_group_id
canonical_count
expected_quota
rounded_quota
candidate_precursor_count
generated_count_before_filter
generated_count_after_filter
dedup_removed_count
filter_removed_count
cursor_state
major_group
recipe_id
payload_collision_count
action_collision_count
missing_reason
```

## 4.2 卡在 generator 是 sampling problem，不是 score problem

G13-G19 的失败说明：不能再只换一个 generator 名字。我们需要把自然生成器建模成一个约束采样问题：

$$
\min_G D_{major}(P_G,P_{ref}) + \lambda D_{tail}(P_G,P_{ref})
$$

subject to：

$$
missing\_tail\_group=0,
$$

$$
max\_share \le \tau,
$$

$$
collision=0,
$$

$$
action\_apply\_error=0.
$$

这不是调阈值，而是 generator distribution matching。

## 4.3 卡在 B 线没有 future operator

FPO7D 的 precision 只有 `0.01149`，说明当前 signal-reservoir SNR gate 没有捕捉好动作。它可能只是在看局部 response 或简化信号，而好动作的价值在未来路径上。

B 线下一步必须把目标改成 path-type classification：

```text
FastGood vs SlowBurnGood vs RiskyHighAUV vs SafeLowValue vs BadPath
```

尤其是 SlowBurnGood，因为它正是 immediate-loss 规则看不到的好动作。

## 4.4 卡在 D 线没有生成目标

没有 C 的 density 裁决，也没有 B 的 operator，D 线不能开。generated sandbox 继续停止是正确的。

---

## 5. 是否还在正确道路上

高层方向仍然对。因为项目没有做这些错误事情：

```text
没有把 tail-key pass 写成 density pass；
没有把 G18 best-failed 写成 generator pass；
没有把 1024 distribution-only 当成 5000 density；
没有把 FPO7D 写成 controller；
没有打开 generated sandbox；
没有 fake/proxy/cpu rows；
没有按 dataset 调参。
```

但路线必须更硬。下一步不要继续：

```text
调 FPO7D threshold；
调 AUV threshold；
只看 major PSI；
直接跑 5000 / 10000 / 20000；
把 tail-failed generator 的 density 当结论；
没有 B/C 证据就重开 generated sandbox。
```

---

## 6. 离目标还差多远

离 system-legal local functional controller 至少还差四道硬门：

```text
1. natural generator 同时通过 major fidelity 和 tail fidelity；
2. CoreLike / PathGood / SlowBurnGood density 在保真 full natural panel 上被裁决；
3. 训练当下合法、低成本、跨分布稳定的 FPO 或 controller 过线；
4. selected runtime step_ratio_q90 <= 1.50。
```

离 strict PureKAN functional causal evidence 还要：

```text
official paired replay；
RealFunctional beats AdamWParallel / bestLR / NoOp / Random；
shuffled payload fail；
leave-dataset-out / leave-stratum-out / leave-template-out。
```

离 external-ready Beyond-MLP 还要：

```text
short/full training；
sample efficiency；
calibration；
robustness；
continual / anti-forgetting；
strong baseline 排除；
external reproducibility。
```

当前最准确状态是：

$$
\boxed{
\text{future path 好动作现象可信；tail key 已修好；但 generator 仍无法保住 tail，density/controller/generated 都不能打开。}
}
$$

---

# v9.9.6 下一步完整实验计划

## 7. v9.9.6 总目标

v9.9.6 不再小修 FPO7D、AUV、CoreExpansion，也不直接跑 5000 panel。它的整体目标是：

$$
\boxed{
\text{找出 major+tail generator fidelity 失败的根因，并产出一个能进入 density panel 的自然动作生成器。}
}
$$

同时继续推进 A/B/D 三线：

```text
A 线：继续拆 future path 类型，尤其 SlowBurnGood；
B 线：重写 FuturePathOperator，改成 path-type sketch；
C 线：最高优先级，修 major+tail fidelity；
D 线：严格 gate，只在 A/B/C 给证据后重开 generated sandbox。
```

---

## 8. P0：复现 v9.9.5 边界

### 目标

确认 v9.9.5 的 route、artifact、tail key、generator failure、FPO failure 都能被读取并复现。

### 必须记录

```text
source_route
P1_tail_key_selected
P1_weak_pass
P1_strong_pass
P2_candidate_count
P2_official_pass_count
P2_weak_pass_count
best_failed_generator_id
best_failed_major_PSI
best_failed_tail_PSI
best_failed_missing_tail_count
P3_largest_completed_panel_size
P5_best_FPO
P5_best_precision
P5_best_V_LCB
P5_best_longrisk_UCB
fake/proxy/cpu audit
```

### 通过标准

```text
P0_boundary_pass = 1
fake/proxy/cpu = 0/0/0
```

### 不满足时 Codex 先尝试

```text
如果 v9.9.5 artifacts 读不到：
  检查 out_dir 路径、manifest hash、route json 文件名。

如果 tail key / generator table 缺失：
  检查 runner 是否写入 p1_tail_key_audit_v9950.csv 和 p2_generator_repair_matrix_v9950.csv。

如果 fake/proxy/cpu audit 异常：
  停止后续所有阶段，先修 no-fake audit。
```

---

## 9. P1：missing tail group root-cause audit

### 目标

不要再盲目加 generator。先对 G13-G19，特别是 G18 的 missing tail groups 做逐 group 根因分析。

### 假设

$$
H_{P1}:
\text{missing tail group 不是随机失败，而是 quota/cursor/filter/dedup/recipe 的系统性问题。}
$$

### 必须记录

对每个 generator、每个 tail group 记录：

```text
generator_id
tail_group_id
major_group_id
canonical_tail_count
canonical_major_count
reference_tail_prob
reference_major_prob
expected_tail_quota_float
expected_tail_quota_int
quota_rounded_to_zero
candidate_precursor_count
sampled_before_filter
sampled_after_filter
filter_removed_count
dedup_removed_count
payload_collision_count
action_collision_count
cursor_start
cursor_end
cursor_exhausted
refill_attempt_count
recipe_id
recipe_available
missing_reason
```

### missing_reason 分类

```text
MR1-quota-rounded-to-zero
MR2-no-precursor-available
MR3-filter-deleted-tail
MR4-dedup-or-collision-deleted-tail
MR5-cursor-collapse
MR6-major-tail-quota-conflict
MR7-recipe-not-implemented
MR8-tail-key-too-fine
MR9-unknown
```

### 判断标准

P1 不是 pass/fail 阶段，而是必须完成 attribution：

```text
assigned_missing_reason_fraction >= 0.95
unknown_missing_reason_fraction <= 0.05
```

### 可视化

```text
p1_missing_tail_reason_stacked_bar.svg
p1_tail_group_expected_vs_generated_scatter.svg
p1_quota_rounding_heatmap.svg
p1_filter_dedup_loss_waterfall.svg
p1_major_tail_conflict_matrix.svg
```

### 不满足时 Codex 先尝试

```text
如果 assigned fraction < 0.95：
  增加 per-tail debug trace，不要继续 P2。

如果 MR1 占比高：
  改 quota rounding：rare group min quota = 1；再用 importance weights 修 major 分布。

如果 MR2 占比高：
  找 canonical precursor replay source；如果 precursor 真的不存在，说明 tail key 仍含不可生成 group。

如果 MR3 占比高：
  输出 filter reason；将 filter 从 hard drop 改成 repair/refill。

如果 MR4 占比高：
  dedup 后必须 tail refill；payload collision 不允许静默删除。

如果 MR5 占比高：
  改 cursor 为 per-tail independent cursor，不共享 global cursor。

如果 MR6 占比高：
  用 minimum-cost flow sampler，不再手写 quota。

如果 MR8 占比高：
  回到 tail key 合并，优先合并 support<5 且 outcome-free 的相近 group。
```

---

## 10. P2：G20-G29 major+tail generator repair matrix

### 目标

基于 P1 根因，不再随机新增 generator，而是用不同采样理论修复 major+tail 冲突。

### Generator candidates

```text
G20-minquota-tail-refill-generator
  每个 tail group 至少 1 quota，major 分布通过 importance weights 修正。

G21-mincost-flow-major-tail-generator
  把 major 和 tail quota 当成双边约束，解最小代价流。

G22-tail-first-major-residual-generator
  先填 tail，剩余容量按 major residual 填。

G23-major-first-tail-residual-generator
  先保 major，再对 missing tail 做 residual refill。

G24-per-tail-independent-cursor-generator
  每个 tail group 单独 cursor，防 global cursor collapse。

G25-tail-repair-after-filter-generator
  filter/dedup 后检查 missing tail，再 refill。

G26-canonical-tail-precursor-replay-generator
  从 canonical tail precursor replay，再加合法轻扰动。

G27-entropy-regularized-generator
  用 entropy regularization 控制 max share。

G28-importance-weighted-density-generator
  允许采样偏 tail，但 density 用 importance weight 修正；只作为 diagnostic，不直接 official。

G29-two-stage-mixture-generator
  G21 main + G26 rare-tail replay mixture。
```

### 必须记录

```text
generator_id
major_PSI
major_JS
major_max_share
major_entropy_ratio
tail_PSI
tail_JS
missing_tail_group_count
tail_coverage
support_less_3_generated_fraction
old_action_collision_count
payload_collision_count
action_apply_error_linf_max
throughput_rows_per_sec_estimate
official_density_eligible
weak_pass
official_pass
```

### Pass 标准

Weak pass：

```text
major_PSI <= 0.08
major_JS <= 0.10
tail_PSI <= 0.12
tail_JS <= 0.12
missing_tail_group_count <= 3
major_max_share <= 0.40
```

Official pass：

```text
major_PSI <= 0.05
major_JS <= 0.08
tail_PSI <= 0.05
tail_JS <= 0.08
missing_tail_group_count = 0
major_max_share <= 0.35
major_entropy_ratio >= 0.90
old_action_collision_count = 0
payload_collision_count = 0
```

### 可视化

```text
p2_generator_major_tail_pareto.svg
p2_generator_missing_tail_bar.svg
p2_major_vs_tail_PSI_scatter.svg
p2_max_share_vs_entropy.svg
p2_generator_collision_audit.svg
```

### 不满足时 Codex 先尝试

```text
如果没有 weak pass：
  不跑 branch-horizon；回到 P1 missing reason top-2，做针对性修复。

如果 major pass but tail fail：
  增加 tail residual refill，不调 major threshold。

如果 tail pass but major fail：
  增加 major projection / min-cost correction，不删除 tail。

如果 missing_tail_group_count > 0：
  逐 group 输出 expected quota / actual count / missing reason；优先修缺失 group。

如果 max_share > 0.35：
  加 entropy cap 或 per-major max quota。

如果 collision > 0：
  collision 后必须 refill，不能静默删除。

如果 G21/G29 成本高：
  先做 distribution-only 1024，不进入 branch-horizon。
```

---

## 11. P3：1024 branch-horizon pilot and sequential density panel

### 目标

只有 P2 至少有 weak generator pass，才打开 branch-horizon pilot。只有 1024 pilot 质量过，才 sequentially 打开 5000 / 10000 / 20000 panel。

### 必须记录

```text
panel_size
generator_id
action_count
branch_horizon_rows_expected
branch_horizon_rows_actual
completion_rate
rows_per_sec
peak_gpu_mb
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
RiskyHighAUV_count
SafeLowValue_count
BadPath_count
LongRisk_rate
bad_rate
null_rate
memory_fail_rate
offdiag_fail_rate
```

### Density decision

对于每个 target $T$：

$$
\hat p_T = \frac{N_T}{N}.
$$

用 Wilson CI 记录：

$$
LCB_T, UCB_T.
$$

判断：

```text
if CoreLike_LCB >= 0.03 or PathGood_LCB >= 0.03 or SlowBurnGood_LCB >= 0.03:
    natural_density_sufficient = 1
elif CoreLike_UCB < 0.03 and PathGood_UCB < 0.03 and SlowBurnGood_UCB < 0.03 at N >= 10000:
    natural_density_insufficient = 1
else:
    continue sequential panel
```

### 可视化

```text
p3_density_curve_corelike_pathgood_slowburn.svg
p3_wilson_ci_vs_panel_size.svg
p3_path_type_stacked_bar.svg
p3_risk_memory_offdiag_by_path_type.svg
p3_throughput_vs_panel_size.svg
```

### 不满足时 Codex 先尝试

```text
如果 1024 branch-horizon completion < 1：
  修 materializer，不继续 5000。

如果 rows/sec 过低：
  分 chunk、resume、clear cache；不要降低 correctness。

如果 1024 density 极低但 generator weak only：
  不下自然源不足结论；优先找 official fidelity generator。

如果 5000 后 LCB < 0.03 <= UCB：
  继续 10000，不 premature stop。

如果 10000 后 UCB < 0.03：
  可以判定 natural source density likely insufficient，但先检查 tail coverage 是否仍 pass。

如果 density sufficient 但 path risk 高：
  进入 A/B 线重验，不直接 controller。
```

---

## 12. P4：A 线 future path type revalidation on faithful panel

### 目标

在保真 natural panel 上验证好路径类型是否仍然存在。

### 路径类型定义

```text
FastGood:
  V1_LCB > 0, V20_LCB > 0, V80_LCB > 0, V240_LCB > 0,
  longrisk_UCB <= 0.05,
  memory/offdiag_UCB <= 0.05.

SlowBurnGood:
  V1_LCB <= 0,
  V20_LCB > 0,
  V80_LCB > 0,
  V240_LCB > 0,
  longrisk_UCB <= 0.05,
  memory/offdiag_UCB <= 0.05.

RiskyHighAUV:
  RAUV_LCB > 0,
  longrisk_UCB > 0.20 or memory/offdiag_UCB > 0.20.

SafeLowValue:
  longrisk_UCB <= 0.05,
  memory/offdiag_UCB <= 0.05,
  RAUV_LCB <= 0.

BadPath:
  V240_LCB <= 0 or longrisk_UCB > 0.50.
```

### 必须记录

```text
path_type
action_count
rate
LCB/UCB
V1_LCB
V5_LCB
V20_LCB
V80_LCB
V240_LCB
RAUV_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO_drop
LSO_drop
LTO_drop
LFO_drop
```

### Pass 标准

```text
FastGood_count + SlowBurnGood_count >= 87
FastGood_or_SlowBurnGood_LCB >= 0.03
V240_LCB > 0
longrisk_UCB <= 0.05
memory/offdiag_UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
```

### 可视化

```text
p4_future_path_curves_by_type.svg
p4_slowburn_vs_fastgood_scatter.svg
p4_auv_vs_longrisk_scatter.svg
p4_path_type_leaveout_heatmap.svg
```

### 不满足时 Codex 先尝试

```text
如果 SlowBurnGood 少：
  检查 h1 条件是否太严；先用 SlowBurnGoodRelaxed，但不能放宽 h240 risk。

如果 RiskyHighAUV 多：
  增强 risk-adjusted path gate，不调 AUV。

如果 path good 只在某 dataset/family：
  记录为 leaveout fail，不按 dataset 调参。

如果 path type 与 old Core77 不一致：
  做 intersection/old-only/new-only 对照。
```

---

## 13. P5：B 线 FuturePathOperator v8

### 目标

停止 FPO7D 阈值修补，转向 path-type prediction。FPO 要预测 FastGood / SlowBurnGood，而不是当前响应大小。

### Candidate sketches

```text
FPO8A-slowburn-signal-reservoir-sketch
FPO8B-tiny-virtual-adamw-path-3step
FPO8C-jvp-gradient-transport-risk-veto
FPO8D-memory-hardtail-delayed-gain-sketch
FPO8E-fastgood-slowburn-two-head-sketch
FPO8F-risky-high-AUV-veto-sketch
FPO8G-signal-channel-drift-diffusion-sketch
FPO8H-lowrank-future-path-operator-sketch
```

### 必须记录

```text
fpo_id
feature_cost_q90_ms
TopK87_FastGood_precision
TopK87_SlowBurnGood_precision
TopK87_PathGood_precision
V_LCB
V240_LCB
RAUV_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO_drop
LSO_drop
LTO_drop
LFO_drop
false_positive_type_distribution
false_negative_type_distribution
```

### Weak pass

```text
TopK87_PathGood_precision >= 0.50
V_LCB > 0
longrisk_UCB <= 0.10
feature_cost_q90_ms <= 1.50
```

### Strong pass

```text
TopK87_PathGood_precision >= 0.75
accepted_count >= 87
V_LCB > 0
V240_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory/offdiag_UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
feature_cost_q90_ms <= 0.50
```

### 可视化

```text
p5_fpo_precision_vs_cost.svg
p5_fpo_value_vs_longrisk.svg
p5_fpo_false_positive_types.svg
p5_fpo_false_negative_types.svg
p5_fpo_leaveout_heatmap.svg
```

### 不满足时 Codex 先尝试

```text
如果 precision 低但 risk 低：
  FPO 是 veto，不是 selector；增加 value/path direction sketch。

如果 value 正但 longrisk 高：
  加 RiskyHighAUV veto，不调 value threshold。

如果 SlowBurnGood 全漏：
  增加 delayed-gain features，不看 immediate loss。

如果 cost 高：
  减少 virtual samples；cache JVP/VJP；只保留 top cheap features。

如果 false positives 都是 RiskyHighAUV：
  强化 risk-adjusted path gate。

如果 false negatives 都是 OldOnly/Core77：
  回到 A 线对这些动作做 operator contrast。
```

---

## 14. P6：existing-action controller gate

### 目标

只有 C 线 density sufficient 或 B 线 FPO strong pass 时才打开。

### 必须记录

```text
controller_id
source_condition
accepted_count
precision
V_LCB
V240_LCB
RAUV_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO_drop
LSO_drop
LTO_drop
LFO_drop
feature_cost_q90_ms
payload_apply_cost_q90_ms
dataset_name_used
outcome_field_used
future_label_used
```

### Pass 标准

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
dataset_name_used = 0
outcome_field_used = 0
future_label_used = 0
```

### 不满足时 Codex 先尝试

```text
如果 accepted_count < 87：
  回到 C 线 density；不要放宽 risk。

如果 precision 高但 LDO 高：
  查 group/template/family support，不按 dataset 调参。

如果 risk 高：
  加 hard veto，不调 precision threshold。

如果 feature cost 高：
  降级为 diagnostic，不 official。
```

---

## 15. P7：generated sandbox reopen gate

### 目标

generated route 只能在有明确目标时打开。

### 允许打开的条件

```text
Condition D1:
  C 线证明 natural density insufficient。

Condition D2:
  B 线 FPO strong pass，可以作为生成目标。

Condition D3:
  A 线 path type 机制清楚，且可用 commit-time proxy 近似。
```

### Sandbox 规模

```text
64 generated actions only
branch-horizon required
no fake/proxy/cpu
action_apply_error_linf_max = 0
```

### Pass 标准

```text
generated_PathGood_precision >= 0.25
generated_V_LCB > 0
generated_longrisk_UCB <= 0.10
new_positive_created_rate >= 0.10
longrisk_created_rate <= 0.10
```

### 不满足时 Codex 先尝试

```text
如果 generated value-negative：
  不调 blend ratio；先查 path type 是否 BadPath。

如果 generated longrisk high：
  加 RiskyHighAUV veto；检查 memory/offdiag。

如果 generated low-risk low-value：
  判为 SafeLowValue，不直接失败；检查 h240+ delayed gain。

如果 generated collision high：
  修 payload/action uniqueness，不继续 outcome。
```

---

## 16. P8：runtime 与 paired replay boundary

### 目标

只有 P6 controller pass 后才打开。

### Runtime pass

```text
step_ratio_q90 <= 1.50
peak_memory_ratio <= 1.05
feature_cost_q90_ms within budget
payload_apply_error_linf_max = 0
no CPU offload
```

### Paired replay pass

```text
RealFunctional beats AdamWParallel
RealFunctional beats bestLR
RealFunctional beats NoOp
RealFunctional beats Random
shuffled payload fail
paired replay LDO/LSO/LTO/LFO pass
```

### 不满足时 Codex 先尝试

```text
如果 runtime fail due feature cost：
  compress FPO features；cache sketch；lower sample count。

如果 payload apply cost high：
  use sparse apply / fused apply；do not CPU offload。

如果 paired replay fail but controller diagnostics pass：
  check mismatch between diagnostic horizon and replay protocol。

如果 shuffled payload passes：
  controller is not action-specific；stop official route。
```

---

## 17. v9.9.6 最终判断规则

v9.9.6 必须输出下面四种 route 之一。

### Route A：Tail Fidelity Closed, Density Pending

```text
major+tail generator official pass;
1024 branch-horizon pass;
5000 panel running or complete;
density still inconclusive.
```

### Route B：Natural Density Sufficient

```text
major+tail official pass;
CoreLike/PathGood/SlowBurnGood LCB >= 0.03;
进入 controller/FPO path。
```

### Route C：Natural Density Insufficient

```text
major+tail official pass;
10000+ panel UCB < 0.03;
existing natural AP0 harvesting insufficient;
D line generated route eligible。
```

### Route D：Tail Fidelity Still Fails

```text
no major+tail generator pass;
missing tail unresolved or major/tail conflict unresolved;
P3+ gate-blocked;
下一轮继续 generator/root-cause repair。
```

### Route E：FPO Mechanism Emerges

```text
FPO strong pass;
controller gate eligible even before density full closure;
需要 runtime validation。
```

---

## 18. 一句话总结

v9.9.5 不是没进展，它把 tail key 修到可采样；但 generator 仍不能保住 tail，所以不能裁决自然好动作密度。v9.9.6 的核心不是继续调分数，而是把 missing tail 的根因查清楚，做 major+tail 约束采样，然后才允许真正的 density panel、controller、generated sandbox 和 runtime。

