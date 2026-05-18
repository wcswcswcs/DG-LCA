# DG-KAN v9.9.6 结果解读与 v9.9.7 下一步计划

> 本文基于 v9.9.6 `Tail Fidelity Root Cause / Natural Density / FPO Rewrite` 的真实结果整理。目标不是小修某个 generator 或某个 FPO 阈值，而是回答一个更底层的问题：自然 AP0 扩流到底能否产生一个足够保真的动作分布，并据此判断好动作密度、future path 机制和 controller 是否有继续价值。
>
> 公式均使用 `$...$` 或 `$$...$$`，Typora 友好。

---

## 0. 一句话结论

v9.9.6 的真实结论是：

$$
\boxed{
\text{missing-tail 的归因已经完成，但 G20-G29 仍没有产生 major+tail fidelity 合格的 natural generator。}
}
$$

这不是能力成功。controller、selected runtime、paired replay、short/full training 都没有打开。可是它也不是空转，因为它把问题从“tail key 可能错了 / generator 可能没调好”推进到更明确的采样瓶颈：

```text
P1 root-cause attribution complete = 1
assigned_missing_reason_fraction = 1.0
unknown_missing_reason_fraction = 0.0

MR1 quota rounded = 30
MR5 cursor collapse = 12
MR6 major-tail conflict = 55
MR8 tail key too fine = 162

G20-G29 official / weak pass = 0 / 0
best failed = G26-canonical-tail-precursor-replay-generator
G26 major PSI = 0.029000497474792362
G26 tail PSI = 0.24015656834967286
G26 missing tail = 24

density panels completed = 0
FPO v8 weak / strong = 0 / 0
best FPO = FPO8G-signal-channel-drift-diffusion-sketch
FPO8G precision = 0.011494252873563218
FPO8G V LCB = -0.383912815103318
controller = not_run
generated sandbox allowed = 0
```

所以当前状态不是：

```text
自然 AP0 没有好动作。
```

而是：

```text
当前 generator 还没有可靠采样到 natural AP0 的关键 tail，
因此不能裁决自然动作密度，也不能打开 controller 或 generated sandbox。
```

---

## 1. 我对 v9.9.6 的独立判断

### 1.1 有进展，但不是能力进展

v9.9.6 最大进展是 P1：missing-tail root-cause attribution 完成。之前我们只知道 G13-G19 或 G20-G29 没过 fidelity，但不知道 tail 到底是被 quota 归零、cursor 塌缩、major-tail 冲突、filter 删除，还是 tail key 本身过细。现在 v9.9.6 明确给出归因：`MR8-tail-key-too-fine` 是最大项，`MR6-major-tail-conflict` 是第二项。

这说明后续不能再只做：

```text
G30: 多加一点 quota；
G31: 多 oversample 一点 tail；
G32: 多 replay 一点 precursor。
```

因为如果 tail key 本身在当前 pilot size 下不可采样，或者 major 与 tail 约束不可同时满足，继续堆 generator 名字没有意义。

### 1.2 v9.9.6 的失败不是“generator 调得不够”，而是“采样约束可能不可行”

现在核心已经是一个约束采样问题。我们想让 generator 分布 $P_G$ 逼近 canonical natural AP0 reference 分布 $P_{ref}$，同时满足：

$$
D_{major}(P_G,P_{ref}) \le \epsilon_M
$$

$$
D_{tail}(P_G,P_{ref}) \le \epsilon_T
$$

$$
missing\_tail\_group = 0
$$

$$
collision=0, \quad action\_apply\_error=0.
$$

但如果某些 tail group 在 reference 中极稀疏，或者 tail key 被切得太细，那么在 $n=1024$ 的 pilot 中要求 `missing_tail_group = 0` 可能本身就很难，甚至和 major 保真冲突。

对一个 tail group $g$，如果真实概率是 $p_g$，在 $n$ 个样本中没采到的概率约为：

$$
P(\text{miss }g)=(1-p_g)^n.
$$

如果很多 $p_g$ 都很小，那么 `missing_tail_group = 0` 会变成非常强的覆盖约束。v9.9.6 的 MR8 占 162，说明我们可能在用一个过细 tail key 给 generator 设定了几乎不可达的目标。

### 1.3 G26 的结果说明 major fidelity 和 tail fidelity 是分裂的

G26 的 major PSI 已经是 `0.0290`，这个数并不糟糕；但 tail PSI 是 `0.2402`，missing tail 仍有 24 个。也就是说，G26 在粗分布上接近，但没保住尾部。

这和 v9.9.2-v9.9.5 的趋势一致：

```text
粗分布可修；
尾部保真困难；
好动作很可能藏在尾部；
因此不能用粗分布过线来裁决好动作密度。
```

这也是本阶段最重要的数学 insight：

$$
\boxed{
\text{GoodAction density 是 tail-conditional density 问题，不是 global major distribution 问题。}
}
$$

如果：

$$
P(Good)=\sum_z P(Good\mid z)P(z),
$$

而 $P(Good\mid z)$ 在少数 tail group 上很高，那么只保 major distribution 不能保证 $P(Good)$ 估计正确。

---

## 2. 四线进展评估

## 2.1 A 线：未来训练路径

v9.9.6 这轮没有推进 A 线，因为 P2 fidelity 没过，P4 future path 被 gate-block。不能因此说 A 线失败。

已有结果已经支持：Core77、OldOnly、Core+RiskClean、SlowBurnGood 等路径类型不是 immediate loss descent 能解释的。v9.9.6 对 A 线的意义是：A 线必须迁移到 **tail-fidelity 通过后的 natural panel** 上重新验证。

当前 A 线状态：

```text
future path phenomenon: credible diagnostic
future path official mechanism: not closed
future path on faithful natural panel: not tested
```

下一步 A 线不能只看 AUV 或 RAUV，要继续按路径类型拆：

```text
FastGood:
  h1 就好，h20/h80/h240 也好。

SlowBurnGood:
  h1 不强甚至为负，但 h20/h80/h240 变好，longrisk 低。

RiskyHighAUV:
  AUV 高，但 longrisk / memory / offdiag 高。

SafeLowValue:
  risk 很低，但 value 不足。

BadPath:
  value 差且 risk 高。
```

真正重要的是 SlowBurnGood，因为它最符合我们现在的核心思路：

$$
\boxed{
\text{好 functional update 不是当前一步降 loss 最多，而是把后续训练带到更好路径。}
}
$$

## 2.2 B 线：训练当下 FuturePathOperator

v9.9.6 的 FPO v8 失败很硬：best `FPO8G-signal-channel-drift-diffusion-sketch` precision 只有 `0.01149`，V LCB 为 `-0.3839`。这说明当前 FPO v8 不只是阈值差，而是方向没对。

现在 B 线的问题不是成本，而是目标错配。FPO8G 名字上接近 signal-channel / drift-diffusion，但实测没有抓住好 future path。它可能仍在估计局部响应、梯度一致性或短期统计，而不是估计动作对未来训练路径的传播效应。

B 线下一步必须从“score feature”改成“路径类型预测器”：

```text
不是预测 action 是否 immediate positive；
不是预测 response magnitude；
不是预测 risk clean；
而是预测 action 属于 FastGood / SlowBurnGood / RiskyHighAUV / SafeLowValue / BadPath 哪一类。
```

### B 线当前结论

```text
FPO v8 weak pass = 0
FPO v8 strong pass = 0
best FPO precision ≈ 0.0115
best FPO value negative
```

因此不能继续小修 FPO8G。

## 2.3 C 线：自然 AP0 动作扩流

C 线仍然是最高优先级，也是 v9.9.6 的主 blocker。

当前 C 线进展链条是：

```text
v9.8.9: natural extension generator entrypoint missing
v9.9.0: entrypoint/materializer landed, single/16/256 smoke pass
v9.9.1: 1024 pilot distribution mismatch
v9.9.2: G5 major distribution pass
v9.9.3: G5 tail fidelity fail
v9.9.4: G7-G12 all fail major+tail fidelity
v9.9.5: tail key TK2 pass, but G13-G19 fail
v9.9.6: root-cause attribution complete, but G20-G29 still fail
```

这不是无进展。它把 C 线从“没有入口”推进到了“知道为什么 major+tail fidelity 同时过不了”。但这条线仍没闭合，因为：

```text
largest completed density panel = 0
CoreLike / PathGood density cannot be estimated
5000 / 10000 / 20000 density panel cannot open
```

C 线的核心问题已经不是“跑不跑 5000”，而是：

$$
\boxed{
\text{我们是否能构造一个在 pilot size 下可行、又不丢失 good-action tail 的 natural sampler？}
}
$$

## 2.4 D 线：generated update

D 线继续停止是正确的。

现在没有任何条件支持重开 generated route：

```text
A 线只是事后 future path 诊断；
B 线没有训练当下合法 FPO；
C 线没有保真 density 裁决。
```

在这种情况下重开 APG / APGU / APGX，只会重复过去的问题：payload 合法、branch-horizon 能跑，但动作 value-negative / high-longrisk。

D 线只允许在以下任一条件满足后重开 64-action sandbox：

```text
1. C 线证明 natural AP0 好动作密度不足；
2. B 线找到可用 FPO，可以作为生成目标；
3. A 线给出明确 path type 机制，可以转成生成目标。
```

---

## 3. 为什么感觉还是非常慢

这个感觉是对的。v9.9.6 没有打开：

```text
controller
selected runtime
official paired replay
short/full training
```

从系统能力角度看，进度仍然很慢。

但从科学推进看，v9.9.6 很重要，因为它证明：

```text
不是 tail key 没审计；
不是 root cause 没归因；
不是只缺一个 G20-G29 profile；
而是当前 major+tail fidelity 目标在采样上仍不可闭合。
```

这阻止我们错误地继续做两件事：

```text
1. 用 tail-failed generator 的低 density 证明 natural AP0 源不足；
2. 在没有密度裁决和 FPO 的情况下重开 generated route。
```

---

## 4. 当前真正卡在哪里

## 4.1 卡在 tail fidelity 的可行性定义

P1 显示 MR8-tail-key-too-fine 是最大 root cause。这个结果说明问题不只是 generator repair，而是 tail fidelity gate 本身可能需要重定义为“可采样的 tail fidelity”。

不能直接把 tail fidelity 放松到无意义，但也不能要求一个在有限 pilot 下几乎不可达的 missing-tail 全覆盖。

需要区分：

```text
不可丢的 actionable tail：
  与 Core77 / OldOnly / SlowBurnGood / RiskClean precursor 有关的 tail。

普通稀有 tail：
  稀有但与好动作机制无关，不能让它无限阻塞 density panel。
```

但注意：official generator 不能使用 outcome label 作为采样规则。因此 actionable tail 的定义必须用 commit-time / precursor / recipe / structural axes，而不是 GoodAction label 本身。

## 4.2 卡在 major-tail 多边际约束冲突

MR6 major-tail conflict 有 55 个，说明 major quota 与 tail quota 不是简单嵌套关系。下一步要从硬 quota 转向多边际约束优化，比如 iterative proportional fitting、min-cost flow、entropy-regularized sampling。

## 4.3 卡在 FPO 仍没有测到 future path benefit

FPO8G 的失败说明 signal-channel / drift-diffusion 这个名字还没有真正落地成 future path operator。

真正要预测的是：

$$
\Delta\theta \mapsto (V_1,V_5,V_{20},V_{80},V_{240},LongRisk,Memory,Offdiag).
$$

而不是：

$$
\Delta\theta \mapsto \text{当前一步响应大小}.
$$

## 4.4 卡在 density 裁决不能打开

只要没有 official 或 weak major+tail generator，P3 largest completed panel 只能是 0。现在所有关于 natural AP0 好动作密度的判断仍然是悬而未决。

---

## 5. 是否还在正确道路上

高层方向仍然对，因为 v9.9.6 做对了三件事：

```text
1. 没有把 tail-key pass 或 root-cause pass 写成 generator pass；
2. 没有把 G26 major PSI 好写成 density pass；
3. 没有在 P2 不过时打开 density/controller/generated/runtime。
```

但具体执行路线必须更硬。下一轮不要继续：

```text
调 FPO8G threshold；
调 AUV / RAUV threshold；
继续只加 G30/G31/G32 名字；
直接跑 5000 / 10000 / 20000；
把 tail-failed generator 的 density 当结论；
没有 B/C 证据就重开 generated sandbox。
```

正确方向是：

```text
先做 tail fidelity feasibility audit；
再做 tail key v3；
再做多边际采样 generator；
再跑 sequential density；
再重验 future path；
最后才谈 FPO/controller/generated。
```

---

## 6. 离目标还差多远

离 system-legal local functional controller 至少还差四道门：

```text
1. natural generator 同时通过可行的 major fidelity 和 actionable-tail fidelity；
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

当前最准确状态是：

$$
\boxed{
\text{future path 好动作现象可信；root cause 已归因；但 tail sampling 仍不可闭合，density/controller/generated 都不能 official。}
}
$$

---

# v9.9.7 下一步完整实验计划

## 7. v9.9.7 总体目标

v9.9.7 的目标不是继续小修某个 generator，而是回答三个第一性问题：

```text
Q1. 当前 tail fidelity gate 在 pilot size 下是否可行？
Q2. 如果不可行，如何构造不使用 outcome label、但能保住 good-action precursor tail 的 tail key？
Q3. 如果 tail fidelity 可行，natural AP0 action stream 中 CoreLike / PathGood / SlowBurnGood 的真实密度是否足够？
```

总目标写成：

$$
\boxed{
\text{建立可采样、可审计、可裁决自然好动作密度的 major+tail generator。}
}
$$

v9.9.7 的最低有效推进是：

```text
1. 给出 tail fidelity gate 的 feasibility report；
2. 如果 tail key 过细，落地 tail key v3；
3. 至少一个 generator 通过 major+actionable-tail weak gate；
4. 如果通过，跑 1024 -> 5000 sequential density；
5. 如果不通过，给出明确 root cause 和下一步 Codex action。
```

---

## 8. P0：复现 v9.9.6 boundary

### 目标

确认 v9.9.6 的失败边界可复现，并禁止把 P1 root-cause pass 写成 generator pass。

### 必须记录

```text
source_route_v9960
P1_attribution_complete
P1_assigned_missing_reason_fraction
P1_unknown_missing_reason_fraction
P1_root_cause_counts_by_reason
P2_G20_G29_official_pass_count
P2_G20_G29_weak_pass_count
P2_best_failed_generator
P2_best_failed_major_psi
P2_best_failed_tail_psi
P2_best_failed_missing_tail
P3_largest_completed_panel
P5_best_fpo
P5_best_fpo_precision
P5_best_fpo_V_LCB
no_fake / no_proxy / no_cpu
```

### 通过标准

```text
P0_boundary_pass = 1
fake/proxy/cpu = 0/0/0
```

### 如果不满足，Codex 先尝试

```text
1. 检查 v9.9.6 artifact path 是否正确；
2. 检查 route_decision_v9960.json 是否存在；
3. 检查 P1/P2 CSV row count 是否被 partial rerun 覆盖；
4. 禁止继续 P1-P9，直到 boundary 复现。
```

---

## 9. P1：Tail fidelity feasibility audit

### 目标

判断当前 major+tail fidelity gate 是否在 pilot size 下可采样。不能再默认 `missing_tail_group=0` 是合理约束。

### 核心假设

$$
H_1:
\text{v9.9.6 的 tail failure 主要来自 tail key / quota feasibility，而不是普通 generator bug。}
$$

### 必须记录

对每个 tail group $g$：

```text
tail_group_id
major_group_id
canonical_count
canonical_probability
expected_count_at_1024
expected_count_at_5000
expected_count_at_10000
expected_count_at_20000
prob_miss_at_1024 = (1-p_g)^1024
prob_miss_at_5000 = (1-p_g)^5000
quota_floor
quota_ceil
rounded_quota
precursor_available_count
recipe_available_count
filter_removal_count
dedup_removal_count
collision_count
cursor_visit_count
missing_reason_v9960
```

全局记录：

```text
tail_group_count
support_lt_3_fraction
support_lt_5_fraction
support_lt_10_fraction
expected_missing_tail_at_1024
expected_missing_tail_at_5000
feasible_tail_group_count_at_1024
feasible_tail_group_count_at_5000
tail_coverage_possible_at_1024
tail_coverage_possible_at_5000
```

### 判断标准

当前 tail gate 可行，当且仅当：

```text
expected_missing_tail_at_1024 <= 5
feasible_tail_group_count_at_1024 / tail_group_count >= 0.95
support_lt_3_fraction <= 0.10
```

如果不满足，但 5000 可行：

```text
expected_missing_tail_at_5000 <= 5
feasible_tail_group_count_at_5000 / tail_group_count >= 0.95
```

则允许 1024 pilot 做 diagnostic，但 official fidelity gate 必须按 5000 判断。

如果 5000 仍不可行，则当前 tail key 不可用于 official fidelity gate。

### 可视化

```text
tail_group_support_histogram.svg
expected_missing_vs_panel_size.svg
prob_miss_by_tail_group_rank.svg
tail_group_feasibility_heatmap.svg
missing_reason_pareto_v9960.svg
```

### 如果不满足，Codex 先尝试

```text
如果 support_lt_3_fraction 太高：
  合并 tail key 的最低层，不动 major key；
  优先合并同 recipe_family / same precursor class / same action-shape 的 tail。

如果 expected_missing_at_1024 高但 5000 可行：
  将 fidelity official gate 从 1024 调整为 5000；
  1024 只做 smoke，不做 terminal route。

如果 precursor_available_count = 0：
  说明这个 tail group 在 generator recipe 中不可达；
  回到 P2 recipe coverage repair。

如果 quota rounded = 0：
  改 stochastic rounding 或 minimum expected quota，不要直接硬设 1 导致 major 崩。

如果 filter_removal_count 高：
  列出 filter 名称；
  优先放宽非 safety-critical filter；
  safety-critical filter 不允许绕过。
```

---

## 10. P2：Tail key v3 设计与审计

### 目标

构造一个不使用 outcome label、但能覆盖 good-action precursor tail 的新 tail key。

### tail key 约束

Tail key v3 允许使用：

```text
major group id
candidate template / recipe family
precursor action family
step bucket
action norm bucket
payload norm bucket
memory/offdiag-safe proxy bucket
future-path-independent structural bucket
```

禁止使用：

```text
CoreLike label
PathGood label
SlowBurnGood label
future outcome
V_integrated
AUV/RAUV
longrisk label
GradeAB label
dataset-specific threshold
```

### 候选 tail keys

```text
TK3A-major_recipe_precursor
TK3B-major_recipe_precursor_norm
TK3C-major_recipe_memory_offdiag_proxy
TK3D-major_template_step_norm
TK3E-major_recipe_action_shape
TK3F-major_recipe_precursor_merged_low_support
```

### 必须记录

```text
tail_key_id
tail_group_count
support_lt_3_fraction
support_lt_5_fraction
support_lt_10_fraction
major_js
major_psi
tail_js
tail_psi_under_reference_self_sample
coverage_of_Core77_precursors_diagnostic
coverage_of_OldOnly_precursors_diagnostic
coverage_of_SlowBurnGood_precursors_diagnostic
outcome_label_used = 0/1
```

注意：Core77 / OldOnly / SlowBurnGood precursor coverage 只能用于诊断 tail key 是否保住已知好动作区域，不能作为 official sampling rule。

### 判断标准

Tail key v3 通过：

```text
outcome_label_used = 0
support_lt_3_fraction <= 0.10
tail_group_count <= 512
self_sample_tail_psi <= 0.05
self_sample_tail_js <= 0.08
Core77_precursor_coverage_diagnostic >= 0.90
OldOnly_precursor_coverage_diagnostic >= 0.90
```

### 可视化

```text
tail_key_candidate_support_bars.svg
tail_key_core77_oldonly_coverage_matrix.svg
tail_key_major_tail_tradeoff.svg
```

### 如果不满足，Codex 先尝试

```text
如果 tail_group_count 太大：
  合并最低支持组；
  保留 recipe family 和 precursor family，不保留过细 continuous bucket。

如果 Core77/OldOnly coverage 低：
  检查这些动作的 structural axes；
  增加不含 outcome 的 precursor class 或 action-shape axis。

如果 self_sample_tail_psi 高：
  tail key 仍过碎或 bins 不稳定；
  改 quantile bins 或 hierarchical bins。
```

---

## 11. P3：G30-G39 major+tail generator repair matrix

### 目标

用多边际约束采样解决 major fidelity 和 tail fidelity 冲突。

### 候选 generator

```text
G30-IPF-hierarchical-major-tail-generator
  使用 iterative proportional fitting 对 major/tail 两个边际同时拟合。

G31-mincostflow-major-tail-quota-generator
  将 tail quota、major quota、recipe availability 写成 min-cost flow。

G32-tail-first-major-corrected-generator
  先覆盖 tail，再用 major correction fill，避免 missing tail。

G33-major-first-tail-reservoir-refill-generator
  先保 major，再用 tail reservoir 补缺失 tail。

G34-stochastic-rounded-tail-quota-generator
  针对 quota rounded=0 的情况做 stochastic rounding。

G35-precursor-bootstrap-tail-generator
  从 canonical precursor 结构 bootstrap，但生成新 action hash。

G36-entropy-regularized-sampler
  约束 max group share，同时最小化 major/tail divergence。

G37-tail-coverage-negative-control
  故意只保 major 不保 tail，必须不过。

G38-tail-oversample-importance-weighted-diagnostic
  允许 importance weight，只做 diagnostic，不 official。

G39-two-stage-feasible-tail-key-v3-generator
  使用 P2 选出的 tail key v3，做 1024/5000 sequential fidelity。
```

### 必须记录

```text
generator_id
action_count
major_psi
major_js
tail_psi
tail_js
missing_tail_group_count
max_group_share
entropy_ratio_min
collision_count
payload_hash_missing
action_apply_linf_max
rows_expected
rows_actual
rows_per_sec
peak_gpu_mb
official_density_eligible
weak_fidelity_pass
official_fidelity_pass
```

### 判断标准

weak fidelity pass：

```text
major_psi <= 0.08
major_js <= 0.10
tail_psi <= 0.08
tail_js <= 0.10
missing_tail_group_count <= 5
max_group_share <= 0.40
collision_count = 0
action_apply_linf_max = 0
```

official fidelity pass：

```text
major_psi <= 0.05
major_js <= 0.08
tail_psi <= 0.05
tail_js <= 0.08
missing_tail_group_count = 0
max_group_share <= 0.35
entropy_ratio_min >= 0.90
collision_count = 0
action_apply_linf_max = 0
```

### 可视化

```text
generator_major_tail_pareto.svg
missing_tail_by_generator.svg
major_tail_conflict_heatmap.svg
generator_throughput_vs_fidelity.svg
```

### 如果不满足，Codex 先尝试

```text
如果 missing_tail_group_count > 0：
  输出每个 missing tail 的 root cause；
  若 root cause 是 quota rounding，转 G34；
  若 root cause 是 precursor unavailable，转 G35；
  若 root cause 是 filter deletion，输出 filter 名并修 filter；
  若 root cause 是 major-tail conflict，转 G30/G31。

如果 major pass 但 tail fail：
  不再调 major quota；
  增加 tail constraints 或 tail key coarsening。

如果 tail pass 但 major fail：
  用 IPF 或 min-cost flow 做 major correction；
  不要简单降低 tail quota。

如果 max_group_share 高：
  加 entropy regularization 或 no-replacement sampling。

如果 rows/sec 低或 GPU 高：
  先跑 distribution-only；
  fidelity pass 后再 branch-horizon。
```

---

## 12. P4：Sequential natural density panel

### 目标

在 official 或 weak fidelity generator 通过后，真实裁决自然 AP0 动作流中的好动作密度。

### Panel 顺序

```text
1024 smoke density
5000 official-minimum density
10000 confirmation density
20000 final density
```

每个 panel 不能复用 outcome rows，必须真实 materialize branch-horizon。

### 目标标签

```text
CoreLike:
  clean, value-positive, low-risk, memory/offdiag safe 的核心好动作。

PathGood:
  future path good，包含 FastGood / SlowBurnGood。

SlowBurnGood:
  h1 不强，但 h20/h80/h240 正向且 longrisk 低。

RiskyHighAUV:
  AUV 高但风险高，用作 reject。

SafeLowValue:
  risk 低但 value 不足，用作边界。
```

### 必须记录

```text
panel_size
generator_id
fidelity_pass_source
CoreLike_count / rate / Wilson_LCB / Wilson_UCB
PathGood_count / rate / Wilson_LCB / Wilson_UCB
SlowBurnGood_count / rate / Wilson_LCB / Wilson_UCB
RiskyHighAUV_count
SafeLowValue_count
BadPath_count
per_major_group_density
per_tail_group_density
per_dataset_density_diagnostic
per_template_density_diagnostic
rows_expected / rows_actual
rows_per_sec
peak_gpu_mb
no_fake/no_proxy/no_cpu
```

### 判断标准

如果：

$$
LCB(CoreLike) \ge 0.03
$$

或：

$$
LCB(PathGood) \ge 0.03
$$

则 natural harvesting route 可继续。

如果：

$$
UCB(CoreLike) < 0.03
$$

且：

$$
UCB(PathGood) < 0.03
$$

则 natural AP0 source density 不足，需要转向 generated update。

如果：

$$
LCB < 0.03 \le UCB,
$$

则继续下一 panel，不下结论。

### 可视化

```text
density_ci_by_panel.svg
density_by_tail_group_heatmap.svg
path_type_composition_by_panel.svg
corelike_vs_pathgood_density_curve.svg
```

### 如果不满足，Codex 先尝试

```text
如果 1024 density 低但 UCB 仍跨 0.03：
  跑 5000，不要提前判死。

如果 5000 UCB < 0.03：
  停止 natural harvesting；
  进入 generated route reopen decision。

如果 panel rows incomplete：
  先实现 resume/chunking；
  禁止用 partial panel 做 density结论。

如果 density 在某些 tail group 高：
  记录 tail-conditional density；
  但不要按 dataset 调参。
```

---

## 13. P5：Future path revalidation on faithful natural panel

### 目标

验证 A 线 future path 类型是否在保真 natural panel 上仍成立。

### 必须记录

```text
action_id
path_type
V1/V5/V20/V80/V240
AUV
RAUV
longrisk
bad/null
memory/offdiag
hard_tail_delta
old_family_delta
per_tail_group
per_template
per_dataset_diagnostic
```

### 判断标准

A 线保真通过：

```text
FastGood_count + SlowBurnGood_count >= 87
RAUV_LCB > 0
LongRisk_UCB <= 0.05
memory/offdiag_UCB <= 0.05
LDO/LSO/LTO <= 0.10
```

### 可视化

```text
future_path_curves_by_type.svg
slowburn_good_horizon_profile.svg
risky_high_auv_vs_slowburn_scatter.svg
path_type_by_tail_group.svg
```

### 如果不满足，Codex 先尝试

```text
如果 FastGood 少但 SlowBurnGood 多：
  B 线 FPO 以 SlowBurnGood 为主目标。

如果 AUV 高但 longrisk 高：
  分离 RiskyHighAUV，不能作为 positive。

如果 PathGood 在某些 tail group 集中：
  检查 tail key 是否捕捉该结构；
  不把 tail group 本身作为 controller 条件。
```

---

## 14. P6：FuturePathOperator v9 重写

### 目标

训练当下低成本预测 path type，尤其是 SlowBurnGood。

### FPO v9 候选

```text
FPO9A-tiny-virtual-adamw-sketch
  action apply 后虚拟 1-3 个 AdamW micro step，只在小 train-memory/hard-tail 子集上估计 delayed value。

FPO9B-jvp-gradient-alignment-sketch
  用 JVP/VJP 估计 action 对下一步梯度方向的影响。

FPO9C-signal-reservoir-drift-diffusion-v2
  估计 stable drift 是否大于 diffusion，不再只看 one-step response。

FPO9D-memory-offdiag-hard-gate-plus-value-sketch
  memory/offdiag 只做 hard gate，value 由 delayed sketch 判断。

FPO9E-slowburn-detector
  专门识别 h1 不强但 h20/h80/h240 变好的动作。

FPO9F-negative-control-immediate-response-only
  只看 immediate response，必须不过。
```

### 必须记录

```text
fpo_id
commit_time_feature_only = 1/0
feature_cost_ms_q50/q90
TopK87_precision_CoreLike
TopK87_precision_PathGood
TopK87_precision_SlowBurnGood
V_LCB
RAUV_LCB
LongRisk_UCB
Bad_UCB
Null_UCB
Memory_UCB
Offdiag_UCB
LDO/LSO/LTO/LFO
false_positive_path_type_distribution
false_negative_path_type_distribution
```

### 判断标准

FPO weak pass：

```text
TopK87 PathGood precision >= 0.60
RAUV_LCB > 0
LongRisk_UCB <= 0.10
Memory/Offdiag_UCB <= 0.10
cost_q90 <= 5 ms
```

FPO strong pass：

```text
TopK87 PathGood precision >= 0.75
SlowBurnGood recall >= 0.50
V_LCB > 0
RAUV_LCB > 0
LongRisk_UCB <= 0.05
Bad_UCB <= 0.05
Null_UCB <= 0.15
Memory/Offdiag_UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
cost_q90 <= 1.50 ms
```

### 可视化

```text
fpo_precision_vs_cost.svg
fpo_false_positive_path_type_bar.svg
fpo_slowburn_recall_curve.svg
fpo_value_risk_scatter.svg
```

### 如果不满足，Codex 先尝试

```text
如果 precision 低但 risk 低：
  当前 FPO 只是 veto，不是 value selector；
  加 delayed value sketch，不要继续调 veto 阈值。

如果 value 正但 longrisk 高：
  加 memory/offdiag hard gate；
  不要降低 value threshold。

如果 cost 高：
  减少 virtual samples；
  用 cached JVP/VJP；
  只保留 stratified memory/hard-tail mini-buffer。

如果 SlowBurnGood recall 低：
  明确训练 slowburn detector；
  不再用 immediate-response-only proxy。
```

---

## 15. P7：Existing-action controller gate

### 打开条件

只有满足以下任一条件才打开：

```text
1. P4 natural density sufficient；
2. P6 FPO strong pass；
3. P5 future path strong pass 且 density count >= 87。
```

### 必须记录

```text
controller_id
accepted_count
precision
V_LCB
RAUV_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO/LSO/LTO/LFO
feature_cost_ms_q90
payload_apply_ms_q90
step_ratio_estimate_q90
```

### 通过标准

```text
accepted_count >= 87
precision >= 0.75
V_LCB > 0
RAUV_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory/offdiag_UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
feature_cost_ms_q90 <= 1.50
```

### 如果不满足，Codex 先尝试

```text
如果 accepted_count < 87：
  回到 P4 density；
  不要放宽 risk gate。

如果 precision 够但 LDO 高：
  分解 support/backfill/quality；
  不按 dataset 调参。

如果 risk 高：
  查 false positives path type；
  过滤 RiskyHighAUV。

如果 cost 高：
  简化 FPO；
  禁止将高成本 probe 写成 official controller。
```

---

## 16. P8：Generated sandbox reopen gate

### 打开条件

只有满足以下之一才允许：

```text
1. natural density insufficient：CoreLike/PathGood UCB < 0.03；
2. FPO strong pass：可作为 generation objective；
3. future path mechanism strong pass：明确 path type generation target。
```

### 生成 sandbox 限制

```text
max generated actions = 64
must include negative controls
must use real payload
must materialize branch-horizon rows
must include no-fake/no-proxy/no-cpu audit
```

### 通过标准

```text
PathGood precision >= 0.50
V_LCB > 0
RAUV_LCB > 0
longrisk_UCB <= 0.10
memory/offdiag_UCB <= 0.10
new_positive_created_rate >= 0.10
```

### 如果不满足，Codex 先尝试

```text
如果 generated value-negative：
  不调 scale；
  查是否路径类型为 SafeLowValue 或 BadPath。

如果 generated high-longrisk：
  查是否 RiskyHighAUV；
  加 memory/offdiag hard gate。

如果 generated safe but low-value：
  检查是否需要 longer horizon；
  不直接写成 success。
```

---

## 17. P9：Selected runtime / paired replay boundary

### 打开条件

只有 P7 controller pass 或 P8 generated sandbox pass 才打开。

### Runtime 指标

```text
step_ratio_q50/q90
feature_compute_ms_q90
payload_apply_ms_q90
kernel_count_q90
sync_count_q90
memory_ratio
selected_action_count_per_step
```

### Paired replay 指标

```text
RealFunctional vs AdamWParallel
RealFunctional vs bestLR
RealFunctional vs NoOp
RealFunctional vs Random
RealFunctional vs shuffled payload
h20/h80/h240 V
longrisk/bad/null
memory/offdiag
```

### 通过标准

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
RealFunctional beats all controls
shuffled payload fails
```

---

## 18. v9.9.7 可视化总表

```text
fig_p1_tail_feasibility_support_histogram.svg
fig_p1_expected_missing_vs_panel_size.svg
fig_p1_missing_reason_pareto.svg
fig_p2_tail_key_candidate_matrix.svg
fig_p3_generator_major_tail_pareto.svg
fig_p3_missing_tail_by_generator.svg
fig_p4_density_ci_by_panel.svg
fig_p4_density_by_tail_group_heatmap.svg
fig_p5_future_path_curves_by_type.svg
fig_p5_slowburn_profile.svg
fig_p6_fpo_precision_vs_cost.svg
fig_p6_false_positive_path_type_bar.svg
fig_p7_controller_gate_waterfall.svg
fig_p8_generated_sandbox_value_risk.svg
```

---

## 19. v9.9.7 stop / pivot rules

### Case A：tail fidelity feasibility 不可行

```text
停止继续 Gxx 名字堆叠；
重构 tail key v3；
只在 tail key v3 通过后继续 generator repair。
```

### Case B：tail key v3 可行，但 generator 仍 fail

```text
转向 min-cost flow / IPF / entropy regularized sampler；
禁止直接跑 5000 density。
```

### Case C：major+tail fidelity pass，但 density insufficient

```text
如果 CoreLike/PathGood UCB < 0.03：
  natural AP0 harvesting route 判不足；
  打开 generated sandbox 64-action。
```

### Case D：density sufficient，但 FPO fail

```text
existing-action harvesting 可继续；
controller 可以先用 natural label-derived candidate pool做离线 selector diagnostic；
但 online controller 仍不能 official，直到 FPO 或 legal mechanism pass。
```

### Case E：FPO pass

```text
打开 P7 controller；
通过后再测 runtime；
runtime 过后再 paired replay。
```

### Case F：generated sandbox pass

```text
进入 generated route v10.0；
但仍需 selected runtime 和 paired replay 才算 system success。
```

---

## 20. 最后判断

v9.9.6 后，项目不应该再说“快成功了”。更准确的状态是：

$$
\boxed{
\text{我们已经看到 future path 好动作现象，也完成 missing-tail root-cause 归因；下一道硬门是构造可采样的 natural-tail generator，并裁决真实好动作密度。}
}
$$

如果 v9.9.7 能解决 tail feasibility 和 generator fidelity，项目会第一次真正进入 natural density science。否则，必须承认 existing natural AP0 harvesting 主线暂时被采样不可行性阻塞，转向 future-path-driven generated update。
