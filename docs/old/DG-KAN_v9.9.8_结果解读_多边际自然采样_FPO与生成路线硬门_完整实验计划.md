# DG-KAN v9.9.7 结果解读与 v9.9.8 完整实验计划

> 本文基于 v9.9.7 `Tail Fidelity Feasibility / Natural Density / FPO Rewrite` 的真实执行结果制定。本文不把任何 diagnostic、未通过 gate 的 density panel、FuturePathOperator、controller、generated sandbox 或 runtime 写成 official success。所有公式使用 `$...$` 或 `$$...$$`，Typora 友好。

---

# 0. 一句话判断

v9.9.7 有进展，但不是能力成功。它把问题从：

```text
当前 tail key 是否过细、是否在 1024/5000 panel 下不可采样？
```

推进到：

```text
现在已经找到一个可采样的 coarse tail key，但 generator 仍然无法同时保住 major distribution 和 tail distribution。
```

更准确地说：

$$
\boxed{
\text{tail key 可行性问题被推进了，但自然动作生成器仍不是 faithful sampler。}
}
$$

这不是 controller success，也不是 density closure。v9.9.7 的真实 route 是：

```text
route = CaseB-TailKeyV3FeasibleGeneratorFail
primary_blocker = tail_key_feasible_but_no_major_tail_generator_passed
secondary_blocker = density_panels_blocked
system_legal_controller_pass = 0
generated_route_status = stopped_generator_fidelity_failed
```

核心数据：

```text
P1 current tail key TK2:
  tail groups = 337
  support<3 fraction = 0.12166172106824925
  expected missing tail @1024 = 72.73671413360148
  expected missing tail @5000 = 5.388789904783137
  official feasible @1024/@5000 = 0/0

P2 tail key v3:
  candidate/pass count = 8/1
  selected = TK3H-major_template_step_coarse
  selected pass = 1

P3 generator repair:
  G30-G39 candidate count = 10
  official/weak pass count = 0/0
  best failed = G33-major-first-tail-reservoir-refill-generator
  G33 major PSI = 0.5578023370995995
  G33 tail PSI = 0.3067280643885967
  G33 missing tail = 0

P4 density:
  largest completed panel = 0
  sufficient/insufficient/inconclusive = 0/0/1

P6 FPO v9:
  weak/strong = 0/0
  best = FPO9C-signal-reservoir-drift-diffusion-v2
  precision = 0.011494252873563218
  V LCB = -0.383912815103318
```

这说明：

```text
1. TK2 确实在有限 panel 下不可采样；
2. TK3H 把 tail key 从不可采样变成可采样；
3. 但 G30-G39 没有一个能同时保 major + tail；
4. G33 已经做到 missing tail = 0，但 major/tail PSI 很差；
5. 因此不能打开 density panel；
6. FPO v9 仍没有训练当下可用的机制；
7. controller / generated / runtime / paired replay 必须继续 gate-blocked。
```

---

# 1. 四线进展判断

## 1.1 A 线：未来训练路径

A 线这轮没有真正推进。P5 future path weak/strong 都是 `0/0`，不是因为 future path 被证伪，而是因为 P3 generator fidelity 没过，P4 density panel 没打开，所以没有可信的新 natural panel 可以重验 future path。

前几轮已经形成的判断仍然成立：Core77、OldOnly、Core+RiskClean 这些动作的未来路径信号是可信的。尤其 SlowBurnGood 这类动作很重要：它们当前一步不一定好，但 h20/h80/h240 后变好。这说明好 functional update 更像是改变未来训练路径的小参数扰动，而不是当前一步最大 loss descent。

v9.9.7 对 A 线的影响是：

```text
future path 需要在保真 natural panel 上重新验证；
如果 natural generator 没有保住 tail，好动作类型可能根本采不到；
因此 A 线暂时受 C 线阻塞。
```

A 线下一步不应只看 AUV。要继续拆：

```text
FastGood：h1/h5/h20/h80/h240 都好；
SlowBurnGood：h1 不强甚至负，但 h20/h80/h240 好；
RiskyHighAUV：AUV 高，但 longrisk / memory / offdiag 高；
SafeLowValue：风险低，但收益弱；
BadPath：收益弱且风险高。
```

## 1.2 B 线：训练当下 FuturePathOperator

B 线继续失败，而且失败很硬。FPO9C 的 precision 只有 `0.01149`，V LCB 为 `-0.3839`。这不是阈值差一点，而是当前 FPO 仍然没有测到 future path benefit。

现在可以明确停止以下方向：

```text
调 FPO9C threshold；
继续调 LC/FOS/FPO 的单分数；
只看 signal-reservoir SNR；
只看当前 response magnitude；
只用 memory/hard-tail risk gate 当 value selector。
```

B 线真正要做的是 path-type prediction：训练当下判断一个动作更像 FastGood、SlowBurnGood、RiskyHighAUV、SafeLowValue 还是 BadPath。尤其要识别 SlowBurnGood，因为它是 immediate loss descent 看不见的关键类型。

## 1.3 C 线：自然 AP0 动作扩流

C 线是本轮主线，也是最大 blocker。

v9.9.5 修好了 tail key 过碎的问题，TK2 groups = 337，support<3 fraction 降到 0.1217；v9.9.6 进一步证明 TK2 在 finite panel 下仍不可行，expected missing tail @5000 还有 5.39；v9.9.7 找到了 TK3H，groups = 89，support<3 fraction = 0.0899，self-sample tail PSI/JS = 0.0078/0.0240，并且 Core77/OldOnly/SlowBurn coverage 分别是 1.0/0.9/0.9815。也就是说，tail key 可行性这一步是真的推进了。

但 P3 说明 generator 还没过。G33 做到了 missing tail = 0，却把 major/tail 分布都弄坏了。它说明：

$$
\boxed{
\text{missing tail = 0 不是 fidelity pass。}
}
$$

真正要满足的是：

$$
D_{major}(P_G,P_{ref}) \le \epsilon_{major}
$$

$$
D_{tail}(P_G,P_{ref}) \le \epsilon_{tail}
$$

$$
missing\_tail = 0
$$

$$
max\_share \le \tau
$$

同时保持 action/payload collision 为 0、action apply error 为 0。

当前 C 线从“tail key 不可采样”推进到了“tail key 可采样但 generator 不会多边际匹配”。这是进展，但不是 closure。

## 1.4 D 线：generated update

D 线继续停止是正确的。

现在 A 线只有旧 panel 的 future path 现象，B 线没有合法 FPO，C 线没有保真 density。此时打开 generated sandbox，只会回到过去多轮 APG/APGA/APGF/APGT 的失败模式：payload 合法、branch-horizon 能跑，但 value-negative / high-longrisk。

D 线只能在以下条件之一成立后打开：

```text
1. C 线证明 natural density 不足，需要生成新动作；
2. B 线找到合法低成本 FuturePathOperator，可以作为 generated objective；
3. A 线在保真 natural panel 上证明某种 path type 机制足够明确，可以直接作为生成目标。
```

在此之前，generated sandbox 继续 `not_run` 是正确 gate。

---

# 2. 现在进展到底如何

从系统能力看，进展很慢。因为到现在还没有：

```text
selected controller；
selected runtime；
official paired replay；
short/full training。
```

但从科学边界看，v9.9.7 是有价值的。它完成了一个关键分解：

```text
v9.9.5：tail key 可采样，但 generator 不过；
v9.9.6：发现 TK2 在 1024/5000 下仍不可行；
v9.9.7：找到可行 TK3H，但 generator 仍不能保 major+tail。
```

这说明问题正在从“tail key 设计”转向“多边际自然采样”。这是一个比阈值调参更本质的问题。

---

# 3. 当前真正卡在哪里

## 3.1 卡点一：自然动作扩流还是没有可信 density panel

P4 largest completed panel = 0，所以 CoreLike / PathGood / SlowBurnGood 的自然密度仍然没有被裁决。

当前不能说：

```text
自然 AP0 没有足够好动作。
```

也不能说：

```text
自然 AP0 一定有足够好动作。
```

因为我们还没有一个 major+tail faithful generator 进入 1024/5000/10000/20000 sequential density panel。

## 3.2 卡点二：tail key v3 可行，但可能过粗

TK3H 过线是进展，但它是 `major_template_step_coarse`。它把 tail group 降到 89 个，解决了 finite panel missing 问题，但可能丢失了 recipe / memory / offdiag / norm 这些细粒度结构。

所以 v9.9.8 不能只问：

```text
TK3H 是否可采样？
```

还要问：

```text
TK3H 是否仍能区分好动作所在尾部？
TK3H 的每个 tail group 内部是否过度混合？
Core77 / OldOnly / SlowBurnGood 是否被压缩到少数 coarse tail group？
```

如果 TK3H 太粗，density panel 即使跑通，也可能把真正的 good-action tail 混进普通尾部，导致密度估计和 FPO 学习都不稳定。

## 3.3 卡点三：generator 不会同时保 major 和 tail

G33 missing tail = 0，但 major PSI = 0.5578，tail PSI = 0.3067。这个形态很典型：为了补 tail，它过度重采某些 reservoir 或 tail groups，破坏了整体自然分布。

这说明下一步不能继续简单加：

```text
major-first；
tail-first；
quota；
oversampling；
reservoir refill；
rejection sampling。
```

需要把问题写成多边际约束采样：

$$
\min_q D(q,p)
$$

subject to：

$$
A_{major} q = m_{major}
$$

$$
A_{tail} q = m_{tail}
$$

$$
q_i \ge 0
$$

并加入 entropy regularization 或 min-cost flow。

## 3.4 卡点四：FPO 仍没有接近未来路径收益

FPO9C precision 接近随机，V LCB 负。它不是未来路径算子，只是又一个 response / SNR / drift-diffusion sketch 失败。

B 线下一步必须基于 path type 做 false positive / false negative 分析：

```text
FPO9C 选中的到底是 RiskyHighAUV、BadPath 还是 SafeLowValue？
FPO9C 漏掉的 SlowBurnGood 有什么共同训练当下特征？
```

否则 B 线会继续盲目造 sketch。

---

# 4. 本轮发现的本质问题

## 4.1 好动作密度是 tail-conditional，不是 global density

我们想估计的是：

$$
P_{natural}(Good)
$$

但如果好动作集中在尾部 group $z$，则：

$$
P_{natural}(Good)=\sum_z P(Good\mid z)P_{natural}(z)
$$

若 generator 的 $P_G(z)$ 不等于 $P_{natural}(z)$，即使 major distribution 很像，估计出来的 $P_G(Good)$ 也不可信。

所以 v9.9.7 的本质是：

$$
\boxed{
\text{我们还没有拿到能代表 } P_{natural}(z_{tail}) \text{ 的样本。}
}
$$

## 4.2 missing tail 被解决，不等于 distribution 被解决

G33 missing tail = 0，但 PSI 很差。这说明：

```text
覆盖所有 tail groups 是必要条件；
但不是充分条件。
```

good natural generator 必须同时做到：

```text
tail 覆盖；
tail 比例；
major 比例；
max share；
entropy；
action uniqueness；
payload realism。
```

## 4.3 现在不是继续补动作，而是要先证明采样过程可信

在 sampling process 不可信时，任何 density result 都没有解释力。

如果用 G33 这种 generator 跑出低 CoreLike density，不能说明 natural AP0 没有好动作；如果跑出高 CoreLike density，也不能说明 natural AP0 有足够好动作。它只说明 G33 的 artificial distribution 里有多少好动作。

---

# 5. v9.9.8 总体目标

v9.9.8 的一句话目标：

$$
\boxed{
\text{把 natural AP0 扩流从“tail key 可行”推进到“major+tail faithful sampling”，并在通过后裁决自然好动作密度。}
}
$$

v9.9.8 不做这些事：

```text
不调 FPO9C threshold；
不调 AUV/RAUV threshold；
不直接跑 5000/10000/20000；
不把 G33 missing tail = 0 写成 generator pass；
不在 density 没过时打开 controller；
不在 B/C 没过时打开 generated sandbox；
不按 dataset 调任何 selector 或 threshold。
```

v9.9.8 要做这些事：

```text
1. 审计 TK3H 是否过粗；
2. 用多边际采样修 major+tail conflict；
3. 先做 distribution-only preflight，不急着跑 branch-horizon；
4. 只有 major+tail pass 后才跑 sequential density；
5. 在保真 panel 上重验 future path type；
6. 用 path-type false positive / false negative 重写 FPO；
7. 只有 C 或 B 过线后才开 controller / generated / runtime。
```

---

# 6. v9.9.8 实验阶段

## P0：复现 v9.9.7 边界

### 目标

确认 v9.9.7 route、tail key、generator repair、FPO、no-fake audit 都能被重新读取，避免在错误 artifact 上继续。

### 假设

$$
H_0: \text{v9.9.7 boundary is reproducible.}
$$

### 必须记录

```text
source_route_v9970
P1_current_tail_key_official_feasible
P2_tail_key_v3_pass
selected_tail_key_id
P3_official_pass_count
P3_weak_pass_count
best_failed_generator_id
P4_largest_completed_panel_size
P6_FPO_best
fake_data_used
proxy_row_used
cpu_offload_used
```

### 通过标准

```text
source_route_v9970 = CaseB-TailKeyV3FeasibleGeneratorFail
selected_tail_key_id = TK3H-major_template_step_coarse
P3_official_pass_count = 0
fake/proxy/cpu = 0/0/0
```

### 不满足时 Codex 先尝试

```text
如果 source route 不一致：检查 v9.9.7 out-dir、manifest hash、route json；
如果 selected tail key 不一致：检查 tail key candidate CSV 是否被覆盖；
如果 fake/proxy/cpu 不为 0：停止后续实验，先修 contract audit。
```

---

## P1：TK3H tail key 粗细度审计

### 目标

判断 TK3H 虽然可采样，但是否过度粗化，导致 good-action tail 被普通动作淹没。

### 假设

$$
H_1: \text{TK3H is feasible but may be too coarse for good-action density estimation.}
$$

### 记录指标

对 TK3H 的每个 tail group 记录：

```text
tail_group_id
canonical_count
generator_count_available
Core77_count
OldOnly_count
SlowBurnGood_count
FastGood_count
RiskyHighAUV_count
BadPath_count
mean_V240
mean_RAUV
longrisk_rate
memory_fail_rate
offdiag_fail_rate
within_group_entropy_recipe
within_group_entropy_memory
good_density_per_tail
```

还要计算：

```text
Core77 tail concentration
OldOnly tail concentration
SlowBurnGood tail concentration
GoodAction Gini over tail groups
within-tail outcome variance
tail purity for path type
```

### 判断标准

TK3H 可继续作为 density tail key，当且仅当：

```text
Core77 coverage >= 0.95
OldOnly coverage >= 0.80
SlowBurnGood coverage >= 0.90
missing high-good-density tail = 0
within-tail path-type entropy 不过高
RiskyHighAUV 与 SlowBurnGood 不被系统性合并进同一少数 tail group
```

如果 TK3H 把 SlowBurnGood 和 RiskyHighAUV 大量混在一起，则它只能作为 sampling tail key，不能作为 FPO/path-type learning key。

### 可视化

```text
fig_p1_tail_good_density_heatmap.svg
fig_p1_tail_path_type_purity.svg
fig_p1_tail_good_gini.svg
fig_p1_within_tail_variance.svg
```

### 不满足时 Codex 先尝试

```text
如果 TK3H 过粗：
  尝试 TK3H_split_by_memory_offdiag；
  尝试 TK3H_split_by_recipe_family；
  只 split high-mixed tail groups，不回到 1000+ group 的过细 key；
  目标是 groups <= 160，support<3 fraction <= 0.12。

如果 TK3H coverage 不足：
  检查 Core77/OldOnly/SlowBurnGood reference join；
  检查 path type 是否来自旧 panel，不应和 new generator output 直接混用。
```

---

## P2：多边际分布保真采样器 G40-G48

### 目标

修复 G33 的问题：missing tail = 0 但 major/tail PSI 很差。v9.9.8 不再做简单 quota，而要做 major+tail 多边际约束。

### 假设

$$
H_2: \text{A multi-marginal sampler can satisfy both major and tail fidelity.}
$$

### 采样器候选

```text
G40-IPF-major-tail-raking-sampler
G41-mincostflow-major-tail-sampler
G42-entropy-regularized-multimarginal-sampler
G43-major-tail-residual-balancing-sampler
G44-tail-conditional-major-refill-sampler
G45-major-conditional-tail-refill-sampler
G46-mixture-IPF-with-tail-replay-sampler
G47-canonical-precursor-tail-local-mutation-sampler
G48-real-train-stream-harvest-sampler
```

这些 generator 必须只使用 commit-time / generation-time 合法字段，不使用 outcome label、future path label、dataset-specific rule。

### 记录指标

```text
generator_id
action_count
major_psi
major_js
major_max_share
major_entropy_ratio_min
tail_psi
tail_js
missing_tail_group_count
tail_coverage
tail_max_share
tail_entropy_ratio_min
collision_action_count
collision_payload_count
action_apply_linf_max
payload_norm_distribution_psi
recipe_distribution_psi
rows_per_second_distribution_only
```

### 通过标准

Official fidelity pass：

```text
major_psi <= 0.05
major_js <= 0.08
major_max_share <= 0.35
major_entropy_ratio_min >= 0.90

tail_psi <= 0.05
tail_js <= 0.08
missing_tail_group_count = 0
tail_coverage >= 0.98
tail_entropy_ratio_min >= 0.90

action_collision = 0
payload_collision = 0
action_apply_linf_max <= 1e-7
```

Weak fidelity pass：

```text
major_psi <= 0.08
tail_psi <= 0.10
missing_tail_group_count <= 1
tail_coverage >= 0.95
```

### 可视化

```text
fig_p2_major_tail_psi_pareto.svg
fig_p2_missing_tail_by_generator.svg
fig_p2_major_vs_tail_heatmap.svg
fig_p2_sampler_entropy_ratio.svg
fig_p2_generator_failure_modes.svg
```

### 不满足时 Codex 先尝试

```text
如果 missing_tail_group_count > 0：
  对每个 missing group 输出 expected quota、rounded quota、precursor count、filter removal、dedup removal、cursor state；
  若 quota rounding 到 0，用 stochastic rounding 或 minimum quota；
  若 precursor 不存在，回到 tail key P1 重审该 group 是否可生成；
  若 dedup 删除，允许 payload-local mutation 但必须 action apply check。

如果 missing tail = 0 但 tail PSI 高：
  不要继续追 missing；改用 IPF/raking 调整比例；
  加 tail over/under sampling residual correction；
  输出每个 tail group 的 signed residual。

如果 tail pass 但 major fail：
  改 hierarchical sampler：先 major quota，再 major 内 tail quota；
  或使用 min-cost flow 同时匹配 major/tail。

如果 major pass 但 tail fail：
  加 tail conditional refill，不要调 major quota；
  对 high-residual tail 做 targeted refill。

如果 action_apply_linf 不为 0：
  停止 P3，先修 payload materialization。
```

---

## P3：1024 branch-horizon fidelity pilot

### 目标

只有 P2 至少 weak pass 后，才对 best generator 跑真实 branch-horizon 1024 pilot。防止 distribution-only pass 但 payload/outcome materializer 出问题。

### 假设

$$
H_3: \text{A fidelity-pass generator remains valid after real branch-horizon materialization.}
$$

### 记录指标

```text
generator_id
action_count = 1024
expected_branch_rows
actual_branch_rows
completion
rows_sec
wallclock
peak_gpu_mb
nan_count
inf_count
duplicate_row_count
action_apply_linf_max
major/tail fidelity after materialization
CoreLike_count/LCB/UCB
PathGood_count/LCB/UCB
SlowBurnGood_count/LCB/UCB
RiskyHighAUV_count
BadPath_count
```

### 通过标准

```text
completion = 1
nan/inf = 0/0
duplicate_row_count = 0
action_apply_linf_max <= 1e-7
major/tail fidelity still weak or official pass
```

### 不满足时 Codex 先尝试

```text
如果 materialization 后 fidelity 漂移：
  检查 distribution-only rows 与 materialized rows 的 action_id/payload_hash join；
  检查 branch-horizon filter 是否删除了某些 tail group；
  检查 GPU OOM / chunking 是否导致 skipped actions。

如果 throughput 太低：
  增大 chunk-actions 前先确认显存；
  启用 action-level resume；
  不允许 CPU offload。
```

---

## P4：sequential natural density panel

### 目标

在保真 generator 上裁决自然 AP0 action stream 是否有足够好动作密度。

### 假设

$$
H_4: \text{Faithful natural AP0 extension has sufficient density of good future-path actions.}
$$

### sequential panel

```text
panel sizes = 1024, 5000, 10000, 20000
```

每个 panel 都要记录：

```text
action_count
CoreLike_count/rate/LCB/UCB
PathGood_count/rate/LCB/UCB
SlowBurnGood_count/rate/LCB/UCB
CoreLike+SlowBurnGood_count/rate/LCB/UCB
RiskyHighAUV_count/rate
BadPath_count/rate
major/tail fidelity post-panel
per-dataset rates
per-template rates
per-family rates
runtime rows/sec
peak_gpu_mb
```

### 判断标准

Density sufficient：

$$
LCB(CoreLike \cup SlowBurnGood) \ge 0.03
$$

并且：

```text
longrisk UCB <= 0.05
memory/offdiag UCB <= 0.05
major/tail fidelity remains pass
```

Density insufficient：

$$
UCB(CoreLike \cup SlowBurnGood) < 0.03
$$

并且 panel size 至少 10000。

Inconclusive：

$$
LCB < 0.03 \le UCB
$$

则继续下一 panel，直到 20000。

### 可视化

```text
fig_p4_density_curve_core_path_slow.svg
fig_p4_density_ci_by_panel.svg
fig_p4_path_type_distribution_by_panel.svg
fig_p4_density_by_dataset_template_family.svg
fig_p4_fidelity_drift_across_panel.svg
```

### 不满足时 Codex 先尝试

```text
如果 density sufficient：
  打开 P6 existing-action controller gate。

如果 density insufficient：
  不要立刻判死项目；
  先检查 tail fidelity 是否在 20000 panel 后仍 pass；
  若 fidelity pass 且 UCB < 0.03，再判定 natural AP0 source density insufficient，打开 D 线 generated objective design。

如果 inconclusive at 20000：
  增加 panel 到 50000 前先估算成本；
  或改用 stratified confidence interval / importance weighted density，但不能伪造 labels。
```

---

## P5：未来路径类型重验

### 目标

在保真 natural panel 上重新验证 A 线，不再只依赖旧 2876 AP0 universe。

### 假设

$$
H_5: \text{Future path types observed in canonical AP0 persist under faithful natural extension.}
$$

### 记录指标

对每类 path type 记录：

```text
FastGood_count
SlowBurnGood_count
RiskyHighAUV_count
SafeLowValue_count
BadPath_count
V1/V5/V20/V80/V240 LCB
RAUV LCB
longrisk UCB
bad/null UCB
memory/offdiag UCB
per-dataset count
per-template count
per-family count
```

### 通过标准

```text
SlowBurnGood_count >= 30 或 rate LCB > 0.005
SlowBurnGood V240 LCB > 0
SlowBurnGood longrisk UCB <= 0.05
RiskyHighAUV longrisk UCB >= 0.30
path type separation clear enough for FPO learning
```

### 不满足时 Codex 先尝试

```text
如果 SlowBurnGood 不复现：
  检查 natural generator 是否丢失旧 SlowBurnGood tail；
  检查 path type threshold 是否过严；
  分别看 Core77-derived vs new-natural actions。

如果 RiskyHighAUV 与 SlowBurnGood 混淆：
  加 memory/offdiag/hard-tail gate 到 path type definition；
  不调 AUV 主阈值。
```

---

## P6：FPO v10 path-type predictor

### 目标

把 B 线从“单分数 response”改成“path type prediction”。训练当下不直接预测 outcome label，而是预测动作是否可能进入 SlowBurnGood / FastGood，同时排除 RiskyHighAUV。

### 假设

$$
H_6: \text{Commit-time legal sketches can classify future path type better than previous FPO scores.}
$$

### FPO 候选

```text
FPO10A-slowburn-vs-risky-binary-sketch
FPO10B-multiclass-path-type-sketch
FPO10C-two-head-value-path-risk-veto
FPO10D-tiny-virtual-adamw-1step-path-classifier
FPO10E-jvp-gradient-transport-path-classifier
FPO10F-memory-hardtail-path-veto-classifier
FPO10G-signal-reservoir-plus-path-type-conformal
```

### 特征合法性

允许：

```text
current train batch response
train-memory response
hard-tail response
per-example gradient mean/variance
JVP/VJP sketch
AdamW alignment
action norm / payload norm
memory/offdiag proxy
cheap virtual AdamW response on train-only buffer
```

禁止：

```text
future outcome
AUV label
V20/V80/V240 label
path type label at commit time
dataset-specific threshold
validation/test metric
```

### 记录指标

```text
FPO id
feature cost q90
precision TopK87
V LCB
V240 LCB
longrisk UCB
bad/null UCB
memory/offdiag UCB
SlowBurnGood recall
RiskyHighAUV false positive rate
BadPath false positive rate
LDO/LSO/LTO/LFO
```

### 通过标准

Weak FPO：

```text
precision >= 0.60
V LCB > 0
longrisk UCB <= 0.10
cost_q90 <= 2.0 ms
```

Strong FPO：

```text
precision >= 0.75
V LCB > 0
V240 LCB > 0
longrisk UCB <= 0.05
bad UCB <= 0.05
null UCB <= 0.15
memory/offdiag UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
cost_q90 <= 1.0 ms
```

### 不满足时 Codex 先尝试

```text
如果 precision 低但 longrisk 低：
  FPO 是 risk veto，不是 value selector；加入 value path head。

如果 value 正但 longrisk 高：
  加 RiskyHighAUV negative class，不调 value threshold。

如果 SlowBurnGood recall 低：
  对 SlowBurnGood 做 class-balanced training；
  加 delayed gain proxy；
  不能使用 future label at commit time。

如果 cost 高：
  减少 virtual samples；
  缓存 JVP/VJP；
  限制到 train-memory + hard-tail mini-buffer。
```

---

## P7：existing-action controller gate

### 开启条件

满足任一条件：

```text
P4 density sufficient；
P6 FPO strong pass；
P6 FPO weak pass 且 P4 density sufficient。
```

### 目标

构造 system-legal minimal controller，不按 dataset 调参。

### 记录指标

```text
accepted_count
coverage
precision
V LCB
V240 LCB
longrisk UCB
bad/null UCB
memory/offdiag UCB
LDO/LSO/LTO/LFO
field red count
feature cost
controller cost
```

### 通过标准

```text
accepted_count >= 87
precision >= 0.75
V LCB > 0
V240 LCB > 0
longrisk UCB <= 0.05
bad UCB <= 0.05
null UCB <= 0.15
memory/offdiag UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
red field count = 0
```

### 不满足时 Codex 先尝试

```text
如果 accepted_count < 87：
  不放松 risk gate；先用 density panel 找 expansion source。

如果 LDO 高：
  做 dataset-blind score normalization；不能按 dataset threshold。

如果 memory/offdiag 高：
  加 hard veto，不调 value score。

如果 value 低：
  回到 FPO false-positive 分析。
```

---

## P8：generated sandbox gate

### 开启条件

仅当满足以下之一：

```text
P4 density insufficient，且 generator fidelity pass；
P6 FPO strong pass；
P5 给出明确 SlowBurnGood 生成目标。
```

### 目标

只跑 64-action sandbox，验证是否能生成 future-path-good actions。不能直接 512/1024 大跑。

### 记录指标

```text
generated_action_count = 64
branch_rows completion
FastGood precision
SlowBurnGood precision
RiskyHighAUV rate
V LCB
V240 LCB
longrisk UCB
memory/offdiag UCB
new positive created rate
longrisk created rate
action apply linf
```

### 通过标准

```text
FastGood or SlowBurnGood precision >= 0.20
V LCB > 0
V240 LCB > 0
longrisk UCB <= 0.10
longrisk_created_rate <= 0.10
```

### 不满足时 Codex 先尝试

```text
如果 high AUV but high risk：
  classify as RiskyHighAUV，do not scale。

如果 low risk but low value：
  classify as SafeLowValue，check longer horizon before scaling。

如果 value-negative：
  stop generated family；do not tune blend ratio blindly。
```

---

## P9：runtime / paired replay boundary

### 开启条件

P7 controller pass。

### Runtime 标准

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
feature cost q90 within budget
no CPU offload
no fake/proxy
```

### Paired replay 标准

```text
RealFunctional beats AdamWParallel rate >= 0.60
RealFunctional beats bestLR rate >= 0.55
NoOp/shuffled payload fail as controls
LDO/LSO/LTO pass
```

---

# 7. v9.9.8 并行执行策略

## Batch 1：当天必须完成

```text
P0 boundary reproduction
P1 TK3H coarseness audit
P2 distribution-only G40-G48 matrix
FPO v10 false-positive/false-negative prep
```

## Batch 2：P2 pass 后立即启动

```text
P3 1024 branch-horizon fidelity pilot
P4 1024 density panel
P5 future path type revalidation on 1024
```

## Batch 3：sequential panel

```text
5000 panel
10000 panel
20000 panel if needed
```

## Batch 4：conditional downstream

```text
P6 FPO v10
P7 controller
P8 generated sandbox
P9 runtime / paired replay
```

---

# 8. 本轮必须落盘 artifacts

```text
p0_boundary_v9980.csv
p1_tail_key_coarseness_audit_v9980.csv
p1_tail_path_type_purity_v9980.csv
p2_generator_distribution_only_matrix_v9980.csv
p2_generator_failure_taxonomy_v9980.csv
p3_branch_horizon_1024_pilot_v9980.csv
p4_sequential_density_panel_v9980.csv
p5_future_path_type_revalidation_v9980.csv
p6_fpo_v10_path_type_predictor_v9980.csv
p7_existing_action_controller_gate_v9980.csv
p8_generated_sandbox_gate_v9980.csv
p9_runtime_paired_replay_boundary_v9980.csv
no_fake_proxy_cpu_audit_v9980.csv
run_manifest_v9980.json
route_decision_v9980.json
```

---

# 9. 本轮必须生成的可视化

```text
fig_p1_tail_good_density_heatmap.svg
fig_p1_tail_path_type_purity.svg
fig_p2_major_tail_psi_pareto.svg
fig_p2_missing_tail_by_generator.svg
fig_p2_major_vs_tail_heatmap.svg
fig_p4_density_curve_core_path_slow.svg
fig_p4_density_ci_by_panel.svg
fig_p5_future_path_type_distribution.svg
fig_p6_fpo_false_positive_negative_matrix.svg
fig_p7_controller_gate_matrix.svg
fig_stop_pivot_matrix_v9980.svg
```

---

# 10. 最终 stop / pivot 规则

## 情况 A：P2 仍无 fidelity pass

如果 G40-G48 没有任何 weak pass：

```text
停止继续加 generator 名字；
输出 multi-marginal infeasibility report；
改走 real train-stream harvesting，而不是 synthetic natural extension generator。
```

## 情况 B：P2 pass，但 P4 density insufficient

如果 fidelity pass 且 full density UCB < 0.03：

```text
判定 natural AP0 source density insufficient；
D 线可以打开 64-action generated sandbox；
generated objective 必须以 SlowBurnGood / FastGood path type 为目标。
```

## 情况 C：P2 pass，P4 density sufficient

如果 density LCB >= 0.03：

```text
existing-action harvesting route 继续；
打开 P7 minimal controller；
不需要立即重开 generated route。
```

## 情况 D：P6 FPO strong pass

```text
打开 controller，即使 density 仍未完全 sufficient，也可以做 FPO-based accepted region scout；
但 paired replay 仍需 runtime pass。
```

## 情况 E：P6 FPO 继续失败

```text
不要继续调阈值；
分析 false positives / false negatives 的 path type；
如果 FPO 连续三轮 precision < 0.20 且 V LCB < 0，停止 FPO 线，转向 direct future-path generator。
```

---

# 11. 最终结论

v9.9.8 的关键不是“再试一个 generator”。它要判断：

$$
\boxed{
\text{是否存在一个同时保住 major distribution 和 good-action tail distribution 的 natural AP0 extension sampler。}
}
$$

如果存在，才能裁决自然好动作密度；如果不存在，就不能继续把 natural density 当成已知量。当前最重要的科学判断是：

$$
\boxed{
\text{好动作不是 global density 问题，而是 tail-conditional density 和 future-path type 问题。}
}
$$

