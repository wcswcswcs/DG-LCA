# DG-KAN v9.9.2 结果解读与 v9.9.3 下一步完整实验计划

> 本文基于 v9.9.2 `Natural Generator Distribution Repair / FuturePathOperator` 的真实结果整理。目标不是给 v9.9.2 找一个好听的 route，而是判断：现在到底有没有进展，四条线各自推进到哪里，为什么仍然慢，真正卡住的东西是什么，以及 v9.9.3 应该怎样设计成能加速、能 fail-fast、能让 Codex 在条件不满足时自动尝试修复的计划。

---

## 0. 一句话判断

v9.9.2 有进展，而且是重要进展，但不是能力成功。

它把上一轮的 blocker：

```text
natural generator 分布失真
```

推进成：

```text
有一个候选 generator 已经通过 major distribution gate，
但好动作密度仍然没有被 full panel 裁决。
```

当前最准确的状态是：

$$
\boxed{
\text{自然动作生成器的粗分布已修好；但它是否保住了稀有好动作区域，还没有被证明。}
}
$$

这句话比 route `R2-DistributionFidelityPassDensityPending` 更准确。因为 v9.9.2 不只是 “density pending”，还暴露了一个更危险的问题：**通过 major distribution gate 的 G5，在 1024 pilot 上 CoreLike / PathGood 密度很低。** 这可能意味着：major axis 看起来对了，但稀有的好动作 recipe 仍然没被采到。

---

## 1. v9.9.2 关键数据复盘

v9.9.2 的 route 是：

```text
route = R2-DistributionFidelityPassDensityPending
primary_blocker = natural_density_full_panel_pending
secondary_blocker = future_operator_sketch_not_opened
system_legal_controller_pass = 0
generated_route_status = stopped_density_full_panel_pending
```

这说明它没有进入 controller、runtime、paired replay 或 short/full training。

### 1.1 P0 / P1：旧 generator 分布错误被复现并定位

v9.9.1 的 boundary 被复现：

```text
source route = R-P1-NaturalGeneratorDistributionMismatch
PSI max = 13.018683555518681
missing major groups = 17
```

v9.9.2 进一步定位原 generator 的错误：

```text
P1 weak/strong = 0 / 0
failure class = F1-cursor-collapse
P1_PSI_major_max = 25.56018339816044
P1_missing_major_group_count = 41
```

独立判断：这是必要进展。它说明上一轮低密度 pilot 不能直接拿来判定 AP0 自然动作源不足，因为 generator 自己已经分布塌缩。

### 1.2 P2：G5 通过 major distribution gate

P2 候选矩阵中，最佳候选是：

```text
G5-hybrid-quota-random-generator
```

它的核心指标是：

```text
action count = 1024
weak/strong = 1 / 1
PSI major max = 0.016435209021362127
JS major max = 0.045276968538283786
max group share = 0.2626953125
entropy ratio min = 0.9355692755194838
```

这说明 G5 在 major axes 上确实比原 generator 好很多。

但这里要非常小心：**major distribution pass 不等于 good-action distribution pass。** 如果好动作集中在更细的组合条件里，例如某些 template、step、memory/offdiag 状态、future-path type、action lineage，那么 major PSI / JS 很低也可能漏掉真正重要的稀有区域。

### 1.3 P3：1024 pilot 的好动作密度很低，但还不能 official 判死

P3 的 density boundary 是：

```text
selected generator = G5-hybrid-quota-random-generator
panel size = 1024
CoreLike count/LCB/UCB = 4 / 0.0015200565834699157 / 0.01000078476017681
PathGood count/LCB/UCB = 3 / 0.000996829459082154 / 0.008578186769099308
density sufficient/insufficient/inconclusive = 0 / 0 / 1
```

报告把 density 记为 inconclusive，是因为 full 5000 / 10000 / 20000 panels 没有打开：

```text
full_density_panels_deferred_set_run_full_density_panels_to_open
```

我的独立判断更尖锐：

$$
\boxed{
\text{P3 不能 official 判死，但它已经是一个强烈警告。}
}
$$

如果 G5 的 1024 pilot 真的是 representative，那么 CoreLike UCB 只有约 `0.0100`，PathGood UCB 只有约 `0.0086`，都明显低于常用 `0.03` coverage 下限。这不是中性结果。它提示两种可能：

```text
可能 A：G5 仍然没有采到稀有好动作 recipe。
可能 B：自然 AP0 action source 的好动作密度确实低于 controller 所需 density。
```

v9.9.2 还不能区分 A 和 B。这正是下一轮的核心。

### 1.4 P5：FuturePathOperator sketch 没有推进成机制

P5 的结果是：

```text
FuturePathOperator sketch weak/strong = 0 / 0
```

这说明 B 线还没有合法训练当下机制。不能因为 A 线 future path 现象强，就跳到 controller。未来路径好动作仍然是事后诊断，还不是可部署的训练当下规则。

### 1.5 P6 / P7：controller 与 generated sandbox 正确 gate-block

```text
P6 controller = not_run
P7 generated sandbox allowed = 0
No-fake rows checked = 3804
fake/proxy/cpu = 0 / 0 / 0
```

这点是对的。v9.9.2 没有把 distribution repair、1024 pilot 或 FuturePathOperator diagnostic 写成 official success。

---

## 2. 四线分别的进展

## 2.1 A 线：未来训练路径

### 当前进展

v9.9.2 没有实际推进 A 线。P3/P4 的重点是 natural generator / density / sketch gate，不是 future path 分解。A 线仍然继承 v9.8.x 的状态：

```text
Core77 / OldOnly / Core+RiskClean 类动作有强 future-path 现象；
但这个现象仍然是事后路径诊断，不是训练当下合法机制。
```

### 我的判断

A 线仍然是最有科学价值的线。到现在为止，最稳定的 insight 是：

$$
\boxed{
\text{好 functional update 不是当前一步降 loss 最多，而是把后续训练带到更好路径。}
}
$$

但 A 线下一步不能只看 AUV 或 V240。它必须拆成路径类型：

```text
FastGood：h1/h5 就好，后面继续好。
SlowBurnGood：h1 不明显好甚至为负，但 h20/h80/h240 变好。
RiskyHighAUV：AUV 高，但 longrisk/memory/offdiag 高。
SafeLowValue：风险低，但 value 不够。
BadPath：value、risk、memory/offdiag 都不好。
```

v9.9.3 中 A 线要在 G5 或修复后的 natural panel 上重新判断这些路径类型的密度和分布，而不是只复用 canonical 2876 的路径现象。

---

## 2.2 B 线：训练当下合法 FuturePathOperator

### 当前进展

v9.9.2 里 P5 weak/strong 仍为 `0 / 0`。这延续了前几轮结论：cheap proxy 很便宜但选错动作，high-cost proxy 太贵且不够好。当前 B 线没有形成可部署机制。

### 我的判断

B 线不是“阈值没调好”，而是目标没抓对。当前 proxy 多数在测：

```text
当前函数响应大小；
当前 hard-tail / memory 响应；
风险清理能力；
局部响应强弱。
```

但我们需要的是：

```text
这个动作是否会改变未来训练路径。
```

因此 v9.9.3 不能继续调 FOS / LC / FO threshold。B 线必须改成 **低成本未来路径算子草图**：用训练当下可算的极小成本，估计动作经过后续 AdamW 后是否仍有正收益，并且不制造 longrisk/memory/offdiag 问题。

---

## 2.3 C 线：自然 AP0 动作扩流

### 当前进展

这是 v9.9.2 的主进展。

v9.8.9：找不到 natural extension generator。

v9.9.0：generator entrypoint 落地，single/16/256 smoke 通过。

v9.9.1：1024 pilot 真实跑了，但 generator 分布失真。

v9.9.2：原 generator 分布错误被定位，G5 修复候选通过 major distribution weak/strong gate。

所以 C 线不是没进展。它已经从：

```text
没有入口
```

推进到：

```text
有一个 major-distribution-faithful candidate。
```

### 当前新问题

G5 虽然通过 major distribution gate，但 P3 的 CoreLike/PathGood count 很低。这里的核心问题是：

$$
\boxed{
\text{G5 是否真的保住了好动作所在的稀有条件分布？}
}
$$

major PSI / JS 只能保证粗分布像旧 AP0，不保证稀有 recipe 也被覆盖。下一步 C 线必须新增 tail fidelity 审计。

---

## 2.4 D 线：generated update

### 当前进展

D 线继续停止。

这是对的。因为：

```text
B 线没有训练当下合法机制；
C 线自然密度还没有 full panel 裁决；
A 线仍然是 future-path diagnostic；
```

此时重开 generated sandbox，只会回到历史上的老问题：

```text
payload 合法；
branch-horizon 能跑；
outcome value-negative / high-longrisk。
```

### 我的判断

D 线不应该主动推进。它只在两种情况下重开：

```text
1. C 线证明自然 AP0 源好动作密度不足；
   这说明必须重新生成动作，而不是继续 harvest。

2. B 线找到低成本 FuturePathOperator；
   这可以作为 generated update 的优化目标。
```

否则 D 线继续停止。

---

## 3. 为什么还是非常慢？

因为项目现在还没有进入系统闭环：

```text
controller -> selected runtime -> paired replay -> short/full training
```

但 v9.9.2 的慢和之前不一样。之前慢在“generator missing / distribution broken”；现在慢在“分布修复后，density 仍然没有裁决”。这已经更接近核心问题。

真正拖慢项目的不是实验数量少，而是每次都需要避免一个错误推进：

```text
v9.9.0 不能把 256 smoke 当 density closure；
v9.9.1 不能把失真 generator 的低密度 pilot 当自然源不足；
v9.9.2 不能把 major distribution pass 当好动作密度 pass。
```

这些 gate 都是对的。但下一轮必须加速：v9.9.3 不应该再停在 1024 pilot 或 major distribution。必须直接处理 tail fidelity 和 sequential full density。

---

## 4. 这轮最重要的 insight

## 4.1 分布保真有两层

v9.9.2 说明，分布保真不能只看 major axes。

第一层是 major fidelity：

```text
dataset / template / step_bucket / payload major group / action source 等粗轴是否像旧 AP0。
```

G5 通过了这一层。

第二层是 tail fidelity：

```text
好动作所在的稀有组合条件是否被采到。
```

例如：

```text
Core77-like precursor groups；
OldOnly-like precursor groups；
RiskCleanButLowValue precursor groups；
SlowBurnGood precursor groups；
memory/offdiag safe + value-positive strata；
future-path-safe strata；
rare template/action-lineage strata。
```

v9.9.2 还没有证明第二层。

因此下一步不能只说 “G5 distribution pass”。必须问：

$$
\boxed{
\text{G5 是否保住了好动作所在的稀有尾部？}
}
$$

## 4.2 如果 G5 的低 density 在 5000/10000 后仍成立，existing-action harvesting 就危险了

当前 1024 pilot 里：

```text
CoreLike count = 4
PathGood count = 3
```

这比 canonical PanelA 的 77 / 2876 rate 低得多。若 5000 / 10000 / 20000 后仍然是这个量级，那说明：

```text
自然 AP0 extension source 不是一个足够密集的好动作来源。
```

这会逼迫路线转成：

```text
FuturePathOperator-driven generated update。
```

也就是不再靠“自然动作池里捞好动作”，而是直接构造能改善未来路径的 update。

## 4.3 如果 tail fidelity 修复后 density 回升，existing-action route 仍可继续

如果发现 G5 缺失的是一些稀有 good-action recipe，并通过 tail-aware generator 修复后，CoreLike/PathGood density 回到或超过 0.03，那么 existing-action harvesting 仍然有价值。

此时下一步不是生成新 primitive，而是：

```text
做 natural-action harvest + FuturePathOperator controller。
```

---

## 5. 当前离目标还差多远

离 system-legal local functional controller 至少还差四道门：

```text
1. 自然生成器必须同时通过 major fidelity 和 tail fidelity；
2. CoreLike / PathGood / SlowBurnGood density 必须在 5000/10000/20000 panel 上被裁决；
3. 必须找到训练当下合法、低成本、跨分布稳定的选择规则，或明确转向生成式 update；
4. selected runtime step_ratio_q90 必须 <= 1.50。
```

离 strict PureKAN functional causal evidence 还要：

```text
official paired replay；
RealFunctional beats AdamWParallel / bestLR / NoOp / Random；
shuffled payload fail；
leave-dataset-out / leave-stratum-out / leave-template-out。
```

当前不是快成功了。最准确状态是：

$$
\boxed{
\text{自然扩流工程已推进到可修复分布；但好动作密度和未来路径机制还没闭合。}
}
$$

---

# v9.9.3 下一步完整实验计划

## 6. v9.9.3 总目标

v9.9.3 的目标不是再证明 G5 major distribution 好，也不是继续调 FuturePathOperator threshold。

本轮要回答一个硬问题：

$$
\boxed{
\text{在分布保真的自然 AP0 extension stream 中，好动作密度到底够不够？}
}
$$

同时并行回答另一个问题：

$$
\boxed{
\text{如果好动作密度不够，是 generator tail fidelity 不够，还是 AP0 自然动作源本身不足？}
}
$$

v9.9.3 的四线结构：

```text
A 线：未来路径类型继续拆，但不再只看 AUV。
B 线：重写 FuturePathOperator sketch，不再调 FOS / LC threshold。
C 线：最高优先级，做 tail fidelity + sequential natural density panel。
D 线：继续严格 gate，只在 A/B/C 给出证据后才允许 generated sandbox。
```

---

## 7. P0：复现 v9.9.2 边界

### 目标

确认 v9.9.2 的关键边界没有回退：

```text
G5 major distribution pass；
P3 density pending；
P5 FuturePathOperator fail；
controller/generated not_run；
no fake/proxy/cpu。
```

### 必须记录

```text
route_v9920
best_generator_id
best_PSI_major_max
best_JS_major_max
best_max_group_share
best_entropy_ratio_min
CoreLike_count_1024
PathGood_count_1024
fake/proxy/cpu counts
```

### 通过标准

```text
best_generator_id = G5-hybrid-quota-random-generator 或更好的 pass candidate
PSI_major_max <= 0.05
JS_major_max <= 0.10
max_group_share <= 0.35
entropy_ratio_min >= 0.85
fake/proxy/cpu = 0 / 0 / 0
```

### 不满足时 Codex 先尝试

```text
1. 若 G5 不再 pass：检查 seed、quota table、cursor provenance、major group reference 是否变了。
2. 若 old/new collision > 0：检查 action_id/payload_hash namespace 是否唯一。
3. 若 fake/proxy/cpu > 0：停止 science runner，只修 materializer contract。
4. 若 PSI/JS 突然变差：复现 G1-G5 matrix，找最小回退 commit。
```

---

## 8. P1：Major + Tail Fidelity Audit

### 目标

判断 G5 不只是 major distribution 像旧 AP0，还要判断它是否保住好动作所在的稀有尾部。

### 假设

$$
H_{tail}:
\text{G5 通过 major fidelity，但可能缺失 CoreLike / PathGood 所在的稀有条件组合。}
$$

### 记录指标

对 canonical AP0 PanelA 和 G5 natural panel 同时记录：

```text
major axes:
  dataset_id
  template_id
  step_bucket
  payload_norm_bucket
  action_norm_bucket
  candidate_origin
  family
  stratum

geometry / future-path precursor axes:
  memory_bucket
  offdiag_bucket
  old_family_bucket
  hard_tail_bucket
  support_bucket
  score_bucket
  value_proxy_bucket
  risk_proxy_bucket
  Core77_precursor_bucket
  OldOnly_precursor_bucket
  RiskClean_precursor_bucket
  SlowBurn_precursor_bucket

action lifecycle axes:
  source_recipe
  cursor_id
  quota_slot
  candidate_template_id
  action_lineage_id
  payload_hash_prefix
```

每个 axis / axis-pair 记录：

```text
old_count
new_count
old_share
new_share
PSI
JS
KL
missing_major_group_count
missing_tail_group_count
max_group_share
entropy_ratio
min_expected_group_count
```

### 通过标准

Major fidelity：

```text
PSI_major_max <= 0.05
JS_major_max <= 0.10
max_major_group_share <= 0.35
entropy_major_min >= 0.85
```

Tail fidelity：

```text
PSI_tail_max <= 0.15
JS_tail_max <= 0.15
missing_tail_group_count = 0 for groups with old_count >= 8
candidate_template_coverage >= 0.80
Core77_precursor_coverage >= 0.80
OldOnly_precursor_coverage >= 0.80
RiskClean_precursor_coverage >= 0.80
```

### 可视化

```text
major axis PSI bar chart
tail axis PSI bar chart
old vs new group share heatmap
candidate_template coverage curve
Core77/OldOnly/RiskClean precursor coverage upset plot
```

### 不满足时 Codex 先尝试

```text
Case 1: cursor collapse
  - 实现 cursor round-robin reset by source_recipe。
  - 每个 major group 设置 min quota。
  - 记录 cursor provenance，禁止连续消耗同一 cursor 超过 K 次。

Case 2: missing major groups
  - 自动生成 missing_major_groups.csv。
  - 对每个 missing group 加 quota fill。
  - 如果没有源数据，回溯 canonical AP0 provenance，找对应 source_recipe。

Case 3: tail group 缺失
  - 不用 outcome label 直接采 CoreLike；只能用 precursor group。
  - 加 tail-aware quota：memory/offdiag safe、old-family-safe、slowburn precursor、risk-clean precursor。
  - 若某 tail group 无法由当前 generator 产生，标记 generator_support_gap。

Case 4: max_group_share 过高
  - 加 group cap，例如每个 group share <= 0.30。
  - 若 cap 后 action_count 不足，增加 random fallback，但必须记录 fallback provenance。

Case 5: candidate_template coverage 低
  - 从 template_id / source_recipe 做 template-balanced sampling。
  - 禁止单 template expansion 伪装成多个 action。
```

---

## 9. P2：Sequential Natural Density Panel

### 目标

在通过 P1 major+tail fidelity 后，真实裁决自然动作池中好动作密度是否足够。

### 分阶段 panel

```text
Panel S0: 1024 actions, already available but rerun with tail audit.
Panel S1: 5000 actions.
Panel S2: 10000 actions.
Panel S3: 20000 actions.
```

每个 panel 只允许在前一 panel 的质量审计过线后打开。

### 标签定义

不要只看 CoreLike。至少记录以下路径类型：

```text
CoreLike:
  高 precision、高 V、低 longrisk、低 bad/null、memory/offdiag safe。

PathGood:
  future path value positive，risk adjusted AUV positive，longrisk 低。

SlowBurnGood:
  h1/h5 不强，h20/h80/h240 变好，longrisk 低。

RiskCleanButLowValue:
  risk clean，但 value 低或旧 label 认为不够强。

RiskyHighAUV:
  AUV 高，但 longrisk/memory/offdiag 高。

BadPath:
  value 低或 longrisk/bad/null/memory/offdiag 高。
```

### 每个 panel 记录

```text
action_count
branch_horizon_rows_expected / actual
completion_rate
rows/sec
peak_gpu_mb
CoreLike_count / rate / LCB / UCB
PathGood_count / rate / LCB / UCB
SlowBurnGood_count / rate / LCB / UCB
RiskCleanButLowValue_count / rate / LCB / UCB
RiskyHighAUV_count / rate / LCB / UCB
BadPath_count / rate / LCB / UCB
per-dataset density
per-template density
per-family density
per-step-bucket density
per-tail-group density
```

### Density pass / fail 标准

Natural density sufficient：

```text
CoreLike_LCB >= 0.03
or PathGood_LCB >= 0.03
or (CoreLike + SlowBurnGood)_LCB >= 0.03 with longrisk_UCB <= 0.05
```

Natural density insufficient：

```text
CoreLike_UCB < 0.03
and PathGood_UCB < 0.03
and (CoreLike + SlowBurnGood)_UCB < 0.03
at N >= 5000 with P1 fidelity pass
```

Inconclusive：

```text
LCB < 0.03 <= UCB
or P1 tail fidelity failed
or materializer incomplete
```

### 可视化

```text
density CI vs panel size plot
CoreLike / PathGood / SlowBurnGood stacked density plot
per-dataset density heatmap
per-template density heatmap
CoreLike count accumulation curve
rare-tail coverage vs good-action rate plot
```

### 不满足条件时 Codex 先尝试

```text
If materializer OOM:
  - action-sharded replay；
  - per-action cache clear；
  - resume manifest；
  - lower branch batch size；
  - do not mark missing rows as negative。

If throughput < 10 rows/sec:
  - write row shard every N actions；
  - remove unused audit in timed path；
  - pre-load payload shard；
  - use pinned async copy only if no CPU offload path enters official metric。

If P1 fidelity pass but density UCB < 0.03 at 5000:
  - run a 1024-action tail-enriched diagnostic panel, not official density；
  - compare tail-enriched vs G5 to see whether missing rare groups explain low density；
  - if tail-enriched recovers density, return to P1 tail repair。

If P1 fidelity pass and density still insufficient at 10000/20000:
  - declare natural AP0 action source density insufficient；
  - do not continue harvesting-only route；
  - open D-line only if B-line has a mechanism target。
```

---

## 10. P3：Density Anatomy and Source Adequacy Decision

### 目标

如果 density 低，判断是 generator 仍不保真，还是 AP0 自然动作源本身不够。

### 分析方式

对每个 stratum 估计：

$$
p_{old}(Y=1 \mid s)
$$

$$
p_{new}(Y=1 \mid s)
$$

以及：

$$
w(s)=\frac{P_{old}(s)}{P_{new}(s)}.
$$

做 importance-weighted density：

$$
\hat p_{iw}=\frac{\sum_s w(s) n_s^{new} \hat p_s^{new}}{\sum_s w(s)n_s^{new}}.
$$

### 判断标准

```text
If raw new density low but importance-weighted density near old density:
  generator sampling weights still wrong.

If raw and weighted density both low:
  natural source likely low-density.

If old reference density differs sharply by tail groups:
  current major distribution gate insufficient; tail group must enter official fidelity gate.
```

### 记录指标

```text
stratum_id
old_action_count
new_action_count
old_good_count
new_good_count
old_good_rate
new_good_rate
density_ratio
importance_weight
weighted_density_contribution
```

### Codex 尝试方向

```text
If weighted density recovers good rate:
  - reweight generator by under-sampled strata；
  - add quota table from high-contribution strata；
  - rerun P1/P2.

If weighted density also low:
  - stop natural harvesting as primary route；
  - document AP0 natural source low density；
  - move to B/D mechanism-driven update.
```

---

## 11. P4：Future Path Type Revalidation on Natural Panel

### 目标

在分布保真 natural panel 上重新确认 future path 现象，不只在 canonical 2876 上成立。

### 要比较的组

```text
Canonical Core77
Canonical OldOnly
Natural CoreLike
Natural PathGood
Natural SlowBurnGood
Natural RiskyHighAUV
Natural BadPath
RandomMatchedNatural
```

### 记录指标

每组、每个 horizon 记录：

```text
V1 / V5 / V20 / V80 / V240 LCB
AUV LCB
RiskAdjustedAUV LCB
LongRisk UCB
Bad UCB
Null UCB
MemoryFail UCB
OffdiagFail UCB
HardTail CEp99 delta
OldFamily loss delta
Cover proxy delta
```

### 判断标准

Future-path mechanism weak pass：

```text
Natural CoreLike or PathGood:
  RiskAdjustedAUV_LCB > 0
  V240_LCB > 0
  longrisk_UCB <= 0.05
  memory/offdiag_UCB <= 0.05
```

Strong pass：

```text
accepted_count >= 87
same quality gates pass
LDO/LSO/LTO/LFO <= 0.10
```

### 可视化

```text
future path curves by group
risk adjusted AUV vs longrisk scatter
SlowBurnGood h1-to-h240 improvement plot
RiskyHighAUV vs CoreLike comparison plot
```

### Codex 尝试方向

```text
If Natural CoreLike future path weaker than canonical Core77:
  - audit generator distribution in future-path precursor strata；
  - compare canonical vs natural payload/action norms and template lineage；
  - add tail fidelity repair.

If Natural PathGood has high AUV but high risk:
  - split PathGood into RiskCleanPathGood and RiskyPathGood；
  - do not use AUV alone.

If no natural good path types found but density panel says good density sufficient:
  - check label definitions / join keys / horizon rows.
```

---

## 12. P5：FuturePathOperator Sketch v5

### 目标

训练当下低成本估计未来路径收益，不再调 LC/FOS/FO old thresholds。

### 设计原则

不使用 future outcome，不使用 dataset name，不使用 validation/test metric。

允许使用：

```text
current train batch logits / gradients
train-memory buffer response
hard-tail mini buffer response
small virtual AdamW sketch
JVP / VJP response
payload norm / action norm / AdamW alignment
memory/offdiag legal proxy
cost estimate
```

### 候选 sketch

```text
FPO1 tiny-virtual-AdamW-1step
FPO2 tiny-virtual-AdamW-3step-shared-batch
FPO3 JVP-memory-response
FPO4 hard-tail-delayed-gain
FPO5 AdamW-aligned-delayed-gain
FPO6 risk-adjusted-delayed-gain
FPO7 value-rank + memory/offdiag veto
FPO8 ensemble hard gate without weighted score
```

### 记录指标

```text
feature_compute_ms_q50/q90/q99
TopK87 precision
V LCB
RiskAdjustedAUV LCB
LongRisk UCB
Bad UCB
Null UCB
Memory UCB
Offdiag UCB
LDO/LSO/LTO/LFO
accepted_count
```

### 通过标准

Weak pass：

```text
TopK87 precision >= 0.65
V_LCB > 0
LongRisk_UCB <= 0.10
Memory/Offdiag_UCB <= 0.10
cost_q90 <= 1.0 ms
```

Strong/controller-ready pass：

```text
accepted_count >= 87
precision >= 0.75
V_LCB > 0
RiskAdjustedAUV_LCB > 0
LongRisk_UCB <= 0.05
Bad_UCB <= 0.05
Null_UCB <= 0.15
Memory/Offdiag_UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
cost_q90 <= 0.50 ms
```

### Codex 尝试方向

```text
If cost too high:
  - reduce virtual steps；
  - cache memory buffer response；
  - use JVP/VJP instead of full apply；
  - batch probe actions together；
  - remove any Python loop in timed path.

If cost low but precision low:
  - inspect accepted action groups；
  - split value proxy and risk veto；
  - do not tune a single threshold；
  - add delayed-gain term rather than current-response term.

If precision high but risk high:
  - add hard veto for memory/offdiag/longrisk proxy；
  - do not add risk as weighted penalty.

If LDO high:
  - check score scale shift by dataset only as diagnostic；
  - use dataset-blind quantile normalization learned on train split；
  - do not branch on dataset name.
```

---

## 13. P6：Existing-Action Controller Gate

### 目标

只有 P2 density 或 P5 sketch 过线，才打开 controller。

### Controller candidates

```text
C1 natural-density-harvest-controller
C2 FPO-sketch-controller
C3 CoreLike + SlowBurnGood hybrid controller
C4 value-rank + hard risk/memory/offdiag veto controller
```

### 记录指标

```text
accepted_count
coverage
precision
V_LCB
RiskAdjustedAUV_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO/LSO/LTO/LFO
feature cost
payload apply cost
controller kernel count
```

### 通过标准

```text
accepted_count >= 87
precision >= 0.75
V_LCB > 0
RiskAdjustedAUV_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory/offdiag_UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
```

### Codex 尝试方向

```text
If accepted_count < 87 but precision/risk clean:
  - do not lower threshold blindly；
  - look for SlowBurnGood expansion；
  - rerun density / FPO to find expansion pool.

If precision low:
  - inspect accepted path types；
  - remove RiskyHighAUV and BadPath strata；
  - hard-veto memory/offdiag fail.

If LDO high:
  - decompose into quality/support/backfill；
  - if support/backfill only, require natural density proof before gate change；
  - no dataset-specific threshold.
```

---

## 14. P7：Generated Sandbox Reopen Gate

### 目标

只在必要且有新目标时打开 generated sandbox。

### Reopen 条件

Generated route allowed only if：

```text
Case A:
  Natural density insufficient on faithful panel
  and P5 FPO weak pass exists.

Case B:
  Natural density sufficient but controller cannot reach accepted_count/leaveout
  and P4 identifies a clear future-path mechanism.

Case C:
  Tail fidelity shows AP0 source misses a necessary recipe
  and generator can explicitly target that recipe without outcome labels.
```

### Sandbox size

```text
64 actions first
then 256 only if 64-action weak pass
```

### Generated pass standard

```text
new positive rate >= 0.10
GradeAB precision >= 0.25 for 64-action sandbox
V_LCB > 0
longrisk_UCB <= 0.10
no-transform sanity pass
payload/apply hash pass
branch-horizon completion = 1.0
```

### Codex 尝试方向

```text
If generated actions high-longrisk:
  - stop variant；
  - inspect memory/offdiag fail first；
  - add hard veto rather than weighted score.

If generated actions value-negative:
  - inspect future-path type；
  - check whether generated update only changes function magnitude but not delayed gain；
  - do not continue same primitive family.

If action apply mismatch:
  - stop science route and fix payload/apply replay.
```

---

## 15. P8：Runtime and Paired Replay Boundary

### 目标

只有 P6 controller pass 后才测 runtime；只有 runtime pass 后才做 paired replay。

### Runtime pass

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
feature_cost_q90 <= 0.50 ms or justified under total step ratio
payload_apply_error_linf = 0
controller no dataset branch
no CPU offload
```

### Paired replay pass

```text
RealFunctional beats AdamWParallel
RealFunctional beats bestLR
RealFunctional beats NoOp
RealFunctional beats Random
shuffled payload fails
LDO/LSO/LTO/LFO still pass
```

### Codex 尝试方向

```text
If runtime fails due feature cost:
  - cache FPO sketch features；
  - batch action scoring；
  - remove Python loop；
  - use persistent workspace.

If runtime fails due payload apply:
  - switch to fused apply for selected action block；
  - compact action representation；
  - preallocate buffers.

If paired replay fails:
  - separate controller error vs action effect error；
  - compare selected action future-path profile with canonical Core77；
  - do not tune dataset-specific threshold.
```

---

## 16. v9.9.3 Stop / Pivot Rules

### Stop natural harvesting route

```text
P1 major+tail fidelity pass
and P2 N >= 10000
and CoreLike_UCB < 0.03
and PathGood_UCB < 0.03
and SlowBurnGood_UCB < 0.03
```

Then：

```text
route = R-NaturalAP0DensityInsufficient
pivot = FuturePathOperatorDrivenGeneratedUpdate
```

### Stop G5 as generator

```text
P1 tail fidelity fail twice
or missing_tail_group_count > 0 for core precursor groups
or density remains low while tail-enriched diagnostic recovers density
```

Then：

```text
route = R-G5MajorPassTailFail
pivot = TailAwareNaturalGenerator
```

### Stop FuturePathOperator sketch

```text
three independent FPO families fail weak pass
and no path-type correlation with future value
```

Then：

```text
route = R-NoLegalFuturePathProxy
pivot = generated update only if density insufficient and new theory available
```

### Open existing-action controller

```text
P2 density sufficient
or P5 FPO strong pass
```

### Open generated sandbox

```text
P2 density insufficient
and P5 FPO weak pass
or P4 future path mechanism strong pass with explicit generator target
```

---

## 17. v9.9.3 最终 route 格式

本轮最后必须输出以下之一：

```text
R1-NaturalDensitySufficientExistingControllerCandidate
R2-NaturalDensityInsufficientGeneratorNeeded
R3-G5MajorPassTailFidelityFail
R4-NaturalDensityStillInconclusiveNeedLargerPanel
R5-FuturePathOperatorWeakPassControllerCandidate
R6-FuturePathOperatorFailDensityPending
R7-GeneratedSandboxReopenedWeakPass
R8-GeneratedSandboxStillStopped
R9-MaterializerOrFidelityEngineeringBlocked
```

---

## 18. 最后判断

v9.9.2 不是失败回退。它把 C 线从 “生成器分布失真” 推进到了 “G5 major distribution pass，但 density pending”。

但它也发出了一个更严肃的警告：

$$
\boxed{
\text{如果 G5 在保真分布下仍只有极低 CoreLike / PathGood 密度，existing-action harvesting 可能不是主路。}
}
$$

因此 v9.9.3 必须把自然动作密度问题真正裁决掉。不能再停在 1024 pilot，也不能只看 major PSI。下一轮的核心是：

```text
先确认 G5 是否保住好动作 tail；
再跑 sequential density；
再决定 existing-action route 是否继续；
最后才讨论 controller 或 generated update。
```
