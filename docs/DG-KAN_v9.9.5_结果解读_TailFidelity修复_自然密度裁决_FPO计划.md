# DG-KAN v9.9.4 结果解读与 v9.9.5 实验计划：Tail Fidelity 修复、自然密度裁决、FuturePathOperator 重写

> 目标读者：不假设读者了解前序所有实验。  
> 写法约束：不把任何未过 gate 的 diagnostic 写成 official pass；不按数据集调参；不使用 fake / proxy / CPU offload；公式使用 `$...$` 或 `$$...$$`，Typora 友好。

---

## 0. 一句话判断

v9.9.4 **有进展，但不是能力成功**。它不是 controller 成功，不是 runtime 成功，也不是 paired replay / short-full 成功。它真正推进的是：我们终于把自然 AP0 扩流从“粗分布修复”推进到“major + tail fidelity 双重审计”，然后发现当前所有 generator repair 都还不能同时保住粗分布和好动作可能所在的稀有尾部分布。

当前最准确状态是：

$$
\boxed{
\text{future path 好动作现象仍可信；但 natural generator 的 tail fidelity 仍未闭合，不能裁决自然好动作密度。}
}
$$

这意味着现在不能说：

```text
自然 AP0 action stream 没有足够好动作。
```

只能说：

```text
当前 generator 还没能生成一个足够像 canonical AP0、尤其像关键 tail 的自然动作样本流。
```

---

## 1. v9.9.4 关键结果复盘

v9.9.4 的主 route 是：

```text
route = CaseC-GeneratorFidelityStillFails
primary_blocker = no_official_major_tail_fidelity_generator_passed
secondary_blocker = density_panels_blocked
system_legal_controller_pass = 0
generated_route_status = stopped_generator_fidelity_failed
```

核心事实：

```text
P1 tail reference audit pass = 1
tail groups = 2191
support<3 tail groups = 2051
missing-rate = 0.0

P2 candidate count = 8
official pass count = 0

closest official-eligible generator = G9-two-stage-major-then-tail-fill-generator
G9 major PSI/JS/max-share = 0.077175 / 0.097658 / 0.402344
G9 tail PSI/JS/missing = 0.178288 / 0.134781 / 1

P3 largest completed density panel = 0
density sufficient / insufficient / inconclusive = 0 / 0 / 1

P4 future path weak / strong = 0 / 0
P5 FPO weak / strong = 0 / 0
P6 controller = not_run
P7 generated sandbox allowed = 0

No-fake audit:
rows checked = 2267
fake / proxy / cpu = 0 / 0 / 0
```

补充 C 线又尝试了 G7-G12 distribution-only 1024，但仍然没有任何候选同时通过 major + tail distribution gate：

```text
supplement route = Supplement-C-NoMajorTailDistributionPass
candidate_count = 6
distribution_pass_count = 0
branch_1024_opened = 0
panel_5000_opened = 0
```

其中几个候选的失败形态很有信息量：

```text
G7 exact-tail-quota-stochastic-refill:
  major PSI/JS/share = 2.506 / 0.503 / 0.614
  tail PSI/JS/missing/coverage = 11.246 / 0.680 / 144 / 0.329
  结论：tail 和 major 都严重失真。

G8 hierarchical-major-tail-quota:
  major PSI/JS/share = 0.458 / 0.237 / 0.535
  tail PSI/JS/missing/coverage = 1.069 / 0.339 / 17 / 0.847
  结论：比 G7 好，但 major/tail 都仍超 gate。

G11 g5-major-tail-replay-mixture:
  major PSI/JS/share = 0.281 / 0.186 / 0.526
  tail PSI/JS/missing/coverage = 0.780 / 0.263 / 13 / 0.859
  结论：tail PSI 是补充候选里相对低的，但 missing tail group 仍然是 13，major 也没过。

G12 rejection-resampled-tail-fidelity-generator:
  major PSI/JS/share = 0.838 / 0.317 / 0.555
  tail PSI/JS/missing/coverage = 7.681 / 0.500 / 10 / 0.806
  结论：missing tail group 最少，但分布整体更差。
```

---

## 2. 四线分别进展如何

### 2.1 A 线：未来训练路径

A 线不是本轮主执行线，因为 P3/P4 被 C 线 fidelity gate 阻断。但补充结果里，A 线对已落地 future rows 做了类型拆解，这个结果非常重要：

```text
landed_future_action_rows = 339
FastGood_count = 129
SlowBurnGood_count = 116
SlowBurnGood_relaxed_count = 148
RiskyHighAUV_count = 79
SafeLowValue_count = 46
BadPath_count = 81
A_line_generation_target_ready = 1
```

关键路径类型：

```text
FastGood:
  actions = 129
  V1/V20/V80/V240 LCB = 0.4777 / 1.4346 / 3.1072 / 5.0035
  RAUV LCB = 2.9206
  longrisk UCB = 0.0289
  memory/offdiag UCB = 0.0289 / 0.0289

SlowBurnGood:
  actions = 116
  V1/V20/V80/V240 LCB = -0.1460 / 0.7635 / 2.0461 / 3.5968
  RAUV LCB = 1.9000
  longrisk UCB = 0.0321
  memory/offdiag UCB = 0.0321 / 0.0321
  mechanism_signal = 1
```

这说明一个核心判断越来越稳：

$$
\boxed{
\text{好 functional update 不一定是当前一步 loss 下降最多，而可能是未来路径逐渐变好。}
}
$$

尤其 `SlowBurnGood` 很关键：它在 h1 上是负的，但后续 h20/h80/h240 持续变好。这比 FastGood 更接近我们要的“未来训练轨迹改善”概念。

A 线当前状态：

```text
现象可信；
路径类型初步成形；
但还不能 controller；
需要在 major+tail fidelity 通过后的自然 panel 上重验。
```

### 2.2 B 线：训练当下合法 FuturePathOperator

主 run 中 P5 没有打开，因为上游 fidelity 不过。补充 run 也没有给出可用 B 线机制。前几轮已经证明：

```text
cheap proxy 很便宜，但 precision / V 崩；
high-cost FO proxy 太贵，且仍不能形成正 V；
FOS / LC 类分数更多是在测当前响应或风险清理，不是在测 future path benefit。
```

B 线当前状态：

```text
不能继续调 LC / FOS threshold；
要重写为真正的 FuturePathOperator sketch；
必须预测未来路径类型，而不是预测当前函数响应大小。
```

下一步 B 线不能再问：

```text
哪个 cheap feature TopK 高？
```

而要问：

```text
训练当下能不能低成本预测 action 属于 FastGood / SlowBurnGood / RiskyHighAUV / BadPath？
```

### 2.3 C 线：自然 AP0 动作扩流

C 线是本轮主线，也是最大 blocker。

v9.9.0 已经证明 natural extension entrypoint 可以落地，single/16/256 smoke 能跑。v9.9.1 证明原 generator 的 1024 pilot 分布严重失真。v9.9.2 找到 G5 让 major distribution 看起来过线。v9.9.3 加 tail fidelity 后发现 G5/G6 不可信。v9.9.4 进一步审计 tail reference 并尝试更多 generator repair，但 official pass count 仍是 0。

C 线当前状态：

$$
\boxed{
\text{generator 工程可行；major distribution 可部分修复；tail fidelity 仍失败。}
}
$$

这不是小事。因为好动作可能位于稀有 tail，若 generator 不覆盖这些 tail，那么任何 density panel 都会低估好动作密度。

更直接地说：

```text
当前不能用 G7-G12 的 1024 distribution-only 失败，直接判定 AP0 natural source 不够；
也不能用 tail-failed generator 跑 5000/10000/20000；
必须先让 generator 同时通过 major + tail fidelity。
```

### 2.4 D 线：generated update

D 线继续停止是正确的。

原因非常简单：

```text
A 线有 future path 现象，但还没有训练当下可用机制；
B 线没有合法低成本 FPO；
C 线没有保真的自然密度裁决；
因此 generated route 没有优化目标。
```

若现在重开 generated sandbox，就会回到过去 APG/APGH/APGL/APGT 的老问题：

```text
payload 合法；
branch-horizon 能跑；
但 action value-negative / high-longrisk。
```

D 线当前状态：

```text
继续 strict gate；
只允许在 A/B/C 至少一条给出可部署目标后，开 64-action sandbox；
禁止直接 512-action 大跑。
```

---

## 3. 为什么感觉还是非常慢

你的感觉是对的。因为系统能力链路仍然没有打开：

```text
controller
-> selected runtime
-> paired replay
-> short/full training
```

从能力视角看，确实慢。

但这轮不是空转。它避免了一个关键错误：用不保真 tail 的 generator 去裁决自然好动作密度。如果没有 v9.9.4，我们很可能会拿 G5/G6/G9 的低密度结果误判为：

```text
AP0 natural action stream 本身没有好动作。
```

现在我们知道，这个结论还不能下。因为：

```text
tail reference audit 过；
support<3 的 tail group 非常多；
G9 仍有 tail PSI 0.178 和 missing 1；
补充 G7-G12 全部没过；
density panel 没打开是正确的。
```

---

## 4. 当前真正卡在哪里

### 4.1 Tail fidelity 是当前第一 blocker

现在不是卡在 action apply，不是卡在 branch-horizon，不是卡在 no-fake，也不是卡在 canonical table。现在卡在：

$$
\boxed{
\hat P_G(z_{tail}) \neq P_{natural}(z_{tail})
}
$$

其中：

```text
P_natural = canonical AP0 的自然动作分布；
P_G = generator 产生的新动作分布；
z_tail = 好动作可能所在的稀有 precursor / lineage / recipe tail。
```

如果 tail 不保真，估计出来的 good density 就没有意义。

### 4.2 Tail group 太碎

P1 显示：

```text
tail groups = 2191
support<3 = 2051
```

这说明 tail reference 本身极其稀疏。这里可能有两个风险：

```text
1. 如果 tail key 太细，任何 generator 都很难 pass；
2. 如果 tail key 太粗，会把好动作 tail 和普通 tail 混在一起，误判 density。
```

所以 v9.9.5 不能只继续加 generator；必须先审计 tail key 是否合理。

### 4.3 Major 与 tail fidelity 在互相冲突

补充候选显示：

```text
G11 tail PSI 相对较低，但 major 仍没过；
G12 missing tail 最少，但 major/tail PSI 更差；
G8 tail coverage 高，但 PSI/JS/share 不过；
G7 exact tail quota 反而 major/tail 都崩。
```

这说明简单 quota 不够。需要一种 hierarchical + support-aware + divergence-minimizing 的采样器，而不是简单 oversampling 或 exact quota。

### 4.4 B 线还没有真正的 future operator

现在我们有 path 类型，但没有训练当下低成本识别。当前 B 线不能只做“响应大小”。它要预测：

```text
FastGood / SlowBurnGood / RiskyHighAUV / SafeLowValue / BadPath
```

尤其要能识别 SlowBurnGood，因为这是 immediate loss descent 看不见的好更新。

---

## 5. 本轮最大的 insight

v9.9.4 最重要的 insight 是：

$$
\boxed{
\text{好动作密度不是整体分布问题，而是 tail-conditional density 问题。}
}
$$

更准确地写：

$$
P(Good)
=
\sum_z P(Good \mid z)P(z).
$$

如果好动作集中在少数 tail $z$ 上，那么即使 generator 的 major distribution 很像，只要 tail $P_G(z)$ 不对，估计出的 $P_G(Good)$ 就会严重偏离真实 $P_{natural}(Good)$。

因此，当前项目不能再只看：

```text
major PSI
major JS
max group share
global entropy ratio
```

必须看：

```text
tail PSI
tail JS
missing tail group
tail coverage
tail-conditional CoreLike / PathGood / SlowBurnGood density
tail-conditioned future path risk
```

---

## 6. 是否还在正确道路上

高层方向仍然对。v9.9.4 没有把粗分布 pass 写成 density closure，没有把 tail-failed generator 写成 official，没有把 diagnostic panel 写成 controller，没有 fake/proxy/cpu，也没有重开 generated sandbox。

但执行上要变得更硬：

```text
不要继续只加 G13/G14/G15 这种名字；
先审计 tail key；
再做可证明保 major+tail 的 generator；
再做 density；
再做 FPO；
最后才 controller/generated。
```

---

## 7. 离目标还差多远

离 system-legal local functional controller 至少还差四道门：

```text
1. natural generator 同时通过 major fidelity 和 tail fidelity；
2. CoreLike / PathGood / SlowBurnGood density 在保真 full natural panel 上被裁决；
3. 训练当下合法、低成本、跨分布稳定的 FuturePathOperator 或 controller 过线；
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

---

# v9.9.5 实验计划：Tail Key 审计、Major+Tail 保真生成器、自然密度裁决、FuturePathOperator 重写

## 8. v9.9.5 总目标

v9.9.5 的目标不是继续“修一个 generator 名字”。本轮要回答：

$$
\boxed{
\text{能否构造一个不使用 outcome label、同时保住 major 与 tail 分布的 natural AP0 extension generator？}
}
$$

如果可以，就打开 sequential density panel，裁决自然动作源中好路径动作的真实密度。

如果不可以，就先不要再讨论 controller / generated sandbox / runtime。因为样本分布不可信，所有 density 结论都不可信。

---

## 9. v9.9.5 总体结构

v9.9.5 分成四条线，但优先级明确：

```text
C 线最高优先级：
  Tail key audit -> Generator repair -> 1024 fidelity -> 5000/10000/20000 density。

A 线并行：
  用已落地 rows 拆 future path 类型；
  在 C 线 pass 后重验 natural panel path type。

B 线并行但限量：
  重写 FuturePathOperator v7；
  不再调 LC/FOS 旧阈值。

D 线严格 gate：
  C 或 B 没过时，不开 generated sandbox；
  过线后只允许 64-action sandbox。
```

---

## 10. P0：复现 v9.9.4 边界

### 目标

确认本轮仍从 v9.9.4 的真实状态出发。

### 必须记录

```text
source_route
P1_tail_reference_audit_pass
tail_group_count
support_lt3_tail_group_count
P2_candidate_count
official_pass_count
closest_generator_id
closest_major_PSI
closest_tail_PSI
P3_largest_completed_panel
P4_future_path_pass
P5_FPO_pass
P6_controller_status
P7_generated_status
fake/proxy/cpu audit
```

### Pass

```text
P0_boundary_pass = 1
if:
  source_route = CaseC-GeneratorFidelityStillFails
  official_pass_count = 0
  no_fake/proxy/cpu = 1
```

### 如果不满足，Codex 先尝试

```text
1. 检查 v9940 artifact path 是否指向最新 run，而不是补充 run 或旧 run。
2. 检查 route_decision_v9940.json 是否存在。
3. 检查 P2 official pass count 是否来自 official-only，不要把 G12 diagnostic upper-bound 算进去。
4. 检查 resume-after-p2 是否造成重复/空 best 路径污染。
```

---

## 11. P1：Tail key reference audit v2

### 目标

判断当前 tail key 是否过细、过粗、或含有不适合 official density 的字段。

### 核心假设

$$
H_{tail-key}:
\text{v9.9.4 的 tail fidelity 失败，部分原因可能是 tail key 过度碎片化。}
$$

P1 不允许使用 outcome label 定义 official tail。允许使用：

```text
dataset_id 作为 stratification / audit，不作为 selector；
family；
step_bucket；
candidate_template；
source recipe；
payload norm bucket；
action norm bucket；
memory/offdiag precursor bucket；
hard-tail precursor bucket；
合法 commit-time precursor。
```

不允许使用：

```text
CoreLike；
PathGood；
SlowBurnGood；
future V；
RAUV；
LongRisk outcome；
GradeAB；
old outcome table；
future branch outcome。
```

### 必须记录

```text
tail_key_version
tail_key_fields
tail_group_count
support_lt3_group_count
support_lt5_group_count
support_lt10_group_count
largest_group_share
entropy
entropy_ratio
tail_group_missing_rate
tail_group_stability_by_seed
tail_group_stability_by_family
tail_group_stability_by_template
tail_key_leakage_audit
```

### 需要输出 4 个 tail key 版本

```text
TK0-original-v9940-tail-key
TK1-merged-support3-tail-key
TK2-hierarchical-major-tail-key
TK3-precursor-only-no-template-tail-key
TK4-memory-offdiag-hardtail-precursor-tail-key
```

### 判断标准

P1 strong pass：

```text
tail_key_leakage_count = 0
tail_group_count <= 512
support_lt3_fraction <= 0.50
largest_group_share <= 0.35
entropy_ratio >= 0.80
```

P1 weak pass：

```text
tail_key_leakage_count = 0
tail_group_count <= 1024
support_lt3_fraction <= 0.75
largest_group_share <= 0.45
entropy_ratio >= 0.70
```

### 如果不满足，Codex 先尝试

```text
如果 support<3 group 占比仍 > 0.75：
  先合并 rare tail group，不要继续加 generator。
  合并策略优先级：
    1. within same major group merge；
    2. within same family/step merge；
    3. merge by memory/offdiag/hardtail precursor；
    4. last resort: rare_tail_bucket。

如果 largest group share > 0.45：
  说明 tail key 过粗或某个字段塌缩；
  检查 payload/action norm bucket 是否全为同一值；
  检查 candidate_template 是否被错误归一化。

如果 leakage audit fail：
  移除 outcome-derived 字段；
  重新生成 tail reference；
  不允许进入 P2。
```

---

## 12. P2：Major+Tail Fidelity Generator Repair Matrix v2

### 目标

在 P1 选出的 tail key 下，构造一个同时通过 major fidelity 与 tail fidelity 的 natural generator。

### Generator 候选

```text
G13-support-aware-hierarchical-generator
  先按 major quota 采样，再在每个 major 内按 merged-tail quota 采样；
  对 support<k 的 tail 使用 rare-tail bucket。

G14-tail-balanced-with-major-projection-generator
  先 tail-balanced 采样，再用 major projection 修正 major PSI。

G15-min-divergence-transport-generator
  把 generator 采样看成 min KL / min PSI transport；
  约束 max_group_share。

G16-canonical-tail-replay-plus-new-residual-generator
  从 canonical tail precursor replay recipe；
  再生成新 payload residual；
  保证 action_id / payload_hash 不 collision。

G17-two-buffer-generator
  一个 buffer 保 major distribution；
  一个 buffer 保 tail coverage；
  交替填充，直到 PSI/JS 双门过线。

G18-sequential-rejection-generator
  每生成一个 action 就在线更新 major/tail divergence；
  超过 divergence budget 的候选直接拒绝。

G19-stratified-random-baseline-v2
  作为负控；
  不应该 strong pass。
```

### 必须记录

```text
generator_id
tail_key_version
action_count
new_action_count
old_action_collision_count
payload_collision_count
major_PSI
major_JS
major_max_share
major_entropy_ratio
tail_PSI
tail_JS
tail_missing_group_count
tail_coverage
tail_entropy_ratio
tail_support_lt3_coverage
rows_sec_distribution_only
branch_horizon_rows_if_opened
action_apply_linf_max_if_opened
```

### Pass

P2 official pass：

```text
major_PSI <= 0.05
major_JS <= 0.08
major_max_share <= 0.35
major_entropy_ratio >= 0.90
tail_PSI <= 0.05
tail_JS <= 0.08
tail_missing_group_count = 0
tail_coverage >= 0.95
tail_entropy_ratio >= 0.90
old_action_collision_count = 0
payload_collision_count = 0
```

P2 weak pass：

```text
major_PSI <= 0.08
major_JS <= 0.12
major_max_share <= 0.40
major_entropy_ratio >= 0.85
tail_PSI <= 0.08
tail_JS <= 0.12
tail_missing_group_count <= 1
tail_coverage >= 0.90
tail_entropy_ratio >= 0.85
```

### 如果不满足，Codex 先尝试

```text
如果 major pass 但 tail fail：
  不调 major quota；
  增加 tail quota 或 rare-tail merge；
  检查 missing tail group 是否 quota rounding 到 0；
  检查 filter 是否删掉该 tail。

如果 tail pass 但 major fail：
  使用 major projection 修正；
  限制 tail oversampling 的 major share；
  降低 rare-tail oversampling 权重。

如果 both fail：
  回到 P1 tail key；
  当前 tail key 可能不可采样。

如果 old_action_collision > 0：
  修 action_id generation；
  不能用 canonical action_id。

如果 payload_collision > 0：
  检查 payload seed / recipe / shard writer；
  确保 payload hash 新增。

如果 distribution-only pass 但 branch-horizon smoke fail：
  检查 payload tensor write；
  action apply replay；
  branch/horizon materializer；
  不要直接跑 5000 panel。
```

---

## 13. P3：Sequential Natural Density Panel

### 目标

只有 P2 official 或 weak pass 后，才裁决自然动作流中好路径动作的密度。

### Panel 顺序

```text
1024 -> 5000 -> 10000 -> 20000
```

每个 panel 都要真实 materialize：

```text
action rows；
payload rows；
branch-horizon rows；
future path labels；
density labels；
no-fake audit。
```

### 要统计的标签

```text
CoreLike
PathGood
FastGood
SlowBurnGood
SlowBurnGoodRelaxed
RiskCleanButLowValue
RiskyHighAUV
BadPath
CoreLikeOrSlowBurn
```

### 每个标签必须记录

```text
count
rate
Wilson_LCB
Wilson_UCB
V_LCB
RAUV_LCB
V240_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
dataset/family/template support
major/tail support
```

### Density decision

```text
if CoreLikeOrSlowBurn_LCB >= 0.03:
  density_sufficient = 1

elif CoreLikeOrSlowBurn_UCB < 0.03 and panel_size >= 10000:
  density_insufficient = 1

else:
  density_inconclusive = 1
  continue next panel
```

### 如果不满足，Codex 先尝试

```text
如果 1024 panel density very low:
  不立即判 density insufficient；
  先检查 P2 fidelity 是否仍 pass；
  检查 tail-conditioned density 是否集中在少数 tail。

如果 5000 panel UCB < 0.03:
  可以提前准备 generated route pivot；
  但若 P2 fidelity marginal，只先修 fidelity。

如果 10000 panel UCB < 0.03:
  判 natural AP0 harvesting 不足；
  D 线可准备 generated sandbox objective。

如果 20000 panel LCB >= 0.03:
  existing-action harvesting route 可继续；
  进入 P6 controller。
```

---

## 14. P4：Future Path Type Revalidation on Fidelity Panel

### 目标

在 P3 保真 natural panel 上重新验证 A 线，不再只依赖旧 339 landed future actions。

### 必须记录

```text
path_type
action_count
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
support_by_dataset
support_by_family
support_by_tail
```

### Pass

A-line mechanism pass：

```text
SlowBurnGood_count >= 64
SlowBurnGood_RAUV_LCB > 0
SlowBurnGood_V240_LCB > 0
SlowBurnGood_longrisk_UCB <= 0.05
SlowBurnGood_memory/offdiag_UCB <= 0.05
FastGood_count + SlowBurnGood_count >= 87
```

### 如果不满足，Codex 先尝试

```text
如果 FastGood 多但 SlowBurnGood 少：
  未来路径机制可能更偏 immediate-good；
  重新评估是否需要 SlowBurnGood 作为 generated target。

如果 SlowBurnGood 多但 tail support 单一：
  检查 tail leakage / tail collapse；
  不能直接 controller。

如果 RiskyHighAUV 多：
  强化 risk-adjusted path type；
  不允许只用 AUV。

如果 SafeLowValue 多：
  检查是否存在 h240 以后才变好的 ultra-slow path；
  作为 diagnostic，不直接 official。
```

---

## 15. P5：FuturePathOperator Sketch v7

### 目标

构造训练当下合法、低成本、能预测 future path type 的 operator sketch。

### 允许输入

```text
current batch gradients
train-memory buffer response
hard-tail mini-buffer response
per-example gradient mean/variance
JVP/VJP sketch
AdamW alignment
action norm / payload norm
memory/offdiag precursor
no dataset_name selector
no future outcome
no old outcome table
```

### Sketch 候选

```text
FPO7A-tiny-virtual-adamw-1step
FPO7B-tiny-virtual-adamw-3step
FPO7C-jvp-vjp-gradient-transport
FPO7D-signal-reservoir-snr-gate
FPO7E-hardtail-memory-delayed-gain
FPO7F-risk-adjusted-path-type-classifier
FPO7G-slowburn-detector
FPO7H-fastgood-slowburn-two-head
```

### 必须记录

```text
fpo_id
feature_count
feature_legality
feature_cost_q90_ms
TopK87_precision_CoreLikeOrSlowBurn
TopK87_V_LCB
TopK87_RAUV_LCB
TopK87_longrisk_UCB
TopK87_bad_UCB
TopK87_null_UCB
TopK87_memory_UCB
TopK87_offdiag_UCB
LDO/LSO/LTO/LFO
path_type_confusion_matrix
false_positive_type
false_negative_type
```

### Pass

P5 strong pass：

```text
TopK87_precision_CoreLikeOrSlowBurn >= 0.75
V_LCB > 0
RAUV_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory/offdiag_UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
feature_cost_q90_ms <= 1.50
```

P5 weak pass：

```text
TopK87_precision >= 0.60
V_LCB > 0
longrisk_UCB <= 0.10
feature_cost_q90_ms <= 3.00
```

### 如果不满足，Codex 先尝试

```text
如果 precision 低但 risk 低:
  proxy 只是 veto，不是 selector；
  增加 value / delayed gain head。

如果 V positive 但 longrisk 高:
  加 hard risk gate；
  不调 value threshold。

如果 SlowBurn false negative 高:
  加 h1-negative-but-h20-positive synthetic contrast；
  不把 h1 response 当硬门。

如果 cost 高:
  降低 virtual samples；
  cache JVP/VJP；
  用 stratified memory mini-buffer；
  禁止 high-cost FPO 进入 official controller。

如果 LDO 高:
  做 dataset-blind calibration；
  不允许 dataset-specific threshold。
```

---

## 16. P6：Existing-Action Controller Gate

### 条件

只有满足以下任一条件才打开：

```text
P3 density_sufficient = 1
or
P5 strong pass = 1
```

### Controller 形式

```text
Accept(a)=1
if:
  FuturePathOperator(a) passes
  and risk/memory/offdiag hard gates pass
  and cost gate pass
```

不允许：

```text
dataset_name branch
outcome-derived feature
old table feature
tail group id direct selector
future label feature
```

### 必须记录

```text
accepted_count
coverage
precision
V_LCB
RAUV_LCB
V240_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_UCB
offdiag_UCB
LDO/LSO/LTO/LFO
template/family/tail support
feature_cost_q90_ms
controller_cost_q90_ms
```

### Pass

```text
accepted_count >= 87
precision >= 0.75
V_LCB > 0
longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory/offdiag_UCB <= 0.05
LDO/LSO/LTO/LFO <= 0.10
```

### 如果不满足，Codex 先尝试

```text
accepted_count < 87:
  如果 density_sufficient=0，回 C/D；
  如果 density_sufficient=1，检查 support split 是否过严。

precision low:
  查看 false positive path type；
  如果 RiskyHighAUV 多，强化 risk path gate；
  如果 SafeLowValue 多，加入 delayed-gain check。

LDO high:
  做 support/backfill decomposition；
  不按 dataset 调 threshold。
```

---

## 17. P7：Generated Sandbox Gate

### 条件

只在以下情况打开 64-action sandbox：

```text
Case 1:
  P3 density_insufficient = 1
  说明 natural harvesting 不够，需要生成。

Case 2:
  P5 strong pass = 1
  说明有训练当下 FPO 可作为生成目标。

Case 3:
  P4 SlowBurnGood mechanism pass = 1
  说明有明确 path type 生成目标。
```

### Sandbox 目标

生成动作不能再只是合法 payload。目标必须是：

```text
FastGood 或 SlowBurnGood；
low longrisk；
memory/offdiag safe；
low bad/null；
cost legal。
```

### 候选生成器

```text
GUP1-FPO-guided-small-delta
GUP2-slowburn-targeted-update
GUP3-memory-offdiag-safe-delayed-gain
GUP4-signal-reservoir-projected-update
GUP5-risk-adjusted-future-path-update
GUP6-negative-control-shuffled-fpo
```

### Pass

```text
generated_action_count = 64
branch_horizon_completion = 1
FastGoodOrSlowBurn precision >= 0.30
V_LCB > 0
longrisk_UCB <= 0.10
new_positive_rate >= 0.10
generated_damage_V_LCB >= -0.10
```

### 如果不满足，Codex 先尝试

```text
如果 generated value-negative:
  不调 scale；
  先查是否目标类型错，是否生成成 SafeLowValue。

如果 high longrisk:
  检查 memory/offdiag/hard-tail gate；
  不允许只因 RAUV 高继续。

如果 new_positive_rate low:
  检查 FPO objective 是否过弱；
  回 B 线，不继续大跑。

如果 negative control pass:
  generator/eval pipeline 有问题；
  停止 D 线。
```

---

## 18. P8：Selected Runtime 与 Paired Replay Boundary

### 条件

只有 P6 controller pass 或 P7 generated sandbox strong pass 才打开。

### Runtime 必须记录

```text
step_ratio_q90
feature_compute_q90
FPO_compute_q90
controller_compute_q90
payload_apply_q90
memory_ratio
kernel_count
sync_count
active_step_count
empty_step_count
```

Pass：

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
empty_step_count = 0
```

### Paired replay 必须记录

```text
RealFunctional vs AdamWParallel
RealFunctional vs bestLR
RealFunctional vs NoOp
RealFunctional vs Random
shuffled payload control
leave-dataset-out
leave-stratum-out
leave-template-out
leave-tail-out
```

Pass：

```text
RealFunctional beats all controls
shuffled payload fails
LDO/LSO/LTO/LFO stable
```

---

## 19. 可视化要求

v9.9.5 必须输出这些图：

```text
1. Tail support histogram for TK0-TK4
2. Major vs Tail PSI scatter for G13-G19
3. Missing tail group heatmap by generator
4. Tail-conditioned density curve for CoreLike / PathGood / SlowBurnGood
5. Sequential panel Wilson CI curve
6. Future path type count bar chart
7. Future path V1/V5/V20/V80/V240 line plot
8. FPO false positive / false negative confusion matrix
9. Controller accepted region value-risk plot
10. Stop/pivot decision matrix
```

---

## 20. v9.9.5 Stop / Pivot 规则

### Stop natural harvesting route

如果在 major+tail pass 的 generator 上：

```text
20000 panel CoreLikeOrSlowBurn UCB < 0.03
```

则判定：

```text
natural AP0 source density insufficient
```

转向 generated route，但只允许基于 P4/P5 机制生成。

### Continue natural harvesting route

如果：

```text
CoreLikeOrSlowBurn LCB >= 0.03
```

则继续 existing-action controller route。

### Stop FPO route

如果：

```text
P5 strong = 0
and P5 weak = 0
and false positive dominated by RiskyHighAUV or BadPath
```

则 FPO v7 失败，不能继续调阈值。

### Open generated sandbox

仅当：

```text
density_insufficient = 1
or FPO strong = 1
or SlowBurnGood mechanism pass = 1
```

否则 generated sandbox 必须保持 not_run。

---

## 21. 本轮最低有效推进

即使 v9.9.5 仍不能 system pass，也必须至少完成下面之一：

```text
1. 找到一个 major+tail fidelity pass 的 generator；
2. 证明当前 tail key 不可采样，需要重新定义 tail key；
3. 在保真 generator 上裁决自然好动作密度；
4. 证明 FPO v7 能或不能识别 SlowBurnGood；
5. 明确 generated route 是否可以因 density insufficient 或 FPO strong 而重开。
```

如果这五个都没有完成，就说明实验仍然在工程循环里，必须进一步缩小 scope，只做 P1/P2 fail-fast。

---

## 22. 最终战略判断

v9.9.4 后，项目的第一优先级不是 controller，不是 runtime，也不是 generated update，而是：

$$
\boxed{
\text{先把 natural AP0 extension generator 的 tail fidelity 做可信。}
}
$$

因为没有可信 tail fidelity，就没有可信 density；没有可信 density，就不能判断 existing-action route 是否可行；没有 density 或 FPO 证据，就没有 generated route 的目标。

这不是保守，而是避免错误地把采样器失败误判为理论失败。
