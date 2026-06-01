# DG-KAN v9.5.8 Group-Stable Legal Rank / Memory-Safe Signal Primitive / Parallel Closure 完整实验计划

> 本计划基于 v9.5.7 `Legal Causal Geometry Rank / Memory-Preserving Primitive / Parallel Closure` 的真实执行结果制定。  
> v9.5.8 不再继续微调 APGM1-APGM8，也不把 legacy GCERT / outcome-derived rank 重新包装成 controller。  
> 本轮目标是把 v9.5.7 中首次出现的强 legal rank 诊断信号，转化为一个 **group-stable、memory-safe、训练当下可用、且可进入 selected runtime 的 functional update 选择规则**；同时并行重做新动作生成器，使它不再只是合法 payload builder，而是真正生成能保住旧知识、压住 long-risk、并有正收益的动作。

---

# 0. 一句话目标

v9.5.8 要回答的问题不是：

```text
APGM2 或 CERT-R2 再调一下能不能过？
```

而是：

```text
我们能不能在不使用 outcome 字段、不按数据集调参的前提下，
用训练当下可见的信息，稳定挑出或生成短期有收益、中期不坏、长期不忘的 functional update 动作？
```

写成公式：

$$
GoodUpdate(a)=1
$$

当且仅当：

$$
V_{20}^{LCB}(a)>0
$$

$$
V_{80}^{LCB}(a)\ge -\epsilon_{80}
$$

$$
V_{240}^{LCB}(a)\ge -\epsilon_{240}
$$

$$
LongRisk_{240}^{UCB}(a)\le \tau_L
$$

$$
Bad^{UCB}(a)\le \tau_B
$$

$$
Null^{UCB}(a)\le \tau_N
$$

$$
MemoryFail^{UCB}(a)\le \tau_M
$$

$$
CoverCollapse^{UCB}(a)\le \tau_C
$$

$$
Cost(a)\le C_{max}
$$

并且这个判断必须来自 commit-time legal fields，而不是 replay 后才知道的结果。

---

# 1. 对 v9.5.7 的独立判断

v9.5.7 的 route 是：

```text
route = R4-APGMGeneratedFrontierFail
success_v9570_strict_purekan_functional = False
success_v9570_full_functional = False
success_v9570_external_ready = False
primary_blocker = apgm_generated_frontier_failed
```

我认为这个 route 对“当前 APGM1-APGM8 generated action family”是合理的，但它没有完整表达本轮最重要的科学变化。

v9.5.7 的真正推进不是 APGM，而是 legal rank。

在 v9.5.6，我们以为最强的 GCERT18 TopK 信号是一个希望，后来发现它用了 outcome-derived fields，因此不能 official。v9.5.7 做了 field legality ledger，明确 legacy full score 有 red fields，best legal score variant 完全复现不了 legacy TopK。这一步很重要，因为它阻止了错误 promotion。

但 v9.5.7 又发现：legal signal 并不是完全没有。P3 的 `ControlTransferImprovement` 有很强 AUC 和正 value LCB，但 long-risk 不闭合；P5 的 small-MLP diagnostic ranker 在 TopK87 上已经接近 official accepted count 的下限，而且 value / long-risk / bad 都接近可用，只是 LDO / LSO drop 太大；P6 的 legacy-to-legal distillation sandbox 也说明，不使用 red fields 的 student 可以学到相当一部分 teacher 排名。

因此，我把 v9.5.7 的真实结论改写为：

$$
\boxed{
\text{legal rank 不再是完全看不见；但它还不够 group-stable，也不能直接 controller。}
}
$$

v9.5.7 的第二个关键发现是 memory blocker。P7 显示 high-longrisk actions 中大约 76% 可以被 memory failure 解释；这说明 long-risk 不是泛泛的风险，而很可能是 functional update 破坏了旧 family / old stratum 的局部 cover 或固定点区域。

因此，v9.5.8 的主线不应该是：

```text
继续调 APGM2；
继续调 CERT-R2 threshold；
继续追 AUC；
继续做小 generator 变体。
```

而应该是：

```text
1. 解释 legal rank 为什么 LDO/LSO drop 极大；
2. 将 legal rank 从 in-domain TopK 变成 group-stable accepted region；
3. 将 memory failure 从诊断字段变成 generator 的硬约束；
4. 设计能保留旧 cover / old-family prediction / signal direction 的新 primitive；
5. 只有当 selected controller 真实过线后，才打开 selected runtime、paired replay、short/full training。
```

---

# 2. v9.5.7 数据复盘

## 2.1 已完成的推进

v9.5.7 完成了以下实事：

```text
field legality ledger pass = 1
legacy rank leakage explained = 1
distillation pass = 1
memory blocker anatomy pass = 1
APGM implementation pass = 1
APGM branch-horizon pass = 1
APGM branch-horizon rows = 12288 / 12288
no fake / no proxy / no CPU offload = 1
manual forward / backward / AdamW update = 1
```

这说明 v9.5.7 不是空跑。它在三个层面推进了系统：

```text
1. 合法字段边界更清楚；
2. legal rank 的上界信号开始出现；
3. APGM 新动作生成与 outcome materializer 工程链路闭合。
```

## 2.2 legal rank 的关键数字

P3 `ControlTransferImprovement`：

```text
AUC_GradeAB = 0.9850698985803052
TopK64 GradeAB precision = 0.484375
TopK64 V_integrated LCB = 0.34107731007668374
TopK64 h240 longrisk UCB = 0.5428703985679182
```

这说明它确实能找到高 value 区域，但同时也会选到大量 long-risk。它不是可部署 rank，而是一个 value 方向信号。

P5 `UB-M1-small-mlp-diagnostic-ranker`：

```text
TopK64 GradeAB precision = 0.921875
TopK64 V_integrated LCB = 0.19684482149968938
TopK64 h240 longrisk UCB = 0.0
TopK87 GradeAB precision = 0.8275862068965517
TopK87 V_integrated LCB = 0.16125862511726052
TopK87 h240 longrisk UCB = 0.054480608015849245
TopK87 bad UCB = 0.054480608015849245
TopK87 memory fail UCB = 0.07282499698983116
LDO / LSO drop max = 0.8385416666666666
```

这组数字非常重要。TopK64 几乎像一个好 accepted region，但 coverage 不够；TopK87 刚好接近 $0.03$ coverage 下限，却在 bad / longrisk UCB 上略高，同时 LDO/LSO 大幅崩。这说明当前 ranker 不是没有能力，而是不稳定。

P6 distillation sandbox：

```text
student red field count = 0
student_teacher_topk87_overlap = 0.7816091954022989
student_true_gradeab_topk87_precision = 0.7241379310344828
student_true_V_lcb = 0.12807606232790217
student_true_longrisk_ucb = 0.0
student_true_bad_ucb = 0.03389313880385762
student_true_null_ucb = 0.10637805008576995
```

这说明 green-field student 能学到一部分合法排序，但还没有达到 official precision / stability。

## 2.3 APGM 的关键数字

APGM 工程链路过了：

```text
generated actions = 512
payload hash missing = 0
certificate hash missing = 0
action apply L∞ max = 0.0
branch-horizon rows = 12288 / 12288
quality audit pass = 1
```

但 APGM outcome 失败：

```text
best = APGM2-PopulationRiskOffDiagonalGate
GradeB precision = 0.0
GradeAB precision = 0.015625
V_integrated LCB = -0.3176437233277188
h240 longrisk UCB = 0.7726149255507754
```

所以 APGM 不是没实现，而是没有生成可用动作。APGM2 名字里有 population-risk / off-diagonal gate，但实际输出仍然 value-negative、高 long-risk。这说明我们目前只是把 population-risk / memory 的想法写进 generator 名字或局部规则里，还没有真正把它变成生成目标。

## 2.4 certificate 的关键数字

`CERT-R2-LegalRankConformalRiskBound`：

```text
TopK64 GradeAB precision = 0.859375
heldout accepted = 87
coverage = 0.08630952380952381
heldout V_integrated LCB = -0.1561252169840407
h240 longrisk UCB = 0.2517897676033668
```

这说明 rank-based certificate 的 TopK 很强，但它在 heldout accepted region 上不能保证 value 和 long-risk。也就是说，它不是一个稳定的线上选择器。

---

# 3. 本轮结论：问题到底在哪里

v9.5.7 之后，不能再说：

```text
完全没有合法信号；
完全没有好动作；
完全没有 ranking 方向；
只是 APGM implementation 没跑通。
```

这些都不准确。

现在更准确的判断是：

$$
\boxed{
\text{已有合法排序线索，但它不稳定；已有 memory 诊断，但 generator 没学会保 memory；已有 TopK 证书，但 heldout region 不安全。}
}
$$

当前卡在四件事。

第一，legal rank 的 TopK 能力和 leave-out 稳定性冲突。P5 的 TopK64/87 很强，但 LDO/LSO drop 到 0.8385，说明它可能依赖某些 dataset / family / stratum / step bucket 的局部模式。我们不能把它写成 dataset-agnostic controller。

第二，value signal 和 long-risk signal 没有被分开处理。P3 的 ControlTransferImprovement 会选到 value-positive 但 long-risk 高的动作。下一步 rank 不能只有一个分数，必须至少有 value rank + long-risk veto + memory veto。

第三，memory blocker 已经成立，但 APGM 没有真的解决它。P7 证明 memory failure 解释了大量 high-longrisk；但 APGM1-APGM8 生成出来的动作仍然 high-longrisk。这说明 generator 不是“memory preserving”，只是“带 memory 名字的合法 payload builder”。

第四，certificate 的目标错了。现在的 certificate 可以模仿某些排名，但不能保证 accepted region 的 $V$ 和 risk。v9.5.8 应该把 certificate 从概率校准转为 rank-safe set guarantee：我选出来的这一批动作，必须整体满足 value LCB、long-risk UCB、bad UCB、memory UCB，而不是只看 AUC 或 ECE。

---

# 4. 与相关工作的连接

Deep Manifold 的启发是：神经网络不是只在固定坐标里降低 loss，而是在训练中移动局部 cover、调整局部坐标、改变曲率，并逐渐形成稳定区域。一个 functional update 如果破坏旧 cover、让 old-family 预测漂移、或让某些局部 patch 过度僵硬，即使短期 value 看起来正，也会在 h240 变成长程风险。

因此，v9.5.8 要把 memory / cover 从诊断项变成生成项。

A Theory of Generalization in Deep Learning 的启发是：降低训练误差的方向不一定影响测试；有些方向只把误差留在训练可见但测试不可见的 reservoir 里。真正有价值的是多个样本一致支持、能转移到 heldout / leave-out 的 signal direction。因此，v9.5.8 的 legal rank 必须测 LDO/LSO drop、batch agreement、leave-one-out transfer、off-diagonal population-risk gain，而不是只看 in-domain TopK。

---

# 5. v9.5.8 核心假设

## H1：legal rank 不是完全不存在，但目前过拟合 group / stratum / step pattern

现象：

```text
P5 TopK87 precision = 0.8276
P5 V LCB = 0.1613
P5 longrisk UCB = 0.0545
但 LDO/LSO drop = 0.8385
```

假设：

```text
ranker 找到的是局部有效模式，不是 group-stable signal channel。
```

判断 H1 成立的标准：

```text
不同 dataset / family / stratum / step bucket 的 TopK precision 方差大；
rank score 在某些 groups 中集中失效；
leave-one-group-out 后 accepted overlap 大幅下降；
score 与 group identity 的 mutual information 偏高。
```

H1 不成立的标准：

```text
LDO/LSO drop 主要来自 target density 太低或统计置信不足，而不是 rank pattern 崩坏。
```

## H2：value signal 与 long-risk / memory signal 必须拆成多头判断

假设：

```text
单一 score 会把 high value 但 high long-risk 的动作排到前面。
```

判断 H2 成立的标准：

```text
TopK by value-only score: V LCB > 0 but longrisk UCB > 0.20；
TopK by memory-only score: longrisk low but V LCB <= 0；
two-stage value + memory veto 明显优于任何单头 score。
```

## H3：APGM failed because it does not preserve old-family local geometry

假设：

```text
APGM 产生 long-risk 的主要原因是 old-family / old-stratum prediction drift、cover entropy collapse、basis rank collapse 或 old gradient conflict。
```

判断 H3 成立的标准：

```text
APGM longrisk actions 中 memory_fail 占比 >= 0.70；
APGM source-to-generated damage 与 memory_fail / cover_collapse 强相关；
APGM generated actions 的 old-family CE / margin / logits drift 显著高于 existing GradeAB actions。
```

## H4：existing-action rank path 比 generated-action path 更接近 v9.5.8 可用系统

假设：

```text
当前 AP0 existing actions 中已经有一些可用区域；先把 legal rank 稳定化，比继续生成新 payload 更可能快速进入 controller。
```

判断 H4 成立的标准：

```text
cross-fitted legal rank controller 能在 heldout + LDO/LSO 上过 weak gate；
APGA generated primitive 仍不能产生足够 GradeAB / memory-safe actions。
```

H4 失败标准：

```text
existing-action legal rank 无法过 group-stable gate，但 APGA 新 primitive 产生足够新 positive。
```

## H5：如果 legal rank 只能通过 distillation 学到，而不能直接由 primitive formula 解释，则 v9.5.8 仍不能 external-ready

假设：

```text
small MLP / distillation 可以作为 diagnostic upper-bound，但 official controller 必须有 minimal interpretable score 或严格 cross-fit artifact。
```

判断 H5 成立的标准：

```text
black-box ranker pass 但 monotone/minimal score fail；
pass 只能依赖 teacher overlap，不能独立解释 value/risk/memory tradeoff。
```

---

# 6. v9.5.8 实验总览

v9.5.8 分两条并行主线。

第一条是 existing-action legal rank path：

```text
canonical AP0 existing actions
-> legal rank stability audit
-> group-stable ranker
-> rank-safe certificate
-> minimal geometry controller
-> selected runtime
```

第二条是 generated-action memory-safe primitive path：

```text
APGM failure autopsy
-> anchor-preserving / memory-nullspace / cover-preserving APGA generator
-> APGA branch-horizon outcomes
-> geometry damage matrix
-> APGA certificate
```

两条线共享同一套 outcome / geometry / memory / runtime metrics，不允许 dataset-specific threshold。

---

# 7. P0：v9.5.7 boundary reproduction

## 目标

确认 v9.5.8 没有跳过 v9.5.7 的真实边界。

## 需要读取的 artifacts

```text
v9570 route_decision_v9570.json
p1_field_legality_legacy_rank_forensic.csv
p3_legal_causal_geometry_feature_factory_v1.csv
p5_legal_rank_upper_bound.csv
p6_legacy_to_legal_distillation_sandbox.csv
p7_memory_blocker_anatomy.csv
p9_apgm_geometry_pass.csv
p10_rank_based_certificate_v5.csv
p16_base_acc_sentinel.csv
contract_audit.csv
no_fake_audit.csv
```

## 记录指标

```text
source_route_v9570
field_legality_ledger_pass
legacy_rank_leakage_explained
legal_best_TopK64_GradeAB_precision
best_legal_ranker_TopK87_GradeAB_precision
best_legal_ranker_V_lcb
best_legal_ranker_longrisk_ucb
LDO_drop_max
LSO_drop_max
memory_failure_explained_fraction
best_APGM_V_lcb
best_APGM_longrisk_ucb
best_certificate_heldout_V_lcb
best_certificate_heldout_longrisk_ucb
system_legal_controller_pass
```

## 通过标准

```text
P0 pass if:
  v9.5.7 route reproduced；
  field legality ledger pass reproduced；
  legacy outcome-derived leakage remains explained；
  APGM generated frontier fail reproduced；
  no-fake / no-proxy / no CPU offload remain pass。
```

## 可视化

```text
v9.5.7 gate waterfall:
  legality -> legal feature -> rank UB -> distillation -> memory -> APGM -> certificate -> system

TopK87 risk bars:
  GradeAB precision, V LCB, longrisk UCB, bad UCB, memory UCB, LDO/LSO drop
```

---

# 8. P1：field legality ledger v2

## 目标

把所有字段分为三类，避免下一轮再出现 GCERT18 / GCERT25 那种 outcome-derived leakage。

## 字段分类

### Green fields

训练当下可用，允许 official：

```text
step
seed id only for split, not feature
action family id if generated by protocol, not dataset id
payload norm / linf / entropy
AdamW cosine / conflict
per-example gradient mean / variance
batch agreement
old-family memory probe before apply
cover statistics before apply
basis effective rank before apply
estimated feature cost
payload apply cost estimate
```

### Yellow fields

只能 diagnostic，不能直接 official：

```text
ranker trained by teacher outcome rank
calibration split aggregate statistics
cross-fitted meta-score
leaveout diagnostic score
```

### Red fields

禁止 official：

```text
V_integrated
risk_score from future outcomes
GradeAB label
longrisk label
bad/null label
horizon outcomes
old v9.3.5 table fields
validation/test metrics
future branch deltas
```

## 记录指标

```text
field_count_total
field_count_green
field_count_yellow
field_count_red
score_variant_count
red_field_used_by_each_score
yellow_field_used_by_each_score
green_only_score_available
leakage_violation_count
```

## 通过标准

```text
P1 pass if:
  every official candidate score uses only green fields；
  every diagnostic score is explicitly marked diagnostic；
  red_field_used_by_official_score = 0；
  yellow_field_promoted_to_official = 0。
```

---

# 9. P2：Grade / target density audit v2

## 目标

不要再把单一 GradeB / T5 / MemorySafeGradeB 当作唯一目标。重新建立多等级 target，让 accepted region 可以在 density 与 safety 之间有明确解释。

## 多等级定义

每个 action 记录：

```text
V20_lcb
V80_lcb
V240_lcb
V_integrated_lcb
h240_longrisk_ucb
bad_ucb
null_ucb
memory_fail_ucb
cover_collapse_ucb
cost_q90
```

定义：

```text
GradeA:
  V20_lcb > 0
  V80_lcb > 0
  V240_lcb > 0
  longrisk_ucb <= 0.05
  bad_ucb <= 0.05
  null_ucb <= 0.15
  memory_fail_ucb <= 0.05

GradeB:
  V_integrated_lcb > 0
  V80_lcb >= -0.05
  longrisk_ucb <= 0.05
  bad_ucb <= 0.05
  null_ucb <= 0.15
  memory_fail_ucb <= 0.10

GradeC:
  V_integrated_lcb > 0
  longrisk_ucb <= 0.10
  bad_ucb <= 0.08
  memory_fail_ucb <= 0.15

Reject:
  longrisk_ucb > 0.20
  or V_integrated_lcb <= 0
  or memory_fail_ucb > 0.20
```

## 记录指标

```text
GradeA_count / coverage
GradeB_count / coverage
GradeC_count / coverage
GradeAB_count / coverage
MemorySafeValuePositive_count
TopK_min_for_coverage_0.03 = ceil(0.03 * action_count)
per-dataset GradeAB count
per-family GradeAB count
per-stratum GradeAB count
per-step-bucket GradeAB count
```

## 通过标准

```text
P2 pass if:
  GradeAB_count >= 87；
  GradeAB has no dataset/family monopoly；
  max family share <= 0.25；
  min non-empty dataset GradeAB count >= 10；
  target density stable across seeds.
```

如果 GradeAB count 不足，但 GradeC count 足够，则 v9.5.8 允许 GradeC 作为 diagnostic controller target，但不能 official。

## 可视化

```text
Grade density bar chart
GradeAB by dataset / family / stratum heatmap
V_integrated vs longrisk scatter
memory_fail vs longrisk scatter
coverage-threshold curve
```

---

# 10. P3：legal rank drop autopsy

## 目标

解释 P5 中 LDO/LSO drop = 0.8385 的来源。这个阶段不训练新 controller，只查 ranker 为什么离开某个 dataset / family / stratum 后会崩。

## 需要测的切片

```text
dataset
seed
family
signal stratum
step bucket
score bucket
payload norm bucket
memory fail bucket
cover entropy bucket
old-family drift bucket
hard-tail fraction bucket
AdamW conflict bucket
```

## 记录指标

```text
ranker_id
fold_id
heldout_group_type
heldout_group_id
TopK64_GradeAB_precision
TopK87_GradeAB_precision
TopK87_V_lcb
TopK87_longrisk_ucb
TopK87_bad_ucb
TopK87_memory_ucb
accepted_overlap_with_full_rank
kendall_tau_train_vs_heldout
score_distribution_shift_KS
group_mutual_information_score
failure_reason
```

## 通过标准

```text
P3 pass if:
  dominant LDO/LSO drop cause is identified；
  every failed heldout group has a tagged failure reason；
  group shift metrics are logged；
  no hidden dataset-name branch is used.
```

## Failure taxonomy

```text
D1-target-density-too-low-in-heldout-group
D2-score-distribution-shift
D3-memory-risk-unseen-in-calibration
D4-value-signal-overfits-family
D5-cover-collapse-not-represented-in-rank
D6-distillation-teacher-pattern-local
D7-action-family-spurious-correlation
D8-statistical-CI-too-wide
```

---

# 11. P4：legal causal geometry feature factory v2

## 目标

不再只构造单个 AUC 很高但 TopK 有风险的 feature，而是构造多头 feature：value、memory、long-risk、cover、cost 各自独立，再用安全规则组合。

## Feature groups

### F1：value direction

```text
ControlTransferImprovement
OffDiagonalAgreement
LeaveOneOutBatchTransfer
AdamWParallelDeltaGap
LinearizedCEGain
LinearizedMarginGain
```

### F2：memory safety

```text
OldFamilyCEProbeDelta
OldFamilyMarginProbeDelta
OldStratumLogitDrift
MemoryGradientConflict
MemoryAnchorCosine
OldFamilySupportCount
```

### F3：cover stability

```text
BasisEffectiveRankDeltaProxy
HardTailCoverEntropyDeltaProxy
ActiveEdgeSupportChange
BasisRoleEntropy
LocalCoverOverlapBeforeAfterProbe
```

### F4：curvature / fixed-point stability

```text
DirectionalJacobianNormProxy
CEp99DirectionalDelta
LocalLipschitzProxy
SymmetricBoundaryResidual
StepwiseFixedPointResidualDelta
```

### F5：cost

```text
feature_compute_ms_q90
payload_apply_ms_q90
extra_kernel_count
memory_ratio_est
```

## 记录指标

每个 feature 记录：

```text
AUC_GradeAB
AUC_longrisk
AUC_memoryfail
TopK64_GradeAB_precision
TopK87_GradeAB_precision
TopK87_V_lcb
TopK87_longrisk_ucb
TopK87_bad_ucb
TopK87_memory_ucb
TopK87_null_ucb
feature_cost_ms_q90
feature_legality_class
monotone_sign_consistency
```

## 通过标准

```text
A single feature is not required to pass.
P4 pass if at least one value feature has:
  TopK87_V_lcb > 0
and at least one independent risk/memory feature can reduce:
  longrisk_ucb <= 0.05
  memory_ucb <= 0.10
without reducing coverage below 0.03.
```

---

# 12. P5：group-stable legal ranker

## 目标

把 P5/P6 的 diagnostic rank 变成 group-stable legal rank。这里允许 small MLP diagnostic，但 official candidate 必须经过 cross-fit + group-drop gate。

## Ranker candidates

```text
R0: monotone additive score
R1: value rank + memory veto
R2: value rank + longrisk veto
R3: value rank + memory + cover veto
R4: group-DRO small MLP ranker
R5: pairwise ranking loss with group-balanced sampling
R6: conformal topK set with group-wise calibration
R7: distillation student from legacy teacher, green fields only, group-DRO fine-tune
R8: negative control shuffled labels
```

## Score form for monotone official candidate

$$
Score(a)=
\alpha_1 ValueScore(a)
-\alpha_2 LongRiskScore(a)
-\alpha_3 MemoryRisk(a)
-\alpha_4 CoverCollapse(a)
-\alpha_5 Cost(a)
$$

with:

$$
\alpha_i \ge 0.
$$

## 记录指标

```text
ranker_id
feature_set_id
green_field_only
training_split_id
calibration_split_id
heldout_split_id
TopK64 precision / V / risk
TopK87 precision / V / risk
TopK128 precision / V / risk
LDO drop
LSO drop
family-out drop
seed-out drop
accepted_overlap_stability
score_shift_KS
negative_control_precision
```

## Official weak pass

```text
TopK87_GradeAB_precision >= 0.75
TopK87_V_integrated_LCB > 0
TopK87_longrisk_UCB <= 0.05
TopK87_bad_UCB <= 0.05
TopK87_null_UCB <= 0.15
TopK87_memory_fail_UCB <= 0.10
coverage >= 0.03
LDO_drop_max <= 0.25
LSO_drop_max <= 0.25
negative_control_pass = 0
```

## Official strong pass

```text
TopK87_GradeAB_precision >= 0.85
TopK87_V_integrated_LCB > 0.10
TopK87_longrisk_UCB <= 0.03
TopK87_bad_UCB <= 0.03
LDO_drop_max <= 0.15
LSO_drop_max <= 0.15
```

## 可视化

```text
TopK curve: K=32,64,87,128,256
risk-value Pareto front
LDO/LSO drop matrix
accepted overlap heatmap
feature contribution waterfall
negative control comparison
```

---

# 13. P6：legacy-to-legal distillation recheck

## 目标

v9.5.7 的 P6 是重要线索，但它来自 legacy teacher sandbox。v9.5.8 要确认：distillation 是不是只是学了 teacher artifact，还是学到了 green-field ranking structure。

## 实验设计

```text
Teacher A: legacy_GCERT18_outcome_diagnostic
Teacher B: UB-M1-small-mlp-diagnostic-ranker
Teacher C: oracle GradeAB rank diagnostic
Teacher D: shuffled teacher negative control
Student 1: green-field linear monotone
Student 2: green-field shallow MLP
Student 3: group-DRO shallow MLP
Student 4: pairwise rank student
```

## 记录指标

```text
teacher_id
student_id
student_red_field_count
student_teacher_kendall_tau
student_topK87_overlap
student_true_GradeAB_precision
student_true_V_lcb
student_true_longrisk_ucb
student_true_bad_ucb
student_LDO_drop
student_LSO_drop
student_negative_control_precision
```

## 通过标准

```text
P6 pass if:
  student_red_field_count = 0；
  true TopK87 GradeAB precision >= 0.75；
  V_lcb > 0；
  longrisk_ucb <= 0.05；
  bad_ucb <= 0.05；
  LDO/LSO drop <= 0.25；
  shuffled teacher fails。
```

如果 P6 pass 但 P5 fail，则 route 进入 `R6-DistilledLegalRankPromisingButNotMinimal`，不能直接 official。

---

# 14. P7：memory blocker anatomy v2

## 目标

把 “memory explains long-risk” 从统计相关推进到可执行生成约束。

## 记录对象

对 canonical AP0、APGM generated、P5 accepted topK、P6 student topK 分别记录：

```text
old_family_CE_delta
old_family_margin_delta
old_family_logit_cosine
old_stratum_CE_delta
old_stratum_margin_delta
memory_gradient_conflict
old_family_support_count
cover_entropy_delta
basis_rank_delta
h240_longrisk
V_integrated
GradeAB
```

## 核心指标

```text
P(longrisk | memory_fail)
P(memory_fail | longrisk)
P(GradeAB | memory_safe)
P(GradeAB | memory_fail)
memory_safe_value_positive_count
memory_safe_value_positive_density
memory_risk_AUC_longrisk
memory_veto_lift
```

## 通过标准

```text
P7 pass if:
  memory failure explains >= 0.70 of high-longrisk；
  memory-safe subset contains >= 87 value-positive actions or a clear generator target；
  memory veto can reduce TopK87 longrisk UCB below 0.05 without killing V_lcb.
```

---

# 15. P8：APGM failure autopsy

## 目标

不要继续猜 APGM 为什么失败。直接比较 source / generated 的 value、memory、cover、risk damage。

## 记录指标

```text
primitive_id
source_action_id
generated_action_id
source_GradeAB
generated_GradeAB
source_V_integrated
generated_V_integrated
Damage_V = generated - source
source_longrisk
generated_longrisk
longrisk_created
source_memory_fail
generated_memory_fail
memory_fail_created
source_cover_entropy
generated_cover_entropy
cover_collapse_created
source_basis_rank
generated_basis_rank
basis_rank_collapse_created
```

## Failure classes

```text
G1-value-destroyed
G2-longrisk-created
G3-memory-fail-created
G4-cover-collapse-created
G5-basis-rank-collapse-created
G6-AdamW-conflict-created
G7-no-op-like-generated
G8-negative-control-like
```

## 通过标准

```text
P8 pass if:
  >= 90% APGM failures assigned to one or more failure classes；
  dominant failure mode identified；
  APGA design constraints are derived from P8 evidence.
```

---

# 16. P9：APGA1-APGA8 memory-safe anchor primitive

## 目标

新生成器不能再只是合法 payload builder。它必须显式保留旧 family / old stratum，并只在 value signal 明确的位置做小更新。

## APGA primitives

```text
APGA1-RankAnchoredTemplateBlend
  使用 P5/P6 legal topK existing actions 作为 anchor，生成小幅 template blend。

APGA2-MemoryNullspaceProjection
  将 update 投影到 old-family gradient conflict 的近似 nullspace。

APGA3-OffDiagonalAgreementMaximizer
  按 population-risk off-diagonal agreement 生成方向，但加入 memory veto。

APGA4-CoverPreservingEdgeUpdate
  限制 basis effective rank / cover entropy 下降。

APGA5-SymmetricBoundaryResidualUpdate
  对称约束 h20 gain 与 h240 risk，避免单向猛拉。

APGA6-AdamWCompatibleResidual
  只允许和 AdamW 主方向不冲突的小 residual。

APGA7-HorizonStagedConservativeUpdate
  先保证 h20 V，再用 h80/h240 risk proxy 剪裁。

APGA8-NegativeControlShuffledAnchor
  shape-preserving shuffled payload negative control。
```

## 生成数量

```text
primitive_count = 8
actions_per_primitive = 64
expected_generated_actions = 512
```

## 记录指标

```text
payload_hash_missing_count
certificate_hash_missing_count
action_apply_linf_max
payload_norm_distribution
memory_veto_rate
cover_veto_rate
value_signal_score_distribution
negative_control_generated
```

## 通过标准

```text
APGA implementation pass if:
  generated actions = 512；
  payload/certificate hash missing = 0；
  action apply L∞ max <= 1e-7；
  negative control generated = 1。
```

---

# 17. P10：APGA branch-horizon outcome materializer

## 目标

真实测 APGA 生成动作，而不是根据 generator 内部分数猜。

## 分支

```text
RealAPGA
AdamWOnly
AdamWParallel
BestLR
NoOp
RandomPayload
ShuffledAPGAPayload
CertificatePassNoPayload
```

## horizons

```text
h = 20, 80, 240
```

## 记录指标

```text
expected_rows
actual_rows
branch_completion_rate
horizon_completion_rate
secondary_delta_completion_rate
unresolved_exception_count
quality_audit_pass
rows_per_sec
wallclock_sec
```

## 通过标准

```text
actual_rows = expected_rows
unresolved_exception_count = 0
label_exclusivity_violation_count = 0
metric_nan_inf_count = 0
quality_audit_pass = 1
```

---

# 18. P11：APGA outcome / geometry pass

## 目标

判断 APGA 是否真的产生可用 frontier。

## 记录指标

对每个 primitive：

```text
GradeA precision
GradeB precision
GradeAB precision
V20_lcb
V80_lcb
V240_lcb
V_integrated_lcb
h240_longrisk_ucb
bad_ucb
null_ucb
memory_fail_ucb
cover_collapse_ucb
new_positive_created_rate
longrisk_created_rate
memory_fail_created_rate
negative_control_pass
```

## Weak pass

```text
GradeAB precision >= 0.50
V_integrated_lcb > 0
h240_longrisk_ucb <= 0.10
bad_ucb <= 0.08
memory_fail_ucb <= 0.15
negative_control_pass = 0
```

## Official candidate pass

```text
accepted_count >= 87
GradeAB precision >= 0.75
V_integrated_lcb > 0
h240_longrisk_ucb <= 0.05
bad_ucb <= 0.05
null_ucb <= 0.15
memory_fail_ucb <= 0.10
coverage >= 0.03
```

If no APGA primitive passes weak gate, route must stop at `R4-GeneratedPrimitiveStillFails` and no controller/runtime should open.

---

# 19. P12：rank-safe certificate v6

## 目标

不再追“概率校准好看”，而是追 accepted region 安全。

## Certificate candidates

```text
CERT6-1-MonotoneValueRiskMemoryScore
CERT6-2-GroupConformalRankBound
CERT6-3-TwoStageValueRankMemoryVeto
CERT6-4-StudentRankWithMemoryConformalSet
CERT6-5-APGAInternalCertificate
CERT6-6-ExistingAndGeneratedUnifiedRank
CERT6-7-NegativeControlCertificate
```

## 记录指标

```text
certificate_id
green_field_only
yellow_field_count
red_field_count
TopK64_GradeAB_precision
TopK87_GradeAB_precision
heldout_accepted
heldout_coverage
heldout_V_lcb
heldout_longrisk_ucb
heldout_bad_ucb
heldout_null_ucb
heldout_memory_ucb
LDO_drop
LSO_drop
ECE
TopK_calibration_gap
```

## 通过标准

```text
certificate official pass if:
  red_field_count = 0；
  heldout_accepted >= 87；
  heldout_coverage >= 0.03；
  heldout_V_lcb > 0；
  heldout_longrisk_ucb <= 0.05；
  heldout_bad_ucb <= 0.05；
  heldout_null_ucb <= 0.15；
  heldout_memory_ucb <= 0.10；
  LDO/LSO drop <= 0.25；
  negative control fails。
```

---

# 20. P13：minimal geometry controller

## 目标

把 P5/P6/P12 的 rank-safe signal 转成一个真正的 minimal controller。

## Controller form

$$
Accept(a)=1
$$

当且仅当：

$$
RankScore(a)\ge q_{cal}
$$

$$
MemoryRisk(a)\le \tau_M
$$

$$
LongRisk(a)\le \tau_L
$$

$$
Cost(a)\le C_{max}
$$

且所有阈值只在 calibration split 上冻结。

## 记录指标

```text
controller_id
feature_count
thresholds
calibration accepted / precision / risk / V
heldout accepted / precision / risk / V
coverage
LDO / LSO / family-out
negative controls
```

## Pass condition

```text
heldout accepted >= 87
coverage in [0.03, 0.15]
GradeAB precision >= 0.75
V_integrated_lcb > 0
longrisk_ucb <= 0.05
bad_ucb <= 0.05
null_ucb <= 0.15
memory_fail_ucb <= 0.10
LDO/LSO drop <= 0.25
feature_count <= 8
red_field_count = 0
```

---

# 21. P14：selected runtime

## 目标

只有 P13 pass 后才测 selected runtime。不能用 preflight runtime 或 no-payload runtime 替代。

## 记录指标

```text
selected_controller_id
selected_action_count
feature_compute_ms_q50/q90
rank_score_ms_q50/q90
certificate_ms_q50/q90
payload_apply_ms_q50/q90
extra_kernel_count
extra_sync_count
step_ratio_q50/q90
memory_ratio
controller_launches_per_active_step_q90
zero_candidate_launch_count
```

## Pass condition

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
zero_candidate_launch_count = 0
no CPU offload
no proxy runtime
```

---

# 22. P15：leaveout / paired replay boundary

## 目标

只有 P13/P14 同时 pass 后，才打开 causal evidence。

## Leaveout tests

```text
leave-dataset-out
leave-stratum-out
leave-family-out
leave-seed-out
```

## Paired replay branches

```text
RealFunctionalSelected
AdamWOnly
AdamWParallel
BestLR
NoOp
RandomPayload
ShuffledPayload
CertificatePassNoPayload
```

## 记录指标

```text
paired_delta_acc
paired_delta_CE
paired_delta_NLL
paired_delta_ECE
paired_delta_CEp99
paired_delta_margin_p10
paired_delta_forgetting
paired_delta_hard_tail
beats_AdamWParallel_rate
beats_BestLR_rate
beats_NoOp_rate
shuffle_control_fail_rate
```

## Pass condition

```text
RealFunctionalSelected beats AdamWParallel and BestLR on V_integrated;
shuffle controls fail;
no dataset-specific collapse;
bad/null/longrisk remain within gate;
runtime remains within selected-runtime envelope.
```

---

# 23. P16：short/full training boundary

## 目标

仍然不打榜，只做功能性边界验证。

## Datasets

```text
MNIST
Fashion-MNIST
KMNIST
```

只作为 fixed protocol diagnostics，不允许 dataset-specific controller。

## Models

```text
LQ-t2-h256 base
MatchedMLP
AdamWStrongLRGridMLP
QuadraticFeatureMLP
DG-KAN selected functional controller
```

## 指标

```text
test_acc
val_loss
NLL
ECE
CEp99
margin_p10
hard-stratum acc
steps_to_target
time_to_target
sample_efficiency
forgetting_old_family
runtime_step_ratio
memory_ratio
```

## Pass condition

```text
DG-KAN selected controller improves at least one of:
  sample efficiency;
  calibration;
  CEp99 / hard-tail;
  forgetting;
  robustness;
  final test acc;
without losing runtime envelope.
```

---

# 24. 可视化总清单

v9.5.8 必须输出这些图：

```text
1. Gate waterfall from v9.5.7 to v9.5.8.
2. Field legality Sankey: green / yellow / red fields into each score.
3. TopK curve for legal rankers: K=32,64,87,128,256.
4. Value-risk Pareto scatter: V_integrated LCB vs h240 longrisk UCB.
5. Memory-risk scatter: memory_fail score vs h240 longrisk.
6. LDO/LSO drop heatmap by ranker.
7. Accepted overlap heatmap across folds.
8. Feature contribution waterfall for selected ranker.
9. APGM damage matrix.
10. APGA source-to-generated damage matrix.
11. Cover entropy / basis rank change histograms.
12. Runtime stacked bar: feature / rank / certificate / payload apply / base step.
13. Paired replay branch comparison if opened.
14. Base-Acc Sentinel continuation chart.
```

---

# 25. 并行执行计划

为了避免继续“一轮只清一个 blocker”，v9.5.8 必须并行跑：

```text
Batch A:
  P0/P1/P2 boundary + legality + target density.

Batch B:
  P3/P4/P5 legal rank drop autopsy + feature factory + group-stable rank.

Batch C:
  P6 distillation recheck.

Batch D:
  P7/P8 memory + APGM failure autopsy.

Batch E:
  P9/P10/P11 APGA generation + outcome.

Batch F:
  P12/P13 certificate/controller, conditional on P5 or P11.

Batch G:
  P14/P15/P16 runtime/paired/short-full, conditional only after controller pass.
```

Preflight rule:

```text
任何新 primitive 必须先跑：
  1-action preflight；
  3-action preflight；
  16-action preflight；
  negative control divergence；
再跑 512-action smoke。
```

---

# 26. Route decision

v9.5.8 最终 route 按以下顺序判断：

```text
R0-BoundaryRegression
  v9.5.7 boundary 或 canonical artifacts 回退。

R1-LegalRankStillGroupUnstable
  legal rank TopK 强，但 LDO/LSO drop 仍过大。

R2-LegalRankAcceptedRegionPass
  existing-action legal rank + certificate 过 controller gate。

R3-DistilledRankPromisingButNotOfficial
  distillation strong，但不能 official。

R4-APGAGeneratedFrontierFail
  APGA implementation/outcome 完整，但 generated frontier 不过。

R5-APGAGeneratedFrontierPass
  APGA 生成动作过 weak/official frontier。

R6-CertificateAcceptedRegionFail
  rank/candidate 有信号，但 certificate heldout region 不安全。

R7-SelectedRuntimeFail
  controller pass，但 runtime 不过。

R8-SystemLegalControllerPass
  controller + selected runtime 同时过线。

R9-PairedReplayFail
  system pass 后 paired replay 不过。

R10-StrictPureKANFunctionalLocalPass
  system + leaveout + paired replay 全过，但 short/full 未完成。

R11-ExternalReadyCandidate
  short/full、robustness、sample efficiency、strong baseline 全部完成。
```

---

# 27. 成功标准

v9.5.8 的最低成功不是 full DG-KAN success，而是以下任一项：

```text
A. group-stable legal rank controller pass；
B. APGA generated frontier pass；
C. 明确证明 existing-action selection 与 APGA generation 都失败，并把失败归因到一个可执行的新机制。
```

v9.5.8 的强成功是：

```text
system_legal_controller_pass = 1
selected_runtime_pass = 1
heldout accepted >= 87
coverage >= 0.03
V_integrated LCB > 0
longrisk UCB <= 0.05
bad UCB <= 0.05
memory UCB <= 0.10
LDO/LSO drop <= 0.25
step_ratio_q90 <= 1.50
```

---

# 28. 最终判断

v9.5.8 的核心不是再造一个名字更复杂的 primitive，而是回答：

$$
\boxed{
\text{legal rank 信号能不能变成 group-stable controller？如果不能，memory-safe generator 能不能直接造出好动作？}
}
$$

如果 existing-action legal rank 可以过 group-stable gate，优先推进 controller/runtime。  
如果 legal rank 仍然 LDO/LSO 崩，而 APGA 能生成好动作，路线转向 self-generating primitive。  
如果两者都失败，就必须承认当前 LQ/AP0 action family 的 functional update 机制仍然不够，需要回到更底层的 KAN geometry primitive 设计，而不是继续调证书阈值。
