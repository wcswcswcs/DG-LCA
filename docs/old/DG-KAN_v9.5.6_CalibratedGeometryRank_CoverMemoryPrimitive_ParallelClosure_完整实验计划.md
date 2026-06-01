# DG-KAN v9.5.6 Calibrated Geometry Rank / Cover-Memory Preserving Primitive / Parallel Closure 完整实验计划

> 本计划基于 v9.5.5 `Trainable Geometry / Signal-Reservoir Primitive / Parallel Closure` 的真实结果制定。  
> v9.5.6 不继续微调 APGR、GCERT18 threshold、T5 阈值或单一 AUC 指标，而是围绕一个更直接的问题设计实验：
>
> $$
> \boxed{
> \text{能否把“训练形状变好”的少量动作，转成训练当下可稳定选择、可校准、可复现、可部署的 functional update？}
> }
> $$

---

# 0. 执行摘要

v9.5.5 的结果不是 functional success，也不是 controller success。它的真实推进是：

```text
1. Trainable Geometry Ledger v2 已经完整落盘；
2. T5-like clean geometry target 真实存在，但数量不够；
3. 多等级几何标签已经建立；
4. signal / reservoir / cover / curvature / memory audit 已并行执行；
5. APGS 的失败主因被定位为 horizon long-risk；
6. APGR1-APGR8 已真实实现并完成 branch-horizon smoke；
7. APGR 仍没有产生 value-positive / horizon-safe / GradeB-positive frontier；
8. GCERT18 出现了很强的 TopK diagnostic signal；
9. 但 GCERT18 calibration / ECE 不过，不能 official controller；
10. source controller、selected runtime、paired replay、short/full training 都没有打开。
```

v9.5.5 的核心数据：

```text
route = R2a-CleanGeometryTargetTooSparse
ledger_rows = 4412
canonical_AP0_rows = 2876
generated_APY/APG/APGS_rows = 512 / 512 / 512

T5_action_count = 57
T5_coverage = 0.019819193324061197
T5_V_integrated_LCB = 0.1960342568901401
T5_h240_longrisk/bad/null = 0.0 / 0.0 / 0.0

GradeA/B/C/D/E_count = 40 / 39 / 18 / 89 / 2690

best_signal = SNR-old-family
best_signal_AUC_GradeB = 0.9302100351642583
best_signal_TopK64_GradeB_precision = 0.296875
best_signal_TopK64_V_integrated_LCB = -0.48271869313319804
best_signal_TopK64_h240_longrisk = 0.390625

cover_collapse_count = 92
basis_effective_rank_delta_mean = -0.5057796221440767
hard_tail_cover_entropy_delta_mean = -9.90125173852573

curvature_delta_UCB = 0.0655680861250561
jacobian_spectral_delta_UCB = 0.0655680861250561
CEp99_delta_UCB = -0.1201715130761731

old_family_fail_count = 4249
old_stratum_fail_count = 1216
forget_risk_UCB = 0.25856111245831553

best_APGR = APGR5-MemoryGuardedResidualUpdate
best_APGR_GradeB_precision = 0.046875
best_APGR_OfficialGeo_precision = 0.0
best_APGR_V_integrated_LCB = -0.8658484647235336
best_APGR_h240_longrisk = 0.78125

best_certificate = GCERT18-TransferRiskBalancedCertificate
GCERT18_AUC_GradeB = 0.992657726818137
GCERT18_TopK64_GradeB_precision = 0.890625
GCERT18_TopK64_V_integrated_LCB = 0.1650359732307112
GCERT18_TopK64_h240_longrisk/bad/null = 0.0 / 0.0 / 0.0
GCERT18_ECE = 0.6217805447724831
geometry_certificate_v3_pass = 0

source_controller/selected_runtime/system = 0 / 0 / 0
```

我的独立判断是：

$$
\boxed{
\text{v9.5.5 的最大新线索不是 APGR，而是 GCERT18 的 TopK 区域。}
}
$$

报告把 primary blocker 写成 `clean_geometry_target_too_sparse` 是合理的，但不完整。因为 v9.5.5 同时暴露出另一个更重要的问题：

$$
\boxed{
\text{我们可能已有很强的“排序型几何信号”，但它不是校准型概率证书。}
}
$$

因此 v9.5.6 的核心不是继续生成更多 APGR 小变体，而是同时解决三个问题：

```text
1. 几何标签的 universe / scope 是否一致；
2. GCERT18 的 TopK 信号能否变成 calibration-frozen accepted region；
3. 新 generator 是否能直接保护 cover / memory / signal，而不是只生成合法 payload。
```

---

# 1. 对 v9.5.5 的独立判断

## 1.1 有进展，但不是能力进展

v9.5.5 的进展主要在实验理解层：

```text
1. Trainable Geometry Ledger v2 比 v9.5.4 更完整；
2. 多等级标签比单一 T5 / GGA 更适合描述动作质量；
3. signal / reservoir / cover / curvature / memory 被拆开审计；
4. APGR implementation 和 branch-horizon materializer 完整闭合；
5. certificate 的强 TopK 诊断信号首次非常明显。
```

但它仍不是 functional success：

```text
1. official target candidate count = 0；
2. raw signal upper-bound pass = 0；
3. reservoir/noise rejection pass = 0；
4. cover stability pass = 0；
5. memory anti-forgetting pass = 0；
6. APGR weak / strong pass = 0；
7. geometry certificate v3 pass = 0；
8. source controller / selected runtime / system pass = 0。
```

## 1.2 T5 不是没有意义，而是数量不足

T5 的质量很干净：

```text
T5 count = 57
V_integrated_LCB = 0.1960342568901401
h240 longrisk = 0.0
bad = 0.0
null = 0.0
```

如果 official coverage 下限仍是：

$$
Coverage \ge 0.03,
$$

在 2876 个 canonical AP0 actions 中，至少需要：

$$
N_{min}=\lceil 0.03 \times 2876\rceil=87.
$$

T5 只有 57 个，缺口是：

$$
87-57=30.
$$

这不是“完全没有好动作”，而是“干净动作数量不够”。

更进一步：

```text
GradeA_count = 40
GradeB_count = 39
GradeC_count = 18
GradeA + GradeB = 79
GradeA + GradeB + GradeC = 97
```

如果 GradeA/B/C 的定义彼此一致，那么：

```text
A+B 距离 87 只差 8 个；
A+B+C 已经超过 87。
```

所以 v9.5.6 不能只盯 T5。必须回答：

```text
1. GradeA/B/C 的风险差异到底是什么？
2. GradeC 中是否存在可接受的低风险子集？
3. A+B 是否可以在 heldout / leaveout 下过线？
4. GCERT18 TopK64 的 GradeB count 是否与 P3 grade count 使用同一 universe？
```

## 1.3 v9.5.5 最大的异常：GCERT18 TopK 很强，但 ECE 很差

GCERT18 的结果很特别：

```text
AUC_GradeB = 0.992657726818137
TopK64_GradeB_precision = 0.890625
TopK64_V_integrated_LCB = 0.1650359732307112
TopK64_h240_longrisk = 0.0
TopK64_bad = 0.0
TopK64_null = 0.0
ECE = 0.6217805447724831
certificate pass = 0
```

这说明它不是普通失败。它在 TopK 选出来的区域非常干净，但它给出的概率分数不可信。

因此它更像：

```text
ranking certificate
```

而不是：

```text
calibrated probability certificate
```

v9.5.6 需要把两者分开：

```text
ranking 是否可用？
calibration 是否可修？
accepted region 是否稳定？
leaveout 是否保持？
```

不能因为 ECE 不过就丢掉 GCERT18；也不能因为 TopK 很强就直接 official。

## 1.4 需要立刻审计 GradeB universe 是否一致

v9.5.5 中有一个必须检查的口径问题：

```text
P3 GradeB_count = 39
GCERT18 TopK64 GradeB precision = 0.890625
```

如果这两个数字来自同一个 action universe，那么 TopK64 里大约有：

$$
64 \times 0.890625 = 57
$$

个 GradeB，这大于总 GradeB count 39，逻辑上不可能。

因此存在三种可能：

```text
1. P3 GradeB_count 只统计 canonical AP0，而 GCERT18 TopK 统计 ledger 全量或某个 split；
2. GCERT18 的 GradeB target 是 GradeA/B combined 或 GradeB-like，不是 P3 的 GradeB；
3. artifact 汇总口径存在 bug。
```

这不是小问题。v9.5.6 的 P1 必须先做 `grade_universe_scope_audit`。在这个 audit 通过前，GCERT18 不能 official。

## 1.5 APGR 工程闭合，但科学失败

APGR1-APGR8 的 implementation 是进展：

```text
generated actions = 512
payload/certificate hash missing = 0 / 0
action apply L∞ max = 0.0
branch-horizon rows = 12288 / 12288
quality audit pass = 1
```

但 APGR outcome 很差：

```text
best APGR5 GradeB precision = 0.046875
OfficialGeo precision = 0.0
V_integrated_LCB = -0.8658484647235336
h240 longrisk = 0.78125
```

这说明 APGR 仍然是“合法动作生成器”，不是“好训练形状生成器”。

## 1.6 cover / memory 是当前几何失败的硬证据

v9.5.5 有一个很重要的分解：

```text
curvature/fixed-point audit pass；
cover stability fail；
memory anti-forgetting fail。
```

这意味着不能说“几何全坏”。更准确是：

```text
曲率没有明显爆；
但是局部 cover 在塌；
旧 family / old stratum 在忘。
```

因此下一轮 generator 不应该再只加 curvature trust region，而要优先保护：

```text
1. basis effective rank；
2. hard-tail cover entropy；
3. old-family / old-stratum performance；
4. signal direction 与 old-family compatibility。
```

---

# 2. 当前阶段判断

## 2.1 现在不是卡在 acc

Base-Acc Sentinel 仍然健康：

```text
rows = 120
LQ mean test acc = 0.6537760416666667
AdamWStrongLRGridMLP = 0.628515625
base_acc_used_for_controller = 0
```

这只能说明 LQ-t2-h256 base 没有 catastrophic fail，不能说明 functional update 已经成功。

当前不是：

```text
KAN 底座完全不能训练；
manual training 不可信；
canonical outcome table 不可信；
branch-horizon materializer 不可信；
payload apply 不可信。
```

当前是：

```text
functional update 的“好训练形状”太少；
生成器造不出足够多这种动作；
证书虽能排序，但校准不够；
controller 未选中；
runtime / paired replay / short-full 没有打开。
```

## 2.2 进度为什么感觉慢

你感觉慢是合理的。最近几轮没有进入“模型能力上升”的正反馈，而是在做：

```text
定义什么是好动作；
定义什么是好训练形状；
修 canonical truth；
排除旧 table；
测试 APX/APY/APG/APGS/APGR；
审计 signal / cover / curvature / memory。
```

这不是没有进度，但还不是最终能力。

v9.5.5 比 v9.5.4 多出来的关键东西是：

```text
1. T5-like clean region 被确认；
2. multi-grade label 可以作为更细的动作质量层级；
3. cover 和 memory 两个具体几何失败点被抓出来；
4. GCERT18 给出强 TopK diagnostic；
5. APGR 被实验证伪为当前形式不够。
```

所以不是“没路”，而是现在必须从“继续造 APGR”转向“用 GCERT18/Grade/cover-memory 信息重构 controller 和 generator”。

---

# 3. v9.5.6 总体目标

v9.5.6 的总体目标是：

$$
\boxed{
\text{把 v9.5.5 的强 ranking diagnostic 转成可校准、可 leaveout、可 runtime 的几何 controller；同时重建保护 cover/memory 的 generator。}
}
$$

最低有效推进目标：

```text
1. 完成 Grade / TopK / certificate target universe scope audit；
2. 判断 GCERT18 是真实 ranking signal，还是口径 artifact；
3. 若 GCERT18 ranking 真有效，构建 rank-only / conformal / accepted-region controller；
4. 建立 A/B/C 多等级几何 target 的 official density 上界；
5. 找到 cover collapse 与 memory fail 的直接机制；
6. 实现 APGC1-APGC8 cover-memory preserving primitive；
7. APGC branch-horizon smoke 完整落盘；
8. 若 controller selected，测 selected runtime；
9. 只有 system pass 后才打开 paired replay / short-full。
```

强目标：

$$
SystemLegalControllerPass=1
$$

$$
Coverage_{heldout}\ge 0.03
$$

$$
V_{integrated}^{LCB}>0
$$

$$
LongRisk_{240}^{UCB}\le 0.05
$$

$$
Bad^{UCB}\le 0.05
$$

$$
Null^{UCB}\le 0.15
$$

$$
StepRatio_{q90}\le 1.50
$$

---

# 4. 本轮不做什么

v9.5.6 不做：

```text
1. 不继续调 APGR5 / APGR7 小参数；
2. 不因为 GCERT18 AUC 高就直接 official；
3. 不因为 GCERT18 ECE 差就直接丢掉；
4. 不继续只看 AUC；
5. 不继续只看 T5 单一二元标签；
6. 不按 dataset 调阈值；
7. 不把 Base-Acc Sentinel 当 functional success；
8. 不在 controller 未过时打开 paired replay；
9. 不在 selected runtime 未测时声明 system pass；
10. 不新增没有假设的 payload builder。
```

允许做：

```text
1. 按 split 冻结 rank threshold；
2. 使用 conformal / Wilson LCB / accepted-region calibration；
3. 使用 GradeA/B/C 多等级 target；
4. 诊断不同 dataset / family / stratum，但不能按它们调参；
5. 并行执行 generator / certificate / runtime preflight；
6. 使用 negative controls；
7. 使用 old-family / old-stratum memory probes；
8. 使用 signal-reservoir SNR 作为局部约束，但不能单独作为 controller。
```

---

# 5. 好动作的新定义

v9.5.6 不再用单个标签定义好动作。每个动作 $a$ 记录一个向量：

$$
G(a)=
(
V_{20},
V_{80},
V_{240},
V_{integrated},
LongRisk_{240},
Bad,
Null,
SignalScore,
ReservoirLeak,
CoverRankDelta,
CoverEntropyDelta,
CurvatureDelta,
JacobianSpectralDelta,
OldFamilyFail,
OldStratumFail,
Cost
).
$$

一个动作不是“acc 涨”就好，而是至少满足：

```text
1. h20 / h80 / h240 不能互相打架；
2. integrated value 为正；
3. long-risk 低；
4. bad/null 低；
5. signal 多于 reservoir；
6. cover 不塌；
7. curvature 不爆；
8. old-family / old-stratum 不忘；
9. 训练当下能被证书或排名找到；
10. runtime 成本可接受。
```

v9.5.6 将候选动作分成 5 级：

```text
Grade A:
  value positive, no long-risk, no bad/null, cover stable, memory stable.

Grade B:
  value positive, no long-risk, no bad/null, curvature safe, mild cover/memory issue allowed.

Grade C:
  value mostly positive, risk low, but cover/memory or one horizon weak.

Grade D:
  short-term useful but horizon-risk / memory-risk visible.

Grade E:
  value negative or high long-risk / bad / null.
```

Controller 不应直接追 GradeA，因为它太稀疏；也不应接受全部 GradeC，因为可能引入风险。v9.5.6 的核心是找：

$$
AcceptRegion \subseteq A \cup B \cup C
$$

使其同时满足 density、value、risk、memory、cost。

---

# 6. 核心假设

## H1：v9.5.5 的 GCERT18 是真实 ranking signal，不是 artifact

H1 成立标准：

```text
grade universe scope audit pass = 1
GCERT18 score 不使用 outcome-derived fields
GCERT18 TopK64 precision 在 calibration / heldout / LDO / LSO 中稳定
TopK64 V_integrated_LCB > 0
TopK64 longrisk/bad/null = 0 或 UCB 过线
negative controls fail
```

H1 失败标准：

```text
TopK GradeB count 与 grade universe 不一致；
或 heldout TopK 崩；
或 GCERT18 使用了 outcome-at-commit / future feature；
或 negative control 也高。
```

## H2：GCERT18 的失败是 calibration failure，而不是 ranking failure

H2 成立标准：

```text
rank metrics pass；
accepted-region metrics pass；
ECE high 但 isotonic / temperature / conformal calibration 后 ECE <= 0.10；
calibrated threshold frozen 后 heldout pass。
```

H2 失败标准：

```text
rank pass 不能转成 accepted-region pass；
或 calibration 后 coverage / value / risk 崩。
```

## H3：T5 太稀疏，但 GradeA+B 或 GradeA+B+safe-C 可能达到 official density

H3 成立标准：

```text
A+B or A+B+safeC accepted count >= 87；
V_integrated_LCB > 0；
h240 longrisk <= 0.05；
bad <= 0.05；
null <= 0.15；
support balance pass；
LDO/LSO 不崩。
```

H3 失败标准：

```text
任何 A/B/C 组合要么 coverage 不足，要么 long-risk / bad / null / memory 失败。
```

## H4：cover collapse 和 memory fail 是 generator 失败主因

H4 成立标准：

```text
APGR/APGS generated actions 的 CoverRankDelta / CoverEntropyDelta / OldFamilyFail 与 long-risk / negative value 显著相关；
repair cover/memory constraints 后 long-risk 降低。
```

H4 失败标准：

```text
cover/memory metrics 与 long-risk/value 没有关系；
或加约束后不改善。
```

## H5：APGC cover-memory primitive 能创造新的 GradeB 或 safe-C actions

H5 成立标准：

```text
APGC generated actions >= 512
branch-horizon rows complete
best APGC GradeB precision >= 0.25
or A+B+safeC precision >= 0.40
V_integrated_LCB > 0
h240 longrisk <= 0.10 diagnostic
no-transform / negative controls behave correctly
```

Official pass 需要更严格：

```text
accepted count >= 87
V_integrated_LCB > 0
h240 longrisk UCB <= 0.05
bad UCB <= 0.05
null UCB <= 0.15
```

## H6：selected controller runtime 可过线

H6 成立标准：

```text
feature/certificate compute q90 measured
payload apply q90 measured
selected step_ratio_q90 <= 1.50
memory_ratio <= 1.05
no-event preservation pass
negative controls fail
```

---

# 7. 实验阶段

## P0：v9.5.5 boundary reproduction

### 目标

确认 v9.5.6 没有绕过 v9.5.5 的真实边界。

### 必须记录

```text
source_route_v9550
ledger_rows
T5_action_count
GradeA/B/C/D/E_count
best_signal_AUC / TopK
cover_stability_pass
curvature_fixed_point_pass
memory_antiforgetting_pass
best_APGR_primitive
best_APGR_GradeB_precision
best_APGR_V_integrated_LCB
best_APGR_h240_longrisk
GCERT18_AUC
GCERT18_TopK64_precision
GCERT18_ECE
system_legal_controller_pass
```

### 成立标准

```text
p0_pass = 1
source_route_v9550 = R2a-CleanGeometryTargetTooSparse
system_legal_controller_pass = 0
```

### 可视化

```text
v9.5.4 -> v9.5.5 route waterfall
T5 / Grade / APGR / GCERT status table
```

---

## P1：Grade universe / TopK scope audit

### 目标

先检查 GradeB count 与 GCERT18 TopK64 GradeB precision 是否使用同一 action universe。

### 方法

构建 `grade_scope_audit_v9560.csv`：

```text
row_id
action_id
source_group
universe_id
is_canonical_ap0
is_generated_APY
is_generated_APG
is_generated_APGS
is_generated_APGR
GradeA
GradeB
GradeC
GradeD
GradeE
GCERT18_score
GCERT18_rank
in_GCERT18_TopK64
split_id
dataset
seed
family
stratum
```

分别统计：

```text
canonical_AP0 Grade counts
full_ledger Grade counts
GCERT18 evaluation universe Grade counts
GCERT18 TopK64 Grade counts
```

### 必须记录

```text
GradeB_count_canonical_AP0
GradeB_count_full_ledger
GradeB_count_GCERT18_eval_universe
TopK64_GradeB_count
TopK64_GradeB_precision
universe_consistency_pass
grade_definition_hash_match
split_definition_hash_match
outcome_field_used_by_certificate_count
future_feature_used_count
```

### 判断标准

Pass：

```text
TopK64_GradeB_count <= GradeB_count_GCERT18_eval_universe
grade definition hash match = 1
split hash match = 1
future/outcome feature used = 0
```

Fail：

```text
TopK64 count exceeds evaluation universe positive count
or GCERT18 target not identical to declared GradeB
or certificate used outcome-derived score.
```

### 可视化

```text
Grade count by universe bar chart
GCERT18 rank histogram by Grade
TopK64 composition stacked bar
```

---

## P2：GCERT18 rank-vs-calibration dissection

### 目标

判断 GCERT18 到底是：

```text
1. 真 ranking signal；
2. calibration failure；
3. scope artifact；
4. split leakage；
5. negative-control fragile。
```

### 方法

在 calibration / heldout / leave-dataset-out / leave-stratum-out 上分别评估：

```text
TopK16 / TopK32 / TopK64 / TopK87 / TopK128
threshold by score quantile
threshold by conformal set
threshold by Wilson LCB risk
```

不允许按 dataset 选阈值。所有阈值只在 calibration split 冻结。

### 必须记录

```text
AUC_GradeA
AUC_GradeB
AUC_GradeAB
AUC_GradeABC
TopK_precision_GradeA/B/AB/ABC
TopK_V_integrated_LCB
TopK_h240_longrisk_UCB
TopK_bad_UCB
TopK_null_UCB
ECE
Brier
reliability_bin_count
calibration_to_heldout_precision_drop
calibration_to_heldout_risk_increase
LDO_drop_max
LSO_drop_max
negative_control_TopK_precision
```

### Pass 标准

Ranking pass：

```text
TopK64 GradeAB precision >= 0.75
TopK64 V_integrated_LCB > 0
TopK64 h240 longrisk UCB <= 0.05
TopK64 bad UCB <= 0.05
TopK64 null UCB <= 0.15
negative control TopK64 GradeAB precision <= base rate + 0.05
```

Calibration pass：

```text
ECE <= 0.10
Brier <= baseline_Brier * 0.75
calibration_to_heldout_drop <= 0.10
```

If ranking pass = 1 but calibration pass = 0：

```text
route = R2b-RankingStrongCalibrationWeak
```

This route should continue to rank-constrained controller rather than discard GCERT18.

### 可视化

```text
Reliability diagram
TopK quality curve K=1..256
Risk vs K curve
Value LCB vs K curve
GCERT score distribution by grade
Calibration-to-heldout drift chart
```

---

## P3：Multi-grade target density audit

### 目标

不再只看 T5。检查 A/B/C 各级是否可以组合成 official density，同时不引入风险。

### Target candidates

```text
T_A_only = GradeA
T_AB = GradeA or GradeB
T_ABC = GradeA or GradeB or safe GradeC
T_T5 = old T5
T_T5_plus_safeC = T5 or safe GradeC
T_rank_GCERT18_TopK87 = GCERT18 top 87 after calibration freeze
```

safe GradeC 条件预注册为：

```text
V_integrated_LCB >= 0
h240_longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
cover_collapse = 0
old_family_fail = 0
```

### 必须记录

```text
target_id
accepted_count
coverage
coverage_LCB
Grade composition
V_integrated_mean/LCB
h20/h80/h240 V_LCB
h240 longrisk/UCB
bad/UCB
null/UCB
cover collapse rate
memory fail rate
dataset/family/stratum support
LDO/LSO drops
```

### Pass 标准

Weak official target:

```text
accepted_count >= 87
V_integrated_LCB > 0
h240 longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
support_balance_pass = 1
```

Strong target:

```text
accepted_count >= 87
V20_LCB > 0
V80_LCB >= 0
V240_LCB >= 0
h240 longrisk = 0
bad = 0
null <= 0.10
cover collapse = 0
memory fail <= 0.05
```

### 可视化

```text
Target density vs risk scatter
Grade composition pie chart
Horizon value heatmap
Support by dataset/family/stratum
```

---

## P4：Cover / memory failure anatomy

### 目标

解释为什么 T5 / GradeB 稀疏，为什么 APGR/APGS 会 high long-risk。

### 方法

比较以下集合：

```text
T5 clean actions
GradeA/B actions
GradeC near miss
GCERT18 TopK64
APGR generated actions
APGS generated actions
long-risk actions
old-family-fail actions
```

### 必须记录

```text
basis_effective_rank_delta
hard_tail_cover_entropy_delta
cover_collapse_flag
old_family_fail
old_stratum_fail
forget_risk
curvature_delta
jacobian_spectral_delta
CEp99_delta
payload_norm
AdamW_cosine
signal_score
reservoir_leak
V_integrated
h240_longrisk
```

### 判断标准

Cover blocker confirmed：

```text
corr(cover_collapse, longrisk) > 0.30
or effect_size cover_collapse between positive and longrisk >= 0.50
or APGR longrisk actions have cover rank delta << positives
```

Memory blocker confirmed：

```text
old_family_fail rate in APGR longrisk >= 2x GradeB
or forget_risk_UCB > 0.20 among generated actions
```

### 可视化

```text
Cover rank delta violin by group
Cover entropy delta violin by group
Old family fail heatmap by primitive
Longrisk vs cover entropy scatter
Memory fail vs V_integrated scatter
```

---

## P5：Signal / reservoir audit v5

### 目标

把 `SNR-old-family` 从单一分数拆成 group-level signal，不再让 high-score high-risk 混进 TopK。

### 方法

基于 generalization paper 的 SNR 思路，对每个 edge/basis group 记录：

$$
SNR_g=
\frac{\mu_g^2}{\sigma_g^2/(b-1)+\epsilon}.
$$

同时记录 population-risk style off-diagonal agreement：

$$
\Omega_B(M)=\frac{1}{b(b-1)}\sum_{a\ne c} r_a^\top K^M_{ac}r_c.
$$

### 必须记录

```text
group_snr_mean
group_snr_min
group_snr_tail
per_example_agreement
off_diagonal_agreement
reservoir_leak
signal_channel_score
old_family_signal_score
hard_tail_signal_score
signal_risk_ratio
TopK64 GradeB precision
TopK64 longrisk
```

### Pass 标准

```text
TopK64 GradeAB precision >= 0.50
TopK64 longrisk <= 0.10 diagnostic
TopK64 V_integrated_LCB > 0
reservoir_leak <= 0.05
control_transfer_improvement > 0
```

### 可视化

```text
Signal vs reservoir plane
SNR by edge/basis group heatmap
TopK signal score vs longrisk scatter
```

---

## P6：Existing-action rank controller candidate

### 目标

先不生成新动作，只测试 GCERT18 / multi-grade target 是否能在 existing AP0/ledger actions 上形成 accepted region。

### Controller form

$$
Accept(a)=1
\iff
Rank_{GCERT18}(a)\le K
\land
RiskUCB(a)\le \tau_R
\land
MemoryUCB(a)\le \tau_M
\land
Cost(a)\le C_{max}.
$$

候选：

```text
C1: GCERT18 rank-only TopK
C2: GCERT18 rank + h240 risk guard
C3: GCERT18 rank + cover guard
C4: GCERT18 rank + memory guard
C5: GCERT18 rank + conformal risk bound
C6: GCERT18 rank + GradeAB-safeC target
C7: negative control shuffled certificate
```

### 必须记录

```text
controller_id
calibration thresholds
heldout accepted count
coverage
Grade composition
V_integrated LCB
h240 longrisk UCB
bad UCB
null UCB
cover collapse rate
memory fail rate
feature cost q90
LDO/LSO metrics
negative-control metrics
```

### Official decision gate

```text
coverage in [0.03, 0.15]
V_integrated_LCB > 0
h240 longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
support_balance_pass = 1
LDO/LSO pass = 1
negative controls fail
```

### 可视化

```text
Accepted region grade composition
Heldout vs calibration bar chart
LDO/LSO drop chart
Risk/value Pareto frontier
```

---

## P7：APGC1-APGC8 cover-memory preserving primitive

### 目标

不再生成“合法但高风险”的 payload，而是生成时显式保护 cover / memory / signal。

### APGC primitive family

```text
APGC1-SignalReservoirMaskedEdgeUpdate:
  只更新 group_snr 高且 reservoir_leak 低的 edge/basis group。

APGC2-CoverRankFloorUpdate:
  若 basis effective rank delta 预计低于阈值，则缩放或拒绝该 group update。

APGC3-HardTailCoverEntropyPreservingUpdate:
  保护 hard-tail cover entropy，避免 hard-tail 区域 cover collapse。

APGC4-OldFamilyOrthogonalResidualUpdate:
  functional delta 对 old-family sensitive gradient 做正交投影，减少遗忘。

APGC5-MemoryGuardedTrustRegionUpdate:
  以 old-family / old-stratum forget risk 作为 trust-region 半径。

APGC6-GCERTGuidedSourceBlend:
  只以 GCERT18 high-rank source actions 为 anchor，做低扰动 blend，不从 low-rank source 生成。

APGC7-SymmetricBoundarySignalUpdate:
  加入 symmetric boundary correction，避免单向漂移与 horizon long-risk。

APGC8-NegativeControlShuffledPayload:
  shape-preserving shuffled payload negative control。
```

### 必须记录

```text
primitive_id
generated_action_count
source_action_count
payload_hash_missing
certificate_hash_missing
action_apply_linf_max
no_transform_equivalence
negative_control_divergence
cover_guard_trigger_rate
memory_guard_trigger_rate
signal_mask_sparsity
runtime_cost_q90
```

### Implementation pass

```text
generated_action_count >= 512
payload/certificate hash missing = 0
action apply L∞ max = 0
negative control does not pass
```

### 可视化

```text
Primitive design matrix
Guard trigger rate by primitive
Payload norm / sparsity distribution
```

---

## P8：APGC branch-horizon smoke

### 目标

真实测量 APGC generated actions 的 h20/h80/h240 结果，不能用 proxy。

### Branches

```text
RealAPGC
AdamWOnly
AdamWParallel
BestLR
NoOp
Random
ShuffledAPGC
CertificatePassNoPayload
```

### Horizons

```text
h20
h80
h240
```

### 必须记录

```text
expected_rows
actual_rows
branch_completion_rate
horizon_completion_rate
secondary_delta_completion_rate
unresolved_exception_count
rows_per_sec
quality_audit_pass
duplicate_row_count
label_exclusivity_violation
```

### Pass 标准

```text
actual_rows = expected_rows
quality_audit_pass = 1
unresolved_exception_count = 0
```

---

## P9：APGC outcome / geometry pass

### 目标

判断 APGC 是否真的产生新的 GradeB / safe-C / official geometry actions。

### 必须记录

```text
best_primitive_id
GradeA/B/C precision
OfficialGeo precision
V_integrated mean/LCB
V20/V80/V240 LCB
h240 longrisk
bad
null
cover collapse
memory fail
new positive created rate
longrisk created rate
Damage V_integrated LCB
source-positive preserved rate
```

### Weak pass

```text
best GradeB precision >= 0.25
V_integrated_LCB > 0
h240 longrisk <= 0.20
new positive created rate >= 0.10
longrisk created rate <= 0.20
```

### Strong pass

```text
accepted count >= 87
V_integrated_LCB > 0
h240 longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
cover collapse <= 0.05
memory fail <= 0.05
```

### 可视化

```text
APGC per-primitive value/risk table
Source-to-generated damage matrix
Horizon value heatmap
Cover/memory change by primitive
```

---

## P10：Geometry certificate v4

### 目标

从 GCERT18 发展为可 official 的证书：既保留强 ranking，也改善 calibration。

### 候选证书

```text
GCERT25-GCERT18RankOnly
GCERT26-GCERT18IsotonicCalibrated
GCERT27-GCERT18ConformalRiskBound
GCERT28-GradeABCOrdinalCertificate
GCERT29-CoverMemoryRiskCertificate
GCERT30-SignalReservoirCoverCertificate
GCERT31-APGCSourceAwareCertificate
GCERT32-NegativeControlCertificate
```

### 必须记录

```text
AUC_GradeA/B/AB/ABC
TopK16/32/64/87 precision
V_integrated_LCB
h240 longrisk_UCB
bad_UCB
null_UCB
ECE
Brier
calibration_bin_table
LDO/LSO drops
negative-control pass
feature cost q90
```

### Pass 标准

Ranking certificate pass：

```text
TopK64 GradeAB precision >= 0.75
TopK64 V_integrated_LCB > 0
TopK64 h240 longrisk_UCB <= 0.05
TopK64 bad_UCB <= 0.05
TopK64 null_UCB <= 0.15
```

Calibration certificate pass：

```text
ECE <= 0.10
Brier <= baseline_Brier * 0.75
calibration_to_heldout_precision_drop <= 0.10
```

Official certificate pass：

```text
ranking pass = 1
calibration pass = 1
LDO/LSO pass = 1
negative control fail = 1
```

### 可视化

```text
Reliability diagram
TopK precision/risk curve
Certificate score by grade violin
LDO/LSO drift chart
```

---

## P11：Minimal geometry controller

### 目标

用最小规则选动作，避免再次变成大 feature patching。

### 允许的 feature groups

最多 5 组：

```text
1. GCERT calibrated score or rank；
2. h240 risk estimate；
3. cover stability guard；
4. memory guard；
5. cost guard。
```

Controller:

$$
Accept(a)=1
\iff
Rank_{cert}(a)\le K
\land
UCB(LongRisk_{240}(a))\le \tau_L
\land
UCB(Bad(a))\le \tau_B
\land
UCB(Null(a))\le \tau_N
\land
UCB(MemoryFail(a))\le \tau_M
\land
Cost(a)\le C_{max}.
$$

### 必须记录

```text
controller_id
feature_count
calibration thresholds
heldout accepted count
coverage
V_integrated_LCB
h240 longrisk_UCB
bad_UCB
null_UCB
memory_fail_UCB
cover_collapse_rate
precision by grade
support balance
feature cost
```

### Pass 标准

```text
feature_count <= 5
coverage in [0.03, 0.15]
V_integrated_LCB > 0
h240 longrisk_UCB <= 0.05
bad_UCB <= 0.05
null_UCB <= 0.15
memory_fail_UCB <= 0.10
support_balance_pass = 1
LDO/LSO pass = 1
```

### 可视化

```text
Accepted actions dashboard
Risk/value/support table
Minimal feature ablation chart
```

---

## P12：Selected runtime

### 目标

只在 controller 过 P11 后测 official selected runtime。不能用 diagnostic runtime 代替。

### 必须记录

```text
selected_controller_id
selected_certificate_id
selected_primitive_id
feature_compute_ms_q90
certificate_compute_ms_q90
payload_apply_ms_q90
kernel_launch_count
sync_count
step_ratio_q50/q90/q99
memory_ratio
no_event_preservation_pass
selected_runtime_pass
```

### Pass 标准

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
no_event_preservation_pass = 1
negative controls fail
```

### 可视化

```text
Step time distribution
Payload apply vs base step time
Memory trace
Kernel/sync count chart
```

---

## P13：Leave-out validation

### 目标

确保不是 dataset / family / stratum 特例。

### Splits

```text
leave-dataset-out
leave-family-out
leave-stratum-out
leave-seed-out
```

### 必须记录

```text
per split accepted count
coverage
V_integrated_LCB
longrisk_UCB
bad_UCB
null_UCB
memory_fail_UCB
feature drift
certificate drift
runtime drift
```

### Pass 标准

```text
all mandatory leaveout splits pass weak gate
no split h240 longrisk_UCB > 0.10
no split V_integrated_LCB <= 0
```

---

## P14：Official paired replay

### 目标

只有 system controller 过线后，才跑 official paired replay。

### Controls

```text
RealFunctional
AdamWParallel
BestLR
NoOp
Random
ShuffledPayload
CertificatePassNoPayload
```

### Metrics

```text
CE
accuracy
NLL
ECE
CEp99
margin_p10
hard-tail acc
old-family acc
old-stratum acc
steps_to_target
time_to_target
memory
```

### Pass 标准

```text
RealFunctional beats AdamWParallel and BestLR on primary value metric
ShuffledPayload fails
CertificatePassNoPayload fails
Old-family forgetting not worse than baseline
runtime envelope remains pass
```

---

## P15：Short / full training boundary

### 目标

只有 P14 pass 后，才打开 short/full training。不能提前拿 Base-Acc Sentinel 代替。

### 必须记录

```text
train/val/test acc
NLL
ECE
CEp99
margin_p10
hard stratum acc
sample efficiency
time_to_target
steps_to_target
runtime
memory
continual retained acc
forgetting
```

### Baselines

```text
MatchedMLP
AdamWStrongLRGridMLP
QuadraticFeatureMLP
LQ-t2-h256 AdamW-only
DG-KAN functional selected
```

### Pass 标准

```text
DG-KAN functional beats LQ AdamW-only and strong MLP baseline on at least one pre-registered axis:
  sample efficiency
  robustness
  calibration
  hard-tail
  continual retention
or test acc with matched cost.

No dataset-specific tuning.
```

---

# 8. 并行执行计划

v9.5.6 必须避免“一轮只发现一个 blocker”。

## Batch A：先验一致性与 target 解析

```text
P0 boundary reproduction
P1 grade scope audit
P2 GCERT18 rank/calibration split
P3 multi-grade target density
```

这些可同一时间跑，不需要 APGC outcome。

## Batch B：机制诊断

```text
P4 cover/memory anatomy
P5 signal/reservoir audit
```

这两项只需要 v9.5.5 ledger。

## Batch C：generator 实现

```text
P7 APGC implementation
P8 APGC branch-horizon smoke
P9 APGC outcome
```

P7 可以在 P1/P2 同时开发；P8/P9 等 P7 完成后跑。

## Batch D：certificate/controller/runtime

```text
P10 geometry certificate v4
P11 minimal controller
P12 selected runtime
```

P10 可以并行研究 GCERT18 calibration；P11 必须等 P2/P3/P10；P12 必须等 P11。

## Batch E：downstream

```text
P13 leaveout
P14 paired replay
P15 short/full training
```

只能在 P11/P12 pass 后打开。

---

# 9. Route decision

```text
R0-BoundaryRegression:
  v9.5.5 boundary 不能复现。

R1-GradeScopeInconsistent:
  Grade count / GCERT TopK universe 不一致，必须先修标签口径。

R2-RankingStrongCalibrationWeak:
  GCERT18 ranking / accepted-region 很强，但 ECE 不过。

R3-CleanGeometryTargetTooSparse:
  T5/A/B/C density 仍不够，且 GCERT TopK 不能形成 accepted region。

R4-CoverMemoryPrimaryBlocker:
  cover collapse / memory fail 是主要风险来源。

R5-APGCGeneratedFrontierFail:
  APGC implementation/smoke 过，但 generated actions 仍 value-negative / high-risk。

R6-CertificateCalibratedControllerFail:
  certificate ranking/calibration 某项过，但 minimal controller 不过。

R7-RuntimeFail:
  controller 过，但 selected runtime 不过。

R8-SystemControllerPass:
  controller + runtime 过，可打开 leaveout / paired replay。

R9-PairedReplayFail:
  system pass 后 paired replay 未打过 controls。

R10-ShortFullReady:
  paired replay pass，可以进入 short/full。
```

---

# 10. Dashboard

必须生成一个 `v9560_dashboard.md`，包含：

```text
1. Route summary；
2. T5 / Grade density；
3. GCERT18 rank vs calibration；
4. TopK value/risk curve；
5. Cover/memory anatomy；
6. APGC primitive outcome；
7. Certificate v4；
8. Controller；
9. Runtime；
10. Leaveout / paired replay boundary；
11. No-fake / no-proxy / no-dataset-tuning audit。
```

核心图：

```text
target_density_vs_risk.png
gcert18_topk_curve.png
gcert18_reliability.png
grade_scope_consistency.png
cover_memory_violin.png
signal_reservoir_plane.png
apgc_primitive_matrix.png
controller_accepted_region.png
selected_runtime_trace.png
```

---

# 11. No-fake / no-leak audit

每一轮都必须记录：

```text
fake_data_used
proxy_row_used
cpu_offload_used
uses_loss_backward
uses_teacher
uses_loss_modification
uses_dataset_name_for_selector
uses_dataset_name_for_controller
uses_validation_or_test_for_controller
uses_future_outcome_for_features
uses_outcome_at_commit
uses_old_table_for_official
diagnostic_promoted_to_official
base_acc_used_for_controller
```

全部必须为 0，除了：

```text
manual_forward/manual_backward/manual_adamw_update = 1/1/1
train_stream_probe = 1
```

---

# 12. 最终判断标准

v9.5.6 不以“又实现了一个 primitive”作为成功。

真正成功只有三种：

## 成功 A：Existing-action controller pass

```text
GCERT18/GCERTv4 让 existing AP0/ledger actions 形成 official accepted region；
controller pass；
runtime pass。
```

## 成功 B：APGC generated frontier pass

```text
APGC 生成了足够多 GradeB / safe-C / official geometry actions；
certificate 可选中；
controller pass；
runtime pass。
```

## 成功 C：明确证伪并进入更大 reset

```text
Grade scope audit 通过；
GCERT18 ranking 不可迁移；
APGC 不能生成正例；
cover/memory/signal 都无法形成 accepted region；
则进入 v9.6.0 architecture-level geometry redesign。
```

v9.5.6 的最差但仍有价值结果：

```text
证明 GCERT18 TopK 是 artifact 或不可迁移；
证明 clean geometry density 在 current action family 中确实不足；
证明需要 architecture-level / basis-level geometry redesign。
```

---

# 13. 给实验执行者的简明指令

这轮不要做：

```text
APGR5 小调参；
GCERT18 threshold 小调参；
只看 AUC；
只看 T5；
只看 h20；
提前 paired replay；
用 Base-Acc Sentinel 报 functional success。
```

这轮必须做：

```text
先查 Grade / GCERT universe；
再拆 GCERT18 ranking vs calibration；
再组合 A/B/C target；
再定位 cover/memory；
再实现 APGC；
再测 certificate/controller/runtime。
```

最终一句话：

> v9.5.6 的目标不是“再找一个分数”，而是判断 v9.5.5 发现的强 TopK 几何信号是否能变成真正可部署的 functional update 选择规则；如果不能，就要明确进入更大范围的几何结构重设计。
