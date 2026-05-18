# DG-KAN v9.5.5 Trainable Geometry / Signal-Reservoir Primitive / Parallel Closure 完整实验计划

> 本计划基于 v9.5.4 `Geometry Definition / Signal-Channel Primitive / Parallel Closure` 的真实结果制定。  
> v9.5.5 不继续微调 APGS、GCERT16 或单个几何阈值。  
> 本轮目标是把“好 functional update 动作”从一个二元标签，改成一张可测、可生成、可筛选、可部署的训练形状表。

---

## 0. 一句话目标

v9.5.5 要回答的问题是：

$$
\boxed{
\text{能不能在训练当下找到或生成一种小参数改动，让 KAN 的训练形状变好，而不是只让某个 horizon 的结果短暂变好？}
}
$$

这里“训练形状变好”不是一个口号。它必须同时满足：

```text
1. h20 短期有收益；
2. h80 中期不坏；
3. h240 长期不出 long-risk；
4. bad/null 低；
5. hard-tail 不被放大；
6. cover 不塌缩；
7. curvature / local Jacobian 不爆；
8. old-family memory 不忘；
9. update 方向来自稳定信号，而不是 minibatch noise；
10. 训练当下可以用证书判断；
11. payload apply 和 feature/certificate cost 仍能过 runtime gate。
```

v9.5.5 的主线不是：

```text
APGS7 再调一下；
GCERT16 再调一下；
SC7 TopK 再调一下；
T5 阈值再放松一点。
```

v9.5.5 的主线是：

```text
先把“好训练形状”量出来；
再找出旧 AP0 / APY / APG / APGS 里有没有这种动作；
再判断训练当下有没有足够信号识别它；
如果没有，就直接生成这种动作；
最后才允许进入 controller / runtime / paired replay。
```

---

## 1. 对 v9.5.4 的独立判断

### 1.1 v9.5.4 有进展，但不是能力成功

v9.5.4 的 route 是：

```text
route = R3-LegalGeometrySignalAbsent
success_v9540_strict_purekan_functional = False
success_v9540_full_functional = False
success_v9540_external_ready = False
primary_blocker = commit_time_geometry_signal_absent
```

这不是 functional success，也不是 controller success。

但它不是没进展。v9.5.4 已经完成了这些硬推进：

```text
Geometry Outcome Ledger rows = 3900
canonical AP0 rows = 2876
generated APY rows = 512
generated APG rows = 512
required field missing / NaN / Inf = 0 / 0 / 0
identity join pass = 1
```

也就是说，现在我们第一次把 AP0、APY、APG 的 outcome 和训练形状指标放到同一张 ledger 里。这个是后面一切几何判断的地基。

### 1.2 v9.5.4 的关键好消息：T5 是真实但太薄的 clean target

P2 扫了 329 个 geometry targets：

```text
weak geometry target = 32
official geometry target = 0
best relaxed target = T5-RelaxedV80NonNegative
action count = 57
coverage = 0.019819193324061197
V_integrated LCB = 0.1960342568901401
h240 longrisk = 0.0
bad/null = 0.0 / 0.0
```

这说明不是完全没有“好训练形状”。

但 coverage gate 如果还是 $0.03$，在 2876 个 action 里至少需要：

$$
\lceil 0.03 \times 2876 \rceil = 87
$$

T5 只有 57 个 action，差 30 个。它不是垃圾 target，但还不能 official。

独立判断：

$$
\boxed{
\text{T5 说明 clean geometry target 存在；问题是太稀疏，不能直接当 controller target。}
}
$$

### 1.3 我不完全同意 “legal geometry signal absent” 的字面说法

v9.5.4 报告说 primary blocker 是 `commit_time_geometry_signal_absent`。我认为这个表述过粗。

P3 的 best signal 是：

```text
SC7-leave-one-out-transfer
AUC_OfficialGeoCandidate = 0.9706870146967946
TopK64 precision = 0.125
TopK64 V_integrated LCB = -0.260665499652037
```

P10 的 best certificate 是：

```text
GCERT16-OfficialMinimalGeometryCertificate
AUC_OfficialGeoCandidate = 0.9977771191464138
TopK64 precision = 0.21875
TopK64 longrisk = 0.1875
ECE = 0.6297824644891717
```

这两个 AUC 都很高，说明不是“完全没有信号”。真正的问题是：

```text
1. 全局排序看起来强；
2. 但 top accepted region 不干净；
3. TopK64 的 V_integrated LCB 不够；
4. long-risk 仍高；
5. certificate calibration 很差；
6. 不能形成 online accept set。
```

所以更准确的 blocker 是：

$$
\boxed{
\text{存在 diagnostic signal，但还没有能直接选动作的 commit-time signal。}
}
$$

这很重要。下一步不应该再问“有没有任何 signal”，而应该问：

```text
为什么 AUC 很高，但 TopK / coverage / risk / calibration 不过？
```

### 1.4 v9.5.4 的 GeoScore 是最接近可用的一条线，但还没过

P6 的 GeoScore diagnostic：

```text
best = GS1-VSignalRisk
heldout accepted = 91
coverage = 0.06594202898550725
V_integrated LCB = 0.04369212799578348
h240 longrisk = 0.0
```

这很有价值，因为它说明“多指标组合”比单个 APG / APGS generator 更接近可用区域。

但它 official 失败：

```text
best bad event = 0.06593406593406594 > 0.05
OfficialGeo precision 仍低
source controller pass = 0
```

独立判断：

$$
\boxed{
\text{GeoScore 说明路线不是完全没信号；但它还不是一个可上线的动作选择器。}
}
$$

### 1.5 APGS 失败很严重，不是阈值问题

APGS1-APGS8 工程上是过的：

```text
generated actions = 512
payload/certificate hash missing = 0 / 0
action apply L∞ max = 0.0
branch-horizon rows = 12288 / 12288
unresolved exception = 0
rows/sec = 21.05364247842129
```

但结果很差：

```text
best APGS = APGS7-SymmetricBoundaryResidualUpdate
OfficialGeo precision = 0.015625
OutcomeGood precision = 0.03125
V_integrated LCB = -0.9218810936698074
h240 longrisk = 0.765625
```

P9 damage 更直接：

```text
best damage row = APGS1-SignalChannelEdgeMask
new positive created rate = 0.0
longrisk created rate = 0.75
Damage integrated LCB = -2.3023029125421157
```

这说明 APGS 不是“还差一点”。它生成的是合法 payload，但不是改善训练形状的 payload。

### 1.6 Base-Acc Sentinel 仍是健康检查，不是 functional success

v9.5.4 复用前序 Base-Acc Sentinel：

```text
rows = 120
LQ mean test acc = 0.6537760416666667
MatchedMLP mean test acc = 0.562890625
QuadraticFeatureMLP mean test acc = 0.40234375
AdamWStrongLRGridMLP = 0.628515625
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
```

所以可以说：

```text
LQ-t2-h256 base 没死；
fixed sentinel 下比 MatchedMLP 和 StrongLRGridMLP 都高；
但 functional update 还没证明赢 MLP。
```

不能说：

```text
DG-KAN functional controller 已经赢 MLP。
```

---

## 2. 相关工作的启发：我们应该怎么定义“好训练形状”

### 2.1 Deep Manifold 给的启发

Deep Manifold 的核心启发可以翻译成很直接的话：

```text
神经网络不是在固定空间里拟合；
训练每一步都会移动局部覆盖区域；
好的训练不是只让 loss 降，而是让局部区域逐渐稳定；
曲率、边界、局部 cover、固定点区域决定了能不能又快又稳地学。
```

这对 DG-KAN 很重要。KAN 的 edge/basis 结构天然像很多小局部函数拼起来，所以 functional update 如果只猛拉 loss，很容易：

```text
1. 把某些 edge/basis 拉过头；
2. 让 hard-tail 变坏；
3. 让 h20 好但 h240 出风险；
4. 让旧 family 被忘掉；
5. 让局部曲率或 Jacobian 变大，训练后续不稳。
```

因此，一个好的 functional update 不应该只是：

```text
这个动作 V_integrated > 0。
```

它还应该是：

```text
这个动作让局部 cover 更均衡；
让曲率不爆；
让 hard-tail 不变坏；
让旧知识不掉；
让训练轨迹更接近稳定区域。
```

### 2.2 Generalization / signal-reservoir 理论给的启发

Generalization 论文给的启发也可以翻译成很直接的话：

```text
不是所有能降低训练误差的方向都会帮助测试；
有些方向只是记住训练集噪声；
有些方向才是真正能传到测试集的信号方向；
一个好 update 要尽量走信号方向，不要走噪声方向。
```

它还给了一个很适合我们使用的筛选思想：

$$
\mu_k^2 > \frac{\sigma_k^2}{b-1}
$$

也就是：一个参数方向只有在 minibatch 内多个样本给出一致支持时，才值得更新。换成 DG-KAN 的语言：

```text
某条 edge / basis / group 的 functional delta，
如果只被少数样本推动，
或者样本之间方向冲突很大，
就不应该被当作好动作。
```

所以 v9.5.5 不应该只做一个 global SNR。应该按下面粒度记录：

```text
edge-level SNR；
basis-level SNR；
class/hard-tail SNR；
old-family SNR；
AdamW-compatible SNR；
leave-one-out transfer SNR。
```

---

## 3. v9.5.5 总体目标

v9.5.5 的目标不是直接拿到 full functional success，而是建立一个更准确的判断系统：

$$
\boxed{
\text{判断一个动作是否真的改善了 KAN 的训练形状，并生成这种动作。}
}
$$

本轮最低有效推进：

```text
1. 完成 Trainable Geometry Ledger v2；
2. 解释 T5 为什么只有 57 个动作；
3. 解释 SC7 / GCERT16 为什么 AUC 高但 TopK 差；
4. 分解 signal / cover / curvature / memory 对 geometry target 的贡献；
5. 设计 APGR1-APGR8，不再只是合法 payload，而是带 signal + cover + curvature + memory 约束的 payload；
6. branch-horizon smoke 完整落盘；
7. 如果 APGR 不过，必须能判定失败原因：source 不够、signal 弱、cover 损伤、curvature 损伤、memory 损伤、horizon 风险、runtime 过慢；
8. 只有在 controller gate 过线时才打开 selected runtime / paired replay。
```

---

## 4. 硬约束

继续遵守：

```text
no teacher
no self-teacher
no distillation
no loss modification
no label smoothing
no focal / margin / calibration auxiliary loss
no dataset-specific threshold
no validation/test feature at commit time
no future outcome at commit time
no old table official usage
no fake / proxy rows
no CPU offload
KAN path 不使用 PyTorch loss.backward graph
manual forward / backward / AdamW update 保持
functional update 仍是 update rule，不是 loss trick
```

Functional update 仍写成：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{AdamW}
+
\Delta\theta_{functional}
$$

不允许写成：

$$
L = CE + \lambda L_{functional}
$$

允许使用：

```text
commit-time train batch statistics；
per-example gradients from manual backward；
edge/basis grouped gradient mean/variance；
cheap HVP / finite-difference curvature probe；
activation cover / basis cover statistics；
old-family replay mini buffer for diagnostics and memory guard；
branch-horizon replay for outcome labels；
leave-dataset-out and leave-stratum-out diagnostics；
Base-Acc Sentinel，且不用于 controller。
```

---

## 5. 新定义：Trainable Geometry Card

每个 action $a$ 都要有一张卡：

```text
action_id
source_action_id
primitive_id
payload_hash
step / dataset / seed / family / stratum
V20, V80, V240
V_integrated
bad20, bad80, bad240
null20, null80, null240
longrisk240
CEp99_delta_20/80/240
margin_p10_delta_20/80/240
NLL_delta_20/80/240
ECE_delta_20/80/240
hard_tail_fraction_delta
edge_snr_mean / p10 / p90
basis_snr_mean / p10 / p90
signal_transfer_score
reservoir_leak_score
cover_entropy_before / after / delta
cover_balance_delta
basis_effective_rank_delta
active_edge_count_delta
curvature_proxy_before / after / delta
jacobian_spectral_proxy_delta
fixed_point_residual_delta
boundary_symmetry_score
adamw_conflict_cosine
old_family_margin_delta
old_family_forget_risk
feature_compute_ms
certificate_compute_ms
payload_apply_ms
memory_ratio_if_selected
```

然后不再只给一个 `Good = 0/1`。先给四个等级：

```text
A: official candidate
B: diagnostic good but too sparse / too costly
C: safe but weak value
D: risky or destructive
E: replay / label / cost not trusted
```

A 级动作至少满足：

$$
V_{20}^{LCB} > 0
$$

$$
V_{80}^{LCB} \ge -\epsilon_{80}
$$

$$
V_{240}^{LCB} \ge -\epsilon_{240}
$$

$$
LongRisk_{240}^{UCB} \le \tau_L
$$

$$
Bad^{UCB} \le \tau_B
$$

$$
Null^{UCB} \le \tau_N
$$

$$
SNR_{group}^{LCB} > \tau_S
$$

$$
CoverDamage^{UCB} \le \tau_C
$$

$$
CurvatureDamage^{UCB} \le \tau_K
$$

$$
ForgetRisk^{UCB} \le \tau_F
$$

$$
Cost \le C_{max}
$$

---

## 6. 核心假设

### H1：T5 不是偶然，它代表“干净但太薄”的几何区域

成立标准：

```text
T5-like target 在 LDO/LSO 中仍保持：
  V_integrated LCB > 0
  h240 longrisk = 0
  bad/null <= 0.05 / 0.15
  至少 3 datasets 都有正例
  family/stratum support 不坍缩
```

失败标准：

```text
T5 只来自单 dataset / 单 family / 单 step bucket / 单 payload bucket。
```

### H2：SC7 / GCERT16 的高 AUC 不是假的，但 top-region ranking 错了

成立标准：

```text
SC7 或 GCERT16 在随机 pair ranking 上强；
但 TopK risk/value/cost 不过；
score 分布存在 high-score high-risk 区域。
```

失败标准：

```text
AUC 高来自 leakage、outcome-derived feature 或 identity artifact。
```

### H3：好动作需要 signal + cover + curvature + memory 同时约束

成立标准：

```text
单项 signal / cover / curvature / memory 任何一个都不能独立过 TopK；
组合后 TopK64 / TopK128 的 V、longrisk、bad/null 同时改善。
```

失败标准：

```text
某个单项已经足够选出 official action，组合不是必要。
```

### H4：现有 APGS 失败是因为它没有直接优化 Trainable Geometry Card

成立标准：

```text
APGS source-to-generated damage 显示：
  new positive created rate = 0
  longrisk created rate high
  cover / curvature / memory 至少一个明显损伤
```

失败标准：

```text
APGS 失败主要是样本量太小或 source bad，扩大后过线。
```

### H5：APGR 新 primitive 必须用群组 SNR 和边界约束生成 update

成立标准：

```text
APGR branch-horizon smoke 中至少一个 primitive：
  generated action count >= 64
  V_integrated LCB > 0
  h240 longrisk <= 0.10
  bad <= 0.05
  null <= 0.15
  TopK64 geometry precision >= 0.50 diagnostic
```

official 成立标准更高：

```text
coverage >= 0.03
precision / value / risk / support 全部过 gate
```

### H6：如果 raw signal upper bound 都不能转成 TopK，则必须换 action space，不再 feature search

成立标准：

```text
High-capacity legal probe AUC high but TopK64/128 多次失败；
APGR 也不能生成正例；
则判定当前 action space / generation objective 错。
```

---

## 7. 并行执行总览

v9.5.5 不再串行等一个大 runner 结束后才知道下一层失败。分成 5 条并行线：

```text
Lane A: target / label / T5 anatomy
Lane B: signal / reservoir / SNR / transfer
Lane C: cover / curvature / memory geometry
Lane D: APGR generator + branch-horizon smoke
Lane E: certificate / controller / runtime boundary
```

每条线必须独立落盘，互相不能用 future outcome at commit time。

---

# Part I. Boundary 与 Ledger

## P0. v9.5.4 boundary reproduction

### 目标

确认 v9.5.4 的结果没有回退。

### 必须记录

```text
source_route_v9540
ledger_rows
canonical_ap0_rows
generated_apy_rows
generated_apg_rows
weak_geometry_target_count
official_geometry_target_count
selected_target_id
selected_action_count
selected_coverage
selected_V_integrated_lcb
selected_h240_longrisk
best_signal_feature_id
best_signal_AUC
best_signal_TopK64_precision
best_geoscore_id
best_geoscore_accepted_count
best_apgs_primitive
best_apgs_V_integrated_lcb
best_certificate_id
best_certificate_AUC
best_certificate_TopK64_precision
system_legal_controller_pass
```

### Pass 标准

```text
P0 pass if:
  v9.5.4 route reproduced；
  no-fake audit reproduced；
  ledger / APGS / GCERT boundary reproduced。
```

### 可视化

```text
v9.3.5 -> v9.5.4 route timeline；
T5 / GeoScore / APGS / GCERT key metric waterfall；
AUC vs TopK precision scatter。
```

---

## P1. Trainable Geometry Ledger v2

### 目标

在 v9.5.4 ledger 基础上加上 signal、cover、curvature、memory、cost 字段，形成一张统一 action 表。

### 要做什么

对以下 action 集合全部生成 card：

```text
canonical AP0 = 2876
generated APY = 512
generated APG = 512
generated APGS = 512
新增 APGR smoke actions = 之后 P7 写入
```

### 必须记录

```text
row_count_total
row_count_by_source
missing_required_field_count
nan_count
inf_count
identity_join_pass
commit_time_field_count
diagnostic_field_count
future_outcome_leakage_count
feature_cost_recorded
```

### Pass 标准

```text
missing_required_field_count = 0
nan_count = 0
inf_count = 0
identity_join_pass = 1
future_outcome_leakage_count = 0
feature_cost_recorded = 1
```

### 可视化

```text
field missing heatmap；
action source composition bar；
feature compute cost histogram；
commit-time vs diagnostic field matrix。
```

---

# Part II. Target 与好训练形状定义

## P2. T5 anatomy and target-density audit

### 目标

弄清楚 T5 为什么干净但只有 57 个动作。

### 要做什么

对 T5 和附近 target 做分解：

```text
T5 original；
T5 + h80 stricter；
T5 + h20 stricter；
T5 + bad/null stricter；
T5 + cover constraint；
T5 + curvature constraint；
T5 + memory constraint；
T5 without each condition。
```

### 必须记录

```text
target_id
action_count
coverage
coverage_lcb
V20_lcb
V80_lcb
V240_lcb
V_integrated_lcb
h240_longrisk
bad_rate
null_rate
accepted_dataset_count
accepted_family_count
accepted_stratum_count
max_dataset_share
max_family_share
max_stratum_share
support_balance_pass
leave_dataset_out_min_count
leave_stratum_out_min_count
```

### 判断标准

T5 可作为下一阶段 official target 候选，必须满足：

```text
action_count >= 87
coverage >= 0.03
V_integrated_lcb > 0
h240_longrisk <= 0.05
bad_rate <= 0.05
null_rate <= 0.15
accepted_dataset_count >= 3
support_balance_pass = 1
```

如果 T5 仍只有 57 左右，但全部干净，则 route 不应是 target absent，而是：

```text
R2a-CleanGeometryTargetTooSparse
```

### 可视化

```text
condition ablation waterfall；
coverage vs V_integrated scatter；
h20/h80/h240 heatmap；
T5 by dataset/family/stratum stacked bars；
why-action-excluded Pareto chart。
```

---

## P3. Multi-grade geometry label instead of binary target

### 目标

不要再把动作硬压成一个好/坏标签。先给每个动作分级。

### 分级

```text
Grade A: official-grade geometry action
Grade B: clean but too sparse / too low coverage
Grade C: safe but weak value
Grade D: value-positive but risky
Grade E: destructive / invalid
```

### 必须记录

```text
grade
grade_reason
V_score
risk_score
signal_score
cover_score
curvature_score
memory_score
cost_score
```

总分：

$$
GeoScore(a)
=
w_V Score_V(a)
-
w_R Score_R(a)
+
w_S Score_S(a)
+
w_C Score_C(a)
-
w_K Score_K(a)
-
w_F Score_F(a)
-
w_T Cost(a)
$$

但每个分量必须单独报告，不能只报告总分。

### 判断标准

```text
Grade A count >= 87 才能进入 official target；
Grade B count >= 87 可以进入 diagnostic generator；
Grade D 不能进入 controller，但用于 failure autopsy。
```

### 可视化

```text
Grade composition bar；
GeoScore component radar；
V vs risk vs memory 3D scatter；
Grade transition across horizon plot。
```

---

# Part III. Signal / Reservoir / SNR

## P4. Signal-channel raw upper-bound v4

### 目标

解释 SC7 AUC 高但 TopK 差的原因。

### 要做什么

对每个 action 计算：

```text
leave-one-out transfer score；
off-diagonal agreement score；
per-example gradient agreement；
edge-level SNR；
basis-level SNR；
hard-tail SNR；
old-family SNR；
AdamW-compatible SNR；
reservoir-leak proxy；
```

SNR 形式：

$$
SNR_g
=
\frac{\mu_g^2}{\sigma_g^2/(b-1)+\epsilon}
$$

其中 $g$ 是 edge / basis / family / hard-tail group。

### 必须记录

```text
feature_id
feature_cost_ms_q50/q90
AUC_GradeA
AUC_GradeB
AUC_T5
TopK64_precision_GradeA/B/T5
TopK128_precision_GradeA/B/T5
TopK64_V_integrated_lcb
TopK64_h240_longrisk
TopK64_bad/null
calibration_ece
monotone_sign_pass
```

### 判断标准

raw signal upper-bound pass：

```text
TopK64 GradeA/B precision >= 0.50
TopK64 V_integrated_lcb > 0
TopK64 h240_longrisk <= 0.10
TopK64 bad <= 0.05
feature_cost_ms_q90 <= 0.20
```

如果 AUC 高但 TopK 仍差，必须输出：

```text
high_AUC_low_TopK_failure = 1
failure_reason = class_imbalance / high_score_high_risk / calibration_bad / leakage / source_mismatch
```

### 可视化

```text
ROC + PR + TopK curve 同图；
score histogram by grade；
TopK risk waterfall；
AUC-vs-TopK scatter；
SNR group heatmap。
```

---

## P5. Reservoir / noise rejection audit

### 目标

判断动作是否只是降低训练误差但不帮助泛化。

### 必须记录

```text
train_only_improvement
control_transfer_improvement
train_to_control_transfer_gap
reservoir_leak_score
signal_channel_score
shuffle_payload_pass
random_payload_pass
noisy_label_probe_delta
```

### 判断标准

```text
signal_channel_score positive；
reservoir_leak_score low；
shuffled payload 不通过；
random payload 不通过；
train_to_control_transfer_gap 不为负。
```

### 可视化

```text
signal vs reservoir plane；
train improvement vs control improvement scatter；
shuffle/random negative-control comparison；
noise rejection waterfall。
```

---

# Part IV. Cover / Curvature / Memory

## P6. Cover stability audit

### 目标

判断动作是否让 KAN 的 edge/basis 覆盖更健康。

### 指标

```text
activation_cover_entropy_before/after/delta
basis_cover_entropy_before/after/delta
edge_active_count_before/after/delta
basis_effective_rank_before/after/delta
per-class cover entropy delta
hard-tail cover entropy delta
cover collapse count
cover over-concentration count
```

### Pass 标准

```text
cover_collapse_count = 0
basis_effective_rank_delta >= -epsilon_rank
hard_tail_cover_entropy_delta >= -epsilon_cover
per-class max cover gap <= tau_cover_gap
```

### 可视化

```text
cover entropy histogram before/after；
effective rank before/after bar；
cover collapse heatmap by family；
edge/basis active count distribution。
```

---

## P7. Curvature / fixed-point stability audit

### 目标

判断动作是否导致局部曲率、Jacobian 或固定点残差变坏。

### 指标

```text
curvature_proxy_before/after/delta
HVP_norm_proxy
Jacobian_spectral_proxy_before/after/delta
local_Lipschitz_proxy_delta
fixed_point_residual_delta
CEp99_delta
margin_p10_delta
hard_tail_fraction_delta
oscillation_proxy_h20_h80_h240
```

局部稳定目标：

$$
\rho_{local}(a) < 1
$$

如果不能直接测谱半径，用 cheap proxy：

$$
\widehat{\rho}_{local}
=
\frac{\|f_{\theta+\Delta\theta}(x+\delta)-f_{\theta+\Delta\theta}(x)\|}{\|\delta\|+\epsilon}
$$

### Pass 标准

```text
curvature_delta_ucb <= tau_curv
jacobian_spectral_proxy_delta <= tau_jac
fixed_point_residual_delta <= 0
CEp99_delta <= tau_tail
margin_p10_delta >= -tau_margin
```

### 可视化

```text
curvature vs longrisk scatter；
Jacobian proxy vs V_integrated scatter；
hard-tail CEp99 before/after plot；
h20/h80/h240 oscillation plot。
```

---

## P8. Memory / anti-forgetting audit

### 目标

判断动作是否伤害旧 family、old stratum 或 hard examples。

### 指标

```text
old_family_margin_delta
old_family_CE_delta
old_family_fail_count
old_stratum_fail_count
memory_probe_acc_delta
memory_probe_NLL_delta
forget_risk_score
AdamW_conflict_with_memory_gradient
```

### Pass 标准

```text
old_family_fail_count = 0
old_stratum_fail_count = 0
forget_risk_ucb <= tau_forget
memory_probe_acc_delta >= -epsilon_acc
```

### 可视化

```text
old-family margin delta heatmap；
forget risk histogram；
memory gradient conflict scatter；
family-level pass/fail table。
```

---

# Part V. Generator Reset

## P9. APGS failure autopsy under Trainable Geometry Card

### 目标

不能只说 APGS fail。要知道 APGS 是怎么 fail 的。

### 必须记录

```text
primitive_id
source_action_count
generated_action_count
new_positive_created_rate
source_positive_preserved_rate
longrisk_created_rate
cover_damage_rate
curvature_damage_rate
memory_damage_rate
signal_damage_rate
Damage_V_integrated_lcb
primary_damage_mode
```

### 判断标准

如果 APGS 的主失败是：

```text
signal_damage
```

则 APGR 要改 signal-channel update。

如果主失败是：

```text
cover_damage / curvature_damage / memory_damage
```

则 APGR 要加对应 guard。

### 可视化

```text
source-to-generated damage waterfall；
primitive-by-damage heatmap；
new positive vs longrisk scatter；
source preserve Sankey diagram。
```

---

## P10. APGR1-APGR8 geometry-signal primitive implementation

### 目标

实现 8 个新 primitive。它们不能只是合法 payload builder，必须直接使用 signal / cover / curvature / memory guard。

### APGR primitives

```text
APGR1-GroupSNRProjectedEdgeUpdate
  只保留 edge/basis group 中 SNR 过线的 functional delta。

APGR2-SignalReservoirMaskedUpdate
  保留 leave-one-out transfer positive 的方向，抑制 reservoir-leak 高的方向。

APGR3-CoverBalancedEdgeUpdate
  生成时约束 basis cover entropy / effective rank 不下降。

APGR4-CurvatureTrustRegionUpdate
  使用 cheap HVP / Jacobian proxy 限制曲率伤害。

APGR5-MemoryGuardedResidualUpdate
  对 old-family gradient conflict 做投影或缩放。

APGR6-SymmetricBoundaryKANUpdate
  使用正负边界对称约束，防止单向 drift。

APGR7-GeoScoreConstrainedBlend
  组合 signal、cover、curvature、memory 四类 guard，做 conservative blend。

APGR8-NegativeControlShuffledPayload
  shape-preserving shuffled negative control，必须不过。
```

### 必须记录

```text
primitive_id
generated_action_count
payload_hash_missing_count
certificate_hash_missing_count
action_apply_error_linf_max
action_apply_error_relative_max
cosine_logged_applied_min
feature_cost_ms_q90
payload_apply_ms_q90
```

### Pass 标准

```text
generated_action_count >= 64 per primitive
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max <= 1e-7
APGR8_negative_control_pass = 0
```

### 可视化

```text
per-primitive payload norm histogram；
apply error histogram；
feature cost and payload cost bar；
certificate component distribution。
```

---

## P11. APGR branch-horizon smoke

### 目标

真实测 APGR 动作在 branches 和 horizons 上的结果。

### Branches

```text
RealAPGR
AdamWOnly
AdamWParallel
bestLR
NoOp
RandomPayload
ShuffledAPGRPayload
CertificatePassNoPayload
```

### Horizons

```text
20
80
240
```

### 必须记录

```text
expected_rows
actual_rows
branch_completion_rate
horizon_completion_rate
secondary_delta_completion_rate
rows_per_sec
unresolved_exception_count
quality_audit_pass
```

### Pass 标准

```text
actual_rows = expected_rows
unresolved_exception_count = 0
quality_audit_pass = 1
```

### 可视化

```text
branch-horizon completion heatmap；
rows/sec trace；
quality audit table；
per-branch V distribution。
```

---

## P12. APGR outcome / geometry pass

### 目标

判断 APGR 是否真的生成好训练形状。

### 必须记录

```text
primitive_id
OfficialGeo precision
GradeA precision
GradeB precision
OutcomeGood precision
V20_lcb
V80_lcb
V240_lcb
V_integrated_lcb
h240_longrisk
bad_rate
null_rate
cover_damage_rate
curvature_damage_rate
memory_damage_rate
new_positive_created_rate
source_positive_preserved_rate
```

### Weak pass

```text
V_integrated_lcb > 0
h240_longrisk <= 0.10
bad_rate <= 0.05
null_rate <= 0.15
GradeA/B precision TopK64 >= 0.25
```

### Strong pass

```text
coverage >= 0.03
V_integrated_lcb > 0
h240_longrisk <= 0.05
bad_rate <= 0.05
null_rate <= 0.15
support_balance_pass = 1
```

### 可视化

```text
primitive performance radar；
h20/h80/h240 V heatmap；
longrisk vs V scatter；
source-to-generated geometry Sankey；
negative control comparison。
```

---

# Part VI. Certificate / Controller / Runtime

## P13. Geometry certificate v3

### 目标

把 P4-P8 的训练形状信号压成一个小证书，但不能只追 AUC。

### Certificate fields

```text
signal_snr_lcb
transfer_score_lcb
reservoir_leak_ucb
cover_damage_ucb
curvature_damage_ucb
memory_forget_ucb
horizon_risk_ucb
cost_ucb
```

### 接受规则

$$
CertPass(a)=1
$$

当且仅当：

$$
SignalLCB(a)>\tau_S
$$

$$
ReservoirLeakUCB(a)<\tau_R
$$

$$
CoverDamageUCB(a)<\tau_C
$$

$$
CurvatureDamageUCB(a)<\tau_K
$$

$$
ForgetRiskUCB(a)<\tau_F
$$

$$
CostUCB(a)<\tau_T
$$

### 必须记录

```text
AUC_GradeA/B/T5
TopK64_precision_GradeA/B/T5
TopK128_precision_GradeA/B/T5
TopK64_V_integrated_lcb
TopK64_h240_longrisk
TopK64_bad/null
ECE
calibration_to_heldout_drift
monotone_sign_pass
feature_count
cost_q90
```

### Pass 标准

```text
TopK64 GradeA/B precision >= 0.50
TopK64 V_integrated_lcb > 0
TopK64 h240_longrisk <= 0.10
TopK64 bad <= 0.05
ECE <= 0.10
cost_q90 <= 0.20 ms
```

### 可视化

```text
calibration plot；
TopK precision curve；
certificate component radar；
ECE reliability diagram；
score vs longrisk scatter。
```

---

## P14. Minimal geometry controller

### 目标

只在 certificate 过线后才运行 controller search。

### Controller form

$$
Accept(a)=1
$$

当且仅当：

$$
CertPass(a)=1
$$

$$
LCB(V_{integrated}(a))>0
$$

$$
UCB(LongRisk_{240}(a))\le\tau_L
$$

$$
UCB(Bad(a))\le\tau_B
$$

$$
UCB(Null(a))\le\tau_N
$$

$$
Cost(a)\le C_{max}
$$

### 必须记录

```text
accepted_count_cal
accepted_count_heldout
coverage_cal/heldout
precision_cal/heldout
V_integrated_lcb_cal/heldout
h240_longrisk_cal/heldout
bad/null_cal/heldout
precision_lcb
bad_ucb
support_balance_pass
leave_dataset_out_pass
leave_stratum_out_pass
```

### Pass 标准

```text
coverage_heldout in [0.03, 0.15]
V_integrated_lcb_heldout > 0
h240_longrisk_heldout <= 0.05
bad_event_heldout <= 0.05
null_event_heldout <= 0.15
precision_lcb >= 0.75 if binary precision is used
support_balance_pass = 1
LDO/LSO diagnostic not catastrophic
```

### 可视化

```text
accepted distribution by dataset/family/stratum；
heldout vs calibration metric bars；
LDO/LSO spider plot；
threshold sensitivity around frozen point。
```

---

## P15. Selected runtime

### 目标

只有 P14 过后，才测 selected runtime。

### 必须记录

```text
selected_controller_id
selected_primitive_id
feature_compute_ms_q50/q90
certificate_compute_ms_q50/q90
payload_apply_ms_q50/q90
kernel_launch_count
sync_count
step_ratio_q90
memory_ratio
empty_step_launch_count
active_step_launches_q90
```

### Pass 标准

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
empty_step_launch_count = 0
selected_controller_runtime_measured = 1
not_offline_materializer_time = 1
```

### 可视化

```text
runtime waterfall；
per-step latency histogram；
feature/cert/payload cost stacked bar；
active-step launch count histogram。
```

---

# Part VII. Conditional downstream

## P16. Conditional paired replay

只有 P14 和 P15 都过，才打开。

### 必须记录

```text
RealFunctional vs AdamWParallel
RealFunctional vs bestLR
RealFunctional vs NoOp
RealFunctional vs Random
RealFunctional vs ShuffledPayload
CE / NLL / ECE / margin / acc / hard-tail
h20 / h80 / h240
```

### Pass 标准

```text
RealFunctional beats AdamWParallel >= 0.60
RealFunctional beats bestLR >= 0.60
NoOp/Random/ShuffledPayload 不通过
V_integrated_lcb > 0
bad/null/longrisk gate still pass
```

---

## P17. Conditional short/full training boundary

只有 P16 过，才进入 short/full training。

### 指标

```text
train acc
val/test acc
CE
NLL
ECE
CEp99
margin_p10
time_to_target
steps_to_target
sample_efficiency
forgetting
robustness
runtime
memory
```

### MLP 对照

```text
MatchedMLP
AdamWStrongLRGridMLP
QuadraticFeatureMLP
LQ base no functional
LQ + APGR functional
```

### Pass 标准

```text
functional LQ beats LQ base in at least one non-acc metric；
functional LQ does not lose against StrongLRGridMLP on main metrics；
functional improvement not caused by dataset-specific tuning；
leave-dataset-out not catastrophic。
```

---

# 8. Route decision table

| Route | 条件 | 下一步 |
|---|---|---|
| R0-BoundaryRegression | P0 不能复现 v9.5.4 | 修 artifact / runner |
| R1-GeometryLedgerFail | P1 ledger 不完整 | 修 ledger |
| R2-CleanGeometryTargetTooSparse | T5/T-like clean target 存在但 count < 87 | 转向 generator 增密 |
| R3-NoDeployableSignal | AUC 和 TopK 都失败 | 换 action space / primitive |
| R4-HighAUCLowTopK | AUC 高但 TopK 失败 | 做 score calibration / risk gating |
| R5-CoverCurvatureMemoryDamage | cover/curvature/memory 任一主导失败 | 加对应 guard |
| R6-APGRImplementationFail | APGR payload/apply/materializer 未闭合 | 修 implementation |
| R7-APGRValueFail | APGR 生成合法但 value/risk 不过 | 重设 generator objective |
| R8-CertificateFail | APGR 有好动作但 certificate 选不中 | 重做 certificate |
| R9-ControllerFail | certificate 过但 heldout/leaveout fail | controller split / support 修复 |
| R10-RuntimeFail | controller 过但 step_ratio_q90 > 1.50 | runtime 优化 |
| R11-PairedReplayFail | system pass 但 causal controls 不过 | action value 重审 |
| R12-ShortFullFail | paired replay 过但 full training 不稳 | long-run / robustness 重审 |
| R13-SystemPass | P14-P16 过 | 进入 official short/full |

---

# 9. 实验看板

本轮所有结果必须汇总到一张 dashboard：

```text
Boundary:
  v9540 route reproduced
  ledger pass

Target:
  T5 count / coverage / V / longrisk / bad / null
  Grade A/B/C/D/E count

Signal:
  SC7 AUC
  SC7 TopK64 precision
  SNR TopK64 precision
  reservoir leak TopK64

Geometry:
  cover pass
  curvature pass
  memory pass
  hard-tail pass

Generator:
  APGR generated count
  APGR branch-horizon rows
  best APGR V_lcb
  best APGR longrisk
  best APGR damage

Certificate:
  best cert TopK64 precision
  best cert longrisk
  best cert ECE

Controller:
  accepted count
  coverage
  V_lcb
  bad/null/longrisk
  LDO/LSO

Runtime:
  step_ratio_q90
  memory_ratio
  payload_apply_q90

Base sentinel:
  LQ mean test acc
  MLP mean test acc
  StrongLRGridMLP mean test acc
  base_acc_used_for_controller = 0
```

---

# 10. 并行执行安排

第一批并行：

```text
P0 boundary reproduction
P1 ledger v2
P2 T5 anatomy
P4 signal upper-bound
P6 cover audit
P7 curvature audit
P8 memory audit
Base-Acc Sentinel continuation
```

第二批并行：

```text
P9 APGS failure autopsy
P10 APGR implementation
P11 APGR smoke materializer
P13 certificate v3 prefit diagnostic
P15 runtime preflight microbench
```

第三批条件执行：

```text
P12 APGR outcome
P14 minimal controller
P15 selected runtime
P16 paired replay
P17 short/full
```

原则：

```text
不要等 P2 完全结束才启动 P4/P6/P7/P8；
不要等 controller 才测 APGR apply/runtime microbench；
不要在 certificate 失败时继续跑 paired replay；
不要在 runtime 未选中时报告 official runtime。
```

---

# 11. 最终判断标准

v9.5.5 不是必须拿到 full success，但必须回答：

```text
1. T5-like clean geometry target 是真的太稀疏，还是定义不对？
2. SC7/GCERT16 高 AUC 为什么 TopK 不行？
3. signal / cover / curvature / memory 哪个最能解释 good/bad geometry？
4. APGS 到底破坏了什么？
5. APGR 是否能创造新的 positive geometry actions？
6. 如果 APGR 不能，失败是 source、signal、cover、curvature、memory、horizon 还是 runtime？
7. 有没有一个小证书可以在训练当下选出好动作？
```

只要 v9.5.5 能把这些问题回答清楚，就不是没进展。真正不能接受的是：继续换一个 APGS 变体、看一个 AUC、调一个阈值，然后又进入下一轮。

---

# 12. 本轮成功的最低定义

最低成功：

```text
Trainable Geometry Ledger v2 pass；
T5 anatomy 完整；
high-AUC-low-TopK 原因明确；
APGS damage 主因明确；
APGR1-APGR8 至少 implementation + smoke pass；
至少一个 APGR primitive 在 diagnostic weak gate 上超过 APGS；
no-fake / no-proxy / no-dataset-tuning audit pass。
```

中等成功：

```text
APGR 产生 count >= 87 的 Grade B 或 better actions；
V_integrated_lcb > 0；
h240 longrisk <= 0.10；
bad/null 过 weak gate；
certificate TopK64 precision >= 0.50 diagnostic。
```

强成功：

```text
minimal geometry controller heldout coverage in [0.03,0.15]；
V_integrated_lcb > 0；
h240 longrisk <= 0.05；
bad <= 0.05；
null <= 0.15；
selected runtime step_ratio_q90 <= 1.50。
```

只有强成功才允许打开 official paired replay。
