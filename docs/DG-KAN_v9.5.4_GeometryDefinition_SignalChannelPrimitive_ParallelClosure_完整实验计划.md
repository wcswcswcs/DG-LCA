# DG-KAN v9.5.4 Geometry Definition / Signal-Channel Primitive / Parallel Closure 完整实验计划

> 本计划基于 v9.5.3 `Geometry Quality Functional Update / Signal Channel Primitive` 的真实执行结果制定。  
> v9.5.4 不再小修 APG1、GCERT8、SNR gate、GoodGeomAction threshold，也不把 acc 当成判断 functional update 好坏的唯一标准。  
> 本轮要解决的是更根本的问题：
>
> $$
> \boxed{
> \text{什么样的参数改动，真正让网络的训练形状变好？}
> }
> $$
>
> 这里的“训练形状变好”不是一个口号，而要被拆成可以测的东西：短期有用、中期不坏、长期安全、不是噪声方向、局部覆盖不塌、曲率不炸、旧知识不忘、能在训练当下被识别、系统成本仍然低。

---

# 0. 执行摘要

v9.5.3 的真实结果是：实验链路继续推进，但没有 functional success。

```text
route = R2-NoGoodGeometryActionInCanonicalAP0
base_candidate = LQ-t2-h256
success_v9530_strict_purekan_functional = False
success_v9530_full_functional = False
success_v9530_external_ready = False
```

本轮的真实进展：

```text
GeometryCard rows = 3388
canonical AP0 rows = 2876
generated APY rows = 512
GeometryCard missing / NaN / Inf = 0
feature cost recorded = 1
commit-time fields separated = 1
APG generated actions = 512
APG branch-horizon rows = 9216 / 9216
APG unresolved exception = 0
```

这说明 v9.5.3 不是 runner 没跑通，也不是 payload / certificate / branch-horizon materializer 没闭合。它已经第一次把“几何质量”相关字段真实落盘，并且让 APG1-APG8 真实生成动作、真实跑 outcomes。

但科学结果仍然失败：

```text
GoodGeomAction count = 5
GoodGeomAction coverage = 0.0017385257301808068
GGA weak pass = 0
GGA official pass = 0
best SNR gate = Omega-batch-action
best SNR AUC_GGA = 0.5108043788672061
best SNR TopK64 GGA precision = 0.0
best APG = APG1-SNRProjectedEdgeUpdate
best APG GGA precision = 0.03125
best APG V_integrated LCB = -0.4061862277341243
best APG h240 long-risk = 0.6875
best certificate = GCERT8-MinimalControllerCertificate
best certificate AUC_GGA = 0.9899049685968893
best certificate TopK64 GGA precision = 0.125
geometry certificate weak pass = 0
system legal controller pass = 0
```

所以 v9.5.3 的核心结论不是“几何方向没用”，也不是“APG1 再调一下”。更准确的判断是：

$$
\boxed{
\text{我们终于开始测训练形状，但现在的好形状定义太稀疏，现有生成器也没有造出这种形状。}
}
$$

v9.5.4 的目标是：

```text
1. 重新定义“好几何”到底是什么；
2. 检查 canonical AP0 里是否真的有足够多这种动作；
3. 如果没有，判断是定义太严，还是动作空间真的缺；
4. 不再只用一个二元 GGA 标签，而是建立一张 Geometry Outcome Ledger；
5. 用 signal-channel / cover / curvature / memory 四类证据设计新动作；
6. 并行验证新动作、证书、运行时间、Base-Acc Sentinel；
7. 只有这些过线，才打开 controller、paired replay、short/full training。
```

---

# 1. v9.5.3 独立数据判断

## 1.1 这轮有进展，但不是能力进展

v9.5.3 的进展主要有三类。

第一，GeometryCard 真正落盘。

```text
geometry_card_rows = 3388
canonical_ap0_card_rows = 2876
generated_apy_card_rows = 512
missing_required_field_count = 0
nan_count = 0
inf_count = 0
feature_cost_recorded = 1
commit_time_fields_separated = 1
```

这说明我们现在已经可以给每个动作记录一张“训练形状卡”。这不是 final controller，但它是必要的地基。

第二，target-vs-geometry 有弱但真实的关系。

```text
best_geometry_group = T_C_StrictAllH
best_group_action_count = 16
best_group_snr_mean = 0.6534713038783501
best_group_cover_delta_mean = -0.07027188124691248
best_group_curv_delta_mean = 0.0704833897282479
best_group_forget_risk_mean = 0.05570613797463307
target_vs_geometry_pass = 1
```

这说明严格多 horizon 好动作不是完全随机的。它们在 SNR 和遗忘风险上确实有不同形状。但这还远远不够，因为 TopK 选择和生成动作都没有过。

第三，APG 工程链路闭合。

```text
APG generated actions = 512
payload/certificate hash missing = 0 / 0
action apply L∞ max = 0.0
branch-horizon rows = 9216 / 9216
unresolved exception = 0
no-transform metric diff = 0.0
```

这说明 APG1-APG8 不是没实现。问题是它们生成出来的动作没有成为好动作。

## 1.2 v9.5.3 的主失败不是 APG implementation

最好的 APG 是：

```text
APG1-SNRProjectedEdgeUpdate
GGA precision = 0.03125
V_integrated LCB = -0.4061862277341243
h240 long-risk = 0.6875
```

如果 TopK64 里 GGA precision 只有 `0.03125`，大约就是 2 个好动作；h240 long-risk 接近 69%。这不是“阈值差一点”，也不是“生成动作少了一点”。这是生成目标错了。

更直接地说：

$$
\boxed{
\text{APG 能生成合法动作，但不是能改善训练形状的动作。}
}
$$

## 1.3 GoodGeomAction 太稀疏，不能直接 official

v9.5.3 的 GoodGeomAction：

```text
GGA count = 5
GGA coverage = 0.0017385257301808068
GGA h240 long-risk = 0.0
GGA V_integrated LCB = 0.1307816569689571
GGA bad event = 0.0
GGA null event = 0.0
```

这 5 个动作本身看起来不错：long-risk 是 0，V_integrated LCB 是正的，bad/null 都是 0。但数量太少。

如果 official coverage 下限仍按 $0.03$，在 $2876$ 个 action 里至少需要：

$$
\lceil 0.03 \times 2876 \rceil = 87
$$

而现在只有 5 个：

$$
5 \ll 87
$$

所以不能把它写成 controller target pass。它只能说明：当前定义下，存在极少数几何上不错的动作，但不够组成训练系统。

## 1.4 GCERT8 的高 AUC 不能误读成证书成功

GCERT8 的结果很容易误导：

```text
AUC_GGA = 0.9899049685968893
TopK64 GGA precision = 0.125
TopK64 longrisk = 0.0
geometry_certificate_weak_pass = 0
```

AUC 很高，但 TopK64 precision 只有 `0.125`。如果 TopK64 precision 是 0.125，大约是 8 / 64。对于 online controller 来说，它不是问“全局排序好不好看”，而是问：

```text
我真正要接受的那一批动作，够不够多，够不够好，能不能保 coverage，能不能过 runtime。
```

高 AUC + 低 TopK precision 的组合说明：

$$
\boxed{
\text{GCERT8 可能能识别极少数极端样本，但不能产出可部署 accepted region。}
}
$$

所以 v9.5.4 不能继续调 GCERT8 阈值。应该先解决正例太少和目标定义问题。

## 1.5 v9.5.3 的真正结论

我不完全接受 `R2-NoGoodGeometryActionInCanonicalAP0` 的字面说法。更精确的 route 应该是：

```text
R2-GoodGeometryDefinitionTooSparseAndGeneratorNotAligned
```

原因是：

```text
1. canonical AP0 中有 5 个 GGA，不是 0；
2. strict target 组有 geometry signal；
3. memory anti-forgetting diagnostic 是过的；
4. 但 GGA 支持度太低，SNR gate 认不出，APG 生成不出，GCERT8 也不能转 controller。
```

所以这轮的本质是：

$$
\boxed{
\text{几何方向值得继续，但现在的定义、生成器和证书没有形成可用训练规则。}
}
$$

---

# 2. 相关工作的启发

## 2.1 Deep Manifold 给我们的启发

Deep Manifold 的核心说法可以转成很实用的实验语言：神经网络训练不是只把 loss 拉低，而是在移动一堆局部小片区，让它们形成稳定路径。它强调：

```text
1. 网络表示是 stacked piecewise manifolds；
2. node covers 会在训练中移动；
3. curvature 会积累；
4. plasticity 会先升后降；
5. fixed-point regions 稳定后能力才出现；
6. symmetric boundary conditions 能减少 drift 和 oscillation。
```

这对 DG-KAN 的启发是：functional update 不是“多加一块 delta 让 h20 loss 降”。它应该像一个小的数值修正，让 KAN 的 edge/basis 覆盖更健康、曲率更稳、局部路径更不抖、旧知识不被冲掉。

因此 v9.5.4 不能只看：

```text
V20, V80, V240, long-risk
```

还要看：

```text
cover entropy 是否下降太多；
edge basis 是否过度集中；
局部 Jacobian / Lipschitz 是否暴涨；
hard-tail 是否被放大；
old-family probe 是否退化；
动作是否让表示路径更稳定，而不是更僵硬。
```

## 2.2 Generalization / Signal Channel 论文给我们的启发

Generalization 论文的核心启发是：训练中有些方向会影响测试，有些方向只是在训练集里消耗误差，对测试不可见。它把这些方向分成 signal channel 和 reservoir，并提出一个简单思想：

```text
多个样本一致支持的方向更像 signal；
只有少数样本支持、方差大的方向更像 noise / memorization。
```

它的 per-parameter gate 可写成：

$$
\mu_k^2 > \frac{\sigma_k^2}{b-1}
$$

这对 DG-KAN 非常重要。我们现在的 APG / APY 生成器太像 heuristic payload builder。下一步应该让动作先通过“多样本一致支持”检查：

```text
同一 batch 多个样本支持这个 edge/basis 改动；
hard-tail 样本支持，但普通样本不反对；
动作方向和 AdamW 不冲突，或者只在 AdamW 低 SNR 的地方补充；
动作不走高方差噪声方向。
```

所以 v9.5.4 的几何定义要加入 signal channel 约束，而不是只看 outcome label。

---

# 3. 什么才是“好的训练形状”

一个好的 functional update 动作 $a$，不是单纯让 acc 涨，也不是单纯让某个 horizon 的 loss 降。它应该同时满足以下九件事。

## 3.1 短期帮忙

$$
V_{20}^{LCB}(a)>0
$$

这表示动作在短期不是纯扰动。

## 3.2 中期不坏

$$
V_{80}^{LCB}(a) \ge -\epsilon_{80}
$$

不能 h20 好，但 h80 明显坏。

## 3.3 长期安全

$$
LongRisk_{240}^{UCB}(a) \le \tau_L
$$

如果 h240 long-risk 高，就不能 official。

## 3.4 不是噪声方向

对 action 所影响的参数组 $G(a)$，记录：

$$
SNR_G(a)=\frac{\|\mu_G\|^2}{\operatorname{tr}(\Sigma_G)/(b-1)+\epsilon}
$$

其中 $\mu_G$ 是 batch 内平均梯度，$\Sigma_G$ 是 batch 内梯度方差。实际文件中应避免符号混淆，落盘字段用：

```text
grad_mean_sq_group
grad_var_trace_group
snr_group = grad_mean_sq_group / (grad_var_trace_group / (batch_size - 1) + eps)
```

pass 条件不是单独 SNR 高，而是：

```text
SNR 高；
且与 value / long-risk 有同向关系；
且 TopK 能选出好动作。
```

## 3.5 局部覆盖不塌

KAN 的 basis / edge function 要保持覆盖能力。记录：

```text
basis_activation_entropy_before / after
basis_dead_count_before / after
edge_family_entropy_before / after
cover_concentration_before / after
hard_tail_cover_count_before / after
```

要求：

$$
\Delta CoverEntropy(a) \ge -\tau_C
$$

$$
\Delta DeadBasisCount(a) \le \tau_D
$$

这不是要求 cover 永远不变，而是不允许动作把表示压进少数 basis / edge。

## 3.6 曲率不炸

记录：

```text
local_jacobian_spectral_proxy_before / after
hessian_trace_proxy_before / after
local_lipschitz_proxy_before / after
CEp99_delta
margin_p10_delta
```

要求：

$$
\Delta Curvature(a) \le \tau_K
$$

$$
\Delta CEp99(a) \le \tau_{tail}
$$

## 3.7 旧知识不忘

记录：

```text
old_family_probe_loss_delta
old_family_margin_delta
old_family_logit_drift
forget_risk
```

要求：

$$
ForgetRisk(a) \le \tau_F
$$

v9.5.3 的 memory audit diagnostic 过了，说明这个维度可能相对容易保住；但它不能单独定义好动作。

## 3.8 训练当下能识别

动作不能只能事后知道好。必须有 commit-time certificate：

```text
signal certificate
cover certificate
curvature certificate
memory certificate
cost certificate
```

要求：

```text
certificate features 不使用 future outcome；
不使用 dataset name；
不使用 validation/test；
不使用 posthoc branch result；
threshold 在 calibration split 冻结；
heldout / leaveout 不崩。
```

## 3.9 成本低

记录：

```text
feature_compute_ms
certificate_compute_ms
payload_generate_ms
payload_apply_ms
controller_step_ratio_q90
memory_ratio
```

官方要求：

$$
StepRatio_{q90} \le 1.50
$$

$$
MemoryRatio \le 1.05
$$

---

# 4. v9.5.4 总体目标

v9.5.4 的总体目标是：

$$
\boxed{
\text{把“好动作”从 outcome-only label 改成 geometry + signal + value + safety 的向量评分，并用它设计新的动作生成器。}
}
$$

v9.5.4 不追求直接 full success。最低有效推进目标是：

```text
1. Geometry Outcome Ledger 真实落盘；
2. 证明当前 GGA=5 是定义过严、动作空间不足，还是生成器没对齐；
3. 找到至少一个可解释几何维度和 V / long-risk / forgetting 有稳定关系；
4. 设计并 materialize APGS1-APGS8 geometry-signal action family；
5. 新动作 branch-horizon smoke 全量落盘；
6. 至少一个 APGS primitive 在 value / long-risk / cover / curvature / memory 上比 APG1 明显更好；
7. 几何证书不是只看 AUC，而是 TopK、coverage、long-risk、ECE 同时过 weak gate；
8. selected runtime preflight 仍不超过 step_ratio_q90 1.50；
9. 若仍失败，明确判定是：目标过严、AP0/APG 动作空间不足、legal signal 不足、还是 KAN base 几何限制。
```

---

# 5. 本轮不做什么

不做：

```text
1. 不继续调 APG1 的 SNR threshold；
2. 不继续调 GCERT8 threshold；
3. 不继续只追 AUC_GGA；
4. 不把 5 个 GGA 当 official target；
5. 不把 TopK64 precision = 0.125 写成 certificate pass；
6. 不按 MNIST / Fashion / KMNIST 调任何 rule；
7. 不用 Base-Acc Sentinel 做 controller；
8. 不打开 paired replay / short-full，除非 selected controller + runtime 已过线；
9. 不用旧 v9.3.5 outcome table；
10. 不把 diagnostic geometry score 写成 official success。
```

允许做：

```text
1. dataset-level failure diagnosis；
2. leave-dataset-out / leave-stratum-out；
3. oracle upper-bound；
4. negative controls；
5. APG/APY/APX/AP0 comparison；
6. geometry score diagnostic；
7. signal-channel raw probe；
8. APGS new primitive smoke；
9. selected runtime preflight。
```

---

# 6. 核心假设

## H1：v9.5.3 的 GGA 太稀疏，不等于几何方向失败

H1 认为：GGA count = 5 说明当前定义下正例太少，但不能说明“好几何不存在”。可能有三种情况：

```text
A. 定义太严，筛掉了许多可用动作；
B. 定义合理，但 canonical AP0 动作空间缺少这种动作；
C. 定义合理，AP0 有少量动作，但生成器无法稳定造出来。
```

H1 成立标准：

```text
存在 relaxed geometry target 或 vector target：
  action_count >= 87
  V_integrated_LCB > 0
  h240_longrisk <= 0.05
  bad_event <= 0.05
  null_event <= 0.15
  support_balance_pass = 1
```

H1 失败标准：

```text
所有 relaxed / vector geometry target 都无法达到 coverage 0.03，或一旦达到 coverage 就 bad/null/longrisk 失控。
```

## H2：严格多 horizon target 有几何信号，但当前单一 SNR gate 不够

H2 认为：T_C strict group 的 SNR / forget risk signal 说明方向对，但 Omega-batch-action 太粗。

H2 成立标准：

```text
至少一个 geometry feature group 在 leave-seed / leave-stratum 下：
  AUC >= 0.65
  TopK64 target precision >= 0.20
  TopK64 longrisk <= 0.15
  sign consistency >= 0.80
```

H2 失败标准：

```text
所有 commit-time geometry features 在 TopK 上接近随机。
```

## H3：好的动作应同时落在 signal channel 和 stable cover zone

H3 认为：只靠 signal/SNR 不够，只靠 cover/curvature 也不够。好动作应满足：

$$
Signal(a)=1
$$

$$
CoverStable(a)=1
$$

$$
CurvatureSafe(a)=1
$$

$$
MemorySafe(a)=1
$$

H3 成立标准：

```text
组合 score 比任何单项 score 的 TopK precision、long-risk、V_LCB 更好。
```

H3 失败标准：

```text
组合后正例仍极稀疏，或组合 score 仍不能区分 long-risk。
```

## H4：APG 失败是因为生成器没有解约束问题

H4 认为 APG1-APG8 仍是 heuristic。它们没有直接求解：

$$
\max_{\Delta\theta} V_{20}+V_{80}+V_{240}
$$

subject to:

$$
LongRisk_{240}\le\tau_L,
\quad
Bad\le\tau_B,
\quad
CoverCollapse\le\tau_C,
\quad
CurvatureIncrease\le\tau_K,
\quad
ForgetRisk\le\tau_F.
$$

H4 成立标准：

```text
APGS objective-solved primitive 显著优于 APG1：
  GGA-like precision higher;
  V_integrated_LCB > 0;
  h240_longrisk <= 0.15;
  cover/curv pass = 1;
  forget risk pass = 1。
```

H4 失败标准：

```text
APGS 也生成不出任何 value-positive / longrisk-safe / cover-stable action。
```

## H5：高 AUC 证书不够，必须看 TopK 和 coverage

H5 认为 GCERT8 的 AUC 很高但仍不能用，因为 TopK / coverage 不够。

H5 成立标准：

```text
新证书必须同时满足：
  AUC >= 0.70
  TopK64 precision >= 0.25
  TopK64 longrisk <= 0.10
  accepted_count >= 87 or extrapolated coverage >= 0.03
  ECE <= 0.10
  LDO / LSO sign consistency pass
```

H5 失败标准：

```text
AUC 高但 TopK / coverage / long-risk 任一失败。
```

---

# 7. 新数据对象：Geometry Outcome Ledger

v9.5.4 新建一张完整表：

```text
geometry_outcome_ledger_v9540.csv
```

每行对应一个 action。至少包含以下字段。

## 7.1 Identity fields

```text
action_id
source_action_id
candidate_id
event_id
primitive_id
primitive_family
payload_hash
certificate_hash
dataset
seed
step
family_id
stratum_id
bucket_id
```

## 7.2 Outcome fields

```text
V20_ctrl
V80_ctrl
V240_ctrl
V_integrated
V20_lcb
V80_lcb
V240_lcb
V_integrated_lcb
weak_CP_h20
weak_CP_h80
weak_CP_h240
bad_event_rate
null_event_rate
h240_longrisk
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
shuffle_control_pass
```

## 7.3 Signal-channel fields

```text
grad_mean_sq_group
grad_var_trace_group
snr_group
snr_edge_mean
snr_edge_p10
snr_edge_p90
action_projection_signal
adamw_alignment_cosine
adamw_conflict_rate
population_risk_rate_proxy
loo_transfer_proxy
```

## 7.4 Cover fields

```text
basis_activation_entropy_before
basis_activation_entropy_after
basis_activation_entropy_delta
basis_dead_count_before
basis_dead_count_after
basis_dead_count_delta
edge_family_entropy_before
edge_family_entropy_after
cover_concentration_before
cover_concentration_after
hard_tail_cover_count_before
hard_tail_cover_count_after
```

## 7.5 Curvature / stability fields

```text
jacobian_spectral_proxy_before
jacobian_spectral_proxy_after
jacobian_spectral_proxy_delta
hessian_trace_proxy_before
hessian_trace_proxy_after
hessian_trace_proxy_delta
local_lipschitz_proxy_before
local_lipschitz_proxy_after
local_lipschitz_proxy_delta
CEp99_delta
margin_p10_delta
NLL_delta
ECE_delta
```

## 7.6 Memory fields

```text
old_family_probe_loss_delta
old_family_margin_delta
old_family_logit_drift
forget_risk
old_family_fail_count
memory_probe_count
```

## 7.7 Runtime fields

```text
feature_compute_ms
certificate_compute_ms
payload_generate_ms
payload_apply_ms
controller_eval_ms
estimated_step_ratio_q90
measured_step_ratio_q90_if_selected
memory_ratio
```

---

# 8. 新标签：不要只用一个 GoodGeomAction

v9.5.4 不再只使用一个二元 GGA。改成四级标签。

## 8.1 Level A：OutcomeGood

```text
OutcomeGood = 1
```

当且仅当：

$$
V_{integrated}^{LCB}>0
$$

$$
LongRisk_{240}\le0.05
$$

$$
BadEvent\le0.05
$$

$$
NullEvent\le0.15
$$

## 8.2 Level B：SignalGood

```text
SignalGood = 1
```

当且仅当：

$$
SNR_G(a)\ge\tau_S
$$

$$
AdamWConflict(a)\le\tau_A
$$

$$
PopulationRiskRateProxy(a)>0
$$

## 8.3 Level C：GeometryStable

```text
GeometryStable = 1
```

当且仅当：

$$
\Delta CoverEntropy(a)\ge -\tau_C
$$

$$
\Delta Curvature(a)\le\tau_K
$$

$$
\Delta CEp99(a)\le\tau_{tail}
$$

$$
ForgetRisk(a)\le\tau_F
$$

## 8.4 Level D：OfficialGeoCandidate

```text
OfficialGeoCandidate = 1
```

当且仅当：

$$
OutcomeGood(a)=1
$$

$$
SignalGood(a)=1
$$

$$
GeometryStable(a)=1
$$

$$
Cost(a)\le C_{max}
$$

v9.5.3 的 GGA 很可能相当于 Level D 的过强交集。v9.5.4 必须同时报告 A/B/C/D 各自密度，避免一开始就因为 Level D 稀疏而看不见机制。

---

# 9. 实验阶段

## P0：v9.5.3 boundary reproduction

### 目标

确认 v9.5.3 的失败边界稳定，避免基于不稳定 artifact 继续。

### 必须记录

```text
route_v9530
source_route_v9520
canonical_full_control_outcome_ready
geometry_card_rows
GGA_count
GGA_coverage
best_snr_AUC_GGA
best_snr_TopK64_GGA_precision
best_apg_primitive
best_apg_GGA_precision
best_apg_V_integrated_lcb
best_apg_h240_longrisk
best_certificate_id
best_certificate_AUC_GGA
best_certificate_TopK64_GGA_precision
best_certificate_TopK64_longrisk
system_legal_controller_pass
fake_data_used
proxy_row_used
cpu_offload_used
```

### 判断标准

P0 pass：

```text
route_v9530 = R2-NoGoodGeometryActionInCanonicalAP0
geometry_card_pass = 1
apg_implementation_pass = 1
apg_preflight_branch_horizon_pass = 1
system_legal_controller_pass = 0
fake/proxy/offload = 0
```

### 可视化

```text
p0_v9530_boundary_ladder.svg
p0_gga_density_vs_target_density.svg
p0_apg_vs_ap0_outcome_scatter.svg
p0_certificate_auc_vs_topk.svg
```

---

## P1：Geometry Outcome Ledger construction

### 目标

为 AP0 / APY / APG / future APGS actions 建统一表，避免每轮各自算一套字段。

### 输入

```text
canonical AP0 full table v9480/v9490+
APY outcomes v9520
APG outcomes v9530
GeometryCard v9530
Base-Acc Sentinel manifest
```

### 必须记录

完整记录第 7 节所有字段。

### 判断标准

P1 pass：

```text
ledger_rows >= 3388
required_field_missing_count = 0
nan_count = 0
inf_count = 0
identity_join_pass = 1
commit_time_field_separation_pass = 1
feature_cost_recorded = 1
outcome_field_complete = 1
```

### 可视化

```text
p1_geometry_ledger_field_coverage_heatmap.svg
p1_action_family_count_bar.svg
p1_signal_cover_curv_memory_pairplot.svg
p1_missing_field_zero_audit.svg
```

---

## P2：Target-density and geometry-lattice audit

### 目标

检查 v9.5.3 的 GGA 稀疏，到底是定义太严还是动作空间真的缺。

### 设置

构造多组 target：

```text
T0: OutcomeGood only
T1: OutcomeGood + SignalGood
T2: OutcomeGood + GeometryStable
T3: SignalGood + GeometryStable
T4: OutcomeGood + SignalGood + GeometryStable
T5: relaxed T4 with V80 non-negative only
T6: hard-tail-safe target
T7: memory-safe target
T8: strict official target
```

每个 target 都扫多个阈值：

```text
V_integrated_lcb threshold
h240 longrisk threshold
bad/null threshold
snr threshold
cover entropy threshold
curvature threshold
forget risk threshold
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
bad_event_rate
null_event_rate
snr_mean
cover_delta_mean
curv_delta_mean
forget_risk_mean
support_family_count
support_stratum_count
support_balance_pass
jaccard_with_v9530_GGA
jaccard_with_T_C
```

### 判断标准

Weak geometry target pass：

```text
action_count >= 32
V_integrated_lcb > 0
h240_longrisk <= 0.10
bad_event_rate <= 0.10
null_event_rate <= 0.20
support_family_count >= 16
```

Official geometry target pass：

```text
action_count >= 87
coverage >= 0.03
V_integrated_lcb > 0
h240_longrisk <= 0.05
bad_event_rate <= 0.05
null_event_rate <= 0.15
support_balance_pass = 1
```

### 可视化

```text
p2_target_lattice_density_quality_frontier.svg
p2_target_jaccard_heatmap.svg
p2_target_value_risk_tradeoff.svg
p2_target_support_balance.svg
p2_gga_relaxation_path.svg
```

---

## P3：Signal-channel raw upper-bound audit

### 目标

检查“多样本一致支持”的信号能否解释好动作。不是只看一个 Omega-batch-action。

### Features

```text
SC1: parameter-group SNR
SC2: edge-family SNR
SC3: hard-tail-only SNR
SC4: normal-vs-hardtail agreement
SC5: AdamW-compatible SNR
SC6: AdamW-conflict relief SNR
SC7: leave-one-out transfer proxy
SC8: off-diagonal batch agreement
SC9: signal/reservoir projection proxy
SC10: SNR × cover-stability interaction
```

### 必须记录

```text
feature_id
feature_group
compute_ms_q50/q90
AUC_OutcomeGood
AUC_GeometryStable
AUC_OfficialGeoCandidate
AUC_LongRisk
TopK16_precision
TopK32_precision
TopK64_precision
TopK128_precision
TopK64_longrisk
TopK64_V_integrated_lcb
sign_consistency_leave_seed
sign_consistency_leave_dataset
sign_consistency_leave_stratum
```

### 判断标准

P3 weak pass：

```text
AUC_OfficialGeoCandidate >= 0.65
TopK64_precision >= 0.15
TopK64_longrisk <= 0.20
compute_ms_q90 <= 0.25
sign_consistency_leave_seed >= 0.70
```

P3 strong pass：

```text
AUC_OfficialGeoCandidate >= 0.75
TopK64_precision >= 0.25
TopK64_longrisk <= 0.10
compute_ms_q90 <= 0.15
sign_consistency_leave_dataset >= 0.70
sign_consistency_leave_stratum >= 0.70
```

### 可视化

```text
p3_signal_feature_auc_bar.svg
p3_signal_topk_precision_curve.svg
p3_signal_longrisk_topk_curve.svg
p3_snr_vs_value_scatter.svg
p3_signal_leaveout_stability.svg
```

---

## P4：Cover / curvature / hard-tail stability audit

### 目标

判断网络的局部覆盖和曲率是否能解释 long-risk 与 value。

### Metrics

```text
cover_entropy_delta
basis_dead_count_delta
edge_family_entropy_delta
cover_concentration_delta
hard_tail_cover_count_delta
jacobian_spectral_proxy_delta
hessian_trace_proxy_delta
local_lipschitz_proxy_delta
CEp99_delta
margin_p10_delta
```

### 必须记录

```text
metric_id
metric_group
mean_by_target
median_by_target
p10/p90_by_target
effect_size_OfficialGeoCandidate_vs_rest
AUC_OfficialGeoCandidate
AUC_LongRisk
TopK64_precision
TopK64_longrisk
monotone_sign_pass
leaveout_sign_pass
```

### 判断标准

P4 pass：

```text
at least one cover metric and one curvature metric:
  AUC_LongRisk >= 0.65
  monotone_sign_pass = 1
  leaveout_sign_pass = 1

and combined cover+curvature score:
  TopK64_OfficialGeoCandidate_precision >= 0.15
  TopK64_longrisk <= 0.15
```

### 可视化

```text
p4_cover_entropy_delta_by_target.svg
p4_curvature_delta_by_target.svg
p4_longrisk_vs_curvature_scatter.svg
p4_hardtail_cover_shift.svg
p4_cover_curvature_combined_frontier.svg
```

---

## P5：Memory / anti-forgetting audit v2

### 目标

v9.5.3 的 memory diagnostic 过了，但要确认它是不是只是“所有动作都不太忘”，还是能帮助选动作。

### 必须记录

```text
forget_risk
old_family_fail_count
old_family_probe_loss_delta
old_family_margin_delta
old_family_logit_drift
new_family_gain
old_new_tradeoff
AUC_OfficialGeoCandidate
AUC_LongRisk
TopK64_precision
TopK64_forget_risk
```

### 判断标准

P5 pass：

```text
forget_risk_mean <= 0.08
old_family_fail_count = 0 in selected TopK
memory metric does not conflict with V_integrated:
  corr(memory_safe_score, V_integrated) >= -0.10
```

P5 strong pass：

```text
memory score improves TopK64 longrisk or TopK64 forgetting when added to signal+cover score.
```

### 可视化

```text
p5_memory_forget_histogram.svg
p5_old_new_tradeoff_scatter.svg
p5_memory_score_added_value.svg
p5_family_forgetting_heatmap.svg
```

---

## P6：Geometry score assembly

### 目标

把 signal / cover / curvature / memory / outcome 组合成一个可解释 score，不允许变成 50-feature 黑箱。

### Score form

每个 candidate score 最多 8 个 terms：

$$
GeoScore(a)
=
+w_V VScore(a)
+w_S SignalScore(a)
+w_C CoverScore(a)
-w_K CurvatureRisk(a)
-w_L LongRiskScore(a)
-w_F ForgetRisk(a)
-w_N NullRisk(a)
-w_T Cost(a)
$$

要求：

```text
w_i >= 0
feature_group_count <= 8
threshold frozen on calibration
no dataset-specific branch
```

### 必须记录

```text
score_id
included_terms
weights
calibration_split
threshold
accepted_count_cal/heldout
coverage_cal/heldout
precision_OutcomeGood
precision_OfficialGeoCandidate
bad_event_rate
null_event_rate
h240_longrisk
V_integrated_lcb
support_balance_pass
TopK precision curves
LDO/LSO diagnostic
```

### 判断标准

P6 weak pass：

```text
accepted_count_heldout >= 32
V_integrated_lcb > 0
h240_longrisk <= 0.10
bad_event <= 0.10
null_event <= 0.20
support_balance_pass = 1
```

P6 official pass：

```text
accepted_count_heldout >= 87
coverage in [0.03, 0.15]
V_integrated_lcb > 0
h240_longrisk <= 0.05
bad_event <= 0.05
null_event <= 0.15
support_balance_pass = 1
```

### 可视化

```text
p6_geoscore_quality_coverage_frontier.svg
p6_geoscore_terms_ablation.svg
p6_geoscore_calibration_heldout_drift.svg
p6_geoscore_family_balance.svg
```

---

## P7：APGS1-APGS8 geometry-signal primitive implementation

### 目标

不再让 generator 只是合法 payload builder。APGS 必须直接根据 signal / cover / curvature / memory 生成动作。

### Primitive families

```text
APGS1-SignalChannelEdgeMask
  只更新高 SNR edge/basis group。

APGS2-AdamWCompatibleSignalResidual
  只在 AdamW 低 SNR 或低覆盖区域补 residual，避免和 AdamW 主方向冲突。

APGS3-CoverBalanceEdgeUpdate
  生成动作时约束 basis entropy，不允许 dead basis 增加。

APGS4-CurvatureClippedLocalUpdate
  对局部 Jacobian / Hessian proxy 做 clip，防止曲率暴涨。

APGS5-HardTailSafeRepair
  只修 hard-tail CEp99 / margin_p10，但要求普通样本 loss 不明显变坏。

APGS6-MemoryAnchoredSignalUpdate
  生成动作时加入 old-family probe anchor，减少遗忘。

APGS7-SymmetricBoundaryResidualUpdate
  以 AdamW direction 和 functional residual 两侧约束，避免单向 drift。

APGS8-NegativeControlRandomSignalMatched
  norm / sparsity / cost matched random control。
```

### 必须记录

```text
primitive_id
generated_action_count
payload_hash_missing
certificate_hash_missing
action_apply_error_linf_max
action_apply_error_relative_max
cosine_logged_applied_min
feature_compute_ms
payload_generate_ms
certificate_compute_ms
```

### 判断标准

P7 implementation pass：

```text
primitive_count = 8
generated_action_count >= 512
payload_hash_missing = 0
certificate_hash_missing = 0
action_apply_error_linf_max <= 1e-6
negative_control_materialized = 1
```

### 可视化

```text
p7_apgs_payload_norm_by_primitive.svg
p7_apgs_sparsity_by_primitive.svg
p7_apgs_apply_error.svg
p7_apgs_generation_cost.svg
```

---

## P8：APGS branch-horizon smoke outcome

### 目标

真实测 APGS 生成动作是否比 APG / APY 更接近好训练形状。

### 设置

```text
APGS primitives = APGS1-APGS8
actions per primitive = 64
branches = RealAPGS, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledAPGS, CertificatePassNoPayload
horizons = 20, 80, 240
```

### 必须记录

```text
primitive_id
action_count
branch_horizon_rows_expected
branch_horizon_rows_actual
completion_rate
unresolved_exception_count
V20_lcb
V80_lcb
V240_lcb
V_integrated_lcb
h240_longrisk
bad_event
null_event
OutcomeGood_precision
OfficialGeoCandidate_precision
cover_curvature_pass
memory_pass
beats_controls_rate
negative_control_pass
```

### 判断标准

APGS weak pass：

```text
at least one primitive:
  OfficialGeoCandidate_precision >= 0.10
  V_integrated_lcb > 0
  h240_longrisk <= 0.20
  bad_event <= 0.10
  null_event <= 0.20
  negative_control_pass = 1
```

APGS strong pass：

```text
at least one primitive:
  OfficialGeoCandidate_precision >= 0.20
  V_integrated_lcb > 0
  h240_longrisk <= 0.10
  bad_event <= 0.05
  null_event <= 0.15
  cover_curvature_pass = 1
  memory_pass = 1
```

### 可视化

```text
p8_apgs_value_risk_by_primitive.svg
p8_apgs_horizon_profiles.svg
p8_apgs_geometry_metrics_by_primitive.svg
p8_apgs_vs_apg_vs_apy.svg
p8_negative_control_comparison.svg
```

---

## P9：Source-to-generated geometry damage matrix

### 目标

判断 APGS 是保留了好 source，还是继续像 AP0b/AP0f、APX、APY 一样破坏 source。

### 必须记录

```text
source_action_id
generated_action_id
primitive_id
source_OutcomeGood
generated_OutcomeGood
source_GeometryStable
generated_GeometryStable
source_V_integrated
generated_V_integrated
source_longrisk
generated_longrisk
source_cover_delta
generated_cover_delta
source_curv_delta
generated_curv_delta
source_forget_risk
generated_forget_risk
damage_value
damage_geometry
damage_longrisk
positive_preserved
new_positive_created
longrisk_created
```

### 判断标准

P9 pass：

```text
positive_preserved_rate >= 0.50
longrisk_created_rate <= 0.10
new_positive_created_rate >= 0.05
Damage_integrated_LCB >= -0.05
```

### 可视化

```text
p9_source_to_generated_damage_heatmap.svg
p9_positive_preservation_by_primitive.svg
p9_longrisk_created_by_primitive.svg
p9_damage_value_vs_geometry.svg
```

---

## P10：Geometry certificate v2

### 目标

训练当下选动作。证书必须是 effect certificate，不是 construction certificate。

### Certificate features

```text
signal_score
cover_score
curvature_risk
forget_risk
longrisk_proxy
cost_score
adamw_compatibility
hard_tail_safety
```

### 必须记录

```text
certificate_id
feature_groups
feature_count
AUC_OutcomeGood
AUC_OfficialGeoCandidate
AUC_LongRisk
TopK16_precision
TopK32_precision
TopK64_precision
TopK128_precision
TopK64_longrisk
TopK64_V_integrated_lcb
ECE
Brier_score
calibration_slope
leave_dataset_out_auc
leave_stratum_out_auc
cost_ms_q90
```

### 判断标准

P10 weak pass：

```text
AUC_OfficialGeoCandidate >= 0.70
TopK64_precision >= 0.15
TopK64_longrisk <= 0.15
TopK64_V_integrated_lcb > 0
ECE <= 0.15
cost_ms_q90 <= 0.25
```

P10 strong pass：

```text
AUC_OfficialGeoCandidate >= 0.80
TopK64_precision >= 0.25
TopK64_longrisk <= 0.10
TopK64_V_integrated_lcb > 0
ECE <= 0.10
leave_dataset_out_auc >= 0.65
leave_stratum_out_auc >= 0.65
cost_ms_q90 <= 0.15
```

### 可视化

```text
p10_certificate_reliability_curve.svg
p10_certificate_topk_precision.svg
p10_certificate_topk_longrisk.svg
p10_certificate_leaveout_matrix.svg
p10_certificate_ablation.svg
```

---

## P11：Minimal geometry controller

### 目标

只在 P6 / P8 / P10 至少 weak pass 后打开。形成一个不按 dataset 调参的 minimal controller。

### Controller form

$$
Accept(a)=1
$$

当且仅当：

$$
Certificate(a)\ge\tau_{cert}
$$

$$
LongRiskProxy(a)\le\tau_L
$$

$$
Cost(a)\le C_{max}
$$

$$
Support(a)\ge\tau_S
$$

### 必须记录

```text
controller_id
selected_primitive_id
selected_certificate_id
thresholds
calibration_split
heldout_split
accepted_count
coverage
OutcomeGood_precision
OfficialGeoCandidate_precision
bad_event
null_event
h240_longrisk
V_integrated_lcb
support_balance_pass
family_count
stratum_count
dataset_name_used
future_outcome_used
validation_test_used
```

### 判断标准

P11 pass：

```text
coverage in [0.03, 0.15]
OfficialGeoCandidate_precision >= 0.75 or OutcomeGood_precision >= 0.75
bad_event <= 0.05
null_event <= 0.15
h240_longrisk <= 0.05
V_integrated_lcb > 0
support_balance_pass = 1
dataset_name_used = 0
future_outcome_used = 0
validation_test_used = 0
```

### 可视化

```text
p11_controller_quality_coverage_frontier.svg
p11_controller_cal_heldout_drift.svg
p11_controller_family_stratum_balance.svg
p11_controller_failure_modes.svg
```

---

## P12：Selected runtime preflight and official runtime

### 目标

只有 P11 pass 后打开。测 selected controller 的真实 online path。

### 必须记录

```text
controller_id
selected_primitive_id
selected_certificate_id
online_step_count
active_step_count
accepted_action_count
feature_compute_ms_q90
certificate_compute_ms_q90
payload_generate_ms_q90
payload_apply_ms_q90
controller_eval_ms_q90
kernel_count
sync_count
step_time_q90
mlp_step_time_q90
step_ratio_q90
memory_ratio
audit_outside_timed_path
cpu_offload_used
proxy_row_used
```

### 判断标准

P12 pass：

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
audit_outside_timed_path = 1
cpu_offload_used = 0
proxy_row_used = 0
```

### 可视化

```text
p12_runtime_waterfall.svg
p12_step_ratio_by_step_bucket.svg
p12_payload_apply_time.svg
p12_memory_trace.svg
p12_controller_cost_vs_quality.svg
```

---

## P13：Leave-dataset-out / leave-stratum-out

### 目标

确认 controller 不是 dataset-specific，也不是某个 family/stratum 的偶然。

### 设置

```text
LDO:
  train/calibrate on two datasets, evaluate third.

LSO:
  hold out one stratum or family bucket.
```

### 必须记录

```text
split_type
heldout_id
accepted_count
coverage
OfficialGeoCandidate_precision
bad_event
null_event
h240_longrisk
V_integrated_lcb
step_ratio_q90
beats_adamwparallel
beats_bestlr
shuffle_control_pass
```

### 判断标准

P13 pass：

```text
at least 2 / 3 LDO splits:
  h240_longrisk <= 0.10
  V_integrated_lcb >= -0.02
  bad_event <= 0.10

at least 70% LSO splits:
  task_safe = 1
  shuffle_control_pass = 1
```

Strong pass：

```text
all LDO splits pass weak criteria
mean V_integrated_lcb > 0
```

### 可视化

```text
p13_ldo_matrix.svg
p13_lso_matrix.svg
p13_leaveout_value_risk_frontier.svg
p13_dataset_tuning_audit.svg
```

---

## P14：Official paired replay

### 目标

只有 P11-P13 pass 后打开。验证 RealFunctional 是否真的优于 controls。

### Branches

```text
RealFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledPayload
CertificatePassNoPayload
```

### Horizons

```text
20, 80, 240
```

### 必须记录

```text
action_id
controller_id
branch_id
horizon
V_ctrl
CEp99_delta
margin_p10_delta
NLL_delta
ECE_delta
bad_event
null_event
longrisk
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
shuffle_control_pass
```

### 判断标准

P14 pass：

```text
RealFunctional beats AdamWParallel >= 0.55
RealFunctional beats bestLR >= 0.55
RealFunctional beats NoOp >= 0.60
ShuffledPayload pass rate <= 0.10
bad_event <= 0.05
h240_longrisk <= 0.05
```

### 可视化

```text
p14_paired_replay_branch_matrix.svg
p14_real_vs_controls_value.svg
p14_horizon_value_profile.svg
p14_shuffle_negative_control.svg
```

---

## P15：Short/full training boundary

### 目标

只有 P14 pass 后打开。检查 functional update 是否在真实训练里有持续收益。

### Metrics

```text
test_acc
val_acc
train_acc
CE
NLL
ECE
CEp99
margin_p10
time_to_target
steps_to_target
sample_efficiency
forgetting
robust_noise_acc
runtime
memory
```

### Baselines

```text
LQ-t2-h256 base AdamW
MatchedMLP
AdamWStrongLRGridMLP
QuadraticFeatureMLP
NoFunctional
RandomFunctional
ShuffledFunctional
```

### 判断标准

Short-run pass：

```text
no catastrophic fail
RealFunctional not worse than base by > 0.005 acc
CEp99 not worse than base by > epsilon
runtime pass maintained
```

Full-run diagnostic pass：

```text
RealFunctional beats LQ base or improves sample efficiency;
RealFunctional does not lose to StrongLRGridMLP on all datasets;
calibration / CEp99 / forgetting not worse.
```

Official full success 暂不在 v9.5.4 强求，除非 P11-P14 全部提前通过。

### 可视化

```text
p15_acc_curves.svg
p15_loss_curves.svg
p15_calibration_curves.svg
p15_time_to_target.svg
p15_runtime_memory_curves.svg
p15_baseline_comparison_matrix.svg
```

---

# 10. 并行执行计划

为了避免一轮只清一个 blocker，v9.5.4 必须并行。

## Batch A：truth and target

```text
P0 boundary reproduction
P1 Geometry Outcome Ledger
P2 target-density and geometry-lattice audit
```

这些可以最先跑，目的是判断 GGA=5 是不是定义问题。

## Batch B：raw mechanism probes

```text
P3 signal-channel upper-bound
P4 cover/curvature audit
P5 memory audit
```

这三个并行跑，互不阻塞。

## Batch C：new primitive implementation

```text
P7 APGS1-APGS8 implementation
P8 APGS branch-horizon smoke
P9 source-to-generated damage
```

P7 可以和 Batch A/B 同时准备；P8 等 P7 preflight 通过后跑。

## Batch D：certificate / controller / runtime

```text
P10 geometry certificate
P11 minimal controller
P12 selected runtime
```

只有 P6/P8/P10 至少 weak pass 后才打开 P11。

## Batch E：downstream only if system pass

```text
P13 leaveout
P14 paired replay
P15 short/full
```

不得提前 official。

---

# 11. Stop / pivot rules

## Stop 1：如果所有 relaxed geometry targets 都太稀疏

条件：

```text
P2 没有任何 target action_count >= 32 且 V_integrated_lcb > 0 且 longrisk <= 0.10。
```

判断：

```text
当前 AP0/APY/APG action universe 没有足够 geometry-improving actions。
```

下一步：

```text
停止 selector/certificate，重设 action generator；不要继续调 target threshold。
```

## Stop 2：如果 raw signal / cover / curvature 全部不区分

条件：

```text
P3/P4/P5 所有 features TopK64 precision 接近 base rate。
```

判断：

```text
当前 commit-time features 看不见好动作。
```

下一步：

```text
从 selector 路线转向 self-generating / self-certifying primitive。
```

## Stop 3：如果 APGS 仍和 APG 一样 long-risk 高

条件：

```text
best APGS h240_longrisk > 0.50 and V_integrated_lcb < 0。
```

判断：

```text
新 primitive 没有真正解几何问题。
```

下一步：

```text
不调 APGS threshold；改成更低层 edge/basis update objective，或回到 base architecture geometry。
```

## Stop 4：如果 certificate AUC 高但 TopK 不行

条件：

```text
AUC >= 0.80 but TopK64 precision < 0.15。
```

判断：

```text
证书只会排序极端点，不能用于 accepted region。
```

下一步：

```text
停止 certificate promotion，重构 target / feature / generator。
```

---

# 12. 最终 route 决策

v9.5.4 的 route 必须落在以下之一。

```text
R0-BoundaryReproductionFailed
R1-GeometryLedgerFailed
R2-GeometryTargetTooSparse
R3-LegalGeometrySignalAbsent
R4-APGSPrimitiveValueFail
R5-CertificateTopKFail
R6-ControllerQualityFail
R7-SelectedRuntimeFail
R8-LeaveoutFail
R9-PairedReplayFail
R10-GeometryFunctionalLocalPass
```

其中 `R10-GeometryFunctionalLocalPass` 的最低条件：

```text
P1 pass
P2 weak or official target pass
P8 APGS weak/strong pass
P10 certificate weak/strong pass
P11 controller pass
P12 runtime pass
```

如果 P11/P12 没过，不能写 local functional pass。

---

# 13. v9.5.4 预期结论格式

最终复盘必须明确回答：

```text
1. GGA=5 是定义太严，还是 action universe 太弱？
2. 哪些几何指标和 value / long-risk / forgetting 有真实关系？
3. signal-channel SNR 是否能帮助识别好动作？
4. cover / curvature / hard-tail 是否解释 APY/APG long-risk？
5. APGS 是否比 APG/APY 更好？
6. 证书是否从 AUC-only 变成 TopK-useful？
7. selected controller 是否存在？
8. runtime 是否过线？
9. Base-Acc Sentinel 是否仍健康？
10. 下一步是 controller、generator、architecture，还是 target reset？
```

---

# 14. 最终判断标准

v9.5.4 不是为了刷 MNIST / Fashion-MNIST / KMNIST acc。Base-Acc Sentinel 继续保留，但不能进入 controller。

本轮真正的成功不是“acc 涨”，而是：

$$
\boxed{
\text{我们能否把好动作定义成一个可测、可生成、可筛选、可运行的训练形状。}
}
$$

如果 v9.5.4 能证明：

```text
1. 好训练形状在 action universe 中存在；
2. 它不只是 outcome-only label；
3. 它能被 signal / cover / curvature / memory 解释；
4. APGS 能生成更多这种动作；
5. 证书能在 TopK 里选中；
6. runtime 仍可控；
```

那项目就真正从 “target / primitive / certificate 全失败” 进入 “geometry-aware functional controller” 阶段。

如果 v9.5.4 失败，也必须明确失败在哪里：

```text
好几何太稀疏；
commit-time signal 看不见；
APGS 造不出来；
证书 TopK 不行；
runtime 太慢；
或者 base KAN 几何本身不足。
```

这比继续调 APG1 / CERT30 / GCERT8 更有价值。
