# DG-KAN v9.5.1 Primitive Family Reset / Multi-Horizon Objective / Constructive Certificate 完整实验计划

> 本计划基于 v9.5.0 `Canonical Frontier Mechanism / Self-Certifying Primitive / Parallel Closure` 的真实复盘结果制定。  
> v9.5.1 不继续调 `NegHardTailFraction`、`UB2`、`SG1-SG4`、`CERT14-CERT18` 的阈值，也不继续把 AP0 opaque action 当作可以靠事后 legal feature 识别的对象。  
> 本轮目标是重置 primitive family：先把目标标签拆成多 horizon / 多控制分支的向量目标，再判断 robust action 的 commit-time 信息上界是否真实存在；如果 existing AP0 selection 仍不可见，则直接构造新的 objective-solved / effect-certified primitive。

---

# 0. 执行摘要

v9.5.0 的结果可以用一句话概括：

$$
\boxed{\text{canonical robust frontier 存在，但当前 legal probe、mechanism、generator、certificate 全部抓不住它。}}
$$

这不是 truth base 问题。v9.4.8 recovery / v9.4.9 已经把 canonical full outcome universe 做到了 `51768 / 51768` rows，并且 quality audit pass。v9.5.0 继续复现了这个 boundary。现在失败点非常集中：

```text
canonical truth base ready = 1
YRobust_count = 120
YStableHorizon_count = 113
YStrictAllH_count = 16
legal_upper_bound_probe_pass = 0
mechanism_pass = 0
constructive_generator_pass = 0
certificate_effect_valid_pass = 0
source_controller_pass = 0
selected_runtime_pass = 0
system_legal_controller_pass = 0
```

这轮不能被解读为“没有进展”。真实进展是：

```text
1. YRobust / YStableHorizon / YStrictAllH 的关系被量化；
2. YRobust 的 h80 blind spot 被暴露；
3. high-capacity legal probe 失败，说明当前 legal representation 上界很低；
4. mechanism anatomy 失败，说明当前 positive frontier 没有稳定 scalar/cluster pattern；
5. SG1-SG4 constructive generators 真实 materialize 但 value/risk 失败；
6. CERT14-CERT18 真实评估但没有 effect-valid separation；
7. runtime preflight 可以通过，但没有 controller，所以不能 official。
```

但从系统能力角度看，用户觉得“慢、没进度”是合理的，因为还没有进入：

```text
selected controller；
selected runtime；
LDO/LSO；
official paired replay；
short/full functional training；
continual / anti-forgetting。
```

v9.5.1 的核心转向是：

```text
从：
  现有 AP0/APx action 上找 legal selector 或调 certificate

转为：
  重新定义 multi-horizon robust objective，
  然后实现能从 construction 产生 effect certificate 的新 primitive family。
```

---

# 1. 对 v9.5.0 的独立判断

## 1.1 v9.5.0 有进展，但不是能力进展

v9.5.0 不是 functional success：

```text
route = R9-PrimitiveFamilyResetRequired
success_v9500_strict_purekan_functional = False
success_v9500_full_functional = False
success_v9500_external_ready = False
```

但它的进展很关键。它把问题从：

```text
canonical truth base 是否可信？
canonical robust frontier 是否存在？
现有 legal feature 是否太弱？
```

推进到：

```text
即使 canonical truth base 和 robust frontier 都成立，
当前 legal upper-bound probe、mechanism anatomy、constructive generator、certificate 仍全部失败。
```

这说明下一步不是继续修表、修 replay、修 source panel，也不是继续调一个阈值，而是必须重置 primitive family。

## 1.2 当前最深的科学问题

v9.5.0 暴露出一个比 “legal feature weak” 更深的问题：当前目标标签本身不是一个简单的一维 robust label。

关键数据：

```text
YRobust_count = 120
YStableHorizon_count = 113
YStrictAllH_count = 16
Jaccard_YRobust_YStableHorizon = 0.9416666666666667
Jaccard_YRobust_YStrictAllH = 0.13333333333333333
P_h80_weak_given_YRobust = 0.175
P_h80_bad_given_YRobust = 0.058333333333333334
V_ctrl_h20_lcb_YRobust = 0.3013685750004344
V_ctrl_h80_lcb_YRobust = -0.8039138915476802
V_ctrl_h240_lcb_YRobust = 0.08296381891423209
objective_conflict_class = OC1-YRobustHasH80BlindSpot
```

这说明 current YRobust 更像：

```text
h20 有正收益 + h240 不长风险
```

但它并不保证 h80 positive。这样做可以得到足够 density，但会带来两个问题：

```text
1. 如果 controller 学 YRobust，它可能学到 h20/h240 两端好但 h80 中间坏的 action；
2. 如果要求 StrictAllH，action 只有 16 个，density 太稀疏，难以支撑 coverage gate。
```

所以 v9.5.1 不能再只把 `YRobust` 当唯一 target。必须把目标改成 vector / Pareto objective：

$$
Y(a)=\{V_{20}(a), V_{80}(a), V_{240}(a), B_{20}(a), B_{80}(a), B_{240}(a), LR_{240}(a), S(a), C(a)\}.
$$

最终 accept 不是预测一个二值标签，而是满足：

$$
LCB(V_{20}) > 0,
$$

$$
LCB(V_{80}) \ge -\epsilon_{80},
$$

$$
LCB(V_{240}) \ge 0 \quad \text{or} \quad UCB(LongRisk_{240}) \le \tau_{LR},
$$

$$
UCB(Bad_h) \le \tau_b \quad \forall h \in \{20,80,240\},
$$

$$
LCB(Support) \ge \tau_s,
$$

$$
Cost \le C_{max}.
$$

## 1.3 当前 legal selection 路线基本被证伪到很强程度

v9.5.0 的 P2 high-capacity legal upper-bound probe 失败：

```text
UB0-ScalarFeatureProbe:
  AUC_YRobust = 0.5375105829704886
  TopK64 YRobust precision = 0.03125
  TopK64 LongRisk = 0.546875

UB1-TensorSketchCentroidProbe:
  AUC_YRobust = 0.4966134494436381
  TopK64 YRobust precision = 0.03125
  TopK64 LongRisk = 0.6875

UB2-HighCapacityLegalKNNProbe:
  AUC_YRobust = 0.51014755684567
  TopK64 YRobust precision = 0.015625
  TopK64 LongRisk = 0.65625
```

这很重要。它不是说“所有可能 legal information 都不存在”，但它说明：在当前记录的 legal representation 上，即使用高容量 KNN / centroid，也看不见 robust frontier。

因此，继续做下面这些属于小修小补：

```text
NegHardTailFraction threshold；
PayloadNorm / PayloadLinf threshold；
StateNLL / Step / CandidateId topK；
UB2 KNN distance threshold；
CERT18 threshold；
SG2 trust-region radius 微调。
```

## 1.4 当前 constructive generator 不是差一点，而是方向错了

v9.5.0 的 P5 constructive source generator v3 已经完整 materialize：

```text
generated actions = 256
branch-horizon rows = 4608
payload hash missing = 0
certificate hash missing = 0
action apply L∞ max = 0.0
```

所以这不是 implementation missing。真实结果是：

```text
best_generator = SG2-HorizonGuardedTrustRegion
h20 weak CP = 0.1875
h20 V_ctrl LCB = -1.5456723592527004
h240 long-risk = 0.71875
YRobust precision = 0.0625
constructive_generator_pass = 0
```

如果 K=64，则 `YRobust precision = 0.0625` 只对应约 4 个 robust actions；`h240 long-risk = 0.71875` 对应约 46 个 long-risk actions。这个不是 “再调一点 radius” 能解决的。

本质是：SG1-SG4 的 certificate / construction 仍然是 heuristic，不是 effect-solved。它们能生成合法 payload，但不能保证真实控制分支 outcome。

## 1.5 当前 certificate 不是 effect certificate

v9.5.0 的 P6：

```text
best_certificate = CERT18-MinimalDistilledLegalCertificate
AUC_YRobust = 0.5284178387150466
TopK64 YRobust precision = 0.015625
TopK64 LongRisk = 0.796875
certificate_effect_valid_pass = 0
```

TopK64 只有 1 个 YRobust，却有约 51 个 long-risk。一个 certificate 如果在 topK 中富集 long-risk，而不是富集 robust action，它就不是 certificate，只是 construction metadata。

所以 v9.5.1 需要从：

```text
certificate = 描述 action 怎么构造
```

转为：

```text
certificate = 对 action effect 给出可验证 lower/upper bound
```

也就是 certificate 必须至少包含：

$$
\widehat{LCB}(V_{20}),
$$

$$
\widehat{LCB}(V_{80}),
$$

$$
\widehat{UCB}(LongRisk_{240}),
$$

$$
\widehat{UCB}(Bad_h),
$$

$$
\widehat{Cost}.
$$

并且这些 certificate fields 要在 branch-horizon outcomes 上有 monotone effect validity。

## 1.6 有在数据集上训练吗？acc 如何？和 MLP 比如何？

v9.5.0 没有打开 official functional short/full training。它复用了 Base-Acc Sentinel，属于隔离健康检查，不用于 selector/generator/controller。

当前固定 sentinel 结果是：

```text
sentinel rows = 120
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0..9
model_count = 4
mean_test_acc_LQ = 0.6537760416666667
mean_test_acc_MLP = 0.562890625
mean_test_acc_QuadraticFeatureMLP = 0.40234375
mean_test_acc_AdamWStrongLRGridMLP = 0.628515625
LQ_minus_MLP = +0.09088541666666672
LQ_minus_AdamWStrongLRGridMLP = +0.025260416666666674
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
```

结论：LQ-t2-h256 base 没有 catastrophic fail，而且 fixed sentinel 下比 MatchedMLP 高约 9.09 个百分点，比 AdamWStrongLRGridMLP 高约 2.53 个百分点。但这不是 functional DG-KAN 成功，因为 selected functional controller、paired replay、short/full functional training 都没有打开。

## 1.7 当前进度判断

现在的状态不是“快成功了”，也不是“没路”。更准确是：

$$
\boxed{\text{truth base 已闭合，AP0 robust frontier 已确认，但 AP0 frontier 对当前 legal representation 是不可见的，现有 generator/certificate 也不能复现它。}}
$$

离目标还差：

```text
1. 一个真正 value-producing / horizon-safe 的 primitive family；
2. 一个 effect-valid certificate；
3. 一个 minimal legal controller；
4. selected-controller runtime；
5. LDO / LSO；
6. official paired replay；
7. short/full/sample-efficiency/continual/robustness；
8. external fair advantage。
```

---

# 2. v9.5.1 总体目标

v9.5.1 的总体目标是：

$$
\boxed{\text{重置 primitive family，使 action 在生成时就带有 multi-horizon effect certificate，而不是事后从 opaque AP0 action 中猜测。}}
$$

本轮不追求直接 full success。最低有效目标是回答四个问题：

```text
Q1. 当前 YRobust / YStableHorizon / YStrictAllH 哪个目标适合 official controller？
Q2. 如果使用更完整 raw legal tensors，高容量 upper-bound 是否仍看不见 robust action？
Q3. 如果 selection 路线失败，新的 objective-solved primitive 是否能生成 h20 value-positive 且 h80/h240 safe 的 actions？
Q4. 新 certificate 是否能在真实 branch-horizon outcomes 上变成 effect-valid sufficient statistic？
```

强目标是：

```text
primitive_family_reset_pass = 1
multi_horizon_objective_pass = 1
constructive_generator_pass = 1
certificate_effect_valid_pass = 1
source_controller_pass = 1
selected_runtime_pass = 1
```

最低可接受推进是：

```text
1. 明确当前 target label 是否必须从 YRobust 改为 multi-objective vector；
2. 明确 legal selection route 是否彻底 dead；
3. 至少一个 APX primitive 在 64-action smoke 上达到 h20 value-positive + h240 long-risk controlled；
4. 如果 APX 全失败，明确失败是 objective solver、basis restriction、horizon dynamics、还是 KAN base/action space 本身导致。
```

---

# 3. 本轮明确不做什么

v9.5.1 不做：

```text
1. 不继续调 SG1-SG4 threshold / radius / support_lcb；
2. 不继续调 CERT14-CERT18 threshold；
3. 不继续扩大 current scalar legal feature soup；
4. 不把 UB2 高容量 probe 换 K 值后继续当 selector；
5. 不把 YRobust 当唯一 target；
6. 不把 Base-Acc Sentinel 当 functional success；
7. 不按 MNIST / Fashion-MNIST / KMNIST 调 controller 或 generator；
8. 不在 controller 未选中时打开 official paired replay；
9. 不把 runtime preflight 写成 official runtime；
10. 不用 old v9.3.5 outcome table。
```

允许做：

```text
1. dataset/family/horizon diagnostic；
2. raw legal tensor upper-bound probe，但只能 diagnostic；
3. high-cost micro-rollout discovery，但不能直接 official，除非 cost 进 budget；
4. APX primitive matrix 并行 smoke；
5. certificate effect validity audit；
6. Base-Acc Sentinel continuation；
7. selected-controller runtime preflight；
8. LDO/LSO only after controller pass。
```

---

# 4. 核心定义

## 4.1 Control value

对 action $a$、branch $b$、horizon $h$，定义 outcome value：

$$
V_b(a,h)=Score_b(a,h)-Score_{NoOp}(a,h).
$$

functional 相对最强 control 的 value：

$$
V_{ctrl}(a,h)=V_{RealFunctional}(a,h)-\max_{b\in \mathcal{B}_{ctrl}}V_b(a,h).
$$

其中：

$$
\mathcal{B}_{ctrl}=\{AdamWOnly, AdamWParallel, bestLR, NoOp, Random\}.
$$

## 4.2 Multi-horizon objective vector

每个 action 的目标不再是单个 $YRobust$，而是：

$$
\mathbf{y}(a)=
\left(
V_{20},V_{80},V_{240},
Bad_{20},Bad_{80},Bad_{240},
LongRisk_{240},Support,Cost
\right).
$$

定义 integrated robust score：

$$
J_{robust}(a)=
\lambda_{20}LCB(V_{20})
+\lambda_{80}LCB(V_{80})
+\lambda_{240}LCB(V_{240})
-\beta_{LR}UCB(LongRisk_{240})
-\beta_B\max_h UCB(Bad_h)
-\beta_C Cost(a).
$$

默认权重固定为：

```text
lambda_20 = 1.0
lambda_80 = 1.0
lambda_240 = 0.5
beta_LR = 2.0
beta_B = 2.0
beta_C = 0.1
```

这些权重只用于 diagnostic ranking，不能按 dataset 调。

## 4.3 Official target candidates

本轮同时评估四个 target，不提前假定 YRobust 是唯一正确 target。

### Target A：Legacy YRobust

```text
T_A = original YRobust from v9.4.9/v9.5.0
```

要求：

```text
h20 value positive；
h240 long-risk safe；
不强制 h80 weak positive。
```

### Target B：Stable-Horizon Robust

```text
T_B = YStableHorizon
```

要求：

```text
h20 value positive；
h80 不 bad / 不 catastrophic；
h240 long-risk safe；
跨 horizon value curve 不出现大幅反转。
```

### Target C：Strict-All-Horizon

```text
T_C = YStrictAllH
```

要求：

```text
h20 / h80 / h240 全部 weak or strong positive。
```

### Target D：Integrated Robust Score

```text
T_D = J_robust(a) >= tau_J
```

不转成单一 oracle label前，先作为排序 / Pareto frontier 使用。

---

# 5. 核心假设

## H0：v9.5.0 boundary 可复现

H0 成立标准：

```text
route_v9500 = R9-PrimitiveFamilyResetRequired
canonical_full_control_outcome_ready = 1
legal_upper_bound_probe_pass = 0
constructive_generator_pass = 0
certificate_effect_valid_pass = 0
system_legal_controller_pass = 0
```

H0 失败标准：

```text
v9.5.0 artifact 不能复现；canonical table / YRobust / probe / generator / certificate metrics 与复盘差异超过 tolerance。
```

## H1：当前 YRobust 不是 sufficient official target

H1 认为 YRobust 有 density，但 h80 blind spot 会误导 controller。

H1 成立标准：

```text
P_h80_weak_given_YRobust <= 0.25
V_ctrl_h80_lcb_YRobust < 0
YStrictAllH_count / action_count < 0.03
```

H1 失败标准：

```text
YRobust 在 h20/h80/h240 均有正 LCB，且 StrictAllH density 也足够。
```

## H2：当前 legal representation 不足以 select robust frontier

H2 成立标准：

```text
raw legal upper-bound probe AUC_TB < 0.65
raw legal upper-bound TopK64 precision_TB < 0.15
raw legal upper-bound TopK64 long-risk > 0.25
LDO/LSO 不稳定或接近随机。
```

H2 失败标准：

```text
raw legal tensor upper-bound AUC >= 0.75
TopK64 robust precision >= 0.25
TopK64 long-risk <= 0.10
LDO/LSO drop <= 0.08。
```

若 H2 失败，说明当前 hand-crafted features 不够，但 legal information 存在，应进入 feature distillation。若 H2 成立，selection route 停止，进入 constructive primitive。

## H3：robust frontier 的机制不在 marginal scalar，而在 action-state-gradient interaction

H3 成立标准：

```text
marginal scalar feature max AUC < 0.60；
raw tensor interaction probe 或 local constrained optimization 能提升；
robust positive 与 gradient-action alignment / hard-tail descent / AdamW conflict relief 有稳定关系。
```

H3 失败标准：

```text
所有 raw tensor interaction、gradient-action、micro-response features 仍接近随机。
```

## H4：objective-solved primitive 可以产生 robust source actions

H4 成立标准：

在 64-action smoke 上至少一个 APX primitive 达到：

```text
h20 weak CP >= 0.50
h20 V_ctrl LCB > 0
h80 V_ctrl LCB >= -0.05
h240 long-risk <= 0.15
YStableHorizon precision >= 0.20
bad_event_h20/h80/h240 <= 0.10
payload/action apply error L∞ = 0
```

强 pass：

```text
h20 weak CP >= 0.60
h20 V_ctrl LCB > 0.10
h80 V_ctrl LCB > 0
h240 long-risk <= 0.10
YStableHorizon precision >= 0.30
```

## H5：effect certificate 可以成为 minimal controller 的 sufficient statistic

H5 成立标准：

```text
AUC_TB >= 0.75
AUC_LongRisk >= 0.75 or inverse AUC <= 0.25 with correct sign
TopK64 YStable precision >= 0.25
TopK64 long-risk <= 0.10
ECE <= 0.08
monotone sign pass = 1
certificate ablation leaves <= 5 fields
```

H5 失败标准：

```text
certificate AUC < 0.65
TopK64 robust precision near base rate
TopK64 long-risk high
calibration-to-heldout drift > 0.10
```

## H6：controller 必须来自 certificate，不来自 opaque feature soup

H6 成立标准：

minimal controller 只使用：

```text
LCB_hat_V20
LCB_hat_V80
LCB_hat_V240
UCB_hat_LongRisk240
UCB_hat_BadMax
SupportLCB
Cost
```

且满足：

```text
feature_count <= 7
all signs monotone
threshold frozen on calibration split
no dataset branch
```

## H7：runtime 只在 selected controller 后 official

H7 成立标准：

```text
selected_controller_pass = 1
payload apply path measured
controller feature/certificate compute measured
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
```

Runtime preflight 只能证明 implementation 可行，不算 system pass。

---

# 6. 数据合同

## 6.1 Canonical action table

每个 action 必须记录：

```text
action_id
candidate_id
event_id
primitive_family_id
primitive_id
payload_hash
certificate_hash
state_before_hash
optimizer_state_hash
rng_state_hash
batch_sequence_hash
branch_config_hash
horizon_config_hash
label_config_hash
dataset
seed
step
family_id
bucket_id
horizon_origin
payload_norm
payload_linf
payload_role_entropy
```

Pass：

```text
duplicate_action_id_count = 0
payload_hash_missing = 0
certificate_hash_missing = 0
action_apply_error_linf_max <= 1e-7
no_transform_equivalence_pass = 1 for NT controls
```

## 6.2 Multi-horizon outcome table

每个 action × branch × horizon row 记录：

```text
action_id
branch_id
horizon
V_branch
V_ctrl
weak_CP
strong_CP
bad_event
null_event
long_risk
CE_delta
NLL_delta
ECE_delta
margin_p10_delta
CEp99_delta
curvature_delta
horizon_state_hash
label_exclusivity_pass
```

Branch：

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

Horizon：

```text
20
80
240
```

Pass：

```text
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1.0
label_exclusivity_violation_count = 0
metric_nan_count = 0
metric_inf_count = 0
duplicate_outcome_row_id_count = 0
```

## 6.3 Certificate table

每个 generated action 必须记录：

```text
action_id
primitive_id
payload_hash
certificate_hash
certificate_schema_version
commit_time_available
uses_dataset_name
uses_future_outcome
uses_outcome_at_commit
uses_validation_or_test
cert_LCB_V20
cert_LCB_V80
cert_LCB_V240
cert_UCB_BadMax
cert_UCB_LongRisk240
cert_support_lcb
cert_cost_estimate_ms
cert_norm_bound
cert_curvature_bound
cert_linearization_error_bound
cert_adamw_conflict_score
cert_hardtail_descent_score
cert_noharm_avg_score
```

Pass：

```text
certificate_fields_complete = 1
payload_hash_binding = 1
legality_violation_count = 0
uses_dataset_name = 0
uses_future_outcome = 0
uses_outcome_at_commit = 0
```

## 6.4 Runtime table

每 selected-controller timed step 记录：

```text
step_id
active_step
candidate_count
accepted_count
certificate_compute_time_ms
controller_score_time_ms
payload_lookup_time_ms
payload_apply_time_ms
base_step_time_ms
total_step_time_ms
step_ratio
peak_memory_mb
memory_ratio
controller_kernel_launch_count
controller_sync_count
audit_outside_timed_path
```

Pass：

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
audit_outside_timed_path = 1
no_event_preservation_pass = 1
base_adamw_equivalence_on_zero_candidate_steps = 1
```

---

# 7. v9.5.1 实验阶段

---

## P0：v9.5.0 boundary reproduction

### 目标

复现 v9.5.0，不允许跳过失败边界直接进入新 primitive。

### 记录指标

```text
route_v9500
canonical_full_control_outcome_ready
YRobust_count
YStableHorizon_count
YStrictAllH_count
P_h80_weak_given_YRobust
UB2_AUC_YRobust
UB2_TopK64_YRobust_precision
UB2_TopK64_LongRisk
best_cluster_purity_YRobust
constructive_best_generator_id
constructive_best_h20_weak_CP
constructive_best_h20_V_ctrl_lcb
constructive_best_h240_longrisk
constructive_best_YRobust_precision
best_certificate_id
best_certificate_AUC_YRobust
best_certificate_TopK64_YRobust_precision
best_certificate_TopK64_LongRisk
runtime_preflight_pass
official_runtime_pass
system_legal_controller_pass
```

### 判断标准

P0 pass：

```text
v9.5.0 route reproduced；
canonical truth ready = 1；
legal/generator/certificate/controller still fail；
no fake/proxy/offload = 1。
```

### 可视化

```text
p0_v9500_route_ladder.svg
p0_probe_generator_certificate_failure_matrix.svg
p0_frontier_exists_but_unusable.svg
```

---

## P1：multi-objective label audit v2

### 目标

判断当前 `YRobust` 是否适合作为 official target，还是必须转为 multi-objective vector / integrated robust score。

### 实验设计

对 canonical AP0 full table 重算四类 target：

```text
T_A = legacy YRobust
T_B = YStableHorizon
T_C = YStrictAllH
T_D = IntegratedRobustScore thresholded frontier
```

对每个 target 计算 density、support、horizon value curve、long-risk、dataset/family/step 分布。

### 记录指标

```text
action_count
T_A_count
T_B_count
T_C_count
T_D_count
T_A_coverage
T_B_coverage
T_C_coverage
T_D_coverage
Jaccard_TA_TB
Jaccard_TA_TC
Jaccard_TB_TC
P_h20_weak_given_T*
P_h80_weak_given_T*
P_h240_weak_given_T*
P_h80_bad_given_T*
P_h240_longrisk_given_T*
V_ctrl_h20_lcb_T*
V_ctrl_h80_lcb_T*
V_ctrl_h240_lcb_T*
V_ctrl_integrated_lcb_T*
family_support_count_T*
dataset_support_count_T*
max_family_share_T*
max_dataset_share_T*
```

### 判断标准

Target 可用于 official controller 必须满足：

```text
coverage >= 0.03
support_balance_pass = 1
V_ctrl_h20_lcb > 0
V_ctrl_h80_lcb >= -0.05
h240_longrisk <= 0.10
max_family_share <= 0.50
```

如果只有 T_A 过 coverage，但 h80 LCB 大负，则不得直接用 T_A 训练 controller；改用 T_D 或 T_B。

如果 T_C 太稀疏但质量最高，则用 T_C 作为 strong diagnostic，不作为 coverage target。

### 可视化

```text
p1_target_overlap_venn.svg
p1_horizon_value_curve_by_target.svg
p1_target_density_vs_quality.svg
p1_h80_blindspot_heatmap.svg
p1_family_dataset_support_by_target.svg
```

---

## P2：raw legal information upper-bound v2

### 目标

确认 “legal selection route” 是否真的 dead。v9.5.0 的 UB2 只证明当前 legal representation 失败；P2 要加入更完整但仍 commit-time legal 的 raw tensor/features。

### Legal 输入范围

允许：

```text
current batch logits
current batch CE/margin/entropy
hard-tail sample mask from train batch only
manual gradients from current batch
AdamW delta
candidate payload tensor
KAN basis activation statistics
edge-role statistics
layerwise payload geometry
pre-update state stats
support stats from calibration split
```

禁止：

```text
dataset_name branch
validation/test metric
future outcome
post-update horizon outcome
outcome label at commit time
old v9.3.5 table
```

### Probe candidates

```text
UB0 scalar baseline from v9.5.0
UB3 raw-logit tensor probe
UB4 gradient-action bilinear probe
UB5 hard-tail response sketch
UB6 basis-edge activation interaction probe
UB7 full legal tensor random feature map
UB8 small MLP upper-bound diagnostic
```

UB8 仅为 diagnostic upper-bound，不能 official。

### 记录指标

```text
probe_id
input_group
input_dim
feature_cost_ms_q90
memory_ratio
AUC_TA
AUC_TB
AUC_TC
AUC_LongRisk
TopK16_precision_TB
TopK64_precision_TB
TopK64_longrisk
TopK273_precision_TB
LDO_AUC_drop_max
LSO_AUC_drop_max
calibration_to_heldout_drift
uses_illegal_feature_count
```

### 判断标准

Legal information upper-bound pass：

```text
AUC_TB >= 0.75
TopK64_precision_TB >= 0.25
TopK64_longrisk <= 0.10
LDO_AUC_drop_max <= 0.08
LSO_AUC_drop_max <= 0.10
```

Weak pass：

```text
AUC_TB >= 0.65
TopK64_precision_TB >= 0.15
TopK64_longrisk <= 0.25
```

若 P2 fail，则 existing-action selection route 停止，P3-P6 转向 constructive primitive。

### 可视化

```text
p2_probe_auc_bar.svg
p2_topk_precision_longrisk_tradeoff.svg
p2_raw_tensor_probe_roc_pr.svg
p2_leaveout_drop_heatmap.svg
p2_cost_vs_signal_scatter.svg
```

---

## P3：robust source mechanism anatomy v2

### 目标

不是继续 cluster scalar features，而是对 robust actions 做机制解剖：到底是 hard-tail 修复、AdamW conflict relief、basis edge specialization、还是 horizon dynamics。

### 实验设计

对 T_B / T_D positives、near misses、long-risk negatives 构建 matched triplets：

```text
positive robust action；
near positive but h80 fail；
near positive but h240 long-risk；
legal TopK false positive；
random negative。
```

计算机制维度：

```text
hard_tail_gradient_alignment
average_gradient_noharm
AdamW_conflict_relief
basis_edge_locality
payload_low_rank_structure
margin_tail_repair_vector
curvature_bound
horizon_value_slope
support_memory_similarity
```

### 记录指标

```text
mechanism_id
effect_size_positive_vs_negative
AUC_TB
AUC_LongRisk
cluster_purity_TB
cluster_longrisk_rate
prototype_reconstruction_error
mechanism_stability_LDO
mechanism_stability_LSO
miss_reason_distribution
```

### 判断标准

Mechanism pass：

```text
best_cluster_purity_TB >= 0.25
cluster_longrisk_rate <= 0.15
effect_size >= 0.50
mechanism_stability_LDO_drop <= 0.10
```

若 mechanism fail，说明 robust frontier 仍然 outcome-only relative to available semantics；不再尝试 distill old actions。

### 可视化

```text
p3_mechanism_effect_size_matrix.svg
p3_positive_near_miss_damage_sankey.svg
p3_horizon_value_slope_by_mechanism.svg
p3_mechanism_cluster_purity.svg
```

---

## P4：primitive family reset specification

### 目标

定义 APX family。APX 不是现有 AP0 action 的 transform，也不是 SG1-SG4 radius tweak；它直接从 current state / batch / gradient / KAN basis 生成 payload，并同时生成 effect certificate。

### APX primitive candidates

#### APX1：Constrained Tail Descent QP

求解：

$$
\Delta^*=
\arg\min_{\Delta\in\mathcal{S}}
\hat{g}_{tail}^\top\Delta
+\lambda\|\Delta\|_2^2
$$

subject to：

$$
\hat{g}_{avg}^\top\Delta + \frac{1}{2}\hat{L}_{avg}\|\Delta\|_2^2 \le \epsilon_{avg},
$$

$$
\|\Delta\|_{\infty} \le r_{\infty},
$$

$$
\|\Delta\|_2 \le r_2.
$$

#### APX2：AdamW-Conflict Orthogonal Residual

构造：

$$
\Delta=Proj_{\perp \Delta_{AdamW}}(-\hat{g}_{tail})
$$

并加入 no-harm constraint。

#### APX3：Horizon Guarded Two-Scale Update

构造短期 descent 与长 horizon conservative shrink：

$$
\Delta=\alpha\Delta_{short}+\beta\Delta_{stable},
$$

其中 $\alpha,\beta$ 由 fixed rule 决定，不按 dataset 调。

#### APX4：Low-Rank Edge Local Repair

只在 low-rank edge subspace 中求解：

$$
\Delta = U r V^\top,
$$

rank 固定为 1 或 2，用于限制 payload apply 成本和 long-risk。

#### APX5：Basis-Response Matched Repair

使用 KAN basis activation response，选取 hard-tail active basis 的 local repair。

#### APX6：No-Harm Conservative Shrink

如果 certificate lower bound 不足，则自动 shrink 到 no-op：

$$
\Delta'=\gamma\Delta,
$$

$$
\gamma=\min\left(1,\frac{Margin_{cert}}{Risk_{cert}+\epsilon}\right).
$$

#### APX7：Ensemble Intersection Primitive

只接受同时满足 APX1/APX3/APX4 certificate 的 intersection action。

#### APX8：Randomized Orthogonal Negative Control

与 APX2 形状相同，但方向随机正交，用于证明 outcome 不是 payload norm artifact。

### 记录指标

```text
primitive_id
generated_action_count
payload_hash_missing
certificate_hash_missing
action_apply_linf_max
certificate_fields_complete
commit_time_available
uses_dataset_name
uses_future_outcome
uses_outcome_at_commit
payload_norm
payload_linf
payload_rank
edge_role_sparsity
basis_locality_score
estimated_apply_cost_ms
```

### 判断标准

P4 pass：

```text
APX1-APX8 generated_action_count expected achieved；
payload/certificate hash missing = 0；
action_apply_linf_max <= 1e-7；
legality violation = 0；
negative control APX8 generated。
```

### 可视化

```text
p4_apx_payload_geometry.svg
p4_apx_certificate_field_completeness.svg
p4_apx_apply_error_distribution.svg
p4_apx_cost_distribution.svg
```

---

## P5：single-action / mini-panel deterministic preflight

### 目标

防止再次出现 v9.4.5-v9.4.7 那种 no-transform / runner 语义问题。所有 APX outcome 前先做 1-action、3-action、16-action preflight。

### 记录指标

```text
preflight_level
primitive_id
action_count
branch_count
horizon_count
payload_hash_match_rate
state_before_hash_match_rate
optimizer_state_hash_match_rate
rng_state_hash_match_rate
batch_sequence_hash_match_rate
horizon_state_hash_match_rate
metric_abs_diff_max
label_match_rate
negative_control_divergence_present
```

### 判断标准

P5 pass：

```text
1-action pass = 1
3-action pass = 1
16-action pass = 1
no-transform equivalence pass = 1
negative controls diverge = 1
```

### 可视化

```text
p5_preflight_ladder.svg
p5_hash_match_grid.svg
p5_negative_control_divergence.svg
```

---

## P6：APX branch-horizon smoke outcome

### 目标

真实测量 APX actions 的 matched-control branch-horizon outcomes，不用 proxy / fake / old table。

### 实验设计

先做 smoke：

```text
APX1-APX8 each 64 actions
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledPayload, CertificatePassNoPayload
horizons = 20, 80, 240
```

总 rows：

$$
8 \times 64 \times 8 \times 3 = 12288.
$$

如果 resource 不够，则先做 APX1-APX4 + APX8，每个 64 actions，不允许低于 64，否则不能评估 TopK64。

### 记录指标

```text
primitive_id
action_count
branch_horizon_rows_expected
branch_horizon_rows_actual
branch_completion_rate
horizon_completion_rate
secondary_delta_completion_rate
h20_weak_CP
h20_strong_CP
h20_V_ctrl_mean
h20_V_ctrl_lcb
h80_weak_CP
h80_V_ctrl_lcb
h80_bad_event
h240_weak_CP
h240_V_ctrl_lcb
h240_longrisk
YRobust_precision
YStableHorizon_precision
YStrictAllH_precision
J_robust_mean
J_robust_lcb
bad_event_max
null_rate_max
support_balance_pass
```

### 判断标准

Smoke weak pass：

```text
h20_weak_CP >= 0.50
h20_V_ctrl_lcb > 0
h80_V_ctrl_lcb >= -0.05
h240_longrisk <= 0.15
YStableHorizon_precision >= 0.20
```

Smoke strong pass：

```text
h20_weak_CP >= 0.60
h20_V_ctrl_lcb > 0.10
h80_V_ctrl_lcb > 0
h240_longrisk <= 0.10
YStableHorizon_precision >= 0.30
YStrictAllH_precision >= 0.05
```

如果 APX8 negative control 也过，则 outcome definition 或 branch isolation 有问题，P6 fail。

### 可视化

```text
p6_apx_horizon_value_curves.svg
p6_apx_longrisk_bar.svg
p6_apx_yrobust_ystable_precision.svg
p6_apx_control_branch_beats_matrix.svg
p6_apx_negative_control_comparison.svg
```

---

## P7：source-to-generated / construction damage audit

### 目标

判断 APX 是在创造 value，还是仅仅改写/破坏已有 source。对于 direct APX，source 可以是 AdamW delta / NoOp / nearest AP0 oracle diagnostic。

### 记录指标

```text
primitive_id
reference_source_type
paired_action_count
Damage_h20_mean
Damage_h20_median
Damage_h80_mean
Damage_h240_mean
source_positive_lost_rate
source_negative_fixed_rate
new_positive_created_rate
longrisk_created_rate
payload_distance_to_adamw
payload_distance_to_nearest_ap0
```

### 判断标准

P7 pass：

```text
new_positive_created_rate >= 0.15
longrisk_created_rate <= 0.10
Damage_h20_lcb >= 0 for generated-vs-source
```

### 可视化

```text
p7_damage_waterfall.svg
p7_source_to_generated_sankey.svg
p7_payload_distance_vs_value.svg
```

---

## P8：effect-valid certificate v4

### 目标

在 APX generated actions 上验证 certificate 是否真的预测 effect，而不是描述 construction。

### Certificate candidates

```text
CERT20-LinearizedDescentBound
CERT21-NoHarmAverageBound
CERT22-HorizonGuardBound
CERT23-CurvatureRiskBound
CERT24-AdamWConflictReliefBound
CERT25-CompositeEffectCertificate
CERT26-MinimalMonotoneCertificate
```

### 记录指标

```text
certificate_id
primitive_id
AUC_TA
AUC_TB
AUC_TC
AUC_LongRisk
TopK16_TB_precision
TopK64_TB_precision
TopK64_LongRisk
TopK273_TB_precision
ECE_TB
Brier_TB
monotone_sign_pass
calibration_to_heldout_drift
LDO_drop
LSO_drop
field_count
ablation_drop_per_field
```

### 判断标准

Certificate weak pass：

```text
AUC_TB >= 0.70
TopK64_TB_precision >= 0.20
TopK64_LongRisk <= 0.15
ECE_TB <= 0.10
monotone_sign_pass = 1
```

Certificate strong pass：

```text
AUC_TB >= 0.75
TopK64_TB_precision >= 0.25
TopK64_LongRisk <= 0.10
ECE_TB <= 0.08
LDO_drop <= 0.08
LSO_drop <= 0.10
field_count <= 7
```

### 可视化

```text
p8_certificate_roc_pr.svg
p8_certificate_topk_enrichment.svg
p8_certificate_calibration_curve.svg
p8_certificate_ablation_waterfall.svg
p8_certificate_longrisk_tradeoff.svg
```

---

## P9：minimal certificate controller

### 目标

只在 P6/P8 过线后，构建 minimal monotone controller。不得使用 outcome label at commit time，不得按 dataset 分支。

### Controller form

$$
Accept(a)=1
\iff
CertLCB(V_{20})>\tau_{20}
\land
CertLCB(V_{80})>\tau_{80}
\land
CertUCB(LongRisk_{240})\le\tau_{LR}
\land
CertUCB(BadMax)\le\tau_B
\land
SupportLCB\ge\tau_S
\land
Cost\le C_{max}.
$$

### 记录指标

```text
controller_id
primitive_id
certificate_id
calibration_split
heldout_split
accepted_count_cal
accepted_count_heldout
coverage_cal
coverage_heldout
precision_TB_cal
precision_TB_heldout
bad_event_heldout
longrisk_heldout
null_rate_heldout
V_ctrl_lcb_h20_heldout
V_ctrl_lcb_h80_heldout
support_balance_pass
accepted_family_count
max_family_share
LDO_pass
LSO_pass
thresholds_frozen
uses_dataset_name
uses_future_outcome
```

### 判断标准

Controller pass：

```text
coverage_heldout in [0.03, 0.15]
precision_TB_heldout >= 0.75
bad_event_heldout <= 0.05
longrisk_heldout <= 0.10
null_rate_heldout <= 0.15
V_ctrl_lcb_h20_heldout > 0
V_ctrl_lcb_h80_heldout >= -0.05
support_balance_pass = 1
LDO_pass = 1
LSO_pass = 1
```

### 可视化

```text
p9_controller_frontier.svg
p9_calibration_heldout_drift.svg
p9_ldo_lso_heatmap.svg
p9_accepted_population_composition.svg
```

---

## P10：selected controller online runtime

### 目标

只有 selected controller 存在后才测 official runtime。P8 runtime preflight 不算。

### 记录指标

```text
runtime_candidate_id
controller_id
primitive_id
step_count
active_step_count
accepted_action_count
certificate_compute_time_ms_q90
controller_score_time_ms_q90
payload_lookup_time_ms_q90
payload_apply_time_ms_q90
base_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
memory_ratio
controller_kernel_launches_per_active_step_q90
controller_syncs_per_active_step_q90
zero_candidate_controller_kernel_count
audit_outside_timed_path
```

### 判断标准

Runtime pass：

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
zero_candidate_controller_kernel_count = 0
controller_launches_per_active_step_q90 <= 2
controller_syncs_per_active_step_q90 <= 1
audit_outside_timed_path = 1
```

### 可视化

```text
p10_runtime_waterfall.svg
p10_step_ratio_distribution.svg
p10_payload_apply_cost_by_primitive.svg
p10_memory_timeline.svg
```

---

## P11：system integration gate

### 目标

整合 primitive、certificate、controller、runtime，决定是否能打开 paired replay / short/full。

### 记录指标

```text
primitive_generation_pass
branch_horizon_outcome_pass
certificate_effect_valid_pass
controller_pass
runtime_pass
manual_forward_pass
manual_backward_pass
manual_adamw_update_pass
uses_loss_backward
uses_teacher
uses_loss_modification
uses_dataset_name
uses_validation_or_test
uses_future_outcome
fake_data_used
proxy_row_used
cpu_offload_used
official_eligible
system_legal_controller_pass
```

### 判断标准

System pass：

```text
primitive_generation_pass = 1
certificate_effect_valid_pass = 1
controller_pass = 1
runtime_pass = 1
contract_audit_pass = 1
official_eligible = 1
system_legal_controller_pass = 1
```

### 可视化

```text
p11_system_gate_ladder.svg
p11_contract_audit_matrix.svg
p11_failure_route_tree.svg
```

---

## P12：official paired replay

### 目标

只有 P11 pass 后打开。判断 APX functional update 是否真的胜过 controls。

### Branches

```text
RealFunctional_APX
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledPayload
CertificatePassNoPayload
```

### 记录指标

```text
branch_id
dataset
seed
step
horizon
final_train_loss
val_acc
test_acc
NLL
ECE
CEp99
margin_p10
curvature
local_lipschitz
sample_efficiency_auc
steps_to_target
time_to_target
real_beats_adamwparallel
real_beats_bestlr
real_beats_noop
real_beats_random
shuffle_control_fail
```

### 判断标准

Paired replay pass：

```text
RealFunctional_APX beats AdamWParallel on primary value metric；
RealFunctional_APX beats bestLR；
RealFunctional_APX beats NoOp；
ShuffledPayload fails；
CertificatePassNoPayload fails；
paired confidence interval lower bound > 0；
no catastrophic bad-event increase。
```

### 可视化

```text
p12_paired_replay_delta_distribution.svg
p12_branch_comparison_radar.svg
p12_shuffle_control_failure.svg
p12_value_vs_risk_tradeoff.svg
```

---

## P13：short/full training and MLP comparison boundary

### 目标

只有 P11/P12 pass 后打开 official functional training。Base-Acc Sentinel 继续跑，但不用于 controller。

### 记录指标

```text
dataset
seed
model_id
controller_id
primitive_id
training_budget
train_acc
val_acc
test_acc
NLL
ECE
CEp99
margin_p10
step_time_q90
memory_peak
params
forward_flops
backward_flops
sample_efficiency_auc
time_to_target
steps_to_target
```

### Baselines

```text
MatchedMLP
AdamWStrongLRGridMLP
QuadraticFeatureMLP
LQ-t2-h256 base no functional
LQ-t2-h256 + APX functional
NoOp functional
Shuffled functional
```

### 判断标准

Short-run pass：

```text
APX functional improves LQ base or matched MLP on at least one non-accuracy metric without worsening safety；
step_ratio_q90 <= 1.50；
no dataset-specific tuning。
```

Full-run external-ready candidate：

```text
APX functional beats AdamWStrongLRGridMLP or improves sample efficiency / calibration / robustness with matched cost；
paired replay confirms causality；
LDO/LSO pass；
shuffled controls fail。
```

### 可视化

```text
p13_acc_by_dataset_seed.svg
p13_lq_vs_mlp_gap.svg
p13_sample_efficiency_auc.svg
p13_calibration_ece.svg
p13_runtime_cost_vs_accuracy.svg
```

---

## P14：continual / anti-forgetting boundary

### 目标

验证 Beyond-MLP 不只是单任务 acc，而是可能在 stability / forgetting 上有优势。

### 记录指标

```text
task_order
old_task_acc_before
old_task_acc_after
new_task_acc
forgetting
backward_transfer
forward_transfer
old_task_NLL_delta
old_task_CEp99_delta
old_task_margin_drift
retained_accuracy_auc
```

### 判断标准

Continual weak pass：

```text
forgetting lower than matched MLP by >= 0.02 absolute；
new task acc not worse by > 0.01；
old task CEp99 not worse；
```

Strong pass：

```text
retained accuracy AUC improves；
backward transfer nonnegative；
calibration improves or stays neutral；
```

### 可视化

```text
p14_forgetting_curves.svg
p14_retained_accuracy_auc.svg
p14_task_order_matrix.svg
p14_old_task_margin_drift.svg
```

---

# 8. 并行执行设计

为避免继续“一轮清一个 blocker”，v9.5.1 必须并行执行。

## Batch A：target / label / truth lane

```text
P0 boundary reproduction
P1 multi-objective label audit
old/new target consistency
YRobust/YStable/YStrict density and horizon conflict
```

输出：

```text
p1_target_decision.json
```

## Batch B：legal information lane

```text
P2 raw legal upper-bound probes
P3 mechanism anatomy
feature legality audit
```

输出：

```text
p2_legal_information_route.json
```

## Batch C：primitive implementation lane

```text
P4 APX1-APX8 implementation
P5 deterministic preflight
payload/certificate/action apply closure
```

输出：

```text
p5_apx_preflight_route.json
```

## Batch D：outcome lane

```text
P6 APX smoke outcomes
P7 damage audit
P8 certificate effect validity
```

输出：

```text
p8_certificate_route.json
```

## Batch E：baseline / runtime lane

```text
Base-Acc Sentinel continuation
payload apply microbench
certificate compute microbench
selected runtime preflight template
```

输出：

```text
p10_runtime_preflight_route.json
```

---

# 9. Route decision table

```text
R0-ReproductionFail:
  P0 fail；先修 artifact / runner。

R1-TargetConflictUnresolved:
  P1 发现 YRobust/YStable/YStrict 无 official target；先修 objective definition。

R2-LegalInformationUpperBoundExists:
  P2 raw legal upper-bound pass；进入 feature distillation / minimal certificate。

R3-LegalSelectionDeadConstructiveNeeded:
  P2 fail；existing AP0 selection route 停止。

R4-APXImplementationFail:
  P4/P5 fail；先修 APX payload/certificate/action apply。

R5-APXGeneratedValueFail:
  APX outcomes fail；primitive objective still wrong。

R6-CertificateEffectFail:
  APX generates value but certificate cannot identify；redesign certificate。

R7-ControllerSupportCollapse:
  certificate works locally but heldout/LDO/LSO fails；support/regularization issue。

R8-RuntimeFail:
  controller pass but step_ratio_q90 > 1.50；runtime is primary blocker。

R9-SystemPassPairedReplayOpen:
  controller + runtime pass；open P12.

R10-PairedReplayCausalFail:
  system pass but RealFunctional does not beat controls；functional value target wrong。

R11-ShortFullExternalCandidate:
  P12/P13 pass；prepare external-ready package。
```

---

# 10. Stop conditions

本轮必须避免继续小修小补。以下情况立即停止对应路线：

```text
1. If UB8 raw legal upper-bound AUC_TB < 0.65:
   stop existing AP0 selection route。

2. If SG1-SG4 / CERT14-CERT18 are the best again:
   do not tune them；they are deprecated baselines。

3. If APX negative control passes:
   stop outcome evaluation；branch/control semantics bug。

4. If APX h20 V_ctrl LCB remains < 0 for all primitives:
   do not build controller；primitive objective fail。

5. If APX h20 positive but h240 longrisk > 0.50:
   stop short-horizon-only objective；horizon safety fail。

6. If certificate TopK64 longrisk > 0.25:
   do not open controller。

7. If controller accepted_count_heldout = 0:
   route as support collapse, not success。

8. If selected runtime not measured:
   no system pass。
```

---

# 11. 最终交付物

v9.5.1 必须落盘：

```text
p0_v9500_boundary_reproduction.csv
p1_multi_objective_label_audit_v2.csv
p2_raw_legal_upper_bound_probe_v2.csv
p3_mechanism_anatomy_v2.csv
p4_apx_primitive_family_spec.csv
p5_apx_preflight_ladder.csv
p6_apx_branch_horizon_smoke_outcome.csv
p7_apx_damage_audit.csv
p8_effect_certificate_v4.csv
p9_minimal_certificate_controller.csv
p10_selected_controller_runtime.csv
p11_system_integration_gate.csv
p12_official_paired_replay_boundary.csv
p13_short_full_training_boundary.csv
p14_continual_antiforgetting_boundary.csv
base_acc_sentinel_v9510.csv
no_fake_audit_v9510.csv
contract_audit_v9510.csv
route_decision_v9510.json
failure_taxonomy_v9510.csv
```

必须生成可视化：

```text
p1_target_overlap_venn.svg
p1_horizon_value_curve_by_target.svg
p2_probe_auc_bar.svg
p2_topk_precision_longrisk_tradeoff.svg
p3_mechanism_effect_size_matrix.svg
p4_apx_payload_geometry.svg
p5_hash_match_grid.svg
p6_apx_horizon_value_curves.svg
p6_apx_control_branch_beats_matrix.svg
p7_source_to_generated_sankey.svg
p8_certificate_calibration_curve.svg
p9_controller_frontier.svg
p10_runtime_waterfall.svg
p13_lq_vs_mlp_gap.svg
p14_forgetting_curves.svg
```

---

# 12. 最终判断

v9.5.1 的判断标准很明确：

```text
如果 raw legal tensor upper-bound 仍失败，而 APX constructive primitives 也全部失败：
  当前 LQ-t2-h256 + AP0/APX local update action space 可能不足，需要更深层 primitive family reset，甚至回到 KAN base / update basis 设计。

如果 raw legal upper-bound 成功但 distillation 失败：
  信息存在，但 official cost/minimality 路线未闭合。

如果 APX 能生成 robust actions但 certificate 失败：
  primitive 有价值，但 certificate 不充分。

如果 certificate/controller pass 但 runtime fail：
  科学路线有希望，工程 runtime 是 blocker。

如果 controller/runtime/paired replay pass：
  才能进入真正的 short/full functional training 与 Beyond-MLP evidence。
```

当前最重要的一句话是：

$$
\boxed{\text{不要再从 opaque AP0 action 中猜好 action；要让新 primitive 在生成时就带来可验证的 multi-horizon effect certificate。}}
$$
