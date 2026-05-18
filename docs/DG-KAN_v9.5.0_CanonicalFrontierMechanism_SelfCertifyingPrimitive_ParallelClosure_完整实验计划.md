# DG-KAN v9.5.0 Canonical Frontier Mechanism / Self-Certifying Primitive / Parallel Closure 完整实验计划

> 本计划基于 v9.4.9 `Canonical Legal Observability / Source Generator / Certificate Parallel Closure` 的真实执行结果制定。  
> v9.4.9 的 terminal route 是：
>
> ```text
> route = R1-FrontierExistsButLegalOpaque
> base_candidate = LQ-t2-h256
> success_v9490_strict_purekan_functional = False
> success_v9490_full_functional = False
> success_v9490_external_ready = False
> primary_blocker = canonical_frontier_legal_generator_certificate_failed
> ```
>
> v9.5.0 不继续做 AP0 static feature threshold patch，也不继续堆 AP0l/AP0r/AP0u/AP0v 这类小 generator 变体。  
> 本轮目标是回答一个更根本的问题：
>
> $$
> \text{canonical robust source frontier 已存在，但它到底是可学习、可生成、可证书化，还是 outcome-only accidental island？}
> $$
>
> 如果它可学习，就把 high-capacity legal upper bound 蒸馏成 minimal certificate。  
> 如果它不可学习，但可生成，就实现 self-certifying constructive primitive。  
> 如果既不可学习也不可生成，就停止当前 AP0/AP0x 路线，回到 functional primitive 设计层。

---

# 0. 执行摘要

v9.4.9 的关键事实是：

```text
canonical full control outcome ready = 1
canonical rows = 51768 / 51768
quality audit pass = 1
YRobust original count = 120
YRobust manual recompute match rate = 1.0
best legal feature = NegHardTailFraction
best legal AUC_YRobust = 0.5375105829704886
best legal TopK64 YRobust precision = 0.03125
best legal TopK64 h240 longrisk = 0.546875
existing no-transform sanity pass = 1
existing real transform generator preserve pass = 0
direct generator pass = 0
best direct generator = VG3-TailMarginConservativeSource
best direct h20 weak CP = 0.375
best direct h20 V_ctrl LCB = -0.16784700022835225
best direct h240 longrisk = 0.625
certificate effect-valid pass = 0
best certificate AUC_YRobust = 0.536384252539913
best certificate TopK64 YRobust precision = 0.03125
runtime preflight pass = 1
official runtime pass = 0
system legal controller pass = 0
```

这说明项目已经越过两个旧 blocker：

```text
1. old outcome table 不可信的问题已经通过 canonical table rebuild 解决；
2. no-transform replay semantics 已经闭合，source/clone 等价不再是主 blocker。
```

但 v9.4.9 也把问题推到了更尖锐的位置：

```text
1. canonical AP0 robust frontier 存在；
2. current 41 legal action-effect features 几乎看不见 frontier；
3. real transform generators 会破坏 robust source value；
4. direct generator 没有生成 h20 value-positive / h240 safe source；
5. effect-valid certificate 无法区分 YRobust 与 long-risk；
6. controller、selected runtime、paired replay、short/full training 全部仍未打开。
```

因此 v9.5.0 的核心不是：

```text
继续调 NegHardTailFraction threshold；
继续调 PayloadNorm / StateNLL / HardTailFraction；
继续改 AP0l/AP0r/AP0u/AP0v 的半径；
继续调 CERT13 threshold；
继续把 oracle K16 当 selector；
继续用 Base-Acc Sentinel 证明 functional success。
```

v9.5.0 的核心是：

```text
1. 信息上界：用高容量但合法的 commit-time tensor probe 判断 YRobust 是否原则上可见；
2. 机制发现：从 120 个 YRobust actions 与 matched hard negatives 中找出稳定机制；
3. 生成重构：从 transform-existing-action 改成 objective-solved constructive primitive；
4. 证书重构：certificate 必须预测 effect，不只是 construction metadata；
5. 并行验证：analysis / generator / certificate / runtime / base sentinel 同时跑，避免一轮只清一个 blocker。
```

v9.5.0 的最低有效成功不是 full functional success，而是明确落到以下三种 route 之一：

```text
Route A: Legal upper-bound probe sees YRobust frontier。
  说明 current hand-crafted features 不够，需要蒸馏 minimal certificate。

Route B: Legal upper-bound probe 仍失败，但 constructive generator 产生 new robust actions。
  说明 AP0 frontier outcome-only，但可以通过 self-certifying primitive 生成。

Route C: Legal upper-bound probe 和 constructive generator 都失败。
  说明当前 AP0/AP0x action family 不是正确 functional primitive，需要回到更深层 primitive redesign。
```

---

# 1. 独立判断：v9.4.9 说明了什么

## 1.1 canonical truth base 已经不再是 blocker

v9.4.8 recovery 后，canonical table 已经达到：

```text
canonical_row_count_expected = 51768
canonical_row_count_actual = 51768
quality_audit_pass = 1
old_table_quarantine_enforced = 1
```

v9.4.9 P0 复现了这一点。因此 v9.5.0 不能再把主要路线写成 materializer completion。现在 table 已经能支撑 official diagnostic，只是 controller / generator / certificate 都不过。

这一步很重要，因为 v9.4.7 之前很多结论都被旧 table drift 污染；现在判断必须基于 canonical_v9480/v9490 truth。

## 1.2 YRobust label 本身闭合，但目标定义要被正确理解

v9.4.9 P1 证明：

```text
action_count = 2876
YRobust_original_count = 120
manual recompute match rate = 1.0
mismatch count = 0
YRobust_h20_weak_rate = 1.0
YRobust_h80_weak_rate = 0.175
YRobust_h240_longrisk_rate = 0.0
h80_required_by_original_definition = 0
```

这澄清了一个容易误读的点：YRobust 不是 “h20/h80/h240 三个 horizon 全 weak CP”。它的核心是：

```text
h20 immediate positive；
h20 value positive；
h240 long-risk safe；
不要求 h80 weak。
```

因此 v9.5.0 不应该在没有定义审计的情况下随意换成 “all horizon weak” 目标。但 v9.5.0 必须同时记录多个目标版本，因为如果只优化当前 YRobust，可能会选中 h80 中间不稳定的 action。

本轮将同时记录：

```text
YRobust_v9490: original h20 value positive + h240 long-risk safe。
YStrictAllH: h20/h80/h240 weak CP all positive and no long-risk。
YNoLongRiskOnly: h240 long-risk safe regardless of h20 value。
YImmediateOnly: h20 weak + h20 V_ctrl positive。
YStableHorizon: h20 positive + h80 non-bad + h240 non-long-risk。
```

这些目标不能被 dataset 调参；它们只是 objective diagnostics，用来判断当前 label 是否过窄、过宽或 horizon-inconsistent。

## 1.3 legal feature failure 不是“小阈值问题”

v9.4.9 的 41 个 legal/action-effect features 中，best 是：

```text
NegHardTailFraction
AUC_YRobust = 0.5375105829704886
AUC_LongRisk = 0.41508897166902475
TopK16 YRobust precision = 0.0
TopK64 YRobust precision = 0.03125
TopK64 h240 longrisk = 0.546875
```

YRobust base rate 是：

$$
\frac{120}{2876}=0.0417246175.
$$

TopK64 precision = 0.03125 等于 TopK64 只有：

$$
64 \times 0.03125 = 2
$$

个 YRobust actions。它甚至低于 base rate 的期望：

$$
64 \times 0.0417246 \approx 2.67.
$$

这说明 current best legal feature 没有 enrichment。AUC 约 0.537 也接近随机。更糟糕的是 TopK64 h240 longrisk = 0.546875，说明它不但找不到 robust source，还会选到大量 long-risk source。

所以 v9.5.0 不应继续做：

```text
feature threshold patch；
feature monotone weight patch；
feature count 从 41 加到 80 但仍是同类 marginal features。
```

## 1.4 generator failure 不是 materialization failure，而是 value preservation / construction objective failure

v9.4.9 P4 证明 existing generator chain 可以真实运行：

```text
generated action = 192
branch-horizon rows = 3456
payload/certificate hash missing = 0
action apply L∞ max = 0.0
no-transform sanity pass = 1
```

这排除了 payload/apply/replay 作为主因。真正失败是 transform：

```text
AP0l h20 weak CP = 0.1875, h20 V_ctrl LCB = -1.6241, h240 longrisk = 0.75
AP0r h20 weak CP = 0.1875, h20 V_ctrl LCB = -1.9379, h240 longrisk = 0.75
AP0u h20 weak CP = 0.25,   h20 V_ctrl LCB = -1.5237, h240 longrisk = 0.9375
AP0v h20 weak CP = 0.1875, h20 V_ctrl LCB = -1.4569, h240 longrisk = 0.625
```

如果 no-transform 对 oracle panel 可以 preserve，而 transform 全部破坏，说明 transform 不只是“稍微不够保守”，而是在破坏产生 robust value 的关键几何结构。

Direct generator 也不过：

```text
best = VG3-TailMarginConservativeSource
h20 weak CP = 0.375
h20 V_ctrl LCB = -0.167847
h240 longrisk = 0.625
YRobust precision = 0.125
```

它比 random/old AP0x 有一些方向感，但 h20 V_ctrl LCB 仍为负，h240 longrisk 太高。

因此 v9.5.0 的 generator 不能再只是 “tail margin conservative / residual / projection” 的 heuristic。它必须变成 objective-solved constructive generator：先定义要优化的 value-risk constrained update，再生成 payload。

## 1.5 certificate failure 不是 calibration 小问题

v9.4.9 P6：

```text
best certificate = CERT13-MinimalMonotoneCompositeCertificate
AUC_YRobust = 0.536384252539913
AUC_LongRisk = 0.41916336377998953
TopK16 YRobust precision = 0.0
TopK64 YRobust precision = 0.03125
TopK64 longrisk = 0.59375
P(YRobust | cert pass) = 0.03125
P(LongRisk | cert pass) = 0.59375
ECE_YRobust = 0.6683514548369317
```

这不是 threshold 问题。一个有效证书至少应该满足：

$$
P(YRobust \mid CertPass) \gg P(YRobust),
$$

并且：

$$
P(LongRisk \mid CertPass) \ll P(LongRisk).
$$

但现在 certificate-pass 的 YRobust precision 低于 base rate，longrisk 还很高。继续调 CERT13 的 threshold 没意义。

## 1.6 Base-Acc Sentinel 有训练，但不是 functional success

v9.4.9 的 Base-Acc Sentinel 是复用 v9.4.7/v9.4.8：

```text
sentinel_row_count = 120
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
model_count = 4
mean_test_acc_LQ = 0.6537760416666667
mean_test_acc_MLP = 0.562890625
mean_test_acc_QuadraticFeatureMLP = 0.40234375
mean_test_acc_AdamWStrongLRGridMLP = 0.628515625
LQ_minus_MLP = +0.09088541666666672
LQ_minus_AdamWStrongLRGridMLP = +0.025260416666666674
base_acc_used_for_controller = 0
```

这说明 LQ-t2-h256 base 没有 catastrophic fail，而且在 fixed sentinel 下比 MatchedMLP / StrongLRGridMLP 都高。但它不是 official functional training，因为：

```text
source controller 未选中；
selected runtime 未打开；
paired replay 未打开；
short/full functional validation 未打开；
base acc 没有用于 controller。
```

v9.5.0 继续保留 Base-Acc Sentinel，但它仍是健康监控，不是系统成功指标。

---

# 2. v9.5.0 总体目标

v9.5.0 的总体目标是：

$$
\boxed{
\text{判断 canonical robust frontier 是否能被 legal information 识别、被 constructive primitive 生成、被 effect-valid certificate 证明。}
}
$$

v9.5.0 不以 final acc 或 full-run 作为主目标。本轮在 controller/system 之前，必须先回答三个问题：

```text
Q1: 120 个 canonical YRobust actions 是否在 commit-time legal tensor space 中可分？
Q2: 如果可分，能否蒸馏成 <= 6 个 feature groups 的 minimal certificate/controller？
Q3: 如果不可分，能否构造 new self-certifying source primitive，让 action 生成时自带可验证的 descent / horizon-safety / cost certificate？
```

v9.5.0 的强目标：

```text
legal_upper_bound_probe_pass = 1
或 constructive_generator_pass = 1
并且 effect_valid_certificate_pass = 1
source_controller_pass = 1
selected_runtime_pass = 1
system_legal_controller_pass = 1
```

v9.5.0 的最低有效推进目标：

```text
1. high-capacity legal upper-bound probe 完整跑完；
2. YRobust / YStrict / YStableHorizon 多目标一致性审计完成；
3. robust source mechanism anatomy 完成；
4. 至少 3 条 constructive generator routes 完整 materialize branch-horizon rows；
5. certificate 的 effect-validity 被重新测量；
6. 明确 route 到：FeatureDistillation / ConstructivePrimitive / PrimitiveReset 三者之一。
```

---

# 3. 硬约束

继续遵守：

```text
no teacher
no self-teacher
no distillation loss
no auxiliary task loss
no loss modification
no label smoothing / focal loss / class weight
no dataset-specific selector/controller/generator branch
no validation/test metric at commit time
no outcome-at-commit feature
no old v9.3.5 outcome table official usage
no fake data
no proxy rows
no CPU offload
no PyTorch loss.backward graph in official KAN path
manual forward / backward / AdamW update must remain official
functional update remains update rule, not loss trick
```

Functional update form stays：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{AdamW}
+
\Delta\theta_{functional}.
$$

Task loss remains：

$$
L_{task}=CE(y,p_\theta(x)).
$$

Allowed diagnostics：

```text
high-capacity legal probe trained on calibration outcomes for upper-bound only；
contrastive representation analysis；
oracle source anatomy；
source prototype mining；
canonical outcome table label reanalysis；
stratified / leave-out diagnostics；
parallel Base-Acc Sentinel；
runtime preflight outside official controller。
```

Allowed official components only if they pass legality audit：

```text
commit-time features；
commit-time certificate tensors；
pre-registered generator equations；
calibration-frozen thresholds；
selected runtime measured on online sequential path；
no dataset_name / no validation / no test / no future outcome。
```

---

# 4. 核心假设

## H1：YRobust frontier 在 legal tensor space 中可能可分，但 hand-crafted scalar features 不够

v9.4.9 只证明 41 个 manually designed legal features 不够。它没有证明所有 commit-time legal information 都不可用。

H1 成立标准：

```text
高容量 legal tensor probe 在 cross-fit heldout 上：
AUC_YRobust >= 0.75
AUC_LongRisk >= 0.75
TopK64 YRobust precision >= 0.20
TopK64 longrisk <= 0.20
leave-dataset AUC drop <= 0.08
leave-stratum AUC drop <= 0.10
```

H1 失败标准：

```text
高容量 probe 使用完整 legal tensor bundle 后仍：
AUC_YRobust < 0.65
TopK64 YRobust precision < 0.10
或 LDO/LSO 崩溃。
```

若 H1 成立，v9.5.0 进入 feature distillation / minimal certificate。  
若 H1 失败，v9.5.0 不能继续扩 hand-crafted features，应转向 constructive primitive。

## H2：当前 YRobust objective 需要和 stricter horizon objectives 一起审计

H2 认为当前 YRobust 是 useful but incomplete target。它不要求 h80 weak，因此它可能允许中间 horizon value drop。

H2 成立标准：

```text
YRobust_v9490 与 YStableHorizon / YStrictAllH 的 overlap 和 divergence 被量化；
如果 YRobust 高但 h80 failure 大量存在，则 certificate 需要显式记录 h80 non-bad 或 h80 damage constraint。
```

H2 失败标准：

```text
YRobust_v9490 与 stricter horizon objectives 完全一致或差异很小；
当前 h80 不要求不是问题。
```

## H3：现有 transform generators 破坏 source value，因为它们没有保留 robust source 的局部机制

H3 成立标准：

```text
oracle-seeded no-transform pass；
real transform generator fail；
source-positive lost rate high；
payload geometry drift or action-response drift explains value loss；
mechanism-preserving perturbation has lower damage than AP0l/AP0r/AP0u/AP0v。
```

H3 失败标准：

```text
transform failure 主要来自 bug / replay mismatch / source join mismatch；
修 bug 后 transform preserve pass。
```

## H4：direct generator 当前失败是 objective formulation failure，而不是 materialization failure

H4 成立标准：

```text
payload/certificate/action apply/branch-horizon rows all close；
best generator h20 V_ctrl LCB remains negative；
h240 longrisk remains high；
therefore generator equations do not solve robust source objective。
```

H4 失败标准：

```text
new direct objective-solved generator produces h20 positive and h240 safe actions。
```

## H5：effect-valid certificate 必须 predict outcome effect, not construction metadata

H5 成立标准：

```text
certificate components tied to predicted CE decrease, margin repair, longrisk guard, support, and cost；
CERT score has monotone sign；
AUC_YRobust >= 0.75；
AUC_LongRisk >= 0.75；
TopK64 YRobust precision >= 0.20；
TopK64 longrisk <= 0.20；
ECE_YRobust <= 0.10。
```

H5 失败标准：

```text
certificate remains near random or selects high long-risk rows。
```

## H6：如果 H1-H5 都失败，current AP0/AP0x family is not the right primitive

H6 成立标准：

```text
legal upper-bound probe fail；
constructive generators fail；
certificate fail；
source mechanism not stable across splits；
then route = R-PrimitiveFamilyResetRequired。
```

H6 失败标准：

```text
any of upper-bound/legal-generator/certificate routes produces official source controller candidate。
```

---

# 5. 数据合同

## 5.1 Canonical action table

所有阶段必须以 canonical_v9480/v9490 full table 为 source of truth：

```text
action_id
candidate_id
event_id
source_step
source_dataset
source_seed
source_family
source_bucket
source_horizon
payload_hash
state_before_hash
branch_config_hash
horizon_config_hash
label_config_hash
canonical_outcome_row_id
```

不得读取旧 v9.3.5 outcome table 作为 official truth。旧表只能用于 drift history diagnostic。

## 5.2 Multi-objective labels

每个 action 必须记录：

```text
YRobust_v9490
YImmediateOnly
YNoLongRiskOnly
YStableHorizon
YStrictAllH
WeakCP_h20
WeakCP_h80
WeakCP_h240
StrongCP_h20
StrongCP_h80
StrongCP_h240
V_ctrl_h20
V_ctrl_h80
V_ctrl_h240
LongRisk_h20
LongRisk_h80
LongRisk_h240
BadEvent_h20
BadEvent_h80
BadEvent_h240
NullEvent_h20
NullEvent_h80
NullEvent_h240
```

必须记录 action-level aggregation：

```text
YRobust_action
YStrict_action
YStable_action
longrisk_any_action
h20_positive_action
h80_nonbad_action
h240_safe_action
```

## 5.3 Legal tensor bundle

为了判断 information upper bound，每个 action 的 commit-time legal tensor bundle 包括：

```text
batch logits summary tensor
hard-tail logits tensor sketch
CE / NLL / entropy / margin vector sketch
manual gradient sketch
gradient norm by role
AdamW update sketch
functional payload tensor sketch
payload rolewise norm
payload sign / sparsity / low-rank sketch
cosine(payload, AdamW)
cosine(payload, -grad)
state tail statistics
basis usage statistics
family / bucket / horizon / support features
runtime cost features
```

所有 tensor sketch 必须满足：

```text
uses_dataset_name = 0
uses_validation_or_test = 0
uses_future_outcome = 0
uses_outcome_at_commit = 0
commit_time_available = 1
hash_logged = 1
cost_logged = 1
```

## 5.4 Generator payload contract

每个 generated action 必须记录：

```text
generator_id
generator_family
source_action_id if seeded
payload_hash
certificate_hash
action_apply_error_linf
action_apply_error_relative
action_apply_cosine
payload_norm
payload_linf
payload_role_entropy
payload_tail_selectivity
payload_adamw_cosine
payload_negative_grad_cosine
construction_objective_value
construction_constraint_violation_count
```

Pass：

```text
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max <= 1e-7
action_apply_cosine_min >= 0.999999
branch_horizon_completion_rate = 1.0
```

## 5.5 Certificate contract

Certificate must contain effect predictions：

```text
cert_value_lcb_h20
cert_value_lcb_h80
cert_value_lcb_h240
cert_longrisk_ucb_h240
cert_bad_ucb_h20
cert_bad_ucb_h80
cert_bad_ucb_h240
cert_null_ucb
cert_support_lcb
cert_cost_q90_est
cert_descent_alignment
cert_tail_margin_repair
cert_control_advantage_proxy
cert_monotone_score
cert_pass
```

Official certificate legality：

```text
certificate_hash_bound_to_payload_hash = 1
certificate_commit_time_available = 1
certificate_uses_outcome_at_commit = 0
certificate_uses_dataset_name = 0
certificate_uses_validation_or_test = 0
certificate_fields_complete = 1
```

Effect-valid pass：

```text
AUC_YRobust >= 0.75
AUC_LongRisk >= 0.75
TopK64_YRobust_precision >= 0.20
TopK64_h240_longrisk <= 0.20
ECE_YRobust <= 0.10
monotone_sign_pass = 1
```

---

# 6. 实验阶段

---

## P0：v9.4.9 boundary reproduction and route lock

### 目标

确认 v9.4.9 boundary 稳定，防止在错误 artifact 或旧 truth table 上继续。

### 执行

读取：

```text
v9490 route_decision_v9490.json
p0_v9480_recovery_boundary_reproduction.csv
p1_yrobust_definition_consistency_audit.csv
p3_legal_action_effect_feature_factory_v2.csv
p4_canonical_existing_generator_preservation.csv
p5_direct_robust_source_generator_v2.csv
p6_effect_valid_certificate_v2.csv
contract_audit.csv
failure_table.csv
```

### 必须记录

```text
route_v9490
canonical_full_control_outcome_ready
canonical_row_count_expected
canonical_row_count_actual
quality_audit_pass
YRobust_definition_consistency_pass
YRobust_action_count
best_legal_feature_id
best_legal_AUC_YRobust
best_legal_TopK64_YRobust_precision
existing_no_transform_sanity_pass
existing_generator_preserve_pass
direct_generator_pass
certificate_effect_valid_pass
runtime_preflight_pass
system_legal_controller_pass
fake_data_used
proxy_row_used
cpu_offload_used
```

### Pass 标准

```text
route_v9490 = R1-FrontierExistsButLegalOpaque
canonical_full_control_outcome_ready = 1
quality_audit_pass = 1
YRobust_definition_consistency_pass = 1
legal_feature_pass = 0
existing_generator_preserve_pass = 0
direct_generator_pass = 0
certificate_effect_valid_pass = 0
fake/proxy/cpu_offload = 0
```

### 可视化

```text
p0_v9490_boundary_ladder.svg
p0_frontier_vs_legal_gap_summary.svg
p0_generator_certificate_failure_grid.svg
```

---

## P1：multi-objective robust label audit

### 目标

确认当前 YRobust 目标是否足够，避免 certificate / generator 学错目标。P1 不修改 official target，只并行记录多个 label variants。

### 假设

H2：YRobust 不要求 h80 weak，这可能是合理的，也可能导致中间 horizon blind spot。必须量化。

### 执行

从 canonical full table 重算每个 action 的：

```text
YRobust_v9490
YImmediateOnly
YNoLongRiskOnly
YStableHorizon
YStrictAllH
```

定义：

$$
YImmediateOnly(a)=1
\iff
WeakCP_{h20}(a)=1
\land
V_{ctrl,h20}(a)>0.
$$

$$
YNoLongRiskOnly(a)=1
\iff
LongRisk_{h240}(a)=0.
$$

$$
YStableHorizon(a)=1
\iff
WeakCP_{h20}(a)=1
\land
BadEvent_{h80}(a)=0
\land
LongRisk_{h240}(a)=0.
$$

$$
YStrictAllH(a)=1
\iff
WeakCP_{h20}(a)=1
\land
WeakCP_{h80}(a)=1
\land
WeakCP_{h240}(a)=1
\land
LongRisk_{h240}(a)=0.
$$

### 必须记录

```text
action_count
YRobust_count
YImmediateOnly_count
YNoLongRiskOnly_count
YStableHorizon_count
YStrictAllH_count
Jaccard_YRobust_YStableHorizon
Jaccard_YRobust_YStrictAllH
P_YStrict_given_YRobust
P_YStable_given_YRobust
P_h80_weak_given_YRobust
P_h80_bad_given_YRobust
P_h240_longrisk_given_h20weak
V_ctrl_h20_lcb_by_label
V_ctrl_h80_lcb_by_label
V_ctrl_h240_lcb_by_label
```

### 判断标准

P1 pass：

```text
all labels recomputed with no mismatch；
label variants and overlap reported；
objective_conflict_class assigned。
```

Conflict classes：

```text
OC0: YRobust aligns with strict/stable labels。
OC1: YRobust has h80 blind spot。
OC2: h20 immediate signal conflicts with h240 safety。
OC3: robust positives are too sparse under strict target。
```

### 可视化

```text
p1_label_variant_overlap_upset.svg
p1_horizon_value_curves_by_label.svg
p1_yrobust_vs_strict_confusion.svg
p1_longrisk_given_immediate_bar.svg
```

---

## P2：high-capacity legal upper-bound probe

### 目标

判断 “YRobust 看不见” 是 hand-crafted feature 不够，还是 commit-time legal information 本身不够。

### 假设

H1：完整 legal tensor bundle 可能包含可分信息，只是 v9.4.9 的 41 个 scalar features 没抓到。

### 执行

构造三个 probe 层级，全部只用 commit-time legal inputs：

```text
UB0-ScalarFeatureProbe:
  使用 v9.4.9 41 scalar features，作为 baseline。

UB1-TensorSketchProbe:
  使用 logits/gradient/payload/state sketch，轻量 MLP / logistic model。

UB2-HighCapacityLegalProbe:
  使用 tensor sketches + rolewise payload + state-tail representations。
  只作为 upper-bound diagnostic，不可直接 official。
```

训练方式：

```text
5-fold seed split；
leave-dataset-out；
leave-family-out；
leave-step-bucket-out；
threshold frozen on calibration fold；
no dataset-specific branch；
class imbalance handled only by reporting PR/lift，不改采样和 loss for official。
```

### 必须记录

```text
probe_id
input_group
feature_count_or_tensor_dim
commit_time_available
uses_dataset_name
uses_outcome_at_commit
train_split
calibration_split
heldout_split
AUC_YRobust
AUC_LongRisk
PR_AUC_YRobust
PR_lift_YRobust
TopK16_YRobust_precision
TopK32_YRobust_precision
TopK64_YRobust_precision
TopK64_LongRisk_rate
TopK128_YRobust_precision
TopK128_LongRisk_rate
ECE_YRobust
Brier_YRobust
LDO_AUC_mean
LDO_AUC_drop_max
LSO_AUC_mean
LSO_AUC_drop_max
cost_q90_ms
memory_ratio
```

### 判断标准

Upper-bound pass：

```text
AUC_YRobust >= 0.75
AUC_LongRisk >= 0.75
TopK64_YRobust_precision >= 0.20
TopK64_LongRisk_rate <= 0.20
ECE_YRobust <= 0.10
LDO_AUC_drop_max <= 0.08
LSO_AUC_drop_max <= 0.10
```

Weak upper-bound pass：

```text
AUC_YRobust >= 0.68
TopK64_YRobust_precision >= 0.12
TopK64_LongRisk_rate <= 0.35
```

Fail：

```text
UB2 AUC_YRobust < 0.65
或 TopK64_YRobust_precision < 0.10
或 LDO/LSO collapse。
```

### Route consequence

```text
if UB2 pass:
  route_next = R2-LegalInformationExistsDistillCertificate

elif UB2 weak pass:
  route_next = R2w-LegalInformationWeakMechanismMiningRequired

else:
  route_next = R3-LegalInformationInsufficientConstructivePrimitiveRequired
```

### 可视化

```text
p2_probe_roc_pr_curves.svg
p2_topk_precision_longrisk_frontier.svg
p2_leave_dataset_auc_heatmap.svg
p2_feature_group_ablation.svg
p2_probe_calibration_curve.svg
```

---

## P3：robust source mechanism anatomy

### 目标

找到 120 个 YRobust actions 的机制结构：它们到底靠什么成为 robust？是 payload geometry、state-tail、gradient alignment、family context、horizon profile，还是某种 interaction。

### 执行

构造 matched negative sets：

```text
N1-hard-tail-near-miss:
  h20 weak but h240 longrisk。

N2-immediate-fail:
  h20 not weak but h240 safe。

N3-near-score-negative:
  legal feature score close to YRobust but label negative。

N4-family-matched-negative:
  same family/bucket/horizon but not YRobust。

N5-random-negative:
  stratified random non-YRobust。
```

对每组做 contrastive anatomy。

### 必须记录

```text
positive_count
negative_count_by_type
matched_family_balance
matched_step_bucket_balance
matched_payload_norm_balance
feature_effect_size_by_group
mutual_information_by_group
Cohen_d_by_group
AUC_by_group
interaction_AUC_pairwise
interaction_AUC_triple
cluster_count
cluster_purity_YRobust
cluster_longrisk_rate
cluster_LDO_stability
cluster_LSO_stability
prototype_count
prototype_reconstruction_error
prototype_payload_geometry_summary
prototype_state_tail_summary
```

### 机制候选

```text
M1-tail-margin-local-repair:
  robust actions concentrate on hard-tail margin repair without increasing global norm。

M2-adamw-compatible-low-conflict:
  robust actions align with AdamW on common roles but add orthogonal tail correction。

M3-low-rank-edge-localization:
  robust actions are low-rank and localized to a few edge/basis roles。

M4-support-memory-pattern:
  robust actions correspond to high-support repeated family pattern。

M5-state-tail-only-outcome-island:
  robust actions only separable by outcome, not commit-time mechanism。
```

### 判断标准

P3 mechanism pass：

```text
at least one mechanism cluster has:
YRobust precision >= 0.20
LongRisk rate <= 0.20
cluster action count >= 32
LDO stability pass
LSO stability pass
```

P3 fail：

```text
no stable cluster or interaction separates robust from hard negatives；
dominant miss remains M8-outcome-only-pattern。
```

### 可视化

```text
p3_robust_vs_nearmiss_umap.svg
p3_mechanism_cluster_heatmap.svg
p3_payload_geometry_by_label.svg
p3_state_tail_by_label.svg
p3_interaction_auc_matrix.svg
p3_prototype_gallery.svg
```

---

## P4：feature distillation if legal upper-bound exists

### 目标

如果 P2 证明 legal information exists，不能直接用 opaque high-capacity probe official。必须蒸馏成 minimal, monotone, cheap certificate/controller。

### 执行条件

```text
run P4 only if P2 upper-bound pass or weak pass。
```

### 设计

从 UB2 probe 中 distill：

```text
D0-single-group baselines；
D1-two-interaction certificate；
D2-monotone 5-feature certificate；
D3-rolewise linear certificate；
D4-low-rank prototype distance certificate；
D5-cost-aware minimal certificate。
```

Certificate score form：

$$
S_{cert}(a)
=
\alpha_1 ValueLCB_{h20}(a)
-
\alpha_2 LongRiskUCB_{h240}(a)
-
\alpha_3 BadUCB(a)
+
\alpha_4 SupportLCB(a)
-
\alpha_5 Cost(a)
+
\alpha_6 PrototypeMatch(a).
$$

Constraints：

```text
alpha_i >= 0 for monotone terms；
feature groups <= 6；
no dataset branch；
threshold frozen on calibration fold；
latency q90 recorded；
component ablation required。
```

### 必须记录

```text
certificate_id
source_probe_id
feature_group_count
monotone_constraint_pass
AUC_YRobust
AUC_LongRisk
TopK64_YRobust_precision
TopK64_LongRisk_rate
ECE_YRobust
cost_q90_ms
memory_ratio
ablation_drop_each_group
LDO_pass
LSO_pass
```

### 判断标准

P4 pass：

```text
AUC_YRobust >= 0.75
AUC_LongRisk >= 0.75
TopK64_YRobust_precision >= 0.20
TopK64_LongRisk_rate <= 0.20
ECE_YRobust <= 0.10
feature_group_count <= 6
cost_q90_ms <= 0.20
LDO/LSO pass
```

P4 fail：

```text
high-capacity signal cannot distill to cheap legal certificate。
```

### 可视化

```text
p4_distillation_performance_vs_complexity.svg
p4_certificate_ablation_waterfall.svg
p4_topk_certificate_frontier.svg
p4_certificate_calibration.svg
```

---

## P5：constructive source generator v3

### 目标

如果 legal observability is weak, construct new action rather than finding AP0 action. Generator must solve a constrained local objective and emit its own certificate.

### 设计原则

Current transform generators fail because they mutate existing source payloads. v9.5.0 generators should construct payloads from state/gradient/tail objective:

$$
\Delta^*
=
\arg\min_{\Delta \in \mathcal{A}}
\widehat{CE}_{tail}(\theta+\Delta)
+
\lambda_{norm}\|\Delta\|^2
+
\lambda_{conflict}\max(0,-\cos(\Delta,\Delta_{AdamW}))
+
\lambda_{risk}\widehat{Risk}_{h240}(\Delta).
$$

Because official path must stay graph-free/manual, implementation should use manual linearized approximations, not PyTorch loss.backward.

### Generator candidates

#### SG1-LinearizedTailDescentConstrained

Constructs last-edge / low-rank payload by solving linearized hard-tail CE descent under norm and AdamW compatibility constraints.

Records：

```text
linearized_tail_descent
norm_constraint_violation
adamw_conflict_constraint
predicted_h20_value_lcb
predicted_h240_longrisk_ucb
```

#### SG2-HorizonGuardedTrustRegion

Uses small trust-region update with explicit horizon-risk penalty estimated from canonical support bins.

Records：

```text
trust_region_radius
risk_bin_id
risk_ucb
support_lcb
predicted_margin_repair
```

#### SG3-PrototypeProjectedRobustSource

Uses P3 robust prototypes. Projects current legal state onto nearest robust prototype direction, then scales to trust-region budget.

Records：

```text
prototype_id
prototype_distance
prototype_support
projection_residual
prototype_match_score
```

#### SG4-AdamWOrthogonalTailRepair

Keeps AdamW-compatible common update and adds orthogonal tail correction only if conflict is low.

Records：

```text
cos_tail_adamw
orthogonal_component_norm
tail_repair_ratio
common_update_preservation
```

#### SG5-NoTransformSelectorUpperBoundDiagnostic

No-transform on oracle source remains diagnostic only, used to confirm materializer/replay.

### Must not do

```text
No oracle source as official input；
No outcome label at commit；
No dataset-specific generator route；
No AP0 transform considered pass unless effect preservation passes；
No generator pass without branch-horizon outcomes。
```

### Must record

```text
generator_id
generated_action_count
payload_hash_missing_count
certificate_hash_missing_count
action_apply_error_linf_max
branch_horizon_rows_expected
branch_horizon_rows_actual
h20_weak_CP
h20_strong_CP
h20_V_ctrl_lcb
h80_weak_CP
h80_V_ctrl_lcb
h240_weak_CP
h240_V_ctrl_lcb
h240_longrisk
YRobust_precision
YStableHorizon_precision
YStrictAllH_precision
bad_event_rate
null_rate
source_positive_lost_rate if seeded
Damage_mean
Damage_median
support_balance_pass
```

### Pass standards

Weak generator pass：

```text
h20_weak_CP >= 0.60
h20_V_ctrl_lcb > 0
h240_longrisk <= 0.20
YRobust_precision >= 0.20
action_count >= 64
support_balance_pass = 1
```

Strong generator pass：

```text
h20_weak_CP >= 0.75
h20_V_ctrl_lcb > 0.10
h240_longrisk <= 0.10
YRobust_precision >= 0.30
YStableHorizon_precision >= 0.20
LDO/LSO diagnostic pass
```

Fail：

```text
best generator h20 V_ctrl_lcb <= 0
or h240_longrisk > 0.50
or YRobust_precision <= base rate + 0.05。
```

### 可视化

```text
p5_generator_value_risk_frontier.svg
p5_generator_horizon_curve.svg
p5_generator_payload_geometry.svg
p5_generator_certificate_scatter.svg
p5_generator_support_balance.svg
```

---

## P6：effect-valid certificate v3

### 目标

对 P4/P5 的 candidate certificate 做 effect-validity audit。Certificate 必须预测实际 canonical outcomes，而不是描述 construction metadata。

### Inputs

```text
canonical AP0 actions；
SG1-SG4 generated actions；
matched hard negatives；
no-transform sanity positives；
stratified random negatives。
```

### Certificate candidates

```text
CERT14-ValueRiskSupportCostCertificate
CERT15-PrototypeEffectCertificate
CERT16-HorizonGuardCertificate
CERT17-GeneratorNativeCertificate
CERT18-MinimalDistilledLegalCertificate
```

### Must record

```text
certificate_id
input_action_family
certificate_feature_count
monotone_sign_pass
AUC_YRobust
AUC_YStableHorizon
AUC_YStrictAllH
AUC_LongRisk
TopK16_YRobust_precision
TopK64_YRobust_precision
TopK64_LongRisk_rate
P_YRobust_given_cert_pass
P_LongRisk_given_cert_pass
ECE_YRobust
Brier_YRobust
cost_q90_ms
memory_ratio
LDO_pass
LSO_pass
```

### Pass

```text
AUC_YRobust >= 0.75
AUC_LongRisk >= 0.75
TopK64_YRobust_precision >= 0.20
TopK64_LongRisk_rate <= 0.20
P_YRobust_given_cert_pass >= 4 * base_rate_YRobust
P_LongRisk_given_cert_pass <= 0.20
ECE_YRobust <= 0.10
monotone_sign_pass = 1
cost_q90_ms <= 0.20
```

### 可视化

```text
p6_certificate_roc_pr.svg
p6_certificate_topk_frontier.svg
p6_certificate_calibration.svg
p6_certificate_longrisk_lift.svg
p6_certificate_component_ablation.svg
```

---

## P7：minimal source controller candidate

### 目标

Only after P4 or P6 pass, build a minimal controller. This is not a broad threshold search.

### Controller form

$$
Accept(a)=1
\iff
CertValueLCB(a)>0
\land
CertLongRiskUCB(a)\le\tau_r
\land
CertBadUCB(a)\le\tau_b
\land
CertSupportLCB(a)\ge\tau_s
\land
CertCost(a)\le C_{max}.
$$

### Must record

```text
controller_id
certificate_id
generator_id if generated
calibration_split
heldout_split
thresholds_frozen
accepted_count_cal
accepted_count_heldout
coverage_heldout
precision_YRobust_heldout
bad_event_heldout
longrisk_heldout
null_rate_heldout
YStableHorizon_precision
YStrictAllH_precision
precision_lcb
longrisk_ucb
support_balance_pass
accepted_family_count
max_family_share
max_stratum_share
uses_dataset_name
uses_validation_or_test
uses_outcome_at_commit
```

### Pass

```text
coverage_heldout in [0.03, 0.15]
YRobust_precision_heldout >= 0.75
longrisk_heldout <= 0.05
bad_event_heldout <= 0.05
null_rate_heldout <= 0.15
precision_lcb >= 0.75
longrisk_ucb <= 0.05
support_balance_pass = 1
uses_dataset_name = 0
uses_validation_or_test = 0
uses_outcome_at_commit = 0
```

Note：如果 YRobust target base is action-level rather than row-level, controller metrics must report both action-level and row-level. Do not mix denominators.

### 可视化

```text
p7_controller_cal_heldout_frontier.svg
p7_controller_support_balance.svg
p7_controller_failure_taxonomy.svg
p7_controller_threshold_stability.svg
```

---

## P8：selected runtime preflight and official runtime

### 目标

如果 controller selected，测 selected payload apply + certificate compute + score + accept + base step 的 online sequential runtime。

### Runtime modes

```text
RT0-no-controller-base-reference
RT1-certificate-only-preflight
RT2-selected-controller-no-payload-apply
RT3-selected-controller-with-payload-apply
RT4-selected-controller-with-audit-outside-timed-path
```

### Must record

```text
runtime_candidate_id
controller_id
certificate_id
generator_id
runtime_mode
step_count
active_step_count
accepted_action_count
zero_candidate_step_count
controller_kernel_launch_count
controller_launches_per_active_step_q90
controller_sync_count
certificate_compute_time_ms_q90
score_accept_time_ms_q90
payload_lookup_time_ms_q90
payload_apply_time_ms_q90
base_train_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
peak_memory_mb
memory_ratio
audit_outside_timed_path
no_event_preservation_pass
base_adamw_equivalence_on_zero_event_steps
```

### Pass

```text
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
controller_launches_per_active_step_q90 <= 2
payload_apply_time_ms_q90 <= 0.20 or justified by total step ratio
no_event_preservation_pass = 1
base_adamw_equivalence_on_zero_event_steps = 1
audit_outside_timed_path = 1
```

### 可视化

```text
p8_runtime_waterfall.svg
p8_step_ratio_distribution.svg
p8_payload_apply_vs_base_step.svg
p8_active_step_launch_histogram.svg
```

---

## P9：system integration gate

### 目标

只在 P7 decision + P8 runtime 同时过线时，打开 system legal controller。

### Must record

```text
system_candidate_id
base_candidate
controller_id
certificate_id
generator_id
canonical_truth_version
candidate_count
action_count
accepted_count
decision_gate_pass
runtime_gate_pass
certificate_gate_pass
generator_gate_pass
manual_forward_pass
manual_backward_pass
manual_adamw_update_pass
no_teacher
no_loss_modification
no_dataset_specific_controller
no_validation_test_leakage
no_old_table_official
system_legal_controller_pass
official_eligible
```

### Pass

```text
decision_gate_pass = 1
runtime_gate_pass = 1
certificate_gate_pass = 1
manual contract pass = 1
no-fake/no-proxy/no-leakage pass = 1
```

### 可视化

```text
p9_system_gate_matrix.svg
p9_contract_audit_matrix.svg
p9_route_ladder.svg
```

---

## P10：leave-dataset-out / leave-stratum-out boundary

### 目标

防止 controller/generator/certificate 只在 MNIST/Fashion/KMNIST 的 combined calibration 上过线。

### Must record

```text
leaveout_type
left_out_dataset
left_out_family
left_out_step_bucket
left_out_horizon_bucket
controller_id
accepted_count
coverage
precision_YRobust
longrisk
bad_event
null_rate
precision_lcb
longrisk_ucb
support_balance_pass
AUC_YRobust
AUC_LongRisk
thresholds_frozen_from_training_folds
```

### Pass

```text
all leave-dataset-out folds pass weak decision gate；
max AUC drop <= 0.08；
coverage never collapses to 0；
longrisk <= 0.08 in every leaveout fold；
no fold uses dataset-specific branch。
```

### 可视化

```text
p10_ldo_lso_heatmap.svg
p10_leaveout_coverage_precision.svg
p10_leaveout_failure_modes.svg
```

---

## P11：diagnostic paired replay scout

### 目标

Only after P9 pass, run diagnostic paired replay scout. This is not official full-run yet; it checks whether selected functional updates beat controls locally.

### Branches

```text
RealFunctionalSelected
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledPayload
ShuffledCertificate
```

### Horizons

```text
20
80
240
```

### Must record

```text
replay_action_count
branch_count
horizon_count
completion_rate
V_ctrl_mean
V_ctrl_lcb
beats_adamwparallel_rate
beats_bestlr_rate
beats_noop_rate
beats_random_rate
shuffle_control_fail_rate
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
calibration_delta
curvature_delta
local_lipschitz_delta
```

### Pass

```text
V_ctrl_lcb > 0
beats_adamwparallel_rate >= 0.60
beats_bestlr_rate >= 0.60
shuffle_control_fail_rate >= 0.80
CEp99_delta non-worse
ECE/NLL non-worse
```

### 可视化

```text
p11_paired_replay_branch_bars.svg
p11_horizon_value_curves.svg
p11_shuffle_control_comparison.svg
```

---

## P12：official paired replay

### 目标

If P11 scout passes, run official paired replay with frozen controller/generator/certificate.

### Pass

```text
completion_rate = 1.0
V_ctrl_lcb > 0
beats AdamWParallel / bestLR / NoOp / Random with pre-registered thresholds
shuffle controls fail
no dataset-specific tuning
```

### 可视化

```text
p12_official_paired_replay_summary.svg
p12_control_advantage_ci.svg
p12_branch_horizon_grid.svg
```

---

## P13：short/full functional training boundary

### 目标

Only after P12 pass, open short/full training. This answers whether local selected functional updates improve task trajectories, not just local replay labels.

### Must record

```text
dataset
seed
model
controller_id
generator_id
certificate_id
train_steps
train_loss_curve
val_loss_curve
test_acc
val_acc
ECE
NLL
CEp99
margin_p10
time_to_target
steps_to_target
ValLossAUC_step
ValLossAUC_time
step_ratio_q90
memory_ratio
functional_event_count
accepted_event_count
bad_event_rate_online
```

### Baselines

```text
LQ-t2-h256 AdamW
MatchedMLP
AdamWStrongLRGridMLP
QuadraticFeatureMLP
NoFunctionalControl
ShuffledFunctionalPayload
```

### Pass

```text
Functional LQ beats LQ AdamW on at least one causal metric；
Functional LQ does not regress test acc / ECE / NLL；
Functional LQ beats matched controls in paired replay；
No dataset-specific controller tuning；
StepRatio q90 <= 1.50。
```

### 可视化

```text
p13_train_val_curves.svg
p13_acc_vs_time.svg
p13_val_loss_auc.svg
p13_calibration_curves.svg
p13_event_accept_timeline.svg
```

---

## P14：Base-Acc Sentinel continuation

### 目标

继续健康监控，不让 base 崩坏被误认为 functional controller 失败。Sentinel 不用于 selector/generator/certificate/controller。

### Must record

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
models = LQ-t2-h256, MatchedMLP, AdamWStrongLRGridMLP, QuadraticFeatureMLP
mean_test_acc
std_test_acc
mean_val_acc
ECE
NLL
training_time
LQ_minus_MLP
LQ_minus_AdamWStrongLRGridMLP
LQ_catastrophic_fail
base_acc_used_for_controller = 0
```

### Pass

```text
LQ_catastrophic_fail = 0
base_acc_used_for_controller = 0
sentinel_complete = 1
```

### 可视化

```text
p14_base_acc_by_dataset.svg
p14_lq_vs_mlp_gap_by_seed.svg
p14_base_calibration.svg
```

---

# 7. 并行执行计划

v9.5.0 必须避免“一轮只清一个 blocker”。建议并行分四条线。

## Batch A：analysis-only，无需新 materializer

```text
P0 boundary reproduction
P1 multi-objective label audit
P2 legal upper-bound probe
P3 robust source anatomy
P14 Base-Acc Sentinel continuation
```

预计产出：

```text
information upper-bound decision；
objective conflict class；
mechanism clusters；
base health report。
```

## Batch B：generator implementation and smoke

```text
P5 SG1-SG4 generator preflight；
16-action preflight；
64-action smoke；
branch-horizon materialization；
action apply replay；
certificate tensor write。
```

Stop rule：

```text
If 16-action preflight has action apply error or branch-horizon incomplete, do not run 64-action smoke。
If 64-action h20 V_ctrl LCB <= -0.5 and longrisk > 0.5 for all generators, do not run full scale。
```

## Batch C：certificate and controller conditional

```text
P4 distillation if P2 pass；
P6 certificate v3 if P4/P5 candidate exists；
P7 minimal source controller only if certificate effect-valid pass。
```

Stop rule：

```text
No certificate TopK64 precision lift -> no controller。
No controller decision pass -> no selected runtime official。
```

## Batch D：runtime preflight

```text
P8 runtime preflight can run with dummy selected controller path；
Official runtime only if P7 selects controller。
```

Stop rule：

```text
runtime_preflight_pass can be recorded diagnostic；
official_runtime_pass requires selected controller。
```

---

# 8. Route decision table

At end of v9.5.0, choose exactly one route:

```text
R0-BoundaryRegression
  canonical truth / YRobust definition / no-fake audit regressed。

R1-LegalUpperBoundFail
  high-capacity legal probe cannot identify YRobust。

R2-LegalUpperBoundPassDistillationFail
  high-capacity legal probe works, but no cheap minimal certificate can be distilled。

R3-MechanismClusterFoundGeneratorFail
  robust mechanism found, but generator cannot produce value-positive actions。

R4-ConstructiveGeneratorPassCertificateFail
  generator produces useful actions, but certificate cannot identify them.

R5-CertificateControllerPassRuntimeFail
  decision pass, runtime fail.

R6-SystemLegalControllerPassPairedReplayPending
  decision + runtime pass, downstream not yet open.

R7-PairedReplayFail
  system pass but local causal advantage fails.

R8-StrictPureKANFunctionalLocalSuccess
  system pass + paired replay pass + leaveout pass.

R9-PrimitiveFamilyResetRequired
  legal upper-bound, mechanism, generator, certificate all fail.
```

---

# 9. 最终成功标准

v9.5.0 strict local success requires：

```text
canonical truth pass = 1
YRobust definition pass = 1
legal or constructive route pass = 1
certificate effect-valid pass = 1
source controller pass = 1
selected runtime pass = 1
system legal controller pass = 1
no-fake/no-proxy/no-offload = 1
no dataset-specific controller = 1
```

Functional local success additionally requires：

```text
LDO/LSO pass = 1
paired replay pass = 1
RealFunctional beats AdamWParallel / bestLR / NoOp / Random
shuffle controls fail
```

External-ready success remains deferred until：

```text
short/full training pass；
StrongLRGridMLP and QuadraticFeatureMLP baselines included；
sample efficiency or calibration or robustness advantage shown；
continual / anti-forgetting measured；
step_ratio_q90 <= 1.50；
memory_ratio <= 1.05。
```

---

# 10. v9.5.0 预期解读

## 如果 P2 high-capacity legal probe 过

说明 v9.4.9 的 failure 不是 “legal information absent”，而是 hand-crafted features 不够。下一步是 distillation and minimal certificate。

## 如果 P2 fail 但 P5 constructive generator 过

说明 AP0 robust frontier 是 outcome-only for selection，但可以通过 constructive primitive 生成新 robust actions。下一步是 generator-native certificate and controller。

## 如果 P2/P5/P6 全 fail

说明当前 AP0/AP0x family 很可能不是正确 primitive。此时继续调 feature、certificate、AP0x 半径都属于小修小补，应切换到 deeper functional primitive family reset。

---

# 11. 本轮最重要的原则

v9.5.0 不再问：

```text
能不能再找到一个更好的 scalar feature？
能不能再调一个 generator threshold？
能不能让 CERT13 稍微准一点？
```

v9.5.0 只问：

$$
\boxed{
\text{canonical robust action 的机制是否在 commit time 可见？如果不可见，我们能否生成一个自带 effect certificate 的新 action？}
}
$$

如果这个问题回答不了，项目应该及时承认当前 AP0/AP0x action family 到了路线边界，而不是继续在 MNIST/Fashion/KMNIST 上刷分或在 41 个 feature 上微调。
