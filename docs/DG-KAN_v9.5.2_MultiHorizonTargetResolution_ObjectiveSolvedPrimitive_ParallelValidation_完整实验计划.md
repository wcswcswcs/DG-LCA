# DG-KAN v9.5.2 Multi-Horizon Target Resolution / Objective-Solved Primitive / Parallel Validation 完整实验计划

> 本计划基于 v9.5.1 `Primitive Family Reset / Multi-Horizon Objective / Constructive Certificate` 的真实执行结果制定。  
> v9.5.1 的 terminal route 是：
>
> ```text
> route = R1-TargetConflictUnresolved
> base_candidate = LQ-t2-h256
> success_v9510_strict_purekan_functional = False
> success_v9510_full_functional = False
> success_v9510_external_ready = False
> ```
>
> v9.5.2 的核心目标不是继续调 APX / CERT 阈值，也不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，而是解决一个更根本的问题：
>
> $$
> \boxed{
> \text{functional update 的 official multi-horizon target 到底应该是什么，以及这个 target 是否有足够密度、可见性、可生成性和可证书性？}
> }
> $$

---

# 0. 执行摘要

v9.5.1 的真实进展是把工程链路推进到了一个比较干净的状态：canonical truth base 可用，APX1-APX8 真实生成，payload/certificate/action apply/preflight/branch-horizon materializer 都闭合；但是它同时证明当前路线没有产生 functional success。

关键数据如下：

```text
canonical_full_control_outcome_ready = 1
multi_horizon_objective_pass = 0
selected_official_target_candidate = T_C
T_A_count = 120
T_B_count = 25
T_C_count = 16
T_D_count = 64
legal_information_upper_bound_pass = 0
best_raw_probe_id = UB4-gradient-action-bilinear-probe
best_raw_AUC_TB = 0.44422307962118557
best_raw_TopK64_precision_TB = 0.0
mechanism_pass = 0
best_cluster_purity_TB = 0.0625
apx_primitive_family_spec_pass = 1
apx_preflight_pass = 1
branch_horizon_rows_expected_actual = 12288 / 12288
best_apx_primitive = APX7-EnsembleIntersectionPrimitive
best_apx_h20_weak_CP = 0.125
best_apx_h20_V_ctrl_lcb = -1.6380800012017955
best_apx_h240_longrisk = 0.84375
certificate_effect_valid_pass = 0
best_certificate = CERT24-AdamWConflictReliefBound
best_certificate_AUC_TB = 0.6041257367387033
best_certificate_TopK64_TB_precision = 0.0
best_certificate_TopK64_LongRisk = 0.734375
system_legal_controller_pass = 0
```

v9.5.1 最关键的发现不是“APX7 不行”或“CERT24 不行”，而是：

$$
\boxed{
\text{目标定义本身出现严格性与支持度冲突。}
}
$$

四个 target 的状态说明了这一点：

| target | count | coverage | h20 weak | h80 weak | h240 weak | h240 long-risk | integrated V LCB | official pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `T_A-LegacyYRobust` | 120 | 0.041725 | 1.0 | 0.175 | 0.566667 | 0.0 | -0.530451 | 0 |
| `T_B-StableHorizon` | 25 | 0.008693 | 1.0 | 0.84 | 0.72 | 0.0 | 0.595976 | 0 |
| `T_C-StrictAllH` | 16 | 0.005563 | 1.0 | 1.0 | 1.0 | 0.0 | 0.784806 | 0 |
| `T_D-IntegratedRobustScoreTop` | 64 | 0.022253 | 0.625 | 0.6875 | 0.71875 | 0.015625 | 0.487061 | 0 |

按 full action count $N=2876$ 计算，coverage 下限 $0.03$ 要求 action count 至少为：

$$
K_{min}=\lceil 0.03 \times 2876 \rceil = 87.
$$

因此：

```text
T_A: 120 actions，支持度够，但 h80 / integrated value 不够。
T_B: 25 actions，质量较好，但少 62 个 action 才到 coverage 下限。
T_C: 16 actions，最严格，但少 71 个 action 才到 coverage 下限。
T_D: 64 actions，折中最好，但仍少 23 个 action 才到 coverage 下限。
```

这说明 v9.5.2 不能再从 APX/CERT 的阈值继续修。下一轮首先要解决：

```text
1. official target 是否应是单一二元 label，还是 multi-horizon vector objective；
2. 如果是 binary target，是否存在 coverage >= 0.03 且 integrated value > 0 的可用 target；
3. 如果不存在，是否需要用 portfolio / staged controller / generated action density 增加支持度；
4. 如果存在，legal information upper-bound 是否能看见；
5. 如果看不见，是否能构造 objective-solved primitive，而不是继续从 opaque AP0/APX action 中筛。 
```

---

# 1. 独立数据判断

## 1.1 这轮是否有进展

有进展，但不是能力进展，而是 target-level 和 implementation-level 的进展。

v9.5.1 至少完成了以下真实推进：

```text
1. 复现 v9.5.0 boundary，没有跳过 PrimitiveFamilyResetRequired；
2. 将 legacy YRobust 目标升级为 T_A/T_B/T_C/T_D multi-horizon target audit；
3. APX1-APX8 primitive family spec 真实 materialize；
4. 512 个 generated actions 完整写出 payload / certificate；
5. action apply L∞ max = 0.0；
6. deterministic preflight ladder 过；
7. 12288 / 12288 branch-horizon outcome rows 完整落盘；
8. APX8 negative control 没有误通过；
9. certificate v4 的 effect-validity 真实被测，而不是 schema-only；
10. P9-P14 被正确 gate-block，没有把 diagnostic 写成 official。
```

这说明最近的慢不是因为系统一直空转，而是因为每轮都在把一个隐藏假设拆掉。v9.5.1 拆掉的隐藏假设是：

$$
\boxed{
\text{只要有 canonical robust frontier，就自然存在一个可 official 的 multi-horizon target。}
}
$$

现在看，这个假设不成立。

## 1.2 为什么感觉慢

你的感觉是合理的。最近的进展主要是“实验可信性”和“路线排除”，不是“模型能力提升”。从 v9.4.7 到 v9.5.1 的节奏大致是：

```text
v9.4.7: 修掉 no-transform replay semantics，旧 outcome table quarantine。
v9.4.8: canonical outcome universe 重建，canonical frontier 存在。
v9.4.9: canonical truth 上 legal / generator / certificate 均失败。
v9.5.0: high-capacity legal probe、mechanism anatomy、SG1-SG4、CERT14-CERT18 均失败。
v9.5.1: APX1-APX8 工程闭合，但 multi-horizon target 本身冲突，APX/CERT 仍失败。
```

这不是完全没进度，但它确实还没有进入 selected controller、selected runtime、paired replay 或 short/full training 的正反馈闭环。v9.5.2 必须改变执行策略：

```text
不要一轮只清一个 blocker；
要并行跑 target lattice、legal upper-bound、APX re-score、APY generator、certificate 和 runtime preflight；
同时设置 preflight stop conditions，避免跑完整大 runner 后才发现 target 不成立。
```

## 1.3 当前真正卡在哪里

当前 primary blocker 不是 canonical truth base，不是 payload，不是 materializer，不是 no-transform replay，也不是 Base-Acc。真正卡在：

$$
\boxed{
\text{multi-horizon official target 没有闭合。}
}
$$

进一步分解为四个子问题。

第一，target strictness 与 support 冲突。T_A 支持度够，但 h80 weak 只有 $0.175$，integrated V LCB 为 $-0.530451$；T_B/T_C 质量好，但 coverage 分别只有 $0.008693$ 和 $0.005563$，远低于 $0.03$；T_D 最接近，但 coverage 也只有 $0.022253$。

第二，legal information upper-bound 对更严格 target 看不见。best raw probe 是 `UB4-gradient-action-bilinear-probe`，但 AUC_TB 只有 $0.4442$，TopK64 T_B precision 为 $0.0$，TopK64 long-risk 为 $0.609375$。这不是阈值问题，而是当前 legal representation 对 multi-horizon robust target 方向错了。

第三，APX generator 不是 value-producing primitive。APX1-APX8 工程链路过了，但 best APX7 的 h20 weak CP 只有 $0.125$，h20 V_ctrl LCB 为 $-1.6381$，h240 long-risk 高达 $0.84375$。这说明当前 APX primitive 只是合法生成 payload，没有解决 multi-horizon constrained value objective。

第四，certificate 不是 effect-valid certificate。CERT24 的 AUC_TB 虽然有 $0.6041$，但 TopK64 T_B precision 是 $0.0$，TopK64 long-risk 是 $0.734375$，说明 certificate 高分区没有选到 stable-horizon good actions，反而选到了大量 long-risk。

## 1.4 是否在正确道路上

高层方向仍然正确，因为项目没有把 Base-Acc Sentinel、APX preflight、branch-horizon rows、certificate diagnostic 或 runtime boundary 写成 success，也没有为了追指标按 dataset 调参。

但具体路线必须再 pivot。v9.5.2 不能继续做：

```text
APX7 threshold 微调；
CERT24 threshold 微调；
UB4 topK 微调；
继续用 T_A 作为唯一 target；
继续用 T_C 作为唯一 strict target；
继续用单一 binary label 承载所有 horizon 目标；
继续在 APX1-APX8 上加小变体。
```

正确路线是：

```text
1. 先解决 multi-horizon target 的定义与支持度；
2. 再判断 legal information upper-bound 是否存在；
3. 如果 legal upper-bound 不存在，停止 selection 主线；
4. 并行实现 objective-solved primitive，而不是 heuristic primitive；
5. certificate 必须预测 vector effect，而不是 binary target；
6. 只有 target + primitive + certificate 三者同时过，才打开 controller/runtime。
```

## 1.5 离目标还差多远

离 system-legal local functional controller 至少还差四道门：

```text
1. official multi-horizon target closure；
2. legal observability 或 objective-solved generation closure；
3. effect-valid certificate / minimal controller closure；
4. selected controller runtime closure。
```

离 strict PureKAN functional causal evidence 还要：

```text
leave-dataset-out；
leave-stratum-out；
official paired replay；
RealFunctional beats AdamWParallel / bestLR / NoOp / Random；
shuffle controls fail。
```

离 external-ready Beyond-MLP 还要：

```text
short/full training；
sample efficiency；
calibration / robustness；
continual / anti-forgetting；
strong MLP baseline 排除；
external reproducibility。
```

因此当前不是“快成功了”。更准确的状态是：

$$
\boxed{
\text{truth base 和工程链路基本可信；现在卡在 multi-horizon target 与 objective-solved primitive。}
}
$$

---

# 2. v9.5.2 总体目标

v9.5.2 的总体目标是：

$$
\boxed{
\text{解决 multi-horizon target 的严格性/支持度冲突，并验证是否存在 objective-solved functional primitive 能产生可证书化的 multi-horizon value。}
}
$$

v9.5.2 不以 full functional success 为最低目标。最低有效推进是：

```text
1. 明确 official target 不是 T_A/T_B/T_C/T_D 中哪一个简单照搬，而是经过 preregistered target lattice 选择；
2. 找到一个 coverage >= 0.03、integrated value LCB > 0、h240 long-risk 可控、support balance 过的 target，或证明不存在；
3. 如果 target 存在，测 legal upper-bound 是否能识别；
4. 如果 legal upper-bound 不存在，停止 selection 主线，转向 constructive generation；
5. 实现 APY1-APY8 objective-solved primitive；
6. 对 APY 做 deterministic preflight、branch-horizon smoke、target alignment、damage、certificate；
7. 若 APY 和 certificate 过线，才打开 minimal controller 和 selected runtime；
8. 若 APY 也失败，明确进入 action primitive mathematical redesign，而不是继续 APX patch。
```

---

# 3. 硬约束

本轮继续遵守：

```text
no teacher
no self-teacher
no distillation
no loss modification
no label smoothing
no focal / margin / calibration loss
no sampler / class weight
no CPU offload
no fake / proxy rows
KAN path 不使用 PyTorch loss.backward graph
official controller 不使用 dataset_name 分支
no validation/test metric at commit time
no future outcome feature
no outcome-at-commit feature
no source-measured gap / formula proxy for official
PureKANConv / PureKANFormer 继续 deferred
```

Functional update 仍然是 update rule，不是 loss：

$$
\theta_{t+1}=\theta_t+\Delta\theta_{AdamW}+\Delta\theta_{functional}.
$$

任务 loss 仍然是标准 CE：

$$
L_{task}=CE(y,p_\theta(x)).
$$

v9.5.2 新增硬约束：

```text
不能把 T_A / T_B / T_C / T_D 之一直接 hard-code 为 official target；
不能把 APX7 best primitive 继续阈值化当成新路线；
不能把 CERT24 的 AUC_TB=0.604 当成 effect-valid pass；
不能以 TopK64 T_B precision=0 的 certificate 进入 controller；
不能因为 T_C strict quality 高就忽视 coverage=0.00556；
不能因为 T_A coverage 过 0.03 就忽视 h80 blind spot 和 integrated V LCB negative；
不能在 target conflict 未解决前打开 paired replay；
不能在 controller 未选中前打开 selected runtime official；
不能把 Base-Acc Sentinel 当 functional success。
```

允许做：

```text
target lattice / Pareto frontier diagnostic；
legal information upper-bound probe；
mechanism anatomy；
objective-solved primitive generation；
multi-horizon vector certificate；
portfolio / staged target diagnostic；
Base-Acc Sentinel continuation；
per-dataset diagnostics，但不得 dataset-specific tuning。
```

---

# 4. 核心定义

## 4.1 Horizon value vector

对每个 action $a$，定义：

$$
\mathbf{V}(a)=(V_{20}(a),V_{80}(a),V_{240}(a)).
$$

其中：

$$
V_h(a)=V_{RealFunctional}(a,h)-\max_{b\in Controls}V_b(a,h).
$$

Controls 包含：

```text
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ShuffledFunctionalPayload when available
```

## 4.2 Integrated value

定义 weighted integrated value：

$$
V_{int}(a;w)=w_{20}V_{20}(a)+w_{80}V_{80}(a)+w_{240}V_{240}(a),
$$

其中：

$$
w_{20}+w_{80}+w_{240}=1,
$$

且 v9.5.2 的默认非数据集调参权重为：

$$
w=(0.30,0.40,0.30).
$$

h80 权重稍高，因为 v9.5.0/v9.5.1 已经暴露 legacy YRobust 的 h80 blind spot。

## 4.3 Multi-horizon risk

定义：

$$
R_{long}(a)=\mathbb{1}[LongRisk_{240}(a)=1],
$$

$$
R_{bad}(a)=\max_h BadEvent_h(a),
$$

$$
R_{null}(a)=\max_h NullEvent_h(a).
$$

## 4.4 Target candidate family

v9.5.2 不直接选 T_A/T_B/T_C/T_D，而是构造 target lattice：

$$
T(\alpha,\beta,\gamma,\rho,\eta,w)=1
$$

当且仅当：

$$
V_{20}(a) \ge \alpha,
$$

$$
V_{80}(a) \ge \beta,
$$

$$
V_{240}(a) \ge \gamma,
$$

$$
R_{long}(a) \le \rho,
$$

$$
V_{int}(a;w) \ge \eta.
$$

其中 $\alpha,\beta,\gamma,\eta$ 只允许来自全局 preregistered grid，不允许按 dataset 调参。

## 4.5 Official target minimum

一个 target candidate 至少要满足：

$$
Coverage(T)\ge 0.03,
$$

$$
LCB(V_{int}|T)>0,
$$

$$
LongRisk_{240}(T)\le 0.05,
$$

$$
BadEvent(T)\le 0.05,
$$

$$
NullEvent(T)\le 0.15,
$$

$$
SupportBalance(T)=1.
$$

如果没有任何 $T$ 满足这些条件，则不能进入 controller，应进入 action density / primitive objective reset。

---

# 5. 核心假设

## H1：v9.5.1 的 target conflict 是真实 blocker，不是 APX/CERT 局部失败

H1 认为 v9.5.1 的 route 不应只读成 APX fail，而应读成 target definition fail。

H1 成立标准：

```text
T_A 支持度过，但 integrated value 或 h80 weak 不过；
T_B/T_C 质量更高，但 coverage 远低于 0.03；
T_D 接近但仍低于 0.03；
没有一个 T_A/T_B/T_C/T_D 同时满足 official target minimum。
```

H1 失败标准：

```text
重新审计发现某个 target 实际满足 coverage/value/risk/support，只是上一轮 route 判断错误。
```

## H2：存在一个 Pareto-relaxed target 可以同时满足 coverage 与 multi-horizon safety

H2 认为 T_A 太弱、T_C 太稀疏，但在 target lattice 中可能存在一个 $T_E$：

```text
coverage >= 0.03
integrated V LCB > 0
h80 不崩
h240 long-risk <= 0.05
support balance pass
```

H2 成立标准：

```text
至少一个 target lattice candidate official_target_pass = 1。
```

H2 失败标准：

```text
所有 preregistered target candidates 要么 coverage <0.03，要么 integrated V LCB <=0，要么 h240 long-risk >0.05。
```

若 H2 失败，应判定 current action universe 的 multi-horizon robust density 不足，进入 source/action primitive density reset。

## H3：当前 legal information upper-bound 对 corrected target 仍可能为零

H3 认为即使 target 重定义，现有 legal features/probes 也可能看不见。v9.5.1 对 T_B 的 best AUC 只有 $0.4442$，说明 selection route 可能已经没有上界。

H3 成立标准：

```text
对 selected target T_E，best high-capacity legal probe AUC < 0.65；
TopK64 target precision <= max(0.05, 2 * base_rate)；
TopK64 long-risk >= 0.30。
```

H3 失败标准：

```text
某个 legal upper-bound probe 对 T_E AUC >=0.75，且 TopK64 precision 显著高于 base rate，long-risk <=0.10。
```

若 H3 成立，不再从 AP0/APX existing action 做 selector patch，转向 objective-solved generator。

## H4：APX primitive 失败是 objective mismatch，而不是工程未闭合

H4 认为 APX1-APX8 已经工程闭合，但它们不是在解 multi-horizon constrained objective。

H4 成立标准：

```text
APX preflight pass = 1；
branch-horizon rows complete = 1；
best APX V_ctrl LCB remains negative；
best APX h240 long-risk remains high；
APX rescore under T_E still fails。
```

H4 失败标准：

```text
APX rescore under selected T_E 后发现某个 APX primitive 过 target gate。
```

## H5：objective-solved APY primitive 可以比 heuristic APX 更接近 target

H5 认为新 primitive 必须直接优化：

$$
\max_{\Delta\theta} \quad \widehat{V}_{int}(\Delta\theta)
-\lambda_{long}\widehat{R}_{long}(\Delta\theta)
-\lambda_{bad}\widehat{R}_{bad}(\Delta\theta)
-\lambda_{cost}C(\Delta\theta),
$$

并满足：

$$
\|\Delta\theta\|\le r,
$$

$$
\cos(\Delta\theta,\Delta\theta_{AdamW})\ge c_{min} \quad \text{or controlled orthogonal residual bound.}
$$

H5 成立标准：

```text
至少一个 APY primitive 在 smoke outcome 中满足 target weak pass：
coverage equivalent >=0.03 or accepted_count >= K_min_on_panel
V_int LCB >0
h240 longrisk <=0.05 or <=0.10 weak
bad/null gates pass
```

H5 失败标准：

```text
所有 APY primitives 与 APX 类似：h20 V LCB negative，h240 longrisk high，target precision near base rate。
```

## H6：certificate 必须预测 vector effect，而不是 binary label

H6 认为 CERT24 失败的原因是 certificate 只在标量 score 上排序，而 multi-horizon target 需要预测：

$$
\widehat{V}_{20},\widehat{V}_{80},\widehat{V}_{240},\widehat{R}_{long},\widehat{Bad},\widehat{Null},\widehat{Cost}.
$$

H6 成立标准：

```text
certificate vector calibration pass；
AUC_target >=0.75；
AUC_longrisk >=0.75；
TopK64 target precision >=0.30；
TopK64 longrisk <=0.10；
ECE_target <=0.08；
calibration_to_heldout_drift <=0.10。
```

H6 失败标准：

```text
certificate still has TopK64 target precision = 0 or longrisk high。
```

## H7：Base-Acc Sentinel 只能作为 base health，不是 functional success

H7 成立标准：

```text
Base-Acc Sentinel rows complete；
base_acc_used_for_controller = 0；
no dataset-specific tuning；
LQ catastrophic fail = 0。
```

H7 失败标准：

```text
Base-Acc Sentinel 被用于 selector/controller 或出现 LQ catastrophic regression。
```

---

# 6. 数据合同

## 6.1 Target lattice table

每个 target candidate 必须记录：

```text
target_id
alpha_h20
beta_h80
gamma_h240
eta_integrated
rho_longrisk
w20
w80
w240
action_count
coverage
coverage_lcb
h20_weak_rate
h80_weak_rate
h240_weak_rate
h20_V_mean
h80_V_mean
h240_V_mean
h20_V_lcb
h80_V_lcb
h240_V_lcb
V_integrated_mean
V_integrated_lcb
h240_longrisk_rate
bad_event_rate
null_event_rate
support_balance_pass
accepted_family_count
accepted_stratum_count
max_family_share
max_stratum_share
jaccard_TA
jaccard_TB
jaccard_TC
jaccard_TD
official_target_pass
weak_target_pass
reason_if_failed
```

## 6.2 Legal upper-bound probe table

每 probe 必须记录：

```text
probe_id
probe_class
legal_feature_groups
uses_dataset_name
uses_validation_or_test
uses_future_outcome
uses_outcome_at_commit
uses_source_measured_gap
feature_cost_ms_q50
feature_cost_ms_q90
memory_ratio
AUC_target
AUC_longrisk
AUC_bad
AUC_null
PR_AUC_target
TopK16_target_precision
TopK64_target_precision
TopK64_longrisk
TopK64_bad
TopK64_null
LDO_AUC_drop_max
LSO_AUC_drop_max
probe_pass
```

## 6.3 APY primitive table

每 generated action 必须记录：

```text
primitive_id
action_id
source_action_id
payload_hash
certificate_hash
objective_id
trust_region_radius
solver_status
solver_iterations
linearized_V20_hat
linearized_V80_hat
linearized_V240_hat
risk_long_hat
bad_hat
null_hat
payload_norm
payload_linf
cos_adamw
cos_negative_grad
action_apply_linf
commit_time_available
uses_dataset_name
uses_future_outcome
uses_outcome_at_commit
```

## 6.4 Branch-horizon outcome table

每 generated action 的 outcome 必须记录：

```text
action_id
primitive_id
branch_id
horizon
V_ctrl
weak_CP
strong_CP
bad_event
null_event
long_risk
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
row_quality_pass
```

## 6.5 Certificate table

每 certificate 必须记录：

```text
certificate_id
primitive_id
action_id
V20_hat
V80_hat
V240_hat
Vint_hat
Bad_hat
Null_hat
LongRisk_hat
Support_hat
Cost_hat
certificate_score
certificate_pass
payload_hash_bound
calibration_bin
target_label
longrisk_label
AUC_target
AUC_longrisk
ECE_target
ECE_longrisk
TopK64_target_precision
TopK64_longrisk
monotone_sign_pass
calibration_to_heldout_drift
```

## 6.6 Runtime table

只有 controller 选中后才 official runtime。必须记录：

```text
runtime_candidate_id
controller_id
primitive_id
step_count
active_step_count
accepted_action_count
payload_apply_time_ms_q50
payload_apply_time_ms_q90
feature_compute_ms_q90
certificate_score_ms_q90
controller_decision_ms_q90
base_step_time_ms_q90
total_step_time_ms_q90
mlp_step_time_ms_q90
step_ratio_q90
memory_ratio
selected_runtime_pass
```

---

# 7. 实验阶段

---

## P0：v9.5.1 boundary reproduction

### 目标

确认 v9.5.1 的 target conflict / APX / CERT boundary 可复现，不在不稳定 artifact 上继续。

### 假设

H1/P0：v9.5.1 的 primary blocker 是 target conflict，不是 canonical truth base 或 materializer failure。

### 必须记录

```text
route_v9510
source_route_v9500
canonical_full_control_outcome_ready
multi_horizon_objective_pass
T_A_count
T_B_count
T_C_count
T_D_count
T_A_coverage
T_B_coverage
T_C_coverage
T_D_coverage
best_raw_probe_id
best_raw_AUC_TB
best_raw_TopK64_precision_TB
best_cluster_purity_TB
apx_primitive_family_spec_pass
apx_preflight_pass
apx_branch_horizon_rows_expected
apx_branch_horizon_rows_actual
best_apx_primitive_id
best_apx_h20_weak_CP
best_apx_h20_V_ctrl_lcb
best_apx_h240_longrisk
best_certificate_id
best_certificate_AUC_TB
best_certificate_TopK64_TB_precision
system_legal_controller_pass
fake_data_used
proxy_row_used
cpu_offload_used
```

### 判断标准

P0 pass：

```text
route_v9510 = R1-TargetConflictUnresolved
canonical_full_control_outcome_ready = 1
APX materialization/preflight/outcome rows complete
system_legal_controller_pass = 0
fake/proxy/offload = 0
```

### 可视化

```text
p0_v9510_boundary_ladder.svg
p0_target_conflict_summary.svg
p0_apx_cert_boundary.svg
```

---

## P1：multi-horizon target lattice resolution

### 目标

解决 T_A/T_B/T_C/T_D 的严格性/支持度冲突，确定是否存在 official-eligible multi-horizon target。

### 假设

H2：存在一个 Pareto-relaxed target $T_E$，能同时满足 coverage、integrated value、long-risk、bad/null 和 support balance。

### 实验设计

构造 target lattice：

```text
alpha_h20_grid = {-0.05, 0.00, 0.02, 0.05, 0.10}
beta_h80_grid  = {-0.05, 0.00, 0.02, 0.05, 0.10}
gamma_h240_grid = {-0.05, 0.00, 0.02, 0.05, 0.10}
eta_integrated_grid = {0.00, 0.02, 0.05, 0.10}
rho_longrisk_grid = {0.00, 0.02, 0.05, 0.10}
w_grid = {(0.30,0.40,0.30), (0.33,0.34,0.33), (0.25,0.50,0.25), (0.40,0.30,0.30)}
```

每个 target candidate 按以下公式生成：

$$
T_E(a)=1
\iff
V_{20}(a)\ge\alpha
\land
V_{80}(a)\ge\beta
\land
V_{240}(a)\ge\gamma
\land
V_{int}(a;w)\ge\eta
\land
LongRisk_{240}(a)\le\rho.
$$

同时保留两个 diagnostic target：

```text
T_Portfolio: 允许分层接受 h20-positive / h80-positive / h240-safe action 组合，但必须报告 per-horizon risk。
T_Staged: stage1 h20 value, stage2 h80 veto, stage3 h240 long-risk veto。
```

T_Portfolio / T_Staged 只能 diagnostic，除非它们最终可写成单一 commit-time rule。

### 必须记录

```text
target_id
threshold_tuple
weight_tuple
action_count
coverage
coverage_lcb
h20_weak_rate
h80_weak_rate
h240_weak_rate
h20_V_lcb
h80_V_lcb
h240_V_lcb
V_integrated_lcb
h240_longrisk_rate
bad_event_rate
null_event_rate
support_balance_pass
accepted_family_count
accepted_stratum_count
jaccard_to_TA_TB_TC_TD
official_target_pass
weak_target_pass
failure_reason
```

### 判断标准

P1 official target pass：

$$
Coverage(T_E)\ge0.03,
$$

$$
LCB(V_{int}|T_E)>0,
$$

$$
LongRisk_{240}(T_E)\le0.05,
$$

$$
BadEvent(T_E)\le0.05,
$$

$$
NullEvent(T_E)\le0.15,
$$

$$
SupportBalance(T_E)=1.
$$

Weak target pass：

$$
Coverage(T_E)\ge0.03,
$$

$$
LCB(V_{int}|T_E)>0,
$$

$$
LongRisk_{240}(T_E)\le0.10.
$$

P1 fail：

```text
no target candidate satisfies weak target pass。
```

若 P1 fail，v9.5.2 route 直接进入：

```text
R1-CanonicalActionDensityInsufficientForMultiHorizonTarget
```

并停止 controller / runtime official path。

### 可视化

```text
p1_target_lattice_coverage_vs_Vint.svg
p1_target_lattice_longrisk_vs_coverage.svg
p1_target_pareto_frontier.svg
p1_target_jaccard_heatmap.svg
p1_horizon_value_curve_by_target.svg
p1_target_support_balance_bars.svg
```

---

## P2：target stability and leave-out sanity

### 目标

验证 P1 选出的 target 不是某个 dataset、seed、family、step bucket 的 artifact。允许诊断 dataset 差异，但不得按 dataset 调参。

### 假设

H2b：若 target 真正适合作为 official objective，它应在 leave-dataset / leave-family / leave-stratum 中不完全崩。

### 设置

```text
Leave-dataset-out:
  MNIST heldout
  Fashion-MNIST heldout
  KMNIST heldout

Leave-family-out:
  top 8 high-volume family each held out

Leave-stratum-out:
  signal strata bucket held out

Leave-step-bucket-out:
  early/mid/late event step buckets held out
```

### 必须记录

```text
target_id
split_type
heldout_entity
train_count
heldout_count
coverage_train
coverage_heldout
V_integrated_lcb_train
V_integrated_lcb_heldout
longrisk_train
longrisk_heldout
bad_event_train
bad_event_heldout
null_event_train
null_event_heldout
support_balance_heldout
coverage_drop
longrisk_increase
value_lcb_drop
```

### 判断标准

P2 pass：

```text
no heldout dataset has coverage = 0；
coverage_drop_max <= 0.50 relative；
longrisk_heldout <= 0.10 for weak pass or <=0.05 for strong pass；
at least 2/3 datasets satisfy weak target pass；
no family dominates accepted actions above 0.50 share。
```

P2 fail：

```text
target only exists in one dataset/family/step bucket；
coverage collapses to 0 in any main dataset；
longrisk explodes in leaveout。
```

### 可视化

```text
p2_leave_dataset_target_matrix.svg
p2_leave_family_target_matrix.svg
p2_leave_stratum_target_matrix.svg
p2_target_coverage_drop_waterfall.svg
```

---

## P3：legal information upper-bound v3 on selected target

### 目标

判断 corrected target 是否在 commit time 可见。如果 high-capacity legal upper-bound 都看不见，selection 路线应停止。

### 假设

H3：当前 legal information upper-bound 很可能仍然失败，但必须在 corrected target 上重新测，而不是只看 T_B。

### Probe candidates

```text
UB0-scalar-feature-probe-v3
UB1-tensor-sketch-centroid-v3
UB2-legal-KNN-v3
UB3-gradient-action-bilinear-v3
UB4-hard-tail-response-v3
UB5-basis-edge-locality-v3
UB6-raw-logit-tail-tensor-v3
UB7-small-mlp-diagnostic-legal-v3
UB8-linearized-multihorizon-response-v3
```

所有 probe 只能使用 commit 前信息。`UB7-small-mlp` 只能作为 upper-bound diagnostic，不能直接 official。

### 必须记录

```text
probe_id
feature_groups
AUC_target
AUC_longrisk
AUC_bad
AUC_null
PR_AUC_target
TopK16_target_precision
TopK64_target_precision
TopK64_longrisk
TopK64_bad_event
feature_cost_ms_q90
memory_ratio
LDO_AUC_drop_max
LSO_AUC_drop_max
uses_dataset_name
uses_future_outcome
uses_outcome_at_commit
probe_pass
```

### 判断标准

Legal upper-bound strong pass：

$$
AUC_{target}\ge0.75,
$$

$$
AUC_{longrisk}\ge0.75,
$$

$$
TopK64Precision_{target}\ge0.30,
$$

$$
TopK64LongRisk\le0.10.
$$

Weak pass：

$$
AUC_{target}\ge0.68,
$$

$$
TopK64Precision_{target}\ge0.20,
$$

$$
TopK64LongRisk\le0.20.
$$

P3 fail：

```text
all probes below weak pass。
```

If P3 fail:

```text
existing-action selection route stopped；
feature distillation not run；
controller not run from legal selection；
continue only constructive primitive route。
```

### 可视化

```text
p3_probe_auc_bar.svg
p3_topk_precision_vs_longrisk.svg
p3_probe_cost_vs_signal.svg
p3_legal_upper_bound_roc_pr.svg
p3_leaveout_probe_stability.svg
```

---

## P4：mechanism anatomy v3

### 目标

理解 selected target positives 与 near-misses 的机制差异，避免盲目继续加 feature 或 primitive。

### 假设

H3b：如果 target positives 是 outcome-only pattern，则 legal selection 很难；如果存在 cluster/prototype，则可转化为 certificate/generator constraints。

### 分析对象

```text
positive target actions
near-miss actions:
  high h20 but bad h80
  high h20/h80 but h240 long-risk
  positive integrated V but bad/null
  legal-probe false positives
  legal-probe false negatives
```

### 必须记录

```text
cluster_id
cluster_size
target_purity
longrisk_rate
bad_event_rate
null_event_rate
mean_V20
mean_V80
mean_V240
mean_payload_norm
mean_cos_adamw
mean_tail_fraction
mean_support_memory_similarity
prototype_reconstruction_error
miss_reason
```

Miss reasons：

```text
M1-h80-middle-horizon-collapse
M2-h240-longrisk-collapse
M3-integrated-value-negative
M4-bad-null-conflict
M5-low-support
M6-legal-feature-invisible
M7-generator-distortion
M8-outcome-only-pattern
M9-label-target-definition-conflict
```

### 判断标准

P4 mechanism pass：

```text
at least one cluster has target_purity >=0.30 and longrisk_rate <=0.10；
prototype_reconstruction_error <=0.30；
mechanism can be expressed using commit-time tensors or generator constraints。
```

P4 fail：

```text
best cluster purity <0.10 or dominant miss remains M8-outcome-only-pattern。
```

### 可视化

```text
p4_positive_vs_nearmiss_umap.svg
p4_cluster_purity_longrisk.svg
p4_mechanism_feature_shift.svg
p4_miss_reason_stacked_bar.svg
p4_horizon_failure_typology.svg
```

---

## P5：APX1-APX8 re-score under corrected target

### 目标

确认 APX failure 是否只是 target mismatch，还是 primitive family universal failure。

### 假设

H4：APX 在 corrected target 下仍失败。

### 必须记录

```text
primitive_id
generated_action_count
branch_horizon_rows
coverage_equivalent
h20_weak_CP
h80_weak_CP
h240_weak_CP
V20_lcb
V80_lcb
V240_lcb
V_integrated_lcb
h240_longrisk
target_precision
T_E_precision
source_to_generated_damage
new_positive_created_rate
longrisk_created_rate
apx_rescore_pass
```

### 判断标准

P5 pass：

```text
some APX primitive satisfies weak target gate under T_E。
```

P5 fail：

```text
all APX primitives have V_integrated_lcb <=0 or h240_longrisk >0.10 or target_precision near base rate。
```

### 可视化

```text
p5_apx_target_precision_by_primitive.svg
p5_apx_horizon_value_curves.svg
p5_apx_longrisk_by_primitive.svg
p5_apx_damage_matrix.svg
```

---

## P6：APY1-APY8 objective-solved primitive family

### 目标

实现一组真正围绕 multi-horizon objective 生成 action 的 primitive，而不是继续 heuristic payload builder。

### Primitive candidates

### APY1：ConstrainedMultiHorizonQP

求解近似 constrained QP：

$$
\max_{\Delta\theta}\quad \widehat{V}_{int}(\Delta\theta)-\lambda_{long}\widehat{R}_{long}(\Delta\theta)-\lambda_{bad}\widehat{Bad}(\Delta\theta)
$$

subject to：

$$
\|\Delta\theta\|_2\le r,
$$

$$
\widehat{V}_{80}(\Delta\theta)\ge \beta,
$$

$$
\widehat{R}_{long}(\Delta\theta)\le \rho.
$$

### APY2：H80MiddleHorizonRepair

针对 v9.5.0/v9.5.1 的 h80 blind spot，显式优化 h80 response：

$$
\max_{\Delta\theta} \quad \widehat{V}_{80}(\Delta\theta)-\lambda\widehat{R}_{240}(\Delta\theta).
$$

### APY3：LongRiskBarrierPrimitive

优先降低 h240 long-risk，再保 h20/h80 value：

$$
\max_{\Delta\theta}\quad \widehat{V}_{20}+\widehat{V}_{80}-\lambda_{barrier}\max(0,\widehat{R}_{240}-\rho)^2.
$$

### APY4：AdamWCompatibleResidualQP

在 AdamW 方向附近加入 constrained residual：

$$
\Delta\theta=\Delta\theta_{AdamW}+\Delta\theta_{res},
$$

$$
\cos(\Delta\theta_{res},\Delta\theta_{AdamW})\ge c_{min}\quad \text{or}\quad \Delta\theta_{res}\perp \Delta\theta_{AdamW}.
$$

### APY5：SupportMemoryPrototypePrimitive

使用 P4 mechanism prototype 中的 support-memory pattern，但只作为 generator constraint，不作为 outcome proxy。

### APY6：BasisEdgeLocalityPrimitive

限制更新在 low-risk basis/edge locality 中：

```text
last-edge / low-rank edge blocks only；
no global dense perturbation；
explicit payload cost budget。
```

### APY7：PortfolioMicroActionPrimitive

生成多个小 payload 候选，controller 只允许选择一小部分，测试是否单 action 过稀疏但 portfolio 能提高 density。

### APY8：NegativeControlRandomOrthogonal

随机正交 negative control，必须不通过。

### 必须记录

```text
primitive_id
generated_action_count
solver_status
solver_iterations
payload_hash_missing_count
certificate_hash_missing_count
action_apply_linf_max
commit_time_available
uses_dataset_name
uses_future_outcome
uses_outcome_at_commit
preflight_single_pass
preflight_three_pass
preflight_sixteen_pass
negative_control_divergence_present
```

### 判断标准

P6 implementation pass：

```text
all APY primitives materialized；
generated_action_count_expected = generated_action_count_actual；
payload/certificate hash missing = 0；
action_apply_linf_max <= 1e-7；
preflight ladder pass；
negative control generated。
```

P6 science pass 由 P7 outcome 决定。

### 可视化

```text
p6_apy_payload_norm_distribution.svg
p6_apy_solver_status.svg
p6_apy_cos_adamw_distribution.svg
p6_apy_preflight_ladder.svg
```

---

## P7：APY branch-horizon smoke outcome

### 目标

真实测量 APY primitives 是否产生 multi-horizon value-positive、horizon-safe frontier。

### 设置

```text
APY primitives = APY1-APY8
actions_per_primitive = 128 preferred, 64 minimum
branches = RealAPY, AdamWOnly, AdamWParallel, bestLR, NoOp, Random
horizons = 20, 80, 240
```

### 必须记录

```text
primitive_id
generated_action_count
branch_horizon_rows_expected
branch_horizon_rows_actual
branch_completion_rate
horizon_completion_rate
secondary_delta_completion_rate
rows_per_sec
wallclock_sec
h20_weak_CP
h80_weak_CP
h240_weak_CP
h20_V_lcb
h80_V_lcb
h240_V_lcb
V_integrated_lcb
h240_longrisk
bad_event_rate
null_rate
target_precision
YStable_precision
YStrict_precision
support_balance_pass
```

### 判断标准

APY weak pass：

$$
V_{int,LCB}>0,
$$

$$
LongRisk_{240}\le0.10,
$$

$$
TargetPrecision\ge0.20,
$$

and at least:

```text
accepted_count_equivalent >= 64 on smoke panel
negative control APY8 weak pass = 0
```

APY strong pass：

$$
V_{int,LCB}>0,
$$

$$
LongRisk_{240}\le0.05,
$$

$$
TargetPrecision\ge0.30.
$$

P7 fail：

```text
all APY primitives have V_int_lcb <=0 or longrisk >0.10。
```

### 可视化

```text
p7_apy_horizon_value_curves.svg
p7_apy_target_precision_by_primitive.svg
p7_apy_longrisk_by_primitive.svg
p7_apy_vint_vs_longrisk_pareto.svg
p7_apy_negative_control_check.svg
```

---

## P8：source-to-generated preservation / improvement audit

### 目标

判断 APY 是保留 source frontier、创造新 positive，还是制造 long-risk。

### 必须记录

```text
primitive_id
source_panel_id
source_positive_count
generated_positive_count
source_positive_lost_rate
source_negative_fixed_rate
new_positive_created_rate
longrisk_created_rate
Damage_h20_mean
Damage_h20_lcb
Damage_h80_mean
Damage_h80_lcb
Damage_h240_mean
Damage_h240_lcb
Damage_integrated_lcb
preservation_pass
improvement_pass
```

### 判断标准

Preservation pass：

```text
source_positive_lost_rate <=0.25
Damage_integrated_lcb >= -0.02
longrisk_created_rate <=0.05
```

Improvement pass：

```text
new_positive_created_rate >=0.10
Damage_integrated_lcb >0
longrisk_created_rate <=0.10
```

### 可视化

```text
p8_damage_matrix_by_primitive.svg
p8_source_generated_sankey.svg
p8_new_positive_vs_longrisk.svg
p8_preservation_improvement_frontier.svg
```

---

## P9：effect-valid vector certificate v5

### 目标

构建能预测 multi-horizon vector effect 的 certificate，而不是 binary label score。

### Certificate candidates

```text
CERT25-VectorLinearizedValueCertificate
CERT26-H80GuardCertificate
CERT27-LongRiskBarrierCertificate
CERT28-AdamWCompatibilityCertificate
CERT29-SupportMemoryVectorCertificate
CERT30-BasisEdgeLocalityCertificate
CERT31-PortfolioActionCertificate
CERT32-MinimalMonotoneVectorCertificate
```

Certificate score：

$$
S_{cert}(a)=
\widehat{V}_{int}(a)
-\lambda_{long}\widehat{LongRisk}_{240}(a)
-\lambda_{bad}\widehat{Bad}(a)
-\lambda_{null}\widehat{Null}(a)
+\lambda_s\widehat{Support}(a)
-\lambda_c\widehat{Cost}(a).
$$

### 必须记录

```text
certificate_id
primitive_id
AUC_target
AUC_longrisk
AUC_bad
AUC_null
ECE_target
ECE_longrisk
TopK16_target_precision
TopK64_target_precision
TopK64_longrisk
TopK64_bad_event
calibration_to_heldout_drift
monotone_sign_pass
certificate_effect_valid_pass
```

### 判断标准

Certificate weak pass：

$$
AUC_{target}\ge0.70,
$$

$$
AUC_{longrisk}\ge0.70,
$$

$$
TopK64Precision_{target}\ge0.20,
$$

$$
TopK64LongRisk\le0.20.
$$

Certificate strong pass：

$$
AUC_{target}\ge0.78,
$$

$$
AUC_{longrisk}\ge0.78,
$$

$$
TopK64Precision_{target}\ge0.30,
$$

$$
TopK64LongRisk\le0.10,
$$

$$
ECE_{target}\le0.08.
$$

### 可视化

```text
p9_certificate_roc_pr.svg
p9_certificate_reliability_diagram.svg
p9_topk_target_longrisk.svg
p9_certificate_ablation.svg
p9_vector_prediction_scatter.svg
```

---

## P10：minimal source certificate controller

### 目标

只有 P1 target pass、P7 APY pass、P9 certificate pass 后，才选择 controller。

### Controller form

$$
Accept(a)=1
\iff
S_{cert}(a)\ge\tau
\land
\widehat{LongRisk}_{240}(a)\le\tau_{long}
\land
\widehat{Bad}(a)\le\tau_b
\land
\widehat{Support}(a)\ge\tau_s
\land
Cost(a)\le C_{max}.
$$

### Splits

```text
seed folds: 3 folds
leave-dataset-out: 3 folds
leave-family-out: top families
leave-stratum-out: signal strata
```

Thresholds must be frozen on calibration fold and never changed per dataset.

### 必须记录

```text
controller_id
primitive_id
certificate_id
target_id
thresholds
calibration_fold
heldout_fold
accepted_count_cal
accepted_count_heldout
coverage_cal
coverage_heldout
target_precision_cal
target_precision_heldout
V_integrated_lcb_heldout
longrisk_heldout
bad_event_heldout
null_rate_heldout
support_balance_pass
accepted_family_count
accepted_stratum_count
max_family_share
max_stratum_share
dataset_name_used
validation_or_test_used
controller_pass
```

### 判断标准

P10 decision pass：

$$
Coverage_{heldout}\in[0.03,0.15],
$$

$$
TargetPrecision_{heldout}\ge0.30 \quad \text{or target-specific official threshold from P1},
$$

$$
LCB(V_{int,heldout})>0,
$$

$$
LongRisk_{heldout}\le0.10 \quad \text{weak},
$$

$$
LongRisk_{heldout}\le0.05 \quad \text{strong},
$$

$$
BadEvent_{heldout}\le0.05,
$$

$$
NullRate_{heldout}\le0.15.
$$

No support collapse:

```text
accepted_count_heldout > 0 in every main fold；
accepted_family_count >= 16 weak, >=32 strong；
max_family_share <=0.50。
```

### 可视化

```text
p10_controller_frontier.svg
p10_calibration_to_heldout_drift.svg
p10_leaveout_matrix.svg
p10_support_balance.svg
```

---

## P11：selected controller runtime

### 目标

测 selected controller 的 online runtime，不能用 preflight 或 no-controller smoke 替代。

### 必须记录

```text
runtime_candidate_id
controller_id
primitive_id
certificate_id
step_count
active_step_count
accepted_action_count
feature_compute_ms_q90
certificate_compute_ms_q90
controller_decision_ms_q90
payload_apply_ms_q90
base_step_ms_q90
total_step_ms_q90
mlp_step_ms_q90
step_ratio_q90
memory_ratio
audit_outside_timed_path
selected_runtime_pass
```

### 判断标准

Runtime weak pass：

$$
StepRatio_{q90}\le1.50,
$$

$$
MemoryRatio\le1.05.
$$

Runtime strong pass：

$$
StepRatio_{q90}\le1.20.
$$

### 可视化

```text
p11_runtime_waterfall.svg
p11_payload_apply_distribution.svg
p11_step_ratio_by_active_step.svg
p11_memory_ratio_trace.svg
```

---

## P12：system integration gate

### 目标

组合 P1/P7/P9/P10/P11 survivor，判断是否 system legal。

### 必须记录

```text
system_candidate_id
target_id
primitive_id
certificate_id
controller_id
runtime_candidate_id
target_pass
primitive_pass
certificate_pass
controller_pass
runtime_pass
contract_pass
no_fake_pass
no_proxy_pass
no_dataset_tuning_pass
system_legal_controller_pass
official_eligible
```

### 判断标准

System pass：

```text
target_pass = 1
primitive_pass = 1
certificate_pass = 1
controller_pass = 1
runtime_pass = 1
no fake/proxy/offload = 1
no dataset-specific branch = 1
```

If P12 fails, P13-P15 remain boundary not_run.

### 可视化

```text
p12_system_gate_dashboard.svg
p12_failure_ladder.svg
p12_contract_audit_matrix.svg
```

---

## P13：leave-dataset-out / leave-stratum-out official

### 目标

只有 P12 pass 后打开，验证 controller 不是 pooled artifact。

### 必须记录

```text
split_type
heldout_entity
target_id
controller_id
primitive_id
certificate_id
accepted_count
coverage
V_integrated_lcb
longrisk
bad_event
null_rate
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
shuffle_control_pass
```

### 判断标准

LDO pass：

```text
at least 2/3 heldout datasets pass weak decision gate；
no heldout dataset coverage = 0；
no dataset-specific threshold branch。
```

LSO pass：

```text
at least 70% heldout strata pass task-safe and long-risk constraints。
```

### 可视化

```text
p13_ldo_matrix.svg
p13_lso_matrix.svg
p13_shuffle_control.svg
```

---

## P14：official paired replay

### 目标

验证 RealFunctional/APY selected update 是否在 strong controls 下有 causal advantage。

### 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledPayload
```

### 必须记录

```text
branch
horizon
dataset
seed
accepted_count
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
V_ctrl
beats_adamwparallel
beats_bestlr
beats_noop
beats_random
shuffle_control_gap
```

### 判断标准

Paired replay pass：

$$
BeatRate(RealFunctional, AdamWParallel)\ge0.50,
$$

$$
BeatRate(RealFunctional, bestLR)\ge0.50,
$$

$$
ShuffleControlPass=1,
$$

and no degradation in task safety.

### 可视化

```text
p14_paired_replay_value_curves.svg
p14_real_vs_controls_heatmap.svg
p14_shuffle_control_distribution.svg
```

---

## P15：short/full training and Base-Acc Sentinel continuation

### 目标

只有 P14 pass 后打开 functional short/full training。Base-Acc Sentinel 可并行继续，但不得用于 controller。

### Base-Acc Sentinel

继续记录：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0..9
models = LQ-t2-h256, MatchedMLP, AdamWStrongLRGridMLP, QuadraticFeatureMLP
train_acc
val_acc
test_acc
ECE
NLL
CEp99
margin_p10
step_time
memory
base_acc_used_for_controller = 0
```

### Functional short/full training

只有 P14 pass 后记录：

```text
final_test_acc
best_val_acc
time_to_target
steps_to_target
val_loss_auc_step
val_loss_auc_time
ECE
NLL
CEp99
margin_p10
robustness_score
sample_efficiency
```

### 判断标准

Short-run pass：

```text
RealFunctional does not underperform AdamWStrongLRGridMLP by more than 0.005 on mean test acc；
functional has at least one of: better calibration, CEp99, margin, sample efficiency。
```

Full-run pass：

```text
matched controls beaten on at least one preregistered non-accuracy axis and no catastrophic acc drop。
```

### 可视化

```text
p15_base_acc_sentinel_trend.svg
p15_short_run_acc_loss_curves.svg
p15_sample_efficiency.svg
p15_calibration_robustness.svg
```

---

## P16：continual / anti-forgetting boundary

### 目标

只有 P14/P15 pass 后打开。验证 functional update 是否带来持续学习或抗遗忘优势。

### 设置

```text
task sequence: MNIST -> Fashion-MNIST -> KMNIST and reversed orders
evaluation after each task on all previous tasks
same model size and training budget
no dataset-specific controller threshold
```

### 必须记录

```text
sequence_id
step
task_id
current_task_acc
previous_task_acc
retained_accuracy
forgetting
backward_transfer
forward_transfer
old_task_CEp99
old_task_margin_p10
controller_accept_rate
functional_update_count
```

### 判断标准

Continual weak pass：

```text
mean forgetting <= matched MLP forgetting - 0.01
or retained accuracy >= matched MLP retained accuracy + 0.01
without current task catastrophic drop。
```

### 可视化

```text
p16_forgetting_matrix.svg
p16_retained_accuracy_curves.svg
p16_forward_backward_transfer.svg
```

---

# 8. 并行执行策略

v9.5.2 必须减少“一轮一个 blocker”的串行浪费。执行分成 5 个并行批次。

## Batch A：Target resolution

```text
P0 boundary reproduction
P1 target lattice
P2 target leaveout sanity
```

预计产物：

```text
selected_target_id or target_absent_route
p1_target_lattice_coverage_vs_Vint.svg
p2_leave_dataset_target_matrix.svg
```

Stop condition：

```text
若 P1 weak target pass = 0，则停止 P3-P11 official path，只保留 diagnostic generator density exploration。
```

## Batch B：Legal upper-bound and mechanism

```text
P3 legal upper-bound probe
P4 mechanism anatomy
```

与 Batch A 同时准备，但只有 selected target 后 official 评分。

Stop condition：

```text
若 P3 fail，则不运行 feature distillation / legal selector controller。
```

## Batch C：Existing APX re-score and APY preflight

```text
P5 APX re-score
P6 APY implementation/preflight
```

APY implementation 可先用 P1 provisional target top candidates 作为 generator calibration，不用 heldout outcome at commit。

## Batch D：APY outcome and certificate

```text
P7 APY branch-horizon outcome
P8 damage audit
P9 vector certificate
```

Stop condition：

```text
若 P7 fail，P9 只做 diagnostic，不跑 P10 controller。
若 P9 fail，P10/P11 不打开。
```

## Batch E：Health and runtime boundary

```text
Base-Acc Sentinel continuation
runtime preflight only
no selected runtime official until P10 pass
```

---

# 9. Route decision table

| Route | 条件 | 下一步 |
|---|---|---|
| `R1-TargetAbsent` | P1 没有 weak target pass | 重建 action density / candidate source，不再跑 selector/controller |
| `R2-TargetExistsLegalInvisible` | P1 pass, P3 fail | 停止 selection route，转 constructive primitive |
| `R3-TargetExistsMechanismFound` | P1 pass, P4 pass | 将 mechanism 转成 APY constraints / certificate features |
| `R4-APXTargetMismatchOnly` | APX re-score under T_E pass | 可以考虑 APX controller，但必须 P9/P10/P11 过 |
| `R5-APXUniversalFail` | APX re-score fail | APX family 停止作为主线 |
| `R6-APYImplementationFail` | APY payload/preflight 不闭合 | 修 implementation，不讨论 science |
| `R7-APYValueFail` | APY outcome 完整但 value/risk 不过 | primitive objective 失败，重设数学形式 |
| `R8-CertificateFail` | APY value pass but P9 fail | 重做 vector certificate |
| `R9-ControllerFail` | P9 pass but P10 fail | controller/support/crossfit 问题 |
| `R10-RuntimeFail` | P10 pass but P11 fail | runtime/payload apply 问题 |
| `R11-SystemPassPairedReplayOpen` | P12 pass | 打开 P13/P14/P15 |

---

# 10. 预期结论模板

v9.5.2 结束时必须能回答：

```text
1. 是否存在 coverage >=0.03 的 official multi-horizon target？
2. 如果存在，它与 T_A/T_B/T_C/T_D 的关系是什么？
3. 如果不存在，是 action density 不足还是 target 过严？
4. legal upper-bound 是否能看见 corrected target？
5. APX 是 target mismatch 还是 primitive universal failure？
6. APY objective-solved primitive 是否产生 value-positive/horizon-safe frontier？
7. certificate 是否能预测 vector effect？
8. 是否有 controller 被选中？
9. selected runtime 是否过线？
10. Base LQ 是否保持健康？
```

最终不能写模糊结论。必须落入 route decision table 中一个 route。

---

# 11. 最小 artifact 清单

```text
route_decision_v9520.json
p0_v9510_boundary_reproduction.csv
p1_target_lattice_resolution.csv
p1_target_lattice_trace_v9520.csv
p2_target_leaveout_sanity.csv
p3_legal_upper_bound_probe_v3.csv
p4_mechanism_anatomy_v3.csv
p5_apx_rescore_under_target.csv
p6_apy_primitive_spec_preflight.csv
p7_apy_branch_horizon_smoke_outcome.csv
p8_apy_damage_audit.csv
p9_vector_certificate_v5.csv
p10_minimal_source_certificate_controller.csv
p11_selected_controller_runtime.csv
p12_system_integration_gate.csv
p13_leaveout_boundary.csv
p14_official_paired_replay_boundary.csv
p15_short_full_base_acc_boundary.csv
p16_continual_boundary.csv
no_fake_audit_v9520.csv
contract_audit_v9520.csv
failure_table_v9520.csv
run_manifest.json
artifact_hashes_v9520.csv
```

---

# 12. 最终判断

v9.5.2 的核心不是继续证明“APX 可以生成 payload”，这已经在 v9.5.1 做到了。v9.5.2 要回答的是：

$$
\boxed{
\text{在 canonical truth 下，是否存在一个支持度足够、multi-horizon 安全、可 legal 生成或可证书化的 functional update target？}
}
$$

如果答案是否定的，项目下一步必须回到更底层的 functional action primitive 数学设计，而不是继续 patch controller。

如果答案是肯定的，才有资格进入：

```text
minimal controller -> selected runtime -> LDO/LSO -> paired replay -> short/full training。
```

