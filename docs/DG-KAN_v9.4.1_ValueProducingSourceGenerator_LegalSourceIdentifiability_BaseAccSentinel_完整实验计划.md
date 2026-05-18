# DG-KAN v9.4.1 Value-Producing Source Generator / Legal Source Identifiability / Base-Acc Sentinel 完整实验计划

> 本计划基于 v9.4.0 `Source Action Selection / Candidate Source Rebuild / Horizon-Robust Controller` 的真实执行结果制定。  
> v9.4.1 不继续做 AP0 静态特征阈值搜索，也不把 oracle source selector 写成 official controller。  
> 本轮目标是把问题从“full AP0 universe 里有 oracle-good source，但 legal selector 看不见”推进到“能否生成 commit-time 自带证书的 value-producing source action”。

---

# 0. 执行摘要

v9.4.0 的核心结果是：

```text
route = R3-LegalSourceSelectorOpaque
base_candidate = LQ-t2-h256
success_v9400_strict_purekan_functional = False
success_v9400_full_functional = False
success_v9400_external_ready = False
```

v9.4.0 不是失败倒退。它纠正了 v9.3.9 的一个关键误读：v9.3.9 的 64-action source panel 很差，但这个 panel 不是 full AP0 universe 的代表样本，而是 `first_N_payload_available` convenience slice。v9.4.0 显示：

```text
full_action_count = 2876
source_panel_action_count = 64
source_panel_fraction = 0.022253129346314324
full_weak_CP_row_rate = 0.1035002318034307
source_panel_weak_CP_row_rate = 0.0625
selection_bias_PSI = 25.80667683866949
selection_bias_KL = 2.434227117471533
route_flip_full_vs_source_panel = 1
```

这说明 v9.3.9 的 source panel bad 不能推出 AP0 全体 source bad。

同时，v9.4.0 的 oracle source selector 证明 full AP0 universe 中存在较强 upper-bound source frontier：

```text
best_oracle_selector_id = OS2-weak-CP-oracle-source-selector
K = 64
weak_CP_precision_all_rows = 0.734375
strong_CP_precision_all_rows = 0.3854166666666667
V_ctrl_lcb_all = 0.11206787715966769
long_risk_rate = 0.109375
support_balance_pass = 1
accepted_family_count = 34
max_family_share = 0.09375
weak_CP_h20 / h80 / h240 = 0.703125 / 0.828125 / 0.671875
oracle_source_present = 1
```

但这仍不是 official，因为 oracle selector 使用 outcome labels。真正失败的是 legal commit-time source selector：

```text
best_legal_selector_id = LSS1-single-feature-topK-LSB-PayloadLinf
feature_auc_weak_CP = 0.5861789765860932
weak_CP_precision_h20 = 0.140625
weak_CP_precision_all_rows = 0.171875
strong_CP_precision_all_rows = 0.0625
V_ctrl_lcb_all = -0.24349444404221637
long_risk_rate = 0.2760416666666667
legal_source_selector_pass = 0
```

因此，本轮的本质结论是：

$$
\boxed{
\text{AP0 full universe has oracle-good sources, but AP0 source value is legally opaque.}
}
$$

v9.4.1 的主线不能是继续筛 `PayloadLinf / StateNLL / score bucket / support count`。v9.4.1 必须实现真实 AP0b-AP0f value-producing source generators，使 source action 在生成时自带 commit-time certificate，而不是事后从 opaque AP0 action 中猜哪一个好。

---

# 1. 当前是否有数据集训练、acc 与 MLP 对比？

## 1.1 v9.4.0 没有做新的 short/full dataset training

v9.4.0 runner 做的是：

```text
读取 v9.3.5 full AP0 outcome universe；
读取 v9.3.9 source panel；
读取 v9.3.3 durable payload；
执行 full-vs-panel diagnosis；
执行 source provenance；
执行 oracle source selector upper-bound；
执行 legal source selector capacity；
然后按 gate 阻断未 materialize 的 source generator / controller / runtime。
```

它不是 full-run 训练实验，不是 MNIST / Fashion-MNIST / KMNIST 打榜实验，也没有生成 `final_acc / best_val_acc / test_acc` 这样的 full-run 指标。

v9.4.0 中 P5-P13 全部 gate-blocked：

```text
P5 h20 immediate smoke = not_run
P6 horizon extension = not_run
P7 source-to-generated transform damage = not_run
P8 certificate redesign = not_run
P9 minimal source certificate controller = not_run
P10 selected online runtime = not_run
P11 system integration = not_run
P12 leaveout / paired replay = not_run
P13 short/full/sample-efficiency/continual robustness = not_run
```

所以本轮不能回答：

```text
DG-KAN full-run acc 是多少？
functional KAN test acc 是否超过 MLP？
KAN 在 MNIST/Fashion/KMNIST 上是否 beating MLP？
```

因为这些实验没有打开。

## 1.2 v9.4.0 使用了数据集，但不是用于调榜

v9.4.0 使用的是已有 train-stream / outcome universe，它依赖 dataset/seed/step/family/horizon 等字段做诊断。但本轮没有按 dataset 调参，也没有 dataset-specific selector/controller：

```text
uses_dataset_name_for_selector/controller = 0/0
uses_validation_or_test = 0
uses_future_outcome_for_features = 0
uses_outcome_at_commit = 0
```

这符合项目原则：可以诊断不同数据集 / family / step / horizon 的失败模式，但不能为了这些数据集调 selector 或 threshold。

## 1.3 为什么现在不能直接和 MLP 比 acc

当前还没有 system-legal controller。没有 controller，就没有 selected functional update path；没有 selected functional update path，就不能打开：

```text
leave-dataset-out
leave-stratum-out
official paired replay
short-run
full-run
sample-efficiency
continual / anti-forgetting
MLP / AdamW / bestLR / QuadraticFeatureMLP comparison
```

因此，v9.4.0 阶段正确的比较对象不是 final acc，而是：

```text
source action 的 control-positive precision；
source action 的 V_ctrl LCB；
source action 的 long-risk；
source selector 的 legal observability；
source panel representativeness；
source generator 是否 materialized；
controller 是否 official eligible。
```

v9.4.1 会新增一个 **Base-Acc Sentinel**，并行跑固定配置的 LQ-t2-h256 与 matched MLP，记录 train/val/test acc，但它只作为健康监控，不用于调 selector/controller。

---

# 2. 对 v9.4.0 的独立判断

## 2.1 真进展：v9.3.9 的 bad source panel 被解释清楚了

v9.3.9 停在：

```text
source_AP0_same_panel_bad
```

当时同源 AP0 source panel 的结果是：

```text
source_AP0_weak_CP_precision = 0.07291666666666667
source_AP0_long_risk_rate = 0.3072916666666667
source_AP0_V_ctrl_lcb = -1.4980123384538895
```

v9.4.0 进一步发现，这个 panel 不是 value selector，而是：

```text
selection_rule = v9380_first_N_payload_available_slice
reason = source_panel_is_first_N_payload_available_convenience_slice_not_value_selector
uses_outcome_in_source_selection = 0
uses_dataset_name_in_source_selection = 0
uses_future_step_in_source_selection = 0
```

它不违规，但它也没有任何理由能选到 good source。source panel 的 representativeness 失败非常强：

```text
selection_bias_PSI = 25.80667683866949
selection_bias_KL = 2.434227117471533
max_family_gap = 0.38030250347705147
max_step_bucket_gap = 0.5111700278164117
max_score_bucket_gap = 0.26881954102920724
route_flip_full_vs_source_panel = 1
```

这说明：v9.3.9 的失败不能扩大解释成 AP0 full universe 没有好 source。

## 2.2 真进展：AP0 source frontier 不是 absent

v9.4.0 的 oracle source selector 给出了重要 upper-bound：

```text
K = 64
weak_CP_precision_all_rows = 0.734375
strong_CP_precision_all_rows = 0.3854166666666667
V_ctrl_lcb_all = 0.11206787715966769
long_risk_rate = 0.109375
support_balance_pass = 1
accepted_family_count = 34
max_family_share = 0.09375
```

这说明 AP0 full universe 中确实有相当好的 source actions。虽然 long-risk `0.109375` 略高于 strict `0.10`，但它不是 value-poor source panel 那种失败。

独立地看，这里有两个含义：

1. 不应该判定 `AP0CandidateSourceOracleAbsent`。
2. 不应该继续使用 first-64 convenience source panel 去生成 AP1/AP8 variants。

## 2.3 真失败：legal selector 与 oracle selector 的差距太大

oracle K64 的 weak CP precision 是：

$$
P_{oracle}=0.734375.
$$

best legal K64 的 all-row weak CP precision 是：

$$
P_{legal}=0.171875.
$$

差距为：

$$
\Delta P=0.734375-0.171875=0.5625.
$$

如果按 64 个 source actions 粗略计算，oracle 约能选中：

$$
64\times0.734375=47
$$

个 weak-CP source，而 best legal selector 约只能选中：

$$
64\times0.171875=11
$$

个。差了约 36 个 good source。

同时，best legal selector 的风险也明显更高：

```text
oracle long-risk = 0.109375
legal long-risk = 0.2760416666666667
```

这不是“换个阈值”能解决的轻微 gap。当前 legal features 基本看不见 source value。

## 2.4 不能把 v9.4.0 读成 source generator 失败

v9.4.0 中：

```text
source_generator_materialized = 0
AP0b-AP0f real source generators = not materialized
```

所以本轮没有证明 AP0b/AP0c/AP0d/AP0e/AP0f 不行。它只证明：repo 当前还没有真实 value-producing source generator，导致 h20 smoke、horizon extension、certificate redesign、controller、runtime、paired replay、short/full 全部不能打开。

这点很重要：下一步不是改 legal selector 小特征，而是实现真实 source generator。

---

# 3. 当前项目进度

## 3.1 已经完成的关键资产

```text
1. LQ-t2-h256 base anchor；
2. manual forward / backward / AdamW update contract；
3. primary same-run labels；
4. action apply replay；
5. durable AP0 payload package；
6. full matched-control outcome universe；
7. AP0 weak control-positive oracle existence；
8. AP0 legal observability failure diagnosis；
9. AP1-AP8 generation/outcome smoke chains；
10. source panel bias diagnosis；
11. AP0 source oracle upper-bound；
12. no-fake / no-proxy / no-dataset-tuning audit。
```

这些都是扎实进展。

## 3.2 当前没完成的硬门

```text
1. legal source selector pass；
2. real AP0b-AP0f value-producing source generator；
3. immediate h20 source direction pass；
4. h80/h240 horizon extension pass；
5. effect-valid source certificate；
6. minimal source certificate controller；
7. selected-controller online runtime；
8. system legal controller；
9. LDO / LSO；
10. official paired replay；
11. short/full training；
12. MLP / strong baseline comparison；
13. sample efficiency / continual / anti-forgetting。
```

## 3.3 离目标还有多远

离 **system-legal local functional controller** 还至少差三步：

```text
1. 真实生成 value-producing source actions；
2. 证明这些 source actions 在 h20/h80/h240 有正 V_ctrl 且 long-risk 可控；
3. 用 commit-time certificate 选择它们，并且 selected runtime 过 step ratio。
```

离 **strict PureKAN functional causal evidence** 还需要：

```text
LDO / LSO；
official paired replay；
RealFunctional beats AdamWParallel / bestLR；
shuffle controls fail。
```

离 **external-ready full functional success** 还需要：

```text
short/full training；
MLP / StrongLRGrid / QuadraticFeatureMLP controls；
robustness；
sample efficiency；
continual / anti-forgetting；
external reproducibility。
```

所以现在不能说“快成功了”。更准确地说：

$$
\boxed{
\text{项目已经从 StableAccept repair 进入 source-action generation 的核心阶段，但还没有进入 full training / MLP comparison 阶段。}
}
$$

---

# 4. v9.4.1 总体目标

v9.4.1 的总体目标是：

$$
\boxed{
\text{实现并验证 real value-producing source generators，使 source action 在生成时自带可审计 certificate。}
}
$$

v9.4.1 的核心问题不是：

```text
PayloadLinf 阈值再怎么调？
StateNLL 是否再加一个 bucket？
source panel 能不能换成另一个 first-N slice？
AP7 epsilon 是否再小一点？
```

而是：

```text
1. 能不能在 commit time 生成 immediate-positive source action？
2. 能不能让 source action 的 h20 positive direction 不依赖 oracle labels？
3. 能不能让 source action 在 h80/h240 不产生 long-risk？
4. 能不能用 certificate 而不是 posthoc outcome 选择 source？
5. 能不能在不调数据集的前提下打开 system controller？
```

v9.4.1 的最低有效推进目标：

```text
1. AP0b-AP0f 至少 3 个真实 source generator materialized；
2. 每个 source generator 生成 durable payload + commit-time certificate；
3. action apply error 全量测量；
4. h20 immediate smoke outcome 完成；
5. 至少一个 source generator h20 weak_CP_precision >= 0.35 且 V_ctrl_lcb_h20 > 0；
6. 若 h20 pass，则完成 h80/h240 horizon extension；
7. 至少一个 certificate feature 对 generated-source weak CP AUC >= 0.65，或明确判定 certificate insufficient；
8. Base-Acc Sentinel 记录 LQ-t2-h256 vs matched MLP 的 train/val/test acc，但不用于调参。
```

v9.4.1 强目标：

```text
source_generator_materialized = 1
h20_immediate_direction_pass = 1
horizon_extension_pass = 1
certificate_sufficiency_pass = 1
source_certificate_controller_pass = 1
selected_runtime_pass = 1
system_legal_controller_pass = 1
```

---

# 5. 硬约束

继续遵守：

```text
no teacher
no self-teacher
no distillation
no auxiliary loss
no loss modification
no label smoothing
no focal / margin / calibration loss
no sampler / class weight
no CPU offload
no fake / proxy rows
KAN path 不使用 PyTorch loss.backward graph
official selector/controller 不使用 dataset_name 分支
不使用 validation/test metric at commit time
不使用 future outcome at commit time
不把 oracle selector 写成 official
不把 diagnostic AP0 old outcomes 写成 new source generator outcomes
不把 Base-Acc Sentinel 用于 selector/controller selection
```

functional update 仍然是 update rule：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{AdamW}
+
\Delta\theta_{functional}.
$$

任务 loss 仍是标准 CE：

$$
L_{task}=CE(y,p_\theta(x)).
$$

允许做：

```text
1. dataset-level diagnostic；
2. source family / horizon / step bucket diagnostic；
3. oracle selector upper-bound diagnostic；
4. calibration split frozen thresholds；
5. cross-fitted source certificate controller；
6. fixed-config LQ-vs-MLP Base-Acc Sentinel；
7. parallel source generator smoke；
8. offline materializer batching；
9. selected-controller online runtime measurement。
```

---

# 6. 核心假设

## H0：v9.3.9 bad source panel 是 sampling/provenance problem，不是 AP0 full universe absence

H0 已由 v9.4.0 初步支持，但 v9.4.1 需要防止再采样错误。

成立标准：

```text
source panels generated by stratified source sampler have PSI <= 0.20 vs full AP0 universe；
route_flip_full_vs_panel = 0；
family/step/score/payload bucket gaps <= 0.10；
source panel weak_CP rate confidence interval contains full universe weak_CP row rate。
```

失败标准：

```text
new source panels 仍出现 PSI > 1.0 或 route flip；
后续 generator 使用了 biased panel。
```

## H1：static legal selector cannot identify AP0 oracle source frontier

v9.4.0 best legal selector AUC 只有 `0.5861789765860932`，all-row weak CP precision 只有 `0.171875`。H1 认为继续筛静态 AP0 selector 不会带来 official source selection。

成立标准：

```text
在 source-balanced panels 与 full AP0 universe 上，static legal selectors 均满足：
AUC_weak_CP < 0.65；
K64 weak_CP_precision_all < 0.35；
V_ctrl_lcb_all <= 0；
long_risk_rate >= 0.15。
```

失败标准：

```text
一个 pre-registered legal selector 在 crossfit / leave-family / leave-step 上达到：
K64 weak_CP_precision_all >= 0.50；
V_ctrl_lcb_all > 0；
long_risk_rate <= 0.15。
```

若 H1 失败，可以选择 legal selector 进入 controller；否则必须走 source generator。

## H2：value-producing source generators can create immediate-positive h20 actions

H2 认为，AP0b-AP0f 这类 generator 如果直接使用 manual gradient / tail margin / AdamW residual / low-rank edge basis 生成 source，而不是从 opaque AP0 payload 中挑选，应该能得到更高 h20 positive rate。

成立标准：

```text
至少一个 AP0b-AP0f generator：
  generated_action_count >= 256；
  action_apply_error_linf_max <= 1e-6；
  weak_CP_precision_h20 >= 0.35；
  V_ctrl_lcb_h20 > 0；
  bad_event_rate_h20 <= 0.10；
  null_rate_h20 <= 0.30。
```

强成立标准：

```text
weak_CP_precision_h20 >= 0.50；
V_ctrl_lcb_h20 > 0.05；
real_beats_adamwparallel_h20 >= 0.60；
real_beats_bestlr_h20 >= 0.50。
```

失败标准：

```text
所有 AP0b-AP0f h20 weak_CP_precision < 0.20 或 V_ctrl_lcb_h20 <= 0。
```

## H3：h20 positive direction is necessary but not sufficient; long horizon must be guarded

v9.3.6-v9.3.9 显示 horizon-robust source/action 很稀疏，long-risk 可以主导 failure。H3 认为 h20 pass 后必须立刻测 h80/h240，不能只看 immediate direction。

成立标准：

```text
对 h20 survivors：
  weak_CP_precision_all >= 0.40；
  V_ctrl_lcb_all > 0；
  long_risk_rate <= 0.15；
  weak_CP_h240 >= 0.25。
```

强成立标准：

```text
weak_CP_precision_all >= 0.55；
V_ctrl_lcb_all > 0.05；
long_risk_rate <= 0.10；
horizon_robust_action_rate >= 0.03。
```

失败标准：

```text
h20 pass 但 h240 long_risk_rate > 0.30，且 certificate 无法过滤。
```

## H4：effect-valid certificate can select generated sources better than static AP0 features

H4 认为 source generator 生成时可以产生更有解释力的 certificate：descent LCB、tail risk UCB、horizon risk UCB、support LCB、cost。它们应该比 `PayloadLinf / StateNLL` 更接近 outcome。

成立标准：

```text
certificate AUC_weak_CP >= 0.65；
certificate AUC_long_risk >= 0.65；
cert-pass weak_CP_precision_all >= cert-fail weak_CP_precision_all * 1.8；
cert-pass long_risk_rate <= cert-fail long_risk_rate * 0.70；
monotone_sign_pass = 1。
```

强成立标准：

```text
certificate AUC_weak_CP >= 0.75；
certificate AUC_long_risk >= 0.75；
cert-pass weak_CP_precision_all >= 0.50；
cert-pass long_risk_rate <= 0.10；
V_ctrl_lcb_cert_pass > 0。
```

失败标准：

```text
certificate AUC <= 0.55 或 cert-pass/failed 没有 lift。
```

## H5：controller 必须 crossfit，不能在单 split / 单 panel 上 promotion

成立标准：

```text
seed-fold crossfit pass；
leave-family-out pass；
leave-step-bucket-out pass；
leave-dataset-out diagnostic pass；
coverage not zero in any official fold；
no dataset_name branch。
```

失败标准：

```text
calibration pass but heldout coverage collapses；
leave-family/leave-step 崩；
threshold 依赖 dataset。
```

## H6：Base-Acc Sentinel 只监控，不参与 selection

成立标准：

```text
LQ-vs-MLP fixed-config training complete；
records train_acc / val_acc / test_acc / loss / ECE / runtime；
Base-Acc results not used in source selector / controller thresholds；
route_decision records base_acc_used_for_controller = 0。
```

失败标准：

```text
Base-Acc Sentinel 被用于选择 AP0b-AP0f、threshold 或 dataset-specific route。
```

---

# 7. v9.4.1 action/source generator designs

## AP0b：LastEdgeLinearizedDescentSource

目标：生成一个明确降低 batch CE 的 last-edge / last-layer source update。

生成形式：

$$
\Delta\theta_{AP0b}
=
-\eta_s P_{last}\left(g_t\right)
\cdot M_{tail},
$$

其中 $P_{last}$ 是 last-edge/last-layer projection，$M_{tail}$ 是 hard-tail mask，不使用 outcome labels。

commit-time certificate：

$$
DescentCert(e)=-g_t^T\Delta\theta_{AP0b},
$$

$$
NormCert(e)=\frac{\|\Delta\theta_{AP0b}\|}{\|\Delta\theta_{AdamW}\|+\epsilon},
$$

$$
TailSafeCert(e)=\widehat{\Delta Margin}_{p10}(e)-\lambda\widehat{\Delta CE}_{p99}(e).
$$

必须记录：

```text
linearized_CE_delta
linearized_margin_p10_delta
grad_dot_delta
cos_delta_negative_grad
cos_delta_adamw
norm_ratio
hard_tail_fraction
certificate_hash
payload_hash
```

## AP0c：AdamWResidualOrthogonalSource

目标：生成不重复 AdamW、但能补充 AdamW 盲区的 residual source。

$$
\Delta\theta_{AP0c}
=
P_{\perp \Delta\theta_{AdamW}}(-g_{tail}).
$$

其中：

$$
P_{\perp a}(b)=b-\frac{a^Tb}{\|a\|^2+\epsilon}a.
$$

certificate：

```text
cos_delta_adamw near 0
cos_delta_negative_grad > 0
residual_tail_descent_lcb > 0
conflict_rate <= threshold
```

## AP0d：TailMarginRepairSource

目标：直接修复 CEp99 / margin p10 tail，而不是优化均值。

source objective：

$$
\max_{\Delta\theta}
\widehat{\Delta Margin}_{p10}(\Delta\theta)
-ho\widehat{\Delta CE}_{p99}(\Delta\theta)
-\lambda\|\Delta\theta\|^2.
$$

certificate：

```text
margin_p10_lcb
CEp99_risk_ucb
tail_selectivity
wrong_confidence_reduction_estimate
```

## AP0e：CurvatureGuardedLowRankEdgeSource

目标：限制 curvature / Lipschitz drift，避免 long-risk。

约束：

$$
\Delta\theta^T\widehat{H}_{diag}\Delta\theta\le \tau_H,
$$

$$
\|\Delta\theta\|_\infty\le \tau_\infty.
$$

certificate：

```text
curvature_proxy_ucb
linf_norm
low_rank_energy_ratio
basis_entropy_delta_estimate
horizon_guard_score
```

## AP0f：SupportMemorySource

目标：只在历史 calibration support 充足的 family/horizon/neighborhood 中生成 source。

Empirical-Bayes support：

$$
SupportLCB(e)=\hat p(e)-z_\alpha\sqrt{\frac{\hat p(e)(1-\hat p(e))}{n_{eff}(e)+\epsilon}}.
$$

certificate：

```text
support_lcb
bad_ucb_EB
null_ucb_EB
family_support_count
horizon_support_count
neighborhood_shift_risk
```

## AP0g：Controls

必须同时生成 controls：

```text
NoOpSource
RandomNormMatchedSource
ScaledAdamWSource
ShuffledPayloadSource
ShuffledCertificateSource
OracleSourceDiagnosticOnly
```

OracleSourceDiagnosticOnly 只能用于 upper-bound，不得 official。

---

# 8. 数据合同

## 8.1 source generator row

```text
source_action_id
candidate_id
event_id
primitive_id
source_generator_id
source_generator_version
dataset
seed
step
family_id
bucket_id
horizon
payload_hash
certificate_hash
payload_tensor_written
certificate_tensor_written
commit_time_available
uses_dataset_name
uses_outcome_at_commit
uses_future_step
uses_validation_or_test
```

Pass：

```text
generated_action_count > 0
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
commit_time_available = 1
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_step = 0
```

## 8.2 action apply contract

```text
action_apply_error_linf
action_apply_error_relative
action_apply_cosine_logged_applied
replay_success
payload_shape
payload_dtype
payload_device
payload_load_time_ms
payload_apply_time_ms
```

Pass：

```text
action_apply_error_linf_max <= 1e-6
action_apply_error_relative_max <= 1e-4
action_apply_cosine_min >= 0.999999
replay_success_rate = 1.0
```

## 8.3 source outcome contract

Branches：

```text
RealSource
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ScaledAdamW
ShuffledSourcePayload
ShuffledCertificate
CertificatePassNoPayload
```

Horizons：

```text
h20
h80
h240
optional h640
```

Metrics：

```text
weak_CP_label
strong_CP_label
horizon_robust_CP_label
long_risk_label
safe_good_label
bad_event_label
null_event_label
V_ctrl
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
real_beats_adamwparallel
real_beats_bestlr
real_beats_noop
real_beats_random
```

## 8.4 Base-Acc Sentinel contract

```text
model_id
model_family
KAN_or_MLP
dataset
seed
steps_or_epochs
train_acc
val_acc
test_acc
train_loss
val_loss
test_loss
ECE
NLL
CEp99
margin_p10
step_time_q90
memory_ratio
params
forward_flops
backward_flops
base_acc_used_for_controller
```

Pass for sentinel integrity：

```text
base_acc_used_for_controller = 0
hyperparams_fixed_before_run = 1
dataset_specific_tuning = 0
same_seed_schedule = 1
same_budget = 1
```

---

# 9. 实验阶段

---

## P0：v9.4.0 boundary reproduction and source-panel bias lock

### 目标

复现 v9.4.0 boundary，锁定 current blocker，并防止 v9.4.1 再使用 biased source panel。

### 假设

H0：v9.3.9 的 bad source panel 是 convenience sampling artifact，不代表 full AP0 universe。

### 必须记录

```text
route_v9400
full_action_count
source_panel_action_count
source_panel_fraction
source_panel_join_pass
full_weak_CP_row_rate
source_panel_weak_CP_row_rate
full_strong_CP_row_rate
source_panel_strong_CP_row_rate
full_long_risk_rate
source_panel_long_risk_rate
full_V_ctrl_lcb
source_panel_V_ctrl_lcb
selection_bias_PSI
selection_bias_KL
max_family_gap
max_step_bucket_gap
max_score_bucket_gap
route_flip_full_vs_source_panel
source_selection_rule
source_selection_reason
uses_outcome_in_source_selection
uses_dataset_name_in_source_selection
uses_future_step_in_source_selection
```

### 判断标准

P0 pass：

```text
v9.4.0 boundary reproduced；
source panel bias quantified；
biased panel is banned from official generator scale-up；
source_panel_used_for_official = 0。
```

### 可视化

```text
p0_full_vs_source_panel_rate_bar.svg
p0_source_panel_bias_psi_kl.svg
p0_family_step_score_gap_heatmap.svg
p0_panel_route_flip_dashboard.svg
```

---

## P1：stratified source panel builder

### 目标

建立 official source-generator smoke panel，避免 first-N payload convenience slice。

### 假设

如果使用 stratified panel，source panel 的 family/step/score/payload 分布应接近 full AP0 universe。

### 实现

构建三种 panel：

```text
PANEL-S64: 64 actions, stratified by family × horizon × score bucket × step bucket
PANEL-S256: 256 actions, stratified by family × horizon × score bucket × step bucket
PANEL-S864: 864 actions, used for scale-up and certificate calibration
```

不得使用 outcome labels 进行 official panel sampling。允许使用 outcome labels 做 representativeness audit。

### 必须记录

```text
panel_id
panel_action_count
panel_row_count
sampling_rule
sampling_seed
family_distribution
horizon_distribution
step_bucket_distribution
score_bucket_distribution
payload_norm_bucket_distribution
PSI_vs_full
KL_vs_full
max_family_gap
max_step_bucket_gap
max_score_bucket_gap
max_payload_bucket_gap
oracle_rate_diagnostic_only
source_panel_used_for_official
```

### 判断标准

P1 pass：

```text
PSI_vs_full <= 0.20
KL_vs_full <= 0.10
max_family_gap <= 0.10
max_step_bucket_gap <= 0.10
max_score_bucket_gap <= 0.10
source_panel_used_for_official = 1
```

If P1 fails：

```text
不能继续 AP0b-AP0f official generator smoke；必须先修 sampler。
```

### 可视化

```text
p1_panel_distribution_vs_full.svg
p1_panel_balance_table.svg
p1_panel_sampling_sankey.svg
p1_panel_bias_metrics.svg
```

---

## P2：Base-Acc Sentinel，LQ-t2-h256 vs matched MLP

### 目标

回答“当前有没有在数据集上训练，acc 和 MLP 如何”的监控需求，但不让 acc 结果影响 source/controller 选择。

### 假设

Base LQ-t2-h256 应保持合理任务能力；若 base 本身显著劣化，functional source/controller 结果会难解释。

### 设置

```text
models:
  LQ-t2-h256 manual PureKAN base
  matched-parameter MLP
  QuadraticFeatureMLP diagnostic baseline

datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2 initially
  0..9 after source/controller pass

training:
  fixed training budget
  fixed LR grid chosen before run or single fixed LR
  no dataset-specific hyperparameter
  no functional update in sentinel primary run
```

### 必须记录

```text
model_id
model_family
dataset
seed
params
forward_flops
backward_flops
train_acc
val_acc
test_acc
train_loss
val_loss
test_loss
ECE
NLL
CEp99
margin_p10
step_time_q50
step_time_q90
memory_peak
memory_ratio
LQ_minus_MLP_test_acc
LQ_minus_MLP_ECE
LQ_minus_MLP_NLL
base_acc_used_for_controller
```

### 判断标准

P2 sentinel complete：

```text
all dataset × seed × model rows complete；
base_acc_used_for_controller = 0；
dataset_specific_tuning = 0；
no validation/test used by source/controller。
```

Health diagnostic：

```text
LQ catastrophic fail if mean test_acc_LQ < mean test_acc_MLP - 0.05。
```

This is not a stop condition by itself. It triggers base health autopsy, not source/controller threshold tuning.

### 可视化

```text
p2_base_acc_lq_vs_mlp.svg
p2_val_loss_curves.svg
p2_ece_nll_comparison.svg
p2_step_time_memory_comparison.svg
p2_dataset_seed_variance.svg
```

---

## P3：real AP0b-AP0f source generator materialization

### 目标

实现真实 source generator，不再停在 `source_generator_materialized = 0`。

### 假设

H2：至少一种 generator 能生成 immediate-positive h20 source actions。

### 实现

在 `PANEL-S256` 上并行生成：

```text
AP0b-LastEdgeLinearizedDescentSource
AP0c-AdamWResidualOrthogonalSource
AP0d-TailMarginRepairSource
AP0e-CurvatureGuardedLowRankEdgeSource
AP0f-SupportMemorySource
AP0g-controls
```

每个 primitive 至少生成：

```text
64 actions for smoke；
256 actions for official-minimum source generation；
864 actions for scale-up if P4 passes。
```

### 必须记录

```text
source_generator_id
primitive_id
source_action_count
payload_tensor_written
certificate_tensor_written
payload_hash_missing_count
certificate_hash_missing_count
action_apply_error_measured
action_apply_error_linf_max
action_apply_error_relative_max
action_apply_cosine_min
commit_time_available
uses_dataset_name
uses_outcome_at_commit
uses_future_step
certificate_schema_version
certificate_fields_complete
```

### 判断标准

P3 materialization pass：

```text
source_generator_materialized = 1
primitive_materialized_count >= 3
generated_action_count_total >= 256
payload_tensor_written = 1
certificate_tensor_written = 1
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max <= 1e-6
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_step = 0
```

### 可视化

```text
p3_generator_action_count.svg
p3_payload_certificate_hash_audit.svg
p3_action_apply_error_hist.svg
p3_generator_norm_geometry.svg
p3_commit_time_legality_dashboard.svg
```

---

## P4：h20 immediate direction smoke

### 目标

验证 generated source 是否在 immediate horizon 有正方向。P4 是 v9.4.1 的第一道科学 gate。

### 假设

H2：至少一个 value-producing source generator h20 positive。

### Branches

```text
RealSource
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ScaledAdamW
ShuffledSourcePayload
ShuffledCertificate
CertificatePassNoPayload
```

### 必须记录

```text
source_generator_id
primitive_id
source_action_count
branch_horizon_row_count_expected
branch_horizon_row_count_actual
horizon = h20
weak_CP_precision_h20
strong_CP_precision_h20
V_ctrl_mean_h20
V_ctrl_lcb_h20
bad_event_rate_h20
null_rate_h20
long_risk_rate_h20
CEp99_delta_mean_h20
margin_p10_delta_mean_h20
ECE_delta_mean_h20
NLL_delta_mean_h20
curvature_delta_mean_h20
real_beats_adamwparallel_h20
real_beats_bestlr_h20
real_beats_noop_h20
real_beats_random_h20
quality_audit_pass
```

### 判断标准

P4 weak pass：

```text
weak_CP_precision_h20 >= 0.35
V_ctrl_lcb_h20 > 0
bad_event_rate_h20 <= 0.10
null_rate_h20 <= 0.30
real_beats_adamwparallel_h20 >= 0.50
```

P4 strong pass：

```text
weak_CP_precision_h20 >= 0.50
V_ctrl_lcb_h20 > 0.05
bad_event_rate_h20 <= 0.05
real_beats_adamwparallel_h20 >= 0.60
real_beats_bestlr_h20 >= 0.50
```

If P4 fails for all generators：

```text
route = R4-GeneratedSourceImmediateDirectionFail
next = redesign source objective, not certificate/controller threshold。
```

### 可视化

```text
p4_h20_weak_cp_by_primitive.svg
p4_h20_vctrl_lcb_by_primitive.svg
p4_h20_real_vs_controls.svg
p4_h20_bad_null_composition.svg
p4_h20_tail_metrics.svg
```

---

## P5：h80/h240 horizon extension and long-risk audit

### 目标

验证 h20 positive source 是否能延伸到 h80/h240，避免 short-only / long-risk failure。

### 假设

H3：h20 positive 必须受 horizon guard 约束，否则不能进入 controller。

### 必须记录

```text
source_generator_id
primitive_id
horizon
weak_CP_precision
strong_CP_precision
V_ctrl_mean
V_ctrl_lcb
long_risk_rate
horizon_robust_CP_action_count
horizon_robust_CP_action_rate
short_only_CP_action_count
long_risk_action_count
weak_CP_h20
weak_CP_h80
weak_CP_h240
V_ctrl_lcb_h20
V_ctrl_lcb_h80
V_ctrl_lcb_h240
long_risk_h240
```

### 判断标准

P5 weak pass：

```text
weak_CP_precision_all >= 0.40
V_ctrl_lcb_all > 0
long_risk_rate <= 0.15
weak_CP_h240 >= 0.25
```

P5 strong pass：

```text
weak_CP_precision_all >= 0.55
V_ctrl_lcb_all > 0.05
long_risk_rate <= 0.10
horizon_robust_action_rate >= 0.03
```

If P5 fails but P4 passed：

```text
route = R5-ImmediatePositiveButHorizonFragile
next = horizon-guarded source redesign。
```

### 可视化

```text
p5_horizon_value_curves.svg
p5_h20_h80_h240_cp_heatmap.svg
p5_long_risk_by_primitive.svg
p5_short_only_vs_horizon_robust.svg
p5_vctrl_lcb_by_horizon.svg
```

---

## P6：effect-valid certificate sufficiency audit

### 目标

判断 generator certificate 是否能作为 effect-valid sufficient statistic，而不是 construction-valid metadata。

### 假设

H4：source generator 的 commit-time certificate 能识别 generated source 的 value/risk。

### Certificate components

```text
DescentLCB
TailRiskUCB
LongRiskUCB
NullUCB
SupportLCB
CostEstimate
GradientConflict
CurvatureGuard
HorizonGuard
PayloadNormRatio
```

### 必须记录

```text
certificate_id
source_generator_id
primitive_id
cert_pass
cert_score
DescentLCB
TailRiskUCB
LongRiskUCB
NullUCB
SupportLCB
CostEstimate
AUC_weak_CP
AUC_strong_CP
AUC_long_risk
AUC_null
P_weak_CP_given_cert_pass
P_weak_CP_given_cert_fail
P_longrisk_given_cert_pass
P_longrisk_given_cert_fail
Lift_weak
Lift_longrisk
monotone_sign_pass
calibration_ECE
```

### 判断标准

P6 weak pass：

```text
AUC_weak_CP >= 0.65
AUC_long_risk >= 0.65
Lift_weak >= 1.8
P_longrisk_given_cert_pass <= 0.70 * P_longrisk_given_cert_fail
monotone_sign_pass = 1
```

P6 strong pass：

```text
AUC_weak_CP >= 0.75
AUC_long_risk >= 0.75
P_weak_CP_given_cert_pass >= 0.50
P_longrisk_given_cert_pass <= 0.10
V_ctrl_lcb_cert_pass > 0
```

If P6 fails：

```text
route = R6-CertificateNotEffectValid
next = redesign certificate, not threshold search。
```

### 可视化

```text
p6_certificate_auc_bar.svg
p6_cert_pass_fail_lift.svg
p6_certificate_calibration_curve.svg
p6_certificate_component_ablation.svg
p6_cert_score_vs_vctrl.svg
```

---

## P7：minimal source certificate controller

### 目标

用 commit-time certificate 选择 source actions，形成 legal source controller。P7 不允许 oracle labels at commit。

### Accept rule

$$
Accept(e)=1
\iff
LCB(V_{cert}(e))>0
\land
UCB(Bad(e))\le\tau_b
\land
UCB(Null(e))\le\tau_n
\land
UCB(LongRisk(e))\le\tau_h
\land
LCB(Support(e))\ge\tau_s
\land
C(e)\le C_{max}.
$$

Monotone score：

$$
S(e)=
 a_1 DescentLCB(e)
-a_2 TailRiskUCB(e)
-a_3 LongRiskUCB(e)
-a_4 NullUCB(e)
+a_5 SupportLCB(e)
-a_6 Cost(e),
$$

with:

```text
a_i >= 0
feature_group_count <= 6
thresholds frozen on calibration split
no dataset branch
```

### Splits

```text
seed-fold crossfit
leave-family-out
leave-step-bucket-out
leave-score-bucket-out
leave-dataset-out diagnostic
```

### 必须记录

```text
controller_id
source_generator_id
primitive_id
feature_set
thresholds
calibration_fold
heldout_fold
split_type
accepted_count_cal
accepted_count_heldout
coverage_heldout
weak_CP_precision_heldout
strong_CP_precision_heldout
V_ctrl_lcb_heldout
bad_event_rate_heldout
null_rate_heldout
long_risk_rate_heldout
horizon_robust_action_rate
accepted_family_count
accepted_step_bucket_count
max_family_share
max_step_bucket_share
dataset_name_used
uses_outcome_at_commit
```

### 判断标准

P7 source controller weak pass：

```text
coverage_heldout >= 0.03
weak_CP_precision_heldout >= 0.50
V_ctrl_lcb_heldout > 0
long_risk_rate_heldout <= 0.15
accepted_family_count >= 16
max_family_share <= 0.50
coverage_heldout > 0 in all official folds
dataset_name_used = 0
uses_outcome_at_commit = 0
```

P7 strong pass：

```text
coverage_heldout >= 0.03
weak_CP_precision_heldout >= 0.65
V_ctrl_lcb_heldout > 0.05
long_risk_rate_heldout <= 0.10
horizon_robust_action_rate >= 0.03
```

### 可视化

```text
p7_controller_precision_coverage_frontier.svg
p7_controller_vctrl_longrisk_pareto.svg
p7_crossfit_fold_matrix.svg
p7_leaveout_stability_heatmap.svg
p7_support_balance_sunburst.svg
```

---

## P8：selected source runtime and payload apply

### 目标

测 selected source controller 的 online runtime，不把 offline materializer 或 oracle mask 混入 timed path。

### 必须记录

```text
runtime_candidate_id
controller_id
source_generator_id
step_count
active_step_count
accepted_action_count
candidate_count
zero_candidate_controller_kernel_count
zero_candidate_controller_sync_count
controller_launches_per_active_step_q90
controller_syncs_per_active_step_q90
feature_compute_time_ms_q90
certificate_compute_time_ms_q90
score_accept_time_ms_q90
payload_apply_time_ms_q90
base_train_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
memory_ratio
payload_apply_error_linf_max
payload_apply_cosine_min
offline_materializer_in_timed_path
disk_payload_lookup_in_timed_path
audit_outside_timed_path
```

### 判断标准

P8 runtime pass：

```text
zero_candidate_controller_kernel_count = 0
zero_candidate_controller_sync_count = 0
controller_launches_per_active_step_q90 <= 2
controller_syncs_per_active_step_q90 <= 1
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
payload_apply_error_linf_max <= 1e-6
offline_materializer_in_timed_path = 0
audit_outside_timed_path = 1
```

### 可视化

```text
p8_runtime_waterfall.svg
p8_payload_apply_time_distribution.svg
p8_step_ratio_q50_q90.svg
p8_controller_launches_per_active_step.svg
p8_memory_ratio.svg
```

---

## P9：system integration gate

### 目标

组合 generated source + certificate controller + selected runtime，判断是否能进入 official downstream。

### 必须记录

```text
system_candidate_id
source_generator_id
controller_id
runtime_candidate_id
source_generator_materialized
h20_immediate_direction_pass
horizon_extension_pass
certificate_sufficiency_pass
source_controller_pass
selected_runtime_pass
official_eligible
system_legal_controller_pass
coverage_heldout
weak_CP_precision_heldout
V_ctrl_lcb_heldout
long_risk_rate_heldout
step_ratio_q90
memory_ratio
no_fake
no_proxy
uses_dataset_name
uses_outcome_at_commit
base_acc_used_for_controller
```

### 判断标准

P9 pass：

```text
official_eligible = 1
system_legal_controller_pass = 1
source_generator_materialized = 1
h20_immediate_direction_pass = 1
horizon_extension_pass = 1
certificate_sufficiency_pass = 1
source_controller_pass = 1
selected_runtime_pass = 1
uses_dataset_name = 0
uses_outcome_at_commit = 0
base_acc_used_for_controller = 0
```

### 可视化

```text
p9_system_gate_dashboard.svg
p9_quality_runtime_pareto.svg
p9_route_ladder.svg
```

---

## P10：leave-dataset-out / leave-stratum-out

### 目标

只有 P9 pass 后 official 打开。证明 source generator/controller 不是 pooled artifact。

### 设置

```text
Leave-dataset-out:
  calibrate on two datasets, evaluate third

Leave-family-out:
  hold out high-volume source/action families

Leave-step-bucket-out:
  hold out early/mid/late train-stream buckets

Leave-score-bucket-out:
  hold out low/mid/high score buckets
```

### 必须记录

```text
split_type
heldout_entity
source_generator_id
controller_id
accepted_count
coverage
weak_CP_precision
strong_CP_precision
V_ctrl_lcb
long_risk_rate
horizon_robust_action_rate
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
real_beats_adamwparallel
real_beats_bestlr
dataset_name_used
```

### 判断标准

P10 pass：

```text
coverage > 0 in every main split；
macro weak_CP_precision >= 0.50；
macro V_ctrl_lcb > 0；
macro long_risk_rate <= 0.15；
dataset_name_used = 0；
no split-specific threshold。
```

### 可视化

```text
p10_leaveout_matrix.svg
p10_leave_dataset_out_dashboard.svg
p10_leave_family_step_score_heatmap.svg
p10_leaveout_failure_modes.svg
```

---

## P11：diagnostic paired replay scout

### 目标

只有 P9 pass 后运行；P11 仍是 diagnostic，不允许影响 P7 thresholds。

### 必须记录

```text
source_generator_id
controller_id
dataset
seed
horizon
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
real_beats_adamwparallel
real_beats_bestlr
real_beats_noop
real_beats_random
task_safe
diagnostic_downstream_used_for_controller
```

### 判断标准

Scout pass：

```text
BeatRate_Real_vs_AdamWParallel >= 0.55
BeatRate_Real_vs_bestLR >= 0.50
task_safe = 1
diagnostic_downstream_used_for_controller = 0
```

### 可视化

```text
p11_paired_replay_scout_pareto.svg
p11_branch_win_matrix.svg
p11_shuffle_control_matrix.svg
```

---

## P12：official paired replay

### 目标

只有 P10 pass 后 official 打开。验证 functional source/controller 的 local causal advantage。

### Branches

```text
RealSourceFunctional
AdamWOnly
AdamWParallel
bestLR
NoOp
Random
ScaledAdamW
ShuffledSourcePayload
ShuffledCertificate
ShuffledSourceScore
ShuffledAcceptBit
FunctionalChannelShuffled
TailMaskShuffled
DatasetRouteShuffled
EventRouteShuffled
```

### 判断标准

$$
BeatRate_{RealSourceFunctional\ vs\ AdamWParallel}\ge 0.60,
$$

$$
BeatRate_{RealSourceFunctional\ vs\ bestLR}\ge 0.60.
$$

Task safety：

$$
Acc_{RealSourceFunctional}\ge Acc_{AdamW}-0.005.
$$

Shuffle controls must fail:

```text
ShuffledSourcePayload = fail
ShuffledCertificate = fail
ShuffledSourceScore = fail
ShuffledAcceptBit = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
```

### 可视化

```text
p12_official_paired_replay_pareto.svg
p12_macro_beat_rate.svg
p12_signal_stratum_win_matrix.svg
p12_shuffle_control_fail_dashboard.svg
```

---

## P13：short/full training and MLP comparison

### 目标

只有 P12 pass 后打开 official functional training comparison。这里才正式回答 functional DG-KAN acc 与 MLP 的关系。

### 设置

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  short-run: 0,1,2
  full-run: 0..9

models/controls:
  LQ-t2-h256 + selected functional source controller
  LQ-t2-h256 AdamWOnly
  LQ-t2-h256 AdamWParallel
  matched MLP
  StrongLRGrid MLP
  QuadraticFeatureMLP
  NoOp source
  Random source
  Shuffled source/controller controls
```

### 必须记录

```text
dataset
seed
model_id
controller_id
steps_or_epochs
train_acc
val_acc
test_acc
best_val_acc
final_acc
train_loss
val_loss
test_loss
ECE
NLL
CEp99
margin_p10
curvature
local_lipschitz
ValLossAUC_step
ValLossAUC_time
time_to_target
steps_to_target
functional_event_count
candidate_rate
coverage
bad_event_rate
null_rate
long_risk_rate
step_ratio_q90
memory_ratio
params
forward_flops
backward_flops
strong_baseline_beaten
```

### 判断标准

Task safety：

$$
Acc_{functional}\ge Acc_{AdamWOnly}-0.005.
$$

MLP comparison diagnostic：

$$
Acc_{functional}\ge Acc_{matched\ MLP}-0.005
$$

or at least one structural advantage:

$$
ECE_{functional}<ECE_{matched\ MLP},
$$

$$
NLL_{functional}<NLL_{matched\ MLP},
$$

$$
CEp99_{functional}<CEp99_{matched\ MLP},
$$

$$
ValLossAUC_{time,functional}<ValLossAUC_{time,MLP}.
$$

Full functional pass：

```text
beats AdamWParallel or bestLR in macro metrics；
not explained by StrongLRGrid；
not explained by QuadraticFeatureMLP；
shuffle controls fail；
step_ratio_q90 <= 1.50；
memory_ratio <= 1.05。
```

### 可视化

```text
p13_acc_lq_functional_vs_mlp.svg
p13_val_loss_auc_time.svg
p13_time_to_target.svg
p13_ece_nll_cepp99_comparison.svg
p13_sample_efficiency_curve.svg
p13_strong_baseline_dashboard.svg
```

---

## P14：continual / anti-forgetting diagnostic

### 目标

只有 P13 初步 pass 后打开。验证 beyond-MLP 不只是 single-task acc，而是在 continual / stability 上有结构优势。

### 设置

```text
Task sequence examples:
  MNIST -> Fashion-MNIST -> KMNIST
  Fashion-MNIST -> MNIST -> KMNIST
  KMNIST -> MNIST -> Fashion-MNIST

No dataset-specific controller branch.
```

### 必须记录

```text
sequence_id
model_id
controller_id
task_id
step
current_task_acc
previous_task_acc
retained_accuracy
forgetting
backward_transfer
forward_transfer
old_task_CEp99
old_task_margin_p10
old_task_ECE
functional_event_count
long_risk_rate
step_ratio_q90
memory_ratio
```

### 判断标准

Continual diagnostic pass：

$$
Forgetting_{functional}\le Forgetting_{MLP}-\epsilon,
$$

or:

$$
RetainedAcc_{functional}\ge RetainedAcc_{MLP}+\epsilon.
$$

This does not replace P12/P13; it is a higher-level external advantage diagnostic.

### 可视化

```text
p14_forgetting_curve.svg
p14_retained_accuracy_matrix.svg
p14_backward_forward_transfer.svg
p14_old_task_tail_risk.svg
```

---

# 10. 并行执行计划

为了加快实验，v9.4.1 按下面并行批次执行。

## Batch A：Source panel and generator implementation

```text
A1: P0 boundary reproduction
A2: P1 stratified panel builder
A3: AP0b/AP0c generator implementation
A4: AP0d/AP0e/AP0f generator implementation
A5: action apply replay/hash audit
```

产物：

```text
p0_boundary_v9410.csv
p1_stratified_source_panel.csv
p3_real_source_generator.csv
source_payload_trace_v9410.csv
source_certificate_trace_v9410.csv
action_apply_replay_trace_v9410.csv
```

## Batch B：Outcome materialization

```text
B1: P4 h20 smoke for all primitives
B2: P5 h80/h240 extension only for P4 survivors
B3: control branch quality audit
B4: retry manifest / materializer throughput audit
```

产物：

```text
p4_h20_immediate_source_smoke.csv
p5_horizon_extension_longrisk.csv
source_outcome_trace_v9410.csv
branch_horizon_completion_trace_v9410.csv
materializer_worker_trace_v9410.csv
```

## Batch C：Certificate and controller

```text
C1: P6 certificate sufficiency
C2: P7 crossfit source controller
C3: leave-family / leave-step / leave-score diagnostics
C4: controller failure autopsy
```

产物：

```text
p6_certificate_sufficiency.csv
p7_minimal_source_certificate_controller.csv
controller_frontier_trace_v9410.csv
leaveout_source_controller_trace_v9410.csv
controller_failure_autopsy_v9410.csv
```

## Batch D：Runtime

```text
D1: payload apply microbench for each source primitive
D2: selected controller runtime after P7 survivor
D3: no-event preservation / base AdamW equivalence
D4: online/offline runtime isolation audit
```

产物：

```text
p8_selected_source_runtime.csv
runtime_component_trace_v9410.csv
no_event_preservation_trace_v9410.csv
runtime_isolation_audit_v9410.csv
```

## Batch E：Base-Acc Sentinel

```text
E1: LQ-t2-h256 fixed-config training
E2: matched MLP fixed-config training
E3: QuadraticFeatureMLP diagnostic baseline
E4: acc/loss/calibration dashboard
```

产物：

```text
p2_base_acc_sentinel.csv
base_acc_training_trace_v9410.csv
lq_vs_mlp_acc_dashboard_v9410.svg
```

Batch E 不参与 controller choice。

---

# 11. Failure taxonomy

```text
R0-BoundaryUnstable
R1-SourcePanelSamplerStillBiased
R2-StaticLegalSelectorUnexpectedPass
R3-SourceGeneratorNotMaterialized
R4-GeneratedSourceImmediateDirectionFail
R5-ImmediatePositiveButHorizonFragile
R6-CertificateNotEffectValid
R7-SourceControllerHeldoutFail
R8-SelectedRuntimeFail
R9-SystemLegalSourceControllerPass
R10-LeaveoutFail
R11-PairedReplayFail
R12-ShortFullTrainingFail
R13-MLPComparisonFailButLocalFunctionalPass
R14-ExternalReadyCandidate
```

Failure table 必须记录：

```text
F1_source_panel_sampling_bias
F2_static_legal_selector_opaque
F3_source_generator_missing
F4_action_apply_error
F5_h20_immediate_direction_fail
F6_horizon_long_risk_fail
F7_certificate_no_effect_lift
F8_controller_support_collapse
F9_runtime_payload_apply_too_slow
F10_base_acc_sentinel_catastrophic
F11_dataset_tuning_detected
F12_outcome_at_commit_violation
F13_diagnostic_promoted_to_official
F14_short_full_not_open
```

---

# 12. Required artifacts

```text
run_manifest.json
route_decision.json
aggregate_decision.json
contract_audit_v9410.csv
provenance_audit_v9410.csv
failure_table_v9410.csv
artifact_hashes_v9410.csv

p0_boundary_reproduction_v9410.csv
p1_stratified_source_panel_builder.csv
p2_base_acc_sentinel_lq_vs_mlp.csv
p3_real_value_producing_source_generator.csv
p4_h20_immediate_direction_smoke.csv
p5_horizon_extension_longrisk_audit.csv
p6_effect_valid_certificate_sufficiency.csv
p7_minimal_source_certificate_controller.csv
p8_selected_source_online_runtime.csv
p9_system_integration_gate_v9410.csv
p10_leaveout_boundary_v9410.csv
p11_diagnostic_paired_replay_scout_v9410.csv
p12_official_paired_replay_v9410.csv
p13_short_full_training_mlp_comparison_v9410.csv
p14_continual_antiforgetting_v9410.csv

source_payload_trace_v9410.csv
source_certificate_trace_v9410.csv
action_apply_replay_trace_v9410.csv
source_outcome_trace_v9410.csv
branch_horizon_completion_trace_v9410.csv
controller_frontier_trace_v9410.csv
runtime_component_trace_v9410.csv
base_acc_training_trace_v9410.csv

figures/
```

---

# 13. 最终判断标准

v9.4.1 不能用下面任何结果宣称 success：

```text
source panel balanced；
oracle source selector pass；
source generator materialized；
action apply error = 0；
h20 smoke 有一点 weak CP；
certificate schema complete；
Base-Acc Sentinel 中 LQ acc 高；
runtime microbench 快。
```

只有下面组合才允许进入 system route：

```text
source_generator_materialized = 1
h20_immediate_direction_pass = 1
horizon_extension_pass = 1
certificate_sufficiency_pass = 1
source_controller_pass = 1
selected_runtime_pass = 1
no_fake = 1
no_proxy = 1
uses_dataset_name = 0
uses_outcome_at_commit = 0
base_acc_used_for_controller = 0
```

只有 P9 pass 后，才能打开 LDO/LSO、paired replay 和 short/full training。

最终一句话：

> v9.4.1 的价值不是“把 AP0 再筛一遍”，而是第一次让 value-producing source action 在 commit time 真实生成、真实落盘、真实 replay、真实 matched-control evaluation，并用 effect-valid certificate 判断它是否能成为 system-legal functional update primitive。
