# DG-KAN v9.4.5 Robust Source Generator / Objective-Aligned Certificate / Parallel Closure 完整实验计划

> 本计划基于 v9.4.4 `Source Value Objective Audit / Oracle-Seeded Generator Reset / Parallel Baseline` 的真实执行结果制定。  
> v9.4.4 的 terminal route 是：
>
> ```text
> route = R3-GeneratorDestructive
> base_candidate = LQ-t2-h256
> success_v9440_strict_purekan_functional = False
> success_v9440_full_functional = False
> success_v9440_external_ready = False
> primary_blocker = oracle_source_survivor_transform_generator_destructive
> ```
>
> v9.4.5 不再继续调 AP0b-AP0q 的阈值，也不把 oracle survivor 直接当 controller。v9.4.5 的核心任务是：
>
> $$
> \boxed{
> \text{从“找到好 source”推进到“生成并保留好 source 的 value，同时让证书和 runtime 可部署”。}
> }
> $$
>
> 本轮不以 MNIST / Fashion-MNIST / KMNIST 打榜为目标。数据集可以用于诊断 failure mode、support、horizon、base health，但不能用于 dataset-specific selector、generator、threshold 或 controller。

---

# 0. 执行摘要

v9.4.4 的关键结果不是“终于成功”，也不是“source frontier absent”。它真正纠正了 v9.4.3 的一个错误边界：full AP0 universe 中存在小而强的 horizon-robust oracle source survivor。

核心数据：

```text
P1 full AP0 universe:
  weak_CP_h20_rate = 0.10674547983310154
  V_ctrl_h20_LCB = -0.7504334275695862
  P(LongRisk_h240 | WeakCP_h20) = 0.8013029315960912

P2 exhaustive oracle:
  best = ORC-D-HorizonRobust-K16
  h20 weak CP = 0.8125
  h20 V_ctrl LCB = 0.13772944106165844
  h240 long-risk = 0.0
  weak_source_survivor_pass = 1
  strong_source_survivor_pass = 1

P4 legal selector:
  best = LS-E-state_NLL_proxy
  best AUC weak CP = 0.5750092242383822
  best topK h20 weak CP = 0.125
  best topK h20 V_ctrl LCB = -0.2595459073949538
  best topK h240 long-risk = 0.828125

P5 oracle-seeded AP0b-AP0f generator:
  generated actions = 80
  branch-horizon rows = 2160 / 2160
  source-positive lost rate = 0.8888888888888888
  Damage median = -0.8996349092340097
  best generated h20 weak CP = 0.125
  best generated h20 V_ctrl LCB = -0.81038825849006
  best generated h240 long-risk = 0.75

P6 direct AP0l-AP0q generator:
  generated actions = 96
  branch-horizon rows = 2592 / 2592
  best = AP0l-LinearizedTrustRegionSource
  best h20 weak CP = 0.125
  best h20 V_ctrl LCB = -0.8829934048526893
  best h240 long-risk = 0.75

P7 effect certificate:
  AUC weak CP = 0.5015054877456535
  AUC long-risk = 0.4984089703765437
  monotone sign pass = 0

P10 Base-Acc Sentinel strong baseline:
  rows = 120
  datasets = MNIST, Fashion-MNIST, KMNIST
  seeds = 0..9
  mean_test_acc_LQ = 0.6537760416666667
  mean_test_acc_MatchedMLP = 0.562890625
  mean_test_acc_AdamWStrongLRGridMLP = 0.628515625
  mean_test_acc_QuadraticFeatureMLP = 0.40234375
  LQ_minus_MLP = +0.09088541666666672
  LQ_minus_AdamWStrongLRGridMLP = +0.025260416666666674
  base_acc_used_for_controller = 0
```

因此，本轮后的真实判断是：

$$
\boxed{
\text{AP0 full universe 里有好 source；但 legal selector 看不见，现有 generator 会破坏，当前 certificate 不具备效果含义。}
}
$$

v9.4.5 的最低有效推进不是“训练 acc 提升”，而是完成以下判定：

```text
1. 现有 generator destructiveness 是否可以被 objective-solved / value-preserving generator 消除？
2. oracle survivor 的 value 是否能被合法 generator 保留，而不是被 AP0b-AP0f transform 破坏？
3. 不使用 oracle outcome 时，是否能通过 legal source objective 直接生成 h20 value-positive 且 h240 safe 的 source action？
4. certificate 是否能从 construction-valid 变成 effect-valid？
5. 若选出 controller，selected payload runtime 是否能重新进入 step_ratio_q90 <= 1.50？
6. Base-Acc Sentinel 是否继续显示 LQ base 不 catastrophic，并与 stronger MLP baseline 保持可解释差距？
```

---

# 1. 独立数据判断

## 1.1 v9.4.4 是进展，但不是能力成功

v9.4.4 的最大进展是纠正了 v9.4.3 的 `FullAP0OracleSourceFrontierAbsent` 结论。v9.4.3 的 ORC64 已经接近 gate，但 $V_{ctrl}$ LCB 略负；v9.4.4 改用 exhaustive AP0 source frontier 后找到了 `ORC-D-HorizonRobust-K16`，同时满足：

$$
WeakCP_{h20}=0.8125,
$$

$$
LCB(V_{ctrl,h20})=0.13772944106165844 > 0,
$$

$$
LongRisk_{h240}=0.
$$

这说明 AP0 full universe 中确实存在小而强的 source survivor。它不是 deployable，因为它使用 outcome/oracle 信息；但作为 upper bound，它非常重要。它说明当前问题不是：

```text
AP0 full universe 完全没有 value-positive source；
KAN base 完全没有 functional 信号；
source outcome materializer 不可信；
full control outcome universe 不能支撑诊断。
```

当前问题更具体：

```text
1. legal selector 不能识别这个 K16 survivor；
2. 现有 AP0b-AP0f transform 会把 survivor value 破坏掉；
3. direct AP0l-AP0q generator 没有自己产生 value-positive / horizon-safe frontier；
4. certificate 对 weak CP 和 long-risk 几乎随机。
```

这就是为什么 route `R3-GeneratorDestructive` 比 `FullAP0OracleSourceFrontierAbsent` 更接近本质。

## 1.2 weak CP 不是足够目标

v9.4.4 的 P1 显示：

```text
P(Vctrl_h20 > 0 | WeakCP_h20) = 1.0
P(LongRisk_h240 | WeakCP_h20) = 0.8013029315960912
Corr(WeakCP_h20, V_ctrl_h20) = 0.3665950128325245
```

这说明 h20 weak CP 对 h20 value 有正相关，但它几乎不约束 h240 long-risk。换句话说，旧目标：

$$
WeakCP_{h20}=1
$$

不是官方 source objective。新的 source objective 至少应是：

$$
Y_{robust}(a)=1
\iff
WeakCP_{h20}(a)=1
\land LCB(V_{ctrl,h20}(a))>0
\land LongRisk_{h240}(a)\le \tau_{long}
\land Support(a)=1.
$$

其中 $\tau_{long}$ 初始设为 $0.10$，strong diagnostic 可以用 $0.0$。

v9.4.5 以后，所有 source / generator / certificate 都必须同时报告：

```text
h20 weak CP；
h20 V_ctrl LCB；
h80 weak CP；
h240 long-risk；
horizon robust action coverage；
source-positive lost rate；
certificate lift on weak CP；
certificate lift on long-risk。
```

只报告 weak CP 会重新造成错误路线。

## 1.3 legal selector 当前失败得很远

v9.4.4 的 best legal selector 是 `LS-E-state_NLL_proxy`：

```text
AUC weak CP = 0.5750092242383822
AUC long-risk = 0.3980992738567873
topK h20 weak CP = 0.125
topK h20 V_ctrl LCB = -0.2595459073949538
h240 long-risk = 0.828125
```

和 oracle K16 比：

```text
Oracle K16 h20 weak CP = 0.8125
Legal topK h20 weak CP = 0.125
gap = 0.6875

Oracle K16 h20 V_ctrl LCB = +0.1377
Legal topK h20 V_ctrl LCB = -0.2595

Oracle K16 h240 long-risk = 0.0
Legal topK h240 long-risk = 0.828125
```

这不是换一个 threshold 能解决的问题。当前 legal features 主要是边际状态特征、payload几何特征或弱 proxy，它们没有捕捉到 “action × state × horizon response” 的交互。

## 1.4 generator destructiveness 是当前最硬 blocker

v9.4.4 最有说服力的实验是 oracle-seeded generator preservation。它直接排除了一个借口：不是因为 source 选错，才导致 generator 差；即使从 P2 oracle survivor 出发，AP0b-AP0f 也把 value 破坏掉。

关键指标：

```text
source_positive_horizon_count = 225
source_positive_lost_rate = 0.8888888888888888
Damage_median = -0.8996349092340097
best generated h20 weak CP = 0.125
best generated h20 V_ctrl LCB = -0.81038825849006
best generated h240 long-risk = 0.75
```

这意味着 AP0b-AP0f 不是 “value-preserving transform”。它更像是 construction-valid transform：payload 合法、hash 合法、action apply 合法，但 effect 方向被破坏。

v9.4.5 不能再给 AP0d / AP0l 调小参数，而应换问题：

$$
\boxed{
\text{如何生成一个本身解 value/risk constrained objective 的 source action？}
}
$$

## 1.5 direct generator 仍是 heuristic，不是 objective solver

AP0l-AP0q direct generator 已经真实 materialize：

```text
payload_tensor_written = 1
certificate_tensor_written = 1
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max = 0.0
commit_time_available = 1
uses_dataset_name = 0
uses_outcome_at_commit = 0
```

但是 best direct primitive 仍然只有：

```text
h20 weak CP = 0.125
h20 V_ctrl LCB = -0.8829934048526893
h240 long-risk = 0.75
```

所以 direct generator 的实现链路已经通，但目标函数不对。它没有在生成时真正优化：

$$
\max_{\Delta\theta} \widehat{V}_{20}(\Delta\theta) - \lambda_{risk}\widehat{R}_{240}(\Delta\theta) - \lambda_{cost}C(\Delta\theta).
$$

v9.4.5 需要从 heuristic payload builder 转向 objective-solved source generator。

## 1.6 Base-Acc Sentinel 是好消息，但不是 functional success

v9.4.4 的 Base-Acc Sentinel 是目前为止更强的 base health 检查：

```text
MNIST / Fashion-MNIST / KMNIST
seeds = 0..9
model_count = 4
rows = 120
LQ mean test acc = 0.6537760416666667
MatchedMLP = 0.562890625
AdamWStrongLRGridMLP = 0.628515625
QuadraticFeatureMLP = 0.40234375
```

因此 fixed-config sentinel 下：

$$
Acc_{LQ} - Acc_{MatchedMLP} = 0.09088541666666672,
$$

$$
Acc_{LQ} - Acc_{StrongLRGridMLP} = 0.025260416666666674.
$$

这说明 LQ-t2-h256 base 没有 catastrophic fail，并且在这个 sentinel protocol 下强于 MatchedMLP 和 AdamWStrongLRGridMLP。但它不是 official functional success，因为：

```text
base_acc_used_for_controller = 0；
source controller 未打开；
selected runtime 未打开；
paired replay 未打开；
short/full functional training 未打开；
这些 acc 没有用来调 selector/generator/controller。
```

所以回答 “有没有在数据集上训练，acc 如何” 时必须精确：有训练，但只是 Base-Acc Sentinel；不是 official DG-KAN functional training。

---

# 2. v9.4.5 总体目标

v9.4.5 的总体目标不是继续救 AP0d、AP0l 或 certificate threshold，而是把 source generation 从 heuristic transform 改成 objective-aligned generation，并明确分解三种失败：

```text
1. Selector failure：好 source 存在，但 legal selector 看不见。
2. Transform failure：好 source 被 generator transform 破坏。
3. Objective failure：direct generator 本身没有产生 value-positive / horizon-safe action。
```

v9.4.5 的核心命题是：

$$
\boxed{
\exists G_{legal}\;\text{such that}\;G_{legal}(s_t,b_t,g_t)\rightarrow (\Delta\theta, Cert),
\quad
LCB(V_{20})>0,
\quad
LongRisk_{240}\le 0.10,
\quad
C\le C_{max}.
}
$$

其中 $G_{legal}$ 不允许使用 dataset name、validation/test、future outcome、outcome-at-commit 或 oracle label。Oracle survivor 只能用于 diagnostic upper bound 和 generator preservation test。

v9.4.5 的强目标：

```text
robust_source_generator_pass = 1
certificate_effect_valid_pass = 1
source_controller_pass = 1
selected_runtime_pass = 1
system_legal_controller_pass = 1
```

v9.4.5 的最低有效推进目标：

```text
1. 完成 oracle survivor source-objective decomposition；
2. 完成 AP0r-AP0w objective-aligned generator implementation；
3. 完成 oracle-seeded preservation test，判断 destructiveness 是否可被消除；
4. 完成 legal/direct generator smoke，判断不用 oracle 能否生成 robust source；
5. 完成 effect-valid certificate redesign；
6. 完成 Base-Acc Sentinel strong baseline continuation；
7. 如果 upstream pass，完成 selected payload runtime；
8. 给出明确 route：selector fail / transform fail / direct generator fail / certificate fail / runtime fail / system pass。
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
official selector/controller 不使用 dataset_name 分支
official selector/controller 不使用 validation/test metric
official selector/controller 不使用 future outcome
official selector/controller 不使用 outcome-at-commit
Base-Acc Sentinel 不用于 selector/generator/controller
oracle survivor 不进入 official path，只用于 upper-bound diagnostic
```

Functional update 仍是 update rule，不是 loss trick：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{AdamW-equivalent}
+
\Delta\theta_{functional}.
$$

任务损失仍是标准 CE：

$$
L_{task}=CE(y,p_\theta(x)).
$$

允许使用：

```text
manual gradient / AdamW direction；
train-batch logits / CE / margin / entropy；
commit-time linearized response；
local trust-region line search based on train-batch manual forward；
train-stream calibration support；
family/horizon/support diagnostics；
oracle survivor as diagnostic seed only；
parallel Base-Acc Sentinel；
parallel generator smoke / certificate audit / runtime microbench。
```

---

# 4. 核心假设

## H1：v9.4.4 的 source objective mismatch 是真实的

H1 认为 weak CP 不能作为唯一 source objective。h20 weak CP 与 h20 value 相关，但不能排除 h240 long-risk。

H1 成立标准：

```text
P(LongRisk_h240 | WeakCP_h20) >= 0.50
Corr(WeakCP_h20, V_ctrl_h20) > 0
but weak-only topK long-risk > 0.10
```

H1 失败标准：

```text
重新计算后 weak CP 同时稳定约束 V_ctrl_h20 和 h240 long-risk。
```

若 H1 成立，所有后续 gate 使用 robust source objective，而不是 weak CP-only。

## H2：generator destructiveness 不是不可避免，而是 AP0b-AP0f transform 目标错了

H2 认为 AP0b-AP0f 破坏 oracle survivor，是因为 transform 没有显式保留 $V_{20}$ / horizon safety，而不是所有生成式 source action 都必然破坏 value。

H2 成立标准：

```text
至少一个 AP0r-AP0w 在 oracle-seeded preservation test 中：
  source_positive_lost_rate <= 0.25
  Damage_median >= -0.05
  h20 weak CP >= 0.60
  h20 V_ctrl LCB > 0
  h240 long-risk <= 0.10
```

H2 失败标准：

```text
所有 AP0r-AP0w 从 oracle survivor 出发仍：
  source_positive_lost_rate > 0.50
  或 Damage_median < -0.20
  或 h20 V_ctrl LCB <= 0
  或 h240 long-risk > 0.10
```

若 H2 失败，应停止 transform route，进入 direct objective solver / raw source selection / new primitive design。

## H3：legal source selector 当前看不见 oracle survivor，但 effect certificate 可能提供新 observability

H3 认为 state_NLL_proxy、PayloadNorm、AdamWConflictLow 这类边际特征不足；必须生成 action-effect certificate，记录 action 对 CE/margin/tail/AdamW conflict 的 commit-time response。

H3 成立标准：

```text
certificate_AUC_robust_source >= 0.70
certificate_AUC_longrisk >= 0.70
P(robust_source | cert_pass) >= 0.60
P(longrisk | cert_pass) <= 0.10
monotone_sign_pass = 1
certificate_cost_q90_ms <= 0.20
```

H3 失败标准：

```text
certificate AUC 仍接近随机，或 cert-pass 不降低 long-risk。
```

## H4：direct generator 必须从 heuristic builder 变成 constrained objective solver

H4 认为 AP0l-AP0q 失败不是因为 direct generation 方向本身不可能，而是因为它们没有真正解一个 value/risk constrained objective。

H4 成立标准：

```text
至少一个 direct generator AP0r-AP0w：
  generated_action_count >= 64
  h20 weak CP >= 0.60
  h20 V_ctrl LCB > 0
  h240 long-risk <= 0.10
  bad_event_h20 <= 0.05
  support_balance_pass = 1
```

H4 失败标准：

```text
所有 direct generator h20 weak CP <= 0.25 或 V_ctrl LCB <= 0 或 h240 long-risk >= 0.50。
```

## H5：Base-Acc Sentinel 只作为健康监控，不能替代 functional evidence

H5 认为 LQ base 目前不 catastrophic，但这个 acc 不能证明 functional controller 成功。

H5 成立标准：

```text
Base-Acc Sentinel complete；
base_acc_used_for_controller = 0；
LQ_catastrophic_fail = 0；
LQ_minus_AdamWStrongLRGridMLP_mean_test_acc >= -0.02。
```

H5 失败标准：

```text
LQ base 在 fixed sentinel 下大幅低于 matched MLP / StrongLRGridMLP，或被用于调 controller。
```

## H6：实验慢的主要原因是串行暴露 blocker，必须改成 preflight + parallel batches

H6 认为 v9.4.1 的 0-row、v9.4.2 的 source panel bad、v9.4.4 的 generator destructive 都应该通过更强 preflight 和并行矩阵更快暴露。

H6 成立标准：

```text
所有新 generator 在 single-action preflight 中先通过：
  payload written；
  certificate written；
  action apply replay；
  one-action outcome row sink；
  branch/horizon completion；

再进入 8-action smoke；
再进入 64-action scale；
不同 primitive / source panel / certificate / Base-Acc Sentinel 并行运行。
```

---

# 5. v9.4.5 数据合同

## 5.1 Source action identity contract

每个 source action 记录：

```text
source_action_id
source_action_hash
source_origin
source_panel_id
source_selection_mode
source_selection_legality
source_payload_hash
source_certificate_hash
source_family_id
source_bucket_id
source_horizon_tag
source_step_id
source_dataset
source_seed
source_oracle_used_for_diagnostic
source_used_for_official
```

Pass：

```text
duplicate_source_action_id_count = 0
source_payload_hash_missing = 0
source_certificate_hash_missing = 0
source_oracle_used_for_official = 0
source_selection_legality_pass = 1
```

## 5.2 Generated action contract

每个 generated action 记录：

```text
generated_action_id
generator_id
generator_family
source_action_id
payload_hash
certificate_hash
payload_tensor_written
certificate_tensor_written
action_apply_error_linf
action_apply_error_relative
action_apply_cosine
commit_time_available
uses_dataset_name
uses_outcome_at_commit
uses_future_step
uses_validation_or_test
```

Pass：

```text
payload_tensor_written = 1
certificate_tensor_written = 1
payload_hash_missing_count = 0
certificate_hash_missing_count = 0
action_apply_error_linf_max <= 1e-7
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_future_step = 0
uses_validation_or_test = 0
```

## 5.3 Source outcome contract

每个 source/generated action 必须在 branches × horizons 下记录：

```text
action_id
source_action_id
generator_id
branch
horizon
CE_delta
CEp99_delta
NLL_delta
ECE_delta
margin_p10_delta
curvature_delta
V_ctrl
WeakCP
StrongCP
LongRisk
BadEvent
NullEvent
TaskSafe
beats_AdamWParallel
beats_bestLR
beats_NoOp
beats_Random
outcome_row_id
outcome_hash
```

Branches：

```text
RealFunctional or RealGenerated
AdamWParallel
AdamWOnly
bestLR
NoOp
Random
ShuffledPayload
CertificatePassNoPayload
SourceAP0Baseline
```

Horizons：

```text
h20
h80
h240
```

Pass：

```text
branch_horizon_row_count_actual = branch_horizon_row_count_expected
branch_completion_rate = 1.0
horizon_completion_rate = 1.0
secondary_delta_completion_rate = 1.0
label_exclusivity_violation_count = 0
duplicate_outcome_row_id_count = 0
NaN/Inf count = 0
```

## 5.4 Robust source labels

定义：

$$
Y_{weak20}(a)=1 \iff WeakCP_{h20}(a)=1.
$$

$$
Y_{value20}(a)=1 \iff LCB(V_{ctrl,h20}(a))>0.
$$

$$
Y_{safe240}(a)=1 \iff LongRisk_{h240}(a)\le 0.10.
$$

$$
Y_{robust}(a)=Y_{weak20}(a)\land Y_{value20}(a)\land Y_{safe240}(a)\land Support(a)=1.
$$

Strong diagnostic：

$$
Y_{robust,strong}(a)=Y_{weak20}(a)\land Y_{value20}(a)\land LongRisk_{h240}(a)=0.
$$

所有 selector、generator、certificate 都必须报告 $Y_{weak20}$、$Y_{value20}$、$Y_{safe240}$ 和 $Y_{robust}$，不得只报告 weak CP。

## 5.5 Certificate contract

新 certificate 不再只是 construction certificate，而必须包含 action-effect fields：

```text
cert_id
cert_schema_version
payload_hash_bound
pred_CE_delta_h20
pred_margin_delta_h20
pred_tail_CE_delta
pred_tail_margin_delta
adamw_alignment
negative_grad_alignment
trust_region_radius
scale_alpha
line_search_accept_bit
linearization_residual_ucb
tail_risk_ucb
horizon_guard_score
support_lcb
cost_estimate_ms
certificate_pass
```

Effect-valid pass：

```text
AUC_robust_source >= 0.70
AUC_longrisk >= 0.70
P(Y_robust=1 | cert_pass) >= 0.60
P(LongRisk_h240=1 | cert_pass) <= 0.10
monotone_sign_pass = 1
certificate_cost_q90_ms <= 0.20
```

---

# 6. Generator candidates

## AP0r：Objective-Solved Last-Edge Trust Region Source

目标：不再从旧 source payload 做盲目 transform，而是在最后若干 edge-basis 参数上解一个小 trust-region objective。

生成形式：

$$
\Delta\theta_{AP0r}
=
\arg\min_{\Delta\theta \in \mathcal{S}_{last}}
\widehat{CE}_{20}(\theta+\Delta\theta)
+
\lambda_{tail}\widehat{TailRisk}_{240}(\theta+\Delta\theta)
+
\lambda_{adamw}\max(0,-\cos(\Delta\theta,\Delta\theta_{AdamW}))
$$

约束：

$$
\|\Delta\theta\|_2 \le r \|\Delta\theta_{AdamW}\|_2,
$$

$$
\widehat{TailRisk}_{240}(\Delta\theta) \le \tau_{tail},
$$

$$
\widehat{\Delta CE}_{lin} < 0.
$$

实现：

```text
只使用 manual gradient / train-batch CE / margin / logits；
只更新 LQ last-edge / selected low-rank edge subset；
不改变 task loss；
不使用 dataset name；
不使用 outcome label；
certificate 记录 objective terms、trust radius、linearized residual。
```

## AP0s：Oracle-Preservation Diagnostic Shrink-Project Source

目标：只用于 diagnostic，回答 “如果 source 是 oracle survivor，是否可以设计 transform 不破坏它”。

生成形式：

$$
\Delta\theta_{AP0s}
=
\alpha \cdot P_{safe}(\Delta\theta_{source}),
$$

其中 $P_{safe}$ 是由 train-batch tail constraints 和 AdamW alignment 构造的合法投影，$\alpha$ 由 commit-time line search 选择。

注意：

```text
oracle source 只能用于 diagnostic；
AP0s 不可 official；
若 AP0s 都不能 preserve，说明 transform route 基本失败。
```

Pass：

```text
source_positive_lost_rate <= 0.25
Damage_median >= -0.05
h20 V_ctrl LCB > 0
h240 long-risk <= 0.10
```

## AP0t：Tail-Projected Horizon Guard Source

目标：显式降低 h240 long-risk，而不是只追 h20 weak CP。

生成：

$$
\Delta\theta_{AP0t}
=
\Delta\theta_{descent}
-
\beta P_{tail}(\Delta\theta_{descent}),
$$

其中 $P_{tail}$ 来自当前 batch hard-tail examples 的 margin/CE response direction。

Certificate：

```text
hard_tail_fraction
pred_tail_CE_delta
pred_tail_margin_delta
tail_projection_norm
tail_risk_ucb
horizon_guard_score
```

## AP0u：AdamW-Compatible Residual Source

目标：避免 functional action 与 AdamW 主方向冲突。

生成：

$$
\Delta\theta_{AP0u}
=
\Delta\theta_{AdamW-residual}
-
Proj_{conflict}(\Delta\theta_{AdamW-residual}).
$$

约束：

$$
\cos(\Delta\theta_{AP0u},\Delta\theta_{AdamW})\ge 0.
$$

## AP0v：Low-Rank Edge Objective Solver

目标：在 FullEdge / LQ low-rank 子空间中产生可解释 source update。

生成：

$$
\Delta W = U_k A V_k^T,
$$

其中 $U_k,V_k$ 来自 train-batch edge response covariance，$A$ 由 constrained least-squares 解出。

Pass 关注：

```text
low-rank rank <= 8
payload apply cost low
h20 value positive
h240 long-risk controlled
```

## AP0w：No-Transform Source Acceptance Diagnostic

目标：判断是否 “只要不 transform，source 本身就能保住”。这不是 official route，而是对比 AP0b-AP0f destructiveness 的 sanity baseline。

```text
Input = P2 oracle survivor or legal-selected source panel
Output = unchanged source payload with updated certificate
```

如果 AP0w oracle-seeded pass，而 AP0r-AP0v fail，说明 generator 不该改 source payload，而应优先做 legal source selector / certificate。  
如果 AP0w legal-seeded fail，但 oracle-seeded pass，说明 selector 是主 blocker。  
如果 AP0w oracle-seeded也 fail，说明 outcome table/source matching有 bug，需回到 source identity audit。

---

# 7. 实验阶段

---

## P0：v9.4.4 boundary reproduction and route correction

### 目标

复现 v9.4.4 的关键边界，确认本轮不是在错误 artifact 上继续。尤其要确认：

```text
frontier_absent = 0；
legal_selector_capacity_pass = 0；
generator_destructive_fail = 1；
direct_generator_pass = 0；
certificate_effect_valid_pass = 0；
base_acc_sentinel_pass = 1；
system_legal_controller_pass = 0。
```

### 必须记录

```text
source_run_id
source_artifact_hash
route_v9440
best_oracle_selector
best_oracle_K
best_oracle_h20_weak_CP
best_oracle_h20_V_ctrl_lcb
best_oracle_h240_long_risk
best_legal_feature
best_legal_topK_h20_weak_CP
best_legal_topK_h20_V_ctrl_lcb
best_legal_topK_h240_long_risk
oracle_seeded_source_positive_lost_rate
oracle_seeded_Damage_median
direct_best_primitive
direct_best_h20_weak_CP
direct_best_h20_V_ctrl_lcb
direct_best_h240_longrisk
certificate_AUC_weak_CP
certificate_AUC_longrisk
mean_test_acc_LQ
mean_test_acc_MLP
mean_test_acc_AdamWStrongLRGridMLP
system_legal_controller_pass
```

### Pass

```text
P0 pass if all v9.4.4 headline metrics reproduced within tolerance；
frontier_absent = 0；
primary blocker remains generator destructive or stricter downstream blocker。
```

### 可视化

```text
p0_route_ladder_v9430_to_v9440.svg
p0_oracle_vs_legal_source_gap.svg
p0_generator_destructive_summary.svg
p0_base_acc_sentinel_strong_baseline.svg
```

---

## P1：source objective decomposition v2

### 目标

把 source objective 从 weak CP-only 改成 robust value/horizon objective，明确哪些 rows 是：

```text
weak but long-risk；
value positive but weak label missing；
horizon robust；
h20 positive but h240 risky；
legally visible vs legally hidden。
```

### 必须记录

```text
action_id
WeakCP_h20
WeakCP_h80
WeakCP_h240
V_ctrl_h20
V_ctrl_h80
V_ctrl_h240
V_ctrl_h20_LCB
LongRisk_h240
BadEvent_h20
NullEvent_h20
HorizonRobustCP
Y_robust
Y_robust_strong
family_id
bucket_id
step_bucket
payload_norm_bucket
state_NLL_bucket
AdamW_conflict_bucket
```

### 指标

```text
P(Vctrl_h20 > 0 | WeakCP_h20)
P(LongRisk_h240 | WeakCP_h20)
P(Y_robust | WeakCP_h20)
Corr(WeakCP_h20, V_ctrl_h20)
Corr(WeakCP_h20, LongRisk_h240)
Corr(V_ctrl_h20, LongRisk_h240)
Y_robust_base_rate
Y_robust_by_family
Y_robust_by_horizon
Y_robust_by_step_bucket
```

### Pass

P1 diagnostic pass：

```text
all full AP0 universe actions have robust labels；
Y_robust label quality pass；
objective mismatch quantified；
no NaN/Inf/label exclusivity violation。
```

### 可视化

```text
p1_weakcp_vs_value_horizon_phase_diagram.svg
p1_longrisk_given_weakcp_by_family.svg
p1_vctrl_h20_distribution_by_label.svg
p1_robust_source_density_heatmap.svg
p1_oracle_k16_neighborhood_profile.svg
```

---

## P2：oracle survivor anatomy and legal invisibility audit

### 目标

深入分析 `ORC-D-HorizonRobust-K16` 为什么好、为什么 legal selector 看不见。不是为了 official 使用 oracle，而是为了设计 generator/certificate。

### 必须记录

```text
oracle_action_id
oracle_selector_id
rank_in_oracle
family_id
bucket_id
step_id
state_NLL
state_CEp99
state_margin_p10
payload_norm
payload_linf
payload_entropy
adamw_alignment
negative_grad_alignment
true_delta_norm
tail_response_proxy
support_count
support_lcb
legal_feature_scores
legal_rank_by_each_feature
legal_selector_miss_reason
```

### 判断标准

P2 pass：

```text
all K16 oracle survivor actions joined with legal feature table；
legal rank distribution computed；
miss reason assigned for >= 95% oracle survivor actions。
```

Miss reasons：

```text
M1-state-feature-not-distinctive
M2-payload-geometry-not-distinctive
M3-AdamW-alignment-missing
M4-tail-risk-proxy-inverted
M5-support-too-low
M6-family/step sparse
M7-feature-cost-not-measured
M8-outcome-only-pattern
```

### 可视化

```text
p2_oracle_survivor_legal_rank_hist.svg
p2_oracle_vs_full_feature_radar.svg
p2_legal_miss_reason_bar.svg
p2_oracle_neighborhood_umap.svg
```

---

## P3：generator preflight matrix

### 目标

避免再次出现 “大 runner 跑完才发现 0 rows / apply failure / missing certificate”。每个 generator 先过单 action，再过 8-action smoke，再过 64-action scale。

### 输入 generator

```text
AP0r Objective-Solved Last-Edge Trust Region Source
AP0s Oracle-Preservation Diagnostic Shrink-Project Source
AP0t Tail-Projected Horizon Guard Source
AP0u AdamW-Compatible Residual Source
AP0v Low-Rank Edge Objective Solver
AP0w No-Transform Source Acceptance Diagnostic
```

### Source panel types

```text
S0_oracle_K16_diagnostic_only
S1_legal_topK_state_NLL
S2_legal_topK_AdamWConflictLow
S3_random_matched_to_oracle_family_step
S4_representative_PANEL_S256_sample
S5_direct_no_source
```

### 必须记录

```text
generator_id
source_panel_type
preflight_stage
input_action_count
output_action_count
payload_tensor_written
certificate_tensor_written
payload_hash_missing
certificate_hash_missing
action_apply_error_linf_max
action_apply_cosine_min
one_action_outcome_row_written
branch_horizon_completion_rate
unresolved_exception_count
wallclock_sec
```

### Pass

Single-action preflight pass：

```text
output_action_count >= 1
payload_tensor_written = 1
certificate_tensor_written = 1
action_apply_error_linf_max <= 1e-7
one_action_outcome_row_written = 1
unresolved_exception_count = 0
```

8-action smoke pass：

```text
branch_horizon_completion_rate = 1.0
label quality pass = 1
```

64-action scale ready：

```text
all above pass；
rows/sec >= 20；
no duplicate action_id/outcome_row_id。
```

### 可视化

```text
p3_generator_preflight_matrix.svg
p3_action_apply_error_by_generator.svg
p3_row_sink_preflight_status.svg
```

---

## P4：oracle-seeded generator preservation test

### 目标

测试 AP0r-AP0w 是否能在 oracle survivor 输入上保留 source value。这个阶段是 diagnostic-only，不可 official。

### 必须记录

```text
generator_id
source_panel_id = ORC-D-HorizonRobust-K16
source_action_count
generated_action_count
branch_horizon_rows_expected
branch_horizon_rows_actual
source_h20_weak_CP
source_h20_V_ctrl_LCB
source_h240_longrisk
generated_h20_weak_CP
generated_h20_V_ctrl_LCB
generated_h240_longrisk
Damage_mean
Damage_median
Damage_LCB
source_positive_horizon_count
source_positive_lost_count
source_positive_lost_rate
source_negative_fixed_rate
horizon_robust_coverage
support_balance_pass
```

Damage 定义：

$$
Damage(a,g)=V_{ctrl}(g)-V_{ctrl}(a).
$$

### Pass

Generator preservation pass：

```text
source_positive_lost_rate <= 0.25
Damage_median >= -0.05
generated_h20_weak_CP >= 0.60
generated_h20_V_ctrl_LCB > 0
generated_h240_longrisk <= 0.10
horizon_robust_coverage >= 0.10
```

Strong pass：

```text
source_positive_lost_rate <= 0.10
Damage_median >= 0
generated_h20_weak_CP >= 0.75
generated_h240_longrisk = 0
```

### Route interpretation

```text
If AP0w pass but AP0r-AP0v fail:
  transform is destructive; prioritize legal selector / no-transform source controller.

If AP0r/AP0s/AP0t pass:
  transform route remains viable; move to legal/direct seed tests.

If all fail including AP0w:
  source identity / oracle matching bug suspected; return to P1/P2 audit.
```

### 可视化

```text
p4_source_to_generated_damage_violin.svg
p4_source_positive_lost_rate_bar.svg
p4_oracle_seeded_value_curve_by_horizon.svg
p4_generator_preservation_pareto.svg
```

---

## P5：legal-seeded and representative-seeded generator test

### 目标

判断 generator 在不使用 oracle seed 时是否能从 legal source panels / representative panels 中产生 robust actions。

### 输入

```text
S1_legal_topK_state_NLL
S2_legal_topK_AdamWConflictLow
S3_random_matched_to_oracle_family_step
S4_representative_PANEL_S256_sample
```

### 必须记录

```text
generator_id
source_panel_type
source_action_count
generated_action_count
h20_weak_CP
h20_V_ctrl_LCB
h80_weak_CP
h240_weak_CP
h240_longrisk
bad_event_h20
null_event_h20
Y_robust_rate
horizon_robust_coverage
support_balance_pass
family_count
max_family_share
step_bucket_balance
```

### Pass

Weak legal-seeded pass：

```text
generated_action_count >= 64
h20_weak_CP >= 0.40
h20_V_ctrl_LCB > -0.05
h240_longrisk <= 0.30
```

Official candidate pass：

```text
generated_action_count >= 64
h20_weak_CP >= 0.60
h20_V_ctrl_LCB > 0
h240_longrisk <= 0.10
bad_event_h20 <= 0.05
support_balance_pass = 1
```

### 可视化

```text
p5_generator_by_source_panel_heatmap.svg
p5_legal_seed_value_horizon_frontier.svg
p5_support_balance_by_generator.svg
```

---

## P6：direct objective-solved generator test

### 目标

判断 AP0r/AP0t/AP0u/AP0v 在无 oracle seed、无 source panel selection 的情况下，是否能直接生成 robust source action。

### 必须记录

```text
generator_id
generated_action_count
branch_horizon_rows_expected
branch_horizon_rows_actual
payload_hash_missing
certificate_hash_missing
action_apply_error_linf_max
h20_weak_CP
h20_V_ctrl_LCB
h80_weak_CP
h240_weak_CP
h240_longrisk
bad_event_h20
null_event_h20
Y_robust_rate
horizon_robust_coverage
cost_ms_q90
memory_ratio
```

### Pass

Direct generator pass：

```text
generated_action_count >= 96
h20_weak_CP >= 0.60
h20_V_ctrl_LCB > 0
h240_longrisk <= 0.10
bad_event_h20 <= 0.05
cost_ms_q90 <= 0.30
```

Strong pass：

```text
h20_weak_CP >= 0.75
h20_V_ctrl_LCB > 0.05
h240_longrisk <= 0.05
Y_robust_rate >= 0.30
```

### 可视化

```text
p6_direct_generator_frontier.svg
p6_direct_generator_value_vs_longrisk.svg
p6_cost_vs_value_pareto.svg
```

---

## P7：effect-valid certificate redesign

### 目标

把 certificate 从 construction-valid 转成 effect-valid。上一轮 AUC weak CP 约 0.5015、AUC long-risk 约 0.4984，几乎随机；v9.4.5 必须测试新 certificate 是否真的能分离 value / risk。

### Certificate candidates

```text
CERT6-LinearizedValueTailGuard
CERT7-TrustRegionResidualUCB
CERT8-AdamWConflictHorizonGuard
CERT9-HybridObjectiveCertificate
CERT10-CheapSelectedCertificate
```

### 必须记录

```text
certificate_id
generator_id
action_count
certificate_pass_count
AUC_weak_CP_h20
AUC_Vctrl_positive_h20
AUC_robust_source
AUC_longrisk_h240
PR_lift_robust
PR_lift_longrisk_inverse
P_Yrobust_given_cert_pass
P_Yrobust_given_cert_fail
P_longrisk_given_cert_pass
P_longrisk_given_cert_fail
monotone_sign_pass
calibration_ECE
certificate_cost_ms_q50
certificate_cost_ms_q90
feature_missing_rate
```

### Pass

```text
AUC_robust_source >= 0.70
AUC_longrisk_h240 >= 0.70
P_Yrobust_given_cert_pass >= 0.60
P_longrisk_given_cert_pass <= 0.10
P_Yrobust_given_cert_pass >= 2.0 * P_Yrobust_given_cert_fail
monotone_sign_pass = 1
calibration_ECE <= 0.05
certificate_cost_ms_q90 <= 0.20
```

Weak diagnostic pass：

```text
AUC_robust_source >= 0.65
or AUC_longrisk_h240 >= 0.65
but not sufficient for controller。
```

### 可视化

```text
p7_certificate_roc_pr_curves.svg
p7_certificate_calibration_curve.svg
p7_cert_pass_vs_fail_value_horizon.svg
p7_certificate_cost_pareto.svg
p7_certificate_component_ablation.svg
```

---

## P8：minimal source certificate controller

### 目标

只有当 P5/P6 至少有一个 generator pass 且 P7 certificate pass，才运行 controller。Controller 不允许使用 oracle labels、dataset name 或 validation/test。

### Controller form

$$
Accept(a)=1
\iff
LCB(\widehat{V}_{20}(a))>0
\land
UCB(\widehat{LongRisk}_{240}(a))\le \tau_{long}
\land
LCB(\widehat{Support}(a))\ge \tau_s
\land
Cost(a)\le C_{max}.
$$

### 必须记录

```text
controller_id
generator_id
certificate_id
calibration_split_id
heldout_split_id
thresholds
accepted_count_cal
accepted_count_heldout
coverage_cal
coverage_heldout
h20_weak_CP_cal
h20_weak_CP_heldout
h20_V_ctrl_LCB_cal
h20_V_ctrl_LCB_heldout
h240_longrisk_cal
h240_longrisk_heldout
bad_event_h20_heldout
null_event_h20_heldout
support_balance_pass
family_count
max_family_share
uses_dataset_name
uses_outcome_at_commit
uses_validation_or_test
```

### Pass

```text
uses_dataset_name = 0
uses_outcome_at_commit = 0
uses_validation_or_test = 0
accepted_count_heldout >= min_count_for_coverage
coverage_heldout in [0.03, 0.15]
h20_weak_CP_heldout >= 0.60
h20_V_ctrl_LCB_heldout > 0
h240_longrisk_heldout <= 0.10
bad_event_h20_heldout <= 0.05
support_balance_pass = 1
```

### 可视化

```text
p8_controller_value_risk_coverage_frontier.svg
p8_calibration_to_heldout_drift.svg
p8_controller_support_balance.svg
p8_accepted_action_family_distribution.svg
```

---

## P9：selected source online runtime

### 目标

若 P8 controller pass，测 selected payload runtime。不能只测 no-payload smoke，也不能把 materializer/offline replay 混进 timed path。

### 必须记录

```text
runtime_candidate_id
controller_id
generator_id
certificate_id
runtime_mode
step_count
active_step_count
candidate_count
accepted_count
zero_candidate_controller_kernel_count
zero_candidate_controller_sync_count
feature_compute_time_ms_q90
certificate_compute_time_ms_q90
score_accept_time_ms_q90
payload_lookup_time_ms_q90
payload_apply_time_ms_q90
base_train_step_time_ms_q90
total_step_time_ms_q90
step_ratio_q90
memory_ratio
control_outcome_materializer_in_timed_path
offline_audit_in_timed_path
disk_payload_lookup_in_timed_path
payload_preloaded
```

### Pass

```text
control_outcome_materializer_in_timed_path = 0
offline_audit_in_timed_path = 0
disk_payload_lookup_in_timed_path = 0
zero_candidate_controller_kernel_count = 0
step_ratio_q90 <= 1.50
memory_ratio <= 1.05
payload_apply_error_linf_max <= 1e-7
```

Strong pass：

```text
step_ratio_q90 <= 1.20
payload_apply_time_ms_q90 <= 0.05
```

### 可视化

```text
p9_selected_runtime_waterfall.svg
p9_payload_apply_time_distribution.svg
p9_step_ratio_vs_accept_rate.svg
p9_runtime_cost_vs_value_frontier.svg
```

---

## P10：Base-Acc Sentinel strong baseline continuation

### 目标

继续固定配置 base health 监控，回答“有没有在数据集上训练、acc 如何、和 MLP 比如何”。这不用于 controller，不做 dataset-specific tuning。

### 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2,3,4,5,6,7,8,9
models:
  LQ-t2-h256
  MatchedMLP
  AdamWStrongLRGridMLP
  QuadraticFeatureMLP
optional:
  MLPWideMatchedParams
  MLPStrongAugOffNoTuning
```

所有 hyperparams 在运行前冻结。不可根据 dataset 单独调参。

### 必须记录

```text
dataset
seed
model_id
train_acc
val_acc
test_acc
train_CE
val_CE
test_CE
ECE
NLL
CEp99
margin_p10
step_time_ms_q90
memory_ratio
params
FLOPs_forward
FLOPs_backward
hyperparams_fixed_before_run
dataset_specific_tuning
base_acc_used_for_controller
```

### Pass

```text
sentinel_complete = 1
base_acc_used_for_controller = 0
dataset_specific_tuning = 0
LQ_catastrophic_fail = 0
LQ_minus_AdamWStrongLRGridMLP_mean_test_acc >= -0.02
```

### 可视化

```text
p10_test_acc_by_dataset_model.svg
p10_LQ_minus_baseline_acc_distribution.svg
p10_CE_ECE_NLL_by_model.svg
p10_step_time_memory_by_model.svg
```

### 解释规则

```text
If LQ > MLP:
  只能写 base sentinel healthy；不能写 functional success。

If LQ < StrongMLP:
  说明 base may be weak；但不能调 controller by dataset。

If LQ catastrophic:
  functional route 暂停，先回 base。
```

---

## P11：system gate

### 目标

组合 P8 selected controller 和 P9 selected runtime，判断是否可以打开 official leave-out / paired replay。

### 必须记录

```text
system_candidate_id
controller_id
generator_id
certificate_id
runtime_candidate_id
official_eligible
source_controller_pass
selected_runtime_pass
system_legal_controller_pass
coverage_heldout
h20_weak_CP_heldout
h20_V_ctrl_LCB_heldout
h240_longrisk_heldout
bad_event_h20_heldout
step_ratio_q90
memory_ratio
base_acc_sentinel_pass
uses_dataset_name_for_selector/controller
uses_validation_or_test_for_controller
uses_future_outcome_for_features
uses_outcome_at_commit
diagnostic_promoted_to_official
fake_data_used
proxy_row_used
cpu_offload_used
```

### Pass

```text
official_eligible = 1
source_controller_pass = 1
selected_runtime_pass = 1
system_legal_controller_pass = 1
no fake/proxy/offload = 1
no dataset/validation/test/outcome leakage = 1
```

### 可视化

```text
p11_system_gate_dashboard.svg
p11_contract_audit_matrix.svg
p11_route_decision_ladder.svg
```

---

## P12：leave-dataset-out / leave-stratum-out

### 目标

只有 P11 pass 才打开。验证 controller 不是 dataset 或 stratum artifact。

### Leave-dataset-out

```text
train/calibrate on MNIST + Fashion-MNIST, evaluate KMNIST
train/calibrate on MNIST + KMNIST, evaluate Fashion-MNIST
train/calibrate on Fashion-MNIST + KMNIST, evaluate MNIST
```

### Leave-stratum-out

```text
hold out family bucket
hold out score bucket
hold out step bucket
hold out horizon-risk bucket
hold out payload norm bucket
```

### 必须记录

```text
split_type
heldout_name
controller_id
generator_id
certificate_id
coverage
h20_weak_CP
h20_V_ctrl_LCB
h240_longrisk
bad_event_h20
null_event_h20
support_balance_pass
step_ratio_q90
memory_ratio
```

### Pass

```text
macro h20_weak_CP >= 0.60
macro h20_V_ctrl_LCB > 0
macro h240_longrisk <= 0.10
no single leave-dataset split has h240_longrisk > 0.15
no single leave-dataset split has V_ctrl_LCB <= -0.05
coverage in [0.03,0.15] for macro and at least 2/3 datasets
```

### 可视化

```text
p12_leave_dataset_metrics.svg
p12_leave_stratum_heatmap.svg
p12_controller_drift_by_split.svg
```

---

## P13：official paired replay

### 目标

只有 P11 和 P12 pass 后打开。验证 functional source action 真正打过 AdamWParallel / bestLR / NoOp / Random / shuffled controls。

### Branches

```text
RealGeneratedFunctional
AdamWParallel
AdamWOnly
bestLR
NoOp
Random
ShuffledPayload
ShuffledCertificate
CertificatePassNoPayload
```

### 必须记录

```text
action_id
controller_id
generator_id
certificate_id
branch
horizon
CE_delta
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
test_acc_delta_diagnostic_only
V_ctrl
beats_AdamWParallel
beats_bestLR
beats_NoOp
beats_Random
beats_ShuffledPayload
```

### Pass

```text
RealGenerated beats AdamWParallel rate >= 0.60
RealGenerated beats bestLR rate >= 0.55
RealGenerated beats NoOp rate >= 0.70
RealGenerated beats Random rate >= 0.70
ShuffledPayload does not pass same gate
CertificatePassNoPayload does not pass same gate
V_ctrl_LCB > 0
h240 long-risk <= 0.10
```

### 可视化

```text
p13_paired_replay_branch_delta.svg
p13_real_vs_controls_paired_scatter.svg
p13_shuffled_control_failure.svg
p13_horizon_value_curve.svg
```

---

## P14：short/full functional training boundary

### 目标

只有 P13 pass 才打开。回答真正的 task-level functional question：DG-KAN functional update 是否在固定配置、无数据集调参下对 training trajectory 有实质帮助。

### Runs

```text
short-run: 240 steps
mid-run: 1000 steps
full-run: fixed full budget
seeds: at least 0..4 initially, 0..9 if pass
models:
  LQ-t2-h256 base AdamW
  LQ-t2-h256 + selected functional controller
  MatchedMLP
  AdamWStrongLRGridMLP
  QuadraticFeatureMLP
```

### 必须记录

```text
dataset
seed
model_id
controller_id
train_acc
val_acc
test_acc
train_CE
val_CE
test_CE
ECE
NLL
CEp99
margin_p10
ValLossAUC_step
ValLossAUC_time
time_to_target
steps_to_target
step_time_ms_q90
memory_ratio
functional_accept_rate
functional_payload_apply_count
functional_bad_event_rate
functional_longrisk_rate
```

### Pass

Short-run diagnostic pass：

```text
DG-KAN functional ValLossAUC_step better than LQ base by >= 2%
no ECE/NLL degradation > 5%
step_ratio_q90 <= 1.50
```

Full functional pass：

```text
DG-KAN functional mean test acc >= LQ base + statistically meaningful margin
or DG-KAN functional improves sample efficiency/time-to-target by >= 10%
while ECE/NLL/CEp99 not worse than base
and beats MatchedMLP / StrongLRGridMLP under fixed protocol or explains non-beat via sample-efficiency/robustness advantage.
```

### 可视化

```text
p14_training_curves_acc_loss.svg
p14_val_loss_auc_by_model.svg
p14_time_to_target_by_model.svg
p14_calibration_robustness_by_model.svg
p14_functional_event_timeline.svg
```

---

# 8. 并行执行计划

v9.4.5 必须减少“一轮一个 blocker”的串行感。推荐并行批次如下。

## Batch A：立即并行，目标是 1 小时内暴露 row/materializer/blocker

```text
A1: P0 v9.4.4 boundary reproduction
A2: P1 source objective decomposition v2
A3: P2 oracle survivor anatomy
A4: P10 Base-Acc Sentinel continuation
A5: P3 generator single-action preflight for AP0r-AP0w
```

Batch A 的失败不应阻塞 Base-Acc Sentinel。若 A5 有任何 0-row / missing hash / apply error，立即停止对应 generator，不等完整 runner。

## Batch B：generator smoke 并行

```text
B1: AP0r oracle-seeded preservation smoke
B2: AP0s oracle-seeded preservation smoke
B3: AP0t oracle-seeded preservation smoke
B4: AP0u oracle-seeded preservation smoke
B5: AP0v oracle-seeded preservation smoke
B6: AP0w no-transform diagnostic
B7: AP0r/AP0t/AP0v direct smoke
```

每个 smoke 先 8 actions，过后自动 scale 到 64 actions。未过 preflight 的 generator 不进入 scale。

## Batch C：certificate / legal selector 并行

```text
C1: CERT6-CERT10 on oracle-seeded generated actions
C2: CERT6-CERT10 on legal-seeded generated actions
C3: legal feature capacity re-audit using robust labels
C4: certificate component ablation
```

## Batch D：controller/runtime conditional

只有当 P5/P6 和 P7 pass 后启动：

```text
D1: P8 minimal source certificate controller
D2: P9 selected runtime
D3: P11 system gate
```

## Batch E：downstream conditional

只有当 P11 pass 后启动：

```text
E1: P12 LDO/LSO
E2: P13 official paired replay
E3: P14 short/full functional training
```

---

# 9. Route decision table

## R0-SystemPass

```text
P8 source controller pass = 1
P9 selected runtime pass = 1
P11 system_legal_controller_pass = 1
```

下一步：LDO/LSO + paired replay。

## R1-OracleSourceIdentityBug

```text
AP0w no-transform oracle-seeded fails unexpectedly
or oracle survivor cannot be reproduced
```

下一步：回到 source identity / outcome join audit。

## R2-TransformStillDestructive

```text
AP0w oracle-seeded pass
but AP0r-AP0v all fail preservation
```

下一步：停止 transform route；优先 legal selector or raw source controller。

## R3-SelectorOpaqueButGeneratorCanPreserve

```text
oracle-seeded preservation pass
legal-seeded fail
direct fail
```

下一步：source selector / legal observability redesign。

## R4-DirectGeneratorObjectiveFail

```text
oracle-seeded preservation pass
direct generator fail
```

下一步：direct objective solver redesign；不要调 certificate。

## R5-CertificateEffectFail

```text
generator pass
certificate AUC / lift / monotone fail
```

下一步：certificate redesign；不要调 controller threshold。

## R6-RuntimeFail

```text
source controller pass
selected runtime step_ratio_q90 > 1.50
```

下一步：payload apply / certificate compute / scheduler optimization。

## R7-BaseCatastrophic

```text
Base-Acc Sentinel catastrophic fail
```

下一步：暂停 functional route，回到 LQ base / optimizer / manual training audit。

---

# 10. 关键停止条件

v9.4.5 明确禁止继续以下小修：

```text
1. AP0d threshold 微调；
2. AP0l trust-region 半径小幅扫描但不改 objective；
3. certificate threshold 微调；
4. weak CP-only controller；
5. legal feature 单变量 topK 继续换名字；
6. oracle survivor 直接 official；
7. Base-Acc Sentinel 当作 functional success；
8. 没有 selected controller 就测 official runtime；
9. 没有 system pass 就开 paired replay / short-full。
```

停止 transform route 的条件：

```text
AP0s/AP0r/AP0t/AP0u/AP0v 在 oracle-seeded preservation test 全部：
  source_positive_lost_rate > 0.50
  或 Damage_median < -0.20
  或 h240 long-risk > 0.30
```

停止 direct generator route 的条件：

```text
AP0r/AP0t/AP0u/AP0v direct smoke 全部：
  h20 weak CP <= 0.25
  或 h20 V_ctrl LCB <= -0.20
  或 h240 long-risk >= 0.50
```

停止 certificate route 的条件：

```text
所有 CERT6-CERT10：
  AUC_robust_source < 0.60
  且 AUC_longrisk < 0.60
  且 monotone_sign_pass = 0
```

---

# 11. 必须落盘的 artifact

```text
p0_v9440_boundary_reproduction.csv
p1_source_objective_decomposition_v2.csv
p2_oracle_survivor_anatomy.csv
p3_generator_preflight_matrix.csv
p4_oracle_seeded_generator_preservation.csv
p5_legal_representative_seeded_generator.csv
p6_direct_objective_solved_generator.csv
p7_effect_valid_certificate_redesign.csv
p8_minimal_source_certificate_controller.csv
p9_selected_source_online_runtime.csv
p10_base_acc_sentinel_strong_baseline_continuation.csv
p11_system_integration_gate_v9450.csv
p12_leaveout_boundary_v9450.csv
p13_official_paired_replay_v9450.csv
p14_short_full_training_boundary_v9450.csv
route_decision.json
contract_audit.json
provenance_audit.json
failure_table_v9450.csv
run_manifest.json
hash_manifest.json
```

---

# 12. 必须可视化的图

```text
p0_route_ladder_v9430_to_v9450.svg
p1_weakcp_value_longrisk_phase_diagram.svg
p2_oracle_survivor_legal_rank_hist.svg
p3_generator_preflight_matrix.svg
p4_generator_preservation_damage_violin.svg
p4_source_positive_lost_rate_by_generator.svg
p5_generator_by_source_panel_heatmap.svg
p6_direct_generator_value_longrisk_pareto.svg
p7_certificate_roc_pr_calibration.svg
p7_certificate_pass_fail_value_horizon.svg
p8_controller_value_risk_coverage_frontier.svg
p9_selected_runtime_waterfall.svg
p10_base_acc_sentinel_boxplot.svg
p11_system_gate_dashboard.svg
p12_leaveout_heatmap.svg
p13_paired_replay_branch_delta.svg
p14_training_curve_if_opened.svg
```

---

# 13. 最终判断

v9.4.4 之后，项目不是没有进度，而是进度集中在“排除错误路线、收窄真实 blocker”。但用户感觉慢是合理的，因为最近几轮多次在工程闭合与诊断边界之间推进，还没有产生正向 functional capability。

v9.4.5 必须改变推进方式：

```text
从：
  一个大 runner 串行暴露下一个 blocker

改成：
  preflight + generator matrix + certificate matrix + base acc sentinel 并行验证。
```

本轮最关键的科学问题是：

$$
\boxed{
\text{好 source 已经被 oracle 找到；我们能否用 legal objective-solved generator 生成或保留这种好 source？}
}
$$

如果答案是 yes，项目进入 source certificate controller 和 selected runtime。  
如果答案是 no，就不应继续修 AP0b-AP0q，而要承认当前 functional source primitive 设计不足，进入更彻底的 primitive redesign。

