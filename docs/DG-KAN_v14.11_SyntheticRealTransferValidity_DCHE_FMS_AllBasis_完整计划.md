# DG-KAN v14.11：Synthetic-to-Real Transfer Validity + D-CHE FMS Definition Reset + All-Basis Parallel Substrate 完整计划

生成时间：2026-05-30（Asia/Singapore）

> 本计划基于 v14.10 真实结果复盘。v14.10 的最新合法状态是：D-CHE substrate eligibility 仍成立，best synthetic 已达到 `S3-DCHESyntheticFMSPass`，`synthetic_task_family_pass_count = 5/7`；但 best real 只有 `2/9`，route 为 `R2-DCHESyntheticDoesNotTransfer`，`official_s5_reached = 0`，`promotion_allowed = 0`。因此 v14.11 的核心不是继续加 F-CHE token，而是验证 synthetic S3 到底是否是 real-transfer 的有效前置门，并重新定义 train-stream-only real-transfer FMS value。

---

# 0. 项目总目标与当前进展

## 0.1 总目标

DG-KAN 的项目总目标仍然是：

$$
\boxed{
\text{在 strict FC-PureKAN substrate/base 上，利用 functional update 获得比普通 AdamW/backprop 更好的训练过程、几何和模型。}
}
$$

这里的“更好”不能只等于某个局部 source row 或某个 synthetic task pass，而必须同时满足：

```text
1. 表达力不打折；
2. forward / backward / step / memory 与 MLP 可比；
3. 收敛轨迹不慢于 MLP / AdamW controls；
4. 几何健康：train-probe coupling、signal/reservoir/noise、tail、calibration 不坏；
5. functional update 的收益必须独立于 AdamWParallel、NoOp、RandomMatched、generic MLP-FMS / generic optimizer controls；
6. functional direction 不能使用 validation/test/future/query，也不能使用 CEp99/NLL/ECE/LineC/AUCtime audit metric 生成方向；
7. 不能按 dataset / seed 分支，不能把局部 positive row 写成 success。
```

最终 claim 形式必须是：

$$
\boxed{
\text{PureKAN substrate + functional update}
>
\text{same PureKAN substrate + AdamW / matched controls}
}
$$

而不是：

```text
D-CHE substrate eligibility = success；
synthetic S3 = real success；
local source positive = functional success；
MLP/generic FMS positive = KAN-specific success。
```

## 0.2 v14.10 当前真实进展

v14.10 证明了三件事。

第一，D-CHE 已经是当前最强的 Non-RAT substrate。v14.10 读取 v14.9 Line D substrate artifact 后确认：

```text
D-CHE dataset-seed pass = 9/9
D-CHE official FMS eligible = 1
D-CHE best candidate = D-CHE24-HighDegreeLateEnable
best mean delta = +0.033203
best LineC = 1.000000
min NLL ratio = 0.477970
median step ratio = 0.251653
```

第二，D-CHE-FMS official synthetic 初始失败，但修正 candidate mismatch、streaming per-example gradient memory、300-step all-FCHE/FB 后，best synthetic 达到：

```text
route = S3-DCHESyntheticFMSPass
synthetic_task_family_pass_count = 5/7
synthetic_task_pass = {X1:1, X2:0, X3:1, X4:1, X5:1, X6:0, X7:1}
```

第三，synthetic S3 没有 transfer 到 real 3x3：

```text
route = R2-DCHESyntheticDoesNotTransfer
real_dataset_seed_pass_count = 2/9
official_s5_reached = 0
promotion_allowed = 0
```

已经尝试过：

```text
1. v14.9 D-CHE substrate candidate 对齐；
2. streaming per-example gradient memory repair；
3. D-CHE17 synthetic all-FCHE/FB 300-step S3；
4. real 3x3 200-step / 400-step transfer；
5. Case B: F-CHE-RT1 / RT2 / RT3；
6. post-plan F-CHE-RT4 composite trust；
7. RT4 strength sensitivity。
```

这些没有打开 S4/S5。v14.10 不能继续在本轮内新增 F-CHE8、action token、controller、reset route，也不能用 real dataset/seed fail pattern 或 CEp99/ECE/LineC/AUCtime audit 指标反推方向。

## 0.3 v14.11 的核心定位

v14.11 不再问：

```text
再加哪个 F-CHE 变体？
再调哪个 strength / interval / trust？
怎么把 2/9 变成 9/9？
```

v14.11 只问三个更根本的问题：

$$
\boxed{
Q1: \text{synthetic S3 是否真的预测 real transfer？}
}
$$

$$
\boxed{
Q2: \text{是否存在 legal train-stream-only real-transfer value，能在 commit 前预测/约束 real-transfer 风险？}
}
$$

$$
\boxed{
Q3: \text{D-CHE-FMS 的失败是 FMS definition 错，还是 D-CHE substrate 只适合 synthetic、不适合 real functional transfer？}
}
$$

如果 Q1/Q2 不成立，不能继续扩 F-CHE token；必须重定义 functional value 或退回 substrate/base 设计。

---

# 1. 各线当前进度百分比

这些百分比是基于 gate 通过情况、artifact 完整性、机制清晰度和距离 official success 的综合估计。

| 线 | 当前完成度 | 当前判断 |
|---|---:|---|
| 代码 / provenance / finalizer 审计 | **99%** | artifact surface、required manifest、forbidden audit 均强；不是 blocker |
| Historical FHQ / B320-current | **85% frozen** | 历史强，但 label-informed init 禁用，不能 official |
| Line C 几何审计 | **88%** | 审计稳定；v14.10 中 LineC 不是主 blocker |
| PopRisk / FMS infrastructure | **94%** | persistent state、per-example gradient、streaming memory、projection telemetry 成熟 |
| Generic MLP-FMS / MLP control | **35%-40%** | 作为 confound/control 有价值；不能写成 KAN success |
| Rational substrate | **85%** | 仍稳定，但 optimizer-state/reset route 已被 generic confound 打回 |
| Rational-FMS causal specificity | **10%-15%** | v14.9 R3 generic confound 后主线降级 |
| D-CHE substrate | **85%** | 9/9 substrate，当前最强 Non-RAT substrate |
| D-CHE-FMS synthetic | **55%-60%** | best synthetic 5/7，S3 已打开；但不等于 real success |
| D-CHE-FMS real transfer | **10%-15%** | best real 2/9，S4 未打开 |
| Synthetic-to-real validity | **0%-10%** | v14.11 新主线；目前只知道 S3 不充分 |
| Train-stream real-transfer value | **0%-5%** | v14.11 新主线；尚未验证 |
| D-FOU substrate | **55%-60%** | 6/9 exploration，值得 hardening 到 9/9 |
| D-RBF / FastKAN substrate | **50%-55%** | 6/9 exploration，但 task-health 仍不稳 |
| D-WAV substrate | **20%-25%** | 2/9，低预算保留 |
| Non-RAT official FMS proof | **0%-5%** | D-CHE 已 eligible，但 FMS real 未成；FOU/RBF/WAV 不能 official proof |
| Official S5 functional success | **0%** | 尚未达成 |
| 整体 next-gen MLP claim | **40%-48%** | D-CHE substrate 与 synthetic S3 是进展，但 real-transfer 失败使 claim 仍远 |

---

# 2. v14.10 独立分析

## 2.1 有进展吗？

有。v14.10 至少有两个真实进展。

第一，D-CHE 从“Non-RAT 仍不确定”推进到“official FMS eligible substrate”。这改变了项目格局：之前 functional 基本绑定 Rational，现在 D-CHE 也成为一个可测试 functional update 的严格 substrate。

第二，D-CHE-FMS 在修正 candidate mismatch、streaming per-example gradient memory、300-step all-FCHE/FB 后，best synthetic 达到 5/7。说明 D-CHE 上并不是完全没有 FMS signal。

但这两个进展都不能 promotion。原因是：real 3x3 transfer 只有 2/9，S4/S5 都没打开。

## 2.2 为什么仍然感觉慢？

因为项目最终目标不是 synthetic S3，而是 real transfer 和 official S5。v14.10 暴露了一个非常重要的问题：

$$
\boxed{
\text{synthetic S3 并不保证 real 3x3 transfer。}
}
$$

这比“又没过”更严重。它说明我们现在的 synthetic proof gate 可能不够贴近 real training dynamics。继续把 synthetic 5/7 当作唯一 real-entry 信号，可能会反复制造“synthetic positive -> real fail”的循环。

## 2.3 当前真正卡在哪里？

v14.10 的 blocker 不是 D-CHE substrate，也不是 LineC，也不是 step/memory。

当前 blocker 是：

$$
\boxed{
\text{D-CHE-FMS 的 synthetic causal value 没有 real-transfer validity。}
}
$$

拆开看有三层：

```text
1. Synthetic gate validity blocker:
   X1/X3/X4/X5/X7 pass 不能预测 real 3x3 pass。

2. Real-transfer value blocker:
   train-stream FMS state 当前没有同时控制 source、AUCtime、CEp99/NLL/ECE tail。

3. Confound blocker:
   MLP/generic source 上界多次高于 D-CHE source；必须保留 generic control，不能把 generic FMS 写成 D-CHE-specific。
```

## 2.4 不能继续做什么？

v14.11 明确禁止以下路线：

```text
1. 继续新增 F-CHE8 / F-CHE9；
2. 继续 strength / interval / trust 小网格；
3. 启动 controller；
4. 使用 real dataset/seed fail pattern 设计 direction；
5. 使用 CEp99/ECE/NLL/LineC/AUCtime audit metric 生成方向；
6. 把 synthetic S3 或 real 2/9 写成 success；
7. 把 D-CHE substrate eligibility 写成 FMS success；
8. 把 MLP/generic FMS positive 写成 KAN-specific success。
```

这也是为了避免重犯 v9 / v12 的旧错：局部 positive 后不断寻找“更好动作”。v14.11 不做动作搜索。

---

# 3. v14.11 核心假设

## H1：Synthetic S3 当前不是充分的 real-transfer gate

v14.10 已经出现：

```text
best synthetic = 5/7
best real = 2/9
```

因此需要验证：

$$
\boxed{
\text{当前 synthetic task-family battery 与 real transfer 是否存在稳定预测关系？}
}
$$

如果没有，下一步不能继续用 synthetic 5/7 作为 real-entry gate，必须改 gate 或增加 real-like train-stream precondition。

## H2：D-CHE-FMS 失败不是 degree safety 不足，而是 real-transfer value 不足

F-CHE1..F-CHE7、FB1..FB3、RT1..RT4 已经覆盖 degree-wise value、degree constraint、projection retention、train split agreement、degree energy safety 等方向。它们没能让 real transfer 超过 2/9。

因此下一步不能继续 degree-FMS token，而要重新定义：

$$
\boxed{
\text{train-stream-only real-transfer value。}
}
$$

## H3：D-CHE 是有效 substrate，但 FMS 可能需要两层边界

D-CHE 的 substrate 能力强，说明它有训练/表达/LineC 基础；FMS 失败说明它不能直接把 generic FMS value 变成 real transfer。可能需要：

```text
1. value path：generic population-risk / train-stream loss cotangent；
2. transfer path：train split agreement / micro-horizon loss integral / tail proxy / margin stability；
3. basis safety path：degree-energy / high-degree late-enable / readout-degree decoupling。
```

但这三者不能变成任意 token 搜索，必须先做 validity audit。

## H4：D-FOU / D-RBF 的 6/9 exploration 是并行机会，但不能抢 D-CHE FMS 主线

D-FOU 和 D-RBF 已到 6/9 exploration，应继续 hardening 到 9/9；但未到 9/9 前，不能进入 official FMS proof。

---

# 4. v14.11 实验总览

v14.11 分为七条线。

```text
Line R:
  implementation / provenance / no-action-search audit。

Line E:
  synthetic-to-real transfer validity audit。

Line V:
  train-stream real-transfer value validity。

Line F-CHE:
  D-CHE FMS definition reset，只在 Line E/V 过后执行。

Line D:
  all-basis substrate hardening：D-FOU / D-RBF / D-WAV / Rational monitor。

Line M:
  MLP / generic FMS confound control。

Line Z:
  final route / no-go boundary / next hypothesis queue。
```

---

# 5. Line R：Implementation / Provenance / No-Action-Search Audit

## 5.1 目标

确保 v14.11 不是 v14.10 的 token extension，也不是 action-search 复发。

## 5.2 必须检查

```text
1. no new F-CHE8/F-CHE9 token before Line E/V validity；
2. no controller；
3. no dataset-name branch；
4. no seed-specific scaling；
5. no validation/test/future/query in direction；
6. no CEp99/NLL/ECE/LineC/AUCtime audit metric as direction source；
7. MLP/generic controls marked non-promotable；
8. D-CHE substrate eligibility not counted as FMS success；
9. synthetic S3 not counted as real success；
10. real 2/9 not counted as S4/S5。
```

## 5.3 Required artifacts

```text
v1411_code_review_manifest.csv
v1411_no_action_search_audit.csv
v1411_forbidden_information_audit.csv
v1411_required_artifact_manifest.csv
v1411_plan_compliance_readback.md
```

## 5.4 Failure route

如果发现任何 action-search violation：

```text
R0-ActionSearchOrAuditDirectionViolation
```

并停止。

---

# 6. Line E：Synthetic-to-Real Transfer Validity Audit

## 6.1 目标

回答：

$$
\boxed{
\text{当前 synthetic S3 是否对 real transfer 有预测力？}
}
$$

这不是为了从 real failure 反推 direction，而是为了判断 synthetic gate 是否有效。

## 6.2 数据来源

读取已完成 artifacts：

```text
1. v14.3 Rational synthetic S3 + real 4/9；
2. v14.4 Rational real 6/9；
3. v14.5 action-bank oracle diagnostics；
4. v14.9 D-CHE substrate status；
5. v14.10 D-CHE synthetic 5/7 + real 2/9；
6. MLP/generic FMS controls。
```

## 6.3 记录字段

每个 method / task / seed / loss 记录：

```text
family
substrate_candidate
method
synthetic_task
synthetic_seed
loss_interface
synthetic_source_vs_control
synthetic_AUCtime
synthetic_CEp99_delta
synthetic_NLL_delta
synthetic_ECE_delta
synthetic_LineC
synthetic_pass
method_family_pass_count
real_dataset
real_seed
real_source_vs_control
real_AUCtime
real_CEp99_delta
real_NLL_delta
real_ECE_delta
real_LineC
real_pass
matched_method_id
is_promotable_candidate
```

## 6.4 分析

计算：

$$
\operatorname{corr}(synthetic\_source, real\_source)
$$

$$
\operatorname{corr}(synthetic\_AUCtime, real\_AUCtime)
$$

$$
\operatorname{AUC}_{predict}(synthetic\_features \rightarrow real\_pass)
$$

并做 leave-one-family / leave-one-dataset audit：

```text
1. train on Rational, test on D-CHE；
2. train on D-CHE synthetic, test on D-CHE real；
3. train on MNIST/Fashion, test on KMNIST；
4. CE-only vs Brier-only split。
```

## 6.5 Gate

Synthetic-to-real validity pass 需要：

$$
AUC_{predict} \ge 0.70
$$

并且：

$$
\operatorname{Spearman}(synthetic\_source, real\_source) \ge 0.30
$$

至少在 D-CHE internal split 上成立。

## 6.6 Failure route

如果不成立：

```text
R1-SyntheticGateNotPredictiveForRealTransfer
```

此时不允许继续 D-CHE real FMS training。下一步必须重建 real-transfer synthetic battery 或直接改成 train-stream transfer value gate。

---

# 7. Line V：Train-Stream Real-Transfer Value Validity

## 7.1 目标

回答：

$$
\boxed{
\text{是否存在一个只使用 train-stream 的 value proxy，能预测 FMS real-transfer 风险？}
}
$$

这一步不能使用 validation/test/future，也不能使用 LineC/CEp99/NLL/ECE/AUCtime audit 作为方向源。

## 7.2 候选 value proxy

在每个 FMS commit 前，从 train batch 拆两个 split：

```text
B1 = update / value-fit split
B2 = counterfactual validation split, train-stream only
```

允许使用：

```text
current train loss
per-example train loss
per-example gradient SNR
train split agreement
train margin p10
train logit RMS
train entropy
train q90/q95 loss
projection value retention
degree energy telemetry
FMS state drift
```

不允许使用：

```text
validation/test/future/query
LineC hard target
CEp99/NLL/ECE/AUCtime audit
Brier audit gate as direction
real dataset/seed branch
```

## 7.3 Value definition

定义 train-stream transfer utility：

$$
U_{train}
=
\Delta L_{B2}^{proxy}
-
\lambda_q \Delta q95Loss_{B2}
-
\lambda_m \max(0,-\Delta margin_{p10,B2})
-
\lambda_r \Delta RMSDrift_{B2}
-
\lambda_h \Delta EntropyCollapse_{B2}
$$

其中 $\Delta L_{B2}^{proxy}$ 是 FMS candidate 相对 NoOp / AdamW step 的 train-split loss improvement。这里不是选择 action bank；只评估一个预注册 FMS definition 是否在 train-stream 上有一致 value。

## 7.4 Controls

必须比较：

```text
NoOpMatchedOverhead
AdamWParallelDirectionControl
RandomMatchedNorm
SameActiveFractionRandomMask
MLP/generic FMS control
D-CHE AdamW baseline
```

## 7.5 Gate

Train-stream value validity pass 需要：

$$
\operatorname{AUC}(U_{train} \rightarrow real\_pass) \ge 0.70
$$

或者在没有新增 real run 的情况下，至少：

$$
\operatorname{AUC}(U_{train} \rightarrow synthetic\_pass) \ge 0.75
$$

并且 generic control 不能完全解释：

$$
U_{D-CHE} - U_{MLP/generic} > 0
$$

## 7.6 Failure route

如果 train-stream value 无法预测 synthetic/real pass：

```text
R2-TrainStreamTransferValueUnobservable
```

此时禁止继续 F-CHE definition variants。

---

# 8. Line F-CHE：D-CHE FMS Definition Reset

## 8.1 进入条件

只有满足下面条件才执行：

```text
Line R pass；
Line E synthetic-to-real validity pass 或 Line V train-stream value validity pass；
D-CHE substrate eligibility = 1；
no generic_fms_confound hard fail。
```

如果 Line E/V 都不通过，不进入 Line F-CHE。

## 8.2 不允许的东西

```text
1. 不允许 F-CHE8 token extension；
2. 不允许 action bank；
3. 不允许 controller；
4. 不允许 dataset/seed branch；
5. 不允许用 LineC/tail/AUC audit 生成 direction；
6. 不允许 strength/interval 小网格；
7. 不允许从 real fail pattern 反推 rule。
```

## 8.3 允许的机制：FMS definition，不是 action

v14.11 只允许三类机制，每一类都必须预注册，不可根据 row-level positive 扩展。

### FMS-D1：Generic value + degree constraint as safety only

形式：

$$
\Delta\theta_{FMS}
=
\Pi_{degree-safe}(\Delta\theta_{generic-FMS})
$$

degree telemetry 只做 constraint，不做 value source。

记录：

```text
value_retention_after_projection
projection_rejection_fraction
cos_projected_vs_generic
degree_energy_before/after
high_degree_fraction
```

### FMS-D2：Train-stream transfer utility gated FMS

形式：

$$
commit = 1[U_{train} > \tau_U]\cdot 1[value\_retention > \tau_R]
$$

这是 accept/reject 单一 FMS definition，不是从多个动作里选。

### FMS-D3：Continuous low-amplitude degree-phase FMS

形式：

$$
\theta_{t+1}
=
\theta_t
-
\eta_t T(c_t) P_{degree-safe} M_{Adam}^{-1/2}m_t
$$

其中 $T(c_t)$ 是 persistent population-risk utility state，不是 one-shot event。

## 8.4 Synthetic gate

D-CHE synthetic S3 gate：

```text
>= 5/7 synthetic task families pass；
within passed family, >=2/3 seeds or >=2 loss interfaces pass where applicable；
source_vs_best_control >= 0.005；
AUCtime_ratio <= 1.0；
CEp99/NLL/ECE non-harm；
LineC pass；
step/memory pass；
generic_fms_confound = 0。
```

## 8.5 Real gate

只有 synthetic S3 + Line V validity pass 后才打开 real 3x3。

Exploration S4：

```text
real_dataset_seed_pass_count >= 6/9
```

Official S5：

```text
real_dataset_seed_pass_count = 9/9
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.25
promotion_allowed = 1 only after audit/code review
```

## 8.6 Failure route

```text
R3-DCHESyntheticPassRealTransferFail:
  synthetic S3 pass but real <6/9。

R4-DCHENoTrainStreamValue:
  FMS cannot pass train-stream value validity。

R5-DCHEGenericFMSConfound:
  MLP/generic FMS explains D-CHE source/value。

R6-DCHEFMSDefinitionNoGo:
  FMS-D1/D2/D3 all fail synthetic S3.
```

---

# 9. Line D：All-Basis Substrate Parallel

## 9.1 目标

继续推进所有 active non-BSpline basis，但不让它们在未成 substrate 前进入 official FMS proof。

## 9.2 D-FOU：从 6/9 推向 9/9

当前：

```text
D-FOU substrate = 6/9
best = D-FOU26-NoMaterializeLifetimeAuditV2
exploration = 1
official FMS eligible = 0
```

v14.11 允许尝试：

```text
D-FOU27-LowFreqIdentityResidualV2
D-FOU28-BandwiseSNRSafeWarmup
D-FOU29-PhaseStableBandMixNoHighFreq
D-FOU30-NoMaterializeLifetimeV3
```

Gate：

```text
full 3x3 substrate >=6/9 for exploration；
full 3x3 substrate =9/9 for official FMS eligibility；
LineC pass rate >=0.30；
NLL ratio <=2；
step ratio <=2.5；
mean_delta_vs_MLP >=-0.05。
```

## 9.3 D-RBF / FastKAN：从 6/9 推向 9/9

当前：

```text
D-RBF substrate = 6/9
best = D-RBF25-WidthConditionGuardNoTaskBranch
```

v14.11 允许尝试：

```text
D-RBF26-ActiveCenterOccupancyV2
D-RBF27-WidthConditionIdentityResidual
D-RBF28-CompactBumpNoDenseMaterialization
D-RBF29-GaussianLocalK4TaskHealth
```

重点指标：

```text
center_occupancy_entropy
empty_center_fraction
width_condition_p99
out_of_grid_fraction
identity_residual_norm
task-health pass
```

## 9.4 D-WAV：低预算保留

当前：

```text
D-WAV substrate = 2/9
```

只允许低预算机制验证：

```text
D-WAV25-TriangularSupportV3
D-WAV26-ScaleOccupancyNoTailTarget
D-WAV27-LocalSupportOverlapDamping
```

如果仍 <6/9，则不进入 FMS proof。

## 9.5 Rational

Rational 不继续 reset / optimizer-state route。只做 no-regression monitor：

```text
RAT substrate monitor
RAT-FMS historical S3/S4b reference
no new reset/action/controller
```

---

# 10. Line M：MLP / Generic Confound Control

## 10.1 目标

判断 FMS signal 是否 generic，而不是 D-CHE / KAN-specific。

## 10.2 必跑 controls

```text
M0-MLP-AdamW
M1-MLP-GenericFMS
M2-MLP-TrainStreamTransferUtilityFMS
M3-MLP-ProjectionMatchedControl
M4-MLP-NoOpMatchedOverhead
```

## 10.3 判断

如果：

$$
MLP\_FMS - MLP\_AdamW \ge D\text{-}CHE\_FMS - D\text{-}CHE\_AdamW,
$$

则不能 claim D-CHE-specific functional leverage。

如果 D-CHE-FMS 成功且 MLP-FMS 不成功，才允许讨论 KAN-specific / Chebyshev-specific leverage。

---

# 11. Line C：Geometry / Tail / Calibration Audit

Line C 只做审计，不生成方向。

必须记录：

```text
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
CEp99
NLL
ECE
Brier
margin_p10
AUCtime
LineC multisketch pass
source_vs_best_control
```

必须明确写入：

```text
direction_uses_linec = 0
direction_uses_cep99_nll_ece_auc = 0
```

---

# 12. 必须生成的 artifacts

```text
v1411_progress_table.csv
v1411_code_review_manifest.csv
v1411_no_action_search_audit.csv
v1411_forbidden_information_audit.csv
v1411_synthetic_real_alignment.csv
v1411_synthetic_real_predictivity_summary.csv
v1411_train_stream_value_proxy.csv
v1411_train_stream_value_validity.csv
v1411_dche_fms_definition_results.csv
v1411_dche_real_transfer_results.csv
v1411_all_basis_substrate_status.csv
v1411_fourier_substrate_hardening.csv
v1411_rbf_substrate_hardening.csv
v1411_wavelet_substrate_monitor.csv
v1411_mlp_generic_controls.csv
v1411_linec_tail_audit.csv
v1411_failure_taxonomy.csv
v1411_route_decision.json
v1411_no_go_boundary.md
v1411_next_hypothesis_queue.md
v1411_required_artifact_manifest.csv
v1411_code_review_packet.zip
```

---

# 13. 必须可视化

```text
fig_v1411_progress_by_line.svg
fig_synthetic_to_real_predictivity.svg
fig_synthetic_task_family_vs_real_pass_heatmap.svg
fig_train_stream_value_vs_real_pass.svg
fig_dche_fms_synthetic_family_heatmap.svg
fig_dche_real_transfer_3x3_matrix.svg
fig_source_auc_tail_failure_decomposition.svg
fig_all_basis_substrate_matrix.svg
fig_mlp_vs_dche_fms_comparison.svg
fig_no_action_search_audit.svg
```

---

# 14. Stop / Go 决策

## Case A：Synthetic-to-real validity 失败

Route：

```text
R1-SyntheticGateNotPredictiveForRealTransfer
```

结论：

```text
不能继续 F-CHE/FMS variant；
必须重建 synthetic battery 或转向 train-stream value validity。
```

## Case B：Train-stream value 不可观测

Route：

```text
R2-TrainStreamTransferValueUnobservable
```

结论：

```text
不能打开 real 3x3；
不能用 audit metric 反推；
需要新的 value definition。
```

## Case C：D-CHE synthetic S3 通过但 real <6/9

Route：

```text
R3-DCHESyntheticPassRealTransferFail
```

结论：

```text
记录 source/AUC/tail/LineC failure taxonomy；
不能按 dataset/seed 调参；
下一步必须重建 real-transfer gate。
```

## Case D：D-CHE real >=6/9 but <9/9

Route：

```text
S4-DCHERelTransferExplorationPositive
```

结论：

```text
不能 promotion；
只允许机制级 missing-state decomposition；
不能 action search。
```

## Case E：D-CHE real 9/9

Route：

```text
S5-DCHENonRATFunctionalSuccess
```

需要：

```text
all strict gates pass；
generic controls fail；
MLP analog does not explain；
code review pass；
provenance pass；
promotion_allowed = 1。
```

---

# 15. Codex 执行要求

Codex 必须按顺序执行：

```text
1. Line R audit；
2. Line E synthetic-to-real validity；
3. Line V train-stream value validity；
4. 若 E/V 过，执行 Line F-CHE；否则 fail-closed；
5. Line D all-basis substrate hardening；
6. Line M MLP generic control；
7. Line C audit；
8. finalizer / no-go / next queue。
```

Codex 禁止：

```text
1. 新增 F-CHE8 或 action token；
2. 启动 controller；
3. strength / interval / trust 小网格；
4. 按 dataset/seed branch；
5. 使用 CEp99/NLL/ECE/LineC/AUCtime audit metric 生成 direction；
6. synthetic S3 未过时打开 real；
7. real 2/9 写成 success；
8. D-CHE substrate eligibility 写成 FMS success；
9. D-FOU/D-RBF 6/9 exploration 写成 official eligibility；
10. 把 MLP/generic positive 写成 KAN-specific。
```

---

# 16. 最终总结

v14.10 的价值不是“D-CHE 已经成功”，而是它给出了一个清晰的新边界：

$$
\boxed{
\text{D-CHE 是强 substrate，D-CHE-FMS 能 synthetic 5/7，}
\text{但 synthetic S3 无法 transfer 到 real 3x3。}
}
$$

因此 v14.11 不能继续小修 F-CHE，也不能继续动作搜索。下一步必须先验证 synthetic gate 的 real-transfer validity，再定义合法 train-stream real-transfer value。如果这两件事不成立，继续跑 D-CHE-FMS 只会重复“synthetic positive / real fail”的循环。

