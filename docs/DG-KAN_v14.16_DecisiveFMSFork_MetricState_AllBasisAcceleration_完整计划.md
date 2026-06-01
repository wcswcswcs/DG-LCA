# DG-KAN v14.16：Decisive FMS Causal Exit Test + Metric-State Rebuild + All-Basis Parallel Acceleration 完整计划

> 版本：v14.16 execution plan  
> 生成时间：2026-05-30 Asia/Singapore  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 关键词：functional update 不停止；停止 action/token 搜索；并行 decisive fork；FMS 因果退出测试；population-risk metric-state rebuild；D-CHE / D-FOU / D-RBF / D-WAV all-basis substrate 加速  
> 硬约束：strict FC-PureKAN；no active B-spline budget；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific branch；no validation/test/future/query direction；LineC / CEp99 / NLL / ECE / AUCtime 只能作为 audit / gate，不能生成方向；不新增 action token；不启动 controller；不使用 action bank；不新增 F-CHE8/F-CHE9 或 FMS-M6/M7 这类 token 搜索。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是寻找一个局部好看的 update rule。项目要证明：

$$
\boxed{
\text{PureKAN substrate + functional update}
>
\text{same PureKAN substrate + ordinary AdamW / strong controls}
}
$$

其中 functional update 必须满足：

```text
1. 表达力不打折；
2. forward / backward / step / memory 与 MLP 可比；
3. 收敛路径更健康，AUC-step / AUC-time 不输；
4. 几何更好：train-probe coupling、signal/reservoir/noise、tail、calibration 不坏；
5. 收益必须击败 NoOp / RandomMatched / AdamWParallel / SameActiveFraction /
   SameProjection / GenericOptimizerState / MLP analog controls；
6. 不依赖 label-informed init；
7. 不依赖 CE-tail / LineC / validation / test / future / dataset branch 生成方向。
```

最终 claim 不是：

```text
1. synthetic task 过了；
2. 某个 real-lite row positive；
3. predictor AUC 局部高；
4. MLP generic optimizer positive；
5. substrate eligibility 过了；
6. action-bank oracle 有局部 upper bound。
```

而是：

$$
\boxed{
\text{functional update 本身产生了 control-resistant、可迁移、低成本、几何健康的训练收益。}
}
$$

---

## 0.2 v14.15 当前真实状态

v14.15 已执行：

```text
Line R: provenance / forbidden / no-action-search audit
Line P: predictor robustness fork
Line I: intervention causality fork
Line B: boundary / curriculum role reset
Line F: D-CHE bounded real-lite
Line D: all-basis substrate acceleration
Line M: MLP / generic controls
Line C/Z: audit / route / no-go / next queue
```

v14.15 的最终 route 为：

```text
route = R15-ParallelForkNoCausalValueAllBasisBlocked
minimum_success = S3-DCHESyntheticFMSPass
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

关键结果：

```text
Line P:
  best_feature = P2-recovery_lag
  best_auc_mean = 0.9090909091
  best_leaveout_min = 0.5
  best_spearman_source = 0.0342218002
  line_p_exploration_gate_pass = 0
  route = R-P-PredictorWeak

Line I:
  line_i_rows = 270
  mean_direction_incremental_gain = -0.017652423
  control_equivalent_fraction = 0.9466666667
  bad_event_fraction = 0.5022222222
  real_lite_pass_count = 0
  route = R-I-DirectionControlEquivalent

Line B:
  best_boundary = BND3-DegreeEnergyLimiter
  best_source_vs_best_control = -0.0230504837
  min_control_equivalent_fraction = 0.7777777778
  route = R-B-BoundaryIsHarmlessNull

Line F:
  line_f_real_lite_pass_count = 0 / 9
  source_vs_best_control_mean = -0.0324013222
  control_equivalent_fraction = 0.9259259259
  route = R-F-RealLiteBelow4

Line D:
  best_non_dche_family = D-FOU
  best_non_dche_dataset_seed_pass_count = 2 / 9
  official_fms_eligible_family_count = 0
  route = R-D-SubstrateStillBlocked

Line M:
  required controls = 7
  executed controls = 7
  missing controls = 0
  route = M-ControlsCompleteNoGenericPositive
```

这意味着：v14.15 没有 fail-fast，且没有违反 no-action-search 纪律；但也没有任何能力突破。

---

# 1. 当前各线进展百分比

| 线 | 当前完成度 | 当前判断 |
|---|---:|---|
| 代码 / provenance / finalizer 审计 | 99% | artifact、manifest、forbidden / no-action-search audit 基本闭合 |
| Historical FHQ / B320-current | 85% frozen | 历史强，但 label-informed init 已禁用，不作为 official base claim |
| Line C 几何审计 | 88% | 审计稳定；只能做 gate / failure explanation，不能生成方向 |
| PopRisk / FMS infrastructure | 94% | per-example gradient、persistent state、projection telemetry 成熟 |
| D-CHE substrate | 85% | 当前最强 Non-RAT substrate，9/9 eligibility 历史成立 |
| D-CHE-FMS synthetic | 55%-60% | 历史 best synthetic 5/7；但 real transfer 不可靠 |
| Transfer observability | 25%-30% | predictor 有局部 AUC，但 leaveout / Spearman 不稳 |
| FMS direction causal value | 0%-5% | Line I control-equivalent，高度负面 |
| Boundary / curriculum FMS | 0%-5% | Line B harmless-null，F real-lite 0/9 |
| D-CHE-FMS real transfer | 5%-10% | v14.15 F 线 0/9；历史 real 2/9 / Rational 6/9 不可 promotion |
| Rational substrate | 85% | 稳定，但 reset / optimizer-state route 被 generic confound 打回 |
| Rational-FMS causal specificity | 10%-15% | monitor / no-regression；不作为当前主 carrier |
| D-FOU substrate | 35%-45% | 历史有 6/9；v14.15 best non-D-CHE 仍 2/9 |
| D-RBF / FastKAN substrate | 30%-40% | workspace/telemetry 可审计，但 task-health 未稳 |
| D-WAV substrate | 15%-25% | 弱线索，低预算保留 |
| Non-RAT official FMS proof | 0%-5% | 除 D-CHE 外无 family 进入 proof；D-CHE real-transfer 未成 |
| Generic MLP-FMS / controls | 35%-40% | control 价值高；不能写成 KAN-specific success |
| Official S5 functional success | 0% | 尚未达成 |
| 整体 next-gen MLP claim | 33%-41% | 比 v14.14 更低：并行 fork 证明当前 FMS 方向与 boundary 都没有 causal value |

说明：整体完成度下调，不是因为工程倒退，而是因为 v14.15 给出更强的负证据：当前 FMS direction、boundary、real-lite、non-D-CHE substrate 都没有打开。

---

# 2. 对 v14.15 结果的独立判断

## 2.1 这次不是没跑，而是跑出了硬 no-go

v14.15 最大变化是：它不再串行等待某个 gate，而是并行执行 predictor、intervention、boundary、real-lite、all-basis、controls。这个设计本身是对的。

但是结果很明确：

```text
Line P:
  predictor 有表面 AUC，但 leaveout_min 只有 0.5，Spearman 近似 0；
  所以 predictor 不是稳健 transfer value。

Line I:
  270 rows，mean_direction_incremental_gain 为负；
  control_equivalent_fraction 接近 0.95；
  所以 FMS direction 没有独立 causal value。

Line B:
  best boundary 仍为负 source；
  control-equivalent fraction 仍高；
  boundary / curriculum 不是当前解法。

Line F:
  D-CHE real-lite 0/9；
  source_vs_best_control_mean 为负；
  所以 bounded real-lite 没有打开。

Line D:
  D-FOU / D-RBF / D-WAV 仍未达到 >=6/9；
  all-basis 替代 carrier 没出现。
```

这说明当前慢，不是因为 Codex 没执行，而是因为我们现在的 functional definition 没有 causal value。

---

## 2.2 真正 blocker：不是 observability，而是 intervention

v14.13 / v14.14 / v14.15 共同说明：

```text
能看见一点东西：
  recovery_lag / degree_projection_rejection_fraction 等 feature 有局部 AUC。

但 intervention 不成立：
  FMS direction 与 matched controls 等价；
  boundary harmless-null；
  real-lite 不转化；
  non-D-CHE basis 没打开。
```

所以当前 blocker 是：

$$
\boxed{
\text{observability} \not\Rightarrow \text{causal intervention}
}
$$

更具体：

```text
1. predictor 可以排序一些状态；
2. 但 FMS direction 在这些状态上没有比 matched random / same active fraction /
   same projection rejection / same value retention controls 更强；
3. boundary / abstention / plasticity 也没有转化成 source/AUC/tail/LineC 同位；
4. 所以不能继续把 predictor 当 value source；
5. 也不能继续新增 FMS token 去“碰运气”。
```

---

## 2.3 为什么这不是“再努力一点就行”

v14.15 已经覆盖了以前会诱导我们继续小修的空间：

```text
selector:
  always-on
  E4 high-score
  inverse E4
  random matched active fraction
  NoOp-safe abstention

boundary:
  no boundary
  degree projection safety
  low-amplitude boundary
  rejection fraction cap
  value retention floor

real-lite:
  D-CHE bounded FMS candidates
  matched controls

all-basis:
  D-FOU32..36
  D-RBF30..34
  D-WAV29..32
```

如果继续新增：

```text
FMS-M6
F-CHE8
BND6
K-RT
K-AUC
controller
action bank
reset
```

这会回到 v9 / v12 的旧错误：看到局部信号就继续寻找“好动作”。

v14.16 必须避免这个错误。

---

# 3. 当前路线是否正确？

## 3.1 正确的部分

```text
1. functional update 不应该停止；
2. promotion fail-closed 是对的；
3. exploration continue-open 是对的；
4. no-action-search 纪律必须保留；
5. D-CHE 是当前最强 Non-RAT carrier；
6. all-basis substrate 不能停止；
7. MLP/generic controls 必须保留；
8. LineC/tail/AUC 必须做 audit，但不能做 direction。
```

## 3.2 错误的部分

```text
1. 继续把 FMS 当 direction generator；
2. 继续试图从 transfer predictor 直接生成 update；
3. 继续做 boundary-only token；
4. 继续在 D-CHE 上 real-lite 小修；
5. 继续逐版串行排错；
6. 让 Codex 在单个 gate fail 后停止，或者在 gate fail 后临时扩 token。
```

最重要的判断：

$$
\boxed{
\text{当前 FMS direction-generator 路线应进入 decisive exit test。}
}
$$

不是停止 functional update，而是停止当前这类 FMS direction token。

---

# 4. v14.16 的核心策略：三天内完成 decisive fork

v14.16 不再写成长串 token。它只做四个并行 decisive tests：

```text
Fork A:
  当前 FMS direction generator 是否 no-go？

Fork B:
  FMS 是否应改成 metric-state / optimizer-boundary，而非 direction？

Fork C:
  D-CHE 是否是错误 carrier，是否需要换 D-FOU / D-RBF？

Fork D:
  all-basis substrate 是否能在并行加速下打开替代 carrier？
```

v14.16 的目标不是马上 S5，而是在一轮内做出 stop-go：

```text
1. 关闭 CurrentFMSDirection route；
2. 或证明 Metric-State FMS 有 causal value；
3. 或找到替代 substrate carrier；
4. 或明确 current functional formulation no-go。
```

---

# 5. Line R：审计、硬约束与加速协议

## 5.1 目标

保证本轮是 decisive fork，不是 action/token search。

## 5.2 必须记录

```text
required_artifact_missing_count
forbidden_information_violation_count
no_action_search_violation_count
new_fms_token_count
new_fche_token_count
action_token_count
controller_executed
reset_route_used
dataset_name_branch_used
seed_specific_scale_used
direction_uses_validation_test_future_query
direction_uses_linec_cep99_nll_ece_auctime
promotion_allowed
exploration_continue_open
```

## 5.3 硬停条件

只在以下条件 hard stop：

```text
required_artifact_missing_count > 0
forbidden_information_violation_count > 0
no_action_search_violation_count > 0
direction_uses_validation_test_future_query = 1
direction_uses_linec_cep99_nll_ece_auctime = 1
dataset_name_branch_used = 1
seed_specific_scale_used = 1
new_fms_token_count > 0
new_fche_token_count > 0
action_token_count > 0
controller_executed = 1
reset_route_used = 1
```

## 5.4 非硬停条件

这些不能让 Codex 停整轮：

```text
Line A fail
Line B fail
Line C fail
Line D fail
MLP/generic controls positive
D-FOU / D-RBF <6/9
D-CHE real-lite <4/9
LineC/tail/AUC fail
```

它们只写 route / failure taxonomy / next hypothesis。

---

# 6. Line A：Current FMS Direction Exit Test

## 6.1 目标

直接回答：

$$
\boxed{
\text{当前 FMS direction 是否有独立 causal value？}
}
$$

不是看 predictor AUC，不是看 real-lite row，而是看同 selector / 同 norm / 同 projection / 同 boundary 下，FMS direction 是否优于 matched controls。

## 6.2 实验设计

固定 selector / boundary / norm / projection：

```text
selector:
  S0 always-on
  S1 predictor-high-score
  S4 NoOp-safe abstention

boundary:
  B0 none
  B1 degree projection safety
  B4 value-retention floor

direction:
  D0 AdamW
  D1 current FMS
  D2 random matched norm
  D3 same-active-fraction random
  D4 same-projection-rejection random
  D5 same-value-retention random
  D6 AdamWParallelDirection control
```

这不是新 method token；这是 causal dissection matrix。

## 6.3 核心估计量

$$
\Delta_{FMS}
=
Outcome(S,D1,B)
-
\max_{c\in \{D2,D3,D4,D5,D6\}} Outcome(S,c,B).
$$

记录：

```text
source_vs_best_control
AUCtime_ratio
CEp99_delta
NLL_delta
ECE_delta
LineC_pass
control_equivalent
bad_event
step_time_ratio
memory_ratio
```

## 6.4 判定标准

Current FMS direction 继续资格：

```text
mean(source_vs_best_control) > 0.005
control_equivalent_fraction <= 0.40
bad_event_fraction <= 0.30
real_lite_pass_count >= 3/9
```

若不满足：

```text
route = R-A-CurrentFMSDirectionNoGo
```

并且：

```text
no more current FMS direction methods
no FMS-M6/M7
no F-CHE8/F-CHE9
```

---

# 7. Line B：Metric-State FMS Rebuild

## 7.1 核心判断

如果 FMS direction no-go，functional update 不应停止，而应从：

```text
FMS as direction generator
```

改成：

```text
FMS as persistent metric-state / training boundary
```

这与 population-risk paper 的思想更一致：每步不是造一个新方向，而是用 drift-vs-diffusion 估计训练信号质量，改变 optimizer metric / plasticity。

## 7.2 数学定义

对参数组或 basis role $r$，定义：

$$
\mu_r=\frac{1}{b}\sum_i g_{i,r},
$$

$$
\Sigma_r=\frac{1}{b}\sum_i (g_{i,r}-\mu_r)(g_{i,r}-\mu_r)^T.
$$

定义 population-risk utility：

$$
U_r=\mu_r^T M_r^{-1}\mu_r-\frac{1}{b-1}\operatorname{tr}(M_r^{-1}\Sigma_r).
$$

维护 persistent state：

$$
c_{r,t+1}=\beta c_{r,t}+(1-\beta)\operatorname{clip}(U_{r,t},u_{min},u_{max}).
$$

更新不是新方向，而是 metric scaling：

$$
\Delta\theta_t
=
-\eta_t T(c_t) M_{Adam,t}^{-1/2} m_t.
$$

其中 $T(c_t)$ 只能调 plasticity / trust region / abstention，不直接生成新的方向。

## 7.3 预注册 variants

只允许三类，不允许扩 token：

```text
MS1-ParameterMetricState
MS2-DegreeRoleMetricState
MS3-BasisGroupMetricState
```

每类只允许：

```text
beta = fixed from plan
clip = fixed from plan
update_interval = fixed from plan
```

不扫网格。

## 7.4 Controls

```text
C0 AdamW
C1 SameT(c)RandomDirection
C2 SameActiveFraction
C3 SameMetricScaleRandomPermutation
C4 MLP-MetricState
C5 NoOpMatchedOverhead
```

## 7.5 判定

探索成功：

```text
real_lite_pass_count >= 3/9
source_vs_best_control_mean > 0
control_equivalent_fraction <= 0.60
```

强探索成功：

```text
real_lite_pass_count >= 4/9
source_vs_best_control_mean >= 0.005
control_equivalent_fraction <= 0.50
```

若 MS1/MS2/MS3 全 fail：

```text
route = R-B-MetricStateFMSNoGo
```

---

# 8. Line C：Transfer Observability Deconfounding

## 8.1 目标

v14.15 的 Line P 有 AUC 高但 leaveout 低的问题。Line C 要判断 predictor 是否有真实可用性，还是只是在 artifact split 上过拟合。

## 8.2 必跑 leaveout

```text
leave-dataset-out
leave-seed-out
leave-loss-interface-out
leave-method-out
leave-control-out
leave-synthetic-family-out
leave-time-budget-out
```

## 8.3 指标

```text
AUC_mean
AUC_leaveout_min
AUC_leaveout_median
Spearman_source
precision_at_top10
recall_at_top10
false_positive_rate_on_controls
incremental_auc_over_controls
calibration_error_of_predictor
```

## 8.4 判定

Predictor 只允许作为 exploration gate，当且仅当：

$$
AUC_{leaveout,min}\ge0.60
$$

且：

$$
Spearman_{source}\ge0.20.
$$

如果不满足：

```text
predictor 不允许 gating / abstention / boundary。
```

注意：predictor 不允许直接生成 direction。

---

# 9. Line D：D-CHE Real-Lite Mechanism Verification

## 9.1 目标

只用 Line A/B/C 中通过的机制进入 D-CHE real-lite，不再单独造 D-CHE FMS method。

## 9.2 输入

```text
from Line A:
  Current FMS direction only if not no-go

from Line B:
  MS1/MS2/MS3 if exploration success

from Line C:
  predictor only if leaveout robust
```

## 9.3 评价

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
baseline = D-CHE-AdamW
controls = SameActiveFraction / SameMetricScale / RandomMatched / MLP analog
```

## 9.4 Gate

Weak exploration：

```text
real_lite_pass_count >= 3/9
```

Meaningful exploration：

```text
real_lite_pass_count >= 4/9
```

S4 exploration：

```text
real_lite_pass_count >= 6/9
```

S5 promotion：

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
controls fail
```

---

# 10. Line E：All-Basis Substrate Acceleration

## 10.1 目标

D-CHE 不能成为唯一 carrier。v14.15 已证明 D-FOU / D-RBF / D-WAV 仍 blocked，但 v14.9 也曾有 D-FOU / D-RBF 6/9 历史信号。Line E 要用并行预算重新确认：

```text
D-FOU 是否能稳定回到 >=6/9；
D-RBF/FastKAN 是否能稳定回到 >=6/9；
D-WAV 是否仍应低预算保留；
D-CHE 是否保持 no-regression。
```

## 10.2 D-FOU

候选不新增 token，只用预注册机制族：

```text
FOU-A low-frequency identity residual
FOU-B bandwise SNR warmup
FOU-C phase-stable band mix
FOU-D no-materialize lifetime
FOU-E high-frequency quarantine
```

记录：

```text
band_energy_low
band_energy_mid
band_energy_high
phase_drift
high_freq_ratio
bandwise_snr
step_ratio
memory_ratio
LineC_pass_rate
mean_delta
worst_delta
NLL_ratio
```

## 10.3 D-RBF / FastKAN

```text
RBF-A active center occupancy
RBF-B width condition guard
RBF-C compact bump no-dense materialization
RBF-D identity residual
RBF-E Gaussian local K4 task-health
```

记录：

```text
center_occupancy_entropy
empty_center_fraction
width_condition_p99
out_of_grid_fraction
residual_over_base
active_center_fraction
step_ratio
memory_ratio
LineC_pass_rate
mean_delta
worst_delta
NLL_ratio
```

## 10.4 D-WAV

低预算保留：

```text
WAV-A triangular support
WAV-B scale occupancy
WAV-C support overlap damping
WAV-D local-tail coverage audit
```

## 10.5 Gate

Exploration substrate：

```text
family_dataset_seed_pass_count >= 6/9
```

Official FMS eligibility：

```text
family_dataset_seed_pass_count = 9/9
```

Non-D-CHE 未到 6/9 时，不允许 official FMS proof，但也不能 hard stop整轮。

---

# 11. Line M：MLP / Generic Controls

## 11.1 目标

所有 positive-looking KAN/FMS result 必须排除 generic optimizer explanation。

## 11.2 必跑 controls

```text
MLP-AdamW
MLP-MetricState-MS1
MLP-MetricState-MS2
MLP-SameActiveFraction
MLP-SameMetricScaleRandomPermutation
MLP-RandomMatchedNorm
MLP-NoOpMatchedOverhead
GenericOptimizerStateControl
AdamWParallelDirectionControl
```

## 11.3 判断

如果：

```text
MLP/generic control >= KAN-FMS result
```

则 route：

```text
R-M-GenericControlExplainsGain
```

不能 claim KAN-specific functional success。

---

# 12. 加速执行策略

v14.16 必须并行运行，不允许再串行等待一个 gate。

## 12.1 并行组

```text
GPU group 0:
  Line A current FMS exit test + Line M controls

GPU group 1:
  Line B metric-state FMS rebuild

GPU group 2:
  Line C transfer observability deconfounding

GPU group 3:
  Line E all-basis substrate acceleration

GPU group 4, if available:
  Line D D-CHE real-lite verification for passed Line A/B mechanisms
```

## 12.2 最低完成要求

```text
Line A:
  full selector-direction-boundary contrast table

Line B:
  MS1/MS2/MS3 + controls

Line C:
  complete leaveout table

Line D:
  at least D-CHE real-lite for every mechanism passing Line A/B

Line E:
  D-FOU and D-RBF full substrate rows;
  D-WAV low-budget monitor;
  D-CHE no-regression

Line M:
  controls for every positive-looking result

Line Z:
  route / no-go / failure taxonomy / next hypothesis
```

## 12.3 Codex 失败后的继续方向

如果 Line A fails：

```text
record CurrentFMSDirectionNoGo；
do not add direction tokens；
continue Line B / E / M。
```

如果 Line B fails：

```text
record MetricStateFMSNoGo；
do not tune beta/clip/grid；
continue all-basis substrate；
prepare FMS definition exit report。
```

如果 Line C fails：

```text
record TransferPredictorNotRobust；
predictor cannot be used for gating；
continue Line B/E。
```

如果 Line D fails：

```text
record D-CHERealLiteNoGo under current mechanisms；
continue Line E；
do not add F-CHE tokens。
```

如果 Line E fails:

```text
record AllBasisCarrierStillBlocked；
do not enter Non-D-CHE FMS proof；
continue no-regression monitor。
```

If all A/B/C/D/E fail:

```text
route = R16-CurrentFMSDefinitionNoCausalValue
next hypothesis:
  either redefine functional update at optimizer/theory level,
  or return to substrate/base architecture as main bottleneck.
```

---

# 13. 必须生成的 artifacts

```text
v1416_route_decision.json
v1416_progress_table.csv
v1416_required_artifact_manifest.csv
v1416_forbidden_information_audit.csv
v1416_no_action_search_audit.csv

Line A:
  v1416_a_current_fms_exit_matrix.csv
  v1416_a_direction_control_equivalence.csv
  v1416_a_failure_taxonomy.csv

Line B:
  v1416_b_metric_state_results.csv
  v1416_b_metric_state_controls.csv
  v1416_b_population_risk_state_trace.csv

Line C:
  v1416_c_transfer_predictor_leaveout.csv
  v1416_c_predictor_control_deconfound.csv

Line D:
  v1416_d_dche_real_lite.csv
  v1416_d_dche_controls.csv

Line E:
  v1416_e_fou_substrate.csv
  v1416_e_rbf_substrate.csv
  v1416_e_wav_monitor.csv
  v1416_e_dche_no_regression.csv
  v1416_e_allbasis_summary.csv

Line M:
  v1416_m_mlp_generic_controls.csv

Line C audit:
  v1416_linec_tail_auc_audit.csv

Line Z:
  v1416_no_go_boundary.md
  v1416_next_hypothesis_queue.md
  v1416_code_review_packet.zip
```

---

# 14. 必须可视化

```text
fig_a_direction_vs_controls_heatmap.svg
fig_a_control_equivalent_fraction.svg
fig_b_metric_state_trace.svg
fig_b_metric_state_vs_controls.svg
fig_c_leaveout_auc_matrix.svg
fig_c_predictor_calibration.svg
fig_d_dche_real_lite_pass_matrix.svg
fig_e_allbasis_pass_heatmap.svg
fig_e_fou_rbf_task_workspace_pareto.svg
fig_m_mlp_vs_kan_control_gap.svg
fig_z_route_dashboard.svg
```

---

# 15. 成功 / 失败路线

## 15.1 最小成功

```text
S0-ParallelForkExecuted:
  all lines R/A/B/C/D/E/M/Z executed with no missing artifact.
```

## 15.2 探索成功

```text
S1-FMSDirectionCausalValue:
  Line A passes current FMS direction gate.

S2-MetricStateFMSPositive:
  Line B reaches >=3/9 real-lite or source/control positive with control_equivalent <=0.60.

S3-TransferPredictorRobust:
  Line C leaveout_min >=0.60 and Spearman >=0.20.

S4-AllBasisAlternativeCarrier:
  any non-D-CHE family reaches >=6/9 substrate exploration.

S4-DCHENewRealLitePositive:
  D-CHE real-lite >=4/9 under a mechanism that passes Line A or B.
```

## 15.3 Official S5

```text
S5-OfficialFunctionalSuccess:
  real_dataset_seed_pass_count = 9/9
  source_vs_best_control >= 0.005
  AUCtime_ratio <= 1.0
  CEp99_delta <= 0.05
  NLL_delta <= 0.02
  ECE_delta <= 0.02
  LineC_pass = 1
  step_time_ratio <= 1.25
  memory_ratio <= 1.25
  controls fail
  forbidden audit pass
  no-action-search audit pass
  code review pass
  promotion_allowed = 1
```

## 15.4 No-go routes

```text
R-A-CurrentFMSDirectionNoGo:
  Line A proves direction control-equivalent.

R-B-MetricStateFMSNoGo:
  MS1/MS2/MS3 fail matched controls.

R-C-TransferPredictorNotRobust:
  predictor AUC high only in non-robust splits.

R-D-DCHENoRealLiteTransfer:
  D-CHE remains <4/9 under all passing mechanisms.

R-E-AllBasisCarrierBlocked:
  D-FOU / D-RBF / D-WAV all <6/9.

R16-CurrentFMSDefinitionNoCausalValue:
  A/B/C/D/E all fail; current FMS definition should be exited.
```

---

# 16. 最终执行判断

v14.16 的目标不是“再试一个方法”，而是用并行 decisive tests 回答：

$$
\boxed{
\text{FMS 是 direction 有用，metric-state 有用，predictor 有用，boundary 有用，}
\text{还是全部只是 control-equivalent?}
}
$$

如果答案是“全部 control-equivalent”，就必须停止当前 FMS 定义，不再用下一个 token 拖延。

如果 Metric-State 或 all-basis carrier 出现 positive，再进入下一轮 focused confirmation。

这才是加速。
