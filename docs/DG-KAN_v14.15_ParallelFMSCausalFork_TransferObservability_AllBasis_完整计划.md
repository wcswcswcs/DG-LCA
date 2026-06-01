# DG-KAN v14.15：Parallel FMS Causal Fork + Transfer Observability + All-Basis Substrate 加速计划

> 版本：v14.15 execution plan  
> 基于：v14.14 `FMS CausalValue TransferBoundary AllBasisContinueOpen` 真实结果复盘，以及 v9 / v12 / v14 系列反复出现的 action-search 教训。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 核心修正：promotion 必须 fail-closed；exploration 必须 continue-open；但 exploration 不能退化为“继续找好动作”。本计划把下一步从串行 token 迭代改成并行的机制分叉实验，用最少轮数判断 FMS 是否还有因果价值、D-CHE 是否仍是 functional carrier、以及其他 basis 是否能尽快进入 substrate gate。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是只找到一个能跑的 KAN basis。项目要证明：

$$
\boxed{
\text{strict FC-PureKAN base} + \text{functional update}
>
\text{same base} + \text{ordinary backprop / AdamW controls}
}
$$

并且必须同时满足：

```text
1. 表达力不打折；
2. forward / backward / step / memory 与 MLP 可比；
3. 收敛轨迹更健康，AUC-time 不差；
4. 几何更好：signal channel 更干净、noise 不进 signal channel、reservoir 不困住真实信号；
5. functional update 的收益必须独立于 NoOp / RandomMatched / AdamWParallel / generic optimizer-state controls；
6. functional direction 不得使用 validation / test / future / query，不得使用 LineC / CEp99 / NLL / ECE / AUCtime audit 指标生成方向；
7. 不按 dataset / seed 调参，不做 teacher / distillation / loss modification / sampler / class weight。
```

## 0.2 当前状态一句话

v14.14 的真实状态是：

$$
\boxed{
\text{有 transfer observability 线索，但 FMS causal value 仍 control-equivalent；}
\text{D-CHE substrate 仍是主 carrier，但 real-transfer 没打开。}
}
$$

v14.14 的关键事实：

```text
route = R4-AllBasisSubstrateBlocked
minimum_success = S3-DCHESyntheticFMSPass
official_s5_reached = 0
promotion_allowed = 0

E4 best AUC mean = 0.9091，但 leaveout AUC min = 0.5，Spearman = 0.0342。
Q source_vs_best_control_mean = -0.0326。
Q control_equivalent_fraction = 0.8889。
Boundary real-lite = 0/9。
Boundary control_equivalent_fraction = 0.9259。
Line D best non-D-CHE = 2/9。
Line D official FMS eligible family count = 0。
```

这不是能力成功。它说明：

```text
1. E4 能看见某些 state，但不稳健；
2. 当前 FMS-M / boundary FMS 不能利用这个 observability；
3. 当前 FMS effect 大多被 matched controls 解释；
4. D-CHE 外其它 basis 还没达到 functional proof 资格；
5. 继续扩 FMS-M6/M7、F-CHE8/F-CHE9、action/controller/reset route 都是在重复旧错误。
```

## 0.3 当前各线进展百分数

| 线 | 当前完成度 | 相比 v14.13 | 判断 |
|---|---:|---:|---|
| 代码 / provenance / finalizer 审计 | 99% | 0 | 工程闭包强，不是当前 blocker。 |
| Historical FHQ / B320-current | 85% frozen | 0 | 历史强 anchor，但 label-informed init 禁用 official claim。 |
| Line C 几何审计 | 88% | 0 | 审计稳定，只能做 gate / failure explanation，不能生成方向。 |
| PopRisk / FMS infrastructure | 94% | 0 | per-example gradient、persistent state、projection telemetry 成熟。 |
| D-CHE substrate | 85% | 0 | 当前最强 Non-RAT substrate，9/9 substrate eligibility。 |
| D-CHE-FMS synthetic | 55%-60% | 0 | 历史 best synthetic 5/7，S3 已打开，但不预测 real。 |
| Transfer observability | 30%-35% | +5 | E4 AUC mean 高，但 leaveout min 只有 0.5，robustness 不够。 |
| FMS causal realization | 0%-5% | 0 | Q/V3 主断点仍是 ControlEquivalent。 |
| Boundary-style FMS | 0%-5% | 0 | F-B1/B2/B3 0/9，且 control-equivalent 更高。 |
| D-CHE-FMS real transfer | 5%-10% | -5 | v14.14 boundary real-lite 0/9；历史 real 2/9 不足。 |
| Rational substrate | 85% | 0 | 稳定，但 reset / optimizer-state route 已被 generic confound 打回。 |
| Rational-FMS causal specificity | 10%-15% | 0 | monitor / no-regression，不作为当前主 carrier。 |
| D-FOU substrate | 35%-45% | 0 | 历史 6/9 信号未稳定复现；当前 best 2/9。 |
| D-RBF / FastKAN substrate | 30%-40% | 0 | workspace 有进展，task-health 不稳。 |
| D-WAV substrate | 15%-25% | 0 | 弱线索，低预算保留。 |
| Non-RAT official FMS proof | 0%-5% | 0 | 除 D-CHE 外无 family 进入 proof；D-CHE real transfer 未成。 |
| Official S5 functional success | 0% | 0 | 尚未达成。 |
| 整体 next-gen MLP claim | 35%-43% | -3 | 诊断更清楚，但当前 FMS causal value 仍未成立。 |

---

# 1. v14.14 独立分析：有进展吗？

## 1.1 有诊断进展，但没有能力进展

v14.14 的进展不是模型变强，而是定位更准：

```text
E4 找到一个高 AUC mean 的 transfer feature；
Q 证明当前 FMS effect 仍被 matched controls 解释；
B/F 证明 boundary-only 不是当前解法；
Line D 证明 D-CHE 外 basis 仍没有进入 FMS proof 资格。
```

因此当前不能写：

```text
functional update 成功；
D-CHE real transfer 成功；
boundary FMS 成功；
all-basis substrate 成功。
```

只能写：

```text
transfer observability 有弱/局部线索；
当前 FMS direction / boundary realization 未形成独立 causal value；
下一步必须并行做 causal fork，而不是继续串行新增 token。
```

## 1.2 当前真正问题不是“没看到 signal”，而是“signal 不能造成独立 effect”

E4 的 `best_auc_mean = 0.9091` 很诱人，但它的 `leaveout_auc_min = 0.5` 和 `Spearman = 0.0342` 说明这个 feature 的稳健性不足。更关键的是，Q 直接显示：

$$
source\_vs\_best\_control\_mean < 0,
$$

并且：

$$
control\_equivalent\_fraction \approx 0.889.
$$

这意味着：

```text
观测量可能能区分某些状态；
但当前 FMS update 并没有比 matched controls 更好地利用这些状态；
functional update 还没有 causal value。
```

## 1.3 当前不能继续串行 v14.x 小修

v14.14 已经覆盖：

```text
Line R / E4 / Q / B/F / D / C / Z；
F-B1/F-B2/F-B3 boundary definitions；
C0..C5 和 G4/G5 matched controls；
D-FOU26..31、D-RBF25..29、D-WAV25..28 substrate candidates。
```

继续做：

```text
FMS-M6/M7；
F-CHE8/F-CHE9；
B4/B5/B6；
action token / controller / action bank；
reset route；
strength/lambda/interval grid；
基于 real fail pattern 的分支；
使用 audit metric 反推 direction；
```

都属于重蹈 v9/v12 的旧错。

---

# 2. 本质问题：FMS 不是缺一个更好动作，而是缺 causal intervention

## 2.1 过去反复犯过的错误

必须把 v9 / v12 的教训写入硬约束：

```text
1. actuatability pass 不等于 causality；
2. oracle frontier 不等于 legal observability；
3. pre-event observability 不闭合时不能做 controller；
4. harmless-null / control-equivalent 不能算 safe-good；
5. local positive row 不能驱动 action/token expansion；
6. dataset/seed failure slice 只能诊断，不能生成 branch；
7. runtime/provenance/control outcome 未闭合不能 promotion。
```

v14.14 的 `ControlEquivalent` 正是这个旧坑的现代版本：

$$
\boxed{
\text{看起来有 movement，但 movement 没有独立于 controls 的 value。}
}
$$

## 2.2 FMS 需要从 direction generator 重新拆成三层

当前 FMS 混了三件事：

```text
1. State predictor：什么时候可能有 transfer value？
2. Direction generator：往哪里更新参数？
3. Boundary / trust region：哪些更新不能做，哪些方向要压低？
```

v14.14 的结果说明：

```text
State predictor 有一些信号；
Direction generator 没有 causal value；
Boundary-only 也没有转化为 value。
```

因此 v14.15 不再问“哪个 FMS method 过”，而是并行回答：

```text
Q1: predictor 是否真的稳健？
Q2: direction 是否真的有 causal value？
Q3: boundary 是否只能防坏，不能创造 value？
Q4: 若 direction 无效，FMS 是否应退化为 boundary-state / curriculum-state？
Q5: D-CHE 之外是否有更适合承载 FMS 的 substrate？
```

---

# 3. v14.15 总体策略：并行 decisive fork，不再串行小修

v14.15 不再采用：

```text
E4 -> Q -> B -> F -> D -> 下一版
```

这种串行推进。它改成并行的五条 decisive fork：

```text
Line P: Predictor robustness fork
Line I: Intervention causality fork
Line B: Boundary-only / curriculum-state fork
Line F: D-CHE bounded real-lite causal fork
Line D: All-basis substrate acceleration fork
```

每条线都有独立 route。任何一条 fail 都不能 hard stop 整轮，除非出现 forbidden / action-search / artifact 缺失。

---

# 4. Line P：Predictor robustness fork

## 4.1 目标

判断 E4 看到的 transfer feature 是真实、稳健、可泛化的观测量，还是特定 split / artifact 的偶然相关。

## 4.2 假设

### H-P1：E4 feature 只是 split-specific

如果 E4 AUC mean 高，但 leaveout / seed / loss / dataset 外推低，则不能作为 FMS gate。

### H-P2：E4 feature 只在 controls 上也有效

如果 controls 同样有高 AUC，则它不是 FMS-specific observability，只是 generic train-state predictor。

### H-P3：E4 feature 可作为 abstention / risk flag，但不能作为 value signal

如果 E4 能预测 bad event，但不能预测 source advantage，应转成 safety boundary，而不是 value direction。

## 4.3 实验设计

对以下 feature family 做 robustness audit：

```text
P1-degree_projection_rejection_fraction
P2-recovery_lag
P3-value_retention_after_degree_projection
P4-cos_projected_vs_generic
P5-drift_diffusion_group_utility
P6-split_window_consistency
P7-micro_horizon_loss_integral
P8-update_state_disagreement
P9-degree_energy_stability
```

每个 feature 在以下 split 上报告：

```text
leave-dataset-out
leave-seed-out
leave-loss-interface-out
leave-method-out
leave-control-out
leave-synthetic-family-out
```

## 4.4 必须记录指标

```text
feature_id
auc_mean
auc_median
auc_leaveout_min
auc_leaveout_std
spearman_source
spearman_auc
spearman_tail
precision_at_top10
recall_at_top10
false_positive_rate_on_controls
incremental_auc_over_controls
calibration_ece_of_predictor
coverage_at_threshold
```

## 4.5 Gate

探索 gate：

$$
AUC_{mean} \ge 0.65,
$$

$$
AUC_{leaveout,min} \ge 0.55,
$$

$$
incremental\_auc\_over\_controls \ge 0.05.
$$

Promotion-enabling gate：

$$
AUC_{mean} \ge 0.75,
$$

$$
AUC_{leaveout,min} \ge 0.65,
$$

$$
Spearman_{source} \ge 0.30,
$$

$$
incremental\_auc\_over\_controls \ge 0.10.
$$

如果 Line P fail，不允许 hard stop；进入 Line I/B/F 继续诊断，但 route 标记为：

```text
R-P-PredictorNotRobustButExplorationContinues
```

## 4.6 可视化

```text
fig_p_predictor_auc_leaveout_heatmap.svg
fig_p_feature_vs_source_scatter.svg
fig_p_incremental_auc_over_controls.svg
fig_p_precision_recall_topk.svg
fig_p_calibration_curve.svg
```

---

# 5. Line I：Intervention causality fork

## 5.1 目标

把 predictor、direction、boundary 三件事正交拆开，判断 FMS direction 是否有独立 causal value。

## 5.2 核心 factorial design

构造三轴 factorial：

```text
Selector axis:
  S0 = always-on
  S1 = E4 predictor high-score
  S2 = inverse E4 predictor
  S3 = random matched active fraction
  S4 = NoOp-safe abstention selector

Direction axis:
  D0 = AdamW direction
  D1 = Generic FMS direction
  D2 = Degree-projected FMS direction
  D3 = Random matched-norm direction
  D4 = AdamWParallelDirection control
  D5 = zero direction / NoOp

Boundary axis:
  B0 = no boundary
  B1 = degree projection safety
  B2 = low-amplitude boundary
  B3 = rejection-fraction cap
  B4 = value-retention floor
```

这不是 action bank；它是一次性 causal decomposition matrix。不能后续继续扩 S/D/B token。

## 5.3 关键对照

必须报告：

```text
FMS direction vs random matched direction under same selector/boundary
E4 selector vs random selector under same direction/boundary
boundary vs no boundary under same selector/direction
NoOp vs FMS under same overhead
AdamWParallel control vs FMS
```

## 5.4 核心 causal estimand

$$
\Delta_{direction}
=
Outcome(S,D1,B)-Outcome(S,D3,B).
$$

$$
\Delta_{selector}
=
Outcome(S1,D,B)-Outcome(S3,D,B).
$$

$$
\Delta_{boundary}
=
Outcome(S,D,B1)-Outcome(S,D,B0).
$$

FMS direction 有 causal value 必须满足：

$$
\Delta_{direction} > 0.
$$

如果只有 selector 有效，而 direction 无效，则 FMS 只能做 gate / abstention，不是 direction generator。

## 5.5 必须记录指标

```text
selector_id
direction_id
boundary_id
source_vs_best_control
auc_time_ratio
CEp99_delta
NLL_delta
ECE_delta
LineC_pass
control_equivalent
bad_event
NoOp_equivalent
random_direction_equivalent
selector_incremental_gain
direction_incremental_gain
boundary_incremental_gain
step_time_ratio
memory_ratio
```

## 5.6 Gate

探索 gate：

```text
mean(direction_incremental_gain) >= 0.002
control_equivalent_fraction <= 0.60
bad_event_fraction <= 0.30
real_lite_pass_count >= 3/9
```

Meaningful gate：

```text
mean(direction_incremental_gain) >= 0.005
control_equivalent_fraction <= 0.40
bad_event_fraction <= 0.20
real_lite_pass_count >= 4/9
```

S4 exploration：

```text
real_lite_pass_count >= 6/9
controls cannot explain
step_time_ratio <= 1.50 exploratory
```

S5 仍严格：

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
```

## 5.7 可视化

```text
fig_i_factorial_direction_selector_boundary_heatmap.svg
fig_i_direction_gain_vs_random.svg
fig_i_control_equivalent_fraction.svg
fig_i_bad_event_breakdown.svg
fig_i_real_lite_pass_matrix.svg
```

---

# 6. Line B：Boundary-only / curriculum-state fork

## 6.1 为什么需要 Line B

v14.14 的 boundary FMS 0/9 不能证明 boundary 思路错，只证明当前 F-B1/B2/B3 无效。更关键的是：如果 Line I 显示 direction 无 causal value，但 selector/boundary 有安全价值，FMS 可能应被重定义为 training boundary / curriculum state，而不是 direction generator。

这条线不是新增 action，而是检验 **FMS role reset**：

```text
FMS-as-direction  ->  FMS-as-boundary
```

## 6.2 预注册 boundary roles

```text
BND1-AbstentionOnly:
  FMS 只决定何时不更新 / 降低 update，不生成方向。

BND2-PlasticityScheduler:
  FMS 只调 role-wise plasticity，不改变方向。

BND3-DegreeEnergyLimiter:
  FMS 只限制 D-CHE high-degree energy growth。

BND4-RecoveryLagSuppressor:
  若 train-stream recovery lag 高，则延迟或降低 FMS projection。

BND5-DriftDiffusionTrustRegion:
  用 PopRisk drift-diffusion 设 trust-region 半径，不改 direction。
```

## 6.3 必须记录指标

```text
boundary_id
abstention_rate
plasticity_mean
plasticity_min
plasticity_max
degree_energy_growth
high_degree_fraction_delta
recovery_lag_delta
drift_diffusion_ratio
source_vs_best_control
AUCtime_ratio
tail_delta
LineC_pass
control_equivalent_fraction
NoOp_equivalent_fraction
```

## 6.4 Gate

Boundary 不能只“不坏”，必须改善 trajectory 或减少 bad event：

```text
bad_event_fraction decrease >= 0.20 vs FMS-direction
source_vs_best_control_mean >= -0.002
AUCtime_ratio_mean improves vs FMS-direction
control_equivalent_fraction <= 0.60
```

若 boundary 只与 NoOp 等价，route：

```text
R-B-BoundaryIsHarmlessNull
```

---

# 7. Line F：D-CHE bounded real-lite causal fork

## 7.1 目标

D-CHE 仍是最强 Non-RAT carrier。Line F 不新增 F-CHE8/F-CHE9，不做 action search，只把 Line P/I/B 中通过 exploration gate 的机制组合拿到 D-CHE real-lite 上做 bounded confirmation。

## 7.2 候选来源

只允许三类候选进入 Line F：

```text
1. Line I 中 direction_incremental_gain 过探索 gate的组合；
2. Line B 中 bad_event_fraction 明显下降且非 NoOp-equivalent 的 boundary；
3. Line P 中 predictor robust 且用于 abstention / schedule 的 feature。
```

不允许直接新增方法。

## 7.3 真实 3x3 real-lite 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_steps = 200 or 400, fixed global schedule
batch_size = fixed global setting
loss = CE, Brier if pre-registered
no dataset/seed branch
```

## 7.4 必须记录

```text
real_lite_dataset_seed_pass_count
source_vs_best_control_mean
source_vs_best_control_min
AUCtime_ratio_mean
AUCtime_ratio_max
CEp99_delta_max
NLL_delta_max
ECE_delta_max
LineC_fail_count
control_equivalent_fraction
step_time_ratio_mean
memory_ratio_mean
failure_taxonomy
```

## 7.5 Gate

```text
Exploration weak: real_lite_pass_count >= 3/9
Exploration meaningful: real_lite_pass_count >= 4/9
S4 exploration: real_lite_pass_count >= 6/9
S5 official: real_dataset_seed_pass_count = 9/9 with all strict gates
```

Line F fail 不得 hard stop；必须输出 failure taxonomy 并继续 Line D/M/C/Z。

---

# 8. Line D：All-basis substrate acceleration fork

## 8.1 目标

避免 D-CHE 单点依赖，同时加速 D-FOU / D-RBF / D-WAV substrate 线。v14.14 说明 D-CHE 外 best 只有 2/9，不能 official FMS proof。但这不应导致全线停止。

## 8.2 D-FOU 方向

当前目标：从 2/9 或历史 6/9 信号重新推进到稳定 >=6/9 exploration substrate。

预注册候选：

```text
D-FOU32-LowFreqIdentityResidualV3
D-FOU33-BandwiseSNRWarmupV2
D-FOU34-PhaseStableBandMixNoHighFreqV2
D-FOU35-NoMaterializeLifetimeV4
D-FOU36-HighFrequencyQuarantineV2
```

必须测：

```text
band_energy_low/mid/high
phase_drift
high_freq_ratio
bandwise_snr
workspace raw/incremental/step
task mean/worst
NLL/ECE/CEp99
LineC
```

## 8.3 D-RBF / FastKAN 方向

预注册候选：

```text
D-RBF30-ActiveCenterOccupancyV3
D-RBF31-WidthConditionIdentityResidualV2
D-RBF32-CompactBumpNoDenseMaterializationV2
D-RBF33-GaussianLocalK4TaskHealthV2
D-RBF34-CenterOccupancyWarmupNoTaskBranch
```

必须测：

```text
center_occupancy_entropy
empty_center_fraction
width_condition
out_of_grid_fraction
residual_over_base
workspace raw/incremental/step
task mean/worst
NLL/ECE/CEp99
LineC
```

## 8.4 D-WAV 方向

低预算保留，不抢主线。

```text
D-WAV29-TriangularSupportV4
D-WAV30-ScaleOccupancyNoTailTargetV2
D-WAV31-LocalSupportOverlapDampingV2
D-WAV32-LocalTailCoverageAuditV2
```

## 8.5 Gate

Exploration substrate：

```text
family_dataset_seed_pass_count >= 6/9
step_ratio <= 1.75
memory_ratio <= 1.75
mean_delta >= -0.05
worst_delta >= -0.10
LineC_pass_rate >= 0.30
```

Official FMS eligibility：

```text
family_dataset_seed_pass_count = 9/9
strict source/task/tail/LineC/efficiency gates pass
```

未达 6/9 不能进入 official FMS proof，但不是整轮 hard stop。

## 8.6 可视化

```text
fig_d_family_pass_heatmap.svg
fig_d_workspace_task_pareto.svg
fig_d_nonrat_telemetry_matrix.svg
fig_d_linec_tail_task_breakdown.svg
```

---

# 9. Line M：MLP / generic controls

## 9.1 目标

判断任何 apparent FMS improvement 是否只是 generic optimizer / train-state trick。

## 9.2 必须比较

```text
MLP-AdamW
MLP-FMS-generic
MLP-FMS-boundary-only
MLP-SameActiveFractionControl
MLP-RandomMatchedNorm
MLP-GenericOptimizerStateControl
MLP-NoOpMatchedOverhead
```

## 9.3 判定

如果 MLP 与 D-CHE 都同等提升：

```text
Generic optimizer effect, not KAN-specific.
```

如果 D-CHE 明显强于 MLP：

```text
Possible basis-specific leverage.
```

如果 MLP controls 解释 D-CHE：

```text
FMS-specific claim rejected.
```

---

# 10. Line C：几何与 failure audit

Line C 继续只做 audit/gate，不做 direction。

必须记录：

```text
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
KernelDrift
margin_p10
CEp99
NLL
ECE
Brier
AUCtime
LineC_pass
source_vs_best_control
control_equivalent
bad_event
```

必须生成：

```text
fig_c_signal_reservoir_noise_matrix.svg
fig_c_failure_taxonomy_stacked.svg
fig_c_source_auc_tail_linec_coincidence.svg
fig_c_control_equivalent_vs_bad_event.svg
```

---

# 11. 执行加速策略

## 11.1 不再串行等待单一 gate

v14.15 必须并行执行：

```text
Line P predictor robustness
Line I causal intervention factorial
Line B boundary role reset
Line F D-CHE real-lite fork
Line D all-basis substrate acceleration
Line M controls
```

不能等 Line P 过了才开始 Line I/B/D。因为 v14.14 已经证明单线串行太慢。

## 11.2 GPU / job 分组建议

```text
GPU group 0:
  Line P + Line M, light diagnostics

GPU group 1:
  Line I factorial, D-CHE only

GPU group 2:
  Line F real-lite, D-CHE only

GPU group 3:
  Line D D-FOU / D-RBF / D-WAV substrate
```

如果资源少，则按优先级：

```text
1. Line I
2. Line F
3. Line P
4. Line D D-FOU/RBF
5. Line B
6. Line M
7. Line D D-WAV
```

## 11.3 最低完成要求

本轮不能只跑 smoke。最低完成要求：

```text
Line P: >= 1 full leaveout table
Line I: full 3-axis factorial at least reduced matrix
Line B: BND1..BND5 all executed
Line F: at least 3 D-CHE mechanism candidates x 3x3 real-lite
Line D: at least D-FOU + D-RBF full substrate rows
Line M: matched controls for every promoted-looking result
Line Z: route/no-go/failure taxonomy
```

---

# 12. Stop / continue contract

## 12.1 Promotion fail-closed

S5 必须同时满足：

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
forbidden audit pass
no-action-search audit pass
code review pass
promotion_allowed = 1
```

## 12.2 Exploration continue-open

以下情况不能 hard stop：

```text
Line P AUC < 0.75
Line I direction gain fail
Line B boundary is harmless-null
Line F real-lite <4/9
Line D non-D-CHE <6/9
MLP/generic controls positive
LineC/tail/AUC fail
```

这些只能写 route / failure taxonomy / next hypothesis，不能阻止其它线执行。

## 12.3 Hard stop 条件

只有以下情况 hard stop：

```text
required_artifact_missing_count > 0
forbidden_information_violation_count > 0
no_action_search_violation_count > 0
direction uses validation/test/future/query
direction uses LineC/CEp99/NLL/ECE/AUCtime
dataset-name branch
seed-specific scale
action token / controller / action bank / reset route
fake / proxy / CPU offload
```

---

# 13. Codex 必须执行的文件与 artifact

## 13.1 建议新增 runner

```text
experiments/run_v1415_parallel_fms_causal_fork_transfer_allbasis.py
```

## 13.2 必须生成 CSV

```text
v1415_progress_table.csv
v1415_predictor_robustness.csv
v1415_predictor_leaveout.csv
v1415_intervention_factorial.csv
v1415_intervention_causal_effects.csv
v1415_boundary_role_reset.csv
v1415_dche_real_lite.csv
v1415_dche_controls.csv
v1415_allbasis_substrate_acceleration.csv
v1415_mlp_generic_controls.csv
v1415_linec_tail_audit.csv
v1415_failure_taxonomy.csv
v1415_required_artifact_manifest.csv
v1415_forbidden_information_audit.csv
v1415_no_action_search_audit.csv
v1415_code_review_manifest.csv
v1415_route_decision.json
v1415_next_hypothesis_queue.md
```

## 13.3 必须生成图

```text
fig_v1415_progress_dashboard.svg
fig_p_predictor_leaveout_heatmap.svg
fig_i_factorial_effect_heatmap.svg
fig_b_boundary_role_matrix.svg
fig_f_dche_real_lite_matrix.svg
fig_d_allbasis_substrate_pareto.svg
fig_c_failure_taxonomy.svg
```

---

# 14. Codex fallback 指令

如果 Line P 失败：

```text
不要停止。
继续 Line I/B/F/D。
记录 R-P-PredictorWeak。
不要新增 predictor token。
```

如果 Line I 显示 FMS direction control-equivalent：

```text
停止扩 direction method。
进入 Line B boundary/curriculum interpretation。
记录 R-I-DirectionControlEquivalent。
```

如果 Line B 也是 NoOp-equivalent：

```text
记录 R-B-HarmlessNullBoundary。
不要新增 boundary token。
继续 Line D/M/C/Z。
```

如果 Line F real-lite <4/9：

```text
记录 failure taxonomy。
不得新增 F-CHE8/FMS-M6。
继续 Line D 与 Line M。
```

如果 Line D D-FOU/RBF <6/9：

```text
继续输出 substrate blocker taxonomy。
不得 official FMS proof。
不得把 workspace positive 写成 substrate success。
```

如果任何 line 出现 local positive row：

```text
不得基于 local positive 新增 action/token。
只能在 route 中标记 LocalPositiveNoPromotion。
```

---

# 15. 本轮预期结论分叉

## 15.1 最好情况

```text
Line I 证明 FMS direction 有 causal value；
Line F real-lite >=6/9；
D-FOU/RBF 至少一个 >=6/9；
下一轮进入 real S4/S5 confirm。
```

## 15.2 中等情况

```text
FMS direction control-equivalent；
但 boundary/curriculum 减少 bad event；
D-CHE real-lite 提升到 3/9 或 4/9；
下一轮围绕 boundary-state 而不是 direction 设计。
```

## 15.3 负面但有价值情况

```text
Predictor 不稳；direction 无 causal value；boundary NoOp-equivalent；D-FOU/RBF 仍 blocked。
结论：当前 FMS definition no-go，项目应转向 substrate/base 或新的 functional theory。
```

## 15.4 必须避免的伪进展

```text
local positive row；
control-equivalent endpoint；
NoOp-safe but no source；
workspace-only substrate；
synthetic-only pass；
real-lite but not strict；
MLP generic positive；
audit metric improvement without source/control gap。
```

---

# 16. 总结

v14.15 的核心不是“再跑一个版本”，而是把目前的模糊失败拆成可判定的并行因果问题：

$$
\boxed{
\text{FMS 到底是 predictor 有用、direction 有用、boundary 有用，还是都只是 control-equivalent？}
}
$$

如果 direction 有用，继续 functional update 主线。  
如果 direction 无用但 boundary 有用，FMS 重定义为 training boundary / curriculum state。  
如果二者都无用，停止 FMS 小修，转入 substrate / 新理论。  
如果 D-FOU/RBF 打开 substrate，则 functional carrier 不再只依赖 D-CHE。

本轮必须加速：并行执行，不串行等 gate；但必须守住 no-action-search、no audit-direction、no dataset-branch、no promotion inflation 的硬边界。
