# DG-KAN v15.6：Stability-Nullspace Split-Consensus Functional Update + All-Basis 并行加速完整计划

生成时间：2026-05-31（Asia/Singapore）

> 本计划基于 v15.5 `SplitConsensusMetricStability AllBasisAcceleration` 真实结果复盘。  
> 本计划不把 G7R source-positive、trust-scalar diagnostic、substrate-only rows、MLP/generic controls 写成 promotion。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler/class weight；no dataset-name branch；no seed-specific scale；no fake/proxy/CPU offload；functional direction 不使用 validation/test/future/query；LineC/CEp99/NLL/ECE/AUCtime/Brier 只能作为 audit/gate/failure taxonomy，不能作为方向源；不启动 action bank/controller/reset route；不新增 G9/G10、FU9/FU10、F-CHE8/F-CHE9。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找到一个局部指标好看的 update。总目标是：

$$
\boxed{
\text{在 strict FC-PureKAN base 上，通过 functional update 获得比 ordinary backprop / AdamW 更好的训练几何和模型。}
}
$$

具体必须同时满足：

```text
1. 表达力不打折，最好优于同规模 MLP；
2. forward / backward / step / memory 接近 MLP；
3. 训练轨迹健康，AUC-step / AUC-time 不输；
4. functional update 的收益必须独立于 AdamW / Cautious / MGUP / Random / NoOp / MLP generic controls；
5. 几何更好：source gain、LineC、tail、calibration、signal/reservoir/noise 同位；
6. functional direction 必须来自 train stream / loss-interface generic signal，不能从 audit metric 反推。
```

最终要证明的是：

$$
\boxed{
\text{PureKAN base + functional update}
>
\text{PureKAN base + ordinary AdamW / matched controls.}
}
$$

## 0.2 v15.5 真实状态

v15.5 的最终 route 是：

```text
route = R4-LineCOrTailDominated
minimum_success = S1-StableSplitConsensusSubspaceObservable
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
contract_unclosed_rows = 0
deep_coverage_unclosed_rows = 0
```

v15.5 的关键事实是：

```text
Line S:
  split-consensus subspace observable。
  line_s_gate_pass = 1
  line_s_best_snr = 1.9560
  K=2 sensitivity best SNR = 10.5196

Line G:
  best = G7R-D-CHE-MetricOnlyReplay
  real_lite_pass_count = 1/9
  source_vs_best_control_mean = 0.4615
  control_equivalent_fraction = 0.0
  bad_event_fraction = 1.0
  tail_fail_fraction = 1.0
  linec_fail_fraction = 0.7778

Line B:
  best_proxy = B10-projection_retention_drift
  bad-event AUC = 0.8094
  leaveout min AUC = 0.25
  因此 proxy 不稳健，不能作为 gating / direction。

Line D:
  D-FOU / D-RBF / D-WAV = 0/9
  无 non-D-CHE official FU carrier。

Line M:
  MLP/generic controls 没有完全解释 G7R source gain。
```

所以 v15.5 不是“没信号”，而是：

$$
\boxed{
\text{G7 split-consensus metric-only 有真实 source gain，}
\text{但 source gain 与 tail/LineC bad-event 绑定在一起。}
}
$$

当前不能继续问：

```text
能不能再加 G9/G10？
能不能再换一个 trust scalar？
能不能启动 controller？
能不能用 LineC/tail failure pattern 反推方向？
```

这些都会重蹈 v9/v12 的 action-search 错误。

当前必须问：

$$
\boxed{
\text{如何在不使用 audit metric 作为方向源的前提下，}
\text{把 source-carrying component 和 instability-carrying component 解耦？}
}
$$

---

# 1. 各线进展百分数

| 线 | 当前完成度 | 判断 |
|---|---:|---|
| 代码 / provenance / finalizer 审计 | 99% | required / forbidden / no-action-search / contract 闭合 |
| Historical B320 / FHQ anchor | 85% frozen | 历史强，但 label-informed init 禁用，不能作最终 official claim |
| Line C 几何审计 | 88% | 审计成熟；只能解释失败，不能生成方向 |
| PopRisk / FU / FMS infrastructure | 95% | per-example gradient、second moment、projection telemetry、controls 都成熟 |
| Current FU / FMS / proximal / ST-FU family | 0%-3% | v15.0-v15.3 已基本 no-go |
| Split-consensus signal observability | 55%-60% | v15.04/v15.5 均证明 subspace observable |
| Split-consensus G7 metric-only source | 40%-45% | source positive 且 control_equivalent=0；但不稳定 |
| Split-consensus stability control | 5%-10% | G7S trust scalar 未解决 bad event |
| Train-stream bad-event proxy | 15%-20% | B10 AUC 高但 leaveout 崩，不能用于 gating |
| D-CHE substrate | 85% | 当前最强 Non-RAT carrier，历史 9/9 eligibility |
| D-CHE stable real-transfer FU | 10%-15% | G7R/G7S real-lite 1/9，S4/S5 未打开 |
| Rational substrate | 85% | 稳定；reset / optimizer-state route 被 generic confound 打回 |
| D-FOU substrate | 20%-30% | v15.5 0/9；历史 6/9 未稳定复现 |
| D-RBF / FastKAN substrate | 20%-30% | v15.5 0/9；task-health 不稳 |
| D-WAV substrate | 15%-20% | v15.5 0/9；低预算保留 |
| Non-D-CHE official FU proof | 0%-5% | 目前不可打开 |
| Official S5 functional success | 0% | 尚未达成 |
| 整体 next-gen MLP claim | 30%-38% | 比 v15.3 上调，但仍远；有 signal，无 stability |

---

# 2. 独立分析：这次到底有没有进展？

## 2.1 有进展：source signal 重新站住了

v15.0-v15.3 的状态很糟：optimizer-aware FU、CR-FU、function-metric、proximal、split-transfer operator 都没有 real-lite success，且多数 positive-looking rows 被 generic controls 解释。

v15.04 / v15.5 的变化是：

```text
1. split-consensus subspace 可观测；
2. G7 metric-only 的 source_vs_best_control_mean 明显为正；
3. control_equivalent_fraction = 0；
4. MLP/generic controls 没有完全解释 G7R source gain。
```

这说明 split-consensus metric-only 不是普通 random matched update，也不是纯 generic optimizer artifact。

## 2.2 但它不是成功：source 与 instability 绑定

v15.5 的 best 方法 G7R：

```text
source_vs_best_control_mean = 0.4615
bad_event_fraction = 1.0
tail_fail_fraction = 1.0
linec_fail_fraction = 0.7778
real_lite_pass_count = 1/9
```

这说明：

$$
\boxed{
\text{G7R 的 source gain 不是无效，}
\text{但它沿着会触发 tail / LineC instability 的方向进入模型。}
}
$$

G7S1..G7S5 试过 train-stream-only trust scalar，但本质结果是：

```text
1. G7S3 唯一把 bad_event 从 9/9 降到 8/9，并得到 1/9 pass；
2. G7S4/G7S5 压低 trust scalar 后，source retention 只有约 0.55；
3. tail 有下降，但 LineC 仍高；
4. scalar damping 没有选择性保留 source gain。
```

因此不能继续做：

```text
更强 damping；
更低 strength；
更复杂 scalar trust；
G9/G10。
```

这只会把 source 和 bad-event 一起削掉。

---

# 3. 当前真正 blocker

当前 blocker 是：

$$
\boxed{
\text{source-carrying component 与 instability-carrying component 未被解耦。}
}
$$

更具体地说：

```text
1. split-consensus subspace 中存在 source-positive direction；
2. 但该 direction 同时包含 tail/LineC instability tangent；
3. train-stream scalar proxy 无法稳定识别 bad-event；
4. all-basis alternative carrier 没打开；
5. 继续 action/token 会重复 v9/v12 的旧错误。
```

v15.6 的核心不是继续找“更好的动作”，而是把 G7R 更新分解为：

$$
u_{G7}
=
u_{source}
+
u_{hazard}
+
u_{residual}.
$$

目标是保留 $u_{source}$，移除或约束 $u_{hazard}$，同时不使用 audit metric 作为方向源。

---

# 4. v15.6 总假设

## H1：G7R 中 source 与 hazard 在参数空间可部分分离

如果 source 与 hazard 完全共线，那么任何稳定化都会同时杀死 source，当前路线需要 substrate/base 重设。

如果存在部分可分离结构，则可以通过 train-stream stability tangent nullspace 或 constrained projection 降低 bad-event，同时保留 source。

判据：

$$
\operatorname{source\_retention\_after\_hazard\_null} \ge 0.60
$$

且：

$$
\operatorname{hazard\_overlap}(u_{G7}) \le 0.70.
$$

## H2：bad-event 不应靠 scalar predictor，而应靠 tangent constraint

v15.5 的 B10 proxy AUC 高但 leaveout 崩，说明 scalar predictor 不稳健。下一步不把 proxy 当 selector，而把 train-stream stability functional 的梯度当作约束方向。

## H3：G7R 的 source gain 应先复现，再稳定

不允许在没有复现 G7R source gain 的情况下直接测试 stabilization。Line G0 必须复现：

```text
source_vs_best_control_mean > 0.20
control_equivalent_fraction <= 0.20
```

否则 v15.6 先 route 到 `R1-G7RSourceNotReproduced`，但不能 hard stop Line D / Line S audit。

## H4：D-CHE 不是唯一长期 carrier

D-FOU / D-RBF / D-WAV 虽然 v15.5 都是 0/9，但仍要并行推进 substrate。不能因为 G7R 有 source gain 就放弃 all-basis。

---

# 5. v15.6 实验总览

v15.6 名称：

```text
DG-KAN v15.6
Source-Stability Decoupled Split-Consensus Functional Update
+
All-Basis 并行加速
```

核心实验线：

```text
Line R:
  implementation / provenance / no-action-search audit。

Line S:
  split-consensus subspace reproducibility。

Line U:
  source-hazard tangent decomposition，只做 readback / audit。

Line N:
  stability-nullspace split-consensus update。

Line Q:
  constrained source-retaining stability projection。

Line P:
  path / micro-horizon / transfer audit，只解释不生成方向。

Line D:
  all-basis substrate acceleration。

Line M:
  MLP / generic controls。

Line C:
  geometry / tail / LineC audit。

Line Z:
  route / no-go / exhaustion certificate / next queue。
```

---

# 6. Line U：source-hazard tangent decomposition

## 6.1 目的

判断 G7R 的 source component 和 hazard component 是否可分离。

这一步不提交 update，只记录方向分解。

## 6.2 方向定义

令：

$$
u = u_{G7}
$$

是 G7 metric-only split-consensus update。

定义 train-stream stability functionals $\psi_j(\theta)$：

```text
psi_1: train loss q95
psi_2: train loss q90
psi_3: logit RMS
psi_4: entropy collapse proxy
psi_5: split loss disagreement
psi_6: recovery lag h1/h2/h4
psi_7: D-CHE high-degree energy fraction
psi_8: degree entropy collapse
psi_9: update cosine disagreement with AdamW
psi_10: projection retention drift
```

这些只能使用 current train stream、current logits、current loss、D-CHE degree telemetry、AdamW/FU state，不使用 validation/test/future/query，不使用 LineC/CEp99/NLL/ECE/AUCtime/Brier 作为方向源。

每个 $\psi_j$ 的 tangent：

$$
v_j = \nabla_\theta \psi_j.
$$

构造 hazard tangent matrix：

$$
V=[v_1,\dots,v_m].
$$

使用 Adam second moment 或 D-CHE role-wise second moment 作为 metric：

$$
M_t = \operatorname{diag}(\sqrt{v^{Adam}_t}+\epsilon)
$$

或 role-block approximation。

计算：

$$
P_V u
=
V(V^\top M_t V + \lambda I)^{-1} V^\top M_t u.
$$

hazard overlap：

$$
h(u)=\frac{\|P_V u\|_{M_t}}{\|u\|_{M_t}+\epsilon}.
$$

source retention after nulling hazard：

$$
r(u)=
\frac{\langle u-P_Vu, u\rangle_{M_t}}
{\|u\|_{M_t}^2+\epsilon}.
$$

## 6.3 记录指标

```text
hazard_overlap
source_retention_after_null
hazard_tangent_rank
hazard_tangent_condition
per_proxy_dirderiv_before
per_proxy_dirderiv_after
cos_u_g7_adam
cos_u_g7_hazard
degree_role_hazard_overlap
readout_role_hazard_overlap
high_degree_role_hazard_overlap
```

## 6.4 判断

如果：

```text
hazard_overlap >= 0.80
source_retention_after_null <= 0.30
```

说明 source 与 hazard 强耦合。下一步不能继续 scalar trust 或 nullspace projection，应记录：

```text
R-U-SourceHazardColinear
```

如果：

```text
hazard_overlap <= 0.70
source_retention_after_null >= 0.60
```

说明值得进入 Line N / Q。

---

# 7. Line N：Stability-nullspace split-consensus update

## 7.1 目的

对 G7R update 做 train-stream hazard-null projection，不新增方向 family。

## 7.2 更新公式

原始 update：

$$
u_{G7}
$$

hazard-null update：

$$
u_N
=
u_{G7}
-
V(V^\top M_t V + \lambda I)^{-1}V^\top M_t u_{G7}.
$$

最终提交：

$$
\Delta\theta
=
\eta \cdot P_{safe}(u_N)
+
\Delta\theta_{decoupled}.
$$

其中 $P_{safe}$ 只做 D-CHE degree / parameter safety，不定义 value。

## 7.3 预注册 variants

```text
N0-G7R-Replay
N1-LossQuantileNull
N2-LogitRMSNull
N3-EntropyRecoveryNull
N4-DegreeEnergyNull
N5-CombinedHazardNull-rank4
N6-CombinedHazardNull-rank8
```

这不是 G9/G10；N1-N6 都是同一个 G7R update 的 train-stream stability-nullspace projection。禁止新增 N7/N8。

## 7.4 Controls

必须比较：

```text
C0-D-CHE-AdamW
C1-CautiousAdamW
C2-MGUP
C3-G7R-RandomMatchedNorm
C4-RandomNullspaceSameRank
C5-SameHazardNullRandomDirection
C6-NoOpMatchedOverhead
C7-MLP-SameHazardNullControl
C8-SameSourceRetentionRandomDirection
```

## 7.5 Exploration gate

```text
real_lite_pass_count >= 3/9
source_vs_best_control_mean > 0
control_equivalent_fraction <= 0.60
bad_event_fraction <= 0.70
source_retention_after_null >= 0.50
```

## 7.6 Meaningful gate

```text
real_lite_pass_count >= 4/9
source_vs_best_control_mean >= 0.005
control_equivalent_fraction <= 0.50
bad_event_fraction <= 0.50
tail_fail_fraction <= 0.60
linec_fail_fraction <= 0.60
```

## 7.7 S4 exploration

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail fail reduced vs G7R
LineC fail reduced vs G7R
controls fail
```

---

# 8. Line Q：Constrained source-retaining stability projection

## 8.1 目的

如果 Line N 只是粗暴投掉 hazard component，可能 source 也被投掉。Line Q 直接求一个 constrained projection，显式要求 source retention。

## 8.2 Solver

求：

$$
u_Q
=
\arg\min_u
\|u-u_{G7}\|_{M_t}^2
+
\rho\|u\|_{M_t}^2
$$

约束：

$$
\nabla\psi_j^\top u \le \delta_j
$$

并且：

$$
\frac{\langle u,u_{G7}\rangle_{M_t}}
{\|u_{G7}\|_{M_t}^2+\epsilon}
\ge r_{min}.
$$

第一版只使用固定参数：

```text
r_min = 0.60
delta_j = 0 for increasing-risk proxies
rho = 1e-3
max_constraints = 4 or 8
```

禁止网格搜索。若 Q 失败，不能临时调 r_min / delta_j / rho；只能进入 Q failure taxonomy。

## 8.3 预注册 variants

```text
Q1-SourceRetainingLossQuantileConstraint
Q2-SourceRetainingLogitEntropyConstraint
Q3-SourceRetainingDegreeEnergyConstraint
Q4-SourceRetainingCombinedConstraint-r4
Q5-SourceRetainingCombinedConstraint-r8
```

不允许 Q6/Q7。

## 8.4 Failure taxonomy

```text
Q-F1: infeasible_constraints
Q-F2: source_retention_low
Q-F3: control_equivalent
Q-F4: bad_event_not_reduced
Q-F5: overhead_blocked
```

---

# 9. Line P：Path / micro-horizon audit

Line P 不生成方向，只解释失败。

## 9.1 必须记录

```text
B1 loss_delta_h1/h2/h4
B2 loss_delta_h1/h2/h4
micro_horizon_integral
recovery_lag
logit_rms_drift_h1/h2/h4
entropy_drift_h1/h2/h4
train_loss_q95_drift_h1/h2/h4
degree_energy_drift_h1/h2/h4
path_source_retention
path_hazard_growth
```

## 9.2 Failure classes

```text
P1-SourceLostAfterStabilityProjection
P2-HazardNotRemoved
P3-MicroHorizonGoodRealBad
P4-ControlEquivalentPath
P5-OverheadDominated
P6-LineCOrTailDominated
P7-SourceHazardColinear
P0-OK
```

---

# 10. Line D：All-basis substrate parallel acceleration

D-CHE 不能成为唯一 carrier。v15.6 必须并行推进：

## 10.1 D-FOU

预注册方向：

```text
D-FOU67-LowFreqIdentityResidualV5
D-FOU68-BandwiseConsensusMetricV2
D-FOU69-PhaseStableBandMixV3
D-FOU70-HighFrequencyQuarantineV2
D-FOU71-NoMaterializeLifetimeV4
```

必须记录：

```text
low/mid/high_band_energy
high_freq_ratio
phase_drift
bandwise_snr
workspace_raw_ratio
workspace_incremental_ratio
step_ratio
task_mean_delta
task_worst_delta
LineC_pass_rate
```

## 10.2 D-RBF / FastKAN

预注册方向：

```text
D-RBF65-CompactBumpIdentityResidualV3
D-RBF66-ActiveCenterOccupancyRepairV3
D-RBF67-WidthConditionGuardV3
D-RBF68-GaussianLocalK4NoDenseMaterializationV2
D-RBF69-CenterSNRWarmupV2
```

必须记录：

```text
center_occupancy_entropy
empty_center_fraction
width_p01
width_p99
active_center_snr
out_of_grid_fraction
workspace_incremental_ratio
step_ratio
task_mean_delta
LineC_pass_rate
```

## 10.3 D-WAV

预注册方向：

```text
D-WAV57-TriangularSupportV5
D-WAV58-ScaleOccupancyHardeningV3
D-WAV59-SupportOverlapDampingV3
D-WAV60-LocalTailCoverageAuditV2
```

必须记录：

```text
scale_occupancy_entropy
support_overlap
local_tail_coverage_proxy
scale_energy_ratio
workspace_incremental_ratio
step_ratio
task_mean_delta
LineC_pass_rate
```

## 10.4 Substrate gates

Exploration substrate gate：

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
step_ratio <= 1.25
memory_ratio <= 1.25
source/tail/LineC no-regression
```

未到 6/9 时不允许 official FU proof，但不能 hard stop 整轮。

---

# 11. Line M：MLP / generic controls

每个 positive-looking result 必须比较：

```text
MLP-AdamW
MLP-CautiousAdamW
MLP-MGUP
MLP-SplitConsensusMetricNoProjection
MLP-SameHazardNullControl
MLP-SameTrustScalarControl
MLP-RandomNullspaceSameRank
MLP-NoOpMatchedOverhead
```

如果 MLP/generic controls 解释结果，则：

```text
KAN_specific_claim_allowed = 0
promotion_allowed = 0
```

---

# 12. 统一记录指标

## 12.1 Source / task

```text
source_vs_best_control
source_vs_adamw
real_lite_pass
dataset_seed_pass
AUCtime_ratio
CEp99_delta
NLL_delta
ECE_delta
Brier_delta
margin_p10_delta
```

## 12.2 Stability

```text
bad_event
bad_event_fraction
tail_fail_fraction
linec_fail_fraction
train_loss_q95_drift
logit_rms_drift
entropy_collapse
degree_energy_drift
high_degree_energy_fraction
```

## 12.3 Decomposition

```text
hazard_overlap
source_retention_after_null
hazard_tangent_rank
hazard_tangent_condition
per_proxy_dirderiv_before
per_proxy_dirderiv_after
nullspace_removal_fraction
projection_retention
cos_u_g7_adam
cos_u_g7_hazard
```

## 12.4 Efficiency

```text
step_time_ratio
memory_ratio
extra_solve_time_ms
subspace_build_time_ms
constraint_solve_time_ms
overhead_fraction
```

## 12.5 Controls

```text
control_equivalent
control_equivalent_fraction
generic_control_explains
same_hazard_null_random_pass
same_source_retention_random_pass
mlp_control_pass
```

---

# 13. 必须生成的 figures

```text
fig_v156_source_vs_bad_event_scatter.svg
fig_v156_hazard_overlap_vs_source_retention.svg
fig_v156_nullspace_before_after_proxy_dirderiv.svg
fig_v156_source_retention_vs_tail_linec.svg
fig_v156_line_g_nq_method_heatmap.svg
fig_v156_line_b_proxy_leaveout_heatmap.svg
fig_v156_micro_horizon_path.svg
fig_v156_controls_comparison.svg
fig_v156_allbasis_substrate_heatmap.svg
fig_v156_failure_taxonomy.svg
fig_v156_gate_dashboard.svg
```

---

# 14. 成功 / 失败判定

## 14.1 S1：Signal observable

```text
Line S gate pass = 1
G7R source_vs_best_control_mean > 0.20
control_equivalent_fraction <= 0.20
```

## 14.2 S2：Stability decoupling exploration

```text
real_lite_pass_count >= 3/9
bad_event_fraction <= 0.70
tail_fail_fraction < G7R_tail_fail_fraction
linec_fail_fraction < G7R_linec_fail_fraction
source_retention >= 0.50
```

## 14.3 S3：Meaningful stabilized positive

```text
real_lite_pass_count >= 4/9
source_vs_best_control_mean >= 0.005
bad_event_fraction <= 0.50
control_equivalent_fraction <= 0.50
```

## 14.4 S4：Real-transfer exploration

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail fail reduced vs AdamW and G7R
LineC fail reduced vs AdamW and G7R
controls fail
```

## 14.5 S5：Official functional success

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

---

# 15. Stop / continue 制度

## 15.1 Promotion fail-closed

S5 没达成时：

```text
promotion_allowed = 0
official_success_reached = 0
```

不能把 S1/S2/S3/S4 写成 S5。

## 15.2 Exploration continue-open

这些不能 hard stop：

```text
Line S fail
Line U source-hazard colinear
Line N fail
Line Q infeasible
Line P bad
Line D all-basis fail
MLP/generic controls positive
D-CHE real-lite <4/9
LineC/tail/AUC fail
overhead high
proxy leaveout fail
```

必须进入 fallback ladder / failure taxonomy / exhaustion certificate。

## 15.3 真正 hard stop

只有这些可以 hard stop：

```text
required_artifact_missing_count > 0
forbidden_information_violation_count > 0
no_action_search_violation_count > 0
direction uses validation/test/future/query
direction uses LineC/CEp99/NLL/ECE/AUCtime/Brier
dataset-name branch
seed-specific scale
action token / controller / action bank / reset route
fake / proxy / CPU offload
```

---

# 16. Fallback ladder

## 16.1 Line U fallback

```text
U-FB1: hazard tangent normalization audit
U-FB2: role-wise hazard overlap audit
U-FB3: degree-wise hazard overlap audit
U-FB4: control residualized hazard overlap audit
U-FB5: source-hazard colinearity certificate
```

## 16.2 Line N fallback

```text
N-FB1: nullspace rank sensitivity r=2/4/8
N-FB2: metric sensitivity identity / AdamV / DegreeRole
N-FB3: source retention floor audit
N-FB4: random nullspace matched controls
N-FB5: exhaustion certificate
```

## 16.3 Line Q fallback

```text
Q-FB1: feasibility certificate
Q-FB2: constraint active set readback
Q-FB3: source retention failure readback
Q-FB4: matched random constrained controls
Q-FB5: exhaustion certificate
```

## 16.4 Line D fallback

```text
D-FB1: v14.9/v15.5 historical replay comparison
D-FB2: gate mismatch audit
D-FB3: budget mismatch audit
D-FB4: candidate mismatch audit
D-FB5: family-specific blocker taxonomy
D-FB6: substrate exhaustion certificate
```

---

# 17. 并行加速执行策略

v15.6 必须并行，不再串行：

```text
GPU group 0:
  Line S + Line U source-hazard tangent decomposition。

GPU group 1:
  Line N stability-nullspace G7 update。

GPU group 2:
  Line Q constrained projection update。

GPU group 3:
  Line D D-FOU + D-RBF substrate acceleration。

GPU group 4:
  Line D D-WAV + D-CHE/Rational no-regression + Line P path audit。

GPU group 5 if available:
  Line M MLP/generic controls。
```

最低完成合同：

```text
Line R:
  complete

Line S:
  K=2/4/8, batch=128/256 sensitivity

Line U:
  hazard overlap table for G7R/G7S3/top controls

Line N:
  N0..N6 + controls

Line Q:
  Q1..Q5 + controls

Line P:
  top-2 N/Q methods + controls micro-horizon

Line D:
  D-FOU and D-RBF full rows;
  D-WAV low-budget;
  D-CHE/Rational no-regression

Line M:
  controls for every positive-looking result

Line Z:
  route / no-go / exhaustion certificate / next queue
```

---

# 18. Codex 执行要求

Codex 必须执行：

```text
1. 不许新增 G9/G10/FU9/FU10/F-CHE8/F-CHE9。
2. 不许 action bank / controller / reset。
3. 不许用 real fail pattern 反推 direction。
4. 不许用 LineC/tail/AUC/calibration audit metric 生成方向。
5. 不许单线失败后停止整轮。
6. 不许把 trust scalar / proxy / substrate-only 写成 promotion。
7. 必须写 source-hazard decomposition。
8. 必须写 matched controls。
9. 必须写 failure taxonomy。
10. 必须写 exhaustion certificate。
```

---

# 19. 最终判断

v15.5 的真正结论是：

$$
\boxed{
\text{split-consensus metric-only 有 source-positive / control-resistant signal，}
\text{但 source 与 tail/LineC instability 绑定。}
}
$$

所以 v15.6 不该继续找新 update，而该问：

$$
\boxed{
\text{source-carrying tangent 和 hazard-carrying tangent 是否可分离？}
}
$$

如果可分离，就用 stability-nullspace / constrained projection 稳定 G7。

如果不可分离，就必须接受：

```text
R-SourceHazardColinear
```

然后转向 substrate/base architecture，而不是继续 G-token 搜索。
