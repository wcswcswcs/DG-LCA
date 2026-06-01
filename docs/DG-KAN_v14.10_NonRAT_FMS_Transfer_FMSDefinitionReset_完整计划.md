# DG-KAN v14.10：Non-RAT FMS Transfer + FMS Causal Definition Reset + All-Basis Parallel 完整计划

> 版本：v14.10 execution plan  
> 生成时间：2026-05-29 / Asia-Singapore  
> 目标：在不重蹈 v9/v12 “寻找好动作 / action search” 错误的前提下，利用 v14.9 打开的 D-CHE 9/9 substrate eligibility，正式验证 Non-RAT basis 上是否存在 functional update 的 causal value；同时关闭 optimizer-state reset 小修路线，继续 all-basis substrate 并行推进。  
> 公式格式：Typora 友好；只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no label-informed initialization；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只作为 audit / gate，不能作为 direction source；不新增 action token；不启动 controller；不把 endpoint-only / control-equivalent / diagnostic / substrate eligibility 写成 promotion。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找一个局部好看的几何指标。项目目标是：

$$
\boxed{\text{strict FC-PureKAN substrate/base + loss-interface-generic functional update}}
$$

在表达力、效率、收敛、几何与泛化审计上稳定优于：

$$
\boxed{\text{same substrate/base + ordinary backprop / AdamW / matched controls}}
$$

最终 claim 必须同时满足：

```text
1. 表达力不打折；
2. forward / backward / step / memory 与 MLP 可比；
3. AUC-step / AUC-time 不慢于 MLP / same-base AdamW；
4. task/source 改善不是 NoOp / random / AdamWParallel / generic optimizer reset controls 可解释；
5. LineC / signal-reservoir-noise / tail / calibration 不坏；
6. functional update 的收益来自可审计 causal mechanism，而不是 action search / endpoint artifact / control-equivalent effect。
```

## 0.2 v14.9 当前进展

v14.9 的结果分成两条线，必须分开判断。

### Line F：FMS causal audit

v14.9 已完成 FMS causal specificity audit。最终状态是：

```text
route = R3-GenericOptimizerStateResetConfound
best_fms_method = F2-RAT-FMS-NoTransport
best_fms_strict_pass_count = 0 / 9
best_fms_endpoint_vs_adamw_pass_count = 2 / 9
interaction_endpoint_pass_count = 0 / 9
interaction_strict_pass_count = 0 / 9
fms_specific_source_delta_mean = -0.13275567690531412
generic_optimizer_state_confound = 1
direction_control_equivalent = 1
mlp_analog_confound = 1
official_s5_reached = 0
promotion_allowed = 0
```

结论：当前 Rational-FMS / optimizer-state-reset 路线没有证明 FMS-specific causal value。generic optimizer-state controls 明显更强，例如 `G4 EventMatchedRandomReset endpoint-vs-AdamW = 7/9`，而 best FMS endpoint 只有 `2/9`。

因此 v14.10 禁止继续：

```text
reset 小修；
affected-mask topK / threshold 网格；
controller；
K-RT / K-AUC / K-FL action token；
endpoint-only promotion；
把 random/full/generic reset 可解释的结果写成 FMS-specific。
```

### Line D：all-basis substrate repair

v14.9 的真正正进展来自 Line D。经过 gate 修正、hardening20 和 3-seed LineC audit 后，Non-RAT substrate 状态变为：

```text
route = S4e-NonRATSubstrateEligibleNoFMSProof
D-CHE substrate = 9 / 9
D-FOU substrate = 6 / 9
D-RBF substrate = 6 / 9
D-WAV substrate = 2 / 9
official_fms_proof_executed = 0
official_s5_reached = 0
promotion_allowed = 0
```

D-CHE 的关键指标：

```text
best_mean_delta_vs_MLP = 0.033203125
best_LineC_pass_rate = 1.0
min_NLL_ratio_vs_MLP = 0.4779699915362838
median_train_step_ratio_vs_MLP = 0.2516527133750648
```

这说明 D-CHE 已经达到 **official FMS eligibility**，但 v14.9 没有预注册 Non-RAT official FMS proof 的 method set / controls / gates / required artifacts，因此不能在 v14.9 内临时执行或追认成功。

## 0.3 v14.10 的核心判断

v14.9 不是“没有进展”。它有两个非常重要的结论：

```text
1. Rational-FMS reset / optimizer-state route 当前被 generic optimizer-state dynamics 解释，必须关闭。
2. D-CHE 已经成为第一个 9/9 Non-RAT official-FMS-eligible substrate，必须单独预注册 functional proof。
```

因此 v14.10 不再问：

```text
如何继续修 zero reset？
如何继续找 action？
如何启动 controller？
如何再调 K-RT / K-AUC？
```

而是问：

$$
\boxed{\text{FMS 是否能在 D-CHE 这类 Non-RAT substrate 上产生不被 generic controls 解释的 causal value？}}
$$

这一步非常关键：如果 D-CHE-FMS 也失败，则说明当前 FMS definition 本身仍未成立；如果 D-CHE-FMS 成功，则说明 Rational 的失败可能是 substrate-specific 或 optimizer-state-confounded，而 functional update 仍有可救的 non-RAT 路径。

---

# 1. 当前各线进展百分比

> 百分比是基于 gate、artifact 完整性、机制清晰度、离 official success 的距离的人工估计，不是报告中的官方字段。

| 线 | 当前完成度 | 判断 |
|---|---:|---|
| 代码 / provenance / finalizer 审计 | 99% | artifact、forbidden audit、no-action-search audit 很成熟，不是当前 blocker |
| Historical FHQ / B320-current | 85% frozen | 历史强，但 label-informed init 已禁用，不能 official |
| Line C 几何审计 | 88% | LineC / tail / signal-reservoir audit 稳定，但只做审计，不能做 direction source |
| PopRisk / FMS infrastructure | 94% | persistent state、per-example gradient、projection trace、FMS runner 成熟 |
| Rational substrate | 85% | 仍是稳定 substrate，但 Rational-FMS specificity 当前失败 |
| Rational-FMS causal specificity | 10%-15% | v14.9 R3 generic confound，FMS-specific value 未证明 |
| Optimizer-state reset / action route | closed | v14.8/v14.9 已阻断；不再作为主线 |
| Generic optimizer-state confound audit | 90% | 机制判别很清楚，generic controls 更强 |
| D-CHE substrate | 85% | 9/9 Non-RAT official-FMS-eligible substrate，是本轮最大进展 |
| D-CHE official FMS proof | 0% | 尚未执行，v14.10 主线 |
| D-FOU substrate | 55%-60% | 6/9 exploration opened，需要补到 9/9 或保留 exploration |
| D-RBF substrate | 50%-55% | 6/9 exploration opened，但 task-health / width/center 仍需稳定 |
| D-WAV substrate | 20%-25% | 2/9，历史有局部线索但当前仍弱 |
| Non-RAT official FMS overall | 0%-5% | 只有 eligibility，没有 FMS proof |
| MLP analog / generic control | 35%-40% | 可作为 confound/control；不是 KAN success |
| 整体 next-gen MLP claim | 40%-48% | 比 v14.8 更清楚，D-CHE 带来新机会，但 S5 仍为 0 |

---

# 2. v14.9 独立分析

## 2.1 Line F 的结果不是“FMS 接近成功”，而是 reset/FMS route 被打回

v14.9 的 factorial causal decomposition 非常硬：

```text
best FMS endpoint = 2/9
FMS + generic transport interaction endpoint = 0/9
FMS-specific source delta mean < 0
generic reset controls endpoint = 6/9 to 7/9
```

这说明当前 FMS 方向没有显示出超越 generic optimizer-state dynamics 的 causal value。更严重的是，`direction removed` 和 `random matched direction` controls 也能达到 3-4/9 endpoint，但 specificity 仍为 0。这意味着：

$$
\boxed{\text{当前 FMS value path 可能并没有提供独立有效 direction。}}
$$

因此，继续 reset / action / mask / controller 是重复 v9/v12 的老错误。

## 2.2 Line D 的结果是真进展：D-CHE 从 substrate repair 进入 official FMS eligibility

v14.9 的 Line D 经过 gate 修正后，出现了一个非常重要的新状态：

```text
D-CHE = 9/9 substrate
D-FOU = 6/9 substrate exploration
D-RBF = 6/9 substrate exploration
D-WAV = 2/9 substrate
```

这说明之前 “Non-RAT basis 全部不可用” 的判断已经过时。D-CHE 不是 functional success，但它是一个新的可测试 substrate。D-CHE 的 step ratio 很低，NLL ratio 好，LineC pass rate 1.0，mean delta 还为正。这使它成为 v14.10 最应该验证的对象。

## 2.3 不能把 D-CHE substrate eligibility 写成 FMS success

D-CHE 9/9 只说明：

```text
D-CHE 可以作为 official FMS proof 的候选 substrate。
```

它不说明：

```text
D-CHE-FMS 成功；
Non-RAT functional 成功；
PureKAN functional update 成功；
S5 达成。
```

v14.10 必须预注册完整 D-CHE-FMS proof protocol，不能在 v14.9 内拼接结果。

---

# 3. 当前核心 blocker

当前 blocker 分成两类。

## 3.1 Functional blocker：FMS causal value 未成立

我们已经证明：

```text
FMS infrastructure 可运行；
FMS / optimizer-state / reset / transport artifacts 完整；
Rational-FMS 曾经有 synthetic / real partial signal；
但 v14.9 difference-in-differences 失败；
generic optimizer-state controls 解释了 endpoint gain。
```

因此，真正 blocker 是：

$$
\boxed{\text{FMS direction / value state 是否真的带来额外 causal value，尚未证明。}}
$$

## 3.2 Substrate blocker：Non-RAT 只有 D-CHE 到 official eligibility

当前基函数状态不是均匀的：

```text
D-CHE:
  已达到 official FMS eligibility，下一步应执行 FMS proof。

D-FOU / D-RBF:
  已打开 exploration，但不够 official；下一步应 harden 到 9/9 或明确 blocker。

D-WAV:
  仍弱，不能进入 FMS proof。

Rational:
  substrate 稳，但 FMS-specific causal claim 被 confound。
```

所以 v14.10 必须 **功能线和 substrate 线分开推进**：

```text
Line F-CHE:
  在 D-CHE 上做 official Non-RAT FMS proof。

Line D-FOU/RBF/WAV:
  继续 substrate hardening，但不进入 official FMS proof。

Line F-RAT:
  不继续 reset route，只做 no-regression / causal reference。
```

---

# 4. v14.10 总体实验目标

v14.10 的目标不是找动作，不是修 reset，不是启动 controller，而是回答三个问题。

## Q1：D-CHE 上的 FMS 是否能打过 same-substrate AdamW / controls？

这是主问题。D-CHE 9/9 substrate 已经具备 official FMS proof eligibility。现在需要测试：

$$
\boxed{\text{D-CHE + FMS} > \text{D-CHE + AdamW / matched controls}}
$$

如果成功，这说明 FMS 不只是 Rational-specific，也不只是 reset confound；如果失败，则说明当前 FMS definition 仍未成立。

## Q2：D-FOU / D-RBF 能否从 6/9 exploration 推到 9/9 eligibility？

D-FOU 和 D-RBF 已经不是 0。v14.10 应继续 harden，但不能把它们直接拉进 FMS proof。

## Q3：Rational-FMS 是否仍有任何 non-reset causal value？

不再做 reset route。Rational 只保留为 causal reference：如果 D-CHE-FMS 成功而 Rational-FMS 不成功，说明 FMS 的适配可能与 basis family 强相关。

---

# 5. Line R：代码 / provenance / anti-regression 审计

## 5.1 目标

确保 v14.10 不重蹈 v9/v12 的 action-search 错误，也不把 substrate eligibility 写成 functional success。

## 5.2 必须审计字段

每个 method / row 必须记录：

```text
uses_label_in_init
uses_y_for_stats
uses_validation_for_direction
uses_test_for_direction
uses_future_for_direction
uses_query_for_direction
uses_linec_for_direction
uses_cep99_for_direction
uses_nll_for_direction
uses_ece_for_direction
uses_auctime_for_direction
uses_dataset_name_branch
uses_seed_specific_scale
is_action_token_extension
controller_executed
promotion_allowed
```

Gate：

```text
所有 uses_*_for_direction 必须为 0；
is_action_token_extension 必须为 0；
controller_executed 必须为 0；
promotion_allowed 只有 S5 后才允许为 1。
```

如果出现任一 violation，route 必须是：

```text
R0-ProvenanceOrActionSearchViolation
```

---

# 6. Line F-CHE：D-CHE official FMS proof

## 6.1 核心假设

D-CHE 的 degree-channel substrate 已经达到 9/9 substrate gate。FMS 可能在 Chebyshev degree-energy coordinate 上比在 Rational denominator/group coordinate 上更容易形成 stable training boundary。

形式上，我们要测试：

$$
\Delta_{CHE-FMS}
=
(\text{D-CHE-FMS} - \text{D-CHE-AdamW})
-
(\text{matched controls} - \text{D-CHE-AdamW}).
$$

只有当：

$$
\Delta_{CHE-FMS} > 0
$$

并且 task / AUC / tail / LineC / efficiency 全部过门，才允许写成 functional success。

## 6.2 Methods

### Baselines

```text
C0-D-CHE-AdamW
C1-D-CHE-AdamW-NoOpMatchedOverhead
C2-D-CHE-AdamW-RandomMatchedNorm
C3-D-CHE-AdamW-AdamWParallelDirectionControl, non-promotable
C4-D-CHE-AdamW-GenericOptimizerStateControl
```

### FMS candidates

```text
F-CHE1-GenericParameterFMS-NoBasisProjection
F-CHE2-DegreeWiseFMS
F-CHE3-DegreeEnergyTrustRegionFMS
F-CHE4-LowDegreeAnchorHighDegreeLateEnableFMS
F-CHE5-ReadoutDegreeDecoupledFMS
F-CHE6-PhaseScheduleDegreeFMS
F-CHE7-ValuePreservingDegreeProjectionFMS
```

注意：这些不是 action token。它们是预注册的 **FMS definition variants**，每个必须有明确的 metric-state / constraint definition，不允许在看到 dataset/seed 结果后新增。

## 6.3 FMS direction 合法信息

允许使用：

```text
current train batch
train-stream per-example gradient
loss-interface generic output cotangent
D-CHE degree-energy telemetry
train-stream logits / margins / entropy, only as train proxy
optimizer state, if not reset-search
```

禁止使用：

```text
validation / test / future / query
LineC / CEp99 / NLL / ECE / AUCtime / Brier audit
labels except through current supervised training loss interface
CE-specific tail formula
dataset name / seed-specific branch
```

## 6.4 Synthetic proof gate

先跑 synthetic X1-X7：

```text
tasks = X1..X7
seeds = 0,1,2
loss_interfaces = CE,Brier
```

S3-CHE synthetic pass：

```text
>= 5 / 7 task families pass
within each passed family: >= 2 / 3 seeds or >= 2 loss interfaces pass
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.25
```

若 S3 不过，不能打开 real short-run。

## 6.5 Real 3x3 proof gate

只有 S3-CHE 过后，才打开：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_steps = 200 / 400 confirmation, if needed
```

S4 exploration：

```text
real_dataset_seed_pass_count >= 6 / 9
promotion_allowed = 0
```

S5 official：

```text
real_dataset_seed_pass_count = 9 / 9
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.25
forbidden audit pass
code review pass
promotion_allowed = 1
```

## 6.6 必须记录的 F-CHE 字段

```text
method
dataset
seed
task_family
loss_interface
source_vs_adamw
source_vs_best_control
AUCtime_ratio
CEp99_delta
NLL_delta
ECE_delta
LineC_pass
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
step_time_ratio
memory_ratio
parameter_update_norm
fms_state_norm
degree_energy_before
degree_energy_after
high_degree_energy_fraction
degree_entropy
degree_gate_active_fraction
degree_projection_rejection_fraction
value_retention_after_degree_projection
cos_projected_vs_generic
control_equivalent_flag
```

---

# 7. Line D：All-basis substrate hardening

Line D 不能因为 D-CHE 进入 proof 就停止。它并行推进，但必须遵守：

```text
Non-RAT 未到 9/9 substrate 不允许 official FMS proof。
```

## 7.1 D-FOU：6/9 exploration to 9/9 eligibility

目标：判断 Fourier low-frequency / no-materialize lifetime 是否能成为第二个 official-eligible substrate。

候选：

```text
D-FOU27-LowFreqIdentityResidualV2
D-FOU28-BandwiseSNRWarmupV2
D-FOU29-PhaseStableBandMixV2
D-FOU30-HighFrequencyQuarantineV2
```

Gate：

```text
full 3x3 substrate >= 6/9 -> exploration remains open
full 3x3 substrate = 9/9 -> official FMS eligible
```

必须记录：

```text
band_energy_low
band_energy_high
high_freq_ratio
phase_drift
bandwise_snr
step_time_ratio
memory_ratio
mean_delta_vs_MLP
NLL_ratio
LineC_pass_rate
```

## 7.2 D-RBF / FastKAN：6/9 exploration to task-health substrate

候选：

```text
D-RBF27-CompactBumpIdentityResidualV2
D-RBF28-ActiveCenterOccupancyRepairV2
D-RBF29-WidthConditionGuardV2
D-RBF30-GaussianLocalK4NoDenseMaterialization
```

目标不是直接 FMS，而是修复 task-health / center occupancy。

必须记录：

```text
center_occupancy_entropy
empty_center_fraction
width_condition
out_of_grid_fraction
active_center_fraction
mean_delta_vs_MLP
NLL_ratio
LineC_pass_rate
```

## 7.3 D-WAV：2/9，低预算继续

Wavelet 当前明显低于 D-CHE/FOU/RBF，不应主投，但不能放弃。

候选：

```text
D-WAV27-TriangularSupportStableV3
D-WAV28-ScaleOccupancyHardeningV3
D-WAV29-LocalTailCoverageGuardV3
D-WAV30-SupportOverlapDampingV3
```

目标：先回到 >=6/9 exploration，否则继续保持低预算。

## 7.4 Rational monitor

Rational 不再跑 reset/optimizer-state route，只做：

```text
D-RAT-Monitor-v1410
```

记录是否出现 regression：

```text
source_vs_adamw
LineC_pass
CEp99_delta
NLL_delta
step_time_ratio
memory_ratio
```

---

# 8. Line M：MLP / generic optimizer control

MLP line 不再作为主突破口，只作为 generic confound control。

比较：

```text
M0-MLP-AdamW
M1-MLP-GenericFMS
M2-MLP-DegreeAnalogFMS, hidden-block analog
M3-MLP-RandomMatchedNorm
M4-MLP-GenericOptimizerStateControl
```

目的：判断若 D-CHE-FMS 有正结果，它是否只是 generic optimizer/FMS，而不是 Chebyshev/KAN-specific。

KAN-specific gain 定义为：

$$
\Delta_{KAN-specific}
=
(\text{D-CHE-FMS}-\text{D-CHE-AdamW})
-
(\text{MLP-FMS}-\text{MLP-AdamW}).
$$

如果：

$$
\Delta_{KAN-specific} \le 0,
$$

则不能 claim KAN-specific functional advantage。

---

# 9. Line C：Manifold-channel audit

Line C 继续做审计，不做方向源。

必须记录：

```text
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
LineC_pass
train_probe_coupling_r2
signal_mass_topk
reservoir_fraction
kernel_drift_proxy
CEp99
NLL
ECE
Brier
margin_p10
```

必须区分：

```text
metric_used_as_direction = 0
metric_used_as_gate = 1
```

如果任一 candidate 使用 LineC / tail / calibration audit 生成方向，则 route：

```text
R0-AuditMetricDirectionViolation
```

---

# 10. 必须生成的 artifact

```text
v1410_route_decision.json
v1410_progress_table.csv
v1410_forbidden_information_audit.csv
v1410_no_action_search_audit.csv
v1410_dche_fms_synthetic_results.csv
v1410_dche_fms_synthetic_summary.csv
v1410_dche_fms_real_results.csv
v1410_dche_fms_real_summary.csv
v1410_dche_fms_controls.csv
v1410_dche_degree_telemetry.csv
v1410_dche_projection_retention.csv
v1410_mlp_generic_controls.csv
v1410_all_basis_substrate_status.csv
v1410_fourier_substrate_hardening.csv
v1410_rbf_substrate_hardening.csv
v1410_wavelet_substrate_hardening.csv
v1410_rational_monitor.csv
v1410_linec_tail_audit.csv
v1410_failure_taxonomy.csv
v1410_required_artifact_manifest.csv
v1410_code_review_packet.zip
v1410_no_go_boundary.md
v1410_next_hypothesis_queue.md
```

---

# 11. 必须生成的可视化

```text
fig_v1410_progress_by_line.svg
fig_dche_synthetic_5of7_heatmap.svg
fig_dche_real_3x3_pass_matrix.svg
fig_dche_source_auc_tail_scatter.svg
fig_dche_degree_energy_before_after.svg
fig_dche_projection_value_retention.svg
fig_dche_fms_vs_controls_diffindiff.svg
fig_mlp_vs_dche_fms_gain.svg
fig_all_basis_substrate_status.svg
fig_fourier_rbf_wavelet_substrate_heatmap.svg
fig_linec_tail_audit_by_method.svg
fig_failure_taxonomy.svg
```

---

# 12. Failure 后 Codex 必须先尝试什么

## Case A：D-CHE synthetic S3 不过

Codex 不能直接新增 F-CHE8。必须先输出：

```text
1. failure 是否来自 source；
2. 是否来自 AUCtime；
3. 是否来自 CEp99/NLL/ECE；
4. 是否来自 LineC；
5. 是否来自 step/memory；
6. 是否被 random / AdamWParallel / generic controls 解释；
7. degree telemetry 是否显示 high-degree energy 失控或过度抑制。
```

允许的预注册 fallback 只有：

```text
F-CHE-FB1-ValuePathOnly
F-CHE-FB2-DegreeConstraintOnly
F-CHE-FB3-GenericFMSPlusDegreeSafetyProjection
```

不允许：

```text
新增 reset action；
新增 controller；
用 LineC/CEp99/NLL/ECE/AUCtime 设计 direction；
按 task family / dataset / seed 调参数。
```

## Case B：D-CHE synthetic 过，但 real 3x3 < 6/9

不允许直接调 real hyperparameter。必须先做 transfer decomposition：

```text
synthetic pass family 对应 real fail dataset-seed 的 feature 是否匹配；
source / AUC / tail / LineC 哪个失败；
FMS value 是否被 controls 解释；
D-CHE degree telemetry 是否 train-stream stable；
real fail 是否与 specific dataset/seed 相关，但不能用其做 branch。
```

允许 fallback：

```text
F-CHE-RT1-TrainSplitAgreement, pre-registered
F-CHE-RT2-ValueRetentionTrust, pre-registered
F-CHE-RT3-DegreeEnergySafetyProjection, pre-registered
```

如果仍 < 6/9，route：

```text
R2-DCHESyntheticDoesNotTransfer
```

## Case C：D-CHE real 6/9 到 8/9，但不到 9/9

不允许降低 gate。必须输出 missing-state table：

```text
dataset
seed
source_fail
AUC_fail
CEp99_fail
NLL_fail
ECE_fail
LineC_fail
step_fail
control_equivalent
```

只有当 missing states 共享同一个 train-stream-visible mechanism 时，才允许下一版计划讨论机制；不能在 v14.10 内临时新增 method。

## Case D：D-CHE S5 过，但 MLP / generic controls 同样解释

route：

```text
R3-GenericFMSConfound
```

结论：FMS 可能是 generic optimizer improvement，不是 KAN/CHE-specific breakthrough。

## Case E：D-FOU / D-RBF 到 9/9

不能自动执行 FMS proof。必须写：

```text
NonRATSubstrateEligibleNoFMSProof
```

然后进入 v14.11 预注册 proof。除非本 v14.10 已经明确定义 corresponding F-FOU / F-RBF method set 和 controls。

---

# 13. Stop / go route 定义

```text
S0-ExecutionCompleteNoPromotion:
  all required artifacts complete, no S3/S4/S5.

S3-DCHESyntheticFMSPass:
  D-CHE synthetic >=5/7 pass, real not yet open or not complete.

S4-DCHRealTransferExplorationPositive:
  real 3x3 >=6/9 but <9/9.

S5-OfficialFunctionalSuccess:
  real 3x3 = 9/9 and all strict gates pass.

R1-DCHESyntheticFMSFail:
  D-CHE synthetic <5/7.

R2-DCHESyntheticDoesNotTransfer:
  synthetic passes but real <6/9.

R3-GenericFMSConfound:
  MLP/generic controls explain gain.

R4-DCHESubstrateRegression:
  D-CHE no longer 9/9 in confirmation.

R5-NonRATSubstrateStillPartial:
  D-FOU/D-RBF/D-WAV fail to reach 9/9.

R0-ProvenanceOrActionSearchViolation:
  any forbidden source, action token, controller, or audit-metric direction violation.
```

---

# 14. 最终判断

v14.10 的核心不是继续修 reset，也不是回到 Rational action search，而是抓住 v14.9 打开的真正新机会：

$$
\boxed{\text{D-CHE 已经成为第一个 9/9 Non-RAT official-FMS-eligible substrate。}}
$$

下一步必须用完整、预注册、control-resistant 的 protocol 回答：

$$
\boxed{\text{D-CHE 上的 FMS 是否有真实 causal value？}}
$$

如果答案是 yes，functional update 仍然是项目突破口，但 basis family 可能从 Rational 转到 Chebyshev。  
如果答案是 no，则当前 FMS definition 需要再次重构，不能再靠 reset、controller、action bank 或局部 positive row 拖延。

