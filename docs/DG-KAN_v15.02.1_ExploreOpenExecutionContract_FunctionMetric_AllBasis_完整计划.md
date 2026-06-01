# DG-KAN v15.2.1：Explore-Open Execution Contract + Function-Metric FU + All-Basis 并行加速完整计划

> 版本：v15.2.1 execution plan  
> 生成日期：2026-05-31（Asia/Singapore）  
> 目标：修正 v15.2 的核心执行问题——虽然写了 exploration continue-open，但实际计划仍容易让 Codex 在 Line G / P / D 单线 fail 后浅尝辄止、直接写 no-go。v15.2.1 不新增 action bank，不新增 controller，不做局部 positive 驱动的 token search，而是把 **探索义务** 写成可执行合同：每条线必须完成预注册 failure-class fallback ladder、最低实验预算、机制分解和 cross-control 后，才允许写 no-go。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no label-informed init；direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只做 audit / gate，不能生成方向。  
> 制度：promotion fail-closed；exploration continue-open；no-go 只能在 **探索合同完成** 后写出。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是寻找某个局部好看的 update token。项目目标是：

$$
\boxed{
\text{在 strict FC-PureKAN base 上，通过 functional update 获得比普通 AdamW/backprop 更好的训练路径和模型。}
}
$$

最终必须证明：

$$
\boxed{
\text{PureKAN base + functional update}
>
\text{PureKAN base + AdamW / strong optimizer controls}
}
$$

并且必须同时满足：

```text
1. 表达力不打折，最好优于同规模 MLP；
2. forward / backward / step / memory 与 MLP 可比；
3. AUC-step / AUC-time 不劣于 MLP / KAN AdamW；
4. source / tail / calibration / LineC 不坏；
5. 收益不能被 AdamW、Cautious、MGUP、second moment、decay、random matched controls 解释；
6. functional update 必须 loss-interface-generic，而不是 CE-specific；
7. functional update 不能退化成 action/token/controller search。
```

## 0.2 当前真实状态

v15.0 / v15.01 已经给出硬结论：

```text
v15.0:
  optimizer-aware FU implemented；
  second moment、cautious alignment、MGUP、SophiaDiag、BlockSecondMoment、
  decoupled decay、SOAPLite diagnostic 均已覆盖；
  但 official FU success = 0。

v15.01:
  CR-FU solver implemented；
  residualizer 已投掉 Adam/Cautious/MGUP/decay + matched-random/same-active/
  same-projection/same-value-retention controls；
  FunctionSpaceProximalResidual 已改为固定 alpha candidates 的 train-B1 objective；
  但 real_lite_pass_count = 1/9；
  source_vs_best_control_mean < 0；
  control_equivalent_fraction 高；
  generic_optimizer_explains_crfu = 1。
```

所以当前结论是：

$$
\boxed{
\text{FU 作为额外 residual direction 没有独立 causal value。}
}
$$

v15.2 试图改成 **gradient-aligned function-metric update**，方向是对的；但 v15.2 的执行问题是：

$$
\boxed{
\text{它仍然太像一次 decisive fork，缺少强制 Codex 深挖的探索合同。}
}
$$

## 0.3 各线当前进展百分数

| 线 | 当前完成度 | 当前判断 |
|---|---:|---|
| 代码 / provenance / finalizer 审计 | 99% | artifact、manifest、forbidden/no-action-search audit 基本闭合 |
| Historical B320/FHQ anchor | 85% frozen | 历史强，但 label-informed init 已禁用，不能作为最终 claim |
| Line C 几何审计 | 88% | 审计稳定，只能做 gate / explanation，不能生成方向 |
| PopRisk / FU infrastructure | 95% | per-example gradient、alignment、second moment、decay、controls、residualizer 均成熟 |
| Current FU / CR-FU causal value | 0%-5% | 额外 direction / residual direction 基本 no-go |
| Function-metric FU | 0%-10% | v15.2 计划定义了方向，但尚未证明，且探索合同不足 |
| D-CHE substrate | 85% | 历史 9/9 eligibility，仍是最强 Non-RAT substrate |
| D-CHE current FU real-lite | 5%-10% | v15.01 real-lite 仅 1/9，不能 promotion |
| Generic MLP / optimizer controls | 50% | generic controls 很强，必须继续作为 confound control |
| Rational substrate | 85% | 稳定，但 reset / optimizer-state route 已被 generic confound 打回 |
| D-FOU substrate | 25%-35% | 历史 6/9 未稳定复现，当前需要 cross-version + fresh hardening |
| D-RBF / FastKAN substrate | 25%-35% | workspace 有进展但 task-health 不稳 |
| D-WAV substrate | 15%-25% | 弱线索，低预算保留 |
| Non-D-CHE official FU proof | 0%-5% | 无 family 达到 FU eligibility |
| Official S5 functional success | 0% | 尚未达成 |
| 整体 next-gen MLP claim | 28%-36% | current FU family 已 no-go，function-metric 是新定义但必须深挖 |

---

# 1. v15.2 的问题：写了 continue-open，但不够可执行

v15.2 的核心方向是：

$$
\Delta\theta_t = -\eta P_{safe}P_{align}M_{func,t}^{-1/2}g_t.
$$

这比旧 FU 更合理，因为它不再让 FU 与 AdamW 竞争方向，而是把 functional information 作为 metric / preconditioner / trust region。

但 v15.2 仍然有三个执行问题。

## 1.1 问题一：Line fail 后没有足够深的同线 fallback ladder

v15.2 规定：

```text
Line G fail；
Line P fail；
Line D fail；
不能 hard stop。
```

但它没有规定：

```text
Line G fail 后，Codex 必须怎么继续；
Line G 是 scale fail、alignment fail、control-equivalent fail、overhead fail、tail fail 时，
分别必须执行哪些预注册分解；
Line P alpha solver 不过时，是否要检查 objective/commit/projection 哪个环节断；
Line D 0/9 时，是否要复核 candidate mismatch / gate mismatch / budget mismatch。
```

因此 Codex 很容易执行完 G0-G8 / P1-P5 后写：

```text
R15_2-CurrentFunctionalDefinitionNoGo
```

这就变成浅尝辄止。

## 1.2 问题二：no-go 触发过早

v15.2 的 Case D 是：

```text
Line G/P/X 全 fail，Line D 也 fail -> R15_2-CurrentFunctionalDefinitionNoGo。
```

这个逻辑太快。它只说明 **method surface** 失败，不说明：

```text
1. failure class 是否被正确识别；
2. implementation / scale / schedule / projection / control-equivalence 是否分清；
3. same mechanism 下是否完成最低深度修复；
4. all-basis 是真实 0/9，还是 candidate/gate/budget mismatch；
5. positive-looking generic controls 是否提示应做 generic optimizer branch。
```

## 1.3 问题三：next_hypothesis_queue 不能只是下一版文档

之前很多计划把 next hypothesis 写在最后，但 Codex 当前轮不会执行。这样会造成：

```text
每轮都“知道了一个问题”；
下一轮再写一个机制；
但当前轮没有继续探索。
```

v15.2.1 要把一部分 next hypothesis 升级成 **本轮必须执行的 fallback ladder**。

---

# 2. v15.2.1 的核心制度：探索合同

## 2.1 Promotion fail-closed

S5 仍然严格，不能降低：

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

## 2.2 Exploration continue-open

以下情况不能 hard stop：

```text
Line G function metric fail；
Line P proximal metric fail；
Line X transfer audit fail；
Line D all-basis fail；
Line M generic controls positive；
D-CHE real-lite <4/9；
D-FOU / D-RBF / D-WAV <6/9；
LineC/tail/AUC fail；
overhead high；
metric not discriminative；
source mean negative；
control-equivalent high。
```

它们只能触发 failure-class fallback。

## 2.3 真正 hard stop

只有这些才 hard stop：

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

## 2.4 No-go 成立条件

No-go 只能在以下条件全部满足后写出：

```text
1. Line R artifact / provenance / forbidden / no-action-search audit 完成；
2. Line G 主方法 + G-fallback ladder 完成；
3. Line P 主方法 + P-fallback ladder 完成；
4. Line X transfer/operator audit 完成；
5. Line D all-basis cross-version reconciliation + fresh hardening 完成；
6. Line M controls 对所有 positive-looking rows 完成；
7. Line C failure taxonomy 完成；
8. next_hypothesis_queue 中“本轮可执行项”已执行或明确 budget-deferred；
9. 每条线写出 exploration_exhaustion_certificate。
```

---

# 3. 关键原则：继续探索，但不重蹈 v9/v12 的 action-search 错误

## 3.1 禁止的探索

Codex 不许做：

```text
1. 新增 FU9/FU10；
2. 新增 F-CHE8/F-CHE9；
3. action bank / controller；
4. reset route；
5. strength/lambda/interval 小网格；
6. 按 dataset/seed fail pattern 写 branch；
7. 用 real audit metric 反推 direction；
8. 局部 positive 后临时造 token；
9. 把 real-lite 写成 promotion；
10. 把 MLP/generic positive 写成 KAN-specific。
```

## 3.2 允许的探索

Codex 允许且必须做：

```text
1. 同一预注册机制内的 failure-class fallback；
2. scale sanity，但只能用全局 fixed small set，不按 dataset/seed 分支；
3. implementation sanity；
4. decomposition；
5. matched controls；
6. cross-version reconciliation；
7. runtime / overhead breakdown；
8. no-regression monitor；
9. generic optimizer branch 记录，但不写 KAN-specific；
10. next hypothesis 中本轮可执行的 deterministic fallback。
```

## 3.3 判断标准

如果一个 fallback 改变了 **机制类别**，它必须在计划里预注册；否则不能执行。

如果一个 fallback 只是为了判断 failure class，例如：

```text
scale too small?
projection kills value?
alignment rejects too much?
metric saturates?
overhead dominates?
candidate mismatch?
```

则必须执行，因为它不是 action search，而是 mechanism diagnosis。

---

# 4. 总体实验设计

v15.2.1 分为八条线，必须并行：

```text
Line R: provenance / no-action-search / implementation readback。
Line G: Gradient-aligned function-metric update + fallback ladder。
Line P: Proximal metric solver + fallback ladder。
Line X: Exact transfer / response operator audit。
Line D: All-basis substrate acceleration + cross-version reconciliation。
Line M: MLP / generic optimizer controls。
Line C: LineC / tail / calibration / failure taxonomy audit。
Line Z: route / no-go / exploration exhaustion certificate。
```

并行执行建议：

```text
GPU group 0:
  Line G D-CHE function-metric update + G fallback ladder。

GPU group 1:
  Line P proximal metric solver + P fallback ladder。

GPU group 2:
  Line D D-FOU / D-RBF substrate acceleration。

GPU group 3:
  Line D D-WAV + Rational no-regression + D-CHE no-regression。

GPU group 4 if available:
  Line X transfer operator audit on completed G/P rows。
```

---

# 5. Line R：实现与审计

## 5.1 目标

确保本轮不是 action search，也不是 audit-direction leak。

## 5.2 必须输出

```text
v1521_route_decision.json
v1521_progress_table.csv
v1521_required_artifact_manifest.csv
v1521_forbidden_information_audit.csv
v1521_no_action_search_audit.csv
v1521_method_surface_manifest.csv
v1521_code_review_packet.zip
```

## 5.3 额外审计字段

```text
new_fu_token_count
new_fche_token_count
action_bank_used
controller_executed
reset_route_used
dataset_name_branch_used
seed_specific_scale_used
audit_metric_used_for_direction
validation_test_future_query_used_for_direction
linec_used_for_direction
tail_metric_used_for_direction
dynamic_method_generation_count
exploration_fallback_executed_count
exploration_exhaustion_certificate_written
```

---

# 6. Line G：Gradient-Aligned Function-Metric Update

## 6.1 目标

验证 KAN function/basis information 是否能作为 **metric** 改善普通 gradient update。

普通梯度：

$$
g_t=\nabla_\theta \mathcal L_{train}(\theta_t).
$$

function/basis metric：

$$
M_{func,t}=\operatorname{diag}(m_{func,t})
$$

或 role/block form：

$$
M_{func,t}^{(r)}=\bar m_{r,t}I.
$$

总 metric：

$$
M_{total,t}
=
\sqrt{v_t}
+
\lambda_f m_{func,t}
+
\lambda_c m_{curv,t}
+
\epsilon.
$$

更新：

$$
u_t=-\frac{g_t}{M_{total,t}}.
$$

alignment gate：

$$
q_i=1\{u_{t,i}(-g_{t,i})>0\}.
$$

最终：

$$
\Delta\theta_t=q\odot u_t+\Delta\theta_{decoupled}.
$$

## 6.2 主方法

```text
G0-D-CHE-AdamW
G1-D-CHE-CautiousAdamW
G2-D-CHE-MGUP
G3-D-CHE-AdamVFunctionMetric
G4-D-CHE-DegreeRoleFunctionMetric
G5-D-CHE-OutputJacobianDiagMetric
G6-D-CHE-GradientSNRMetric
G7-D-CHE-CurvatureClippedFunctionMetric
G8-D-CHE-CombinedAdamV_FunctionMetric_Alignment
```

## 6.3 Line G failure-class fallback ladder

Line G 主方法跑完后，不允许直接 no-go。必须执行以下 fallback ladder。

### G-FB1：Scale sanity，不是网格搜索

目的：判断 metric effect 是否太小或太大。

只允许全局固定三档：

```text
lambda_f in {0.25, 1.0, 4.0}
```

仅对 top-2 main methods 执行，不按 dataset / seed 分支。

记录：

```text
metric_effect_norm
update_norm_ratio_vs_adamw
alignment_keep_fraction
source_vs_best_control
control_equivalent_fraction
bad_event_fraction
```

若三档均 control-equivalent，记录：

```text
G-Fail-MetricScaleNotCause
```

### G-FB2：Projection-kills-value audit

目的：判断 basis-safe projection 是否杀掉 value。

对 top-2 main methods 比较：

```text
NoProjectionDiagnostic
ProjectionAuditOnly
ProjectionCommit
```

注意：

```text
NoProjectionDiagnostic 不能 promotion；
只能判断 projection 是否杀掉 source。
```

记录：

```text
pre_projection_source_proxy
post_projection_source
value_retention_after_projection
projection_rejection_fraction
cos_projected_vs_unprojected
```

### G-FB3：Alignment rejection audit

目的：判断 cautious / alignment gate 是否过强。

比较：

```text
HardAlignment
SoftAlignment025
SoftAlignment050
RoleAlignmentOnly
NoAlignmentDiagnostic
```

注意：

```text
NoAlignmentDiagnostic 不可 promotion。
```

记录：

```text
anti_alignment_fraction
keep_fraction
rolewise_keep_fraction
source_vs_best_control
tail_fail_delta
LineC_fail_delta
```

### G-FB4：Metric degeneracy audit

目的：判断 metric 是否近似常数、退化为 AdamV 或被 controls 解释。

记录：

```text
metric_mean
metric_std
metric_p10
metric_p90
metric_entropy
corr_metric_with_adam_v
corr_metric_with_grad_snr
corr_metric_with_degree_energy
```

若：

```text
metric_std / metric_mean < 0.05
```

则 route:

```text
G-Fail-MetricDegenerateConstant
```

### G-FB5：Continuous boundary schedule probe

目的：判断 single-event function metric 是否不适合，是否必须持续边界。

只允许两个预注册 schedule：

```text
ScheduleEarlyWeakLateStrong
ScheduleConstantLowAmplitude
```

不得按 dataset/seed 调度。

### G-FB6：G-line exhaustion certificate

只有 G0-G8 + G-FB1..G-FB5 完成后，才允许写：

```text
G-LineExhausted = 1
```

---

# 7. Line P：Proximal Metric Solver

## 7.1 目标

验证是否存在固定低秩 function-space proximal metric，比 direct metric 更可靠。

形式：

$$
\alpha^*
=
\arg\min_\alpha
\mathcal L_{B1}(\theta+U\alpha)
+
\rho\|\alpha\|_2^2
+
\tau\|U\alpha\|_{M_t}^2.
$$

固定 alpha candidates：

```text
0, 0.025, 0.05, 0.10, 0.20
```

## 7.2 主方法

```text
P0-D-CHE-AdamW
P1-D-CHE-AdamWSubspaceProx
P2-D-CHE-PerExampleGradLowRankProx
P3-D-CHE-DegreeRoleProx
P4-D-CHE-RandomSketchFunctionProx
P5-D-CHE-AdamWPlusFPUResidual
```

## 7.3 Line P fallback ladder

### P-FB1：Objective decomposition

记录每个 alpha：

```text
B1_loss_delta
B2_loss_delta
prox_penalty
trust_penalty
alpha_selected
alpha_selected_frequency
NoOp_selected_frequency
```

若 NoOp 选择率高于 0.80，route:

```text
P-Fail-ProximalObjectivePrefersNoOp
```

### P-FB2：Subspace quality audit

比较 U 的来源：

```text
AdamWSubspace
GradLowRankSubspace
DegreeRoleSubspace
RandomSketchSubspace
```

记录：

```text
subspace_rank
subspace_grad_capture_fraction
subspace_condition
subspace_overlap_with_adamw
subspace_overlap_with_controls
```

### P-FB3：B1/B2 mismatch audit

如果 B1 改善但 B2 不改善，说明 overfit current train split。

记录：

```text
B1_improve_count
B2_improve_count
B1_B2_disagreement_rate
```

若 disagreement > 0.60，route:

```text
P-Fail-TrainSplitOverfit
```

### P-FB4：Commit effect audit

比较：

```text
selected alpha but no commit
commit alpha
random alpha same norm
same subspace random alpha
```

判断是否真正来自 proximal objective。

### P-FB5：P-line exhaustion certificate

只有 P0-P5 + P-FB1..P-FB4 完成后，才允许写：

```text
P-LineExhausted = 1
```

---

# 8. Line X：Exact Transfer / Response Operator Audit

## 8.1 目标

避免再次忽略 v9 教训：local / synthetic / B1 positive 不等于真实 functional value。

Line X 不生成方向，只审计。

## 8.2 必须记录

```text
train_motion_norm
probe_motion_norm
train_probe_ridge_R2
transfer_operator_condition
signal_channel_energy
reservoir_energy
noise_leakage_proxy
kernel_drift_norm
delta_logit_RMS
source_vs_best_control
```

## 8.3 X-fallback

如果 G/P 有 positive-looking rows，但 X 不支持，必须写：

```text
X-Fail-LocalPositiveNoTransfer
```

不能进入 promotion。

---

# 9. Line D：All-Basis Substrate Acceleration

## 9.1 目标

不能把 D-CHE 作为唯一 carrier。继续推进：

```text
D-FOU
D-RBF / FastKAN
D-WAV
Rational monitor
D-CHE no-regression
```

## 9.2 D-FOU fallback ladder

主候选：

```text
D-FOU47-LowFreqIdentityResidualV5
D-FOU48-BandwiseSNRWarmupV3
D-FOU49-PhaseStableBandMixV3
D-FOU50-NoMaterializeLifetimeV3
D-FOU51-HighFreqQuarantineV2
```

如果全部 <6/9，必须执行：

```text
D-FOU-FB1 Candidate lineage replay from v14.9 6/9 rows
D-FOU-FB2 Gate mismatch audit
D-FOU-FB3 Budget mismatch audit
D-FOU-FB4 Low-frequency-only task-health hardening
D-FOU-FB5 Phase drift / band occupancy failure decomposition
```

## 9.3 D-RBF / FastKAN fallback ladder

主候选：

```text
D-RBF45-ActiveCenterOccupancyV3
D-RBF46-WidthConditionGuardV3
D-RBF47-CompactBumpNoDenseV3
D-RBF48-IdentityResidualV3
D-RBF49-GaussianLocalK4TaskHealth
```

如果全部 <6/9，必须执行：

```text
D-RBF-FB1 Center occupancy collapse audit
D-RBF-FB2 Width condition collapse audit
D-RBF-FB3 Identity residual ablation
D-RBF-FB4 FastKAN-style Gaussian local support replay
D-RBF-FB5 Task-health vs workspace conflict decomposition
```

## 9.4 D-WAV fallback ladder

主候选：

```text
D-WAV41-TriangularSupportV5
D-WAV42-ScaleOccupancyV4
D-WAV43-SupportOverlapDampingV3
D-WAV44-LocalTailCoverageAudit
```

如果全部 <6/9，必须执行：

```text
D-WAV-FB1 Scale occupancy audit
D-WAV-FB2 Support overlap audit
D-WAV-FB3 Low-scale only replay
D-WAV-FB4 Local-tail coverage decomposition
```

## 9.5 D-line gates

Exploration substrate gate：

```text
family_dataset_seed_pass_count >= 6/9
step_ratio <= 1.75
memory_ratio <= 1.75
mean_delta >= -0.05
worst_delta >= -0.10
LineC_pass_rate >= 0.30
```

Official FU eligibility：

```text
family_dataset_seed_pass_count = 9/9
step_ratio <= 1.50
memory_ratio <= 1.50
mean_delta >= -0.02
worst_delta >= -0.05
LineC_pass_rate >= 0.60
```

Line D fail 不能 hard stop Line G/P/X。

## 9.6 D-line exhaustion certificate

必须对每个 family 写：

```text
family_exhaustion_certificate
```

字段：

```text
main_candidates_executed
fallbacks_executed
historical_replay_checked
gate_mismatch_checked
budget_mismatch_checked
true_non_reproducibility
next_kernel_hypothesis
```

---

# 10. Line M：MLP / Generic Optimizer Controls

## 10.1 目标

每一个 positive-looking KAN result 必须跑 matched generic controls。

## 10.2 必须比较

```text
MLP-AdamW
MLP-CautiousAdamW
MLP-MGUP
MLP-AdamVFunctionMetricAnalog
MLP-FunctionSpaceProxAnalog
MLP-RandomMatchedNorm
MLP-SameActiveFraction
MLP-NoOpMatchedOverhead
```

## 10.3 Positive-looking 定义

若某 KAN row 满足任一：

```text
source_vs_best_control > 0
real_lite_pass = 1
LineC improves
tail improves
```

则必须进入 Line M matched controls。

若 MLP/generic controls 解释收益：

```text
route = R-M-GenericOptimizerExplainsGain
KAN-specific claim = 0
```

---

# 11. Line C：Audit only

Line C 只记录，不生成方向。

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
source_vs_control
AUCtime_ratio
LineC_fail_reason
```

禁止：

```text
LineC hard target direction
CE-tail direction
AUCtime direction
calibration-target direction
```

---

# 12. 必须生成的 artifacts

```text
v1521_route_decision.json
v1521_progress_table.csv
v1521_required_artifact_manifest.csv
v1521_forbidden_information_audit.csv
v1521_no_action_search_audit.csv
v1521_method_surface_manifest.csv
v1521_code_review_packet.zip

v1521_line_g_function_metric_results.csv
v1521_line_g_function_metric_controls.csv
v1521_line_g_fallback_results.csv
v1521_line_g_exhaustion_certificate.csv

v1521_line_p_proximal_results.csv
v1521_line_p_proximal_controls.csv
v1521_line_p_fallback_results.csv
v1521_line_p_exhaustion_certificate.csv

v1521_line_x_transfer_operator_audit.csv
v1521_line_x_signal_reservoir_noise.csv
v1521_line_x_kernel_drift.csv

v1521_line_d_allbasis_substrate_results.csv
v1521_line_d_family_summary.csv
v1521_line_d_substrate_failure_table.csv
v1521_line_d_family_exhaustion_certificates.csv

v1521_line_m_mlp_generic_controls.csv
v1521_line_c_linec_tail_audit.csv
v1521_no_go_boundary.md
v1521_next_hypothesis_queue.md
```

---

# 13. 必须可视化

```text
fig_v1521_progress_by_line.svg
fig_v1521_exploration_depth_by_line.svg
fig_v1521_function_metric_vs_controls.svg
fig_v1521_g_fallback_ladder.svg
fig_v1521_p_fallback_ladder.svg
fig_v1521_alignment_keep_fraction_by_role.svg
fig_v1521_metric_distribution_by_role.svg
fig_v1521_real_lite_pass_heatmap.svg
fig_v1521_source_vs_control_scatter.svg
fig_v1521_transfer_r2_vs_source.svg
fig_v1521_noise_leakage_vs_source.svg
fig_v1521_allbasis_substrate_matrix.svg
fig_v1521_allbasis_family_exhaustion.svg
fig_v1521_task_efficiency_pareto.svg
fig_v1521_failure_taxonomy_heatmap.svg
fig_v1521_step_time_breakdown.svg
```

---

# 14. 最终 route 规则

## 14.1 Success routes

### S2-FunctionMetricExplorationPositive

```text
real_lite_pass_count >= 3/9
source_vs_best_control_mean > 0
control_equivalent_fraction <= 0.60
```

### S3-FunctionMetricMeaningful

```text
real_lite_pass_count >= 4/9
source_vs_best_control_mean >= 0.005
control_equivalent_fraction <= 0.50
```

### S4-RealTransferExploration

```text
real_lite_pass_count >= 6/9
AUCtime median <= 1.05
tail fail reduced vs AdamW
LineC fail <= AdamW
```

### S5-Official

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
promotion_allowed = 1
```

## 14.2 No-go routes

No-go routes require exhaustion certificates.

```text
R-G-FunctionMetricExhausted:
  G-LineExhausted = 1 and no G success.

R-P-ProximalMetricExhausted:
  P-LineExhausted = 1 and no P success.

R-X-LocalPositiveNoTransfer:
  positive G/P rows exist but X shows no train-probe transfer.

R-D-AllBasisSubstrateExhausted:
  all family exhaustion certificates complete and no family >=6/9.

R15_2_1-CurrentFunctionalDefinitionNoGo:
  R-G + R-P + R-X + R-D all satisfied,
  Line M controls explain positives,
  Line Z no remaining executable fallback.
```

---

# 15. Codex 执行要求

Codex 必须：

```text
1. 不早停。
2. 每条线都写 progress 表。
3. 每条线都写 fallback 执行状态。
4. 每条线都写 exhaustion certificate。
5. 每个 positive-looking row 都跑 matched controls。
6. 每个 fail 都写 failure class。
7. Line Z 必须区分：
   - promotion no-go；
   - exploration no-go；
   - budget-deferred；
   - implementation blocker；
   - theoretical no-go。
```

Codex 不许：

```text
1. 写“all failed”但没有 fallback ladder；
2. 写 no-go 但没有 exhaustion certificate；
3. 因 Line G fail 跳过 Line D；
4. 因 Line D fail 跳过 Line G/P；
5. 因 controls positive 停止所有线；
6. 把 next_hypothesis_queue 当成已执行探索；
7. 新增 action/token/controller；
8. 降低 official gate；
9. 用 audit metric 做方向。
```

---

# 16. 最终一句话

v15.2 的方向是对的，但执行合同太弱，容易让 Codex 浅尝辄止。

v15.2.1 的核心修正是：

$$
\boxed{
\text{不放宽 promotion gate；但把 exploration 写成必须完成的 fallback ladder。}
}
$$

这样既不会重蹈 v9/v12 的 action-search 错误，也不会让 Codex 一看到主 gate fail 就停止。