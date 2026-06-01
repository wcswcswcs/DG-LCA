# DG-KAN v12.35：All-Basis Substrate Health + Basis-Specific Functional Co-location 完整计划

> 版本：v12.35 execution plan  
> 基于：v12.34.2 `AllBasisSubstrateFunctionalRepair` 真实复盘  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；不使用 label-informed initialization；functional direction 不使用 label、CE vector、permuted-label CE、validation/test、future outcome、query batch 或 LineC hard target；CE / NLL / ECE / CEp99 只能作为审计与坏化约束。

---

# 0. 项目总目标与当前进展

## 0.1 总目标

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找到某个只在一两个诊断指标上好看的更新规则。项目目标是：

$$
\boxed{
\text{构建 label-free strict FC-PureKAN efficient substrate，并通过 loss-agnostic functional update 得到比 ordinary backprop / AdamW controls 更好的训练几何和模型。}
}
$$

最终成功至少要同时满足：

```text
1. strict FC-PureKAN；不依赖 label-informed init；不引入 non-KAN learnable path。
2. forward / backward / step / memory 与 MLP 可比。
3. 表达力不打折，task / AUC / calibration 不靠慢训练或坏 tail 换来。
4. LineC / Manifold-Channel geometry 健康：CouplingR2、NoiseSignalLeak、RealSignalReservoirRatio、tail、ECE 不坏。
5. functional update 的收益必须 loss-agnostic、precommit、control-resistant，不能用 CE/label 设计方向。
6. 最终要证明：base/substrate + functional update > same base/substrate + ordinary AdamW / strong controls。
```

## 0.2 当前项目状态

v12.34.2 的真实结论是：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R5-MLPFunctionalNoGo_BasisSpecificOnly also applies
minimum_success = S1-AnyFamilySubstratePass
official_success_reached = 0
promotion_allowed = 0
basis_workspace_rows = 504
basis_workspace_pass_count = 172
basis_workspace_strong_pass_count = 0
substrate_gate_pass_count = 11
substrate_near_pass_count = 15
healthy_base_gate_pass_count = 0
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
mlp_functional_mj_pass_rows = 0
```

这意味着：

```text
1. v12.34.2 只达到 S1：至少存在 active basis substrate。
2. 没有达到 S2/S3：basis-specific functional P3/P4 都是 0 pass。
3. 没有达到 S5：没有 official functional success。
4. MLP M-J closure 仍然 no-go，因此 generic MLP functional 不是当前可依赖主线。
5. all-basis substrate map、family telemetry、basis-specific repair、Non-RAT lifetime fallback、focused Fourier P3 和 required artifact 审计都已执行。
```

## 0.3 当前关键变化

v12.34.2 之后，我们不应再继续：

```text
1. 只扩大 Rational alias 或 B-RAT objective token；
2. 只扩大 Fourier B-FOU functional token；
3. 继续 M-J 类 MLP generic objective；
4. 把 CEp99 / NLL / ECE 当作 direction source；
5. 把 foreach-off 打开的 lifetime 当作健康 substrate。
```

新的判断是：

$$
\boxed{
\text{每个 active basis 先成为 efficient substrate；如果 substrate 不健康，functional 可以修，但必须有可观测、可执行、打过 controls 的 basis-specific channel co-location mechanism。}
}
$$

---

# 1. 各条线当前进展百分比

这些百分比不是 artifact 中的官方数值，而是根据 gate 通过情况、机制清晰度、可复现性、代码审计完整度和距离 official success 的综合估计。

| 线 | 当前完成度 | 状态判断 |
|---|---:|---|
| Line R：代码 / provenance / finalizer 审计 | 97% | artifacts、required manifest、code packet、provenance clean，已很成熟。 |
| Historical FHQ / B320-current 工程能力 | 85% frozen | 历史上强，但 label-informed init 已禁止，不能 official。 |
| Label-free FHQ / A-DYN monitor | 26% | 仍无 label-free near-anchor；继续低预算 monitor。 |
| Line C：Manifold-Channel 几何审计 | 87% | 审计工具稳定，但不能作为 functional direction source。 |
| Precommit / loss-agnostic value source | 20% | MLP M-J no-go，basis-specific source 仍未成形。 |
| Generic MLP functional update | 23% | 线已经建立，但当前 observable family v2 no-go。 |
| KAN / basis-specific functional official | 8% | P3/P4 pass 仍为 0，尚未 work。 |
| Classic no-BSpline portfolio 总体 | 67% | substrate map 完成；Rational 为唯一稳定 substrate family。 |
| Rational family | 84% | 11 substrate pass、15 near substrate 主要来自 Rational；但 healthy base / functional repair 未过。 |
| Chebyshev family | 46% | 当前 all-basis map 中无 substrate pass；incremental lifetime 与 task-health 未闭合。 |
| Fourier family | 49% | foreach-off 可打开部分 lifetime，focused P3 0 pass；LineC/CouplingR2 没有 gain。 |
| RBF / FastKAN family | 45% | foreach-off RBF workspace pass，但 task/AUC/LineC 全面坏化。 |
| Wavelet family | 42% | 部分 lifetime 迹象，但 task/AUC/LineC 坏化，未成 substrate。 |
| 整体 next-gen MLP claim | 51%-55% | substrate 有进展，functional 和 healthy base 仍是大缺口。 |

---

# 2. v12.34.2 结果独立分析

## 2.1 有进展，但不是 S5

v12.34.2 不是没有进展。它做对了三件事：

```text
1. 不再只看 Rational，而是把 Rational / Chebyshev / Fourier / RBF / Wavelet 全部纳入 substrate map。
2. 引入 basis-specific functional P3 runner，并用 NoOp / Random / AdamWParallel / SNR controls 审计。
3. 对 Non-RAT 做了 foreach-off lifetime fallback 和 focused Fourier P3，不是只停在 initial workspace fail。
```

但是它没有达成核心成功：

```text
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
mlp_functional_mj_pass_rows = 0
healthy_base_gate_pass_count = 0
```

所以 v12.34.2 的正确定位是：

$$
\boxed{
\text{S1 reached: substrate exists; S2/S3/S5 not reached.}
}
$$

## 2.2 all-basis substrate map 的真实含义

执行规模：

```text
candidates = 44
families = D-RAT,D-CHE,D-FOU,D-RBF,D-WAV
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
workspace_rows = 396 initial, 504 after fallback/merge
```

核心结论：

```text
Rational 是唯一稳定打开 workspace / substrate 的 family。
D-CHE / D-FOU / D-RBF / D-WAV 在 initial substrate map 中 workspace pass = 0。
strong workspace pass = 0，说明没有 healthy final substrate。
```

这不是说 Non-RAT 数学上失败，而是说明当前实现/测量口径下，它们还不能成为可靠 substrate。

## 2.3 Non-RAT foreach-off 是真实进展，但不是健康 substrate

foreach-off fallback 后：

```text
RBF: 18/18 workspace pass
Fourier: 12/45 workspace pass
Wavelet: 5/18 workspace pass
Chebyshev: 0 workspace pass
```

但 task/AUC/LineC 全面坏化。例如 Fourier rows 的 mean_delta_vs_MLP 约 `-0.314`，AUC_time 到 `4.3-7.0`，LineC 是 `0/12`；RBF rows mean_delta 约 `-0.454`，LineC 只有 `14/27`；Wavelet rows mean_delta 约 `-0.547`，LineC `0/15`。

这说明：

$$
\boxed{
\text{打开 lifetime 不等于得到 substrate。}
}
$$

foreach-off 降低了一些系统压力，但破坏了 task-health 和 LineC geometry。下一步不能把 foreach-off 当作默认修复；它只是暴露 lifetime / optimizer-state overlap 的诊断工具。

## 2.4 basis-specific functional P3 的失败不是 tail，而是 channel co-location

Rational P3：

```text
base_candidates = D-RAT34, D-RAT28, D-RAT26
functional_candidates = B-RAT1..B-RAT5
p3_rows = 135
p3_pass_rows = 0
p4_pass_rows = 0
```

Focused Fourier P3：

```text
base_candidates = D-FOU12,D-FOU14,D-FOU16
functional_candidates = B-FOU1..B-FOU5
p3_executed_rows = 60
p3_pass_rows = 0
p4_pass_rows = 0
```

关键失败不是 CEp99 tail，而是：

```text
CouplingR2_no_gain
NoiseSignalLeak_no_drop
RealSignalReservoirRatio_no_drop
LineC_not_preserved
```

也就是说，functional update 不是简单缺一个 tail-safe rule，而是没有改变 signal/reservoir/noise channel 的有效方向。

## 2.5 MLP functional closure 的意义

MLP M-J1..M-J3：

```text
mlp_functional_candidate_rows = 81
mlp_functional_control_rows = 486
mlp_functional_linec_rows = 810
mlp_functional_mj_pass_rows = 0
mlp_functional_no_go_current_family_v2 = 1
```

这支持一个重要判断：

$$
\boxed{
\text{当前 loss-agnostic functional update 不是一个容易泛化到 MLP 的通用 optimizer trick。}
}
$$

如果 functional 后续能 work，更可能需要依赖 explicit basis coordinate 和 basis-specific telemetry，而不是继续 MLP-generic objective。

---

# 3. 当前核心 blocker

当前 blocker 不再是“有没有 substrate”，而是更精确的三层问题。

## 3.1 Rational：有 substrate，但 functional 不能修

Rational 现在已经是唯一稳定 substrate family，但它还没有 healthy base，也没有 basis-specific functional P3/P4。当前问题是：Rational telemetry 有了，但 B-RAT1..B-RAT5 没有把 telemetry 转换成真正的 channel co-location improvement。

核心假设：

$$
\boxed{
\text{Rational functional repair 不能只控制 denominator / derivative，它必须控制 readout-rational coupling 对 LineC projector 的影响。}
}
$$

## 3.2 Non-RAT：lifetime 和 task-health 分裂

Chebyshev / Fourier / RBF / Wavelet 的问题各不相同：

```text
Chebyshev:
  incremental lifetime 仍卡住，没有 substrate pass。

Fourier:
  foreach-off 可以打开部分 lifetime，但 LineC=0/3 focused P3，CouplingR2 没有 gain。

RBF/FastKAN:
  foreach-off workspace pass，但 task/AUC 大幅坏化。

Wavelet:
  部分 lifetime 迹象，但 task/AUC/LineC 不健康。
```

核心假设：

$$
\boxed{
\text{Non-RAT 不缺少“更激进的 functional”，而缺少能保住 task-health 的 substrate path。}
}
$$

## 3.3 Functional：没有 basis-channel co-location value source

当前 P3/P4 为 0 的根因是：functional direction 没有同时做到：

```text
source_vs_control positive；
CouplingR2 gain；
NoiseSignalLeak drop；
RealSignalReservoirRatio drop；
LineC preservation；
tail/calibration non-harm。
```

所以 v12.35 不应该继续“扩 B-RAT/B-FOU/M-J token”，而要建立 **basis response dictionary + channel co-location solver**。

---

# 4. v12.35 核心假设

## H1：每个 basis family 可以先成为 substrate，不要求 self-healthy

Gate S substrate 只要求：

```text
workspace / step / memory 过；
expression 不灾难；
task 有最低可训练性；
LineC 不灾难；
没有 label/CE/provenance violation。
```

Healthy base gate 可以留给后续，但 substrate gate 不能太低，否则 functional 无落点。

## H2：Functional repair 必须 basis-specific，而不是通用 MLP trick

MLP M-J no-go 后，下一步 functional 预算应转向：

```text
Rational tangent/readout coupling；
Chebyshev degree-energy channel；
Fourier frequency-band channel；
RBF center/width occupancy channel；
Wavelet scale/support channel。
```

## H3：basis-specific repair 必须先建 response dictionary

不能继续直接构造 B-RAT/B-FOU token 然后看 P3。必须先做：

$$
\text{basis actuator} \rightarrow \Delta \text{telemetry} \rightarrow \Delta \text{LineC} \rightarrow \Delta \text{task/tail}
$$

否则每轮都会得到“有些 row task positive，有些 row LineC positive，但不同位”。

## H4：Non-RAT lifetime 修复不能破坏 task-health

foreach-off 是诊断，不是 solution。必须证明：

$$
\text{lifetime repair} \not\Rightarrow \text{task/AUC/LineC collapse}
$$

否则即使 workspace pass，也不能进入 substrate gate。

---

# 5. 实验总流程

v12.35 分为 8 条线。它不是继续扩大局部 token，而是重建 substrate-health 和 channel co-location。

```text
Line R: code/provenance/implementation readback
Line S: substrate-health map v2
Line Q: basis response dictionary
Line B: basis-specific channel co-location functional repair
Line N: Non-RAT task-health substrate repair
Line M: MLP functional closure final monitor
Line C: Manifold-Channel audit
Line Z: finalizer / route / no-go boundary
```

---

# 6. Line R：代码与实现审查

## 6.1 目标

确认 v12.35 没有把 proxy alias、diagnostic response 或 CE/label 信息混入 official direction。

## 6.2 必须记录

```text
file_path
symbol
line_range
candidate_family
uses_label
uses_ce_vector
uses_validation_or_test
uses_linec_hard_target
uses_query_batch
uses_dataset_name_branch
uses_proxy_alias
exact_kernel_implemented
mapped_to_existing_primitive
functional_direction_source
control_direction_source
```

## 6.3 Gate

任何 official candidate 必须满足：

```text
uses_label = 0
uses_ce_vector = 0
uses_validation_or_test = 0
uses_linec_hard_target = 0
uses_query_batch = 0
uses_dataset_name_branch = 0
provenance_violation = 0
```

如果 Codex 找不到核心实现文件和 line range，则 route 必须为：

```text
R0-CodeReviewSurfaceIncomplete
```

---

# 7. Line S：Substrate-health map v2

## 7.1 目标

把 substrate gate 从单纯 workspace 改成 **workspace + minimum task-health + LineC non-catastrophic**。

## 7.2 Candidate family

```text
Rational: D-RAT34..D-RAT39 plus one new D-RAT40 response-ready substrate.
Chebyshev: D-CHE16..D-CHE19 plus one lifetime-health candidate.
Fourier: D-FOU16..D-FOU19 plus frequency-stable substrate.
RBF/FastKAN: D-RBF12..D-RBF16 plus compact-capacity substrate.
Wavelet: D-WAV11..D-WAV15 plus support-stable substrate.
```

## 7.3 Metrics

```text
workspace_raw_ratio
workspace_incremental_ratio
step_ratio
strong_workspace_pass
expression_smoke_pass
mean_delta_vs_MLP
worst_delta_vs_MLP
AUC_time_ratio_vs_MLP
CEp99_delta_vs_MLP
NLL_delta_vs_MLP
ECE_delta_vs_MLP
LineC_pass_count
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
substrate_gate_pass
substrate_near_pass
healthy_base_gate_pass
```

## 7.4 Gate S：Efficient substrate

A candidate passes substrate gate if:

$$
workspace\_raw\_ratio \le 1.05,
$$

$$
workspace\_incremental\_ratio \le 1.75,
$$

$$
step\_ratio \le 1.75,
$$

$$
mean\_delta\_vs\_MLP \ge -0.05,
$$

$$
worst\_delta\_vs\_MLP \ge -0.10,
$$

$$
AUCtime\_ratio\_vs\_MLP \le 2.0,
$$

and:

$$
LineC\_pass\_rate \ge 0.30.
$$

This is intentionally weaker than healthy base gate but stronger than workspace-only. It prevents treating a fast but collapsed substrate as usable.

## 7.5 Gate H：Healthy base

A candidate passes healthy base gate if:

$$
mean\_delta\_vs\_MLP \ge 0,
$$

$$
worst\_delta\_vs\_MLP \ge -0.01,
$$

$$
AUCtime\_ratio\_vs\_MLP \le 1.0,
$$

$$
CEp99\_delta\_vs\_MLP \le 0.05,
$$

$$
LineC\_pass\_rate \ge 0.80.
$$

## 7.6 Failure actions

```text
If workspace pass but task/AUC/LineC collapse:
  Do not run official functional P3.
  Move to Line N task-health repair first.

If task positive but LineC collapse:
  Run Line Q response dictionary to identify geometry-risk channels.

If LineC positive but task collapse:
  Run role/readout coupling audit; check whether basis channel is decoupled from classifier readout.

If workspace fail only:
  Run lifetime waterfall; do not modify task recipe first.
```

---

# 8. Line Q：Basis response dictionary

## 8.1 目标

建立每个 basis family 的可执行响应字典：

$$
\Delta u_j \rightarrow
(\Delta telemetry, \Delta LineC, \Delta task, \Delta tail, \Delta controls)
$$

这里 $\Delta u_j$ 是一个小的、loss-agnostic、precommit actuator，不使用 CE/label。

## 8.2 Family-specific telemetry

### Rational

```text
den_p01
den_p99
r_prime_p95
r_double_prime_p95
tangent_condition
tangent_top_eigen_share
group_function_diversity
readout_rational_coupling
per_group_update_norm
```

### Chebyshev

```text
degree_energy_k
high_degree_energy_ratio
recurrence_max_abs
degree_tangent_condition
degree_top_eigen_share
```

### Fourier

```text
frequency_band_energy
high_freq_energy_ratio
phase_stability
sincos_amplitude_norm
frequency_noise_leak_proxy
```

### RBF / FastKAN

```text
center_occupancy_entropy
dead_center_fraction
width_p01
width_p99
out_of_grid_fraction
center_gradient_balance
```

### Wavelet

```text
scale_energy
support_occupancy
support_overlap
scale_dead_fraction
local_tail_coverage
```

## 8.3 Required output

```text
v1235_basis_response_dictionary.csv
v1235_basis_response_controls.csv
v1235_basis_response_linec.csv
v1235_basis_response_task_tail.csv
v1235_response_colocation_summary.csv
```

## 8.4 Response dictionary gate

A family has a usable response dictionary if at least one actuator family satisfies:

$$
\Delta CouplingR^2 \ge 0.01,
$$

$$
\Delta NoiseSignalLeak \le 0.0,
$$

$$
\Delta RealSignalReservoirRatio \le 0.0,
$$

$$
CEp99\_delta \le 0.05,
$$

and:

$$
control\_gap \ge 0.
$$

This is not official functional success; it only permits Line B construction.

---

# 9. Line B：Basis-specific channel co-location functional repair

## 9.1 目标

从 response dictionary 中构造 functional event，不再直接扩 B-RAT/B-FOU token。

Functional repair must solve:

$$
\min_a
\quad
-\alpha \widehat{\Delta CouplingR^2}(a)
+\beta \widehat{\Delta NoiseSignalLeak}(a)
+\gamma \widehat{\Delta RealSignalReservoirRatio}(a)
+\eta \widehat{TailRisk}(a)
+\lambda \|a\|_2^2
$$

subject to:

$$
\widehat{TaskNonHarm}(a) \le \epsilon,
$$

$$
\widehat{Drift}(a) \le \tau.
$$

## 9.2 Allowed actuators

```text
Rational:
  tangent trust region, denominator-slope guard, group diversity transport, readout-rational decoupling.

Chebyshev:
  degree-energy damping, high-degree late-enable, degree tangent trust region.

Fourier:
  frequency-band damping, phase-stability guard, high-frequency noise leak veto.

RBF/FastKAN:
  center occupancy rebalance, width condition guard, OOG boundary repair.

Wavelet:
  scale energy balance, local support occupancy repair, support-overlap guard.
```

## 9.3 Controls

Every functional event must be compared against:

```text
NoOpMatchedOverhead
RandomMatchedNorm
AdamWParallelDirection, non-promotable control
SNR-only audit
Telemetry-shuffled actuator
Role-energy matched random actuator
```

## 9.4 P3 gate

A row passes P3 if:

$$
source\_vs\_best\_control \ge 0.005,
$$

$$
\Delta CouplingR^2 \ge 0.02,
$$

$$
\Delta NoiseSignalLeak \le -0.005,
$$

$$
\Delta RealSignalReservoirRatio \le -0.005,
$$

$$
CEp99\_delta \le 0.05,
$$

$$
NLL\_delta \le 0.02,
$$

$$
ECE\_delta \le 0.02.
$$

## 9.5 P4 short-run gate

P4 opens only if P3 has at least:

```text
3x3 all-row pass for a family, or
>= 7/9 pass plus bootstrap CI lower >= 0 for control gap.
```

P4 official requires:

$$
AUCtime_{source} \le AUCtime_{base},
$$

$$
source\_vs\_best\_control \ge 0.005,
$$

$$
LineC\_pass\_rate \ge 0.80,
$$

$$
CEp99\_delta \le 0.05.
$$

---

# 10. Line N：Non-RAT task-health substrate repair

## 10.1 Goal

Non-RAT families must not be judged only by workspace. They need substrate health.

## 10.2 Chebyshev

Hypothesis:

```text
Chebyshev fails because degree energy / recurrence live-set makes task and LineC unstable.
```

Try:

```text
degree-energy cap
late-enable K4 from K3
degree-normalized readout
readout grad recompute
optimizer-state delayed allocation
```

## 10.3 Fourier

Hypothesis:

```text
Low-frequency Fourier can be efficient but lacks interaction/expression; high frequency risks noise leak.
```

Try:

```text
lowfreq + identity residual
frequency band cap
phase-stability guard
frequency-band response dictionary
no high-frequency expansion unless NoiseSignalLeak audit remains non-harm
```

## 10.4 RBF / FastKAN

Hypothesis:

```text
Compact RBF has workspace path, but expression and task health collapse because center coverage is too thin.
```

Try:

```text
K_active 2 -> 4 controlled increase
center occupancy rebalance
width guard
out-of-grid guard
compact bump basis alternative
```

## 10.5 Wavelet

Hypothesis:

```text
Local wavelet support can be efficient but current scale/support placement is task-hostile.
```

Try:

```text
support occupancy balancing
scale diversity cap
hat/triangle only, no Morlet heavy path
support-overlap guard
```

---

# 11. Line M：MLP functional closure monitor

## 11.1 Goal

Do not spend major budget on MLP functional unless there is a new mechanism. Current M-J closure is no-go.

## 11.2 MLP closure status

If M-J family remains 0 pass, record:

```text
MLPFunctionalNoGo_CurrentLossAgnosticObservableFamily_v3
```

Then transfer functional budget to basis-specific repair.

## 11.3 Allowed continuation

Only run MLP again if there is a new mechanism not equivalent to M-J:

```text
new precommit observable family;
new control-resistant response dictionary;
new theorem-inspired SNR estimator not already covered.
```

---

# 12. Visualizations

v12.35 must generate:

```text
fig_progress_percent_by_line.svg
fig_family_substrate_heatmap.svg
fig_workspace_vs_task_health_pareto.svg
fig_nonrat_lifetime_vs_task_collapse.svg
fig_basis_telemetry_vs_linec_delta.svg
fig_response_dictionary_colocation_scatter.svg
fig_functional_control_gap_by_family.svg
fig_linec_noise_reservoir_delta_by_functional.svg
fig_ce_nll_ece_safety_by_candidate.svg
fig_mlp_functional_closure_no_go.svg
fig_route_sankey_v1235.svg
```

---

# 13. Required artifacts

```text
v1235_route_decision.json
v1235_required_artifact_manifest.csv
v1235_progress_by_line.csv
v1235_code_provenance_audit.csv
v1235_basis_substrate_health.csv
v1235_family_substrate_summary.csv
v1235_family_telemetry.csv
v1235_basis_response_dictionary.csv
v1235_response_colocation_summary.csv
v1235_basis_functional_p3.csv
v1235_basis_functional_p4.csv
v1235_nonrat_task_health_repair.csv
v1235_mlp_functional_closure.csv
v1235_failure_table.csv
v1235_next_hypothesis_queue.md
v1235_code_review_packet.zip
```

---

# 14. Stop / go routes

```text
S1-SubstrateHealthPass:
  At least one active basis passes substrate gate with minimum task-health.

S2-ResponseDictionaryPass:
  At least one active basis has usable response dictionary.

S3-BasisFunctionalP3Pass:
  At least one basis-specific functional repair passes P3.

S4-BasisFunctionalP4Pass:
  P4 short-run passes task/LineC/tail/control.

S5-OfficialFunctionalSuccess:
  label-free strict FC-PureKAN substrate + functional repair passes official controls.

R1-NoSubstrateHealth:
  workspace may pass, but no active basis has minimum task-health.

R2-SubstrateButNoFunctionalRepair:
  substrate exists, but response dictionary or P3/P4 fails.

R3-NonRATLifetimeTaskHealthSplit:
  Non-RAT lifetime opens but task/AUC/LineC collapse.

R4-RationalOnlySubstrate:
  only Rational substrate exists; Non-RAT blocked.

R5-MLPFunctionalNoGo:
  MLP functional closure remains 0 pass.

R6-LowValueTokenGridDetected:
  all new attempts are local token variants with no mechanism novelty.
```

---

# 15. Codex failure-handling rules

If a gate fails, Codex must not simply stop unless all required fallback for that route is executed.

```text
If substrate workspace passes but task health fails:
  Run task-health repair and response dictionary before final no-go.

If functional P3 fails with CouplingR2_no_gain:
  Build response dictionary; do not add new token grid first.

If Non-RAT lifetime opens but task collapses:
  Run family-specific task-health repair, not P3 official.

If MLP closure fails:
  Mark no-go and reduce MLP functional budget; do not keep M-J grid.

If artifacts missing:
  route = R0-ArtifactIncomplete.

If implementation readback incomplete:
  route = R0-CodeReviewSurfaceIncomplete.
```

---

# 16. Expected outcome of v12.35

Minimum success:

```text
1. A substrate-health map that separates workspace-only pass from usable substrate.
2. A response dictionary for at least one active basis.
3. Clear route deciding whether Rational-only substrate remains the path, or whether Non-RAT can be made task-healthy.
```

Strong success:

```text
At least one family reaches S2 response dictionary pass.
```

Official success:

```text
At least one basis-specific functional repair reaches P4 and beats controls.
```

Most likely route if current blocker persists:

```text
R2-SubstrateButNoFunctionalRepair
or
R4-RationalOnlySubstrate
```

But v12.35 must answer a deeper question than v12.34.2:

$$
\boxed{
\text{Is the failure due to missing healthy substrate, or due to missing basis-specific functional channel co-location?}
}
$$

