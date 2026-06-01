# DG-KAN v12.33：Rational Telemetry Kernel、Non-RAT Lifetime Repair、MLP Functional Closure 完整计划

> 版本：v12.33 execution plan  
> 基于：v12.32 `RationalTailKernel / NonRATFusedKernels / MLPFunctionalNoGo` 真实结果复盘  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；classic no-BSpline；不使用 label-informed initialization；no teacher / distillation / loss modification / sampler / class weight / dataset-name branch；functional direction 不使用 label、CE vector、permuted-label CE、validation/test、future outcome、query batch 或 LineC hard target；CE / NLL / ECE / CEp99 只作为审计和坏化约束，不能作为方向源或训练目标。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的最终目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是寻找一个只在某个局部 diagnostic 上好看的 KAN 变体。项目目标是：

$$
\boxed{
\text{构建 label-free strict FC-PureKAN base，}
\text{并通过 loss-agnostic functional update 得到比普通反向传播更好的训练几何和模型。}
}
$$

最终要证明的是：

$$
\boxed{
\text{PureKAN base + functional update}
>
\text{same PureKAN base + ordinary backprop / AdamW controls}
}
$$

同时必须满足：

```text
1. 表达力不打折，最好比同规模 MLP 更强；
2. forward / backward / step / memory 与 MLP 可比；
3. 收敛轨迹健康，AUC-step / AUC-time 不输；
4. 几何健康：CouplingR2、NoiseSignalLeak、RealSignalReservoirRatio、tail、calibration 不坏；
5. functional update 必须独立击败 NoOp / RandomMatchedNorm / AdamWParallel / SNR-only controls；
6. functional direction 必须 loss-agnostic，不允许针对 CE / NLL / ECE / CEp99 设计。
```

## 0.2 当前项目进展

当前项目经过 v12.26 之后已经明确禁止 `label-informed init`，因此历史 B320-current / FHQ 强结果只能作为工程参考，不能作为 official label-free claim。v12.30 起主线转为：

```text
1. Classic no-BSpline basis portfolio 作为主并行线；
2. MLP functional 作为 generic functional mechanism 对照线；
3. Label-free FHQ / A-DYN 只做低预算 monitor；
4. functional update 只有在 label-free base 或 family near-pass 出现后才能 official re-entry。
```

v12.31 的真实进展是 Rational workspace 明显打开：Rational workspace repair 达到 `45/45` pass、`26/45` strong pass，但 task / LineC / AUC / tail 没有同位；MLP functional M-G1..M-G5 没有 exploration / official pass。

v12.32 的真实进展进一步明确：

```text
1. Rational D-RAT16..D-RAT23 在 strict workspace gate 下 72/72 pass；
2. Rational strong workspace 0/72，incremental ratio 最低约 1.594，仍高于 strong gate 1.35；
3. Rational task / LineC 有正信号，但 AUC-time 与 CEp99 tail 不能闭合；
4. D-RAT20/D-RAT23 能降低 CEp99 delta，但 task / LineC / AUC 崩坏；
5. Chebyshev / Fourier exact no-materialize kernel 的 manual gradcheck 与 A4 smoke 通过，但 incremental memory gate 未开；
6. MLP M-H1..M-H3 在 windows=3/5/10 下没有 aggregate pass；
7. A-DYN monitor 没有 label-free near-anchor；
8. Provenance audit 通过，required artifacts 缺失为 0，fallback depth 6 已执行。
```

因此 v12.32 的合法状态是：

```text
route = R1-RationalTailStabilityBlocked
minimum_success = S1-RationalWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
basis_workspace_pass_count = 72
basis_workspace_strong_pass_count = 0
rational_auc_near_pass_count = 0
nonrat_s3_true_kernel_count = 0
mlp_functional_aggregate_pass_rows = 0
mlp_functional_no_go_current_family = 1
line_a_near_anchor_pass_count = 0
```

## 0.3 各线完成度：v12.31 到 v12.32

| 线 | v12.31 | v12.32 | 变化 | 解释 |
|---|---:|---:|---:|---|
| 代码 / provenance / finalizer 审计 | 92% | 94% | +2 | v12.32 required artifacts 与 provenance clean，新增 exact-kernel audit，但仍需人工审查 Rational 是否只是 proxy alias。 |
| Historical FHQ / B320-current 工程能力 | 85% frozen | 85% frozen | 0 | 历史强参考，但 label-informed init 已禁止，不能 official。 |
| Label-free FHQ / A-DYN monitor | 31% | 30% | -1 | A-DYN 无 near-anchor，继续低预算 monitor。 |
| Line C 几何审计 | 84% | 85% | +1 | 审计稳定，能暴露 task / LineC / tail 分裂；仍不是 direction source。 |
| Precommit / loss-agnostic value source | 24% | 22% | -2 | MLP M-H family no-go 进一步说明 generic value source 不成立。 |
| KAN functional official | 8% | 7% | -1 | 无 label-free base / family near-pass，functional 仍不能 official。 |
| MLP functional update | 20% | 22% | +2 | M-H1..M-H3 线完整跑完，但 0 pass；进展主要是 no-go 证据。 |
| Classic no-BSpline portfolio 总体 | 55% | 60% | +5 | Rational workspace 72/72 pass，Cheby/Fourier exact kernels有实证；但无 FamilyNearPass。 |
| Rational family | 70% | 76% | +6 | workspace 彻底打开到 S1，但 tail / AUC / task / LineC 同位失败。 |
| Chebyshev | 45% | 52% | +7 | exact no-materialize kernel gradcheck/A4 smoke 通过，但 incremental memory 仍 block。 |
| Fourier | 38% | 46% | +8 | exact no-materialize kernel gradcheck/A4 smoke 通过，但 incremental memory 仍 block。 |
| Wavelet | 42% | 40% | -2 | 本轮未推进，仍是 task / memory / LineC blocker。 |
| RBF / FastKAN | 38% | 36% | -2 | 本轮未推进，仍是 expression / compact-capacity blocker。 |
| 整体 next-gen MLP claim | 49%-53% | 50%-54% | +1 | Rational / exact-kernel 带来真实进展，但 official base 与 functional 仍未闭合。 |

注意：这些百分比不是实验 artifact 中的官方字段，而是根据 gate、blocker 清晰度、机制成熟度和距离 official success 的综合估计。

---

# 1. v12.32 独立分析

## 1.1 这次有进展吗？

有，但不是 S5 / official success。v12.32 最大进展是：

$$
\boxed{
\text{Rational 从 workspace blocker 进入 tail-stability / task-AUC-LineC 同位 blocker。}
}
$$

Rational 的 72/72 strict workspace pass 是实质进展。这说明 Rational 不再是“连系统成本都站不住”的状态。但 0/72 strong pass 也说明它还不是最终效率形态，特别是 incremental memory 仍未达 strong gate。

另一个进展是 Chebyshev / Fourier 的 exact no-materialize kernel 有了 manual gradcheck 与 A4 smoke 证据。这说明 Non-RAT exact kernel 已不只是 wrapper / alias。它们现在的 blocker 更精确地变成：

$$
\boxed{
\text{incremental memory accounting / workspace lifetime 没闭合。}
}
$$

MLP functional 线也有进展，但进展是 no-go 证据：M-H1..M-H3 没有 aggregate pass，说明当前 MLP loss-agnostic observable family 不能作为 generic functional update 机制。

## 1.2 为什么仍然感觉慢？

因为这轮没有任何 promotion：

```text
S5 official success = 0
Rational S2 = 0
Non-RAT S3 = 0
MLP functional S4 = 0
A-DYN near-anchor = 0
```

从最终目标看，这确实还没有能力突破。但从研究诊断看，v12.32 把问题从“也许继续 alias / objective 小修能救”推进成更清楚的结构性 blocker：

```text
Rational：需要真正 denominator / derivative telemetry，不是继续 output-geometry proxy token。
Chebyshev/Fourier：需要 workspace lifetime repair，不是证明 kernel 存在。
MLP functional：当前 M-H observable family no-go，不应继续扩同类 objective。
```

## 1.3 当前真正卡在哪里？

当前主 blocker 有三个。

### Blocker 1：Rational 的 proxy tail mechanism 不足

D-RAT16/17/18/19/22 可以给 positive mean task signal，LineC 常在 20/27 以上；但 AUC-time 明显大于 1.0，CEp99 tail 坏化严重，因此 `candidate_auc_near_pass_v1232 = 0`。D-RAT20/D-RAT23 可以降低 CEp99 delta，却导致 task / LineC / AUC 崩坏。

这说明 Rational 的问题不是“tailnorm 强度不对”，而是：

$$
\boxed{
\text{当前 output-level proxy tail repair 没有控制 rational function 本身的 denominator / derivative / tangent geometry。}
}
$$

也就是说，我们不能再继续在 `D-RAT proxy tail / output geometry` token family 上排列组合。下一步必须看到 rational 的内部函数几何：denominator、derivative、tangent metric、group diversity、function slope。

### Blocker 2：Non-RAT exact kernel 只是 forward/backward correctness，不是 lifetime success

Chebyshev / Fourier 的 exact no-materialize kernel已通过 manual gradcheck 与 A4 smoke，但 incremental memory gate 未开。这说明：

```text
kernel correctness != training system success
```

当前更可能的瓶颈是：

```text
readout grad materialization
basis derivative temporary
optimizer state overlap
LineC / A4 probe 生命周期
workspace buffer 未复用
profile 期间 peak lifetime 没释放
```

所以 Non-RAT 下一步应该做 lifetime/allocator 层审计，而不是继续加 K 或做 expression 小修。

### Blocker 3：MLP functional no-go 更可信

M-H1/M-H2/M-H3 都是合法的 loss-agnostic source：只读取 unlabeled inputs、activations、logits、random cotangent / microprobe response，不使用 label/CE 作为方向源。它们跑了 243 candidate rows、1458 controls、2430 LineC rows，但 aggregate pass 为 0。

这说明当前 functional update 很可能不是通用 MLP optimizer trick。若 functional 未来成立，它更可能需要 KAN / Rational / basis coordinate 的显式结构。

---

# 2. v12.33 核心假设

## H1：Rational 的当前 blocker 是函数内部几何不可见，而不是 workspace

v12.32 已经给出 Rational strict workspace 72/72 pass，因此 v12.33 不再继续扩大 D-RAT alias。新的假设是：

$$
\boxed{
\text{Rational tail / AUC / LineC 不同位，是因为当前机制没有观测并控制 denominator / derivative / tangent geometry。}
}
$$

需要验证：

```text
1. CEp99 bad rows 是否对应 den_p01 低、r_prime_p99 高、r_double_prime 高、group diversity collapse？
2. AUC_time fail 是否对应 denominator / derivative 轨迹不稳定？
3. LineC pass 下降是否对应 tangent rank / output derivative spectrum collapse？
4. 能否用 loss-agnostic rational telemetry 预测 tail / AUC / LineC badness？
```

## H2：Non-RAT blocker 是 workspace lifetime，不是 exact kernel correctness

Chebyshev/Fourier exact kernels存在，A4 smoke通过，但 incremental memory未过。新的假设是：

$$
\boxed{
\text{Non-RAT 需要 full-step lifetime repair：forward no-materialize 还不够，backward/readout/update workspace 也必须 no-materialize 或 recompute。}
}
$$

## H3：MLP functional 当前 observable family 应进入 no-go boundary

M-H1..M-H3 没有 aggregate pass，因此 v12.33 不再扩同类 MLP objective。只做一次 stronger-control confirm / no-go closure。

若继续失败，记录：

```text
MLPFunctionalNoGo_CurrentLossAgnosticObservableFamily
```

之后 functional 预算转向 basis-specific functional，而不是 MLP generic functional。

## H4：Functional update 只在 family near-pass 上 re-entry

没有 label-free near-base 或 family near-pass 时，functional 只能 shadow。当前最接近是 Rational，因此 functional re-entry 条件应改为：

```text
Rational S2 near-pass 或 Non-RAT S3 near-pass 出现后，才允许 basis-specific functional diagnostic。
```

---

# 3. Line R：代码、provenance、实现语义审计

## 3.1 目标

确保 v12.33 不把 proxy alias、wrapper、smoke、diagnostic 写成 kernel success，也不让 CE/NLL/ECE/CEp99 进入 direction source。

## 3.2 必须审查的代码面

Codex 必须在复盘中写清楚以下文件和 symbol：

```text
dgkan/diagnostics/basis_workspace.py
  V1233_BASIS_CANDIDATES
  workspace gate
  component peak / lifetime accounting
  exact kernel audit

dgkan/models/fc_purekan_primitives.py
  rational group implementation
  denominator computation
  numerator computation
  derivative / telemetry hook
  Chebyshev / Fourier no-materialize kernels

dgkan/functional/mlp_functional.py
  M-H no-go closure
  control implementations

experiments/run_v1233_*.py
  runner orchestration
  finalizer
  artifact manifest
```

## 3.3 必须落盘 artifact

```text
v1233_code_review_manifest.csv
v1233_core_symbol_map.json
v1233_provenance_audit.csv
v1233_forbidden_direction_audit.csv
v1233_required_artifact_manifest.csv
v1233_fallback_manifest.csv
```

## 3.4 Gate

任何 official / exploration candidate 如果出现：

```text
uses_y_for_stats = 1
uses_label_for_direction = 1
uses_ce_vector_for_direction = 1
uses_query_batch = 1
uses_validation_for_commit = 1
uses_dataset_name_branch = 1
kernel_is_wrapper_only = 1, but marked exact_kernel = 1
```

则 route 必须为：

```text
R0-ProvenanceOrImplementationViolation
```

---

# 4. Line D-RAT：Rational Telemetry Kernel

## 4.1 目标

把 Rational 从 output-level proxy tail repair，升级为 rational function internal geometry telemetry 与 controlled kernel。

核心目标：

$$
\boxed{
\text{Rational candidate 同时满足 workspace、task、AUC、LineC、tail audit。}
}
$$

## 4.2 候选设计

v12.33 不再继续 D-RAT16..D-RAT23 的局部 alias 扩展，而是引入新的 telemetry / derivative family。

```text
D-RAT24-DenDerivativeTelemetryKernel
  暴露 per-sample denominator p01 / p99、r_prime、r_double_prime、group diversity。

D-RAT25-DenDerivativeTelemetryRecomputeBackward
  forward 不保存 den/r' 大 tensor；backward recompute；测试 memory / correctness。

D-RAT26-TangentTrustRegionNoCE
  用 label-free rational tangent metric 限制 update / output geometry，不用 CE tail。

D-RAT27-DenSlopeGuardNoCE
  对 denominator / derivative 做 loss-agnostic guard，作为 architecture/kernel guard，不作为 CE-directed optimization。

D-RAT28-GroupDiversityPreservingRational
  防止 group rational function collapse，记录 group_function_diversity / group_dead_fraction。

D-RAT29-LineCStableTangentMix
  以 LineC 审计为 evaluation，不把 LineC hard target作为方向源；用 label-free tangent spectrum 做 mix。

D-RAT30-LowMemoryTelemetryStrong
  D-RAT24 的 low-memory strong gate版本，目标 incremental <= 1.35。

D-RAT31-TelemetryAblationControl
  只加 telemetry 不改变 forward，用于确认 telemetry 本身不改变 task。
```

## 4.3 必须记录字段：`v1233_rational_telemetry.csv`

```text
run_id
candidate_id
dataset
seed
epoch
step
workspace_raw_ratio
workspace_incremental_ratio
step_ratio
strong_workspace_pass
mean_delta_vs_MLP
worst_delta_vs_MLP
AUC_step_ratio
AUC_time_ratio
LineC_pass_count
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
CEp99_delta
NLL_delta
ECE_delta
margin_p10_delta
logit_entropy_p01
logit_entropy_p99
top1_top2_gap_p01
den_min
den_p01
den_p99
den_condition
r_prime_p95
r_prime_p99
r_double_prime_p95
tangent_rank
tangent_top_eigen_share
group_function_diversity
group_dead_fraction
telemetry_available
telemetry_materialized_bytes
exact_fused_rational_kernel
uses_ce_tail_direction
failure_reason
```

## 4.4 Rational S2 gate

Exploration S2-near：

$$
workspace\_raw\_ratio \le 1.05
$$

$$
workspace\_incremental\_ratio \le 1.75
$$

$$
step\_ratio \le 1.75
$$

$$
mean\_delta\_vs\_MLP \ge -0.005
$$

$$
worst\_delta\_vs\_MLP \ge -0.025
$$

$$
AUC\_time\_ratio \le 1.05
$$

$$
LineC\_pass\_count \ge 20/27
$$

$$
CEp99\_delta \le 0.5
$$

Official S2：

$$
workspace\_incremental\_ratio \le 1.35
$$

$$
step\_ratio \le 1.40
$$

$$
mean\_delta\_vs\_MLP \ge 0
$$

$$
worst\_delta\_vs\_MLP \ge -0.010
$$

$$
AUC\_time\_ratio \le 1.00
$$

$$
LineC\_pass\_count = 27/27
$$

$$
CEp99\_delta \le 0.05
$$

注意：`CEp99_delta` 是审计坏化约束，不是优化目标。

## 4.5 失败后 Codex 必须继续尝试

如果 workspace pass 但 tail fail：

```text
1. 检查 den_p01 / r_prime_p99 是否解释 CEp99；
2. 若解释，进入 D-RAT25/D-RAT27；
3. 若不解释，进入 output logit entropy / top1-top2 gap proxy；
4. 不允许直接用 CEp99 梯度或 CE-tail loss。
```

如果 task pass 但 LineC fail：

```text
1. 检查 tangent_rank / top_eigen_share；
2. 运行 D-RAT29 LineCStableTangentMix；
3. 增加 group diversity guard；
4. 不允许用 LineC hard target 生成方向。
```

如果 strong workspace fail：

```text
1. telemetry recompute backward；
2. den/r' 不落地，只采样 small telemetry batch；
3. group workspace reuse；
4. readout grad recompute；
5. optimizer state overlap audit。
```

---

# 5. Line D-KER：Non-RAT Incremental Memory Lifetime Repair

## 5.1 目标

Chebyshev / Fourier exact no-materialize kernel 已经存在，但 S3 未开。v12.33 的目标不是证明 kernel correctness，而是证明 full-step incremental memory lifetime 能闭合。

## 5.2 候选

```text
D-CHE12-LifetimeRecomputeBackward-K3
D-CHE13-FusedReadoutGradNoMaterialize-K3
D-CHE14-OptimizerStateLifetimeReuse-K3
D-CHE15-FullStepNoMaterialize-K3

D-FOU12-LifetimeRecomputeBackward-K2
D-FOU13-FusedReadoutGradNoMaterialize-K2
D-FOU14-SincosSharedWorkspace-K2
D-FOU15-FullStepNoMaterialize-K2
```

低预算 monitor：

```text
D-RBF11-CompactExpressionRepair-Monitor
D-WAV10-HatWaveletLifetimeRepair-Monitor
```

## 5.3 必须记录字段：`v1233_nonrat_lifetime.csv`

```text
candidate_id
family
dataset
seed
exact_kernel_implemented
gradcheck_pass
A4_smoke_pass
basis_activation_bytes
basis_derivative_bytes
readout_grad_bytes
coeff_grad_bytes
optimizer_state_bytes
temp_workspace_bytes
raw_peak_ratio
incremental_peak_ratio
step_ratio
workspace_lifetime_overlap_count
largest_live_tensor_name
largest_live_tensor_bytes
materializes_basis
materializes_derivative
recomputes_basis_backward
fuses_readout_grad
S3_exploration_pass
S3_official_pass
failure_reason
```

## 5.4 S3 gate

Exploration S3：

$$
exact\_kernel\_implemented = 1
$$

$$
gradcheck\_pass = 1
$$

$$
A4\_smoke\_pass = 1
$$

$$
incremental\_peak\_ratio \le 1.75
$$

$$
step\_ratio \le 1.75
$$

Official S3：

$$
incremental\_peak\_ratio \le 1.35
$$

$$
step\_ratio \le 1.40
$$

并且不能有 full basis / derivative materialization：

$$
materializes\_basis = 0
$$

$$
materializes\_derivative = 0
$$

## 5.5 失败后 Codex 必须继续尝试

如果 raw pass 但 incremental fail：

```text
1. 输出 lifetime waterfall；
2. 找 largest_live_tensor；
3. 尝试 recompute backward；
4. 尝试 fused readout grad；
5. 尝试 optimizer state delayed allocation；
6. 再测 profile，不能只写 WorkspaceBlocked。
```

如果 exact kernel pass 但 A4 fail：

```text
1. Chebyshev modest K increase；
2. Fourier lowfreq + identity residual；
3. 只允许在 S3 exploration仍通过时增加 capacity。
```

---

# 6. Line M：MLP Functional Mechanism Closure

## 6.1 目标

v12.32 的 M-H1..M-H3 已经没有 aggregate pass。v12.33 不再扩同类 MLP objective，而是做一次 closure test，回答：

$$
\boxed{\text{当前 loss-agnostic MLP observable family 是否应正式 no-go？}}
$$

## 6.2 候选

```text
M-I1-CloneProbeCovarianceUpperBound
  non-promotable upper-bound，判断 response information 是否理论上有信号。

M-I2-PrecommitUnlabeledResponseStability
  只用 commit 前无标签 perturbation response。

M-I3-ControlResidualizedMicroProbe
  对 M-H family 做 matched-control residualization。

M-I4-ArchitectureNeutralSNRTransport
  只使用 optimizer-observable no-label stats，复核 SNR-like generic mechanism。
```

## 6.3 必须记录字段：`v1233_mlp_functional_closure.csv`

```text
candidate_id
window
dataset
seed
train_seed_base
source_vs_noop
source_vs_best_control
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
NLL_delta
ECE_delta
matched_control_gap
linec_pass_count
aggregate_pass
promotion_allowed
uses_label_for_direction
uses_ce_for_direction
failure_reason
```

## 6.4 Gate

Exploration：

$$
source\_vs\_best\_control \ge 0.005
$$

$$
LineC\_pass\_count \ge 4/5
$$

$$
CEp99\_delta \le 0.05
$$

Official：

$$
source\_vs\_best\_control \ge 0.01
$$

$$
LineC\_pass\_count = 5/5
$$

$$
NLL\_delta \le 0
$$

$$
ECE\_delta \le 0.01
$$

如果 M-I1 upper-bound 有信号但 M-I2/M-I3 没信号，结论是：

```text
response-level information exists but not deployable precommit.
```

如果全 fail，则记录：

```text
MLPFunctionalNoGo_CurrentLossAgnosticObservableFamily
```

---

# 7. Line F：Basis-specific functional shadow

## 7.1 目标

Functional 不抢主预算。只有 Rational S2-near 或 Non-RAT S3-near 出现后，才运行 basis-specific functional shadow。

## 7.2 候选

Rational：

```text
F-RAT1-TangentMetricGeometryMaintenance
F-RAT2-DenDerivativeSafeMaintenance
F-RAT3-GroupDiversityRefresh
```

Chebyshev/Fourier：

```text
F-CHE1-DegreeEnergyDampingNoCE
F-FOU1-FrequencyEnergyDampingNoCE
```

## 7.3 Gate

Functional shadow 必须先满足：

```text
base_family_near_pass = 1
```

否则只记录 skipped reason，不运行 P4。

---

# 8. Line A：Label-free FHQ / A-DYN Monitor

FHQ / A-DYN 不再是主预算。每轮只做 no-regression monitor：

```text
A-DYN1..A-DYN5
train_size = 512
val/test = 256
epochs = 3
seeds = 0,1,2
```

若出现 near-anchor，再开独立计划；否则不抢 Rational / Non-RAT kernel 预算。

---

# 9. 可视化要求

必须生成：

```text
fig_v1233_progress_over_versions.svg
fig_v1233_rational_workspace_tail_pareto.svg
fig_v1233_rational_den_derivative_vs_CEp99.svg
fig_v1233_rational_auc_linec_tail_heatmap.svg
fig_v1233_nonrat_lifetime_waterfall.svg
fig_v1233_nonrat_incremental_memory_by_component.svg
fig_v1233_mlp_functional_control_gap.svg
fig_v1233_family_status_matrix.svg
fig_v1233_route_dashboard.svg
```

---

# 10. Stop / continue 规则

v12.33 必须继续执行 fallback，但不能继续低价值 alias 网格。

## 10.1 不允许 final stop 的情况

```text
1. Rational workspace pass 但没有 telemetry artifact；
2. Rational tail fail 但没有 den/derivative/tangent autopsy；
3. Non-RAT incremental memory fail 但没有 lifetime waterfall；
4. MLP functional fail 但没有 matched-control closure；
5. required artifact missing；
6. implementation readback missing。
```

route：

```text
R0-ExploreDepthIncomplete
```

## 10.2 允许 final stop 的情况

```text
1. official success reached；
2. Rational telemetry no-go boundary closed；
3. Non-RAT lifetime no-go boundary closed；
4. MLP functional no-go closure complete；
5. hard compute budget exhausted and all planned fallback levels executed；
6. no-go boundary includes next mechanism-level hypothesis, not local alias grid。
```

---

# 11. 最终 route 定义

```text
R0-ProvenanceOrImplementationViolation
R0-ExploreDepthIncomplete
R1-RationalTelemetryKernelBlocked
R2-RationalTailTaskLineCNotColocated
R3-RationalS2NearPass
R4-RationalS2OfficialPass
R5-NonRATLifetimeBlocked
R6-NonRATS3NearPass
R7-NonRATS3OfficialPass
R8-MLPFunctionalNoGoClosed
R9-MLPFunctionalGenericPositive
R10-BasisSpecificFunctionalShadowPositive
R11-S5OfficialFunctionalSuccess
```

---

# 12. 本轮预期判断

我对 v12.33 的预期不是立刻拿到 S5，而是回答三个更本质的问题：

```text
1. Rational 的 task/AUC/tail blocker 是否能被 denominator/derivative/tangent telemetry 解释并修复？
2. Chebyshev/Fourier 的 exact kernels 能否从 correctness 进入 full-step lifetime success？
3. MLP functional 是否应在当前 loss-agnostic observable family 下正式 no-go？
```

如果 Rational telemetry 能解释 CEp99 / AUC / LineC 的失败，并且某个 D-RAT candidate 达到 S2-near，那么下一轮才进入 Rational-focused longer run。

如果 Non-RAT lifetime repair 打开 S3，那么下一轮才进入 Chebyshev / Fourier task hardening。

如果 MLP functional closure 继续 0 pass，则 functional update 主预算应只保留 basis-specific path，不再做 generic MLP objective 扩展。

