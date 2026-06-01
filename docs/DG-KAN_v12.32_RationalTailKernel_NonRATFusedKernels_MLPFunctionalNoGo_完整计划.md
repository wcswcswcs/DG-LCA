# DG-KAN v12.32：Rational Tail-Stable Kernel、Non-Rational 真 Fused Kernel、MLP Functional No-Go 复核完整计划

> 版本：v12.32 execution plan  
> 基于：v12.31 `BasisKernelFirst MLPFunctionalNoGoTest` 真实结果复盘  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 总原则：不针对单个数据集调参；不降低 gate；不使用 teacher / distillation / loss modification / sampler / class weight；不使用 label-informed initialization；functional direction 不使用 label、CE vector、permuted-label CE、validation/test、future outcome、query batch 或 LineC hard target；CE / NLL / ECE / CEp99 只能作为审计和坏化约束。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的最终目标不是在 MNIST / Fashion-MNIST / KMNIST 上刷分，也不是找一个只在某个 toy geometry 指标上好看的更新。项目目标是：

$$
\boxed{
\text{构建 label-free strict FC-PureKAN base，并通过 loss-agnostic functional update 获得比普通 backprop/AdamW 更好的训练几何和模型。}
}
$$

这里的“更好”必须同时满足：

```text
1. 表达力不打折，最好优于同规模 MLP；
2. forward / backward / step / memory 与 MLP 可比；
3. 收敛轨迹健康，AUC-step / AUC-time 不输；
4. 几何更健康：train-probe coupling、signal/reservoir/noise、tail、calibration 不坏；
5. functional update 的收益必须独立于 NoOp / RandomMatchedNorm / AdamWParallel / SNR-only / MLP analog controls；
6. functional direction 必须 loss-agnostic，不能为 CE / CEp99 / ECE / NLL 设计方向。
```

最终要证明的是：

$$
\boxed{
\text{label-free PureKAN base + loss-agnostic functional update}
>
\text{同一 base + ordinary AdamW / backprop controls}
}
$$

而不是只证明：

$$
\text{某个 KAN base 比 MLP accuracy 高一点。}
$$

也不是只证明：

$$
\text{某个 functional event 在 CouplingR2 上升。}
$$

## 0.2 当前进展概览

v12.31 的真实结果可以概括为：

```text
route = R2-BasisTaskLineCNotColocated
minimum_success = S1-BasisWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
basis_family_near_pass_count = 0
mlp_functional_exploration_pass_rows = 0
line_a_near_anchor_pass_count = 0
```

也就是说，本轮没有 S5、没有 classic FamilyNearPass、没有 MLP functional generic positive、没有 label-free FHQ near-anchor。

但是 v12.31 不是无进展。它完成了三件重要的事情：

```text
1. Basis workspace truth audit 不再停留在 alias scout，而是进入 P0 workspace / peak / step 口径；
2. Rational workspace 明显打开：initial D-RAT 36/54 workspace pass，22/54 strong pass；后续 Rational multi-step repair 达到 45/45 workspace pass，26/45 strong pass；
3. MLP functional M-G1..M-G5 完整跑 matched controls，确认当前 MLP loss-agnostic functional source 没有 generic positive。
```

当前项目的核心状态是：

```text
历史 FHQ / B320-current：证明过系统效率与 task 能力，但 label-informed init 已禁止，不能 official。
label-free FHQ：没有 near-anchor，只能 monitor。
classic no-BSpline：Rational 有明确正进展；Chebyshev / Fourier / RBF / Wavelet 仍主要 workspace blocked。
functional update：MLP 上没有 generic positive；KAN 上没有 label-free near-anchor 支撑 official re-entry。
```

## 0.3 当前各线进度百分比：v12.30 -> v12.31

这些百分比不是官方 artifact 字段，而是基于 gate、artifact、机制清晰度和离 official success 的距离做的阶段判断。

| 线 | v12.30 | v12.31 | 变化 | 判断 |
|---|---:|---:|---:|---|
| Line R：代码 / provenance / finalizer 审计 | 89% | 92% | +3 | 继续进步。workspace core 下沉到 `dgkan/diagnostics/basis_workspace.py`，MLP functional core 下沉到 `dgkan/functional/mlp_functional.py`，但仍要审查是否只是 wrapper。 |
| Historical FHQ / B320-current 工程能力 | 85% frozen | 85% frozen | 0 | 只作历史 reference。label-informed init 已禁止，不能 official。 |
| Label-free FHQ / A-DYN monitor | 34% | 31% | -3 | A-DYN1..5 全部差于 MLP，未恢复 near-anchor。 |
| Line C 几何审计 | 83% | 84% | +1 | 审计稳定，能暴露 task-LineC 分裂；仍不能作为 direction source。 |
| Precommit / loss-agnostic value source | 28% | 24% | -4 | MLP M-G1..5 均未 pass；generic functional value source 更不乐观。 |
| KAN functional update official | 10% | 8% | -2 | 没有 label-free base / family near-pass，functional 仍只能 shadow。 |
| MLP functional update | 18% | 20% | +2 | 线已建立并测试更强 M-G，但 0 pass；进展主要是 no-go 证据。 |
| Classic no-BSpline portfolio 总体 | 48% | 55% | +7 | Rational workspace 打开是实质进展；其余 family 仍 workspace blocked。 |
| Rational family | 60% | 70% | +10 | workspace 45/45 pass；task/LineC/AUC/tail 未同位。最接近 family success。 |
| Chebyshev family | 50% | 45% | -5 | raw/step 可低，但 incremental memory blocker 未解决，不能 task hardening。 |
| Wavelet family | 45%-50% | 42% | -5 | raw/step 部分可低，但 incremental memory blocker；任务未进入 official。 |
| RBF / FastKAN family | 35%-40% | 38% | 0到+2 | raw/step 还行，incremental memory 高，A4/task 未恢复。 |
| Fourier family | 35%-40% | 38% | 0到+2 | raw/step 还行，incremental memory 高，expression/task 未恢复。 |
| 整体 next-gen MLP claim | 47%-51% | 49%-53% | +2 | Rational 打开 workspace 带来正进展，但 official base / functional 仍未闭合。 |

---

# 1. 对 v12.31 结果的独立分析

## 1.1 这次有进展吗？

有，但不是 S5 成功，也不是 functional update 成功。它的真实进展是 **basis workspace 问题被拆开，并且 Rational 这条经典基函数线第一次从 memory/workspace blocker 中明显松动**。

v12.30 的结论是 classic basis memory blocker 很硬：raw memory ratio 约 $2.10$，incremental peak ratio 约 $9.97$。v12.31 做了更细的 workspace truth audit，并在 Rational 上继续多步 profile repair。结果是：

```text
Rational initial workspace：36/54 pass，22/54 strong pass；
Rational multi-step repair：45/45 pass，26/45 strong pass。
```

这意味着 Rational 的系统 blocker 已经从：

```text
memory/workspace 不能过
```

推进到：

```text
workspace 能过，但 task / LineC / AUC / tail 无法同位。
```

这是有价值的阶段性进展。

## 1.2 为什么还不能 promotion？

因为 Rational 仍没有形成 FamilyNearPass。最接近的结果来自 `D-RAT10-FusedDenNumReadoutGrad` 的 3 epoch + lr=0.0015：

```text
mean_delta_vs_MLP = +0.016059
worst_delta_vs_MLP = -0.019531
max_AUC_time_ratio_vs_MLP = 1.089952
LineC_pass_count = 20/27
max_CEp99_delta_vs_MLP = +2.608434
near pass = 0
```

这个结果有正信号：mean 和 worst 都比之前好，AUC 接近，LineC 也不是崩坏。但 CEp99 tail 严重坏化，且 AUC_time 仍大于 1。由于 CEp99 / NLL / ECE 只能作为审计与坏化约束，不能作为 direction source 或 loss target，所以不能用 CE-tail trick 修它。

后续 tailnorm output-geometry repair 也没有闭合：

```text
D-RAT13：CEp99 明显改善，但 task / AUC / LineC 全面失败；
D-RAT14：LineC 和 mean 较好，但 worst / AUC_time / CEp99 仍失败；
D-RAT15：介于两者之间，也未过 near-pass。
```

所以 v12.31 的真正 blocker 是：

$$
\boxed{
\text{Rational workspace 已经可行，但 task / LineC / AUC / tail stability 不能同位。}
}
$$

## 1.3 非 Rational family 的含义

Chebyshev、Fourier、RBF、Wavelet 不是数学上失败，而是这轮仍没有通过 workspace gate。更精确地说：

```text
raw ratio 往往可以低于 1.5；
incremental ratio 仍高于 3.0；
因此不能进入 task hardening。
```

这说明问题不在算子主计算本身，而在：

```text
basis activation materialization；
basis derivative materialization；
readout grad / coeff grad peak；
optimizer state overlap；
temporary workspace lifetime；
runner 中现有 alias 没有真正做到 no-materialize fused kernel。
```

更关键的是，v12.31 明确记录 `exact_kernel_implemented=0`。也就是说，这轮做的是 workspace truth audit 和 existing low-memory path 映射，不是已经实现了计划字面意义上的真实 fused CUDA/Triton kernel。下一步如果还继续 alias，就会停留在假进展。

## 1.4 MLP functional update 的意义

v12.31 在 MLP 上跑了 M-G1..M-G5：

```text
candidate_rows = 405
control_rows = 2430
linec_rows = 4050
mlp_functional_exploration_pass_rows = 0
mlp_functional_official_pass_rows = 0
```

有些局部 row 能正过 NoOp / control 且 LineC 5/5，但 CouplingR2_delta 为负，整体 mean_vs_control 为负。因此不能说 MLP 上存在 generic positive。

这对项目很重要。它说明：

```text
当前 loss-agnostic functional source 不是通用 optimizer 机制；
至少 M-G1..M-G5 这类 response/covariance/value source 不足以打过 matched controls；
functional update 如果要成功，很可能需要 KAN / Rational / basis coordinate 的显式结构，而不是 MLP 上的通用 logit/activation geometry trick。
```

## 1.5 当前是否在正确道路上？

是，但需要再次升级层级。

对的地方：

```text
1. 不再主投 label-free FHQ token 小修；
2. classic basis 成为主并行线是对的；
3. MLP functional 同步验证是对的；
4. 不把 workspace smoke / skipped rows / diagnostic 写成 promotion 是对的；
5. 不用 CEp99 设计方向是对的。
```

需要改变的地方：

```text
1. Rational 不能继续 D-RAT + epoch/lr + tailnorm 小网格；
2. 非 Rational 不能继续 alias 映射，必须做真实 fused/no-materialize kernel；
3. MLP functional 不能继续 M-G single-objective 变体，应该进入 mechanism-level no-go 或全新机制；
4. functional update 不应抢 basis kernel / Rational tail-stability 的主预算。
```

---

# 2. v12.32 总体策略

v12.32 的核心是：

$$
\boxed{
\text{Basis-first 继续，但从 alias search 升级为 kernel / tail-stability / task-LineC co-location 机制实验。}
}
$$

具体分三条主线：

```text
Line D-RAT：Rational Tail-Stable Kernel 主线。
Line D-KER：非 Rational 真 fused / no-materialize kernel 主线。
Line M：MLP functional 机制 no-go 复核，不再大规模扩 M-G。
```

低预算监控：

```text
Line A：label-free FHQ / A-DYN monitor，仅防止错过意外恢复。
Line C：Manifold-Channel audit，继续作为所有候选的 geometry safety。
Line R：代码 / provenance / exact kernel / implementation readback 审计。
```

计算预算建议：

```text
Line D-RAT Rational: 40%
Line D-KER true kernels: 30%
Line M MLP functional: 10%
Line A/F FHQ functional shadow: 5%-10%
Line C/R/Z audit/finalizer: 10%-15%
```

---

# 3. 核心假设

## H1：Rational 当前不是 workspace blocked，而是 tail-stability / trajectory blocked

v12.31 已经证明 Rational workspace 可以打开。新的 hypothesis 是：

$$
\boxed{
\text{Rational 的 task/LineC 近似成立，但 denominator / derivative / logit-tail 几何不稳定，导致 CEp99、AUC-time 与 worst row 失败。}
}
$$

因此下一步不应继续扫 lr / epoch / tailnorm，而要直接审计并控制：

```text
denominator p01 / min；
rational derivative p95 / p99；
rational second derivative；
logit norm p99；
unlabeled entropy floor；
top1-top2 unlabeled margin p01；
LineC signal/reservoir/noise；
CEp99 / NLL / ECE as audit-only badness constraints。
```

## H2：非 Rational family 的主 blocker 是 basis/derivative/readout grad materialization

Chebyshev / Fourier / RBF / Wavelet 的 raw step 可以低，但 incremental memory 过高。新的 hypothesis 是：

$$
\boxed{
\text{现有 alias 仍然 materialize basis / derivative / readout grad workspace，必须实现真正 no-materialize fused kernel 才能继续判断 family 价值。}
}
$$

## H3：MLP functional 当前 family 已形成 no-go，继续轻量变体价值低

MLP M-G1..M-G5 0 pass。新的 hypothesis 是：

$$
\boxed{
\text{当前 MLP loss-agnostic functional value source family 不成立；若 functional update 有价值，可能需要显式 basis coordinate 或新的 population-risk observable，而不是 M-G family 小修。}
}
$$

---

# 4. Line R：代码、kernel、provenance 审计

## 4.1 目标

防止 v12.32 继续出现“名字像 fused kernel，但实际只是 alias / wrapper”的假进展。

## 4.2 必须记录的 artifact

```text
v1232_core_symbol_map.json
v1232_kernel_implementation_manifest.csv
v1232_exact_kernel_audit.csv
v1232_provenance_audit.csv
v1232_forbidden_token_audit.csv
v1232_code_review_packet.zip
```

## 4.3 每个 kernel candidate 必须记录

```text
candidate_id
family
actual_file_path
actual_symbol
exact_kernel_implemented
uses_triton
uses_cuda_extension
uses_torch_autograd_graph
uses_loss_backward
materializes_basis_tensor
materializes_derivative_tensor
materializes_readout_grad_tensor
basis_tensor_shape_if_any
derivative_tensor_shape_if_any
workspace_temp_bytes_static
uses_label_or_ce_for_direction
uses_y_for_stats
forbidden_token_present
manual_gradcheck_pass
max_forward_abs_error
max_grad_rel_error
```

## 4.4 硬门

如果一个 candidate 声称是 fused/no-materialize kernel，则必须满足：

$$
exact\_kernel\_implemented = 1,
$$

$$
materializes\_basis\_tensor = 0,
$$

$$
materializes\_derivative\_tensor = 0.
$$

否则它只能是 `workspace_audit_alias`，不能作为 kernel success。

---

# 5. Line D-RAT：Rational Tail-Stable Kernel 主线

## 5.1 总目标

把 v12.31 的 Rational workspace progress 推向真正 FamilyNearPass：

$$
\boxed{
\text{Rational must close workspace + task + AUC + LineC + tail together.}
}
$$

## 5.2 候选族

### D-RAT-K1：True fused rational denominator/readout kernel

目标：把 D-RAT10 的 workspace positive 从 existing low-memory path 升级为真实 kernel。

必须实现：

```text
1. fused numerator / denominator eval；
2. fused derivative recompute in backward；
3. fused readout grad accumulation；
4. no persistent dense basis / derivative materialization；
5. manual correctness vs reference。
```

记录：

```text
num_eval_ms
den_eval_ms
division_ms
readout_grad_ms
denominator_grad_ms
workspace_temp_bytes
raw_peak_ratio
incremental_peak_ratio
step_ratio
```

### D-RAT-K2：Denominator / derivative stability mechanism

不使用 CE / label，只控制 rational 函数本身的无标签几何。

记录：

```text
den_min_batch
den_p01_batch
den_condition_batch
r_prime_p95
r_prime_p99
r_double_prime_p95
r_double_prime_p99
rational_output_p99
logit_norm_p99
unlabeled_entropy_mean
unlabeled_entropy_p05
unlabeled_top1_top2_gap_p01
```

候选：

```text
D-RAT16-DenP01FloorNoCE
D-RAT17-RPrimeCapNoCE
D-RAT18-TangentNormTrustNoCE
D-RAT19-DenSlopeJointStabilityNoCE
```

这些机制不能读标签，不能以 CEp99 为目标，只能用 CEp99/NLL/ECE 做 audit。

### D-RAT-K3：Calibration-preserving output geometry

v12.31 tailnorm 显示：强 RMS norm 能修 CEp99 但伤 task/LineC，RMS mix 能保 LineC 但 tail/AUC 不闭合。因此下一步要换成更平滑的 output geometry mechanism。

候选：

```text
D-RAT20-LogitSpectrumClipStopGrad
D-RAT21-EntropyFloorUnlabeled
D-RAT22-TopGapFloorNoLabel
D-RAT23-LogitNormEMAAnchorNoCE
```

硬约束：

```text
不使用 CEp99 / NLL / ECE 生成方向；
不使用 label；
不做 label smoothing；
不改 loss；
不按 dataset 分支。
```

## 5.3 实验流程

### P0：kernel/correctness/workspace

每个 D-RAT16..23 先跑：

```text
MNIST,Fashion-MNIST,KMNIST
seeds=0,1,2
train_size=512
workspace_warmup_steps=5
workspace_profile_steps=20
```

通过 gate：

$$
raw\_peak\_ratio \le 1.05,
$$

$$
incremental\_peak\_ratio \le 1.75,
$$

$$
step\_ratio \le 1.75.
$$

Strong gate：

$$
incremental\_peak\_ratio \le 1.35,
$$

$$
step\_ratio \le 1.40.
$$

### P1：3 epoch trajectory triage

只对 P0 pass candidates 跑：

```text
train_size=512
val/test=256
epochs=3
lr in {0.0015, 0.002}
```

必须记录：

```text
mean_delta_vs_MLP
worst_delta_vs_MLP
max_AUC_time_ratio_vs_MLP
max_AUC_step_ratio_vs_MLP
max_CEp99_delta_vs_MLP
max_NLL_delta_vs_MLP
max_ECE_delta_vs_MLP
LineC_pass_count
LineC_CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
```

### P2：8 epoch hardening only for near triage

进入 P2 的条件：

```text
mean_delta_vs_MLP >= -0.005
worst_delta_vs_MLP >= -0.030
max_AUC_time_ratio_vs_MLP <= 1.20
LineC_pass_count >= 20/27
max_CEp99_delta_vs_MLP <= 1.0
```

P2 gate：

$$
mean\_delta \ge 0,
$$

$$
worst\_delta \ge -0.01,
$$

$$
AUCtime \le 1.00,
$$

$$
CEp99\_delta \le 0.05,
$$

$$
LineC\_pass\_count \ge 25/27.
$$

## 5.4 Rational failure fallback

如果 workspace fail：

```text
1. chunked readout grad；
2. denominator recompute backward；
3. grouped workspace reuse；
4. optimizer-state delayed allocation；
5. fp16 denominator state audit, only if correctness pass。
```

如果 task/worst fail but LineC ok：

```text
1. identity residual gain schedule；
2. late-enable rational nonlinear residual；
3. role-wise residual scale cap；
4. no CE-specific scheduler。
```

如果 CEp99/NLL/ECE fail：

```text
1. denominator / derivative stability audit；
2. unlabeled logit spectrum stability；
3. entropy floor / logit norm p99 cap；
4. tangent trust region；
5. no CE-tail gradient, no label smoothing, no loss change。
```

If LineC fail：

```text
1. role-wise tangent split；
2. coupling/reservoir audit by role；
3. remove candidate if improvement is task-only but geometry-bad；
4. do not add dataset-specific controller。
```

---

# 6. Line D-KER：Non-Rational 真 fused / no-materialize kernel 主线

## 6.1 总目标

不再让 Chebyshev / Fourier / RBF / Wavelet 停在 “raw ratio good, incremental memory bad”。v12.32 必须实现至少两个真实 no-materialize fused microkernel vertical slices。

优先顺序：

```text
1. Chebyshev fused recurrence kernel：最容易验证无 materialized basis。
2. Fourier fused low-frequency sincos/readout kernel：计算简单，expression 是后续问题。
3. RBF local active-center recompute kernel。
4. Hat/Triangle Wavelet local support recompute kernel。
```

## 6.2 Chebyshev fused recurrence

实现：

```text
D-CHE10-FusedRecurrenceNoMaterialize-K3
D-CHE11-FusedRecurrenceNoMaterialize-K4
```

公式：

$$
T_0(x)=1,
$$

$$
T_1(x)=x,
$$

$$
T_{k+1}(x)=2xT_k(x)-T_{k-1}(x).
$$

要求：在 kernel 内 recurrence + readout accumulation，不写出 `[B,D,K]`。

记录：

```text
cheby_degree_energy
high_degree_energy_ratio
recurrence_max_abs
basis_materialized = 0
incremental_peak_ratio
step_ratio
A4_expression_score
LineC
```

Fallback：

```text
if workspace fail: fuse readout grad and coeff grad;
if expression fail: K4 + identity residual, but no high-degree blind increase;
if task fail: degree-energy damping / late-enable high-degree。
```

## 6.3 Fourier fused low-frequency kernel

实现：

```text
D-FOU10-FusedSincosLowFreqK2
D-FOU11-FusedSincosLowFreqK4SharedAmp
```

要求：kernel 内计算 sin/cos 并直接累加，不写出 `[B,D,2K]`。

记录：

```text
frequency_count
sincos_eval_ms
basis_materialized
spectral_entropy
high_freq_energy_ratio
NoiseSignalLeak
CEp99
```

Fallback：

```text
if expression fail: lowfreq + identity residual + small learned amplitude;
if noise/tail fail: high-frequency energy cap;
if workspace fail: precompute frequency constants / fused readout grad。
```

## 6.4 RBF / FastKAN local active-center kernel

实现：

```text
D-RBF11-LocalRBFK4NoMaterialize
D-RBF12-LocalRBFK8RecomputeBackward
```

要求：只计算 active centers，不写 dense basis。

记录：

```text
K_total
K_active
exp_count_per_sample
active_center_entropy
dead_center_fraction
out_of_grid_fraction
basis_materialized
```

Fallback：

```text
if expression fail: K_active 4->8, train-stream quantile centers;
if workspace fail: recompute exp in backward;
if task fail: center occupancy rebalance, not CE-specific。
```

## 6.5 Wavelet local support kernel

实现：

```text
D-WAV10-FusedHatTriangleK4NoMaterialize
D-WAV11-FusedHatTriangleK8ScaleDiversity
```

公式：

$$
\psi(x)=\max(1-|x|,0).
$$

记录：

```text
active_support_count
scale_energy
scale_dead_fraction
local_tail_coverage
basis_materialized
CEp99 audit
LineC
```

Fallback：

```text
if task collapse: reduce scale diversity, late-enable wavelet residual;
if expression fail: K8 local support, not Morlet heavy path;
if workspace fail: reuse RBF local kernel skeleton。
```

---

# 7. Line M：MLP functional mechanism no-go / transfer test

## 7.1 目标

v12.31 已经显示 M-G1..M-G5 0 pass。v12.32 不再扩 M-G 单项 objective，而是做更明确的问题判定：

$$
\boxed{
\text{在 MLP 上，当前 loss-agnostic functional geometry maintenance 是否存在 generic mechanism？}
}
$$

## 7.2 候选

只保留三类机制，不再无限扩：

```text
M-H1-UnlabeledMicroProbeResponsePredictor
M-H2-RandomCotangentLowRankResponseController
M-H3-ActivationSpectrumGuardWithMatchedControls
```

这些 candidate 只能使用：

```text
unlabeled train-stream inputs；
activations；
logits；
random cotangent sketches；
precommit clone response that does not use labels/CE。
```

不能使用：

```text
labels；
CE vector；
permuted CE；
validation/test；
future outcome；
dataset-name branch。
```

## 7.3 Gate

Row-level exploration：

$$
source\_vs\_control \ge 0.005,
$$

$$
LineC\_pass\_count \ge 4/5,
$$

$$
CouplingR2\_delta \ge 0,
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

Aggregate pass requires：

```text
>= 8/9 dataset/seed pass，且 no single dataset concentration。
```

## 7.4 No-go rule

If M-H1..H3 fail with controls:

```text
MLPFunctionalNoGo_CurrentLossAgnosticObservableFamily = 1
```

Then do not spend more budget on MLP functional until a new theoretical observable is introduced.

---

# 8. Line A / F：Label-free FHQ monitor and KAN functional shadow

## 8.1 目标

FHQ / A-DYN 不再是主线，但需要 low-budget monitor，防止错过 accidental recovery。

候选：

```text
A-DYN baseline monitor only；
no label-informed init；
no trainprobe；
no y_for_stats。
```

进入 functional shadow 的条件：

```text
mean_delta_vs_MLP >= -0.005
worst_delta_vs_MLP >= -0.02
AUC_time <= 1.10
LineC_pass >= 5/9
```

否则 Line F 不跑。

---

# 9. 必须记录的指标

## 9.1 Basis workspace fields

```text
candidate_id
family
exact_kernel_implemented
basis_materialized
derivative_materialized
readout_grad_materialized
forward_peak_bytes
backward_peak_bytes
update_peak_bytes
raw_peak_ratio_vs_mlp
incremental_peak_ratio_vs_mlp
step_ratio_vs_mlp
forward_ms
backward_ms
update_ms
workspace_temp_bytes
optimizer_state_bytes
```

## 9.2 Task / trajectory fields

```text
mean_delta_vs_MLP
worst_delta_vs_MLP
near_pass_rate
AUC_step_ratio_vs_MLP
AUC_time_ratio_vs_MLP
NLL_delta_vs_MLP
ECE_delta_vs_MLP
CEp99_delta_vs_MLP
margin_p10_delta_vs_MLP
classwise_worst_delta
```

## 9.3 LineC fields

```text
CouplingR2
CouplingR2_delta
NoiseSignalLeak
NoiseSignalLeak_delta
RealSignalReservoirRatio
RealSignalReservoirRatio_delta
LineC_pass_count
KernelDrift
signal_mass_topk
reservoir_fraction
```

## 9.4 Rational-specific fields

```text
den_min_batch
den_p01_batch
den_condition_batch
r_prime_p95
r_prime_p99
r_double_prime_p95
r_double_prime_p99
rational_output_p99
logit_norm_p99
unlabeled_entropy_mean
unlabeled_entropy_p05
top1_top2_gap_p01
```

## 9.5 Functional fields

```text
functional_source_id
control_id
source_vs_noop
source_vs_control
control_gap
LineC_pass_count
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
NLL_delta
ECE_delta
uses_label
uses_ce_vector
uses_validation_or_test
uses_query_batch
precommit_available
```

---

# 10. 必须生成的可视化

```text
fig_v1232_progress_by_line.svg
fig_v1232_workspace_pareto_by_family.svg
fig_v1232_rational_tail_task_linec_tradeoff.svg
fig_v1232_rational_denominator_derivative_vs_CEp99.svg
fig_v1232_nonrat_incremental_memory_breakdown.svg
fig_v1232_family_expression_task_linec_radar.svg
fig_v1232_mlp_functional_control_gap.svg
fig_v1232_linec_task_colocation_scatter.svg
fig_v1232_route_dashboard.svg
```

特别要求：

```text
1. Rational 图必须同时画 mean_delta、worst_delta、AUC_time、CEp99、LineC，不允许单独画 task success。
2. Non-RAT 图必须显示 raw peak 和 incremental peak 的差异。
3. MLP functional 图必须显示 source_vs_control 与 CouplingR2_delta，不允许只报 LineC pass_count。
```

---

# 11. Codex 执行规则与 fallback

## 11.1 不允许 fail-fast

Promotion fail closed，但 exploration 必须继续到规定深度。若某 line 失败，必须执行对应 fallback；若未执行，route 必须为：

```text
R0-ExploreDepthIncomplete
```

## 11.2 Rational fallback depth

```text
Depth 1: true kernel / workspace / correctness
Depth 2: denominator / derivative stability
Depth 3: task/AUC hardening
Depth 4: tail-stability no-CE repair
Depth 5: LineC role-wise audit
Depth 6: no-go boundary or new mechanism proposal
```

## 11.3 Non-RAT fallback depth

```text
Depth 1: exact_kernel_implemented audit
Depth 2: no-materialize workspace test
Depth 3: A4 expression smoke
Depth 4: task triage only if workspace + A4 pass
Depth 5: family-specific no-go or next fused-kernel mechanism
```

## 11.4 MLP functional fallback depth

```text
Depth 1: M-H1/H2/H3 run with controls
Depth 2: feature ablation and sign sanity
Depth 3: matched control gap audit
Depth 4: no-go boundary for current observable family
```

---

# 12. Success / route definitions

## S1：Rational workspace and kernel truth opened

```text
exact_kernel_implemented = 1 or explicitly audited existing kernel path
workspace pass across 3x3
strong pass >= 6/9 for at least one candidate
```

## S2：Rational near-pass

```text
mean_delta >= 0
worst_delta >= -0.01
AUC_time <= 1.00
CEp99_delta <= 0.05
LineC_pass >= 25/27
```

## S3：Non-RAT true kernel opened

```text
exact_kernel_implemented = 1
workspace pass >= 8/9
A4 expression pass
```

## S4：MLP functional generic positive

```text
MLP functional >= 8/9 pass
source_vs_control >= 0.005
LineC majority/all pass
CEp99/NLL/ECE non-harm
```

## S5：KAN functional official re-entry

Only after a label-free KAN or classic basis near-pass exists：

```text
functional event beats NoOp / Random / AdamWParallel / SNR-only
LineC all pass
CEp99/NLL/ECE non-harm
train-shuffle robust
no label / CE / query / validation / dataset-name branch
```

## Failure routes

```text
R1-RationalTailStabilityBlocked
R2-BasisTaskLineCNotColocated
R3-NonRATWorkspaceBlocked
R4-MLPFunctionalNoGoCurrentFamily
R5-NoLabelFreeBaseForFunctional
R6-ExactKernelNotImplemented
R0-ExploreDepthIncomplete
```

---

# 13. v12.32 最终判断标准

v12.32 不以“再跑了多少候选”为成功。成功必须至少满足一个：

```text
1. Rational 达到 S2 near-pass；
2. 至少一个 non-RAT family 达到 S3 true kernel opened；
3. MLP functional 达到 S4 generic positive；
4. 明确证明 MLP functional current family no-go，并把 functional 主线限定到 KAN/basis coordinate；
5. 提供可执行的新 mechanism，不只是继续 token grid。
```

如果以上都没有发生，但 artifacts 完整，则仍是科学 no-go，不是项目成功。

---

# 14. 一句话执行摘要

v12.31 证明 Rational workspace 已经打开，但 task / LineC / AUC / tail 不能同位；非 Rational family 仍被 incremental memory 阻塞；MLP functional 没有 generic positive。v12.32 不应继续 alias/token 网格，而应做三件事：

```text
1. Rational：从 workspace repair 升级到 denominator / derivative / tail-stability kernel；
2. Chebyshev/Fourier/RBF/Wavelet：实现真实 no-materialize fused kernels，而不是继续 alias；
3. MLP functional：做 mechanism-level no-go / positive test，停止轻量 M-G 变体扩散。
```

