# DG-KAN v12.31：Basis-Kernel-First、MLP Functional 机制检验与 Label-Free FHQ 监控完整计划

> 版本：v12.31 execution plan  
> 日期：2026-05-26  
> 基于：v12.30 `BasisFirst FunctionalAlsoMLP` 真实结果复盘  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 硬约束：strict FC-PureKAN；不使用 label-informed initialization；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no fake / proxy / CPU offload；functional direction 不使用 label、CE vector、permuted-label CE、validation/test、future outcome、query batch 或 LineC hard target；CE / NLL / ECE / CEp99 只能作为审计和坏化约束。

---

# 0. 项目总目标与当前进展

## 0.1 总目标

DG-KAN 项目的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是单独找到一个更快的 KAN layer。总目标是：

$$
\boxed{
\text{构建 label-free strict FC-PureKAN base，并通过 loss-agnostic functional update 获得比普通反向传播更好的训练几何与模型。}
}
$$

最终 claim 必须同时满足：

```text
1. base 是 label-free strict FC-PureKAN，不使用 trainprobe / y_for_stats / label-informed initialization；
2. forward / backward / step / memory 与 MLP 可比；
3. 表达力不打折，不能只靠降低复杂度换速度；
4. AUC-step / AUC-time 不输 MLP，不能只看 final accuracy；
5. LineC 几何健康：CouplingR2、NoiseSignalLeak、RealSignalReservoirRatio、tail、calibration 不坏；
6. functional update 必须 loss-agnostic，不针对 CE 设计；
7. functional update 必须击败 NoOp、RandomMatchedNorm、AdamWParallel、SNR-only、MLP analog 等 controls；
8. 数据集只能作为诊断切片，不能作为 controller / threshold / route 的条件。
```

当前我们已经知道：历史 B320-current / FHQ 工程能力很强，但它依赖 label-informed trainprobe initialization，因此不能再作为 official label-free base。v12.26.1-v12.28 证明，去掉 label-informed init 后，当前 label-free FHQ / A-LF / A-S / A-F / A-R family 没有恢复 B320-current 的 signal geometry。v12.30 把重点转向 classic no-BSpline basis 与 MLP functional 机制检验，但也没有产生 FamilyNearPass 或 MLP functional positive。

---

## 0.2 各条线当前进度：与上次 v12.28 对比

> 百分比是基于当前 gate、artifact 覆盖、机制清晰度和可继续性做的项目管理估计，不是报告中的官方数值。`能力进展` 指离 official success 的距离；`证据进展` 指我们是否更清楚地知道 blocker 在哪里。

| 线 | v12.28 估计 | v12.30 估计 | 变化 | 当前判断 |
|---|---:|---:|---:|---|
| Line R：代码 / provenance / finalizer 审计 | 84% | **89%** | +5 | core 逻辑下沉到 `dgkan`，code packet / finalizer 更完整；这是工程审计进展，不是模型能力进展。 |
| Historical B320-current / FHQ 工程能力 | 85% frozen | **85% frozen** | 0 | 仍是历史强工程参考，但 label-informed init 被禁止，不能 official。 |
| Line A：label-free FHQ / signal-frame base | 38% | **34%** | -4 | A-DYN 只是低预算 monitor，无 near-anchor；继续扩 FHQ token 价值降低。 |
| Line C：Manifold-Channel 几何审计 | 82% | **83%** | +1 | 审计更成熟；仍不能作为 direction source。 |
| Line T：precommit / loss-agnostic value source | 33% | **28%** | -5 | label-free KAN 和 MLP functional 都没有找到可用 value source。 |
| Line F：KAN functional update | 12% | **10%** | -2 | 没有 label-free near-anchor，functional 只能 shadow；不能 official。 |
| Line M：MLP functional update | 新增 | **18%** | +18 | 已建立 MLP functional runner 与 controls，但 M-F1..M-F6 全部 0 pass；证明当前轻量 value source 不够。 |
| Line D：classic no-BSpline basis portfolio | 43% | **48%** | +5 | 27 个 alias、scout/hardening/fallback 都执行；证据更清楚，但 FamilyNearPass 仍为 0。 |
| Classic basis memory/workspace closure | 35% | **30%** | -5 | memory re-audit 后 raw peak 约 2.10、incremental peak 约 9.97，说明 memory blocker 更硬。 |
| 整体 next-gen MLP claim | 48%-52% | **47%-51%** | -1 | 路线更干净，但 official base + functional 都未闭合。 |

### 0.2.1 为什么 Line D 百分比上升，但整体略降

v12.30 做了更多 classic basis 真实 scout / hardening / fallback，因此我们对 basis blocker 的认识更完整；这让 Line D 的**证据完成度**提高。但 memory accounting re-audit 表明当前 classic basis workspace 相对 MLP 的额外峰值更严重，因此**能力完成度**没有提高，甚至更远。

---

# 1. v12.30 结果独立分析

## 1.1 本轮实际做了什么

v12.30 的核心变化不是继续 FHQ token grid，而是：

```text
1. 建立 classic no-BSpline alias registry：
   D-RAT1..D-RAT6, D-CHE1..D-CHE5, D-WAV1..D-WAV5, D-RBF1..D-RBF6, D-FOU1..D-FOU5。

2. 新增 A-DYN low-budget FHQ monitor：
   A-DYN1..A-DYN5，全部 uses_y_for_stats=0 / forbidden_token_present=0。

3. 新增 MLP functional diagnostic：
   M-F1..M-F5，后续 M-F6；controls 包括 TaskOnlyAdamW、NoOp、RandomMatchedNorm、AdamWParallel、SNR-only。

4. 执行 classic basis scout / hardening / fallback。

5. 执行 MLP functional M-F1..M-F6 + matched controls。

6. 执行 memory accounting re-audit。

7. 将部分核心逻辑从 runner 下沉到 `dgkan/diagnostics/classic_basis.py`、`dgkan/functional/mlp_functional.py`、`dgkan/training/eval.py`。
```

这说明 v12.30 不是第一关失败就停。它执行到 fallback depth 6，且 finalizer 给出：

```text
route = R1-BasisPortfolioNoProgress
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_family_near_pass_count = 0
basis_family_progress_count = 0
mlp_functional_candidate_rows = 486
mlp_functional_control_rows = 2916
mlp_functional_exploration_pass_rows = 0
mlp_functional_official_pass_rows = 0
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
```

所以 v12.30 的终止是计划内 hard budget exhausted，不是 Codex 又在第一个 gate fail 后偷停。

---

## 1.2 主要正进展

### 1.2.1 Classic basis 终于被系统性 scout / hardening / fallback

之前 classic basis 经常是零散 smoke 或 family-specific focused run。v12.30 第一次把 active no-BSpline family 都拉到同一个 BasisFirst 口径下：

```text
Rational
Chebyshev
Wavelet
RBF / FastKAN
Fourier
```

这使得我们不再只能说“这些 basis 没有 FamilyPass”，而能更具体地说：

```text
Rational：scout 时 task mean 接近，但 hardening 后 LineC 和 memory 崩；
Wavelet：D-WAV5 在 hardening 中 task 相对最好，但 step/memory/worst 失败；
Fourier：fallback 有一点 LineC / task 改善，但仍远离 near-pass；
Chebyshev：task/worst 仍差；
RBF：expression / task 没有恢复。
```

### 1.2.2 MLP functional 线建立起来了

这是重要结构进展。以前 functional update 总是在 KAN 上兜圈，很难判断问题来自：

```text
KAN parameterization；
base 不稳定；
functional value source 本身不成立；
或者只是我们没有找到对的 controls。
```

v12.30 新增 MLP functional diagnostic 后，已经可以问：

$$
\boxed{
\text{loss-agnostic functional update 是否是通用机制，还是 KAN-specific，还是当前根本没有 value source？}
}
$$

当前结果是：M-F1..M-F6 在 MLP 上也没有打过 matched controls。这个负结果很有价值，因为它说明当前 lightweight loss-agnostic objectives 不是简单迁移到 MLP 就能 work。

### 1.2.3 代码结构进步

把 classic basis registry、memory accounting、MLP functional sources 和 shared eval 从 runner 下沉到 `dgkan` 是必要的。它不产生科学 success，但能降低后续 Codex 在 runner 里偷换语义、复制代码、漏审 core 实现的风险。

---

## 1.3 主要负结论

### 1.3.1 Classic basis 的 memory blocker 变得更硬

v12.30 memory accounting 修复后：

```text
raw peak ratio ≈ 2.10
incremental peak ratio ≈ 9.97
```

这说明旧 memory 口径不是误伤。相反，incremental 口径把 family 训练本身相对 MLP 的额外峰值暴露得更明显。也就是说，classic basis 当前不是只差 task recipe，**workspace / basis materialization / update live-set** 是硬 blocker。

这也解释了为什么继续扩 alias/token 很危险：如果每个 candidate 都带着类似 memory/workspace 负担，task 小幅改善也没有系统价值。

### 1.3.2 Task / worst / LineC 仍没有同位

v12.30 hardening / fallback 中没有任何 FamilyNearPass：

```text
basis_family_near_pass_count = 0
exploration_pass_rows = 0
```

一些候选某一项接近：

```text
D-RAT4 scout: mean_delta 接近 0，LineC 9/9；但 step/memory/worst 不过。
D-WAV5 hardening: task 相对最好，LineC 5/9；但 step/memory/worst 不过。
D-FOU4 fallback: LineC 3/9，有一点改善；但 task/worst/memory 仍远。
```

这说明当前 classic basis 的问题不是单一 gate，而是四项无法同位：

$$
\boxed{
\text{task mean} + \text{worst/AUC} + \text{LineC} + \text{memory/step}
}
$$

### 1.3.3 MLP functional 目前没有 generic positive

合并后 MLP functional canonical artifact：

```text
candidate_rows = 486
control_rows = 2916
linec_rows = 4860
gate_rows = 18
mlp_functional_exploration_pass_rows = 0
mlp_functional_official_pass_rows = 0
```

M-F6 input-jitter 说明无标签输入扰动一致性可以保持 LineC 不撕裂，但不能击败 matched controls。因此 MLP functional 不能 promotion。

这给了一个重要判断：

$$
\boxed{
\text{继续给 M-F family 增加同类单项正则式 objective，大概率只是低价值搜索。}
}
$$

---

# 2. 当前核心问题：不是“没跑够”，而是机制层级不对

v12.30 之后，核心问题不再是：

```text
有没有再多跑几个 candidate？
有没有再调几个 scale / temperature / rank？
```

而是：

$$
\boxed{
\text{当前 basis implementation 与 functional value source 的机制层级都不够。}
}
$$

具体表现为两条硬边界：

## 2.1 Basis 线硬边界

Classic basis 当前仍然像“通过 PyTorch/Triton wrapper 试不同 primitive spec”，但真正 blocker 是 basis/workspace 的 lower-level memory model：

```text
1. basis activation 是否 materialize 了过大的 [B, d, K] 或 role-wise workspace；
2. coefficient/readout grad 是否产生巨大临时张量；
3. update live-set 是否和 forward/backward workspace 叠加；
4. MLP reference 的 incremental peak 太低，KAN family 的额外临时量被放大；
5. raw peak ratio 约 2.10 也已经高于 desired envelope。
```

因此下一步不能只是 D-RAT7、D-WAV6 这种 alias 扩展，而要做：

$$
\boxed{
\text{basis kernel/workspace memory-first redesign。}
}
$$

## 2.2 Functional 线硬边界

MLP 上 M-F1..M-F6 全部失败，说明当前 value source family 没有 generic signal。KAN 上 label-free base 又没有 near-anchor，因此 functional update 没有合法承载体。下一步 functional 不应继续轻量 objective token，而应回答两个更基础的问题：

```text
1. 在 MLP 上是否存在 loss-agnostic precommit geometry maintenance 机制？
2. 如果 MLP 不存在，是否只有 KAN / explicit basis coordinates 才能提供 functional update 的有效 actuator？
```

---

# 3. v12.31 总体策略

v12.31 不再继续 v12.30 的 alias 扩展。它采用三条主线：

```text
Line D：Basis Kernel / Workspace Memory-First Redesign
  目标：先打穿 classic basis 的 memory/workspace blocker，再谈 task hardening。

Line M：MLP Functional Mechanism No-Go / New Mechanism Test
  目标：判断 functional update 是否可能是通用 MLP mechanism；如果当前 objective family 不行，必须做机制级替代而不是 token 扩展。

Line A/F：Label-Free KAN Monitor + Functional Re-entry Gate
  目标：只监控 label-free FHQ / classic near-pass，不再把没有 near-anchor 的 base 强行接 functional。
```

计算预算建议：

```text
Line D basis kernel/workspace: 50%-55%
Line M MLP functional mechanism: 20%-25%
Line A/F label-free KAN monitor + shadow functional: 10%-15%
Line R/C/Z audit/finalizer/visualization: 10%-15%
```

---

# 4. Line D：Classic no-BSpline basis kernel / workspace memory-first redesign

## 4.1 总目标

Line D 的目标不再是“再试一些 basis aliases”，而是：

$$
\boxed{
\text{为每个 active family 找到真实 lower-level memory model，并至少让一个 family 同时满足 memory + task/LineC near condition。}
}
$$

Active family：

```text
Rational
Wavelet
Chebyshev
RBF / FastKAN
Fourier
```

Frozen：

```text
B-spline
```

---

## 4.2 共同 P0：workspace truth audit

### 目标

在继续训练前，先定位 memory peak 到底来自哪里。

### 必须记录 CSV：`v1231_basis_workspace_truth.csv`

字段：

```text
run_id
family
candidate_id
dataset
seed
batch_size
hidden_dim
basis_K
group_count
forward_peak_bytes
backward_peak_bytes
update_peak_bytes
raw_peak_bytes
incremental_peak_bytes
mlp_raw_peak_bytes
mlp_incremental_peak_bytes
raw_memory_ratio_vs_mlp
incremental_memory_ratio_vs_mlp
basis_activation_bytes
basis_derivative_bytes
readout_grad_bytes
coeff_grad_bytes
optimizer_state_bytes
workspace_temp_bytes
saved_tensor_bytes
num_custom_kernels
num_torch_ops
num_gemm_calls
num_exp_calls
num_sin_cos_calls
num_gather_scatter_calls
uses_dense_basis_materialization
uses_recompute_backward
uses_chunked_grad
uses_fused_update
```

### 判断标准

进入 task hardening 前，family 必须至少达到 exploratory workspace gate：

$$
raw\_memory\_ratio \le 1.50,
$$

$$
incremental\_memory\_ratio \le 3.00,
$$

$$
step\_ratio \le 1.75.
$$

强目标：

$$
raw\_memory\_ratio \le 1.20,
$$

$$
incremental\_memory\_ratio \le 2.00,
$$

$$
step\_ratio \le 1.30.
$$

### Codex fallback

如果 memory fail：

```text
1. 不进入 task grid。
2. 自动运行 component peak profiler。
3. 识别 top peak source：basis activation / derivative / readout grad / optimizer state / temp workspace。
4. 针对 top source 执行 family-specific kernel repair。
5. 修复后重跑 workspace truth，不允许直接跳到 task。
```

---

## 4.3 Rational：最高优先级

### 当前状态

Rational 在 scout 中最接近 task/LineC，但 hardening 后 memory 和 LineC 退化。v12.30 显示 Rational memory ratio raw 约 2.10，incremental 约 9.97；hardening 后 LineC 只有 3/9，mean/worst 也不够。

### 假设

$$
\boxed{
\text{Rational 的主要 blocker 是 readout/denominator/hidden residual workspace 与 update live-set，而不是 rational 函数本身表达力。}
}
$$

### 候选

```text
D-RAT7-FusedGroupRationalNoMaterialize
D-RAT8-RecomputeDenominatorBackward
D-RAT9-ChunkedReadoutGradNoPersistentBasis
D-RAT10-FusedDenNumReadoutGrad
D-RAT11-LowMemGroupSharedDenomPlusLineC
D-RAT12-RationalWorkspaceMinStrongDiag
```

### 必须记录

除了 workspace truth 外，还要记录：

```text
den_min
den_p01
den_condition
r_prime_p95
r_double_prime_p95
group_function_diversity
readout_grad_peak_bytes
den_grad_peak_bytes
num_grad_peak_bytes
hidden_residual_peak_bytes
LineC_CouplingR2
LineC_pass_count
NoiseSignalLeak
RealSignalReservoirRatio
CEp99
NLL
ECE
```

### 进入 task hardening 标准

$$
raw\_memory\_ratio \le 1.50,
$$

$$
incremental\_memory\_ratio \le 3.00,
$$

$$
step\_ratio \le 1.60,
$$

且 denominator safety：

$$
den\_p01 > 10^{-3}.
$$

### Task hardening gate

```text
mean_delta_vs_MLP >= -0.015
worst_delta_vs_MLP >= -0.035
AUC_time_ratio_vs_MLP <= 1.10
LineC pass >= 5/9
CEp99 <= MLP + tolerance
```

### Codex fallback

如果 Rational memory fail：

```text
1. 分离 readout grad、denominator grad、numerator grad peak。
2. 若 readout grad 是 top source：启用 chunked readout grad + recompute activation。
3. 若 denominator grad 是 top source：启用 fused den/num derivative kernel，不保存 powers。
4. 若 optimizer state 是 top source：启用 bf16 state diagnostic 和 update-after-free schedule。
5. 仍 fail：降低 group count 或 hidden residual，但必须重新跑 A4/LineC，不允许只看 memory。
```

如果 Rational task fail but memory pass：

```text
1. 先查 LineC：CouplingR2 / NoiseSignalLeak / ReservoirRatio。
2. 如果 CouplingR2 低：加 linec residual mix，但不增加 persistent workspace。
3. 如果 CEp99/NLL/ECE 坏：只做 audit，不用 CE 设计方向。
4. 如果 worst/AUC 坏：做 early trajectory analysis，不按 dataset 调参。
```

---

## 4.4 Wavelet：第二优先级之一

### 当前状态

D-WAV5 在 hardening 中 task 相对最好、LineC 达到 5/9，但 step/memory/worst 都不过。Wavelet 不是完全没信号，但 local basis workspace 和 trajectory 不健康。

### 假设

$$
\boxed{
\text{local wavelet 可以提供 tail / local stroke coverage，但当前 scale/support implementation 太贵且 worst-row 不稳。}
}
$$

### 候选

```text
D-WAV6-HatWaveletNoMaterializeLocal2
D-WAV7-SharedScaleLocalWavelet
D-WAV8-ScaleEntropyCappedWavelet
D-WAV9-TailCoverageNoExtraWorkspace
```

### Gate

```text
raw_memory_ratio <= 1.50
incremental_memory_ratio <= 3.00
step_ratio <= 1.75
mean_delta_vs_MLP >= -0.030
worst_delta_vs_MLP >= -0.060
LineC pass >= 5/9
```

### Codex fallback

```text
如果 memory fail：只允许 local support K=2/4，禁止 Morlet/MexicanHat heavy path。
如果 task fail：检查 active_support_count、scale_entropy、tail_coverage。
如果 LineC fail：尝试 shared-scale / scale entropy cap，不增加 basis count。
```

---

## 4.5 Chebyshev：task-trajectory repair，不再加 degree

### 当前状态

Chebyshev 已有 efficiency/expression 路径，但 task / worst / AUC / LineC 坏。

### 假设

$$
\boxed{
\text{Chebyshev 的问题是 global polynomial 高阶能量导致 task trajectory / signal-reservoir 不稳。}
}
$$

### 候选

```text
D-CHE6-K3DegreeEnergyCap
D-CHE7-LateEnableK4
D-CHE8-OrthogonalInputNormNoExtraMem
D-CHE9-RoleWiseLowDegreeResidual
```

### Gate

```text
high_degree_energy_ratio <= threshold
raw_memory_ratio <= 1.50
step_ratio <= 1.60
mean_delta_vs_MLP >= -0.030
LineC pass >= 4/9 exploratory, 6/9 official
```

### Codex fallback

```text
如果 high_degree_energy 高：降低高阶 scale，不增加 K。
如果 task fail：late-enable K4，不做 K6。
如果 LineC fail：检查 RealSignalReservoirRatio 是否升高。
```

---

## 4.6 RBF / FastKAN：expression repair while preserving efficiency

### 当前状态

RBF / FastKAN 的 compact path 有 efficiency 路径，但 expression / task 不够。

### 假设

$$
\boxed{
\text{当前 compact RBF K_active 太窄，无法覆盖 expression；但 dense RBF 不能回归。}
}
$$

### 候选

```text
D-RBF7-CompactK4PlusLinearResidual
D-RBF8-QuantileCentersFixedWidth
D-RBF9-TriangularBumpNoExp
D-RBF10-LocalRBFExpressionPatch
```

### Gate

```text
A4 expression pass required before task hardening
raw_memory_ratio <= 1.50
step_ratio <= 1.75
out_of_grid_fraction low
dead_center_fraction <= 0.30
```

### Codex fallback

```text
如果 A4 fail：增加 local residual，而不是 dense K。
如果 exp 成本高：triangular bump / exp2 approximation diagnostic。
如果 dead center 高：quantile centers + wider fixed width。
```

---

## 4.7 Fourier：low-frequency expression repair，严防 noise leak

### 当前状态

Fourier 效率路径已有，但 lowfreq compact version expression 不够；fallback 有一点 LineC，但 task gap 大。

### 假设

$$
\boxed{
\text{Fourier 需要少量 multi-scale expression capacity，但不能让高频噪声进入 signal channel。}
}
$$

### 候选

```text
D-FOU6-LowFreqPlusIdentityResidual
D-FOU7-LateEnableK4WithEnergyCap
D-FOU8-PhaseAmplitudeSharedNoExtraMem
D-FOU9-FourierNoiseLeakGuard
```

### Gate

```text
A4 expression pass required
high_freq_energy_ratio <= threshold
NoiseSignalLeak <= MLP + tolerance
raw_memory_ratio <= 1.50
step_ratio <= 1.75
```

### Codex fallback

```text
如果 expression fail：加 identity residual 或 K4 late-enable。
如果 NoiseSignalLeak 高：降低 high frequency energy，不调 CE。
如果 memory fail：sincos shared / no dense basis materialization。
```

---

# 5. Line M：MLP functional mechanism test

## 5.1 当前结论

v12.30 的 MLP functional：

```text
candidate_rows = 486
control_rows = 2916
linec_rows = 4860
mlp_functional_exploration_pass_rows = 0
mlp_functional_official_pass_rows = 0
```

M-F6 input-jitter 能保持部分 LineC non-tearing，但不能击败 matched controls。继续在 M-F1..M-F6 这类单项 objective 上加变体，预期价值低。

## 5.2 新假设

### H-M1：Functional update 不是通用 MLP 机制

如果 MLP 上没有任何 loss-agnostic functional value source 能击败 controls，而 KAN / classic basis 后续能击败 controls，则 functional update 可能依赖显式 basis coordinate。

### H-M2：当前 M-F family 太弱，需要 response-based but precommit-safe microprobe

允许在 commit 前用 unlabeled train-stream clone probe 观察几何 response，但不能用 label / CE / validation / future。也就是说，functional value 可以来自：

```text
unlabeled activation response
logit covariance response
random cotangent response
input perturbation consistency response
optimizer-state-free geometry response
```

但不能来自：

```text
CE descent
label residual
validation performance
LineC hard target
future trajectory
```

### H-M3：如果 MLP 上仍没有 signal，则 MLP functional 线给出 no-go boundary

不再无限试 M-F token，而是输出：

```text
MLPFunctionalNoGo_CurrentLossAgnosticFeatureFamily
```

## 5.3 实验设计

### M0：matched-control audit stability

记录：

```text
method
candidate_id
control_id
dataset
seed
window
source_norm
control_norm
matched_norm_error
CouplingR2_delta
NoiseSignalLeak_delta
ReservoirRatio_delta
NLL_delta
ECE_delta
CEp99_delta
source_vs_noop
source_vs_best_control
control_gap
```

Gate：NoOp / Random / AdamWParallel / SNR-only 必须全部存在，同 batch/window/seed 匹配。

### M1：precommit microprobe value source

新增候选：

```text
M-G1-UnlabeledCloneResponseCov
M-G2-RandomCotangentResponseStability
M-G3-InputPerturbationConsistencyTransport
M-G4-HiddenCovarianceTransportWithControlResidual
M-G5-ActivationSubspaceGuardThenTaskNeutral
```

注意：这些不是训练 loss。它们只在 cloned precommit state 上用无标签几何 response 生成 direction。

Gate：

$$
source\_vs\_best\_control \ge 0.005,
$$

$$
\Delta CouplingR2 \ge 0.02,
$$

$$
\Delta NoiseSignalLeak \le 0,
$$

$$
\Delta CEp99 \le 0.05,
$$

且 train-shuffle robust：

```text
>= 2/3 train-shuffle seeds positive
```

### M2：MLP short-run only if M1 pass

如果 M1 不过，不开 M2。若 M1 过，跑短训：

```text
MLP + AdamW
MLP + NoOp event
MLP + best M-G functional event
MLP + AdamWParallel control
```

记录：

```text
AUC_step
AUC_time
acc_delta
NLL
ECE
CEp99
LineC trajectory
amortized overhead
control gap
```

## 5.4 Codex fallback

如果 M-F/M-G 全 fail：

```text
1. 不再增加同类 single-objective functional token。
2. 输出 MLP functional no-go boundary。
3. 将 functional 主预算转回 KAN/classic near-pass base。
```

如果 MLP pass 但 KAN fail：

```text
说明 functional 是 generic optimizer-like mechanism；要在 KAN 上找 actuator / parameterization mismatch。
```

如果 MLP fail 但 KAN/classic pass：

```text
说明 functional 可能需要 KAN basis coordinate；后续以 KAN-specific functional 为主。
```

---

# 6. Line A/F：Label-free FHQ monitor 与 functional re-entry gate

Line A 不再是主预算线。v12.28-v12.30 已经显示，label-free FHQ token/fame family 没有 near-anchor。v12.31 只允许：

```text
1. A-DYN monitor；
2. 不超过 10%-15% 计算预算；
3. 只在出现 label-free near-anchor 时重开 functional bridge。
```

Near-anchor gate：

```text
mean_delta_vs_MLP >= -0.005
worst_delta_vs_MLP >= -0.015
AUC_time_ratio <= 1.05
LineC pass >= 5/9
memory/step 不劣于 MLP envelope
```

如果没有 near-anchor：

```text
Line F official remains closed
functional only shadow diagnostic
```

---

# 7. Line C：Manifold-Channel Diagnostics

Line C 继续作为审计，不作为 direction source。

必须记录：

```text
CouplingR2
CouplingCorr
NoiseSignalLeak
RealSignalReservoirRatio
KernelDrift
CEp99
NLL
ECE
MarginP10
LineC pass count
LineC all-pass / majority-pass
multi-sketch variance
```

解释规则：

```text
good:
  CouplingR2 高，NoiseSignalLeak 低，ReservoirRatio 低，tail/calibration 不坏。

bad task-only:
  accuracy/task margin 提高，但 LineC 低或 CEp99/NLL/ECE 坏。

bad geometry-only:
  LineC 提高，但 task/control margin 不够。

bad memory:
  LineC/task 局部好，但 workspace/memory 不可接受。
```

---

# 8. Line R：代码审查与 implementation readback

Codex 每轮必须在复盘中写清楚：

```text
1. 新增或修改的核心代码文件；
2. 关键 class/function/kernel；
3. 使用的 tensor shape；
4. 是否 materialize basis；
5. backward / update 是否 fused / recompute / chunked；
6. 是否读取 label / CE / validation / dataset name；
7. artifact 字段与 gate 的关系；
8. 如果结果失败，失败发生在实现层还是机制层。
```

缺失则 route：

```text
R0-ImplementationReadbackIncomplete
```

---

# 9. 必须生成的 artifact

## 9.1 Basis line

```text
v1231_basis_workspace_truth.csv
v1231_basis_component_peak.csv
v1231_basis_microkernel_correctness.csv
v1231_basis_hardening.csv
v1231_basis_linec.csv
v1231_family_status.csv
```

## 9.2 MLP functional line

```text
v1231_mlp_functional_candidates.csv
v1231_mlp_functional_controls.csv
v1231_mlp_functional_linec.csv
v1231_mlp_functional_gate.csv
v1231_mlp_functional_no_go.md
```

## 9.3 FHQ monitor

```text
v1231_adyn_monitor.csv
v1231_label_free_near_anchor.csv
```

## 9.4 Finalizer

```text
v1231_route_decision.json
v1231_required_artifact_manifest.csv
v1231_fallback_manifest.csv
v1231_no_go_boundary.md
v1231_next_hypothesis_queue.md
v1231_code_review_packet.zip
```

---

# 10. 必须可视化

```text
fig_progress_by_line.svg
fig_basis_memory_truth_raw_vs_incremental.svg
fig_basis_workspace_waterfall_by_family.svg
fig_rational_memory_component_breakdown.svg
fig_classic_task_linec_memory_pareto.svg
fig_mlp_functional_control_gap.svg
fig_mlp_functional_linec_vs_control.svg
fig_label_free_fhq_monitor.svg
fig_route_waterfall.svg
```

---

# 11. v12.31 route 定义

## Success routes

```text
S1-BasisWorkspaceOpened:
  至少一个 classic family 过 workspace exploratory gate。

S2-FamilyNearPass:
  至少一个 classic family 同时满足 task/worst/AUC/LineC/memory near gate。

S3-MLPFunctionalGenericPositive:
  MLP functional M-G family 打过 matched controls，并 task/tail/LineC 不坏。

S4-KANFunctionalReentryOpened:
  label-free KAN/classic near-anchor 出现，functional P3 可打开。

S5-OfficialFunctionalSuccess:
  source_vs_control、LineC all-pass、tail/calibration、train-shuffle robustness 全部通过。
```

## Fail routes

```text
R1-BasisWorkspaceStillBlocked:
  no family 过 workspace exploratory gate。

R2-BasisTaskLineCNotColocated:
  有 family 过 workspace，但 task/LineC/worst 不同位。

R3-MLPFunctionalNoGoCurrentFamily:
  MLP M-G family 仍全部 fail controls。

R4-LabelFreeKANNearAnchorMissing:
  FHQ/classic 没有 near-anchor，functional official closed。

R0-ExploreDepthIncomplete:
  gate fail 后没有执行规定 fallback。

R0-ImplementationReadbackIncomplete:
  缺代码实现解释。
```

---

# 12. 最终判断

v12.30 已经说明：

```text
1. classic basis 不是完全没价值，但当前 memory/workspace 与 task-LineC 同位没有闭合；
2. MLP functional diagnostic 很重要，但当前 M-F1..M-F6 没有 generic positive；
3. label-free FHQ token route 边际收益低；
4. 继续扩同类 alias / value token 会变成低价值网格搜索。
```

v12.31 的核心是换层级：

$$
\boxed{
\text{basis 线从 candidate alias 扩展升级为 kernel/workspace memory-first redesign；}
}
$$

$$
\boxed{
\text{functional 线从轻量 objective token 升级为 MLP generic mechanism/no-go 检验。}
}
$$

如果 v12.31 仍然失败，但能够给出明确：

```text
Rational memory source；
Wavelet task/LineC source；
RBF/Fourier expression source；
MLP functional no-go boundary；
```

那也是有效进展。不能接受的是继续排列同类 token，得到又一个 `R1-BasisPortfolioNoProgress`。

