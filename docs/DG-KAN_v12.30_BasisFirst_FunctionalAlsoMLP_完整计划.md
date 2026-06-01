# DG-KAN v12.30：Basis-First 主线重排 + Functional Update 同时验证 MLP 的完整实验计划

> 版本：v12.30 execution plan  
> 基于：v12.28 / v12.29 的 label-free base recovery 结果、classic no-BSpline basis 进展、以及 functional S4a-to-S5 失败边界。  
> 本版核心修正：如果 functional update 在当前 label-free PureKAN 上仍然太难，下一阶段必须把 **classic function basis 做 work** 放到更高优先级；同时，functional update 不能只在 KAN 上验证，也必须在 **MLP** 上同步验证，区分“通用 optimizer / geometry maintenance”与“KAN-specific synergy”。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；label-free initialization only；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no fake / proxy / CPU offload；functional direction 不使用 label、CE vector、permuted-label CE、validation/test、future outcome、query batch 或 LineC hard target；CE / NLL / ECE / CEp99 只能作为审计与坏化约束，不能作为 functional direction source。

---

# 0. 项目总目标、当前进展与本版战略调整

## 0.1 项目总目标

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是只做一个更快的 KAN primitive。最终目标是：

$$
\boxed{
\text{构建 label-free strict FC-PureKAN base，并通过 loss-agnostic functional update 获得比普通反向传播更好的训练几何和模型。}
}
$$

最终要证明：

$$
\boxed{
\text{label-free PureKAN base + functional update}
>
\text{same label-free PureKAN base + ordinary backprop / AdamW controls}
}
$$

并且必须同时满足：

```text
1. 表达力不打折，不能靠低表达或过平滑换几何。
2. forward / backward / step / memory 与 MLP 可比。
3. 训练轨迹健康，AUC-step / AUC-time 不输 MLP 或不输 base。
4. 几何健康：CouplingR2、NoiseSignalLeak、RealSignalReservoirRatio、tail、calibration 不坏，最好改善。
5. functional update 的收益必须击败 NoOp / RandomMatchedNorm / AdamWParallel / SNR-only / matched role controls。
6. 不使用 label-informed initialization，不使用 CE / label / validation / test / future outcome 设计 functional direction。
```

其中 functional update 不是主 task optimizer。默认训练仍然是：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{task}
+
\lambda_t\Delta\theta_{func},
$$

其中：

```text
Δθ_task:
  普通 supervised task optimizer，例如 manual AdamW。
  它可以使用训练标签，因为 supervised training 本身使用 label。

Δθ_func:
  loss-agnostic geometry maintenance event。
  它不能直接读取 label、CE vector、permuted-label CE、validation/test、future outcome、query batch 或 LineC hard target。
```

## 0.2 当前进展简述

历史 FHQ / B320-current 证明过：strict FC-PureKAN 可以做到强效率、强 task、健康 AUC 与 LineC non-tearing。但 B320-current 使用了 label-informed trainprobe initialization，因此已经退役为历史 reference，不能再作为 official base、functional anchor 或 external-ready claim。

从 v12.26.1 开始，项目切换到 label-free-only。v12.27 / v12.28 做了多轮 label-free signal-source / early frame-formation 尝试，但尚未恢复 label-free near-anchor，也没有恢复 label-free S4a functional re-entry。

v12.28 的关键事实：

```text
route = R1-LabelFreeBaseRecoveryMissing
line_a_near_anchor_pass_count = 0
line_a_official_pass_count = 0
line_f_exploration_gate_pass = 0
line_f_official_gate_pass = 0
classic_meaningful_progress_count = 0
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
```

这说明 v12.28 不是 Codex 没跑够，而是现有 label-free static / early frame token family 没有形成可用机制。

## 0.3 本版战略调整

v12.29 的重点是 label-free dynamic signal-frame formation，但你指出两个关键问题：

```text
1. 如果 functional update 太难，应该先把其他函数基做 work。
2. functional update 也应该考虑 MLP，而不是只在 KAN 上验证。
```

我同意。v12.30 因此做三项战略重排。

### 调整一：Basis-first portfolio 升级为主线之一

Classic no-BSpline basis 不再只是低预算支线，而是下一阶段的 **主并行线**。原因是：

```text
1. label-free FHQ/B320-like base 暂时没有恢复 near-anchor。
2. functional update 需要一个健康 base 才有稳定机制可验证。
3. Rational / Chebyshev / Wavelet / RBF / Fourier 中，多数已经不是纯 efficiency fantasy；它们分别卡在 memory、expression、task trajectory 或 LineC。
4. 如果能先做出一个 label-free classic-basis base，就可以绕开当前 FHQ label-free signal-frame formation 的瓶颈。
```

本版 active basis：

```text
D-RAT: Rational
D-CHE: Chebyshev
D-WAV: Wavelet
D-RBF: RBF / FastKAN
D-FOU: Fourier
```

B-spline 继续 frozen，不投入 active budget。

### 调整二：Functional update 必须同时验证 MLP

如果 functional update 是一个真正的 loss-agnostic geometry maintenance 机制，它不应该只在 KAN 上能测。MLP 上至少应该出现以下三种情况之一：

```text
Case M1: MLP + functional 也有改善。
  说明 functional update 是通用 optimizer / geometry maintenance 方法。
  然后要问 KAN 是否提供更强 synergy。

Case M2: MLP + functional 无效，但 KAN + functional 有效。
  说明收益可能是 KAN-specific geometry / basis coordinate 贡献。

Case M3: MLP 与 KAN 都无效。
  说明当前 functional value source 仍然不成立，不应该继续在 KAN 上小修。
```

因此 v12.30 新增：

```text
Line M: Functional update on MLP。
```

它不是替代 KAN 主线，而是 functional update 的必要 control 和机制定位实验。

### 调整三：Functional 线降级为 cross-architecture diagnostic，直到 base 恢复

在没有 label-free near-anchor 之前，functional update 不能 official；但它不能停。新的定位是：

```text
1. 在 MLP 上验证通用 loss-agnostic functional mechanism。
2. 在 candidate KAN / classic-basis near-pass 上做 cloned diagnostic。
3. 只有 base 过 near-anchor 或 family near-pass 后，才允许 S4a re-entry。
4. 只有 S4a robust 后才进 S5 official。
```

---

# 1. 各条线当前完成度与 v12.30 目标

> 百分比不是 artifact 官方字段，而是根据 gate、机制清晰度、实现成熟度和距离最终目标的阶段性估计。

| 线 | v12.28/v12.29 前状态 | v12.30 目标 | 当前完成度 | 本版目标完成度 |
|---|---:|---|---:|---:|
| Line R：代码 / provenance / finalizer 审计 | artifacts 完整，provenance clean | 强制 readback，尤其 basis kernels 与 MLP functional | 84% | 90% |
| Historical FHQ / B320-current | 历史强，但 label-informed frozen | 仅作历史 reference | 85% frozen | frozen |
| Line A：label-free FHQ/B320-like base | no near-anchor | 低预算 monitor，不再主投 token 网格 | 38% | 45% |
| Line D：classic no-BSpline basis | 无 FamilyPass；Rational 最接近 | Basis-first 主并行线 | 43%-45% | 60% |
| D-RAT Rational | L3/A4 有路，task/LineC/memory 未闭合 | memory + task + LineC targeted repair | 55%-60% | 70% |
| D-CHE Chebyshev | TaskBlocked | degree-energy / task trajectory repair | 50%-55% | 62% |
| D-WAV Wavelet | TaskBlocked / ExpressionBlocked | local wavelet task stability | 45%-50% | 58% |
| D-RBF RBF/FastKAN | ExpressionBlocked | compact capacity repair | 35%-40% | 50% |
| D-FOU Fourier | ExpressionBlocked | lowfreq + residual expression repair | 35%-40% | 50% |
| Line C：Manifold-Channel 审计 | 审计可用 | basis + MLP + functional 统一审计 | 82% | 88% |
| Line M：Functional on MLP | 新增 | 验证 functional 是否通用 | 0%-10% | 35% |
| Line F：Functional on KAN | label-free 后 S4a=0 | 仅在 near-anchor / near-family 上 re-entry | 12%-15% | 25% |
| 整体 next-gen MLP claim | label-free base + functional 未闭合 | 先恢复 base/basis，再 functional | 48%-52% | 58%-62% |

v12.30 不要求一次完成 final success；它的 minimum useful success 是：

```text
1. 至少一个 classic no-BSpline family 从当前 blocker 推进一个明确阶段；
2. MLP functional line 完成第一轮 controlled diagnostic；
3. Functional 是否通用 / KAN-specific / 当前无效 形成清楚分叉判断；
4. label-free FHQ 不再消耗主预算于低价值 token grid。
```

---

# 2. 核心假设

## H-D：Classic basis 可以先于 functional update 取得 base 突破

当前 FHQ/B320-current 的历史能力证明了系统效率可达，但 label-free 去掉 trainprobe 后，signal channel 恢复失败。Classic basis 可能提供不同的 label-free inductive bias：

```text
Rational:
  通过 group rational / denominator-safe tangent geometry 提供更连续的 learnable edge function。

Chebyshev:
  通过低阶正交多项式提供稳定的全局低阶函数族。

Wavelet:
  通过 local support 捕捉局部 stroke / hard-tail structure。

RBF/FastKAN:
  通过 compact local centers 提供局部 smooth cover。

Fourier:
  通过低频 periodic basis 提供全局 smooth interaction reference。
```

如果 H-D 成立，则至少一个 family 会从当前 blocker 阶段推进到：

```text
FamilyNearPass 或 FamilyPass。
```

## H-M：Loss-agnostic functional update 应该能在 MLP 上暴露机制

如果 functional update 的本质是 signal/reservoir/noise geometry maintenance，而不是 KAN-specific basis correction，那么 MLP 上也应该出现可测信号。

形式上，比较：

$$
\Delta_G^{MLP}
=
G(MLP+FU)-G(MLP+AdamW),
$$

$$
\Delta_G^{KAN}
=
G(KAN+FU)-G(KAN+AdamW).
$$

若：

$$
\Delta_G^{MLP} > 0,
$$

且：

$$
\Delta_G^{KAN} \approx \Delta_G^{MLP},
$$

则 functional update 更像通用 optimizer / geometry method。

若：

$$
\Delta_G^{MLP} \le 0,
$$

但：

$$
\Delta_G^{KAN} > 0,
$$

则说明 KAN basis coordinate 与 functional update 有 synergy。

若二者都不成立，则当前 functional value source 不成立。

## H-C：LineC 是审计工具，不是方向源

LineC 继续用于审计：

```text
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
CEp99
NLL
ECE
AUC-time
```

但 LineC hard target 不允许进入 functional direction source。Functional direction 只能来自：

```text
T1: precommit loss-agnostic model-state features。
T1B: optimizer-observable but loss-agnostic features, exploration only unless explicitly promoted。
T2: clone-probe diagnostic, no promotion。
T3: audit-only CE/label metrics, never direction source。
```

## H-A：FHQ label-free base 仍可保留，但不是下一轮主预算

v12.28 已经表明 A-F/A-R token family 没有 near-anchor。v12.30 只保留少量 label-free dynamic signal-frame experiment，作为对照，不再把 50% 以上预算投在 FHQ token 网格。

---

# 3. 总执行结构

v12.30 分为七条线。

```text
Line R:
  Code / provenance / implementation readback。

Line D:
  Classic no-BSpline basis-first portfolio。

Line C:
  Unified Manifold-Channel Geometry Diagnostics。

Line M:
  Functional update on MLP。

Line F:
  Functional update on KAN / classic family near-pass candidates。

Line A:
  Label-free FHQ dynamic signal-frame monitor。

Line Z:
  Finalizer, route, no-go boundary, next-hypothesis queue。
```

预算建议：

```text
Line D classic basis: 45%-50%
Line M functional on MLP: 15%-20%
Line F functional on KAN/classic near-pass: 10%-15%
Line A label-free FHQ monitor: 10%-15%
Line C/R/Z shared audit: 10%-15%
```

如果资源紧张，优先级是：

```text
D-RAT > Line M > D-CHE/D-WAV > D-RBF/D-FOU > Line A > Line F shadow。
```

---

# 4. Line R：代码与 provenance 审计

## 4.1 目标

确保本轮不会再次出现：

```text
1. label-informed init 被误用；
2. LineC hard target 被误用为 direction source；
3. MLP functional 使用 CE vector 作为方向；
4. classic basis runner 只写 hypothesis 不执行；
5. efficiency path 走了 fallback autograd 却写成 fused/manual；
6. Codex 只写 route，不解释关键实现。
```

## 4.2 必须审查的代码面

Codex 必须在复盘中指出以下实现的真实文件、class/function、line range、调用链、关键 tensor shape。

```text
R0 entrypoint / finalizer / route decision。
R1 classic basis candidate registry。
R2 Rational forward/backward/update path。
R3 Chebyshev fused recurrence path。
R4 Wavelet local support kernel path。
R5 RBF/FastKAN compact center path。
R6 Fourier lowfreq fused path。
R7 MLP functional event path。
R8 KAN/classic functional event path。
R9 LineC metric computation。
R10 feature tier provenance: T1/T1B/T2/T3。
R11 controls: NoOp / Random / AdamWParallel / SNR-only / matched role controls。
R12 timing/memory profiler。
R13 label-free audit and forbidden token audit。
```

## 4.3 必须落盘 artifact

```text
v1230_code_review_manifest.csv
v1230_core_symbol_map.json
v1230_feature_provenance_table.csv
v1230_forbidden_token_audit.csv
v1230_implementation_readback.md
v1230_route_decision.json
```

## 4.4 Gate

若任一 official / exploration candidate 有：

```text
uses_y_for_stats = 1
uses_trainprobe_token = 1
uses_label_for_init = 1
uses_ce_vector_for_functional_direction = 1
uses_validation_for_commit = 1
uses_query_batch_for_commit = 1
uses_dataset_name_branch = 1
uses_linec_hard_target_for_direction = 1
```

则 route 必须是：

```text
R0-ProvenanceViolation
```

---

# 5. Line D：Classic no-BSpline basis-first portfolio

## 5.1 共同目标

Line D 的目标不再是“看看有没有一个过”。每个 active family 都必须推进到可审计状态：

```text
FamilyPass
FamilyNearPass
EfficiencyBlocked
ExpressionBlocked
TaskBlocked
GeometryBlocked
MemoryBlocked
RejectedForThisVersion
```

本轮目标不是所有 family 全部 FamilyPass，而是至少满足：

```text
1. Rational 推进一个硬 blocker；或
2. Chebyshev / Wavelet 中至少一个从 TaskBlocked 推到 FamilyNearPass；或
3. RBF / Fourier 中至少一个从 ExpressionBlocked 推到 TaskBlocked / FamilyNearPass。
```

## 5.2 统一 gate

### Efficiency gate

Exploratory：

$$
step\_ratio_{q90}\le 1.25,
$$

$$
memory\_ratio_{q90}\le 1.20.
$$

Official：

$$
step\_ratio_{q90}\le 1.10,
$$

$$
memory\_ratio_{q90}\le 1.00.
$$

Strong target：

$$
step\_ratio_{q90}\le 1.00,
$$

$$
memory\_ratio_{q90}\le 0.80.
$$

### Expression gate

必须通过 A4 expression battery：

```text
E1 pairwise product
E2 composition
E6 rotated pairwise
E8 random quadratic
```

Exploratory：

$$
\Delta R^2_{min}\ge -0.03.
$$

Official：

$$
\Delta R^2_{min}\ge -0.015.
$$

### Task gate

Exploratory：

$$
mean\_delta\ge -0.010,
$$

$$
worst\_delta\ge -0.030,
$$

$$
AUCtime\_ratio\le 1.10,
$$

$$
ECE\_delta\le 0.03.
$$

Official：

$$
mean\_delta\ge -0.005,
$$

$$
worst\_delta\ge -0.015,
$$

$$
AUCtime\_ratio\le 1.00,
$$

$$
ECE\_delta\le 0.02.
$$

### Geometry gate

Exploratory：

```text
LineC pass rate >= 5/9
CouplingR2 >= MLP - 0.03
NoiseSignalLeak <= MLP + 0.03
RealSignalReservoirRatio <= MLP + 0.05
CEp99 <= MLP + tolerance
```

Official：

```text
LineC pass rate >= 8/9
CouplingR2 >= MLP - 0.02
NoiseSignalLeak <= MLP + 0.02
RealSignalReservoirRatio <= MLP + 0.03
CEp99 / NLL / ECE not worse than MLP by strict tolerance
```

## 5.3 D-RAT：Rational 优先线

### 当前状态

Rational 已经不再是纯 kernel fantasy。历史上有 L3 / A4 可行候选，但 A5 task、LineC、memory 没有同时闭合。近期 D39-D43 repair 未产生 meaningful progress。因此 Rational 下一步不能泛扫 task，而要做 memory + task geometry 的 targeted repair。

### 假设

$$
\boxed{
\text{Rational 的当前 blocker 是 memory/workspace 与 task-geometry 不同位，而不是 rational basis 数学上不可训练。}
}
$$

### 候选方向

```text
D-RAT1: GroupWorkspaceRecomputeV3
  denominator / numerator forward recompute，减少 activation save。

D-RAT2: ReadoutGradChunkedV2
  chunked readout gradient，降低 memory peak。

D-RAT3: DenStateFP16Checkpoint
  denominator state fp16/bf16 audit + stability check。

D-RAT4: GroupRationalSharedDenom
  group-shared denominator，降低 denominator gradient state。

D-RAT5: RationalLineCResidualMix
  保留 best task path，加入 loss-agnostic LineC residual guard。

D-RAT6: RationalTaskTrajectoryWarmNoExtraMemV2
  只修 AUC/time，不增加 memory。
```

### 必须记录

```text
candidate_id
family = Rational
step_ratio_q90
memory_ratio_q90
backward_ratio_q90
update_ratio_q90
num_denominator_params
den_min
den_p01
den_condition
r_prime_p95
r_double_prime_p95
A4_E1_delta
A4_E6_delta
A4_E8_delta
mean_delta_vs_mlp
worst_delta_vs_mlp
AUC_step_ratio
AUC_time_ratio
ECE_delta
NLL_delta
CEp99_delta
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
LineC_pass_count
blocker_label
```

### 失败后 Codex 必须尝试

```text
If memory fail:
  尝试 readout chunking、activation recompute、denominator state fp16、workspace reuse。

If efficiency pass but expression fail:
  尝试 group count / hidden residual / low-rank sidecar，不增加 dense expansion。

If expression pass but task fail:
  尝试 denominator damping、output scale schedule、role energy cap、AUC trajectory warmup。

If task pass but LineC fail:
  尝试 LineC residual guard / coupling-preserving readout，不能使用 LineC hard target作为方向。
```

## 5.4 D-CHE：Chebyshev task-geometry repair

### 当前状态

Chebyshev 有 efficiency / A4 通过历史，但 task trajectory / AUC / LineC 不健康。下一步不能继续盲目提高 degree。

### 候选方向

```text
D-CHE1: K3-degreeEnergyCap
D-CHE2: K4-lateEnableHighDegree
D-CHE3: K4-roleWiseEnergyCap
D-CHE4: K3-K4 hybrid with low-degree dominant residual
D-CHE5: LineC-aware energy monitor, no direction use
```

### 关键指标

```text
degree_energy_k
high_degree_energy_ratio
recurrence_max_abs
basis_condition_proxy
AUC_time_ratio
LineC_pass_count
NoiseSignalLeak
RealSignalReservoirRatio
```

### 失败后 Codex 尝试

```text
If task collapse:
  降 degree，启用 late-enable，降低 high-degree energy。

If LineC fail:
  加 role energy cap、basis energy smooth schedule。

If efficiency fail:
  不扩 degree，优化 recurrence fusion。
```

## 5.5 D-WAV：Wavelet local support repair

### 当前状态

Wavelet local/hat path 已有 efficiency 可行，但 task/LineC 不稳。不能转向 heavy Morlet / MexicanHat 大路径。

### 候选方向

```text
D-WAV1: HatWavelet-local-K4-scaleStable
D-WAV2: TriangleWavelet-K4-occupancyBalanced
D-WAV3: Haar-lite local support diagnostic
D-WAV4: scale-diversity small residual, no heavy exp/sin
D-WAV5: local-tail coverage monitor
```

### 关键指标

```text
active_support_count
support_occupancy_entropy
scale_energy
scale_dead_fraction
local_tail_coverage
CEp99_delta
LineC_pass_count
```

### 失败后 Codex 尝试

```text
If expression fail:
  小幅增加 K_active 或增加 identity residual。

If task fail:
  降 scale diversity，限制 local wavelet amplitude。

If LineC fail:
  测 active support 是否集中在 hard-tail；若否，换 support placement。
```

## 5.6 D-RBF：RBF / FastKAN compact expression repair

### 当前状态

RBF / FastKAN 已有 compact L3 efficiency path，但 A4 expression 不过。不能回到 dense RBF。

### 候选方向

```text
D-RBF1: FastKAN-fixed-center-K4
D-RBF2: CompactLocalRBF-Kactive4
D-RBF3: CompactLocalRBF-Kactive8, only if efficiency remains pass
D-RBF4: train-stream quantile centers
D-RBF5: triangular / bump approximation to reduce exp cost
D-RBF6: identity + compact RBF residual
```

### 关键指标

```text
K_total
K_active
exp_count_per_sample
active_center_entropy
dead_center_fraction
out_of_grid_fraction
A4_E1/E6/E8
LineC_pass_count
```

### 失败后 Codex 尝试

```text
If A4 expression fail:
  从 Kactive4 增到 Kactive8；加入 identity residual；调整 center stats。

If efficiency fail:
  回退到 Kactive4 或 approximate bump，不准 dense materialization。

If task fail:
  检查 dead center 与 out-of-grid，做 center quantile repair。
```

## 5.7 D-FOU：Fourier low-frequency expression repair

### 当前状态

Fourier efficiency 可以很好，但低频 compact version A4 expression 不够。不能直接加高频硬拟合。

### 候选方向

```text
D-FOU1: Fourier-K2-lowfreq-plus-identity
D-FOU2: Fourier-K4-lowfreq-plus-linearResidual
D-FOU3: small learned amplitude residual, fixed frequency
D-FOU4: late-enable K4 from K2
D-FOU5: multi-scale low-K, no high-frequency explosion
```

### 关键指标

```text
spectral_entropy
high_freq_energy_ratio
phase_drift
A4_E1/E6/E8
NoiseSignalLeak
CEp99_delta
LineC_pass_count
```

### 失败后 Codex 尝试

```text
If expression fail:
  加 low-K residual / identity coupling，不直接上 high-frequency。

If noise leak high:
  降频率、加 amplitude cap、late-enable。

If efficiency fail:
  fused sincos / fixed frequency only。
```

---

# 6. Line M：Functional update on MLP

## 6.1 目标

Line M 不是为了证明 MLP 也一定更好，而是为了给 functional update 建立 cross-architecture 机制定位。

要回答：

```text
M-Q1: loss-agnostic functional update 在普通 MLP 上是否有可测 geometry/task benefit？
M-Q2: 如果 MLP 有 benefit，KAN 是否有更强 benefit？
M-Q3: 如果 MLP 无 benefit，KAN 的 benefit 是否真是 KAN-specific？
M-Q4: 如果 MLP 和 KAN 都无 benefit，当前 functional value source 是否应该重建？
```

## 6.2 MLP baseline

必须至少包含：

```text
MLP-h160-AdamW
MLP-h160-AdamW + NoOpMatchedOverhead
MLP-h160-AdamW + RandomMatchedNorm
MLP-h160-AdamW + AdamWParallelDirection audit
MLP-h160-AdamW + SNR-only audit
MLP-h160-AdamW + LossAgnosticFunctionalEvent
```

如果计算允许，增加：

```text
MLP same-param as best KAN family
MLP same-step-time as best KAN family
MLP same-wall-clock budget
```

## 6.3 Loss-agnostic MLP functional event 候选

Official direction 只能使用 T1 precommit loss-agnostic features。

```text
M-F1: activation covariance transport
  只使用 unlabeled activation covariance，做低频 whitening / condition maintenance。

M-F2: logit covariance stabilization
  使用 unlabeled logits covariance，不读 label，不读 CE。

M-F3: random-cotangent Jacobian sketch shaping
  使用随机 cotangent，不使用 CE gradient。

M-F4: hidden spectral balance
  控制 hidden effective rank / top eigen share，避免 collapse。

M-F5: geometry-safe weight anchor
  使用 weight-space proximity and activation drift，不能使用 validation loss。
```

T1B exploration 允许记录 optimizer-observable opaque features，例如 base AdamW update norm / direction statistics，但 promotion 时必须单独标注：

```text
uses_optimizer_update_as_observable = 1
promotion_scope = optimizer-observable, not pure state-only
```

## 6.4 记录指标

```text
method
architecture = MLP
functional_candidate_id
uses_label_for_direction
uses_ce_vector_for_direction
uses_validation_for_commit
uses_linec_hard_target_for_direction
source_vs_noop_acc_delta
source_vs_control_acc_delta
NLL_delta
ECE_delta
CEp99_delta
AUC_step_delta
AUC_time_delta
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
control_gap_vs_best
train_shuffle_pass_count
LineC_pass_count
functional_overhead_ratio
```

## 6.5 Gate

Exploration：

$$
source\_vs\_noop \ge 0,
$$

$$
source\_vs\_control \ge -0.003,
$$

$$
\Delta CouplingR^2 \ge 0.01,
$$

且：

```text
CEp99 / NLL / ECE 不明显坏；
LineC pass count >= 3/5；
functional overhead <= 1.05。
```

Official MLP functional success：

$$
source\_vs\_control \ge 0.005,
$$

$$
\Delta CouplingR^2 \ge 0.02,
$$

$$
\Delta NoiseSignalLeak \le 0,
$$

$$
\Delta RealSignalReservoirRatio \le 0,
$$

并且：

```text
LineC majority/all pass；
CEp99 / NLL / ECE 不坏；
train-shuffle robust；
matched controls checked；
no label / CE / validation / future / dataset branch。
```

## 6.6 解释规则

```text
MLP functional pass, KAN functional fail:
  functional 机制可能通用，但 KAN base geometry 未承载；先修 KAN base。

MLP functional fail, KAN functional pass:
  functional 可能是 KAN-specific synergy，优先做 KAN mechanism proof。

MLP functional pass, KAN functional pass:
  做 factorial interaction，判断 KAN 是否有额外增益。

MLP functional fail, KAN functional fail:
  暂停 functional 小修，重建 value source。
```

---

# 7. Line F：Functional on KAN / classic near-pass candidates

## 7.1 开启条件

Functional on KAN 只有在以下至少一个条件成立时进入 S4a re-entry：

```text
1. Line A label-free FHQ near-anchor passed；
2. Line D 某个 classic family FamilyNearPass；
3. Line M MLP functional 出现 official 或 strong exploratory positive，需要测试 KAN transfer。
```

否则 Line F 只允许 shadow diagnostic。

## 7.2 Factorial matrix

必须构造矩阵：

| Architecture | Base optimizer | Functional | 目的 |
|---|---|---|---|
| MLP | AdamW | off | MLP baseline |
| MLP | AdamW | on | functional generic effect |
| label-free FHQ / A-line | AdamW/manual AdamW | off | KAN base effect |
| label-free FHQ / A-line | AdamW/manual AdamW | on | KAN + FU effect |
| Rational / classic family best | AdamW/manual | off | basis family base effect |
| Rational / classic family best | AdamW/manual | on | basis + FU effect |

Functional effect：

$$
E_{FU}^{arch}=Metric(arch+FU)-Metric(arch).
$$

KAN-specific synergy：

$$
S_{KAN}=
E_{FU}^{KAN}-E_{FU}^{MLP}.
$$

Classic-basis synergy：

$$
S_{basis}=
E_{FU}^{basis}-E_{FU}^{MLP}.
$$

## 7.3 Controls

每个 functional row 必须有：

```text
NoOpMatchedOverhead
RandomMatchedNorm
AdamWParallelDirection
SNR-only
ShuffledPayload/Event
MatchedRoleEnergyRandom
MLPAnalogFunctional, when comparing KAN
```

## 7.4 Gate

KAN functional official success：

```text
base_near_or_pass = 1
loss_agnostic_direction = 1
source_vs_control >= 0.005 or bootstrap CI lower >= 0
LineC majority/all pass
CEp99 / NLL / ECE not worse
AUC_time not worse
functional overhead <= 1.05
train-shuffle robust
matched controls checked
```

若 functional improves MLP but not KAN，不能写 DG-KAN success，只能写：

```text
generic functional optimizer positive, KAN synergy not established。
```

---

# 8. Line A：Label-free FHQ dynamic signal-frame monitor

## 8.1 目标

Line A 保留，但不再作为唯一主线。目标是验证 dynamic signal-frame formation 是否能比 static token init 更接近 near-anchor。

## 8.2 候选

```text
A-DYN1: learnable signal frame warmup with no labels
A-DYN2: early self-predictive frame alignment
A-DYN3: optimizer-observable but label-free frame refresh
A-DYN4: overcomplete structured frame with rank guard
A-DYN5: role-energy balanced FHQ with LineC audit only
```

## 8.3 Gate

Near-anchor：

```text
mean_delta_vs_mlp >= -0.005
worst_delta_vs_mlp >= -0.015
AUC_time_ratio <= 1.05 exploratory, <= 1.00 official
LineC_pass_rate >= 5/9 exploratory, >= 8/9 official
ECE/NLL/CEp99 not worse
step_ratio <= 1.10
memory_ratio <= 1.00
```

若 no near-anchor，Line F on A-line remains shadow only。

---

# 9. Line C：Unified Manifold-Channel Diagnostics

## 9.1 目标

Line C 统一用于：

```text
1. MLP functional；
2. FHQ label-free base；
3. classic basis family；
4. KAN functional。
```

不要只在 KAN 上测。MLP 也必须记录 signal/reservoir/noise。

## 9.2 指标定义

Train-probe coupling：

$$
\Delta U_B=f_{\theta_{t+\Delta}}(B)-f_{\theta_t}(B),
$$

$$
\Delta U_Q=f_{\theta_{t+\Delta}}(Q)-f_{\theta_t}(Q),
$$

$$
A_t=\arg\min_A \|\Delta U_Q-A\Delta U_B\|_F^2+\lambda\|A\|_F^2,
$$

$$
CouplingR^2=
1-
\frac{\|\Delta U_Q-A_t\Delta U_B\|_F^2}{\|\Delta U_Q\|_F^2+\epsilon}.
$$

Signal/reservoir：

$$
\hat K_{BB}=\Phi\Phi^T,
$$

$$
\hat W_B=\sum_{\tau=t}^{t+\Delta}\hat K_{BB}(\tau).
$$

从 $\hat W_B$ 谱分解得到 $P_{sig}$ 与 $P_{res}$。

Audit metrics：

$$
RealSignalReservoirRatio=
\frac{\|P_{res}r_{real}\|^2}{\|r_{real}\|^2+\epsilon},
$$

$$
NoiseSignalLeak=
\frac{\|P_{sig}r_{noise}\|^2}{\|r_{noise}\|^2+\epsilon}.
$$

注意：$r_{real}$ 和 $r_{noise}$ 可能使用 label / CE residual，因此它们是 audit-only，不允许作为 functional direction source。

## 9.3 必须记录

```text
v1230_linec_unified.csv
v1230_train_probe_coupling.csv
v1230_signal_reservoir_audit.csv
v1230_tail_calibration_audit.csv
```

字段：

```text
architecture
basis_family
candidate_id
method
functional_on
control_id
dataset
seed
window
batch_size
sketch_dim
output_rank
CouplingR2
CouplingCorr
KernelDrift
NoiseSignalLeak
RealSignalReservoirRatio
signal_effective_rank
reservoir_fraction
top_eigen_share
CEp99
NLL
ECE
margin_p10
```

---

# 10. 实验阶段设计

## P0：Code audit and candidate registry

目标：确保所有候选合法。

必做：

```text
1. py_compile 所有 modified files。
2. registry dump。
3. forbidden token audit。
4. implementation readback。
5. MLP functional path audit。
6. classic basis path audit。
```

通过标准：

```text
required_artifact_missing_count = 0
forbidden_token_count = 0
unknown_core_code_count = 0
```

## P1：Classic basis parallel scout

目标：并行跑所有 active basis 的最小 scout。

设置：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 512
val/test = 256
epochs = 3
LineC batch = 32
sketch_dim = 8
```

每个 family 至少 3-6 个候选，Rational 6-8 个。

通过：

```text
family_best_status != no_signal
```

## P2：Family-specific hardening

对每个 family 的 top candidates：

```text
train_size = 1024
val/test = 512
epochs = 8
LineC batch = 64
sketch_dim = 24
```

输出 family status。

如果 family 当前 blocker 没推进，Codex 必须执行 family-specific fallback，不允许只写 no-go。

## P3：MLP functional diagnostic

在 MLP 上跑 M-F1..M-F5：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
windows = 3,5,10
train-shuffle seeds >= 3
controls = NoOp, Random, AdamWParallel, SNR-only
```

必须输出：

```text
v1230_mlp_functional_candidates.csv
v1230_mlp_functional_controls.csv
v1230_mlp_functional_gate.csv
```

## P4：Factorial transfer to KAN / classic near-pass

只有 P2 或 P3 有 signal 才执行。

```text
MLP functional positive -> test same functional family on best KAN/classic near-pass。
Classic FamilyNearPass -> test functional on that family。
FHQ near-anchor -> test functional on FHQ。
```

## P5：Joint confirmation

如果出现：

```text
FamilyNearPass + functional positive
或 MLP functional positive + KAN transfer positive
```

跑 5-seed short confirm。

## P6：Stop / no-go / next-hypothesis queue

如果所有路径失败，必须写清楚：

```text
1. 哪个 family 是 EfficiencyBlocked / ExpressionBlocked / TaskBlocked / GeometryBlocked；
2. MLP functional 是否有通用信号；
3. KAN-specific functional synergy 是否不存在；
4. 下一轮该修 family、functional value source，还是 base architecture。
```

---

# 11. 必须生成的可视化

```text
fig_v1230_family_status_matrix.svg
fig_v1230_basis_efficiency_pareto.svg
fig_v1230_basis_expression_radar.svg
fig_v1230_task_auc_time_by_family.svg
fig_v1230_linec_by_family.svg
fig_v1230_mlp_functional_control_gap.svg
fig_v1230_mlp_vs_kan_functional_factorial.svg
fig_v1230_task_linec_colocation_heatmap.svg
fig_v1230_tail_calibration_failure_modes.svg
fig_v1230_route_waterfall.svg
```

特别重要的是 `fig_v1230_mlp_vs_kan_functional_factorial.svg`。它必须展示：

```text
MLP + FU vs MLP
KAN + FU vs KAN
Classic basis + FU vs Classic basis
controls
```

否则无法判断 functional update 是通用方法还是 KAN-specific synergy。

---

# 12. 成功标准

## Minimum Success A：basis-first 有真实推进

满足任一：

```text
1. Rational 从 Memory/Task/Geometry blocker 推到 FamilyNearPass；
2. Chebyshev 或 Wavelet 从 TaskBlocked 推到 FamilyNearPass；
3. RBF 或 Fourier 从 ExpressionBlocked 推到 TaskBlocked 或 FamilyNearPass；
4. 至少一个 classic family official gate 的一个 hard blocker 被稳定修掉，且不是单 seed / 单 dataset。
```

## Minimum Success B：MLP functional 机制定位成立

满足：

```text
MLP functional 跑完 matched controls；
至少一个 candidate 有 reproducible positive or clean no-go；
可以明确判断 functional 是 generic / KAN-specific / 当前 value source invalid。
```

## Minimum Success C：functional transfer path 打开

满足：

```text
MLP functional positive 或 basis family near-pass；
在 KAN/classic candidate 上复验；
至少达到 S4a exploration。
```

## Official Success

满足：

```text
label-free base or classic family base passes near/official gate；
functional event is loss-agnostic；
source_vs_control >= 0.005 or bootstrap CI lower >= 0；
LineC majority/all pass；
CEp99/NLL/ECE/AUC-time not worse；
train-shuffle robust；
matched controls checked；
no provenance violation。
```

---

# 13. Route definitions

```text
S1-BasisFamilyPass
  至少一个 active no-BSpline family official pass。

S2-BasisFamilyNearPass
  至少一个 family near-pass，允许 functional re-entry。

S3-MLPFunctionalGenericPositive
  MLP functional positive，KAN transfer待测。

S4-KANFunctionalSynergyPositive
  KAN positive, MLP negative/weak，提示 KAN-specific synergy。

S5-OfficialFunctionalSuccess
  label-free base/classic base + functional official success。

R1-BasisPortfolioNoProgress
  所有 family blocker 未推进，且 fallback 执行完。

R2-MLPFunctionalNoSignal
  MLP functional matched-control no-go。

R3-FunctionalGenericButNoKANSynergy
  MLP positive，但 KAN/classic transfer 不成立。

R4-LabelFreeBaseStillMissing
  FHQ label-free near-anchor仍缺失，但 basis/MLP线另有结果。

R0-ProvenanceViolation
  label / CE / validation / LineC hard target 等被违规使用。

R0-ExploreDepthIncomplete
  gate fail 后未执行预注册 fallback。
```

---

# 14. Codex 执行规则：不允许 fail-fast

Codex 不能在以下情况下停止：

```text
1. Rational memory fail 后，没有执行 memory fallback。
2. RBF/Fourier expression fail 后，没有执行 compact capacity fallback。
3. MLP functional fail 后，没有执行 feature ablation / matched controls / no-go boundary。
4. Classic family 未执行，却写成 rejected。
5. Functional on KAN 没有 base，直接写 functional no-go。
```

Final stop 只允许：

```text
official_success_reached = 1
或 hard_compute_budget_exhausted = 1 且 all_fallback_levels_executed = 1
或 user_stop_flag = 1
```

若只写 next hypothesis 但未执行，必须 route：

```text
R0-ExploreDepthIncomplete
```

---

# 15. 本版最终判断

v12.30 不再把所有希望压在 label-free FHQ/B320-like frame recovery 上，也不再把 functional update 只放在 KAN 内部自循环。新的主策略是：

$$
\boxed{
\text{先把 classic no-BSpline basis 做到能承载任务和几何；同时在 MLP 上验证 functional update 是否有通用机制。}
}
$$

如果 classic basis 先 work，它可以成为新的 label-free PureKAN base。如果 MLP functional 先 work，它可以帮助判断 functional update 是通用 optimizer 还是 KAN-specific geometry。如果二者都不 work，则项目应回到 primitive/base design，而不是继续在当前 functional scorer 上小修。

本版真正要避免的是：

```text
1. 继续在 A-F/A-R token 上低价值排列组合；
2. functional update 只在 KAN 上失败但不知道是否 MLP 也失败；
3. classic family 只写 hypothesis 不执行；
4. 用 CE/NLL/ECE 作为方向源；
5. 用单 seed / 单数据集 / 单 sketch partial signal 写成 progress。
```

一句话：**v12.30 的重点是 Basis-first + Functional-on-MLP。先让至少一个非 B-spline basis 真正接近可用，同时用 MLP 把 functional update 的机制属性判清楚。**

