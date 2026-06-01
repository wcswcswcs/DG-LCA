# DG-KAN v13.4：Operator-Level Basis-Channel Functional Update + Substrate Split 战略计划

> 生成时间：2026-05-27 Asia/Singapore  
> 版本定位：v13.4 strategic plan  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 核心要求：不按数据集调参；不使用 teacher / distillation / loss modification / sampler / class weight / dataset-name branch；不使用 label-informed initialization；CEp99 / NLL / ECE / LineC hard target 只能作为审计与坏化约束；functional direction 可以接收 **generic loss-interface cotangent**，但不能 hardcode CE-specific 公式。

---

# 0. 项目总目标与当前进展

## 0.1 总目标

DG-KAN 的总目标不是找到一个在 MNIST/Fashion-MNIST/KMNIST 上偶然表现好的模型，也不是继续构造越来越复杂的 functional token。项目真正目标是：

$$
\boxed{
\text{构建一个 label-free strict FC-PureKAN substrate/base，}
\text{并通过 loss-interface-generic functional update}
\text{获得比同一 base 的 ordinary backprop/AdamW 更好的训练几何和模型。}
}
$$

最终 claim 必须同时满足：

```text
1. 表达力不打折；
2. forward / backward / step / memory 与 MLP 可比；
3. 训练轨迹不慢，AUC-step / AUC-time 不输；
4. 几何健康：train-probe coupling、signal/reservoir/noise、tail、calibration 不坏；
5. functional update 的收益必须击败 NoOp / RandomMatchedNorm / AdamWParallel / SNR-only / MLP analog controls；
6. functional update 不针对 CE 设计，CE / NLL / ECE / CEp99 只能作为审计。
```

正式目标是：

$$
\boxed{
\text{PureKAN substrate + functional update}
>
\text{same PureKAN substrate + ordinary backprop / controls}
}
$$

不是只证明：

$$
\text{某个 synthetic row 有 P3 improvement}
$$

也不是只证明：

$$
\text{某个 basis 能跑得快}
$$

---

## 0.2 当前最新进展状态

v13.3 的结果说明：

```text
route = R4-DiagonalAndLowRankMetricNoGo
minimum_success = S1-EfficientControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
basis_natural_p3_pass_count = 0
lowrank_metric_pass_count = 0
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
nonrat_rescue_s1_count = 0
mlp_analog_pass_count = 0
full_basis_param_update_rows = 168
provenance_violation_count = 0
```

因此，当前项目只保留了：

```text
S1: Rational efficient controllable substrate 存在。
```

还没有：

```text
S2: healthy base；
S3: synthetic task-family robust functional mechanism；
S4: MLP or KAN functional generic positive；
S5: official real-data functional success。
```

v13.3 的重要进展不是能力成功，而是证伪了一个关键假设：

$$
\boxed{
\text{“diagonal/local metric 不够，所以加 low-rank/block/family-balanced metric 就会 work”}
\text{这个解释不成立。}
}
$$

BM8-BM15 的 low-rank / block / family-balanced / control-residualized natural updates 都没有打开 P3。最接近的 rows 仍然卡在 source、CouplingR2、NoiseSignalLeak、ReservoirRatio 等核心门，而 metric condition 本身非常健康，说明失败不是 Woodbury 或数值病态导致。

---

# 1. 各条线当前进展百分比与上次比较

下表中的百分比是研究完成度估计，不是 artifact 官方字段。它综合了 gate 通过情况、机制清晰度、代码可信度和距离 official success 的距离。

| 线 | 上次 v13.2 后估计 | 本次 v13.3 后估计 | 变化 | 判断 |
|---|---:|---:|---:|---|
| Line R：代码 / provenance / finalizer 审计 | 99% | 99% | 0 | 工程闭包很强；不是当前 blocker。 |
| Historical FHQ / B320-current 工程能力 | 85% frozen | 85% frozen | 0 | 历史强，但 label-informed init 禁用，不能 official。 |
| Label-free FHQ / A-DYN monitor | 22% | 20% | -2 | 多轮无 near-anchor，仅保留低预算 monitor。 |
| Line C 几何审计 | 88% | 88% | 0 | 审计可靠，但不能作为 direction source。 |
| Rational substrate | 85% | 85% | 0 | 仍是唯一稳定 S1 substrate。 |
| Non-RAT substrate | 35%-42% | 30%-38% | -5 | D-CHE12/D-CHE20 focused rescue 都没打开 S1。 |
| Full basis-param writeback implementation | 55% | 60% | +5 | 真实参数写回继续成立，168 rows 不是 readout proxy。 |
| Basis-natural functional mechanism | 18% | 10% | -8 | v13.2 有 X7-only 局部 pass；v13.3 BM8-BM15 全部 P3=0。 |
| Synthetic mechanism proof | 25% | 5%-10% | -15 | synthetic task-family pass 0/7，不能再说接近。 |
| MLP analog control | 20% | 18% | -2 | MLP analog closure 完成但 pass=0。 |
| Functional official / real short-run | 0%-5% | 0%-5% | 0 | real short-run gate 未打开。 |
| 整体 next-gen MLP claim | 43%-48% | 38%-43% | -5 | v13.3 证明当前 metric family 方向更弱，路线必须重置。 |

这次要主动下调完成度。原因不是工程倒退，而是 v13.3 让我们看清楚：真实参数写回虽然已实现，但当前 metric / update 结构没有抓住任务族稳健的 functional mechanism。

---

# 2. 对 v13.3 的独立分析

## 2.1 这次有没有进展？

有，但不是能力进展。v13.3 的价值是：

```text
1. 证明 BM8/BM9/BM10/BM14 这类低秩/块度量没有打开 P3；
2. 证明 BM11/BM12/BM13/BM15 fallback 也没有打开 P3；
3. 证明 Non-RAT focused rescue 仍然不能形成 S1；
4. 证明 MLP analog 仍然没有 generic positive；
5. 证明 full_basis_param_update_rows = 168，仍是真实 basis-param update，不是 readout proxy。
```

这轮不是以前那种“runner 没跑到关键阶段”。它跑到了关键阶段，但给了一个更硬的 no-go。

---

## 2.2 为什么这不是小失败，而是路线失败

v13.2 的 BN5 在 X7 上有局部 P3 pass，容易让我们以为：

```text
只要把 diagonal/local metric 换成 low-rank 或 block metric，
coverage 就会扩大。
```

v13.3 直接否定了这个解释。BM8/BM9/BM10/BM14 没有保住 v13.2 的 X7 positive，BM11/BM12/BM13/BM15 也没有打开新的 family。最接近 rows 的 pattern 仍然是：

```text
source 不足；
CouplingR2 坏化；
NoiseSignalLeak 坏化；
ReservoirRatio 坏化；
有时 CEp99/NLL/ECE 也坏。
```

metric condition 的范围很健康：

```text
min ≈ 1.000007
max ≈ 1.006437
mean ≈ 1.001049
```

所以不能再说：

```text
数值条件病态；
Woodbury inverse 爆了；
rank 太低；
block group 太粗。
```

这些都不是主因。

---

## 2.3 当前真正的问题

我现在认为真正问题是：

$$
\boxed{
\text{我们仍然在用 parameter-space metric 近似一个 operator-level 问题。}
}
$$

当前 update 形式大致是：

$$
\Delta\theta
=
-\eta M_{\theta}^{-1} g_{\theta}.
$$

即使 $M_\theta$ 加了 diagonal、SNR、low-rank、block、Kronecker，它仍然直接在参数空间里 precondition 梯度。问题是：KAN 的 functional advantage 不应该来自参数坐标本身，而应该来自 **basis channel function** 的可控变化。

也就是说，我们真正想控制的是：

$$
Z = \Phi_\theta(X),
$$

以及：

$$
\Delta Z = \Phi_{\theta+\Delta\theta}(X)-\Phi_\theta(X).
$$

而不是直接希望某个 $M_\theta^{-1}g_\theta$ 自动产生好的 $\Delta Z$。

v13.3 的 failure pattern 说明当前 update 经常做到：

```text
source_vs_best 略正；
局部 reservoir 有时改善；
CEp99 有时改善；
```

但同时：

```text
NoiseSignalLeak 变坏；
CouplingR2 不稳；
ReservoirRatio 不稳；
跨 X-family 不稳。
```

这说明它不是在控制 basis channel 的几何，而是在参数空间里制造了某种局部扰动。

---

# 3. v13.4 的核心重置：从 parameter metric 转向 operator-level basis-channel solve

## 3.1 新假设

v13.4 不再继续问：

```text
哪种参数度量 M_theta 更好？
```

而是问：

```text
能否先在 basis-channel space 求一个想要的 Delta Z，
再把 Delta Z 投影回真实 basis 参数？
```

新 functional update 形式变成两阶段：

### Stage 1：在 basis-channel space 中求目标位移

令：

$$
Z = \Phi_\theta(X)
$$

为某个 basis family 的 channel activation / basis output。输出 logits 记为：

$$
Y = H_\theta(Z).
$$

从 generic loss interface 得到 output cotangent：

$$
\delta_Y = \frac{\partial \ell}{\partial Y}.
$$

我们先求一个 channel displacement：

$$
\Delta Z^\star
=
\arg\min_{\Delta Z}
\left[
\langle \delta_Y, J_{Z\to Y}\Delta Z\rangle
+
\lambda_z \|\Delta Z\|_{K_Z}^2
+
\lambda_s R_{snr}(\Delta Z)
+
\lambda_t R_{tail-proxy}(\Delta Z)
\right].
$$

这里：

```text
1. δ_Y 来自 generic loss interface，不 hardcode CE；
2. R_tail-proxy 不使用 CEp99 / NLL / ECE，只使用无标签 logit / basis telemetry；
3. LineC / CEp99 / NLL / ECE 只在之后审计。
```

### Stage 2：把 $\Delta Z^\star$ 投影回真实参数

然后解：

$$
\Delta\theta^\star
=
\arg\min_{\Delta\theta}
\|J_{\theta\to Z}\Delta\theta-\Delta Z^\star\|_2^2
+
\rho\|\Delta\theta\|_{M_\theta}^2.
$$

接受条件首先不是 task gain，而是 actuation fidelity：

$$
\operatorname{ActuationError}
=
\frac{\|J_{\theta\to Z}\Delta\theta^\star-\Delta Z^\star\|}
{\|\Delta Z^\star\|+\epsilon}
\le \tau_{act}.
$$

如果 actuation error 很高，说明该 basis substrate 不能执行我们想要的 channel displacement。此时不应该继续调 update scale，而应该回到 substrate 或 basis parametrization。

---

## 3.2 为什么这是本质变化

旧路线：

```text
g_theta -> M_theta^{-1} g_theta -> hope LineC/task improves
```

新路线：

```text
loss cotangent -> desired basis-channel movement DeltaZ
-> parameter projection J_theta->Z
-> verify actuation fidelity
-> only then future probe / controls
```

它多了一个关键可解释中间层：

```text
basis-channel displacement target DeltaZ
```

这能回答之前一直答不出来的问题：

```text
functional update 到底想让 basis 发生什么变化？
参数更新有没有真的实现这个变化？
失败是 value source 错，还是 basis 执行不了？
```

---

# 4. v13.4 实验总结构

v13.4 分为六条线：

```text
Line R：Implementation / provenance / code review。
Line S：Substrate split：Efficient substrate vs channel-controllable substrate。
Line O：Operator-level basis-channel solve。
Line P：Parameter projection / actuation fidelity。
Line X：Synthetic task-family proof。
Line D：Non-RAT S1 substrate design。
Line M：MLP analog coordinate-channel control。
Line Z：finalizer / route / no-go boundary。
```

---

# 5. Line R：代码与实现审计

## 5.1 目标

确认 v13.4 真正实现 operator-level basis-channel functional update，而不是又退化成：

```text
parameter metric preconditioner；
readout-feature proxy；
frozen feature table transport；
LineC hard target direction；
CEp99/NLL/ECE direction。
```

## 5.2 必须审查的实现面

Codex 必须在复盘里写清楚以下文件、函数、行号、shape：

```text
1. 每个 basis family 的 Z = Phi_theta(X) 如何定义；
2. J_theta->Z 如何计算或 sketch；
3. J_Z->Y 如何计算或 sketch；
4. DeltaZ target 如何生成；
5. DeltaTheta projection 如何求解；
6. true parameter writeback 路径；
7. optimizer state 是否 transport；
8. loss-interface cotangent 是否 generic；
9. CE / NLL / ECE / CEp99 是否只作为 audit；
10. 是否有 label / validation / future / dataset-name leakage。
```

## 5.3 必须落盘

```text
v134_code_review_manifest.csv
v134_basis_channel_manifest.csv
v134_jacobian_operator_manifest.csv
v134_loss_interface_audit.csv
v134_forbidden_information_audit.csv
v134_parameter_writeback_trace.csv
v134_optimizer_state_transport_trace.csv
```

## 5.4 硬门

如果出现以下任一情况，route 直接为：

```text
R0-OperatorFunctionalNotImplemented
```

```text
readout_feature_proxy_only = 1
feature_table_proxy_only = 1
full_basis_param_update_rows = 0
basis_channel_target_rows = 0
actuation_error_missing = 1
loss_interface_generic = 0
forbidden_information_violation = 1
```

---

# 6. Line S：Substrate split

## 6.1 目标

把 substrate 分成两层：

```text
S1-EfficientSubstrate:
  跑得动，有最小表达和最小训练信号。

S1C-ChannelControllableSubstrate:
  不一定 healthy，但能以低 actuation error 执行 DeltaZ target。

S2-HealthyBase:
  basis-only 已能通过 task/AUC/LineC/tail。
```

v13.3 已经证明，仅有 S1 不够。v13.4 的新目标是找到 S1C。

## 6.2 指标

```text
workspace_raw_ratio
workspace_incremental_ratio
step_ratio
mean_delta_vs_MLP
worst_delta_vs_MLP
AUCtime_ratio
LineC_pass_rate
basis_channel_rank
basis_channel_effective_rank
J_theta_to_Z_rank
J_theta_to_Z_condition
actuation_error_random_target
actuation_error_loss_target
actuation_error_channel_target
```

## 6.3 Gate S1C

一个 substrate 进入 functional mechanism proof 需要：

$$
workspace\_incremental\_ratio \le 2.0,
$$

$$
step\_ratio \le 1.75,
$$

$$
basis\_channel\_rank\_ratio \ge 0.25,
$$

$$
actuation\_error\_{random} \le 0.35,
$$

$$
actuation\_error\_{loss} \le 0.45.
$$

如果 S1 有但 S1C 没有，则 functional 不进入 P3；应回到 basis parametrization。

---

# 7. Line O：Operator-level basis-channel solve

## 7.1 目标

构造 $\Delta Z^\star$，并证明它不是 CE-specific，不是 LineC-target-specific，不是 post-hoc 目标。

## 7.2 候选

```text
O1-LossCotangentChannelNewtonDiag
O2-LossCotangentChannelSNR
O3-PopRiskOffdiagChannel
O4-TrainProbeCouplingPreservingChannel
O5-NoiseQuarantineUnlabeledProxy
O6-ReservoirReleaseUnlabeledProxy
O7-BalancedChannelSolve
```

## 7.3 记录指标

```text
delta_z_norm
delta_z_rank
delta_z_effective_rank
delta_z_alignment_with_loss_cotangent
delta_z_alignment_with_adamw_channel
delta_z_snr_positive_fraction
delta_z_offdiag_score
delta_z_predicted_logit_drift
delta_z_predicted_tail_proxy
```

## 7.4 Gate O

```text
delta_z_norm > 0
delta_z_effective_rank >= 2
predicted_logit_drift <= drift_budget
loss_interface_generic = 1
uses_CE_formula_specific = 0
uses_LineC_target = 0
```

---

# 8. Line P：Parameter projection / actuation fidelity

## 8.1 目标

把 $\Delta Z^\star$ 投影成真实参数更新，并检查是否真的执行了该 $\Delta Z$。

## 8.2 候选求解器

```text
P1-DiagonalProjection
P2-BlockProjection
P3-LowRankWoodburyProjection
P4-ConjugateGradientJtJProjection
P5-RoleWiseProjection
P6-TrustRegionProjection
```

## 8.3 必须记录

```text
projection_solver
projection_residual
actuation_error
actual_delta_z_norm
actual_vs_target_delta_z_cosine
actual_delta_y_norm
actual_logit_drift
actual_basis_rank_change
actual_basis_condition_change
parameter_update_norm
optimizer_state_transport_applied
```

## 8.4 Gate P

$$
actuation\_error \le 0.35,
$$

$$
cos(\Delta Z_{actual},\Delta Z^\star) \ge 0.50,
$$

$$
actual\_logit\_drift \le drift\_budget.
$$

如果 P 失败，不能进入 synthetic proof；说明不是 metric 问题，而是 basis parameterization 不可控。

---

# 9. Line X：Synthetic task-family proof

## 9.1 目标

只在 O/P 通过后执行 synthetic X1-X7。v13.3 最大问题是直接从 parameter metric 跳到 X family，缺少 actuation fidelity 中间层。v13.4 必须先过 O/P。

## 9.2 Tasks

```text
X1 pairwise product
X2 rotated quadratic
X3 random quadratic
X4 local bump / tail cluster
X5 low/high frequency mixture
X6 two-manifold split
X7 noisy signal / reservoir mixture
```

## 9.3 Gate X

Official synthetic gate：

```text
>=5/7 task families pass
and no single task family accounts for all pass rows
and leave-one-family-out pass >= 4/6
```

Task-family pass row 需要：

$$
source\_vs\_best \ge 0.005,
$$

$$
\Delta CouplingR^2 \ge 0.02,
$$

$$
\Delta NoiseSignalLeak \le 0,
$$

$$
\Delta ReservoirRatio \le 0,
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

Exploration gate 可以允许：

```text
>=3/7 pass
```

但不能 promotion；只能进入 mechanism debugging。

---

# 10. Line D：Non-RAT substrate design

## 10.1 目标

v13.3 的 functional 仍然绑定 Rational substrate，因为 Non-RAT rescue S1=0。v13.4 必须继续推进 Non-RAT，但方式不是继续 D-CHE20 小修。

## 10.2 方向

### Chebyshev

```text
CHE-S1 recurrence-in-register kernel；
CHE-S2 degree-energy normalized basis；
CHE-S3 delayed high-degree activation；
CHE-S4 degree-wise readout recompute。
```

### Fourier

```text
FOU-S1 fused sincos low-frequency kernel；
FOU-S2 frequency-band sparse readout；
FOU-S3 phase-stable low-rank residual；
FOU-S4 high-frequency quarantine。
```

### RBF/FastKAN

```text
RBF-S1 local active-center no-materialize kernel；
RBF-S2 width-shared compact bump；
RBF-S3 OOG boundary recenter；
RBF-S4 center occupancy repair。
```

### Wavelet

```text
WAV-S1 hat-wavelet support-local kernel；
WAV-S2 scale-normalized support bank；
WAV-S3 overlap-controlled local tail coverage；
WAV-S4 support occupancy guard。
```

## 10.3 Gate

Non-RAT 至少要达到 S1C，不只是 S1：

```text
workspace_incremental_ratio <= 2.0
task not catastrophic
LineC_pass_rate >= 0.20
actuation_error_random <= 0.45
```

否则不能进入 basis-functional proof。

---

# 11. Line M：MLP analog coordinate-channel control

## 11.1 目标

确认新 operator-channel solve 是否是 KAN-specific，还是普通 reparameterization trick。

## 11.2 MLP analog

```text
M1 hidden activation channel solve
M2 hidden whitening + inverse readout compensation
M3 layerwise balanced coordinate transport
M4 LoRA-like hidden subspace channel solve
```

## 11.3 判断

如果 MLP analog 也通过：

```text
functional mechanism may be generic, not KAN-specific.
```

如果 MLP 不通过而 KAN 通过：

```text
basis-coordinate advantage exists.
```

如果都不通过：

```text
operator-level functional solve no-go under current formulation.
```

---

# 12. Stop / Go 规则

## 12.1 不允许继续的方向

以下情况不得继续扩网格：

```text
1. BM8-BM15 rank/block/metric 小修；
2. BN4-BN7 diagonal/local metric 小修；
3. readout-feature proxy transport；
4. frozen feature-table transport；
5. LineC hard target direction；
6. CEp99/NLL/ECE direction；
7. Non-RAT workspace fail 后直接 functional P3；
8. synthetic 5/7 fail 后 real short-run。
```

## 12.2 允许继续的方向

仅允许继续以下方向：

```text
1. operator-level basis-channel target；
2. parameter projection actuation fidelity；
3. true full basis parameter writeback；
4. substrate S1C controllability；
5. Non-RAT S1C design；
6. MLP analog channel control。
```

## 12.3 Route definitions

```text
S1: Efficient substrate.
S1C: Channel-controllable substrate.
S2: Healthy base.
S3: Synthetic 5/7 functional proof.
S4: Real-data short-run open.
S5: Official functional success.

R0: Implementation/provenance fail.
R1: No S1C substrate.
R2: O/P actuation fail.
R3: Synthetic proof fail.
R4: Non-RAT substrate fail but Rational mechanism open.
R5: MLP analog explains mechanism.
R6: Mechanism no-go under current operator formulation.
```

---

# 13. 必须生成的可视化

```text
fig_progress_by_line.svg
fig_basis_channel_rank_by_family.svg
fig_actuation_error_by_solver.svg
fig_delta_z_target_vs_actual.svg
fig_operator_solve_source_vs_control.svg
fig_synthetic_family_pass_heatmap.svg
fig_nonrat_s1c_substrate_status.svg
fig_mlp_analog_vs_kan_operator_solve.svg
fig_failure_taxonomy.svg
```

---

# 14. 最终判断

v13.3 已经足够说明：

```text
parameter-space diagonal / low-rank / block metric 不是出路；
继续扩 BM8-BM15 是低价值搜索；
Non-RAT 仍没进入可 functional 的 substrate；
MLP analog 没有 generic positive；
Rational 仍是唯一 substrate，但 functional 不能只绑定 Rational 的 parameter metric。
```

v13.4 必须换问题层级：

$$
\boxed{
\text{从 parameter metric update}
\quad
\rightarrow
\quad
\text{operator-level basis-channel solve + parameter projection。}
}
$$

如果 v13.4 仍然不能产生 S1C 或 synthetic 5/7，那么我们应正式记录：

```text
current PureKAN basis-channel functional update formulation no-go
```

然后回到 substrate/base architecture，而不是继续 functional metric 搜索。
