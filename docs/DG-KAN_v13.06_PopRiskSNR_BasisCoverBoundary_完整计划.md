# DG-KAN v13.6：Population-Risk SNR Functional Update + Basis-Cover Boundary Stabilization 完整计划

> 版本：v13.6 strategic execution plan  
> 基于：v13.5 `DecisiveOracleTarget / Legal Observability / Substrate Reset` 真实结果；重新阅读 `A Theory of Generalization in Deep Learning` 与 `Deep Manifold Part 2: Neural Network Mathematics` 后的机制重构  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 硬约束：strict FC-PureKAN；no active B-spline；no label-informed initialization；no teacher / distillation / sampler / class weight / dataset-name branch；不使用 validation/test/future/query batch 生成 functional direction；CEp99 / NLL / ECE / LineC hard target 只能作为审计与坏化约束，不能作为方向源。

---

# 0. 项目总目标与当前进展

## 0.1 总目标

DG-KAN 的总目标不是在 MNIST-family 上打榜，也不是找一个能让某个几何分数上升的小技巧。总目标是构建：

$$
\boxed{
\text{label-free strict FC-PureKAN efficient substrate/base}
+
\text{loss-interface-generic functional update}
>
\text{same base + ordinary backprop / AdamW controls}
}
$$

这个目标有四个不可降级的要求：

```text
1. 表达力不打折：至少不弱于同规模 MLP，最好更强。
2. 系统成本不打折：forward / backward / step / memory 与 MLP 可比。
3. functional update 不是 CE 专用 trick：不能针对 CEp99 / NLL / ECE / LineC hard target 设计方向。
4. 几何收益必须真实：不能只是 CouplingR2 或 output movement，必须改善 signal / reservoir / noise 结构，并击败 matched controls。
```

最终 claim 只能来自：

```text
PureKAN substrate/base + functional
vs
same PureKAN substrate/base + AdamW / ordinary backprop / matched controls
```

而不是：

```text
label-informed B320-current；
oracle future target；
LineC hard-target direction；
readout-feature proxy；
frozen feature-table transport。
```

## 0.2 当前最重要进展

v13.5 结果说明：Rational 仍是当前唯一稳定到达 `S1C-ChannelControllableSubstrate` 的 family。它有真实 basis-channel actuation，且 `full_basis_param_update_rows=56`，不是 readout-feature proxy，也不是 feature-table proxy。

但是 v13.5 同时给出了一个很硬的负结论：

```text
route = R4-OracleTargetNoUpperBound
minimum_success = S1C-ChannelControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
substrate_s1_count = 11
substrate_s1c_count = 1
nonrat_s1c_count = 0
oracle_rows = 56
oracle_pass_rows = 0
oracle_task_family_pass_count = 0
oracle_upper_bound_pass = 0
precommit_visibility_pass = 0
sequential_oracle_pass_count = 0
mlp_oracle_pass_count = 0
full_basis_param_update_rows = 56
```

这说明：

$$
\boxed{
\text{当前问题已经不是“能不能执行 basis-channel movement”，}
\text{而是“当前 functional target 是否有 population value”。}
}
$$

v13.5 的 oracle 甚至允许 future / label / LineC diagnostic 目标作为 upper-bound，但仍然没有打过 synthetic success。这意味着：继续 O7/O8/O9、BM/BN、scale、trust-region、response dictionary、readout-feature proxy 都不是下一步。

## 0.3 重新阅读文献后的关键修正

`A Theory of Generalization in Deep Learning` 给出一个更自然的 functional value source：不要手造 oracle $\Delta Z$，而要用训练时可见的 per-example gradient mean / covariance 构造 population-risk gate。核心对象是：

$$
\bar g_B = \frac{1}{b}\sum_{a\in B} g_a,
$$

$$
\Sigma_B = \frac{1}{b}\sum_{a\in B}(g_a-\bar g_B)(g_a-\bar g_B)^T,
$$

$$
A_B = \bar g_B\bar g_B^T - \frac{1}{b-1}\Sigma_B.
$$

对于 diagonal preconditioner，最直接的参数级 gate 是：

$$
q_k = 1\left\{\mu_k^2 > \frac{\sigma_k^2}{b-1}\right\}.
$$

这说明我们过去把 `loss-agnostic` 执行成近似 `label-blind` 太强了。正确约束应该是：

```text
禁止：CE-specific formula、CEp99/NLL/ECE/LineC hard target direction、validation/test/future/query/dataset branch。
允许：通过统一 loss interface 读取当前训练 loss 的 output cotangent，并计算 per-example gradient statistics。
```

也就是说：

$$
\boxed{
\text{loss-agnostic} \neq \text{label-blind};
\quad
\text{应该是 loss-interface-generic。}
}
$$

`Deep Manifold Part 2` 的启发是：网络训练是 boundary-conditioned fixed-point iteration；node covers / coordinates 随训练移动，plasticity 先升后降。因此 KAN basis 不应只看作静态函数库，而应看作可训练 cover。functional update 的职责不是一次性几何扰动，而是：

$$
\boxed{
\text{population-risk signal selection}
+
\text{basis-cover boundary stabilization}
}
$$

## 0.4 当前各线完成度：与上一轮 v13.5 前估计对比

| 线 | v13.4 后估计 | v13.5 后估计 | 变化 | 当前判断 |
|---|---:|---:|---:|---|
| Line R：代码 / provenance / finalizer 审计 | 99% | **99%** | 0 | 工程闭包强，required artifacts 和 forbidden audit 干净。 |
| Historical FHQ / B320-current 工程能力 | 85% frozen | **85% frozen** | 0 | 历史强，但 label-informed init 已禁用，不能 official。 |
| Label-free FHQ / A-DYN monitor | 18% | **15%** | -3 | 已不是主线，继续低预算 monitor 即可。 |
| Line C 几何审计 | 88% | **88%** | 0 | 审计可用，但不能直接作为 direction source。 |
| Rational S1 efficient substrate | 85% | **85%** | 0 | 仍是唯一稳定 substrate family。 |
| Rational S1C channel controllability | 55% | **60%** | +5 | v13.5 再次确认 S1C；但只是可执行，不代表有价值。 |
| Non-RAT S1C | 25%-35% | **20%-30%** | -5 | FOU/CHE vertical slice actuation 有局部信号，但 workspace/LineC/S1C 不过。 |
| Full basis-param writeback | 75% | **80%** | +5 | 56 rows 写回真实 basis 参数，proxy flags 为 0。 |
| Operator-level basis-channel solve | 45% | **50%** | +5 | O/P 可执行性更扎实；但 target value 失败。 |
| Oracle value target | 新增 | **0%-5%** | - | 56 oracle rows 全失败，task-family pass 0/7。 |
| Legal precommit observability | 5%-10% | **0%-5%** | -5 | oracle_positive_rows=0，因此 legal visibility 不能先成立。 |
| Population-risk SNR functional | 新增 | **0%** | - | 尚未实现，是下一轮主线。 |
| Basis-cover boundary stabilization | 新增 | **0%** | - | 尚未实现，是下一轮主线。 |
| MLP analog functional | 15% | **10%-15%** | -3 | MLP oracle pass=0；generic MLP route仍无正结果。 |
| Basis-specific functional official | 0%-5% | **0%-5%** | 0 | S5 仍为 0。 |
| 整体 next-gen MLP claim | 39%-44% | **36%-42%** | -3 | v13.5 证伪了 oracle target upper-bound，路线需要重构。 |

这个下调不是工程倒退，而是科学判断更严格：我们已经知道当前 operator target 即使用 oracle 也不够，必须换 functional value source。

---

# 1. v13.5 独立分析

## 1.1 v13.5 的正进展

v13.5 不是空转。它完成了四件重要事情：

```text
1. 证明 Rational S1C 仍成立：current substrate 可以执行 reachable basis-channel movement。
2. 证明真实 basis-param writeback 已经不是 blocker：full_basis_param_update_rows=56。
3. 证明 legal/provenance 干净：provenance_violation_count=0，forbidden_information_violation_count=0。
4. 用 oracle upper-bound 关闭了当前 O-target route：oracle_rows=56，oracle_pass_rows=0。
```

以前我们不知道：失败是因为 actuator 不行、target 不行、legal visibility 不行，还是 substrate 不行。v13.5 把它拆清楚了：

```text
actuator / projection：Rational S1C 可执行；
target value：oracle upper-bound 都不行；
legal visibility：因为 oracle positive 为 0，不可能先成立；
Non-RAT：还不能作为 real short-run substrate；
MLP analog：也没有解释同一机制。
```

## 1.2 v13.5 的坏消息

坏消息比好消息更关键：

$$
\boxed{
\text{如果连 oracle target 都不能带来 future advantage，}
\text{那继续寻找 legal precommit proxy 没意义。}
}
$$

这与过去许多轮问题不同。过去是：

```text
有 oracle，legal feature 看不到；
有 P3，P4 不转化；
有 actuation，target 不对。
```

现在是：

```text
连 oracle upper-bound 都没有。
```

所以下一步不能继续：

```text
O7/O8/O9 target 小修；
BN/BM metric 回退；
response dictionary vN；
real short-run；
LineC / CEp99 / NLL / ECE direction；
readout-feature proxy；
frozen feature-table transport。
```

这些已经被 v13.5 stop rule 明确禁止，也确实没有科学价值。

## 1.3 结合文献后的重新解释

v13.5 的失败并不说明 functional update 没希望。它说明我们选错了 functional value source。

之前的思路是：

$$
\delta_Y \to \Delta Z^\star \to \Delta\theta^\star \to \text{future probe}
$$

但 `A Theory of Generalization` 的启发是：真实训练中的可泛化方向不是任意 $\Delta Z$，而是 per-example gradient 的 drift-vs-diffusion 结构。即使某个 $\Delta Z$ 能改善 LineC audit，它也可能只是诊断空间里的好动作，不是 population-risk safe 的训练动作。

因此 functional update 应该变成：

$$
\boxed{
\Delta\theta_{func}
=
-\eta\,P_{cover}\,P_{SNR}\,M_{basis}^{-1}\,\bar g
}
$$

其中：

```text
P_SNR:
  由 per-example gradient mean / variance 决定哪些参数或 basis channels 有 population signal。

M_basis:
  每类 basis 的 natural metric / telemetry metric。

P_cover:
  Deep Manifold 启发下的 basis-cover boundary stabilization。
```

这样做的关键区别是：

```text
旧路线：先手造一个几何目标，再问它有没有未来收益。
新路线：先从训练边界条件中筛选 signal，再用 basis cover 约束它怎样进入坐标系统。
```

---

# 2. 新核心假设

## H1：当前失败不是 actuator failure，而是 value-source failure

v13.4 / v13.5 已经证明 Rational S1C 可以执行 channel movement。失败来自 oracle target upper-bound 为 0。因此下一步要验证：population-risk SNR 是否比 oracle $\Delta Z$ 更自然。

## H2：loss-interface-generic SNR 是合法 functional source

Functional update 可以使用当前训练 loss 的 output cotangent $\delta_i = \partial \ell_i / \partial y_i$，但不能 hardcode CE，也不能使用 CEp99/NLL/ECE/LineC hard target。若 loss 换成 Brier、MSE、DPO-style preference loss，接口应该一致。

## H3：KAN 的优势应该来自 basis-cover telemetry，而不是 generic SNR 本身

如果 `MLP + SNR` 也强，那么这是 generic optimizer improvement，不是 KAN-specific。KAN 的主张必须是：

$$
\text{KAN + basis-SNR + cover-boundary}
>
\text{MLP + parameter-SNR}
$$

## H4：每个 basis 不必自己成为 healthy base，但必须成为 controllable substrate

基函数的职责是：

```text
1. 高效；
2. 表达不差；
3. 提供有分辨率的 channel telemetry；
4. functional safety projection 不会把大部分 update 拒掉；
5. 能在训练中形成可控 cover。
```

它不必单独解决全部 AUC / CEp99 / LineC，但必须能承载 functional update。

## H5：如果 SNR-population-risk baseline 都失败，应停止 functional 主线并转 substrate/base architecture

这是本轮必须接受的 decisive rule。如果最朴素的 population-risk SNR 都没有带来任何 signal，那么继续复杂 functional 机制是不理性的。

---

# 3. 实验总结构

v13.6 不再延续 v13.5 oracle target。新计划分为八条线：

```text
Line R：Code / provenance / implementation readback。
Line P：Population-risk SNR baseline，先跑 MLP 和 Rational。
Line B：Basis-channel SNR + basis metric。
Line G：Basis-cover boundary stabilization。
Line S：Substrate/base architecture reset，重点 Non-RAT controllable substrate。
Line X：Synthetic family proof，验证 X1-X7。
Line M：MLP analog 对照。
Line Z：Finalizer / no-go boundary / next hypothesis queue。
```

执行顺序：

```text
P0 implementation audit
P1 per-example gradient statistics correctness
P2 parameter-level SNR baseline
P3 basis-channel SNR baseline
P4 basis-cover boundary stabilization
P5 synthetic X1-X7 proof
P6 real-data short-run, gated
P7 substrate reset for Non-RAT, parallel
P8 final route
```

---

# 4. Line R：代码 / provenance / implementation readback

## 4.1 目标

确认 v13.6 真的实现了 population-risk SNR functional update，而不是又变成 feature-table proxy、oracle target 或 CE-tail direction。

## 4.2 必须审查的实现路径

Codex 必须在结果复盘中写明以下文件、函数、line range、tensor shape、数学对象：

```text
R0 runner entrypoint
R1 loss-interface cotangent extraction
R2 per-example gradient collection
R3 gradient mean / covariance / variance estimator
R4 parameter-level SNR mask
R5 basis-channel aggregation
R6 basis-specific telemetry and cover guard
R7 update writeback path
R8 optimizer state interaction
R9 controls implementation
R10 timing / memory accounting
R11 artifact finalizer
```

## 4.3 必须落盘 artifact

```text
v136_code_review_manifest.csv
v136_loss_interface_audit.csv
v136_per_example_gradient_manifest.csv
v136_snr_estimator_audit.csv
v136_update_writeback_trace.csv
v136_forbidden_information_audit.csv
v136_timing_memory_audit.csv
```

## 4.4 硬门

如果出现任一情况，route 直接为 `R0-ImplementationInvalid`：

```text
uses_validation_or_test_for_direction = 1
uses_future_outcome_for_direction = 1
uses_ce_tail_metric_for_direction = 1
uses_linec_hard_target_for_direction = 1
uses_label_informed_initialization = 1
readout_feature_proxy_only = 1
feature_table_proxy_only = 1
per_example_gradient_missing = 1
```

---

# 5. Line P：Population-risk SNR baseline

## 5.1 目标

先验证最简单、最可解释的 population-risk update 是否有用。不要一开始就做 basis-specific complex transport。

## 5.2 方法

对每个模型 / task window，收集 per-example gradient：

$$
g_i = J_\theta(x_i)^T\delta_i,
$$

其中：

$$
\delta_i = \frac{\partial \ell_i}{\partial y_i}.
$$

计算 batch mean 与 variance：

$$
\mu_k = \frac{1}{b}\sum_i g_{i,k},
$$

$$
\sigma_k^2 = \frac{1}{b-1}\sum_i (g_{i,k}-\mu_k)^2.
$$

SNR gate：

$$
q_k = 1\left\{\mu_k^2 > \tau\frac{\sigma_k^2}{b-1}+\epsilon\right\}.
$$

Functional update：

$$
\Delta\theta_{SNR}
=
-\eta\,q\odot\operatorname{AdamDirection}(\theta).
$$

也测试 soft gate：

$$
q_k = \operatorname{sigmoid}\left(
\alpha\left(
\log(\mu_k^2+\epsilon)-\log\left(\frac{\sigma_k^2}{b-1}+\epsilon\right)-\log \tau
\right)
\right).
$$

## 5.3 必跑对象

```text
MLP-h160
Rational D-RAT S1C substrate
Rational D-RAT substrate near variants
Chebyshev best workspace candidate, if S1-ready
Fourier best workspace candidate, if S1-ready
RBF / Wavelet monitor, only if workspace/task-health not catastrophic
```

## 5.4 Controls

```text
C0 AdamW
C1 NoOpMatchedOverhead
C2 RandomMatchedNorm
C3 AdamWParallelDirection
C4 Old-SNR-only, previous implementation
C5 ShuffledPerExampleGradient
C6 SameMaskRandomSign
C7 SameActiveFractionRandomMask
```

## 5.5 必须记录字段

```text
model_id
family
dataset
seed
synthetic_task
loss_interface
batch_size
microbatch_count
per_example_gradient_method
snr_tau
snr_active_fraction
snr_active_fraction_by_role
snr_active_fraction_by_basis_group
mean_mu2
mean_sigma2_over_bminus1
snr_median
snr_p90
snr_p99
update_norm
removed_update_norm_fraction
cos_snr_adamw
cos_snr_random
source_vs_adamw
source_vs_best_control
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
NLL_delta
ECE_delta
AUC_step_delta
AUC_time_delta
step_time_overhead
memory_overhead
```

## 5.6 判断标准

Exploration pass：

$$
source\_vs\_best\_control \ge 0.002,
$$

$$
\Delta CouplingR^2 \ge 0,
$$

$$
\Delta NoiseSignalLeak \le 0.005,
$$

$$
\Delta RealSignalReservoirRatio \le 0.005,
$$

$$
\Delta CEp99 \le 0.10,
$$

$$
overhead_{step} \le 1.10.
$$

Official synthetic pass：

$$
source\_vs\_best\_control \ge 0.005,
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

$$
\Delta CEp99 \le 0.05,
$$

$$
\Delta NLL \le 0.02,
$$

$$
\Delta ECE \le 0.02.
$$

Synthetic task-family gate：

```text
X1-X7 至少 5/7 family pass；
不能只靠 X7；
至少 2 个 seeds pass；
MLP-SNR 与 KAN-basis-SNR 必须同时报告。
```

## 5.7 失败后 Codex 必须尝试

如果 SNR active fraction 太低：

```text
try soft q;
try EMA mu/sigma over 3 windows;
try role-wise threshold normalization;
do not lower task/LineC gates.
```

如果 SNR active fraction 太高：

```text
increase tau;
cap active fraction per role;
compare SameActiveFractionRandomMask.
```

如果 MLP-SNR pass 但 KAN-SNR fail：

```text
basis metric / cover guard wrong；
进入 Line B/G，不继续 parameter-level SNR。
```

如果 KAN-SNR pass 但 MLP-SNR fail：

```text
basis coordinate 有价值；
进入 basis-channel SNR and real-data short-run。
```

如果两者都 fail：

```text
record PopRiskSNRNoGo_CurrentImplementation;
转 substrate/base architecture。
```

---

# 6. Line B：Basis-channel SNR + basis metric

## 6.1 目标

把 parameter-level SNR 提升到 basis-channel / cover level。理由：KAN 的优势不应来自普通参数 mask，而应来自显式 basis coordinate。

## 6.2 通用定义

对每个 basis channel $c$，把相关参数集合记为 $\theta_c$。聚合 per-example gradient：

$$
g_{i,c}=\operatorname{Agg}_{k\in\theta_c}(g_{i,k}).
$$

例如用 norm-projected scalar：

$$
g_{i,c}=\frac{\langle g_{i,\theta_c}, \mu_{\theta_c}\rangle}{\|\mu_{\theta_c}\|+\epsilon}.
$$

计算：

$$
SNR_c = \frac{\mu_c^2}{\sigma_c^2/(b-1)+\epsilon}.
$$

Basis-channel mask：

$$
q_c = 1\{SNR_c > \tau_c\}.
$$

最终 update：

$$
\Delta\theta
=
-\eta \sum_c q_c P_c M_c^{-1}\mu_{\theta_c}.
$$

## 6.3 Family-specific telemetry

### Rational

```text
den_p01
den_p99
r_prime_p95
r_double_prime_p95
tangent_condition
group_function_diversity
readout_rational_coupling
per_group_snr
```

Safety projection：

```text
denominator guard
slope guard
curvature guard
group diversity floor
```

### Chebyshev

```text
degree_energy_k
high_degree_ratio
degreewise_snr
recurrence_max_abs
```

Safety projection：

```text
high-degree cap
degree-energy damping
recurrence stability guard
```

### Fourier

```text
band_energy
high_frequency_ratio
phase_drift
bandwise_snr
frequency_noise_leak_proxy
```

Safety projection：

```text
high-frequency quarantine
phase stability guard
band energy cap
```

### RBF / FastKAN

```text
center_occupancy_entropy
dead_center_fraction
width_condition
out_of_grid_fraction
centerwise_snr
```

Safety projection：

```text
occupancy rebalance
width condition guard
OOG boundary guard
```

### Wavelet

```text
scale_energy
support_overlap
local_tail_coverage
scalewise_snr
```

Safety projection：

```text
scale energy cap
support overlap guard
local tail occupancy guard
```

## 6.4 必须记录

```text
basis_family
basis_channel_count
channel_snr_active_fraction
channel_snr_entropy
per_family_active_fraction
basis_safety_rejection_fraction
basis_safety_rejection_reason
post_safety_update_norm
channel_update_cos_with_adamw
channel_update_cos_with_param_snr
source_vs_best_control
LineC deltas
tail/calibration deltas
```

## 6.5 Gate

Basis-channel SNR 必须显著优于 parameter-level SNR：

$$
source\_vs\_control(BasisSNR)
\ge
source\_vs\_control(ParamSNR)+0.002.
$$

或：

$$
LineCScore(BasisSNR)
\ge
LineCScore(ParamSNR)+0.05.
$$

且 safety rejection fraction 不能过高：

$$
rejection\_fraction \le 0.70.
$$

如果 rejection fraction > 0.70，说明 basis substrate 不可控，应返回 substrate design。

---

# 7. Line G：Basis-cover boundary stabilization

## 7.1 目标

吸收 Deep Manifold 的启发：functional update 不是单步扰动，而是在训练中提供弱、对称、离散的 boundary condition，防止 moving covers 过度漂移或坍塌。

## 7.2 三阶段 plasticity schedule

### Phase 1：plasticity-open

```text
epoch/window early
SNR gate loose
cover guard only rejects catastrophic moves
目标：允许 basis cover 展开
```

### Phase 2：cover-alignment

```text
middle window
SNR gate normal
basis channel occupancy / energy / denominator / frequency / degree / scale 开始约束
目标：让 signal channel 稳定形成
```

### Phase 3：fixed-point consolidation

```text
late window
SNR gate conservative
cover guard stronger
目标：防止 noise leak、tail drift、high-frequency / curvature explosion
```

## 7.3 Boundary functional form

每个 basis family 定义 cover debt：

$$
D_{cover} = D_{occupancy}+D_{condition}+D_{tail}+D_{drift}.
$$

Functional update 只允许在：

$$
D_{cover}^{after} \le D_{cover}^{before}+\epsilon_D
$$

时提交。

## 7.4 必须记录

```text
phase
cover_debt_before
cover_debt_after
cover_debt_delta
plasticity_index
cover_entropy
cover_drift
basis_channel_rank
fixed_point_stability_proxy
boundary_rejection_count
boundary_accept_count
```

## 7.5 Gate

```text
boundary_accept_rate >= 0.20
catastrophic_cover_violation_rate = 0
source_vs_best_control not worse by more than 0.001
LineC not worse
CEp99/NLL/ECE not worse
```

如果 boundary guard 让所有 update 变成 no-op：

```text
route = R2-BoundaryOverConstrained
try phase-specific loosening, not global gate lowering
```

---

# 8. Line S：Substrate / base architecture reset

## 8.1 目标

v13.5 表明当前 substrate 上 oracle value target upper-bound fail。因此必须并行重启 substrate/base architecture，但不是回到随机 basis sweep，而是围绕 “can support SNR + cover boundary” 设计 substrate。

## 8.2 Candidate families

Active：

```text
Rational
Chebyshev
Fourier
RBF / FastKAN
Wavelet
```

Frozen：

```text
B-spline
```

## 8.3 Substrate-SNR gate

一个 basis 不再只看 workspace pass。它必须满足：

```text
workspace pass;
expression smoke pass;
minimum task not catastrophic;
channel_snr_entropy above floor;
safety_rejection_fraction not too high;
cover telemetry finite and non-degenerate.
```

Formal gate：

$$
step\_ratio \le 1.75,
$$

$$
incremental\_memory\_ratio \le 1.75,
$$

$$
mean\_delta\_vs\_MLP \ge -0.05,
$$

$$
AUCtime\_ratio \le 2.0,
$$

$$
channel\_snr\_entropy \ge 0.20,
$$

$$
safety\_rejection\_fraction \le 0.70.
$$

## 8.4 Family-specific next candidates

### Rational

```text
RAT-SNR1-denSlopeTelemetrySubstrate
RAT-SNR2-groupDiversityFloorSubstrate
RAT-SNR3-readoutRationalDecoupledSubstrate
```

### Chebyshev

```text
CHE-SNR1-degreeEnergyLowKSubstrate
CHE-SNR2-lateHighDegreeEnableSubstrate
CHE-SNR3-recurrenceStableSubstrate
```

### Fourier

```text
FOU-SNR1-lowBandAnchorSubstrate
FOU-SNR2-highBandQuarantineSubstrate
FOU-SNR3-phaseStableSubstrate
```

### RBF / FastKAN

```text
RBF-SNR1-compactOccupancySubstrate
RBF-SNR2-widthConditionedSubstrate
RBF-SNR3-oogBoundarySubstrate
```

### Wavelet

```text
WAV-SNR1-hatScaleStableSubstrate
WAV-SNR2-supportOverlapGuardSubstrate
WAV-SNR3-localTailCoverageSubstrate
```

## 8.5 Failure rules

If a family passes workspace but SNR telemetry degenerates:

```text
status = WorkspaceOnly_NotFunctionalSubstrate
```

If it has SNR telemetry but task catastrophic:

```text
status = SignalVisible_TaskCollapsed
```

If it is task viable but cover guard rejects all updates:

```text
status = TaskViable_CoverNotControllable
```

---

# 9. Line X：Synthetic family proof

## 9.1 目标

用 X1-X7 测 population-risk SNR 和 basis-cover boundary 是否跨任务族有效。

## 9.2 Methods

```text
AdamW
AdamW + parameter SNR
AdamW + basis-channel SNR
AdamW + basis-channel SNR + cover boundary
MLP + parameter SNR
MLP + hidden-channel SNR analog
```

## 9.3 Gate

必须达到：

```text
5 / 7 synthetic task family pass；
每个 pass 至少 2 seeds；
不能只靠 X7；
source_vs_best_control >= 0.005；
LineC non-harm；
CEp99 / NLL / ECE non-harm。
```

如果 `AdamW + parameter SNR` 已经解决大部分任务：

```text
route = GenericPopRiskOptimizerPositive
KAN-specific claim not allowed yet
```

如果 `basis-channel SNR + cover boundary` 明显优于 MLP-SNR：

```text
route = BasisCoordinateFunctionalAdvantageCandidate
```

If all fail:

```text
route = PopRiskSNRFunctionalNoGo
return to substrate/base architecture
```

---

# 10. Line M：MLP analog

## 10.1 目标

判断 functional update 是 generic optimizer 还是 KAN/basis-specific。

## 10.2 MLP methods

```text
MLP-AdamW
MLP-AdamW-ParameterSNR
MLP-HiddenChannelSNR
MLP-HiddenCovBoundary
MLP-WeightBalanceBoundary
```

## 10.3 判断

```text
MLP-SNR pass, KAN-SNR pass:
  generic optimizer positive; KAN-specific claim requires additional margin.

MLP-SNR fail, KAN basis-SNR pass:
  KAN explicit basis coordinate likely contributes.

MLP-SNR pass, KAN fail:
  KAN substrate / basis metric wrong.

Both fail:
  SNR route no-go under current implementation.
```

---

# 11. Line C：Manifold-channel audit

Line C 继续记录，但不生成 direction。

## 11.1 Metrics

```text
CouplingR2
CouplingCorr
NoiseSignalLeak
RealSignalReservoirRatio
KernelDrift
CEp99
NLL
ECE
Brier
margin_p10
train-probe displacement R2
signal effective rank
reservoir fraction
```

## 11.2 Role

```text
Audit only.
No direction source.
No threshold tuning by dataset.
No query/future/validation/test usage.
```

---

# 12. Required artifacts

```text
v136_route_decision.json
v136_progress_table.csv
v136_code_review_manifest.csv
v136_loss_interface_audit.csv
v136_per_example_gradient_stats.csv
v136_snr_parameter_update.csv
v136_snr_basis_channel_update.csv
v136_basis_cover_boundary.csv
v136_substrate_snr_gate.csv
v136_synthetic_family_results.csv
v136_mlp_analog_results.csv
v136_linec_audit.csv
v136_controls.csv
v136_failure_table.csv
v136_no_go_boundary.md
v136_next_hypothesis_queue.md
v136_required_manifest.csv
v136_code_review_packet.zip
```

---

# 13. Required visualizations

```text
fig_progress_lines_v1235_to_v136.svg
fig_snr_distribution_by_role.svg
fig_snr_active_fraction_by_family.svg
fig_parameter_snr_vs_basis_snr.svg
fig_basis_cover_debt_before_after.svg
fig_linec_deltas_by_method.svg
fig_ce_tail_calibration_by_method.svg
fig_synthetic_5of7_heatmap.svg
fig_mlp_vs_kan_snr_comparison.svg
fig_substrate_snr_health_matrix.svg
fig_failure_taxonomy.svg
```

---

# 14. Stop / go rules

## 14.1 Promotion rules

No promotion unless:

```text
1. implementation valid;
2. no forbidden information;
3. synthetic 5/7 pass;
4. matched controls beaten;
5. LineC non-harm;
6. CEp99/NLL/ECE non-harm;
7. efficiency overhead acceptable;
8. real-data short-run pass, if opened.
```

## 14.2 No-go rules

If parameter-SNR and basis-SNR both fail across MLP/KAN:

```text
R4-PopRiskSNRNoGo
```

If parameter-SNR works for MLP but not KAN:

```text
R5-KANSubstrateMetricMismatch
```

If basis-SNR works only for Rational and not other basis:

```text
R6-RationalSpecificFunctionalOnly
```

If cover-boundary rejects everything:

```text
R7-BasisCoverOverConstrained
```

If substrate telemetry degenerates:

```text
R8-SubstrateNotFunctionalControllable
```

---

# 15. Codex execution priorities

Codex must not stop after first fail. It must execute in this order:

```text
1. MLP parameter-SNR baseline.
2. Rational parameter-SNR baseline.
3. Rational basis-channel SNR.
4. Rational basis-cover boundary.
5. At least two Non-RAT substrate-SNR scouts: one Fourier, one Chebyshev or RBF.
6. Synthetic X1-X7 for any method with exploration pass.
7. MLP analog comparison.
8. Final no-go / next-hypothesis queue.
```

If step 1 fails due implementation, fix implementation before any KAN run.

If step 1 passes but all KAN fail, do not continue KAN token search; debug basis metric.

If Rational basis-SNR passes but Non-RAT fail, record Rational-specific route and design Non-RAT substrate separately.

---

# 16. 最终判断

v13.5 关闭了 oracle-target route，但没有关闭 functional update 本身。它告诉我们：

$$
\boxed{
\text{手造 basis-channel }\Delta Z\text{ 不是正确 value source。}
}
$$

结合两篇文献，下一步应该把 functional update 改成：

$$
\boxed{
\text{population-risk signal selection}
+
\text{basis-cover boundary stabilization}
}
$$

这不是小修，而是换掉 functional update 的信息接口：从 oracle / geometry target，换成训练时可见的 per-example gradient SNR；从一次性 perturbation，换成 basis cover 的弱边界维护。
