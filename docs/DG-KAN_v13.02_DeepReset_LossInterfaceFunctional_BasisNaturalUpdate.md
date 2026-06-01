# DG-KAN v13.2 深度重置计划：Loss-Interface Functional Update + Basis-Natural Update

> 版本：v13.2 strategic reset  
> 目标：停止 response-token / readout-proxy / 小网格路线，重新定义 functional update 的合法信息接口和作用对象。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no active B-spline；禁止 label-informed initialization；CE / NLL / ECE / CEp99 只能作为审计与坏化约束，不能作为 functional 目标函数。  

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是寻找某个只在单一 diagnostic 指标上好看的更新。最终目标是：

$$
\boxed{
\text{构建 label-free strict FC-PureKAN efficient substrate，}
\text{并用 loss-interface-generic functional update 获得比普通反向传播更好的训练几何和模型。}
}
$$

最终需要证明：

$$
\boxed{
\text{KAN substrate + functional update}
>
\text{same KAN substrate + AdamW / ordinary backprop controls}
}
$$

并且：

```text
1. 表达力不打折；
2. forward / backward / step / memory 与 MLP 可比；
3. AUC-step / AUC-time 不输；
4. LineC 几何不坏，最好改善；
5. CEp99 / NLL / ECE / Brier / margin tail 不坏；
6. functional 收益必须击败 NoOp / RandomMatchedNorm / AdamWParallel / SNR-only / MLP analog controls；
7. functional 机制不能针对 CE 设计，必须通过 generic loss-cotangent interface 工作。
```

## 0.2 当前进展事实

当前事实不是“完全没进展”，但也不能继续粉饰。

```text
1. Historical B320/FHQ 工程能力很强，但因 label-informed trainprobe init 被禁，不能作为 official base claim。
2. Label-free FHQ/A-DYN 多轮没有恢复 near-anchor。
3. Classic no-BSpline all-basis map 已完成，Rational 是唯一稳定 substrate family。
4. Rational workspace / telemetry 已经打开，但 healthy base 仍未成立。
5. Non-RAT family 还没有 usable substrate；Chebyshev/Fourier 有 exact kernel 正确性，但 lifetime/task-health 未闭合。
6. MLP functional 多轮 0 pass，当前 generic observable family 更接近 no-go。
7. Response dictionary / readout-feature transport / P3 token-style functional repair 没有产生 P3/P4 survivor。
8. v13.0 证明 readout-feature transport proxy 不够；full basis-parameter coordinate surgery 尚未执行。
```

因此当前阶段应从 v12.x 的“继续扩 token / response dictionary”切换到 v13.2 的新核心问题：

$$
\boxed{
\text{functional update 应该使用什么合法信息，作用在什么真实参数坐标上，才能改变未来训练几何？}
}
$$

---

# 1. 我们之前的关键误区

## 1.1 把 loss-agnostic 误写成 label-blind

过去几轮把 `loss-agnostic` 执行成了近似 `label-blind`：functional direction 不能读 label、不能读 CE、甚至不能读由 loss 产生的 output cotangent。结果是，functional 只能看 logits、activations、random cotangent、unlabeled geometry，然后试图预测 label-audited hard release。

这个要求过强，且可能不是我们真正想要的。

真正应该禁止的是：

```text
1. 针对 CE 的专门公式；
2. 用 CEp99 / NLL / ECE / LineC hard target 直接生成方向；
3. 用 validation/test/future/query batch；
4. dataset-name branch；
5. label-informed initialization。
```

但 supervised training 本身需要 boundary condition。Functional update 可以使用一个 **loss-interface-generic output cotangent**：

$$
\delta_y = \frac{\partial \ell(y, target)}{\partial y},
$$

只要 functional 模块不关心这个 cotangent 来自 CE、Brier、MSE、DPO 或其他 loss。

也就是说：

```text
禁止：CE-specific functional update。
允许：loss-interface-generic functional update。
```

这是本计划最重要的定义修正。

## 1.2 把 functional update 当成一次 additive perturbation

旧路线基本是：

$$
\theta' = \theta + \lambda d_{func}.
$$

然后看 one-step / P3 / P4 是否改善。多轮结果说明，这种 direction 多数只是移动 output/readout 表面，没有改变未来训练会走的 tangent / basis coordinate。

functional update 需要从 “additive output perturbation” 改成：

$$
\boxed{
\text{basis-coordinate-aware preconditioned update / coordinate transport / optimizer-state transport。}
}
$$

## 1.3 把 basis 失败和 functional 失败混在一起

用户提出的判断是正确的：基函数首先应解决效率和可控坐标问题；AUC、tail、LineC 健康可以交给 functional update 修。但是 substrate 不能坏到不可修。

因此要区分三层：

```text
S0 WorkspaceOnly:
  跑得动，但任务/几何灾难，不能进入 functional official。

S1 EfficientControllableSubstrate:
  跑得动，有表达，有 telemetry，训练不灾难，可以进入 basis-specific functional repair。

S2 HealthyBase:
  basis-only 已经接近 MLP，可以直接 official。
```

functional update 不应要求 basis-only 已经 S2；它应该在 S1 上证明修复能力。

---

# 2. 当前各线完成度估计

| 线 | 当前完成度 | 判断 |
|---|---:|---|
| 代码 / provenance / finalizer 审计 | 98% | artifact discipline 已经成熟 |
| Historical FHQ/B320 工程能力 | 85% frozen | 强，但 label-informed init 禁用 |
| Label-free FHQ/A-DYN | 20%-25% | 多轮未恢复 near-anchor，降级 monitor |
| LineC 几何审计 | 80%-85% | 作为审计可用，但不能直接作方向源 |
| Rational substrate | 80%-85% | 当前唯一稳定 S1 候选，但不是 healthy base |
| Chebyshev substrate | 35%-40% | correctness/A4 有路，lifetime/task-health 未闭合 |
| Fourier substrate | 40%-45% | exact kernel 有路，lifetime/expression/task 未闭合 |
| RBF/FastKAN substrate | 35%-40% | compact efficiency 有路，expression/task 不足 |
| Wavelet substrate | 35%-40% | local kernel 有路，task/LineC 不健康 |
| MLP functional | 10%-15% | 当前 observable family no-go 倾向强 |
| Basis-specific functional | 5%-10% | P3/P4 仍未成立，需重定义信息接口 |
| Full basis-parameter surgery | 0%-5% | 尚未真正执行 |
| 整体 next-gen MLP claim | 40%-45% | 工程基础强，但 official 科学闭环未开 |

这个表故意比之前更保守。因为 v13.0 暴露：很多之前的 functional 结果只是 readout-feature proxy，不是真正参数坐标更新。

---

# 3. 新核心假设

## H1：存在 efficient controllable substrate，不必先是 healthy base

对于 Rational / Chebyshev / Fourier / RBF / Wavelet，每个 family 先争取 S1：

$$
S1 = Efficiency + Expression + NonCatastrophicTask + Telemetry + ControllableCoordinates.
$$

S1 不要求 CEp99 / AUC / LineC 全过，但要求不灾难：

$$
mean\_delta \ge -0.05,
$$

$$
worst\_delta \ge -0.12,
$$

$$
AUCtime\_ratio \le 2.0,
$$

$$
LineC\_pass\_rate \ge 0.20,
$$

并且必须有 basis telemetry。

## H2：functional update 应使用 generic loss cotangent，而不是 CE-specific 或 label-blind

定义统一接口：

```python
delta_y = loss_interface.output_cotangent(logits, targets_or_boundary)
```

functional 模块只接收 `delta_y`，不读取 CE 类名，不计算 CEp99，不用 validation/test，不按 dataset 分支。

为了证明它不是 CE-specific，必须至少在两个 loss interface 下跑 smoke：

```text
CE cotangent
Brier / squared-probability cotangent 或 MSE-logit diagnostic cotangent
```

如果 functional 只在 CE cotangent 下成立，在 Brier/MSE 下完全失效，则不能 claim loss-agnostic。

## H3：basis-natural update 比 response-token 更可能成立

新的 functional update 形式是：

$$
\Delta\theta_{func}
=
-\eta \; P_{safe}\; P_{snr}\; M_{basis}^{-1} g_{basis}.
$$

其中：

```text
g_basis:
  由真实 basis 参数的 per-example gradient / cotangent 得到。

M_basis:
  由输入、basis tangent、basis derivative、occupancy、group statistics 构造，不用 CE。

P_snr:
  population-risk / SNR gate，使用 per-example gradient mean/variance，不硬编码 CE。

P_safe:
  basis-specific safety projection，例如 Rational denominator/derivative guard，Fourier band cap。
```

这不是一次 output perturbation，而是一个真实 basis-coordinate preconditioner。

## H4：MLP functional 线应作为机制判别，而非主路线

如果 MLP 上同样成立，functional 是通用 optimizer / geometry mechanism；如果 MLP 不成立但 KAN basis 成立，说明 KAN 显式 basis coordinate 是关键；如果两者都不成立，则当前 value source 不成立。

---

# 4. 总实验路线

v13.2 分成八条线：

```text
Line R: Code / implementation audit。
Line S: Efficient controllable substrate map。
Line B: Basis-natural functional update。
Line X: Parameter-level synthetic mechanism proof。
Line P: Real-data short-run repair。
Line M: MLP analog control。
Line C: Manifold-channel audit。
Line Z: Finalizer / no-go / next hypothesis。
```

其中最重要的是 Line B 和 Line X。没有真实参数级 functional update，就不再进入 P4/P5 宣称。

---

# 5. Line R：代码与实现语义审计

## 目标

确保 functional update 真的作用在真实 basis 参数上，而不是 frozen feature table 或 readout proxy。

## Codex 必须定位

```text
1. 每个 basis family 的真实参数：basis coeff / rational numerator / denominator / frequency / center / width / scale / readout。
2. per-example gradient 获取路径。
3. manual / analytic backward 路径。
4. optimizer state 路径。
5. functional update 写回参数的路径。
6. 是否读 label / CE / validation / future / dataset name。
7. 是否只是 readout-feature proxy。
```

## 必须落盘

```text
v132_code_surface.csv
v132_basis_param_manifest.csv
v132_functional_writeback_trace.csv
v132_loss_interface_audit.csv
v132_forbidden_information_audit.csv
```

## Gate

若任何 official functional candidate 满足以下任一项，直接 route fail：

```text
readout_feature_proxy_only = 1
full_basis_param_update = 0
loss_interface_generic = 0
uses_validation_or_test = 1
uses_dataset_name_branch = 1
uses_ce_specific_formula = 1
uses_label_for_init = 1
```

---

# 6. Line S：Efficient controllable substrate map

## 目标

所有 active no-BSpline basis 都按 S0/S1/S2 评级。

Active family：

```text
Rational
Chebyshev
Fourier
RBF / FastKAN
Wavelet
```

B-spline 继续 frozen。

## 指标

### Efficiency

```text
forward_ratio
backward_ratio
step_ratio
raw_memory_ratio
incremental_memory_ratio
optimizer_state_ratio
workspace_peak_ratio
```

### Expression

```text
E1_pairwise_R2
E2_composition_R2
E6_rotated_pairwise_R2
E8_random_quadratic_R2
expression_delta_vs_MLP
```

### Minimal trainability

```text
mean_delta_vs_MLP
worst_delta_vs_MLP
AUC_step_ratio
AUC_time_ratio
near_pass_rate
ECE_delta
CEp99_delta
```

### Telemetry availability

Rational：

```text
den_p01
den_p99
r_prime_p95
r_double_prime_p95
tangent_condition
group_diversity
readout_rational_coupling
```

Chebyshev：

```text
degree_energy
high_degree_ratio
degree_tangent_condition
recurrence_max_abs
```

Fourier：

```text
band_energy
high_freq_ratio
phase_stability
frequency_gradient_norm
```

RBF/FastKAN：

```text
center_occupancy
width_condition
out_of_grid_fraction
dead_center_fraction
```

Wavelet：

```text
scale_energy
support_overlap
active_support_entropy
local_tail_coverage
```

## S1 Gate

$$
step\_ratio \le 1.75,
$$

$$
incremental\_memory\_ratio \le 2.0,
$$

$$
mean\_delta \ge -0.05,
$$

$$
worst\_delta \ge -0.12,
$$

$$
AUCtime\_ratio \le 2.0,
$$

$$
LineC\_pass\_rate \ge 0.20,
$$

$$
telemetry\_available = 1.
$$

S1 是进入 functional repair 的最低门，不是 final base success。

---

# 7. Line B：Basis-natural functional update

## 目标

构造真实参数级、loss-interface-generic、basis-specific functional update。

## 通用公式

每个样本的 loss cotangent：

$$
\delta_i=\frac{\partial \ell(y_i,target_i)}{\partial y_i}.
$$

每个样本的 basis 参数梯度：

$$
g_i=J_{\theta}(x_i)^T\delta_i.
$$

batch mean 和 variance：

$$
\mu=\frac{1}{b}\sum_i g_i,
$$

$$
\sigma^2=\frac{1}{b-1}\sum_i (g_i-\mu)^2.
$$

SNR gate：

$$
SNR_k = \mu_k^2 - \frac{\sigma_k^2}{b-1}.
$$

basis metric：

$$
M_{basis}=\mathbb{E}[J_\theta(x)^TJ_\theta(x)] + \alpha R_{basis} + \rho I.
$$

functional update：

$$
\Delta\theta_{func}=-\eta P_{safe}P_{snr}M_{basis}^{-1}\mu.
$$

这里 $P_{safe}$ 与 $M_{basis}$ 是 family-specific；$\mu,\sigma$ 来自 generic loss interface，不硬编码 CE。

## Family-specific safety

### Rational

```text
Denominator floor guard
r' / r'' slope guard
Tangent condition trust region
Group diversity preservation
Readout-rational decoupling
```

### Chebyshev

```text
Degree-energy damping
High-degree late-enable
Degree tangent trust region
Recurrence bound guard
```

### Fourier

```text
Frequency-band damping
High-frequency noise-leak veto
Phase stability guard
Low-frequency anchor preservation
```

### RBF / FastKAN

```text
Center occupancy rebalance
Width condition guard
OOG boundary recenter
Dead center revival
```

### Wavelet

```text
Scale-energy balance
Support-overlap guard
Local-tail coverage repair
Active support entropy preservation
```

## Controls

每个 functional candidate 必须对照：

```text
C0 TaskOnly AdamW
C1 NoOpMatchedOverhead
C2 RandomMatchedNorm
C3 AdamWParallelDirection
C4 SNR-only diagonal gate
C5 MLP analog functional
C6 shuffled loss-cotangent diagnostic
```

## P3 Gate

$$
source\_vs\_best\_control \ge 0.005
$$

或 bootstrap lower CI $
\ge 0$。

并且：

$$
\Delta CouplingR^2 \ge 0.01,
$$

$$
\Delta NoiseSignalLeak \le 0,
$$

$$
\Delta RealSignalReservoirRatio \le 0,
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

注意：CEp99/NLL/ECE 是审计，不参与方向生成。

---

# 8. Line X：Parameter-level synthetic mechanism proof

## 目标

先在真实 module / 真实参数上证明 functional update 可以改变未来训练几何，而不是继续在真实数据上盲试。

Synthetic tasks：

```text
X1 pairwise product
X2 rotated quadratic
X3 random quadratic
X4 low/high frequency mixture
X5 local bump / tail cluster
X6 two-manifold class split
X7 label-noise split audit
```

## 要求

所有 synthetic proof 必须：

```text
1. 实例化真实 model module；
2. 修改真实 basis 参数；
3. 使用真实 manual/analytic backward；
4. 写回 optimizer state 或明确 no optimizer state；
5. 禁止 feature-table proxy。
```

## Gate

Synthetic success：

$$
future\_train\_loss\_AUC_{func} < future\_train\_loss\_AUC_{AdamW}
$$

并且：

$$
LineC_{func} \ge LineC_{AdamW}
$$

在至少 5/7 synthetic tasks 上成立。

如果 synthetic 都失败，不进入 real P4。

---

# 9. Line P：Real-data short-run repair

只允许在以下条件下打开：

```text
1. basis family 有 S1 substrate；
2. Line B P3 pass；
3. Line X synthetic pass；
4. implementation audit pass。
```

训练形式：

$$
\theta_{t+1}=\theta_t + \Delta\theta_{task} + \lambda_t \Delta\theta_{func}.
$$

functional event 低频触发，不替代 task optimizer。

记录：

```text
val_acc
val_loss_auc_step
val_loss_auc_time
source_vs_control
CEp99
NLL
ECE
Brier
margin_p10
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
functional_event_count
accepted_event_count
rejected_event_count
amortized_overhead
```

Gate：

$$
Acc_{func}\ge Acc_{base}-0.003,
$$

$$
AUCtime_{func}\le AUCtime_{base},
$$

$$
source\_vs\_best\_control \ge 0.005,
$$

$$
CEp99_{func}\le CEp99_{base}+0.05.
$$

---

# 10. Line M：MLP analog control

MLP analog 不能再只跑 weak objective。它要作为机制判别。

候选：

```text
M1 Hidden orthogonal rotation + inverse readout compensation
M2 Hidden whitening + readout ridge compensation
M3 Layerwise weight balancing + optimizer-state transport
M4 Activation covariance preconditioned update
M5 Generic loss-cotangent SNR gate
```

判断：

```text
If MLP analog works and KAN works:
  functional is generic, not KAN-specific.

If MLP fails and KAN basis works:
  KAN explicit basis coordinates matter.

If both fail:
  current functional mechanism no-go.
```

---

# 11. Line C：Audit only

继续记录：

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
top1_top2_gap
classwise tail
```

LineC 不允许作为 direction source，只能作为 audit。

---

# 12. 必须可视化

```text
fig_substrate_s1_map.svg
fig_basis_telemetry_vs_task.svg
fig_basis_telemetry_vs_linec.svg
fig_snr_positive_fraction.svg
fig_functional_vs_controls.svg
fig_synthetic_future_probe_auc.svg
fig_real_short_run_auc_time.svg
fig_ce_tail_nll_ece_audit.svg
fig_mlp_vs_kan_functional_comparison.svg
fig_no_go_boundary.svg
```

---

# 13. Failure -> Codex 必须先尝试什么

## Case A：没有 S1 substrate

Codex 不能跑 functional。必须先做：

```text
1. fused/no-materialize kernel repair；
2. workspace lifetime repair；
3. expression capacity repair；
4. minimal trainability rescue。
```

## Case B：有 S1 substrate，但 Line B P3 fail

Codex 必须：

```text
1. 输出 per-example gradient SNR histogram；
2. 输出 basis metric condition；
3. 输出 safety projection rejection reason；
4. 尝试 family-specific metric repair；
5. 尝试 MLP analog对照。
```

不能直接写 no-go。

## Case C：P3 pass，但 synthetic fail

Codex 必须：

```text
1. 检查是否仍是 feature-table proxy；
2. 检查 optimizer-state transport；
3. 检查 update norm / clipping；
4. 检查 future probe是否被 task optimizer覆盖。
```

## Case D：Synthetic pass，但 real fail

Codex 必须：

```text
1. 做 train-shuffle robustness；
2. 做 dataset-agnostic failure slice；
3. 做 control-residualized update；
4. 做 tail / calibration audit；
5. 不能按 dataset 调参。
```

## Case E：MLP analog pass

Codex 必须：

```text
1. 给出 generic functional route；
2. 不得 claim KAN-specific；
3. 对 KAN / MLP 做 paired factorial design。
```

---

# 14. Stop rule

允许停止的条件：

```text
1. official success reached；
2. 所有 S1 substrate family 都已完成 Line B/X/M 判定；
3. hard compute budget exhausted 且每个 failure case 都有 next hypothesis / no-go boundary；
4. user_stop_flag = 1。
```

不允许停止：

```text
1. 只跑了 readout-feature proxy；
2. 只做了 response dictionary；
3. 只扩 token 网格；
4. synthetic 没用真实参数；
5. P3 fail 后没有输出 gradient/SNR/metric diagnosis。
```

---

# 15. 最终判断

v13.2 的核心不是继续寻找一个更漂亮的 diagnostic row，而是重新定义 functional update 的合法信息接口与真实作用对象。

$$
\boxed{
\text{Functional update 不应是 label-blind output perturbation，}
\text{而应是 loss-interface-generic 的 basis-natural parameter update。}
}
$$

如果这条路失败，我们才有资格说当前 functional update 假设在 strict FC-PureKAN 下不成立。否则继续跑旧 token 只是浪费时间。

