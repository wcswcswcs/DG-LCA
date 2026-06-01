# DG-KAN v13.9：Signal-to-Cover Functional Update 与 Basis Substrate Architecture 重置完整计划

> 版本：v13.9 execution plan  
> 基于：v13.8 `Generic PopRisk + KAN SNR-to-Basis Lift` 真实结果复盘，以及 `A Theory of Generalization in Deep Learning` 与 `Deep Manifold Part 2` 的重新启发  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no label-informed initialization；CEp99 / NLL / ECE / LineC hard target 只能作为 audit / gate，不能作为 direction source；functional direction 只允许来自 current train batch、generic loss interface、train-stream per-example gradient statistics、basis telemetry 与 optimizer state；不能使用 validation / test / future / query batch 生成方向。

---

# 0. 项目总目标与当前进展

## 0.1 总目标

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是寻找一个只在某个 toy score 上好看的 functional trick。总目标是：

$$
\boxed{
\text{构建一个 label-free strict FC-PureKAN / KAN-basis substrate，}
\text{在表达力和效率不输 MLP 的条件下，}
\text{通过 functional update 获得比普通反向传播更好的训练几何和模型。}
}
$$

更具体地说，最终成功必须满足：

```text
1. strict FC-PureKAN / no non-KAN trainable parameter；
2. no label-informed init；
3. forward / backward / step / memory 与 MLP 可比；
4. 表达力不打折；
5. task trajectory 不比 MLP 慢，AUC-step / AUC-time 不坏；
6. LineC 几何更健康：真实信号进入 signal channel，噪声不进入 signal channel，真实信号不困在 reservoir；
7. CEp99 / NLL / ECE / Brier / margin 不坏；
8. functional update 必须打过 AdamW / AdamWParallel / RandomMatchedNorm / NoOp / SNR-only controls；
9. functional update 必须是 loss-interface-generic，不是 CE-specific；
10. 不允许按数据集调 controller。
```

最终主张不是：

$$
\text{KAN base 单独比 MLP 好。}
$$

也不是：

$$
\text{MLP 上一个 generic SNR optimizer 有一点正信号。}
$$

而是：

$$
\boxed{
\text{KAN explicit basis cover} + \text{functional update}
>
\text{same KAN basis} + \text{ordinary AdamW/backprop controls}
}
$$

## 0.2 当前总状态

v13.8 的合法结论是：

```text
route = R1-GenericSNROptimizerNoGo
minimum_success = S1-ImplementationReadback
promotion_allowed = 0
official_success_reached = 0
kan_real_short_run_open_allowed = 0
```

本轮闭合事实：

```text
1. MLP generic optimizer 10-seed 未确认：budgeted official = 0/3 dataset pass，full-size MLP repair = 0/3 dataset pass。
2. K0 signal transfer gate 通过：median retention = 0.8716974258422852，median cosine = 0.7423757314682007。
3. K1-K7 official 只达到 KAN S3 3/7、S4 1/7；后续 200-step repair 最高也只到 S3 4/7、S4 2/7。
4. K1 targeted longer repair 对 X2/X3/X5/X6 只达到 1/4 task pass。
5. Non-RAT substrate-health pass = 0；FOU20、RBF17、WAV16 exact vertical slices 仍被 workspace gate 拒绝。
6. required artifact missing = 0，forbidden/provenance violation = 0。
7. MLP-only positive 没有被写成 KAN promotion。
```

因此当前不能继续把 v13.8 当作“差一点”。它说明：

$$
\boxed{
\text{generic PopRisk-SNR signal 并非完全不存在；}
\text{但它不能稳定成为 10-seed generic optimizer，}
\text{也没有被 KAN basis cover 转化成 KAN-specific advantage。}
}
$$

---

# 1. 各条线进展百分比：v13.7 后估计 vs v13.8 后估计

这些百分比不是 artifact 中的官方字段，而是根据 gate 通过情况、机制清晰度、代码实现程度和距离 official success 的综合估计。

| 线 | v13.7 后估计 | v13.8 后估计 | 变化 | 判断 |
|---|---:|---:|---:|---|
| Line R：代码 / provenance / finalizer 审计 | 99% | **99%** | 0 | 工程闭包强，不是 blocker。 |
| Historical FHQ / B320-current 工程能力 | 85% frozen | **85% frozen** | 0 | 历史强，但 label-informed init 已禁用，不能 official。 |
| Label-free FHQ / A-DYN monitor | 10%-12% | **8%-10%** | -2 | 已非主线，继续低预算 monitor。 |
| Line C 几何审计 | 88% | **88%** | 0 | 审计可用，但不能作为 direction source。 |
| Rational S1 efficient substrate | 85% | **85%** | 0 | 仍是唯一稳定 substrate family。 |
| Rational S1C channel controllability | 60% | **60%** | 0 | 可执行 basis-channel movement，但 value 未成立。 |
| Population-risk SNR implementation | 80% | **86%** | +6 | K1-K7 surface、writeback trace、full-size repair、artifact 更完整。 |
| Generic MLP optimizer claim | 35% | **18%-22%** | -13 到 -17 | v13.7 的 5-seed positive 未能扩展到 10-seed；G6/G7/G8/G9 full-size 全 fail。 |
| MLP optimizer line as diagnostic | 50% | **45%** | -5 | 作为对照仍有价值，但不再主投小修。 |
| K0 SNR transfer audit | 新增/未充分估计 | **75%** | +75 | median retention / cosine 过 gate，说明 basis/group lift 不是完全丢信号。 |
| KAN SNR-to-basis lift | 8%-10% | **25%-30%** | +17 到 +20 | 有局部 source；official 3/7，200-step 4/7；仍未到 S3。 |
| KAN basis-cover functional official | 0%-5% | **0%-5%** | 0 | S3/S4/S5 未达成，real short-run 不允许打开。 |
| Non-RAT substrate / S1C | 10%-20% | **8%-15%** | -2 到 -5 | FOU/RBF/WAV exact vertical slice 仍 workspace fail。 |
| Classic no-BSpline portfolio 总体 | 60%-65% | **58%-62%** | -2 | Rational 仍唯一可用 substrate，其他 basis 还不能进入 functional proof。 |
| Signal-to-cover mechanism | 新增 | **0%-5%** | +5 | 目前只是下一步假设，尚未实现。 |
| 整体 next-gen MLP claim | 42%-48% | **38%-44%** | -4 | v13.8 证伪 MLP generic 10-seed，也没打开 KAN S3；整体应下调。 |

---

# 2. v13.8 结果独立分析

## 2.1 这次有没有进展？

有，但不是能力突破。

v13.8 相比 v13.7 的最大进展不是模型变强，而是把之前混在一起的三个问题拆开了：

```text
Q1: MLP PopRisk-SNR 是否是稳定 generic optimizer？
Q2: KAN basis/group lift 是否把 parameter-level SNR signal 丢掉？
Q3: KAN 是否能把 generic SNR signal 转成 basis-specific advantage？
```

v13.8 的答案是：

```text
Q1: 否。MLP 10-seed generic optimizer 未确认；full-size G6/G7/G8/G9 都不能 3/3 dataset pass。
Q2: 不支持。K0 signal transfer gate 通过，median retention = 0.8717，median cosine = 0.7424。
Q3: 还不能。K1-K7 有局部 source，但官方只到 S3 3/7，200-step 最高 S3 4/7，低于 >=5/7 gate。
```

所以它不是“没跑出来任何东西”，而是非常明确地告诉我们：

$$
\boxed{
\text{问题不再是“有没有 PopRisk signal”，}
\text{也不是“basis lift 完全丢 signal”，}
\text{而是“signal 没有变成稳定的 basis cover 改善”。}
}
$$

## 2.2 这次为什么仍然不能 promotion？

因为所有 official success 都是 0：

```text
MLP generic 10-seed confirmed = 0
KAN S3 >=5/7 = 0
KAN S4 >=5/7 = 0
Non-RAT substrate-health pass = 0
KAN real short-run open allowed = 0
promotion_allowed = 0
```

即使 KAN 200-step repair 后达到 S3 4/7、S4 2/7，它仍然没有达到 synthetic family coverage gate。K4F freeze-cluster fallback 也没有打开 X1 / X4 / X7 family；Non-RAT FOU20 / RBF17 / WAV16 exact vertical slice 仍被 workspace gate 拒绝；MLP G7 full-size 10-seed 也 0/3 dataset pass。因此停止继续小修是正确的。

## 2.3 不是 source 完全没有，而是 source 不稳定且不落入正确 channel

v13.8 的 K1-K7 rows 显示局部 source 可以很强，有些 X4 / X7 行 source 数值不低；但它们不能稳定满足 LineC / reservoir / tail 条件，且 X2 / X3 / X5 / X6 覆盖不足。

这说明：

```text
1. 当前 KAN update 能推动某些 synthetic family 的 immediate source；
2. 但是这个推动不稳定进入 test-visible / signal channel；
3. 它也没有稳定把 noise 留在 reservoir 或避免 tail / calibration 坏化；
4. 因此它不是 functional geometry success。
```

这正好对应 generalization 文档的 signal/reservoir 观点：好 update 不是“让输出动得更大”，而是让真实 signal 进入 signal channel、让 noise 不进入 signal channel、让 residual signal 不困在 reservoir。

## 2.4 为什么 MLP 线要降级？

v13.7 曾经给出 MLP-SNR 的 synthetic positive 和 5-seed real positive，因此我之前把 MLP generic optimizer 线看得偏乐观。v13.8 完成了 full-size 10-seed repair：G6 TrainLossQuantileTrust、G7 LogitNormTrust、G8 ActiveFractionSchedule、G9 PerExampleGradientClip 都没有确认 3/3 dataset pass。尤其 G7 full-size 10-seed 中，MNIST seed0 虽有 source/AUC 改善，但 NLL_delta 超过 gate；Fashion-MNIST 与 KMNIST 最接近 rows 仍 source_vs_adamw < 0 且 AUC_time_ratio > 1。

所以 MLP 线现在应该变成：

```text
reference / diagnostic / sanity control。
```

不应该继续主投：

```text
G10/G11/G12 generic SNR optimizer 小修。
```

除非出现新理论，不再扩 MLP SNR trust 小网格。

---

# 3. 结合两篇文档后的新判断

## 3.1 Generalization paper 给出的真正启发

Generalization 文档的关键不是“再做一个 SNR optimizer”，而是：训练有效性取决于 **coherent drift vs stochastic diffusion**。它给了 per-example gradient 的 population-risk结构：

$$
\bar g_B=\frac{1}{b}\sum_i g_i,
$$

$$
\Sigma_B=\frac{1}{b}\sum_i(g_i-\bar g_B)(g_i-\bar g_B)^T,
$$

$$
A_B=\bar g_B\bar g_B^T-\frac{1}{b-1}\Sigma_B.
$$

对 diagonal gate，它对应：

$$
\mu_k^2>\frac{\sigma_k^2}{b-1}.
$$

v13.6 / v13.7 / v13.8 证明：这个方向不能只做成 one-shot update，也不能简单做成 generic MLP optimizer。它需要和模型的 channel / cover 结构绑定。

换句话说：

$$
\boxed{
\text{PopRisk-SNR 是边界条件，不是最终功能。}
}
$$

它告诉我们哪些训练信号是 coherent 的，但并不自动告诉 KAN 哪个 basis cover 应该承载这些信号。

## 3.2 Deep Manifold 给出的真正启发

Deep Manifold 文档强调：网络训练不是固定坐标里的静态优化，而是 moving coordinates / shifting node covers / boundary-conditioned iteration。也就是说，basis 不是静态函数库，而是 cover / chart：

```text
Rational: denominator / slope / curvature / group cover
Fourier: frequency-band cover
Chebyshev: degree-energy cover
RBF: center / width / local occupancy cover
Wavelet: scale / support cover
```

所以 functional update 不应该只是：

```text
在已有参数上乘一个 SNR mask。
```

而应该是：

$$
\boxed{
\text{把 population-risk signal 分配到合适的 basis cover，}
\text{并让 cover 在训练中形成稳定的 signal channel。}
}
$$

这就是 v13.8 之后的核心新方向：

$$
\boxed{
\text{SNR-to-BasisLift} \rightarrow \text{Signal-to-Cover Formation}
}
$$

---

# 4. 当前真正卡在哪里

## 4.1 blocker 不是 signal transfer 丢失

K0 已经通过：

```text
median retention = 0.8716974258422852
median cosine = 0.7423757314682007
```

这说明 parameter-level SNR signal 映射到 basis/group coordinate 时，不是完全消失。继续只修 `cos_group_vs_param`、`signal_retention_group` 的边际价值有限。

## 4.2 blocker 是 signal 没有落到稳定 cover

当前真正问题是：

$$
\boxed{
\text{basis/group coordinate 能保留一部分 signal，}
\text{但这些 signal 没有形成稳定的 cover specialization。}
}
$$

症状是：

```text
1. K1-K7 有局部 source；
2. 但 LineC / reservoir / tail 不稳定；
3. X2/X3/X5/X6 覆盖不足；
4. 200-step 仍不能到 >=5/7；
5. Freeze-cluster 也不能修复 X1/X4/X7；
6. Non-RAT 没有 substrate-health，因此不能验证 basis-general functional。
```

这不是再调 cover threshold 能解决的。需要让 basis cover 在训练过程中更明确地承担不同类型的 signal。

## 4.3 Non-RAT 仍然是系统性短板

当前只有 Rational 是稳定 substrate。Fourier、RBF、Wavelet、Chebyshev 仍主要卡在 workspace/lifetime 或 task-health。v13.8 的 FOU20、RBF17、WAV16 exact vertical slice 显示：raw/step 有时接近 gate，但 incremental memory 仍高，且 exact_kernel_implemented=0 或仍 materializes basis/derivative/readout grad。

因此，不能说“所有基函数都可以交给 functional update 修”。更准确是：

```text
Rational 可进入 signal-to-cover functional 主线；
Non-RAT 先必须成为 controllable substrate；
否则没有可操作对象。
```

---

# 5. v13.9 核心假设

## H1：PopRisk signal 是有用边界，但需要 cover 分配机制

假设：

$$
\text{parameter-SNR} \rightarrow \text{basis/group-SNR}
$$

不是最终机制。真正需要的是：

$$
\text{parameter-SNR} \rightarrow \text{cover assignment} \rightarrow \text{cover plasticity schedule}.
$$

判定：如果 signal-to-cover 后 KAN 在 synthetic >=5/7 上通过，而 K1-K7 不通过，则说明过去的 fixed basis lift 缺少 cover formation。

## H2：Rational 的问题不是 denominator catastrophe，而是 cover specialization 不够

Rational rejection audits 多次没有显示 den / r' / r'' 数值灾难。因此 Rational blocker 更可能是：

```text
group cover 没有分化成 task-relevant channels；
或 readout/numerator/denominator/projection roles 承载 signal 的方式错配。
```

判定：如果 group specialization / cover assignment 明显改善 LineC 与 synthetic coverage，则 H2 成立。

## H3：MLP generic optimizer 线应降级为 control，不再主投

v13.8 已经补齐 G6/G7/G8/G9 full-size 10-seed repair，仍 0/3 dataset pass。因此 MLP-SNR 不应继续作为主线扩展。

判定：只保留 MLP-AdamW 与 best MLP-SNRBlend 作为 reference / control，不再扩 G10/G11 小修。

## H4：Non-RAT 需要真 exact no-materialize substrate，不是 functional 修补

FOU20/RBF17/WAV16 都显示 workspace / incremental memory 仍未过。不能把这些 family 送进 functional proof。

判定：Non-RAT 只有在 substrate-health 过后才进入 Line K。

---

# 6. v13.9 实验总线

v13.9 分为六条线。

```text
Line R: Code / provenance / implementation readback
Line G: MLP generic optimizer closure as control
Line K0: Signal-to-cover audit
Line K1: Rational signal-to-cover mechanism
Line D: Non-RAT exact substrate vertical slices
Line C: Manifold-channel audit only
Line Z: finalizer / no-go / next hypothesis
```

重点不是继续 SNR threshold 小网格，而是回答：

$$
\boxed{
\text{generic PopRisk signal 能否被 KAN basis cover 稳定承载？}
}
$$

---

# 7. Line R：代码与 provenance 审计

## 7.1 目标

确认 v13.9 没有把旧 K1-K7、MLP G6-G9、小修 token 伪装成新机制。

## 7.2 必须记录

输出：`v139_code_provenance_audit.csv`

字段：

```text
file
symbol
line_start
line_end
changed
change_reason
uses_label_in_init
uses_y_for_stats
uses_validation_for_direction
uses_test_for_direction
uses_future_for_direction
uses_query_batch_for_direction
uses_ce_tail_as_direction
uses_linec_target_as_direction
uses_generic_loss_interface
uses_per_example_gradients
uses_basis_telemetry
uses_cover_assignment
uses_true_parameter_writeback
uses_readout_feature_proxy
uses_feature_table_proxy
```

## 7.3 Gate

```text
forbidden_information_violation_count = 0
readout_feature_proxy_only = 0
feature_table_proxy_only = 0
uses_label_in_init = 0
```

若失败，route = `R0-ProvenanceViolation`。

---

# 8. Line G：MLP generic optimizer closure as control

## 8.1 目标

不再主投 MLP SNR 小修；只保留一个 closure/control baseline。

## 8.2 方法

只跑：

```text
MLP-AdamW
MLP-best-v13.8-SNRBlend
MLP-best-v13.8-G7-LogitNormTrust
```

设置：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..4 for cheap monitor
train_size = 1024
val/test = 512
epochs = 3
```

## 8.3 记录

输出：`v139_mlp_control_monitor.csv`

字段：

```text
dataset
seed
method
source_vs_adamw
AUC_time_ratio
CEp99_delta
NLL_delta
ECE_delta
pass
```

## 8.4 Gate

这条线不允许 promotion 为 KAN success。若 MLP 仍不过，则记录：

```text
GenericMLPSNRStopped_v13_9 = 1
```

若 MLP 突然大幅过，也只能写：

```text
Generic optimizer signal exists; KAN-specific advantage not established.
```

---

# 9. Line K0：Signal-to-Cover Audit

## 9.1 目标

不再只看 parameter-SNR 到 group-SNR 的保留率，而是测 signal 是否形成了稳定 cover specialization。

## 9.2 核心定义

对每个 sample $i$、参数或 basis channel $k$，有 per-example gradient：

$$
g_{i,k}=J_{\theta_k}(x_i)^T \delta_i.
$$

定义 parameter-level SNR：

$$
SNR_k=\frac{\mu_k^2}{\sigma_k^2/(b-1)+\epsilon}.
$$

对 basis cover group $c$，定义：

$$
SNR_c=\frac{\left(\sum_{k\in c}\mu_k\right)^2}{\sum_{k\in c}\sigma_k^2/(b-1)+\epsilon}.
$$

但新增重点不是 $SNR_c$ 本身，而是 cover specialization：

$$
CoverPurity(c)=\frac{\|\mathbb{E}_{i\in B_c} g_i\|^2}{\mathbb{E}_{i\in B_c}\|g_i\|^2+\epsilon}.
$$

cover churn：

$$
CoverChurn_t=1-\frac{|A_c(t)\cap A_c(t-1)|}{|A_c(t)\cup A_c(t-1)|+\epsilon}.
$$

signal-to-cover co-location score：

$$
STCScore =
\operatorname{median}_c(SNR_c \cdot CoverPurity(c) \cdot (1-CoverChurn_c)).
$$

## 9.3 必须记录

输出：`v139_signal_to_cover_audit.csv`

字段：

```text
dataset
seed
task
loss_interface
method
basis_family
candidate
parameter_snr_active_fraction
group_snr_active_fraction
basis_snr_active_fraction
signal_retention_group
cos_group_vs_param
cover_purity_mean
cover_purity_p10
cover_churn_mean
cover_specialization_entropy
cover_load_gini
signal_to_cover_score
false_drop_fraction
false_keep_fraction
readout_signal_mass
numerator_signal_mass
denominator_signal_mass
projection_signal_mass
residual_signal_mass
LineC_CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
CEp99
NLL
ECE
```

## 9.4 Gate

Exploratory cover gate：

$$
\operatorname{median}(signal\_retention\_group)\ge0.70,
$$

$$
\operatorname{median}(cos\_group\_vs\_param)\ge0.60,
$$

$$
\operatorname{median}(cover\_purity)\ge0.20,
$$

$$
\operatorname{mean}(cover\_churn)\le0.50.
$$

如果 K0 signal retention 过但 cover purity / churn 失败，说明问题不是 signal 丢失，而是 cover formation 失败。

---

# 10. Line K1：Rational Signal-to-Cover Mechanism

## 10.1 目标

把 Rational 从固定 group-SNR update 改成 dynamic cover formation。

不再继续：

```text
K1-K7 小修；
K4F freeze-cluster；
cover threshold / tau 网格；
G6-G9 MLP trust 小修。
```

## 10.2 新 candidate families

### K8：Rational Cover Assignment by Gradient Clustering

根据 current train batch 的 per-example gradient signature，把样本映射到 Rational groups：

$$
a_i = \operatorname{cluster}(g_i^{RAT}).
$$

更新时只放大与该 cluster 一致的 group。

候选：

```text
K8-RAT-GradientClusterCover-k4
K8-RAT-GradientClusterCover-k8
```

### K9：Rational Cover Split-Merge

如果某 group cover load 过高且 internal gradient conflict 高，则 split；如果某 group dead 且 low SNR，则 merge / freeze。

$$
Conflict(c)=1-CoverPurity(c).
$$

规则：

```text
if load(c) high and Conflict(c) high:
  split group c into c1/c2 by gradient direction
if load(c) low and SNR(c) low:
  freeze or merge
```

候选：

```text
K9-RAT-CoverSplitMerge-lite
K9-RAT-CoverSplitMerge-noMerge
```

### K10：Parameter-SNR then Cover Consolidation

Phase 1 用 parameter-SNR 保留 generic signal；Phase 2 再把 high-SNR parameters 聚合进 stable cover。

```text
Phase 1: parameter-SNR, weak cover guard
Phase 2: group consolidation, medium cover guard
Phase 3: fixed-point consolidation, stronger cover guard
```

候选：

```text
K10-RAT-ParamSNRThenCover-3phase
K10-RAT-ParamSNRThenCover-slowConsolidate
```

### K11：Readout-Basis Decoupled SNR

把 readout signal 和 basis shape signal 分开：

```text
readout path:
  follows generic AdamW/SNR blend
basis shape path:
  follows cover-stability SNR
```

候选：

```text
K11-RAT-ReadoutBasisDecoupledSNR
K11-RAT-ReadoutFirstBasisConsolidate
```

### K12：Reservoir-Aware Cover Growth without LineC Target

不使用 LineC hard target。只用 per-example gradient variance 和 basis telemetry 判断某些 groups 是否应扩大 plasticity。LineC 只 audit。

候选：

```text
K12-RAT-VarianceReservoirProxyCoverGrowth
K12-RAT-LowVarianceSignalCoverGrowth
```

## 10.3 执行设置

Synthetic official：

```text
tasks = X1..X7
seeds = 0,1,2
loss_interfaces = CE,Brier
train_steps = 200
batch_size = 32
methods = RAT-AdamW, K8..K12
controls = AdamW, NoOp, RandomMatchedNorm, AdamWParallel, ParameterSNR-only, old K4/K7 best
```

## 10.4 记录

输出：`v139_rat_signal_to_cover_training.csv`

字段：

```text
task
seed
loss_interface
method
phase
step
source_vs_best_control
source_vs_adamw
AUC_time_ratio
CEp99_delta
NLL_delta
ECE_delta
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
LineC_majority_pass
LineC_all_pass
cover_purity
cover_churn
cover_load_gini
cover_split_count
cover_merge_count
cover_freeze_count
group_diversity
den_p01
r_prime_p99
r_double_prime_p99
active_fraction
update_norm
control_gap
pass_s3
pass_s4
```

## 10.5 Gate

KAN S3 synthetic gate：

```text
>= 5 / 7 synthetic task families pass；
within each passed family, >= 2 / 3 seeds or >=2 loss interfaces pass；
source_vs_best_control >= 0.005；
AUC_time_ratio <= 1.0；
CEp99_delta <= 0.05；
NLL_delta <= 0.02；
ECE_delta <= 0.02；
LineC majority pass；
```

KAN S4 cover gate：

```text
S3 pass；
LineC all-pass or >=4/5 multisketch pass；
cover_churn_mean <= 0.35；
cover_purity_mean improves over old K1-K7 by >= 20%；
NoiseSignalLeak non-harm；
RealSignalReservoirRatio non-harm；
```

只有 S3 或 S4 过，才允许 real short-run triage。

---

# 11. Line D：Non-RAT substrate vertical slices

## 11.1 目标

Non-RAT 不进入 functional proof，除非先成为 substrate-health candidate。

## 11.2 优先级

```text
D1 Fourier exact no-materialize vertical slice。
D2 RBF compact local exact vertical slice。
D3 Wavelet hat local exact vertical slice。
D4 Chebyshev degree-energy exact vertical slice。
```

## 11.3 必须记录

输出：`v139_nonrat_exact_substrate.csv`

字段：

```text
family
candidate
exact_kernel_implemented
materializes_basis
materializes_derivative
materializes_readout_grad
raw_memory_ratio
incremental_memory_ratio
step_ratio
workspace_gate_pass
hardening_executed
LineC_executed
mean_delta_vs_MLP
worst_delta_vs_MLP
AUC_time_ratio
LineC_pass_rate
substrate_health_pass
```

## 11.4 Gate

Substrate-health gate：

$$
raw\_memory\_ratio \le 1.15,
$$

$$
incremental\_memory\_ratio \le 2.00,
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
AUCtime\_ratio \le 2.0,
$$

$$
LineC\_pass\_rate \ge 0.30.
$$

如果 exact kernel still materializes basis / derivative / readout grad，则不得写成 exact repair success。

---

# 12. Line C：Manifold-channel audit

Line C 只做审计，不做方向源。

## 12.1 必须记录

输出：`v139_linec_audit.csv`

字段：

```text
dataset_or_task
seed
loss_interface
method
CouplingR2
CouplingCorr
NoiseSignalLeak
RealSignalReservoirRatio
signal_mass_topk
reservoir_fraction
CEp99
NLL
ECE
Brier
margin_p10
logit_norm
logit_drift
```

## 12.2 Gate

Functional candidate 必须满足：

```text
CouplingR2_delta >= 0 or no worse than control；
NoiseSignalLeak_delta <= 0.01；
RealSignalReservoirRatio_delta <= 0.01；
CEp99/NLL/ECE non-harm；
```

但 LineC 不能用于生成方向。

---

# 13. 必须生成的可视化

```text
fig_v139_progress_by_line.svg
fig_v139_signal_retention_vs_cover_purity.svg
fig_v139_cover_churn_by_method.svg
fig_v139_k1_old_vs_k8_k12_synthetic_heatmap.svg
fig_v139_linec_task_family_heatmap.svg
fig_v139_mlp_control_monitor.svg
fig_v139_nonrat_workspace_waterfall.svg
fig_v139_rational_cover_load_gini.svg
fig_v139_synthetic_pass_matrix.svg
fig_v139_failure_taxonomy.svg
```

---

# 14. 失败后 Codex 必须优先尝试什么

## Case A：K0 signal retention 过，但 cover purity / churn fail

不要继续调 SNR tau。先尝试：

```text
1. K8 gradient clustering cover；
2. K9 split/merge；
3. K10 delayed consolidation；
4. K11 readout-basis decoupling。
```

## Case B：K8/K9 让 cover purity 上升，但 source_vs_control 不够

不要降低 source gate。先尝试：

```text
1. 增加 Phase 1 parameter-SNR plasticity；
2. 延后 cover consolidation；
3. readout follows AdamW/SNR, basis follows cover-SNR；
4. 检查 cover 是否过早冻结。
```

## Case C：source 好，但 LineC / reservoir / tail fail

不要用 LineC target 生成方向。先尝试：

```text
1. 以 per-example gradient variance 做 noise proxy；
2. 减少 low-purity cover 的 update；
3. 加强 Phase 3 consolidation；
4. 检查 group diversity 和 den/r'/r'' telemetry。
```

## Case D：MLP G line 又出现局部 positive

不要写成 KAN success。只记录：

```text
Generic optimizer diagnostic positive；
KAN-specific advantage still unproven。
```

不再扩 MLP small repair，除非 KAN S3 已经打开，需要 matched generic baseline。

## Case E：Non-RAT workspace fail

不要进入 functional proof。先做：

```text
1. exact no-materialize kernel；
2. derivative recompute；
3. readout grad recompute；
4. optimizer state delayed allocation；
5. lifecycle waterfall。
```

## Case F：Non-RAT task-health collapse after workspace pass

不要按数据集调参。先做：

```text
Fourier: high-frequency quarantine / band-energy cap；
Chebyshev: degree-energy cap / late high-degree enable；
RBF: center occupancy / width guard；
Wavelet: support overlap / scale guard。
```

---

# 15. Route 定义

```text
R0-ProvenanceViolation:
  forbidden information or missing artifact。

R1-GenericSNROptimizerNoGo:
  MLP generic optimizer not confirmed, and KAN S3/S4 not reached。

R2-SignalRetentionPassCoverFormationFail:
  K0 retention/cos pass, but cover purity/churn fail。

R3-CoverFormationNoTaskAdvantage:
  cover improves, but source/control gate fails。

R4-CoverTaskPositiveLineCFail:
  source/control positive, but LineC/reservoir/tail fails。

R5-KANSpecificS3Opened:
  KAN S3 synthetic >=5/7 pass, real short-run allowed。

R6-KANSpecificS4Opened:
  S3 pass plus robust LineC/cover pass。

R7-NonRATSubstrateOpened:
  at least one Non-RAT family passes substrate-health gate。

R8-S5OfficialCandidate:
  real short-run 3x3 task/AUC/tail/LineC/control pass。
```

---

# 16. 最终判断

v13.8 不是完全没进展。它证明：

```text
1. MLP generic SNR 的 5-seed positive 不能扩展成稳定 10-seed claim；
2. KAN basis/group lift 并没有完全丢失 parameter-level SNR signal；
3. KAN blocker 是 signal 没有稳定形成 basis cover；
4. Non-RAT 仍然不能进入 functional proof；
5. 继续 MLP SNR 小修和 K1-K7 小修已经没有意义。
```

下一步的核心不是继续调 SNR，而是：

$$
\boxed{
\text{把 coherent population-risk signal 映射为稳定的 KAN basis cover。}
}
$$

如果 v13.9 仍然不能让 Rational 在 synthetic >=5/7 上达到 S3，那么我们需要更强的 substrate/base architecture reset，而不是继续 functional update 搜索。
