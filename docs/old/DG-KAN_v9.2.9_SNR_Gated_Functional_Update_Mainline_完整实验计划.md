# DG-KAN v9.2.9 SNR-Gated Functional Update Mainline 完整实验计划

> 本计划基于 DG-KAN 总计划、v9.2.7 真实复盘，以及论文 **A Theory of Generalization in Deep Learning** 的理论启发制定。  
> 本计划的核心判断是：**我们已经可以开始认真探索最重要的 functional update，但必须以受控、可审计、task-safe、population-risk-safe 的方式打开。**  
> 它不是 teacher，不是 distillation，不是 loss modification，不是 label smoothing，不是 class-weight，也不是 sampler。  
> 它必须是 **update rule**，并且必须在当前 FC-PureKAN LQ near-pass base 上证明它能带来几何、校准、鲁棒性或 KMNIST miss-row 的真实改善，同时不破坏 CE-only AdamW base 与系统 gate。

---

## 0. 我们根据总计划进行到哪一步了

### 0.1 总计划的阶段定位

DG-KAN 总计划定义的终极目标是：

$$
\boxed{
\text{Next-Gen Beyond-MLP}
=
\text{Clean FullEdge PureKAN}
+
\text{Graph-Free Manual Training}
+
\text{Kernel-Native Efficiency}
+
\text{Task-Safe Functional Update}
+
\text{External Fair Advantage}
+
\text{Scalable Conv/Transformer Extension}.
}
$$

截至 v9.2.7，我们已经推进到以下位置：

```text
Clean FullEdge / FC-PureKAN:
  已经出现 LQ-t2-h256 这个 strict FC-PureKAN near-pass primitive。

Graph-free manual training:
  LQ path 继续遵守 manual forward / manual backward / manual update，不使用 PyTorch loss.backward graph。

Kernel-native efficiency:
  LQ 在 official compact/recompute P4 gate 下通过 forward/backward/step/memory。

Task trainability:
  P5 robust near-pass 已成立，但 full-pass 未达成。
  v9.2.7 robust 10-seed = 24/30 near-pass，macro delta = -0.0057666699。

Functional update:
  还没有正式打开。
  现在正处于可以打开 gated functional diagnostic 的阶段。

External fair / Conv / Former:
  仍然 deferred。
```

因此，当前不是早期 “basis-source / kernel-native 还没闭合” 阶段，也不是 “PureKAN 已经 full-pass / Beyond-MLP 成功” 阶段。最准确定位是：

$$
\boxed{
\text{FC-PureKAN LQ 已经 P4 pass + robust near-pass；现在可以启动 task-safe functional update 主线。}
}
$$

### 0.2 是否可以开始探索最重要的 functional update

可以，但必须明确边界。

可以开始的原因：

```text
1. LQ 已通过 strict FC-PureKAN equivalence audit；
2. LQ 已通过 P4 official compact/recompute system gate；
3. LQ 已达到 P5 robust near-pass；
4. 当前 blocker 已从系统实现转为 KMNIST miss rows、margin、lift conditioning、basis usage 与泛化稳定性；
5. functional update 正是总计划中的核心优势来源。
```

不能做的事情：

```text
1. 不能用 functional update 代替 AdamW-only full-pass；
2. 不能把 functional update 写进 loss；
3. 不能引入 teacher / self-teacher / distillation；
4. 不能使用 label smoothing / focal / margin loss；
5. 不能使用 sampler / class weight；
6. 不能用 functional update 结果掩盖 AdamW-only base 的真实状态；
7. 不能提前打开 PureKANConv / PureKANFormer。
```

本计划将 functional update 定义为：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW-equivalent}}
+
\lambda_f \Delta\theta_{\text{functional-safe}}.
$$

其中：

$$
\Delta\theta_{\text{functional-safe}}
=
\Pi_{\text{task-safe}}
\left(
q_{\text{SNR}}
\odot
d_{\text{geometry}}
\right).
$$

这里 $d_{\text{geometry}}$ 是我们已有的 curvature / high-curvature / role-wise functional direction；$q_{\text{SNR}}$ 是受论文启发的 population-risk-safe gate；$\Pi_{\text{task-safe}}$ 是不伤 CE descent 的投影或 guard。

---

## 1. 论文启发与我们要做到更好的地方

### 1.1 论文给出的核心理论启发

论文把训练输出空间分成 signal channel 和 reservoir。其核心对象是 cumulative dissipation Gramian：

$$
W_S(s,T)
=
\int_s^T
P_g(\tau,s)^T
K_{SS}(\tau)
P_g(\tau,s)
d\tau.
$$

其中：

```text
range(W_S):
  signal channel，训练真正耗散 loss 且会影响 test 的方向。

ker(W_S):
  reservoir，训练残差可能存在，但对 test prediction 不可见。
```

论文还把 minibatch gradient 分解为：

$$
\hat g_k=\mu_k+\xi_k,
$$

其中 $\mu_k$ 是 coherent drift，$\xi_k$ 是 centered fluctuation。文章的 drift-diffusion 观点是：

```text
coherent population signal:
  以 O(T) 累积；

minibatch noise:
  以 O(sqrt(eta T / b)) 扩散。
```

最后，它推导出一个 practical SNR gate。对 diagonal preconditioner，参数 $k$ 只有在下面条件成立时才应该被更新：

$$
\mu_k^2>\frac{\sigma_k^2}{b-1}.
$$

这给我们的启发是：functional update 不应该只是“让曲率更低”，而应该只允许 **signal-dominant** 的几何 correction 进入参数更新。

### 1.2 我们不能直接照搬论文

这篇文章是重要参考，但我们要做到更好，原因有四点。

第一，论文的 practical rule 主要是 architecture-agnostic per-parameter SNR gate；而我们的模型是 FC-PureKAN，有清楚的 edge-basis roles：

```text
identity lift channel
quadratic T2 channel
optional T3 / Legendre channel
output coefficient
basis normalization / scale
```

因此我们不能只做 scalar per-parameter mask。我们应该做：

```text
role-aware SNR
basis-channel-aware SNR
layer-aware SNR
dataset/miss-row diagnostic SNR
```

第二，论文的 gate 可以作用到 optimizer 本体；我们当前不应先 gate AdamW 主更新。因为 LQ AdamW-only 已经 near-pass，直接 gate AdamW 可能破坏 task trainability。更稳妥的第一阶段是：

$$
\Delta\theta_{\text{AdamW}}
\quad\text{保持不动，}
$$

只对 functional correction 做 SNR gate：

$$
\Delta\theta_{\text{functional-safe}}
=
q_{\text{SNR}}
\odot
\Delta\theta_{\text{functional}}.
$$

第三，论文主要强调 population-risk-safe signal/noise；我们还必须保留 DG-KAN 的几何目标：

```text
curvature
Jacobian norm
local Lipschitz
basis usage entropy
lift condition number
KMNIST miss-row margin
```

所以我们的 functional update 应该是：

$$
\boxed{
\text{population-risk-safe}
\cap
\text{geometry-aware}
\cap
\text{task-safe}
\cap
\text{PureKAN-role-aware}.
}
$$

第四，论文不是 KAN-specific。它不能证明 PureKAN 架构优于 MLP。我们要额外加：

```text
strong MLP baseline challenge
QuadraticFeatureMLP diagnostic
NoOp / Random / ShuffledSNR controls
system overhead gate
```

---

## 2. v9.2.9 整体目标

v9.2.9 的整体目标是：

$$
\boxed{
\text{在 LQ FC-PureKAN near-pass base 上，证明或证伪 SNR-gated functional update 的独特优势。}
}
$$

这不是单纯追 final accuracy。它要回答五个问题。

### Q1：functional update 能否安全打开

Functional update 必须满足：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

如果它提高 curvature 但伤 task，就失败。

### Q2：functional update 能否改善几何

至少满足：

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

或者：

$$
LocalLipschitz_{\text{functional}}\leq0.90LocalLipschitz_{\text{AdamW}}.
$$

### Q3：functional update 能否改善泛化相关指标

至少改善一个：

$$
ECE_{\text{functional}}\leq ECE_{\text{AdamW}},
$$

$$
NLL_{\text{functional}}\leq NLL_{\text{AdamW}},
$$

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}}.
$$

### Q4：functional update 能否解决 KMNIST miss rows

v9.2.7 的 miss rows 全部来自 KMNIST seeds `0/3/4/5/7/9`。v9.2.9 要测：

$$
\Delta Acc_{\text{KMNIST,functional}}
-
\Delta Acc_{\text{KMNIST,AdamW}}
\geq0.005.
$$

更强目标：

$$
\Delta Acc_{\text{macro,functional}}\geq0.
$$

### Q5：functional update 的收益是否超过 NoOp / Random / ShuffledSNR controls

Functional update 必须超过三类 control：

```text
NoOpMatchedOverhead:
  同样频率、同样系统开销，但不改变参数。

RandomDirection:
  同样 norm / role / frequency，但方向随机。

ShuffledSNRMask:
  使用相同 active ratio，但打乱 role/channel mask。
```

如果 functional 只赢 AdamW，不赢 controls，不能说明 functional causality。

---

## 3. 硬约束

### 3.1 Clean training contract

所有 official candidate 必须满足：

```text
loss_type = CE
label_smoothing = 0
external_teacher_used = 0
self_teacher_used = 0
teacher_logits_used = 0
distillation_used = 0
geometry_loss_used = 0
sampler_changed = 0
class_weight_used = 0
cpu_offload_used = 0
uses_loss_backward = 0
fake_data_used = 0
proxy_row_used = 0
```

训练目标保持：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

Functional update 是 update rule：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW-equivalent}}
+
\lambda_f\Delta\theta_{\text{functional-safe}}.
$$

不允许：

$$
L=CE+\lambda L_{\text{geo}}.
$$

### 3.2 FC-PureKAN contract

LQ 必须继续满足：

$$
h_j=\sum_i c^0_{ij}B_0(x_i),
$$

其中：

$$
B_0(x)=x.
$$

第二层：

$$
y_c=\sum_j c^1_{jc,0}h_j+\sum_j c^1_{jc,2}T_2(h_j).
$$

Functional update 不得引入 non-edge trainable params。所有 update 必须作用于已有 FC-PureKAN edge-owned parameters。

### 3.3 Deferred architecture rule

本轮继续保持：

```text
PureKANConv_status = deferred_until_FC_PureKAN_P5_fullpass_or_user_unlock
PureKANFormer_status = deferred_until_FC_PureKAN_P5_fullpass_or_user_unlock
```

不得执行 Conv / Former measured candidate。

### 3.4 Official baseline rule

Official 主比较仍是 same-parameter MLP-match：

$$
\left|
\frac{Params_{\text{KAN}}-Params_{\text{MLP-match}}}
{Params_{\text{MLP-match}}}
\right|
\leq0.05.
$$

同时保留强基线诊断：

```text
QuadraticFeatureMLP
same-shape MLP
hidden-bracket MLP
```

但这些 diagnostic 不能替代 official route。

---

## 4. Functional update 的设计

### 4.1 SNR estimator

对一个 batch 拆成 $M$ 个 microbatches，得到每个 microbatch 的 manual gradient：

$$
g^{(m)}_r,\quad m=1,\dots,M.
$$

其中 $r$ 是 role，例如：

```text
lift_identity
quadratic_coeff
output_linear
basis_scale
```

role-wise mean：

$$
\mu_r=\frac{1}{M}\sum_{m=1}^{M}g^{(m)}_r.
$$

role-wise variance：

$$
\sigma_r^2=\frac{1}{M-1}\sum_{m=1}^{M}\|g^{(m)}_r-\mu_r\|^2.
$$

role-wise SNR：

$$
SNR_r=
\frac{\|\mu_r\|^2}{\sigma_r^2/(M-1)+\epsilon}.
$$

per-parameter SNR：

$$
SNR_k=
\frac{\mu_k^2}{\sigma_k^2/(M-1)+\epsilon}.
$$

hard gate：

$$
q_k=\mathbb{1}(SNR_k>\tau).
$$

smooth gate：

$$
q_k=
\sigma
\left(
\frac{\log(SNR_k+\epsilon)-\log(\tau)}{T}
\right).
$$

### 4.2 Geometry direction

Functional direction 不直接由 SNR 给出。SNR 只是 gate。方向来自 DG-KAN 原有几何机制：

```text
Quadratic coefficient curvature direction
High-curvature lift direction
Basis usage balancing direction
Local Lipschitz reduction direction
Role-wise functional correction
```

记为：

$$
d_{\text{geometry}}.
$$

### 4.3 Task-safe projection

给定 task gradient $g$ 与 proposed functional step $\delta_f$，如果：

$$
g^T\delta_f>0,
$$

说明该 functional step 在一阶上增加 CE，应投影掉伤 task 分量：

$$
\delta_{\text{safe}}
=
\delta_f
-
\frac{\max(0,g^T\delta_f)}{\|g\|^2+\epsilon}g.
$$

然后：

$$
\Delta\theta_{\text{functional-safe}}
=
q_{\text{SNR}}\odot\delta_{\text{safe}}.
$$

### 4.4 Functional update 事件触发

Functional 不应每步无脑触发。触发条件：

```text
event_stride = 8 or 16
bad_margin_event = CEp99 high or margin_p10 low
curvature_event = curvature ratio high
SNR_event = active fraction within target range
```

事件触发必须记录：

```text
event_id
step
trigger_type
role_active_fraction
snr_mean_by_role
snr_p10_by_role
snr_p90_by_role
functional_norm
task_projection_norm_removed
holdout_loss_before
holdout_loss_after
```

---

## 5. Candidate 设计

### 5.1 Base candidates

```text
B0-MLP-match:
  official same-parameter MLP baseline.

QF-QuadraticFeatureMLP:
  diagnostic strong baseline.

LQ0-LQ-t2-h256-AdamW:
  current strict FC-PureKAN near-pass base.

LQ1-LQ-t2-h256-fanin-output-scale:
  v9.2.7 best repair reference.
```

### 5.2 Functional candidates

```text
F0-AdamWOnly:
  no functional update; official base reference.

F1-CurrentQuadraticCoeffFunctional:
  current quadratic coefficient functional direction without SNR gate.

F2-CurrentHighCurvatureLiftFunctional:
  current high-curvature lift functional direction without SNR gate.

F3-SNRGatedQuadraticCoeffFunctional:
  q_SNR gates quadratic coefficient functional direction.

F4-SNRGatedHighCurvatureLiftFunctional:
  q_SNR gates lift high-curvature direction.

F5-RoleSNRGatedFunctional:
  role-level SNR gate, with per-role active/inactive decisions.

F6-ParamSNRGatedFunctional:
  per-parameter SNR gate, highest granularity.

F7-SNRGatedAdamWDiagnostic:
  applies SNR gate to AdamW update itself; diagnostic only, not official replacement.

F8-SNRGatedFunctionalPlusTaskProjection:
  F3/F4 plus task-safe projection.

F9-SNRGatedFunctionalPlusHoldoutGuard:
  F8 plus pre/post holdout guard.

F10-SNRGatedFunctionalEventController:
  event-triggered version using margin/curvature/SNR triggers.
```

### 5.3 Control candidates

```text
C0-NoOpMatchedOverhead:
  same event frequency and measurement overhead, no parameter change.

C1-RandomDirectionMatchedNorm:
  random direction with same norm and role distribution.

C2-ShuffledSNRMask:
  same active ratio as SNR gate, but shuffled across parameters/channels.

C3-InvertedSNRMask:
  updates low-SNR directions; should perform worse.

C4-GeometryOnlyNoSNR:
  geometry direction without SNR.

C5-SNROnlyNoGeometry:
  SNR gate applied to zero/AdamW diagnostic direction, to separate gate from geometry.
```

---

## 6. 实验阶段

## P0：current stage recap and contract audit

### 目标

确认当前已经满足 functional re-entry 前提：LQ strict FC-PureKAN、P4 pass、P5 robust near-pass；同时确认 no-teacher/no-loss/no-fake/no-proxy contract。

### 必须记录

```text
candidate_id
candidate_family
purekan_equivalence_pass
p4_pass
p5_near_pass
p5_full_pass
loss_type
label_smoothing
teacher_used
distillation_used
geometry_loss_used
uses_loss_backward
fake_data_used
proxy_row_used
cpu_offload_used
purekanconv_status
purekanformer_status
```

### 判断标准

Functional diagnostic can open only if：

```text
purekan_equivalence_pass = 1
p4_pass = 1
p5_near_pass = 1
loss_type = CE
teacher_used = 0
geometry_loss_used = 0
uses_loss_backward = 0
```

### 可视化

```text
p0_functional_open_gate_dashboard.svg
p0_contract_heatmap.svg
p0_route_position_diagram.svg
```

---

## P1：SNR instrumentation and estimator validation

### 目标

验证 ghost-batch / microbatch SNR 估计可用、稳定、系统开销可接受。P1 不做 functional update，只测 SNR。

### 设置

```text
candidate = LQ0 and LQ1
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
microbatches_per_batch = 4,8
batch_size = 128
functional_update = off
```

### 必须记录

```text
dataset
seed
step
role
microbatch_count
grad_mean_norm
grad_var_norm
SNR_mean
SNR_p10
SNR_p50
SNR_p90
SNR_active_fraction_tau1
SNR_active_fraction_tau2
per_param_SNR_histogram
basis_channel_SNR
lift_identity_SNR
quadratic_coeff_SNR
output_linear_SNR
SNR_compute_time_ms
SNR_memory_MB
```

### 判断标准

Estimator stability：

$$
CV(SNR_{\text{role}})\leq0.50
$$

over repeated batches for major roles.

Overhead gate：

$$
T_{\text{SNR-compute}}\leq0.20T_{\text{step}}.
$$

Active fraction sanity：

$$
0.05\leq ActiveFraction\leq0.80.
$$

如果 active fraction 接近 0 或 1，说明 gate 无信息量，需调整 $\tau$ 或 temperature。

### 可视化

```text
p1_snr_by_role_violin.svg
p1_snr_active_fraction_by_dataset.svg
p1_snr_overhead_bar.svg
p1_snr_vs_margin_scatter.svg
p1_basis_channel_snr_heatmap.svg
```

---

## P2：one-step population-risk and task-safety audit

### 目标

验证论文启发的 SNR gate 是否真的预测 holdout / population-risk one-step improvement，并验证 functional direction 不伤 CE。

### 方法

每个 batch 切分：

```text
train microbatch:
  用于计算 AdamW gradient、functional direction、SNR gate。

exchangeability holdout microbatch:
  不参与 update，用于 one-step before/after CE evaluation。
```

不使用 validation/test 选择 candidate。holdout 仅用于 one-step causality audit。

### 必须记录

```text
candidate_id
dataset
seed
step
functional_mode
snr_gate_type
active_fraction
predicted_population_improvement
actual_holdout_loss_before
actual_holdout_loss_after
actual_holdout_delta
task_gradient_dot_functional_step
projection_removed_norm
functional_step_norm
adamw_step_norm
cos_functional_task
bad_step
CEp99_before
CEp99_after
margin_p10_before
margin_p10_after
```

### 判断标准

Prediction validity：

$$
Corr(\widehat{\Delta R}_{\text{population}},\Delta L_{\text{holdout}})\geq0.30.
$$

Task safety：

$$
BadStepRate\leq0.05.
$$

Holdout non-harm：

$$
\frac{\#\{\Delta L_{\text{holdout}}\leq0\}}{\#\text{events}}\geq0.70.
$$

SNR gate usefulness：

SNR-gated functional must beat geometry-only no-SNR on bad step rate or holdout delta:

$$
BadStepRate_{\text{SNR}}\leq BadStepRate_{\text{NoSNR}},
$$

or:

$$
\Delta L_{\text{holdout,SNR}}\leq\Delta L_{\text{holdout,NoSNR}}.
$$

### 可视化

```text
p2_predicted_vs_actual_holdout_delta.svg
p2_bad_step_rate_by_candidate.svg
p2_projection_removed_norm.svg
p2_margin_before_after_one_step.svg
p2_snr_gate_vs_no_snr.svg
```

---

## P3：short-run functional safety and mechanism audit

### 目标

用 20/50-step short runs 判断 functional update 是否在短期内安全，并观察是否改善 CE tail、margin、curvature、basis usage。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
steps = 20,50
candidates = F0,F1,F2,F3,F4,F5,F8,F9,C0,C1,C2,C3,C4
```

### 必须记录

```text
candidate_id
dataset
seed
step
train_loss
holdout_loss
val_acc_proxy
CE_p50
CE_p90
CE_p99
margin_p10
ECE_proxy
NLL_proxy
curvature
jacobian_norm
local_lipschitz
basis_usage_entropy
lift_condition_number
effective_rank
functional_event_count
active_fraction_by_role
snr_mean_by_role
step_ratio
memory_ratio
```

### 判断标准

Short-run safety：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005
$$

on proxy validation after 50 steps.

Geometry improvement：

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}
$$

or:

$$
LocalLipschitz_{\text{functional}}\leq0.90LocalLipschitz_{\text{AdamW}}.
$$

Tail improvement：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}}
$$

or:

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}}.
$$

Control causality：

Functional candidate must outperform NoOp and Random on at least one mechanism metric without task harm.

### 可视化

```text
p3_short_run_loss_curve.svg
p3_curvature_ratio_bar.svg
p3_ce_tail_by_candidate.svg
p3_margin_p10_by_candidate.svg
p3_basis_usage_entropy_trace.svg
p3_task_geometry_pareto.svg
```

---

## P4：full 10-seed functional re-entry

### 目标

在完整 P5 trainability setting 下比较 AdamW-only 和 functional update。P4 是本计划的核心。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
epochs = 20
baseline = MLP-match
base = LQ0 / LQ1
functional_candidates = selected from P3 survivors
controls = NoOpMatchedOverhead, RandomDirection, ShuffledSNRMask
```

### 必须记录

```text
candidate_id
functional_mode
dataset
seed
val_acc
test_acc
train_acc
val_loss
test_loss
delta_vs_MLP_match
delta_vs_AdamW_LQ
near_pass
full_pass
CE_p50
CE_p90
CE_p99
margin_p10
wrong_confidence_p95
ECE
NLL
curvature_ratio
jacobian_norm_ratio
local_lipschitz_ratio
basis_usage_entropy
lift_condition_number
effective_rank
functional_event_count
active_fraction_by_role
SNR_mean_by_role
SNR_active_fraction_by_role
step_ratio
memory_ratio
functional_update_time_ratio
```

### 判断标准

Task safety：

$$
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005.
$$

Functional macro improvement：

$$
\Delta Acc_{\text{macro,functional}}
-
\Delta Acc_{\text{macro,AdamW}}
\geq0.003.
$$

KMNIST miss-row repair：

$$
\Delta Acc_{\text{KMNIST,functional}}
-
\Delta Acc_{\text{KMNIST,AdamW}}
\geq0.005.
$$

Geometry pass：

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

Calibration / tail pass：

At least one：

$$
ECE_{\text{functional}}\leq ECE_{\text{AdamW}},
$$

$$
NLL_{\text{functional}}\leq NLL_{\text{AdamW}},
$$

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}}.
$$

Control pass：

Functional must beat NoOp and Random on geometry/tail while not losing task:

$$
Curvature_{\text{functional}}<Curvature_{\text{NoOp}},
$$

$$
Curvature_{\text{functional}}<Curvature_{\text{Random}},
$$

and:

$$
Acc_{\text{functional}}\geq Acc_{\text{Random}}.
$$

System pass：

$$
StepRatio_{\text{functional}}\leq1.50,
$$

$$
MemoryRatio_{\text{functional}}\leq1.05.
$$

### 可视化

```text
p4_macro_delta_vs_adamw.svg
p4_kmnist_miss_recovery.svg
p4_seedwise_win_matrix.svg
p4_functional_task_geometry_pareto.svg
p4_curvature_vs_accuracy.svg
p4_ece_nll_bar.svg
p4_snr_active_fraction_heatmap.svg
p4_controls_comparison.svg
```

---

## P5：strong baseline and mechanism challenge

### 目标

防止把 quadratic lifted feature 的通用优势误判为 KAN functional advantage。Functional 必须和 strong baselines 对比。

### Baselines

```text
B0-MLP-match
B1-same-shape MLP
B2-hidden-bracket MLP
B3-QuadraticFeatureMLP
B4-LQ-forbidden-MLP-hidden diagnostic
```

### 必须记录

```text
baseline_id
params
params_ratio
flops_ratio
forward_ratio
backward_ratio
step_ratio
memory_ratio
dataset
seed
acc
ECE
NLL
CEp99
margin_p10
curvature
local_lipschitz
```

### 判断标准

KAN functional not explained by quadratic baseline if：

$$
Acc_{\text{LQ-functional}}\geq Acc_{\text{QuadraticFeatureMLP}}-0.005,
$$

and at least one geometry/calibration metric is better:

$$
Curvature_{\text{LQ-functional}}<Curvature_{\text{QuadraticFeatureMLP}},
$$

or:

$$
ECE_{\text{LQ-functional}}\leq ECE_{\text{QuadraticFeatureMLP}}.
$$

If this fails, route must record：

```text
quadratic_feature_family_explains_task_gain = 1
```

### 可视化

```text
p5_lq_functional_vs_quadratic_mlp.svg
p5_strong_baseline_system_table.md
p5_task_geometry_baseline_pareto.svg
```

---

## P6：robustness and noisy-signal diagnostic

### 目标

论文强调 signal/noise separation。我们要验证 SNR-gated functional update 是否在 noisy labels / input perturbation 下优于 AdamW-only。

### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
candidates = AdamW-only, best functional, NoOp, Random
```

### 必须记录

```text
noise_type
noise_level
dataset
seed
candidate
acc_clean_test
acc_noisy_test
acc_drop
CEp99
margin_p10
ECE
NLL
curvature
SNR_active_fraction
signal_channel_proxy
noise_channel_proxy
bad_step_rate
```

### 判断标准

Robustness useful：

$$
AccDrop_{\text{functional}}\leq AccDrop_{\text{AdamW}}.
$$

Noise-tail pass：

$$
CEp99_{\text{functional}}\leq CEp99_{\text{AdamW}}.
$$

SNR gate relevance：

As noise increases, low-SNR active fraction should decrease:

$$
ActiveFraction_{\text{noise}=0.20}
<
ActiveFraction_{\text{noise}=0.05}.
$$

### 可视化

```text
p6_noise_robustness_curve.svg
p6_snr_active_fraction_vs_noise.svg
p6_noise_ce_tail.svg
p6_signal_noise_channel_proxy.svg
```

---

## P7：route decision and external-fair gate

### 目标

根据 P0-P6 判断 functional update 是否成为正式主线，以及是否允许进入 external fair validation。

### Route cases

```text
R1-FunctionalPrimaryAdvantage:
  functional improves task or KMNIST miss rows, improves geometry/tail, passes controls and system.

R2-FunctionalGeometryOnly:
  functional improves curvature/ECE/NLL but task unchanged; still valuable but not full advantage.

R3-FunctionalUnsafe:
  functional improves geometry but hurts task beyond tolerance.

R4-SNRGateNotUseful:
  SNR-gated variants do not outperform NoSNR/NoOp/Random controls.

R5-QuadraticBaselineExplainsGain:
  LQ functional is not better than QuadraticFeatureMLP diagnostic.

R6-FunctionalSystemOverheadFail:
  functional breaks step/memory gate.

R7-AdamWOnlyStillBest:
  AdamW-only remains best; functional branch should be redesigned.

R8-ExternalFairReady:
  functional passes task/geometry/system and strong baseline challenge; open external fair validation.
```

### route_decision.json 必须记录

```text
route
best_functional_candidate
base_candidate
p4_pass
p5_near_pass
functional_task_safe
functional_geometry_pass
functional_calibration_pass
functional_kmnist_repair_pass
functional_control_pass
functional_system_pass
quadratic_baseline_challenge_pass
noise_robustness_pass
external_fair_ready
primary_blocker
next_required_implementation
success_v929_functional_opened
success_v929_functional_advantage
success_v929_external_ready
```

---

## 7. Required artifacts

```text
run_manifest.json
contract_audit_v929.csv
functional_open_gate_v929.csv
snr_instrumentation_v929.csv
one_step_population_risk_audit_v929.csv
short_run_functional_safety_v929.csv
full_functional_reentry_10seed_v929.csv
strong_baseline_challenge_v929.csv
robustness_noise_diagnostic_v929.csv
functional_event_trace_v929.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_purekan_equivalence_fail
F3_functional_open_gate_fail
F4_snr_estimator_unstable
F5_snr_overhead_fail
F6_one_step_task_safety_fail
F7_functional_task_drop
F8_functional_geometry_no_gain
F9_control_causality_fail
F10_kmnist_repair_fail
F11_quadratic_baseline_explains_gain
F12_system_overhead_fail
F13_robustness_no_gain
F14_external_fair_not_ready
F15_fake_or_proxy_violation
F16_artifact_missing
```

---

## 8. 第一轮执行顺序

```text
Step 1:
  P0 确认 LQ base 的 functional open gate。
  如果 P0 不过，不打开 functional。

Step 2:
  P1 只做 SNR instrumentation，不更新参数。
  先确认 SNR gate 有信息量，且 overhead 可接受。

Step 3:
  P2 做 one-step population-risk / task-safety audit。
  验证 SNR gate 是否预测 holdout improvement。

Step 4:
  P3 做 20/50-step short-run functional safety。
  淘汰 task-unsafe 或 controls 过不了的 functional variants。

Step 5:
  P4 做 10-seed full functional re-entry。
  这是本轮核心。

Step 6:
  P5 做 strong baseline challenge。
  判断 functional advantage 是否被 QuadraticFeatureMLP 解释。

Step 7:
  P6 做 noisy-signal robustness diagnostic。
  验证 SNR gate 是否真的 suppress noise。

Step 8:
  P7 route decision。
  只有 R8-ExternalFairReady 才允许后续 external fair。
```

---

## 9. 停止条件

### 成功停止

Minimum functional success：

```text
P0 gate pass
P1 SNR stable
P2 one-step task-safe
P4 full functional re-entry task-safe
functional geometry pass
functional system pass
controls pass
```

Strong functional success：

```text
Minimum functional success
+
KMNIST miss-row repair
+
calibration/tail improvement
+
strong baseline challenge pass
+
robustness/noise pass
```

External-ready success：

```text
Strong functional success
+
macro delta vs MLP >= 0
or clear geometry/calibration/robustness advantage without task drop
```

### 失败停止

```text
1. LQ base no longer reproduces P4 near-pass；
2. SNR estimator active fraction collapses to 0 or 1；
3. SNR overhead exceeds 20% of step；
4. one-step bad-step rate > 5%；
5. functional loses task by more than 0.005；
6. functional does not beat NoOp / Random / Shuffled controls；
7. functional gain is explained by QuadraticFeatureMLP；
8. functional breaks step/memory gate；
9. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 10. 最终解释规则

### Case A：functional improves geometry and task

可以声明：

```text
FC-PureKAN functional update gives task-safe geometry-aware advantage on LQ near-pass base.
```

若 macro delta reaches full-pass：

```text
FC-PureKAN + functional reaches AdamW-comparable or beyond-MLP under current task family.
```

### Case B：functional improves geometry but not task

必须声明：

```text
Functional update provides geometry / calibration evidence, but not task advantage.
```

### Case C：functional unsafe

必须声明：

```text
Current functional direction is not task-safe; SNR gate or projection insufficient.
```

### Case D：SNR gate works but only as optimizer diagnostic

必须声明：

```text
Population-risk SNR gate is informative, but current geometry direction is weak.
```

### Case E：Quadratic baseline explains gain

必须声明：

```text
LQ family remains useful, but the observed advantage is not yet uniquely KAN-functional.
```

---

## 11. 最终建议

v9.2.9 的一句话策略是：

$$
\boxed{
\text{以 LQ near-pass base 为地基，正式打开 SNR-gated、geometry-aware、task-safe functional update 主线。}
}
$$

当前最重要的问题不是继续优化系统 kernel，也不是继续调 optimizer，而是：

```text
1. functional update 是否能在不改 loss、不用 teacher 的条件下改善 generalization-relevant signal？
2. SNR gate 是否能区分 signal-dominant 和 noise-dominant update direction？
3. KMNIST miss rows 是否能被 functional update 修复？
4. geometry improvement 是否能转化为 ECE/NLL/robustness/CE-tail 改善？
5. functional advantage 是否能经受 NoOp/Random/ShuffledSNR/QuadraticFeatureMLP controls？
```

只有这些成立，functional update 才能成为 DG-KAN 的真正核心贡献。
