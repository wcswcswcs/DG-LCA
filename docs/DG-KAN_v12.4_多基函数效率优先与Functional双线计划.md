# DG-KAN v12.4：多基函数效率优先与 Functional Update 双线验证计划

> 版本：v12.4 draft  
> 目标：从 v12.3 的 LQ frame family 失败中抽象出真正问题，转向 primitive-level basis design；同时保持 functional update 诊断线并行推进。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。

---

## 0. 一句话判断

v12.3 不是没有进展，但它不是能力成功。它完成了一个重要的排错：**同一类 LQ quadratic frame 的修修补补已经不应该继续作为主线**。C8/C13 能把局部 E1 pairwise cover 做到接近 1，但不能同时满足 rotated/random quadratic coverage、condition、task stability；C10 能局部降低 condition，但 expression 和 task 又掉下去。这说明当前 blocker 不是某个 seed、某个数据集、某个 threshold，而是：

$$
\boxed{
\text{当前 LQ rank-1 quadratic lift/frame 不能同时满足 coverage、condition、task stability 与 MLP-like efficiency。}
}
$$

因此 v12.4 的路线应当转为：

$$
\boxed{
\text{多基函数 primitive-level design}
+
\text{MLP-like forward/backward efficiency gate}
+
\text{并行 functional update control-resistant audit。}
}
$$

Functional update 不能等到所有 base 都完美后才看，但也不能在 base 不合格时冒充 official success。因此本计划采用双线：

```text
Line A: Basis Primitive Efficiency + Base Qualification
  目标是找到 strict FC-PureKAN base，先过前馈/反传/step/memory 与表达力硬门。

Line B: Basis-aware Functional Update Diagnostic
  与 Line A 并行运行 one-step / five-step / cloned rollout。
  只要 base 未过，只能写 diagnostic；base 过后才允许 official functional training。
```

---

## 1. 本轮实验结果的独立分析

### 1.1 v12.3 的真实进展

v12.3 的工程纪律是好的。它没有调低 functional gate，没有按 MNIST/Fashion/KMNIST 名称分支，没有使用 teacher/distillation/loss modification/class weights，也没有把 diagnostic/not_run row 写成 success。它还把 C0/C8/C10/C11-C17 全量 triage 做完，避免了只看 top-K expression candidate 的偏差。

真正的进展是三点。

第一，**timing 已经不是当前第一 blocker**。C0/C8/C10/C11/C12/C15/C16 的 clean step 基本在 `1.25x` 附近或以内；C13/C14/C17 超过 system gate。A7 hook path 极慢，说明 diagnostic hook 不能混进 architecture clean timing。下一步所有效率 gate 必须分离：

```text
architecture_clean_timing
online_functional_overhead
offline_diagnostic_hook_cost
```

第二，**局部 pairwise cover 可以结构性修复，但不能推广为全局 base success**。C8/C13 的 E1 LS 接近 `0.992`，说明 signed-pair / block-orthogonal pair cover 对原始 pairwise-product 有强局部修复；但 E6 rotated pairwise 和 E8 random quadratic coverage 仍低，而且 task stability 和 condition 没同时过。

第三，**trainable capacity 与 frozen/global coverage 分裂**。C11-C17 的 E1 trainable R2 多数接近 `0.99`，说明很多候选经过训练能学 pairwise；但 frozen/global span coverage 低，说明这些 basis/frame 不能在训练初期提供稳定、可用、condition 好的 coordinate system。这正是你关心的“几何稳定性”问题。

### 1.2 v12.3 没有能力成功

v12.3 的 route 是：

```text
route = R6-LQFrameFamilyExhausted
base_qualified = False
functional_open = False
promoted_candidates = []
```

这意味着：

```text
1. 没有 base survivor；
2. 没有 official functional audit；
3. 没有 short-run / full-run functional training；
4. 没有 next-gen MLP claim。
```

所以这轮不能写成 “LQ 已经快成功了，只差一点调参”。更准确的判断是：

$$
\boxed{
\text{LQ quadratic frame family 已经完成一轮充分排查，应转向 basis primitive-level design。}
}
$$

### 1.3 当前真正卡在哪里

现在不是简单的 “KAN 表达力不够”。更准确是：

$$
\boxed{
\text{KAN basis coordinate 没有同时做到：可表达、可训练、低条件数、低系统代价。}
}
$$

具体拆开：

```text
Expression:
  训练足够久能拟合 pairwise，但 frozen/global basis coverage 不够。

Condition:
  C10 能降低 condition，但牺牲 task/expression；C8/C13 提升局部 coverage，但 condition 和 task 变坏。

Task stability:
  没有 candidate 在 5-epoch triage 中同时满足 mean delta、worst-row safety、coverage、condition 与 system gate。

Functional:
  base 未合格，官方 functional 不能打开；过去 diagnostic functional 也长期没打过 strong controls。

Efficiency:
  clean architecture timing 对若干 LQ variants 已接近 MLP，但这只是必要条件，不是 base success。
```

### 1.4 是否还在正确道路上

方向仍是正确的，但推进方式必须变。继续做 LQ frame 小修会慢而且低效，因为我们已经看到同类候选的 trade-off：

```text
局部 E1 cover 好 -> rotated/random coverage、condition、task 掉；
condition 好 -> coverage/task 掉；
trainable R2 好 -> frozen/global coordinate 仍差；
clean timing 好 -> base gate 仍不过。
```

因此下一步不是 “调 C8/C10/C13 的参数”，而是扩大 basis family，但以严格 efficiency gate 为第一门。

---

## 2. 总目标

DG-KAN 的目标仍然是：

$$
\boxed{
\text{PureKAN base} + \text{functional update}
\rightarrow
\text{比 PureKAN base + ordinary backprop 更好的模型。}
}
$$

这个目标必须拆成三个可验证层次。

### 2.1 Base minimum claim

找到一个 strict FC-PureKAN primitive，使其满足：

```text
1. 表达力不低于 same-param / same-FLOPs MLP；
2. clean forward/backward/step/memory 与 MLP comparable；
3. 训练 task 不低于 MLP envelope；
4. geometry certificate 不出现 rank / cover / tail collapse。
```

### 2.2 Functional minimum claim

在同一个合格 PureKAN base 上，functional update 满足：

```text
1. 不损伤 task；
2. 不损伤 expression；
3. 改善 geometry / calibration / tail / signal separation 至少一类；
4. 打过 NoOp、RandomMatchedNorm、AdamWParallel、SNR-only、ShuffledPayload、MLP analog controls；
5. amortized overhead 仍在 MLP-comparable envelope 内。
```

### 2.3 Strong claim

最终希望证明：

$$
\boxed{
\text{KAN basis primitive 提供表达/几何坐标优势，functional update 提供训练轨迹/几何维护优势。}
}
$$

两者必须区分，不允许把 KAN base 的收益说成 functional update，也不允许把 functional 的收益说成 KAN architecture。

---

## 3. 先确定要测哪些基函数

参考 awesome-kan 中列出的 KAN 方向，当前最相关的 basis families 包括：B-spline / efficient-kan、FastKAN / RBF、FasterKAN / RSWAF、FourierKAN、ChebyKAN、OrthogPolyKAN、JacobiKAN、Wav-KAN / cuda-Wavelet-KAN、TaylorKAN、BSRBF / function-combination KAN、Rational/KAT 等。

但我们不能把所有东西一起上 full training。第一步要按 **效率可达性** 和 **functional metric 可定义性** 排优先级。

### 3.1 Tier 0：保留对照，不再主修

#### T0-LQ-current

用途：当前 base 对照。

```text
C0-R2-LQ-current
C8-signed-pair-cover-lift
C10-data-pca-whiten-h64
C13-block-orthogonal-pair-cover
```

处理方式：只作为 reference / failure slice，不继续小修 LQ frame。

#### T0-MLP controls

```text
MLP-same-param-AdamW
MLP-same-FLOPs / same-step-AdamW
QuadraticFeatureMLP diagnostic
```

用途：拆分 KAN vs MLP、quadratic-feature shortcut、functional analog。

---

### 3.2 Tier 1：第一批必须并行测的 basis families

#### B1：Activation / ReLU / RSWAF family

候选名：

```text
B1a-ReLU-KAN-local-hinge
B1b-FasterKAN-RSWAF
B1c-AF-KAN-silu-gelu-mish-lite
B1d-piecewise-linear-hinge-KAN
```

优先原因：

```text
1. forward/backward 都是 pointwise activation + GEMM，最有希望接近 MLP；
2. derivative 简单，manual/fused backward 容易；
3. local hinge/activation occupancy 可直接定义 cover entropy；
4. functional update 可定义为 hinge/activation region rebalance、slope cap、active-region entropy maintenance。
```

主要风险：

```text
1. 可能退化为普通 activation MLP 的变体；
2. 表达力可能不如 spline/RBF；
3. 需要 strict audit，确保参数属于 edge basis，不是普通 MLP hidden path。
```

第一优先级配置：

```text
K = 4, 8, 16 local hinges / activation components
hidden = same-param matched
basis init = identity-like + small residual
```

---

#### B2：Compact RBF / FastKAN family

候选名：

```text
B2a-GaussianRBF-K4-compact
B2b-FastKAN-RBF-approx-spline
B2c-LUT-RBF-K8-local
B2d-BSRBF-lite-local-only
```

优先原因：

```text
1. RBF / FastKAN 是 KAN 中最自然的 spline approximation 之一；
2. local support 好，geometry / curvature / occupancy 容易定义；
3. functional metric 直接：coefficient L2/H1/H2、center/width safety、occupancy rebalance；
4. 如果 K 很小且不 materialize dense basis，有机会 MLP-like。
```

主要风险：

```text
1. 过去 dense RBF / AB-RBF 已多次因 basis materialization 太重失败；
2. exp 代价可能高；
3. 必须避免 [B,d,out,K] 或 [B,d,K] 大张量进入 backward live set。
```

第一优先级配置：

```text
K = 4 or 8
basis = local compact / fast approximate Gaussian
backward = recompute basis or fused coeffgrad
no dense edge-out basis materialization
```

---

#### B3：Orthogonal polynomial family

候选名：

```text
B3a-ChebyKAN-K3/K4/K6
B3b-LegendreKAN-K3/K4/K6
B3c-JacobiKAN-K3/K4
B3d-OrthogPolyKAN-shared-recurrence
```

优先原因：

```text
1. recurrence 计算简单，forward/backward 可做到 O(K)；
2. 不需要 grid / knots / exp；
3. 对 global smooth function 与 polynomial interaction 可能比 LQ frame 更自然；
4. spectral coefficient energy 可以直接作为 geometry metric。
```

主要风险：

```text
1. global support 可能导致 tail / perturb drift；
2. 高阶 polynomial condition 会爆；
3. 需要严格 input normalization，否则数值不稳。
```

第一优先级配置：

```text
K <= 6
x normalized to [-1,1]
recurrence-based basis eval
degree-energy damping init
```

---

#### B4：Low-frequency Fourier family

候选名：

```text
B4a-FourierKAN-sin-cos-K2/K4
B4b-FourierKAN-lowfreq-plus-linear
B4c-Fourier-local-windowed-diagnostic
```

优先原因：

```text
1. sin/cos basis 对高频/周期结构有表达优势；
2. derivative 与 curvature 可解析；
3. spectral bias / high-frequency stress battery 可直接测试；
4. low K 时计算代价可控。
```

主要风险：

```text
1. global support 可能造成非局部扰动；
2. sin/cos kernel 代价可能高于 activation/poly；
3. 对图像分类未必有优势，不能只因 INR 任务好就主推。
```

第一优先级配置：

```text
K = 2,4
frequency init = low-frequency only
amplitude small residual
```

---

#### B5：Wavelet / multiscale local family

候选名：

```text
B5a-HaarWaveletKAN-lite
B5b-MorletWaveletKAN-lite
B5c-MexicanHat/RickerWaveletKAN-lite
B5d-cuda-Wavelet-KAN diagnostic, if code usable
```

优先原因：

```text
1. wavelet 兼具 locality 与 scale decomposition；
2. 对 hard-tail、局部 stroke/shape 可能比 global polynomial 更稳；
3. functional geometry 可定义为 scale-energy balance 与 local tail correction。
```

主要风险：

```text
1. basis 参数多，scale/shift 管理容易变重；
2. backward 可能复杂；
3. 没有 fused path 前不能进入 full task。
```

第一优先级配置：

```text
scales = 2 or 3
fixed scale grid first
learnable coefficient only first
```

---

### 3.3 Tier 2：有价值但必须先 microbench 过线

#### B6：Local B-spline / EfficientKAN / MatrixKAN family

候选名：

```text
B6a-BSpline-order1-local
B6b-BSpline-order2-local
B6c-BSpline-order3-local-efficient
B6d-MatrixKAN-style-spline-mat
```

优先原因：原始 KAN 的自然 basis，解释性好。  
风险：过去 dense spline/basis 容易慢，必须 local-support + no dense materialization。

进入条件：

```text
architecture clean step <= 1.35x
backward memory <= 1.10x
no [B,d,out,K] materialization
```

#### B7：Rational / KAT family

候选名：

```text
B7a-RationalKAT-lite-safe-den
B7b-fast-rational-Horner
B7c-tangent-metric-rational-diagnostic
```

优先原因：计算形态可能接近 activation，表达强。  
风险：denominator safety 与 tangent metric 复杂，旧 Rational functional update 失败过。

进入条件：

```text
denominator p01 safe
clean step <= 1.35x
rational tangent metric condition <= 1e4
```

#### B8：Hybrid function-combination family

候选名：

```text
B8a-BSRBF-lite
B8b-FC-KAN-lite-combo
B8c-poly+RBF two-branch-lite
B8d-activation+orthopoly-lite
```

优先原因：单一 basis 可能不够，组合能覆盖 locality + global smoothness。  
风险：最容易变慢、变成参数堆砌、贡献不可解释。

进入条件：必须有至少两个 single-family basis 已通过 microbench，并且 hybrid overhead 小于：

$$
\Delta T_{step} \leq 0.15,
$$

$$
\Delta M_{peak} \leq 0.10.
$$

---

### 3.4 Tier 3：只做 diagnostic，不进 full training

```text
Dense B-spline large-K
Dense RBF / AB-RBF large-K
large-K Fourier
high-degree polynomial K>8
unfused wavelet with learnable scale/shift
any implementation requiring large [B,d,out,K] tensor
```

这些可以用来做 upper-bound / teacher-free expression reference，但不能成为 next-gen MLP 候选。

---

## 4. 统一 architecture 约束

所有 candidate 必须遵守：

```text
1. strict FC-PureKAN；
2. no ordinary MLP hidden path；
3. no learnable LayerNorm / hidden stem / non-KAN head；
4. CE-only；
5. no teacher / no distillation / no loss modification / no class weights；
6. no dataset-name branch；
7. manual or graph-free path where possible；
8. clean architecture timing 与 diagnostic/functional hook timing 分开。
```

对每个 basis family，都要记录：

```text
basis_family
basis_name
K
basis_order
local_support
global_support
uses_exp
uses_sin_cos
uses_division
uses_gather_scatter
uses_dense_basis_tensor
manual_forward
manual_backward
manual_update
strict_purekan_pass
nonkan_param_count
edge_param_count
param_ratio_vs_mlp
```

---

## 5. Line A：Basis Primitive Efficiency + Base Qualification

### A0：basis implementation contract

#### 目标

确认候选是 strict PureKAN，并能正确 forward/backward/update。

#### 必须记录

```text
method
basis_family
basis_variant
K
hidden_dim
param_count
param_ratio_vs_mlp
edge_param_count
nonkan_param_count
manual_forward_available
manual_backward_available
manual_update_available
uses_loss_backward
uses_torch_autograd_graph
rollback_max_error
grad_relerr_max
grad_cos_min
nan_inf_count
```

#### 通过标准

$$
nonKAN = 0,
$$

$$
rollback\_max\_error < 10^{-8},
$$

$$
grad\_relerr < 10^{-4},
$$

$$
grad\_cos > 0.999.
$$

如果某 candidate 没有 manual backward，但 PyTorch autograd path clean 且只是 microbench 阶段，可以保留为 diagnostic；不能进入 final claim。

---

### A1：clean efficiency microbench

#### 目标

第一步就是回答用户强调的问题：**前馈/反传/代价能不能和 MLP 比肩**。任何 basis 如果第一门不过，不进入 task full run。

#### 设置

```text
batch_size = 128, 256, 512
hidden_dim = same-param matched
input_dim = 784
num_classes = 10
depth = 2 first, then 3/4 diagnostic
warmup_steps = 50
measure_steps = 200
precision = fp32 first, bf16 diagnostic only after grad pass
```

#### 对照

```text
MLP-same-param-AdamW
MLP-same-FLOPs-or-step
C0-R2-LQ-current
best historical LQ clean path
```

#### 必须记录

```text
forward_q50_ms
forward_q90_ms
backward_q50_ms
backward_q90_ms
update_q50_ms
update_q90_ms
step_q50_ms
step_q90_ms
forward_ratio_q90_vs_mlp
backward_ratio_q90_vs_mlp
update_ratio_q90_vs_mlp
step_ratio_q90_vs_mlp
peak_allocated_mb
peak_reserved_mb
compact_memory_ratio
conservative_memory_ratio
kernel_count_forward
kernel_count_backward
kernel_count_update
allocation_count
sync_count
largest_temp_tensor_mb
```

#### Gate

Exploratory pass：

$$
T_{step,q90}/T_{MLP,q90} \leq 1.50,
$$

$$
M_{peak}/M_{MLP} \leq 1.25.
$$

Base candidate pass：

$$
T_{forward,q90}/T_{MLP,q90} \leq 1.25,
$$

$$
T_{backward,q90}/T_{MLP,q90} \leq 1.35,
$$

$$
T_{step,q90}/T_{MLP,q90} \leq 1.25,
$$

$$
M_{peak}/M_{MLP} \leq 1.05.
$$

Final target：

$$
T_{step,q90}/T_{MLP,q90} \leq 1.10,
$$

$$
M_{peak}/M_{MLP} \leq 1.00.
$$

#### 可视化

```text
fig_a1_step_ratio_by_basis.svg
fig_a1_forward_backward_update_stacked_bar.svg
fig_a1_memory_ratio_by_basis.svg
fig_a1_kernel_count_vs_step_ratio.svg
fig_a1_temp_tensor_waterfall.svg
fig_a1_batch_scaling_curve.svg
```

#### 不满足时 Codex 先尝试

```text
If step fail but forward/backward component identified:
  1. remove dense basis tensor;
  2. recompute basis in backward;
  3. fuse basis eval + edge mix;
  4. replace gather/scatter with local contiguous table;
  5. reduce K or use recurrence;
  6. torch.compile warm path;
  7. Triton microkernel only after component attribution.

If memory fail:
  1. live-set timeline;
  2. no cache / short cache / recompute variants;
  3. avoid storing basis outputs;
  4. stream coeffgrad;
  5. split diagnostic hook from architecture timing;
  6. check optimizer state memory parity.

If grad fail:
  1. small-shape autograd reference;
  2. role-wise gradient comparison;
  3. disable mixed precision;
  4. isolate input grad vs coeff grad;
  5. mark candidate diagnostic only until fixed.
```

---

### A2：expression and coordinate coverage battery

#### 目标

避免再出现 v12.3 的问题：trainable pairwise R2 好看，但 frozen/global coordinate coverage 差。

#### Synthetic targets

```text
E0-additive
E1-pairwise-product
E2-composition
E3-local-XOR
E4-high-frequency
E5-noise-stress
E6-rotated-pairwise-product
E7-piecewise-local
E8-random-quadratic-form
E9-smooth-nonpolynomial
E10-local-bump-mixture
```

#### 协议

```text
B0 short fit: 60 steps
B1 standard fit: 1000 steps
B2 high-budget fit: 5000 steps
B3 frozen-basis readout LS
B4 random-label / noise stress diagnostic
```

#### 必须记录

```text
target
protocol
basis_family
method
val_R2
test_R2
delta_vs_mlp
frozen_readout_R2
trainable_best_R2
steps_to_R2_0_95
condition_proxy
basis_effective_rank
basis_entropy
dead_basis_fraction
basis_output_norm_p95
```

#### Gate

候选不能只靠高预算训练补救。需要：

$$
R^2_{B1,target} \geq R^2_{MLP,target} - 0.02
$$

for at least E1/E2 and one of E6/E8, and:

$$
R^2_{B3,frozen} \geq 0.65
$$

on at least two interaction targets, unless the family is explicitly non-quadratic and shows strong B1/B2 trainability plus good condition.

#### 可视化

```text
fig_a2_expression_heatmap_basis_target.svg
fig_a2_frozen_vs_trainable_scatter.svg
fig_a2_steps_to_r2_by_basis.svg
fig_a2_basis_rank_entropy.svg
fig_a2_failure_target_sankey.svg
```

#### 不满足时 Codex 先尝试

```text
If frozen coverage low but trainable R2 high:
  1. improve initialization / input normalization;
  2. add identity residual scale;
  3. orthogonalize or whiten basis outputs;
  4. use low-coherence initialization;
  5. test small K increase within efficiency budget.

If B1/B2 both low:
  1. basis family under-expressive at current K;
  2. try K+2 or local/global hybrid lite;
  3. if still low, demote family to diagnostic.

If only high-frequency target fails:
  1. do not overfit the basis to high-frequency unless task/tail requires it;
  2. record as spectral limitation.
```

---

### A3：5-epoch task triage

#### 目标

判断 basis 是否在真实 task 上保持 MLP envelope，同时不能按 dataset 调参。

#### 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
train_size = 1024 or 1536
val_size = 512
test_size = 512
epochs = 5
batch_size = 128
```

#### 对照

```text
MLP-same-param-AdamW
MLP-same-step/FLOPs-AdamW
C0-R2-LQ-current
best basis candidate per family
```

#### 必须记录

```text
val_acc
val_loss
test_acc
NLL
ECE
CEp99
margin_mean
margin_p10
val_loss_auc_step
val_loss_auc_time
classwise_acc
step_time_q90
memory_ratio
basis_entropy
effective_rank_hidden
perturb_logit_drift_p95
signal_consistency_real
noise_leak
```

#### Gate

$$
\Delta Acc_{macro} \geq -0.005,
$$

$$
near\_pass\_rate \geq 0.80,
$$

$$
worst\_row\_delta \geq -0.025,
$$

$$
ValLossAUC_{time} \leq 1.05 \cdot ValLossAUC_{MLP,time},
$$

$$
ECE \leq ECE_{MLP}+0.02.
$$

Dataset 只作为 slice 诊断，不允许 candidate 或超参按 dataset name 选择。

#### 可视化

```text
fig_a3_val_acc_delta_heatmap_dataset_seed.svg
fig_a3_loss_vs_time_by_basis.svg
fig_a3_task_efficiency_pareto.svg
fig_a3_ece_tail_pareto.svg
fig_a3_classwise_failure_heatmap.svg
```

#### 不满足时 Codex 先尝试

```text
If all datasets fail:
  1. basis family likely bad or optimization unstable;
  2. adjust global init / residual scale / LR schedule;
  3. do not dataset-tune.

If one dataset slice fails:
  1. diagnose classwise, tail, perturb drift, basis entropy;
  2. derive global fix only, e.g. normalization, condition repair, tail-safe init;
  3. rerun all datasets with same fix.

If task ok but time/AUC-time fail:
  1. reduce hook overhead;
  2. optimize update path;
  3. do not claim base candidate until wall-clock passes.
```

---

### A4：20/30-epoch survivor confirmation

Only candidates passing A1/A2/A3 enter this stage.

#### 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
budget = 20 or 30 epoch equivalent
```

#### Gate

$$
Acc_{basis} \geq Acc_{MLP} - 0.003
$$

or paired CI low no worse than `-0.005`, and:

$$
T_{step,q90}/T_{MLP,q90} \leq 1.25,
$$

$$
ValLossAUC_{time,basis} \leq ValLossAUC_{time,MLP},
$$

$$
ECE_{basis} \leq ECE_{MLP},
$$

$$
GeoHardGate = pass.
$$

---

## 6. Line B：Basis-aware Functional Update Diagnostic

Line B 与 Line A 并行，但结论分级：

```text
base_not_qualified:
  只允许 diagnostic functional audit。

base_qualified:
  允许 official one-step/five-step functional audit。

base_qualified + functional_control_pass:
  允许 short-run functional training。
```

### B0：为每类 basis 定义 functional geometry direction

#### B1 Activation / RSWAF / ReLU family

```text
G-act-occupancy-rebalance
G-act-slope-cap
G-act-region-entropy
G-act-tail-margin-residual
```

指标：

```text
active_region_entropy
slope_p95
dead_hinge_fraction
tail_region_coverage
```

#### B2 RBF / FastKAN family

```text
G-rbf-occupancy-rebalance
G-rbf-curvature-smoothing-with-compensation
G-rbf-center-width-condition-repair
G-rbf-tail-local-refit
```

指标：

```text
basis_occupancy_entropy
dead_basis_fraction
center_width_condition
curvature_debt
```

#### B3 Orthogonal polynomial family

```text
G-poly-degree-energy-damping
G-poly-spectral-whitening
G-poly-low-degree-preserving-tail-repair
```

指标：

```text
degree_energy_distribution
high_degree_energy_ratio
recurrence_condition_proxy
spectral_entropy
```

#### B4 Fourier family

```text
G-fourier-highfreq-damping
G-fourier-phase-stability
G-fourier-lowfreq-signal-projection
```

指标：

```text
frequency_energy
highfreq_energy_ratio
phase_drift
perturb_drift
```

#### B5 Wavelet family

```text
G-wavelet-scale-energy-balance
G-wavelet-local-tail-correction
G-wavelet-dead-scale-revival
```

指标：

```text
scale_energy_distribution
local_support_entropy
hard_tail_scale_coverage
```

#### B6 B-spline family

```text
G-spline-knot-occupancy-rebalance
G-spline-second-difference-smoothing
G-spline-knot-condition-repair
```

指标：

```text
knot_entropy
empty_knot_fraction
second_difference_energy
total_variation
```

#### B7 Rational / KAT family

```text
G-rational-denominator-safety
G-rational-tangent-metric-step
G-rational-curvature-with-den-safety
```

指标：

```text
denominator_p01
denominator_condition
tangent_metric_condition
r_prime_p95
r_double_prime_energy
```

---

### B1：one-step / five-step cloned audit

#### 目标

回答 functional update 是否有独立价值，而不是又被 AdamWParallel / RandomMatchedNorm 解释。

#### 对照矩阵

```text
D0-TaskOnlyAdamW
D1-NoOpMatchedOverhead
D2-RandomMatchedNorm
D3-AdamWParallelDirection
D4-ShuffledPayload
D5-ShuffledEvent
D6-SNR-only
D7-GeometryOnlyNoSNR
D8-BasisAwareFunctional
D9-BasisAwareFunctional-AdamWOrthogonal
D10-BasisAwareFunctional-SNRProjected
D11-MLPAnalogGeometryMaintenance
D12-QuadraticFeatureMLPAnalog
```

#### 必须记录

```text
basis_family
method
candidate_id
step_id
train_descent
holdout_descent
holdout_descent_ratio
bad_step
bad_step_rate
lambda_selected
lambda_backtrack_count
cos_to_adamw
cos_to_random
functional_norm_ratio
GeoDebt_delta
ECE_delta
CEp99_delta
margin_p10_delta
rank_delta
basis_entropy_delta
perturb_drift_delta
signal_consistency_delta
noise_leak_delta
step_overhead_ms
memory_overhead_mb
control_gap_vs_best_control
```

#### Gate

$$
holdout\_descent\_ratio \geq 0.95,
$$

$$
bad\_step\_rate \leq 0.02,
$$

$$
GeoDebt_{functional} \leq GeoDebt_{task} - 0.10|GeoDebt_{task}|,
$$

$$
control\_gap > 0
$$

with paired CI low above `0` or at least stable positive across basis families.

#### 可视化

```text
fig_b1_control_gap_by_basis.svg
fig_b1_holdout_descent_vs_geodebt.svg
fig_b1_lambda_backtracking_hist.svg
fig_b1_functional_vs_adam_cosine.svg
fig_b1_bad_step_rate_heatmap.svg
fig_b1_control_resistance_pareto.svg
```

#### 不满足时 Codex 先尝试

```text
If task-safe but control_gap <= 0:
  1. do not lower gate;
  2. compute AdamW-orthogonal residual direction;
  3. remove components parallel to RandomMatchedNorm / AdamWParallel;
  4. redesign basis-specific geometry target.

If holdout descent fails:
  1. reduce lambda with backtracking;
  2. add SNR projection;
  3. veto high tail-risk regions;
  4. if still fail, demote direction.

If GeoDebt improves but rank/expression collapses:
  1. add compensation term;
  2. preserve effective rank and basis entropy;
  3. reject pure smoothing direction.

If overhead fails:
  1. event every K steps;
  2. cache basis stats;
  3. low-rank sketch;
  4. offline diagnostic only until amortized overhead passes.
```

---

### B2：basis-specific functional micro-training

Only run for basis candidates that pass A1 exploratory and B1 diagnostic positive.

#### 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
budget = 5 epochs cloned micro-training
functional_event_stride = 16, 32, 64
same stride across all datasets
```

#### Methods

```text
Base-AdamW
Base-AdamW-NoOpOverhead
Base-AdamW-RandomMaintenance
Base-AdamW-AdamWParallelMaintenance
Base-AdamW-SNR-only
Base-AdamW-BasisAwareFunctional
MLP-AdamW-AnalogFunctional
```

#### Gate

$$
Acc_{functional} \geq Acc_{base} - 0.003,
$$

$$
ValLossAUC_{time,functional} \leq 1.05 \cdot ValLossAUC_{time,base},
$$

$$
ECE_{functional} \leq ECE_{base},
$$

$$
GeoDebt_{functional} \leq GeoDebt_{base} - 10\%,
$$

and it must beat matched controls.

---

## 7. Good Geometry Certificate V1

Good geometry must not be a single curvature score. Define:

$$
\theta' \succ_G \theta
$$

if all hard gates pass and at least one Pareto axis improves without collapse.

### 7.1 Hard gates

```text
TaskSlack pass
ValLossAUC_time pass
ECE/NLL pass
Expression retention pass
Effective rank pass
Basis entropy pass
Tail risk pass
Efficiency pass
Control resistance pass, for functional claim
```

### 7.2 Pareto axes

```text
Condition:
  kernel_condition_proxy
  recurrence_condition_proxy
  lift_condition_proxy

Cover:
  basis_entropy
  active_region_entropy
  dead_basis_fraction
  scale_energy_entropy

Stability:
  perturb_logit_drift_p95
  local_jacobian_norm_p95
  curvature_debt

Signal/noise:
  signal_consistency_real
  signal_consistency_noise
  noise_leak
  offdiag_population_proxy

Tail/calibration:
  CEp99
  margin_p10
  wrong_confidence_p95
  ECE
  NLL
```

### 7.3 Geometry Debt definition

A first V1 diagnostic score may be reported, but not used alone for pass/fail:

$$
GeoDebt
=
 z(Condition)
+ z(PerturbDrift)
+ z(CurvatureDebt)
+ z(TailRisk)
+ z(CoverCollapseRisk)
+ z(NoiseLeak).
$$

Important rule:

```text
Lower GeoDebt is only meaningful if expression/task hard gates pass.
```

---

## 8. 并行执行安排

### Batch 1：basis microbench 并行

Run all Tier-1 families with small K:

```text
B1 Activation / RSWAF / ReLU
B2 Compact RBF / FastKAN
B3 Chebyshev / Legendre / Jacobi
B4 Fourier low-K
B5 Wavelet lite
```

Expected output:

```text
v124_basis_manifest.csv
v124_efficiency_microbench.csv
v124_grad_correctness.csv
v124_efficiency_failure_table.csv
fig_a1_*.svg
```

### Batch 2：expression battery 并行

Only A1 exploratory survivors run A2.

Expected output:

```text
v124_expression_battery.csv
v124_frozen_readout.csv
v124_basis_condition.csv
fig_a2_*.svg
```

### Batch 3：5-epoch triage

Only A1 + partial A2 survivors run A3.

Expected output:

```text
v124_task_triage.csv
v124_task_trace.csv
v124_geometry_snapshot.csv
v124_signal_noise.csv
v124_tail_calibration.csv
fig_a3_*.svg
```

### Batch 4：functional diagnostic in parallel

Run B1 cloned audit for all basis candidates that pass A1 exploratory, even if A3 not yet passed. Mark rows by:

```text
functional_status = diagnostic_base_not_qualified
functional_status = official_base_qualified
```

Expected output:

```text
v124_functional_one_step.csv
v124_functional_five_step.csv
v124_control_matrix.csv
v124_lambda_backtracking.csv
fig_b1_*.svg
```

### Batch 5：survivor confirmation

Only candidates passing A1/A2/A3 and showing B1 diagnostic value enter A4/B2.

---

## 9. Route decision

### R1：EfficiencyAllFail

```text
No basis family passes A1 exploratory.
```

Next:

```text
Implement fused/manual kernel for the top two families only.
Do not run task/functional full training.
```

### R2：EfficiencyPassExpressionFail

```text
At least one family is MLP-like, but expression battery fails.
```

Next:

```text
Increase K within budget, improve init/normalization, or try lite hybrid.
```

### R3：ExpressionPassTaskFail

```text
Basis can express synthetic targets but fails real task triage.
```

Next:

```text
Global optimizer/initialization repair; no dataset-specific tuning.
```

### R4：BaseQualifiedFunctionalFail

```text
Base passes, but functional update does not beat controls.
```

Next:

```text
Keep base as KAN candidate; redesign basis-aware functional target.
```

### R5：FunctionalDiagnosticPositiveBaseNotYetQualified

```text
Functional direction improves geometry vs controls on cloned audit, but base still not qualified.
```

Next:

```text
Continue base repair; do not write functional success.
```

### R6：BasisAndFunctionalCandidateReady

```text
At least one basis base passes, and functional diagnostic beats controls.
```

Next:

```text
Run 20/30-epoch 3-seed short training, then 10-seed confirm.
```

---

## 10. 成功标准

### 10.1 Basis base success

A basis base is successful if:

$$
Acc_{KAN} \geq Acc_{MLP} - 0.003,
$$

$$
ValLossAUC_{time,KAN} \leq ValLossAUC_{time,MLP},
$$

$$
ECE_{KAN} \leq ECE_{MLP},
$$

$$
T_{step,q90}/T_{MLP,q90} \leq 1.25,
$$

$$
M_{peak}/M_{MLP} \leq 1.05,
$$

and expression hard gates pass.

### 10.2 Functional update success

Functional update is successful if, on the same qualified basis base:

$$
Acc_{functional} \geq Acc_{base} - 0.003,
$$

$$
ValLossAUC_{time,functional} \leq ValLossAUC_{time,base},
$$

$$
ECE_{functional} \leq ECE_{base},
$$

$$
GeoDebt_{functional} \leq GeoDebt_{base} - 10\%,
$$

and:

```text
beats NoOpMatchedOverhead
beats RandomMatchedNorm
beats AdamWParallelDirection
beats SNR-only
beats ShuffledPayload/Event
not explained by MLP analog
```

### 10.3 Next-gen MLP candidate

Only if both base and functional success hold, and external fair checks pass, can we call it a next-gen MLP candidate.

---

## 11. 当前建议的优先级

### Highest priority

```text
B1 Activation / RSWAF / ReLU family
B2 Compact RBF / FastKAN family
B3 Orthogonal polynomial family
```

Reason:

```text
These have the best chance to pass MLP-like forward/backward first.
```

### Medium priority

```text
B4 Fourier low-K
B5 Wavelet lite
B6 Local B-spline efficient
B7 Rational/KAT safe
```

Reason:

```text
They may have expression/geometry advantages but need strict efficiency microbench first.
```

### Diagnostic only

```text
Hybrid BSRBF / FC-KAN combo
large-K dense spline/RBF/Fourier
high-degree polynomial
```

Reason:

```text
They are likely too expensive unless single-family efficiency is already solved.
```

---

## 12. 最终执行口径

This plan should be implemented as:

```text
DG-KAN v12.4 Multi-Basis MLP-Like Primitive Screen
+
Basis-Aware Functional Update Diagnostic
```

The key discipline:

```text
1. Efficiency first for every new basis.
2. Expression second.
3. Task stability third.
4. Functional update diagnostic runs in parallel but official promotion is gated.
5. Dataset is only a failure slice, never a tuning condition.
6. If a basis is slow, do not rescue it with functional update.
7. If functional update does not beat controls, do not lower gates.
```

