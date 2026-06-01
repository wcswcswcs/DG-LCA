# DG-KAN v9.2.2 PureFullEdge Compositional Trainability and Margin-Stability 补充实验计划

> 本文件是 v9.2 的第二个补充计划，版本号为 **v9.2.2**。  
> v9.2.1 已经真实执行到 `R8-OptimizerSanityNotEnough`：P4-pass candidate 的 AdamW-only trainability 仍未修复，P7 functional update 没有打开，external fair 也没有打开。  
> v9.2.2 不继续做小修小补，不把 functional update 用来救一个 AdamW-only 不成立的 base，也不退回 transitional linear-SiLU route。  
> 本计划的核心判断是：**当前 P5 blocker 已经不是 basis conditioning，也不是 kernel-native feasibility，而是 FullEdge candidate 的有效表达力 / compositionality / margin dynamics 不足。**  
> 因此 v9.2.2 的目标是：在不离开 PureKAN contract 的前提下，判断当前 FullEdge primitive 是否因为 “单层 additive edge model / active-k correction 太弱 / flattened source 缺少局部结构 / margin-logit 病态” 而无法达到 MLP-match trainability，并据此决定下一步是做 compositional FullEdge、patch-local KANConv primitive，还是淘汰当前 S3-B2-P4 线。

---

## 0. 最新状态与独立判断

### 0.1 v9.2.1 达到了什么

v9.2.1 的执行没有达到 trainability repair。最终 summary artifact 指向：

```text
results/real_rerun_20260506/v921_trainability_repair_summary_20260509T150000Z/
```

最终 route 为：

```text
route = R8-OptimizerSanityNotEnough
best_candidate = S3-B2-P4-F0
best_correction_channel_variant = Z_lr0005_T2_r4
success_v921_trainability_repair = false
success_v921_functional_opened = false
success_v921_external_fair_opened = false
```

已确认的事实：

```text
1. v9.2 的 P5 failure 被真实复现：
   S3-B2-P4-F0 在 T2/rank4/lr0.002 下三任务三 seed 仍是 0/9 near-pass，
   macro delta vs MLP-match = -0.100389。

2. CE tail / margin / logit 诊断确认 failure 不是记录错误：
   mean CEp99 = 82.916358，
   mean train loss head2048 = 1.875384，
   wrong confidence p95 多数为 1.0，
   margin p10 为负。

3. 容量修复没有成功：
   T3/rank4 与 T2/rank8 都保持 P4 pass，
   但 P5 仍 0/9 near-pass，
   macro delta 分别为 -0.113222 与 -0.107222。

4. 小型 lr sanity 有显著改善但仍未达到 near-pass：
   T2/rank4/lr0.0005 保持 P4 pass，
   macro delta 改善到 -0.028222，
   mean CEp99 降到 9.676426，
   但仍是 0/9 near-pass。

5. Identity-only diagnostic 真实训练后远低于 MLP-match：
   macro delta = -0.107778，
   且 P4 kernel-native gate fail。
```

这些事实说明：v9.2.1 并不是完全没有进展。它把 P5 failure 从 “不知道为什么训不起来” 推进为更具体的现象：

```text
1. update scale / logit tail 是真实问题，因为 lr0.0005 显著改善；
2. 但 optimizer sanity 不足以修复 trainability；
3. 增加 T3 或 rank8 没有带来 task gain；
4. identity-only 不能当作 MLP-equivalent；
5. 当前 flattened-source active-k FullEdge candidate 的有效表达力不足。
```

### 0.2 为什么不能相信 route 结论本身就结束

`R8-OptimizerSanityNotEnough` 这个 route 名称容易让人误解为“optimizer 不是问题，所以继续换别的”。独立看数据，更准确的判断是：

$$
\boxed{
\text{lr0.0005 把 macro gap 从约 } -0.10 \text{ 修到 } -0.028 \text{，说明 early update / logit scale 是重要因素；但它不是唯一因素。}
}
$$

如果只是 optimizer 问题，lr / warmup / init sanity 应该能把 gap 推到 near-pass。但实际仍是 0/9 near-pass。因此 P5 blocker 不是纯 optimizer，也不是纯系统；更可能是：

```text
1. 当前 candidate 是过度压缩的 active-k correction；
2. 当前 full-edge 形式在 flattened source 上有效交互能力不足；
3. 当前 architecture 可能更接近 additive edge classifier，而不是 MLP-like compositional learner；
4. 当前 CE tail / margin dynamics 显示 confident wrong samples 没有被修掉；
5. identity channel 和 nonlinear correction channel 没有形成稳定协同。
```

### 0.3 v9.2.2 的核心问题

v9.2.2 要回答：

$$
\boxed{
\text{当前 FullEdge-AdamW 失败，是因为单层/浅层 additive edge model 表达力不足，还是因为 scale/margin 训练病态，还是因为 flattened source 不适合 vision？}
}
$$

只要这个问题没有回答，继续做下面任何事情都会偏：

```text
继续加 functional update；
继续外部公平验证；
继续优化 P4 kernel；
继续换 basis 名字；
继续小 lr sweep；
继续把 transitional route 当终极路线。
```

---

## 1. v9.2.2 总体目标

v9.2.2 的整体目标是建立一个从 **P5 failure 机制诊断** 到 **Pure FullEdge trainability repair** 的完整闭环：

$$
\boxed{
\text{P5 failure}
\rightarrow
\text{additivity / interaction / margin diagnosis}
\rightarrow
\text{compositional FullEdge or source repair}
\rightarrow
\text{P4-pass + P5-near-pass survivor}.
}
$$

v9.2.2 的 minimum success 是：

```text
1. 明确判定当前 S3-B2-P4-F0 的 P5 failure 主要属于哪类机制：
   additive/compositionality fail、margin/logit pathology、source fail、capacity-system tradeoff 或 implementation metric issue。

2. 至少一个不违反 PureKAN contract 的修复候选仍通过 P4 kernel-native gate。

3. 至少一个修复候选达到 P5 near-pass。
```

P5 near-pass 定义为：

$$
Acc_{\text{KAN-AdamW}}\geq Acc_{\text{MLP-match}}-0.01
$$

在至少 6/9 rows 上成立，并且 macro mean gap 满足：

$$
\Delta Acc_{\text{macro}}\geq -0.01.
$$

v9.2.2 strong success 是：

$$
Acc_{\text{KAN-AdamW}}\geq Acc_{\text{MLP-match}}
$$

在 macro mean 上成立，同时仍通过 P4 kernel-native gate：

$$
T_{\text{forward,KAN}}\leq1.25T_{\text{forward,MLP-match}},
$$

$$
T_{\text{backward,KAN}}\leq1.50T_{\text{backward,MLP-match}},
$$

$$
T_{\text{step,KAN}}\leq1.50T_{\text{step,MLP-match}},
$$

$$
M_{\text{peak,KAN}}\leq1.05M_{\text{peak,MLP-match}}.
$$

v9.2.2 不以 functional update 或 external fair success 为目标。只有 P5 near-pass 后，functional update 才能 gated open；只有 P5 pass 或 near-pass + functional useful 后，external fair 才能 open。

---

## 2. 不可违反的 PureKAN contract

v9.2.2 继续执行 v9.2.1 的 contract，并把 compositional route 的边界写清楚。

### 2.1 Clean training contract

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

训练目标必须是：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

P5 阶段只允许 AdamW-equivalent update：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW-equivalent}}.
$$

P5 阶段不允许：

$$
\Delta\theta_{\text{functional}}.
$$

### 2.2 FullEdge equivalence contract

每一层 official FullEdge 必须能写成：

$$
h^{\ell+1}_j=\sum_i\phi^{\ell}_{ij}(h^{\ell}_i).
$$

每条 edge function 必须能写成：

$$
\phi^{\ell}_{ij}(x)
=
\sum_k c^{\ell}_{ij,k}B^{\ell}_k(\tilde{x}).
$$

允许 identity edge-basis channel：

$$
B_0(x)=x.
$$

允许 nonlinear edge-correction channel：

$$
\sum_{k\geq1}c_{ij,k}B_k(\tilde{x}).
$$

禁止外部 residual shortcut：

$$
h^{\ell+1}=F_{\text{edge}}(h^\ell)+h^\ell W_{\text{skip}},
$$

如果 $W_{\text{skip}}$ 是普通 trainable Linear。

禁止普通 MLP hidden path：

$$
h^{\ell+1}=\sigma(h^\ell U)V.
$$

除非它被严格证明等价为 edge coefficient tensor 的合法因式分解：

$$
C_{i,j,k}
=
\sum_r u_{i,r}v_{j,r}a_{k,r},
$$

并且没有引入普通 hidden activation $\sigma(hU)$。

### 2.3 Compositional PureKAN contract

v9.2.2 允许多层 FullEdge composition，因为多层 KAN 本身就是 PureKAN：

$$
h^{1}_j=\sum_i\phi^{0}_{ij}(x_i),
$$

$$
h^{2}_m=\sum_j\phi^{1}_{jm}(h^{1}_j),
$$

$$
\text{logit}_c=\sum_m\phi^{2}_{mc}(h^{2}_m).
$$

这不等于 MLP。关键区别是：每一层的可学习非线性都在 edge function $\phi_{ij}$ 内，而不是 node activation $\sigma(\sum_i w_ix_i)$。

因此，v9.2.2 可以测试 `depth=2/3` 的 FullEdge candidate，但必须记录：

```text
each_layer_full_edge_equivalence_pass = 1
ordinary_mlp_hidden_activation_used = 0
external_residual_shortcut_used = 0
trainable_nonkan_preprocessor_used = 0
```

### 2.4 Fixed patch/local source boundary

固定 patch pooling / local average / local contrast normalization 可以作为 diagnostic source：

```text
fixed_preprocessing_used = 1
trainable_preprocessing_used = 0
```

但 trainable Conv / Linear preprocessing 不允许进入 FC FullEdge official route。若固定 patch source 明显优于 flattened source，route 应转向 KANConv-like patch-local FullEdge primitive，而不是把 trainable Conv 接到 FC FullEdge 前面。

---

## 3. 核心假设

### H1：当前 S3-B2-P4-F0 的根本问题是 additive / interaction capacity 不足

如果当前 candidate 实质上接近单层 additive edge classifier：

$$
y_j=\sum_i\phi_{ij}(x_i),
$$

那么它不含显式 feature interaction：

$$
\frac{\partial^2 y_j}{\partial x_a \partial x_b}=0,\quad a\ne b.
$$

MLP 的隐藏层则可以产生交互项。v9.2.2 要检查当前 candidate 是否缺少 cross-feature interaction，尤其在 KMNIST 这种结构更复杂的数据集上表现为大 gap。

H1 成立标准：

```text
1. 当前 K0 的 empirical cross-feature interaction score 显著低于 MLP-match；
2. K0 在 synthetic interaction probe 上低于 MLP-match；
3. depth-2/3 compositional FullEdge 显著提升 P5 trainability。
```

定量标准：

$$
InteractionScore_{\text{K0}}\leq0.5InteractionScore_{\text{MLP-match}},
$$

且：

$$
Acc_{\text{CompositionalFullEdge}}\geq Acc_{\text{K0}}+0.05.
$$

### H2：lr0.0005 改善说明 margin/logit dynamics 是必要修复，但不是充分修复

v9.2.1 中 lr0.0005 将 macro delta 从约 `-0.100389` 改善到 `-0.028222`，CEp99 从 `82.916358` 降到 `9.676426`。这说明 update scale 与 CE tail 有重要影响。

H2 成立标准：

若 lower lr / output scale / correction init 继续降低 CE tail：

$$
CE_{p99,\text{new}}<0.5CE_{p99,\text{K0}},
$$

但仍满足：

$$
Acc_{\text{new}}<Acc_{\text{MLP-match}}-0.01,
$$

则说明 margin/logit repair 是必要但不足，下一步必须修 architecture capacity 或 source。

### H3：identity edge-basis channel 不等价于 MLP hidden layer

Identity-only diagnostic 远低于 MLP-match，macro delta `-0.107778`，且 P4 fail。H3 认为这是预期现象：单层 identity edge-basis 近似线性分类器，不应被当成 MLP-equivalent。它的用途是诊断 edge-basis channel 是否能提供基础线性 mixing，而不是证明模型等价 MLP。

H3 成立标准：

若 identity-only 低于 MLP-match 但 compositional identity + correction FullEdge 提升明显，则说明问题不是 identity basis 不合法，而是当前 architecture 缺少 composition。

### H4：nonlinear edge-correction channel 当前不是“干扰”，而是容量和组织方式不足

v9.2.1 的 capacity repair T3/r4、T2/r8 没有改善，说明简单增加 rank 或换 T3 不足。H4 假设：非线性 correction 需要更合理的组织方式，例如：

```text
multi-channel T2+T3；
layer-wise correction；
input layer correction + output layer identity；
gated correction coefficient within edge function；
deeper composition；
patch-local source。
```

H4 成立标准：

若多层 / patch-local / organized correction 比单层 rank 增加带来更大 gain：

$$
Acc_{\text{organized-correction}}-Acc_{\text{K0}}\geq0.05,
$$

而 rank-only 仍无效，则支持 H4。

### H5：flattened source 对 vision 不够，fixed patch/local source 可能是必要方向

如果 flattened scalar input 对 vision 的局部结构利用不足，FullEdge FC 可能很难追上 MLP-match。H5 测试固定 patch source是否能显著改善。

H5 成立标准：

$$
Acc_{\text{FixedPatchSource}}-Acc_{\text{FlatSource}}\geq0.03.
$$

如果 H5 成立，v9.2.2 应输出：

```text
next_required_implementation = KANConv-like patch-local FullEdge primitive
```

而不是继续在 flattened FC 上加 basis。

### H6：P4 仍是硬约束，但 P4-pass 不代表 architecture sufficient

所有候选必须通过 P4 kernel-native gate：

$$
T_{\text{forward,KAN}}\leq1.25T_{\text{forward,MLP-match}},
$$

$$
T_{\text{backward,KAN}}\leq1.50T_{\text{backward,MLP-match}},
$$

$$
T_{\text{step,KAN}}\leq1.50T_{\text{step,MLP-match}},
$$

$$
M_{\text{peak,KAN}}\leq1.05M_{\text{peak,MLP-match}}.
$$

但 P4 只是资格门槛，不是成功。P5 trainability 是单独 gate。

---

## 4. Candidate 设计

### 4.1 Baselines

```text
B0-MLP-match:
  同参数量 MLP-match，CE-only，AdamW。

K0-S3-B2-P4-F0-lr0005:
  v9.2.1 当前 best optimizer sanity candidate，
  FullEdge equivalence pass，P4 pass，但 P5 fail。

K0-lr002-repeat:
  原 P5 failure reference。
```

### 4.2 Interaction diagnostics

```text
X0-AdditiveSynthetic:
  synthetic additive target，用于确认 KAN additive layer 能拟合 additive 函数。

X1-PairwiseProductSynthetic:
  synthetic pairwise interaction target，例如 x_a x_b。

X2-XORPatchSynthetic:
  local binary/XOR-like interaction target。

X3-RealDataInteractionProbe:
  对 MNIST/FMNIST/KMNIST batch 做 finite-difference cross-feature interaction score。
```

### 4.3 Compositional FullEdge candidates

所有 candidate 都必须满足每层 FullEdge equivalence。

```text
D1-Depth1-K0:
  当前 single-layer / shallow baseline。

D2-Depth2-IdentityCorrection:
  两层 FullEdge。第一层 identity + nonlinear correction，第二层 edge head。

D3-Depth2-InputCorrectionOnly:
  第一层使用 nonlinear correction，输出层主要 identity edge-basis。

D4-Depth2-OutputCorrectionOnly:
  第一层 identity edge-basis，输出层 nonlinear correction。

D5-Depth3-LightCorrection:
  三层轻量 FullEdge，每层 correction channel 很小。

D6-Depth2-SharedBasisGrouped:
  depth2 + grouped shared basis，避免 dense materialization。

D7-Depth2-LowRankRankSweep:
  depth2 + edge coefficient low-rank r = 2,4,8。
```

### 4.4 Nonlinear edge-correction organization candidates

```text
C1-T2-only:
  当前 T2 对照。

C2-T2T3:
  两个 Chebyshev correction channels。

C3-T2T3-SiLU:
  Chebyshev + SiLU edge-basis channel。

C4-Legendre-P2P3:
  换正交族。

C5-BoundedRational2:
  bounded rational diagnostic，必须先过 NaN/Inf、gradient tail 和 P4。

C6-PiecewiseLinear2:
  non-recursive active-k piecewise linear diagnostic。
```

### 4.5 Source candidates

```text
S3-EdgeAffineNorm:
  当前 survivor source。

S4-FixedPatchPool:
  fixed patch pooling diagnostic source。

S6-FixedPatchPoolEdgeAffineNorm:
  fixed patch/local source + EdgeAffineNorm。

S7-FixedLocalContrast:
  fixed local contrast source，非 trainable。

S8-PatchFlattenMultiChannel:
  fixed patch-to-channel source，模拟 KANConv 输入但不引入 trainable Conv。
```

### 4.6 Scale/margin candidates

只允许小型 sanity，不允许扩大为 optimizer sweep。

```text
Z0-current-lr0005:
  v9.2.1 best sanity。

Z1-output-scale-fanin:
  按 fan-in 理论设置 output scale。

Z2-correction-init-zero:
  correction channel 初始为 0。

Z3-correction-init-small:
  correction channel 小尺度初始化。

Z4-block-update-ratio-capped-diagnostic:
  只作为 diagnostic，检查 update/param 爆炸；若作为 official 需单独论证。
```

---

## 5. 实验阶段

## P0：contract / equivalence / metric sanity audit

### 目标

确保所有 v9.2.2 candidate 不离开 PureKAN，且 P5 acc/loss/margin 统计路径没有不一致。

### 必须记录

```text
candidate_id
candidate_family
depth
source_id
basis_id
parameterization_id
loss_type
label_smoothing
external_teacher_used
self_teacher_used
uses_loss_backward
full_edge_equivalence_pass
each_layer_full_edge_equivalence_pass
ordinary_mlp_hidden_activation_used
external_residual_shortcut_used
ordinary_linear_skip_used
trainable_preprocessor_used
lowrank_edge_factorization_pass
hidden_activation_introduced
materializes_dense_edge_tensor
metric_acc_loss_same_logits
metric_train_head_same_batch
official_eligible
```

### 判断标准

Official candidate 必须满足：

```text
full_edge_equivalence_pass = 1
each_layer_full_edge_equivalence_pass = 1
ordinary_mlp_hidden_activation_used = 0
external_residual_shortcut_used = 0
ordinary_linear_skip_used = 0
trainable_preprocessor_used = 0
uses_loss_backward = 0
label_smoothing = 0
metric_acc_loss_same_logits = 1
```

### 可视化

```text
p0_contract_equivalence_matrix.svg
p0_candidate_depth_lattice.svg
p0_forbidden_path_heatmap.svg
p0_metric_sanity_table.md
```

---

## P1：P5 failure reproduction with interaction and margin diagnostics

### 目标

复现 K0 的 P5 failure，并补充分布、交互、margin、logit、CE tail 诊断。P1 不做模型修复。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
candidate = K0-S3-B2-P4-F0-lr0005 and K0-lr002-repeat
baseline = MLP-match
functional_update = off
epochs = 20
batch_size = 128
```

### 必须记录

```text
dataset
seed
candidate
epoch
train_acc
test_acc
train_loss
test_loss
CE_p50
CE_p90
CE_p99
logit_norm_mean
logit_norm_p95
correct_margin_mean
correct_margin_p10
wrong_confidence_p95
ECE
NLL
interaction_score_diag
interaction_score_offdiag
finite_diff_pair_score
identity_channel_contribution_norm
correction_channel_contribution_norm
correction_to_identity_ratio
grad_norm_identity_channel
grad_norm_correction_channels
update_over_param_identity_channel
update_over_param_correction_channels
```

### 判断标准

P5 failure confirmed：

$$
Acc_{\text{K0}}\leq Acc_{\text{MLP-match}}-0.01
$$

on at least 6/9 rows。

Margin pathology confirmed：

$$
CE_{p99}>5CE_{p50}
$$

or:

$$
WrongConfidence_{p95}>0.90.
$$

Interaction deficit confirmed：

$$
InteractionScore_{\text{K0}}\leq0.5InteractionScore_{\text{MLP-match}}.
$$

### 可视化

```text
p1_acc_loss_curve.svg
p1_ce_tail_quantiles.svg
p1_margin_distribution.svg
p1_wrong_confidence_hist.svg
p1_interaction_score_bar.svg
p1_update_over_param_by_channel.svg
```

---

## P2：synthetic interaction diagnosis

### 目标

在可控合成任务上区分 additive capacity 和 interaction capacity。若当前 FullEdge 单层只能拟合 additive 目标，不能拟合 pairwise/XOR-like 目标，则 P5 failure 的本质就是 compositionality 缺失。

### 数据

构造 synthetic targets：

```text
T0-additive:
  y = sum_i a_i f_i(x_i)

T1-pairwise-product:
  y = sum_{(i,j)} a_{ij} x_i x_j

T2-local-xor:
  binary local XOR / parity-like target

T3-composition:
  y = f_2(sum_i f_{1,i}(x_i))
```

所有 synthetic 只用于机制诊断，不用于 official external success。

### 必须记录

```text
target_type
candidate
depth
train_loss
test_loss
train_acc_or_r2
test_acc_or_r2
interaction_score
fit_R2
P4_step_ratio
P4_memory_ratio
```

### 判断标准

Additive pass：

$$
R^2_{\text{additive}}\geq0.95.
$$

Interaction fail：

$$
R^2_{\text{pairwise}}<0.80
$$

for depth1 but depth2 improves by：

$$
R^2_{\text{depth2}}-R^2_{\text{depth1}}\geq0.10.
$$

If this holds, v9.2.2 should prioritize compositional FullEdge.

### 可视化

```text
p2_synthetic_fit_r2.svg
p2_depth_vs_interaction_fit.svg
p2_additive_vs_pairwise_loss_curve.svg
```

---

## P3：Compositional FullEdge trainability

### 目标

测试 depth2/depth3 Pure FullEdge composition 是否能修复 AdamW-only trainability，同时保持 P4 gate。

### 设置

```text
candidates = D1-D7
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
functional_update = off
epochs = 20
```

### 必须记录

```text
candidate
depth
dataset
seed
params_ratio_vs_mlp_match
forward_ratio
backward_ratio
step_ratio
memory_ratio
train_acc
test_acc
train_loss
test_loss
delta_vs_mlp_match
CE_p99
margin_p10
interaction_score
layer1_activation_rank
layer2_activation_rank
basis_channel_usage
identity_channel_contribution_norm
correction_channel_contribution_norm
full_edge_equivalence_pass
```

### 判断标准

P4 gate：

$$
T_{\text{forward,KAN}}\leq1.25T_{\text{forward,MLP-match}},
$$

$$
T_{\text{backward,KAN}}\leq1.50T_{\text{backward,MLP-match}},
$$

$$
T_{\text{step,KAN}}\leq1.50T_{\text{step,MLP-match}},
$$

$$
M_{\text{peak,KAN}}\leq1.05M_{\text{peak,MLP-match}}.
$$

Compositional repair pass：

$$
Acc_{\text{depth2/3}}\geq Acc_{\text{K0}}+0.05.
$$

P5 near-pass：

$$
Acc_{\text{depth2/3}}\geq Acc_{\text{MLP-match}}-0.01
$$

on at least 6/9 rows。

### 可视化

```text
p3_depth_accuracy_bar.svg
p3_depth_task_system_pareto.svg
p3_depth_interaction_score.svg
p3_activation_rank_by_layer.svg
p3_ce_tail_by_depth.svg
```

---

## P4：Nonlinear edge-correction organization under PureKAN contract

### 目标

如果 depth alone 不够，测试 nonlinear edge-correction channel 的组织方式，而不是只做 rank 增加。所有候选仍必须是 edge-basis channels，不允许外部 residual。

### 设置

```text
candidates = C1-C6
base_depth = best from P3 or D1 if P3 fails
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
functional_update = off
```

### 必须记录

```text
candidate
basis_channels
correction_channel_count
edge_factorization_rank
dataset
seed
P4_forward_ratio
P4_backward_ratio
P4_step_ratio
P4_memory_ratio
train_acc
test_acc
delta_vs_mlp_match
CE_p99
margin_p10
correction_channel_usage_entropy
dominant_correction_channel_fraction
grad_norm_by_basis_channel
update_over_param_by_basis_channel
```

### 判断标准

Organized correction pass：

$$
Acc_{\text{organized-correction}}\geq Acc_{\text{K0}}+0.05.
$$

Correction utilization pass：

$$
DominantCorrectionChannelFraction\leq0.70,
$$

$$
UsageEntropy\geq0.50.
$$

P4 gate must remain pass.

### 可视化

```text
p4_correction_channel_accuracy.svg
p4_correction_channel_usage_heatmap.svg
p4_basis_grad_norm_bar.svg
p4_correction_pareto.svg
```

---

## P5：Fixed source / patch-local diagnostic

### 目标

判断 vision task 是否需要 patch/local source。如果 fixed patch/local source 显著提升，说明下一步应转向 KANConv-like patch-local FullEdge，而不是继续 FC flattened source。

### 设置

```text
source candidates = S3,S4,S6,S7,S8
base candidate = best P3/P4 survivor
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
functional_update = off
```

### 必须记录

```text
source_id
fixed_preprocessing_used
trainable_preprocessing_used
dataset
seed
conditioning_number
dominant_basis_fraction
dead_basis_fraction
source_variance_mean
patch_locality_score
train_acc
test_acc
delta_vs_mlp_match
CE_p99
margin_p10
step_ratio
memory_ratio
```

### 判断标准

Patch source evidence：

$$
Acc_{\text{PatchSource}}\geq Acc_{\text{FlatSource}}+0.03.
$$

Official FC source remains clean only if:

```text
trainable_preprocessing_used = 0
```

If patch source helps but FC full-edge still fails, route should recommend KANConv-like primitive.

### 可视化

```text
p5_source_accuracy_bar.svg
p5_source_condition_vs_accuracy.svg
p5_patch_locality_score.svg
p5_source_task_system_pareto.svg
```

---

## P6：Margin / scale repair for best PureKAN candidate

### 目标

在 best P3/P4/P5 survivor 上修 CE tail 与 margin dynamics。P6 不是 optimizer sweep，而是验证 v9.2.1 中 lr0.0005 的启示：update scale 和 logit scale 是必要但不足的修复。

### 设置

```text
candidate = best P3/P4/P5 survivor
lr = 0.0005, 0.001
correction_channel_init_scale = 0.0, 0.01
output_scale = fanin, current
warmup = none, 5% steps
```

### 必须记录

```text
candidate
lr
correction_channel_init_scale
output_scale
warmup
dataset
seed
early_loss_slope
early_margin_slope
train_acc_final
test_acc_final
train_loss_final
test_loss_final
CE_p99
logit_norm_mean
margin_p10
update_over_param_identity
update_over_param_correction
bad_update_rate
step_ratio
memory_ratio
```

### 判断标准

Margin repair pass：

$$
CE_{p99,\text{new}}<0.5CE_{p99,\text{base}}.
$$

Trainability repair pass：

$$
Acc_{\text{new}}\geq Acc_{\text{base}}+0.03.
$$

If CE tail repairs but accuracy remains below MLP-match by more than 0.01, architecture capacity remains blocker.

### 可视化

```text
p6_margin_scale_heatmap.svg
p6_ce_tail_by_scale.svg
p6_early_margin_slope.svg
p6_update_over_param_trace.svg
```

---

## P7：P5 survivor confirmation and gated functional re-entry decision

### 目标

只在出现 P5 near-pass candidate 后做 final confirmation。P7 决定是否允许重新打开 functional update。

### 选择规则

不得用 test set 选择。先过滤：

```text
FullEdgeEquivalencePass = 1
NoExternalResidualPass = 1
P4KernelNativePass = 1
```

然后按 validation macro score 选 candidate：

$$
Score =
\Delta Acc_{\text{val,macro}}
-
0.5\max(0,StepRatio-1.50)
-
0.5\max(0,MemoryRatio-1.05)
-
0.1\max(0,CE_{p99}-CE_{p99,\text{MLP}}).
$$

### 必须记录

```text
candidate
dataset
seed
val_acc
test_acc
train_acc
val_loss
test_loss
delta_vs_mlp_match
ECE
NLL
CE_p50
CE_p90
CE_p99
margin_p10
interaction_score
step_ratio
memory_ratio
selection_score
selected_by_rule
functional_open_allowed
```

### 判断标准

P5 near-pass：

$$
Acc_{\text{KAN}}\geq Acc_{\text{MLP-match}}-0.01
$$

on at least 6/9 rows，并且：

$$
\Delta Acc_{\text{macro}}\geq -0.01.
$$

Functional open allowed：

```text
functional_open_allowed = 1
```

only if P5 near-pass is achieved.

### 可视化

```text
p7_survivor_score_table.md
p7_dataset_gap_bar.svg
p7_task_system_pareto.svg
p7_functional_open_decision.svg
```

---

## 6. Route decision

### Route cases

```text
R1-PureFullEdgeP5Repaired:
  PureKAN FullEdge candidate reaches P5 near-pass/pass and keeps P4 gate.

R2-CompositionalFullEdgeRequired:
  Depth2/3 improves significantly; depth1 additive architecture is insufficient.

R3-PatchLocalRequired:
  Fixed patch/local source improves significantly; move toward KANConv-like primitive.

R4-MarginScaleNecessaryButInsufficient:
  CE tail/logit scale improves but task gap remains > 0.01.

R5-CapacitySystemTradeoff:
  Higher-capacity edge-correction improves task but breaks P4; custom kernel needed.

R6-InteractionCapacityFail:
  Synthetic/finite-difference interaction tests show current FullEdge lacks feature interaction.

R7-SourceDepthFail:
  Identity/additive/compositional flattened-source routes all fail; flattened FC FullEdge should be deprioritized.

R8-ContractFail:
  Candidate requires external residual, MLP path, trainable preprocessing, teacher, loss modification, fake/proxy, or loss.backward.

R9-NoRepair:
  No candidate improves K0 by at least 0.03 or reaches P5 near-pass.
```

### route_decision.json 必须记录

```text
route
best_candidate
best_depth
best_source
best_basis
best_correction_channel
full_edge_equivalence_pass
no_external_residual_pass
p4_kernel_native_pass
p5_near_pass
p5_pass
interaction_deficit_confirmed
margin_pathology_confirmed
patch_source_evidence
functional_open_allowed
primary_blocker
next_required_implementation
success_v922_trainability_repair
success_v922_functional_opened
success_v922_external_fair_opened
```

---

## 7. Required artifacts

```text
run_manifest.json
contract_equivalence_audit_v922.csv
p1_p5_failure_interaction_margin_diagnostics.csv
p1_loss_margin_logit_trace.csv
p2_synthetic_interaction_diagnostics.csv
p3_compositional_full_edge_trainability.csv
p3_compositional_full_edge_trace.csv
p4_edge_correction_organization.csv
p4_correction_channel_usage.csv
p5_fixed_source_patch_diagnostic.csv
p6_margin_scale_repair.csv
p7_survivor_confirmation.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy:

```text
F1_contract_violation
F2_full_edge_equivalence_fail
F3_external_residual_violation
F4_metric_sanity_fail
F5_p5_failure_not_reproduced
F6_interaction_capacity_fail
F7_compositional_depth_breaks_p4
F8_compositional_depth_no_task_gain
F9_edge_correction_no_gain
F10_patch_source_no_gain
F11_margin_scale_no_gain
F12_p5_near_pass_fail
F13_functional_not_opened
F14_fake_or_proxy_violation
F15_artifact_missing
```

---

## 8. 第一轮执行顺序

v9.2.2 的执行顺序必须固定：

```text
Step 1:
  P0 contract / equivalence / metric sanity。
  先证明所有候选没有 external residual、MLP hidden path、trainable preprocessing。

Step 2:
  P1 复现 K0 P5 failure，并补 interaction / margin / logit / CE tail 诊断。

Step 3:
  P2 做 synthetic interaction diagnostic。
  判断 depth1 additive architecture 是否缺少 feature interaction。

Step 4:
  P3 做 compositional FullEdge trainability。
  这是 v9.2.2 的核心，不得跳过。

Step 5:
  P4 做 nonlinear edge-correction organization。
  不是外部 residual，不是 MLP path，只是 edge-basis channel 组织。

Step 6:
  P5 做 fixed patch/local source diagnostic。
  若 fixed patch source 强提升，则转向 KANConv-like primitive。

Step 7:
  P6 做 margin / scale repair。
  只在 best PureKAN candidate 上做小型 sanity。

Step 8:
  P7 survivor confirmation。
  只有 P5 near-pass 后，才允许 functional update 重新打开。
```

---

## 9. 停止条件

### 成功停止

v9.2.2 minimum success：

```text
P5 near-pass achieved
FullEdgeEquivalencePass = 1
NoExternalResidualPass = 1
P4 kernel-native gate retained
contract clean
no fake/proxy
```

v9.2.2 strong success：

```text
P5 pass achieved
macro mean >= MLP-match
P4 kernel-native gate retained
functional open allowed
```

### 失败停止

```text
1. P5 failure cannot be reproduced；
2. metric sanity shows acc/loss logging inconsistency；
3. any candidate needs external residual / MLP hidden path / trainable preprocessing；
4. depth2/depth3 FullEdge breaks P4 and no kernel plan exists；
5. depth2/depth3 FullEdge gives no task gain over K0；
6. patch/local fixed source gives no task gain；
7. margin/scale repair lowers CE tail but cannot reduce task gap below 0.01；
8. no candidate reaches P5 near-pass；
9. fake/proxy/offload/loss modification violation occurs。
```

---

## 10. 最终解释规则

### Case A：P5 repaired by compositional PureKAN

可以声明：

```text
v9.2.2 repaired AdamW-only trainability through compositional FullEdge PureKAN without leaving PureKAN.
```

仍不能声明 functional advantage 或 external fair success，除非后续 P7/P8 打开并通过。

### Case B：Depth helps but breaks P4

必须声明：

```text
FullEdge requires compositionality, but current kernel-native implementation cannot yet support it; next required implementation is fused compositional FullEdge kernel.
```

### Case C：Patch source helps

必须声明：

```text
Vision task requires patch-local source; next route should be KANConv-like patch-local FullEdge rather than flattened FC FullEdge.
```

### Case D：Margin improves but task gap remains

必须声明：

```text
Update/logit scale pathology was real but not sufficient; architecture capacity remains blocker.
```

### Case E：No repair works

必须声明：

```text
S3-B2-P4 active-k shared edge-correction is kernel-native but not a viable official FullEdge trainability candidate.
```

---

## 11. 最终建议

v9.2.2 的一句话策略是：

$$
\boxed{
\text{不要继续修 P4，也不要用 functional update 救 P5；先证明 Pure FullEdge 需要什么形式的 compositionality 才能达到 AdamW trainability。}
}
$$

当前最重要的问题不是 “再试一个 basis”，也不是 “再调一个 lr”，而是：

```text
1. 当前 depth1 / shallow FullEdge 是否只是 additive classifier？
2. 是否需要 depth2/depth3 compositional FullEdge 才能产生 feature interaction？
3. 是否需要 fixed patch/local source 才适合 vision？
4. margin/logit pathology 是否只是表象，还是主要 blocker？
5. 能否在 PureKAN contract 和 P4 kernel-native gate 内修到 P5 near-pass？
```

只有回答这些问题，v9.2 才能决定下一步是继续 FC FullEdge、转向 KANConv、还是重新设计 kernelized compositional edge primitive。
