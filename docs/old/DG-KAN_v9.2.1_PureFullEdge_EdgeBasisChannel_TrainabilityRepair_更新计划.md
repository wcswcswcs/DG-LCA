# DG-KAN v9.2.1 Pure FullEdge Edge-Basis Channel Trainability Repair 补充实验计划

> 本文件是 v9.2 的补充计划，版本号为 **v9.2.1**，不另起 v9.3。  
> 本版计划专门修正上一版中 “residual” 表述可能导致路线偏移的问题。  
> 本计划明确：**official route 必须仍然是 Clean FullEdge PureKAN，不允许外部 residual shortcut、不允许普通 Linear/MLP 路径、不允许 trainable Conv/Linear preprocessing。**  
> 后续所有 “residual” 相关说法统一改为 **nonlinear edge-correction channel** 或 **edge-basis correction channel**。  
> 允许的结构必须能代数等价写成：
>
> $$
> y_j=\sum_i\phi_{ij}(x_i).
> $$
>
> 其中 identity channel、nonlinear correction channel、low-rank coefficient factorization 都必须属于同一个 edge-function system，而不是外部 MLP/Linear residual path。

---

## 0. 当前状态与本计划的核心纠偏

### 0.1 v9.2 已经推进到哪里

v9.2 的真实执行已经把问题推进到了一个更清楚的位置。v9.1 中 planned basis family 的 dense manual full-edge forward/backward 已补齐，BAS0-BAS9 的 gradcheck 全部通过，部分 basis 能完成 512-sample overfit，但 raw-input conditioning 是 `0/10 basis pass`，没有任何 basis 同时通过 conditioning、fit、gradcheck、overfit。v9.2 通过引入 source-basis 组合推进了这个 blocker，`S3-EdgeAffineNorm + B2-Chebyshev3Residual` 在 MNIST、Fashion-MNIST、KMNIST 和 seeds 0/1/2 上 `9/9` 通过 conditioning，max condition 为 `81.249916`，max dominant fraction 为 `0.662755`，dead fraction 为 `0`。

随后 v9.2 的 P4 也被推进。first-wave 的 `P1 IdentityPlusLowRankResidual` P4 失败，route 是 `R6-ComputeFail-KernelizationRequired`；后续新增 `P3 SharedBasisLowRank` 和 `P4 ActiveKSharedResidual`，并用 compiled forward/backward 与 foreach update 真实重跑 P3/P4。最终 `S3-B2-P4-F0` active-k shared residual 在 hidden4096/rank1 和 hidden4096/rank4 下均真实通过 kernel-native gate。

但是，继续打开 P5 后，AdamW-only trainability 没有通过。三任务三 seed 的 9 行均低于同参数 MLP-match 的 `-0.01` tolerance。最终 route 是：

```text
route = R4-AdamWTrainabilityFail
best_candidate = S3-B2-P4-F0
success_v92_basis_source = true
success_v92_full_edge_trainability = false
success_v92_functional_advantage = false
success_v92_external_fair = false
```

因此，v9.2.1 不应该打开 P6 functional，也不应该打开 P7 external fair。当前唯一正确任务是：**在不偏离 PureKAN contract 的前提下修复或证伪 P5 AdamW-only trainability。**

### 0.2 为什么必须修正 “Residual” 说法

上一版计划使用了 “residual path / residual capacity repair / hybrid residual basis” 这些词。这个说法有歧义。如果 residual 被理解为：

$$
y = F_{\text{KAN}}(x) + xW_{\text{skip}},
$$

或者：

$$
h_{l+1}=h_l+F_{\text{KAN}}(h_l),
$$

并且 $W_{\text{skip}}$ 是普通 Linear/MLP-style trainable block，那么这会直接偏离终极目标。它会变成：

```text
KAN + Linear shortcut
KAN + MLP residual block
KAN + non-KAN trainable path
```

这种结构不能作为 official route。

本计划允许的是另一种东西：**edge function 内部的 identity basis channel 与 nonlinear edge-correction channel**。也就是说，每条 edge 的函数可以写成：

$$
\phi_{ij}(x_i)
=
c_{ij,0}B_0(x_i)
+
\sum_{k=1}^{K}c_{ij,k}B_k(x_i),
$$

其中：

$$
B_0(x)=x.
$$

如果把它写成：

$$
\phi_{ij}(x_i)=w^{(0)}_{ij}\tilde{x}_i+\rho_{ij}(\tilde{x}_i),
$$

这里的 $w^{(0)}_{ij}\tilde{x}_i$ 不是外部 linear skip，而是 edge function 的 identity basis channel；$\rho_{ij}$ 也不是外部 residual block，而是同一条 edge 内部的 nonlinear correction channels。整层仍然是：

$$
y_j=\sum_i\phi_{ij}(x_i).
$$

因此，本计划从措辞和 contract 上做如下修改：

```text
禁止使用 external residual shortcut 作为 official route；
禁止普通 Linear/MLP hidden path；
禁止 trainable Conv/Linear preprocessing；
允许 edge-internal identity basis channel；
允许 edge-internal nonlinear correction channel；
允许 edge coefficient tensor 的低秩因式分解；
允许 fixed diagnostic preprocessing，但不得算作 trainable PureKAN path；
如果 patch/local trainable source 有必要，必须进入 KANConv 路线，而不是 FC FullEdge official route。
```

### 0.3 v9.2.1 的定位

v9.2.1 的核心目标是：

$$
\boxed{
\text{在 FullEdge PureKAN contract 下修复 P5 AdamW-only trainability。}
}
$$

它不是：

```text
不是回到 transitional linear-SiLU route；
不是加入 MLP residual；
不是做 optimizer sweep；
不是用 functional update 救一个 AdamW-only 不成立的 base；
不是先跑 external fair。
```

它是：

```text
1. 保持 y_j = sum_i phi_ij(x_i)；
2. 检查 identity edge-basis channel 是否成立；
3. 检查 nonlinear edge-correction channels 是否容量不足或干扰 task；
4. 检查 edge coefficient low-rank factorization 是否过窄；
5. 检查 source/normalization 是否压掉 task-useful variation；
6. 检查 logit/margin/CE tail 是否病态；
7. 在 P4 kernel-native gate 内修复 P5。
```

---

## 1. v9.2.1 整体目标

v9.2.1 的整体目标是建立一个完整、可审计、不偏离 PureKAN 的 P5 trainability 修复闭环：

$$
\boxed{
\text{P4-pass FullEdge primitive}
\rightarrow
\text{AdamW-only trainability diagnostic}
\rightarrow
\text{edge-basis channel / coefficient factorization / source / scale repair}
\rightarrow
\text{P5 survivor confirmation}.
}
$$

v9.2.1 的 minimum success 是：

```text
1. P5 failure 可复现；
2. 指标 sanity 通过；
3. 至少一个修复候选仍满足 FullEdge equivalence 和 P4 kernel-native gate；
4. 至少一个修复候选达到 P5 near-pass。
```

P5 near-pass 定义为：

$$
Acc_{\text{KAN-AdamW}}\geq Acc_{\text{MLP-match}}-0.01
$$

在至少 6/9 rows 上成立，并且 macro mean gap 满足：

$$
\Delta Acc_{\text{macro}}\geq -0.01.
$$

v9.2.1 strong success 是：

$$
Acc_{\text{KAN-AdamW}}\geq Acc_{\text{MLP-match}}
$$

在 macro mean 上成立，同时仍通过 kernel-native gate：

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

本阶段不以 functional update 成功为目标。Functional update 只有在 P5 near-pass 后才允许作为 gated diagnostic 打开。

---

## 2. 不可违反的 PureKAN contract

v9.2.1 的 official candidate 必须同时通过以下 contract。

### 2.1 Clean training contract

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

也不允许：

$$
L=CE+\lambda L_{\text{geo}}.
$$

### 2.2 FullEdge equivalence contract

official candidate 必须能写成：

$$
y_j=\sum_i\phi_{ij}(x_i).
$$

每条 edge function 必须能写成：

$$
\phi_{ij}(x_i)
=
\sum_k c_{ij,k}B_k(\tilde{x}_i).
$$

其中 $B_0(x)=x$ 是允许的 identity edge-basis channel。非线性部分必须是 edge-basis correction channel：

$$
\sum_{k\geq1} c_{ij,k}B_k(\tilde{x}_i).
$$

### 2.3 NoExternalResidualPass

以下结构不允许进入 official route：

$$
y=F_{\text{edge}}(x)+xW_{\text{skip}},
$$

如果 $W_{\text{skip}}$ 是外部普通 trainable Linear。

以下结构也不允许：

```text
ordinary Linear shortcut outside edge function
ordinary MLP hidden layer
trainable Conv/Linear preprocessing
trainable source projection not expressible as edge function
KAN output + non-KAN residual branch
```

### 2.4 LowRankEdgeCoefficientFactorizationPass

low-rank 只能作为 edge coefficient tensor 的因式分解。FullEdge dense coefficient 可以写成：

$$
C_{i,j,k}.
$$

允许的低秩形式是：

$$
C_{i,j,k}
=
\sum_{r=1}^{R}u_{i,r}v_{j,r}a_{k,r}.
$$

这种写法仍然等价于：

$$
y_j=\sum_i\sum_k C_{i,j,k}B_k(\tilde{x}_i).
$$

但是如果实现变成：

$$
r=\sigma(xU),
$$

$$
y=rV,
$$

其中 $r$ 是普通 hidden activation，而不是 edge coefficient factorization 下的 basis contraction，那么这就是 MLP-like hidden projection，不允许进入 official route。

因此每个 low-rank candidate 必须记录：

```text
coefficient_factorization_equivalent = 1
hidden_activation_introduced = 0
ordinary_linear_hidden_path = 0
```

### 2.5 Fixed preprocessing boundary

固定 preprocessing 可以作为 diagnostic source，例如：

```text
fixed per-feature standardization
fixed clipping
fixed patch pooling
fixed local average
fixed local contrast normalization
```

但 trainable preprocessing 不允许进入 FC FullEdge official route。若发现 vision task 必须依赖 trainable patch/local source，则 route 应转向 KANConv，而不是把 trainable Conv/Linear preprocessor 接到 FC FullEdge 前面。

---

## 3. 核心假设

### H0：P5 failure 是真实可复现的，不是 metric artifact

H0 要求重新复现 `S3-B2-P4-F0` 的 P5 failure，并补充更细诊断。H0 成立标准：

$$
Acc_{\text{KAN}}\leq Acc_{\text{MLP-match}}-0.01
$$

在至少 6/9 rows 上仍成立，同时 no-fake/no-proxy/contract 全部通过。

如果 H0 不成立，即 P5 repeat 接近通过，则先修 reproduction protocol，而不是改模型。

### H1：高 train acc 但高 CE loss 暗示 margin / logit-scale 或 tail-risk 问题

H1 关注 P5 中出现的 train acc 不低但 train loss 高的现象。若模型不是完全不会分类，而是 margin 不够、错误样本 CE tail 极大或 logit scale 不稳，就需要优先修初始化与 update scale，而不是继续加基函数。

H1 成立标准：

$$
TrainAcc>0.85
$$

且：

$$
TrainLoss>1.5.
$$

或者：

$$
CE_{p99}>5CE_{p50}.
$$

或者：

$$
Margin_{p10}<0.
$$

或者：

$$
WrongConfidence_{p95}>0.90.
$$

### H2：identity edge-basis channel 必须单独验证

Hybrid FullEdge 的合法形式是：

$$
\phi_{ij}(x)=c_{ij,0}B_0(x)+\sum_{k\geq1}c_{ij,k}B_k(x),
$$

其中：

$$
B_0(x)=x.
$$

H2 假设 identity edge-basis channel 应提供稳定线性混合。如果 identity-only 都远低于 MLP-match，则问题不在 nonlinear edge-correction，而在 architecture/source/depth 本身。

H2 成立标准：

Identity-only candidate 若满足：

$$
Acc_{\text{IdentityOnly}}\geq Acc_{\text{MLP-match}}-0.02,
$$

则 identity edge-basis channel 近似有效。

若：

$$
Acc_{\text{IdentityOnly}}<Acc_{\text{MLP-match}}-0.05,
$$

说明当前 FullEdge source/depth/architecture 不是 MLP-equivalent，不能把 P5 failure 归咎于 nonlinear basis。

### H3：nonlinear edge-correction channel 可能容量不足，也可能干扰 task

H3 检查 nonlinear edge-correction channels 的作用。它不是 external residual path，而是 edge function 内部的 $B_k$ channels。

若：

$$
Acc_{\text{Identity+Correction}}<Acc_{\text{IdentityOnly}}-0.02,
$$

说明 nonlinear correction channel 干扰 task。

若：

$$
Acc_{\text{Identity+Correction}}>Acc_{\text{IdentityOnly}}+0.02,
$$

说明 nonlinear edge-correction channel 有正贡献。

若所有 correction capacity variants 都不提升，则当前 basis / parameterization 不是有效 task correction。

### H4：当前 active-k shared edge-correction 容量过低

P4 为了过系统 gate，将 Chebyshev nonlinear correction 压到 single active channel。H4 假设 P5 failure 的主因是有效 nonlinear edge-correction capacity 过低。

H4 成立标准：

在保持 FullEdge equivalence 和 P4 pass 的前提下，增加 active correction channels / factorization rank / grouped correction 后：

$$
Acc_{\text{candidate}}\geq Acc_{\text{S3-B2-P4-F0}}+0.05
$$

on macro mean，或至少在 KMNIST 上提升：

$$
Acc_{\text{candidate,KMNIST}}\geq Acc_{\text{S3-B2-P4-F0,KMNIST}}+0.05.
$$

### H5：EdgeAffineNorm 修复了 conditioning，但可能压掉 task-useful variation

v9.2 的 survivor source 是 S3-EdgeAffineNorm。H5 假设：S3 让 basis 数值条件变健康，但可能过度标准化，损失 raw/local contrast 信息。

H5 成立标准：

如果 fixed PatchPool 或 PatchPool + EdgeAffineNorm 相比 flattened EdgeAffineNorm 提升：

$$
Acc_{\text{PatchSource}}-Acc_{\text{FlatS3}}\geq0.03,
$$

则说明 vision task 需要 local source，下一步应转向 KANConv-like patch-local edge-function primitive。

### H6：P4 system solution 可能过度压缩，需要寻找 capacity-system Pareto

H6 假设：当前 `S3-B2-P4-F0` 通过 P4 是因为参数化被压得过窄。需要寻找仍能 P4 pass 的更大 edge-correction rank / channel / grouped capacity。

H6 成立标准：

存在 candidate 满足：

$$
StepRatio\leq1.50,
$$

$$
MemoryRatio\leq1.05,
$$

且：

$$
Acc_{\text{candidate}}\geq Acc_{\text{S3-B2-P4-F0}}+0.05.
$$

若所有容量增强候选都 P4 fail，则说明需要更深 kernel work，而不是简单增加 capacity。

### H7：P5 failure 不应通过大规模 optimizer sweep 解决

v9.2.1 只允许小型、预注册 optimizer sanity，不允许把 lr sweep 当主线。

允许矩阵：

```text
lr = 0.0005, 0.001, 0.002
correction_channel_init_scale = 0.0, 0.01, 0.05
warmup = none, 5% steps
output_scale = current, mlp_logit_matched
```

若只有某个组合有效，必须解释机制，例如 correction channel early dynamics、logit scale 或 update/param ratio，而不是写成“optimizer 调好了”。

---

## 4. Candidate 设计

### 4.1 Baselines

```text
B0-KB-MLP-match:
  同参数量 MLP-match，AdamW，CE-only。

K0-S3-B2-P4-F0:
  v9.2 当前 best P4-pass FullEdge active-k shared edge-correction candidate。

K0-repeat:
  P5 failure reproduction candidate。
```

### 4.2 Edge-basis channel ablation candidates

```text
I0-IdentityEdgeOnly:
  只保留 B0(x)=x identity edge-basis channel。

I1-IdentityEdgeOnlySameParam:
  保持参数量接近 K0，通过 legal edge coefficient factorization 增加参数，不引入 MLP hidden path。

I2-IdentityPlusFrozenCorrection:
  nonlinear edge-correction channels 存在但冻结，检查 forward contribution 是否扰动。

I3-CorrectionOnly:
  去掉 identity channel，只保留 nonlinear edge-correction channels。

I4-IdentityPlusTrainableCorrection:
  当前 K0 等价 candidate。
```

所有 I candidates 都必须满足：

$$
y_j=\sum_i\phi_{ij}(x_i).
$$

### 4.3 Nonlinear edge-correction capacity candidates

```text
C1-Chebyshev-T2-only:
  当前 active T2 correction 对照。

C2-Chebyshev-T3-only:
  检查三阶 Chebyshev correction。

C3-Chebyshev-T2T3:
  两个 active correction channels。

C4-Chebyshev-T1T2T3:
  三个 Chebyshev correction channels。

C5-Chebyshev-T2T3-SiLU:
  添加 SiLU edge-basis correction channel。

C6-Legendre-P2P3:
  换正交族，检查 Chebyshev 特异性。

C7-Chebyshev-T2T3-rank8:
  增加 edge coefficient factorization rank。

C8-GroupedEdgeCorrection-g4:
  分组 output edge-correction，提高 capacity 但控制 system。
```

所有 C candidates 必须先通过：

```text
FullEdgeEquivalencePass
NoExternalResidualPass
LowRankEdgeCoefficientFactorizationPass
P4 kernel-native gate
```

P4 不过不得进入 P5。

### 4.4 Source / normalization candidates

```text
S1-StdClip-BestCorrection:
  per-feature standardization + clipping。

S2-FanInNorm-BestCorrection:
  fan-in normalized input。

S3-EdgeAffineNorm-BestCorrection:
  当前 survivor source。

S4-FixedPatchPool-BestCorrection:
  fixed patch pooling source，只作为 FC diagnostic source，不是 trainable Conv。

S6-FixedPatchPoolEdgeAffineNorm-BestCorrection:
  fixed patch/local source 后再 EdgeAffineNorm。
```

如果 S4/S6 大幅提升，route 应指向 KANConv-like patch-local edge-function primitive，而不是在 FC FullEdge 中加 trainable Conv/Linear source。

### 4.5 Initialization / scale sanity candidates

```text
Z0-current-init:
  当前初始化。

Z1-zero-correction-init:
  nonlinear correction coefficient 初始为 0，仅 identity channel 起步。

Z2-small-correction-init-001:
  correction init scale = 0.01。

Z3-small-correction-init-005:
  correction init scale = 0.05。

Z4-output-scale-calibrated:
  output/logit scale 初始化按 MLP-match logit norm 对齐。
```

这些是初始化，不是 loss 修改。

### 4.6 Gated functional candidates

默认不打开。只有 P5 near-pass 后才允许：

```text
F0-AdamW-only
F1-CorrectionChannelOnlyFunctional
F2-HighCurvatureCorrectionFunctional
F3-NoOpMatchedOverhead
F4-RandomDirection
```

Functional update 只能作用在 nonlinear edge-correction channels 或 high-curvature channels，不允许破坏 identity edge-basis channel。

---

## 5. 实验阶段

## P0：contract / equivalence audit

### 目标

确保 v9.2.1 不因为修 trainability 而偏离 PureKAN。P0 必须在所有 candidate 上检查 FullEdge 等价、低秩因式分解合法性、无外部 residual、无 non-KAN trainable path。

### 必须记录

```text
candidate_id
candidate_family
source_id
basis_id
parameterization_id
functional_id
loss_type
label_smoothing
external_teacher_used
self_teacher_used
geometry_loss_used
uses_loss_backward
fake_data_used
proxy_row_used
full_edge_equivalence_pass
edge_function_formula
identity_edge_basis_channel_used
nonlinear_edge_correction_channels
external_residual_shortcut_used
ordinary_linear_skip_used
ordinary_mlp_hidden_path_used
trainable_preprocessor_used
lowrank_is_edge_coefficient_factorization
hidden_activation_introduced
materializes_dense_edge_tensor
official_eligible
```

### 判断标准

Official candidate 必须满足：

```text
loss_type = CE
label_smoothing = 0
external_teacher_used = 0
self_teacher_used = 0
uses_loss_backward = 0
full_edge_equivalence_pass = 1
external_residual_shortcut_used = 0
ordinary_linear_skip_used = 0
ordinary_mlp_hidden_path_used = 0
trainable_preprocessor_used = 0
lowrank_is_edge_coefficient_factorization = 1
hidden_activation_introduced = 0
```

### 可视化

```text
p0_contract_equivalence_heatmap.svg
p0_candidate_lattice_purekan.svg
p0_forbidden_path_matrix.svg
p0_lowrank_equivalence_diagram.svg
```

---

## P1：P5 failure reproduction and expanded diagnostics

### 目标

复现 `S3-B2-P4-F0` 的 P5 failure，并补齐 loss、margin、logit、CE tail、gradient、edge-basis channel contribution 诊断。P1 不做修复，只定位 failure 形态。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 9984
test_size = 2000
epochs = 20
batch_size = 128
candidate = S3-B2-P4-F0
baseline = same-param MLP-match
functional_update = off
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
correct_logit_mean
top_wrong_logit_mean
correct_margin_mean
correct_margin_p10
wrong_confidence_p95
ECE
NLL
grad_norm_identity_channel
grad_norm_correction_channels
grad_norm_output
update_over_param_identity_channel
update_over_param_correction_channels
update_over_param_output
identity_channel_contribution_norm
correction_channel_contribution_norm
correction_to_identity_ratio
step_ratio
memory_ratio
```

### 判断标准

P5 failure reproduction 成立：

$$
Acc_{\text{KAN}}\leq Acc_{\text{MLP-match}}-0.01
$$

在至少 6/9 rows 上成立。

Margin/logit pathology 成立：

$$
TrainAcc>0.85
$$

且：

$$
TrainLoss>1.5
$$

或：

$$
CE_{p99}>5CE_{p50}.
$$

### 可视化

```text
p1_train_test_acc_curve.svg
p1_train_test_loss_curve.svg
p1_ce_quantile_curve.svg
p1_margin_distribution.svg
p1_logit_norm_distribution.svg
p1_gradient_update_channel_norms.svg
p1_identity_vs_correction_contribution_ratio.svg
```

---

## P2：identity edge-basis / nonlinear edge-correction ablation

### 目标

判断 P5 failure 是 identity edge-basis channel 不成立、nonlinear edge-correction channel 不成立，还是二者耦合有问题。

### 设置

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
epochs = 20
functional_update = off
candidates = I0,I1,I2,I3,I4
```

### 必须记录

```text
candidate
dataset
seed
params
params_ratio_vs_mlp_match
forward_ratio
backward_ratio
step_ratio
memory_ratio
train_acc_final
test_acc_final
train_loss_final
test_loss_final
delta_vs_mlp_match
identity_channel_norm
correction_channel_norm
correction_to_identity_ratio
correct_margin_p10
CE_p99
ECE
NLL
full_edge_equivalence_pass
no_external_residual_pass
```

### 判断标准

Identity edge-basis sufficient：

$$
Acc_{\text{IdentityOnly}}\geq Acc_{\text{MLP-match}}-0.02.
$$

Correction channel interference：

$$
Acc_{\text{Identity+Correction}}<Acc_{\text{IdentityOnly}}-0.02.
$$

Correction channel useful：

$$
Acc_{\text{Identity+Correction}}>Acc_{\text{IdentityOnly}}+0.02.
$$

Identity source/depth fail：

$$
Acc_{\text{IdentityOnly}}<Acc_{\text{MLP-match}}-0.05.
$$

### 可视化

```text
p2_identity_correction_ablation_bar.svg
p2_delta_vs_mlp_by_channel_ablation.svg
p2_correction_to_identity_ratio.svg
p2_margin_by_channel_ablation.svg
p2_task_system_pareto_channel_ablation.svg
```

---

## P3：nonlinear edge-correction capacity repair under P4 gate

### 目标

在不破坏 P4 kernel-native gate、也不引入外部 residual/MLP path 的前提下，恢复 nonlinear edge-correction capacity。

### 设置

```text
candidates = C1,C2,C3,C4,C5,C6,C7,C8
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
epochs = 20
functional_update = off
```

### 必须记录

P4 metrics：

```text
candidate
basis_channels
edge_correction_rank
group_count
params_ratio_vs_mlp_match
forward_time_ratio
backward_time_ratio
step_ratio
memory_ratio
forward_FLOPs_ratio
backward_FLOPs_ratio
kernel_count
small_kernel_count
kernel_native_pass
full_edge_equivalence_pass
lowrank_edge_factorization_pass
```

P5 metrics：

```text
candidate
dataset
seed
train_acc
test_acc
train_loss
test_loss
delta_vs_mlp_match
CE_p99
correct_margin_p10
ECE
NLL
identity_channel_contribution_norm
correction_channel_contribution_norm
grad_norm_correction_channels
update_over_param_correction_channels
```

### 判断标准

P4 pass：

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

Capacity repair pass：

$$
Acc_{\text{candidate}}\geq Acc_{\text{K0}}+0.05
$$

on macro mean，或者：

$$
Acc_{\text{candidate,KMNIST}}\geq Acc_{\text{K0,KMNIST}}+0.05.
$$

P5 near-pass：

$$
Acc_{\text{candidate}}\geq Acc_{\text{MLP-match}}-0.01
$$

on at least 6/9 rows。

### 可视化

```text
p3_correction_capacity_vs_accuracy.svg
p3_correction_capacity_vs_step_memory.svg
p3_pareto_task_vs_kernel_gate.svg
p3_margin_gain_by_correction_channel.svg
p3_edge_correction_rank_heatmap.svg
```

---

## P4：source / normalization repair

### 目标

检查 EdgeAffineNorm 是否只是让 basis conditioning 变好，但压掉了 task-useful variation；同时测试 fixed patch/local source 是否能改善 vision task trainability。任何 trainable patch source 都不得进入 FC FullEdge official route。

### 设置

使用 P3 的 best correction-capacity candidate，替换 source：

```text
S1-StdClip
S2-FanInNorm
S3-EdgeAffineNorm
S4-FixedPatchPool
S6-FixedPatchPoolEdgeAffineNorm
```

### 必须记录

```text
source_id
dataset
seed
conditioning_number
dominant_basis_fraction
dead_basis_fraction
train_acc
test_acc
delta_vs_mlp_match
train_loss
test_loss
CE_p99
margin_p10
source_variance_mean
source_variance_p10
source_variance_p90
patch_locality_score
step_ratio
memory_ratio
fixed_preprocessing_used
trainable_preprocessing_used
```

### 判断标准

Source repair pass：

$$
Acc_{\text{source}}\geq Acc_{\text{S3}}+0.03.
$$

Patch/local source evidence：

$$
Acc_{\text{PatchSource}}\geq Acc_{\text{FlatSource}}+0.03.
$$

同时：

$$
trainable\_preprocessing\_used=0.
$$

P4 gate 仍需通过：

$$
StepRatio\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p4_source_accuracy_bar.svg
p4_source_condition_vs_accuracy.svg
p4_source_variance_distribution.svg
p4_patchpool_vs_flat_margin.svg
p4_source_task_system_pareto.svg
```

---

## P5：initialization and update-scale sanity

### 目标

确认 P5 failure 是否由 correction channel 初始化、logit scale 或 update/param ratio 造成。P5 是 sanity，不是 optimizer sweep。不得把 P5 扩大成调参刷结果。

### 设置

只允许预注册小矩阵：

```text
lr = 0.0005,0.001,0.002
correction_channel_init_scale = 0.0,0.01,0.05
warmup = none,5% steps
output_scale = current,mlp_logit_matched
```

候选使用 P3/P4 的 best survivor。

### 必须记录

```text
lr
correction_channel_init_scale
warmup
output_scale
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
update_over_param_identity_channel
update_over_param_correction_channels
update_over_param_output
bad_update_rate
step_ratio
memory_ratio
```

### 判断标准

Scale repair evidence：

如果某个初始化 / scale 设置带来：

$$
Acc_{\text{new}}-Acc_{\text{current}}\geq0.03,
$$

并且：

$$
CE_{p99,\text{new}}<0.5CE_{p99,\text{current}},
$$

则认为 correction channel scale / logit scale 是主要 blocker。

如果所有预注册组合均仍低于 MLP-match 超过 5%，则 P5 failure 不是简单 optimizer / init 问题。

### 可视化

```text
p5_lr_scale_heatmap.svg
p5_early_loss_slope.svg
p5_logit_norm_by_init.svg
p5_update_over_param_trace.svg
p5_ce_tail_by_init.svg
```

---

## P6：P4+P5 survivor confirmation

### 目标

确认经过 P1-P5 诊断后是否存在真正的 AdamW-only trainable FullEdge survivor。

### 设置

只允许使用预注册选择规则：

```text
先过滤 FullEdgeEquivalencePass；
再过滤 NoExternalResidualPass；
再过滤 P4 kernel-native pass；
再按 macro trainability score 选择；
不得用 test set 选择 candidate。
```

Score：

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
ECE
NLL
delta_vs_mlp_match
step_ratio
memory_ratio
forward_ratio
backward_ratio
CE_p50
CE_p90
CE_p99
margin_p10
selection_score
selected_by_rule
full_edge_equivalence_pass
no_external_residual_pass
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

P5 pass：

$$
Acc_{\text{KAN}}\geq Acc_{\text{MLP-match}}
$$

on macro mean，并且 P4 gate pass。

### 可视化

```text
p6_survivor_score_table.md
p6_dataset_gap_bar.svg
p6_train_val_test_curve.svg
p6_task_system_pareto.svg
p6_ce_margin_summary.svg
```

---

## P7：gated functional update re-entry

### 目标

只有 P6 near-pass 后才打开。验证 functional update 是否在 trainable FullEdge base 上带来 geometry、robustness、convergence 优势。Functional update 不允许用于拯救 P5 失败 base。

### 设置

```text
Functional candidates:
  F0-AdamW-only
  F1-CorrectionChannelOnlyFunctional
  F2-HighCurvatureCorrectionFunctional
  F3-NoOpMatchedOverhead
  F4-RandomDirection
```

### 必须记录

```text
functional_mode
dataset
seed
test_acc
delta_vs_adamw
curvature_ratio
jacobian_norm_ratio
local_lipschitz_ratio
ECE_delta
NLL_delta
robustness_proxy
bad_step_rate
holdout_descent_ratio
functional_update_time_ratio
functional_event_count
identity_channel_modified
correction_channels_modified
```

### 判断标准

Functional causality：

$$
R_{\text{curv,Functional}}<R_{\text{curv,NoOp}},
$$

$$
R_{\text{curv,Functional}}<R_{\text{curv,Random}},
$$

$$
Acc_{\text{Functional}}\geq Acc_{\text{NoOp}}-0.002.
$$

Functional useful：

$$
R_{\text{curv,Functional}}\leq0.90R_{\text{AdamW}},
$$

$$
Acc_{\text{Functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

Functional update 必须满足：

```text
identity_channel_modified = 0
correction_channels_modified = 1
```

### 可视化

```text
p7_functional_curvature_bar.svg
p7_functional_task_geometry_pareto.svg
p7_noop_random_control_matrix.svg
p7_functional_event_timeline.svg
p7_identity_vs_correction_update_trace.svg
```

---

## 6. Route decision

### Route cases

```text
R1-PureFullEdgeTrainabilityRepaired:
  P4-pass FullEdge candidate reaches P5 near-pass or pass,
  with FullEdgeEquivalencePass and NoExternalResidualPass.

R2-PureFullEdgeNearPassButNotFull:
  Candidate reaches -0.01 tolerance but not macro >= MLP.

R3-EdgeCorrectionCapacityBlocker:
  More nonlinear edge-correction capacity improves task,
  but P4 gate breaks.

R4-SourceBlocker:
  Patch/local or alternative source needed;
  flat source remains weak.

R5-IdentityEdgeBasisBlocker:
  identity edge-basis channel cannot approach MLP-match,
  indicating source/depth/architecture problem.

R6-EdgeCorrectionInterference:
  identity channel works but nonlinear correction hurts task,
  requiring correction scale / gating redesign.

R7-MarginLogitPathology:
  high train acc + high CE tail dominates;
  need scale/margin dynamics repair.

R8-OptimizerSanityNotEnough:
  small lr/init/warmup matrix cannot recover trainability.

R9-NoTrainabilityRepair:
  all repairs fail;
  current active-k shared edge-correction architecture should be retired or moved to patch/conv primitive.

R10-ContractOrArtifactFail:
  no-teacher/no-loss/no-fake/manual/PureKAN contract or required artifacts fail.
```

### route_decision.json 必须记录

```text
route
best_candidate
best_source
best_basis
best_parameterization
best_correction_channel_variant
best_init_variant
full_edge_equivalence_pass
no_external_residual_pass
lowrank_edge_factorization_pass
p4_kernel_native_pass
p5_near_pass
p5_pass
functional_opened
functional_pass
primary_blocker
next_required_implementation
success_v921_trainability_repair
success_v921_functional_opened
success_v921_external_fair_opened
```

---

## 7. Required artifacts

```text
run_manifest.json
contract_equivalence_audit_v921.csv
p1_p5_failure_reproduction_diagnostics.csv
p1_loss_margin_logit_trace.csv
p2_identity_correction_channel_ablation.csv
p3_edge_correction_capacity_kernel_gate.csv
p3_edge_correction_capacity_trainability.csv
p4_source_repair_conditioning.csv
p4_source_repair_trainability.csv
p5_init_update_scale_sanity.csv
p6_survivor_confirmation.csv
p7_functional_reentry_if_opened.csv
p7_functional_event_trace_if_opened.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_full_edge_equivalence_fail
F3_external_residual_violation
F4_lowrank_not_edge_factorization
F5_p5_failure_not_reproduced
F6_metric_sanity_fail
F7_identity_edge_basis_fail
F8_edge_correction_interference
F9_capacity_repair_breaks_p4
F10_capacity_repair_no_task_gain
F11_source_repair_no_gain
F12_margin_logit_pathology
F13_optimizer_sanity_no_repair
F14_p5_near_pass_fail
F15_functional_reentry_not_opened
F16_functional_causality_fail
F17_fake_or_proxy_violation
F18_artifact_missing
```

---

## 8. 第一轮执行顺序

v9.2.1 的执行顺序必须固定：

```text
Step 1:
  P0 contract / equivalence audit。
  所有 candidate 先证明没有 external residual shortcut 或 non-KAN path。

Step 2:
  P1 复现 P5 failure，并补全 loss / margin / logit / CE tail / gradient / edge-basis channel contribution 诊断。

Step 3:
  P2 做 identity edge-basis / nonlinear edge-correction channel ablation。
  先判断 identity edge-basis channel 是否能接近 MLP-match。

Step 4:
  P3 做 nonlinear edge-correction capacity repair。
  每个 capacity candidate 必须先过 P4 kernel-native gate。

Step 5:
  P4 做 source / normalization repair。
  特别检查 fixed patch/local source 是否显著优于 flat source。

Step 6:
  P5 做小型 init / update-scale sanity。
  不允许扩大成 optimizer sweep。

Step 7:
  P6 survivor confirmation。
  用预注册 score 选择，不用 test set 挑 candidate。

Step 8:
  只有 P6 near-pass 后，才打开 P7 functional update。
```

---

## 9. 停止条件

### 成功停止

v9.2.1 minimum success：

```text
P5 near-pass achieved
FullEdgeEquivalencePass = 1
NoExternalResidualPass = 1
P4 kernel-native gate retained
contract clean
no fake/proxy
```

v9.2.1 strong success：

```text
P5 pass achieved
macro mean >= MLP-match
P4 kernel-native gate retained
P7 functional opens and improves geometry without task drop
```

### 失败停止

```text
1. P5 failure cannot be reproduced；
2. metric sanity shows acc/loss logging inconsistency；
3. any candidate requires external residual / ordinary Linear skip / MLP hidden path；
4. identity edge-basis channel is far below MLP and fixed patch/local source does not help；
5. all nonlinear edge-correction capacity repairs either break P4 or fail to improve task by >= 0.03；
6. margin/logit pathology persists under scale/init repair；
7. no candidate reaches P5 near-pass；
8. any teacher/loss/fake/proxy/offload violation occurs。
```

---

## 10. 最终解释规则

### Case A：P5 repaired under PureKAN contract

可以声明：

```text
v9.2.1 repaired AdamW-only trainability for the P4-pass FullEdge primitive without leaving PureKAN.
```

但仍不能声明 functional advantage 或 external fair success，除非 P7 和后续 external validation 打开并通过。

### Case B：capacity repair helps but breaks P4

必须声明：

```text
Current FullEdge primitive has a capacity-system tradeoff; next work requires kernelization for the higher-capacity edge-correction path.
```

### Case C：identity edge-basis channel fails

必须声明：

```text
The current flattened-source FullEdge architecture is not MLP-equivalent under AdamW; next work should move to patch-local or deeper compositional FullEdge.
```

### Case D：nonlinear edge-correction channel interferes

必须声明：

```text
The nonlinear edge-correction channel is task-interfering under current scale/init; redesign correction gating or correction-channel initialization.
```

### Case E：fixed patch/local source helps

必须声明：

```text
Vision FullEdge should move toward KANConv-like patch-local edge-function primitive rather than flattened scalar source.
```

### Case F：no repair works

必须声明：

```text
S3-B2-P4 active-k shared edge-correction is kernel-native but not trainable enough; retire it as official candidate and keep it only as diagnostic.
```

---

## 11. 最终建议

v9.2.1 的一句话策略是：

$$
\boxed{
\text{不要用 functional update 救一个 AdamW-only 不成立的 FullEdge；先在不离开 PureKAN 的前提下查清并修复 P5 trainability。}
}
$$

当前最重要的不是继续优化 P4，也不是直接打开 P6/P7，而是回答：

```text
1. identity edge-basis channel 是否能接近 MLP？
2. nonlinear edge-correction channel 是容量不足还是干扰 task？
3. 高 train acc / 高 CE loss 是否来自 margin/logit pathology？
4. EdgeAffineNorm 是否压掉 task-useful variation？
5. 能否在 P4 gate 内增加 edge-correction capacity？
```

只有当这些问题得到回答，并且至少一个 candidate 达到 P5 near-pass，v9.2 才应该继续进入 functional update 与 external fair validation。
