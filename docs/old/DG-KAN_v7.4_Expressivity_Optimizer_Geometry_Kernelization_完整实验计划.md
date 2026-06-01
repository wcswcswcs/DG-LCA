# DG-KAN v7.4：表达力来源、优化可达性、几何 Pareto 与 Dense-Preserving Kernelization 完整实验计划

> 本计划基于 v7.0-v7.3 的真实实验链路重新制定。核心修正是：**不再把 classwise-safe 当主目标，也不把 optimizer 小调当主线**。项目真正要回答的是：KAN 的函数空间是否确实强于 MLP；如果 dense KAN 强，为什么 grouped / low-rank / hidden shrink / geometry-constrained / kernel-friendly 版本会掉；这是有效表达力被结构约束压坏，还是 optimizer 没把可表达函数训出来；最后，这个优势能否在 graph-free、strict PureKAN、kernel-native、S2/S1 efficiency 条件下保留。

---

## 0. 当前实验数据的本质判断

### 0.1 已经不能说“KAN 表达力不行”

v7.0 的 dense manual KAN `D3-dense-poly2-gate-d3-warmup-smooth-240` 已经在 3 个 primary datasets 上打开了 basic Beyond-MLP task gate：

```text
D3 val acc mean  = 0.8670
D3 test acc mean = 0.8218
D3 val gap vs MLP = +0.0206
datasets KAN >= MLP = 3/3
```

同时 D3 的校准和 NLL 也优于 MLP：

```text
MLP val ECE = 0.0603
D3  val ECE = 0.0537

MLP val NLL = 0.5516
D3  val NLL = 0.4675

MLP test ECE = 0.0906
D3  test ECE = 0.0481

MLP test NLL = 0.7986
D3  test NLL = 0.5968
```

因此，当前不能把问题归因成“KAN 基函数天然弱于 MLP”。更准确的判断是：

$$
\boxed{
\text{dense KAN / dense poly2\_gate 的函数族已经表现出超过 MLP 的表达力信号。}
}
$$

---

### 0.2 也不能说“只是 optimizer 小参数没调好”

v7.0-v7.2 已经做过多轮 optimizer / recipe / compression probe：

```text
optimizer / lr / warmup / weight decay / label smoothing:
  grouped/T3/G2 路线仍未闭合 task gap。

NoSync:
  真实降低 step，但仍未进 S2。

SGD:
  降低 update ratio 和少量 memory，但 task signal 被破坏。

hidden dim 32 / 48:
  step 有下降，但 task gap 明显变差，memory 仍约 1.29。

low-rank primitive:
  没保住 H3/H4 task signal，memory 也没接近 S2，部分 gradient gate 不完整。
```

所以普通 optimizer 小调不应再作为主线。下一步如果研究 optimizer，必须是为了回答“优化可达性”这个本质问题，而不是继续扫 AdamW 小参数。

---

### 0.3 现在最准确的 blocker

当前 blocker 不是 classwise，不是简单 optimizer，也不是“KAN basis 弱”。真正 blocker 是三件事的耦合：

$$
\boxed{
\text{effective expressivity under constraints}
+
\text{optimization reachability}
+
\text{kernel-native efficiency}.
}
$$

已有数据形成了很清楚的结构：

```text
dense D3:
  task 强，ECE/NLL 强，但不是 strict/kernel-native/efficient。

strict H3:
  证明 strict KAN head 可以保留正向 task signal；
  但 memory / step 严重失败。

cached C3:
  macro gap 可以超过 +2%，统计显著；
  但仍不是 efficient。

G2/T3:
  efficiency 强，但 task gap 大。

grouped / low-rank / hidden shrink:
  memory 或 step 某些方向有改善；
  但有效表达力明显下降。

NoSync / SGD:
  说明 update overhead 是一部分开销；
  但不是根因。
```

因此 v7.4 的目标应该从“修某个数字”转成“判别机制”。

---

## 1. v7.4 总目标

v7.4 的目标是建立一条机制证据链：

$$
\boxed{
\text{表达力来源}
\rightarrow
\text{约束后的有效表达力}
\rightarrow
\text{优化可达性}
\rightarrow
\text{几何 Pareto}
\rightarrow
\text{dense-preserving kernelization}.
}
$$

具体来说，v7.4 要回答五个核心问题：

```text
Q1:
  dense poly2_gate 为什么赢？
  是 gate、poly2、dense cross-channel、strict head、depth、训练 recipe，还是参数量？

Q2:
  grouped / low-rank / hidden shrink 为什么掉？
  是函数类不够，还是 optimizer 训不到，还是参数化病态？

Q3:
  几何性质到底有帮助，还是在伤害表达力？
  它应该是硬约束、弱正则，还是只做 diagnostic？

Q4:
  高效实现应该先压缩再训练，还是先保 dense 表达力再 kernelize？
  目前证据更支持后者，但需要实验确认。

Q5:
  如果 dense-preserving kernelization 失败，应该设计什么新的低 live-set primitive？
```

v7.4 的成功不要求一次完成所有终极目标，但必须在机制上给出明确路线选择。

---

## 2. 成功标准

### 2.1 表达力优势标准

KAN candidate 的 macro task advantage 至少满足：

$$
\Delta Acc_{\text{macro,val}}\geq0.02.
$$

并且统计稳定：

$$
CI_{95\%,macro}^{low}>0.
$$

$$
p_{\text{Holm}}<0.05.
$$

Test consistency：

$$
\Delta Acc_{\text{macro,test}}\geq0.015.
$$

### 2.2 优化可达性标准

对一个 compressed / efficient student，定义 supervised 与 distillation 的差异：

$$
\Delta Acc_{\text{distill}}
=
Acc_{\text{student-distill}}
-
Acc_{\text{student-supervised}}.
$$

若：

$$
\Delta Acc_{\text{distill}}\geq0.02
$$

并且：

$$
Acc_{\text{student-distill}}\geq Acc_{\text{MLP}}-0.01,
$$

说明该结构的表达力可能够，当前主要 blocker 是 optimizer / training objective。

若：

$$
Acc_{\text{student-distill}}\leq Acc_{\text{D3}}-0.03,
$$

且 distillation 提升小于 $0.01$，说明该 compressed student 有效表达力不足。

### 2.3 几何 Pareto 标准

几何不再作为硬目标，而是 Pareto 维度。对：

$$
L = L_{\text{task}}+\lambda L_{\text{geo}},
$$

如果存在 $\lambda^*>0$ 使：

$$
Acc_{\lambda^*}\geq Acc_{\lambda=0}-0.005,
$$

且：

$$
ECE_{\lambda^*}<ECE_{\lambda=0}
$$

或：

$$
NLL_{\lambda^*}<NLL_{\lambda=0},
$$

或：

$$
ValLossAUC_{\lambda^*}<ValLossAUC_{\lambda=0},
$$

则几何作为 weak regularizer 有价值。

如果任意 $\lambda>0$ 都满足：

$$
Acc_{\lambda}<Acc_{\lambda=0}-0.02,
$$

且没有明显 ECE/NLL/AUC 改善，则几何约束伤害表达力，不应作为主线硬约束。

### 2.4 Efficiency 标准

S2 diagnostic gate：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

S1 official gate：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

Strong efficiency：

$$
r_{\text{mem}}\leq0.95,
$$

$$
r_{\text{step}}\leq1.20.
$$

完整成功标准：

$$
\boxed{
\text{StrictPass}
\land
\text{GradPass}
\land
\text{MacroSignificantPass}
\land
\text{OptimizationOrExpressivityExplained}
\land
(\text{S2Pass} \lor \text{S1Pass})
}
$$

---

## 3. 当前离目标还差多远

### 3.1 Task macro 已经接近或局部达到

`H3` 已经有稳定正信号：

```text
H3 mean val gap = +0.0185
CI95 low = +0.0134
Holm p = 3.17e-06
test gap = +0.0192
```

`C3` 更强：

```text
C3 10-seed mean val gap = +0.0214
CI95 low = +0.0158
Holm p = 2.19e-07
test gap = +0.0208
```

所以 task macro 层面并不是空白。现在的问题是这个优势尚未进入 efficient envelope。

---

### 3.2 Efficiency 差距仍然很大

`C3` 10-seed confirmation：

```text
memory ratio = 1.3069
step ratio = 3.3751
```

到 S2：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

以 C3 为例，memory 需要降低：

$$
1-\frac{1.05}{1.3069}\approx19.66\%.
$$

step 需要提速：

$$
1-\frac{1.50}{3.3751}\approx55.56\%.
$$

这不是小修能解决的，需要 materialization-free kernelization 或新 primitive。

---

### 3.3 Optimizer 问题还没被彻底判明

已有证据不支持“普通 optimizer 小调能解决”。但还不能完全排除“优化器/训练目标导致 efficient student 训不到 dense teacher function”。必须通过 distillation / stronger optimizer 做判别。

当前缺口是：

```text
还没有系统回答：
  efficient student 是否能通过 D3/C3 teacher distillation 接近 MLP 或 dense teacher？

如果能：
  optimizer/objective 是主因。

如果不能：
  efficient student 函数类不足。
```

---

### 3.4 Geometry 还没有被科学判定

目前还没有完整的 task-geometry Pareto 曲线。之前如果把几何性质硬保，会误导路线。v7.4 必须直接测：

```text
geometry 约束是否改善 ECE / NLL / AUC？
它是否伤害 accuracy？
它是否只在某个小 λ 范围内有用？
```

---

## 4. v7.4 禁止事项

第一，不允许使用 fake data、proxy rows、固定占位 ratio 或手填结论。未实现项必须写为：

```text
not_implemented
not_run
not_applicable
metric_unavailable
```

第二，不允许把 classwise-safe 当主目标。classwise 只作为局部模式诊断，不作为主 gate。

第三，不允许继续主攻简单 optimizer sweep，例如：

```text
lr high / low
weight decay high / low
label smoothing
class weight
sampler
focal loss
target margin
```

这些已经做过大量测试，不应继续作为主线。

第四，不允许把 dense oracle 当 final success。D3/C3/H3 证明表达力，不等于 kernel-native success。

第五，不允许把 grouped/low-rank 的 task 下降直接解释为 KAN 不行。必须先做 distillation/reachability。

第六，不允许把几何性质作为未验证硬约束。必须做 Pareto。

第七，不允许先 aggressive compression 再尝试补 task。已有结果提示 low-rank、hidden shrink、grouped 会丢有效表达力；下一步应优先保表达力再 kernelize。

第八，不允许在没有 live-set attribution 的情况下盲写 kernel。必须先定位 materialized tensor、kernel count、top memory/time source。

第九，不允许用 test accuracy 调参。test 只用于最终报告。

第十，没有 GradPass 的 candidate 不能进入 route selection。

---

## 5. 核心假设

### H1：dense poly2_gate 的优势来自函数结构，而不是训练 recipe

H1 假设：D3/C3 的优势来自 dense cross-channel poly2_gate 结构，包括 gate、poly2 residual、dense interaction，而不是单纯 warmup、smoothing、depth 或 head。

需要比较：

```text
A0-D3-full
A1-D3-no-gate
A2-D3-no-poly2
A3-D3-no-dense-cross-channel
A4-D3-depth2
A5-D3-no-warmup
A6-D3-no-smoothing
A7-D3-strict-head
A8-MLP-matched-param
```

H1 成立标准：

$$
Acc_{\text{D3-full}}-Acc_{\text{D3-no-gate}}\geq0.01.
$$

或者：

$$
Acc_{\text{D3-full}}-Acc_{\text{D3-no-cross}}\geq0.02.
$$

并且 D3-full 的 ECE/NLL 不劣于主要 ablation。

若 H1 不成立，v7.4 不应继续围绕 poly2_gate kernelization，而要重新定位 gain source。

---

### H2：压缩 / 分组 / 低秩导致 effective expressivity 损失

H2 假设：当前 grouped、low-rank、hidden shrink 等约束把 KAN 的有效函数族压得太小，导致 task signal 丢失。

需要系统比较：

```text
dense
grouped g2/g4/g8/g16
overlap-grouped
low-rank r1/r2/r4/r8/r16
block-sparse
shared-basis
hidden shrink
factorized dense gate
```

H2 成立标准：

在同等 optimizer、同等训练预算下，如果：

$$
Acc_{\text{compressed}}\leq Acc_{\text{dense}}-0.02,
$$

且 distillation 也不能恢复，则说明压缩结构表达力不足。

若 compressed candidate 保持：

$$
Acc_{\text{compressed}}\geq Acc_{\text{MLP}}-0.01
$$

且 efficiency 接近 S2，则该方向可继续。

---

### H3：optimizer 可达性必须通过 teacher distillation / stronger optimizer 判断

H3 假设：如果 efficient student 有足够表达力，D3/C3 teacher 可以通过 distillation 或 stronger optimizer 把它训练到接近 MLP。

Distillation loss：

$$
L =
L_{\text{CE}}
+
\alpha T^2 KL(p_{\text{teacher}}^T||p_{\text{student}}^T)
+
\beta L_{\text{feature}}.
$$

需要测试：

```text
O0-supervised-AdamW
O1-supervised-AdanLite
O2-basis-wise-AdamW
O3-edge-wise-update-normalized-AdamW
O4-diagonal-Fisher-preconditioned
O5-blockwise-Shampoo-lite
O6-LBFGS-small-batch-diagnostic
O7-distill-logit-from-D3
O8-distill-feature-from-D3
O9-distill-logit+feature-from-D3
O10-distill-from-C3
```

H3 optimizer/objective blocker 成立标准：

$$
Acc_{\text{student-distill}}-Acc_{\text{student-supervised}}\geq0.02.
$$

并且：

$$
Acc_{\text{student-distill}}\geq Acc_{\text{MLP}}-0.01.
$$

H3 expressivity blocker 成立标准：

$$
Acc_{\text{student-distill}}\leq Acc_{\text{D3}}-0.03.
$$

且：

$$
Acc_{\text{student-distill}}-Acc_{\text{student-supervised}}<0.01.
$$

---

### H4：几何约束会形成 task-geometry Pareto，而不是天然有利

H4 假设：几何约束可能改善校准/平滑/泛化，但过强会伤害表达力。因此必须 sweep：

$$
L = L_{\text{task}}+\lambda L_{\text{geo}}.
$$

几何项包括：

```text
basis_smoothness
edge_curvature
jacobian_norm
local_lipschitz
coefficient_tv
path_length
```

lambda：

```text
0
1e-5
3e-5
1e-4
3e-4
1e-3
3e-3
1e-2
```

H4 useful geometry 成立标准：

$$
Acc_{\lambda}\geq Acc_{\lambda=0}-0.005.
$$

且：

$$
ECE_{\lambda}<ECE_{\lambda=0}
$$

或：

$$
NLL_{\lambda}<NLL_{\lambda=0}
$$

或：

$$
ValLossAUC_{\lambda}<ValLossAUC_{\lambda=0}.
$$

H4 harmful geometry 成立标准：

$$
Acc_{\lambda}<Acc_{\lambda=0}-0.02
$$

且无明显 ECE/NLL/AUC 收益。

---

### H5：efficiency fail 来自 materialization / kernel fragmentation，而不是 KAN 数学本身

H5 假设：H3/C3 的 memory/step fail 来自 dense basis、gate temp、head temp、grad temp、kernel fragmentation，而不是 poly2_gate 数学不可高效。

H5 成立标准：

live-set attribution 中：

$$
\frac{M_{\text{materialized temps}}}{M_{\text{peak}}}\geq0.35.
$$

或 timing attribution 中：

$$
\frac{T_{\text{materialization/kernel-fragmentation}}}{T_{\text{step}}}\geq0.50.
$$

若成立，下一步应做 materialization-free fused kernel，而不是压缩表达力。

---

### H6：如果 dense-preserving kernelization 仍失败，应设计新 primitive

如果完成：

```text
source-of-gain ablation
distillation / reachability
geometry Pareto
live-set attribution
materialization-free kernelization
factorized / overlap grouped bridge
```

仍无法得到 S2 + task candidate，则当前 dense poly2_gate family 不适合目标效率约束，应设计新 low-live-set function-space primitive。

---

## 6. Candidate 设计

### 6.1 Baselines and oracles

```text
B0-MLP-autograd-reference
B1-G2-O7-efficient-grouped-reference
B2-D3-dense-poly2-gate-oracle
B3-H3-strict-rbf-poly-exp-head
B4-C3-cached-hidden-y-rbf-head
```

### 6.2 Expressivity ablation candidates

```text
A0-D3-full
A1-D3-no-gate
A2-D3-no-poly2
A3-D3-no-dense-cross-channel
A4-D3-depth2
A5-D3-no-warmup
A6-D3-no-smoothing
A7-D3-strict-head
A8-MLP-matched-param
```

### 6.3 Compression / constraint candidates

```text
C0-dense
C1-grouped-g2
C2-grouped-g4
C3-grouped-g8
C4-grouped-g16
C5-overlap-grouped-g8-overlap2
C6-overlap-grouped-g16-overlap2
C7-lowrank-r1
C8-lowrank-r2
C9-lowrank-r4
C10-lowrank-r8
C11-lowrank-r16
C12-blocksparse-25
C13-blocksparse-50
C14-shared-basis
C15-factorized-dense-gate-r4
C16-factorized-dense-gate-r8
C17-hidden32
C18-hidden48
C19-hidden64
```

### 6.4 Optimization reachability candidates

```text
O0-supervised-AdamW
O1-supervised-AdanLite
O2-basis-wise-AdamW
O3-edge-wise-update-normalized-AdamW
O4-diagonal-Fisher-preconditioned
O5-blockwise-Shampoo-lite
O6-LBFGS-small-batch-diagnostic
O7-logit-distill-from-D3
O8-feature-distill-from-D3
O9-logit+feature-distill-from-D3
O10-logit-distill-from-C3
O11-distill-then-supervised-finetune
```

### 6.5 Geometry Pareto candidates

```text
GEO0-lambda0
GEO1-lambda1e-5
GEO2-lambda3e-5
GEO3-lambda1e-4
GEO4-lambda3e-4
GEO5-lambda1e-3
GEO6-lambda3e-3
GEO7-lambda1e-2
```

### 6.6 Kernelization candidates

```text
K0-C3-current
K1-C3-no-full-basis-materialization
K2-C3-gate-temp-streaming
K3-C3-grad-gate-streaming
K4-C3-grad-poly-streaming
K5-C3-fused-transform-mix-backward
K6-C3-fused-head-backward
K7-C3-tile-stream-forward-backward
K8-C3-materialization-free-combo
K9-C3-factorized-gate-r4
K10-C3-factorized-gate-r8
K11-C3-overlap-grouped-dense-bridge
```

### 6.7 New primitive candidates

```text
NP0-TileStreamDensePoly2Gate
NP1-OverlapGroupedPoly2Gate
NP2-BlockSparseDenseGate
NP3-DenseLocalHybridGate
NP4-SharedBasisDenseGate
NP5-EdgeOwnedMixtureOfBasis
NP6-MaterializationFreePrototypeKAN
```

---

## 7. 实验阶段总览

v7.4 分为十二个阶段：

```text
P0: v7.2/C3 reproduction and provenance lock
P1: dense poly2_gate source-of-gain attribution
P2: compression / grouping effective-expressivity curve
P3: optimizer reachability and teacher distillation
P4: geometry constraint Pareto
P5: live-set and kernelization attribution
P6: materialization-free microkernel benchmark
P7: dense-preserving kernelized package
P8: new primitive diagnostic
P9: 10-seed task + AUC verification
P10: full-grid efficiency profiler
P11: candidate co-selection and route decision
P12: artifact and failure audit
```

---

## 8. P0：v7.2/C3 reproduction and provenance lock

### 8.1 目的

确认 v7.4 与 v7.2/C3 可比，锁定 no-fake/no-proxy、strict/manual、gradient correctness、MLP denominator。

### 8.2 必跑对象

```text
B0-MLP-autograd-reference
B1-G2-O7-efficient-grouped-reference
B2-D3-dense-poly2-gate-oracle
B3-H3-strict-rbf-poly-exp-head
B4-C3-cached-hidden-y-rbf-head
```

### 8.3 必须记录字段

```text
run_id
candidate
family
status
fake_data_used
proxy_row_used
uses_loss_backward
uses_torch_autograd_graph
manual_forward_available
manual_backward_available
manual_update_available
nonKAN_param_count
head_type
head_is_kan
grad_relerr_max
grad_cos_min
val_acc_mean
test_acc_mean
val_gap_vs_MLP
test_gap_vs_MLP
ci95_low
holm_p
memory_ratio_mean
step_ratio_mean
forward_ratio_mean
backward_ratio_mean
reproduction_delta_val_gap
reproduction_delta_efficiency
```

### 8.4 判断标准

P0 pass：

```text
fake_data_used = 0
proxy_row_used = 0
nonKAN_param_count = 0 for strict candidates
manual_backward_available = 1
uses_loss_backward = 0
grad_relerr_max <= 1e-4
grad_cos_min >= 0.999
```

Reproduction pass：

$$
|\Delta Acc_{\text{C3,v74}}-\Delta Acc_{\text{C3,v72}}|\leq0.005.
$$

### 8.5 可视化

```text
p0_reproduction_task_bar.svg
p0_reproduction_efficiency_bar.svg
p0_contract_heatmap.svg
p0_candidate_lineage_table.md
```

---

## 9. P1：dense poly2_gate source-of-gain attribution

### 9.1 目的

确认 dense KAN 的 task advantage 来自哪里。P1 是表达力归因实验，不做 efficiency 优化。

### 9.2 实验对象

```text
D3-full
D3-no-gate
D3-no-poly2
D3-no-dense-cross-channel
D3-depth2
D3-no-warmup
D3-no-smoothing
D3-strict-head
MLP-matched-param
```

### 9.3 设置

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2,3,4

steps:
  240

train/val/test:
  1536/512/512

logging:
  every 20 steps
```

### 9.4 必须记录字段

```text
candidate
dataset
seed
train_acc
val_acc
test_acc
train_loss
val_loss
test_loss
ECE
NLL
val_gap_vs_MLP
test_gap_vs_MLP
feature_effective_rank
margin_mean
margin_p10
logit_norm_mean
confidence_mean
wrong_confidence_mean
gate_activation_mean
gate_activation_entropy
poly2_contribution_norm
cross_channel_contribution_norm
ablation_delta_acc
ablation_delta_loss
```

### 9.5 判断标准

Gate contribution：

$$
Acc_{\text{D3-full}}-Acc_{\text{D3-no-gate}}\geq0.01.
$$

Cross-channel contribution：

$$
Acc_{\text{D3-full}}-Acc_{\text{D3-no-cross}}\geq0.02.
$$

Poly2 contribution：

$$
Acc_{\text{D3-full}}-Acc_{\text{D3-no-poly2}}\geq0.01.
$$

If all ablations produce small delta：

$$
|\Delta Acc|<0.005,
$$

then D3 advantage may be training recipe / depth / head rather than poly2_gate.

### 9.6 可视化

```text
p1_ablation_val_gap_bar.svg
p1_ablation_loss_curve.svg
p1_poly2_gate_contribution_bar.svg
p1_feature_rank_by_ablation.svg
p1_margin_distribution_by_ablation.svg
p1_source_of_gain_dashboard.svg
```

---

## 10. P2：compression / grouping effective-expressivity curve

### 10.1 目的

判断压缩、分组、低秩到底损失多少有效表达力。P2 不以 classwise 为目标，而是画出表达力-效率曲线。

### 10.2 实验对象

```text
Dense-poly2-gate
Grouped-g2
Grouped-g4
Grouped-g8
Grouped-g16
OverlapGrouped-g8-overlap2
OverlapGrouped-g16-overlap2
LowRank-r1
LowRank-r2
LowRank-r4
LowRank-r8
LowRank-r16
BlockSparse-25%
BlockSparse-50%
SharedBasis
Hidden32
Hidden48
Hidden64
FactorizedDenseGate-r4
FactorizedDenseGate-r8
```

### 10.3 必须记录字段

```text
candidate
constraint_type
constraint_strength
rank
group_size
overlap_size
sparsity
hidden_dim
val_acc
test_acc
val_gap_vs_dense
val_gap_vs_MLP
ECE
NLL
feature_rank
margin_p10
memory_ratio_mean
step_ratio_mean
param_count
kernel_count
materialized_tensor_count
```

### 10.4 判断标准

Compression preserves expression if：

$$
Acc_{\text{compressed}}\geq Acc_{\text{dense}}-0.01.
$$

Compression hurts expression if：

$$
Acc_{\text{compressed}}\leq Acc_{\text{dense}}-0.02.
$$

Strong compression useful if：

$$
Acc_{\text{compressed}}\geq Acc_{\text{MLP}}-0.01
$$

and:

$$
r_{\text{mem}}\leq1.05,\quad r_{\text{step}}\leq1.50.
$$

### 10.5 可视化

```text
p2_accuracy_vs_constraint_strength.svg
p2_efficiency_vs_constraint_strength.svg
p2_expression_efficiency_pareto.svg
p2_rank_group_hidden_heatmap.svg
p2_feature_rank_vs_accuracy.svg
```

---

## 11. P3：optimizer reachability and teacher distillation

### 11.1 目的

区分“表达力不够”与“optimizer 训不出来”。这是 v7.4 最关键的实验之一。

### 11.2 Teacher

```text
teacher:
  D3 dense oracle
  C3 cached strict candidate
```

### 11.3 Students

```text
G2/T3 efficient grouped
Grouped-g2/g4/g8/g16
OverlapGrouped
LowRank-r2/r4/r8/r16
BlockSparse
Best compressed candidate from P2
```

### 11.4 Training recipes

```text
supervised CE baseline
AdamW
AdanLite
basis-wise AdamW
edge-wise update-normalized AdamW
diagonal Fisher preconditioner
blockwise Shampoo-lite diagnostic
LBFGS small-batch diagnostic
logit distillation
feature distillation
logit + feature distillation
distill then supervised finetune
```

### 11.5 必须记录字段

```text
student
teacher
recipe
optimizer
distill_temperature
alpha_logit
beta_feature
val_acc
test_acc
val_gap_vs_MLP
val_gap_vs_teacher
distill_improvement_vs_supervised
ECE
NLL
student_teacher_logit_KL
student_teacher_logit_cos
feature_CKA
feature_rank
margin_p10
grad_norm_mean
update_norm_mean
cos_update_grad
bad_step_rate
memory_ratio_mean
step_ratio_mean
```

### 11.6 判断标准

Optimizer / reachability issue if：

$$
Acc_{\text{student-distill}}\geq Acc_{\text{MLP}}-0.01.
$$

or:

$$
Acc_{\text{student-distill}}-Acc_{\text{student-supervised}}\geq0.02.
$$

Expressivity / constraint issue if：

$$
Acc_{\text{student-distill}}\leq Acc_{\text{D3}}-0.03.
$$

and distillation improves little：

$$
Acc_{\text{student-distill}}-Acc_{\text{student-supervised}}<0.01.
$$

### 11.7 可视化

```text
p3_distill_improvement_bar.svg
p3_student_teacher_gap.svg
p3_feature_cka_heatmap.svg
p3_optimizer_reachability_dashboard.svg
p3_accuracy_vs_efficiency_by_recipe.svg
```

---

## 12. P4：geometry constraint Pareto

### 12.1 目的

验证几何约束是否帮助泛化，还是伤害表达力。几何不再是硬目标，而是 Pareto 维度。

### 12.2 实验对象

```text
D3 / H3 / C3
Best compressed candidate
Best distillable student
```

### 12.3 Geometry losses

```text
basis_smoothness
edge_curvature
jacobian_norm
local_lipschitz
coefficient_tv
path_length
```

### 12.4 Lambda sweep

```text
lambda_geo:
  0
  1e-5
  3e-5
  1e-4
  3e-4
  1e-3
  3e-3
  1e-2
```

### 12.5 必须记录字段

```text
candidate
lambda_geo
geometry_loss_type
train_acc
val_acc
test_acc
ECE
NLL
ValLossAUC_step
ValLossAUC_time
geometry_metric_value
basis_smoothness
edge_curvature
jacobian_norm
local_lipschitz
feature_rank
margin_p10
memory_ratio_mean
step_ratio_mean
```

### 12.6 判断标准

Useful geometry regularization if：

$$
Acc_{\lambda}\geq Acc_{\lambda=0}-0.005.
$$

and:

$$
ECE_{\lambda}<ECE_{\lambda=0}
$$

or:

$$
NLL_{\lambda}<NLL_{\lambda=0}.
$$

Harmful geometry constraint if：

$$
Acc_{\lambda}<Acc_{\lambda=0}-0.02.
$$

and no strong calibration/AUC improvement appears.

### 12.7 可视化

```text
p4_accuracy_vs_geometry_lambda.svg
p4_ece_vs_geometry_lambda.svg
p4_nll_vs_geometry_lambda.svg
p4_geometry_metric_vs_accuracy.svg
p4_geometry_pareto_frontier.svg
```

---

## 13. P5：live-set and kernelization attribution

### 13.1 目的

解释 C3/H3/D3 为什么效率失败。P5 必须定位 materialization / kernel fragmentation。

### 13.2 必跑对象

```text
MLP
D3
H3
C3
Best compressed candidate
Best distillable student
```

### 13.3 Shape grid

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

batch sizes:
  128,256,512

depths:
  native depth

warmup / measure:
  50 / 200
```

### 13.4 必须记录字段

```text
candidate
dataset
batch_size
depth
peak_allocated_MB
peak_reserved_MB
memory_ratio
step_ratio
forward_ratio
backward_ratio
update_ratio
basis_temp_MB
gate_temp_MB
hidden_y_cache_MB
head_temp_MB
grad_gate_MB
grad_poly_MB
dense_output_temp_MB
manual_cache_MB
optimizer_state_MB
largest_live_tensor_MB
materialized_tensor_count
torch_op_count
triton_kernel_count
kernel_launch_proxy_count
python_loop_count
top1_memory_source
top2_memory_source
top3_memory_source
unknown_memory_fraction
```

### 13.5 判断标准

Attribution pass：

```text
top3 memory sources identified
unknown_memory_fraction <= 0.10
component timing explains >= 90% step time
```

Materialization blocker：

$$
\frac{M_{\text{materialized temps}}}{M_{\text{peak}}}\geq0.35.
$$

Kernel fragmentation blocker：

$$
\frac{T_{\text{fragmented ops}}}{T_{\text{step}}}\geq0.50.
$$

### 13.6 可视化

```text
p5_live_set_waterfall.svg
p5_step_time_waterfall.svg
p5_materialized_tensor_count.svg
p5_kernel_count_by_phase.svg
p5_batch_scaling_memory_step.svg
```

---

## 14. P6：materialization-free microkernel benchmark

### 14.1 目的

在完整 package 之前，先验证局部 kernel 是否能真实降低 memory/time，并保持梯度正确。

### 14.2 Microkernels

```text
MK0-current-poly2-gate-forward
MK1-fused-poly2-gate-forward
MK2-current-poly2-gate-backward
MK3-fused-poly2-gate-backward
MK4-streaming-grad-gate
MK5-streaming-grad-poly
MK6-fused-transform-mix-backward
MK7-tile-stream-dense-gate
MK8-fused-head-backward
MK9-factorized-dense-gate-r4/r8
```

### 14.3 必须记录字段

```text
microkernel
input_shape
output_shape
implementation_status
time_ms_current
time_ms_candidate
time_ratio_vs_current
memory_MB_current
memory_MB_candidate
memory_ratio_vs_current
forward_relerr
grad_relerr
grad_cos
materialized_tensor_count
temp_MB
kernel_count
allocation_count
bandwidth_estimate_GBps
```

### 14.4 判断标准

Microkernel useful if：

$$
\frac{T_{\text{candidate}}}{T_{\text{current}}}\leq0.80
$$

or:

$$
\frac{M_{\text{candidate}}}{M_{\text{current}}}\leq0.80.
$$

Microkernel enters full package if：

```text
grad_relerr < 1e-4
grad_cos > 0.999
time_ratio_vs_current <= 0.90
memory_ratio_vs_current <= 0.95
```

### 14.5 可视化

```text
p6_microkernel_memory_time_pareto.svg
p6_grad_correctness_lollipop.svg
p6_materialization_reduction_bar.svg
p6_kernel_count_reduction_bar.svg
```

---

## 15. P7：dense-preserving kernelized package

### 15.1 目的

把 P6 有效 microkernel 集成到 C3/D3/H3 的 full package。原则是：先保表达力，再压效率；不要先用低秩/分组把表达力砍掉。

### 15.2 Candidate packages

```text
PK0-C3-current
PK1-C3-no-full-basis-materialization
PK2-C3-gate-temp-streaming
PK3-C3-grad-gate-streaming
PK4-C3-grad-poly-streaming
PK5-C3-fused-transform-mix-backward
PK6-C3-fused-head-backward
PK7-C3-tile-stream-forward-backward
PK8-C3-materialization-free-combo
PK9-C3-factorized-gate-r4
PK10-C3-factorized-gate-r8
PK11-C3-overlap-grouped-dense-bridge
```

### 15.3 必须记录字段

```text
package
parent
implementation_status
uses_fused_poly2
uses_streaming_grad_gate
uses_streaming_grad_poly
uses_fused_head
uses_tile_stream
factor_rank
overlap_group
memory_ratio_mean
step_ratio_mean
forward_ratio_mean
backward_ratio_mean
memory_improvement_vs_parent
step_improvement_vs_parent
val_acc_mean
test_acc_mean
val_gap_vs_MLP
val_gap_vs_parent
ECE
NLL
grad_relerr_max
grad_cos_min
materialized_tensor_count
kernel_count_total
```

### 15.4 判断标准

Expression preservation：

$$
Acc_{\text{package}}\geq Acc_{\text{parent}}-0.005.
$$

or at least：

$$
Acc_{\text{package}}\geq Acc_{\text{MLP}}+0.01.
$$

Efficiency useful：

$$
\frac{M_{\text{package}}}{M_{\text{parent}}}\leq0.85
$$

or:

$$
\frac{T_{\text{package}}}{T_{\text{parent}}}\leq0.70.
$$

S2：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

### 15.5 可视化

```text
p7_kernelized_package_pareto.svg
p7_parent_vs_package_memory_step.svg
p7_expression_preservation_bar.svg
p7_s2_s1_boundary.svg
```

---

## 16. P8：new primitive diagnostic

### 16.1 目的

如果 C3/D3 family 仍难以 kernelize，设计新的低 live-set function-space primitive。目标不是 classwise，而是保留 KAN 表达力并降低 live set。

### 16.2 Candidates

```text
NP0-TileStreamDensePoly2Gate
NP1-OverlapGroupedPoly2Gate
NP2-BlockSparseDenseGate
NP3-DenseLocalHybridGate
NP4-SharedBasisDenseGate
NP5-EdgeOwnedMixtureOfBasis
NP6-MaterializationFreePrototypeKAN
```

### 16.3 必须记录字段

```text
candidate
primitive_type
live_set_design
expression_design
strict_pass
grad_pass
val_acc_mean
test_acc_mean
val_gap_vs_MLP
ECE
NLL
feature_rank
margin_p10
memory_ratio_mean
step_ratio_mean
basis_temp_MB
gate_temp_MB
materialized_tensor_count
kernel_count_total
```

### 16.4 判断标准

New primitive useful：

$$
\Delta Acc_{\text{macro,val}}\geq0.02.
$$

S2-near：

$$
r_{\text{mem}}\leq1.10,
$$

$$
r_{\text{step}}\leq1.70.
$$

S2：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

### 16.5 可视化

```text
p8_new_primitive_pareto.svg
p8_expression_vs_live_set.svg
p8_feature_rank_vs_efficiency.svg
```

---

## 17. P9：10-seed task + AUC verification

### 17.1 目的

对通过 P3/P7/P8 的候选做 10-seed task 和 AUC 验证。

### 17.2 必跑对象

```text
MLP
D3
C3
best-distillable-student
best-kernelized-package
best-new-primitive
```

### 17.3 设置

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0..9

steps:
  240

logging:
  every 20 steps
```

### 17.4 必须记录字段

```text
candidate
dataset
seed
train_acc
val_acc
test_acc
val_loss
test_loss
ECE
NLL
val_gap_vs_MLP
test_gap_vs_MLP
ValLossAUC_step
ValLossAUC_time
ValAccAUC_step
ValAccAUC_time
time_to_target_acc
time_to_target_loss
feature_rank
margin_p10
local_failure_audit
```

### 17.5 判断标准

MacroSignificantPass：

$$
\Delta Acc_{\text{macro,val}}\geq0.02.
$$

$$
CI_{95\%,macro}^{low}>0.
$$

$$
p_{\text{Holm}}<0.05.
$$

Test consistency：

$$
\Delta Acc_{\text{macro,test}}\geq0.015.
$$

Time AUC：

$$
ValLossAUC_{\text{time,KAN}}\leq ValLossAUC_{\text{time,MLP}}.
$$

### 17.6 可视化

```text
p9_seedwise_val_gap_boxplot.svg
p9_bootstrap_ci.svg
p9_val_loss_vs_time.svg
p9_val_loss_auc_time_bar.svg
p9_task_efficiency_pareto.svg
```

---

## 18. P10：full-grid efficiency profiler

### 18.1 目的

对 P9 selected candidates 做 full-grid efficiency profiler。

### 18.2 Shape grid

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

batch sizes:
  128,256,512

depths:
  native,2,3,4

warmup / measure:
  50 / 200
```

### 18.3 必须记录字段

```text
candidate
dataset
batch_size
depth
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
forward_ratio_mean
backward_ratio_mean
update_ratio_mean
peak_allocated_MB
peak_reserved_MB
largest_live_tensor_MB
top1_memory_source
top2_memory_source
top3_memory_source
kernel_count_total
torch_op_count
triton_kernel_count
materialized_tensor_count
```

### 18.4 判断标准

S2：

$$
r_{\text{mem}}\leq1.05.
$$

$$
r_{\text{step}}\leq1.50.
$$

S1：

$$
r_{\text{mem}}<1.00.
$$

$$
r_{\text{step}}\leq1.35.
$$

### 18.5 可视化

```text
p10_efficiency_pareto.svg
p10_batch_depth_heatmap_memory.svg
p10_batch_depth_heatmap_step.svg
p10_memory_source_waterfall.svg
```

---

## 19. P11：candidate co-selection and route decision

### 19.1 Survivor 类型

```text
S0:
  StrictPass + GradPass + S1Pass + MacroSignificantPass + TimeAUCPass

S1:
  StrictPass + GradPass + S2Pass + MacroSignificantPass

S2:
  MacroSignificantPass but efficiency fail

S3:
  S2/S1 efficiency pass but task advantage not significant

S4:
  distillation shows optimizer reachability but raw training fails

S5:
  distillation also fails, indicating effective expressivity loss

S6:
  geometry improves calibration without task damage

S7:
  geometry hurts expression

S8:
  gradient fail

S9:
  no improvement
```

### 19.2 必须记录字段

```text
candidate
survivor_type
strict_pass
grad_pass
s2_pass
s1_pass
macro_significant_pass
time_auc_pass
distillation_reachable
geometry_pareto_status
memory_ratio_mean
step_ratio_mean
val_gap_vs_MLP
test_gap_vs_MLP
ci95_low
holm_p
ECE_delta
NLL_delta
ValLossAUC_time_delta
route_recommendation
primary_blocker
next_required_implementation
```

### 19.3 可视化

```text
p11_expression_optimization_efficiency_pareto.svg
p11_survivor_type_dashboard.svg
p11_route_candidate_scorecard.svg
p11_geometry_pareto_summary.svg
```

---

## 20. P12：artifact and failure audit

### 20.1 Artifacts

```text
run_manifest.json
provenance_audit.csv
candidate_registry.csv
gate_config.json
p0_reproduction.csv
p1_source_of_gain_attribution.csv
p2_expression_constraint_curve.csv
p3_optimizer_reachability_distillation.csv
p4_geometry_pareto.csv
p5_live_set_kernel_attribution.csv
p6_microkernel_benchmark.csv
p7_kernelized_package.csv
p8_new_primitive.csv
p9_10seed_task_auc.csv
p10_fullgrid_efficiency.csv
p11_candidate_selection.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

### 20.2 Failure taxonomy

```text
F1_macro_task_fail
F2_optimizer_reachability_fail
F3_effective_expressivity_loss
F4_geometry_hurts_expression
F5_geometry_no_benefit
F6_memory_fail
F7_step_time_fail
F8_kernelization_fail
F9_gradient_correctness_fail
F10_distillation_only
F11_fake_or_proxy_violation
F12_nonKAN_violation
F13_artifact_missing
```

### 20.3 总图

```text
figures/p11_expression_optimization_efficiency_pareto.svg
figures/p11_geometry_pareto_summary.svg
figures/p11_survivor_type_dashboard.svg
figures/route_decision_dashboard.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 21. 第一轮推荐执行顺序

v7.4 不应一开始全量铺开，应该先做最能区分机制的实验。

### Step 1：P1 source-of-gain attribution

先回答 dense D3/C3/H3 到底为什么赢。尤其要确认：

```text
gate 是否关键？
poly2 是否关键？
dense cross-channel 是否关键？
strict head 是否关键？
```

如果 H1 不成立，不要继续围绕 poly2_gate kernelization。

### Step 2：P3 distillation reachability

这是区分“表达力不够”还是“optimizer 不够”的关键。

如果 grouped / low-rank student 经 D3/C3 distillation 后能接近 MLP，则主线转为 KAN-specific optimizer / distillation / training objective。  
如果 distillation 也救不回来，则说明结构约束有效表达力不足。

### Step 3：P4 geometry Pareto

验证几何约束是否伤表达力。不要把几何当硬目标；先画 Pareto 曲线。

### Step 4：P5 attribution + P6 microkernel

只有确认 dense signal 来源后，才做 materialization-free kernelization。优先目标：

```text
fused-transform-mix-backward
streaming grad_gate
streaming grad_poly
tile-stream dense gate
```

### Step 5：P7/P8 package 和新 primitive

如果 C3 kernelization 保表达力且效率下降明显，继续 package。  
如果 C3 kernelization 始终太重，直接进入 new low-live-set primitive。

---

## 22. 停止条件

### 22.1 成功停止

出现以下任一情况，立即写复盘：

```text
S2 + MacroSignificantPass
S1 + MacroSignificantPass
S1 + MacroSignificantPass + TimeAUCPass
```

### 22.2 失败停止

出现以下任一情况，停止对应路线：

```text
1. dense ablation 显示 poly2_gate 不是 gain source；
2. distillation 无法让 compressed student 提升 >=1%；
3. geometry lambda 一加就导致 task loss >2%，且 calibration/AUC 无明显收益；
4. materialization-free kernel 后 step 仍 >2.0 或 memory >1.20；
5. new primitive macro gap < +0.01；
6. route candidate grad correctness fail 且无法修复；
7. no-fake/no-proxy audit fail。
```

---

## 23. 成功与失败解释规则

### Case A：distillation 能救 grouped / low-rank

如果：

$$
Acc_{\text{student-distill}}\geq Acc_{\text{MLP}}-0.01,
$$

则说明 student 的表达力可能够，optimizer / objective 是主问题。下一轮应该研究 KAN-specific optimizer，而不是换 primitive。

### Case B：distillation 也救不了

如果：

$$
Acc_{\text{student-distill}}\leq Acc_{\text{D3}}-0.03,
$$

说明 efficient student 有效表达力不足。下一轮必须改结构。

### Case C：geometry 伤 task

如果：

$$
Acc_{\lambda}<Acc_{\lambda=0}-0.02,
$$

且 ECE/NLL/AUC 没补回来，则几何约束不能作为硬目标。

### Case D：geometry 有 Pareto 收益

如果：

$$
Acc_{\lambda}\geq Acc_{\lambda=0}-0.005
$$

且 ECE/NLL/AUC 改善，则几何可以作为 weak regularizer。

### Case E：kernelized dense preserves expression and reaches S2

如果：

$$
Acc_{\text{kernelized}}\geq Acc_{\text{dense}}-0.005,
$$

且：

$$
r_{\text{mem}}\leq1.05,\quad r_{\text{step}}\leq1.50,
$$

则 v7.4 达成真正关键突破。

### Case F：kernelization improves efficiency but hurts expression

说明 materialization-free implementation 改变了有效函数或压缩太强。需要回到 dense-preserving design。

### Case G：no improvement

如果表达力归因、distillation、geometry、kernelization 都没有带来可用 candidate，则当前 poly2_gate family 应暂停，设计新 function-space primitive。

---

## 24. 最终建议

v7.4 的一句话策略是：

$$
\boxed{
\text{主线应是表达力归因、优化可达性、几何 Pareto 与 dense-preserving kernelization，而不是 classwise-safe 或 optimizer 小调。}
}
$$

当前最准确的判断是：

```text
1. KAN basis / dense poly2_gate 不是弱于 MLP；
2. dense KAN 已经有超过 MLP 的表达力信号；
3. grouped / low-rank / hidden shrink 等高效约束会显著损伤有效表达力；
4. 普通 optimizer 小调不是主因；
5. KAN-specific optimizer 仍值得研究，但必须通过 distillation/reachability 实验证明；
6. 几何性质不能硬保，必须作为 Pareto 维度；
7. efficiency 需要 materialization-free kernelization，而不是继续 Python/Torch 小修。
```

如果 v7.4 能证明：

$$
\text{dense advantage source明确}
+
\text{distillation/reachability判断清楚}
+
\text{geometry Pareto明确}
+
\text{kernelized candidate 保表达力并进 S2}
$$

那么项目就会从“有表达力信号但系统未闭合”推进到“真正理解 KAN 为什么以及何时强于 MLP”。
