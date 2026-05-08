# DG-KAN v7.2：显著 Beyond-MLP 优势、Kernel-Native Efficiency 与 Strict PureKAN-NG 闭环实验计划

> 本计划基于 v7.1 final run `results/real_rerun_20260505/v71_poly2gate_kernel_final_20260505T221338Z` 制定。v7.1 的核心进展是：**strict KAN head 已经保住了 Beyond-MLP task signal**，最佳 strict candidate `H3-D3-dense-poly2-gate-d3-KAN-head-rbf-poly-exp` 的 mean validation gap 为 `+0.0189`，3/3 datasets `KAN >= MLP`。  
> 但这还不能称为“显著超越 MLP”。当前优势幅度仍偏小，缺少显著性检验、置信区间、ValLossAUC/time AUC、完整 full-gradient gate，而且 H3 的 memory ratio `1.3031`、step ratio `3.6324` 远未进入 S2。  
> 因此 v7.2 的目标从 “basic Beyond-MLP pass” 升级为：**在 strict PureKAN / graph-free / manual backward 条件下，证明对 MLP 有统计显著、实践显著、效率成立的优势。**

---

## 0. 当前结论：现在是否对 MLP 有显著优势？

答案是：

$$
\boxed{
\text{目前还不能说有显著优势，只能说 task mean accuracy 出现了稳定正信号。}
}
$$

v7.1 的 task 侧确实有突破。最佳 strict candidate `H3` 的结果是：

```text
B0 MLP:
  val acc = 0.8440
  test acc = 0.7951

H3 strict rbf-poly-exp head:
  val acc = 0.8628
  test acc = 0.8181
  val gap vs MLP = +0.0189
```

H3 分 dataset：

```text
MNIST:
  MLP val = 0.8919
  H3 val  = 0.9121
  gap = +0.0202

Fashion-MNIST:
  MLP val = 0.8066
  H3 val  = 0.8275
  gap = +0.0208

KMNIST:
  MLP val = 0.8333
  H3 val  = 0.8490
  gap = +0.0156
```

这说明 H3 是一个真实的 strict task improvement，不是一个只在单数据集上碰巧超过 MLP 的点。但是，如果我们把“显著优势”定义为 “不是稍微超过，而是在统计和实践上都能站住”，那么当前仍不够，原因有四个。

第一，mean validation gap 是：

$$
0.8628-0.8440=0.0188.
$$

这只有约 $1.9$ 个百分点，接近但没有明显超过 $2\%$ 的实践显著阈值。MNIST 和 Fashion-MNIST 超过 $2\%$，但 KMNIST 是 $1.56\%$，还不够强。

第二，v7.1 没有报告 bootstrap confidence interval、paired seed test、per-dataset corrected $p$ value 或 effect size。因此不能判断这个 $+1.89\%$ 是统计显著优势，还是 3-seed / 512-val split 下的中等波动。

第三，test mean 虽然高于 MLP，但还需要 per-dataset test gap、seedwise variance 和 classwise collapse audit。尤其 KMNIST 的 H3 test 为 `0.7201`，必须确认它相对 MLP 的 test gap是否同向，不能只看 macro mean。

第四，H3 不是 efficient candidate：

$$
r_{mem}=1.3031,
$$

$$
r_{step}=3.6324.
$$

S2 要求：

$$
r_{mem}\leq1.05,\quad r_{step}\leq1.50.
$$

所以当前最多能说：

$$
\boxed{
\text{H3 在 strict task accuracy 上有正向信号，但还没有证明显著、稳健、efficient 的 Beyond-MLP 优势。}
}
$$

---

## 1. v7.2 实验整体目标

v7.2 的整体目标是把 v7.1 的 “basic task pass” 升级为真正的 “significant Beyond-MLP candidate”。

v7.2 不再满足于：

$$
Acc_{KAN}\geq Acc_{MLP}-0.01.
$$

v7.2 要验证：

$$
\boxed{
\text{KAN 是否在统计显著、实践显著、效率成立、校准不差的条件下优于 MLP。}
}
$$

v7.2 的最低成功标准是：

$$
\boxed{
\text{StrictPass} \land \text{GradPass} \land \text{S2Pass} \land \text{SignificantTaskPass}.
}
$$

v7.2 的正式成功标准是：

$$
\boxed{
\text{StrictPass} \land \text{GradPass} \land \text{S1Pass} \land \text{SignificantTaskPass}.
}
$$

v7.2 的强成功标准是：

$$
\boxed{
\text{StrictPass} \land \text{GradPass} \land \text{S1Pass} \land \text{StrongSignificantBeyondPass}.
}
$$

其中：

```text
StrictPass:
  zero non-KAN trainable params
  no loss.backward graph
  manual forward / backward / update

GradPass:
  full gradient relerr/cos gate pass
  one-step loss descent pass
  rollback exactness pass

S2Pass:
  memory_ratio <= 1.05
  step_ratio <= 1.50

S1Pass:
  memory_ratio < 1.00
  step_ratio <= 1.35

SignificantTaskPass:
  not merely slightly above MLP;
  requires practical gap, confidence intervals, seed stability, and no classwise collapse.
```

---

## 2. 新的“显著优势”判定标准

### 2.1 Practical significance gate

v7.2 定义实践显著优势为：

$$
\Delta Acc_{macro,val}\geq0.02.
$$

也就是说，KAN 的 macro validation accuracy 至少比 MLP 高 $2$ 个百分点。

同时要求至少两个 primary datasets 满足：

$$
\Delta Acc_{val,d}\geq0.02.
$$

并且任何 dataset 不允许明显退化：

$$
\Delta Acc_{val,d}\geq-0.005.
$$

这里：

$$
\Delta Acc_{val,d}
=
Acc_{KAN,val,d}-Acc_{MLP,val,d}.
$$

### 2.2 Statistical significance gate

每个 candidate 必须使用 seed-level 和 sample-level 两类统计检验。

Seed-level paired test：

```text
same dataset
same seed
same train/val/test split
compare KAN vs MLP validation accuracy
```

必须记录：

```text
paired_mean_gap
paired_std_gap
paired_t_stat
paired_p_value
wilcoxon_p_value
```

Sample-level bootstrap：

```text
bootstrap over validation examples
bootstrap over seeds
10000 bootstrap samples recommended
```

必须记录：

```text
ci95_low
ci95_high
bootstrap_p_value
```

Statistical pass 要求：

$$
CI_{95\%,macro}^{low}>0.
$$

并且至少两个 datasets 满足：

$$
CI_{95\%,d}^{low}>0.
$$

多数据集 / 多候选比较需要 Holm correction：

$$
p_{corrected}<0.05.
$$

### 2.3 Effect size gate

Accuracy gap 容易受样本数影响，因此 v7.2 还记录 Cohen's $h$：

$$
h = 2\arcsin(\sqrt{p_{KAN}}) - 2\arcsin(\sqrt{p_{MLP}}).
$$

Effect size 不作为单独硬 gate，但必须报告：

```text
cohen_h_macro
cohen_h_by_dataset
```

强优势建议目标：

$$
|h| \geq 0.05
$$

on at least two datasets.

### 2.4 Test consistency gate

Validation win 不能只停在 validation。v7.2 要求 test 同向：

$$
\Delta Acc_{macro,test}\geq0.015.
$$

并且：

$$
\min_d \Delta Acc_{test,d}\geq-0.005.
$$

如果 val win 但 test collapse，candidate 不能称为显著优势，只能称为 validation diagnostic。

### 2.5 Calibration / NLL gate

显著优势不能以 calibration 变差为代价。必须满足：

$$
ECE_{KAN}\leq ECE_{MLP}+0.005.
$$

$$
NLL_{KAN}\leq NLL_{MLP}+0.01.
$$

Strong gate 要求：

$$
ECE_{KAN}<ECE_{MLP}.
$$

$$
NLL_{KAN}<NLL_{MLP}.
$$

### 2.6 ValLossAUC / time AUC gate

v7.1 没有完整证明 time AUC。v7.2 必须记录：

$$
AUC_{step}^{val\_loss}
=
\sum_i
\frac{L_i+L_{i-1}}{2}
(s_i-s_{i-1}).
$$

$$
AUC_{time}^{val\_loss}
=
\sum_i
\frac{L_i+L_{i-1}}{2}
(t_i-t_{i-1}).
$$

Significant efficient advantage 要求：

$$
AUC_{time,KAN}^{val\_loss}
\leq
AUC_{time,MLP}^{val\_loss}.
$$

如果 KAN accuracy 更高但 time AUC 更差很多，它是 expressivity proof，不是 efficient Beyond-MLP proof。

### 2.7 Classwise non-collapse gate

必须记录每个 dataset 的 classwise accuracy。禁止通过牺牲少数类换取 macro mean。

Classwise pass 要求：

$$
\max_c \left(Acc_{MLP,c}-Acc_{KAN,c}\right)\leq0.05.
$$

并且 hard class pair 不允许退化超过：

$$
0.03.
$$

---

## 3. v7.1 结果复盘与问题定位

### 3.1 已经达成的内容

v7.1 真实达成了以下内容：

```text
1. v7.1 real-only runner 跑通；
2. no-fake / no-proxy 审计通过；
3. strict KAN head 替代 linear head 在 task 侧成功；
4. H1/H2/H3/H4/H5/H6 全部通过 basic task gate；
5. H3 是当前最佳 strict task candidate；
6. K0/K0s grouped g2 证明 structured bridge 有 task potential；
7. K1/K2 grouped g8/g16 memory 接近 S2。
```

这说明 v7.0 中的 dense D3 signal 不是只靠普通 linear head。strict KAN head 可以保住 basic Beyond-MLP task improvement。

### 3.2 仍未达成的内容

v7.1 没有达成完整目标，核心原因是：

```text
1. H3 efficiency fail:
   memory ratio = 1.3031
   step ratio = 3.6324

2. H4 efficiency fail:
   memory ratio = 1.2956
   step ratio = 2.6287

3. K0/K0s step fail:
   K0 memory ratio = 1.1489
   K0 step ratio = 5.7211

4. K1/K2 Python grouped implementation step fail:
   K1 memory ratio = 1.0430
   K1 step ratio = 18.6210
   K2 memory ratio = 1.0463
   K2 step ratio = 36.0729

5. full grad relerr/cos gate 未完成；
6. 没有 confidence interval / p-value / effect size；
7. 没有 ValLossAUC/time AUC。
```

### 3.3 当前真正 blocker

当前 blocker 不是 task accuracy 本身，也不是简单 optimizer。当前 blocker 是：

$$
\boxed{
\text{efficiency\_kernelization + significant\_advantage\_proof}
}
$$

也就是说，需要同时解决两个问题：

```text
1. 把 H3/H4/K0/K1/K2 做成 S2/S1 kernel-native candidate；
2. 证明该 candidate 对 MLP 有显著优势，而不是略高一点。
```

---

## 4. v7.2 禁止事项

第一，不允许使用 fake data、proxy rows、固定占位 ratio 或手填结论。所有未实现内容必须写成：

```text
not_implemented
not_run
not_applicable
metric_unavailable
```

第二，不允许把 H3/H4 的 basic pass 写成显著优势。必须通过 significance gate。

第三，不允许把 H3/H4 的 task pass 写成完整成功。它们没有 S2 efficiency。

第四，不允许把 K1/K2 的 memory near-pass 写成成功。它们的 step ratio 极慢，目前只是 memory model reference。

第五，不允许继续以 optimizer sweep 为主线。已有结果显示 task signal 的关键不是已有 optimizer 小参数。

第六，不允许跳过 full gradient correctness。任何 route-eligible candidate 必须有：

```text
grad_relerr_max
grad_cos_min
forward_relerr
one-step loss delta
rollback error
```

第七，不允许用 test accuracy 调参。candidate selection 只能用 validation、holdout、efficiency 和 diagnostic metrics；test 只用于最终报告。

第八，不允许把非 KAN trainable head 或普通 dense mixer 作为 mainline success。

第九，不允许在没有 S2 的情况下进行 official task claim。S2 只允许 diagnostic claim，S1/S0 才能 official。

第十，不允许只报 mean ratio。每个 candidate 必须报告：

```text
mean
std
min
max
worst shape
batch/depth heatmap
```

第十一，不允许只报告 validation gap。必须报告：

```text
confidence interval
p value
effect size
test consistency
ECE/NLL
ValLossAUC_time
classwise non-collapse
```

---

## 5. v7.2 核心假设

### H1：当前 task gain 是真实但尚未证明显著

H1 假设：H3/H4 的 validation gain 不是偶然，但 v7.1 的 3-seed / 512-val split 还不足以证明显著优势。

H1 成立标准：

$$
CI_{95\%,macro}^{low}>0.
$$

至少两个 datasets：

$$
CI_{95\%,d}^{low}>0.
$$

并且：

$$
\Delta Acc_{macro,val}\geq0.02.
$$

若 H1 不成立，则 H3/H4 只能算 weak positive signal，不得作为 Beyond-MLP claim。

---

### H2：H4 是更现实的 kernelization 起点

H3 task 最好，但 step ratio 为 `3.6324`。H4 task 仍过 basic gate，val gap `+0.0169`，step ratio 降到 `2.6287`。H2 假设：H4 是更现实的 dense kernelization 起点。

H2 成立标准：

```text
H4 reproduction keeps val gap >= +0.010
H4 kernelized variant retains task within 0.01 of H4
H4 kernelized variant improves step by >=40%
H4 kernelized variant improves memory by >=15%
```

即：

$$
\frac{T_{H4-kernel}}{T_{H4}}\leq0.60.
$$

$$
\frac{M_{H4-kernel}}{M_{H4}}\leq0.85.
$$

---

### H3：H3/H4 efficiency fail 来自 materialization，而不是数学必然

H3 假设：H3/H4 的 slow path 来自 dense basis、gate activation、grad gate、grad poly、rbf head temp 的 materialization，而不是 poly2-gate 数学不可高效。

H3 成立要求 P2 attribution 中至少一项成立：

$$
\frac{M_{gate/basis/temp}}{M_{peak}}\geq0.35.
$$

or:

$$
\frac{T_{materialization}}{T_{step}}\geq0.60.
$$

H3 repair 成功标准：

$$
\frac{M_{kernelized}}{M_{parent}}\leq0.80.
$$

or:

$$
\frac{T_{kernelized}}{T_{parent}}\leq0.70.
$$

---

### H4：K0/K1/K2 的 step fail 来自 Python/grouped fragmentation

K0/K1/K2 目前 step 很慢，但 K1/K2 memory 已接近 S2。H4 假设：这是 implementation failure，不是 grouped primitive failure。

H4 成立标准：

$$
\frac{T_{fused}}{T_{python}}\leq0.30
$$

for K0, and:

$$
\frac{T_{fused}}{T_{python}}\leq0.20
$$

for K1/K2.

同时要求：

$$
r_{mem,fused}\leq1.05.
$$

and:

$$
Acc_{val,fused}\geq Acc_{val,parent}-0.005.
$$

---

### H5：显著优势必须在 test / classwise / calibration 上不崩

H5 假设：真正的 Beyond-MLP 不是 validation macro trick，而是多指标一致改善。

H5 成立标准：

$$
\Delta Acc_{macro,test}\geq0.015.
$$

$$
ECE_{KAN}\leq ECE_{MLP}+0.005.
$$

$$
NLL_{KAN}\leq NLL_{MLP}+0.01.
$$

Classwise：

$$
\max_c(Acc_{MLP,c}-Acc_{KAN,c})\leq0.05.
$$

---

### H6：如果 H4/K0/K1 kernelization 仍无法达到 S2，则 poly2-gate family 需要新 primitive

H6 是 Stop/Go 假设。如果完成：

```text
H4 materialization-free kernel
H3 head kernelization
K0 fused grouped-g2
K1/K2 fused grouped-g8/g16
full gradient correctness
significance audit
```

仍没有 S2 + significant candidate，则不能继续小修小补，应设计新的 function-space primitive。

---

## 6. Candidate 设计

### 6.1 Baselines

```text
B0-MLP-autograd-reference
B1-D3-dense-linear-oracle
B2-H3-strict-rbf-poly-exp-head
B3-H4-depth2-strict-poly2-gate
B4-K0-grouped-g2-poly2-gate
B5-K0s-grouped-g2-shuffle
B6-K1-grouped-g8
B7-K2-grouped-g16
```

### 6.2 Dense H4 kernelization candidates

```text
D0-H4-current
D1-H4-no-full-basis-materialization
D2-H4-gate-temp-streaming
D3-H4-grad-gate-streaming
D4-H4-grad-poly-streaming
D5-H4-materialization-free-forward
D6-H4-materialization-free-backward
D7-H4-fused-forward-backward
D8-H4-factorized-gate-r4
D9-H4-factorized-gate-r8
D10-H4-tiled-dense-gate
D11-H4-S2-combo
```

### 6.3 H3 head kernelization candidates

```text
R0-H3-current
R1-H3-rbf-head-no-full-basis
R2-H3-rbf-head-streaming-distance
R3-H3-rbf-head-fused-forward
R4-H3-rbf-head-fused-backward
R5-H3-rbf-head-shared-buffer
R6-H3-rbf-head-S2-combo
```

### 6.4 Grouped kernelization candidates

```text
G0-K0-current-python-g2
G1-K0-fused-g2-forward
G2-K0-fused-g2-backward
G3-K0-fused-g2-forward-backward
G4-K0-fused-g2-shuffle
G5-K1-fused-g8-forward-backward
G6-K2-fused-g16-forward-backward
G7-K1-fused-g8-with-g2-bridge
G8-K2-fused-g16-with-rank2-bridge
G9-grouped-S2-combo
```

### 6.5 Hybrid candidates

```text
Y0-H4-dense-layer1+K1-grouped-layer2
Y1-H4-dense-head+K1-body
Y2-K1-body+H3-strict-head
Y3-K0-g2-first-layer+K1-g8-hidden
Y4-K1-g8-body+rank2-crossgroup-bridge
Y5-H4-factorized-gate-r4+K1-memory-path
```

---

## 7. 实验阶段总览

v7.2 分为十六个阶段：

```text
P0: v7.1 reproduction and contract
P1: significance audit for H3/H4
P2: full gradient correctness gate
P3: efficiency live-set attribution
P4: H4 dense materialization-free kernelization
P5: H3 strict head kernelization
P6: K0/K1/K2 grouped fused kernelization
P7: hybrid dense-grouped candidate construction
P8: microkernel benchmark and correctness
P9: 3-dataset task gate for S2-near candidates
P10: full-step efficiency profiler for selected candidates
P11: ValLossAUC/time AUC verification
P12: one-step probe and rollback
P13: candidate co-selection
P14: official task re-entry if S1/S0
P15: sample efficiency / robustness diagnostic
P16: route decision and artifact audit
```

P0-P3 是复现、显著性、梯度和 attribution。  
P4-P7 是核心 kernelization。  
P8-P12 是 correctness + task + efficiency 证据链。  
P13-P16 是路线选择和最终结论。

---

## 8. P0：v7.1 reproduction and contract

### 8.1 目的

确认 v7.2 与 v7.1 final run 可比，并锁定 route-eligible candidate 的 purity / no-fake / no-proxy / manual path。

### 8.2 必跑对象

```text
B0-MLP-autograd-reference
B1-B3-dense-linear-oracle
B2-H3-strict-rbf-poly-exp-head
B3-H4-depth2-strict-poly2-gate
B4-K0-grouped-g2-poly2-gate
B5-K0s-grouped-g2-shuffle
B6-K1-grouped-g8
B7-K2-grouped-g16
```

### 8.3 必须记录字段

```text
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
kan_trainable_param_count
head_type
head_is_kan
edge_param_count
grad_relerr_max
grad_cos_min
memory_ratio_mean
step_ratio_mean
forward_ratio_mean
backward_ratio_mean
val_acc_mean
test_acc_mean
ECE
NLL
reproduction_delta_val_acc
reproduction_delta_memory_ratio
reproduction_delta_step_ratio
```

### 8.4 判断标准

Reproduction pass：

$$
|\operatorname{Acc}_{val,H3,v72}-0.8628|\leq0.015.
$$

$$
|\operatorname{Acc}_{val,H4,v72}-0.8609|\leq0.015.
$$

Efficiency reproduction pass：

$$
|r_{mem,H3,v72}-1.3031|\leq0.05.
$$

$$
|r_{step,H4,v72}-2.6287|\leq0.20.
$$

StrictPass：

```text
nonKAN_param_count = 0
manual_backward_available = 1
uses_loss_backward = 0
uses_torch_autograd_graph = 0
fake_data_used = 0
proxy_row_used = 0
```

### 8.5 可视化

```text
p0_reproduction_task_bar.svg
p0_reproduction_efficiency_bar.svg
p0_contract_heatmap.svg
p0_candidate_position_pareto.svg
```

---

## 9. P1：significance audit for H3/H4

### 9.1 目的

P1 专门回答用户关心的问题：现在对比 MLP 是否有显著优势。P1 不做新 kernel，只对 H3/H4 进行显著性验证。

### 9.2 必跑对象

```text
MLP-autograd-reference
H3-strict-rbf-poly-exp-head
H4-depth2-strict-poly2-gate
B3-dense-linear-oracle
```

### 9.3 设置

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2,3,4,5,6,7,8,9

train/val/test:
  1536/512/512

steps:
  240

logging:
  every 20 steps
```

### 9.4 必须记录字段

```text
dataset
seed
candidate
val_acc
test_acc
train_acc
val_loss
test_loss
ECE
NLL
val_gap_vs_MLP
test_gap_vs_MLP
classwise_acc
hard_pair_acc
confidence_mean
wrong_confidence_mean
ValLossAUC_step
ValLossAUC_time
wall_clock_time_sec
```

### 9.5 统计字段

```text
candidate
dataset
mean_val_gap
std_val_gap
mean_test_gap
std_test_gap
paired_t_stat
paired_p_value
wilcoxon_p_value
bootstrap_ci95_low
bootstrap_ci95_high
holm_corrected_p
cohen_h
seed_win_count
dataset_pass_practical
dataset_pass_statistical
```

### 9.6 判断标准

SignificantTaskPass：

```text
macro mean val gap >= +0.02
macro bootstrap CI95 low > 0
at least two datasets mean val gap >= +0.02
at least two datasets CI95 low > 0
Holm corrected p < 0.05
macro test gap >= +0.015
no dataset test gap < -0.005
ECE/NLL not worse
classwise non-collapse
```

公式：

$$
\Delta Acc_{macro,val}\geq0.02.
$$

$$
CI_{95\%,macro}^{low}>0.
$$

$$
p_{Holm}<0.05.
$$

### 9.7 可视化

```text
p1_seedwise_val_gap_boxplot.svg
p1_bootstrap_ci_by_dataset.svg
p1_val_test_gap_scatter.svg
p1_effect_size_cohen_h_bar.svg
p1_classwise_noncollapse_heatmap.svg
p1_significance_dashboard.svg
```

---

## 10. P2：full gradient correctness gate

### 10.1 目的

补齐 v7.1 route 中缺失的 full grad relerr/cos gate。

### 10.2 必跑对象

```text
H3
H4
K0
K0s
K1
K2
all newly implemented D/H/R/G/Y candidates
```

### 10.3 检查设置

```text
small check:
  batch = 8

task-shape check:
  batch = 128

datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seed:
  0
```

### 10.4 必须记录字段

```text
candidate
dataset
batch_size
forward_relerr_max
loss_relerr
grad_relerr_max
grad_relerr_mean
grad_cos_min
grad_cos_mean
finite_grad
finite_forward
finite_loss
nan_count
inf_count
one_step_loss_before
one_step_loss_after
one_step_loss_delta
rollback_error
```

### 10.5 判断标准

GradPass：

$$
\operatorname{grad\_relerr\_max}\leq10^{-4}.
$$

$$
\operatorname{grad\_cos\_min}\geq0.999.
$$

OneStepPass：

$$
\Delta L_{\text{train}}<0.
$$

Rollback pass：

$$
\operatorname{rollback\_error}<10^{-8}.
$$

If candidate fails GradPass, it cannot enter route selection even if task accuracy is high.

### 10.6 可视化

```text
p2_grad_relerr_lollipop.svg
p2_grad_cos_bar.svg
p2_one_step_loss_delta.svg
p2_rollback_error_bar.svg
```

---

## 11. P3：efficiency live-set attribution

### 11.1 目的

解释 H3/H4/K0/K1/K2 为什么效率失败，找到可修 source。

### 11.2 必跑对象

```text
MLP-autograd-reference
H3
H4
K0
K1
K2
```

### 11.3 Shape grid

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

batch sizes:
  128
  256
  512

depths:
  native depth

warmup / measure:
  50 / 200
```

### 11.4 必须记录字段

#### Timing

```text
forward_time_ms
backward_time_ms
update_time_ms
step_time_ms
dense_gate_forward_ms
basis_eval_ms
rbf_head_forward_ms
gate_temp_ms
grad_gate_ms
grad_poly_ms
grouped_loop_ms
shuffle_ms
crossgroup_bridge_ms
kernel_launch_overhead_proxy_ms
python_overhead_ms
```

#### Memory

```text
peak_allocated_MB
peak_reserved_MB
MLP_peak_allocated_MB
memory_ratio
dense_basis_MB
gate_activation_MB
gate_temp_MB
grad_gate_MB
grad_poly_MB
rbf_head_temp_MB
grouped_output_MB
shuffle_temp_MB
crossgroup_bridge_temp_MB
manual_cache_MB
optimizer_state_MB
largest_live_tensor_MB
top1_memory_source
top2_memory_source
top3_memory_source
unknown_memory_fraction
```

#### Kernelization

```text
torch_op_count
triton_kernel_count
elementwise_kernel_count
gemm_count
allocation_proxy_count
materialized_tensor_count
python_loop_count
group_loop_count
```

### 11.5 判断标准

Attribution pass：

```text
top3 memory sources identified
unknown_memory_fraction <= 0.10
component timing explains >= 90% step time
```

Dense materialization blocker：

$$
\frac{M_{\text{gate/basis/temp}}}{M_{\text{peak}}}\geq0.35.
$$

or:

$$
\frac{T_{\text{dense materialization}}}{T_{\text{step}}}\geq0.60.
$$

Grouped implementation blocker：

$$
\frac{T_{\text{python/group loop}}}{T_{\text{step}}}\geq0.50.
$$

### 11.6 可视化

```text
p3_step_time_waterfall.svg
p3_memory_live_set_waterfall.svg
p3_top_memory_sources_bar.svg
p3_kernel_count_by_candidate.svg
p3_dense_vs_grouped_efficiency_pareto.svg
```

---

## 12. P4：H4 dense materialization-free kernelization

### 12.1 目的

H4 是当前最现实的 dense task+efficiency 折中点。P4 从 H4 出发，减少 materialization 和 kernel fragmentation。

### 12.2 Candidate packages

```text
D0-H4-current
D1-H4-no-full-basis-materialization
D2-H4-gate-temp-streaming
D3-H4-grad-gate-streaming
D4-H4-grad-poly-streaming
D5-H4-materialization-free-forward
D6-H4-materialization-free-backward
D7-H4-fused-forward-backward
D8-H4-factorized-gate-r4
D9-H4-factorized-gate-r8
D10-H4-tiled-dense-gate
D11-H4-S2-combo
```

### 12.3 必须记录字段

```text
package
implementation_status
materializes_basis
materializes_gate
materializes_grad_gate
factor_rank
tile_size
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
forward_ratio_mean
backward_ratio_mean
memory_improvement_vs_H4
step_improvement_vs_H4
peak_allocated_MB
basis_temp_MB
gate_temp_MB
grad_gate_MB
grad_poly_MB
kernel_count_total
torch_op_count
triton_kernel_count
grad_relerr_max
grad_cos_min
val_acc_mean
test_acc_mean
val_gap_vs_MLP
ECE
NLL
```

### 12.4 判断标准

Kernelization useful：

$$
\frac{M_{\text{candidate}}}{M_{\text{H4}}}\leq0.85.
$$

or:

$$
\frac{T_{\text{candidate}}}{T_{\text{H4}}}\leq0.70.
$$

S2Pass：

$$
r_{mem}\leq1.05.
$$

$$
r_{step}\leq1.50.
$$

Task preservation：

$$
Acc_{val,candidate}\geq Acc_{val,H4}-0.01.
$$

or:

$$
Acc_{val,candidate}\geq Acc_{val,MLP}-0.01.
$$

### 12.5 可视化

```text
p4_h4_kernelization_pareto.svg
p4_h4_memory_reduction_waterfall.svg
p4_h4_task_preservation_bar.svg
p4_h4_s2_boundary_plot.svg
```

---

## 13. P5：H3 strict head kernelization

### 13.1 目的

H3 是 task winner，但 step 更慢。P5 只在 H4 task preservation 失败或 H3 head-specific bottleneck 可修时作为并行路线。

### 13.2 Candidate packages

```text
R0-H3-current
R1-H3-rbf-head-no-full-basis
R2-H3-rbf-head-streaming-distance
R3-H3-rbf-head-fused-forward
R4-H3-rbf-head-fused-backward
R5-H3-rbf-head-shared-buffer
R6-H3-rbf-head-S2-combo
```

### 13.3 必须记录字段

```text
package
implementation_status
rbf_temp_MB
head_basis_MB
head_grad_temp_MB
head_forward_ms
head_backward_ms
memory_ratio_mean
step_ratio_mean
memory_improvement_vs_H3
step_improvement_vs_H3
grad_relerr_max
grad_cos_min
val_acc_mean
test_acc_mean
ECE
NLL
```

### 13.4 判断标准

H3 head repair useful：

$$
\frac{T_{\text{candidate}}}{T_{\text{H3}}}\leq0.70.
$$

or:

$$
\frac{M_{\text{candidate}}}{M_{\text{H3}}}\leq0.85.
$$

Task preservation：

$$
Acc_{val,candidate}\geq Acc_{val,H3}-0.005.
$$

### 13.5 可视化

```text
p5_h3_head_kernelization_bar.svg
p5_h3_head_task_preservation.svg
p5_h3_head_efficiency_pareto.svg
```

---

## 14. P6：K0/K1/K2 grouped fused kernelization

### 14.1 目的

K0/K1/K2 代表 memory-friendly structured route。P6 的目标是用 fused kernel 消除 Python/grouped loop overhead。

### 14.2 Candidate packages

```text
G0-K0-current-python-g2
G1-K0-fused-g2-forward
G2-K0-fused-g2-backward
G3-K0-fused-g2-forward-backward
G4-K0-fused-g2-shuffle
G5-K1-fused-g8-forward-backward
G6-K2-fused-g16-forward-backward
G7-K1-fused-g8-with-g2-bridge
G8-K2-fused-g16-with-rank2-bridge
G9-grouped-S2-combo
```

### 14.3 必须记录字段

```text
package
group_size
uses_shuffle
bridge_rank
implementation_status
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
forward_ratio_mean
backward_ratio_mean
memory_improvement_vs_parent
step_improvement_vs_parent
group_loop_count
python_loop_count
kernel_count_total
triton_kernel_count
grouped_output_MB
shuffle_temp_MB
bridge_temp_MB
grad_relerr_max
grad_cos_min
val_acc_mean
test_acc_mean
val_gap_vs_MLP
ECE
NLL
```

### 14.4 判断标准

Fused grouped useful：

$$
\frac{T_{\text{fused}}}{T_{\text{python}}}\leq0.30.
$$

Task preservation：

$$
Acc_{val,fused}\geq Acc_{val,parent}-0.005.
$$

S2Pass：

$$
r_{mem}\leq1.05.
$$

$$
r_{step}\leq1.50.
$$

If memory passes but task fails, candidate is efficiency diagnostic only.  
If task passes but step remains high, candidate requires lower-level kernelization.

### 14.5 可视化

```text
p6_grouped_fused_step_speedup.svg
p6_grouped_memory_task_pareto.svg
p6_group_loop_elimination_bar.svg
p6_grouped_s2_boundary_plot.svg
```

---

## 15. P7：hybrid dense-grouped candidate construction

### 15.1 目的

如果 pure dense H4 kernelization 仍 memory-heavy，而 pure grouped K1/K2 task gap 大，则 P7 构造 hybrid candidate。

### 15.2 Candidate packages

```text
Y0-H4-dense-layer1+K1-grouped-layer2
Y1-H4-dense-head+K1-body
Y2-K1-body+H3-strict-head
Y3-K0-g2-first-layer+K1-g8-hidden
Y4-K1-g8-body+rank2-crossgroup-bridge
Y5-H4-factorized-gate-r4+K1-memory-path
```

### 15.3 必须记录字段

```text
candidate
dense_layer_count
grouped_layer_count
hybrid_pattern
memory_ratio_mean
step_ratio_mean
forward_ratio_mean
backward_ratio_mean
val_acc_mean
test_acc_mean
val_gap_vs_MLP
ECE
NLL
feature_rank
margin_p10
classwise_acc
grad_relerr_max
grad_cos_min
```

### 15.4 判断标准

Hybrid useful：

$$
Acc_{val,candidate}\geq Acc_{val,MLP}-0.01.
$$

and:

$$
r_{mem}\leq1.05.
$$

$$
r_{step}\leq1.50.
$$

If hybrid improves task but violates S2, it is diagnostic only.

### 15.5 可视化

```text
p7_hybrid_efficiency_task_pareto.svg
p7_hybrid_pattern_bar.svg
p7_hybrid_classwise_delta.svg
```

---

## 16. P8：microkernel benchmark and correctness

### 16.1 目的

对 P4-P7 中有效组件做 microkernel benchmark，确认哪些值得进入 full package。

### 16.2 Microkernels

```text
MK0-poly2-gate-forward-current
MK1-poly2-gate-forward-fused
MK2-poly2-gate-backward-current
MK3-poly2-gate-backward-fused
MK4-grad-gate-streaming
MK5-grad-poly-streaming
MK6-rbf-head-forward-fused
MK7-rbf-head-backward-fused
MK8-grouped-g2-forward-fused
MK9-grouped-g2-backward-fused
MK10-grouped-g8-forward-backward-fused
```

### 16.3 必须记录字段

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
grad_relerr
grad_cos
forward_relerr
materialized_tensor_count
temp_MB
kernel_count
allocation_count
bandwidth_estimate_GBps
```

### 16.4 判断标准

Microkernel useful：

$$
\frac{T_{\text{candidate}}}{T_{\text{current}}}\leq0.80.
$$

or:

$$
\frac{M_{\text{candidate}}}{M_{\text{current}}}\leq0.80.
$$

Microkernel enters package if：

```text
grad_relerr < 1e-4
grad_cos > 0.999
time_ratio_vs_current <= 0.90
memory_ratio_vs_current <= 0.95
```

### 16.5 可视化

```text
p8_microkernel_memory_time_pareto.svg
p8_microkernel_grad_correctness.svg
p8_materialization_reduction_bar.svg
```

---

## 17. P9：3-dataset task gate for S2-near candidates

### 17.1 目的

P9 对 P4-P7 中达到 S2-near 或 task-preserving 的 candidates 运行 3 datasets x 3 seeds task gate。

### 17.2 必跑对象

```text
MLP-autograd-reference
H3
H4
K0
best-H4-kernelized
best-H3-head-kernelized
best-grouped-fused
best-hybrid
```

### 17.3 设置

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

train/val/test:
  1536/512/512

steps:
  240

logging:
  every 20 steps
```

### 17.4 必须记录字段

```text
dataset
seed
candidate
train_acc_curve
val_acc_curve
train_loss_curve
val_loss_curve
val_acc
test_acc
ECE
NLL
val_gap_vs_MLP
test_gap_vs_MLP
ValLossAUC_step
ValLossAUC_time
feature_rank
margin_p10
classwise_acc
KMNIST_hard_pair_acc
step_time_ms
memory_ratio_mean
```

### 17.5 判断标准

Basic task pass：

$$
Acc_{val,KAN,d}\geq Acc_{val,MLP,d}-0.01
$$

for all datasets.

Significant diagnostic candidate：

$$
\Delta Acc_{macro,val}\geq0.02.
$$

Task preservation：

$$
Acc_{val,candidate}\geq Acc_{val,parent}-0.01.
$$

### 17.6 可视化

```text
p9_val_acc_by_dataset_seed.svg
p9_val_loss_vs_step.svg
p9_val_loss_vs_time.svg
p9_task_efficiency_pareto.svg
p9_classwise_heatmap.svg
```

---

## 18. P10：full-step efficiency profiler for selected candidates

### 18.1 目的

对 P9 selected candidates 运行 full-step profiler，确认 efficiency gate。

### 18.2 必跑对象

```text
MLP-autograd-reference
H3
H4
K0
best-H4-kernelized
best-H3-head-kernelized
best-grouped-fused
best-hybrid
```

### 18.3 Shape grid

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

batch sizes:
  128
  256
  512

depths:
  native depth

warmup / measure:
  50 / 200
```

### 18.4 必须记录字段

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
grad_relerr_max
grad_cos_min
kernel_count_total
torch_op_count
triton_kernel_count
allocation_proxy_count
largest_live_tensor_MB
top1_memory_source
top2_memory_source
top3_memory_source
```

### 18.5 判断标准

S2：

$$
r_{mem}\leq1.05.
$$

$$
r_{step}\leq1.50.
$$

S1：

$$
r_{mem}<1.00.
$$

$$
r_{step}\leq1.35.
$$

Strong efficiency：

$$
r_{mem}\leq0.95.
$$

$$
r_{step}\leq1.20.
$$

### 18.6 可视化

```text
p10_efficiency_pareto_selected.svg
p10_batch_depth_memory_heatmap.svg
p10_batch_depth_step_heatmap.svg
p10_s1_s2_boundary_plot.svg
```

---

## 19. P11：ValLossAUC/time AUC verification

### 19.1 目的

验证 candidate 是否只是慢而准，还是真正 wall-clock 有优势。

### 19.2 必跑对象

```text
MLP-autograd-reference
H3
H4
best-S2-candidate
best-S1-candidate, if exists
```

### 19.3 设置

```text
steps:
  240
  optional 480 for final candidate

logging:
  every 20 steps

seeds:
  0,1,2,3,4

datasets:
  MNIST
  Fashion-MNIST
  KMNIST
```

### 19.4 必须记录

```text
val_acc_at_120
val_acc_at_240
val_acc_at_480
test_acc_at_240
test_acc_at_480
ValLossAUC_step_0_240
ValLossAUC_time_0_240
ValLossAUC_step_0_480
ValLossAUC_time_0_480
late_slope_val_acc
late_slope_val_loss
overfit_index
time_to_target
```

### 19.5 判断标准

Time AUC pass：

$$
ValLossAUC_{time,KAN}\leq ValLossAUC_{time,MLP}.
$$

Step AUC pass：

$$
ValLossAUC_{step,KAN}\leq ValLossAUC_{step,MLP}.
$$

If accuracy improves but time AUC fails badly, candidate remains expressivity proof.

### 19.6 可视化

```text
p11_val_loss_auc_time_bar.svg
p11_val_loss_vs_time.svg
p11_late_slope_bar.svg
p11_time_to_target_bar.svg
```

---

## 20. P12：one-step probe and rollback

### 20.1 目的

对 selected candidate 做 one-step descent 和 rollback。任何进入 route 的 candidate 必须通过。

### 20.2 必跑对象

```text
best-H4-kernelized
best-H3-head-kernelized
best-grouped-fused
best-hybrid
```

### 20.3 必须记录字段

```text
candidate
dataset
train_loss_before
train_loss_after
holdout_loss_before
holdout_loss_after
val_loss_before
val_loss_after
train_loss_delta
holdout_loss_delta
val_loss_delta
bad_step_flag
rollback_error
update_norm
update_over_param
grad_norm
gate_ablation_delta_loss
poly2_ablation_delta_loss
crossgroup_ablation_delta_loss
```

### 20.4 判断标准

OneStepPass：

$$
\Delta L_{\text{train}}<0.
$$

Holdout safe：

$$
\Delta L_{\text{holdout}}\leq0.02L_{\text{holdout,before}}.
$$

Rollback：

$$
rollback\_error<10^{-8}.
$$

Ablation contribution：

$$
|\Delta L_{\text{ablation}}|>10^{-4}.
$$

### 20.5 可视化

```text
p12_loss_before_after.svg
p12_bad_step_heatmap.svg
p12_update_norm_vs_loss_delta.svg
p12_ablation_delta_bar.svg
```

---

## 21. P13：candidate co-selection

### 21.1 目的

统一选择 candidate。不能只按 accuracy 或 efficiency。

### 21.2 Survivor 类型

```text
S0:
  S1 efficiency + SignificantTaskPass + ECE/NLL not worse

S1:
  S2 efficiency + SignificantTaskPass

S2:
  S2 efficiency + BasicTaskPass but not significant

S3:
  task significant but efficiency fail

S4:
  efficiency pass but task gap persists

S5:
  gradient fail

S6:
  no improvement
```

### 21.3 必须记录字段

```text
candidate
survivor_type
strict_pass
grad_pass
s2_pass
s1_pass
basic_task_pass
significant_task_pass
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
val_acc_mean
test_acc_mean
val_gap_vs_MLP
test_gap_vs_MLP
ci95_low
ci95_high
holm_p
ECE
NLL
ValLossAUC_time_delta
classwise_noncollapse_pass
route_recommendation
```

### 21.4 可视化

```text
p13_efficiency_task_significance_pareto.svg
p13_survivor_type_dashboard.svg
p13_metric_radar_chart.svg
p13_route_candidate_scorecard.svg
```

---

## 22. P14：official task re-entry if S1/S0

### 22.1 打开条件

Official task opens only if：

```text
survivor_type == S0
S1Pass = 1
GradPass = 1
StrictPass = 1
OneStepPass = 1
no_fake = 1
no_proxy = 1
```

### 22.2 必跑对象

```text
MLP-autograd-reference
best-v72-official-candidate
H3 reference
H4 reference
```

### 22.3 必须记录字段

```text
train_loss_curve
val_loss_curve
test_acc
val_acc
ECE
NLL
ValLossAUC_step
ValLossAUC_time
ValAccAUC_step
ValAccAUC_time
time_to_target_loss
time_to_target_acc
step_time_ms
wall_clock_time_sec
backward_memory_ratio
samples_per_second
feature_rank
margin_p10
classwise_acc
seedwise_failure_reason
```

### 22.4 判断标准

Official task pass：

$$
Acc_{KAN,d}\geq Acc_{MLP,d}-0.01
$$

for all datasets.

Significant official pass：

$$
\Delta Acc_{macro,val}\geq0.02,
$$

$$
CI_{95\%,macro}^{low}>0,
$$

$$
p_{Holm}<0.05.
$$

Memory pass：

$$
r_{mem}<1.00.
$$

Wall-clock pass：

$$
ValLossAUC_{time,KAN}\leq ValLossAUC_{time,MLP}.
$$

### 22.5 可视化

```text
p14_official_val_loss_vs_time.svg
p14_official_accuracy_vs_time.svg
p14_official_task_efficiency_pareto.svg
p14_seedwise_delta_plot.svg
```

---

## 23. P15：sample efficiency / robustness diagnostic

### 23.1 目的

只有在出现 S2 + significant task candidate 后运行。它用于判断是否有 Beyond-MLP 的更强维度。

### 23.2 Data fractions

```text
5%
10%
25%
50%
100%
```

### 23.3 Noise settings

```text
label_noise:
  5%
  10%
  20%

input_noise:
  gaussian sigma 0.05
  gaussian sigma 0.10
  random erasing small
```

### 23.4 必跑对象

```text
MLP-autograd-reference
H3
best-v72-significant-candidate
```

### 23.5 必须记录

```text
data_fraction
noise_type
noise_level
val_acc
test_acc
accuracy_drop_vs_clean
ECE
NLL
sample_efficiency_auc
robustness_auc
confidence_mean
wrong_confidence_mean
margin_p10
feature_rank
```

### 23.6 判断标准

Sample efficiency useful：

$$
AUC_{data,KAN}>AUC_{data,MLP}.
$$

Robustness useful：

$$
AccDrop_{KAN}<AccDrop_{MLP}
$$

for at least two noise settings.

Calibration useful：

$$
ECE_{KAN}<ECE_{MLP}-0.01.
$$

### 23.7 可视化

```text
p15_accuracy_vs_data_fraction.svg
p15_sample_efficiency_auc_bar.svg
p15_accuracy_under_noise.svg
p15_ece_under_noise.svg
p15_robustness_auc_bar.svg
```

---

## 24. P16：route decision and artifact audit

### 24.1 Route cases

```text
R1-SignificantPureKANNG-Pass:
  S1 + SignificantTaskPass + ECE/NLL/time AUC not worse.
  Move to 5-seed/10-seed confirm and patch/token expansion.

R2-S2-SignificantTaskPass:
  S2 + SignificantTaskPass but memory still >1.0.
  Continue S1 memory repair.

R3-TaskPositiveButNotSignificant:
  task gap positive but fails significance/practical threshold.
  Increase seeds or revise task claim.

R4-TaskSignificantEfficiencyFail:
  significant task advantage but no S2.
  Continue kernelization.

R5-EfficiencyPassTaskNotSignificant:
  efficiency solved but task advantage not significant.
  Focus expressivity/generalization.

R6-H4KernelizationUseful:
  H4 kernelization improves efficiency substantially but not yet S2.
  Continue materialization-free kernel.

R7-GroupedKernelizationUseful:
  grouped fused path reduces step >=70% and preserves task.
  Continue grouped route.

R8-NoImprovement:
  no candidate improves over v7.1.
  Need new primitive.

R9-ContractFail:
  fake/proxy/nonKAN/grad fail.
  Reject candidate.
```

### 24.2 JSON 输出字段

```text
route
best_candidate
best_family
best_memory_ratio
best_step_ratio
best_backward_ratio
best_forward_ratio
best_val_acc
best_test_acc
val_gap_vs_MLP
test_gap_vs_MLP
ci95_low
ci95_high
holm_p
cohen_h
ECE_delta
NLL_delta
ValLossAUC_time_delta
survivor_type
strict_pass
grad_pass
s2_pass
s1_pass
basic_task_pass
significant_task_pass
official_task_opened
diagnostic_task_opened
primary_blocker
next_required_implementation
no_fake
no_proxy
```

### 24.3 Artifacts

```text
run_manifest.json
provenance_audit.csv
candidate_registry.csv
gate_config.json
p0_contract.csv
p1_significance_audit.csv
p2_full_gradient_correctness.csv
p3_efficiency_live_set_attribution.csv
p4_h4_kernelization.csv
p5_h3_head_kernelization.csv
p6_grouped_fused_kernelization.csv
p7_hybrid_candidates.csv
p8_microkernel_benchmark.csv
p9_task_gate.csv
p9_task_trace.csv
p10_efficiency_profiler.csv
p11_val_loss_auc_time.csv
p12_one_step_probe.csv
p13_candidate_selection.csv
p14_official_task_reentry.csv
p15_sample_robustness.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

### 24.4 Failure taxonomy

```text
F1_memory_fail
F2_step_time_fail
F3_gradient_correctness_fail
F4_task_not_significant
F5_validation_test_inconsistent
F6_classwise_collapse
F7_time_auc_fail
F8_dense_kernelization_fail
F9_grouped_kernelization_fail
F10_hybrid_fail
F11_nonKAN_violation
F12_fake_or_proxy_violation
F13_artifact_missing
```

### 24.5 必须生成总图

```text
figures/p13_efficiency_task_significance_pareto.svg
figures/p13_survivor_type_dashboard.svg
figures/p16_route_decision_dashboard.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 25. 第一轮推荐执行顺序

v7.2 不建议一开始全量跑 P0-P16。第一轮应该先回答最关键的问题。

### Step 1：P1 significance audit

先回答：

```text
H3/H4 是否真的显著优于 MLP？
还是只是 +1%-2% 的弱 positive signal？
```

如果 H3/H4 不能通过 significance gate，v7.2 不应再用“显著超越”措辞，只能说“正向 task signal”。

### Step 2：P2 full gradient correctness

确保 H3/H4/K0/K1/K2 和新 kernelized candidates 都有完整 GradPass。没有 GradPass，不进入 route。

### Step 3：P3 attribution

定位 H4/H3/K0/K1 的真实 efficiency blocker。不要盲写 fused kernel。

### Step 4：优先做 H4 materialization-free kernelization

H4 是当前最现实起点。目标先把：

$$
r_{step}=2.6287
$$

压到：

$$
r_{step}\leq1.50.
$$

同时 memory 从：

$$
r_{mem}=1.2956
$$

压到：

$$
r_{mem}\leq1.05.
$$

### Step 5：并行做 K1/K2 fused grouped kernel

K1/K2 memory 已经接近 S2，说明 memory model 有价值。只要 fused implementation 大幅降低 step，就可能成为最快进入 S2 的路线。

---

## 26. 成功与失败解释规则

### Case A：S1 + significant task pass

如果某 candidate 满足：

$$
r_{mem}<1.00,
$$

$$
r_{step}\leq1.35,
$$

并且：

$$
\Delta Acc_{macro,val}\geq0.02,
$$

$$
CI_{95\%,macro}^{low}>0,
$$

$$
p_{Holm}<0.05,
$$

则可以说 v7.2 建立了显著 Beyond-MLP PureKAN-NG candidate。

### Case B：S2 + significant task pass

如果 S2 成立且 significant task pass，但 memory 仍 $>1.00$，则 candidate 是 strong diagnostic success。下一轮继续 S1 memory repair。

### Case C：task positive but not significant

如果 val gap 为正，但 macro gap $<2\%$ 或 CI lower bound 不大于 0，则不能称为显著优势。只能写：

```text
positive but not statistically/practically significant
```

### Case D：task significant but efficiency fail

如果 H3/H4 通过 significance gate，但 S2 仍 fail，则当前 blocker 是 kernelization。不能称为 complete PureKAN-NG success。

### Case E：efficiency pass but task not significant

如果 kernelized candidate 进入 S2/S1，但 task advantage 不显著，则下一步要回到 expressivity / architecture，而不是再只修 kernel。

### Case F：K1/K2 fused 仍极慢

如果 fused grouped implementation 后 step 仍远高于 S2，则 grouped memory model 可能与当前 poly2-gate task signal 不兼容，需要新 primitive。

### Case G：no improvement

如果 v7.2 没有 candidate 在 significance 或 efficiency 任一方向实质改善，下一轮应停止小修，设计新 function-space primitive。

---

## 27. 最终建议

v7.2 的一句话策略是：

$$
\boxed{
\text{先证明 H3/H4 是否显著优于 MLP，再把 H4/K1/K2 做成 kernel-native S2/S1 candidate。}
}
$$

当前不能说“已经显著超过 MLP”。更准确的说法是：

```text
v7.1 已经证明 strict PureKAN 可以在 mean validation accuracy 上超过 MLP；
但优势幅度约 1.9%，尚未通过统计显著、实践显著、test consistency 和 efficiency gate。
```

因此 v7.2 的目标必须比 v7.1 更严格：

```text
1. macro val gap >= +2%
2. bootstrap CI lower bound > 0
3. corrected p < 0.05
4. macro test gap >= +1.5%
5. ECE/NLL not worse
6. no classwise collapse
7. S2 or S1 efficiency
8. full GradPass
```

只有满足这些条件，才能说：

$$
\boxed{
\text{PureKAN-NG 对 MLP 有显著优势。}
}
$$
