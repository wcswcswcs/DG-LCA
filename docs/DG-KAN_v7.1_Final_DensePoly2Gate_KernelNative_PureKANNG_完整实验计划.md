# DG-KAN v7.1 Final：Dense Poly2-Gate 表达力迁移到 Kernel-Native PureKAN-NG 的完整实验计划

> 本最终方案整合了两份方案：一份强调 **Dense-to-Kernel bridge、D3 advantage attribution、dense efficiency attribution、microkernel / fused kernelization、bridge / distillation / AUC / robustness / patch-token scaling**；另一份专家方案强调 **Strict purity gate、strict KAN head repair、candidate registry / provenance lock、structured poly2_gate candidate ladder、live-set repair candidates、第一轮执行顺序和明确停止条件**。  
> 本方案不是补丁，而是 v7.1 的完整最终实验计划。它以 v7.0 最新复盘为准：dense manual KAN 的 `D3-dense-poly2-gate-d3-warmup-smooth-240` 已经真实打开 basic Beyond-MLP task accuracy gate，但 dense D3 还不是 graph-free、kernel-native、S1/S2 official candidate；FastOpt 已经大幅降低 optimizer update overhead，但 D3 仍然 efficiency 失败。因此 v7.1 的核心任务是把 dense `poly2_gate` 的表达力迁移到 strict、graph-free、kernel-native、低内存 PureKAN-NG candidate。

---

## 0. v7.1 的总目标

v7.1 的目标不是继续证明 dense D3 accuracy 高，也不是继续盲调 grouped/T3 optimizer。v7.1 要回答一个更明确、更有价值的问题：

$$
\boxed{
\text{能否把 dense } poly2\_gate \text{ 的 Beyond-MLP 表达力压缩、稀疏化、低秩化、kernel 化，并保持 PureKAN contract？}
}
$$

v7.0 已经给出两个互补事实。

第一，official full run 中，`G2-O7-ManualAdamW-lr-high-fulltrain-240` 是最好的 grouped/T3/G2 candidate：

```text
memory ratio = 1.0311948157
step ratio = 0.8665947909
backward ratio = 0.7092462146
forward ratio = 1.0750472898
val acc mean = 0.8138020833
test acc mean = 0.7645399306
val gap vs MLP = -0.0325520833
route = R3-S2-TaskImproved
official task opened = false
diagnostic task opened = true
```

它在 primary datasets 上没有通过 basic Beyond-MLP task gate：

```text
MNIST gap = -0.0176
Fashion-MNIST gap = -0.0150
KMNIST gap = -0.0651
```

所以 grouped/T3/G2 是高效的 S2 diagnostic improvement，但不是 Beyond-MLP pass。

第二，用户继续要求后的 dense manual KAN targeted run 中，`D3-dense-poly2-gate-d3-warmup-smooth-240` 首次真实打开 basic Beyond-MLP task accuracy gate：

```text
val acc mean = 0.8670
test acc mean = 0.8218
val gap vs MLP = +0.0206
datasets within 1% MLP = 3/3
datasets KAN >= MLP = 3/3
```

D3 在各 dataset 上：

```text
MNIST:
  MLP val acc = 0.8861
  D3 val acc  = 0.9167
  gap = +0.0306
  D3 test acc = 0.9225

Fashion-MNIST:
  MLP val acc = 0.8268
  D3 val acc  = 0.8281
  gap = +0.0013
  D3 test acc = 0.8125

KMNIST:
  MLP val acc = 0.8262
  D3 val acc  = 0.8561
  gap = +0.0299
  D3 test acc = 0.7305
```

D3 的 calibration / NLL 也优于 MLP：

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

但是 dense D3 没有达到完整 PureKAN-NG hard goal。原始 dense D3 efficiency 是：

```text
D3 raw:
  step ratio mean = 2.7489
  memory ratio mean = 1.1982
  forward ratio mean = 2.5314
  backward ratio mean = 1.9199
  update ms mean = 0.9273
  survivor = FAIL 9/9
```

FastOpt 后：

```text
D3 fastopt:
  step ratio mean = 1.6810
  memory ratio mean = 1.1778
  forward ratio mean = 2.4131
  backward ratio mean = 1.9863
  update ms mean = 0.1171
  survivor = FAIL 9/9
```

FastOpt 说明 optimizer update overhead 已经大幅降低，但 dense D3 仍不能进入 S2：

$$
r_{\text{step,D3-fastopt}}=1.6810>1.50,
$$

$$
r_{\text{mem,D3-fastopt}}=1.1778>1.05.
$$

因此 v7.1 的路线判断是：

$$
\boxed{
\text{accuracy problem has an oracle solution; efficiency, purity, kernelization are now the blockers.}
}
$$

---

## 1. v7.1 的成功标准

v7.1 不允许只以一个指标宣布成功。一个 candidate 必须同时满足 purity、correctness、efficiency、task 四类条件。

### 1.1 最低可接受成功标准

$$
\boxed{
\text{StrictPass} \land \text{GradPass} \land \text{S2Pass} \land \text{BasicBeyondPass}
}
$$

这表示：candidate 已经是 strict graph-free PureKAN，梯度正确，至少满足 S2 diagnostic efficiency，并通过 basic Beyond-MLP task gate。

### 1.2 正式目标成功标准

$$
\boxed{
\text{StrictPass} \land \text{GradPass} \land \text{S1Pass} \land \text{BasicBeyondPass}
}
$$

这表示：candidate 可进入 official route，因为 memory 已低于 MLP，step 也满足 official gate。

### 1.3 强成功标准

$$
\boxed{
\text{StrictPass} \land \text{GradPass} \land \text{S1Pass} \land \text{StrongBeyondPass}
}
$$

StrongBeyondPass 要求：

$$
\operatorname{Acc}_{val,KAN}\geq\operatorname{Acc}_{val,MLP}+0.005
$$

on at least two datasets, while:

$$
\operatorname{ECE}_{KAN}\leq\operatorname{ECE}_{MLP},
$$

$$
\operatorname{NLL}_{KAN}\leq\operatorname{NLL}_{MLP},
$$

and:

$$
\operatorname{ValLossAUC}_{time,KAN}\leq\operatorname{ValLossAUC}_{time,MLP}.
$$

---

## 2. Gate 定义

### 2.1 Strict purity gate

每个 candidate 必须记录：

```text
non_kan_trainable_param_count
kan_trainable_param_count
uses_torch_loss_backward
uses_torch_autograd_graph
manual_forward
manual_backward
manual_update
head_type
head_is_kan
edge_param_count
edge_param_count_delta
```

StrictPass 定义为：

$$
\text{StrictPass}
=
[\text{non\_kan\_trainable\_param\_count}=0]
\land
[\text{uses\_torch\_loss\_backward}=0]
\land
[\text{uses\_torch\_autograd\_graph}=0]
\land
[\text{manual\_forward}=1]
\land
[\text{manual\_backward}=1]
\land
[\text{manual\_update}=1].
$$

任何使用 trainable linear readout、普通 MLP head、LayerNorm head、non-KAN classifier 的 candidate 可以作为 diagnostic oracle，但不能作为 strict route winner。

### 2.2 Gradient correctness gate

对每个新 manual layer、head、bridge、fused backward、kernel-native path，都必须做 gradient correctness check。

必须记录：

```text
grad_relerr_max
grad_relerr_mean
grad_cos_min
grad_cos_mean
finite_grad
forward_relerr_max
one_step_loss_before
one_step_loss_after
one_step_loss_delta
rollback_error
```

GradPass 定义：

$$
\text{GradPass}
=
[\text{grad\_relerr\_max}\leq10^{-4}]
\land
[\text{grad\_cos\_min}\geq0.999]
\land
[\text{finite\_grad}=1].
$$

OneStepPass 定义：

$$
\text{OneStepPass}
=
[\Delta L_{\text{train}}<0]
\land
[\text{finite\_loss}=1]
\land
[\text{rollback\_error}<10^{-8}].
$$

### 2.3 Efficiency gate

所有 efficiency ratio 必须使用同 shape、同 run、真实 measured MLP denominator，不能跨 run 用未审计 denominator。

定义：

$$
r_{mem}=\frac{\text{peak\_allocated\_MB}_{KAN}}{\text{peak\_allocated\_MB}_{MLP}}.
$$

$$
r_{step}=\frac{\text{step\_time\_ms}_{KAN}}{\text{step\_time\_ms}_{MLP}}.
$$

$$
r_{fwd}=\frac{\text{forward\_time\_ms}_{KAN}}{\text{forward\_time\_ms}_{MLP}}.
$$

$$
r_{bwd}=\frac{\text{backward\_time\_ms}_{KAN}}{\text{backward\_time\_ms}_{MLP}}.
$$

S2 diagnostic gate：

$$
r_{mem}\leq1.05,
$$

$$
r_{step}\leq1.50.
$$

S1 official gate：

$$
r_{mem}<1.00,
$$

$$
r_{step}\leq1.35.
$$

S0 strong gate：

$$
r_{mem}<1.00,
$$

$$
r_{step}\leq1.20.
$$

### 2.4 Basic Beyond-MLP task gate

Primary datasets：

```text
MNIST
Fashion-MNIST
KMNIST
```

Seeds：

```text
0,1,2
```

BasicBeyondPass：

$$
\forall d\in D_{primary}:
\quad
Acc_{KAN,d}\geq Acc_{MLP,d}-0.01.
$$

并且：

$$
\left|\{d:Acc_{KAN,d}\geq Acc_{MLP,d}\}\right|\geq2.
$$

### 2.5 Strong Beyond-MLP gate

StrongBeyondPass：

$$
\left|\{d:Acc_{KAN,d}\geq Acc_{MLP,d}+0.005\}\right|\geq2.
$$

并且：

$$
ECE_{KAN}\leq ECE_{MLP},
$$

$$
NLL_{KAN}\leq NLL_{MLP},
$$

$$
\operatorname{ValLossAUC}_{time,KAN}\leq\operatorname{ValLossAUC}_{time,MLP}.
$$

Validation loss AUC 使用梯形积分：

$$
\operatorname{AUC}_{time}^{val\_loss}
=
\sum_{i=1}^{n}
\frac{L_i+L_{i-1}}{2}
\cdot
(t_i-t_{i-1}).
$$

Step 维度 AUC：

$$
\operatorname{AUC}_{step}^{val\_loss}
=
\sum_{i=1}^{n}
\frac{L_i+L_{i-1}}{2}
\cdot
(s_i-s_{i-1}).
$$

---

## 3. v7.1 禁止事项

第一，不允许使用 fake data、proxy rows、固定占位 ratio 或手填结论。所有未实现内容必须写为：

```text
not_implemented
not_run
not_applicable
metric_unavailable
```

第二，不允许把 dense D3 targeted run 当 final success。D3 是 task-expression oracle，不是 official kernel-native S1/S0 candidate。

第三，不允许继续只围绕 grouped/T3 做 optimizer 大扫。已有 targeted probes 已说明，简单 optimizer、lr、schedule、weight decay、smoothing、input normalization、head 调整没有闭合 KMNIST gap。

第四，不允许把 non-KAN head、MLP stem、普通 dense head、LayerNorm head 当 mainline success。

第五，不允许只看 validation accuracy。每个 candidate 必须同时记录：

```text
memory ratio
step ratio
forward ratio
backward ratio
ValLossAUC by step
ValLossAUC by time
ECE
NLL
feature rank
margin
classwise accuracy
KMNIST hard pairs
```

第六，不允许用 test accuracy 调参。candidate selection 只能依据 train/val/holdout 和 diagnostic metrics；test accuracy 只在最终报告中展示。

第七，没有 S2 efficiency 时不允许 official task claim。task 好但 efficiency 不过，只能作为 expressivity diagnostic。

第八，没有 S1/S0 时不允许 official task re-entry。S2 只允许 diagnostic task。

第九，不允许把 teacher distillation 单独作为 final proof。distillation 可以用于诊断，但最终 claim 必须报告 teacher-free 或 supervised-only path。

第十，不允许跳过 efficiency attribution。任何 dense-to-kernel candidate 如果 efficiency 失败，必须拆清：

```text
forward materialization
backward live set
kernel count
update temp
activation / gate temp
cross-group bridge temp
```

第十一，不允许跳过 strict KAN head repair。如果 best dense candidate 依赖 trainable linear head，它必须先被标记为 oracle，不得进入 official route。

---

## 4. 核心假设

### H1：D3 的 accuracy 优势来自 dense cross-channel poly2_gate，而不是单纯 optimizer 或训练预算

v7.0 的 grouped/T3 optimizer、schedule、lr、head、input normalization probes 未闭合 KMNIST gap；dense manual KAN stack 才首次超过 MLP。H1 假设：D3 的优势主要来自 dense cross-channel function interaction 与 poly2_gate。

H1 成立证据包括：

```text
D3 feature_effective_rank > G2/T3 feature_effective_rank
D3 margin_p10 > G2/T3 margin_p10
D3 classwise KMNIST hard pairs 明显改善
D3 cross_group_correlation 比 grouped/T3 更强
D3 gate / poly2 ablation 导致 val acc 明显下降
```

H1 判断标准：

$$
\operatorname{rank}_{D3}\geq1.10\operatorname{rank}_{G2},
$$

or:

$$
\operatorname{margin}_{p10,D3}\geq1.10\operatorname{margin}_{p10,G2}.
$$

Hard-pair improvement：

$$
\Delta Acc_{\text{hard-pair}}\geq0.03.
$$

Gate-driven advantage：

$$
\operatorname{Acc}_{val,D3}
-
\operatorname{Acc}_{val,D3\text{-gate-disabled}}
\geq0.02.
$$

如果 H1 不成立，而 optimizer/regularization alone 解释了 D3 的优势，则 v7.1 应转向 optimizer/regularization；否则进入 dense-to-kernel bridge。

### H2：D3 的效率失败主要来自 dense forward/backward materialization，而不是 update overhead

FastOpt 已将 update mean 从 `0.9273 ms` 降到 `0.1171 ms`，但 D3-fastopt 仍然 step/memory fail。H2 假设剩余效率 blocker 是 dense forward/backward live set 和 materialization。

H2 成立要求至少一个 source 满足：

$$
\frac{M_{\text{source}}}{G_{\text{peak}}}\geq0.25,
$$

or:

$$
\frac{T_{\text{source}}}{T_{\text{step}}}\geq0.25,
$$

where source belongs to:

```text
dense_poly2_gate_activation
dense_gate_temp
dense_grad_gate
dense_grad_poly
dense_cross_channel_output
dense_elementwise_launches
```

H2 repair target：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

如果 update overhead 已解决但 forward/backward 仍重，则不能继续只优化 optimizer update。

### H3：low-rank / sparse cross-group bridge 可以恢复 D3 的主要 task gain，并保持 T3/G2 efficiency envelope

H3 假设：不需要 full dense D3 才能得到 task gain。edge-owned low-rank / sparse bridge 可以让 T3/G2 获得足够 cross-group interaction，同时保持 S2/S1。

候选 bridge：

```text
rank1 cross-group bridge
rank2 cross-group bridge
rank4 cross-group bridge
block-sparse group bridge
static group shuffle + rank1 bridge
g16/g8 hybrid one layer
late-phase cross-group residual
```

H3 成立标准：

$$
\Delta Acc_{\text{val}}\geq0.02
$$

relative to grouped G2/T3 baseline, and:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

H3 strong 标准：

$$
\operatorname{Acc}_{val,KAN}\geq\operatorname{Acc}_{val,MLP}-0.01
$$

on all primary datasets, and at least two datasets satisfy:

$$
\operatorname{Acc}_{val,KAN}\geq\operatorname{Acc}_{val,MLP}.
$$

如果 bridge improves accuracy but violates S2, it is diagnostic only.

### H4：strict KAN readout 可以替代 dense targeted run 中的 trainable linear head

v7.0 dense targeted run 的 D3 仍可能使用了 linear readout / head，因此 strict zero non-KAN trainable 条件没有作为 official gate 重新证明。H4 假设：

$$
Acc(\text{KAN stack + KAN readout})
\approx
Acc(\text{KAN stack + linear readout}).
$$

H4 成立标准：

```text
1. KAN readout 后 mean val acc 相对 linear head 下降 <= 0.005；
2. KMNIST val acc 下降 <= 0.010；
3. ECE/NLL 不明显恶化；
4. non_kan_trainable_param_count = 0；
5. gradient correctness pass。
```

如果 H4 不成立，则必须先设计 strict KAN classifier head，而不是直接 kernelize dense stack。

### H5：dense poly2_gate 可以被 kernel-native fused implementation 压到 S2，但需要 materialization-free forward/backward

H5 假设：D3 不一定必须慢。如果 dense poly2_gate 被写成 materialization-free fused kernel，避免显式 gate activation、gate gradient、poly temp 和 cross-channel temp，则 D3-derived candidate 可能进入 S2。

候选实现：

```text
fused dense poly2_gate forward
fused dense poly2_gate backward
single-kernel gate + mix
streaming grad_gate accumulation
materialization-free gate residual
tile-stream dense gate
low-rank factorized dense gate
```

H5 useful 标准：

$$
\frac{M_{\text{kernelized-D3}}}{M_{\text{D3-fastopt}}}\leq0.90,
$$

and:

$$
\frac{T_{\text{kernelized-D3}}}{T_{\text{D3-fastopt}}}\leq0.90.
$$

H5 S2 标准：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

H5 S1 标准：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

### H6：D3 的 Beyond-MLP signal 必须补 ValLossAUC/time AUC

D3 targeted run 证明了 accuracy/ECE/NLL，但没有测 ValLossAUC/time AUC。H6 假设：D3 的优势可能来自更慢路径或更长训练，因此必须补 AUC。

H6 成立要求 D3 或 D3-derived candidate 满足：

$$
\operatorname{ValLossAUC}_{time,KAN}\leq\operatorname{ValLossAUC}_{time,MLP},
$$

or at least:

$$
\operatorname{ValLossAUC}_{step,KAN}\leq\operatorname{ValLossAUC}_{step,MLP}.
$$

如果 D3 accuracy high but time AUC much worse, it remains expressivity proof rather than efficient Beyond-MLP proof.

### H7：Beyond-MLP improvement 不是 validation 偶然性，而是 seed/dataset 稳定优势

H7 假设 dense/structured KAN 的 improvement 在 primary datasets 和 seeds 上稳定，不是某个 split 的偶然。

H7 成立标准：

```text
1. 3 datasets x 3 seeds basic gate pass；
2. 至少 2 datasets 的 dataset-mean gap >= +0.005；
3. seed std 不显著大于 MLP；
4. test acc mean 同方向改善，不能出现 val win 但 test 全面 collapse；
5. classwise accuracy 没有由少数 class 的严重退化换来总 acc 提升。
```

### H8：如果 D3-derived efficient bridge 仍无法闭合 task gap，则 grouped/T3 需要新 primitive，而不是继续 optimizer

H8 是 Stop/Go 假设。如果完成：

```text
D3 advantage attribution
strict KAN head repair
dense efficiency attribution
low-rank/sparse bridge
fused dense-gate kernelization
optimizer matched control
```

仍然没有 candidate 满足 S2 + task improved，则当前 grouped/T3 family 的 expressivity bridge 不够。下一轮应做新的 function-space primitive，而不是继续调 optimizer。

---

## 5. Candidate 设计

v7.1 candidate 不是无序组合，而是从 dense D3 oracle 到 kernel-native strict candidate 的迁移链。

### 5.1 Baselines

| id | candidate | 用途 |
|---|---|---|
| `B0` | `MLP-autograd-fulltrain-240` | task denominator |
| `B1` | `T3-launch-fused-step+ManualAdamW` | efficient grouped baseline |
| `B2` | `G2-O7-ManualAdamW-lr-high-fulltrain-240` | official best grouped baseline |
| `B3` | `D3-dense-poly2-gate-d3-linear-head` | dense oracle accuracy upper bound |
| `B4` | `D3-fastopt-dense-poly2-gate-d3-linear-head` | dense oracle with update overhead reduced |

`B3/B4` 只能是 oracle，不允许作为 final strict success。

### 5.2 Strict KAN head candidates

目标是验证 H4，去掉 dense targeted run 中可能存在的 trainable linear head。

| id | candidate | 设计 |
|---|---|---|
| `H0` | `KAN-head-poly2-gate-classwise` | 每个 class 一个 KAN readout edge group |
| `H1` | `KAN-head-poly2-silu-classwise` | 更稳定的 classwise quadratic readout |
| `H2` | `KAN-head-rbf-poly-exp-classwise` | RBF-like strict head |
| `H3` | `KAN-head-shared-basis-classwise` | 共享 basis，class-specific coefficient |
| `H4` | `No-head-energy-score` | 以 class prototypes / KAN distance score 替代 linear classifier |

每个 head 必须记录：

```text
head_param_count
head_is_kan
head_grad_relerr_max
head_grad_cos_min
head_forward_time_ms
head_backward_time_ms
val_acc_delta_vs_linear_head
ECE_delta_vs_linear_head
NLL_delta_vs_linear_head
```

如果所有 strict head 都比 linear head 低超过 `0.02` mean val acc，则 v7.1 不进入 final route，应先做 head 架构。

### 5.3 Structured poly2_gate candidates

目标是验证 H3：是否能保留 dense D3 表达力而不保留 dense live set。

| family | id | candidate | 设计意图 |
|---|---|---|---|
| grouped local | `K1-g8` | `GroupedPoly2Gate-g8` | 最接近 T3 grouped route，低 live set |
| grouped local | `K1-g16` | `GroupedPoly2Gate-g16` | group size 与 v6.20 T3 对齐 |
| grouped local | `K1-g32` | `GroupedPoly2Gate-g32` | 更接近 dense，但仍可 block/fuse |
| shuffle | `K2-s2` | `GroupedPoly2Gate+FixedShuffle2` | 固定 permutation，无新增 trainable 非 KAN 参数 |
| shuffle | `K2-s4` | `GroupedPoly2Gate+FixedShuffle4` | 多层固定 shuffle 增强跨组传播 |
| low rank KAN-owned | `K3-r1` | `GroupedPoly2Gate+KANLowRank-r1` | 低秩跨组 correction，但参数必须 KAN-owned |
| low rank KAN-owned | `K3-r2` | `GroupedPoly2Gate+KANLowRank-r2` | 表达力与效率折中 |
| low rank KAN-owned | `K3-r4` | `GroupedPoly2Gate+KANLowRank-r4` | 接近 dense oracle |
| shared basis | `K4-shared` | `GroupedPoly2Gate+SharedBasis` | 降低参数和 optimizer state |
| streamed | `K5-stream` | `StreamedGroupedPoly2Gate` | 不 materialize full basis/gate temp |
| fused | `K6-fused` | `FusedGroupedPoly2GateForwardBackward` | kernel-native final target |

所有 trainable cross-group 参数必须归类为 KAN function parameters；若实现为普通 dense linear mixer，则 candidate 只能作为 diagnostic，不允许 StrictPass。

### 5.4 Live-set repair candidates

目标是验证 H2/H5，并把 K5/K6/F candidates 推向 S2/S1。

| id | candidate | 预期作用 |
|---|---|---|
| `M1` | `no-full-basis-materialization` | basis 按 chunk 生成并立即消费 |
| `M2` | `gate-inplace-reuse` | gate sigmoid workspace 复用 |
| `M3` | `hidden-cache-minimal` | 只保存 backward 必需 cache |
| `M4` | `manual-checkpoint-local-only` | 可选局部重算，必须记录 recompute cost |
| `M5` | `foreach-fastopt` | 保留 v7.0 fastopt 成果 |
| `M6` | `optimizer-state-trim` | 对共享/低秩参数减少 state |
| `M7` | `fused-update-no-cpu-diagnostics` | 去掉训练期 CPU norm 诊断 |
| `M8` | `gate-temp-streaming` | gate temp 流式消费，不保留完整 gate activation |
| `M9` | `grad-gate-streaming` | streaming grad_gate accumulation |
| `M10` | `cross-channel-output-no-materialize` | cross-channel output 直接写入下一阶段 buffer |

每个 memory repair 必须同时记录：

```text
memory_delta_vs_parent
step_delta_vs_parent
grad_relerr_delta_vs_parent
task_delta_vs_parent
```

若 memory 降低来自牺牲 correctness 或 task，则不能算 repair 成功。

---

## 6. 实验阶段总览

v7.1 分为十八个阶段：

```text
P0: Runner / provenance / contract lock
P1: Dense oracle source-of-gain audit
P2: Strict KAN head repair
P3: Structured candidate task transfer
P4: Gradient correctness and one-step lock
P5: Efficiency profiler and live-set attribution
P6: D3-kernelization microkernel smoke
P7: Low-rank / sparse cross-group bridge under T3 envelope
P8: Fused dense-gate / materialization-free D3 package
P9: D3-to-T3 distillation and initialization diagnostic
P10: Optimizer / regularization matched control
P11: 3-dataset diagnostic task gate
P12: Selected-candidate full-step efficiency profiler
P13: Longer-budget and ValLossAUC/time AUC verification
P14: Sample efficiency / robustness / calibration diagnostic
P15: Patch/token scaling smoke
P16: Candidate selection by efficiency + task + beyond score
P17: Official task re-entry if S1/S0
P18: Route decision and artifact audit
```

P0-P2 是 purity 和 source-of-gain 阶段。  
P3-P8 是 dense-to-kernel architecture 阶段。  
P9-P10 是 training recipe diagnostic 阶段。  
P11-P15 是 task / efficiency / scaling evidence 阶段。  
P16-P18 是 route selection 阶段。

---

## 7. P0：Runner / provenance / contract lock

### 7.1 目的

建立 v7.1 的 no-fake / no-proxy 实验入口，避免 targeted probe、oracle run、official gate 混淆。

建议新增或复用 runner：

```text
experiments/run_gafu_v71_real.py
```

建议命令：

```bash
python experiments/run_gafu_v71_real.py \
  --packages V7_1_ALL \
  --out-dir results/real_rerun_20260505/v71_poly2gate_kernel_all_<timestamp> \
  --fresh \
  --device auto \
  --wandb-project DG-KAN \
  --wandb-group v71-poly2gate-kernel-20260505 \
  --wandb-name-prefix v71-poly2gate-kernel
```

### 7.2 必须落盘 artifact

```text
run_manifest.json
provenance_audit.csv
candidate_registry.csv
gate_config.json
p0_contract.csv
p0_reproduction_check.csv
```

`candidate_registry.csv` 必须记录：

```text
candidate_id
candidate_name
family
oracle_or_mainline
strict_status
head_type
implementation_status
expected_stage
allowed_in_route_selection
```

### 7.3 P0 必须记录字段

```text
variant
family
status
fake_data_used
proxy_row_used
uses_loss_backward
uses_torch_autograd_graph
manual_forward_available
manual_backward_available
manual_update_available
uses_triton_kernel
uses_foreach_update
nonKAN_param_count
edge_param_count
edge_param_count_delta
head_type
head_is_kan
grad_relerr_max
grad_cos_min
memory_ratio_mean
step_ratio_mean
backward_ratio_mean
forward_ratio_mean
update_ratio_mean
val_acc_mean
test_acc_mean
val_ECE_mean
val_NLL_mean
v70_reference_metric
v71_reproduction_metric
reproduction_delta
```

### 7.4 P0 判断标准

P0 provenance pass：

```text
allow_fake_data=False hard-coded
fake/proxy audit nonzero count = 0
all candidate status in {measured, not_implemented, gated_not_run}
no measured row may contain placeholder ratios
```

P0 contract pass：

```text
fake_data_used = 0
proxy_row_used = 0
uses_loss_backward = 0 for mainline candidates
manual_backward_available = 1
grad_relerr < 1e-4
grad_cos > 0.999
```

D3 task reproduction pass：

$$
|\operatorname{Acc}_{val,D3,v71}-0.8670|\leq0.015.
$$

G2 official reproduction pass：

$$
|\operatorname{Acc}_{val,G2,v71}-0.8138|\leq0.015.
$$

D3-fastopt efficiency reproduction pass：

$$
|r_{\text{step,D3-fastopt,v71}}-1.6810|\leq0.10,
$$

$$
|r_{\text{mem,D3-fastopt,v71}}-1.1778|\leq0.05.
$$

### 7.5 可视化

```text
p0_contract_heatmap.svg
p0_reproduction_accuracy_bar.svg
p0_reproduction_efficiency_bar.svg
p0_family_position_pareto.svg
p0_candidate_registry_table.md
```

---

## 8. P1：Dense oracle source-of-gain audit

### 8.1 目的

验证 H1，确定 D3 的成功到底来自 `poly2_gate`、dense cross-channel interaction、depth、warmup/smoothing、head，还是 optimizer recipe。

### 8.2 实验矩阵

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

steps:
  240

train_mode:
  fulltrain_cycle

hidden:
  64

depth_dense:
  2,3

basis:
  8
```

### 8.3 候选

```text
D0-MLP-reference
D1-dense-poly2-silu-base-d2
D2-dense-poly2-silu-base-d3
D3-dense-poly2-gate-d3
D4-dense-poly3-d2
D5-dense-poly2-gate-d2
D6-dense-poly2-gate-d3-no-label-smoothing
D7-dense-poly2-gate-d3-no-gate-learnable
D8-dense-poly2-gate-d3-gate-frozen
D9-dense-poly2-gate-d3-input-layer-only-gate
D10-dense-poly2-gate-d3-hidden-layer-only-gate
D11-dense-poly2-gate-d3-output-layer-only-gate
D12-dense-poly2-gate-d3-no-warmup
D13-dense-poly2-gate-d3-no-smooth
```

### 8.4 必须记录字段

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
val_ECE
test_ECE
val_NLL
test_NLL
val_gap_vs_MLP
test_gap_vs_MLP
margin_p10
margin_p50
feature_effective_rank
confidence_mean
wrong_confidence_mean
param_count
head_type
head_param_count
gate_param_count
poly_param_count
wall_clock_time_sec
trace_every_20_steps
```

### 8.5 Ablation 字段

```text
gate_enabled
gate_disabled_val_acc
gate_ablation_delta_acc
poly2_enabled
poly2_disabled_val_acc
poly2_ablation_delta_acc
cross_channel_enabled
cross_channel_disabled_val_acc
cross_channel_ablation_delta_acc
depth_ablation_delta_acc
warmup_smooth_ablation_delta_acc
```

### 8.6 判断标准

H1 成立：

```text
D3 / D5 / D6 系列稳定优于 no-gate / frozen-gate；
D3 poly2_gate 在 3 datasets mean 上保持 best 或 near-best；
去 gate ablation mean val acc 下降 >= 0.005；
KMNIST 去 gate 或替换 gate 后 val acc 下降 >= 0.010；
D3 ECE/NLL 不劣于去 gate ablation。
```

Gate-driven advantage：

$$
\operatorname{Acc}_{val,D3}
-
\operatorname{Acc}_{val,D3\text{-gate-disabled}}
\geq0.02.
$$

Representation-driven advantage：

$$
\operatorname{rank}_{D3}\geq1.10\operatorname{rank}_{G2}.
$$

or:

$$
\operatorname{margin}_{p10,D3}\geq1.10\operatorname{margin}_{p10,G2}.
$$

Hard-pair improvement：

$$
\Delta Acc_{\text{hard-pair}}\geq0.03.
$$

### 8.7 可视化

```text
p1_val_gap_by_candidate.svg
p1_gate_ablation_delta_by_dataset.svg
p1_ece_nll_scatter.svg
p1_learning_curves_val_loss.svg
p1_feature_rank_comparison.svg
p1_margin_distribution.svg
p1_kmnist_classwise_acc_heatmap.svg
p1_advantage_taxonomy_dashboard.svg
```

---

## 9. P2：Strict KAN head repair

### 9.1 目的

验证 H4，让 dense/structured KAN 不依赖 trainable linear readout。若 strict head 不能替代 linear head，后续 kernelization 即使成功也不能称为 full PureKAN-NG。

### 9.2 实验矩阵

```text
base stack:
  D3 dense poly2_gate d3

heads:
  linear reference
  KAN poly2_gate classwise
  KAN poly2_silu classwise
  KAN rbf/poly-exp classwise
  shared-basis KAN classwise
  energy/prototype score

datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

steps:
  240
```

### 9.3 候选

```text
H0-linear-head-reference
H1-KAN-head-poly2-gate-classwise
H2-KAN-head-poly2-silu-classwise
H3-KAN-head-rbf-poly-exp-classwise
H4-KAN-head-shared-basis-classwise
H5-no-head-energy-score
```

### 9.4 必须记录字段

```text
head_type
head_param_count
head_is_kan
non_kan_trainable_param_count
kan_trainable_param_count
head_grad_relerr_max
head_grad_cos_min
head_forward_time_ms
head_backward_time_ms
head_update_time_ms
val_acc
test_acc
val_acc_delta_vs_linear_head
ECE
NLL
ECE_delta_vs_linear_head
NLL_delta_vs_linear_head
KMNIST_val_acc_delta_vs_linear_head
```

### 9.5 判断标准

H4 pass：

```text
strict head mean val drop <= 0.005
strict head KMNIST val drop <= 0.010
ECE/NLL 不明显恶化
non_kan_trainable_param_count = 0
gradient correctness pass
```

如果所有 strict head 都比 linear head 低超过 `0.02` mean val acc，则 v7.1 不进入 final route，应先做 head 架构。

### 9.6 可视化

```text
p2_head_val_delta.svg
p2_head_ece_delta.svg
p2_head_nll_delta.svg
p2_purity_param_breakdown.svg
p2_head_efficiency_pareto.svg
```

---

## 10. P3：Structured candidate task transfer

### 10.1 目的

验证 H3 的 task 部分：structured / kernel-friendly candidate 是否保留 dense D3 的 accuracy。P3 先用 KMNIST targeted 筛选，再进入 3 datasets x 3 seeds。

### 10.2 第一层 targeted screening

```text
dataset:
  KMNIST

seed:
  0

steps:
  240

candidate families:
  K1, K2, K3, K4, K5, K6
```

Targeted 晋级标准：

$$
Acc_{\text{candidate,KMNIST,seed0}}
\geq
Acc_{\text{MLP,KMNIST,seed0}}-0.01.
$$

并且：

$$
Acc_{\text{candidate,KMNIST,seed0}}
\geq
Acc_{\text{G2,KMNIST,seed0}}+0.03.
$$

### 10.3 第二层 primary gate

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

steps:
  240
```

只让 targeted top candidates 进入。

### 10.4 Candidate families

```text
K1-g8 / K1-g16 / K1-g32
K2-fixed-shuffle2 / K2-fixed-shuffle4
K3-KANLowRank-r1 / r2 / r4
K4-shared-basis
K5-streamed-grouped-poly2-gate
K6-fused-grouped-poly2-gate-forward-backward
```

### 10.5 必须记录字段

```text
candidate_family
candidate_id
group_size
shuffle_count
cross_rank
shared_basis
streaming_mode
fused_mode
strict_head_type
val_acc
test_acc
train_acc
dataset_gap_vs_mlp
dataset_gap_vs_dense_D3
seed_std
ECE
NLL
feature_rank
margin_p10
classwise_acc
confusion_matrix
nonKAN_param_count
edge_param_count_delta
```

### 10.6 判断标准

Primary gate 晋级标准：

```text
BasicBeyondPass = 1
GradPass = 1
StrictPass = 1 or strict-repair-pending with clear reason
```

Structured candidate useful：

$$
\Delta Acc_{\text{val}}\geq0.02
$$

relative to G2/T3 baseline.

Bridge strong：

$$
\operatorname{Acc}_{val,KAN}\geq\operatorname{Acc}_{val,MLP}-0.01
$$

on all primary datasets.

### 10.7 可视化

```text
p3_task_pareto_val_gap_vs_param_count.svg
p3_dataset_gap_heatmap.svg
p3_seed_boxplot.svg
p3_classwise_delta_heatmap.svg
p3_kmnist_targeted_screening_bar.svg
```

---

## 11. P4：Gradient correctness and one-step lock

### 11.1 目的

确保所有新结构不是“看起来准但梯度错”。每个 P3 晋级 candidate 必须跑 gradient correctness 和 one-step probe。

### 11.2 必跑检查

```text
small gradient check:
  batch 8/16

task-shape gradient check:
  batch 128

one-step probe:
  real train batch

finite value audit:
  forward / loss / grad / update
```

### 11.3 必须记录字段

```text
candidate
dataset
batch_size
grad_relerr_max
grad_relerr_mean
grad_cos_min
grad_cos_mean
forward_relerr_vs_reference
loss_before
loss_after
loss_delta
nan_count
inf_count
rollback_error
bad_step_flag
update_norm
update_over_param
grad_norm
gate_ablation_delta_loss
poly2_ablation_delta_loss
crossgroup_ablation_delta_loss
```

### 11.4 判断标准

GradPass：

$$
\operatorname{grad\_relerr\_max}<10^{-4}.
$$

$$
\operatorname{grad\_cos\_min}>0.999.
$$

OneStepPass：

$$
\Delta L_{\text{train}}<0.
$$

Rollback pass：

$$
\operatorname{rollback\_error}<10^{-8}.
$$

Ablation contribution：

$$
|\Delta L_{\text{gate/poly2/crossgroup ablation}}|>10^{-4}.
$$

Gradient fail candidate 不允许进入 route best，只能保留在 failure table。

### 11.5 可视化

```text
p4_grad_relerr_by_candidate.svg
p4_grad_cos_by_candidate.svg
p4_one_step_loss_delta.svg
p4_ablation_delta_bar.svg
p4_rollback_error_bar.svg
```

---

## 12. P5：Efficiency profiler and live-set attribution

### 12.1 目的

验证 H2/H3 的 efficiency 部分。P5 对 P3/P4 晋级 candidate 做 full-step profiler 和 live-set attribution。

### 12.2 Profiler shapes

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

batch_sizes:
  128
  256
  512

depths:
  candidate native depth

hidden:
  64

warmup/reps:
  50/200
```

### 12.3 必须记录字段

```text
candidate
dataset
batch_size
depth
forward_time_ms
backward_time_ms
update_time_ms
step_time_ms
peak_allocated_MB
peak_reserved_MB
MLP_peak_allocated_MB
memory_ratio
step_ratio
forward_ratio
backward_ratio
update_ratio
optimizer_state_memory_MB
cache_total_MB
cache_hidden_MB
cache_basis_MB
cache_gate_MB
workspace_temp_MB
largest_temp_tensor_MB
kernel_count_forward
kernel_count_backward
kernel_count_update
num_tensor_allocations
materialized_tensor_count
top1_memory_source
top2_memory_source
top3_memory_source
unknown_memory_fraction
```

### 12.4 判断标准

P5 attribution pass：

```text
top3 memory sources identified
unknown_memory_fraction <= 0.10
component timing explains >= 90% step time
```

S2Pass：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

S1Pass：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

If task pass but efficiency fail, candidate remains expressivity diagnostic.

### 12.5 可视化

```text
p5_efficiency_pareto.svg
p5_live_set_waterfall.svg
p5_temp_tensor_stacked_bar.svg
p5_batch_size_scaling_curve.svg
p5_step_time_waterfall.svg
p5_memory_source_heatmap.svg
```

---

## 13. P6：D3-kernelization microkernel smoke

### 13.1 目的

将 D3 拆成 microkernel，判断哪些部分值得 kernelize。P6 不要求 task pass，只验证局部 kernel 的 memory/time/gradient。

### 13.2 Microkernels

```text
K0-dense-poly2-gate-forward-current
K1-dense-poly2-gate-forward-fused
K2-dense-poly2-gate-backward-current
K3-dense-poly2-gate-backward-fused
K4-dense-grad-gate-streaming
K5-dense-grad-poly-streaming
K6-dense-cross-channel-output-no-materialize
K7-dense-gate+mix-single-kernel
K8-factorized-gate-r4
K9-factorized-gate-r8
K10-factorized-gate-r16
```

### 13.3 必须记录字段

```text
microkernel
implementation_status
input_shape
output_shape
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

### 13.4 判断标准

Microkernel useful if:

$$
\frac{T_{\text{candidate}}}{T_{\text{current}}}\leq0.80
$$

or:

$$
\frac{M_{\text{candidate}}}{M_{\text{current}}}\leq0.80.
$$

Microkernel enters P8 if:

```text
grad_relerr < 1e-4
grad_cos > 0.999
time_ratio_vs_current <= 0.90
memory_ratio_vs_current <= 0.95
```

### 13.5 可视化

```text
p6_microkernel_memory_time_pareto.svg
p6_microkernel_grad_correctness.svg
p6_materialization_reduction_bar.svg
p6_kernel_count_reduction_bar.svg
```

---

## 14. P7：Low-rank / sparse cross-group bridge under T3 envelope

### 14.1 目的

把 D3 的 cross-channel 表达力添加到 T3/G2 上，同时保持 T3 的 S2/S1 efficiency envelope。

### 14.2 Candidate packages

```text
B0-T3-baseline
B1-static-group-shuffle
B2-crossgroup-r1-edge-owned
B3-crossgroup-r2-edge-owned
B4-crossgroup-r4-edge-owned
B5-block-sparse-group-bridge
B6-g16-g8-hybrid-one-layer
B7-late-phase-crossgroup-residual
B8-groupwise-temperature-edge-owned
B9-static-shuffle+rank1
B10-static-shuffle+rank2
```

### 14.3 必须记录字段

```text
candidate
implementation_status
bridge_type
rank
sparsity
group_count
nonKAN_param_count
edge_param_count_delta
manual_backward_available
memory_ratio_mean
step_ratio_mean
forward_ratio_mean
backward_ratio_mean
memory_overhead_vs_T3
step_overhead_vs_T3
grad_relerr_max
grad_cos_min
val_acc_mean
test_acc_mean
val_gap_vs_MLP
ECE
NLL
feature_rank
margin_p10
cross_group_correlation
KMNIST_val_acc
KMNIST_classwise_acc
hard_class_pair_improvement
```

### 14.4 判断标准

Efficiency envelope：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

Official envelope：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

Bridge useful：

$$
\Delta Acc_{\text{val}}\geq0.02
$$

relative to T3/G2 baseline, while:

$$
\Delta r_{\text{step}}\leq0.05.
$$

Bridge strong：

$$
\operatorname{Acc}_{val,KAN}\geq\operatorname{Acc}_{val,MLP}-0.01
$$

on all primary datasets.

### 14.5 可视化

```text
p7_bridge_efficiency_accuracy_pareto.svg
p7_bridge_rank_vs_accuracy.svg
p7_bridge_overhead_bar.svg
p7_kmnist_hard_pair_improvement.svg
p7_cross_group_correlation_delta.svg
```

---

## 15. P8：Fused dense-gate / materialization-free D3 package

### 15.1 目的

P8 是 dense-to-kernel 的主实验。它测试 kernel-native D3-derived full-step packages。

### 15.2 Candidate packages

```text
F0-D3-fastopt-current
F1-D3-fused-forward
F2-D3-fused-backward
F3-D3-fused-forward-backward
F4-D3-materialization-free-gate
F5-D3-streaming-grad-gate
F6-D3-factorized-gate-r4
F7-D3-factorized-gate-r8
F8-D3-factorized-gate-r16
F9-D3-tiled-dense-gate
F10-D3-fused-gate+mix-single-kernel
F11-D3-S2-combo
```

### 15.3 必须记录字段

```text
package
implementation_status
backend
factor_rank
tile_size
materializes_gate_activation
materializes_grad_gate
memory_ratio_mean
step_ratio_mean
forward_ratio_mean
backward_ratio_mean
update_ratio_mean
memory_improvement_vs_D3_fastopt
step_improvement_vs_D3_fastopt
peak_allocated_MB
peak_reserved_MB
gate_temp_MB
grad_gate_MB
poly_temp_MB
cross_channel_temp_MB
kernel_count_total
torch_op_count
triton_kernel_count
grad_relerr_max
grad_cos_min
val_acc_mean
test_acc_mean
ECE
NLL
```

### 15.4 判断标准

Kernelization useful：

$$
\frac{M_{\text{package}}}{M_{\text{D3-fastopt}}}\leq0.90,
$$

or:

$$
\frac{T_{\text{package}}}{T_{\text{D3-fastopt}}}\leq0.90.
$$

S2：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

S1：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

Task preservation：

$$
\operatorname{Acc}_{val,package}\geq\operatorname{Acc}_{val,D3}-0.01.
$$

or if package trades some D3 accuracy for efficiency:

$$
\operatorname{Acc}_{val,package}\geq\operatorname{Acc}_{val,MLP}-0.01.
$$

### 15.5 可视化

```text
p8_d3_kernelization_pareto.svg
p8_materialization_free_memory_bar.svg
p8_d3_accuracy_preservation_bar.svg
p8_s2_s1_boundary_plot.svg
p8_grad_correctness_lollipop.svg
```

---

## 16. P9：D3-to-T3 distillation and initialization diagnostic

### 16.1 目的

判断 D3 的 function-space 是否能传给 efficient T3/bridge candidate。它回答：T3/G2 是表达力不够，还是训练没有找到 D3-like solution。

### 16.2 Candidate recipes

```text
D0-T3-supervised-only
D1-T3-logit-distill-from-D3
D2-T3-feature-distill-from-D3
D3-T3+bridge-r1-logit-distill
D4-T3+bridge-r2-logit-distill
D5-T3+D3-initialization-projection
D6-T3+bridge-r2-supervised-only
D7-T3+bridge-r2-distill-then-supervised
```

### 16.3 必须记录字段

```text
recipe
teacher
student
uses_teacher_logits
uses_teacher_features
distill_temperature
distill_weight
supervised_weight
memory_ratio_mean
step_ratio_mean
val_acc_mean
test_acc_mean
val_gap_vs_MLP
val_gap_vs_D3
ECE
NLL
feature_rank
margin_p10
student_teacher_logit_cos
student_teacher_feature_cka
hard_class_pair_accuracy
```

### 16.4 判断标准

Distillation useful：

$$
\Delta Acc_{\text{val}}\geq0.02
$$

relative to same student supervised-only.

Efficiency must remain：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

Distillation cannot be final if same student does not also have a teacher-free or supervised-only path within $1\%$ of MLP. If it only works with D3 teacher, route must mark:

```text
distillation_diagnostic_only
```

### 16.5 可视化

```text
p9_distillation_accuracy_bar.svg
p9_student_teacher_similarity.svg
p9_distill_efficiency_pareto.svg
p9_hard_pair_recovery.svg
```

---

## 17. P10：Optimizer / regularization matched control

### 17.1 目的

控制训练 recipe 的影响，避免把 architecture gain 误判为 optimizer gain。P10 不做大扫，只做 matched control。

### 17.2 Candidate recipes

```text
O0-ManualAdamW-baseline
O1-ManualAdamW-lr-high
O2-ManualAdamW-lr-low
O3-ManualAdamW-warmup-smooth
O4-ManualAdamW-weightdecay-low
O5-ManualAdamW-weightdecay-high
O6-ManualAdamW-gradclip
O7-ManualAdamW-label-smoothing
O8-ManualAdamW-edge-noise
O9-ManualAdanLite-best
```

All recipes must be run on：

```text
MLP baseline
G2/T3 baseline
D3 dense
best bridge candidate
best kernelized-D3 candidate
```

### 17.3 必须记录字段

```text
recipe
model
lr
weight_decay
grad_clip
warmup
label_smoothing
edge_noise
val_acc_mean
test_acc_mean
ECE
NLL
ValLossAUC_step
ValLossAUC_time
train_val_gap
update_norm_mean
cos_update_grad
bad_step_rate
memory_ratio_mean
step_ratio_mean
```

### 17.4 判断标准

Optimizer-only explains D3 if the same recipe gives G2/T3：

$$
\operatorname{Acc}_{val,G2/T3}\geq\operatorname{Acc}_{val,MLP}-0.01.
$$

Optimizer useful if：

$$
\Delta Acc_{\text{val}}\geq0.02
$$

without leaving S2.

If optimizer helps D3 but not grouped/T3/bridge, architecture remains the main blocker.

### 17.5 可视化

```text
p10_recipe_model_interaction_heatmap.svg
p10_optimizer_val_acc_bar.svg
p10_val_loss_auc_time_bar.svg
p10_train_val_gap_bar.svg
p10_accuracy_vs_efficiency_by_recipe.svg
```

---

## 18. P11：3-dataset diagnostic task gate

### 18.1 目的

P11 是 v7.1 主 task gate，比较 selected candidates on 3 datasets x 3 seeds。P11 不允许只用 KMNIST seed0 或 targeted single probe 做结论。

### 18.2 必跑方法

```text
MLP-autograd-fulltrain-240
G2-O7-ManualAdamW-lr-high-fulltrain-240
D3-dense-poly2-gate-d3-warmup-smooth-240
best-strict-head-candidate
best-structured-candidate
best-bridge-candidate
best-kernelized-D3-candidate
best-distilled-candidate, if allowed
```

### 18.3 设置

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

### 18.4 必须记录字段

```text
dataset
seed
candidate
train_acc_curve
val_acc_curve
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
time_to_val_acc_80pct_MLP
time_to_val_acc_MLP_minus_1pct
step_time_ms
memory_ratio_mean
samples_per_second
feature_rank
margin_p10
classwise_acc
KMNIST_hard_pair_acc
seedwise_failure_reason
```

### 18.5 判断标准

Basic Beyond task gate：

$$
\operatorname{Acc}_{val,KAN}\geq\operatorname{Acc}_{val,MLP}-0.01
$$

on all primary datasets, and at least two datasets satisfy：

$$
\operatorname{Acc}_{val,KAN}\geq\operatorname{Acc}_{val,MLP}.
$$

Strong gate diagnostic：

$$
\operatorname{Acc}_{val,KAN}\geq\operatorname{Acc}_{val,MLP}+0.005
$$

on at least two datasets, and：

$$
\operatorname{ECE}_{KAN}\leq\operatorname{ECE}_{MLP},
$$

$$
\operatorname{NLL}_{KAN}\leq\operatorname{NLL}_{MLP}.
$$

Time gate：

$$
\operatorname{ValLossAUC}_{time,KAN}\leq\operatorname{ValLossAUC}_{time,MLP}.
$$

### 18.6 可视化

```text
p11_val_acc_by_dataset_seed.svg
p11_val_loss_vs_step.svg
p11_val_loss_vs_time.svg
p11_accuracy_vs_time.svg
p11_ece_nll_bar.svg
p11_kmnist_classwise_heatmap.svg
p11_task_efficiency_pareto.svg
```

---

## 19. P12：Selected-candidate full-step efficiency profiler

### 19.1 目的

对 P11 selected candidates 做 final efficiency profiler，避免 task 好但 efficiency 不过。

### 19.2 必跑对象

```text
MLP-autograd-reference
G2/T3 baseline
D3-fastopt
best-strict-head-candidate
best-structured-candidate
best-bridge-candidate
best-kernelized-D3-candidate
best-task-candidate-from-P11
```

### 19.3 Shape grid

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
  2
  3
  4

warmup / measure:
  50 / 200
```

### 19.4 必须记录字段

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

### 19.5 判断标准

S2：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

S1：

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

### 19.6 可视化

```text
p12_efficiency_pareto_selected.svg
p12_batch_depth_memory_heatmap.svg
p12_batch_depth_step_heatmap.svg
p12_memory_source_waterfall_selected.svg
p12_s1_s2_boundary_plot.svg
```

---

## 20. P13：Longer-budget and ValLossAUC/time AUC verification

### 20.1 目的

D3 targeted run steps=240，但 strong gate 缺 ValLossAUC/time AUC。P13 补这个缺口，并判断 candidate 是真正 wall-clock 有效，还是只是慢而准。

### 20.2 必跑对象

```text
MLP-autograd-fulltrain
D3-dense-poly2-gate
best-structured-candidate
best-bridge-candidate
best-kernelized-D3-candidate
```

### 20.3 设置

```text
steps:
  480

logging:
  every 20 steps

seeds:
  0,1,2

datasets:
  MNIST
  Fashion-MNIST
  KMNIST
```

### 20.4 必须记录

```text
val_acc_at_120
val_acc_at_240
val_acc_at_480
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

### 20.5 判断标准

Long budget pass：

$$
\operatorname{Acc}_{val,480}\geq\operatorname{Acc}_{MLP,480}-0.01.
$$

Time AUC pass：

$$
\operatorname{ValLossAUC}_{time,KAN}\leq\operatorname{ValLossAUC}_{time,MLP}.
$$

If accuracy improves but time AUC fails badly, candidate remains expressivity proof, not efficient Beyond-MLP proof.

### 20.6 可视化

```text
p13_long_budget_val_acc_curve.svg
p13_val_loss_auc_time_bar.svg
p13_late_slope_bar.svg
p13_overfit_index_bar.svg
```

---

## 21. P14：Sample efficiency / robustness / calibration diagnostic

### 21.1 目的

Beyond-MLP 不应只看 full-data accuracy。P14 测 sample efficiency、noise robustness 和 calibration。

### 21.2 Data fractions

```text
5%
10%
25%
50%
100%
```

### 21.3 Noise settings

```text
label noise:
  5%, 10%, 20%

input noise:
  gaussian sigma 0.05
  gaussian sigma 0.10
  random erasing small
```

### 21.4 必跑对象

```text
MLP-autograd-reference
D3-dense-poly2-gate
best-structured-candidate
best-bridge-candidate
best-kernelized-D3-candidate
```

### 21.5 必须记录

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

### 21.6 判断标准

Sample efficiency useful：

$$
\operatorname{AUC}_{data,KAN}>\operatorname{AUC}_{data,MLP}.
$$

Robustness useful：

$$
\operatorname{AccDrop}_{KAN}<\operatorname{AccDrop}_{MLP}
$$

for at least two noise settings.

Calibration useful：

$$
\operatorname{ECE}_{KAN}<\operatorname{ECE}_{MLP}-0.01.
$$

### 21.7 可视化

```text
p14_accuracy_vs_data_fraction.svg
p14_sample_efficiency_auc_bar.svg
p14_accuracy_under_noise.svg
p14_ece_under_noise.svg
p14_robustness_auc_bar.svg
```

---

## 22. P15：Patch/token scaling smoke

### 22.1 目的

v7.1 不把 patch/token scaling 作为 final claim，但要开始验证 dense-to-kernel candidate 是否能进入 token setting。

### 22.2 Task

```text
CIFAR-10-small or TinyImageNet-subset

patch size:
  4x4
  8x8

architectures:
  PatchMLP baseline
  T3 grouped patch block
  best-bridge patch block
  best-kernelized-D3 patch block
```

### 22.3 必须记录

```text
dataset
patch_size
num_tokens
hidden_dim
depth
candidate
memory_ratio_mean
step_ratio_mean
forward_ratio_mean
backward_ratio_mean
val_acc
test_acc
ECE
NLL
tokens_per_second
samples_per_second
scaling_slope_memory
scaling_slope_step
```

### 22.4 判断标准

Patch/token diagnostic pass：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50,
$$

and:

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}-0.02.
$$

Scaling advantage：

$$
\operatorname{slope}_{memory,KAN}<\operatorname{slope}_{memory,MLP}
$$

or:

$$
\operatorname{slope}_{step,KAN}<\operatorname{slope}_{step,MLP}.
$$

### 22.5 可视化

```text
p15_patch_token_accuracy.svg
p15_patch_token_memory_step_pareto.svg
p15_token_scaling_slope.svg
p15_tokens_per_second_bar.svg
```

---

## 23. P16：Candidate selection by efficiency + task + beyond score

### 23.1 目的

P16 汇总 P1-P15，选择 v7.1 的 route。不能只看一个指标，也不能让 dense oracle 和 strict candidate 混在一起。

### 23.2 Survivor 类型

```text
B0:
  S1 efficiency + BasicBeyondPass + at least one beyond metric wins

B1:
  S2 efficiency + BasicBeyondPass + task/time AUC not worse

B2:
  BasicBeyondPass but efficiency fails

B3:
  efficiency passes but task gap persists

B4:
  diagnostic improves task by >=2% but not enough for basic gate

B5:
  no improvement over v7.0

B6:
  contract / gradient fail

B7:
  strict head fail

B8:
  dense oracle only; not route-eligible
```

### 23.3 Beyond score

$$
S_{\text{beyond}}
=
w_1S_{\text{eff}}
+
w_2S_{\text{task}}
+
w_3S_{\text{cal}}
+
w_4S_{\text{sample}}
+
w_5S_{\text{robust}}
+
w_6S_{\text{geom}}.
$$

Default weights：

```text
w_eff = 0.25
w_task = 0.25
w_cal = 0.15
w_sample = 0.15
w_robust = 0.10
w_geom = 0.10
```

Efficiency score：

$$
S_{\text{eff}}
=
\frac{1}{2}
\left(
\frac{1}{r_{\text{mem}}}
+
\frac{1}{r_{\text{step}}}
\right).
$$

Task score：

$$
S_{\text{task}}=
\operatorname{Acc}_{KAN}-\operatorname{Acc}_{MLP}.
$$

Calibration score：

$$
S_{\text{cal}}=
\operatorname{ECE}_{MLP}-\operatorname{ECE}_{KAN}.
$$

Sample score：

$$
S_{\text{sample}}=
\operatorname{AUC}_{data,KAN}-\operatorname{AUC}_{data,MLP}.
$$

Robustness score：

$$
S_{\text{robust}}=
\operatorname{AccDrop}_{MLP}-\operatorname{AccDrop}_{KAN}.
$$

Geometry score：

$$
S_{\text{geom}}=
\operatorname{margin}_{p10,KAN}-\operatorname{margin}_{p10,MLP}.
$$

### 23.4 必须记录字段

```text
candidate
family
oracle_or_mainline
survivor_type
strict_pass
grad_pass
s2_pass
s1_pass
basic_beyond_pass
strong_beyond_pass
memory_ratio_mean
step_ratio_mean
val_acc_mean
test_acc_mean
val_gap_vs_MLP
test_gap_vs_MLP
ValLossAUC_time_delta
ECE_delta
NLL_delta
sample_efficiency_auc_delta
robustness_auc_delta
feature_rank_delta
margin_p10_delta
patch_token_pass
S_eff
S_task
S_cal
S_sample
S_robust
S_geom
S_beyond
route_recommendation
```

### 23.5 可视化

```text
p16_beyond_score_bar.svg
p16_efficiency_task_pareto.svg
p16_survivor_type_dashboard.svg
p16_metric_radar_chart.svg
p16_oracle_vs_route_eligible_table.md
```

---

## 24. P17：Official task re-entry if S1/S0

### 24.1 打开条件

Official task opens only if：

```text
survivor_type in {B0}
P4 one-step pass
r_mem < 1.00
r_step <= 1.35
grad pass
strict pass
no fake/proxy
```

Diagnostic task opens if：

```text
survivor_type in {B1, B4}
P4 pass
r_mem <= 1.05
r_step <= 1.50
```

### 24.2 Official methods

```text
MLP-autograd-fulltrain
D3-dense-poly2-gate reference
best-v71-official-candidate
best-v71-candidate+best-recipe
```

### 24.3 必须记录字段

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

### 24.4 判断标准

Official task pass：

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}-0.01
$$

on all datasets.

At least two datasets must satisfy：

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}.
$$

Wall-clock pass：

$$
\operatorname{ValLossAUC}_{time,KAN}\leq\operatorname{ValLossAUC}_{time,MLP}.
$$

Memory pass：

$$
r_{\text{mem}}<1.00.
$$

### 24.5 可视化

```text
p17_val_loss_vs_step.svg
p17_val_loss_vs_time.svg
p17_accuracy_vs_time.svg
p17_task_efficiency_pareto.svg
p17_seedwise_delta_plot.svg
```

---

## 25. P18：Route decision and artifact audit

### 25.1 Route cases

```text
R1-PureKAN-NG-BeyondMLP-Pass:
  S1 efficiency + BasicBeyondPass + at least one beyond metric wins.
  Move to 5-seed/10-seed confirm and patch/token expansion.

R2-Kernelized-D3-S2-TaskPass:
  S2 efficiency + BasicBeyondPass.
  Continue memory repair toward S1.

R3-Dense-Expressivity-KernelizationFail:
  dense D3 task pass but all kernelized/factorized versions fail efficiency.
  Need lower-level fused dense gate kernel or new primitive.

R4-Bridge-TaskImproves-ButGapPersists:
  bridge improves task by >=2% but not enough for basic gate.
  Continue cross-group bridge design.

R5-Bridge-TooExpensive:
  bridge closes task gap but leaves S2 efficiency envelope.
  Diagnostic only.

R6-StrictHeadFail:
  dense D3 relies on non-KAN head; strict KAN head loses >2%.
  Focus strict head architecture before kernelization claim.

R7-OptimizerOnly:
  optimizer/regularization closes gap without architecture changes.
  Continue recipe confirm, but verify no overfitting.

R8-NoImprovementOverV70:
  no candidate improves task or efficiency.
  Need architecture redesign.

R9-ContractFail:
  nonKAN / fake / proxy / gradient fail.
  Reject candidate.
```

### 25.2 JSON 输出字段

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
ECE_delta
NLL_delta
ValLossAUC_time_delta
sample_efficiency_delta
robustness_delta
patch_token_pass
survivor_type
S_beyond
strict_pass
grad_pass
one_step_probe_pass
official_task_opened
diagnostic_task_opened
primary_blocker
next_required_implementation
no_fake
no_proxy
```

### 25.3 Artifact audit

必须生成：

```text
run_manifest.json
provenance_audit.csv
candidate_registry.csv
gate_config.json
p0_contract.csv
p1_dense_oracle_source_of_gain.csv
p1_task_trace.csv
p2_strict_kan_head_repair.csv
p2_head_trace.csv
p3_structured_candidate_task_transfer.csv
p3_structured_task_trace.csv
p4_gradient_one_step_lock.csv
p5_efficiency_live_set_attribution.csv
p6_microkernel_smoke.csv
p7_crossgroup_bridge.csv
p8_fused_dense_gate_package.csv
p9_distillation_diagnostic.csv
p10_optimizer_matched_control.csv
p11_task_gate.csv
p11_task_trace.csv
p12_efficiency_profiler.csv
p13_long_budget_auc.csv
p14_sample_robust_calibration.csv
p15_patch_token_scaling.csv
p16_candidate_selection.csv
p17_official_task.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

Failure taxonomy：

```text
F1_memory_fail
F2_step_time_fail
F3_gradient_correctness_fail
F4_task_gap_KMNIST
F5_dense_kernelization_fail
F6_bridge_too_expensive
F7_optimizer_no_effect
F8_distillation_only
F9_nonKAN_violation
F10_fake_or_proxy_violation
F11_task_auc_fail
F12_calibration_fail
F13_patch_token_fail
F14_strict_head_fail
F15_purity_gate_fail
F16_artifact_missing
```

### 25.4 必须生成总图

```text
figures/p16_efficiency_task_pareto.svg
figures/p16_beyond_score_bar.svg
figures/p16_metric_radar_chart.svg
figures/p18_route_decision_dashboard.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 26. 第一轮推荐执行顺序

v7.1 不建议一开始把 P0-P18 全部跑满。第一轮应该优先做最能减少不确定性的实验。

### Step 1：P1 dense oracle source-of-gain audit

目的：确认 `poly2_gate` / gate / dense cross-channel 是否真是 D3 gain source。

必须先回答：

```text
D3 是因为 gate 赢？
还是因为 depth / smoothing / head / optimizer 赢？
```

若 H1 不成立，不应继续大规模 kernelization。

### Step 2：P2 strict KAN head repair

目的：确认 dense targeted D3 的 success 能否在 zero non-KAN trainable head 下保留。

若 strict head mean val drop > 0.02，应先做 head 架构，不应宣称 PureKAN-NG。

### Step 3：P3 targeted KMNIST structured transfer

目的：用 KMNIST seed0 快速判断 grouped / low-rank / shared / fused 有没有表达力。

进入下一步的 targeted 标准：

$$
Acc_{\text{KMNIST,seed0}}\geq Acc_{\text{MLP,KMNIST,seed0}}-0.01.
$$

并且：

$$
Acc_{\text{KMNIST,seed0}}\geq Acc_{\text{G2,KMNIST,seed0}}+0.03.
$$

### Step 4：P5 efficiency profiler for top candidates

目的：对 P3 top candidates 做 128/256/512 profiler，排除明显慢/爆内存路线。

进入 3 datasets x 3 seeds official task gate 的最低条件：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

### Step 5：只有出现 S2 + task promising candidate 后，才进入 P11/P12/P13

如果第一轮没有 candidate 同时接近 task 与 efficiency，就不要开 sample efficiency、robustness、patch-token。先修架构/核实现。

---

## 27. 明确停止条件

为了避免无限 sweep，v7.1 设置以下停止条件。

### 27.1 成功停止

立即停止并写复盘：

```text
StrictPass=1
GradPass=1
S1Pass=1
BasicBeyondPass=1
```

进入 strong gate 后停止：

```text
StrictPass=1
GradPass=1
S1Pass=1
StrongBeyondPass=1
```

### 27.2 失败停止

出现以下任一情况，停止并写具体 blocker：

```text
1. 所有 strict head candidate mean val drop > 0.02；
2. 所有 structured candidate 与 dense D3 mean val gap < -0.03；
3. fast/fused candidate 仍 memory_ratio > 1.20 或 step_ratio > 2.00；
4. top task candidate gradient correctness fail 且无法修复；
5. provenance audit 发现 fake/proxy/non-real data；
6. bridge candidate improves task but all variants leave S2 and cannot be kernelized；
7. D3 advantage attribution shows gain is not poly2_gate/cross-channel but only non-strict head。
```

失败时不能写“接近成功”，必须写具体 blocker：

```text
strict_head_fail
structured_expressivity_fail
kernelization_efficiency_fail
gradient_correctness_fail
memory_live_set_fail
bridge_too_expensive
not_poly2_gate_source
```

---

## 28. 成功与失败解释模板

### Case A：kernelized D3 reaches S1 and passes task gate

如果某 D3-derived candidate 满足：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35,
$$

且 BasicBeyondPass 通过，则进入 official confirm。若 ECE/NLL/sample/robust 任一 beyond metric 胜出，可标记为：

```text
PureKAN-NG candidate established
```

但仍需 5-seed / 10-seed confirm。

### Case B：kernelized D3 reaches S2 and passes task gate

如果：

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50,
$$

且 task gate 通过，但 memory 仍 $>1.00$，则 route 为：

```text
R2-Kernelized-D3-S2-TaskPass
```

下一轮继续 memory lifetime repair toward S1。

### Case C：dense D3 remains only task winner

如果 dense D3 持续赢 MLP，但所有 kernelized / factorized / bridge candidates 都不能进入 S2，则当前 blocker 是 dense expressivity kernelization。下一轮必须进入 lower-level fused dense-gate kernel，而不是继续调 optimizer。

### Case D：bridge candidate significantly improves but not enough

如果 bridge 提升 val acc $\geq2\%$，但仍未达到 MLP - 1%，说明 cross-group interaction 是正确方向。下一轮继续 bridge rank / sparsity / kernelization。

### Case E：optimizer matched control closes gap

如果同一 optimizer / regularization recipe 让 G2/T3 或 bridge 直接达到 task gate，且 efficiency envelope 保持，则 route 为：

```text
R7-OptimizerOnly
```

但必须检查 train-val gap、ECE、ValLossAUC_time，避免只是过拟合或 test-tuned。

### Case F：task improvement breaks efficiency

如果某 candidate 通过 task gate 但 memory/step 离开 S2，它只能作为 diagnostic。不能作为 PureKAN-NG success。

### Case G：strict head fails

如果 dense D3 的优势依赖 non-KAN linear head，而所有 strict KAN head 都明显退化，则不要继续 kernelization success claim。下一轮主线是 strict classifier head。

### Case H：no candidate improves over v7.0

如果没有 candidate 同时改善 task 或 efficiency，说明 v7.1 的 bridge/kernelization 设计不足，需要重新设计 function-space primitive。

---

## 29. 最终建议

v7.1 的一句话策略是：

$$
\boxed{
\text{以 dense D3 为 task-expression oracle，以 T3/G2 为 efficient kernel base，做 strict head + dense-to-kernel bridge。}
}
$$

v7.0 已经证明：

```text
dense D3 可以超过 MLP；
T3/G2 可以高效但 task gap 未闭合；
dense D3 fastopt 后仍效率失败；
simple optimizer / head / input normalization probes 不能闭合 grouped KMNIST gap。
```

因此 v7.1 不应该再问“有没有表达力”或“有没有效率”。答案分别是：dense 有表达力，T3/G2 有效率。真正问题是：

$$
\boxed{
\text{能否把 D3 的表达力在 strict PureKAN 条件下压缩、稀疏化、低秩化、kernel 化？}
}
$$

如果 v7.1 成功，项目将从“有一个 dense upper-bound 和一个 efficient lower-task candidate”推进到“有一个真正的 PureKAN-NG candidate”。如果 v7.1 失败，失败也会非常有信息量：它会告诉我们 dense D3 的优势到底是否能被当前 grouped / factorized / fused kernel family 承载。
