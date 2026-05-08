# DG-KAN v7.5：Effective Expressivity、Optimization Reachability 与 Dense-Preserving Kernel-Native Bridge 完整实验计划

> 本计划基于 v7.4 real-only 复盘制定。v7.4 的核心结论不是“表达力失败”，也不是“optimizer 小修失败”，而是：**C3 继续证明 strict/cached KAN 的 macro 表达力优势真实存在；A2S/A2C/A4C/A5C 证明低 live-set bridge 可以显著降低 memory，但会损失一部分 effective expressivity；当前真正 blocker 是如何在低 live-set / kernel-native 约束下保住 C3 的有效表达力，并完成全 shape S2/S1。**  
> 本计划明确不再把 classwise-safe、loss 调参、sampler、class weight、focal、target-margin 作为主线。它们最多用于局部诊断，不是项目目标。本轮目标是机制闭环，不是刷榜。

---

## 0. 当前实验结果的本质判断

v7.4 的主 run 为：

```bash
python experiments/run_gafu_v74_real.py \
  --out-dir results/real_rerun_20260506/v74_mechanism_kernel_bridge_5seed_20260506T021447Z \
  --fresh \
  --device auto \
  --candidates B0,C3,A2C,A2S,A4C,A5C \
  --seeds 0,1,2,3,4 \
  --bench-batch-sizes 128,256,512 \
  --bench-warmup 5 \
  --bench-reps 30 \
  --grad-batch-sizes 8,128 \
  --bootstrap-reps 5000
```

本轮遵守 real-only / no-fake / no-proxy 约束：

```text
rows_checked = 55
fake_proxy_nonzero_count = 0
no_fake = true
no_proxy = true
```

route best 是 `C3`：

```text
candidate:
  H3-cached-hidden-y-rbf-head

macro val gap:
  +0.023046875

CI95 low:
  +0.0158854167

Holm p:
  2.90e-04

test gap:
  +0.0244791667

ECE delta:
  -0.001797389

NLL delta:
  -0.099492782

GradPass:
  6/6, max relerr = 6.02e-05

memory ratio:
  1.3069108164

step ratio:
  3.9089837584

route:
  R3-MacroExpressivityPositiveButNotKernelNative
```

这说明：

$$
\boxed{
\text{C3 的 macro 表达力优势真实存在，且不是 loss/sampler/classwise 小修导致。}
}
$$

但它也说明：

$$
\boxed{
\text{C3 不是 kernel-native efficient candidate。}
}
$$

结构性候选 `A2S/A2C/A4C/A5C` 的意义更重要。它们不是小修，而是低 live-set bridge。它们把 `C3` 的 memory ratio 从 `1.3069` 降到 `1.0368-1.2129` 区间，但 macro gap 低于 `C3`：

```text
C3  macro gap = +0.0230
A2S macro gap = +0.0199
A2C macro gap = +0.0190
A4C macro gap = +0.0190
A5C macro gap = +0.0189
```

其中 `A2S` 是当前最接近合流的候选：

```text
A2S val acc = 0.8642
A2S test acc = 0.8182
A2S val gap vs MLP = +0.0199
A2S CI95 low = +0.0125
A2S Holm p = 2.51e-03
A2S test gap = +0.0233
A2S ECE delta = -0.0004
A2S NLL delta = -0.0903
A2S GradPass = 6/6
A2S memory ratio mean = 1.0368
A2S step ratio mean = 1.6691 in main v7.4 run
A2S S2 shapes = 4/9 in main v7.4 run
```

追加 low-cost structural follow-up 中，`A2S` 达到：

```text
macro gap = +0.0199
GradPass = 6/6
memory ratio = 1.0368
step ratio = 1.3759
S2 shapes = 5/9
```

但仍不是成功，因为：

```text
1. macro gate 仍差极小幅度，没有达到 +0.0200；
2. full-grid S2 没闭合；
3. 真实 live-set / kernel-count attribution 仍未完成；
4. materialized_tensor_count 仍未测，不能声称 H5 attribution pass。
```

因此 v7.5 的本质目标不是继续把 `A2S` 调高一点，也不是继续把 `C3` 快一点，而是回答：

$$
\boxed{
\text{C3 的有效表达力能否通过 A2S 类低 live-set bridge 保留下来，并用 fused/materialization-free kernel 完成全 shape S2？}
}
$$

---

## 1. 当前进展总结

### 1.1 已经确定的正向进展

第一，macro 表达力已经真实存在。`C3` 的 macro val gap 为 `+0.0230`，CI95 lower bound 为 `+0.0159`，Holm p 为 `2.90e-04`，test gap 为 `+0.0245`，同时 ECE/NLL 均优于 MLP。这说明当前不是“KAN basis 弱于 MLP”。

第二，strict/manual/gradient 路径没有倒退。`C3/A2S/A5C` 均通过 full gradient correctness，`C3` max relerr 为 `6.02e-05`，`A2S` max relerr 为 `9.27e-05`。这说明主问题不是梯度公式错误。

第三，低 live-set bridge 方向有真实突破。`A2S/A2C` 把 memory ratio 拉到 `1.0368`，已经低于 S2 mean gate 的 `1.05`，这比 C3 的 `1.3069` 大幅前进。它说明不是所有 task-strong KAN 都必然 memory-heavy；结构桥接有效。

第四，v7.4 没有做 loss/sampler/classwise 小修，失败形式直接来自结构候选本身。这一点很重要，因为它避免了把问题重新误导到 loss trick。当前失败不是 focal/margin/class weighting 没调好，而是 structural bridge 尚未同时保住 macro 与 full-grid S2。

### 1.2 已经被降级的方向

下面方向不应再作为主线：

```text
1. classwise-safe:
   只作为局部模式诊断，不作为 hard gate。

2. loss / sampler / class weight / focal / target-margin:
   v7.4 没有把它们作为主线，本轮失败不是这些因素导致。

3. AdamW 小参数 sweep:
   过去多轮已说明普通 lr / wd / smoothing 小扫不是主因。

4. hidden shrink:
   hidden48/32 会显著损失 task signal，memory 仍约 1.29，不能作为闭合路线。

5. naive low-rank:
   没保住 H3/H4 task signal，且 gradient gate 曾不完整。

6. Python-level vectorization:
   可以降 loop overhead，但常常增加 materialization 和 memory live-set，不是 final kernel-native 解法。
```

---

## 2. 问题所在：不是表达力不行，而是约束下的 effective expressivity 没闭合

当前最准确的分解是：

```text
C3:
  表达力强；
  macro significant pass；
  grad pass；
  efficiency fail。

A2S:
  low live-set 成功；
  grad pass；
  macro 接近 pass；
  mean memory 接近或已经低于 S2；
  full-grid S2 shape fail。

A2C/A4C:
  macro 接近；
  memory/step 部分接近；
  gradient pass 不完整或不稳定。

A5C:
  gradient pass；
  macro 接近；
  step/memory 仍不稳定。
```

这说明当前不是一个单点问题，而是一个三角约束：

$$
\boxed{
\text{expression preservation}
+
\text{low live-set}
+
\text{kernel-native step stability}
}
$$

C3 占据 expression 角，但 live-set 太大。A2S 占据 low live-set 角，但 expression 还差一点，shape-level step/memory 还不稳定。v7.5 的目标是把 C3 与 A2S 合流，而不是继续在任意一端做小修。

---

## 3. 是否在正确道路上

是，但路线必须进一步收敛。

v7.4 支持三条判断：

```text
1. dense/cached KAN 的 macro 表达力真实存在；
2. low live-set bridge 能接近 C3 的 task，而且大幅降低 memory；
3. 剩余问题是机制性问题：为什么 A2S 差那 0.3%-0.4% macro gap，以及为什么 full-grid S2 没闭合。
```

所以当前正确道路不是：

```text
继续调 loss
继续扫 optimizer 小参数
继续压 hidden
继续追 classwise
```

而是：

```text
1. C3 vs A2S source-of-loss attribution；
2. A2S 的 distillation reachability；
3. A2S 的 full-grid shape-specific S2 attribution；
4. fused transform+mix backward / streaming head / gate / grad temp；
5. 若 A2S cannot recover expression，则设计 new low-live-set primitive。
```

---

## 4. 离目标还差多远

### 4.1 C3 到 S2 的差距

C3 当前：

$$
r_{mem}=1.3069,
$$

$$
r_{step}=3.9090.
$$

S2 要求：

$$
r_{mem}\leq1.05,
$$

$$
r_{step}\leq1.50.
$$

memory 需要降低：

$$
1-\frac{1.05}{1.3069}\approx19.66\%.
$$

step 需要提速：

$$
1-\frac{1.50}{3.9090}\approx61.63\%.
$$

这说明 C3 自身不适合作为直接工程收敛对象。它是 teacher / reference / expression oracle，而不是短期 S2 target。

### 4.2 A2S 到 macro gate 的差距

A2S 当前：

$$
\Delta Acc_{macro,val}=+0.0199.
$$

macro gate：

$$
\Delta Acc_{macro,val}\geq+0.0200.
$$

差距约：

$$
0.0200-0.0199\approx0.0001.
$$

从 task 数值看，A2S 已经非常接近。但不能四舍五入为成功。下一步需要用机制性方法确认这 `0.0001-0.003` 的 gap 是 sampling noise、optimization reachability，还是结构表达力确实不足。

### 4.3 A2S 到 S2 的差距

main v7.4 中 A2S：

$$
r_{mem,mean}=1.0368,
$$

$$
r_{step,mean}=1.6691.
$$

按 mean step 到 S2：

$$
1-\frac{1.50}{1.6691}\approx10.13\%.
$$

但 full-grid S2 的难点不只是 mean。A2S 的 per-shape 结果显示：

```text
bs=512 memory ratio ≈ 1.0816 > 1.05
部分 shape step ratio > 1.50
S2 shapes = 4/9 in main run
S2 shapes = 5/9 in low-cost follow-up
```

因此 A2S 到 S2 的真正差距是 shape stability，而不是均值差一点。需要把 bs=512 memory 从 `1.0816` 拉到 `<=1.05`，约需降低：

$$
1-\frac{1.05}{1.0816}\approx2.92\%.
$$

并把 worst-shape step 降到 `<=1.50`。

### 4.4 当前最大缺口

当前最大缺口不是 task，也不是 memory mean，而是：

$$
\boxed{
\text{缺少真实 live-set / kernel-count attribution，导致还不知道 A2S full-grid S2 fail 的根源。}
}
$$

v7.4 明确指出 `materialized_tensor_count` 仍是 `not_measured`，不能声称 materialization 占 peak 的比例，也不能随意写“fused kernel 一定能解决”。v7.5 必须先补这个洞。

---

## 5. v7.5 总目标

v7.5 的整体目标是：

$$
\boxed{
\text{把 A2S 从 near-macro + near-S2 推进到 MacroSignificantPass + full-grid S2Pass。}
}
$$

同时用 C3 作为 teacher / expression oracle，回答：

$$
\boxed{
\text{A2S 差 C3 的表达力到底能否通过 distillation 或轻量 bridge 恢复？}
}
$$

v7.5 不以“再调一个超参”为目标，而是建立以下机制结论：

```text
1. A2S macro gap 低于 C3 的原因；
2. A2S full-grid S2 fail 的 live-set / kernel-count source；
3. A2S 是否 optimization-reachable；
4. 如果 A2S 结构不够，应该增加哪种最低 live-set 的 expression bridge；
5. 如果 A2S 结构够，应该设计什么 KAN-specific optimizer / distillation objective；
6. 如果 A2S shape fail 是 kernel问题，应该 fused 哪些 microkernel。
```

v7.5 最低成功标准：

$$
\boxed{
\text{StrictPass}+\text{GradPass}+\text{MacroSignificantPass}+\text{FullGridS2Pass}.
}
$$

v7.5 正式成功标准：

$$
\boxed{
\text{StrictPass}+\text{GradPass}+\text{MacroSignificantPass}+\text{FullGridS1Pass}+\text{TimeAUCPass}.
}
$$

如果 v7.5 无法达到完整成功，也必须输出明确 route：

```text
optimizer_reachability
effective_expressivity_loss
kernel_materialization_fail
new_primitive_required
```

---

## 6. v7.5 禁止事项

第一，不允许使用 fake data、proxy row、固定占位 ratio 或手填 pass。所有未实现项必须写为：

```text
not_implemented
not_run
not_applicable
metric_unavailable
```

第二，不允许把 `A2S macro +0.0199` 四舍五入为 `+0.0200`。gate 不过就是不过。

第三，不允许把 mean memory pass 写成 full-grid S2 pass。必须看 9/9 shapes。

第四，不允许回到 loss/sampler/class weight/focal/target-margin 作为主线。

第五，不允许把 classwise 作为 hard gate。classwise 只记录为 local failure audit，不参与主 route gate，除非出现极端 collapse。

第六，不允许把 C3 当最终 candidate。C3 是 expression teacher，不是 S2/S1 candidate。

第七，不允许在没有 attribution 的情况下盲写 fused kernel。P1/P2 必须先定位 full-grid fail source。

第八，不允许把 optimizer 小调叫 optimizer research。只有 distillation/reachability 或 KAN-specific optimizer 才是本轮 optimizer 研究。

第九，不允许继续 hidden shrink / naive low-rank 作为主线，除非它通过 distillation 或 attribution 证明可保表达力。

第十，没有 GradPass 的 candidate 不能进入 route best。

---

## 7. 核心假设

### H1：A2S 是当前最有希望的 bridge，但缺失少量 C3 expression component

A2S 已经做到：

```text
macro gap close to +0.02
GradPass = 6/6
memory mean around 1.0368
S2 shapes 4/9 or 5/9
```

H1 假设：A2S 的表达力损失不是大规模函数类不足，而是缺少少量 C3 中的 nonlinear stack interaction。

H1 成立标准：

$$
Acc_{C3}-Acc_{A2S}\leq0.005.
$$

并且 feature / logit 差异集中在少数 component：

```text
logit residual rank small
feature CKA high
teacher-student KL low
margin distribution close
```

量化标准：

$$
CKA(A2S,C3)\geq0.90.
$$

$$
KL(p_{C3}||p_{A2S})\leq0.05.
$$

如果 H1 成立，下一步应加最小 live-set residual bridge，而不是回到 full C3。

---

### H2：A2S macro gap 可通过 C3 teacher distillation 恢复，而不增加 inference live-set

H2 假设：A2S 结构已经足够，当前差距主要来自 optimization / objective。用 C3 teacher 的 logit 或 feature distillation 可以让 A2S 通过 macro gate，而不改变 inference 结构。

Distillation loss：

$$
L =
L_{\text{CE}}
+
\alpha T^2 KL(p_{C3}^{T}||p_{A2S}^{T})
+
\beta L_{\text{feature}}.
$$

H2 成立标准：

$$
Acc_{\text{A2S-distill}}-Acc_{\text{A2S-supervised}}\geq0.002.
$$

并且：

$$
\Delta Acc_{\text{macro,val,A2S-distill}}\geq0.0200.
$$

同时：

$$
r_{mem}\leq1.05,
$$

$$
r_{step}\leq1.50
$$

on full-grid after training/inference measurement.

如果 distillation 提升 macro 但 full-grid S2 不变，则 H2 说明 task gap 是 optimization/objective；S2 仍需 kernel path 修复。

如果 distillation 不能提升：

$$
Acc_{\text{A2S-distill}}-Acc_{\text{A2S-supervised}}<0.001,
$$

则说明 A2S 结构有效表达力不足，需要 structural bridge。

---

### H3：A2S full-grid S2 fail 来自 shape-specific live-set / kernel fragmentation，不是数学不可高效

A2S mean memory 已经过 S2，但 bs=512 memory 超过 `1.05`，部分 shape step 超过 `1.50`。H3 假设：这是 implementation / materialization 问题。

H3 成立标准：

P1/P2 attribution 必须满足：

```text
top3 memory sources identified
unknown_memory_fraction <= 0.10
component timing explains >= 90% step time
materialized_tensor_count measured
kernel_count_total measured
```

并且发现至少一个 source 满足：

$$
\frac{M_{\text{source}}}{M_{\text{peak}}}\geq0.15
$$

或：

$$
\frac{T_{\text{source}}}{T_{\text{step}}}\geq0.20.
$$

H3 repair 成立标准：

$$
r_{mem,bs512}\leq1.05,
$$

$$
r_{step,worstshape}\leq1.50.
$$

---

### H4：A2S 需要的 expression bridge 应该是局部/稀疏/streaming，而不是 dense stack 回退

H4 假设：如果 A2S 结构确实不足，只需要极小额外 nonlinear interaction，而不需要回到 full C3。

候选 bridge：

```text
B1-A2S+single-layer-poly2-residual
B2-A2S+tile-stream-cross-feature-gate
B3-A2S+rank2-edge-owned-nonlinear-bridge
B4-A2S+prototype-distance-head-lite
B5-A2S+overlap-grouped-local-dense-bridge
B6-A2S+C3-residual-student-projection
```

H4 成立标准：

$$
\Delta Acc_{\text{macro,val}}\geq0.002
$$

relative to A2S, and:

$$
\Delta r_{mem}\leq0.01,
$$

$$
\Delta r_{step}\leq0.05.
$$

也就是说，只允许极小 overhead 换取足够 macro recovery。

---

### H5：几何约束不是本轮主线，但应作为 Pareto diagnostic 防止误判

H5 假设：几何可能有助于 calibration/NLL/AUC，但不能牺牲 macro。v7.5 只做小规模 geometry Pareto，不作为 hard target。

定义：

$$
L = L_{\text{task}}+\lambda L_{\text{geo}}.
$$

H5 有效标准：

$$
Acc_{\lambda}\geq Acc_{\lambda=0}-0.005
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

如果：

$$
Acc_{\lambda}<Acc_{\lambda=0}-0.02,
$$

则几何约束伤害表达力，本轮应冻结 geometry。

---

### H6：如果 A2S + distillation + minimal bridge + fused kernel 仍不能闭合，则需要 new primitive

H6 是 Stop/Go 假设。如果完成：

```text
A2S source-of-loss attribution
C3 teacher distillation
minimal expression bridge
full-grid live-set attribution
materialization-free fused package
```

仍没有 `MacroSignificantPass + FullGridS2Pass`，则当前 A2S/C3 family 需要被替换。下一轮应设计 new low-live-set primitive，不再继续局部修补。

---

## 8. Candidate 设计

### 8.1 Baselines and teachers

```text
B0-MLP-autograd-reference
B1-G2-O7-efficient-grouped-reference
B2-C3-cached-hidden-y-rbf-head-teacher
B3-A2S-low-live-set-bridge
B4-A2C-low-live-set-bridge
B5-A5C-late-poly2-reference
```

### 8.2 A2S distillation candidates

```text
D0-A2S-supervised-reference
D1-A2S-logit-distill-C3-T2-alpha0.25
D2-A2S-logit-distill-C3-T4-alpha0.25
D3-A2S-logit-distill-C3-T4-alpha0.50
D4-A2S-feature-distill-C3-beta0.10
D5-A2S-logit+feature-distill-C3
D6-A2S-distill-then-supervised-finetune
D7-A2S-C3-margin-matching
```

### 8.3 Minimal expression bridge candidates

```text
E0-A2S-reference
E1-A2S-single-layer-poly2-residual
E2-A2S-first-layer-tiny-poly2-residual
E3-A2S-late-layer-tiny-poly2-residual
E4-A2S-tile-stream-cross-feature-gate
E5-A2S-rank2-edge-owned-nonlinear-bridge
E6-A2S-rank4-edge-owned-nonlinear-bridge
E7-A2S-prototype-distance-head-lite
E8-A2S-overlap-grouped-local-dense-bridge
E9-A2S-C3-residual-student-projection
```

### 8.4 A2S kernelization candidates

```text
K0-A2S-current
K1-A2S-fused-head-forward
K2-A2S-fused-head-backward
K3-A2S-gate-temp-streaming
K4-A2S-grad-gate-streaming
K5-A2S-grad-poly-streaming
K6-A2S-no-full-basis-materialization
K7-A2S-fused-transform-mix-backward
K8-A2S-tile-stream-forward-backward
K9-A2S-materialization-free-combo
K10-A2S-bs512-memory-specialized-tile-stream
K11-A2S-worstshape-step-specialized-fused-path
```

### 8.5 New primitive candidates, only if needed

```text
NP0-LinearBody+KANMixtureHead
NP1-TileStreamDensePoly2Gate-lite
NP2-OverlapGroupedPoly2Gate-lite
NP3-BlockSparseDenseGate-lite
NP4-EdgeOwnedMixtureOfBasis
NP5-PrototypeDistanceKANLowLiveSet
NP6-MaterializationFreeLocalDenseKAN
```

---

## 9. 实验阶段总览

v7.5 分为十一个阶段：

```text
P0: v7.4 reproduction and route lock
P1: C3 vs A2S source-of-loss attribution
P2: A2S full-grid live-set and kernel-count attribution
P3: C3-teacher distillation reachability
P4: minimal expression bridge construction
P5: materialization-free A2S microkernel benchmark
P6: A2S fused / tile-stream full package
P7: geometry Pareto diagnostic
P8: 10-seed task + AUC verification
P9: full-grid efficiency profiler
P10: candidate co-selection and route decision
P11: artifact and failure audit
```

P0-P2 是定位阶段。  
P3-P4 判别 optimization vs effective expressivity。  
P5-P6 解决 kernel-native efficiency。  
P7 只做 Pareto diagnostic。  
P8-P11 做最终证据闭环。

---

## 10. P0：v7.4 reproduction and route lock

### 10.1 目的

确认 v7.5 与 v7.4 可比，并锁定 no-fake / no-proxy / strict / grad / baseline denominator。

### 10.2 必跑对象

```text
B0-MLP-autograd-reference
C3-cached-hidden-y-rbf-head
A2S-low-live-set-bridge
A2C-low-live-set-bridge
A5C-late-poly2-reference
```

### 10.3 必须记录字段

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
ECE_delta
NLL_delta
memory_ratio_mean
step_ratio_mean
forward_ratio_mean
backward_ratio_mean
S2_shape_count
reproduction_delta_val_gap
reproduction_delta_memory_ratio
reproduction_delta_step_ratio
```

### 10.4 判断标准

P0 pass：

```text
fake_data_used = 0
proxy_row_used = 0
nonKAN_param_count = 0
manual_backward_available = 1
uses_loss_backward = 0
grad_relerr_max <= 1e-4
grad_cos_min >= 0.999
```

C3 task reproduction：

$$
|\Delta Acc_{C3,v75}-0.0230|\leq0.005.
$$

A2S task reproduction：

$$
|\Delta Acc_{A2S,v75}-0.0199|\leq0.005.
$$

A2S efficiency reproduction：

$$
|r_{mem,A2S,v75}-1.0368|\leq0.03.
$$

$$
|r_{step,A2S,v75}-r_{step,A2S,v74}|\leq0.20.
$$

### 10.5 可视化

```text
p0_reproduction_task_bar.svg
p0_reproduction_efficiency_bar.svg
p0_contract_heatmap.svg
p0_c3_a2s_position_pareto.svg
```

---

## 11. P1：C3 vs A2S source-of-loss attribution

### 11.1 目的

解释 A2S 为什么比 C3 低约 `0.0031` macro gap。P1 不是修复，而是定位表达力差异。

### 11.2 实验对象

```text
C3
A2S
A2C
A5C
MLP
```

### 11.3 设置

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2,3,4

steps:
  240

logging:
  every 20 steps
```

### 11.4 必须记录字段

#### Task metrics

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
ValLossAUC_step
ValLossAUC_time
```

#### Teacher-student similarity

```text
teacher_candidate
student_candidate
logit_KL
logit_cos
feature_CKA_layer1
feature_CKA_layer2
feature_CKA_head
feature_rank
margin_mean
margin_p10
margin_p50
confidence_mean
wrong_confidence_mean
```

#### Component contribution

```text
head_output_norm
body_output_norm
poly2_residual_norm
linear_body_norm
gate_activation_mean
gate_activation_entropy
C3_minus_A2S_logit_residual_rank
C3_minus_A2S_logit_residual_norm
top_residual_modes
```

### 11.5 判断标准

A2S close-to-C3 if：

$$
Acc_{C3}-Acc_{A2S}\leq0.005.
$$

Teacher-student similarity high if：

$$
CKA(A2S,C3)\geq0.90.
$$

and:

$$
KL(p_{C3}||p_{A2S})\leq0.05.
$$

If C3-A2S residual is low rank：

```text
top 2 residual modes explain >= 80% of logit residual norm
```

then minimal bridge should be low-rank / projection-like. If residual is high-rank, A2S structural expressivity is insufficient and full new primitive may be needed.

### 11.6 可视化

```text
p1_c3_a2s_val_gap_by_dataset.svg
p1_teacher_student_logit_kl.svg
p1_feature_cka_heatmap.svg
p1_logit_residual_spectrum.svg
p1_margin_distribution_c3_vs_a2s.svg
p1_source_of_loss_dashboard.svg
```

---

## 12. P2：A2S full-grid live-set and kernel-count attribution

### 12.1 目的

补齐 v7.4 未完成的 H5 attribution。必须定位 A2S full-grid S2 fail 的真实 source，而不是猜 materialization。

### 12.2 必跑对象

```text
MLP
C3
A2S
A2C
A5C
```

### 12.3 Shape grid

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
  native
  2
  3
```

### 12.4 必须记录字段

#### Efficiency summary

```text
candidate
dataset
batch_size
depth
peak_allocated_MB
peak_reserved_MB
MLP_peak_allocated_MB
memory_ratio
step_ratio
forward_ratio
backward_ratio
update_ratio
S2_pass
```

#### Live-set attribution

```text
largest_live_tensor_MB
basis_temp_MB
gate_temp_MB
head_temp_MB
grad_gate_MB
grad_poly_MB
linear_body_temp_MB
poly2_residual_temp_MB
manual_cache_MB
optimizer_state_MB
allocator_padding_MB
reserved_unallocated_MB
top1_memory_source
top2_memory_source
top3_memory_source
unknown_memory_fraction
```

#### Kernel / op counts

```text
materialized_tensor_count
torch_op_count
triton_kernel_count
elementwise_kernel_count
gemm_count
allocation_proxy_count
kernel_launch_proxy_count
python_loop_count
layout_conversion_count
```

#### Timing components

```text
head_forward_ms
head_backward_ms
body_forward_ms
body_backward_ms
gate_eval_ms
poly2_eval_ms
loss_delta_ms
optimizer_update_ms
kernel_launch_proxy_ms
layout_conversion_ms
```

### 12.5 判断标准

Attribution pass：

```text
top3 memory sources identified
unknown_memory_fraction <= 0.10
component timing explains >= 90% step time
materialized_tensor_count measured
kernel_count_total measured
```

A2S memory fail source found if：

$$
\frac{M_{\text{source}}}{M_{\text{peak}}}\geq0.15.
$$

A2S step fail source found if：

$$
\frac{T_{\text{source}}}{T_{\text{step}}}\geq0.20.
$$

If bs512 memory fail is allocator/reserved, prioritize allocator/lifetime trim. If it is head/gate temp, prioritize streaming head/gate. If it is kernel count, prioritize fused backward.

### 12.6 可视化

```text
p2_a2s_memory_waterfall_by_shape.svg
p2_a2s_step_time_waterfall_by_shape.svg
p2_bs512_memory_source_bar.svg
p2_kernel_count_by_phase.svg
p2_materialized_tensor_count_heatmap.svg
p2_shape_s2_fail_dashboard.svg
```

---

## 13. P3：C3-teacher distillation reachability

### 13.1 目的

判断 A2S 缺的 macro gap 是 optimization/objective 问题，还是 structure/effective expressivity 问题。

### 13.2 Candidates

```text
D0-A2S-supervised-reference
D1-A2S-logit-distill-C3-T2-alpha0.25
D2-A2S-logit-distill-C3-T4-alpha0.25
D3-A2S-logit-distill-C3-T4-alpha0.50
D4-A2S-feature-distill-C3-beta0.10
D5-A2S-logit+feature-distill-C3
D6-A2S-distill-then-supervised-finetune
D7-A2S-C3-margin-matching
```

### 13.3 设置

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2,3,4

steps:
  240

teacher:
  C3 frozen checkpoint from same seed
```

### 13.4 必须记录字段

```text
candidate
dataset
seed
teacher_seed
distill_temperature
alpha_logit
beta_feature
train_acc
val_acc
test_acc
val_gap_vs_MLP
test_gap_vs_MLP
val_gap_vs_C3
ECE
NLL
ValLossAUC_step
ValLossAUC_time
teacher_student_KL
teacher_student_logit_cos
feature_CKA
distill_improvement_vs_A2S_supervised
memory_ratio_inference
step_ratio_inference
memory_ratio_training
step_ratio_training
```

### 13.5 判断标准

Optimization reachability pass：

$$
Acc_{A2S-distill}-Acc_{A2S-supervised}\geq0.002.
$$

and:

$$
\Delta Acc_{\text{macro,val,A2S-distill}}\geq0.0200.
$$

If distillation gives task pass without inference overhead increase, route becomes：

```text
optimizer_objective_reachability
```

If distillation fails：

$$
Acc_{A2S-distill}-Acc_{A2S-supervised}<0.001,
$$

route becomes：

```text
effective_expressivity_loss
```

and P4 minimal expression bridge becomes mandatory.

### 13.6 可视化

```text
p3_distill_macro_gap_bar.svg
p3_distill_vs_supervised_delta.svg
p3_teacher_student_KL_curve.svg
p3_feature_CKA_distill_heatmap.svg
p3_task_efficiency_inference_pareto.svg
```

---

## 14. P4：Minimal expression bridge construction

### 14.1 目的

如果 P3 证明 A2S 结构不够，P4 加最小 expression bridge。原则是只补 A2S 缺失的 C3 residual，不回到 full C3。

### 14.2 Candidates

```text
E0-A2S-reference
E1-A2S-single-layer-poly2-residual
E2-A2S-first-layer-tiny-poly2-residual
E3-A2S-late-layer-tiny-poly2-residual
E4-A2S-tile-stream-cross-feature-gate
E5-A2S-rank2-edge-owned-nonlinear-bridge
E6-A2S-rank4-edge-owned-nonlinear-bridge
E7-A2S-prototype-distance-head-lite
E8-A2S-overlap-grouped-local-dense-bridge
E9-A2S-C3-residual-student-projection
```

### 14.3 必须记录字段

```text
candidate
bridge_type
bridge_rank
bridge_layer
bridge_param_count
nonKAN_param_count
edge_param_count_delta
strict_pass
grad_pass
val_acc_mean
test_acc_mean
val_gap_vs_MLP
test_gap_vs_MLP
val_gap_vs_A2S
ECE
NLL
memory_ratio_mean
step_ratio_mean
memory_overhead_vs_A2S
step_overhead_vs_A2S
feature_CKA_to_C3
logit_KL_to_C3
```

### 14.4 判断标准

Bridge useful if：

$$
\Delta Acc_{\text{macro,val}}\geq0.002
$$

relative to A2S, and:

$$
\Delta r_{mem}\leq0.01,
$$

$$
\Delta r_{step}\leq0.05.
$$

Bridge pass if：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200,
$$

and:

$$
r_{mem}\leq1.05,
$$

$$
r_{step}\leq1.50
$$

on full-grid.

If bridge improves macro but breaks S2, it is diagnostic-only. If bridge preserves S2 but macro remains below `+0.0200`, it is not enough.

### 14.5 可视化

```text
p4_bridge_macro_vs_overhead.svg
p4_bridge_feature_recovery.svg
p4_bridge_logit_residual_spectrum.svg
p4_bridge_s2_boundary_plot.svg
p4_bridge_candidate_scorecard.svg
```

---

## 15. P5：Materialization-free A2S microkernel benchmark

### 15.1 目的

对 P2 定位出的 A2S bottleneck 做 microkernel 级验证。目标不是直接宣称成功，而是证明某个 fused/streaming kernel 真实降低 memory/time 且梯度正确。

### 15.2 Microkernels

```text
MK0-A2S-current-head-forward
MK1-A2S-fused-head-forward
MK2-A2S-current-head-backward
MK3-A2S-fused-head-backward
MK4-A2S-gate-temp-streaming
MK5-A2S-grad-gate-streaming
MK6-A2S-grad-poly-streaming
MK7-A2S-no-full-basis-materialization
MK8-A2S-fused-transform-mix-backward
MK9-A2S-tile-stream-forward-backward
MK10-A2S-bs512-memory-specialized-tile
MK11-A2S-worstshape-step-specialized-fused-path
```

### 15.3 必须记录字段

```text
microkernel
input_shape
output_shape
dataset
batch_size
current_time_ms
candidate_time_ms
time_ratio_vs_current
current_memory_MB
candidate_memory_MB
memory_ratio_vs_current
forward_relerr
grad_relerr
grad_cos
materialized_tensor_count_current
materialized_tensor_count_candidate
kernel_count_current
kernel_count_candidate
allocation_count_current
allocation_count_candidate
bandwidth_estimate_GBps
```

### 15.4 判断标准

Microkernel useful if：

$$
\frac{T_{candidate}}{T_{current}}\leq0.80
$$

or:

$$
\frac{M_{candidate}}{M_{current}}\leq0.80.
$$

Microkernel enters full package if：

```text
grad_relerr < 1e-4
grad_cos > 0.999
time_ratio_vs_current <= 0.90
memory_ratio_vs_current <= 0.95
```

For bs512 memory repair：

$$
r_{mem,bs512}\leq1.05.
$$

For worst-shape step repair：

$$
r_{step,worstshape}\leq1.50.
$$

### 15.5 可视化

```text
p5_microkernel_memory_time_pareto.svg
p5_microkernel_grad_correctness.svg
p5_materialization_reduction_bar.svg
p5_kernel_count_reduction_bar.svg
p5_bs512_repair_microkernel_bar.svg
```

---

## 16. P6：A2S fused / tile-stream full package

### 16.1 目的

集成 P5 有效 microkernel 到 full training package，验证是否能达成 MacroSignificantPass + FullGridS2Pass。

### 16.2 Candidates

```text
F0-A2S-current
F1-A2S-fused-head-forward
F2-A2S-fused-head-backward
F3-A2S-gate-grad-streaming
F4-A2S-no-full-basis-materialization
F5-A2S-fused-transform-mix-backward
F6-A2S-tile-stream-forward-backward
F7-A2S-bs512-memory-specialized-tile-stream
F8-A2S-worstshape-step-fused-path
F9-A2S-materialization-free-S2-combo
F10-A2S-distilled-materialization-free-combo
F11-A2S-bridge-materialization-free-combo
```

### 16.3 必须记录字段

```text
package
components
implementation_status
strict_pass
grad_pass
uses_fused_head
uses_streaming_gate
uses_streaming_grad_poly
uses_no_full_basis_materialization
uses_tile_stream
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
forward_ratio_mean
backward_ratio_mean
update_ratio_mean
S2_shape_count
S2_fullgrid_pass
S1_fullgrid_pass
val_acc_mean
test_acc_mean
val_gap_vs_MLP
test_gap_vs_MLP
val_gap_vs_A2S
ECE
NLL
ValLossAUC_step
ValLossAUC_time
materialized_tensor_count
kernel_count_total
```

### 16.4 判断标准

Package efficiency pass：

$$
r_{mem,max}\leq1.05,
$$

$$
r_{step,max}\leq1.50.
$$

Macro pass：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200.
$$

Expression preservation：

$$
Acc_{package}\geq Acc_{A2S}-0.002.
$$

If F10 distill combo passes but F0/F9 supervised does not, route marks：

```text
optimization_objective_needed
```

If F11 bridge combo passes, route marks：

```text
minimal_expression_bridge_success
```

### 16.5 可视化

```text
p6_package_efficiency_task_pareto.svg
p6_shape_s2_heatmap.svg
p6_package_task_preservation_bar.svg
p6_val_loss_auc_time_bar.svg
p6_s2_s1_boundary_plot.svg
```

---

## 17. P7：Geometry Pareto diagnostic

### 17.1 目的

几何不是硬目标。本阶段只判断 weak geometry regularizer 是否能改善 ECE/NLL/AUC 而不伤 macro。

### 17.2 Candidates

```text
GEO0-A2S-lambda0
GEO1-A2S-lambda1e-5
GEO2-A2S-lambda3e-5
GEO3-A2S-lambda1e-4
GEO4-A2S-lambda3e-4
GEO5-A2S-lambda1e-3

GEO6-best-package-lambda0
GEO7-best-package-lambda1e-5
GEO8-best-package-lambda1e-4
```

Geometry losses：

```text
basis_smoothness
edge_curvature
jacobian_norm
local_lipschitz
coefficient_tv
path_length
```

### 17.3 必须记录字段

```text
candidate
lambda_geo
geometry_loss_type
train_acc
val_acc
test_acc
val_gap_vs_MLP
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

### 17.4 判断标准

Useful geometry if：

$$
Acc_{\lambda}\geq Acc_{\lambda=0}-0.005
$$

and:

$$
ECE_{\lambda}<ECE_{\lambda=0}
$$

or:

$$
NLL_{\lambda}<NLL_{\lambda=0}
$$

or:

$$
ValLossAUC_{\lambda}<ValLossAUC_{\lambda=0}.
$$

Harmful geometry if：

$$
Acc_{\lambda}<Acc_{\lambda=0}-0.02
$$

and no ECE/NLL/AUC improvement.

Geometry must not decide main route unless it improves Pareto.

### 17.5 可视化

```text
p7_accuracy_vs_geometry_lambda.svg
p7_ece_vs_geometry_lambda.svg
p7_nll_vs_geometry_lambda.svg
p7_geometry_metric_vs_accuracy.svg
p7_geometry_pareto_frontier.svg
```

---

## 18. P8：10-seed task + AUC verification

### 18.1 目的

对最终 candidate 做 10-seed task verification。5-seed near-pass 不能作为 final success。

### 18.2 必跑对象

```text
MLP
C3 teacher
A2S supervised
best A2S distill candidate
best A2S bridge candidate
best A2S fused package
best new primitive if needed
```

### 18.3 设置

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

### 18.4 必须记录字段

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
CI95_low
CI95_high
Holm_p
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

### 18.5 判断标准

MacroSignificantPass：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200.
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

Calibration/NLL：

$$
ECE_{KAN}\leq ECE_{MLP}+0.005.
$$

$$
NLL_{KAN}\leq NLL_{MLP}+0.01.
$$

Time AUC：

$$
ValLossAUC_{\text{time,KAN}}\leq ValLossAUC_{\text{time,MLP}}.
$$

### 18.6 可视化

```text
p8_seedwise_val_gap_boxplot.svg
p8_bootstrap_ci.svg
p8_val_test_gap_scatter.svg
p8_val_loss_vs_time.svg
p8_val_loss_auc_time_bar.svg
p8_task_efficiency_pareto.svg
```

---

## 19. P9：Full-grid efficiency profiler

### 19.1 目的

验证 final candidate 是否真的 9/9 shapes 过 S2/S1。

### 19.2 Shape grid

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
  native
  2
  3

warmup / measure:
  50 / 200
```

### 19.3 必须记录字段

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
S2_pass
S1_pass
failure_reason_by_shape
```

### 19.4 判断标准

FullGridS2Pass：

```text
all 9 primary shapes satisfy S2
```

where:

$$
r_{mem}\leq1.05.
$$

$$
r_{step}\leq1.50.
$$

FullGridS1Pass：

$$
r_{mem}<1.00.
$$

$$
r_{step}\leq1.35.
$$

If only mean passes but shape fails, route remains diagnostic.

### 19.5 可视化

```text
p9_efficiency_pareto.svg
p9_batch_depth_heatmap_memory.svg
p9_batch_depth_heatmap_step.svg
p9_shape_failure_reason_heatmap.svg
p9_s2_s1_boundary_plot.svg
```

---

## 20. P10：candidate co-selection and route decision

### 20.1 Survivor 类型

```text
S0:
  StrictPass + GradPass + FullGridS1Pass + MacroSignificantPass + TimeAUCPass

S1:
  StrictPass + GradPass + FullGridS2Pass + MacroSignificantPass

S2:
  MacroSignificantPass but efficiency fail

S3:
  FullGridS2Pass but macro below +0.0200

S4:
  A2S distillation closes macro, but supervised does not

S5:
  bridge closes macro with acceptable overhead

S6:
  kernelization improves efficiency but hurts expression

S7:
  attribution incomplete

S8:
  gradient fail

S9:
  no improvement
```

### 20.2 必须记录字段

```text
candidate
survivor_type
strict_pass
grad_pass
macro_significant_pass
fullgrid_s2_pass
fullgrid_s1_pass
time_auc_pass
distillation_reachable
bridge_success
attribution_pass
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
val_gap_vs_MLP
test_gap_vs_MLP
CI95_low
Holm_p
ECE_delta
NLL_delta
ValLossAUC_time_delta
route_recommendation
primary_blocker
next_required_implementation
```

### 20.3 Route cases

```text
R1-PureKANNG-S1-Success:
  S1 + MacroSignificantPass + TimeAUCPass.
  Move to external/task scaling.

R2-PureKANNG-S2-DiagnosticSuccess:
  S2 + MacroSignificantPass.
  Continue S1 memory repair.

R3-OptimizationReachability:
  Distillation makes A2S pass macro without inference overhead.
  Need KAN-specific objective / training recipe.

R4-EffectiveExpressivityLoss:
  Distillation fails, and A2S remains below macro gate.
  Need minimal expression bridge or new primitive.

R5-KernelizationFail:
  Macro passes but S2 fails after fused package.
  Need lower-level fused kernel or different live-set design.

R6-AttributionIncomplete:
  materialized_tensor_count/top sources not measured.
  No kernel claim allowed.

R7-NoImprovement:
  Stop local patching, design new primitive.
```

### 20.4 可视化

```text
p10_expression_optimization_efficiency_pareto.svg
p10_survivor_type_dashboard.svg
p10_route_candidate_scorecard.svg
p10_decision_tree.svg
```

---

## 21. P11：artifact and failure audit

### 21.1 Artifacts

```text
run_manifest.json
provenance_audit.csv
candidate_registry.csv
gate_config.json
p0_reproduction.csv
p1_c3_a2s_source_of_loss.csv
p2_fullgrid_live_set_attribution.csv
p3_distillation_reachability.csv
p4_minimal_expression_bridge.csv
p5_microkernel_benchmark.csv
p6_fused_tile_stream_package.csv
p7_geometry_pareto.csv
p8_10seed_task_auc.csv
p9_fullgrid_efficiency.csv
p10_candidate_selection.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

### 21.2 Failure taxonomy

```text
F1_macro_task_fail
F2_fullgrid_memory_fail
F3_fullgrid_step_fail
F4_gradient_correctness_fail
F5_distillation_no_reachability
F6_effective_expressivity_loss
F7_bridge_too_expensive
F8_kernelization_hurts_expression
F9_attribution_incomplete
F10_geometry_hurts_expression
F11_fake_or_proxy_violation
F12_nonKAN_violation
F13_artifact_missing
```

### 21.3 总图

```text
figures/p10_expression_optimization_efficiency_pareto.svg
figures/p10_survivor_type_dashboard.svg
figures/p10_decision_tree.svg
figures/route_decision_dashboard.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 22. 第一轮推荐执行顺序

v7.5 第一轮不要全量铺开。优先做最能回答机制问题的实验。

### Step 1：P2 full-grid attribution

先补 v7.4 最大缺口：

```text
materialized_tensor_count
kernel_count
top memory source
top step source
bs512 memory fail source
worst-shape step fail source
```

没有 P2，就不允许声称 fused kernel 一定能解决。

### Step 2：P3 C3 teacher distillation

同时判别 A2S 是 optimizer/objective 问题还是结构表达力问题。这个实验成本比写 kernel 小，但决策价值很大。

### Step 3：P5 microkernel

只对 P2 找出的 top bottleneck 写 microkernel，不做盲目的大 package。

### Step 4：P4 minimal bridge

如果 P3 证明 A2S 结构不足，再做 minimal expression bridge。不要直接回退到 full C3。

### Step 5：P6 full package

只有 P5 microkernel 有效且 P3/P4 能过 macro，才进入 full package。

---

## 23. 停止条件

### 成功停止

出现以下任一条件，立即写复盘：

```text
FullGridS2Pass + MacroSignificantPass
FullGridS1Pass + MacroSignificantPass
FullGridS1Pass + MacroSignificantPass + TimeAUCPass
```

### 失败停止

出现以下任一情况，停止当前路线：

```text
1. P2 attribution 仍无法测 materialized_tensor_count / top source；
2. distillation 提升 <0.001，且 bridge 提升 <0.002；
3. fused package 后 A2S step worst-shape 仍 >1.80；
4. fused package 后 bs512 memory 仍 >1.08；
5. bridge overhead >0.02 memory 或 >0.10 step；
6. final route candidate GradPass fail；
7. no-fake/no-proxy audit fail。
```

---

## 24. 成功与失败解释规则

### Case A：A2S distillation passes macro and fused package passes S2

说明 A2S 结构大体足够，主要问题是 optimizer/objective 和 kernel path。下一轮重点是把 distillation 固化为 KAN-specific training recipe，并继续 S1 memory repair。

### Case B：A2S distillation fails but minimal bridge passes

说明 A2S 本身缺失少量表达力，C3 的优势可以通过低 overhead bridge 恢复。下一轮重点是 bridge kernelization。

### Case C：A2S distillation and bridge both fail

说明 low live-set bridge 结构不够，必须设计 new primitive。不要继续修 A2S。

### Case D：kernelization improves efficiency but macro drops

说明 fused/tile-stream 改变了有效函数或约束过强。需要 dense-preserving design，而不是继续压 memory。

### Case E：macro passes but shape S2 fails

说明 route 是 `kernelization_fail`，不是 task fail。下一轮继续 full-grid shape-specific repair。

### Case F：S2 passes but macro below +0.0200

说明 route 是 `effective_expression_loss`。下一轮回到 expression bridge 或 distillation。

---

## 25. 最终建议

v7.5 的一句话策略是：

$$
\boxed{
\text{以 A2S 为主 candidate，以 C3 为 teacher，先判别 reachability，再做 attribution-driven fused kernel。}
}
$$

当前不应该继续小修小补。最关键的不是再把某个指标调高一点，而是明确：

```text
1. A2S 与 C3 的表达力差异来自哪里；
2. C3 teacher 能不能把 A2S 教到 macro pass；
3. A2S full-grid S2 fail 的真实 memory/time source 是什么；
4. 哪个 microkernel 能真实降低 source；
5. 最小 expression bridge 是否能不增加 live-set 就恢复 C3 的优势。
```

如果这些实验闭合，项目就会从：

```text
C3 有表达力但太重；
A2S 轻但差一点；
```

推进到：

```text
A2S/C3 bridge 既保表达力，又 full-grid S2。
```

这才是真正的 PureKAN-NG kernel-native Beyond-MLP 路线。
