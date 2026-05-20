# DG-KAN v12.5.2：效率路线、流形-信号通道几何诊断与 Functional Update 三线完整计划

> 本文件整合 `DG-KAN_v12.5_效率与Functional双线_实验结果分析与下一步计划.md` 与 `DG-KAN_v12.5.1_TrainTestCouplingGeometryDiagnostics_补丁计划.md`。
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。
> 原则：不降低 gate，不按数据集调参，不用 teacher / distillation / loss modification / sampler / class weight，不把 diagnostic 写成 official success。

---

## 0. 一句话结论

v12.4 不是完全没进展，但仍不是科学成功。它的真实进展是：

$$
\boxed{
\text{多基函数路线已经把候选从“广泛扫描”收敛到 Gated Legendre + Quadratic 一族，}
}
$$

并且：

$$
\boxed{
\text{B47 证明 manual backward / manual CE step 是可行的，}
\text{但 Python-level manual backward 不能解决效率门。}
}
$$

当前真正 blocker 已经不是单纯 autograd backward，也不是单纯表达力，而是：

$$
\boxed{
\text{strict PureKAN basis primitive 还没有同时满足 expression、task stability、MLP-like efficiency。}
}
$$

Functional update 仍然要继续并行验证，但只能以 diagnostic 形式进行。正式 functional success 必须等 base gate 打开后才能成立。同时，Train-Probe Coupling / Signal-Reservoir / Noise Leakage 不能再藏在 functional 附属指标里，而必须成为独立 Line C。

---

## 1. 当前实验结果独立分析

## 1.1 这次有进展吗？

有，但不是最终能力进展，而是边界变清楚了。

v12.4 完成了以下真实推进：

1. 从 v12.3 的 LQ frame family 小修，转向 primitive-level multi-basis screen。
2. ReLU/RSWAF hinge、RBF/FastKAN、Chebyshev、Fourier、Wavelet、B-spline、Rational 等基函数已做过一轮效率优先扫描。
3. Poly2 / signed-pair 证明局部 pairwise coverage 有价值，但无法覆盖 rotated/random quadratic 与 task stability。
4. B14/B21/B22/B23/B27/B33/B36/B42/B45 证明 **Gated Legendre + Quadratic** 是当前最接近的 strict FC-PureKAN primitive family。
5. B42 fast-reuse forward path 做到了数值等价，说明部分工程路径可压缩。
6. B47 manual backward / manual CE step 证明了“绕开 torch autograd graph”的方向是可行的。
7. Functional diagnostic 有 positive rows，但 base gate 关闭，因此不能 official。

所以，本轮不是“又原地失败”，而是把问题从：

```text
到底该不该继续多基函数？
```

推进成：

```text
现在只剩少数有希望的 basis family，
但必须进入 lower-level CUDA/Triton fused forward+backward 才能继续。
```

## 1.2 什么没有达成？

没有任何 candidate 同时通过：

```text
A1 efficiency
A2 expression
A3 task/base
```

所以当前仍然是：

```text
base_qualified = false
functional_open = false
functional_diagnostic_positive = false in latest route
```

这意味着不能声明：

```text
PureKAN base 成功；
Functional update 成功；
DG-KAN 已经优于 MLP；
可以进入 external fair；
可以打开 Conv / Former。
```

当前最好只能说：

```text
Gated Legendre + Quadratic 是目前最接近的 primitive family；
B47 证明 manual path 可做，但真正效率需要 fused kernel；
functional update 仍是并行 diagnostic。
```

---

## 2. 关键候选的真实含义

## 2.1 B21d：表达力最强，但 task 没过

B21d 的价值很高。它说明 basis-specific residual quadratic norm 和 branch-specific normalization 确实能修 expression：

```text
E1 / E2 / E6 / E8 delta 接近 0 或略正
frozen R2 全部超过 0.89
A2 expression pass
```

这说明 Legendre + quadratic sketch 不是“表达力不够”的低级失败。它可以在 synthetic expression 上接近 MLP 或超过 MLP。

但 B21d 没有通过 A3。原因是 task stability、AUC-time、ECE hard gates 没有一起过。

独立判断：

$$
\boxed{
\text{B21d 是 expression anchor，不是 base survivor。}
}
$$

它应该保留为后续 fused kernel 的 expression reference。

## 2.2 B23b：task accuracy 很强，但 AUC-time / ECE max 失败

B23b low-temperature repair 把 task mean delta 推到了约 $+0.00998$，near pass 也到 $0.8889$。这说明：

```text
logit temperature / branch scale 对 early confidence 和 task accuracy 有真实作用。
```

但 B24 compiled task audit 里 AUC-time 出现极大异常，后续 B25 steady-state accounting 证明 B24 的 $224x$ 到 $280x$ 是 compile/warmup accounting artifact。修正后仍未通过 near pass、AUC max、ECE hard gates。

独立判断：

$$
\boxed{
\text{B23b 证明低温初始化能改善 task/ECE，}
\text{但它没有解决 worst-row stability 和 true AUC-time。}
}
$$

后续不能只继续调 temperature。

## 2.3 B27c：schedule 有帮助，但不是解决方案

B27c 通过 A2，AUC-time mean 降到约 $1.02$，ECE mean delta 也明显变好，但：

```text
worst delta 仍差；
near pass rate 低；
AUC-time max 仍高。
```

独立判断：

$$
\boxed{
\text{schedule repair 可以改善平均轨迹，}
\text{但不能解决最差 split 和 task stability。}
}
$$

这说明不能把问题降级为“再调 schedule”。

## 2.4 B33c / B36c：最接近 task，但 AUC-time 和 ECE 不闭合

B33c / B36c 都出现过 task mean delta $+0.0052$ 这种接近成功的信号，并且 A1+A2 可通过。但 near pass、AUC-time、ECE 没有同时满足。

独立判断：

$$
\boxed{
\text{Gated Legendre + Quadratic 已经进入 near-survivor 区域，}
\text{但它的训练轨迹不稳定。}
}
$$

这更像是 coordinate / branch scale / kernel path 与 optimizer dynamics 的联合问题，而不是单个超参问题。

## 2.5 B42/B45：fast-reuse 是工程进展，不是 base success

B42 fast-reuse forward equivalence max abs error 为 $0$，这很重要。它说明部分 forward 可以被重写成等价低成本路径。B45 又进一步改善了 ECE mean delta。

但它们都没有通过 A3：

```text
B42b task mean delta 约 -0.00933，near pass 约 0.2222；
B45a ECE mean 改善，但 task mean delta 仍约 -0.00933，near pass 约 0.3333。
```

独立判断：

$$
\boxed{
\text{fast-reuse 可以降低工程冗余，}
\text{但当前 directskip / GatedHybrid family 的 task geometry 仍不稳定。}
}
$$

## 2.6 B47：最重要的系统诊断

B47 是本轮最关键的系统诊断，不应该被简单看成失败。

B47 做了三件正确的事：

```text
1. 新增 manual forward / manual backward；
2. 绕开 torch autograd graph；
3. 梯度 correctness 基本通过。
```

梯度正确性：

```text
grad relerr max ≈ 0.000177
grad cos min ≈ 0.99999988
output max abs error ≈ 8.94e-08
```

Manual CE step 也确实降低了 backward 和 total step：

```text
custom autograd eager step ratio:
  4.13x - 5.81x

manual CE step ratio:
  2.62x - 2.79x
```

但 B47 仍失败，因为 forward ratio 太高：

```text
B47b forward ratio ≈ 6.80x
B47b backward ratio ≈ 1.87x
B47b step ratio ≈ 2.62x
B47b memory ratio ≈ 1.12x
```

这说明：

$$
\boxed{
\text{manual backward 已经不是唯一问题；}
\text{forward basis/norm/projection 的多算子组合和 kernel launch 才是当前主 blocker。}
}
$$

因此，下一步不能再只写 Python-level manual backward。必须做 lower-level fused CUDA/Triton forward+backward kernel。

---

## 3. 当前卡在哪里？

当前 blocker 可以分成四层。

## 3.1 系统层 blocker：forward / kernel launch / 多算子组合

B47 的 forward ratio 约 $6.8x$，而 backward ratio 已经压到约 $1.87x$。这说明最耗时的已经不是“反传公式不会写”，而是：

```text
basis-specific input norm；
Legendre K=4 recurrence；
quadratic sketch；
direct edge-basis skip；
gating / branch scale；
projection / readout；
Python/Torch op 边界；
临时 tensor materialization；
kernel launch fragmentation。
```

当前 torch-level / Python-level 组合已经到头。要继续效率路线，必须进入：

```text
Triton fused forward kernel；
Triton fused backward kernel；
single-call full-layer kernel；
CUDA graph fixed-shape capture；
workspace/lifetime planning；
basis materialization elimination。
```

## 3.2 表达-任务-效率三目标冲突

当前每个 candidate 都只解决一部分：

| family | 已解决 | 没解决 |
|---|---|---|
| B21d | expression | task/AUC/ECE |
| B23b | task mean / ECE mean | AUC-time / ECE max / stability |
| B27c | AUC mean / ECE mean | worst row / near pass |
| B33/B36 | A1+A2 + task mean signal | AUC max / ECE / near pass |
| B42/B45 | forward equivalence / fast reuse | task |
| B47 | manual backward correctness | forward/step efficiency |

这说明真正目标不是“把某个 gate 调松一点”，而是找到：

$$
\boxed{
\text{一个表达力、训练稳定性、系统效率共同闭合的 primitive。}
}
$$

## 3.3 Functional update blocker：base 未过 + control resistance 未过

Functional diagnostic 继续有价值，但不能 official。原因有两个：

1. base gate 没打开，functional 只能 diagnostic；
2. positive rows 没有形成 strong-control-resistant evidence。

也就是说，即使某些 row 上 functional control gap 为正，也不能证明：

$$
FunctionalUpdate > AdamWParallelDirection
$$

或者：

$$
FunctionalUpdate > RandomMatchedNorm
$$

更不能证明它能带来独立几何优势。

## 3.4 数据集问题只能做诊断，不能做调参依据

MNIST / Fashion-MNIST / KMNIST 暴露的 failure 不一样，这很有用。但不能引入：

```text
dataset_name branch；
dataset-specific threshold；
dataset-specific schedule；
dataset-specific norm；
dataset-specific temperature。
```

不同数据集只能用于 failure slice：

```text
哪类 hard direction 不稳定；
哪个 split AUC max 爆；
哪个 class tail CEp99 高；
哪种 basis coverage 不够。
```

官方候选必须保持 dataset-agnostic。

---

## 4. 现在是否在正确道路上？

我的判断是：

$$
\boxed{
\text{方向是对的，但推进方式必须升级。}
}
$$

对的地方：

```text
1. 没有降低 gate；
2. 没有把 diagnostic 写成 success；
3. 没有按 dataset 调参；
4. 已从 LQ frame 小修转到 multi-basis primitive；
5. 已收敛到 Gated Legendre + Quadratic 这个较强 family；
6. 已通过 B47 证明 manual path 可行；
7. 继续保留 functional diagnostic，但没有强行 official。
```

不对或需要停止的地方：

```text
1. 不能继续 B42/B45/B46 的 schedule/scale 小网格；
2. 不能继续 Python-level manual backward 作为效率主修；
3. 不能只靠 torch.compile / warmup accounting；
4. 不能再广泛平均扫所有 basis family；
5. 不能在 base 未过时让 functional short-run 进入 official。
```

因此，v12.5.2 应该明确升级为：

$$
\boxed{
\text{Fused Kernel First + Functional Diagnostic Parallel。}
}
$$

---

## 5. 离目标还差多远？

按模块拆分：

| 模块 | 当前状态 | 离目标 |
|---|---|---|
| 表达力 | Gated Legendre + Quadratic 已接近；B21d 很强 | 中等距离 |
| task mean | B23/B33/B36 有正信号 | 接近但不稳定 |
| worst-row / near pass | 仍失败 | 较远 |
| AUC-time | 平均可接近，max 仍爆 | 较远 |
| ECE | mean 可改善，max 仍失败 | 中等 |
| forward efficiency | B47 forward ratio 约 $6.8x$ | 很远 |
| backward efficiency | B47 backward ratio 约 $1.87x$ | 中等偏远 |
| step efficiency | B47 step ratio 约 $2.62x$ | 较远 |
| memory | B47 memory ratio 约 $1.12x$ | 中等 |
| functional update | diagnostic only，不 beat strong controls | 很远 |
| next-gen MLP claim | 不能 claim | 很远 |

一句话：

$$
\boxed{
\text{科学上还没成功；工程上已经定位到必须 lower-level kernel。}
}
$$

---

---

# 6. v12.5.2 总实验目标

v12.5.2 不再做“小修小补”。它把 v12.5 与 v12.5.1 整合成一个三线并行计划：

$$
\boxed{
\text{Line A 做出 MLP-like strict PureKAN base；}
\text{Line C 判断训练几何是否健康；}
\text{Line B 验证 functional update 是否有独立贡献。}
}
$$

三条线的最终关系是：

```text
Line A:
  fused CUDA/Triton efficiency-first base qualification。

Line C:
  manifold-channel geometry diagnostics。
  测 train-probe coupling、signal channel、reservoir、noise leakage。

Line B:
  functional update diagnostic / official gated re-entry。
```

v12.5.2 必须明确回答四个问题：

```text
1. Gated Legendre + Quadratic 或更简单 primitive 是否能通过 lower-level fusion 进入 MLP-like envelope？
2. 如果不能，是 forward basis 复杂度不可接受，还是实现没有 kernel-native？
3. 一个 near-pass base 是否真的有健康的 train-test/probe coupling，而不是用坏几何换任务？
4. Functional update 是否能在 Line C 指标上击败 AdamWParallel / Random / SNR-only / MLPAnalog controls？
```

本轮不要求马上得到 final 10-seed success，但必须把“效率、几何、functional 因果性”拆开测清楚。

---

# 7. 三线总览

## Line A：效率 / Base Primitive 路线

目标：

$$
\boxed{
\text{做出一个 A1 + A2 + A3 同时过的 strict PureKAN base。}
}
$$

优先候选：

```text
A-family-1:
  Gated Legendre + Quadratic, B21/B33/B36/B42/B45/B47 lineage。

A-family-2:
  LiteGated Legendre direct + quadratic sketch, B30/B31 lineage。

A-family-3:
  SimpleFastTaskGeometry:
    Activation / RSWAF / hinge-like edge basis + minimal quadratic interaction。
```

Line A 的核心不是再扫 50 个 config，而是对 top family 做 kernel-native implementation，必要时使用 lower-level CUDA/Triton fused forward+backward kernel。

## Line C：Manifold-Channel Geometry Diagnostics 路线

目标：

$$
\boxed{
\text{判断一个 base 或 functional update 的训练运动是否 test/probe-visible、}
\text{真实信号是否进入 signal channel、噪声是否泄漏进 signal channel。}
}
$$

Line C 是独立主线，不是 functional 的附属指标。它服务两个 gate：

```text
A -> C:
  A 线候选即使 task/efficiency near-pass，也必须没有明显 coupling collapse / noise leakage / reservoir trapping。

B -> C:
  B 线 functional 必须在 Line C 指标上改善 base，并击败 strong controls。
```

## Line B：Functional Update 路线

目标：

$$
\boxed{
\text{并行验证 functional update 是否有 control-resistant signal-channel geometry advantage。}
}
$$

Line B 不等待 final base，但它的输出必须标注：

```text
diagnostic_base_not_qualified
```

只有 Line A base qualified 且 Line C 没有明显几何崩坏后，Line B 才能转 official。

---

# 8. Line A：效率 / Base Primitive 详细实验计划

## A0：合同锁定与候选冻结

### 目标

冻结 v12.5.2 不再无限扩张候选。只允许三类候选进入：

```text
K1: GatedLQFull
  Gated Legendre K=4 + quadratic sketch + direct edge-basis skip

K2: LiteGatedLQ
  direct Legendre readout + low-cost quadratic sketch

K3: SimpleFastTaskGeometry
  RSWAF / hinge / low-degree orthogonal + minimal interaction
```

### 必须记录

```text
candidate_id
family
basis_type
hidden_dim
basis_order
quadratic_rank
directskip_scale
branch_scale
input_norm_type
temperature_init
nonKAN_param_count
edge_param_count
manual_forward_available
manual_backward_available
uses_loss_backward
uses_torch_autograd_graph
diagnostic_only
official_capable
```

### 通过标准

```text
nonKAN_param_count = 0
official_capable = 1
uses_loss_backward = 0 for manual/fused candidates
no dataset_name branch
no teacher / no loss modification
```

### 不满足时 Codex 先尝试

如果某候选 official_capable 失败：

```text
1. 检查 learnable norm / ordinary Linear / MLP shortcut 是否混入；
2. 把固定统计 norm 标记为 fixed buffer，不作为 learnable non-KAN；
3. 若 directskip 是 ordinary MLP hidden path，移除或改成 edge-owned basis readout；
4. 仍无法 strict，则 candidate 降级 diagnostic。
```

---

## A1：Fused forward truth audit

### 假设 H-A1

B47 的 forward ratio 约 $6.8x$ 不是数学复杂度必然，而是 torch/Python 多算子和 launch fragmentation 导致。若把 input norm、Legendre recurrence、quadratic sketch、directskip projection 合成少量 fused kernel，forward ratio 可以显著下降。

### 实验对象

```text
F0:
  current B47 manual CE path

F1:
  fused Legendre basis forward only

F2:
  fused input norm + Legendre basis forward

F3:
  fused input norm + Legendre + quadratic sketch forward

F4:
  fused input norm + Legendre + quadratic + directskip readout prep

F5:
  full-layer fused forward, output buffer directly consumed by projection
```

### 必须记录

```text
candidate_id
kernel_impl
forward_ratio
forward_ms_q50
forward_ms_q90
kernel_count_forward
custom_kernel_count_forward
torch_op_count_forward
allocation_count_forward
temp_bytes_forward
materialized_basis_bytes
materialized_quad_bytes
max_abs_forward_error_vs_reference
forward_cos_vs_reference
```

### 判断标准

探索通过：

$$
T_{forward,new} \le 2.0T_{forward,MLP}
$$

正式进入下一阶段：

$$
T_{forward,new} \le 1.50T_{forward,MLP}
$$

目标：

$$
T_{forward,new} \le 1.25T_{forward,MLP}
$$

正确性：

$$
\|f_{new}-f_{ref}\|_{\infty}\le 10^{-6}
$$

或：

$$
\operatorname{cos}(f_{new},f_{ref})\ge 0.999999.
$$

### 可视化

```text
fig_A1_forward_ratio_bar.svg
fig_A1_forward_time_waterfall.svg
fig_A1_kernel_count_vs_forward_time.svg
fig_A1_materialized_bytes_bar.svg
fig_A1_forward_error_hist.svg
```

### 不满足时 Codex 先尝试

若 forward ratio 仍 $>2.0$：

```text
1. 统计 kernel_count_forward 和 allocation_count_forward；
2. 若 kernel_count 高，优先 single-call fused forward；
3. 若 materialized_basis_bytes 高，改为 basis on-the-fly + immediate consume；
4. 若 Legendre recurrence 慢，改成 closed recurrence unroll K=4；
5. 若 quadratic sketch 慢，预计算 projection layout，避免 einsum；
6. 若 input norm 慢，合并 norm 到 basis kernel；
7. 若 full K1 仍慢，切到 K2 LiteGatedLQ。
```

若 forward correctness 失败：

```text
1. 用 fp64 reference 检查 Legendre recurrence；
2. 单独验证 input norm；
3. 单独验证 quadratic sketch；
4. 单独验证 directskip；
5. 按 F1 -> F5 二分定位误差。
```

---

## A2：Fused backward / analytic adjoint

### 假设 H-A2

B47 证明 parameter-gradient manual backward 可行，但 backward ratio 仍约 $1.87x$，memory ratio 约 $1.12x$。通过 fused backward、two-stage reduction、workspace reuse，可以把 backward ratio 降到 $1.5x$ 以内，并把 memory ratio 控制到 $1.05x$ 以内。

### 实验对象

```text
B0:
  B47 manual CE current

B1:
  fused param-gradient only

B2:
  fused param-gradient + no materialized basis

B3:
  fused dx + param-gradient full layer

B4:
  fused two-stage reduction for basis/branch params

B5:
  fused backward + update-prep workspace reuse

B6:
  CUDA graph captured forward+backward fixed shape
```

### 必须记录

```text
candidate_id
backward_impl
grad_relerr_max
grad_cos_min
param_grad_linf
input_grad_relerr_max
input_grad_cos_min
backward_ratio
backward_ms_q90
step_ratio
memory_ratio
kernel_count_backward
allocation_count_backward
workspace_temp_mb
basis_recomputed
basis_materialized
atomic_add_count_proxy
reduction_strategy
```

### 判断标准

正确性：

$$
\operatorname{grad\_relerr}_{max}<10^{-4}
$$

$$
\operatorname{grad\_cos}_{min}>0.999.
$$

效率探索门：

$$
T_{backward}/T_{MLP}\le1.75
$$

$$
T_{step}/T_{MLP}\le1.75
$$

正式门：

$$
T_{backward}/T_{MLP}\le1.50
$$

$$
T_{step}/T_{MLP}\le1.50
$$

目标门：

$$
T_{step}/T_{MLP}\le1.25.
$$

Memory：

$$
M_{peak}/M_{MLP}\le1.05
$$

目标：

$$
M_{peak}/M_{MLP}\le1.00.
$$

### 可视化

```text
fig_A2_backward_ratio_bar.svg
fig_A2_grad_correctness_scatter.svg
fig_A2_memory_workspace_waterfall.svg
fig_A2_reduction_strategy_pareto.svg
fig_A2_step_ratio_vs_memory_ratio.svg
```

### 不满足时 Codex 先尝试

若 grad correctness fail：

```text
1. 先只测 param-gradient，不测 input dx；
2. 对 Legendre branch、quadratic branch、directskip branch 分别 gradcheck；
3. 检查 norm scale 的 backward 是否漏乘；
4. 检查 batch reduction 与 output-class reduction 维度；
5. 检查 contiguous layout 和 stride；
6. 若 relerr 在 1e-4 附近，记录 fp32/fp64 对比，不手动改 pass。
```

若 backward ratio 仍高：

```text
1. 看 reduction 是否使用 atomic；
2. 改为 block-level local reduction + second-pass reduction；
3. 合并 branch param gradients；
4. 减少 separate kernel calls；
5. 尝试 persistent CTA per output tile 或 per hidden tile；
6. 若 dx 不影响当前 architecture，可先测 no-dx diagnostic，但 official full-layer 需要 dx。
```

若 memory fail：

```text
1. 禁止 materialized basis；
2. 禁止 materialized quadratic tensor；
3. forward cache 只保留 x / norm stats / compact metadata；
4. backward recompute basis；
5. workspace 使用 preallocated buffer；
6. 记录 actual CUDA peak，而不是只记录 manual cache。
```

---

## A3：Combined kernel gate

### 目标

把 A1/A2 中最好的 forward 与 backward 组合，测 full training step。

### 必须记录

```text
candidate_id
forward_impl
backward_impl
update_impl
manual_ce_step
uses_torch_autograd_graph
forward_ratio_q90
backward_ratio_q90
optimizer_ratio_q90
step_ratio_q90
memory_ratio_q90
kernel_count_total
allocation_count_total
samples_per_second
compile_warmup_steps
steady_state_accounting_used
```

### 通过标准

探索：

$$
T_{step,q90}/T_{MLP,q90}\le1.50
$$

$$
M_{q90}/M_{MLP,q90}\le1.10.
$$

official base efficiency：

$$
T_{step,q90}/T_{MLP,q90}\le1.25
$$

$$
M_{q90}/M_{MLP,q90}\le1.05.
$$

如果 target 是 first scientific re-entry，可以接受：

$$
T_{step,q90}/T_{MLP,q90}\le1.50
$$

但必须在报告中写成 exploratory，不可称 final MLP-like。

### 可视化

```text
fig_A3_full_step_waterfall.svg
fig_A3_step_ratio_distribution.svg
fig_A3_memory_ratio_trace.svg
fig_A3_kernel_count_trace.svg
```

### 不满足时 Codex 先尝试

若 forward 已过、backward 没过：

```text
继续 A2 reduction/workspace，不做 task。
```

若 backward 已过、forward 没过：

```text
继续 A1 full-layer fused forward，不做 task。
```

若二者单独过、combined 不过：

```text
检查 boundary copy / layout transform / CUDA sync；
实现 single-call full-layer forward+backward；
尝试 CUDA graph capture；
检查 optimizer update 是否成为新瓶颈。
```

---

## A4：Expression qualification

### 目标

确保 fused kernel 没有破坏表达力。这个阶段不看 dataset task success，只看 expression。

### 必须测试

```text
E0 additive
E1 pairwise product
E2 composition
E3 local-XOR
E4 high-frequency
E5 noise-stress
E6 rotated pairwise
E8 random quadratic
global matrix-span R2
```

### 必须记录

```text
target_id
candidate_id
trainable_R2
frozen_readout_R2
delta_vs_MLP
delta_vs_B21d
global_matrix_span_R2
effective_rank
basis_usage_entropy
dead_basis_fraction
branch_energy_legendre
branch_energy_quadratic
branch_energy_directskip
```

### 通过标准

A2 expression pass：

$$
\Delta R^2_{E1,E2,E6,E8} \ge -0.01
$$

或者至少：

$$
R^2_{frozen,E1,E2,E6,E8}\ge0.89
$$

并且：

$$
\text{dead\_basis\_fraction}\le0.30.
$$

### 不满足时 Codex 先尝试

若 E1/E6/E8 失败：

```text
1. 恢复 B21d 的 branch-specific norm / residual quadratic norm；
2. 检查 quadratic projection scale；
3. 增加 quadratic rank，但必须先估算 efficiency；
4. 若 LiteGated fails expression，回到 full GatedLQ；
5. 不用 dataset task 结果修 expression。
```

若 expression pass 但 effective rank 低：

```text
1. 调整 branch scale；
2. 加 orthogonal init；
3. 减少 directskip 初始占比；
4. 检查 feature-scale calibration。
```

---

## A5：5-epoch / short-budget task triage

### 目标

只让 A1+A2 过的 candidate 进入 task triage。task triage 只判断是否值得长跑，不做 final claim。

### 数据设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
train_size = 1024 or 2048
val_size = 512
test_size = 512
budget = 5 epochs and optional 10 epochs
```

### 对照

```text
MLP-same-param-AdamW
MLP-same-step-or-same-FLOPs
QuadraticFeatureMLP-diagnostic
B21d-expression-anchor
B23b-task-anchor
B36c-near-task-anchor
new fused K1/K2/K3 candidates
```

### 必须记录

任务：

```text
val_acc
test_acc
mean_delta_vs_mlp
worst_delta_vs_mlp
near_pass_rate
val_loss_auc_step
val_loss_auc_time_mean
val_loss_auc_time_max
time_to_target_loss
time_to_target_acc
```

校准 / tail：

```text
ECE_mean_delta
ECE_max_delta
NLL_delta
CEp99_delta
margin_p10_delta
wrong_confidence_p95
```

稳定性：

```text
loss_spike_count
bad_step_rate
gradient_spike_p95
update_over_param_norm
branch_scale_trace
branch_energy_trace
```

效率：

```text
step_ratio_q90
memory_ratio_q90
samples_per_sec
steady_state_epoch_accounting
```

### 通过标准

A3 exploratory pass：

$$
\Delta Acc_{mean}\ge-0.005
$$

$$
\Delta Acc_{worst}\ge-0.02
$$

$$
near\_pass\_rate\ge0.80
$$

$$
AUCtime_{mean}\le1.10
$$

$$
AUCtime_{max}\le1.25
$$

$$
ECE_{max\_delta}\le0.02.
$$

A3 strong pass：

$$
\Delta Acc_{mean}\ge0
$$

$$
\Delta Acc_{worst}\ge-0.01
$$

$$
near\_pass\_rate\ge0.889
$$

$$
AUCtime_{mean}\le1.05
$$

$$
AUCtime_{max}\le1.10
$$

$$
ECE_{mean}\le ECE_{MLP}
$$

并且：

$$
T_{step}/T_{MLP}\le1.25.
$$

### 不满足时 Codex 先尝试

若 task mean 好但 worst row 差：

```text
1. 不按 dataset 调参；
2. 记录 failure slice；
3. 检查 branch energy 是否 seed-dependent；
4. 调整全局 initialization / branch scale；
5. 尝试 global warmup-cosine / gradient clipping；
6. 若仍不稳，降低 directskip 或 quadratic初始占比。
```

若 AUC-time fail 但 accuracy pass：

```text
1. 检查 steady-state timing；
2. 检查 loss curve early spike；
3. 使用全局 schedule，不按 dataset；
4. 尝试 smaller initial logit gain；
5. 若 AUC max 由单个 split outlier 导致，保留 outlier 分析但不特调。
```

若 ECE fail：

```text
1. 调低全局 logit temperature；
2. 记录 ECE vs margin；
3. 检查 overconfidence tail；
4. 不使用 calibration loss，不使用 label smoothing。
```

---

## A6：20/30-epoch confirmation

### 目标

只有 A5 pass 的 candidate 进入。确认短程不是偶然。

### 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
budget = 20 or 30 epochs
```

### 必须记录

全部 A5 指标，加：

```text
best_epoch
plateau_epoch
generalization_gap
calibration_curve_bins
confusion_matrix
classwise_accuracy
classwise_ECE
time_to_target_curve
```

### 通过标准

Base qualification：

$$
base\_qualified =
A1\_efficiency\_pass
\land A2\_expression\_pass
\land A3\_task\_pass.
$$

只有此时：

```text
functional_open = candidate for official one-step / short-run
```

### 不满足时 Codex 先尝试

若 5-epoch pass 但 20/30 fail：

```text
1. 判断是 late overfit、late instability 还是 underfit；
2. 若 late overfit，进入 signal-channel/noise leak diagnostic；
3. 若 late instability，降低 branch scale 或 add global schedule damping；
4. 若 underfit，增加 capacity 但必须先通过 A1 microbench。
```

---

---

# 9. Line C：Manifold-Channel Geometry Diagnostics / 流形-信号通道几何诊断

## C0：为什么需要单独提升为 Line C

v12.5 原计划有两条主线：

```text
Line A: 多基函数 / fused CUDA-Triton efficiency-first base qualification。
Line B: functional update diagnostic / official gated re-entry。
```

这还不够。因为我们的核心研究目标不是只做一个更快 KAN，也不是只找一个 functional update，而是证明：

$$
\boxed{\text{KAN base 提供可训练表达坐标，functional update 改善训练几何。}}
$$

这里的“几何好”不能只用 curvature、condition、basis entropy 或 kernel drift 表示。必须测训练运动是否能传到 probe/test-visible 方向：

$$
\boxed{\text{好几何 = train motion 更可预测 probe motion，真实信号进入 signal channel，噪声不泄漏进 signal channel。}}
$$

因此 v12.5.2 中三条线并行：

```text
Line A: Efficiency / primitive / fused kernel line。
Line B: Functional update diagnostic line。
Line C: Manifold-channel geometry diagnostics line。
```

Line C 不替代 A/B。它是 A/B 的几何测量与 gate：

```text
A 线候选必须通过 Line C 的被动诊断，才能说 base 几何有潜力。
B 线 functional 必须在 Line C 上击败 controls，才能说 functional 有独立几何贡献。
```

---

## C1：Line C 的总目标

Line C 要回答三个问题：

```text
C-Q1: 一个 basis primitive 是否让真实任务信号更容易进入 test-visible signal channel？
C-Q2: 一个 functional update 是否让 train motion 更稳定地预测 probe motion？
C-Q3: 噪声是否被 functional update 推进了 signal channel，从而增加过拟合风险？
```

这三件事都不能通过 dataset-specific tuning 解决。MNIST / Fashion-MNIST / KMNIST 只能作为 failure slice，不能作为 official controller 条件。

---

## C2：核心定义

从 train stream 中拆两个 batch：

```text
B = update batch
Q = probe batch
```

在窗口 $[t,t+\Delta]$ 内记录 logits 位移：

$$
\Delta U_B=f_{\theta_{t+\Delta}}(B)-f_{\theta_t}(B)
$$

$$
\Delta U_Q=f_{\theta_{t+\Delta}}(Q)-f_{\theta_t}(Q).
$$

拟合 ridge predictor：

$$
A_t=\arg\min_A \|\Delta U_Q-A\Delta U_B\|_F^2+\lambda\|A\|_F^2.
$$

Train-probe coupling：

$$
CouplingR^2=
1-
\frac{\|\Delta U_Q-A_t\Delta U_B\|_F^2}
{\|\Delta U_Q\|_F^2+\epsilon}.
$$

$$
CouplingCorr=
\operatorname{corr}(\operatorname{vec}(A_t\Delta U_B),\operatorname{vec}(\Delta U_Q)).
$$

对 batch 样本构造 projected gradient / logit-Jacobian sketch：

$$
\hat K_{BB}=\Phi\Phi^\top.
$$

窗口累积：

$$
\hat W_B=\sum_{\tau=t}^{t+\Delta}\hat K_{BB}(\tau).
$$

由 $\hat W_B$ 的谱分解得到 signal projector $P_{sig}$ 与 reservoir projector $P_{res}$。

真实信号困在 reservoir 的比例：

$$
RealSignalReservoirRatio=
\frac{\|P_{res}r_{real}\|^2}{\|r_{real}\|^2+\epsilon}.
$$

噪声泄漏进 signal channel 的比例：

$$
NoiseSignalLeak=
\frac{\|P_{sig}r_{noise}\|^2}{\|r_{noise}\|^2+\epsilon}.
$$

Kernel drift 只作为配套指标，不作为越小越好的目标：

$$
KernelDrift=
\frac{\|\hat K(t)-\hat K(0)\|_F}{\|\hat K(0)\|_F+\epsilon}.
$$

解释规则：

```text
good feature learning:
  KernelDrift 高，但 CouplingR2 高、NoiseSignalLeak 低、tail 不坏。

bad tearing:
  KernelDrift 高，CouplingR2 低、NoiseSignalLeak 高、CEp99 / ECE 变坏。

lazy underfit:
  KernelDrift 低，但 RealSignalReservoirRatio 高、task 不下降。
```

---

## C3：Line C 必须记录的 CSV 字段

### C3.1 `v125_train_probe_coupling.csv`

```text
run_id
candidate_id
basis_family
method
update_type
control_id
dataset
seed
step
window_size
batch_size_B
batch_size_Q
ridge_lambda
train_logit_drift_l2
probe_logit_drift_l2
CouplingR2
CouplingCorr
coupling_residual_norm
coupling_stability_across_splits
KernelDrift
CEp99_delta
ECE_delta
margin_p10_delta
official_gate_open
```

### C3.2 `v125_signal_reservoir_sketch.csv`

```text
run_id
candidate_id
basis_family
method
update_type
control_id
dataset
seed
step
window_size
sketch_dim
signal_effective_rank
signal_mass_topk
reservoir_fraction
top_eigen_share
dissipation_condition
RealSignalReservoirRatio
NoiseSignalLeak
SNR_positive_fraction
real_noise_gap
official_gate_open
```

### C3.3 `v125_manifold_channel_diagnostics.csv`

```text
run_id
candidate_id
basis_family
method
dataset
seed
step
base_qualified
functional_official_open
CouplingR2
CouplingCorr
RealSignalReservoirRatio
NoiseSignalLeak
KernelDrift
effective_rank_hidden
basis_occupancy_entropy
basis_dead_fraction
lift_condition_proxy
perturb_logit_drift_p95
CEp99
ECE
NLL
step_time_ratio
memory_ratio
```

---

## C4：判断标准

### C4.1 Base candidate 的 Line C 被动诊断标准

Base candidate 不是必须在 Line C 上全面赢 MLP 才能进入下一步，但必须不能出现明显 tearing：

$$
CouplingR^2_{KAN}\ge CouplingR^2_{MLP}-0.02.
$$

$$
NoiseSignalLeak_{KAN}\le NoiseSignalLeak_{MLP}+0.02.
$$

$$
CEp99_{KAN}\le CEp99_{MLP}+\epsilon_{tail}.
$$

如果 base 在 task/efficiency 上 near-pass，但 Line C 显示：

```text
CouplingR2 大幅低于 MLP；
NoiseSignalLeak 高；
RealSignalReservoirRatio 高；
CEp99 / ECE 变坏；
```

则该 candidate 不能称为 good-geometry base，只能称为 task/efficiency near-pass candidate。

### C4.2 Functional candidate 的 Line C official 标准

Functional candidate 必须相对同一 base 改善：

$$
CouplingR^2_{func}\ge CouplingR^2_{base}+\delta_c.
$$

$$
RealSignalReservoirRatio_{func}\le RealSignalReservoirRatio_{base}-\delta_r.
$$

$$
NoiseSignalLeak_{func}\le NoiseSignalLeak_{base}-\delta_n.
$$

默认：

$$
\delta_c=0.02,\quad \delta_r=0.02,\quad \delta_n=0.02.
$$

同时必须满足 task / tail / efficiency non-harm：

$$
Acc_{func}\ge Acc_{base}-0.003.
$$

$$
CEp99_{func}\le CEp99_{base}+\epsilon_{tail}.
$$

$$
ECE_{func}\le ECE_{base}+\epsilon_{ece}.
$$

$$
T_{amortized,func}/T_{base}\le 1.05.
$$

还必须击败 strong controls：

```text
NoOpMatchedOverhead
RandomMatchedNorm
AdamWParallelDirection
SNR-only
GeometryOnlyNoSNR
ShuffledPayload
MLPAnalogGeometryMaintenance
```

判断公式：

$$
CouplingR^2_{func}>CouplingR^2_{AdamWParallel}.
$$

$$
NoiseSignalLeak_{func}<NoiseSignalLeak_{RandomMatched}.
$$

$$
RealSignalReservoirRatio_{func}<RealSignalReservoirRatio_{SNRonly}.
$$

---

## C5：必须生成的图

```text
fig_coupling_predicted_vs_actual.svg
fig_coupling_r2_by_candidate.svg
fig_coupling_r2_vs_task_delta.svg
fig_signal_spectrum.svg
fig_real_signal_reservoir_ratio.svg
fig_noise_signal_leak.svg
fig_kernel_drift_vs_coupling.svg
fig_noise_leak_vs_ECE.svg
fig_real_signal_reservoir_vs_auc_time.svg
fig_functional_control_gap_manifold_channel.svg
```

每张图必须按 candidate family 上色：

```text
MLP baseline
GatedLegendreQuadratic / B42-B47 family
Activation / RSWAF / hinge family
RBF / FastKAN family
Orthogonal polynomial family
Fourier / Wavelet / B-spline family
Functional candidate
Strong controls
```

---

## C6：Codex 执行要求

Codex 不要把 Line C 写成 optional appendix。它是 v12.5.2 的第三条并行主线。

### C6.1 最小实现顺序

```text
C0: 实现 train/probe batch split 与 windowed logit displacement recorder。
C1: 实现 ridge coupling predictor，并写 v125_train_probe_coupling.csv。
C2: 实现 projected gradient / logit-Jacobian sketch，写 v125_signal_reservoir_sketch.csv。
C3: 增加 noise residual audit，写 NoiseSignalLeak。
C4: 将 C-line 指标接入 functional one-step / five-step cloned audit。
C5: 生成 Line C figures。
```

### C6.2 不满足条件时 Codex 先尝试什么

如果 `CouplingR2` 数值不稳定：

```text
1. 增大 probe batch Q；
2. 对 logits 做 classwise centering；
3. 增加 ridge_lambda；
4. 使用 PCA sketch 后再拟合 A_t；
5. 缩短窗口 Delta，避免非线性漂移太大。
```

如果 `NoiseSignalLeak` 过高：

```text
1. 先检查 shuffled-label residual 是否构造正确；
2. 检查 P_sig 阈值是否过宽；
3. 改用 top-energy signal projector；
4. 将 functional candidate 改为 AdamW-orthogonal residual direction；
5. 禁止直接降低 gate。
```

如果 `RealSignalReservoirRatio` 过高：

```text
1. 检查 basis / lift effective rank；
2. 检查 signal spectrum 是否 top-eigen collapse；
3. 对 base primitive 尝试 condition-preserving init；
4. 对 functional update 尝试 signal-projected geometry repair；
5. 不按 dataset name 调参。
```

如果 Line C 指标改善但 task 变差：

```text
1. 该 candidate 不 promotion；
2. 检查 logit drift / CEp99 / margin_p10；
3. 降低 lambda 或做 task-safe backtracking；
4. 保留为 diagnostic，不写 official success。
```

---

## C7：Line C 的最终路线条件

v12.5.2 成功不是只看一个指标。正式推进条件是：

```text
Line A:
  至少一个 base candidate 达到 MLP-like efficiency + expression + task near-pass。

Line C:
  该 base 没有明显 train-probe coupling collapse / noise leakage / reservoir trapping。

Line B:
  functional update 在 Line C 指标上改善 base，并击败 strong controls。
```

只有三者同时成立，才能进入 short-run functional training。

最终判断：

$$
\boxed{
\text{Efficiency line 证明 KAN 能跑得像 MLP；}
\text{Geometry diagnostics 证明它不是用坏几何换任务；}
\text{Functional line 证明 update 能独立改善 signal-channel geometry。}
}
$$

---

# 10. Line B：Functional Update 并行验证计划

Line B 现在只负责 functional update 方向、controls、cloned audit 与 official re-entry。Train-probe coupling、signal/reservoir、noise leakage 的计算和 gate 不再藏在 Line B 内部，而是由 Line C 作为独立几何诊断主线提供。

## B0：Functional route 原则

Functional update 继续做，但必须分两种状态：

```text
diagnostic:
  base not qualified；只做 cloned one-step / five-step / control matrix。

official:
  base qualified；可以做 short-run functional training。
```

v12.5.2 中，Line B 默认是 diagnostic。若 Line A 提前产生 base_qualified candidate，则 Line B 自动切 official。

---

## B1：Functional update 的新目标定义

不要再把 functional update 定义成：

```text
直接让 train loss 降更多；
直接让 curvature 更低；
直接模仿 AdamW。
```

新的目标是：

$$
\boxed{
\text{Functional update 应该改善 signal-channel geometry，}
\text{并且不损失 task / expression / efficiency。}
}
$$

具体目标：

```text
1. 训练 motion 更能预测 probe motion；
2. 真实信号更少困在 reservoir；
3. 噪声更少进入 signal channel；
4. hard tail / ECE 不坏；
5. basis cover / rank 不塌；
6. 击败 strong controls。
```

---

---

## B2：Functional candidates

这些 candidate 的几何目标一律从 Line C 读取，不再单独定义一套 curvature-only score。只保留可解释、basis-aware、低成本方向。

### Candidate F1：SNRProjectedGeometry

更新方向先通过 SNR gate：

$$
SNR_k=\mu_k^2-\frac{\sigma_k^2}{b-1}.
$$

仅允许：

$$
SNR_k>0
$$

的方向进入 geometry correction。

### Candidate F2：BasisOccupancyRebalance

目标是防止 basis dead / branch collapse：

```text
increase active basis entropy；
reduce dead basis fraction；
preserve logits through small compensation。
```

### Candidate F3：Lift / Norm Condition Repair

针对 Gated Legendre + Quadratic：

```text
Legendre branch condition；
quadratic branch scale；
directskip branch dominance；
input norm residual mix。
```

必须是 function-preserving 或 task-safe，不允许大幅移动 logits。

### Candidate F4：TailStabilityCorrection

只在 hard-tail event 上做 small correction：

```text
CEp99；
margin_p10；
wrong_confidence_p95；
tail coupling。
```

### Candidate F5：AdamW-Orthogonal Functional Residual

从 AdamWParallel 中减去同向分量，只测 functional 是否有互补信息：

$$
d_{func}^{\perp}=d_{func}-
\frac{\langle d_{func},d_{adam}\rangle}{\|d_{adam}\|^2+\epsilon}d_{adam}.
$$

### Candidate F6：SignalReservoirSeparationRepair

目标直接改善：

```text
RealSignalReservoirRatio
NoiseSignalLeak
CouplingR2
```

而不是直接最小化 CE。

---

## B3：Controls

每个 functional candidate 必须比较：

```text
C0 TaskOnlyAdamW
C1 NoOpMatchedOverhead
C2 RandomMatchedNorm
C3 AdamWParallelDirection
C4 SNR-only
C5 GeometryOnlyNoSNR
C6 ShuffledPayload
C7 ShuffledEvent
C8 MLPAnalogGeometryMaintenance
C9 QuadraticFeatureMLPAnalog
C10 BestLR / scalar step matched control
```

禁止：

```text
只和 AdamW baseline 比；
不和 AdamWParallel 比；
不和 RandomMatchedNorm 比；
不和 MLPAnalog 比。
```

---

## B4：One-step / five-step cloned audit

### 设置

```text
base snapshots:
  B21d expression anchor
  B23b task anchor
  B36c near-task anchor
  B42/B45 fast-reuse anchor
  any new A-line fused candidate

datasets:
  MNIST, Fashion-MNIST, KMNIST

seeds:
  0,1,2

mode:
  cloned weights only；
  no pollution of real training path。
```

### 必须记录

```text
candidate_id
snapshot_id
dataset
seed
event_id
update_type
control_id
lambda_selected
lambda_backtracking_steps
accepted
train_descent
probe_descent
holdout_descent_ratio
bad_step
bad_step_reason
CouplingR2_delta
CouplingCorr_delta
RealSignalReservoirRatio_delta
NoiseSignalLeak_delta
CEp99_delta
margin_p10_delta
ECE_proxy_delta
rank_delta
basis_entropy_delta
dead_basis_delta
cost_ms
memory_delta
```

### 判断标准

Diagnostic positive：

$$
holdout\_descent\_ratio\ge0.95
$$

$$
bad\_step\_rate\le0.02
$$

$$
\Delta CouplingR^2>0
$$

$$
\Delta NoiseSignalLeak<0
$$

且 beat at least:

```text
RandomMatchedNorm
NoOpMatchedOverhead
ShuffledPayload
```

Official promotable：

```text
base_qualified = true
```

并且：

$$
GeoGain_{func}>GeoGain_{AdamWParallel}+\epsilon
$$

$$
GeoGain_{func}>GeoGain_{SNR-only}+\epsilon
$$

$$
GeoGain_{func}>GeoGain_{MLPAnalog}+\epsilon.
$$

默认：

$$
\epsilon=0.01.
$$

### 不满足时 Codex 先尝试

若 functional 被 AdamWParallel 解释：

```text
1. 做 AdamW-orthogonal residual；
2. 做 control-gap residual target；
3. 不调低 threshold；
4. 不扩展到 short-run。
```

若 functional task-safe 但 geometry 没提升：

```text
1. 更换 geometry target 为 coupling/noise leak；
2. 不继续 curvature-only；
3. 检查是否表达/rank被压低。
```

若 geometry 提升但 task 不安全：

```text
1. 降 lambda；
2. 加 train-stream probe backtracking；
3. 加 tail veto；
4. 若仍不安全，candidate 停止。
```

---

## B5：Official functional short-run

只在 Line A 产生 base_qualified 后运行。

### 方法

```text
Base-AdamW
Base-AdamW + NoOpMatchedOverhead
Base-AdamW + RandomMatchedNorm
Base-AdamW + AdamWParallelMaintenance
Base-AdamW + SNR-only
Base-AdamW + BestFunctionalMaintenance
MLP + analogous maintenance
QuadraticFeatureMLP + analogous maintenance
```

### 通过标准

$$
Acc_{func}\ge Acc_{base}-0.003
$$

$$
AUCtime_{func}\le AUCtime_{base}
$$

$$
ECE_{func}\le ECE_{base}
$$

$$
GeoDiagnostic_{func}\ge GeoDiagnostic_{base}+10\%
$$

并且：

$$
Func > AdamWParallel,\quad Func > RandomMatched,\quad Func > MLPAnalog.
$$

---

---

# 11. v12.5.2 必须输出的 artifacts

## 11.1 Efficiency / kernel artifacts

```text
v125_contract_manifest.csv
v125_forward_kernel_truth.csv
v125_backward_kernel_truth.csv
v125_grad_correctness.csv
v125_full_step_efficiency.csv
v125_kernel_count_breakdown.csv
v125_allocation_trace.csv
v125_workspace_lifetime.csv
v125_efficiency_failure_table.csv
```

## 11.2 Expression / task artifacts

```text
v125_expression_battery.csv
v125_frozen_readout.csv
v125_matrix_span.csv
v125_task_triage.csv
v125_task_trace.csv
v125_task_failure_table.csv
v125_task_confirm20.csv
```

## 11.3 Line C geometry diagnostics artifacts

```text
v125_train_probe_coupling.csv
v125_signal_reservoir_sketch.csv
v125_noise_leak_audit.csv
v125_manifold_channel_diagnostics.csv
```

其中 `v125_train_probe_coupling.csv` 至少包含：

```text
run_id
candidate_id
basis_family
method
update_type
control_id
dataset
seed
step
window_size
batch_size_B
batch_size_Q
ridge_lambda
train_logit_drift_l2
probe_logit_drift_l2
CouplingR2
CouplingCorr
coupling_residual_norm
coupling_stability_across_splits
KernelDrift
CEp99_delta
ECE_delta
margin_p10_delta
official_gate_open
```

`v125_signal_reservoir_sketch.csv` 至少包含：

```text
run_id
candidate_id
basis_family
method
update_type
control_id
dataset
seed
step
window_size
sketch_dim
signal_effective_rank
signal_mass_topk
reservoir_fraction
top_eigen_share
dissipation_condition
RealSignalReservoirRatio
NoiseSignalLeak
SNR_positive_fraction
real_noise_gap
official_gate_open
```

`v125_manifold_channel_diagnostics.csv` 至少包含：

```text
run_id
candidate_id
basis_family
method
dataset
seed
step
base_qualified
functional_official_open
CouplingR2
CouplingCorr
RealSignalReservoirRatio
NoiseSignalLeak
KernelDrift
effective_rank_hidden
basis_occupancy_entropy
basis_dead_fraction
lift_condition_proxy
perturb_logit_drift_p95
CEp99
ECE
NLL
step_time_ratio
memory_ratio
```

## 11.4 Functional artifacts

```text
v125_functional_direction_audit.csv
v125_one_step_probe.csv
v125_five_step_probe.csv
v125_control_matrix.csv
v125_lambda_backtracking.csv
v125_functional_route.json
```

## 11.5 Route / provenance artifacts

```text
v125_route_decision.json
v125_run_manifest.json
v125_provenance_audit.csv
v125_no_fake_audit.csv
v125_hash_manifest.csv
```

---

# 12. 必须生成的可视化

## 12.1 Efficiency figures

```text
fig_A1_forward_ratio_bar.svg
fig_A1_forward_waterfall.svg
fig_A2_backward_ratio_bar.svg
fig_A2_memory_workspace_waterfall.svg
fig_A3_step_ratio_distribution.svg
fig_A3_kernel_count_trace.svg
fig_A3_step_vs_memory_pareto.svg
```

## 12.2 Expression / task figures

```text
fig_expression_delta_by_target.svg
fig_frozen_r2_by_candidate.svg
fig_task_mean_worst_delta.svg
fig_near_pass_by_candidate.svg
fig_auc_time_mean_max.svg
fig_ece_mean_max.svg
fig_classwise_failure_heatmap.svg
```

## 12.3 Line C manifold-channel diagnostics figures

```text
fig_coupling_predicted_vs_actual.svg
fig_coupling_r2_by_candidate.svg
fig_coupling_r2_vs_task_delta.svg
fig_signal_spectrum.svg
fig_real_signal_reservoir_ratio.svg
fig_noise_signal_leak.svg
fig_kernel_drift_vs_coupling.svg
fig_noise_leak_vs_ECE.svg
fig_real_signal_reservoir_vs_auc_time.svg
fig_functional_control_gap_manifold_channel.svg
```

每张 Line C 图必须按 candidate family 或 update/control family 分组：

```text
MLP baseline
GatedLegendreQuadratic / B42-B47 family
Activation / RSWAF / hinge family
RBF / FastKAN family
Orthogonal polynomial family
Fourier / Wavelet / B-spline family
Functional candidate
Strong controls
```

## 12.4 Functional figures

```text
fig_functional_control_gap.svg
fig_functional_bad_step_rate.svg
fig_functional_lambda_acceptance.svg
fig_functional_geo_gain_vs_task_slack.svg
fig_functional_vs_adamwparallel.svg
```

## 12.5 Failure taxonomy

```text
fig_failure_taxonomy_heatmap.svg
fig_route_tree.svg
fig_candidate_progression_b21_to_b47.svg
```

---

# 13. Route decision tree

## R0：FusedBaseQualified

条件：

```text
A1 efficiency pass
A2 expression pass
A3 task pass
Line C no obvious geometry collapse
no-fake pass
strict PureKAN pass
```

动作：

```text
打开 B5 official functional short-run。
```

## R1：FusedEfficiencyFail

条件：

```text
expression 或 task 有希望，但 step / forward / memory fail。
```

动作：

```text
继续 fused CUDA/Triton kernel；
不继续 schedule/scale task 小修；
不打开 official functional。
```

Codex 先尝试：

```text
1. forward component timing truth；
2. full-layer fused forward smoke；
3. fused backward local/two-stage reduction；
4. workspace lifetime planning；
5. 若 forward fusion收益不足，转 SimpleFastTaskGeometry。
```

## R2：ExpressionPassTaskFail

条件：

```text
A1/A2 pass，A3 fail。
```

动作：

```text
global normalization / branch-scale / schedule repair；
不按 dataset 调参；
Line C 继续诊断是否是 coupling / reservoir / noise leakage 问题；
functional 仍 diagnostic。
```

## R3：TaskPassExpressionFail

条件：

```text
task mean 好，但 expression battery fail。
```

动作：

```text
不能作为 base；
回到 expression preserving design；
Lite family 降级 diagnostic。
```

## R4：LineCGeometryCollapse

条件：

```text
task/efficiency near-pass，
但 CouplingR2 明显低于 MLP，或 NoiseSignalLeak 高，或 RealSignalReservoirRatio 高，或 CEp99/ECE 变坏。
```

动作：

```text
candidate 不 promotion；
优先查 basis/rank/condition/branch dominance；
不得为了 task near-pass 忽略几何崩坏；
不按数据集调参。
```

## R5：FunctionalControlEquivalent

条件：

```text
functional diagnostic positive but not beating AdamWParallel / Random / SNR-only / MLPAnalog。
```

动作：

```text
不调低 gate；
改为 AdamW-orthogonal residual / signal-reservoir target / control-gap residual；
继续 cloned diagnostic。
```

## R6：NewPrimitiveNeeded

条件：

```text
K1/K2 fused 后仍无法进入 A1 或 A3。
```

动作：

```text
转向 SimpleFastTaskGeometry：
  Activation/RSWAF hinge + minimal quadratic；
  compact RBF/FastKAN only with fused local K；
  local spline only with no dense materialization。
```

---

# 14. Codex 执行优先级

## 第一优先级：B47 forward bottleneck truth

Codex 先做：

```text
1. 读取 B47 forward path；
2. 拆分 input norm / Legendre / quadratic / directskip / projection 时间；
3. 输出 per-component time；
4. 实现 F1/F2 fused forward smoke；
5. 做 correctness audit；
6. 若 F2 有明显收益，再做 F3/F4。
```

成功门：

$$
T_{forward,F2}
\le
0.75T_{forward,B47}.
$$

否则，说明 forward fusion 收益不足，应简化 primitive。

## 第二优先级：B47 backward reduction truth

Codex 并行做：

```text
1. param-gradient branch split gradcheck；
2. local reduction kernel；
3. two-stage reduction；
4. no-materialized-basis backward；
5. compare to B47 manual CE step。
```

成功门：

$$
T_{backward,new}
\le
0.8T_{backward,B47}
$$

并保持：

$$
grad\_relerr<10^{-4}.
$$

## 第三优先级：Line C 独立实现

Codex 不要把 Line C 写成 optional appendix。最小实现顺序：

```text
1. train/probe batch split；
2. windowed logit displacement recorder；
3. ridge coupling predictor；
4. projected gradient / logit-Jacobian sketch；
5. signal/reservoir projector；
6. real residual / shuffled residual audit；
7. figures and route hooks。
```

若 `CouplingR2` 不稳定，Codex 先尝试：

```text
1. 增大 probe batch Q；
2. 对 logits 做 classwise centering；
3. 增大 ridge_lambda；
4. PCA sketch 后再拟合 A_t；
5. 缩短窗口 Delta。
```

若 `NoiseSignalLeak` 过高，Codex 先尝试：

```text
1. 检查 shuffled-label residual 构造；
2. 检查 P_sig 阈值是否过宽；
3. 改用 top-energy signal projector；
4. functional candidate 改为 AdamW-orthogonal residual direction；
5. 禁止降低 gate。
```

若 `RealSignalReservoirRatio` 过高，Codex 先尝试：

```text
1. 检查 basis / lift effective rank；
2. 检查 signal spectrum 是否 top-eigen collapse；
3. 对 base primitive 尝试 condition-preserving init；
4. 对 functional update 尝试 signal-projected geometry repair；
5. 不按 dataset name 调参。
```

## 第四优先级：K2 Lite rescue only if K1 system fail

如果 K1 full GatedLQ fused 仍太慢：

```text
1. 回到 B30/B31 LiteGated family；
2. 只补 expression，不继续 task-only；
3. 增加 minimal quadratic rank；
4. 保持 A1 priority。
```

## 第五优先级：Functional diagnostic 不停跑

只要有 snapshot，就并行跑：

```text
Line C train-probe coupling；
Line C signal/reservoir sketch；
one-step/five-step controls；
functional control gap。
```

但所有结果必须写：

```text
official_gate_open = 0
```

直到 base_qualified 且 Line C 没有明显几何崩坏。

---

# 15. 最终成功标准

## 15.1 Base success

$$
BaseSuccess =
StrictPureKAN
\land A1
\land A2
\land A3
\land NoLineCCollapse.
$$

展开：

$$
T_{step}/T_{MLP}\le1.25
$$

$$
M/M_{MLP}\le1.05
$$

$$
\Delta R^2_{expr}\ge-0.01
$$

$$
\Delta Acc_{mean}\ge-0.005
$$

$$
near\_pass\_rate\ge0.80
$$

$$
AUCtime_{mean}\le1.10
$$

并且：

$$
CouplingR^2_{KAN}\ge CouplingR^2_{MLP}-0.02
$$

$$
NoiseSignalLeak_{KAN}\le NoiseSignalLeak_{MLP}+0.02.
$$

Strong base success：

$$
\Delta Acc_{mean}\ge0
$$

$$
AUCtime_{mean}\le1.05
$$

$$
ECE_{base}\le ECE_{MLP}
$$

$$
T_{step}/T_{MLP}\le1.25.
$$

## 15.2 Functional success

Functional success 只能在 BaseSuccess 后定义：

$$
FunctionalSuccess =
BaseSuccess
\land TaskSafe
\land LineCImprove
\land ControlResistant.
$$

其中：

$$
TaskSafe:
Acc_{func}\ge Acc_{base}-0.003.
$$

$$
LineCImprove:
CouplingR^2_{func}>CouplingR^2_{base}+0.02.
$$

$$
RealSignalReservoirRatio_{func}
<
RealSignalReservoirRatio_{base}-0.02.
$$

$$
NoiseSignalLeak_{func}
<
NoiseSignalLeak_{base}-0.02.
$$

$$
ControlResistant:
Func>AdamWParallel,
\quad Func>RandomMatched,
\quad Func>SNROnly,
\quad Func>MLPAnalog.
$$

## 15.3 三线合并成功

三线同时成立才允许进入 official short-run functional training：

```text
Line A:
  至少一个 strict PureKAN base 达到 MLP-like efficiency + expression + task near-pass。

Line C:
  该 base 没有明显 train-probe coupling collapse / noise leakage / reservoir trapping。

Line B:
  functional update 在 Line C 指标上改善 base，并击败 strong controls。
```

---

# 16. 本轮不做什么

v12.5.2 明确禁止：

```text
1. 不继续 B42/B45/B46 的小网格 scale/schedule；
2. 不按数据集调参；
3. 不降低 A1/A2/A3/Line C/functional gate；
4. 不把 diagnostic positive 写成 official；
5. 不用 teacher / distillation / modified loss；
6. 不把 kernel warmup artifact 当 AUC-time；
7. 不在 base 未合格时跑 official functional short-run；
8. 不重新大范围扫所有 basis；
9. 不把 Line C 当 optional appendix；
10. 不把 kernel drift 小当成唯一几何目标。
```

---

# 17. 最终建议

我的建议是：

$$
\boxed{
\text{v12.5.2 必须同时保留效率路线、流形-信号通道几何诊断、functional update 路线，}
\text{但三者职责不同。}
}
$$

效率路线负责回答：

```text
是否存在 MLP-like strict PureKAN primitive？
```

Line C 负责回答：

```text
这个 primitive 的训练运动是否 test/probe-visible，真实信号是否进入 signal channel，噪声是否泄漏？
```

functional 路线负责回答：

```text
在合格 base 上，functional update 是否能提供 control-resistant signal-channel geometry advantage？
```

现在的关键是不要把这三件事混在一起。当前 base 未合格，functional 只能 diagnostic；但如果停止 functional diagnostic，又会导致 base 合格后没有可接入的 functional hypothesis。因此最佳推进方式是：

```text
Line A:
  lower-level fused CUDA/Triton forward+backward kernel，优先 K1/K2 top families。

Line C:
  train-probe coupling + signal/reservoir + noise leak，作为独立几何诊断主线。

Line B:
  basis-aware functional update + strong controls，持续 cloned diagnostic。

Merge gate:
  只有 Line A base_qualified 且 Line C no-collapse 后，Line B 才转 official functional training。
```

一句话：

$$
\boxed{
\text{不要再用 Python-level 手写 backward 或 schedule 小修消耗主线；}
\text{现在必须进入 fused kernel，同时让 functional update 在 Line C 几何诊断上准备好。}
}
$$
