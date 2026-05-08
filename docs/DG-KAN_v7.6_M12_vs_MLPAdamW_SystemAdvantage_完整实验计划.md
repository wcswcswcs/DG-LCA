# DG-KAN v7.6：M12 对 MLP-AdamW 系统级优势确认与 PureKAN-NG S1 闭环完整实验计划

> 本计划基于 v7.5 最新最终复盘制定。v7.5 的关键结果是：`M12 = A2S-fused-linear-silu-packed-generic-head-M5init-logit-distill-from-C3-T4-alpha025` 首次同时满足 `StrictPass + GradPass + MacroSignificantPass + FullGridS2Pass`。  
> 这意味着 v7.5 的最低成功标准已经达成：M12 不再只是 dense oracle，不再只是 task-only winner，也不是 mean-S2 擦边，而是 9/9 shapes 通过 S2 的 strict PureKAN candidate。  
> 但这还不是终极系统级胜利。v7.6 的目标是判断：M12 相比 MLP-AdamW 的优势是否稳定、可复现、公平、具备 wall-clock 意义，并且能否继续从 S2 推到 S1。

---

## 0. 当前结论

### 0.1 是否达到目标？

如果目标是 v7.5 的最低成功标准：

$$
\text{StrictPass}+\text{GradPass}+\text{MacroSignificantPass}+\text{FullGridS2Pass},
$$

那么答案是：

$$
\boxed{\text{已经达到。}}
$$

v7.5 最终确认 run 中，`M12` 的结果是：

```text
strict_pass = 1
grad_pass = 1
macro_pass_v75 = 1
fullgrid_s2_pass = 1
success_v75_minimum = 1
```

task / macro 结果：

```text
B0 / MLP-AdamW reference:
  val acc  = 0.8443
  test acc = 0.7949

M12:
  val acc  = 0.8651
  test acc = 0.8163
  macro val gap = +0.0208
  CI95 low = +0.0129
  Holm p = 1.40e-03
  test gap = +0.0214
  ECE delta = +0.0024
  NLL delta = -0.0914
```

gradient correctness：

```text
GradPass = 6/6
max relerr = 8.07e-05
```

efficiency：

```text
memory ratio mean = 1.0182
step ratio mean = 1.3688
forward ratio mean = 2.0993
backward ratio mean = 1.1763
S2 shapes = 9/9
```

逐 shape S2：

```text
MNIST bs128: memory 0.9952, step 1.4490
MNIST bs256: memory 1.0110, step 1.3643
MNIST bs512: memory 1.0482, step 1.3917

Fashion-MNIST bs128: memory 0.9952, step 1.3589
Fashion-MNIST bs256: memory 1.0110, step 1.3522
Fashion-MNIST bs512: memory 1.0482, step 1.3338

KMNIST bs128: memory 0.9952, step 1.3622
KMNIST bs256: memory 1.0110, step 1.3599
KMNIST bs512: memory 1.0482, step 1.3466
```

因此，v7.5 已经完成了过去几轮一直没有完成的合流：

$$
\boxed{
\text{macro task advantage}
+
\text{strict manual gradient correctness}
+
\text{full-grid S2 efficiency}
}
$$

---

### 0.2 对比 MLP-AdamW，是否有显著优势？

如果只看当前 v7.5 预设口径，即 5 seeds、MNIST/Fashion-MNIST/KMNIST、B0/MLP-AdamW reference，那么答案是：

$$
\boxed{\text{M12 对 MLP-AdamW 有统计显著的 macro accuracy 优势。}}
$$

证据是：

$$
\Delta Acc_{\text{macro,val}}=+0.0208>+0.0200.
$$

$$
CI_{95\%,macro}^{low}=+0.0129>0.
$$

$$
p_{\text{Holm}}=1.40\times 10^{-3}<0.05.
$$

test 同向：

$$
\Delta Acc_{\text{macro,test}}=+0.0214.
$$

所以在 task accuracy 上，M12 已经不是“稍微超过”。它跨过了预设的 practical threshold 和 statistical threshold。

但是如果问题是：

$$
\boxed{\text{M12 是否已经全面系统级优于 MLP-AdamW？}}
$$

答案仍然是：

$$
\boxed{\text{尚未完全证明。}}
$$

原因是：

```text
1. 当前是 5-seed confirm，不是 10-seed / 20-seed final confirm。
2. macro gap 只比 +0.0200 gate 高 +0.0008，安全边际不厚。
3. M12 使用 C3 teacher distillation，需要 MLP-distill 公平对照。
4. M12 过的是 S2，不是 S1。
5. ECE delta = +0.0024，只是 within gate，不是 ECE 优于 MLP。
6. NLL 显著更好，但 ValLossAUC_time / time_to_target 未闭合。
7. forward ratio = 2.0993，forward path 仍偏慢。
8. scaling / robustness / larger train size / patch-token 尚未证明。
```

因此，当前最准确的结论是：

$$
\boxed{
\text{M12 是第一个 S2-MacroSignificant strict PureKAN-NG candidate；}
}
$$

但还不能写成：

$$
\boxed{
\text{PureKAN-NG 已经全面系统级超越 MLP-AdamW。}
}
$$

---

## 1. 实验进展分析

### 1.1 从 dense oracle 到 strict S2 candidate 的路线已经闭合

早期 dense D3 证明了 PureKAN / poly2_gate family 有超过 MLP 的表达力，但 dense D3 不是 strict kernel-native candidate。后续 H3/C3 证明 strict KAN head 和 cached stack 可以保住 macro positive signal，但 efficiency 仍失败。v7.5 的 M12 则首次把表达力、strict gradient correctness 和 full-grid S2 同时闭合。

这条路线的机制可以总结为：

```text
D3:
  表达力 oracle。

H3/C3:
  strict head + cached hidden-y，把 dense task signal 转成 strict KAN signal。

A2S/D3:
  low-live-set bridge，把 memory 接近 S2。

M5:
  macro + S2 接近闭合，但 gradient relerr 有一行略超 gate。

M12:
  保留 M5 初始化轨迹；
  使用 packed generic V63 head；
  使用 FP32 stable SiLU derivative 等价式；
  修复 GradPass；
  最终进入 FullGridS2。
```

其中，M12 不是通过放宽 gate 成功，而是通过数值等价重排让 manual backward 更贴近 autograd。关键导数写法是：

```python
sig = torch.sigmoid(y_i)
silu_prime = sig + y_i * (sig - sig.square())
delta = delta * silu_prime
```

这个形式避免了 FP64 temp，保持 full-grid S2；而 FP64 derivative 虽然也能修 GradPass，但会使 bs512 memory ratio 升到 `1.0621`，导致 S2 失败。因此 v7.5 的最终成功来自结构、owner/runtime、manual backward 数值稳定化的共同闭合，而不是小修小补。

---

### 1.2 当前已经排除的主线

v7.5 之前的大量结果已经说明，下列方向不应作为 v7.6 主线：

```text
1. classwise-safe:
   不作为硬目标，只作为 local failure audit。

2. loss / sampler / class weight / focal / target margin:
   不应再作为主修方向。

3. AdamW 小参数 sweep:
   lr / wd / smoothing 已经不是根因。

4. hidden shrink:
   会损 task signal，且 memory 没有按比例闭合。

5. naive low-rank:
   没保住 task signal或 GradPass。

6. pure recompute-y checkpoint:
   memory 能降，但 step 代价过大。

7. FP64 derivative:
   能修 GradPass，但破坏 full-grid S2 memory。
```

M12 的成功说明：应该继续沿着 **expression-preserving low-live-set + packed/fused implementation + numerically stable manual backward** 这条线，而不是回到外部 loss trick。

---

## 2. 当前问题所在

### 2.1 M12 已经解决了 “有没有可行候选” 的问题

在 v7.5 之前，我们一直没有一个 candidate 同时满足：

```text
macro significant
gradient correctness
full-grid S2
strict PureKAN
```

M12 解决了这个问题。现在问题从：

$$
\text{是否存在一个可行 PureKAN-NG candidate？}
$$

变成：

$$
\text{这个 candidate 是否足够稳定、公平、快速、可扩展？}
$$

这是项目阶段的根本变化。

---

### 2.2 M12 的 task advantage 有显著性，但安全边际不厚

M12 的 macro gap 是：

$$
+0.0208.
$$

v7.5 gate 是：

$$
+0.0200.
$$

安全边际：

$$
+0.0208 - +0.0200 = +0.0008.
$$

这说明它已经过线，但仍然需要 10-seed / 20-seed 复现。当前 5-seed 结果足以说明 v7.5 最低成功，但不足以支撑最终论文级别的 robust claim。

---

### 2.3 M12 是 S2，不是 S1

M12 mean memory：

$$
r_{\text{mem,mean}}=1.0182.
$$

M12 mean step：

$$
r_{\text{step,mean}}=1.3688.
$$

S1 要求：

$$
r_{\text{mem}}<1.00,
$$

$$
r_{\text{step}}\leq1.35.
$$

mean memory 距离 S1：

$$
1-\frac{1.00}{1.0182}\approx1.79\%.
$$

mean step 距离 S1：

$$
1-\frac{1.35}{1.3688}\approx1.37\%.
$$

但 full-grid S1 要看 worst shape。当前 worst memory 是：

$$
r_{\text{mem,worst}}=1.0482.
$$

要到 `<1.00` 需要降低：

$$
1-\frac{1.00}{1.0482}\approx4.60\%.
$$

当前 worst step 是：

$$
r_{\text{step,worst}}=1.4490.
$$

要到 `<=1.35` 需要提速：

$$
1-\frac{1.35}{1.4490}\approx6.83\%.
$$

这说明 S1 不远，但需要非常精准的 margin repair，而不是大改架构。大改可能破坏 M12 刚刚闭合的表达力。

---

### 2.4 Forward path 是当前效率内部的主要短板

M12 的 full step 过 S2，但 forward ratio 是：

$$
r_{\text{forward}}=2.0993.
$$

backward ratio 是：

$$
r_{\text{backward}}=1.1763.
$$

这说明 backward 已经接近 MLP，真正明显偏慢的是 forward/head/stack 路径。下一步如果只看 step ratio，会掩盖 forward 仍然偏重的问题。v7.6 必须拆：

```text
head_forward_ms
stack_forward_ms
packed_head_owner_overhead
linear_silu_forward_temp
kernel_count_forward
materialized_tensor_count_forward
```

---

### 2.5 Teacher distillation 的公平性尚未证明

M12 使用 C3 teacher logit distill：

```text
teacher = C3
temperature = 4
alpha = 0.25
```

这不是问题，但必须公平比较：

```text
MLP-AdamW
MLP + C3 distill
MLP + self-distill
M12 no-distill
M12 self-distill
M12 + C3 distill
```

如果 MLP + C3 distill 追平 M12，则 M12 的优势主要来自 teacher objective，而不是 KAN structure。反之，如果 M12 + C3 distill 仍明显优于 MLP + C3 distill，则可以说 M12 对 C3 teacher signal 有更好的结构吸收能力或 inductive bias。

---

### 2.6 Wall-clock 意义尚未闭合

M12 的 step ratio 是：

$$
1.3688.
$$

这意味着每步仍慢于 MLP-AdamW。如果 final accuracy 高，但 wall-clock AUC 更差，那么系统级优势要降级为 quality advantage。

v7.6 必须验证：

$$
ValLossAUC_{time,M12}\leq ValLossAUC_{time,MLP}.
$$

或者至少：

$$
time\_to\_target(M12)\leq1.10\cdot time\_to\_target(MLP).
$$

---

## 3. 是否在正确道路上

现在明确在正确道路上。

原因是 v7.5 的 M12 不是单点小修，而是首次完成了过去所有路线一直缺失的合流：

$$
\boxed{
\text{strict PureKAN}
+
\text{manual GradPass}
+
\text{macro significant accuracy}
+
\text{full-grid S2 efficiency}
}
$$

这说明项目不再只是 dense oracle 证明，也不再只是 kernel benchmark。M12 已经是一个真实可训练、真实测量、真实过 gate 的 PureKAN-NG candidate。

但下一步不能急着宣布终极胜利。正确路线是：

```text
1. 复现稳定性；
2. 公平对照；
3. wall-clock 收敛；
4. S1 margin；
5. 机制消融；
6. 扩展性。
```

换句话说，v7.6 的核心不是 “再找一个新 candidate”，而是判断 M12 是否能从 **阶段性成功** 升级为 **系统级优势**。

---

## 4. 离目标还差多远

### 4.1 离 v7.5 最低目标

已经达到。

$$
\boxed{
\text{StrictPass}+\text{GradPass}+\text{MacroSignificantPass}+\text{FullGridS2Pass}
}
$$

已经在 M12 上同时成立。

---

### 4.2 离 MLP-AdamW 显著 accuracy 优势

在 5-seed 当前口径下已经达到：

$$
\Delta Acc_{\text{macro,val}}=+0.0208.
$$

$$
CI_{95\%,macro}^{low}=+0.0129.
$$

$$
p_{\text{Holm}}=1.40\times10^{-3}.
$$

但需要 10-seed 或 20-seed confirm 来确认安全边际。

---

### 4.3 离 S1

mean S1 差距很小：

```text
memory mean gap to S1 ≈ 1.79%
step mean gap to S1 ≈ 1.37%
```

full-grid worst-shape 差距更大但仍可修：

```text
memory worst gap to S1 ≈ 4.60%
step worst gap to S1 ≈ 6.83%
```

v7.6 可以把 S1 当作明确目标，但不能牺牲 macro gap。

---

### 4.4 离全面系统优势

还缺：

```text
10/20-seed reproducibility
MLP-distill fairness
teacher-free / teacher-dependent decomposition
ValLossAUC_time
time_to_target
S1
forward path attribution
scaling / robustness
geometry Pareto
```

所以当前最终定位是：

$$
\boxed{
\text{M12 已经是第一个 S2-MacroSignificant PureKAN-NG candidate；}
}
$$

但还不是：

$$
\boxed{
\text{全面系统级超过 MLP-AdamW 的最终答案。}
}
$$

---

## 5. v7.6 总体目标

v7.6 的整体目标是把 M12 从 v7.5 最低成功推进到系统级优势确认：

$$
\boxed{
\text{M12 是否真正、稳定、公平、wall-clock 有意义地优于 MLP-AdamW？}
}
$$

v7.6 要回答六个核心问题：

```text
Q1:
  M12 的 +2.08% macro gap 是否在 10/20 seeds 下稳定？

Q2:
  M12 的优势是否仍然存在于 MLP + C3 distill 公平对照下？

Q3:
  M12 的 teacher dependency 有多强？teacher-free M12 是否仍有优势？

Q4:
  M12 能否从 S2 推到 S1，而不破坏 macro gap 和 GradPass？

Q5:
  M12 是否在 ValLossAUC_time / time_to_target 上有系统意义？

Q6:
  M12 的成功机制到底来自 packed generic head、M5 init、C3 distill、stable SiLU derivative、还是 fused A2S stack 的组合？
```

v7.6 最低成功标准：

$$
\boxed{
\text{StrictPass}
+\text{GradPass}
+\text{10SeedMacroSignificantPass}
+\text{FullGridS2Pass}
+\text{FairnessPass}
}
$$

v7.6 正式成功标准：

$$
\boxed{
\text{StrictPass}
+\text{GradPass}
+\text{10SeedMacroSignificantPass}
+\text{FullGridS1Pass}
+\text{FairnessPass}
+\text{TimeAUCPass}
}
$$

v7.6 强成功标准：

$$
\boxed{
\text{S1}
+\text{TimeAUCPass}
+\text{ECE/NLL improvement}
+\text{MLP-distill superiority}
+\text{sample/robust/scaling advantage}
}
$$

---

## 6. v7.6 禁止事项

第一，不允许回到 loss、sampler、class weight、focal、target margin、classwise-safe 主线。v7.6 的主问题不是刷榜。

第二，不允许把 5-seed 成功直接写成终极成功。必须做 10-seed 或 20-seed confirm。

第三，不允许只比较 raw MLP-AdamW。因为 M12 用了 C3 teacher distillation，必须加入 MLP-distill 和 M12 teacher-free 对照。

第四，不允许只看 mean ratio。FullGridS2 / FullGridS1 必须看 9/9 shapes 和 worst shape。

第五，不允许只看 final accuracy。必须记录 ValLossAUC_step、ValLossAUC_time、time_to_target_loss、time_to_target_acc。

第六，不允许为了 S1 margin repair 破坏 GradPass 或 macro gap。任何 S1 repair 都必须重新跑 full gradient correctness 和 task gate。

第七，不允许把 FP64 derivative 作为成功修复。它会增加 live set，已知会破坏 bs512 S2；只允许作为 diagnostic。

第八，不允许把 posthoc calibration 和 training-time architecture claim 混在一起。posthoc temperature scaling 可以报告，但不能作为 M12 architecture 本身的校准优势。

第九，不允许把 geometry 当硬目标。geometry 只作为 Pareto 维度；如果伤 task 或 time-to-target，就不能进入主路线。

第十，不允许在 scaling/robustness 未测前把 M12 的优势外推到大数据或 token task。

---

## 7. 核心假设

### H1：M12 的 macro accuracy advantage 对 MLP-AdamW 稳定显著

H1 假设 M12 的优势不是 5-seed 偶然。

H1 成立标准：

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

Seed-wise stability：

$$
win\_rate_{seed}\geq0.70.
$$

如果 10-seed 后：

$$
0.018\leq \Delta Acc_{\text{macro,val}}<0.020,
$$

则进入 20-seed confirm，不直接判死。若 20-seed 仍低于 +0.020，则只能说 M12 是 positive signal，不是 robust significant advantage。

---

### H2：M12 的优势不是 C3 distillation-only

H2 假设 C3 teacher 对 M12 有帮助，但 M12 的结构也有独立贡献。

必须比较：

```text
MLP-AdamW
MLP + C3 logit distill
MLP + self-distill
M12 supervised-only
M12 + self-distill
M12 + C3 distill
A2S + C3 distill
C3 teacher
```

H2 成立标准：

$$
Acc_{\text{M12+C3distill}} - Acc_{\text{MLP+C3distill}} \geq0.010.
$$

并且：

$$
NLL_{\text{M12+C3distill}}\leq NLL_{\text{MLP+C3distill}}+0.01.
$$

若：

$$
Acc_{\text{MLP+C3distill}}\approx Acc_{\text{M12+C3distill}},
$$

则优势主要来自 teacher objective，而不是 KAN architecture。此时 route 应标记为：

```text
teacher_objective_explains_advantage
```

---

### H3：M12 的 teacher dependency 可以被量化

H3 假设 M12 不是只能依赖 C3 teacher 才工作；teacher 可能是 boost，但不是唯一来源。

Teacher dependency 定义：

$$
D_{\text{teacher}}
=
Acc_{\text{M12+C3distill}}
-
Acc_{\text{M12-supervised}}.
$$

强依赖：

$$
D_{\text{teacher}}\geq0.010.
$$

中等依赖：

$$
0.005\leq D_{\text{teacher}}<0.010.
$$

弱依赖：

$$
D_{\text{teacher}}<0.005.
$$

如果强依赖成立，M12 的 official recipe 必须显式包含 C3 teacher，并且公平对照必须包含 MLP-distill。

---

### H4：M12 能以小幅 live-set / forward repair 推到 S1

M12 当前离 S1 很近。H4 假设：S1 gap 来自可修的 head temp、manual cache lifetime、packed owner state、allocator padding 或 forward temp，而不是表达力必要 live-set。

H4 成立标准：

FullGridS1Pass：

$$
r_{\text{mem,max}}<1.00.
$$

$$
r_{\text{step,max}}\leq1.35.
$$

Task preservation：

$$
\Delta Acc_{\text{macro,val,S1}}\geq
\Delta Acc_{\text{macro,val,M12}}-0.003.
$$

也就是说，如果 M12-S1 把 macro gap 从 `+0.0208` 打到 `+0.017` 以下，则 S1 修复失败。

---

### H5：M12 的系统价值必须通过 time-to-target / AUC 证明

H5 假设 M12 虽然 step 慢于 MLP，但可能因为更好 final accuracy / NLL / loss curve 而具备 wall-clock 意义。

定义：

$$
AUC_{time}^{val\_loss}
=
\sum_i
\frac{L_i+L_{i-1}}{2}(t_i-t_{i-1}).
$$

H5 成立标准：

$$
ValLossAUC_{time,M12}\leq ValLossAUC_{time,MLP}.
$$

或：

$$
time\_to\_target\_acc(M12)\leq1.10\cdot time\_to\_target\_acc(MLP).
$$

其中：

$$
Acc_{target}=Acc_{\text{MLP-final}}+0.005.
$$

如果 M12 只在 final accuracy 上赢，但 time-to-target 和 AUC_time 全部更差，则它是 quality success，不是 wall-clock system success。

---

### H6：M12 成功机制必须可消融解释

H6 假设 M12 成功来自以下组件组合：

```text
packed generic V63 head
M5 initialization trajectory
C3 logit distillation
A2S fused linear-SiLU stack
FP32 stable SiLU derivative
```

H6 成立标准：

移除组件后应出现对应退化：

```text
no C3 distill:
  macro gap drops by >= 0.005

no packed head:
  step/memory or GradPass worsens

old SiLU derivative:
  GradPass fails or relerr approaches > 1e-4

no M5 init:
  macro gap drops or convergence worsens

unfused stack:
  FullGridS2 fails
```

如果某组件移除后完全不变，应简化 final recipe。

---

### H7：M12 的优势需要至少在一个扩展维度成立

H7 假设 M12 不是只在当前 small fixed task 上有效。至少需要一个扩展维度支持：

```text
sample efficiency
noise robustness
larger train size
hidden width scaling
patch/token smoke
geometry Pareto
```

H7 任一成立标准：

Sample efficiency：

$$
AUC_{data,M12}>AUC_{data,MLP}.
$$

Noise robustness：

$$
AccDrop_{M12}<AccDrop_{MLP}
$$

for at least two noise settings.

Scaling smoke：

$$
\Delta Acc_{\text{macro,val}}\geq0.010
$$

under larger train size or hidden size, while:

$$
r_{\text{mem}}\leq1.10,
$$

$$
r_{\text{step}}\leq1.70.
$$

---

## 8. Candidate 设计

### 8.1 Baselines

```text
B0-MLP-AdamW-reference
B1-MLP-AdanLite-reference
B2-MLP-C3-logit-distill-T4-alpha025
B3-MLP-self-distill
B4-C3-teacher
B5-A2S-supervised
B6-A2S-C3-distill
B7-M12-final-v75
```

### 8.2 M12 confirmation candidates

```text
M12-FINAL:
  packed generic V63 head
  FP32 stable SiLU derivative
  C3 distill T4 alpha0.25
  M5 initialization trajectory

M12-NODISTILL:
  same architecture, no C3 teacher

M12-SELFDISTILL:
  same architecture, self-distill

M12-MLPTEACHER:
  same architecture, MLP teacher

M12-RESEED:
  independent rerun seeds

M12-LONGBUDGET:
  480 steps

M12-NOSTABLESILU:
  old derivative form, diagnostic only

M12-NOPACKEDHEAD:
  unpacked generic head, diagnostic only

M12-NOM5INIT:
  same architecture, non-M5 init trajectory
```

### 8.3 MLP fairness candidates

```text
MLP-AdamW
MLP-AdanLite
MLP-C3-logit-distill-T2-alpha025
MLP-C3-logit-distill-T4-alpha025
MLP-C3-logit-distill-T4-alpha050
MLP-self-distill
MLP-longbudget-480
```

### 8.4 S1 repair candidates

```text
S1-0-M12-current
S1-1-head-cache-trim
S1-2-packed-owner-state-trim
S1-3-manual-cache-lifetime-trim
S1-4-bs512-head-temp-streaming
S1-5-forward-temp-fusion
S1-6-update-buffer-lifetime-trim
S1-7-allocator-padding-control
S1-8-forward-kernel-count-trim
S1-9-M12-S1-combo
```

### 8.5 Calibration candidates

```text
CAL0-M12-current
CAL1-M12-temperature-posthoc
CAL2-M12-train-time-logit-scale-edge-owned
CAL3-M12-weak-geometry-lambda1e-5
CAL4-M12-weak-geometry-lambda1e-4
CAL5-M12-distill-temperature-adjusted
```

### 8.6 Robustness / scaling candidates

```text
R0-MLP-AdamW
R1-MLP-distill
R2-M12-final
R3-M12-S1-combo, if exists
R4-M12-no-distill
```

---

## 9. 实验阶段总览

v7.6 分为十三个阶段：

```text
P0: v7.5 M12 reproduction and provenance lock
P1: 10-seed / 20-seed M12 significance confirmation
P2: MLP-AdamW / MLP-distill fairness comparison
P3: teacher dependency and optimization reachability audit
P4: M12 mechanism ablation
P5: S1 memory/step margin attribution
P6: S1 repair package
P7: ValLossAUC / time-to-target verification
P8: ECE / NLL / calibration refinement
P9: sample efficiency and robustness expansion
P10: scaling smoke
P11: geometry Pareto diagnostic
P12: candidate co-selection and route decision
P13: artifact and failure audit
```

P0-P4 确认 M12 是否真实、公平、可解释。  
P5-P6 把 S2 推向 S1。  
P7-P10 判断系统级优势是否成立。  
P11 只做 geometry Pareto，不作为硬目标。  
P12-P13 输出最终 route。

---

## 10. P0：v7.5 M12 reproduction and provenance lock

### 10.1 目的

确认 v7.6 与 v7.5 final M12 可比。P0 不寻找新结论，只验证 baseline、runner、no-fake/no-proxy、manual path、gradient correctness、MLP denominator。

### 10.2 必跑对象

```text
B0-MLP-AdamW-reference
M12-FINAL
C3-teacher
A2S-reference
```

### 10.3 必须记录字段

```text
run_id
artifact_root
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
teacher_used
teacher_candidate
distill_temperature
distill_alpha
grad_relerr_max
grad_cos_min
macro_val_gap
ci95_low
holm_p
test_gap
ECE_delta
NLL_delta
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
forward_ratio_mean
backward_ratio_mean
S2_shape_count
S1_shape_count
reproduction_delta_macro_gap
reproduction_delta_memory
reproduction_delta_step
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

M12 task reproduction：

$$
|\Delta Acc_{\text{M12,v76}}-0.0208|\leq0.005.
$$

M12 efficiency reproduction：

$$
|r_{\text{mem,M12,v76}}-1.0182|\leq0.02.
$$

$$
|r_{\text{step,M12,v76}}-1.3688|\leq0.10.
$$

### 10.5 可视化

```text
p0_m12_reproduction_task_bar.svg
p0_m12_reproduction_efficiency_bar.svg
p0_contract_heatmap.svg
p0_candidate_lineage_table.md
```

---

## 11. P1：10-seed / 20-seed M12 significance confirmation

### 11.1 目的

确认 M12 的 `+0.0208` macro gap 是否稳定。P1 是 v7.6 第一优先级，先于 S1 repair 和 scaling。

### 11.2 设置

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0..9

optional seeds:
  0..19 if 10-seed margin is close

steps:
  240

train/val/test:
  1536/512/512

bootstrap:
  10000 samples
```

### 11.3 必跑对象

```text
MLP-AdamW-reference
M12-FINAL
C3-teacher, reference only
```

### 11.4 必须记录字段

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
AdaptiveECE
NLL
Brier
val_gap_vs_MLP
test_gap_vs_MLP
seedwise_win
ValLossAUC_step
ValLossAUC_time
time_to_target_acc
time_to_target_loss
```

统计字段：

```text
macro_val_gap_mean
macro_val_gap_std
macro_val_gap_ci95_low
macro_val_gap_ci95_high
paired_t_p
wilcoxon_p
holm_p
cohen_h
seed_win_rate
dataset_gap_mean
dataset_gap_ci95_low
```

### 11.5 判断标准

10-seed MacroSignificantPass：

$$
\Delta Acc_{\text{macro,val}}\geq0.0200.
$$

$$
CI_{95\%,macro}^{low}>0.
$$

$$
p_{\text{Holm}}<0.05.
$$

$$
\Delta Acc_{\text{macro,test}}\geq0.015.
$$

Seed stability：

$$
win\_rate_{seed}\geq0.70.
$$

If:

$$
0.018\leq\Delta Acc_{\text{macro,val}}<0.020,
$$

then run 20-seed confirmation.

If:

$$
\Delta Acc_{\text{macro,val}}<0.015,
$$

route becomes:

```text
NoReproduction
```

### 11.6 可视化

```text
p1_seedwise_val_gap_boxplot.svg
p1_bootstrap_ci_macro.svg
p1_dataset_gap_ci.svg
p1_val_test_gap_scatter.svg
p1_seed_win_rate_bar.svg
```

---

## 12. P2：MLP-AdamW / MLP-distill fairness comparison

### 12.1 目的

回答 M12 相比 MLP-AdamW 的优势是否公平。因为 M12 使用 C3 teacher distillation，所以必须比较 MLP 在相同 teacher objective 下的表现。

### 12.2 必跑对象

```text
MLP-AdamW
MLP-AdanLite
MLP-C3-logit-distill-T4-alpha0.25
MLP-C3-logit-distill-T4-alpha0.50
MLP-self-distill
M12-no-distill
M12-C3-distill
M12-self-distill
C3-teacher
```

### 12.3 必须记录字段

```text
candidate
optimizer
teacher_used
teacher_type
distill_temperature
distill_alpha
train_acc
val_acc
test_acc
val_loss
test_loss
ECE
AdaptiveECE
NLL
Brier
ValLossAUC_step
ValLossAUC_time
time_to_target_acc
time_to_target_loss
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
forward_ratio_mean
backward_ratio_mean
```

### 12.4 判断标准

M12 vs MLP-AdamW pass：

$$
Acc_{M12} - Acc_{MLP-AdamW}\geq0.0200.
$$

M12 vs MLP-distill fairness pass：

$$
Acc_{M12-distill} - Acc_{MLP-distill}\geq0.0100.
$$

NLL fairness：

$$
NLL_{M12-distill}\leq NLL_{MLP-distill}+0.01.
$$

Efficiency fairness：

$$
r_{\text{mem,M12}}\leq1.05.
$$

$$
r_{\text{step,M12}}\leq1.50.
$$

If MLP-distill closes the gap:

$$
Acc_{M12-distill}-Acc_{MLP-distill}<0.005,
$$

route becomes:

```text
teacher_objective_explains_advantage
```

If M12 remains ahead:

$$
Acc_{M12-distill}-Acc_{MLP-distill}\geq0.010,
$$

route becomes:

```text
KAN_structure_distillation_advantage
```

### 12.5 可视化

```text
p2_fairness_val_gap_bar.svg
p2_distill_effect_by_model.svg
p2_nll_ece_fairness_scatter.svg
p2_task_efficiency_fairness_pareto.svg
p2_mlp_vs_m12_learning_curves.svg
```

---

## 13. P3：teacher dependency and optimization reachability audit

### 13.1 目的

判断 M12 本身是否足够强，还是必须依赖 C3 teacher 才能超过 MLP-AdamW。

### 13.2 Candidates

```text
M12-supervised-only
M12-C3-logit-distill
M12-C3-feature-distill
M12-C3-logit+feature
M12-self-distill
M12-distill-then-supervised-finetune
A2S-supervised
A2S-C3-distill
MLP-C3-distill
```

### 13.3 必须记录字段

```text
candidate
teacher_candidate
teacher_used
distill_loss
supervised_loss
feature_loss
teacher_student_KL
teacher_student_logit_cos
feature_CKA
val_acc
test_acc
val_gap_vs_MLP
val_gap_vs_C3
ECE
NLL
distill_improvement_vs_supervised
memory_ratio_inference
step_ratio_inference
memory_ratio_training
step_ratio_training
```

### 13.4 判断标准

Teacher dependency strong：

$$
Acc_{M12-distill}-Acc_{M12-supervised}\geq0.010.
$$

Teacher dependency medium：

$$
0.005\leq Acc_{M12-distill}-Acc_{M12-supervised}<0.010.
$$

Teacher dependency weak：

$$
Acc_{M12-distill}-Acc_{M12-supervised}<0.005.
$$

Teacher-free M12 pass：

$$
Acc_{M12-supervised}-Acc_{MLP}\geq0.015.
$$

If teacher dependency is strong, distillation must be declared part of the official training recipe.

### 13.5 可视化

```text
p3_teacher_dependency_bar.svg
p3_teacher_student_KL_curve.svg
p3_feature_CKA_heatmap.svg
p3_distill_vs_supervised_efficiency.svg
```

---

## 14. P4：M12 mechanism ablation

### 14.1 目的

解释 M12 为什么成功。v7.6 不应只报告 M12 成功，而应把成功机制拆清楚。

### 14.2 Candidates

```text
A0-M12-full
A1-M12-no-C3-distill
A2-M12-no-packed-head
A3-M12-old-SiLU-derivative
A4-M12-FP64-SiLU-derivative
A5-M12-no-M5-init
A6-M12-unfused-stack
A7-M12-generic-head-only
A8-M12-packed-head-only
A9-M12-no-stable-owner-layout
```

### 14.3 必须记录字段

```text
candidate
ablation_type
grad_relerr_max
grad_cos_min
macro_val_gap
test_gap
ECE_delta
NLL_delta
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
S2_shape_count
S1_shape_count
head_forward_ms
head_backward_ms
stack_forward_ms
stack_backward_ms
parameter_owner_count
materialized_tensor_count
kernel_count_total
```

### 14.4 判断标准

A component is essential if removing it causes any of:

$$
\Delta Acc_{\text{macro,val}}\leq-0.005.
$$

or:

$$
r_{\text{step}}>1.50.
$$

or:

$$
r_{\text{mem}}>1.05.
$$

or:

$$
\operatorname{grad\_relerr}_{max}>10^{-4}.
$$

Stable SiLU derivative essential if old derivative fails GradPass but stable derivative passes under identical setting.

Packed head essential if no-packed-head increases memory/step or destabilizes GradPass.

Distill essential if no-distill drops macro gap by at least:

$$
0.005.
$$

### 14.5 可视化

```text
p4_mechanism_ablation_task_bar.svg
p4_mechanism_ablation_efficiency_bar.svg
p4_grad_relerr_by_silu_form.svg
p4_parameter_owner_count_vs_step.svg
p4_m12_mechanism_dashboard.svg
```

---

## 15. P5：S1 memory/step margin attribution

### 15.1 目的

M12 已经 S2。P5 专门解释为什么还不是 S1，并定位可修的 1-7% margin。

### 15.2 Shape grid

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
```

### 15.3 必须记录字段

```text
candidate
dataset
batch_size
peak_allocated_MB
peak_reserved_MB
memory_ratio
step_ratio
forward_ratio
backward_ratio
update_ratio
head_temp_MB
stack_temp_MB
manual_cache_MB
optimizer_state_MB
packed_owner_state_MB
allocator_padding_MB
reserved_unallocated_MB
largest_live_tensor_MB
materialized_tensor_count
kernel_count_total
torch_op_count
triton_kernel_count
top1_memory_source
top2_memory_source
top3_memory_source
top1_time_source
top2_time_source
top3_time_source
unknown_memory_fraction
unknown_time_fraction
```

### 15.4 判断标准

Attribution pass：

```text
top3 memory sources identified
unknown_memory_fraction <= 0.10
top3 time sources identified
unknown_time_fraction <= 0.10
kernel_count_total measured
materialized_tensor_count measured
```

Actionable memory source：

$$
\frac{M_{source}}{M_{peak}}\geq0.02.
$$

Actionable time source：

$$
\frac{T_{source}}{T_{step}}\geq0.03.
$$

Because M12 is close to S1, small sources matter.

### 15.5 可视化

```text
p5_s1_memory_waterfall.svg
p5_s1_step_time_waterfall.svg
p5_shape_memory_heatmap.svg
p5_shape_step_heatmap.svg
p5_forward_backward_ratio_breakdown.svg
```

---

## 16. P6：S1 repair package

### 16.1 目的

把 M12 从 FullGridS2 推到 FullGridS1，同时不破坏 macro gap、NLL、GradPass。

### 16.2 Candidates

```text
S1-0-M12-current
S1-1-head-cache-trim
S1-2-packed-owner-state-trim
S1-3-manual-cache-lifetime-trim
S1-4-bs512-head-temp-streaming
S1-5-forward-temp-fusion
S1-6-update-buffer-lifetime-trim
S1-7-allocator-padding-control
S1-8-forward-kernel-count-trim
S1-9-M12-S1-combo
```

### 16.3 必须记录字段

```text
candidate
components
implementation_status
grad_relerr_max
grad_cos_min
macro_val_gap
test_gap
ECE_delta
NLL_delta
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
forward_ratio_mean
backward_ratio_mean
S1_shape_count
S2_shape_count
memory_improvement_vs_M12
step_improvement_vs_M12
macro_delta_vs_M12
```

### 16.4 判断标准

S1 repair useful：

$$
r_{\text{mem,max,new}}<r_{\text{mem,max,M12}}-0.02.
$$

or:

$$
r_{\text{step,max,new}}<r_{\text{step,max,M12}}-0.05.
$$

FullGridS1Pass：

$$
r_{\text{mem,max}}<1.00.
$$

$$
r_{\text{step,max}}\leq1.35.
$$

Task preservation：

$$
\Delta Acc_{\text{macro,val,new}}\geq
\Delta Acc_{\text{macro,val,M12}}-0.003.
$$

If S1 repair reaches S1 but macro falls below:

$$
+0.0200,
$$

it is not final success.

### 16.5 可视化

```text
p6_s1_repair_pareto.svg
p6_s1_shape_pass_heatmap.svg
p6_task_preservation_bar.svg
p6_s1_boundary_plot.svg
```

---

## 17. P7：ValLossAUC / time-to-target verification

### 17.1 目的

判断 M12 是否有 wall-clock 意义，而不只是 final accuracy 更高。

### 17.2 必跑对象

```text
MLP-AdamW
MLP-distill
M12-FINAL
M12-S1-combo, if exists
C3-teacher
```

### 17.3 设置

```text
steps:
  240
  optional 480

logging:
  every 10 steps

seeds:
  0..9
```

### 17.4 必须记录字段

```text
candidate
dataset
seed
wall_clock_time_sec
step
val_loss
val_acc
train_loss
train_acc
ValLossAUC_step
ValLossAUC_time
ValAccAUC_step
ValAccAUC_time
time_to_target_loss
time_to_target_acc
samples_per_second
early_loss_slope
mid_loss_slope
late_loss_slope
```

### 17.5 判断标准

TimeAUCPass：

$$
ValLossAUC_{time,M12}\leq ValLossAUC_{time,MLP}.
$$

Target pass：

$$
time\_to\_target\_acc(M12)\leq1.10\cdot time\_to\_target\_acc(MLP).
$$

Strong convergence pass：

$$
time\_to\_target\_acc(M12)\leq time\_to\_target\_acc(MLP).
$$

If M12 wins final accuracy but loses all time-AUC metrics badly, route becomes:

```text
quality_advantage_but_wallclock_not_yet
```

### 17.6 可视化

```text
p7_val_loss_vs_time.svg
p7_val_acc_vs_time.svg
p7_val_loss_auc_time_bar.svg
p7_time_to_target_bar.svg
p7_step_ratio_vs_step_saving_scatter.svg
```

---

## 18. P8：ECE / NLL / calibration refinement

### 18.1 目的

M12 的 NLL 明显优于 MLP，但 ECE 只是 within gate。P8 判断是否可以在不伤 task/efficiency 的条件下改善 calibration。

### 18.2 Candidates

```text
CAL0-M12-current
CAL1-M12-temperature-posthoc
CAL2-M12-logit-scale-edge-owned
CAL3-M12-weak-geometry-lambda1e-5
CAL4-M12-weak-geometry-lambda1e-4
CAL5-M12-distill-temperature-adjusted
```

### 18.3 必须记录字段

```text
candidate
calibration_method
posthoc_or_train_time
val_acc
test_acc
ECE
AdaptiveECE
NLL
Brier
macro_gap
memory_ratio
step_ratio
temperature_value
logit_norm_mean
confidence_mean
wrong_confidence_mean
```

### 18.4 判断标准

Calibration improvement：

$$
ECE_{new}<ECE_{M12}.
$$

and:

$$
Acc_{new}\geq Acc_{M12}-0.002.
$$

and:

$$
r_{\text{step,new}}\leq1.50.
$$

Strong calibration pass：

$$
ECE_{new}<ECE_{MLP}.
$$

If only posthoc temperature improves ECE, report separately as posthoc calibration, not architecture-level training improvement.

### 18.5 可视化

```text
p8_reliability_diagram.svg
p8_ece_nll_bar.svg
p8_confidence_histogram_correct_wrong.svg
p8_accuracy_calibration_pareto.svg
```

---

## 19. P9：sample efficiency and robustness expansion

### 19.1 目的

判断 M12 是否代表函数空间结构优势，而不是只在当前 fixed train size 上有效。

### 19.2 Data fractions

```text
train_size:
  256
  512
  1024
  1536
  4096, if available
```

### 19.3 Noise settings

```text
label_noise:
  5%
  10%
  20%

input_noise:
  gaussian sigma 0.05
  gaussian sigma 0.10

corruption:
  random erasing small
  mild affine shift
```

### 19.4 必跑对象

```text
MLP-AdamW
MLP-distill
M12-FINAL
M12-S1-combo, if exists
```

### 19.5 必须记录字段

```text
candidate
dataset
seed
train_size
noise_type
noise_level
val_acc
test_acc
val_loss
test_loss
ECE
NLL
accuracy_drop_vs_clean
sample_efficiency_auc
robustness_auc
memory_ratio
step_ratio
time_to_target_acc
```

### 19.6 判断标准

Sample efficiency pass：

$$
AUC_{data,M12}>AUC_{data,MLP}.
$$

Robustness pass：

$$
AccDrop_{M12}<AccDrop_{MLP}
$$

for at least two noise settings.

If M12 only wins clean fixed-size task but loses robustness/sample efficiency, route remains clean-task success, not broad advantage.

### 19.7 可视化

```text
p9_accuracy_vs_train_size.svg
p9_sample_efficiency_auc_bar.svg
p9_accuracy_under_noise.svg
p9_robustness_auc_bar.svg
p9_ece_under_noise.svg
```

---

## 20. P10：scaling smoke

### 20.1 目的

验证 M12 在 hidden width、batch size、larger train split 或 small token/patch task 上是否有合理 scaling。

### 20.2 Settings

```text
hidden_dim:
  48
  64
  96
  128

batch:
  128
  256
  512

optional tasks:
  CIFAR10-small
  EMNIST-subset
  Fashion/KMNIST larger train split
```

### 20.3 必须记录字段

```text
candidate
hidden_dim
batch_size
dataset
train_size
val_acc
test_acc
macro_gap_vs_MLP
memory_ratio
step_ratio
forward_ratio
backward_ratio
S2_pass
S1_pass
samples_per_second
parameter_count
kernel_count_total
```

### 20.4 判断标准

Scaling smoke pass：

$$
\Delta Acc_{\text{macro,val}}\geq0.010.
$$

and:

$$
r_{\text{mem}}\leq1.10.
$$

and:

$$
r_{\text{step}}\leq1.70.
$$

Full scaling pass requires S2:

$$
r_{\text{mem}}\leq1.05,
$$

$$
r_{\text{step}}\leq1.50.
$$

### 20.5 可视化

```text
p10_hidden_scaling_accuracy.svg
p10_hidden_scaling_efficiency.svg
p10_batch_scaling_heatmap.svg
p10_task_scaling_pareto.svg
```

---

## 21. P11：geometry Pareto diagnostic

### 21.1 目的

Geometry 不是 hard gate，但如果要 claim functional KAN advantage，需要判断是否存在 geometry Pareto benefit。

### 21.2 Geometry losses

```text
basis_smoothness
edge_curvature
jacobian_norm
local_lipschitz
coefficient_tv
path_length
```

### 21.3 Lambda sweep

```text
lambda_geo:
  0
  1e-6
  3e-6
  1e-5
  3e-5
  1e-4
```

### 21.4 必须记录字段

```text
candidate
lambda_geo
geometry_loss_type
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
memory_ratio
step_ratio
```

### 21.5 判断标准

Geometry Pareto useful if：

$$
Acc_{\lambda}\geq Acc_{\lambda=0}-0.005.
$$

and one of:

$$
ECE_{\lambda}<ECE_{\lambda=0},
$$

$$
NLL_{\lambda}<NLL_{\lambda=0},
$$

$$
ValLossAUC_{\lambda}<ValLossAUC_{\lambda=0}.
$$

Harmful if:

$$
Acc_{\lambda}<Acc_{\lambda=0}-0.02
$$

and no metric improves.

### 21.6 可视化

```text
p11_geometry_pareto_frontier.svg
p11_accuracy_vs_geometry.svg
p11_ece_nll_vs_geometry.svg
p11_val_loss_auc_vs_geometry.svg
```

---

## 22. P12：candidate co-selection and route decision

### 22.1 Survivor 类型

```text
S0:
  StrictPass + GradPass + FullGridS1Pass + 10seed MacroSignificantPass + TimeAUCPass + FairnessPass

S1:
  StrictPass + GradPass + FullGridS2Pass + 10seed MacroSignificantPass + FairnessPass

S2:
  FullGridS2 + MacroSignificant, but fairness vs MLP-distill unclear

S3:
  MacroSignificant but no TimeAUCPass

S4:
  TimeAUCPass but macro gap unstable

S5:
  MLP-distill closes the gap; teacher objective explains advantage

S6:
  S1 repair hurts expression

S7:
  Scaling/robustness fail

S8:
  Gradient fail

S9:
  no improvement
```

### 22.2 必须记录字段

```text
candidate
survivor_type
strict_pass
grad_pass
macro_10seed_pass
fullgrid_s2_pass
fullgrid_s1_pass
time_auc_pass
fairness_pass
calibration_pass
sample_efficiency_pass
robustness_pass
geometry_pareto_pass
memory_ratio_mean
memory_ratio_max
step_ratio_mean
step_ratio_max
forward_ratio_mean
backward_ratio_mean
val_gap_vs_MLP
test_gap_vs_MLP
val_gap_vs_MLP_distill
CI95_low
Holm_p
ECE_delta
NLL_delta
ValLossAUC_time_delta
route_recommendation
primary_blocker
next_required_implementation
```

### 22.3 Route cases

```text
R1-FullSystemAdvantage:
  S1 + 10seed MacroSignificant + TimeAUCPass + FairnessPass.
  Claim system-level PureKAN-NG advantage.

R2-S2QualityAdvantage:
  S2 + MacroSignificant + FairnessPass, but S1/time AUC incomplete.
  Claim efficient diagnostic success, continue S1/time work.

R3-DistillationObjectiveAdvantage:
  MLP-distill catches up.
  Claim teacher/objective explains much of gain; need architecture-independent comparison.

R4-AccuracyOnlyAdvantage:
  MacroSignificant but time AUC fails.
  Claim quality improvement but not wall-clock system advantage.

R5-S1RepairHurtsExpression:
  Efficiency repair breaks macro.
  Return to dense-preserving S1 design.

R6-ScalingFail:
  Clean small-data success only.
  Need stronger primitive or scaling redesign.

R7-NoReproduction:
  10seed/20seed does not reproduce v7.5.
  Treat v7.5 as promising but unstable.
```

### 22.4 可视化

```text
p12_system_pareto_accuracy_efficiency_time.svg
p12_survivor_type_dashboard.svg
p12_route_decision_tree.svg
p12_final_scorecard.svg
```

---

## 23. P13：artifact and failure audit

### 23.1 Required artifacts

```text
run_manifest.json
provenance_audit.csv
candidate_registry.csv
gate_config.json
p0_reproduction.csv
p1_10seed_significance.csv
p2_fairness_mlp_distill.csv
p3_teacher_dependency.csv
p4_m12_mechanism_ablation.csv
p5_s1_attribution.csv
p6_s1_repair.csv
p7_val_loss_auc_time.csv
p8_calibration_refinement.csv
p9_sample_robustness.csv
p10_scaling_smoke.csv
p11_geometry_pareto.csv
p12_candidate_selection.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

### 23.2 Failure taxonomy

```text
F1_10seed_macro_fail
F2_mlp_distill_closes_gap
F3_teacher_free_fail
F4_time_auc_fail
F5_s1_memory_fail
F6_s1_step_fail
F7_s1_repair_hurts_expression
F8_forward_ratio_fail
F9_calibration_fail
F10_scaling_fail
F11_robustness_fail
F12_geometry_hurts_expression
F13_gradient_correctness_fail
F14_fake_or_proxy_violation
F15_artifact_missing
```

### 23.3 Required figures

```text
figures/p12_system_pareto_accuracy_efficiency_time.svg
figures/p12_final_scorecard.svg
figures/p12_route_decision_tree.svg
figures/failure_taxonomy_heatmap.svg
```

---

## 24. 第一轮推荐执行顺序

v7.6 第一轮不要全量展开。最重要的是先确认 v7.5 最低成功是否能升级成稳定系统结论。

### Step 1：P1 10-seed confirm

先确认 M12 的 macro gap 是否稳定超过 `+0.0200`。如果 10-seed 不稳，不要做 S1 和 scaling。

### Step 2：P2 fairness vs MLP-distill

如果 MLP-distill 追平 M12，那么下一步应研究 distillation objective，而不是宣称 KAN architecture 全面优势。

### Step 3：P7 time-to-target / AUC

如果 M12 final accuracy win 但 wall-clock AUC 很差，系统 claim 必须降级。

### Step 4：P5-P6 S1 repair

只有在 M12 10-seed 和 fairness pass 后，S1 repair 才值得投入。否则 S1 repair 可能只是优化一个不稳的候选。

### Step 5：P4 mechanism ablation

如果 P1/P2 通过，做 mechanism ablation，把 M12 成功原因写清楚。

---

## 25. 停止条件

### 25.1 成功停止

出现以下任一情况立即写复盘：

```text
10seed MacroSignificantPass + FullGridS2Pass + FairnessPass
10seed MacroSignificantPass + FullGridS1Pass + FairnessPass
10seed MacroSignificantPass + FullGridS1Pass + FairnessPass + TimeAUCPass
```

### 25.2 失败停止

出现以下情况停止对应路线：

```text
1. 10seed macro gap < +0.015；
2. MLP-distill 与 M12 gap < +0.005；
3. M12 teacher-free 与 M12-distill 差距 > +0.010，且 MLP-distill 追平；
4. S1 repair 让 macro gap 掉到 < +0.018；
5. time AUC 比 MLP 差 >10%；
6. GradPass 任一 route candidate fail 且无法用等价数值重排修复；
7. no-fake/no-proxy audit fail。
```

---

## 26. 成功与失败解释规则

### Case A：M12 10-seed 稳定，MLP-distill 追不上

说明 M12 不是单纯 teacher objective 的产物，而是 KAN architecture + distillation reachability 的优势。此时可以写：

```text
M12 has a robust architecture-conditioned advantage under the current task family.
```

### Case B：M12 10-seed 稳定，但 MLP-distill 追平

说明 v7.5 的优势很大部分来自 C3 teacher / distillation objective。M12 仍是 strict efficient implementation success，但不能宣称 architecture 独有优势。

### Case C：M12 10-seed 不稳定

说明 v7.5 是一个真实但边际较薄的阶段性成功。下一步应回到 effective expressivity margin，而不是直接 S1 repair。

### Case D：S1 repair 成功且 macro 保持

这是系统突破：从 S2 diagnostic 进入 S1 official candidate。

### Case E：S1 repair 破坏 macro

说明 M12 的剩余 live-set 或 forward path 可能承载 effective expressivity / numerical stability。必须回到 dense-preserving S1 design。

### Case F：TimeAUCPass 失败

M12 是 quality win，不是 wall-clock win。系统 claim 必须降级为 accuracy/NLL advantage。

### Case G：Scaling/robustness fail

M12 可能是当前 small-task/small-data 的结构成功，不能外推到更广任务。

---

## 27. 最终建议

v7.6 的一句话策略是：

$$
\boxed{
\text{把 M12 从 v7.5 最低成功，升级为可复现、公平、wall-clock 有意义、可解释的 MLP-AdamW 系统优势。}
}
$$

当前最准确的状态是：

```text
1. v7.5 达成最低成功标准。
2. M12 对 B0/MLP-AdamW reference 有 macro significant accuracy advantage。
3. M12 full-grid S2 成立，9/9 shapes pass。
4. M12 还不是 S1。
5. M12 的 ECE 只是 within gate，不是更优。
6. M12 使用 C3 distillation，必须做 MLP-distill fairness。
7. M12 还缺 time-to-target / ValLossAUC_time。
8. M12 的 generalization / scaling / robustness 还没证明。
```

因此，下一步不是小修小补，而是系统确认：

$$
\boxed{
\text{reproducibility}
+
\text{fairness}
+
\text{S1 margin}
+
\text{time-to-target}
+
\text{mechanism}
+
\text{scaling}
}
$$

只有这些闭合，才能从：

```text
M12 is the first S2-MacroSignificant strict PureKAN-NG candidate.
```

升级为：

```text
M12 establishes a robust system-level advantage over MLP-AdamW.
```
