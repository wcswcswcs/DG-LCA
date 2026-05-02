# DG-KAN Stage C-next v2 详细实验计划：放大非 Identity 修正并重新检验 Router 价值

> 文件名：`DG-KAN_StageC-next_v2_详细实验计划.md`  
> 对象：DG-KAN / DG-LCA 的 Stage C-next v2 补充实验  
> 定位：实验执行协议，不涉及具体代码实现  
> 公式格式：Typora 友好，统一使用 `$...$` 与 `$$...$$`  
> 核心目标：在上一轮 Stage C-next 发现 **identity baseline 过强、correction router 未通过 gate** 之后，重新构造更能暴露非 identity 修正项价值的实验条件，检验 learned router 是否在更深、更难、更大 span、更强 branch 的 DG-KAN 中具有真实必要性。

---

## 0. 背景与本轮实验要回答的问题

上一轮 Stage C-next 的核心发现是：在当前 residual DG-KAN 设定下，单层 local VJP 非常接近 identity。真实输入端 credit 可以写成：

$$
g_h = g_{out} + \Delta g
$$

其中：

$$
\Delta g = g_h - g_{out}
$$

上一轮已经将 router 目标从直接预测完整 $g_h$ 改成预测非 identity 修正项 $\Delta g$。这个设计是正确的，但实验显示：

1. 单层 correction ratio 在 Pareto regime 中通常只有 $0.16$ 到 $0.20$ 左右；
2. `scalar_correction` 只能带来 $7\%$ 到 $12\%$ 的相对增益，没有达到 $>20\%$ 的 gate；
3. ridge / low-rank correction 有一点方向信号，但 norm miscalibration 严重，反而让 full relerr 变差；
4. depth 从 $2$ 加到 $8$ 后，identity relerr 没有升高，反而下降；
5. KMNIST / EMNIST 等复杂数据确实让 AdamW 的 correction 更大，但原 Pareto 超参迁移后 accuracy gap 太大；
6. one-step sanity 中，oracle correction 相比 identity 只有约 $2.5\%$ 的实际 loss descent 增益；
7. N5-lite joint-training precheck 中，identity adjoint 与 full / analytic adjoint 的训练结果非常接近，accuracy gap 只有 $0\%$ 到 $0.5\%$。

这说明：

$$
\boxed{
\text{当前单层 residual DG-KAN 的 non-identity correction 太小，learned router 很难证明必要性。}
}
$$

但是，这还不能直接说明 learned router 没有研究价值。更合理的怀疑是：

$$
\boxed{
\text{当前模型太浅、数据太简单、routing unit 太细，导致 identity baseline 过强。}
}
$$

因此 Stage C-next v2 的目标不是继续重复单层 correction router，而是系统回答：

> **在更深模型、更复杂数据集、更大 span、更强 branch 但几何仍可控的条件下，identity baseline 是否会变弱？learned residual correction router 是否能稳定打败 identity？**

---

## 1. 本轮实验的核心假设

Stage C-next v2 要检验四个假设。

### 假设 H1：单层 router 粒度太细

当前单个 residual block 是：

$$
h_{k+1}=h_k+\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))
$$

每层都是 $I + \text{small correction}$，所以单层 VJP 很接近 identity：

$$
g_k \approx g_{k+1}
$$

如果把 routing unit 从单层改成 span：

$$
h_{k+s}=F_{k+s-1}\circ\cdots\circ F_k(h_k)
$$

则对应 VJP 为：

$$
g_k=J_{k:k+s}^{\top}g_{k+s}
$$

随着 $s$ 增大，非 identity correction 可能累积，identity baseline 可能变弱。

本轮将重点测试：

$$
s\in\{1,2,4,8\}
$$

其中 $s=1$ 是上一轮单层 setting，$s=2,4$ 是主要测试对象，$s=8$ 只作为 stress test。

---

### 假设 H2：复杂数据会增加 non-identity correction，但需要重新寻找 Pareto regime

上一轮 KMNIST / EMNIST 显示 AdamW 的 correction ratio 明显更大，但原 MNIST Pareto 超参直接迁移后 accuracy gap 达到 $3.6\%$ 到 $5.2\%$，不适合作为 router 主结论。

因此本轮不直接使用旧 Pareto 设置，而是在每个复杂数据集上重新搜索：

$$
\text{test gap}<2\%
$$

同时保持：

$$
0.4 < \text{branch / AdamW} < 0.8
$$

$$
\text{amp p95}<1.5
$$

$$
\text{J cond / AdamW}<1.1
$$

满足这些条件的 checkpoint 才进入 router distillation。

---

### 假设 H3：router 失败主要来自 norm miscalibration，而不是完全没有方向信号

上一轮 ridge / low-rank correction 的 correction cosine 有一定信号，但 correction norm 过大，导致 full relerr 恶化。因此本轮 router 不再只优化 correction MSE，而是显式拆成：

1. correction direction；
2. correction norm；
3. full VJP improvement over identity；
4. one-step actual descent gain。

新的 correction router 应写为：

$$
\hat g_h = g_{out} + \widehat{\Delta g}
$$

其中：

$$
\widehat{\Delta g}=\hat s(h,g_{out})\cdot \hat u(h,g_{out})
$$

并要求：

$$
\|\hat u\|\approx 1
$$

$$
\hat s\approx \|\Delta g^{teacher}\|
$$

这样可以避免方向略对但 norm 爆炸的问题。

---

### 假设 H4：如果所有稳定 setting 中 identity 仍然足够好，那么 learned router 应降级

如果在更深、更难、更大 span 条件下仍然满足：

$$
\operatorname{relerr}_{identity}<0.2
$$

并且 identity adjoint joint training 仍接近 analytic adjoint，则说明 residual DG-KAN 的局部 credit transport 本身就是 identity-dominant。此时 learned router 不应作为主线，Stage D 应优先推进：

$$
\boxed{
\text{IdentityAdj / AnalyticAdj + functional update + memory accounting}
}
$$

---

## 2. 本轮实验总路线

Stage C-next v2 分为八个阶段。

1. **CNv2-0：复现与数据准备**  
   复核上一轮 MNIST / Fashion 结论，并准备更复杂数据集与模型配置。

2. **CNv2-1：Hard-dataset Pareto search**  
   在 KMNIST、EMNIST Balanced、CIFAR-10 small 上重新寻找满足性能与几何约束的 Pareto checkpoint。

3. **CNv2-2：Span-VJP geometry audit**  
   对 $s=1,2,4,8$ 的 span VJP 进行 identity relerr、correction ratio、amplification、noise gain、credit rank 审计。

4. **CNv2-3：Norm-calibrated residual correction router**  
   使用 direction + norm 分解设计 correction router，避免上一轮 norm miscalibration。

5. **CNv2-4：Span correction router distillation**  
   在 span-level 目标上训练 router，检验 router 是否能打败 span identity baseline。

6. **CNv2-5：One-step 与 micro-training usefulness**  
   检查 router credit 是否真的比 identity 带来更好的 actual descent 和短程训练表现。

7. **CNv2-6：Identity / analytic / router joint-training precheck**  
   用更深模型和更复杂数据复查 identity adjoint 是否仍接近 analytic adjoint。

8. **CNv2-7：Residual stress ablation**  
   不去掉 residual 作为主线，但加入 scaled residual、gated residual、non-residual KAN 作为诊断。

---

## 3. 关键 notation 与指标定义

### 3.1 单层 VJP 与 correction

单层 residual block：

$$
h_{k+1}=F_k(h_k)
$$

真实 VJP：

$$
g_k^{teacher}=J_k^{\top}g_{k+1}
$$

identity baseline：

$$
g_k^{id}=g_{k+1}
$$

非 identity correction：

$$
\Delta g_k^{teacher}=g_k^{teacher}-g_{k+1}
$$

correction ratio：

$$
r_{corr,k}=
\frac{
\|\Delta g_k^{teacher}\|
}{
\|g_k^{teacher}\|+\epsilon
}
$$

identity relative error：

$$
\operatorname{relerr}_{id,k}
=
\frac{
\|g_{k+1}-g_k^{teacher}\|
}{
\|g_k^{teacher}\|+\epsilon
}
$$

---

### 3.2 Span VJP

span $s$ 表示从第 $k$ 层跨到第 $k+s$ 层：

$$
F_{k:k+s}=F_{k+s-1}\circ\cdots\circ F_k
$$

span teacher VJP：

$$
g_k^{teacher,s}=J_{k:k+s}^{\top}g_{k+s}
$$

span identity baseline：

$$
g_k^{id,s}=g_{k+s}
$$

span correction：

$$
\Delta g_k^{teacher,s}=g_k^{teacher,s}-g_{k+s}
$$

span identity relerr：

$$
\operatorname{relerr}_{id}^{s}=
\frac{
\|g_{k+s}-g_k^{teacher,s}\|
}{
\|g_k^{teacher,s}\|+\epsilon
}
$$

span correction ratio：

$$
r_{corr}^{s}=
\frac{
\|\Delta g_k^{teacher,s}\|
}{
\|g_k^{teacher,s}\|+\epsilon
}
$$

---

### 3.3 Router gain over identity

如果 router 给出：

$$
\hat g_k=g_{k+s}+\widehat{\Delta g}_k
$$

router full relerr 为：

$$
\operatorname{relerr}_{router}=
\frac{
\|\hat g_k-g_k^{teacher,s}\|
}{
\|g_k^{teacher,s}\|+\epsilon
}
$$

相对 identity 的增益：

$$
\operatorname{gain}_{rel}=
\frac{
\operatorname{relerr}_{id}^{s}-\operatorname{relerr}_{router}
}{
\operatorname{relerr}_{id}^{s}+\epsilon
}
$$

beat rate：

$$
\operatorname{beat\_rate}
=
\mathbb P
\left(
\operatorname{relerr}_{router}<\operatorname{relerr}_{id}
\right)
$$

---

### 3.4 One-step usefulness

对 hidden state 做一次小更新：

$$
h\leftarrow h-\eta \hat g
$$

actual loss delta：

$$
\Delta L_{actual}=L_{after}-L_{before}
$$

相对 identity 的 one-step gain：

$$
\operatorname{gain}_{step}=
\frac{
\Delta L_{identity}-\Delta L_{router}
}{
|\Delta L_{identity}|+\epsilon
}
$$

因为 $\Delta L$ 越负越好，所以如果 router 让 loss 降得更多，则 $\operatorname{gain}_{step}>0$。

---

### 3.5 Branch ratio 与 regime

forward branch ratio：

$$
r_{branch,k}=
\frac{
\|\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))\|
}{
\|h_k\|+\epsilon
}
$$

相对 AdamW 的 branch strength：

$$
\text{branch / AdamW}=
\frac{
 r_{branch}^{method}
}{
 r_{branch}^{AdamW}
}
$$

本轮 regime 不用单一超参定义，而用结果指标定义：

| regime | 判定标准 |
|---|---|
| conservative | branch / AdamW $<0.3$，几何稳定但 branch under-active |
| Pareto | test gap $<2\%$，branch / AdamW 在 $0.4$ 到 $0.8$，几何优于 AdamW |
| active | branch / AdamW 在 $0.8$ 到 $1.2$，精度追平但几何优势减弱 |
| over-active | branch / AdamW $>2$ 或 Jacobian / $\phi'$ 明显爆炸 |

---

## 4. 全局实验配置与 W&B config

每个 run 必须记录以下 config。

| config key | 含义 |
|---|---|
| `exp/stage` | 固定为 `stage_c_next_v2` |
| `exp/phase` | `CNv2_0` 到 `CNv2_7` |
| `exp/seed` | 随机种子 |
| `data/name` | `mnist`, `fashion_mnist`, `kmnist`, `emnist_balanced`, `cifar10_small` |
| `data/train_size` | 训练集大小 |
| `data/val_size` | 验证集大小 |
| `data/test_size` | 测试集大小 |
| `model/depth` | block 数 |
| `model/hidden_dim` | hidden dimension |
| `model/basis_count` | KAN basis 数量 |
| `model/kan_connectivity` | `dense`, `grouped`, `sparse` |
| `model/residual_type` | `standard`, `scaled`, `gated`, `non_residual` |
| `model/alpha` | residual scale |
| `model/branch_target` | 若使用 branch-ratio targeting，则记录目标值 |
| `update/type` | `adamw`, `diag_to_sobolev`, `sobolev`, `diag` |
| `update/coeff_lr` | KAN coefficient learning rate |
| `update/rest_lr` | 非 KAN 参数 learning rate |
| `update/warmup_frac` | diag-to-Sobolev warmup fraction |
| `update/sobolev_alpha` | Sobolev alpha |
| `update/sobolev_beta` | Sobolev beta |
| `update/rho` | damping |
| `credit/span` | $s=1,2,4,8$ |
| `router/type` | `zero`, `scalar`, `norm_calibrated_scalar`, `ridge`, `diag`, `diag_lowrank`, `direction_norm`, `derivative_informed` |
| `router/rank` | low-rank rank |
| `router/norm_clip_tau` | correction norm trust-region 上限 |
| `router/loss_type` | `mse`, `direction_norm`, `full_gain`, `descent_aware` |
| `audit/batch_size` | audit batch size |
| `resource/device` | GPU 型号 |
| `resource/precision` | `fp32`, `bf16`, `fp16` |

---

## 5. CNv2-0：复现与准备

### 5.1 目标

CNv2-0 用于复现上一轮 Stage C-next 结果，并确保新实验环境与历史结论一致。它不是主实验，但如果复现失败，后续结果无法解释。

必须复现三件事：

1. MNIST / Fashion 的 identity baseline 很强；
2. Pareto regime 的 test gap 小，同时几何优于 AdamW；
3. correction router 上一轮没有稳定打败 identity。

---

### 5.2 数据集与模型

使用：

```text
MNIST small
Fashion-MNIST small
```

模型：

```text
depth = 2
hidden_dim = 32
basis_count = 8
residual_type = standard
```

seeds：

```text
0,1,2,3,4
```

---

### 5.3 必须记录的指标

| metric key | 含义 |
|---|---|
| `cnv2_0/test_acc` | 测试准确率 |
| `cnv2_0/test_gap_vs_adamw` | 相对 AdamW 的 test gap |
| `cnv2_0/branch_over_adamw` | branch / AdamW |
| `cnv2_0/phi_prime_over_adamw` | $\phi'_{p95}$ / AdamW |
| `cnv2_0/jcond_over_adamw` | Jacobian condition / AdamW |
| `cnv2_0/identity_relerr` | identity relerr |
| `cnv2_0/correction_ratio` | correction ratio |
| `cnv2_0/amp_p95` | credit amplification p95 |
| `cnv2_0/no_kan_acc_drop` | no-KAN ablation drop |

---

### 5.4 成功标准

CNv2-0 通过需要满足：

1. MNIST / Fashion 上 Pareto regime 的 test gap 约小于 $1\%$ 到 $2\%$；
2. Pareto regime 的 $\phi'_{p95}$ 和 Jacobian condition 低于 AdamW；
3. identity relerr 与上一轮大致一致；
4. active regime 的 correction ratio 明显高于 Pareto，但几何变差。

如果 CNv2-0 失败，必须先检查训练代码、数据 split、随机种子、functional update 超参，不进入后续实验。

---

## 6. CNv2-1：Hard-dataset Pareto search

### 6.1 目标

CNv2-1 解决上一轮数据复杂度不足的问题。目标不是直接训练 router，而是在更复杂数据集上重新找到可用的 Pareto regime。

复杂数据集包括：

| 数据集 | 作用 |
|---|---|
| Fashion-MNIST | 继续作为中等难度主验证 |
| KMNIST | 比 MNIST 更难，上一轮 correction ratio 更大 |
| EMNIST Balanced | 类别更多，更容易产生复杂 credit |
| CIFAR-10 small | 视觉结构更复杂，需要 ConvStem-DG-KAN |

CIFAR-10 small 不作为第一批 gate，只有 KMNIST / EMNIST 的 Pareto search 通过后进入。

---

### 6.2 搜索空间

第一轮搜索：

```text
dataset = Fashion-MNIST, KMNIST, EMNIST Balanced
hidden_dim = 64
basis_count = 16
depth = 4
residual_type = standard
```

优化超参：

```text
coeff_lr = 0.3, 0.5, 0.7, 1.0
rest_lr = 0.001, 0.003
warmup_frac = 0.10, 0.25
alpha = 1.0, 1.5
```

第二轮只保留第一轮中接近 Pareto 的配置，加 seeds 到 $5$。

---

### 6.3 Pareto 判定标准

硬数据集上的 Pareto checkpoint 需要满足：

$$
\text{test gap vs AdamW}<2\%
$$

$$
0.4<\text{branch / AdamW}<0.8
$$

$$
\text{amp p95}<1.5
$$

$$
\text{J cond / AdamW}<1.1
$$

$$
\phi'_{p95}/\phi'_{p95}^{AdamW}<1.0
$$

如果无法同时满足，则记录为 `pareto_not_found`，不能进入 router 主实验。

---

### 6.4 W&B 指标

| metric key | 含义 |
|---|---|
| `cnv2_1/train_acc` | train accuracy |
| `cnv2_1/test_acc` | test accuracy |
| `cnv2_1/test_gap_vs_adamw` | test gap |
| `cnv2_1/loss_auc_val` | validation loss AUC |
| `cnv2_1/branch_over_adamw` | branch / AdamW |
| `cnv2_1/no_kan_acc_drop` | no-KAN ablation drop |
| `cnv2_1/phi_prime_p95` | $\phi'_{p95}$ |
| `cnv2_1/phi_prime_over_adamw` | $\phi'_{p95}$ ratio |
| `cnv2_1/jacobian_condition` | Jacobian condition |
| `cnv2_1/jcond_over_adamw` | condition ratio |
| `cnv2_1/identity_relerr_s1` | single-block identity relerr |
| `cnv2_1/correction_ratio_s1` | single-block correction ratio |
| `cnv2_1/amp_p95_s1` | single-block amp p95 |
| `cnv2_1/pareto_score` | 综合 Pareto score |

Pareto score 可以定义为：

$$
S_{pareto}
=
\text{test\_gap}
+
0.5\max(0,\text{Jcond ratio}-1)
+
0.5\max(0,\phi' \text{ ratio}-1)
+
0.2|\text{branch ratio}-0.6|
$$

越低越好。

---

### 6.5 可视化

W&B 必须有以下图：

1. **Pareto search scatter**  
   x-axis: test gap；y-axis: J cond / AdamW；color: branch / AdamW；size: $\phi'$/AdamW。

2. **Branch vs accuracy plot**  
   x-axis: branch / AdamW；y-axis: test acc；color: dataset。

3. **Geometry vs accuracy Pareto front**  
   x-axis: J cond / AdamW；y-axis: test gap；color: coeff lr。

4. **Heatmap: coeff lr × rest lr**  
   color: Pareto score。

5. **no-KAN ablation plot**  
   x-axis: branch / AdamW；y-axis: no-KAN acc drop。

---

## 7. CNv2-2：Span-VJP geometry audit

### 7.1 目标

CNv2-2 是本轮最重要的诊断实验。它测试：当 routing unit 从单层变成 span 时，identity baseline 是否变弱。

span 设置：

$$
s\in\{1,2,4,8\}
$$

其中：

- $s=1$：上一轮单层 setting；
- $s=2$：最小 span router；
- $s=4$：主 stress setting；
- $s=8$：只在深度足够且几何稳定时做。

---

### 7.2 实验输入

使用 CNv2-1 找到的 Pareto checkpoint，同时保留 AdamW 和 active checkpoint 作为对照。

推荐 first pass：

```text
dataset = Fashion-MNIST, KMNIST
model depth = 4, 8
hidden_dim = 64
basis_count = 16
regime = AdamW, Pareto, active
span = 1, 2, 4
seeds = 0,1,2,3,4
```

只有当 $s=4$ 下几何仍稳定，才做 $s=8$。

---

### 7.3 核心指标

| metric key | 含义 |
|---|---|
| `cnv2_2/span_{s}/identity_cos` | span identity cosine |
| `cnv2_2/span_{s}/identity_relerr` | span identity relerr |
| `cnv2_2/span_{s}/correction_ratio` | span correction ratio |
| `cnv2_2/span_{s}/amp_mean` | span amplification mean |
| `cnv2_2/span_{s}/amp_p95` | span amplification p95 |
| `cnv2_2/span_{s}/noise_gain_0_05` | noise gain $\sigma=0.05$ |
| `cnv2_2/span_{s}/noise_gain_0_1` | noise gain $\sigma=0.1$ |
| `cnv2_2/span_{s}/credit_PR` | credit participation rank |
| `cnv2_2/span_{s}/rank90` | credit rank90 |
| `cnv2_2/span_{s}/rank95` | credit rank95 |
| `cnv2_2/span_{s}/backward_branch_ratio` | backward branch correction size |

---

### 7.4 Router-worthy setting 判定

一个 setting 被认为值得训练 router，需要满足：

$$
\operatorname{relerr}_{identity}^{span}>0.25
$$

$$
r_{corr}^{span}>0.30
$$

$$
\text{amp p95}<1.5
$$

$$
\text{test gap}<2\%
$$

如果一个 setting 的 identity relerr 很低，例如：

$$
\operatorname{relerr}_{identity}^{span}<0.15
$$

则不进入 router 训练，因为 identity baseline 太强，router 很难产生实质价值。

---

### 7.5 可视化

1. **Span vs identity relerr**  
   x-axis: span $s$；y-axis: identity relerr；color: regime。

2. **Span vs correction ratio**  
   x-axis: span $s$；y-axis: correction ratio；color: dataset。

3. **Span geometry heatmap**  
   rows: dataset × regime；columns: span；color: amp p95。

4. **Router-worthy map**  
   scatter: identity relerr vs amp p95；highlight settings satisfying router-worthy criteria。

5. **Credit rank over span**  
   x-axis: span；y-axis: PR / rank90。

---

## 8. CNv2-3：Norm-calibrated residual correction router

### 8.1 目标

上一轮 ridge / low-rank correction 的主要失败原因是 norm miscalibration。本轮 router 必须显式校准 correction norm。

Teacher：

$$
\Delta g^{teacher}=g_k^{teacher,s}-g_{k+s}
$$

Router 输出：

$$
\widehat{\Delta g}=\hat s\cdot \hat u
$$

其中：

$$
\|\hat u\|\approx1
$$

$$
\hat s\approx\|\Delta g^{teacher}\|
$$

最终 credit：

$$
\hat g_k=g_{k+s}+\widehat{\Delta g}
$$

---

### 8.2 Router 类型

| router | 说明 |
|---|---|
| zero correction | 等价 identity baseline |
| scalar correction | $\widehat{\Delta g}=c\cdot g_{out}$ 或 learned scalar |
| norm-calibrated scalar | scalar direction + learned norm predictor |
| ridge correction | centered adaptive ridge |
| clipped ridge correction | ridge + norm clipping |
| diag correction | conditional diagonal correction |
| diag-lowrank correction | diagonal + low-rank direction |
| direction-norm router | separate direction head and norm head |
| derivative-informed correction | 使用 $\phi'$ / branch features 作为条件 |
| analytic oracle | teacher correction |

---

### 8.3 Router loss

Direction loss：

$$
\mathcal L_{dir}=1-
\frac{
\langle \widehat{\Delta g},\Delta g^{teacher}\rangle
}{
\|\widehat{\Delta g}\|\|\Delta g^{teacher}\|+\epsilon
}
$$

Norm log loss：

$$
\mathcal L_{norm}=\left(
\log(\|\widehat{\Delta g}\|+\epsilon)
-
\log(\|\Delta g^{teacher}\|+\epsilon)
\right)^2
$$

Full VJP loss：

$$
\mathcal L_{full}=\frac{
\|g_{k+s}+\widehat{\Delta g}-g_k^{teacher,s}\|^2
}{
\|g_k^{teacher,s}\|^2+\epsilon
}
$$

Total loss：

$$
\mathcal L
=
\lambda_{dir}\mathcal L_{dir}
+
\lambda_{norm}\mathcal L_{norm}
+
\lambda_{full}\mathcal L_{full}
$$

推荐 sweep：

```text
lambda_dir = 1.0
lambda_norm = 0.1, 0.3, 1.0
lambda_full = 0.1, 1.0
```

---

### 8.4 Trust region 与 clipping

为防止 norm 爆炸，加入：

$$
\|\widehat{\Delta g}\| \leq \tau \|g_{k+s}\|
$$

建议：

$$
\tau\in\{0.2,0.5,1.0\}
$$

记录 clipping 频率：

| metric key | 含义 |
|---|---|
| `cnv2_3/router/clip_rate` | correction 被 clipping 的比例 |
| `cnv2_3/router/pre_clip_norm_ratio` | clipping 前 norm ratio |
| `cnv2_3/router/post_clip_norm_ratio` | clipping 后 norm ratio |

---

### 8.5 Router 指标

| metric key | 含义 |
|---|---|
| `cnv2_3/router/full_cos` | full credit cosine |
| `cnv2_3/router/full_relerr` | full credit relerr |
| `cnv2_3/router/gain_rel_vs_identity` | 相对 identity 的 relerr gain |
| `cnv2_3/router/beat_rate` | 单样本 beat identity 比例 |
| `cnv2_3/router/correction_cos` | correction direction cosine |
| `cnv2_3/router/correction_relerr` | correction relerr |
| `cnv2_3/router/correction_norm_ratio` | predicted / teacher correction norm |
| `cnv2_3/router/correction_norm_log_error` | norm log error |
| `cnv2_3/router/adjoint_residual_norm` | normalized adjoint residual |
| `cnv2_3/router/linearity_residual` | router linearity residual |
| `cnv2_3/router/time_ms` | router time |
| `cnv2_3/router/memory_mb` | router memory |

---

### 8.6 成功标准

中等成功：

$$
\operatorname{gain}_{rel}>0.2
$$

$$
\cos_{corr}>0.7
$$

$$
0.8<\text{correction norm ratio}<1.2
$$

$$
\operatorname{beat\_rate}>0.6
$$

强成功：

$$
\operatorname{gain}_{rel}>0.3
$$

$$
\cos_{corr}>0.8
$$

$$
\operatorname{beat\_rate}>0.7
$$

并且 one-step gain over identity $>5\%$。

如果 router 只在 full cosine 上表现好，但 correction cosine 低、norm ratio 失控，则判定为失败。

---

## 9. CNv2-4：Span correction router distillation

### 9.1 目标

CNv2-4 将 CNv2-3 的 correction router 扩展到 span-level VJP。

对于 span $s$：

$$
\hat g_k=g_{k+s}+\widehat{\Delta g}^{s}
$$

其中：

$$
\Delta g^{s}=g_k^{teacher,s}-g_{k+s}
$$

---

### 9.2 训练设置

仅对 CNv2-2 中满足 router-worthy 的 setting 训练 router。若没有 setting 满足：

$$
\operatorname{relerr}_{identity}^{span}>0.25
$$

则记录 `no_router_worthy_setting`，不强行训练。

训练 / eval split：

| split | 说明 |
|---|---|
| train-same | 同 dataset / same regime / same span |
| val-same | heldout batch |
| test-same | same setting heldout |
| test-cross-seed | same setting different seed |
| test-cross-regime | train Pareto, eval active / AdamW |
| test-cross-span | train span 2, eval span 4 |
| test-random-credit | random Gaussian credit |
| test-mixed-credit | task + random mixed credit |

---

### 9.3 指标

| metric key | 含义 |
|---|---|
| `cnv2_4/span_{s}/router_full_relerr` | span router full relerr |
| `cnv2_4/span_{s}/identity_relerr` | span identity relerr |
| `cnv2_4/span_{s}/gain_rel` | router gain over identity |
| `cnv2_4/span_{s}/correction_cos` | correction cosine |
| `cnv2_4/span_{s}/correction_norm_ratio` | norm ratio |
| `cnv2_4/span_{s}/beat_rate` | beat identity rate |
| `cnv2_4/span_{s}/cross_seed_cos` | cross seed fidelity |
| `cnv2_4/span_{s}/cross_regime_cos` | cross regime fidelity |
| `cnv2_4/span_{s}/cross_span_cos` | cross span fidelity |
| `cnv2_4/span_{s}/ood_random_cos` | random credit OOD |
| `cnv2_4/span_{s}/ood_mixed_cos` | mixed credit OOD |

---

### 9.4 可视化

1. **Span router gain heatmap**  
   rows: dataset × regime；columns: span；color: gain over identity。

2. **Correction cosine vs span**  
   x-axis: span；y-axis: correction cosine；color: router type。

3. **Norm calibration scatter**  
   x-axis: teacher correction norm；y-axis: predicted correction norm。

4. **Cross-span matrix**  
   train span × eval span，颜色为 gain / cosine。

5. **Cross-regime matrix**  
   train regime × eval regime，颜色为 correction cosine。

---

## 10. CNv2-5：One-step 与 micro-training usefulness

### 10.1 目标

CNv2-5 不只看 fidelity，还要看 router credit 是否能改善实际训练方向。

先做 one-step hidden-state update，再做 short micro-training precheck。

---

### 10.2 One-step update

比较 credit source：

| source | 含义 |
|---|---|
| analytic oracle | exact span VJP |
| identity | $g_{k+s}$ |
| scalar correction | 上轮稳定 baseline |
| norm-calibrated router | 本轮主方法 |
| ridge / lowrank clipped | 结构化 router |
| random credit | 负对照 |

记录：

| metric key | 含义 |
|---|---|
| `cnv2_5/one_step/loss_before` | update 前 loss |
| `cnv2_5/one_step/loss_after` | update 后 loss |
| `cnv2_5/one_step/actual_delta` | actual loss delta |
| `cnv2_5/one_step/gain_vs_identity` | 相对 identity 的下降增益 |
| `cnv2_5/one_step/sign_agreement` | 是否下降 |
| `cnv2_5/one_step/update_norm` | hidden update norm |
| `cnv2_5/one_step/phi_prime_after` | update 后 $\phi'$ |
| `cnv2_5/one_step/jcond_after` | update 后 J condition |

成功标准：

$$
\operatorname{gain}_{step}>0.05
$$

并且 sign agreement 不低于 identity。

---

### 10.3 Micro-training precheck

做短程训练，例如：

```text
steps = 100, 500
```

比较：

| method | credit |
|---|---|
| analytic span adjoint | oracle |
| identity span adjoint | identity |
| router span correction | learned correction |
| random / feedback | negative baseline |

记录：

| metric key | 含义 |
|---|---|
| `cnv2_5/micro/train_loss_auc` | short training loss AUC |
| `cnv2_5/micro/val_loss_after` | 结束时 val loss |
| `cnv2_5/micro/val_acc_after` | 结束时 val acc |
| `cnv2_5/micro/gap_vs_identity` | 相对 identity 的提升 |
| `cnv2_5/micro/gap_vs_analytic` | 相对 analytic 的差距 |
| `cnv2_5/micro/branch_ratio_after` | 训练后 branch ratio |
| `cnv2_5/micro/phi_prime_p95_after` | 训练后 $\phi'$ |
| `cnv2_5/micro/jcond_after` | 训练后 condition |

Router 进入 Stage D 的最低条件：

$$
\text{micro val loss improvement vs identity}>2\%
$$

或：

$$
\text{micro val acc improvement vs identity}>0.5\%
$$

---

## 11. CNv2-6：Identity / analytic / router joint-training precheck

### 11.1 目标

CNv2-6 在更深模型、更复杂数据和 span setting 下重新检查：identity adjoint 是否仍然接近 analytic adjoint。如果 identity 仍然接近 analytic，那么 learned router 不应进入 Stage D 主线。

---

### 11.2 方法比较

| method | credit |
|---|---|
| FullBP / analytic | full analytic local adjoint |
| IdentityAdj | identity credit |
| SpanIdentityAdj | span identity credit |
| RouterCorrectionAdj | learned correction router |

更新方式统一使用：

$$
\text{diag-to-Sobolev functional update}
$$

非 KAN 参数统一使用 AdamW。

---

### 11.3 关键指标

| metric key | 含义 |
|---|---|
| `cnv2_6/test_acc` | test accuracy |
| `cnv2_6/acc_gap_vs_analytic` | 相对 analytic gap |
| `cnv2_6/val_loss_auc` | val loss AUC |
| `cnv2_6/auc_ratio_vs_analytic` | AUC ratio |
| `cnv2_6/branch_ratio` | branch ratio |
| `cnv2_6/phi_prime_p95` | $\phi'$ |
| `cnv2_6/jcond` | Jacobian condition |
| `cnv2_6/memory_peak_mb` | peak memory |
| `cnv2_6/step_time_ms` | step time |

---

### 11.4 决策标准

如果：

$$
\text{IdentityAdj acc gap vs analytic}<1\%
$$

且：

$$
\text{AUC ratio vs analytic}<1.05
$$

则说明 identity 仍然足够强，router 不进入主线。

如果 router 满足：

$$
\text{Router acc improvement vs identity}>0.5\%
$$

或：

$$
\text{Router val loss AUC improvement vs identity}>2\%
$$

且没有显著增加 instability，则 router 可以进入 Stage D ablation，但仍不作为唯一主方法。

---

## 12. CNv2-7：Residual stress ablation

### 12.1 目标

CNv2-7 用来回答：是否需要改变架构来让 learned router 有价值？

原则是：

> 不直接把 non-residual 当主线，只作为诊断。

---

### 12.2 模型变体

| variant | block 形式 | 作用 |
|---|---|---|
| standard residual | $h+\alpha KAN(LN(h))$ | 主线 |
| scaled residual | $h+\beta\alpha KAN(LN(h))$ | 放大 branch |
| gated residual | $h+\gamma(h)KAN(LN(h))$ | 学习 branch gate |
| target-branch residual | 加 branch ratio control loss | 直接控制 branch |
| non-residual | $KAN(LN(h))$ | stress / 负对照 |

---

### 12.3 记录指标

| metric key | 含义 |
|---|---|
| `cnv2_7/identity_relerr` | identity relerr |
| `cnv2_7/correction_ratio` | correction ratio |
| `cnv2_7/router_gain` | router gain over identity |
| `cnv2_7/test_acc` | test accuracy |
| `cnv2_7/jcond` | Jacobian condition |
| `cnv2_7/amp_p95` | amplification p95 |
| `cnv2_7/phi_prime_p95` | $\phi'$ |
| `cnv2_7/diverged` | 是否发散 |

---

### 12.4 判断逻辑

如果 scaled / gated residual 能满足：

$$
\operatorname{relerr}_{identity}>0.25
$$

且：

$$
\text{test gap}<2\%
$$

$$
\text{amp p95}<1.5
$$

那么该架构可以作为 router stress setting。

如果 non-residual 让 identity 变弱，但 Jacobian condition 或 amplification 爆炸，则 non-residual 只能作为负对照，不能进入主线。

---

## 13. W&B Dashboard 总设计

Stage C-next v2 建议建立 8 个 dashboard 页面。

### Page 1：Overview

展示：

1. 各 phase run 数量和 failure 数；
2. dataset × model × regime summary；
3. test acc vs geometry scatter；
4. identity relerr distribution；
5. router gain distribution。

---

### Page 2：Hard-dataset Pareto Search

必须包含：

1. test gap vs J cond ratio scatter；
2. branch / AdamW vs test acc；
3. coeff lr × rest lr heatmap；
4. alpha × coeff lr heatmap；
5. no-KAN drop vs branch ratio；
6. Pareto front plot。

---

### Page 3：Span Geometry

必须包含：

1. span vs identity relerr；
2. span vs correction ratio；
3. span vs amp p95；
4. span vs credit rank90；
5. router-worthy setting map；
6. dataset / regime / span heatmap。

---

### Page 4：Correction Router Calibration

必须包含：

1. correction norm predicted vs teacher scatter；
2. correction cosine by router type；
3. full relerr vs identity relerr；
4. gain over identity bar chart；
5. beat rate by router；
6. clipping rate by trust-region threshold。

---

### Page 5：Span Router

必须包含：

1. span router gain heatmap；
2. train span × eval span matrix；
3. train regime × eval regime matrix；
4. correction cosine vs span；
5. cost vs gain plot。

---

### Page 6：One-step / Micro-training Usefulness

必须包含：

1. actual loss delta by credit source；
2. gain over identity by credit source；
3. predicted vs actual descent scatter；
4. micro-training loss curves；
5. micro-training val acc bar chart。

---

### Page 7：Joint-training Precheck

必须包含：

1. IdentityAdj vs AnalyticAdj accuracy gap；
2. IdentityAdj vs AnalyticAdj AUC ratio；
3. RouterAdj vs IdentityAdj improvement；
4. memory / time bar chart；
5. branch ratio and geometry after training。

---

### Page 8：Failure Analysis

必须包含 failure table，以及以下 failure 类型统计：

| failure type | 含义 |
|---|---|
| `pareto_not_found` | 复杂数据上找不到可用 Pareto |
| `identity_too_strong` | identity relerr 过低，不值得训 router |
| `router_no_gain` | router 未打败 identity |
| `router_bad_correction_cos` | correction 方向学不好 |
| `router_norm_miscalibrated` | norm 校准失败 |
| `one_step_no_gain` | one-step 不优于 identity |
| `micro_train_no_gain` | micro-training 不优于 identity |
| `geometry_collapse` | amp / J cond / $\phi'$ 失控 |
| `non_residual_unstable` | non-residual 失稳 |

---

## 14. Failure table 设计

每个 failure 必须记录：

| 字段 | 含义 |
|---|---|
| `failure/type` | failure 类型 |
| `failure/phase` | CNv2 phase |
| `failure/dataset` | 数据集 |
| `failure/model_depth` | depth |
| `failure/hidden_dim` | hidden dim |
| `failure/basis_count` | basis count |
| `failure/span` | span |
| `failure/regime` | regime |
| `failure/router_type` | router 类型 |
| `failure/metric_name` | 触发指标 |
| `failure/metric_value` | 指标值 |
| `failure/threshold` | 阈值 |
| `failure/diagnosis` | 诊断 |
| `failure/recommended_action` | 建议动作 |

---

## 15. 第一轮最小执行包

为了避免一次性展开过大，Stage C-next v2 第一轮只执行四个包。

### Package A：Hard-dataset Pareto search

```text
dataset = Fashion-MNIST, KMNIST
model = depth 4, hidden 64, basis 16
coeff_lr = 0.3, 0.5, 0.7, 1.0
rest_lr = 0.001, 0.003
alpha = 1.0, 1.5
seeds = 0,1,2
```

目标：找到 test gap < $2\%$ 且 geometry 可控的 Pareto checkpoint。

---

### Package B：Span geometry audit

```text
dataset = Fashion-MNIST, KMNIST
model = depth 4, 8
span = 1, 2, 4
regime = AdamW, Pareto, active
seeds = 0,1,2,3,4
```

目标：找到 identity relerr > $0.25$ 且 amp p95 < $1.5$ 的 router-worthy setting。

---

### Package C：Norm-calibrated correction router

只在 Package B 找到的 router-worthy setting 上跑。

```text
router = zero, scalar, norm_calibrated_scalar, clipped_ridge, diag_lowrank, direction_norm
span = selected from Package B
seeds = 0,1,2,3,4
```

目标：gain over identity > $20\%$。

---

### Package D：One-step + micro-training usefulness

对 Package C 中表现最好的 router 做：

```text
one-step hidden update
micro-training 100 / 500 steps
```

目标：证明 router credit 不只是 fidelity 更好，而且 actual descent / short training 也优于 identity。

---

## 16. Stage D 决策规则

### 情况 1：没有任何 router-worthy setting

若所有稳定 setting 都满足：

$$
\operatorname{relerr}_{identity}<0.2
$$

则结论是：

$$
\boxed{
\text{residual DG-KAN local credit transport is identity-dominant.}
}
$$

Stage D 不再推进 learned router，改为：

$$
\text{IdentityAdj / AnalyticAdj + functional update + memory accounting}
$$

---

### 情况 2：存在 router-worthy setting，但 router 仍无 gain

若：

$$
\operatorname{relerr}_{identity}>0.25
$$

但 router gain 仍低于 $20\%$，则说明当前 router 结构不足。Stage D 仍不使用 router，只保留 router 作为 later work。

---

### 情况 3：router fidelity 有 gain，但 one-step 无 gain

说明 fidelity 不等于 update usefulness。需要 descent-aware router loss，不进入 Stage D 主线。

---

### 情况 4：router fidelity 和 one-step 都有 gain

如果满足：

$$
\operatorname{gain}_{rel}>0.2
$$

$$
\operatorname{gain}_{step}>0.05
$$

并且 micro-training 优于 identity，则 router 可以进入 Stage D ablation。

---

### 情况 5：non-residual 让 router 有用但几何崩溃

不能把 non-residual 作为主线。结论应写为：

> Router becomes useful only when the architecture loses the geometry-preserving property, which is not the desired DG-KAN regime.

---

## 17. 最终预期结论模板

如果 Stage C-next v2 成功，可以写：

> By increasing routing span and moving to harder classification tasks, we identify regimes where the identity baseline is no longer sufficient while DG-KAN geometry remains stable. In these regimes, norm-calibrated residual correction routers improve over identity routing both in VJP fidelity and one-step descent usefulness.

如果 Stage C-next v2 失败，但 identity 仍强，可以写：

> Even under deeper models, larger spans, and harder datasets, stable residual DG-KAN remains identity-dominant at the local credit level. This suggests that the main practical route is not learned router replacement, but identity / analytic adjoint local training combined with functional-space updates and low-memory sufficient statistics.

这两个结果都可接受。前者支持 learned router，后者支持 analytic / identity adjoint 作为主线。

---

## 18. 总结

Stage C-next v2 的核心不是“继续强推 learned router”，而是严肃检验 learned router 是否真的有必要。

它通过以下方式放大非 identity correction：

1. 从单层 router 扩展到 span router；
2. 从 MNIST / Fashion 扩展到 KMNIST / EMNIST / CIFAR-10 small；
3. 从固定 Pareto 超参改为每个数据集重新搜索 Pareto；
4. 从普通 correction router 改为 norm-calibrated correction router；
5. 从 fidelity 指标扩展到 one-step 和 micro-training usefulness；
6. 从单纯 residual 主线加入 scaled / gated / non-residual ablation。

最终要回答：

$$
\boxed{
\text{learned router 的失败是因为实验太简单，还是因为 residual DG-KAN 天然 identity-dominant？}
}
$$

只有当 router 在稳定的、非 identity 充足的 setting 中明确打败 identity，它才应该进入 Stage D 主线。否则 Stage D 应转向 analytic / identity adjoint + functional update + memory accounting。

