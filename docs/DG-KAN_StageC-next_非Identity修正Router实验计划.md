# DG-KAN Stage C-next 补充实验计划：放大非 Identity 修正并验证 Router 价值

> 文件名：`DG-KAN_StageC-next_非Identity修正Router实验计划.md`  
> 对象：DG-KAN / DG-LCA Stage C 后续实验  
> 版本：v1.0  
> 公式格式：Typora 友好，统一使用 `$...$` 与 `$$...$$`  
> 核心目标：在 Stage C 已证明 analytic KAN adjoint 正确、Pareto regime 改善 credit geometry、但 identity router 过强的基础上，重新设计实验，使 learned router 的任务从“预测完整 VJP”改为“学习 identity 之外的非平凡修正项”，并在更深、更强 branch、更复杂数据中检验 router 是否真的有价值。

---

## 0. 为什么需要 Stage C-next

Stage C 的结果显示，DG-KAN 的 core KAN analytic adjoint 与 full residual block adjoint 都已经通过严格正确性验证。也就是说，KAN primitive 的局部 VJP：

$$
g_x=
\Phi'(x)^\top g_y
$$

以及 full residual block 的局部 VJP：

$$
g_h=
g_{out}
+
D\operatorname{LN}(h)^\top
\left(
\alpha \Phi'(x)^\top g_{out}
\right)
$$

都可以和 autograd VJP 对齐到数值精度级别。

但是 Stage C 也发现了一个新的问题：因为当前 DG-KAN block 是 residual 结构：

$$
h_{out}=h+
\alpha \operatorname{KAN}(\operatorname{LN}(h))
$$

所以 full block VJP 可以写成：

$$
g_h=g_{out}+\Delta g
$$

其中：

$$
\Delta g=
D\operatorname{LN}(h)^\top
\left(
\alpha\Phi'(x)^\top g_{out}
\right)
$$

当 KAN branch 较弱或中等强度时，$\Delta g$ 相比 $g_{out}$ 较小，因此最简单的 identity router：

$$
\hat g_h=g_{out}
$$

已经非常接近真实 VJP。这导致原来的 learned router 任务：

$$
C^\phi(h,h_{out})[g_{out}]\approx g_h
$$

不再是一个能充分检验 router 价值的任务。因为真实目标 $g_h$ 的主体已经由 $g_{out}$ 解释，learned router 只是在一个很强的 baseline 上做很小的改进。

因此 Stage C-next 的核心改变是：

$$
\boxed{
\text{不要再让 router 直接学习完整 } g_h，
\text{而是学习非 identity 修正 } \Delta g=g_h-g_{out}。
}
$$

新的 router 形式应为：

$$
\hat g_h
=
g_{out}
+
R^\phi(h,h_{out})[g_{out}]
$$

其中：

$$
R^\phi(h,h_{out})[g_{out}]
\approx
\Delta g^{teacher}
=
g_h^{teacher}-g_{out}
$$

Stage C-next 不是为了人为制造不稳定性，也不是为了去掉 residual。Residual 是 DG-KAN 稳定 credit transport 的核心设计，不应作为主线移除。Stage C-next 的目标是在保持 residual 主线的同时，通过更深模型、更强 branch、更复杂数据、更明确的 correction-targeted router，检验 learned router 是否能在 identity 之外提供实际增益。

---

## 1. Stage C-next 的核心问题

Stage C-next 需要回答六个问题。

### 1.1 非 identity 修正项到底有多大

首先要测量：

$$
\Delta g = g_h^{teacher}-g_{out}
$$

是否足够大、足够稳定、足够结构化。如果 $\Delta g$ 本身很小，那么 learned router 没有发挥空间；如果 $\Delta g$ 太大且不稳定，则 credit geometry 可能已经恶化。

因此我们要定义 correction strength：

$$
r_{\Delta,k}
=
\frac{
\|\Delta g_k\|
}{
\|g_{out,k}\|+\epsilon
}
$$

以及 full identity error：

$$
\operatorname{relerr}_{id,k}
=
\frac{
\|g_{out,k}-g_h^{teacher}\|
}{
\|g_h^{teacher}\|+\epsilon
}
=
\frac{
\|\Delta g_k\|
}{
\|g_h^{teacher}\|+\epsilon
}
$$

如果 $r_{\Delta}$ 长期低于 $0.05$，identity 已经几乎完美，learned router 很难显示价值。如果 $r_{\Delta}$ 在 $0.1$ 到 $0.4$ 之间，且 credit amplification 仍然可控，这是最适合测试 router 的区间。如果 $r_{\Delta}>0.8$ 且 Jacobian condition 或 noise gain 明显升高，则该设置可能已经进入不稳定区。

---

### 1.2 Router 能不能学会 correction，而不是重复 identity

新的目标不是：

$$
\hat g_h\approx g_h
$$

而是：

$$
\hat{\Delta g}
=
R^\phi(h,h_{out})[g_{out}]
\approx
\Delta g^{teacher}
$$

最终输出：

$$
\hat g_h=g_{out}+\hat{\Delta g}
$$

这要求 router 不再靠 identity path 作弊。评估时必须同时报告：

1. correction-level fidelity；
2. full-VJP fidelity；
3. improvement over identity；
4. one-step local update gain over identity。

如果一个 router 的 full cosine 很高，但 correction cosine 很低，说明它只是靠 identity path 获得高分，并没有真正学会 KAN branch correction。

---

### 1.3 哪些模型 / 数据 / branch 强度会让 identity 不再免费获胜

Stage C 使用 depth $2$、hidden dim $32$、basis count $8$ 的小模型，因此每层 VJP 接近 identity 是合理的。Stage C-next 要系统检查：

1. 增加 depth 是否会让 identity baseline 变弱；
2. 增加 hidden dim / basis count 是否让 branch correction 更复杂；
3. 更复杂数据集是否需要更强非 identity correction；
4. 调大 branch activity 是否能让 learned router 有发挥空间；
5. 这些变化是否仍然保持 Jacobian condition、credit amplification、noise gain 可控。

---

### 1.4 Pareto regime 是否仍然是最适合 router 的 regime

Stage C 已经显示 Pareto regime 在精度和几何之间最好。Stage C-next 要进一步判断：

$$
\text{Pareto regime}
\rightarrow
\text{moderate } \Delta g
\rightarrow
\text{stable correction target}
\rightarrow
\text{router gain over identity}
$$

是否成立。

理想情况是：

1. conservative 的 $\Delta g$ 太小，router 没必要；
2. active 的 $\Delta g$ 大，但 noise gain、condition、rank 变差；
3. Pareto 的 $\Delta g$ 适中，identity 不是完美，但 credit geometry 仍然健康，因此最适合 learned correction router。

---

### 1.5 Analytic adjoint 是否仍应作为 Stage D 主线

如果 correction router 无法稳定打败 identity，而 analytic adjoint 已经正确、快速、低误差，那么 Stage D 应该优先使用：

$$
\boxed{
\text{AnalyticAdj-KAN-Sobolev}
}
$$

而不是 Router-KAN-Sobolev。

Stage C-next 需要给出明确决策：

1. learned correction router 是否足以进入 Stage D；
2. 如果不能，是否只保留 analytic adjoint；
3. 如果 correction router 只在 stress setting 有价值，是否作为后续探索而非主线。

---

### 1.6 是否需要去掉 residual

Stage C-next 不以去掉 residual 为主线。非 residual KAN 只作为 stress / negative control。

原因是 residual 结构本身是 DG-KAN 信息保持 interface 的关键：

$$
h_{out}=h+\text{controlled update}
$$

如果为了让 router 显得更有价值而去掉 residual，可能只是人为制造了不稳定、病态、难学的 credit transport。这并不能证明 DG-LCA 更强。

因此实验主线是：

$$
\boxed{
\text{保留 residual，但系统放大、隔离、学习 residual correction。}
}
$$

---

## 2. 总体实验结构

Stage C-next 分成七个部分。

1. **CN0：checkpoint 与 branch regime 复核**  
   准备不同 branch strength、不同 depth、不同数据集的 checkpoint，并复核其 test accuracy、branch ratio、$\phi'_{p95}$、Jacobian condition、identity error。

2. **CN1：Correction geometry audit**  
   直接分析 $\Delta g=g_h-g_{out}$ 的大小、秩、稳定性、noise sensitivity、与 $g_{out}$ 的关系。

3. **CN2：Residual correction router distillation**  
   训练 router 学习 $\Delta g$，而不是完整 $g_h$。

4. **CN3：Identity stress test**  
   系统增加 depth、branch target、dataset complexity、hidden dim、basis count，看 identity 什么时候不再足够。

5. **CN4：One-step correction usefulness**  
   用 correction router 的 credit 做 hidden-state 和 parameter-level one-step update，检验是否真的比 identity 带来更好的 actual descent。

6. **CN5：AnalyticAdj joint-training precheck**  
   先用 analytic adjoint 和 identity credit 分别做轻量 joint-training precheck，判断 Stage D 该优先走 analytic adjoint 还是 learned router。

7. **CN6：Non-residual / scaled residual ablation**  
   作为后置 ablation，检验没有 residual 时 identity 是否失效，以及失效是否伴随几何崩坏。

---

## 3. 数据集与模型设置

### 3.1 数据集顺序

Stage C-next 不应直接跳到大规模任务。推荐顺序如下。

| 优先级 | 数据集 | 作用 |
|---:|---|---|
| 1 | MNIST small | 与 Stage B/C 结果直接衔接，便于机制诊断 |
| 2 | Fashion-MNIST small | 已观察到同类 branch under-active，适合作为主要复核 |
| 3 | KMNIST | 类别结构更复杂，仍然便宜，适合测试 identity 是否减弱 |
| 4 | EMNIST Balanced small | 更多类别，测试 output-side credit 复杂度 |
| 5 | CIFAR-10 small | 更复杂视觉任务，需 ConvStem-DG-KAN |
| 6 | CIFAR-100 small | 后续 stress，不进入第一轮 |

Two Moons 只用于 sanity，不作为 Stage C-next 的主要结论来源。

---

### 3.2 模型族

主线仍然使用 residual DG-KAN：

$$
h_{k+1}
=
h_k+
\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))
$$

#### Model S：Small residual DG-KAN

用于 MNIST / Fashion / KMNIST：

| setting | values |
|---|---|
| hidden dim | $32,64$ |
| depth | $2,4$ |
| basis count | $8,16$ |
| KAN connectivity | dense |
| alpha init | $1.0$ |

#### Model M：Medium residual DG-KAN

用于 harder small datasets：

| setting | values |
|---|---|
| hidden dim | $128$ |
| depth | $4,8$ |
| basis count | $16$ |
| KAN connectivity | dense or grouped |
| alpha init | $1.0$ |

#### Model C：ConvStem-DG-KAN

用于 CIFAR-10 small：

$$
x\rightarrow \operatorname{ConvStem}(x)\rightarrow h_0
$$

然后：

$$
h_{k+1}=h_k+
\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))
$$

建议第一轮 CIFAR-10 small 使用：

| setting | value |
|---|---:|
| hidden dim | $128$ |
| depth | $4$ |
| basis count | $16$ |
| KAN connectivity | grouped |

---

### 3.3 Branch strength 控制方式

Stage C-next 不用单一 coefficient learning rate 定义 regime，而使用实际结果指标定义 regime。

主要监控：

$$
r_{branch,k}
=
\frac{
\|\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))\|
}{
\|h_k\|+\epsilon
}
$$

相对 AdamW：

$$
\operatorname{branchOverAdamW}
=
\frac{
 r_{branch}^{method}
}{
 r_{branch}^{AdamW}
}
$$

目标区间：

| regime | branch / AdamW | 用途 |
|---|---:|---|
| weak | $0.2$ 到 $0.3$ | identity 极强，稳定性上限 |
| pareto | $0.4$ 到 $0.6$ | 主线，精度接近且几何稳定 |
| active | $0.8$ 到 $1.2$ | 精度优先，测试 credit complexity |
| over-active | $>2.0$ | failure boundary，不作为主线 |

实现手段包括：

1. 改变 KAN coefficient lr：$0.1,0.3,1.0$；
2. 改变 residual scale：$\alpha\in\{0.5,1.0,2.0\}$；
3. 使用 branch-ratio target loss；
4. 使用 depth / hidden dim / basis count 提高 correction complexity。

需要强调：$\alpha$ 和 coefficient lr 都只是实现手段，regime 的定义应由实际 branch ratio、精度 gap、$\phi'_{p95}$、Jacobian condition 共同决定。

---

## 4. 全局 W&B config

每个 run 必须记录以下 config，保证不同实验能够合并分析。

| config key | 示例 | 含义 |
|---|---|---|
| `exp/stage` | `stage_c_next` | 实验阶段 |
| `exp/phase` | `CN1_correction_audit` | 子阶段 |
| `exp/seed` | `0` | 随机种子 |
| `data/name` | `MNIST`, `FashionMNIST`, `KMNIST`, `CIFAR10Small` | 数据集 |
| `data/train_size` | `6000` | 训练集大小 |
| `data/val_size` | `1000` | 验证集大小 |
| `data/test_size` | `1000` | 测试集大小 |
| `model/family` | `ResidualDGKAN`, `ConvStemDGKAN` | 模型族 |
| `model/depth` | `2`, `4`, `8` | block 数 |
| `model/hidden_dim` | `32`, `64`, `128` | hidden dimension |
| `model/basis_count` | `8`, `16` | KAN basis 数量 |
| `model/connectivity` | `dense`, `grouped` | KAN 连接方式 |
| `model/alpha_init` | `1.0` | 初始 residual scale |
| `model/alpha_trainable` | `true`, `false` | residual scale 是否可训练 |
| `train/update_type` | `adamw`, `diag_to_sobolev` | 更新方式 |
| `train/coeff_lr` | `0.1`, `0.3`, `1.0` | KAN coeff lr |
| `train/rest_lr` | `0.003` | 非 KAN 参数 lr |
| `train/warmup_frac` | `0.25` | diag-to-Sobolev warmup |
| `regime/name` | `weak`, `pareto`, `active`, `overactive` | 根据实际指标定义的 regime |
| `router/task` | `full_vjp`, `residual_correction` | router 学习目标 |
| `router/type` | `zero_correction`, `ridge_correction`, `diag_lowrank_correction` | router 类型 |
| `router/rank` | `0`, `4`, `8`, `16`, `32` | low-rank rank |
| `router/input_features` | `h,h_out,g_out,phi_prime_stats` | router 条件输入 |
| `audit/credit_type` | `task_bp`, `random`, `mixed` | credit 类型 |
| `audit/batch_size` | `256`, `512`, `1024` | audit batch size |
| `resource/device` | `cuda` | 设备 |

---

## 5. CN0：Checkpoint 与 regime 复核

### 5.1 目标

CN0 训练并复核一组 checkpoint，确保每个 checkpoint 确实对应预期 branch activity regime。只有 CN0 复核通过的 checkpoint 才能进入 CN1-C5。

CN0 不是只看 test accuracy，而是要同时确认：

1. branch ratio；
2. no-KAN ablation；
3. $\phi'_{p95}$；
4. Jacobian condition；
5. credit amplification；
6. active basis coverage；
7. identity VJP error。

---

### 5.2 第一轮 checkpoint 矩阵

第一轮建议只做以下组合，避免一开始过宽。

| dataset | depth | hidden dim | basis | coeff lr | rest lr | alpha | seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 2 | 32 | 8 | $0.1,0.3,1.0$ | $0.003$ | $1.0$ | 5 |
| Fashion | 2 | 32 | 8 | $0.1,0.3,1.0$ | $0.003$ | $1.0$ | 5 |
| KMNIST | 2 | 32 | 8 | $0.3,1.0$ | $0.003$ | $1.0$ | 5 |
| MNIST | 4 | 64 | 16 | $0.3,1.0$ | $0.003$ | $1.0$ | 3 |
| Fashion | 4 | 64 | 16 | $0.3,1.0$ | $0.003$ | $1.0$ | 3 |

第二轮再加入 CIFAR-10 small。

---

### 5.3 CN0 metrics

任务指标：

| metric key | 含义 |
|---|---|
| `cn0/train_acc` | train accuracy |
| `cn0/val_acc` | validation accuracy |
| `cn0/test_acc` | test accuracy |
| `cn0/test_gap_vs_adamw` | 相对 AdamW 的 test gap |
| `cn0/loss_auc_val` | validation loss AUC |

Branch activity：

| metric key | 含义 |
|---|---|
| `cn0/branch/global_mean_ratio` | 平均 branch ratio |
| `cn0/branch/layer_{k}_ratio` | 每层 branch ratio |
| `cn0/branch/over_adamw` | branch / AdamW |
| `cn0/ablation/no_kan_test_acc` | 关闭 KAN branch 后 test acc |
| `cn0/ablation/no_kan_acc_drop` | full acc - no-KAN acc |
| `cn0/ablation/kan_logit_delta_norm` | KAN branch 对 logits 的贡献 |

Geometry：

| metric key | 含义 |
|---|---|
| `cn0/kan/phi_prime_p95` | $\phi'_{p95}$ |
| `cn0/kan/phi_prime_max` | $\phi'_{max}$ |
| `cn0/kan/curvature` | $\int |\phi''|^2$ |
| `cn0/jacobian/condition_max` | 最大 Jacobian condition |
| `cn0/jacobian/condition_mean` | 平均 Jacobian condition |
| `cn0/credit/amplification_p95` | credit amplification p95 |

Basis coverage：

| metric key | 含义 |
|---|---|
| `cn0/basis/active_fraction` | active basis fraction |
| `cn0/basis/dead_fraction` | dead basis fraction |
| `cn0/input/out_of_grid_fraction` | 输入落出 grid 比例 |

Identity difficulty：

| metric key | 含义 |
|---|---|
| `cn0/identity/cos` | identity router cosine |
| `cn0/identity/relerr` | identity relative error |
| `cn0/correction/norm_ratio` | $\|\Delta g\|/\|g_{out}\|$ |
| `cn0/correction/backward_branch_ratio` | backward branch ratio |

---

### 5.4 CN0 成功条件

一个 checkpoint 可以进入后续实验，需要满足：

1. 没有 NaN / Inf；
2. active basis fraction 不低于 $0.6$；
3. out-of-grid fraction 不超过 $0.02$；
4. 对于 Pareto checkpoint，test gap < $1\%$；
5. 对于 Pareto checkpoint，$\phi'_{p95}$ 和 Jacobian condition 均低于 AdamW；
6. 对于 active checkpoint，精度接近 AdamW，但几何不能完全失控；
7. 对于 over-active checkpoint，只用于 failure boundary，不进入主 router 训练。

---

## 6. CN1：Correction geometry audit

### 6.1 目标

CN1 直接分析非 identity 修正项：

$$
\Delta g=g_h^{teacher}-g_{out}
$$

这一步的目标是回答：

1. $\Delta g$ 有多大；
2. $\Delta g$ 是否低秩；
3. $\Delta g$ 是否稳定；
4. $\Delta g$ 是否随 branch ratio 增大而变复杂；
5. 哪个 regime 最适合 learned correction router。

---

### 6.2 Credit types

每个 checkpoint、每层 block、每个 batch，使用三类 output credit：

| credit type | 定义 | 作用 |
|---|---|---|
| `task_bp` | 真实任务 BP credit | 主分析 |
| `random` | Gaussian credit，norm 匹配 task credit | 测 operator 全局行为 |
| `mixed` | task + random | 测 OOD robustness |

Mixed credit：

$$
g_{out}^{mix}
=
\lambda g_{out}^{task}+(1-\lambda)g_{out}^{rand}
$$

其中：

$$
\lambda\in\{0.25,0.5,0.75\}
$$

---

### 6.3 CN1 metrics

Correction magnitude：

| metric key | 含义 |
|---|---|
| `cn1/correction/layer_{k}/norm` | $\|\Delta g\|$ |
| `cn1/correction/layer_{k}/norm_ratio_to_gout` | $\|\Delta g\|/\|g_{out}\|$ |
| `cn1/correction/layer_{k}/norm_ratio_to_teacher` | $\|\Delta g\|/\|g_h^{teacher}\|$ |
| `cn1/correction/global/mean_norm_ratio` | 全层平均 norm ratio |
| `cn1/correction/global/worst_norm_ratio` | 最差层 norm ratio |

Identity baseline difficulty：

| metric key | 含义 |
|---|---|
| `cn1/identity/layer_{k}/cos` | $\cos(g_{out},g_h^{teacher})$ |
| `cn1/identity/layer_{k}/relerr` | identity relerr |
| `cn1/identity/global/mean_relerr` | 平均 identity relerr |
| `cn1/identity/global/worst_relerr` | 最差层 relerr |

Correction alignment：

| metric key | 含义 |
|---|---|
| `cn1/alignment/layer_{k}/delta_vs_gout_cos` | $\cos(\Delta g,g_{out})$ |
| `cn1/alignment/layer_{k}/delta_vs_teacher_cos` | $\cos(\Delta g,g_h^{teacher})$ |
| `cn1/alignment/global/delta_vs_gout_cos_mean` | 全层平均 |

Correction rank：

对 correction 矩阵：

$$
\Delta G_k=[\Delta g_k^{(1)},\dots,\Delta g_k^{(B)}]
$$

做 SVD，记录：

| metric key | 含义 |
|---|---|
| `cn1/rank/layer_{k}/E4` | top-4 energy |
| `cn1/rank/layer_{k}/E8` | top-8 energy |
| `cn1/rank/layer_{k}/E16` | top-16 energy |
| `cn1/rank/layer_{k}/PR` | participation rank |
| `cn1/rank/layer_{k}/rank90` | rank90 |
| `cn1/rank/layer_{k}/rank95` | rank95 |

Regime relation：

| metric key | 含义 |
|---|---|
| `cn1/relation/branch_vs_delta_corr` | branch ratio 与 correction norm 相关性 |
| `cn1/relation/phi_prime_vs_delta_corr` | $\phi'_{p95}$ 与 correction norm 相关性 |
| `cn1/relation/jcond_vs_delta_corr` | J condition 与 correction norm 相关性 |

---

### 6.4 CN1 可视化

必须包含以下图：

1. **Correction norm ratio by regime**  
   x-axis: regime；y-axis: $\|\Delta g\|/\|g_{out}\|$。

2. **Identity relerr by regime and depth**  
   x-axis: depth；y-axis: identity relerr；color: regime。

3. **Branch ratio vs correction norm scatter**  
   x-axis: branch / AdamW；y-axis: correction norm ratio；color: dataset。

4. **Correction rank heatmap**  
   layer × regime，颜色为 rank90 或 PR。

5. **Correction alignment histogram**  
   $\cos(\Delta g,g_{out})$ 的分布。

6. **Identity relerr vs test accuracy gap**  
   判断 identity difficulty 是否伴随性能变化。

---

### 6.5 CN1 结论标准

CN1 不以 pass/fail 为主，而是寻找适合 router 的测试区间。推荐判定：

| correction ratio | 解释 |
|---:|---|
| $<0.05$ | identity 几乎完美，不适合验证 learned router |
| $0.05$ 到 $0.15$ | 弱 correction，适合 sanity |
| $0.15$ 到 $0.40$ | 主 router 区间，非 identity 修正明显但仍可控 |
| $0.40$ 到 $0.80$ | stress 区间，router 有价值但几何风险增大 |
| $>0.80$ | 可能过强，需检查 condition / noise gain |

Stage C-next 的主实验应优先使用 correction ratio 在 $0.15$ 到 $0.40$ 的 checkpoint。

---

## 7. CN2：Residual correction router distillation

### 7.1 目标

CN2 是 Stage C-next 的核心。它重做 router distillation，但学习目标换成：

$$
\Delta g^{teacher}=g_h^{teacher}-g_{out}
$$

router 预测：

$$
\hat{\Delta g}=R^\phi(h,h_{out})[g_{out}]
$$

最终 credit：

$$
\hat g_h=g_{out}+\hat{\Delta g}
$$

这样 identity baseline 对应：

$$
\hat{\Delta g}=0
$$

如果 learned router 不能比 zero correction 更好，就说明它没有学到非 identity 修正。

---

### 7.2 Router 类型

#### Router 0：zero correction

$$
\hat{\Delta g}=0
$$

这等价于 identity router，是最重要 baseline。

#### Router 1：static ridge correction

$$
\hat{\Delta g}=Wg_{out}
$$

其中：

$$
W=
\arg\min_W
\sum_n
\|Wg_{out}^{(n)}-\Delta g^{(n)}\|^2
+
\lambda\|W\|_F^2
$$

需要使用 centered adaptive ridge，避免 tiny correction scale 下过拟合。

#### Router 2：conditioned diagonal correction

$$
\hat{\Delta g}=D(h,h_{out})\odot g_{out}
$$

#### Router 3：conditioned low-rank correction

$$
\hat{\Delta g}
=
D(h,h_{out})\odot g_{out}
+
U(h,h_{out})(V(h,h_{out})^\top g_{out})
$$

rank：

$$
r\in\{4,8,16,32\}
$$

#### Router 4：derivative-informed correction

使用 KAN derivative statistics 作为条件输入：

$$
\phi'_{p50},\phi'_{p95},\phi'_{max},
\text{branch ratio},
\text{layer id}
$$

router 仍然保持对 $g_{out}$ 线性，只让条件参数依赖这些统计。

#### Router 5：analytic correction oracle

$$
\Delta g^{oracle}=g_h^{teacher}-g_{out}
$$

用于上限，不参与学习。

---

### 7.3 Router loss

Correction normalized MSE：

$$
\mathcal L_{\Delta}
=
\frac{
\|\hat{\Delta g}-\Delta g^{teacher}\|^2
}{
\|\Delta g^{teacher}\|^2+\epsilon
}
$$

Correction cosine loss：

$$
\mathcal L_{cos,\Delta}
=
1-
\frac{
\langle \hat{\Delta g},\Delta g^{teacher}\rangle
}{
\|\hat{\Delta g}\|\|\Delta g^{teacher}\|+\epsilon
}
$$

Full VJP loss：

$$
\mathcal L_{full}
=
\frac{
\|g_{out}+\hat{\Delta g}-g_h^{teacher}\|^2
}{
\|g_h^{teacher}\|^2+\epsilon
}
$$

Norm calibration：

$$
\mathcal L_{norm}
=
\left(
\frac{
\|\hat{\Delta g}\|
}{
\|\Delta g^{teacher}\|+\epsilon
}
-1
\right)^2
$$

Total loss：

$$
\mathcal L_{router}
=
\mathcal L_{\Delta}
+
\lambda_{cos}\mathcal L_{cos,\Delta}
+
\lambda_{full}\mathcal L_{full}
+
\lambda_{norm}\mathcal L_{norm}
$$

建议第一轮：

$$
\lambda_{cos}=0.5,
\quad
\lambda_{full}=0.2,
\quad
\lambda_{norm}=0.1
$$

---

### 7.4 数据拆分

每个 router 训练集由四元组组成：

$$
(h,h_{out},g_{out},\Delta g^{teacher})
$$

拆分：

| split | 来源 | 用途 |
|---|---|---|
| train_router | checkpoint training batches | router 训练 |
| val_router | heldout batch same checkpoint | early stopping |
| test_same_regime | 同 regime 不同 batch | 同分布泛化 |
| test_cross_seed | 同 regime 不同 seed | seed 泛化 |
| test_cross_regime | train on one regime, eval on others | regime 泛化 |
| test_random_credit | random credit | OOD credit |
| test_mixed_credit | task + random | OOD mixed |

第一轮每个 regime 至少收集：

$$
N_{credit}\geq 4096
$$

如果 correction ratio 很小，需要更大样本：

$$
N_{credit}\geq 16384
$$

---

### 7.5 CN2 metrics

Correction fidelity：

| metric key | 含义 |
|---|---|
| `cn2/correction/cos` | $\cos(\hat{\Delta g},\Delta g)$ |
| `cn2/correction/relerr` | correction relerr |
| `cn2/correction/norm_ratio` | $\|\hat{\Delta g}\|/\|\Delta g\|$ |
| `cn2/correction/cos_p05` | correction cosine p05 |
| `cn2/correction/relerr_p95` | correction relerr p95 |

Full VJP fidelity：

| metric key | 含义 |
|---|---|
| `cn2/full/cos` | $\cos(g_{out}+\hat\Delta g,g_h)$ |
| `cn2/full/relerr` | full relerr |
| `cn2/full/norm_ratio` | full norm ratio |

Gain over identity：

Identity full relerr：

$$
\operatorname{relerr}_{id}
=
\frac{
\|g_{out}-g_h\|
}{
\|g_h\|+\epsilon
}
$$

Router full relerr：

$$
\operatorname{relerr}_{router}
=
\frac{
\|g_{out}+\hat\Delta g-g_h\|
}{
\|g_h\|+\epsilon
}
$$

Absolute gain：

$$
G_{abs}
=
\operatorname{relerr}_{id}-\operatorname{relerr}_{router}
$$

Relative gain：

$$
G_{rel}
=
\frac{
\operatorname{relerr}_{id}-\operatorname{relerr}_{router}
}{
\operatorname{relerr}_{id}+\epsilon
}
$$

W&B keys：

| metric key | 含义 |
|---|---|
| `cn2/gain/abs_over_identity` | absolute relerr gain |
| `cn2/gain/rel_over_identity` | relative gain |
| `cn2/gain/beat_identity_rate` | router relerr < identity relerr 的比例 |

Linearity：

| metric key | 含义 |
|---|---|
| `cn2/linearity/scale_error_0_5` | scale test |
| `cn2/linearity/scale_error_2_0` | scale test |
| `cn2/linearity/superposition_error` | superposition test |

Cost：

| metric key | 含义 |
|---|---|
| `cn2/cost/router_time_ms` | router 时间 |
| `cn2/cost/analytic_vjp_time_ms` | analytic VJP 时间 |
| `cn2/cost/router_memory_mb` | router memory |
| `cn2/cost/router_params` | router 参数量 |

---

### 7.6 CN2 可视化

必须包含：

1. **Correction cosine by regime**  
   x-axis: regime；y-axis: correction cosine；color: router type。

2. **Gain over identity bar chart**  
   x-axis: router type；y-axis: relative gain over identity。

3. **Correction norm vs router gain scatter**  
   x-axis: $\|\Delta g\|/\|g_{out}\|$；y-axis: gain over identity。

4. **Full relerr before / after correction**  
   paired plot：identity relerr vs router relerr。

5. **Rank vs fidelity Pareto**  
   x-axis: router rank；y-axis: correction cosine；point size: router time。

6. **Cross-regime matrix**  
   train regime × eval regime，颜色为 correction cosine 和 gain over identity。

7. **Cost vs gain Pareto**  
   x-axis: router time；y-axis: gain over identity。

---

### 7.7 CN2 成功标准

Weak success：

$$
\cos(\hat\Delta g,\Delta g)>0.6
$$

且：

$$
G_{rel}>0.1
$$

Medium success：

$$
\cos(\hat\Delta g,\Delta g)>0.8
$$

$$
\operatorname{relerr}_{\Delta}<0.5
$$

$$
G_{rel}>0.2
$$

Strong success：

$$
\cos(\hat\Delta g,\Delta g)>0.9
$$

$$
G_{rel}>0.3
$$

且 one-step local update 优于 identity。

如果 router 只能在 over-active / unstable regime 中获得 gain，而在 Pareto regime 中没有 gain，则不能进入 Stage D 主线。

---

## 8. CN3：Identity stress test

### 8.1 目标

CN3 系统寻找 identity baseline 什么时候不够。它不是为了摧毁 residual，而是为了找到：

$$
\text{identity 不再完美，但 credit geometry 仍然可控}
$$

的实验区间。

---

### 8.2 Stress axes

#### Depth stress

$$
K\in\{2,4,8\}
$$

关注随着 depth 增加：

1. identity relerr 是否上升；
2. correction norm 是否上升；
3. credit amplification 是否仍可控；
4. router gain 是否上升。

#### Branch strength stress

通过 coeff lr、alpha、branch target 控制：

$$
\operatorname{branchOverAdamW}
\in
\{0.3,0.5,0.8,1.0,1.5\}
$$

#### Dataset stress

顺序：

$$
\text{MNIST}
\rightarrow
\text{Fashion}
\rightarrow
\text{KMNIST}
\rightarrow
\text{EMNIST}
\rightarrow
\text{CIFAR-10 small}
$$

#### Model capacity stress

$$
hidden\ dim\in\{32,64,128\}
$$

$$
basis\ count\in\{8,16\}
$$

---

### 8.3 CN3 metrics

| metric key | 含义 |
|---|---|
| `cn3/identity/relerr` | identity relerr |
| `cn3/identity/cos` | identity cosine |
| `cn3/correction/norm_ratio` | correction norm ratio |
| `cn3/router/gain_over_identity` | router gain |
| `cn3/credit/amplification_p95` | amplification p95 |
| `cn3/credit/noise_gain_p95` | noise gain p95 |
| `cn3/jacobian/condition` | J condition |
| `cn3/task/test_acc` | test accuracy |
| `cn3/task/gap_vs_adamw` | gap vs AdamW |

---

### 8.4 CN3 可视化

1. **Depth vs identity relerr**  
   x-axis: depth；y-axis: identity relerr；color: dataset。

2. **Branch ratio vs identity relerr**  
   x-axis: branch / AdamW；y-axis: identity relerr。

3. **Branch ratio vs credit amplification**  
   判断 branch 强度增大是否导致 credit 不稳定。

4. **Router gain vs identity relerr**  
   判断 identity 越难，router 是否越有价值。

5. **Accuracy gap vs geometry gain**  
   x-axis: accuracy gap；y-axis: J condition reduction。

---

### 8.5 CN3 成功标准

理想 stress 区间满足：

$$
\operatorname{relerr}_{id}>0.2
$$

但：

$$
\text{credit amplification p95}<1.5
$$

$$
\text{test gap}<2\%
$$

$$
\text{Jacobian condition}<2\times \text{AdamW condition}
$$

如果能找到这种区间，就可以作为 learned correction router 的主验证场景。

---

## 9. CN4：One-step correction usefulness

### 9.1 目标

CN4 检查 learned correction router 不只是指标上接近 $\Delta g$，而是真的能改善 update。

比较 credit source：

| source | 定义 |
|---|---|
| oracle | $g_h^{teacher}$ |
| identity | $g_{out}$ |
| zero correction | identity 等价 |
| ridge correction | $g_{out}+Wg_{out}$ |
| low-rank correction | $g_{out}+R^\phi[g_{out}]$ |
| analytic correction | exact $g_h$ |
| random | random credit |

---

### 9.2 Hidden-state one-step update

对 block input hidden state 做：

$$
h\leftarrow h-\eta \hat g_h
$$

记录：

$$
\Delta L
=
L_{after}-L_{before}
$$

---

### 9.3 Parameter-level one-step functional update

对 KAN coefficients 做一次 functional update：

$$
a\leftarrow a-
\eta(M+\rho I)^{-1}
\nabla_a L
$$

分别使用不同 credit source 计算局部更新方向，比较实际 loss 变化。

---

### 9.4 CN4 metrics

| metric key | 含义 |
|---|---|
| `cn4/hidden/pred_descent` | hidden-state predicted descent |
| `cn4/hidden/actual_descent` | hidden-state actual descent |
| `cn4/hidden/descent_ratio` | actual / predicted |
| `cn4/param/pred_descent` | parameter-level predicted descent |
| `cn4/param/actual_descent` | parameter-level actual descent |
| `cn4/param/descent_ratio` | actual / predicted |
| `cn4/sign_agreement_rate` | sign agreement |
| `cn4/gain_over_identity_actual` | actual descent gain over identity |
| `cn4/update_norm_function` | function update norm |
| `cn4/phi_prime_p95_after` | update 后 $\phi'_{p95}$ |
| `cn4/jacobian_condition_after` | update 后 J condition |

---

### 9.5 CN4 可视化

1. predicted vs actual descent scatter；
2. actual descent by credit source；
3. gain over identity by router；
4. update norm vs actual descent；
5. before / after $\phi'_{p95}$；
6. before / after Jacobian condition。

---

### 9.6 CN4 成功标准

Analytic credit：

$$
\text{sign agreement}>0.85
$$

Correction router：

$$
\text{sign agreement}>0.75
$$

并且：

$$
\text{actual descent gain over identity}>10\%
$$

如果 correction router 的 VJP fidelity 通过但 one-step actual descent 不优于 identity，则不能进入 Stage D 主线。

---

## 10. CN5：AnalyticAdj joint-training precheck

### 10.1 目标

CN5 是 Stage D 的预检查。它不做完整大规模训练，只用小模型比较：

1. BP-KAN functional update；
2. analytic adjoint functional update；
3. identity credit functional update；
4. correction router functional update。

这一步要回答：

> 在真正更新参数时，analytic adjoint 是否明显优于 identity？correction router 是否能超过 identity？

---

### 10.2 方法组合

| method | credit source | update |
|---|---|---|
| BP-KAN-Sobolev | full BP | diag-to-Sobolev |
| AnalyticAdj-KAN-Sobolev | analytic full block VJP | diag-to-Sobolev |
| IdentityAdj-KAN-Sobolev | $g_h=g_{out}$ | diag-to-Sobolev |
| CorrectionRouter-KAN-Sobolev | $g_h=g_{out}+\hat\Delta g$ | diag-to-Sobolev |
| Random-KAN-Sobolev | random credit | diag-to-Sobolev |

---

### 10.3 第一轮设置

| dataset | depth | hidden dim | basis | seeds |
|---|---:|---:|---:|---:|
| MNIST | 2, 4 | 32, 64 | 8, 16 | 5 |
| Fashion | 2, 4 | 32, 64 | 8, 16 | 5 |
| KMNIST | 2 | 32 | 8 | 3 |

---

### 10.4 Metrics

| metric key | 含义 |
|---|---|
| `cn5/train/loss` | train loss |
| `cn5/val/loss` | val loss |
| `cn5/test/acc` | test acc |
| `cn5/acc_gap_vs_bp` | gap vs BP |
| `cn5/loss_auc` | loss AUC |
| `cn5/steps_to_target_acc` | steps to target |
| `cn5/credit/mean_cos_vs_bp` | credit cosine vs BP |
| `cn5/credit/worst_cos_vs_bp` | worst-layer cosine |
| `cn5/credit/norm_ratio` | norm ratio |
| `cn5/descent/sign_agreement` | sign agreement |
| `cn5/memory/peak_allocated_mb` | peak memory |
| `cn5/compute/step_time_ms` | step time |

---

### 10.5 CN5 成功标准

Analytic adjoint 成功：

$$
\text{accuracy gap vs BP}<2\%
$$

且：

$$
\text{memory or compute improves over full BP}
$$

Correction router 成功：

$$
\text{accuracy gap vs BP}<3\%
$$

并且优于 IdentityAdj：

$$
\text{acc}_{router}>
\text{acc}_{identity}+0.5\%
$$

或：

$$
\text{loss AUC}_{router}<0.9\times
\text{loss AUC}_{identity}
$$

如果 AnalyticAdj 成功但 CorrectionRouter 不成功，Stage D 主线应采用 AnalyticAdj，而 learned router 留作后续。

---

## 11. CN6：Scaled residual 与 non-residual ablation

### 11.1 目标

CN6 不是主线，只是为了回答：

> 如果我们减弱或去掉 identity path，router 是否更有必要？这种必要性是否伴随几何崩坏？

---

### 11.2 模型变体

#### Base residual

$$
h_{out}=h+
\alpha\operatorname{KAN}(\operatorname{LN}(h))
$$

#### Scaled residual

$$
h_{out}=h+
\beta\alpha\operatorname{KAN}(\operatorname{LN}(h))
$$

其中：

$$
\beta\in\{0.5,1.0,2.0,4.0\}
$$

#### Gated residual

$$
h_{out}=h+
\gamma(h)\alpha\operatorname{KAN}(\operatorname{LN}(h))
$$

#### Non-residual KAN

$$
h_{out}=\operatorname{KAN}(\operatorname{LN}(h))
$$

Non-residual 只作为 stress / negative control。

---

### 11.3 CN6 metrics

| metric key | 含义 |
|---|---|
| `cn6/identity/relerr` | identity relerr |
| `cn6/router/gain_over_identity` | router gain |
| `cn6/jacobian/condition` | Jacobian condition |
| `cn6/credit/amplification_p95` | credit amplification |
| `cn6/noise/gain_p95` | noise gain |
| `cn6/test_acc` | test accuracy |
| `cn6/diverged` | 是否发散 |

---

### 11.4 CN6 判定

如果 non-residual 让 identity relerr 升高，但 Jacobian condition 和 amplification 也大幅失控，则它只能作为负对照，不能作为主线。

如果 scaled residual 中存在某个 $\beta$，使得：

$$
\operatorname{relerr}_{id}>0.2
$$

且：

$$
\text{amplification p95}<1.5
$$

$$
\text{test gap}<2\%
$$

那么这个 setting 可以作为 learned router 的主 stress test。

---

## 12. W&B dashboard 设计

Stage C-next 需要建立 8 个 dashboard 页面。

### Page 1：Checkpoint regime map

图表：

1. branch / AdamW by method；
2. test acc gap vs AdamW；
3. no-KAN acc drop；
4. $\phi'_{p95}$ by regime；
5. Jacobian condition by regime；
6. active basis fraction。

---

### Page 2：Correction geometry

图表：

1. correction norm ratio by regime；
2. identity relerr by regime；
3. branch ratio vs correction norm scatter；
4. correction rank heatmap；
5. correction alignment histogram。

---

### Page 3：Residual correction router

图表：

1. correction cosine curve；
2. correction relerr curve；
3. gain over identity bar chart；
4. full VJP relerr before / after correction；
5. rank vs fidelity Pareto；
6. cross-regime matrix。

---

### Page 4：Identity stress test

图表：

1. depth vs identity relerr；
2. branch ratio vs identity relerr；
3. branch ratio vs amplification；
4. router gain vs identity relerr；
5. dataset complexity vs identity relerr。

---

### Page 5：One-step usefulness

图表：

1. predicted vs actual descent scatter；
2. actual descent by credit source；
3. gain over identity in actual descent；
4. before / after $\phi'_{p95}$；
5. before / after Jacobian condition。

---

### Page 6：Joint-training precheck

图表：

1. train / val loss curves；
2. test accuracy by method；
3. loss AUC by method；
4. credit cosine vs BP；
5. memory vs accuracy Pareto；
6. step time vs accuracy。

---

### Page 7：Scaled residual / non-residual ablation

图表：

1. beta vs identity relerr；
2. beta vs Jacobian condition；
3. beta vs amplification p95；
4. beta vs router gain；
5. non-residual failure table。

---

### Page 8：Failure analysis

需要展示 failure table，并按 failure type 分组。

---

## 13. Failure table

所有 failure 都必须结构化记录。

| failure type | 触发条件 | 解释 |
|---|---|---|
| `identity_too_strong` | correction ratio < $0.05$ | 不适合验证 router |
| `correction_unstable` | correction ratio 高且 amplification 高 | branch correction 不稳定 |
| `router_no_gain` | gain over identity < $0.05$ | router 无价值 |
| `router_bad_correction_cos` | correction cosine < $0.6$ | correction 学不准 |
| `router_norm_miscalibrated` | correction norm ratio < $0.5$ 或 > $1.5$ | norm 校准失败 |
| `cross_regime_fail` | cross-regime gain < $0$ | router 不泛化 |
| `one_step_no_gain` | actual descent 不优于 identity | router 不支持更新 |
| `analytic_adj_not_useful` | AnalyticAdj 不优于 IdentityAdj | analytic correction 对训练贡献不足 |
| `scaled_residual_unstable` | amplification p95 > $2$ 或 J cond > $5\times$ AdamW | scale stress 失控 |
| `non_residual_unstable` | non-residual 发散或 J condition 极高 | 非残差不可用 |

Failure table 字段：

| field | 含义 |
|---|---|
| `failure/type` | failure 类型 |
| `failure/phase` | CN0-CN6 |
| `failure/dataset` | 数据集 |
| `failure/model` | 模型 |
| `failure/regime` | regime |
| `failure/router_type` | router 类型 |
| `failure/layer` | 层号 |
| `failure/seed` | seed |
| `failure/metric_name` | 触发指标 |
| `failure/metric_value` | 数值 |
| `failure/threshold` | 阈值 |
| `failure/diagnosis` | 诊断 |
| `failure/recommended_action` | 建议 |

---

## 14. 第一轮最小执行包

为了避免过宽，第一轮 Stage C-next 只做以下实验。

### Experiment N1：Correction geometry audit

数据集：MNIST、Fashion-MNIST、KMNIST。  
模型：depth $2$，hidden $32$，basis $8$。  
Regime：conservative、Pareto、active、AdamW。  
Seeds：5。

目标：确认 $\Delta g$ 在不同 regime 中的大小、秩、稳定性。

成功：找到 correction ratio 在 $0.15$ 到 $0.40$ 且 amplification p95 < $1.5$ 的 setting。

---

### Experiment N2：Residual correction router

数据集：MNIST、Fashion-MNIST。  
模型：depth $2$，hidden $32$。  
Router：zero correction、ridge correction、diagonal correction、diag-lowrank correction。  
Regime：Pareto、active。  
Seeds：5。

目标：判断 router 是否能学会 $\Delta g$ 并打败 identity。

成功：gain over identity > $20\%$，correction cosine > $0.8$。

---

### Experiment N3：Depth stress

数据集：MNIST、Fashion-MNIST。  
Depth：$2,4,8$。  
Hidden：$64$。  
Regime：Pareto。  
Seeds：3。

目标：判断深度增加后 identity baseline 是否变弱，router gain 是否增加。

成功：depth $4$ 或 $8$ 中 identity relerr > $0.2$，但 amplification p95 < $1.5$。

---

### Experiment N4：Dataset stress

数据集：Fashion-MNIST、KMNIST、EMNIST Balanced small。  
模型：depth $4$，hidden $64$，basis $16$。  
Regime：Pareto。  
Seeds：3。

目标：判断更复杂数据是否让 non-identity correction 更必要。

成功：identity relerr 上升，router gain 上升，test gap < $2\%$。

---

### Experiment N5：AnalyticAdj vs IdentityAdj precheck

数据集：MNIST、Fashion-MNIST。  
模型：depth $2,4$，hidden $32,64$。  
方法：BP-KAN-Sobolev、AnalyticAdj-KAN-Sobolev、IdentityAdj-KAN-Sobolev、CorrectionRouter-KAN-Sobolev。  
Seeds：5。

目标：判断 Stage D 该优先 analytic adjoint 还是 identity / correction router。

成功：AnalyticAdj gap vs BP < $2\%$；CorrectionRouter 优于 IdentityAdj 至少 $0.5\%$ accuracy 或 $10\%$ loss AUC。

---

## 15. Stage D 决策规则

### 情况 A：AnalyticAdj 成功，IdentityAdj 也接近

结论：当前 residual DG-KAN 不需要 learned router。Stage D 主线采用：

$$
\text{AnalyticAdj-KAN-Sobolev}
$$

并把 learned router 留作后续。

---

### 情况 B：AnalyticAdj 明显优于 IdentityAdj

结论：KAN branch correction 对训练有实际贡献。Stage D 应保留 analytic adjoint，并继续研究 correction router。

---

### 情况 C：Correction router 在 Pareto / depth stress 中打败 identity

结论：learned router 有价值，可以进入 Stage D 的 Router-KAN-Sobolev 路线。

---

### 情况 D：Router 只有在 over-active / non-residual 中有效

结论：router 价值来自不稳定困难场景，不进入主线。

---

### 情况 E：更深 / 更复杂数据中 identity 失效，但 Pareto 仍稳

这是最理想结果。说明 residual DG-KAN 在真实任务中仍保持几何健康，同时 learned correction router 变得必要。

---

## 16. 最终预期结论

如果 Stage C-next 成功，预期结论应是：

> Stage C showed that residual DG-KAN local VJP is close to identity, making identity routing a strong baseline. Stage C-next reframes router learning as residual correction prediction. In regimes where KAN branch correction is non-trivial but geometry remains stable, learned correction routers can reduce identity residual error and improve one-step descent. This clarifies when learned local credit routing is necessary, and when analytic adjoint or identity routing is sufficient.

中文表述：

> Stage C 说明 residual DG-KAN 的 local VJP 很接近 identity，因此 identity routing 是强 baseline。Stage C-next 将 router 学习重新定义为 residual correction prediction。在 KAN branch correction 非平凡但几何仍稳定的 regime 中，learned correction router 应当减少 identity 剩余误差，并改善 one-step descent。这能明确 learned local credit routing 何时必要，以及何时 analytic adjoint 或 identity routing 已经足够。

---

## 17. 一句话总结

Stage C-next 的核心不是去掉 residual，也不是盲目换复杂数据集，而是：

$$
\boxed{
\text{保留 residual 稳定主线，放大并隔离非 identity correction，检验 router 是否能学会 identity 之外的那部分 credit。}
}
$$

只有当 learned router 能在 correction-level 明确打败 zero correction / identity，并且在 one-step local update 或 joint-training precheck 中带来实际收益，才能说 learned router 对 DG-KAN 有必要。否则 Stage D 应优先使用 analytic KAN adjoint，而不是强行推进 learned router。
