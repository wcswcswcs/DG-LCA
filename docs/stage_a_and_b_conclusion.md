# Stage A and Stage B 实验结论总结

> 文件名：`stage_a_and_b_conclusion.md`  
> 主题：DG-KAN / DG-LCA 当前实验阶段结论  
> 范围：Stage A、Stage B、Stage B-next、MNIST classification drilldown  
> 公式格式：Typora 友好，使用 `$...$` 与 `$$...$$`

---

## 0. 总体结论

目前实验已经把最初的模糊问题拆成了较清楚的几条结论：

1. **DG-KAN 架构本身是可训练的。**  
   Stage A 说明 DG-KAN 在多个 toy / small classification 数据集上没有数值崩溃，能够稳定训练。

2. **DG-KAN 的局部几何明显比普通 KAN 稳定。**  
   Stage A 中普通 KAN 的 Jacobian condition 达到 $10^8$ 量级，而 DG-KAN 降到个位数或接近个位数。这说明 residual / geometry-aware 设计确实改善了局部谱性质。

3. **Functional-space update 对 KAN edge function 是有价值的，但它不是通用 optimizer。**  
   Stage B 证明：functional update 应该主要作用在 KAN edge coefficients 上；非 KAN 参数仍应使用 AdamW / Adam 这类成熟优化器。粗暴地把非 KAN 参数也换成 SGD / functional update 会导致欠拟合。

4. **`diag`、`sobolev`、`diag_to_sobolev` 分工已经比较清楚。**  
   - `diag`：早期下降最快，适合 warmup；
   - `sobolev`：最终函数拟合质量最好，几何更可控；
   - `diag_to_sobolev`：当前最稳的 schedule，能同时保留 `diag` 的速度和 `sobolev` 的最终质量。

5. **RKHS 当前不进入主线。**  
   RKHS 在 raw condition 很高时失败；即使通过 damping 降低 condition 后，性能仍然明显劣化。因此当前 RKHS 失败不只是 condition number 问题，也可能是 kernel geometry 与当前 RBF-KAN 参数化不匹配。

6. **MNIST small 的 hybrid failure 已经被重新解释。**  
   这不是泛化失败，也不是 basis coverage 问题，而是 functional update 的 effective coefficient step 太小，导致 KAN branch under-active。提高 KAN coefficient learning rate 后，hybrid 可以追上甚至略超 AdamW，但高步长会削弱几何优势。

7. **目前可以进入 Stage C，但要带着三个 branch activity regime 进入。**  
   Stage C 应该同时测试：
   - conservative regime：branch under-active，几何最稳；
   - Pareto regime：精度 gap < 1%，几何明显优于 AdamW；
   - accuracy-parity regime：精度追平或略超 AdamW，但几何优势减弱。

---

## 1. 关键 notation

### 1.1 Residual DG-KAN block

当前分类实验使用的核心结构可以写成：

$$
h_{k+1}
=
h_k
+
\alpha_k \operatorname{KAN}_k(\operatorname{LN}(h_k))
$$

其中 KAN block 为：

$$
\operatorname{KAN}_k(x)_j
=
\sum_i \phi_{k,ji}(x_i)
$$

edge function 使用 basis 表示：

$$
\phi_{ji}(t)
=
\sum_m a_{jim}B_m(t)
$$

当前实现中常用 RBF basis：

$$
B_m(t)
=
\exp(-\gamma(t-c_m)^2)
$$

---

### 1.2 KAN local VJP / adjoint

对 KAN-like 层：

$$
y_j=\sum_i\phi_{ji}(x_i)
$$

前向扰动为：

$$
\delta y_j
=
\sum_i \phi'_{ji}(x_i)\delta x_i
$$

所以局部 VJP 为：

$$
g_{x_i}
=
\sum_j g_{y_j}\phi'_{ji}(x_i)
$$

即：

$$
g_x
=
\Phi'(x)^\top g_y
$$

这说明 KAN 的 credit transport 直接由 edge function 的一阶导 $\phi'(x)$ 控制。

---

### 1.3 Functional-space update

普通 coefficient update：

$$
a
\leftarrow
a-\eta\nabla_a\mathcal L
$$

函数空间预条件更新：

$$
a
\leftarrow
a-\eta(M+\rho I)^{-1}\nabla_a\mathcal L
$$

其中 $M$ 是函数空间 metric / Gram matrix，$\rho I$ 是 damping。

Sobolev metric：

$$
\|\phi\|_{\mathcal H}^2
=
\int \phi(t)^2dt
+
\alpha\int \phi'(t)^2dt
+
\beta\int \phi''(t)^2dt
$$

对应 Gram matrix：

$$
M_{mn}
=
\int B_mB_n
+
\alpha\int B_m'B_n'
+
\beta\int B_m''B_n''
$$

---

### 1.4 Branch ratio

Residual block 中 KAN branch 的相对贡献定义为：

$$
r_{\text{branch},k}
=
\frac{
\|\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))\|
}{
\|h_k\|+\epsilon
}
$$

全局 branch ratio 是各层 / batch 的平均值：

$$
r_{\text{branch}}
=
\operatorname{mean}_{k,n}
r_{\text{branch},k}^{(n)}
$$

为了和 AdamW 比较，定义：

$$
\text{branch / AdamW}
=
\frac{
r_{\text{branch}}^{\text{method}}
}{
r_{\text{branch}}^{\text{AdamW}}
}
$$

这个指标衡量的是：

> 当前方法下，KAN branch 的输出强度相当于 AdamW 下的多少。

经验解释：

| branch / AdamW | 解释 |
|---:|---|
| $<0.2$ | under-active，KAN branch 基本没起作用 |
| $0.3$ 到 $0.6$ | Pareto 区间，精度接近且几何更稳 |
| $0.6$ 到 $1.2$ | active，精度优先 |
| $>2$ | over-active，可能导致 $\phi'$ 和 Jacobian 爆炸 |

---

### 1.5 No-KAN ablation

关闭 KAN branch 等价于把 residual block 改成：

$$
h_{k+1}=h_k
$$

也就是：

$$
\alpha_k\operatorname{KAN}_k(\operatorname{LN}(h_k))=0
$$

定义 no-KAN accuracy drop：

$$
\Delta_{\text{no-KAN}}
=
\operatorname{Acc}_{\text{full}}
-
\operatorname{Acc}_{\text{no-KAN}}
$$

如果 $\Delta_{\text{no-KAN}}$ 很小，说明 KAN branch 对最终分类贡献很小。

---

## 2. Stage A 结论：DG-KAN 架构可训练且几何稳定

### 2.1 Stage A 做了什么

Stage A 主要验证 DG-KAN 架构本身是否可训练，是否数值稳定，以及是否相较普通 KAN 有更健康的局部几何。

已有 Stage A S1 结果包括：

- 共 $105$ 个 run；
- 数据集：Two Moons、MNIST、Fashion-MNIST；
- 模型：多种 KAN / DG-KAN / baseline；
- seeds：$5$；
- 全部使用 GPU；
- 无 NaN / Inf。

---

### 2.2 主要结论

#### 结论 A1：DG-KAN 可训练，数值稳定

所有 DG-KAN run 均可正常收敛，没有 NaN / Inf。这说明当前 residual DG-KAN 结构至少具备基础可训练性。

---

#### 结论 A2：DG-KAN 几何稳定性明显优于普通 KAN

Stage A 的关键结果是 Jacobian condition：

| dataset | KAN condition | DG-KAN condition |
|---|---:|---:|
| Two Moons | $5.64\times 10^8$ | $4.18$ |
| MNIST | $4.21\times 10^8$ | $2.00$ |
| Fashion-MNIST | $1.54\times 10^8$ | $1.60$ |

这说明普通 KAN 在当前设置中局部 Jacobian 极其病态，而 DG-KAN 的 residual / normalized / geometry-aware 结构显著改善了谱性质。

---

#### 结论 A3：DG-KAN 精度尚未全面超过普通 KAN 或 AdamW baseline

Stage A 的几何结论很强，但性能结论更保守。当前不能声称：

$$
\text{DG-KAN accuracy} > \text{all baselines}
$$

更准确的表述是：

> DG-KAN 显著改善局部几何稳定性，并保持可训练性；但最终精度和收敛速度仍需要通过 Stage B 的 functional update 与 optimizer 设计继续提升。

---

### 2.3 Stage A 对后续的意义

Stage A 证明：

1. residual DG-KAN 是合理主架构；
2. Jacobian condition 是必要监控指标；
3. 单纯稳定不等于高精度；
4. 后续需要专门研究 KAN edge coefficients 的更新几何。

这直接引出了 Stage B：

> 在真实 BP credit 下，只改变 KAN edge function 的 coefficient 更新几何，观察是否改善收敛、稳定性和函数几何。

---

## 3. Stage B 初始结果：functional update 的基本角色

### 3.1 Stage B 目标

Stage B 不训练 router，只使用真实 BP credit，验证：

> 改变 KAN edge function coefficient 的更新几何，是否能改善收敛速度、最终拟合质量和局部几何稳定性。

---

### 3.2 回归 pilot 的核心结论

回归任务包括：

- $y=\sin x$；
- $y=\sin x+0.3\sin 5x$；
- $y=\sin x_1+\cos x_2$。

比较方法包括：

- `coeff_sgd`;
- `coeff_adam`;
- `coeff_adamw`;
- `diag`;
- `sobolev`;
- `rkhs`。

核心发现：

| 方法 | 主要特点 |
|---|---|
| `diag` | 收敛最快，loss AUC 最低 |
| `sobolev` | 最终 test loss 最好 |
| `rkhs` | 当前明显失败，condition number 过高 |
| `coeff_adamw` | 稳定 baseline，但回归最终拟合不如 Sobolev |

---

### 3.3 `diag` 的定位

`diag` 是 coefficient-wise preconditioner：

$$
M=\operatorname{diag}(m_1,\dots,m_M)
$$

更新为：

$$
a_m
\leftarrow
a_m
-
\eta
\frac{1}{m_m+\rho}
\frac{\partial\mathcal L}{\partial a_m}
$$

它不显式建模 basis 之间的函数空间耦合，但能快速修正不同 coefficient 的尺度。

实验结论：

> `diag` 是很强的 early-stage optimizer，适合作为 warmup。

---

### 3.4 `sobolev` 的定位

Sobolev update 使用函数空间 metric，考虑函数值、斜率和曲率。它更符合 KAN edge function 的函数属性。

回归任务中，Sobolev 在三个任务上最终 test loss 最好，尤其在高频任务 $y=\sin x+0.3\sin 5x$ 上优势明显。

实验结论：

> `sobolev` 是更好的 final-quality / geometry-aware optimizer，但不总是最快。

---

### 3.5 `rkhs` 的定位

RKHS raw metric 的 precondition condition number 约为：

$$
\kappa(M_{\text{RKHS}})\approx 4720.9
$$

明显高于 Sobolev / diag 的约 $29.65$。结果在回归任务中严重劣化。

后续 damping / lengthscale 修复后，即使 condition 降到较低，性能仍然远差于 Sobolev。因此当前结论是：

> RKHS 暂不进入主线。当前失败不只是 condition number 问题，也可能是 kernel geometry 与当前 RBF-KAN 参数化不匹配。

---

## 4. Stage B-next：系统实验结论

### 4.1 B0：pilot reproduction

B0 复现了初始趋势：

1. `diag` 仍然是最快下降方法；
2. `sobolev` 仍然是最终 test loss 最好的主方法；
3. `rkhs_raw` 仍然失败；
4. `sign_agreement=1.0` 饱和，不能区分方法。

因此后续不应只看 sign agreement，而应看：

- `loss_auc_val`;
- `test_loss`;
- `descent/ratio_median`;
- `descent/pred_actual_corr`;
- `bad_step_total_positive_loss_increase`.

---

### 4.2 B1：`diag_to_sobolev` schedule 成立

`diag_to_sobolev` 的定义：

前 $T_w$ steps 使用 diagonal Sobolev preconditioner：

$$
d_t^{\text{diag}}
=
\operatorname{diag}(M_{\text{Sob}}+\rho I)^{-1}\nabla_a\mathcal L
$$

之后切换到 full Sobolev preconditioner：

$$
d_t^{\text{sob}}
=
(M_{\text{Sob}}+\rho I)^{-1}\nabla_a\mathcal L
$$

实验结果：

- 9 个 task / warmup 组合全部 weak success；
- `test_loss <= 1.1 * Sobolev`;
- `val_AUC <= 1.1 * diag`.

关键结论：

> `diag_to_sobolev` 能同时保留 `diag` 的早期速度和 `sobolev` 的最终质量，是当前最可靠的 default functional update schedule。

推荐默认：

| 任务类型 | 推荐 warmup |
|---|---|
| 1D / 高频回归 | $0.10$ |
| 2D 回归 | $0.25$ |
| 分类任务 | 需要按 branch ratio 调整 |

---

### 4.3 B2：classification hybrid 的必要性

Two Moons 和 MNIST small 结果表明：

> functional update 不应该替代所有参数的 optimizer。

`all_functional_other_sgd` 在 Two Moons 和 MNIST 上都明显欠拟合。原因是：

- KAN edge coefficients 有函数空间结构，适合 functional update；
- stem、head、LayerNorm、普通 linear 参数没有自然的一维函数空间 metric，仍然需要 AdamW。

因此分类任务的正确默认形式是：

$$
\boxed{
\text{KAN coefficients: functional update}
+
\text{non-KAN parameters: AdamW}
}
$$

---

### 4.4 Two Moons hybrid 结论

Two Moons 上 hybrid 方法接近 AdamW accuracy，同时有更稳定的几何：

| method | test acc | max J cond | $\phi'_{p95}$ |
|---|---:|---:|---:|
| all AdamW | $0.9973$ | $4.86$ | $0.0613$ |
| hybrid Sobolev | $0.9967$ | $2.91$ | $0.0337$ |
| hybrid diag-to-Sobolev | $0.9967$ | $2.74$ | $0.0291$ |

结论：

> Two Moons 上，hybrid functional update 可以在几乎不损失 accuracy 的情况下显著降低 Jacobian condition 和 $\phi'_{p95}$。

---

### 4.5 MNIST 初始 hybrid failure

初始 MNIST small 上，hybrid 没追上 AdamW：

| method | train acc | test acc | max J cond | $\phi'_{p95}$ |
|---|---:|---:|---:|---:|
| all AdamW | $0.9689$ | $0.9133$ | $1.79$ | $0.0440$ |
| hybrid Sobolev | $0.9287$ | $0.8993$ | $1.12$ | $0.0139$ |

关键判断：

> MNIST hybrid 初始失败不是泛化失败，因为 train acc 也明显低；它更像优化不足 / KAN branch under-active。

这个判断后来被 MNIST drilldown 证实。

---

### 4.6 B3：Sobolev sweep 结论

在高频任务 $y=\sin x+0.3\sin 5x$ 上，Sobolev $\alpha$ 过大会过度平滑。

稳定区间：

$$
\alpha\in[0,10^{-2}]
$$

推荐默认：

$$
\alpha=10^{-2},\quad \rho=10^{-3}
$$

过大平滑项，例如 $\alpha=1$，会显著压低曲率，但 test loss 严重劣化。

结论：

> Sobolev 不是“越平滑越好”，而是提供函数几何控制。任务需要的曲率不能被过度抑制。

---

### 4.7 B4：RKHS repair 失败

RKHS repair 中，damping 和 lengthscale 调整可以降低 condition number，但性能仍然远差于 Sobolev。

因此：

> RKHS 失败不只是 condition number 问题。当前 RKHS metric 的 inductive bias / kernel scale / update geometry 与当前 RBF-KAN coefficient 参数化不匹配。

当前决策：

$$
\boxed{
\text{RKHS 不进入 Stage C / D 主方法}
}
$$

---

## 5. MNIST classification drilldown 结论

### 5.1 MNIST failure 的核心问题

MNIST small 上 hybrid 初始失败的主要原因是：

$$
\boxed{
\text{functional update effective step too small}
\rightarrow
\text{KAN branch under-active}
}
$$

不是：

- 泛化失败；
- basis coverage mismatch；
- 随机噪声；
- 单纯模型容量不足。

---

### 5.2 Under-active 证据

MNIST 初始结果：

| method | branch ratio | branch / AdamW | no-KAN acc drop | test acc |
|---|---:|---:|---:|---:|
| all AdamW | $0.2795$ | $1.000$ | $0.0203$ | $0.9133$ |
| hybrid diag | $0.0481$ | $0.172$ | $0.0037$ | $0.8987$ |
| hybrid Sobolev | $0.0449$ | $0.161$ | $0.0033$ | $0.8993$ |

解释：

1. hybrid 的 branch ratio 只有 AdamW 的 $16\%$ 到 $17\%$；
2. 关闭 KAN branch 后，AdamW accuracy 掉约 $2.03\%$；
3. 关闭 KAN branch 后，hybrid accuracy 只掉约 $0.33\%$ 到 $0.37\%$；
4. 所以 hybrid 中的 KAN branch 对最终分类几乎没有充分贡献。

结论：

> hybrid functional update 初始设置下，KAN branch 没有真正参与分类。

---

### 5.3 Fashion-MNIST 复现 under-active 模式

Fashion-MNIST 上也出现同样现象：

| method | branch / AdamW | no-KAN acc drop |
|---|---:|---:|
| hybrid diag | $0.204$ | $0.0007$ |
| hybrid Sobolev | $0.191$ | $0.0003$ |

说明：

> KAN branch under-active 不是 MNIST split 的偶然现象，而是当前 small classification 设置下的共性问题。

---

### 5.4 Basis coverage 不是主因

MNIST drilldown 记录到：

- active basis fraction 约 $0.75$；
- out-of-grid fraction 约 $10^{-4}$ 到 $8\times 10^{-4}$；
- 没有触发 basis coverage failure。

因此：

$$
\boxed{
\text{basis coverage mismatch is not the primary cause}
}
$$

---

### 5.5 提高 coefficient lr 可以修复 under-active

高步长实验：

| method | coeff lr | rest lr | test acc | branch / AdamW |
|---|---:|---:|---:|---:|
| hybrid Sobolev | $1.0$ | $0.003$ | $0.9140$ | $0.831$ |
| hybrid diag-to-Sobolev | $1.0$ | $0.003$ | $0.9134$ | $0.834$ |
| hybrid diag | $1.0$ | $0.003$ | $0.9102$ | $0.888$ |
| all AdamW | - | - | $0.9114$ | $1.000$ |

结论：

> 当 KAN branch 的相对强度从 $0.16$ 到 $0.20$ 提升到约 $0.8$ 时，MNIST 分类精度恢复到 AdamW 水平甚至略超。

这说明原始 failure 的核心不是 functional update 方向错误，而是 effective update strength 太小。

---

### 5.6 高步长恢复精度，但牺牲部分几何优势

高步长下：

| method | test acc | $\phi'_{p95}$ | max J cond |
|---|---:|---:|---:|
| all AdamW | $0.9114$ | $0.0440$ | $1.7646$ |
| hybrid Sobolev | $0.9140$ | $0.0648$ | $2.1416$ |
| hybrid diag-to-Sobolev | $0.9134$ | $0.0621$ | $2.0942$ |

解释：

> 高 coefficient lr 可以恢复精度，但 $\phi'_{p95}$ 和 Jacobian condition 会升高，部分几何稳定优势消失。

---

### 5.7 Pareto regime 是当前最重要的分类发现

中等步长：

$$
\text{coeff\_lr}=0.3,\quad \text{rest\_lr}=0.003
$$

结果：

| method | test acc | test gap | branch / AdamW | $\phi'_{p95}$ | max J cond |
|---|---:|---:|---:|---:|---:|
| hybrid Sobolev | $0.9048$ | $0.0066$ | $0.417$ | $0.0273$ | $1.3643$ |
| hybrid diag-to-Sobolev | $0.9032$ | $0.0082$ | $0.430$ | $0.0236$ | $1.3243$ |
| all AdamW | $0.9114$ | $0.0000$ | $1.000$ | $0.0440$ | $1.7646$ |

这满足：

$$
\text{test acc gap}<1\%
$$

同时：

$$
\text{Jacobian condition}<0.8\times \text{AdamW}
$$

$$
\phi'_{p95}<0.8\times \text{AdamW}
$$

结论：

> `coeff_lr=0.3, rest_lr=0.003` 是当前最有价值的 geometry-preserving Pareto regime。

---

### 5.8 Naive Adam moment 失败

Adam-style moment 初筛显示：

- simple momentum 没明显帮助；
- naive Adam moment + Sobolev 会导致 branch 爆炸；
- branch ratio 达到 AdamW 的 $45\times$ 到 $58\times$；
- $\phi'_{p95}$ 达到 $6$ 到 $10$；
- Jacobian condition 达到 $10^3$ 量级；
- test accuracy 反而下降。

结论：

> Adam-style moment 不能直接叠加到 Sobolev preconditioner 上。若继续研究，必须加入 trust region、update clipping、branch-ratio control 或 function-delta normalization。

---

## 6. 当前最稳的方法结论

### 6.1 回归任务默认方法

推荐：

$$
\boxed{
\text{diag-to-Sobolev}
}
$$

其中：

- 1D / 高频回归：warmup frac $0.10$；
- 2D 回归：warmup frac $0.25$；
- Sobolev 默认 $\alpha=10^{-2},\rho=10^{-3}$。

---

### 6.2 分类任务默认方法

分类任务应使用 hybrid：

$$
\boxed{
\text{KAN coefficients: functional update}
+
\text{non-KAN parameters: AdamW}
}
$$

当前建议维护三个 regime：

| regime | 典型设置 | 作用 |
|---|---|---|
| conservative | coeff lr 小，如 $0.1$ | 几何很稳，但 branch under-active |
| Pareto | coeff lr 约 $0.3$，rest lr $0.003$ | 精度 gap < 1%，几何明显更稳 |
| accuracy parity | coeff lr 约 $1.0$，rest lr $0.003$ | 精度追平 / 略超 AdamW，但几何优势减弱 |

---

### 6.3 当前不推荐的方法

| 方法 | 当前决策 | 原因 |
|---|---|---|
| `all_functional_other_sgd` | 不进入主线 | 非 KAN 参数欠拟合 |
| `rkhs_raw` | 不进入主线 | condition number 高且性能差 |
| repaired RKHS | 暂停 | 即使 condition 降低，性能仍差 |
| naive Sobolev + Adam moment | 不进入主线 | branch 爆炸，Jacobian 失控 |

---

## 7. 当前仍未证明的内容

尽管 Stage A/B 取得了清楚进展，但以下内容还没有被证明：

1. **learned credit router 尚未验证。**  
   Stage B 使用真实 BP credit 或解析更新，尚未证明 $C_k^\phi$ 能稳定学习 local VJP。

2. **analytic KAN adjoint 尚未系统验证。**  
   Stage C 需要比较 analytic KAN VJP 与 autograd VJP。

3. **multi-layer self-routed credit 尚未验证。**  
   尚未证明 router 误差不会随深度爆炸。

4. **low-memory sufficient statistics 尚未验证。**  
   尚未比较 full BP、checkpointing、analytic KAN adjoint、sufficient statistics 的显存和计算开销。

5. **DG-LCA 不能被描述为 BP 替代。**  
   当前只能说 functional-space update 和 DG-KAN 架构在局部实验上有价值。

---

## 8. 对 Stage C 的建议

Stage C 应该研究：

$$
\text{analytic KAN adjoint}
\quad
\text{and}
\quad
\text{learned local credit router}
$$

建议带入三个分类 regime：

| regime | 目的 |
|---|---|
| conservative | 看几何最稳时 VJP / router 是否最容易 |
| Pareto | 看当前最实用平衡点 |
| accuracy parity | 看高精度但几何更强时 credit 是否更难 |

Stage C 的关键问题是：

> Functional update 改善的几何，是否会让 local credit transport 更稳定、更容易被 router 学习？

也就是验证链条：

$$
\text{functional update}
\rightarrow
\text{healthier KAN geometry}
\rightarrow
\text{more stable credit transport}
\rightarrow
\text{easier local router}
$$

---

## 9. 目前可以写进论文/报告的核心结论

### 9.1 架构层面

> Residual DG-KAN 是可训练且几何更稳定的 KAN-like 主架构。相比普通 KAN，DG-KAN 显著降低 Jacobian condition。

---

### 9.2 更新几何层面

> KAN edge coefficients 更适合使用 function-space preconditioning，而不是普通 coefficient-space update。Sobolev metric 在回归任务上显著改善最终 test loss；diagonal preconditioning 显著改善早期收敛速度；diag-to-Sobolev schedule 同时保留两者优势。

---

### 9.3 分类任务层面

> Classification 中 functional update 必须采用 hybrid optimizer：KAN coefficients 使用 functional update，非 KAN 参数使用 AdamW。初始 hybrid 在 MNIST 上的失败来自 KAN branch under-active，而不是泛化失败或 basis coverage 问题。

---

### 9.4 Branch activity 层面

> Branch ratio 和 no-KAN ablation 是解释 DG-KAN 分类性能的关键诊断指标。KAN branch 太弱时模型欠拟合；branch 过强时几何失控；中间存在 Pareto regime，可用小于 1% 的 accuracy gap 换取更稳定的 Jacobian 和更低的 $\phi'_{p95}$。

---

### 9.5 方法边界

> 当前结果支持 DG-KAN functional update 主线，但还不支持 learned router、macro-router 或低显存训练 claim。这些需要 Stage C/D/E 继续验证。

---

## 10. 下一步优先级

### Priority 1：Stage C analytic KAN adjoint

验证：

$$
g_x=\Phi'(x)^\top g_y
$$

与 autograd VJP 一致性。

指标：

- VJP cosine；
- relative error；
- norm ratio；
- compute time；
- memory cost。

---

### Priority 2：Stage C learned local router

比较：

- analytic KAN adjoint；
- learned diagonal router；
- learned low-rank router；
- synthetic gradient；
- feedback alignment。

必须在 conservative / Pareto / accuracy-parity 三个 regime 下都测。

---

### Priority 3：Fashion-MNIST Pareto 复核

当前 Pareto regime 已在 MNIST 上成立，需要在 Fashion-MNIST 上复核：

$$
\text{coeff\_lr}=0.3,\quad \text{rest\_lr}=0.003
$$

以及：

$$
\text{coeff\_lr}=1.0,\quad \text{rest\_lr}=0.003
$$

---

### Priority 4：Branch-ratio targeted schedule

从固定 coefficient lr 进一步升级到 branch-ratio 控制：

$$
\eta_{\text{coeff},t+1}
=
\eta_{\text{coeff},t}
\exp\left(
\gamma(r_{\text{target}}-r_t)
\right)
$$

其中：

$$
r_t=
\frac{
\|\alpha\operatorname{KAN}(\operatorname{LN}(h))\|
}{
\|h\|+\epsilon
}
$$

目标：

$$
r_{\text{target}}\in\{0.4,0.6,0.8\}
$$

---

### Priority 5：Trust-region Adam moment

重新测试 Adam-style moment，但必须加入：

- function-delta clipping；
- branch ratio upper bound；
- $\phi'_{p95}$ clipping；
- Jacobian condition guard。

否则 naive Adam moment 会导致 branch 爆炸。

---

## 11. 最终阶段性判断

当前阶段结论可以概括为：

$$
\boxed{
\text{DG-KAN architecture is trainable and geometrically stable}
}
$$

$$
\boxed{
\text{Functional-space update is useful for KAN edge coefficients}
}
$$

$$
\boxed{
\text{diag-to-Sobolev is the strongest current update schedule}
}
$$

$$
\boxed{
\text{Classification requires hybrid optimization and branch activity control}
}
$$

$$
\boxed{
\text{MNIST failure is explained by KAN branch under-activity}
}
$$

$$
\boxed{
\text{Stage C can proceed, but learned credit routing and low-memory claims remain unproven}
}
