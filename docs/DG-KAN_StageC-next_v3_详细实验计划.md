# DG-KAN Stage C-next v3 详细实验计划：Derivative-aware Span Correction Router 与 Router Stage-D 决策实验

> 文件名：`DG-KAN_StageC-next_v3_详细实验计划.md`  
> 对象：DG-KAN / DG-LCA 的 Stage C-next v3 补充实验  
> 定位：实验执行协议，不涉及具体代码实现  
> 公式格式：Typora 友好，统一使用 `$...$` 与 `$$...$$`  
> 核心目标：在 Stage C-next v2 已经证明“更难数据 + 更大 span 可以制造 router-worthy setting，但当前 router 仍未过主线 gate”的基础上，进一步判断 **learned router 是否还有进入 Stage D 的资格**。v3 不再泛泛扩大 sweep，而是集中修复 v2 暴露出的核心失败：**correction direction 学不准**。

---

## 0. 当前判断：v3 为什么需要重新设计

Stage C 和 Stage C-next v2 已经把 learned router 的处境变得很清楚。

最初的 Stage C 发现，residual DG-KAN 的 full-block VJP 很接近 identity。对一个 block：

$$
h_{out}=h+\alpha\operatorname{KAN}(\operatorname{LN}(h))
$$

真实 credit 可以写成：

$$
g_h=g_{out}+\Delta g
$$

其中：

$$
\Delta g
=
D\operatorname{LN}(h)^\top
\left(
\alpha\Phi'(x)^\top g_{out}
\right)
$$

因此 identity router：

$$
\hat g_h=g_{out}
$$

已经是很强 baseline。Stage C-next v1 把 router 目标改成学习非 identity 修正项：

$$
\Delta g=g_h-g_{out}
$$

但发现 correction router 还没有稳定打败 identity。Stage C-next v2 进一步用 hard dataset、span router、norm-calibrated router 放大 correction，找到了很多 router-worthy setting，并且 `direction_norm` router 有明显进步：全局 gain 约 $13\%$，局部 setting 最高 gain 约 $25\%$。但 v2 仍未达到 Stage D 主线 gate：

$$
\text{global gain}<20\%
$$

$$
\text{global correction cosine}<0.7
$$

$$
\text{global one-step gain}<5\%
$$

v2 的 failure table 也表明，主要失败已经从上一轮的 norm miscalibration 转向：

$$
\boxed{
\text{correction direction 学得不够好}
}
$$

因此 v3 的目标不是继续问“有没有 router-worthy setting”，因为 v2 已经找到了。v3 要回答更尖锐的问题：

> **在 router-worthy setting 中，使用 derivative-aware 特征、direction-first loss、更大 router 数据和 micro-training usefulness 评估后，learned correction router 是否能够稳定打败 identity，并且是否值得进入 Stage D？**

如果 v3 仍然失败，则 learned router 应降级为 later work，Stage D 主线应转向：

$$
\boxed{
\text{AnalyticAdj / IdentityAdj + functional update + memory accounting}
}
$$

---

## 1. v3 的核心假设

v3 检验四个假设。

### 1.1 假设一：v2 router 失败的主因是 correction direction 不足

v2 已经通过 `direction_norm` 降低了 norm miscalibration，说明单纯 clip / norm calibration 不够。现在需要提升：

$$
\cos(\widehat{\Delta g},\Delta g)
$$

而不是只控制：

$$
\frac{\|\widehat{\Delta g}\|}{\|\Delta g\|}
$$

因此 v3 的主训练目标要从 MSE-first 改为 direction-first。

### 1.2 假设二：router 需要 KAN derivative-aware 特征

Correction 来自 KAN branch 的局部 Jacobian。对 core KAN：

$$
g_x=\Phi'(x)^\top g_y
$$

span correction 的主要结构也来自每层的 $\Phi'(x)$、LayerNorm、residual branch 和 credit 之间的耦合。因此如果 router 只看 hidden states 和 credit，可能难以学准 correction direction。v3 要显式加入 derivative-aware 特征，例如：

$$
\phi'_{p95},\quad \phi'_{mean},\quad \Phi'(x)g,\quad \text{branch credit norm},\quad \text{per-layer correction sketch}
$$

更准确地说，v3 要比较：

$$
\text{hidden-only router}
\quad
\text{vs}
\quad
\text{derivative-aware router}
$$

### 1.3 假设三：span correction 可以被分解为一阶 branch correction 与高阶交叉项

对 residual block：

$$
F_\ell=I+B_\ell
$$

span-$s$ VJP 是：

$$
J_{k:k+s}^\top
=
(I+A_k^\top)(I+A_{k+1}^\top)\cdots(I+A_{k+s-1}^\top)
$$

其中 $A_\ell$ 是 branch Jacobian。展开后：

$$
J_{k:k+s}^\top g
=
g
+
\sum_{\ell=k}^{k+s-1} A_\ell^\top g
+
\text{higher-order cross terms}
$$

所以 span correction：

$$
\Delta g_{span}
=
J_{k:k+s}^\top g-g
$$

可以近似分解为：

$$
\Delta g_{span}
\approx
\sum_{\ell=k}^{k+s-1}\Delta g_\ell^{first}
+
\Delta g^{cross}
$$

v3 要测清楚 router 难点到底来自：

1. 每层一阶 correction 学不准；
2. 高阶 cross terms 比例太大；
3. correction direction 本身噪声太大；
4. 样本不足或训练目标不合适。

### 1.4 假设四：如果 router 真有价值，它必须在 micro-training 中带来收益

v2 的 one-step usefulness 只能说明局部 hidden-state loss 是否下降，但仍不足以决定 Stage D。v3 要加入短程 micro-training：

> 固定一个 checkpoint，在相同 functional update 设置下，用 identity credit、analytic span credit、router correction credit 做 $T_{micro}$ 步局部训练，比较 loss AUC、test accuracy、几何指标和计算成本。

如果 router 的 fidelity gain 不能转化成 micro-training gain，则不进入 Stage D 主线。

---

## 2. v3 的总体执行路线

v3 分成七个阶段：

1. **V3-A：固定 router-worthy setting 和数据集**  
   不再大范围搜索，锁定 v2 已经证明有价值的 Fashion / KMNIST span2 setting。

2. **V3-B：Span correction decomposition audit**  
   分解 $\Delta g_{span}$ 为一阶 correction 与 cross-term，判断任务难度来源。

3. **V3-C：构建 derivative-aware router dataset**  
   增大 router 样本量，加入 derivative features、layerwise correction sketches 和 normalized targets。

4. **V3-D：Direction-first correction router 训练**  
   比较 hidden-only、derivative-aware、decomposition-aware、norm-calibrated router。

5. **V3-E：Cross-setting generalization**  
   测 router 是否只在单一 setting 过拟合，还是能跨 seed、span、dataset、regime 泛化。

6. **V3-F：One-step 与 micro-training usefulness**  
   检查 router 是否带来实际 loss 下降和短程训练收益。

7. **V3-G：Stage D 决策**  
   根据 fidelity、usefulness、cost、稳定性决定 router 是否进入 Stage D。

---

## 3. 实验对象与固定 setting

v3 不再把 MNIST small 作为主 router 测试，因为 v2 已经说明 MNIST 的 identity baseline 太强。v3 第一轮聚焦四个 setting：

| setting id | dataset | depth | hidden | basis | regime | span | 理由 |
|---|---|---:|---:|---:|---|---:|---|
| `F8_A_s2` | Fashion-MNIST | 8 | 64 | 16 | AdamW | 2 | v2 中局部 gain 最高之一 |
| `F8_P_s2` | Fashion-MNIST | 8 | 64 | 16 | Pareto | 2 | geometry 稳，且 correction 不太小 |
| `K8_P_s2` | KMNIST | 8 | 64 | 16 | Pareto | 2 | v2 one-step 局部 gain 较高 |
| `K8_A_s2` | KMNIST | 8 | 64 | 16 | active | 2 | correction 更大，用作 stress |

其中：

- `F8_A_s2` 是目前 learned router 最接近成功的 setting；
- `F8_P_s2` 是最符合 DG-KAN 主线的 geometry-preserving setting；
- `K8_P_s2` 用来验证 hard dataset Pareto；
- `K8_A_s2` 用来验证强 correction 但较高几何风险的情况。

如果资源允许，第二轮加入：

| setting id | dataset | depth | hidden | basis | regime | span |
|---|---|---:|---:|---:|---|---:|
| `F4_A_s1` | Fashion-MNIST | 4 | 64 | 16 | AdamW | 1 |
| `F8_P_s4` | Fashion-MNIST | 8 | 64 | 16 | Pareto | 4 |
| `K4_P_s1` | KMNIST | 4 | 64 | 16 | Pareto | 1 |

但第一轮只以上述四个 span2 setting 为主，避免实验发散。

---

## 4. V3-A：Checkpoint 复核与固定样本协议

### 4.1 目标

V3-A 的目标不是重新搜索 Pareto，而是复核 v2 选中的 checkpoint 是否仍满足 router-worthy 条件，并为后续 router 训练提供固定数据 split。

router-worthy 条件为：

$$
\operatorname{relerr}_{identity}>0.25
$$

$$
\text{correction ratio}>0.30
$$

$$
\text{amp p95}<1.5
$$

$$
\text{test gap}<2\%
$$

对于 active stress setting，允许：

$$
\text{amp p95}<1.8
$$

但必须单独标记为 stress，不作为主线成功证据。

### 4.2 Router 数据集构造

每个 setting 构造 router dataset：

$$
\mathcal D_{router}
=
\{h_k,h_{k+s},g_{k+s},g_k^{teacher},\Delta g\}
$$

其中：

$$
\Delta g=g_k^{teacher}-g_{k+s}
$$

teacher 使用 analytic sequential span VJP：

$$
g_k^{teacher}=J_{k:k+s}^\top g_{k+s}
$$

样本来源包括：

1. task BP credit；
2. random normalized credit；
3. mixed credit；
4. teacher-forced intermediate credit。

第一轮推荐样本量：

| split | samples |
|---|---:|
| train router | $8192$ |
| validation router | $2048$ |
| test same setting | $2048$ |
| test cross-seed | $2048$ |
| test random credit | $2048$ |
| test mixed credit | $2048$ |

如果计算成本太高，最低不要低于：

$$
N_{train}=4096
$$

因为 v2 的方向学习不足可能部分来自样本量太小。

### 4.3 必须记录的 W&B config

| config key | 含义 |
|---|---|
| `stage` | `stage_c_next_v3` |
| `setting_id` | `F8_A_s2`, `F8_P_s2`, `K8_P_s2`, `K8_A_s2` |
| `dataset` | `fashion_mnist`, `kmnist` |
| `depth` | 8 |
| `hidden_dim` | 64 |
| `basis_count` | 16 |
| `span` | 2 |
| `regime` | `adamw`, `pareto`, `active` |
| `router_train_samples` | 4096 / 8192 |
| `router_val_samples` | 2048 |
| `credit_types_train` | task / random / mixed / pooled |
| `credit_types_eval` | task / random / mixed |
| `teacher_type` | analytic sequential span VJP |
| `target_type` | full correction $\Delta g$ |

### 4.4 V3-A metrics

| metric key | 含义 |
|---|---|
| `v3a/checkpoint/test_acc` | checkpoint test accuracy |
| `v3a/checkpoint/test_gap_vs_adamw` | 与 AdamW 差距 |
| `v3a/checkpoint/branch_over_adamw` | branch / AdamW |
| `v3a/checkpoint/phi_prime_over_adamw` | $\phi'_{p95}$ / AdamW |
| `v3a/checkpoint/jcond_over_adamw` | Jacobian condition / AdamW |
| `v3a/span/identity_relerr` | span identity relerr |
| `v3a/span/correction_ratio` | $\|\Delta g\|/\|g_k^{teacher}\|$ |
| `v3a/span/amp_p95` | amplification p95 |
| `v3a/span/noise_gain` | noise gain |
| `v3a/router_data/train_samples` | router 训练样本量 |
| `v3a/router_data/credit_norm_mean` | credit norm 均值 |
| `v3a/router_data/correction_norm_mean` | correction norm 均值 |
| `v3a/router_data/correction_norm_p95` | correction norm 95 分位 |

### 4.5 V3-A 可视化

1. **Checkpoint regime table**：setting × test gap × branch / AdamW × J condition。
2. **Identity relerr by setting**：bar chart。
3. **Correction ratio by setting**：bar chart。
4. **Credit norm distribution**：task / random / mixed credit 的 norm histogram。
5. **Correction norm distribution**：$\|\Delta g\|$ histogram。
6. **Router sample coverage plot**：样本数量 vs correction norm coverage。

### 4.6 V3-A 通过条件

若某个 setting 不满足：

$$
\operatorname{relerr}_{identity}>0.25
$$

或：

$$
\text{correction ratio}>0.30
$$

则该 setting 不进入 V3-D 主实验。

若某个 Pareto setting 的 test gap 超过 $2\%$，则它只能作为 stress，不能作为主线证据。

---

## 5. V3-B：Span correction decomposition audit

### 5.1 目标

V3-B 要回答：

> span correction 为什么难学？它主要是一阶 branch correction，还是高阶 cross terms？

对 span $s=2$：

$$
J_{k:k+2}^\top
=
(I+A_k^\top)(I+A_{k+1}^\top)
$$

展开：

$$
J_{k:k+2}^\top g
=
g
+
A_{k+1}^\top g
+
A_k^\top g
+
A_k^\top A_{k+1}^\top g
$$

所以 correction 为：

$$
\Delta g
=
A_{k+1}^\top g
+
A_k^\top g
+
A_k^\top A_{k+1}^\top g
$$

定义一阶 correction：

$$
\Delta g^{first}
=
A_{k+1}^\top g
+
A_k^\top g
$$

高阶项：

$$
\Delta g^{cross}
=
\Delta g-\Delta g^{first}
$$

如果：

$$
\|\Delta g^{cross}\|\ll \|\Delta g\|
$$

说明 router 主要需要学一阶 branch correction。  
如果 cross term 很大，则简单 router 会很难。

### 5.2 需要计算的量

对每个 setting、每个 span、每个 layer window：

| quantity | 定义 |
|---|---|
| `delta_total` | $\Delta g=g_k^{teacher}-g_{k+s}$ |
| `delta_first` | 一阶 branch correction 近似 |
| `delta_cross` | $\Delta g-\Delta g^{first}$ |
| `delta_per_layer_l` | 每层一阶 correction |
| `delta_direction_residual` | $\Delta g$ 与 $\Delta g^{first}$ 的方向差 |
| `cross_ratio` | $\|\Delta g^{cross}\|/\|\Delta g\|$ |
| `first_cos` | $\cos(\Delta g^{first},\Delta g)$ |

### 5.3 W&B metrics

| metric key | 含义 |
|---|---|
| `v3b/decomp/first_cos` | 一阶 correction 与总 correction cosine |
| `v3b/decomp/first_relerr` | $\|\Delta g^{first}-\Delta g\|/\|\Delta g\|$ |
| `v3b/decomp/cross_ratio` | $\|\Delta g^{cross}\|/\|\Delta g\|$ |
| `v3b/decomp/layer0_contrib_ratio` | 第 1 个 block correction 占比 |
| `v3b/decomp/layer1_contrib_ratio` | 第 2 个 block correction 占比 |
| `v3b/decomp/per_layer_alignment` | per-layer correction 之间 cosine |
| `v3b/decomp/correction_snr` | correction norm / correction residual noise |
| `v3b/decomp/credit_type` | task / random / mixed |

### 5.4 V3-B 可视化

1. **First-order approximation scatter**  
   x-axis: $\|\Delta g\|$，y-axis: $\|\Delta g^{first}\|$。

2. **First-order cosine bar**  
   setting × `first_cos`。

3. **Cross-term ratio heatmap**  
   setting × layer-window，颜色为 `cross_ratio`。

4. **Per-layer contribution stacked bar**  
   每个 span window 中，不同 layer correction 的占比。

5. **Correction PCA plot**  
   UMAP / PCA 可选，比较 task / random / mixed credit 下 correction 分布。

### 5.5 V3-B 结论如何影响后续 router

若：

$$
\text{first\_cos}>0.8
$$

且：

$$
\text{cross\_ratio}<0.3
$$

则后续 router 可以使用 decomposition-aware 一阶 features。

若：

$$
\text{cross\_ratio}>0.5
$$

则说明 span correction 里高阶 interaction 很重要，简单 low-rank router 很难成功，应限制 span 或使用更强结构。

---

## 6. V3-C：Derivative-aware router dataset 与 feature 设计

### 6.1 目标

V3-C 构造多种 router 输入特征，判断 correction direction 学不准是否来自缺少 derivative information。

### 6.2 Feature families

#### Family 0：Hidden-only features

作为 baseline，只使用：

$$
h_k,\quad h_{k+s},\quad g_{k+s}
$$

#### Family 1：Branch scalar features

加入每层 scalar diagnostics：

$$
r_{branch},\quad \phi'_{p95},\quad \phi'_{mean},\quad Jcond,\quad amp
$$

这些特征用于 norm head 或 gating。

#### Family 2：Derivative statistics features

对每层 KAN derivative matrix $\Phi'(x)$ 提取：

| feature | 含义 |
|---|---|
| row norm | 每个 output dim 的 derivative 强度 |
| column norm | 每个 input dim 的 derivative 强度 |
| diagonal proxy | $\partial y_i/\partial x_i$ 类似的对角近似 |
| top singular sketch | $\Phi'(x)$ 的低秩 sketch |
| pooled $\phi'$ moments | mean / std / p95 / max |

#### Family 3：Analytic first-order correction sketch

使用 V3-B 的一阶 correction 近似：

$$
\Delta g^{first}
$$

或其低维 sketch 作为 router feature。

这不是 oracle target，而是用于判断：

> 如果给 router 结构化 derivative hint，direction 是否显著改善？

必须清楚区分：

- 直接使用 exact $\Delta g$ 是 oracle，不允许；
- 使用一阶 / layerwise analytic sketch 是 structured hint，作为 derivative-aware ablation。

#### Family 4：Layerwise correction features

对 span $s=2$，分别提供：

$$
\Delta g_0^{first},\quad \Delta g_1^{first}
$$

router 再学习如何组合。

### 6.3 W&B config

| config key | 含义 |
|---|---|
| `feature_family` | `hidden_only`, `scalar`, `derivative_stats`, `first_order_sketch`, `layerwise_first_order` |
| `feature_dim` | feature 维度 |
| `uses_phi_prime` | 是否显式使用 $\phi'$ |
| `uses_first_order_sketch` | 是否使用一阶 correction sketch |
| `uses_layerwise_features` | 是否使用 layerwise correction |
| `sketch_rank` | derivative sketch rank |
| `feature_normalization` | none / layernorm / zscore |
| `credit_normalization` | norm / rms / none |

### 6.4 Feature usefulness metrics

| metric key | 含义 |
|---|---|
| `v3c/feature/mutual_cos_with_delta` | feature-induced direction 与 $\Delta g$ 的 cosine |
| `v3c/feature/linear_probe_cos` | 线性 probe 从 feature 预测 correction 的 cosine |
| `v3c/feature/linear_probe_relerr` | 线性 probe relerr |
| `v3c/feature/feature_norm_stability` | feature norm 稳定性 |
| `v3c/feature/cross_seed_probe_cos` | cross-seed probe cosine |

### 6.5 可视化

1. **Feature family comparison bar**：linear probe cosine / relerr。
2. **Feature norm distribution**：防止 feature 爆炸。
3. **Feature vs correction norm scatter**。
4. **Feature PCA by setting**：查看不同 setting 是否分布漂移。
5. **Cross-seed feature probe matrix**。

### 6.6 通过条件

若 derivative-aware feature 的 linear probe 已经明显优于 hidden-only：

$$
\text{cos gain}>0.1
$$

或：

$$
\text{relerr reduction}>20\%
$$

则进入 V3-D 主 router 训练。若 feature probe 都无提升，则复杂 router 也很可能失败，应停止 router 主线。

---

## 7. V3-D：Direction-first correction router 训练

### 7.1 目标

V3-D 是 v3 的核心实验。它训练 correction router：

$$
\widehat{\Delta g}=R^\phi(\text{features},g_{k+s})
$$

最终输出：

$$
\hat g_k=g_{k+s}+\widehat{\Delta g}
$$

与 v2 不同，v3 不再以 full-VJP MSE 为主，而是先学 correction direction，再学 norm。

### 7.2 Router families

#### Router 0：zero correction

$$
\widehat{\Delta g}=0
$$

等价于 identity。

#### Router 1：scalar correction

$$
\widehat{\Delta g}=s\cdot g_{k+s}
$$

作为最弱 correction baseline。

#### Router 2：direction-norm router

分解为方向和范数：

$$
\widehat{\Delta g}
=
\widehat r\cdot
\frac{\widehat u}{\|\widehat u\|+\epsilon}
$$

其中：

$$
\widehat r\approx\|\Delta g\|
$$

$$
\widehat u\approx \frac{\Delta g}{\|\Delta g\|}
$$

#### Router 3：derivative-aware direction-norm router

同 Router 2，但输入加入 derivative features。

#### Router 4：decomposition-aware router

预测 layerwise weights：

$$
\widehat{\Delta g}
=
w_0 \Delta g_0^{first}
+
w_1 \Delta g_1^{first}
+
R_{residual}
$$

其中 $R_{residual}$ 学习一阶近似之外的残差。

#### Router 5：low-rank correction router

$$
\widehat{\Delta g}
=
U(f)V(f)^\top g
$$

必须结合 norm clipping 与 direction loss，否则容易重现 v2 的 norm miscalibration。

#### Router 6：analytic oracle

$$
\widehat{\Delta g}=\Delta g
$$

只作为上限。

### 7.3 Loss 设计

Direction loss：

$$
\mathcal L_{dir}
=
1-
\frac{
\langle \widehat{\Delta g},\Delta g\rangle
}{
\|\widehat{\Delta g}\|\|\Delta g\|+\epsilon
}
$$

Norm loss：

$$
\mathcal L_{norm}
=
\left(
\log
\frac{
\|\widehat{\Delta g}\|+\epsilon
}{
\|\Delta g\|+\epsilon
}
\right)^2
$$

Full VJP loss：

$$
\mathcal L_{full}
=
\frac{
\|g_{k+s}+\widehat{\Delta g}-g_k^{teacher}\|^2
}{
\|g_k^{teacher}\|^2+\epsilon
}
$$

One-step proxy loss，若可计算：

$$
\mathcal L_{descent}
=
-\Delta L_{actual}^{proxy}
$$

总 loss：

$$
\mathcal L
=
\lambda_{dir}\mathcal L_{dir}
+
\lambda_{norm}\mathcal L_{norm}
+
\lambda_{full}\mathcal L_{full}
+
\lambda_{descent}\mathcal L_{descent}
$$

第一轮建议：

$$
\lambda_{dir}=1.0
$$

$$
\lambda_{norm}=0.3
$$

$$
\lambda_{full}=0.3
$$

$$
\lambda_{descent}=0
$$

第二轮再加入 descent-aware loss。

### 7.4 Router training protocol

每个 setting 至少训练以下 router：

| router | feature family | samples | seeds |
|---|---|---:|---:|
| zero | none | - | - |
| scalar | scalar | 8192 | 5 |
| direction_norm_hidden | hidden_only | 8192 | 5 |
| direction_norm_derivative | derivative_stats | 8192 | 5 |
| decomposition_aware | first_order_sketch | 8192 | 5 |
| lowrank_direction | derivative_stats | 8192 | 5 |
| analytic_oracle | oracle | - | - |

训练时必须固定 router train/val/test split，避免数据泄漏。

### 7.5 W&B metrics

Core correction metrics：

| metric key | 含义 |
|---|---|
| `v3d/correction/cos` | $\cos(\widehat{\Delta g},\Delta g)$ |
| `v3d/correction/relerr` | $\|\widehat{\Delta g}-\Delta g\|/\|\Delta g\|$ |
| `v3d/correction/norm_ratio` | $\|\widehat{\Delta g}\|/\|\Delta g\|$ |
| `v3d/correction/cos_p05` | correction cosine 低分位 |
| `v3d/correction/relerr_p95` | correction relerr 高分位 |
| `v3d/correction/norm_ratio_p95` | norm ratio 高分位 |

Full VJP metrics：

| metric key | 含义 |
|---|---|
| `v3d/full/relerr_identity` | identity relerr |
| `v3d/full/relerr_router` | router full relerr |
| `v3d/full/gain_over_identity` | 相对 identity 的 relerr gain |
| `v3d/full/cos_router` | full VJP cosine |
| `v3d/full/beat_rate` | sample-level 打败 identity 的比例 |

定义：

$$
\text{gain}
=
\frac{
\operatorname{relerr}_{identity}
-
\operatorname{relerr}_{router}
}{
\operatorname{relerr}_{identity}+\epsilon
}
$$

Generalization metrics：

| metric key | 含义 |
|---|---|
| `v3d/gen/same_setting_cos` | 同 setting test correction cos |
| `v3d/gen/cross_seed_cos` | cross-seed correction cos |
| `v3d/gen/cross_credit_type_cos` | task/random/mixed credit 泛化 |
| `v3d/gen/cross_dataset_cos` | cross-dataset，可选 |
| `v3d/gen/cross_regime_cos` | cross-regime，可选 |

Cost metrics：

| metric key | 含义 |
|---|---|
| `v3d/cost/router_time_ms` | router 推理时间 |
| `v3d/cost/analytic_span_vjp_time_ms` | analytic span VJP 时间 |
| `v3d/cost/router_memory_mb` | router 显存 |
| `v3d/cost/router_params` | router 参数量 |

### 7.6 V3-D 可视化

1. **Correction cosine by router**  
   x-axis: router type；y-axis: correction cos；color: setting。

2. **Gain over identity by router**  
   bar chart，显示 mean ± std。

3. **Router fidelity vs cost Pareto**  
   x-axis: router time；y-axis: gain over identity；point size: memory。

4. **Correction norm calibration scatter**  
   x-axis: $\|\Delta g\|$；y-axis: $\|\widehat{\Delta g}\|$。

5. **Full relerr identity vs router scatter**  
   每个点一个 sample，直观看 router 是否打败 identity。

6. **Feature family comparison**  
   hidden-only vs derivative-aware vs decomposition-aware。

7. **Cross-seed generalization matrix**。

### 7.7 成功标准

v3 的 router 主成功标准分三级。

#### Weak pass

在至少两个 setting 上：

$$
\text{gain}>0.15
$$

$$
\text{beat rate}>0.65
$$

$$
\text{correction cos}>0.6
$$

#### Medium pass

在至少两个主 setting 上：

$$
\text{gain}>0.20
$$

$$
\text{beat rate}>0.75
$$

$$
\text{correction cos}>0.70
$$

$$
0.7<\text{norm ratio}<1.3
$$

#### Strong pass

在 Fashion 和 KMNIST 的 span2 setting 上：

$$
\text{gain}>0.25
$$

$$
\text{beat rate}>0.8
$$

$$
\text{correction cos}>0.8
$$

并且 router 的 one-step gain 超过 $5\%$，micro-training gain 超过 $3\%$。

若无法达到 Medium pass，router 不进入 Stage D 主线。

---

## 8. V3-E：Cross-setting generalization

### 8.1 目标

Router 如果只在一个 setting 上有效，不能说明机制成立。V3-E 检查 router 是否能跨 seed、credit type、regime 和 dataset 泛化。

### 8.2 Matrix 设计

训练 / 测试 matrix：

| train \ eval | F8_A_s2 | F8_P_s2 | K8_P_s2 | K8_A_s2 |
|---|---:|---:|---:|---:|
| F8_A_s2 | cos/gain | cos/gain | cos/gain | cos/gain |
| F8_P_s2 | cos/gain | cos/gain | cos/gain | cos/gain |
| K8_P_s2 | cos/gain | cos/gain | cos/gain | cos/gain |
| pooled | cos/gain | cos/gain | cos/gain | cos/gain |

第一轮可以只做 same-dataset cross-regime 与 pooled，不必立刻做 full cross-dataset。

### 8.3 W&B metrics

| metric key | 含义 |
|---|---|
| `v3e/cross/setting_train` | train setting |
| `v3e/cross/setting_eval` | eval setting |
| `v3e/cross/correction_cos` | correction cos |
| `v3e/cross/full_gain` | full gain |
| `v3e/cross/beat_rate` | beat rate |
| `v3e/cross/norm_ratio` | norm ratio |
| `v3e/cross/degradation_vs_same` | 相对 same-setting 降幅 |

### 8.4 可视化

1. **Cross-setting heatmap：correction cos**。
2. **Cross-setting heatmap：gain over identity**。
3. **Pooled vs specialized bar chart**。
4. **Generalization degradation plot**。

### 8.5 通过条件

Pooled router 若能在所有主 setting 上满足：

$$
\text{gain}>0.15
$$

且：

$$
\text{correction cos}>0.6
$$

则说明 router 不是纯粹单 setting 过拟合。

若 specialized router 成功但 pooled 失败，则 Stage D 若使用 router，需要 online / per-regime calibration。

---

## 9. V3-F：One-step 与 micro-training usefulness

### 9.1 目标

V3-F 要回答：

> Router fidelity gain 是否能转化成真实训练收益？

v2 表明 fidelity gain 和 one-step gain 不完全一致。因此 v3 加入 micro-training。

### 9.2 One-step hidden update

与 v2 保持一致，但只在 v3 router 通过的 setting 上测。

比较 credit source：

| source | 含义 |
|---|---|
| identity | $\hat g=g_{out}$ |
| analytic oracle | exact span VJP |
| direction_norm_v2 | v2 最强 router |
| derivative-aware router | v3 router |
| decomposition-aware router | v3 router |
| random | 负对照 |

指标：

| metric key | 含义 |
|---|---|
| `v3f/one_step/actual_delta` | actual loss delta |
| `v3f/one_step/gain_vs_identity` | 相对 identity gain |
| `v3f/one_step/sign_agreement` | 是否下降 |
| `v3f/one_step/update_norm` | hidden update norm |
| `v3f/one_step/correction_cos` | correction cos |

通过条件：

$$
\text{gain vs identity}>5\%
$$

在至少两个 setting 上成立。

### 9.3 Micro-training protocol

固定 checkpoint，运行短程训练：

$$
T_{micro}\in\{50,100,200\}
$$

只更新 KAN coefficients，非 KAN 参数可选择冻结或继续 AdamW。第一轮建议：

| mode | KAN coeff update | non-KAN params |
|---|---|---|
| local-only | functional update | frozen |
| hybrid | functional update | AdamW |

Credit sources：

| source | 含义 |
|---|---|
| full analytic | exact span analytic VJP |
| identity | identity credit |
| v2 direction_norm | best v2 router |
| v3 derivative-aware | best v3 router |
| v3 decomposition-aware | best v3 router |

记录 micro-training 的 loss AUC、final val loss、test acc、geometry metrics。

### 9.4 Micro-training metrics

| metric key | 含义 |
|---|---|
| `v3f/micro/train_loss_auc` | micro-train loss AUC |
| `v3f/micro/val_loss_auc` | micro-val loss AUC |
| `v3f/micro/final_val_loss` | final val loss |
| `v3f/micro/final_test_acc` | final test acc |
| `v3f/micro/gap_vs_analytic` | 相对 analytic credit gap |
| `v3f/micro/gain_vs_identity` | 相对 identity gain |
| `v3f/micro/branch_ratio_after` | 训练后 branch ratio |
| `v3f/micro/phi_prime_p95_after` | 训练后 $\phi'_{p95}$ |
| `v3f/micro/jcond_after` | 训练后 Jacobian condition |
| `v3f/micro/diverged` | 是否发散 |

### 9.5 Micro-training 可视化

1. **Micro loss curves**：identity vs analytic vs v3 router。
2. **Final val loss bar chart**。
3. **Gain vs identity bar chart**。
4. **Geometry after micro-training**：$\phi'_{p95}$、J condition。
5. **Router fidelity vs micro-training gain scatter**。

### 9.6 Micro-training 成功标准

Router 进入 Stage D 的最低要求：

$$
\text{micro gain vs identity}>3\%
$$

且：

$$
\text{gap vs analytic}<5\%
$$

如果 router fidelity 有提升但 micro-training 没提升，则不进入 Stage D。

---

## 10. V3-G：Cost 与 Stage D readiness

### 10.1 目标

即使 router 更准，如果它比 analytic span VJP 更贵，也不应该进入 Stage D 主线。V3-G 比较成本。

### 10.2 Cost metrics

| metric key | 含义 |
|---|---|
| `v3g/cost/router_forward_ms` | router forward time |
| `v3g/cost/router_train_ms_per_step` | router train step time |
| `v3g/cost/analytic_span_vjp_ms` | analytic span VJP time |
| `v3g/cost/autograd_span_vjp_ms` | autograd span VJP time |
| `v3g/cost/router_memory_mb` | router memory |
| `v3g/cost/analytic_memory_mb` | analytic VJP temp memory |
| `v3g/cost/router_params` | router 参数量 |
| `v3g/cost/router_over_analytic_time` | router / analytic time |

### 10.3 Cost 可视化

1. **Cost vs gain Pareto**。
2. **Router memory vs correction cos**。
3. **Analytic VJP vs router time bar chart**。
4. **Router params vs micro-training gain**。

### 10.4 Stage D readiness 条件

Router 只有同时满足下面条件，才进入 Stage D 主线：

$$
\text{gain}>0.20
$$

$$
\text{correction cos}>0.70
$$

$$
\text{one-step gain}>5\%
$$

$$
\text{micro gain}>3\%
$$

并且：

$$
T_{router}<T_{analytic\ span\ VJP}
$$

或：

$$
M_{router}<M_{analytic\ span\ VJP}
$$

若不满足，则 Stage D 继续以 AnalyticAdj / IdentityAdj 为主，router 只作为 ablation。

---

## 11. W&B Dashboard 设计

Stage C-next v3 建议建立 8 个 dashboard 页面。

### Page 1：V3 Overview

展示所有 setting 的：

- test gap；
- branch / AdamW；
- identity relerr；
- correction ratio；
- amp p95；
- router best gain；
- one-step gain；
- micro-training gain。

### Page 2：Checkpoint and Router-worthy Map

图表：

1. setting table；
2. branch / AdamW bar；
3. identity relerr bar；
4. correction ratio bar；
5. amp p95 bar；
6. router-worthy pass/fail heatmap。

### Page 3：Correction Decomposition

图表：

1. first-order cosine bar；
2. cross-term ratio heatmap；
3. per-layer contribution stacked bar；
4. first-order approximation scatter；
5. cross-ratio vs router gain scatter。

### Page 4：Feature Probe

图表：

1. feature family vs linear probe cosine；
2. feature family vs relerr；
3. derivative feature norm distribution；
4. feature PCA by setting；
5. cross-seed feature generalization matrix。

### Page 5：Router Fidelity

图表：

1. correction cos by router；
2. full gain over identity by router；
3. norm calibration scatter；
4. beat rate bar；
5. full relerr identity vs router scatter；
6. router fidelity vs cost Pareto。

### Page 6：Cross-setting Generalization

图表：

1. train/eval heatmap for correction cosine；
2. train/eval heatmap for gain；
3. pooled vs specialized comparison；
4. cross-setting degradation bar。

### Page 7：Usefulness

图表：

1. one-step gain bar；
2. micro-training loss curves；
3. final val loss bar；
4. router fidelity vs micro gain scatter；
5. geometry after micro-training。

### Page 8：Failure Analysis

展示 failure table，按类型聚合：

- router bad direction；
- norm miscalibration；
- no gain over identity；
- cross-setting fail；
- one-step no gain；
- micro no gain；
- router cost too high；
- pareto invalid。

---

## 12. Failure table

每个失败都要落入 failure table。

| failure type | 触发条件 | 解释 |
|---|---|---|
| `setting_not_router_worthy` | identity relerr <= 0.25 或 correction ratio <= 0.30 | setting 太容易 |
| `amp_too_high` | amp p95 > 1.5 主线 / >1.8 stress | credit geometry 不稳 |
| `feature_no_signal` | derivative feature probe 没有提升 | derivative features 无效 |
| `first_order_bad` | first-order cos < 0.6 | correction 主要来自高阶项 |
| `router_bad_direction` | correction cos < 0.6 | direction 学不准 |
| `router_norm_bad` | norm ratio < 0.7 或 > 1.3 | norm 校准失败 |
| `router_no_gain` | gain <= 0 | 没打败 identity |
| `router_gate_fail` | gain < 0.2 或 cos < 0.7 | 未过主 gate |
| `cross_setting_fail` | cross-setting cos < 0.5 | 泛化失败 |
| `one_step_no_gain` | one-step gain <= 0 | 实际下降无收益 |
| `micro_no_gain` | micro gain <= 0 | 短程训练无收益 |
| `cost_too_high` | router 比 analytic VJP 更贵且无收益 | 不适合部署 |

字段：

| field | 含义 |
|---|---|
| `failure/stage` | V3-A 到 V3-G |
| `failure/setting_id` | setting |
| `failure/dataset` | dataset |
| `failure/router_type` | router |
| `failure/span` | span |
| `failure/seed` | seed |
| `failure/metric_name` | 触发 metric |
| `failure/metric_value` | metric value |
| `failure/threshold` | threshold |
| `failure/diagnosis` | 诊断 |
| `failure/recommended_action` | 建议 |

---

## 13. 第一轮最小执行包

如果资源有限，第一轮只做下面四个 setting：

```text
F8_A_s2
F8_P_s2
K8_P_s2
K8_A_s2
```

并只跑五个实验：

1. V3-A checkpoint 复核；
2. V3-B correction decomposition；
3. V3-C feature probe；
4. V3-D router training；
5. V3-F one-step + micro-training。

Router 类型第一轮只跑：

```text
zero
scalar
direction_norm_hidden
direction_norm_derivative
decomposition_aware
analytic_oracle
```

不再跑大量 ridge / lowrank，除非 direction-aware router 通过 weak gate。

第一轮最低决策标准：

- 如果 derivative-aware 或 decomposition-aware router 在至少两个 setting 上满足：
  $$
  \text{gain}>0.20,\quad \text{correction cos}>0.70
  $$
  则继续 V3-E / V3-G。
- 如果没有任何 router 达到：
  $$
  \text{gain}>0.15,\quad \text{correction cos}>0.60
  $$
  则停止 router 主线，进入 Stage D analytic / identity 主线。

---

## 14. Stage D 决策规则

v3 完成后，按下面规则决定 Stage D。

### 情况 A：router 通过 medium gate

如果 v3 router 满足：

$$
\text{gain}>0.20
$$

$$
\text{correction cos}>0.70
$$

$$
\text{one-step gain}>5\%
$$

$$
\text{micro gain}>3\%
$$

则 Stage D 加入：

$$
\text{RouterAdj-DGKAN}
$$

作为主方法之一。

### 情况 B：router fidelity 通过但 usefulness 不通过

如果 correction cos / gain 通过，但 one-step 或 micro-training 不通过，说明 fidelity 指标不足以保证训练价值。Stage D 不使用 router，但保留 router 分析作为附录。

### 情况 C：router 仍未通过 weak gate

则 learned router 停止作为主线。Stage D 使用：

$$
\text{FullBP-DGKAN}
$$

$$
\text{AnalyticAdj-DGKAN}
$$

$$
\text{IdentityAdj-DGKAN}
$$

并专注 memory accounting 与 functional update。

### 情况 D：only active / unstable setting 成功

如果只有 active / high amplification setting 成功，而 Pareto setting 失败，则不能 claim router 是 DG-KAN 主线的一部分。因为 router 价值来自几何变差的场景，而非稳定 DG-KAN 主线。

---

## 15. 预期结论模板

如果 v3 成功，可以写：

> Stage C-next v3 shows that learned routers become useful only when the routing target is formulated as a span-level non-identity correction and the router is supplied with derivative-aware features. In router-worthy Fashion-MNIST and KMNIST settings, derivative-aware correction routers reduce identity error by over $20\%$ and improve one-step / micro-training descent, suggesting that learned local credit correction can become useful beyond identity routing.

如果 v3 失败，应写：

> Stage C-next v3 further stress-tests the learned router hypothesis using harder datasets, span-level VJP targets, derivative-aware features, direction-first losses, and micro-training usefulness tests. Although these settings amplify non-identity correction and improve over previous norm-calibrated routers, no learned router consistently beats the identity baseline with sufficient fidelity and training usefulness. This suggests that for residual DG-KAN, local credit transport is effectively handled by identity / analytic adjoints, and learned routers should be treated as optional rather than central.

---

## 16. 当前推荐下一步

从当前证据出发，v3 是 learned router 的最后一次主线资格测试。它不应再继续大范围盲扫，而应集中在：

$$
\boxed{
\text{span2}
+
\text{hard dataset}
+
\text{derivative-aware features}
+
\text{direction-first loss}
+
\text{micro-training usefulness}
}
$$

如果这组实验仍不过关，就应果断推进 Stage D 的 analytic / identity adjoint 主线。
