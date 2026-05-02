# DG-KAN Optimizer v3.4 实验计划：Unified Full-Start 与真正 All-Functional Update

> 版本：v3.4 draft  
> 日期：2026-05-02  
> 目标：在 v3.3 已经确认 `U-FULL-f085-alphaFixed1` 是统一 Pareto profile 的基础上，真正测试“非 KAN 参数也使用 functional / geometry-aware update”的可行性。  
> 公式格式：Typora 友好，全文只使用 `$...$` 与 `$$...$$`。

---

## 0. 本轮实验的一句话目标

本轮实验要回答两个问题。

第一个问题是：

$$
\boxed{
\text{full_sobolev_gram from start + alphaFixed1 + branch_final_scale=0.85}
}
$$

能不能作为 DG-KAN 在 Fashion-MNIST 与 KMNIST 上的统一 profile 继续保留，并且是否有必要只做一个很小的 expression rescue。

第二个问题更重要：

$$
\boxed{
\text{能不能把非 KAN 参数也从 AdamW 中拿出来，改成真正的 functional / geometry-aware update？}
}
$$

这里的重点是：上一轮 v3.3 的 `UO-PGAdam / UO-NormPGAdam / UO-PostAdam` 并没有真正测试这个问题。它们测试的是把 KAN functional direction 接入 AdamW-style moment dynamics，而不是给非 KAN 参数设计自己的 function-space 或 activation-space metric。因此，上一轮可以排除的是 AdamW-style unified optimizer，不能排除真正的 all-functional optimizer。

本轮把真正的 all-functional optimizer 命名为 **AFU：All-Functional Update**。

---

## 1. 当前结果如何决定下一轮设计

v3.3 已经给出三个非常清楚的信号。

第一，统一 full-start profile 已经有强证据。`U-FULL-f085-alphaFixed1` 在 Fashion-MNIST 与 KMNIST 上同时改善 accuracy mean、validation-loss AUC、geometry 与 ECE。Fashion 上它略弱于 dataset-specific 的 `F-V3-HARD-base30`，但依然明显强于 AdamW；KMNIST 上它比 v3.2 的 full-base 更好，尤其 no-KAN expression 从约 $0.663$ 提升到约 $0.696$。所以它已经是合理的统一 Pareto profile。

第二，KMNIST 的 expression rescue 已经基本成功，但严格 gate 还差一点点。当前差距不是大问题，而是：

$$
0.7000 - 0.6959 \approx 0.0041.
$$

因此，下一轮不应该再大范围搜索 branch schedule。最多只允许小幅检查 `branch_final_scale=0.875` 或 `0.90` 是否能把 expression ratio 推过 $0.70$，而不能牺牲 AUC 与 geometry。

第三，上一轮 unified optimizer 的失败不能解释为“all-functional 不可行”。`PGAdam / NormPGAdam / PostAdam` 的失败主要是 Adam moment 与 functional geometry 不对齐，出现 branch 爆炸、AUC 变差、$
\phi'$ 与 Jacobian condition 恶化。真正应该测试的是：对每类非 KAN 参数定义相应的 metric，而不是把所有东西都塞回 Adam moment。

因此，v3.4 的设计原则是：

```text
保留 U-FULL-f085 作为统一 profile baseline。
不再测试 AdamW-style UO。
真正实现 AFU：每类参数都有自己的 metric。
按参数组逐层推进，不一次性全替换。
先方向审计，再训练，再扩 seed。
```

---

## 2. 当前 hybrid optimizer 与目标 AFU 的区别

当前最强主线仍是 hybrid：

$$
\boxed{
\text{KAN coefficients: Sobolev functional update}
+
\text{non-KAN parameters: AdamW}
}
$$

KAN edge function 写成：

$$
\phi_{ji}(t)=\sum_m a_{jim}B_m(t).
$$

KAN coefficient 的 functional update 是：

$$
a_{t+1}=a_t-\eta_a(M_{KAN}+\rho I)^{-1}\nabla_aL.
$$

其中：

$$
M_{KAN}
=
\int BB^\top dt
+
\alpha_s\int B'B'^\top dt
+
\beta_s\int B''B''^\top dt.
$$

而真正的 AFU 目标不是继续保留 rest AdamW，而是把非 KAN 参数也改成 metric-aware update：

$$
\boxed{
\theta_{g,t+1}=\theta_{g,t}-\eta_gP_g\nabla_{\theta_g}L
}
$$

这里 $g$ 表示参数组，例如 head、stem、LayerNorm、bias 等。不同参数组必须有不同的 $P_g$。不能把 KAN 的 Sobolev Gram 粗暴套给所有参数，因为 stem/head/LayerNorm 不是 RBF edge function coefficient。

这也是本轮最关键的思想：

$$
\boxed{
\text{All-functional 不是 all-Sobolev，而是 every parameter group gets its natural geometry.}
}
$$

---

## 3. AFU 的参数组与 metric 设计

### 3.1 KAN coefficient：保持 full Sobolev from start

KAN coefficient 继续使用 v3.3 的统一 full-start 设置：

```text
metric = full_sobolev_gram
alpha_mode = fixed1
branch_final_scale = 0.85
```

更新为：

$$
\Delta a=-\eta_a(M_{KAN}+\rho I)^{-1}\nabla_aL.
$$

这部分不再重新发明。它是 AFU 的核心锚点。

---

### 3.2 KAN bias：constant-function diagonal update

KAN bias 可以看成 edge function 之外的常数函数项。它不需要 Sobolev full Gram，但可以使用 output-gradient diagonal metric。

对每个 KAN bias $b_j$，定义：

$$
D_{b,j}=\operatorname{EMA}\left[\mathbb E_B[g_{y,j}^2]\right]+\rho.
$$

更新为：

$$
\Delta b_j=-\eta_b\frac{\nabla_{b_j}L}{\sqrt{D_{b,j}}}.
$$

这不是 Adam，因为不使用一阶 momentum，也不使用 Adam 的 bias correction；它只是用输出梯度的二阶尺度做 diagonal functional normalization。

需要记录：

```text
kan_bias/update_norm
kan_bias/update_over_param
kan_bias/diag_metric_mean
kan_bias/diag_metric_p95
```

---

### 3.3 Head Linear：feature-space covariance update

分类 head 通常是：

$$
z=Wh+b.
$$

如果把 head 看成从 representation $h$ 到 logits $z$ 的线性函数，那么它自然的函数空间 metric 来自 feature covariance：

$$
G_h=\mathbb E_B[hh^\top]+\rho I.
$$

head weight 的更新为：

$$
\Delta W=-\eta_W\nabla_WL\,G_h^{-1}.
$$

head bias 使用：

$$
\Delta b=-\eta_b\frac{\nabla_bL}{D_b}.
$$

其中 $D_b$ 可以取 batch logit-gradient variance：

$$
D_b=\operatorname{EMA}\left[\mathbb E_B[g_z^2]\right]+\rho.
$$

head hidden dimension 当前约为 $96$，所以 full covariance inverse 是可接受的。为了稳定，使用 EMA covariance：

$$
\bar G_{h,t}=\beta\bar G_{h,t-1}+(1-\beta)G_{h,t}.
$$

建议：

```text
cov_ema_beta = 0.95
rho_head = 1e-2
head_metric = full_cov
```

需要记录：

```text
head/cov_condition
head/cov_eig_min
head/cov_eig_max
head/update_norm
head/update_over_param
head/raw_grad_norm
head/precond_grad_norm
head/cos_raw_precond
head/predicted_descent
head/shadow_actual_delta
```

---

### 3.4 Stem Linear：input covariance update，先从 diagonal 开始

stem 是：

$$
h_0=W_{stem}x+b.
$$

它的自然 metric 同样来自输入 covariance：

$$
G_x=\mathbb E_B[xx^\top]+\rho I.
$$

完整更新为：

$$
\Delta W_{stem}=-\eta_{stem}\nabla_WL\,G_x^{-1}.
$$

但输入维度是 $784$，full covariance 每步更新和求逆会明显增加成本，而且可能在小 batch 下病态。因此本轮主线先用 diagonal covariance：

$$
D_x=\mathbb E_B[x^2]+\rho.
$$

更新为：

$$
\Delta W_{stem, :, i}=-\eta_{stem}\frac{\nabla_{W_{:,i}}L}{D_{x,i}}.
$$

可选的 full covariance stem 只做 P0/P1 方向审计，不进入正式训练，除非 diagonal stem 非常稳定且收益明显。

建议：

```text
stem_metric = diag_cov
rho_stem = 1e-2
cov_ema_beta = 0.95
```

需要记录：

```text
stem/cov_diag_min
stem/cov_diag_max
stem/cov_diag_p95
stem/update_norm
stem/update_over_param
stem/cos_raw_precond
stem/shadow_actual_delta
```

---

### 3.5 LayerNorm affine：diagonal Fisher-style update

LayerNorm affine 参数是：

$$
y=\gamma\hat x+\beta.
$$

它们不是函数基系数，但可以用局部输出梯度构造 diagonal Fisher-style metric。

对 $
\gamma$：

$$
F_\gamma=\operatorname{EMA}\left[\mathbb E_B[(g_y\odot \hat x)^2]\right]+\rho.
$$

对 $
\beta$：

$$
F_\beta=\operatorname{EMA}\left[\mathbb E_B[g_y^2]\right]+\rho.
$$

更新为：

$$
\Delta \gamma=-\eta_{LN}\frac{\nabla_\gamma L}{\sqrt{F_\gamma}},
$$

$$
\Delta \beta=-\eta_{LN}\frac{\nabla_\beta L}{\sqrt{F_\beta}}.
$$

LayerNorm 的更新必须非常保守，因为它影响所有后续 KAN input scale。建议一开始只在 head LayerNorm 上测试，然后再扩到 block LayerNorm 与 stem LayerNorm。

建议：

```text
ln_metric = diag_fisher
ln_lr_mult = 0.3 to 0.5 relative to rest_lr
rho_ln = 1e-2
```

需要记录：

```text
ln/gamma_update_norm
ln/beta_update_norm
ln/update_over_param
ln/fisher_mean
ln/fisher_p95
ln/output_scale_drift
ln/kan_input_std_drift
ln/oog_fraction
```

---

### 3.6 Residual alpha：本轮默认无更新

因为 v3.3 主线已经选择：

```text
alpha_mode = fixed1
```

所以主实验里 residual alpha 不更新。这样能避免把 expression rescue 与 alpha 自适应混在一起。

如果后续要测试 learnable alpha 的 AFU，可以定义 scalar geometry metric：

$$
G_{\alpha,k}=\mathbb E_B[\|\operatorname{KAN}_k(\operatorname{LN}(h_k))\|^2]+\rho.
$$

更新为：

$$
\Delta \alpha_k=-\eta_\alpha\frac{\nabla_{\alpha_k}L}{G_{\alpha,k}}.
$$

但这不进入 v3.4 core。

---

## 4. 本轮方法命名

本轮所有正式方法都以 `U-FULL-f085-alphaFixed1` 为基础。

### 4.1 Baselines

| method | 含义 |
|---|---|
| `AdamW-alphaFixed1` | 全参数 AdamW baseline |
| `F-V3-HARD-base30-v32best` | Fashion-specific best，仅作为上界对照 |
| `U-FULL-f085-Hybrid` | 当前统一 profile，KAN coeff functional，non-KAN AdamW |

### 4.2 AFU component methods

| method | KAN coeff | KAN bias | head | stem | LayerNorm | 目的 |
|---|---|---|---|---|---|---|
| `AFU-0-Hybrid` | Sobolev | AdamW/rest | AdamW | AdamW | AdamW | 当前主线 baseline |
| `AFU-1-BranchLocal` | Sobolev | diag functional | AdamW | AdamW | AdamW | 只测试 branch-local 非 KAN 参数 |
| `AFU-2-HeadCov` | Sobolev | diag functional | feature-cov | AdamW | AdamW | 测 head functional update |
| `AFU-3-HeadCov-LNHead` | Sobolev | diag functional | feature-cov | AdamW | head LN diag | 测最小 rest functional |
| `AFU-4-StemDiag` | Sobolev | diag functional | feature-cov | diag-cov | AdamW | 测 stem covariance update |
| `AFU-5-FullDiagLite` | Sobolev | diag functional | feature-cov | diag-cov | all LN diag | 第一个真正 all-functional-lite |
| `AFU-6-FullKFACLite` | Sobolev | diag functional | full-cov | diag-cov | all LN diag | 正式 all-functional candidate |

本轮不再运行：

```text
PGAdam
NormPGAdam
PostAdam
PostAdam-trust
naive all-rest-SGD except tiny negative smoke
```

它们已经不是这轮问题的正面对照。

---

## 5. 实验阶段总览

v3.4 分为九个阶段。它们不是平行大扫，而是逐步收敛。

```text
P0: AFU implementation smoke
P1: direction and shadow-step audit
P2: single-component AFU ablation
P3: cumulative AFU 3-seed comparison
P4: expression rescue micro-check
P5: AFU 5-seed confirm
P6: AFU 10-seed final confirm
P7: mechanism and representation audit
P8: wall-clock / memory / decision report
```

P0/P1 没过时，不允许进入 P2。P2 没有发现至少一个安全 component，不允许进入 full all-functional。P3 里 full AFU 如果明显欠拟合，就不扩 5-seed。

---

# 第一阶段：P0 AFU implementation smoke

## 6. P0 目标

P0 只验证实现能跑，并检查每个 parameter-group metric 是否数值健康。它不做方法结论。

这里最重要的是避免两类假失败：

```text
1. metric condition 太病态，导致 update 爆炸。
2. update 太小，导致 non-KAN 参数基本不动。
```

## 7. P0 设置

```text
datasets = Fashion-MNIST, KMNIST
train / val / test = 512 / 128 / 128
epochs = 1
seeds = 0
model = h32 / depth2 / basis8
alpha_mode = fixed1
profile = full_sobolev_gram from start
branch_final_scale = 0.85
```

方法：

```text
AdamW-alphaFixed1
AFU-0-Hybrid
AFU-1-BranchLocal
AFU-2-HeadCov
AFU-4-StemDiag
AFU-5-FullDiagLite
```

`AFU-6-FullKFACLite` 可以只在 Fashion smoke 跑一次，因为它包含最多 metric。

## 8. P0 必须记录

```text
run_failed
nan_or_inf_count
loss_is_finite
acc_is_finite
trust_clip_rate
kan/gram_condition
head/cov_condition
head/cov_eig_min
head/cov_eig_max
stem/cov_diag_min
stem/cov_diag_max
ln/fisher_mean
ln/fisher_p95
per_group/update_norm
per_group/update_over_param
per_group/raw_grad_norm
per_group/precond_grad_norm
per_group/cos_raw_precond
per_group/bad_shadow_step_rate
```

P0 必须输出每个 run 的：

```text
runs/{run_id}/curves.csv
runs/{run_id}/group_update_audit.csv
runs/{run_id}/metric_condition_audit.csv
```

## 9. P0 通过条件

P0 通过要求：

$$
\text{run failures}=0.
$$

$$
\text{NaN/Inf count}=0.
$$

KAN Sobolev Gram condition：

$$
1<\operatorname{cond}(M_{KAN}+\rho I)<10^4.
$$

Head covariance condition：

$$
1<\operatorname{cond}(G_h+\rho I)<10^5.
$$

每个参数组的 update ratio 满足：

$$
10^{-5}<\frac{\|\Delta \theta_g\|}{\|\theta_g\|+\epsilon}<10^{-1}.
$$

如果某组超过 $10^{-1}$，说明太激进；低于 $10^{-5}$，说明该组几乎没有被有效更新。

---

# 第二阶段：P1 direction and shadow-step audit

## 10. P1 目标

P1 在正式训练前，用同一个 batch 比较不同 update direction 是否合理。这个阶段非常重要，因为 all-functional update 很容易出现“最终训练失败但不知道是哪个组错了”的问题。

P1 不训练完整模型，而是做 shadow step：保存当前参数，计算一个候选 update，临时应用，重新计算同一个 mini-batch loss，再回滚。

对每个参数组 $g$，记录：

$$
\cos(d_g^{raw}, d_g^{AFU})
=
\frac{\langle d_g^{raw},d_g^{AFU}\rangle}{\|d_g^{raw}\|\|d_g^{AFU}\|+\epsilon}.
$$

同时记录 predicted descent：

$$
\widehat{\Delta L}_g=\langle \nabla_{\theta_g}L,\Delta \theta_g\rangle.
$$

以及 actual shadow loss delta：

$$
\Delta L_g^{shadow}=L(\theta+\Delta\theta_g)-L(\theta).
$$

## 11. P1 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0,1,2
model = h96 / depth4 / basis24
batch_size = 256
profile = U-FULL-f085-alphaFixed1
```

方法方向：

```text
raw SGD direction
AdamW direction, no parameter update, only audit
KAN Sobolev direction
head covariance direction
stem diagonal covariance direction
LayerNorm diagonal Fisher direction
full AFU combined direction
```

## 12. P1 必须记录

```text
per_group/direction_cos_raw_vs_afu
per_group/norm_ratio_afu_vs_raw
per_group/metric_norm
per_group/predicted_descent
per_group/shadow_actual_delta
per_group/pred_actual_ratio
per_group/bad_shadow_step
combined/shadow_actual_delta
combined/bad_shadow_step
combined/update_norm
combined/update_over_param
```

bad shadow step 定义为：

$$
\widehat{\Delta L}<0
\quad\text{but}\quad
\Delta L^{shadow}>0.
$$

## 13. P1 可视化

必须生成：

```text
direction cosine heatmap: parameter group x method
norm ratio heatmap
predicted vs actual descent scatter
bad shadow step rate bar
metric condition bar
```

这些图用来决定哪些 component 可以进入 P2。

## 14. P1 通过条件

单个参数组进入 P2 的条件：

$$
\operatorname{bad\_shadow\_rate}<0.10.
$$

$$
\operatorname{median}\left(\frac{\|d^{AFU}\|}{\|d^{raw}\|+\epsilon}\right)<10.
$$

$$
\operatorname{median}(\Delta L^{shadow})<0.
$$

如果某个组在 P1 不通过，该组不进入 P2 正式训练。

---

# 第三阶段：P2 single-component AFU ablation

## 15. P2 目标

P2 逐个替换 non-KAN 参数组，判断哪些 functional update 是安全的、哪些会造成欠拟合或不稳定。

P2 不追求最终最强结果，而是要回答：

```text
head functional update 是否安全？
stem functional update 是否安全？
LayerNorm functional update 是否安全？
KAN bias functional update 是否安全？
```

## 16. P2 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0,1,2
epochs = Fashion 30, KMNIST 20
model = h96 / depth4 / basis24
alpha_mode = fixed1
branch_final_scale = 0.85
metric = full_sobolev_gram from start
```

方法：

```text
AdamW-alphaFixed1
AFU-0-Hybrid
AFU-1-BranchLocal
AFU-2-HeadCov
AFU-HeadOnly
AFU-StemOnly
AFU-LNOnly
```

其中：

```text
AFU-HeadOnly:
  仅 head 从 AdamW 换成 feature-cov update，其余 non-KAN 仍 AdamW。

AFU-StemOnly:
  仅 stem 从 AdamW 换成 diag-cov update。

AFU-LNOnly:
  仅 LayerNorm affine 从 AdamW 换成 diag Fisher update。
```

## 17. P2 必须记录

标准任务指标：

```text
train_loss
val_loss
test_loss
train_acc
val_acc
test_acc
val_loss_auc
val_auc_improvement_vs_adamw
```

表达力指标：

```text
branch_over_adamw
no_kan_test_acc
no_kan_acc_drop
no_kan_drop_over_adamw
kan_logit_delta_norm_mean
kan_margin_contribution_mean
```

几何指标：

```text
phi_prime_p95
phi_prime_max
jacobian_condition_max
curvature_energy
credit_amplification_p95
```

组更新指标：

```text
per_group/update_norm_auc
per_group/update_over_param_mean
per_group/update_over_param_p95
per_group/bad_shadow_step_rate
per_group/cos_raw_precond_mean
```

## 18. P2 判定

一个 component 被认为“安全”需要满足：

$$
\text{acc gap vs AFU-0-Hybrid}<0.5\%.
$$

$$
\text{AUC improvement vs AdamW}>0.
$$

$$
\phi\text{ red and }J\text{ red not worse than Hybrid by more than }10\%.
$$

如果 component 同时提高 KMNIST no-KAN ratio：

$$
\Delta_{noKAN}^{component}>\Delta_{noKAN}^{Hybrid},
$$

则标记为 expression-positive component。

如果 component 造成 train acc 明显下降：

$$
\text{train acc gap vs Hybrid}>1.5\%,
$$

则标记为 rest-underfit，不进入 cumulative AFU。

---

# 第四阶段：P3 cumulative AFU 3-seed comparison

## 19. P3 目标

P3 把 P2 通过的 component 按保守到激进累积，测试真正的 all-functional-lite 是否可行。

P3 的核心问题是：

$$
\boxed{
\text{是否可以完全移除 non-KAN AdamW，同时不损失 accuracy / AUC / expression？}
}
$$

## 20. P3 方法

默认候选：

```text
AFU-0-Hybrid
AFU-2-HeadCov
AFU-3-HeadCov-LNHead
AFU-4-StemDiag
AFU-5-FullDiagLite
AFU-6-FullKFACLite
```

其中 `AFU-5-FullDiagLite` 是第一个真正意义上的 all-functional-lite：

```text
KAN coeff: Sobolev full Gram
KAN bias: diag functional
head: feature covariance
stem: diagonal activation covariance
LayerNorm: diagonal Fisher
alpha: fixed1, no update
```

也就是说，`AFU-5` 没有使用 AdamW 更新任何 trainable parameter。

`AFU-6` 与 `AFU-5` 的区别是：head 使用 full covariance，且所有 covariance 使用 EMA stabilized solve。

## 21. P3 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0,1,2
epochs = Fashion 30, KMNIST 20
profile = U-FULL-f085-alphaFixed1
```

学习率初始建议：

```text
KAN coeff lr = current U-FULL-f085 value
head functional lr = rest_lr * 0.5
stem functional lr = rest_lr * 0.3
LayerNorm functional lr = rest_lr * 0.3
bias functional lr = rest_lr * 0.5
```

注意：这不是大扫，只是为了防止非 KAN functional update 一开始过强。P3 只允许一个 conservative lr 设置。如果欠拟合明显，再进入 P3b 小范围 lr rescue。

## 22. P3 记录指标

除 P2 指标外，还必须记录：

```text
optimizer/uses_adamw_any_group
optimizer/adamw_group_count
optimizer/functional_group_count
optimizer/total_update_norm_by_group
optimizer/group_update_share
optimizer/rest_update_norm_vs_kan_update_norm
optimizer/nonkan_update_norm_vs_kan_update_norm
```

组更新占比定义：

$$
S_g=\frac{\|\Delta\theta_g\|}{\sum_j\|\Delta\theta_j\|+\epsilon}.
$$

如果某个 non-KAN 组 $S_g$ 长期接近 $0$，说明这个 functional update 实际没有起作用。

## 23. P3 可视化

必须生成：

```text
scorecard by method and dataset
loss curves with seed mean ± std
branch/no-KAN curves
per-group update norm stacked area
per-group update_over_param violin
metric condition over epoch
head covariance condition over epoch
stem covariance range over epoch
LayerNorm scale drift over epoch
Pareto plot: x = AUC imp, y = acc, size = noKAN, color = method
Pareto plot: x = phi red, y = noKAN, size = acc
```

## 24. P3 通过条件

AFU-lite weak pass：

$$
\text{acc gap vs Hybrid}<1.0\%.
$$

$$
\text{AUC improvement vs AdamW}>5\%.
$$

$$
\phi\text{ red}>20\%,\quad J\text{ red}>20\%.
$$

$$
\text{branch/AdamW}\in[0.50,0.90].
$$

AFU-lite medium pass：

$$
\text{acc gap vs Hybrid}<0.5\%.
$$

$$
\text{AUC improvement vs AdamW}>8\%.
$$

$$
\text{no-KAN ratio}>0.70.
$$

$$
\text{ECE reduction}>10\%.
$$

AFU-lite strong pass：

$$
\text{no AdamW groups remain}.
$$

$$
\text{acc not worse than Hybrid by more than }0.3\%.
$$

$$
\text{AUC not worse than Hybrid by more than }1\%.
$$

$$
\text{geometry not worse than Hybrid by more than }10\%.
$$

如果 `AFU-5/6` 不过，但 `AFU-2/3` 过，则结论是 partial AFU viable，而不是 all-functional viable。

---

# 第五阶段：P4 expression rescue micro-check

## 25. P4 目标

P4 只在以下情况下触发：

```text
AFU-0-Hybrid 或 AFU candidate 在 KMNIST 上 no-KAN ratio 仍低于 0.70，
但其他指标已经满足。
```

P4 不做大扫，只做两个点：

```text
branch_final_scale = 0.875
branch_final_scale = 0.90
```

其他全部不动。

## 26. P4 判定

P4 成功要求 KMNIST：

$$
\text{no-KAN ratio}>0.70.
$$

同时：

$$
\text{AUC improvement}>8\%.
$$

$$
\text{acc gap vs AdamW}<0.5\%.
$$

$$
\phi\text{ red}>25\%.
$$

如果 $f=0.90$ 让 no-KAN 过线但 phi/J 明显恶化，则不采用。

---

# 第六阶段：P5 AFU 5-seed confirm

## 27. P5 目标

P5 是正式确认阶段。只扩展 P3 中最好的两个 AFU 候选。

候选数量最多为两个：

```text
best partial AFU
best full AFU-lite
```

对照固定为：

```text
AdamW-alphaFixed1
AFU-0-Hybrid U-FULL-f085
F-V3-HARD-base30-v32best for Fashion upper bound
```

## 28. P5 设置

```text
datasets = Fashion-MNIST, KMNIST
seeds = 0,1,2,3,4
model = h96 / depth4 / basis24
alpha_mode = fixed1
epochs = Fashion 30, KMNIST 20
```

## 29. P5 主要 scorecard

| metric | Fashion target | KMNIST target |
|---|---:|---:|
| acc gap vs AdamW | $\leq 0$ preferred | $<0.5\%$ |
| acc gap vs Hybrid | $<0.5\%$ | $<0.5\%$ |
| AUC improvement vs AdamW | $>8\%$ | $>8\%$ |
| phi reduction | $>25\%$ | $>25\%$ |
| J reduction | $>20\%$ | $>20\%$ |
| branch/AdamW | $0.50$ to $0.90$ | $0.50$ to $0.90$ |
| no-KAN ratio | $>0.70$ | $>0.70$ |
| ECE reduction | $>10\%$ | $>10\%$ |

P5 之后，如果 full AFU-lite 不过，但 partial AFU 过，则不要强行推进 full AFU。

---

# 第七阶段：P6 10-seed final confirm

## 30. P6 目标

P6 只运行一个最终 AFU candidate。如果 P5 没有明确 candidate，则不运行 P6。

设置：

```text
seeds = 0..9
methods = AdamW, Hybrid-U-FULL-f085, final-AFU
optional Fashion upper bound = F-V3-HARD-base30
```

P6 输出最终报告：

```text
final_scorecard.csv
paired_delta_vs_adamw.csv
paired_delta_vs_hybrid.csv
mechanism_summary.csv
wallclock_summary.csv
failure_table.csv
```

## 31. P6 决策规则

### 31.1 AFU 成为主线

如果 final AFU 满足：

$$
\text{no AdamW groups remain}
$$

并且两个数据集都满足 medium pass，则：

```text
AFU becomes the new main optimizer candidate.
Hybrid becomes strong baseline.
```

### 31.2 Partial AFU 成立

如果只有 head / bias / LN 等部分 component 通过，而 stem 或 full rest functional 失败，则：

```text
Partial AFU is viable.
Hybrid remains default.
Safe non-KAN functional components may be included as optional profile.
```

### 31.3 AFU 失败

如果 all-functional-lite 明显低于 Hybrid：

$$
\text{acc gap vs Hybrid}>1\%
$$

或：

$$
\text{AUC improvement vs AdamW}<5\%,
$$

则结论是：

```text
True all-functional update is not yet viable with current non-KAN metrics.
Hybrid remains necessary.
```

注意这比上一轮结论更强，因为这一次才真正测试了 non-KAN functional metrics。

---

# 第八阶段：P7 mechanism and representation audit

## 32. P7 目标

P7 用来解释 AFU 的成败，尤其判断 non-KAN functional update 是帮助 expression，还是导致 underfit。

## 33. P7 必须记录的机制指标

### 33.1 Expression

```text
branch/global/branch_over_adamw
branch/layer_k/output_norm_ratio
ablation/no_kan_test_acc
ablation/no_kan_acc_drop
ablation/no_kan_drop_over_adamw
ablation/kan_logit_delta_norm_mean
ablation/kan_margin_contribution_mean
```

### 33.2 Representation drift

```text
repr/stem_output_norm_mean
repr/stem_output_std
repr/head_input_cov_condition
repr/head_input_effective_rank
repr/layer_k_ln_output_std
repr/layer_k_kan_input_oog_fraction
```

如果 AFU 失败但 KAN branch 正常，需要检查是否 head/stem representation 学得太慢。

### 33.3 Per-group learning dynamics

```text
group/head/update_norm_auc
group/stem/update_norm_auc
group/ln/update_norm_auc
group/kan_coeff/update_norm_auc
group/head/grad_norm_auc
group/stem/grad_norm_auc
group/ln/grad_norm_auc
group/kan_coeff/grad_norm_auc
```

### 33.4 Calibration and confidence

```text
calibration/ece
calibration/nll
confidence/correct_mean
confidence/wrong_mean
margin/mean
margin/p95
```

## 34. P7 关键图

必须生成：

```text
branch vs no-KAN scatter
no-KAN ratio over epoch
KAN margin contribution over epoch
head covariance condition over epoch
stem update norm over epoch
LayerNorm scale drift over epoch
per-group update share stacked area
representation effective rank over epoch
AUC improvement vs no-KAN ratio scatter
phi red vs no-KAN ratio scatter
```

## 35. P7 解释规则

如果 AFU acc 低、train acc 也低，并且 head/stem update norm 很小，则判为：

```text
nonKAN_functional_underfit
```

如果 AFU acc 低，但 update norm 很大，head covariance condition 高，loss 有 spike，则判为：

```text
nonKAN_metric_unstable
```

如果 AFU acc 好，但 no-KAN ratio 低，则判为：

```text
branch_expression_incomplete
```

如果 AFU geometry 坏，则判为：

```text
functional_metric_conflict
```

---

# 第九阶段：P8 wall-clock, memory, and final decision

## 36. P8 目标

AFU 可能比 Hybrid 更慢，因为 head covariance、stem covariance、LayerNorm Fisher 都有额外统计。P8 必须判断它是否值得。

## 37. P8 记录指标

```text
compute/step_time_ms
compute/epoch_time_sec
compute/total_train_time_sec
compute/kan_metric_solve_time_ms
compute/head_cov_build_time_ms
compute/head_cov_solve_time_ms
compute/stem_metric_time_ms
compute/ln_metric_time_ms
memory/peak_allocated_mb
memory/time_ratio_vs_adamw
memory/time_ratio_vs_hybrid
```

Target-matched：

```text
target/reached_relaxed_loss
target/epochs_to_relaxed_loss
target/time_to_relaxed_loss
target/reached_adamw_final_loss
target/time_to_adamw_final_loss
```

## 38. P8 可视化

```text
step time bar
metric overhead breakdown stacked bar
time-to-relaxed-loss bar
AUC vs step-time Pareto plot
accuracy vs memory Pareto plot
no-KAN ratio vs step-time plot
```

## 39. P8 决策

如果 AFU 只带来很小指标收益，但时间超过 Hybrid 太多：

$$
\text{time ratio vs Hybrid}>1.3,
$$

则不应作为 default，只能作为 analysis profile。

如果 AFU 改善 no-KAN / AUC / calibration，并且时间不超过：

$$
1.2\times \text{Hybrid},
$$

则可以保留为 candidate。

---

# 第十部分：Failure table

## 40. Failure tags

每个 run 都要自动打标签。

| failure type | 触发条件 | 含义 |
|---|---|---|
| `run_failed` | 运行错误 | 实现或环境问题 |
| `nan_or_inf` | loss / metric 非有限 | 数值不稳定 |
| `nonkan_underfit` | train acc 比 Hybrid 低 $>1.5\%$ | 非 KAN functional update 太弱 |
| `head_metric_unstable` | head condition $>10^5$ 或 shadow bad rate高 | head metric 问题 |
| `stem_metric_unstable` | stem update ratio 过大或 loss spike | stem metric 问题 |
| `ln_metric_unstable` | LN output std drift $>30\%$ | LN metric 破坏尺度 |
| `branch_underactive` | branch/A $<0.5$ 且 no-KAN ratio低 | KAN branch 没用起来 |
| `branch_overactive` | branch/A $>1.0$ | branch 过强 |
| `expression_incomplete` | no-KAN ratio $<0.7$ | 表达力贡献不足 |
| `geometry_not_preserved` | phi/J reduction 低于 gate | 几何优势消失 |
| `cov_condition_bad` | covariance condition 超阈值 | metric 病态 |
| `wallclock_too_slow` | time ratio vs Hybrid $>1.3$ | 工程不划算 |

输出：

```text
failure_table.csv
failure_summary_by_method.csv
failure_summary_by_dataset.csv
failure_summary_by_param_group.csv
```

可视化：

```text
failure counts by method
failure counts by parameter group
nonKAN_underfit vs update_norm scatter
cov_condition_bad vs AUC scatter
expression_incomplete vs branch/A scatter
```

---

# 第十一部分：最终报告应该怎么写

## 41. 如果 AFU 成功

如果 `AFU-5` 或 `AFU-6` 在两个数据集上都过 medium gate，可以写：

```text
We extend GA-FU from a hybrid optimizer into an all-functional update rule.
KAN edge coefficients are updated in Sobolev function space, while linear and normalization parameters are updated using activation-space or Fisher-style metrics.
The resulting AFU matches or improves the hybrid optimizer on accuracy, validation-loss AUC, geometry, and branch expression.
```

中文：

```text
我们把 GA-FU 从 hybrid optimizer 推进到 all-functional update。
KAN edge coefficients 继续使用 Sobolev 函数空间更新，head/stem/LayerNorm 则使用 activation covariance 或 diagonal Fisher-style metric。
实验表明 AFU 可以在不依赖 non-KAN AdamW 的情况下保持或提升 accuracy、AUC、几何和 branch expression。
```

## 42. 如果只有 partial AFU 成功

如果 head covariance 或 branch-local update 成功，但 full AFU-lite 失败，可以写：

```text
Partial non-KAN functional updates are viable, especially for the classifier head, but a fully all-functional DG-KAN optimizer remains challenging. The hybrid design remains the clean default.
```

中文：

```text
部分非 KAN 参数的 functional-style update 是可行的，尤其是分类 head；但完全移除 non-KAN AdamW 仍然困难。因此 hybrid 仍然是默认主线。
```

## 43. 如果 AFU 失败

如果 AFU 全部失败，可以写：

```text
A true all-functional optimizer was tested with parameter-group-specific metrics. The result shows that KAN coefficient functional update is robust, but non-KAN functional updates currently underfit or destabilize the representation. This supports keeping AdamW for non-KAN parameters.
```

中文：

```text
我们测试了真正的 all-functional optimizer，并为不同非 KAN 参数组设计了对应 metric。结果显示 KAN coefficient 的 functional update 仍然稳健，但非 KAN 参数的 functional update 当前要么欠拟合，要么破坏 representation。因此 non-KAN AdamW 仍然有必要。
```

这个结论才是对“所有参数都 functional update”问题的正面回答。

---

# 第十二部分：推荐执行顺序

## 44. 最小执行路线

建议按以下顺序执行：

```text
1. P0 smoke：所有 AFU metric 路径必须先数值通过。
2. P1 direction audit：筛掉不安全的 parameter group metric。
3. P2 component ablation：单独测试 head / stem / LN / bias。
4. P3 cumulative AFU：只组合 P2 中安全的 component。
5. P5 5-seed confirm：只扩展最多两个候选。
6. P6 10-seed final：只有候选明确时才跑。
7. P7/P8 机制和工程分析。
```

不要从一开始就跑 full AFU 10 seeds。否则如果失败，很难知道是 head、stem、LN 还是 bias 造成的。

---

# 45. 本轮最重要的预期结果

我对下一轮的预期是：

```text
HeadCov 最可能成功。
KAN bias functional update 大概率安全。
LayerNorm functional update 风险中等，可能需要很小 lr。
StemDiag 风险最大，可能欠拟合或拖慢 early learning。
Full AFU-lite 未必能赢 Hybrid，但会给出真正答案。
```

如果最后只有 `AFU-2-HeadCov` 成功，也已经很有价值，因为它说明：

$$
\boxed{
\text{非 KAN 参数不是不能 functional update，而是需要合适的 parameter-group metric。}
}
$$

如果 `AFU-5/6` 成功，那就可以把项目推进到更强的叙事：

$$
\boxed{
\text{DG-KAN can be trained by a fully geometry-aware optimizer, not merely a hybrid one.}
}
$$

如果 `AFU-5/6` 失败，也同样有价值，因为这会把当前主线彻底固定为：

$$
\boxed{
\text{KAN coefficients require functional update; non-KAN parameters still require AdamW.}
}
$$

这比上一轮 AdamW-style UO 的失败结论更准确，也更能回答你的原始问题。

