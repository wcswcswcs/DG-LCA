# DG-LCA v0.3 实验验证方案（详细 W&B 执行版）

> 版本：v0.2-detailed  
> 对象：DG-LCA v0.3 保守实验版  
> 定位：实验执行协议，不涉及具体代码实现  
> 公式格式：Typora 友好，使用 `$...$` 与 `$$...$$`  
> 核心目标：用可复现实验验证 **local VJP distillation + functional-space preconditioning + adapter-only memory tradeoff** 是否成立，并明确失败边界。

---

## 0. 这版实验方案解决什么问题

上一版方案只给了 Gate 和指标大纲，不足以指导执行。本版把实验拆成可直接落地的协议，明确：

1. 每个 Gate 到底验证什么假设；
2. 每个实验使用哪些数据集、模型、方法和 baseline；
3. 每个 run 的 W&B config 应记录什么；
4. 每个 step / epoch / audit 应上传什么 metric；
5. W&B dashboard 应画什么图；
6. 如何判断收敛性；
7. 如何公平比较 DG-LCA 与 baseline；
8. 达到什么程度算成功，失败后如何收缩。

本方案不试图证明：

$$
\text{DG-LCA replaces BP}
$$

而是验证：

$$
\text{local VJP can be distilled}
+
\text{functional preconditioning can stabilize structured modules}
+
\text{multi-layer routing error can be measured}
+
\text{memory tradeoff can be audited}
$$

---

## 1. 总体实验假设与判定路径

### 1.1 大目标

验证 DG-LCA v0.3 的保守命题：

> 在 KAN-like、adapter、LoRA、小型 residual block 等结构化模块中，BP 的局部 VJP 操作是否可以被条件线性 router 蒸馏；函数空间预条件更新是否能改善局部优化稳定性；这些局部机制是否在特定场景中带来可解释的 memory / compute / performance tradeoff。

核心公式为：

$$
\hat g_k
=
C_k^\phi(h_k,h_{k+1})[g_{k+1}]
\approx
G_k^{-1}J_k^\top G_{k+1}g_{k+1}
$$

以及：

$$
a
\leftarrow
 a-\eta(M+\rho I)^{-1}\nabla_a\mathcal L
$$

其中第一式验证 **learned local VJP**，第二式验证 **functional-space preconditioning**。

---

### 1.2 小目标

本实验方案分成五个 Gate。

| Gate | 名称 | 验证问题 | 失败后的处理 |
|---|---|---|---|
| Gate 0 | Credit rank diagnostic | 真实 credit 是否可压缩 | 收缩到 adapter / KAN / top-$r$ 子空间 |
| Gate 1 | Single-block VJP router | 单个 block 的 VJP 是否能被 router 学准 | 不做 sequential routing |
| Gate 2 | KAN functional update | 函数空间预条件是否优于普通参数更新 | functional update 降级为可选 preconditioner |
| Gate 3 | Sequential routing audit | 多层 learned router 串联是否误差可控 | 不做 macro-router / tree routing |
| Gate 4 | Adapter-only memory test | 是否有实际 memory / compute / performance tradeoff | 不宣称低显存优势 |

推荐执行顺序：

$$
\boxed{
\text{Gate 0}
\rightarrow
\text{Gate 1}
\rightarrow
\text{Gate 2}
\rightarrow
\text{Gate 3}
\rightarrow
\text{Gate 4}
}
$$

---

## 2. 全局实验设计规范

### 2.1 所有实验必须记录的 W&B config

每个 W&B run 必须包含以下 config 字段，保证不同方法可以横向比较。

#### 实验元信息

| config key | 含义 | 示例 |
|---|---|---|
| `exp/gate` | Gate 编号 | `gate1_vjp_router` |
| `exp/group` | 实验组 | `mnist_mlp8_router_rank_sweep` |
| `exp/method` | 方法名称 | `diag_lowrank_router` |
| `exp/baseline_type` | baseline 类型 | `bp`, `synthetic_gradient`, `feedback_alignment` |
| `exp/seed` | 随机种子 | `0`, `1`, `2`, `3`, `4` |
| `exp/run_id_manual` | 人工可读 run 名 | `g1_mnist_mlp8_r16_seed0` |

#### 数据集配置

| config key | 含义 |
|---|---|
| `data/name` | 数据集名称 |
| `data/train_size` | 训练样本数 |
| `data/val_size` | 验证样本数 |
| `data/test_size` | 测试样本数 |
| `data/batch_size_train` | 训练 batch size |
| `data/batch_size_eval` | 评估 batch size |
| `data/preprocess` | 数据预处理方式 |
| `data/augmentation` | 是否使用增强 |

#### 模型配置

| config key | 含义 |
|---|---|
| `model/name` | 模型名称，例如 `mlp`, `res_mlp`, `kan_mlp`, `small_cnn`, `adapter_resnet` |
| `model/depth` | block 数 |
| `model/hidden_dim` | hidden dimension |
| `model/block_type` | block 类型 |
| `model/num_params` | 总参数量 |
| `model/trainable_params` | 可训练参数量 |
| `model/frozen_backbone` | 是否冻结主干 |

#### Router 配置

| config key | 含义 |
|---|---|
| `router/enabled` | 是否启用 router |
| `router/type` | `identity`, `diagonal`, `diag_lowrank`, `block_diag`, `attention_linear`, `nonlinear_mlp` |
| `router/rank` | low-rank rank |
| `router/condition_inputs` | 使用哪些条件，例如 `h_k`, `h_k_h_kplus1`, `sketch` |
| `router/linear_in_credit` | 是否结构性保证对 credit 线性 |
| `router/num_params` | router 参数量 |
| `router/update_frequency` | router 更新频率 |
| `router/audit_frequency` | exact VJP audit 频率 |

#### Functional update 配置

| config key | 含义 |
|---|---|
| `func_update/enabled` | 是否启用函数空间更新 |
| `func_update/type` | `none`, `sobolev`, `rkhs`, `diag_precond` |
| `func_update/damping_rho` | damping 系数 $\rho$ |
| `func_update/sobolev_alpha` | Sobolev 一阶项权重 $\alpha$ |
| `func_update/sobolev_beta` | Sobolev 二阶项权重 $\beta$ |
| `func_update/kernel` | RKHS kernel 类型 |
| `func_update/basis_count` | basis / spline 数量 |

#### Optimizer 与训练预算

| config key | 含义 |
|---|---|
| `optim/name` | `sgd`, `adam`, `adamw`, `precond_sgd` |
| `optim/lr` | 学习率 |
| `optim/weight_decay` | weight decay |
| `optim/scheduler` | scheduler 类型 |
| `train/max_steps` | 最大训练步数 |
| `train/max_epochs` | 最大 epoch 数 |
| `train/early_stop_patience` | early stopping patience |
| `train/grad_clip` | gradient clipping |

#### 资源配置

| config key | 含义 |
|---|---|
| `resource/gpu_type` | GPU 类型 |
| `resource/num_gpus` | GPU 数量 |
| `resource/precision` | `fp32`, `bf16`, `fp16` |
| `resource/checkpointing` | 是否使用 checkpointing |
| `resource/compile` | 是否使用 compile / graph optimization |

---

### 2.2 所有实验必须上传的 W&B scalar metrics

这些 metric 每个 run 都要有，便于统一 dashboard 比较。

#### 训练与验证指标

| metric key | 频率 | 含义 |
|---|---:|---|
| `train/loss` | every step | 训练 loss |
| `train/acc` | every step 或 epoch | 训练 accuracy，回归任务可省略 |
| `val/loss` | every epoch / audit | 验证 loss |
| `val/acc` | every epoch / audit | 验证 accuracy |
| `test/loss` | final | 测试 loss |
| `test/acc` | final | 测试 accuracy |
| `train/grad_norm` | every step | 可训练参数梯度范数 |
| `train/update_norm` | every step | 参数更新范数 |
| `train/lr` | every step | 当前学习率 |

#### 收敛性指标

| metric key | 频率 | 含义 |
|---|---:|---|
| `conv/steps_to_target_val_loss` | final | 达到目标 val loss 所需 step |
| `conv/steps_to_target_val_acc` | final | 达到目标 val acc 所需 step |
| `conv/time_to_target_sec` | final | 达到目标所需 wall-clock time |
| `conv/loss_auc` | final | 训练曲线下面积，越小越好 |
| `conv/val_loss_slope_recent` | epoch | 最近窗口的 val loss 下降斜率 |
| `conv/plateau_step` | final | 判定进入 plateau 的 step |
| `conv/diverged` | final | 是否发散，0/1 |
| `conv/nan_count` | every epoch | NaN / Inf 次数 |

#### 资源指标

| metric key | 频率 | 含义 |
|---|---:|---|
| `perf/step_time_ms` | every step | 每步耗时 |
| `perf/samples_per_sec` | every step | 吞吐 |
| `perf/epoch_time_sec` | epoch | 每个 epoch 时间 |
| `memory/peak_allocated_mb` | every epoch / final | peak allocated memory |
| `memory/peak_reserved_mb` | every epoch / final | peak reserved memory |
| `memory/model_params_mb` | final | 模型参数显存估计 |
| `memory/optimizer_mb` | final | optimizer state 显存估计 |
| `memory/activation_or_interface_mb` | audit | activation 或 interface 显存估计 |
| `memory/router_mb` | final | router 参数与激活显存估计 |
| `memory/stats_audit_mb` | final | SVD、audit、sketch buffer 显存 |

#### 比较指标

| metric key | 含义 |
|---|---|
| `compare/acc_gap_vs_bp` | 与 BP 的 accuracy 差距 |
| `compare/loss_gap_vs_bp` | 与 BP 的 validation loss 差距 |
| `compare/memory_reduction_vs_bp_pct` | 相对 vanilla BP 的显存下降百分比 |
| `compare/memory_reduction_vs_ckpt_pct` | 相对 checkpointing 的显存下降百分比 |
| `compare/wallclock_overhead_vs_bp_pct` | 相对 BP 的 wall-clock 开销 |
| `compare/wallclock_overhead_vs_ckpt_pct` | 相对 checkpointing 的 wall-clock 开销 |

---

### 2.3 收敛性怎么判断

不能只看最后一个点。每个方法至少报告四类收敛性。

#### 固定 step 收敛

所有方法训练相同 step 数，例如：

$$
T \in \{1000, 5000, 20000\}
$$

比较：

1. final train loss；
2. final val loss；
3. final test acc；
4. loss curve AUC。

loss curve AUC 定义为：

$$
\text{AUC}_{\text{loss}}
=
\frac{1}{T}\sum_{t=1}^{T}\mathcal L_t
$$

越小代表同等训练预算下整体收敛更快。

---

#### 目标阈值收敛

设定目标值来自 BP baseline，例如：

$$
\mathcal L_{\text{target}}
=
\mathcal L_{\text{BP,final}}+\delta
$$

或：

$$
\text{Acc}_{\text{target}}
=
\text{Acc}_{\text{BP,final}}-\delta
$$

记录：

$$
\text{steps\_to\_target}
$$

和：

$$
\text{time\_to\_target}
$$

推荐分类任务使用：

$$
\delta_{\text{acc}}\in\{1\%,2\%,3\%\}
$$

回归任务使用：

$$
\delta_{\text{loss}}=0.05\times |\mathcal L_{\text{BP,final}}|
$$

---

#### Plateau 收敛

定义最近窗口 $W$ 内验证 loss 的相对改善：

$$
\Delta_W
=
\frac{\mathcal L_{t-W}^{\text{val}}-\mathcal L_t^{\text{val}}}
{|\mathcal L_{t-W}^{\text{val}}|+\epsilon}
$$

若连续 $P$ 个窗口满足：

$$
\Delta_W < \tau
$$

则认为进入 plateau。

建议默认：

$$
W=500\ \text{steps},\quad P=3,\quad \tau=0.005
$$

上传：

- `conv/plateau_step`；
- `conv/val_loss_slope_recent`；
- `conv/is_plateaued`。

---

#### 稳定性收敛

一个方法即使 final loss 好，如果经常出现尖峰、NaN、梯度爆炸，也不能算稳定。必须记录：

| metric key | 含义 |
|---|---|
| `stability/loss_spike_count` | loss 突增次数 |
| `stability/grad_explosion_count` | grad norm 超阈值次数 |
| `stability/nan_or_inf_count` | NaN / Inf 次数 |
| `stability/diverged` | 是否发散 |
| `stability/seed_std_val_loss` | 不同 seed 的 val loss 标准差 |
| `stability/seed_std_acc` | 不同 seed 的 accuracy 标准差 |

loss spike 可定义为：

$$
\mathcal L_t > \mathcal L_{t-1}^{\text{EMA}} \times 1.5
$$

---

### 2.4 如何与 baseline 公平比较

每个核心实验必须进行三种比较。

#### Step-matched comparison

所有方法训练相同 step 数。

回答：

> 同样训练步数下，谁的 loss / accuracy 更好？

---

#### Wall-clock-matched comparison

所有方法使用相同 wall-clock 时间预算。

回答：

> 同样时间下，DG-LCA 是否因为 router / audit overhead 变慢？

---

#### Memory-budget-matched comparison

限定相同 peak memory budget，例如：

$$
M_{\max}=4\text{GB}, 8\text{GB}, 16\text{GB}
$$

回答：

> 在相同显存上限下，哪个方法能用更大 batch、更深模型或更高分辨率？

---

### 2.5 多 seed 与统计报告

每个关键配置至少：

$$
N_{\text{seed}}=5
$$

资源受限时可先用 $3$ 个 seed 做探索，但正式结果必须用 $5$ 个 seed。

W&B 需要按 group 聚合：

- mean；
- std；
- 95% bootstrap CI；
- best seed；
- worst seed。

正式表格不只报告 best run，而要报告：

$$
\text{mean} \pm \text{std}
$$

并附带 worst-case。

---

## 3. W&B Dashboard 总设计

建议为每个项目建立 7 个 dashboard 页面。

### 3.1 Page 1：Overview

目标：快速判断每个 Gate 是否通过。

必须包含：

1. run summary table；
2. method vs final val loss bar chart；
3. method vs test acc bar chart；
4. method vs peak memory bar chart；
5. method vs step time bar chart；
6. pass / fail scorecard。

推荐图表：

| 图 | x | y | group / color |
|---|---|---|---|
| Final accuracy | method | `test/acc` | dataset |
| Final loss | method | `val/loss_final` | model |
| Memory | method | `memory/peak_allocated_mb` | dataset |
| Time-to-target | method | `conv/time_to_target_sec` | dataset |
| Pass score | Gate | pass rate | method |

---

### 3.2 Page 2：Gate 0 Credit Rank

必须包含：

1. layer-wise $E_{64}^{(2)}$ heatmap；
2. layer-wise $E_{128}^{(2)}$ heatmap；
3. singular value scree plots；
4. participation rank over training；
5. subspace overlap heatmap；
6. hidden-space vs adapter-subspace rank comparison。

推荐 W&B metrics：

| metric key | 含义 |
|---|---|
| `credit/layer_{k}/E16_var` | top 16 variance energy |
| `credit/layer_{k}/E32_var` | top 32 variance energy |
| `credit/layer_{k}/E64_var` | top 64 variance energy |
| `credit/layer_{k}/E128_var` | top 128 variance energy |
| `credit/layer_{k}/participation_rank` | participation rank |
| `credit/layer_{k}/entropy_rank` | entropy rank |
| `credit/layer_{k}/spectral_decay_slope` | 奇异值衰减斜率 |
| `credit/layer_{k}/batch_overlap_top64` | batch 间 top-64 子空间 overlap |
| `credit/layer_{k}/epoch_overlap_top64` | epoch 间 top-64 子空间 overlap |
| `credit/global/mean_E64_var` | 全层平均 top-64 energy |
| `credit/global/worst_E64_var` | 最差层 top-64 energy |
| `credit/global/mean_PR` | 全层平均 participation rank |

---

### 3.3 Page 3：Gate 1 VJP Router Fidelity

必须包含：

1. router validation cosine curve；
2. per-layer cosine heatmap；
3. relative error heatmap；
4. norm ratio histogram；
5. linearity residual curve；
6. rank vs fidelity Pareto plot；
7. stale curve。

推荐图表：

| 图 | x | y | color |
|---|---|---|---|
| Router train curve | step | `router/train/cos_mean` | router type |
| Router val curve | step | `router/val/cos_mean` | router type |
| Rank Pareto | `router/rank` | `router/val/cos_mean` | memory |
| Norm calibration | teacher norm | predicted norm | router type |
| Stale curve | stale steps | `router/stale/cos` | block |
| Linearity | step | `router/val/lin_resid` | router type |

---

### 3.4 Page 4：Gate 2 KAN Functional Update

必须包含：

1. train / val loss curves；
2. convergence-to-target curves；
3. $\max |\phi'|$ over time；
4. $\int |\phi''|^2$ over time；
5. Jacobian spectrum curves；
6. predicted vs actual descent scatter；
7. damping / smoothness hyperparameter heatmap。

推荐 W&B metrics：

| metric key | 含义 |
|---|---|
| `kan/phi_prime_max` | 所有 edge function 最大斜率 |
| `kan/phi_prime_p95` | 斜率 95 分位 |
| `kan/phi_double_prime_energy_mean` | 平均曲率能量 |
| `kan/phi_double_prime_energy_max` | 最大曲率能量 |
| `kan/sobolev_norm_mean` | 平均 Sobolev norm |
| `kan/jacobian_sigma_max` | 局部 Jacobian 最大奇异值 |
| `kan/jacobian_sigma_min` | 局部 Jacobian 最小奇异值 |
| `kan/jacobian_condition` | Jacobian 条件数 |
| `kan/precond_matrix_condition` | $M+\rho I$ 条件数 |
| `descent/pred` | predicted descent |
| `descent/actual` | actual descent |
| `descent/ratio` | actual / predicted |
| `descent/sign_agreement` | predicted 与 actual 是否同号 |

---

### 3.5 Page 5：Gate 3 Sequential Routing

必须包含：

1. per-layer credit cosine heatmap；
2. depth vs cosine profile；
3. teacher-forced vs self-routed comparison；
4. drift over depth；
5. credit norm over depth；
6. training loss for each routing method；
7. predicted vs actual descent over training。

推荐 W&B metrics：

| metric key | 含义 |
|---|---|
| `routing/layer_{k}/cos_bp_teacher_forced` | teacher-forced 与 BP 的 cosine |
| `routing/layer_{k}/cos_bp_self` | self-routed 与 BP 的 cosine |
| `routing/layer_{k}/relerr_self` | self-routed relative error |
| `routing/layer_{k}/norm_ratio_self` | self-routed norm ratio |
| `routing/layer_{k}/drift_self_vs_tf` | self-routed 与 teacher-forced drift |
| `routing/global/mean_cos_self` | 全层平均 self cosine |
| `routing/global/worst_cos_self` | 最差层 self cosine |
| `routing/global/mean_drift` | 平均 drift |
| `routing/global/error_amplification` | 误差放大因子 |
| `routing/global/credit_norm_growth` | credit norm 随深度增长率 |

---

### 3.6 Page 6：Gate 4 Adapter Memory / Compute

必须包含：

1. peak memory stacked bar；
2. wall-clock vs validation loss；
3. throughput vs method；
4. memory vs accuracy Pareto；
5. audit frequency vs performance；
6. router overhead pie / bar；
7. adapter-subspace gradient cosine curves。

推荐 W&B metrics：

| metric key | 含义 |
|---|---|
| `adapter/grad_cos_adapter_subspace` | adapter 子空间 gradient cosine |
| `adapter/grad_cos_hidden_space` | hidden-space gradient cosine |
| `adapter/projected_credit_rank` | adapter credit rank |
| `adapter/performance_gap_vs_bp` | 与 adapter BP 性能差距 |
| `memory/interface_mb` | interface memory |
| `memory/router_param_mb` | router 参数显存 |
| `memory/router_activation_mb` | router activation 显存 |
| `memory/local_update_mb` | local update 短期显存 |
| `memory/audit_buffer_mb` | audit buffer 显存 |
| `compute/router_time_ms` | router 耗时 |
| `compute/audit_time_ms` | audit 耗时 |
| `compute/local_update_time_ms` | local update 耗时 |
| `compute/router_overhead_pct` | router 额外开销百分比 |
| `compute/audit_overhead_pct` | audit 额外开销百分比 |

---

### 3.7 Page 7：Failure Analysis

必须包含：

1. 哪些层 rank 高；
2. 哪些层 VJP router 学不准；
3. teacher-forced 好但 self-routed 坏的案例；
4. norm ratio 爆炸案例；
5. router stale 速度；
6. functional update 过平滑案例；
7. predicted descent 与 actual descent 不一致案例；
8. memory 优势被 router / audit overhead 吃掉的案例。

W&B 应上传 failure table，字段包括：

| 字段 | 含义 |
|---|---|
| `failure/type` | 失败类型 |
| `failure/layer` | 所在层 |
| `failure/step` | 发生 step |
| `failure/method` | 方法 |
| `failure/metric_value` | 对应 metric 值 |
| `failure/diagnosis` | 初步诊断 |
| `failure/action` | 建议修正 |

---

## 4. Gate 0：Credit 子空间诊断

### 4.1 验证目标

Gate 0 验证 learned router 的必要条件：真实 credit 是否在低维结构化子空间中。

要验证的假设：

$$
\text{observed credit directions are compressible}
$$

具体为：

$$
E_r^{(2)}
=
\frac{\sum_{i=1}^{r}s_i^2}{\sum_i s_i^2}
$$

是否在较小的 $r$ 下已经接近 $1$。

---

### 4.2 实验矩阵

#### 数据集

| 难度 | 数据集 | 目的 |
|---|---|---|
| Toy | Two Moons | 可视化 credit 子空间 |
| Toy | Circles | 非线性边界 sanity check |
| Toy | $y=\sin x$ | 回归 sanity check |
| Small | MNIST | 低复杂度真实分类 |
| Small | Fashion-MNIST | 稍难视觉分类 |
| Small | CIFAR-10 | 小型自然图像 |
| Optional | CIFAR-100 | 类别更多，credit 可能更复杂 |
| Optional | AG News / SST-2 | 文本 adapter 子空间 |

#### 模型

| 模型 | depth | hidden dim | 目的 |
|---|---:|---:|---|
| MLP | 4 | 256 / 512 | 浅层基线 |
| MLP | 8 | 256 / 512 | 中等深度 |
| Residual MLP | 8 | 256 / 512 | 信息保持结构 |
| KAN-like MLP | 4 / 8 | basis 8 / 16 / 32 | 函数模块 |
| Small CNN | 4 / 8 blocks | channels 32 / 64 | 图像 block |
| Frozen backbone + adapter | 4 / 8 adapters | adapter dim 16 / 32 / 64 | 低维子空间 |

#### Credit 来源

| credit source | 作用 |
|---|---|
| full BP credit | 主诊断对象 |
| local exact VJP credit | block-local 对照 |
| adapter-projected credit | 低维结构化子空间 |
| random feedback credit | 负对照 |
| local loss credit | 与局部监督比较 |
| synthetic-gradient credit | 与 learned gradient 比较 |

---

### 4.3 收集频率

每个训练 run 在以下 step 收集 credit：

$$
S=\{0,100,500,1000,2000,5000,10000,T_{\text{final}}\}
$$

如果训练较短，可按 epoch 收集：

$$
\text{epoch}\in\{0,1,2,5,10,20,50,\text{final}\}
$$

每次收集至少使用：

$$
B_{\text{eval}}\in\{128,256,512,1024\}
$$

必须做 batch-size sensitivity，因为小 batch 会制造假低秩。

---

### 4.4 W&B 上传指标

#### 每层指标

对每层 $k$ 上传：

| metric key | 计算 |
|---|---|
| `credit/layer_{k}/E16_var` | $E_{16}^{(2)}$ |
| `credit/layer_{k}/E32_var` | $E_{32}^{(2)}$ |
| `credit/layer_{k}/E64_var` | $E_{64}^{(2)}$ |
| `credit/layer_{k}/E128_var` | $E_{128}^{(2)}$ |
| `credit/layer_{k}/PR` | $1/\sum_i p_i^2$ |
| `credit/layer_{k}/entropy_rank` | $\exp(-\sum_i p_i\log(p_i+\epsilon))$ |
| `credit/layer_{k}/spectral_decay_slope` | log singular value 的拟合斜率 |
| `credit/layer_{k}/rank_90` | 达到 $E_r^{(2)}\geq 0.9$ 的最小 $r$ |
| `credit/layer_{k}/rank_95` | 达到 $E_r^{(2)}\geq 0.95$ 的最小 $r$ |

#### 子空间稳定性

定义 top-$r$ 子空间基 $U_t^{(r)}$ 与 $U_{t'}^{(r)}$ 的 overlap：

$$
\text{overlap}_r(t,t')
=
\frac{1}{r}\left\|{U_t^{(r)}}^\top U_{t'}^{(r)}\right\|_F^2
$$

上传：

| metric key | 含义 |
|---|---|
| `credit/layer_{k}/batch_overlap_top64` | 不同 batch 的 top-64 overlap |
| `credit/layer_{k}/epoch_overlap_top64` | 不同 epoch 的 top-64 overlap |
| `credit/layer_{k}/seed_overlap_top64` | 不同 seed 的 top-64 overlap |
| `credit/global/mean_batch_overlap_top64` | 全层平均 batch overlap |
| `credit/global/worst_batch_overlap_top64` | 最差层 batch overlap |

#### 全局聚合指标

| metric key | 含义 |
|---|---|
| `credit/global/mean_E64_var` | 全层平均 $E_{64}^{(2)}$ |
| `credit/global/worst_E64_var` | 最差层 $E_{64}^{(2)}$ |
| `credit/global/mean_E128_var` | 全层平均 $E_{128}^{(2)}$ |
| `credit/global/worst_E128_var` | 最差层 $E_{128}^{(2)}$ |
| `credit/global/mean_PR` | 全层平均 PR |
| `credit/global/max_PR` | 最大 PR |
| `credit/global/hidden_vs_adapter_rank_ratio` | hidden-space rank / adapter rank |

---

### 4.5 W&B 可视化

必须做以下图：

1. **Layer × Step heatmap**：颜色为 `credit/layer_k/E64_var`；
2. **Layer × Step heatmap**：颜色为 `credit/layer_k/PR`；
3. **Scree plot**：不同层的 normalized $s_i^2$；
4. **Rank-to-energy curve**：$r$ vs $E_r^{(2)}$；
5. **Subspace overlap heatmap**：不同 epoch / batch 之间的 overlap；
6. **Hidden vs Adapter rank bar chart**；
7. **Batch-size sensitivity line plot**：batch size vs $E_{64}^{(2)}$。

---

### 4.6 Gate 0 成功标准

#### 弱通过

多数层满足：

$$
E_{128}^{(2)}>0.8
$$

并且：

$$
\text{batch\_overlap}_{64}>0.5
$$

#### 中等通过

多数层满足：

$$
E_{64}^{(2)}>0.8
$$

或：

$$
E_{128}^{(2)}>0.9
$$

并且：

$$
\text{batch\_overlap}_{64}>0.6
$$

adapter / KAN-like 子空间满足：

$$
\text{rank}_{\text{adapter}} < 0.5\times \text{rank}_{\text{hidden}}
$$

#### 强通过

多个模型与数据集上同时满足：

$$
E_{64}^{(2)}>0.8
$$

$$
\text{batch\_overlap}_{64}>0.7
$$

$$
\text{epoch\_overlap}_{64}>0.6
$$

---

### 4.7 Gate 0 失败判定

若多数层满足：

$$
E_{128}^{(2)}<0.5
$$

或：

$$
\text{batch\_overlap}_{64}<0.3
$$

则 full hidden-space learned router 不成立。后续必须收缩到：

1. adapter / LoRA 子空间；
2. KAN-like 模块；
3. top-$r$ projected credit；
4. exact local VJP；
5. functional preconditioning alone。

---

## 5. Gate 1：Single-block Local VJP Router

### 5.1 验证目标

验证单个 block 上是否存在低成本条件线性 router，使得：

$$
\hat g_k=C_k^\phi(h_k,h_{k+1})[g_{k+1}]
$$

逼近：

$$
g_k^{\text{teacher}}
=
G_k^{-1}J_k^\top G_{k+1}g_{k+1}
$$

此 Gate 不证明低成本训练，只证明 router architecture 是否有表达力。

---

### 5.2 数据拆分

每个 block 的 VJP distillation 数据由四元组构成：

$$
(h_k,h_{k+1},g_{k+1},g_k^{\text{teacher}})
$$

必须拆成：

| split | 来源 | 用途 |
|---|---|---|
| train-router | 训练 batch / checkpoint | 训练 router |
| val-router | 不同 batch，同一训练阶段 | early stopping 与超参选择 |
| test-router-in-stage | 不同 batch，同一 checkpoint | 测同分布泛化 |
| test-router-cross-stage | 不同 checkpoint | 测训练阶段泛化 |
| test-router-ood-credit | random / mixed / self-routed credit | 测 credit 分布外泛化 |

---

### 5.3 实验矩阵

#### 模型与 block

| block type | 数据集 | 目的 |
|---|---|---|
| MLP block | MNIST, Fashion-MNIST | 基础验证 |
| residual MLP block | MNIST, CIFAR-10 | 残差结构 |
| CNN block | CIFAR-10 | 图像局部结构 |
| KAN-like block | regression, MNIST | 函数模块 |
| adapter block | CIFAR-10 / AG News | 低维子空间 |

#### Router 类型

| router | 形式 | 作用 |
|---|---|---|
| identity | $C[g]=g$ | 最弱 baseline |
| diagonal | $C[h][g]=D(h)\odot g$ | 只学缩放 |
| diagonal + low-rank | $D(h)\odot g+U(h)(V(h)^\top g)$ | 主方法 |
| block diagonal | group-wise linear | head / channel 分组 |
| attention-linear | $A_{tok}(h)gB_{chan}(h)^\top$ | token 模型 |
| nonlinear MLP | arbitrary $C(h,g)$ | 负对照，测线性破坏 |

#### Rank sweep

$$
r\in\{0,4,8,16,32,64\}
$$

其中 $r=0$ 表示 diagonal-only。

---

### 5.4 Baseline

必须比较：

1. exact local VJP teacher，作为 oracle；
2. identity router；
3. diagonal router；
4. random feedback matrix；
5. learned random projection；
6. synthetic-gradient style predictor；
7. feedback alignment；
8. direct feedback alignment；
9. nonlinear router negative control。

---

### 5.5 W&B 上传指标

#### Router 训练指标

| metric key | 频率 | 含义 |
|---|---:|---|
| `router/train/loss_vjp_norm` | step | normalized MSE |
| `router/train/cos_mean` | step | train cosine 均值 |
| `router/train/relerr_mean` | step | train relative error |
| `router/train/norm_ratio_mean` | step | predicted / teacher norm |
| `router/train/lin_resid` | step | 线性性 residual |
| `router/val/loss_vjp_norm` | eval | val normalized MSE |
| `router/val/cos_mean` | eval | val cosine 均值 |
| `router/val/cos_p05` | eval | cosine 5 分位，反映坏样本 |
| `router/val/relerr_mean` | eval | val relative error |
| `router/val/relerr_p95` | eval | relative error 95 分位 |
| `router/val/norm_ratio_mean` | eval | norm ratio 均值 |
| `router/val/norm_ratio_p95` | eval | norm ratio 95 分位 |
| `router/val/lin_resid` | eval | 线性性 residual |

#### 每层指标

| metric key | 含义 |
|---|---|
| `router/layer_{k}/cos_mean` | 第 $k$ 层 cosine 均值 |
| `router/layer_{k}/cos_p05` | 第 $k$ 层 cosine 5 分位 |
| `router/layer_{k}/relerr_mean` | 第 $k$ 层 relerr 均值 |
| `router/layer_{k}/relerr_p95` | 第 $k$ 层 relerr 95 分位 |
| `router/layer_{k}/norm_ratio_mean` | 第 $k$ 层 norm ratio |
| `router/layer_{k}/lin_resid` | 第 $k$ 层线性性 residual |

#### 泛化测试

| metric key | 含义 |
|---|---|
| `router/scale_test/a0_5_error` | $C(0.5g)$ vs $0.5C(g)$ error |
| `router/scale_test/a2_error` | $C(2g)$ vs $2C(g)$ error |
| `router/superposition/error` | $C(g_1+g_2)$ vs $C(g_1)+C(g_2)$ error |
| `router/ood_credit/cos_mean` | OOD credit 输入下 cosine |
| `router/cross_stage/cos_mean` | 跨 checkpoint 泛化 cosine |

#### Stale 指标

在 block 参数更新 $\tau$ step 后重新评估：

| metric key | 含义 |
|---|---|
| `router/stale/cos_tau_10` | 10 step 后 cosine |
| `router/stale/cos_tau_50` | 50 step 后 cosine |
| `router/stale/cos_tau_100` | 100 step 后 cosine |
| `router/stale/relerr_tau_100` | 100 step 后 relerr |
| `router/stale/decay_rate` | stale 衰减率 |

#### 成本指标

| metric key | 含义 |
|---|---|
| `router/num_params` | router 参数量 |
| `router/param_ratio_vs_block` | router 参数量 / block 参数量 |
| `router/forward_time_ms` | router forward 耗时 |
| `router/train_time_ms` | router 训练耗时 |
| `router/memory_mb` | router 显存 |
| `router/flops_estimate` | FLOPs 估计 |

---

### 5.6 W&B 可视化

必须创建：

1. **router val cosine curve**：step vs `router/val/cos_mean`；
2. **router relerr curve**：step vs `router/val/relerr_mean`；
3. **per-layer heatmap**：layer × step，颜色为 cosine；
4. **rank Pareto**：x=`router/rank`，y=`router/val/cos_mean`，点大小=`router/memory_mb`；
5. **norm calibration scatter**：x=$\|g^{teacher}\|$，y=$\|\hat g\|$；
6. **linearity residual curve**；
7. **stale curve**：$\tau$ vs cosine；
8. **OOD credit performance bar**。

---

### 5.7 Gate 1 收敛性判定

router 训练不是看 loss 下降就行，而看 validation fidelity 是否 plateau。

router converged 当满足：

$$
\Delta \text{cos}_{W}<0.005
$$

且：

$$
\Delta \text{relerr}_{W}<0.01
$$

连续 $P=3$ 个评估窗口成立。

上传：

- `router/conv/plateau_step`；
- `router/conv/best_val_cos`；
- `router/conv/best_val_relerr`；
- `router/conv/overfit_gap_cos`，定义为 train cos - val cos。

---

### 5.8 Gate 1 成功标准

#### 弱通过

KAN-like / small MLP block：

$$
\cos>0.9
$$

$$
\operatorname{relerr}<0.35
$$

$$
0.7<\rho<1.3
$$

#### 中等通过

MLP / CNN / adapter block：

$$
\cos>0.95
$$

$$
\operatorname{relerr}<0.2
$$

$$
0.8<\rho<1.2
$$

$$
r_{\text{lin}}<0.05
$$

#### 强通过

同时满足：

1. 多层、多 checkpoint 上 $\cos>0.95$；
2. OOD credit 上 $\cos>0.9$；
3. stale 100 step 后仍 $\cos>0.9$；
4. router overhead 小于 local VJP teacher 的 30%。

---

### 5.9 Gate 1 失败判定

若出现以下任一情况，不进入 Gate 3：

$$
\cos<0.85
$$

或：

$$
\operatorname{relerr}>0.5
$$

或：

$$
\rho>1.5\quad\text{or}\quad\rho<0.5
$$

或：

$$
r_{\text{lin}}>0.1
$$

修正路线：

1. 增大 rank；
2. 改为 block-diagonal / attention-linear router；
3. 限制到 adapter 子空间；
4. 只学习 top-$r$ credit projection；
5. 回退到 exact local VJP。

---

## 6. Gate 2：KAN-like Functional-Space Update

### 6.1 验证目标

验证函数空间预条件更新是否比普通 coefficient update 更稳定、更快或更平滑。

KAN-like 层：

$$
y_j=\sum_i\phi_{ji}(x_i)
$$

普通 coefficient update：

$$
a\leftarrow a-\eta\nabla_a\mathcal L
$$

functional preconditioned update：

$$
a\leftarrow a-\eta(M+\rho I)^{-1}\nabla_a\mathcal L
$$

---

### 6.2 实验矩阵

#### 数据集

| 类型 | 数据集 / 任务 | 目的 |
|---|---|---|
| 1D regression | $y=\sin x$ | 平滑函数 sanity check |
| 1D regression | $y=\sin x+0.3\sin 5x$ | 中频函数 |
| 1D regression | smooth step | 局部变化 |
| 2D regression | $y=\sin x_1+\cos x_2$ | 多输入函数 |
| classification | Two Moons | 非线性分类 |
| classification | MNIST | 简单视觉 |
| classification | Fashion-MNIST | 中等视觉 |
| classification | CIFAR-10 small | 更难视觉 |

#### 方法

| method | 更新 |
|---|---|
| coefficient SGD | $a\leftarrow a-\eta\nabla_a\mathcal L$ |
| coefficient Adam | Adam on coefficients |
| coefficient AdamW | AdamW on coefficients |
| diagonal preconditioner | diagonal approximation of $M$ |
| Sobolev preconditioner | $(M_{Sob}+\rho I)^{-1}$ |
| RKHS preconditioner | $(M_{RKHS}+\rho I)^{-1}$ |
| optional K-FAC / NG | 更强预条件 baseline |

#### Hyperparameter sweep

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

建议 sweep：

$$
\alpha\in\{0,10^{-4},10^{-3},10^{-2},10^{-1}\}
$$

$$
\beta\in\{0,10^{-5},10^{-4},10^{-3},10^{-2}\}
$$

$$
\rho\in\{10^{-6},10^{-5},10^{-4},10^{-3},10^{-2}\}
$$

学习率 sweep：

$$
\eta\in\{10^{-4},3\times 10^{-4},10^{-3},3\times 10^{-3},10^{-2}\}
$$

---

### 6.3 W&B 上传指标

#### 任务指标

| metric key | 含义 |
|---|---|
| `train/loss` | train loss |
| `val/loss` | validation loss |
| `test/loss` | test loss |
| `train/acc` | train accuracy |
| `val/acc` | validation accuracy |
| `test/acc` | test accuracy |
| `generalization/gap_loss` | val loss - train loss |
| `generalization/gap_acc` | train acc - val acc |

#### 收敛指标

| metric key | 含义 |
|---|---|
| `conv/loss_auc` | loss curve AUC |
| `conv/steps_to_target_val_loss` | 达到目标 val loss 的 step |
| `conv/steps_to_target_val_acc` | 达到目标 val acc 的 step |
| `conv/time_to_target_sec` | 达到目标耗时 |
| `conv/plateau_step` | plateau step |
| `conv/largest_stable_lr` | 最大稳定学习率 |

#### 函数形状指标

| metric key | 含义 |
|---|---|
| `kan/phi_prime_max` | 最大 $|\phi'|$ |
| `kan/phi_prime_mean` | 平均 $|\phi'|$ |
| `kan/phi_prime_p95` | $|\phi'|$ 95 分位 |
| `kan/phi_double_prime_energy_mean` | 平均 $\int|\phi''|^2$ |
| `kan/phi_double_prime_energy_max` | 最大 $\int|\phi''|^2$ |
| `kan/sobolev_norm_mean` | 平均 Sobolev norm |
| `kan/sobolev_norm_max` | 最大 Sobolev norm |
| `kan/coefficient_norm` | coefficient norm |
| `kan/update_norm_coeff` | coefficient update norm |
| `kan/update_norm_function` | function-space update norm |

#### Jacobian 与 credit 稳定性

| metric key | 含义 |
|---|---|
| `kan/jacobian_sigma_max` | $\sigma_{max}(J)$ |
| `kan/jacobian_sigma_min` | $\sigma_{min}(J)$ |
| `kan/jacobian_condition` | $\sigma_{max}/(\sigma_{min}+\epsilon)$ |
| `kan/credit_amplification_mean` | $\|g_x\|/\|g_y\|$ 均值 |
| `kan/credit_amplification_p95` | credit amplification 95 分位 |

#### Descent alignment

| metric key | 含义 |
|---|---|
| `descent/pred` | predicted descent |
| `descent/actual` | actual descent |
| `descent/ratio` | actual / predicted |
| `descent/sign_agreement` | sign 是否一致，0/1 |
| `descent/sign_agreement_rate` | 最近窗口一致率 |
| `descent/pred_negative_actual_positive_count` | 预测下降但实际上升次数 |

#### Metric 条件数

| metric key | 含义 |
|---|---|
| `metric/precond_condition` | $\kappa(M+\rho I)$ |
| `metric/precond_min_eig` | 最小特征值 |
| `metric/precond_max_eig` | 最大特征值 |
| `metric/damping_rho` | damping |
| `metric/alpha` | Sobolev $\alpha$ |
| `metric/beta` | Sobolev $\beta$ |

---

### 6.4 W&B 可视化

必须画：

1. **train / val loss curve**：比较 Adam、Sobolev、RKHS；
2. **steps-to-target bar chart**；
3. **loss AUC bar chart**；
4. **$\max |\phi'|$ over time**；
5. **$\int |\phi''|^2$ over time**；
6. **Jacobian condition over time**；
7. **predicted vs actual descent scatter**；
8. **damping heatmap**：x=$\rho$，y=$\alpha$ 或 $\beta$，颜色为 val loss / condition；
9. **Pareto plot**：x=`kan/phi_double_prime_energy_mean`，y=`val/loss`；
10. **learning-rate stability heatmap**。

---

### 6.5 Gate 2 成功标准

#### 弱通过

相比 Adam：

1. final validation loss 不差于 Adam 超过 2%；
2. $\max |\phi'|$ 更低；
3. $\int |\phi''|^2$ 更低；
4. Jacobian condition 更稳定。

#### 中等通过

至少两个数据集上满足任一主要收益：

$$
\text{steps\_to\_target reduction}>20\%
$$

或：

$$
\text{val loss improvement}>5\%
$$

或在同等 accuracy 下：

$$
\text{curvature reduction}>30\%
$$

并且：

$$
\text{descent sign agreement rate}>0.75
$$

#### 强通过

在 MNIST / Fashion-MNIST / CIFAR-10 small 上同时满足：

1. validation accuracy 不低于 Adam；
2. convergence speed 更快；
3. $\kappa(J)$ 更低或更稳定；
4. 对学习率更不敏感；
5. $\text{descent sign agreement rate}>0.8$。

---

### 6.6 Gate 2 失败判定

若出现以下任一情况，functional update 降级为辅助方法：

1. validation loss 明显差于 Adam 超过 5%；
2. 过平滑导致 train loss 也降不下去；
3. $\kappa(M+\rho I)$ 过高且需要极大 damping；
4. `descent/sign_agreement_rate < 0.6`；
5. 学习率稳定性不如 Adam；
6. compute overhead 过高且没有优化收益。

---

## 7. Gate 3：Sequential Learned Routing Audit

### 7.1 验证目标

验证 learned router 串联后是否仍能产生可用 credit。

真实递推：

$$
g_k^{true}=C_k^{true}g_{k+1}^{true}
$$

learned self-routed 递推：

$$
\hat g_k=C_k^\phi(h_k,h_{k+1})[\hat g_{k+1}]
$$

关键风险是误差累积：

$$
e_k=C_k^{true}e_{k+1}+E_kg_{k+1}^{true}+E_ke_{k+1}
$$

---

### 7.2 实验矩阵

#### 数据集

| 数据集 | 模型 | 目的 |
|---|---|---|
| Two Moons | 4 / 8 block MLP | sanity check |
| MNIST | 4 / 8 / 12 block MLP | 主实验 |
| Fashion-MNIST | 4 / 8 / 12 block MLP | 稍难分类 |
| CIFAR-10 small | small CNN / residual MLP | 图像任务 |
| Optional AG News | adapter transformer | 低维 adapter routing |

#### 方法

| method | 说明 |
|---|---|
| exact BP | oracle baseline |
| local exact VJP sequential | 局部精确 VJP，但仍串行 |
| learned router teacher-forced | 输入 BP credit，测 router 本身 |
| learned router self-routed | 输入上一层预测 credit，测部署模式 |
| learned router + scheduled mixing | 混合 teacher / predicted credit |
| synthetic gradient | baseline |
| feedback alignment | baseline |
| direct feedback alignment | baseline |
| local loss | baseline |
| random feedback | negative control |

---

### 7.3 W&B 上传指标

#### 每层 credit fidelity

对每层 $k$ 上传：

| metric key | 含义 |
|---|---|
| `routing/layer_{k}/cos_bp_teacher_forced` | teacher-forced 与 BP cosine |
| `routing/layer_{k}/cos_bp_self` | self-routed 与 BP cosine |
| `routing/layer_{k}/relerr_teacher_forced` | teacher-forced relerr |
| `routing/layer_{k}/relerr_self` | self-routed relerr |
| `routing/layer_{k}/norm_ratio_teacher_forced` | teacher-forced norm ratio |
| `routing/layer_{k}/norm_ratio_self` | self-routed norm ratio |
| `routing/layer_{k}/drift_self_vs_tf` | self vs teacher-forced drift |
| `routing/layer_{k}/credit_norm_bp` | BP credit norm |
| `routing/layer_{k}/credit_norm_self` | self-routed credit norm |

#### 全局 routing 指标

| metric key | 含义 |
|---|---|
| `routing/global/mean_cos_tf` | teacher-forced 平均 cosine |
| `routing/global/mean_cos_self` | self-routed 平均 cosine |
| `routing/global/worst_cos_self` | self-routed 最差层 cosine |
| `routing/global/mean_relerr_self` | self-routed 平均 relerr |
| `routing/global/max_relerr_self` | self-routed 最大 relerr |
| `routing/global/mean_drift` | 平均 drift |
| `routing/global/max_drift` | 最大 drift |
| `routing/global/error_amplification` | 误差放大因子 |
| `routing/global/credit_norm_growth` | credit norm 随深度增长率 |

误差放大因子可定义为：

$$
\text{error\_amplification}
=
\frac{\|\hat g_0-g_0^{BP}\|}{\|\hat g_{K-1}-g_{K-1}^{BP}\|+\epsilon}
$$

#### 训练与 descent 指标

| metric key | 含义 |
|---|---|
| `train/loss` | train loss |
| `val/loss` | val loss |
| `test/acc` | test accuracy |
| `descent/pred_total` | 总 predicted descent |
| `descent/actual_total` | 总 actual descent |
| `descent/sign_agreement_rate` | sign agreement rate |
| `stability/divergence_count` | 发散次数 |

---

### 7.4 W&B 可视化

必须画：

1. **Layer × Step heatmap**：self-routed cosine；
2. **Depth profile**：layer index vs cosine；
3. **Teacher-forced vs self-routed line chart**；
4. **Drift over depth**；
5. **Credit norm over depth**；
6. **Error amplification over training**；
7. **Training loss by method**；
8. **Validation loss by method**；
9. **Predicted vs actual descent scatter**；
10. **4 / 8 / 12 block comparison bar chart**。

---

### 7.5 收敛性与比较方式

Gate 3 必须同时报告：

1. step-matched final accuracy；
2. wall-clock-matched final accuracy；
3. loss AUC；
4. time-to-target；
5. routing fidelity over training。

不能只报告 final accuracy，因为 self-routed 方法可能：

1. 前期看起来下降；
2. 中期 credit drift；
3. 后期发散或 plateau。

---

### 7.6 Gate 3 成功标准

#### 弱通过

4-block 模型：

$$
\text{routing/global/mean\_cos\_self}>0.8
$$

且 train loss 稳定下降，无发散。

#### 中等通过

8-block 模型：

$$
\text{routing/global/mean\_cos\_self}>0.8
$$

$$
\text{routing/global/worst\_cos\_self}>0.6
$$

并且：

$$
\text{accuracy gap vs BP}<3\%
$$

或：

$$
\text{val loss gap vs BP}<10\%
$$

#### 强通过

12-block 模型：

$$
\text{routing/global/mean\_cos\_self}>0.85
$$

$$
\text{routing/global/worst\_cos\_self}>0.7
$$

并且：

$$
\text{accuracy gap vs BP}<2\%
$$

$$
\text{descent sign agreement rate}>0.75
$$

---

### 7.7 Gate 3 失败判定

若 8-block 模型满足任一情况：

$$
\text{routing/global/mean\_cos\_self}<0.5
$$

或：

$$
\text{routing/global/worst\_cos\_self}<0.3
$$

或：

$$
\text{accuracy gap vs BP}>5\%
$$

或 teacher-forced 好但 self-routed 坏，则说明主要问题是：

$$
\text{distribution shift} + \text{error accumulation}
$$

此时不能推进 macro-router / tree routing。

---

## 8. Gate 4：Adapter-only Memory / Compute Test

### 8.1 验证目标

验证 DG-LCA 是否在 adapter / LoRA 等低维结构化模块中有实际训练收益。

模型形式：

$$
h_{k+1}
=
h_k
+
F_k^{frozen}(h_k)
+
A_k(h_k;\psi_k)
$$

只训练 adapter 参数 $\psi_k$。

---

### 8.2 实验矩阵

#### 数据集

| 类型 | 数据集 | 目的 |
|---|---|---|
| Image | CIFAR-10 | 主实验 |
| Image | CIFAR-100 | 更难分类 |
| Image | Tiny-ImageNet small | 中等规模 |
| Text | AG News | 文本分类 |
| Text | SST-2 | 情感分类 |
| Text | IMDB | 长文本分类 |

#### 模型

| 模型 | adapter 类型 |
|---|---|
| frozen small ResNet | bottleneck adapter |
| frozen small ViT | bottleneck adapter / LoRA |
| frozen Transformer encoder | adapter / LoRA |
| KAN-like adapter | edge-function adapter |

#### 方法

| method | 说明 |
|---|---|
| full BP adapter training | adapter 标准训练 |
| adapter BP + checkpointing | 强 memory baseline |
| local BP through adapter | 只对 adapter 局部反传 |
| exact local VJP + local update | 精确局部 VJP 版本 |
| learned router + local update | DG-LCA 部署版本 |
| synthetic gradient adapter | credit baseline |
| feedback alignment adapter | credit baseline |
| local loss adapter | local objective baseline |
| standard LoRA training | 实用 baseline |

---

### 8.3 W&B 上传指标

#### 性能指标

| metric key | 含义 |
|---|---|
| `val/loss` | validation loss |
| `val/acc` | validation accuracy |
| `test/loss` | test loss |
| `test/acc` | test accuracy |
| `compare/acc_gap_vs_adapter_bp` | 与 adapter BP 的 accuracy gap |
| `compare/loss_gap_vs_adapter_bp` | 与 adapter BP 的 loss gap |
| `conv/steps_to_target_val_acc` | 达到目标 acc 的 step |
| `conv/time_to_target_sec` | 达到目标 acc 的时间 |

#### Memory accounting

必须拆分记录，而不是只报 peak memory。

| metric key | 含义 |
|---|---|
| `memory/peak_allocated_mb` | 实际 peak allocated |
| `memory/peak_reserved_mb` | 实际 peak reserved |
| `memory/params_mb` | 参数显存 |
| `memory/optimizer_mb` | optimizer state |
| `memory/interfaces_mb` | block interface memory |
| `memory/router_param_mb` | router 参数 |
| `memory/router_activation_mb` | router activation |
| `memory/local_update_mb` | local update 短期 memory |
| `memory/audit_buffer_mb` | exact VJP audit buffer |
| `memory/stats_mb` | rank / metric / sketch 统计 |
| `memory/reduction_vs_bp_pct` | 相对 BP 降低百分比 |
| `memory/reduction_vs_ckpt_pct` | 相对 checkpointing 降低百分比 |

#### Compute accounting

| metric key | 含义 |
|---|---|
| `perf/step_time_ms` | 每步耗时 |
| `perf/samples_per_sec` | 吞吐 |
| `compute/forward_time_ms` | forward 耗时 |
| `compute/router_time_ms` | router 耗时 |
| `compute/audit_time_ms` | audit 耗时 |
| `compute/local_update_time_ms` | local update 耗时 |
| `compute/metric_time_ms` | metric / preconditioner 耗时 |
| `compute/router_overhead_pct` | router overhead |
| `compute/audit_overhead_pct` | audit overhead |
| `compute/total_overhead_vs_bp_pct` | 总 overhead vs BP |
| `compute/total_overhead_vs_ckpt_pct` | 总 overhead vs checkpointing |

#### Adapter credit 指标

| metric key | 含义 |
|---|---|
| `adapter/credit_E64_var` | adapter credit top-64 energy |
| `adapter/credit_PR` | adapter credit participation rank |
| `adapter/grad_cos_adapter_subspace` | adapter 子空间 gradient cosine |
| `adapter/grad_cos_hidden_space` | hidden-space gradient cosine |
| `adapter/router_cos` | router VJP cosine |
| `adapter/router_relerr` | router relative error |
| `adapter/router_stale` | router stale 指标 |

---

### 8.4 W&B 可视化

必须画：

1. **Memory stacked bar**：params / optimizer / activations / router / audit / stats；
2. **Peak memory vs method**；
3. **Accuracy vs peak memory Pareto**；
4. **Validation loss vs wall-clock time**；
5. **Throughput vs method**；
6. **Audit frequency vs accuracy / memory / time**；
7. **Router overhead pie chart**；
8. **Adapter credit rank over training**；
9. **Performance gap vs adapter BP table**。

---

### 8.5 Gate 4 成功标准

#### 弱通过

相比 adapter BP：

$$
\text{accuracy gap}<3\%
$$

$$
\text{val loss gap}<10\%
$$

$$
\text{peak memory reduction}>15\%
$$

此时 wall-clock 可以略高。

#### 中等通过

相比 adapter BP + checkpointing：

$$
\text{peak memory reduction}>10\%
$$

$$
\text{accuracy gap}<2\%
$$

$$
\text{wall-clock overhead}<1.5\times
$$

$$
\text{router overhead}<20\%
$$

#### 强通过

相比 adapter BP + checkpointing：

$$
\text{peak memory reduction}>20\%
$$

$$
\text{accuracy gap}<1\%
$$

$$
\text{wall-clock overhead}<1.2\times
$$

并且：

$$
\text{adapter/router\_cos}>0.9
$$

且 exact VJP audit 不需要每步做。

---

### 8.6 Gate 4 失败判定

若出现以下任一情况，不能宣称低显存优势：

1. 只比 vanilla BP 省，但不如 checkpointing；
2. wall-clock 超过 checkpointing 的 2 倍；
3. accuracy gap 超过 3%；
4. router / audit memory 吃掉大部分节省；
5. 必须每步 exact VJP teacher 才稳定；
6. deployment mode 明显差于 distillation mode。

---

## 9. 消融实验设计

### 9.1 Router rank 消融

比较：

$$
r\in\{0,4,8,16,32,64\}
$$

记录：

1. `router/val/cos_mean`；
2. `router/val/relerr_mean`；
3. `router/memory_mb`；
4. `router/forward_time_ms`；
5. downstream `val/loss`；
6. downstream `test/acc`。

可视化：rank vs fidelity vs cost Pareto。

---

### 9.2 State metric 消融

比较：

1. $G_k=I$；
2. diagonal variance metric；
3. whitened diagonal metric；
4. damped diagonal metric。

所有 inverse 使用：

$$
G_k^{-1}\rightarrow(G_k+\rho I)^{-1}
$$

记录：

1. VJP cosine；
2. norm ratio；
3. router relerr；
4. metric condition number；
5. noise amplification。

---

### 9.3 Functional metric 消融

比较：

1. no preconditioning；
2. diagonal preconditioning；
3. Sobolev $\alpha$ only；
4. Sobolev $\beta$ only；
5. Sobolev $\alpha+\beta$；
6. RKHS；
7. different damping $\rho$。

记录：

1. val loss；
2. convergence speed；
3. $\max|\phi'|$；
4. $\int|\phi''|^2$；
5. Jacobian condition；
6. descent sign agreement。

---

### 9.4 Exact VJP audit frequency 消融

比较：

| setting | 含义 |
|---|---|
| every step | 每步 exact VJP |
| every 5 steps | 每 5 step audit |
| every 20 steps | 每 20 step audit |
| every 100 steps | 每 100 step audit |
| warmup only | warmup 后不 audit |
| no audit | 完全 learned router |

记录：

1. router stale；
2. self-routed drift；
3. final performance；
4. compute overhead；
5. memory overhead。

---

### 9.5 Teacher-forced vs self-routed 消融

必须分开报告：

teacher-forced：

$$
\hat g_k=C_k(h_k,h_{k+1})[g_{k+1}^{BP}]
$$

self-routed：

$$
\hat g_k=C_k(h_k,h_{k+1})[\hat g_{k+1}]
$$

如果 teacher-forced 好但 self-routed 坏，结论是：

$$
\text{router fitted local VJP but deployment distribution shift failed}
$$

不能 claim learned routing 可部署。

---

## 10. 第一轮最小可执行实验包

如果资源有限，第一轮只做下面 6 个实验，而不是全部展开。

---

### Experiment A：Credit rank diagnostic

| 项目 | 设置 |
|---|---|
| 数据 | MNIST, Fashion-MNIST, CIFAR-10 |
| 模型 | 4-layer MLP, 8-layer MLP, KAN-like MLP |
| 方法 | full BP credit, adapter-projected credit, random feedback credit |
| seeds | 5 |
| 主要 W&B 图 | rank heatmap, scree plot, overlap heatmap |

成功：

$$
E_{128}^{(2)}>0.9
$$

至少在 KAN-like / adapter 子空间成立。

---

### Experiment B：Single-block VJP distillation

| 项目 | 设置 |
|---|---|
| 数据 | MNIST, CIFAR-10 |
| block | MLP block, KAN block, adapter block |
| router | diagonal, rank 8, rank 16, rank 32 |
| teacher | local exact VJP |
| seeds | 5 |
| 主要 W&B 图 | cosine curve, relerr curve, rank Pareto, stale curve |

成功：

$$
\cos>0.95
$$

$$
\operatorname{relerr}<0.2
$$

$$
0.8<\rho<1.2
$$

---

### Experiment C：Router OOD 与 stale 测试

| 项目 | 设置 |
|---|---|
| 输入 credit | BP credit, random credit, mixed credit, self-routed credit |
| stale | $\tau=10,50,100,200$ steps |
| 指标 | OOD cosine, stale decay, norm ratio |

成功：

$$
\cos_{OOD}>0.9
$$

且：

$$
\cos_{\tau=100}>0.9
$$

---

### Experiment D：KAN functional preconditioning

| 项目 | 设置 |
|---|---|
| 数据 | 1D regression, 2D regression, MNIST, Fashion-MNIST |
| baseline | coefficient Adam, AdamW, diagonal preconditioner |
| methods | Sobolev, RKHS |
| seeds | 5 |
| 主要 W&B 图 | loss curve, curvature curve, Jacobian condition, descent scatter |

成功：

1. validation loss 不差；
2. 收敛更快或 loss AUC 更小；
3. $\max|\phi'|$ 更低；
4. $\int|\phi''|^2$ 更低；
5. descent sign agreement rate $>0.75$。

---

### Experiment E：8-block sequential routing

| 项目 | 设置 |
|---|---|
| 数据 | MNIST, Fashion-MNIST |
| 模型 | 4 / 8 / 12 block residual MLP |
| methods | BP, local exact VJP, teacher-forced, self-routed, synthetic gradient, feedback alignment |
| seeds | 5 |
| 主要 W&B 图 | layer cosine heatmap, drift plot, loss curves |

成功：

8-block self-routed：

$$
\text{mean cosine}>0.8
$$

$$
\text{worst-layer cosine}>0.6
$$

accuracy gap vs BP：

$$
<3\%
$$

---

### Experiment F：Adapter-only memory test

| 项目 | 设置 |
|---|---|
| 数据 | CIFAR-10 或 AG News |
| 模型 | frozen small ResNet / frozen Transformer encoder |
| methods | adapter BP, adapter BP + checkpointing, exact local VJP, learned router, synthetic gradient |
| seeds | 5 |
| 主要 W&B 图 | memory stacked bar, accuracy-memory Pareto, wall-clock curves |

成功：

相比 adapter BP + checkpointing：

$$
\text{peak memory reduction}>10\%
$$

$$
\text{accuracy gap}<2\%
$$

$$
\text{wall-clock overhead}<1.5\times
$$

---

## 11. 最终论文级判定标准

### 11.1 最低可发表结果

满足：

1. Gate 1 single-block router 在 KAN / adapter / small MLP 上 $\cos>0.9$；
2. Gate 2 functional preconditioning 在 KAN-like 上比 Adam 更稳定；
3. 明确报告 sequential routing 的失败边界。

结论只能写：

> Local VJP distillation and functional-space updates are feasible in selected structured modules.

不能写：

> DG-LCA replaces BP.

---

### 11.2 中等强度结果

满足：

1. Gate 0 显示结构化子空间 credit 可压缩；
2. Gate 1 在多个 block 上 $\cos>0.95$；
3. Gate 2 收敛速度或稳定性显著优于 Adam；
4. Gate 3 在 8-block self-routed 中 accuracy gap $<3\%$。

可以 claim：

> DG-LCA components can approximate local BP operations under structured, low-dimensional conditions.

---

### 11.3 强结果

满足：

1. Gate 3 在 12-block 中仍稳定；
2. Gate 4 相比 adapter BP + checkpointing 有显存优势；
3. performance gap $<1\%-2\%$；
4. audit 不需要每步 exact VJP；
5. overhead 可控。

可以 claim：

> In structured adapter-like regimes, learned local VJP plus functional preconditioning can partially replace selected BP components with practical memory tradeoffs.

---

## 12. 失败结果也要如何记录

每次失败都要上传到 W&B failure table，而不是只在文字里描述。

### 12.1 必须记录的失败类型

| failure type | 触发条件 |
|---|---|
| `credit_high_rank` | $E_{128}^{(2)}<0.5$ |
| `credit_unstable_subspace` | overlap top-64 < 0.3 |
| `router_low_cosine` | router cosine < 0.85 |
| `router_norm_explosion` | norm ratio > 1.5 |
| `router_linearity_fail` | linearity residual > 0.1 |
| `router_stale_fast` | stale 100 step 后 cosine < 0.8 |
| `self_routed_drift` | self vs teacher drift 快速上升 |
| `descent_mismatch` | predicted descent < 0 but actual > 0 |
| `functional_oversmooth` | train loss 降不下去且 curvature 很低 |
| `memory_no_gain` | 不如 checkpointing 省显存 |
| `overhead_too_high` | wall-clock 超 baseline 2 倍 |

### 12.2 failure table 字段

| 字段 | 含义 |
|---|---|
| `failure/type` | 失败类型 |
| `failure/gate` | Gate 编号 |
| `failure/dataset` | 数据集 |
| `failure/model` | 模型 |
| `failure/method` | 方法 |
| `failure/layer` | 层号 |
| `failure/step` | step |
| `failure/metric_name` | 触发 metric |
| `failure/metric_value` | metric 值 |
| `failure/threshold` | 阈值 |
| `failure/diagnosis` | 诊断 |
| `failure/recommended_action` | 建议动作 |

---

## 13. 最终建议

这套实验不要追求一次性证明 DG-LCA 成为新训练范式。第一篇最稳的目标是：

$$
\boxed{
\text{single-block local VJP fidelity}
+
\text{KAN functional preconditioning stability}
+
\text{sequential routing boundary}
+
\text{strict memory accounting}
}
$$

如果 Gate 1 和 Gate 2 成功，即使 Gate 3 / Gate 4 没有完全成功，也有清晰贡献：

1. 证明某些结构化模块的 local VJP 可蒸馏；
2. 证明函数空间预条件能改善 KAN-like 模块稳定性；
3. 系统刻画 learned credit routing 的失败模式；
4. 给出比“替代 BP”更诚实的研究边界。

