# DG-KAN v6.4：Memory-Passing Graph-Free DWM2 与任务保持型 Functional Correction 详细实验计划

> 修订版说明：本计划基于 v6.3 no-fake 真实重跑结果重新校准。v6.3 已经真实跑通 P0-P4，P2 选出 `DWM2-poly2-compiled`，P3 选出 `ManualAdanLite` 为当前最强 task learner；但 P4 只有 partial winner，没有完整 all-dataset winner；P5 LightSmooth、P6 functional no-autograd smoke、P7-P9 confirm 均为 `not_run`。因此 v6.4 不能继承任何旧 proxy row 或固定数字结论。本轮计划的第一优先级是让 `DWM2-poly2-compiled` 的 backward memory 过线，第二优先级是把 `ManualAdanLite` 从 partial candidate 修成跨数据集稳定 winner，第三优先级才是真实验证 task-preserving functional correction。

---

## 0. 实验整体目标与本轮定位

本项目的终极目标不是让某个 KAN 变体在小数据集上跑出不错 accuracy，而是构建一个真正可用的 **PureKAN graph-free functional training system**。这个系统必须同时满足六类要求：模型结构必须是无 ordinary non-KAN 参数的 PureKAN；训练路径必须不依赖 PyTorch backward graph；forward 与 backward 的时间必须接近 MLP；backward memory 必须低于 MLP；收敛速度必须按 wall-clock time-to-target 具有竞争力；最终还要在 accuracy、validation-loss AUC、ECE 与 geometry 上超过或至少不弱于 `MLP-AdamW` 与 `PureKAN-AdamW`。

最终系统至少要满足：

$$
\frac{T_{\text{forward,KAN}}}{T_{\text{forward,MLP}}} \leq 1.25,
$$

$$
\frac{T_{\text{backward,KAN}}}{T_{\text{backward,MLP}}} \leq 1.25,
$$

$$
\frac{T_{\text{step,KAN}}}{T_{\text{step,MLP}}} \leq 1.25,
$$

$$
\frac{M_{\text{backward,KAN}}}{M_{\text{backward,MLP}}} < 1.00,
$$

并最终希望达到更强的 memory 目标：

$$
\frac{M_{\text{backward,KAN}}}{M_{\text{backward,MLP}}} \leq 0.80.
$$

任务性能方面，最终候选必须满足：

$$
\operatorname{Acc}_{KAN} \geq \max(\operatorname{Acc}_{MLP-AdamW},\operatorname{Acc}_{PureKAN-AdamW}),
$$

$$
\operatorname{ValLossAUC}_{time,KAN} \leq \operatorname{ValLossAUC}_{time,MLP-AdamW},
$$

$$
\operatorname{ECE}_{KAN} \leq \operatorname{ECE}_{MLP-AdamW}.
$$

这里必须明确：本计划中 `ValLossAUC` 指 validation loss 曲线下面积，**越低越好**；如果记录 accuracy AUC，字段必须写成 `ValAccAUC`，且 **越高越好**。后续所有 CSV、图表和 gate 都必须避免使用模糊字段名 `AUC`。

几何方面，最终候选至少要在一个 geometry 指标上优于 MLP 或 PureKAN reference，例如：

$$
\phi'_{p95,KAN}<\phi'_{p95,reference},
$$

或：

$$
\operatorname{Curvature}_{KAN}<\operatorname{Curvature}_{reference}.
$$

v6.3 真实重跑说明路线已经从“是否能 graph-free”进入“是否能 memory-pass 并稳定赢 task”的阶段。`DWM2-poly2-compiled` 已经出现明确 step-time 正信号：

```text
DWM2-poly2-compiled:
  step_ratio = 0.833302
  bmem_ratio = 1.144985
```

这意味着它在当前 P2 口径下 step time 已经优于 MLP reference，但 backward memory 仍高于 MLP。v6.4 的第一硬目标就是把：

$$
M_{\text{backward}}/M_{\text{MLP}}=1.144985
$$

压到：

$$
M_{\text{backward}}/M_{\text{MLP}}<1.0.
$$

如果按最终目标 $0.8$ 计算，当前还需要约：

$$
1-\frac{0.8}{1.144985}\approx 0.301
$$

也就是大约 $30\%$ 的 backward memory reduction。因此，本轮不是重新搜索 Dense RBF、U-FULL、TFU、FNG 或 SparseInterp 大空间，而是把实验主线压缩为：

$$
\boxed{\text{DWM2-poly2-compiled memory pass}}
+
\boxed{\text{ManualAdanLite all-dataset stability}}
+
\boxed{\text{task-preserving functional geometry correction}}.
$$

换句话说，本轮不尝试让 functional update 独立替代 Adam-like task learner，而是把 functional update 降级并重定义为 **不破坏任务下降的几何校正项**。新的组合方向写成：

$$
\Delta\theta_{new}=\Delta\theta_{task}+\lambda_t\Delta\theta_{geo},
$$

其中 $\Delta\theta_{task}$ 来自 `ManualAdanLite` 或其它通过 P2 的 task learner，$\Delta\theta_{geo}$ 只允许在通过 direction-quality gate 后作为小幅 correction 注入训练。

---

## 1. v6.3 真实重跑复盘与事实口径

v6.3 的真实重跑建立了两个必要前提。第一，实验不再接受 fake data、固定代理行或 mock/proxy 结论；P5/P6 中原先可能存在的固定代理行不得作为结果使用。第二，P0-P4 已经通过 W&B 和本地 CSV 留下真实 trace；P5-P9 因 P4 未产生完整 winner 而正确门禁为 `not_run`。

本轮计划必须使用下面事实口径。

### 1.1 P0/P1 的含义

P0 baseline contract 通过，说明 runner、核心构件、no-fake 防护与 artifact 记录流程具备基本可运行性。P1 component/kernel/cache profiler 生成 42 行真实 profiling 结果，主要失败来源为：

```text
F1_kernel_launch_overhead: 16
F2_transform_cost: 8
F3_cache_memory: 1
```

这说明当前瓶颈不是一个单独 kernel，而是 kernel launch、transform cost、cache memory 三类问题共同存在。因此 v6.4 的 P1 不应只报告总 memory，而必须拆成 phase-local peak memory 与 component-level runtime。

### 1.2 P2 的含义

P2 选出：

```text
selected_id = DWM2-poly2-compiled
step_ratio = 0.833302
bmem_ratio = 1.144985
```

这是真实进展，但不是完整成功。它说明 speed path 出现突破；memory path 仍未过线。v6.4 的 memory audit 必须围绕这个 exact method id 展开。历史或旧文档中的 `DWM2-poly2-fused-forward-adjoint` 如果指代同一实现，可以作为技术别名，但 CSV 和 W&B 中必须统一使用：

```text
DWM2-poly2-compiled
```

### 1.3 P3 的含义

P3 对四个 optimizer 做了真实训练验证，结果为：

```text
ManualAdanLite:
  dataset_pass_count = 3
  mean_acc = 0.815755
  time_win_count = 5
  p3_pass = 1

ManualAdam:
  dataset_pass_count = 3
  mean_acc = 0.813802
  time_win_count = 4
  p3_pass = 1

ManualAdamW:
  dataset_pass_count = 3
  mean_acc = 0.813802
  time_win_count = 4
  p3_pass = 1

ManualNesterovAdam:
  dataset_pass_count = 3
  mean_acc = 0.806424
  time_win_count = 0
  p3_pass = 0
```

因此 `ManualAdanLite` 是当前最强 task learner，但 `ManualAdam` 与 `ManualAdamW` 仍是有价值的 task-only baselines。v6.4 不能只跑 AdanLite；必须保留 AdamW/Adam baseline 用于判断 AdanLite 是否稳定真实提升。

### 1.4 P4 的含义

P4 没有得到完整 all-dataset winner。`A2-ManualAdanLite` 是 best partial candidate，而不是 confirmed survivor：

```text
A2-ManualAdanLite:
  mean_acc = 0.809896
  gap_vs_mlp = -0.026042
  pass_count = 2
  conclusion = partial
```

其它 partial candidates 包括 `A0-ManualAdamW`、`A3-ManualWinLite`、`A4-Lookahead-ManualAdamW`、`A6-WarmupCosine-ManualAdamW`。`A7 RoleWiseLR InputHeavy` 与 `A8 RoleWiseLR OutputWarmup` 明显退化，必须从主线候选集中删除或隔离。

所以 v6.4 的任务不是“确认 A2 已经成功”，而是验证：

$$
\boxed{\text{A2-ManualAdanLite 能否从 partial candidate 修成 all-dataset stable winner。}}
$$

### 1.5 P5/P6/P7/P8/P9 的事实口径

v6.3 中：

```text
P5 LightSmooth compatibility = not_run
P6 functional no-autograd smoke = not_run
P7 joint selection3 = not_run
P8 confirm5 = not_run
P9 confirm10 = not_run
```

原因是 empirical LightSmooth audit 与 empirical functional update smoke 尚未真实实现，固定代理行被禁止。v6.4 不允许引用任何“P5 已通过”或“P6 已失败且有具体数值”的旧 proxy 结论。P5/P6 必须在本轮作为首次真实 empirical audit 重新实现、重新运行、重新判断。

---

## 2. 当前离终极目标还差什么

### 2.1 Memory 缺口

当前最强 primitive 的 backward memory ratio 是：

$$
\frac{M_{\text{backward,DWM2}}}{M_{\text{backward,MLP}}}=1.144985.
$$

要达到 basic pass，需要降到：

$$
\frac{M_{\text{backward,DWM2}}}{M_{\text{backward,MLP}}}<1.0.
$$

要达到 final target，需要降到：

$$
\frac{M_{\text{backward,DWM2}}}{M_{\text{backward,MLP}}}\leq0.8.
$$

memory 可能来源包括 manual cache、hidden cache、delta buffer、temporary polynomial feature、workspace temporary、optimizer state、buffer allocation policy 和 MLP reference cache policy 不公平。v6.4 必须把它们逐项拆开，而不是继续只看一个 `bmem_ratio`。

### 2.2 Task stability 缺口

P3 已经显示 `DWM2-poly2-compiled` 能学任务，`ManualAdanLite` 是当前最佳；但 P4 只得到 partial winner。v6.4 必须从 single-run / partial pass 推进到：

```text
P2: short-budget recipe selection
P4: all-dataset stability repair
P6: 3-seed full-budget selection
P7: 5-seed confirm
P8: 10-seed final confirm
```

### 2.3 Functional correction 缺口

v6.3 没有真实运行 P6。v6.4 要第一次真实验证：

$$
\Delta\theta_{new}=\Delta\theta_{task}+\lambda_t\Delta\theta_{geo}.
$$

这里 functional correction 的基本要求不是“单独让 loss 下降”，而是在保持 task direction 质量的同时改善 geometry 或 ECE。它必须满足：

$$
\cos(d_{new},d_{task})\geq0.85,
$$

$$
\frac{\operatorname{HoldoutDescent}(d_{new})}{\operatorname{HoldoutDescent}(d_{task})}\geq0.95,
$$

$$
\operatorname{BadStepRate}(d_{new})\leq0.02.
$$

如果 correction 经 adaptive backtracking 后 $\lambda_t$ 总被压到 $0$，它不能算通过，只能说明 correction 不可用。

### 2.4 Geometry / ECE 缺口

P5 LightSmooth 尚未真实运行。v6.4 要验证 LightSmooth 或 geometry correction 是否可以在不牺牲 accuracy 和 wall-clock 的情况下改善：

```text
phi_prime_p95
curvature
jacobian_condition_proxy
ECE
NLL
logit drift
feature drift
```

如果 geometry 改善需要明显牺牲 task 或 memory，它不能进入 confirm。

### 2.5 Statistical confirmation 缺口

当前结论仍来自 P0-P4 的 rerun 与 partial candidate。最终 claim 必须通过 paired seed-level analysis，至少输出：

```text
paired_delta_mean
paired_delta_std
bootstrap_CI_low
bootstrap_CI_high
seedwise_failure_reason
```

P6/P7/P8 不能只看 mean；必须看 paired delta 与 failure case。

---

## 3. v6.4 核心假设

v6.4 的实验不是线性跑表，而是围绕六个假设展开。每个假设必须有明确实验、指标和成立标准。

### H1：`DWM2-poly2-compiled` 的 backward memory 可以通过 cache/streaming/buffer repair 压到 MLP 以下

当前 memory ratio 是 $1.144985$。如果 manual cache、delta buffer 和 temp polynomial feature 是主要来源，那么通过 `noHiddenCache`、`deltaStreaming`、`bufferReuse`、`noTemp-poly2`、`bf16 cache` 等修复，应能把 backward memory 降到 $<1.0$，同时不显著破坏 step time 和 gradient correctness。

H1 成立标准：

$$
\frac{M_{\text{backward}}}{M_{\text{MLP}}}<1.0,
$$

$$
\frac{T_{\text{step}}}{T_{\text{MLP}}}\leq1.10,
$$

$$
\operatorname{relerr}(\nabla)<10^{-4},
$$

$$
\cos(\nabla_{manual},\nabla_{reference})>0.999.
$$

如果只能达到 $M_{backward}/M_{MLP}\leq1.05$，则标记为 exploratory pass，不允许进入 final confirm。

### H2：`ManualAdanLite` 是当前最优 task learner，但需要 stability repair

v6.3 P3 说明 `ManualAdanLite` 是 P3 最强候选；P4 说明它只是 partial winner。v6.4 假设 `ManualAdanLite` 的不稳定来自 optimizer dynamics、restart policy、beta schedule 或 dataset-specific early phase，而不是 primitive 本身完全不可用。

H2 成立标准：

$$
\operatorname{Acc}_{candidate}\geq\operatorname{Acc}_{MLP}-0.005
$$

on all datasets, and at least two datasets satisfy:

$$
T_{target,candidate}\leq T_{target,MLP}.
$$

### H3：functional update 应作为 task-preserving correction，而不是主 optimizer

v6.4 不再测试 standalone functional direction 作为 task learner。新假设是：functional geometry direction 只要足够小，并通过 backtracking 保持 task descent，就可以改善 geometry / ECE 而不牺牲 accuracy。

H3 成立标准：

$$
\cos(d_{new},d_{task})\geq0.85,
$$

$$
\frac{\operatorname{HoldoutDescent}(d_{new})}{\operatorname{HoldoutDescent}(d_{task})}\geq0.95,
$$

$$
\operatorname{BadStepRate}(d_{new})\leq0.02,
$$

且至少一个 geometry 或 calibration 指标优于 task-only：

$$
\Delta\operatorname{Curvature}_{new}<\Delta\operatorname{Curvature}_{task}
$$

或：

$$
\Delta\operatorname{ECE}_{new}<\Delta\operatorname{ECE}_{task}.
$$

### H4：LightSmooth 只能作为低频事件，不能作为 every-step optimizer

v6.4 首次真实验证 LightSmooth compatibility。假设 LightSmooth 如果有效，应以 low-frequency event 或 late-phase event 形式使用，而不是每 step smoothing。

H4 成立标准：

$$
\operatorname{Acc}_{smooth}\geq\operatorname{Acc}_{task-only}-0.005,
$$

$$
\frac{T_{step+smooth}}{T_{task-only}}\leq1.10,
$$

并且 geometry 至少改善 $5\%$。

### H5：hard role-wise LR 已失败，但温和 role-wise optimizer state 仍值得验证

v6.3 中 `A7 RoleWiseLR InputHeavy` 与 `A8 RoleWiseLR OutputWarmup` 明显退化，所以 v6.4 不继续 hard role-wise LR。只验证：

```text
same LR, role-wise beta2
same LR, role-wise grad clipping
same LR, role-wise update normalization
```

H5 成立标准是它们至少不比 `ManualAdanLite` baseline 差，并能改善 P4 instability。

### H6：如果 DWM2-poly2 memory pass 但 task 失败，应该小幅增强表达力，而不是回到 dense RBF

如果 memory 已过但 KMNIST 或 Fashion 仍欠拟合，则优先测试：

```text
DWM2-poly3
DWM2-poly2+silu-base
DWM2-gated-poly2
DWM2-RBFK2-small-correction
```

但这些 fallback 必须先过 P1 efficiency gate，不能为了 task 牺牲 memory-pass 目标。

---

## 4. 实验阶段总览

v6.4 分为十一个阶段：

```text
P0: implementation contract and artifact sanity
P1: DWM2-poly2 memory finalization and phase-local memory audit
P2: DWM2-poly2 accelerated task recipe search
P3: functional correction direction-quality audit
P4: all-dataset acceleration stability repair
P5: empirical geometry / LightSmooth integration
P6: 3-seed full-budget candidate selection
P7: 5-seed confirm
P8: 10-seed final confirm
P9: fallback primitive decision
P10: failure diagnosis and route decision
```

P1 和 P3 是本轮硬门禁。P1 决定是否具备进入终极效率目标的资格；P3 决定 functional correction 是否能进入训练闭环。P5/P6 不允许使用 proxy row，必须真实实现、真实运行。

---

## 5. P0：Implementation Contract and Artifact Sanity

### 5.1 目的

P0 确认所有候选仍然是 strict PureKAN graph-free training system，并且实验产物具备可追溯性。任何使用 `loss.backward()`、PyTorch autograd graph、隐藏 ordinary Linear head/stem/LN 参数的候选只能作为 reference，不能进入主线。

### 5.2 必跑对象

```text
MLP-autograd-reference
MLP-manual-linear-reference
DWM2-poly2-compiled-current
DWM2-poly2-compiled-memoryOptimized
DWM2-poly2-compiled+LightSmooth-hook
DWM2-poly2-compiled+FunctionalCorrection-hook
DWM2-poly3-reference
DWM2-poly2-silu-base-reference
```

### 5.3 必须记录指标

```text
method_id
primitive_family
implementation_version
edge_param_count
base_param_count
residual_param_count
mixing_param_count
nonKAN_param_count
loss_backward_used
torch_autograd_graph_used
manual_forward_available
manual_backward_available
manual_update_available
gradient_correctness_available
rollback_max_error
edge_coverage
base_coverage
residual_coverage
mixing_coverage
manual_cache_MB
optimizer_state_MB
parameter_MB
uses_fake_data
wandb_run_id
artifact_hash
source_commit
```

### 5.4 通过标准

Graph-free candidate 必须满足：

```text
nonKAN_param_count = 0
loss_backward_used = 0
torch_autograd_graph_used = 0
manual_forward_available = 1
manual_backward_available = 1
manual_update_available = 1
rollback_max_error < 1e-8
edge_coverage = 1
uses_fake_data = 0
```

如果任何主线候选不满足这些条件，则不得进入 P1。

### 5.5 可视化

必须画 **implementation contract heatmap**。横轴是 candidate，纵轴是 invariant，包括 `nonKAN=0`、`no loss.backward`、`manual backward`、`rollback`、`coverage`、`no fake data`。绿色代表通过，红色代表失败。

---

## 6. P1：DWM2-poly2 Memory Finalization and Phase-Local Memory Audit

### 6.1 目的

P1 的唯一核心目标是把 `DWM2-poly2-compiled` 的 backward memory ratio 从 $1.144985$ 压到 $<1.0$。本阶段不做任务成功结论，只做 efficiency、memory、gradient correctness 和 cache attribution。

### 6.2 方法包

```text
P1-M0-current:
  当前 DWM2-poly2-compiled

P1-M1-cacheCompressed-bf16:
  manual cache 中的 x / hidden / intermediate 使用 bf16 保存，backward 时转回 fp32 或混合精度计算

P1-M2-noHiddenCache:
  不保存 hidden cache，manual backward 时从 input 和 parameters 重算必要 hidden

P1-M3-deltaStreaming:
  不保留完整 delta buffer，逐层 streaming consume delta

P1-M4-bufferReuse:
  forward / backward / update 复用 temporary buffer，避免重复 allocation

P1-M5-noTemp-poly2:
  不显式保存 x^2 / polynomial features，backward 中重算

P1-M6-fusedForwardBackwardCachePolicy:
  forward cache 与 backward adjoint 使用统一 cache policy，删除重复存储

P1-M7-allMemoryOptimized:
  M1 + M2 + M3 + M4 + M5 的组合
```

### 6.3 数据配置

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

batch_sizes:
  128
  256
  512

hidden_dim:
  64

depth:
  2
  4

warmup_steps:
  50

measure_steps:
  200
```

### 6.4 必须记录指标

P1 必须同时记录全局 peak 与 phase-local peak。全局指标包括：

```text
forward_time_ms
manual_backward_time_ms
update_time_ms
step_time_ms
forward_time_ratio_vs_MLP
backward_time_ratio_vs_MLP
step_time_ratio_vs_MLP
peak_allocated_MB
peak_reserved_MB
backward_memory_ratio_vs_MLP
manual_cache_MB
optimizer_state_MB
parameter_MB
workspace_temp_MB
kernel_count_forward
kernel_count_backward
gemm_count
elementwise_kernel_count
poly_eval_kernel_count
buffer_reuse_count
dtype_cache
grad_relerr
grad_cos
```

phase-local memory 必须记录：

```text
peak_forward_MB
peak_loss_delta_MB
peak_backward_adjoint_MB
peak_update_MB
peak_optimizer_state_MB
peak_total_step_MB
```

cache decomposition 必须记录：

```text
cache_x_MB
cache_hidden_MB
cache_delta_MB
cache_temp_MB
cache_poly_MB
cache_index_MB
cache_misc_MB
```

### 6.5 判断标准

Exploratory pass:

$$
\frac{T_{step}}{T_{MLP}}\leq1.20,
$$

$$
\frac{M_{backward}}{M_{MLP}}\leq1.05.
$$

Memory pass:

$$
\frac{T_{step}}{T_{MLP}}\leq1.10,
$$

$$
\frac{M_{backward}}{M_{MLP}}<1.00.
$$

Final memory target:

$$
\frac{M_{backward}}{M_{MLP}}\leq0.80.
$$

Gradient correctness 必须满足：

$$
\operatorname{relerr}(\nabla)<10^{-4},
$$

$$
\cos(\nabla_{manual},\nabla_{reference})>0.999.
$$

如果某 memory optimization 通过 memory gate 但 gradient correctness 失败，则视为 invalid，不可进入 P2。

### 6.6 可视化

P1 必须生成以下图：

1. **Memory decomposition stacked bar**：分解 `parameter_MB`、`optimizer_state_MB`、`cache_x_MB`、`cache_hidden_MB`、`cache_delta_MB`、`cache_poly_MB`、`workspace_temp_MB`。
2. **Phase-local peak memory waterfall**：展示 forward、loss delta、backward adjoint、update、optimizer state 的局部 peak。
3. **Step-time waterfall**：分解 forward、manual backward、update、optimizer-state update 的耗时。
4. **Memory-time Pareto**：横轴 backward memory ratio，纵轴 step time ratio，点大小表示 gradient relerr。
5. **Batch-scaling curve**：横轴 batch size，纵轴 memory ratio 与 step ratio。
6. **Cache source Pareto**：横轴 cache component MB，纵轴 contribution to peak memory。

---

## 7. P2：DWM2-poly2 Accelerated Task Recipe Search

### 7.1 目的

在 P1 survivor 上验证任务学习能力和 wall-clock 收敛速度。P2 不加入 functional correction，只比较 task learner。P2 的目标不是最终 confirm，而是筛选出进入 P3/P4 的 task backbone。

### 7.2 候选优化器

```text
T0-ManualAdamW
T1-ManualAdam
T2-ManualAdanLite
T3-ManualWinLite
T4-Lookahead-ManualAdamW
T5-WarmupCosine-ManualAdamW
T6-ManualAdamW-beta2low
T7-ManualAdamW-gradClip
T8-ManualAdanLite-restart
T9-ManualWinLite-restart
T10-ManualAdanLite-betaSchedule
T11-ManualAdanLite-gradClip
```

不再主线运行：

```text
InputHeavy role-wise LR
OutputWarmup role-wise LR
hard role LR schedule
```

如果需要 role-wise variant，只允许：

```text
same lr + role-wise beta2
same lr + role-wise grad norm clipping
same lr + role-wise update normalization
```

### 7.3 数据与预算

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

short_budget:
  20 epochs equivalent

full_budget_for_survivors:
  30 epochs equivalent
```

P2 选择只允许使用 train/val/profiler 指标。test set 可以记录用于 sanity，但不得用于调参。P6 以后才允许使用 test 指标做 confirm 报告。

### 7.4 必须记录指标

任务指标：

```text
train_loss
val_loss
train_acc
val_acc
test_acc_sanity
val_loss_auc_by_step
val_loss_auc_by_time
val_acc_auc_by_step
val_acc_auc_by_time
ECE
NLL
margin_mean
margin_p10
classwise_val_acc
```

收敛指标：

```text
time_to_target_loss
steps_to_target_loss
time_to_target_acc
steps_to_target_acc
early_loss_slope_step
early_loss_slope_time
mid_loss_slope_step
mid_loss_slope_time
loss_spike_count
restart_count
restart_reason
```

效率指标：

```text
manual_step_time_ms
forward_time_ms
backward_time_ms
update_time_ms
total_wall_clock_sec
samples_per_second
peak_memory_MB
backward_memory_ratio
step_time_ratio
```

optimizer dynamics：

```text
optimizer_state_norm_m
optimizer_state_norm_v
update_norm_input
update_norm_block
update_norm_output
update_over_param_input
update_over_param_block
update_over_param_output
role_update_share_input
role_update_share_block
role_update_share_output
gradient_norm_by_role
cos_update_grad
cos_update_previous
```

representation 指标：

```text
feature_effective_rank_input
feature_effective_rank_block
feature_effective_rank_output
class_centroid_separation
feature_norm_mean
feature_norm_p95
```

### 7.5 判断标准

P2 使用三层 gate，避免过早杀掉 near-miss。

Exploratory pass:

$$
\operatorname{Acc}_{candidate}\geq\operatorname{Acc}_{MLP}-0.015
$$

on all datasets，并且没有任何 dataset 出现 catastrophic failure。

Selection pass:

$$
\operatorname{Acc}_{candidate}\geq\operatorname{Acc}_{MLP}-0.005
$$

on at least two datasets。

Convergence pass 至少满足一个：

$$
T_{target-loss,candidate}<T_{target-loss,MLP},
$$

或：

$$
\operatorname{ValLossAUC}_{time,candidate}<\operatorname{ValLossAUC}_{time,MLP}.
$$

Efficiency 必须满足：

$$
\frac{T_{step}}{T_{MLP}}\leq1.20,
$$

$$
\frac{M_{backward}}{M_{MLP}}<1.05.
$$

只有同时满足 Selection pass 与 Efficiency pass 的方法才能进入 P4 稳定性修复；Exploratory pass 只能进入 diagnostic，不进入 confirm。

### 7.6 可视化

P2 必须生成：

1. **Validation loss vs step** 与 **validation loss vs wall-clock**。两张图必须同时存在。
2. **Accuracy vs wall-clock**。
3. **Time-to-target bar chart**：分别画 target loss 与 target accuracy。
4. **Task-efficiency Pareto**：横轴 step time ratio，纵轴 validation/test accuracy，点大小为 backward memory ratio。
5. **Role update share trajectory**：input/block/output update share 随 step 变化。
6. **Optimizer dynamics trace**：momentum norm、RMS norm、cos_update_grad、restart marker。
7. **Dataset failure heatmap**：dataset × optimizer，显示 task fail、time fail、memory fail。

---

## 8. P3：Functional Correction Direction-Quality Audit

### 8.1 目的

P3 验证 functional update 是否可以作为 **correction**，而不是主 direction。P3 不是完整训练，而是 direction-quality audit 和 one-step probe。只有 P3 过，functional correction 才能进入 P5 训练集成。

组合方向为：

$$
d_{new}=d_{task}+\lambda d_{geo}.
$$

$d_{task}$ 来自 P2 survivor，例如 `ManualAdanLite`、`ManualAdamW` 或 `ManualWinLite`。$d_{geo}$ 来自 geometry / calibration correction。

### 8.2 候选 correction

```text
C0-task-only
C1-task-plus-smallSob-lambda0.02
C2-task-plus-smallSob-lambda0.05
C3-task-plus-LightSmooth-lambda0.02
C4-task-plus-LightSmooth-lambda0.05
C5-task-plus-curvature-lambda0.02
C6-task-plus-curvature-lambda0.05
C7-task-plus-ECE-lambda0.02
C8-adaptive-lambda-descent-safe
C9-adaptive-lambda-geometry-first-safe
```

### 8.3 Direction audit batch

对每个 dataset / seed / task learner / correction method，抽取：

```text
train batch
holdout batch
validation mini-batch
```

分别评估 candidate direction。所有 direction probe 必须在 cloned weights 上执行，不能污染真实训练权重。

### 8.4 必须记录指标

Direction 质量：

```text
cos_new_task
projection_on_task
norm_new_over_task
angle_new_task_degree
holdout_descent_task
holdout_descent_new
holdout_descent_ratio
train_descent_task
train_descent_new
train_descent_ratio
bad_step_task
bad_step_new
bad_step_rate_new
```

真实 one-step probe：

```text
actual_train_loss_before
actual_train_loss_after
actual_holdout_loss_before
actual_holdout_loss_after
actual_val_loss_before
actual_val_loss_after
rollback_error_after_probe
```

Geometry / calibration：

```text
phi_change_task
phi_change_new
curvature_change_task
curvature_change_new
ECE_change_proxy
NLL_change_proxy
logit_drift
feature_drift
margin_change
```

Adaptive lambda：

```text
initial_lambda
accepted_lambda
lambda_backtrack_count
fallback_to_task_only
lambda_zero_rate
```

### 8.5 判断标准

Functional correction survives P3 only if:

$$
\cos(d_{new},d_{task})\geq0.85,
$$

$$
\frac{\operatorname{HoldoutDescent}(d_{new})}{\operatorname{HoldoutDescent}(d_{task})}\geq0.95,
$$

$$
\operatorname{BadStepRate}(d_{new})\leq0.02,
$$

and rollback is exact:

$$
\operatorname{RollbackError}<10^{-8}.
$$

It must also improve at least one non-task metric:

$$
\Delta\phi'_{p95}(d_{new})<\Delta\phi'_{p95}(d_{task}),
$$

or:

$$
\Delta\operatorname{Curvature}(d_{new})<\Delta\operatorname{Curvature}(d_{task}),
$$

or:

$$
\Delta\operatorname{ECE}(d_{new})<\Delta\operatorname{ECE}(d_{task}).
$$

If all corrections fail, v6.4 decision is:

```text
functional correction not ready;
use graph-free accelerated task learner only;
do not enter geometry-maintenance confirm.
```

### 8.6 可视化

P3 必须生成：

1. **Direction-quality scatter**：横轴 `cos_new_task`，纵轴 `holdout_descent_ratio`，颜色表示 geometry improvement。
2. **Geometry gain vs task cost Pareto**：横轴 `1 - holdout_descent_ratio`，纵轴 geometry reduction。
3. **Lambda acceptance histogram**：检查 adaptive lambda 是否总是被压到 0。
4. **Bad-step heatmap**：dataset × correction method。
5. **Predicted vs actual descent scatter**：横轴 predicted descent，纵轴 actual holdout descent。
6. **Rollback error table**：确保 probe 不污染权重。

---

## 9. P4：All-Dataset Acceleration Stability Repair

### 9.1 目的

P4 专门修复 v6.3 中 `A2-ManualAdanLite` 只有 partial winner 的问题。P4 不引入 functional correction，只验证 task learner 稳定性。

### 9.2 候选

```text
A0-ManualAdamW
A2-ManualAdanLite
A2b-ManualAdanLite-betaSchedule
A2c-ManualAdanLite-gradClip
A2d-ManualAdanLite-restart-on-plateau
A3-ManualWinLite
A4-Lookahead-ManualAdamW
A6-WarmupCosine-ManualAdamW
```

删除主线：

```text
A7 RoleWiseLR InputHeavy
A8 RoleWiseLR OutputWarmup
```

### 9.3 数据与 seed

```text
datasets:
  MNIST
  Fashion-MNIST
  KMNIST

seeds:
  0,1,2

budget:
  same as P2 full_budget_for_survivors
```

### 9.4 必须记录指标

P4 继承 P2 所有 task、efficiency、convergence 指标，并额外记录：

```text
seedwise_pass
seedwise_failure_reason
dataset_pass_count
all_dataset_winner
instability_code
loss_spike_epoch
restart_effective
best_epoch_by_val_loss
final_epoch_gap
```

### 9.5 判断标准

P4 candidate becomes all-dataset winner if:

$$
\operatorname{Acc}_{candidate}\geq\operatorname{Acc}_{MLP}-0.005
$$

on all datasets, and:

$$
\operatorname{ValLossAUC}_{time,candidate}\leq\operatorname{ValLossAUC}_{time,MLP}
$$

on at least two datasets, and:

$$
\frac{M_{backward}}{M_{MLP}}<1.0.
$$

If memory is not yet $<1.0$, candidate can be marked:

```text
P4 task-stable but memory-blocked
```

and must return to P1 instead of entering P6 confirm.

### 9.6 可视化

1. **Seedwise accuracy strip plot**：每个 dataset × optimizer 的 seed-level accuracy。
2. **Time-to-target paired bar**：candidate 与 MLP 的 paired comparison。
3. **Instability heatmap**：dataset × seed × optimizer。
4. **Loss spike timeline**：标出 restart / plateau / spike。
5. **P4 winner scorecard**：task、time、memory 三栏 pass/fail。

---

## 10. P5：Empirical Geometry / LightSmooth Integration

### 10.1 目的

P5 是首次真实 empirical LightSmooth / geometry integration audit。不能使用 proxy row，也不能继承旧固定结论。P5 只在 P1 memory pass 且 P4 task-stable 后运行。

### 10.2 训练模式

```text
GTrain0-task-only-baseline
GTrain1-task+correction-every-10-steps
GTrain2-task+correction-geometry-debt-triggered
GTrain3-task+epoch-level-LightSmooth
GTrain4-task+late-phase-correction-only
GTrain5-task+adaptive-lambda-correction
```

默认设置：

```text
correction_interval = 10 steps
late_phase_start = 60% training
geometry_debt_threshold = phi_ratio > 1.10 or curvature_ratio > 1.10
max_lightsmooth_events = 2
```

### 10.3 必须记录指标

任务指标：

```text
accuracy
val_loss_auc_by_time
val_loss_auc_by_step
ECE
NLL
margin_mean
margin_p10
```

geometry 指标：

```text
phi_prime_p95
curvature
jacobian_condition_proxy
geometry_debt
geometry_reduction_vs_task_only
```

event 指标：

```text
correction_event_count
accepted_correction_count
rejected_correction_count
smooth_event_count
accepted_smooth_count
avg_lambda
fallback_rate
correction_overhead_ms
amortized_step_time_ms
backward_memory_ratio
logit_drift
feature_drift
KL_before_after
```

### 10.4 判断标准

Correction / LightSmooth integration survives if:

$$
\operatorname{Acc}_{corrected}\geq\operatorname{Acc}_{task-only}-0.005,
$$

$$
\operatorname{ValLossAUC}_{time,corrected}\leq\operatorname{ValLossAUC}_{time,task-only}+0.02,
$$

and at least one geometry metric improves by $>5\%$:

$$
\frac{\operatorname{Curvature}_{corrected}}{\operatorname{Curvature}_{task-only}}<0.95,
$$

or:

$$
\frac{\phi'_{p95,corrected}}{\phi'_{p95,task-only}}<0.95.
$$

Overhead must satisfy:

$$
\frac{T_{step+correction}}{T_{task-only}}\leq1.10.
$$

### 10.5 可视化

1. **Training curve with correction events**：val loss、accuracy、phi、curvature，标出 event。
2. **Geometry-debt curve**：检查 geometry debt 是否真的下降。
3. **Event overhead plot**：每个 event 的 wall-clock overhead。
4. **Task-only vs corrected Pareto**：横轴 accuracy，纵轴 geometry reduction。
5. **Logit drift vs geometry reduction scatter**。
6. **ECE curve with correction events**。

---

## 11. P6：3-Seed Full-Budget Candidate Selection

### 11.1 目的

P6 验证候选不是短程或 seed0 偶然。P6 是第一个允许正式报告 test set 的 selection 阶段，但 P6 后不得根据 test set 结果回调超参。

### 11.2 方法

```text
seeds:
  0,1,2

datasets:
  MNIST
  Fashion-MNIST
  KMNIST

methods:
  MLP-autograd-reference
  MLP-manual-linear-reference
  DWM2-poly2-task-only-best
  DWM2-poly2-task+correction-best
  DWM2-poly2-task+LightSmooth-best, optional
```

### 11.3 必须记录指标

P6 记录 P5 所有指标，并额外记录 paired delta：

```text
paired_acc_delta_vs_MLP
paired_val_loss_auc_time_delta_vs_MLP
paired_memory_delta_vs_MLP
paired_ECE_delta_vs_MLP
paired_geometry_delta_vs_MLP
paired_time_to_target_delta_vs_MLP
seedwise_failure_reason
```

### 11.4 通过标准

P6 candidate survives if:

1. at least two datasets accuracy $\geq$ MLP;
2. no dataset accuracy below MLP by more than $1\%$;
3. all datasets backward memory ratio $<1.0$;
4. at least two datasets time-to-target faster than MLP;
5. correction version must improve geometry or ECE compared with task-only version.

If task-only passes but correction fails, route decision is:

```text
DWM2-poly2 graph-free task learner survives;
functional correction remains future work.
```

---

## 12. P7：5-Seed Confirm

P7 只跑 P6 survivor。

### 12.1 设置

```text
seeds:
  0,1,2,3,4

methods:
  MLP-autograd-reference
  MLP-manual-linear-reference
  best DWM2-poly2 task-only
  best DWM2-poly2 task+correction, if P6 passed
```

### 12.2 必须记录统计

```text
paired_delta_mean
paired_delta_std
bootstrap_CI_low
bootstrap_CI_high
seedwise_failure_reason
```

这些字段至少要覆盖：

```text
accuracy
ValLossAUC_time
ECE
backward_memory_ratio
time_to_target_loss
time_to_target_acc
phi_prime_p95
curvature
```

### 12.3 成功标准

```text
paired acc delta vs MLP >= 0 on at least two datasets
paired ValLossAUC-time delta vs MLP <= 0 on at least two datasets
backward memory ratio < 1.0 on all datasets
no catastrophic seed
ECE or geometry improves on at least two datasets
```

如果 P7 失败，必须输出 route case：

```text
A: efficiency pass, task fail
B: task pass, memory fail
C: task + efficiency pass, geometry fail
D: seed instability
E: all fail
```

---

## 13. P8：10-Seed Final Confirm

P8 是最终确认，不允许再调参。

### 13.1 方法

```text
seeds:
  0..9

methods:
  MLP-autograd-reference
  MLP-manual-linear-reference
  Final-DWM2-poly2-task-only
  Final-DWM2-poly2-task+functional-correction, if passed
```

### 13.2 必须输出 artifact

```text
final_scorecard.csv
paired_delta_vs_mlp.csv
time_to_target_summary.csv
memory_summary.csv
geometry_summary.csv
calibration_summary.csv
failure_table.csv
route_decision.json
aggregate_decision.json
```

### 13.3 最终 claim gate

只有同时满足下面条件，才可以写：

```text
PureKAN graph-free functional training system is established.
```

Accuracy:

$$
\operatorname{Acc}_{KAN}\geq\operatorname{Acc}_{MLP}
$$

on at least two datasets, and no dataset gap worse than $1\%$.

Memory:

$$
\frac{M_{backward,KAN}}{M_{backward,MLP}}<1.0
$$

on all datasets.

Convergence:

$$
T_{target,KAN}\leq T_{target,MLP}
$$

on at least two datasets.

Calibration or geometry:

$$
\operatorname{ECE}_{KAN}<\operatorname{ECE}_{MLP}
$$

or:

$$
\phi'_{p95,KAN}<\phi'_{p95,MLP}.
$$

If candidate reaches memory $<1.0$ but not $\leq0.8$，最终只能写：

```text
near-final graph-free PureKAN system;
terminal memory target remains open.
```

不能写 backward memory 终极目标已达成。

---

## 14. P9：Fallback Primitive Decision

如果 DWM2-poly2 无法同时过 memory + task，P9 决定是否切换 primitive。

### 14.1 fallback options

```text
F0-DWM2-poly3
F1-DWM2-poly2+silu-base
F2-DWM2-gated-poly2
F3-DWM2-RBFK2-small-correction
F4-RationalKAT-lite-fastpoly
F5-SparseInterp fused rewrite
```

### 14.2 decision logic

如果：

```text
DWM2-poly2 memory pass, task fail
```

则优先试：

```text
DWM2-poly3
DWM2-poly2+silu-base
DWM2-gated-poly2
```

如果：

```text
DWM2-poly2 task pass, memory fail
```

则优先继续：

```text
cacheCompressed-bf16
deltaStreaming
noHiddenCache
bufferReuse
```

如果：

```text
DWM2-poly2 time fail
```

则重新评估：

```text
RationalKAT-lite
SparseInterp fused kernel
```

如果：

```text
DWM2-poly2 task + memory both fail
```

则 route decision 必须写明：

```text
DWM2-poly2 cannot be terminal primitive under current implementation.
```

---

## 15. P10：Failure Diagnosis and Route Decision

v6.4 最后必须输出清晰 route decision，而不是只说失败。

### 15.1 route cases

```text
R1:
  DWM2-poly2 passes memory and task, correction fails.
  Decision: use DWM2-poly2 accelerated graph-free task learner; functional geometry remains future work.

R2:
  DWM2-poly2 passes memory, task, and correction.
  Decision: proceed to 5/10-seed final confirm or declare final if confirm already passed.

R3:
  DWM2-poly2 passes task but memory remains > 1.0.
  Decision: focus on memory kernel; do not run more confirm seeds.

R4:
  DWM2-poly2 passes memory but task fails.
  Decision: increase expressivity with poly3 / silu-base / gated-poly2.

R5:
  DWM2-poly2 fails both task and memory.
  Decision: abandon DWM2-poly2 as final primitive; shift to RationalKAT or SparseInterp.

R6:
  No graph-free primitive can meet memory/time.
  Decision: current PureKAN primitive family cannot satisfy final efficiency target under current implementation.
```

### 15.2 failure taxonomy

必须统计：

```text
F1_memory_fail
F2_step_time_fail
F3_gradient_correctness_fail
F4_task_underfit
F5_convergence_slow
F6_acceleration_unstable
F7_correction_bad_direction
F8_geometry_no_gain
F9_calibration_worse
F10_seed_instability
F11_gated_not_run
F12_artifact_integrity_fail
```

---

## 16. 本轮必须生成的 artifact

```text
p0_contract.csv
p0_contract_heatmap.svg
p1_memory_finalization.csv
p1_memory_decomposition.csv
p1_phase_local_memory.csv
p1_runtime_decomposition.csv
p1_gradient_correctness.csv
p2_task_recipe.csv
p2_task_trace.csv
p2_time_to_target.csv
p2_optimizer_dynamics.csv
p3_functional_correction_direction.csv
p3_one_step_probe.csv
p3_direction_gate_summary.csv
p4_acceleration_stability.csv
p4_seedwise_trace.csv
p5_geometry_integration.csv
p5_event_trace.csv
p6_confirm3.csv
p7_confirm5.csv
p8_confirm10.csv
failure_table.csv
route_decision.json
aggregate_decision.json
figures/
```

---

## 17. 必须画的图

### 17.1 Efficiency dashboard

展示每个候选的：

```text
forward ratio
backward ratio
step ratio
backward memory ratio
```

目标是直观看出 DWM2-poly2 是否真的进入 MLP-like 区域。

### 17.2 Memory decomposition stacked bar

每个候选分解：

```text
parameters
optimizer state
manual cache
x cache
hidden cache
delta cache
poly temp cache
workspace temp
```

这张图决定 P1 的方向。

### 17.3 Phase-local peak memory waterfall

每个候选必须画：

```text
forward peak
loss-delta peak
backward-adjoint peak
update peak
optimizer-state peak
total-step peak
```

用来定位 `bmem_ratio` 到底来自哪个阶段。

### 17.4 Loss by step and by wall-clock

必须同时画：

```text
val loss vs step
val loss vs seconds
```

否则不能 claim 收敛快。

### 17.5 Time-to-target plot

目标 loss 与目标 acc 两张 bar chart：

```text
method vs time_to_target_loss
method vs time_to_target_acc
```

### 17.6 Direction-quality scatter

横轴：

$$
\cos(d_{new},d_{task})
$$

纵轴：

$$
\frac{\operatorname{HoldoutDescent}(d_{new})}{\operatorname{HoldoutDescent}(d_{task})}
$$

颜色表示 geometry improvement。

### 17.7 Predicted vs actual descent scatter

横轴是 predicted descent，纵轴是 actual holdout descent。用于发现 functional correction 的有限步长偏差。

### 17.8 Geometry integration curve

画：

```text
epoch vs phi_prime_p95
epoch vs curvature
epoch vs ECE
```

并标出 correction / LightSmooth event。

### 17.9 Task-efficiency-geometry Pareto

三张图：

```text
accuracy vs step_time_ratio
accuracy vs backward_memory_ratio
geometry_reduction vs accuracy
```

### 17.10 Seedwise paired delta plot

P6-P8 必须画：

```text
paired accuracy delta
paired ValLossAUC-time delta
paired memory delta
paired ECE delta
paired geometry delta
```

### 17.11 Failure taxonomy heatmap

横轴是方法，纵轴是 failure type：

```text
memory_fail
time_fail
task_fail
correction_fail
geometry_fail
calibration_fail
seed_instability
artifact_integrity_fail
```

---

## 18. 实验纪律与防泄漏规则

1. P0-P5 只允许使用 train/val/profiler 指标做选择。test 指标只能作为 sanity 记录，不能用于调参。
2. P6 可以报告 test set，但 P6 后不能根据 test result 继续改 recipe。
3. P7/P8 是 confirm，不允许再调超参。
4. 每次实验必须记录 W&B run、local CSV、run.log、artifact hash、source commit。
5. 不允许 `--allow-fake-data`。
6. 不允许 fixed proxy row 写成 pass。
7. `not_run` 必须保持 `not_run`，不能在 aggregate summary 中被当作 0 或 pass。
8. 所有 formula 中的 AUC 必须明确是 `ValLossAUC` 还是 `ValAccAUC`。

---

## 19. 最终建议

v6.4 的一句话方向是：

$$
\boxed{\text{先把 DWM2-poly2-compiled 推成真正 memory-pass 的 graph-free task learner，}}
$$

$$
\boxed{\text{再把 functional update 改成 task-preserving geometry correction。}}
$$

不要再尝试让 functional update 单独替代 task learner。当前最合理的结构是：

$$
\boxed{\Delta\theta=\Delta\theta_{accelerated\ task}+\lambda_t\Delta\theta_{functional\ geometry\ correction}.}
$$

如果 v6.4 成功，我们就能从：

```text
graph-free primitive partial candidate
```

推进到：

```text
graph-free memory-pass PureKAN training system candidate
```

如果 v6.4 失败，也能明确知道失败来自：

```text
memory limit
task expressivity limit
acceleration instability
functional correction quality limit
```

中的哪一个，而不是继续陷入“大方向是否正确”的不确定性。
