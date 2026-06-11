# DG-KAN v22.07：Metric-as-Dynamics Functional Update + Basis Efficiency Closure + 4GPU 完整实验计划

> 版本：v22.07 execution plan  
> 目标读者：不假设读者熟悉全部历史实验；本文开头说明项目目标、v22.06 真实状态和为什么必须换 Functional Update 思路。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler/class weight；no dataset-name branch；no seed-specific rule；no fake/proxy/CPU offload；functional direction 不使用 validation/test/future/query；LineC / CEp99 / NLL / ECE / Brier / AUCtime 只能作为 audit、debt readback 和 gate，不能生成 update direction。  
> 本版核心修正：Functional Update 不再是“metric 名字 + update token”的工程堆叠，而是训练动力系统中的 **source-estimation → function-space target → metric-defined solver → optimizer-state integration → source preservation** 算法链。

---

## 0. 项目总目标与 v22.06 后的真实状态

DG-KAN 的总目标是建立一个 strict FC-PureKAN 训练系统，使：

$$
\boxed{
\text{PureKAN base + functional update}
>
\text{PureKAN base + ordinary backprop / AdamW controls}
}
$$

这里“更好”不是某个 horizon 局部 source 为正，也不是某个 ActuationR2 很高，而是同时满足：

```text
1. carrier 的 forward / backward / step / memory 接近 same-param MLP；
2. functional update 的收益不能被 AdamW、SGD、NoOp、random、stable-random、same-overhead controls 解释；
3. source 必须从早期形成，并在 h3200 / h4800 后仍保留；
4. tail / LineC / ECE / Brier / NLL / AUCtime debt 不能在 terminal phase 爆发；
5. 如果只在 MLP 成功，只能写成 generic training-dynamics insight，不能写成 KAN-specific；
6. 只有 KAN 在同机制下超过 MLP / controls，才允许讨论 KAN-specific functional advantage。
```

v22.06 的真实状态如下。

```text
Code:
  核心 required path 基本自包含，compile/import/source-chain/LineC/terminal-retention/metric-solver tests 通过。
  但 entire historical repo 仍不应混入 CodeRoute；CodeRoute 只能声明 v22.06 required path pass。

Efficiency:
  D-CHE / D-FOU 已经是稳定主 carrier。
  D-CHE S1 pass = 1；D-FOU S1 pass = 1。
  D-RAT / D-RBF 出现单 batch production fused pass，但 multibatch probe 仍显示 OfficialFusedBlocked。
  因此 D-RAT / D-RBF 不能写成 robust official closure，只能写成 single-path production evidence + multibatch stability blocker。

Functional:
  v22.06 route = C3-ActuationOnly。
  C1 observability rows 多，C3 actuation rows 多，但 C4 early-source / h3200-source rows 为 0。
  best h3200 source 来自 MLP-V2206-C4-T10G0-OptTransport-lr150，source_h3200 > 0，但 source_h800 < 0，所以不是 retained source，而是 late/local recovery。
  Terminal preservation 与 KAN mapping 正确 fail-closed，没有进入。
```

v22.06 最重要的结论不是“metric 没用”，而是：

$$
\boxed{
\text{我们终于从 flat-gradient metric proxy 进入了 readout-level metric solver，}
\text{但 solver 能 actuation，仍不能形成 early retained source。}
}
$$

这说明 Functional Update 的核心 blocker 已经从“参数能不能动”变成：

$$
\boxed{
\text{什么 train-only signal 才值得被写入函数空间，并能在训练动力学中保留？}
}
$$

---

## 1. 第一性原则：Functional Update 到底应该是什么

普通训练每一步做：

$$
\theta_{t+1} = \theta_t + u_{\mathrm{opt},t},
$$

其中 $u_{\mathrm{opt},t}$ 来自 AdamW / SGD / Momentum 等 optimizer。

Functional Update 不应该被理解为：

```text
在 AdamW 后面加一个小 residual；
或每隔几步加一个启发式方向；
或把某个 metric 名字包装成 update token。
```

Functional Update 应该被理解为训练动力系统里的外部控制 / 边界条件：

$$
\theta_{t+1}
=
\theta_t
+
u_{\mathrm{opt},t}
+
u_{\mathrm{FU},t},
$$

其中 $u_{\mathrm{FU},t}$ 必须满足：

```text
1. 来自 train-stream 可观测信号；
2. 在函数空间中定义目标 Δf，而不是只在 flat 参数空间里过滤梯度；
3. 通过 metric-defined solver 找到可执行参数更新；
4. 与 optimizer state 协同，不被后续训练洗掉；
5. 被 LineC / tail / ECE / Brier / AUCtime 事后审计，但这些 audit metric 不生成方向。
```

形式上，我们要从：

$$
g \rightarrow \mathrm{filter}(g)
$$

升级为：

$$
\boxed{
 u^\star
=
\arg\min_u
\|J_\theta u - \Delta f_{\mathrm{target}}\|_{G_f}^2
+
\rho\|u\|_P^2
}
$$

这里：

```text
J_theta u:
  参数更新造成的函数变化。

Δf_target:
  train-stream 可得的目标函数变化。

G_f:
  函数空间 metric，例如 L2、Fisher、Sobolev、RKHS、basis-channel、low-NDS。

P:
  参数空间 / block / optimizer-state regularizer。
```

v22.06 已经开始做 readout exact solve，但仍不够。因为：

```text
1. S1 readout solve 能实现 target，但 target 本身不产生 early source；
2. S2 hidden residual 能打开一些 source，但会破坏 actuation / transfer evidence；
3. T8/T9/T10 把 hidden residual 管住后，source 又消失；
4. 因此当前问题不是 solver 名字不够，而是 source target、metric constraint 和 optimizer integration 没形成闭环。
```

---

## 2. 从训练研究中抽取的算法原则，而不是机械换 optimizer

### 2.1 Muon / NDS：长期保留取决于曲率代价，不只是 source amplitude

Muon curvature 研究提示：一阶 gain 相近时，长期差异可能来自二阶 curvature penalty 和 Normalized Directional Sharpness。对我们来说，h3200 source 能正但 h4800 / h3200 不足，可能意味着 update 的一阶收益可见，但二阶曲率代价太高。

因此 v22.07 必须记录：

$$
I^{(1)} = \langle g, u\rangle,
$$

$$
I^{(2)} \approx \frac{1}{2} u^T H u,
$$

以及：

$$
\mathrm{NDS}(u)
= \frac{u^T H u}{\|u\|^2 + \epsilon}.
$$

判断方式不是“用 Muon 替代 AdamW”，而是：

```text
如果 early source 成功但 terminal erosion 高，并且 NDS 高，下一步做 low-NDS projection / block-balanced update。
如果 NDS 低但 source 仍掉，问题不是 curvature，而是 target / debt / dataset heterogeneity。
```

### 2.2 Nora / row-wise angular stability：source preservation 要区分径向和角向运动

Nora 的启发不是“换 Nora optimizer”，而是矩阵参数有两种运动：

```text
radial movement:
  改变 row norm。

angular movement:
  改变 row direction。
```

对 hidden/readout matrix、D-CHE degree-readout、D-FOU band-readout 来说，source 可能主要存在于角向子空间；后期 optimizer 的径向 jitter 或角速度漂移可能侵蚀 source。

v22.07 要记录：

```text
row_norm_drift
row_angular_velocity
radial_update_fraction
orthogonal_update_fraction
source_projection_radial
source_projection_angular
```

并测试真正的 row-orthogonal source-preserving update，而不是只把 mechanism 命名为 Nora。

### 2.3 SOAP / block coordinates：metric 必须放在稳定 block coordinate 中

SOAP 的启发是，Adam-like moment 可以在 slowly-changing preconditioner eigenbasis 中运行。对我们来说，metric 不应该只定义在 flat 参数向量上，而应该定义在 block 坐标中：

```text
MLP:
  hidden matrix block；readout matrix block。

D-CHE:
  low-degree bank；high-degree reservoir；degree-readout block。

D-FOU:
  low-frequency bank；high-frequency reservoir；band-readout block。

D-RAT:
  numerator tangent；denominator tangent；num/den joint block。

D-RBF:
  local-center block；width block；readout block。
```

v22.07 的 metric solver 必须输出 block-level energy 和 block-level source attribution，不再只输出 global update norm。

### 2.4 Schedule-Free / AdEMAMix：source formation 和 source preservation 是不同时间尺度

Schedule-free / averaging 的启发是 fast iterate 和 slow iterate 角色不同；AdEMAMix 的启发是近期梯度和旧梯度都可能重要。对我们来说，source update 至少要分成两个状态：

```text
short source state:
  负责 h100/h400/h800 source formation。

long source state:
  负责 h1600/h3200/h4800 source preservation。
```

v22.07 不再问“一个 update 同时解决所有 horizon”，而是明确分阶段：

```text
Stage F: formation。
Stage C: consolidation。
Stage P: preservation。
```

### 2.5 Signal-channel / reservoir：update 必须写进可迁移通道

generalization theory 的启发是：训练能移动的方向不一定能泛化。真正有用的方向必须进入 signal channel，而不是 reservoir。

v22.07 对每个 update 记录：

```text
signal_channel_projection
reservoir_projection
noise_in_signal_channel
source_in_reservoir
train_to_probe_transfer
```

如果一个 update ActuationR2 高但 source=0，说明它能动，但动在错误通道。v20-v22.06 已经反复出现这种现象，所以 v22.07 不再把 high ActuationR2 当作成功前兆。

### 2.6 Constrained inference manifolds / diffeomorphism：低维 source path 不够，必须保留信息体积

如果 source trajectory 变得低维，但 information volume collapse，说明它可能只是退化压缩，不是健康推理 / 表示流形。

v22.07 必须记录：

```text
intrinsic_dimension
info_volume
condition_number_of_source_subspace
neighbor_preservation
fold_proxy
local_stretch_p95
```

这将用于判断 function update 是否是平滑可逆变形，而不是局部撕裂。

---

## 3. 哪些旧方向可以重开，哪些不能重开

### 3.1 可以重开，但必须换语义

#### Function metric / proximal 线

旧版 function-metric 线失败过，但当时多是 parameter-pullback / proxy / old gate / old carrier。v22.05 的 metric proxy 也失败了。但这不等于 Sobolev / RKHS / Fisher / basis metric 失败。v22.07 允许重开，但必须是：

```text
显式 Jθu function displacement；
显式 Δf_target；
显式 G_f residual；
solver residual / JVP / VJP / CG 记录；
不能只是 flat vector smoothing。
```

#### Basis-natural / operator-level basis-channel 线

v13.2 / v13.3 / v13.4 的 basis-natural 和 operator-level basis-channel 当时受旧 kernel、旧 metric、旧 target 限制。现在 D-CHE / D-FOU 已经高效，D-RAT / D-RBF 也有 production path 线索，因此可以重开：

```text
D-CHE degree-channel metric；
D-FOU frequency-channel metric；
D-RAT joint numerator/denominator tangent metric；
D-RBF local RKHS metric。
```

#### PopRisk / SNR

旧 PopRisk/SNR 作为 one-shot mask 或 parameter SNR 不稳定，但 signal-channel 理论仍然重要。v22.07 允许作为 source estimator / noise-in-signal-channel estimator，不允许直接作为 update mask。

#### Split-consensus

旧 split-consensus 线经常 control-equivalent，但它仍可以作为 source observability estimator。它不能单独提交 update，除非通过 C1/C2/C3 gate。

### 3.2 不应重开为主线

```text
action bank / controller；
reset route；
cover objective 小修；
M31/M32 同族修补；
G/N/Q token 扩展；
只调 scale / floor / lookahead / alt period；
把 high ActuationR2 当 functional success；
把 KAN efficiency pass 当 KAN functional pass。
```

这些方向过去的问题不是简单 bug，而是目标函数和 route 语义不对。它们最多作为 audit/control，不进入主预算。

---

## 4. v22.07 实验总结构

v22.07 分成五条并行线。

```text
Line A: Code / semantic / solver truth gate。
Line B: Basis efficiency full-loop closure。
Line C: Functional Update — source estimator and metric solver。
Line D: Functional Update — source preservation and optimizer-state integration。
Line E: KAN source-channel mapping。
```

所有实验必须使用 4GPU dynamic queue。不得等待单条线跑完再更新计划。

---

## 5. Line A：代码、语义和 solver truth gate

### 5.1 目标

防止再出现：

```text
名字是 metric solver，实际是 flat-gradient smoothing；
名字是 row-orthogonal，实际没有 radial/angular decomposition；
结果包 pass，但 final zip 中缺源码；
continuation 写进复盘，但代码没打包；
```

### 5.2 必须执行

在最终 zip 解压目录中执行：

```text
python -m compileall -q dgkan experiments tests
required_import_closure.py
self_contained_import_check.py
linec_golden_tests.py
source_chain_tests.py
terminal_retention_tests.py
metric_solver_unit_tests.py
semantic_alias_audit.py
kernel_gradcheck.py
profiler_phase_tests.py
```

### 5.3 必须打包文件

Codex 必须生成：

```text
v22_07_code_review_packet.zip
v22_07_results_bundle.zip
```

`v22_07_code_review_packet.zip` 必须包含：

```text
00_README.md
02_SOURCE_TREE/dgkan/**/*.py
02_SOURCE_TREE/experiments/**/*.py
02_SOURCE_TREE/tests/**/*.py
03_IMPORT_CLOSURE/*
04_LINEC_CORRECTNESS/*
05_SOURCE_CHAIN_RETENTION_DEBT/*
06_MECHANISM_SEMANTIC_CONTRACTS/*
07_FUNCTION_SPACE_SOLVER/*
08_BASIS_KERNELS_AND_PROFILERS/*
09_GPU_QUEUE/*
10_RESULTS_POINTERS/*
packet_manifest.csv
packet_sha256_manifest.csv
```

### 5.4 代码 gate

必须满足：

```text
required_source_files = 100%
compileall_returncode = 0
self_contained_import_check = 1
LineC fast/channel golden pass
source-chain tests pass
terminal-retention tests pass
metric-solver tests pass
semantic_alias_undeclared = 0
kernel_gradcheck pass
profiler_phase_tests pass
```

如果不满足：

```text
不能写 scientific no-go；
只能写 CodeAuditBlocked；
Codex 必须先修 packet / import / schema / solver correctness。
```

---

## 6. Line B：Basis efficiency full-loop closure

### 6.1 D-CHE / D-FOU 主 carrier reconfirm

D-CHE / D-FOU 现在是主 functional carrier。v22.07 只做 reconfirm，不再大撒网。

D-CHE variants：

```text
CHE21-R2-low-degree-k3-official
CHE21-R4-k5-gradbuf-triton-ablation
```

D-FOU variants：

```text
FOU21-R3-tablelookup-bandreadout-official
```

记录：

```text
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
functional_direction_ms
metric_solver_ms
LineC_audit_ms
horizon_readback_ms
same_kernel_functional_runner_proof
fallback_kernel_used
official_fused_kernel_complete
no_materialize_complete
```

通过标准：

```text
forward <= 1.25
step <= 1.25
memory <= 1.05
same_kernel_functional_runner_proof = 1
fallback_kernel_used = 0
```

### 6.2 D-RAT / D-RBF multibatch officialization

v22.06 出现一个口径问题：single batch production fused pass 了，但 multibatch probe 仍 blocked。因此 v22.07 必须把 D-RAT / D-RBF 从 “any production pass” 升级到 **multi-batch robust production pass**。

D-RAT 必须测：

```text
batch = 128, 256, 512, 1024
forward_ratio
backward_ratio
step_ratio
memory_ratio
numerator_eval_ms
denominator_eval_ms
reciprocal_ms
safety_guard_ms
telemetry_ms
rational_den_min
rational_den_p01
gradcheck_relerr
same_kernel_runner_proof
```

D-RBF 必须测：

```text
batch = 128, 256, 512, 1024
forward_ratio
backward_ratio
step_ratio
memory_ratio
active_center_fraction
mean_local_K
local_gather_ms
exp_eval_ms
readout_matmul_ms
local_backward_ms
dense_materialized_bytes
same_kernel_runner_proof
```

D-RAT / D-RBF robust official gate：

```text
>= 3 / 4 batch sizes pass
forward <= 1.50
step <= 1.50
memory <= 1.10
gradcheck_pass = 1
same_kernel_runner_proof = 1
```

如果失败，Codex 先尝试：

```text
D-RAT:
  separate numerator/denominator/reciprocal micro waterfall；
  telemetry-free training path；
  reciprocal approximation vs torch reciprocal；
  denominator safety separated from training timing。

D-RBF:
  active-center threshold repair；
  local K reduction；
  exp approximation；
  no dense materialization proof；
  local backward fused proof。
```

### 6.3 D-RAT / D-RBF functional smoke only after robust near-E1

D-RAT / D-RBF 只有满足：

```text
near_E1_rows >= 3 batch sizes
forward <= 3.0
step <= 2.0
memory <= 1.2
gradcheck = 1
```

才允许 limited FU smoke。未满足前不能进入 full functional matrix。

---

## 7. Line C：Functional Update source-estimator and metric solver

Line C 是 v22.07 的算法核心。它分四阶段。

### 7.1 C0：source-estimator no-commit audit

先不提交 update，只问：哪些 train-only 指标能预测 early source？

候选 estimator：

```text
E0 current loss cotangent magnitude
E1 train split B1/B2 transfer gain
E2 per-example gradient drift/diffusion SNR
E3 signal-channel / reservoir projection
E4 class-balanced early source density
E5 hidden/readout source decomposition
E6 optimizer-state conflict score
E7 low-NDS score
E8 information-volume / no-fold score
E9 KAN low-degree / low-frequency channel source score
```

记录：

```text
precommit_score
future_audit_source_h100/h400/h800/h1600/h3200
AUC_predict_h800_positive
AUC_predict_h3200_positive
Spearman_score_vs_source_h800
Spearman_score_vs_source_h3200
precision_at_topk
recall_at_topk
control_equivalent_fraction
```

注意：future source 只能用于 audit，不用于 direction。

C0 pass：

```text
AUC_predict_h800_positive >= 0.70
precision_at_top20 >= 0.50
control_equivalent_fraction <= 0.50
```

若 C0 不过：

```text
不进入 solver 大矩阵；
Codex 先尝试 estimator repair：
  class-balanced split；
  drift/diffusion window 改为 2-stage；
  hidden/readout separation；
  dataset-invariant z-score；
  remove estimator rows dominated by controls。
```

### 7.2 C1：target construction and contrast

将 estimator 变成函数空间目标 $\Delta f_{target}$。

目标族：

```text
T0 loss-cotangent target
T1 split-transfer target
T2 population-SNR target
T3 signal-channel target
T4 smooth-manifold target
T5 low-NDS target
T6 dual-memory target
T7 hidden-block source target
T8 class-balanced source target
T9 information-volume preserving target
T10 basis-channel target for KAN
```

对照：

```text
random matched target
sign-flip target
corrupt target
same-norm target
same-ActuationR2 random target
```

记录：

```text
target_norm
target_L2_energy
target_Fisher_energy
target_Sobolev_energy
target_RKHS_energy
target_NDS
target_info_volume
target_neighbor_preservation
target_signal_projection
target_reservoir_projection
B1_gain
B2_transfer_gain
B3_safety_gain
random_target_gap
sign_flip_gap
corrupt_gap
```

C1 pass：

```text
B2_transfer_gain >= random + 0.005
B3_safety_gain >= -0.005
random/sign-flip/corrupt controls fail
```

若 C1 不过：

```text
Codex 不调 lr；先换 target construction：
  reduce class imbalance；
  remove high-reservoir target component；
  add low-frequency / low-degree projection；
  add information-volume floor；
  re-run target contrast only。
```

### 7.3 C2：true metric solver

C2 必须真正求解：

$$
 u^\star = \arg\min_u \|J_\theta u - \Delta f_{target}\|_{G_f}^2 + \rho\|u\|_P^2.
$$

Solver levels：

```text
S1 ReadoutExactSolve
S2 HiddenBlockCGSolve
S3 BlockGaussNewtonSolve
S4 KANBasisChannelSolve
S5 OptimizerStateIntegratedSolve
```

Metric families：

```text
G0 L2 output metric
G1 diagonal Fisher metric
G2 PopRisk/SNR diagonal metric
G3 Sobolev-H1 hidden metric
G4 RKHS-KNN hidden graph metric
G5 Fisher-RKHS hybrid
G6 low-NDS curvature metric
G7 KAN basis-channel metric
G8 signal-reservoir metric
G9 information-volume constrained metric
```

必须记录：

```text
projection_residual_Gf
ActuationR2
ActuationCosine
B1_gain
B2_transfer_gain
B3_safety_gain
solve_time_ms
JVP_count
VJP_count
CG_iterations
solver_rank
condition_estimate
metric_energy_L2/Fisher/Sobolev/RKHS
NDS
info_volume
fold_proxy
function_displacement_cosine
parameter_update_cosine
```

C2 pass：

```text
projection_residual_Gf <= 0.35
ActuationR2 >= 0.60
B2_transfer_gain >= 0.005
B3_safety_gain >= -0.005
solve_time_ms <= 1.25 * baseline_update_ms for exploration
```

若 C2 不过：

```text
Codex 先尝试 solver repair，而不是 functional rerun：
  damping rho sweep fixed small set；
  CG iterations 5/10/20；
  Nyström rank 8/16；
  block restriction readout-only / hidden-only / block-combined；
  target rescale to avoid saturation；
  check JVP/VJP finite-difference correctness。
```

### 7.4 C3：source formation horizon matrix

只有 C0/C1/C2 通过的 candidate 进入 source formation。

Horizon：

```text
h100, h400, h800, h1600, h2400, h3200
```

记录：

```text
source_vs_best_control_h100/h400/h800/h1600/h2400/h3200
row_positive_count_each_horizon
retention_h1600_over_h800
retention_h3200_over_h1600
matched_control_source
stable_random_source
NoOp_source
AdamW_source
SGD_source
LineC_debt_peak/final/recovery
CEp99_debt_peak/final/recovery
ECE_debt_peak/final/recovery
Brier_debt_peak/final/recovery
AUCtime_ratio
```

C3 source formation pass：

```text
source_h100 >= 0.005
source_h400 >= 0.005
source_h800 >= 0.005
source_h1600 >= 0.005
source_h3200 >= 0.005
row_h3200_positive_count >= 6/9
matched controls fail
stable-random fail
AUCtime_ratio <= 1.10
LineC/ECE/Brier/tail debt not exploded
```

若 C3 不过：

```text
If C1/C2 pass but source_h100/h400/h800 negative:
  target has actuation but not source; return to estimator/target, not solver.

If source_h100/h400 positive but h800 fails:
  source formation is too transient; try short/long state split and lower NDS.

If h800 positive but h1600/h3200 fails:
  enter optimizer-state integration before terminal preservation.
```

---

## 8. Line D：Optimizer-state integration and terminal preservation

Line D 只在 C3 通过后运行。

### 8.1 D0 terminal autopsy

记录：

```text
optimizer cumulative projection on source direction
AdamW m/v cosine with source
SGD/momentum cosine with source
row_norm_drift
angular_velocity
radial_update_fraction
source_hidden_fraction
source_readout_fraction
tail / LineC / ECE / Brier transition h3200 -> h4800
classwise source collapse
dataset/seed localization
```

### 8.2 D1 preservation mechanisms

机制：

```text
P1 source-preserving projection
P2 row-orthogonal source update
P3 low-NDS matrix-block update
P4 dual-memory source state
P5 debt-aware terminal update
P6 schedule-free slow-state source preservation
P7 optimizer-state transport with source-aligned m/v
```

D1 pass：

```text
source_h4800 >= 0.005
R4800_over_3200 >= 0.50
row_h4800_positive_count >= 7/9
matched controls fail
stable-random fail
AUCtime_ratio <= 1.05
LineC/ECE/Brier/tail debt not worse
independent init-offset rerun pass
```

若 D1 不过：

```text
If optimizer projection is negative:
  try source-preserving projection and row-orthogonal update.

If debt transition explodes:
  try debt-aware terminal update.

If dataset/seed localized:
  mark dataset-heterogeneous; do not tune per dataset.

If controls share h4800 behavior:
  mark control-equivalent drift and stop same family.
```

---

## 9. Line E：KAN source-channel mapping

Line E 只在 MLP C3 或 D1 有 pass 后进入。KAN 不再直接复制 MLP update。

### 9.1 MLP source decomposition

记录：

```text
hidden_source_energy
readout_source_energy
source_subspace_rank
source_singular_values
source_reconstruction_error
block_alignment
row_angular_velocity
info_volume
```

### 9.2 KAN mapping candidates

D-CHE：

```text
CHE-low-degree-bank-commit
CHE-degree-readout-block-solve
CHE-readout-only-commit
CHE-basis-estimate-readout-commit
CHE-low-NDS-degree-update
```

D-FOU：

```text
FOU-low-frequency-bank-commit
FOU-band-readout-block-solve
FOU-readout-only-commit
FOU-basis-estimate-readout-commit
FOU-low-NDS-frequency-update
```

D-RAT / D-RBF：仅在 robust official / near-E1 pass 后 limited smoke。

D-RAT：

```text
RAT-joint-num-den-tangent-metric
RAT-readout-only-commit
RAT-den-safety-preserving-update
```

D-RBF：

```text
RBF-local-RKHS-source-channel
RBF-local-center-readout-commit
RBF-width-safe-source-update
```

### 9.3 KAN success gate

```text
KAN h100/h400/h800 >= 0.005
KAN h1600/h3200 >= 0.005
KAN R4800_over_3200 >= 0.50
row_h4800_positive_count >= 7/9
KAN_specific_delta_vs_MLP_same_metric >= 0.005
same-param MLP control available
matched controls fail
```

If fail：

```text
If MLP succeeds and KAN fails:
  source-channel mismatch; inspect low-degree/low-frequency/readout mapping.

If KAN only late rebound:
  not retained source; do not promote.

If KAN source opens but efficiency fails:
  route as InefficientButSourceSignalCarrier and repair kernel separately.
```

---

## 10. 4GPU dynamic queue

v22.07 必须真正并行。

```text
GPU0:
  Line A + MLP C0/C1/C2/C3 metric-solver matrix。

GPU1:
  D-CHE / D-FOU efficiency reconfirm + KAN source-channel mapping smoke。

GPU2:
  D-RAT / D-RBF multibatch officialization + limited smoke if near-E1。

GPU3:
  controls / stable-random / independent rerun / figures / packet generation。
```

必须落盘：

```text
v22_07_runnable_queue.csv
v22_07_gpu_assignment_manifest.csv
v22_07_gpu_utilization_timeline.csv
v22_07_idle_violation.csv
v22_07_queue_drain_report.csv
v22_07_deferred_items.csv
```

规则：

```text
if runnable_queue_nonempty and any_gpu_idle > 10 min:
  execution_contract_violation = 1
  final route cannot be completed-no-go
```

---

## 11. 必须生成的可视化

```text
code_truth_dashboard.svg
semantic_alias_heatmap_parameter.svg
semantic_alias_heatmap_function_displacement.svg
metric_operator_energy_dashboard.svg
projection_residual_vs_B2_transfer.svg
source_trajectory_h100_to_h6400.svg
terminal_erosion_autopsy.svg
optimizer_projection_on_source.svg
NDS_vs_retention.svg
Sobolev_RKHS_energy_vs_source.svg
info_volume_and_fold_proxy_trace.svg
signal_reservoir_projection.svg
KAN_source_channel_mapping_heatmap.svg
D-CHE_D-FOU_efficiency_dashboard.svg
D-RAT_D-RBF_multibatch_efficiency_dashboard.svg
D-RAT_component_waterfall.svg
D-RBF_component_waterfall.svg
gpu_utilization_timeline.svg
```

---

## 12. v22.07 成功标准

### 12.1 Minimum execution success

```text
S0.14 code gate pass；
D-CHE / D-FOU reconfirm pass；
D-RAT / D-RBF multibatch status closed；
C0/C1/C2/C3 raw matrices complete；
required figures complete；
4GPU queue drained without violation。
```

### 12.2 Functional algorithmic progress

```text
At least one legal train-only source estimator predicts early source better than random；
At least one target beats random/sign/corrupt controls on B2 transfer；
At least one true solver produces C2 actuation without proxy fallback；
At least one candidate forms h100/h400/h800 early chain or produces a clear source-observer no-go。
```

### 12.3 Functional exploration success

```text
source_h100/h400/h800 >= 0.005；
source_h1600/h3200 >= 0.005；
row_h3200_positive_count >= 6/9；
matched controls fail；
stable-random fail；
debt not exploded。
```

### 12.4 Productive terminal source

```text
source_h4800 >= 0.005；
R4800_over_3200 >= 0.50；
row_h4800_positive_count >= 7/9；
AUCtime_ratio <= 1.05；
LineC / ECE / Brier / CEp99 debt not worse；
independent init-offset rerun pass。
```

### 12.5 KAN-specific success

```text
KAN productive terminal source pass；
KAN_specific_delta_vs_MLP_same_metric >= 0.005；
corresponding basis efficiency pass；
source-channel mapping evidence exists；
controls fail。
```

### 12.6 Official success

```text
9/9 real dataset-seed pass；
source_vs_best_control >= 0.005；
AUCtime_ratio <= 1.0；
CEp99_delta <= 0.05；
NLL_delta <= 0.02；
ECE_delta <= 0.02；
Brier_delta <= 0.02；
LineC_pass = 1；
step_time_ratio <= 1.25；
memory_ratio <= 1.25；
promotion_allowed = 1。
```

---

## 13. 最终判断

v22.07 的核心不是再加 metric 名字，而是：

$$
\boxed{
\text{从“metric solver 能 actuation”推进到“metric solver 能生成 retained source”。}
}
$$

如果 v22.07 仍然是：

```text
C1/C3 pass；
C4 early source fail；
h3200 local positive；
h4800 not entered；
KAN mapping not entered；
```

那么结论不能再写成“继续调 metric solver”。必须写成：

```text
当前 train-only source estimator family 不足；
需要新的 source observability theory，而不是更多 solver/metric variants。
```

如果 v22.07 打开 MLP source formation 但 KAN 失败，下一轮才进入真正 KAN source-channel theory；如果 MLP 都不能 early source chain，则继续 KAN-FU 只是在浪费 GPU。
