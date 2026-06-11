# DG-KAN v22.06：Training-Dynamics / Metric-as-Geometry Functional Update + Basis Efficiency Full-Loop 完整计划

> 版本：v22.06 execution plan  
> 目标读者：不假设读者熟悉全部历史实验；本文开头先说明项目目标、当前进展、这轮为什么要改方向。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler/class weight；no dataset-name branch；no seed-specific rule；no fake/proxy/CPU offload；functional direction 不使用 validation/test/future/query；LineC / CEp99 / NLL / ECE / Brier / AUCtime 只能作为 audit、debt readback 和 gate，不能生成方向。

---

## 0. 项目总目标与 v22.05 后的真实状态

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，而是建立一个可与 MLP 竞争甚至超越 MLP 的 strict FC-PureKAN 训练系统。目标可以写成：

$$
\boxed{
\text{PureKAN base + functional update}
>
\text{PureKAN base + ordinary backprop / AdamW controls}
}
$$

这里“更好”必须同时包含四个维度：

```text
1. 表达与任务不劣于同参数 MLP；
2. forward / backward / step / memory 接近同参数 MLP；
3. functional update 的收益打过 AdamW / SGD / NoOp / random / stable-random / same-overhead controls；
4. 好处能长期留存，不只是 h800/h3200 局部 positive，且 tail / LineC / calibration / AUC debt 不爆。
```

v22.05 后，当前进展不能说完全没有，但也不能写成成功。

第一，效率路线已经有真实阶段性进展。D-CHE 与 D-FOU 已经是当前可用的主 KAN carrier；v22.05 中 D-CHE / D-FOU 继续保持 full-loop efficient carrier 状态。D-RAT / D-RBF 也不再完全停留在旧的 forward-blocked 表里：D-RBF 的 official fused transition 已经有后续打开迹象，D-RAT 也在后续 continuation 中补了 rational-k4 trainpath；但这些 continuation 必须在 v22.06 重新用一个自包含 code packet 审计，不能只靠追加复盘文字。

第二，functional update 没有能力突破。v22.04 最接近成功的 D1b candidate 已经接近 $R_{4800/3200}=0.50$，但 v22.05 的 metric-first 尝试反而没有形成 early/h3200 source chain。主 route 是 `F0-MetricNoEffect`：metric-first update 没有把 h3200/h4800 source retention 推过 gate。

第三，v22.05 的最大 insight 不是“metric 没用”，而是“metric 还没有真正进入算法”。当前代码中的 metric-first update 主要是 parameter-vector projection proxy：它对 flat gradient 做 smoothing、Fisher-like scaling、RKHS-like smoothing 或 low-NDS filtering，然后把结果当作 update。它还不是显式的函数空间问题：

$$
u^\star = \arg\min_u
\|J_\theta u-\Delta f_{target}\|_{G_f}^2+\rho\|u\|_P^2.
$$

因此 v22.05 只能说明：**当前 parameter-pullback proxy 形式的 metric-first FU 没有打开**。它不能说明 Sobolev / RKHS / Fisher / basis-channel metric 这个算法思想失败。

v22.06 的核心转向是：

$$
\boxed{
\text{从 metric-as-filter 转向 metric-as-geometry / metric-as-solver。}
}
$$

---

## 1. v22.06 要解决的本质问题

### 1.1 问题 A：代码和 artifact 必须彻底闭包

v22.05 的初始 code packet 基本可审，但后续 T9/T16、D-RAT/D-RBF repair continuation 没有完整合并进同一个源码包。这样会造成一个反复出现的问题：复盘文件说 route 更新了，但我无法确认这些 continuation 是否使用了同一套源码、同一套 metric 定义、同一套 seed 语义、同一套 source-chain gate。

v22.06 必须把代码审计变成第一硬门：

```text
最终 zip 解压目录 = 唯一审计对象；
不能在原始 repo 环境里 pass，而在 zip 里缺文件；
continuation runner、kernel、metric、mechanism 全部要进入同一 packet；
所有 route 必须能从 packet 内源码 + command journal + raw matrix 复算。
```

### 1.2 问题 B：metric 不能只做 flat-gradient smoothing

当前 metric helper 的核心形态大致是：

```text
flat gradient g
-> metric_project_vector(g, metric_name)
-> normalized projected vector
-> UpdateTensor
```

这不是严格意义上的 function-space metric。真正的 function-space metric 至少要出现：

```text
J_theta u：参数更新造成的函数变化；
G_f：函数空间 metric；
Δf_target：train-stream 可得的目标函数变化；
projection residual：J_theta u 是否真的实现 target；
metric energy：这个函数变形是否低能量、低撕裂、低曲率。
```

v22.06 要把 metric 放进 solver，而不是只当 update 过滤器。

### 1.3 问题 C：source formation 与 terminal preservation 是两件事

v22.04 告诉我们，有些 MLP candidate retention ratio 接近或超过 0.50，但 early source chain 不闭合；另一些 early chain 过了，但 h4800 retention 不足。v22.05 metric-first update没有形成 early source，说明它甚至没有进入 terminal retention 可评估状态。

因此 v22.06 不再把所有 FU 混成一个目标，而拆成两个阶段：

```text
Stage 1: source formation
  目标是 h100/h400/h800 先形成跨数据集 source chain。

Stage 2: source preservation
  目标是 h3200 -> h4800 不被 erosion，R4800/3200 过 gate。
```

如果 Stage 1 不过，就不要谈 terminal preservation；如果 Stage 1 过但 Stage 2 不过，就做 erosion autopsy 和 preservation solver。

### 1.4 问题 D：KAN 的问题不是效率，而是 source-channel writeback

D-CHE / D-FOU 已经足够快。它们仍然没有 retained source，说明当前 KAN blocker 更像：

```text
MLP source 可以写入 hidden/readout matrix；
KAN source 没有写入 low-degree / low-frequency / readout-stable channel；
KAN 的 degree/frequency/basis bank 可能把 source 写进 reservoir。
```

v22.06 必须做 KAN source-channel mapping，而不是把 MLP metric update 直接搬到 D-CHE/D-FOU。

### 1.5 问题 E：D-RAT / D-RBF 不能再停在“micro 有希望”

v22.03/v22.04/v22.05 都显示 D-RAT / D-RBF 的 micro-kernel 有明显改善，但 official fused production path 和 full-loop functional runner 仍不够清楚。v22.06 必须把它们推到明确状态：

```text
D-RAT: ProductionFusedPass / OfficialFusedBlocked / SafetyBlocked / RejectedForThisVersion
D-RBF: ProductionFusedPass / OfficialFusedBlocked / MaterializationBlocked / RejectedForThisVersion
```

---

## 2. 哪些旧方向可以重开，哪些必须停止

### 2.1 可以重开的方向

#### 2.1.1 Function metric / basis natural update

v15.02.1 的 function-metric line 失败过，v13.2/v13.3 的 basis-natural / low-rank / block metric 也没有打开。但它们失败时的环境与现在不同：当时 D-CHE/D-FOU 的 full-loop efficiency 没站稳，source-retention/h4800 gate 没成熟，LineC/debt/route 也没现在严格。

现在可以重开，但必须换语义：

```text
旧语义：function metric 是一个 update token。
新语义：function metric 定义函数空间几何，进入 projection solver。
```

#### 2.1.2 Operator-level basis-channel functional

v13.4 的 operator-level basis-channel target 思路值得重开。因为它比很多 Fxx 工程搜索更接近真正 functional update：先定义函数/hidden/basis channel target，再投影到参数。

v22.06 将重开：

```text
D-CHE degree-channel operator target；
D-FOU band-channel operator target；
D-RAT rational tangent operator target；
D-RBF local-kernel operator target。
```

#### 2.1.3 PopRisk / SNR

PopRisk/SNR 过去多次失败，尤其 v22 系列里作为 one-shot source selector 表现不佳。但它仍然符合 signal channel 理论：只要它作为 precommit metric / source-risk estimator，而不是直接生成 flat update，就仍然有价值。

v22.06 将只允许 PopRisk/SNR 作为：

```text
metric weight；
source-risk audit；
train-only candidate selector；
不能直接当 final update。
```

#### 2.1.4 MLP functional line

MLP 不是普通 control。MLP 是判断 functional update 机制是否 generic-feasible 的实验台。v22.06 必须继续 MLP，因为如果 MLP 都不能形成 source chain，KAN 上继续调 source-channel 没意义。

### 2.2 必须停止或降预算的方向

```text
action bank / controller / reset route：不重启。
cover objective 小修：v13.11 已显示 objective invalid，不重启。
M31/M32 同族修补：v19 已经独立确认失败，不重启。
继续扩大 Fxx/D1x terminal floor / lookahead / scale：停止主预算。
只追 ActuationR2：停止作为主目标。
只追 h6400 late positive：停止作为成功证据。
```

---

## 3. 当前研究进展如何转化为 v22.06 Functional Update 算法

本节只更新 functional update 主线。代码审计、basis efficiency、D-CHE / D-FOU / D-RAT / D-RBF 的执行结构保持原计划不变。

v22.06 的 functional update 不再以“新增更多 update token”为中心，而是从第一性原则重新定义为：

$$
\boxed{
\text{在训练动力学中，用 train-stream 可观测信号构造函数空间边界条件，}
\text{再通过可审计的函数空间求解器写入模型，并保护该 source 不被后续训练侵蚀。}
}
$$

这句话拆成四个必要条件：

```text
1. Observability：训练流里必须能看见可能泛化的 signal，不能靠 validation/test/future。
2. Actuatability：模型参数必须能真实实现这个函数变化，不只是 ActuationR2 高。
3. Integrability：这个函数变化必须能和后续优化器动力学兼容，不被 AdamW/SGD/动量立刻洗掉。
4. Preservation：source 必须从 h100/h400/h800 形成，并在 h3200/h4800 仍保留，不能只靠 late rebound。
```

若四者任一不成立，functional update 都不应 promotion。

### 3.1 第一性原则：FU 不是一步 update，而是训练向量场的受控扰动

普通训练可以写成离散动力系统：

$$
\theta_{t+1}=\theta_t+U_{opt}(\theta_t,B_t,s_t),
$$

其中 $B_t$ 是 batch，$s_t$ 是 optimizer state。早期我们把 FU 当成：

$$
\theta_{t+1}=\theta_t+U_{opt}+u_{FU}.
$$

这太弱。它只是在当前点加一个向量。v22.06 修正为：FU 应该改变训练向量场的边界条件、状态和函数空间投影：

$$
\theta_{t+1}=\theta_t+
\Pi_{\mathcal{C}_t,G_f}\bigl(U_{opt}(\theta_t,B_t,s_t), s^{source}_t, \Delta f^{target}_t\bigr),
$$

其中 $\Pi_{\mathcal{C}_t,G_f}$ 是在函数空间 metric $G_f$ 下的约束投影，$s^{source}_t$ 是 train-stream source state，$\Delta f^{target}_t$ 是训练流可观测的目标函数变化。

这意味着 v22.06 的 FU 不是“AdamW + residual trick”，而是以下四层算法：

```text
C1: source observability：看见信号。
C2: function-space actuation：把信号变成可执行函数变化。
C3: optimizer-state integration：让这个变化进入后续训练动力学。
C4: terminal preservation：防止 h3200 -> h4800 被侵蚀。
```

### 3.2 从优化器研究中抽出的原则

下面这些研究进展不应被机械变成“换 optimizer”，而应该转成 functional update 的算法约束。

#### 3.2.1 AdamW / decoupled decay：regularization 与 source value 必须解耦

AdamW 的启发是：正则化不应该混在 loss gradient 里。对 FU 来说，degree damping、frequency damping、weight decay、denominator damping、source preservation 都不能被写成 source 本身。每个 positive 都必须有：

```text
decay-only control；
random source + same decay control；
FU + no decay control；
FU + decoupled decay control。
```

若 decay-only 也产生同等 h4800 retention，则 route 是 `RegularizationExplainsSource`，不能写 FU success。

#### 3.2.2 Schedule-Free / Lookahead / SWA / EMA：fast iterate 与 slow iterate 分工

Schedule-free / iterate averaging / Lookahead / SWA 类思想的共同点是：训练中有 fast state 负责探索，有 slow state 负责稳定。v22.06 不再只做一个 slow EMA，而要显式记录：

```text
fast_iterate_source
slow_iterate_source
fast_slow_function_gap
slow_state_retention_h3200_to_h4800
source_written_to_fast_only_or_slow
```

如果 fast source 到 h3200 很强，但 slow state 不保留，则 h4800 erosion 的原因可能是 source 没进入可保存状态。此时 Codex 不能继续加大 FU 幅度，而要改成 slow-state writer。

#### 3.2.3 AdEMAMix / older gradients：旧梯度可能仍然有信息

单一 EMA 同时照顾近期与远期信息很难。v22.06 用 dual-memory source state：

$$
s_t^{short}=\beta_s s_{t-1}^{short}+(1-\beta_s)\hat{s}_t,
$$

$$
s_t^{long}=\beta_l s_{t-1}^{long}+(1-\beta_l)\hat{s}_t,
$$

其中 $\hat{s}_t$ 是 train-stream source estimate。判断不是 short/long 谁大，而是：

```text
short_state predicts h800/h1600 source；
long_state predicts h3200/h4800 retention；
short-long agreement predicts stable source；
short-only source 是否更容易 terminal erosion。
```

#### 3.2.4 Muon / SOAP / Nora：matrix/block 几何比逐参数几何更重要

MLP 的 source 主要存在 hidden/readout 矩阵里；D-CHE / D-FOU 的 source 应该存在 degree-readout / band-readout block 里。v22.06 不直接换 Muon / SOAP / Nora，而提取三个约束：

```text
Muon/NDS：低曲率方向比大 norm 方向更可能留存。
SOAP：adaptive moment 应该在慢变 block basis 中运行。
Nora：矩阵 row 的 radial drift 与 angular velocity 要稳定。
```

必须记录：

```text
block_update_norm
row_radial_component
row_angular_component
row_norm_drift
angular_velocity
within_block_NDS
cross_block_NDS
source_retention_delta_after_block_projection
```

若 MLP source 主要来自 matrix-block geometry，而 KAN source 写入 degree/frequency bank 后失效，则 route 是 `KANBlockCoordinateMismatch`，不能继续把 MLP update 直接搬到 KAN。

#### 3.2.5 Natural gradient / K-FAC / Shampoo：metric 是局部几何，不是装饰

真正的 metric-first FU 必须包含 $J_\theta u$，而不是 flat gradient smoothing。v22.06 的核心 solver 是：

$$u^\star=
\arg\min_u
\|J_\theta u-\Delta f^{target}\|_{G_f}^2+
\rho\|u\|_P^2.
$$

这里 $G_f$ 可以是 L2、diag Fisher、Sobolev、RKHS、Fisher-RKHS、basis-channel metric。若实现里没有 JVP/VJP 或 readout exact solve，只能标记为 `MetricProxy`，不能写成 `MetricSolver`。

#### 3.2.6 SAM / sharpness / NDS：source 不能走高曲率捷径

h3200 source 到 h4800 被侵蚀，可能是因为 update 方向的二阶代价高。v22.06 必须记录：

$$
I^{(1)}=-\langle g,u\rangle,
$$

$$
I^{(2)}=\frac{1}{2}u^THu,
$$

$$
NDS(u)=\frac{u^THu}{\|u\|^2+\epsilon}.
$$

如果 $I^{(1)}$ 高但 $NDS$ 高，source 可能短期强、后期掉。此时 Codex 不应继续放大 source，而应尝试 low-NDS projection 或 block-balanced update。

#### 3.2.7 Cautious optimizer：alignment 是诊断，不是硬规则

如果 proposed FU 与当前 gradient 方向冲突，不能简单丢弃，因为有些有用的轨迹变形可能短期 anti-gradient。但必须记录：

```text
FU_vs_grad_cosine
FU_vs_optimizer_step_cosine
FU_vs_momentum_cosine
aligned_source_retention
anti_aligned_source_retention
```

若 anti-aligned FU 在 h800 有 source 但 h4800 掉，说明它可能是短期跳跃而非 stable signal。若 anti-aligned FU 反而 h4800 留存，则 alignment mask 不能作为硬门。

#### 3.2.8 Signal channel / reservoir：source 必须进入 test-visible 通道

Generalization theory 的启发是：训练能移动很多方向，但只有 signal channel 中的 coherent signal 会 transfer；reservoir 里的运动可能对 test/probe 不可见。v22.06 需要估计：

```text
signal_channel_projection
reservoir_projection
source_to_reservoir_leakage
noise_in_signal_channel
stable_random_transfer
split_transfer_operator_gain
```

这不是 audit-only。C1 source observability 必须优先选择 signal-channel projection 高、reservoir leakage 低的 target。

#### 3.2.9 Manifold / diffeomorphism：FU 应是平滑可逆变形，不是撕裂

将 FU 看成函数空间形变。成功的 FU 应满足：

```text
no tearing：相近样本不应被拉到完全不同方向；
no folding：局部邻域结构不应被翻转；
bounded deformation：Jacobian condition 不爆；
non-degenerate information volume：source path 不应塌成退化低维线。
```

因此 v22.06 记录：

```text
neighbor_preservation
fold_proxy
Jacobian_condition
source_path_intrinsic_dim
info_volume
info_volume_drop_h3200_to_h4800
```

### 3.3 哪些旧方向可以重开，哪些不能

可以重开的旧方向，必须换语义：

```text
Function metric / v15.02.1：旧版是 metric token/proxy；可重开为 JVP/VJP metric solver。
Basis natural update / v13.2-v13.3：旧版 D-CHE/DFOU efficiency 未站稳；现在可在 efficient carrier 上重测 block-natural metric。
Operator-level basis-channel / v13.4：可重开为 function target -> basis/readout projection solver。
PopRisk/SNR / v13.6-v13.8：可重开为 source observability，不直接当 update mask。
Split-consensus / v15.04-v15.7：可重开为 signal-channel target estimator，不再当 trust scalar。
Optimizer-state transport / v14.7-v14.9：可重开为 FU-to-optimizer-state integration，不再当 reset action。
```

必须停止或降预算：

```text
action bank / controller；
reset route；
cover objective 小修；
M31/M32 同族修补；
G/N/Q-token 扩展；
只追 h6400 late positive；
只提高 h800/h3200 source amplitude；
只换 optimizer 名字但不记录函数空间效果。
```

---

## 4. v22.06 实验总体结构

v22.06 分四条主线同步执行：

```text
Line A: S0.13 代码/指标/机制硬审计。
Line B: basis efficiency full-loop closure。
Line C: Metric-as-Geometry FU solver。
Line D: KAN source-channel mapping。
```

四张 GPU 动态队列：

```text
GPU0:
  S0.13 + MLP metric solver + terminal preservation autopsy。

GPU1:
  D-CHE/D-FOU metric solver + source-channel mapping。

GPU2:
  D-RAT/D-RBF production fused repair + limited metric smoke。

GPU3:
  controls / independent rerun / figures / queue refill / h4800 extension。
```

硬规则：

```text
runnable_queue 非空 && 任一 GPU idle > 10 min
=> execution_contract_violation = 1
=> 不能写 completed no-go。
```

---

## 5. Line A：S0.13 代码、指标、机制语义硬门

v22.06 必须先通过 S0.13，否则不进入 science route。

### 5.1 Codex 必须打包的文件

最终必须生成：

```text
v22_06_code_review_packet.zip
v22_06_results_bundle.zip
```

`v22_06_code_review_packet.zip` 必须包含：

```text
00_README.md
02_SOURCE_TREE/dgkan/**/*.py
02_SOURCE_TREE/experiments/**/*.py
02_SOURCE_TREE/tests/**/*.py
03_IMPORT_CLOSURE/
04_LINEC_CORRECTNESS/
05_SOURCE_CHAIN_RETENTION_DEBT/
06_MECHANISM_SEMANTIC_CONTRACTS/
07_FUNCTION_SPACE_METRICS/
08_BASIS_KERNELS_AND_PROFILERS/
09_GPU_QUEUE_CONTRACT/
10_RESULTS_POINTERS/
packet_manifest.csv
packet_sha256_manifest.csv
```

必须包含以下关键源文件：

```text
dgkan/fu/core.py
dgkan/fu/source_chain.py
dgkan/fu/terminal_retention.py
dgkan/fu/debt_accounting.py
dgkan/fu/function_space_metrics.py
dgkan/fu/metric_projection.py
dgkan/fu/metric_solver.py
dgkan/fu/jacobian_sketch.py
dgkan/fu/sobolev_metric.py
dgkan/fu/rkhs_metric.py
dgkan/fu/fisher_metric.py
dgkan/fu/basis_channel_metric.py
dgkan/fu/mechanisms.py
dgkan/metrics/linec.py
dgkan/kernels/fused_chebyshev_k3.py
dgkan/kernels/fused_fourier_k2.py
dgkan/kernels/fused_rational_k4.py
dgkan/kernels/fused_rbf.py
dgkan/kernels/rational_fused.py
dgkan/kernels/rbf_sparse.py
dgkan/profiling/efficiency_v22_06.py
experiments/run_v22_06_s013_truth_gate.py
experiments/run_v22_06_metric_solver_fu.py
experiments/run_v22_06_terminal_preservation.py
experiments/run_v22_06_kan_source_mapping.py
experiments/run_v22_06_drat_drbf_officialization.py
experiments/run_v22_06_finalize.py
```

### 5.2 必须通过的代码测试

```text
compileall = 0 error；
required import closure = 0 error；
self_contained_import_check = 1；
LineC fast/channel golden pass；
source-chain tests pass；
terminal-retention tests pass；
debt accounting tests pass；
metric solver unit tests pass；
semantic alias audit pass；
kernel gradcheck pass；
profiler phase tests pass；
```

### 5.3 semantic alias audit

对所有 active mechanisms，在同一 model / batch 上生成 update tensor，记录：

```text
pairwise cosine；
relative L2 distance；
norm ratio；
support overlap；
role-wise energy；
function displacement cosine。
```

若两个机制满足：

$$
\cos(u_i,u_j)>0.999,
$$

$$
0.99\le \frac{\|u_i\|}{\|u_j\|}\le 1.01,
$$

且 function displacement cosine > 0.999，则它们属于同一个 alias group。alias group 未声明时，S0.13 fail。

---

## 6. Line B：basis efficiency full-loop closure

### 6.1 D-CHE / D-FOU reconfirm

目标：确认 metric-solver FU 也使用同一 efficient kernel，而不是只有 profiler 快。

候选：

```text
D-CHE:
  CHE21-R2-low-degree-k3-official
  CHE21-R4-k5-gradbuf-triton-ablation

D-FOU:
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

成功标准：

```text
D-CHE:
  >=2 variants × >=3 batch sizes S1 pass；
  forward <=1.25；step <=1.25；memory <=1.05；
  fallback_kernel_used=0。

D-FOU:
  FOU-R3 >=3 batch sizes S1 pass；
  forward <=1.25；step <=1.25；memory <=1.05；
  fallback_kernel_used=0。
```

### 6.2 D-RAT production officialization

D-RAT 不能停在 micro numerator/denominator benchmark。v22.06 要求 production trainpath：

```text
forward + manual CE backward + parameter update + safety telemetry separated。
```

记录：

```text
numerator_eval_ms
denominator_eval_ms
reciprocal_ms
safety_guard_ms
telemetry_ms
trainpath_forward_ms
trainpath_backward_ms
trainpath_step_ms
memory_ratio
rational_den_min
rational_den_p01
rational_gradcheck_relerr
same_kernel_runner_proof
```

D-RAT near-E1：

```text
forward <=3.0；step <=2.0；memory <=1.2；gradcheck pass。
```

D-RAT official：

```text
forward <=1.5；step <=1.5；memory <=1.1；den safety pass；same-kernel runner proof = 1。
```

### 6.3 D-RBF production officialization

D-RBF 必须从 micro local-K 转成 production sparse-local path。

记录：

```text
active_center_fraction
mean_local_K
local_gather_ms
exp_eval_ms
readout_matmul_ms
local_backward_ms
dense_materialized_bytes
memory_ratio
forward_ratio
step_ratio
same_kernel_runner_proof
```

成功标准同 D-RAT official。若 active_center_fraction > 0.80 且 forward 仍 > 3.0，判为 locality-not-realized。

---

## 7. Line C：Functional Update 主线重写 —— Source-Observation → Function-Space Actuation → Optimizer Integration → Terminal Preservation

这是 v22.06 的核心更新。上一版 v22.06 把重点放在 metric projection solver，但仍容易把 metric 做成 flat-gradient filter。新版 Line C 明确规定：Functional update 必须是一套训练动力学算法，而不是一组工程 token。

Line C 由七个阶段组成：

```text
C0: semantic truth gate，证明不是 proxy / alias。
C1: source observability，找到 train-stream 可观测的泛化信号。
C2: function-space target construction，构造 Δf_target。
C3: metric actuation solver，求解 Jθu ≈ Δf_target。
C4: optimizer-state integration，把 FU 写入后续训练动力学。
C5: terminal preservation，保护 h3200 source 到 h4800。
C6: causal controls and independent confirmation。
```

每一阶段都有失败分支。Codex 不能跳过前一阶段直接跑长 horizon。

### 7.0 C0：semantic truth gate，先证明“这是真 FU 算法”

C0 必须在最终 zip 解压目录里执行，不得使用原始 repo 环境。每个 functional mechanism 必须生成一条 semantic contract：

```text
mechanism_id
mechanism_family
source_observer_type
target_constructor_type
metric_operator_type
solver_type
commit_type
optimizer_integration_type
preservation_type
uses_future_or_validation
uses_audit_metric_for_direction
is_proxy
is_alias_of
```

硬规则：

```text
如果没有 JVP/VJP、readout exact solve 或 block-Jacobian sketch，不能标成 MetricSolver，只能标成 MetricProxy。
如果两个 mechanism 的 update cosine > 0.999、function displacement cosine > 0.999、role energy profile 相同，则必须合并为同一 alias group。
如果 direction 使用 LineC/ECE/Brier/tail/AUCtime 生成，直接 forbidden。
```

C0 输出：

```text
v22_06_functional_semantic_contract.csv
v22_06_functional_alias_matrix.csv
v22_06_function_displacement_alias_matrix.csv
v22_06_forbidden_direction_audit.csv
```

### 7.1 C1：source observability，不先求 update，先证明训练流能看见 signal

C1 的目标不是提交参数，而是回答：

$$
\boxed{
\text{训练流中是否存在一个 precommit 可观测量，能预测 retained source，}
\text{而不是只预测 train loss drop 或 late rebound？}
}
$$

C1 只允许使用：

```text
current train batch；
train split B1/B2/B3；
per-example gradients / output cotangents；
current logits / hidden activations；
optimizer state m/v；
basis-channel telemetry；
current source state；
```

禁止使用：

```text
validation/test；
future horizon outcome；
LineC/ECE/Brier/tail/AUCtime 作为 direction source；
dataset name / seed branch。
```

C1 必须同时测五类 observability：

#### C1-a：split-transfer observability

把当前 train batch 分成 $B_1,B_2,B_3$。一个 target 或 source estimator 必须在 $B_1$ 上形成 gain，并在 $B_2$ 上 transfer，不应只在 $B_1$ 自嗨。

记录：

```text
B1_gain
B2_transfer_gain
B3_safety_gain
B2_over_B1_ratio
split_transfer_variance
sign_flip_B2_gain
random_matched_B2_gain
corrupt_target_B2_gain
```

C1-a pass：

```text
B2_transfer_gain > random_matched_B2_gain + 0.005；
B2_over_B1_ratio >= 0.25；
sign_flip_B2_gain < 0；
corrupt_target_B2_gain <= random_matched_B2_gain。
```

#### C1-b：drift-diffusion / PopRisk observability

对 per-example gradient 或 output cotangent 估计 coherent drift 与 diffusion：

$$
\mu = \mathbb{E}_i[g_i],
$$

$$
\sigma^2 = \mathbb{E}_i\|g_i-\mu\|^2,
$$

$$
SNR = \frac{\|\mu\|^2}{\sigma^2/(B-1)+\epsilon}.
$$

记录：

```text
grad_drift_norm
grad_diffusion_norm
source_snr
output_cotangent_snr
snr_by_layer
snr_by_role
snr_by_basis_channel
```

C1-b pass 不是 SNR 高，而是：

```text
SNR estimator 在 locked historical rows 上预测 h3200/h4800 retention AUC >= 0.65；
在 live precommit run 中不使用 future label；
不能只预测 h800 source，必须预测 R4800_over_3200 或 terminal erosion risk。
```

#### C1-c：signal-channel / reservoir observability

用 train split 构造一个 cheap signal-channel proxy。对候选 target $\Delta f$，估计：

```text
source_channel_projection
reservoir_projection
source_to_reservoir_leakage
noise_in_signal_channel
stable_random_transfer_gap
```

C1-c pass：

```text
source_channel_projection >= reservoir_projection；
source_to_reservoir_leakage <= 0.35；
stable_random_transfer_gap >= 0.005。
```

#### C1-d：optimizer-compatibility observability

记录候选 source 与当前优化器动力学是否兼容：

```text
cos_target_grad
cos_target_adamw_step
cos_target_sgd_step
cos_target_momentum
cos_target_old_long_ema
optimizer_conflict_score
```

这不是 hard mask。它只决定 C4 的 integration mode。

#### C1-e：manifold deformation observability

记录候选 target 是否像平滑变形：

```text
neighbor_preservation_prefit
fold_proxy_prefit
hidden_sobolev_energy_prefit
rkhs_graph_energy_prefit
info_volume_prefit
```

C1-e 如果失败，该 target 仍可进入 C2 smoke，但不能进入 C4/C5 terminal preservation。

### 7.2 C2：function-space target construction，先定义 Δf，再谈参数更新

C2 构造 $\Delta f_{target}$。目标不再是“梯度方向”本身，而是一个 train-stream 合法、跨 split 一致、可保留的函数变化。

Target families：

```text
T0 Loss-cotangent baseline target，仅作 baseline。
T1 Split-transfer target，最大化 B2/B1 transfer。
T2 Drift-diffusion target，强调 coherent drift。
T3 Signal-channel target，减少 reservoir leakage。
T4 Smooth-manifold target，低 Sobolev / RKHS energy。
T5 Low-NDS target，降低曲率惩罚。
T6 Dual-memory target，short state 负责 source formation，long state 负责 terminal retention。
T7 Block-structured target，限制在 hidden/readout 或 degree/band-readout block。
T8 KAN low-degree / low-frequency target。
```

C2 必须输出：

```text
target_id
target_family
target_norm_L2
target_norm_Fisher
target_Sobolev_H1_energy
target_RKHS_energy
target_NDS_proxy
target_signal_projection
target_reservoir_projection
target_info_volume
target_neighbor_preservation
target_fold_proxy
```

C2 target gate：

```text
B2_transfer_gain > random matched；
source_channel_projection > reservoir_projection；
metric_energy not exploded；
no forbidden information；
```

若 C2 target gate 不过，不能跑 solver 长程，只记录 `TargetObservableFailed` 或 `TargetObservableNoRetentionRisk`。

### 7.3 C3：metric actuation solver，真正求 $J_\theta u \approx \Delta f_{target}$

C3 是 v22.06 与 v22.05 最大区别。v22.05 的 metric 是 flat-gradient proxy；v22.06 必须实现可审计 solver。

核心优化问题：

$$u^\star=
\arg\min_u
\|J_\theta u-\Delta f_{target}\|_{G_f}^2+
\rho\|u\|_P^2.
$$

Solver levels：

```text
S0 Flat proxy baseline：只作对照，不可 promotion。
S1 Readout exact solve：只解 readout / final mixing block。
S2 Diagonal JVP/VJP sketch：低成本近似。
S3 Low-rank JVP/VJP CG：rank 4/8/16。
S4 Block-Jacobian solver：hidden/readout、degree-readout、band-readout block。
S5 KAN basis-channel solver：D-CHE degree / D-FOU band / D-RAT num-den / D-RBF local center。
```

Metric families：

```text
G0 L2 output metric。
G1 Diag Fisher probability metric。
G2 PopRisk/SNR diagonal metric。
G3 Hidden Sobolev-H1 metric。
G4 RKHS-KNN graph metric。
G5 Fisher-RKHS hybrid。
G6 Low-NDS curvature metric。
G7 KAN basis-channel metric。
G8 Ensemble metric with learned fixed weights from train-only observability.
```

C3 必须记录：

```text
solver_level
metric_family
target_family
projection_residual_Gf
ActuationR2
function_displacement_norm
parameter_update_norm
solve_time_ms
JVP_count
VJP_count
CG_iterations
solver_rank
condition_estimate
B2_transfer_after_actuation
random_target_actuation_gap
```

C3 pass：

```text
ActuationR2 >= 0.50；
projection_residual_Gf 比 S0 flat proxy 降低 >= 20%；
B2_transfer_after_actuation > matched random + 0.005；
solve_time_ms 不超过 baseline step 的 25%，或标记为 expensive-diagnostic；
```

C3 若不过，Codex 只能尝试：

```text
readout-only solve；
lower rank；
higher damping rho；
smaller target norm；
block-restricted solve；
```

不能直接进入 h4800。

### 7.4 C4：optimizer-state integration，FU 必须进入后续训练动力学

C4 回答：$u^\star$ 提交后，普通 optimizer 是否会接纳它，还是把它当噪声洗掉。

C4 比较五种 integration mode：

```text
I0 Direct parameter commit：baseline。
I1 Source-state commit：写入 short/long source state，不直接大幅改参数。
I2 Optimizer-state transport：把 FU 对应的方向写入 m/v 或 momentum state。
I3 Boundary-condition attach：在一个窗口内把 FU 当作训练边界条件，影响后续 update projection。
I4 Source-preserving optimizer projection：后续 optimizer 不允许破坏 source direction。
I5 Block-matrix integration：MLP hidden/readout、D-CHE degree-readout、D-FOU band-readout block separately integrate。
```

必须记录：

```text
FU_param_commit_norm
FU_function_commit_norm
optimizer_state_m_projection_on_FU
optimizer_state_v_projection_on_FU
post_commit_grad_alignment
cumulative_optimizer_projection_h100_to_h800
cumulative_optimizer_projection_h3200_to_h4800
fast_slow_function_gap
short_long_source_agreement
```

C4 pass：

```text
h100/h400/h800 source >= 0.005；
row_h800_positive_count >= 6/9；
cumulative optimizer projection on source is not strongly negative；
matched controls fail。
```

### 7.5 C5：terminal preservation，专门攻 h3200 -> h4800 erosion

C5 不再生成更大的 source，而是保护已有 h3200 source。

给定 h3200 source direction $s_{3200}$，对后续函数变化 $\Delta f_t$，计算：

$$
E_t^{erase}=
-\min\left(0,\langle \Delta f_t,s_{3200}\rangle_{G_f}\right).
$$

如果 $E_t^{erase}>0$，按 metric projection 去掉破坏分量：

$$
\Delta f_t^{safe}=
\Delta f_t-
\frac{\min(0,\langle \Delta f_t,s_{3200}\rangle_{G_f})}
{\|s_{3200}\|_{G_f}^2+\epsilon}s_{3200}.
$$

C5 preservation modes：

```text
P0 No preservation baseline。
P1 Metric orthogonal source projection。
P2 Low-NDS source projection。
P3 Dual-memory source preservation。
P4 Row-orthogonal block preservation。
P5 Debt-aware preservation，优先偿还 tail/ECE/Brier/LineC debt。
P6 Info-volume preservation，防止 source subspace collapse。
P7 Signal-channel preservation，防止 source -> reservoir leakage。
```

记录：

```text
source_h3200
source_h4000
source_h4800
R4800_over_3200
R6400_over_4800
erase_energy_h3200_to_h4800
removed_destructive_component_norm
source_channel_projection_drop
reservoir_leakage_increase
NDS_h3200_to_h4800
info_volume_drop
hidden_source_fraction_drop
readout_source_fraction_drop
tail_debt_transition
LineC_debt_transition
ECE_debt_transition
Brier_debt_transition
```

C5 productive gate：

```text
source_h4800 >= 0.005；
R4800_over_3200 >= 0.50；
row_h4800_positive_count >= 7/9；
matched controls fail；
stable-random controls fail；
AUCtime_ratio <= 1.05；
tail/LineC/ECE/Brier debt 不恶化。
```

### 7.6 C6：causal controls 与独立确认

任何 C5 positive 必须执行：

```text
NoOp matched overhead；
random matched norm；
stable random source state；
same metric random target；
same ActuationR2 random target；
optimizer-only control；
decay-only control；
MLP same-metric control；
KAN same-metric control；
independent init offsets；
matched batch stream rerun。
```

若同 metric random target 也过，route 是 `MetricRegularizesButTargetNotSpecific`。若 optimizer-only 过，route 是 `OptimizerDynamicsExplainsSource`。若 MLP 过而 KAN 不过，route 是 `GenericMLPSourceKANMappingFailed`。

### 7.7 C7：horizon protocol 与执行规则

Horizon ladder：

```text
h100 / h400 / h800: source formation。
h1600 / h2400 / h3200: source consolidation。
h4000 / h4800 / h6400: terminal preservation。
```

进入条件：

```text
C1/C2 过才进入 C3。
C3 过才进入 C4。
C4 early source 过才进入 C5。
C5 positive 才进入 C6 independent confirmation。
```

这样避免继续大规模跑无效 h6400 late rebound。

---

## 8. Line D：KAN source-channel mapping（保留原目标，但改为 metric/optimizer-aware mapping）

Line D 只在 MLP 或 C3/C4 中出现可解释 source 后执行。KAN 不再直接复制 MLP update，而是先做 source decomposition，再做 channel mapping。

### 8.1 MLP source decomposition

对 MLP source，记录：

```text
hidden_source_energy
readout_source_energy
bias_source_energy
source_subspace_rank
source_singular_values
matrix_block_alignment
row_radial_component
row_angular_component
source_reconstruction_error
source_NDS
source_info_volume
```

如果 MLP source 主要在 readout，不允许直接写 KAN basis；先做 KAN readout-only commit。如果 MLP source 主要在 hidden matrix 的低秩方向，KAN 应先映射到 degree-readout / band-readout，而不是高阶 / 高频 basis 参数。

### 8.2 D-CHE source mapping

D-CHE channels：

```text
low-degree source bank；
high-degree reservoir；
degree-readout matrix；
readout-only；
basis-estimate / readout-commit；
low-NDS degree block；
Sobolev degree metric。
```

记录：

```text
low_degree_source_energy
high_degree_reservoir_energy
degree_entropy
source_to_reservoir_leakage
basis_commit_vs_readout_commit
degree_block_NDS
low_degree_Sobolev_energy
KAN_metric_projection_residual
```

D-CHE mapping pass：

```text
low_degree_source_energy > high_degree_reservoir_energy；
source_to_reservoir_leakage <= 0.35；
readout-only 或 degree-readout commit 至少一个形成 h100/h400/h800 source。
```

### 8.3 D-FOU source mapping

D-FOU channels：

```text
low-frequency source bank；
high-frequency reservoir；
band-readout matrix；
readout-only；
basis-estimate / readout-commit；
low-NDS frequency block；
frequency Sobolev metric。
```

记录同 D-CHE，替换 degree 为 frequency/band。

D-FOU mapping pass：

```text
low_frequency_source_energy > high_frequency_reservoir_energy；
source_to_reservoir_leakage <= 0.35；
band-readout 或 readout-only commit 至少一个形成 h100/h400/h800 source。
```

### 8.4 D-RAT / D-RBF source mapping smoke

D-RAT / D-RBF 只有在 Line B official fused runner 过后才能进入 limited mapping。否则只做 channel audit，不做 full FU。

D-RAT 记录：

```text
numerator_tangent_source
denominator_tangent_source
num_den_coupling_energy
denominator_safety_margin
rational_tangent_metric_residual
```

D-RBF 记录：

```text
local_center_source_energy
readout_source_energy
active_center_fraction
local_K_source_consistency
RKHS_local_energy
```

### 8.5 KAN source-channel success gate

```text
KAN h100/h400/h800 >= 0.005；
KAN h1600/h3200 >= 0.005；
R4800_over_3200 >= 0.50；
row_h4800_positive_count >= 7/9；
KAN_specific_delta_vs_MLP_same_metric >= 0.005；
controls fail；
D-CHE / D-FOU / D-RAT / D-RBF 的效率 gate 对应 carrier pass。
```

如果 KAN 仍不形成 early source chain，结论不是“metric FU 失败”，而是：

```text
KAN source-channel writer not found；
continue MLP metric solver until S3 terminal success；
only then remap with stricter channel constraints。
```

---

## 9. 必须记录的总指标

### 9.1 代码/执行指标

```text
compileall_returncode
required_import_error_count
self_contained_import_check
required_source_missing_count
semantic_alias_undeclared_count
queue_nonempty_idle_violation_count
per_gpu_busy_seconds
per_gpu_idle_while_queue_nonempty_seconds
```

### 9.2 效率指标

```text
forward_ms
backward_ms
optimizer_update_ms
metric_solver_ms
functional_commit_ms
LineC_audit_ms
horizon_readback_ms
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
same_kernel_runner_proof
fallback_kernel_used
```

### 9.3 Functional source and training-dynamics indicators

```text
source_h100/h400/h800/h1600/h2400/h3200/h4000/h4800/h6400
R1600_over_800
R3200_over_1600
R4800_over_3200
R6400_over_4800
row_positive_count_by_horizon
control_equivalent_fraction
matched_random_gap
stable_random_gap
source_derivative_h3200_to_h4000
source_derivative_h4000_to_h4800
source_formation_step
source_preservation_step
cumulative_optimizer_projection_on_source
optimizer_state_m_projection_on_source
optimizer_state_v_projection_on_source
fast_slow_function_gap
short_long_source_agreement
erase_energy_h3200_to_h4800
```

### 9.4 Function-space metric and geometry indicators

```text
projection_residual_Gf
ActuationR2
metric_energy_L2
metric_energy_Fisher
metric_energy_PopRisk
metric_energy_Sobolev
metric_energy_RKHS
basis_channel_energy
source_channel_projection
reservoir_projection
source_to_reservoir_leakage
NDS
raw_gradient_NDS
within_block_NDS
cross_block_NDS
first_order_gain
second_order_penalty
info_volume
info_volume_drop_h3200_to_h4800
neighbor_preservation
fold_proxy
Jacobian_condition
row_radial_component
row_angular_component
row_norm_drift
angular_velocity
```

### 9.5 Debt and safety indicators

```text
tail_debt_peak/final/recovery
LineC_debt_peak/final/recovery
ECE_debt_peak/final/recovery
Brier_debt_peak/final/recovery
AUCtime_ratio
loss_spike_count
bad_step_rate
calibration_terminal_erosion_score
classwise_terminal_drop
```

---


## 10. 必须生成的可视化

```text
1. code_truth_dashboard.svg
2. semantic_alias_heatmap.svg
3. function_displacement_alias_heatmap.svg
4. efficiency_dashboard_DCHE_DFOU_DRAT_DRBF.svg
5. source_observability_predictor_auc.svg
6. split_transfer_target_contrast.svg
7. metric_energy_vs_h4800_retention.svg
8. NDS_vs_terminal_erosion.svg
9. Sobolev_RKHS_energy_vs_source.svg
10. projection_residual_vs_B2_transfer.svg
11. source_trajectory_h100_to_h6400.svg
12. terminal_preservation_projection_trace.svg
13. optimizer_cumulative_projection_h3200_to_h4800.svg
14. short_long_source_state_trace.svg
15. debt_transition_h3200_to_h4800.svg
16. signal_channel_reservoir_projection.svg
17. info_volume_and_fold_proxy_trace.svg
18. MLP_source_decomposition_hidden_readout.svg
19. KAN_source_channel_mapping_heatmap.svg
20. D-RAT_component_waterfall.svg
21. D-RBF_component_waterfall.svg
22. GPU_utilization_timeline.svg
```

---

## 11. 失败时的 Codex 自动尝试方向

### 11.1 如果 S0.13 失败

Codex 必须优先修：

```text
missing source files；
self-contained import；
mechanism alias declaration；
kernel gradcheck；
LineC/source-chain/debt tests。
```

不能继续跑 science rows。

### 11.2 如果 C1 source observability 失败

Codex 不允许继续跑昂贵 h4800 solver。按顺序尝试：

```text
1. 增大 train split batch，但不得使用 validation/test。
2. 从 parameter gradient 改成 output cotangent / logit displacement observability。
3. 从 sample-level 改成 block-level：hidden/readout、degree-readout、band-readout。
4. 从 single-step observability 改成 short-window h100 observability。
5. 若仍失败，标记 SourceObservableMissing，停止该 target family。
```

### 11.3 如果 C2 target construction 失败

按顺序尝试：

```text
1. 降低 target norm；
2. 改用 split-transfer target；
3. 改用 signal-channel target；
4. 限制到 low-NDS / low-Sobolev subspace；
5. 若 random/sign-flip/corrupt target 同样好，标记 TargetNotSpecific。
```

### 11.4 如果 C3 projection solver 失败

按顺序尝试：

```text
1. readout-only exact solve；
2. lower rank JVP/VJP sketch；
3. 增大 damping rho；
4. block-restricted solve；
5. diagonal metric fallback；
6. 若仍失败，标记 SolverBlocked，不进入 h4800。
```

### 11.5 如果 C4 source formation 失败

按顺序尝试：

```text
1. 检查 target 是否只在 B1 gain；若 B2 transfer 不过，回到 C1。
2. 检查 optimizer conflict；若强冲突，尝试 optimizer-state transport。
3. 检查 ActuationR2；若低，回到 C3。
4. 检查 metric energy / NDS；若高，切 low-NDS / Sobolev constrained target。
5. 若 MLP 失败，停止 KAN mapping；先修 source observability。
6. 若 MLP 成功而 KAN 失败，进入 Line D source-channel mapping。
```

### 11.6 如果 C5 terminal preservation 失败

根据 autopsy 分类自动尝试：

```text
OptimizerErosion:
  source-preserving optimizer projection；
  optimizer-state transport；
  row-orthogonal block update；
  slow-state commit。

DebtErosion:
  debt-aware terminal FU；
  reduce target norm；
  delay source amplification；
  tail/calibration-safe preservation，但 audit metric 不生成方向。

HighNDS_Erosion:
  low-NDS projection；
  block-balanced update；
  row-angular update。

InfoVolumeCollapse:
  info-volume preservation；
  RKHS/Sobolev smooth target；
  no-fold constraint。

DatasetLocalizedErosion:
  split-transfer reweighting；
  dataset-invariant source estimator；
  若需要 dataset-name branch，则停止。

ControlEquivalentTerminalDrift:
  mark no-go；
  do not tune scale / floor / lookahead。
```

### 11.7 如果 KAN source-channel mapping 失败

按顺序尝试：

```text
1. readout-only commit；
2. basis-estimate / readout-commit；
3. D-CHE low-degree source bank；
4. D-FOU low-frequency source bank；
5. degree-readout / band-readout block solve；
6. 若仍无 early chain，标记 KANSourceChannelWriterMissing。
```

### 11.8 如果 D-RAT/D-RBF official fused 失败

按原 Line B 执行，不改变 functional gate：

```text
D-RAT:
  numerator / denominator / reciprocal / safety / telemetry component repair。

D-RBF:
  active-center / local gather / exp approx / no-dense materialization repair。
```

---

## 12. v22.06 成功标准

### 12.1 Minimum success

```text
S0.13 pass；
D-CHE / D-FOU efficiency reconfirm pass；
D-RAT or D-RBF at least one production fused official pass or clear component blocker；
C1 source observability produces at least one legal target family；
C3 metric solver reaches actuation gate for at least one metric/target/solver；
all raw matrices and figures complete。
```

### 12.2 Functional algorithmic progress

```text
C1 train-only source estimator predicts h3200 or R4800_over_3200 better than random；
C2 target beats random/sign-flip/corrupt target on B2 transfer；
C3 projection residual beats flat proxy by >= 20%；
C4 produces h100/h400/h800 early source chain；
semantic alias audit shows mechanisms are not identical proxies。
```

### 12.3 Functional exploration success

```text
source_h100/h400/h800 all >= 0.005；
source_h1600/h3200 >= 0.005；
row_h3200_positive_count >= 6/9；
matched controls fail；
stable-random controls fail；
tail/LineC/ECE/Brier debt not exploded。
```

### 12.4 Productive terminal source

```text
source_h4800 >= 0.005；
R4800_over_3200 >= 0.50；
row_h4800_positive_count >= 7/9；
matched controls fail；
stable-random controls fail；
AUCtime_ratio <= 1.05；
LineC / ECE / Brier / tail debt not worse；
independent init-offset rerun pass。
```

### 12.5 KAN-specific success

```text
KAN productive terminal source pass；
KAN_specific_delta_vs_MLP_same_metric >= 0.005；
D-CHE / D-FOU / D-RAT / D-RBF corresponding efficiency gate pass；
source-channel mapping shows low-degree / low-frequency / readout channel carries source；
controls fail；
no audit-metric-directed update。
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

v22.06 的关键词不是“更多 metric variants”，也不是“继续堆 Fxx”。真正关键词是：

$$
\boxed{
\text{Functional update 必须从工程化 update 搜索，升级为训练动力学中的 source-channel 控制算法。}
}
$$

v22.05 的负结果不能写成“metric 没用”。它只能说明 flat-gradient metric proxy 没用。v22.06 必须真正回答：

```text
1. train-stream 是否能看见 retained source？
2. Δf_target 是否跨 split transfer？
3. Jθu 是否能在函数空间 metric 下实现 target？
4. optimizer state 是否会接纳还是洗掉 source？
5. h3200 source 是否能被保护到 h4800？
6. KAN 是否有对应 source-channel？
```

如果 v22.06 仍失败，也必须输出明确路线：

```text
SourceObservableMissing；
TargetNotSpecific；
SolverBlocked；
OptimizerErosion；
DebtErosion；
HighNDS_Erosion；
InfoVolumeCollapse；
KANSourceChannelWriterMissing；
MetricProxyNoGo。
```

这才能避免继续“跑了很多机制，但不知道为什么失败”。本版 functional update 的底线是：先把问题变成可诊断的训练动力学系统，再讨论 KAN-specific breakthrough。
