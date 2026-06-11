# DG-KAN v22.09：Retained-Source Observability / Metric-as-Dynamics Functional Update / Basis Efficiency Closure / 4GPU 完整计划

> 版本：v22.09 execution plan  
> 生成时间：2026-06-06  
> 本计划基于 v22.08 实验结果、v22.07/v22.06 的失败边界、D-CHE/D-FOU/D-RAT/D-RBF 效率线进展，以及近期优化器 / 泛化 / 神经网络流形研究启发。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler/class weight；no dataset-name branch；no seed-specific rule；functional direction 不使用 validation/test/future/query；LineC、CEp99、NLL、ECE、Brier、AUCtime 只能做 audit、debt readback 和 gate，不能生成方向。

---

## 0. 项目总目标、当前真实状态与 v22.09 的定位

DG-KAN 项目的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是证明某个局部指标更好，而是建立一个 strict FC-PureKAN 训练系统：

$$
\boxed{
\text{PureKAN base + functional update}
>
\text{PureKAN base + ordinary backprop / AdamW controls}
}
$$

这个“大于”必须同时包含：

```text
1. task 表现不弱于同参数 MLP；
2. forward / backward / optimizer step / memory 接近同参数 MLP；
3. functional update 的收益超过 AdamW / SGD / NoOp / random / stable-random / same-overhead controls；
4. source 能从早期留到长期 horizon，而不是 h800 或 h3200 局部 positive；
5. tail / LineC / calibration / AUC debt 不爆；
6. 若 MLP 也成功，必须写成 generic training-dynamics insight，不能写成 KAN-specific；
7. 若 KAN 比同机制 MLP 更强，才可以讨论 KAN-specific advantage。
```

v22.08 的真实 route 是：

```text
route = R-PostNoGoTrainFlowCommutatorRepeatVerifyBlocked
promotion_allowed = 0
D-CHE / D-FOU S1 pass = 1 / 1
D-RAT / D-RBF multibatch closed = 0
C0 retained-source observer pass rows = 0
official early-chain positive rows = 0
retained h800+h3200 positive rows = 13
C2 true block solver pass rows = 0
C3 source formation pass rows = 0
post-no-go C-O12 train-flow commutator fresh C3 pass rows = 1
post-no-go C-O12 repeat verify robust fresh rows = 0
terminal preservation not entered
KAN mapping not entered
```

这说明 v22.08 不是能力突破。它把 blocker 进一步定位为：

$$
\boxed{
\text{我们还没有合法的 train-only retained-source certificate。}
}
$$

也就是说，现在不是单纯缺 metric、缺 solver、缺 actuation，也不是缺更多 Fxx / C-Ox 名字。当前缺的是一个能在提交 update 之前判断“这个方向会形成 retained source”的可部署观测原理。

v22.09 因此不继续扩大 C-O1..C-O12 同族 observer，不继续堆 metric 名字，不继续调 target scale / cap / floor。v22.09 的核心目标是：

$$
\boxed{
\text{从 one-step actuation / metric proxy / observer feature 组合，}
\text{升级到 retained-source training dynamics。}
}
$$

换句话说，Functional Update 应该从“某一步参数怎么动”变成“训练动力系统怎样被边界条件、函数空间 metric、optimizer state 和 source channel 共同塑形”。

---

## 1. v22.08 结果的独立问题诊断

### 1.1 代码审计问题：结果可分析，但 code packet 不可写成完全自包含

v22.08 的结果 bundle 中 `v22_08_required_source_files.csv` 写了 39 个 required files 全部存在，`v22_08_import_closure.csv` 写 `self_contained_import_check=1`。但对用户收到的 `v22_08_code_review_packet.zip` 解压后独立检查，实际只包含 21 个 `.py` 文件，缺少大量 required source：

```text
dgkan/fu/source_chain.py
dgkan/fu/terminal_retention.py
dgkan/fu/debt_accounting.py
dgkan/fu/function_space_metrics.py
dgkan/fu/metric_projection.py
dgkan/fu/jacobian_sketch.py
dgkan/fu/sobolev_metric.py
dgkan/fu/rkhs_metric.py
dgkan/fu/fisher_metric.py
dgkan/fu/basis_channel_metric.py
dgkan/metrics/linec.py
dgkan/kernels/fused_chebyshev_k3.py
dgkan/kernels/fused_fourier_k2.py
dgkan/kernels/fused_rational_k4.py
dgkan/kernels/fused_rbf.py
dgkan/kernels/rational_fused.py
dgkan/kernels/rbf_sparse.py
dgkan/profiling/efficiency_v22_06.py
experiments/run_v22_06_common.py
experiments/run_v22_06_s013_truth_gate.py
experiments/run_v22_06_metric_solver_fu.py
experiments/run_v22_06_terminal_preservation.py
experiments/run_v22_06_kan_source_mapping.py
experiments/run_v22_06_drat_drbf_officialization.py
experiments/run_v22_06_finalize.py
```

独立 import scan 发现至少 12 个核心模块 import 失败，例如：

```text
dgkan.fu.mechanisms:
  No module named 'dgkan.fu.function_space_metrics'

dgkan.fu.metric_solver:
  No module named 'dgkan.fu.basis_channel_metric'

experiments.run_v22_08_drat_drbf_multibatch:
  cannot import run_v22_07_drat_drbf_multibatch
```

因此，v22.09 的第一硬门是：

$$
\boxed{
\text{CodeRoute pass 必须基于最终 zip 解压目录，而不是原始 repo 环境。}
}
$$

如果 final zip 不能独立 import required path，不能写 scientific no-go，也不能写 implementation-correct no-go。

### 1.2 效率线结果：D-CHE/D-FOU 站住，D-RAT/D-RBF 仍未 multibatch official closure

v22.08 对 D-CHE / D-FOU 继续给出 full-loop efficient carrier 证据：

```text
D-CHE:
  S1_pass_rows = 6
  best_forward_ratio = 0.9305
  best_step_ratio = 0.4668
  best_memory_ratio = 0.9913
  decision = OfficialEfficientCarrier

D-FOU:
  S1_pass_rows = 3
  best_forward_ratio = 1.0087
  best_step_ratio = 0.5089
  best_memory_ratio = 0.9888
  decision = OfficialEfficientCarrier
```

这意味着 D-CHE / D-FOU 不再是当前 Functional Update 失败的主要 runtime blocker。

D-RAT / D-RBF 有 near-E1 进展，但没有 robust official closure：

```text
D-RAT:
  near_E1_rows = 4
  robust_production_pass_rows = 0
  best_forward_ratio = 1.6740
  best_step_ratio = 0.5053
  decision = NearE1LimitedSmokeAllowed
  blocker = forward_ratio; component_telemetry_incomplete

D-RBF:
  near_E1_rows = 4
  robust_production_pass_rows = 0
  best_forward_ratio = 1.7653
  best_step_ratio = 0.6422
  decision = NearE1LimitedSmokeAllowed
  blocker = forward_ratio; component_telemetry_incomplete
```

repair attempts 之后仍是：

```text
D-RAT:
  MicroNearE1RunnerBlocked
  blocker = official_fused_missing; functional_runner_kernel_mismatch

D-RBF:
  MicroNearE1RunnerBlocked
  blocker = official_fused_missing; functional_runner_kernel_mismatch
```

所以 v22.09 的效率策略是：

```text
D-CHE / D-FOU:
  只做 reconfirm + metric/FU runner integration，不再重复大 census。

D-RAT / D-RBF:
  做 robust multibatch officialization + component telemetry completion。
  在 >=3/4 batch sizes 过 official gate 之前，不进入 full functional matrix。
```

### 1.3 Functional Update 结果：C-O12 有 fresh 迹象，但 repeat verification 失败

v22.08 的基本 observer family 全部没有 official early-chain label：

```text
C-O1..C-O5:
  positive_official_early_chain_rows = 0
  C0_retained_source_observer_pass = 0
```

post-no-go C-O6..C-O11 也一样：train-only pass rows 可以很多，但 fresh C3 / official C3 全部为 0。

唯一值得严肃看的是 C-O12 Train-Flow Commutator：

```text
C-O12 train-only pass rows = 48
fresh C3 pass rows = 1
official C3 pass rows = 0
verify route = TrainFlowCommutatorRepeatVerifyBlocked
repeat robust fresh rows = 0
```

这说明 C-O12 不是成功，但它比 C-O1..C-O11 更接近真正的训练动力学：它不是只看一个静态 feature，而是看 train flows 的组合关系。它的失败也很关键：fresh smoke 有一次打开，但 repeat 不稳。v22.09 不应该继续 C-O12 小修，而应该把它提升为更严谨的 **Train-Flow Algebra Certificate**：不要只看一个 commutator proxy，而要看多 split、多 micro-horizon、多 optimizer state 下的 vector field consistency。

---

## 2. 第一性原则：Functional Update 应该是什么

### 2.1 普通 optimizer 做什么

普通训练每一步做：

$$
\theta_{t+1}=\theta_t-\eta g_t.
$$

AdamW / SGD / Momentum 主要回答：

```text
当前 batch loss 的梯度方向怎么变成参数更新？
```

### 2.2 Functional Update 不应只是另一个 optimizer

Functional Update 的目标不是替代 AdamW 的 step，也不是在梯度上加一个小 residual。它要回答：

```text
模型作为函数 f_θ，应该朝哪个 train-only 可观测的方向移动，
这个移动会不会形成 retained source，
并且这个 source 会不会进入长期 signal channel？
```

因此更正确的对象是函数位移：

$$
\Delta f = J_\theta u.
$$

一个合格的 FU 不应该只满足：

```text
ActuationR2 高；
当前 B2 transfer gain 好；
h3200 某些 row positive。
```

它必须满足：

```text
1. source observer 在提交前合法可见；
2. target 是 train-only 且可 transfer；
3. solver 真正让 Jθu 接近目标；
4. update 写入 optimizer / slow state / block coordinate，而不是孤立一步；
5. h100/h400/h800/h1600/h3200/h4800 形成 retained chain；
6. debt 不爆，controls 解释不了。
```

### 2.3 现阶段最重要的问题不是 metric，而是 source observability

v22.05 证明 metric-as-flat-filter 没用。v22.06 证明 readout exact solve 能做 actuation，但不能形成 early source。v22.07 证明 C-O1..C-O7 observer family local no-go。v22.08 证明 C-O1..C-O12 中只有 train-flow commutator 有一次 fresh C3 迹象，但 repeat verify 失败。

所以现在最核心的问题是：

$$
\boxed{
\text{什么 train-only 信号能在提交前证明一个方向会形成 retained source？}
}
$$

metric 不是没用，但 metric 必须服务于这个问题。没有 retained-source certificate，metric solver 只会更精确地执行一个错误 target。

---

## 3. 当前研究进展转化为算法原则

### 3.1 Generalization signal-channel / reservoir

相关泛化理论把输出空间沿训练轨迹分成 signal channel 与 reservoir。对我们而言，这意味着 FU 的目标不是“让输出动”，而是：

$$
\boxed{
\text{让 coherent train-stream signal 进入 signal channel，}
\text{同时避免 noise 进入 signal channel。}
}
$$

实验化为：

```text
1. 记录 source displacement 是否位于 train-flow 可重复子空间；
2. 记录 stable-random / sign-flip / corrupt target 是否也能解释同一 source；
3. 记录 noise-reservoir leakage；
4. 只允许 signal-channel positive 且 control-nullspace positive 的 target 进入 solver。
```

### 3.2 Deep Manifold / boundary-conditioned iteration

Deep Manifold 视角提醒我们：神经网络不是静态 fixed point，而是在训练中通过边界条件与残差迭代构造 fixed-point region。FU 因此不应是孤立动作，而应是训练动力系统的边界条件控制。

实验化为：

```text
1. FU 不是一次 commit 后结束；
2. FU 必须写入 optimizer state / slow source state / block coordinate；
3. 观察 train flow 在 B1/B2/B3 split 下的组合是否稳定；
4. 记录 flow commutator、flow associativity、flow path invariance。
```

### 3.3 Muon / NDS 曲率视角

Muon curvature 研究说明，一阶 gain 相近时，长期差异可能来自二阶曲率惩罚和 Normalized Directional Sharpness。对应到 FU：

```text
source_h800 / h3200 高不够；
如果 NDS 高，h4800 仍可能被 erosion；
如果 source observer 选择的是高曲率方向，metric solver 可能越精确越糟。
```

v22.09 记录：

$$
I^{(1)} = -g^T u,
$$

$$
I^{(2)} = \frac{1}{2}u^T H u,
$$

$$
\text{NDS}(u)=\frac{u^T H u}{\|u\|^2+\epsilon}.
$$

### 3.4 Nora / row-wise angular stability

Nora 的启发不是“把 optimizer 换成 Nora”，而是矩阵 block 的 source update 不应该引发 row norm jitter 和 angular velocity 爆炸。

v22.09 记录：

```text
row_norm_drift
row_angular_velocity
radial_source_component
tangential_source_component
source_retention_delta_after_row_orthogonal_projection
```

### 3.5 SOAP / block coordinate basis

SOAP 的启发是：moment / preconditioner 不一定应该在 flat parameter coordinate 中运行，而应该在 slowly-changing block/eigenbasis 中运行。对应到 DG-KAN：

```text
MLP:
  hidden block / readout block

D-CHE:
  low-degree / high-degree / degree-readout block

D-FOU:
  low-frequency / high-frequency / band-readout block

D-RAT:
  numerator / denominator / readout block

D-RBF:
  local-center / width / readout block
```

### 3.6 AdEMAMix / Schedule-Free / fast-slow memory

h3200 到 h4800 erosion 说明 source 需要 slow memory，而 source formation 又需要 fast response。v22.09 不再只用一个 EMA，而是显式记录：

```text
fast_source_state
slow_source_state
old_gradient_source_alignment
source_state_age
state overwrite fraction
```

### 3.7 Constrained inference manifolds / information volume

低维 source path 不等于好 source。如果 source 被压缩成退化方向，它可能 h3200 好但 h4800 不稳。v22.09 记录：

```text
source_path_intrinsic_dim
source_info_volume
subspace_condition_number
fold_proxy
neighbor_preservation
```

---

## 4. 旧路线的重新分类：哪些可以重开，哪些不能再耗预算

### 4.1 可以在新语义下重开的路线

#### 4.1.1 v15.02.1 function metric

旧结论是 function-metric fallback 没打开，control-equivalent 高。但当时 metric 多半是 proxy 或 update token，没有进入 retained-source observer 与 metric solver。现在可重开，但只允许作为：

```text
function-space metric operator + solver energy + terminal preservation audit
```

不能再作为：

```text
flat-gradient metric filter
```

#### 4.1.2 v13.2 / v13.3 basis natural update

真实 basis-parameter writeback 和 low-rank/block natural metric 值得重开。现在 D-CHE / D-FOU efficiency 已经站住，基础条件比当时更好。重开方式：

```text
basis-channel metric pullback；
low-degree / low-frequency source bank；
operator-level KAN source mapping。
```

#### 4.1.3 v13.4 operator-level basis-channel target

这是比很多 Fxx trick 更接近真正 FU 的路线。重开方式：

```text
operator-level target -> metric energy -> Jθ solver -> source-chain gate
```

#### 4.1.4 v13.6 / v13.7 PopRisk/SNR

旧 PopRisk/SNR one-shot 失败，后续也有负相关，但 population-risk / drift-diffusion 思想不能判死。重开方式：

```text
coherent drift / diffusion observer；
source-state certificate；
not direct mask / not one-shot update。
```

#### 4.1.5 C-O12 train-flow commutator

v22.08 中 C-O12 是唯一出现 fresh C3 pass 的 post-no-go family，但 repeat failed。它不能直接继续小修，但应升级为 Train-Flow Algebra Certificate。

### 4.2 不应再重启的路线

```text
action bank / controller；
reset route；
cover objective 小修；
M31/M32 同族小修；
G/N/Q-token 名字扩展；
more terminal floor / lookahead / scale / hold；
ActuationR2-only target；
LineC / ECE / Brier directed update；
dataset/seed branch。
```

原因：这些路线过去已经多次显示 control-equivalent、oracle upper-bound insufficient、cover objective invalid、或不能 independent confirmation。继续投入会制造“看起来很多实验”的错觉。

---

## 5. v22.09 总体实验结构

v22.09 分为五条线：

```text
Line A: Code / metric / solver / packaging truth gate
Line B: Basis efficiency closure
Line C: Retained-source observability reset
Line D: Metric-as-dynamics solver
Line E: KAN source-channel mapping
```

四张 GPU 的默认动态队列：

```text
GPU0:
  Line A truth gate；
  D-CHE/D-FOU reconfirm；
  MLP observer C0/C1。

GPU1:
  retained-source observer C-O13/C-O14/C-O15；
  metric solver C2/C3；
  MLP terminal preservation。

GPU2:
  D-RAT/D-RBF multibatch officialization；
  D-RAT/D-RBF component telemetry；
  limited smoke if near-E1.

GPU3:
  KAN source-channel mapping；
  controls/fresh repeat；
  figures/final packet.
```

必须输出：

```text
v22_09_runnable_queue.csv
v22_09_gpu_assignment_manifest.csv
v22_09_gpu_utilization_timeline.csv
v22_09_idle_violation.csv
v22_09_deferred_items.csv
v22_09_queue_drain_report.csv
```

如果 runnable queue 非空且任一 GPU idle 超过 10 分钟：

```text
execution_contract_violation = 1
final route 不能写 completed no-go
```

---

## 6. Line A：Code / Metric / Solver / Packaging Truth Gate

### 6.1 目的

修复 v22.08 最大实现问题：final code packet 不自包含，但 result CSV 声称 required files 存在。

### 6.2 必须打包的源码

Codex 必须生成：

```text
v22_09_code_review_packet.zip
```

并且最终 zip 解压目录必须包含至少：

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
dgkan/fu/retained_source_certificate.py
dgkan/fu/train_flow_commutator.py
dgkan/fu/source_state_dynamics.py
dgkan/fu/optimizer_state_integration.py

dgkan/metrics/linec.py

dgkan/kernels/fused_chebyshev_k3.py
dgkan/kernels/fused_fourier_k2.py
dgkan/kernels/fused_rational_k4.py
dgkan/kernels/fused_rbf.py
dgkan/kernels/rational_fused.py
dgkan/kernels/rbf_sparse.py

dgkan/profiling/efficiency_v22_06.py
dgkan/profiling/efficiency_v22_09.py

experiments/run_v22_09_common.py
experiments/run_v22_09_s016_truth_gate.py
experiments/run_v22_09_basis_efficiency_closure.py
experiments/run_v22_09_retained_source_observer.py
experiments/run_v22_09_metric_dynamics_solver.py
experiments/run_v22_09_terminal_preservation.py
experiments/run_v22_09_kan_mapping.py
experiments/run_v22_09_finalize.py
```

### 6.3 自包含测试

测试必须在解压后的 zip 目录里执行，而不是在原始 repo 中执行。

```bash
python -m compileall -q dgkan experiments tests
python experiments/run_v22_09_s016_truth_gate.py --self-contained-check 1
```

### 6.4 必须记录

```text
required_source_files.csv
packet_manifest.csv
packet_sha256_manifest.csv
compileall.csv
import_closure.csv
self_contained_import_check
missing_required_files
csv_claimed_exists_but_zip_missing_count
legacy_import_errors_excluded
forbidden_direction_audit.csv
semantic_contract.csv
semantic_alias_matrix.csv
function_displacement_alias_matrix.csv
```

### 6.5 Gate

```text
required_source_files_exist = 1
csv_claimed_exists_but_zip_missing_count = 0
compileall_pass = 1
required_import_errors = 0
self_contained_import_check = 1
forbidden_direction_violation = 0
undeclared_semantic_alias_pairs = 0
```

如果任一不满足：

```text
CodeRoute = R0-CodePacketNotSelfContained
scientific_route = blocked
```

---

## 7. Line B：Basis efficiency closure

### 7.1 目标

保持 D-CHE / D-FOU 主 carrier 状态，同时推动 D-RAT / D-RBF 从 near-E1 到 robust multibatch official closure。

### 7.2 D-CHE / D-FOU reconfirm

候选：

```text
D-CHE:
  CHE21-R2-low-degree-k3
  CHE21-R4-k5-gradbuf

D-FOU:
  FOU21-R3-tablelookup-bandreadout
```

记录：

```text
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
functional_runner_same_kernel
fallback_kernel_used
LineC_audit_ms
horizon_readback_ms
metric_solver_overhead_ms
```

Gate：

```text
forward <= 1.25
step <= 1.25
memory <= 1.05
same_kernel_functional_runner_proof = 1
fallback_kernel_used = 0
```

### 7.3 D-RAT robust officialization

D-RAT 现在不是完全 blocked，而是 multibatch robust 未闭合。v22.09 必须拆 component telemetry：

```text
numerator_eval_ms
denominator_eval_ms
reciprocal_ms
numden_fused_ms
safety_clamp_ms
derivative_telemetry_ms
train_path_without_telemetry_ms
component_sum_vs_total_error
```

Batch grid：

```text
batch = 128, 256, 512, 1024
```

Gate：

```text
>=3/4 batch sizes:
  forward <= 1.25
  step <= 1.25
  memory <= 1.05
component_telemetry_complete = 1
functional_runner_kernel_match = 1
official_fused_kernel_complete = 1
```

如果 fail：

```text
D-RAT decision = NearE1LimitedSmokeOnly
D-RAT not allowed into full FU matrix
```

### 7.4 D-RBF robust officialization

记录：

```text
active_center_fraction
mean_local_K
local_gather_ms
exp_eval_ms
local_backward_ms
dense_materialization_bytes
no_dense_materialization_proof
component_sum_vs_total_error
```

Gate 与 D-RAT 相同。

### 7.5 D-RAT / D-RBF limited functional smoke

只有满足 near-E1 且 gradcheck pass，但未 official closure 时，允许 limited smoke：

```text
scope = MNIST seed0 only
horizon = h800 / h1600
no promotion
```

目的只判断是否完全 no-source，不允许写 success。

---

## 8. Line C：Retained-source observability reset

### 8.1 为什么要 reset observer

v22.07/v22.08 证明：C-O1..C-O12 这些 observer family 中，大多数只在 train-only retrospective 或 selected candidate 上有局部信号，fresh C3 / official C3 不成立。继续调同族 observer 是局部 no-go。

v22.09 只允许三个新 observer 原理进入主预算：

```text
C-O13 Train-Flow Algebra Certificate
C-O14 Drift-Diffusion Signal-Channel Certificate
C-O15 Block-Coordinate Source-State Certificate
```

其它 observer 只能作为 audit baseline。

---

### 8.2 C-O13：Train-Flow Algebra Certificate

#### 8.2.1 第一性动机

如果一个方向是真正 train-flow stable 的 source，它应该不强依赖某个 split 的顺序。对两个 train split $B_1,B_2$，定义微训练流：

$$
F_{B}(\theta)=\theta+U_B(\theta),
$$

其中 $U_B$ 是若干合法 train steps 或 solver commit。定义交换子：

$$
\mathcal{K}_{12}(\theta)
=
F_{B_2}(F_{B_1}(\theta))
-
F_{B_1}(F_{B_2}(\theta)).
$$

如果 $\|\mathcal{K}_{12}\|$ 很大，说明更新方向依赖 batch 顺序，可能是噪声或局部路径 artifact。若 commutator 小且 source drift 大，才可能形成 retained source。

#### 8.2.2 记录指标

```text
commutator_norm
commutator_norm_ratio
commutator_function_cosine
B1_then_B2_source_gain
B2_then_B1_source_gain
flow_order_gap
multi_split_associativity_error
micro_horizon_stability_h1_h2_h4_h8
optimizer_state_commutator
source_direction_reproducibility
```

#### 8.2.3 Gate

C-O13 observer pass 必须满足：

```text
AUC_predict_official_early_chain >= 0.75
precision_at_top20 >= 0.50
heldout_precision_at_top20 >= 0.40
commutator_norm_ratio <= 0.25
flow_order_gap <= 0.05
control_equivalent_fraction <= 0.10
random/sign/corrupt gap positive
```

如果 C-O13 有 fresh C3 pass，必须 independent repeat：

```text
repeat_count >= 3
robust_fresh_C3_pass_rows >= 2/3
```

---

### 8.3 C-O14：Drift-Diffusion Signal-Channel Certificate

#### 8.3.1 第一性动机

一个 retained source 应该表现为 coherent drift 大于 idiosyncratic diffusion。对 per-example function displacement $d_i$：

$$
\mu = \frac{1}{n}\sum_i d_i,
$$

$$
\sigma^2 = \frac{1}{n-1}\sum_i \|d_i-\mu\|^2.
$$

定义：

$$
\text{DDR} = \frac{\|P_{sig}\mu\|^2}{\sigma^2 + \epsilon}.
$$

其中 $P_{sig}$ 是 train-only signal-channel projection，不允许用 validation/test/future。

#### 8.3.2 Signal-channel projection 的合法构造

候选：

```text
split-consensus tangent basis；
train-flow low-commutator subspace；
per-example gradient drift subspace；
block-coordinate source state subspace；
control-null residual subspace。
```

不能使用：

```text
LineC hard label;
validation/test;
future horizon outcome;
dataset-name branch;
seed-specific scale。
```

#### 8.3.3 记录指标

```text
drift_norm
diffusion_norm
DDR
signal_channel_energy
reservoir_energy
noise_into_signal_ratio
source_minus_control_gap
split_consensus_cosine
source_direction_rank
source_info_volume
```

#### 8.3.4 Gate

```text
DDR top20 precision >= 0.50
heldout precision >= 0.40
noise_into_signal_ratio <= 0.25
control-null residual source positive
```

---

### 8.4 C-O15：Block-Coordinate Source-State Certificate

#### 8.4.1 动机

Source 不是 flat 向量，而应该在 block coordinate 中持续存在。对应不同 carrier：

```text
MLP:
  hidden block / readout block

D-CHE:
  low-degree bank / degree-readout / readout

D-FOU:
  low-frequency bank / band-readout / readout

D-RAT:
  numerator / denominator / readout

D-RBF:
  local-center / width / readout
```

#### 8.4.2 记录指标

```text
block_source_energy
block_source_SNR
block_source_age
block_source_decay_rate
block_source_overwrite_fraction
row_norm_drift
row_angular_velocity
radial_component
tangential_component
source_state_alignment_with_optimizer_momentum
```

#### 8.4.3 Gate

```text
block source state predicts early-chain AUC >= 0.75
block source age survives >= h800
source overwrite fraction <= 0.40
row angular velocity stable
```

---

## 9. Line D：Metric-as-dynamics solver

### 9.1 核心变化

旧 solver 是：

```text
C0 observer -> C1 target -> C2 solver -> C3 source formation
```

但 v22.07/v22.08 说明，如果 C0 不可靠，C2/C3 没意义。v22.09 的顺序是：

```text
C0 retained-source certificate
-> C1 target contrast
-> C2 solver with block recompute
-> C3 optimizer-state integrated source formation
-> C4 terminal preservation
```

### 9.2 C1 target contrast

Target family：

```text
T0 loss-cotangent
T1 split-transfer
T2 drift-diffusion signal target
T3 train-flow algebra target
T4 block-source-state target
T5 low-NDS target
T6 information-volume-preserving target
T7 control-null residual target
```

Controls：

```text
random matched target
sign-flipped target
corrupt target
stable-random target
same-ActuationR2 random target
same-DDR random target
```

C1 pass：

```text
B2_transfer_gain > random by >= 0.02
B3_safety_gain >= -0.02
random/sign/corrupt controls fail
not control equivalent
```

### 9.3 C2 solver

Solver levels：

```text
S1 readout exact solve
S2 hidden/readout joint block CG
S3 block Gauss-Newton / low-rank CG
S4 optimizer-state integrated solve
S5 KAN basis-channel solve
```

C2 必须为每个 target/block 独立 recompute，不能复用旧 diagnostics。

记录：

```text
projection_residual_Gf
ActuationR2
B2_transfer_gain
B3_safety_gain
JVP_count
VJP_count
CG_iterations
condition_estimate
solve_time_ms
parameter_update_norm
function_displacement_norm
metric_energy_L2
metric_energy_Fisher
metric_energy_Sobolev
metric_energy_RKHS
NDS
```

C2 pass：

```text
projection_residual_Gf <= 0.50
ActuationR2 >= 0.20
B2_transfer_gain >= 0.02
solve_time_ms <= budget
```

### 9.4 C3 source formation

C3 不再只是 direct commit。必须比较：

```text
I0 direct parameter commit
I1 optimizer momentum injection
I2 slow source-state injection
I3 block-coordinate source memory
I4 schedule-free fast/slow iterate injection
I5 row-orthogonal matrix update
```

C3 pass：

```text
source_h100 >= 0.005
source_h400 >= 0.005
source_h800 >= 0.005
source_h1600 >= 0.005
source_h3200 >= 0.005
row_h3200_positive_count >= 6/9
matched controls fail
debt not exploded
```

### 9.5 C4 terminal preservation

只有 C3 过才跑。

记录：

```text
source_h4000
source_h4800
source_h6400
R4800_over_3200
R6400_over_3200
optimizer_projection_on_source
source_state_decay
source_state_overwrite
tail/LineC/ECE/Brier debt transition
```

C4 pass：

```text
source_h4800 >= 0.005
R4800_over_3200 >= 0.50
row_h4800_positive_count >= 7/9
stable-random control fails
AUCtime <= 1.05
debt not worse
```

---

## 10. Line E：KAN source-channel mapping

只有 MLP C3 过后进入 KAN mapping。

### 10.1 MLP source decomposition

记录：

```text
hidden_source_energy
readout_source_energy
source_subspace_rank
source_singular_values
source_reconstruction_error
hidden/readout function displacement cosine
source info volume
row angular velocity
```

### 10.2 KAN mapping candidates

```text
D-CHE:
  low-degree bank
  degree-readout
  readout-only
  basis-estimate/readout-commit
  low-NDS degree block

D-FOU:
  low-frequency bank
  band-readout
  readout-only
  basis-estimate/readout-commit
  low-NDS band block

D-RAT:
  numerator block
  denominator block
  numerator/denominator tangent block
  readout-only

D-RBF:
  local-center block
  width block
  local-center/readout block
```

### 10.3 KAN gate

```text
KAN source_h100/h400/h800 >= 0.005
KAN source_h1600/h3200 >= 0.005
R4800_over_3200 >= 0.50
KAN_specific_delta_vs_MLP_same_metric >= 0.005
controls fail
```

If MLP C3 pass but KAN fails：

```text
KANSourceChannelMismatchConfirmed
```

If MLP C3 fails：

```text
KANMappingNotEntered
```

---

## 11. 必须记录的统一指标

### 11.1 Code / provenance

```text
required_source_files
packet_manifest
sha256_manifest
compileall
import_closure
self_contained_import_check
legacy_import_error_excluded_count
forbidden_direction_violation_count
semantic_alias_pairs
undeclared_alias_pairs
```

### 11.2 Efficiency

```text
forward_ms
backward_ms
step_ms
memory_peak
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
functional_direction_ms
metric_solver_ms
LineC_audit_ms
horizon_readback_ms
component_telemetry_complete
functional_runner_kernel_match
```

### 11.3 Functional source

```text
source_h100
source_h400
source_h800
source_h1600
source_h2400
source_h3200
source_h4000
source_h4800
source_h6400
R4800_over_3200
row_h3200_positive_count
row_h4800_positive_count
control_equivalent_fraction
```

### 11.4 Observer

```text
AUC_predict_early_chain
AUC_predict_h3200
AUC_predict_h4800
precision_at_top20
heldout_precision_at_top20
random/sign/corrupt_gap
commutator_norm_ratio
DDR
source_info_volume
source_path_intrinsic_dim
```

### 11.5 Solver

```text
projection_residual_Gf
ActuationR2
B2_transfer_gain
B3_safety_gain
JVP_count
VJP_count
CG_iterations
condition_estimate
metric_energy_L2/Fisher/Sobolev/RKHS
NDS
solve_time_ms
```

### 11.6 Debt / safety

```text
tail_debt_peak/final/recovery
LineC_debt_peak/final/recovery
ECE_debt_peak/final/recovery
Brier_debt_peak/final/recovery
AUCtime_ratio
classwise_source_loss
```

---

## 12. 必须可视化

```text
1. basis efficiency dashboard:
   D-CHE / D-FOU / D-RAT / D-RBF forward/backward/step/memory.

2. D-RAT / D-RBF component waterfall:
   numerator / denominator / reciprocal / local K / exp / telemetry / readout.

3. retained-source observer dashboard:
   AUC / precision@20 / heldout precision / random gap.

4. train-flow commutator plot:
   commutator_norm_ratio vs h3200 source.

5. drift-diffusion plot:
   DDR vs h3200/h4800 source.

6. solver plot:
   projection_residual_Gf vs ActuationR2 vs B2 gain.

7. source horizon trajectory:
   h100..h6400 per candidate.

8. source-state overwrite plot:
   source_state_age / overwrite_fraction / h4800 retention.

9. KAN mapping plot:
   MLP source subspace vs KAN low-degree/low-frequency/readout projection.

10. 4GPU timeline:
   start/end/busy/idle per GPU, not final snapshot only.
```

---

## 13. 判断标准与失败后 Codex 先尝试方向

### 13.1 如果 code packet 不自包含

```text
route = R0-CodePacketNotSelfContained
next_codex_action:
  rebuild packet from final unzip root;
  compare required_source_files.csv against actual zip entries;
  rerun compile/import inside temp unzip dir.
```

### 13.2 如果 D-CHE / D-FOU efficiency regression

```text
next_codex_action:
  check fallback kernel path;
  compare profiler path vs functional runner path;
  separate metric_solver overhead from training step;
  rerun batch 32/128/256.
```

### 13.3 如果 D-RAT / D-RBF multibatch still blocked

```text
next_codex_action:
  D-RAT: component telemetry -> remove derivative telemetry from train path -> rational-k4 fused num/den reciprocal.
  D-RBF: local K active mask -> no dense materialization proof -> exp approximation -> local backward fused.
```

### 13.4 如果 C-O13/C-O14/C-O15 all fail

```text
route = RetainedSourceObserverLocalNoGo_v22.09
next_codex_action:
  stop C-O family variants;
  write observer no-go boundary;
  propose a new first-principles certificate, not more thresholds.
```

### 13.5 如果 C0 passes but C2 fails

```text
next_codex_action:
  reduce target rank;
  block-restricted solve;
  increase damping;
  use low-rank CG;
  do not enter C3.
```

### 13.6 如果 C2 passes but C3 fails

```text
next_codex_action:
  compare direct commit vs optimizer-state injection;
  add slow source-state;
  test block-coordinate commit;
  analyze source overwritten by optimizer.
```

### 13.7 If MLP C3 passes but C4 fails

```text
next_codex_action:
  terminal erosion autopsy;
  source-preserving late projection;
  slow source-state preservation;
  row-orthogonal / low-NDS update.
```

### 13.8 If MLP C3/C4 pass but KAN fails

```text
route = KANSourceChannelMismatchConfirmed
next_codex_action:
  source subspace decomposition;
  D-CHE low-degree vs readout commit;
  D-FOU low-frequency vs band-readout commit;
  do not copy MLP update directly.
```

---

## 14. v22.09 最终交付物

Codex 必须打包：

```text
v22_09_code_review_packet.zip
v22_09_results_bundle.zip
```

结果目录必须包含：

```text
v22_09_final_route.json
v22_09_code_truth_gate.csv
v22_09_required_source_files.csv
v22_09_import_closure.csv
v22_09_semantic_contract.csv
v22_09_semantic_alias_matrix.csv
v22_09_efficiency_full_loop_reconfirm.csv
v22_09_drat_drbf_multibatch_summary.csv
v22_09_retained_source_observer_summary.csv
v22_09_train_flow_algebra_certificate.csv
v22_09_drift_diffusion_certificate.csv
v22_09_block_source_state_certificate.csv
v22_09_metric_solver_matrix.csv
v22_09_source_formation_matrix.csv
v22_09_terminal_preservation_matrix.csv
v22_09_kan_mapping_matrix.csv
v22_09_failure_taxonomy.csv
v22_09_queue_drain_report.json
figures/*.svg
logs/*.log
```

---

## 15. v22.09 的最低有效进展定义

这轮不要求直接 S5，但不能再只输出同样 no-go。最低有效进展必须至少满足以下之一：

```text
A. Code closure:
   final zip 自包含，missing required source = 0，import errors = 0。

B. Efficiency closure:
   D-RAT 或 D-RBF 至少一个 robust multibatch official closure。

C. Observer progress:
   C-O13 / C-O14 / C-O15 中至少一个 heldout precision@20 >= 0.40。

D. Functional progress:
   MLP C3 early+h3200 source chain 打开，且 controls fail。

E. Theory boundary:
   若 A-D 都失败，必须输出 RetainedSourceObserverLocalNoGo_v22.09，明确哪些 first-principles observer 被排除，不能继续同族小修。
```

---

## 16. 最终总结

v22.09 的核心不是继续 “找一个好 update”。真正要解决的是：

$$
\boxed{
\text{在训练前可见的信息中，是否存在 retained source 的证书？}
}
$$

如果没有这个证书，metric、solver、optimizer、KAN carrier 都只是在执行一个不知道是否可留存的 target。v22.09 要把 functional update 从工程变体池推进到算法层：

```text
retained-source certificate
→ metric-defined target
→ block-coordinate solver
→ optimizer-state integration
→ source preservation
→ KAN source-channel mapping
```

这才是下一步可能真正破局的方向。
