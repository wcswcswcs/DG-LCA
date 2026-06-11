# DG-KAN v22.10：Constructive Retained-Source Functional Update + Basis Efficiency Closure + 4GPU 完整计划

> 版本：v22.10 execution plan  
> 生成时间：2026-06-07  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler/class weight；no dataset-name branch；no seed-specific rule；no validation/test/future/query 生成 functional direction；LineC / CEp99 / NLL / ECE / Brier / AUCtime 只能作为 audit、debt readback 和 gate，不能生成方向。  
> 这版计划必须同时推进两条线：**Functional Update 算法突破** 与 **basis efficiency closure**。D-CHE/D-FOU 继续作为主 functional carrier，D-RAT/D-RBF 继续 robust officialization，不允许再次只躺在 blocked 表里。

---

## 0. 项目总目标与当前真实状态

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，而是建立一个 strict FC-PureKAN 训练系统，使其在表达、效率、训练稳定性和泛化几何上成为 MLP 的可竞争替代。总目标是：

$$
\boxed{
\text{PureKAN base + functional update}
>
\text{PureKAN base + ordinary backprop / AdamW controls}
}
$$

这个目标必须同时满足：

```text
1. 表达与任务不劣于同参数 MLP；
2. forward / backward / step / memory 接近同参数 MLP；
3. functional update 的收益打过 AdamW / SGD / NoOp / random / stable-random / same-overhead controls；
4. 好处能长期留存，不只是 h800/h3200 局部 positive；
5. tail / LineC / ECE / Brier / AUCtime debt 不爆；
6. 若 MLP-FU 成功，只能写成 generic training-dynamics insight；
7. 若 KAN 在同机制下超过 MLP，才可以讨论 KAN-specific functional advantage。
```

v22.09 的真实状态是：

```text
Code:
  v22.09 required path 基本自包含，compile/import 能在最终 zip 解压目录通过；
  但 import_closure.csv 中 self_contained_import_check=0 与 clean-unzip self-test=1 存在口径冲突；
  缺少独立 kernel_gradcheck artifact，需要补齐。

Efficiency:
  D-CHE / D-FOU 继续是 OfficialEfficientCarrier；
  D-RAT / D-RBF 仍未 robust official closure，当前 blocker 主要是 forward_ratio 与 component telemetry / functional runner kernel mismatch。

Functional:
  C0 retained-source certificate pass rows = 0；
  official early-chain positive rows = 0；
  retained h800+h3200 positive rows = 13；
  BVFR / OSH / ROST / IVSC / FSMR / TRAC 等 post-boundary family 有大量 train-only pass，但 fresh C3 / official C3 全部为 0；
  C2/C3/C4 pass rows = 0 / 0 / 0；
  KAN mapping 正确 fail-closed。
```

因此 v22.10 的核心判断是：

$$
\boxed{
\text{当前问题不是“执行不了 update”，而是“没有合法 train-only 证书能构造 retained source”。}
}
$$

更进一步，v22.09 暴露了一个比“observer 不够强”更本质的问题：当前 official early-chain positive rows 为 0。在没有正例的情况下，继续训练或调优 observer classifier 没有意义。下一轮必须从 **observer-as-filter** 转成 **constructive retained-source generator**。

---

## 1. v22.10 的核心思想：从“找证书”转向“构造 retained source”

过去几轮主要问：

```text
给定一批历史候选 update，能不能用 train-only signal 找出会 retained 的候选？
```

v22.09 证明：这条路线已经到 local no-go。原因不是某个 observer 少调了参数，而是当前 candidate pool 里没有 official early-chain positive label。继续对 C-O13/C-O14/C-O15、BVFR、OSH、ROST、IVSC、FSMR、TRAC 做同族小修，只会继续得到：

```text
train-only pass 很多；
fresh C3 = 0；
official C3 = 0。
```

v22.10 改问：

$$
\boxed{
\text{能不能用 train-only micro-dynamics 构造一个 source direction，而不是从旧候选里筛？}
}
$$

也就是说，Functional Update 的算法对象从：

```text
score(update candidate)
```

改成：

```text
solve for update direction that maximizes coherent train-flow signal and minimizes diffusion / curvature / control-equivalent components.
```

---

## 2. Functional Update 的第一性定义

普通训练一步可以写成：

$$
\theta_{t+1} = F_B(\theta_t, s_t),
$$

其中 $B$ 是 train batch，$s_t$ 是 optimizer state。Functional Update 不能再被看成孤立参数扰动，而应该是训练动力系统中的 control：

$$
(\theta_t, s_t)
\rightarrow
(\theta_t + u, s_t + v).
$$

我们真正想要的是：这个 control 在后续训练流中产生 retained source：

$$
Source_{h100}, Source_{h400}, Source_{h800}, Source_{h1600}, Source_{h3200}, Source_{h4800} > 0.
$$

因此 v22.10 采用三层算法：

```text
Layer 1: train-only source atom generation
  通过当前 train stream 的 micro-probe 生成一组可控 source atoms。

Layer 2: retained-source variational solve
  在这些 source atoms 上求一个低曲率、低扩散、低 control-equivalent 的方向。

Layer 3: metric-as-dynamics commit
  把这个方向写入参数和 optimizer state，并在 h100-h4800 验证 source 是否 retained。
```

---

## 3. 外部研究给 v22.10 的算法启发

### 3.1 Muon / NDS：不要只最大化一阶 gain

Muon curvature 研究指出，Muon 和 Adam 的一阶收益可相近，但 Muon 的优势来自更低的二阶曲率惩罚，且这个惩罚主要由更低的 Normalized Directional Sharpness 决定。v22.10 因此不再把 source amplitude 当唯一目标，而要求记录：

$$
I^{(1)} = \langle g, u \rangle,
$$

$$
I^{(2)} = \frac{1}{2}u^T H u,
$$

$$
NDS(u)=\frac{u^T H u}{\|u\|^2+\epsilon}.
$$

Functional direction 必须满足：

```text
同等 source 下 NDS 更低；
同等 NDS 下 source 更高；
不允许靠高曲率方向换短期 h800 source。
```

### 3.2 Nora：矩阵 block 的径向/角向稳定性

Nora 的 row-wise orthogonal momentum projection 启发我们：对矩阵参数，保留 row norm 和 angular velocity 的稳定性可能比单纯缩小 learning rate 更重要。v22.10 不会把 “Nora” 作为名字套到机制上，而是记录并控制：

```text
row_norm_drift;
row_angular_velocity;
radial_update_fraction;
tangential_update_fraction;
source_retention_delta;
```

特别是 MLP hidden/readout、D-CHE degree-readout、D-FOU band-readout、D-RAT numerator/denominator、D-RBF local-center/readout 都必须分 block 记录。

### 3.3 SOAP / Shampoo 系列：不要在 flat 参数空间做 metric

SOAP 的关键启发是：Adam-like moment 可以在 preconditioner eigenbasis / slowly changing coordinate basis 中运行。对我们来说，functional source state 不应是 flat vector，而应是 block-coordinate state：

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

### 3.4 AdEMAMix / Schedule-Free：source formation 和 source preservation 是两个时间尺度

单一 EMA 不能同时重视近期梯度和旧梯度。v22.10 把 source state 拆成：

$$
S_{fast,t} = \beta_f S_{fast,t-1} + (1-\beta_f)d_t,
$$

$$
S_{slow,t} = \beta_s S_{slow,t-1} + (1-\beta_s)d_t.
$$

其中：

```text
fast state:
  负责 h100/h400/h800 source formation；

slow state:
  负责 h1600/h3200/h4800 source preservation。
```

### 3.5 Signal channel / reservoir：训练能动不等于能泛化

Generalization theory 的 signal-channel / reservoir 观点告诉我们，训练能移动的方向不一定是泛化方向。Functional Update 必须最大化 coherent drift into signal channel，并避免 noise into signal channel。

因此 v22.10 不再只记录 B2 transfer gain，而记录：

$$
DDR = \frac{\|P_{sig}\mu\|^2}{\sigma^2+\epsilon},
$$

其中 $\mu$ 是跨 split / micro-trajectory 的平均函数位移，$\sigma^2$ 是 idiosyncratic diffusion。

### 3.6 Deep Manifold：FU 是边界条件控制，不是孤立动作

Deep Manifold 的 boundary-conditioned iteration 视角提示：神经网络不是一开始就有固定点，而是在训练中通过边界条件和残差迭代形成 fixed-point region。Functional Update 应当被设计为训练流的边界条件，而不是单步动作。

因此 v22.10 要记录：

```text
boundary_flow_replay_error;
flow_order_commutator;
state_holonomy;
source_state_age;
fixed-point residual decrease;
```

但这些不再作为 observer score 直接筛候选，而是进入 variational objective。

---

## 4. 哪些旧路线可以重开，哪些必须停止

### 4.1 可以重开，但必须换语义

```text
Split-consensus / transfer operator:
  不再作为 observer score，而作为 source atom generator。

PopRisk / SNR:
  不再作为 one-shot mask，而作为 drift-diffusion signal-channel estimator。

Function-space / operator-level basis-channel:
  不再只看 ActuationR2，而进入 Jθu ≈ Δf_target 的 metric solver。

Basis-natural / block metric:
  不再做 diagonal metric grid，而做 block-coordinate source-state solve。

Graph-free analytic adjoint:
  不再只是效率路线，而用于 cheap JVP/VJP source atoms。

MLP functional update:
  保留为主算法实验台，因为 KAN mapping 需要先有 MLP retained source。
```

### 4.2 不应重启的路线

```text
Action bank / controller；
reset route；
cover objective 小修；
M31/M32 同族小修；
G/N/Q-token 扩展；
late rebound 放大；
high ActuationR2-only target；
retrospective selector-only；
```

原因：这些路线过去要么被 controls 解释，要么只产生 late rebound，要么是 hindsight，不是合法 retained-source generator。

---

## 5. v22.10 总执行结构

v22.10 分为 7 个阶段，所有阶段必须并行但按 gate 合并。

```text
S0.17: Code / packet / solver truth gate
S1: Basis efficiency closure
S2: Source atom generation
S3: Retained-source variational solve
S4: Metric-as-dynamics commit
S5: Horizon source formation / terminal retention
S6: KAN source-channel mapping
S7: final route / no-go / next hypothesis
```

---

## 6. S0.17：代码 / packet / solver truth gate

### 6.1 目标

保证最终交付的 zip 在解压目录中自包含，而不是只在 Codex 原始 repo 环境中 pass。

### 6.2 必须执行

```text
1. unzip final code packet 到 clean temp dir；
2. 在 clean temp dir 运行 compileall；
3. 在 clean temp dir 运行 required import closure；
4. required_source_files.csv 中 exists=1 的文件必须真实存在于 zip；
5. LineC fast/channel tests；
6. source-chain / terminal-retention tests；
7. metric solver tests；
8. source atom generator tests；
9. variational solver tests；
10. semantic alias audit；
11. kernel gradcheck；
12. profiler phase tests；
13. forbidden direction audit。
```

### 6.3 硬标准

```text
self_contained_import_check = 1
csv_claimed_exists_but_zip_missing_count = 0
required_import_errors = 0
semantic_alias_undeclared_pairs = 0
forbidden_direction_violations = 0
kernel_status_consistency = 1
```

若不满足，route 必须写：

```text
R0-CodePacketOrImplementationClosureFail
```

并且不允许写 scientific no-go。

---

## 7. S1：Basis efficiency closure

### 7.1 D-CHE / D-FOU reconfirm

D-CHE / D-FOU 只做 full-loop reconfirm，不再重复大 census。

必须记录：

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
```

成功标准：

```text
D-CHE:
  >= 2 variants × >= 3 batch sizes S1 pass；
  forward <= 1.25；
  step <= 1.25；
  memory <= 1.05；
  same_kernel_functional_runner_proof = 1；
  fallback_kernel_used = 0。

D-FOU:
  FOU-R3 >= 3 batch sizes S1 pass；
  forward <= 1.25；
  step <= 1.25；
  memory <= 1.05；
  same_kernel_functional_runner_proof = 1；
  fallback_kernel_used = 0。
```

### 7.2 D-RAT / D-RBF robust officialization

D-RAT / D-RBF 必须继续推进，但不进入 full functional matrix，除非 robust efficiency gate 过。

测试 batch：

```text
batch = 128, 256, 512, 1024
```

D-RAT 必须记录：

```text
numerator_eval_ms
denominator_eval_ms
reciprocal_ms
numden_fused_ms
backward_num_ms
backward_den_ms
denominator_safety_p01
denominator_safety_min
component_telemetry_complete
functional_runner_kernel_match
```

D-RBF 必须记录：

```text
active_center_fraction
mean_local_K
local_gather_ms
exp_eval_ms
basis_materialized_bytes
local_backward_ms
width_grad_ms
center_grad_ms
component_telemetry_complete
functional_runner_kernel_match
```

Robust official 标准：

```text
>= 3/4 batch sizes:
  forward <= 1.25
  step <= 1.25
  memory <= 1.05
component_telemetry_complete = 1
functional_runner_kernel_match = 1
official_fused_kernel_complete = 1
```

如果仍失败，D-RAT / D-RBF 只允许 limited smoke，不进入 full FU。

---

## 8. S2：Source atom generation

### 8.1 为什么需要 source atoms

v22.09 的 official early-chain positive rows 为 0，所以继续筛旧 candidates 没意义。S2 要主动生成一组 train-only source atoms。

每个 source atom 是一个小的可回滚函数位移：

$$
a_j = \Delta f_j = f_{\theta+u_j}(x)-f_\theta(x)
$$

其中 $u_j$ 只能由当前 train stream 构造，不能使用 validation/test/future。

### 8.2 Source atom families

```text
A1: split-transfer atom
  用 B1 构造方向，在 B2 上读 immediate train transfer。

A2: drift-diffusion atom
  用多个 micro-batches 的 mean drift / diffusion 构造方向。

A3: control-null residual atom
  去掉 AdamW/SGD/random/stable-random span 后的 residual source。

A4: low-NDS atom
  在一阶 gain 相近的方向中选择低 NDS 方向。

A5: row-orthogonal matrix atom
  在矩阵 block 中只保留 tangential / angular-stable component。

A6: fast-slow source-state atom
  fast state 负责 early source，slow state 负责 retention。

A7: train-flow commutator atom
  惩罚对 batch order 高敏感的方向。

A8: KAN low-degree / low-frequency atom
  只在 D-CHE low-degree 或 D-FOU low-frequency channel 生成 source atom。
```

### 8.3 Source atom 记录指标

每个 atom 必须记录：

```text
atom_id
carrier
block_role
uses_future_or_validation
uses_audit_metric_for_direction
B1_gain
B2_transfer_gain
B3_safety_gain
random_gap
sign_flip_gap
corrupt_target_gap
control_null_residual_norm
commutator_norm_ratio
flow_order_gap
DDR
NDS
metric_energy_L2
metric_energy_Fisher
metric_energy_Sobolev
metric_energy_RKHS
row_norm_drift
row_angular_velocity
radial_fraction
tangential_fraction
source_atom_norm
source_atom_function_cosine_with_controls
```

### 8.4 Source atom pass 标准

一个 atom 只能进入 S3，如果满足：

```text
B2_transfer_gain > random_matched + 0.0005
B3_safety_gain >= -0.0005
random_gap > 0
sign_flip_gap > 0
corrupt_target_gap > 0
control_null_residual_norm > 0
commutator_norm_ratio <= 0.10
NDS <= control_NDS_median
uses_future_or_validation = 0
uses_audit_metric_for_direction = 0
```

若没有任何 atom pass，route 写：

```text
S2-SourceAtomGenerationNoGo
```

Codex fallback：

```text
1. 降低 source atom norm，但不降低 pass gate；
2. 增加 micro-batch split count；
3. 改用 block-restricted atom；
4. 若仍失败，停止同族 atom 扩展，输出 no-go。
```

---

## 9. S3：Retained-source variational solve

### 9.1 目标

用 S2 source atoms 直接构造方向，而不是给已有候选打分。

令 $A=[a_1,a_2,\ldots,a_m]$ 是 source atom 矩阵。求：

$$
w^* = \arg\max_w
\left[
D_{signal}(Aw)
- \lambda D_{diffusion}(Aw)
- \mu E_{metric}(Aw)
- \nu E_{control}(Aw)
- \gamma E_{curvature}(Aw)
\right]
$$

约束：

$$
\|w\|_1 \leq \tau,
$$

$$
\operatorname{sign\_flip\_gap}(Aw) > 0,
$$

$$
\operatorname{random\_gap}(Aw) > 0.
$$

### 9.2 Objective components

```text
D_signal:
  split-coherent drift into estimated signal channel。

D_diffusion:
  micro-trajectory variance / idiosyncratic diffusion。

E_metric:
  L2 / Fisher / Sobolev / RKHS / basis-channel energy。

E_control:
  projection onto AdamW / SGD / random / stable-random control span。

E_curvature:
  low-NDS / curvature penalty。
```

### 9.3 S3 记录指标

```text
source_combo_atom_count
source_combo_l1_norm
signal_drift_score
diffusion_score
DDR
control_projection_fraction
random_gap
sign_flip_gap
corrupt_gap
metric_energy_L2
metric_energy_Fisher
metric_energy_Sobolev
metric_energy_RKHS
NDS
solve_time_ms
selected_atoms
selected_atom_weights
```

### 9.4 S3 pass 标准

```text
DDR >= 1.0
control_projection_fraction <= 0.25
random_gap > 0
sign_flip_gap > 0
corrupt_gap > 0
NDS <= control_NDS_median
metric_energy_Sobolev <= control_Sobolev_p50 or RKHS_tearing <= control_p50
```

若 S3 fail，Codex 可先尝试：

```text
1. l1 sparsity 更强；
2. 只用 low-NDS atoms；
3. 只用 control-null atoms；
4. 只用 block-restricted atoms；
5. 若仍 fail，输出 S3-VariationalSourceSolveNoGo。
```

---

## 10. S4：Metric-as-dynamics commit

### 10.1 目标

把 S3 得到的函数方向 $\Delta f^*$ 写回参数与 optimizer state。

求解：

$$
u^* = \arg\min_u
\|J_\theta u - \Delta f^*\|_{G_f}^2
+ \rho\|u\|_P^2.
$$

但 v22.10 不能只做 readout solve。必须分层：

```text
S4.1 readout exact solve
S4.2 hidden/readout block CG solve
S4.3 matrix-block source-state solve
S4.4 optimizer-state integrated solve
S4.5 KAN basis-channel solve, gated after MLP success
```

### 10.2 Solver 必须记录

```text
solver_level
block_role
JVP_count
VJP_count
CG_iterations
projection_residual_Gf
ActuationR2
B2_transfer_gain
solve_time_ms
update_norm
function_displacement_norm
function_displacement_cos_with_target
hidden_source_energy
readout_source_energy
optimizer_state_write_fraction
```

### 10.3 Solver pass 标准

```text
projection_residual_Gf <= 0.40
ActuationR2 >= 0.20
B2_transfer_gain > random_matched
function_displacement_cos_with_target >= 0.30
uses_future_or_validation = 0
```

若 S4 fail，Codex 可先尝试：

```text
1. readout-only commit；
2. block-restricted hidden/readout commit；
3. lower rank CG sketch；
4. higher damping rho；
5. optimizer-state write instead of direct parameter commit；
6. 若仍 fail，输出 S4-MetricDynamicsCommitNoGo。
```

---

## 11. S5：Horizon source formation / terminal retention

只有 S2/S3/S4 全过，才进入 horizon test。

### 11.1 必测 horizons

```text
h100, h400, h800, h1600, h2400, h3200, h4000, h4800, h6400
```

### 11.2 必测 controls

```text
AdamW
SGD
Momentum
NoOpMatchedOverhead
RandomMatchedNorm
StableRandom
SameAtomsRandomWeights
SameMetricRandomTarget
SameSolverRandomTarget
SignFlipTarget
CorruptTarget
```

### 11.3 记录指标

```text
source_vs_best_control_h100/h400/h800/h1600/h2400/h3200/h4000/h4800/h6400
row_positive_count_h100/h400/h800/h1600/h3200/h4800
R4800_over_3200
R6400_over_4800
tail_debt_peak/final/recovery
LineC_debt_peak/final/recovery
ECE_debt_peak/final/recovery
Brier_debt_peak/final/recovery
AUCtime_ratio
control_equivalent_fraction
optimizer_projection_on_source
source_overwrite_fraction
source_hidden_fraction
source_readout_fraction
```

### 11.4 C3 source formation gate

```text
source_h100 >= 0.005
source_h400 >= 0.005
source_h800 >= 0.005
source_h1600 >= 0.005
source_h3200 >= 0.005
row_h3200_positive_count >= 6/9
matched controls fail
stable-random fail
debt not exploded
```

### 11.5 C4 terminal retention gate

```text
source_h4800 >= 0.005
R4800_over_3200 >= 0.50
row_h4800_positive_count >= 7/9
AUCtime <= 1.05
tail / LineC / ECE / Brier debt not worse
independent rerun pass
```

若 C3 过但 C4 不过，进入 terminal preservation，不进入 KAN mapping。

---

## 12. S6：KAN source-channel mapping

只有 MLP C3 或 C4 过后，KAN mapping 才进入。

### 12.1 MLP source decomposition

记录：

```text
hidden_source_energy
readout_source_energy
source_subspace_rank
source_singular_values
source_reconstruction_error
hidden/readout function displacement cosine
source information volume
```

### 12.2 KAN mapping candidates

```text
D-CHE low-degree bank
D-CHE degree-readout
D-CHE readout-only
D-CHE basis-estimate / readout-commit

D-FOU low-frequency bank
D-FOU band-readout
D-FOU readout-only
D-FOU basis-estimate / readout-commit

D-RAT numerator / denominator tangent block, only if robust official efficiency pass
D-RBF local-center / readout block, only if robust official efficiency pass
```

### 12.3 KAN success gate

```text
KAN h100/h400/h800 >= 0.005
KAN h1600/h3200 >= 0.005
R4800_over_3200 >= 0.50
KAN_specific_delta_vs_MLP_same_metric >= 0.005
same-mechanism MLP control available
matched controls fail
```

If MLP succeeds but KAN fails, route：

```text
KANSourceChannelMismatchConfirmed
```

If MLP fails, route：

```text
KANMappingNotEntered
```

---

## 13. 4GPU dynamic queue

v22.10 必须使用真实动态队列，不只是静态分配。

### 13.1 必须落盘

```text
v22_10_runnable_queue.csv
v22_10_gpu_assignment_manifest.csv
v22_10_gpu_utilization_timeline.csv
v22_10_idle_violation.csv
v22_10_deferred_items.csv
v22_10_queue_drain_report.csv
```

### 13.2 GPU 分工

```text
GPU0:
  S0.17 code / source atom generator tests / MLP source atom experiments。

GPU1:
  Metric-as-dynamics solver / MLP S3-S5 horizon tests。

GPU2:
  D-CHE/D-FOU reconfirm + KAN mapping gated runs。

GPU3:
  D-RAT/D-RBF robust officialization + limited smoke。
```

### 13.3 硬规则

```text
runnable_queue 非空 && any GPU idle > 10 min
=> execution_contract_violation = 1
=> final route 不能写 completed-no-go。
```

---

## 14. 必须可视化

### 14.1 Code / coverage dashboard

```text
required source file presence matrix
compile/import status table
semantic alias heatmap
kernel status consistency table
```

### 14.2 Basis efficiency dashboard

```text
D-CHE/D-FOU full-loop timing bar
D-RAT/D-RBF multibatch forward/step/memory curve
component waterfall for D-RAT numerator/denominator/reciprocal
component waterfall for D-RBF local gather/exp/local backward
```

### 14.3 Source atom dashboard

```text
source atom B2 gain vs random gap
source atom DDR vs NDS
commutator_norm_ratio vs source score
control projection fraction histogram
```

### 14.4 Variational solve dashboard

```text
selected atom weights bar
signal / diffusion / metric / control / curvature objective decomposition
projection residual vs ActuationR2 scatter
function displacement cosine heatmap
```

### 14.5 Horizon source dashboard

```text
source trajectory h100-h6400
R4800_over_3200 bar
row positive count by horizon
debt recovery curves
controls comparison plot
```

### 14.6 KAN mapping dashboard

```text
MLP hidden/readout source energy
D-CHE low-degree vs degree-readout source transfer
D-FOU low-frequency vs band-readout source transfer
KAN vs MLP same-mechanism source delta
```

---

## 15. 最低有效进展定义

v22.10 不能再只输出同类 no-go。最低有效进展必须至少满足以下之一：

```text
A. Code closure:
  final zip 自包含，missing required source = 0，import errors = 0，self_contained_import_check 口径一致。

B. Efficiency closure:
  D-RAT 或 D-RBF 至少一个 robust multibatch official closure。

C. Source atom progress:
  至少一个 source atom family 通过 S2 gate，且 random/sign/corrupt controls fail。

D. Constructive solve progress:
  S3 variational source solve 通过，进入 S4 commit。

E. Functional progress:
  MLP C3 early+h3200 source chain 打开，且 controls fail。

F. Theory boundary:
  如果 A-E 都失败，必须输出 ConstructiveRetainedSourceNoGo_v22.10，明确 source atom / variational solve / metric commit 哪一层失败。
```

---

## 16. Final route taxonomy

```text
R0-CodePacketOrImplementationClosureFail
R1-D-RAT-D-RBF-RobustEfficiencyBlocked
R2-SourceAtomGenerationNoGo
R3-VariationalSourceSolveNoGo
R4-MetricDynamicsCommitNoGo
R5-MLPSourceFormationOpenedTerminalBlocked
R6-MLPTerminalRetentionOpened
R7-KANSourceChannelMismatchConfirmed
R8-KANRetainedSourceOpened
R9-OfficialPromotionCandidate
```

---

## 17. v22.10 的最终原则

v22.10 不再问：

```text
哪个 observer score 更好？
哪个 metric 名字更好？
哪个 Fxx 变体更好？
```

而是问：

$$
\boxed{
\text{能否从 train-only micro-dynamics 构造一个 coherent、low-diffusion、low-curvature、control-resistant 的 retained-source direction？}
}
$$

这才是 Functional Update 从工程搜索转向算法突破的方向。

