# DG-KAN v22.11：Arbitrary-Loss Constructive Functional Update + Basis Efficiency Closure + 4GPU 完整计划

> 版本：v22.11 execution plan  
> 生成时间：2026-06-07  
> 核心修正：v22.10 已经打开 loss-agnostic source-side constructive path，但 S1 basis efficiency 被 arbitrary-loss / upstream-gradient contract 拦住；同时 v22.10 的 source horizon 仍主要是 target-retention / geometry replay，不是任意任务 loss 下的真实训练动力学。因此 v22.11 必须把 Functional Update 从“label-free target replay”推进为“arbitrary-loss retained-source training dynamics”。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific branch；no validation/test/future/query 生成方向；LineC / ECE / Brier / AUCtime / tail 只能 audit / readback / gate，不能生成方向；**优化与效率 runner 不得针对 CE**，必须通过 generic loss-interface 或 arbitrary upstream cotangent。  

---

## 0. 项目总目标与 v22.10 后的真实状态

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找到一个 CE 分类任务专用技巧。项目目标是：

$$
\boxed{
\text{在 strict FC-PureKAN base 上，通过 loss-agnostic functional update 改善训练动力学，}
\text{得到比普通 backprop / AdamW controls 更好的模型。}
}
$$

这里的“更好”必须同时满足：

```text
1. source 不是 single-row positive，而是 h100/h400/h800/h1600/h3200/h4800/h6400 连续留存；
2. source 打过 AdamW / SGD / Momentum / NoOp / random / stable-random / same-overhead controls；
3. 更新方向不针对 CE，而是只依赖 generic loss-interface 的 output cotangent 或完全 label-free train-stream geometry；
4. 任务 loss 可以是 CE、MSE、ranking loss、policy loss、DPO loss、PDE residual loss 等；算法不能使用 CE-specific 公式；
5. D-CHE / D-FOU / D-RAT / D-RBF 的 forward / backward / update / memory 必须在 arbitrary upstream gradient 下测量；
6. KAN-specific claim 必须在同机制 MLP control 之外成立；
7. 所有 audit metric 只用于 readback，不用于 direction。
```

v22.10 的真实状态必须分开看。

第一，代码包本身有进步。v22.10 的 required source files 存在，解压后的 required import closure 可以过，source-chain / terminal-retention / metric-solver / source-atom / variational-solver / constructive-commit tests 都落盘通过。这个比 v22.08 那种 final zip 不自包含的情况好。

第二，source-side constructive path 有明显进展。v22.10 中：

```text
S2 source atom pass rows = 2；
S3 variational source solve pass rows = 1；
S4 metric dynamics commit pass rows = 1；
C3 / C4 pass rows = 7 / 7；
PID C3 / C4 pass rows = 3 / 3；
KAN mapping decision = KANRetainedSourceOpened。
```

这说明：通过 label-free logit geometry 构造 source atoms，再用 variational solve 和 readout commit，可以形成一个能被 PID / replay 保住的 target-retention source。

第三，不能把 v22.10 写成 official breakthrough。两个原因很硬：

```text
1. Basis efficiency formal route 被 loss-agnostic efficiency contract 拦住：
   optimization_loss_agnostic_contract_pass = 0；
   CE-targeted efficiency rows invalidated = 1。

2. v22.10 的 horizon source 主要是在 TinyConstructiveMLP / target-retention replay / geometry-energy dynamics 下证明 source 可保，
   不是任意任务 loss 下的真实 training loss dynamics。
```

所以 v22.11 的核心目标不是“继续放大 v22.10 的 positive”，而是把它接到真实任意 loss 的训练系统里：

$$
\boxed{
\text{v22.11 的主问题：v22.10 的 constructive source 能否在 arbitrary-loss training dynamics 中留存？}
}
$$

---

## 1. v22.10 的本质诊断

### 1.1 代码审计结论

v22.10 的 required path 可用，但下一轮仍必须强制 final zip 自检。S0.18 需要在最终交付 zip 解压目录中执行：

```text
python -m compileall -q dgkan experiments tests
required import closure
clean unzip import closure
required_source_files.csv 与 packet_manifest.csv 一致性检查
LineC fast/channel golden tests
source-chain / terminal-retention tests
loss-interface / arbitrary-cotangent tests
kernel gradcheck
profiler phase tests
semantic non-collapse audit
```

如果 `required_source_files.csv` 声称存在的文件没有真实进入 final zip，或者 `self_contained_import_check=0`，则：

```text
CodeRoute = R0-CodePacketNotSelfContained
scientific no-go 禁止写入
```

### 1.2 v22.10 source-side 为什么值得保留

v22.10 做对了一件事：它不再只给已有 update 打分，而是构造 source atoms，再做 variational source solve，再 commit 到函数空间方向。这是从 observer/filter 转向 constructive source generation 的第一步。

但是 v22.10 的 source atoms 是从当前 train logits 的 label-free geometry 里构造的，主要是：

```text
A10 split-consensus atom；
A11 causal split transport atom；
control-balanced L1 variational combination；
readout all-train actuation repair；
PID source-state replay。
```

这些东西说明“存在可控的函数位移源”，但还没说明“它对任意 loss 的训练有用”。

### 1.3 v22.10 最大风险：source replay 不等于 task training

v22.10 horizon runner 继续训练时使用的是 target-retention / geometry-energy dynamics，而不是某个 arbitrary loss 的真实训练目标。换句话说，它证明的是：

```text
如果训练动力学以维持 target displacement 为核心，source 可以被保住。
```

但项目真正要证明的是：

```text
普通任务 loss 继续训练时，这个 functional source 仍能带来长期收益。
```

因此 v22.11 必须新增 **arbitrary-loss training replay**，不能再只读 target-retention source。

---

## 2. 第一性原则：Functional Update 到底应该是什么

普通训练可以写成：

$$
\theta_{t+1}=\theta_t + U_{base}(\theta_t, \delta_t),
$$

其中 $\delta_t$ 是当前任务 loss 对模型输出的 cotangent：

$$
\delta_t = \frac{\partial \mathcal{L}}{\partial f_\theta(x_t)}.
$$

这里 $\mathcal{L}$ 不能写死成 CE。它可以是任意可微任务 loss。functional update 应该是一个训练动力系统中的 control term：

$$
\theta_{t+1}
=
\theta_t
+
U_{base}(\theta_t,\delta_t)
+
U_{FU}(\theta_t,\mathcal{S}_t,G_f,\mathcal{M}_t),
$$

其中：

```text
S_t:
  train-stream source state，只使用当前/过去 train-stream 可见信息；

G_f:
  函数空间 metric，例如 L2 / Fisher / Sobolev / RKHS / basis-channel metric；

M_t:
  optimizer state / slow state / source memory。
```

Functional update 不是替代 loss，也不是 CE-specific optimizer。它应该做三件事：

```text
1. 识别 coherent source：当前 train-stream 中哪些函数方向具有跨 split / 跨 micro trajectory 一致性；
2. 在函数空间 metric 下提交低能量、低曲率、低 control-equivalent 的位移；
3. 把 source 写入 optimizer state / slow state / KAN source-channel，使它在后续 arbitrary-loss training 中不被洗掉。
```

因此 v22.11 的 Functional Update 不再以 Fxx 名字为中心，而以这个链条为中心：

```text
upstream cotangent / label-free geometry
→ source atoms
→ retained-source variational solve
→ metric-as-dynamics commit
→ arbitrary-loss horizon training
→ terminal source preservation
→ KAN source-channel mapping
```

---

## 3. 当前研究进展给我们的算法启发

### 3.1 Muon / curvature / NDS

Muon 曲率视角提示：一阶收益相近时，长期训练差异可能来自二阶曲率惩罚和 Normalized Directional Sharpness，而不是 update norm 更大。因此 v22.11 不应只看 `source_h3200`，还要记录：

$$
I^{(1)} = \langle g, u \rangle,
$$

$$
I^{(2)} = u^T H u,
$$

$$
NDS = \frac{u^T H u}{\|u\|^2 + \epsilon}.
$$

如果某个 FU 方向在 h800/h3200 source 很强，但 $NDS$ 高，则它可能在 h4800/h6400 被训练动力学侵蚀。

### 3.2 Nora / row-wise angular stability

Nora 的启发不是“换 optimizer”，而是：矩阵参数的 row norm 和 angular velocity 必须稳定。对我们来说，MLP hidden/readout、D-CHE degree-readout、D-FOU band-readout、D-RAT numerator/denominator、D-RBF local-center/readout 都要记录：

```text
row_norm_drift
row_angular_velocity
radial_component_fraction
tangential_component_fraction
source_projection_radial
source_projection_tangential
```

如果 source 主要写入 radial jitter 或高 angular velocity 方向，可能短期有效但长期不稳。

### 3.3 SOAP / block coordinate moment

SOAP 类思路提示：adaptive moment 不一定要在 flat parameter basis 中运行，可以在 slowly-changing block/eigenbasis 中运行。v22.11 因此要把 source state 分成 block-coordinate：

```text
MLP hidden / readout；
D-CHE low-degree / high-degree / degree-readout；
D-FOU low-frequency / high-frequency / band-readout；
D-RAT numerator / denominator / readout；
D-RBF local-center / width / readout。
```

### 3.4 Schedule-Free / AdEMAMix / fast-slow state

v22.10 的 PID/source-state replay 给出了 positive 线索，但它还是 target-retention replay。下一步要把 fast/slow state 真正接入 arbitrary-loss training：

```text
fast source state:
  负责 h100/h400/h800 source formation；

slow source state:
  负责 h1600/h3200/h4800/h6400 retention；

source preservation state:
  负责防止后续 base optimizer 更新破坏 source。
```

### 3.5 Signal channel / reservoir

泛化理论的 signal-channel / reservoir 视角提醒我们：训练能移动的方向不一定能泛化。FU 必须让 coherent source 进入 signal channel，而不是 reservoir。v22.11 必须记录：

```text
signal_channel_energy
reservoir_energy
noise_into_signal_ratio
control_null_source_residual
source_channel_drift
source_channel_diffusion
```

### 3.6 Deep Manifold / boundary-conditioned iteration

Deep Manifold 的固定点 / boundary-conditioned iteration 视角提醒我们：FU 不是孤立一步动作，而是改变训练动力学边界条件。v22.11 的 PID / fast-slow state / optimizer-state integration 都应该看作训练动力系统控制，而不是 target replay trick。

---

## 4. v22.11 总执行结构

v22.11 分七条线并行执行。

```text
Line A: S0.18 code / loss-interface / metric / kernel truth gate
Line B: arbitrary-loss / upstream-cotangent basis efficiency closure
Line C: constructive source atoms on real train-stream, not toy-only
Line D: retained-source variational solve under arbitrary upstream cotangent
Line E: metric-as-dynamics commit + optimizer-state integration
Line F: arbitrary-loss horizon training verification
Line G: KAN source-channel mapping and D-RAT/D-RBF limited entry
```

这七条线必须用 4GPU dynamic queue 执行，不能串行等待。

---

## 5. Line A：S0.18 代码 / loss-interface / metric / kernel truth gate

### 5.1 目标

保证 v22.11 的代码包、loss-interface 和效率 runner 都能独立复核，且不再 CE-targeted。

### 5.2 必须打包的文件

Codex 最终必须打包：

```text
dgkan/fu/loss_interface.py
dgkan/fu/upstream_cotangent.py
dgkan/fu/source_atoms.py
dgkan/fu/variational_source_solver.py
dgkan/fu/constructive_commit.py
dgkan/fu/metric_solver.py
dgkan/fu/function_space_metrics.py
dgkan/fu/jacobian_sketch.py
dgkan/fu/basis_channel_metric.py
dgkan/fu/source_chain.py
dgkan/fu/terminal_retention.py
dgkan/fu/debt_accounting.py
dgkan/fu/source_state_dynamics.py
dgkan/fu/optimizer_state_integration.py
dgkan/profiling/efficiency_v22_11.py
dgkan/kernels/fused_chebyshev_k3.py
dgkan/kernels/fused_fourier_k2.py
dgkan/kernels/fused_rational_k4.py
dgkan/kernels/fused_rbf.py
experiments/run_v22_11_*.py
```

### 5.3 loss-interface 硬要求

新增统一接口：

```python
class LossInterface:
    def value(self, logits, target_or_task_data): ...
    def cotangent(self, logits, target_or_task_data): ...
    def name(self): ...
```

必须至少实现并测试：

```text
GenericUpstreamCotangent:
  直接给定 delta，不关心 loss 类型；

ClassificationCEAdapter:
  只作为 adapter，不能在 FU 内写 CE-specific 公式；

RegressionMSEAdapter:
  用于回归式 arbitrary loss；

PairwiseRankingAdapter:
  用于 pair/ranking style cotangent；

PolicyPreferenceAdapter / DPO-like cotangent smoke:
  只验证接口，不做任务 claim。
```

### 5.4 S0.18 通过标准

```text
required_source_files = 100% exists in final zip；
self_contained_import_check = 1；
loss_interface_tests = pass；
arbitrary_upstream_cotangent_tests = pass；
CE_specific_formula_in_core_FU = 0；
LineC / ECE / Brier / AUCtime used_for_direction = 0；
semantic_alias_undeclared_pairs = 0；
kernel_status_consistency = 1。
```

如果任一失败：

```text
Route = R0-CodeOrLossInterfaceGateFailed
禁止 scientific no-go
```

---

## 6. Line B：arbitrary-loss / upstream-gradient basis efficiency closure

### 6.1 为什么必须做

v22.10 的效率 blocker 是：旧 efficiency runner 依赖 CE-targeted trainpath，因此 formal S1 被 invalidated。v22.11 必须把 basis efficiency 定义改成：

```text
给定任意 output cotangent delta，basis forward/backward/update 是否仍接近 MLP。
```

这与具体 loss 无关。

### 6.2 实验对象

```text
D-CHE:
  CHE21-R2-low-degree-k3
  CHE21-R4-k5-gradbuf

D-FOU:
  FOU21-R3-tablelookup-bandreadout

D-RAT:
  fused rational k4 / k2 production trainpath

D-RBF:
  fused local RBF k4 / k2 no-dense trainpath

MLP:
  same-param reference
```

### 6.3 upstream cotangent suite

每个 basis 必须在这些 cotangent 下测：

```text
Delta-Gaussian:
  random normal matched norm；

Delta-StableRandom:
  deterministic stable random；

Delta-SourceTarget:
  v22.10 source target displacement；

Delta-LossCEAdapter:
  CE adapter cotangent，只作为 one adapter，不得用于算法特化；

Delta-MSEAdapter:
  regression adapter cotangent；

Delta-RankingAdapter:
  pairwise ranking adapter cotangent。
```

### 6.4 记录指标

```text
carrier
variant
cotangent_type
batch_size
forward_ms
backward_vjp_ms
parameter_update_ms
functional_commit_ms
full_step_ms
forward_memory_mb
backward_memory_mb
workspace_memory_mb
kernel_count
fused_kernel_used
fallback_kernel_used
component_telemetry_complete
functional_runner_kernel_match
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
gradcheck_pass
upstream_cotangent_contract_pass
```

### 6.5 成功标准

D-CHE / D-FOU：

```text
>= 3/4 batch sizes pass；
>= 4/6 cotangent types pass；
forward_ratio <= 1.25；
step_ratio <= 1.25；
memory_ratio <= 1.05；
fallback_kernel_used = 0。
```

D-RAT / D-RBF：

```text
near-E1:
  forward_ratio <= 2.0；
  step_ratio <= 1.75；
  memory_ratio <= 1.20；

robust official:
  >= 3/4 batch sizes pass；
  >= 3/6 cotangent types pass；
  forward_ratio <= 1.25；
  step_ratio <= 1.25；
  memory_ratio <= 1.05；
  component_telemetry_complete = 1。
```

如果 D-RAT / D-RBF 不过：

```text
只允许 limited smoke；禁止 full FU matrix。
```

---

## 7. Line C：real train-stream constructive source atoms

### 7.1 目标

v22.10 的 source atoms 是 TinyConstructiveMLP 上的 label-free logit geometry。v22.11 要把 source atom generation 移到真实 train-stream / arbitrary loss-interface 环境。

### 7.2 允许的信息

允许：

```text
current train-stream logits；
current hidden/readout/basis activations；
current loss-interface output cotangent delta；
past train-stream source state；
current optimizer state；
current batch/micro-split statistics。
```

禁止：

```text
validation/test/future outcome；
dataset-name branch；
seed-specific branch；
CE-specific formula；
LineC/ECE/Brier/AUCtime as direction；
classwise CE-tail hard target as direction。
```

### 7.3 source atom families

每个 source atom 都是函数空间位移 $a_j=\Delta f_j$。

```text
A1 upstream-cotangent drift atom：
  用 loss-interface cotangent 的跨 split coherent drift；

A2 label-free logit geometry atom：
  继承 v22.10 的 split consensus / causal split transport；

A3 drift-diffusion signal-channel atom：
  coherent drift / diffusion ratio 高，且 control-null residual 非零；

A4 train-flow commutator atom：
  micro trajectory order 不敏感；

A5 low-NDS atom：
  一阶 gain 不弱、NDS 低；

A6 row-orthogonal matrix atom：
  控制 radial jitter 和 angular velocity；

A7 fast-slow source-state atom：
  fast state 负责 early source，slow state 负责 retained source；

A8 KAN low-degree / low-frequency atom：
  D-CHE 低阶、D-FOU 低频、D-RAT num/den tangent、D-RBF local-center/readout；

A9 control-null residual atom：
  投掉 AdamW / SGD / random / stable-random span 后仍有残差；

A10 arbitrary-cotangent invariant atom：
  在 CE/MSE/ranking/random upstream cotangent 下方向稳定。
```

### 7.4 记录指标

```text
atom_id
carrier
cotangent_type
uses_labels_for_direction
uses_loss_formula_specific_direction
B1_gain_generic
B2_transfer_gain_generic
B3_safety_gain_generic
random_gap
sign_flip_gap
corrupt_gap
control_projection_fraction
control_null_residual_norm
DDR
commutator_norm_ratio
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
S2_source_atom_pass
blocker
```

### 7.5 S2 gate

一个 atom 进入 S3，必须满足：

```text
uses_loss_formula_specific_direction = 0；
B2_transfer_gain_generic > random + margin；
B3_safety_gain_generic >= -tolerance；
random_gap > 0；
sign_flip_gap > 0；
corrupt_gap > 0；
control_projection_fraction <= 0.35；
commutator_norm_ratio <= 0.45；
NDS <= control_NDS_p50 或在 source gain 明显更高时进入 low-NDS repair bucket；
至少 2 个 cotangent_type 下保持同向。
```

如果没有 atom 过：

```text
先尝试降低 norm_scale；
再尝试增加 split_count；
再尝试 block-restricted atom；
再尝试 control-null residual atom；
仍失败则输出 SourceAtomNoGo_v22.11，不进入 variational solve。
```

---

## 8. Line D：retained-source variational solve

### 8.1 目标

v22.10 的 variational solve 通过 `control_balanced_l1` 找到了 A10/A11 组合。v22.11 要在 real train-stream / arbitrary cotangent 下做同样的构造式组合。

### 8.2 目标函数

令 $A=[a_1,a_2,\ldots,a_m]$，组合位移为：

$$
\Delta f^* = Aw.
$$

求解：

$$
w^*=
\arg\max_w
\left[
D_{signal}(Aw)
-
\lambda D_{diffusion}(Aw)
-
\mu E_{metric}(Aw)
-
\nu E_{control}(Aw)
-
\gamma E_{curvature}(Aw)
-
\xi E_{loss\text{-}specific}(Aw)
\right].
$$

其中 $E_{loss\text{-}specific}$ 惩罚只在单一 loss adapter 下有效的方向。

### 8.3 记录指标

```text
selected_atoms
selected_weights
source_combo_l1_norm
signal_drift_score
diffusion_score
DDR
control_projection_fraction
control_null_residual_norm
metric_energy_L2/Fisher/Sobolev/RKHS
NDS
loss_adapter_invariance_score
random_gap
sign_flip_gap
corrupt_gap
source_combo_function_cosine_with_controls
S3_variational_source_solve_pass
```

### 8.4 S3 gate

```text
DDR >= 1.0；
control_projection_fraction <= 0.25；
random_gap / sign_gap / corrupt_gap > 0；
NDS <= max(control_NDS_median, 0.40) 或 low-NDS repair pass；
Sobolev/RKHS energy <= control p50 或 metric-benefit tradeoff pass；
loss_adapter_invariance_score >= 0.50。
```

失败时 Codex 先尝试：

```text
1. control-balanced L1；
2. nonnegative source weights；
3. entropy-regularized atom mixture；
4. low-NDS constrained solve；
5. block-coordinate atom mixture；
6. fast/slow atom split。
```

---

## 9. Line E：metric-as-dynamics commit + optimizer-state integration

### 9.1 目标

将 $\Delta f^*$ 写回参数和 optimizer state。不能只做 readout target replay。必须按层级测试：

```text
S1 readout exact solve；
S2 hidden/readout block CG；
S3 matrix-block source-state solve；
S4 optimizer-state integrated solve；
S5 KAN basis-channel solve。
```

### 9.2 commit 公式

$$
u^*=
\arg\min_u
\|J_\theta u - \Delta f^*\|_{G_f}^2
+
\rho\|u\|_P^2.
$$

但 v22.11 必须新增 optimizer-state integration：

$$
\mathcal{M}_{t+1}
=
\beta \mathcal{M}_t
+
(1-\beta)\Pi_{source}(u^*),
$$

后续 base update 需要使用 source-preserving projection：

$$
U_{base}^{safe}
=
U_{base}
-
\mathbf{1}_{\langle J U_{base}, s_t\rangle_{G_f}<0}
\frac{\langle J U_{base}, s_t\rangle_{G_f}}{\|s_t\|_{G_f}^2+\epsilon}s_t.
$$

### 9.3 记录指标

```text
solver_level
block_role
JVP_count
VJP_count
CG_iterations
projection_residual_Gf
ActuationR2
B2_transfer_gain
random_matched_B2_transfer_gain
function_displacement_cos_with_target
solve_time_ms
update_norm
function_displacement_norm
condition_estimate
optimizer_state_write_fraction
source_state_alignment
source_state_decay_rate
base_update_destructive_projection_fraction
loss_agnostic_contract_pass
S4_metric_dynamics_commit_pass
```

### 9.4 S4 gate

```text
projection_residual_Gf <= 0.40；
ActuationR2 >= 0.20；
B2_transfer_gain > random；
function_displacement_cos_with_target >= 0.30；
loss_agnostic_contract_pass = 1；
uses_CE_specific_formula = 0。
```

失败时 Codex 先尝试：

```text
1. readout exact solve；
2. readout all-train fit；
3. hidden/readout block CG；
4. low-rank Nyström J sketch；
5. optimizer-state-only write；
6. source-state integrated solver；
7. KAN basis-channel solve。
```

---

## 10. Line F：arbitrary-loss horizon training verification

### 10.1 目标

这是 v22.11 的主 gate。v22.10 的 target-retention replay 必须升级为任意 loss 的真实训练验证。

### 10.2 Horizon runner

每个候选必须跑：

```text
h100,h400,h800,h1600,h2400,h3200,h4000,h4800,h6400
```

继续训练时使用 generic loss-interface：

```text
loss_adapter = CE / MSE / Ranking / GenericUpstreamCotangent / PolicyPreference smoke
```

但 FU 算法不能读取 loss adapter 类型，也不能使用 CE-specific 公式。

### 10.3 source 定义

记录两套 source：

```text
function-retention source:
  target displacement retention vs controls；

loss-interface source:
  arbitrary loss-interface improvement vs best controls。
```

公式：

$$
Source^{func}_{h}
=
Score^{func}_{FU,h}
-
\max_c Score^{func}_{c,h}.
$$

$$
Source^{loss}_{h}
=
\left(L_{control,h}^{best}-L_{FU,h}\right).
$$

其中 $L$ 来自 generic loss-interface，不是 CE-specific。

### 10.4 记录指标

```text
loss_adapter_name
source_func_h100...h6400
source_loss_h100...h6400
row_positive_count_h100...h6400
control_equivalent_fraction_h100...h6400
R4800_over_3200_func
R4800_over_3200_loss
AUC_loss_step
AUC_loss_time
per_example_loss_q95_debt
per_example_loss_q99_debt
LineC_generic_loss_coupling
metric_debt_Sobolev/RKHS/Fisher
source_state_alignment
source_state_decay_rate
optimizer_destructive_projection
wall_clock_time
step_time
memory
```

### 10.5 C3/C4 gate

C3 source formation：

```text
source_func_h100/h400/h800/h1600/h3200 >= 0.005；
source_loss_h800/h1600/h3200 >= 0 或 task-loss neutral-positive；
row_positive_count_h3200 >= 6/9；
matched controls fail；
loss-agnostic debt not exploded。
```

C4 terminal retention：

```text
source_func_h4800 >= 0.005；
R4800_over_3200_func >= 0.50；
row_h4800_positive_count >= 7/9；
source_loss_h4800 >= 0 或 loss-neutral with strong function-retention source；
stable-random / same-source-random / sign-flip / corrupt controls fail；
AUC_loss_time <= 1.05 * best control；
per-example loss tail debt not worse。
```

如果 C3/C4 只在 function-retention source 过，但 loss-interface source 失败，则 route 写：

```text
TargetRetentionOnly_NotTaskUseful
```

不能 promotion。

---

## 11. Line G：KAN source-channel mapping

### 11.1 进入条件

只有当 MLP 或 generic source path 满足：

```text
C3_source_formation_pass = 1
C4_terminal_retention_pass = 1
loss-interface source non-negative
```

才进入 KAN mapping。

### 11.2 KAN mapping 目标

不要再用轻量 toy KAN 直接证明。v22.11 必须在 D-CHE / D-FOU full functional runner 中执行：

```text
D-CHE low-degree bank；
D-CHE degree-readout；
D-CHE readout-only；
D-FOU low-frequency bank；
D-FOU band-readout；
D-FOU readout-only。
```

D-RAT / D-RBF 只有在 Line B 达到 robust official 或 near-E1 + limited-smoke allowance 时进入：

```text
D-RAT numerator/denominator tangent block；
D-RBF local-center/readout block。
```

### 11.3 KAN 成功标准

```text
KAN source_func_h100/h400/h800/h1600/h3200 >= 0.005；
KAN R4800_over_3200_func >= 0.50；
KAN source_loss_h3200/h4800 non-negative；
KAN_specific_delta_vs_MLP_same_metric >= 0.005；
controls fail；
full functional runner kernel match = 1；
loss-agnostic efficiency gate pass。
```

如果 KAN 失败：

```text
如果 MLP C4 也失败：KANMappingNotEntered；
如果 MLP C4 过但 KAN 失败：KANSourceChannelMismatchConfirmed；
如果 KAN function source 过但 loss source 不过：KANTargetRetentionOnly；
如果 KAN source 过但 efficiency fail：KANEfficiencyContractBlocked。
```

---

## 12. 4GPU 动态队列

v22.11 必须显式使用 4GPU dynamic queue。

```text
GPU0:
  S0.18 code/loss-interface tests；
  D-CHE/D-FOU arbitrary-cotangent efficiency reconfirm；
  D-CHE KAN mapping。

GPU1:
  real-train-stream source atom generation；
  variational source solve；
  MLP arbitrary-loss horizon training。

GPU2:
  D-RAT/D-RBF arbitrary-cotangent multibatch efficiency；
  D-RAT/D-RBF component telemetry / repair。

GPU3:
  metric commit solver；
  PID/source-state arbitrary-loss horizon；
  D-FOU KAN mapping；
  finalizer / figures / packet audit。
```

必须落盘：

```text
v22_11_runnable_queue.csv
v22_11_gpu_assignment_manifest.csv
v22_11_gpu_utilization_timeline.csv
v22_11_idle_violation.csv
v22_11_queue_drain_report.csv
```

硬标准：

```text
queue_drained = 1；
execution_contract_violation = 0；
如果 runnable queue 非空且任一 GPU idle > 10 分钟，不能写 completed no-go。
```

---

## 13. 必须可视化

```text
1. arbitrary-cotangent efficiency dashboard:
   carrier × cotangent_type × forward/backward/step/memory。

2. source atom dashboard:
   DDR vs NDS；B2 gain vs random gap；control projection vs source norm。

3. variational solve dashboard:
   objective decomposition；selected atom weights；control projection residual。

4. commit dashboard:
   projection residual vs ActuationR2；B2 transfer vs random；solve_time vs residual。

5. horizon source trajectory:
   source_func and source_loss from h100 to h6400。

6. source preservation dashboard:
   optimizer destructive projection；source state decay；R4800/h3200。

7. KAN mapping dashboard:
   MLP source decomposition vs D-CHE/D-FOU source-channel energy。

8. failure taxonomy heatmap:
   source atom / variational / commit / horizon / KAN / efficiency blockers。
```

---

## 14. v22.11 最低有效进展定义

v22.11 不能只输出同类 no-go。最低有效进展必须满足至少一项：

```text
A. Code closure:
   final zip 自包含，required source missing=0，required import errors=0，loss-interface tests pass。

B. Arbitrary-loss efficiency closure:
   D-CHE / D-FOU 在 arbitrary upstream cotangent 下 S1 pass；
   或 D-RAT / D-RBF 至少一个 robust official closure。

C. Source construction progress:
   real train-stream source atoms 在至少两个 loss adapters / cotangent types 下通过 S2。

D. Functional progress:
   arbitrary-loss horizon 中 MLP C3/C4 pass，且 source_loss 不为负。

E. KAN progress:
   D-CHE 或 D-FOU 在 full runner 中 KAN source-channel pass，且 loss source 不为负。

F. Theory boundary:
   如果 A-E 都失败，输出 ArbitraryLossConstructiveFU_NoGo_v22.11，明确失败层级：
   source atom / variational solve / commit / horizon loss dynamics / KAN channel / efficiency。
```

---

## 15. 决策规则

### Case 1：source-side 过，但 arbitrary-loss efficiency 不过

结论：

```text
Source-side constructive FU 有算法线索，但不能 promotion。
下一步优先 arbitrary-loss efficiency runner。
```

### Case 2：efficiency 过，但 source_loss 不过

结论：

```text
Target-retention source 不能转成任务 loss source。
下一步重新定义 source atoms，不能继续 PID/replay。
```

### Case 3：MLP arbitrary-loss source 过，KAN 不过

结论：

```text
Generic FU 成立，KAN source-channel mismatch。
下一步做 KAN basis-channel mapping，不做 MLP 小修。
```

### Case 4：MLP 和 KAN 都过 source，但 KAN 不优于 MLP

结论：

```text
FU 是 generic optimizer/training-dynamics insight，不是 KAN-specific。
```

### Case 5：KAN 过 source 且 efficiency 过

结论：

```text
进入 v22.12 S4/S5 official real-task confirmation。
```

---

## 16. 最终总结

v22.10 的最大进展是 constructive source path 终于打开：source atoms、variational solve、commit、PID/source-state replay、KAN source-channel 都出现正证据。但这不是 official success，因为：

```text
1. efficiency 被 arbitrary-loss/upstream-gradient contract 拦住；
2. horizon source 主要是 target-retention / geometry replay，不是任意 task loss training；
3. KAN mapping 仍是 lightweight source-channel proof，不是 full real KAN functional training proof。
```

因此 v22.11 的核心不是继续扩大 source replay，而是：

$$
\boxed{
\text{把 constructive FU 接入 arbitrary-loss training dynamics，}
\text{并重新建立 basis efficiency 的 upstream-cotangent official runner。}
}
$$

只有当 source 在任意 loss-interface 下仍留存，并且 basis 在任意 upstream gradient 下仍高效，DG-KAN 才能从工程正信号进入算法/系统成功。
