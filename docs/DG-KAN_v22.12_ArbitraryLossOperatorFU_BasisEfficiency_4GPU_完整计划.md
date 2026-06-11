# DG-KAN v22.12：Arbitrary-Loss Operator Functional Update + Kernel-Native Cotangent Efficiency + 4GPU 完整计划

> 版本：v22.12 execution plan  
> 生成时间：2026-06-07  
> 目标读者：不假设读者熟悉全部历史实验；本文开头先说明项目总目标、v22.11 真实进展、当前 blocker 和本轮为什么要改方向。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline official path；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific rule；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / ECE / Brier / AUCtime / tail metrics 只能作为 audit / gate / debt readback，不能生成 direction；**functional update 与 efficiency runner 不得针对 CE 设计，CE 只能作为 `LossInterface` 的一个 adapter。**

---

# 0. 项目总目标与 v22.11 后的真实状态

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找一个只在 CE 分类上有效的训练技巧。项目目标是：

$$
\boxed{
\text{strict FC-PureKAN base + arbitrary-loss functional update}
>
\text{strict FC-PureKAN base + ordinary backprop / AdamW controls}
}
$$

这里“更好”必须同时满足四类要求：

```text
1. functional update 的方向不针对 CE，不依赖某个具体任务 loss 的公式；
2. source 在普通继续训练中留存，不只是 target replay 或 h3200 局部 positive；
3. KAN carrier 的 forward / VJP / update / memory 接近同参数 MLP；
4. controls、debt、LineC、calibration、AUCtime、random/sign/corrupt target 都不能解释结果。
```

v22.11 的正向进展很明确：

```text
1. 代码包 required path 基本自包含，required source files 29/29，compile/import/LineC/source-chain/loss-interface tests 通过。
2. S1 runner 已从 CE trainpath 改成 arbitrary upstream cotangent manual-VJP runner。
3. Source-side constructive path 从 v22.10 的 toy target-retention，推进到 v22.11 的 real loss-interface continuation：S2/S3/S4 pass rows = 3/4/2。
4. S5 arbitrary-loss horizon 有 C3=3、C4=1、source_loss_nonnegative=9。
5. S6 KAN mapping 出现 D-FOU readout-source-channel positive rows。
```

但 v22.11 还不能写成最终 scientific success。独立代码和数据审计显示：

```text
1. S1 efficiency 仍是 manual upstream-VJP closure，不是 kernel-native official fused closure；official_fused_kernel_complete_rows = 0。
2. D-FOU / D-RBF 通过 arbitrary-cotangent efficiency gate，但 D-CHE / D-RAT 没有 robust pass。
3. S5 的 best arbitrary-loss horizon 主要来自 Delta-MSEAdapter；CE 与 ranking adapter 下大量 source_func/source_loss 是负的。
4. S6 KAN pass 来自 D-FOU readout source-state replay；D-CHE 多数 row 只是 KANTargetRetentionOnly。
5. S6 代码中 `efficiency_pass = D-CHE_pass or D-FOU_pass` 是全局 carrier gate，不是 per-carrier gate；这会把 D-FOU 的效率 pass 误传播到 D-CHE mapping row 的 `loss_agnostic_efficiency_gate_pass` 字段。
6. Finalizer 的 `promotion_allowed=1` 只检查 D-CHE/D-FOU 任一 pass 与 KAN route opened，没有要求 official fused rows，也没有要求多 adapter robustness；因此它只能算 exploration promotion，不能算 official success。
```

因此我把 v22.11 的独立 route 改写为：

```text
R9a-ArbitraryLossConstructiveSourceOpened-NeedsAdapterRobustnessAndKernelNativeClosure
```

这不是回退。v22.11 解决了 v22.10 的关键问题：**functional source 可以在 arbitrary loss-interface 下构造并部分留存**。但它还没解决两个更高层问题：

```text
1. 这个 source 是否对多个 loss adapters 都稳定，而不是只对 MSE-like target replay 有效；
2. KAN carrier 是否在 kernel-native arbitrary-cotangent training path 中仍高效，并且 source_loss 不只是 target-retention 正。
```

v22.12 的核心目标就是解决这两个问题。

---

# 1. v22.12 的总判断：从“source displacement”升级为“loss-interface operator”

v22.10 / v22.11 的 constructive FU 已经证明：可以构造一个 label-free / loss-agnostic target displacement，并把它写入模型函数中。下一步不能继续只优化一个固定的 target displacement，而要把 FU 定义为一个 **operator**：

$$
\boxed{
\Delta f = T_\theta(\delta)
}
$$

其中：

```text
δ:
  任意 loss interface 给出的 output cotangent，例如 CE、MSE、ranking、preference、source-target、random cotangent。

T_θ:
  当前模型状态下的 functional update operator。

Δf:
  模型函数应该发生的变化。
```

普通训练只知道：

$$
\delta = \frac{\partial \mathcal{L}}{\partial f_\theta(x)}.
$$

Functional update 需要学会：

```text
这个 cotangent 中哪一部分是 coherent signal，哪一部分是 reservoir / noise / control-equivalent；
应该把 signal 写进哪个函数空间通道；
后续 optimizer 训练时如何不把它擦掉；
如果换成另一个 loss adapter，同一个 operator 是否仍然合理。
```

这就是 v22.12 的第一性原则：

$$
\boxed{
\text{Functional update 不应是一个固定 update vector，}
\text{而应是一个把任意上游 cotangent 变成 retained source 的训练动力学 operator。}
}
$$

---

# 2. 当前研究进展如何进入 v22.12

v22.12 不机械替换 optimizer，而是从 optimizer / 稳定训练 / 泛化理论中抽出算法原则。

## 2.1 Muon / low-NDS 的启发

Muon curvature 研究说明：一阶收益相近时，长期训练差异可能来自更低的二阶曲率代价与 Normalized Directional Sharpness，而不是更大的 update norm。v22.12 因此不再只追 `source_func_h3200` 或 `source_loss_h3200`，而是要求记录：

```text
first_order_gain
second_order_penalty
NDS
within_block_NDS
cross_block_NDS
source_retention_delta_vs_NDS
```

如果某个 FU 在 MSE adapter 下短期好，但 NDS 高、Sobolev/RKHS energy 高、h4800 或 h6400 掉，就应判为 high-curvature source，不应继续加 replay scale。

## 2.2 Nora / row-wise angular stability 的启发

Nora 的核心启发不是“换 Nora optimizer”，而是：矩阵参数的 row norm 与 angular velocity 稳定性影响训练稳定。v22.12 对 MLP hidden/readout、D-CHE degree-readout、D-FOU band-readout、D-RAT numerator/denominator、D-RBF local-center/readout 都要记录：

```text
row_norm_drift
row_angular_velocity
radial_component_fraction
tangential_component_fraction
source_alignment_tangential_vs_radial
```

如果 source 被写进 radial jitter，而不是稳定 angular transport，那么它可能在 h3200 后被擦掉。

## 2.3 SOAP / block coordinate 的启发

SOAP 的启发是：Adam-like moment 可以在 slowly-changing coordinate basis 中运行。v22.12 因此不再把 source-state 存成 flat vector，而要按 block 记录：

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

每个 block 都要有自己的 source age、overwrite fraction、slow-state memory、control projection。

## 2.4 AdEMAMix / Schedule-Free 的启发

AdEMAMix 告诉我们，single EMA 不能同时保留近期梯度与长期旧梯度；Schedule-Free / averaging 也提示 fast iterate 与 slow iterate 需要分工。v22.12 把 source state 分成：

```text
fast source state:
  负责 h100/h400/h800 source formation；

slow source state:
  负责 h1600/h3200/h4800/h6400 retention；

terminal preservation state:
  负责防止 optimizer destructive projection。
```

## 2.5 Generalization signal-channel / reservoir 的启发

训练能移动的方向不等于能泛化的方向。Functional update 应识别 coherent signal drift，而不是只看某个 split 的 target retention。v22.12 将 source atom / source operator 评估改为：

```text
signal_drift_energy
reservoir_energy
noise_into_signal_ratio
control_null_residual_norm
drift_diffusion_ratio
microtrajectory_order_consistency
```

## 2.6 Deep Manifold / boundary-conditioned iteration 的启发

如果神经网络是在训练中形成 fixed-point region，那么 FU 不是孤立的一步参数扰动，而是训练流的边界条件控制。v22.12 因此把 FU 设计成：

```text
precommit operator construction
metric commit
optimizer-state integration
finite-window source-state replay
terminal preservation
```

而不是“一个 update 后就交给 AdamW”。

---

# 3. Part A：代码与语义审计硬门

v22.12 首先必须修 v22.11 的两个代码/route 问题：carrier-specific efficiency gate 与 promotion semantics。

## 3.1 必须修复的代码问题

### A1. S6 KAN mapping 的 per-carrier efficiency gate

当前 v22.11 中：

```python
efficiency_pass = D-CHE_pass or D-FOU_pass
```

这必须改成：

```python
efficiency_pass_by_carrier = {
  "D-CHE": D-CHE_pass,
  "D-FOU": D-FOU_pass,
  "D-RAT": D-RAT_pass,
  "D-RBF": D-RBF_pass,
}
```

S6 每一行都必须使用自己的 carrier gate。

记录：

```text
carrier
carrier_specific_efficiency_pass
wrong_global_efficiency_gate_detected
wrong_global_efficiency_gate_fixed
```

硬标准：

```text
wrong_global_efficiency_gate_detected = 0
carrier_specific_efficiency_pass_consistency = 1
```

### A2. Finalizer promotion semantics

v22.11 的 `promotion_allowed=1` 不能继续作为 official success。v22.12 finalizer 必须区分：

```text
exploration_promotion_allowed:
  可以进入下一阶段；

official_promotion_allowed:
  可以写成 scientific success。
```

Official promotion 必须同时满足：

```text
S0 code truth pass;
S1 arbitrary-cotangent efficiency pass with native/kernel or explicitly accepted manual path;
S5 arbitrary-loss horizon pass on >=2 loss adapters;
S6 KAN source-channel pass with carrier-specific efficiency;
source_loss not negative beyond tolerance;
controls fail;
debt not exploded.
```

### A3. CE-specific helper 隔离

允许历史文件保留 CE helper，但 official direction path 必须满足：

```text
official_direction_path_uses_CE_specific_formula = 0
CE_adapter_only_used_as_LossInterface = 1
legacy_CE_helper_not_on_official_path = 1
```

需要 grep + runtime provenance：

```text
cross_entropy
one_hot
softmax
CEp99
LineC
ECE
Brier
AUCtime
```

其中 CE adapter 可以出现，但不能出现在 core FU source construction 的专用公式中。

### A4. S2/S3/S4 loss-interface provenance

v22.12 要明确区分：

```text
uses_loss_interface_cotangent_for_direction = 1
uses_loss_formula_specific_direction = 0
uses_adapter_name_branch = 0
uses_labels_for_adapter_only = 1/0
uses_labels_for_direction = 0
```

CE adapter 使用 labels 是合法的，因为 loss adapter 本身需要 labels；但 core FU 不得根据 adapter name 或 CE formula 调分支。

---

# 4. Part B：基函数效率路线

v22.12 的效率线不再问“是否 CE 下快”，而是问：

$$
\boxed{
\text{给定任意 upstream cotangent } \delta,
\text{basis 的 forward / VJP / update / commit 是否接近 MLP？}
}
$$

## 4.1 统一 arbitrary-cotangent efficiency runner

每个 basis 在同一套 cotangent suite 下测：

```text
Delta-Gaussian
Delta-StableRandom
Delta-SourceTarget
Delta-LossCEAdapter
Delta-MSEAdapter
Delta-RankingAdapter
Delta-PreferenceAdapter smoke
```

注意：CE/RANK/MSE 都只是 adapter，不得进入 basis-specific branch。

每个 row 记录：

```text
carrier
variant
cotangent_type
batch_size
forward_ms
manual_vjp_ms
parameter_update_ms
functional_commit_ms
full_step_ms
forward_memory_mb
backward_memory_mb
workspace_memory_mb
kernel_count
fused_kernel_used
manual_upstream_vjp_used
fallback_kernel_used
official_fused_kernel_complete
component_telemetry_complete
functional_runner_kernel_match
forward_ratio_vs_mlp
vjp_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
gradcheck_pass
upstream_cotangent_contract_pass
```

## 4.2 D-CHE repair

v22.11 中 D-CHE best ratio 局部很好，但 batch/cotangent pass 只有 1/1，整体 blocked。D-CHE 下一步不应该继续大网格，而要拆原因：

```text
1. forward_ratio 只在某些 cotangent / batch pass？
2. VJP path 是否仍 materialize dense degree basis？
3. degree-readout contraction 是否和 arbitrary cotangent runner 共用 kernel？
4. D-CHE k3/k5 是否在 source target cotangent 下退化？
```

Codex 失败后先尝试：

```text
CHE22.12-R1 k3 low-degree upstream-VJP fused path;
CHE22.12-R2 k5 gradbuf no-materialize path;
CHE22.12-R3 readout-only arbitrary-cotangent fast path;
CHE22.12-R4 dense-degree debug path with component telemetry only。
```

通过标准：

```text
>=3/4 batch sizes pass;
>=4/6 non-smoke cotangent types pass;
forward_ratio <= 1.25;
step_ratio <= 1.25;
memory_ratio <= 1.05;
fallback_kernel_used = 0.
```

## 4.3 D-FOU reconfirm

D-FOU 是 v22.11 最稳的 arbitrary-cotangent carrier。v22.12 只需做 reconfirm，不要继续消耗大预算。

必须记录：

```text
FOU-R3 tablelookup-bandreadout official path;
low-frequency path;
source-target cotangent path;
loss-adapter cotangent path;
same_kernel_functional_runner_proof;
```

成功标准同上。

## 4.4 D-RAT repair

D-RAT v22.11 near_E1 rows 很多，但 robust pass 仍失败。D-RAT 的问题不是 memory，而是不同 batch/cotangent 下 forward ratio 不稳。

Codex 失败后先尝试：

```text
RAT22.12-R1 numerator/denominator fused VJP;
RAT22.12-R2 reciprocal approximation + safety clamp;
RAT22.12-R3 telemetry-free train path;
RAT22.12-R4 num/den blockwise update with shared cotangent buffer;
RAT22.12-R5 denominator safety audit separated from training path。
```

记录：

```text
numerator_eval_ms
denominator_eval_ms
reciprocal_ms
safety_clamp_ms
telemetry_ms
train_path_ms
audit_path_ms
denominator_min/p01/condition
num_den_update_cosine
```

通过标准：

```text
near-E1:
  forward_ratio <= 2.0;
  step_ratio <= 1.75;
  memory_ratio <= 1.20;

robust official:
  >=3/4 batch sizes pass;
  >=3/6 cotangent types pass;
  forward_ratio <= 1.25;
  step_ratio <= 1.25;
  memory_ratio <= 1.05;
  component_telemetry_complete = 1.
```

## 4.5 D-RBF repair

D-RBF v22.11 pass 了 arbitrary-cotangent efficiency，但仍要确保不是 micro/path artifact。D-RBF 重点是 local support 是否真的转化成 runtime。

Codex 失败后先尝试：

```text
RBF22.12-R1 local-K4 no-dense train path;
RBF22.12-R2 local-K2 compact support path;
RBF22.12-R3 active-center mask hard sparse path;
RBF22.12-R4 exp approximation path;
RBF22.12-R5 local backward fused scatter path。
```

记录：

```text
active_center_fraction
mean_local_K
basis_materialized_bytes
exp_eval_ms
local_gather_ms
local_backward_ms
center_grad_snr
width_grad_snr
```

通过标准同 D-RAT。

---

# 5. Part C：Functional Update 总体算法

v22.12 的 FU 不再以单个 displacement 为中心，而以 operator 为中心。

## 5.1 Loss-interface operator

定义任意 loss adapter 的 output cotangent：

$$
\delta_a = \frac{\partial \mathcal{L}_a}{\partial f_\theta(x)}.
$$

Functional update operator：

$$
\Delta f_a = T_\theta(\delta_a).
$$

目标不是让某一个 adapter 的 source 好，而是让 $T_\theta$ 在一组 adapter 上稳定：

$$
\mathcal{A}=\{CE, MSE, Ranking, SourceTarget, Random, Preference\}.
$$

Operator 目标：

$$
\max_T
\mathbb{E}_{a\in\mathcal{A}}
\left[-\langle \delta_a, T_\theta(\delta_a)\rangle\right]
-
\lambda D_{diffusion}(T)
-
\mu E_{metric}(T)
-
\nu E_{control}(T)
-
\gamma NDS(T)
-
\kappa Var_{adapter}(T).
$$

其中：

```text
first term:
  任意 loss cotangent 下的线性化 loss improvement；

D_diffusion:
  split / microtrajectory 中的非一致扩散；

E_metric:
  Sobolev / RKHS / Fisher / basis-channel energy；

E_control:
  AdamW / SGD / random / stable-random controls 可解释部分；

NDS:
  high-curvature directional sharpness；

Var_adapter:
  同一 operator 对不同 loss adapter 的方向差异。
```

这才是 “metric 知道怎么用”：metric 不再只是打分，而是作为 operator 的能量与约束。

## 5.2 Source operator atoms

v22.12 的 source atoms 不再是固定 $a_j=\Delta f_j$，而是 operator atoms：

$$
A_j: \delta \mapsto \Delta f_j.
$$

候选：

```text
O1 UpstreamCotangentDrift:
  跨 split coherent upstream-cotangent drift。

O2 SignalReservoirProjection:
  把 δ 投到 train-stream signal channel，压低 reservoir/noise。

O3 ControlNullOperator:
  去除 AdamW/SGD/random/stable-random span 后的 residual。

O4 LowNDSOperator:
  保留一阶收益，降低 NDS。

O5 TrainFlowCommutatorOperator:
  选择 microtrajectory order-insensitive direction。

O6 FastSlowSourceOperator:
  fast state 形成 source，slow state 保护 source。

O7 RowOrthogonalMatrixOperator:
  控制 row norm drift 与 angular velocity。

O8 KANBasisChannelOperator:
  D-CHE low-degree、D-FOU low-frequency、D-RAT num/den、D-RBF local-center/readout。

O9 AdapterInvariantOperator:
  同一个 operator 在 CE/MSE/ranking/source target 上方向一致。
```

每个 operator atom 记录：

```text
operator_id
carrier
block_role
cotangent_types_supported
uses_labels_for_adapter_only
uses_loss_formula_specific_direction
operator_linearity_error
operator_homogeneity_error
adapter_direction_variance
B2_transfer_gain_by_adapter
source_loss_gain_by_adapter
control_projection_fraction
control_null_residual_norm
DDR
NDS
metric_energy_L2/Fisher/Sobolev/RKHS
row_angular_velocity
radial_fraction
tangential_fraction
operator_function_cosine_with_controls
```

## 5.3 Operator variational solve

把 operator atoms 组合：

$$
T_w = \sum_j w_j A_j.
$$

求：

$$
w^* = \arg\max_w
\left[
\mathbb{E}_{a\in\mathcal{A}} -\langle \delta_a, T_w(\delta_a)\rangle
-
\lambda D_{diffusion}(T_w)
-
\mu E_{metric}(T_w)
-
\nu E_{control}(T_w)
-
\gamma NDS(T_w)
-
\kappa Var_{adapter}(T_w)
\right].
$$

记录：

```text
operator_combo_atom_count
operator_combo_l1_norm
adapter_invariance_score
expected_loss_linear_gain
source_loss_gain_CE/MSE/Ranking/SourceTarget
control_projection_fraction
NDS
metric_energy
operator_variational_pass
```

通过标准：

```text
adapter_invariance_score >= 0.50
expected_loss_linear_gain > random / sign / corrupt controls
control_projection_fraction <= 0.25
NDS <= control_NDS_median
metric_energy <= control_metric_p50 or justified by gain
operator_variational_pass = 1
```

## 5.4 Metric-as-dynamics commit

函数空间方向：

$$
\Delta f_a^* = T_{w^*}(\delta_a).
$$

参数 commit：

$$
u^* = \arg\min_u
\mathbb{E}_{a\in\mathcal{A}}
\|J_\theta u - \Delta f_a^*\|_{G_f}^2
+
\rho\|u\|_P^2
+
\eta E_{block}(u).
$$

Solver 分层：

```text
S4.1 readout exact solve
S4.2 readout all-train fit
S4.3 hidden/readout block CG
S4.4 optimizer-state write only
S4.5 source-state integrated all-train
S4.6 KAN basis-channel block solve
```

记录：

```text
projection_residual_Gf
ActuationR2
B2_transfer_gain_by_adapter
function_displacement_cos_with_target
function_displacement_cos_with_controls
JVP_count
VJP_count
solve_time_ms
commit_time_ms
source_state_write_fraction
optimizer_state_write_fraction
block_energy_hidden/readout/basis
```

通过标准：

```text
projection_residual_Gf <= 0.25
ActuationR2 >= 0.70
B2_transfer_gain > random/sign/corrupt controls
function_displacement_cos_with_target >= 0.70
commit_time_ratio_vs_mlp <= 1.25
```

---

# 6. Part D：Arbitrary-loss horizon training

v22.11 的 S5 best row 主要来自 Delta-MSEAdapter。v22.12 必须证明 source 不只在一个 adapter 上成立。

## 6.1 Horizon protocol

对每个 selected operator/commit：

```text
Adapters:
  Delta-MSEAdapter
  Delta-LossCEAdapter
  Delta-RankingAdapter
  Delta-SourceTarget
  Delta-StableRandom control

Horizons:
  h100, h400, h800, h1600, h2400, h3200, h4000, h4800, h6400
```

每个 FU attempt 与 controls 使用同一 adapter，不允许 FU 根据 adapter 名称走特殊 branch。

## 6.2 记录指标

```text
loss_adapter_name
attempt
source_func_h*
source_loss_h*
row_positive_count_h*
control_equivalent_fraction_h*
R4800_over_3200_func
R4800_over_3200_loss
AUC_loss_step
AUC_loss_time
AUC_loss_time_ratio_vs_best_control
per_example_loss_q95_debt
per_example_loss_q99_debt
LineC_generic_loss_coupling
metric_debt_Sobolev/RKHS/Fisher
source_state_alignment
source_state_decay_rate
optimizer_destructive_projection
TargetRetentionOnly_NotTaskUseful
C3_source_formation_pass
C4_terminal_retention_pass
```

## 6.3 C3/C4 gates

C3 source formation gate：

```text
source_func_h100 >= 0.005
source_func_h400 >= 0.005
source_func_h800 >= 0.005
source_func_h1600 >= 0.005
source_func_h3200 >= 0.005
source_loss_h3200 >= -1e-6
row_positive_count_h3200 >= 6/9
matched controls fail
```

C4 terminal gate：

```text
source_func_h4800 >= 0.005
R4800_over_3200_func >= 0.50
source_loss_h4800 >= -1e-6
row_positive_count_h4800 >= 7/9
stable-random control fails
AUC_loss_time_ratio <= 1.05
debt not exploded
```

Adapter robustness gate：

```text
C3 pass on >= 2 non-random adapters;
C4 pass on >= 1 non-random adapter for exploration;
C4 pass on >= 2 non-random adapters for official promotion;
CE adapter may pass or fail, but failure must not be hidden by MSE-only success.
```

If CE fails but MSE/ranking pass：

```text
route = LossAdapterSpecificSourceOpened_NotUniversal
```

If only source_func passes but source_loss negative：

```text
route = TargetRetentionOnly_NotTaskUseful
```

---

# 7. Part E：KAN source-channel mapping

v22.11 的 KAN positive 来自 D-FOU readout replay，D-CHE 多数是 target retention only。v22.12 要把 KAN mapping 分成三类：

```text
readout source channel;
basis-estimate / readout-commit;
true basis-channel commit.
```

## 7.1 Carrier-specific mapping gate

每个 KAN row 必须使用自己的 carrier efficiency pass：

```text
D-CHE row -> D-CHE_pass
D-FOU row -> D-FOU_pass
D-RAT row -> D-RAT_pass
D-RBF row -> D-RBF_pass
```

如果 carrier-specific efficiency 不过：

```text
KANSourceExistsEfficiencyBlocked
```

不能因为另一个 carrier 过而放行。

## 7.2 Mapping strategies

```text
K1 readout-only source-state replay
K2 basis-estimate / readout-commit
K3 low-degree D-CHE bank commit
K4 low-frequency D-FOU bank commit
K5 D-RAT numerator/denominator tangent block commit
K6 D-RBF local-center/readout commit
K7 optimizer-state integrated source replay
```

## 7.3 KAN metrics

```text
KAN_source_func_h*
KAN_source_loss_h*
KAN_specific_delta_vs_MLP_same_metric
R4800_over_3200_func
R4800_over_3200_loss
carrier_specific_efficiency_pass
full_functional_runner_kernel_match
basis_channel_energy
readout_channel_energy
basis_estimate_fit_cosine
basis_commit_projection_residual
source_state_decay_rate
```

## 7.4 KAN gates

Exploration pass：

```text
carrier_specific_efficiency_pass = 1
KAN_source_func_h100/h400/h800/h1600/h3200 >= 0.005
KAN_source_loss_h3200 >= -1e-6
R4800_over_3200_func >= 0.50
KAN_specific_delta_vs_MLP_same_metric >= 0.005
controls fail
```

Official pass：

```text
exploration pass
source_loss_h4800 >= -1e-6
row_positive_count_h4800 >= 7/9
>=2 independent seeds / reruns
same source-channel method not target-retention-only
```

If only readout-only passes：

```text
route = KANReadoutSourceOpened_BasisChannelStillOpen
```

If basis-estimate fit high but source_loss negative：

```text
route = KANBasisActuationTargetRetentionOnly
```

---

# 8. Controls and forbidden-information policy

Each S2/S3/S4/S5/S6 run must compare:

```text
NoOpMatchedOverhead
RandomMatchedNorm
StableRandom
SignFlipTarget
CorruptTarget
SameSolverRandomTarget
AdamWParallelDirection
SGDParallelDirection
OptimizerStateOnlyNoSource
TargetOnlyNoLossControl
```

Forbidden:

```text
validation/test/future/query;
dataset-name branch;
seed-specific branch;
CE-specific source formula;
LineC/ECE/Brier/AUCtime/tail as direction;
classwise CE-tail hard target;
loss adapter name branch.
```

Allowed:

```text
current train-stream logits;
current hidden/readout/basis activations;
current upstream cotangent from generic LossInterface;
past train-stream source state;
current optimizer state;
current batch/micro-split statistics;
labels only inside supervised loss adapter, not inside FU core.
```

---

# 9. 4GPU dynamic execution plan

v22.12 must use the four GPUs explicitly and produce real queue artifacts.

```text
GPU0:
  S0 code truth gate;
  D-CHE arbitrary-cotangent repair;
  D-CHE KAN mapping.

GPU1:
  D-FOU reconfirm;
  MLP operator source / S5 horizon;
  D-FOU KAN mapping.

GPU2:
  D-RAT arbitrary-cotangent repair;
  D-RAT limited KAN mapping only if S1 near/official pass.

GPU3:
  D-RBF arbitrary-cotangent repair;
  cross-adapter controls;
  report/finalizer/figures.
```

Must output:

```text
v22_12_runnable_queue.csv
v22_12_gpu_assignment_manifest.csv
v22_12_gpu_utilization_timeline.csv
v22_12_idle_violation.csv
v22_12_queue_drain_report.json
v22_12_deferred_items.csv
```

Execution contract:

```text
if runnable_queue_nonempty and any GPU idle > 10 minutes:
  execution_contract_violation = 1
  final route cannot be completed_no_go
```

---

# 10. Required visualizations

## 10.1 Code / semantic dashboard

```text
source-file closure heatmap
CE-specific formula usage path graph
LineC/audit metric direction firewall graph
semantic alias cosine heatmap
per-carrier efficiency gate consistency table
```

## 10.2 Efficiency dashboard

```text
forward ratio by carrier × cotangent × batch
VJP ratio by carrier × cotangent × batch
step ratio by carrier × cotangent × batch
memory ratio by carrier × cotangent × batch
component waterfall: forward / VJP / update / commit / audit
kernel-native vs manual-upstream-VJP comparison
```

## 10.3 Functional operator dashboard

```text
operator atom score Pareto: expected loss gain vs control projection
adapter invariance heatmap
source_func trajectory by adapter
source_loss trajectory by adapter
source_state decay curve
optimizer destructive projection curve
C3/C4 pass matrix by adapter × attempt
```

## 10.4 KAN mapping dashboard

```text
KAN source_func trajectory by carrier
KAN source_loss trajectory by carrier
readout vs basis channel energy bar
KAN_specific_delta_vs_MLP_same_metric scatter
basis actuation residual vs retained source scatter
```

---

# 11. Failure taxonomy and Codex next actions

## Case A：code packet or semantic gate fails

If:

```text
self_contained_import_check = 0
or CE_specific_formula_in_core_FU = 1
or wrong_global_efficiency_gate_detected = 1
```

Codex must first repair code and rerun S0 only. No scientific no-go is allowed.

## Case B：D-CHE arbitrary-cotangent efficiency fails

Try in order:

```text
1. k3 low-degree fused VJP;
2. k5 gradbuf no-materialize;
3. readout-only fast path;
4. component telemetry isolation;
5. if still fails, classify D-CHE-ArbitraryCotangentKernelBlocked.
```

## Case C：D-RAT efficiency fails

Try in order:

```text
1. num/den fused forward;
2. reciprocal approximation;
3. remove telemetry from train path;
4. denom safety path separated;
5. if still fails, D-RAT limited smoke only.
```

## Case D：source operator only works on Delta-MSEAdapter

Do not promote. Try:

```text
1. adapter-balanced variational objective;
2. cotangent norm normalization;
3. loss-adapter invariant atom weighting;
4. source-loss-aware horizon replay;
5. if still MSE-only, route = AdapterSpecificSourceOpened_NotUniversal.
```

## Case E：source_func passes but source_loss negative

Do not promote. Try:

```text
1. add source_loss term to variational objective;
2. reduce target-retention replay scale;
3. enforce loss-adapter source_loss nonnegative at h3200;
4. if still fails, TargetRetentionOnly_NotTaskUseful.
```

## Case F：KAN D-FOU passes but D-CHE fails

Do not write KAN-general success. Try:

```text
1. D-CHE readout-only source-state replay;
2. D-CHE low-degree bank commit;
3. D-CHE basis-estimate/readout-commit;
4. if still target-only, route = DFOUReadoutOnlyKANSourceOpened_DCHEMismatch.
```

## Case G：all functional gates pass but efficiency official fused rows still zero

Write:

```text
FunctionalExplorationSuccess_SystemEfficiencyOfficializationPending
```

and next action must be kernel-native arbitrary-cotangent path.

---

# 12. Final route definitions

v22.12 finalizer must choose one of:

```text
R0-CodeOrSemanticGateFailed
R1-ArbitraryCotangentEfficiencyBlocked
R2-SourceOperatorNoGo
R3-AdapterSpecificSourceOpened_NotUniversal
R4-TargetRetentionOnly_NotTaskUseful
R5-MLPArbitraryLossSourceOpened_KANNotEntered
R6-DFOUReadoutSourceOpened_DCHEMismatch
R7-KANSourceExistsEfficiencyBlocked
R8-FunctionalExplorationSuccess_SystemEfficiencyOfficializationPending
R9-ArbitraryLossOperatorFU_KANExplorationSuccess
R10-OfficialPromotionReady
```

Official promotion requires:

```text
S0 pass;
S1 pass with carrier-specific gates;
S2/S3/S4 pass;
S5 C3 on >=2 adapters and C4 on >=1 adapter;
S6 KAN source on carrier-specific efficient carrier;
source_loss nonnegative at h3200/h4800;
controls fail;
debt not exploded;
no CE-specific direction;
4GPU contract pass.
```

If fused official rows remain zero but all functional gates pass, route must be R8, not R10.

---

# 13. Minimum effective progress

v22.12 cannot return another vague no-go. At least one of the following must be delivered:

```text
A-CodeTruth:
  S6 carrier-specific efficiency bug fixed and finalizer promotion semantics split.

B-Efficiency:
  D-CHE or D-RAT arbitrary-cotangent robust pass improves from v22.11.

C-Functional:
  C3/C4 source-loss pass on at least two non-random adapters.

D-KAN:
  D-FOU retained source confirmed with carrier-specific efficiency and source_loss nonnegative;
  or D-CHE source-channel opens.

E-Theory boundary:
  If all above fail, output precise no-go:
  which layer failed: operator atom, variational solve, commit, adapter horizon, KAN mapping, or efficiency.
```

---

# 14. Final expected interpretation

The main scientific question for v22.12 is:

$$
\boxed{
\text{Can we build a loss-interface operator } T_\theta
\text{ that converts arbitrary upstream cotangent into retained source,}
\text{ rather than a target replay that only works for one adapter?}
}
$$

If yes, DG-KAN functional update finally becomes a general training-dynamics algorithm.

If no, the result is still useful: we will know whether the current constructive FU is only a target-retention mechanism, whether the source side is adapter-specific, or whether KAN carrier source-channel mapping is still the central blocker.
