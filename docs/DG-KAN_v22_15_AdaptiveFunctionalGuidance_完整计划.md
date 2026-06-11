# DG-KAN v22.15 Adaptive Functional Guidance 完整实验计划

> 版本：v22.15-AFG / Adaptive Functional Guidance  
> 目标：把 functional update 从“单次 source 写入 + 事后修补”升级为“训练过程中的自适应函数空间控制律”。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no validation/test/future/query direction；no dataset-name branch；no seed-specific rule；no sampler/class weight；no audit metric as direction；CE/MSE/Ranking/Preference 只能作为 LossInterface 或 train-only loss geometry provider，不允许 core FU 按 adapter 名称分支；auxiliary loss 只能作为 upper-bound diagnostic，不能作为 strict official FU。

---

# 0. 这版计划为什么必须改方向

v22.14 以前的 functional update 很多时候还是这种思路：

```text
先构造一个 source；
让模型继续普通训练；
等到 h3200/h4800 看 source 是否还在；
如果不在，再尝试 periodic replay / prox / anchor 修补。
```

这个思路的根本问题是：**source 被擦掉之后再修，往往已经晚了。** 训练过程不是一条被动轨迹，functional update 也不应该是外部补丁。它应该在训练过程中持续判断：普通梯度这一步是否会破坏未来可保留的 source，是否正在把 useful signal 推进 high-curvature / reservoir / readout-only channel，是否需要调整当前 step 的函数空间方向。

因此 v22.15 的主命题是：

$$
\boxed{
\text{Functional update 不是一次性 update，}
\text{而是训练流中的自适应函数空间控制律。}
}
$$

更具体地说，当前普通训练每步做：

$$
g_t = \nabla_\theta L_t,
\qquad
\theta_{t+1}=\theta_t+u_t^{AdamW}.
$$

v22.15 要做的是：

$$
\delta_t = \frac{\partial L_t}{\partial f_{\theta_t}},
\qquad
s_t = T_{\theta_t}(\delta_t, \mathcal{C}_t, z_t),
$$

其中 $\delta_t$ 是当前 loss 给出的 output cotangent，$\mathcal{C}_t$ 是 train-only loss geometry context，例如 pointwise mask、pair incidence matrix、preference graph、sequence mask，$z_t$ 是当前 source-state / slow-memory。

最终参数更新不是等 source 掉了再修，而是在当前 step 就解一个带 source-control 的更新：

$$
u_t^*
=
\arg\min_u
\left[
\langle g_t,u\rangle
+
\frac{1}{2\eta_t}\|u-u_t^{base}\|_{P_t}^2
+
\lambda_t\|J_tu-q_t\|_{G_f}^2
+
\mu_t\|B_tJ_tu-q^{pair}_t\|_{G_{pair}}^2
+
\nu_t\|P_{control}J_tu\|^2
\right].
$$

这里 $u_t^{base}$ 是 AdamW/SGD 给出的普通 step，$J_t$ 是当前模型 Jacobian，$q_t$ 是希望本步在函数空间推进或保护的 source velocity，$B_t$ 是 pairwise / preference geometry 的 incidence operator。关键是 $\lambda_t,\mu_t,\nu_t$ 不是固定超参，而是由 train-only risk controller 自适应决定。

v22.15 要验证的不是“能不能补回来”，而是：

$$
\boxed{
\text{能否在 source 被擦掉之前，预测并约束普通反传的破坏性方向，}
\text{从而驾驭整个训练过程。}
}
$$

---

# 1. 当前研究进展的真实启发

## 1.1 效率路线已经不再是 D-CHE / D-FOU 的主要科学瓶颈

v22.12 还是 manual upstream-VJP，`official_fused_kernel_complete_rows=0`，所以那时只能说明 arbitrary cotangent manual path 有希望，不能说明 kernel-native official 成功。v22.13 开始把 D-CHE / D-FOU 推进到 generic output-cotangent fused VJP，v22.14 repair 继续把 D-CHE / D-FOU 的 native arbitrary-cotangent efficiency 推向 robust grid。这个趋势说明：D-CHE / D-FOU 已经可以作为 v22.15 adaptive FU 的主效率 carrier。

v22.15 不应该继续把效率路线写成单独 profiler，而要测完整 controller-in-loop 的成本：

```text
ordinary forward/backward;
output cotangent extraction;
T(delta, C, z) source operator;
cheap risk monitor;
adaptive low-rank prox solve;
KAN basis/readout commit;
source-state update;
full training step wall-clock;
memory.
```

也就是说，效率路线要服务于 adaptive FU，而不是只证明 basis 单独快。

## 1.2 v22.12 到 v22.14 已经说明：只构造 source 不够，必须控制训练流

v22.12 的核心改变是从固定 displacement $\Delta f$ 走向 loss-interface operator：

$$
\Delta f = T_\theta(\delta).
$$

v22.13 用 source-anchor retention 让多 adapter horizon 出现强信号，但它依赖 anchor retention，不能直接写成 strict no-loss FU。v22.14 把 anchor split 成 operator-only、periodic source-state、auxiliary loss anchor、optimizer-prox、source-state projector。结果告诉我们：

```text
operator_only 不够；
auxiliary loss 能作为 upper bound 说明 source 可以被保护，但它改了 loss；
optimizer_prox_jacobian 在 CE/MSE 上出现 no-loss-modification 的真实信号；
Ranking/Preference 仍然失败；
random/stable-random controls 没有跟着通过。
```

这带来一个核心 insight：

$$
\boxed{
T_\theta(\delta) \text{ 负责产生 source，}
\text{但 source 的长期价值取决于训练流控制律。}
}
$$

因此 v22.15 不再主打“一次 update 后观察 h4800”，而是主打 **operator-in-loop + predictive source-risk controller**。

## 1.3 KAN 现在的问题不是“算得慢”，而是 basis 没承载 source

v22.14 的 K18/K19/K20 指向一个很清楚的问题：当前 target 很大程度是 readout-dominated。readout projection 很高，basis projection 很低；direct basis-state operator 能解出 basis update，但 source_loss / retention gate 没过。这说明不能再把 MLP/readout target 硬塞给 KAN basis。

v22.15 的 KAN 目标必须改成：

$$
\boxed{
\Delta b_t = T^B_{\theta_t}(\delta_t,\mathcal{C}_t,z^B_t),
\qquad
\Delta f^B_t = J^B_t\Delta b_t.
}
$$

也就是说，KAN 的独特价值必须来自 basis-native source velocity，而不是 readout replay。

## 1.4 哪些旧方向可能因为代码或问题定义改变而值得重开

以下旧方向不能按过去的失败直接判死，但必须换身份：它们不能作为最终 update 直接 promotion，只能作为 adaptive controller 的组成部分或风险预测器。

### 1.4.1 F53/F70/F71/F72/F73/F74 terminal source 近似成功行

这些旧行的共同特征是 h3200 能打开，h4800 掉。过去继续调 terminal floor 没意义，但它们对 v22.15 很有价值：它们提供了 source 在进入 late washout 之前的轨迹样本。也就是说，它们可以训练或校准 `washout risk predictor`。

保留方式：

```text
不再作为 update family；
作为 source-risk 监督数据和 ablation baseline；
提取 h800->h1600、h1600->h3200、h3200->h4800 的衰减特征；
用来判断 adaptive controller 是否真的提前介入。
```

### 1.4.2 Train-loss selector / train-loss h400-h800 predictor

过去 train-loss selector 不能作为 direction，因为很容易变成选择器作弊或 post-hoc filter。但 v22.0 的结果说明 train-loss h400/h800 对 early-chain 有预测力。v22.15 可以把它降级为 **risk feature**，而不是 update selector。

允许用途：

```text
作为 risk controller 的输入之一；
只影响 lambda_t / source refresh rate / prox strength；
不能直接选择 dataset/seed/mechanism；
不能用 validation/test/future。
```

### 1.4.3 Split-consensus / control-null / metric observer

这些路线过去作为 filter/observer 不够，因为它们只是在候选池里筛 update。v22.15 可以重开，但必须进入 $T_\theta(\delta,\mathcal{C},z)$ 的定义：

```text
split-consensus -> coherent source subspace;
control-null -> 去掉 AdamW/SGD/random 可解释分量;
metric observer -> G_f / NDS / Fisher/Sobolev energy；
row-angular stability -> risk feature 和 block update regularizer。
```

### 1.4.4 D-CHE corrected layout 之后的旧 D-CHE no-go

v22.12 的 K14 corrected layout 说明历史 D-CHE source mismatch 有一部分可能是 layout bug 污染。v22.15 要重开 D-CHE，但不再用 readout replay route，而是在 corrected layout 上做 basis-native adaptive controller。

### 1.4.5 Row-axis control-null 对 Ranking/Preference 的污染

v22.13 修过 row-axis control-null，因为它会擦掉 Ranking/Preference 的 row-wise cotangent。v22.14 Ranking/Preference 仍失败，说明问题没有完全由这个 bug 解释，但它提醒 v22.15：pairwise loss 不能被 pointwise whitening/control-null 误删。Ranking/Preference 必须走 loss-geometry-aware route。

## 1.5 哪些旧方向应停止作为主线

以下方向不再作为主线：

```text
1. 单次 FU commit 后等 h4800；
2. 固定 periodic replay，不看 source risk；
3. 只调 terminal floor / lookahead / hold / fallback；
4. 把 auxiliary loss anchor 当 strict FU；
5. readout target replay 到 KAN basis；
6. 用 source_func 正忽略 source_loss 负；
7. carrier-level OR efficiency gate；
8. MSE-only success 写成 arbitrary-loss success。
```

它们可以作为 diagnostic / upper bound / negative control，但不能进入 official promotion。

---

# 2. v22.15 总体目标

v22.15 的实验总目标分成四个层次。

## 2.1 训练控制目标

证明 adaptive FU 能在训练过程中提前预测普通梯度对 source 的破坏，并自适应调整 step：

$$
\boxed{
\text{Predict before source is erased, guide before trajectory leaves source manifold.}
}
$$

具体目标：

```text
1. 不等 source_retained < threshold 才修；
2. 用 predicted next-step source drift / destructive projection / curvature risk 提前调节 lambda_t；
3. source controller 进入每个训练 step 的 update rule；
4. full task loss 不改，strict official 不允许 auxiliary loss；
5. random/stable-random/sign-flip controls 不能跟着过。
```

## 2.2 Functional update 算法目标

把 FU 从：

$$
\Delta f = T_\theta(\delta)
$$

升级为：

$$
\boxed{
\Delta f_t = T_{\theta_t}(\delta_t,\mathcal{C}_t,z_t),
\qquad
u_t^* = \Pi_{source}(u_t^{base}; \Delta f_t, z_t, r_t).
}
$$

也就是说，FU 不是只产生 direction，还要带 source-state、risk controller、adaptive prox。

## 2.3 KAN carrier 目标

证明 D-CHE / D-FOU 不只是 high-efficiency readout carrier，而能成为 basis-native adaptive controller 的 efficient carrier：

$$
\boxed{
\text{D-CHE / D-FOU basis state can carry adaptive FU source with MLP-like step cost.}
}
$$

## 2.4 Task-level 目标

只有在 functional 和 KAN carrier gates 通过之后，才进入 task proof：

```text
KAN+AdaptiveFU > KAN+AdamW/BP;
KAN+AdaptiveFU >= same-param MLP controls;
convergence faster;
forgetting lower;
expression metrics better;
calibration/tail debt not worse.
```

---

# 3. 核心算法：Adaptive Functional Guidance

## 3.1 核心状态

每个 run 维护以下 train-only 状态：

```text
source_state z_t:
  当前希望保留/推进的函数空间 source velocity。

basis_source_state z^B_t:
  KAN basis-native source state。

source_age:
  当前 source 自生成以来的 step 数。

source_usefulness:
  当前 source 对 task loss 的线性化贡献。

source_retention:
  当前模型 displacement 在 source 方向上的保留程度。

washout_risk:
  预测未来 50/100/200 steps 内 source 被擦掉的概率。

controller_strength lambda_t:
  本步 source-prox 的强度。
```

## 3.2 每步流程

v22.15 的核心训练循环如下。

```python
for step in range(total_steps):
    logits = model(x_train_batch)
    loss = loss_adapter.value(logits, train_targets)

    # ordinary task gradient; loss itself is not modified
    g_theta = grad(loss, model.parameters())
    u_base = adamw_candidate_step(g_theta, optimizer_state)

    # output cotangent and train-only geometry context
    delta = loss_adapter.cotangent(logits, train_targets)
    C = loss_adapter.geometry_context(logits, train_targets)

    # source update is not necessarily every step expensive solve;
    # it has cheap every-step monitor and adaptive refresh
    if step >= warmup and should_refresh_source(step, risk_state, source_age):
        s_new = T_theta(delta, C, z_source, model_state)
        z_source = adaptive_source_state_update(z_source, s_new, delta, C)

    # before applying u_base, predict its effect on source
    predicted = predict_next_source_state(
        model=model,
        u_base=u_base,
        z_source=z_source,
        delta=delta,
        C=C,
    )

    risk = washout_risk_controller(predicted, source_history, optimizer_state)
    lambda_t = controller_strength(risk, source_usefulness, curvature, source_age)

    # adaptive FU is part of the step, not a late repair
    u_star = solve_source_guided_step(
        u_base=u_base,
        z_source=z_source,
        delta=delta,
        C=C,
        lambda_t=lambda_t,
        model=model,
    )

    apply_update(model, u_star)
    update_optimizer_state(optimizer_state, g_theta, u_star)

    record_train_only_diagnostics()
```

The important point is that FU is **always present as a controller**, but its strength can be almost zero when the predicted ordinary step is safe. This differs from the previous reactive approach:

```text
old reactive prox:
  wait until source_retained is low, then repair.

v22.15 adaptive guidance:
  predict whether the next ordinary step will reduce future retained source;
  adjust the current step before source is lost.
```

## 3.3 Risk controller

The controller estimates risk from train-only quantities:

$$
r_t = \sigma\left(
 a_0
 +a_1(-\widehat{\Delta S}_{t+1})
 +a_2 D^{destruct}_t
 +a_3 NDS_t
 +a_4 E^{control}_t
 +a_5 B^{pair}_t
 +a_6 E^{basis\_leak}_t
 +a_7 \mathbf{1}[source\_loss\_boundary]
\right).
$$

Where:

```text
predicted_source_drift:
  predicted change in source projection after u_base.

destructive_projection:
  how much J_t u_base points against z_t.

NDS:
  normalized directional sharpness of the proposed step.

control_energy:
  fraction explainable by AdamW/SGD/random/stable-random controls.

pairwise_risk:
  whether pairwise margin structure is being collapsed.

basis_leak:
  whether KAN basis source is leaking to readout/reservoir.

source_loss_boundary:
  whether source_func is retained but source_loss is about to turn negative.
```

Then:

$$
\lambda_t = \lambda_{min} + (\lambda_{max}-\lambda_{min})r_t.
$$

If $source\_loss\_boundary=1$ and $source\_usefulness<0$, the controller must **release or refresh** source, not blindly preserve it.

## 3.4 Source-state update

The source state is not a fixed anchor. It evolves:

$$
z_t = \operatorname{Norm}\left((1-\alpha_t)\mathcal{T}_{t-1\rightarrow t}(z_{t-1}) + \alpha_t s_t\right).
$$

The gate $\alpha_t$ is:

$$
\alpha_t
=
\operatorname{clip}\left(
\alpha_0
+c_1\operatorname{Gain}(s_t)
+c_2\operatorname{Coherence}(s_t)
-c_3\operatorname{NDS}(s_t)
-c_4\operatorname{ControlProj}(s_t)
-c_5\mathbf{1}[source\_loss(s_t)<0],
0,
\alpha_{max}
\right).
$$

This prevents two old failure modes:

```text
1. source never refreshes and becomes stale;
2. source refreshes too often and becomes noise-following target drift.
```

## 3.5 Strict official vs diagnostic mechanisms

Strict official adaptive FU allows:

```text
task loss unchanged;
operator reads δ, C, train-stream state, optimizer state;
controller changes update rule / optimizer step;
source-state memory train-only;
no validation/test/future/query;
no adapter-name branch.
```

Diagnostic-only mechanisms:

```text
auxiliary loss anchor;
explicit source retention loss added to training objective;
source chosen using future horizon labels from same run;
post-hoc h4800 selector;
readout replay into KAN basis without basis-native objective.
```

---

# 4. Part A：代码审计与语义闭合

## 4.1 目标

v22.15 的第一步必须确保 code packet 和 route semantics 支撑 adaptive controller，不允许再出现 packet 缺 transitive dependency、finalizer 忽略执行合同、auxiliary loss 被写成 strict success、operator core 偷读 adapter name 的问题。

## 4.2 必须新增或修复的源码

Codex 必须新增或更新：

```text
dgkan/fu/adaptive_controller.py
  risk controller, lambda scheduler, source-state update.

dgkan/fu/loss_geometry.py
  pointwise context, pair incidence context, preference graph context.

dgkan/fu/source_state_transport.py
  z_{t-1} -> z_t transport and stale-source detection.

dgkan/fu/source_guided_step.py
  source-aware optimizer-prox solve without loss modification.

dgkan/fu/basis_native_controller.py
  D-CHE / D-FOU basis-state adaptive source solve.

dgkan/profiling/adaptive_fu_efficiency.py
  full controller-in-loop timing.

experiments/run_v22_15_s0_truth.py
experiments/run_v22_15_adaptive_mlp_lab.py
experiments/run_v22_15_loss_geometry_operator.py
experiments/run_v22_15_kan_basis_controller.py
experiments/run_v22_15_efficiency_controller_loop.py
experiments/run_v22_15_task_readback.py
experiments/run_v22_15_finalize.py
```

The packet must include all transitive dependencies, especially model definitions and profiling dependencies:

```text
dgkan/models/**
dgkan/kernels/**
dgkan/profiling/**
dgkan/fu/**
experiments/run_v22_15_*.py
```

## 4.3 S0 checks

Record:

```text
clean_unzip_compileall_pass
clean_unzip_import_pass
missing_transitive_dependency_count
operator_core_adapter_name_branch_count
loss_formula_branch_in_core_count
uses_loss_modification_for_strict_rows
auxiliary_rows_diagnostic_only
context_provider_adapter_name_allowed_only_in_LossInterface
risk_controller_uses_validation_test_future
finalizer_reads_execution_contract
finalizer_blocks_auxiliary_official
adaptive_controller_unit_tests_pass
source_state_transport_unit_tests_pass
pairwise_geometry_unit_tests_pass
basis_controller_layout_unit_tests_pass
```

Pass criteria:

```text
clean_unzip_compileall_pass = 1
clean_unzip_import_pass = 1
missing_transitive_dependency_count = 0
operator_core_adapter_name_branch_count = 0
loss_formula_branch_in_core_count = 0
risk_controller_uses_validation_test_future = 0
uses_loss_modification_for_strict_rows = 0
auxiliary_rows_diagnostic_only = 1
finalizer_blocks_auxiliary_official = 1
adaptive_controller_unit_tests_pass = 1
```

## 4.4 Synthetic controller unit tests

Before real runs, Codex must create toy tests where source is known.

Test 1: ordinary step destroys source.

```text
Construct z and u_base with <J u_base, z> < 0.
Expected: risk high, lambda_t high, corrected update has larger <J u_star, z>.
```

Test 2: ordinary step already supports source.

```text
Construct z and u_base with <J u_base, z> > 0.
Expected: risk low, lambda_t near zero, u_star close to u_base.
```

Test 3: source_loss boundary negative.

```text
Construct z with positive retention but negative linearized source_loss.
Expected: controller releases/refreshed source, not preserve stale z.
```

Test 4: pairwise antisymmetry.

```text
Construct pair incidence B.
Expected: pairwise operator preserves antisymmetric margin direction; row-mean whitening cannot erase B-space signal.
```

If these unit tests fail, no scientific no-go is allowed; Codex must repair Part A first.

---

# 5. Part B：Efficiency route for adaptive FU

## 5.1 目标

The efficiency question is no longer only:

```text
Is D-CHE/D-FOU forward/VJP close to MLP?
```

It is now:

```text
Can D-CHE/D-FOU run adaptive FU controller in the full training loop with MLP-like overhead?
```

The timing must include every component of adaptive guidance.

## 5.2 Carriers and variants

Primary:

```text
D-FOU R2 tablelookup-bandreadout native operator-step
D-FOU R3 lowfreq source-state fused commit
D-CHE R2 k5 gradbuf no-materialize native VJP
D-CHE R4 corrected-layout source-state fused commit
```

Secondary only if generic arbitrary-cotangent fused backward exists:

```text
D-RBF generic backward_from_grad_logits
D-RAT generic backward_from_grad_logits
```

If D-RBF/D-RAT still lack generic backward contract, they remain blocked and cannot borrow D-CHE/D-FOU success.

## 5.3 Timing grid

Run:

```text
batch_size: 128, 256, 512, 1024
hidden: 64, 128
cotangent/context:
  CE pointwise
  MSE pointwise
  Ranking pairwise
  Preference pairwise smoke
  SourceTarget diagnostic
  StableRandom control
  Gaussian stress
controller_mode:
  no_controller
  cheap_risk_monitor_only
  adaptive_readout_prox
  adaptive_basis_prox
  adaptive_coupled_basis_readout
```

## 5.4 Metrics

Record per row:

```text
carrier
variant
batch_size
hidden
cotangent_type
geometry_context_type
controller_mode
forward_ms
backward_ms
cotangent_ms
operator_T_ms
risk_monitor_ms
source_state_update_ms
prox_solve_ms
basis_solve_ms
readout_solve_ms
commit_ms
optimizer_state_update_ms
full_step_ms
full_loop_ratio_vs_mlp
controller_overhead_ratio
memory_peak_mb
basis_activation_bytes
source_state_bytes
JVP_count
VJP_count
official_fused_kernel_complete
manual_upstream_vjp_used
grad_rel_error
native_vs_autograd_gradcheck
controller_contract_pass
outlier_reason
```

## 5.5 Efficiency pass criteria

Exploration pass:

```text
official_fused_kernel_complete = 1
manual_upstream_vjp_used = 0
grad_rel_error <= 1e-5
full_loop_ratio_vs_mlp <= 1.50
memory_ratio_vs_mlp <= 1.15
controller_contract_pass = 1
```

Official adaptive efficiency pass:

```text
D-CHE or D-FOU has >=1 variant passing all non-smoke cotangent/context rows;
full_loop_ratio_vs_mlp <= 1.35;
controller_overhead_ratio <= 0.25;
memory_ratio_vs_mlp <= 1.10;
no missing batch 1024 row;
no carrier-level OR gate;
no manual VJP counted as official.
```

## 5.6 If not satisfied, Codex first attempts

If risk monitor dominates time:

```text
1. Cache H, basis features, and low-rank source projections.
2. Replace exact JVP with readout/basis first-order approximation.
3. Compute risk every mini-step but exact projection every K=10 steps.
4. Record approximation error; do not hide it.
```

If prox_solve dominates time:

```text
1. Lower rank_cap from 32 to 16 or 8.
2. Use warm-started CG with previous basis.
3. Fuse Gram computation with forward cache.
4. Fall back to readout-only for MLP lab, but not for KAN basis claim.
```

If D-CHE has outliers:

```text
1. Check corrected layout contract first.
2. Split forward/VJP/commit timing.
3. Try low-degree active bank only.
4. Do not average out Gaussian or pairwise outliers.
```

If D-FOU has outliers:

```text
1. Inspect tablelookup memory movement.
2. Check frequency buffer reuse.
3. Split lowfreq source-state commit from readout commit.
4. If Gaussian stress only fails, mark stress blocked but still require real adapter rows.
```

---

# 6. Part C：Functional update main experiments

## 6.1 Overall hypotheses

### H1: Reactive correction is too late

If the controller waits until measured source retention drops below a threshold, the representation has already shifted. Predictive risk should intervene earlier and improve h3200/h4800/h6400 source_loss.

Prediction:

```text
predictive adaptive prox > reactive threshold prox;
predictive adaptive prox requires fewer large corrections;
predictive adaptive prox has lower optimizer destructive projection before h1600;
source_loss does not flip negative as often.
```

### H2: Adaptive FU should be a feedback controller, not a fixed periodic replay

Fixed periodic replay is blind to whether the current step helps or hurts source. Adaptive controller should use risk and source usefulness to modulate strength.

Prediction:

```text
adaptive controller beats periodic_source_state at same or lower compute;
adaptive lambda_t correlates with future source drift;
controller interventions happen before large source decay.
```

### H3: Complex losses require geometry context $\mathcal{C}$, not only cotangent $\delta$

CE/MSE are mostly pointwise. Ranking/Preference depend on pairwise margins. Flattened $\delta$ alone can destroy pairwise antisymmetry.

Prediction:

```text
T(delta) fails or weakly passes Ranking/Preference;
T(delta, B) improves pairwise margin source_loss;
row-axis whitening/control-null that ignores B hurts Ranking/Preference.
```

### H4: Source retention must be source_loss-aware

A retained displacement is not enough. If source_func is positive but source_loss is negative, the controller must release or refresh the source.

Prediction:

```text
source_loss-aware release reduces target-retention-only rows;
source_func-only preservation increases stale source / negative source_loss events.
```

### H5: KAN value requires basis-native controller

Readout replay cannot prove KAN-specific value. D-CHE/D-FOU must carry adaptive source in basis state or low-bank state.

Prediction:

```text
basis-native controller increases basis_channel_energy_fraction;
KAN_source_loss_h3200/h4800 improves versus readout-only;
KAN_specific_delta_vs_MLP_same_operator becomes nonnegative;
controller-in-loop efficiency remains acceptable.
```

---

# 7. Experiment C1：MLP Adaptive Guidance Lab

## 7.1 Goal

Use MLP as the clean source dynamics laboratory. We first ask whether adaptive controller really improves training-flow source retention before moving to KAN.

## 7.2 Modes

Run the following modes:

```text
M0 AdamW baseline
M1 operator_only one_commit diagnostic
M2 fixed_periodic_source_state
M3 reactive_threshold_jacobian_prox
M4 predictive_adaptive_readout_prox
M5 predictive_adaptive_source_state_prox
M6 continuous_lowrank_guidance
M7 auxiliary_loss_anchor upper_bound diagnostic
M8 random_matched_controller control
M9 stable_random_controller control
M10 sign_flip_source control
```

Definitions:

```text
reactive_threshold_jacobian_prox:
  intervention only after measured source_retention < threshold.

predictive_adaptive_readout_prox:
  lambda_t based on predicted next-step source drift; uses readout low-rank solve.

predictive_adaptive_source_state_prox:
  same as above but z_t refreshes adaptively.

continuous_lowrank_guidance:
  lambda_t can be nonzero every step; exact solve amortized / low-rank approximate.
```

## 7.3 Adapters and seeds

Adapters:

```text
Delta-LossCEAdapter
Delta-MSEAdapter
Delta-RankingAdapter
Delta-PreferenceAdapter-smoke
Delta-StableRandom-control
Delta-RandomMatched-control
```

Seeds:

```text
2215, 2216, 2217
```

Horizon:

```text
h100, h400, h800, h1600, h2400, h3200, h4000, h4800, h6400
```

## 7.4 Metrics

Record every horizon and every 25 steps for internal dynamics:

```text
source_func_h*
source_loss_h*
source_retention_h*
R4800_over_3200_func
R6400_over_4800_func
source_loss_flip_count
source_stale_release_count
source_refresh_count
mean_source_age
source_state_alignment
source_state_decay_rate
predicted_source_drift
actual_next_source_drift
washout_risk
controller_lambda_t_mean
controller_lambda_t_p95
intervention_count
intervention_lead_time
prox_residual_before
prox_residual_after
optimizer_destructive_projection
control_projection_fraction
NDS
metric_energy_Fisher
metric_energy_Sobolev
row_angular_velocity
radial_fraction
tangential_fraction
AUC_loss_step
AUC_loss_time
train_loss_final
calibration_debt_readback
random_control_pass
stable_random_control_pass
```

## 7.5 Pass criteria

C1 exploration pass:

```text
CE and MSE:
  predictive_adaptive modes pass C3 on >=2/3 seeds;
  source_func_h3200 > 0;
  source_loss_h3200 >= 0;
  controls fail.
```

C1 official MLP adaptive pass:

```text
CE and MSE:
  predictive_adaptive modes pass C4 on >=2/3 seeds;
  source_func_h4800 > 0;
  source_loss_h4800 >= 0;
  R4800_over_3200_func >= 0.50;
  C4 pass rate > reactive_threshold_jacobian_prox;
  C4 pass rate > fixed_periodic_source_state;
  random/stable/sign controls fail;
  AUC_loss_time not worse than AdamW by >5%.
```

Additional requirement:

```text
adaptive controller must show positive intervention_lead_time:
  median intervention must occur before measured source_retention crosses failure threshold.
```

## 7.6 If not satisfied, Codex first attempts

If predictive adaptive does not beat reactive:

```text
1. Inspect risk AUC for source drop within 50/100/200 steps.
2. If risk predictor is bad, move to C2 risk-model repair.
3. If risk predictor is good but controller does not help, lower prox_scale and rank_cap; check overcorrection.
4. If intervention happens too late, raise risk threshold or add source derivative term.
```

If source_func positive but source_loss negative:

```text
1. Enable source_loss-aware release.
2. Penalize preserving stale source when linearized source_loss < 0.
3. Do not increase retention strength.
4. Mark row target-retention-only if repeated.
```

If controls pass:

```text
1. Tighten control projection penalty.
2. Check whether controller simply preserves arbitrary source.
3. Add sign_flip and corrupt-source controls.
4. Disable claim until controls fail.
```

---

# 8. Experiment C2：Predictive source-risk model

## 8.1 Goal

Train-only risk prediction is the core difference between adaptive guidance and late repair. C2 asks whether we can predict source washout before it happens.

## 8.2 Label definition

For risk model calibration, labels can be computed from prior training-run artifacts, but within a new official run the controller can only use current and past train-stream information.

Define future washout label for analysis:

$$
y_t^{H}=1
\quad \text{if} \quad
S_{t+H}<\tau_S
\text{ or }
source\_loss_{t+H}<\tau_L.
$$

Use horizons:

```text
H = 50, 100, 200
```

This label is only for fitting/evaluating the risk predictor across historical/train-only runs, not for direction in the same run.

## 8.3 Features

Record candidate features:

```text
S_t current source retention
Delta_S_t first derivative
Delta2_S_t second derivative
predicted_Ju_base_source_projection
optimizer_destructive_projection
control_projection_fraction
NDS_current
metric_energy_current
source_loss_linear_gain
source_loss_boundary_distance
row_angular_velocity
radial_fraction
source_age
source_refresh_rate
pairwise_margin_velocity
pairwise_antisymmetry_error
basis_channel_energy_fraction
basis_to_readout_leakage
```

Forbidden features:

```text
validation/test metrics;
future horizon from same run as runtime signal;
dataset name;
seed;
LineC/ECE/Brier/AUCtime as direction.
```

## 8.4 Metrics

```text
AUC_predict_washout_H50
AUC_predict_washout_H100
AUC_predict_washout_H200
precision_at_top20pct_risk
recall_at_FPR30
median_lead_time_steps
false_positive_intervention_rate
missed_washout_rate
feature_ablation_importance
cross_seed_AUC
cross_adapter_AUC
```

## 8.5 Pass criteria

Exploration:

```text
AUC_predict_washout_H100 >= 0.70
recall_at_FPR30 >= 0.65
median_lead_time_steps >= 50
```

Official adaptive-risk pass:

```text
AUC_predict_washout_H100 >= 0.75
AUC_predict_washout_H200 >= 0.70
recall_at_FPR30 >= 0.80
median_lead_time_steps >= 100
cross_seed_AUC >= 0.65
cross_adapter_AUC >= 0.60
```

## 8.6 If not satisfied, Codex first attempts

If risk model fails globally:

```text
1. Fall back to model-free risk law using predicted_Ju_base_source_projection and source_loss boundary.
2. Remove noisy features and test derivative-only controller.
3. Increase logging frequency to every 10 steps for a short diagnostic run.
4. Do not return to fixed periodic replay as mainline.
```

If risk model works only for CE/MSE:

```text
1. Add pairwise margin features for Ranking/Preference.
2. Separate pointwise and pairwise risk heads.
3. Do not claim arbitrary-loss adaptive guidance.
```

---

# 9. Experiment C3：Loss-geometry-aware operator

## 9.1 Goal

CE/MSE success does not prove complex-loss success. Ranking/Preference need pairwise geometry. C3 upgrades the operator from:

$$
T_\theta(\delta)
$$

to:

$$
\boxed{T_\theta(\delta,\mathcal{C})}.
$$

## 9.2 Geometry contexts

Pointwise losses:

```text
C = identity / per-sample weights / mask.
```

Ranking losses:

```text
C = B, where (Bf)_{ij}=f(x_i)-f(x_j).
```

Preference losses:

```text
C = preference pair graph with chosen/rejected pairs.
```

Composite losses:

```text
C = block context combining pointwise and pairwise incidence.
```

The core operator may read $\mathcal{C}$, but not adapter name. If the same $\delta$ and same $\mathcal{C}$ are passed under a renamed adapter, output must be identical.

## 9.3 Operator objective

Pointwise:

$$
T_{point}(\delta,z)
=
\arg\min_{\Delta f}
\left[
\langle \delta,\Delta f\rangle
+
\lambda\|\Delta f\|_{G_f}^2
+
\mu NDS(\Delta f)
+
\nu\|P_{control}\Delta f\|^2
+
\rho\|\Delta f-z\|^2
\right].
$$

Pairwise:

$$
T_{pair}(\delta,B,z)
=
\arg\min_{\Delta f}
\left[
\langle \delta,\Delta f\rangle
+
\lambda\|\Delta f\|_{G_f}^2
+
\mu\|B\Delta f\|_{G_{pair}}^2
+
\nu\|P_{control}^{pair}B\Delta f\|^2
+
\rho\|B\Delta f-Bz\|^2
\right].
$$

The pairwise term is not auxiliary loss. It is part of the train-only operator geometry used to construct the update direction.

## 9.4 Metrics

```text
operator_context_type
adapter_renaming_pass_with_same_delta_C
same_delta_different_C_output_change_allowed
pairwise_margin_gain_h*
pairwise_order_accuracy_gain_h*
pairwise_antisymmetry_error
pairwise_delta_projection_residual
row_collapse_score
source_func_h*
source_loss_h*
ranking_source_loss_h*
preference_source_loss_h*
control_projection_pointwise
control_projection_pairwise
NDS_pointwise
NDS_pairwise
```

## 9.5 Pass criteria

Pointwise pass:

```text
CE and MSE C4 on >=2/3 seeds using adaptive controller;
controls fail;
no adapter-name branch.
```

Pairwise exploration pass:

```text
Ranking C3 on >=2/3 seeds;
pairwise_margin_gain_h3200 > 0;
source_loss_h3200 >= 0;
pairwise_antisymmetry_error <= 0.05;
controls fail.
```

Pairwise official pass:

```text
Ranking C4 on >=2/3 seeds;
Preference smoke C3 on >=2/3 seeds;
source_loss_h4800 >= 0;
pairwise_margin_gain_h4800 > 0;
random/stable/sign controls fail.
```

## 9.6 If not satisfied, Codex first attempts

If Ranking source_func positive but source_loss negative:

```text
1. Check pairwise_antisymmetry_error.
2. Disable row-mean whitening in pairwise path.
3. Lower control-null strength in B-space.
4. Add source_loss_boundary release for pairwise source.
```

If Ranking source_func negative:

```text
1. Verify B incidence matrix and sign convention.
2. Test same δ with identity C vs pair C.
3. Check whether operator collapses row differences.
4. Add pairwise projection residual to objective.
```

If Preference fails but Ranking passes:

```text
1. Separate preference graph weights from ranking incidence.
2. Add chosen/rejected margin normalization.
3. Keep Preference as smoke until stable; do not block Ranking official unless plan requires preference.
```

---

# 10. Experiment C4：Adaptive source-state and release dynamics

## 10.1 Goal

Prevent stale source preservation. Adaptive guidance must know when to preserve, when to refresh, and when to release.

## 10.2 Modes

```text
S0 fixed_source_anchor
S1 EMA_source_state
S2 gain_coherence_gated_source_state
S3 source_loss_boundary_release
S4 dual_memory_fast_slow_source
S5 transport_aware_source_state
S6 random_source_state_control
```

## 10.3 Source-state rules

Fast/slow source:

$$
z_t^{fast}=\beta_f z_{t-1}^{fast}+(1-\beta_f)s_t,
\qquad
z_t^{slow}=\beta_s z_{t-1}^{slow}+(1-\beta_s)s_t.
$$

Guidance source:

$$
z_t = a_tz_t^{fast}+(1-a_t)z_t^{slow}.
$$

The mixing $a_t$ depends on risk and source usefulness:

$$
a_t = \operatorname{clip}(a_0+c_1 r_t-c_2 source\_age-c_3\mathbf{1}[source\_loss<0],0,1).
$$

## 10.4 Metrics

```text
source_age_mean
source_age_p95
fast_slow_alignment
source_refresh_count
source_release_count
stale_source_count
source_loss_boundary_events
source_loss_recovered_after_release
source_retention_after_refresh
source_func_h*
source_loss_h*
AUC_loss_time
controller_lambda_mean
controller_lambda_p95
```

## 10.5 Pass criteria

```text
source_loss_boundary_release reduces negative source_loss_h4800 rows by >=30% versus fixed_source_anchor;
source_func_h4800 not reduced by >20%;
C4 pass rate improves over EMA-only;
random_source_state_control fails;
AUC_loss_time not worse than AdamW by >5%.
```

## 10.6 If not satisfied, Codex first attempts

If release destroys useful source:

```text
1. Make release soft: reduce lambda_t instead of resetting z.
2. Use slow state as fallback.
3. Raise source_loss negative threshold from 0 to -1e-4.
```

If stale source remains:

```text
1. Increase source refresh frequency.
2. Add source_age penalty.
3. Add coherence drop trigger.
```

---

# 11. Experiment C5：KAN basis-native adaptive controller

## 11.1 Goal

KAN must stop relying on readout replay. The goal is to define and test adaptive FU directly in D-CHE / D-FOU basis state.

## 11.2 Basis-native update

Let $b$ denote basis parameters / low-degree or low-frequency state. The basis controller solves:

$$
\Delta b_t^*
=
\arg\min_{\Delta b}
\left[
\langle \delta_t,J^B_t\Delta b\rangle
+
\lambda_t\|J^B_t\Delta b-z_t\|_{G_f}^2
+
\mu_tNDS(J^B_t\Delta b)
+
\nu_t\|P_{control}J^B_t\Delta b\|^2
+
\rho_t\|\Delta b\|_{G_B}^2
\right].
$$

For pairwise losses:

$$
\Delta b_t^*
=
\arg\min_{\Delta b}
\left[
\langle \delta_t,J^B_t\Delta b\rangle
+
\lambda_t\|B_tJ^B_t\Delta b-B_tz_t\|_{G_{pair}}^2
+
\rho_t\|\Delta b\|_{G_B}^2
\right].
$$

## 11.3 KAN modes

```text
K0 KAN+AdamW baseline
K1 readout adaptive controller diagnostic
K2 basis-only adaptive controller
K3 coupled basis+readout controller with leakage penalty
K4 low-degree D-CHE controller
K5 low-frequency D-FOU controller
K6 basis-native pairwise controller
K7 random basis-source control
K8 readout-to-basis transfer diagnostic only
```

## 11.4 Metrics

```text
carrier
variant
controller_mode
basis_channel_energy_fraction
readout_channel_energy_fraction
basis_projection_cosine
basis_projection_residual
basis_operator_residual
basis_condition_number
basis_update_norm
basis_to_readout_leakage
readout_to_basis_transfer_success
KAN_source_func_h*
KAN_source_loss_h*
KAN_specific_delta_vs_MLP_same_operator
KAN_vs_readout_only_delta
source_state_decay_rate_basis
controller_full_loop_ratio_vs_mlp
basis_controller_step_ms
basis_controller_memory_mb
controls_pass_count
```

## 11.5 Pass criteria

KAN exploration pass:

```text
basis_channel_energy_fraction >= 0.30;
basis_projection_residual <= 0.70;
KAN_source_func_h3200 > 0;
KAN_source_loss_h3200 >= 0;
controls fail;
carrier adaptive efficiency exploration pass = 1.
```

KAN official basis-carrier pass:

```text
basis_channel_energy_fraction >= 0.50;
KAN_source_func_h4800 > 0;
KAN_source_loss_h4800 >= 0;
KAN_specific_delta_vs_MLP_same_operator >= 0;
KAN_vs_readout_only_delta >= 0;
controller_full_loop_ratio_vs_mlp <= 1.35;
>=2/3 seeds pass.
```

## 11.6 If not satisfied, Codex first attempts

If basis coverage is still low:

```text
1. Do not tune gain first.
2. Compute basis Gram by degree/frequency bank.
3. Whiten basis features within low-degree/low-frequency bank.
4. Increase active low-bank width only after coverage audit.
5. If basis_projection_cosine < 0.20 after whitening, mark current strict basis tangent unsuitable for this target.
```

If basis source_func positive but source_loss negative:

```text
1. Add source_loss boundary term to basis objective.
2. Release stale basis source.
3. Compare readout-only diagnostic to basis-only; do not claim KAN value.
```

If D-CHE fails but D-FOU passes:

```text
1. Promote D-FOU as primary basis carrier only.
2. Keep D-CHE as secondary efficiency carrier.
3. Do not write KAN-general success.
```

---

# 12. Part D：Task-level proof after mechanism gates

## 12.1 When to run task proof

Task proof only runs when:

```text
C1 MLP adaptive pointwise pass;
C3 pairwise exploration pass or explicit pointwise-only route;
C5 KAN basis exploration pass;
Part B adaptive efficiency pass;
controls fail.
```

If KAN basis gate fails, task rows may be readback only and cannot repair direction.

## 12.2 Task matrix

Datasets:

```text
MNIST
FashionMNIST
KMNIST
```

Seeds:

```text
0, 1, 2
```

Variants:

```text
MLP+AdamW
MLP+SGD
MLP+AdaptiveFU
KAN+AdamW
KAN+AdaptiveFU-readout-diagnostic
KAN+AdaptiveFU-basis-official
KAN+AdaptiveFU-coupled-basis-readout
```

## 12.3 Metrics

```text
final_train_loss
final_test_loss_readback
final_train_accuracy
final_test_accuracy_readback
NLL_delta_vs_MLP
ECE_delta_vs_MLP
Brier_delta_vs_MLP
AUC_loss_step
AUC_loss_time
AUC_loss_time_ratio_vs_best_control
time_to_train_loss_threshold
time_to_accuracy_threshold
forgetting_after_shift
source_func_task_h*
source_loss_task_h*
source_decay_rate_task
Jacobian_spectrum
feature_effective_rank
margin_distribution
calibration_debt
tail_loss_q95/q99
full_loop_step_ms
memory_ratio_vs_mlp
```

## 12.4 Pass criteria

Functional task value pass:

```text
KAN+AdaptiveFU beats KAN+AdamW on >=8/9 rows in AUC_loss_time or final train loss;
no calibration/tail debt explosion;
source retention mechanism pass remains valid.
```

MLP superiority claim:

```text
KAN+AdaptiveFU beats or matches same-param MLP on >=7/9 rows by test accuracy/NLL;
AUC_loss_time_ratio_vs_best_control <= 1.0 on >=7/9 rows;
full_loop_step_ratio_vs_mlp <= 1.35;
mechanism gates pass.
```

If task improves but mechanism gates fail:

```text
route = TaskReadbackImproved_MechanismNotProven
```

---

# 13. Controls and forbidden information firewall

Every official row must include:

```text
NoOpMatchedOverhead
RandomMatchedNorm
StableRandom
SignFlipSource
CorruptSource
SameControllerRandomSource
AdamWParallelDirection
SGDParallelDirection
AuxiliaryLossUpperBoundDiagnostic
```

The controls must have their own source anchors/states when applicable. They cannot inherit FU source.

Forbidden:

```text
validation/test/future/query direction;
dataset-name branch;
seed-specific rule;
adapter-name branch in FU core;
loss formula branch in FU core;
LineC/ECE/Brier/AUCtime/tail as direction;
auxiliary loss as strict official;
readout-only KAN row as basis-carrier success;
manual VJP row as official fused kernel row.
```

Allowed:

```text
current train batch logits;
current train batch labels inside LossInterface only;
current output cotangent δ;
train-only geometry context C;
current hidden/readout/basis activations;
past train-stream source state;
optimizer state;
train-only risk features;
historical training-run risk labels for offline predictor calibration, not same-run future for direction.
```

---

# 14. Required artifacts

## 14.1 Code / truth artifacts

```text
v22_15_code_review_packet.zip
v22_15_code_review_packet_manifest.csv
v22_15_required_source_files.csv
v22_15_clean_unzip_compileall.log
v22_15_clean_unzip_import_closure.log
v22_15_semantic_firewall.csv
v22_15_adapter_renaming_tests.csv
v22_15_loss_geometry_context_tests.csv
v22_15_adaptive_controller_unit_tests.csv
v22_15_finalizer_semantics_tests.csv
```

## 14.2 Efficiency artifacts

```text
v22_15_adaptive_efficiency_matrix.csv
v22_15_DCHE_adaptive_officialization.csv
v22_15_DFOU_adaptive_officialization.csv
v22_15_controller_component_timing.csv
v22_15_full_loop_ratio_matrix.csv
v22_15_native_gradcheck.csv
v22_15_manual_vs_native_vjp_comparison.csv
```

## 14.3 Functional artifacts

```text
v22_15_mlp_adaptive_guidance_matrix.csv
v22_15_source_risk_prediction_matrix.csv
v22_15_loss_geometry_operator_matrix.csv
v22_15_source_state_release_matrix.csv
v22_15_controller_intervention_log.csv
v22_15_source_dynamics_timeseries.csv
v22_15_control_attribution_matrix.csv
v22_15_source_func_loss_horizon_matrix.csv
```

## 14.4 KAN artifacts

```text
v22_15_KAN_basis_controller_matrix.csv
v22_15_basis_projection_coverage.csv
v22_15_basis_vs_readout_adaptive_ablation.csv
v22_15_KAN_vs_MLP_same_operator.csv
v22_15_basis_source_state_dynamics.csv
v22_15_basis_controller_efficiency.csv
```

## 14.5 Task artifacts

```text
v22_15_task_eval_matrix.csv
v22_15_convergence_speed_matrix.csv
v22_15_forgetting_readback_matrix.csv
v22_15_expression_metrics_matrix.csv
v22_15_calibration_debt_matrix.csv
```

## 14.6 GPU / execution artifacts

```text
v22_15_runnable_queue.csv
v22_15_gpu_assignment_manifest.csv
v22_15_gpu_utilization_timeline.csv
v22_15_idle_violation.csv
v22_15_queue_drain_report.json
v22_15_command_journal.csv
v22_15_deferred_items.csv
```

---

# 15. 4GPU dynamic execution plan

GPU0:

```text
Part A S0 code truth;
D-CHE adaptive efficiency;
D-CHE basis controller.
```

GPU1:

```text
MLP adaptive guidance lab C1;
source risk prediction C2.
```

GPU2:

```text
D-FOU adaptive efficiency;
D-FOU basis controller;
KAN vs MLP same operator.
```

GPU3:

```text
loss-geometry operator C3;
source-state release C4;
controls;
task readback only after gates pass;
finalizer/figures.
```

Dynamic refill:

```text
If GPU0 finishes early:
  take D-FOU efficiency outlier or basis coverage jobs.

If GPU1 finishes early:
  run additional seeds for C1/C2.

If GPU2 finishes early:
  run KAN controls and coupled basis-readout ablation.

If GPU3 finishes early:
  run Ranking/Preference pairwise ablations and controller controls.
```

Execution contract:

```text
if runnable_queue_nonempty and any_gpu_idle_minutes > 10:
    execution_contract_violation = 1
    final route cannot be completed promotion
```

---

# 16. Final route taxonomy

```text
R0-CodeOrSemanticGateFailed
R1-AdaptiveEfficiencyBlocked
R2-RiskPredictionNoGo
R3-AdaptiveControllerNoGo
R4-PointwiseAdaptiveFUOpened_PairwiseNoGo
R5-SourceFuncOnly_SourceLossNoGo
R6-AuxiliaryUpperBoundOnly_StrictNoLossNoGo
R7-MLPAdaptiveFUOpened_KANBasisNoGo
R8-KANReadoutAdaptiveOpened_BasisStillBlocked
R9-KANBasisAdaptiveExplorationOpened_TaskNotProven
R10-AdaptiveFUMechanismPass_TaskReadbackPending
R11-TaskImproved_MechanismNotProven
R12-OfficialAdaptiveFU_KANCarrierReady
R13-OfficialDGKANBeatsMLPReady
```

Official adaptive FU mechanism requires:

```text
S0 pass;
adaptive efficiency pass on D-CHE or D-FOU;
CE/MSE C4 on >=2/3 seeds;
Ranking C3 or explicit pointwise-only route;
risk controller intervention lead time positive;
source_loss_h4800 >= 0;
random/stable/sign controls fail;
uses_loss_modification_for_retention = 0;
no adapter-name branch;
no validation/test/future/query direction.
```

Official KAN carrier requires:

```text
basis_channel_energy_fraction >= 0.50;
KAN_source_loss_h4800 >= 0;
KAN_specific_delta_vs_MLP_same_operator >= 0;
controller full-loop ratio <= 1.35;
>=2/3 seeds pass;
controls fail.
```

Official DG-KAN > MLP requires:

```text
mechanism pass;
KAN carrier pass;
KAN+AdaptiveFU beats or matches same-param MLP on >=7/9 task rows;
AUC_loss_time improves;
calibration/tail debt not exploded;
step/memory within gate.
```

---

# 17. What Codex should do first if results are bad

## 17.1 If adaptive controller fails completely

Do not return to fixed periodic replay. First:

```text
1. Check whether risk predictor has any lead-time signal.
2. If yes, repair controller strength / overcorrection.
3. If no, use model-free destructive projection law.
4. Reduce exact solve frequency but keep every-step cheap guard.
5. Output which layer failed: risk, source operator, prox solve, source_loss boundary, or controls.
```

## 17.2 If CE/MSE pass but Ranking/Preference fail

Do not claim arbitrary-loss. First:

```text
1. Enable pairwise context B.
2. Disable row-axis whitening in pairwise path.
3. Record pairwise antisymmetry error.
4. Add pairwise source_loss boundary.
5. Treat Preference as smoke if Ranking becomes stable first.
```

## 17.3 If KAN basis fails

Do not promote readout rows. First:

```text
1. Run basis coverage audit by degree/frequency bank.
2. Whiten basis Gram.
3. Try low-degree/low-frequency active-bank expansion.
4. Try coupled basis+readout with leakage penalty.
5. If basis_projection_cosine still <0.20, mark current basis tangent unsuitable and report architecture-level blocker.
```

## 17.4 If efficiency fails

Do not average away outliers. First:

```text
1. Split timing into operator_T, risk, prox, commit.
2. Replace exact risk with cached approximation.
3. Reduce rank_cap.
4. Fuse basis/readout Gram computation.
5. Keep D-RBF/D-RAT blocked unless generic backward contract exists.
```

---

# 18. Final expected interpretation

v22.15 should answer one central scientific question:

$$
\boxed{
\text{Can functional update become an adaptive controller of training dynamics,}
\text{rather than a source patch applied before or after ordinary BP?}
}
$$

A strong positive result would mean:

```text
1. The model predicts destructive gradient flow before source is erased.
2. The controller changes the current step without modifying task loss.
3. Source_func and source_loss stay positive to h4800/h6400.
4. Controls fail.
5. D-CHE/D-FOU can run the controller efficiently.
6. KAN basis, not just readout, carries the source.
```

A useful negative result would still be precise:

```text
risk prediction exists but controller cannot use it;
risk prediction does not exist with current observables;
pointwise losses work but pairwise geometry fails;
source can be controlled in MLP but not KAN basis;
KAN basis is not expressive enough in current strict FC-PureKAN tangent;
efficiency works for ordinary cotangent but not controller-in-loop.
```

The project should not measure success by inventing another Fxx label. It should measure success by whether the training trajectory itself becomes guided: ordinary BP still supplies task gradients, but adaptive FU controls which parts of those gradients are allowed to become long-lived function-space changes.

