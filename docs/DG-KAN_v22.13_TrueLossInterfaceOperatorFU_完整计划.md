# DG-KAN v22.13：True Loss-Interface Operator FU + Kernel-Native Cotangent Efficiency + Corrected KAN Source Carrier 完整计划

> 版本：v22.13 execution plan  
> 生成日期：2026-06-08  
> 目标读者：Codex / 实验执行者 / 后续论文复盘读者  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 核心主题：从 constructive single-displacement source 升级为 true loss-interface operator。  
> 硬约束：strict FC-PureKAN；no active B-spline official path；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific rule；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / ECE / Brier / AUCtime / tail metrics 只能作为 audit / gate / debt readback，不能生成 direction；functional update 与 efficiency runner 不得针对 CE 设计，CE 只能作为 `LossInterface` 的一个 adapter。

---

# 0. 一句话目标

v22.13 的目标不是“把 v22.12 的 MSE C4 继续调强”，也不是“把 D-CHE K14 的 h4800 source_loss 微修成正”。这版要正式验证一个更高层命题：

$$
\boxed{
\text{Functional Update 的算法突破不是构造一个固定 } \Delta f，
\text{而是定义一个把任意 loss cotangent } \delta
\text{ 转成 retained、task-useful、low-curvature } \Delta f
\text{ 的 operator } T_\theta。
}
$$

也就是：

$$
\boxed{
\Delta f = T_\theta(\delta),
\qquad
\delta = \frac{\partial \mathcal{L}}{\partial f_\theta(x)}
}
$$

v22.13 必须同时推进两条路线：

```text
Efficiency 路线：
  证明 D-FOU / D-CHE 能在 kernel-native arbitrary-cotangent / operator-step 路径中接近 MLP。

Functional Update 路线：
  证明同一个 role-blind loss-interface operator 能跨多个 loss adapters 产生长期 retained、task-useful source。
```

最终科学 claim 不是：

```text
某个 MSE adapter 下 source_func 很大。
```

而是：

$$
\boxed{
\text{同一个 } T_\theta \text{ 在不同 loss cotangent 下都改善训练动力学：}
\text{表达能力更好、遗忘更少、收敛更快、效率不牺牲。}
}
$$

---

# 1. v22.12 的真实结论与 v22.13 的继承关系

## 1.1 v22.12 的真实进展

v22.12 的 route 是：

```text
R7-KANSourceExistsEfficiencyBlocked
exploration_promotion_allowed = 1
official_promotion_allowed = 0
```

关键事实：

```text
S0.19 code / semantic truth gate pass = 1
D-CHE / D-FOU / D-RAT / D-RBF S1 pass = 0 / 1 / 0 / 0
official_fused_kernel_complete_rows = 0
S2/S3/S4 pass rows = 2 / 3 / 3
S5 C3 adapter count = 2: Delta-MSEAdapter, Delta-RankingAdapter
S5 C4 adapter count = 1: Delta-MSEAdapter
KAN route = S6-DFOUSourceOpened_DCHESourceExistsEfficiencyBlocked
KAN pass rows = 2
```

这说明 v22.12 有真实推进，但不能 promotion：

```text
1. arbitrary-loss interface 已经进入 FU pipeline；
2. source-side constructive path 不是空的；
3. MSE adapter 下 source_func/source_loss 能到 C4；
4. Ranking adapter 能到 C3，但 h4800 source_loss 仍失败；
5. CE / GenericSourceTarget / StableRandom 暴露了 source_func 与 source_loss 分裂；
6. D-FOU K13 打开 KAN retained source；
7. D-CHE K14 corrected layout 打开 source-exists，但 D-CHE efficiency 与 h4800 source_loss 阻塞；
8. 所有 efficiency 成果仍是 manual upstream-VJP / official fused rows = 0。
```

因此 v22.12 最大改变不是某个具体 row，而是问题定义变了：

$$
\boxed{
\text{single-displacement constructive source}
\quad \longrightarrow \quad
\text{loss-interface operator } T_\theta(\delta)
}
$$

## 1.2 v22.13 必须继承的 insight

v22.13 必须继承四点：

```text
1. Constructive FU 比 observer/filter 路线更有前途。
2. 但 constructive FU 不能停留在 fixed Δf，必须变成 δ -> Δf 的 operator。
3. metric 不是没用，而是必须进入 operator definition，不能只是事后 filter/readback。
4. KAN 的价值不是“也能做 FU”，而是可能用低阶/低频 basis carrier 更高效、更稳定地实现 Tθ。
```

v22.13 因此不再把主线叫：

```text
h4800 repair
terminal source-loss fix
D-CHE K14 closure
```

而叫：

```text
True Loss-Interface Operator FU
+ Kernel-Native Cotangent Efficiency
+ Corrected KAN Source Carrier
```

---

# 2. v22.13 的总假设

v22.13 围绕五个核心假设设计。

## H1：v22.12 的 O10 不是 true arbitrary-loss operator，而是 role-aware / adapter-suite consensus

v22.12 的 `O10_NonRandomSourceLossBalancedOperator` 有研究价值，但它不能直接写成 true arbitrary-loss success。原因是：它把 non-random adapters 作为一个已知 suite 构造 consensus，并且事实上利用了 adapter role 的结构。

v22.13 对 O10 的定位：

```text
O10 可以保留为 diagnostic scaffold；
O10 不允许计入 official role-blind operator success；
O10 的有效成分必须被抽象成 adapter-name-free 的 operator atom。
```

预测：

```text
如果 O10 的成功来自 role-aware consensus：
  它会继续在 MSE 强，在 Ranking 局部强，在 CE/source-target 上暴露 source_loss 或 gauge 问题。

如果能抽象成 role-blind operator：
  去掉 adapter-name branch 后，至少两个非随机 adapters 仍能 C3，且至少两个非随机 adapters 能 C4。
```

## H2：真正的 FU 应是 cotangent preconditioner，不是 target generator

旧路线像：

$$
\Delta f = \Delta f_{target}
$$

v22.13 要验证的是：

$$
\Delta f = T_\theta(\delta)
$$

其中 $T_\theta$ 只能看到：

```text
current train-stream logits / activations
current upstream cotangent δ
current train micro-split statistics
current optimizer/source state
current KAN basis/readout state
```

不能看到：

```text
adapter name
loss formula branch
dataset name
seed
validation/test/future/query
LineC/ECE/Brier/AUCtime/tail as direction
```

更具体地，v22.13 要测试的 operator 形式是：

$$
T_\theta(\delta)
=
-P_{signal,\theta}
G_\theta^{-1}
P_{control-null,\theta}
P_{low-curvature,\theta}\delta
$$

其中：

```text
P_low-curvature:
  去除高 NDS / 高 Sobolev / 高 Fisher 曲率方向。

P_control-null:
  去除 AdamW / SGD / random / stable-random 可解释方向。

Gθ^{-1}:
  函数空间 metric preconditioner。

P_signal:
  投到 train micro-split coherent signal subspace，避免 reservoir noise。
```

## H3：operator law invariant，不等于输出方向 identical

一个重要修正：adapter robustness 不能理解为不同 adapters 的 $T_\theta(\delta_a)$ 必须完全同向。

CE、MSE、Ranking 的 cotangent 几何本来就不同。真正应该不变的是 operator law，而不是所有输出方向。

错误要求：

$$
\cos(T_\theta(\delta_{CE}),T_\theta(\delta_{MSE})) \approx 1
$$

正确要求：

```text
同一个 Tθ 无 adapter-name branch；
Tθ 的 linearity / homogeneity / stability 在不同 adapter 下成立；
Tθ 在每个 adapter 下都降低 NDS 或 control projection；
Tθ 在每个 adapter 下都不把 source_loss 推负；
Tθ 在 holdout adapter 上仍保持 task-useful retention。
```

v22.13 需要记录：

$$
\text{GainRatio}_a
=
\frac{-\langle \delta_a,T_\theta(\delta_a)\rangle}
{\|\delta_a\|^2+\epsilon}
$$

$$
\text{NDSReduction}_a
=
\frac{\text{NDS}(\delta_a)-\text{NDS}(T_\theta(\delta_a))}
{\text{NDS}(\delta_a)+\epsilon}
$$

$$
\text{ControlNullRatio}_a
=
1-
\frac{\|P_{control}T_\theta(\delta_a)\|}
{\|T_\theta(\delta_a)\|+\epsilon}
$$

## H4：metric 必须定义 operator，而不是筛选 update

observer/filter 路线是：

```text
先产生很多 update；
再用 metric / observer 挑一个。
```

v22.13 要做的是：

```text
metric 本身定义什么是健康的函数更新。
```

因此主 operator 写作：

$$
T_\theta(\delta)
=
\arg\min_{\Delta f}
\left[
\langle \delta,\Delta f\rangle
+
\lambda \|\Delta f\|_{G_\theta}^2
+
\mu \operatorname{NDS}(\Delta f)
+
\nu \operatorname{NoiseLeak}(\Delta f)
+
\omega \operatorname{ControlProj}(\Delta f)
+
\kappa \operatorname{AdapterLawDebt}(T_\theta)
\right]
$$

然后写回参数：

$$
u^*
=
\arg\min_u
\left[
\|J_\theta u - T_\theta(\delta)\|_{G_\theta}^2
+
\rho\|u\|_{P_\theta}^2
+
\eta E_{block}(u)
+
\zeta E_{source-state}(u)
\right]
$$

这里 $\nu^*$ 表示参数空间 commit direction。为了避免和 control coefficient $\nu$ 混淆，代码里可以命名为 `u_star`。

## H5：KAN 的价值是 operator carrier，不是 target replay carrier

KAN-specific claim 不应是：

```text
KAN 某个 readout replay source_func 很大。
```

而应该是：

$$
\boxed{
\text{D-CHE / D-FOU 的 low-degree / low-frequency carrier}
\text{是否能更便宜、更稳定地实现 } T_\theta(\delta)?
}
$$

v22.12 的 D-CHE K14 表明旧 D-CHE negative 可能被 readout layout bug 污染；但 K14 同时说明 D-CHE 仍卡在 efficiency 与 h4800 source_loss。因此 v22.13 必须：

```text
1. 用 corrected layout 重新验证 D-CHE source carrier；
2. 用 corrected layout 重新验证 D-FOU K13；
3. 不允许 D-CHE 借 D-FOU efficiency pass；
4. 不允许 readout target retention 写成 basis-channel success。
```

---

# 3. Part A：代码审计与语义真值门

## 3.1 目标

Part A 的目标不是走形式，而是防止 v22.13 再次把 role-aware / manual / layout artifact 写成 scientific progress。

必须回答：

```text
1. 源码包是否自包含？
2. 计划要求的文件是否都在？
3. official operator path 是否没有 adapter-name branch？
4. official operator path 是否没有 CE/MSE/Ranking loss formula branch？
5. O10-style diagnostic 是否被正确降级？
6. D-CHE/D-FOU readout/basis layout 是否被单元测试锁住？
7. S1 efficiency 是否区分 manual_upstream_vjp 与 official fused native path？
8. finalizer 是否区分 exploration promotion 与 official promotion？
```

## 3.2 必须包含的源码文件

Codex 必须输出：

```text
v22_13_code_review_packet.zip
```

至少包含：

```text
dgkan/loss_interface.py
dgkan/fu/operator_core.py
dgkan/fu/operator_atoms_v22_13.py
dgkan/fu/metric_geometry.py
dgkan/fu/control_nullspace.py
dgkan/fu/source_state.py
dgkan/fu/source_loss.py
dgkan/fu/operator_commit.py
dgkan/fu/operator_horizon.py
dgkan/kan/layout_contracts.py
dgkan/kan/corrected_readout_solve.py
dgkan/kernels/dfou_native_cotangent.py
dgkan/kernels/dche_native_cotangent.py
dgkan/profiling/efficiency_v22_13.py
experiments/run_v22_13_s0_truth_gate.py
experiments/run_v22_13_operator_semantic_tests.py
experiments/run_v22_13_operator_construct.py
experiments/run_v22_13_operator_horizon.py
experiments/run_v22_13_kan_corrected_mapping.py
experiments/run_v22_13_efficiency_native.py
experiments/run_v22_13_task_eval.py
experiments/run_v22_13_finalize.py
```

如果实际沿用旧文件，必须在：

```text
v22_13_compatibility_manifest.csv
```

写明：

```text
shim_file
target_file
target_exists
reason
risk
official_path_imports_this_file
```

## 3.3 自包含与 import closure

必须跑：

```bash
python -m compileall -q dgkan experiments
python experiments/run_v22_13_s0_truth_gate.py --check import_closure --source-root . --out-dir <official_dir>
```

记录：

```text
source_tree_complete
compileall_ok
core_import_pass
import_error_count
required_source_files_present
csv_claimed_exists_but_zip_missing_count
artifact_index_complete
```

硬标准：

```text
source_tree_complete = 1
compileall_ok = 1
core_import_pass = 1
import_error_count = 0
required_source_files_present = 1
csv_claimed_exists_but_zip_missing_count = 0
```

失败时 Codex 先做：

```text
1. 补齐缺失 source；
2. 若是 shim，补 target 或把 runner 改到真实文件；
3. 重跑 compileall/import；
4. 不允许进入 Part B/C；
5. final route = R0-CodeTruthFailed。
```

## 3.4 Role-blind operator semantic gate

### 3.4.1 静态 grep

检查 official FU core path 中是否出现：

```text
Delta-MSEAdapter
Delta-LossCEAdapter
Delta-RankingAdapter
GenericSourceTarget
StableRandom
cross_entropy
one_hot
softmax
ranking
mse_loss
adapter_name
loss_adapter_name
if adapter
if loss
```

允许：

```text
LossInterface adapter 自己内部出现 loss formula；
experiment matrix / CSV / finalizer 出现 adapter name；
diagnostic O10 path 出现 adapter-suite logic，但必须 official_path = 0。
```

不允许：

```text
official operator construction 根据 adapter name 选择公式；
official operator construction 根据 CE/MSE/Ranking 选择 branch；
official commit path 使用 loss formula special case。
```

记录：

```text
adapter_name_branch_count
official_core_loss_formula_branch_count
CE_specific_formula_in_official_direction_path
MSE_specific_formula_in_official_direction_path
Ranking_specific_formula_in_official_direction_path
O10_diagnostic_only_pass
official_operator_role_blind_static_pass
```

硬标准：

```text
adapter_name_branch_count = 0
official_core_loss_formula_branch_count = 0
CE_specific_formula_in_official_direction_path = 0
O10_diagnostic_only_pass = 1
official_operator_role_blind_static_pass = 1
```

### 3.4.2 Runtime provenance

每个 official FU row 必须落盘：

```text
operator_id
uses_adapter_name_for_direction
uses_loss_formula_for_direction
uses_labels_for_adapter_only
uses_labels_for_fu_core
uses_validation_test_future_query
uses_audit_metric_for_direction
runtime_provenance_hash
```

硬标准：

```text
uses_adapter_name_for_direction = 0
uses_loss_formula_for_direction = 0
uses_labels_for_fu_core = 0
uses_validation_test_future_query = 0
uses_audit_metric_for_direction = 0
```

### 3.4.3 Adapter renaming black-box test

构造两个等价 adapters：

```text
AdapterA: name = Delta-MSEAdapter
AdapterB: name = BananaAdapter_9231
```

它们输出相同 cotangent $\delta$。official operator 必须输出相同 $T_\theta(\delta)$。

记录：

```text
adapter_renaming_output_cosine
adapter_renaming_output_rel_error
adapter_renaming_commit_rel_error
adapter_renaming_pass
```

硬标准：

```text
adapter_renaming_output_cosine >= 0.999
adapter_renaming_output_rel_error <= 1e-6
adapter_renaming_commit_rel_error <= 1e-6
adapter_renaming_pass = 1
```

### 3.4.4 Operator law tests

对每个 official operator 测：

$$
T_\theta(\alpha \delta_1 + \beta \delta_2)
\approx
\alpha T_\theta(\delta_1) + \beta T_\theta(\delta_2)
$$

$$
T_\theta(c\delta)
\approx cT_\theta(\delta)
$$

记录：

```text
operator_linearity_error
operator_homogeneity_error
operator_lipschitz_ratio
cotangent_permutation_equivariance_error
cotangent_sign_flip_consistency
operator_law_pass
```

硬标准：

```text
operator_linearity_error <= 0.15 for linear operators
operator_homogeneity_error <= 0.10 for homogeneous operators
operator_lipschitz_ratio <= 5.0
cotangent_permutation_equivariance_error <= 1e-5
operator_law_pass = 1
```

说明：如果某个 operator 设计为非线性，例如 clipping / spectral threshold / PID source-loss control，则必须标记：

```text
operator_class = nonlinear_metric_operator
linearity_gate_required = 0
homogeneity_gate_required = 0
```

但仍必须通过：

```text
adapter renaming test
loss formula branch test
stability test
control tests
```

## 3.5 Corrected KAN layout truth gate

v22.12 的 D-CHE K14 发现 `frozen_readout_features` 的自然顺序是 `(hidden,basis)`，而旧 solve 直接 reshape 到 `w2(hidden,class,basis)`，会混淆 class/basis layout。v22.13 必须把这个问题锁成单元测试。

记录：

```text
carrier
w2_layout_contract
frozen_readout_feature_layout
legacy_layout_fit_cosine
legacy_layout_projection_residual
corrected_layout_fit_cosine
corrected_layout_projection_residual
layout_unit_test_pass
```

对 D-CHE 与 D-FOU 都要跑 immediate actuation test：

```text
D-CHE legacy layout
D-CHE corrected layout
D-FOU legacy layout
D-FOU corrected layout
```

硬标准：

```text
corrected_layout_fit_cosine >= 0.99
corrected_layout_projection_residual <= 0.10
legacy_vs_corrected_gap_detected = 1
layout_unit_test_pass = 1
```

如果 D-FOU legacy 与 corrected 差异也大：

```text
D-FOU K13 不允许沿用旧 pass；
必须重跑 K13_corrected。
```

## 3.6 Promotion semantics

finalizer 必须分开：

```text
exploration_promotion_allowed
official_promotion_allowed
scientific_claim_allowed
```

定义：

```text
exploration_promotion_allowed:
  可以进入下一版，但不能写 official success。

official_promotion_allowed:
  可以写成本轮 official operator/KAN/efficiency 成功。

scientific_claim_allowed:
  可以写成“FU 改善纯梯度反传的表达、遗忘、收敛、效率”。
```

硬标准：

```text
manual_upstream_vjp row 不能算 official fused success；
O10 diagnostic row 不能算 true role-blind operator success；
MSE-only C4 不能算 arbitrary-loss success；
source_func positive + source_loss negative 不能算 task-useful source；
D-CHE source-exists + D-CHE efficiency fail 不能算 D-CHE official carrier success；
D-FOU pass 不能借给 D-CHE；
readout-only pass 不能写成 full basis-channel pass。
```

---

# 4. Part B：True Loss-Interface Operator 理论与实现

## 4.1 LossInterface contract

每个 loss adapter 只负责产生 output cotangent：

$$
\delta_a = \frac{\partial \mathcal{L}_a}{\partial f_\theta(x)}
$$

LossInterface 必须输出：

```text
cotangent: tensor with same shape as model output
cotangent_norm
output_geometry_hint optional
adapter_internal_loss_value optional
labels_used_inside_adapter
```

LossInterface 不允许输出：

```text
special FU direction
special source target
adapter-specific metric weight
future/horizon readback
validation/test signal
```

官方 operator 的输入必须是：

```text
current model state θ
current train batch activations/logits
cotangent δ
train micro-split statistics
optimizer/source state
basis/readout state
```

## 4.2 Operator definition

v22.13 的 official operator 统一接口：

```python
Delta_f, telemetry = operator.apply(
    model_state=theta,
    train_batch=batch,
    activations=acts,
    logits=f_theta_x,
    cotangent=delta,
    optimizer_state=opt_state,
    source_state=source_state,
    basis_state=basis_state,
)
```

禁止接口：

```python
operator.apply(adapter_name="Delta-MSEAdapter", ...)
operator.apply(loss_type="ce", ...)
operator.apply(dataset="MNIST", ...)
operator.apply(seed=0, ...)
operator.apply(validation_metrics=..., ...)
```

## 4.3 Metric-as-geometry objective

v22.13 的主问题是：给定 $\delta$，构造一个既有 loss descent，又能长期留存的 $\Delta f$。

函数空间 objective：

$$
\mathcal{J}_{op}(\Delta f;\delta)
=
\langle \delta,\Delta f\rangle
+
\lambda \Delta f^\top G_\theta \Delta f
+
\mu \operatorname{NDS}(\Delta f)
+
\nu \|P_{control}\Delta f\|^2
+
\omega \operatorname{NoiseLeak}(\Delta f)
+
\xi \operatorname{CurvaturePenalty}(\Delta f)
$$

operator：

$$
T_\theta(\delta)
=
\arg\min_{\Delta f}\mathcal{J}_{op}(\Delta f;\delta)
$$

其中：

```text
Gθ:
  function-space metric，可由 Fisher / Sobolev / RKHS / KAN basis Gram 组合。

NDS:
  normalized directional sharpness，用于避免高曲率短期好、长期崩的方向。

P_control:
  AdamW / SGD / random / stable-random span，避免 FU 只是普通 optimizer/noise 的别名。

NoiseLeak:
  train micro-split 不一致或 corrupt-label 方向相似的成分。

CurvaturePenalty:
  row angular jitter / high-frequency / high-degree / reservoir leakage。
```

参数写回 objective：

$$
\mathcal{J}_{commit}(u;\Delta f)
=
\|J_\theta u - \Delta f\|_{G_\theta}^2
+
\rho\|u\|_{P_\theta}^2
+
\eta E_{block}(u)
+
\zeta E_{overwrite}(u, source\_state)
$$

$$
u^* = \arg\min_u \mathcal{J}_{commit}(u;T_\theta(\delta))
$$

代码中用：

```text
u_star
```

不要用希腊字母变量名。

## 4.4 Source state dynamics

Functional update 不是 one-step patch，而是训练动力学边界条件。v22.13 需要显式维护 source state：

$$
s_{t+1}
=
\beta s_t
+
(1-\beta)\Pi_{retain,\theta}T_\theta(\delta_t)
$$

commit 可以写成：

$$
\Delta f^{final}_t
=
T_\theta(\delta_t)
+
\alpha_s s_t
-
\alpha_c P_{destructive}(g_t,s_t)
$$

其中：

```text
s_t:
  retained source state。

Π_retain:
  low-NDS / control-null / split-coherent / KAN low-bank projection。

P_destructive:
  optimizer 后续可能擦掉 source 的 destructive projection。
```

记录：

```text
source_state_norm
source_state_age
source_state_decay_rate
source_state_overwrite_fraction
source_state_alignment_with_current_delta
source_state_alignment_with_controls
optimizer_destructive_projection
retain_projection_residual
```

---

# 5. Part C：Operator families

v22.13 不再无限尝试 Fxx 小修，而是明确测试少数 operator families。每个 family 都必须是 $\delta \mapsto \Delta f$，不是固定 displacement。

## 5.1 LIO-1：LowNDS Green Operator

核心假设：v22.12 中 O4 LowNDS 有通过迹象；metric 应作为 operator geometry，而不是事后 filter。

定义：

$$
T_{Green}(\delta)
=
-(G_\theta + \lambda H_{NDS}+\epsilon I)^{-1}\delta
$$

近似实现：

```text
1. 在 output / readout / low-bank feature space 构造 Gram。
2. 用 diagonal + low-rank approximation 近似 Gθ。
3. 用 micro-batch finite difference 估计 NDS curvature diagonal。
4. 输出 preconditioned cotangent direction。
```

记录：

```text
operator_id = LIO1_LowNDSGreen
metric_type
rank
cg_iterations
NDS_before
NDS_after
NDS_reduction
linearized_loss_gain
GainRatio_by_adapter
metric_energy_by_adapter
```

通过标准：

```text
NDS_reduction >= 0.30 on >=3 non-random adapters
GainRatio_a > 0 on >=3 non-random adapters
control_projection_fraction <= 0.25
StableRandom does not pass C3/C4
```

Codex 失败后先尝试：

```text
1. 从 full metric 降级到 diagonal Fisher/Sobolev metric；
2. 从 all-output 降级到 readout feature metric；
3. 降低 low-rank rank = 4/8/12；
4. 如果 NDS 下降但 source_loss 负，加 source_loss boundary term；
5. 如果仍失败，标记 LowNDSMetricNotSufficient，不继续 scale sweep。
```

## 5.2 LIO-2：Split-Coherent Control-Null Operator

核心假设：旧 split-consensus / control-nullspace 作为 filter 不稳定，但作为 operator projection 可能有价值。

构造：

```text
1. 把 train batch 切成 A/B/C micro-splits。
2. 对每个 split 得到 cotangent δ_A, δ_B, δ_C。
3. 构造 split-coherent subspace P_signal。
4. 构造 AdamW/SGD/random/stable-random span 的 P_control。
5. 输出：
```

$$
T_{SCN}(\delta)
=
-P_{signal}(I-P_{control})G^{-1}\delta
$$

记录：

```text
split_count
signal_subspace_rank
split_pair_cosine_mean
split_pair_cosine_min
control_span_rank
control_projection_before
control_projection_after
control_null_residual_norm
noise_into_signal_ratio
corrupt_projection_fraction
```

通过标准：

```text
split_pair_cosine_mean_after >= split_pair_cosine_mean_before
control_projection_after <= 0.25 * control_projection_before
corrupt_projection_fraction <= 0.50 * train_signal_projection
source_loss_h3200 >= -1e-6 on >=2 adapters
```

Codex 失败后先尝试：

```text
1. 降低 signal_subspace_rank；
2. 用 robust median consensus 替代 mean consensus；
3. 增加 random/stable-random control span；
4. 如果 Ranking 掉，检查 pairwise antisymmetry 是否被 split projection 破坏；
5. 如果 CE 掉，检查 class-gauge centering 是否造成 signal loss。
```

## 5.3 LIO-3：KAN Low-Bank Spectral Operator

核心假设：KAN 的优势来自低阶/低频 carrier。v22.13 要把 KAN 从 target replay carrier 变成 operator carrier。

定义：

$$
T_{KAN}(\delta)
=
-P_{lowbank}G_{basis}^{-1}P_{control-null}\delta
$$

其中：

```text
D-CHE:
  low-degree Chebyshev bank。

D-FOU:
  low-frequency Fourier bank。

D-RBF:
  local compact support bank，只作为 secondary。

D-RAT:
  numerator/denominator tangent block，只作为 diagnostic。
```

记录：

```text
carrier
basis_bank
basis_Gram_condition
spectral_rank
low_degree_energy_fraction
low_frequency_energy_fraction
high_degree_reservoir_leakage
high_frequency_reservoir_leakage
basis_channel_energy
readout_channel_energy
basis_commit_projection_residual
basis_estimate_fit_cosine
KAN_specific_delta_vs_MLP_same_metric
```

通过标准：

```text
basis_estimate_fit_cosine >= 0.99 for corrected layout diagnostic
basis_commit_projection_residual <= 0.10 for corrected layout diagnostic
KAN_source_func_h3200 >= 0.005
KAN_source_loss_h3200 >= -1e-6
KAN_source_loss_h4800 >= -1e-6 for official
KAN_specific_delta_vs_MLP_same_metric >= 0.005 or same source with better efficiency
carrier_specific_efficiency_pass = 1
```

Codex 失败后先尝试：

```text
1. 如果 D-CHE source exists but efficiency blocked：优先做 D-CHE kernel-native officialization；
2. 如果 D-CHE h4800 source_loss 负：加 slow source-state retention，而不是增大 replay scale；
3. 如果 D-FOU K13 corrected 后下降：定位旧 K13 是否 layout artifact；
4. 如果 readout-only pass 但 basis fail：写 KANReadoutSourceOpened_BasisChannelBlocked，不强行 full-basis claim。
```

## 5.4 LIO-4：Row-Angular / Matrix-Block Operator

核心假设：v22.12 引入 row-wise angular stability 启发。source 如果写成 radial jitter，长期容易被 optimizer 洗掉；写成 stable angular transport 更可能保留。

分解：

$$
u = u_{radial}+u_{tangent}
$$

其中：

$$
u_{radial}
=
\frac{\langle u,w\rangle}{\|w\|^2+\epsilon}w
$$

$$
u_{tangent}=u-u_{radial}
$$

operator：

$$
T_{angular}(\delta)
=
J_\theta \Pi_{tangent} u^*(\delta)
$$

记录：

```text
row_norm_drift
row_angular_velocity
radial_fraction
tangential_fraction
source_alignment_radial
source_alignment_tangential
hidden_block_energy
readout_block_energy
basis_block_energy
```

通过标准：

```text
tangential_fraction >= 0.60 for matrix/basis blocks
row_norm_drift not worse than AdamW by >10%
source_loss_h3200 nonnegative
optimizer_destructive_projection lower than raw commit
```

Codex 失败后先尝试：

```text
1. 只对 hidden/readout matrix block 使用 angular projection；
2. 对 KAN w2 readout 使用 class/basis corrected layout 后再投影；
3. 如果 loss gain 大幅下降，改为 radial cap 而不是 radial removal；
4. 如果仍失败，保留为 diagnostic，不继续独立路线。
```

## 5.5 LIO-5：Dual-Memory Retained Operator

核心假设：v22.0 显示 early/h3200 可以打开但 h4800 断链；v22.12 显示 MSE C4 但 Ranking/CE terminal 失败。问题可能是 source 没进入 slow state。

source state：

$$
s^{fast}_{t+1}=\beta_f s^{fast}_t+(1-\beta_f)T_\theta(\delta_t)
$$

$$
s^{slow}_{t+1}=\beta_s s^{slow}_t+(1-\beta_s)\Pi_{retain}T_\theta(\delta_t)
$$

commit：

$$
\Delta f_t
=T_\theta(\delta_t)
+\alpha_f s^{fast}_t
+\alpha_s s^{slow}_t
$$

记录：

```text
fast_source_norm
slow_source_norm
fast_slow_cosine
slow_source_age
slow_source_decay_rate
source_overwrite_fraction
h3200_to_h4800_decay
```

通过标准：

```text
R4800_over_3200_func >= 0.50
source_loss_h4800 >= -1e-6 on >=2 non-random adapters for official
slow_source_decay_rate <= control median
optimizer_destructive_projection <= control median
```

Codex 失败后先尝试：

```text
1. 降低 fast weight，提高 slow retention；
2. 加 control-null projection 到 slow state；
3. 加 low-bank projection 到 KAN slow state；
4. 如果 h3200 好 h4800 仍掉，输出 RetentionStateInsufficient，不继续 terminal floor 小扫。
```

## 5.6 LIO-6：Source-Loss Boundary Operator

核心假设：v22.12 最大 functional blocker 是 source_func 与 source_loss 分裂。source_func 代表 displacement 留住了，source_loss 才代表 task-useful source。

v22.13 可以保留 source-loss boundary，但它必须作为 operator objective term，不是 adapter-specific replay PID。

定义：

$$
\mathcal{J}_{source-loss}
=
\tau\max(0, -\widehat{\Delta \mathcal{L}}_{source})^2
$$

加入：

$$
T_\theta(\delta)
=
\arg\min_{\Delta f}
\left[
\mathcal{J}_{op}(\Delta f;\delta)+
\mathcal{J}_{source-loss}
\right]
$$

记录：

```text
source_func_h*
source_loss_h*
source_func_loss_gap_h*
TargetRetentionOnly_NotTaskUseful
source_loss_boundary_active_fraction
boundary_term_energy
```

通过标准：

```text
source_loss_h3200 >= -1e-6 on >=2 adapters
source_loss_h4800 >= -1e-6 on >=2 adapters for official
TargetRetentionOnly_NotTaskUseful = 0
boundary_term_energy not dominating total energy
```

Codex 失败后先尝试：

```text
1. 如果 boundary dominates，降低 boundary weight；
2. 如果 source_func 掉光，说明 boundary 太强或 source target 错；
3. 如果 MSE pass / Ranking fail，加入 pairwise antisymmetry preserving projection；
4. 如果 CE source_loss 负，检查 generic logit gauge / row-mean handling；
5. 如果 StableRandom source_loss 被修正为 pass，说明 boundary 被滥用，立即 fail。
```

---

# 6. Part D：Functional Update horizon protocol

## 6.1 Adapter suite

Official horizon 必须覆盖：

```text
Delta-MSEAdapter
Delta-LossCEAdapter
Delta-RankingAdapter
Delta-PreferenceAdapter-smoke
Delta-GenericSourceTarget
Delta-StableRandom-control
Delta-RandomMatched-control
```

其中：

```text
MSE / CE / Ranking / Preference / GenericSourceTarget:
  非随机 adapters，但 GenericSourceTarget 如果本身 scale/pathology 明显，必须单独标记 diagnostic。

StableRandom / RandomMatched:
  controls，不计入 non-random adapter pass count。
```

v22.13 新增 adapter holdout protocol：

```text
operator hyperparameter tune adapters:
  MSE + Ranking + CE smoke

holdout adapters:
  Preference + GenericSourceTarget normalized variant
```

注意：official operator 不允许 adapter-name branch；adapter holdout 只用于评价泛化，不允许在 run 中切分公式。

## 6.2 Horizon points

所有 official horizon 至少记录：

```text
h100
h400
h800
h1600
h2400
h3200
h4000
h4800
h6400
```

如果预算受限，必须至少跑：

```text
h100, h400, h800, h1600, h3200, h4800
```

不能只跑 h3200/h4800。

## 6.3 每个 horizon row 必须记录的核心字段

### Identity / provenance

```text
run_id
seed
dataset
carrier
model_family
operator_id
operator_family
loss_adapter_name
adapter_is_control
adapter_seen_in_operator_tuning
same_operator_parameters_across_adapters
uses_adapter_name_for_direction
uses_loss_formula_for_direction
uses_labels_for_fu_core
uses_validation_test_future_query
uses_audit_metric_for_direction
```

### Operator law / geometry

```text
operator_linearity_error
operator_homogeneity_error
operator_lipschitz_ratio
adapter_renaming_output_rel_error
GainRatio_by_adapter
NDS_before
NDS_after
NDS_reduction
metric_energy_L2
metric_energy_Fisher
metric_energy_Sobolev
metric_energy_RKHS
control_projection_before
control_projection_after
ControlNullRatio
noise_into_signal_ratio
corrupt_projection_fraction
```

### Actuation / commit

```text
projection_residual_Gf
ActuationR2
B2_transfer_gain
function_displacement_cos_with_operator_target
function_displacement_cos_with_controls
JVP_count
VJP_count
solve_time_ms
commit_time_ms
source_state_write_fraction
optimizer_state_write_fraction
block_energy_hidden
block_energy_readout
block_energy_basis
```

### Source / retention

```text
source_func_h100
source_func_h400
source_func_h800
source_func_h1600
source_func_h2400
source_func_h3200
source_func_h4000
source_func_h4800
source_func_h6400

source_loss_h100
source_loss_h400
source_loss_h800
source_loss_h1600
source_loss_h2400
source_loss_h3200
source_loss_h4000
source_loss_h4800
source_loss_h6400

row_positive_count_h3200
row_positive_count_h4800
R1600_over_800_func
R3200_over_1600_func
R4800_over_3200_func
R6400_over_4800_func
R4800_over_3200_loss
source_state_decay_rate
source_state_overwrite_fraction
optimizer_destructive_projection_h*
TargetRetentionOnly_NotTaskUseful
```

### Task readback

这些只能 readback / gate，不能生成 direction：

```text
train_loss_h*
train_accuracy_h*
AUC_loss_step
AUC_loss_time
AUC_loss_time_ratio_vs_best_control
time_to_threshold_80pct
time_to_threshold_90pct
NLL_delta
ECE_delta
Brier_delta
LineC_fast
LineC_channel
per_example_loss_q95_debt
per_example_loss_q99_debt
```

## 6.4 C3/C4 gates

### C3：source formation pass

对单个 non-random adapter：

```text
source_func_h100 >= 0.005
source_func_h400 >= 0.005
source_func_h800 >= 0.005
source_func_h1600 >= 0.005
source_func_h3200 >= 0.005
source_loss_h3200 >= -1e-6
row_positive_count_h3200 >= 7/9
StableRandom / RandomMatched controls fail
control_projection_after <= 0.25
TargetRetentionOnly_NotTaskUseful = 0
```

探索级 C3：

```text
C3 pass on >=2 non-random adapters
```

官方级 C3：

```text
C3 pass on >=3 non-random adapters
including at least one non-MSE adapter
```

### C4：terminal retention pass

对单个 non-random adapter：

```text
source_func_h4800 >= 0.005
source_loss_h4800 >= -1e-6
R4800_over_3200_func >= 0.50
R4800_over_3200_loss >= 0.25 or source_loss_h4800 absolute positive margin >= 1e-5
row_positive_count_h4800 >= 7/9
StableRandom / RandomMatched controls fail
AUC_loss_time_ratio_vs_best_control <= 1.05
debt not exploded
```

探索级 C4：

```text
C4 pass on >=1 non-MSE adapter
or C4 pass on MSE + C3 pass on Ranking/Preference
```

官方级 C4：

```text
C4 pass on >=2 non-random adapters
including at least one non-MSE adapter
CE failure cannot be hidden by MSE-only success
```

### C5：h6400 robustness pass

如果 h4800 通过，必须扩到 h6400：

```text
source_func_h6400 >= 0.005
source_loss_h6400 >= -1e-6
R6400_over_4800_func >= 0.50
no late StableRandom convergence into same source
```

如果 h6400 失败：

```text
route = H4800RetainedButH6400Fragile
```

不允许直接 full scientific claim。

## 6.5 如果 adapter 失败，如何判读

```text
MSE C4, Ranking C3, CE fail:
  route = LossInterfacePartiallyOpened_CENotSolved

MSE C4 only:
  route = MSESpecificSource_NotArbitraryLoss

source_func pass but source_loss negative:
  route = TargetRetentionOnly_NotTaskUseful

StableRandom pass:
  route = ControlEquivalentOrSourceDefinitionBroken

CE source_func negative from h100:
  diagnostic = LogitGaugeOrHighCurvatureCECotangentBlocked

Ranking source_func positive but source_loss h4800 negative:
  diagnostic = PairwiseSourceLossRetentionBlocked

GenericSourceTarget huge negative source_loss:
  diagnostic = AdapterScalePathologyOrTargetReplayNotTaskUseful
```

---

# 7. Part E：Kernel-native arbitrary-cotangent efficiency

## 7.1 目标

v22.12 的效率证据是：

```text
manual_upstream_vjp_rows = 360
official_fused_kernel_complete_rows = 0
D-FOU pass
D-CHE/D-RAT/D-RBF near/blocked
```

v22.13 的效率目标必须升级：

$$
\boxed{
\text{不是只测 basis forward，也不是 manual VJP，}
\text{而是测 } \delta \rightarrow T_\theta(\delta) \rightarrow u^* \rightarrow commit
\text{ 的 kernel-native full operator step。}
}
$$

## 7.2 Efficiency runner fields

每个 row 记录：

```text
carrier
variant
operator_id
cotangent_type
batch_size
hidden
basis_dim
forward_ms
native_vjp_ms
manual_vjp_ms
basis_gram_ms
spectral_truncation_ms
operator_apply_ms
param_solve_ms
functional_commit_ms
optimizer_update_ms
full_operator_step_ms
full_training_step_ms
forward_memory_mb
backward_memory_mb
workspace_memory_mb
basis_materialized_bytes
kernel_count
fused_kernel_used
manual_upstream_vjp_used
fallback_kernel_used
official_fused_kernel_complete
component_telemetry_complete
functional_runner_kernel_match
forward_ratio_vs_mlp
vjp_ratio_vs_mlp
operator_step_ratio_vs_mlp
full_step_ratio_vs_mlp
memory_ratio_vs_mlp
gradcheck_pass
upstream_cotangent_contract_pass
```

## 7.3 Official efficiency gates

Exploration efficiency E1：

```text
forward_ratio_vs_mlp <= 1.75
operator_step_ratio_vs_mlp <= 1.75
full_step_ratio_vs_mlp <= 1.50
memory_ratio_vs_mlp <= 1.10
gradcheck_pass = 1
fallback_kernel_used = 0
```

Official native S1：

```text
official_fused_kernel_complete = 1
manual_upstream_vjp_used = 0
fused_kernel_used = 1
functional_runner_kernel_match = 1
component_telemetry_complete = 1
forward_ratio_vs_mlp <= 1.25
vjp_ratio_vs_mlp <= 1.25
operator_step_ratio_vs_mlp <= 1.35
full_step_ratio_vs_mlp <= 1.25
memory_ratio_vs_mlp <= 1.10
gradcheck_pass = 1
upstream_cotangent_contract_pass = 1
```

Official operator-carrier success：

```text
S1 native pass on >=3 batch sizes
S1 native pass on >=4 cotangent types
S1 native pass on at least one non-MSE adapter cotangent
no manual row counted as official
```

## 7.4 D-FOU officialization first

D-FOU 是 v22.12 最稳 carrier。v22.13 先把它 officialize。

主候选：

```text
FOU22.13-R1-lowfreq-bandreadout-native-vjp
FOU22.13-R2-tablelookup-bandreadout-native-operator-step
FOU22.13-R3-lowfreq-source-state-fused-commit
```

必须验证：

```text
1. arbitrary cotangent native VJP；
2. low-frequency basis Gram 不 materialize dense basis；
3. operator apply 与 training runner 使用同一 kernel；
4. K13 corrected layout 后 source pass 仍成立；
5. MSE/Ranking/CE cotangent 都不走 special branch。
```

失败时 Codex 先尝试：

```text
1. 固定 frequency table buffer，避免重复构造；
2. 合并 band-readout contraction 与 upstream VJP；
3. 把 source-state commit 融入 readout update kernel；
4. 若 operator_step 慢但 raw VJP 快，定位 basis Gram / spectral truncation；
5. 若 corrected K13 掉，输出 DFOUK13LegacyLayoutArtifactRisk，不允许沿用旧 K13 promotion。
```

## 7.5 D-CHE corrected layout + native officialization

D-CHE 是 v22.13 不能跳过的 carrier，因为 K14 证明 D-CHE source-exists 可能被 layout 修复打开。

主候选：

```text
CHE22.13-R1-k3-corrected-layout-native-vjp
CHE22.13-R2-k5-gradbuf-no-materialize-native-vjp
CHE22.13-R3-lowdegree-readout-operator-step
CHE22.13-R4-corrected-layout-source-state-fused-commit
```

必须验证：

```text
1. corrected readout layout unit test pass；
2. K14 source-exists row 用新 layout 重跑；
3. D-CHE 自己的 carrier_specific_efficiency_pass = 1；
4. h4800 source_loss 修复不能靠 replay scale；
5. D-CHE row 不借 D-FOU efficiency。
```

失败时 Codex 先尝试：

```text
1. 如果 forward_ratio blocked：检查 Chebyshev recurrence 是否 materialize；
2. 如果 VJP blocked：合并 degree recurrence 与 upstream contraction；
3. 如果 step blocked：分离 telemetry/audit path，不放进 train path；
4. 如果 h4800 source_loss 负：加 slow source-state + low-degree retention objective；
5. 如果 D-CHE source-exists 但 efficiency blocked：route = DCHE_SourceExists_EfficiencyBlocked。
```

## 7.6 D-RBF / D-RAT secondary route

D-RBF 有局部 pass rows，但 robustness 不够；D-RAT forward/step 较慢。v22.13 不把它们作为主线，但保留 secondary diagnostic。

D-RBF 尝试：

```text
RBF22.13-R1-local-K4-native-vjp
RBF22.13-R2-local-backward-fused-scatter
RBF22.13-R3-active-center-hard-sparse
```

D-RAT 尝试：

```text
RAT22.13-R1-num-den-native-vjp
RAT22.13-R2-reciprocal-approx-no-telemetry
RAT22.13-R3-denom-safety-separated
```

停止条件：

```text
如果 D-FOU official native pass 且 D-CHE source carrier 正在推进，D-RBF/D-RAT 不消耗主预算。
如果 D-FOU/D-CHE 都 blocked，D-RBF 可作为 fallback efficient local carrier。
D-RAT 只有在 step_ratio <= 1.50 near-E1 时进入 functional mapping。
```

---

# 8. Part F：KAN corrected source mapping

## 8.1 目标

KAN mapping 的目标是证明：

$$
\boxed{
\text{KAN carrier 能实现 } T_\theta(\delta)
\text{ 并长期保留 task-useful source。}
}
$$

不是证明：

```text
某个 readout target replay 能拟合一个 Δf。
```

## 8.2 KAN mapping candidates

### K15：D-FOU K13 corrected-layout revalidation

目的：确认 v22.12 D-FOU K13 不是 legacy layout artifact。

记录：

```text
legacy_K13_source_func_h3200
legacy_K13_source_loss_h3200
legacy_K13_source_loss_h4800
corrected_K13_source_func_h3200
corrected_K13_source_loss_h3200
corrected_K13_source_loss_h4800
corrected_vs_legacy_delta
carrier_specific_efficiency_pass
```

通过标准：

```text
corrected_K13_source_func_h3200 >= 0.005
corrected_K13_source_loss_h3200 >= -1e-6
corrected_K13_source_loss_h4800 >= -1e-6
carrier_specific_efficiency_pass = 1
```

失败时：

```text
如果 corrected K13 掉：旧 K13 降级为 layout-uncertain diagnostic；
如果 corrected K13 更强：D-FOU 进入 primary KAN operator carrier。
```

### K16：D-CHE K14 h4800 source-loss repair without replay-scale sweep

目的：K14 已打开 D-CHE source-exists，但 h4800 source_loss 负，且 efficiency blocked。K16 只允许用 operator/source-state 修，不允许简单 fan_scale 小扫。

尝试：

```text
K16a corrected-layout low-degree Green operator
K16b corrected-layout source-loss boundary operator
K16c corrected-layout slow source-state low-degree retention
K16d corrected-layout control-null low-degree operator
```

记录：

```text
basis_estimate_fit_cosine
basis_commit_projection_residual
KAN_source_func_h3200
KAN_source_func_h4800
KAN_source_loss_h3200
KAN_source_loss_h4800
D-CHE_efficiency_pass
source_loss_boundary_active_fraction
slow_source_decay_rate
```

通过标准：

```text
KAN_source_func_h3200 >= 0.005
KAN_source_loss_h3200 >= -1e-6
KAN_source_func_h4800 >= 0.005
KAN_source_loss_h4800 >= -1e-6
D-CHE_efficiency_pass = 1 for official
```

失败时：

```text
如果 h4800 source_loss 仍负：route = DCHE_SourceExists_SourceLossBlocked；
如果 efficiency 仍不过：route = DCHE_SourceExists_EfficiencyBlocked；
如果二者都过：D-CHE official carrier candidate。
```

### K17：True basis-channel spectral commit

目的：区分 readout-only source 与 basis-channel source。

定义：

```text
readout commit:
  only w2 / readout update。

basis-channel commit:
  w1 / basis parameters / low-degree or low-frequency bank update。

hybrid commit:
  low-bank basis + readout joint update。
```

记录：

```text
commit_channel
readout_channel_energy
basis_channel_energy
basis_to_readout_energy_ratio
basis_channel_source_func_h*
basis_channel_source_loss_h*
readout_only_ablation_source_func_h*
readout_only_ablation_source_loss_h*
```

通过标准：

```text
basis_channel_energy > 0
basis_channel_source_loss_h3200 >= -1e-6
basis_channel_source_loss_h4800 >= -1e-6 for official
basis_channel improves over readout-only by >=0.005 source_loss or same source with lower memory/time
```

失败时：

```text
如果 readout-only pass / basis fail：route = KANReadoutSourceOpened_BasisChannelBlocked；
不写成 KAN basis success。
```

---

# 9. Part G：Task-level evidence：表达能力、抗遗忘、收敛速度

v22.13 不能只做 source metrics。项目总 claim 是 functional update 改善普通 BP 的缺点。因此必须加 task-level evidence，但这些 task metrics 只能作为 readback/gate，不能生成 direction。

## 9.1 Baselines

必须比较：

```text
same-param MLP + AdamW
same-param MLP + SGD/Momentum
strict FC-PureKAN + AdamW
strict FC-PureKAN + ordinary BP controls
strict FC-PureKAN + FU operator
MLP + same FU operator generic control
KAN + random matched FU control
KAN + stable random FU control
KAN + O10 diagnostic role-aware control
```

## 9.2 Datasets / seeds

最低矩阵：

```text
MNIST seeds 0/1/2
Fashion-MNIST seeds 0/1/2
KMNIST seeds 0/1/2
```

如果资源允许：

```text
CIFAR-10 grayscale smoke
synthetic regression adapter smoke
pairwise ranking adapter smoke
```

注意：额外任务只作为 adapter generality smoke，不改变主 promotion gate。

## 9.3 表达能力指标

记录：

```text
final_train_loss
final_test_loss_readback
final_train_accuracy
final_test_accuracy_readback
NLL_delta_vs_MLP
ECE_delta_vs_MLP
Brier_delta_vs_MLP
function_rank_effective
feature_covariance_rank
class_margin_mean
class_margin_q10
low_degree_energy_fraction
low_frequency_energy_fraction
```

表达能力通过标准：

```text
KAN+FU final accuracy >= same-param MLP on >=7/9 rows
KAN+FU final NLL not worse than MLP by >0.02 on >=7/9 rows
KAN+FU feature/function rank not collapsed
KAN+FU source improvements not paid by calibration collapse
```

## 9.4 抗遗忘指标

这里的“遗忘”不是必须引入外部 continual benchmark，而是先在当前 train stream 中定义 source forgetting：FU 写入的 source 在普通继续训练中是否被 washout。

记录：

```text
source_forgetting_h800_to_h3200
source_forgetting_h3200_to_h4800
old_micro_split_loss_delta_h*
new_micro_split_loss_delta_h*
source_state_decay_rate
optimizer_destructive_projection
source_overwrite_fraction
control_matched_forgetting_rate
```

抗遗忘通过标准：

```text
R3200_over_1600_func >= 0.40
R4800_over_3200_func >= 0.50
source_loss_h4800 >= -1e-6
source_forgetting lower than AdamW/random controls by >=25%
```

如果要加 continual smoke，只允许：

```text
train-only split stream；
no validation/test direction；
old-task readback only；
no replay buffer unless baseline also has it。
```

## 9.5 收敛速度指标

记录：

```text
AUC_loss_step
AUC_loss_time
AUC_accuracy_step
AUC_accuracy_time
time_to_80pct_best_control
time_to_90pct_best_control
steps_to_threshold
wallclock_to_threshold
operator_overhead_ratio
full_step_ratio_vs_MLP
```

收敛速度通过标准：

```text
AUC_loss_time_ratio_vs_best_control <= 1.05
or time_to_threshold improves by >=10% while final/debt not worse
operator_overhead_ratio <= 1.35 exploration
operator_overhead_ratio <= 1.25 official
```

---

# 10. Part H：Controls 与 forbidden information firewall

## 10.1 必须 controls

每个 S2/S3/S4/S5/S6/S7 run 必须有：

```text
NoOpMatchedOverhead
RandomMatchedNorm
StableRandom
SignFlipCotangent
CorruptTarget
SameSolverRandomTarget
AdamWParallelDirection
SGDParallelDirection
MomentumParallelDirection
OptimizerStateOnlyNoSource
TargetOnlyNoLossControl
O10RoleAwareDiagnosticControl
MLPGenericFUControl
KANOrdinaryBPControl
```

## 10.2 Control attribution

记录：

```text
control_id
control_norm_match
control_time_match
control_projection_fraction
control_source_func_h*
control_source_loss_h*
control_C3_pass
control_C4_pass
candidate_minus_best_control_source_func
candidate_minus_best_control_source_loss
```

通过标准：

```text
candidate_source_func_h3200 >= best_control_source_func_h3200 + 0.005
candidate_source_loss_h3200 >= best_control_source_loss_h3200 + 1e-6
candidate_source_func_h4800 >= best_control_source_func_h4800 + 0.005 for official
candidate_source_loss_h4800 >= best_control_source_loss_h4800 + 1e-6 for official
StableRandom C3/C4 = 0
```

## 10.3 Forbidden information

严格禁止：

```text
validation/test/future/query as direction
dataset-name branch
seed-specific branch
CE-specific source formula
MSE-specific source formula
Ranking-specific source formula
LineC/ECE/Brier/AUCtime/tail as direction
classwise CE-tail hard target
loss adapter name branch
manual result inspection affecting same-run direction
```

允许：

```text
current train-stream logits
current train-stream activations
current upstream cotangent δ from generic LossInterface
past train-stream source state
current optimizer state
current batch/micro-split statistics
labels inside supervised loss adapter only
metric geometry derived from train stream only
```

---

# 11. Part I：哪些旧方向重开，哪些停止

## 11.1 因代码/layout 问题必须重开的方向

### D-CHE low-degree / readout source writer

原因：v22.12 K14 corrected layout 证明旧 D-CHE negative 可能被 class/basis layout bug 污染。

重开方式：

```text
不重开旧 K1-K13 原样；
只用 corrected layout + role-blind operator + source_loss gate 重开；
必须区分 source-exists、source-loss、efficiency 三个 gate。
```

### D-FOU K13 retained source

原因：我独立判断 D-FOU legacy solve 也可能受 layout 影响。

重开方式：

```text
K13_corrected 必须跑；
旧 K13 只能作为 prior evidence，不能作为 v22.13 official row。
```

## 11.2 旧 observer/filter 可作为 operator component 重开

```text
Split-consensus:
  不再做 selector；
  改为 P_signal 的 train micro-split coherent subspace。

Control-nullspace:
  不再筛 update；
  改为 P_control-null，去除 AdamW/SGD/random/stable-random 可解释方向。

Metric observer:
  不再事后打分；
  改为 Gθ / NDS / Sobolev / Fisher operator energy。

PopRisk / SNR:
  不再直接决定 update；
  改为 NoiseLeak / signal-reservoir separation term。

Train-loss selector:
  不作为 direction selector；
  只作为 source_loss / optimizer destructive projection diagnostic。

F53 terminal near-miss:
  不作为主线继续小扫；
  只作为 h3200->h4800 destructive projection localization diagnostic。
```

## 11.3 必须停止的方向

```text
1. terminal floor scale sweep；
2. lookahead gate 小修；
3. hold/fallback 小修；
4. O10 role-aware consensus 伪装成 official arbitrary-loss operator；
5. MSE-only C4 继续放大；
6. readout target replay 写成 basis-channel success；
7. D-CHE 借 D-FOU efficiency；
8. manual VJP ratio pass 写成 official fused pass；
9. source_func positive 但 source_loss 负写成 functional success。
```

停止原因：

```text
v22.0 已经证明 h3200 continuous 不等于 h4800 retention；
v22.12 已经证明 MSE C4 不等于 arbitrary-loss success；
继续小修只会产生更漂亮的 diagnostic row，不会产生算法突破。
```

---

# 12. Part J：4GPU dynamic execution plan

## 12.1 初始分工

```text
GPU0:
  S0 code / semantic truth gate；
  D-FOU kernel-native officialization；
  K15 D-FOU corrected K13 revalidation。

GPU1:
  role-blind operator construction；
  LIO-1 / LIO-2 / LIO-5 MLP operator horizon；
  adapter holdout horizon。

GPU2:
  D-CHE corrected layout officialization；
  K16/K17 D-CHE corrected mapping；
  D-CHE h4800 source-loss repair。

GPU3:
  controls；
  task-level expression/forgetting/convergence readback；
  finalizer / figures / packet generation。
```

## 12.2 Dynamic refill rules

```text
If GPU0 finishes D-FOU early:
  take D-CHE native VJP batch512 repair;
  then KAN mapping controls;
  then efficiency figures.

If GPU1 finishes operator horizon early:
  run independent seed offset;
  then holdout adapter;
  then MLP generic FU control.

If GPU2 finishes D-CHE early:
  run D-FOU corrected K13 independent confirmation;
  then D-RBF fallback;
  then KAN basis/readout ablation.

If GPU3 finishes controls early:
  run StableRandom extended h6400;
  then task-level convergence readback;
  then code packet validation.
```

## 12.3 Queue artifacts

必须输出：

```text
v22_13_runnable_queue.csv
v22_13_gpu_assignment_manifest.csv
v22_13_gpu_utilization_timeline.csv
v22_13_idle_violation.csv
v22_13_queue_drain_report.json
v22_13_deferred_items.csv
```

硬规则：

```text
if runnable_queue_nonempty and any_gpu_idle > 10 minutes:
    execution_contract_violation = 1
    official_promotion_allowed = 0
```

---

# 13. Required artifacts

## 13.1 Code truth artifacts

```text
v22_13_code_review_packet.zip
v22_13_required_source_files.csv
v22_13_compileall_report.csv
v22_13_import_closure.csv
v22_13_semantic_firewall.csv
v22_13_operator_role_blind_tests.csv
v22_13_adapter_renaming_tests.csv
v22_13_operator_law_tests.csv
v22_13_corrected_layout_tests.csv
v22_13_promotion_semantics_tests.csv
```

## 13.2 Efficiency artifacts

```text
v22_13_native_efficiency_truth_table.csv
v22_13_native_kernel_gradcheck.csv
v22_13_official_fused_status_matrix.csv
v22_13_operator_step_efficiency.csv
v22_13_component_timing_waterfall.csv
v22_13_manual_vs_native_vjp_comparison.csv
v22_13_DFOU_officialization_matrix.csv
v22_13_DCHE_officialization_matrix.csv
```

## 13.3 Operator FU artifacts

```text
v22_13_operator_atom_matrix.csv
v22_13_metric_geometry_matrix.csv
v22_13_variational_operator_solve.csv
v22_13_operator_commit_matrix.csv
v22_13_adapter_horizon_matrix.csv
v22_13_adapter_holdout_matrix.csv
v22_13_source_loss_boundary_matrix.csv
v22_13_source_state_dynamics.csv
v22_13_control_attribution_matrix.csv
```

## 13.4 KAN artifacts

```text
v22_13_K15_DFOU_corrected_K13.csv
v22_13_K16_DCHE_corrected_source_loss_repair.csv
v22_13_K17_basis_channel_commit.csv
v22_13_KAN_mapping_matrix.csv
v22_13_readout_vs_basis_ablation.csv
v22_13_KAN_vs_MLP_same_operator.csv
```

## 13.5 Task-level artifacts

```text
v22_13_task_eval_matrix.csv
v22_13_convergence_speed_matrix.csv
v22_13_forgetting_readback_matrix.csv
v22_13_expression_metrics_matrix.csv
v22_13_calibration_debt_matrix.csv
```

## 13.6 Figures

```text
v22_13_operator_law_dashboard.svg
v22_13_metric_as_operator_pareto.svg
v22_13_adapter_horizon_source_func.svg
v22_13_adapter_horizon_source_loss.svg
v22_13_control_projection_dashboard.svg
v22_13_DFOU_native_efficiency.svg
v22_13_DCHE_corrected_layout_efficiency.svg
v22_13_KAN_source_channel_dashboard.svg
v22_13_convergence_forgetting_expression_dashboard.svg
v22_13_gpu_utilization_dashboard.svg
```

---

# 14. Final route definitions

v22.13 finalizer 必须选择一个 route：

```text
R0-CodeTruthFailed
R1-OperatorSemanticGateFailed
R2-OnlyRoleAwareConsensus_NotTrueOperator
R3-KernelNativeEfficiencyBlocked
R4-MSEOnlySource_NotArbitraryLoss
R5-TargetRetentionOnly_NotTaskUseful
R6-OperatorC3Opened_C4TerminalBlocked
R7-DFOUCorrectedSourceOpened_DCHEBlocked
R8-DCHESourceExists_EfficiencyOrSourceLossBlocked
R9-TrueLossInterfaceOperatorExplorationSuccess_KANPending
R10-KANOperatorCarrierExplorationSuccess_OfficialFusedPending
R11-OfficialOperatorCarrierReady_TaskEvidencePending
R12-FullScientificPromotionReady
```

## 14.1 Exploration promotion

允许进入下一轮，如果：

```text
S0 pass
role-blind semantic gate pass
至少一个 official operator atom pass
C3 pass on >=2 non-random adapters
D-FOU or D-CHE corrected KAN source evidence exists
controls fail
```

## 14.2 Official operator promotion

允许写 true loss-interface operator FU success，如果：

```text
S0 pass
operator role-blind static/runtime/renaming tests pass
O10 not counted as official
C3 pass on >=3 non-random adapters
C4 pass on >=2 non-random adapters
including at least one non-MSE adapter
source_loss_h4800 nonnegative
StableRandom / RandomMatched fail
operator law metrics pass
adapter holdout pass
```

## 14.3 Official KAN carrier promotion

允许写 KAN carrier success，如果：

```text
official operator promotion pass
carrier_specific_efficiency_pass = 1
native kernel official_fused_kernel_complete_rows > 0
manual_upstream_vjp_used = 0 for official rows
KAN_source_loss_h4800 >= -1e-6
KAN_specific_delta_vs_MLP_same_metric >= 0.005
or KAN_same_source_with_better_efficiency = 1
readout-only / basis-channel claim separated
```

## 14.4 Full scientific promotion

允许写“functional update 改善纯梯度反传缺点”如果：

```text
official operator promotion pass
official KAN carrier promotion pass
KAN+FU beats same-param MLP on >=7/9 task rows
KAN+FU beats PureKAN ordinary BP on >=7/9 task rows
source forgetting lower than controls by >=25%
AUC_loss_time_ratio_vs_best_control <= 1.05
step_time_ratio_vs_MLP <= 1.25
memory_ratio_vs_MLP <= 1.25
LineC/ECE/Brier/NLL debt not exploded
```

---

# 15. Failure taxonomy and Codex next actions

## Case A：code / semantic gate fails

触发：

```text
compileall fail
import closure fail
adapter-name branch in official operator
loss formula branch in official operator
layout tests fail
promotion semantics fail
```

Codex action：

```text
1. 修源码与 semantic firewall；
2. 重跑 S0 only；
3. 不跑长实验；
4. route = R0 or R1；
5. 不写 scientific no-go。
```

## Case B：operator 只能以 O10 role-aware consensus 通过

触发：

```text
O10 diagnostic pass
role-blind operators fail
adapter renaming test fail
```

Codex action：

```text
1. 抽取 O10 有效成分：low-NDS、source-loss boundary、control-null；
2. 移除 adapter suite branch；
3. 重新实现为 δ-only operator；
4. 重跑 adapter renaming / operator law tests；
5. route = R2-OnlyRoleAwareConsensus_NotTrueOperator。
```

## Case C：MSE C4 but Ranking/CE fail

触发：

```text
MSE C4 = 1
Ranking C4 = 0
CE C3/C4 = 0
```

Codex action：

```text
1. 不放大 MSE；
2. 记录 GainRatio / NDSReduction / ControlNullRatio by adapter；
3. 对 Ranking 加 pairwise antisymmetry preserving projection；
4. 对 CE 检查 generic logit gauge / class mean handling，不写 CE branch；
5. 加 adapter holdout；
6. route = R4-MSEOnlySource_NotArbitraryLoss if still fails。
```

## Case D：source_func pass but source_loss negative

触发：

```text
source_func_h3200/h4800 positive
source_loss_h3200/h4800 negative
```

Codex action：

```text
1. 标记 TargetRetentionOnly_NotTaskUseful；
2. 增加 source-loss boundary term；
3. 降低 target replay / displacement norm；
4. 提高 control-null / low-curvature penalty；
5. 如果 source_func 与 source_loss 长期分裂，停止该 operator family。
```

## Case E：StableRandom / RandomMatched 通过

触发：

```text
StableRandom C3 or C4 pass
RandomMatched C3 or C4 pass
```

Codex action：

```text
1. 禁止 promotion；
2. 检查 source metric 是否只是 norm/retention artifact；
3. 加 control attribution；
4. 加 source_loss / task-useful gate；
5. route = ControlEquivalentOrSourceDefinitionBroken。
```

## Case F：D-FOU corrected pass, D-CHE fail

触发：

```text
D-FOU K13 corrected pass
D-CHE K16 fail or efficiency blocked
```

Codex action：

```text
1. D-FOU 作为 primary KAN operator carrier；
2. D-CHE 写成 source-exists/blocked，不写 KAN-general success；
3. 继续 D-CHE native VJP officialization；
4. 如果 D-CHE source_loss 负，停止 scale sweep，改 source-state/metric objective。
```

## Case G：native fused rows remain zero

触发：

```text
official_fused_kernel_complete_rows = 0
manual_upstream_vjp_rows > 0
```

Codex action：

```text
1. 不写 official efficiency success；
2. D-FOU 先做 native VJP；
3. D-CHE 同步 corrected layout native VJP；
4. final route 最多 R10，不得 R12。
```

## Case H：task improves but source not retained

触发：

```text
accuracy/loss improves
source_func/source_loss retention fails
```

Codex action：

```text
1. 写 task improvement diagnostic；
2. 不写 functional update mechanism success；
3. 检查是否普通 optimizer / regularization artifact；
4. 加 MLP/KAN ordinary BP controls。
```

## Case I：source retained but convergence/efficiency worse

触发：

```text
source C4 pass
AUC_loss_time worse > 1.05
step_time_ratio > 1.35 exploration or >1.25 official
memory_ratio > 1.25
```

Codex action：

```text
1. 写 FunctionalRetainedSource_SystemCostBlocked；
2. 优先 kernel-native operator step；
3. 减少 Gram rank / CG iterations；
4. 把 expensive metric 从 train path 移到 amortized source-state path。
```

---

# 16. Minimum effective progress

v22.13 不能返回模糊 no-go。至少交付以下之一：

```text
A-CodeTruth:
  role-blind operator semantic gate + corrected layout tests + promotion split 全部落盘。

B-Operator:
  至少一个非 O10 role-blind operator 通过 S2/S3/S4，并在 >=2 non-random adapters C3。

C-Adapter:
  至少一个非 MSE adapter 达到 C4，或者明确定位为什么 CE/Ranking source_loss 失败。

D-Efficiency:
  D-FOU 或 D-CHE 至少一个 native official fused arbitrary-cotangent row > 0。

E-KAN:
  D-FOU K13 corrected 重验完成；D-CHE K14 source-exists 的 h4800 source_loss 或 efficiency blocker 被明确定位。

F-Theory:
  如果全部失败，必须输出 operator-level no-go：失败发生在 semantic、metric operator、commit、adapter horizon、KAN carrier、还是 efficiency。
```

---

# 17. 最终解释模板

v22.13 最终复盘必须按下面格式解释，不允许只写 route 名。

```text
1. Code truth:
   源码是否完整？role-blind 是否真实？layout 是否修正？

2. Efficiency:
   D-FOU/D-CHE 是否 native official？manual VJP 是否仍是 blocker？

3. Operator:
   是否有 true δ -> Δf operator？还是只有 O10 role-aware diagnostic？

4. Adapter robustness:
   MSE/Ranking/CE/Preference 哪些 C3/C4？source_func/source_loss 是否分裂？

5. KAN carrier:
   D-FOU corrected K13 是否成立？D-CHE K14 是否推进到 efficiency/source_loss closure？

6. Task-level value:
   表达能力、抗遗忘、收敛速度是否相对 MLP/PureKAN BP controls 有真实改善？

7. Next action:
   如果失败，下一步是修 semantic、operator geometry、adapter source_loss、KAN carrier、还是 kernel officialization？
```

最终解释必须避免：

```text
1. 把 MSE-only 写成 arbitrary-loss；
2. 把 source_func 写成 task-useful；
3. 把 manual VJP 写成 official fused；
4. 把 D-FOU 成功写成 D-CHE/KAN-general；
5. 把 role-aware O10 写成 true role-blind operator；
6. 把 task metric improvement 写成 FU mechanism proof。
```

---

# 18. 结语

v22.13 的核心不是继续堆工程，而是把 functional update 正式定义成一个训练动力学算子：

$$
\boxed{
\delta
\xrightarrow{\quad T_\theta \quad}
\Delta f
\xrightarrow{\quad J_\theta^\dagger / G_\theta \quad}
u^*
\xrightarrow{\quad source\ state \quad}
\text{retained task-useful source}
}
$$

如果这条链成立，DG-KAN 才能真正证明：

```text
1. functional update 不是 observer/filter；
2. functional update 不是 MSE target replay；
3. functional update 能改善普通梯度反传的 source washout / high-curvature / slow convergence；
4. KAN low-degree / low-frequency basis 是高效 carrier，而不是慢速装饰。
```

v22.13 的成功标准因此很高，但也更清楚。它要么打开 true loss-interface operator FU，要么给出精确 no-go：当前失败到底是 metric geometry 不够、source_loss 定义不够、adapter law 不够、KAN carrier 不够，还是 kernel-native 系统路径不够。
