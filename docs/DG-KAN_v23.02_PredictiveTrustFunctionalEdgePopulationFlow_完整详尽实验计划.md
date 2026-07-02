# DG-KAN v23.02 Predictive-Trust Functional Edge Population Flow 完整详尽实验计划

创建时间：2026-07-02

本文档是在 v23.01 的真实实验结果之后重写的下一版实验计划。它保留 v23.00R/v23.01 中已经显露出来的正信号：Functional $L^2$ edge Gram、D-CHE K9 depth3、BlockSNR / degree-BlockSNR 对 C2 visual coordinate formation 的帮助；同时把当前真正的硬 blocker 明确收敛到 **finite-step safety trust efficiency** 和 **taskwise C2 robustness**。本计划不把“继续扫超参”当作研究路线，也不把“每步强行 finite-step 拒绝”当成可 promotion 的算法。我们要验证一个更清楚、更优雅的理论命题：

$$
\boxed{
\text{KAN-FU 的下一步不是再找 output signal channel，}
\text{而是在 Functional }L^2\text{ edge 几何中，}
\text{用 population gate 形成任务坐标，}
\text{再用可预测的 finite-step trust region 保证 safety。}
}
$$

v23.01 已经给出关键事实。Functional edge Gram + Qpop block gate 在 D-CHE K9 depth3 的双任务总体 C2 上有正信号，尤其 gate floor 提到 0.3 后 E4/E5 打过 random matched；但 local_patch interaction 的 taskwise 中位数仍为负，说明 C2 formation 还不够全面。另一方面，严格 finite-step guard 可以让 F5/no-debt 闭合，但依赖 heavy rejection；当 guard cadence 放松到 every 2 或 every 3 时 accept/scale 改善，但 F5/no-debt 立刻失败。因此 v23.02 的目标不是降低 no-debt gate，不是把 heavy rejection 写成成功，而是把 strict finite-step guard 的真实判断转化成一个**低成本、可预测、可审计、可连续训练**的 trust rule。

本文档写给后续实现者和 Codex。它必须被当成实验契约，而不是方向提纲。每一个 Part 都写清楚实验目标、假设、公式、实现方式、必须落盘字段、通过标准、失败解释，以及不满足条件时 Codex 允许优先尝试的修复。所有公式使用 Typora 友好的 `$` 或 `$$`，不使用 `\[\]`。

---

## 一、理论收敛：从 heavy rejection 到 predictive trust

我们现在已经不再处于“KAN 是否有 C2 visual capacity”这个阶段。v22.93 证明 true deep PureKAN 架构在 AdamW 下有 visual architecture signal；v22.94 证明 AdamW witness 中被 metric branch 删除的 freedom 是真实的，释放后可以打开 C2，但 F5/no-debt 会崩；v23.00R continuation 证明 true dense FunctionalGram $s=0$ 接入训练后能产生 C2 信号，而 FunctionalGram $s=1$ derivative-heavy 版本失败；v23.01 进一步证明 FunctionalGram $s=0$ + BlockSNR / degree-BlockSNR 可以在顶层 Part E 通过 C2 formation gate，但 F5/no-debt 只能靠重度 finite-step rejection diagnostic pass。

因此新的理论层次应该是：

$$
G_{edge}^{L2} = \int_{-1}^{1} \psi(u)\psi(u)^T\,du + \epsilon I,
$$

$$
M_t = (G_{edge}^{L2})^{-1/2} Q_{pop,t} (G_{edge}^{L2})^{-1/2},
$$

$$
\theta^+ = \theta + \alpha_t \Delta\theta_t,
$$

其中 $G_{edge}^{L2}$ 是 D-CHE / D-FOUR 边函数空间的 formation-phase Hilbert 几何；$Q_{pop,t}$ 是在这个几何中计算的 population-risk gate，负责让 coherent task directions 进入更新；$\alpha_t$ 是由 train-only trust predictor 或 trust controller 给出的有限步安全尺度。真正的 v23.02 问题不是 $G_{edge}$ 是否漂亮，也不是 $Q_{pop}$ 是否单独在诊断指标上赢，而是如下组合是否成立：

$$
\boxed{
\text{Functional }L^2\text{ edge geometry}
+
\text{Block-level population gate}
+
\text{Predictive finite-step trust}
\Rightarrow
\text{C2 formation + efficient F5 no-debt。}
}
$$

这里的 trust 不应该是 brute-force accept/reject。v23.01 的 F2 finite-step trust 已经证明 exact candidate evaluation 可以闭合 F5，但 median accept rate 约 0.05，scale mean 约 0.00625，skip count 很大。这说明 exact guard 是一个真实 safety oracle，但不能直接作为 optimizer。v23.02 要做的是从这个 oracle 中学习或估计一个可提前预测的 trust radius，使大部分 step 在进入 exact guard 前已经接近安全尺度。

理论上，我们把 safety 写成有限步约束：

$$
D_c(\theta + \alpha\Delta\theta) - D_c(\theta) \le 0,
\quad
c\in\{\mathrm{Brier},\mathrm{ECE},\mathrm{tail95},\mathrm{tail99},\mathrm{margin10}\}.
$$

理想 trust radius 是：

$$
\alpha^* = \max\left\{\alpha\in(0,1]:\max_c \Delta D_c(\theta,\alpha\Delta\theta)\le 0\right\}.
$$

v23.01 的 exact finite-step guard 通过离散搜索近似 $\alpha^*$，但效率太低。v23.02 的核心任务是构造一个 train-only predictor：

$$
\widehat{\alpha}_t = f_{trust}(x_t),
$$

其中 $x_t$ 不能包含 held/test 信息，必须由当前训练状态、guard split 上的当前 debt、task/debt gradients、population gate density、FunctionalGram norm、历史 accept/reject statistics 等组成。只有当 $\widehat{\alpha}_t$ 能在保持 F5/no-debt 的同时显著提高 accept rate 和 scale mean，才能称为真正进展。

---

## 二、全局目标与成功边界

v23.02 的整体目标不是马上进入真实任务 official，而是在 C2/F5 positive-control 上把 v23.01 的 diagnostic 可行性推进为 official 可接受的训练律。最低要求如下。

首先，C2 formation 必须在 taskwise 层面成立。v23.01 的 E4/E5 总体通过是正信号，但 local_patch interaction 的 median 仍为负。因此 v23.02 不能只看双任务合并 median。local_patch_interaction 和 rotation_sensitive 必须分别达到 formation gate；否则 C2 formation 仍然只是部分任务阳性。

其次，F5/no-debt 必须不再依赖 heavy rejection。v23.02 的 official safety gate 要求在 full 15 seeds、双 C2 任务上同时满足：

```text
F5_no_debt_count >= 12/15
component_non_positive_rows >= 12/15
finite_step_accept_rate_median >= 0.20
finite_step_scale_mean_median >= 0.05
finite_step_skip_count_median <= 0.70 * total_steps
C2_coverage_retention_vs_no_safety >= 0.70
random_veto_matched_gap >= 8/15
overhead_ratio <= 2.5 for positive-control, <= 2.0 preferred
```

第三，Qpop 的贡献必须继续被验证。v23.01 的 Part D 顶层 qpop_best_minus_random 只是小正数，repair root 曾有更强正数，但不够稳。v23.02 必须同时报告 BlockSNR / degree-BlockSNR 与 random-matched gate、gate-density-matched random、same-compute no-op 的差异。如果 Qpop 不能稳定超过 random matched，则最终结论应写成 “FunctionalGram + trust 有效，Qpop contribution unproven”，不能强行宣称 population gate 成功。

第四，Part E official controls 必须补齐。v23.01 的复盘明确说当前 runner 验证了核心路径，但默认 Part E 仍是缩小版 D-CHE K9 depth3，尚未实现 official 要求的 MLP raw / MLP composite / full basis-depth 矩阵。v23.02 必须把这些 controls 接入 full positive-control，否则不能 promotion。

第五，real-task preflight 只能在 C2/F5 full positive-control official pass 之后运行。真实任务最低进展标准是打破历史边界，而不是立即 official：visual coverage 必须高于 0/18，no-debt 必须高于 0/30，beats_MLP_composite 必须高于 1/30。若 C2/F5 正式未过，real-task 禁止运行。

---

## 三、实验矩阵总览

v23.02 不再设计十几个互相嵌套的复杂 Part，而采用七个强约束 Part。每个 Part 都有多个方案，多个方案用于机制验证，不用于 runtime 选 winner。

Part A 做代码身份、数据身份、artifact hygiene 和 full controls readiness。Part B 锁定历史边界和 v23.01 当前边界。Part C 做 FunctionalGram 与 Qpop formation 的 taskwise official matrix。Part D 做 predictive trust 估计器的 oracle learning / calibration / ablation。Part E 把 predictive trust 接入 C2/F5 full positive-control。Part F 做 efficiency and failure decomposition，判断当前 safety 是否真正从 heavy rejection 变成 usable trust. Part G 只有在前面 gate 全过时做 limited real-task preflight。

所有 Part 都必须输出 `part_X_next_actions_for_codex.json`。该文件必须写：dominant_blocker、allowed_actions、forbidden_actions、max_repair_rounds、rerun_commands、evidence_to_record_next。Codex 不允许在没有 `next_actions` 的情况下继续修复。

---

# Part A：实现身份、control 完整性与 safety rollback smoke

Part A 的目标是确保 v23.02 没有在实现层面偷换问题。必须确认 FunctionalGram、Qpop、predictive trust、exact guard、MLP controls 和 same-compute controls 都是真实实现，而不是 placeholder 或 stub。

## A.1 代码与模型身份

Codex 首先运行 compile/import/static scan。必须检查这些模块：

```text
dgkan/fu/edge_functional_gram.py
dgkan/fu/finite_step_trust_region.py
dgkan/fu/predictive_trust_radius.py
dgkan/fu/trust_feature_extractor.py
dgkan/optim/functional_population_trust_flow.py
experiments/run_v23_02_predictive_trust_functional_population_flow.py
```

如果新模块尚未存在，Codex 应创建它们。`predictive_trust_radius.py` 要包含至少三个 trust predictor 方案：analytic quadratic predictor、online isotonic/logistic predictor、PID trust-radius controller。`trust_feature_extractor.py` 要统一产生所有 trust features，避免每个 scheme 重复写不同特征导致结果不可比。

Part A 必须记录：

```text
compile_pass
import_pass
static_scan_pass
true_depth2_purekan_constructed
true_depth3_purekan_constructed
edge_coefficients_changed
mlp_tensors_changed_in_primary
mlp_raw_control_available
mlp_composite_control_available
same_compute_control_available
random_matched_gate_control_available
functional_gram_s0_available
functional_gram_s1_diagnostic_available
qpop_block_gate_available
qpop_degree_block_gate_available
exact_finite_step_guard_available
predictive_trust_analytic_available
predictive_trust_online_available
predictive_trust_pid_available
runtime_selector_used
metric_winner_selection_used
candidate_update_selection_used
held_test_usage
used_fake_data_rows
```

通过标准是所有身份字段正确：no MLP stem/readout in primary, no external product feature, no new edge function, no runtime selection, all controls available.

## A.2 Finite-step rollback smoke

由于 v23.02 的核心是 trust，Part A 必须证明 rollback 真实可靠。构造一个 toy model 和 toy optimizer state，执行：

1. 保存模型参数和 optimizer state；
2. 做一个明确改变参数和 Adam moments 的 step；
3. 调用 restore；
4. 比较参数、moment、step count、gate EMA、trust predictor state 是否完全恢复。

记录：

```text
snapshot_restore_param_error
snapshot_restore_optimizer_error
snapshot_restore_gate_state_error
snapshot_restore_predictor_state_error
rollback_restores_rejected_candidate
accepted_candidate_keeps_state
```

通过标准：所有 restore error 小于 $10^{-12}$，accepted candidate 不被回滚，rejected candidate 必须回到 step 前状态。

## A.3 Control matrix readiness

v23.01 的一个限制是 Part E 默认矩阵仍为缩小版 D-CHE K9 depth3，没有 full basis-depth 和 MLP controls。v23.02 Part A 必须先确认后续 matrix 能生成：

```text
basis: dche_k5, dche_k9, dfour_default
depth: depth2, depth3
primary: E4/E5 FunctionalGram BlockSNR s0
controls: AdamW, FunctionalGram AdamW no-Qpop, RandomMatchedGate, same-compute no-op, MLP raw, MLP composite
safety: no-safety, exact finite-step, analytic trust, online trust, PID trust, random trust matched
```

如果任一 control 缺失，Part A 失败。Codex 不允许用“暂时先跑 D-CHE K9 depth3”通过 Part A。可以在 Part C 做 diagnostic reduced run，但 official Part E/F 必须 full controls ready.

---

# Part B：历史边界与 v23.01 当前边界锁定

Part B 要把当前路线的历史边界写成 hard lock，防止后续把 diagnostic pass 写成 official pass。

必须锁定以下事实。v22.94 已证明 AdamW witness 和 missing freedom decomposition 有效，ordinary repair 能打开 C2，但 F5 no-debt 最好只有 $2/15$，tailsafe full run 会杀 C2。v23.00R continuation 证明 dense functional $G_{edge}(s=0)$ + Qpop + finite-step acceptance 可以在 forced/sanity 条件下通过 F5，但 heavy rejection 是明确 caveat。v23.01 顶层结果为 Part E `E_C2FormationPass`、Part F `F_SafetyDiagnosticOnlyHeavyRejection`、Part G `G_PositiveControlFailed`，最终 `promotion_allowed=0`。v23.01 的 E4/E5 总体 C2 positive 主要由 rotation_sensitive 强阳性拉动，local_patch_interaction median 仍为负。v23.01 的 F5/no-debt 在 exact finite-step 下可闭合，但 accept rate 和 scale mean 太低，guard every 2/3 提高效率但 F5 失败。

Part B 必须输出：

```text
v22_94_witness_pass
v22_94_decomposition_pass
v22_94_ordinary_c2_opened
v22_94_F5_best
v22_94_tailsafe_kills_C2
v23_00R_functional_s0_positive
v23_00R_finite_step_guard_diagnostic_pass
v23_00R_accept_rate_caveat
v23_01_part_e_route
v23_01_part_f_route
v23_01_part_g_route
v23_01_final_promotion_allowed
v23_01_E4E5_C2_coverage
v23_01_E4E5_random_gap
v23_01_local_patch_median
v23_01_rotation_sensitive_median
v23_01_F2_accept_rate
v23_01_F2_scale_mean
v23_01_guard_every2_F5
v23_01_guard_every3_F5
```

Part B 通过标准是没有 missing 历史字段。若某个字段 artifact 缺失，Codex 必须读取对应 CSV/JSON 重新汇总，不能写 unknown 后继续。

---

# Part C：FunctionalGram + Qpop 的 taskwise C2 formation official audit

Part C 的目标是把 v23.01 的 C2 formation 从“合并 median 通过”升级成“taskwise 通过 + full controls 通过”。它不测试 safety，只测试 formation.

## C.1 核心假设

假设 C1：Functional $L^2$ edge Gram $s=0$ 是 formation phase 的有效 KAN curve geometry。

假设 C2：BlockSNR 和 degree-BlockSNR 是比 diagonal SNR 更合适的 population gate，因为 KAN edge modes 需要成组变化。

假设 C3：v23.01 的 local_patch negative median 不是结构性失败，而是 gate_floor / training horizon / block granularity 尚未合适。

## C.2 方案矩阵

Part C 必须同时跑以下 fixed schemes：

```text
C0_AdamW_control
C1_FunctionalGram_AdamW_s0_no_Qpop
C2_FunctionalGram_DiagonalSNR_s0
C3_FunctionalGram_BlockSNR_layer_s0
C4_FunctionalGram_BlockSNR_degree_s0
C5_FunctionalGram_BlockSNR_edgebank_s0
C6_FunctionalGram_BlockSNR_class_conditional_s0
C7_FunctionalGram_RandomMatchedGate_s0
C8_FunctionalGram_SameComputeNoOp_s0
C9_FunctionalGram_s1_derivative_diagnostic
```

其中 C9 不作为候选，只用于确认 derivative-heavy metric 是否仍压 formation。C7 的 random matched gate 必须匹配 gate density、block structure 和 update norm，不能只是随机 mask。

Basis/depth 矩阵：

```text
basis = dche_k5, dche_k9, dfour_default
depth = depth2, depth3
```

Task 矩阵：

```text
local_patch_interaction
rotation_sensitive
```

Seed：official C2 至少 15 seeds；若为了 debug 可先 5 seeds，但不得 promotion。

## C.3 实现细节

FunctionalGram $s=0$ 必须通过 quadrature 计算：

$$
G_{edge} = \int_{-1}^{1}\psi(u)\psi(u)^Tdu + \epsilon I.
$$

不允许退回 diagonal mode weight。每个 scheme 必须记录 `G_edge_type=functional_l2`、`functional_gram_condition`、`functional_gram_ridge`、`functional_gram_quadrature_points`。如果使用 trace normalization，也必须记录 normalization 前后的 condition 和 trace。

Qpop 必须在 FunctionalGram whitened coordinate 中计算。对 block $b$：

$$
\tilde g_a = G_{edge}^{-1/2}g_a,
$$

$$
\mathrm{SNR}_b = \frac{\|\mu_b\|^2}{\operatorname{tr}(\Sigma_b)/(n_b-1)+\epsilon}.
$$

Block gate 可写成：

$$
q_b = \sigma\left(\beta(\log \mathrm{SNR}_b - \tau)\right),
$$

并加 floor：

$$
q_b^{final}=q_{floor}+(1-q_{floor})q_b.
$$

v23.01 中 gate_floor=0.3 对 E4/E5 起了作用，因此 Part C 要比较 $q_{floor}\in\{0.2,0.3,0.4\}$，但这个比较只能在 fixed scheme 维度做，不能按 seed 选 winner。

## C.4 必须记录的字段

每一行必须记录：

```text
basis_key
depth
task
seed
scheme
train_steps
gate_floor
stat_warmup_steps
population_grad_examples
C2_accuracy_initial
C2_accuracy_final
C2_accuracy_improvement
C2_coverage_initial
C2_coverage_final
C2_coverage_improvement
gate_density_mean
gate_density_median
gate_entropy
block_snr_mean
block_snr_top_decile
random_matched_gate_density
gate_norm_ratio_vs_random
functional_gram_condition
functional_gram_trace
mode_energy_low_mid_high
source_guard_C2_coverage
source_guard_C2_accuracy
wall_time_ratio
overhead_ratio
used_fake_data_rows
held_test_usage
```

Merge summary 必须按 overall、taskwise、basiswise、depthwise 分组输出。

## C.5 通过标准

Part C official pass 必须满足以下条件：

```text
E4 or E5 fixed scheme on dche_k9/depth3:
  both tasks taskwise C2_coverage_improvement_median >= 0.03
  both tasks taskwise C2_accuracy_improvement_median >= 0.20
  overall C2_coverage_improvement_median >= 0.08
  random_gap overall >= 0.04
  random_gap taskwise >= 0.02
  gate_density_median in [0.25, 0.85]
  overhead_ratio <= 2.5
  used_fake_data_rows = 0
  held_test_usage = 0
```

D-CHE K9 depth3 是主 carrier，但 Part C 也要报告 D-CHE K5 和 D-FOUR。如果 D-CHE K9 过而其他 basis 失败，可以继续，但必须记录 basis-specific conclusion。若只有 rotation_sensitive 过而 local_patch 不过，则 Part C 不能 official pass，只能 diagnostic pass。

## C.6 失败时 Codex 允许探索

如果 local_patch 仍失败，Codex 允许优先尝试：

```text
gate_floor = 0.35 or 0.4
train_steps = 200 or 240
population_grad_examples = 8
block definition = edgebank or degree+edgebank hybrid
stat_warmup_steps = 20
width/depth only if identity remains true PureKAN
```

禁止：

```text
新增 patch product feature
新增 MLP stem/readout
删除 rotation_sensitive 或只报告 overall median
按 seed 选择 passing rows
降低 taskwise gate
```

如果 random matched gate 同样强，Codex 必须改进 random matched control 的 density/norm matching audit，不能删 random matched。

---

# Part D：Qpop 贡献与 gate 结构审计

Part D 的目标是回答：C2 formation 是 FunctionalGram 本身造成的，还是 Qpop 真有贡献？v23.01 中 Qpop 顶层只比 random 多约 0.028，repair root 曾多 0.18，说明贡献存在但不稳。Part D 要把这个问题做 solid。

## D.1 方案

D 只使用 Part C 中 C2 taskwise 表现合格或接近合格的 carrier，优先 D-CHE K9 depth3。比较以下 gates：

```text
D0_no_gate_FunctionalGramAdamW
D1_random_density_matched
D2_random_density_norm_matched
D3_diagonal_SNR
D4_layer_block_SNR
D5_degree_block_SNR
D6_edgebank_block_SNR
D7_class_conditional_block_SNR
D8_lowrank_block_AB_sketch_rank4
D9_lowrank_block_AB_sketch_rank8
D10_negative_control_adjusted_SNR
```

D8/D9 必须做 layer/block low-rank，不允许全参数 eigendecomposition。v23.00R 的 full-parameter eigendecomposition 曾卡住，因此这是硬约束。

## D.2 记录字段

```text
gate_type
gate_density
gate_norm_ratio
C2_coverage_improvement
C2_accuracy_improvement
random_gap
negative_label_update_mass
MLP_friendly_update_mass
source_shuffle_update_mass
snr_C2_gain_correlation
source_guard_gate_cosine
gate_seed_stability
block_snr_variance
lowrank_runtime
lowrank_memory_peak
```

## D.3 通过标准

Part D 通过不要求所有 gates 都成功，但至少一个非-random Qpop gate 必须满足：

```text
C2_coverage_median - best_random_matched_coverage_median >= 0.05
C2_accuracy_median - best_random_matched_accuracy_median >= 0.05
source_guard_gate_cosine >= 0.70
gate_seed_stability >= 0.65
overhead_ratio <= 2.5
```

如果 Part C 已过但 Part D 不过，则最终解释要写成：FunctionalGram 可能是主因，Qpop contribution unproven。此时仍可进入 Part E 的 trust study，但不能把 population gate 写成成功机制。

## D.4 Codex 修复

如果 Qpop 被 random matched 打平，Codex 可以尝试 block granularity、EMA beta、gate floor、cohort split、class-conditional SNR。不能降低 random gate 匹配强度，也不能删除 random controls。如果 low-rank 卡住，只能做 layer/block sketch，不可全参数 eigendecomposition。

---

# Part E：Predictive trust radius 建模

Part E 是 v23.02 的核心。它不直接跑 official safety，而是学习或估计一个 $\widehat{\alpha}$，用于在 exact finite-step guard 前预测安全尺度。

## E.1 数据生成

使用 Part C/D 通过或接近通过的 C2 formation schemes，主要是：

```text
E4_FunctionalGram_BlockSNR_s0
E5_FunctionalGram_BlockSNR_degree_s0
```

在 training loop 中，每隔 $k$ 步记录一个 candidate update $\Delta\theta_t$。对每个 candidate，用 train-only guard split 评估离散尺度：

$$
\alpha\in\{1,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0\}.
$$

定义 oracle safe scale：

$$
\alpha^*_t = \max\{\alpha:\Delta D_c(\theta_t+\alpha\Delta\theta_t)\le 0,\forall c\}.
$$

必须记录所有 candidate 的 features、oracle scale、accept/reject reason。这个阶段不 promotion，不训练主模型，只生成 trust dataset。

## E.2 Trust features

每个 candidate 必须记录：

```text
step
scheme
task
seed
current_NLL
current_accuracy
current_Brier
current_ECE
current_tail95
current_tail99
current_margin10
update_norm_L2
update_norm_Gedge
update_norm_per_layer
qpop_gate_density
qpop_gate_entropy
block_snr_mean
block_snr_top_decile
predicted_task_descent_linear
predicted_debt_delta_linear_Brier/ECE/tail95/tail99/margin10
predicted_debt_delta_quadratic_Brier/ECE/tail95/tail99/margin10
debt_gradient_cos_task_gradient_per_component
recent_accept_rate_ema
recent_scale_mean_ema
recent_reject_reason_histogram
functional_gram_condition
mode_energy_low_mid_high
oracle_alpha_star
oracle_reject_reasons
oracle_safe_component_vector
```

Quadratic debt prediction 可用低成本 directional finite difference：

$$
\widehat{\Delta D_c}(\alpha)
= a_c\alpha + b_c\alpha^2,
$$

其中 $a_c$ 来自 gradient directional derivative，$b_c$ 可由一个 small probe scale $\alpha_0$ 拟合：

$$
b_c = \frac{D_c(\theta+\alpha_0\Delta\theta)-D_c(\theta)-a_c\alpha_0}{\alpha_0^2+\epsilon}.
$$

## E.3 多方案 trust predictor

Part E 至少实现四类 predictor。

### E-A Analytic quadratic trust

对每个 debt component 解：

$$
a_c\alpha + b_c\alpha^2 \le 0.
$$

取所有 component 的最大安全交集：

$$
\widehat\alpha = \min_c \alpha_c^{safe}.
$$

若 $b_c$ 不稳定，采用 clipping：

$$
b_c\leftarrow \operatorname{clip}(b_c,b_{min},b_{max}).
$$

### E-B Online isotonic / logistic trust

训练一个 train-only small predictor，不使用 held/test。输入为上面的 features，目标为 oracle accept label 或 oracle scale bucket。可选模型：

```text
isotonic regression on risk_score
logistic regression on handcrafted features
small monotone MLP prohibited unless fully diagnostic; official uses linear/logistic only
```

risk score 可写成：

$$
r_t = \max_c \widehat{\Delta D_c}(1) + \lambda\|\Delta\theta_t\|_{Gedge}.
$$

predictor 输出安全概率 $p_t(\alpha)$，选择最大 $\alpha$ 使 $p_t(\alpha)\ge p_0$。

### E-C PID / adaptive trust radius

不训练 predictor，只维护全局或 per-scheme trust radius $r_t$：

$$
\log r_{t+1}=\log r_t+\eta_p(a_t-a^*)-\eta_i\sum_{\tau\le t}(a^*-a_\tau),
$$

其中 $a_t$ 是近期 accept indicator，$a^*$ 是目标 accept rate，例如 0.35。candidate scale 为：

$$
\widehat\alpha_t=\min(1,r_t/\|\Delta\theta_t\|_{Gedge}).
$$

这是一条非常简单的 control baseline。如果它打过复杂 predictor，说明不需要复杂安全模型。

### E-D Component-risk trust with persistent guard

维持一个固定 train-only persistent guard split。对每个 component 维护 recent violation EMA：

$$
v_{c,t}=\rho v_{c,t-1}+(1-\rho)\max(0,\Delta D_{c,t}).
$$

scale rule：

$$
\widehat\alpha_t = \frac{\alpha_0}{1+\sum_c \lambda_c v_{c,t}}.
$$

这个方案测试是否可以用非常简单的 component risk state 替代 exact per-step line-search。

## E.4 通过标准

Part E 不是最终 safety pass，它验证 predictor 是否值得接入训练。通过标准：

```text
oracle_scale_prediction_spearman >= 0.50
safe_accept_precision >= 0.90
unsafe_recall >= 0.80
median_predicted_scale / median_oracle_scale in [0.5, 1.5]
predictor_overhead <= 0.25 * training_step_time
held_test_usage = 0
```

对每个 predictor 都要记录 calibration curve、false-safe rows、over-conservative rows。若 predictor precision 低，不能进入 Part F。若 predictor 过度保守，表现为 median predicted scale 接近 0，也不能进入 Part F。

## E.5 Codex 修复

如果 analytic quadratic 不准，Codex 可以调整 probe scale、b clipping、component-specific scale。若 online predictor 过拟合，Codex 必须增加 source/guard split、cross-fit trust dataset，不得读 held/test。若 PID trust 震荡，Codex 可以调 $a^*$、$\eta_p$、$\eta_i$，但必须记录 stability。

---

# Part F：Predictive trust 接入 C2/F5 full positive-control

Part F 把 Part E 通过的 trust predictor 接入训练。它是 v23.02 的核心 official gate。

## F.1 Scheme matrix

固定 formation base：

```text
E4_FunctionalGram_BlockSNR_s0
E5_FunctionalGram_BlockSNR_degree_s0
```

Safety / trust schemes：

```text
F0_no_safety_control
F1_component_conflict_veto
F2_exact_finite_step_guard_all
F3_analytic_quadratic_predictive_trust
F4_online_logistic_predictive_trust
F5_PID_adaptive_trust_radius
F6_component_risk_persistent_guard_trust
F7_predictive_trust_plus_exact_fallback
F8_random_trust_matched_control
```

F7 是实际最可能成功的折中：先用 predictor 给 $\widehat\alpha$，然后只在 high-risk 或周期性 steps 上用 exact guard 校验。它必须记录 exact fallback frequency。

## F.2 必须记录字段

```text
scheme
safety_type
task
seed
C2_coverage_no_safety
C2_coverage_with_safety
C2_coverage_retention_vs_F0
C2_accuracy_retention_vs_F0
F5_no_debt
F5_no_debt_count
component_non_positive_Brier/ECE/tail95/tail99/margin10
finite_step_accept_rate
finite_step_scale_mean
finite_step_skip_count
predictor_scale_mean
predictor_scale_median
oracle_check_frequency
exact_guard_fallback_count
false_safe_count
false_reject_count
reject_reason_histogram
random_veto_matched_gap
overhead_ratio
wall_time_ratio
trust_model_overhead
used_fake_data_rows
held_test_usage
```

## F.3 Official pass 标准

对至少一个 fixed scheme group，必须同时满足：

```text
basis=dche_k9, depth=depth3
both tasks C2 formation already passed in Part C
F5_no_debt_count >= 12/15
component_non_positive_rows >= 12/15
C2_coverage_retention_vs_F0 >= 0.70
C2_accuracy_retention_vs_F0 >= 0.80
finite_step_accept_rate_median >= 0.20
finite_step_scale_mean_median >= 0.05
finite_step_skip_count_median <= 0.70 * total_steps
random_veto_matched_gap >= 8/15
overhead_ratio <= 2.5
held_test_usage=0
```

如果 F2 exact guard passes F5 but fails accept/scale，必须标记 diagnostic only，不得 official。若 F7 predictor+fallback 通过，即使 exact guard F2 是 diagnostic，也可进入 Part G。

## F.4 Codex 修复

若 F5 不过，Codex 只能从 Part E predictor 特征、component-wise trust、guard split 稳定性入手，不得降低 no-debt gate。若 accept/scale 不过但 F5 过，Codex 必须提高 predictor scale 或减少 exact guard rejection，不能用更低 edge_lr 假装提升安全。若 C2 retention 不过，Codex 应检查 predictor 是否过度保守，或是否 component risk 把 task-relevant blocks全压掉。

---

# Part G：Full control positive-control matrix

Part G 在 Part C/F 通过后运行。它补齐 v23.01 缺失的 official controls。

## G.1 必须包含的模型和 controls

```text
Primary:
  FunctionalGram BlockSNR + predictive trust scheme from Part F

Controls:
  KAN AdamW
  FunctionalGram AdamW no Qpop
  FunctionalGram BlockSNR no safety
  FunctionalGram RandomMatchedGate + same trust
  Same-compute no-op
  Same-gate-density random
  MLP raw
  MLP composite-coordinate
  MLP same-parameter budget if available
```

Basis/depth：

```text
dche_k5 depth2/depth3
dche_k9 depth2/depth3
dfour_default depth2/depth3
```

Primary success 可集中在 dche_k9 depth3，但 controls 必须完整记录。

## G.2 通过标准

```text
primary beats KAN AdamW in C2 coverage median
primary beats FunctionalGram AdamW no-Qpop in C2/F5 joint score
primary beats RandomMatchedGate in C2/F5 joint score
primary does not lose to MLP raw/composite on positive-control teacher rows
F5 official pass remains true
no fake data, no held/test, no runtime selector
```

Joint score 定义为：

$$
J = \Delta\mathrm{C2Coverage} + 0.5\Delta\mathrm{C2Accuracy} + 0.2\mathbf{1}_{F5} - 0.1\mathrm{OverheadPenalty}.
$$

这个 score 只用于 report，不作为 runtime winner selection。所有 gate 仍以绝对指标为准。

---

# Part H：Efficiency and trust predictor decomposition

Part H 不再是可选复盘，而是 official 前必须解释 trust 的效率来源。

必须回答：

```text
predictive trust 是否真正减少 exact guard 的工作量？
accept/scale 提升来自 predictor 还是来自降低 step size？
哪些 debt component 仍是主要 reject reason？
F5 成功是否依赖少数 seed 或少数 task？
local_patch 与 rotation_sensitive 的 trust 行为是否不同？
```

记录：

```text
trust_efficiency_gain_vs_exact_guard
trust_false_safe_rate
trust_false_reject_rate
trust_calibration_ECE
scale_distribution_by_task
reject_reason_distribution_by_task
accept_rate_by_seed
scale_mean_by_seed
debt_component_violation_histogram
```

Part H 若发现 success 只来自低 edge_lr、低 scale 或少数 seed，必须把 Part G 降级为 diagnostic pass，不允许 real-task。

---

# Part I：Limited real-task preflight

只有 Part G/H official pass 后运行。任务：

```text
MNIST
FashionMNIST
KMNIST
Wine
Spam
```

矩阵必须保留：

```text
Primary v23.02 fixed scheme
KAN AdamW
FunctionalGram AdamW
RandomMatchedGate
MLP raw
MLP composite
same-compute no-op
```

记录：

```text
dataset
seed
final_NLL
accuracy
ECE
Brier
tail95
tail99
margin10
no_debt
visual_coverage
tabular_coverage
source_guard
beats_AdamW
beats_MLP_raw
beats_MLP_composite
beats_random_matched
overhead
trust_accept_rate
trust_scale_mean
trust_skip_count
```

最低进展标准：

```text
visual_coverage > historical 0/18
no_debt > historical 0/30
beats_MLP_composite > historical 1/30
trust_accept_rate_median >= 0.15
```

如果这些都没有改善，结论是 positive-control 仍未迁移。不能进入 full-loop。

---

# Part J：Failure decomposition and final route

Part J 生成最终报告。必须按以下 route 写结果：

```text
A_or_B_identity_failed
C_C2FormationTaskwiseFailed
D_QpopContributionUnproven
E_TrustPredictorUncalibrated
F_SafetyEnvelopeFailed
F_SafetyDiagnosticOnlyHeavyRejection
G_PositiveControlFailed
H_TrustEfficiencyFailed
I_RealTaskPreflightFailed
I_RealTaskPreflightProgress
OfficialCandidatePass
```

Final route 不允许手写成功。必须从 artifacts 聚合。

---

## Codex 自动探索总规则

每个失败必须生成 `part_X_next_actions_for_codex.json`。以下是全局 repair cookbook。

如果 Part C local_patch 不过，优先：gate_floor、train_steps、block granularity、population_grad_examples、stat_warmup。禁止新增 feature 或删除 local_patch。

如果 Part D Qpop 不过，优先：block definition、EMA、class-conditional SNR、low-rank layer sketch。禁止降低 random matched control。

如果 Part E trust predictor 不准，优先：quadratic probe scale、component-specific clipping、cross-fit trust dataset、PID target accept rate。禁止读取 held/test。

如果 Part F F5 不过，优先：predictive scale calibration、component-risk state、persistent guard split。禁止降低 no-debt gate。

如果 Part F F5 过但 heavy rejection，优先：提高 predictor scale、exact fallback cadence、trust-radius warmup。禁止用小 lr 假装 efficiency。

如果 Part G controls 不过，不能删除 controls；必须写明是 MLP-dominance、random-gate dominance、same-compute dominance 还是 overhead dominance。

如果 Part I real-task 不过，不能回头把 positive-control写成 official；必须写迁移失败。

---

## 预期结论形式

v23.02 最理想的结果是：

$$
\boxed{
\text{FunctionalGram }s=0 + \text{BlockSNR} + \text{predictive trust}
\text{ 在 C2/F5 上 official pass，且 trust efficiency 达标。}
}
$$

次优但仍有价值的结果是：C2 taskwise pass，F5 仍 heavy rejection。这说明 trust predictor未成，路线继续卡 safety efficiency。

负结果也必须清楚。若 C2 taskwise不过，说明 v23.01 的 C2 pass 是合并 median 假阳性。若 Qpop 不过，说明 FunctionalGram 是主因，population gate仍不可靠。若 predictive trust 不过，说明 F5/no-debt 是需要 exact finite-step oracle 的强非线性约束。若 real-task 不迁移，说明 positive-control机制仍不足以支撑真实任务。

这版计划的核心不是再写复杂工程，而是把当前真正有效的东西收敛成可验证算法：

$$
\boxed{
G_{edge}^{L2}
+
Q_{pop}^{block}
+
\widehat\alpha_{trust}
}
$$

它要回答的只有三个问题：

1. Functional $L^2$ edge geometry 是否能稳定形成 C2？
2. Block-level population gate 是否真的优于 random matched gate？
3. Predictive trust 是否能在不 heavy rejection 的情况下闭合 F5/no-debt？

这三个问题回答清楚，项目才会从 diagnostic 可行性进入真正 candidate 阶段。
