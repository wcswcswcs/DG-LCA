# DG-KAN v22.13 独立审计总结与 v22.14 完整新计划

> 版本：independent audit + v22.14 execution plan  
> 生成日期：2026-06-08  
> 目标：独立复核 v22.13 代码与结果，不盲信复盘；同时给出下一版确实推进 **basis efficiency** 与 **functional update algorithm** 的完整实验计划。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 固定硬约束：strict FC-PureKAN；no active B-spline official path；no teacher；no distillation；no sampler/class weight；no dataset-name branch；no seed-specific rule；no validation/test/future/query for direction；LineC/ECE/Brier/AUCtime/task metrics 只能 readback/gate，不能生成 direction。  

---

# 0. 我对 v22.13 的总判断

v22.13 是到目前为止最重要的一版之一，但我不会把它写成完整 scientific success。我的独立判断是：

$$
\boxed{
\text{v22.13 在 efficiency 和 loss-interface operator 上有实质突破，}
\text{但 KAN carrier 没闭合，且 operator horizon 的 official 口径需要降级。}
}
$$

更具体地说：

```text
真实进展：
  1. 代码包基本自包含，required source 23/23 存在，clean compile/import 通过。
  2. D-FOU 与 D-CHE 已经从 v22.12 的 manual upstream-VJP 推进到 native arbitrary-cotangent fused VJP。
  3. official_fused_kernel_complete_rows 从 v22.12 的 0 推到 v22.13 的 84。
  4. FU 从 O10 role-aware consensus 降级后，真正使用了 role-blind operator core。
  5. S5 horizon 首次在 CE / MSE / Ranking / Preference-smoke 上达到 C3=4，并在 CE / Ranking / Preference-smoke 上达到 C4=3。
  6. StableRandom / RandomMatched 没有通过，说明本轮不是 random control 同步开闸。
```

但也有几个不能忽略的阻塞：

```text
主要阻塞：
  1. official KAN carrier promotion = 0，KAN_source_channel_pass_rows = 0。
  2. D-CHE / D-FOU readout-only rows 有弱 source，但 basis_channel_energy = 0；basis_linearized_w1 rows 有 basis energy，但 projection residual 约 0.998，source 不稳。
  3. KAN_specific_delta_vs_MLP_same_metric 全部为负，约 -0.40 到 -0.45；KAN 没有超过 MLP same-operator baseline。
  4. task readback 中 KAN+FU 明显改善 KAN+AdamW，但远没超过 same-param MLP；KAN+FU accuracy >= MLP 只有 1/9。
  5. S5 official horizon pass 依赖 `low_lr_source_anchor_w0p10_lr0p02`，也就是 horizon training loop 中加入了 source-anchor retention loss；这更像“operator + train-only anchor/proximal dynamics”，不能直接写成纯 operator one-shot FU 成功。
  6. 结果包里的 `v22_13_idle_violation.csv` 明确写 `execution_contract_violation = 1`，而 v22.13 计划规定 execution contract violation 时 `official_promotion_allowed = 0`。finalizer 没把这个 blocker 纳入 official route。
  7. `v22_13_source_state_dynamics.csv` 实际为空文件，这说明 source-state dynamics artifact 没有按计划有效落盘。
```

因此我建议把 v22.13 的 route 从复盘中的：

```text
R9-TrueLossInterfaceOperatorExplorationSuccess_KANPending
official_promotion_allowed = 1
```

独立改写为：

```text
R9a-RoleBlindOperatorAnchorExplorationSuccess_EfficiencyOpened_KANBasisCarrierNoGo
exploration_promotion_allowed = 1
official_operator_promotion_allowed_strict = 0
official_kan_carrier_promotion_allowed = 0
scientific_claim_allowed = 0
```

这不是否定 v22.13。相反，v22.13 的价值很大：它证明我们已经从 v22.12 的 single-displacement / role-aware scaffold，推进到了一个真正可执行的 `cotangent -> functional displacement` operator pipeline，并且 kernel-native arbitrary-cotangent efficiency 也终于打开了。但它也暴露出下一步真正的问题：

$$
\boxed{
\text{我们现在不是缺一个更强 readout replay，}
\text{而是缺一个能把 } T_\theta(\delta) \text{ 写进 KAN basis-state channel 的算法。}
}
$$

---

# 1. 独立审计方法

我没有直接接受复盘结论，而是做了以下独立复核：

```text
1. 解包：
   /mnt/data/v22_13_code_review_packet.zip
   /mnt/data/v22_13_results_bundle.zip

2. 统计 zip 内容：
   code_review_packet entries = 287
   results_bundle entries = 287

3. 在 clean extracted source 上执行：
   python -m compileall -q dgkan experiments

4. 手动 import 核心模块：
   dgkan.fu.operator_atoms_v22_13
   dgkan.fu.operator_core
   dgkan.fu.operator_horizon
   dgkan.profiling.efficiency_v22_13
   dgkan.kernels.dfou_native_cotangent
   dgkan.kernels.dche_native_cotangent
   dgkan.kan.corrected_readout_solve
   experiments.run_v22_13_operator_construct
   experiments.run_v22_13_efficiency_native
   experiments.run_v22_13_kan_corrected_mapping
   experiments.run_v22_13_finalize

5. 对照 v22.13 计划中 required source list，逐一确认文件存在。

6. 手工阅读并审计关键源码：
   dgkan/fu/operator_core.py
   dgkan/fu/operator_atoms_v22_13.py
   dgkan/fu/control_nullspace.py
   dgkan/fu/metric_geometry.py
   dgkan/fu/operator_commit.py
   experiments/run_v22_13_operator_horizon.py
   dgkan/profiling/efficiency_v22_13.py
   dgkan/kernels/dfou_native_cotangent.py
   dgkan/kernels/dche_native_cotangent.py
   dgkan/kernels/fused_fourier_k2.py
   dgkan/kernels/fused_chebyshev_k3.py
   experiments/run_v22_13_kan_corrected_mapping.py
   dgkan/kan/corrected_readout_solve.py
   dgkan/kan/layout_contracts.py
   experiments/run_v22_13_finalize.py

7. 独立聚合 CSV / JSON：
   v22_13_code_truth_gate.csv
   v22_13_required_source_files.csv
   v22_13_packet_required_compare.csv
   v22_13_semantic_firewall.csv
   v22_13_operator_law_tests.csv
   v22_13_adapter_renaming_tests.csv
   v22_13_native_efficiency_truth_table.csv
   v22_13_official_fused_status_matrix.csv
   v22_13_adapter_horizon_matrix.csv
   v22_13_KAN_mapping_matrix.csv
   v22_13_K17_basis_channel_commit.csv
   v22_13_task_eval_matrix.csv
   v22_13_idle_violation.csv
```

我把复盘当作提示，而不是证据本身。以下结论来自源码、CSV、JSON、artifact 与我自己的聚合。

---

# 2. Part A：代码审计结论

## 2.1 文件完整性

v22.13 计划要求至少包含 23 个核心源码文件。独立检查结果：

```text
required_source_files_present = 23/23
csv_claimed_exists_but_zip_missing_count = 0
compileall_status = 0
core_import_fail_count = 0
```

必须包含的关键文件都在：

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
experiments/run_v22_13_common.py
experiments/run_v22_13_s0_truth_gate.py
experiments/run_v22_13_operator_semantic_tests.py
experiments/run_v22_13_operator_construct.py
experiments/run_v22_13_operator_horizon.py
experiments/run_v22_13_kan_corrected_mapping.py
experiments/run_v22_13_efficiency_native.py
experiments/run_v22_13_task_eval.py
experiments/run_v22_13_finalize.py
```

因此这轮不是 v22.11/v22.12 早期那种“结果表有，源码缺”问题。代码包可以被当作有效实验包审计。

## 2.2 与计划要求的一致性

大体一致，但不是完全一致。

满足的部分：

```text
1. role-blind operator core 存在。
2. O10 没有继续计入 official operator path。
3. adapter renaming tests 6/6 pass。
4. operator law tests 6/6 pass，但这些 operator 被正确标记为 nonlinear norm-capped operator，不再伪称线性。
5. corrected KAN layout tests D-CHE / D-FOU 都 pass。
6. D-FOU / D-CHE native arbitrary-cotangent fused path 存在 `backward_from_grad_logits`。
7. finalizer 区分了 exploration / official operator / official KAN / scientific claim。
```

不完全一致或需要修正的部分：

```text
1. v22_13_idle_violation.csv 标记 execution_contract_violation = 1。
   按计划，这应该强制 official_promotion_allowed = 0。
   但 finalizer 的 _route_decision 没读取 idle_violation，也没有把 queue contract 加入 official route。

2. v22_13_source_state_dynamics.csv 是空文件。
   原因是 operator horizon row 没有把 `_attempts()` 中的 `source_state_attempt` 字段传播到 row，导致 source-state dynamics artifact 没有有效记录。

3. 计划中要求 `v22_13_compileall_report.csv` 和 `v22_13_import_closure.csv`；实际落盘是 `.log` 文件。
   这不影响 compile/import 事实，但 artifact naming 与计划不完全一致。

4. results bundle 解包后没有包含 nested `v22_13_code_review_packet.zip` 与 `v22_13_results_bundle.zip` 本体；artifact index 中宣称 repo official dir 里存在这些 zip。
   因为用户已单独上传 code packet 与 results bundle，所以这不妨碍本次审计；但不能把 results bundle 自身写成完全自封闭 artifact closure。

5. S5 horizon pass 依赖 source-anchor retention loss。这个机制在源码中是：
   `loss = loss + retention_weight * F.mse_loss(displacement_now, source_anchor)`。
   这应被明确标记为 auxiliary retention loss 或 proximal dynamics，而不是纯 one-shot functional update。
```

## 2.3 Role-blind operator core 是否正确？

我独立阅读了 `dgkan/fu/operator_core.py`。它的核心函数是：

```text
apply_operator(operator_id, logits, cotangent, source_state=None, norm_scale=..., seed=...)
```

它只消费 `cotangent`，不读取 adapter name、dataset name、labels、validation/test/future/query。LIO1/LIO2/LIO3/LIO5/LIO6/LIO7 的逻辑都是对 `descent = -cotangent` 做低曲率平滑、control-null projection、source-state mixing 或 row-mean boundary 修正。官方 core 没有 CE/MSE/Ranking 分支。

我手工 grep official core path 后，看到的 `adapter_name` 只出现在 telemetry 字段名 `uses_adapter_name_for_direction`，不是用于分支；`ranking` 只在注释里说明 row-axis control 可能会抹掉 ranking cotangent，也不是公式分支。因此 role-blind core 这点基本成立。

但是要注意，`run_v22_13_operator_horizon.py` 会根据每个 adapter 重新构造 `delta = adapter.cotangent(...)`，然后使用同一个 operator_id 生成不同的 `T(delta)`。这符合 loss-interface operator 定义。也就是说，v22.13 不是要求不同 adapter 输出同一个方向，而是要求同一个 operator law 作用在不同 cotangent 上。

更准确的数学表述是：

$$
\Delta f_a = T_\theta(\delta_a),
\qquad
\delta_a = \frac{\partial \mathcal{L}_a}{\partial f_\theta(x)}.
$$

这条证据链比 v22.12 的 O10 role-aware consensus 干净。

## 2.4 Operator law 的真实性

v22.13 的 operator law tests 显示：

```text
operator_linearity_error ≈ 0.30
operator_homogeneity_error = 0.5
cotangent_permutation_equivariance_error ≈ 0
operator_law_pass = 1
```

表面上 linearity/homogeneity 很差，但源码里把这些 operator 标记成：

```text
nonlinear_norm_capped_operator
```

这是合理的。因为所有 operator 最后都经过：

$$
\text{normalize\_update}(x) = x \cdot \frac{s\sqrt{n}}{\|x\|}
$$

所以它们本来就不是线性 operator。v22.13 在这一点上比 v22.12 更诚实：没有把 norm-capped nonlinear operator 伪装成线性算子。

不过这也意味着下一版不能再用“operator linearity”作为主要理论 claim。更正确的 claim 应该是：

$$
\boxed{
T_\theta \text{ 是 role-blind、permutation-equivariant、stable、norm-controlled 的 nonlinear cotangent preconditioner。}
}
$$

而不是：

$$
T_\theta \text{ 是线性 functional operator。}
$$

## 2.5 Corrected KAN layout 是否锁住？

是。v22.13 的 layout test 结果非常清楚：

```text
D-CHE legacy_layout_fit_cosine ≈ -0.075
D-CHE legacy_layout_projection_residual ≈ 1.342
D-CHE corrected_layout_fit_cosine ≈ 0.99996
D-CHE corrected_layout_projection_residual ≈ 0.00927

D-FOU legacy_layout_fit_cosine ≈ 0.529
D-FOU legacy_layout_projection_residual ≈ 0.883
D-FOU corrected_layout_fit_cosine ≈ 0.99966
D-FOU corrected_layout_projection_residual ≈ 0.0262
```

这说明 v22.12 K14 发现的 layout bug 不是小问题。它确实影响 D-CHE，也影响 D-FOU 的旧 readout solve。v22.13 已经把这个问题转成单元测试，是正确的。

但注意：layout 修复只证明 readout target 可以被正确写入，不证明 KAN basis-channel source 成功。v22.13 后面的 KAN no-go 说明问题已经从“layout 错”升级为“basis carrier objective 不对”。

## 2.6 代码审计最终结论

代码审计结论：

```text
S0 code truth: pass
source completeness: pass
compile/import closure: pass
role-blind core: pass
corrected layout truth: pass
promotion semantics split: partially pass
artifact/queue contract: blocked for official promotion
source-state dynamics artifact: incomplete
```

因此我不会说 v22.13 代码无效。它是有效实验包。但我会修正 promotion 口径：

```text
exploration_promotion_allowed = 1
official_operator_promotion_allowed_strict = 0
reason:
  execution_contract_violation = 1;
  source-anchor retention loss counted in horizon success;
  source_state_dynamics artifact empty;
  KAN carrier not closed.
```

---

# 3. Part B：基函数效率路线审计

## 3.1 v22.13 是否优化了基函数效率？

是，而且这是本轮最硬的系统进展。

v22.12 的核心 blocker 是：

```text
official_fused_kernel_complete_rows = 0
manual_upstream_vjp_rows = 360
```

v22.13 推进到：

```text
official_fused_kernel_complete_rows = 84
manual_upstream_vjp_rows = 0
D-FOU official fused rows = 36
D-CHE official fused rows = 48
D-RBF official fused rows = 0
D-RAT official fused rows = 0
```

这说明 D-FOU / D-CHE 终于不只是 manual upstream-VJP profiler，而是有了真正的 arbitrary-cotangent fused VJP path。

源码证据也支持这一点：

```text
dgkan/kernels/fused_fourier_k2.py:
  backward_from_grad_logits(model, x, grad_logits, logits, h)

dgkan/kernels/fused_chebyshev_k3.py:
  backward_from_grad_logits(model, x, grad_logits, logits, h, use_grad_buffers=...)
```

这些函数不再只接受 supervised labels，而是接受任意 `grad_logits`，并调用 Triton kernels 计算 `w1` / `w2` gradients。v22.13 的 native gradcheck 也是 native-vs-autograd relative error，而不是只看 CE label backward。

因此我同意：

$$
\boxed{
\text{D-FOU / D-CHE arbitrary-cotangent efficiency 已经从 manual exploration 进入 native official evidence 阶段。}
}
$$

## 3.2 当前表现如何？

官方 summary：

```text
D-FOU:
  profile_rows = 36
  native_E1_pass_rows = 35
  official_fused_kernel_complete_rows = 36
  manual_upstream_vjp_rows = 0
  best_operator_step_ratio = 0.872
  best_full_step_ratio = 0.676
  best_memory_ratio = 0.989
  decision = OfficialNativeRowsPresent

D-CHE:
  profile_rows = 48
  native_E1_pass_rows = 48
  official_fused_kernel_complete_rows = 48
  manual_upstream_vjp_rows = 0
  best_operator_step_ratio = 0.856
  best_full_step_ratio = 0.700
  best_memory_ratio = 0.989
  decision = OfficialNativeRowsPresent

D-RBF:
  official_fused_kernel_complete_rows = 0
  blocker = arbitrary_cotangent_fused_backward_contract_missing

D-RAT:
  official_fused_kernel_complete_rows = 0
  blocker = arbitrary_cotangent_fused_backward_contract_missing
```

独立聚合后我会补充一个更细的判断：

```text
D-FOU:
  median operator_step_ratio ≈ 1.095
  median full_step_ratio ≈ 0.769
  median memory_ratio ≈ 0.995
  但 R1 lowfreq-bandreadout-native-vjp 在 Delta-Gaussian 上有明显 outlier：
    batch256 forward_ratio ≈ 5.53, operator_step_ratio ≈ 3.07
    batch512 forward_ratio ≈ 5.03, operator_step_ratio ≈ 1.34

D-CHE:
  median operator_step_ratio ≈ 1.073
  median full_step_ratio ≈ 0.765
  median memory_ratio ≈ 0.995
  但 R1 k3-corrected-layout-native-vjp 在 Delta-Gaussian 上也有 outlier：
    batch256 operator_step_ratio ≈ 1.58
    batch512 operator_step_ratio ≈ 1.44

D-RBF:
  forward ratio 本身不算非常差，但 VJP ratio 约 1.26 到 3.73，且没有 arbitrary-cotangent fused backward contract。

D-RAT:
  forward ratio 与 VJP 都偏慢，VJP ratio 可到 4.4+，且没有 arbitrary-cotangent fused backward contract。
```

因此不能把 “D-FOU / D-CHE carrier pass” 泛化为 “所有 variant 都 robust pass”。更精确地说：

$$
\boxed{
\text{D-FOU / D-CHE 已经有 official-native rows，}
\text{但 v22.14 需要 variant-level robust pass，不能只做 carrier-level OR。}
}
$$

## 3.3 效率路线卡在哪里？

目前的效率 blocker 已经从 v22.12 的 “有没有 fused arbitrary-cotangent path” 变成了三个更精确的问题。

### 3.3.1 D-FOU / D-CHE 的 variant robustness

D-FOU 和 D-CHE 各有好 variant，但也有 bad outlier。v22.14 应该把 official pass 从 carrier-level 改成 variant-level：

```text
variant_robust_pass requires:
  all non-smoke cotangent types pass ratio gate;
  all batch sizes pass ratio gate;
  no Delta-Gaussian outlier;
  no hidden fallback/autograd path;
  grad_rel_error pass.
```

如果只保留 best row 或 carrier OR，就会高估效率。

### 3.3.2 D-RBF / D-RAT 仍没有 generic cotangent fused backward

D-RBF / D-RAT 的 status 非常明确：

```text
arbitrary_cotangent_fused_backward_contract_missing
```

这两条线不能借 D-FOU / D-CHE 的成功。v22.14 如果继续推进 D-RBF / D-RAT，第一步不是 ratio sweep，而是实现：

```text
backward_from_grad_logits(model, x, grad_logits, logits, cache)
```

否则所有 profiler 都只能算 official blocked。

### 3.3.3 Efficiency 必须服务于 basis-state operator，而不只是 readout/operator-step profiler

v22.13 的 efficiency operator_id 对 KAN 主要是 `LIO3_KANLowBankSpectral`，但 KAN mapping 失败说明当前 operator target 并没有被 basis channel 承载。v22.14 的 efficiency runner 必须测：

$$
\delta \rightarrow T_\theta(\delta) \rightarrow \Delta s_{basis} \rightarrow \Delta f
$$

也就是 basis-state operator 的 full loop，而不是只测 output-cotangent VJP。

## 3.4 效率路线结论

v22.13 的效率路线是成功推进：

```text
D-FOU/D-CHE native arbitrary-cotangent officialization: yes
D-RBF/D-RAT officialization: no
variant-level robustness: partially open, not fully closed
basis-state operator efficiency: not yet measured
```

v22.14 不能再把主要工程预算放在 D-FOU/D-CHE 是否可能快；它们已经快了。下一步是：

```text
1. 固化 D-FOU/D-CHE 的 robust official variants；
2. 给 D-RBF/D-RAT 加 generic backward_from_grad_logits，或明确降级；
3. 把 efficiency runner 接到 KAN basis-state operator，而不是 readout replay。
```

---

# 4. Part C：Functional Update 路线审计

## 4.1 v22.13 是否推动了 functional update？

是，推动很明显。但它还不是最终算法突破。

v22.13 的 FU 证据链是：

```text
S2 operator atom pass rows = 5
S3 variational solve pass rows = 6
S4 operator metric commit pass rows = 3
S5 C3 adapter pass count = 4
S5 C4 adapter pass count = 3
StableRandom_or_RandomMatched_pass = 0
```

这比 v22.12 强很多。v22.12 是 MSE C4、Ranking C3、CE 大多失败；v22.13 变成：

```text
C3 pass:
  Delta-LossCEAdapter
  Delta-MSEAdapter
  Delta-RankingAdapter
  Delta-PreferenceAdapter-smoke

C4 pass:
  Delta-LossCEAdapter
  Delta-RankingAdapter
  Delta-PreferenceAdapter-smoke
```

具体 horizon row：

```text
Delta-LossCEAdapter:
  source_func_h3200 ≈ 1.147
  source_func_h4800 ≈ 1.137
  source_loss_h3200 ≈ 0.239
  source_loss_h4800 ≈ 0.212
  C3=1, C4=1

Delta-MSEAdapter:
  source_func_h3200 ≈ 0.428
  source_func_h4800 ≈ 0.204
  source_loss_h3200 ≈ 5.128
  source_loss_h4800 ≈ 2.055
  C3=1, C4=0
  reason: R4800_over_3200_func ≈ 0.478 < 0.50

Delta-RankingAdapter:
  source_func_h3200 ≈ 1.110
  source_func_h4800 ≈ 0.889
  source_loss_h3200 ≈ 0.121
  source_loss_h4800 ≈ 0.119
  C3=1, C4=1

Delta-PreferenceAdapter-smoke:
  source_func_h3200 ≈ 0.903
  source_func_h4800 ≈ 0.774
  source_loss_h3200 ≈ 0.339
  source_loss_h4800 ≈ 0.339
  C3=1, C4=1

Delta-GenericSourceTarget / normalized holdout:
  strongly fails

StableRandom / RandomMatched:
  strongly fails
```

这说明 v22.13 确实不是 MSE-only 了。CE 和 Ranking 的 source_loss 也为正，这是 v22.12 没做到的。

## 4.2 这个结果说明了什么 insight？

### Insight 1：metric-as-operator 比 metric-as-filter 更接近正确方向

v22.13 中表现最关键的方向是：

```text
LIO2_SplitCoherentControlNull
LIO7_LowNDSControlNull
```

它们不是从已有 update 里筛，而是直接定义：

$$
T_\theta(\delta)
= \text{Normalize}\big(P_{control-null} P_{low-curvature}(-\delta)\big).
$$

这比旧 observer/filter 路线更接近算法思想。metric 终于进入了 operator 的构造，而不是事后解释。

### Insight 2：row-axis control-null 的修复很重要

v22.13 里移除了 default control span 中的 `row_axis_control`，理由是它会擦掉 PairwiseRanking / Preference 这类 row-wise cotangent。这个修复很重要，因为 Ranking/Preference 正是在这一轮打开的。

这说明之前某些 ranking/preference/row-wise loss interface 失败，可能不是算法完全不行，而是 control-nullspace 定义把真实 signal 当成 control 删掉了。

### Insight 3：source_loss 与 target retention 终于一起变正

v22.12 的典型问题是 source_func 过、source_loss 负。v22.13 在 CE/Ranking/Preference 上同时 source_func 和 source_loss 为正，这比单纯 target retention 有意义。

这说明 $T_\theta(\delta)$ 至少在 MLP train-stream 实验台上能把 loss cotangent 变成 task-useful source，而不只是保留一个无关 displacement。

### Insight 4：MSE 不到 C4 反而是重要信号

MSE 的 source_loss 很大、一直为正，但 C4 不通过，因为 terminal retention ratio 不足。这说明 C4 gate 没有被 source_loss 巨大值直接绕过。这个 gate 有用。

MSE 不过 C4 的含义不是 MSE 失败，而是：

$$
\boxed{
\text{source_loss gain 与 function-retention ratio 是不同维度。}
}
$$

这对下一版很重要：我们不能只追 source_loss，也不能只追 source_func。

## 4.3 最大 caveat：horizon pass 依赖 source-anchor retention loss

这是我对 v22.13 最重要的保留意见。

v22.13 一开始的 official GPU horizon 是 NoGo；后面通过 repair sweep 找到：

```text
attempt = low_lr_source_anchor_w0p10_lr0p02
norm_scale = 5.12
```

源码中对应逻辑是：

```python
loss = adapter.value(logits_now, task_data)
if retention_active:
    displacement_now = logits_now - base_logits
    loss = loss + retention_weight * F.mse_loss(displacement_now, source_anchor)
```

也就是说，后续训练不是纯 AdamW on original adapter loss，而是加了一个 train-only source-anchor retention loss。虽然 controls 也各自 retain 自己的 anchor，因此比较相对公平，但这仍然改变了训练动力学。

在项目硬约束里，“no loss modification” 一直是强约束。因此这条证据应被降级为：

```text
role-blind operator + train-only anchor/proximal retention dynamics opened
```

而不是：

```text
pure functional update operator alone solved retention
```

如果我们想保留这条路线，需要在 v22.14 做一个严格拆分：

```text
Condition A: operator-only, no anchor, no auxiliary loss
Condition B: operator + periodic source state, no auxiliary loss
Condition C: operator + auxiliary source-anchor loss
Condition D: operator + optimizer-prox source anchor, no loss modification
Condition E: controls with matched own-anchor
```

只有 A/B/D 通过，才能接近 official operator claim。C 通过只能算 exploration：它证明 source anchor 对抗遗忘有价值，但不能单独证明 FU 没有通过修改 loss 取巧。

## 4.4 Task-level readback 说明什么？

v22.13 的 task readback 不能支持 scientific claim。

独立聚合结果：

```text
MLP+FU+AdamW:
  avg_test_acc ≈ 0.737 vs MLP+AdamW ≈ 0.743
  avg_train_acc 提高，但 avg_NLL_delta_vs_MLP ≈ +0.040
  NLL <= MLP rows = 3/9
  AUC_loss_time_ratio ≈ 0.923，说明训练 loss 收敛更快

KAN+FU+AdamW:
  avg_test_acc ≈ 0.668 vs KAN+AdamW ≈ 0.564
  明显改善 KAN ordinary BP
  但仍显著低于 MLP+AdamW ≈ 0.743
  KAN+FU accuracy >= MLP rows = 1/9
  avg_NLL_delta_vs_MLP ≈ +0.304
```

解释：

```text
1. FU 对 KAN ordinary BP 有明显帮助。
2. FU 对 MLP 的训练速度可能有帮助，但泛化没有稳定超过 MLP baseline。
3. KAN+FU 尚未超过 same-param MLP。
4. 当前 task_eval 使用的是短步数、小 subset readback，且 MLP+FU/KAN+FU 使用 task CE cotangent，不是完整 v22.13 final LIO2 horizon protocol；所以只能作为 smoke/readback。
```

因此不能写：

```text
functional update 已证明表达能力更好、抗遗忘、更快收敛。
```

最多写：

```text
functional update operator 在 synthetic train-stream horizon 上打开 source retention；
在 task readback 中 KAN+FU 明显优于 KAN+AdamW，但尚未超过 MLP，也未形成 scientific proof。
```

## 4.5 FU 路线结论

v22.13 的 FU 路线有真实推动：

$$
\boxed{
\text{从 O10 role-aware scaffold 进入了 role-blind } T_\theta(\delta) \text{ 实验台。}
}
$$

但仍卡在：

```text
1. operator-only 与 anchor-retention 没拆开；
2. source-anchor 以 auxiliary loss 形式实现，触碰 no loss modification 约束；
3. 只有一个 official seed / norm-scale / attempt，缺少独立 replication；
4. source-state dynamics artifact 为空，抗遗忘机制没有充分记录；
5. task-level claim 未达成；
6. KAN carrier 没接上。
```

下一步不能继续把 norm_scale 和 retention_weight 小扫做成 v22.14。真正要做的是：

$$
\boxed{
\text{把 source-anchor 从 loss 项改造成 optimizer/proximal/source-state dynamics，}
\text{并证明 operator 本身产生的是可被 KAN basis-state 承载的 source。}
}
$$

---

# 5. Part D：KAN carrier 审计

## 5.1 KAN 是否推动？

没有形成 official KAN carrier success。

v22.13 的 KAN route：

```text
S6-KANCorrectedCarrierNoGo
KAN_source_channel_pass_rows = 0
DFOU_corrected_source_exists = 0
DCHE_corrected_source_exists = 0
DFOU_efficiency_pass = 1
DCHE_efficiency_pass = 1
blocker = KANSourceChannelMismatchConfirmed
```

注意这里非常关键：D-FOU 和 D-CHE 的 efficiency 都过了。因此 KAN 的主要 blocker 已经不再是 v22.12 的 D-CHE efficiency blocked，而是 **source carrier mismatch**。

## 5.2 KAN rows 的真实含义

读出每个 row：

```text
D-FOU corrected_readout_layout_low_degree_source_replay:
  basis_estimate_fit_cosine ≈ 0.968
  projection_residual ≈ 0.281
  KAN_source_func_h3200 ≈ 0.0111
  KAN_source_func_h4800 ≈ 0.00627
  KAN_source_loss_h3200 ≈ 0.00525
  KAN_source_loss_h4800 ≈ 0.00103
  KAN_specific_delta_vs_MLP_same_metric ≈ -0.417
  readout_channel_energy = 1
  basis_channel_energy = 0
  decision = KANSourceChannelMismatchConfirmed

D-CHE corrected_readout_layout_low_degree_source_replay:
  basis_estimate_fit_cosine ≈ 0.938
  projection_residual ≈ 0.369
  KAN_source_func_h3200 ≈ 0.0273
  KAN_source_func_h4800 ≈ 0.0241
  KAN_source_loss_h3200 ≈ 0.00807
  KAN_source_loss_h4800 ≈ 0.00266
  KAN_specific_delta_vs_MLP_same_metric ≈ -0.401
  readout_channel_energy = 1
  basis_channel_energy = 0
  decision = KANSourceChannelMismatchConfirmed

basis_linearized_w1 rows:
  basis_channel_energy = 1
  basis_estimate_fit_cosine ≈ 0.06 到 0.085
  projection_residual ≈ 0.998
  source_func small / unstable
  gain16 出现 source_loss collapse
  K17_basis_channel_pass = 0
```

这说明：

```text
1. corrected readout rows 能保留一点 source，但它们是 readout-only，不是 basis-channel source。
2. basis-linearized rows 真正动了 basis channel，但拟合不了当前 target，projection residual 接近 1。
3. KAN 的 source_func/source_loss 虽然有绝对正值，但远弱于 MLP same-operator baseline。
4. KAN_specific_delta_vs_MLP_same_metric 全部为负，不能写成 KAN carrier advantage。
```

## 5.3 这暴露出什么数学问题？

当前 $T_\theta(\delta)$ 产生的是 output-space displacement。readout solve 可以拟合它，因为 readout tangent space 对 logits 直接线性；但 basis channel 的 tangent space不同，特别是 $w_1$ 改变的是 hidden representation 与 basis feature，不是直接 output displacement。

用数学说，v22.13 实际在做：

$$
\Delta f^* = T_\theta(\delta)
$$

然后试图找：

$$
\Delta w_1^* = \arg\min_{\Delta w_1}\|J_{w_1}\Delta w_1 - \Delta f^*\|^2.
$$

v22.13 的结果显示：

$$
\frac{\|J_{w_1}\Delta w_1 - \Delta f^*\|}{\|\Delta f^*\|} \approx 1.
$$

也就是当前 $\Delta f^*$ 根本不在 basis tangent space 中，或者至少以当前线性化 / damping / rank 设置不可达。

所以 v22.14 不能继续用 readout target replay 去逼 basis。真正应该变成：

$$
\Delta w_1^*
= \arg\min_v
\left[
\langle \delta, J_{w_1}v\rangle
+ \lambda \|v\|_{G_B}^2
+ \mu \operatorname{NDS}(J_{w_1}v)
+ \nu \operatorname{ReadoutLeak}(J_{w_1}v)
\right].
$$

也就是说，KAN FU 不能先生成 arbitrary output target，再让 basis 硬拟合；它要直接在 basis tangent / basis state 中定义 operator。

## 5.4 哪些旧方向因为代码问题值得重开？

### 5.4.1 D-CHE legacy mismatch 结论需要重写

v22.12 K14 和 v22.13 layout tests 证明：旧 D-CHE readout layout 错。所有基于 legacy layout 的 D-CHE source mismatch 结论都应该降级。

但 v22.13 corrected-layout 后仍然没有 KAN source carrier success，所以新的结论是：

```text
D-CHE 不是因为 layout 无法写 source；
D-CHE 是因为 source 主要停留在 readout target，basis-state carrier 没学到 retained source。
```

### 5.4.2 D-FOU K13 旧 pass 需要降级

v22.12 的 D-FOU K13 曾经看起来 clean pass。但 v22.13 corrected K13 revalidation 显示：

```text
absolute source 有弱正值；
但 KAN_specific_delta_vs_MLP_same_metric ≈ -0.417；
basis channel 没通过。
```

所以旧 K13 不能再写成 KAN retained source success。它最多说明 D-FOU readout/source replay 可产生弱 positive。

### 5.4.3 Split-consensus / control-nullspace 作为 operator component 值得继续

v22.13 的 LIO2/LIO7 说明 control-nullspace 和 low-NDS 方向不是废的。之前 observer/filter 路线失败，不代表这些几何量没有价值；它们应该作为 $T_\theta$ 的构成，而不是事后 selector。

### 5.4.4 Row-axis control-null 需要作为 diagnostic 保留

移除 row-axis control 后 Ranking/Preference 打开。这说明 row-axis 方向可能是真实 signal。v22.14 应该保留 row-axis as diagnostic，而不是重新默认投掉它。

### 5.4.5 F53/terminal floor 仍不应作为主线

v22.0 已经证明大量 terminal floor / lookahead / hold / fallback 可以到 h3200，但 h4800 断链。v22.13 的进展不是靠 terminal floor，而是靠 operator + source-anchor。因此 terminal repair 只能继续作为 destructive projection localization，不应再作为突破主线。

---

# 6. v22.14 总体目标

v22.14 的名字建议改为：

```text
DG-KAN v22.14
Basis-State Loss-Interface Operator FU
+ Source-Anchor Proximal Dynamics
+ Robust Native Cotangent Efficiency
```

总目标：

$$
\boxed{
\text{把 v22.13 的 role-blind } T_\theta(\delta)
\text{ 从 MLP/readout horizon success，推进到 KAN basis-state carrier success。}
}
$$

具体目标分四层：

```text
A. Code truth:
   修正 finalizer official promotion 口径；修复 source_state_dynamics 空 artifact；锁住 no-loss-modification / anchor mechanism 标记。

B. Efficiency:
   固化 D-FOU/D-CHE robust native variants；不要只用 carrier OR；D-RBF/D-RAT 要么实现 generic backward_from_grad_logits，要么明确 blocked。

C. Functional Update:
   拆分 operator-only、auxiliary source-anchor、optimizer-prox source-anchor；证明不是靠修改 loss 才成功。

D. KAN carrier:
   从 output displacement target 转向 basis-state operator，直接优化 basis tangent / basis source state。

E. Task value:
   证明 KAN+FU 不只是优于 KAN+AdamW，还要开始接近或超过 same-param MLP，并在收敛速度/遗忘/表达能力 readback 上有一致改善。
```

v22.14 的核心科学假设：

$$
\boxed{
\text{v22.13 已经找到一个有效的 role-blind cotangent preconditioner，}
\text{但它的 retained source 主要由 readout/anchor 承载；}
\text{KAN 要成功，必须把 operator 定义在 basis-state tangent 中。}
}
$$

---

# 7. Part A：v22.14 代码与语义修复计划

## 7.1 A1：finalizer official promotion 必须读取 execution contract

### 问题

v22.13 计划规定：

```text
if runnable_queue_nonempty and any_gpu_idle > 10 minutes:
    execution_contract_violation = 1
    official_promotion_allowed = 0
```

但 v22.13 finalizer 的 `_route_decision()` 没读取 `v22_13_idle_violation.csv`。结果是：

```text
v22_13_idle_violation.csv:
  execution_contract_violation = 1

v22_13_final_decision.json:
  official_promotion_allowed = 1
```

这是 route 口径 bug。

### v22.14 修复

finalizer 必须读取：

```text
v22_14_idle_violation.csv
v22_14_queue_drain_report.json
```

并执行：

```text
if execution_contract_violation == 1:
    official_promotion_allowed = 0
    official_operator_promotion_allowed = 0
    route_suffix += "_ExecutionContractViolation"
```

记录字段：

```text
execution_contract_violation
queue_drained
runnable_queue_nonempty_at_idle
max_idle_gap_sec
official_blocked_by_execution_contract
```

成功标准：

```text
execution_contract_violation = 0
queue_drained = 1
or official_promotion_allowed = 0 if violation = 1
```

不满足时 Codex 先做：

```text
1. 修 dynamic queue runner，不跑长实验；
2. 重建 queue artifacts；
3. 只重跑 finalizer；
4. 不允许继续把 official 写成 1。
```

## 7.2 A2：source_state_dynamics artifact 必须非空

### 问题

v22.13 的 `v22_13_source_state_dynamics.csv` 是空文件。源码原因是 `_attempts()` 中有 `source_state_attempt`，但 `_evaluate_attempt_gpu()` 没把它复制到 row。

### v22.14 修复

horizon row 必须新增：

```text
source_state_attempt
source_state_type
source_state_write_fraction
source_state_anchor_norm
source_state_alignment_h*
source_state_decay_rate_h*
source_state_to_control_projection_h*
source_state_to_task_loss_projection_h*
```

成功标准：

```text
v22_14_source_state_dynamics.csv rows >= FU_attempt_rows
source_state_attempt field non-empty
source_state_decay_rate_h3200/h4800 finite
```

不满足时 Codex 先做：

```text
1. 修 row decoration；
2. 重跑 short smoke horizon h100/h400；
3. 不进入 full h6400。
```

## 7.3 A3：anchor mechanism 必须显式分类

v22.13 的 source-anchor retention 是有效修复，但它修改了 training loss。v22.14 必须显式区分：

```text
anchor_mechanism_type:
  none
  periodic_update
  auxiliary_loss_anchor
  optimizer_prox_anchor
  source_state_projector
  parameter_ema_reentry

uses_loss_modification_for_retention:
  0/1

official_strict_anchor_allowed:
  1 only if uses_loss_modification_for_retention = 0
```

记录：

```text
retention_weight
retention_loss_value_h*
anchor_gradient_norm_h*
anchor_to_task_gradient_cos_h*
anchor_update_equivalent_norm_h*
optimizer_prox_residual_h*
```

通过标准：

```text
strict official operator success:
  uses_loss_modification_for_retention = 0
  C3 >= 3 adapters
  C4 >= 2 adapters
  source_loss_h4800 >= 0

auxiliary-loss exploration:
  uses_loss_modification_for_retention = 1
  can count only as exploration, not strict official.
```

不满足时 Codex 先做：

```text
1. 把 auxiliary source-anchor loss 改写为 optimizer-prox update；
2. 保留 auxiliary loss 作为 diagnostic upper bound；
3. 重跑 same operator / same norm_scale / same adapters；
4. 记录 strict vs auxiliary gap。
```

## 7.4 A4：artifact naming 与 non-empty required artifact gate

v22.14 需要增加 required artifact gate：

```text
required_artifact_exists
required_artifact_nonempty
required_artifact_min_rows
required_artifact_schema_pass
```

对以下 artifact 强制 min rows：

```text
source_state_dynamics: >= FU_attempt_rows
control_attribution_matrix: >= control_rows
native_efficiency_truth_table: >= planned_rows
KAN_mapping_matrix: >= planned_KAN_rows
task_eval_matrix: >= planned_task_rows
```

同时把计划中的 `.csv` / `.log` 命名对齐，避免 `compileall_report.csv` 在计划里有、实际只有 `.log`。

## 7.5 A5：independent replication gate

v22.13 的 best route 基本是 seed 2213 + norm_scale 5.12 + selected attempt。v22.14 必须新增 independent replication：

```text
seeds = 2213, 2214, 2215
operator payload seeds independent
adapter order shuffled
norm_scale fixed from precommit, not chosen from same horizon
```

通过标准：

```text
replicated_C3_adapter_count_median >= 3
replicated_C4_adapter_count_median >= 2
minimum C3 adapter count across seeds >= 2
StableRandom/RandomMatched pass count = 0
```

如果独立 seed 失败：

```text
route = OperatorHorizonSingleSeedOnly_NotOfficial
```

---

# 8. Part B：v22.14 efficiency 计划

## 8.1 目标

v22.14 efficiency 的目标不是重新证明 D-FOU/D-CHE 可能快，而是固化 robust official path，并让 efficiency 服务于 basis-state operator。

总问题：

$$
\boxed{
\text{D-FOU / D-CHE 能否在 } \delta \rightarrow T_\theta(\delta) \rightarrow \Delta s_{basis}
\text{ 的完整 operator-step 中仍保持 MLP-like？}
}
$$

## 8.2 E1：variant-level robust officialization

### 实验对象

D-FOU：

```text
FOU22.14-R2-tablelookup-bandreadout-native-operator-step
FOU22.14-R3-lowfreq-source-state-fused-commit
FOU22.14-R4-basis-state-operator-step
```

D-CHE：

```text
CHE22.14-R2-k5-gradbuf-no-materialize-native-vjp
CHE22.14-R3-lowdegree-readout-operator-step
CHE22.14-R4-corrected-layout-source-state-fused-commit
CHE22.14-R5-basis-state-operator-step
```

D-RBF/D-RAT：

```text
only if backward_from_grad_logits implemented;
otherwise official_status = blocked, no ratio sweep promotion.
```

### Cotangent suite

```text
Delta-Gaussian
Delta-StableRandom
Delta-LossCEAdapter
Delta-MSEAdapter
Delta-RankingAdapter
Delta-PreferenceAdapter-smoke
Delta-GenericHoldout
```

### Batch grid

```text
batch_size = 128, 256, 512, 1024
hidden = 64, 128
repeat = 5
warmup = 3
```

### 记录指标

```text
carrier
variant
operator_stage
cotangent_type
batch_size
hidden
basis_dim
forward_ms
native_vjp_ms
operator_apply_ms
basis_state_solve_ms
readout_solve_ms
source_state_update_ms
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
native_grad_rel_error
upstream_cotangent_contract_pass
functional_runner_kernel_match
variant_robust_pass
carrier_robust_pass
```

### variant-level pass 标准

```text
for all non-smoke cotangents and all batch sizes:
  official_fused_kernel_complete = 1
  manual_upstream_vjp_used = 0
  fallback_kernel_used = 0
  native_grad_rel_error <= 5e-3
  forward_ratio_vs_mlp <= 1.35
  vjp_ratio_vs_mlp <= 1.35
  operator_step_ratio_vs_mlp <= 1.35
  full_step_ratio_vs_mlp <= 1.35
  memory_ratio_vs_mlp <= 1.10
```

Exploration pass 可放宽：

```text
>=80% rows pass ratio gate
no official_fused missing
no manual VJP
```

### 不满足时 Codex 先尝试

如果 D-FOU R2/R3 pass 但 R1 Gaussian outlier：

```text
1. R1 降级，不作为 official robust variant；
2. 只保留 R2/R3 做 official path；
3. 对 Gaussian outlier 做 component timing，不继续用 carrier OR 掩盖。
```

如果 D-CHE R1 outlier：

```text
1. 用 R2 gradbuf / R4 source-state fused commit 替代；
2. 检查 Gaussian cotangent 下 w1_grad kernel block size；
3. 不因 R3/R4 pass 而宣称 R1 robust。
```

如果 basis-state operator step 超时：

```text
1. 降低 basis tangent solve rank；
2. 缓存 basis Gram / Fisher diagonal；
3. 使用 block-CG 而非 dense solve；
4. 若仍 >1.35，basis-state path 写成 functional exploration, not efficiency official。
```

## 8.3 E2：D-RBF / D-RAT generic cotangent fused backward

D-RBF / D-RAT 只有在实现以下函数后才能进入 official path：

```text
fused_rbf_local.backward_from_grad_logits(model, x, grad_logits, logits, cache)
fused_rational_k4.backward_from_grad_logits(model, x, grad_logits, logits, cache)
```

记录：

```text
arbitrary_cotangent_backward_available
native_grad_rel_error
vjp_ratio_vs_mlp
reciprocal_ms
exp_eval_ms
local_gather_ms
local_backward_ms
numerator_grad_ms
denominator_grad_ms
condition_safety_ms
```

通过标准同 E1。

如果 2 次实现尝试后仍 blocked：

```text
route = RBF_RAT_GenericCotangentKernelBlocked
D-RBF/D-RAT 不再进入 v22.14 official carrier budget。
```

---

# 9. Part C：v22.14 Functional Update 算法计划

## 9.1 总体数学定义

v22.14 的 FU 继续沿用 loss-interface operator：

$$
\delta_a = \frac{\partial \mathcal{L}_a}{\partial f_\theta(x)},
\qquad
\Delta f_a = T_\theta(\delta_a).
$$

但 v22.14 要明确拆开两个问题：

```text
Creation:
  operator 是否能产生 task-useful source direction？

Retention:
  source 是否能在普通训练/optimizer dynamics 中不被遗忘？
```

因此训练动力学写成：

$$
\theta_{t+1}
= \theta_t + U_t\big(T_\theta(\delta_t), g_t, s_t\big),
$$

其中：

```text
Tθ:
  role-blind cotangent operator；

g_t:
  ordinary training gradient；

s_t:
  source state / anchor / memory；

U_t:
  commit + optimizer/proximal dynamics。
```

v22.13 把 $s_t$ 作为 auxiliary loss 写进 training loop。v22.14 必须测试能否把 $s_t$ 作为 optimizer dynamics，而不是修改 task loss。

## 9.2 F1：operator-only vs anchor-retention 分离实验

### 假设

H-F1：v22.13 的 C4 主要来自 $T_\theta(\delta)$ 与 source-anchor dynamics 的组合；operator-only 可能只能产生 C3 或短期 source。

预测：

```text
1. operator-only 条件下，C3 adapter count 会下降，C4 可能消失。
2. auxiliary_loss_anchor 会复现 v22.13 C4。
3. optimizer_prox_anchor 若能复现 auxiliary_loss_anchor，说明 source-anchor 可转成合法 train-only optimizer dynamics。
4. controls with own-anchor 仍不能 pass，说明 anchor 不是单纯保留任意 random displacement。
```

### 实验条件

同一个 operator payload、同一个 norm_scale、同一批 adapters，跑：

```text
F1-A operator_only:
  retention_weight = 0
  periodic_interval = 0
  uses_loss_modification_for_retention = 0

F1-B periodic_source_state:
  periodic_interval = 800
  periodic_scale = 0.006
  retention_weight = 0
  uses_loss_modification_for_retention = 0

F1-C auxiliary_loss_anchor:
  retention_weight = 0.10
  retention_start_step = 1
  retention_stop_step = 6400
  uses_loss_modification_for_retention = 1

F1-D optimizer_prox_anchor:
  no auxiliary loss;
  optimizer step solves source-prox update;
  uses_loss_modification_for_retention = 0

F1-E source_state_projector:
  no auxiliary loss;
  project destructive gradient component away from retained source state;
  uses_loss_modification_for_retention = 0

F1-F controls_own_anchor:
  RandomMatched / StableRandom / SameSolverRandomTarget / SignFlip / Corrupt each retains own anchor。
```

### Optimizer-prox anchor 定义

不用修改 task loss，而是在每步梯度后求：

$$
u_t^* = \arg\min_u
\left[
\langle g_t, u\rangle
+ \frac{1}{2\eta}\|u\|_P^2
+ \lambda \|J_t u - r_t\|_{G_f}^2
\right],
$$

其中 $r_t$ 是 retained source residual：

$$
r_t = \Delta f_{anchor} - \big(f_{\theta_t}(x)-f_{\theta_0}(x)\big).
$$

这和 auxiliary loss 在数学上相关，但实验语义不同：它是 optimizer update rule，不是改 task loss。v22.14 要两者分开记录。

### Source-state projector 定义

计算 ordinary gradient step 的 functional displacement：

$$
\Delta f_g = J_t(-\eta g_t).
$$

若它破坏 source anchor：

$$
\langle \Delta f_g, r_t \rangle < 0,
$$

则投掉 destructive component：

$$
\Delta f_g^{safe}
= \Delta f_g
- \frac{\min(0,\langle \Delta f_g,r_t\rangle)}{\|r_t\|^2+\epsilon}r_t.
$$

再用 low-rank commit 近似写回参数。

### 记录指标

```text
operator_id
anchor_mechanism_type
uses_loss_modification_for_retention
norm_scale
loss_adapter_name
adapter_seen_in_operator_tuning
source_func_h100/h400/h800/h1600/h2400/h3200/h4000/h4800/h6400
source_loss_h100/h400/h800/h1600/h2400/h3200/h4000/h4800/h6400
R3200_over_1600_func
R4800_over_3200_func
R6400_over_4800_func
R4800_over_3200_loss
row_positive_count_h*
control_equivalent_fraction_h*
source_state_alignment_h*
source_state_decay_rate_h*
optimizer_destructive_projection_h*
anchor_gradient_norm_h*
anchor_to_task_gradient_cos_h*
retention_loss_value_h*
optimizer_prox_residual_h*
AUC_loss_step
AUC_loss_time
AUC_loss_time_ratio_vs_best_control
C3_source_formation_pass
C4_terminal_retention_pass
C5_h6400_retention_pass
TargetRetentionOnly_NotTaskUseful
```

### 判断标准

Exploration success：

```text
C3 pass on >=3 non-random adapters
C4 pass on >=2 non-random adapters
StableRandom/RandomMatched fail
source_loss_h4800 >= -1e-6
```

Strict official operator success：

```text
Exploration success
uses_loss_modification_for_retention = 0
adapter holdout pass
independent seeds >=2/3 pass
source_state_dynamics artifact non-empty
```

Auxiliary-only success：

```text
F1-C passes but F1-D/F1-E fail
route = AuxiliaryAnchorUpperBound_OptimizerProxNotClosed
```

不满足时 Codex 先尝试：

```text
If operator_only fails but auxiliary_anchor passes:
  implement optimizer_prox_anchor, do not tune retention_weight further.

If optimizer_prox_anchor fails:
  reduce J_t solve rank, add diagonal Fisher P, try source_state_projector.

If controls with own anchor pass:
  anchor definition is too generic; add source_loss/control-null constraint before retention.

If CE/Ranking pass but MSE fails C4:
  inspect R4800_over_3200_func and source_loss tradeoff; do not force output-direction similarity.

If all anchors fail on independent seeds:
  route = SourceRetentionDynamicsNotSolved; return to source-state theory.
```

## 9.3 F2：new role-blind operator families

v22.14 不应该只调 LIO2/LIO7 scale。要增加真正的 metric-as-operator variants。

### LIO8：Fisher-Sobolev Resolvent Operator

定义：

$$
T_8(\delta)
= -\left(G_F + \lambda G_S + \mu I\right)^{-1}\delta.
$$

其中：

```text
G_F: train-stream Fisher diagonal/block approximation
G_S: Sobolev / low-curvature penalty
I: damping
```

记录：

```text
fisher_energy_before/after
sobolev_energy_before/after
resolvent_cg_iterations
resolvent_residual
NDS_reduction
loss_gain_ratio
```

### LIO9：Adapter-Whitened Cotangent Operator

不同 loss adapter 的 cotangent norm 和 row structure 差异很大。LIO9 不看 adapter name，只对 cotangent 自身 whitening：

$$
\tilde{\delta}
= \frac{\delta - \operatorname{mean}_{row}(\delta)}{\sqrt{\operatorname{Var}_{row}(\delta)+\epsilon}}.
$$

然后：

$$
T_9(\delta)=P_{control-null}P_{low-curvature}(-\tilde{\delta}).
$$

记录：

```text
row_whitening_gain
class_gauge_invariance_error
adapter_gain_ratio_variance
```

### LIO10：Signal-Reservoir Split Operator

用 train micro-splits 构造 signal 子空间：

$$
P_{signal} = \operatorname{TopEig}\left(\sum_i \delta_i\delta_i^T\right).
$$

然后：

$$
T_{10}(\delta) = -P_{signal}P_{control-null}\delta.
$$

记录：

```text
signal_rank
signal_projection_gain
reservoir_leakage
noise_into_signal_ratio
split_pair_cosine_mean/min
```

### LIO11：Source-Loss Boundary Operator without adapter branch

v22.13 的 source_loss 是结果，不是 operator 目标。LIO11 用 train-only first-order source-loss boundary：

$$
T_{11}(\delta)
= \arg\min_v
\left[
\langle \delta,v\rangle
+ \lambda \operatorname{NDS}(v)
+ \mu \|P_{control}v\|^2
+ \nu \max(0, \langle \delta_{next-proxy},v\rangle)^2
\right].
$$

这里 $\delta_{next-proxy}$ 只能来自 current train micro-split，不允许未来 horizon。

### Operator family gate

每个 LIO operator 必须记录：

```text
operator_class
uses_adapter_name_for_direction
uses_loss_formula_for_direction
adapter_renaming_pass
operator_lipschitz_ratio
cotangent_permutation_equivariance_error
class_gauge_invariance_error
GainRatio_CE/MSE/Ranking/Preference/Holdout
NDS_reduction
ControlNullRatio
source_loss_proxy_gain
```

通过标准：

```text
adapter_renaming_pass = 1
uses_adapter_name_for_direction = 0
uses_loss_formula_for_direction = 0
operator_lipschitz_ratio <= 5
NDS_reduction >= 0 or justified by gain
control_projection_after <= 0.25
GainRatio positive on >=4/5 non-random cotangents
```

## 9.4 F3：adapter robustness 与 holdout

v22.14 adapter suite：

```text
Seen during operator construction:
  Delta-LossCEAdapter
  Delta-MSEAdapter
  Delta-RankingAdapter

Holdout / smoke:
  Delta-PreferenceAdapter
  Delta-GenericSourceTargetNormalized
  Delta-HuberRegressionAdapter
  Delta-FocalLikeAdapter diagnostic only, not CE-specific official

Controls:
  Delta-StableRandom-control
  Delta-RandomMatched-control
  Delta-SignFlipTarget-control
  Delta-CorruptTarget-control
```

注意：FocalLike 只能作为 extra loss-interface smoke，不能让 operator core读取 focal formula。

通过标准：

```text
C3 >= 3 seen/non-random adapters
C4 >= 2 seen/non-random adapters
at least one non-MSE C4
at least one holdout C3
controls fail
independent seed pass
```

如果 only seen adapters pass but holdout fails：

```text
route = AdapterSuiteOverfit_NotUniversalOperator
```

---

# 10. Part D：KAN basis-state carrier 计划

## 10.1 核心问题

v22.13 的 KAN no-go 不是简单“source 不正”，而是：

```text
readout-only 能写一点 source；
basis-linearized w1 真正动 basis，但 projection residual ≈ 1；
KAN_specific_delta_vs_MLP 全负。
```

这说明当前 output target $T_\theta(\delta)$ 不适合直接由 basis tangent 拟合。

v22.14 的 KAN 问题改成：

$$
\boxed{
\text{能否直接在 KAN basis-state tangent 中定义 } T^B_\theta(\delta)，
\text{而不是先生成 readout target 再硬拟合 basis？}
}
$$

## 10.2 K18：basis tangent coverage audit

### 假设

H-K18：v22.13 basis-linearized rows 失败，是因为 $T_\theta(\delta)$ 主要落在 readout tangent，而不在 basis tangent。

### 实验

对 D-CHE / D-FOU 分别计算：

```text
J_readout
J_basis_w1
J_basis_state_lowrank
J_basis_plus_readout
```

然后把 $T_\theta(\delta)$ 投影到不同 tangent space：

$$
\Pi_R T,
\qquad
\Pi_B T,
\qquad
\Pi_{B+R}T.
$$

### 记录指标

```text
carrier
operator_id
loss_adapter_name
target_norm
readout_projection_residual
basis_projection_residual
basis_plus_readout_projection_residual
readout_projection_cosine
basis_projection_cosine
basis_plus_readout_projection_cosine
basis_tangent_rank
readout_tangent_rank
basis_condition_number
readout_condition_number
basis_energy_needed_for_same_delta
source_loss_proxy_readout
source_loss_proxy_basis
source_loss_proxy_basis_plus_readout
```

### 判断标准

如果：

```text
basis_projection_residual >= 0.80
readout_projection_residual <= 0.40
```

则说明 v22.13 target 是 readout-dominated，不能继续强行 basis replay。进入 K19 basis-state operator。

如果：

```text
basis_projection_residual <= 0.40
```

但 horizon 仍失败，则说明不是 reachability，而是 basis dynamics / optimizer washout 问题，进入 K20 slow basis-state retention。

## 10.3 K19：direct basis-state operator

### 数学定义

不再先求 $\Delta f=T(\delta)$，再拟合 basis；而是直接求 basis update：

$$
\Delta b^*
= \arg\min_{\Delta b}
\left[
\langle \delta, J_b\Delta b\rangle
+ \lambda \|\Delta b\|_{G_B}^2
+ \mu \operatorname{NDS}(J_b\Delta b)
+ \nu \|P_{control}J_b\Delta b\|^2
+ \omega \operatorname{ReservoirLeak}(\Delta b)
\right].
$$

然后：

$$
\Delta f_B = J_b\Delta b^*.
$$

这里 $b$ 可以是：

```text
D-CHE:
  low-degree w1 bank
  degree-readout bank
  cheby pair-cross low-rank state

D-FOU:
  low-frequency w1 bank
  band-readout bank
  frequency source-state
```

### 实现路线

```text
K19-A basis_w1_diagonal_green:
  diagonal Fisher/Sobolev approximation on w1.

K19-B basis_lowrank_cg:
  low-rank J_b CG solve, rank cap 32/64.

K19-C basis_state_buffer:
  do not permanently change model parameter first;
  maintain optimizer source-state buffer and inject through fused basis carrier.

K19-D basis_readout_coupled:
  solve basis first, then minimal readout correction only for residual.
```

### 记录指标

```text
basis_operator_id
basis_state_type
basis_rank_cap
basis_cg_iterations
basis_operator_solve_ms
basis_operator_residual
basis_update_norm
basis_channel_energy
readout_channel_energy
basis_to_readout_energy_ratio
basis_projection_cosine_with_descent
basis_source_loss_proxy
KAN_source_func_h*
KAN_source_loss_h*
KAN_specific_delta_vs_MLP_same_metric
R4800_over_3200_func
basis_state_decay_rate
basis_optimizer_destructive_projection
full_operator_step_ratio_vs_MLP
memory_ratio_vs_MLP
```

### 通过标准

Exploration KAN basis pass：

```text
basis_channel_energy >= 0.50
KAN_source_func_h100/h400/h800/h1600/h3200 >= 0.005
KAN_source_loss_h3200 >= -1e-6
R4800_over_3200_func >= 0.50
KAN_specific_delta_vs_MLP_same_metric >= -0.05
controls fail
```

Official KAN basis pass：

```text
basis exploration pass
KAN_source_loss_h4800 >= -1e-6
KAN_specific_delta_vs_MLP_same_metric >= 0.005
basis_channel_energy >= 0.50
carrier_specific_efficiency_pass = 1
full_operator_step_ratio_vs_MLP <= 1.35
independent seed pass >= 2/3
```

如果 KAN remains negative vs MLP：

```text
route = KANBasisStateReachableButNotCompetitive
```

如果 basis projection residual remains high：

```text
route = KANBasisTangentCoverageNoGo
```

## 10.4 K20：readout-to-basis source transfer

v22.13 readout-only rows 有弱 positive，但 basis rows失败。v22.14 要测试 source 是否可以先在 readout形成，再迁移到 basis slow-state。

定义：

$$
\Delta f_t
= (1-\alpha_t)\Delta f_R + \alpha_t \Delta f_B,
\qquad
\alpha_t = \min(1, t/\tau).
$$

实验：

```text
K20-A tau=800
K20-B tau=1600
K20-C tau=3200
K20-D adaptive tau based on source_loss_proxy
```

记录：

```text
alpha_t_h*
readout_source_energy_h*
basis_source_energy_h*
source_transfer_success_h*
source_loss_h*
readout_to_basis_cosine_h*
basis_state_decay_rate_h*
```

通过标准：

```text
basis_source_energy_h3200 >= 0.50
basis_source_energy_h4800 >= 0.50
source_func_h4800 >= 0.005
source_loss_h4800 >= -1e-6
KAN_specific_delta_vs_MLP >= 0.005
```

不满足时 Codex 先尝试：

```text
If readout positive but basis transfer fails:
  reduce alpha speed; add basis-prox; inspect basis tangent residual.

If basis transfer destroys source_loss:
  add source_loss proxy to basis objective; do not increase replay scale.

If source remains readout-only:
  route = ReadoutSourceCannotMigrateToBasis; stop readout replay promotion.
```

## 10.5 K21：KAN carrier architecture boundary

如果 K18/K19/K20 都显示 basis tangent coverage不足，可以测试一个严格隔离的 architecture boundary：

```text
basis_source_state is optimizer/source state, not model parameter;
no extra inference-time parameters for official strict path;
optional low-rank basis adapter is diagnostic only unless parameter budget matched.
```

新增 diagnostic：

```text
K21-A optimizer-only basis source state
K21-B matched-param low-rank basis adapter diagnostic
K21-C source-state buffer fused into kernel training only, zeroed at inference
```

严格 official 只允许 K21-A / K21-C。K21-B 只能作为 architecture hypothesis。

---

# 11. Part E：task-level expression / forgetting / convergence plan

## 11.1 目标

v22.13 task readback 只证明 KAN+FU > KAN+AdamW，不能证明 KAN+FU > MLP。v22.14 必须把 task-level evidence 与 operator/KAN gates 分开，但要开始接近最终科学 claim。

最终要证明：

$$
\boxed{
\text{Functional update 改善纯梯度反传：表达更强、遗忘更少、收敛更快。}
}
$$

但 task metrics 仍然只能 readback，不能生成 direction。

## 11.2 实验矩阵

Datasets：

```text
MNIST
FashionMNIST
KMNIST
```

Seeds：

```text
0, 1, 2
```

Train sizes：

```text
1024 smoke
4096 confirmation
full optional only after gates pass
```

Steps：

```text
80 smoke
400 medium
1200 retention confirmation
```

Variants：

```text
MLP+AdamW
MLP+SGD
MLP+FU_operator_only
MLP+FU_optimizer_prox_anchor
MLP+FU_auxiliary_anchor_diagnostic
KAN+AdamW
KAN+FU_readout_only_diagnostic
KAN+FU_basis_state_operator
KAN+FU_basis_state_operator_prox_anchor
KAN+RandomMatchedFU_control
KAN+StableRandomFU_control
```

## 11.3 记录指标

```text
final_train_loss
final_test_loss_readback
final_train_accuracy
final_test_accuracy_readback
NLL_delta_vs_MLP
NLL_delta_vs_KAN_AdamW
accuracy_delta_vs_MLP
accuracy_delta_vs_KAN_AdamW
AUC_loss_step
AUC_loss_time
AUC_loss_time_ratio_vs_best_control
steps_to_threshold_70/80/90pct
wallclock_sec
step_time_ratio_vs_MLP
memory_ratio_vs_MLP
ECE_delta
Brier_delta
LineC_readback
per_example_loss_q95/q99
forgetting_after_stream_shift
source_func_h*
source_loss_h*
source_state_decay_h*
function_rank_effective
basis_effective_rank
basis_source_energy
readout_source_energy
```

## 11.4 判断标准

Exploration task value：

```text
KAN+FU_basis_state_operator beats KAN+AdamW on >=7/9 rows by NLL or accuracy
AUC_loss_time_ratio_vs_KAN_AdamW <= 0.90
no ECE/Brier explosion
```

Scientific task gate：

```text
KAN+FU beats same-param MLP on >=7/9 rows
KAN+FU beats KAN+AdamW on >=8/9 rows
AUC_loss_time_ratio_vs_best_control <= 1.05
source forgetting lower than controls by >=25%
step_time_ratio_vs_MLP <= 1.35
memory_ratio_vs_MLP <= 1.20
ECE/Brier/NLL debt not exploded
```

如果只 KAN+FU > KAN+AdamW but < MLP：

```text
route = KANBackpropImprovedButMLPNotBeaten
```

这仍是 useful progress，但不是最终 DG-KAN claim。

---

# 12. Controls 与 forbidden-information firewall

每个 v22.14 official row 必须有 controls：

```text
NoOpMatchedOverhead
RandomMatchedNorm
StableRandom
SameSolverRandomTarget
SignFlipTarget
CorruptTarget
AdamWParallelDirection
SGDParallelDirection
AnchorOnlyNoOperator
OperatorOnlyNoAnchor
AuxiliaryAnchorRandomSource
OptimizerProxRandomSource
```

每个 row 必须记录：

```text
uses_validation_test_future_query
uses_audit_metric_for_direction
uses_adapter_name_for_direction
uses_loss_formula_for_direction
uses_loss_modification_for_retention
uses_labels_for_fu_core
uses_labels_for_adapter_only
adapter_seen_in_operator_tuning
holdout_adapter
```

硬 forbidden：

```text
validation/test/future/query direction
LineC/ECE/Brier/AUCtime/tail direction
dataset-name branch
seed-specific rule
CE/MSE/Ranking formula branch inside operator core
readout-only claim as basis-channel success
manual VJP row as official fused success
auxiliary loss anchor as strict no-loss-modification official success
```

---

# 13. 4GPU dynamic queue plan

v22.14 必须真正执行 dynamic queue，而不是 staged commands。

## 13.1 初始分工

```text
GPU0:
  S0/A code truth and finalizer route tests
  D-FOU robust native efficiency
  D-FOU basis-state operator efficiency

GPU1:
  F1 operator-only vs anchor-retention matrix
  F2 new LIO operator families
  adapter holdout / independent seed horizon

GPU2:
  D-CHE robust native efficiency
  K18/K19/K20 D-CHE basis-state carrier
  D-CHE source-loss / basis-state retention

GPU3:
  D-RBF/D-RAT generic backward attempts
  task-level readback
  controls, figures, packet generation
```

## 13.2 Dynamic refill

```text
If GPU0 finishes D-FOU early:
  take D-CHE K18 projection audit;
  then D-RBF generic backward smoke.

If GPU1 finishes horizon early:
  run independent seed 2214/2215;
  then holdout adapter;
  then source-state artifact validation.

If GPU2 finishes D-CHE basis audit early:
  run D-FOU K19 basis-state operator;
  then readout-to-basis transfer K20.

If GPU3 finishes task readback early:
  run extra controls;
  then artifact non-empty gate;
  then finalizer.
```

## 13.3 Queue artifacts

必须输出且 route 读取：

```text
v22_14_runnable_queue.csv
v22_14_gpu_assignment_manifest.csv
v22_14_gpu_utilization_timeline.csv
v22_14_idle_violation.csv
v22_14_queue_drain_report.json
v22_14_deferred_items.csv
```

硬标准：

```text
execution_contract_violation = 0
queue_drained = 1
if violation = 1:
  official_promotion_allowed = 0
```

---

# 14. v22.14 required artifacts

## 14.1 Code truth

```text
v22_14_code_review_packet.zip
v22_14_required_source_files.csv
v22_14_compileall_report.csv
v22_14_import_closure.csv
v22_14_semantic_firewall.csv
v22_14_operator_role_blind_tests.csv
v22_14_adapter_renaming_tests.csv
v22_14_operator_law_tests.csv
v22_14_corrected_layout_tests.csv
v22_14_promotion_semantics_tests.csv
v22_14_execution_contract_tests.csv
v22_14_artifact_nonempty_gate.csv
```

## 14.2 Efficiency

```text
v22_14_native_efficiency_truth_table.csv
v22_14_variant_robust_efficiency_matrix.csv
v22_14_native_kernel_gradcheck.csv
v22_14_official_fused_status_matrix.csv
v22_14_basis_state_operator_step_efficiency.csv
v22_14_DFOU_officialization_matrix.csv
v22_14_DCHE_officialization_matrix.csv
v22_14_DRBF_DRAT_generic_backward_matrix.csv
```

## 14.3 FU operator

```text
v22_14_operator_atom_matrix.csv
v22_14_metric_geometry_matrix.csv
v22_14_variational_operator_solve.csv
v22_14_operator_commit_matrix.csv
v22_14_operator_only_vs_anchor_matrix.csv
v22_14_adapter_horizon_matrix.csv
v22_14_adapter_holdout_matrix.csv
v22_14_source_state_dynamics.csv
v22_14_source_loss_boundary_matrix.csv
v22_14_control_attribution_matrix.csv
v22_14_independent_replication_matrix.csv
```

## 14.4 KAN carrier

```text
v22_14_K18_basis_tangent_coverage.csv
v22_14_K19_basis_state_operator.csv
v22_14_K20_readout_to_basis_transfer.csv
v22_14_KAN_mapping_matrix.csv
v22_14_KAN_vs_MLP_same_operator.csv
v22_14_basis_vs_readout_ablation.csv
v22_14_basis_state_source_dynamics.csv
```

## 14.5 Task readback

```text
v22_14_task_eval_matrix.csv
v22_14_convergence_speed_matrix.csv
v22_14_forgetting_readback_matrix.csv
v22_14_expression_metrics_matrix.csv
v22_14_calibration_debt_matrix.csv
```

---

# 15. Final routes for v22.14

v22.14 finalizer 必须选择一个 route：

```text
R0-CodeTruthFailed
R1-OperatorSemanticGateFailed
R2-ArtifactOrExecutionContractFailed
R3-EfficiencyVariantRobustnessBlocked
R4-OperatorOnlyNoGo_AuxiliaryAnchorOnly
R5-OptimizerProxAnchorNoGo
R6-RoleBlindOperatorStrictExplorationPass
R7-KANBasisTangentCoverageNoGo
R8-KANBasisStateReachableButNotRetained
R9-KANBasisStateRetainedButNotCompetitiveWithMLP
R10-KANBasisCarrierExplorationPass
R11-KANBackpropImprovedButMLPNotBeaten
R12-OfficialOperatorAndCarrierReady_TaskEvidencePending
R13-FullScientificPromotionReady
```

Promotion definitions：

## 15.1 Exploration promotion

```text
S0 pass
execution_contract_violation = 0 or official flag = 0
role-blind operator pass
C3 >= 3 non-random adapters
C4 >= 2 non-random adapters in any anchor mode
controls fail
```

## 15.2 Strict official operator promotion

```text
Exploration promotion
uses_loss_modification_for_retention = 0
source_state_dynamics non-empty
independent seeds >=2/3 pass
C4 includes at least one non-MSE adapter
holdout adapter C3 pass
StableRandom/RandomMatched fail
```

## 15.3 Official KAN carrier promotion

```text
Strict official operator promotion
D-FOU or D-CHE variant_robust_efficiency_pass = 1
basis_channel_energy >= 0.50
KAN_source_loss_h4800 >= -1e-6
KAN_specific_delta_vs_MLP_same_metric >= 0.005
full_operator_step_ratio_vs_MLP <= 1.35
memory_ratio_vs_MLP <= 1.20
independent seeds >=2/3 pass
```

## 15.4 Full scientific promotion

```text
Official KAN carrier promotion
KAN+FU beats same-param MLP on >=7/9 task rows
KAN+FU beats KAN+AdamW on >=8/9 task rows
source forgetting lower than controls by >=25%
AUC_loss_time_ratio_vs_best_control <= 1.05
LineC/ECE/Brier/NLL debt not exploded
```

---

# 16. v22.14 的最小有效进展要求

v22.14 不能返回模糊 no-go。至少要交付以下之一：

```text
A-CodeTruth:
  finalizer 正确读取 execution contract；source_state_dynamics 非空；anchor/loss-modification 标记完整。

B-Efficiency:
  D-FOU/D-CHE 至少一个 variant-level robust official path；basis-state operator-step timing 落盘。

C-Operator:
  operator-only / auxiliary-anchor / optimizer-prox-anchor 三者拆清，知道 v22.13 pass 的来源。

D-KAN:
  K18 basis tangent coverage 解释 KAN mismatch，或 K19/K20 打开 basis-state source。

E-Task:
  KAN+FU 至少稳定超过 KAN+AdamW，并定位距离 MLP 的主要差距。
```

如果只完成 A+B，但 C/D 没进展，也有价值；因为 v22.13 的最大不确定性会被清掉。如果 C 证明 auxiliary anchor 是唯一有效机制，也有价值；因为我们就知道下一步必须从 optimizer-prox/source-state theory 突破，而不是继续调 operator scale。

---

# 17. 最后结论

v22.13 的最大启发是：

$$
\boxed{
\text{loss-interface operator 这条路是对的，}
\text{metric/control-null 也终于开始以正确方式发挥作用；}
\text{但 source retention 不能靠 readout replay 或 auxiliary loss 一直撑着。}
}
$$

v22.14 应该把问题推进到更本质的一层：

$$
\boxed{
\text{Functional update 必须成为一个 } (\delta, G_\theta, s_t) \mapsto U_t
\text{ 的训练动力学算子，}
\text{并且 KAN 的 basis-state channel 必须能承载这个 source。}
}
$$

如果 v22.14 成功，我们才可以开始写：

```text
FU 不只是构造 output displacement，
而是改变纯梯度反传的训练动力学：
  更快进入 signal channel；
  source 不被 optimizer 洗掉；
  KAN basis carrier 能高效承载 retained source。
```

如果 v22.14 失败，也会是非常有价值的失败，因为它会明确告诉我们：

```text
1. v22.13 的 success 到底来自 operator，还是来自 auxiliary anchor；
2. KAN basis tangent 是否根本覆盖不了当前 T(delta) target；
3. D-FOU/D-CHE efficiency 是否已经足够，不再是主 blocker；
4. 下一步是否要从 operator design 转向 source-state / basis-state representation theory。
```
