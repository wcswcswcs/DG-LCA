# DG-KAN v17.1 三段式完整计划：代码审计 + 基函数效率突破 + AdamW-free / Multi-Mechanism Functional Update

> 版本：v17.1 revised execution plan  
> 生成时间：2026-06-02  
> 适用方式：从 v17.1 开始，每一轮实验复盘和下一步计划都必须固定回答三件事：  
> **1）代码审计是否正确、是否和计划一致；2）基函数效率是否优化、当前表现如何；3）functional update 是否推进，若没有推进，得到什么 insight，下一步怎么推进。**  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬边界：strict FC-PureKAN；no active B-spline official path；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit / gate / debt readback，不能作为方向源。

---

# 0. 这版计划先说清楚：以后每一轮必须固定三段式汇报

用户已经明确要求：每个计划和每个实验审核，都必须回答以下三部分。

```text
Part A: 代码审计
  代码是否正确？
  是否能自包含 import / compile？
  LineC、source、retention、debt、route、efficiency profiler 是否按计划实现？
  代码实现是否和计划语义一致？
  有没有指标实现错误导致我们在错误湖里原地打转？

Part B: 基函数效率
  每个 basis 和同参数量 MLP 比，forward / backward / optimizer update / functional update / memory 到底如何？
  是 forward 慢、backward 慢、update 慢、audit 污染，还是显存问题？
  本轮是否真正优化了 basis kernel，而不是只测了一张 census 表？

Part C: Functional update
  FU 是否推动了？
  如果没有推动，是 source 不存在、source 不留存、debt 不恢复、controls 解释、AdamW 洗掉 source，还是 carrier 不适合？
  得到什么 insight？
  下一步怎么推进？
```

从 v17.1 开始，任何 final report、route decision、no-go boundary、next plan 都必须按这三部分组织。不能只给一个 route 名字，也不能只写一堆 method token。

---

# 1. 项目总目标与当前真实状态

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找一个局部 source 正的 update。项目目标是：

$$
\boxed{
\text{构建一个 strict FC-PureKAN / KAN-like efficient model，}
\text{其 forward/backward/update/memory 接近同参数量 MLP，}
\text{并通过 functional update 得到比普通梯度训练更好的长期训练动力学。}
}
$$

更具体地说，最终想证明：

$$
\boxed{
\text{PureKAN base + functional update}
>
\text{PureKAN base + ordinary backprop / standard optimizer controls}
}
$$

同时也必须和同参数量 MLP 对齐：

```text
1. forward 不应明显慢于 same-param MLP；
2. backward 不应明显慢于 same-param MLP；
3. optimizer update / functional update overhead 不能把 step 拖垮；
4. backward peak memory 不应明显高于 same-param MLP；
5. functional update 的收益必须超过 AdamW / SGD / Momentum / random / NoOp / matched controls；
6. 如果 MLP-FU 也成功，必须写成 generic training-dynamics insight，不能写成 KAN-specific；
7. 只有 KAN carrier 明显优于 same-param MLP same-mechanism control，才可以讨论 KAN-specific functional advantage。
```

当前真实状态：

```text
1. v17 已完成较大规模执行：S0、LineC/update/kernel correctness gate、full carrier x mechanism h100 screen、945-row h800/h1600 extension、mandatory efficiency census、code review packet、4GPU shard execution都有覆盖。
2. v17 route 是 R8-CarrierSpecificPartialPositive，但 best source 字段是 single-row max，不是 carrier x mechanism 的 9-row mean。
3. v17 不能写成 capability progress，因为 S4 real-transfer 和 S5 official 都没有执行/通过。
4. v17 的 source / retention / debt / route 字段仍存在解释风险，必须在 v17.1 先修正。
5. v16.4.1 / v17 都显示：局部 source 有时出现，但 source 没有稳定留存到长期 horizon，KAN-specific advantage 仍没有站住。
6. basis efficiency 方面已有 truth table 雏形，但很多 basis 仍没有完整 raw phase matrix 和 family-specific kernel repair closure。
```

因此 v17.1 的本质不是“继续加一个 functional update 名字”，而是：

$$
\boxed{
\text{先把代码、指标、route、efficiency 的尺子修正；}
\text{再用 AdamW-free / AdamW-optional 的 FU 全矩阵裁决机制；}
\text{同时推进 basis kernel efficiency。}
}
$$

---

# 2. 各条线当前进展百分数

| 线 | 当前完成度 | 判断 |
|---|---:|---|
| 代码 compile / basic S0 | **95%-98%** | v17 S0 过，但还不能替代 deeper correctness；LineC / route / retention / debt semantics 仍需复核 |
| LineC exception policy | **85%-90%** | MeasurementInvalid 路径已改善；但 LineC-fast 仍只是 cheap audit，不等于完整 signal-channel geometry |
| source / retention / route aggregation | **50%-60%** | v17 有 source 表，但 single-row max 和 retention 字段定义可能误导，需要重写 |
| debt accounting | **30%-40%** | v16 有动态几何概念，v17 extension 主要看 source，tail/LineC/calibration/AUC debt recovery 还不完整 |
| FU update semantics | **70%-80%** | UpdateTensor 有进展，但 step-like / gradient-like / cotangent-like / function-space 仍需严格区分 |
| AdamW coupling audit | **60%-70%** | 已有 coupling / overwrite 方向，但 AdamW-free FU 没作为主线充分裁决 |
| MLP functional dynamics | **35%-45%** | 当前 strongest source 常来自 MLP；必须作为 active line，不是 control |
| D-CHE functional dynamics | **20%-30%** | 有少量 source，但 h1600 / debt recovery / KAN-specific 没闭合 |
| LQ reanchor / late attach | **25%-35%** | 历史有价值；当前不能 official，但必须保留 reanchor + smoke |
| Rational / D-RAT monitor | **20%-30% functional / 70%-80% monitor** | monitor 可用；reset/controller/action route 不重启 |
| D-FOU efficiency / substrate | **30%-40%** | step/memory 局部可承受，forward/basis eval 是主瓶颈，kernel repair 优先 |
| D-RBF / FastKAN | **20%-30%** | active-center / local support 方向有价值，但当前效率和 task-health 未稳 |
| D-WAV | **15%-20%** | 低预算保留，重点是 sparse support / local backward |
| Efficiency truth table | **45%-50%** | 有 phase timing，但 raw full matrices、audit separation、fused-status、same-param mapping 仍需硬化 |
| Basis kernel repair | **10%-15%** | 现在还偏 census，不够 repair；v17.1 必须 family-specific repair |
| 4GPU dynamic queue | **60%-70%** | 有 shard / drain 记录；真正 work stealing / dynamic refill 未完整 |
| S4 / S5 scientific success | **0%** | 尚未达成 |
| 整体 next-gen MLP claim | **25%-33%** | 有执行成熟度和局部线索；没有 KAN-specific functional success，也没有 full efficiency closure |

---

# 3. v17.1 固定报告模板：每轮都必须这样写

从 v17.1 开始，每轮实验复盘必须包含以下三大段。下面不是提纲，而是硬报告合同。

## 3.1 Part A：代码审计报告必须回答的问题

每轮必须明确回答：

```text
A1. 本轮代码是否能 clean compile / import？
A2. 所有核心 runner 是否能从 code packet 自包含 import？
A3. 本轮关键指标实现在哪里？是否有源码行 / function 名 / test 名？
A4. LineC 是否按计划实现？异常是否被记为 MeasurementInvalid，而不是 geometry fail？
A5. source、retention、debt、route aggregation 是否按数学定义实现？
A6. update 的 sign / kind / space 是否正确？
A7. AdamW coupling 是否被显式标注？是否存在 FU 实际上通过 AdamW.step() 提交的情况？
A8. 多机制矩阵是否真的语义不同，还是只是 token 不同、实现同一个 branch？
A9. efficiency profiler 是否把 forward/backward/update/functional/audit/horizon readback 分开？
A10. 代码实现是否和计划要求一致？若不一致，是 deliberate deviation、bug，还是 budget-deferred？
```

报告中必须给出：

```text
compileall_ok
import_error_count
linec_golden_pass_count / total
update_semantics_pass_count / total
retention_formula_pass
route_aggregation_pass
adamw_coupling_rows
semantic_alias_rows
efficiency_profiler_correctness_pass
kernel_gradcheck_pass_count / total
code_plan_mismatch_count
measurement_invalid_count
```

如果这段不完整，final route 不能写成科学 no-go。

## 3.2 Part B：基函数效率报告必须回答的问题

每轮必须明确回答：

```text
B1. 每个 basis 和 same-param MLP 的 forward ratio 是多少？
B2. backward ratio 是多少？
B3. optimizer update ratio 是多少？
B4. functional direction / projection / commit overhead 是多少？
B5. backward peak memory ratio 是多少？
B6. basis activation bytes、functional state bytes、optimizer state memory 分别是多少？
B7. 慢在哪里：forward、backward、update、functional、audit、horizon readback，还是 memory？
B8. 本轮是否不仅测了 census，还执行了 family-specific efficiency repair？
B9. repair 后哪些 phase 改善，哪些没改善？
B10. 哪个 basis 具备进入 next functional proof 的工程可能？
```

报告中必须给出：

```text
v17_1_efficiency_truth_table.csv
v17_1_efficiency_blocker_table.csv
v17_1_family_repair_summary.csv
v17_1_same_param_mlp_mapping.csv
v17_1_audit_cost_separation.csv
v17_1_kernel_gradcheck_results.csv
v17_1_efficiency_dashboard.svg
```

如果没有完整 truth table，不能说“basis 不行”。只能说：

```text
EfficiencyUnknown / MeasurementIncomplete
```

## 3.3 Part C：functional update 报告必须回答的问题

每轮必须明确回答：

```text
C1. FU 是否带来了 source？是 single-row source，还是 carrier x mechanism 9-row mean source？
C2. source 是否留存？h100 -> h800 -> h1600 -> h3200 的 retention 是多少？
C3. tail / LineC / calibration / AUC debt 是否恢复？
C4. controls 是否解释 positive？
C5. MLP same-mechanism 是否也 positive？如果是 generic insight 还是 KAN-specific？
C6. AdamW 是否洗掉 FU source？FU + AdamW、FU + SGD/Momentum、FU-primary、FU-only 的差异是什么？
C7. 如果 FU 没推动，得到的 insight 是什么？下一步是 AdamW-free、slow-state、matrix-block、PopRisk/SNR，还是 carrier repair？
C8. 是否有明确 no-go？还是只是当前机制未充分实现？
```

报告中必须给出：

```text
v17_1_functional_source_matrix.csv
v17_1_source_retention_matrix.csv
v17_1_debt_recovery_matrix.csv
v17_1_control_attribution.csv
v17_1_kan_vs_mlp_attribution.csv
v17_1_adamw_overwrite_diagnostics.csv
v17_1_mechanism_noncollapse.csv
v17_1_functional_no_go_boundary.md
v17_1_next_hypothesis_queue.md
```

如果 source 没有推动，报告必须写清楚是哪一种：

```text
SourceAbsent
SourceSingleRowOnly
SourceWashedOut
DebtNotRecovered
ControlEquivalent
MLPGenericOnly
KANCarrierFailed
EfficiencyPolluted
ImplementationInvalid
```

---

# 4. v17.1 第一阶段：S0.1 代码审计硬门

v17.1 第一阶段不是跑科学实验，而是跑代码与指标正确性。S0.1 不通过时，不能进入 functional update / basis efficiency route。否则会继续用错误指标和错误实现原地打转。

## 4.1 Codex 必须打包给审查的文件

Codex 必须生成：

```text
v17_1_code_review_packet.zip
```

压缩包根目录必须包含：

```text
00_README.md
packet_manifest.csv
packet_sha256_manifest.csv
```

并包含以下目录。

### 4.1.1 `01_ENVIRONMENT/`

必须包含：

```text
python_version.txt
pip_freeze.txt
conda_env_export.yml
cuda_version.txt
torch_version.txt
gpu_info.txt
git_commit.txt
git_status.txt
run_host.txt
```

`git_status.txt` 不能只写 clean / dirty，必须列出 changed files。若有本轮修改外的旧 dirty files，必须标为：

```text
preexisting_dirty_file = 1
```

### 4.1.2 `02_SOURCE_TREE/`

必须包含源文件快照。至少包含：

```text
dgkan/models/fc_purekan_primitives.py
dgkan/models/fc_purekan_lq.py
dgkan/kernels/fused_hinge_quadratic.py
dgkan/diagnostics/basis_workspace.py
dgkan/functional/mlp_functional.py
dgkan/metrics/linec.py
dgkan/metrics/dynamic_geometry.py
dgkan/functional/update_tensor.py
dgkan/functional/functional_mechanisms.py
dgkan/functional/manual_optimizers.py
dgkan/functional/adamw_coupling_audit.py
dgkan/profiling/efficiency_profiler.py
dgkan/profiling/kernel_gradcheck.py
experiments/run_v17_1_code_correctness_gate.py
experiments/run_v17_1_functional_matrix.py
experiments/run_v17_1_basis_efficiency_repair.py
experiments/run_v17_1_finalize.py
experiments/run_v17_common.py
experiments/run_v17_full_codeaudit_adamwfree_fu_basis_efficiency_4gpu.py
experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py
experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py
experiments/run_v149_line_d_all_basis_substrate_repair.py
```

如果某些历史文件被拆分或重命名，必须提供：

```text
compatibility_shim_manifest.csv
symbol_renaming_manifest.csv
```

### 4.1.3 `03_IMPORT_CLOSURE/`

必须包含：

```text
compileall_report.txt
import_closure_results.csv
missing_module_report.csv
runner_import_results.csv
symbol_resolution_table.csv
```

要求：

```text
compileall_ok = 1
core_runner_import_error_count = 0
missing_module_count = 0
```

如果有 missing module，不能用 skip / mock 绕过；必须修复或把 S0.1 判 fail。

### 4.1.4 `04_LINEC_CORRECTNESS/`

必须包含：

```text
linec_source_readback.md
linec_symbol_map.csv
linec_golden_tests.py
linec_golden_results.csv
linec_exception_policy.csv
linec_measurement_invalid_examples.csv
linec_fast_vs_channel_comparison.csv
```

LineC golden tests 必须覆盖：

```text
NoOp:
  expected: source≈0, no false positive, valid=1

RandomMatchedNorm:
  expected: no systematic positive, valid=1

KnownTransfer:
  expected: CouplingR2 positive, split transfer positive

NoiseLeak:
  expected: NoiseSignalLeak worsens / leak detected

ReservoirOnly:
  expected: reservoir ratio high, transfer low

BatchPermutation:
  expected: coupling drops

ExceptionPath:
  expected: MeasurementInvalid, not geometry_fail

ScaleInvariance:
  expected: harmless output scale does not create fake source

ControlEquivalence:
  expected: matched control positive marks control_equivalent
```

LineC 规则：

```text
MeasurementInvalid 不进入 pass/fail mean。
MeasurementInvalid 会阻断 final route。
LineC exception 不得被写成 bad geometry。
LineC 只做 audit / gate / debt readback，不生成 direction。
```

### 4.1.5 `05_UPDATE_SEMANTICS/`

必须包含：

```text
update_tensor_source_readback.md
update_tensor_kind_contract.csv
update_sign_finite_difference_tests.py
update_sign_finite_difference_results.csv
update_space_contract.csv
update_writeback_trace.csv
rollback_trace.csv
```

必须把 update 分成：

```text
gradient_like:
  表示 dL/dtheta，optimizer 需要做 -lr * grad。

step_like:
  表示直接加到参数上的 delta theta。

cotangent_like:
  表示 output/function-space cotangent，不能直接写参数。

function_space:
  表示 function displacement，需要 projection / pullback 后才能写参数。
```

必须测试：

$$
L(\theta + \epsilon u_{step}) < L(\theta)
$$

当 $u_{step}$ 是预期下降方向时成立。若符号反了，S0.1 fail。

### 4.1.6 `06_ADAMW_COUPLING_AUDIT/`

必须包含：

```text
adamw_coupling_source_readback.md
adamw_coupled_parameter_map.csv
fu_submit_path_map.csv
adamw_overwrite_diagnostics.csv
fu_vs_adamw_cosine.csv
fu_vs_sgd_momentum_cosine.csv
optimizer_state_touch_map.csv
```

必须标明每个 FU method：

```text
uses_adamw_step_for_commit
writes_to_grad_then_adamw_step
manual_step_commit
fu_primary_commit
fu_only_commit
alternating_commit
role_partition_commit
```

v17.1 的原则是：

$$
\boxed{
\text{backprop / loss-interface signal 可以用，AdamW 不再默认必须用。}
}
$$

AdamW 只能是一个条件，不是默认主干。

### 4.1.7 `07_FUNCTIONAL_MECHANISM_NONCOLLAPSE/`

必须包含：

```text
mechanism_semantic_contract.csv
mechanism_source_readback.md
mechanism_noncollapse_tests.py
mechanism_noncollapse_results.csv
semantic_alias_matrix.csv
candidate_to_internal_path_map.csv
```

如果两个 method token 最终执行完全相同的 update path，必须标为：

```text
semantic_alias = 1
```

并且不能把它们当成两套独立机制。

### 4.1.8 `08_EFFICIENCY_PROFILER_CORRECTNESS/`

必须包含：

```text
efficiency_profiler_source_readback.md
phase_timing_unit_tests.py
phase_timing_unit_test_results.csv
audited_timing_phase_definitions.csv
audit_cost_separation_tests.csv
memory_profiler_tests.csv
same_param_mlp_mapping_tests.csv
```

必须区分：

```text
forward_only_ms
backward_grad_ms
optimizer_update_ms
functional_direction_ms
functional_projection_ms
functional_commit_ms
linec_audit_ms
tail_calibration_audit_ms
horizon_readback_ms
step_total_ms
```

如果 `linec_audit_ms` 或 `horizon_readback_ms` 混入 training step time，必须标记：

```text
audit_cost_polluted = 1
```

### 4.1.9 `09_KERNEL_GRADCHECK/`

必须包含：

```text
kernel_gradcheck_results.csv
basis_forward_correctness.csv
basis_backward_correctness.csv
fused_kernel_status.csv
dense_materialization_audit.csv
manual_vs_autograd_gradcheck.csv
```

每个 basis 都要有：

```text
forward_relerr
backward_relerr
grad_cosine
finite_rate
dense_basis_materialized
no_materialize_claim_allowed
official_fused_kernel_complete
```

如果 `official_fused_kernel_complete=0`，不能写 official kernel efficiency success。

### 4.1.10 `10_RAW_EXPERIMENT_MATRICES/`

必须包含 raw matrices，不允许只有 summary：

```text
raw_h100_matrix.csv
raw_h400_matrix.csv
raw_h800_matrix.csv
raw_h1600_matrix.csv
raw_h3200_matrix.csv
raw_controls_matrix.csv
raw_efficiency_matrix.csv
raw_linec_debt_matrix.csv
raw_tail_calibration_debt_matrix.csv
```

每个 raw row 必须包含：

```text
carrier
mechanism
dataset
seed
horizon
control_group
source_vs_best_control
source_vs_adamw
source_vs_sgd
source_vs_mlp_same_mechanism
source_retention_h800_over_h100
source_retention_h1600_over_h800
tail_debt_peak
tail_debt_final
tail_recovery_rate
LineC_debt_peak
LineC_debt_final
LineC_recovery_rate
AUCtime_ratio
CEp99_delta
NLL_delta
ECE_delta
Brier_delta
step_ratio
memory_ratio
```

### 4.1.11 `11_FIGURES_AND_DASHBOARDS/`

必须包含：

```text
code_audit_dashboard.svg
efficiency_phase_dashboard.svg
basis_efficiency_pareto.svg
source_retention_curves.svg
debt_recovery_curves.svg
control_attribution_heatmap.svg
kan_vs_mlp_attribution.svg
adamw_overwrite_dashboard.svg
gpu_utilization_dashboard.svg
```

### 4.1.12 `12_FAILURE_TAXONOMY/`

必须包含：

```text
failure_taxonomy.csv
no_go_boundary.md
next_hypothesis_queue.md
exhaustion_certificate.csv
```

Failure taxonomy 必须使用固定类：

```text
ImplementationInvalid
MetricInvalid
RouteAggregationInvalid
EfficiencyUnknown
ForwardBlocked
BackwardBlocked
UpdateBlocked
MemoryBlocked
AuditCostPolluted
SourceAbsent
SourceSingleRowOnly
SourceWashedOut
DebtNotRecovered
ControlEquivalent
MLPGenericOnly
KANCarrierFailed
AdamWWashesFU
SemanticAlias
BudgetDeferred
```

### 4.1.13 `13_REPRO_COMMANDS/`

必须包含：

```text
run_all.sh
run_s0_code_gate.sh
run_linec_tests.sh
run_update_semantics_tests.sh
run_efficiency_census.sh
run_functional_matrix.sh
run_basis_repair.sh
reproduce_route.sh
```

---

## 4.2 S0.1 成功标准

S0.1 必须全部满足：

```text
compileall_ok = 1
core_runner_import_error_count = 0
linec_golden_pass = 1
update_semantics_pass = 1
source_retention_formula_pass = 1
debt_accounting_formula_pass = 1
route_aggregation_unit_pass = 1
efficiency_profiler_correctness_pass = 1
kernel_gradcheck_exploration_pass = 1
semantic_alias_unexplained_count = 0
packet_manifest_missing_count = 0
```

如果 S0.1 fail，Codex 不能继续跑大实验。它必须先按 blocker 修复：

```text
ImportClosureFail:
  补齐 missing modules / compatibility shims / symbol maps。

LineCInvalid:
  修 LineC 实现和 golden tests，不得把 exception 写成 geometry fail。

UpdateSignInvalid:
  统一 UpdateTensor.kind 和 apply semantics。

RetentionFormulaInvalid:
  修 retention 公式和 route readback。

DebtAccountingInvalid:
  补 tail/LineC/calibration/AUC debt peak/final/recovery。

EfficiencyProfilerInvalid:
  分离 audit/horizon readback cost，重测 phase timing。

SemanticAliasInvalid:
  删除 alias 或标记 smoke，不得冒充 multi-mechanism。
```

---

# 5. v17.1 第二阶段：基函数效率主线

## 5.1 目标

基函数效率不是附属问题。DG-KAN 如果要作为 next-gen MLP alternative，basis 的 forward / backward / update / memory 必须和 same-param MLP 匹敌。可以略慢，但不能慢太多，更不能靠巨大 forward/backward 开销换取局部 source。

v17.1 的效率目标分三层。

### 5.1.1 Exploration envelope

$$
\frac{T_{forward,basis}}{T_{forward,MLP}} \le 1.75
$$

$$
\frac{T_{backward,basis}}{T_{backward,MLP}} \le 1.75
$$

$$
\frac{T_{step,basis}}{T_{step,MLP}} \le 1.75
$$

$$
\frac{M_{backward,basis}}{M_{backward,MLP}} \le 1.25
$$

### 5.1.2 Strong engineering envelope

$$
\frac{T_{forward,basis}}{T_{forward,MLP}} \le 1.35
$$

$$
\frac{T_{backward,basis}}{T_{backward,MLP}} \le 1.40
$$

$$
\frac{T_{step,basis}}{T_{step,MLP}} \le 1.35
$$

$$
\frac{M_{backward,basis}}{M_{backward,MLP}} \le 1.05
$$

### 5.1.3 Official efficiency envelope

$$
\frac{T_{forward,basis}}{T_{forward,MLP}} \le 1.25
$$

$$
\frac{T_{backward,basis}}{T_{backward,MLP}} \le 1.30
$$

$$
\frac{T_{step,basis}}{T_{step,MLP}} \le 1.25
$$

$$
\frac{M_{backward,basis}}{M_{backward,MLP}} \le 1.00
$$

若 official fused kernel 未完成，最多只能写 exploration efficiency，不允许写 official efficiency success。

---

## 5.2 必须比较的 carrier

```text
C0: same-param MLP reference
C1: D-CHE
C2: LQ
C3: Rational / D-RAT
C4: D-FOU
C5: D-RBF / FastKAN
C6: D-WAV
C7: current best FHQ / B320/B314 anchor if available as historical reference only
```

每个 carrier 必须和 same-param MLP 对齐：

```text
same trainable parameter count within ±5%
same hidden width family if possible
same batch size grid
same dtype
same train/eval mode
same loss interface
same optimizer family for efficiency-only measurement
```

---

## 5.3 Phase-level efficiency truth table

每个 carrier x batch size x implementation variant 必须记录：

```text
batch_size
hidden_dim
param_count
same_param_mlp_param_count
param_count_ratio
forward_only_ms
backward_grad_ms
optimizer_update_ms
functional_direction_ms
functional_projection_ms
functional_commit_ms
linec_audit_ms
tail_calibration_audit_ms
horizon_readback_ms
step_total_ms
forward_peak_memory_mb
backward_peak_memory_mb
optimizer_state_memory_mb
basis_activation_bytes
functional_state_bytes
temp_workspace_bytes
kernel_count_forward
kernel_count_backward
num_gemm_calls
num_trig_calls
num_exp_calls
num_div_calls
num_scatter_calls
num_gather_calls
dense_basis_materialized
official_fused_kernel_complete
audit_cost_polluted
```

Batch grid：

```text
batch_size = 8, 32, 128, 256
```

如 256 超预算，可 budget-defer，但 8/32/128 必须完成。

---

## 5.4 Family-specific efficiency repair

### 5.4.1 D-CHE repair

已知风险：D-CHE 可能同时混入 basis forward、degree recurrence、functional split-gradient、LineC / horizon readback 成本。v17.1 必须拆清楚。

候选 repair：

```text
CHE-R0 current reference
CHE-R1 recurrence no-materialize forward
CHE-R2 fused degree eval
CHE-R3 analytic backward degree recurrence
CHE-R4 low-degree active bank
CHE-R5 fused degree-role projection update
CHE-R6 audit/horizon readback separated timing
CHE-R7 cache degree basis across micro-splits, diagnostic only
```

必须记录：

```text
degree_basis_eval_ms
degree_recurrence_ms
readout_mix_ms
degree_projection_ms
linec_pack_ms
horizon_readback_ms
no_materialize_basis = 0/1
max_materialized_tensor_shape
```

如果 `CHE-R6` 显示 slow 主要来自 audit/horizon readback，不能把它写成 training inefficiency。必须拆成：

```text
training_step_ratio
audit_augmented_step_ratio
```

### 5.4.2 D-FOU repair

已知风险：D-FOU 当前局部 step/memory 可承受，但 forward ratio 偏高。主攻 forward。

候选 repair：

```text
FOU-R0 current reference
FOU-R1 sincos precompute
FOU-R2 recurrence sincos
FOU-R3 fused band eval
FOU-R4 low-frequency active bank
FOU-R5 table lookup diagnostic
FOU-R6 band-readout fused backward
FOU-R7 high-frequency quarantine no-materialize
```

必须记录：

```text
sincos_eval_ms
band_mix_ms
band_readout_ms
band_backward_ms
low_mid_high_band_energy
high_freq_ratio
phase_drift
bandwise_snr
```

如果 forward 仍 >1.75，但 step/memory 可承受，应写：

```text
ForwardKernelBlocked
```

而不是 FunctionalNoGo。

### 5.4.3 LQ repair

LQ 历史有 near-pass 和 repaired anchor 线索，但效率和 reanchor 不稳。v17.1 要把 LQ 作为 monitor + late attach candidate，而不是直接丢。

候选 repair：

```text
LQ-R0 current/reanchor reference
LQ-R1 Legendre recurrence
LQ-R2 fixed-frame forward cache
LQ-R3 fused projection update
LQ-R4 projection update foreach/persistent optimizer
LQ-R5 late attach fixed frame smoke
LQ-R6 low-rank frame active columns
```

必须记录：

```text
legendre_eval_ms
projection_forward_ms
projection_update_ms
frame_cache_hit_rate
active_frame_rank
macro_delta_vs_mlp
near_pass_count
```

### 5.4.4 Rational / D-RAT repair

Rational 现在是 monitor，不重启 reset/controller/action，但可以做 minimum FU smoke 和 efficiency repair。

候选 repair：

```text
RAT-R0 current reference
RAT-R1 Horner numerator/denominator eval
RAT-R2 branchless denominator safety
RAT-R3 grouped vectorized rational eval
RAT-R4 derivative telemetry separated from training
RAT-R5 block update for numerator/denominator group
RAT-R6 denominator safety readback no materialize
```

必须记录：

```text
numerator_eval_ms
denominator_eval_ms
division_ms
derivative_telemetry_ms
den_min
den_p01
den_condition
r_prime_p99
r_double_prime_p99
```

### 5.4.5 D-RBF / FastKAN repair

候选 repair：

```text
RBF-R0 current reference
RBF-R1 active-center local K4
RBF-R2 compact support no dense materialization
RBF-R3 width condition guard
RBF-R4 gaussian lookup table diagnostic
RBF-R5 sparse local backward
RBF-R6 center occupancy warmup smoke
```

必须记录：

```text
active_center_fraction
empty_center_fraction
width_p01
width_p99
center_eval_ms
local_backward_ms
dense_basis_materialized
```

### 5.4.6 D-WAV repair

候选 repair：

```text
WAV-R0 current reference
WAV-R1 triangular support index-only
WAV-R2 sparse support backward
WAV-R3 scale occupancy audit
WAV-R4 support-overlap damping readback
WAV-R5 local-tail coverage audit
```

必须记录：

```text
support_size_mean
support_overlap_mean
scale_occupancy_entropy
sparse_backward_ms
dense_support_materialized
```

---

# 6. v17.1 第三阶段：Functional Update 主线

## 6.1 核心重置：FU 需要 backprop signal，但不必绑定 AdamW

v17.1 的核心变化：AdamW 不再是默认主干。我们要测试：

$$
\boxed{
\text{FU 本身是否无效，还是 AdamW + FU 的组合方式错了？}
}
$$

反传 / loss-interface 给的是任务压力：

$$
g_t = \nabla_\theta L(\theta_t)
$$

或者输出空间 cotangent：

$$
\delta_t = \nabla_f L(f_{\theta_t})
$$

AdamW 只是使用这个信息的一种 optimizer，不是 FU 必须绑定的更新方式。

---

## 6.2 Carrier matrix

v17.1 必跑 carriers：

```text
C1-D-CHE:
  当前最强 KAN carrier，full mechanism matrix。

C2-MLP:
  active functional dynamics discovery line，full mechanism matrix。
  不是普通 control。

C3-LQ:
  reanchor + promotion-disabled late attach smoke。

C4-Rational / D-RAT:
  monitor + minimum FU smoke，不重启 reset/controller/action。

C5-D-FOU:
  substrate/efficiency repair + minimum FU smoke。

C6-D-RBF/FastKAN:
  substrate/efficiency repair + minimum FU smoke。

C7-D-WAV:
  low-budget smoke + efficiency repair。
```

---

## 6.3 Mechanism matrix

每个 active carrier 至少跑下列机制。D-CHE 和 MLP 必须 full matrix；LQ/Rational/FOU/RBF/WAV 至少 smoke subset。

### M1: AdamW-primary + FU residual

旧路线，作为 baseline / control，不再默认主线。

$$
\Delta\theta = \Delta\theta_{AdamW} + \eta_{fu} u_{FU}
$$

记录：

```text
fu_vs_adamw_cosine
adamw_overwrite_cosine
source_retention
control_equivalence
```

### M2: SGD / Momentum-primary + FU

测试 AdamW 是否洗掉 FU source。

$$
\Delta\theta = \Delta\theta_{SGD/Momentum} + \eta_{fu} u_{FU}
$$

若 M1 fail 而 M2 source 留存，说明 AdamW coupling 可能是 blocker。

### M3: FU-primary

反传只提供 loss pressure，主要更新由 FU 决定。

$$
\Delta\theta = \operatorname{FU}(g_t, \text{basis state}, \text{train split state})
$$

### M4: FU-only smoke

不作为 final training，只检测 FU 是否有基本下降能力和 source channel 写入能力。

### M5: Alternating FU / gradient

不把 FU 每步混进 optimizer，而是：

```text
k steps gradient optimizer
1 step FU pulse
k steps recovery
```

比较 recovery：

```text
AdamW recovery
SGD recovery
Momentum recovery
Schedule-free / slow-state recovery
No optimizer recovery
```

### M6: Slow-state FU

解决 h800 source 到 h1600 washout。

$$
s_{t+1} = \beta s_t + (1-\beta)P_{signal}(u_t)
$$

更新：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{base,t}
+
\eta_s P_{safe}(s_t)
$$

其中 $P_{signal}$ 只能来自 train-stream split-consensus / PopRisk / SNR，不来自 validation/test/LineC/tail/AUC。

### M7: Matrix-block FU

针对 MLP hidden matrix、D-CHE degree-readout、FOU band-readout、LQ projection frame、Rational group block。

记录：

```text
block_source_norm
block_source_retention
block_update_cosine
matrix_rank_change
spectral_norm_change
```

### M8: PopRisk / SNR signal-channel FU

用 per-example gradient 的 drift-diffusion 判断 coherent signal。

对参数块 $k$：

$$
\mu_k = \frac{1}{B}\sum_i g_{i,k}
$$

$$
\sigma_k^2 = \frac{1}{B-1}\sum_i \|g_{i,k}-\mu_k\|^2
$$

$$
SNR_k = \frac{\|\mu_k\|^2}{\sigma_k^2/(B-1)+\epsilon}
$$

只允许 train-stream per-example gradient，不使用 validation/test/future。

### M9: Function-space operator FU

先定义输出函数位移，再 pullback 到参数。

$$
\alpha^* = \arg\min_\alpha L_{B1}(f_\theta + J_{B1}U\alpha) + \rho\|\alpha\|^2
$$

然后：

$$
\Delta\theta = U\alpha^*
$$

### M10: Role-partition optimizer

不同参数角色使用不同更新机制：

```text
basis params:
  FU / function-space / PopRisk update

readout / mixing params:
  SGD / momentum / matrix-block update

scale / decay params:
  decoupled shrink only

safety params:
  frozen or slow EMA
```

这个机制专门测试：过去 hybrid 架构中 AdamW 是否只是因为 non-KAN params 才必要。

---

## 6.4 Functional metrics

每个 carrier x mechanism x dataset x seed x horizon 必须记录：

```text
source_vs_best_control_h100
source_vs_best_control_h400
source_vs_best_control_h800
source_vs_best_control_h1600
source_vs_best_control_h3200
source_retention_h800_over_h100
source_retention_h1600_over_h800
source_retention_h3200_over_h1600
dataset_seed_pass_h800
dataset_seed_pass_h1600
worst_source_h1600
control_equivalent_fraction
source_vs_mlp_same_mechanism
KAN_specific_delta_vs_MLP_same_mechanism
AdamW_overwrite_cosine_h800
AdamW_overwrite_cosine_h1600
FU_vs_grad_cosine
FU_vs_AdamW_cosine
FU_vs_SGD_cosine
FU_vs_momentum_cosine
tail_debt_peak
tail_debt_final
tail_recovery_rate
LineC_debt_peak
LineC_debt_final
LineC_recovery_rate
calibration_debt_peak
calibration_debt_final
calibration_recovery_rate
AUCtime_ratio
step_ratio
memory_ratio
```

Retention 定义：

$$
Retention_{h800/h100}
=
\frac{\max(0, Source_{h800})}{\max(\epsilon, Source_{h100})}
$$

$$
Retention_{h1600/h800}
=
\frac{\max(0, Source_{h1600})}{\max(\epsilon, Source_{h800})}
$$

Debt recovery 定义：

$$
Recovery_H
=
1 - \frac{Debt_H}{Debt_{peak}+\epsilon}
$$

---

## 6.5 Functional gates

### S2: weak productive dynamics

```text
carrier x mechanism 9-row mean source_h800 >= 0.005
dataset_seed_pass_count_h800 >= 3/9
source_retention_h800_over_h100 >= 0.40
tail_recovery_rate_h800 >= 0.40 or LineC_recovery_rate_h800 >= 0.40
AUCtime_ratio_h800 <= 1.10
matched controls fail
```

### S3: productive debt recovery

```text
carrier x mechanism 9-row mean source_h1600 >= 0.005
dataset_seed_pass_count_h1600 >= 4/9
source_retention_h1600_over_h800 >= 0.50
tail_recovery_rate_h1600 >= 0.60
LineC_recovery_rate_h1600 >= 0.60
AUCtime_ratio_h1600 <= 1.05
RandomPulse + same recovery fails
RecoveryOnly fails
NoOpMatchedOverhead fails
```

### S4: real-transfer exploration

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail debt recovered
LineC debt recovered
controls fail
```

### S5: official success，不降低

```text
real_dataset_seed_pass_count = 9/9
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.25
official fused kernel gate pass where required
controls fail
promotion_allowed = 1
```

---

# 7. 4GPU 动态并行执行

v17.1 必须使用 4 张 GPU，不允许本质串行。

## 7.1 GPU primary queues

```text
GPU0:
  S0.1 code tests；D-CHE M1-M10；D-CHE kernel repair。

GPU1:
  MLP M1-M10；MLP same-param reference；MLP slow-state/matrix-block FU。

GPU2:
  LQ reanchor/smoke；Rational monitor/smoke；LQ/RAT efficiency repair。

GPU3:
  D-FOU / D-RBF / D-WAV substrate + smoke + efficiency repair。
```

## 7.2 Fallback queues

任一 GPU primary queue 完成后必须自动补：

```text
1. missing Line M controls
2. missing efficiency census rows
3. all-basis smoke rows
4. h1600/h3200 extension for top source-retaining rows
5. figure generation
6. code audit packet finalization
```

## 7.3 必须输出 queue artifacts

```text
v17_1_runnable_queue.csv
v17_1_gpu_assignment_manifest.csv
v17_1_gpu_utilization_dashboard.csv
v17_1_idle_violation.csv
v17_1_deferred_items.csv
v17_1_queue_drain_report.csv
```

硬规则：

```text
if runnable_queue_not_empty and any_gpu_idle_minutes > 10:
    execution_contract_violation = 1
    final_route_completed_no_go_allowed = 0
```

---

# 8. Stop / continue 规则

## 8.1 不能 hard stop 的情况

以下情况不能 hard stop，只能进入 fallback / taxonomy / next queue：

```text
D-CHE fail
MLP fail
LQ reanchor fail
Rational smoke fail
D-FOU/RBF/WAV < 6/9
任一 mechanism fail
h800 fail
h1600 fail
h3200 fail
MLP/generic controls positive
LineC/tail/AUC immediate fail
overhead high
random pulse explains result
AdamW washes FU source
source single-row positive only
efficiency blocked but smoke not executed
```

## 8.2 真正 hard stop

只有以下可以 hard stop：

```text
required artifact missing
compile/import closure fail
LineC golden tests fail
update semantics fail
source retention formula fail
debt accounting formula fail
efficiency profiler correctness fail
forbidden information violation
no-action-search violation
direction uses validation/test/future/query
direction uses LineC/CEp99/NLL/ECE/AUCtime/Brier
dataset-name branch
seed-specific scale
action token / controller / action bank / reset route
fake / proxy / CPU offload
```

---

# 9. 必须生成的可视化

## 9.1 代码审计

```text
code_audit_dashboard.svg
linec_golden_results.svg
update_semantics_signcheck.svg
route_aggregation_unit_tests.svg
```

## 9.2 基函数效率

```text
efficiency_phase_stacked_bar.svg
forward_ratio_by_basis.svg
backward_ratio_by_basis.svg
update_ratio_by_basis.svg
memory_ratio_by_basis.svg
basis_efficiency_pareto.svg
kernel_blocker_heatmap.svg
audit_cost_pollution_bar.svg
```

## 9.3 Functional update

```text
source_horizon_curves.svg
source_retention_heatmap.svg
debt_recovery_curves.svg
control_attribution_heatmap.svg
kan_vs_mlp_same_mechanism.svg
adamw_overwrite_cosine.svg
mechanism_noncollapse_matrix.svg
```

## 9.4 4GPU 执行

```text
gpu_utilization_dashboard.svg
queue_drain_timeline.svg
idle_violation_timeline.svg
```

---

# 10. v17.1 final route 必须如何写

最终 route 不能只写一个 `R8-PartialPositive`。必须分三段。

## 10.1 Code route

示例：

```text
CodeRoute = S0_1-CodeCorrectnessPassed
```

或：

```text
CodeRoute = R-CodeInvalid-LineCMeasurementInvalid
```

## 10.2 Efficiency route

示例：

```text
EfficiencyRoute = R-EfficiencyForwardBlocked-D-FOU
```

或：

```text
EfficiencyRoute = S2-EfficiencyExplorationEnvelopeOpened-D-FOU-LQ
```

## 10.3 Functional route

示例：

```text
FunctionalRoute = R-SourceWashedOut-MLPGenericOnly
```

或：

```text
FunctionalRoute = S2-WeakProductiveDynamics-MLP-M6SlowState
```

Final summary 必须同时写：

```text
code_audit_status
efficiency_status
functional_status
next_action_code
next_action_efficiency
next_action_functional
```

---

# 11. 最终判断

v17.1 的目标不是“再跑一个更复杂实验”。它的目标是把项目重新拉回三个硬问题：

```text
1. 代码和指标是否正确？
2. basis 是否真正接近 MLP 的效率 envelope？
3. FU 是否真的形成长期 source retention + debt recovery，而不是局部 positive？
```

如果代码和指标不正确，我们不能相信任何 no-go 或 positive。  
如果 basis 慢，我们不能把 functional 失败全归因于机制。  
如果 FU 只在 MLP 上 positive，我们要把它写成 generic training-dynamics insight，而不是 KAN-specific。  
如果 AdamW 洗掉 FU，我们要停止默认 AdamW + FU。  
如果 source 单 row positive 但 9-row mean / h1600 retention 不成立，不能写成 progress。  

v17.1 的核心句是：

$$
\boxed{
\text{每轮先审代码，再审效率，再审 functional；}
\text{三者分开结论，最后再合并路线。}
}
$$
