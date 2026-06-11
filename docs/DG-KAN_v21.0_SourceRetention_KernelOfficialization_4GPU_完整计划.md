# DG-KAN v21.0：Source-Retention Functional Update + Basis Kernel Officialization + 4GPU 动态并行完整计划

> 版本：v21.0 execution plan  
> 生成时间：2026-06-03  
> 适用阶段：v20 之后，v21 之前  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 固定复盘结构：每轮必须分成三部分：**代码审计**、**基函数效率**、**functional update**。  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no label-informed init；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / Brier / AUCtime 只能作为 audit / gate / debt readback，不能作为 direction source。

---

# 0. 项目总目标与当前真实状态

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，而是证明：

$$
\boxed{
\text{strict FC-PureKAN base + functional update}
>
\text{same-param MLP / PureKAN ordinary gradient baselines}
}
$$

这里的“大于”必须同时满足四类条件：

```text
1. 表达与任务：final task / loss / calibration 不弱于 same-param MLP。
2. 效率：forward / backward / optimizer update / functional commit / memory 接近 same-param MLP。
3. Functional causality：functional update 的收益不能被 AdamW、SGD/Momentum、random matched、NoOp overhead、recovery-only、MLP generic control 解释。
4. 动态几何：短期 bad update 可以存在，但 source 必须长期留存，tail / LineC / calibration / AUC debt 必须偿还。
```

v20 的真实状态必须更清楚地写成：

```text
代码：
  artifact 矩阵较完整，但结果包没有包含完整 .py source tree，不能算独立代码审计闭合。

效率：
  D-CHE 与 D-FOU 终于有 official-ish efficiency pass rows；这是实质推进。
  但 officialized kernel 是否真正接入 full functional training loop 尚未闭合。

Functional update：
  MLP M16 retained source 没有复现；KAN retained source = 0。
  M2 / F7 / F8 / F9 / F10 等能制造 h800/h1600 source，但 h3200/h4800 不留。
  H4 high ActuationR2 rows 多，但 source-success rows = 0，说明 actuation 不是主瓶颈，target / source observability 才是主瓶颈。
```

因此 v21.0 不再做“继续某个 FU token 小修”。它必须回答两个硬问题：

$$
\boxed{
\text{Functional update 的 source 如何进入长期可保留 signal channel？}
}
$$

以及：

$$
\boxed{
\text{D-CHE / D-FOU 的高效 kernel 是否能成为 full-loop official basis kernel？}
}
$$

---

# 1. v20 结果的独立判断：哪些是进展，哪些不是

## 1.1 代码审计判断

v20 的 artifact 自检显示 S0.4 pass、LineC/debt complete、queue/drain 完整，但上传的结果包没有包含完整 `.py` 源码。因此 v20 只能被认定为：

```text
artifact-analysis usable = 1
independent-source-code-audit-complete = 0
```

这不是小问题。我们现在已经多次被“指标实现 / route aggregation / mechanism name 与实现不一致”误导。如果下一轮仍缺源码，就不允许写：

```text
CodeRoute = CodeMetricMechanismPassed
```

只能写：

```text
CodeRoute = ArtifactOnlyReview_SourceMissing
```

v21.0 的第一硬门是：**完整源码必须打包，不打包不进入 scientific route**。

## 1.2 基函数效率判断

v20 的效率路线有实质推进。D-CHE 和 D-FOU 出现 official-ish pass rows，说明之前“所有 KAN basis 都 forward blocked”已经不准确。当前更精确的判断是：

```text
D-CHE:
  CHE20-R4-k3-gradbuf-triton-officialize 在 batch 8/32/128/256 上 forward/step/memory 已接近 MLP envelope。

D-FOU:
  FOU20-R4-k4-triton-no-materialize-officialize 在至少一个 batch 上进入 MLP-like envelope。

LQ / D-RAT / D-RBF / D-WAV:
  仍然 forward blocked，不能作为主 FU carrier。
```

因此 v21.0 不应再平均撒网。效率主线只做两个一级目标：

```text
E-main-1: D-CHE officialized kernel full-loop integration。
E-main-2: D-FOU officialized kernel full-loop integration。
```

LQ / D-RAT / D-RBF / D-WAV 只做 targeted repair smoke，不抢主预算。

## 1.3 Functional update 判断

v20 没有 functional success。关键事实是：

```text
MLP M16 retained h3200/h4800 = 0/0。
KAN retained h3200/h4800 = 0/0。
H4 high ActuationR2 rows = 15，但 source-success rows = 0。
F7/F8/F9/F10 能把 source 推到 h800/h1600，但撑不到 h3200/h4800。
```

这说明 functional update 当前不是“没有任何 source”，而是：

$$
\boxed{
\text{source 可以被制造，但没有进入长期 retained signal channel。}
}
$$

而 H4 的结果说明：

$$
\boxed{
\text{能 actuation 不等于 source target 正确。}
}
$$

所以 v21.0 的 FU 主线不再是“找更强 source”，而是：

```text
source dynamics decomposition：为什么强 source 会 washout？
source target reconstruction：什么 train-stream target 才会留存？
source-channel writer：如何把 source 写入 slow / matrix / output / basis channel？
KAN carrier adaptation：为什么 MLP 上的 source dynamics 不能迁移到 D-CHE/D-FOU？
```

---

# 2. 之前被判错的方向：哪些可能因代码/指标问题还有希望，哪些必须停止

## 2.1 可以重开的旧方向

这些方向不能按旧结论彻底关掉，因为旧实现或旧指标存在问题，或者它们的思想需要用 v21 的新语义重测。

### 2.1.1 PopRisk / SNR

旧版 PopRisk/SNR 很多时候是一次性 parameter-SNR 或简单 per-example gradient gate。v20 的 source estimator 没有给出 retained source；但这不证明 population-risk / signal-channel 思路错。新的 generalization 理论把 output space 分为 signal channel 与 reservoir，并强调 coherent population signal 通过 drift 累积，idiosyncratic noise 更像 diffusion。v21 应把 PopRisk/SNR 改成 **source-retention predictor**，而不是直接 update rule。

重开方式：

```text
1. 不直接用 parameter SNR 提交 update。
2. 用 train-stream leave-one-point transfer / cross-split transfer 构造 source score。
3. 用 h800/h1600/h3200 retention 做校准。
4. 若 Spearman <= 0，不能做 source selector，只能做 failure evidence。
```

### 2.1.2 Split-consensus / G7R lineage

旧 G7R / split-consensus 曾出现 source，但常伴随 bad event 或 washout。旧做法是直接提交 metric update。v21 只允许它作为 **source estimator** 或 **candidate target generator**，不能直接作为 final update。

重开方式：

```text
split-consensus estimates source target;
actuation solver executes target;
long-horizon source/debt tests decide whether target is useful。
```

### 2.1.3 Function-space / operator-level actuation

v20 显示 high ActuationR2 但 source-success = 0。这不能说明 function-space actuation 无效；它说明 target 错。v13.4 的 operator-level basis-channel路线也曾卡在 projection / actuation fidelity，但现在 H4 已经显示 actuation 可以很高，因此 v21 应重开 function-space route，但必须换 source target。

重开方式：

```text
不再问“能不能 actuation”；
改问“哪个 train-stream output displacement target 能长期留 source”。
```

### 2.1.4 MLP functional update

MLP 不能被降级为 control。v19 的 retained MLP source、v20 的 non-repro，都说明 MLP 是研究 FU source dynamics 的必要实验台。

重开方式：

```text
用 MLP 分解 source dynamics：M2 strong-washout、M15 weak-stable、F9/F10 hold。
如果 MLP 上都没有 retained source，KAN 上不应继续堆 carrier。
```

### 2.1.5 Graph-free analytic adjoint / manual update

早期 graph-free analytic adjoint 证明 manual gradient correctness 可行，但 primitive/kernel efficiency 当时没过。现在 D-CHE/D-FOU kernel efficiency 已经有进展，应把 graph-free / analytic backward 思路与 D-CHE/D-FOU kernel officialization 合并。

重开方式：

```text
officialized D-CHE/D-FOU kernel + analytic backward + FU commit profiler。
```

## 2.2 不应该重启的旧方向

以下方向已有明确反证，不能再浪费主预算。

```text
1. action bank / controller：过去 oracle upper bound 不足，且容易回到 v9 的“找好动作”错误。
2. reset route / optimizer-state reset：generic optimizer confound 已多次解释。
3. cover objective 小修：oracle cover objective 已 invalid。
4. G-token / N-token / Q-token 扩展：只是给旧 update 换名字。
5. M31/M32 same-family 修补：H10 partial positive 未复现，H12 negative，H13 已判 no actionable same-family repair。
6. dataset/seed branch：违反项目目标。
7. audit-metric-directed update：LineC/tail/AUC/calibration 只能 audit，不可生成 direction。
```

---

# 3. 当前研究进展对 v21 的启发

v21 不机械搬 optimizer，而是抽取训练动力学原则。

## 3.1 Schedule-free / slow iterate 的启发

Schedule-free / averaged-iterate 方法提醒我们：fast iterate 和 slow iterate 可能承担不同角色。v20 的核心失败是 h800/h1600 source 不能撑到 h3200/h4800，因此 FU 可能需要写入 slow state，而不是直接作为 fast parameter perturbation。

v21 的对应假设：

$$
\boxed{
\text{FU source 必须进入 slow source state，才不会被后续训练 washout。}
}
$$

## 3.2 AdEMAMix / long-memory gradient 的启发

历史 gradient 不一定是噪声。短期 source 和长期 source 可能需要双时间尺度：

$$
s_t^{short}=\beta_s s_{t-1}^{short}+(1-\beta_s)u_t,
$$

$$
s_t^{long}=\beta_l s_{t-1}^{long}+(1-\beta_l)u_t.
$$

v21 不把它当 optimizer 名字，而是用它测试：

```text
M2 strong source 是否能被 long state 留住？
M15 weak stable source 是否可作为 anchor？
```

## 3.3 Muon / SOAP / matrix-block optimizer 的启发

很多更新不应逐参数定义。KAN 的 D-CHE degree-readout、D-FOU band-readout、MLP hidden matrix 都是 matrix/block 结构。逐坐标 AdamW 或逐坐标 FU 可能会把 source 打散。

v21 的对应假设：

$$
\boxed{
\text{source 应在 matrix/block channel 中定义和留存，而不是在逐参数坐标中留存。}
}
$$

## 3.4 Signal channel / reservoir 理论的启发

训练运动是否有泛化意义，不应只看 train loss 或 ActuationR2。真正关键是：source 是否进入 signal channel，噪声是否留在 reservoir，训练运动能否 transfer 到其它 train split / probe split。

v21 的对应假设：

$$
\boxed{
\text{high ActuationR2 + source fail 表明 source target 错，不是执行器错。}
}
$$

## 3.5 Deep Manifold / boundary-conditioned iteration 的启发

Functional update 不一定是单步好方向，而可能是训练边界条件。短期 bad debt 可以存在，但必须在长期 fixed-point trajectory 中被偿还。

v21 的对应假设：

$$
\boxed{
\text{FU 是 trajectory boundary / source-channel writer，而不是 AdamW residual。}
}
$$

---

# 4. v21 总体实验结构

v21 分成三条互锁主线：

```text
Part A: Code / Metric / Mechanism Audit
  确保源码完整、指标定义正确、route 不再被 single-row / missing debt / prototype 名字误导。

Part B: Basis Kernel Efficiency Officialization
  把 D-CHE / D-FOU 从 v20 official-ish pass 推到 full-loop official kernel evidence。

Part C: Source-Channel Functional Update
  先理解 MLP source dynamics，再重构 KAN source-channel writer，不再继续 M31/M32 同族小修。
```

四张 GPU 并行：

```text
GPU0:
  Part A S0.5 checks + D-CHE kernel officialization + D-CHE source-channel writer。

GPU1:
  MLP source dynamics dissection + slow-state / matrix-block / schedule-free FU。

GPU2:
  D-FOU kernel officialization + D-FOU source-channel smoke。

GPU3:
  LQ/D-RAT targeted repair smoke + D-RBF/D-WAV sparse smoke + controls / figures / packet generation。
```

如果 `runnable_queue.csv` 非空但任一 GPU idle 超过 10 分钟，则：

```text
execution_contract_violation = 1
completed_no_go_allowed = 0
```

---

# 5. Part A：代码审计与证据完整性硬门

## 5.1 总目标

Part A 的目标是防止继续被错误指标/错误实现拖着原地打转。v21 必须先证明：

```text
1. 源码完整；
2. LineC / debt / retention / route aggregation 正确；
3. mechanism 名字和实现一致；
4. efficiency profiler 的 phase timing 能代表真实 full-loop；
5. kernel correctness 与 fused/no-materialize status 可审计。
```

## 5.2 Codex 必须打包的文件

Codex 必须输出：

```text
v21_code_review_packet.zip
```

包内必须包含：

```text
00_README.md
01_ENVIRONMENT/
02_SOURCE_TREE/
03_IMPORT_CLOSURE/
04_LINEC_FAST_CHANNEL/
05_RETENTION_DEBT_ROUTE/
06_UPDATE_SEMANTICS/
07_FUNCTIONAL_MECHANISM_CONTRACTS/
08_OPTIMIZER_COUPLING/
09_EFFICIENCY_PROFILER/
10_KERNEL_OFFICIALIZATION/
11_RAW_EXPERIMENT_MATRICES/
12_GPU_QUEUE/
13_FIGURES/
14_FAILURE_TAXONOMY/
15_REPRO_COMMANDS/
packet_manifest.csv
packet_sha256_manifest.csv
```

`02_SOURCE_TREE` 至少必须包含：

```text
dgkan/**/*.py
experiments/run_v21_*.py
experiments/run_v20_*.py
experiments/run_v19_*.py
```

特别要求包含：

```text
dgkan/metrics/linec.py
dgkan/fu/mechanisms.py
dgkan/fu/source_state.py
dgkan/fu/function_space_actuation.py
dgkan/fu/matrix_block.py
dgkan/fu/poprisk_source.py
dgkan/profiling/efficiency_v21.py
dgkan/kernels/che_official.py
dgkan/kernels/fou_official.py
experiments/run_v21_s05_truth_gate.py
experiments/run_v21_efficiency_officialization.py
experiments/run_v21_mlp_source_dynamics.py
experiments/run_v21_kan_source_channel_writer.py
experiments/run_v21_function_space_target_reset.py
experiments/run_v21_finalize.py
```

如果某历史 runner 被合并进 generic runner，必须提供：

```text
compatibility_shim.py
compatibility_manifest.csv
legacy_to_current_runner_map.csv
```

## 5.3 S0.5 硬门

### S0.5-1 import / compile closure

必须执行：

```bash
python -m compileall -q dgkan experiments
python experiments/run_v21_s05_truth_gate.py --check-import-closure 1
```

通过标准：

```text
compileall_ok = 1
import_error_count = 0
historical_proxy_guard_count reported separately
```

历史 proxy 文件可以存在，但必须列入：

```text
legacy_proxy_isolation_manifest.csv
```

不能被 v21 runner import。

### S0.5-2 LineC-fast / LineC-channel golden tests

必须有 golden fixtures：

```text
NoOpNull
RandomMatchedNormNull
KnownTrainSplitTransfer
KnownNoiseLeak
KnownReservoirOnly
BatchPermutationMismatch
ExceptionPathMeasurementInvalid
ScaleInvariance
ControlEquivalence
LongHorizonDebtRecoveryToy
```

通过标准：

```text
LineC_fast_golden_pass = 1
LineC_channel_golden_pass = 1
ExceptionPath route = R0-LineCMeasurementInvalid
MeasurementInvalid rows excluded from pass/fail means
```

### S0.5-3 Retention / debt / route aggregation tests

必须测试：

$$
Retention_{h1600/h800}=\frac{\max(0, Source_{h1600})}{\max(\epsilon, Source_{h800})}.
$$

如果 `Source_h800 <= 0` 且 `Source_h1600 > 0`，只能写：

```text
late_rebound = 1
retained = 0
```

必须测 debt：

$$
Recovery(H)=1-\frac{Debt_{final}(H)}{Debt_{peak}+\epsilon}.
$$

debt 包括：

```text
tail_debt_CEp99
tail_debt_NLL
linec_fast_debt
linec_channel_debt
calibration_debt_ECE
calibration_debt_Brier
auctime_debt
```

通过标准：

```text
source_retention_formula_pass = 1
debt_recovery_formula_pass = 1
single_row_max_cannot_open_route = 1
route_uses_grouped_9row_mean = 1
```

### S0.5-4 Mechanism semantic contract

每个 FU mechanism 必须有：

```text
mechanism_id
natural_language_description
mathematical_update_definition
parameter_roles_touched
uses_adamw_primary
uses_sgd_momentum_primary
uses_fu_primary
uses_function_space_target
uses_matrix_block_state
uses_slow_state
direction_source
forbidden_metric_used_for_direction
matched_controls
implementation_entrypoint
unit_test_name
```

如果机制只是 gradient mask，不允许命名为：

```text
MatrixBlockFU
FunctionSpaceOperatorFU
SlowStateSourceWriter
```

必须降级为 smoke 名：

```text
GradientMaskSmoke
ScaledGradientSmoke
```

### S0.5-5 Efficiency profiler correctness

必须分别测：

```text
forward_only_ms
backward_grad_ms
SGD_update_ms
AdamW_update_ms
FU_commit_ms
basis_specific_fused_update_ms
functional_direction_ms
functional_projection_ms
linec_audit_ms
horizon_readback_ms
forward_peak_memory
backward_peak_memory
optimizer_state_memory
basis_activation_bytes
workspace_bytes
```

通过标准：

```text
phase_timing_separated = 1
audit_cost_separated = 1
same_param_mlp_present = 1
full_loop_timing_present = 1
profiler_path_matches_training_path = 1
```

如果 update phase 只用 SGD step 代替 AdamW / FU commit，必须标记：

```text
optimizer_update_truth_incomplete = 1
```

不能作为 final efficiency evidence。

---

# 6. Part B：Basis Kernel Efficiency Officialization

## 6.1 总目标

v21 的效率目标不是继续证明 “KAN 可能快”，而是把 v20 的 D-CHE/D-FOU official-ish pass 推成 full-loop official evidence：

$$
\boxed{
\text{D-CHE 或 D-FOU 在 full functional training loop 中，}
\text{forward/backward/step/memory 接近 same-param MLP。}
}
$$

## 6.2 Hypothesis E1：D-CHE / D-FOU profiler pass 能在 full-loop 保持

### 实验 E1a：D-CHE full-loop officialization

候选：

```text
CHE21-R4-k3-gradbuf-triton-official
CHE21-R2-low-degree-k3-official
CHE21-R4-k5-gradbuf-triton-ablation
```

必须比较：

```text
same-param MLP
D-CHE reference torch path
D-CHE v20 officialized profiler path
D-CHE v21 full-loop official path
```

记录指标：

```text
forward_only_ms
backward_grad_ms
AdamW_update_ms
FU_commit_ms
functional_direction_ms
step_total_ms
linec_audit_ms
horizon_readback_ms
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
full_loop_matches_profiler_ratio
kernel_count
triton_kernel_count
dense_materialization_bytes
basis_activation_bytes
workspace_bytes
gradcheck_relerr
gradcheck_cos
```

通过标准 E1a-exploration：

```text
forward_ratio <= 1.25
backward_ratio <= 1.30
step_ratio <= 1.25
memory_ratio <= 1.10
gradcheck_relerr <= 1e-4
dense_materialization_bytes <= 0.05 * reference_dense_bytes
```

强标准 E1a-official-like：

```text
forward_ratio <= 1.15
backward_ratio <= 1.20
step_ratio <= 1.15
memory_ratio <= 1.05
full_loop_matches_profiler_ratio <= 1.10
```

如果不满足，Codex 必须先尝试：

```text
1. 分离 audit / horizon readback cost；
2. 检查 training path 是否没有调用 officialized kernel；
3. 检查 dense basis materialization 是否重新出现；
4. 对 batch 32/128/256 画 component waterfall；
5. 若 only backward fail，优先 analytic backward；
6. 若 only forward fail，优先 fused degree-readout contraction。
```

### 实验 E1b：D-FOU full-loop officialization

候选：

```text
FOU21-R4-k4-triton-no-materialize-official
FOU21-R2-lowfreq-k2-stream-official
FOU21-R3-tablelookup-bandreadout-official
```

记录指标与 D-CHE 相同，额外记录：

```text
sincos_eval_ms
frequency_table_ms
band_eval_ms
band_readout_contraction_ms
bandwise_backward_ms
high_frequency_energy_fraction
low_frequency_active_fraction
```

通过标准同 E1a。

如果不满足，Codex 必须先尝试：

```text
1. sincos recurrence vs table lookup vs torch trig 三分对照；
2. fused band eval + readout contraction；
3. high-frequency quarantine；
4. no-materialize [B,H,K] audit；
5. bandwise analytic backward。
```

## 6.3 Hypothesis E2：LQ / D-RAT 可以局部修复，但不占主线

### 实验 E2a：LQ targeted repair smoke

候选：

```text
LQ21-R1-legendre-recurrence
LQ21-R2-fixed-frame-cache
LQ21-R3-fused-projection-update
```

只需跑 batch 32/128。

进入下一轮条件：

```text
forward_ratio <= 2.0
step_ratio <= 1.5
memory_ratio <= 1.1
```

否则维持 monitor。

### 实验 E2b：D-RAT targeted repair smoke

候选：

```text
RAT21-R1-horner-eval
RAT21-R2-branchless-denominator
RAT21-R3-telemetry-separated
```

通过条件：

```text
forward_ratio <= 2.0
step_ratio <= 1.5
denominator_min_safe = 1
```

## 6.4 Hypothesis E3：D-RBF / D-WAV 暂时只能 sparse-local smoke

只做：

```text
D-RBF21-localK4-noDense-smoke
D-WAV21-sparseSupportBackward-smoke
```

若 forward ratio 仍 > 3.0，则写：

```text
family_status = SparseLocalNotRuntimeEffective_CurrentImplementation
```

不进入 FU proof。

---

# 7. Part C：Functional Update Source-Retention 主线

## 7.1 总目标

v21 的 FU 目标不是“找到更强 h800 source”。目标是：

$$
\boxed{
\text{找到能从 h800 留到 h3200/h4800 的 source-channel writer。}
}
$$

也就是说，成功必须同时满足：

```text
source positive;
source retained;
debt recovered;
controls fail;
KAN vs MLP attribution clear;
full-loop efficiency acceptable。
```

## 7.2 Hypothesis F1：MLP source dynamics 是 FU 机制发现的主实验台

v20 说明 MLP M16 retained source 不复现，但 M2、M15、F9/F10 给出了三个不同 dynamics：

```text
M2:
  strong h800 source, strong washout。

M15:
  weak but more stable source。

F9/F10:
  source can be held to h1600, not h3200/h4800。
```

### 实验 F1：MLP source dynamics decomposition

候选：

```text
MLP-F1-M2-strong-source
MLP-F2-M15-weak-stable
MLP-F3-F9-hold
MLP-F4-F10-cycle-hold
MLP-F5-M2-plus-M15-anchor
MLP-F6-M2-source-with-slow-anchor
MLP-F7-M2-source-with-matrix-block-retention
```

核心公式：

$$
s_t^{fast}=u_t^{M2},
$$

$$
s_t^{anchor}=u_t^{M15},
$$

$$
s_t^{slow}=\beta s_{t-1}^{slow}+(1-\beta)P_{signal}(s_t^{fast},s_t^{anchor}).
$$

提交更新不是 raw $u_t^{M2}$，而是：

$$
\Delta\theta_t = \eta_f u_t^{fast} + \eta_s s_t^{slow}.
$$

固定参数，不做网格搜索：

```text
beta = 0.98
eta_fast = existing M2 safe scale
eta_slow = 0.25 * eta_fast
commit_interval = 50
```

记录指标：

```text
source_h100/h400/h800/h1600/h3200/h4800
source_retention_h1600_over_h800
source_retention_h3200_over_h1600
source_retention_h4800_over_h3200
source_washout_rate
fast_source_norm
anchor_source_norm
slow_state_norm
cos_fast_anchor
cos_fast_slow
cos_slow_gradient
cos_slow_adamw
AdamW_overwrite_cosine
tail_debt_peak/final/recovery
LineC_fast_debt_peak/final/recovery
LineC_channel_debt_peak/final/recovery
ECE_debt_peak/final/recovery
Brier_debt_peak/final/recovery
AUCtime_ratio
matched_control_source
random_pulse_same_schedule_source
NoOp_overhead_source
```

F1 exploration success：

```text
9-row mean source_h3200 >= 0.005
source_retention_h3200_over_h1600 >= 0.50
pass_count_h3200 >= 4/9
LineC_channel_recovery_h3200 >= 0.50 or tail_recovery_h3200 >= 0.50
matched controls fail
```

F1 strong success：

```text
9-row mean source_h4800 >= 0.005
source_retention_h4800_over_h3200 >= 0.50
pass_count_h4800 >= 5/9
tail_recovery_h4800 >= 0.60
LineC_channel_recovery_h4800 >= 0.60
random pulse + same schedule fails
```

如果不满足，Codex 必须先尝试：

```text
1. 如果 M2 strong source washout：提高 slow-state retention，不能提高 raw M2 amplitude。
2. 如果 M15 too weak：用 M15 只做 anchor，不做 primary source。
3. 如果 controls explain：切换到 source-channel target validation，不继续 same family。
4. 如果 debt 不恢复：先分析 debt peak vs slow state，不改方向。
5. 如果 h3200 positive but h4800 negative：加 h6400 diagnostic，不 promotion。
```

## 7.3 Hypothesis F2：AdamW 洗掉 FU source

### 实验 F2：optimizer overwrite / AdamW-free comparison

在 MLP F1 top candidates 与 D-CHE/D-FOU source candidates 上比较：

```text
O0 AdamW-primary + FU residual
O1 SGD-primary + FU
O2 Momentum-primary + FU
O3 Schedule-free/slow-state primary + FU
O4 FU-primary + gradient cotangent only
O5 FU-only smoke
O6 Role-partition optimizer
```

记录：

```text
AdamW_cumulative_projection_on_FU
SGD_cumulative_projection_on_FU
Momentum_cumulative_projection_on_FU
slow_state_projection_on_FU
overwrite_cosine_h100/h800/h1600/h3200
source_retention under each optimizer
debt recovery under each optimizer
```

判断 AdamW 阻碍成立：

```text
AdamW-primary source_h1600 or h3200 <= 0
SGD/Momentum/slow-state source_h3200 >= 0.005
same candidate controls fail
AdamW cumulative projection on FU <= -0.20
```

如果 AdamW 阻碍成立，下一步 Codex 直接尝试：

```text
FU-primary / Momentum-primary / schedule-free primary，
并把 AdamW 降级为 control，不再当 default training path。
```

## 7.4 Hypothesis F3：high ActuationR2 失败是 target 错，不是 actuation 错

### 实验 F3：function-space target reset

v20 的高 ActuationR2 + zero source-success 表明执行器能动，但目标不对。v21 要构造新的 output-space source target。

训练 batch 拆成：

```text
B1: target-build split
B2: transfer-check split
B3: safety/check split
```

构造 candidate output displacement target $\Delta f$，只允许来自 train-stream：

```text
T1 loss-cotangent descent target
T2 split-consensus transfer target
T3 leave-one-point population-risk target
T4 MLP-retained-source-aligned target
T5 random matched target control
T6 sign-flipped target control
```

Actuation solver：

$$
\Delta\theta^* = \arg\min_{\Delta\theta}\|J\Delta\theta-\Delta f\|_2^2 + \lambda\|\Delta\theta\|_M^2.
$$

记录：

```text
ActuationR2
projection_residual_norm
B1_gain
B2_transfer_gain
B3_safety_gain
random_target_gain
sign_flipped_target_gain
source_h800/h1600/h3200
source_retention
LineC_channel_debt
controls
```

F3 target success：

```text
ActuationR2 >= 0.50
B2_transfer_gain > random target by >= 0.005
source_h1600 >= 0.005
source_h3200 >= 0.005
random target source <= 0
sign-flipped target source <= 0
```

如果 high ActuationR2 但 source <= 0，Codex 必须先尝试：

```text
1. target corruption contrast；
2. source target vs random target matched ActuationR2；
3. B1-only vs B1+B2 target；
4. lower-rank target projection；
5. 如果全部 fail，写 TargetObservableNoGo，不继续 actuation solver 小修。
```

## 7.5 Hypothesis F4：KAN carrier 需要自己的 source-channel writer，不是直接搬 MLP update

### 实验 F4：D-CHE / D-FOU source-channel writer

前提：D-CHE / D-FOU officialized kernel pass，至少 E1 exploration pass。

KAN source-channel writer 分三类：

```text
KSW1 basis-estimate readout-commit
  用 basis/degree/frequency channel 估计 source，但只提交 readout / low-degree readout update。

KSW2 low-degree source bank
  只允许 D-CHE low-degree / D-FOU low-frequency bank 承载 slow source。

KSW3 dual-bank source-reservoir
  source bank slow commit，reservoir bank fast absorb / decay，不让 fast hazard 进入 source bank。
```

D-CHE 版本：

```text
CHE-KSW1-degree-to-readout
CHE-KSW2-lowdegree-source-bank
CHE-KSW3-dualbank-source-reservoir
```

D-FOU 版本：

```text
FOU-KSW1-band-to-readout
FOU-KSW2-lowfreq-source-bank
FOU-KSW3-dualband-source-reservoir
```

记录：

```text
basis_channel_source_score
readout_source_score
low_degree_energy
high_degree_energy
low_frequency_energy
high_frequency_energy
source_bank_norm
reservoir_bank_norm
source_to_reservoir_leak
source_retention_h800/h1600/h3200/h4800
debt recovery
same-mechanism MLP source
KAN_specific_delta
full-loop efficiency ratio
```

KAN source-channel success：

```text
source_h3200 >= 0.005
source_retention_h3200_over_h1600 >= 0.50
pass_count_h3200 >= 4/9
KAN_specific_delta_vs_MLP_same_mechanism >= 0.005
full_loop_step_ratio <= 1.25
memory_ratio <= 1.10
controls fail
```

如果 D-CHE/D-FOU source still fail，Codex 必须先尝试：

```text
1. 只在 readout commit，不动 basis；
2. 只在 low-degree / low-frequency commit；
3. fast reservoir decay，不改 source bank；
4. 如果 readout-only works but basis-write fails，说明 carrier actuation path 错；
5. 如果 all fail but MLP works，说明 KAN carrier source-channel mismatch，需要新 carrier，不继续同 family FU。
```

## 7.6 Hypothesis F5：PopRisk/SNR estimator 需要重新定义为 retention predictor

### 实验 F5：source-retention estimator audit

Estimator candidates：

```text
P1 parameter SNR baseline
P2 output-space leave-one-point transfer
P3 split-consensus displacement SNR
P4 signal-channel LineC-channel drift
P5 matrix-block spectral source score
P6 slow-state stability score
P7 random / shuffled controls
```

不提交 update，只预测：

```text
source_h1600
source_h3200
source_h4800
washout
late rebound
```

记录：

```text
Spearman_h1600
Spearman_h3200
Spearman_h4800
AUC_retained_vs_washout
leave-dataset-out_AUC
leave-seed-out_AUC
incremental_AUC_vs_controls
false_positive_rate
false_negative_rate
```

通过标准：

```text
leave-dataset-out AUC >= 0.65
leave-seed-out AUC >= 0.65
Spearman_h3200 >= 0.30
incremental_AUC_vs_controls >= 0.10
```

如果不通过：

```text
Estimator 不允许用于 direction；
只能作为 failure taxonomy。
```

---

# 8. Controls / attribution hard requirements

每个 positive-looking FU candidate 必须比较：

```text
NoOpMatchedOverhead
RandomMatchedNorm
RandomMatchedActuationR2
SignFlippedTarget
SameActuationR2RandomTarget
SameScheduleRandomPulse
RecoveryOnly
AdamWExtraStepsMatchedTime
SGDMomentumMatched
MLP same mechanism
Same carrier no-FU
Same efficiency kernel no-FU
```

KAN-specific claim 只有在以下条件成立时允许：

```text
source retained in KAN;
same-mechanism MLP not retained or KAN delta >= 0.005;
random / NoOp / recovery-only controls fail;
efficiency pass;
debt recovered;
no forbidden direction source。
```

---

# 9. Success criteria

## S0.5：代码/指标/机制正确性

全部必须通过：

```text
source code included
compile/import closure
LineC fast/channel golden
retention/debt formulas
route aggregation
mechanism semantic contracts
efficiency profiler correctness
kernel gradcheck
gpu queue contract
```

## S1：efficiency officialization

D-CHE 或 D-FOU 至少一个满足：

```text
forward_ratio <= 1.25
backward_ratio <= 1.30
step_ratio <= 1.25
memory_ratio <= 1.10
full_loop_matches_profiler_ratio <= 1.10
gradcheck pass
no-materialize proof
```

## S2：weak FU source retention

任一 carrier × mechanism：

```text
9-row mean source_h1600 >= 0.005
pass_count_h1600 >= 3/9
source_retention_h1600_over_h800 >= 0.40
matched controls fail
```

## S3：productive retained source

任一 carrier × mechanism：

```text
9-row mean source_h3200 >= 0.005
pass_count_h3200 >= 4/9
source_retention_h3200_over_h1600 >= 0.50
tail_recovery_h3200 >= 0.50
LineC_channel_recovery_h3200 >= 0.50
AUCtime_ratio_h3200 <= 1.10
matched controls fail
```

## S4：KAN-specific functional source

KAN carrier：

```text
source_h3200 >= 0.005
pass_count_h3200 >= 5/9
KAN_specific_delta_vs_MLP_same_mechanism >= 0.005
full_loop_step_ratio <= 1.25
memory_ratio <= 1.10
controls fail
```

## S5：official success，不降低

```text
real_dataset_seed_pass_count = 9/9
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_channel_pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.10
official kernel gate pass
controls fail
promotion_allowed = 1
```

---

# 10. Metrics to record

## 10.1 Code audit metrics

```text
source_file_count
expected_source_file_count
missing_source_files
compileall_ok
import_error_count
linec_fast_golden_pass
linec_channel_golden_pass
retention_formula_pass
debt_formula_pass
route_aggregation_pass
mechanism_contract_pass
profiler_phase_pass
kernel_gradcheck_pass
```

## 10.2 Efficiency metrics

```text
carrier
kernel_variant
batch_size
param_count
same_param_mlp_id
forward_only_ms
backward_grad_ms
SGD_update_ms
AdamW_update_ms
FU_commit_ms
basis_specific_fused_update_ms
functional_direction_ms
functional_projection_ms
functional_commit_ms
linec_audit_ms
horizon_readback_ms
step_total_ms
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
kernel_count
triton_kernel_count
dense_materialization_bytes
workspace_bytes
basis_activation_bytes
optimizer_state_bytes
full_loop_matches_profiler_ratio
gradcheck_relerr
gradcheck_cos
```

## 10.3 Functional source metrics

```text
carrier
mechanism
horizon
source_vs_best_control
source_retention_from_previous_horizon
source_washout_rate
late_rebound_flag
pass_count
worst_source
paired_ci_low
paired_ci_high
```

## 10.4 Debt metrics

```text
tail_debt_CEp99_peak/final/recovery
tail_debt_NLL_peak/final/recovery
LineC_fast_debt_peak/final/recovery
LineC_channel_debt_peak/final/recovery
ECE_debt_peak/final/recovery
Brier_debt_peak/final/recovery
AUCtime_ratio
```

## 10.5 Optimizer/FU interaction metrics

```text
FU_vs_gradient_cosine
FU_vs_AdamW_cosine
FU_vs_SGD_cosine
FU_vs_momentum_cosine
AdamW_cumulative_projection_on_FU
Momentum_cumulative_projection_on_FU
slow_state_norm
fast_state_norm
source_bank_norm
reservoir_bank_norm
source_to_reservoir_leak
```

## 10.6 Actuation metrics

```text
ActuationR2
projection_residual_norm
B1_gain
B2_transfer_gain
B3_safety_gain
random_target_gain
same_actuation_random_target_source
sign_flipped_target_source
```

---

# 11. Visualizations required

## 11.1 Code audit dashboard

```text
source completeness bar
LineC golden matrix
retention/debt formula unit test table
route aggregation counterexample table
mechanism semantic alias heatmap
```

## 11.2 Efficiency dashboard

```text
forward/backward/update/commit stacked bars
D-CHE dense vs no-materialize timing waterfall
D-FOU trig vs recurrence vs table lookup waterfall
step ratio vs memory ratio Pareto
full-loop vs profiler ratio scatter
basis family efficiency heatmap
```

## 11.3 Functional source dashboard

```text
source retention curves h100/h400/h800/h1600/h3200/h4800
washout vs retained heatmap
late rebound timeline
MLP M2/M15/F9/F10 source dynamics comparison
KAN vs MLP same-mechanism source delta
```

## 11.4 Debt dashboard

```text
tail debt peak/final/recovery curves
LineC-channel debt curves
ECE/Brier calibration debt curves
source vs debt scatter
AUCtime ratio vs source retention scatter
```

## 11.5 Actuation / target dashboard

```text
ActuationR2 vs source scatter
random target vs train-source target bar
projection residual vs source retention
B1/B2/B3 gain waterfall
```

## 11.6 GPU execution dashboard

```text
gpu utilization over time
idle violation table
queue drain timeline
per-GPU job mix stacked bar
```

---

# 12. 4GPU execution plan

## Round 0：S0.5 code / metric truth gate

```text
GPU0:
  compile/import/LineC/retention/debt/route tests。
GPU1:
  mechanism semantic contracts / update semantics tests。
GPU2:
  efficiency profiler correctness / kernel gradcheck。
GPU3:
  packet manifest / sha256 / figure smoke / queue setup。
```

Round 0 必须先过。失败时不能进入 scientific route。

## Round 1：efficiency officialization + MLP source dynamics

```text
GPU0:
  D-CHE CHE21-R4/R2 full-loop efficiency。
GPU1:
  MLP F1 source dynamics M2/M15/F9/F10。
GPU2:
  D-FOU FOU21-R4/R2/R3 full-loop efficiency。
GPU3:
  LQ/RAT/RBF/WAV targeted repair smoke + controls。
```

## Round 2：source-channel writer experiments

```text
GPU0:
  D-CHE KSW1/KSW2/KSW3。
GPU1:
  MLP slow-state / matrix-block / AdamW-free。
GPU2:
  D-FOU KSW1/KSW2/KSW3。
GPU3:
  F3 function-space target reset + controls。
```

## Round 3：long-horizon confirmation

```text
GPU0:
  top D-CHE h3200/h4800。
GPU1:
  top MLP h3200/h4800。
GPU2:
  top D-FOU h3200/h4800。
GPU3:
  all matched controls + debt readback + final packet。
```

---

# 13. Failure handling / Codex fallback contract

## 13.1 If code packet missing source

Codex must stop scientific route and produce:

```text
missing_source_files.csv
source_repack_command.sh
packet_rebuild_log.md
```

No scientific no-go allowed.

## 13.2 If D-CHE / D-FOU full-loop efficiency fails

Codex must produce:

```text
component_waterfall.csv
profiler_vs_full_loop_diff.csv
dense_materialization_trace.csv
kernel_invocation_trace.csv
```

Then attempt:

```text
audit cost separation;
training path kernel binding check;
no-materialize memory audit;
analytic backward fallback;
```

## 13.3 If MLP source washout remains

Codex must not add new FU token. It must first run:

```text
M2 strong-source vs M15 weak-stable anatomy;
slow-state retention;
AdamW overwrite diagnostic;
matrix-block state diagnostic;
```

## 13.4 If MLP source retained but KAN source fails

Codex must run:

```text
KAN source-channel mismatch audit;
readout-only commit;
low-degree / low-frequency source bank;
basis-estimate readout-commit;
```

## 13.5 If high ActuationR2 but source fails again

Codex must run:

```text
random target matched ActuationR2;
train-source target vs random target;
B1-only vs B1+B2 target;
target corruption contrast;
```

Then write:

```text
TargetObservableNoGo
```

if all fail.

## 13.6 If PopRisk/SNR predictor has negative Spearman

Codex must not use it for direction. It must write:

```text
PopRiskEstimatorInvalid_CurrentImplementation
```

and only continue with alternative source estimators.

---

# 14. Final route decision

Possible routes:

```text
R0-CodeAuditIncomplete
R1-EfficiencyOfficializationFailed
R2-MLPSourceDynamicsNoRetention
R3-MLPGenericRetained_KANCarrierFail
R4-KANSourceChannelWriterPartial
R5-ActuationTargetObservableNoGo
R6-PopRiskEstimatorInvalid
R7-DCHESourceRetainedEfficiencyPass
R8-DFOUSourceRetainedEfficiencyPass
R9-OfficialS5FunctionalSuccess
```

A completed no-go is only allowed if:

```text
S0.5 pass = 1
Efficiency officialization attempted for D-CHE and D-FOU
MLP source dynamics F1 completed
KAN source-channel F4 completed if efficiency pass
Function-space target reset F3 completed
Controls completed
4GPU queue drained
no idle violation
```

---

# 15. One-page operational summary for Codex

v21.0 is not a token-extension experiment. It must deliver:

```text
1. Complete source-code audit packet.
2. D-CHE / D-FOU full-loop efficiency officialization.
3. MLP source dynamics dissection: M2 strong washout, M15 weak stable, F9/F10 hold.
4. KAN source-channel writer on D-CHE / D-FOU, not direct MLP update transplant.
5. Function-space target reset: high ActuationR2 must be paired with useful train-source target.
6. Complete debt recovery curves.
7. Dynamic 4GPU queue evidence.
```

The final answer cannot be “partial source exists.” It must say one of:

```text
source retained and why;
source washed out and why;
efficiency officialized and where;
efficiency failed and which component;
source target observable or not;
KAN carrier mismatch or KAN-specific advantage.
```

