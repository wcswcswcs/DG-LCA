# DG-KAN v18.0：Evidence-First Breakthrough Plan  
## 代码审计闭环 + AdamW-free / Signal-State Functional Update + Basis Kernel Efficiency Breakthrough + 4GPU 动态并行

> 版本：v18.0 execution plan  
> 生成时间：2026-06-02  
> 目标读者：接手项目的人可以只读本文件，知道项目总目标、当前进展、为什么之前原地打转、下一步该怎么执行。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline official path；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能做 audit / gate / debt readback，不能生成方向。  
> 当前服务器资源：4 张 GPU。v18.0 必须以 4GPU dynamic queue 执行，不允许本质串行。

---

# 0. 项目总目标与当前真实状态

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找一个局部 positive 的 update。项目目标是：

$$
\boxed{
\text{在 strict FC-PureKAN / KAN-like carrier 上，}
\text{用 functional update 改善训练动力学，}
\text{并达到 MLP-like efficiency。}
}
$$

更具体地说，最终系统必须同时满足：

```text
1. 基函数 / carrier 的 forward、backward、参数更新、显存与同参数量 MLP 可比；
2. functional update 的收益不能被 AdamW、SGD、random、NoOp、matched overhead、recovery-only、MLP/generic controls 解释；
3. 短期 bad update 可以出现，但必须在长期 horizon 中偿还 debt；
4. source 必须从 h100/h800 延续到 h1600/h3200，而不是 single-row spike 或 late rebound；
5. 如果 MLP-FU 成功而 KAN 不成功，必须写成 generic functional dynamics，不得写 KAN-specific；
6. 只有 KAN carrier 在同机制下优于 same-param MLP，并且 efficiency gate 通过，才允许 KAN-specific claim。
```

v17.1 当前真实状态：

```text
CodeRoute:
  S0_1-CodeCorrectnessPassed。

EfficiencyRoute:
  R-EfficiencyForwardBlocked-D-CHE-D-FOU-D-RAT-D-RBF-D-WAV-LQ。

FunctionalRoute:
  R-SourceSingleRowOnlyOrWashedOut。

S4/S5:
  未打开。

关键事实：
  full carrier x mechanism h100 screen 已跑；
  945-row h800/h1600 extension 已跑；
  h3200 有 1 个 positive mean group，但 retained_candidate_count=0，late_rebound_count=1；
  route 中 best source 是 single-row max，不能当机制级成功；
  LineC exception policy 已修复为 MeasurementInvalid；
  真实 tail/LineC/calibration/AUC debt recovery 仍未完整测量；
  non-MLP basis 的主要效率 blocker 是 forward / basis eval / dense materialization，不是显存。
```

v18.0 的核心判断是：

$$
\boxed{
\text{下一步不能继续“跑更多同名机制”。}
\text{必须同时突破两个硬问题：}
}
$$

```text
主线 A：Functional Update
  为什么 source 不留存？AdamW 是否洗掉 FU？FU 是否需要 slow-state / signal-channel / matrix-block 实现？

主线 B：Basis Efficiency
  为什么 basis forward 慢？如何把 D-FOU / D-CHE 等 family 从 dense/materialized path 推进到 fused/no-materialize/kernel-native path？
```

---

# 1. v17.1 暴露的根本问题

## 1.1 代码/指标层：S0.1 通过不等于科学判断可靠

v17.1 让代码审计比以前好，但仍不能直接开始 S4/S5。原因是：

```text
1. LineC golden 通过，只说明当前 cheap proxy 没有基础 bug；
   它不是完整 signal-channel / reservoir geometry。
2. source retention 公式修正了，但真实 debt accounting 没完整测量。
3. route 仍保留 single-row max 字段，容易被误读成机制级成功。
4. M3 / M7 / M9 等 mechanism 仍有 prototype/smoke 成分，名字大于实现。
5. efficiency profiler 的 optimizer_update_ms 仍需要区分 AdamW / SGD / FU commit / fused update。
```

所以 v18.0 第一阶段必须是 **S0.2 Metric and Implementation Truth Gate**。不通过 S0.2，不允许写科学 no-go，也不允许写 promotion。

---

## 1.2 Efficiency 层：显存不是主 blocker，forward / basis eval 才是主 blocker

v17.1 的趋势显示：

```text
MLP:
  fastest and stable.

D-FOU:
  step/memory 局部可承受，但 forward 约 4-6x。

D-CHE:
  memory 接近 MLP，但 forward 约 4-5x，backward 约 1.3-1.6x。

LQ:
  memory 接近 MLP，但 forward / backward 仍偏慢。

D-RAT:
  denominator / rational eval 仍贵。

D-RBF:
  forward 约 9-10x，local support 未转成效率。

D-WAV:
  forward 约 13-19x，当前只能 low-priority smoke。
```

所以 efficiency 不能继续只写：

```text
ForwardBlocked。
```

必须进入 family-specific kernel repair：

```text
D-FOU:
  首先突破 forward kernel。

D-CHE:
  首先突破 no-materialize degree recurrence / fused contraction。

LQ:
  首先突破 recurrence + projection update fusion。

D-RAT:
  首先突破 branchless denominator / vectorized rational eval。

D-RBF/D-WAV:
  只有实现真正 sparse-local fused path 后才值得扩大实验。
```

---

## 1.3 Functional 层：当前最大问题不是“没有 source”，而是 source 不留存

v17.1 的关键现象是：

```text
h800:
  MLP / M2-SGDMomentumPrimaryFU 有 grouped mean positive。

h1600:
  MLP source 被洗掉；
  D-CHE 只有极弱正均值，低于 gate。

h3200:
  D-CHE / M2 出现 late rebound；
  但 h1600 是负的，因此不是 retention。
```

这意味着：

$$
\boxed{
\text{current FU can produce local / short-horizon source,}
\text{but it does not reliably enter a retained signal channel.}
}
$$

v18.0 不再问“哪个 FU token h800 更高”，而问：

```text
1. AdamW 是否把 FU source 洗掉？
2. SGD/Momentum/slow-state 是否能保留 source？
3. source 是否应该写入 slow state，而不是直接写参数？
4. source 是否应该在 matrix/block space 保留，而不是逐参数保留？
5. source 是否其实是 MLP/generic functional dynamics，而不是 KAN-specific？
6. KAN basis 是否效率太差，导致 source/debt dynamics 被 runtime path 污染？
```

---

# 2. 哪些旧方向可能因代码/指标问题仍有推进希望？

v18.0 不能盲目重启旧路线，但也不能把所有旧 no-go 都当最终证伪。下面给出分级。

## 2.1 可重新开启，但必须换实现/指标的方向

### 2.1.1 PopRisk / SNR

旧证据：v13.6-v13.8 曾把 functional source 从 future/oracle target 改成 train-stream per-example gradient statistics。方向正确，但后来没有把它做成长期 retained signal-state。

当前判断：

```text
可以重开。
但不能只是 SNR mask / preconditioner；
必须变成 source-channel state:
  short EMA
  long EMA
  drift-diffusion score
  cross-split consistency
  h800/h1600 retention
```

核心假设：

$$
SNR_k=\frac{\mu_k^2}{\sigma_k^2/(b-1)+\epsilon}
$$

如果 source 要长期留下，不能只用 one-step $SNR_k$，而要维护：

$$
s_{t+1}=\beta s_t+(1-\beta)P_{\text{signal}}(g_t)
$$

并测试 $s_t$ 是否预测 h1600/h3200 source retention。

---

### 2.1.2 Split-consensus / G7R

旧证据：v15.04-v15.6 发现 split-consensus metric-only 有 source，但 bad event 高，source/hazard 静态不可分。后来 v16 进一步表明 recovery 未成功。

当前判断：

```text
可以作为 signal observability 继续用；
不应直接作为 update；
应该转成 source estimator / slow-state writer。
```

原因：如果 G7R 的 source/hazard 短期共线，静态 nullspace 会杀 source；但这不证明它不能作为 slow boundary condition。

---

### 2.1.3 Function-space / operator-level FU

旧证据：v13.2-v13.4 尝试 loss-interface cotangent -> basis parameter / operator-level update，但当时实现与 carrier/efficiency未闭合。

当前判断：

```text
需要重写成真实 function-space operator，不允许 M9=0.5*gradient 这种伪实现。
```

真实目标：

$$
\Delta f_B = J_B U\alpha
$$

不是：

$$
\Delta\theta = c g_\theta
$$

必须记录：

```text
function_displacement_norm
train_split_transfer
operator_rank
projection_error
parameter_writeback_error
source retention
controls
```

---

### 2.1.4 Graph-free analytic adjoint / manual update

旧证据：v6.1 证明 manual gradient correctness 可以过，但 P2 efficiency 没过，主要被 primitive/kernel/cache 拖慢。

当前判断：

```text
可以重开，而且必须作为 basis efficiency 的核心。
旧失败不能说明 manual path 错；
它说明 Python/Torch-level kernel/cache 没写成 MLP-like。
```

v18.0 要把 graph-free 经验迁移到 D-FOU/D-CHE：

```text
D-FOU no-materialize sincos recurrence；
D-CHE no-materialize Chebyshev recurrence；
manual/analytic backward；
fused contraction。
```

---

### 2.1.5 MLP functional update

旧证据：v14.2 以后多次显示 MLP 比 KAN 更容易出 generic source；v17.1 h800 positive 也来自 MLP/M2。

当前判断：

```text
必须作为 active discovery line，不再只是 control。
```

目的不是证明 KAN，而是先回答：

```text
functional dynamics 本身能否在最干净的 dense carrier 上成立？
```

如果 MLP 都不能 h1600/h3200 retention，KAN 上继续堆 carrier 是低效的。

---

## 2.2 目前不应重启的方向

```text
1. action bank / controller / action token：
   v14.5 已显示 oracle upper bound 不足，旧 controller route 不可重启。

2. reset / optimizer-state transport route：
   已多次被 generic optimizer-state confound 打回。

3. cover_purity / cover_churn / cover_specialization objective：
   v13.11 已证明 cover objective invalid。
   除非重新定义成 dynamic signal-channel cover，否则不重启。

4. G-token / N-token / Q-token 小扩展：
   这些是旧机制表面扩展，不是破局。

5. dataset/seed branch / threshold tuning：
   不允许。
```

---

# 3. 当前研究进展对 v18.0 的启发

v18.0 不是把最新 optimizer 直接搬来跑，而是抽取训练动力学原则。

## 3.1 Signal-channel / reservoir 理论

启发：

```text
source 必须进入 signal channel；
reservoir 中的变化不转 test-visible signal；
noise 进入 signal channel 才是真风险。
```

v18.0 的改动：

```text
1. source_retention 不再只看 h800；
2. 必须记录 signal-channel projection；
3. PopRisk/SNR 作为 source writer；
4. LineC-channel 作为低频审计。
```

---

## 3.2 Deep Manifold / boundary-conditioned iteration

启发：

```text
training is moving boundary-conditioned iteration；
短期 perturbation 可能是 fixed-point region 形成过程的一部分；
好几何不是单步不坏，而是长期能收敛到更好区域。
```

v18.0 的改动：

```text
1. bad update 不直接判死；
2. debt recovery 必须真实测量；
3. FU 被测试为 trajectory boundary / slow-state writer，而不是 AdamW residual。
```

---

## 3.3 Schedule-Free / AdEMAMix / slow-state optimizers

启发：

```text
source 可能需要 slow state；
旧梯度/旧 source 可能在长 horizon 仍有用；
fast iterate 和 slow iterate 应该分离。
```

v18.0 的改动：

```text
1. SlowStateFU 是主机制，不是 fallback。
2. 记录 short/long source agreement。
3. 比较 fast source 和 slow source 的 h1600/h3200 retention。
```

---

## 3.4 Muon / SOAP / block optimizer

启发：

```text
逐参数 AdamW 可能不是矩阵/基函数参数的自然坐标；
source 可能要在 block/matrix space 中保留。
```

v18.0 的改动：

```text
1. MLP matrix-block FU 主线；
2. D-CHE degree-readout block FU；
3. D-FOU band-readout block FU；
4. LQ projection frame block FU；
5. matched random block controls。
```

---

## 3.5 AdamW / decoupled weight decay / cautious alignment

启发：

```text
AdamW 不必是 FU 默认主干；
weight decay 是 recovery / consolidation 的一种机制，不是 FU value source；
alignment 是 conflict diagnostic，不是绝对 veto。
```

v18.0 的改动：

```text
1. AdamW-primary 降级为 control condition；
2. 增加 SGD/Momentum-primary、FU-primary、slow-state FU；
3. 记录 AdamW overwrite cosine；
4. decoupled decay 只作为 recovery condition，必须有 decay-only/random-pulse controls。
```

---

# 4. v18.0 总体架构：两条主线、一个代码硬门

v18.0 由三部分组成：

```text
Part A: S0.2 代码/指标/Profiler 正确性硬门。
Part B: Basis Kernel Efficiency Breakthrough。
Part C: Functional Update Mechanism Breakthrough。
```

三个部分都必须执行。不能因为 Part C 失败而跳过 Part B，也不能因为 Part B forward blocked 而不跑 Part C smoke。两条科学主线必须并行。

---

# 5. Part A：S0.2 代码/指标/Profiler 正确性硬门

## 5.1 总目标

防止错误实现、错误指标、错误汇总方式继续让项目原地打转。

S0.2 必须先完成：

```text
1. 代码包自包含；
2. LineC-fast / LineC-channel 可区分；
3. source retention 公式正确；
4. debt accounting 真实测量；
5. route aggregation 不被 single-row max 误导；
6. update semantics 与实际机制一致；
7. optimizer coupling 明确；
8. efficiency profiler phase timing 可信；
9. kernel correctness / gradcheck 过；
10. 机制 semantic non-collapse。
```

## 5.2 Codex 必须打包的审查文件

Codex 必须生成：

```text
v18_code_review_packet.zip
```

包结构必须如下：

```text
00_README.md
01_ENVIRONMENT/
  python_version.txt
  torch_cuda_versions.txt
  gpu_info.txt
  git_status.txt
  pip_freeze.txt
02_SOURCE_TREE/
  dgkan/
  experiments/
  tests/
03_IMPORT_CLOSURE/
  import_all_core_modules.py
  import_closure_report.csv
  missing_dependency_report.csv
04_LINEC_CORRECTNESS/
  linec_fast_golden_tests.py
  linec_channel_golden_tests.py
  linec_exception_policy_tests.py
  linec_golden_results.csv
05_RETENTION_AND_DEBT/
  source_retention_unit_tests.py
  debt_accounting_unit_tests.py
  synthetic_debt_recovery_fixture.csv
  debt_formula_results.csv
06_ROUTE_AGGREGATION/
  route_aggregation_tests.py
  single_row_vs_group_mean_fixture.csv
  route_aggregation_results.csv
07_UPDATE_SEMANTICS/
  update_tensor_contract.md
  update_sign_finite_difference_tests.py
  update_kind_space_source_report.csv
08_OPTIMIZER_COUPLING/
  adamw_overwrite_diagnostics.py
  optimizer_coupling_map.csv
  adamw_free_training_loop_tests.py
09_FUNCTIONAL_MECHANISMS/
  mechanism_semantic_contracts.md
  mechanism_noncollapse_tests.py
  mechanism_update_similarity_matrix.csv
  mechanism_unit_test_results.csv
10_EFFICIENCY_PROFILER/
  efficiency_profiler_phase_tests.py
  same_param_mlp_match_tests.py
  audit_cost_separation_tests.py
  profiler_correctness_results.csv
11_KERNEL_CORRECTNESS/
  dche_gradcheck.py
  dfou_gradcheck.py
  lq_gradcheck.py
  drat_gradcheck.py
  drbf_gradcheck.py
  dwav_gradcheck.py
  kernel_gradcheck_results.csv
12_EXPERIMENT_RUNNERS/
  run_v18_code_gate.py
  run_v18_basis_efficiency_breakthrough.py
  run_v18_functional_update_breakthrough.py
  run_v18_merge_finalize.py
13_RAW_MATRICES/
  raw_h100.csv
  raw_h800.csv
  raw_h1600.csv
  raw_h3200.csv
  raw_efficiency.csv
  raw_debt.csv
14_FIGURES/
  required_figures_manifest.csv
15_GPU_QUEUE/
  runnable_queue.csv
  gpu_assignment_manifest.csv
  gpu_utilization_dashboard.csv
  idle_violation.csv
  queue_drain_report.csv
packet_manifest.csv
packet_sha256_manifest.csv
```

如果 packet 缺任一 required section，S0.2 fail。

## 5.3 S0.2 通过标准

必须同时满足：

```text
compileall_ok = 1
import_error_count = 0 for v18 core
LineC-fast golden pass = 1
LineC-channel golden pass = 1
MeasurementInvalid 不进入 geometry fail 均值
source_retention formula tests pass
debt_accounting tests pass
route aggregation tests pass
update sign tests pass
AdamW-free training loop tests pass
efficiency profiler phase tests pass
same-param MLP match tests pass
kernel gradcheck pass for all smoke families
mechanism semantic non-collapse pass
raw matrices present
```

如果不满足，Codex 不能启动正式科学 route。它必须先修 S0.2。

## 5.4 S0.2 不满足时 Codex 的继续方向

```text
Case A: import closure fail
  修 missing imports / compatibility shims；
  重新生成 import_closure_report.csv。

Case B: LineC golden fail
  不允许进入 FU；
  修 linec_fast / linec_channel 实现；
  MeasurementInvalid 必须独立 route。

Case C: retention/debt formula fail
  不允许写 source-retention / productive-debt 结论；
  修公式和 fixture。

Case D: mechanism non-collapse fail
  若两个 mechanism cosine > 0.98 且 outcomes 等价，
  合并或重写 mechanism；
  不允许用 alias 充当多机制。

Case E: efficiency profiler fail
  拆 forward/backward/update/audit；
  确保 update phase 含 live gradients；
  AdamW/SGD/FU commit 分开。
```

---

# 6. Part B：Basis Kernel Efficiency Breakthrough

## 6.1 总目标

把基函数效率从“census”推进到“kernel repair”。

目标不是所有 basis 立刻过 official，而是每个 active family 都必须被裁决为：

```text
FamilyEfficiencyPass
FamilyNearPass
ForwardKernelBlocked
BackwardKernelBlocked
UpdatePathBlocked
MemoryBlocked
AuditCostPolluted
RejectedForThisVersion
```

不能再只有 vague ForwardBlocked。

## 6.2 统一效率指标

每个 family、每个 batch size 必须记录：

```text
batch_size = 8, 32, 128, 256
hidden_dim = matched same-param MLP hidden
param_count
param_count_ratio_vs_mlp
forward_only_ms
backward_grad_ms
optimizer_update_adamw_ms
optimizer_update_sgd_ms
functional_direction_ms
functional_projection_ms
functional_commit_ms
linec_audit_ms
tail_calibration_audit_ms
horizon_readback_ms
step_total_ms
forward_peak_memory
backward_peak_memory
optimizer_state_memory
basis_activation_bytes
functional_state_bytes
workspace_temp_bytes
kernel_count
gemm_count
trig_count
exp_count
scatter_gather_count
dense_basis_materialized
official_fused_kernel_complete
```

同时记录 ratio：

```text
forward_ratio_vs_mlp
backward_ratio_vs_mlp
adamw_update_ratio_vs_mlp
sgd_update_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
audit_overhead_fraction
```

## 6.3 效率 Gate

Exploration gate：

```text
forward_ratio_vs_mlp <= 1.75
backward_ratio_vs_mlp <= 1.75
step_ratio_vs_mlp <= 1.75
memory_ratio_vs_mlp <= 1.10
audit_overhead_fraction <= 0.20
```

Near-pass gate：

```text
forward_ratio_vs_mlp <= 2.25
step_ratio_vs_mlp <= 1.75
memory_ratio_vs_mlp <= 1.10
```

Official efficiency gate：

```text
forward_ratio_vs_mlp <= 1.25
backward_ratio_vs_mlp <= 1.40
step_ratio_vs_mlp <= 1.25
memory_ratio_vs_mlp <= 1.00
official_fused_kernel_complete = 1 where kernel path claimed
```

## 6.4 Family-specific repair plan

### 6.4.1 D-FOU：Priority 1

Hypothesis:

```text
D-FOU 当前主要慢在 sin/cos / Fourier basis forward；
如果 no-materialize recurrence + fused band contraction 实现正确，
forward ratio 可从 4-6x 降到 <=1.75x。
```

必须实现并比较：

```text
FOU-R0 current reference
FOU-R1 sincos precompute table
FOU-R2 recurrence no-materialize
FOU-R3 fused band eval + readout contraction
FOU-R4 low-frequency active bank
FOU-R5 analytic backward for band-readout
FOU-R6 table lookup smoke
```

必须记录：

```text
trig_time_ms
band_eval_time_ms
band_readout_time_ms
basis_materialization_bytes
recurrence_error_vs_reference
band_grad_relerr
phase_drift
high_freq_ratio
bandwise_snr
```

判断：

```text
If R2/R3 forward <=1.75 and gradcheck pass:
  D-FOU becomes efficiency near-mainline.
If all R1-R6 forward >2.25:
  D-FOU remains ForwardKernelBlocked; stop FU proof, keep substrate smoke only.
```

### 6.4.2 D-CHE：Priority 2

Hypothesis:

```text
D-CHE 是最强 KAN carrier，但 forward 4-5x 来自 dense degree materialization；
no-materialize Chebyshev recurrence + fused degree-readout contraction 能显著降低 forward。
```

必须实现：

```text
CHE-R0 current reference
CHE-R1 recurrence no-materialize forward
CHE-R2 fused degree-readout contraction
CHE-R3 analytic backward recurrence
CHE-R4 low-degree active bank
CHE-R5 audit-separated LineC/horizon readback
```

指标：

```text
degree_eval_ms
degree_contraction_ms
degree_materialization_bytes
recurrence_max_abs_error
grad_relerr
degree_energy_low/mid/high
linec_audit_overhead_fraction
```

判断：

```text
If no-materialize CHE forward <=1.75:
  D-CHE remains main KAN carrier.
If CHE remains forward >2.25:
  D-CHE functional evidence must be interpreted as runtime-limited; no official efficiency claim.
```

### 6.4.3 LQ：Priority 3

Hypothesis:

```text
LQ historical near-pass 可能仍有 value；
当前 blocker 是 forward + projection/update path；
fixed-frame recurrence + fused projection update 可能恢复 LQ as late-attach carrier.
```

必须实现：

```text
LQ-R0 current/reanchored
LQ-R1 Legendre recurrence no-materialize
LQ-R2 fixed-frame cache
LQ-R3 fused projection update
LQ-R4 late-attach no-reanchor smoke
LQ-R5 projection-frame block update
```

判断：

```text
If LQ near-pass >=8/9 and efficiency near-pass:
  open promotion-disabled functional smoke M1/M2/M4/M7.
If reanchor fail:
  still run one-shot late-attach diagnostic, promotion_allowed=0.
```

### 6.4.4 D-RAT / Rational

Hypothesis:

```text
Rational 可能是 efficient analytic function carrier，但 denominator/derivative telemetry 与 branch eval 太慢。
```

必须实现：

```text
RAT-R0 current reference
RAT-R1 Horner numerator/denominator
RAT-R2 branchless denominator safety
RAT-R3 grouped vectorized eval
RAT-R4 derivative telemetry separated
RAT-R5 block update smoke
```

注意：

```text
不重启 reset/controller/action route。
```

### 6.4.5 D-RBF / FastKAN

Hypothesis:

```text
RBF/FastKAN 只有在 compact local K / no dense materialization 真正成立时才有希望。
```

必须实现：

```text
RBF-R0 current
RBF-R1 active center occupancy
RBF-R2 compact local K4
RBF-R3 no dense materialization
RBF-R4 sparse local backward
RBF-R5 width condition guard
```

如果 RBF-R3/R4 仍 forward >2.25，停止 RBF official FU proof。

### 6.4.6 D-WAV

Hypothesis:

```text
Wavelet 当前太慢，只有 sparse support index-only 才值得保留。
```

必须实现：

```text
WAV-R0 current
WAV-R1 triangular support index-only
WAV-R2 sparse support backward
WAV-R3 support-overlap damping readback
```

若 forward 仍 >2.25，则 D-WAV 保留 low-budget monitor。

## 6.5 必须可视化

```text
efficiency_family_forward_ratio_heatmap.svg
efficiency_family_backward_ratio_heatmap.svg
efficiency_family_step_ratio_heatmap.svg
efficiency_memory_ratio_heatmap.svg
basis_runtime_waterfall_by_family.svg
forward_component_waterfall_DFOU.svg
forward_component_waterfall_DCHE.svg
dense_materialization_bytes_by_family.svg
audit_overhead_fraction_by_family.svg
same_param_mlp_comparison_table.svg
kernel_repair_before_after_pareto.svg
```

---

# 7. Part C：Functional Update Breakthrough

## 7.1 总目标

v18.0 的 FU 不再围绕 AdamW residual。核心目标是：

$$
\boxed{
\text{找到能把 train-stream source 写入长期 retained signal channel 的 functional mechanism。}
}
$$

具体要回答：

```text
1. AdamW 是否 wash out FU source？
2. FU 是否在 MLP 上先成立？
3. FU 是否需要 slow-state / schedule-free state 才能留存？
4. FU 是否应在 matrix/block space 中定义？
5. KAN carrier 是否有 source-specific advantage？
6. h3200 late rebound 是随机还是 delayed signal migration？
```

## 7.2 Carrier matrix

每个 carrier 的执行强度：

```text
MLP:
  full functional mechanism matrix。
  目的：generic FU discovery。

D-CHE:
  full functional mechanism matrix。
  目的：KAN carrier primary。

D-FOU:
  full M1/M2/M3/M4 if efficiency near-pass or R1/R2 smoke positive；
  otherwise minimum smoke only。

LQ:
  reanchor + late-attach smoke；
  if near-pass >=8/9, run M1/M2/M4/M7.

D-RAT:
  monitor + M1/M2/M3/M7 smoke；
  no reset/controller/action。

D-RBF:
  M1/M3/M4 smoke only unless compact kernel near-pass.

D-WAV:
  M1/M3 smoke only.
```

## 7.3 Mechanism matrix

### M1：AdamW-primary + FU residual baseline

Purpose:

```text
保留旧路线作为 baseline，不再默认它正确。
```

Variants:

```text
M1a AdamW + FU residual
M1b AdamW + Cautious-style diagnostic
M1c AdamW + FU but AdamW moment frozen around FU pulse
M1d AdamW + FU with overwrite diagnostic
```

Metrics:

```text
FU_vs_AdamW_cosine
AdamW_overwrite_cosine_h100_to_h1600
source_h800/h1600
debt_recovery
control_equivalent
```

If fails:

```text
Do not declare FU no-go.
Route to M2/M3/M4/M5.
```

---

### M2：SGD / Momentum-primary + FU

Purpose:

```text
测试 AdamW 是否洗掉 FU。
```

Variants:

```text
M2a SGD + FU
M2b Momentum + FU
M2c NesterovMomentum + FU
M2d SGD/Momentum alternating FU
```

Hypothesis pass:

```text
If M2 retains h1600 source while M1 washes out,
AdamW is likely interfering with FU.
```

Metrics:

```text
source_h800_mean
source_h1600_mean
source_retention_h1600_over_h800
optimizer_overwrite_cosine
debt_recovery
AUCtime
```

---

### M3：FU-primary / FU-only

Purpose:

```text
测试 FU 是否可以作为主 optimizer，而不是 AdamW supplement。
```

Variants:

```text
M3a FU-primary gradient cotangent
M3b FU-only smoke
M3c FU-primary + decoupled shrink
M3d FU-primary + role-partition update
```

Code requirement:

```text
不能只是 mean-centered gradient；
必须有 mechanism contract：
  source estimator
  parameter writeback
  role map
  controls
```

---

### M4：Slow-state / schedule-free source writer

Purpose:

```text
解决 h800 source 到 h1600 washout。
```

Form:

$$
s_{t+1}^{short}=\beta_s s_t^{short}+(1-\beta_s)u_t
$$

$$
s_{t+1}^{long}=\beta_l s_t^{long}+(1-\beta_l)u_t
$$

Commit:

$$
\Delta\theta_t = \eta_s P_{\text{safe}}(s_t^{long})
$$

Variants:

```text
M4a short/long EMA FU
M4b schedule-free fast/slow iterate FU
M4c lookahead-style FU consolidation
M4d AdEMAMix-style two-memory FU
```

Pass evidence:

```text
h1600/h3200 source retention improves vs M1/M2;
debt recovery not worse;
controls fail.
```

---

### M5：Matrix-block FU

Purpose:

```text
source 可能在 matrix/block space 才能保留。
```

Targets:

```text
MLP hidden weight matrix；
D-CHE degree-readout block；
D-FOU band-readout block；
LQ projection frame；
Rational numerator/denominator group block。
```

Variants:

```text
M5a block momentum FU
M5b orthogonalized update diagnostic
M5c SOAP-like rotated-space diagnostic
M5d random matched block control
```

Important:

```text
Do not claim Muon/SOAP optimizer success.
Claim only whether block-space FU retains source.
```

---

### M6：PopRisk / SNR signal-channel FU

Purpose:

```text
source selection should favor coherent population signal over idiosyncratic noise.
```

Form:

$$
\mu_k=\frac{1}{b}\sum_i g_{i,k}
$$

$$
\sigma_k^2=\frac{1}{b-1}\sum_i(g_{i,k}-\mu_k)^2
$$

$$
SNR_k=\frac{\mu_k^2}{\sigma_k^2/(b-1)+\epsilon}
$$

Variants:

```text
M6a parameter SNR FU
M6b role-wise SNR FU
M6c split-consensus SNR FU
M6d slow-state SNR FU
M6e PopRisk residual noise suppressor
```

Pass evidence:

```text
source channel projection positive;
noise-channel leakage not increased long-term;
h1600/h3200 retention improves.
```

---

### M7：Function-space operator FU

Purpose:

```text
直接在 function/output displacement 上定义 update，而不是伪装成 0.5 * gradient。
```

Form:

$$
\alpha^\star=
\arg\min_\alpha
\mathcal L_{B_1}(f_\theta+J_{B_1}U\alpha)
+
\lambda
\mathcal L_{B_2}(f_\theta+J_{B_2}U\alpha)
+
\rho\|\alpha\|^2
$$

Then:

$$
\Delta\theta=U\alpha^\star
$$

Required:

```text
U must be explicit subspace:
  Adam subspace
  per-example gradient low-rank subspace
  basis-channel subspace
  matrix-block subspace
  random matched control subspace
```

Must record:

```text
operator_rank
solve_time_ms
projection_error
actual_delta_fidelity
split_transfer_gain
matched_control_gain
```

---

### M8：Dynamic geometry / debt recovery FU

Purpose:

```text
测试短期 bad update 是否是可偿还 debt。
```

Variants:

```text
M8a pulse-once + SGD recovery
M8b pulse-once + Momentum recovery
M8c pulse-once + slow-state recovery
M8d pulse-once + decoupled shrink recovery
M8e pulse-every-K with cooldown
```

Must record real debt:

```text
tail_debt_peak
tail_debt_final
tail_recovery_rate
LineC_debt_peak
LineC_debt_final
LineC_recovery_rate
calibration_debt_peak/final
AUC_debt_peak/final
```

---

### M9：Late attach / phase-specific FU

Purpose:

```text
FU may be harmful early but useful after carrier stabilizes, or vice versa.
```

Variants:

```text
M9a early-only FU
M9b mid-only FU
M9c late-only FU
M9d snapshot late attach
M9e attach after LineC-channel stabilization
```

Not allowed:

```text
dataset-specific timing。
```

Timing must be based on train-step phase or train-stream state only.

---

### M10：Recovery-only / control deconfound

Purpose:

```text
确保 positive 不是 recovery mechanism 自己带来的。
```

Controls:

```text
NoOp matched overhead
Random pulse + same recovery
Recovery-only
AdamW extra steps matched time
SGD extra steps matched time
Random block same rank
Same active fraction
Same norm
Same projection retention
MLP same mechanism
```

## 7.4 FU 指标

每个 carrier x mechanism x dataset x seed x horizon 必须记录：

```text
carrier
mechanism
dataset
seed
horizon
source_vs_best_control
source_vs_random
source_vs_noop
source_vs_adamw
source_vs_sgd
source_retention_h800_over_h100
source_retention_h1600_over_h800
source_retention_h3200_over_h1600
dataset_seed_pass
worst_source
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
AdamW_overwrite_cosine
optimizer_overwrite_cosine
FU_vs_grad_cosine
FU_vs_optimizer_step_cosine
signal_channel_projection
reservoir_projection
noise_signal_leakage_delta
control_equivalent
KAN_specific_delta_vs_MLP_same_mechanism
step_time_ratio
memory_ratio
```

## 7.5 FU Gates

### S2 weak signal

```text
carrier x mechanism 9-row mean source_h800 >= 0.005
dataset_seed_pass_count_h800 >= 3/9
source_retention_h800_over_h100 >= 0.40
at least one of:
  tail_recovery_rate_h800 >= 0.40
  LineC_recovery_rate_h800 >= 0.40
AUCtime_ratio_h800 <= 1.10
matched controls fail
```

### S3 retained source

```text
carrier x mechanism 9-row mean source_h1600 >= 0.005
dataset_seed_pass_count_h1600 >= 4/9
source_retention_h1600_over_h800 >= 0.50
tail_recovery_rate_h1600 >= 0.50
LineC_recovery_rate_h1600 >= 0.50
controls fail
```

### S3b delayed source

For late rebound:

```text
h1600 mean <= 0
h3200 mean >= 0.005
late_rebound_repeats_in >= 2 independent reruns
random/control late rebound absent
debt at h3200 recovered
```

This is exploration only, not promotion.

### S4 real-transfer exploration

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail debt recovered
LineC debt recovered
controls fail
```

### S5 official success

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
controls fail
promotion_allowed = 1
```

## 7.6 FU 可视化

```text
source_retention_curves_by_carrier_mechanism.svg
h800_h1600_h3200_source_heatmap.svg
debt_recovery_curves_tail_linec_calibration_auc.svg
AdamW_overwrite_cosine_heatmap.svg
MLP_vs_KAN_same_mechanism_delta.svg
signal_channel_vs_reservoir_projection.svg
mechanism_noncollapse_cosine_matrix.svg
control_attribution_waterfall.svg
late_rebound_repeatability_plot.svg
```

---

# 8. 4GPU Dynamic Queue

## 8.1 GPU roles

Initial assignment:

```text
GPU0:
  D-FOU kernel repair + D-CHE kernel repair smoke.

GPU1:
  MLP full functional mechanism matrix.

GPU2:
  D-CHE full functional mechanism matrix.

GPU3:
  LQ/Rational/D-RBF/D-WAV functional smoke + kernel repair.
```

Dynamic refill:

```text
If GPU0 queue drains:
  take remaining D-CHE/D-FOU kernel jobs, then FU controls.

If GPU1 queue drains:
  take FU h1600/h3200 extensions or MLP controls.

If GPU2 queue drains:
  take D-CHE h1600/h3200 or kernel repair.

If GPU3 queue drains:
  take LQ/RAT/RBF/WAV missing smoke, then all-basis efficiency rows.
```

## 8.2 Required queue artifacts

```text
v18_runnable_queue.csv
v18_gpu_assignment_manifest.csv
v18_gpu_utilization_dashboard.csv
v18_idle_violation.csv
v18_deferred_items.csv
v18_queue_drain_report.csv
```

Hard rule:

```text
If runnable_queue nonempty and any GPU idle > 10 minutes:
  execution_contract_violation = 1
  final route cannot be completed-no-go.
```

## 8.3 Job priorities

```text
P0:
  S0.2 code correctness.

P1:
  D-FOU/D-CHE kernel repair;
  MLP functional full matrix;
  D-CHE functional full matrix.

P2:
  LQ/RAT functional smoke;
  D-RBF/D-WAV sparse kernel smoke;
  controls.

P3:
  h1600/h3200 extension for top source-retaining rows;
  late rebound repeatability;
  figures / dashboards.
```

---

# 9. v18.0 成功标准

## S0.2 code correctness

All code/metric/profiler gates pass.

## S1 execution coverage

Must complete:

```text
S0.2 packet
D-FOU kernel repair
D-CHE kernel repair
MLP full FU matrix
D-CHE full FU matrix
LQ/RAT/RBF/WAV smoke
full efficiency truth table
raw debt matrices
4GPU queue drain
required figures
```

## S2 efficiency exploration

At least one non-MLP basis satisfies:

```text
forward_ratio <= 1.75
step_ratio <= 1.75
memory_ratio <= 1.10
gradcheck pass
```

## S3 efficiency near-breakthrough

At least one non-MLP basis satisfies:

```text
forward_ratio <= 1.50
backward_ratio <= 1.50
step_ratio <= 1.50
memory_ratio <= 1.05
```

## S2-FU weak signal

At least one carrier x mechanism satisfies S2 weak signal.

## S3-FU retained source

At least one carrier x mechanism satisfies S3 retained source.

## S4 real-transfer exploration

At least one carrier x mechanism satisfies real-lite 6/9.

## S5 official

Strict official gate remains unchanged.

---

# 10. Stop / continue rules

## 10.1 Not allowed to hard stop

Codex must not hard stop when:

```text
D-CHE FU fail
MLP FU fail
D-FOU kernel fail
D-CHE kernel fail
LQ reanchor fail
Rational smoke fail
D-RBF/D-WAV fail
h800 fail
h1600 fail
LineC immediate fail
tail immediate fail
AdamW controls positive
MLP generic positive
late rebound appears
efficiency forward blocked
```

These must go to:

```text
fallback ladder
failure taxonomy
next hypothesis queue
deferred_items with exact budget reason
```

## 10.2 Allowed hard stop

Only these hard stop:

```text
required artifact missing
forbidden information violation
no-action-search violation
direction uses validation/test/future/query
direction uses LineC/CEp99/NLL/ECE/AUCtime/Brier
dataset-name branch
seed-specific scale
action token/controller/reset route
fake/proxy/CPU offload
S0.2 code correctness fail
```

## 10.3 If condition fails, Codex must try these next

### If FU source h800 positive but h1600 washed out

Try in same run queue:

```text
M4 slow-state FU
M6 slow-state SNR FU
M5 matrix-block FU
AdamW overwrite diagnostic
SGD/Momentum recovery
```

### If AdamW overwrite cosine is negative

Try:

```text
SGD/Momentum-primary
FU-primary
AdamW moment freeze around FU pulse
role-partition optimizer
```

### If MLP succeeds but KAN fails

Try:

```text
D-CHE matrix-block FU
D-FOU if efficiency near-pass
LQ late attach
KAN carrier reparameterization
```

### If KAN succeeds but MLP fails

Run:

```text
KAN-specific attribution
same-param MLP stronger controls
S4 real-lite
efficiency hard gate
```

### If all FU mechanisms fail

Do not add tokens. Write:

```text
Current observable FU family no-go.
Next should be new signal-channel estimator or new carrier primitive.
```

### If D-FOU forward remains >2.25

Try:

```text
table lookup
lower band count
fused contraction
sincos recurrence
band-readout fused backward
```

### If D-CHE forward remains >2.25

Try:

```text
low-degree active bank
no-materialize recurrence
fused contraction
audit separation
```

### If debt metrics missing

Experiment invalid. Fix debt accounting before any route.

---

# 11. Required final artifacts

v18.0 must output:

```text
v18_route_decision.json
v18_code_review_packet.zip
v18_efficiency_truth_table.csv
v18_efficiency_blocker_table.csv
v18_kernel_repair_matrix.csv
v18_functional_raw_horizon_matrix.csv
v18_source_retention_matrix.csv
v18_debt_accounting_matrix.csv
v18_control_attribution.csv
v18_kan_vs_mlp_attribution.csv
v18_adamw_overwrite_diagnostics.csv
v18_mechanism_noncollapse_matrix.csv
v18_gpu_queue_drain_report.csv
v18_failure_taxonomy.csv
v18_next_hypothesis_queue.md
v18_no_go_boundary.md
figures/*.svg
```

---

# 12. 最后判断

v18.0 的核心不是继续扩大实验表面，而是把项目从“局部 positive / vague no-go”推进到两个可裁决的问题：

$$
\boxed{
\text{Functional: source 为什么不留存？}
}
$$

$$
\boxed{
\text{Efficiency: basis 为什么 forward 慢？}
}
$$

如果 v18.0 能做到以下任一项，就是实质进展：

```text
1. D-FOU 或 D-CHE forward ratio 被真正压到 <=1.75；
2. MLP FU 在 h1600/h3200 形成 retained source；
3. D-CHE/KAN FU 在同机制下超过 MLP 且 source retained；
4. AdamW 被证明会 wash out FU，且 AdamW-free FU 保留 source；
5. PopRisk/SNR slow-state 能预测并保留 h1600 source；
6. late rebound 在 repeated runs 中复现并伴随 debt recovery。
```

如果这些都没有发生，v18.0 必须明确写：

```text
No current FU observable family has retained source.
No current basis family has MLP-like forward kernel.
Next route requires either a new carrier primitive or a new signal-channel estimator.
```

而不是再继续补 token。
