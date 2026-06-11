
# DG-KAN v22.0：Early-Source Retention Functional Update + D-CHE/D-FOU Full-Loop Kernel Officialization + 4GPU 动态并行完整计划

> 版本：v22.0 execution plan  
> 生成时间：2026-06-04  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 固定三段式：**Part A 代码审计**、**Part B 基函数效率**、**Part C Functional Update**。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / tail / calibration / AUC 只能作为 audit / gate / debt readback，不能生成方向；不能把 single-row positive、late rebound、control-equivalent、smoke、diagnostic 写成 promotion。

---

# 0. 项目总目标与 v21.01 当前真实状态

## 0.1 项目总目标

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，而是证明：

$$
\boxed{
\text{strict FC-PureKAN base} + \text{functional update}
>
\text{same-param MLP / PureKAN ordinary backprop controls}
}
$$

这里的“大于”必须同时满足四类要求：

```text
1. 表达与任务：
   不低于 same-param MLP，不能只在一个 dataset/seed 上侥幸好。

2. 效率：
   forward / backward / optimizer update / full step / memory 接近同参数量 MLP；
   KAN basis 不能靠慢很多换一点局部 gain。

3. Functional value：
   functional update 的收益必须超过 AdamW / SGD / Momentum / random matched / NoOp overhead / recovery-only / MLP generic controls。

4. 动态几何：
   允许短期 bad debt，但 source 必须长期留存，tail / LineC / calibration / AUC debt 必须被偿还。
```

最终不是证明：

$$
\text{某个 h800 source positive}
$$

而是证明：

$$
\boxed{
\text{source 从 early horizon 进入长期 signal channel，}
\text{并在 h3200/h4800/h6400 仍能打过 controls。}
}
$$

## 0.2 v21.01 的真实状态

v21.01 的 route 是：

```text
route = R2-LateReboundNoContinuousRetention
CodeRoute = S0_6-CodeMetricMechanismPassed
EfficiencyRoute = D-CHE_E1=8;D-FOU_E1=3;D-CHE_S1=6;D-FOU_S1=2
FunctionalRoute = MLP_h3200=0;KAN_h3200=0;KAN_h4800=0;independent=0;late_rebound=26
promotion_allowed = 0
```

这轮不是完全没有进展。它有两个真实推进：

```text
1. D-CHE / D-FOU efficiency repeat/warmup 路线继续打开；
   D-CHE 和 D-FOU 在部分 repeat rows 中已接近甚至优于 same-param MLP step envelope。

2. Functional update 的 blocker 更清楚：
   不是没有 late positive；
   而是没有 continuous retention。
```

但 v21.01 没有达成能力进展：

```text
1. MLP_h3200 = 0；
2. KAN_h3200 = 0；
3. KAN_h4800 = 0；
4. independent_confirmation_pass = 0；
5. late_rebound_groups 很多，但 continuous_retention_groups = 0；
6. stable-random / random controls 在 late horizon 也会 positive；
7. AdamW washout first-step 不足以解释失败；
8. official fused kernel status 与 efficiency repeat table 仍有口径不一致。
```

因此 v22.0 的核心不是继续扩大 late rebound，而是正面解决：

$$
\boxed{
\text{如何生成 early source chain，而不是 late rebound？}
}
$$

以及：

$$
\boxed{
\text{如何把 D-CHE / D-FOU 的 efficiency repeat pass 变成 full-loop official kernel evidence？}
}
$$

---

# 1. v22.0 的核心反思：为什么一直原地打转

## 1.1 原地打转的真正原因

过去很多版本不是没有执行，而是“尺子”和“问题定义”反复错位：

```text
1. 把 single-row max 当成机制进展。
2. 把 h3200 late positive 当成 source retention。
3. 把 ActuationR2 高当成 functional success。
4. 把 PopRisk/SNR readback 当成合法 direction selector。
5. 把 efficiency profiler pass 当成 full-loop kernel success。
6. 把 mechanism smoke 的名字当成真实机制实现。
7. 把 AdamW + FU residual 的失败当成 FU 全家族失败。
8. 把 delayed migration 当成可继续小修的希望。
```

v22.0 必须改成“证据优先”：

```text
任何 positive 必须回答：
  是 early source，还是 late rebound？
  是 continuous retention，还是 delayed migration？
  是 target-specific，还是 random/stable-random control 也能做到？
  是 KAN-specific，还是 MLP/generic 更强？
  是真实机制，还是 mechanism name 大于实现？
  是 profiler pass，还是 full-loop pass？
```

## 1.2 哪些旧方向可能因为代码/指标问题仍有推进希望

这些方向不能按旧结论直接判死，但必须换语义重开：

### 1.2.1 PopRisk / SNR

过去问题：

```text
1. 早期 PopRisk/SNR 多为 one-shot parameter SNR 或 readback；
2. v21.01 中 PopRisk/SNR 没有形成合法 retained-source selector；
3. 有些 estimator 只能事后解释 early source，不能 precommit 生成 direction。
```

重开条件：

```text
PopRisk/SNR 不再直接提交 parameter mask；
改成 early-source-chain estimator：
  预测 h100/h400/h800 是否形成正 source；
  预测 h1600/h3200 是否留存；
  必须在 train-only precommit 下工作；
  必须击败 stable-random / train-loss-drop selector。
```

### 1.2.2 Split-consensus / G7R lineage

过去问题：

```text
G7R 曾有 source，但 source/hazard 共线，tail/LineC bad event 高。
```

重开条件：

```text
Split-consensus 不再作为 direct update；
改成 source-target estimator：
  只负责提出 output-space / low-degree source target；
  不直接提交参数更新；
  必须看 early-chain retention，而不是 h800/h3200 late positive。
```

### 1.2.3 Function-space actuation

过去问题：

```text
H4 / v20 / v21 显示 high ActuationR2 可以做到；
但 high ActuationR2 source-success rows = 0。
```

重开条件：

```text
ActuationR2 只能证明“能动”；
下一步必须换 target definition：
  loss-cotangent target；
  cross-split consensus target；
  low-degree/low-frequency source-channel target；
  noise-reservoir separating target；
  random / sign-flip / corrupt target controls。
```

### 1.2.4 Graph-free analytic adjoint / operator-level basis channel

过去问题：

```text
早期 graph-free / operator-level 路线被效率和实现不成熟卡住；
D-CHE / D-FOU kernel 现在已接近 MLP-like envelope，旧失败可能不再完全适用。
```

重开条件：

```text
只在 D-CHE / D-FOU officialized kernel 上重开；
必须用 full-loop timing；
必须测 target actuation fidelity 与 source-retention，而不是只测 gradient correctness。
```

### 1.2.5 MLP functional update

过去问题：

```text
v19 MLP retained source 在 v20/v21 未复现；
但 MLP 仍是最强 generic FU 实验台。
```

重开条件：

```text
MLP 不再只是 control；
作为 source dynamics laboratory：
  M2 strong-washout；
  M15 weak-stable；
  M43 h1600 weak；
  F9/F10 hold/cycle；
  这些要用于理解 source dynamics，而不是追求 MLP promotion。
```

## 1.3 哪些旧方向不应重启

以下路线已经被反复证伪或目标错误，不再消耗主预算：

```text
1. action bank / controller；
2. reset route；
3. cover objective 小修；
4. G/N/Q-token 扩展；
5. M31/M32 同族小修；
6. 再调 alt period / warmup / fu_lr / target threshold；
7. dataset-name branch / seed-specific scale；
8. audit-metric-directed update；
9. 把 late positive 扩大成 route。
```

原因：

```text
1. action bank / controller 在旧版本显示 upper bound 不足或不可部署；
2. cover objective 已有 oracle validation invalid 证据；
3. M31/M32 partial positive 已经 independent confirmation 失败；
4. v21.01 F22-F36 一批修补都没有把 late rebound 变成 continuous retention。
```

---

# 2. 当前研究进展对 v22.0 的启发

v22.0 不机械换 optimizer，而是抽取训练动力学原则。

## 2.1 Signal channel / reservoir 理论

理论启发：

```text
训练运动只有进入 signal channel 才会 transfer；
噪声进入 signal channel 是真正风险；
reservoir 中的运动可能对 test 不可见；
grokking / delayed migration 可以理解为 signal 从 reservoir 迁移到 signal channel。
```

对 DG-KAN 的直接含义：

```text
late rebound 不等于 retained source；
h6400 positive 可能只是 delayed migration 或 control-equivalent drift；
我们必须构造 early-source-chain，让 source 从 h100/h400/h800 开始进入 signal channel。
```

v22.0 因此新增：

```text
SourceChainScore
EarlySourceDensity
SignalChannelEntry
ReservoirToSignalMigration
ControlEquivalentLateDrift
```

## 2.2 Deep Manifold / boundary-conditioned iteration

理论启发：

```text
神经网络不是在固定坐标里一次性收敛；
训练会构造 fixed-point regions；
boundary condition 决定轨迹；
短期扰动可以是必要的，但必须稳定成 fixed-point region。
```

对 DG-KAN 的直接含义：

```text
Functional update 不是 one-step correction；
应该是 boundary condition / source-channel writer；
如果只制造 late rebound 而没有 early chain，就没有稳定 fixed-point region。
```

## 2.3 Schedule-Free / fast-slow iterate

优化器启发：

```text
fast iterate 负责探索；
slow / averaged iterate 负责保留；
long-horizon stability 可能依赖 fast-state decay / averaging / schedule-free dynamics。
```

对 DG-KAN 的直接含义：

```text
M2 strong source washout 可能说明 raw source 不进入 slow state；
v22.0 要测 source 是否写入 slow state，而不是只看 parameter update。
```

## 2.4 AdEMAMix / dual-memory gradients

优化器启发：

```text
single EMA 不能同时重视近梯度和旧梯度；
dual EMA 可以同时保留 immediate past 与 older gradients。
```

对 DG-KAN 的直接含义：

```text
M2 strong-washout 与 M15 weak-stable 可能分别对应 short source 与 long source；
v22.0 要把 strong source 和 stable source 分解，而不是二选一。
```

## 2.5 Muon / SOAP / matrix-block optimization

优化器启发：

```text
matrix/block 参数不一定适合逐坐标 AdamW；
block-level orthogonalization / eigenbasis Adam 可能更适合保留结构化 source。
```

对 DG-KAN 的直接含义：

```text
D-CHE degree-readout；
D-FOU band-readout；
MLP hidden matrix；
这些应按 block/source-bank 更新，而不是逐参数 mask。
```

## 2.6 Cautious optimizer 的正确用法

启发：

```text
update 和 gradient 冲突是重要诊断；
但不能简单 hard mask，因为反向 update 可能是 productive boundary perturbation。
```

对 DG-KAN 的直接含义：

```text
Cautious 只做 conflict diagnostic：
  aligned source 是否更 retain？
  anti-aligned source 是否只是 late rebound？
  conflict 是否预测 washout？
```

---

# 3. v22.0 总体设计

v22.0 由四个并行工作包组成：

```text
Part A：代码审计 / metric truth / artifact closure
Part B：D-CHE / D-FOU efficiency officialization
Part C：functional update source-retention breakthrough
Part D：4GPU dynamic queue / execution acceleration
```

三个主问题：

```text
Q1:
  v21.01 的 efficiency pass 是否能变成 full-loop official evidence？

Q2:
  v21.01 的 late rebound 能否转成 early-source continuous retention？

Q3:
  能否构造合法 train-only source estimator，预测 h800 前 source chain，而不是事后解释 h6400 late positive？
```

---

# 4. Part A：代码审计 / metric truth / artifact closure

## 4.1 整体目标

v22.0 第一阶段不是训练，而是确保：

```text
1. 源码包自包含；
2. route 口径一致；
3. official fused kernel status 不冲突；
4. LineC / debt / retention / source-chain 指标正确；
5. mechanism 名称和实现一致；
6. 4GPU artifacts 完整。
```

如果 Part A 不通过，不允许写 scientific no-go。

## 4.2 Codex 必须打包的文件

Codex 必须输出：

```text
v22_code_review_packet.zip
```

目录结构必须是：

```text
00_README.md
01_ENVIRONMENT/
02_SOURCE_TREE/
03_IMPORT_CLOSURE/
04_LINEC_CORRECTNESS/
05_RETENTION_SOURCECHAIN_DEBT/
06_ROUTE_AGGREGATION/
07_FUNCTIONAL_MECHANISM_CONTRACTS/
08_OPTIMIZER_COUPLING/
09_EFFICIENCY_KERNELS/
10_PROFILER_CORRECTNESS/
11_4GPU_QUEUE/
12_RESULTS_TREE/
13_FIGURES/
14_FAILURE_TAXONOMY/
15_REPRO_COMMANDS/
packet_manifest.csv
packet_sha256_manifest.csv
```

必须包含以下源码：

```text
dgkan/metrics/linec.py
dgkan/metrics/debt.py
dgkan/fu/core.py
dgkan/fu/mechanisms.py
dgkan/fu/source_channel.py
dgkan/fu/source_state.py
dgkan/fu/function_space_actuation.py
dgkan/fu/matrix_block.py
dgkan/fu/poprisk_source.py
dgkan/fu/poprisk_snr.py
dgkan/fu/slow_state.py
dgkan/fu/source_chain.py
dgkan/kernels/che_official.py
dgkan/kernels/fou_official.py
dgkan/kernels/cheby_fused.py
dgkan/kernels/fourier_fused.py
dgkan/kernels/v17_basis.py
dgkan/profiling/efficiency_v20.py
dgkan/profiling/efficiency_v21.py
dgkan/profiling/efficiency_v22.py
experiments/run_v22_s07_truth_gate.py
experiments/run_v22_efficiency_officialization.py
experiments/run_v22_source_chain_dynamics.py
experiments/run_v22_function_space_target_reset.py
experiments/run_v22_kan_source_writer.py
experiments/run_v22_mlp_source_lab.py
experiments/run_v22_merge_finalize.py
```

如果任何文件是 compatibility shim，必须在：

```text
v22_compatibility_shim_manifest.csv
```

写清楚：

```text
shim_file
target_file
target_exists
reason
risk
```

## 4.3 必跑检查

### 4.3.1 Import closure

必须跑：

```bash
python -m compileall -q dgkan experiments
python experiments/run_v22_s07_truth_gate.py --check import_closure
```

记录：

```text
compileall_ok
modules_scanned
import_error_count
import_errors
proxy_guard_modules
core_import_pass
```

通过标准：

```text
compileall_ok = 1
core_import_pass = 1
import_error_count_core = 0
```

历史 proxy 脚本可以存在，但必须写入：

```text
legacy_proxy_guard_manifest.csv
```

且不能被 v22 runner import。

### 4.3.2 Official fused status consistency

必须比较三张表：

```text
v22_efficiency_truth_table.csv
v22_kernel_gradcheck_results.csv
v22_official_fused_status_matrix.csv
```

要求：

```text
同一 carrier / variant / batch 的 official_fused_kernel_complete 语义一致；
如果 truth table 写 1，kernel status 不能写 0；
如果 status 写 0，truth table 只能写 official_like_pass，不得写 official_pass。
```

通过标准：

```text
official_status_conflict_count = 0
```

否则 final route 必须是：

```text
R0-OfficialKernelStatusInconsistent
```

### 4.3.3 Source retention / source chain

新增指标：

$$
R_{800/400}
=
\frac{\max(0,S_{800})}{\max(\epsilon,S_{400})}
$$

$$
R_{1600/800}
=
\frac{\max(0,S_{1600})}{\max(\epsilon,S_{800})}
$$

$$
R_{3200/1600}
=
\frac{\max(0,S_{3200})}{\max(\epsilon,S_{1600})}
$$

定义 source chain：

```text
early_source_chain =
  S_h100 >= 0
  and S_h400 >= 0
  and S_h800 >= 0

continuous_retention_chain =
  S_h800 >= 0.005
  and S_h1600 >= 0.005
  and S_h3200 >= 0.005
  and R_1600_800 >= 0.40
  and R_3200_1600 >= 0.40

late_rebound =
  S_h800 < 0
  and S_h3200 > 0.005 or S_h4800 > 0.005
```

必须单元测试：

```text
positive h6400 but negative h800 => late_rebound, not retained
single-row positive but group mean negative => not mechanism positive
random control positive late => control_equivalent_late_drift
```

### 4.3.4 Debt recovery

必须记录完整 debt：

```text
tail_debt_peak
tail_debt_final_h800/h1600/h3200/h4800
tail_recovery_rate_h800/h1600/h3200/h4800

LineC_debt_peak
LineC_debt_final_h800/h1600/h3200/h4800
LineC_recovery_rate_h800/h1600/h3200/h4800

ECE_debt_peak
Brier_debt_peak
NLL_debt_peak
AUCtime_debt
```

debt recovery 公式：

$$
Recovery_D(h)
=
1 -
\frac{D_{final}(h)}{D_{peak}+\epsilon}
$$

通过标准：

```text
debt_metric_empty_count = 0
measurement_invalid_excluded = 1
```

### 4.3.5 Mechanism semantic contract

每个 FU mechanism 必须有：

```text
mechanism_id
mechanism_family
source_state_used
target_space
commit_space
uses_adamw_primary
uses_sgd_momentum_primary
uses_slow_state
uses_matrix_block
uses_function_space_actuation
uses_poprisk_snr
uses_linec_for_direction
uses_validation_test_future
semantic_implementation_summary
toy_expected_behavior
matched_controls
```

如果实现只是 gradient mask，不得命名为：

```text
FunctionSpaceOperatorFU
MatrixBlockFU
SourceChannelWriter
```

必须降级为：

```text
GradientMaskSmoke
```

通过标准：

```text
mechanism_name_overclaims_count = 0
```

## 4.4 Part A 失败时 Codex 先尝试什么

如果 `import_error_count_core > 0`：

```text
1. 补齐缺失源码；
2. 如果是 shim，补 target 文件或移除 shim；
3. 重跑 import closure；
4. 不允许进入 Part B/C。
```

如果 `official_status_conflict_count > 0`：

```text
1. 统一 official-like 与 official 的定义；
2. truth table 降级为 official_like_pass；
3. kernel status 补 gradcheck / no-materialize proof；
4. 重建 route。
```

如果 `debt_metric_empty_count > 0`：

```text
1. 补 LineC-channel / ECE / Brier / NLL debt；
2. 不允许写 S2/S3 functional route；
3. 只能写 S0 fail。
```

---

# 5. Part B：D-CHE / D-FOU Kernel Officialization

## 5.1 目标

v21.01 显示 D-CHE / D-FOU 在 repeat/warmup 下已经进入 MLP-like envelope。v22.0 要把这条线推进成真正 official kernel evidence。

目标：

$$
\boxed{
\text{D-CHE / D-FOU high-efficiency variants 在 profiler 与 full functional loop 中同时通过。}
}
$$

不是继续问：

```text
D-CHE / D-FOU 是否可能快？
```

而是问：

```text
它们是否已经能作为 v22 functional update carrier 的高效实现？
```

## 5.2 候选

D-CHE 主候选：

```text
CHE22-R2-low-degree-k3-official
CHE22-R4-k3-gradbuf-triton-official
CHE22-R4-k5-gradbuf-triton-official
CHE22-R5-batch512-forwardfix
```

D-FOU 主候选：

```text
FOU22-R2-lowfreq-k2-stream-official
FOU22-R3-tablelookup-bandreadout-official
FOU22-R4-k4-triton-no-materialize-official
FOU22-R5-batch512-forwardfix
```

MLP reference：

```text
same-param MLP batch 8/32/128/256/512
```

## 5.3 实验

每个 candidate 必须跑四种 timing：

```text
1. isolated profiler timing
2. full training loop timing
3. full functional update loop timing
4. audit-separated timing
```

batch grid：

```text
batch = 8, 32, 128, 256, 512
```

每个 batch 跑：

```text
warmup_steps = 50
measure_steps = 200
repeat = 3
```

## 5.4 必须记录指标

### Timing

```text
forward_only_ms
backward_grad_ms
optimizer_update_ms
fu_direction_ms
fu_commit_ms
linec_audit_ms
horizon_readback_ms
step_total_ms
full_loop_step_ms
full_functional_loop_step_ms
```

### Memory

```text
forward_peak_memory
backward_peak_memory
step_peak_memory
basis_activation_bytes
grad_buffer_bytes
functional_state_bytes
optimizer_state_bytes
audit_buffer_bytes
```

### Kernel status

```text
official_fused_kernel_complete
no_materialize_complete
gradcheck_pass
analytic_backward_pass
uses_torch_family_repair
fallback_to_reference_count
dense_basis_materialized
kernel_source_sha256
```

### Ratios

```text
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
full_loop_step_ratio_vs_mlp
functional_loop_step_ratio_vs_mlp
```

## 5.5 成功标准

### E1 exploration efficiency

```text
forward_ratio_vs_mlp <= 1.75
step_ratio_vs_mlp <= 1.50
memory_ratio_vs_mlp <= 1.10
gradcheck_pass = 1
```

### S1 official kernel efficiency

```text
official_fused_kernel_complete = 1
no_materialize_complete = 1
gradcheck_pass = 1
forward_ratio_vs_mlp <= 1.25
step_ratio_vs_mlp <= 1.25
memory_ratio_vs_mlp <= 1.10
full_loop_step_ratio_vs_mlp <= 1.25
functional_loop_step_ratio_vs_mlp <= 1.35
official_status_conflict_count = 0
```

### E2 batch512 robustness

```text
batch512_forward_ratio_vs_mlp <= 1.50
batch512_step_ratio_vs_mlp <= 1.50
batch512_memory_ratio_vs_mlp <= 1.10
```

## 5.6 可视化

必须生成：

```text
v22_efficiency_forward_ratio_by_batch.svg
v22_efficiency_step_ratio_by_batch.svg
v22_efficiency_memory_ratio_by_batch.svg
v22_full_loop_vs_profiler_timing.svg
v22_training_vs_audit_cost_waterfall.svg
v22_kernel_status_heatmap.svg
v22_batch512_blocker_breakdown.svg
```

## 5.7 失败时 Codex 先尝试什么

如果 D-CHE batch512 forward blocked：

```text
1. 检查 low-degree k3/k5 是否 materialize；
2. 降低 degree active bank，只保留 k3；
3. 把 degree-readout contraction 与 recurrence fuse；
4. 分离 LineC/horizon readback；
5. 重跑 batch512 only。
```

如果 D-FOU repeat pass 不稳定：

```text
1. 比较 tablelookup vs recurrence；
2. 固定 frequency buffer；
3. band-readout contraction fuse；
4. 重测 batch 128/256/512；
5. 如果仍不稳，保留 D-CHE 为主 carrier，D-FOU 降为 secondary。
```

如果 official status 冲突：

```text
1. truth table 降级 official-like；
2. 补源码 / gradcheck；
3. 不能把 efficiency 写成 official success。
```

---

# 6. Part C：Functional Update Source-Retention Breakthrough

## 6.1 目标

v21.01 的核心 functional blocker 是：

```text
late rebound groups = 26
continuous retention = 0
```

因此 v22.0 的 functional 目标不是“更强 h6400 positive”，而是：

$$
\boxed{
\text{生成 early source chain，并把它连续留存到 h3200/h4800。}
}
$$

## 6.2 核心假设

### H-FU1：当前目标只制造 delayed migration，不制造 early source

证据：

```text
KAN 多数 positive 是 h800 负、h3200/h4800 正；
stable-random control 也能 late positive。
```

预测：

```text
如果换成 early-source target，h100/h400/h800 必须先变正；
否则 late rebound 仍是 control-equivalent drift。
```

### H-FU2：strong source 与 stable source 是不同成分

证据：

```text
M2 类机制 h800 强，但 h1600/h3200 washout；
M15 类机制弱但更稳定；
F9/F10 能延到 h1600，但 h3200 失败。
```

预测：

```text
若用 dual-memory / source anchor 把 strong source 和 stable source 合并，
h800 source 不应牺牲 h3200 retention。
```

### H-FU3：function-space actuation 不是瓶颈，target 定义才是瓶颈

证据：

```text
high ActuationR2 rows 出现，但 source-success rows = 0。
```

预测：

```text
如果 target 定义正确，同等 ActuationR2 下应有 source chain；
如果 random/sign-flip/corrupt target 也 late positive，则 target 不可用。
```

### H-FU4：KAN source 需要 low-degree / low-frequency carrier，不应直接写全 basis

证据：

```text
D-CHE / D-FOU efficiency 已打开，但 functional source 未留存；
KAN late positive 多，continuous retention 少。
```

预测：

```text
低阶 / 低频 source bank + readout-only commit 可能比 full basis commit 更能保留 source。
```

### H-FU5：AdamW 不是唯一 blocker，但 optimizer state 仍影响 washout

证据：

```text
first-step AdamW overwrite 不能解释失败；
但 long-horizon source 仍会 washout。
```

预测：

```text
需要 cumulative optimizer projection，不只看 first-step cosine。
```

---

# 7. Functional Line C1：MLP Source Dynamics Laboratory

## 7.1 目标

MLP 不是 control，而是 source dynamics 实验台。先在 MLP 上回答：

```text
strong-washout source、weak-stable source、hold/cycle source 的区别是什么？
```

如果 MLP 上都不能产生 h3200/h4800 retention，KAN 上继续调 target 大概率低效。

## 7.2 候选

```text
MLP-C1-M2-StrongWashout
MLP-C2-M15-WeakStable
MLP-C3-M43-CycleHold
MLP-C4-M2PlusM15-DualMemory
MLP-C5-M2PlusM15-SlowAnchor
MLP-C6-M2WithScheduleFreeAveraging
MLP-C7-M2WithMatrixBlockSource
MLP-C8-M2WithCautiousConflictDiagnosticOnly
```

## 7.3 记录指标

```text
source_h100/h400/h800/h1600/h2400/h3200/h4800
source_derivative_h400_h800
source_derivative_h800_h1600
source_derivative_h1600_h3200
source_derivative_h3200_h4800
source_retention_h1600_over_h800
source_retention_h3200_over_h1600
tail_debt_recovery
LineC_debt_recovery
ECE_debt_recovery
Brier_debt_recovery
optimizer_cumulative_projection_on_source
momentum_state_alignment
slow_state_source_projection
hidden_matrix_source_projection
readout_source_projection
random_matched_source
stable_random_source
```

## 7.4 成功标准

MLP source dynamics S2：

```text
h800_source_mean >= 0.005
h1600_source_mean >= 0.005
h3200_source_mean >= 0.005
source_retention_h1600_over_h800 >= 0.40
source_retention_h3200_over_h1600 >= 0.40
stable_random_control_h3200 < candidate_h3200 - 0.005
```

MLP source dynamics S3：

```text
h4800_source_mean >= 0.005
tail_recovery_h3200 >= 0.50
LineC_recovery_h3200 >= 0.50
ECE/Brier not worse than best control
```

## 7.5 失败时 Codex 先尝试什么

如果 M2 strong source still washout：

```text
1. 增加 slow anchor，但不调 dataset/seed；
2. 把 source 写入 matrix-block slow state；
3. 比较 AdamW / Momentum / Schedule-free recovery；
4. 若仍 washout，M2 降级为 strong-source diagnostic，不再作为 update。
```

如果 M15 weak-stable too weak：

```text
1. 只放大 slow component，不放大 fast component；
2. 增加 dual-memory mixing；
3. 若 h800 仍 <0.005，M15 作为 stability reference。
```

如果 stable-random h3200 也 positive：

```text
1. 禁止 claim retained source；
2. 转入 control-equivalent late drift taxonomy；
3. 只保留 source-chain positive。
```

---

# 8. Functional Line C2：Early-Source Target Discovery

## 8.1 目标

构造能在 h100/h400/h800 就产生 source 的 target，而不是 late rebound。

## 8.2 Target families

```text
T1-loss-cotangent-target
T2-cross-split-consensus-target
T3-low-degree-CHE-target
T4-low-frequency-FOU-target
T5-readout-only-target
T6-source-reservoir-separating-target
T7-MLP-retained-target-transfer
TCTRL-random-matched-target
TCTRL-stable-random-target
TCTRL-sign-flip-target
TCTRL-corrupt-target
```

## 8.3 Target observability metrics

```text
ActuationR2
B1_gain
B2_transfer_gain
B3_safety_gain
target_random_gap
target_signflip_gap
target_corrupt_gap
early_source_h100
early_source_h400
early_source_h800
stable_random_late_positive_rate
```

## 8.4 Source-chain metrics

```text
early_source_density = mean(1[S_h100>0] + 1[S_h400>0] + 1[S_h800>0]) / 3
early_chain_pass = S_h100>=0 and S_h400>=0 and S_h800>=0.005
continuous_retention_pass = early_chain_pass and S_h1600>=0.005 and S_h3200>=0.005
late_rebound_only = S_h800<0 and S_h3200>0.005
```

## 8.5 成功标准

Target S2：

```text
B2_transfer_gain > random_B2_gain + 0.01
early_source_density >= 0.67
h800_source_mean >= 0.005
stable_random_h800_source < candidate_h800_source
```

Target S3：

```text
h1600_source_mean >= 0.005
h3200_source_mean >= 0.005
retention_h1600_over_h800 >= 0.40
retention_h3200_over_h1600 >= 0.40
random/signflip/corrupt controls fail
debt recovery not worse than controls
```

## 8.6 失败时 Codex 先尝试什么

如果 ActuationR2 高但 source 失败：

```text
1. 标记 target_wrong_not_actuation_wrong；
2. 切换 target family，不继续提高 ActuationR2；
3. 对比 loss target vs low-degree/low-frequency target；
4. 若所有 target local transfer 好但 h800 source 失败，说明 target-to-training dynamics 不成立。
```

如果 random target late positive：

```text
1. 加入 stable_random_late_drift baseline；
2. 禁止 h3200/h4800 alone 进入 gate；
3. 强制 early_chain criterion。
```

---

# 9. Functional Line C3：KAN Source-Channel Writer

## 9.1 目标

验证 D-CHE / D-FOU 是否存在能承载 retained source 的 channel。

KAN 不应直接复制 MLP update。要把 KAN source 写入稳定低阶 / 低频 / readout channel。

## 9.2 D-CHE writer candidates

```text
CHE-W1-basis-estimate-readout-commit
CHE-W2-low-degree-k3-source-bank
CHE-W3-degree-readout-block-writer
CHE-W4-low-degree-source-bank-plus-slow-state
CHE-W5-reservoir-separate-high-degree-null
CHE-W6-MLP-retained-target-to-CHE-low-degree
```

## 9.3 D-FOU writer candidates

```text
FOU-W1-band-estimate-readout-commit
FOU-W2-low-frequency-k2-source-bank
FOU-W3-band-readout-block-writer
FOU-W4-low-frequency-source-bank-plus-slow-state
FOU-W5-high-frequency-reservoir-null
FOU-W6-MLP-retained-target-to-FOU-lowfreq
```

## 9.4 必须记录指标

```text
source_h100/h400/h800/h1600/h3200/h4800
early_source_density
continuous_retention_chain
late_rebound_only
low_degree_energy_fraction
high_degree_reservoir_energy
low_frequency_energy_fraction
high_frequency_reservoir_energy
readout_commit_norm
basis_commit_norm
source_bank_norm
reservoir_bank_norm
source_to_reservoir_leakage
KAN_vs_MLP_same_target_delta
```

## 9.5 成功标准

KAN writer S2：

```text
h800_source_mean >= 0.005
early_source_density >= 0.67
KAN_vs_MLP_same_target_delta >= -0.02
stable_random_h800_control_fail = 1
```

KAN writer S3：

```text
h1600_source_mean >= 0.005
h3200_source_mean >= 0.005
retention_h1600_over_h800 >= 0.40
retention_h3200_over_h1600 >= 0.40
tail_recovery_h3200 >= 0.50
LineC_recovery_h3200 >= 0.50
```

KAN-specific S4 exploration：

```text
KAN_source_mean_h3200 - MLP_same_mechanism_source_mean_h3200 >= 0.005
or
KAN achieves same source with better efficiency / lower memory
```

## 9.6 失败时 Codex 先尝试什么

如果 low-degree / low-frequency writer late rebounds only：

```text
1. 不调 scale；
2. 检查 source estimator 是否 early-chain；
3. 把 commit 从 basis 改成 readout-only；
4. 若 readout-only 仍 late only，说明 target 不对。
```

如果 readout-only source positive but basis source negative：

```text
1. 记录 basis_carrier_mismatch；
2. 允许 KAN writer 只在 readout/low-bank commit；
3. 不再强行 full-basis FU。
```

如果 D-CHE fail but D-FOU pass：

```text
1. D-FOU 进入 primary KAN FU carrier；
2. D-CHE 保留 efficiency + baseline；
3. 不按历史 D-CHE 偏好覆盖证据。
```

---

# 10. Functional Line C4：PopRisk / SNR Retention Estimator Reset

## 10.1 目标

重建一个合法 train-only estimator，预测 retained source，而不是事后解释 late positive。

## 10.2 Estimator candidates

```text
E1-per-example-gradient-drift-diffusion
E2-output-space-cross-split-transfer
E3-source-chain-early-readout-free
E4-low-degree-signal-energy
E5-low-frequency-signal-energy
E6-matrix-block-source-coherence
E7-noise-reservoir-separation-score
ECTRL-train-loss-drop
ECTRL-gradient-norm
ECTRL-random-score
```

## 10.3 训练/选择协议

Estimator 只能使用：

```text
current train stream；
B1/B2/B3 split；
current logits；
per-example gradients；
basis activation telemetry；
optimizer state；
no validation/test/future/query；
no LineC/tail/AUC/calibration direction.
```

Estimator 不直接更新参数，先做 selector audit：

```text
top-k selected target；
bottom-k rejected target；
random selected target；
matched score control。
```

## 10.4 指标

```text
spearman_with_h800_source
spearman_with_h1600_source
spearman_with_h3200_source
AUC_predict_early_chain
AUC_predict_continuous_retention
AUC_predict_late_rebound
precision_topk_retained
recall_topk_retained
false_positive_late_rebound_rate
```

## 10.5 成功标准

Estimator S2：

```text
AUC_predict_early_chain >= 0.70
precision_topk_retained >= 0.30
false_positive_late_rebound_rate <= 0.50
beats train_loss_drop / grad_norm controls
```

Estimator S3：

```text
AUC_predict_continuous_retention >= 0.75
leave-one-dataset-out AUC >= 0.60
leave-one-seed-out AUC >= 0.60
```

## 10.6 失败时 Codex 先尝试什么

If all estimators fail:

```text
1. 不再扩大 target family；
2. 输出 SourceObservabilityNoGo；
3. functional update 下一步必须转向 supervised-but-legal? no, not allowed; instead reset representation/channel definition；
4. 只保留 MLP source lab 与 efficiency officialization。
```

If estimator predicts late rebound but not retention:

```text
1. 标记 late-drift predictor；
2. 禁止用于 update；
3. 用作 failure taxonomy only。
```

---

# 11. 4GPU 动态队列

## 11.1 总原则

v22.0 必须真正利用 4 张 GPU。不是静态 shard 完成就算，而是动态 runnable queue。

必须生成：

```text
v22_runnable_queue.csv
v22_gpu_assignment_manifest.csv
v22_gpu_utilization_dashboard.csv
v22_idle_violation.csv
v22_deferred_items.csv
v22_queue_drain_report.csv
```

硬规则：

```text
if runnable_queue_nonempty and any_gpu_idle_minutes > 10:
    execution_contract_violation = 1
    completed_no_go_allowed = 0
```

## 11.2 初始 GPU 分配

```text
GPU0:
  Part A truth gate；
  D-CHE kernel officialization；
  D-CHE writer W1-W6。

GPU1:
  MLP source dynamics lab；
  MLP M2/M15/M43/F9/F10 experiments；
  estimator validation on MLP。

GPU2:
  D-FOU kernel officialization；
  D-FOU writer W1-W6；
  D-FOU batch512 repair。

GPU3:
  target discovery T1-T7；
  PopRisk/SNR estimator E1-E7；
  controls / figures / packet generation。
```

## 11.3 Dynamic refill rules

If GPU0 finishes D-CHE early:

```text
take next D-FOU officialization repeat；
then KAN writer controls；
then figures.
```

If GPU1 finishes MLP early:

```text
run MLP independent confirmation；
then train-only estimator leaveout；
then controls.
```

If GPU2 finishes D-FOU early:

```text
run D-CHE batch512 fix；
then LQ/RAT targeted smoke；
then kernel status consistency.
```

If GPU3 finishes target discovery early:

```text
run KAN target long horizon；
then stable-random controls；
then packet.
```

---

# 12. Required artifacts

v22.0 必须输出：

## 12.1 Code / metric

```text
v22_code_review_packet.zip
v22_import_closure.csv
v22_required_source_files.csv
v22_linec_fast_golden.csv
v22_linec_channel_golden.csv
v22_debt_metric_unit_tests.csv
v22_source_chain_unit_tests.csv
v22_route_aggregation_unit_tests.csv
v22_mechanism_semantic_contract.csv
```

## 12.2 Efficiency

```text
v22_efficiency_truth_table.csv
v22_kernel_gradcheck_results.csv
v22_official_fused_status_matrix.csv
v22_full_loop_efficiency.csv
v22_functional_loop_efficiency.csv
v22_audit_cost_waterfall.csv
v22_batch512_repair_matrix.csv
```

## 12.3 Functional

```text
v22_mlp_source_dynamics_matrix.csv
v22_target_discovery_matrix.csv
v22_kan_source_writer_matrix.csv
v22_poprisk_estimator_matrix.csv
v22_source_retention_matrix.csv
v22_debt_accounting_matrix.csv
v22_control_attribution_matrix.csv
v22_adamw_overwrite_trajectory.csv
v22_late_rebound_taxonomy.csv
```

## 12.4 GPU / execution

```text
v22_runnable_queue.csv
v22_gpu_assignment_manifest.csv
v22_gpu_utilization_dashboard.csv
v22_idle_violation.csv
v22_deferred_items.csv
v22_queue_drain_report.csv
```

## 12.5 Figures

```text
v22_source_trajectory_grid.svg
v22_source_chain_vs_late_rebound.svg
v22_target_transfer_vs_retention.svg
v22_mlp_strong_vs_stable_source.svg
v22_kan_lowbank_source_writer.svg
v22_poprisk_estimator_auc.svg
v22_efficiency_full_loop_dashboard.svg
v22_kernel_status_consistency.svg
v22_gpu_utilization_dashboard.svg
```

---

# 13. Final gates

## S0：code / metric truth

```text
source_tree_complete = 1
compileall_ok = 1
core_import_error_count = 0
LineC_fast_golden_pass = 1
LineC_channel_golden_pass = 1
debt_metric_empty_count = 0
source_chain_unit_tests_pass = 1
route_aggregation_tests_pass = 1
mechanism_name_overclaims_count = 0
official_status_conflict_count = 0
```

## E1：efficiency exploration

```text
forward_ratio_vs_mlp <= 1.75
step_ratio_vs_mlp <= 1.50
memory_ratio_vs_mlp <= 1.10
gradcheck_pass = 1
```

## S1：official kernel efficiency

```text
official_fused_kernel_complete = 1
no_materialize_complete = 1
forward_ratio_vs_mlp <= 1.25
step_ratio_vs_mlp <= 1.25
memory_ratio_vs_mlp <= 1.10
full_loop_step_ratio_vs_mlp <= 1.25
functional_loop_step_ratio_vs_mlp <= 1.35
```

## FU-S2：weak early-source chain

```text
h800_source_mean >= 0.005
early_source_density >= 0.67
stable_random_h800_control_fail = 1
source_retention_h1600_over_h800 >= 0.30
```

## FU-S3：continuous retention

```text
h1600_source_mean >= 0.005
h3200_source_mean >= 0.005
source_retention_h1600_over_h800 >= 0.40
source_retention_h3200_over_h1600 >= 0.40
tail_recovery_h3200 >= 0.50
LineC_recovery_h3200 >= 0.50
stable_random_h3200_control_fail = 1
```

## FU-S4：KAN-specific exploration

```text
KAN_h3200_source_mean >= 0.005
KAN_h4800_source_mean >= 0.005
KAN_vs_MLP_same_mechanism_delta_h3200 >= 0.005
or KAN_same_source_with_better_efficiency = 1
controls fail
```

## S5：official success

```text
real_dataset_seed_pass_count = 9/9
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
Brier_delta <= 0.02
LineC_pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.25
official fused kernel gate pass
controls fail
promotion_allowed = 1
```

---

# 14. 不允许 hard stop 的情况

以下失败不能让 Codex 停止整轮：

```text
D-CHE writer fail；
D-FOU writer fail；
MLP source lab fail；
one target family fail；
PopRisk estimator fail；
batch512 efficiency fail；
h4800 fail；
stable-random late positive；
AdamW-free fail；
LineC immediate bad；
tail immediate bad；
official-like kernel pass but official status conflict。
```

必须进入：

```text
fallback ladder；
failure taxonomy；
next hypothesis queue；
deferred item with reason。
```

真正 hard stop 只有：

```text
source tree missing；
import closure core fail；
forbidden information violation；
direction uses validation/test/future/query；
direction uses LineC/tail/AUC/calibration；
dataset-name branch；
seed-specific scale；
fake/proxy/CPU offload；
required artifact missing；
runnable queue nonempty + GPU idle violation。
```

---

# 15. v22.0 最终必须回答的问题

这版不能再给模糊结论。最终必须回答：

```text
1. D-CHE / D-FOU efficiency 是否真正 officialized？
2. initial officialization 与 repeat/warmup 的差异来自哪里？
3. 是否存在 early-source chain？
4. late rebound 是否全部被 stable-random / control-equivalent 解释？
5. MLP 的 strong-washout 与 weak-stable source 的机制差异是什么？
6. 哪个 train-only estimator 能预测 retained source？
7. KAN 的 low-degree / low-frequency source bank 是否能承载 source？
8. function-space high ActuationR2 失败是 target 错还是 dynamics 错？
9. AdamW 是否长期 washout source？
10. 如果没有 retained source，是否应暂时停止 functional update family，转向 source observability theory？
```

---

# 16. 总结

v21.01 不是完全失败。它证明：

```text
1. D-CHE / D-FOU 的效率路线正在打开；
2. KAN 上确实有大量 late rebound / delayed migration；
3. MLP 上有 weak h1600 source 但 h3200 washout；
4. 目前没有 continuous retention；
5. source target theory 不够。
```

v22.0 的核心不是继续放大 late rebound，而是：

$$
\boxed{
\text{把 late migration 转成 early-source chain，}
\text{并把 D-CHE / D-FOU efficiency repeat pass 变成 full-loop official evidence。}
}
$$

如果 v22.0 仍没有 FU-S2 / FU-S3，那么下一步不能继续调 target family；必须承认当前 train-only source observability 不足，functional update 主线要从 update 设计转向 source theory reset。
