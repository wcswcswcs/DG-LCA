# DG-KAN v20.0：Source-Channel Functional Update + Basis Kernel Officialization + 4GPU 动态并行完整计划

> 版本：v20.0 execution plan  
> 生成时间：2026-06-03  
> 目标：停止“每轮换方法名但结论相同”的循环，把问题拆成两个必须同时突破的主线：  
> **主线 A：Functional Update 是否能把 source 写入长期可保留的 signal channel。**  
> **主线 B：KAN 基函数是否能在 forward / backward / update / memory 上进入 same-param MLP 的效率包络。**  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no label-informed initialization；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / Brier / AUCtime 只能作为 audit / gate / debt readback，不能作为方向源；不重启 action bank / controller / reset route。

---

# 0. 项目总目标与 v20 当前状态

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，而是在 strict FC-PureKAN 上证明：

$$
\boxed{
\text{PureKAN base + functional update}
>
\text{same-param MLP / PureKAN ordinary gradient baseline}
}
$$

这个不等式必须同时包含三类证据：

```text
1. Functional evidence:
   functional update 的 source 不只是 h800 局部正，而要进入 h1600/h3200/h4800 retained signal channel；
   bad debt 可以偿还；matched controls 不能解释。

2. KAN-specific evidence:
   如果 MLP 同机制也成功，结果只能写成 generic training-dynamics insight；
   只有 KAN carrier 在 same mechanism 下额外胜出，才允许写 KAN-specific。

3. Efficiency evidence:
   basis forward / backward / update / memory 接近 same-param MLP；
   不能靠慢很多的 basis path 换取局部 source。
```

v19 的真实状态是：

```text
CodeRoute:
  S0_3-CodeMetricMechanismPassed。

EfficiencyRoute:
  E2-D-FOU-Pass; E2-D-CHE-Pass。

FunctionalRoute:
  S3-MLPGenericRetainedSource;
  H10M31PartialPositive;
  H11IndependentConfirmationFailed;
  H12M32PostAdamWSmokeNegative;
  H13H10H11AnatomyNoRepro;
  NoActionableSourceStateFamilyRepair。
```

这说明 v19 有两个真实推进：

```text
1. Efficiency:
   D-CHE / D-FOU 从全家族 ForwardBlocked 推进到 E2 exploration efficiency pass。

2. Functional:
   MLP 上出现了 retained generic source；
   但 KAN carrier 上没有稳定 retained source。
```

但 v19 仍不是项目目标成功，因为：

```text
1. H10 M31 的 D-CHE partial positive 没有被 H11 independent rerun 复现；
2. H12 post-AdamW recompute M32 为负；
3. H13 判断 same-family M31/M32 没有可继续小修方向；
4. H4 function-space actuation route 有 readback precedence 需要复核；
5. D-CHE / D-FOU efficiency 只是 E2 exploration pass，还不是 official fused/no-materialize kernel closure；
6. H10-H13 continuation source code 必须完整打包复审；
7. S4/S5 仍为 0。
```

所以 v20 的总目标不是“继续 v19 的 M31/M32”，而是：

$$
\boxed{
\text{把 MLP retained source 拆明白，}
\text{把 KAN source-channel writer 重构，}
\text{把 D-CHE / D-FOU efficiency E2 推到 official-ish full-loop closure。}
}
$$

---

# 1. v20 必须回答的科学问题

v20 不允许最后只写一个模糊 no-go。至少必须回答下面八个问题。

## 1.1 Functional update 问题

**Q1. MLP M16 retained source 为什么能成立？**

v19 中 MLP / M16 是当前唯一 retained source。它可能来自：

```text
1. MLP dense matrix carrier 更适合写入 source；
2. M16 two-phase / alternating 机制真的写入了 slow signal；
3. SGD/Momentum 或非 AdamW 路径保留了 source；
4. LineC-filtered style source writer 只是偶然匹配 MLP controls；
5. source 其实来自 recovery dynamics，而不是 functional update。
```

v20 必须把 MLP M16 拆成机制层级，而不是直接拿去 KAN 上套。

**Q2. AdamW 是否在 KAN 上洗掉 functional source？**

我们不能继续默认 `AdamW + FU residual`。必须比较：

```text
AdamW-primary + FU residual；
SGD/Momentum-primary + FU；
FU-primary；
FU-only smoke；
slow-state FU；
matrix/block FU；
function-space actuation FU；
role-partition optimizer。
```

如果 MLP/FU 在非 AdamW 条件下能留 source，而 AdamW 条件下不能，说明 AdamW 可能在洗掉 FU source。

**Q3. KAN carrier 为什么无法承载 MLP retained source？**

D-CHE / D-FOU 的效率已经出现 E2 pass，但 functional M16/M17 在 KAN 上没有 retained source。v20 要判断失败来自：

```text
1. KAN basis-channel source 不可见；
2. source 写入了 basis 但没有 actuation 到 output；
3. source 写到 high-degree/high-frequency hazard channel；
4. KAN readout / basis state 耦合方式不对；
5. current update still parameter-gradient-like，而不是 function-space source writer；
6. KAN kernel repair 后的 efficient path 和 functional path 不一致。
```

**Q4. H4 function-space actuation 是否被 route 误判？**

v19 的 H4 初始 smoke `ActuationR2` 很低，但 extended actuation 读数又很高。v20 必须复核：

```text
H4 extended actuation 是否真实；
它是否只是一种 diagnostic R2，而非 source-producing actuation；
为什么 route 没进入 long horizon；
是否 finalizer precedence 错误；
如果 actuation fidelity 真实存在，能否用它做 KAN source-channel writer。
```

**Q5. PopRisk / SNR 是否应该作为 source-state，而不是 one-shot gate？**

v13.6 one-shot PopRiskSNR no-go；v13.7 continuous SNR 在 MLP 上有 generic positive。过去这条线可能因为实现预算、代码浅层和 cover objective 偏移没有走到底。v20 要重开的是：

```text
PopRisk/SNR slow-state source writer，
不是旧的 one-shot SNR gate。
```

## 1.2 Basis efficiency 问题

**Q6. D-FOU / D-CHE E2 pass 能否变成 full-loop official-ish efficiency？**

v19 的 D-CHE / D-FOU exploration pass 很重要，但它必须经过：

```text
1. official fused/no-materialize kernel flag；
2. kernel gradcheck；
3. full training loop timing；
4. functional direction overhead；
5. LineC/horizon audit cost separated；
6. same-param MLP reference；
7. multi-batch scaling。
```

**Q7. LQ / Rational 是否还有效率推进价值？**

LQ 和 Rational 仍没有 functional success，但它们历史上都不是纯废线。v20 中它们不抢主预算，但必须给出：

```text
LQ forward / projection update bottleneck是否可修；
Rational denominator / Horner / branchless eval 是否可修；
如果不能修，给出明确退出 official path 的证据。
```

**Q8. RBF / Wavelet 是否应降级为 smoke-only？**

D-RBF / D-WAV 当前 forward ratio 太差。v20 中它们只做 sparse-local kernel smoke。如果 sparse-local fused path 仍不能把 forward ratio 降到 2.0 内，就不再进入 v20 functional official proof。

---

# 2. 哪些旧路线可能因为代码/指标问题仍有推进希望，哪些必须关闭

## 2.1 可按新语义重开的旧方向

### 2.1.1 PopRisk / SNR

v13.6 one-shot PopRiskSNR route 是 no-go，但 v13.7 说明 continuous SNR 在 MLP 上有 generic positive。旧问题是：它被当成 parameter gate / cover objective，而不是 source-channel slow-state writer。

v20 重开方式：

```text
不再直接用 SNR mask 更新参数；
先把 per-example gradient drift/diffusion 写入 slow state；
再测试该 slow state 是否预测 MLP M16 source retention；
最后才尝试 KAN basis-channel write-in。
```

### 2.1.2 Split-consensus / G7 lineage

v15.04-v15.6 说明 split-consensus metric-only 有 source，但 tail/LineC bad event 高、source/hazard 静态分离失败。旧错误是把 split-consensus 直接当 update。

v20 重开方式：

```text
split-consensus 只做 source estimator；
不直接提交参数 update；
source 要经过 function-space actuation / slow-state / matrix-block writer。
```

### 2.1.3 Function-space / operator-level actuation

v13.4 / H4 都提示 actuation 可能是关键。旧问题是 actuation fidelity 不足或 route readback 不一致。

v20 重开方式：

```text
先验证 output-space displacement 能否被 KAN basis/readout参数执行；
如果 ActuationR2 >= 0.70，才进入 long-horizon；
如果 ActuationR2 低，不能再用 parameter-gradient FU 冒充 function-space FU。
```

### 2.1.4 Graph-free analytic adjoint

v6.1 证明 graph-free manual gradient correctness 方向是成立的，但当时 primitive/cache/kernel efficiency 不过。v19 现在 D-CHE / D-FOU kernel 有 E2 pass，所以 graph-free / analytic adjoint 需要重新连接到 current efficient kernels。

v20 重开方式：

```text
D-CHE / D-FOU efficient path 上实现 graph-free / analytic adjoint smoke；
测真实 backward memory / forward / update；
如果 manual path 可行，FU-primary 才有可信实现基础。
```

### 2.1.5 MLP functional update

v19 的 MLP retained source 是当前最重要 FU 线索。MLP 不再只是 control，而是机制发现主线。

## 2.2 不应重启的旧方向

下面这些路线已经被足够证伪或容易重蹈旧错，v20 不允许重启：

```text
1. action bank / controller / action-token 扩展；
2. Rational reset / optimizer-state reset route；
3. cover_purity / cover_churn / cover_specialization 作为 objective；
4. G-token / N-token / Q-token 小变体扩展；
5. M31/M32 same-family alt-period/fu-lr 小修；
6. dataset-name branch / seed-specific scale；
7. 用 LineC / CEp99 / NLL / ECE / AUCtime 生成 direction。
```

如果 Codex 想走这些方向，必须自动写入 `forbidden_route_attempt.csv`，并转向预注册 fallback。

---

# 3. 当前研究进展给 v20 的启发

v20 不把外部优化器直接搬来当“再试一个 optimizer”。我们只抽取训练动力学原则。

## 3.1 Signal channel / reservoir theory

Generalization theory 里的核心启发是：训练输出空间可分为 signal channel 和 reservoir。coherent population signal 能 transfer；噪声如果进入 signal channel 才危险；reservoir 中的噪声对 test 不可见。对我们来说，FU 的目标不是单步 source，而是把 source 写入长期 signal channel。

v20 的指标因此从：

```text
source_h800 是否为正
```

升级为：

```text
source_h800/h1600/h3200/h4800 是否连续留存；
source 是否投影到 signal-channel readback；
noise 是否泄漏进 signal channel；
reservoir 是否只是暂存而没有 transfer。
```

## 3.2 Schedule-free / slow iterate

Schedule-free 和 averaging 研究启发 v20：训练可能需要 fast state 负责探索，slow state 负责保留 source。v19 的 MLP retained source 很可能不是单步 update 成功，而是某种 slow-state / alternating structure 起作用。

v20 因此设计：

```text
FU 不直接写参数；
先写入 source slow state；
只有当 slow state 与 split-consensus / PopRisk / LineC-channel readback 同向时，才小幅 actuation。
```

## 3.3 Dual-memory / old gradients

AdEMAMix 说明 old gradients 可能长期有用。v20 把 MLP M16 拆解为：

```text
short source state；
long source state；
short-long agreement；
short-long conflict；
old-source retention。
```

如果 long state 对 h3200/h4800 retention 有解释力，这就是一个真正推进。

## 3.4 Matrix/block optimizer research

Muon/SOAP 类研究说明 matrix/block space 可能比逐参数 AdamW 更适合 hidden-layer 更新。v20 不直接替换成 Muon，而是把 functional source 写入：

```text
MLP hidden matrix；
D-CHE degree-readout block；
D-FOU band-readout block；
LQ projection frame；
Rational numerator/denominator group block。
```

并记录 block update 是否比 parameter update 更能留住 source。

## 3.5 Dynamic manifold / boundary-conditioned training

Deep Manifold 的启发是：网络训练是 moving-coordinate / boundary-conditioned iteration，不是固定坐标的单步优化。v20 因此允许 early debt，但必须测：

```text
debt 是否被偿还；
fixed-point region 是否稳定；
source 是否在后续 horizon 保留。
```

---

# 4. v20 总体实验结构

v20 分为三条主线，每条都必须独立输出结果，不允许混成一个 vague route。

```text
Part A: Code / Metric / Mechanism audit
  确保指标、代码、route 和 mechanism 语义不再误导实验。

Part B: Basis kernel efficiency officialization
  把 D-CHE / D-FOU E2 pass 推到 official-ish full-loop closure；
  LQ/Rational/RBF/WAV 只做 targeted repair / smoke。

Part C: Functional update source-channel breakthrough
  以 MLP retained source 为机制发现锚点，
  重构 KAN source-channel writer，
  不再继续 M31/M32 同族小修。
```

四张 GPU 必须动态执行，不允许串行等待。

---

# 5. Part A：Code / Metric / Mechanism Audit

## 5.1 目标

v20 的 S0.4 目标是确保这轮不会因为错误实现、缺文件、route 口径或 mechanism 命名膨胀而再次原地打转。

必须通过的硬门：

```text
A0. Code packet completeness。
A1. H10-H13 source closure。
A2. LineC-fast / LineC-channel golden tests。
A3. Debt accounting completeness。
A4. Retention formula and route aggregation。
A5. Mechanism semantic contract。
A6. Efficiency profiler phase correctness。
A7. Kernel gradcheck and fused status。
A8. Repro command journal。
```

## 5.2 Codex 必须打包的文件

Codex 必须生成：

```text
v20_code_review_packet.zip
```

包内必须有：

```text
00_README.md
01_ENVIRONMENT/
02_SOURCE_TREE/
03_IMPORT_CLOSURE/
04_LINEC_FAST_CHANNEL/
05_RETENTION_DEBT_ROUTE/
06_MECHANISM_SEMANTICS/
07_FUNCTIONAL_SOURCE_CHANNEL/
08_OPTIMIZER_COUPLING/
09_EFFICIENCY_KERNELS/
10_EXPERIMENT_RUNNERS/
11_RAW_MATRICES/
12_FIGURES/
13_FAILURE_TAXONOMY/
14_REPRO_COMMANDS/
15_GPU_QUEUE/
packet_manifest.csv
packet_sha256_manifest.csv
```

必须包含源码：

```text
dgkan/metrics/linec.py
dgkan/fu/mechanisms.py
dgkan/fu/source_channel.py
dgkan/fu/slow_state.py
dgkan/fu/function_space_actuation.py
dgkan/fu/matrix_block.py
dgkan/profiling/efficiency_v20.py
dgkan/kernels/cheby_fused.py
dgkan/kernels/fourier_fused.py
dgkan/kernels/lq_fused.py
dgkan/kernels/rational_fused.py
dgkan/kernels/rbf_sparse.py
dgkan/kernels/wavelet_sparse.py
experiments/run_v20_s04_truth_gate.py
experiments/run_v20_mlp_retained_source_anatomy.py
experiments/run_v20_kan_source_channel_writer.py
experiments/run_v20_basis_kernel_officialization.py
experiments/run_v20_function_space_actuation.py
experiments/run_v20_merge_finalize.py
```

如果 H10-H13 continuation 仍然影响判断，必须包含：

```text
experiments/run_v19_h10_m31_full.py
experiments/run_v19_h11_m31_independent.py
experiments/run_v19_h12_m32_postadamw.py
experiments/run_v19_h13_anatomy.py
```

如果这些文件不存在，Codex 必须写：

```text
missing_source_reason
cannot_independently_audit_continuation = 1
final_route_blocked_by_missing_source = 1
```

## 5.3 S0.4 代码正确性检查

必须执行：

```bash
python -m compileall -q dgkan experiments
python experiments/run_v20_s04_truth_gate.py --check import_closure
python experiments/run_v20_s04_truth_gate.py --check linec_golden
python experiments/run_v20_s04_truth_gate.py --check debt_route
python experiments/run_v20_s04_truth_gate.py --check mechanism_semantics
python experiments/run_v20_s04_truth_gate.py --check efficiency_profiler
python experiments/run_v20_s04_truth_gate.py --check kernel_gradcheck
```

## 5.4 LineC / debt / route 检查

必须证明：

```text
LineC exception -> MeasurementInvalid，不进入 geometry fail 均值；
LineC-fast 只做 cheap split audit；
LineC-channel 记录 source-channel / reservoir / noise leakage readback；
CEp99 / NLL / ECE / Brier / AUCtime 只做 debt readback；
source retention 只在前一 horizon source > 0 时定义；
h3200 late rebound 不写成 retention；
single-row max 不打开 mechanism success。
```

Retention 公式：

$$
R_{H_2/H_1}
=
\frac{\max(0,S_{H_2})}{\max(\epsilon,S_{H_1})}.
$$

Debt recovery 公式：

$$
Recovery_H(D)
=
\frac{D_{peak}-D_H}{D_{peak}+\epsilon}.
$$

Group success 必须基于 9-row carrier × mechanism mean：

$$
\bar S_{c,m,H}
=
\frac{1}{9}\sum_{d,s} S_{c,m,d,s,H}.
$$

## 5.5 Mechanism semantic contract

每个 mechanism 必须有：

```text
mechanism_id
mechanism_family
is_true_implementation
is_smoke_only
update_kind
update_space
optimizer_primary
uses_adamw_primary
uses_sgd_momentum_primary
uses_slow_state
uses_matrix_block
uses_function_space_actuation
uses_poprisk_snr
matched_controls
expected_invariance
invalid_if_name_only_mask
```

如果 `M7-MatrixBlockFU` 仍只是 first-half row mask，它必须标记：

```text
is_true_implementation = 0
smoke_only = 1
cannot_close_matrix_block_route = 1
```

如果 `M9-FunctionSpaceOperatorFU` 仍只是 `0.5 * gradient`，必须标记：

```text
cannot_close_function_space_route = 1
```

---

# 6. Part B：Basis Kernel Efficiency Officialization

## 6.1 总目标

v19 的效率进展必须被 officialize。v20 的 efficiency 目标是：

$$
\boxed{
\text{D-CHE / D-FOU 的 E2 exploration pass 要进入 full-loop official-ish closure。}
}
$$

同时，LQ/Rational 做 targeted repair，RBF/WAV 做 sparse-local smoke。

## 6.2 统一 efficiency 指标

每个 basis 必须记录：

```text
carrier
kernel_variant
batch_size
hidden_size
parameter_count
same_param_mlp_id
forward_only_ms
basis_eval_ms
readout_contraction_ms
backward_input_ms
backward_param_ms
optimizer_update_ms
fu_direction_ms
fu_commit_ms
linec_audit_ms
horizon_readback_ms
step_total_ms
forward_peak_memory_mb
backward_peak_memory_mb
optimizer_state_memory_mb
basis_activation_bytes
functional_state_bytes
dense_materialized_bytes
kernel_count
custom_kernel_count
graph_break_count
compile_time_ms
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
audit_overhead_ratio
official_fused_kernel_complete
no_materialize_complete
gradcheck_pass
full_loop_timing_pass
```

## 6.3 D-FOU officialization

### Hypothesis B1

D-FOU 的 v19 pass 说明 Fourier family 不是天然慢，而是旧实现 forward/band eval 太差。如果使用 fused low-frequency recurrence / no-materialize band eval，它可以进入 MLP-like timing envelope。

### Implementation candidates

```text
FOU20-R2-officialize-lowfreq-k2-stream
FOU20-R3-fused-band-readout
FOU20-R4-k4-triton-no-materialize-officialize
FOU20-R5-bandwise-analytic-backward
FOU20-R6-full-loop-functional-overhead
```

### 必测指标

```text
sin_cos_eval_ms
band_eval_ms
band_readout_ms
band_backward_ms
basis_materialization_bytes
frequency_table_bytes
full_loop_forward_ratio
full_loop_step_ratio
functional_direction_overhead_ratio
```

### Gate

Exploration efficiency：

```text
forward_ratio_vs_mlp <= 1.50
step_ratio_vs_mlp <= 1.50
memory_ratio_vs_mlp <= 1.10
```

Official-ish kernel closure：

```text
forward_ratio_vs_mlp <= 1.25
step_ratio_vs_mlp <= 1.25
memory_ratio_vs_mlp <= 1.05
gradcheck_pass = 1
no_materialize_complete = 1
full_loop_timing_pass = 1
audit_overhead_ratio <= 0.15
```

### 不满足条件时 Codex 先尝试

```text
If forward_ratio > 1.50:
  try recurrence table precompute;
  try fused band-readout contraction;
  try reduce K active bands;
  try batch-major memory layout;
  try torch.compile/triton warm path;
  record component waterfall.

If backward_ratio > 1.50:
  implement bandwise analytic backward;
  split input-gradient and param-gradient timing;
  test no-materialize recompute backward.

If memory_ratio > 1.10:
  audit dense_materialized_bytes;
  remove [B,H,K] basis tensor;
  separate audit/horizon readback memory.
```

## 6.4 D-CHE officialization

### Hypothesis B2

D-CHE 是当前最强 KAN carrier。v19 中 D-CHE E2 pass 说明 no-materialize low-degree kernel 有希望。关键是确认 efficient path 与 functional path 一致。

### Implementation candidates

```text
CHE20-R2-low-degree-k3-triton-officialize
CHE20-R3-k4-triton-no-materialize-officialize
CHE20-R4-k3-gradbuf-triton-officialize
CHE20-R5-Clenshaw-readout-fusion
CHE20-R6-degree-role-functional-full-loop
```

### 必测指标

```text
cheby_recurrence_ms
degree_readout_contraction_ms
degree_backward_ms
degree_materialization_bytes
low_degree_active_fraction
high_degree_energy_fraction
full_loop_step_ratio
functional_direction_overhead_ratio
```

### Gate

同 D-FOU，但额外要求：

```text
low_degree_active_fraction >= 0.70
high_degree_energy_fraction <= 0.30
full-loop functional path uses same kernel variant as efficiency path
```

### 不满足条件时 Codex 先尝试

```text
If forward > 1.50:
  try Clenshaw recurrence;
  fuse degree-readout contraction;
  reduce active degree K but keep same param count via readout/share trick;
  test no-materialize recurrence.

If functional path slower than efficiency path:
  separate FU direction overhead;
  cache basis-channel telemetry;
  avoid LineC audit in training loop;
  confirm no dense readback in FU.
```

## 6.5 LQ / Rational / RBF / Wavelet

LQ and Rational remain secondary but not abandoned.

```text
LQ20:
  recurrence + fixed-frame cache + fused projection update;
  pass if forward <= 2.0 and step <= 1.5.

RAT20:
  Horner numerator/denominator;
  branchless denominator guard;
  telemetry separated from training path;
  pass if forward <= 2.0 and denominator safety unchanged.

RBF20:
  compact local K4 no dense materialization;
  pass if forward <= 2.0 in smoke.

WAV20:
  sparse support index-only forward/backward;
  pass if forward <= 2.5 in smoke.
```

---

# 7. Part C：Functional Update Source-Channel Breakthrough

## 7.1 总体目标

v20 的 FU 目标不是“再找一个 update 方法”，而是：

$$
\boxed{
\text{把 source 写入长期可保留的 signal channel，}
\text{并证明 KAN carrier 是否能承载这种 source。}
}
$$

必须同时判断：

```text
1. MLP retained source 机制是什么；
2. KAN 为什么不能承载同机制；
3. AdamW 是否洗掉 source；
4. function-space actuation 是否可行；
5. PopRisk/SNR slow-state 是否能预测 source retention；
6. matrix/block writer 是否优于 parameter writer。
```

## 7.2 统一 FU 指标

每个 candidate 必须记录：

```text
carrier
mechanism
optimizer_primary
source_state_type
update_space
actuation_method
horizon
source_vs_best_control
source_vs_noop
source_vs_random
source_vs_adamw
source_vs_same_recovery
source_vs_mlp_same_mechanism
source_retention_h800_over_h100
source_retention_h1600_over_h800
source_retention_h3200_over_h1600
source_retention_h4800_over_h3200
retained_candidate
late_rebound_candidate
washout_candidate
LineC_fast_debt_peak
LineC_fast_debt_final
LineC_fast_recovery
LineC_channel_debt_peak
LineC_channel_debt_final
LineC_channel_recovery
CEp99_debt_peak
CEp99_debt_final
CEp99_recovery
NLL_debt_peak
ECE_debt_peak
Brier_debt_peak
AUCtime_ratio
AdamW_overwrite_cosine
SGD_overwrite_cosine
slow_state_alignment
PopRisk_SNR_score
ActuationR2
ActuationResidualNorm
control_equivalent
KAN_specific_delta
```

## 7.3 Line F1：MLP M16 retained source anatomy

### Hypothesis F1

MLP M16 retained source is a real generic training-dynamics phenomenon. It is caused by a two-phase / alternating / slow-state source writer that writes source into dense matrix signal channel.

### Experiments

Run the same M16 mechanism with controlled ablations:

```text
F1-MLP-M16-replay-exact
F1a-remove-two-phase-warmup
F1b-remove-alternation
F1c-remove-LineC-filtered-source-estimator
F1d-replace-source-state-with-random-same-norm
F1e-SGD-momentum-only
F1f-AdamW-primary-only
F1g-slow-state-short-only
F1h-slow-state-long-only
F1i-short-long-agreement-only
F1j-matrix-block-hidden-weight-writer
F1k-output-readout-only-writer
```

### Required outputs

```text
mlp_m16_anatomy_matrix.csv
mlp_m16_hidden_projection.csv
mlp_m16_optimizer_overwrite.csv
mlp_m16_slow_state_trace.csv
mlp_m16_source_channel_trace.csv
```

### Gate

M16 mechanism is considered understood if:

```text
replay retained source >= 0.005 through h4800;
at least one ablation kills retention by >= 50%;
random same-norm source state does not retain;
control-equivalent fraction <= 0.50;
source channel trace predicts h4800 source with Spearman >= 0.30.
```

### 如果不满足，Codex 先尝试

```text
If replay fails:
  verify batch stream / seed / initialization match;
  rerun x3 independent offsets;
  if not reproducible, downgrade M16 to stochastic artifact.

If no ablation kills retention:
  retention may be recovery-only or generic training artifact;
  run recovery-only / NoOp overhead / random-source controls.

If slow-state traces do not predict retention:
  switch to matrix-block projection trace and hidden subspace trace.
```

## 7.4 Line F2：AdamW washout / optimizer coupling diagnostic

### Hypothesis F2

AdamW may wash out functional source, especially on KAN carriers. FU needs backprop/loss-interface signal, but not necessarily AdamW.

### Experiments

For MLP M16, D-CHE, D-FOU, run:

```text
F2a AdamW-primary + FU residual
F2b SGD-primary + FU
F2c Momentum-primary + FU
F2d FU-primary, gradient cotangent only
F2e FU-only smoke
F2f schedule-free / slow averaged-state FU
F2g role-partition optimizer:
    basis params = FU / source-state writer
    readout params = SGD or matrix-block
    scale/safety params = decoupled decay only
```

### AdamW overwrite metric

$$
O_{AdamW,H}
=
\frac{\langle \Delta\theta_{AdamW,0:H}, u_{FU}\rangle}
{\|\Delta\theta_{AdamW,0:H}\|\|u_{FU}\|+\epsilon}.
$$

If $O_{AdamW,H}<0$ and source decays, AdamW is likely overwriting FU.

### Gate

AdamW washout hypothesis supported if:

```text
AdamW-primary source_h1600/h3200 decays;
non-AdamW or slow-state condition retains source;
AdamW cumulative overwrite cosine <= -0.20;
matched controls fail.
```

### 如果不满足，Codex 先尝试

```text
If all optimizers wash out:
  source estimator likely wrong; go to F3/F4 source-channel writer.

If only AdamW works:
  AdamW not blocker; keep AdamW as recovery but still require KAN-specific attribution.

If SGD/Momentum works only on MLP:
  KAN carrier actuation is blocker; go to F5/F6.
```

## 7.5 Line F3：PopRisk / SNR slow-state source writer

### Hypothesis F3

Per-example gradient drift-diffusion can estimate population signal. But it should write to a slow source state, not directly mask gradients.

### Source score

For per-example gradients $g_i$, define:

$$
\bar g=\frac{1}{b}\sum_i g_i,
$$

$$
\Sigma_B=\frac{1}{b-1}\sum_i (g_i-\bar g)(g_i-\bar g)^T.
$$

For metric $M$:

$$
\Omega_B(M)
=
\bar g^T M\bar g
-
\frac{1}{b-1}\operatorname{tr}(M\Sigma_B).
$$

This score is not directly a direction. It updates a slow state:

$$
z_{t+1}
=
\beta z_t+(1-\beta)P_{signal}(\bar g,\Sigma_B,M).
$$

### Experiments

```text
F3a MLP-PopRiskSlowState
F3b MLP-PopRiskSlowState + matrix-block projection
F3c D-CHE-PopRiskDegreeReadoutState
F3d D-FOU-PopRiskBandReadoutState
F3e D-CHE-PopRiskReadoutOnlyActuation
F3f D-FOU-PopRiskReadoutOnlyActuation
```

### Gate

```text
PopRisk score must predict source_retention_h1600/h3200 with Spearman >= 0.25;
source_h1600 >= 0.005 for at least 4/9 rows;
random same-score permutation control fails;
noise-channel leak does not increase at h1600.
```

### 如果不满足，Codex 先尝试

```text
If PopRisk score is noisy:
  increase micro-split K;
  use exact per-batch variance instead of EMA;
  switch from parameter-level to block-level score.

If score predicts MLP but not KAN:
  use KAN readout-only actuation;
  avoid basis coefficient actuation until H4 actuation passes.
```

## 7.6 Line F4：Function-space actuation solver

### Hypothesis F4

Parameter-gradient FU fails because it does not execute desired output displacement. If function-space displacement can be actuated with high R2, KAN source writer may become viable.

### Desired displacement

Let source estimator propose output-space displacement $\Delta f_{sig}$ on train split $B$.

Solve:

$$
\alpha^*
=
\arg\min_\alpha
\|J_B U\alpha - \Delta f_{sig}\|_2^2
+
\rho\|\alpha\|_2^2.
$$

Commit:

$$
\Delta\theta=U\alpha^*.
$$

Actuation fidelity:

$$
ActuationR2
=1-
\frac{\|J_B U\alpha^* - \Delta f_{sig}\|^2}
{\|\Delta f_{sig}-\bar{\Delta f}_{sig}\|^2+\epsilon}.
$$

### Experiments

```text
F4a MLP function-space actuation replay
F4b D-CHE readout-only actuation
F4c D-CHE degree-readout actuation
F4d D-FOU band-readout actuation
F4e LQ projection-frame actuation smoke
F4f D-RAT group-readout actuation smoke
```

### Gate

```text
ActuationR2 >= 0.70;
ActuationResidualNorm <= 0.40;
source_h800 >= 0.005;
matched random target actuation fails;
LineC-channel debt not unrecovered by h1600.
```

### 如果不满足，Codex 先尝试

```text
If ActuationR2 < 0.20:
  target displacement not reachable; reduce subspace or switch carrier.

If ActuationR2 high but source fails:
  source estimator wrong; go back to F3.

If ActuationR2 high on MLP but low on KAN:
  KAN actuation basis wrong; test readout-only / low-degree-only / band-readout-only.
```

## 7.7 Line F5：Matrix/block source writer

### Hypothesis F5

FU source is matrix/block structured. Parameter-wise gradient masks destroy it. MLP retained source may live in hidden weight matrix; D-CHE/D-FOU source may need degree/band readout blocks.

### Experiments

```text
F5a MLP hidden-matrix block writer
F5b MLP output-readout block writer
F5c D-CHE degree-readout block writer
F5d D-CHE low-degree block writer
F5e D-FOU band-readout block writer
F5f LQ projection-frame block writer
```

Use block whitening / orthogonalized residual diagnostic, not full optimizer replacement.

### Metrics

```text
block_source_energy
block_noise_energy
block_rank
block_orthogonality_error
block_update_norm
source_retention
debt_recovery
controls
```

### Gate

```text
source_h1600 >= 0.005;
retention_h1600_over_h800 >= 0.50;
block random rotation control fails;
same active fraction control fails.
```

## 7.8 Line F6：KAN source-channel writer reconstruction

This line is only opened after F1-F5 identify at least one retained MLP mechanism or one high-actuation KAN subspace.

### Candidate writers

```text
F6a D-CHE readout-only source writer
F6b D-CHE low-degree-source / high-degree-reservoir writer
F6c D-FOU low-band-source / high-band-reservoir writer
F6d D-CHE source estimator + readout actuation + basis slow decay
F6e D-FOU source estimator + band-readout actuation
```

### Gate

```text
carrier x mechanism mean source_h1600 >= 0.005;
source_h3200 retained >= 0.50 of h1600;
dataset_seed_pass_h1600 >= 4/9;
KAN_specific_delta_vs_MLP_same_mechanism >= 0.005;
matched controls fail;
full-loop efficiency gate pass.
```

### 如果不满足，Codex 先尝试

```text
If source appears only in readout but not basis:
  keep readout writer; do not force basis coefficient update.

If source appears in basis but efficiency fails:
  link to Part B kernel repair before rerun.

If KAN-specific delta remains <=0:
  write generic FU insight, not KAN-specific; continue efficiency line separately.
```

---

# 8. Controls and attribution

Every positive result must compare against:

```text
NoOpMatchedOverhead
RandomMatchedNorm
RandomMatchedDirection
SameActiveFractionRandomMask
SameSourceStateRandomPermutation
RecoveryOnly
AdamWExtraStepsMatchedTime
SGDMomentumExtraStepsMatchedTime
MLP same carrier/mechanism analog
same efficiency path without FU
same FU path without source estimator
```

A result is control-equivalent if:

```text
best_control_source >= candidate_source - 0.005
```

KAN-specific claim requires:

```text
candidate_KAN_source - MLP_same_mechanism_source >= 0.005
```

with same horizon and same retention/debt gates.

---

# 9. 4GPU dynamic queue

v20 must use four GPUs as a dynamic queue, not static shard labels.

## 9.1 Queue files

Codex must output:

```text
v20_runnable_queue.csv
v20_gpu_assignment_manifest.csv
v20_gpu_utilization_dashboard.csv
v20_idle_violation.csv
v20_deferred_items.csv
v20_queue_drain_report.csv
```

## 9.2 GPU primary queues

```text
GPU0:
  Part A S0.4 tests;
  D-CHE kernel officialization;
  D-CHE source-channel writers.

GPU1:
  MLP M16 anatomy;
  AdamW-free / slow-state / matrix-block FU.

GPU2:
  D-FOU kernel officialization;
  D-FOU source-channel smoke;
  efficiency waterfalls.

GPU3:
  H4 actuation solver;
  LQ/Rational/RBF/WAV smoke;
  matched controls;
  figure generation.
```

## 9.3 Fill rules

If a GPU finishes its primary queue, Codex must pull jobs in this order:

```text
1. missing controls for positive rows;
2. full-loop efficiency confirmation for D-CHE / D-FOU;
3. h4800/h6400 extension for retained-source candidates;
4. H4 actuation rerun;
5. figure / dashboard generation;
6. code packet finalization.
```

Hard rule:

```text
If runnable_queue nonempty and any GPU idle > 10 min:
  execution_contract_violation = 1
  final route cannot be completed-no-go.
```

---

# 10. Success gates

## S0.4 Code / metric / mechanism gate

Must pass:

```text
compile/import closure;
LineC fast/channel golden;
debt accounting complete;
route aggregation unit tests;
mechanism semantic contract;
H10-H13 source closure;
efficiency profiler phase correctness;
kernel gradcheck;
repro command journal.
```

## S1 Efficiency official-ish closure

For D-CHE or D-FOU:

```text
forward_ratio_vs_mlp <= 1.25
step_ratio_vs_mlp <= 1.25
memory_ratio_vs_mlp <= 1.05
audit_overhead_ratio <= 0.15
gradcheck_pass = 1
no_materialize_complete = 1
full_loop_timing_pass = 1
```

## S2 Weak functional source-channel success

```text
carrier x mechanism 9-row mean source_h1600 >= 0.005
dataset_seed_pass_h1600 >= 3/9
source_retention_h1600_over_h800 >= 0.40
LineC_channel_recovery_h1600 >= 0.40 or CEp99_recovery_h1600 >= 0.40
AUCtime_ratio_h1600 <= 1.10
matched controls fail
```

## S3 Productive retained source

```text
carrier x mechanism mean source_h3200 >= 0.005
dataset_seed_pass_h3200 >= 4/9
source_retention_h3200_over_h1600 >= 0.50
CEp99_recovery_h3200 >= 0.60
LineC_channel_recovery_h3200 >= 0.60
AUCtime_ratio_h3200 <= 1.05
RandomMatchedNorm fails
RecoveryOnly fails
NoOpMatchedOverhead fails
```

## S4 KAN-specific exploration

```text
D-CHE or D-FOU source_h3200 >= 0.005
KAN_specific_delta_vs_MLP_same_mechanism >= 0.005
efficiency S1 pass
controls fail
dataset_seed_pass >= 6/9
```

## S5 official success, not lowered

```text
real_dataset_seed_pass_count = 9/9
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_channel_pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.25
official fused/no-materialize kernel pass if KAN carrier
controls fail
promotion_allowed = 1
```

---

# 11. Required metrics and visualizations

## 11.1 Tables

```text
v20_code_truth_gate.csv
v20_mechanism_semantic_contract.csv
v20_efficiency_truth_table.csv
v20_efficiency_waterfall.csv
v20_kernel_gradcheck.csv
v20_mlp_m16_anatomy.csv
v20_optimizer_washout_matrix.csv
v20_poprisk_slow_state_matrix.csv
v20_function_space_actuation_matrix.csv
v20_matrix_block_fu_matrix.csv
v20_kan_source_writer_matrix.csv
v20_controls_attribution.csv
v20_source_retention_matrix.csv
v20_debt_accounting_matrix.csv
v20_gpu_queue_drain_report.csv
v20_failure_taxonomy.csv
v20_route_decision.json
```

## 11.2 Figures

```text
fig_efficiency_forward_backward_step_by_carrier.svg
fig_efficiency_component_waterfall_dche_dfou.svg
fig_kernel_dense_vs_no_materialize.svg
fig_mlp_m16_ablation_retention.svg
fig_source_retention_horizon_curves.svg
fig_debt_recovery_curves.svg
fig_adamw_overwrite_cosine.svg
fig_poprisk_score_vs_source_retention.svg
fig_function_space_actuation_r2_vs_source.svg
fig_matrix_block_source_energy.svg
fig_kan_vs_mlp_same_mechanism_delta.svg
fig_gpu_utilization_timeline.svg
fig_failure_taxonomy_heatmap.svg
```

---

# 12. Failure taxonomy and next actions

v20 finalizer must produce a concrete route, not vague no-go.

## Case A: MLP M16 not reproducible

Route:

```text
R-A1-MLPGenericSourceNotReproducible
```

Action:

```text
Close M16 as stochastic artifact;
continue PopRisk/SNR and matrix-block source discovery on MLP;
do not port M16 to KAN.
```

## Case B: MLP retained source reproducible, KAN still fails

Route:

```text
R-B1-GenericFURetained-KANCarrierActuationFail
```

Action:

```text
Focus on KAN source-channel writer and actuation;
prioritize D-CHE/D-FOU readout-only / low-degree / low-band writers;
do not repeat same MLP mechanism directly on basis coefficients.
```

## Case C: AdamW washes out FU

Route:

```text
R-C1-AdamWWashoutConfirmed
```

Action:

```text
Move AdamW to baseline/control only;
advance FU-primary / SGD-Momentum / slow-state / role-partition optimizer.
```

## Case D: ActuationR2 remains low

Route:

```text
R-D1-FunctionSpaceActuationNotReachable
```

Action:

```text
Do not claim function-space FU;
switch to readout-only or block-level subspace;
revise carrier before further FU.
```

## Case E: D-FOU / D-CHE official efficiency fails

Route:

```text
R-E1-EfficiencyOfficializationFailed
```

Action:

```text
Keep functional experiments only on MLP;
continue basis kernel repair;
do not spend large FU budget on inefficient KAN path.
```

## Case F: D-FOU / D-CHE efficiency passes but FU fails

Route:

```text
R-F1-EfficientCarrierButNoSourceWriter
```

Action:

```text
Efficiency line succeeded;
functional blocker is source-channel theory;
continue MLP source theory and KAN actuation, not kernel repair.
```

## Case G: KAN-specific source retained

Route:

```text
R-G1-KANSpecificFunctionalExplorationPositive
```

Action:

```text
Run S4 real-lite 6/9 confirmation;
full controls;
h6400 extension;
prepare S5 protocol.
```

---

# 13. Final summary for Codex

v20 的执行重点不是“再跑一批方法”，而是交付两个明确突破或明确裁决：

```text
1. Efficiency:
   D-CHE / D-FOU 的 v19 E2 pass 是否能进入 official-ish full-loop efficiency closure。

2. Functional:
   MLP retained source 的机制是什么；
   KAN carrier 为什么不能 retain；
   是否存在 source-channel writer / actuation path 能让 D-CHE or D-FOU retain source。
```

Codex 不能再用 partial source / single-row max / late rebound / smoke-only mechanism / ForwardBlocked 来结束。每个失败必须写清楚：

```text
是 source estimator 失败？
是 actuation 失败？
是 optimizer washout？
是 debt 不恢复？
是 control-equivalent？
是 KAN carrier 不如 MLP？
是 kernel efficiency 不够？
```

v20 的最低可接受结果是：即使没有 S4/S5，也必须至少明确回答：

```text
MLP retained source 是否可复现；
AdamW 是否洗掉 source；
D-CHE/D-FOU efficiency 是否 official-ish 通过；
H4 actuation route 是否真实可用；
PopRisk/SNR slow-state 是否预测 retention；
KAN source-channel writer 是否有一个 9-row mean positive retained candidate。
```

只有这样，下一轮才不是继续原地打转。
