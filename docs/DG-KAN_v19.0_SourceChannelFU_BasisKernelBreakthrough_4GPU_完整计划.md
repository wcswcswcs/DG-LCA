# DG-KAN v19.0：Source-Channel Functional Update + Basis Kernel Efficiency Breakthrough + 4GPU 动态并行完整计划

> 版本：v19.0 execution plan  
> 生成时间：2026-06-02  
> 计划目标：停止“每轮换名字但同样 no-go”的原地打转，正面解决两个硬问题：  
> **1）functional update 为什么不能形成长期留存的 productive source；**  
> **2）KAN 基函数为什么仍不能达到同参数量 MLP 的 forward / backward / update / memory envelope。**  
> 公式格式：Typora 友好，仅使用 `$...$` 与 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no label-informed initialization；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / Brier / AUCtime 只能作为 audit / debt readback / final gate，不能作为方向源。

---

# 0. 本计划为什么必须大改

我们现在的问题不是“还差一个 FU token”，也不是“还差一个 trust scalar”，而是整个实验系统一直把不同层次的问题混在一起：

```text
代码/指标问题：
  LineC-fast 可用，但 LineC-channel / ECE / Brier debt 未完成；
  route 仍可能被 single-row max 误导；
  source retention / debt recovery 必须重新定义并实际落盘。

functional update 问题：
  source 在部分 horizon 出现，但没有连续留存；
  MLP 上中期 source 更明显，KAN-specific advantage 没站住；
  当前很多 mechanism 名字大于实现，只是 gradient mask / scaled gradient prototype。

basis efficiency 问题：
  non-MLP basis 主要被 forward / basis eval / dense materialization 卡住；
  memory ratio 多数接近 MLP，说明不是显存第一 blocker；
  census 已经反复做过，下一步必须做 family-specific kernel repair。
```

v18 的真实状态可以简化为：代码预检过了，但 evidence-complete 没过；fresh functional matrix 跑了 945 rows，但没有 h800/h1600 retained candidate；只有 D-CHE / M2 在 h3200 出现 late rebound；non-MLP efficiency pass rows 仍为 0。当前 route 是 `R1-S0_2DebtMetricsIncomplete`，functional route 是 `S3b-LateReboundNeedsIndependentRerun`，efficiency route 是 `R-EfficiencyForwardBlocked-D-CHE-D-FOU-D-RAT-D-RBF-D-WAV-LQ`。

所以 v19.0 的定位不是 v18 的“小修版”，而是：

$$
\boxed{
\text{先把代码尺子修到能判断，再用 4GPU 并行裁决真正的新 functional 机制和 basis kernel repair。}
}
$$

---

# 1. 项目总目标与当前进展

DG-KAN 的总目标仍然是：

$$
\boxed{
\text{在 strict FC-PureKAN base 上，通过 functional update 获得比普通 backprop / AdamW 更好的训练动力学，}
\text{同时保持与同参数量 MLP 接近的计算效率。}
}
$$

最终不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，而是证明一个更一般的系统结论：

```text
1. KAN basis / carrier 提供可用函数坐标；
2. functional update 能把训练运动推入更稳定的 signal channel；
3. 短期 bad debt 可以被长期训练动力学偿还；
4. 结果必须打过 AdamW、SGD/Momentum、random、NoOp、recovery-only、same-overhead、same-rank、same-norm、same-MLP mechanism controls；
5. 若 MLP 同样成功，写成 generic training-dynamics insight；只有 KAN 明显更强，才写 KAN-specific；
6. 每个 basis 的 forward / backward / update / memory 必须接近同参数量 MLP，不能靠慢很多换指标。
```

当前真实进展：

```text
代码审计：
  v17/v18 已把 import、LineC-fast、update semantics、kernel gradcheck 体系搭起来；
  但 debt accounting、LineC-channel、route aggregation 和 mechanism semantic contract 仍不足。

functional update：
  多轮证明 AdamW-coupled / residualized / function-metric / proximal / split-transfer / split-consensus static stabilization 没有达到 S4/S5；
  但 MLP/M2 h800 source、D-CHE/M2 h3200 late rebound、早期 PopRisk/SNR 线索说明“functional signal 完全不存在”这个结论还不能下。

basis efficiency：
  D-CHE、D-FOU、LQ、D-RAT、D-RBF、D-WAV 都没有达到 full MLP-like envelope；
  但 D-FOU/D-CHE 的 memory 接近 MLP，说明最值得先攻 forward / kernel path。
```

---

# 2. 这次必须解决的核心问题

v19.0 不再问一个模糊问题：

```text
functional update 有没有用？
```

而是拆成八个必须被裁决的问题。

## 2.1 代码/指标问题

### Q-Code-1：LineC 是否仍在误导我们？

LineC-fast 已经从异常即 fail 改成 MeasurementInvalid，这是正确修复。但它目前仍主要是 split CE improvement proxy。真正的“好几何”应该看 train motion 是否能进入可泛化 signal channel，而不是只看即时 CE improvement。

v19.0 必须实现两层 LineC：

```text
LineC-fast：
  cheap split-improvement audit，用于快速筛查明显撕裂 / noise leak。

LineC-channel：
  低频 signal-channel / reservoir / train-motion-to-probe-motion 审计，用于 h800/h1600/h3200 debt recovery。
```

LineC-fast 可以跑每个 row，LineC-channel 可以只跑 top candidates 和 matched controls，但必须真实落盘。

### Q-Code-2：source retention 是否被正确计算？

retention 不能等于某个 horizon 的 source 正数。必须定义为跨 horizon 保留：

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

$$
Retention_{h3200/h1600}
=
\frac{\max(0, Source_{h3200})}{\max(\epsilon, Source_{h1600})}
$$

如果前一 horizon mean source 非正，则后一 horizon 转正只能写作 late rebound，不能写作 retained source。

### Q-Code-3：debt recovery 是否真实测量？

短期坏 update 不能直接判死，但必须测 debt 是否被偿还。v18 的 debt rows 不完整，因此 v19.0 必须真实记录：

$$
Debt_{metric,peak} = \max_{\tau \in [t,t+H]} \max(0, metric(\tau)-metric_{baseline}(\tau))
$$

$$
Debt_{metric,final} = \max(0, metric(t+H)-metric_{baseline}(t+H))
$$

$$
RecoveryRate_{metric,H}
=
1-
\frac{Debt_{metric,final}}{Debt_{metric,peak}+\epsilon}
$$

其中 metric 至少包括：

```text
CEp99 / NLL / ECE / Brier / LineC-fast / LineC-channel / AUCtime。
```

### Q-Code-4：mechanism 名字是否大于实现？

v17/v18 中不少机制是 prototype，例如：

```text
MatrixBlockFU = 前半 rows mask；
FunctionSpaceOperatorFU = 0.5 * gradient；
SlowStateFU = 简单 gradient EMA。
```

v19.0 必须新增 **Mechanism Semantic Contract**：每个机制必须有数学定义、代码路径、toy correctness test、matched control、非坍缩审计。如果实现只是 prototype，必须命名为 smoke，不能用来关闭理论路线。

---

## 2.2 Functional update 问题

### Q-FU-1：AdamW 是否正在洗掉 FU source？

过去很多实验默认：

$$
\Delta\theta = \Delta\theta_{AdamW} + \eta \Delta\theta_{FU}
$$

但这只是 hybrid 架构时代的遗留默认。现在要测试：functional update 需要 backprop / loss cotangent，但不必绑定 AdamW。v19.0 要把 AdamW 从默认主干降级为一个条件。

必须比较：

```text
AdamW-primary + FU residual；
SGD-primary + FU；
Momentum-primary + FU；
Schedule-free / averaged-state + FU；
FU-primary；
FU-only smoke；
role-partition optimizer；
slow-state source writer；
matrix-block source writer。
```

并记录 AdamW overwrite：

$$
OverwriteCos_H
=
\cos\left(
\sum_{\tau=t}^{t+H}\Delta\theta_{AdamW,\tau},
\Delta\theta_{FU,t}
\right)
$$

以及 AdamW 对 FU direction 的累计投影：

$$
OverwriteProj_H
=
\frac{\left\langle\sum_{\tau=t}^{t+H}\Delta\theta_{AdamW,\tau},\Delta\theta_{FU,t}\right\rangle}
{\|\Delta\theta_{FU,t}\|^2+\epsilon}
$$

若 $OverwriteProj_H < -0.3$ 且 source 在 AdamW recovery 下消失，而在 SGD / slow-state recovery 下保留，则说明 AdamW 可能阻碍 FU。

### Q-FU-2：source 是不是没有进入 signal channel？

外部 generalization theory 给我们的核心启发是：训练中的 output space 可以分为 signal channel 与 reservoir，coherent population signal 才会 transfer，而 noise / idiosyncratic memorization 应该被压在 reservoir 中。因此 v19.0 不再只看 source_h800，而要看 source 是否进入 signal channel。

定义 train split 的 per-example output cotangent $\delta_i$ 与参数 Jacobian $J_i$：

$$
g_i = J_i^T \delta_i
$$

定义 population-risk / SNR source state：

$$
\mu = \frac{1}{B}\sum_i g_i
$$

$$
\sigma^2 = \frac{1}{B-1}\sum_i (g_i-\mu)^2
$$

$$
SNR = \frac{\mu^2}{\sigma^2/B + \epsilon}
$$

但 v19.0 不能只做 one-shot SNR。必须构造 slow-state signal writer：

$$
s_{t+1} = \beta s_t + (1-\beta)P_{signal}(u_t)
$$

其中 $P_{signal}$ 可由 split-consensus、PopRisk/SNR 或 function-space transfer operator 给出。

### Q-FU-3：MLP h800 source washout 到底说明什么？

v18 继续显示 MLP / M2 在 h800 有中期 source，但 h1600/h3200 洗掉。v19.0 必须把 MLP 当主实验台，不再只是 control。

要回答：

```text
1. MLP M2 h800 source 是真实 generic source 还是 horizon artifact？
2. 它为什么 h1600 洗掉？
3. slow-state / schedule-free / AdEMAMix-like dual memory 是否能保住？
4. 如果 MLP 都保不住 source，KAN 上继续堆 carrier 是否有意义？
```

### Q-FU-4：D-CHE h3200 late rebound 是噪声还是 delayed signal migration？

D-CHE / M2 在 h3200 有 late rebound，但 h100/h400/h800/h1600 均不连续。它不能 promotion，但它是线索。

v19.0 必须做 independent rerun：

```text
D-CHE / M2 exact rerun x3；
matched controls x3；
horizon = h800, h1600, h2400, h3200, h4800；
full debt / LineC-channel / AdamW overwrite / source-channel audit。
```

如果 late rebound 可复现且 controls 解释不了，则进入 delayed signal migration route；否则标记 stochastic rebound 并关闭。

---

## 2.3 Basis efficiency 问题

### Q-Eff-1：每个 basis 到底慢在哪里？

现在不能再只写 ForwardBlocked。必须给每个 basis 输出 waterfall：

```text
basis eval；
readout contraction；
backward input；
backward parameter；
optimizer update；
functional direction；
LineC audit；
horizon readback；
workspace / cache / materialization。
```

### Q-Eff-2：哪个 basis 最可能先被修成 MLP-like？

当前数据指向：

```text
D-FOU：memory 接近 MLP，但 forward 4-5x，优先做 sin/cos recurrence + fused band eval。
D-CHE：当前最强 KAN carrier，memory 接近 MLP，但 forward 4-6x，优先做 no-materialize recurrence。
LQ：forward / update 慢，作为 reanchor + fused projection repair。
D-RAT：rational eval / denominator path 慢，作为 branchless Horner/vectorized repair。
D-RBF / D-WAV：forward 太慢，除非 sparse-local fused path，否则低预算 smoke。
```

---

# 3. 之前被验证错误但可能因代码/指标问题仍有希望的路线

v19.0 不盲目重启旧路线，但要区分“真正证伪”和“可能被错误实现/错误指标误伤”。

## 3.1 可以重开的路线

### R-Open-1：PopRisk / SNR

v13.6-v13.8 已经把 source 从 future/oracle target 改成 train-stream per-example gradient statistics，这是合法方向。后来它没有形成 KAN-specific proof，但当时我们还没有现在的 debt accounting / LineC-channel / AdamW-free / slow-state。现在它值得作为 **source-channel slow-state writer** 重开，而不是作为 one-shot parameter SNR gate。

重开条件：

```text
必须有 slow state；
必须测 h800/h1600/h3200 retention；
必须有 MLP active line；
必须有 matched random SNR / same-active controls。
```

### R-Open-2：Split-consensus / G7R lineage

v15.04/v15.5 找到过 split-consensus metric-only source，但 bad event 高；v15.6/v15.7 静态 source/hazard 分离失败；v15.8/v16 后的 recovery 没闭合。这个路线不能按旧 G-token 继续，但可以作为 **source estimator** 重开：

```text
不再让 split-consensus 直接提交 parameter update；
它只负责构造 P_signal 或 slow source state；
提交由 slow-state / matrix-block / function-space operator 完成。
```

### R-Open-3：Function-space / operator-level basis-channel update

v13.2-v13.4 做过 loss-interface cotangent、basis-natural、operator-level basis-channel functional。这些路线当时受限于 code/import、projection、未来 probe、basis efficiency。现在要重开其中合法部分：

```text
loss-interface cotangent -> output-space displacement -> basis/channel actuation fidelity。
```

但必须禁止 future outcome / oracle target 进入方向；future 只能 diagnostic。

### R-Open-4：Graph-free analytic adjoint / manual update

v6.1 显示 graph-free gradient correctness 可以通过，但效率失败。现在 basis efficiency 已经定位 forward/kernel bottleneck，所以 graph-free path 不应被关闭，而应和 D-FOU/D-CHE kernel repair 合并。

### R-Open-5：MLP functional update

v14.2 有 generic MLP-FMS signal，v16-v18 也持续显示 MLP 是 strongest h800 source line。它不能丢。v19.0 把 MLP 当“FU 机制实验台”。

---

## 3.2 不应重启的路线

```text
1. action bank / controller / action token：v14.5 已显示 action-bank upper bound 不足，且 v9 系列长期陷入找动作错误。
2. reset route / optimizer-state reset：v14.8-v14.9 已被 generic optimizer-state confound 打回。
3. cover_purity / cover_churn / cover_specialization objective：v13.11 oracle validation 已判 invalid。
4. G-token / N-token / Q-token 小扩展：v15.0-v15.7 已证明同类静态小修无法形成 S4/S5。
5. dataset/seed branch 或 audit-metric-directed update：违反项目硬约束。
```

---

# 4. 当前研究进展给我们的启发

v19.0 借鉴研究进展时，不是把某个 optimizer 直接搬进来，而是抽取训练动力学原则。

## 4.1 Schedule-free / iterate averaging

启发：source 可能需要写入 slow state，而不是只提交到 fast iterate。v19.0 因此必须比较：

```text
fast-only FU；
slow-state FU；
schedule-free style averaged FU；
lookahead / EMA source retention。
```

## 4.2 AdEMAMix / older gradients

启发：旧梯度可能长期有用，单一 EMA 可能保不住 source。v19.0 因此加入 dual-memory source state：

$$
s^{short}_{t+1}=\beta_s s^{short}_t+(1-\beta_s)u_t
$$

$$
s^{long}_{t+1}=\beta_l s^{long}_t+(1-\beta_l)u_t
$$

只在 short / long agreement 或受控 disagreement 时提交。

## 4.3 Muon / SOAP / matrix-block optimization

启发：很多有意义的训练结构存在于 matrix/block space，而不是逐参数坐标。v19.0 因此必须实现真正 matrix-block FU，而不是前半 rows mask。

目标 block：

```text
MLP hidden weight matrix；
D-CHE degree-readout block；
D-FOU band-readout block；
LQ projection frame block；
Rational numerator/denominator group block。
```

## 4.4 Population-risk / signal-reservoir theory

启发：source 必须进入 signal channel；reservoir 中的变化可能 test-invisible；噪声进入 signal channel 才是危险。v19.0 因此把 FU 定义为：

$$
\boxed{\text{source-channel writer + carrier actuator + trajectory retention process}}
$$

而不是：

$$
\boxed{\text{one-step parameter perturbation}}
$$

## 4.5 Deep Manifold / boundary-conditioned iteration

启发：神经网络不是一步到 fixed point，而是在 boundary-conditioned iteration 中形成 fixed-point regions。v19.0 因此允许短期 bad debt，但要求长期 recovery；并将 FU 理解为 trajectory boundary，而不是单步安全 update。

---

# 5. v19.0 总体实验结构

v19.0 分成四条主线，并行执行。

```text
Line 0: S0.3 Code / Metric / Mechanism Truth Gate
  确保尺子正确，不能继续被错误指标误导。

Line A: Functional Update Breakthrough Matrix
  不再默认 AdamW + FU；测试 source-channel writer / slow-state / matrix-block / AdamW-free / function-space operator。

Line B: Basis Kernel Efficiency Breakthrough
  不再只 census；对 D-FOU / D-CHE / LQ / D-RAT / D-RBF / D-WAV 做 family-specific repair。

Line C: 4GPU Dynamic Execution + Evidence Packaging
  所有实验必须动态队列化，不允许有 runnable job 时 GPU 空闲。
```

---

# 6. Line 0：S0.3 Code / Metric / Mechanism Truth Gate

S0.3 是 v19.0 的第一硬门。没过 S0.3，不允许写 scientific no-go。

## 6.1 Codex 必须打包的文件

Codex 必须生成：

```text
v19_code_review_packet.zip
v19_results_bundle.zip
```

`v19_code_review_packet.zip` 必须包含：

```text
00_README.md
01_ENVIRONMENT/
  conda_env.yaml
  pip_freeze.txt
  git_status.txt
  cuda_info.txt
  exact_commands.sh

02_SOURCE_TREE/
  dgkan/
  experiments/
  tests/
  setup.cfg or pyproject.toml if present

03_IMPORT_CLOSURE/
  compileall_report.json
  import_closure_report.csv
  missing_imports.csv
  legacy_proxy_isolation.csv

04_LINEC_CORRECTNESS/
  linec_fast_source.py snapshot
  linec_channel_source.py snapshot
  linec_golden_tests.csv
  linec_exception_policy.csv
  linec_measurement_invalid_rows.csv
  linec_fast_vs_channel_consistency.csv

05_RETENTION_AND_DEBT/
  retention_formula_tests.csv
  debt_formula_tests.csv
  debt_metric_availability.csv
  tail_debt_golden.csv
  linec_debt_golden.csv
  calibration_debt_golden.csv
  auc_debt_golden.csv

06_ROUTE_AGGREGATION/
  route_aggregation_unit_tests.csv
  single_row_positive_counterexample.csv
  grouped_mean_recompute.csv
  retention_recompute.csv
  controls_explain_recompute.csv

07_UPDATE_SEMANTICS/
  update_tensor_schema.py snapshot
  update_sign_finite_difference.csv
  update_space_contract.csv
  update_kind_contract.csv
  update_writeback_trace.csv

08_FUNCTIONAL_MECHANISM_CONTRACTS/
  mechanism_manifest.csv
  mechanism_math_contract.md
  mechanism_semantic_noncollapse.csv
  mechanism_toy_correctness.csv
  mechanism_control_mapping.csv

09_OPTIMIZER_COUPLING/
  adamw_overwrite_diagnostic.csv
  sgd_overwrite_diagnostic.csv
  momentum_overwrite_diagnostic.csv
  slow_state_overwrite_diagnostic.csv
  role_partition_optimizer_manifest.csv

10_EFFICIENCY_PROFILER/
  profiler_source.py snapshot
  profiler_phase_unit_tests.csv
  optimizer_update_phase_tests.csv
  audit_cost_separation_tests.csv
  same_param_mlp_match_tests.csv

11_KERNEL_CORRECTNESS/
  kernel_gradcheck.csv
  dense_vs_nomaterialize_correctness.csv
  analytic_backward_correctness.csv
  fused_kernel_status.csv

12_RAW_EXPERIMENT_MATRICES/
  all raw functional matrices
  all raw efficiency matrices
  all raw debt matrices
  all raw controls matrices

13_GPU_QUEUE/
  runnable_queue.csv
  gpu_assignment_manifest.csv
  gpu_utilization_dashboard.csv
  idle_violation.csv
  queue_drain_report.csv

14_FAILURE_TAXONOMY/
  failure_taxonomy.csv
  no_go_boundary.md
  next_hypothesis_queue.csv

packet_manifest.csv
packet_sha256_manifest.csv
```

`v19_results_bundle.zip` 必须包含完整 official result tree：

```text
route_decision.json
all gate recompute json/csv
all required manifests
all raw matrices
all figures
all execution logs
all code packet sha manifest
```

缺任一关键文件，不能写 completed no-go。

## 6.2 S0.3 必须通过的测试

### Test 0.3.1：LineC-fast / LineC-channel correctness

必须通过：

```text
NoOpNull
RandomMatchedNormNull
KnownTransferPositive
NoiseLeakPositive
ReservoirOnlyPositive
BatchPermutationMismatch
ExceptionMeasurementInvalid
ScaleInvariance
ControlEquivalence
LongHorizonRecoverySynthetic
```

通过标准：

```text
LineC exception => MeasurementInvalid，不进入 pass/fail mean。
LineC-fast 与 LineC-channel 在 synthetic known-transfer 上方向一致。
LineC-channel 不允许读取 validation/test/future。
```

### Test 0.3.2：Retention / debt 公式

必须验证：

```text
h800 positive but h1600 negative => retention_h1600/h800 = 0。
h1600 negative but h3200 positive => late_rebound，不是 retained。
debt peak = 0 时 recovery undefined，不得写成 1。
missing debt metric => EvidenceIncomplete，不得写成 pass。
```

### Test 0.3.3：Mechanism semantic contract

每个 mechanism 必须声明：

```text
source_space = parameter / function / output / block / basis-channel；
uses_optimizer_primary = AdamW / SGD / Momentum / none / schedule_free / role_partition；
uses_slow_state = 0/1；
uses_matrix_block = 0/1；
uses_poprisk_snr = 0/1；
uses_function_space_operator = 0/1；
implementation_is_prototype = 0/1。
```

如果 `implementation_is_prototype=1`，该 mechanism 只能 smoke，不能 no-go 该 theoretical family。

### Test 0.3.4：Efficiency profiler correctness

必须将 update phase 拆成：

```text
SGD_update_ms
AdamW_update_ms
manual_FU_commit_ms
basis_specific_update_ms
functional_direction_ms
functional_projection_ms
functional_commit_ms
```

如果仍只用 SGD `.step()` 代表所有 update，efficiency truth table 不合格。

---

# 7. Line A：Functional Update Breakthrough Matrix

Line A 的目标不是再跑更多名字，而是围绕下面四个核心假设做 decisive tests。

## 7.1 Hypothesis FU-H1：AdamW 洗掉 FU source

### 目的

验证 FU 与 AdamW 绑定是否是错误前提。

### 实验设计

对 `MLP`、`D-CHE`、`D-FOU-smoke`、`LQ-smoke` 同时跑：

```text
H1-A AdamW-primary + FU residual
H1-B SGD-primary + FU residual
H1-C Momentum-primary + FU residual
H1-D FU-primary, no AdamW optimizer state
H1-E FU-only smoke, no ordinary optimizer step
H1-F Alternating: k gradient steps + 1 FU pulse
H1-G Schedule-free / averaged-state + FU
H1-H Role partition:
     basis params = FU / function-space;
     readout/mixing params = SGD/Momentum or matrix optimizer;
     scale/decay params = decoupled shrink only.
```

### 记录指标

```text
source_h100/h400/h800/h1600/h3200
source_retention_h800/h100
source_retention_h1600/h800
source_retention_h3200/h1600
AdamW_overwrite_cos_h800/h1600
optimizer_cumulative_projection_on_FU
FU_vs_grad_cos
FU_vs_optimizer_step_cos
tail_debt_peak/final/recovery
LineC_fast_debt_peak/final/recovery
LineC_channel_debt_peak/final/recovery
ECE_debt_peak/final/recovery
AUCtime_ratio
control_equivalent_fraction
same_mechanism_MLP_delta
```

### 判断标准

FU-H1 成立，如果：

```text
AdamW-primary 下 source h800 positive 但 h1600/h3200 washout；
SGD/Momentum/FU-primary/slow-state 下 source retention >= 0.50；
AdamW_overwrite_projection_h1600 <= -0.30；
matched controls fail；
recovery-only fail。
```

若 AdamW-free 也没有 retention，则不能说 AdamW 是 blocker；转入 FU-H2 / FU-H3。

### 不满足条件时 Codex 先尝试

```text
如果 FU-primary 数值不稳定：
  自动降低 FU step scale 到 0.5x, 0.25x；
  但必须记录 scale，不允许 dataset/seed branch。

如果 FU-only 训练发散：
  改成 alternating FU/gradient；
  加 decoupled norm clamp；
  不能转回 AdamW-primary 作为唯一路线。

如果 AdamW overwrite diagnostic 缺失：
  不允许写 H1 no-go；先补 diagnostic。
```

---

## 7.2 Hypothesis FU-H2：source 需要 slow-state retention

### 目的

解决 h800 source 有、h1600/h3200 washout 的问题。

### 实验设计

对 MLP/M2 与 D-CHE/M2 两条线做 slow-state source writer：

```text
H2-A single EMA source state
H2-B dual EMA source state short+long
H2-C schedule-free averaged iterate FU
H2-D lookahead source consolidation
H2-E source state only updates when split-consensus and PopRisk agree
H2-F source state anti-alignment diagnostic, 不 hard mask
```

定义：

$$
u_t = P_{signal}(u_t)
$$

$$
s^{short}_{t+1}=\beta_s s^{short}_t+(1-\beta_s)\nu_t
$$

$$
s^{long}_{t+1}=\beta_l s^{long}_t+(1-\beta_l)\nu_t
$$

提交 update：

$$
\Delta\theta_t = \eta_f u_t + \eta_s P_{carrier}(s^{long}_t)
$$

或只提交 slow state：

$$
\Delta\theta_t = \eta_s P_{carrier}(s^{long}_t)
$$

### 记录指标

```text
short_state_norm
long_state_norm
short_long_cosine
source_state_projection_to_current_gradient
source_state_projection_to_signal_channel
state_age_effective_steps
source_retention_h1600/h800
source_retention_h3200/h1600
washout_rate
late_rebound_rate
debt recovery metrics
```

### 判断标准

FU-H2 成立，如果：

```text
single-step FU source washout；
slow-state FU source_retention_h1600/h800 >= 0.50；
source_retention_h3200/h1600 >= 0.50；
tail or LineC-channel recovery >= 0.60；
controls fail。
```

### 不满足条件时 Codex 先尝试

```text
如果 short/long cosine 长期 < 0：
  改为 disagreement-no-commit protocol；
  继续记录 source observability，不 hard stop。

如果 slow state source 变成 control-equivalent：
  增加 random-same-age-state control；
  增加 shuffled-split source state control。

如果 source state norm 爆：
  加 role-wise norm cap；
  不允许按 dataset/seed 调 cap。
```

---

## 7.3 Hypothesis FU-H3：source 应该定义在 matrix/block space，不是逐参数 space

### 目的

验证逐参数 FU 是否把 matrix/block structure 打散。

### 实验设计

必须实现真正 matrix-block FU，而不是 row mask：

```text
H3-A MLP hidden matrix block FU
H3-B D-CHE degree-readout block FU
H3-C D-FOU band-readout block FU
H3-D LQ projection-frame block FU
H3-E Rational numerator/denominator group block FU
```

每个 block 计算 gradient matrix $G$，再比较：

```text
raw gradient block；
orthogonalized block update；
SOAP-like rotated diagonal preconditioned block；
low-rank r=4/r=8 block source state；
random same-rank matrix control；
row/column shuffled block control。
```

矩阵 update 可写作：

$$
\Delta W = -\eta U V^T
$$

其中 $U,V$ 来自 current train-stream block gradient 的 SVD / sketch，但不能使用 validation/test/future。

### 记录指标

```text
block_update_rank
block_update_spectral_norm
block_update_fro_norm
block_source_retention
block_random_control_gap
row_col_shuffle_gap
orthogonalization_time_ms
block_precondition_time_ms
source_horizon metrics
debt recovery metrics
```

### 判断标准

FU-H3 成立，如果：

```text
matrix/block FU h1600 source mean >= 0.005；
source_retention_h1600/h800 >= 0.50；
random same-rank and shuffled controls fail；
AUCtime_ratio_h1600 <= 1.05；
step overhead <= 1.25 or efficiency-smoke route only。
```

### 不满足条件时 Codex 先尝试

```text
如果 orthogonalization overhead 高：
  降到 low-rank r=4 sketch；
  或降低 precondition frequency，每 20/50 steps 更新一次 basis。

如果 block FU source 只在 MLP 出现：
  标记 generic matrix-dynamics insight；
  转入 KAN carrier block redesign。

如果 KAN block source 被 MLP controls 解释：
  不能写 KAN-specific；
  继续 basis carrier repair，而不是扩 FU token。
```

---

## 7.4 Hypothesis FU-H4：functional update 应该是 function-space operator，不是 parameter perturbation

### 目的

重启 v13.4 operator-level basis-channel 思路，但修正 previous code/proxy 问题。

### 实验设计

用 train split $B_1,B_2,B_3$ 构造合法 operator-level update：

```text
B1: estimate source cotangent / output displacement proposal。
B2: check train-split transfer。
B3: check no immediate collapse。
```

令：

$$
\Delta f_{B} = J_B U\alpha
$$

解：

$$
\alpha^* = \arg\min_\alpha
L_{B1}(f+J_{B1}U\alpha)
+
\lambda L_{B2}(f+J_{B2}U\alpha)
+
\rho\|\alpha\|^2
$$

但提交前必须测 actuation fidelity：

$$
ActuationR2 = R^2(\Delta f_{target}, \Delta f_{actual})
$$

### 记录指标

```text
B1_gain
B2_transfer_gain
B3_safety_gain
actuation_R2
actuation_cosine
parameter_norm
function_displacement_norm
projection_residual_norm
source_retention_h800/h1600/h3200
debt recovery
matched operator random control
```

### 判断标准

FU-H4 成立，如果：

```text
B2_transfer_gain > 0；
actuation_R2 >= 0.50；
h1600 source mean >= 0.005；
source_retention_h1600/h800 >= 0.50；
operator random controls fail。
```

### 不满足条件时 Codex 先尝试

```text
如果 actuation_R2 < 0.20：
  不跑 long horizon；先修 basis-channel U 或 projection solver。

如果 B1 gain 有但 B2 transfer 无：
  加 PopRisk/SNR source filter；
  不能把 B1 local positive 写成 progress。

如果 operator solve 太慢：
  限制 U 到低秩 r=4/r=8；
  只跑 top carriers。
```

---

# 8. Line B：Basis Kernel Efficiency Breakthrough

Line B 的目标是让至少一个 non-MLP basis 进入 MLP-like efficiency envelope，并明确每个 basis 的状态：

```text
FamilyPass
FamilyNearPass
KernelForwardBlocked
KernelBackwardBlocked
UpdateBlocked
AuditCostPolluted
RejectedForThisVersion
```

## 8.1 通用 profiler 与 gate

每个 basis 必须记录：

```text
forward_only_ms
basis_eval_ms
readout_contraction_ms
backward_input_ms
backward_param_ms
optimizer_update_ms_AdamW
optimizer_update_ms_SGD
optimizer_update_ms_manualFU
functional_direction_ms
functional_commit_ms
LineC_fast_audit_ms
LineC_channel_audit_ms
horizon_readback_ms
forward_peak_memory
backward_peak_memory
basis_activation_bytes
materialized_basis_bytes
workspace_temp_bytes
kernel_count
custom_kernel_count
```

探索效率门：

$$
T_{forward}/T_{MLP} \le 1.75
$$

$$
T_{step}/T_{MLP} \le 1.50
$$

$$
M_{backward}/M_{MLP} \le 1.10
$$

官方效率门：

$$
T_{forward}/T_{MLP} \le 1.25
$$

$$
T_{step}/T_{MLP} \le 1.25
$$

$$
M_{backward}/M_{MLP} \le 1.00
$$

## 8.2 D-FOU：第一优先级 forward kernel repair

### 假设

D-FOU 不是 fundamentally slow；它主要慢在 sin/cos / band eval 未 fused。

### 实验

```text
FOU-R0 current torch reference
FOU-R1 sincos recurrence no-materialize
FOU-R2 precomputed frequency table
FOU-R3 fused band eval + readout contraction
FOU-R4 analytic backward for band-readout
FOU-R5 low-frequency active bank with high-band quarantine
```

### 记录指标

```text
sincos_eval_ms
band_eval_ms
band_readout_contract_ms
materialized_band_bytes
high_band_energy_fraction
bandwise_snr
phase_drift
forward_ratio
step_ratio
memory_ratio
```

### 判断

D-FOU repair 成立，如果：

```text
forward ratio <= 1.75；
step ratio <= 1.50；
memory ratio <= 1.10；
functional smoke M1/M3/M4 至少有 source_h800 > 0 on 3/9 或证明无 signal。
```

### 不满足时 Codex 先尝试

```text
如果 R1 recurrence 数值误差大：
  降低 frequency band；
  fp32-only；
  对比 table lookup。

如果 R3 fused contraction 太慢：
  合并 band dimension 到 GEMM；
  或只保留 low-frequency active bank。

如果 high_band_energy_fraction 低：
  自动尝试 high-band quarantine，不作为 dataset tuning。
```

## 8.3 D-CHE：第二优先级 no-materialize recurrence

### 假设

D-CHE 是当前最强 KAN carrier，但 forward 慢来自 dense degree tensor materialization。

### 实验

```text
CHE-R0 current torch reference
CHE-R1 Clenshaw / recurrence eval
CHE-R2 no-materialize degree-readout contraction
CHE-R3 analytic backward
CHE-R4 low-degree active bank
CHE-R5 audit/horizon readback separation
```

### 记录指标

```text
degree_eval_ms
degree_tensor_bytes
degree_readout_contract_ms
backward_degree_ms
high_degree_energy_fraction
degree_entropy
forward_ratio
step_ratio
memory_ratio
```

### 判断

D-CHE repair 成立，如果：

```text
forward ratio <= 1.75；
step ratio <= 1.50；
memory ratio <= 1.10；
D-CHE/M2 late rebound rerun gives reproducible positive or route closed as stochastic。
```

### 不满足时 Codex 先尝试

```text
如果 no-materialize backward 错：
  先 keep forward fused + autograd backward for smoke；
  同时落 gradcheck fail，不写 official。

如果 degree_eval 仍 >3x：
  限制 low-degree active bank K<=4/8；
  不做 dataset-specific degree schedule。
```

## 8.4 LQ / Rational / D-RBF / D-WAV

这些为 secondary repair。

```text
LQ:
  recurrence + fixed-frame cache + fused projection update。

D-RAT:
  Horner numerator/denominator + branchless denominator + telemetry separation。

D-RBF:
  compact local K4/K8 + active-center occupancy + no dense materialization。

D-WAV:
  triangular support index-only + sparse support backward。
```

它们必须跑 correctness + efficiency smoke，但不抢 D-FOU/D-CHE 4GPU 主预算。

---

# 9. 4GPU 动态执行计划

v19.0 使用四张 GPU，必须输出动态队列证据。

## 9.1 GPU 分工

```text
GPU0:
  Line 0 S0.3 tests；
  D-CHE FU-H1/H2/H4；
  D-CHE kernel repair R1-R5。

GPU1:
  MLP FU-H1/H2/H3；
  MLP M2 washout deep dive；
  slow-state / schedule-free / dual memory experiments。

GPU2:
  D-FOU kernel repair R1-R5；
  D-FOU FU smoke M1/M3/M4；
  efficiency waterfall figures。

GPU3:
  LQ / D-RAT / D-RBF / D-WAV repair smoke；
  D-CHE M2 late rebound independent rerun；
  controls / figures / packet generation when queue available。
```

## 9.2 动态队列规则

必须生成：

```text
v19_runnable_queue.csv
v19_gpu_assignment_manifest.csv
v19_gpu_utilization_dashboard.csv
v19_idle_violation.csv
v19_deferred_items.csv
v19_queue_drain_report.csv
```

硬规则：

```text
如果 runnable_queue 非空，任一 GPU idle > 10 min：
  execution_contract_violation = 1；
  final route 不能写 completed no-go。
```

每个 job 必须有：

```text
job_id
line
carrier
mechanism
priority
estimated_runtime
assigned_gpu
start_time
end_time
status
artifact_path
if_failed_next_attempt
```

---

# 10. 必须记录的指标

## 10.1 Code / audit 指标

```text
compileall_ok
import_error_count
missing_required_files
LineC_fast_golden_pass
LineC_channel_golden_pass
retention_formula_pass
debt_formula_pass
route_aggregation_pass
update_sign_pass
mechanism_noncollapse_pass
efficiency_profiler_phase_pass
kernel_gradcheck_pass
packet_manifest_complete
```

## 10.2 Functional 指标

```text
carrier
mechanism
optimizer_primary
uses_adamw
uses_sgd
uses_momentum
uses_slow_state
uses_matrix_block
uses_poprisk_snr
uses_function_space_operator
implementation_is_prototype
source_h100
source_h400
source_h800
source_h1600
source_h2400
source_h3200
source_h4800
source_retention_h800_h100
source_retention_h1600_h800
source_retention_h3200_h1600
late_rebound_flag
late_rebound_repro_count
tail_debt_peak/final/recovery
LineC_fast_debt_peak/final/recovery
LineC_channel_debt_peak/final/recovery
ECE_debt_peak/final/recovery
Brier_debt_peak/final/recovery
AUCtime_ratio
control_equivalent_fraction
matched_random_gap
same_rank_gap
same_norm_gap
same_overhead_gap
same_mechanism_MLP_delta
AdamW_overwrite_cos
AdamW_overwrite_projection
```

## 10.3 Efficiency 指标

```text
forward_only_ms
basis_eval_ms
readout_contraction_ms
backward_input_ms
backward_param_ms
optimizer_update_ms_SGD
optimizer_update_ms_AdamW
optimizer_update_ms_manualFU
functional_direction_ms
functional_commit_ms
LineC_fast_audit_ms
LineC_channel_audit_ms
horizon_readback_ms
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
basis_activation_bytes
materialized_basis_bytes
workspace_temp_bytes
kernel_count
custom_kernel_count
official_fused_kernel_complete
```

---

# 11. 必须可视化的图

```text
1. source trajectory by carrier/mechanism: h100/h400/h800/h1600/h3200/h4800。
2. source retention heatmap: carrier x mechanism。
3. late rebound map: h1600 negative -> h3200/h4800 positive cases。
4. debt recovery curves: tail / LineC-fast / LineC-channel / ECE / Brier / AUCtime。
5. AdamW overwrite projection curves。
6. MLP vs D-CHE same mechanism source trajectory。
7. PopRisk/SNR signal-state norm and short/long agreement。
8. matrix-block update rank vs source retention。
9. efficiency waterfall per family。
10. dense materialized vs no-materialize timing bar。
11. forward ratio vs step ratio vs memory bubble plot。
12. 4GPU utilization timeline。
13. route aggregation dashboard: single-row max vs 9-row mean vs pass count。
```

---

# 12. 成功标准

## S0.3：代码/指标/机制真值门

必须全部满足：

```text
compile/import closure pass；
LineC-fast + LineC-channel golden pass；
retention/debt formula pass；
route aggregation pass；
update semantics pass；
mechanism semantic contract pass；
efficiency profiler phase pass；
kernel gradcheck pass；
required packet complete。
```

否则不得写 scientific no-go。

## S2：weak productive functional signal

一个 carrier × mechanism group 必须满足：

```text
9-row mean source_h800 >= 0.005；
dataset_seed_pass_count_h800 >= 3/9；
source_retention_h800/h100 >= 0.40；
tail or LineC-channel recovery >= 0.40；
AUCtime_ratio_h800 <= 1.10；
matched controls fail。
```

## S3：retained productive dynamics

必须满足：

```text
9-row mean source_h1600 >= 0.005；
dataset_seed_pass_count_h1600 >= 4/9；
source_retention_h1600/h800 >= 0.50；
tail recovery >= 0.60；
LineC-channel recovery >= 0.60；
AUCtime_ratio_h1600 <= 1.05；
random/same-rank/same-overhead/recovery-only controls fail。
```

## S3b：delayed signal migration

如果 h1600 没过，但 h3200/h4800 late rebound 出现，必须满足：

```text
independent rerun count >= 3；
positive rebound reproduced >= 2/3；
h3200 or h4800 mean source >= 0.005；
matched controls fail；
late rebound not caused by route single-row max；
debt recovery completed。
```

否则 late rebound 只能 diagnostic。

## E2：basis efficiency exploration

一个 non-MLP basis family 必须满足：

```text
forward_ratio <= 1.75；
step_ratio <= 1.50；
memory_ratio <= 1.10；
correctness pass；
audit cost separated。
```

## E3：basis efficiency official candidate

必须满足：

```text
forward_ratio <= 1.25；
step_ratio <= 1.25；
memory_ratio <= 1.00；
official fused or no-materialize kernel complete；
training path not audit-polluted。
```

## S5：official success，不降低

最终仍然严格：

```text
real_dataset_seed_pass_count = 9/9
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC-channel pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.25
controls fail
code/provenance/no-action audit pass
promotion_allowed = 1
```

---

# 13. 不满足条件时的 fallback 方向

## Case A：S0.3 不过

```text
不要跑科学矩阵。
先修对应代码：
  LineC invalid -> 修 LineC-channel / MeasurementInvalid；
  debt missing -> 补 debt readback；
  mechanism prototype -> 降级 smoke 或补真实实现；
  profiler invalid -> 修 phase profiler。
```

## Case B：MLP h800 source 仍 washout

```text
优先尝试 slow-state / dual-memory / schedule-free / PopRisk source writer。
如果仍 washout：
  判断当前 observable 不足，转向 signal-channel estimator；
  不要继续在 KAN 上堆 carrier。
```

## Case C：MLP source retained，KAN source 不 retained

```text
functional dynamics 可行，但 KAN carrier 不承载。
优先修 D-CHE / D-FOU carrier block 和 basis kernel。
```

## Case D：D-CHE late rebound 不复现

```text
关闭 delayed signal migration route；
不要围绕 h3200 单次 positive 小修。
```

## Case E：D-CHE late rebound 复现

```text
转入 delayed migration route：
  扩 h2400/h3200/h4800；
  加 source-channel audit；
  加 controls；
  只在复现后进入 S3b。
```

## Case F：D-FOU / D-CHE kernel repair forward 仍 > 3x

```text
停止同实现小修。
D-FOU：改 table lookup / low-frequency active bank。
D-CHE：改 low-degree active bank / Clenshaw eval。
仍不行则标 KernelForwardBlocked。
```

## Case G：mechanism positive 被 controls 解释

```text
不能 promotion。
如果 MLP 同机制也 positive：写 generic insight。
如果 random same-rank controls positive：说明 source estimator 无独立性。
```

---

# 14. v19.0 最终必须输出的结论类型

v19.0 结束时必须至少能给出这些明确结论之一：

```text
1. AdamW blocks FU source / AdamW does not block FU source。
2. MLP has retained functional source / MLP source washes out。
3. D-CHE late rebound reproducible / stochastic。
4. PopRisk/SNR slow-state predicts source retention / does not。
5. Matrix-block FU improves retention / does not。
6. D-FOU kernel repair reaches forward <=1.75 / remains forward blocked。
7. D-CHE kernel repair reaches forward <=1.75 / remains forward blocked。
8. LineC-channel/debt metrics complete / still incomplete。
```

如果 v19.0 仍只能输出：

```text
route=no-go
source partial
forward blocked
```

则执行不合格，因为它没有推进问题边界。

---

# 15. 最终总结

v19.0 的核心不是“再大胆试几个方法”，而是改变实验哲学：

$$
\boxed{
\text{functional update 不再是 AdamW 的小补丁，}
\text{而是 source-channel writer + carrier actuator + trajectory retention process。}
}
$$

同时：

$$
\boxed{
\text{basis efficiency 不再是 census，}
\text{而是 family-specific kernel repair 与 no-materialize implementation。}
}
$$

本轮必须真正推进两件事：

```text
Functional：
  解释 source 为什么不留存，并验证 AdamW-free / slow-state / matrix-block / function-space 路线。

Efficiency：
  至少让 D-FOU 或 D-CHE 的 forward ratio 明确下降，或给出不可行证据。
```

如果这两件事都没有发生，就不是科学失败，而是实验设计失败。
