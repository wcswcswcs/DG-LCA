# DG-KAN v21.01：Source-Retention First Functional Update + D-CHE/D-FOU Kernel Officialization + 4GPU 动态并行完整计划

> 版本：v21.01  
> 目标：不再继续“局部 source / late rebound / weak h1600”的循环。v21.01 必须同时推进两条硬主线：  
> 1. **Functional update**：从“找一个更强的局部 update”转向“让 source 写入长期可保留的 signal channel”。  
> 2. **KAN 基函数效率**：从“efficiency census / official-like rows”转向“D-CHE / D-FOU official kernel full-loop closure”，同时对 LQ / Rational / RBF / Wavelet 做有边界的 targeted repair。  
>
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no label-informed init；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / Brier / AUCtime 只能作为 audit / gate / debt readback，不能生成方向。

---

# 0. 项目总目标与 v21 后的真实状态

DG-KAN 的总目标仍然是：

$$
\boxed{
\text{在 strict FC-PureKAN base 上，通过 functional update 获得比普通 AdamW/backprop 更好的长期训练动力学，}
}
$$

同时必须满足：

$$
\boxed{
\text{KAN basis 的 forward / backward / update / memory 与 same-param MLP 可比，不能靠慢很多换局部 source。}
}
$$

这里的“更好”不是某个 row 的 source 大，也不是 h800 一时 positive，而是：

```text
1. source 能从 h800/h1600 留到 h3200/h4800，必要时延伸到 h6400；
2. tail / LineC / calibration / AUC debt 能被偿还；
3. matched controls / random target / recovery-only / same-actuation random 不能解释；
4. 如果 MLP 也成功，必须写成 generic training-dynamics insight，不是 KAN-specific；
5. 如果 KAN 成功，必须同时证明 same-mechanism KAN > MLP，并且基函数效率过关；
6. D-CHE / D-FOU 的高效 kernel 必须进入真实 full functional loop，而不是只在 profiler path 中 pass。
```

v21 的真实状态不是“完全失败”，也不是“快成功”。它给了两个清楚结论：

```text
效率线：
  D-CHE / D-FOU 已经有 strong efficiency rows，best forward / step 已进入 same-param MLP envelope。
  但 code packet 缺核心 fused kernel 依赖，kernel_gradcheck 与 efficiency truth table 的 official_fused 状态不一致。
  所以 efficiency 是真实进展，但还不是 official closure。

Functional update 线：
  MLP 有 weak h1600 source，但 h3200 washout。
  KAN 没有 weak h1600 / productive h3200 / h4800 retained source。
  Function-space target 可局部 actuation，高 ActuationR2 不等于 retained source。
  PopRisk/SNR 当前不能作为 direction source。
```

因此 v21.01 的核心不是“再调 M43 / KSW2 / actuation rank”，而是：

$$
\boxed{
\text{把 source-retention 问题本身当成第一研究对象。}
}
$$

---

# 1. v21.01 必须解决的根问题

## 1.1 代码 / 指标根问题

v21 复盘里写了 CodeRoute pass，但 code packet 不是自包含的。缺少核心依赖时，任何“代码正确”都只能算 runner 自检，不能算独立审计。

v21.01 必须解决：

```text
Q-Code-1:
  代码包是否自包含？是否能在干净环境 import core v21.01 runners？

Q-Code-2:
  LineC-fast / LineC-channel / debt / retention / route aggregation 是否和计划语义一致？

Q-Code-3:
  mechanism 名称是否真实反映实现？
  如果只是 gradient mask，就不能叫 matrix-block FU；
  如果只是 scaled gradient，就不能叫 function-space operator。

Q-Code-4:
  efficiency truth table 的 official_fused_kernel_complete 是否与 kernel_gradcheck / source code 一致？
```

如果这些不过，v21.01 不能写 scientific no-go。

---

## 1.2 基函数效率根问题

v21 已经证明 D-CHE / D-FOU 不是天然慢。现在要解决的是：

```text
Q-Eff-1:
  D-CHE / D-FOU 的 profiler pass 是否能在 full functional training loop 中保持？

Q-Eff-2:
  official_fused / no-materialize kernel 是否真的存在于源码，而不是 result table 标记？

Q-Eff-3:
  LineC/horizon readback cost 是否和 training cost 分离？

Q-Eff-4:
  LQ / Rational / RBF / Wavelet 是否还有必要继续作为主线，还是只保留 targeted smoke？
```

v21.01 的效率目标不是再做 census，而是把 D-CHE / D-FOU 推到：

$$
\boxed{
\text{full-loop official kernel evidence}
}
$$

---

## 1.3 Functional update 根问题

v21 的功能更新没有卡在“完全没有 source”，而是卡在：

```text
Q-FU-1:
  h800/h1600 source 为什么会在 h3200/h4800 washout？

Q-FU-2:
  MLP weak source 为什么比 KAN 更容易出现？
  它是 generic matrix dynamics，还是当前 KAN carrier 写入方式不对？

Q-FU-3:
  D-CHE / D-FOU function-space target 可以局部 actuation，但为什么不能 retention？

Q-FU-4:
  late positive / delayed rebound 是否是随机 trajectory effect，还是可以被设计成 retained signal migration？

Q-FU-5:
  AdamW 是否长期洗掉 FU source？如果不是第一步反向 overwrite，那是不是后续 optimizer dynamics 造成 washout？

Q-FU-6:
  KAN source-channel 是否应该由 basis 估计 source、readout/low-degree/low-frequency bank 提交，而不是直接改全 basis 参数？
```

v21.01 的 functional 目标是从 source amplitude 转向 source dynamics：

$$
\boxed{
\text{不是追求 h800 source 最大，而是追求 h800 -> h1600 -> h3200 -> h4800 source 连续留存。}
}
$$

---

# 2. 哪些旧方向可以重开，哪些必须关闭

## 2.1 可以重开，但必须换语义的方向

### 2.1.1 PopRisk / SNR

过去 PopRisk / SNR 出现过 one-shot 或短期版本失败，也出现过 cover 转换失败。但这不等于 population-risk / SNR 思想已经被证伪。当前 generalization theory 的启发是：coherent population signal 应该在 signal channel 里形成 drift，而 idiosyncratic noise 应该被压进 reservoir / diffusion。v21 中 PopRisk/SNR 没形成合法 precommit selector，说明当前 estimator 不对，而不是这个方向没有价值。

v21.01 允许重开 PopRisk/SNR，但必须改成：

```text
1. 不再作为 one-shot update；
2. 不再直接按 parameter SNR mask；
3. 改为 source-retention estimator；
4. 必须预测 h3200/h4800 retention，而不是 h100/h800 local gain；
5. 不能用 early-source readback 当 direction selector。
```

### 2.1.2 Split-consensus / G7R lineage

split-consensus 曾出现过 source signal，但伴随 bad debt / instability；后来静态 hazard-null / projection 会杀 source。现在应重开为：

```text
source estimator；
not direct parameter update；
not trust scalar；
not G-token extension。
```

### 2.1.3 Function-space actuation

v20/v21 都显示 high ActuationR2 不等于 source success。这不是 function-space route 死了，而是 target 错了。v21.01 重开 function-space，但研究对象从：

```text
能不能 actuation？
```

改成：

```text
什么 output target 能 retained？
```

### 2.1.4 MLP functional update

MLP 不是 control，而是 active mechanism discovery line。v19 有 MLP retained source，v20 打回，v21 有 weak h1600 source但 h3200 washout。它仍是判断 FU 是否 generic 可行的主实验台。

### 2.1.5 Graph-free / analytic adjoint / kernel repair

早期 graph-free 失败多卡在 primitive/cache/kernel 效率，不等于 analytic-adjoint 方向没价值。现在 D-CHE / D-FOU kernel efficiency 已经打开，v21.01 要把 graph-free / no-materialize / fused path 和 functional loop 绑定起来。

---

## 2.2 不应重启的方向

以下方向已经多轮显示上界不足、control-equivalent、或指标目标错误，v21.01 不允许继续消耗主预算：

```text
1. action bank / controller / online action search；
2. reset route / optimizer-state reset route；
3. cover objective 小修；
4. M31/M32 同家族修补；
5. G-token / N-token / Q-token 扩展；
6. dataset-name branch / seed-specific scale；
7. 用 LineC / CEp99 / NLL / ECE / Brier / AUCtime 生成 update direction；
8. 单纯调 cycle period / FU lr / target threshold / alt period；
9. single-row max promotion；
10. high ActuationR2 promotion。
```

---

# 3. 当前研究进展对 v21.01 的启发

v21.01 不是机械替换 optimizer，而是抽取训练动力学原则。

## 3.1 Signal channel / reservoir 理论

我们采用如下解释框架：

```text
coherent source:
  应进入 signal channel，并在 h3200/h4800 留存。

noise / local memorization:
  应留在 reservoir 或被 diffusion 抑制。

failure:
  source 只在 h800/h1600 出现，随后 washout；
  或 high actuation target 不进入 signal channel。
```

因此 v21.01 要记录：

```text
source_channel_projection
reservoir_projection
noise_signal_leak
train_split_transfer
source_transition_derivative
```

而不是只看单个 source 数值。

## 3.2 Boundary-conditioned training / fixed-point view

短期 bad update 不一定失败，但它必须被训练动力学吸收并进入稳定 fixed-point region。v21.01 不把 immediate debt 当 hard fail，但要求：

```text
h800/h1600 debt 开始下降；
h3200/h4800 debt 充分恢复；
source 同时留存；
matched controls 不能解释。
```

## 3.3 Schedule-free / slow iterate / dual memory

Schedule-free 与 AdEMAMix 类研究的启发是：source 可能需要 fast / slow state 分离。v21.01 不直接照搬 optimizer，而测试：

```text
fast update 产生 source；
slow source state 保留 source；
commit 由 source retention gate 控制；
AdamW / SGD / schedule-free 作为条件比较。
```

## 3.4 Matrix / block optimizer

Muon / SOAP / Newton-Muon 类研究提示：对矩阵/block 参数，逐参数 AdamW 可能不是合适坐标。KAN 的 degree-readout、band-readout、readout projection、basis bank 都是 block / matrix 结构。v21.01 要把 matrix-block FU 变成真实实现，不再是“前半 rows mask”。

---

# 4. v21.01 实验总结构

v21.01 分成三大部分，每一部分都有硬输出。

```text
Part A:
  Code / metric / mechanism truth gate。
  目标：确保结果可信，不再让错误指标/缺源码制造假 no-go。

Part B:
  Basis kernel officialization。
  目标：D-CHE / D-FOU 的 efficiency pass 进入 full functional training loop。

Part C:
  Source-retention functional update。
  目标：回答 source 为什么 washout，构造能 h3200/h4800 retained 的 target / writer。
```

所有实验都走 4GPU dynamic queue，不允许静态分配后空卡。

---

# 5. Part A：S0.6 Code / Metric / Mechanism Truth Gate

## 5.1 Codex 必须打包什么

Codex 必须在实验开始和结束各生成一次：

```text
v21_01_code_review_packet.zip
v21_01_results_bundle.zip
```

`v21_01_code_review_packet.zip` 必须包含：

```text
00_README.md
packet_manifest.csv
packet_sha256_manifest.csv

02_SOURCE_TREE/
  dgkan/**/*.py
  experiments/**/*.py
  tests/**/*.py

03_IMPORT_CLOSURE/
  import_closure.csv
  import_error_details.csv
  compileall.log
  self_contained_import_check.log

04_LINEC_CORRECTNESS/
  linec_fast_source.py
  linec_channel_source.py
  linec_golden_tests.py
  linec_golden_results.csv
  linec_measurement_invalid_tests.csv

05_RETENTION_AND_DEBT/
  retention_formula_tests.py
  retention_formula_results.csv
  debt_accounting_tests.py
  debt_accounting_results.csv
  route_aggregation_tests.py
  route_aggregation_results.csv

06_UPDATE_SEMANTICS/
  update_tensor_source.py
  update_sign_finite_difference_tests.py
  update_sign_results.csv
  update_space_kind_contract.csv
  optimizer_coupling_contract.csv

07_FUNCTIONAL_MECHANISMS/
  mechanisms_source.py
  mechanism_semantic_contract.csv
  mechanism_noncollapse_tests.py
  mechanism_noncollapse_results.csv
  source_writer_source.py
  function_space_target_source.py

08_EFFICIENCY_KERNELS/
  che_official_source.py
  fou_official_source.py
  lq_kernel_source.py
  rat_kernel_source.py
  rbf_kernel_source.py
  wav_kernel_source.py
  kernel_gradcheck_tests.py
  kernel_gradcheck_results.csv
  official_fused_status_matrix.csv

09_PROFILER_CORRECTNESS/
  profiler_source.py
  profiler_phase_tests.py
  profiler_phase_results.csv
  profiler_vs_full_loop_tests.csv
  profiler_vs_full_loop_results.csv

10_EXPERIMENT_RUNNERS/
  all v21.01 runners
  command_journal.sh
  command_journal.csv
  exact_env.yml
  git_diff.patch
```

`v21_01_results_bundle.zip` 必须包含：

```text
official_v21_01/
  route_decision.json
  gate_recompute.json
  code_route.json
  efficiency_route.json
  functional_route.json
  required_manifest.csv
  forbidden_audit.csv
  no_action_search_audit.csv

raw_matrices/
  source_retention_matrix.csv
  functional_raw_horizon_matrix.csv
  debt_accounting_matrix.csv
  target_contrast_matrix.csv
  source_dynamics_matrix.csv
  source_transition_matrix.csv
  controls_attribution_matrix.csv
  efficiency_truth_table.csv
  full_loop_efficiency_matrix.csv
  kernel_repair_matrix.csv
  gpu_queue_matrix.csv

figures/
  all required SVG/PNG
```

如果代码包缺少任何被 import 的 `.py` 文件，S0.6 失败，scientific route 禁止写 completed no-go。

---

## 5.2 S0.6 代码正确性测试

Codex 必须运行：

```bash
python -m compileall -q dgkan experiments tests
python tests/v21_01/test_import_closure.py
python tests/v21_01/test_linec_fast_channel.py
python tests/v21_01/test_retention_debt_route.py
python tests/v21_01/test_update_semantics.py
python tests/v21_01/test_functional_mechanism_contracts.py
python tests/v21_01/test_efficiency_profiler.py
python tests/v21_01/test_kernel_gradcheck.py
```

必须记录：

```text
compileall_ok
import_error_count
missing_source_file_count
linec_fast_golden_pass
linec_channel_golden_pass
retention_formula_pass
debt_accounting_pass
route_aggregation_pass
update_sign_pass
mechanism_noncollapse_pass
profiler_phase_pass
kernel_gradcheck_pass
official_fused_status_consistency_pass
```

## 5.3 S0.6 hard gate

S0.6 通过条件：

```text
missing_source_file_count = 0
import_error_count = 0
compileall_ok = 1
linec_fast_golden_pass = 1
linec_channel_golden_pass = 1
retention_formula_pass = 1
debt_accounting_pass = 1
route_aggregation_pass = 1
update_sign_pass = 1
mechanism_noncollapse_pass = 1
profiler_phase_pass = 1
kernel_gradcheck_pass = 1
official_fused_status_consistency_pass = 1
```

若不满足：

```text
Codex 必须先修 code packet / imports / metrics。
不得启动 functional matrix。
不得写 route = scientific no-go。
```

---

# 6. Part B：Basis Kernel Officialization

## 6.1 总目标

v21.01 的 efficiency 线只主攻 D-CHE / D-FOU。其它 basis 保留低预算 targeted smoke。

目标：

$$
\boxed{
\text{证明 D-CHE / D-FOU 的 high-efficiency rows 不只是 profiler artifact，}
\text{而是可以进入 full functional training loop 的 official kernel path。}
}
$$

---

## 6.2 Line E0：official fused status reconciliation

### 假设

v21 的 efficiency table 与 kernel_gradcheck status 不一致。v21.01 必须先统一定义。

### 实验

对 D-CHE / D-FOU 每个 official-like row，记录：

```text
kernel_id
source_file
uses_triton
uses_cuda_cpp
uses_torch_reference
uses_dense_materialization
no_materialize_complete
forward_fused
backward_fused
update_fused
functional_direction_fused
gradcheck_pass
full_loop_uses_kernel
profiler_uses_kernel
```

### 判定

只有同时满足：

```text
source_file_exists = 1
gradcheck_pass = 1
no_materialize_complete = 1
profiler_uses_kernel = 1
full_loop_uses_kernel = 1
```

才可标记：

```text
official_kernel_usable = 1
```

如果不满足，Codex 必须自动尝试：

```text
1. 补齐源码；
2. 修 shim / import；
3. 改 profiler 和 full-loop 使用同一 kernel；
4. 若 fused path 未实现，降级为 exploration，不允许 official-like。
```

---

## 6.3 Line E1：D-CHE full-loop officialization

### 假设

D-CHE CHE21-R2 / CHE21-R4 在 profiler 中已经进入 MLP-like envelope；如果它在 full functional loop 中仍保持效率，就可以把 D-CHE 作为主 KAN carrier 继续推进 functional source writer。

### 实验

候选：

```text
CHE21-R2-low-degree-k3-official
CHE21-R4-k3-gradbuf-triton-official
CHE21-R4-k5-gradbuf-triton-official
```

测试条件：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
batch = 32,128,256
train_steps = 800 for efficiency loop
functional_runner = enabled and disabled
LineC/horizon audit = separated
```

### 记录指标

```text
forward_only_ms
backward_grad_ms
optimizer_update_ms
fu_direction_ms
fu_commit_ms
linec_audit_ms
horizon_readback_ms
step_total_ms

forward_ratio_vs_same_param_mlp
backward_ratio_vs_same_param_mlp
update_ratio_vs_same_param_mlp
step_ratio_vs_same_param_mlp
memory_ratio_vs_same_param_mlp

basis_eval_ms
degree_recurrence_ms
readout_contraction_ms
grad_buffer_ms
kernel_launch_count
dense_materialized_bytes
workspace_bytes
```

### 判断标准

E1 exploration pass：

```text
forward_ratio <= 1.25
backward_ratio <= 1.40
step_ratio <= 1.25
memory_ratio <= 1.10
full_loop_uses_kernel = 1
audit_cost_separated = 1
```

E1 official candidate：

```text
forward_ratio <= 1.15
backward_ratio <= 1.25
step_ratio <= 1.15
memory_ratio <= 1.05
kernel_gradcheck_pass = 1
no_materialize_complete = 1
same-param MLP matched = 1
```

如果不满足，Codex 先尝试：

```text
CHE-FB1:
  检查 profiler path vs full-loop path 是否使用同一 kernel。

CHE-FB2:
  分离 LineC / horizon readback cost。

CHE-FB3:
  替换 dense degree materialization 为 recurrence contraction。

CHE-FB4:
  降低 active degree k，但必须记录 expression/task impact。

CHE-FB5:
  如果 batch32 fail 但 batch128/256 pass，记录 launch-overhead class，不关闭 family。
```

---

## 6.4 Line E2：D-FOU full-loop officialization

### 假设

D-FOU 的 high-efficiency pass 表明 Fourier basis 不是天然慢；它可能成为比 D-CHE 更适合 source-channel 的低频 carrier。

### 实验

候选：

```text
FOU21-R2-lowfreq-k2-official
FOU21-R3-tablelookup-bandreadout-official
FOU21-R4-k4-triton-no-materialize-official
```

测试同 E1。

### 记录指标

额外记录：

```text
sincos_eval_ms
frequency_table_lookup_ms
band_readout_contraction_ms
bandwise_backward_ms
low_frequency_energy
mid_frequency_energy
high_frequency_energy
band_energy_entropy
```

### 判断标准

同 E1。

如果不满足，Codex 先尝试：

```text
FOU-FB1:
  sin/cos recurrence vs table lookup vs torch trig 对比。

FOU-FB2:
  fused band eval + readout contraction。

FOU-FB3:
  bandwise analytic backward。

FOU-FB4:
  low-frequency active bank k2/k4 comparison。

FOU-FB5:
  若 forward pass 但 functional direction slow，拆 functional direction overhead。
```

---

## 6.5 Line E3：LQ / Rational / RBF / Wavelet targeted smoke

这些 family 不进入主 functional proof，只做针对性修复。

```text
LQ:
  recurrence + fixed-frame cache + fused projection update。

D-RAT:
  Horner eval + branchless denominator + telemetry separation。

D-RBF:
  compact local K4 + no dense materialization smoke。

D-WAV:
  sparse support index + sparse backward smoke。
```

目标只是判断：

```text
是否从 ForwardBlocked 进入 NearEfficiency。
```

不允许它们占用 GPU0/GPU1 主预算。

---

# 7. Part C：Functional Update Source-Retention 计划

## 7.1 总体思想

v21 的功能更新失败不是因为完全没有 source，而是：

```text
MLP:
  h1600 weak source -> h3200 washout。

KAN:
  late positive / delayed rebound，但没有连续 retention。

Function-space:
  high ActuationR2 -> no source success。
```

因此 v21.01 functional update 不再追求“更强 h800 source”，而是围绕三类 source dynamics 设计实验：

```text
D1:
  strong-washout source，例如 MLP M2/M43。

D2:
  weak-stable source，例如 M15-like weak positive。

D3:
  late-rebound source，例如 D-CHE / KAN delayed positive。
```

目标是把其中至少一类变成：

```text
retained source:
  h800 positive
  h1600 positive
  h3200 positive
  h4800 positive
  controls fail
  debt recovered
```

---

## 7.2 Line F0：source dynamics decomposition

### 假设

当前 source failure 不是单一失败，而是三种动力学混在一起：

```text
washout:
  h800/h1600 positive, h3200 negative。

weak-stable:
  h800/h1600/h3200 small positive, below threshold。

late-rebound:
  h800 negative, h3200/h4800 positive。
```

如果不分解这三类，就会继续误把 late rebound 当 retention，或误把 h800 source 当 progress。

### 实验

固定对比：

```text
MLP-M2 / MLP-M43:
  strong-washout line。

MLP-M15-like:
  weak-stable line。

D-CHE-KSW2-like / D-CHE-M2-like:
  late-rebound line。

D-CHE/D-FOU loss-target function-space:
  high-actuation-no-retention line。
```

### 记录指标

每个 carrier × mechanism × dataset × seed 记录：

```text
source_h100
source_h400
source_h800
source_h1600
source_h2400
source_h3200
source_h4800
source_h6400

source_derivative_h400_800
source_derivative_h800_1600
source_derivative_h1600_3200
source_derivative_h3200_4800
source_derivative_h4800_6400

retention_h1600_over_h800
retention_h3200_over_h1600
retention_h4800_over_h3200

washout_flag
late_rebound_flag
weak_stable_flag
retained_flag

tail_debt_peak
tail_debt_final
tail_recovery_rate
LineC_debt_peak
LineC_debt_final
LineC_recovery_rate
ECE_debt_peak
ECE_debt_final
Brier_debt_peak
Brier_debt_final
AUCtime_ratio
```

### 判定

F0 不 promotion，只分类。成功是输出可靠 taxonomy：

```text
source_dynamics_classified_fraction >= 0.95
ambiguous_fraction <= 0.05
```

如果 ambiguous 高，Codex 自动：

```text
1. 延长 horizon 到 h6400；
2. 增加 h2400 中间点；
3. 检查 controls 是否同样 late rebound；
4. 检查 route aggregation 是否把 single row 当 mean。
```

### 可视化

```text
source_trajectory_by_class.svg
source_derivative_phase_portrait.svg
debt_recovery_by_class.svg
carrier_mechanism_source_heatmap.svg
late_rebound_vs_retention_scatter.svg
```

---

## 7.3 Line F1：MLP source-retention mechanism dissection

### 假设

MLP 是当前最容易出现 source 的平台。如果 MLP 都不能产生 h3200/h4800 retained source，KAN 上继续堆方法意义很低。

### 实验问题

```text
Q1:
  MLP h1600 source 为什么 h3200 washout？

Q2:
  MLP weak-stable source 是否能通过 slow-state / dual-memory 提升到 threshold？

Q3:
  strong-washout source 和 weak-stable source 是否可组合？
```

### 候选机制

```text
F1-A: StrongSource M2/M43 replay
F1-B: WeakStable M15 replay
F1-C: StrongSource + WeakStable slow anchor
F1-D: Dual-memory source state
F1-E: Schedule-free source iterate
F1-F: AdEMAMix-style short/long source memory
F1-G: Matrix-block source writer on hidden weight
F1-H: SOAP-like rotated source state, low-rank diagnostic
F1-I: Cautious conflict diagnostic, not hard mask
```

### 核心公式

dual-memory source state：

$$
s_t^{short} = \beta_s s_{t-1}^{short} + (1-\beta_s) u_t
$$

$$
s_t^{long} = \beta_l s_{t-1}^{long} + (1-\beta_l) u_t
$$

source commit：

$$
u_t^{commit} =
\lambda_s s_t^{short}
+
\lambda_l s_t^{long}
$$

但提交必须经过 train-stream precommit checks：

```text
split-consensus gain positive
same-actuation random target fail
debt proxy not exploding
no validation/test/future/query
```

### 记录指标

```text
short_state_norm
long_state_norm
short_long_cosine
commit_norm
commit_cosine_with_gradient
commit_cosine_with_adamw
commit_cosine_with_momentum
cumulative_adamw_overwrite_projection
source_channel_projection
hidden_weight_block_projection
readout_projection
h800/h1600/h3200/h4800 source
retention ratios
debt recovery
controls
```

### 判定

F1 weak success：

```text
MLP source_h3200 >= 0.005
retention_h3200_over_h1600 >= 0.50
dataset_seed_pass_h3200 >= 4/9
controls fail
```

F1 strong success：

```text
MLP source_h4800 >= 0.005
retention_h4800_over_h3200 >= 0.50
dataset_seed_pass_h4800 >= 5/9
tail_recovery_rate_h4800 >= 0.60
LineC_recovery_rate_h4800 >= 0.60
controls fail
```

如果失败，Codex 自动尝试：

```text
F1-FB1:
  判断是 source amplitude 不够，还是 washout。
  若 amplitude 不够，尝试 strong+weak combination。
  若 washout，尝试 slow-state / schedule-free / dual memory。

F1-FB2:
  若 random matched source 也成功，标记 generic control-equivalent，停止该 family。

F1-FB3:
  若 only h1600 positive，增加 h2400/h6400，判断是 delayed decay 还是 hard washout。

F1-FB4:
  若 AdamW overwrite projection 强负，转 AdamW-free / SGD / schedule-free。
```

---

## 7.4 Line F2：AdamW-free / optimizer-dynamics decoupling

### 假设

AdamW 不是 functional update 的必要组成。它可能不是第一步反向覆盖 FU，但可能在 h1600->h3200 的长期过程中洗掉 source。

### 实验

对 MLP 和 D-CHE / D-FOU 同机制比较：

```text
O0: AdamW-primary + FU residual
O1: SGD-primary + FU
O2: Momentum-primary + FU
O3: Schedule-free primary + FU
O4: FU-primary + SGD recovery
O5: FU-only smoke
O6: role-partition optimizer
O7: matrix-block optimizer, Muon-like diagnostic
O8: SOAP-like rotated low-rank diagnostic
```

### 记录指标

```text
source trajectory h100..h6400
cumulative_optimizer_projection_on_fu
optimizer_displacement_parallel_to_fu
optimizer_displacement_orthogonal_to_fu
momentum_state_cosine_with_fu
second_moment_weighted_fu_norm
fast_iterate_source
slow_iterate_source
average_iterate_source
weight_decay_parallel_to_fu
weight_decay_orthogonal_to_fu
```

### 判定

AdamW-washout hypothesis supported if：

```text
AdamW-primary h3200 source < 0
and SGD/Momentum/ScheduleFree h3200 source >= 0.005
and controls fail
and debt recovery not worse
```

If no optimizer condition retains source：

```text
AdamW is not primary blocker。
转 F3 target reset / F4 KAN writer。
```

---

## 7.5 Line F3：function-space target reset

### 假设

v21 的 high ActuationR2 说明 actuation 能做到；失败来自 target 定义不对。loss-cotangent target 可短期 transfer，但不 retained。

### Target families

```text
T0: loss-cotangent target
T1: cross-split consensus target
T2: low-frequency / low-degree target
T3: weak-stable source target
T4: readout-only target
T5: reservoir-excluding target
T6: source-channel projection target
TCTRL: random matched target
TCTRL: sign-flip target
TCTRL: corrupt target
TCTRL: same-ActuationR2 random target
```

### 实验

先做 target observability：

```text
B1 gain
B2 transfer gain
B3 safety gain
ActuationR2
random/sign/corrupt contrast
```

再做 long-horizon source test：

```text
h800/h1600/h3200/h4800/h6400 source
debt recovery
controls
```

### 记录指标

```text
target_norm
target_rank
target_low_frequency_fraction
target_low_degree_fraction
target_readout_fraction
ActuationR2
projection_residual_norm
B2_transfer_gain
B3_safety_gain
same_actuation_random_source
target_source_horizon_curve
```

### 判定

Target success：

```text
ActuationR2 >= 0.50
B2_transfer_gain > random by >= 0.05
source_h3200 >= 0.005
retention_h3200_over_h1600 >= 0.50
same-ActuationR2 random target fails
sign-flip / corrupt fail
debt recovered
```

如果 all targets high ActuationR2 but no retained source：

```text
actuation is not blocker；
source observability / target theory invalid；
stop actuation rank tuning；
go to source-channel estimator redesign。
```

---

## 7.6 Line F4：KAN source-channel writer

### 假设

KAN carrier 没有 retained source，是因为 commit 方式错：basis 参数被直接写入会进入 reservoir / rebound，而不是 signal channel。正确方式可能是：

```text
basis channel estimates source；
readout / low-degree / low-frequency bank commits source；
high-degree / high-frequency bank acts as reservoir。
```

### D-CHE candidates

```text
KCHE-A: degree-estimate / readout-commit
KCHE-B: low-degree source bank k3
KCHE-C: low-degree source bank k5
KCHE-D: dual-bank signal/reservoir
KCHE-E: delayed migration writer
KCHE-F: MLP-source projected to D-CHE low-degree bank
```

### D-FOU candidates

```text
KFOU-A: band-estimate / readout-commit
KFOU-B: low-frequency source bank k2
KFOU-C: low-frequency source bank k4
KFOU-D: dual-band signal/reservoir
KFOU-E: delayed migration writer
KFOU-F: MLP-source projected to D-FOU low-frequency bank
```

### 记录指标

```text
basis_estimator_source
readout_commit_norm
low_degree_source_energy
high_degree_reservoir_energy
low_frequency_source_energy
high_frequency_reservoir_energy
source_channel_projection
reservoir_projection
degree_entropy
band_entropy
source_h800/h1600/h3200/h4800
retention
debt recovery
KAN_specific_delta_vs_MLP_same_mechanism
```

### 判定

KAN writer weak success：

```text
KAN source_h3200 >= 0.005
dataset_seed_pass_h3200 >= 3/9
retention_h3200_over_h1600 >= 0.50
same-mechanism MLP not strictly better by >0.005
controls fail
```

KAN writer strong success：

```text
KAN source_h4800 >= 0.005
dataset_seed_pass_h4800 >= 5/9
tail_recovery_rate_h4800 >= 0.60
LineC_recovery_rate_h4800 >= 0.60
KAN_specific_delta_vs_MLP_same_mechanism >= 0.005
efficiency full-loop pass
controls fail
```

If KAN writer fails but MLP retained success exists：

```text
carrier mismatch supported；
continue basis-channel redesign, not FU method tuning。
```

If KAN writer and MLP both fail：

```text
current FU observable insufficient；
stop adding update variants；
return to source-target theory。
```

---

# 8. Full carrier × mechanism matrix

## 8.1 Carriers

```text
C0: MLP
  active FU mechanism discovery line。

C1: D-CHE officialized kernel
  main KAN carrier。

C2: D-FOU officialized kernel
  main alternate KAN carrier。

C3: LQ
  targeted forward repair + smoke only。

C4: D-RAT
  monitor + rational eval repair smoke only。

C5: D-RBF
  sparse local support smoke only。

C6: D-WAV
  sparse support smoke only。
```

## 8.2 Mechanism families

```text
F0 source dynamics classification
F1 MLP dual-memory / source-retention mechanisms
F2 AdamW-free / optimizer-dynamics decoupling
F3 function-space target reset
F4 KAN source-channel writer
F5 controls and attribution
```

## 8.3 Mandatory controls

每个 positive-looking result 必须比较：

```text
NoOpMatchedOverhead
RandomMatchedNorm
RandomMatchedActuationR2
SameHorizonRandomTarget
SignFlipTarget
CorruptTarget
RecoveryOnly
AdamWParallelDirection
SGDMomentumParallelDirection
SameKernelNoFU
SameSourceStateRandomCommit
MLP same-mechanism control
```

---

# 9. 4GPU 动态调度

## 9.1 GPU 分工

```text
GPU0:
  S0.6 code tests；
  D-CHE E1 full-loop officialization；
  D-CHE F4 source-channel writer。

GPU1:
  MLP F0/F1/F2 source dynamics；
  MLP optimizer-dynamics comparison；
  MLP controls。

GPU2:
  D-FOU E2 full-loop officialization；
  D-FOU F4 source-channel writer。

GPU3:
  F3 function-space target contrast；
  LQ/RAT/RBF/WAV smoke；
  controls / figure / packet generation；
  backup queue worker。
```

## 9.2 动态队列要求

必须生成：

```text
v21_01_runnable_queue.csv
v21_01_gpu_assignment_manifest.csv
v21_01_gpu_utilization_dashboard.csv
v21_01_idle_violation.csv
v21_01_deferred_items.csv
v21_01_queue_drain_report.csv
```

硬规则：

```text
if runnable_queue_nonempty and any_gpu_idle_minutes > 10:
    execution_contract_violation = 1
    completed_no_go_allowed = 0
```

如果某 GPU 的 primary queue 完成，必须自动取：

```text
1. pending controls；
2. pending h3200/h4800 extension；
3. pending efficiency full-loop rows；
4. pending code/figure artifacts；
5. pending low-priority smoke。
```

---

# 10. 成功标准

## 10.1 S0.6 code success

```text
missing_source_file_count = 0
import_error_count = 0
all metric tests pass
all mechanism semantic tests pass
official fused status consistency pass
```

## 10.2 E1/E2 efficiency success

```text
D-CHE or D-FOU full-loop:
  forward_ratio <= 1.25
  backward_ratio <= 1.40
  step_ratio <= 1.25
  memory_ratio <= 1.10
  kernel_gradcheck_pass = 1
  full_loop_uses_kernel = 1
  audit_cost_separated = 1
```

Official efficiency candidate：

```text
forward_ratio <= 1.15
step_ratio <= 1.15
memory_ratio <= 1.05
official_kernel_usable = 1
```

## 10.3 Functional weak success

```text
source_h3200 >= 0.005
dataset_seed_pass_h3200 >= 3/9
retention_h3200_over_h1600 >= 0.50
tail_recovery_rate_h3200 >= 0.40 or LineC_recovery_rate_h3200 >= 0.40
matched controls fail
```

## 10.4 Functional productive success

```text
source_h4800 >= 0.005
dataset_seed_pass_h4800 >= 5/9
retention_h4800_over_h3200 >= 0.50
tail_recovery_rate_h4800 >= 0.60
LineC_recovery_rate_h4800 >= 0.60
ECE_debt_recovery_rate_h4800 >= 0.60
Brier_debt_recovery_rate_h4800 >= 0.60
AUCtime_ratio_h4800 <= 1.05
controls fail
```

## 10.5 KAN-specific success

```text
KAN source_h4800 >= 0.005
KAN dataset_seed_pass_h4800 >= 5/9
KAN_specific_delta_vs_MLP_same_mechanism >= 0.005
D-CHE or D-FOU full-loop efficiency pass
controls fail
```

## 10.6 Official S5 不降低

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
official kernel gate pass
controls fail
promotion_allowed = 1
```

---

# 11. 必须记录的指标清单

## 11.1 Code metrics

```text
compileall_ok
import_error_count
missing_source_file_count
packet_sha256_count
linec_fast_golden_pass
linec_channel_golden_pass
retention_formula_pass
debt_accounting_pass
route_aggregation_pass
update_sign_pass
mechanism_noncollapse_pass
profiler_phase_pass
kernel_gradcheck_pass
official_fused_status_consistency_pass
```

## 11.2 Efficiency metrics

```text
forward_only_ms
backward_grad_ms
optimizer_update_ms
fu_direction_ms
fu_commit_ms
linec_audit_ms
horizon_readback_ms
step_total_ms
forward_ratio_vs_mlp
backward_ratio_vs_mlp
update_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
kernel_launch_count
dense_materialized_bytes
workspace_bytes
full_loop_uses_kernel
profiler_uses_kernel
```

## 11.3 Source metrics

```text
source_h100
source_h400
source_h800
source_h1600
source_h2400
source_h3200
source_h4800
source_h6400
source_derivative_h800_1600
source_derivative_h1600_3200
source_derivative_h3200_4800
retention_h1600_over_h800
retention_h3200_over_h1600
retention_h4800_over_h3200
washout_flag
late_rebound_flag
weak_stable_flag
retained_flag
```

## 11.4 Debt metrics

```text
tail_debt_peak
tail_debt_final
tail_recovery_rate
LineC_debt_peak
LineC_debt_final
LineC_recovery_rate
ECE_debt_peak
ECE_debt_final
ECE_recovery_rate
Brier_debt_peak
Brier_debt_final
Brier_recovery_rate
AUCtime_ratio
```

## 11.5 Functional mechanism metrics

```text
short_state_norm
long_state_norm
short_long_cosine
commit_norm
commit_cosine_with_gradient
commit_cosine_with_adamw
cumulative_optimizer_projection_on_fu
source_channel_projection
reservoir_projection
target_ActuationR2
target_B2_transfer_gain
same_actuation_random_source
KAN_specific_delta_vs_MLP_same_mechanism
control_equivalent_fraction
```

---

# 12. 必须可视化的图

```text
1. code_truth_gate_dashboard.svg
2. efficiency_full_loop_vs_profiler_waterfall.svg
3. D-CHE_kernel_component_waterfall.svg
4. D-FOU_kernel_component_waterfall.svg
5. source_trajectory_horizon_grid.svg
6. source_dynamics_phase_portrait.svg
7. washout_vs_retention_heatmap.svg
8. debt_recovery_phase_plane.svg
9. target_actuation_vs_retention_scatter.svg
10. MLP_vs_KAN_same_mechanism_source.svg
11. optimizer_overwrite_projection_curve.svg
12. low_degree_low_frequency_source_energy.svg
13. control_attribution_matrix.svg
14. gpu_utilization_dashboard.svg
```

---

# 13. 决策规则与 Codex 自动 fallback

## 13.1 如果 S0.6 失败

不能跑科学实验。Codex 必须：

```text
1. 补齐 source tree；
2. 修 import closure；
3. 修 route aggregation tests；
4. 修 official_fused_status consistency；
5. 重打 v21_01_code_review_packet.zip。
```

## 13.2 如果 D-CHE / D-FOU full-loop efficiency 失败

Codex 必须先判断：

```text
profiler path != full-loop path？
audit cost mixed？
kernel not actually used？
batch launch overhead？
dense materialization returned？
```

然后执行对应 fallback，不允许直接写 basis fail。

## 13.3 如果 MLP source 仍 h3200 washout

Codex 必须先运行：

```text
dual-memory source state；
schedule-free source iterate；
strong+weak source combination；
AdamW-free comparison；
cumulative overwrite diagnosis。
```

只有这些都失败，才能写：

```text
MLP source-retention current theory fail。
```

## 13.4 如果 function-space target high ActuationR2 but no source

Codex 必须运行：

```text
target contrast；
same-ActuationR2 random control；
source-channel target；
low-frequency/low-degree target；
readout-only target。
```

不允许继续调 actuation rank。

## 13.5 如果 KAN source writer fail

如果 MLP retained success exists：

```text
carrier mismatch route；
继续 KAN source-channel redesign。
```

如果 MLP also fail：

```text
source observability theory fail；
停止新增 FU token；
回到 target/source theory。
```

## 13.6 如果 late rebound 出现

不能当 success。Codex 必须：

```text
1. 独立 rerun x3；
2. matched controls；
3. h2400/h3200/h4800/h6400 trajectory；
4. 判定 stochastic rebound / delayed migration / control-equivalent。
```

---

# 14. v21.01 最终必须回答的问题

v21.01 结束时，不能再只给 route。必须回答：

```text
1. 代码包是否自包含？缺文件是否为 0？
2. D-CHE / D-FOU 的 high-efficiency kernel 是否能 full-loop officialize？
3. MLP source 是否能从 h1600 留到 h3200/h4800？
4. AdamW 是否长期 washout FU source？
5. high ActuationR2 失败是 target 错，还是 actuation 错？
6. KAN 是否有可写入 source 的 low-degree / low-frequency channel？
7. PopRisk/SNR 是否能合法预测 retained source？如果不能，是否停止用它生成 direction？
8. 是否至少有一个 S2/S3 functional route 打开？
```

如果这些问题没有回答，即使跑了很多 rows，也算执行设计失败。

---

# 15. v21.01 的最终一句话

$$
\boxed{
\text{v21.01 不再寻找“更强局部 update”，}
\text{而是要把 source-retention 变成可测、可解释、可复现的训练动力学。}
}
$$

同时：

$$
\boxed{
\text{D-CHE / D-FOU 的效率进展必须从 profiler pass 变成 full-loop official kernel evidence。}
}
$$
