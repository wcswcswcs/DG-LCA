# DG-KAN v22.01：Early-Source Retention / Terminal-Collapse Autopsy / D-CHE-D-FOU Officialization / D-RAT-D-RBF Active Repair / 4GPU 动态并行完整计划

> 版本：v22.01 execution plan  
> 生成时间：2026-06-04  
> 目标：停止围绕 late rebound 做表面修补，直接解决 source retention、terminal collapse、KAN carrier writeback、basis efficiency officialization 四个核心问题。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline official path；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / Brier / AUCtime 只能作为 audit / gate / debt readback，不能作为方向源。

---

# 0. 项目总目标与 v22 后的真实状态

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是让某个 horizon 的 source 临时为正。项目目标是：

$$
\boxed{
\text{构建 strict FC-PureKAN base + functional update 的训练系统，}
\text{在效率接近 same-param MLP 的前提下，获得更好的长期训练动力学。}
}
$$

最终必须同时满足：

```text
1. KAN basis 的 forward / backward / update / memory 与同参数量 MLP 可比。
2. Functional update 的收益不能被 AdamW、SGD、NoOp、RandomMatched、StableRandom、RecoveryOnly 或 MLP same-mechanism controls 解释。
3. Source 不能只是 h800 / h3200 局部正值，而必须形成 continuous retention chain。
4. Tail / LineC / calibration / AUCtime debt 可以短期产生，但必须在长 horizon 被偿还。
5. 如果 functional update 只在 MLP 上成立，要写成 generic training-dynamics insight，不能写成 KAN-specific。
6. 只有 KAN 在同机制、同参数量、同效率约束下胜过 MLP，才允许讨论 KAN-specific functional advantage。
```

v22 的真实状态是：

```text
代码：
  v22 code packet 语法可编译，但不是自包含包；缺 dgkan.fu.core、linec.py、efficiency_v20/v21、run_v21_01_common 等关键依赖。
  source-chain early rule 使用 >=0，导致 zero-control 被错误计为 early-source-chain。

效率：
  D-CHE / D-FOU 继续进入 MLP-like envelope。
  D-CHE E1=6、S1=6；D-FOU E1=3、S1=3。
  但源码闭包和 official fused status 仍未完全统一。
  D-RAT / D-RBF 没有实质推进，不能继续被动躺在 ForwardBlocked 表里。

Functional update：
  v21.01 是 late rebound without continuous retention。
  v22 推进到 h3200 continuous candidate，但 h4800 全部失败。
  candidate_h3200=12，candidate_h4800=0，promotion_allowed=0。
  真正 blocker 已从 “没有 h3200 source” 变成 “h3200 到 h4800 terminal collapse”。
```

因此 v22.01 不是继续扩大 F57-F74，也不是继续调 KSW2 / M43 / terminal floor。v22.01 要强制回答：

$$
\boxed{
\text{为什么 source 能到 h3200，却不能跨过 h4800？}
}
$$

以及：

$$
\boxed{
\text{D-CHE / D-FOU 的效率 pass 能否变成 full-loop official kernel evidence，}
\text{D-RAT / D-RBF 是否存在可修复的 efficient carrier path？}
}
$$

---

# 1. 本轮必须解决的核心问题

## 1.1 代码与指标问题：先修尺子，否则所有实验继续误导

v22 暴露了两个会直接导致原地打转的问题。

第一个是 code packet 不自包含。复盘里写 CodeRoute pass，但独立解压后缺少核心依赖。这意味着 Codex 运行环境里的自检通过，不能等价于外部审计者可以复现和复核实现。

第二个是 source-chain early rule 错误。当前规则：

```python
early = int(s100 >= 0 and s400 >= 0 and s800 >= 0)
```

会把 AdamW control / zero source 也计为 early source。v22.01 必须改为：

$$
EarlyChain = 1
\Longleftrightarrow
Source_{h100} \ge \tau_s,
Source_{h400} \ge \tau_s,
Source_{h800} \ge \tau_s,
ControlEquivalent = 0.
$$

其中：

$$
\tau_s = 0.005.
$$

如果为了探索需要 relaxed 版本，可以另记：

$$
\tau_{weak}=0.001,
$$

但 relaxed early chain 不能打开 S2/S3/S4，只能进入 diagnostic queue。

## 1.2 Functional update 问题：当前 blocker 是 terminal collapse，不是 h800 source 不足

v22 之后不能再说 “没有 source”。更准确是：

```text
MLP terminal-source family 能形成 h100 -> h3200 continuous chain；
但 h4800 terminal collapse。
KAN 仍没有 true retained source；
D-CHE / D-FOU 的 late positive 很多，但不是 continuous retention。
```

所以 v22.01 的 functional 目标不再是 “找更强 h800/h3200 source”，而是建立 source transition model：

$$
SourceTrajectory = \{S_{100}, S_{400}, S_{800}, S_{1600}, S_{2400}, S_{3200}, S_{4000}, S_{4800}, S_{6400}\}.
$$

并明确判定：

```text
strong-washout：早期强，但 h3200/h4800 掉。
weak-stable：早期弱，但长程稳定。
late-rebound：早期负，后期正。
terminal-collapse：到 h3200 连续正，但 h4800 掉。
continuous-retention：h100/h400/h800/h1600/h3200/h4800 连续正并打过 controls。
```

v22.01 的关键是把 terminal-collapse 分解成可诊断因果路径，而不是继续调 terminal floor。

## 1.3 效率问题：D-CHE / D-FOU 已经不是天然慢，D-RAT / D-RBF 不能继续缺席

D-CHE / D-FOU 已经有 profiler repeat rows 进入 MLP-like envelope。这是实质工程进展。下一步不是继续 census，而是 officialization closure：

```text
同一个 fused/no-materialize kernel 必须被 profiler、full training loop、functional runner 同时使用。
truth table、kernel_gradcheck、official_fused_status_matrix 必须口径一致。
```

但 D-RAT / D-RBF 不能继续只在历史表里 ForwardBlocked。v22.01 必须给它们 active repair：

```text
D-RAT：拆 numerator / denominator / reciprocal / derivative telemetry / safety path。
D-RBF：拆 dense materialization / active-center / compact support / local K / Gaussian exp / local fused backward。
```

D-RAT / D-RBF 不要求本轮 official pass，但必须至少达到 near-E1 或明确给出不可修复 blocker。

---

# 2. 哪些旧方向可能因为代码 / 指标问题仍有推进希望

## 2.1 可以重开的方向，但必须换语义

### PopRisk / SNR

v13.6-v13.8 的 PopRisk/SNR 线失败不能简单写成 “population-risk 不行”。当时主要问题是 one-shot parameter SNR、cover objective 误用、basis lift 不稳定、后续 route 转向 cover formation。当前理论启发是 signal channel / reservoir：我们要的是能预测 retained source 的 train-only estimator，而不是 one-step gradient SNR。

v22.01 中 PopRisk/SNR 可以重开，但只能作为：

```text
precommit source-retention predictor；
source-channel noise estimator；
not direction by itself unless heldout train split selector pass。
```

### Split-consensus / G7R lineage

v15.04-v15.7 的 split-consensus 曾经给出 source signal，但后来 source/hazard 共线、static projection 失败。旧结论只能说明 “static source-hazard separation 不行”，不能说明 “split-consensus 不能作为 source estimator”。

v22.01 中 split-consensus 不允许直接提交 parameter update，只能用于：

```text
cross-split target observability；
early-source selector；
control deconfounding；
source-chain predictor。
```

### Function-space actuation

v20/v21 显示 high ActuationR2 不产生 retained source。这说明 actuation 不是主瓶颈，target 定义才是主瓶颈。但 function-space actuation 仍有价值：它能把 “target 错” 和 “actuation 做不到” 分开。

v22.01 中 function-space 只负责测试：

```text
target 是否 observable；
target 是否 transfer；
target 是否比 random/signflip/corrupt target 更可能形成 early source chain。
```

### MLP functional update

MLP 不是普通 control，而是 active functional lab。v22 的 h3200 continuous candidates 主要在 MLP 上出现，说明 MLP 是研究 source dynamics 的最好平台。

MLP-FU 方向继续推进，但结论必须分清：

```text
MLP 成功、KAN 不成功：generic dynamics + KAN carrier mismatch。
MLP 不成功、KAN 不成功：current observable/target theory 不足。
KAN 成功、MLP 不成功：KAN-specific functional carrier。
```

### Graph-free analytic adjoint / kernel-first 路线

v6.1 的 graph-free analytic adjoint 失败在 P2 efficiency，而不是梯度正确性。现在 D-CHE / D-FOU kernel 已经明显改善，所以 graph-free / no-materialize / analytic backward 应作为 efficiency officialization 的基础，不是旧失败路线。

### D-RAT / D-RBF

D-RAT / D-RBF 过去失败大多混合了实现慢、dense basis materialization、denominator/center telemetry 没拆、task-health 不稳。不能因为旧实现慢就判死。

v22.01 中它们是 active repair 线，但不允许抢 D-CHE / D-FOU 主预算。

## 2.2 不应重启的方向

以下方向不应消耗 v22.01 主预算：

```text
action bank / controller / action token；
reset route / optimizer-state reset family；
cover objective 小修；
M31/M32 同族修补；
G-token / N-token / Q-token 扩展；
late rebound scale / alt period / terminal floor 小调；
dataset-name branch / seed-specific scale；
用 LineC/tail/calibration/AUC 反推方向。
```

原因：

```text
cover objective 已被 oracle validation 判为 invalid；
action-bank oracle upper bound 不足；
M31/M32 partial positive 未被 independent confirmation 复现；
late rebound 已经大量出现，但 h4800 retained source 仍为 0；
继续调 late rebound 只会重复 terminal-collapse。
```

---

# 3. 当前研究进展给我们的启发

## 3.1 Signal channel / reservoir：source 必须进入可泛化通道

当前 generalization 理论把 output space 分成 signal channel 和 reservoir。真正会泛化的是 signal channel 中的 coherent population signal；reservoir 中的变化可能对 test 不可见。对应到 DG-KAN：

```text
h3200 late positive 不等于 source 成功；
stable-random h6400 positive 是危险信号；
只有 train-stream target 能产生 early continuous source chain，才可能是 signal-channel writer。
```

v22.01 因此要求每个 candidate 记录 source trajectory、LineC-channel、tail/calibration/AUC debt，并和 stable-random / random-matched controls 比较。

## 3.2 Deep Manifold / boundary-conditioned iteration：FU 是训练边界条件，不是单步修正

Deep Manifold 的启发是：训练在移动坐标和边界条件下构造 fixed-point regions。Functional update 不应被理解成 “单步参数补丁”，而应被理解成 “改变训练路径的边界条件”。因此：

```text
短期坏不一定失败；
late rebound 也不一定成功；
关键是 source 是否形成稳定 fixed-point region，并在 h4800/h6400 保持。
```

v22.01 记录 source derivative、optimizer displacement、debt transition，就是为了判定训练路径是否真的进入了稳定 source region。

## 3.3 Schedule-Free / AdEMAMix / SOAP / Muon / Cautious 的启发

这些研究不能机械搬成 “换 optimizer”。它们提供的是训练动力学原则：

```text
Schedule-Free / SF-NorMuon：
  fast iterate / slow iterate / averaging 对 long-horizon stability 重要。

AdEMAMix：
  short memory 和 long memory 同时存在，旧梯度可能长期有用。

SOAP / Shampoo / Muon：
  matrix/block coordinate 可能比逐参数 AdamW 更适合 hidden/readout/basis block。

Cautious Optimizers：
  update-gradient alignment 是 conflict diagnostic，而不是硬 mask。
```

v22.01 将这些启发转化为具体实验：

```text
terminal collapse autopsy 中记录 optimizer cumulative projection；
MLP source lab 中测试 short/long source memory；
KAN source writer 中测试 readout/block/low-degree/low-frequency writer；
alignment 只作为诊断，不作为默认 hard filter。
```

---

# 4. v22.01 总体实验结构

v22.01 分为三大部分，每部分都必须有可判定结果。

```text
Part A: S0.8 Code / Metric / Artifact Truth Gate
  修代码闭包、source-chain bug、official kernel status、queue artifacts。

Part B: Basis Efficiency Officialization + D-RAT/D-RBF Active Repair
  D-CHE / D-FOU full-loop official closure；
  D-RAT / D-RBF active repair + limited smoke；
  LQ / D-WAV 低预算保留。

Part C: Functional Update Source-Retention Program
  修 terminal collapse；
  建 train-only early-source selector；
  重设 function-space target；
  迁移 MLP source dynamics 到 KAN source-channel writer；
  判定 D-RAT/D-RBF 是否有 carrier signal。
```

---

# 5. Part A：S0.8 Code / Metric / Artifact Truth Gate

## 5.1 目标

S0.8 不是形式审计，而是防止错误指标/标准导致继续原地打转。只有 S0.8 全部通过，才允许写 scientific no-go 或 promotion-precondition。

## 5.2 Codex 必须打包的文件

Codex 必须生成：

```text
v22_01_code_review_packet.zip
v22_01_results_bundle.zip
```

`v22_01_code_review_packet.zip` 必须包含：

```text
00_README.md
packet_manifest.csv
packet_sha256_manifest.csv

02_SOURCE_TREE/
  dgkan/**/*.py
  experiments/**/*.py
  scripts/**/*.py if used

03_IMPORT_CLOSURE/
  v22_01_py_compile.log
  v22_01_import_closure.csv
  v22_01_missing_dependency_report.csv

04_LINEC_CORRECTNESS/
  linec.py source snapshot
  v22_01_linec_fast_golden.csv
  v22_01_linec_channel_golden.csv
  v22_01_linec_measurement_invalid_tests.csv

05_SOURCE_CHAIN_AND_RETENTION/
  source_chain.py source snapshot
  v22_01_source_chain_unit_tests.csv
  v22_01_retention_formula_tests.csv
  v22_01_route_reaggregation_tests.csv

06_DEBT_ACCOUNTING/
  debt_accounting.py source snapshot
  v22_01_tail_debt_tests.csv
  v22_01_linec_debt_tests.csv
  v22_01_ece_brier_debt_tests.csv
  v22_01_auctime_debt_tests.csv

07_FUNCTIONAL_MECHANISMS/
  mechanisms.py
  source_channel.py
  source_state.py
  function_space_actuation.py
  matrix_block.py
  poprisk_snr.py
  v22_01_mechanism_semantic_contract.csv
  v22_01_mechanism_noncollapse_tests.csv

08_EFFICIENCY_KERNELS/
  cheby_fused.py
  fourier_fused.py
  rational_fused.py if implemented
  rbf_local_fused.py if implemented
  efficiency_v20.py
  efficiency_v21.py
  efficiency_v22.py
  efficiency_v22_01.py
  v22_01_kernel_gradcheck.csv
  v22_01_official_fused_status_matrix.csv

09_EXPERIMENT_RUNNERS/
  run_v22_01_s08_truth_gate.py
  run_v22_01_efficiency_officialization.py
  run_v22_01_drat_drbf_active_repair.py
  run_v22_01_terminal_collapse_autopsy.py
  run_v22_01_mlp_source_lab.py
  run_v22_01_function_space_target_reset.py
  run_v22_01_kan_source_writer.py
  run_v22_01_controls_and_finalize.py

10_GPU_QUEUE/
  v22_01_runnable_queue.csv
  v22_01_gpu_assignment_manifest.csv
  v22_01_gpu_utilization_dashboard.csv
  v22_01_idle_violation.csv
  v22_01_deferred_items.csv
  v22_01_queue_drain_report.csv
```

如果任何 required source 缺失，不允许写：

```text
CodeRoute = CodeMetricMechanismPassed
```

只能写：

```text
CodeRoute = SourcePacketIncomplete
```

## 5.3 S0.8 必修检查

### 5.3.1 自包含 import closure

必须在解压后的 packet 内执行：

```bash
python -m compileall -q dgkan experiments
python scripts/import_scan.py --root . --out 03_IMPORT_CLOSURE/v22_01_import_closure.csv
```

通过标准：

```text
compileall_ok = 1
core_import_error_count = 0
legacy_proxy_system_exit_count 可存在，但必须列入 isolated_legacy_proxy.csv
missing_required_dependency_count = 0
```

### 5.3.2 source-chain rule 修复

必须修正：

$$
EarlyChain = 1
\Longleftrightarrow
S_{100}\ge0.005,
S_{400}\ge0.005,
S_{800}\ge0.005,
ControlEquivalent=0.
$$

单元测试必须覆盖：

```text
all-zero control -> early_source_chain = 0
all-zero source but h3200 positive -> late_rebound only
h100/h400/h800 positive but random control explains -> early_source_chain = 0
h100/h400/h800/h1600/h3200 positive but h4800 negative -> terminal_collapse
h100/h400/h800/h1600/h3200/h4800 positive -> continuous_retention
```

### 5.3.3 route reaggregation

必须从 raw matrices 重新聚合：

```text
candidate_early_chain
candidate_continuous_h3200
candidate_h4800
candidate_h6400
late_rebound_groups
terminal_collapse_groups
control_equivalent_groups
```

不得使用 single-row max 打开 route。

### 5.3.4 official kernel status consistency

以下文件必须一致：

```text
v22_01_efficiency_truth_table.csv
v22_01_kernel_gradcheck.csv
v22_01_official_fused_status_matrix.csv
```

如果 truth table 写 `official_fused_kernel_complete=1`，则 kernel_gradcheck 必须也写：

```text
official_fused_kernel_complete = 1
gradcheck_pass = 1
no_materialize_complete = 1
implementation_path 指向真实 source file
```

否则该 row 只能写：

```text
official_like_timing_only = 1
official_fused_kernel_complete = 0
```

### 5.3.5 4GPU dynamic queue

必须输出：

```text
runnable_queue.csv
assignment_manifest.csv
utilization_dashboard.csv
idle_violation.csv
queue_drain_report.csv
```

硬规则：

```text
runnable_queue 非空 && 任一 GPU idle > 10 min
=> execution_contract_violation = 1
=> final route 不允许写 completed no-go。
```

---

# 6. Part B：Basis Efficiency Officialization + D-RAT/D-RBF Active Repair

## 6.1 总目标

Efficiency 路线必须回答：

```text
1. D-CHE / D-FOU 的 MLP-like efficiency 是否在 full functional loop 中真实成立？
2. D-RAT / D-RBF 是否只是实现慢，还是 basis 本身不适合作为 efficient carrier？
3. 每个 basis 到底卡在 forward、backward、update、audit、memory、kernel fallback 哪一项？
```

## 6.2 Line E-CHE：D-CHE full-loop officialization

### 假设

D-CHE 的 CHE21-R2 / CHE21-R4 系列已经接近 MLP-like efficiency；v22.01 要验证它不是 profiler artifact。

### 实验

候选：

```text
CHE22-R2-low-degree-k3-official-fullloop
CHE22-R4-k3-gradbuf-triton-fullloop
CHE22-R4-k5-gradbuf-triton-fullloop
```

每个候选在：

```text
batch = 8, 32, 128, 256, 512
hidden = official hidden setting
same-param MLP reference
```

上测：

```text
profiler path
full AdamW training path
full functional runner path
LineC/horizon readback separated path
```

### 记录指标

```text
forward_only_ms
backward_grad_ms
optimizer_update_ms
functional_direction_ms
functional_commit_ms
linec_audit_ms
horizon_readback_ms
training_step_ms
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
kernel_grad_relerr
kernel_grad_cosine
no_materialize_complete
official_fused_kernel_complete
fallback_kernel_used
```

### 判断标准

Exploration pass E1-CHE：

```text
forward_ratio_vs_mlp <= 1.25
step_ratio_vs_mlp <= 1.25
memory_ratio_vs_mlp <= 1.10
gradcheck_pass = 1
fallback_kernel_used = 0
```

Official-like S1-CHE：

```text
>= 2 variants × >= 3 batch sizes pass E1-CHE
functional_runner_uses_same_kernel = 1
audit_cost_separated = 1
official_fused_status_consistent = 1
```

如果 batch512 fail，但 128/256 pass，则不能写 failure；写：

```text
CHE-E2-UsableFunctionalEnvelope-Batch512Blocked
```

Codex fallback：

```text
若 profiler pass 但 full-loop fail：检查 kernel fallback、audit contamination、functional_direction overhead。
若 gradcheck fail：回退 torch reference，禁用 official status，保留 timing diagnostic。
若 batch512 fail：先修 launch/block tiling，不阻断 batch128/256 functional tests。
```

## 6.3 Line E-FOU：D-FOU full-loop officialization

### 假设

D-FOU tablelookup / low-frequency / no-materialize path 可以成为第二个 efficient carrier。

### 实验候选

```text
FOU22-R2-lowfreq-k2-stream-fullloop
FOU22-R3-tablelookup-bandreadout-fullloop
FOU22-R4-k4-triton-no-materialize-fullloop
```

同样测 batch 8/32/128/256/512。

### 记录指标

除 CHE 指标外，FOU 还必须记录：

```text
frequency_band_count
low_band_energy
mid_band_energy
high_band_energy
sin_cos_eval_ms
table_lookup_ms
band_readout_contraction_ms
bandwise_backward_ms
high_frequency_quarantine_rate
```

### 判断标准

Exploration pass E1-FOU：

```text
forward_ratio_vs_mlp <= 1.25
step_ratio_vs_mlp <= 1.25
memory_ratio_vs_mlp <= 1.10
gradcheck_pass = 1
fallback_kernel_used = 0
```

Official-like S1-FOU：

```text
>= 1 variant × >= 3 batch sizes pass E1-FOU
functional_runner_uses_same_kernel = 1
official_fused_status_consistent = 1
```

Codex fallback：

```text
若 FOU-R3 tablelookup fast 但 gradient fail：先修 band-readout backward。
若 high frequency band 造成 task instability：保留 low-frequency path，标记 high-band reservoir，不做 dataset tuning。
若 batch512 fail：保留 batch128/256 official-ish route，并安排 tiling repair。
```

## 6.4 Line E-RAT：D-RAT active repair

### 假设

D-RAT 过去慢主要来自 rational numerator/denominator eval、division、denominator safety / derivative telemetry 混入 training path。若拆开并 fused，D-RAT 可能进入 near-E1。

### 实验候选

```text
RAT22-R0-current-reference
RAT22-R1-horner-numden-fused
RAT22-R2-branchless-denominator-safe
RAT22-R3-reciprocal-stabilized-exact
RAT22-R4-telemetry-separated-training-path
RAT22-R5-grouped-vectorized-rational
RAT22-R6-numden-block-update-smoke
```

说明：`reciprocal-stabilized-exact` 必须数值等价，不允许 approximate reciprocal 进入 official；approx reciprocal 只能 diagnostic。

### 记录指标

```text
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
numerator_eval_ms
denominator_eval_ms
reciprocal_ms
derivative_telemetry_ms
denominator_safety_ms
training_path_telemetry_free
rational_den_min
rational_den_p01
rational_den_condition
r_prime_p99
r_double_prime_p99
gradcheck_pass
```

### 近似成功标准 Near-E1-RAT

```text
forward_ratio_vs_mlp <= 3.0
step_ratio_vs_mlp <= 2.0
memory_ratio_vs_mlp <= 1.20
denominator_safety_pass = 1
gradcheck_pass = 1
training_path_telemetry_free = 1
```

若达到 Near-E1-RAT，进入 limited functional smoke，只跑：

```text
RAT-FU-smoke-readout-commit
RAT-FU-smoke-den-slope-safe-source
RAT-FU-smoke-source-chain-selector
```

Codex fallback：

```text
若 denominator safety fail：不跑 functional，先修 denominator parameterization。
若 forward > 3 但 step <= 2：拆 numerator/denominator/telemetry waterfall，继续 targeted kernel repair。
若 Near-E1 达成但 functional source 负：标记 EfficientEnoughNoSource，不要继续调 rational scale。
```

## 6.5 Line E-RBF：D-RBF active repair

### 假设

D-RBF / FastKAN 过去慢可能是因为 dense basis materialization，没有实现 active-center local K。若实现 compact local fused path，D-RBF 仍可能是有价值 carrier。

### 实验候选

```text
RBF22-R0-current-dense-reference
RBF22-R1-active-center-mask
RBF22-R2-compact-local-k4
RBF22-R3-compact-local-k8
RBF22-R4-width-conditioned-local-k4
RBF22-R5-no-dense-materialization-backward
RBF22-R6-local-support-functional-smoke
```

### 记录指标

```text
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
basis_materialized_bytes
active_center_fraction
local_k_effective
empty_center_fraction
width_p01
width_p50
width_p99
rbf_exp_eval_ms
local_gather_ms
local_backward_ms
center_grad_snr
width_grad_snr
gradcheck_pass
```

### 近似成功标准 Near-E1-RBF

```text
forward_ratio_vs_mlp <= 3.0
step_ratio_vs_mlp <= 2.0
memory_ratio_vs_mlp <= 1.20
basis_materialized_bytes reduced by >= 70%
gradcheck_pass = 1
active_center_fraction between 0.05 and 0.60
```

若达到 Near-E1-RBF，进入 limited functional smoke：

```text
RBF-FU-smoke-local-support-source
RBF-FU-smoke-readout-only-commit
RBF-FU-smoke-width-stable-source
```

Codex fallback：

```text
若 active_center_fraction 太低：修 width/init/center occupancy，不按 dataset 调参。
若 memory 下降但 forward 仍慢：拆 exp/local gather/local backward。
若 local path source positive 但 efficiency不够：标记 InefficientButSourceSignal，保留供理论分析，不进入 official。
```

## 6.6 LQ / D-WAV 低预算线

LQ：

```text
只做 recurrence + fused projection update targeted repair。
目标 forward <= 4.0, step <= 2.0，若达不到，不进入 functional smoke。
```

D-WAV：

```text
只做 sparse support smoke。
目标证明 support-index path 是否有 forward 降低趋势；不进入 full functional proof。
```

---

# 7. Part C：Functional Update Source-Retention Program

## 7.1 总体目标

Functional update 本轮必须回答：

```text
1. v22 的 h3200 continuous candidates 为什么 h4800 collapse？
2. 这种 collapse 是 optimizer washout、debt 爆发、dataset/seed heterogeneity、control-equivalent drift，还是 target 错？
3. F46 train-loss selector 是 retrospective 相关性，还是可 precommit 的 causal selector？
4. MLP 的 h3200 chain 能否被保到 h4800？
5. KAN 是否存在 low-degree / low-frequency / readout source-channel writer？
6. D-RAT / D-RBF 是否有任何 carrier-specific source smoke？
```

## 7.2 Line F0：v22 source-chain 修正重聚合

### 目标

在修正 early rule 后，重新计算 v22 全部 functional summary。不得沿用旧 route 数字。

### 输入

```text
v22 raw source_retention_matrix
v22 debt_accounting_matrix
v22 controls matrix
```

### 输出

```text
v22_01_reaggregated_v22_source_chain.csv
v22_01_reaggregated_v22_route.json
v22_01_reaggregated_candidate_diff.csv
```

### 记录指标

```text
old_early_chain
new_early_chain
old_continuous
new_continuous
old_h3200
new_h3200
old_h4800
new_h4800
control_equivalent
changed_reason
```

### 判定

如果修正后 `candidate_h3200` 大幅减少，v22 的 functional progress 必须降级为：

```text
R0-SourceChainRuleInflatedPreviousRoute
```

如果修正后仍有 h3200 continuous candidates，则进入 terminal collapse autopsy。

## 7.3 Line F1：Terminal-collapse autopsy

### 假设

v22 的 h3200 continuous candidate 在 h4800 collapse，可能由以下机制之一导致：

```text
A. optimizer cumulative overwrite；
B. tail / calibration debt delayed explosion；
C. source target 进入 reservoir，不进入 signal channel；
D. dataset/seed-specific source，不具 population coherence；
E. stable-random / random-matched control 同样 late positive，说明 control-equivalent drift；
F. h3200 是短期 alignment，h4800 固定点不稳定。
```

### 实验对象

对 v22 修正后所有 h3200 candidates，以及 top stable-random / random-matched h3200 positives，执行：

```text
horizon = h100, h400, h800, h1200, h1600, h2400, h3200, h4000, h4800, h6400
```

### 记录指标

```text
source_h100...source_h6400
source_derivative_h800_to_h1600
source_derivative_h1600_to_h2400
source_derivative_h2400_to_h3200
source_derivative_h3200_to_h4000
source_derivative_h4000_to_h4800
retention_h3200_over_h800
retention_h4800_over_h3200
optimizer_cumulative_projection_on_source
adamw_projection_on_source
sgd_projection_on_source
momentum_projection_on_source
lookahead_slow_state_projection
tail_debt_peak / final / recovery
LineC_fast_debt_peak / final / recovery
LineC_channel_debt_peak / final / recovery
ECE_debt_peak / final / recovery
Brier_debt_peak / final / recovery
AUCtime_ratio_vs_best_control
dataset_seed_heterogeneity_score
control_equivalent_fraction
stable_random_h4800_positive_fraction
```

### 分类标准

```text
OptimizerWashout:
  cumulative optimizer projection on h3200 source < -0.2
  and source drops h3200 -> h4800.

DebtCollapse:
  source drop aligns with tail/ECE/Brier/LineC debt spike.

ControlDrift:
  stable-random h4800 positive fraction >= candidate h4800 positive fraction * 0.5.

DatasetHeterogeneity:
  positive rows concentrated in <=1 dataset or <=1 seed stratum.

ReservoirTarget:
  high actuation / local transfer but LineC-channel source retention low.

UnknownTerminalCollapse:
  none of above explains.
```

### Codex fallback

```text
若 OptimizerWashout：执行 F2a source-conserving optimizer only。
若 DebtCollapse：执行 F2b debt-aware target filter；注意不能用 audit metrics 生成方向，只能用于 candidate rejection audit。
若 ControlDrift：提高 control standard，停止 late horizon positive route。
若 DatasetHeterogeneity：寻找 dataset-invariant train-only features，不按数据集调参。
若 ReservoirTarget：进入 function-space target reset。
```

## 7.4 Line F2：MLP source lab，解决 h3200 -> h4800 collapse

### 目标

MLP 是 functional dynamics 实验台。v22 的 h3200 continuous chain 主要来自 MLP terminal-source family。F2 要判断：MLP source 能否从 h3200 保到 h4800，并找出保留机制。

### 实验族

```text
F2-M1: MLP strong-source M2 replay
F2-M2: MLP weak-stable M15 replay
F2-M3: MLP terminal F63/F70/F74 replay
F2-M4: strong-source + weak-stable slow anchor
F2-M5: short-memory + long-memory AdEMAMix-style source writer
F2-M6: schedule-free fast/slow iterate source writer
F2-M7: matrix-block hidden-source writer
F2-M8: readout-only terminal source writer
F2-M9: terminal-collapse hold without future leakage
F2-M10: stable-random matched terminal control
```

### 记录指标

除 source/debt/controls 外，额外记录：

```text
hidden_matrix_source_projection
readout_source_projection
source_projection_decay_h3200_h4800
fast_state_norm
slow_state_norm
fast_slow_cosine
old_gradient_memory_cosine
matrix_block_update_norm
source_anchor_norm
source_anchor_drift
```

### 成功标准

MLP-FU-S3：

```text
h100/h400/h800 >= 0.005
h1600/h3200/h4800 >= 0.005
retention_h4800_over_h3200 >= 0.50
at least 5/9 dataset-seed rows positive at h4800
stable-random and random-matched controls fail
AUCtime_ratio <= 1.10
no unresolved debt collapse
```

MLP-FU-S4：

```text
h6400 >= 0.005
retention_h6400_over_h4800 >= 0.50
at least 5/9 rows positive at h6400
```

如果 MLP-FU-S3/S4 成功：

```text
写成 generic functional dynamics progress，不写 KAN-specific。
随后将 identified source writer 迁移到 KAN source-channel。
```

如果 MLP 仍 h4800 collapse：

```text
current train-stream functional source theory insufficient；暂停 KAN large FU matrix，只保留 KAN smoke。
```

## 7.5 Line F3：Precommit train-only selector

### 目标

F46 train-loss selector 在 retrospective holdout 上有价值，但它不是 precommit causal selector。F3 要建立合法 train-only selector。

### 数据划分

每个 train batch 划分：

```text
B1: generate candidate target / update
B2: train-only transfer readback
B3: train-only safety readback
B4: selector holdout train split
```

禁止读取：

```text
future horizon outcome
validation/test
LineC/tail/AUC/calibration as selector input
```

### selector 候选特征

```text
train_loss_drop_h1/h2/h4 proxy
B1->B2 transfer_gain
B1/B2 source agreement
B3 damage proxy from train loss only
per-example gradient coherence
PopRisk/SNR signal estimate
source-vs-random immediate control gap
optimizer conflict cosine
basis-channel low-degree/low-frequency energy
readout-source concentration
```

### 记录指标

```text
selector_auc_for_h4800_retention
selector_precision_at_k
selector_recall_at_k
selector_spearman_h3200
selector_spearman_h4800
selector_train_split_holdout_pass
selector_control_gap
feature_ablation
```

### 成功标准

Selector-S2：

```text
AUC_h4800 >= 0.70
precision_at_top20pct >= 0.50
holdout_train_split_pass = 1
stable-random selector AUC <= 0.55
```

Selector-S3：

```text
precommit-selected candidates in independent rerun:
  h800/h1600/h3200/h4800 continuous retention
  controls fail
```

Codex fallback：

```text
若 retrospective selector pass 但 precommit fail：标记 RetrospectiveOnly，不用于 direction。
若 train_loss feature dominates but control also selected：加入 random/stable controls，不继续用 train_loss alone。
若 PopRisk/SNR negative：重构 estimator，不让它生成方向。
```

## 7.6 Line F4：Function-space target reset

### 目标

H4/v20/v21 已证明 high ActuationR2 不等于 retained source。F4 不再问 actuation 能否高，而是问哪类 output-space target 能形成 early source chain。

### target family

```text
T1 loss-cotangent target
T2 cross-split consensus target
T3 low-degree / low-frequency filtered target
T4 readout-only target
T5 signal-channel target from train-only transfer
T6 reservoir-rejecting target
T7 noise-contrast target
T8 matrix-block hidden target
T9 random matched target
T10 sign-flipped target
T11 corrupt target
```

### 记录指标

```text
ActuationR2
projection_residual_norm
B2_transfer_gain
B3_train_safety_gain
random_target_B2_gain
signflip_B2_gain
corrupt_B2_gain
early_source_chain_rate
h3200_retention_rate
h4800_retention_rate
debt_recovery_rate
```

### 成功标准

Target-S2：

```text
ActuationR2 >= 0.50
B2_transfer_gain > random_target_B2_gain + 0.005
B3_train_safety_gain >= -0.005
early_source_chain_rate > random by >= 2x
```

Target-S3：

```text
precommit target produces h4800 retained source in independent rerun
controls fail
```

Codex fallback：

```text
若 ActuationR2 高但 source=0：target observability fail，不继续修 solver。
若 random target 同样好：target not semantic，reject family。
若 low-frequency/low-degree target 有 early chain：迁移到 KAN writer。
```

## 7.7 Line F5：KAN source-channel writer

### 目标

D-CHE / D-FOU 现在效率可行，但没有 retained source。F5 要测试 KAN 是否存在可写入 source 的 low-degree / low-frequency / readout channel。

### D-CHE writer candidates

```text
CHE-F5a basis-estimate readout-commit
CHE-F5b low-degree source bank commit
CHE-F5c degree-0/1/2-only writeback
CHE-F5d readout-only commit with degree source estimator
CHE-F5e matrix-block degree-readout writer
CHE-F5f source-conserving slow-state writer
```

### D-FOU writer candidates

```text
FOU-F5a band-estimate readout-commit
FOU-F5b low-frequency source bank commit
FOU-F5c high-frequency quarantine + lowband commit
FOU-F5d band-readout matrix-block writer
FOU-F5e source-conserving slow-state writer
```

### 记录指标

```text
basis_source_estimate_norm
readout_commit_norm
basis_commit_norm
low_degree_energy
high_degree_energy
low_frequency_energy
high_frequency_energy
source_bank_drift
readout_source_projection
basis_source_projection
source-chain metrics h100...h6400
debt metrics
controls
```

### 成功标准

KAN-FU-S2：

```text
h100/h400/h800 >= 0.005
h1600/h3200 >= 0.005
at least 4/9 rows h3200 positive
controls fail
```

KAN-FU-S3：

```text
h4800 >= 0.005
retention_h4800_over_h3200 >= 0.50
at least 5/9 rows h4800 positive
KAN_specific_delta_vs_MLP_same_mechanism >= 0.005
D-CHE/D-FOU efficiency S1 pass in same run
```

Codex fallback：

```text
若 readout-only commit works but basis commit fails：basis writeback dilutes source；继续 decoupled readout/basis.
若 low-degree/low-frequency works：禁用 high-degree/high-frequency source commit，只作为 reservoir。
若 all KAN writers fail while MLP succeeds：write CarrierMismatch certificate。
```

## 7.8 Line F6：D-RAT / D-RBF limited functional smoke

只有 Line E-RAT / E-RBF 达到 Near-E1 后执行。

### D-RAT smoke

```text
RAT-F6a denominator-safe readout commit
RAT-F6b numerator-only source commit
RAT-F6c denominator-slope safety boundary + readout commit
RAT-F6d rational block matrix writer
```

### D-RBF smoke

```text
RBF-F6a local-center source estimator + readout commit
RBF-F6b compact-local K source writer
RBF-F6c width-stable local support writer
```

### 成功标准

```text
h800 source >= 0.005 for at least 3/9 rows
no denominator/width safety fail
controls fail at smoke level
```

Smoke 成功不允许 promotion，只允许进入下一轮 full carrier plan。

---

# 8. Controls / Attribution

每个 positive-looking result 必须有：

```text
NoOpMatchedOverhead
AdamW baseline
SGD baseline
Momentum baseline
RandomMatchedNorm
StableRandomSource
RandomTargetSameActuationR2
RecoveryOnly
SameScheduleNoFU
MLP same mechanism
D-CHE / D-FOU same mechanism if applicable
```

记录：

```text
source_vs_best_control
control_equivalent_fraction
random_late_positive_fraction
stable_random_h4800_fraction
MLP_generic_explains
KAN_specific_delta
```

如果 control positive 与 candidate 同阶，route 必须写：

```text
ControlEquivalentLateDrift
```

不能写 functional progress。

---

# 9. 4GPU 动态执行计划

v22.01 必须充分使用四张 GPU。默认队列：

```text
GPU0:
  S0.8 truth gate
  D-CHE efficiency officialization
  D-CHE source-channel writer

GPU1:
  D-FOU efficiency officialization
  D-FOU source-channel writer

GPU2:
  MLP source lab
  terminal-collapse autopsy
  precommit selector

GPU3:
  D-RAT repair
  D-RBF repair
  D-RAT/D-RBF limited smoke
  controls / figures / packet generation fallback
```

动态补位规则：

```text
若 GPU0 primary 完成：接 D-CHE h4800/h6400 extension 或 controls。
若 GPU1 primary 完成：接 D-FOU h4800/h6400 extension 或 controls。
若 GPU2 primary 完成：接 selector independent rerun 或 terminal-collapse figures。
若 GPU3 primary 完成：接 LQ/D-WAV smoke 或 missing efficiency rows。
```

必须生成：

```text
v22_01_runnable_queue.csv
v22_01_gpu_assignment_manifest.csv
v22_01_gpu_utilization_dashboard.csv
v22_01_idle_violation.csv
v22_01_deferred_items.csv
v22_01_queue_drain_report.csv
```

---

# 10. 统一指标定义

## 10.1 Source metrics

```text
source_h100
source_h400
source_h800
source_h1200
source_h1600
source_h2400
source_h3200
source_h4000
source_h4800
source_h6400
source_vs_best_control_h*
source_derivative_h*_to_h*
```

Retention：

$$
Retention_{b/a}=
\frac{\max(0,S_b)}{\max(\epsilon,S_a)}.
$$

Early chain：

$$
S_{100}, S_{400}, S_{800} \ge 0.005.
$$

Continuous h3200：

$$
S_{100}, S_{400}, S_{800}, S_{1600}, S_{3200} \ge 0.005.
$$

Continuous h4800：

$$
S_{100}, S_{400}, S_{800}, S_{1600}, S_{3200}, S_{4800} \ge 0.005.
$$

## 10.2 Debt metrics

```text
tail_CEp99_debt_peak / final / recovery
tail_NLL_debt_peak / final / recovery
ECE_debt_peak / final / recovery
Brier_debt_peak / final / recovery
LineC_fast_debt_peak / final / recovery
LineC_channel_debt_peak / final / recovery
AUCtime_ratio_vs_best_control
```

Recovery：

$$
Recovery(H)=1-\frac{Debt_H}{Debt_{peak}+\epsilon}.
$$

## 10.3 Efficiency metrics

```text
forward_only_ms
backward_grad_ms
optimizer_update_ms
functional_direction_ms
functional_commit_ms
linec_audit_ms
horizon_readback_ms
training_step_ms
forward_peak_memory
backward_peak_memory
optimizer_state_memory
basis_activation_bytes
functional_state_bytes
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
```

## 10.4 Kernel metrics

```text
official_fused_kernel_complete
no_materialize_complete
gradcheck_pass
grad_relerr
grad_cosine
fallback_kernel_used
kernel_count
basis_materialized_bytes
component_time_waterfall
```

## 10.5 Selector metrics

```text
selector_auc_h4800
selector_precision_at_k
selector_recall_at_k
selector_spearman_h3200
selector_spearman_h4800
precommit_selector_pass
retrospective_only_flag
```

---

# 11. 成功标准

## S0.8：代码 / 指标 / artifact pass

```text
source packet self-contained
compileall ok
core import error count = 0
source-chain unit tests pass
LineC fast/channel golden pass
retention/debt formula tests pass
official kernel status consistency pass
4GPU queue artifacts complete
```

## E1：Efficiency exploration pass

```text
forward_ratio_vs_mlp <= 1.25
step_ratio_vs_mlp <= 1.25
memory_ratio_vs_mlp <= 1.10
gradcheck_pass = 1
fallback_kernel_used = 0
```

## E2：D-RAT / D-RBF near-efficiency pass

```text
forward_ratio_vs_mlp <= 3.0
step_ratio_vs_mlp <= 2.0
memory_ratio_vs_mlp <= 1.20
gradcheck_pass = 1
safety gate pass
```

## F-S2：Functional early-source pass

```text
carrier x mechanism 9-row mean:
  h100/h400/h800 >= 0.005
  at least 4/9 rows h800 positive
  controls fail
  no unresolved debt explosion
```

## F-S3：Functional h3200 continuous pass

```text
h100/h400/h800/h1600/h3200 >= 0.005
retention_h3200_over_h800 >= 0.40
at least 5/9 rows h3200 positive
controls fail
```

## F-S4：Functional h4800 retained pass

```text
h100/h400/h800/h1600/h3200/h4800 >= 0.005
retention_h4800_over_h3200 >= 0.50
at least 5/9 rows h4800 positive
stable-random and random-matched controls fail
tail / LineC / ECE / Brier debt recovered >= 0.60
AUCtime_ratio <= 1.05
```

## S5：official success，不降低

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
controls fail
promotion_allowed = 1
```

---

# 12. 必须可视化

v22.01 必须生成以下图表：

```text
1. source_trajectory_by_candidate.svg
   展示 h100...h6400 source curves，标记 terminal collapse。

2. terminal_collapse_waterfall.svg
   h3200 -> h4800 source drop 与 optimizer projection / debt spike 对齐。

3. debt_recovery_heatmap.svg
   tail / LineC / ECE / Brier / AUCtime debt by carrier x mechanism。

4. selector_roc_pr.svg
   train-only selector 对 h4800 retained source 的 ROC / PR。

5. target_contrast_actuation_vs_retention.svg
   ActuationR2 vs h4800 retained source；区分 semantic / random / signflip / corrupt target。

6. efficiency_component_waterfall_DCHE_DFOU_DRAT_DRBF.svg
   forward / backward / update / audit / horizon component。

7. kernel_status_consistency_dashboard.svg
   truth table vs gradcheck vs official status。

8. carrier_source_channel_map.svg
   MLP hidden/readout, D-CHE degree/readout, D-FOU band/readout source projection。

9. gpu_utilization_dashboard.svg
   四卡利用率、idle violation、queue drain。
```

---

# 13. 失败后的执行方向：Codex 不能停在 no-go

## Case A：S0.8 不通过

不允许跑 scientific route。Codex 必须先修：

```text
缺源文件 -> 补 source tree；
source-chain unit test fail -> 修 early/retention rules；
official kernel status mismatch -> 统一 truth/gradcheck/status；
LineC missing -> 补 metrics source；
queue artifacts missing -> 补动态队列。
```

## Case B：D-CHE / D-FOU profiler pass 但 full-loop fail

Codex 必须拆：

```text
kernel fallback；
functional direction overhead；
audit/horizon readback contamination；
batch512 tiling；
MLP reference mismatch。
```

不能直接写 efficiency fail。

## Case C：D-RAT / D-RBF 仍 forward blocked

Codex 必须输出 component waterfall，并至少尝试：

```text
D-RAT: telemetry separation + branchless denominator；
D-RBF: compact local K + no dense materialization。
```

若仍失败，写：

```text
D-RAT/D-RBF RejectedForThisVersion with explicit component blocker。
```

## Case D：h3200 candidates 仍全部 h4800 collapse

Codex 必须执行 terminal-collapse taxonomy，不能直接换 target：

```text
OptimizerWashout -> source-conserving optimizer tests；
DebtCollapse -> target rejection filter；
ControlDrift -> abandon late-positive metric；
DatasetHeterogeneity -> train-only invariant selector；
ReservoirTarget -> function-space target reset。
```

## Case E：MLP 成功但 KAN 失败

写：

```text
GenericFUWorks_KANCarrierMismatch
```

然后只做 KAN source-channel writer，不再扩大 MLP method surface。

## Case F：KAN low-degree / low-frequency writer 成功但 readout/basis混合失败

保留 decoupled writer：

```text
basis-estimate / readout-commit
low-degree / low-frequency source bank
high-degree / high-frequency reservoir only
```

不要强制 full-basis commit。

---

# 14. 本轮结束时必须交付的结论

v22.01 最终复盘必须固定三段式：

## Part A：代码审计

必须回答：

```text
代码包是否自包含？
是否缺文件？
source-chain bug 是否修正？
LineC / debt / retention 是否正确？
official kernel status 是否一致？
4GPU queue 是否闭合？
```

## Part B：基函数效率

必须回答：

```text
D-CHE / D-FOU 是否 full-loop officialize？
D-RAT / D-RBF 是否达到 near-E1？
每个 basis 卡在 forward/backward/update/memory/audit 哪一项？
D-RAT / D-RBF 是否有 active repair 结果，而不是空缺？
```

## Part C：Functional update

必须回答：

```text
修正 source-chain rule 后，v22 的 h3200 candidates 是否仍成立？
h3200 -> h4800 terminal collapse 的主因是什么？
MLP source 是否能 h4800 retained？
train-only selector 是否能 precommit 选择 retained source？
function-space target 是否能从 actuation 变成 retention？
KAN low-degree / low-frequency / readout source writer 是否打开？
D-RAT / D-RBF 是否有 limited source smoke？
```

---

# 15. 最终判断

v22.01 的核心不是继续把 late rebound 做大，而是停止用 late rebound 逃避问题。当前最清楚的事实是：

$$
\boxed{
\text{D-CHE / D-FOU 的效率路线已经有真实希望，}
\text{但 functional route 仍缺 h4800 retained source。}
}
$$

因此本轮必须确实推进两条线：

```text
效率线：
  D-CHE / D-FOU full-loop officialization；
  D-RAT / D-RBF active repair。

Functional 线：
  修 source-chain 指标；
  诊断 terminal collapse；
  建 precommit selector；
  重设 function-space target；
  构建 KAN source-channel writer。
```

如果这轮结束仍只有：

```text
D-CHE / D-FOU efficiency pass；
h3200 positive；
h4800 fail；
D-RAT / D-RBF 没跑；
code packet 不闭包；
```

那就不是科学失败，而是执行失败。

