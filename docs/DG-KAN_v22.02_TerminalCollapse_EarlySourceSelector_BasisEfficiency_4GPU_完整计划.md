# DG-KAN v22.02：Terminal-Collapse 机制裁决 + Early-Source Precommit Selector + D-CHE/D-FOU Officialization + D-RAT/D-RBF Active Repair + 4GPU 动态并行完整计划

> 版本：v22.02 execution plan  
> 生成时间：2026-06-04  
> 当前主结论：v22.01 并非完全原地打转，但仍未达成能力突破。效率线已经把 D-CHE / D-FOU 推入 MLP-like envelope；functional update 线从“h3200 打不开”推进到“h3200 continuous source 有 12 个 group”，但所有 group 在 h4800 terminal collapse，KAN 仍没有 retained source。v22.02 的目标不是继续放大 late rebound，而是明确裁决 h3200→h4800 为什么坍塌，并构造合法的 train-only precommit selector 与 KAN source-channel writer。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no validation / test / future / query 生成 functional direction；LineC / CEp99 / NLL / ECE / Brier / AUCtime 只能作为 audit / debt readback / final gate，不能作为方向源；不启动 action bank / controller / reset route；不把 single-row positive、late rebound、diagnostic oracle、smoke 或 efficiency-only 写成 promotion。

---

# 0. 项目总目标与 v22.02 当前定位

DG-KAN 的总目标仍然是：

$$
\boxed{
\text{在 strict FC-PureKAN base 上，通过 functional update 改善训练动力学，}
\text{最终得到比同参数量 MLP / PureKAN-AdamW 更好的系统。}
}
$$

这里“更好”不是某个 horizon 的 source 为正，也不是某个 basis 在 profiler 里快，而是同时满足：

```text
1. strict FC-PureKAN，无 non-KAN 可学习参数支撑；
2. basis forward / backward / update / memory 接近同参数量 MLP；
3. functional update 的收益超过 AdamW、SGD/Momentum、NoOp、RandomMatched、StableRandom、same-overhead、same-target、same-actuation controls；
4. source 必须形成 early chain，并连续留存到 h3200/h4800，而不是 h3200 late rebound；
5. tail / LineC / ECE / Brier / AUC debt 必须可偿还；
6. 若只有 MLP 成功，写成 generic training-dynamics insight；只有 KAN 同机制显著超过 MLP，才可写 KAN-specific functional advantage；
7. 任何 positive-looking result 都必须通过 independent rerun，不允许用 route single-row max 或 retrospective selector claim。
```

v22.01 的事实边界是：

```text
Code:
  compileall 语法通过，但 code packet 非自包含；LineC / profiler / kernel / FU core 依赖缺失；source-chain zero-control early bug 已修。

Efficiency:
  D-CHE / D-FOU 继续有 S1 official-like rows；
  D-RAT / D-RBF 被主动 repair，但 forward 仍约 9-10x，near-E1=0。

Functional:
  MLP terminal family 可形成 h100/h400/h800/h1600/h3200 continuous source；
  h4800 全部坍塌；
  KAN 没有 early continuous retained source；
  late rebound / delayed migration 仍不能 promotion；
  precommit selector 未打开，当前 selector 多为 retrospective。
```

因此 v22.02 不再问：

```text
哪个 Fxx 还能把 h3200 变得更高？
```

而是问：

$$
\boxed{
\text{为什么 h3200 continuous source 到 h4800 系统性坍塌？}
}
$$

以及：

$$
\boxed{
\text{能否在不看 future outcome 的情况下，提前选择会连续留存到 h4800 的 source target？}
}
$$

并且效率线必须继续回答：

$$
\boxed{
\text{D-CHE / D-FOU 的 MLP-like timing 能否进入完整 functional training loop，}
\text{D-RAT / D-RBF 能否从 ForwardBlocked 推到 near-E1？}
}
$$

---

# 1. v22.02 必须解决的核心问题

## 1.1 问题 A：代码审计必须变成“打包后自包含审计”

v22.01 的 code packet 能编译，但独立 import 失败。根因不是实验逻辑，而是打包审计逻辑：`required_source_files.csv` 可能在原始 repo 环境中生成，而不是在最终 zip 解压目录中验证。v22.02 必须把 code audit 改成：

```text
先打包；
再解压到 clean tmp dir；
只使用解压目录执行 compileall/import/tests；
最终 route 只能读取 clean tmp dir 的审计结果。
```

如果不修这个，后续任何 LineC、kernel、functional mechanism 的 correctness 都不能完全信。

## 1.2 问题 B：D-CHE / D-FOU efficiency 已经不是“有没有希望”，而是 officialization closure

D-CHE / D-FOU 的效率路线现在已经从 `ForwardBlocked` 进入 MLP-like envelope。v22.02 不该继续大规模 census，而应把它们推到 full-loop closure：同一个 kernel 同时用于 profiler、full training loop、functional update runner、LineC/horizon readback separated path。

但 D-RAT / D-RBF 不能再躺在 blocked 表里。它们必须进入 active repair：D-RAT 拆 numerator / denominator / reciprocal / telemetry；D-RBF 拆 dense materialization / active-center / local-K / compact support / fused local backward。D-RAT / D-RBF 先追 near-E1，不抢 full FU 预算。

## 1.3 问题 C：functional update 的 blocker 已经从 h800/h3200 转成 h4800 terminal collapse

v22.01 的核心事实是：MLP 上 h100-h3200 能连续 source，但 h4800 坍塌。KAN 上仍没有 early continuous source。v22.02 必须把 functional update 从“找更多 source family”转成“终端坍塌机制裁决”。

对每个 h3200 continuous group，要问：

```text
h3200->h4800 是 optimizer washout？
是 tail / calibration / LineC debt 延迟爆发？
是 dataset/seed heterogeneity？
是 stable-random / trajectory drift？
是 target 只在中期有效，无法进入 signal channel？
是 source 写在 MLP hidden/readout，而 KAN low-degree/low-frequency bank 不对应？
```

## 1.4 问题 D：precommit selector 仍未打开

v22.01 显示 train-loss h400/h800 可 retrospective 地选择好 trajectory，但不是合法 precommit selector。v22.02 必须把 selector 实验从 retrospective audit 改成真正 precommit protocol：只用提交前可见的 train-stream 统计量，先选择 candidate，再 fresh rerun。

## 1.5 问题 E：以前被判错的方向哪些可重开，哪些不能重开

### 仍可重开，但必须换语义

```text
PopRisk / SNR：
  过去常作为 one-shot parameter SNR 或 retrospective predictor，当前不能直接用。
  v22.02 只允许把它作为 train-only source-chain precommit feature，必须 fresh rerun 验证。

Split-consensus / G7R lineage：
  过去 source/hazard 重叠、trust scalar、小 nullspace 都失败。
  但在 source-chain bug、retention/debt 口径修正后，可以重开为 source estimator，不直接提交原始 update。

Function-space actuation：
  过去高 ActuationR2 但无 retained source。
  这说明 actuation 不是瓶颈，target 定义才是瓶颈；可以重开 target definition，不再追 ActuationR2 本身。

MLP-FU：
  不能丢。它是 generic source dynamics 实验台。现在 MLP 已能 h3200 continuous，正适合做 terminal-collapse autopsy。

Graph-free analytic adjoint / operator-level basis channel：
  早期失败很多来自 efficiency/kernels 不成熟；D-CHE/DFOU kernel 现在改善后，可用更高效 carrier 重做有限 smoke。
```

### 不应重启主预算

```text
Action bank / controller / reset route：
  过去已经多次证明 oracle upper-bound 或 generic optimizer reset confound，不应回去找“好动作”。

Cover objective 小修：
  oracle cover objective 已被判 invalid，不能再把 cover_purity / churn 当核心 objective。

M31/M32 同族修补：
  H10 partial positive 未被 H11/H13 复现，H12 negative，不应继续调 alt period / FU lr / threshold。

G/N/Q-token 扩展：
  过去 source/hazard 静态分不开，继续扩 token 只会重复。

F57-F86 terminal family 小修：
  它们已经把 h3200 打开但 h4800 坍塌。继续调 floor / lookahead / hold / tiny fallback 很可能只重复 terminal collapse。
```

---

# 2. 当前研究进展给 v22.02 的启发

v22.02 不机械“换 optimizer”，而是抽取训练动力学原则。

## 2.1 Signal channel / reservoir：source 必须进入长期可见通道

最新 generalization 理论把输出空间分成 signal channel 与 reservoir。coherent population signal 会通过 drift 积累并 transfer；噪声被困在 reservoir 则对 test 不可见。对应到我们当前结果：h3200 source 不等于 signal-channel source，h4800 terminal collapse 说明它还没有稳定进入长期 signal channel。

因此 v22.02 不再用单个 horizon 的 source 判断成功，而是必须记录：

$$
S_{100}, S_{400}, S_{800}, S_{1600}, S_{2400}, S_{3200}, S_{4000}, S_{4800}
$$

以及 source derivative：

$$
D_{a\to b}=S_b-S_a.
$$

特别关注：

$$
D_{3200\to4800}<0
$$

是否由 optimizer projection、debt explosion 或 target misalignment 解释。

## 2.2 Boundary-conditioned iteration：functional update 应该改变轨迹，不是单步补丁

Deep Manifold 视角提示：训练不是固定坐标下的单步安全移动，而是 boundary-conditioned iteration。functional update 更像改变训练边界条件，使模型进入不同 fixed-point region。v22.02 的 functional 成败因此看 trajectory chain，而不是 one-step source。

## 2.3 Schedule-Free / SF-NorMuon：fast/slow iterate 与长期稳定性

Schedule-Free 将 learning-rate schedule 与 iterate averaging 统一；SF-NorMuon 的近期结果进一步强调 fast iterate 上的 weight decay 对 long-horizon stability 关键。v22.02 的启发不是“换成 SF-NorMuon”，而是要记录并测试：

```text
fast iterate source；
slow/averaged iterate source；
fast-to-slow source transfer；
terminal h3200->h4800 collapse 是否发生在 fast state、slow state 还是二者偏离时。
```

## 2.4 AdEMAMix：短记忆与旧梯度都可能重要

AdEMAMix 指出单一 EMA 不能同时保留近期梯度与旧梯度，使用两条 EMA 可能更好。v22.02 用它来设计 terminal source retention：不是只保存一个 momentum，而是显式记录短记忆 source 与长记忆 source。

$$
short_t=\beta_s short_{t-1}+(1-\beta_s)u_t
$$

$$
long_t=\beta_l long_{t-1}+(1-\beta_l)u_t
$$

然后比较：

```text
short-only commit；
long-only commit；
short+long agreement commit；
short/long disagreement reject；
terminal h4800 retention。
```

## 2.5 SOAP / Muon / spectral update：matrix/block coordinate 可能比逐参数 coordinate 更合适

SOAP 把 Adam 放在 preconditioner eigenbasis 中；Muon 类方法强调矩阵更新的谱结构。对我们来说，启发是：MLP 的 source 可能存在于 hidden/readout matrix block，而 KAN 的 degree/frequency/readout bank 如果逐参数提交，会把 source 变成 reservoir 或 delayed migration。v22.02 因此必须做 MLP source subspace decomposition 与 KAN low-degree / low-frequency bank mapping。

## 2.6 Cautious optimizer：只作为 conflict diagnostic，不作为 hard mask

Cautious optimizer 的启发是 proposed update 与 gradient/momentum 冲突可能导致训练不稳。但我们不能简单“相反就不更新”，因为 productive perturbation 可能短期反向。v22.02 只把 alignment 用作诊断：

```text
FU vs gradient；
FU vs AdamW cumulative update；
FU vs momentum；
FU vs source direction；
terminal collapse 与这些 cosine 的关系。
```

---

# 3. v22.02 总体实验结构

v22.02 分成三大部分，每部分都有硬交付。

```text
Part A: Code / metric / artifact closure
  修自包含打包、source-chain、LineC/debt、route、kernel status、4GPU queue。

Part B: Basis efficiency
  D-CHE / D-FOU full-loop official closure；
  D-RAT / D-RBF active repair；
  LQ / D-WAV low-budget smoke。

Part C: Functional update
  terminal-collapse autopsy；
  online precommit selector；
  source-channel target reset；
  KAN source bank writer；
  MLP-to-KAN source mapping；
  controls and independent rerun。
```

---

# 4. Part A：代码审计与指标正确性硬门 S0.9

## 4.1 目标

S0.9 不是形式检查。它要证明 v22.02 的结果包可以由用户拿到后独立复核。所有 source/debt/efficiency/FU 结论必须来自最终 zip 解压目录，而不是 Codex 原始 repo 环境。

## 4.2 Codex 必须打包的文件

Codex 必须生成两个包：

```text
v22_02_code_review_packet.zip
v22_02_results_bundle.zip
```

### 4.2.1 code_review_packet 必须包含

```text
00_README.md
01_ENVIRONMENT/
  python_version.txt
  pip_freeze.txt
  torch_cuda_info.txt
  gpu_info.txt
  git_status.txt
  git_diff.patch

02_SOURCE_TREE/
  dgkan/fu/core.py
  dgkan/fu/source_chain.py
  dgkan/fu/debt_accounting.py
  dgkan/fu/mechanisms.py
  dgkan/fu/source_channel.py
  dgkan/fu/source_state.py
  dgkan/fu/function_space_actuation.py
  dgkan/fu/matrix_block.py
  dgkan/fu/poprisk_snr.py
  dgkan/fu/poprisk_source.py
  dgkan/fu/terminal_collapse.py
  dgkan/fu/precommit_selector.py
  dgkan/fu/kan_source_bank.py

  dgkan/metrics/linec.py
  dgkan/metrics/calibration.py

  dgkan/kernels/fused_chebyshev_k3.py
  dgkan/kernels/fused_fourier_k2.py
  dgkan/kernels/fused_rational.py
  dgkan/kernels/fused_rbf_local.py
  dgkan/kernels/v17_basis.py
  dgkan/kernels/cheby_fused.py
  dgkan/kernels/fourier_fused.py
  dgkan/kernels/rational_fused.py
  dgkan/kernels/rbf_sparse.py

  dgkan/profiling/efficiency_v17.py
  dgkan/profiling/efficiency_v20.py
  dgkan/profiling/efficiency_v21.py
  dgkan/profiling/efficiency_v22.py
  dgkan/profiling/efficiency_v22_01.py
  dgkan/profiling/efficiency_v22_02.py

  experiments/run_v17_common.py
  experiments/run_v21_efficiency_officialization.py
  experiments/run_v21_01_common.py
  experiments/run_v21_01_source_retention.py
  experiments/run_v22_01_common.py
  experiments/run_v22_02_s09_truth_gate.py
  experiments/run_v22_02_efficiency_officialization.py
  experiments/run_v22_02_terminal_collapse_autopsy.py
  experiments/run_v22_02_precommit_selector.py
  experiments/run_v22_02_source_channel_target_reset.py
  experiments/run_v22_02_kan_source_writer.py
  experiments/run_v22_02_drat_drbf_active_repair.py
  experiments/run_v22_02_merge_finalize.py

03_IMPORT_CLOSURE/
04_LINEC_CORRECTNESS/
05_SOURCE_CHAIN_AND_DEBT/
06_UPDATE_SEMANTICS/
07_FUNCTIONAL_MECHANISM_CONTRACTS/
08_EFFICIENCY_KERNELS/
09_PROFILER_CORRECTNESS/
10_GPU_QUEUE/
11_REPRO_COMMANDS/
packet_manifest.csv
packet_sha256_manifest.csv
```

### 4.2.2 results_bundle 必须包含

```text
v22_02_route_decision.json
v22_02_required_artifact_manifest.csv
v22_02_failure_taxonomy.csv
v22_02_no_go_boundary.md
v22_02_next_hypothesis_queue.md
v22_02_code_audit_summary.csv
v22_02_efficiency_truth_table.csv
v22_02_efficiency_full_loop_table.csv
v22_02_kernel_gradcheck.csv
v22_02_official_fused_status_matrix.csv
v22_02_drat_repair_table.csv
v22_02_drbf_repair_table.csv
v22_02_source_retention_matrix.csv
v22_02_terminal_collapse_autopsy.csv
v22_02_precommit_selector_matrix.csv
v22_02_source_channel_target_matrix.csv
v22_02_kan_source_bank_matrix.csv
v22_02_controls_matrix.csv
v22_02_independent_rerun_matrix.csv
figures/
logs/
```

## 4.3 Clean unzip self-test

Codex 必须在打包后执行：

```bash
rm -rf /tmp/v22_02_packet_check
mkdir -p /tmp/v22_02_packet_check
unzip v22_02_code_review_packet.zip -d /tmp/v22_02_packet_check
cd /tmp/v22_02_packet_check/02_SOURCE_TREE
python -m compileall -q .
python experiments/run_v22_02_s09_truth_gate.py --self-contained-import-check 1 --source-root .
```

必须输出：

```text
v22_02_packet_clean_unzip_compileall.csv
v22_02_packet_clean_unzip_import_closure.csv
v22_02_packet_manifest_vs_required_source_files.csv
```

## 4.4 S0.9 gate

S0.9 通过条件：

```text
compileall_pass = 1
import_error_count = 0
required_source_missing_count = 0
packet_manifest_mismatch_count = 0
linec_fast_golden_pass = 9/9
linec_channel_golden_pass = 8/8
source_chain_unit_tests_pass = 1
debt_accounting_unit_tests_pass = 1
route_aggregation_unit_tests_pass = 1
update_semantics_tests_pass = 1
kernel_gradcheck_pass = 1
profiler_phase_isolation_pass = 1
```

若不满足：

```text
不能跑 scientific route；
Codex 必须先修包和 tests；
但允许继续生成 failure taxonomy 和 repair queue。
```

---

# 5. Part B：基函数效率路线

## 5.1 总目标

v22.02 的效率路线必须同时推进四类 basis：

```text
Primary officialization:
  D-CHE, D-FOU

Active repair:
  D-RAT, D-RBF

Low-budget smoke:
  LQ, D-WAV
```

目标不是“有一个 basis 快就停”，而是形成明确状态表：

```text
OfficialEfficientCarrier
ExplorationEfficientCarrier
NearEfficientCarrier
ForwardKernelBlocked
BackwardKernelBlocked
StepKernelBlocked
SafetyBlocked
MaterializationBlocked
RejectedForThisVersion
```

## 5.2 D-CHE / D-FOU：full-loop official closure

### 假设 E-H1

D-CHE / D-FOU 的 current S1 rows 不是 profiler artifact，而是真实 full functional runner 可用的 efficient carrier。

### 实验

D-CHE 候选：

```text
CHE22-R2-low-degree-k3-official
CHE22-R4-k3-gradbuf-triton-official
CHE22-R4-k5-gradbuf-triton-ablation
```

D-FOU 候选：

```text
FOU22-R2-lowfreq-k2-stream-official
FOU22-R3-tablelookup-bandreadout-official
FOU22-R4-k4-triton-no-materialize-official
```

每个候选执行：

```text
batch = 8, 32, 128, 256, 512
profiler-only timing
full training loop timing
full functional runner timing
LineC/horizon audit separated timing
kernel gradcheck
no-materialize audit
same-param MLP reference
```

### 记录指标

```text
carrier
variant
batch
forward_only_ms
backward_grad_ms
optimizer_update_ms
functional_direction_ms
functional_commit_ms
linec_audit_ms
horizon_readback_ms
training_step_ms
full_loop_step_ms
forward_ratio_vs_mlp
backward_ratio_vs_mlp
update_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
forward_peak_memory
backward_peak_memory
basis_activation_bytes
workspace_temp_bytes
official_fused_kernel_complete
no_materialize_complete
fallback_kernel_used
kernel_gradcheck_pass
functional_runner_uses_same_kernel
```

### 判断标准

D-CHE S1 closure：

```text
>= 2 variants × >= 3 batch sizes pass
forward_ratio_vs_mlp <= 1.25
step_ratio_vs_mlp <= 1.25
memory_ratio_vs_mlp <= 1.05
functional_runner_uses_same_kernel = 1
fallback_kernel_used = 0
official_fused_kernel_complete = 1
kernel_gradcheck_pass = 1
```

D-FOU S1 closure：

```text
>= 1 variant × >= 3 batch sizes pass
forward_ratio_vs_mlp <= 1.25
step_ratio_vs_mlp <= 1.25
memory_ratio_vs_mlp <= 1.05
functional_runner_uses_same_kernel = 1
fallback_kernel_used = 0
official_fused_kernel_complete = 1
kernel_gradcheck_pass = 1
```

### 不满足时 Codex 先尝试

```text
If source packet missing kernel:
  fix packaging and rerun S0.9 only.

If profiler pass but full functional loop fail:
  split full-loop timing into training / functional_direction / audit / horizon_readback.
  If audit dominates, move audit to separate readback path.
  If functional direction dominates, cache source-state projection or reduce per-row recompute.

If batch512 fails but batch128/256 pass:
  mark batch512 ForwardFullLoopBlocked and keep S1 if >=3 batch sizes pass.
  Then try chunked no-materialize batch512 only, not full rerun.

If D-FOU R3 pass but R2/R4 fail:
  freeze R3 as efficiency anchor; do not keep tuning R2/R4 unless R3 lacks gradcheck.
```

## 5.3 D-RAT active repair

### 假设 E-H2

D-RAT 当前失败是 rational forward evaluation / denominator path 的 implementation blocker，而不是 Rational basis 本身不可用。

### 实验

候选：

```text
RAT22-R0-current-reference
RAT22-R1-horner-numerator-denominator
RAT22-R2-branchless-denominator-safety
RAT22-R3-reciprocal-approx-diagnostic
RAT22-R4-telemetry-free-train-path
RAT22-R5-fused-num-den-backward
```

每个候选执行 batch 8 / 32 / 128。只做 efficiency repair，不做 full FU。

### 记录指标

```text
num_eval_ms
den_eval_ms
reciprocal_ms
rational_forward_ms
rational_backward_ms
denominator_min
denominator_p01
denominator_condition
derivative_telemetry_ms
telemetry_on_train_path
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
gradcheck_pass
safety_pass
```

### near-E1 gate

```text
forward_ratio_vs_mlp <= 3.0
step_ratio_vs_mlp <= 2.0
memory_ratio_vs_mlp <= 1.2
gradcheck_pass = 1
safety_pass = 1
```

### 不满足时 Codex 先尝试

```text
If denominator safety fails:
  do not run functional smoke; implement denominator clamp / branchless floor diagnostic and rerun gradcheck.

If forward ratio > 3.0 but denominator eval dominates:
  implement Horner + reciprocal fuse.

If telemetry dominates:
  split telemetry-free train path and audit-only telemetry path.

If gradcheck fails:
  fix analytic derivative before any timing claim.
```

## 5.4 D-RBF active repair

### 假设 E-H3

D-RBF 失败是 dense basis materialization 与 Gaussian exp path 没被 local-support 化，而不是 RBF local support 概念无效。

### 实验

候选：

```text
RBF22-R0-current-reference
RBF22-R1-active-center-mask
RBF22-R2-compact-local-k4-no-dense
RBF22-R3-local-gather-fused-forward
RBF22-R4-local-fused-backward
RBF22-R5-exp-table-approx-diagnostic
```

每个候选 batch 8 / 32 / 128。

### 记录指标

```text
active_center_fraction
avg_local_k
max_local_k
basis_materialized_bytes
local_gather_ms
gaussian_exp_ms
rbf_forward_ms
rbf_backward_ms
center_grad_ms
width_grad_ms
width_condition
out_of_support_fraction
forward_ratio_vs_mlp
backward_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
gradcheck_pass
```

### near-E1 gate

```text
forward_ratio_vs_mlp <= 3.0
step_ratio_vs_mlp <= 2.0
memory_ratio_vs_mlp <= 1.2
gradcheck_pass = 1
basis_materialized_bytes <= 0.25 * dense_reference_bytes
```

### 不满足时 Codex 先尝试

```text
If active_center_fraction too high:
  tighten compact support or width condition; rerun only RBF active-center rows.

If exp dominates:
  compare exp-table diagnostic vs exact exp; diagnostic cannot be official unless gradcheck acceptable.

If dense materialization remains:
  implement no-dense local K path before any functional smoke.
```

## 5.5 LQ / D-WAV low-budget smoke

LQ / D-WAV 不抢主 GPU，但必须给状态更新：

```text
LQ:
  recurrence + fused projection update smoke。

D-WAV:
  sparse support index-only + local backward smoke。
```

只要求记录：forward / backward / step / memory / blocker。除非 near-E1 打开，否则不进入 functional matrix。

---

# 6. Part C：Functional update 主线

## 6.1 Functional 总目标

v22.02 的 FU 目标是：

$$
\boxed{
\text{找到合法 train-stream source target，使 source 从 h100/h400/h800 开始形成，}
\text{连续保留到 h3200/h4800，并打过 controls。}
}
$$

不再把下面这些当成功：

```text
single-row max；
h800-only source；
h3200 positive but h4800 negative；
late rebound with negative h800；
retrospective selector；
high ActuationR2 without retained source；
MLP generic positive claimed as KAN-specific。
```

## 6.2 统一 source-chain 定义

对一个 carrier / mechanism / dataset-seed group，定义：

$$
S_H = \text{source\_vs\_best\_control at horizon } H
$$

Early source chain：

$$
S_{100}\ge \epsilon,
\quad
S_{400}\ge \epsilon,
\quad
S_{800}\ge \epsilon,
\quad
\epsilon=0.005.
$$

Continuous h3200：

$$
S_{100},S_{400},S_{800},S_{1600},S_{2400},S_{3200}\ge \epsilon
$$

and retention ratios:

$$
R_{1600/800}=\frac{\max(0,S_{1600})}{\max(\epsilon,S_{800})},
$$

$$
R_{3200/1600}=\frac{\max(0,S_{3200})}{\max(\epsilon,S_{1600})}.
$$

Productive h4800：

$$
S_{4800}\ge \epsilon,
\quad
R_{4800/3200}=\frac{\max(0,S_{4800})}{\max(\epsilon,S_{3200})}\ge 0.50.
$$

Terminal collapse：

$$
\text{TC}=1
\quad \text{if} \quad
S_{3200}\ge\epsilon \quad \text{and} \quad S_{4800}<\epsilon.
$$

Controls cannot count as source chain. `control_equivalent=1` forces all source-chain gates to 0.

## 6.3 FU-H1：terminal-collapse autopsy

### 假设

h3200→h4800 collapse 是由可测的训练动力学事件导致，而不是随机噪声。可能原因：

```text
OptimizerWashout；
DebtExplosion；
DatasetHeterogeneity；
ControlEquivalentLateDrift；
TargetMisalignment；
SourceReservoirMigration；
UnknownTerminalCollapse。
```

### 实验对象

取 v22/v22.01 所有 h3200 continuous groups，尤其：

```text
MLP-F53 / F63 / F70 / F74 / F77 / F78 / F84-F86 families；
any KAN delayed migration groups；
matched stable-random h3200/h4800 positive controls。
```

### 执行

对每个 group fresh rerun，记录 horizon：

```text
h100, h400, h800, h1600, h2400, h3200, h4000, h4800
```

并在 h2400/h3200/h4000/h4800 保存 checkpoint-level state readback。

### 记录指标

```text
source_h100 ... source_h4800
source_derivative_h100_h400 ... h3200_h4800
source_pass_count_by_dataset_seed
cumulative_optimizer_projection_on_source_h3200_to_h4800
adamw_m_cosine_with_source
adamw_v_preconditioned_cosine_with_source
sgd_momentum_cosine_with_source
update_norm_parallel_to_source
update_norm_orthogonal_to_source
readout_source_energy
hidden_source_energy
basis_source_energy
low_degree_source_energy
low_frequency_source_energy
tail_debt_peak/final/recovery
LineC_channel_debt_peak/final/recovery
ECE_debt_peak/final/recovery
Brier_debt_peak/final/recovery
AUCtime_ratio_h4800
classwise_source_collapse
stable_random_h4800_source
control_equivalent_late_drift
```

Debt recovery formula：

$$
Recovery_H = 1-\frac{Debt_H}{Debt_{peak}+\epsilon}.
$$

### 判断标准

Terminal collapse classified if one class explains at least 70% of collapsed groups:

```text
OptimizerWashout:
  cumulative_optimizer_projection_on_source_h3200_to_h4800 < -0.20
  and source_derivative_h3200_h4800 < -0.05.

DebtExplosion:
  tail/ECE/Brier/LineC debt increases by >= 50% from h3200 to h4800
  and source derivative is negative.

ControlEquivalentLateDrift:
  stable-random matched controls have h4800 positive at comparable rate.

DatasetHeterogeneity:
  collapse is concentrated in one dataset or one seed family.

TargetMisalignment:
  target local transfer is positive but source-channel projection decays.
```

### 不满足时 Codex 先尝试

```text
If UnknownTerminalCollapse > 50%:
  add h3600/h4000/h4400 checkpoints to localize collapse timing.

If optimizer projection not measured:
  instrument cumulative displacement projection and rerun only top collapse groups.

If debt missing:
  run debt readback only from saved checkpoints; do not retrain.

If stable-random controls missing:
  run matched stable-random h3200/h4800 controls before any new FU variant.
```

## 6.4 FU-H2：precommit selector 从 retrospective 变 online

### 假设

v22/v22.01 中 train-loss h400/h800 可以 retrospective 选择好 trajectory，但不是合法 online selector。v22.02 要测试是否存在只用提交前 train-stream 信息的 precommit selector，能在 fresh rerun 中选择 h4800 retained source。

### 合法 precommit features

只允许使用 candidate 提交前可见：

```text
train_loss_h100/h400 from precommit shadow rollout
split-transfer B1->B2 local gain
B3 safety gain
function-space local ActuationR2
source_state_short/long agreement
optimizer-source conflict cosine before commit
basis/readout source energy before commit
LineC-fast audit-only local readback before commit, not direction source
```

禁止使用：

```text
h800/h1600/h3200/h4800 actual outcome；
validation/test/future/query；
LineC/tail/ECE/Brier/AUC as direction optimizer。
```

### 实验

候选 pools：

```text
MLP terminal families F53/F63/F70/F74/F77/F78/F84/F85/F86;
D-CHE source-bank candidates;
D-FOU source-bank candidates;
stable-random matched target controls;
random matched target controls。
```

Phase 1：precommit scoring only。  
Phase 2：选 top-k candidates fresh rerun。  
Phase 3：independent offsets confirmation。

### 记录指标

```text
selector_feature_name
selector_train_only = 1
selector_uses_future = 0
selector_AUC_retained_h4800_on_shadow_pool
selector_precision_at_k
selector_recall_at_k
selected_candidate_count
fresh_h100/h400/h800/h1600/h3200/h4800
independent_offset_h4800
stable_random_selected_rate
random_selected_rate
selector_control_equivalent_fraction
```

### 判断标准

Precommit selector passes if:

```text
selector_uses_future = 0
precision_at_topk_h4800 >= 0.50 on shadow pool
fresh selected groups have:
  h100/h400/h800/h1600/h3200/h4800 all >= 0.005
  productive_h4800_group_count >= 2
  stable_random matched selected h4800 pass = 0
  independent confirmation pass >= 1 group
```

### 不满足时 Codex 先尝试

```text
If selector works retrospectively but fails fresh:
  mark RetrospectiveOnly; do not tune threshold.
  Try adding precommit uncertainty / source_state agreement, not future outcome.

If selector selects random controls:
  add control-equivalence penalty and rerun selector only.

If selector has low recall but high precision:
  keep as narrow gate and run more candidate pool; do not loosen to low precision.

If no feature predicts h4800:
  stop selector variants and move to target theory reset.
```

## 6.5 FU-H3：source target theory reset

### 假设

当前 loss-cotangent / B1 consensus / terminal floor targets can produce local transfer or h3200 source, but fail h4800 because target is not signal-channel-retaining. Need new target definitions.

### Target families

```text
T0 loss-cotangent target, control baseline
T1 cross-split consensus target
T2 source-channel projection target
T3 low-frequency / low-degree target
T4 readout-only target
T5 reservoir-separating target
T6 noise-orthogonal target
T7 random matched target
T8 sign-flip target
T9 corrupt target
```

### Output-space target definition

For a candidate target $\Delta f$, require local train-split transfer:

$$
G_{B2}(\Delta f) > G_{random}(\Delta f)
$$

and source-channel non-noise criterion:

$$
NoiseInSignal(\Delta f) \le NoiseInSignal(control)+\tau.
$$

This is audit/gate, not direction from validation/test.

### 实验

For each target:

```text
compute target from train B1/B2/B3 only;
solve actuation to parameter space;
measure ActuationR2;
commit only if ActuationR2 >= 0.20 and precommit selector passes;
run h100..h4800;
compare random/signflip/corrupt targets at same ActuationR2.
```

### 记录指标

```text
target_family
ActuationR2
projection_residual_norm
B2_transfer_gain
B3_safety_gain
noise_reservoir_score
source_channel_score
same_actuation_random_source
same_actuation_signflip_source
same_actuation_corrupt_source
horizon_source_chain
h4800_productive
```

### 判断标准

Target family passes if:

```text
ActuationR2 >= 0.20
B2_transfer_gain > random by >= 0.05
h4800_productive_group_count >= 2
same-actuation random/signflip/corrupt controls fail
LineC/tail/ECE/Brier debt recovered
```

### 不满足时 Codex 先尝试

```text
If ActuationR2 high but source=0:
  target is wrong; do not tune actuator.

If ActuationR2 low:
  improve solver/rank before judging target.

If random same-actuation also positive:
  target family control-equivalent; close this target.
```

## 6.6 FU-H4：MLP source dynamics decomposition

### 假设

MLP can build h3200 continuous source because dense matrix hidden/readout has a source subspace that current KAN carriers fail to replicate. v22.02 should not copy MLP Fxx family into KAN blindly; it should decompose where MLP source lives.

### 实验

For top MLP terminal groups:

```text
decompose source displacement into hidden matrix, readout matrix, bias/state, optimizer state;
compute matrix-block source direction;
compare Muon/SOAP-like block projection vs parameter projection;
measure source subspace rank and singular spectrum;
map source subspace to D-CHE degree/readout and D-FOU frequency/readout bank。
```

### 记录指标

```text
hidden_source_energy
readout_source_energy
bias_source_energy
optimizer_state_source_energy
source_subspace_rank
source_singular_values
matrix_block_alignment
parameter_alignment
MLP_to_DCHE_projection_retention
MLP_to_DFOU_projection_retention
source_reconstruction_error
```

### 判断标准

MLP source map useful if:

```text
>= 70% source energy localizes to interpretable block;
projected KAN source retention >= 0.30 in low-degree/low-frequency bank;
random block projection retention <= 0.10。
```

### 不满足时 Codex 先尝试

```text
If source is diffuse across all parameters:
  stop MLP-to-KAN projection route; use MLP only as generic FU line.

If source mainly readout:
  test KAN readout-only commit before basis commit.

If source mainly hidden matrix:
  test matrix-block KAN readout/basis block with spectral/orthogonal update。
```

## 6.7 FU-H5：KAN source-channel writer

### 假设

D-CHE / D-FOU efficiency is now sufficient; lack of KAN retained source is a carrier writeback problem. Source should be estimated in basis channel but committed into stable low-degree / low-frequency readout bank, not arbitrary basis params.

### Candidate writers

D-CHE:

```text
KSW-CHE1 low-degree source bank, commit readout only
KSW-CHE2 low-degree source bank, commit degree+readout
KSW-CHE3 basis-estimate / readout-commit split
KSW-CHE4 low-degree source + terminal source-conserving optimizer
```

D-FOU:

```text
KSW-FOU1 low-frequency source bank, commit readout only
KSW-FOU2 low-frequency source bank, commit band+readout
KSW-FOU3 band-estimate / readout-commit split
KSW-FOU4 low-frequency source + terminal source-conserving optimizer
```

### 记录指标

```text
low_degree_source_energy
high_degree_reservoir_energy
low_frequency_source_energy
high_frequency_reservoir_energy
readout_commit_norm
basis_commit_norm
source_bank_drift
source_bank_retention_horizon
h100..h4800 source
debt recovery
KAN_specific_delta_vs_MLP_same_mechanism
controls
```

### 判断标准

KAN writer passes exploration if:

```text
h100/h400/h800/h1600/h3200 all >= 0.005
h4800 >= 0.005
pass_count_h4800 >= 4/9
KAN_specific_delta_vs_MLP_same_mechanism >= 0.005 or KAN unique source channel identified
matched controls fail
source_bank_retention >= 0.50
```

### 不满足时 Codex 先尝试

```text
If readout-only works better than basis+readout:
  freeze basis commit for next run; do not tune basis lr.

If low-degree/low-frequency source dissipates:
  add source-conserving terminal hold only after h3200, not full-training hold.

If high-degree/high-frequency reservoir grows:
  add reservoir damping as recovery, not as source direction.

If MLP same mechanism stronger:
  write generic FU result, not KAN-specific claim。
```

## 6.8 FU-H6：terminal source-conserving dynamics

### 假设

h4800 collapse happens because terminal optimizer dynamics overwrite h3200 source. Need a source-conserving terminal phase, but only after h3200 chain exists. This is not another full-training optimizer switch.

### Experimental conditions

```text
C0 baseline terminal continuation
C1 source-conserving projection: remove negative component along h3200 source direction
C2 source-slow EMA: preserve long source state
C3 dual-memory AdEMAMix-like source state
C4 schedule-free slow iterate source commit
C5 matrix-block spectral source guard
C6 terminal debt-capped recovery
C7 stable-random same guard control
```

### 记录指标

```text
h3200_source
h4000_source
h4800_source
source_conservation_projection_norm
removed_update_norm
terminal_optimizer_cosine_before_after
terminal_debt_recovery
tail/ECE/Brier/LineC debt
step_time_overhead
matched control h4800
```

### 判断标准

```text
h4800_source >= 0.005
h4800 pass_count >= 4/9
R4800/3200 >= 0.50
terminal_debt_recovery >= 0.60
step_time_overhead <= 1.10
stable-random same guard fails
```

### 不满足时 Codex 先尝试

```text
If source-conserving projection preserves source but debt explodes:
  add debt-capped projection; don't increase source strength.

If projection removes too much update and task stalls:
  protect only top-k source subspace rather than full source vector.

If stable-random also benefits:
  route is generic terminal regularization, not functional source。
```

---

# 7. Controls and attribution

Every positive-looking row must compare against:

```text
NoOpMatchedOverhead
AdamW
SGD/Momentum
StableRandomMatchedNorm
RandomMatchedTarget
SameActuationRandomTarget
SameSourceAmplitudeRandom
RecoveryOnly
MLP same mechanism
same kernel without FU
same selector with random target
```

Control-equivalent rule:

$$
ControlEquivalent=1
\quad \text{if}\quad
S_{candidate}\le S_{best\_control}+0.005
$$

or if any matched random has equal or better h4800 chain.

---

# 8. Metrics and visualizations

## 8.1 Mandatory metrics

### Code

```text
compileall_pass
import_error_count
required_source_missing_count
packet_manifest_mismatch_count
linec_fast/channel pass
source_chain_tests pass
debt_tests pass
kernel_gradcheck pass
profiler_phase_isolation pass
```

### Efficiency

```text
forward_ratio_vs_mlp
backward_ratio_vs_mlp
update_ratio_vs_mlp
step_ratio_vs_mlp
memory_ratio_vs_mlp
functional_direction_overhead
linec_audit_overhead
horizon_readback_overhead
basis_materialized_bytes
kernel_fallback_used
```

### Functional

```text
source_h100/h400/h800/h1600/h2400/h3200/h4000/h4800
source_derivatives
retention ratios
early_source_chain
continuous_h3200
productive_h4800
terminal_collapse
pass_count_by_dataset_seed
control_equivalent_fraction
KAN_specific_delta_vs_MLP_same_mechanism
independent_confirmation_pass
tail/LineC/ECE/Brier/AUC debt peak/final/recovery
optimizer projection diagnostics
source subspace diagnostics
```

## 8.2 Mandatory figures

```text
fig_source_chain_horizons_by_carrier.svg
fig_terminal_collapse_waterfall_top12.svg
fig_terminal_collapse_taxonomy_heatmap.svg
fig_optimizer_projection_vs_source_derivative.svg
fig_debt_transition_h3200_h4800.svg
fig_precommit_selector_roc_pr.svg
fig_selector_fresh_vs_retrospective.svg
fig_target_actuation_vs_retention.svg
fig_mlp_source_subspace_spectrum.svg
fig_mlp_to_kan_projection_retention.svg
fig_kan_source_bank_energy.svg
fig_efficiency_full_loop_waterfall.svg
fig_dche_dfou_s1_stability.svg
fig_drat_forward_breakdown.svg
fig_drbf_local_support_breakdown.svg
fig_gpu_utilization_timeline.svg
```

---

# 9. 4GPU dynamic queue

## 9.1 GPU assignment

```text
GPU0:
  S0.9 clean packet test；
  D-CHE efficiency official closure；
  D-CHE KAN source writer。

GPU1:
  MLP terminal-collapse autopsy；
  precommit selector fresh reruns；
  terminal source-conserving dynamics。

GPU2:
  D-FOU efficiency official closure；
  D-FOU KAN source writer；
  D-FOU controls。

GPU3:
  D-RAT active repair；
  D-RBF active repair；
  stable-random / same-actuation controls；
  figures / merge / packet generation when idle。
```

## 9.2 Dynamic queue artifacts

Must output:

```text
v22_02_runnable_queue.csv
v22_02_gpu_assignment_manifest.csv
v22_02_gpu_utilization_timeline.csv
v22_02_idle_violation.csv
v22_02_deferred_items.csv
v22_02_queue_drain_report.csv
```

Idle rule:

```text
If runnable_queue non-empty and any GPU idle > 10 min:
  execution_contract_violation = 1
  final route cannot be completed-no-go.
```

---

# 10. Success gates

## 10.1 S0.9 code gate

As defined in Part A. Without S0.9, no scientific no-go.

## 10.2 E1 / S1 efficiency gates

E1 exploration:

```text
forward_ratio <= 1.75
step_ratio <= 1.75
memory_ratio <= 1.20
gradcheck_pass = 1
```

S1 official-like:

```text
forward_ratio <= 1.25
step_ratio <= 1.25
memory_ratio <= 1.05
full_loop_timing_pass = 1
official_fused_kernel_complete = 1
no_materialize_complete = 1
functional_runner_uses_same_kernel = 1
```

Near-E1 for D-RAT/D-RBF:

```text
forward_ratio <= 3.0
step_ratio <= 2.0
memory_ratio <= 1.2
gradcheck_pass = 1
```

## 10.3 FU-S2 early-chain exploration

```text
carrier x mechanism 9-row mean:
  S100/S400/S800 >= 0.005
pass_count_h800 >= 4/9
controls fail
```

## 10.4 FU-S3 h3200 productive chain

```text
S100/S400/S800/S1600/S2400/S3200 >= 0.005
pass_count_h3200 >= 4/9
R3200/1600 >= 0.50
tail or LineC debt recovery >= 0.40
matched controls fail
```

## 10.5 FU-S4 h4800 productive retention

```text
S4800 >= 0.005
pass_count_h4800 >= 4/9
R4800/3200 >= 0.50
tail_recovery_h4800 >= 0.60
LineC_recovery_h4800 >= 0.60
ECE/Brier debt non-worsening
AUCtime_ratio_h4800 <= 1.05
matched controls fail
stable-random fail
independent confirmation pass >= 1 group
```

## 10.6 FU-S5 official success

Not expected in v22.02 unless S4 opens. S5 requires:

```text
9/9 real dataset-seed pass
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

---

# 11. Failure taxonomy and next-action ladder

## 11.1 Code failure

```text
C0-PacketNotSelfContained
C1-ImportClosureFail
C2-LineCMissing
C3-KernelSourceMissing
C4-SourceChainFormulaBug
C5-ProfilerPhasePolluted
```

Action: fix code/packet first; do not run new science rows.

## 11.2 Efficiency failure

```text
E0-DCHEDFOUProfilerOnlyNotFullLoop
E1-DCHEDFOUFullLoopFail
E2-DRATForwardEvalBlocked
E3-DRBFDenseMaterializationBlocked
E4-KernelGradcheckFail
E5-AuditCostPollutedTiming
```

Action: targeted repair based on blocker; no full FU for non-near-E1 D-RAT/D-RBF.

## 11.3 Functional failure

```text
F0-NoEarlySource
F1-H3200ChainButH4800Collapse
F2-ControlEquivalentLateDrift
F3-RetrospectiveSelectorOnly
F4-TargetObservableNoRetention
F5-KANSourceBankMismatch
F6-OptimizerWashout
F7-DebtExplosion
F8-UnknownTerminalCollapse
```

Action:

```text
F1 -> terminal autopsy and source-conserving terminal dynamics.
F2 -> strengthen stable-random / same-actuation controls, close target if controls explain.
F3 -> fresh precommit selector rerun; no threshold tuning on future outcomes.
F4 -> target theory reset; stop actuation-only improvements.
F5 -> MLP source decomposition and KAN source-bank mapping.
F6 -> terminal optimizer projection guard.
F7 -> debt-capped source writer.
F8 -> add intermediate horizons and projection/debt instrumentation.
```

---

# 12. Expected final outputs

v22.02 final report must have three fixed sections:

```text
Part 1 Code audit:
  Is code correct? Any missing file? Does code match plan?

Part 2 Basis efficiency:
  Did D-CHE/D-FOU officialize? Did D-RAT/D-RBF improve? Where blocked?

Part 3 Functional update:
  Did h4800 retained source open? If not, what exact terminal-collapse class? What next action?
```

Final route must not be vague. It must answer:

```text
1. 是否仍然 code packet 不自包含？
2. D-CHE / D-FOU 是否 full-loop official closure？
3. D-RAT / D-RBF 是否达到 near-E1？
4. 12 个 h3200 continuous groups 为什么 h4800 collapse？
5. 是否存在 legal precommit selector？
6. function-space high ActuationR2 失败是 target 错还是 actuation 错？
7. KAN source-bank writer 是否比 direct basis commit 更好？
8. MLP h3200 source 的 subspace 能否映射到 KAN？
9. 是否至少打开 FU-S3 或 FU-S4？
```

---

# 13. v22.02 最后判断

v22.02 的实验目标不是“继续多跑一些 Fxx 变体”。它必须把当前进展推进到可裁决阶段：

```text
Efficiency:
  D-CHE / D-FOU 从 official-like profiler pass 推到 full-loop kernel closure；
  D-RAT / D-RBF 从被动 rejected 推到 active near-E1 repair or明确关闭。

Functional:
  从 h3200 continuous source 推到 h4800 retained source；
  如果推不上去，必须明确 terminal-collapse 机制和是否存在合法 precommit selector。

Code:
  从运行环境自检通过，推进到最终 zip 自包含独立审计通过。
```

一句话：

$$
\boxed{
\text{v22.02 不再扩大 late rebound；}
\text{它要裁决 terminal collapse，并把 efficiency pass 变成 full-loop 证据。}
}
$$
