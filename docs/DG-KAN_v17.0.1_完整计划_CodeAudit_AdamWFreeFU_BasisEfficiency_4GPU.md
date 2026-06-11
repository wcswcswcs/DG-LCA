# DG-KAN v17 完整计划：代码审计闭环、AdamW-free Functional Update、Basis Kernel Efficiency、4GPU 动态并行

> 版本：v17 execution plan  
> 生成时间：2026-06-02（Asia/Singapore）  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 本计划基于：v16.4.1 真实结果复盘、上传代码包本地解压审查、v12-v16 历史教训、用户明确要求“不要继续用错误指标/标准原地打转”。  
> 硬约束：strict FC-PureKAN；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit / gate / debt readback，不能生成方向。

---

# 0. 总目标、当前状态与 v17 的根本变化

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是让某个局部 metric 变好。项目目标是构建一个 strict FC-PureKAN / KAN-like efficient functional training system，使它在表达力、训练速度、显存、训练轨迹、几何和泛化上可以成为 MLP 的替代方案。最终 claim 需要同时满足两条主线：

$$
\boxed{
\text{Functional update 能带来普通 backprop / optimizer controls 之外的长期训练动力学收益。}
}
$$

以及：

$$
\boxed{
\text{KAN basis 的 forward、backward、update、memory 必须接近同参数量 MLP。}
}
$$

v16.4.1 的真实结果说明：functional update 仍没有形成长期 retained productive dynamics。全局 best h800 source 来自 MLP 的 `B-M4a-MLP-AdamSubspaceProximal-alpha000`，但 h1600 后 source 归零，KAN-specific advantage rows 为 0；basis efficiency truth table 有 114 rows、measured=114，但它只是工程证据，不是 functional promotion。v16.4.1 的 final route 是 `R4-FunctionalSourceNotRetained`，S2/S3/S5 全部为 0，promotion 仍为 0。

v17 的根本变化是：不再继续默认 `AdamW + FU residual` 是 functional update 的自然形态。反传 / loss-interface 提供任务压力是必要的，但 AdamW 只是 optimizer 的一种，不是 functional update 必须绑定的主干。v17 必须正面裁决：

$$
\boxed{
\text{FU 是否被 AdamW 洗掉、抵消或变成 control-equivalent？}
}
$$

同时，v17 也不再允许 basis efficiency 只停留在 “census”。每个 basis family 必须有 family-specific kernel repair，不允许只写 blocked summary 后进入下一轮。

最后，v17 的第一阶段不是实验，而是代码正确性闭环。原因很简单：如果 LineC 实现错误、update 符号混乱、AdamW coupling 没拆开、efficiency profiler 计时污染，那么后续再多实验都会被错误指标/标准拖进“原地打转”。这里的原地打转不是死循环，而是错误测量和错误标准让真实进展无法显现。

---

# 1. 本地代码审查初步结论：v17 必须先过 S0 Code Correctness Gate

我已经把上传包解压到：

```text
/mnt/data/dgkan_code_audit_v17_deep
```

基础结果是：

```text
compileall_ok = True
modules_checked = 75
import_errors = 17
```

这说明代码包语法能过，但核心 runner 不能完整 import。典型缺失包括：

```text
experiments.run_v1410_nonrat_fms_transfer_fms_definition_reset
  missing: experiments.run_v133_task_family_robust_basis_natural
  missing: experiments.run_v1231_basis_kernel_workspace

experiments.run_v144_real_transfer_fms_all_basis_substrate
  missing: experiments.run_v1231_basis_kernel_workspace

experiments.run_v1641_functional_update_efficiency_breakthrough_4gpu
  indirectly missing: experiments.run_v133_task_family_robust_basis_natural
```

这不是小问题。当前包不是自包含审查包，无法单独复现核心 runner、LineC、functional update、efficiency census 的真实语义。因此 v17 的第一硬门是 import closure。

## 1.1 LineC 相关 blocker

多个 runner 从 `experiments.run_v1231_basis_kernel_workspace import linec_metrics`，但上传包中没有 `run_v1231_basis_kernel_workspace.py`，也没有 `linec_metrics` 的真实实现。与此同时，调用方存在把 LineC exception 转成 NaN 再按 fail 处理的逻辑。例如 v14.10 风格实现中：

```python
try:
    lm = linec_metrics(...)
    status = "executed"
except Exception as exc:
    lm = {"CouplingR2": nan, "NoiseSignalLeak": nan, "RealSignalReservoirRatio": nan}
    status = "blocked"
passed = int(fnum(lm["CouplingR2"], -999) >= 0.15 ...)
```

这个逻辑会把 measurement implementation error 包装成 geometry fail。v17 必须改成：

```text
LineC exception = MeasurementInvalid
MeasurementInvalid 不进入 pass/fail 均值
MeasurementInvalid 会阻断 final route
LineC missing implementation 会阻断 S0
```

也就是说，不能再让 “LineC 没测成” 变成 “几何失败”。

## 1.2 AdamW coupling blocker

代码中大量 FU 训练路径仍是：计算某个 projected / functional update，然后写进 `.grad`，最后调用 `AdamW.step()`。这种实现实际测试的是：

```text
AdamW preconditioning / momentum / weight decay / optimizer state
+
FU projected gradient
```

而不是 AdamW-free FU。v17 必须把它拆开成：

```text
Backprop cotangent / train-stream loss pressure
  ≠ AdamW optimizer

Gradient-like tensor
  ≠ Step-like parameter displacement

Functional displacement
  ≠ .grad handed to AdamW
```

如果不拆开，我们就只能证明 “AdamW + FU residual 没成功”，不能证明 functional update 本身没价值。

## 1.3 更新方向语义 blocker

代码中同时存在类似：

```python
add_flat_update(specs, step_update, +lr)
apply_flat_update(model, update, lr)
assign_flat_grad(specs, projected_flat); opt.step()
stateless_probe_update(..., lr, weight_decay)
```

这些路径的符号语义并不统一。有些 update 是 `step-like`，应该直接加到参数；有些 update 是 `gradient-like`，应该经 optimizer 取负方向；有些是 `cotangent-like`，还没有映射成参数方向。v17 必须引入 update type system，否则我们可能一直在测符号混乱的 FU。

## 1.4 多方案 matrix 的 semantic collapse blocker

v16.1-v16.4 表面上有 M1-M8、多 carrier、多 controls，但不少 method 最终会落到同一套底层 update 语义，只是换了名字、pulse/recovery 标签或 readback horizon。v17 必须加 semantic non-collapse audit：每个 method 必须落盘其实际 update source、optimizer coupling、direction type、commit type、carrier role、param mask、sign convention、state dependencies。如果两个 method 的 committed update cosine > 0.999 且 param mask 相同，则它们不能算独立机制。

## 1.5 Efficiency profiler blocker

v16.3/v16.4.1 已开始拆 forward/backward/update/functional/audit/memory，但仍有风险：

```text
1. training cost 与 audit / horizon readback cost 混在 step_total 里；
2. optimizer update phase 有时测的是 AdamW live gradients，有时测的是 functional commit；
3. same-param MLP 只是近似隐藏维匹配，需要记录 param_delta；
4. blocked rows 有 blocker class，但有些缺 phase-level full values；
5. peak memory 要区分 training backward peak、audit peak、horizon readback peak。
```

v17 中 efficiency truth table 不是普通 artifact，而是 scientific route 的前置条件。

---

# 2. Codex 必须打包给审查的完整 code review packet

v17 不能再只打包当前 runner 和几个 CSV。Codex 必须生成一个完整、自包含、可 import、可执行的代码审查包。包名建议：

```text
v17_code_review_packet.zip
```

包内目录结构必须如下：

```text
v17_code_review_packet/
  00_README.md
  01_ENVIRONMENT/
  02_SOURCE_TREE/
  03_IMPORT_CLOSURE/
  04_LINEC_CORRECTNESS/
  05_UPDATE_SEMANTICS/
  06_ADAMW_COUPLING_AUDIT/
  07_FUNCTIONAL_MECHANISM_NONCOLLAPSE/
  08_EFFICIENCY_PROFILER_CORRECTNESS/
  09_KERNEL_GRADCHECK/
  10_EXPERIMENT_RUNNERS/
  11_RESULTS_MANIFESTS/
  12_FIGURES_AND_DASHBOARDS/
  13_FAILURE_TAXONOMY/
  14_REPRO_COMMANDS/
  packet_manifest.csv
  packet_sha256_manifest.csv
```

## 2.1 00_README.md 必须包含

Codex 必须写清楚：

```text
1. repo root absolute path
2. git commit hash
3. git diff summary
4. Python / CUDA / PyTorch / Triton / driver versions
5. conda env export
6. GPU count and GPU names
7. code packet creation command
8. exact command to run S0 tests
9. exact command to reproduce one smoke row
10. known limitations and missing dependencies
```

如果没有 git 仓库或 commit hash，必须写：

```text
git_commit_unavailable=1
reason=...
```

不能留空。

## 2.2 01_ENVIRONMENT 必须包含

```text
python_version.txt
pip_freeze.txt
conda_env.yml
nvidia_smi.txt
torch_cuda_report.json
triton_version.txt
```

`torch_cuda_report.json` 至少包含：

```text
torch.__version__
torch.version.cuda
torch.backends.cudnn.version()
torch.cuda.device_count()
per-device name / memory / capability
```

## 2.3 02_SOURCE_TREE 必须包含完整源代码

必须打包以下目录和文件，不允许只打包 runner：

```text
dgkan/
experiments/
docs/  # 至少打包 v16.4.1-v17 相关 plan/recap
```

最低必须包含下列关键文件；缺任何一个，S0 fail：

```text
dgkan/models/fc_purekan_primitives.py
dgkan/models/fc_purekan_lq.py
dgkan/models/manual_full_edge.py
dgkan/models/edge_layers.py
dgkan/models/edge_functions.py
dgkan/kernels/fused_hinge_quadratic.py
dgkan/kernels/fused_chebyshev_k3.py
dgkan/kernels/fused_fourier_k2.py
dgkan/kernels/fused_hat_wavelet.py
dgkan/kernels/fused_rbf.py
dgkan/functional/manifold_channel_geometry.py
dgkan/functional/geometry.py
dgkan/functional/mlp_functional.py
dgkan/diagnostics/basis_workspace.py
dgkan/diagnostics/classic_basis.py
dgkan/optim/manual_adamw.py
dgkan/optim/foreach_adamw.py
dgkan/training/gradcheck.py
dgkan/training/manual_full_edge.py
experiments/run_v120_good_geometry_battery.py
experiments/run_v1231_basis_kernel_workspace.py
experiments/run_v133_task_family_robust_basis_natural.py
experiments/run_v136_poprisk_snr_basis_cover_boundary.py
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
experiments/run_v144_real_transfer_fms_all_basis_substrate.py
experiments/run_v149_line_d_all_basis_substrate_repair.py
experiments/run_v150_function_update_allbasis_parallel.py
experiments/run_v154_split_consensus_signal_subspace_fu_allbasis.py
experiments/run_v158_dynamics_harness_decoupled_decay_recovery_allbasis.py
experiments/run_v162_multischeme_functional_dynamics_4gpu.py
experiments/run_v163_multischeme_functional_dynamics_efficiency_census_4gpu.py
experiments/run_v1641_functional_update_efficiency_breakthrough_4gpu.py
experiments/run_v17_code_correctness_audit.py
experiments/run_v17_adamw_free_functional_matrix.py
experiments/run_v17_basis_kernel_efficiency_matrix.py
experiments/run_v17_4gpu_scheduler.py
```

如果某历史 runner 已被重构到新 module，必须提供 compatibility shim，并在 `compatibility_manifest.csv` 中说明旧符号映射到新符号。

## 2.4 03_IMPORT_CLOSURE 必须包含

```text
compileall.log
import_closure.csv
import_errors.csv
module_dependency_graph.json
missing_symbol_report.csv
```

`import_closure.csv` 字段：

```text
module
path
import_status
error_type
error_message
imports_experiments_count
imports_dgkan_count
```

S0 通过标准：

```text
compileall_pass = 1
import_error_count = 0
missing_symbol_count = 0
```

如果 import error > 0，不允许进入 v17 science run。

## 2.5 04_LINEC_CORRECTNESS 必须包含

```text
linec_source_manifest.csv
linec_golden_fixture.py
linec_golden_results.csv
linec_exception_policy_test.csv
linec_null_distribution.csv
linec_split_transfer_test.csv
linec_noise_injection_test.csv
linec_signal_reservoir_test.csv
linec_batch_order_invariance_test.csv
linec_seed_stability_test.csv
```

LineC golden tests 必须覆盖以下 fixtures：

```text
1. NoOp update：CouplingR2 delta 接近 0，NoiseSignalLeak 不应系统性下降。
2. RandomMatchedNorm update：不能产生 systematic LineC pass。
3. Signal-aligned linear update：B1/B2 train motion coupling 应上升。
4. Noise injection update：NoiseSignalLeak 应上升。
5. Reservoir-only perturbation：train local motion 有变化，但 probe/test-visible coupling 不应稳定上升。
6. Batch permutation：同一 data content 顺序改变后 LineC metrics 不应大幅变化。
7. Exception path：linec_metrics 抛异常时必须 MeasurementInvalid，不能写几何 fail。
```

LineC 的核心测量输出必须至少包含：

```text
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
measurement_status
measurement_error
valid_for_gate
```

S0 通过标准：

```text
linec_importable = 1
linec_golden_pass_rate = 1.0
noop_false_positive_rate <= 0.05
random_false_positive_rate <= 0.05
measurement_invalid_count = 0 on golden fixtures
exception_as_fail_count = 0
```

## 2.6 05_UPDATE_SEMANTICS 必须包含

```text
update_type_manifest.csv
update_sign_golden_tests.csv
gradient_like_vs_step_like_test.csv
finite_difference_descent_test.csv
fu_commit_rollback_test.csv
role_partition_update_test.csv
```

v17 必须引入 update type system。每个 update object 需要声明：

```text
update_id
direction_type = gradient_like | step_like | cotangent_like | function_displacement | metric_state | unknown
commit_semantics = add_to_param | assign_to_grad_then_optimizer | optimizer_internal | readback_only
sign_convention = descent_positive | ascent_positive | unknown
uses_optimizer = none | adamw | sgd | momentum | schedule_free | manual
uses_weight_decay
uses_momentum_state
uses_second_moment_state
role_mask
param_count_touched
```

S0 通过标准：

```text
unknown_direction_type_count = 0
unknown_sign_convention_count = 0
finite_difference_descent_pass_rate >= 0.95 for intended descent updates
rollback_error_max < 1e-8
```

## 2.7 06_ADAMW_COUPLING_AUDIT 必须包含

```text
adamw_coupling_map.csv
adamw_overwrite_diagnostic.csv
optimizer_state_dependency.csv
fu_vs_adamw_cosine.csv
fu_source_survival_by_recovery_optimizer.csv
```

`adamw_coupling_map.csv` 字段：

```text
method
carrier
line
uses_adamw_step
assigns_fu_to_grad
uses_adamw_momentum
uses_adamw_v
uses_decoupled_weight_decay
fu_committed_directly
optimizer_primary
fu_primary
recovery_optimizer
```

v17 必须能够区分：

```text
AdamW-primary + FU residual
SGD-primary + FU
Momentum-primary + FU
FU-primary
FU-only
Alternating FU / gradient
Role-partition optimizer
Schedule-free / slow-state FU
```

如果某个 method 名字声称 FU-only，但 `uses_adamw_step=1`，S0 fail。

## 2.8 07_FUNCTIONAL_MECHANISM_NONCOLLAPSE 必须包含

```text
mechanism_semantic_manifest.csv
mechanism_update_cosine_matrix.csv
mechanism_param_mask_jaccard.csv
mechanism_state_dependency_matrix.csv
mechanism_noncollapse_summary.csv
```

判断规则：若两个 methods 在同一 carrier/dataset/seed/horizon 上满足：

$$
\cos(u_i,u_j) > 0.999
$$

且：

$$
Jaccard(mask_i, mask_j) > 0.99
$$

且 optimizer state dependencies 相同，则它们为 semantic collapse，只能算一个机制。

S0 通过标准：

```text
collapse_fraction <= 0.25 for claimed independent mechanisms
all claimed mechanisms have distinct actual update semantics
```

## 2.9 08_EFFICIENCY_PROFILER_CORRECTNESS 必须包含

```text
efficiency_profiler_unit_tests.csv
efficiency_phase_timer_tests.csv
audit_cost_separation_test.csv
same_param_mlp_matching_test.csv
memory_peak_reset_test.csv
warmup_vs_measured_test.csv
```

每个 phase 必须独立测：

```text
forward_only
loss_delta
backward_grad
optimizer_update
functional_direction
functional_projection
functional_commit
linec_audit
tail_calibration_audit
horizon_readback
step_total_train_only
step_total_with_audit
```

必须区分：

```text
training_step_total
training_plus_audit_total
horizon_readback_total
```

如果 `step_total` 混入 LineC/horizon audit 而没有单独字段，S0 fail。

## 2.10 09_KERNEL_GRADCHECK 必须包含

```text
kernel_gradcheck_summary.csv
chebyshev_gradcheck.csv
fourier_gradcheck.csv
rbf_gradcheck.csv
wavelet_gradcheck.csv
lq_gradcheck.csv
rational_gradcheck.csv
manual_vs_autograd_gradcheck.csv
```

每个 basis 必须跑小规模 gradcheck：

```text
B = 4 / 8
input_dim = 8 / 16
hidden = 8 / 16
classes = 3 / 5
```

S0 gate：

```text
forward_relerr <= 1e-6
param_grad_relerr <= 1e-4 or grad_cos >= 0.999
input_grad_relerr <= 1e-4 or grad_cos >= 0.999
no NaN / Inf
```

## 2.11 10_EXPERIMENT_RUNNERS 必须包含

```text
run_v17_code_correctness_audit.py
run_v17_adamw_free_functional_matrix.py
run_v17_basis_kernel_efficiency_matrix.py
run_v17_4gpu_scheduler.py
run_v17_finalize.py
```

每个 runner 必须支持：

```text
--device cuda:N
--shard-index
--shard-count
--out-dir
--resume
--reuse-if-present
--fail-on-missing-s0
--smoke-only
```

## 2.12 11_RESULTS_MANIFESTS 必须包含

```text
required_artifact_manifest.csv
forbidden_information_audit.csv
no_action_search_audit.csv
direction_provenance.csv
measurement_validity_manifest.csv
budget_exhaustion_certificate.csv
deferred_items.csv
next_hypothesis_queue.md
no_go_boundary.md
```

## 2.13 packet_manifest.csv 字段

```text
relative_path
sha256
bytes
artifact_type
required
created_by
source_command
```

S0 通过标准：

```text
required_missing_count = 0
sha256_missing_count = 0
packet_self_contained = 1
```

---

# 3. v17 代码实现大改：先建立正确抽象，再跑科学实验

v17 的代码改造必须服务两个目标：一是让 functional update 不再默认 AdamW-coupled；二是让每个 basis 的效率路径可审计、可优化、可比较。

## 3.1 新增 `dgkan/fu/` 子系统

建议新增：

```text
dgkan/fu/__init__.py
dgkan/fu/types.py
dgkan/fu/loss_interface.py
dgkan/fu/update_semantics.py
dgkan/fu/carriers.py
dgkan/fu/mechanisms.py
dgkan/fu/optimizers.py
dgkan/fu/slow_state.py
dgkan/fu/matrix_block.py
dgkan/fu/poprisk_snr.py
dgkan/fu/linec_readback.py
dgkan/fu/controls.py
dgkan/fu/audit.py
```

### 3.1.1 `types.py`

定义强类型：

```python
@dataclass
class UpdateTensor:
    tensor: torch.Tensor
    direction_type: Literal[
        "gradient_like", "step_like", "cotangent_like", "function_displacement", "metric_state"
    ]
    sign_convention: Literal["descent_positive", "ascent_positive", "unknown"]
    carrier: str
    mechanism: str
    role_mask_id: str
    uses_adamw: bool
    uses_momentum: bool
    uses_second_moment: bool
    uses_weight_decay: bool
    provenance: dict
```

任何更新进入参数前必须经过：

```python
validate_update_semantics(update)
```

如果 `direction_type="unknown"` 或 sign 未声明，直接 fail。

### 3.1.2 `loss_interface.py`

只负责生成 train-stream loss cotangent / gradient，不负责 optimizer step。必须支持：

```text
CE cotangent
Brier cotangent diagnostic, if needed
per-example output cotangent
per-example parameter gradients
split B1/B2 cotangent
```

但要明确：CE / Brier 是 loss-interface，不能变成 CE-tail / ECE / LineC direction source。

### 3.1.3 `optimizers.py`

把 optimizer 从 FU 中拆开，至少实现：

```text
AdamWPrimary
SGDPrimary
MomentumPrimary
FUPrimary
FUOnly
AlternatingFUGradient
RolePartitionOptimizer
ScheduleFreeSlowState
MatrixBlockOptimizer
NoOptimizerRecovery
```

这些 optimizer 都必须实现共同接口：

```python
propose_step(loss_signal, model_state, fu_state) -> UpdateTensor
commit_step(model, update)
```

AdamW 不再是默认，而只是其中一个 mode。

### 3.1.4 `mechanisms.py`

机制不再写成 G-token / FU-token，而是明确机制类型：

```text
ProductivePulseRecovery
SplitConsensusMetric
PopRiskDriftDiffusion
FunctionSpaceProximal
OptimizerAwareAlignedFU
CarrierReparameterizedFU
LateAttachSnapshotFU
RecoveryOnlyDeconfound
SlowStateFunctionalUpdate
MatrixBlockFunctionalUpdate
AdamWFreeFU
```

每个 mechanism 必须输出 semantic manifest。

### 3.1.5 `carriers.py`

Carrier 必须是明确对象，不再用 string 到处分支：

```text
D_CHE_Carrier
MLP_Carrier
LQ_Carrier
Rational_Carrier
D_FOU_Carrier
D_RBF_Carrier
D_WAV_Carrier
FHQ_Carrier_Anchor
```

每个 carrier 要提供：

```text
named_roles()
flat_params()
role_masks()
forward_features()
function_space_sketch()
basis_telemetry()
efficiency_kernel_info()
```

## 3.2 新增 `dgkan/efficiency/` 子系统

```text
dgkan/efficiency/profiler.py
dgkan/efficiency/same_param_mlp.py
dgkan/efficiency/kernel_census.py
dgkan/efficiency/phase_timer.py
dgkan/efficiency/memory_timer.py
dgkan/efficiency/repair_registry.py
```

它必须保证所有 basis 和 MLP 使用同一 profiler，输出同一 schema。

## 3.3 LineC 实现迁移到稳定模块

LineC 不应散落在历史 runner。v17 必须把 LineC 收敛到：

```text
dgkan/diagnostics/linec.py
```

函数：

```python
linec_metrics(model, train_batch, probe_batch, *, seed, sketch_dim, update=None) -> LineCResult
```

`LineCResult` 必须包含：

```text
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
measurement_status
measurement_error
valid_for_gate
```

禁止调用方吞异常后写 NaN fail。

---

# 4. v17 实验总结构：两条主线，四张 GPU，全矩阵执行

v17 仍然是两条主线，但每条都必须完整展开。

```text
Mainline 1: Functional Update
Mainline 2: KAN Basis Efficiency
```

二者同等重要。如果 FU 成功但 basis 慢很多，不能 claim next-gen MLP。如果 basis 快但 FU 没有长期收益，也不能 claim functional success。

## 4.1 Full Carrier Matrix

v17 active carriers：

```text
C1-D-CHE
C2-MLP
C3-LQ
C4-Rational
C5-D-FOU
C6-D-RBF/FastKAN
C7-D-WAV
C8-FHQ/B320/B314 anchor diagnostic, promotion disabled if label-informed
```

每个 carrier 的角色：

```text
D-CHE:
  当前最强 KAN functional carrier，必须 full matrix。

MLP:
  主动 functional dynamics discovery line，不只是 control。

LQ:
  reanchor + late attach + AdamW-free smoke；reanchor 未开也跑 promotion-disabled diagnostic。

Rational:
  monitor + FU smoke；不重启 reset/controller/action。

D-FOU/RBF/WAV:
  basis repair + minimum functional smoke；即使 substrate gate 未开，也要低预算测 signal，不允许完全跳过。

FHQ anchor:
  只作 historical / efficiency anchor 和 implementation reference；若涉及 label-informed path，promotion_allowed=0。
```

## 4.2 Full Mechanism Matrix

每个 active carrier 至少要覆盖以下机制族。

### M0 Baselines and controls

```text
AdamW-primary
SGD-primary
Momentum-primary
NoOp matched overhead
Random matched norm
Same active fraction
Same projection retention
Same update norm
Recovery-only
Extra steps matched time
```

### M1 AdamW-primary + FU residual

旧路线。保留作为 baseline，不再作为默认主线。

```text
update = AdamW(g) + FU_residual
```

### M2 SGD/Momentum-primary + FU

测试 AdamW 是否洗掉 FU source：

```text
update = SGD/Momentum(g) + FU
```

### M3 FU-primary

反传只给 loss signal，FU 决定主要参数 step：

```text
update = FU(g, carrier_state, train_split_state)
```

### M4 FU-only smoke

不使用 AdamW / SGD primary，只用 FU step 进行短程训练。目标不是 promotion，而是判断 FU 是否本身具备训练动力。

### M5 Alternating FU / gradient training

```text
k steps gradient optimizer
1 step FU
k steps recovery
```

recovery optimizer 要比较：

```text
AdamW
SGD
Momentum
Schedule-free / EMA
No optimizer recovery
```

### M6 Schedule-free / slow-state FU

解决 h800 source 到 h1600 消失问题。维护 slow source state：

$$
s_{t+1} = \beta s_t + (1-\beta)P_{signal}(u_t)
$$

更新：

$$
\theta_{t+1}=\theta_t+u^{base}_t+\eta_s P_{safe}(s_t)
$$

其中 $P_{signal}$ 只能来自 train-stream split-consensus / PopRisk / SNR，不允许使用 audit metrics。

### M7 Matrix-block FU

在矩阵 / block 空间中定义 FU，不逐坐标打散 source。适用：

```text
MLP hidden weight
D-CHE degree-readout block
LQ projection frame
D-FOU band-readout block
Rational numerator/denominator group
D-RBF center-readout block
```

### M8 PopRisk / SNR signal-state FU

基于 per-example gradient drift-diffusion：

$$
\mu_r = \frac{1}{b}\sum_i g_{i,r}
$$

$$
\sigma_r^2 = \frac{1}{b}\sum_i (g_{i,r}-\mu_r)^2
$$

$$
SNR_r = \frac{\mu_r^2}{\sigma_r^2/(b-1)+\epsilon}
$$

更新只允许使用 train-stream per-example gradients。

### M9 Function-space proximal / split-transfer operator

不是 action search，而是在固定低秩子空间求解：

$$
\alpha^* = \arg\min_\alpha L_{B1}(\theta + U\alpha) + \lambda L_{B2}(\theta + U\alpha) + \rho\|\alpha\|^2
$$

固定 alpha candidate，不允许自适应 search。

### M10 Late attach / snapshot FU

测试 FU 是否不该全程使用，而应该在某个 training phase attach。

```text
early attach
mid attach
late attach
one-shot attach
periodic attach
```

### M11 AdamW-overwrite diagnostic

不作为 method，只做诊断：

$$
\cos\left(\sum_{\tau=t}^{t+H}\Delta\theta^{AdamW}_\tau, u_{FU,t}\right)
$$

判断 AdamW 是否系统性抵消 FU。

---

# 5. Functional update 指标与判断标准

所有 FU rows 必须记录：

```text
carrier
mechanism
optimizer_primary
recovery_optimizer
uses_adamw
uses_sgd
uses_momentum
uses_slow_state
uses_matrix_block
uses_poprisk_snr
uses_function_space_solver
direction_type
commit_semantics
source_vs_best_control_h1/h20/h100/h400/h800/h1600
source_retention_h100/h400/h800/h1600
tail_debt_peak
tail_debt_h800/h1600
tail_recovery_rate_h800/h1600
LineC_debt_peak
LineC_debt_h800/h1600
LineC_recovery_rate_h800/h1600
calibration_debt_peak
AUCtime_ratio_h800/h1600
AdamW_overwrite_cosine
FU_vs_grad_cosine
FU_vs_SGD_cosine
FU_vs_momentum_cosine
FU_vs_slow_state_cosine
matched_control_gap
MLP_attribution_delta
KAN_specific_advantage
semantic_collapse_group
```

## 5.1 S2 weak productive dynamics

$$
source\_vs\_best\_control_{h800} \ge 0.005
$$

$$
source\_retention_{h800} \ge 0.40
$$

且：

$$
tail\_recovery_{h800} \ge 0.40
$$

或：

$$
LineC\_recovery_{h800} \ge 0.40
$$

同时：

$$
AUCtime\_ratio_{h800} \le 1.10
$$

matched controls fail。

## 5.2 S3 productive debt recovery

$$
source\_retention_{h800} \ge 0.50
$$

$$
tail\_recovery_{h800} \ge 0.60
$$

$$
LineC\_recovery_{h800} \ge 0.60
$$

$$
AUCtime\_ratio_{h800} \le 1.05
$$

并且 random pulse、recovery-only、NoOp overhead 都不能解释。

## 5.3 S4 real-transfer exploration

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail debt recovered
LineC debt recovered
controls fail
```

## 5.4 S5 official success

S5 不降低：

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

---

# 6. Basis Kernel Efficiency 主线

v17 的 efficiency 不再只是 census，而是 census + repair + validation。

## 6.1 Mandatory efficiency truth table

每个 carrier / basis / same-param MLP 必须记录：

```text
forward_only_ms
loss_delta_ms
backward_grad_ms
optimizer_update_ms
functional_direction_ms
functional_projection_ms
functional_commit_ms
linec_audit_ms
tail_calibration_audit_ms
horizon_readback_ms
step_total_train_only_ms
step_total_with_audit_ms
forward_peak_memory_mb
backward_peak_memory_mb
optimizer_state_memory_mb
basis_activation_bytes
functional_state_bytes
audit_peak_memory_mb
horizon_peak_memory_mb
kernel_count
small_kernel_count
num_gemm
num_exp
num_trig
num_gather
num_scatter
num_custom_kernel
same_param_mlp_param_count
basis_param_count
param_delta_fraction
```

如果缺任何 phase，final route invalid。

## 6.2 Efficiency gate

Exploration：

```text
forward_ratio <= 1.75
backward_ratio <= 1.75
step_train_only_ratio <= 1.75
backward_memory_ratio <= 1.50
```

Official efficiency：

```text
forward_ratio <= 1.25
backward_ratio <= 1.40
step_train_only_ratio <= 1.25
backward_memory_ratio <= 1.25
```

注意：`step_total_with_audit` 不用于 official training efficiency gate，但必须记录，防止 audit cost 污染。

## 6.3 Family-specific repair

### D-CHE repair

核心问题：D-CHE functional path 混入 basis eval、split-gradient、LineC/horizon readback。v17 要拆：

```text
CHE-R1 training-only phase separation
CHE-R2 degree recurrence cache
CHE-R3 low-degree active bank
CHE-R4 fused degree-role projection update
CHE-R5 no-materialize degree energy readback
CHE-R6 analytic backward for degree bank
```

若 D-CHE forward ratio >1.75：先修 recurrence / fusion。若 backward ratio >1.75：先修 analytic backward。若 audit overhead >35%：LineC/horizon readback 从 train step 中拆离。

### D-FOU repair

v16.3/v16.4.1 已显示 D-FOU 局部 step/memory 可承受，但 forward 慢。v17 重点：

```text
FOU-R1 sin/cos recurrence
FOU-R2 precomputed frequency table
FOU-R3 fused low-frequency band eval
FOU-R4 high-frequency quarantine
FOU-R5 band-readout GEMM fusion
```

若 forward >1.75 且 memory <=1.25，Codex 必须优先修 forward，不要继续 substrate token。

### LQ repair

LQ 历史有 near-pass，但 forward/update 慢。v17 重点：

```text
LQ-R1 fixed-frame forward cache
LQ-R2 fused projection update
LQ-R3 persistent AdamW / foreach update audit
LQ-R4 late attach with frozen projection frame
LQ-R5 rank-limited projection smoke
```

若 update ratio >1.75，必须拆 projection update 与 optimizer update，不能笼统写 LQ slow。

### Rational repair

```text
RAT-R1 branchless denominator eval
RAT-R2 denominator/derivative telemetry separated from train step
RAT-R3 numerator/denominator block update
RAT-R4 rational eval Horner form
RAT-R5 tangent metric smoke, promotion disabled
```

不允许重启 reset/controller/action。

### D-RBF / FastKAN repair

```text
RBF-R1 active-center occupancy
RBF-R2 compact K4 no dense materialization
RBF-R3 width condition guard
RBF-R4 table lookup / local gaussian approximation
RBF-R5 center-readout block update
```

### D-WAV repair

```text
WAV-R1 sparse triangular support
WAV-R2 support index-only backward
WAV-R3 support-overlap damping readback
WAV-R4 scale occupancy telemetry
```

---

# 7. 4GPU 动态队列执行

v17 必须充分使用服务器 4 张显卡。

## 7.1 队列文件

Codex 必须生成：

```text
v17_runnable_queue.csv
v17_gpu_assignment_manifest.csv
v17_gpu_utilization_dashboard.csv
v17_idle_violation.csv
v17_deferred_items.csv
v17_queue_drain_report.csv
```

`v17_runnable_queue.csv` 字段：

```text
job_id
priority
line
carrier
mechanism
stage
estimated_minutes
requires_gpu
preferred_gpu
can_steal
dependencies
status
assigned_gpu
start_time
end_time
artifact_path
```

## 7.2 GPU 默认分配

```text
GPU0:
  S0 tests + D-CHE FU full matrix + D-CHE kernel repair

GPU1:
  MLP FU full matrix + MLP AdamW-free / slow-state / matrix-block tests

GPU2:
  LQ reanchor/smoke + Rational monitor/smoke + LQ/RAT kernel repair

GPU3:
  D-FOU / D-RBF / D-WAV kernel repair + substrate/FU smoke
```

## 7.3 动态补位规则

如果 primary queue 完成，GPU 必须自动拉取：

```text
1. missing S0 tests
2. Line M matched controls
3. efficiency missing rows
4. all-basis smoke rows
5. h1600 extension for top source-retaining rows
6. figure/audit generation
```

若 runnable queue 非空但任一 GPU idle 超过 10 分钟：

```text
execution_contract_violation = 1
final route cannot be completed-no-go
```

---

# 8. v17 实验阶段

## Stage S0: Code Correctness Gate

必须先执行：

```bash
python experiments/run_v17_code_correctness_audit.py \
  --out-dir results/v17_code_audit/official_s0 \
  --device cuda:0
```

S0 输出：

```text
v17_compileall.log
v17_import_closure.csv
v17_linec_golden_results.csv
v17_update_semantics_tests.csv
v17_adamw_coupling_map.csv
v17_mechanism_noncollapse_summary.csv
v17_efficiency_profiler_unit_tests.csv
v17_kernel_gradcheck_summary.csv
v17_s0_route_decision.json
v17_code_review_packet.zip
```

若 S0 fail，不能执行 science run。

## Stage S1: Efficiency Truth + Repair Matrix

并行执行每个 basis 的 efficiency census 与 repair attempts。S1 不看 functional success，只看 basis 能否进入 MLP-like envelope。

## Stage S2: AdamW-coupling Decisive Matrix

每个 active carrier 比较：

```text
AdamW-primary + FU
SGD-primary + FU
Momentum-primary + FU
FU-primary
FU-only
Alternating FU / gradient
Schedule-free slow-state FU
Role-partition optimizer
```

目标：确认 AdamW 是否阻碍 FU。

## Stage S3: Full Functional Mechanism Matrix

执行 M1-M11。D-CHE / MLP full matrix；LQ/Rational/FOU/RBF/WAV minimum smoke。每个 positive-looking row 必须有 controls。

## Stage S4: Long-horizon retention

Top rows 执行：

```text
h = 100, 400, 800, 1600
```

如果 h800 source positive 而 h1600 消失，必须输出 source washout diagnosis，不允许只写 no-go。

## Stage S5: Final route / no-go / next queue

Finalizer 必须把失败拆成：

```text
CodeInvalid
LineCInvalid
UpdateSemanticsInvalid
AdamWCouplingConfound
SourceNotGenerated
SourceNotRetained
DebtNotRecovered
ControlEquivalent
KANSpecificAdvantageAbsent
EfficiencyForwardBlocked
EfficiencyBackwardBlocked
EfficiencyUpdateBlocked
EfficiencyMemoryBlocked
AuditCostPolluted
CarrierSubstrateBlocked
```

---

# 9. 不满足条件时 Codex 必须先尝试的方向

## 9.1 S0 import fail

Codex 必须补齐缺失 runner / compatibility shim，不允许跳过。必须输出：

```text
missing_module
required_by
restored_file
shim_file
unit_test
```

## 9.2 LineC golden fail

先修 LineC，不跑科学实验。检查：

```text
batch split 是否正确
probe split 是否泄漏
noise injection fixture 是否方向正确
NoOp/Random null false positive
exception 是否被误写成 fail
```

## 9.3 update sign fail

先修 update type system。必须证明：

```text
gradient_like handed to optimizer only
step_like direct add only
cotangent_like cannot commit directly
finite-difference descent direction correct
```

## 9.4 AdamW overwrite detected

如果 AdamW cumulative displacement 与 FU source 长期负相关，Codex 必须执行：

```text
SGD recovery
Momentum recovery
No optimizer recovery
FU-primary
slow-state FU
role-partition optimizer
```

不能只写 AdamW + FU no-go。

## 9.5 source h800 positive but h1600 disappears

Codex 必须执行：

```text
slow-state source retention
EMA / schedule-free state
matrix-block source retention
PopRisk/SNR signal-channel audit
AdamW overwrite diagnostic
```

## 9.6 MLP positive but KAN not positive

Codex 必须判断：

```text
FU is generic dynamics?
KAN carrier coordinate wrong?
Basis efficiency too slow / audit polluted?
D-CHE source washed out by AdamW?
```

并把 MLP 继续作为 active line，不允许降级成 control。

## 9.7 Basis forward blocked

Codex 必须做 family-specific repair，不允许新增 token 后重跑。例如 D-FOU forward blocked 时，先比较 recurrence / sincos precompute / fused band eval / table lookup。

## 9.8 Basis update blocked

Codex 必须拆 optimizer update 与 functional commit，尤其 LQ / D-CHE / Rational：

```text
foreach AdamW update
manual fused update
role-block update
projection update only
optimizer state update only
```

## 9.9 Audit cost polluted

Codex 必须把 train step 与 LineC/horizon/tail audit 分离，重新输出 `training_step_total` 与 `training_plus_audit_total`。

---

# 10. Required artifacts for v17 official completion

最低 required artifacts：

```text
v17_s0_route_decision.json
v17_code_review_packet.zip
v17_import_closure.csv
v17_linec_golden_results.csv
v17_update_semantics_tests.csv
v17_adamw_coupling_map.csv
v17_mechanism_noncollapse_summary.csv
v17_efficiency_truth_table.csv
v17_efficiency_phase_breakdown.csv
v17_memory_phase_breakdown.csv
v17_same_param_mlp_manifest.csv
v17_kernel_gradcheck_summary.csv
v17_line_a_dche_fu_matrix.csv
v17_line_b_mlp_fu_matrix.csv
v17_line_c_lq_reanchor_smoke.csv
v17_line_d_rational_monitor_smoke.csv
v17_line_e_allbasis_kernel_repair.csv
v17_line_f_allbasis_fu_smoke.csv
v17_line_m_attribution.csv
v17_h800_summary.csv
v17_h1600_summary.csv
v17_source_washout_diagnosis.csv
v17_adamw_overwrite_diagnostic.csv
v17_direction_provenance.csv
v17_forbidden_information_audit.csv
v17_no_action_search_audit.csv
v17_gpu_assignment_manifest.csv
v17_gpu_utilization_dashboard.csv
v17_idle_violation.csv
v17_queue_drain_report.csv
v17_failure_taxonomy.csv
v17_no_go_boundary.md
v17_next_hypothesis_queue.md
v17_required_artifact_manifest.csv
```

Required figures：

```text
fig_v17_s0_code_gate_dashboard.svg
fig_v17_linec_golden_null_distribution.svg
fig_v17_update_semantics_sign_tests.svg
fig_v17_adamw_overwrite_cosine.svg
fig_v17_source_retention_horizon_curves.svg
fig_v17_debt_recovery_curves.svg
fig_v17_carrier_mechanism_heatmap.svg
fig_v17_efficiency_forward_backward_update_bar.svg
fig_v17_memory_phase_breakdown.svg
fig_v17_same_param_mlp_efficiency_pareto.svg
fig_v17_gpu_utilization_timeline.svg
fig_v17_failure_taxonomy_heatmap.svg
```

---

# 11. v17 final decision rules

## 11.1 Code success

```text
S0-CodeValid = 1
```

requires：

```text
compileall_pass=1
import_error_count=0
LineC golden pass=1
update semantics pass=1
AdamW coupling map complete=1
efficiency profiler tests pass=1
kernel gradcheck pass=1
```

## 11.2 Functional success hierarchy

```text
S1 = execution coverage complete
S2 = weak productive dynamics
S3 = productive debt recovery
S4 = real-transfer exploration
S5 = official success
```

S5 不降低。

## 11.3 Efficiency success hierarchy

```text
E1 = complete truth table
E2 = exploration efficiency envelope
E3 = official efficiency envelope
E4 = efficiency + functional compatible
```

## 11.4 Project-level progress

真正项目进展必须是以下之一：

```text
1. FU source retained to h1600 and controls fail;
2. AdamW-free FU proves AdamW was suppressing source;
3. MLP-FU proves generic functional dynamics and gives transfer path;
4. a KAN carrier achieves KAN-specific advantage;
5. a basis family reaches MLP-like efficiency envelope;
6. LineC / metric bug is fixed and prior no-go is invalidated;
7. full no-go closes current FU family with valid code and efficiency evidence.
```

单纯 `route=no-go` 不算进展；单纯 h800 source 不算进展；单纯 efficiency pass 不算 functional progress。

---

# 12. 最终总结

v17 的核心不是再写一个新 FU 方法，而是先停止错误实现和错误指标继续污染结论。当前上传包显示：compile 能过，但 import closure 不完整；LineC 实现文件不在包内；LineC exception 可能被计成几何 fail；FU 多数路径仍与 AdamW 耦合；update 语义没有强类型；efficiency profiler 仍有 audit cost 污染风险。这些问题都足以让实验“原地打转”。

v17 因此必须先完成：

$$
\boxed{\text{S0 代码/指标/更新语义正确性闭环}}
$$

然后再做：

$$
\boxed{\text{AdamW-free / FU-primary / slow-state / matrix-block functional update full matrix}}
$$

以及：

$$
\boxed{\text{basis kernel efficiency census + family-specific repair}}
$$

最终，v17 的执行标准是：

```text
full carrier matrix
× full mechanism matrix
× mandatory efficiency census
× code correctness gates
× 4GPU dynamic queue
```

只有这样，下一轮结果才不会继续被错误指标、错误标准或错误实现带着原地打转。
