# DG-KAN v16.5：Mechanism-First Functional Update + Basis Kernel Efficiency Breakthrough + 4GPU 动态并行完整计划

> 版本：v16.5 execution plan  
> 生成时间：2026-06-02  
> 基于：v16.4.1 `FunctionalUpdate EfficiencyBreakthrough 4GPU` 实验结果复盘  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 核心修正：本版不再把“多 carrier + 一组代表 FU”当作充分探索；也不再把 efficiency census 当成已经解决效率。v16.5 把项目拆成两个必须同时突破的主问题：  
> 1. **Functional update 机制为什么不能变成长期有效训练动力学。**  
> 2. **每个 basis 为什么还没有进入 MLP-like forward / backward / update / memory envelope。**  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit / gate / debt readback，不能作为方向源；Rational 不重启 reset / controller / action route。

---

# 0. 项目总目标与当前真实状态

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找一个局部指标好看的 update。项目目标是：

$$
\boxed{
\text{在 strict FC-PureKAN base 上，通过 functional update 改善长期训练动力学，}
\text{最终得到比普通 AdamW / backprop 更好的模型。}
}
$$

这个目标同时包含两件事：

```text
A. Functional update 必须有独立长期价值：
   它不能只是 h20/h800 局部 source 正；
   必须在 h800/h1600 之后保留 source；
   必须偿还 tail / LineC / calibration / AUC debt；
   必须打过 AdamW / random / matched / NoOp / recovery-only / MLP/generic controls。

B. Basis / carrier 必须接近 MLP efficiency envelope：
   forward 不能远慢于 same-param MLP；
   backward 不能远慢于 same-param MLP；
   optimizer update 不能成为隐藏 bottleneck；
   functional direction / projection / audit cost 必须拆开；
   backward peak memory / optimizer state / basis activation / functional state 必须清楚。
```

v16.4.1 的真实结论是：执行合同更完整，但 scientific capability 没有推进。它完成了 efficiency truth table、functional smoke、D-CHE/MLP readback、KAN-specific attribution、queue drain 等边界；但最终 route 是：

```text
route = R4-FunctionalSourceNotRetained
minimum_success = S1-ExecutionCoverageCompleted
S2_weak_productive_dynamics_reached = 0
S3_productive_debt_recovery_reached = 0
official_s5_reached = 0
promotion_allowed = 0
```

因此 v16.5 的定位是：

$$
\boxed{
\text{不是继续补一个 FU 名字，}
\text{而是同时做 functional mechanism 机制裁决与 basis kernel efficiency 突破。}
}
$$

---

# 1. v16.4.1 实验结果独立分析

## 1.1 已经取得的真实进展

v16.4.1 的进展主要在执行与工程诊断，不是能力成功。

已完成：

```text
1. Efficiency truth table:
   rows = 114
   measured = 114
   exploration_pass = 6
   official_pass = 5

2. Functional smoke:
   rows = 48
   measured = 48
   complete = 1
   source >= 0.005 rows = 0

3. D-CHE / MLP full matrix:
   复用 v16.3 真实 artifacts，reuse_readback = 1；
   没有伪造 fresh v16.4.1 training。

4. KAN-specific attribution:
   line_m_rows = 55
   KAN_specific_advantage_rows = 0

5. Queue drain:
   idle_violation_count = 0
```

这说明 v16.4.1 没有明显浅尝辄止，也不是 artifact 缺失导致失败。它已经完成了计划要求的 execution coverage。

## 1.2 没有取得能力进展的原因

全局 best h800 source 不是 KAN，而是 MLP：

```text
best = B-M4a-MLP-AdamSubspaceProximal-alpha000
carrier = MLP
source_h800 = 0.14581246508492363
retention_h800 = 0.8296399083402421
tail_recovery_h800 = 0.01966376412104744
LineC_recovery_h800 = 0.0
AUCtime_h800 = 1.0373429473020996
h1600_mean_source = 0.0
```

这说明：

```text
1. h800 有 source，不是完全没信号；
2. source 出现在 MLP，不是 KAN-specific；
3. h1600 source 归零，说明它没有变成长期 retained dynamics；
4. tail / LineC debt 没有被偿还；
5. KAN_specific_advantage_rows = 0，不能写成 KAN functional progress。
```

所以 v16.4.1 真实结论是：

$$
\boxed{
\text{当前 FU family 能制造短中期 source，}
\text{但不能把 source 变成长期 retained productive dynamics。}
}
$$

## 1.3 basis efficiency 侧的新信息

v16.4.1 开始给出 phase-level truth table，但仍不是最终答案。已知信息说明：

```text
1. D-FOU / LQ 某些 rows 的 step / memory 可承受，但 forward 仍极慢。
2. D-CHE / D-RBF / D-WAV 多数 rows 被 forward/backward/update/audit cost 阻断。
3. blocked row 已经写 exact blocker class，例如：
   F7-EfficiencyForwardBlocked
   F8-EfficiencyBackwardBlocked
   F11-AuditCostPolluted
4. efficiency timing 只作为 engineering / basis evidence，不能作为 functional promotion。
```

这意味着 basis efficiency 的下一步不能只做 census，而必须进入 family-specific kernel repair。

---

# 2. 各条线当前进展百分比

| 线 | 当前完成度 | 当前判断 |
|---|---:|---|
| 代码 / provenance / no-action audit | 99% | 工程闭包强，不是当前 blocker |
| 4GPU execution contract | 88% | v16.4.1 queue/drain/idle 已闭合，但还需真正动态 job stealing |
| Efficiency truth table | 45% | rows=114 已测，但需要完整 per-family phase map + repair validation |
| D-CHE substrate | 85% | 仍是当前最强 KAN carrier，但 FU 与 efficiency 都没闭合 |
| D-CHE functional dynamics | 18%-25% | 有局部 source，但不能 h1600 retained；KAN-specific advantage = 0 |
| MLP functional dynamics | 35%-45% | 当前最强 source 来自 MLP；必须作为主动机制发现线 |
| LQ reanchor / late attach | 30%-40% | near/reanchor 线索强，但 gate 未完全打开；需要 promotion-disabled smoke |
| Rational monitor | 70%-80% substrate / 10%-15% functional | 只做 monitor/smoke；不重启 reset/controller/action |
| D-FOU substrate / efficiency | 35%-45% | best non-D-CHE 线索；step/memory 局部可承受，但 forward bottleneck 严重 |
| D-RBF / FastKAN | 20%-30% | task-health 与 efficiency 都不稳 |
| D-WAV | 15%-20% | 弱线索，低预算 smoke |
| Dynamic geometry / debt recovery | 25%-30% | 定义已修正，但 productive debt recovery 尚未出现 |
| Functional mechanism matrix | 35%-40% | D-CHE/MLP 比较完整，其它 carrier 仍不足 |
| Basis kernel repair | 10%-15% | 目前主要是 census；真正 family-specific repair 尚未完成 |
| Official S5 | 0% | 尚未达成 |
| 整体 next-gen MLP claim | 23%-31% | 有执行成熟度，但科学 claim 仍弱 |

---

# 3. 当前最本质的问题

## 3.1 Functional update 的本质问题

过去几轮已经排除很多表面解释：

```text
不是只缺 h800/h1600；
不是只缺 MLP active line；
不是只缺 controls；
不是只缺 multi-carrier；
不是只缺 efficiency census；
不是只缺 no-action audit。
```

真正问题是：

$$
\boxed{
\text{当前 FU 产生的 source 不能穿过 optimizer / training trajectory，}
\text{成为长期 retained source。}
}
$$

更直白地说：

```text
当前 FU 像是在某个 horizon 里制造了一点局部有效移动，
但后续训练没有把这个移动巩固成稳定优势。
```

这说明下一步不能继续只比较：

```text
source_h800 是否正。
```

必须把 FU 分解成：

```text
1. 方向是否正确？
2. 方向是否被 ordinary optimizer 覆盖或抵消？
3. 方向是否进入 signal channel？
4. 方向是否能被后续训练巩固？
5. 如果短期产生 debt，是否能被 recovery mechanism 偿还？
6. 这种现象是 MLP-generic 还是 KAN-carrier-specific？
```

## 3.2 Basis efficiency 的本质问题

basis efficiency 现在不能再笼统说“KAN 慢”。不同 family 的瓶颈不同：

```text
D-CHE:
  可能混合了 basis recurrence / functional split-gradient / LineC horizon readback / audit cost。
  必须把 training cost 与 audit readback cost 分开。

D-FOU:
  当前主要是 forward bottleneck。
  sin/cos、band eval、high-frequency branch、audit readback 是重点。

LQ:
  forward 与 optimizer update 都是 bottleneck。
  需要拆 projection update、persistent AdamW、fixed-frame late attach、foreach/fused update。

D-RBF:
  task-health 和 efficiency 都不稳。
  active-center occupancy、width condition、local K4、no-dense materialization 是重点。

D-WAV:
  support overlap / sparse support backward 仍弱。
  先 low-budget smoke，不进入 official proof。

Rational:
  denominator / derivative telemetry cost、branchless eval、optimizer state cost需要拆；
  不重启 reset/controller/action。
```

因此 v16.5 把 basis efficiency 从“记录 timing”升级成“每个 family 有明确 repair hypothesis”。

---

# 4. v16.5 总体策略

v16.5 同时推进两个主线，不允许互相等待。

```text
主线 A：Functional Update Mechanism Decisive Matrix
  目标：找出 FU 为什么不能长期 retain source；
  同时验证 D-CHE、MLP、LQ、Rational、D-FOU/RBF/WAV smoke。

主线 B：Basis Kernel Efficiency Breakthrough
  目标：每个 basis family 都给出 forward/backward/update/memory 的 blocker 类型与 repair 尝试；
  不能只给 blocked 表，必须有 repair-vs-baseline 对照。
```

四张 GPU 必须全部使用：

```text
GPU0:
  D-CHE functional full matrix + D-CHE efficiency repair。

GPU1:
  MLP functional full matrix + MLP reference efficiency.

GPU2:
  LQ reanchor/smoke + Rational monitor/smoke + LQ/Rational efficiency repair。

GPU3:
  D-FOU / D-RBF / D-WAV substrate + functional smoke + efficiency repair。
```

如果 `runnable_queue.csv` 非空但任何 GPU idle 超过 10 分钟，final route 不允许写 completed no-go。

---

# 5. Functional Update：多机制并行矩阵

## 5.1 Active carriers

```text
C1-D-CHE:
  full matrix M1-M10。

C2-MLP:
  full matrix M1-M10。
  MLP 是主动机制发现线，不是普通 control。

C3-LQ:
  reanchor R0-R10；
  不论 reanchor gate 是否打开，必须跑 promotion-disabled late-attach smoke：
    M1, M2, M4, M7, M9。

C4-Rational:
  monitor + promotion-disabled smoke：
    M1, M2, M3, M5, M9。
  不重启 reset/controller/action route。

C5-D-FOU:
  substrate repair + minimum smoke：
    M1, M2, M3, M4, M9。

C6-D-RBF/FastKAN:
  substrate repair + minimum smoke：
    M1, M3, M4, M9。

C7-D-WAV:
  substrate repair + low-budget smoke：
    M1, M3, M9。
```

## 5.2 Mechanism families

### M1：Productive Pulse + Recovery Dynamics

核心问题：

$$
\text{短期 bad update 是否能被后续训练偿还，并留下 source？}
$$

必须比较：

```text
pulse-once + AdamW recovery
pulse-once + LR cooldown
pulse-once + momentum damping
pulse-once + decoupled weight decay
pulse-once + EMA/SWA consolidation
pulse-every50 + AdamW recovery
early-only pulse
mid-only pulse
late-only pulse
```

记录：

```text
source_h1, source_h20, source_h100, source_h800, source_h1600
source_retention_h800/h1600
tail_debt_peak
tail_debt_final
tail_recovery_rate
LineC_debt_peak
LineC_debt_final
LineC_recovery_rate
AUCtime_h800/h1600
```

### M2：Split-Consensus Signal Metric

核心问题：

$$
\text{多个 train split 都同意的方向，是否更可能 transfer？}
$$

记录：

```text
split_count
consensus_snr
effective_rank
eigengap
projection_retention
B1_gain
B2_gain
B3_gain
random_subspace_control_gain
```

### M3：PopRisk / Drift-Diffusion SNR

核心问题：

$$
\text{coherent population signal 是否比局部 geometry proxy 更可靠？}
$$

公式：

$$
SNR_k=
\frac{\mu_k^2}{\sigma_k^2/(b-1)+\epsilon}
$$

记录：

```text
mu_norm
sigma_norm
snr_mean
snr_p90
snr_active_fraction
snr_update_cos_adam
snr_update_cos_fu
source_retention
debt_recovery
```

### M4：Function-Space Proximal

核心问题：

$$
\text{先在函数空间解小位移，再投回参数，是否比直接参数 update 更可靠？}
$$

固定：

```text
alpha grid = 0, 0.025, 0.05, 0.10, 0.20
no adaptive search
no controller
```

记录：

```text
selected_alpha
proximal_objective_B1
proximal_objective_B2
commit_norm
function_displacement_norm
source_h800/h1600
debt_recovery
```

### M5：Optimizer-Aware Aligned FU

核心问题：

$$
\text{FU 是否因为和 ordinary gradient / momentum / second moment 冲突而失败？}
$$

比较：

```text
raw FU
cautious FU
soft cautious FU
MGUP-style FU
AdamV-scaled FU
SophiaDiagLite FU
block-second-moment FU
```

记录：

```text
cos_fu_neg_grad
sign_agreement_fraction
anti_alignment_fraction
preconditioned_cos
second_moment_scaled_norm
curvature_clip_fraction
source_retention
debt_recovery
```

### M6：Carrier-Reparameterized FU

核心问题：

$$
\text{source/hazard 是否因为 carrier 坐标错而绑死？}
$$

D-CHE variants：

```text
low-degree signal / high-degree residual split
readout-basis decoupled carrier
orthogonal degree bank
output-Jacobian carrier
dual-bank signal/reservoir carrier
```

LQ variants：

```text
fixed frame late attach
fan-in output-scale reanchor
projection-update decoupled carrier
```

D-FOU/RBF/WAV variants：

```text
frequency-band carrier
active-center carrier
support-region carrier
```

### M7：Late-Attach Snapshot FU

核心问题：

$$
\text{FU 是否应该在训练阶段性 attach，而不是全程使用？}
$$

比较：

```text
attach at early checkpoint
attach at mid checkpoint
attach at loss plateau
attach at high-source low-debt window
attach once then remove
```

### M8：Recovery-Only Deconfound

核心问题：

$$
\text{收益是否其实来自 recovery mechanism 本身，而不是 FU？}
$$

比较：

```text
AdamW recovery only
LR cooldown only
decay recovery only
EMA/SWA recovery only
Lookahead recovery only
momentum damping only
NoOp matched overhead
random pulse + same recovery
```

### M9：Train-Split Transfer Operator

核心问题：

$$
\text{FU 是否能在一个 train split 上生成 update，并在另一个 train split 上产生同向收益？}
$$

必须记录：

```text
B1_gain
B2_gain
B3_gain
B2_minus_random
B2_minus_adam_parallel
split_transfer_rate
```

### M10：Path-Type Classifier Diagnostic

这不是 controller，不提交方向。它只做 post-hoc 诊断：

```text
FastGood
SlowBurnGood
RiskyHighSource
BadPath
HarmlessNull
ControlEquivalent
```

目的：判断当前 failure 是否是我们用二分类 good/bad gate 过粗。

---

# 6. Basis Efficiency：每个 family 的 repair 计划

## 6.1 Line P0：完整 truth table 必须先完成

输出：

```text
v165_efficiency_truth_table.csv
```

字段：

```text
family
candidate
batch_size
same_param_mlp_hidden
param_count
forward_ratio_vs_same_param_mlp
backward_ratio_vs_same_param_mlp
optimizer_update_ratio_vs_same_param_mlp
functional_direction_ratio_vs_same_param_mlp
functional_projection_ratio_vs_same_param_mlp
functional_commit_ratio_vs_same_param_mlp
linec_audit_ratio_vs_same_param_mlp
horizon_readback_ratio_vs_same_param_mlp
step_ratio_vs_same_param_mlp
forward_peak_memory_ratio
backward_peak_memory_ratio
optimizer_state_memory_ratio
basis_activation_bytes_ratio
functional_state_bytes_ratio
blocker_class
primary_bottleneck
audit_cost_separated
```

Hard rule：

```text
如果 v165_efficiency_truth_table.csv 缺失，final route invalid。
如果 same-param MLP reference 缺失，final route invalid。
如果 phase-level timing 缺失，final route invalid。
如果 audit cost 没有和 training cost 分开，final route invalid。
```

## 6.2 D-CHE repair

Hypothesis：

```text
D-CHE 当前慢，不只是 Chebyshev basis 慢；
functional split-gradient、degree telemetry、LineC horizon readback、audit cost 混在一起。
```

必须执行：

```text
CHE-E1-training-only-no-LineC
CHE-E2-degree-recurrence-cache
CHE-E3-low-degree-active-bank
CHE-E4-fused-degree-role-projection-update
CHE-E5-no-materialize-degree-energy-readback
CHE-E6-audit-cost-separated
```

判断：

```text
若 training-only 过而 with-audit 不过：
  audit cost pollution 是主 blocker。

若 training-only 仍不过：
  basis recurrence / backward kernel 是主 blocker。
```

## 6.3 D-FOU repair

Hypothesis：

```text
D-FOU 主要是 forward bottleneck，不是 memory bottleneck。
```

必须执行：

```text
FOU-E1-low-frequency-recurrence
FOU-E2-sincos-precompute-shape
FOU-E3-fused-band-k-small
FOU-E4-high-frequency-quarantine
FOU-E5-table-lookup-sincos
FOU-E6-trig-vs-recurrence-table
```

记录：

```text
sin_cos_eval_time
band_eval_time
frequency_count
high_freq_fraction
forward_ratio
backward_ratio
step_ratio
memory_ratio
```

## 6.4 LQ repair

Hypothesis：

```text
LQ 不是显存爆；
主要是 forward + optimizer update path 慢。
```

必须执行：

```text
LQ-E1-update-decomposition
LQ-E2-persistent-adamw-foreach
LQ-E3-fused-projection-update
LQ-E4-fixed-frame-late-attach
LQ-E5-no-rebuild-frame
LQ-E6-projection-cache-reuse
```

记录：

```text
projection_forward_time
projection_update_time
frame_rebuild_time
optimizer_update_time
foreach_update_time
fixed_frame_vs_rebuild_delta
```

## 6.5 D-RBF / FastKAN repair

Hypothesis：

```text
RBF/FastKAN 的主要问题是 active center / width condition / dense materialization。
```

必须执行：

```text
RBF-E1-active-center-occupancy
RBF-E2-width-condition-guard
RBF-E3-local-K4-no-dense
RBF-E4-table-lookup-gaussian
RBF-E5-center-update-separation
RBF-E6-width-frozen-smoke
```

## 6.6 D-WAV repair

Hypothesis：

```text
Wavelet 主要卡在 sparse support backward 与 support overlap。
```

必须执行：

```text
WAV-E1-triangular-index-forward
WAV-E2-sparse-support-backward
WAV-E3-support-overlap-damping
WAV-E4-scale-occupancy-monitor
WAV-E5-index-only-readback
```

## 6.7 Rational repair

Hypothesis：

```text
Rational 主要是 denominator / derivative telemetry 与 optimizer-state cost。
```

必须执行：

```text
RAT-E1-branchless-denominator-eval
RAT-E2-branchless-denominator-readback
RAT-E3-horner-polynomial-eval
RAT-E4-optimizer-state-cost
RAT-E5-denominator-safety-no-reset
```

禁止：

```text
reset route
controller
action bank
state reset variants
```

---

# 7. 4GPU dynamic execution contract

必须生成：

```text
v165_runnable_queue.csv
v165_gpu_assignment_manifest.csv
v165_gpu_utilization_dashboard.csv
v165_idle_violation.csv
v165_deferred_items.csv
v165_queue_drain_report.csv
```

## 7.1 GPU assignment

```text
GPU0:
  D-CHE M1-M10 + CHE efficiency repair

GPU1:
  MLP M1-M10 + MLP same-param references

GPU2:
  LQ reanchor/smoke + Rational monitor/smoke + LQ/RAT efficiency repair

GPU3:
  D-FOU/D-RBF/D-WAV substrate + smoke + efficiency repair
```

## 7.2 Dynamic fill rule

如果某 GPU primary queue 完成，必须按顺序抢占：

```text
1. Line M controls
2. Line P efficiency missing rows
3. all-basis smoke rows
4. h1600 extension for top source-retaining rows
5. figure / audit generation
```

如果 runnable queue 非空但 GPU idle 超过 10 分钟：

```text
execution_contract_fail = 1
final_completed_no_go_allowed = 0
```

---

# 8. 记录指标

## 8.1 Functional metrics

```text
source_vs_best_control_h20/h100/h800/h1600
source_retention_h800/h1600
tail_debt_peak
tail_debt_final
tail_recovery_rate
LineC_debt_peak
LineC_debt_final
LineC_recovery_rate
calibration_debt_peak
calibration_debt_final
AUCtime_h800/h1600
control_equivalent_fraction
KAN_specific_delta_vs_best_MLP
random_pulse_explains
recovery_only_explains
NoOp_overhead_explains
```

## 8.2 Efficiency metrics

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

forward_peak_memory
backward_peak_memory
optimizer_state_memory
basis_activation_bytes
functional_state_bytes
audit_state_bytes
```

## 8.3 Geometry audit metrics

```text
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
NLL_delta
ECE_delta
Brier_delta
margin_p10_delta
signal_channel_energy
reservoir_energy
noise_leakage_proxy
```

注意：

```text
LineC / CEp99 / NLL / ECE / Brier / AUCtime 只能作为 audit / readback / gate。
它们不能用于生成 functional direction。
```

---

# 9. 可视化要求

必须生成：

```text
fig_01_progress_by_line.svg
fig_02_best_source_retention_vs_debt.svg
fig_03_h800_to_h1600_source_collapse.svg
fig_04_KAN_vs_MLP_attribution_matrix.svg
fig_05_efficiency_forward_backward_update_memory_heatmap.svg
fig_06_efficiency_phase_stacked_bar_by_family.svg
fig_07_audit_cost_vs_training_cost.svg
fig_08_basis_family_bottleneck_waterfall.svg
fig_09_4gpu_utilization_timeline.svg
fig_10_queue_drain_report.svg
fig_11_functional_mechanism_failure_taxonomy.svg
fig_12_basis_efficiency_failure_taxonomy.svg
```

---

# 10. Success / no-go gates

## S1：execution coverage success

必须同时满足：

```text
D-CHE M1-M10 complete
MLP M1-M10 complete
LQ reanchor + smoke complete
Rational monitor + smoke complete
D-FOU/RBF/WAV substrate + smoke complete
efficiency truth table complete
4GPU queue drain complete
matched controls complete
required / forbidden / no-action audit pass
```

## S2：weak productive dynamics

```text
source_vs_best_control_h800 >= 0.005
source_retention_h800 >= 0.40
tail_recovery_rate_h800 >= 0.40
or LineC_recovery_rate_h800 >= 0.40
AUCtime_ratio_h800 <= 1.10
matched controls fail
```

## S3：productive debt recovery

```text
source_vs_best_control_h800 >= 0.005
source_retention_h800 >= 0.50
tail_recovery_rate_h800 >= 0.60
LineC_recovery_rate_h800 >= 0.60
AUCtime_ratio_h800 <= 1.05
random pulse + same recovery fails
recovery-only fails
NoOp overhead fails
```

## S4：real-transfer exploration

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail debt recovered
LineC debt recovered
controls fail
```

## S5：official success

不降低：

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
forbidden audit pass
no-action-search audit pass
code review pass
promotion_allowed = 1
```

---

# 11. Stop / continue rules

## 11.1 不能 hard stop 的情况

```text
D-CHE fail
MLP fail
LQ reanchor fail
Rational smoke fail
D-FOU/RBF/WAV <6/9
M1 fail
M2 fail
M3 fail
M4 fail
M5 fail
M6 fail
M7 fail
M8 fail
M9 fail
M10 fail
h800 fail
h1600 fail
MLP/generic controls positive
LineC/tail/AUC immediate fail
overhead high
random pulse explains result
efficiency blocked but repair row not executed
```

这些只能进入：

```text
failure taxonomy
fallback ladder
exhaustion certificate
next hypothesis queue
```

## 11.2 真正 hard stop

只有这些可以 hard stop：

```text
required artifact missing
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

# 12. Codex 执行要求

Codex 必须在实验复盘文件中写明：

```text
1. 本轮新增代码和复用代码在哪些文件。
2. 每个 core update / metric / efficiency repair 的实现文件和函数名。
3. 每个 function direction 使用了哪些信息。
4. 是否使用 validation/test/future/query。
5. 是否使用 LineC/tail/AUC/calibration 作为 direction。
6. 每个 basis 的 forward/backward/update/memory blocker。
7. 每张 GPU 执行了哪些 job。
8. 哪些 job deferred，deferred 的 exact reason。
9. 是否存在 runnable queue 非空但 GPU idle。
10. 如果 route no-go，no-go 是来自 functional、efficiency、controls、attribution、还是 artifact。
```

Codex 不许：

```text
1. 新增 action token。
2. 启动 controller。
3. 使用 action bank。
4. reset route。
5. 按 dataset / seed 写分支。
6. 用 LineC / CEp99 / NLL / ECE / AUCtime / Brier 生成方向。
7. 把 low-budget smoke 写成 official FU proof。
8. 把 efficiency pass 写成 functional success。
9. 把 MLP generic positive 写成 KAN-specific success。
10. 把 blocked row 写成未测。
```

---

# 13. 本轮预期结论形式

v16.5 的最终结果必须落入以下之一。

## Case A：KAN FU 真正打开

```text
route = S4/S5-KANFunctionalProductiveDynamics
条件：
  D-CHE 或其它 KAN carrier 达到 S4/S5；
  MLP/generic controls 不能解释；
  efficiency gate 至少 exploration pass。
```

## Case B：MLP FU 打开，KAN 没打开

```text
route = R-B-GenericMLPFunctionalDynamicsPositiveKANCarrierFail
结论：
  FU 机制可能是 generic training-dynamics insight；
  KAN carrier 仍需 substrate / coordinate / efficiency repair。
```

## Case C：Basis efficiency 打开，FU 没打开

```text
route = R-C-EfficientKANCarrierFoundFunctionalStillNoGo
结论：
  找到 MLP-like basis carrier；
  functional update 仍需新机制。
```

## Case D：FU 与 efficiency 都没打开，但 blocker 明确

```text
route = R-D-MechanismAndEfficiencyBlockersSeparated
结论：
  本轮价值在于拆清 functional blocker 与 basis blocker；
  下一轮必须按 blocker-specific plan，不允许继续混写。
```

## Case E：执行合同失败

```text
route = R0-ExecutionContractFail
触发：
  required artifact missing；
  efficiency truth table incomplete；
  GPU idle while runnable queue nonempty；
  forbidden/no-action violation。
```

---

# 14. 一句话总结

v16.5 的目标不是继续“跑一个 best row 再 no-go”。它要同时回答两个硬问题：

$$
\boxed{
\text{Functional update 是否能形成长期 retained productive dynamics？}
}
$$

以及：

$$
\boxed{
\text{哪个 basis family 能真正接近 same-param MLP 的 forward/backward/update/memory envelope？}
}
$$

如果这两个问题任何一个仍然回答不清，就不允许写 vague no-go。必须给出明确 blocker：

```text
source retention blocker
debt recovery blocker
control-equivalence blocker
KAN-specific attribution blocker
forward kernel blocker
backward kernel blocker
optimizer update blocker
audit-cost pollution blocker
memory-state blocker
```

只有这样，下一轮才不是原地打转。
