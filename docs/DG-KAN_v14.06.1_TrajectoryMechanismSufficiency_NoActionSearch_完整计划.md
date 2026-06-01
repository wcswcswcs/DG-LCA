# DG-KAN v14.6.1：Trajectory Mechanism Sufficiency Test + All-Basis Parallel Plan

> 版本：v14.6.1 修正版  
> 目的：修正 v14.6 中“action bank / action generation”容易重蹈旧路的问题。  
> 核心原则：**不再寻找好动作。Action bank 只能作为一次性 upper-bound 诊断工具，不能作为候选搜索空间。**  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no label-informed initialization；functional direction 不使用 validation/test/future/query；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit / gate，不能生成方向；不能把 MLP-only positive 写成 KAN success；不能把 diagnostic oracle 写成 promotion。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的总目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是继续构造局部好看的 update rule。项目目标是：

$$
\boxed{
\text{label-free strict FC-PureKAN substrate + functional update}
>
\text{same substrate + ordinary AdamW / controls}
}
$$

其中 functional update 必须满足：

```text
1. 不针对 CE 写死，不用 CEp99 / NLL / ECE / LineC / AUCtime 作为 direction source；
2. 可以通过统一 loss interface 使用当前训练 batch 的 output cotangent；
3. 必须使用 train-stream precommit 信息；
4. 必须击败 NoOp / RandomMatchedNorm / AdamWParallel / matched overhead / generic MLP controls；
5. 必须在 real 3x3 上稳定：MNIST、Fashion-MNIST、KMNIST × seeds 0,1,2；
6. 必须保持表达力、效率、AUCtime、tail、calibration 和 Manifold-Channel 几何不坏。
```

最终要证明的是：

$$
\boxed{
\text{functional update 本身改变训练过程，}
\text{而不是靠局部动作、seed 特例或 audit 指标调参。}
}
$$

## 0.2 当前真实状态

截至 v14.5，项目状态是：

```text
Rational-FMS synthetic S3:
  已经成立。

Rational-FMS real S4b:
  已经打开。

v14.4 best actual real 3x3:
  6 / 9 dataset-seed pass。

v14.5 diagnostic oracle:
  core action bank upper bound = 6 / 9；
  extended legal v14.4/v14.5 bank upper bound = 7 / 9。

controller:
  没有执行，因为 oracle upper bound < 9/9。

S5:
  仍然为 0。
```

最重要的新结论不是“再找好动作”，而是：

$$
\boxed{
\text{现有合法 action family 的 upper bound 不足，}
\text{controller / selection policy 不是当前第一 blocker。}
}
$$

剩下未覆盖的主要状态是：

```text
Fashion-MNIST seed2
KMNIST seed2
```

其 failure 主要表现为：

```text
source / tail / LineC 可局部通过；
但 AUCtime 仍 > 1.0。
```

因此当前 blocker 更准确地说是：

$$
\boxed{
\text{AUCtime-only trajectory-cost states 没有被现有 functional family 覆盖。}
}
$$

## 0.3 为什么必须修正 v14.6

原 v14.6 名称里有：

```text
Trajectory-Generating Functional Actions
```

这个说法有风险。它容易让 Codex 回到 v12 以前的老路线：

```text
看到局部 positive row；
发明一个相似 action；
再调 strength / scale / trust / interval；
最后得到更多局部 positive，但 P4/S5 不转化。
```

这版 v14.6.1 明确改为：

$$
\boxed{
\text{Trajectory Mechanism Sufficiency Test}
}
$$

不是：

$$
\boxed{
\text{Action Search}
}
$$

---

# 1. 我们必须记住的旧错误

## 1.1 旧错误 A：把局部 P3 / diagnostic positive 当作动作价值

过去多轮失败都来自同一模式：

```text
1. 构造一个 action；
2. 某个 seed / metric / P3 row 看起来好；
3. 围绕它扩展 scale / sign / role / window / damping；
4. 进入 P4 后不转化；
5. 再找下一个 action。
```

v12.11 / v12.12 已经说明旧 P3 score 与 P4 gain 几乎不相关；LineC score 虽略好但仍不足以 promotion；v12.13-v12.16 又说明 CouplingR2 可移动，但 NoiseSignalLeak / ReservoirRatio / control gap 不闭合。

因此：

$$
\boxed{
\text{局部 positive row 不再允许触发新 action 设计。}
}
$$

## 1.2 旧错误 B：把 action bank 当搜索空间

v14.5 的 action bank 只能回答：

```text
现有已执行合法 methods 的 oracle upper bound 是多少？
```

它不能回答：

```text
下一步该扩哪些 action？
```

因此 action bank 在本计划中只允许有一次性诊断用途：

$$
\boxed{
\text{ActionBank} = \text{Upper-bound diagnostic artifact}
}
$$

不是：

$$
\boxed{
\text{ActionBank} = \text{Candidate pool / Search space / Controller training set}
}
$$

## 1.3 旧错误 C：在 upper bound 不足时调 controller

如果 oracle 从现有 action bank 中逐 dataset-seed 选最佳也不到 9/9，那么 train-stream controller 再聪明也不能超过这个上界。

因此：

```text
If oracle_bank_pass < 9/9:
  controller_executed = 0
  controller_tuning_forbidden = 1
```

v14.5 已经触发了这个条件：

```text
extended_legal_bank_oracle = 7/9
controller_executed = 0
```

本轮不能回头调 controller。

## 1.4 旧错误 D：把 audit 指标变成 direction

这些指标只能用于审计或 gate：

```text
LineC
CEp99
NLL
ECE
AUCtime
Brier
validation/test performance
future outcome
```

不能用它们生成 functional direction，也不能用它们作为 train-time action selection 的输入。

---

# 2. 当前各线进展百分比

这些百分比是根据当前 gate 状态、实现成熟度、机制清晰度和离 official success 的距离估计，不是 artifact 中的官方字段。

| 线 | 当前完成度 | 判断 |
|---|---:|---|
| 代码 / provenance / finalizer 审计 | **99%** | 工程闭包强，不是当前 blocker |
| Historical FHQ / B320-current | **85% frozen** | 历史强，但 label-informed init 已禁用，不能 official |
| Line C 几何审计 | **88%** | 审计稳定；当前失败基本不是 LineC 主导 |
| PopRisk / FMS 实现面 | **93%** | persistent state、per-example gradient、projection trace、real runner 已成熟 |
| Generic MLP-FMS | **45%** | 有 generic signal，但不能写成 KAN success |
| Rational substrate | **85%** | 当前最稳定 substrate |
| Rational-FMS synthetic | **70%** | 已达 S3，KAN-specific synthetic signal 成立 |
| Rational-FMS real-transfer | **62%** | 已达 S4b；actual best 6/9，diagnostic oracle best 7/9，但 S5=0 |
| Action-bank diagnostic | **70% diagnostic / 0% mechanism success** | 已证明现有 bank upper bound 不足 |
| Controller / policy | **0%** | 按 gate 不允许启动 |
| Trajectory mechanism diagnosis | **0%-10%** | v14.6.1 新主线，尚未执行 |
| Wavelet substrate | **35%-40%** | 有线索，但 v14.5 best 仍不足以 official FMS proof |
| Fourier substrate | **35%-40%** | 当前较有希望的 Non-RAT 方向，single-config 3/9，audit best 4/9 |
| RBF / FastKAN substrate | **25%** | compact workspace 有信号，但 task-health collapse |
| Chebyshev substrate | **25%** | compact workspace 有信号，但 task-health / LineC 未闭合 |
| Non-RAT official FMS | **0%-5%** | 没有 full 3x3 substrate pass，不能 official proof |
| Official S5 functional success | **0%** | 尚未达成 |
| 整体 next-gen MLP claim | **43%-50%** | real S4b 是真进展，但现有 action family upper bound 不足 |

---

# 3. 本轮核心假设

## H1：剩余 2/9 不是“选错动作”，而是 trajectory mechanism 缺失

v14.5 已经证明现有 action bank oracle 只有 7/9。因此剩余失败不能先归因于 action selection。

本轮要验证：

$$
\boxed{
\text{Fashion-MNIST seed2 与 KMNIST seed2 是否属于 AUCtime-only trajectory-cost states。}
}
$$

如果它们不是 AUCtime-only，而是存在隐藏 source / tail / LineC failure，那么 trajectory mechanism 方向也不成立。

## H2：AUCtime-only failure 可能来自 optimizer-state mismatch

FMS event 改变参数后，AdamW 的 moment / RMS state 可能仍对应旧参数坐标，导致之后若干 step 出现 recovery lag。

要测：

```text
post_event_update_cos_with_grad
post_event_moment_staleness
rms_state_mismatch
recovery_lag_steps
loss_spike_after_event
```

如果这些指标在 missing states 上显著高于 passed states，则 optimizer-state transport / reset 是机制候选。

## H3：AUCtime-only failure 可能来自 forced functional commit

如果某些 state 下 NoOp / AdamW 本身已经是最优轨迹，强制 commit FMS event 会增加 AUC debt。

要测：

```text
NoOp dominates action?
FMS event causes train-stream loss spike?
event benefit appears only at endpoint but hurts integral?
```

如果是，则需要 NoOp-aware commit 机制，而不是继续造更强 FMS action。

## H4：AUCtime-only failure 可能来自 delayed value mismatch

有些 state 的 generic drift 要先累积，basis projection 太早会拖慢训练轨迹。

要测：

```text
early projection vs delayed projection
early source gain
mid trajectory integral
late endpoint improvement
```

如果 delayed projection consistently lowers AUC debt without audit leakage，才允许机制化。

## H5：如果 mechanism probes 都不能专门修复 AUCtime-only states，则 Rational-FMS 当前路线 upper bound 是 7/9

这必须作为可接受的结论，而不是继续扩 K-token。

---

# 4. v14.6.1 总体实验设计

v14.6.1 分成五条线：

```text
Line R:
  Implementation / provenance / anti-action-search audit。

Line T:
  Missing-state trajectory autopsy，不新增 action。

Line P:
  Minimal mechanism sufficiency probes，不做 action search。

Line D:
  All-basis parallel substrate task-health repair。

Line Z:
  Finalizer / route / no-go / next hypothesis。
```

本轮最小成功不是 S5，而是明确回答：

$$
\boxed{
\text{剩余 2/9 是否能由一种合法 train-stream trajectory mechanism 修复？}
}
$$

---

# 5. Line R：Implementation / Anti-Action-Search Audit

## 5.1 目标

确保 Codex 没有把本轮变成“寻找好动作”。

## 5.2 必须生成 artifact

```text
v1461_code_review_manifest.csv
v1461_forbidden_information_audit.csv
v1461_action_search_violation_audit.csv
v1461_existing_bank_lock_manifest.csv
v1461_mechanism_probe_manifest.csv
v1461_required_manifest.csv
```

## 5.3 Action-search violation 判定

如果出现以下任一情况，route 直接为：

```text
R0-ActionSearchViolation
```

违规项包括：

```text
1. 新增 K-RT8 / K-AUC5 / K-FL3 这种 scalar/token action；
2. 因某个 row 局部 positive 而新增相似 action；
3. 调 strength/lambda/lr/refresh/tail-trust/projection-threshold 小网格；
4. controller 在 oracle < 9/9 时执行；
5. 使用 CEp99/NLL/ECE/LineC/AUCtime audit metric 生成 direction；
6. 使用 dataset-name / seed / validation / test / future / query 生成 direction；
7. 把 diagnostic oracle 或 missing-state probe 写成 promotion；
8. 拼接不同 runs 的局部 positive 宣称 9/9。
```

## 5.4 Existing bank lock

读取 v14.4/v14.5 已执行 rows，仅用于诊断：

```text
K0 / NoOp / AdamW baseline
K8 / lowplasticity
K-RT1..K-RT7
single-refresh
slow-refresh
v14.4 legal repair rows
v14.5 legal repair rows
```

这些 rows 不能被扩展成新 action。必须写入：

```text
row_id
source_artifact
method_name
dataset
seed
metrics_hash
legal_for_diagnostic
promotion_allowed=0
```

---

# 6. Line T：Missing-State Trajectory Autopsy

## 6.1 目标

不新增 action，只解释 v14.5 未覆盖的 states 为什么失败。

重点 states：

```text
Fashion-MNIST seed2
KMNIST seed2
```

对照 states：

```text
pass states:
  oracle bank 已覆盖的 7/9。

near-fail states:
  source / tail / LineC 接近但 AUCtime fail 的 rows。

NoOp / AdamW states:
  相同 dataset-seed 下 baseline 轨迹。
```

## 6.2 记录指标

每个 state / method / window 记录：

```text
dataset
seed
method
source_vs_best_control
AUCtime_ratio
CEp99_delta
NLL_delta
ECE_delta
LineC_pass
CouplingR2_delta
NoiseSignalLeak_delta
ReservoirRatio_delta
step_time_ratio
overhead_ratio
```

新增 trajectory decomposition：

```text
train_loss_integral_early
train_loss_integral_mid
train_loss_integral_late
loss_spike_after_event
loss_recovery_lag_steps
endpoint_source_gain
source_integral_gain
AUC_debt
NoOp_dominance_margin
AdamW_dominance_margin
```

定义：

$$
AUCDebt(a)=\int_0^T [L_a(t)-L_{best}(t)]_+dt.
$$

对离散 steps：

$$
AUCDebt(a)=\sum_{t=1}^{T} [L_a(t)-L_{best}(t)]_+\Delta t.
$$

Optimizer-state mismatch：

```text
adam_moment_cos_before_after
adam_rms_ratio_before_after
post_event_update_cos_grad
post_event_update_cos_adamw
moment_staleness_score
rms_staleness_score
```

定义：

$$
MomentStale =
1-\cos(m_{Adam,t}, g_{post,t}).
$$

$$
RMSMismatch =
\left|
\log
\frac{\|v_{Adam,t}^{1/2}\|}{\|g_{post,t}\|+\epsilon}
\right|.
$$

## 6.3 Missing-state 分类

每个 missing state 必须被分类为以下之一：

```text
C1-AUCOnlyTrajectoryCost:
  source/tail/LineC pass，但 AUCtime fail。

C2-HiddenSourceFail:
  endpoint source positive，但 source_integral negative。

C3-HiddenTailRecoveryLag:
  CEp99/NLL/ECE endpoint acceptable，但 early/mid tail debt 高。

C4-OptimizerStateMismatch:
  moment_staleness / rms_mismatch 高于 pass states。

C5-OverheadDominated:
  raw source OK，但 overhead / step_time 造成 AUCtime fail。

C6-AuditConflict:
  AUCtime fail 与 source/tail/LineC 多重冲突，不是 trajectory-only。

C7-NotEnoughData:
  artifact 不足，必须补 profile，不许新增 action。
```

## 6.4 Line T 判定

Line T 通过机制探针前置条件：

```text
至少一个 missing state 被判为 C1 或 C4；
且 pass states 中同类指标不普遍存在；
且 audit 不显示 LineC/CEp99/NLL/ECE 是主 blocker。
```

若不是：

```text
route = R1-MissingStateNotTrajectoryMechanism
```

并禁止 Line P。

---

# 7. Line P：Minimal Mechanism Sufficiency Probes

## 7.1 原则

Line P 不是 action search。它只允许执行预注册的少数机制 probe，用来验证 Line T 中提出的机制假设。

每个 probe 的目标不是直接 S5，而是回答：

```text
这个机制是否能专门减少 missing states 的 AUCDebt，
同时不破坏已通过的 7/9？
```

如果 probe 只在某个 seed 局部好，不允许扩展为新 action token。

## 7.2 P1：Optimizer-State Transport Probe

### 目标

验证 FMS event 后 AdamW state mismatch 是否造成 AUCtime debt。

### 方法

对同一个 FMS parameter delta，比较：

```text
P1a: no state transport
P1b: zero moment reset on affected parameter groups
P1c: partial moment interpolation
P1d: RMS state recompute from train-stream microbatch
P1e: moment transport by projected post-event gradient
```

这些不是新 direction。参数 delta 不变，只测试 optimizer-state handling。

### 记录

```text
state_transport_mode
affected_param_fraction
moment_staleness_before
moment_staleness_after
rms_mismatch_before
rms_mismatch_after
AUCDebt_before
AUCDebt_after
source_change
tail_change
LineC_change
pass_state_regression_count
```

### Gate

允许进入 next mechanism only if：

```text
missing_state_AUCDebt_reduction >= 20%
pass_state_regression_count = 0
source_vs_control_drop <= 0.002
CEp99/NLL/ECE non-harm
LineC non-harm
```

如果不满足，记录：

```text
P1-OptimizerStateMismatchNotSufficient
```

## 7.3 P2：NoOp-Aware Commit Probe

### 目标

验证 forced commit 是否制造 AUCtime debt。

### 方法

不新增新 update。只比较：

```text
commit FMS event
commit NoOp
commit AdamW-only
```

使用 train-stream split B1/B2 做 micro-horizon proxy，仅用于判断“是否 commit”，不是选择多个动作。

Train-stream proxy：

```text
B1: apply candidate event + 1 micro step；
B2: measure train loss integral, q95 loss, margin p10, logit RMS, entropy, split agreement。
```

### 记录

```text
micro_horizon_loss_integral
micro_horizon_q95_loss
micro_horizon_margin_p10
split_agreement
commit_decision
actual_AUCDebt
actual_source
actual_tail
```

### Gate

NoOp-aware commit 允许继续 only if：

```text
它能在 missing states 中选择 NoOp / AdamW-only 避免 AUCDebt；
同时不拒绝原本通过的 source-positive states；
overall diagnostic upper bound improves from 7/9 to >=8/9。
```

如果只能改善一个 seed 但破坏其他 passed seed：

```text
P2-NoOpPolicyUnstable
```

## 7.4 P3：Micro-Horizon Loss-Integral Probe

### 目标

验证 AUCtime fail 是否能用 train-stream micro-horizon loss integral 预测。

### 方法

对 fixed existing legal event，不改 direction，只测：

$$
I_H=\sum_{h=1}^{H} L_{train}(t+h).
$$

比较：

```text
H = 1,2,4 micro steps
B1/B2 split
NoOp / AdamW / existing FMS event
```

### 记录

```text
micro_horizon_H
I_H_B1
I_H_B2
proxy_rank
actual_AUCtime_rank
rank_correlation
false_accept_count
false_reject_count
```

### Gate

```text
Spearman(proxy_rank, actual_AUCtime_rank) >= 0.60
false_accept_count <= 1 across 9 states
```

否则不能用于 controller。

## 7.5 P4：Delayed Projection Probe

### 目标

验证 basis projection 太早是否造成 trajectory lag。

### 方法

比较：

```text
immediate projection
delayed projection after k AdamW/FMS generic steps
generic first, basis constraint later
```

其中 $k$ 只允许预注册：

```text
k = 1, 2, 4
```

不允许按 dataset / seed 调整。

### 记录

```text
delay_k
early_source_integral
mid_source_integral
late_endpoint_source
AUCDebt
tail_debt
LineC_debt
pass_state_regression
```

### Gate

```text
missing_state_AUCDebt_reduction >= 20%
no pass-state regression
source/tail/LineC non-harm
```

---

# 8. Line O：Mechanism Coverage Certificate

## 8.1 目标

Line O 不再叫 action bank oracle，而叫：

```text
Mechanism coverage certificate
```

它回答：

```text
预注册机制 probe 是否把 upper bound 从 7/9 提高到 9/9？
```

## 8.2 输入

仅允许以下 rows：

```text
v14.5 existing bank rows
Line P mechanism probe rows that passed mechanism gate
NoOp / AdamW baselines
```

不允许：

```text
未执行 imagined rows
seed-specific scale
dataset-specific branch
audit-target direction
```

## 8.3 输出

```text
v1461_mechanism_coverage_certificate.csv
```

字段：

```text
dataset
seed
best_existing_bank_pass
best_mechanism_probe_pass
coverage_before
coverage_after
best_legal_mechanism
dominant_failure_after
oracle_uses_audit_only
promotion_allowed=0
```

## 8.4 判定

```text
coverage_after < 9/9:
  route = R2-MechanismUpperBoundInsufficient
  controller_executed = 0

coverage_after = 9/9:
  route = S4c-MechanismSufficientDiagnostic
  controller may be evaluated in a separate line, promotion still 0
```

---

# 9. Optional Line K：Train-Stream Commit Policy

## 9.1 进入条件

只有当：

```text
Line O coverage_after = 9/9
```

才允许执行。

## 9.2 限制

Commit policy 不是 action search。它只能在以下固定 choices 中选择：

```text
NoOp
AdamW-only
one validated mechanism probe
```

不能选择一堆 K-token 变体。

## 9.3 输入 features

只能使用 train-stream precommit features：

```text
train split B1/B2 loss
train q95 loss
margin p10
logit RMS
entropy
split agreement
per-example gradient SNR
AdamW/FMS state
optimizer-state mismatch telemetry
micro-horizon proxy
```

不能使用：

```text
validation/test
future
LineC
CEp99/NLL/ECE audit
AUCtime audit
dataset name
seed id
```

## 9.4 Gate

Exploration：

```text
controller_real_pass >= 7/9
```

Official：

```text
controller_real_pass = 9/9
```

如果 coverage_after=9/9 但 controller<7/9：

```text
R3-CommitPolicyFail
```

---

# 10. Line D：All-Basis Parallel Substrate Task-Health Repair

## 10.1 目标

Functional 不应让 all-basis 线停掉。Non-RAT 不能进入 official FMS proof，除非先成为 substrate。

## 10.2 Active families

```text
Rational:
  主 functional carrier，监控 no-regression。

Fourier:
  当前最有希望的 Non-RAT substrate，继续 low-frequency identity residual / band occupancy。

RBF / FastKAN:
  compact workspace 已有信号，重点修 task collapse。

Chebyshev:
  compact workspace 已有信号，重点修 degree-energy / task-health / incremental memory。

Wavelet:
  有历史线索，但 v14.5 best 仍不足，继续 scale/support hardening。

B-spline:
  frozen，不进入 active budget。
```

## 10.3 Substrate gate

进入 official FMS proof 前必须满足：

$$
step\_ratio \le 1.75
$$

$$
memory\_ratio \le 1.75
$$

$$
mean\_delta\_vs\_MLP \ge -0.05
$$

$$
worst\_delta\_vs\_MLP \ge -0.10
$$

$$
AUCtime\_ratio \le 2.0
$$

$$
LineC\_pass\_rate \ge 0.30
$$

Non-RAT 若不过 substrate gate：

```text
functional_official_open = 0
status = WorkspaceOnly / TaskHealthBlocked / GeometryBlocked
```

## 10.4 每个 basis 的修复方向

### Fourier

```text
D-FOU-LowFreqIdentityResidual
D-FOU-BandwiseOccupancyWarmup
D-FOU-PhaseStableBandMix
D-FOU-HighFreqQuarantine
```

记录：

```text
band_energy
high_freq_ratio
phase_drift
band_snr
band_occupancy_entropy
```

### RBF / FastKAN

```text
D-RBF-CompactBumpActiveCenter
D-RBF-WidthConditionGuard
D-RBF-IdentityResidual
D-RBF-CenterOccupancyRebalance
```

记录：

```text
center_occupancy
width_condition
out_of_grid_fraction
center_snr
boundary_tail_fraction
```

### Chebyshev

```text
D-CHE-LowDegreeIdentityResidual
D-CHE-HighDegreeLateEnable
D-CHE-DegreeEnergyDamping
D-CHE-RecurrenceNoMaterializeLifetime
```

记录：

```text
degree_energy
high_degree_ratio
degree_snr
recurrence_stability
incremental_memory_ratio
```

### Wavelet

```text
D-WAV-TriangularSupportHealth
D-WAV-ScaleOccupancyBalance
D-WAV-LocalTailCoverage
D-WAV-SupportOverlapDamping
```

记录：

```text
scale_energy
support_overlap
local_tail_occupancy
scale_snr
support_dead_fraction
```

---

# 11. Line C：Manifold-Channel Geometry Audit

Line C 继续做审计，不作为 direction source。

## 11.1 记录指标

```text
CouplingR2
CouplingCorr
NoiseSignalLeak
RealSignalReservoirRatio
signal_mass_topk
reservoir_fraction
KernelDrift
CEp99
NLL
ECE
Brier
margin_p10
AUCtime
source_vs_best_control
```

## 11.2 几何解释规则

如果：

```text
LineC fail = 0
AUCtime fail high
```

说明问题是 trajectory / optimization path，不是 manifold tearing。

如果：

```text
NoiseSignalLeak worsen
```

说明 functional update 把噪声推进 signal channel，不能 promotion。

如果：

```text
ReservoirRatio worsen
```

说明真实信号被困在 reservoir，functional update 也不能 promotion。

---

# 12. 必须生成的 CSV / JSON / SVG

## 12.1 CSV / JSON

```text
v1461_route_decision.json
v1461_code_review_manifest.csv
v1461_forbidden_information_audit.csv
v1461_action_search_violation_audit.csv
v1461_existing_bank_lock_manifest.csv
v1461_missing_state_trajectory_autopsy.csv
v1461_aucdebt_decomposition.csv
v1461_optimizer_state_mismatch.csv
v1461_noop_dominance_audit.csv
v1461_mechanism_probe_manifest.csv
v1461_optimizer_state_transport_probe.csv
v1461_noop_commit_probe.csv
v1461_micro_horizon_integral_probe.csv
v1461_delayed_projection_probe.csv
v1461_mechanism_coverage_certificate.csv
v1461_controller_results.csv
v1461_all_basis_substrate_status.csv
v1461_linec_audit.csv
v1461_failure_table.csv
v1461_no_go_boundary.md
v1461_next_hypothesis_queue.md
```

## 12.2 Figures

```text
fig_missing_state_aucdebt.svg
fig_aucdebt_early_mid_late.svg
fig_optimizer_state_mismatch.svg
fig_noop_dominance.svg
fig_mechanism_probe_effect.svg
fig_mechanism_coverage_before_after.svg
fig_pass_state_regression_heatmap.svg
fig_all_basis_substrate_matrix.svg
fig_linec_vs_aucdebt.svg
fig_failure_taxonomy.svg
```

---

# 13. Route 决策

## R0-ActionSearchViolation

触发条件：

```text
新增 token/action/grid；
局部 positive -> 新 action；
controller 在 upper bound 不足时运行；
audit metric 被用作 direction。
```

## R1-MissingStateNotTrajectoryMechanism

触发条件：

```text
missing states 不是 AUCtime-only / optimizer-state / trajectory-cost；
而是 source/tail/LineC 多重冲突。
```

## R2-MechanismUpperBoundInsufficient

触发条件：

```text
Line P mechanism probes 没有把 coverage_after 推到 9/9。
```

含义：

```text
当前 Rational-FMS route 的 upper bound 仍不足；
不能 controller；
不能继续 action search。
```

## S4c-MechanismSufficientDiagnostic

触发条件：

```text
coverage_after = 9/9
promotion_allowed = 0
```

含义：

```text
机制族存在 upper bound；
可以进入 train-stream commit policy。
```

## R3-CommitPolicyFail

触发条件：

```text
coverage_after = 9/9
controller_real_pass < 7/9
```

## S4d-ControllerExplorationPositive

触发条件：

```text
controller_real_pass >= 7/9
controller_real_pass < 9/9
promotion_allowed = 0
```

## S5-OfficialFunctionalSuccess

触发条件：

```text
real_dataset_seed_pass_count = 9/9
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_pass = 1
provenance pass
code review pass
promotion_allowed = 1
```

---

# 14. Codex 执行要求

Codex 必须在复盘文件中写明：

```text
1. 本轮是否新增 action token；
2. 是否发生 action_search_violation；
3. Line T missing states 如何分类；
4. Line P 每个 mechanism probe 针对哪个 falsifiable hypothesis；
5. 是否存在 pass-state regression；
6. coverage_before / coverage_after；
7. controller 是否被允许执行；
8. all-basis substrate 哪些 family 有进展；
9. 哪些实现代码是核心，需要人工审查；
10. 下一轮是否还有机制假设，还是应停掉 Rational-FMS route。
```

核心代码审查文件必须包括：

```text
experiments/run_v1461_trajectory_mechanism_sufficiency.py
dgkan/functional/fms.py
dgkan/functional/train_stream_counterfactual.py
dgkan/diagnostics/trajectory_aucdebt.py
dgkan/models/fc_purekan_primitives.py
dgkan/diagnostics/basis_workspace.py
```

如果某文件不存在，Codex 必须说明实际对应文件路径。

---

# 15. 成功后才能做什么

只有当：

```text
S4c-MechanismSufficientDiagnostic
```

成立后，才允许研究 commit policy。

只有当：

```text
S5-OfficialFunctionalSuccess
```

成立后，才允许：

```text
1. 写 functional update success；
2. 扩 5-seed / 10-seed；
3. 开 external fair；
4. 做 broader task transfer。
```

---

# 16. 最终判断

v14.6.1 的核心不是“再找好动作”，而是：

$$
\boxed{
\text{先证明剩余 2/9 是什么机制问题，}
\text{再用最小机制 probe 判断是否存在合法修复。}
}
$$

如果机制 probe 不能把 upper bound 补到 9/9，就必须承认：

$$
\boxed{
\text{当前 Rational-FMS action family 的 real-transfer upper bound 是 7/9。}
}
$$

这时不能继续靠 action search 消耗时间；必须换 functional mechanism 或换 substrate。

如果机制 probe 能把 upper bound 补到 9/9，再谈 train-stream commit policy。

这版计划的底线是：

$$
\boxed{
\text{action bank 不是研究路线；}
\text{mechanism sufficiency 才是研究问题。}
}
$$
