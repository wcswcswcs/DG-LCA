# DG-KAN v9.2.39 Anchor-or-Pivot：LQ Base 稳定性重锚、协议归因与 Snapshot Functional Re-entry 完整实验计划

> 本计划基于 v9.2.38 `LQ Base Re-Anchor 与 Snapshot Late-Attach Functional Carrier` 的真实复盘制定。  
> v9.2.38 的 terminal route 是：
>
> ```text
> route = R1-LQBaseAnchorBroken
> base_candidate = LQ-t2-h256
> success_v9238_strict_purekan_functional = False
> success_v9238_full_functional = False
> success_v9238_external_ready = False
> ```
>
> v9.2.38 的关键事实是：
>
> ```text
> Current LQ near-pass = 6/9
> Current LQ macro delta = -0.004500
> Historical LQ near-pass = 8/9
> Historical LQ macro delta = -0.004000
> historical-current macro delta = -0.000500
> TPEA-off P5 equivalence pass = 1
> protocol mismatch detected = 1
> protocol mismatch mode = historical_current_nearpass_drift
> snapshot late-attach / carrier / value / paired replay = not_opened
> ```
>
> 这轮最重要的结论不是 “functional update 失败”，而是：
>
> $$
> \boxed{
> \text{当前 LQ base 的 row-level near-pass 锚点不稳定，functional route 暂时没有合法评估基础。}
> }
> $$
>
> 更微妙的是，当前 LQ 的 macro delta 只比历史差约 $-0.0005$，不是大幅退化；真正失败是 near-pass count 从 $8/9$ 掉到 $6/9$。因此 v9.2.39 不能继续无脑加 functional carrier，也不能立刻放弃 LQ/T2，而要先判断：
>
> $$
> \boxed{
> \text{这是协议漂移、row-level gate 脆弱、LQ base 真实退化，还是 MLP-match 定义漂移？}
> }
> $$
>
> 用户特别提醒：可以诊断不同数据集的问题，但不能针对数据集调参，因为目标不是在这些数据集上打榜。因此本计划继续坚持：dataset 只能作为 evaluation slice / failure diagnosis，不能作为 official controller 或 candidate selection 条件。

---

## 0. 硬约束

本计划继续遵守：

```text
no teacher
no self-teacher
no distillation
no loss modification
no label smoothing
no focal / margin / calibration loss
no sampler / class weight
no CPU offload
no fake / proxy rows
KAN path 不使用 PyTorch loss.backward graph
official controller 不使用 dataset_name 分支
PureKANConv / PureKANFormer 继续 deferred
```

Functional update 仍然是 update rule，不是 loss：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{\text{AdamW}}
+
\Delta\theta_{\text{functional}}.
$$

任务目标保持标准 CE：

$$
L_{\text{task}}=CE(y,p_\theta(x)).
$$

数据集只能作为诊断切片，不能作为 official route 条件：

```text
allowed:
  report MNIST / Fashion-MNIST / KMNIST slice metrics
  report seed-level miss rows
  report which signal strata dominate each dataset
  report leave-dataset-out generalization

forbidden:
  if dataset == Fashion: use controller A
  if dataset == KMNIST: use primitive B
  if dataset == MNIST: abstain
  tune threshold separately per dataset
  choose base candidate because it rescues only one dataset
```

---

# Part I. 对当前结果的独立判断

## 1. v9.2.38 没有达到目标

v9.2.38 没有打开 strict PureKAN functional route。原因不是 paired replay 失败，而是更前置的 base anchor 失败：

```text
lq_base_anchor_pass = 0
current_lq_near_rate = 6/9 = 0.666667
required near-pass rate >= 0.80
current_lq_macro_delta = -0.004500
historical_current_lq_delta = -0.000500
```

因此 snapshot late attach、carrier、value observability、paired replay 全部没有资格打开。

这轮的 route 写成 `R1-LQBaseAnchorBroken` 是合理的，因为预注册 gate 是 near-pass rate，而不是单看 macro delta。但是科学解释上不能过度外推成 “LQ base 崩了”。更准确的解释是：

$$
\boxed{
\text{LQ base 的 aggregate macro 仍接近历史，但 row-level near-pass 稳定性不足。}
}
$$

## 2. 进展是否太慢

我同意：如果继续按照 “一个版本只测一个 gate” 的方式推进，进展会太慢。  
但这几轮并不是没有信息。它们排除了很多错误解释：

```text
v9.2.35:
  VAOP 实现了、contract/grad 过了，但直接混入 base 会破坏 P5。

v9.2.36:
  BPFS 实现了、base-equivalence/contract/grad 过了，但 P5 仍没保住。

v9.2.37:
  TPEA 实现了、training-path equivalence/contract/grad 过了，但 P5 仍没保住。

v9.2.38:
  TPEA-off 与当前 LQ P5 等价，但当前 LQ 自己 near-pass 只有 6/9。
```

这条链路说明：我们从 “functional carrier 是否污染 base” 推进到了 “base reference 本身是否稳定”。这不是倒退，而是把真正的前置锚点暴露出来了。

但是 v9.2.39 必须改变推进方式：不能再串行地做一个小 gate。下一轮应该把所有 base-anchor 解释一次性并行归因，形成 **anchor-or-pivot** 决策。换句话说，v9.2.39 要把问题收敛到两个可执行结论之一：

```text
A. LQ base 可重新锚定，然后继续 snapshot late attach functional。
B. LQ base 在当前协议下不可稳定锚定，必须停止 LQ-based functional route，转入 base repair / base pivot。
```

## 3. 当前真正 blocker

当前 blocker 不是：

```text
functional update 失败；
TPEA 不可实现；
LQ/T2 basis 没价值；
KMNIST / Fashion 要单独调；
paired replay 不可能过。
```

当前 blocker 是：

$$
\boxed{
\text{没有一个当前协议下可复现的 P5 base anchor，functional 的因果评估没有地基。}
}
$$

更具体地说，需要先确认：

```text
1. Current LQ 的 6/9 near-pass 是统计波动还是协议漂移？
2. 失败的 3 rows 是不是都在 near-pass threshold 附近？
3. MLP-match baseline 是否比历史更强，导致 KAN delta 变差？
4. KAN 自身是否退化，还是 baseline / split / seed / hidden / scale / gate 定义漂移？
5. TPEA-off 与 Current LQ 等价已通过，是否意味着 attach infrastructure 可以暂时不再作为主 blocker？
```

---

# Part II. v9.2.39 总体目标

v9.2.39 的总体目标是：

$$
\boxed{
\text{用同一个 runner 一次性解决 LQ anchor 是否可复现，并在 anchor pass 后才恢复 snapshot functional route。}
}
$$

这个目标分六层。

## 1. Anchor diagnosis success

必须定位 `6/9` 的原因。至少要把 LQ anchor 失败归入下面一种主因：

```text
A1-protocol_hash_drift:
  candidate config / data split / seed / MLP-match / gate definition 与历史不一致。

A2-mlp_match_strength_drift:
  MLP-match baseline 变强，导致 KAN delta row-level 变差。

A3-lq_task_degradation:
  KAN raw accuracy / CEp99 / margin 比历史真实变差。

A4-row_threshold_brittleness:
  macro delta 基本一致，但少数 rows 在 -0.01 near-pass threshold 附近随机跨线。

A5-dataset_slice_instability:
  miss rows 集中在某些 evaluation slices，但不能用 dataset-specific tuning 处理。

A6-runner_measurement_artifact:
  当前 runner 的 evaluator、batching、data normalization、checkpoint loading、metric aggregation 有 bug 或不一致。

A7-real_base_instability:
  protocol 无漂移，repeats 后仍低于 8/9，说明 LQ/T2 base 作为 official anchor 不够稳。
```

## 2. LQ re-anchor success

当前协议下 LQ 必须重新满足：

$$
near\_pass\_rate_{\text{LQ}}\geq0.80,
$$

$$
\Delta Acc_{\text{macro,LQ}}\geq-0.01,
$$

并且与历史 reference 的宏观差异满足：

$$
|\Delta Acc_{\text{macro,current LQ}}-\Delta Acc_{\text{macro,historical LQ}}|\leq0.003.
$$

如果 near-pass 只在 repeated runs 中不稳定，必须报告 row-level confidence，而不能直接写 success。

## 3. Gate brittleness clarification

如果 macro delta 过、历史-current macro delta 小，但 near-pass count 不过，则要计算 row margin：

$$
m_{row}
=
\Delta Acc_{row}+0.01.
$$

其中 $m_{row}<0$ 表示该 row 未 near-pass。  
如果所有 miss rows 满足：

$$
|m_{row}|\leq0.002,
$$

则当前问题属于 **gate brittleness / threshold-borderline**，不能直接说 base 崩溃，但也不能把 gate 改掉后立刻算成功。必须做独立 repeated confirmation。

## 4. Snapshot late-attach success

只有 LQ anchor pass 后，才允许恢复 snapshot late attach。Attach 后 inactive 必须满足：

$$
\max_x
\|z_{\text{late-attach-off}}-z_{\text{LQ checkpoint}}\|_\infty
\leq10^{-6}.
$$

No-event replay 必须满足：

$$
\max_{t\leq T}
\|z_{t,\text{late-no-event}}-z_{t,\text{base-continue}}\|_\infty
\leq10^{-5}.
$$

## 5. Functional carrier success

只有 base-ready attach success 后，才测 functional carrier。最低 actuatability gate：

$$
r_{z,\text{tail}}\geq0.10,
$$

$$
r_{\perp,\text{tail}}\geq0.10.
$$

Task safety：

$$
BadEventRate\leq0.05.
$$

## 6. Local functional causality success

Value score 必须预测 grounded Real-vs-control value：

$$
AUC(Y_{\text{beat}})\geq0.70
$$

or:

$$
Corr(S_{\text{value}},V_{\text{grounded}})\geq0.35.
$$

Accept / abstain gate：

$$
Precision_{\text{accepted beats controls}}\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

Official paired replay：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

---

# Part III. 核心假设

## H1：v9.2.38 的失败主要是 row-level near-pass instability，不是 LQ macro collapse

证据是：

$$
\Delta Acc_{\text{macro,current LQ}}=-0.004500,
$$

$$
\Delta Acc_{\text{macro,historical LQ}}=-0.004000,
$$

因此：

$$
\Delta_{\text{historical-current}}=-0.000500.
$$

如果 LQ 真正崩了，macro delta 应该明显偏离历史，而不是只差 $0.0005$。  
H1 成立标准：

```text
historical-current macro delta <= 0.003
miss rows are near threshold
repeated runs show near-pass count fluctuates around 6/9-8/9
```

如果 miss rows 不是 near-threshold，而是显著低于 $-0.01$，则 H1 失败。

## H2：TPEA attach infrastructure 不是当前主 blocker

v9.2.38 显示：

```text
TPEA-off P5 equivalence pass = 1
```

这说明 TPEA-off 与 current LQ 在 P5 上等价。因此，如果 current LQ 自己 anchor fail，就不能再把失败主要归因于 attach contamination。

H2 成立标准：

$$
|\Delta Acc_{\text{TPEA-off}}-\Delta Acc_{\text{CurrentLQ}}|\leq0.001.
$$

## H3：如果 Current LQ 不能重锚，就必须停止 LQ-based functional route

Functional update 不能在 failed base 上继续推进。  
如果 current LQ 在 protocol repair 和 repeated confirmation 后仍满足：

$$
near\_pass\_rate<0.80,
$$

则 route 必须转为：

```text
R4-LQAnchorUnstableBaseRepairRequired
```

此时下一步是 base repair / base pivot，不是 functional carrier。

## H4：如果 LQ anchor pass，但 snapshot attach fail，则 blocker 回到 attach infrastructure

如果 current LQ re-anchor pass，而 snapshot inactive / no-event replay fail，则问题不是 LQ base，而是 late attach implementation。

H4 成立标准：

```text
LQ anchor pass = 1
checkpoint inactive equivalence pass = 0
or no-event replay preservation pass = 0
```

## H5：如果 LQ anchor 与 late attach 都 pass，但 carrier silent，则 blocker 是 functional carrier actuatability

此时应修 functional carrier，而不是 base。

H5 成立标准：

$$
r_{z,\text{tail}}<0.10
$$

or:

$$
r_{\perp,\text{tail}}<0.10.
$$

## H6：如果 carrier active 但 value unobservable，则 blocker 回到 role-wise / control-gap mechanism

此时 functional movement exists，但不等于 control-resistant value。  
H6 成立标准：

$$
r_{z,\text{tail}}\geq0.10,
$$

$$
r_{\perp,\text{tail}}\geq0.10,
$$

but:

$$
AUC(Y_{\text{beat}})<0.70.
$$

## H7：dataset 只能用于诊断，不能用于 route selection

即使 miss rows 集中在 KMNIST 或 Fashion，也不能写：

```text
if dataset == KMNIST: choose base repair X
```

只能写：

```text
if signal_stratum == hard_tail and control_gap_lcb > threshold: trigger functional update
```

H7 成立标准：

```text
dataset_tuning_detected = 0
controller_feature_dataset_name_used = 0
leave-dataset-out pass required before official route
```

---

# Part IV. Candidate 与对照设计

## 1. Base candidates

### B0：Historical LQ Reference

历史 reference 只用于 comparison，不作为 current measured pass。必须记录：

```text
historical near-pass = 8/9
historical macro delta = -0.004000
source artifact hash
source candidate config hash
```

### B1：Current LQ Reproduction

当前 runner 下原样复现：

```text
LQ-t2-h256
same hidden dim
same fan-in output scale
same compact memory mode
same P5 protocol
same MLP-match definition
same train/eval split if possible
```

### B2：Exact-Protocol LQ Replay

把历史 protocol 中所有可恢复字段锁死：

```text
seed protocol
data split
minibatch order
fan-in output scale
hidden dim
lr
epochs
batch size
eval size
metric aggregation
MLP-match param count
```

### B3：Repeated Current LQ

对 Current LQ 做 repeated reruns，不改变 candidate，只测稳定性：

```text
reruns = 3 or 5
same seeds = 0,1,2
same datasets = MNIST,Fashion-MNIST,KMNIST
```

### B4：TPEA-off Equivalence Reference

复现 v9.2.38 TPEA-off equivalence，用于确认 attach infrastructure 不是主 blocker。

### B5：Global Base Repair Candidates

只有在 LQ anchor fail 后打开。所有 repair 只能是全局候选，不允许 per-dataset tuning：

```text
R0-LQ-current
R1-LQ-centered-T2
R2-LQ-fanin-output-scale-confirmed
R3-LQ-orthogonal-lift-init
R4-LQ-hidden224
R5-LQ-hidden256
R6-LQ-hidden288
R7-LQ-basis-balanced-init
R8-LQ-legendre2-loworder
```

Base repair 只允许用于 AdamW-only base；functional update 不许救 base。

---

# Part V. 实验阶段

## P0：v9.2.38 boundary reproduction

### 目标

复现 v9.2.38 boundary，确认不是读取/解析错误。

### 必须记录

```text
route
source_route_v9237
current_lq_near_count
current_lq_row_count
current_lq_near_rate
current_lq_macro_delta
historical_lq_near_count
historical_lq_macro_delta
historical_current_lq_delta
protocol_mismatch_detected
protocol_mismatch_mode
tpea_off_p5_equivalence_pass
snapshot_late_attach_opened
fake_proxy_count
```

### 判断标准

P0 pass：

```text
route = R1-LQBaseAnchorBroken
current_lq_near_count = 6
current_lq_row_count = 9
historical_current_lq_delta approximately -0.0005
fake/proxy = 0
```

### 可视化

```text
p0_boundary_dashboard.svg
p0_lq_anchor_gate_ladder.svg
p0_historical_current_delta.svg
```

---

## P1：LQ anchor miss-row autopsy

### 目标

解释 `6/9` 的具体 miss rows，不允许只看 aggregate route。

### 必须记录

每个 dataset / seed row：

```text
dataset
seed
current_lq_acc
historical_lq_acc
current_mlp_match_acc
historical_mlp_match_acc
current_delta_vs_mlp
historical_delta_vs_mlp
delta_drift
near_pass
near_margin = current_delta_vs_mlp + 0.01
CEp99
margin_p10
ECE
NLL
wrong_confidence_p95
basis_entropy
lift_condition_number
effective_rank
```

### 判断标准

P1 attribution pass：

```text
每个 miss row 都归因到一种 primary mode：
  M1-KAN_acc_dropped
  M2-MLP_baseline_strengthened
  M3-both_changed
  M4-threshold_borderline
  M5-metric_protocol_drift
  M6-data_split_or_seed_drift
  M7-unattributed
```

如果 `M7-unattributed` 超过 10%，P1 fail。

定量：

$$
near\_margin = \Delta Acc_{row}+0.01.
$$

如果 miss row 满足：

$$
|near\_margin|\leq0.002,
$$

则标记为 threshold-borderline。

### 可视化

```text
p1_row_delta_heatmap.svg
p1_near_margin_distribution.svg
p1_kan_vs_mlp_drift_decomposition.svg
p1_miss_row_mechanism_sankey.svg
```

---

## P2：Protocol hash and MLP-match audit

### 目标

确认 current / historical protocol 是否真的一致。

### 必须记录

```text
candidate_config_hash
runner_hash
data_protocol_hash
data_split_hash
seed_protocol_hash
minibatch_order_hash
p5_gate_definition_hash
mlp_match_definition_hash
mlp_match_param_count
kan_param_count
param_ratio
hidden_dim
basis_type
fan_in_output_scale
memory_mode
batch_size
lr
epochs
train_size
eval_size
optimizer_config_hash
manual_update_config_hash
metric_aggregation_hash
```

### 判断标准

Protocol pass：

```text
所有 essential hash match
或 mismatch 有明确 documented reason
```

Essential fields：

```text
candidate_config_hash
data_split_hash
seed_protocol_hash
mlp_match_definition_hash
p5_gate_definition_hash
metric_aggregation_hash
```

如果 essential hash mismatch 且无法解释，route 必须为：

```text
R2-ProtocolMismatchExplainsAnchorBreak
```

### 可视化

```text
p2_protocol_hash_diff_table.svg
p2_param_count_match.svg
p2_metric_gate_diff.svg
```

---

## P3：Current LQ repeated re-anchor

### 目标

判断 `6/9` 是一次性波动还是稳定失败。

### 设置

```text
candidate = LQ-t2-h256
functional_update = off
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
reruns = 3 or 5
same protocol as P2
baseline = MLP-match
```

### 必须记录

```text
rerun_id
dataset
seed
lq_acc
mlp_acc
delta_vs_mlp
near_pass
near_margin
CEp99
margin_p10
ECE
NLL
basis_entropy
lift_condition_number
effective_rank
run_time
memory_ratio
step_ratio
```

Aggregate：

```text
near_pass_rate_mean
near_pass_rate_min
near_pass_rate_max
macro_delta_mean
macro_delta_std
row_near_margin_mean
row_near_margin_std
miss_row_stability
```

### 判断标准

Stable LQ anchor pass：

$$
\min_{rerun} near\_pass\_rate \geq 0.80,
$$

and:

$$
\Delta Acc_{\text{macro,mean}}\geq-0.01.
$$

Borderline anchor diagnostic：

```text
mean near-pass >= 0.80
but min near-pass < 0.80
and all miss near_margin within ±0.002
```

Stable fail：

```text
mean near-pass < 0.80
or repeated miss rows are not threshold-borderline
```

### 可视化

```text
p3_lq_nearpass_rerun_stability.svg
p3_row_margin_over_reruns.svg
p3_macro_delta_ci.svg
p3_miss_row_stability_matrix.svg
```

---

## P4：Anchor decision and base repair gate

### 目标

根据 P1-P3 做硬决策，不允许继续模糊推进 functional。

### Route cases

```text
A-pass:
  Current LQ re-anchored.
  Open snapshot late attach.

B-borderline:
  LQ macro stable but row gate borderline.
  Run independent confirmation; no functional until confirmed.

C-protocol:
  Protocol mismatch explains failure.
  Repair protocol and rerun P3.

D-base-fail:
  Current LQ truly unstable.
  Stop LQ-based functional; open global base repair.

E-mlp-drift:
  MLP-match definition/strength changed.
  Reconcile fairness protocol before functional.
```

### 必须记录

```text
anchor_decision
primary_failure_mode
lq_anchor_pass
lq_borderline_diagnostic
protocol_repair_required
base_repair_required
functional_route_allowed
```

### 判断标准

Functional route allowed only if：

```text
lq_anchor_pass = 1
protocol_mismatch_unresolved = 0
current_lq_nearpass_stable = 1
```

---

## P5：Global LQ base repair if anchor fails

### 目标

如果 LQ anchor 稳定失败，做全局 base repair，而不是 dataset-specific tuning。

### 设置

```text
candidates = R0-R8
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
functional_update = off
baseline = MLP-match
```

### 必须记录

```text
candidate
dataset
seed
acc
mlp_acc
delta_vs_mlp
near_pass
CEp99
margin_p10
ECE
NLL
basis_entropy
lift_condition_number
effective_rank
P4_forward_q90
P4_backward_q90
P4_step_q90
memory_ratio
```

### 判断标准

Base repair pass：

$$
near\_pass\_rate\geq0.80,
$$

$$
\Delta Acc_{\text{macro}}\geq-0.01.
$$

System pass：

$$
forward_{q90}\leq1.25,
$$

$$
backward_{q90}\leq1.50,
$$

$$
step_{q90}\leq1.50,
$$

$$
memory\leq1.05.
$$

Promotion rule：

```text
global macro / system / conditioning best
no dataset-specific branch
no single-dataset rescue candidate promotion
```

### 可视化

```text
p5_base_repair_pareto.svg
p5_base_repair_row_matrix.svg
p5_conditioning_vs_nearpass.svg
```

---

## P6：Snapshot late attach implementation

### 目标

只有 anchor pass 后，恢复 snapshot late attach。

### 必须记录

```text
candidate
attach_mode
base_checkpoint_hash
task_param_hash_before_attach
task_param_hash_after_attach
optimizer_state_hash_before_attach
optimizer_state_hash_after_attach
functional_param_registered_before_attach
functional_param_registered_after_attach
functional_param_zero_init
edge_owned_param_fraction
external_residual_used
ordinary_mlp_path_used
manual_forward
manual_backward
manual_update
uses_loss_backward
lambda_inactive_exact_zero
implemented
implementation_status
```

### 判断标准

Implementation pass：

```text
at least A2 late-attach zero linear tail implemented
at least one of A3 control-gap or A5 role-wise implemented
strict PureKAN contract pass
```

Contract pass：

```text
edge_owned_param_fraction = 1
external_residual_used = 0
ordinary_mlp_path_used = 0
manual_forward = 1
manual_backward = 1
manual_update = 1
uses_loss_backward = 0
```

---

## P7：Checkpoint inactive equivalence and no-event replay

### 目标

证明 attach 不污染 base checkpoint。

### 必须记录

Inactive equivalence：

```text
candidate
dataset
seed
batch_id
base_checkpoint_hash
max_logit_diff_inactive
mean_logit_diff_inactive
task_param_max_diff
optimizer_m_max_diff
optimizer_v_max_diff
functional_zero_norm
lambda_value
lambda_leak_detected
```

No-event replay：

```text
steps = 1,5,20,80
branch = base_continue, attach_off_continue, attach_no_event
train_loss
holdout_loss
CEp99
margin_p10
ECE_proxy
NLL_proxy
curvature
logit_diff_vs_base_continue
task_param_diff_vs_base_continue
optimizer_state_diff_vs_base_continue
step_q90
memory_ratio
```

### 判断标准

Inactive equivalence：

$$
\max_x
\|z_{\text{attach-off}}-z_{\text{LQ checkpoint}}\|_\infty
\leq10^{-6}.
$$

No-event replay：

$$
\max_{t\leq T}
\|z_{t,\text{attach-no-event}}-z_{t,\text{base-continue}}\|_\infty
\leq10^{-5}.
$$

Metric preservation：

$$
|CEp99_{\text{attach-no-event}}-CEp99_{\text{base-continue}}|\leq10^{-5}.
$$

System：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p7_inactive_equivalence.svg
p7_no_event_replay_trace.svg
p7_metric_preservation_trace.svg
p7_attach_system_overhead.svg
```

---

## P8：Functional carrier actuatability

### 目标

只在 base-ready attach 上测试 functional carrier 是否能产生 non-silent / non-AdamW movement。

### 设置

```text
base = qualified LQ or repaired base checkpoint
events = signal strata S1-S8
horizons = 1,5,20,80,240
branches = RealFunctional, AdamWParallel, bestLR, NoOp, Random
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
```

### 必须记录

```text
candidate
event_id
signal_stratum
horizon
branch
actual_logit_delta_norm
actual_tail_logit_delta_norm
actual_nonadamw_delta_norm
r_z
r_z_tail
r_perp_tail
cos_real_adamw
cos_real_bestlr
branch_ratio
effective_derivative
functional_step_norm
task_step_norm
task_safe
bad_event
```

### 判断标准

Carrier non-silent：

$$
r_{z,\text{tail}}\geq0.10.
$$

Non-AdamW component：

$$
r_{\perp,\text{tail}}\geq0.10.
$$

Task safety：

$$
BadEventRate\leq0.05.
$$

### 可视化

```text
p8_functional_movement_distribution.svg
p8_tail_movement_vs_task_safety.svg
p8_nonadamw_component.svg
```

---

## P9：Value observability audit

### 目标

判断 functional score 是否预测 grounded Real-vs-control value。

### 必须记录

```text
candidate
controller
event_id
signal_stratum
horizon
value_score
control_gap_score
tail_value_score
role_score
grounded_value
Y_beat
corr
auc
precision
coverage
bad_event_rate
accepted_event_count
dataset_name_used
posthoc_used_at_commit
validation_used
test_used
step_q90
memory_ratio
```

### 判断标准

Observability pass：

$$
AUC(Y_{\text{beat}})\geq0.70
$$

or:

$$
Corr(S_{\text{value}},V_{\text{grounded}})\geq0.35.
$$

Accept / abstain pass：

$$
Precision\geq0.75,
$$

$$
Coverage\in[0.03,0.15],
$$

$$
BadEventRate\leq0.05.
$$

Legality：

```text
dataset_name_used = 0
posthoc_used_at_commit = 0
validation_used = 0
test_used = 0
```

### 可视化

```text
p9_value_score_vs_grounded_value.svg
p9_auc_precision_coverage.svg
p9_score_components_correlation.svg
p9_accepted_strata_distribution.svg
```

---

## P10：Leave-dataset-out / leave-stratum-out validation

### 目标

防止 controller 隐式 dataset tuning。

### 设置

Leave-dataset-out：

```text
train/calibrate on MNIST + Fashion, evaluate KMNIST
train/calibrate on MNIST + KMNIST, evaluate Fashion
train/calibrate on Fashion + KMNIST, evaluate MNIST
```

Leave-stratum-out：

```text
train/calibrate on all but one signal stratum
evaluate held-out stratum
```

### 必须记录

```text
split_type
heldout
candidate
controller
threshold
precision
coverage
bad_event_rate
task_safe
CEp99_delta
margin_delta
ECE_delta
NLL_delta
curvature_delta
beats_adamwparallel
beats_bestlr
shuffle_control_pass
```

### 判断标准

LDO pass：

$$
Acc_{\text{heldout,Real}}\geq Acc_{\text{heldout,AdamW}}-0.005.
$$

At least two LDO splits satisfy：

$$
BeatRate_{\text{heldout,Real vs AdamWParallel}}\geq0.50,
$$

$$
BeatRate_{\text{heldout,Real vs bestLR}}\geq0.50.
$$

LSO pass：

At least $70\%$ held-out strata task-safe and CE-tail non-worse：

$$
CEp99_{\text{heldout,Real}}\leq CEp99_{\text{AdamW}}+\epsilon.
$$

### 可视化

```text
p10_leave_dataset_out_matrix.svg
p10_leave_stratum_out_matrix.svg
p10_hidden_dataset_tuning_audit.svg
```

---

## P11：Official snapshot late-attach paired replay

### 目标

只有 P1-P10 survivor 可以进入 official paired replay。

### 设置

```text
base = qualified LQ or repaired base checkpoint
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2,3,4
horizons = 20,80,240,640
candidate = best attach survivor
controller = best legal signal-based controller
branches = RealFunctional, AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, InvertedRole, ValueScoreShuffled, FunctionalChannelShuffled, TailMaskShuffled
```

### 必须记录

```text
candidate
controller
dataset
seed
horizon
signal_stratum
branch
CEp99_delta
margin_p10_delta
ECE_delta
NLL_delta
curvature_delta
acc_delta
real_beats_adamwparallel
real_beats_bestlr
real_beats_random
real_beats_noop
task_safe
event_count
coverage
bad_event_rate
step_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Official paired replay pass：

$$
BeatRate_{\text{macro,Real vs AdamWParallel}}\geq0.60,
$$

$$
BeatRate_{\text{macro,Real vs bestLR}}\geq0.60.
$$

Task safety：

$$
Acc_{\text{each slice,Real}}\geq Acc_{\text{AdamW}}-0.005.
$$

Anti-overfit controls fail：

```text
ValueScoreShuffled = fail
FunctionalChannelShuffled = fail
TailMaskShuffled = fail
DatasetRouteShuffled = fail
EventRouteShuffled = fail
InvertedRoleMask = fail
```

System：

$$
StepRatio_{q90}\leq1.50,
$$

$$
MemoryRatio\leq1.05.
$$

### 可视化

```text
p11_official_paired_replay_pareto.svg
p11_macro_beat_rate.svg
p11_signal_stratum_win_matrix.svg
p11_shuffle_control_matrix.svg
p11_system_gate_distribution.svg
```

---

## P12：Short-run validation

### 目标

验证 local causality 是否能在连续训练中保持。只有 P11 survivor 进入。

### 设置

```text
steps = 50,240,640
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole, ValueScoreShuffled
```

### 必须记录

```text
candidate
dataset
seed
steps
train_loss
holdout_loss
val_acc_proxy
CEp99
margin_p10
ECE_proxy
NLL_proxy
curvature
local_lipschitz
basis_usage_entropy
functional_channel_usage_entropy
event_count
coverage
bad_event_rate
step_ratio_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Task safety：

$$
Acc_{\text{functional}}\geq Acc_{\text{AdamW}}-0.005.
$$

Control superiority：

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{bestLR}}.
$$

Mechanism pass，至少一个：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

$$
ECE_{\text{functional}}\leq ECE_{\text{AdamW}},
$$

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

### 可视化

```text
p12_short_run_task_mechanism_pareto.svg
p12_short_run_controls.svg
p12_event_timeline.svg
p12_ce_tail_margin_panel.svg
```

---

## P13：Full 10-seed validation

### 目标

验证 strict PureKAN functional route 是否在 full training 上成立。

### 设置

```text
epochs = 20
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
baseline = MLP-match
diagnostic baseline = QuadraticFeatureMLP
controls = AdamWOnly, AdamWParallel, bestLR, NoOp, Random, ShuffledRole
```

### 必须记录

```text
candidate
dataset
seed
val_acc
test_acc
delta_vs_mlp
delta_vs_adamw
delta_vs_adamwparallel
delta_vs_bestlr
delta_vs_quadratic_feature_mlp
near_pass
full_pass
CEp99
margin_p10
wrong_confidence_p95
ECE
NLL
curvature_ratio
local_lipschitz_ratio
basis_usage_entropy
functional_channel_usage_entropy
event_count
coverage
bad_event_rate
step_ratio_q90
memory_ratio
base_checkpoint_hash
```

### 判断标准

Task safety：

$$
\Delta Acc_{\text{functional-vs-AdamW}}\geq-0.005.
$$

Macro improvement：

$$
\Delta Acc_{\text{macro,functional}}
-
\Delta Acc_{\text{macro,AdamW}}
\geq0.003.
$$

Hard-stratum repair：

$$
\Delta Acc_{\text{hard-stratum,functional}}
-
\Delta Acc_{\text{hard-stratum,AdamW}}
\geq0.005.
$$

Control superiority：

$$
MetricGain_{\text{functional}}>MetricGain_{\text{AdamWParallel}},
$$

$$
MetricGain_{\text{functional}}>MetricGain_{\text{bestLR}}.
$$

Mechanism pass，至少一个：

$$
CEp99_{\text{functional}}<CEp99_{\text{AdamW}},
$$

$$
MarginP10_{\text{functional}}>MarginP10_{\text{AdamW}},
$$

$$
ECE_{\text{functional}}\leq ECE_{\text{AdamW}},
$$

$$
NLL_{\text{functional}}\leq NLL_{\text{AdamW}},
$$

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{AdamW}}.
$$

---

## P14：Robustness / strong baseline / external-ready gate

### 目标

确认 strict PureKAN functional advantage 不是 clean MNIST-family artifact。

### 设置

```text
label_noise = 0.05,0.10,0.20
input_noise = 0.05,0.10
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
strong baselines = MLP-match, hidden-bracket MLP, QuadraticFeatureMLP, LQ/A7c AdamW-only
```

### 判断标准

Robustness pass：

$$
AccDrop_{\text{functional}}\leq AccDrop_{\text{AdamW}}.
$$

Noise-tail pass：

$$
CEp99_{\text{functional}}\leq CEp99_{\text{AdamW}}.
$$

Strong baseline pass：

$$
Acc_{\text{functional}}\geq Acc_{\text{QuadraticFeatureMLP}}-0.005
$$

or:

$$
Curvature_{\text{functional}}\leq0.90Curvature_{\text{QuadraticFeatureMLP}}.
$$

---

# Part VI. Required artifacts

```text
run_manifest.json
contract_audit_v9239.csv
p0_v9238_boundary_reproduction.csv
p1_lq_anchor_miss_row_autopsy.csv
p2_protocol_hash_and_mlp_match_audit.csv
p3_current_lq_repeated_reanchor.csv
p4_anchor_decision_and_base_repair_gate.csv
p5_global_lq_base_repair.csv
p6_snapshot_late_attach_implementation.csv
p7_checkpoint_inactive_noevent_equivalence.csv
p8_functional_carrier_actuatability.csv
p9_value_observability_audit.csv
p10_leave_dataset_and_stratum_out_validation.csv
p11_official_snapshot_late_attach_paired_replay.csv
p12_short_run_functional_validation.csv
p13_full_10seed_functional_validation.csv
p14_robustness_external_ready.csv
lq_anchor_trace_v9239.csv
miss_row_trace_v9239.csv
protocol_hash_diff_v9239.csv
snapshot_attach_trace_v9239.csv
functional_carrier_trace_v9239.csv
value_score_trace_v9239.csv
leave_dataset_out_trace_v9239.csv
paired_replay_branch_trace_v9239.csv
route_decision.json
aggregate_decision.json
failure_table.csv
figures/
```

Failure taxonomy：

```text
F1_contract_violation
F2_v9238_boundary_unstable
F3_dataset_tuning_detected
F4_lq_anchor_fail_unattributed
F5_protocol_hash_drift
F6_mlp_match_definition_drift
F7_data_split_or_seed_protocol_drift
F8_lq_row_threshold_brittleness
F9_lq_true_task_degradation
F10_current_lq_repeated_anchor_fail
F11_base_repair_required
F12_snapshot_late_attach_not_implemented
F13_checkpoint_inactive_equivalence_fail
F14_no_event_replay_preservation_fail
F15_functional_carrier_silent
F16_value_observability_fail
F17_leave_dataset_out_fail
F18_leave_stratum_out_fail
F19_paired_replay_control_equivalent
F20_shuffle_control_pass
F21_functional_lr_equivalent
F22_short_run_task_drop
F23_full_run_no_macro_hard_stratum_gain
F24_adamw_fullpass_fail
F25_strong_baseline_explains_gain
F26_robustness_fail
F27_external_not_ready
F28_fake_or_proxy_violation
F29_artifact_missing
```

---

# Part VII. Route decision

```text
R1-LQAnchorRowDriftDiagnosed:
  Current LQ macro stable, but row-level near-pass drift explains anchor failure.

R2-ProtocolMismatchExplainsAnchorBreak:
  candidate / MLP-match / seed / split / gate protocol mismatch found.

R3-LQBaseReanchored:
  Current LQ passes repeated P5 anchor.

R4-LQAnchorUnstableBaseRepairRequired:
  Current LQ remains below near-pass gate under matched protocol.

R5-BaseRepairPass:
  global base repair restores P4/P5 without dataset tuning.

R6-SnapshotLateAttachPass:
  late attach inactive/no-event preservation passes.

R7-FunctionalCarrierActive:
  functional carrier produces non-silent, non-AdamW movement.

R8-ValueObservabilityPass:
  value score predicts grounded Real-vs-control value.

R9-LeaveDatasetOutPass:
  controller generalizes without dataset-specific tuning.

R10-SnapshotLateAttachPairedReplayPass:
  official paired replay beats AdamWParallel / bestLR.

R11-BasePreservedButFunctionalSilent:
  base preserved, carrier cannot move output.

R12-BasePreservedButValueUnobservable:
  carrier moves output, but value score fails.

R13-StrictPureKANFunctionalShortRunPass:
  short-run task-safe mechanism gain.

R14-StrictPureKANFunctionalFullPass:
  full 10-seed macro / hard-stratum / geometry gain.

R15-ExternalReady:
  strict PureKAN functional route passes task / geometry / system / control / robustness / strong-baseline gates.
```

`route_decision.json` 必须记录：

```text
route
v9238_boundary_pass
dataset_tuning_detected
lq_anchor_pass
lq_anchor_failure_mode
current_lq_near_rate
current_lq_macro_delta
historical_current_lq_delta
miss_row_borderline_rate
protocol_mismatch_detected
mlp_match_drift_detected
base_repair_required
best_base_candidate
snapshot_attach_pass
checkpoint_inactive_equivalence_pass
no_event_replay_preservation_pass
functional_carrier_pass
value_observability_pass
leave_dataset_out_pass
paired_replay_pass
short_run_pass
full_run_pass
external_ready
primary_blocker
next_required_implementation
success_v9239_strict_purekan_functional
success_v9239_full_functional
success_v9239_external_ready
```

---

# Part VIII. 第一轮执行顺序

```text
Step 1:
  P0 复现 v9.2.38 boundary。

Step 2:
  P1 做 miss-row autopsy。
  必须列出当前 3 个 miss rows 的 near_margin、KAN/MLP drift、CEp99、margin、ECE、NLL。

Step 3:
  P2 做 protocol hash audit。
  如果 essential hash mismatch，先修 protocol，不进入 functional。

Step 4:
  P3 做 Current LQ repeated re-anchor。
  判断 6/9 是一次性波动、borderline，还是稳定失败。

Step 5:
  P4 做 anchor decision。
  只有 LQ anchor pass 才允许 functional route。

Step 6:
  如果 LQ anchor fail，P5 做 global base repair。
  不允许 dataset-specific repair。

Step 7:
  若 base anchor pass，P6-P7 做 snapshot late attach 和 no-event replay。

Step 8:
  P8 做 functional carrier actuatability。

Step 9:
  P9 做 value observability。

Step 10:
  P10 做 leave-dataset-out / leave-stratum-out。

Step 11:
  P11 official paired replay。

Step 12:
  P12 short-run。

Step 13:
  P13 full 10-seed。

Step 14:
  P14 robustness / external-ready。
```

---

# Part IX. 停止条件

## Minimum diagnostic success

```text
v9.2.38 boundary reproduced
LQ anchor failure attributed
protocol mismatch found or ruled out
current LQ repeated re-anchor completed
no fake/proxy/offload/loss/teacher violation
```

## Base-ready success

```text
LQ anchor pass
or global base repair pass
+
snapshot late attach inactive equivalence
+
no-event replay preservation
```

## Functional carrier success

```text
Base-ready success
+
functional carrier non-silent
+
task safety holds
```

## Local functional success

```text
Functional carrier success
+
value observability pass
+
leave-dataset-out pass
+
official paired replay beats AdamWParallel / bestLR
```

## Full functional success

```text
Local functional success
+
short/full run task-safe mechanism gain
+
macro or hard-stratum improvement
```

## Failure stop

```text
1. v9.2.38 boundary cannot be reproduced；
2. LQ anchor fail cannot be attributed；
3. protocol mismatch cannot be resolved；
4. current LQ remains unstable after repeated re-anchor；
5. all global base repairs fail P4/P5；
6. snapshot late attach inactive equivalence fails；
7. no-event replay preservation fails；
8. base preserved but functional carrier remains silent；
9. carrier moves but value score remains unobservable；
10. leave-dataset-out fails；
11. paired replay remains control-equivalent；
12. shuffle controls pass, indicating overfit；
13. full run gives no macro / hard-stratum / geometry gain；
14. functional breaks system gate；
15. gains are explained by QuadraticFeatureMLP；
16. any teacher/loss/fake/proxy/offload violation occurs。
```

---

# Part X. 最终解释规则

## Case A：Current LQ re-anchors

可以声明：

```text
v9.2.38 was a row-level anchor instability / protocol boundary, not LQ base collapse.
```

然后进入 snapshot late attach，但不能声明 functional success。

## Case B：Current LQ macro stable but row gate remains unstable

必须声明：

```text
LQ base remains aggregate-stable but row-level near-pass is not robust enough for official functional re-entry.
```

下一步是 protocol/gate stability repair 或 base repair，不是 functional.

## Case C：Protocol mismatch explains failure

必须声明：

```text
The apparent base failure came from protocol drift.
```

修复协议后重跑 anchor，不能直接进入 functional.

## Case D：LQ anchor truly fails

必须声明：

```text
LQ/T2 remains useful historically but is not currently a stable official base anchor.
```

进入 global base repair / pivot。

## Case E：Late attach preserves base but carrier silent

必须声明：

```text
The base is preserved, but functional carrier lacks event-time actuatability.
```

下一步修 carrier，不是 dataset patch。

## Case F：Carrier active but value unobservable

必须声明：

```text
Functional movement exists but is not control-resistant value.
```

下一步回到 role-wise FT7 / control-gap mechanism。

## Case G：Paired replay passes

可以声明：

```text
Strict PureKAN functional has local causal evidence under strong controls.
```

但 full success 仍需 short/full validation。

---

# Part XI. 最终建议

v9.2.39 的一句话策略是：

$$
\boxed{
\text{停止在未锚定的 base 上推进 functional；一次性完成 LQ anchor 归因，能重锚则 late attach，不能重锚则 base repair / pivot。}
}
$$

当前最关键的问题不是：

```text
Fashion 怎么调过；
KMNIST 怎么调过；
MNIST 是否 abstain；
再换哪个 target；
再调哪个 horizon；
再调哪个 value score。
```

而是：

```text
1. Current LQ 的 6/9 是不是 near-pass gate 边界波动？
2. 当前协议是否和历史 LQ 真的一致？
3. MLP-match baseline 是否发生定义或强度漂移？
4. 如果 LQ 仍能重锚，snapshot late attach 是否保持 checkpoint / no-event 等价？
5. 如果 base preserved，functional carrier 是否能产生 non-silent, non-AdamW movement？
6. control-gap / role-wise score 是否能预测 grounded Real-vs-control value？
7. RealFunctional 能否在 official paired replay 中超过 AdamWParallel / best LR？
```

这一步的目标不是 “再跑一个小补丁”，而是明确做出 **Anchor-or-Pivot** 决策：  
要么 LQ 重新成为 functional 的合法地基；要么诚实停止 LQ-based functional route，转向 base repair 或更深的 role-wise edge-function reset。
