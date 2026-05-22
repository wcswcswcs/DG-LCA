# DG-KAN v12.1 Codex 下一步执行计划：Good Geometry Battery 与 Functional Geometry Maintenance 首轮闭环

版本：v12.1  
执行对象：Codex / 实验自动化实现者  
核心原则：并行诊断可以提前跑；official success 必须受 gate 控制。所有公式使用 `$...$` 与 `$$...$$`，不使用方括号公式。

---

## 0. 本轮 Codex 任务的总体目标

本轮不是继续修 v9.2.x 某个 controller，也不是重启 v10 future-path。Codex 的任务是新建一个 empirical runner，完成 DG-KAN v12 的第一轮闭环：

$$
\boxed{
\text{repaired LQ base qualification}
\rightarrow
\text{passive Good Geometry Battery}
\rightarrow
\text{GeometryCertificateV0}
\rightarrow
\text{one-step/five-step functional maintenance audit}
}
$$

本轮不要求直接完成 10-seed final，也不要求直接 claim functional success。它要回答四个更本质的问题：

1. **repaired LQ 是否仍然是合格 strict FC-PureKAN base？**
2. **LQ / repaired LQ 相比 MLP 是否本身有可测 geometry advantage？**
3. **哪些 geometry metrics 真正和收敛、tail、calibration、noise split 有关？**
4. **functional geometry maintenance 在 one-step / five-step 上是否 task-safe，并且是否击败 strong controls？**

如果第 4 个问题不过，不允许进入 full training。

---

## 1. 硬约束

Codex 必须保持以下约束。任何违反约束的结果必须写入 failure table，不能写成 pass。

```text
no teacher
no self-teacher
no distillation
no loss modification
no label smoothing
no focal / margin / calibration loss
no sampler / class weight
no CPU offload as success path
no fake / proxy rows
no dataset-name branch in official controller
KAN path 不使用 PyTorch loss.backward graph as official training path
functional update 是 update rule，不是 loss
```

允许：

```text
使用 dataset_name 作为报告 slice；
报告 MNIST / Fashion-MNIST / KMNIST 分别失败在哪里；
做 leave-dataset-out / leave-stratum-out；
在 diagnostic row 中提前跑下游 scout。
```

禁止：

```text
if dataset == KMNIST: use method A
if dataset == Fashion: lower threshold
if dataset == MNIST: abstain
按数据集单独调 lambda / event threshold / score weight
因为某一个数据集被救回来就 promotion
```

---

## 2. 代码现状与本轮新增文件

上传代码中已经有可复用模块：

```text
dgkan/models/fc_purekan_lq.py
  LQSpec, basis_from_lift, lift_basis_forward, lift_basis_fwd_bwd 等 LQ primitive。

dgkan/functional/snr_gated_lq.py
  LQ role names, step operations, SNR state, existing role-aware helpers。

dgkan/functional/lq_output_space_functional.py
  旧 output-space functional target helpers；本轮只作为 control / diagnostic 参考，不作为主线。

dgkan/artifacts/writer.py
  write_csv_rows, write_json, artifact hash helpers。
```

也有需要注意的问题：

```text
dgkan/functional/geometry.py
  目前只有 coefficient_curvature，远远不够定义 Good Geometry。

experiments/run_gafu_v64.py 和 analyze_gafu_v64.py
  已被 hard-fail stub 替换，不能复活旧 proxy artifacts。
```

本轮 Codex 应新增：

```text
dgkan/functional/manifold_channel_geometry.py
  Good Geometry Battery 的可复用 metric 实现。

dgkan/functional/geometry_certificate.py
  GeometryCertificateV0 gate / Pareto 判断 / JSON summary。

experiments/run_v120_good_geometry_battery.py
  主 runner：P0-P5 并行 gated pipeline。

experiments/analyze_v120_good_geometry_battery.py
  分析器：聚合 CSV、gate、图、failure table、route decision。

experiments/plot_v120_good_geometry_battery.py
  可选；如果 analyzer 内含画图，可以不单独建。
```

不要修改或复活 `run_gafu_v64.py`。如果需要引用 v6.4，只读历史文档，不运行旧 stub。

---

## 3. 输出目录与 artifact contract

默认输出：

```text
results/v12_0_good_geometry_battery/<timestamp>/
```

必须写入：

```text
p0_contract.csv
p1_base_qualification.csv
p1_base_task_trace.csv
p1_efficiency_profile.csv
p2_passive_geometry_snapshot.csv
p2_geometry_checkpoint_trace.csv
p2_geometry_correlation.csv
p3_geometry_certificate.csv
p3_geometry_certificate.json
p4_functional_direction_audit.csv
p4_control_matrix.csv
p4_one_step_probe.csv
p4_five_step_probe.csv
p4_lambda_backtracking.csv
p5_short_run_plan_or_notrun.csv
failure_table.csv
route_decision.json
aggregate_decision.json
provenance_audit.csv
artifact_hashes.csv
figures/
```

所有 gated-not-run 阶段必须写明确 `not_run` row，原因写清楚。例如：

```text
stage = P5
status = not_run
reason = P4_functional_control_resistant_gate_failed
```

不允许用空文件代替 not-run，不允许写固定 proxy rows。

---

## 4. P0：Implementation contract 与 no-fake sanity

### 4.1 目的

P0 只确认实现和实验契约，不做科学结论。

### 4.2 方法集合

P0 至少检查：

```text
B0-MLP-same-param
B1-MLP-same-step-or-same-FLOPs
B2-QuadraticFeatureMLP-diagnostic, if available
B3-LQ-t2-h256
B4-R2-LQ-fanin-output-scale-confirmed
B5-D2-compositional-FullEdge-diagnostic, optional
```

如果代码中没有 QuadraticFeatureMLP，应先实现一个 diagnostic baseline：

```text
x -> fixed or learned quadratic features -> MLP-like classifier
```

但它只能作为 diagnostic/control，不可作为 KAN success。

### 4.3 必须记录字段

```text
method_id
model_family
candidate_id
is_strict_purekan
nonkan_param_count
edge_param_count
base_param_count
residual_param_count
mixing_param_count
manual_forward_available
manual_backward_available
manual_update_available
uses_loss_backward
uses_torch_autograd_graph
loss_is_ce_only
uses_teacher
uses_sampler_weight
uses_dataset_branch
rollback_max_error
artifact_hash
source_commit_or_unknown
```

### 4.4 P0 gate

LQ / repaired LQ official candidate 必须满足：

```text
nonkan_param_count = 0
manual_forward_available = 1
manual_backward_available = 1
manual_update_available = 1
loss_is_ce_only = 1
uses_teacher = 0
uses_sampler_weight = 0
uses_dataset_branch = 0
rollback_max_error < 1e-8
```

若 P0 不过，停止后续 official gate，但可以写 failure diagnostic。

### 4.5 图

生成：

```text
figures/p0_contract_heatmap.svg
```

横轴 method，纵轴 contract items，绿色 pass，红色 fail，灰色 not_applicable。

---

## 5. P1：Base qualification 与公平 baseline

### 5.1 目的

确认 repaired LQ 是可用主 base，并建立 MLP / QuadraticFeatureMLP 公平 baseline。

### 5.2 数据设置

第一轮使用：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
train/val/test = 使用当前项目标准 split；若已有 runner 使用 6000/1000/1000，则沿用；若 full train 可用，记录 split。
budget = 20 或 30 epoch equivalent
batch_size = 沿用 LQ runner 默认；必须记录。
```

为了加速：

```text
Lane A: base task training
Lane B: efficiency profile
Lane C: expression battery
```

三条 lane 可以并行跑；official base pass 必须等 A/B/C gate 都完成。

### 5.3 方法

```text
M0: MLP-same-param-AdamW
M1: MLP-same-step-or-same-FLOPs-AdamW
M2: QuadraticFeatureMLP-AdamW
K0: LQ-t2-h256-AdamW
K1: R2-LQ-fanin-output-scale-confirmed-AdamW
D0: D2-compositional-FullEdge-diagnostic, optional
```

如果 R2 repaired LQ 的 exact implementation 名称在 code 中不同，Codex 要用 manifest 映射：

```text
canonical_method_id = R2-LQ-fanin-output-scale-confirmed
implementation_id = <actual code id>
```

### 5.4 任务指标

```text
train_loss
val_loss
test_loss
train_acc
val_acc
test_acc_sanity
val_loss_auc_step
val_loss_auc_time
val_acc_auc_step
val_acc_auc_time
ECE
NLL
Brier
margin_mean
margin_p10
CEp95
CEp99
wrong_confidence_p95
classwise_val_acc
classwise_test_acc_sanity
confusion_matrix
```

注意：P1 selection 只使用 train/val/profiler；test 可以 sanity 记录，但不要用于调参。

### 5.5 效率指标

```text
forward_time_ms
backward_time_ms
update_time_ms
step_time_ms
forward_time_ratio_vs_MLP
backward_time_ratio_vs_MLP
step_time_ratio_vs_MLP
compact_memory_MB
compact_memory_ratio
conservative_memory_MB
conservative_memory_ratio
peak_reserved_MB
samples_per_second
kernel_count_forward
kernel_count_backward
```

必须同时报告 compact 与 conservative memory。不能只报告对自己有利的 compact memory。

### 5.6 Expression battery

实现一个轻量 expression battery，用同一 base 结构拟合或 probe 以下 target：

```text
E0 additive target
E1 pairwise product target
E2 composition target
E3 local XOR target
E4 high-frequency target
E5 noise-stress target
```

记录：

```text
expression_target_id
train_R2
val_R2
fit_loss
steps_to_R2_090
steps_to_R2_095
expression_rank
```

pairwise product 与 composition target 是核心，不是可选项。

### 5.7 P1 gate

Repaired LQ 进入 P2 的条件：

$$
near\_pass\_rate \ge 0.80,
$$

$$
\Delta Acc_{macro} \ge -0.005
$$

exploratory 可放宽为：

$$
\Delta Acc_{macro} \ge -0.01.
$$

效率：

$$
step\_q90 \le 1.10,
$$

$$
compact\_memory\_ratio \le 1.00.
$$

表达：

```text
pairwise_product_R2 不低于 LQ-t2 historical expectation；
composition target 不输 same-param MLP；
没有明显 rank collapse。
```

如果 P1 失败：

```text
停止 functional。
Codex 只允许尝试 global base repair：fanin/output scale、lift init、hidden capacity、output scale、protocol mismatch audit。
不允许按数据集调参。
```

### 5.8 图

```text
figures/p1_accuracy_delta_by_dataset_seed.svg
figures/p1_val_loss_vs_time.svg
figures/p1_efficiency_dashboard.svg
figures/p1_compact_vs_conservative_memory.svg
figures/p1_expression_battery_R2.svg
figures/p1_task_efficiency_pareto.svg
```

---

## 6. P2：Passive Good Geometry Battery

### 6.1 目的

P2 不改模型，不加 functional update。只沿训练 checkpoint 被动测量 Good Geometry Battery，判断哪些指标有意义。

### 6.2 Checkpoint schedule

对 P1 中每个主要方法保存或重放：

```text
checkpoint_step = 0
checkpoint_step = early 10%
checkpoint_step = mid 50%
checkpoint_step = late 90%
checkpoint_step = final
```

如果训练按 epoch：

```text
checkpoint_epoch = 0, 1, 3, 10, final
```

### 6.3 Reference batches

每个 checkpoint 固定抽取：

```text
R_train_small: train split 中 512 或 1024 样本
R_probe: train-stream holdout 中 512 或 1024 样本
R_val: validation 中 512 或 1024 样本
R_noise: shuffled-label diagnostic batch
R_perturb: input perturbation / augmentation diagnostic batch
```

R_noise 只能用于 diagnostic，不参与训练。

### 6.4 Expressivity / cover metrics

实现函数：

```text
compute_effective_rank(Z)
compute_lift_stats_lq(model, batch)
compute_basis_usage_lq(model, batch)
compute_role_norms_lq(model)
```

记录：

```text
effective_rank_input
effective_rank_lift
effective_rank_hidden
effective_rank_logits
lift_condition_proxy
basis_usage_entropy
dead_basis_fraction
quadratic_channel_norm
linear_channel_norm
lift_norm
output_linear_norm
role_update_share_input
role_update_share_output
role_update_share_quadratic
```

有效秩公式：

$$
p_i=\frac{\sigma_i}{\sum_j\sigma_j+\epsilon},
$$

$$
R_{eff}=\exp\left(-\sum_i p_i\log(p_i+\epsilon)\right).
$$

Basis entropy：

$$
H_{occ}=-\sum_k p_k\log(p_k+\epsilon).
$$

### 6.5 Trainability condition metrics

优先实现低成本 proxy，不要一开始求完整 Jacobian。

#### Output-kernel sketch

对 $m$ 个样本、$q$ 个随机 output probes，估计：

```text
kernel_condition_proxy
kernel_top_share_proxy
kernel_effective_rank_proxy
update_to_output_ratio
```

若手写 Jacobian 太重，先用 finite-difference output response：

$$
Jv \approx \frac{f_{\theta+\epsilon v}(x)-f_\theta(x)}{\epsilon}.
$$

对输入局部稳定可用：

$$
J_xv \approx \frac{f_\theta(x+\epsilon v)-f_\theta(x)}{\epsilon}.
$$

记录：

```text
jacobian_local_norm_median
jacobian_local_norm_p95
kernel_condition_proxy
kernel_top_eigen_share_proxy
update_to_output_ratio_median
update_to_output_ratio_p95
gradient_spike_p95
```

### 6.6 Manifold / perturbation metrics

实现：

```text
compute_input_perturbation_drift(model, batch, eps_list)
compute_aug_logit_drift(model, batch, augmentation_config)
compute_data_tangent_curvature_proxy(model, batch)
```

扰动 sensitivity：

$$
S_{local}(x,v)=\frac{\|f(x+\epsilon v)-f(x)\|_2}{\epsilon\|v\|_2+\epsilon_0}.
$$

记录：

```text
perturb_logit_drift_mean
perturb_logit_drift_p95
perturb_prediction_flip_rate
local_jacobian_norm_median
local_jacobian_norm_p95
data_tangent_curvature_proxy
augmentation_logit_drift_p95
```

### 6.7 Curvature / coefficient metrics

`dgkan/functional/geometry.py` 目前只有 `coefficient_curvature`，需要扩展。

新增：

```text
coefficient_total_variation(coeff)
coefficient_second_difference_energy(coeff)
rolewise_curvature_lq(params, basis)
geometry_debt_lq(model)
```

二阶差分：

$$
D^2c_k=c_{k+1}-2c_k+c_{k-1}.
$$

CurvatureDebt：

$$
CurvatureDebt=\sum_{e,k}p_{e,k}\left(D^2c_{e,k}\right)^2.
$$

对 LQ 的低阶 basis，如果没有 spline-like coefficient grid，就记录 rolewise quadratic coefficient norm、lift condition、perturbation curvature proxy，不要伪造 spline curvature。

### 6.8 Signal / noise metrics

使用 `dgkan/functional/snr_gated_lq.py` 中已有 role-aware helper，扩展输出。

Microbatch count：

```text
microbatch_count = 4 或 8
```

记录：

```text
snr_positive_fraction_global
snr_positive_fraction_by_role
snr_mean_by_role
snr_p10_by_role
snr_p90_by_role
offdiag_population_proxy
signal_consistency_real
signal_consistency_shuffled
noise_leak
real_noise_consistency_gap
```

SNR：

$$
SNR_k=\mu_k^2-\frac{\sigma_k^2}{b-1}.
$$

SignalConsistency：

$$
SignalConsistency=\frac{\|\bar u\|^2}{\frac{1}{S}\sum_s\|u_s-\bar u\|^2+\epsilon}.
$$

NoiseLeak：

$$
NoiseLeak=\frac{Improve_{noise}}{Improve_{real}+\epsilon}.
$$

### 6.9 Tail / calibration metrics

实现：

```text
compute_tail_metrics(logits, labels)
compute_ece(logits, labels, bins=15)
compute_margin_stats(logits, labels)
```

记录：

```text
CE_mean
CE_p90
CE_p95
CE_p99
margin_mean
margin_p10
wrong_confidence_p95
ECE
NLL
Brier
hard_tail_class_distribution
classwise_CEp99
```

### 6.10 P2 outputs

`p2_passive_geometry_snapshot.csv` 每行是一组 method / dataset / seed / checkpoint / split。

必须字段：

```text
stage
method_id
dataset
seed
checkpoint_id
split
train_step
wall_clock_sec
all task metrics
all geometry metrics
all signal/noise metrics
all efficiency metrics available
```

### 6.11 P2 分析

分析器计算 geometry 与以下目标的相关性：

```text
final_val_acc
final_val_loss
val_loss_auc_time
ECE
NLL
CEp99
MarginP10
miss_row_indicator
noise_leak
```

输出：

```text
p2_geometry_correlation.csv
```

字段：

```text
metric_name
target_name
pearson
spearman
kendall
n
sign_expected
interpretation
```

### 6.12 P2 图

```text
figures/p2_effective_rank_trajectory.svg
figures/p2_basis_entropy_trajectory.svg
figures/p2_perturb_drift_distribution.svg
figures/p2_snr_rolewise_heatmap.svg
figures/p2_tail_metrics_trajectory.svg
figures/p2_geometry_target_correlation_heatmap.svg
figures/p2_lq_vs_mlp_geometry_radar.svg
```

### 6.13 P2 失败后的尝试方向

如果 metrics 全部无相关性：

```text
增加 checkpoint frequency；
增加 tail-stratified batches；
增加 leave-dataset-out / leave-stratum-out split；
增加 perturbation directions；
增加 rolewise metrics；
不要把无效 metrics 变成 loss。
```

如果 LQ task 好但 geometry 差：

```text
保留 LQ as base；
functional maintenance 目标优先针对 geometry debt；
不要改 base。
```

如果 LQ geometry 好但 task 差：

```text
base repair 优先；
functional 不进入训练。
```

---

## 7. P3：GeometryCertificateV0

### 7.1 目的

把 P2 的指标转成 official certificate，不再依赖口头解释。

### 7.2 JSON schema

`p3_geometry_certificate.json` 结构：

```json
{
  "version": "v0",
  "reference_method": "R2-LQ-fanin-output-scale-confirmed-AdamW",
  "candidate_method": "...",
  "hard_gates": {
    "task": {},
    "efficiency": {},
    "rank_cover": {},
    "tail_calibration": {},
    "no_harm": {}
  },
  "pareto_metrics": {
    "curvature": {},
    "perturbation": {},
    "signal_noise": {},
    "cover": {},
    "tail": {},
    "calibration": {}
  },
  "pass": false,
  "failure_reasons": []
}
```

### 7.3 Hard gates

Task：

$$
Acc_C\ge Acc_B-0.005.
$$

Exploratory：

$$
Acc_C\ge Acc_B-0.01.
$$

Val loss AUC：

$$
ValLossAUC_{time,C}\le ValLossAUC_{time,B}+\tau_{auc}.
$$

Efficiency：

$$
StepTimeRatio_C\le1.10.
$$

低频 maintenance 用 amortized：

$$
AmortizedStepTimeRatio_C\le1.05.
$$

Rank / cover：

$$
R_{eff,C}\ge R_{eff,B}-\epsilon_R,
$$

$$
H_{occ,C}\ge H_{occ,B}-\epsilon_H.
$$

Tail：

$$
CEp99_C\le CEp99_B+\tau_{tail},
$$

$$
MarginP10_C\ge MarginP10_B-\tau_{margin}.
$$

### 7.4 Pareto improvement

Candidate 需要至少满足一个核心 improvement：

```text
CurvatureDebt improves by >= 10%；
PerturbLogitDrift_p95 improves by >= 10%；
SignalConsistency improves by >= 10%；
NoiseLeak decreases by >= 10%；
ECE decreases by >= 5%；
CEp99 decreases by >= 5%。
```

阈值第一轮可作为 pre-registered exploratory threshold，但不能按数据集改。

### 7.5 图

```text
figures/p3_certificate_gate_heatmap.svg
figures/p3_pareto_frontier_task_geometry_cost.svg
```

---

## 8. P4：Functional geometry maintenance one-step / five-step audit

### 8.1 目的

验证 functional correction 是否具备进入训练的资格。本阶段只在 cloned weights 上 probe，不污染真实训练。

### 8.2 Audit checkpoint

从 P1/P2 中选择：

```text
checkpoint early: loss 正在快速下降
checkpoint mid: loss 下降变慢
checkpoint late: tail / ECE / geometry debt 可能出现
```

每个 dataset / seed 至少一个 mid checkpoint。为了加速，先使用 seeds 0,1,2。

### 8.3 Batch split

每次 audit 使用：

```text
B_update: 用于计算 task step / functional correction
B_probe: train-stream holdout，用于判断 non-harm
B_valmini: validation mini-batch，只记录，不作为训练决策
B_noise: shuffled-label diagnostic，只记录
```

所有 step 在 cloned weights 上应用，然后 rollback。

### 8.4 Candidate directions

必须包含 controls：

```text
D0 TaskOnlyAdamW
D1 NoOpMatchedOverhead
D2 RandomMatchedNorm
D3 AdamWParallelDirection
D4 ShuffledPayload
D5 ShuffledEvent
D6 SNROnlyGate
D7 GeometryOnlyNoSNR
D8 SNRProjectedGeometry
D9 BasisEntropyRebalance
D10 LiftConditionRepair
D11 TailStabilityCorrection
D12 MLPAnalogGeometryMaintenance
D13 QuadraticFeatureMLPAnalogMaintenance
```

建议先实现的 LQ-specific directions：

#### D8 SNRProjectedGeometry

步骤：

```text
1. 计算 task step d_task。
2. 计算 geometry direction d_geo_raw，例如 curvature damping / basis entropy rebalance。
3. 用 rolewise SNR mask 或 soft gate 投影：d_geo = P_snr d_geo_raw。
4. 做 task-safe projection，避免与 d_task 方向冲突过大。
5. backtracking 选择 lambda。
```

组合：

$$
d_{new}=d_{task}+\lambda d_{geo}.
$$

#### D9 BasisEntropyRebalance

目标：提高 basis / lift usage entropy，不改变 output 过多。

第一版可做 conservative parameter-space direction：

```text
对 underused quadratic/lift channels 做小幅 norm rebalance；
对 over-dominant channels 做小幅 damping；
通过 logit drift 与 holdout descent gate 过滤。
```

#### D10 LiftConditionRepair

目标：改善 lift effective rank / condition。

第一版：

```text
对 lift matrix A 的 column norms / covariance 做小幅 whitening-like correction；
必须保持 strict edge-owned parameter；
不能加入 ordinary LayerNorm / non-KAN params。
```

#### D11 TailStabilityCorrection

目标：降低 CEp99、wrong_confidence_p95 或 MarginP10 tail risk。

它不能使用新的 loss，只能作为 update rule 的 candidate direction。它必须击败 AdamWParallelDirection。

### 8.5 Direction quality metrics

每个 candidate 记录：

```text
cos_new_task
projection_on_task
norm_new_over_task
angle_new_task_degree
lambda_initial
lambda_accepted
lambda_backtrack_count
lambda_zero_rate
train_descent_task
train_descent_new
holdout_descent_task
holdout_descent_new
holdout_descent_ratio
valmini_descent_task
valmini_descent_new
bad_step_new
bad_step_rate_new
rollback_error
```

公式：

$$
\cos(d_{new},d_{task})=
\frac{\langle d_{new},d_{task}\rangle}
{\|d_{new}\|\|d_{task}\|+\epsilon}.
$$

$$
HoldoutDescentRatio=
\frac{L_{probe}(\theta)-L_{probe}(\theta+d_{new})}
{L_{probe}(\theta)-L_{probe}(\theta+d_{task})+\epsilon}.
$$

### 8.6 Geometry delta metrics

记录 candidate step 后的 delta：

```text
delta_effective_rank
delta_lift_condition
delta_basis_entropy
delta_curvature_debt
delta_perturb_logit_drift_p95
delta_local_jacobian_norm_p95
delta_signal_consistency
delta_noise_leak
delta_CEp99
delta_margin_p10
delta_ECE_proxy
delta_NLL_proxy
logit_drift_p95
feature_drift_p95
```

### 8.7 One-step gate

Functional candidate 进入 five-step 必须满足：

$$
\cos(d_{new},d_{task})\ge0.85,
$$

$$
HoldoutDescentRatio\ge0.95,
$$

$$
BadStepRate\le0.02,
$$

$$
RollbackError<10^{-8}.
$$

并且至少一个 geometry metric 改善：

```text
CurvatureDebt decrease；
PerturbLogitDrift decrease；
BasisEntropy increase；
CEp99 decrease；
ECE proxy decrease；
NoiseLeak decrease。
```

### 8.8 Control-resistant gate

Functional candidate 必须击败：

```text
NoOpMatchedOverhead；
RandomMatchedNorm；
AdamWParallelDirection；
ShuffledPayload / ShuffledEvent；
SNROnlyGate；
GeometryOnlyNoSNR。
```

判定：

```text
candidate 的 accepted rows 中，GeometryCertificate delta 优于所有 controls；
candidate 的 task non-harm 不差于 controls；
candidate 的 benefit 不能完全由 MLPAnalog 或 QuadraticFeatureMLPAnalog 复现。
```

### 8.9 Five-step audit

通过 one-step 的 candidates 做 five-step cloned rollout：

```text
重复 5 个 mini-step；
每步从 train stream 取 B_update 和 B_probe；
不更新真实训练权重；
记录 cumulative task / geometry / tail / calibration。
```

通过条件：

```text
cumulative holdout loss 不差于 task-only；
bad_step_rate <= 0.02；
geometry improvement 不被第 2-5 步反转；
logit drift 不超过预设 p95；
controls 不能解释。
```

### 8.10 图

```text
figures/p4_direction_cosine_heatmap.svg
figures/p4_holdout_descent_scatter.svg
figures/p4_geometry_delta_by_candidate.svg
figures/p4_control_beat_matrix.svg
figures/p4_lambda_backtracking_histogram.svg
figures/p4_five_step_rollout_curves.svg
figures/p4_bad_step_timeline.svg
```

### 8.11 P4 失败后的 Codex 尝试方向

如果所有 functional candidate task-unsafe：

```text
实现 stronger backtracking；
减小 lambda grid；
增加 projection onto task step；
把 geometry direction 限制到 rolewise SNR-positive subspace；
不要调 dataset-specific threshold。
```

如果 candidate task-safe 但没有 geometry 改善：

```text
替换 geometry direction：basis entropy、lift condition、tail stability、perturb drift；
不要继续只做 coefficient curvature。
```

如果 candidate 改善 geometry 但被 AdamWParallel 解释：

```text
加入 orthogonal-to-AdamW component；
重做 control-gap objective；
要求 functional direction 产生 KAN-specific cover / basis effect；
不进入 short-run。
```

如果 MLP analog 同样改善：

```text
说明这是通用 optimizer geometry trick；
可以保留为 optimizer result，但不能 claim KAN-specific functional synergy；
尝试 LQ-specific basis / lift / cover direction。
```

---

## 9. P5：Short-run controlled training gate

### 9.1 运行条件

仅当 P4 one-step 与 five-step 都通过，并且至少一个 functional candidate control-resistant，才运行 P5。

否则写：

```text
P5 status = not_run
reason = P4_no_control_resistant_functional_candidate
```

### 9.2 方法

```text
T0 LQ-AdamW
T1 LQ-AdamW + NoOpMatchedOverhead
T2 LQ-AdamW + RandomMatchedNorm
T3 LQ-AdamW + AdamWParallelMaintenance
T4 LQ-AdamW + SNROnlyGate
T5 LQ-AdamW + best FunctionalGeometryMaintenance
T6 MLP-AdamW
T7 MLP + analogous GeometryMaintenance
T8 QuadraticFeatureMLP + analogous GeometryMaintenance
```

### 9.3 设置

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
budget = 20 或 30 epochs equivalent
event_frequency = low-frequency only, e.g. every K steps or geometry debt event
```

不得按数据集调 K 或 threshold。若要 event-based，使用统一规则：

```text
geometry_debt > global_threshold
and task_slack > global_threshold
and SNR/offdiag safety pass
and cooldown passed
```

### 9.4 记录

除 P1/P2/P4 指标外，新增：

```text
event_count
accepted_event_count
rejected_event_count
accept_rate
lambda_mean
lambda_p95
fallback_to_task_only_rate
amortized_functional_time_ms
amortized_step_time_ratio
functional_memory_overhead
geometry_before_after_event
refresh_recovery_steps
```

### 9.5 P5 gate

```text
Acc >= LQ-AdamW - 0.005；
ValLossAUC_time <= LQ-AdamW + tolerance；
ECE/NLL 不恶化；
GeometryCertificateV0 pass；
beats NoOp / Random / AdamWParallel / SNR-only；
benefit not fully reproduced by MLP analog / QuadraticFeatureMLP analog；
amortized overhead <= 1.05。
```

如果 P5 过，进入 v12.2 3-seed-to-10-seed confirmation。当前 v12.1 可以只实现 P5 runner 和 gate，不强制跑 10-seed。

---

## 10. 并行执行计划

为了加快进度，Codex 应按 lane 并行实现和运行。official gate 仍按顺序判断。

### Lane A：Base task / efficiency

```text
Implement P0/P1 runner for MLP, QuadraticFeatureMLP, LQ-t2, repaired LQ。
Output p1_base_qualification.csv and figures。
```

### Lane B：Geometry metrics library

```text
Implement manifold_channel_geometry.py。
Unit test each metric on random tensors and LQ model output。
Output p2_passive_geometry_snapshot.csv for cached checkpoints。
```

### Lane C：Signal / SNR / noise diagnostics

```text
Extend snr_gated_lq helpers if needed。
Compute rolewise SNR, signal consistency, shuffled-label noise leak。
```

### Lane D：Certificate analyzer

```text
Implement geometry_certificate.py。
Read P1/P2 CSV and emit p3_geometry_certificate.json/csv。
```

### Lane E：Functional direction audit

```text
Implement P4 cloned-weight one-step/five-step audit。
Must include strong controls。
```

### Lane F：Plotting and route

```text
Generate required figures。
Write route_decision.json and aggregate_decision.json。
```

Codex 可以先实现 Lane B/C/D while Lane A training runs。P4 只能 official 读取通过 P1/P3 的 base，但 diagnostic P4 scaffold 可以先用 available checkpoint smoke。

---

## 11. 命令行接口建议

Runner：

```bash
python experiments/run_v120_good_geometry_battery.py \
  --out-dir results/v12_0_good_geometry_battery/<run_name> \
  --device auto \
  --data-root data \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --packages P0,P1,P2,P3,P4 \
  --fresh
```

快速 smoke：

```bash
python experiments/run_v120_good_geometry_battery.py \
  --out-dir results/v12_0_good_geometry_battery/smoke \
  --device auto \
  --datasets MNIST \
  --seeds 0 \
  --train-size 512 \
  --val-size 256 \
  --test-size 256 \
  --packages P0,P2_METRIC_SMOKE,P4_DIRECTION_SMOKE \
  --fresh
```

Analyzer：

```bash
python experiments/analyze_v120_good_geometry_battery.py \
  --result-dir results/v12_0_good_geometry_battery/<run_name>
```

### CLI 必须支持

```text
--packages
--datasets
--seeds
--train-size
--val-size
--test-size
--checkpoint-schedule
--microbatch-count
--probe-batch-size
--functional-candidates
--controls
--fresh
--no-download
--device
```

如果 data 不存在且 `--no-download`，必须 fail cleanly，不允许 fake data。

---

## 12. 单元测试与 sanity checks

Codex 在跑正式实验前必须加 smoke tests。

### 12.1 Geometry metrics smoke

```text
random tensor effective rank finite；
constant tensor effective rank near 1；
perturbation drift finite；
ECE finite；
curvature functions handle short coefficient arrays；
SNR handles zero variance without NaN。
```

### 12.2 LQ model smoke

```text
LQ forward returns logits shape [B, C]；
manual fwd_bwd returns finite loss and grads；
apply step and rollback exact；
R2 repaired LQ spec can instantiate；
role_names match parameter groups。
```

### 12.3 P4 audit smoke

```text
candidate directions all same parameter shapes；
random matched norm has same norm as functional direction；
AdamWParallelDirection reproducible；
rollback_error < 1e-8；
NoOp leaves logits unchanged；
ShuffledPayload changes payload but not event stats.
```

---

## 13. Failure table schema

每个 failure row：

```text
stage
method_id
dataset
seed
checkpoint_id
failure_code
failure_reason
metric_name
metric_value
threshold
action_recommended
```

Failure codes：

```text
F0_contract_fail
F1_base_task_fail
F2_base_efficiency_fail
F3_expression_fail
F4_geometry_metric_nan
F5_geometry_no_signal
F6_certificate_hard_gate_fail
F7_functional_task_unsafe
F8_functional_no_geometry_gain
F9_control_explains_gain
F10_mlp_analog_explains_gain
F11_dataset_specific_only
F12_overhead_fail
F13_not_run_gate
F14_no_data_or_download_blocked
```

---

## 14. route_decision.json schema

```json
{
  "route": "R0-Unknown",
  "base_candidate": "R2-LQ-fanin-output-scale-confirmed",
  "p0_contract_pass": false,
  "p1_base_pass": false,
  "p2_geometry_battery_complete": false,
  "p3_certificate_ready": false,
  "p4_functional_audit_pass": false,
  "p5_short_run_opened": false,
  "strict_functional_success": false,
  "external_ready": false,
  "primary_blocker": "",
  "next_recommended_action": ""
}
```

Routes：

```text
R1-ContractFail
R2-BaseNotQualified
R3-BaseQualifiedGeometryMeasured
R4-GeometryCertificateReadyNoFunctional
R5-FunctionalTaskUnsafe
R6-FunctionalControlExplained
R7-FunctionalAuditPassShortRunOpened
R8-ShortRunFunctionalPass
R9-ShortRunFunctionalFail
```

---

## 15. 本轮具体实现优先级

Codex 应按以下优先级推进。

### Priority 1：实现 metric library 与 P0/P1/P2 artifact

这一步是主线，不依赖 functional update。先把 geometry 测清楚。

### Priority 2：实现 GeometryCertificateV0

不要等所有 metrics 完美再写 certificate。先用 v0 gates，把 failures 明确暴露出来。

### Priority 3：实现 P4 one-step controls

先保证 controls 完整：NoOp、Random、AdamWParallel、Shuffled、SNR-only、Geometry-only。没有 strong controls 的 functional positive 一律不算。

### Priority 4：实现 LQ-specific geometry direction

先实现 conservative directions：BasisEntropyRebalance、LiftConditionRepair、SNRProjectedGeometry。不要先复活 output-space target factory 作为主线。

### Priority 5：P5 short-run only if P4 passes

P4 不过，P5 必须 not_run。

---

## 16. 不满足条件时的尝试方向

### 16.1 P1 base 不过

尝试：

```text
R2 repaired LQ protocol reproduction；
fanin-output scale audit；
lift init audit；
output scale sweep as global repair；
hidden size only within efficiency envelope；
protocol mismatch / MLP drift audit。
```

不要尝试：

```text
dataset-specific base；
functional event rescue；
Conv / Former；
new action controller。
```

### 16.2 P2 metrics 不稳定或 NaN

尝试：

```text
reduce probe batch；
use finite-difference with larger epsilon；
clip only diagnostic perturbation, not training；
add eps to denominators；
log metric validity mask；
separate metric runtime from training runtime。
```

### 16.3 Geometry metrics 无法区分 MLP / LQ

尝试：

```text
tail-stratified geometry；
classwise hard-tail geometry；
leave-dataset-out geometry correlation；
noise-label split；
augmentation drift；
rolewise LQ metrics。
```

不要：

```text
手工改权重让图好看；
把无效 metric 加进 GeoScore。
```

### 16.4 P4 functional task-unsafe

尝试：

```text
更强 backtracking；
lambda grid = 0.005, 0.01, 0.02, 0.05；
projection to task direction；
SNR-positive role mask；
only late checkpoint audit；
restrict direction norm ratio <= 0.05 or 0.10。
```

不要：

```text
降低 holdout_descent_ratio gate；
删除 bad_step rows；
按 dataset 调 lambda。
```

### 16.5 P4 被 AdamWParallel 解释

尝试：

```text
orthogonalize d_geo against d_task；
focus on basis entropy / lift condition not CE tail；
measure KAN-specific cover delta；
require geometry gain with same CE descent。
```

不要：

```text
继续调 output target；
只报告 RealFunctional row-level beat。
```

### 16.6 P4 被 MLP analog 解释

尝试：

```text
把 claim 改成 generic optimizer geometry，不写 KAN-specific；
设计 LQ-only basis / lift maintenance；
加入 QuadraticFeatureMLP analog 进一步拆解。
```

### 16.7 只在 KMNIST 或 Fashion 有效

尝试：

```text
按 tail / rank / class confusion / perturbation drift 做 dataset-agnostic stratum；
leave-dataset-out；
报告 failure slice；
统一 global threshold。
```

不要：

```text
KMNIST-specific target；
Fashion delayed controller；
MNIST abstain。
```

---

## 17. Codex 最终交付物

Codex 完成本轮后，必须交付：

```text
1. 新增/修改代码列表。
2. 运行命令。
3. 所有 CSV/JSON artifact。
4. figures 目录。
5. route_decision.json。
6. aggregate_decision.json。
7. failure_table.csv。
8. 一份结果复盘 markdown：docs/DG-KAN_v12.0_GoodGeometryBattery_结果复盘.md。
```

结果复盘必须明确写：

```text
哪些 gate 过；
哪些 gate 没过；
哪些阶段 not_run；
functional 是否打开；
若没打开，primary blocker 是什么；
下一步是 base repair、metric repair、direction repair、还是 short-run confirm。
```

不能写：

```text
“看起来有希望，所以算 pass”；
“部分数据集有效，所以打开 full-run”；
“proxy rows support success”；
“carrier movement = functional success”。
```

---

## 18. 本轮成功定义

v12.1 不是最终论文成功。它的成功定义是：

$$
\boxed{
\text{形成一个无 fake、可复现、带 strong controls 的 Good Geometry Battery + functional audit pipeline。}
}
$$

具体成功条件：

```text
P0 contract pass；
P1 repaired LQ base status 被准确确认；
P2 passive geometry metrics 完整落盘；
P3 GeometryCertificateV0 可运行；
P4 至少完成 controls 完整的 one-step / five-step audit；
P5 只在 P4 过时打开，否则正确 not_run；
所有 artifacts 和 figures 完整；
failure taxonomy 清楚。
```

如果 P4 发现一个 control-resistant functional geometry maintenance candidate，那就是 v12.1 的强成功；如果没有，也仍然是有效推进，因为它会明确告诉我们：functional 失败是 task-safety、geometry target、control dominance，还是 MLP analog 可解释。

---

## 19. 最后一条执行提醒

本轮最重要的是**重新定义并测量问题本质**，不是把已有 functional controller 修到某个数据集过线。Codex 应优先保证：

```text
metric 清楚；
gate 清楚；
controls 清楚；
not_run 清楚；
failure 清楚。
```

只有这样，下一轮才知道该推进 functional maintenance、修 geometry metric、还是回到 base repair。
