# DG-KAN v12.20 Fail-Closed Continue 执行日志

> 日期：2026-05-25
> 环境：`conda env kan`
> 计划文档：`docs/DG-KAN_v12.20_FailClosedContinue_LabelFreeFunctional_实验结果分析与下一步计划.md`
> 原则：不伪造数据；promotion fail 后继续执行预注册 continuation diagnostics；所有 diagnostic / upper-bound / exploratory 结果均不得写成 official success。

---

## 1. 计划读取与执行目标

读取计划：

```bash
sed -n '1,260p' docs/DG-KAN_v12.20_FailClosedContinue_LabelFreeFunctional_实验结果分析与下一步计划.md
sed -n '260,620p' docs/DG-KAN_v12.20_FailClosedContinue_LabelFreeFunctional_实验结果分析与下一步计划.md
sed -n '620,1040p' docs/DG-KAN_v12.20_FailClosedContinue_LabelFreeFunctional_实验结果分析与下一步计划.md
sed -n '1040,1480p' docs/DG-KAN_v12.20_FailClosedContinue_LabelFreeFunctional_实验结果分析与下一步计划.md
sed -n '1480,1920p' docs/DG-KAN_v12.20_FailClosedContinue_LabelFreeFunctional_实验结果分析与下一步计划.md
```

本轮必须完成：

```text
Line R: code review surface + continuation queue
Line A: label-free base architecture redesign + A-UB upper-bound diagnostics
Line C: target reset diagnostic C-T1
Line T: visibility v4 / T1A-T1B / T2 upper-bound / no-go
Line I: exploratory actuator capacity, not promotable
Line B: P3/P4 gate audit and not-opened record
Line D: classic no-BSpline monitor
zip: required code review packet
```

---

## 2. 代码修改记录

### 2.1 `dgkan/models/fc_purekan_primitives.py`

新增 v12.20 label-free projector init tokens：

```text
multiframebankp:
  PCA + SRHT/Rademacher + augmentation-stable SVD + local-block + low-frequency frame bank。

selfcondresp:
  activation covariance eigenvectors + random-cotangent residual frame。

covadaptp:
  PCA/covariance initialized projector；配合 runner 中 unlabeled epoch adaptation。
```

这些路径均不读取 label、CE vector、validation/test 或 dataset name。

### 2.2 `experiments/run_v1218_b320_label_free_ablation.py`

新增 y_stats diagnostic modes：

```text
unsupervised_kmeans:
  A-UB4；基于 x_train 的无监督聚类伪类；diagnostic only，不 promotion。

small_label_fraction:
  A-UB5；只给少量样本保留真实 label；oracle upper-bound diagnostic only，不 promotion。
```

新增 candidates：

```text
A23-MultiFrameBank-labelFree
A24-MultiFrameBankDirect125-labelFree
A25-SelfConditionResidualP-labelFree
A26-CovAdaptP-labelFree
A27-MultiFrameLowFreqBias-labelFree
A28-unsupervisedClusterTrainProbe-diagnostic
A29-oracleSmallLabelTrainProbe-diagnostic
```

新增 `apply_unlabeled_projector_adaptation()`，用于 A26 在 epoch 开始时用 unlabeled activation covariance 低频更新 `quad_proj`。该更新不读取 label/CE。

### 2.3 `experiments/run_v1252_efficiency_functional_manifold.py`

修复 smoke 中触发的数值 blocker：

```text
_ridge_coupling:
  torch.linalg.solve 在小批量奇异矩阵上失败。
  修复为 solve -> lstsq -> pinv fallback。
```

该修复不改变 metric 定义，只让同一个 ridge system 在 rank-deficient 情况下可计算。

### 2.4 `experiments/run_v1220_failclosed_continue_label_free_functional.py`

新增 v12.20 continuation runner，写出：

```text
v1220_code_review_manifest.csv
v1220_continuation_execution_manifest.csv
v1220_stop_reason_audit.csv
v1220_diff_intent_table.csv
v1220_core_symbol_map.json
v1220_required_artifact_manifest.csv
v1220_label_free_base_candidates.csv
v1220_label_free_upper_bound.csv
v1220_linec_target_reset.csv
v1220_visibility_scores.csv
v1220_visibility_no_go_upper_bound.csv
v1220_actuator_capacity.csv
v1220_functional_p3.csv
v1220_functional_p4_short.csv
v1220_classic_family_status.csv
v1220_classic_family_new_hypothesis.csv
v1220_classic_family_linec.csv
v1220_route_decision.json
v1220_hash_manifest.json
v1220_code_review_packet.zip
```

---

## 3. Smoke 与 blocker 修复

### 3.1 初始语法检查

```bash
conda run -n kan python -m py_compile \
  dgkan/models/fc_purekan_primitives.py \
  experiments/run_v1218_b320_label_free_ablation.py \
  experiments/run_v1220_failclosed_continue_label_free_functional.py
```

结果：exit 0。

### 3.2 smoke: new frame tokens

命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id v1220_smoke_new_frame_tokens \
  --out-dir results/v12_20_failclosed_continue_label_free_functional/smoke_linea \
  --artifact-prefix v1220_smoke_new_frame_tokens \
  --result-stage V1220_SMOKE_LINEA \
  --summary-stage V1220_SMOKE_LINEA_SUMMARY \
  --route-stage V1220_SMOKE_LINEA_ROUTE \
  --result-scope smoke_new_frame_tokens \
  --datasets MNIST \
  --seeds 0 \
  --ablation-ids A0-labelInit,A23-MultiFrameBank-labelFree,A24-MultiFrameBankDirect125-labelFree,A25-SelfConditionResidualP-labelFree,A26-CovAdaptP-labelFree,A27-MultiFrameLowFreqBias-labelFree,A28-unsupervisedClusterTrainProbe-diagnostic,A29-oracleSmallLabelTrainProbe-diagnostic \
  --train-size 64 --val-size 64 --test-size 64 \
  --epochs 1 --batch-size 32 \
  --task-compile-warmup-steps 0 \
  --measure-linec 1 --linec-batch-size 16 --linec-sketch-batch-size 4 --linec-sketch-dim 4 \
  --no-download
```

第一次失败：

```text
torch.linalg.solve: input matrix is singular
```

修复：

```text
experiments/run_v1252_efficiency_functional_manifold.py::_ridge_coupling
增加 solve -> lstsq -> pinv fallback。
```

第二次失败：

```text
torch.linalg.eigh failed to converge in signal_reservoir_metrics_detailed
```

修复：

```text
experiments/run_v1218_b320_label_free_ablation.py::signal_reservoir_metrics_detailed
对 Gram matrix 显式对称化；
nonfinite grad sketch 直接返回 NaN metrics 使 gate fail；
eigh fallback: jitter sweep -> CPU eigh -> NaN metrics。
```

第三次运行结果：exit 0。

smoke 关键输出：

```text
rows = 10
summary_rows = 10
label_free_ablation_available_count = 5
best_label_free_candidate = A25-SelfConditionResidualP-labelFree
best_label_free_mean_delta_vs_A0 = -0.484375
```

注意：这是 smoke，不进入 official 结论。

### 3.3 smoke: v1220 continuation runner

第一次 runner smoke 失败：

```text
TypeError: int() argument must be ... not 'list'
```

修复：

```text
experiments/run_v1220_failclosed_continue_label_free_functional.py
visibility pass_flag 中将 `and rows` 改为 `and bool(rows)`。
```

重跑命令：

```bash
conda run -n kan python experiments/run_v1220_failclosed_continue_label_free_functional.py \
  --run-id v1220_runner_smoke \
  --out-dir results/v12_20_failclosed_continue_label_free_functional/smoke_runner \
  --linea-root results/v12_20_failclosed_continue_label_free_functional/smoke_linea \
  --v1219-dir results/v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry/official_from_v1218_v1217_v1216_artifacts \
  --v1218-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/support_expand_seed012_seed345_b128_w10_12_15
```

结果：exit 0。

smoke runner 输出：

```text
route = R2-LabelFreeSignalFrameMissing
minimum_success = Success D
continuation_missing_count = 0
code_review_packet_zip_entries = 46
```

---

## 4. Official Line A continuation: epochs=3

共同设置：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
train_size = 1024
val_size = 512
test_size = 512
batch_size = 128
epochs = 3
LineC batch = 64
LineC sketch_batch = 32
LineC sketch_dim = 24
seed_base = 1220000
```

### 4.1 MNIST epochs=3

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id v1220_linea_e3_MNIST_seed012 \
  --out-dir results/v12_20_failclosed_continue_label_free_functional/official_continuation/linea/e3_MNIST \
  --artifact-prefix v1220_linea_e3_MNIST \
  --result-stage V1220_LINEA_E3 \
  --summary-stage V1220_LINEA_E3_SUMMARY \
  --route-stage V1220_LINEA_E3_ROUTE \
  --result-scope v1220_label_free_architecture_redesign_e3 \
  --smoke-not-official 0 --official-training-result-available 1 \
  --protocol-note "v12.20 Line A official continuation: label-free architecture redesign A23-A27 plus A-UB controls; MNIST seeds 0/1/2 epochs=3" \
  --route-impact "feeds v12.20 fail-closed continuation; no promotion without task+LineC pass" \
  --device cuda:0 --datasets MNIST --seeds 0,1,2 \
  --ablation-ids A0-labelInit,A1-noYForStats,A5-orthogonalP-labelFree,A20-shuffledLabelTrainProbe-diagnostic,A21-randomClassCentroid-diagnostic,A22-permutedClassMeanP-diagnostic,A23-MultiFrameBank-labelFree,A24-MultiFrameBankDirect125-labelFree,A25-SelfConditionResidualP-labelFree,A26-CovAdaptP-labelFree,A27-MultiFrameLowFreqBias-labelFree,A28-unsupervisedClusterTrainProbe-diagnostic,A29-oracleSmallLabelTrainProbe-diagnostic \
  --seed-base 1220000 --train-size 1024 --val-size 512 --test-size 512 --epochs 3 --batch-size 128 \
  --task-compile-warmup-steps 2 --measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 32 --linec-sketch-dim 24 --no-download
```

结果：

```text
rows = 45
summary_rows = 15
best_label_free_candidate = A1-noYForStats
best_label_free_mean_delta_vs_A0 = -0.0078125
```

### 4.2 Fashion-MNIST epochs=3

同一参数，`--device cuda:1 --datasets Fashion-MNIST`，输出目录：

```text
results/v12_20_failclosed_continue_label_free_functional/official_continuation/linea/e3_FashionMNIST
```

结果：

```text
rows = 45
summary_rows = 15
best_label_free_candidate = A1-noYForStats
best_label_free_mean_delta_vs_A0 = -0.014322916666666666
```

### 4.3 KMNIST epochs=3

同一参数，`--device cuda:2 --datasets KMNIST`，输出目录：

```text
results/v12_20_failclosed_continue_label_free_functional/official_continuation/linea/e3_KMNIST
```

结果：

```text
rows = 45
summary_rows = 15
best_label_free_candidate = A1-noYForStats
best_label_free_mean_delta_vs_A0 = 0.001953125
```

---

## 5. Official Line A continuation: epochs=8

共同设置与 epochs=3 相同，仅 `--epochs 8`。

### 5.1 MNIST epochs=8

输出目录：

```text
results/v12_20_failclosed_continue_label_free_functional/official_continuation/linea/e8_MNIST
```

结果：

```text
rows = 45
summary_rows = 15
best_label_free_candidate = A1-noYForStats
best_label_free_mean_delta_vs_A0 = 0.0
```

### 5.2 Fashion-MNIST epochs=8

输出目录：

```text
results/v12_20_failclosed_continue_label_free_functional/official_continuation/linea/e8_FashionMNIST
```

结果：

```text
rows = 45
summary_rows = 15
best_label_free_candidate = A1-noYForStats
best_label_free_mean_delta_vs_A0 = -0.005208333333333333
```

### 5.3 KMNIST epochs=8

输出目录：

```text
results/v12_20_failclosed_continue_label_free_functional/official_continuation/linea/e8_KMNIST
```

结果：

```text
rows = 45
summary_rows = 15
best_label_free_candidate = A1-noYForStats
best_label_free_mean_delta_vs_A0 = 0.007161458333333333
```

Official Line A continuation 总训练/Line C 行数：

```text
270 = 3 datasets * 2 epoch budgets * 3 seeds * 15 methods
```

---

## 6. v12.20 continuation runner official

命令：

```bash
conda run -n kan python experiments/run_v1220_failclosed_continue_label_free_functional.py \
  --run-id official_continuation_e3_e8_seed012_joinfix \
  --out-dir results/v12_20_failclosed_continue_label_free_functional/official_continuation \
  --linea-root results/v12_20_failclosed_continue_label_free_functional/official_continuation/linea \
  --v1219-dir results/v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry/official_from_v1218_v1217_v1216_artifacts \
  --v1218-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/support_expand_seed012_seed345_b128_w10_12_15
```

结果：exit 0。

关键输出：

```text
route = R2-LabelFreeSignalFrameMissing
minimum_success = Success D
line_r_pass = 1
linea_rows = 270
label_free_candidate_rows = 13
label_free_official_pass_count = 0
label_free_best_candidate = A1-noYForStats
upper_bound_rows = 6
linec_target_reset_rows = 216
c_t1_spearman_noise = 0.03645397607489652
c_t1_spearman_reservoir = 0.1970492654002053
c_t1_value_source_pass = 0
T1A_visibility_pass = 0
T1A_auc_joint_min = 0.0
T1A_precision_joint_min = 0.0
T1A_recall_joint_min = 0.0
T2_upper_bound_auc_joint_min = 0.03773584905660377
actuator_capacity_rows = 198
actuator_safe_movement_rows = 0
actuator_release_audit_rows = 0
p4_open = 0
continuation_missing_count = 0
required_artifact_missing_count = 0
figure_count = 13
code_review_packet_zip_entries = 46
soft_target_r2_noise_min_T1A = -11956.931957256009
soft_target_r2_reservoir_min_T1A = -401245.5453505141
soft_target_r2_noise_min_T2_upper_bound = -13474.094195762538
soft_target_r2_reservoir_min_T2_upper_bound = -261669.11702240884
```


### 6.1 join 修复与重跑说明

首次 official runner 输出中 soft-release regression 字段为空。检查发现不是 target 无方差，而是 v12.19 的 T1 feature `row_id` 与 T3 audit target `row_id` 最后一段 index 不一致，完整 `row_id` join 只命中 1 行。

核验命令：

```bash
python - <<'PY'
import csv
from pathlib import Path
base=Path('results/v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry/official_from_v1218_v1217_v1216_artifacts')
with (base/'v1219_T1_deployable_features.csv').open(newline='') as f:
    t1=list(csv.DictReader(f))
with (base/'v1219_T3_audit_targets.csv').open(newline='') as f:
    t3={r['row_id']:r for r in csv.DictReader(f)}
print('t1 row ids',len(set(r['row_id'] for r in t1)),'t3 ids',len(t3),'intersect',len(set(r['row_id'] for r in t1)&set(t3)))
PY
```

结果：

```text
t1 row ids = 1728
t3 ids = 1728
intersect = 1
```

修复：`experiments/run_v1220_failclosed_continue_label_free_functional.py::write_visibility_v4` 改为按稳定键 join：

```text
source_run::dataset::seed::window::method::sketch_id
```

重跑 official continuation runner：

```bash
conda run -n kan python experiments/run_v1220_failclosed_continue_label_free_functional.py \
  --run-id official_continuation_e3_e8_seed012_joinfix \
  --out-dir results/v12_20_failclosed_continue_label_free_functional/official_continuation \
  --linea-root results/v12_20_failclosed_continue_label_free_functional/official_continuation/linea \
  --v1219-dir results/v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry/official_from_v1218_v1217_v1216_artifacts \
  --v1218-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/support_expand_seed012_seed345_b128_w10_12_15
```

join 修复后 soft-release regression 结果：

```text
T1A soft_target_r2_noise_min = -11956.931957256009
T1A soft_target_r2_reservoir_min = -401245.5453505141
T1A+T1B soft_target_r2_noise_min = -11956.931957256009
T1A+T1B soft_target_r2_reservoir_min = -401245.5453505141
T2 upper-bound soft_target_r2_noise_min = -13474.094195762538
T2 upper-bound soft_target_r2_reservoir_min = -261669.11702240884
```

这些是实际计算结果，不能解释为 pass；它们支持 soft target 也不可见。

---

## 7. 关键 artifacts

输出目录：

```text
results/v12_20_failclosed_continue_label_free_functional/official_continuation
```

主要 artifacts：

```text
v1220_route_decision.json
v1220_code_review_manifest.csv
v1220_continuation_execution_manifest.csv
v1220_stop_reason_audit.csv
v1220_diff_intent_table.csv
v1220_core_symbol_map.json
v1220_required_artifact_manifest.csv
v1220_label_free_base_candidates.csv
v1220_label_free_upper_bound.csv
v1220_linec_target_reset.csv
v1220_visibility_scores.csv
v1220_visibility_no_go_upper_bound.csv
v1220_visibility_leaveout.csv
v1220_actuator_capacity.csv
v1220_functional_p3.csv
v1220_functional_p4_short.csv
v1220_classic_family_status.csv
v1220_classic_family_new_hypothesis.csv
v1220_classic_family_linec.csv
v1220_hash_manifest.json
v1220_code_review_packet.zip
```

Figures：

```text
fig_v1220_route_waterfall.svg
fig_v1220_failclosed_continue_matrix.svg
fig_v1220_label_free_task_linec_pareto.svg
fig_v1220_projection_frame_diagnostics.svg
fig_v1220_linec_target_reset_correlation.svg
fig_v1220_visibility_T1A_T1B_ablation.svg
fig_v1220_visibility_leaveout_heatmap.svg
fig_v1220_upper_bound_gap.svg
fig_v1220_actuator_capacity_vs_controls.svg
fig_v1220_functional_p3_control_gap.svg
fig_v1220_p4_if_open_loss_vs_step.svg
fig_v1220_p4_if_open_linec_trajectory.svg
fig_v1220_classic_status_dashboard.svg
```

---

## 8. 审计 zip

路径：

```text
results/v12_20_failclosed_continue_label_free_functional/official_continuation/v1220_code_review_packet.zip
```

sha256：

```text
e4dac025396e626032bd1fddbf639f81dec0e263c3ab4d8b4f6fa636841f68d4
```

zip 自检：

```text
entries = 46
contains code/experiments/run_v1220_failclosed_continue_label_free_functional.py = True
contains code/experiments/run_v1218_b320_label_free_ablation.py = True
contains code/dgkan/models/fc_purekan_primitives.py = True
contains code/experiments/run_v1252_efficiency_functional_manifold.py = True
contains review_artifacts/v1220_route_decision.json = True
contains review_artifacts/v1220_label_free_base_candidates.csv = True
contains review_artifacts/v1220_visibility_scores.csv = True
contains review_artifacts/v1220_actuator_capacity.csv = True
```

---

## 9. 最终验证

语法检查：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1220_failclosed_continue_label_free_functional.py \
  experiments/run_v1218_b320_label_free_ablation.py \
  experiments/run_v1252_efficiency_functional_manifold.py \
  dgkan/models/fc_purekan_primitives.py
```

结果：exit 0。

Trailing whitespace 检查：

```text
experiments/run_v1220_failclosed_continue_label_free_functional.py trailing_ws_count = 0
experiments/run_v1218_b320_label_free_ablation.py trailing_ws_count = 0
experiments/run_v1252_efficiency_functional_manifold.py trailing_ws_count = 0
dgkan/models/fc_purekan_primitives.py trailing_ws_count = 0
```

---

## 10. Git 工作区提醒

本轮直接相关新增/修改：

```text
M dgkan/models/fc_purekan_primitives.py
M experiments/run_v1252_efficiency_functional_manifold.py
M/?? experiments/run_v1218_b320_label_free_ablation.py
A experiments/run_v1220_failclosed_continue_label_free_functional.py
A docs/DG-KAN_v12.20_FailClosedContinue_LabelFreeFunctional_执行日志.md
A docs/DG-KAN_v12.20_FailClosedContinue_LabelFreeFunctional_实验结果复盘.md
```

注意：仓库中已有一些未跟踪/历史实验文件状态，本轮没有回滚用户或历史变更。
