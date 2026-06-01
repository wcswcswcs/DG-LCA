# DG-KAN v12.19 执行日志

> 计划文档：`docs/DG-KAN_v12.19_B320Deconfounding_TrueLossAgnosticVisibility_FunctionalReentry_独立分析与下一步计划.md`  
> 执行环境：`conda env kan`  
> 工作目录：`/home/chengshun.wang/DG-LCA`  
> 原则：不编造数据；失败就是失败；P3/P4 必须按 gate 关闭。

---

## 1. 本轮目标理解

v12.19 不是继续扫 functional lambda，而是先处理三个硬问题：

1. B320 current 的强 task anchor 是否依赖 label-informed trainprobe initialization。
2. 是否能找到 label-free B320-like candidate，同时过 task + Line C non-tearing。
3. Line T 必须清理成真正 T1 pre-commit loss-agnostic features，剔除 method identity、clone-probe/post-response drift、label/CE target。

只有 Line T T1-only visibility 通过后，才允许打开 Line I actuator dictionary；只有 Line R/A/C/T/I 全过后才允许 P4。

---

## 2. 代码修改

本轮做了 3 个代码层修改，均纳入 v12.19 zip：

1. `dgkan/models/fc_purekan_primitives.py`
   - 新增 label-free projector 初始化 token：
     - `pcaorthomixp`
     - `augstablep`
     - `randomcotangentstablep`
     - `covwhitenp`
     - `blocklocalaugstablep`
   - 这些 token 只用 train-stream input geometry / augmentation proxy / random cotangent geometry，不使用真实 label 或 CE direction。

2. `experiments/run_v1218_b320_label_free_ablation.py`
   - 新增 v12.19 candidates：
     - A15-PCAOrthoMix-labelFree
     - A16-AugStableP-labelFree
     - A17-RandomCotangentStableP-labelFree
     - A18-InputCovWhitenedOrthoP-labelFree
     - A19-BlockLocalAugStableP-labelFree
     - A20-shuffledLabelTrainProbe-diagnostic
     - A21-randomClassCentroid-diagnostic
     - A22-permutedClassMeanP-diagnostic
   - 新增 `build_y_stats_for_mode`，用于 A20-A22 diagnostic y_stats 模式。
   - 新增 P basis alignment diagnostics：
     - `P_energy_on_top_pca`
     - `P_energy_on_aug_stable_subspace`
     - `P_condition`
     - `quad_feature_std_*`
     - branch/readout norms
   - 修正 summary 里的 label-free candidate 选择：只把 `strict_label_free_rows > 0` 的候选计入 label-free best。

3. `experiments/run_v1219_b320_deconfounding_true_lossagnostic_visibility.py`
   - 新增 v12.19 汇总 runner：
     - Line R semantic audit
     - Line A gate summary
     - Line C protocol autopsy
     - T1/T2/T3 feature tiering
     - T1-only visibility scorer
     - Line I/P4 gate status
     - figures
     - hash manifest
     - code review zip

---

## 3. 初始化 smoke

目的：先确认 A15-A22 新 token / diagnostic y_stats 模式在 CUDA 上能真实训练和测 Line C，不崩溃。

命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id v1219_smoke_init_paths \
  --out-dir results/v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry/smoke_init_paths \
  --artifact-prefix v1219_smoke_init_paths \
  --result-stage V1219_SMOKE_INIT_PATHS \
  --summary-stage V1219_SMOKE_INIT_PATHS_SUMMARY \
  --route-stage V1219_SMOKE_INIT_PATHS_ROUTE \
  --result-scope smoke_init_paths \
  --datasets MNIST \
  --seeds 0 \
  --ablation-ids A15-PCAOrthoMix-labelFree,A16-AugStableP-labelFree,A17-RandomCotangentStableP-labelFree,A18-InputCovWhitenedOrthoP-labelFree,A19-BlockLocalAugStableP-labelFree,A20-shuffledLabelTrainProbe-diagnostic,A21-randomClassCentroid-diagnostic,A22-permutedClassMeanP-diagnostic \
  --train-size 64 --val-size 64 --test-size 64 \
  --epochs 1 --batch-size 32 \
  --task-compile-warmup-steps 0 \
  --measure-linec 1 --linec-batch-size 16 --linec-sketch-batch-size 4 --linec-sketch-dim 4 \
  --no-download
```

结果：

```text
rows = 10
summary_rows = 10
label_free_ablation_available_count = 5
best_label_free_candidate = A19-BlockLocalAugStableP-labelFree
```

该 smoke 只验证路径，不用于 official conclusion。

---

## 4. Line A 正式 Group A/B/C

计划要求的 recovered locked-anchor budget：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
train_size = 1024
val_size = 512
test_size = 512
epochs = 3
batch_size = 128
Line C = enabled, linec_batch_size=64, sketch_dim=24
```

命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id v1219_linea_groupabc_seed012_3x3 \
  --out-dir results/v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry/official_from_v1218_v1217_v1216_artifacts \
  --artifact-prefix v1219_b320_deconfounding_linea \
  --result-stage V1219_B320_DECONFOUNDING_LINEA \
  --summary-stage V1219_B320_DECONFOUNDING_LINEA_SUMMARY \
  --route-stage V1219_B320_DECONFOUNDING_LINEA_ROUTE \
  --result-scope v1219_recovered_anchor_budget_groupabc \
  --smoke-not-official 0 \
  --official-training-result-available 1 \
  --protocol-note "v12.19 recovered locked-anchor budget: datasets MNIST/Fashion-MNIST/KMNIST, seeds 0/1/2, train_size=1024, val/test=512, epochs=3, batch=128; Group A+B+C with same Line C calculator" \
  --route-impact "feeds v12.19 Line A deconfounding; does not by itself open P3/P4" \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --ablation-ids A0-labelInit,A1-noYForStats,A5-orthogonalP-labelFree,A11-orthogonalP-directRead125-labelFree,A12-orthogonalP-identityAmp150-labelFree,A15-PCAOrthoMix-labelFree,A16-AugStableP-labelFree,A17-RandomCotangentStableP-labelFree,A18-InputCovWhitenedOrthoP-labelFree,A19-BlockLocalAugStableP-labelFree,A20-shuffledLabelTrainProbe-diagnostic,A21-randomClassCentroid-diagnostic,A22-permutedClassMeanP-diagnostic \
  --seed-base 1219000 \
  --train-size 1024 --val-size 512 --test-size 512 \
  --epochs 3 --batch-size 128 \
  --task-compile-warmup-steps 2 \
  --measure-linec 1 --linec-batch-size 64 --linec-sketch-batch-size 32 --linec-sketch-dim 24 \
  --no-download
```

输出：

```text
v1219_b320_deconfounding_linea_ablation.csv
v1219_b320_deconfounding_linea_summary.csv
v1219_b320_deconfounding_linea_route.json
```

关键结果：

```text
rows = 135
summary_rows = 15
best strict label-free candidate by task delta = A1-noYForStats
best_label_free_mean_delta_vs_A0 = -0.011284722222222222
```

该 best 没有达到 task-side gate `>= -0.005`，且所有 label-free candidates 的 Line C non-tearing pass rate 都是 0。

---

## 5. v12.19 汇总 runner

命令：

```bash
conda run -n kan python experiments/run_v1219_b320_deconfounding_true_lossagnostic_visibility.py \
  --run-id official_from_v1218_v1217_v1216_artifacts_linea_groupabc_t1_v3_feature_provenance_fix \
  --out-dir results/v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry/official_from_v1218_v1217_v1216_artifacts \
  --v1218-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/support_expand_seed012_seed345_b128_w10_12_15 \
  --v1216-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_sketchdim24_rank5_b128_w10_12_15 \
  --autopsy-dataset MNIST \
  --autopsy-seed 0 \
  --autopsy-train-size 1024 \
  --autopsy-val-size 512 \
  --autopsy-test-size 512 \
  --autopsy-epochs 3 \
  --batch-size 128 \
  --no-download
```

第一次汇总后自检发现 `v1219_feature_provenance_strict_v2.csv` 没有单独写出，只在 tier manifest 内。已修复 runner，并用上述命令重跑。

最终 route：

```text
route = R2-B320LabelInitDependenceDetected
fail_reason = no label-free B320-like candidate passed task+LineC; current B320 remains label-informed diagnostic anchor
T1_only_visibility_pass = 0
p4_open = 0
```

---

## 6. 最终产物目录

```text
results/v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry/official_from_v1218_v1217_v1216_artifacts
```

核心 artifacts：

```text
v1219_route_decision.json
v1219_code_review_delta.csv
v1219_core_semantics_audit.md
v1219_feature_provenance_strict_v2.csv
v1219_forbidden_feature_blocklist.csv
v1219_b320_construction_trace.json
v1219_linec_metric_trace.json
v1219_linea_candidate_gate_summary.csv
v1219_P_basis_alignment_diagnostics.csv
v1219_linea_reservoir_decomposition.csv
v1219_linec_protocol_autopsy.csv
v1219_visibility_feature_tier_manifest.csv
v1219_T1_deployable_features.csv
v1219_T2_clone_probe_diagnostic_features.csv
v1219_T3_audit_targets.csv
v1219_visibility_scores_T1_only.csv
v1219_visibility_scores_T1_plus_T2_diagnostic.csv
v1219_visibility_leaveout_T1_only.csv
v1219_visibility_support_audit.csv
v1219_linei_gate_status.csv
v1219_p4_gate_status.csv
v1219_classic_family_status.csv
v1219_hash_manifest.json
```

Figures：

```text
fig_v1219_route_dashboard.svg
fig_v1219_b320_label_init_ablation_task.svg
fig_v1219_b320_label_init_ablation_linec.svg
fig_v1219_linec_protocol_mismatch.svg
fig_v1219_reservoir_decomposition_a5_vs_mlp.svg
fig_v1219_noise_signal_energy_a5_vs_mlp.svg
fig_v1219_visibility_pr_curve_T1_only.svg
fig_v1219_visibility_auc_leaveout_T1_only.svg
fig_v1219_visibility_support_heatmap.svg
fig_v1219_feature_tier_ablation.svg
fig_v1219_actuator_control_gap.svg
fig_v1219_p4_short_run_if_opened.svg
fig_v1219_label_free_task_delta.svg
fig_v1219_label_free_auc_time.svg
fig_v1219_label_free_linec_scatter.svg
fig_v1219_real_residual_energy_ratio.svg
fig_v1219_noise_signal_energy_ratio.svg
fig_v1219_P_basis_alignment.svg
fig_v1219_A0_locked_vs_recovered_protocol.svg
```

---

## 7. 审计 zip

最终 zip：

```text
results/v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry/official_from_v1218_v1217_v1216_artifacts/v1219_code_review_packet.zip
```

sha256：

```text
e551e83cd531af0bf1cf74cae48fd5ec0fed38fb2ce0292e22b3e9cd4aba7e97
```

zip 自检：

```text
zip_entries = 59
contains code/experiments/run_v1219_b320_deconfounding_true_lossagnostic_visibility.py = yes
contains code/experiments/run_v1218_b320_label_free_ablation.py = yes
contains code/dgkan/models/fc_purekan_primitives.py = yes
contains review_artifacts/v1219_feature_provenance_strict_v2.csv = yes
contains review_artifacts/v1219_core_semantics_audit.md = yes
contains review_artifacts/v1219_visibility_scores_T1_only.csv = yes
```

---

## 8. 验证命令

语法检查：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1219_b320_deconfounding_true_lossagnostic_visibility.py \
  experiments/run_v1218_b320_label_free_ablation.py \
  dgkan/models/fc_purekan_primitives.py
```

结果：exit 0。

Trailing whitespace 检查：

```text
experiments/run_v1219_b320_deconfounding_true_lossagnostic_visibility.py trailing_ws_count = 0
experiments/run_v1218_b320_label_free_ablation.py trailing_ws_count = 0
dgkan/models/fc_purekan_primitives.py trailing_ws_count = 0
```

---

## 9. Git 工作区提醒

执行前后工作区已有大量 unrelated docs 删除/新增和若干旧实验脚本修改。本轮没有回滚这些已有变更。本轮直接相关的新增/修改是：

```text
M dgkan/models/fc_purekan_primitives.py
M experiments/run_v1218_b320_label_free_ablation.py
A experiments/run_v1219_b320_deconfounding_true_lossagnostic_visibility.py
A docs/DG-KAN_v12.19_B320Deconfounding_TrueLossAgnosticVisibility_FunctionalReentry_执行日志.md
A docs/DG-KAN_v12.19_B320Deconfounding_TrueLossAgnosticVisibility_FunctionalReentry_实验结果复盘.md
```

---

## 10. 2026-05-24 23:43 +08 完成性复核

用户追问 `docs/DG-KAN_v12.19_B320Deconfounding_TrueLossAgnosticVisibility_FunctionalReentry_独立分析与下一步计划.md` 是否完成后，执行了计划逐项回读与 artifact 存在性复核。本节只记录复核命令与客观结果，不新增实验数据。

读取计划全文：

```bash
sed -n '1,260p' docs/DG-KAN_v12.19_B320Deconfounding_TrueLossAgnosticVisibility_FunctionalReentry_独立分析与下一步计划.md
sed -n '260,620p' docs/DG-KAN_v12.19_B320Deconfounding_TrueLossAgnosticVisibility_FunctionalReentry_独立分析与下一步计划.md
sed -n '620,1040p' docs/DG-KAN_v12.19_B320Deconfounding_TrueLossAgnosticVisibility_FunctionalReentry_独立分析与下一步计划.md
sed -n '1040,1480p' docs/DG-KAN_v12.19_B320Deconfounding_TrueLossAgnosticVisibility_FunctionalReentry_独立分析与下一步计划.md
```

读取 route decision 关键字段：

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('results/v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry/official_from_v1218_v1217_v1216_artifacts/v1219_route_decision.json')
d=json.loads(p.read_text())
keys=['route','fail_reason','line_r_code_semantics_pass','implementation_readback_missing_refs','cr0_cr15_missing_or_unknown','linea_pass','linea_official_label_free_pass_count','linea_best_candidate_by_task_delta','linea_best_candidate_mean_delta_vs_A0','linec_protocol_impl_consistency_pass','T1_only_visibility_pass','line_i_open','p4_open','code_review_packet_zip','code_review_packet_zip_sha256','hash_manifest_entries']
for k in keys:
    print(f'{k}={d.get(k)}')
PY
```

结果：

```text
route=R2-B320LabelInitDependenceDetected
fail_reason=no label-free B320-like candidate passed task+LineC; current B320 remains label-informed diagnostic anchor
line_r_code_semantics_pass=1
implementation_readback_missing_refs=0
cr0_cr15_missing_or_unknown=0
linea_pass=0
linea_official_label_free_pass_count=0
linea_best_candidate_by_task_delta=A1-noYForStats
linea_best_candidate_mean_delta_vs_A0=-0.011284722222222222
linec_protocol_impl_consistency_pass=1
T1_only_visibility_pass=0
line_i_open=0
p4_open=0
code_review_packet_zip=results/v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry/official_from_v1218_v1217_v1216_artifacts/v1219_code_review_packet.zip
code_review_packet_zip_sha256=e551e83cd531af0bf1cf74cae48fd5ec0fed38fb2ce0292e22b3e9cd4aba7e97
hash_manifest_entries=47
```

检查计划要求的主要 artifacts 是否存在：

```bash
python - <<'PY'
from pathlib import Path
base=Path('results/v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry/official_from_v1218_v1217_v1216_artifacts')
checks=[
'v1219_code_review_delta.csv','v1219_core_semantics_audit.md','v1219_feature_provenance_strict_v2.csv','v1219_forbidden_feature_blocklist.csv','v1219_b320_construction_trace.json','v1219_linec_metric_trace.json','v1219_b320_deconfounding_linea_ablation.csv','v1219_b320_deconfounding_linea_summary.csv','v1219_linea_candidate_gate_summary.csv','v1219_P_basis_alignment_diagnostics.csv','v1219_linea_reservoir_decomposition.csv','v1219_linec_protocol_autopsy.csv','v1219_visibility_feature_tier_manifest.csv','v1219_T1_deployable_features.csv','v1219_T2_clone_probe_diagnostic_features.csv','v1219_T3_audit_targets.csv','v1219_visibility_scores_T1_only.csv','v1219_visibility_scores_T1_plus_T2_diagnostic.csv','v1219_visibility_leaveout_T1_only.csv','v1219_visibility_support_audit.csv','v1219_linei_gate_status.csv','v1219_p4_gate_status.csv','v1219_classic_family_status.csv','v1219_hash_manifest.json','v1219_code_review_packet.zip']
for name in checks:
    print(('OK' if (base/name).exists() else 'MISSING'), name)
PY
```

结果：上述 25 个 artifact 均为 `OK`。

复核 zip：

```bash
python - <<'PY'
import zipfile, hashlib
from pathlib import Path
p=Path('results/v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry/official_from_v1218_v1217_v1216_artifacts/v1219_code_review_packet.zip')
print('zip_exists', p.exists())
print('zip_sha256_actual', hashlib.sha256(p.read_bytes()).hexdigest())
with zipfile.ZipFile(p) as z:
    names=set(z.namelist())
    for name in [
        'code/experiments/run_v1219_b320_deconfounding_true_lossagnostic_visibility.py',
        'code/experiments/run_v1218_b320_label_free_ablation.py',
        'code/dgkan/models/fc_purekan_primitives.py',
        'review_artifacts/v1219_feature_provenance_strict_v2.csv',
    ]:
        print(name, name in names)
    print('entries', len(names))
PY
```

结果：

```text
zip_exists True
zip_sha256_actual e551e83cd531af0bf1cf74cae48fd5ec0fed38fb2ce0292e22b3e9cd4aba7e97
code/experiments/run_v1219_b320_deconfounding_true_lossagnostic_visibility.py True
code/experiments/run_v1218_b320_label_free_ablation.py True
code/dgkan/models/fc_purekan_primitives.py True
review_artifacts/v1219_feature_provenance_strict_v2.csv True
entries 59
```

结论：按 v12.19 计划的定义，本轮达到 `Success B`，即 B320 claim scope finalized。没有达到 `Success A/C/D`，所以没有扩到 10-seed、没有打开 Line I/P4，也没有伪造 functional promotion。

---

## 11. 2026-05-24 23:47 +08 二次硬核验

用户再次追问是否完成后，进一步从结果 CSV/JSON 中核验 gate 逻辑是否真的闭合，尤其检查：Line C 是否含 C0-C5；Line A 是否所有 strict label-free 候选都未过；Line T/I/P4 是否因 gate 关闭。

核验命令：

```bash
python - <<'PY'
import csv, json
from pathlib import Path
base=Path('results/v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry/official_from_v1218_v1217_v1216_artifacts')
route=json.loads((base/'v1219_route_decision.json').read_text())
print('ROUTE')
for k in ['route','fail_reason','line_r_code_semantics_pass','linea_pass','linea_official_label_free_pass_count','linec_protocol_impl_consistency_pass','T1_only_visibility_pass','line_i_open','p4_open']:
    print(k, route.get(k))
print('\nLINEA rows/summaries')
for name in ['v1219_b320_deconfounding_linea_ablation.csv','v1219_b320_deconfounding_linea_summary.csv','v1219_linea_candidate_gate_summary.csv']:
    with (base/name).open(newline='') as f:
        rows=list(csv.DictReader(f))
    print(name, len(rows), rows[0].keys() if rows else [])
print('\nLineC impl ids')
with (base/'v1219_linec_protocol_autopsy.csv').open(newline='') as f:
    rows=list(csv.DictReader(f))
print('rows', len(rows))
print(sorted({r.get('linec_impl_id','') for r in rows}))
print('\nT feature score rows')
for name in ['v1219_visibility_scores_T1_only.csv','v1219_visibility_scores_T1_plus_T2_diagnostic.csv','v1219_visibility_support_audit.csv']:
    with (base/name).open(newline='') as f:
        rows=list(csv.DictReader(f))
    print(name, len(rows), rows[0] if rows else {})
PY
```

关键结果：

```text
route = R2-B320LabelInitDependenceDetected
line_r_code_semantics_pass = 1
linea_pass = 0
linea_official_label_free_pass_count = 0
linec_protocol_impl_consistency_pass = 1
T1_only_visibility_pass = 0
line_i_open = 0
p4_open = 0

v1219_b320_deconfounding_linea_ablation.csv rows = 135
v1219_b320_deconfounding_linea_summary.csv rows = 15
v1219_linea_candidate_gate_summary.csv rows = 15

LineC autopsy rows = 7
LineC impl ids =
  C0-locked-source-shared-v1252
  C1-recovered-b64-sketch32-d24
  C2-v1217-repair-source-shared-v1252
  C3-no-update-baseline
  C4-AdamW-window
  C5-fixed-batch-deterministic-repeat
  summary row

T1 visibility rows = 1728
T1 feature_columns = 31
T1 AUC_joint_min = 0.0
T1 precision_at_k_joint_min = 0.0
T1 recall_at_k_joint_min = 0.0

T1+T2 diagnostic rows = 1728
T1+T2 feature_columns = 41
T1+T2 AUC_joint_min = 0.03773584905660377
T1+T2 precision_at_k_joint_min = 0.0
T1+T2 recall_at_k_joint_min = 0.0

support_concentrated = 0
control_false_positive_rate = 0.0
```

Line A gate 摘要核验命令：

```bash
python - <<'PY'
import csv
from pathlib import Path
base=Path('results/v12_19_b320_deconfounding_true_lossagnostic_visibility_functional_reentry/official_from_v1218_v1217_v1216_artifacts')
with (base/'v1219_linea_candidate_gate_summary.csv').open(newline='') as f:
    rows=list(csv.DictReader(f))
print('candidate_id, strict_label_free_rows, task_pass, linec_pass, official_pass, mean_delta_vs_A0, worst_delta_vs_A0')
for r in rows:
    print(r.get('candidate_id'), r.get('strict_label_free_rows'), r.get('task_side_label_free_pass'), r.get('linec_pass'), r.get('official_label_free_b320_pass'), r.get('mean_delta_vs_A0'), r.get('worst_delta_vs_A0'))
PY
```

关键结果：

```text
A1-noYForStats strict_label_free_rows=9 task_pass=0 linec_pass=0 official_pass=0 mean_delta_vs_A0=-0.011284722222222222 worst_delta_vs_A0=-0.037109375
A5-orthogonalP-labelFree strict_label_free_rows=9 task_pass=0 linec_pass=0 official_pass=0 mean_delta_vs_A0=-0.011935763888888888 worst_delta_vs_A0=-0.0390625
A11-orthogonalP-directRead125-labelFree strict_label_free_rows=9 task_pass=0 linec_pass=0 official_pass=0 mean_delta_vs_A0=-0.011935763888888888 worst_delta_vs_A0=-0.0390625
A12-orthogonalP-identityAmp150-labelFree strict_label_free_rows=9 task_pass=0 linec_pass=0 official_pass=0 mean_delta_vs_A0=-0.012152777777777778 worst_delta_vs_A0=-0.0390625
A15-PCAOrthoMix-labelFree strict_label_free_rows=9 task_pass=0 linec_pass=0 official_pass=0 mean_delta_vs_A0=-0.028428819444444444 worst_delta_vs_A0=-0.083984375
A16-AugStableP-labelFree strict_label_free_rows=9 task_pass=0 linec_pass=0 official_pass=0 mean_delta_vs_A0=-0.037109375 worst_delta_vs_A0=-0.060546875
A17-RandomCotangentStableP-labelFree strict_label_free_rows=9 task_pass=0 linec_pass=0 official_pass=0 mean_delta_vs_A0=-0.016493055555555556 worst_delta_vs_A0=-0.052734375
A18-InputCovWhitenedOrthoP-labelFree strict_label_free_rows=9 task_pass=0 linec_pass=0 official_pass=0 mean_delta_vs_A0=-0.091796875 worst_delta_vs_A0=-0.13671875
A19-BlockLocalAugStableP-labelFree strict_label_free_rows=9 task_pass=0 linec_pass=0 official_pass=0 mean_delta_vs_A0=-0.021267361111111112 worst_delta_vs_A0=-0.04296875
```

zip 二次核验：

```text
sha256 = e551e83cd531af0bf1cf74cae48fd5ec0fed38fb2ce0292e22b3e9cd4aba7e97
entries = 59
contains code/experiments/run_v1219_b320_deconfounding_true_lossagnostic_visibility.py = True
contains code/experiments/run_v1218_b320_label_free_ablation.py = True
contains code/dgkan/models/fc_purekan_primitives.py = True
contains review_artifacts/v1219_route_decision.json = True
contains review_artifacts/v1219_linec_protocol_autopsy.csv = True
contains review_artifacts/v1219_visibility_scores_T1_only.csv = True
```

结论：本次二次硬核验仍支持 `Success B` 完成；不存在满足继续 10-seed、Line I 或 P4 的前置条件。
