# DG-KAN v12.18 B320CodeAudit LossAgnosticFunctional 执行日志

生成时间：2026-05-24 Asia/Singapore

对应计划：

```text
docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional计划.md
```

本日志只记录真实执行过的命令、文件、输出与修复动作；不把未运行的训练、未存在的 ablation 或 gated-not-run 阶段写成成功。

## 1. 环境与约束

工作目录：

```text
/home/chengshun.wang/DG-LCA
```

代码环境：

```text
conda env = kan
```

GPU 可用性检查：

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader
```

输出：

```text
NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 97250 MiB, 0 %
NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 97250 MiB, 0 %
NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 97250 MiB, 0 %
NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 97250 MiB, 0 %
```

## 2. 计划阅读与代码定位

阅读计划文档：

```bash
sed -n '1,260p' docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional计划.md
sed -n '260,620p' docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional计划.md
sed -n '620,1040p' docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional计划.md
sed -n '1040,1280p' docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional计划.md
```

提取 route / label-init / strict Line T 等关键要求：

```bash
rg -n "R2|label-init|label init|ablation|PCA|lowfreq|orthogonal|必须|路线|route" docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional计划.md
```

确认 v12.18 必须包含 transitive code review packet，并且 v124/v120 不能再漏：

```bash
rg --files | rg 'run_v124_multibasis_functional_dual.py|run_v120_good_geometry_battery.py|run_v1217|run_v1216|run_v1215|run_v1283|run_v1252'
```

审查 label-informed trainprobe 初始化位置：

```bash
sed -n '4040,4135p' dgkan/models/fc_purekan_primitives.py
```

核心证据：

```text
SimpleFastTaskGeometryKAN.__init__
if "trainprobe" in variant_lower and y_for_stats is not None:
  使用 y_for_stats 计算 class mean direction
trainprobe_signal_init_uses_labels = 1 if "trainprobe" in variant_lower else 0
trainprobe_signal_init_applied = 1 if probe_dirs is not None else 0
```

审查 v1283/v124 构造链：

```bash
sed -n '270,380p' experiments/run_v1283_b109_classic_family_functional_geometry.py
sed -n '3780,3868p' experiments/run_v1283_b109_classic_family_functional_geometry.py
```

## 3. 新增 v12.18 runner

新增文件：

```text
experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

主要实现内容：

```text
1. 生成 v1218_transitive_dependency_manifest.csv。
2. 生成 CR0-CR15 code review manifest、symbol line map、manual review packet、blocker table。
3. 审计 B320 current label-informed init，并记录 A1-A4 label-free ablation 当前无 comparable artifact。
4. 拆分 audit-only Line C 与 strict deployable observables。
5. 初版先把 v12.17.2 feature provenance 全部判为 forbidden，不打开 P3/P4。
6. 后续修复版新增从 v12.16 repair source 抽取 strict loss-agnostic observables 并实际跑 Line T visibility v2。
7. 生成完整 code review zip，包含 v1218/v1217/v1216/v1215/v1283/v1252/v124/v120 与 dgkan 核心文件。
```

## 4. 初版编译与运行

编译：

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

结果：exit 0。

初版 official run：

```bash
rm -rf results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts && conda run -n kan python experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py --run-id official_from_v1217_v1216_artifacts --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts
```

输出：

```json
{
  "out_dir": "/home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts",
  "route": "R2-B320LabelInitDependenceDetected",
  "p4_open": 0,
  "promotion_allowed": 0
}
```

初版校验发现：

```text
required_count = 22
missing = []
strict_allowed_feature_count = 0
strict_visibility_feature_rows = 0
strict_visibility_blocker = no_strict_v1217_feature_survived_provenance_audit
zip_entries = 44
```

这个结果没有造假，但 Line T 只做了 existing feature audit，没有按计划继续尝试 strict observables repair，因此继续修改。

## 5. Line T 修复推进

检查 v12.16 / v12.17.2 artifact：

```bash
find results/v12_16_b320locked_explicit_signal_reservoir_functional -maxdepth 2 -type f | sort | sed -n '1,120p'
find results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry -maxdepth 2 -type f | sort | sed -n '1,160p'
head -n 3 results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_sketchdim24_rank5_b64_w3_5_10/v1216_linec_v2_sketch_targets.csv
head -n 3 results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts/v1217_release_labels_audit_only.csv
```

确认 v12.16 repair source 有严格无标签/无 CE 行：

```bash
conda run -n kan python -c "import csv; from collections import Counter; p='results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_sketchdim24_rank5_b64_w3_5_10/v1216_linec_v2_sketch_targets.csv'; rows=list(csv.DictReader(open(p))); print('rows',len(rows)); [print(col, Counter(r[col] for r in rows).most_common(20)) for col in ['method','sketch_family','cotangent_family','augmentation_used','loss_agnostic_direction','ce_vector_used_for_direction','label_used_for_direction','dataset_name_used_for_commit']]"
```

输出：

```text
rows 864
method [('U0-NoOp', 108), ('U1-RandomMatchedNorm', 108), ('U4-RandomCotangentVJPEnsemble', 108), ('U5-OrthogonalCotangentVJPEnsemble', 108), ('U6-RoleConditionedPrimitiveActuator', 108), ('U7-ProjectionQuadRoleActuator', 108), ('U8-BranchReadoutRoleActuator', 108), ('U9-MixedLowRankActuatorBasis', 108)]
sketch_family [('multi_window_gradient', 216), ('role_conditioned', 216), ('stable_unstable_output', 216), ('persistent_projector', 216)]
cotangent_family [('loss_agnostic_probe', 648), ('control_noop', 108), ('control_random', 108)]
augmentation_used [('0', 648), ('1', 216)]
loss_agnostic_direction [('1', 864)]
ce_vector_used_for_direction [('0', 864)]
label_used_for_direction [('0', 864)]
dataset_name_used_for_commit [('0', 864)]
```

确认 hard joint support 分布：

```bash
conda run -n kan python -c "import csv; from collections import Counter; p='results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts/v1217_release_labels_audit_only.csv'; rows=list(csv.DictReader(open(p))); print('rows',len(rows)); [print(col, Counter(r[col] for r in rows if r['hard_joint_release']=='1').most_common()) for col in ['source_run','method','sketch_family','dataset','seed','window']]"
```

输出：

```text
rows 864
source_run [('repair_sketchdim24_rank5_b64_w3_5_10', 8)]
method [('U7-ProjectionQuadRoleActuator', 4), ('U9-MixedLowRankActuatorBasis', 4)]
sketch_family [('multi_window_gradient', 2), ('role_conditioned', 2), ('stable_unstable_output', 2), ('persistent_projector', 2)]
dataset [('Fashion-MNIST', 4), ('KMNIST', 4)]
seed [('0', 8)]
window [('10', 8)]
```

修复内容：

```text
1. 新增 DEFAULT_V1216_REPAIR_DIR。
2. 新增 v1218_strict_visibility_repair_attempts.csv。
3. 从 v1216 repair source 抽取 13 个 strict feature：
   signal_rank, reservoir_rank, signal_effective_rank,
   reservoir_fraction, top_eigen_share,
   dissipation_condition, signal_projector_stability, reservoir_projector_stability,
   stable_cotangent_energy, unstable_cotangent_energy,
   augmentation_used, sketch_dim, batch_size。
4. 特征只来自 loss_agnostic_direction=1、ce_vector_used_for_direction=0、label_used_for_direction=0、dataset_name_used_for_commit=0 的行。
5. 使用 v12.17.2 hard release label 只作为 audit target。
6. 实现 leave-dataset/seed/window/method/source-run/actuator-family out。
```

第一次修复 run 后发现 join key 问题：

```text
v12.17.2 release label row_id 末尾是原始行号；
v12.16 source split 字段恒为 0；
按完整 row_id join 会导致 strict features 匹配不到 8 个真实 hard joint labels。
```

修复：

```text
改为 semantic_release_key = source_run, dataset, seed, window, method, sketch_id。
```

## 6. 最终编译与 official run

最终编译与重跑：

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py && rm -rf results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts && conda run -n kan python experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py --run-id official_from_v1217_v1216_artifacts --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts
```

输出：

```json
{
  "out_dir": "/home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts",
  "route": "R2-B320LabelInitDependenceDetected",
  "p4_open": 0,
  "promotion_allowed": 0
}
```

最终输出目录：

```text
/home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts
```

## 7. 最终校验命令

Route 和核心字段读取：

```bash
conda run -n kan python -c "import json; from pathlib import Path; out=Path('results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts'); route=json.loads((out/'v1218_route_decision.json').read_text()); keys=['generated_at','route','fail_reason','transitive_dependency_rows','transitive_missing_count','core_code_review_rows','cr0_cr15_unknown_count','manual_review_pending_count','b320_current_anchor_pass','b320_current_uses_label_init','b320_label_free_ablation_available_count','linec_audit_rows','linec_joint_hard_support_rows','linec_random_joint_false_positive_rows','linec_calibration_pass','feature_provenance_rows','strict_allowed_feature_count','forbidden_feature_count','strict_visibility_feature_rows','strict_visibility_wide_rows','strict_visibility_feature_columns','strict_auc_joint_min','strict_precision_joint_min','strict_recall_joint_min','strict_visibility_pass','strict_visibility_blocker','strict_visibility_required_split_failures','support_concentrated','hard_joint_support_rows','p3_open','p4_open','line_d_rows','line_d_new_hypothesis_implemented','code_review_packet_zip_sha256','hash_manifest_entries']; [print(f'{k}={route.get(k)}') for k in keys]"
```

输出：

```text
generated_at=2026-05-24T06:31:49Z
route=R2-B320LabelInitDependenceDetected
fail_reason=current_B320_uses_label_informed_trainprobe_init;label_free_ablation_missing;strict_loss_agnostic_observables_not_visible
transitive_dependency_rows=23
transitive_missing_count=0
core_code_review_rows=16
cr0_cr15_unknown_count=0
manual_review_pending_count=16
b320_current_anchor_pass=1
b320_current_uses_label_init=1
b320_label_free_ablation_available_count=0
linec_audit_rows=864
linec_joint_hard_support_rows=24
linec_random_joint_false_positive_rows=0
linec_calibration_pass=1
feature_provenance_rows=41
strict_allowed_feature_count=13
forbidden_feature_count=28
strict_visibility_feature_rows=11232
strict_visibility_wide_rows=864
strict_visibility_feature_columns=13
strict_auc_joint_min=0.5
strict_precision_joint_min=0.0
strict_recall_joint_min=0.0
strict_visibility_pass=0
strict_visibility_blocker=strict_auc_joint_below_gate;strict_precision_recall_below_gate;support_concentrated;required_leaveout_not_runnable
strict_visibility_required_split_failures=1
support_concentrated=1
hard_joint_support_rows=8
p3_open=0
p4_open=0
line_d_rows=12
line_d_new_hypothesis_implemented=0
code_review_packet_zip_sha256=82cb3284c1b45575ef03dd4d9d40efce0bd974c3c539fb7609241809387aa39d
hash_manifest_entries=22
```

产物完整性与 zip 检查：

```bash
conda run -n kan python -c "import json,zipfile; from pathlib import Path; out=Path('results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts'); required=['v1218_route_decision.json','v1218_transitive_dependency_manifest.csv','v1218_core_code_review_manifest.csv','v1218_symbol_line_map.json','v1218_feature_provenance_audit.csv','v1218_b320_label_init_audit.csv','v1218_manual_review_packet.md','v1218_review_blocker_table.csv','v1218_linec_audit_metrics.csv','v1218_linec_null_distribution.csv','v1218_linec_threshold_sensitivity.csv','v1218_linec_hard_support.csv','v1218_audit_metric_provenance.csv','v1218_strict_visibility_features.csv','v1218_strict_visibility_scores.csv','v1218_strict_visibility_leaveout.csv','v1218_strict_visibility_repair_attempts.csv','v1218_support_concentration.csv','v1218_p3_matched_controls_status.csv','v1218_p4_gate_status.csv','v1218_classic_family_status.csv','v1218_hash_manifest.json','v1218_code_review_packet.zip']; print('required_count=',len(required)); print('missing=',[x for x in required if not (out/x).exists()]); z=zipfile.ZipFile(out/'v1218_code_review_packet.zip'); names=set(z.namelist()); print('zip_entries=',len(names)); [print('zip_has',x,x in names) for x in ['code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py','code/experiments/run_v124_multibasis_functional_dual.py','code/experiments/run_v120_good_geometry_battery.py','code/dgkan/models/fc_purekan_primitives.py','review_artifacts/v1218_strict_visibility_repair_attempts.csv','review_artifacts/v1218_strict_visibility_scores.csv','review_artifacts/v1218_core_code_review_manifest.csv']]"
```

输出：

```text
required_count= 23
missing= []
zip_entries= 45
zip_has code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py True
zip_has code/experiments/run_v124_multibasis_functional_dual.py True
zip_has code/experiments/run_v120_good_geometry_battery.py True
zip_has code/dgkan/models/fc_purekan_primitives.py True
zip_has review_artifacts/v1218_strict_visibility_repair_attempts.csv True
zip_has review_artifacts/v1218_strict_visibility_scores.csv True
zip_has review_artifacts/v1218_core_code_review_manifest.csv True
```

CR unknown 与 manual pending 检查：

```bash
conda run -n kan python -c "import csv; from pathlib import Path; out=Path('results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts'); rows=list(csv.DictReader(open(out/'v1218_core_code_review_manifest.csv'))); print('unknown_reviews=', [r['review_id'] for r in rows if r['unknown_or_not_inspected']!='0']); blockers=list(csv.DictReader(open(out/'v1218_review_blocker_table.csv'))); print('manual_pending=', sum(1 for r in blockers if r['blocker_type']=='manual_review_pending_for_promotion'))"
```

输出：

```text
unknown_reviews= []
manual_pending= 16
```

空白检查：

```bash
git diff --check -- experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

输出：无输出，exit 0。

## 8. 继续推进：Line A label-free B320 smoke ablation

在用户指出“没有完成不应停下”后，继续推进 Line A。该步骤不是 official anchor 同预算训练，而是一个真实 CUDA smoke ablation，用来检验推荐方向是否值得继续投入，并避免继续只写“缺失”。

新增 runner：

```text
experiments/run_v1218_b320_label_free_ablation.py
```

同步修改 code packet manifest：

```text
experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

修改内容：

```text
1. BASE_REQUIRED_CODE_FILES 增加 experiments/run_v1218_b320_label_free_ablation.py。
2. REQUIRED_ARTIFACTS 增加：
   v1218_b320_label_free_smoke_ablation.csv
   v1218_b320_label_free_smoke_summary.csv
   v1218_b320_label_free_smoke_route.json
```

第一次运行 blocker：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id label_free_smoke_seed0_3x3 --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts --device auto --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --train-size 512 --val-size 256 --test-size 256 --epochs 3 --batch-size 128
```

失败：

```text
ValueError: Expected a torch.device with a specified index or an integer, but got:cuda
```

修复：

```text
device_from_arg("auto") 从 torch.device("cuda") 改为 torch.device("cuda:0")。
```

第二次运行 blocker：

```text
AttributeError: 'Namespace' object has no attribute 'seed'. Did you mean: 'seeds'?
```

修复：

```text
给 v120._load_vision_split 传入的 Namespace 注入 seed。
train_one 按当前 seed 注入 load_args.seed。
run_main 的 probe load 使用 seeds[0] 注入 probe_args.seed。
```

最终编译与运行：

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_label_free_ablation.py && conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id label_free_smoke_seed0_3x3 --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts --device auto --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0 --train-size 512 --val-size 256 --test-size 256 --epochs 3 --batch-size 128
```

输出：

```json
{
  "stage": "V1218_B320_LABEL_FREE_SMOKE_ROUTE",
  "generated_at": "2026-05-24T06:52:52Z",
  "run_id": "label_free_smoke_seed0_3x3",
  "datasets": ["MNIST", "Fashion-MNIST", "KMNIST"],
  "seeds": [0],
  "train_size": 512,
  "val_size": 256,
  "test_size": 256,
  "epochs": 3,
  "rows": 21,
  "summary_rows": 7,
  "label_free_smoke_ablation_available_count": 5,
  "best_label_free_smoke_candidate": "A1-noYForStats",
  "best_label_free_mean_delta_vs_A0": -0.015625,
  "smoke_not_official": 1,
  "official_training_result_available": 0,
  "route_impact": "does_not_close_R2; full comparable label-free B320 budget still missing",
  "no_fake": 1,
  "no_proxy": 1,
  "cpu_offload_used": 0
}
```

随后将同一 continuation 从 seed0 扩展到 seeds 0,1,2：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py --run-id label_free_smoke_seed012_3x3 --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts --device auto --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 512 --val-size 256 --test-size 256 --epochs 3 --batch-size 128
```

输出：

```json
{
  "stage": "V1218_B320_LABEL_FREE_SMOKE_ROUTE",
  "generated_at": "2026-05-24T06:58:12Z",
  "run_id": "label_free_smoke_seed012_3x3",
  "datasets": ["MNIST", "Fashion-MNIST", "KMNIST"],
  "seeds": [0, 1, 2],
  "train_size": 512,
  "val_size": 256,
  "test_size": 256,
  "epochs": 3,
  "rows": 63,
  "summary_rows": 7,
  "label_free_smoke_ablation_available_count": 5,
  "best_label_free_smoke_candidate": "A1-noYForStats",
  "best_label_free_mean_delta_vs_A0": -0.032552083333333336,
  "smoke_not_official": 1,
  "official_training_result_available": 0,
  "route_impact": "does_not_close_R2; full comparable label-free B320 budget still missing",
  "no_fake": 1,
  "no_proxy": 1,
  "cpu_offload_used": 0
}
```

最终 3×3 smoke summary：

```text
A0-labelInit:
  rows = 9
  mean_delta_vs_mlp = 0.022569444444444444
  worst_delta_vs_mlp = -0.0078125
  max_AUC_step_ratio_vs_mlp = 1.0071162337536481

A1-noYForStats:
  rows = 9
  mean_delta_vs_mlp = -0.009982638888888888
  worst_delta_vs_mlp = -0.0625
  mean_delta_vs_A0_labelInit = -0.032552083333333336
  worst_delta_vs_A0_labelInit = -0.0625

A2-randomP-labelFree:
  rows = 9
  mean_delta_vs_mlp = -0.009982638888888888
  worst_delta_vs_mlp = -0.0625
  mean_delta_vs_A0_labelInit = -0.032552083333333336
  worst_delta_vs_A0_labelInit = -0.0625

A3-PCA-P-labelFree:
  rows = 9
  mean_delta_vs_mlp = -0.07942708333333333
  worst_delta_vs_mlp = -0.1484375
  mean_delta_vs_A0_labelInit = -0.10199652777777778

A4-lowfreqP-labelFree:
  rows = 9
  mean_delta_vs_mlp = -0.63671875
  worst_delta_vs_mlp = -0.74609375
  mean_delta_vs_A0_labelInit = -0.6592881944444444

A5-orthogonalP-labelFree:
  rows = 9
  mean_delta_vs_mlp = -0.024739583333333332
  worst_delta_vs_mlp = -0.04296875
  mean_delta_vs_A0_labelInit = -0.047309027777777776
```

重新打包 zip/hash：

```bash
conda run -n kan python -c "import json; from pathlib import Path; from experiments import run_v1218_b320_codeaudit_lossagnostic_functional as v18; out=Path('results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts').resolve(); zip_path=v18.package_code_review_zip(out); hashes=v18.write_hash_manifest(out); route=json.loads((out/'v1218_route_decision.json').read_text()); route['code_review_packet_zip']=v18.rel(zip_path); route['code_review_packet_zip_sha256']=v18.sha256_file(zip_path); route['hash_manifest_entries']=len(hashes); route['post_label_free_smoke_repackaged']=1; v18.write_json(out/'v1218_route_decision.json', route); v18.write_hash_manifest(out); print('zip', zip_path); print('zip_sha256', route['code_review_packet_zip_sha256']); print('hash_manifest_entries', route['hash_manifest_entries'])"
```

输出：

```text
zip /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_sha256 6e53e0c54df91f23469e2b2bc0ee4ab86d7fd8ef4f56a11d85284d3c878aa306
hash_manifest_entries 25
```

3×3 smoke 覆盖后再次重打包：

```text
zip_sha256 0f412595679ef52dc97d24838debd98fb74c45bdead8b99a0401d75cfaa037b8
hash_manifest_entries 25
```

最终 zip 校验：

```bash
conda run -n kan python -c "import json,zipfile; from pathlib import Path; out=Path('results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts'); route=json.loads((out/'v1218_route_decision.json').read_text()); keys=['route','fail_reason','b320_label_free_smoke_ablation_available_count','b320_label_free_smoke_best_candidate','b320_label_free_smoke_best_mean_delta_vs_A0','b320_label_free_smoke_not_official','b320_label_free_official_ablation_still_missing','code_review_packet_zip_sha256','hash_manifest_entries']; [print(f'{k}={route.get(k)}') for k in keys]; required=['v1218_b320_label_free_smoke_ablation.csv','v1218_b320_label_free_smoke_summary.csv','v1218_b320_label_free_smoke_route.json','v1218_code_review_packet.zip']; print('missing_smoke=', [x for x in required if not (out/x).exists()]); z=zipfile.ZipFile(out/'v1218_code_review_packet.zip'); names=set(z.namelist()); [print('zip_has', x, x in names) for x in ['code/experiments/run_v1218_b320_label_free_ablation.py','review_artifacts/v1218_b320_label_free_smoke_ablation.csv','review_artifacts/v1218_b320_label_free_smoke_summary.csv','review_artifacts/v1218_b320_label_free_smoke_route.json']]"
```

输出：

```text
route=R2-B320LabelInitDependenceDetected
fail_reason=current_B320_uses_label_informed_trainprobe_init;label_free_ablation_missing;strict_loss_agnostic_observables_not_visible
b320_label_free_smoke_ablation_available_count=5
b320_label_free_smoke_best_candidate=A1-noYForStats
b320_label_free_smoke_best_mean_delta_vs_A0=-0.032552083333333336
b320_label_free_smoke_not_official=1
b320_label_free_official_ablation_still_missing=1
code_review_packet_zip_sha256=0f412595679ef52dc97d24838debd98fb74c45bdead8b99a0401d75cfaa037b8
hash_manifest_entries=25
missing_smoke= []
zip_has code/experiments/run_v1218_b320_label_free_ablation.py True
zip_has review_artifacts/v1218_b320_label_free_smoke_ablation.csv True
zip_has review_artifacts/v1218_b320_label_free_smoke_summary.csv True
zip_has review_artifacts/v1218_b320_label_free_smoke_route.json True
```

空白检查：

```bash
git diff --check -- experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional_执行日志.md docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional_实验结果复盘.md
```

输出：无输出，exit 0。

## 9. 最终文件位置

Runner：

```text
/home/chengshun.wang/DG-LCA/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
/home/chengshun.wang/DG-LCA/experiments/run_v1218_b320_label_free_ablation.py
```

结果目录：

```text
/home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts
```

代码审查 zip：

```text
/home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
```

Line A smoke ablation artifacts：

```text
/home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_b320_label_free_smoke_ablation.csv
/home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_b320_label_free_smoke_summary.csv
/home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_b320_label_free_smoke_route.json
```

zip sha256：

```text
63802d5700a4269488aaca2071eb41061f88b595635579b1782220702e18cdb1
```

本执行日志：

```text
/home/chengshun.wang/DG-LCA/docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional_执行日志.md
```

实验结果复盘：

```text
/home/chengshun.wang/DG-LCA/docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional_实验结果复盘.md
```

## 10. 继续推进：把 smoke ablation 与 Line T support repair 正式并入 v12.18 产物链

用户指出不能停在未完成状态，因此继续检查当前 v12.18 是否只是口头记录 smoke 结果，而没有进入正式 audit/route。

确认旧状态的关键命令：

```bash
head -n 12 results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_b320_label_init_audit.csv
head -n 12 results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_b320_label_free_smoke_summary.csv
python -c "import json; from pathlib import Path; d=json.loads(Path('results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_route_decision.json').read_text()); keys=['route','fail_reason','b320_label_free_ablation_available_count','b320_label_free_smoke_ablation_available_count','b320_label_free_smoke_best_candidate','b320_label_free_smoke_best_mean_delta_vs_A0','strict_visibility_blocker','support_concentrated','code_review_packet_zip_sha256']; [print(f'{k}={d.get(k)}') for k in keys]"
find results -path '*v12_16*' -name 'v1216_linec_v2_sketch_targets.csv' -print
```

旧状态输出摘要：

```text
v1218_b320_label_init_audit.csv 仍把 A1-A4 写成 no comparable artifact exists。
v1218_b320_label_free_smoke_summary.csv 已存在 A0/A1/A2/A3/A4/A5/MLP rows。
route=R2-B320LabelInitDependenceDetected
fail_reason=current_B320_uses_label_informed_trainprobe_init;label_free_ablation_missing;strict_loss_agnostic_observables_not_visible
b320_label_free_smoke_ablation_available_count=5
b320_label_free_smoke_best_candidate=A1-noYForStats
b320_label_free_smoke_best_mean_delta_vs_A0=-0.032552083333333336
strict_visibility_blocker=strict_auc_joint_below_gate;strict_precision_recall_below_gate;support_concentrated;required_leaveout_not_runnable
support_concentrated=1
code_review_packet_zip_sha256=0f412595679ef52dc97d24838debd98fb74c45bdead8b99a0401d75cfaa037b8

可用 v12.16 Line C source files:
results/v12_16_b320locked_explicit_signal_reservoir_functional/official_3x3_b32_w3_5_10/v1216_linec_v2_sketch_targets.csv
results/v12_16_b320locked_explicit_signal_reservoir_functional/smoke_mnist_seed0/v1216_linec_v2_sketch_targets.csv
results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_sketchdim24_rank5_b64_w3_5_10/v1216_linec_v2_sketch_targets.csv
```

发现问题：

```text
1. smoke ablation 已真实跑完，但正式 v1218_b320_label_init_audit.csv 仍显示 A1-A4 未运行，审计链不自洽。
2. route 里已有 smoke summary 字段，但 fail_reason 仍写 label_free_ablation_missing，没有区分 official-budget 缺失和 small-budget smoke 已执行。
3. Line T support concentration 失败后，没有单独产物说明已有 source 为什么不能安全扩展 visibility promotion。
```

本轮修改文件：

```text
experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

修改内容：

```text
1. 新增 attach_smoke_metrics，把 v1218_b320_label_free_smoke_summary.csv 中的真实 A0/A1/A2/A3/A4/A5 smoke metrics 写入 v1218_b320_label_init_audit.csv。
2. A1-A5 仍保留 official_training_result_available=0，新增 small_budget_training_result_available=1 与 smoke_not_official=1，避免把 smoke 伪装成 official。
3. route fail_reason 改为区分 label_free_official_ablation_missing 与 label_free_smoke_underperforms_labelInit。
4. 新增 v1218_line_t_support_expansion_audit.csv，逐行审计 official/repair/smoke 三个 v12.16 source 是否可用于 Line T support/source 扩展。
5. REQUIRED_ARTIFACTS 增加 v1218_line_t_support_expansion_audit.csv，使该文件进入 code review zip。
```

第一次运行主 runner 前语法检查：

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py
```

输出：无输出，exit 0。

第一次重跑主 runner：

```bash
conda run -n kan python experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py --run-id official_from_v1217_v1216_artifacts --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts --v1216-repair-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_sketchdim24_rank5_b64_w3_5_10
```

blocker 输出：

```text
ValueError: too many values to unpack (expected 2)
```

blocker 原因与修复：

```text
原因：新增 tuple 字段 smoke_id 后，循环仍写成 for ablation_id, allowed in [...]。
修复：改为 for ablation_id, smoke_id, allowed in [...]。
该 blocker 是本轮代码修改错误，不是实验结果。
```

修复后重跑：

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py && conda run -n kan python experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py --run-id official_from_v1217_v1216_artifacts --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts --v1216-repair-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_sketchdim24_rank5_b64_w3_5_10
```

输出：

```json
{
  "out_dir": "/home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts",
  "route": "R2-B320LabelInitDependenceDetected",
  "p4_open": 0,
  "promotion_allowed": 0
}
```

重跑后关键 route 检查：

```bash
python -c "import json; from pathlib import Path; d=json.loads(Path('results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_route_decision.json').read_text()); keys=['route','fail_reason','b320_label_free_ablation_available_count','b320_label_free_smoke_ablation_available_count','b320_label_free_smoke_best_candidate','b320_label_free_smoke_best_mean_delta_vs_A0','b320_label_free_smoke_underperforms_labelinit','b320_label_free_official_ablation_still_missing','strict_visibility_blocker','line_t_support_expansion_audit_rows','line_t_calibrated_release_source_count','line_t_independent_source_runs_available_for_leaveout','line_t_support_expansion_pass','line_t_support_expansion_blocker','p3_open','p4_open','code_review_packet_zip_sha256','hash_manifest_entries']; [print(f'{k}={d.get(k)}') for k in keys]"
```

输出：

```text
route=R2-B320LabelInitDependenceDetected
fail_reason=current_B320_uses_label_informed_trainprobe_init;label_free_official_ablation_missing;label_free_smoke_underperforms_labelInit;strict_loss_agnostic_observables_not_visible
b320_label_free_ablation_available_count=0
b320_label_free_smoke_ablation_available_count=5
b320_label_free_smoke_best_candidate=A1-noYForStats
b320_label_free_smoke_best_mean_delta_vs_A0=-0.032552083333333336
b320_label_free_smoke_underperforms_labelinit=1
b320_label_free_official_ablation_still_missing=1
strict_visibility_blocker=strict_auc_joint_below_gate;strict_precision_recall_below_gate;support_concentrated;required_leaveout_not_runnable
line_t_support_expansion_audit_rows=3
line_t_calibrated_release_source_count=1
line_t_independent_source_runs_available_for_leaveout=1
line_t_support_expansion_pass=0
line_t_support_expansion_blocker=no_second_calibrated_release_source;hard_joint_support_concentrated_in_seed0_window10;official_source_random_false_positives;smoke_source_not_official_independent_source
p3_open=0
p4_open=0
code_review_packet_zip_sha256=63802d5700a4269488aaca2071eb41061f88b595635579b1782220702e18cdb1
hash_manifest_entries=26
```

Line T support expansion audit 实际输出：

```bash
cat results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_line_t_support_expansion_audit.csv
```

关键行摘要：

```text
official_3x3_b32_w3_5_10:
  linec_rows=864
  strict_loss_agnostic_rows=864
  v1217_source_calibration_pass=0
  random_joint_false_positive_rows=4
  control_joint_false_positive_rows=4
  v1217_hard_support_joint_rows=16
  v1217_release_hard_joint_rows=0
  usable_for_promotion_visibility=0
  expansion_status=blocked_by_source_calibration_false_positive;no_v1217_release_hard_joint_rows

repair_sketchdim24_rank5_b64_w3_5_10:
  linec_rows=864
  strict_loss_agnostic_rows=864
  v1217_source_calibration_pass=1
  random_joint_false_positive_rows=0
  control_joint_false_positive_rows=0
  v1217_hard_support_joint_rows=8
  v1217_release_hard_joint_rows=8
  usable_for_promotion_visibility=0
  expansion_status=already_used_but_hard_joint_support_concentrated

smoke_mnist_seed0:
  linec_rows=64
  strict_loss_agnostic_rows=64
  v1217_source_calibration_available=0
  v1217_release_hard_joint_rows=0
  usable_for_promotion_visibility=0
  expansion_status=not_in_v1217_source_calibration;no_v1217_release_hard_joint_rows;smoke_scope_only_not_official_independent_source
```

zip 检查：

```bash
python -c "import zipfile; from pathlib import Path; out=Path('results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts'); z=zipfile.ZipFile(out/'v1218_code_review_packet.zip'); names=set(z.namelist()); print('zip_entries', len(names)); [print('zip_has', x, x in names) for x in ['code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py','code/experiments/run_v1218_b320_label_free_ablation.py','review_artifacts/v1218_b320_label_init_audit.csv','review_artifacts/v1218_b320_label_free_smoke_summary.csv','review_artifacts/v1218_line_t_support_expansion_audit.csv','review_artifacts/v1218_route_decision.json']]"
```

输出：

```text
zip_entries 51
zip_has code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py True
zip_has code/experiments/run_v1218_b320_label_free_ablation.py True
zip_has review_artifacts/v1218_b320_label_init_audit.csv True
zip_has review_artifacts/v1218_b320_label_free_smoke_summary.csv True
zip_has review_artifacts/v1218_line_t_support_expansion_audit.csv True
zip_has review_artifacts/v1218_route_decision.json True
```

最终空白检查：

```bash
git diff --check -- experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional_执行日志.md docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional_实验结果复盘.md
```

输出：无输出，exit 0。

## 11. 是否继续升成 official-budget label-free ablation 的检查

继续检查是否可以安全地把当前 smoke ablation 扩展为 official-budget Line A ablation。

查计划中 Line A 要求：

```bash
sed -n '474,630p' docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional计划.md
```

关键要求：

```text
A0 B320-current-labelInit
A1 B320-current-noYForStats
A2 B320-current-randomP-labelFree
A3 B320-current-PCA-P-labelFree
A4 B320-current-lowfreqP-labelFree
A5 MLP same-param AdamW
A6 MLP same-step/FLOP AdamW

所有 candidate 使用同一数据、同一 seed、同一训练预算。
如果 only label-free ablation fail：
  尝试 PCA/lowfreq/orthogonal label-free init；
  不允许再加入 label-derived class mean trick；
  不改变 functional line 的 loss-agnostic 约束。
```

查 v12.17 anchor monitor：

```bash
head -n 5 results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts/v1217_anchor_monitor.csv
python -c "import json; from pathlib import Path; p=Path('results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts/v1217_route_decision.json'); d=json.loads(p.read_text()); [print(k, d[k]) for k in sorted(d) if any(s in k.lower() for s in ['train','epoch','batch','dataset','seed','budget','b320','anchor','p0','size'])]"
```

关键发现：

```text
v12.17 P0 anchor 是 reused_from_v1216=1 / source_artifact=results/v12_14_b320locked_lossagnostic_functional_mechanism/final_aggregate_official/v1214_b320_anchor_monitor.csv。
v12.17 本身没有 new_b320_training_claimed。
route JSON 没有给出可直接复现 official B320 train budget 的完整 train_size/val_size/test_size/epochs/batch_size 字段。
```

查 v12.14/v12.16 文档与 runner：

```bash
rg -n "train_size|epochs|batch_size|B320 monitor|official_3x3" docs/DG-KAN_v12.14_B320Locked_LossAgnosticFunctionalMechanism_独立分析与下一步计划.md docs/DG-KAN_v12.16_B320Locked_ExplicitSignalReservoirFunctional_实验结果分析与下一步计划.md experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py
```

关键发现：

```text
v12.14 plan 写到 B320 monitor train_size=1024、batch_size=128。
v12.14 P3/P4 short-run 写 epochs=3 or 5。
v12.16 plan 写 base=B320 locked、train_size=1024、budget=short-run 3-5 epochs。
v12.16 runner default probe_train_size=512、probe_val_size=256、probe_test_size=256、functional_batch_size=32。
```

结论：

```text
我没有继续把 A1-A5 直接跑成 official-budget label-free ablation。
原因不是放弃，而是 exact official B320 training budget/protocol 在当前 artifacts 中并未完整落盘；现有 v1218 label-free runner 也明确标记 smoke_not_official=1。
如果我任意选择 train_size=1024、epochs=3 或 5 并把它写成 official，就会把协议假设伪装成事实。
因此本轮保持 R2，并把 official-budget label-free ablation 作为未关闭 blocker，而不编造 official result。
```

## 12. 继续推进：recovered anchor-budget Line A + LineC + A6 control

本节是在“不能把 smoke 伪装成 official”的前提下继续推进 Line A。执行目标是：从 v12.11/v12.12/v12.13/v12.14 代码与文档中恢复一个可审计的 anchor-budget 协议，跑 A0/A1/A2/A3/A4/A5 与显式 A6 MLP control，并把 protocol gate 交给主审计决定。

### 12.1 协议恢复检查

检查命令：

```bash
sed -n '1760,1790p' experiments/run_v1213_b320locked_functional_mechanism_nobspline.py
sed -n '190,220p' experiments/run_v1212_b320locked_functional_value_rebuild_classic_nobspline.py
sed -n '205,220p' experiments/run_v1211_b320_functional_mechanism_classic_nobspline.py
rg -n "probe-train-size|train_size|batch_size|epochs" experiments/run_v1214_b320locked_lossagnostic_functional_mechanism.py
head -n 5 results/v12_14_b320locked_lossagnostic_functional_mechanism/final_aggregate_official/v1214_b320_anchor_monitor.csv
sed -n '1,45p' docs/DG-KAN_v12.14_B320Locked_LossAgnosticFunctionalMechanism_执行复盘.md
```

恢复出的可审计 budget：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
train_size = 1024
val_size = 512
test_size = 512
epochs = 3
batch_size = 128
seed_base candidates = 1211000 / 1212000 / 1213000
```

注意：这是 recovered anchor-budget，不是从 v12.17 artifact 中直接完整落盘的 official protocol。因此必须增加 A0 protocol audit；A0 不同时复现 task gate 与 LineC gate 时，不允许写成 official replacement。

### 12.2 修改的代码

文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

修改内容：

```text
1. label_free_ablation runner 新增 artifact_prefix/result_stage/summary_stage/route_stage/result_scope，支持同一 runner 写 smoke、anchor-budget、seedbase probe 等不同产物。
2. label_free_ablation runner 新增 measure_linec，训练结束后对每个 candidate 执行 one AdamW window LineC audit，记录 CouplingR2 / NoiseSignalLeak / RealSignalReservoirRatio，并按 MLP control 判定 LineC_nontearing。
3. label_free_ablation runner 新增 A5-orthogonalP-labelFree，这是计划 6.5 推荐的 orthogonal label-free init 修复方向。
4. label_free_ablation runner 新增显式 A6 MLP-same-step-FLOP-AdamW control。当前 v124/v1218 budget 下它与 hidden=256 MLP control 同构，但仍实跑并落盘，不手填。
5. codeaudit runner 新增 anchor-budget required artifacts、seedbase probe artifacts、v1218_b320_anchor_budget_protocol_audit.csv。
6. codeaudit route gate 现在要求：A0 recovered-anchor-budget 必须同时 task gate pass 与 LineC reproduction pass，才允许把 label-free anchor-budget 视为 official replacement。
```

### 12.3 seed_base probe

执行了 3 个 A0+MLP probe，用于选择能复现 no-regression task gate 的 seed_base。命令形态如下，每次只改 `--seed-base`、`--run-id`、`--artifact-prefix` 与 `--protocol-note`：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id anchor_budget_seedbase1211000_probe \
  --artifact-prefix v1218_b320_anchor_budget_seedbase1211000_probe \
  --result-stage V1218_B320_ANCHOR_BUDGET_SEEDBASE_PROBE \
  --summary-stage V1218_B320_ANCHOR_BUDGET_SEEDBASE_PROBE_SUMMARY \
  --route-stage V1218_B320_ANCHOR_BUDGET_SEEDBASE_PROBE_ROUTE \
  --result-scope recovered_anchor_budget_seedbase1211000_probe \
  --smoke-not-official 0 \
  --official-training-result-available 0 \
  --protocol-note "A0+MLP seed-base probe for recovered anchor budget; train=1024,val=512,test=512,epochs=3,batch=128,seed_base=1211000" \
  --route-impact "seed-base probe only; not official replacement" \
  --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --device auto \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --seed-base 1211000 \
  --train-size 1024 \
  --val-size 512 \
  --test-size 512 \
  --epochs 3 \
  --batch-size 128 \
  --ablation-ids A0-labelInit
```

probe 结果：

```text
seed_base=1211000:
  A0 mean_delta_vs_mlp=0.022569444444444444
  worst_delta_vs_mlp=0.013671875
  near_pass_rate_vs_mlp=1.0
  max_AUC_step_ratio_vs_mlp=0.9637283017402302
  max_AUC_time_ratio_vs_mlp=0.9637283017402302
  max_ECE_delta_vs_mlp=0.005898520350456238

seed_base=1212000:
  A0 mean_delta_vs_mlp=0.021267361111111112
  worst_delta_vs_mlp=0.0
  near_pass_rate_vs_mlp=1.0
  max_AUC_step_ratio_vs_mlp=0.9395254448115371
  max_AUC_time_ratio_vs_mlp=0.7581381409711518
  max_ECE_delta_vs_mlp=0.01871931552886963

seed_base=1213000:
  A0 mean_delta_vs_mlp=0.016927083333333332
  worst_delta_vs_mlp=0.0
  near_pass_rate_vs_mlp=1.0
  max_AUC_step_ratio_vs_mlp=0.9646083424163914
  max_AUC_time_ratio_vs_mlp=0.8558250559627462
  max_ECE_delta_vs_mlp=0.0164031982421875
```

选择：

```text
seed_base = 1211000
理由：A0 task no-regression gate 复现最稳，worst_delta_vs_mlp 为正，ECE delta 最小。
```

### 12.4 完整 recovered anchor-budget + LineC + A6 运行

编译与完整运行命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_label_free_ablation.py experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py && \
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id label_free_anchor_budget_seed012_3x3_seedbase1211000_linec_b64_s32_d24_a6_demoted \
  --artifact-prefix v1218_b320_label_free_anchor_budget \
  --result-stage V1218_B320_LABEL_FREE_ANCHOR_BUDGET_ABLATION \
  --summary-stage V1218_B320_LABEL_FREE_ANCHOR_BUDGET_SUMMARY \
  --route-stage V1218_B320_LABEL_FREE_ANCHOR_BUDGET_ROUTE \
  --result-scope recovered_v1211_v1212_anchor_budget_3x3_seedbase1211000_linec_b64_s32_d24_a6_demoted \
  --smoke-not-official 0 \
  --official-training-result-available 0 \
  --protocol-note "recovered from v1211 anchor lock/v1212 hardening: train_size=1024,val_size=512,test_size=512,epochs=3,batch=128,seed_base=1211000; LineC retry uses b64/sketch_batch32/sketchdim24 following v12.16 repair source naming; A0 task reproduces anchor no-regression but official claim still depends on protocol audit" \
  --route-impact "keeps_R2 unless b64/sketchdim24 LineC protocol reproduces A0 and label-free nontearing" \
  --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --device auto \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --seed-base 1211000 \
  --train-size 1024 \
  --val-size 512 \
  --test-size 512 \
  --epochs 3 \
  --batch-size 128 \
  --measure-linec 1 \
  --linec-batch-size 64 \
  --linec-sketch-batch-size 32 \
  --linec-sketch-dim 24
```

输出：

```text
rows = 72
summary_rows = 8
label_free_ablation_available_count = 5
best_label_free_candidate = A5-orthogonalP-labelFree
best_label_free_mean_delta_vs_A0 = -0.00390625
official_training_result_available = 0
no_fake = 1
no_proxy = 1
cpu_offload_used = 0
```

### 12.5 主审计重跑

命令：

```bash
conda run -n kan python experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py \
  --run-id official_from_v1217_v1216_artifacts \
  --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts \
  --v1216-repair-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_sketchdim24_rank5_b64_w3_5_10
```

输出：

```text
route = R2-B320LabelInitDependenceDetected
p4_open = 0
promotion_allowed = 0
```

route 关键字段：

```text
fail_reason = current_B320_uses_label_informed_trainprobe_init;label_free_anchor_budget_protocol_not_official;label_free_anchor_budget_A0_does_not_reproduce_locked_anchor;label_free_anchor_budget_LineC_nontearing_failed;strict_loss_agnostic_observables_not_visible
b320_label_free_anchor_budget_ablation_available_count = 5
b320_label_free_anchor_budget_task_strong_pass = 1
b320_label_free_anchor_budget_linec_nontearing_pass = 0
b320_label_free_anchor_budget_a0_task_reproduces_locked_anchor = 1
b320_label_free_anchor_budget_a0_linec_reproduces_locked_anchor = 0
b320_label_free_anchor_budget_a0_reproduces_locked_anchor = 0
b320_label_free_anchor_budget_best_candidate = A5-orthogonalP-labelFree
b320_label_free_anchor_budget_best_mean_delta_vs_A0 = -0.00390625
```

protocol audit：

```text
a0_anchor_budget_gate_pass = 1
a0_anchor_budget_task_reproduces_locked_anchor = 1
a0_anchor_budget_linec_available = 1
a0_anchor_budget_linec_reproduces_locked_anchor = 0
a0_anchor_budget_reproduces_locked_anchor = 0
official_budget_claim_allowed = 0
blocker = recovered-anchor-budget A0 does not reproduce locked B320 task+LineC gates; do not treat label-free ablation as official replacement
```

### 12.6 最终 zip 与检查命令

检查命令：

```bash
bsdtar -tf results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip | wc -l
sha256sum results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
jq 'length' results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_hash_manifest.json
```

输出：

```text
zip_entries = 65
zip_sha256 = 861a71f82db20fdb635612333cd1bee046ac3ed9e3dfa18ce29dbf96eb6a0dea
hash_manifest_entries = 40
```

## 17. 最终权威状态汇总

本执行日志历史追加顺序不是完全按章节号递增；为了防止复现时误读，最终状态以本节为准。

最新代码修改已整合进 zip：

```text
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_sha256 = c670793aeb05646f698bf7f5380078211ebca78675a129c866971ce5fd3b5843
zip_entries = 66
hash_manifest_entries = 41
zip_has code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py = True
zip_has review_artifacts/v1218_strict_visibility_feature_engineering_audit.csv = True
```

最终 route：

```text
route = R2-B320LabelInitDependenceDetected
p4_open = 0
promotion_allowed = 0
strict_visibility_feature_columns = 36
strict_auc_joint_min = 0.028169014084507043
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate;support_concentrated;required_leaveout_not_runnable
```

最终 Line T feature-engineering audit：

```text
feature_engineering_audit_rows = 36
raw_count = 18
derived_count = 18
uses_label_sum = 0
uses_ce_sum = 0
selection_sum = 0
```

最终校验：

```text
conda run -n kan python -m py_compile experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py
py_compile_exit = 0
trailing_whitespace_count = 0
```

## 25. EOF 最新权威索引：U10-U14 dictionary expansion 后

本文件历史追加顺序存在中段插入。以本节为最新权威状态；第 24 节记录了本轮 actuator dictionary expansion / batch-bracket repair 的完整命令与结果，第 23 节中的 `b4ccb...` 是上一阶段中间产物。

最新主审计产物：

```text
result_dir = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_sha256 = 6582b59646528335e6f9628f95edb57a6c85a4fc24fd3fed0b663d4d71dc3e36
zip_entries = 68
hash_manifest_entries = 43
```

最新主 route：

```text
run_id = official_from_v1217_v1216_artifacts_support_expand_seed012_seed345_b128_w10_12_15_context_rank_repair_dict_u10_u14_code_audit_refresh
route = R2-B320LabelInitDependenceDetected
fail_reason = current_B320_uses_label_informed_trainprobe_init;label_free_anchor_budget_protocol_not_official;label_free_anchor_budget_A0_does_not_reproduce_locked_anchor;label_free_anchor_budget_LineC_nontearing_failed;strict_loss_agnostic_observables_not_visible
p4_open = 0
promotion_allowed = 0
strict_visibility_pass = 0
strict_auc_joint_min = 0.05660377358490566
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
line_t_context_rank_repair_best_protocol = context_rank_source_dataset_seed_window/ridge
line_t_context_rank_repair_best_auc_joint_min = 0.4798951048951049
line_t_context_rank_repair_best_precision_joint_min = 0.0
line_t_context_rank_repair_best_recall_joint_min = 0.0
line_t_context_rank_repair_best_pass = 0
```

本轮新增 strict checks：

```text
dict_b192_check:
  run_id = support_expand_seed012_dict_u10_u14_b192_plus_seed345_b128_w10_12_15_strict_check
  strict_auc_joint_min = 0.031914893617021274
  strict_precision_joint_min = 0.0
  strict_recall_joint_min = 0.0
  strict_visibility_pass = 0

dict_b256_check:
  run_id = support_expand_seed012_dict_u10_u14_b256_plus_seed345_b128_w10_12_15_strict_check
  strict_auc_joint_min = 0.024565508021390375
  strict_precision_joint_min = 0.0
  strict_recall_joint_min = 0.0
  strict_visibility_pass = 0
  blocker includes support_concentrated
```

最新验证：

```text
conda run -n kan python -m py_compile experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py
py_compile_exit = 0
trailing_whitespace_count = 0
```

## 28. EOF 最新权威索引：hard-gate margin repair 后

本文件历史追加顺序存在中段插入。以本节为最新权威状态；第 27 节记录了 hard-gate margin repair 的完整命令与结果。

最新主审计产物：

```text
result_dir = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_sha256 = 43262f6c065ce34f5f574c8ce72a254bb699ba1fab6a57950f27323742521a1c
zip_entries = 70
hash_manifest_entries = 45
```

最新主 route：

```text
run_id = official_from_v1217_v1216_artifacts_context_rank_dict_u10_u14_margin_repair_refresh
route = R2-B320LabelInitDependenceDetected
p4_open = 0
promotion_allowed = 0
strict_visibility_pass = 0
strict_auc_joint_min = 0.05660377358490566
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
line_t_context_rank_repair_best_auc_joint_min = 0.4798951048951049
line_t_hard_gate_margin_repair_rows = 56
line_t_hard_gate_margin_usable_rows = 4
line_t_hard_gate_margin_best_source = repair_seed345_sketchdim24_rank5_b128_w10_12_15
line_t_hard_gate_margin_best_threshold = -0.01
line_t_hard_gate_margin_best_joint_support = 16
line_t_hard_gate_margin_best_new_u10_u14_support = 0
line_t_hard_gate_margin_repair_pass = 0
```

最新 margin 结论：

```text
raise hard gate margin 已执行；
可 usable margin rows 没有新增 U10-U14 support；
U10-U14 support 存在时 random false positive 仍存在；
Line T 仍未通过；
P3/P4 仍不允许打开。
```

最新验证：

```text
conda run -n kan python -m py_compile experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py
py_compile_exit = 0
trailing_whitespace_count = 0
```

## 27. EOF 最新权威索引：hard-gate margin repair 后

本文件历史追加顺序存在中段插入。以本节为最新权威状态；第 24-26 节记录了 U10-U14 dictionary expansion，第 27 节记录本轮 `raise hard gate margin` 修复尝试。

### 27.1 代码修改

修改文件：

```text
/home/chengshun.wang/DG-LCA/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

新增内容：

```text
REQUIRED_ARTIFACTS:
  v1218_hard_gate_margin_repair.csv
  v1218_hard_gate_margin_repair_summary.csv

新增函数：
  write_hard_gate_margin_repair(...)

新增 CLI：
  --margin-source-run
```

审计约束：

```text
threshold 只向更严格方向移动：-0.0100, -0.0125, -0.0150, -0.0175, -0.0200, -0.0250, -0.0300
不降低 release threshold
label 只作为 audit target
uses_label_for_feature = 0
uses_ce_for_feature = 0
selection_for_promotion = 0
margin_repair_pass = 0
```

### 27.2 编译

命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py \
  experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py
```

结果：

```text
py_compile_exit = 0
```

### 27.3 主审计刷新命令

命令：

```bash
conda run -n kan python experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py \
  --run-id official_from_v1217_v1216_artifacts_context_rank_dict_u10_u14_margin_repair_refresh \
  --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/support_expand_seed012_seed345_b128_w10_12_15 \
  --v1216-repair-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_sketchdim24_rank5_b128_w10_12_15 \
  --margin-source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_sketchdim24_rank5_b128_w10_12_15 \
  --margin-source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_sketchdim24_rank5_b128_w10_12_15 \
  --margin-source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_dict_expand_u10_u14_sketchdim24_rank5_b128_w10_12_15 \
  --margin-source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_dict_expand_u10_u14_sketchdim24_rank5_b192_w10_12_15 \
  --margin-source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_dict_expand_u10_u14_sketchdim24_rank5_b256_w10_12_15 \
  --margin-source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_dict_expand_u10_u14_sketchdim24_rank5_b128_w10_12_15 \
  --margin-source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_dict_expand_u10_u14_sketchdim24_rank5_b192_w10_12_15 \
  --margin-source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_dict_expand_u10_u14_sketchdim24_rank5_b256_w10_12_15
```

### 27.4 margin sweep 结果

输出文件：

```text
results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_hard_gate_margin_repair.csv
results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_hard_gate_margin_repair_summary.csv
```

summary：

```text
margin_source_count = 8
margin_rows = 56
usable_margin_rows = 4
best_margin_source_run = repair_seed345_sketchdim24_rank5_b128_w10_12_15
best_margin_threshold = -0.01
best_margin_joint_hard_support_rows = 16
best_margin_new_u10_u14_hard_support_rows = 0
best_margin_random_joint_false_positive_rows = 0
best_margin_support_concentrated = 0
margin_repair_pass = 0
pass_reason = diagnostic_only; margin sweep cannot open P3/P4 without strict heldout visibility pass
```

关键明细：

```text
repair_seed012 original / threshold -0.01:
  joint = 12
  new_u10_u14 = 0
  random_fp = 0
  usable = 1

repair_seed345 original / threshold -0.01, -0.0125, -0.015:
  joint = 16
  new_u10_u14 = 0
  random_fp = 0
  usable = 1

repair_seed012_dict B128 / threshold -0.01:
  joint = 20
  new_u10_u14 = 8
  random_fp = 8
  usable = 0

repair_seed345_dict B128 / threshold -0.01:
  joint = 44
  new_u10_u14 = 28
  random_fp = 28
  usable = 0

repair_seed345_dict B128 / threshold -0.02:
  joint = 24
  new_u10_u14 = 16
  random_fp = 16
  usable = 0

repair_seed345_dict B128 / threshold -0.03:
  joint = 8
  new_u10_u14 = 8
  random_fp = 8
  usable = 0

repair_seed345_dict B192/B256:
  stricter thresholds reduce support but random_fp remains tied to U13/U14 support
  no usable U10-U14 margin row observed
```

解释：

```text
raise hard gate margin 没有解决 dictionary expansion 的 blocker。
凡是 U10-U14 hard support 仍存在的 margin 点，random matched control false positive 也仍存在。
可 usable 的 margin 点全部来自原始非 U10-U14 source。
```

### 27.5 最新主 route

```text
run_id = official_from_v1217_v1216_artifacts_context_rank_dict_u10_u14_margin_repair_refresh
route = R2-B320LabelInitDependenceDetected
fail_reason = current_B320_uses_label_informed_trainprobe_init;label_free_anchor_budget_protocol_not_official;label_free_anchor_budget_A0_does_not_reproduce_locked_anchor;label_free_anchor_budget_LineC_nontearing_failed;strict_loss_agnostic_observables_not_visible
p4_open = 0
promotion_allowed = 0
strict_visibility_pass = 0
strict_auc_joint_min = 0.05660377358490566
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
line_t_context_rank_repair_best_auc_joint_min = 0.4798951048951049
line_t_hard_gate_margin_repair_rows = 56
line_t_hard_gate_margin_usable_rows = 4
line_t_hard_gate_margin_best_source = repair_seed345_sketchdim24_rank5_b128_w10_12_15
line_t_hard_gate_margin_best_threshold = -0.01
line_t_hard_gate_margin_best_joint_support = 16
line_t_hard_gate_margin_best_new_u10_u14_support = 0
line_t_hard_gate_margin_repair_pass = 0
```

### 27.6 最新 zip

```text
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_sha256 = 43262f6c065ce34f5f574c8ce72a254bb699ba1fab6a57950f27323742521a1c
zip_entries = 70
hash_manifest_entries = 45
```

zip 内容核验：

```text
code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
review_artifacts/v1218_hard_gate_margin_repair.csv
review_artifacts/v1218_hard_gate_margin_repair_summary.csv
```

本节结论：

```text
hard-gate margin repair 已执行；
更严格 threshold 没有救回 U10-U14；
Line T 仍未通过；
P3/P4 仍不允许打开。
```

## 24. EOF 继续推进：actuator dictionary expansion / batch-bracket repair

触发原因：

```text
用户要求：未完成则继续，不允许没执行完就退出。
计划依据：v12.18 Line T 8.7 失败后动作：
  如果 AUC 高但 precision/recall 低：
    检查 support concentration；
    增加 windows / seeds / actuator dictionary；
    不降低 release threshold。
```

本轮新增修复面：

```text
代码文件：
  /home/chengshun.wang/DG-LCA/experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py

修改函数：
  make_p1_probe_updates(...)

新增 loss-agnostic probe dictionary entries：
  U10-AllRoleLowLiftCotangent
  U11-BranchQuadLowLiftPrimitiveActuator
  U12-DirectLowLiftPrimitiveActuator
  U13-MixedLowRankNoBranchBasis
  U14-MixedBranchQuadDirectLowLiftBasis

审计约束：
  loss_agnostic_direction = 1
  ce_vector_used_for_direction = 0
  label_used_for_direction = 0
  validation_used_for_commit = 0
  dataset_name_used_for_commit = 0
  不修改 hard release threshold
  不修改 P3/P4 gate
```

### 24.1 编译检查

命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py \
  experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py \
  experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

结果：

```text
py_compile_exit = 0
```

### 24.2 第一版 U10-U14 blocker 与修复

第一次尝试使用了 `projection/readout` role，运行时失败：

```text
KeyError: 'projection'
```

原因：

```text
experiments/run_v1214_b320locked_lossagnostic_functional_mechanism.py::ROLE_TOKENS
只支持：
  direct
  quad
  branch
  direct_quad
  branch_quad
```

按代码实际支持面修复：

```text
U11 从 projection 改为 branch_quad
U12 从 readout 改为 direct
U14 改为 U7 + U11 + U12 的 mixed branch/quad/direct low-lift basis
```

修复后再次执行 `py_compile`：

```text
py_compile_exit = 0
```

### 24.3 v12.16 source：seed012 / B128 / dictionary expansion

命令：

```bash
conda run -n kan python experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py \
  --probe-device auto \
  --probe-datasets MNIST,Fashion-MNIST,KMNIST \
  --probe-seeds 0,1,2 \
  --probe-splits 1 \
  --windows 10,12,15 \
  --actuator-windows 5 \
  --functional-batch-size 128 \
  --probe-train-size 512 \
  --probe-val-size 256 \
  --probe-test-size 256 \
  --probe-lr 0.01 \
  --sketch-dim 24 \
  --output-subspace-rank 5 \
  --ridge-lambda 0.001 \
  --actuator-budget-multiplier 8.0 \
  --target-logit-drift 0.04 \
  --safety-batch-factor 2 \
  --no-download \
  --out-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_dict_expand_u10_u14_sketchdim24_rank5_b128_w10_12_15 \
  --fresh
```

结果：

```text
route = R2-EstimatorStillWeak
p1_pass = 0
p2_pass = 0
p3_response_pass = 0
p4_open = 0
p1_best_noise_abs_spearman = 0.07999020895762843
p1_best_reservoir_abs_spearman = 0.14321878022935114
```

Line C support 检查：

```text
rows = 1404
hard_joint_rows = 20
hard_methods =
  U7-ProjectionQuadRoleActuator = 4
  U6-RoleConditionedPrimitiveActuator = 4
  U9-MixedLowRankActuatorBasis = 4
  U10-AllRoleLowLiftCotangent = 4
  U13-MixedLowRankNoBranchBasis = 4
random_joint_fp_rows = 8
noop_joint_fp_rows = 0
```

### 24.4 v12.17 calibration：seed012 B128 dict + seed345 B128 original

命令：

```bash
conda run -n kan python experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py \
  --run-id support_expand_seed012_dict_u10_u14_plus_seed345_b128_w10_12_15 \
  --out-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/support_expand_seed012_dict_u10_u14_plus_seed345_b128_w10_12_15 \
  --source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_dict_expand_u10_u14_sketchdim24_rank5_b128_w10_12_15 \
  --source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_sketchdim24_rank5_b128_w10_12_15
```

结果：

```text
route = R2-ReleaseUnobservableUnderLossAgnosticFeatures
p1_calibrated_source_runs = ['repair_seed345_sketchdim24_rank5_b128_w10_12_15']
seed012_dict_source_calibration_pass = 0
seed012_dict_random_joint_false_positive_rows = 8
seed012_dict_control_joint_false_positive_rate = 0.009259259259259259
p1_joint_hard_support_rows = 36
p2_visibility_repair_best_auc_joint_min = 0.7241671580188679
p2_visibility_repair_best_precision_joint_min = 0.0
p2_visibility_repair_best_recall_joint_min = 0.0
p4_open = 0
```

解释：

```text
新增 U10/U13 确实增加 hard support；
但该 source 因 random control false positive 被 calibration 排除；
不能作为 strict source promoted。
```

### 24.5 按 calibration 建议尝试 batch bracket：seed012 B192/B256

计划/CSV 建议：

```text
expand batch/bootstrap or raise hard gate margin; do not lower release threshold
```

本轮只尝试扩大 batch，不 raise hard gate margin。

B192 命令：

```bash
conda run -n kan python experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py \
  --probe-device auto \
  --probe-datasets MNIST,Fashion-MNIST,KMNIST \
  --probe-seeds 0,1,2 \
  --probe-splits 1 \
  --windows 10,12,15 \
  --actuator-windows 5 \
  --functional-batch-size 192 \
  --probe-train-size 512 \
  --probe-val-size 256 \
  --probe-test-size 256 \
  --probe-lr 0.01 \
  --sketch-dim 24 \
  --output-subspace-rank 5 \
  --ridge-lambda 0.001 \
  --actuator-budget-multiplier 8.0 \
  --target-logit-drift 0.04 \
  --safety-batch-factor 2 \
  --no-download \
  --out-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_dict_expand_u10_u14_sketchdim24_rank5_b192_w10_12_15 \
  --fresh
```

B192 结果：

```text
route = R2-EstimatorStillWeak
rows = 1404
hard_joint_rows = 8
hard_methods =
  U7-ProjectionQuadRoleActuator = 4
  U6-RoleConditionedPrimitiveActuator = 4
random_joint_fp_rows = 0
noop_joint_fp_rows = 0
```

B256 命令：

```bash
conda run -n kan python experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py \
  --probe-device auto \
  --probe-datasets MNIST,Fashion-MNIST,KMNIST \
  --probe-seeds 0,1,2 \
  --probe-splits 1 \
  --windows 10,12,15 \
  --actuator-windows 5 \
  --functional-batch-size 256 \
  --probe-train-size 512 \
  --probe-val-size 256 \
  --probe-test-size 256 \
  --probe-lr 0.01 \
  --sketch-dim 24 \
  --output-subspace-rank 5 \
  --ridge-lambda 0.001 \
  --actuator-budget-multiplier 8.0 \
  --target-logit-drift 0.04 \
  --safety-batch-factor 2 \
  --no-download \
  --out-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_dict_expand_u10_u14_sketchdim24_rank5_b256_w10_12_15 \
  --fresh
```

B256 结果：

```text
route = R2-EstimatorStillWeak
rows = 1404
hard_joint_rows = 0
random_joint_fp_rows = 0
noop_joint_fp_rows = 0
```

### 24.6 v12.17/v12.18：seed012 B192/B256 strict checks

B192 v12.17 命令：

```bash
conda run -n kan python experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py \
  --run-id support_expand_seed012_dict_u10_u14_b192_plus_seed345_b128_w10_12_15 \
  --out-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/support_expand_seed012_dict_u10_u14_b192_plus_seed345_b128_w10_12_15 \
  --source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_dict_expand_u10_u14_sketchdim24_rank5_b192_w10_12_15 \
  --source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_sketchdim24_rank5_b128_w10_12_15
```

B192 v12.17 结果：

```text
p1_calibrated_source_runs =
  repair_seed012_dict_expand_u10_u14_sketchdim24_rank5_b192_w10_12_15
  repair_seed345_sketchdim24_rank5_b128_w10_12_15
p1_joint_hard_support_rows = 24
p2_visibility_repair_best_auc_joint_min = 0.8059269162210339
p2_visibility_repair_best_precision_joint_min = 0.0
p2_visibility_repair_best_recall_joint_min = 0.0
p4_open = 0
```

B192 v12.18 命令：

```bash
conda run -n kan python experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py \
  --run-id support_expand_seed012_dict_u10_u14_b192_plus_seed345_b128_w10_12_15_strict_check \
  --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/support_expand_seed012_dict_u10_u14_b192_plus_seed345_b128_w10_12_15_strict_check \
  --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/support_expand_seed012_dict_u10_u14_b192_plus_seed345_b128_w10_12_15 \
  --v1216-repair-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_dict_expand_u10_u14_sketchdim24_rank5_b192_w10_12_15
```

B192 v12.18 结果：

```text
route = R2-B320LabelInitDependenceDetected
hard_joint_support_rows = 24
strict_auc_joint_min = 0.031914893617021274
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate
line_t_context_rank_repair_best_auc_joint_min = 0.48060344827586204
line_t_context_rank_repair_best_precision_joint_min = 0.0
line_t_context_rank_repair_best_recall_joint_min = 0.0
p4_open = 0
```

B256 v12.17/v12.18 结果摘要：

```text
v12.17 calibrated_sources =
  repair_seed012_dict_expand_u10_u14_sketchdim24_rank5_b256_w10_12_15
  repair_seed345_sketchdim24_rank5_b128_w10_12_15

v12.18 run_id = support_expand_seed012_dict_u10_u14_b256_plus_seed345_b128_w10_12_15_strict_check
hard_joint_support_rows = 16
strict_auc_joint_min = 0.024565508021390375
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate;support_concentrated
line_t_context_rank_repair_best_auc_joint_min = 0.48702830188679247
p4_open = 0
```

### 24.7 seed345 dictionary expansion checks

seed345 B128 命令：

```bash
conda run -n kan python experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py \
  --probe-device auto \
  --probe-datasets MNIST,Fashion-MNIST,KMNIST \
  --probe-seeds 3,4,5 \
  --probe-splits 1 \
  --windows 10,12,15 \
  --actuator-windows 5 \
  --functional-batch-size 128 \
  --probe-train-size 512 \
  --probe-val-size 256 \
  --probe-test-size 256 \
  --probe-lr 0.01 \
  --sketch-dim 24 \
  --output-subspace-rank 5 \
  --ridge-lambda 0.001 \
  --actuator-budget-multiplier 8.0 \
  --target-logit-drift 0.04 \
  --safety-batch-factor 2 \
  --no-download \
  --out-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_dict_expand_u10_u14_sketchdim24_rank5_b128_w10_12_15 \
  --fresh
```

seed345 B128 结果：

```text
route = R2-EstimatorStillWeak
rows = 1404
hard_joint_rows = 44
hard_methods =
  U13-MixedLowRankNoBranchBasis = 20
  U5-OrthogonalCotangentVJPEnsemble = 8
  U6-RoleConditionedPrimitiveActuator = 4
  U11-BranchQuadLowLiftPrimitiveActuator = 4
  U14-MixedBranchQuadDirectLowLiftBasis = 4
  U7-ProjectionQuadRoleActuator = 4
random_joint_fp_rows = 28
noop_joint_fp_rows = 0
```

seed345 B192/B256 结果：

```text
seed345 B192:
  hard_joint_rows = 44
  random_joint_fp_rows = 28
  source_calibration_pass = 0

seed345 B256:
  hard_joint_rows = 40
  random_joint_fp_rows = 24
  source_calibration_pass = 0
```

组合审计：

```text
support_expand_seed012_seed345_dict_u10_u14_b128_w10_12_15:
  route = R1-LineCCalibrationUnstable
  p1_calibrated_source_runs = []
  p1_joint_hard_support_rows = 64
  p2_visibility_repair_best_auc_joint_min = 0.695884771319242
  precision/recall = 0.0 / 0.0

support_expand_seed012_seed345_dict_u10_u14_b192_w10_12_15:
  p1_calibrated_source_runs =
    repair_seed012_dict_expand_u10_u14_sketchdim24_rank5_b192_w10_12_15
  seed345_dict_source_calibration_pass = 0
  p2_visibility_repair_best_auc_joint_min = 0.4813753581661891
  precision/recall = 0.0 / 0.0
```

### 24.8 刷新主 v12.18 审计包

目的：

```text
刷新官方主审计目录的 code_review_packet；
确保 zip 内包含本轮修改后的 v12.16 runner；
主 strict result 仍使用此前更干净的 seed012+seed345 B128 source，不把失败的 dictionary expansion 分支伪装为成功。
```

命令：

```bash
conda run -n kan python experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py \
  --run-id official_from_v1217_v1216_artifacts_support_expand_seed012_seed345_b128_w10_12_15_context_rank_repair_dict_u10_u14_code_audit_refresh \
  --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/support_expand_seed012_seed345_b128_w10_12_15 \
  --v1216-repair-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_sketchdim24_rank5_b128_w10_12_15
```

最新主审计结果：

```text
run_id = official_from_v1217_v1216_artifacts_support_expand_seed012_seed345_b128_w10_12_15_context_rank_repair_dict_u10_u14_code_audit_refresh
route = R2-B320LabelInitDependenceDetected
promotion_allowed = 0
p4_open = 0
hard_joint_support_rows = 28
strict_auc_joint_min = 0.05660377358490566
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate
line_t_context_rank_repair_best_auc_joint_min = 0.4798951048951049
line_t_context_rank_repair_best_precision_joint_min = 0.0
line_t_context_rank_repair_best_recall_joint_min = 0.0
```

最新 zip：

```text
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_sha256 = 6582b59646528335e6f9628f95edb57a6c85a4fc24fd3fed0b663d4d71dc3e36
zip_entries = 68
hash_manifest_entries = 43
```

zip 内容核验：

```text
code/experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py
code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
review_artifacts/v1218_route_decision.json
review_artifacts/v1218_hash_manifest.json
```

本轮结束判定：

```text
actuator dictionary expansion 已真实执行。
新增 U10-U14 能增加 hard support，但新增 support 在 seed345 上伴随 random-control false positive；
B192/B256 batch repair 能压掉部分 false positive，但不能改善 strict visibility；
所有 strict checks 的 precision/recall min 仍为 0。

因此：
  Line T 仍未通过；
  不允许进入 P3；
  不允许进入 P4；
  不允许 promotion；
  不降低 threshold。
```

## 22. 继续推进：seed/window support expansion + method-generalizable context-rank repair

本节是在第 21 节之后继续推进。第 21 节的 blocker 已经不是 `support_concentrated` 或 `required_leaveout_not_runnable`，而是：

```text
strict_auc_joint_below_gate
strict_precision_recall_below_gate
worst_split = leave-method-out / U9-MixedLowRankActuatorBasis
```

继续推进原则：

```text
1. 不降低 release threshold。
2. 不把 label/CE 加入 feature construction。
3. 不把未校准 source 强行加入 promotion visibility。
4. 不打开 P3/P4，除非 Line T hard/robust visibility 真通过。
```

### 22.1 新增 seed678 source

命令：

```bash
conda run -n kan python experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py \
  --probe-device auto --probe-datasets MNIST,Fashion-MNIST,KMNIST \
  --probe-seeds 6,7,8 --probe-splits 1 --windows 10,12,15 --actuator-windows 5 \
  --functional-batch-size 128 --probe-train-size 512 --probe-val-size 256 --probe-test-size 256 \
  --probe-lr 0.01 --sketch-dim 24 --output-subspace-rank 5 --ridge-lambda 0.001 \
  --actuator-budget-multiplier 8.0 --target-logit-drift 0.04 --safety-batch-factor 2 \
  --no-download --out-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed678_sketchdim24_rank5_b128_w10_12_15
```

结果：

```text
run_id = repair_seed678_sketchdim24_rank5_b128_w10_12_15
route = R2-EstimatorStillWeak
p0_pass = 1
p1_pass = 0
p1_best_noise_abs_spearman = 0.117271743127479
p1_best_reservoir_abs_spearman = 0.16378848952471092
p1_best_negative_release_precision = 0.0
p1_best_negative_release_recall = 0.0
p3_max_actuator_sketch_delta_fro = 0.008893998339772224
p3_max_actuator_projector_angle_deg = 0.5695825815200806
p3_best_noise_delta = -0.005090683698654175
p3_best_reservoir_delta = -0.006326198577880859
p4_open = 0
```

解释：

```text
seed678 是真实新增 source，但 v12.16 自身仍是 R2，不是机制成功。
```

### 22.2 seed012+seed345+seed678 v12.17 calibration

命令：

```bash
conda run -n kan python experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py \
  --run-id support_expand_seed012_seed345_seed678_b128_w10_12_15 \
  --out-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/support_expand_seed012_seed345_seed678_b128_w10_12_15 \
  --source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_sketchdim24_rank5_b128_w10_12_15 \
  --source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_sketchdim24_rank5_b128_w10_12_15 \
  --source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed678_sketchdim24_rank5_b128_w10_12_15
```

结果：

```text
run_id = support_expand_seed012_seed345_seed678_b128_w10_12_15
route = R2-ReleaseUnobservableUnderLossAgnosticFeatures
p1_calibrated_source_runs = [
  repair_seed012_sketchdim24_rank5_b128_w10_12_15,
  repair_seed345_sketchdim24_rank5_b128_w10_12_15
]

repair_seed678 source_calibration_pass = 0
repair_seed678 random_joint_false_positive_rows = 0
repair_seed678 control_joint_false_positive_rows = 0
repair_seed678 joint_hard_support_rows = 4

release_label_rows = 1728
hard_joint_rows = 28
hard_joint_by_source:
  repair_seed012_sketchdim24_rank5_b128_w10_12_15 = 12
  repair_seed345_sketchdim24_rank5_b128_w10_12_15 = 16
```

解释：

```text
seed678 没有通过 source calibration，因此不能强行加入 promotion visibility。
v12.17 release labels 仍只包含 seed012/seed345 两个 calibrated sources。
```

### 22.3 mixed-window support expansion

为了继续测试 plan 8.7 的 `增加 windows / seeds / actuator dictionary` 方向，复用已真实存在且校准过的 `repair_seed345_sketchdim24_rank5_b128_w8_10_12`，与 seed012/seed345 w10,12,15 组合。

命令：

```bash
conda run -n kan python experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py \
  --run-id support_expand_seed012_seed345_mixed_w8_10_12_15 \
  --out-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/support_expand_seed012_seed345_mixed_w8_10_12_15 \
  --source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_sketchdim24_rank5_b128_w10_12_15 \
  --source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_sketchdim24_rank5_b128_w8_10_12 \
  --source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_sketchdim24_rank5_b128_w10_12_15
```

v12.17 结果：

```text
run_id = support_expand_seed012_seed345_mixed_w8_10_12_15
route = R2-ReleaseUnobservableUnderLossAgnosticFeatures
p1_calibrated_source_runs = [
  repair_seed012_sketchdim24_rank5_b128_w10_12_15,
  repair_seed345_sketchdim24_rank5_b128_w10_12_15,
  repair_seed345_sketchdim24_rank5_b128_w8_10_12
]
p2_visibility_repair_best_protocol = repair_coupling_spectrum_persistence/mean_noise_reservoir_rank
p2_visibility_repair_best_auc_joint_min = 0.772028777603895
p2_visibility_repair_best_precision_joint_min = 0.0
p2_visibility_repair_best_recall_joint_min = 0.0
p4_open = 0

release_label_rows = 2592
hard_joint_rows = 36
hard_joint_by_source:
  repair_seed012_sketchdim24_rank5_b128_w10_12_15 = 12
  repair_seed345_sketchdim24_rank5_b128_w8_10_12 = 8
  repair_seed345_sketchdim24_rank5_b128_w10_12_15 = 16
hard_joint_by_window:
  10 = 12
  12 = 16
  15 = 8
hard_joint_by_method:
  U5-OrthogonalCotangentVJPEnsemble = 12
  U6-RoleConditionedPrimitiveActuator = 8
  U7-ProjectionQuadRoleActuator = 12
  U9-MixedLowRankActuatorBasis = 4
```

mixed-window v12.18 命令：

```bash
conda run -n kan python experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py \
  --run-id official_from_v1217_mixed_w8_10_12_15_support_context_pending \
  --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_mixed_w8_10_12_15_support_context_pending \
  --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/support_expand_seed012_seed345_mixed_w8_10_12_15 \
  --v1216-repair-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_sketchdim24_rank5_b128_w10_12_15
```

mixed-window v12.18 结果：

```text
run_id = official_from_v1217_mixed_w8_10_12_15_support_context_pending
route = R2-B320LabelInitDependenceDetected
strict_visibility_feature_rows = 121824
strict_visibility_wide_rows = 2592
strict_visibility_feature_columns = 47
hard_joint_support_rows = 36
support_concentrated = 0
strict_visibility_required_split_failures = 0
strict_auc_joint_min = 0.0375
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate

worst_split = leave-method-out / U9-MixedLowRankActuatorBasis
worst_AUC_joint = 0.0375
worst_precision_at_k_joint = 0.0
worst_recall_at_k_joint = 0.0
worst_positive_rows_joint = 4
```

解释：

```text
mixed-window support 增加 hard_joint_rows 到 36，但没有增加 U9 hard_joint support；
主 strict AUC 反而从 0.05660377358490566 降到 0.0375。
因此 mixed-window 不是修复方向，不能作为最终主结果。
```

### 22.4 代码修改：context-rank repair

修改文件：

```text
experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

新增内容：

```text
1. REQUIRED_ARTIFACTS 加入 v1218_strict_visibility_context_rank_repair.csv。
2. 新增 centroid_predict，用于 context-rank 的 centroid scorer 诊断。
3. 新增 context_rank_matrix：
   - group = source_run,dataset,seed,window
   - group = dataset,seed,window
   - group = source_run,dataset,seed
   在每个 precommit candidate cohort 内对 feature 做 rank normalization。
4. 新增 write_strict_visibility_context_rank_repair：
   - scorer = ridge
   - scorer = positive_centroid_dot
   - scorer = positive_centroid_distance
   - scorer = contrast_centroid_dot
   - scorer = contrast_centroid_distance
5. context-rank repair 明确记录：
   uses_label_for_feature = 0
   uses_ce_for_feature = 0
   uses_method_identity_feature = 0
   selection_for_promotion = 0
```

注意：

```text
context-rank repair 使用 label 训练 audit scorer，但 label 不进入 feature construction。
该 repair 是 post-hoc Line T method-generalization diagnostic，不降低 gate，不直接用于 promotion。
```

### 22.5 最终主审计重跑

命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py

conda run -n kan python experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py \
  --run-id official_from_v1217_v1216_artifacts_support_expand_seed012_seed345_b128_w10_12_15_context_rank_repair \
  --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/support_expand_seed012_seed345_b128_w10_12_15 \
  --v1216-repair-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_sketchdim24_rank5_b128_w10_12_15
```

最终主审计结果：

```text
run_id = official_from_v1217_v1216_artifacts_support_expand_seed012_seed345_b128_w10_12_15_context_rank_repair
route = R2-B320LabelInitDependenceDetected
fail_reason = current_B320_uses_label_informed_trainprobe_init;label_free_anchor_budget_protocol_not_official;label_free_anchor_budget_A0_does_not_reproduce_locked_anchor;label_free_anchor_budget_LineC_nontearing_failed;strict_loss_agnostic_observables_not_visible
p4_open = 0
promotion_allowed = 0

strict_visibility_feature_rows = 81216
strict_visibility_wide_rows = 1728
strict_visibility_feature_columns = 47
hard_joint_support_rows = 28
support_concentrated = 0
strict_visibility_required_split_failures = 0
strict_auc_joint_min = 0.05660377358490566
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate
```

context-rank repair 最好结果：

```text
line_t_context_rank_repair_best_protocol = context_rank_source_dataset_seed_window/ridge
line_t_context_rank_repair_best_auc_joint_min = 0.4798951048951049
line_t_context_rank_repair_best_precision_joint_min = 0.0
line_t_context_rank_repair_best_recall_joint_min = 0.0
line_t_context_rank_repair_best_pass = 0
```

context-rank top summaries：

```text
context_rank_source_dataset_seed_window/ridge:
  AUC_joint_min = 0.4798951048951049
  AUC_joint_max = 0.9224759615384616
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  worst_split = leave-dataset-out / KMNIST
  repair_pass = 0

context_rank_dataset_seed_window/ridge:
  AUC_joint_min = 0.4798951048951049
  AUC_joint_max = 0.9224759615384616
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  worst_split = leave-dataset-out / KMNIST
  repair_pass = 0

context_rank_source_dataset_seed_window/positive_centroid_dot:
  AUC_joint_min = 0.35563380281690143
  AUC_joint_max = 0.9632867132867133
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  worst_split = leave-seed-out / 4
  repair_pass = 0
```

composite repair 仍失败：

```text
line_t_composite_repair_best_protocol = min_noise_reservoir_rank
line_t_composite_repair_best_auc_joint_min = 0.19811320754716982
line_t_composite_repair_best_precision_joint_min = 0.0
line_t_composite_repair_best_recall_joint_min = 0.0
line_t_composite_repair_best_pass = 0
```

### 22.6 最新 zip

检查命令：

```bash
bsdtar -tf results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip | rg 'context_rank|run_v1218_b320_codeaudit|route_decision|hash_manifest'
sha256sum results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
bsdtar -tf results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip | wc -l
```

输出：

```text
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_sha256 = b4ccb50b88d657ee1b0b9d036e75da590288da17979be6e646318c01ff8ae10b
zip_entries = 68
hash_manifest_entries = 43

zip_has code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py = True
zip_has review_artifacts/v1218_route_decision.json = True
zip_has review_artifacts/v1218_strict_visibility_context_rank_repair.csv = True
zip_has review_artifacts/v1218_hash_manifest.json = True
```

### 22.7 本节结论

```text
1. seed678 扩 seed 尝试失败：source_calibration_pass=0，不能纳入 calibrated release labels。
2. mixed-window support 虽增加 hard_joint_rows 到 36，但 U9 hard_joint_rows 仍只有 4，strict_auc_joint_min 变差到 0.0375。
3. context-rank method-generalization repair 的最好 AUC_joint_min 只有 0.4798951048951049，precision/recall 仍为 0。
4. Line T 仍不通过。
5. Line I / P3 和 Line B / P4 仍按 gate 关闭。
```

当前能推进的 plan 8.7 方向已经真实尝试：

```text
random cotangent / augmentation / output-subspace drift derived features：已尝试，失败。
support concentration / seeds / windows expansion：已尝试，support 结构修复但 visibility 仍失败；seed678 与 mixed-window 继续失败。
method-generalizable context rank：已尝试，失败。
```

因此当前 blocker 进一步收敛为：

```text
当前 hard-release target 在 strict loss-agnostic feature family 下不可观测；
继续做同类 post-hoc feature/rank/support 小修没有证据能打开 P3/P4。
```

## 20. 文件末尾最终权威状态更新：Line T support expansion 后的 v12.18 结论

本执行日志存在历史追加顺序不完全递增的问题；第 19 节已经记录了本轮继续推进的完整过程，但文件尾部旧状态仍可能造成误读。因此本节作为当前文件末尾的最终权威状态。旧的 `zip_sha256 = c670...`、`zip_entries = 66`、`strict_visibility_feature_columns = 36`、`support_concentrated = 1` 等记录均为上一阶段中间产物。

### 20.1 本轮最终链路命令

生成 matched seed012 source：

```bash
conda run -n kan python experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py \
  --probe-device auto --probe-datasets MNIST,Fashion-MNIST,KMNIST \
  --probe-seeds 0,1,2 --probe-splits 1 --windows 10,12,15 --actuator-windows 5 \
  --functional-batch-size 128 --probe-train-size 512 --probe-val-size 256 --probe-test-size 256 \
  --probe-lr 0.01 --sketch-dim 24 --output-subspace-rank 5 --ridge-lambda 0.001 \
  --actuator-budget-multiplier 8.0 --target-logit-drift 0.04 --safety-batch-factor 2 \
  --no-download --out-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_sketchdim24_rank5_b128_w10_12_15
```

生成 independent seed345 source：

```bash
conda run -n kan python experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py \
  --probe-device auto --probe-datasets MNIST,Fashion-MNIST,KMNIST \
  --probe-seeds 3,4,5 --probe-splits 1 --windows 10,12,15 --actuator-windows 5 \
  --functional-batch-size 128 --probe-train-size 512 --probe-val-size 256 --probe-test-size 256 \
  --probe-lr 0.01 --sketch-dim 24 --output-subspace-rank 5 --ridge-lambda 0.001 \
  --actuator-budget-multiplier 8.0 --target-logit-drift 0.04 --safety-batch-factor 2 \
  --no-download --out-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_sketchdim24_rank5_b128_w10_12_15
```

生成 matched v12.17 visibility source：

```bash
conda run -n kan python experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py \
  --run-id support_expand_seed012_seed345_b128_w10_12_15 \
  --out-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/support_expand_seed012_seed345_b128_w10_12_15 \
  --source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_sketchdim24_rank5_b128_w10_12_15 \
  --source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_sketchdim24_rank5_b128_w10_12_15
```

最终 v12.18 主审计：

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py && \
conda run -n kan python experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py \
  --run-id official_from_v1217_v1216_artifacts_support_expand_seed012_seed345_b128_w10_12_15_role_time_composite \
  --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/support_expand_seed012_seed345_b128_w10_12_15 \
  --v1216-repair-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_sketchdim24_rank5_b128_w10_12_15
```

### 20.2 本轮最终代码修改

修改文件：

```text
experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

主要修改：

```text
1. 支持从 v12.17 route/source_run_dirs 读取多个 calibrated source run，而不是只读取一个 v12.16 repair dir。
2. strict v12.16 long rows 构建时只保留 v12.17 release labels 覆盖到的 rows，避免把缺失 label 当作 negative。
3. 增加 role/time/persistence/projector derived strict features：
   method_role_flags
   window_schedule
   time_persistence_derived
   projector_geometry_derived
4. 增加 Line T support expansion audit：
   v1218_line_t_support_expansion_audit.csv
5. 增加 strict composite repair：
   v1218_strict_visibility_composite_repair.csv
6. 更新审计包 REQUIRED_ARTIFACTS / zip 打包内容。
```

这些修改只改变审计/诊断 runner 与审计产物生成；未降低 release threshold，未把 label/CE 用作 strict feature 构造或 direction 选择。

### 20.3 新 source 与 v12.17 support

matched v12.16 sources：

```text
repair_seed012_sketchdim24_rank5_b128_w10_12_15:
  route = R2-EstimatorStillWeak
  p0_pass = 1
  p1_pass = 0
  p1_best_noise_abs_spearman = 0.18818886064881887
  p1_best_reservoir_abs_spearman = 0.2605555287728709
  p1_best_negative_release_precision = 0.0
  p1_best_negative_release_recall = 0.0
  p3_max_actuator_sketch_delta_fro = 0.008491670712828636
  p3_max_actuator_projector_angle_deg = 2.2786431312561035

repair_seed345_sketchdim24_rank5_b128_w10_12_15:
  route = R2-EstimatorStillWeak
  p0_pass = 1
  p1_pass = 0
  p1_best_noise_abs_spearman = 0.0619349104560141
  p1_best_reservoir_abs_spearman = 0.09456052759880126
  p1_best_negative_release_precision = 0.0
  p1_best_negative_release_recall = 0.0
  p3_max_actuator_sketch_delta_fro = 0.012493130750954151
  p3_max_actuator_projector_angle_deg = 0.6425489187240601
```

matched v12.17 source：

```text
run_id = support_expand_seed012_seed345_b128_w10_12_15
route = R2-ReleaseUnobservableUnderLossAgnosticFeatures
p1_calibrated_source_runs = [
  repair_seed012_sketchdim24_rank5_b128_w10_12_15,
  repair_seed345_sketchdim24_rank5_b128_w10_12_15
]
p2_visibility_repair_best_protocol = repair_coupling_spectrum_persistence/mean_noise_reservoir_rank
p2_visibility_repair_best_auc_joint_min = 0.7844852941176471
p2_visibility_repair_best_precision_joint_min = 0.0
p2_visibility_repair_best_recall_joint_min = 0.0
p4_open = 0
```

v12.17 release support：

```text
release_label_rows = 1728
hard_joint_rows = 28
hard_joint_by_source:
  repair_seed012_sketchdim24_rank5_b128_w10_12_15 = 12
  repair_seed345_sketchdim24_rank5_b128_w10_12_15 = 16
hard_joint_by_window:
  10 = 8
  12 = 12
  15 = 8
hard_joint_by_method:
  U5-OrthogonalCotangentVJPEnsemble = 8
  U6-RoleConditionedPrimitiveActuator = 8
  U7-ProjectionQuadRoleActuator = 8
  U9-MixedLowRankActuatorBasis = 4
```

### 20.4 最终 v12.18 route

```text
run_id = official_from_v1217_v1216_artifacts_support_expand_seed012_seed345_b128_w10_12_15_role_time_composite
route = R2-B320LabelInitDependenceDetected
fail_reason = current_B320_uses_label_informed_trainprobe_init;label_free_anchor_budget_protocol_not_official;label_free_anchor_budget_A0_does_not_reproduce_locked_anchor;label_free_anchor_budget_LineC_nontearing_failed;strict_loss_agnostic_observables_not_visible
p4_open = 0
promotion_allowed = 0

strict_visibility_feature_rows = 81216
strict_visibility_wide_rows = 1728
strict_visibility_feature_columns = 47
hard_joint_support_rows = 28
support_concentrated = 0
strict_visibility_required_split_failures = 0

strict_auc_joint_min = 0.05660377358490566
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate

line_t_support_expansion_pass = 1
line_t_support_expansion_blocker = smoke_source_not_official_independent_source
line_t_family_ablation_best_set = only_augmentation_consistency_proxy
line_t_family_ablation_best_auc_joint_min = 0.5
line_t_family_ablation_best_visibility_pass = 0

line_t_composite_repair_best_protocol = min_noise_reservoir_rank
line_t_composite_repair_best_auc_joint_min = 0.19811320754716982
line_t_composite_repair_best_precision_joint_min = 0.0
line_t_composite_repair_best_recall_joint_min = 0.0
line_t_composite_repair_best_pass = 0
```

关键 leaveout：

```text
base strict worst split:
  leave-method-out / U9-MixedLowRankActuatorBasis
  AUC_joint = 0.05660377358490566
  precision_at_k_joint = 0.0
  recall_at_k_joint = 0.0
  positive_rows_joint = 4
  topk_hits_joint = 0

leave-source-run-out / repair_seed012:
  AUC_joint = 0.4849374021909233
  precision_at_k_joint = 0.0
  recall_at_k_joint = 0.0
  positive_rows_joint = 12

leave-source-run-out / repair_seed345:
  AUC_joint = 0.5772405660377359
  precision_at_k_joint = 0.0
  recall_at_k_joint = 0.0
  positive_rows_joint = 16
```

strict composite repair summary：

```text
mean_noise_reservoir_rank:
  AUC_joint_min = 0.12794811320754718
  AUC_joint_max = 0.8705985915492958
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  repair_pass = 0

min_noise_reservoir_rank:
  AUC_joint_min = 0.19811320754716982
  AUC_joint_max = 0.8911409198113207
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  repair_pass = 0

product_noise_reservoir_rank:
  AUC_joint_min = 0.1792452830188679
  AUC_joint_max = 0.8683978873239436
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  repair_pass = 0
```

### 20.5 最新 zip / 审计包

检查命令：

```bash
bsdtar -tf results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip | rg 'run_v1218_b320_codeaudit|composite_repair|support_expansion|route_decision|hash_manifest'
sha256sum results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
bsdtar -tf results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip | wc -l
```

输出：

```text
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_sha256 = eb970193cf11173e8a8adf450e7cdca112666351f7af79118909cf65ab4d1d30
zip_entries = 67
hash_manifest_entries = 42

zip_has code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py = True
zip_has review_artifacts/v1218_route_decision.json = True
zip_has review_artifacts/v1218_strict_visibility_composite_repair.csv = True
zip_has review_artifacts/v1218_line_t_support_expansion_audit.csv = True
zip_has review_artifacts/v1218_hash_manifest.json = True
```

### 20.6 当前最终判断

本轮已经按文档推荐方向继续推进 Line T，并尝试了 support expansion、独立 seed/window source、role/time/persistence/projector derived features、strict composite repair。结果是：

```text
1. support_concentrated 已从 1 修复为 0。
2. required_leaveout_not_runnable 已从 1 修复为 0。
3. 但 strict visibility 仍失败：
   strict_auc_joint_min = 0.05660377358490566
   strict_precision_joint_min = 0.0
   strict_recall_joint_min = 0.0
4. P4 仍关闭：
   p4_open = 0
   promotion_allowed = 0
```

因此不能声明 v12.18 promotion，也不能打开 Line I/P4。当前 blocker 已从“support/leaveout 不足”推进为更具体的“U9/MixedLowRank 等 method-generalization 下，strict loss-agnostic observables 对 hard joint release 仍不可见”。

## 19. 继续推进：Line T support expansion + matched-window source + strict composite repair

### 19.1 继续推进原因

上一轮最终状态仍有两个 Line T blocker：

```text
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate;support_concentrated;required_leaveout_not_runnable
```

按计划 8.7：

```text
如果 AUC 高但 precision/recall 低：
  检查 support concentration；
  增加 windows / seeds / actuator dictionary；
  不降低 release threshold。
```

本轮继续推进目标：

```text
1. 生成新的独立 strict source，修复 leave-source-run-out 不可运行。
2. 扩大 batch/window，尝试降低 random/control false positive。
3. 对齐两个 source 的 window dictionary，排除 schedule mismatch。
4. 增加 role/time-persistence strict features。
5. 增加 strict composite visibility repair，测试直接 joint ridge 是否因为 joint target 过稀疏而失败。
```

### 19.2 修改的代码

修改文件：

```text
experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

修改内容：

```text
1. strict visibility 不再只读单个 --v1216-repair-dir。
   新增 strict_visibility_source_dirs(v1217_dir, fallback_v1216_dir)，从 v1217_route_decision.json 的 source_run_dirs 读取多个 source。

2. build_strict_v1216_long_rows(...) 改为读取多个 source dirs。
   只保留存在 v1217 audit labels 的 strict rows；
   没有 labels 的 source 不再被隐式当作 hard_release=0。

3. feature provenance / feature-engineering audit 的 source_artifact 改为记录多个 source artifact。

4. 新增 role/time strict features：
   method_role_flags:
     method_is_control
     method_is_random
     method_is_cotangent_vjp
     method_is_role_actuator
     method_is_projector

   window_schedule:
     window_value
     window_log

   time_persistence_derived:
     persistence_reservoir_fraction_absdiff
     persistence_projector_stability_absdiff
     persistence_pred_coupling_absdiff

   projector_geometry_derived:
     projector_stability_gap

5. 新增 add_strict_time_persistence(...)。
   persistence 只使用同 source/dataset/seed/method/sketch/split 下相邻 window 的 label-free columns；
   不使用 actual_CouplingR2_delta，不使用 label/CE。

6. 新增 v1218_strict_visibility_composite_repair.csv。
   规则包括：
     mean_noise_reservoir_rank
     min_noise_reservoir_rank
     product_noise_reservoir_rank

   composite repair 用 hard_noise_release / hard_reservoir_release 训练 audit scorer；
   labels 只作为 visibility evaluation target，不进入 feature construction 或 direction。

7. REQUIRED_ARTIFACTS 新增：
   v1218_strict_visibility_composite_repair.csv
```

### 19.3 新 source 生成与 v12.17 calibration

#### 19.3.1 seed345 / b64 / w3,5,10

命令：

```bash
conda run -n kan python experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py \
  --probe-device auto \
  --probe-datasets MNIST,Fashion-MNIST,KMNIST \
  --probe-seeds 3,4,5 \
  --probe-splits 1 \
  --windows 3,5,10 \
  --actuator-windows 5 \
  --functional-batch-size 64 \
  --probe-train-size 512 \
  --probe-val-size 256 \
  --probe-test-size 256 \
  --probe-lr 0.01 \
  --sketch-dim 24 \
  --output-subspace-rank 5 \
  --ridge-lambda 0.001 \
  --actuator-budget-multiplier 8.0 \
  --target-logit-drift 0.04 \
  --safety-batch-factor 2 \
  --no-download \
  --out-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_sketchdim24_rank5_b64_w3_5_10
```

v12.16 输出：

```text
route = R2-EstimatorStillWeak
p0_pass = 1
p1_pass = 0
p1_best_noise_abs_spearman = 0.08287590999371523
p1_best_reservoir_abs_spearman = 0.03838880992782826
p1_best_negative_release_precision = 0.3333333333333333
p1_best_negative_release_recall = 0.06666666666666667
p3_max_actuator_sketch_delta_fro = 0.012463713996112347
p3_max_actuator_projector_angle_deg = 0.80697101354599
```

v12.17 calibration 命令：

```bash
conda run -n kan python experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py \
  --run-id support_expand_repair_plus_seed345 \
  --out-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/support_expand_repair_plus_seed345 \
  --source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_sketchdim24_rank5_b64_w3_5_10 \
  --source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_sketchdim24_rank5_b64_w3_5_10
```

结果：

```text
repair_seed345_sketchdim24_rank5_b64_w3_5_10:
  source_calibration_pass = 0
  random_joint_false_positive_rows = 12
  control_joint_false_positive_rate = 0.037037037037037035

结论：
  b64 seed345 source 不能作为第二 calibrated source。
```

#### 19.3.2 seed345 / b128 / w3,5,10

命令：

```bash
conda run -n kan python experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py \
  --probe-device auto \
  --probe-datasets MNIST,Fashion-MNIST,KMNIST \
  --probe-seeds 3,4,5 \
  --probe-splits 1 \
  --windows 3,5,10 \
  --actuator-windows 5 \
  --functional-batch-size 128 \
  --probe-train-size 512 \
  --probe-val-size 256 \
  --probe-test-size 256 \
  --probe-lr 0.01 \
  --sketch-dim 24 \
  --output-subspace-rank 5 \
  --ridge-lambda 0.001 \
  --actuator-budget-multiplier 8.0 \
  --target-logit-drift 0.04 \
  --safety-batch-factor 2 \
  --no-download \
  --out-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_sketchdim24_rank5_b128_w3_5_10
```

v12.16 输出：

```text
route = R2-EstimatorStillWeak
p1_pass = 0
p1_best_noise_abs_spearman = 0.0782554674475702
p1_best_reservoir_abs_spearman = 0.06684860090369464
p1_best_negative_release_precision = 0.0
p1_best_negative_release_recall = 0.0
```

v12.17 calibration：

```text
repair_seed345_sketchdim24_rank5_b128_w3_5_10:
  source_calibration_pass = 1
  random_joint_false_positive_rows = 0
  control_joint_false_positive_rows = 0
  joint_hard_support_rows = 4

combined release labels:
  rows = 1728
  hard_joint = 12
  source_runs = repair_sketchdim24_rank5_b64_w3_5_10, repair_seed345_sketchdim24_rank5_b128_w3_5_10
```

v12.18 with this source:

```text
strict_visibility_wide_rows = 1728
strict_visibility_feature_columns = 36
required_split_failures = 0
support_concentrated = 1
strict_auc_joint_min = 0.013986013986013986
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
```

结论：

```text
batch128 修复了 source calibration false positives；
但 hard support 仍集中在 window=10。
```

#### 19.3.3 seed345 / b128 / w8,10,12

命令：

```bash
conda run -n kan python experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py \
  --probe-device auto \
  --probe-datasets MNIST,Fashion-MNIST,KMNIST \
  --probe-seeds 3,4,5 \
  --probe-splits 1 \
  --windows 8,10,12 \
  --actuator-windows 5 \
  --functional-batch-size 128 \
  --probe-train-size 512 \
  --probe-val-size 256 \
  --probe-test-size 256 \
  --probe-lr 0.01 \
  --sketch-dim 24 \
  --output-subspace-rank 5 \
  --ridge-lambda 0.001 \
  --actuator-budget-multiplier 8.0 \
  --target-logit-drift 0.04 \
  --safety-batch-factor 2 \
  --no-download \
  --out-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_sketchdim24_rank5_b128_w8_10_12
```

v12.16 输出：

```text
route = R2-EstimatorStillWeak
p1_pass = 0
p1_best_noise_abs_spearman = 0.036911089524983065
p1_best_reservoir_abs_spearman = 0.18909006151695243
p1_best_negative_release_precision = 0.0
p1_best_negative_release_recall = 0.0
```

v12.17 calibration：

```text
repair_seed345_sketchdim24_rank5_b128_w8_10_12:
  source_calibration_pass = 1
  random_joint_false_positive_rows = 0
  control_joint_false_positive_rows = 0
  joint_hard_support_rows = 8

combined release labels:
  rows = 1728
  hard_joint = 16
  hard_joint by window = {'10': 12, '12': 4}
```

v12.18：

```text
support_concentrated = 0
required_split_failures = 0
strict_auc_joint_min = 0.0
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
```

结论：

```text
window 扩展修复了 support concentration；
但 strict visibility 仍失败。
```

#### 19.3.4 seed345 / b128 / w10,12,15

命令：

```bash
conda run -n kan python experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py \
  --probe-device auto \
  --probe-datasets MNIST,Fashion-MNIST,KMNIST \
  --probe-seeds 3,4,5 \
  --probe-splits 1 \
  --windows 10,12,15 \
  --actuator-windows 5 \
  --functional-batch-size 128 \
  --probe-train-size 512 \
  --probe-val-size 256 \
  --probe-test-size 256 \
  --probe-lr 0.01 \
  --sketch-dim 24 \
  --output-subspace-rank 5 \
  --ridge-lambda 0.001 \
  --actuator-budget-multiplier 8.0 \
  --target-logit-drift 0.04 \
  --safety-batch-factor 2 \
  --no-download \
  --out-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_sketchdim24_rank5_b128_w10_12_15
```

v12.17 calibration：

```text
repair_seed345_sketchdim24_rank5_b128_w10_12_15:
  source_calibration_pass = 1
  random_joint_false_positive_rows = 0
  control_joint_false_positive_rows = 0
  joint_hard_support_rows = 16

combined with original repair source:
  rows = 1728
  hard_joint = 24
  hard_joint by window = {'10': 12, '12': 4, '15': 8}
  p2_visibility_repair_best_protocol = repair_coupling_spectrum_persistence/mean_noise_reservoir_rank
  p2_visibility_repair_best_auc_joint_min = 0.8432365023474179
  p2_visibility_repair_best_precision_joint_min = 0.0
  p2_visibility_repair_best_recall_joint_min = 0.0
```

v12.18 with role/time features:

```text
strict_visibility_feature_columns = 47
strict_auc_joint_min = 0.07535460992907801
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
support_concentrated = 0
required_split_failures = 0
```

composite repair：

```text
best_protocol = mean_noise_reservoir_rank
best_auc_joint_min = 0.39308176100628933
best_precision_joint_min = 0.0
best_recall_joint_min = 0.0
best_pass = 0
```

#### 19.3.5 matched source：seed012 / b128 / w10,12,15 + seed345 / b128 / w10,12,15

命令：

```bash
conda run -n kan python experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py \
  --probe-device auto \
  --probe-datasets MNIST,Fashion-MNIST,KMNIST \
  --probe-seeds 0,1,2 \
  --probe-splits 1 \
  --windows 10,12,15 \
  --actuator-windows 5 \
  --functional-batch-size 128 \
  --probe-train-size 512 \
  --probe-val-size 256 \
  --probe-test-size 256 \
  --probe-lr 0.01 \
  --sketch-dim 24 \
  --output-subspace-rank 5 \
  --ridge-lambda 0.001 \
  --actuator-budget-multiplier 8.0 \
  --target-logit-drift 0.04 \
  --safety-batch-factor 2 \
  --no-download \
  --out-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_sketchdim24_rank5_b128_w10_12_15
```

v12.16 seed012 输出：

```text
route = R2-EstimatorStillWeak
p1_pass = 0
p1_best_noise_abs_spearman = 0.18818886064881887
p1_best_reservoir_abs_spearman = 0.2605555287728709
p1_best_negative_release_precision = 0.0
p1_best_negative_release_recall = 0.0
p3_max_actuator_sketch_delta_fro = 0.008491670712828636
p3_max_actuator_projector_angle_deg = 2.2786431312561035
p3_best_noise_delta = -0.0037438571453094482
p3_best_reservoir_delta = -0.00587308406829834
```

v12.17 matched source command：

```bash
conda run -n kan python experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py \
  --run-id support_expand_seed012_seed345_b128_w10_12_15 \
  --out-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/support_expand_seed012_seed345_b128_w10_12_15 \
  --source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_sketchdim24_rank5_b128_w10_12_15 \
  --source-run results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed345_sketchdim24_rank5_b128_w10_12_15
```

v12.17 matched source 结果：

```text
source_calibration_pass:
  repair_seed012_sketchdim24_rank5_b128_w10_12_15 = 1
  repair_seed345_sketchdim24_rank5_b128_w10_12_15 = 1

random_joint_false_positive_rows:
  both = 0

control_joint_false_positive_rows:
  both = 0

release rows = 1728
hard_joint = 28
hard_joint by source = {
  repair_seed012_sketchdim24_rank5_b128_w10_12_15: 12,
  repair_seed345_sketchdim24_rank5_b128_w10_12_15: 16
}
hard_joint by window = {'10': 8, '12': 12, '15': 8}
hard_joint by method = {
  U5-OrthogonalCotangentVJPEnsemble: 8,
  U6-RoleConditionedPrimitiveActuator: 8,
  U7-ProjectionQuadRoleActuator: 8,
  U9-MixedLowRankActuatorBasis: 4
}

p2_visibility_repair_best_protocol = repair_coupling_spectrum_persistence/mean_noise_reservoir_rank
p2_visibility_repair_best_auc_joint_min = 0.7844852941176471
p2_visibility_repair_best_precision_joint_min = 0.0
p2_visibility_repair_best_recall_joint_min = 0.0
p4_open = 0
```

### 19.4 最终 v12.18 主审计

命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py \
  experiments/run_v1218_b320_label_free_ablation.py

conda run -n kan python experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py \
  --run-id official_from_v1217_v1216_artifacts_support_expand_seed012_seed345_b128_w10_12_15_role_time_composite \
  --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/support_expand_seed012_seed345_b128_w10_12_15 \
  --v1216-repair-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_seed012_sketchdim24_rank5_b128_w10_12_15
```

最终 route：

```text
route = R2-B320LabelInitDependenceDetected
fail_reason = current_B320_uses_label_informed_trainprobe_init;label_free_anchor_budget_protocol_not_official;label_free_anchor_budget_A0_does_not_reproduce_locked_anchor;label_free_anchor_budget_LineC_nontearing_failed;strict_loss_agnostic_observables_not_visible
promotion_allowed = 0
p4_open = 0
```

最终 strict visibility：

```text
strict_visibility_wide_rows = 1728
strict_visibility_feature_rows = 81216
strict_visibility_feature_columns = 47
hard_joint_support_rows = 28
support_concentrated = 0
strict_visibility_required_split_failures = 0
strict_auc_joint_min = 0.05660377358490566
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate
```

最终 support expansion：

```text
line_t_support_expansion_pass = 1
line_t_support_expansion_blocker = smoke_source_not_official_independent_source

usable calibrated sources:
  repair_seed012_sketchdim24_rank5_b128_w10_12_15
  repair_seed345_sketchdim24_rank5_b128_w10_12_15
```

最终 family ablation：

```text
line_t_family_ablation_best_set = only_augmentation_consistency_proxy
line_t_family_ablation_best_auc_joint_min = 0.5
line_t_family_ablation_best_visibility_pass = 0
```

最终 composite repair：

```text
line_t_composite_repair_best_protocol = min_noise_reservoir_rank
line_t_composite_repair_best_auc_joint_min = 0.19811320754716982
line_t_composite_repair_best_precision_joint_min = 0.0
line_t_composite_repair_best_recall_joint_min = 0.0
line_t_composite_repair_best_pass = 0
```

最差 split 定位：

```text
mean_noise_reservoir_rank:
  worst = leave-method-out / U9-MixedLowRankActuatorBasis
  AUC_joint = 0.12794811320754718
  precision = 0.0
  recall = 0.0
  positive_rows = 4
  hits = 0

min_noise_reservoir_rank:
  worst = leave-method-out / U9-MixedLowRankActuatorBasis
  AUC_joint = 0.19811320754716982
  precision = 0.0
  recall = 0.0
  positive_rows = 4
  hits = 0
```

### 19.5 最新 zip

zip 位置：

```text
/home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
```

zip 检查结果：

```text
zip_sha256 = eb970193cf11173e8a8adf450e7cdca112666351f7af79118909cf65ab4d1d30
zip_entries = 67
hash_manifest_entries = 42
zip_has code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py = True
zip_has review_artifacts/v1218_strict_visibility_composite_repair.csv = True
zip_has review_artifacts/v1218_line_t_support_expansion_audit.csv = True
```

### 19.6 本轮最终校验

命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py \
  experiments/run_v1218_b320_label_free_ablation.py

python - <<'PY'
from pathlib import Path
paths=[
Path('experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py'),
Path('experiments/run_v1218_b320_label_free_ablation.py'),
Path('docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional_执行日志.md'),
Path('docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional_实验结果复盘.md'),
]
failed=[]
for p in paths:
    for i,line in enumerate(p.read_text().splitlines(),1):
        if line.rstrip()!=line:
            failed.append((str(p),i))
print('trailing_whitespace_count',len(failed))
for item in failed[:20]: print(item)
PY
```

校验结果：

```text
py_compile_exit = 0
trailing_whitespace_count = 0
```

两个日志位置：

```text
docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional_执行日志.md
docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional_实验结果复盘.md
```

## 16. 继续推进：Line T strict feature engineering repair

### 16.1 继续推进原因

上一轮 `output-subspace drift repair` 后，Line T 仍然失败：

```text
strict_auc_joint_min = 0.3503521126760563
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate;support_concentrated;required_leaveout_not_runnable
```

按计划中“strict loss-agnostic observable 不可见时，继续扩展 random cotangent / augmentation consistency / output subspace drift 等 loss-agnostic 特征”的方向，本轮继续做 Line T 的特征工程修复。

注意：本轮只使用已有 v12.16 strict loss-agnostic source columns 派生特征，不使用 label / CE / permuted label / dataset name 作为 feature，不降低 gate，不把 post-hoc 结果用于 promotion。

### 16.2 修改的代码

修改文件：

```text
experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

修改内容：

```text
1. 在 REQUIRED_ARTIFACTS 中加入：
   v1218_strict_visibility_feature_engineering_audit.csv

2. 扩展 strict_v1216_feature_specs()，从 18 个 raw strict features 扩展到 36 个 features。

3. 新增 18 个 derived strict features：
   basis_occupancy_derived:
     rank_balance
     rank_total_log

   logit_free_spectrum_proxy_derived:
     signal_mass_proxy
     eigen_reservoir_interaction

   projector_geometry_derived:
     projector_stability_ratio
     projector_stability_balance
     dissipation_log

   random_cotangent_logit_jacobian_ensemble:
     cotangent_energy_total_log
     cotangent_energy_ratio
     cotangent_energy_balance
     cotangent_instability_share

   augmentation_consistency_derived:
     augmentation_cotangent_balance
     augmentation_projector_balance

   output_subspace_drift_derived:
     target_norm_ratio
     target_norm_total_log
     pred_joint_release_proxy
     pred_noise_reservoir_balance

   sketch_geometry_derived:
     sketch_work_log

4. 新增 strict_feature_value(source, spec)，统一计算 raw/derived feature_value。

5. 新增 write_strict_feature_engineering_audit(...)，输出每个 feature 的 source_column / formula / input_columns / derived / label 使用审计字段。

6. build_strict_v1216_long_rows(...) 改为调用 strict_feature_value(...)，避免 derived feature 读不到 source_column 时全部落成 0。

7. 为 feature-engineering audit 增加 derived 字段：
   derived=0 表示 raw source column
   derived=1 表示 formula-derived feature
```

本轮没有改动 gate 阈值、promotion 判定、P4 判定，也没有把 label/CE 信息引入 feature matrix。

### 16.3 编译与主审计重跑

命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py \
  experiments/run_v1218_b320_label_free_ablation.py

conda run -n kan python experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py \
  --run-id official_from_v1217_v1216_artifacts \
  --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts \
  --v1216-repair-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_sketchdim24_rank5_b64_w3_5_10
```

主审计输出：

```text
out_dir = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts
route = R2-B320LabelInitDependenceDetected
p4_open = 0
promotion_allowed = 0
```

### 16.4 产物读取命令

命令：

```bash
python - <<'PY'
import json, csv, zipfile, hashlib
from pathlib import Path
base=Path('results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts')
route=json.loads((base/'v1218_route_decision.json').read_text())
keys=['route','fail_reason','promotion_allowed','p4_open','strict_visibility_feature_rows','strict_visibility_feature_columns','strict_auc_joint_min','strict_precision_joint_min','strict_recall_joint_min','strict_visibility_pass','strict_visibility_blocker','line_t_family_ablation_rows','line_t_family_ablation_best_set','line_t_family_ablation_best_auc_joint_min','line_t_family_ablation_best_visibility_pass','code_review_packet_zip','code_review_packet_zip_sha256','hash_manifest_entries']
for k in keys:
    print(f'{k}={route.get(k)}')
with (base/'v1218_strict_visibility_scores.csv').open(newline='') as f:
    print(list(csv.DictReader(f))[0])
with (base/'v1218_strict_visibility_family_ablation.csv').open(newline='') as f:
    fam=list(csv.DictReader(f))
for r in sorted(fam, key=lambda r: float(r['AUC_joint_min']), reverse=True)[:12]:
    print(r['feature_set_name'], r['feature_columns'], r['AUC_joint_min'], r['AUC_joint_max'], r['precision_at_k_joint_min'], r['recall_at_k_joint_min'], r['visibility_pass'])
with (base/'v1218_strict_visibility_feature_engineering_audit.csv').open(newline='') as f:
    audit=list(csv.DictReader(f))
print('feature_audit_rows',len(audit))
print('derived_count',sum(1 for r in audit if r.get('derived')=='1'))
print('raw_count',sum(1 for r in audit if r.get('derived')=='0'))
print('uses_label_sum',sum(int(r['uses_label_for_feature']) for r in audit))
print('uses_ce_sum',sum(int(r['uses_ce_for_feature']) for r in audit))
print('selection_sum',sum(int(r['selection_for_promotion']) for r in audit))
zip_path=base/'v1218_code_review_packet.zip'
print('zip_path',zip_path.resolve())
print('zip_sha256',hashlib.sha256(zip_path.read_bytes()).hexdigest())
with zipfile.ZipFile(zip_path) as z:
    names=z.namelist()
print('zip_entries',len(names))
for n in names:
    if 'strict_visibility_feature_engineering_audit' in n or n.endswith('run_v1218_b320_codeaudit_lossagnostic_functional.py'):
        print('zip_contains',n)
PY
```

### 16.5 Line T feature-engineering 结果

route 关键字段：

```text
route = R2-B320LabelInitDependenceDetected
fail_reason = current_B320_uses_label_informed_trainprobe_init;label_free_anchor_budget_protocol_not_official;label_free_anchor_budget_A0_does_not_reproduce_locked_anchor;label_free_anchor_budget_LineC_nontearing_failed;strict_loss_agnostic_observables_not_visible
promotion_allowed = 0
p4_open = 0

strict_visibility_feature_rows = 31104
strict_visibility_feature_columns = 36
strict_auc_joint_min = 0.028169014084507043
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate;support_concentrated;required_leaveout_not_runnable

line_t_family_ablation_rows = 30
line_t_family_ablation_best_set = only_augmentation_consistency_derived
line_t_family_ablation_best_auc_joint_min = 0.5
line_t_family_ablation_best_visibility_pass = 0
```

`v1218_strict_visibility_scores.csv`：

```text
rows = 864
feature_columns = 36
long_feature_rows = 31104
AUC_noise_min = 0.4630281690140845
AUC_reservoir_min = 0.4299107142857143
AUC_joint_min = 0.028169014084507043
AUC_joint_max = 0.7884615384615384
precision_at_k_joint_min = 0.0
recall_at_k_joint_min = 0.0
required_split_failures = 1
support_concentrated = 1
visibility_exploratory_pass = 0
visibility_hard_pass = 0
visibility_robustness_pass = 0
visibility_pass = 0
```

family ablation AUC_joint_min 排名前 12：

```text
only_augmentation_consistency_derived:
  feature_columns = 2
  AUC_joint_min = 0.5
  AUC_joint_max = 0.5192307692307693
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

only_augmentation_consistency_proxy:
  feature_columns = 1
  AUC_joint_min = 0.5
  AUC_joint_max = 0.5
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

only_basis_occupancy_derived:
  feature_columns = 2
  AUC_joint_min = 0.5
  AUC_joint_max = 0.5
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

only_projector_geometry_derived:
  feature_columns = 3
  AUC_joint_min = 0.5
  AUC_joint_max = 0.9615384615384616
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

only_sketch_geometry:
  feature_columns = 2
  AUC_joint_min = 0.5
  AUC_joint_max = 0.5
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

only_sketch_geometry_derived:
  feature_columns = 1
  AUC_joint_min = 0.5
  AUC_joint_max = 0.5
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

only_output_subspace_drift:
  feature_columns = 5
  AUC_joint_min = 0.47836538461538464
  AUC_joint_max = 0.8230633802816901
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

only_projector_geometry:
  feature_columns = 3
  AUC_joint_min = 0.46153846153846156
  AUC_joint_max = 0.9295774647887324
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

only_output_subspace_drift_derived:
  feature_columns = 4
  AUC_joint_min = 0.41346153846153844
  AUC_joint_max = 0.9128521126760564
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

only_random_cotangent_logit_jacobian_ensemble:
  feature_columns = 4
  AUC_joint_min = 0.2692307692307692
  AUC_joint_max = 0.8380281690140845
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

only_logit_free_spectrum_proxy_derived:
  feature_columns = 2
  AUC_joint_min = 0.22535211267605634
  AUC_joint_max = 0.5384615384615384
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

only_logit_free_spectrum_proxy:
  feature_columns = 2
  AUC_joint_min = 0.19718309859154928
  AUC_joint_max = 0.5384615384615384
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0
```

feature-engineering audit：

```text
feature_audit_rows = 36
derived_count = 18
raw_count = 18
uses_label_sum = 0
uses_ce_sum = 0
selection_sum = 0
```

### 16.6 最新 zip / manifest

zip 位置：

```text
/home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
```

zip 检查结果：

```text
zip_sha256 = c670793aeb05646f698bf7f5380078211ebca78675a129c866971ce5fd3b5843
zip_entries = 66
hash_manifest_entries = 41
zip_contains = code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
zip_contains = review_artifacts/v1218_strict_visibility_feature_engineering_audit.csv
```

route JSON 中的 zip 字段：

```text
code_review_packet_zip = results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
code_review_packet_zip_sha256 = c670793aeb05646f698bf7f5380078211ebca78675a129c866971ce5fd3b5843
hash_manifest_entries = 41
```

### 16.7 本轮结论

```text
1. 本轮已按计划继续扩展 strict loss-agnostic Line T 特征，feature_columns 从 18 增至 36。
2. 新增 18 个 derived features 后，全量 AUC_joint_min 从上一轮 0.3503521126760563 降到 0.028169014084507043。
3. family ablation 的最佳 AUC_joint_min 只有 0.5，且 precision_at_k_joint_min / recall_at_k_joint_min 仍为 0.0。
4. support_concentrated=1 和 required_leaveout_not_runnable 仍存在，说明当前 strict source 仍不足以支持可推广 visibility 结论。
5. 因此 Line T 仍不通过，Line I / P4 仍不能打开，promotion_allowed 仍为 0。
6. 继续在同一批 strict source 上追加代数派生特征，已经没有证据能解决 support 集中和 leaveout 不可运行问题；下一步需要新的独立 strict source / support expansion，而不是继续用同一张表做更多 post-hoc 派生。
```

### 16.8 最终校验

命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py \
  experiments/run_v1218_b320_label_free_ablation.py

python - <<'PY'
from pathlib import Path
paths=[
Path('experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py'),
Path('experiments/run_v1218_b320_label_free_ablation.py'),
Path('docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional_执行日志.md'),
Path('docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional_实验结果复盘.md'),
]
failed=[]
for p in paths:
    for i,line in enumerate(p.read_text().splitlines(),1):
        if line.rstrip()!=line:
            failed.append((str(p),i))
print('trailing_whitespace_count',len(failed))
for item in failed[:20]: print(item)
PY

python - <<'PY'
import json, csv, zipfile, hashlib
from pathlib import Path
base=Path('results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts')
route=json.loads((base/'v1218_route_decision.json').read_text())
print('route',route['route'])
print('p4_open',route['p4_open'])
print('promotion_allowed',route['promotion_allowed'])
print('strict_visibility_feature_columns',route['strict_visibility_feature_columns'])
print('strict_auc_joint_min',route['strict_auc_joint_min'])
print('strict_precision_joint_min',route['strict_precision_joint_min'])
print('strict_recall_joint_min',route['strict_recall_joint_min'])
print('code_review_packet_zip_sha256',route['code_review_packet_zip_sha256'])
with (base/'v1218_strict_visibility_feature_engineering_audit.csv').open(newline='') as f:
    rows=list(csv.DictReader(f))
print('feature_engineering_audit_rows',len(rows))
print('derived_count',sum(1 for r in rows if r.get('derived')=='1'))
zip_path=base/'v1218_code_review_packet.zip'
print('zip_sha256_actual',hashlib.sha256(zip_path.read_bytes()).hexdigest())
with zipfile.ZipFile(zip_path) as z:
    names=set(z.namelist())
print('zip_entries',len(names))
for n in ['code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py','review_artifacts/v1218_strict_visibility_feature_engineering_audit.csv']:
    print('zip_has',n,n in names)
PY
```

输出：

```text
py_compile_exit = 0
trailing_whitespace_count = 0

route = R2-B320LabelInitDependenceDetected
p4_open = 0
promotion_allowed = 0
strict_visibility_feature_columns = 36
strict_auc_joint_min = 0.028169014084507043
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
code_review_packet_zip_sha256 = c670793aeb05646f698bf7f5380078211ebca78675a129c866971ce5fd3b5843
feature_engineering_audit_rows = 36
derived_count = 18
zip_sha256_actual = c670793aeb05646f698bf7f5380078211ebca78675a129c866971ce5fd3b5843
zip_entries = 66
zip_has code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py = True
zip_has review_artifacts/v1218_strict_visibility_feature_engineering_audit.csv = True
```

## 14. 继续推进：Line A label-free reservoir/direct 修复 A6-A14

用户指出不能在未完成时退出后，继续推进 v12.18。此前最新阻塞为：

```text
route = R2-B320LabelInitDependenceDetected
blockers =
  current_B320_uses_label_informed_trainprobe_init
  label_free_anchor_budget_protocol_not_official
  label_free_anchor_budget_A0_does_not_reproduce_locked_anchor
  label_free_anchor_budget_LineC_nontearing_failed
  strict_loss_agnostic_observables_not_visible
```

其中 Line A 的可修方向集中在：

```text
1. A5-orthogonalP-labelFree task-side 已接近可用；
2. 但 LineC_nontearing_all_pass=0；
3. A5 主要失败项是 RealSignalReservoirRatio_above_mlp_plus_0.02，且多数 row 还有 NoiseSignalLeak_above_mlp_plus_0.02。
```

因此继续做两组不读 label / 不读 CE 的修复候选：

```text
A6-A10: 试图降低 / bound quadratic-reservoir 分支。
A11-A14: 试图增强 label-free direct branch，降低 reservoir overuse。
```

### 14.1 代码修改

修改文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

`run_v1218_b320_label_free_ablation.py` 新增：

```text
replace_variant_token()

A6-orthogonalP-active64-labelFree:
  init_variant = stripped + _orthop_activep64

A7-orthogonalP-lowQuad-labelFree:
  init_variant = stripped with quadreadinit125 -> quadreadinit075 and identitytailquad030 -> identitytailquad020, then _orthop

A8-orthogonalP-boundQ-labelFree:
  init_variant = stripped + _orthop_boundq

A9-orthogonalP-lowQuad-boundQ-labelFree:
  init_variant = A7 variant + _orthop_boundq

A10-orthogonalP-strongLowQuad-boundQ-labelFree:
  init_variant = A9 variant with quadreadinit075 -> quadreadinit050 and _quadramp010

A11-orthogonalP-directRead125-labelFree:
  init_variant = stripped + _orthop_directreadinit125

A12-orthogonalP-identityAmp150-labelFree:
  init_variant = stripped + _orthop_identityamp150

A13-orthogonalP-lowQuad-directRead125-labelFree:
  init_variant = A7 variant + _orthop_directreadinit125

A14-orthogonalP-lowQuad-identityAmp150-labelFree:
  init_variant = A7 variant + _orthop_identityamp150
```

所有新增 candidate：

```text
uses_y_for_stats = 0
strict_label_free_init = 1
uses_label_for_direction = 0
uses_ce_vector_for_direction = 0
```

`run_v1218_b320_codeaudit_lossagnostic_functional.py` 新增 A6-A14 的 label-init audit 行，确保主审计和 zip 中可以追踪这些修复候选。

### 14.2 语法校验

命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py
```

结果：

```text
exit_code = 0
```

### 14.3 第一轮 A6-A10 recovered anchor-budget 运行

命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id label_free_anchor_budget_seed012_3x3_seedbase1211000_linec_b64_s32_d24_a6_a10_reservoir_repair \
  --artifact-prefix v1218_b320_label_free_anchor_budget \
  --result-stage V1218_B320_LABEL_FREE_ANCHOR_BUDGET_ABLATION \
  --summary-stage V1218_B320_LABEL_FREE_ANCHOR_BUDGET_SUMMARY \
  --route-stage V1218_B320_LABEL_FREE_ANCHOR_BUDGET_ROUTE \
  --result-scope recovered_v1211_v1212_anchor_budget_3x3_seedbase1211000_linec_b64_s32_d24_a6_a10_reservoir_repair \
  --smoke-not-official 0 \
  --official-training-result-available 0 \
  --protocol-note "recovered v12.11/v12.12/v12.14 anchor-like 3x3 budget; A6-A10 are label-free reservoir/noise repair candidates; no labels/CE used for direction; official claim depends on A0 task+LineC protocol reproduction" \
  --route-impact "keeps_R2 unless label-free task+LineC/protocol gates pass" \
  --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --device auto \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --seed-base 1211000 \
  --train-size 1024 \
  --val-size 512 \
  --test-size 512 \
  --epochs 3 \
  --batch-size 128 \
  --measure-linec 1 \
  --linec-batch-size 64 \
  --linec-sketch-batch-size 32 \
  --linec-sketch-dim 24
```

输出：

```text
rows = 117
summary_rows = 13
label_free_ablation_available_count = 10
best_label_free_candidate = A5-orthogonalP-labelFree
best_label_free_mean_delta_vs_A0 = -0.00390625
cpu_offload_used = 0
no_fake = 1
no_proxy = 1
```

A6-A10 结果概要：

```text
A5 pass 0/9, meanLeak=0.11128202494647768, meanReservoir=0.7062070800198449, meanCoupling=0.1824057448652821
A6 pass 0/9, meanLeak=0.11127888411283493, meanReservoir=0.7062235905064477, meanCoupling=0.18238124705765746
A7 pass 0/9, meanLeak=0.12404098734259605, meanReservoir=0.8426375654008653, meanCoupling=0.16871167121601718
A8 pass 0/9, meanLeak=0.13467421548234093, meanReservoir=0.8196700281567044, meanCoupling=0.15475522796713534
A9 pass 0/9, meanLeak=0.1485407774647077, meanReservoir=0.7652390135659112, meanCoupling=0.1550241495865462
A10 pass 0/9, meanLeak=0.1414537044862906, meanReservoir=0.7604076001379225, meanCoupling=0.16460591445048445
```

结论：

```text
A6-A10 没有解决 LineC non-tearing。
lowerQuad / boundQ 方向没有降低 RealSignalReservoirRatio 到 MLP+0.02 内，且 A7-A10 明显伤害任务指标。
```

### 14.4 第一轮后主审计重跑

命令：

```bash
conda run -n kan python experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py \
  --run-id official_from_v1217_v1216_artifacts \
  --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts \
  --v1216-repair-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_sketchdim24_rank5_b64_w3_5_10
```

输出：

```text
route = R2-B320LabelInitDependenceDetected
p4_open = 0
promotion_allowed = 0
```

第一轮 zip：

```text
zip_entries = 65
zip_sha256 = 61b1595d07cb5401d191dbc88f34fa9d0eeb57716ed8b09fa46bdbdd9ac465d4
```

该 zip 后续被 A11-A14 第二轮覆盖，保留为历史中间结果。

### 14.5 第二轮 A11-A14 direct-branch repair

继续分析后，认为仍有一个合理修复方向：A5/A6 的 direct branch 可能太弱，导致更新过度依赖 reservoir/quadratic 分支。于是补充 A11-A14。

语法校验命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py
```

结果：

```text
exit_code = 0
```

完整 A0-A14 运行命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id label_free_anchor_budget_seed012_3x3_seedbase1211000_linec_b64_s32_d24_a6_a14_reservoir_direct_repair \
  --artifact-prefix v1218_b320_label_free_anchor_budget \
  --result-stage V1218_B320_LABEL_FREE_ANCHOR_BUDGET_ABLATION \
  --summary-stage V1218_B320_LABEL_FREE_ANCHOR_BUDGET_SUMMARY \
  --route-stage V1218_B320_LABEL_FREE_ANCHOR_BUDGET_ROUTE \
  --result-scope recovered_v1211_v1212_anchor_budget_3x3_seedbase1211000_linec_b64_s32_d24_a6_a14_reservoir_direct_repair \
  --smoke-not-official 0 \
  --official-training-result-available 0 \
  --protocol-note "recovered v12.11/v12.12/v12.14 anchor-like 3x3 budget; A6-A10 lower/bound reservoir repair and A11-A14 label-free direct-branch repair candidates; no labels/CE used for direction; official claim depends on A0 task+LineC protocol reproduction" \
  --route-impact "keeps_R2 unless label-free task+LineC/protocol gates pass" \
  --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --device auto \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --seed-base 1211000 \
  --train-size 1024 \
  --val-size 512 \
  --test-size 512 \
  --epochs 3 \
  --batch-size 128 \
  --measure-linec 1 \
  --linec-batch-size 64 \
  --linec-sketch-batch-size 32 \
  --linec-sketch-dim 24
```

输出：

```text
rows = 153
summary_rows = 17
label_free_ablation_available_count = 14
best_label_free_candidate = A5-orthogonalP-labelFree
best_label_free_mean_delta_vs_A0 = -0.00390625
cpu_offload_used = 0
no_fake = 1
no_proxy = 1
```

### 14.6 第二轮最终 summary

`v1218_b320_label_free_anchor_budget_summary.csv` 关键字段：

```text
candidate_id                                            mean_delta_vs_mlp       worst_delta_vs_mlp   mean_delta_vs_A0       worst_delta_vs_A0   max_AUC_time_ratio_vs_mlp   linec_pass
A0-labelInit                                            0.022569444444444444    0.013671875                              0.9340476611616885        0/9
A1-noYForStats                                          0.012152777777777778   -0.015625          -0.010416666666666666  -0.048828125       0.6915868435257259        0/9
A2-randomP-labelFree                                    0.012152777777777778   -0.015625          -0.010416666666666666  -0.048828125       0.6915872276095383        0/9
A3-PCA-P-labelFree                                     -0.019097222222222224   -0.0703125         -0.041666666666666664  -0.083984375       0.8714860359825919        0/9
A4-lowfreqP-labelFree                                  -0.6603732638888888     -0.76953125        -0.6829427083333334    -0.806640625       42333.4644903965          0/9
A5-orthogonalP-labelFree                                0.018663194444444444   -0.005859375       -0.00390625            -0.0390625         0.6578371023388839        0/9
A6-orthogonalP-active64-labelFree                       0.018663194444444444   -0.005859375       -0.00390625            -0.0390625         0.657837163187841         0/9
A7-orthogonalP-lowQuad-labelFree                        0.007161458333333333   -0.01953125        -0.015407986111111112  -0.033203125       0.7214003793617216        0/9
A8-orthogonalP-boundQ-labelFree                        -0.05056423611111111    -0.09765625        -0.07313368055555555   -0.111328125       1.2658369264152824        0/9
A9-orthogonalP-lowQuad-boundQ-labelFree                -0.0724826388888889     -0.126953125       -0.09505208333333333   -0.140625          1.762533492131623         0/9
A10-orthogonalP-strongLowQuad-boundQ-labelFree         -0.0783420138888889     -0.134765625       -0.10091145833333333   -0.1484375         1.9856940546873132        0/9
A11-orthogonalP-directRead125-labelFree                 0.018229166666666668   -0.005859375       -0.004340277777777778  -0.0390625         0.6578555395729052        0/9
A12-orthogonalP-identityAmp150-labelFree                0.018446180555555556   -0.005859375       -0.004123263888888889  -0.0390625         0.6579779676747233        0/9
A13-orthogonalP-lowQuad-directRead125-labelFree         0.007378472222222222   -0.01953125        -0.015190972222222222  -0.033203125       0.7215783321369537        0/9
A14-orthogonalP-lowQuad-identityAmp150-labelFree        0.007161458333333333   -0.017578125       -0.015407986111111112  -0.03515625        0.7217404337588441        0/9
MLP-same-step-FLOP-AdamW                                0.0                     0.0                                      1.072374152799825         9/9
```

第二轮 Line C 细节：

```text
A5  pass 0/9, meanLeak=0.1112774374584357, meanReservoir=0.7062219613128238, meanCoupling=0.1823585640753439
A6  pass 0/9, meanLeak=0.11127393609947628, meanReservoir=0.7062293257978227, meanCoupling=0.18236011351399817
A7  pass 0/9, meanLeak=0.12404166244798237, meanReservoir=0.8426430688963996, meanCoupling=0.1687150209352995
A8  pass 0/9, meanLeak=0.13467423783408272, meanReservoir=0.8196694321102567, meanCoupling=0.15475378669706397
A9  pass 0/9, meanLeak=0.14854084410601193, meanReservoir=0.7652382585737441, meanCoupling=0.15501506107042073
A10 pass 0/9, meanLeak=0.14145335513684484, meanReservoir=0.7604071895281473, meanCoupling=0.1646045696908192
A11 pass 0/9, meanLeak=0.11095200065109465, meanReservoir=0.7053342296017541, meanCoupling=0.18260943509131536
A12 pass 0/9, meanLeak=0.11036044400599268, meanReservoir=0.7057378987471262, meanCoupling=0.18251400828949643
A13 pass 0/9, meanLeak=0.12404547590348455, meanReservoir=0.8424602614508735, meanCoupling=0.16899817181984755
A14 pass 0/9, meanLeak=0.12377550121810701, meanReservoir=0.8423047595553927, meanCoupling=0.16919102604683658
```

失败项计数：

```text
A5/A6/A11/A12:
  RealSignalReservoirRatio_above_mlp_plus_0.02 = 9/9
  NoiseSignalLeak_above_mlp_plus_0.02 = 7/9
  CouplingR2_below_mlp_minus_0.02 = 1/9

A7/A13/A14:
  RealSignalReservoirRatio_above_mlp_plus_0.02 = 9/9
  NoiseSignalLeak_above_mlp_plus_0.02 = 6/9
  CouplingR2_below_mlp_minus_0.02 = 2/9

A8/A10:
  RealSignalReservoirRatio_above_mlp_plus_0.02 = 9/9
  NoiseSignalLeak_above_mlp_plus_0.02 = 7/9
  CouplingR2_below_mlp_minus_0.02 = 2/9

A9:
  RealSignalReservoirRatio_above_mlp_plus_0.02 = 9/9
  NoiseSignalLeak_above_mlp_plus_0.02 = 7/9
  CouplingR2_below_mlp_minus_0.02 = 3/9
```

结论：

```text
1. A11/A12 direct-branch repair 对 meanLeak/meanReservoir 只有极小改善，但没有任何 row 通过 LineC non-tearing。
2. A7/A9/A10/A13/A14 lowerQuad 方向反而提高 reservoir ratio 或损伤任务指标。
3. A8/A9/A10 boundQ 方向没有降低噪声泄漏，且任务指标明显下降。
4. A5 仍是 best label-free candidate，但只能 task-side strong pass，不能关闭 LineC / R2。
```

### 14.7 最终主审计重跑

命令：

```bash
conda run -n kan python experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py \
  --run-id official_from_v1217_v1216_artifacts \
  --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts \
  --v1216-repair-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_sketchdim24_rank5_b64_w3_5_10
```

输出：

```text
route = R2-B320LabelInitDependenceDetected
p4_open = 0
promotion_allowed = 0
```

最终 route 关键字段：

```text
fail_reason = current_B320_uses_label_informed_trainprobe_init;label_free_anchor_budget_protocol_not_official;label_free_anchor_budget_A0_does_not_reproduce_locked_anchor;label_free_anchor_budget_LineC_nontearing_failed;strict_loss_agnostic_observables_not_visible

b320_label_free_anchor_budget_ablation_available_count = 14
b320_label_free_anchor_budget_best_candidate = A5-orthogonalP-labelFree
b320_label_free_anchor_budget_best_mean_delta_vs_A0 = -0.00390625
b320_label_free_anchor_budget_best_max_AUC_time_ratio_vs_mlp = 0.6578371023388839
b320_label_free_anchor_budget_task_strong_pass = 1
b320_label_free_anchor_budget_linec_nontearing_pass = 0

b320_label_free_anchor_budget_a0_gate_pass = 1
b320_label_free_anchor_budget_a0_task_reproduces_locked_anchor = 1
b320_label_free_anchor_budget_a0_linec_reproduces_locked_anchor = 0
b320_label_free_anchor_budget_a0_reproduces_locked_anchor = 0

strict_auc_joint_min = 0.3503521126760563
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate;support_concentrated;required_leaveout_not_runnable
```

Protocol audit：

```text
source = v1218_b320_label_free_anchor_budget_summary.csv
candidate_id = A0-labelInit
anchor_budget_A0_mean_delta_vs_mlp = 0.022569444444444444
anchor_budget_A0_worst_delta_vs_mlp = 0.013671875
anchor_budget_A0_max_AUC_time_ratio_vs_mlp = 0.9340476611616885
anchor_budget_A0_linec_nontearing_rows = 9
anchor_budget_A0_linec_nontearing_pass_rate = 0.0
anchor_budget_A0_linec_nontearing_all_pass = 0
a0_anchor_budget_gate_pass = 1
a0_anchor_budget_task_reproduces_locked_anchor = 1
a0_anchor_budget_linec_reproduces_locked_anchor = 0
a0_anchor_budget_reproduces_locked_anchor = 0
official_budget_claim_allowed = 0
blocker = recovered-anchor-budget A0 does not reproduce locked B320 task+LineC gates; do not treat label-free ablation as official replacement
```

### 14.8 最终 zip 与校验

zip 检查命令：

```bash
bsdtar -tf results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip | rg 'run_v1218_b320_label_free|label_free_anchor_budget_(ablation|summary|route)|label_init_audit|route_decision'
bsdtar -tf results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip | wc -l
sha256sum results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
```

输出：

```text
code/experiments/run_v1218_b320_label_free_ablation.py
review_artifacts/v1218_route_decision.json
review_artifacts/v1218_b320_label_init_audit.csv
review_artifacts/v1218_b320_label_free_anchor_budget_ablation.csv
review_artifacts/v1218_b320_label_free_anchor_budget_summary.csv
review_artifacts/v1218_b320_label_free_anchor_budget_route.json
zip_entries = 65
zip_sha256 = f118d37c084b9de3ff7687859e0cb605dac32fe92cc30a7f74181dd1f28d7bf9
hash_manifest_entries = 40
```

最终 zip 路径：

```text
/home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
```

最终校验：

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py
perl -ne 'print "$ARGV:$.:$_" if /[ \t]$/; close ARGV if eof' experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional_执行日志.md docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional_实验结果复盘.md
```

结果：

```text
py_compile exit_code = 0
trailing_whitespace_check output = empty
```

注：旧 zip hash `861a71...` 与 `61b159...` 只保留在前文历史章节中；本节结束时的可审计 zip hash 为 `f118d37c084b9de3ff7687859e0cb605dac32fe92cc30a7f74181dd1f28d7bf9`，后续 detailed audit 已在第 15 节覆盖为最新 zip。

## 15. 继续推进：LineC RealSignalReservoirRatio detailed audit

上一节虽然确认 A6-A14 全部失败，但仍需要解释 `RealSignalReservoirRatio` 为什么高。继续审查代码后确认：

```text
v1252._signal_reservoir_metrics:
  grad_sketch = CE gradient sketch over real labels, audit-only
  k_mat = grad_sketch @ grad_sketch.T
  p_sig = eigenspace covering 80% gradient-sketch mass
  p_res = I - p_sig

  r_real = CE_real - mean(CE_real)
  RealSignalReservoirRatio = ||p_res @ r_real||^2 / ||r_real||^2

  r_noise = CE_noise - mean(CE_noise)
  NoiseSignalLeak = ||p_sig @ r_noise||^2 / ||r_noise||^2
```

这说明仅看 ratio 不够，需要同时记录分子/分母与 signal eigenspace 结构。

### 15.1 代码修改

修改文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
```

新增：

```text
import torch.nn.functional as F
signal_reservoir_metrics_detailed()
```

新增 raw ablation CSV 列：

```text
linec_signal_mass_topk
linec_reservoir_fraction
linec_top_eigen_share
linec_dissipation_condition
linec_SNR_positive_fraction
linec_signal_top_count
linec_real_residual_energy
linec_real_total_energy
linec_noise_signal_energy
linec_noise_total_energy
```

这些列只用于 audit：

```text
linec_label_used_for_audit_only = 1
linec_label_used_for_direction = 0
linec_ce_vector_used_for_direction = 0
```

### 15.2 语法校验

命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_label_free_ablation.py experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

结果：

```text
exit_code = 0
```

### 15.3 detailed LineC 版 A0-A14 重跑

命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id label_free_anchor_budget_seed012_3x3_seedbase1211000_linec_b64_s32_d24_a6_a14_reservoir_direct_repair_detailed_linec \
  --artifact-prefix v1218_b320_label_free_anchor_budget \
  --result-stage V1218_B320_LABEL_FREE_ANCHOR_BUDGET_ABLATION \
  --summary-stage V1218_B320_LABEL_FREE_ANCHOR_BUDGET_SUMMARY \
  --route-stage V1218_B320_LABEL_FREE_ANCHOR_BUDGET_ROUTE \
  --result-scope recovered_v1211_v1212_anchor_budget_3x3_seedbase1211000_linec_b64_s32_d24_a6_a14_reservoir_direct_repair_detailed_linec \
  --smoke-not-official 0 \
  --official-training-result-available 0 \
  --protocol-note "recovered v12.11/v12.12/v12.14 anchor-like 3x3 budget; A6-A10 lower/bound reservoir repair and A11-A14 label-free direct-branch repair candidates; detailed LineC numerator/denominator audit columns added; no labels/CE used for direction; official claim depends on A0 task+LineC protocol reproduction" \
  --route-impact "keeps_R2 unless label-free task+LineC/protocol gates pass" \
  --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --device auto \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --seed-base 1211000 \
  --train-size 1024 \
  --val-size 512 \
  --test-size 512 \
  --epochs 3 \
  --batch-size 128 \
  --measure-linec 1 \
  --linec-batch-size 64 \
  --linec-sketch-batch-size 32 \
  --linec-sketch-dim 24
```

输出：

```text
rows = 153
summary_rows = 17
label_free_ablation_available_count = 14
best_label_free_candidate = A5-orthogonalP-labelFree
best_label_free_mean_delta_vs_A0 = -0.00390625
cpu_offload_used = 0
no_fake = 1
no_proxy = 1
```

最终 summary 关键变化：

```text
A0-labelInit:
  mean_delta_vs_mlp = 0.022569444444444444
  worst_delta_vs_mlp = 0.013671875
  max_AUC_time_ratio_vs_mlp = 0.9151941736560291
  LineC = 0/9

A5-orthogonalP-labelFree:
  mean_delta_vs_mlp = 0.018663194444444444
  worst_delta_vs_mlp = -0.005859375
  mean_delta_vs_A0 = -0.00390625
  worst_delta_vs_A0 = -0.0390625
  max_AUC_time_ratio_vs_mlp = 0.7776700385171795
  LineC = 0/9

A11-orthogonalP-directRead125-labelFree:
  mean_delta_vs_mlp = 0.018229166666666668
  worst_delta_vs_mlp = -0.005859375
  mean_delta_vs_A0 = -0.004340277777777778
  max_AUC_time_ratio_vs_mlp = 0.7775565950993564
  LineC = 0/9

A12-orthogonalP-identityAmp150-labelFree:
  mean_delta_vs_mlp = 0.018446180555555556
  worst_delta_vs_mlp = -0.005859375
  mean_delta_vs_A0 = -0.004123263888888889
  max_AUC_time_ratio_vs_mlp = 0.7780129834775749
  LineC = 0/9

MLP-same-step-FLOP-AdamW:
  mean_delta_vs_mlp = 0.0
  worst_delta_vs_mlp = 0.0
  max_AUC_time_ratio_vs_mlp = 1.0331432309011304
  LineC = 9/9
```

### 15.4 detailed LineC decomposition

均值统计：

```text
MLP-same-step-FLOP-AdamW:
  linec_signal_mass_topk = 0.8611922595236037
  linec_reservoir_fraction = 0.13880774047639635
  linec_top_eigen_share = 0.6349554161230723
  linec_signal_top_count = 2.3333333333333335
  linec_real_residual_energy = 23.124738832314808
  linec_real_total_energy = 95.83334681722853
  linec_noise_signal_energy = 55.435595217678284
  linec_noise_total_energy = 1015.8765869140625

A0-labelInit:
  linec_signal_mass_topk = 0.8587490717569987
  linec_reservoir_fraction = 0.1412509282430013
  linec_top_eigen_share = 0.42259319954448277
  linec_signal_top_count = 4.111111111111111
  linec_real_residual_energy = 0.8429780337545607
  linec_real_total_energy = 1.7884223577049043
  linec_noise_signal_energy = 97.8177138434516
  linec_noise_total_energy = 711.0140821668837

A5-orthogonalP-labelFree:
  linec_signal_mass_topk = 0.8355417980088128
  linec_reservoir_fraction = 0.16445820199118721
  linec_top_eigen_share = 0.3521610846122106
  linec_signal_top_count = 4.444444444444445
  linec_real_residual_energy = 5.526151098724869
  linec_real_total_energy = 9.12627146144708
  linec_noise_signal_energy = 120.40638648139105
  linec_noise_total_energy = 1104.3729315863716

A11-orthogonalP-directRead125-labelFree:
  linec_signal_mass_topk = 0.8357143269644843
  linec_reservoir_fraction = 0.16428567303551567
  linec_top_eigen_share = 0.35207471748193103
  linec_signal_top_count = 4.444444444444445
  linec_real_residual_energy = 5.513146714203888
  linec_real_total_energy = 9.123155733777416
  linec_noise_signal_energy = 120.28983052571614
  linec_noise_total_energy = 1105.7046237521702

A12-orthogonalP-identityAmp150-labelFree:
  linec_signal_mass_topk = 0.8357236054208543
  linec_reservoir_fraction = 0.16427639457914564
  linec_top_eigen_share = 0.35213640166653526
  linec_signal_top_count = 4.444444444444445
  linec_real_residual_energy = 5.511935541199313
  linec_real_total_energy = 9.124200887564156
  linec_noise_signal_energy = 119.85119077894423
  linec_noise_total_energy = 1106.1815728081597

A7-orthogonalP-lowQuad-labelFree:
  linec_signal_mass_topk = 0.8336505757437812
  linec_reservoir_fraction = 0.1663494242562188
  linec_real_residual_energy = 3.067094405492147
  linec_real_total_energy = 3.628629156284862

A8-orthogonalP-boundQ-labelFree:
  linec_signal_mass_topk = 0.8415789604187012
  linec_reservoir_fraction = 0.15842103958129883
  linec_real_residual_energy = 5.990347385406494
  linec_real_total_energy = 7.342595100402832
```

解释：

```text
1. A5 的 reservoir_fraction=0.16445820199118721，比 MLP 的 0.13880774047639635 只高约 0.02565，不能单独解释 RealSignalReservoirRatio 大幅失败。
2. A5 的 real_total_energy=9.12627146144708，远低于 MLP 的 95.83334681722853；但 A5 real_residual_energy=5.526151098724869，并没有按同等比例降低。
3. 因此 A5 的失败更像是 CE-real 残差能量被压小后，剩余残差相对集中在 p_res reservoir 方向，而不是 eigenspace reservoir_fraction 大幅变宽。
4. A11/A12 direct repair 几乎不改变 signal_mass_topk / reservoir_fraction / residual_energy，解释了为什么它们只带来极小改善。
5. A7/A8 说明 lowerQuad/boundQ 不是正确方向：它们没有改善相对 reservoir alignment，同时会伤任务或 noise。
6. NoiseSignalLeak 方面，A5 noise_signal_energy=120.40638648139105，高于 MLP 的 55.435595217678284，而 noise_total_energy 只从 MLP 的 1015.8765869140625 变到 1104.3729315863716；这说明 noise leak 不是单纯分母效应，确实有更多 noise residual 落入 p_sig。
```

### 15.5 最终主审计重跑

命令：

```bash
conda run -n kan python experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py \
  --run-id official_from_v1217_v1216_artifacts \
  --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts \
  --v1216-repair-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_sketchdim24_rank5_b64_w3_5_10
```

输出：

```text
route = R2-B320LabelInitDependenceDetected
p4_open = 0
promotion_allowed = 0
```

最终 route：

```text
fail_reason = current_B320_uses_label_informed_trainprobe_init;label_free_anchor_budget_protocol_not_official;label_free_anchor_budget_A0_does_not_reproduce_locked_anchor;label_free_anchor_budget_LineC_nontearing_failed;strict_loss_agnostic_observables_not_visible

b320_label_free_anchor_budget_ablation_available_count = 14
b320_label_free_anchor_budget_best_candidate = A5-orthogonalP-labelFree
b320_label_free_anchor_budget_best_mean_delta_vs_A0 = -0.00390625
b320_label_free_anchor_budget_best_max_AUC_time_ratio_vs_mlp = 0.7776700385171795
b320_label_free_anchor_budget_task_strong_pass = 1
b320_label_free_anchor_budget_linec_nontearing_pass = 0

b320_label_free_anchor_budget_a0_gate_pass = 1
b320_label_free_anchor_budget_a0_task_reproduces_locked_anchor = 1
b320_label_free_anchor_budget_a0_linec_reproduces_locked_anchor = 0
b320_label_free_anchor_budget_a0_reproduces_locked_anchor = 0

strict_auc_joint_min = 0.3503521126760563
strict_visibility_pass = 0
```

Protocol audit：

```text
anchor_budget_A0_mean_delta_vs_mlp = 0.022569444444444444
anchor_budget_A0_worst_delta_vs_mlp = 0.013671875
anchor_budget_A0_max_AUC_time_ratio_vs_mlp = 0.9151941736560291
anchor_budget_A0_linec_nontearing_all_pass = 0
a0_anchor_budget_gate_pass = 1
a0_anchor_budget_task_reproduces_locked_anchor = 1
a0_anchor_budget_linec_reproduces_locked_anchor = 0
a0_anchor_budget_reproduces_locked_anchor = 0
official_budget_claim_allowed = 0
```

### 15.6 最终 zip

检查命令：

```bash
bsdtar -tf results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip | rg 'run_v1218_b320_label_free|label_free_anchor_budget_(ablation|summary|route)|route_decision'
bsdtar -tf results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip | wc -l
sha256sum results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
```

输出：

```text
code/experiments/run_v1218_b320_label_free_ablation.py
review_artifacts/v1218_route_decision.json
review_artifacts/v1218_b320_label_free_anchor_budget_ablation.csv
review_artifacts/v1218_b320_label_free_anchor_budget_summary.csv
review_artifacts/v1218_b320_label_free_anchor_budget_route.json
zip_entries = 65
zip_sha256 = 82d438d4641b2d1ee820fa715c2cadf76243b8a2905f8be3c485dc1d0c0b7872
hash_manifest_entries = 40
```

最新 zip 路径仍为：

```text
/home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
```

### 15.7 最终校验

命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py
perl -ne 'print "$ARGV:$.:$_" if /[ \t]$/; close ARGV if eof' experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional_执行日志.md docs/DG-KAN_v12.18_B320CodeAudit_LossAgnosticFunctional_实验结果复盘.md
python - <<'PY'
import csv, json, pathlib, zipfile
root=pathlib.Path('results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts')
route=json.loads((root/'v1218_route_decision.json').read_text())
summary=list(csv.DictReader((root/'v1218_b320_label_free_anchor_budget_summary.csv').open()))
print('route', route['route'])
print('promotion_allowed', route['promotion_allowed'])
print('p4_open', route['p4_open'])
print('best', route['b320_label_free_anchor_budget_best_candidate'], route['b320_label_free_anchor_budget_best_mean_delta_vs_A0'])
print('zip_sha', route['code_review_packet_zip_sha256'])
print('summary_rows', len(summary))
with zipfile.ZipFile(root/'v1218_code_review_packet.zip') as zf:
    names=set(zf.namelist())
    print('zip_entries', len(names))
    for name in ['code/experiments/run_v1218_b320_label_free_ablation.py','review_artifacts/v1218_b320_label_free_anchor_budget_ablation.csv','review_artifacts/v1218_b320_label_free_anchor_budget_summary.csv','review_artifacts/v1218_route_decision.json']:
        print(name, name in names)
PY
```

结果：

```text
py_compile exit_code = 0
trailing_whitespace_check output = empty
route = R2-B320LabelInitDependenceDetected
promotion_allowed = 0
p4_open = 0
best = A5-orthogonalP-labelFree -0.00390625
zip_sha = 82d438d4641b2d1ee820fa715c2cadf76243b8a2905f8be3c485dc1d0c0b7872
summary_rows = 17
zip_entries = 65
code/experiments/run_v1218_b320_label_free_ablation.py = True
review_artifacts/v1218_b320_label_free_anchor_budget_ablation.csv = True
review_artifacts/v1218_b320_label_free_anchor_budget_summary.csv = True
review_artifacts/v1218_route_decision.json = True
```

补充说明：

```text
系统没有 unzip 命令，因此 zip 内容检查使用 bsdtar -tf。
```

### 12.7 本轮修复后的 blocker

已经执行并修复：

```text
1. smoke-only A1-A5 不足以回答 official-budget Line A 的问题，因此恢复了 v12.11/v12.12 anchor budget 并跑 3x3。
2. exact official protocol 没有完整落在 v12.17 route artifact 中，因此增加 A0 protocol audit，防止把 recovered protocol 伪装成 official。
3. 计划中的 A6 MLP same-step/FLOP control 没有显式落盘，因此补充 A6 实跑 control。
4. b32/sketchdim8 LineC 可能与 v12.16 repair source 不匹配，因此按 repair_sketchdim24_rank5_b64_w3_5_10 方向重跑 b64/sketch_batch32/sketchdim24；结论仍未改变。
```

仍未关闭：

```text
1. A5-orthogonalP-labelFree 的 task strong pass = 1，但 LineC_nontearing_all_pass = 0。
2. A0 recovered anchor-budget task gate 复现，但 A0 LineC reproduction = 0，因此 recovered protocol 不能作为 locked-anchor official replacement。
3. 主 route 仍为 R2，不允许 promotion / P4。
```

## 13. 继续推进：Line T output-subspace drift repair 与 family ablation

继续对照计划 8.7 执行 Line T 推荐修复方向。

计划要求：

```text
如果 label-free features AUC 低：
  新增随机 cotangent logit-Jacobian ensemble；
  新增 augmentation consistency；
  新增 output-subspace drift；
  不允许引入 CE gradient sketch。
```

检查发现：

```text
1. 当前 strict feature specs 已包含 random_cotangent_logit_jacobian。
2. 当前 strict feature specs 已包含 augmentation_consistency_proxy，但只是 augmentation_used 二值代理。
3. 当前 strict feature specs 未包含 v12.16 artifact 中已经落盘的 output-subspace drift / predicted drift columns。
4. ridge_predict 已经用训练折统计量做 z-score 标准化，因此没有重复修改 scaling。
```

### 13.1 修改代码

文件：

```text
experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
```

修改内容：

```text
1. strict_v1216_feature_specs 新增 output_subspace_drift family：
   - noise_target_norm
   - reservoir_target_norm
   - pred_noise_delta
   - pred_reservoir_delta
   - pred_coupling_delta

2. 新增 v1218_strict_visibility_family_ablation.csv。

3. 新增 write_strict_visibility_family_ablation：
   - all_current
   - baseline_without_output_subspace_drift
   - only_<family>
   - all_except_<family>

4. family ablation 明确写入：
   uses_label_for_feature=0
   uses_ce_for_feature=0
   selection_for_promotion=0
   reason=post-hoc Line T repair diagnostic; not used to lower gates or claim promotion
```

### 13.2 主审计重跑命令

```bash
conda run -n kan python -m py_compile experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py && \
conda run -n kan python experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py \
  --run-id official_from_v1217_v1216_artifacts \
  --out-dir results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts \
  --v1217-dir results/v12_17_2_b320locked_lossagnostic_target_visibility_functional_geometry/official_from_v1216_artifacts \
  --v1216-repair-dir results/v12_16_b320locked_explicit_signal_reservoir_functional/repair_sketchdim24_rank5_b64_w3_5_10
```

输出：

```text
route = R2-B320LabelInitDependenceDetected
p4_open = 0
promotion_allowed = 0
```

### 13.3 Line T 结果

route 关键字段：

```text
strict_visibility_feature_rows = 15552
strict_visibility_feature_columns = 18
strict_auc_joint_min = 0.3503521126760563
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate;support_concentrated;required_leaveout_not_runnable
line_t_family_ablation_rows = 16
line_t_family_ablation_best_set = baseline_without_output_subspace_drift
line_t_family_ablation_best_auc_joint_min = 0.5
```

family ablation 关键结果：

```text
all_current:
  feature_columns = 18
  AUC_joint_min = 0.3503521126760563
  AUC_joint_max = 0.5769230769230769
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

baseline_without_output_subspace_drift:
  feature_columns = 13
  AUC_joint_min = 0.5
  AUC_joint_max = 0.9436619718309859
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

only_output_subspace_drift:
  feature_columns = 5
  AUC_joint_min = 0.47836538461538464
  AUC_joint_max = 0.8230633802816901
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0

all_except_projector_geometry:
  feature_columns = 15
  AUC_joint_min = 0.5
  AUC_joint_max = 0.9471830985915493
  precision_at_k_joint_min = 0.0
  recall_at_k_joint_min = 0.0
  visibility_pass = 0
```

结论：

```text
1. 按计划新增 output-subspace drift 后没有提高 strict visibility；全量加入反而让 min AUC 从 0.5 降到 0.3503521126760563。
2. 只用 output-subspace drift 的 min AUC=0.47836538461538464，仍低于 0.65 exploratory gate。
3. 所有 family ablation 的 precision/recall min 都是 0.0。
4. 因此不能打开 Line I / P4，也不能降低 release threshold。
```

### 13.4 最终 zip

检查命令：

```bash
bsdtar -tf results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip | rg 'family_ablation|run_v1218_b320_codeaudit|route_decision'
bsdtar -tf results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip | wc -l
sha256sum results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
```

输出：

```text
zip_has code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py
zip_has review_artifacts/v1218_route_decision.json
zip_has review_artifacts/v1218_strict_visibility_family_ablation.csv
zip_entries = 65
zip_sha256 = 861a71f82db20fdb635612333cd1bee046ac3ed9e3dfa18ce29dbf96eb6a0dea
hash_manifest_entries = 40
```

## 18. 文件末尾最终权威状态汇总

本执行日志存在历史追加顺序不完全递增的问题；文件末尾以本节为最终权威状态，旧的 `zip_entries = 65` / `861a...` 记录只是上一轮中间产物。

最新代码修改已整合进 zip：

```text
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_sha256 = c670793aeb05646f698bf7f5380078211ebca78675a129c866971ce5fd3b5843
zip_entries = 66
hash_manifest_entries = 41
zip_has code/experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py = True
zip_has review_artifacts/v1218_strict_visibility_feature_engineering_audit.csv = True
```

最终 route：

```text
route = R2-B320LabelInitDependenceDetected
p4_open = 0
promotion_allowed = 0
strict_visibility_feature_columns = 36
strict_auc_joint_min = 0.028169014084507043
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate;support_concentrated;required_leaveout_not_runnable
```

最终 Line T feature-engineering audit：

```text
feature_engineering_audit_rows = 36
raw_count = 18
derived_count = 18
uses_label_sum = 0
uses_ce_sum = 0
selection_sum = 0
```

最终校验：

```text
conda run -n kan python -m py_compile experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py
py_compile_exit = 0
trailing_whitespace_count = 0
```

## 21. EOF 最终权威索引

本文件历史追加顺序存在中段插入。以本 EOF 索引和第 20 节为最新权威状态；第 18 节中的 `c670...` / `zip_entries=66` / `strict_visibility_feature_columns=36` 是上一阶段中间产物。

最新产物：

```text
result_dir = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_sha256 = eb970193cf11173e8a8adf450e7cdca112666351f7af79118909cf65ab4d1d30
zip_entries = 67
hash_manifest_entries = 42
```

最新 route：

```text
run_id = official_from_v1217_v1216_artifacts_support_expand_seed012_seed345_b128_w10_12_15_role_time_composite
route = R2-B320LabelInitDependenceDetected
p4_open = 0
promotion_allowed = 0
strict_visibility_feature_columns = 47
strict_visibility_wide_rows = 1728
hard_joint_support_rows = 28
support_concentrated = 0
strict_visibility_required_split_failures = 0
strict_auc_joint_min = 0.05660377358490566
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
strict_visibility_pass = 0
strict_visibility_blocker = strict_auc_joint_below_gate;strict_precision_recall_below_gate
line_t_composite_repair_best_protocol = min_noise_reservoir_rank
line_t_composite_repair_best_auc_joint_min = 0.19811320754716982
line_t_composite_repair_best_precision_joint_min = 0.0
line_t_composite_repair_best_recall_joint_min = 0.0
line_t_composite_repair_best_pass = 0
```

最新验证：

```text
conda run -n kan python -m py_compile experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py
py_compile_exit = 0
trailing_whitespace_count = 0
```

## 23. EOF 最新权威索引：context-rank repair 后

本文件历史追加顺序存在中段插入。以本节为最新权威状态；第 21 节中的 `eb970...` / `zip_entries=67` 是上一阶段中间产物，第 22 节记录了完整继续推进过程。

最新产物：

```text
result_dir = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_sha256 = b4ccb50b88d657ee1b0b9d036e75da590288da17979be6e646318c01ff8ae10b
zip_entries = 68
hash_manifest_entries = 43
```

最新 route：

```text
run_id = official_from_v1217_v1216_artifacts_support_expand_seed012_seed345_b128_w10_12_15_context_rank_repair
route = R2-B320LabelInitDependenceDetected
fail_reason = current_B320_uses_label_informed_trainprobe_init;label_free_anchor_budget_protocol_not_official;label_free_anchor_budget_A0_does_not_reproduce_locked_anchor;label_free_anchor_budget_LineC_nontearing_failed;strict_loss_agnostic_observables_not_visible
p4_open = 0
promotion_allowed = 0
strict_visibility_pass = 0
strict_auc_joint_min = 0.05660377358490566
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
line_t_context_rank_repair_best_protocol = context_rank_source_dataset_seed_window/ridge
line_t_context_rank_repair_best_auc_joint_min = 0.4798951048951049
line_t_context_rank_repair_best_precision_joint_min = 0.0
line_t_context_rank_repair_best_recall_joint_min = 0.0
line_t_context_rank_repair_best_pass = 0
```

最新验证：

```text
conda run -n kan python -m py_compile experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py
py_compile_exit = 0
trailing_whitespace_count = 0
```

## 26. EOF 最新权威索引：hard-gate bootstrap repair 后

本文件历史追加顺序存在中段插入。以本节为最新权威状态；第 24/25 节记录了 U10-U14 dictionary expansion / batch-bracket repair，第 27 节记录了 hard-gate margin repair。本节记录最后补做的 hard-gate bootstrap repair。第 23 节中的 `b4ccb...`、旧第 26 节中的 `6582...`、第 27 节中的 `43262...` 均是上一阶段中间产物。

最新主审计产物：

```text
result_dir = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts
zip_path = /home/chengshun.wang/DG-LCA/results/v12_18_b320_codeaudit_lossagnostic_functional/official_from_v1217_v1216_artifacts/v1218_code_review_packet.zip
zip_sha256 = 638ac59eab9c7457ea823028f7e5840d9158e430acbec15f763c41c4df96b79c
zip_entries = 72
hash_manifest_entries = 47
```

最新主 route：

```text
run_id = official_from_v1217_v1216_artifacts_context_rank_dict_u10_u14_margin_bootstrap_refresh
route = R2-B320LabelInitDependenceDetected
fail_reason = current_B320_uses_label_informed_trainprobe_init;label_free_anchor_budget_protocol_not_official;label_free_anchor_budget_A0_does_not_reproduce_locked_anchor;label_free_anchor_budget_LineC_nontearing_failed;strict_loss_agnostic_observables_not_visible
p4_open = 0
promotion_allowed = 0
strict_visibility_pass = 0
strict_auc_joint_min = 0.05660377358490566
strict_precision_joint_min = 0.0
strict_recall_joint_min = 0.0
line_t_context_rank_repair_best_protocol = context_rank_source_dataset_seed_window/ridge
line_t_context_rank_repair_best_auc_joint_min = 0.4798951048951049
line_t_context_rank_repair_best_precision_joint_min = 0.0
line_t_context_rank_repair_best_recall_joint_min = 0.0
line_t_context_rank_repair_best_pass = 0
line_t_hard_gate_margin_repair_rows = 56
line_t_hard_gate_margin_usable_rows = 4
line_t_hard_gate_margin_best_source = repair_seed345_sketchdim24_rank5_b128_w10_12_15
line_t_hard_gate_margin_best_threshold = -0.01
line_t_hard_gate_margin_best_joint_support = 16
line_t_hard_gate_margin_best_new_u10_u14_support = 0
line_t_hard_gate_margin_repair_pass = 0
line_t_hard_gate_bootstrap_repair_rows = 40
line_t_hard_gate_bootstrap_u10_u14_positive_usable_rows = 1
line_t_hard_gate_bootstrap_best_source = repair_seed345_dict_expand_u10_u14_sketchdim24_rank5_b192_w10_12_15
line_t_hard_gate_bootstrap_best_threshold = -0.0125
line_t_hard_gate_bootstrap_best_usable_rate = 0.015625
line_t_hard_gate_bootstrap_repair_pass = 0
```

本轮新增 hard-gate margin/bootstrap repair：

```text
margin_source_count = 8
margin_rows = 56
usable_margin_rows = 4
best_margin_source_run = repair_seed345_sketchdim24_rank5_b128_w10_12_15
best_margin_threshold = -0.01
best_margin_joint_hard_support_rows = 16
best_margin_new_u10_u14_hard_support_rows = 0
margin_repair_pass = 0

bootstrap_rows = 40
bootstrap_source_count = 8
bootstrap_replicates_per_row = 128
u10_u14_rows_with_positive_usable_rate = 1
best_bootstrap_source_run = repair_seed345_dict_expand_u10_u14_sketchdim24_rank5_b192_w10_12_15
best_bootstrap_threshold = -0.0125
best_bootstrap_usable_rate = 0.015625
best_bootstrap_new_u10_u14_support_mean = 23.8125
bootstrap_repair_pass = 0

结论 = margin 没有找到 deterministic usable U10-U14 row；bootstrap 仅出现 1 个低 usable-rate 诊断点，不能开 P3/P4。
```

最新验证：

```text
conda run -n kan python -m py_compile experiments/run_v1216_b320locked_explicit_signal_reservoir_functional.py experiments/run_v1217_b320locked_lossagnostic_target_visibility_functional_geometry.py experiments/run_v1218_b320_codeaudit_lossagnostic_functional.py experiments/run_v1218_b320_label_free_ablation.py
py_compile_exit = 0
trailing_whitespace_count = 0
```
