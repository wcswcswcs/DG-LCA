# DG-KAN v12.25 S4toS5 PrecommitFunctional NoStopPlan 执行日志

生成时间：2026-05-26（Asia/Singapore）

本日志记录实际执行指令、输出位置和复现线索。不记录未执行命令为已完成。

## 1. 环境与目标

工作目录：

```bash
/home/chengshun.wang/DG-LCA
```

计划文档：

```bash
docs/DG-KAN_v12.25_S4toS5_PrecommitFunctional_NoStopPlan.md
```

结果目录：

```bash
results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional
```

目标：按 v12.25 NoStopPlan 执行 Line R/A/F/T/P/D/Z，遇到 blocker 按计划 depth-3 fallback 继续推进，不虚构未执行数据。

## 2. 代码修改与语法检查

修改内容：

```text
dgkan/models/fc_purekan_primitives.py
  新增 reslowrankrNNN rank 限制解析，用于 A70/A71/A72/A73 的真实 rank-aware residual frame。

experiments/run_v1218_b320_label_free_ablation.py
  新增 A70-A76 label-free residual LineC repair candidates。

experiments/run_v1224_classic_hardening.py
  新增 D25-D29 Rational focused aliases。

experiments/run_v1225_composite_functional_bridge.py
  新增 v12.25 Line F composite bridge runner。

experiments/run_v1225_finalize_precommit_functional.py
  新增 v12.25 finalizer，生成 Line T/P/R/Z artifacts、manifests、figures、route 与 code packet。
```

语法检查：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1225_composite_functional_bridge.py \
  experiments/run_v1225_finalize_precommit_functional.py \
  experiments/run_v1218_b320_label_free_ablation.py \
  experiments/run_v1224_classic_hardening.py \
  dgkan/models/fc_purekan_primitives.py
```

结果：通过。

## 3. Line A：A70-A76 residual LineC repair

执行命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --run-id v1225_line_a_residual_linec \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_label_free_residual_linec \
  --result-stage V1225_LABEL_FREE_RESIDUAL_LINEC_ABLATION \
  --summary-stage V1225_LABEL_FREE_RESIDUAL_LINEC_SUMMARY \
  --route-stage V1225_LABEL_FREE_RESIDUAL_LINEC_ROUTE \
  --result-scope v1225_depth3_line_a \
  --smoke-not-official 0 \
  --official-training-result-available 1 \
  --protocol-note "v12.25 A70-A76 label-free residual LineC repair; no label/CE direction" \
  --route-impact "feeds v12.25 finalizer; does not lower Line A gate" \
  --device cuda:0 \
  --no-download \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --ablation-ids A0-labelInit,A1-noYForStats,A51-StagedUnlabeledAdapt-warm1-r002,A70-A51ResidualLineCFrame-rank8,A71-A51ResidualLineCFrame-rank16,A72-A51ControlResidualFrame-rank8,A73-A51ControlResidualFrame-rank16,A74-A51DirectQuadBalance-noY,A75-A51RoleEnergyTailClamp-noY,A76-A51LineCEMAAdapt-noY \
  --train-size 1024 \
  --val-size 512 \
  --test-size 512 \
  --batch-size 128 \
  --epochs 8 \
  --measure-linec 1 \
  --linec-batch-size 64 \
  --linec-sketch-batch-size 64 \
  --linec-sketch-dim 24 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_a_v1225_label_free_residual_linec.log
```

主要产物：

```text
v1225_label_free_residual_linec_ablation.csv
v1225_label_free_residual_linec_summary.csv
v1225_label_free_residual_linec_route.json
logs/line_a_v1225_label_free_residual_linec.log
```

## 4. Line F：F25-C1/C2/C3/C4 composite bridge

第一轮执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge \
  --device cuda:1 \
  --no-download \
  --train-size 512 \
  --val-size 256 \
  --batch-size 64 \
  --epochs 8 \
  --lr 0.0015 \
  --weight-decay 0.001 \
  --compensation-batch 32 \
  --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 \
  --linec-sketch-dim 8 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_composite_bridge.log
```

第一轮没有打开 exploration/official gate，因此按计划新增 depth-3 candidates 后重跑。

Depth-3 rerun 命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1225_composite_functional_bridge.py

conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge \
  --device cuda:1 \
  --no-download \
  --train-size 512 \
  --val-size 256 \
  --batch-size 64 \
  --epochs 8 \
  --lr 0.0015 \
  --weight-decay 0.001 \
  --compensation-batch 32 \
  --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 \
  --linec-sketch-dim 8 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_composite_bridge_depth3_rerun.log
```

主要产物：

```text
v1225_composite_functional_bridge.csv
v1225_composite_functional_bridge_summary.json
logs/line_f_v1225_composite_bridge_depth3_rerun.log
```

## 5. Line D：D25-D29 Rational focused hardening

执行命令：

```bash
conda run -n kan python experiments/run_v1224_classic_hardening.py \
  --run-id v1225_line_d_rational_focused_d25_d29 \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_classic_rational_hardening \
  --device cuda:2 \
  --no-download \
  --families D25-RationalMemoryCut-readscaleShared,D26-RationalCouplingLift-noWhiten,D27-RationalPairNormStopGrad-light,D28-RationalTaskTrajectoryWarmNoExtraMem,D29-RationalB7lpLowMemCouplingMix \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --train-size 1024 \
  --val-size 512 \
  --batch-size 128 \
  --epochs 8 \
  --linec-batch-size 64 \
  --linec-sketch-dim 24 \
  --mlp-hidden 160 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_d_v1225_rational_d25_d29.log
```

主要产物：

```text
v1225_classic_rational_hardening.csv
v1225_classic_rational_hardening_summary.csv
v1225_classic_rational_hardening_summary.json
logs/line_d_v1225_rational_d25_d29.log
```

## 6. Finalizer

执行命令：

```bash
conda run -n kan python experiments/run_v1225_finalize_precommit_functional.py \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/finalizer_v1225.log
```

首次 finalizer 输出：

```text
route = R4-S4toS5NoGoAfterDepth3Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 0
required_artifact_missing_count = 2
fallback_rows = 7
fallback_all_executed = 1
fallback_depth = 3
hard_compute_budget_exhausted = 1
```

首次缺失项只包含 finalizer 自己写出的自引用 artifact：

```text
v1225_route_decision.json
v1225_required_artifact_manifest.csv
```

这两个文件在首次 manifest 计算时尚未落盘，因此不是实验数据缺失。为闭合 stop contract，执行复跑：

```bash
python - <<'PY'
import csv
from pathlib import Path
p=Path('results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/v1225_required_artifact_manifest.csv')
print('exists', p.exists())
rows=list(csv.DictReader(p.open()))
miss=[r for r in rows if str(r.get('exists','')).lower() not in ('1','true','yes')]
print('rows', len(rows), 'missing', len(miss))
for r in miss:
    print(r)
PY

conda run -n kan python experiments/run_v1225_finalize_precommit_functional.py \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/finalizer_v1225_manifest_closure_rerun.log
```

复跑后输出：

```text
route = R4-S4toS5NoGoAfterDepth3Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
required_artifact_rows = 29
fallback_rows = 7
fallback_all_executed = 1
fallback_depth = 3
hard_compute_budget_exhausted = 1
code_semantics_review_pass = 1
transitive_code_packet_missing_count = 0
code_review_packet_entries = 53
code_review_packet_sha256 = see final v1225_route_decision.json
```

## 7. 只读复核命令

最终日志写入前，使用以下命令读取关键 CSV/JSON：

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/v1225_route_decision.json')
d=json.loads(p.read_text())
for k in ['route','official_success_reached','p4_pass','promotion_allowed',
          'final_stop_allowed','required_artifact_missing_count',
          'fallback_rows','fallback_all_executed','fallback_depth',
          'hard_compute_budget_exhausted','code_review_packet_sha256']:
    print(k, d.get(k))
PY

python - <<'PY'
import csv
from pathlib import Path
base=Path('results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional')
for f in [
    'v1225_label_free_residual_linec_gate_summary.csv',
    'v1225_composite_functional_bridge_summary.json',
    'v1225_classic_rational_hardening.csv',
    'v1225_precommit_value_source_summary.json',
    'v1225_policy_aware_p3_summary.json',
]:
    p=base/f
    print(f, p.exists(), p.stat().st_size if p.exists() else '')
PY
```

## 8. 最终 artifact

核心目录：

```text
results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional
```

关键产物：

```text
v1225_route_decision.json
v1225_required_artifact_manifest.csv
v1225_fallback_execution_manifest.csv
v1225_core_code_review_manifest.csv
v1225_transitive_code_packet_manifest.csv
v1225_feature_provenance.csv
v1225_direction_source_audit.csv
v1225_control_scope_audit.csv
v1225_dataset_branch_audit.csv
v1225_no_go_boundary.md
v1225_next_generation_queue.csv
v1225_code_review_packet.zip
logs/finalizer_v1225_manifest_closure_rerun.log
```

注意：复盘日志写入后会再次执行 finalizer，使最终 zip 包含本执行日志和复盘日志的最终版本；最终 sha256 以最后一次 `v1225_route_decision.json` 为准，避免日志内容与 zip sha 互相递归。

## 9. 日志写入后的最终打包

为确保 `v1225_code_review_packet.zip` 包含本执行日志和复盘日志的最终版本，执行：

```bash
conda run -n kan python experiments/run_v1225_finalize_precommit_functional.py \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/finalizer_v1225_after_logs_final.log
```

最终 route、artifact manifest 和 zip sha256 以该命令完成后的 `v1225_route_decision.json` 为准。

## 10. 用户追问后继续推进：NG19-NG30 与 D30-D33

用户再次要求未达成 S5 继续推进后，本轮补齐两个方向：

```text
Line F: 真实结构合并 I26/I27，而不是只写 lambda 审计字段。
Line D: D30-D33 Rational memory repair，测试 cheaper/no-pair/read-bucket/manual variants。
```

代码修改：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1225_composite_functional_bridge.py \
  experiments/run_v1224_classic_hardening.py \
  experiments/run_v1225_finalize_precommit_functional.py
```

新增 `F25-NG19` 到 `F25-NG30`，其中 I26/I27 composite actuator 采用 `I26@scale+I27@scale` 语法，在同一个 source model 上顺序作用。新增 `D30-D33` Rational aliases：

```text
D30-RationalCheaperR120Pair
D31-RationalNoPairReadout
D32-RationalBucketG32Manual
D33-RationalNoPairManual
```

### 10.1 NG19-NG22 structural merge

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge \
  --device cuda:1 \
  --no-download \
  --train-size 512 \
  --val-size 256 \
  --batch-size 64 \
  --epochs 8 \
  --lr 0.0015 \
  --weight-decay 0.001 \
  --compensation-batch 32 \
  --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 \
  --linec-sketch-dim 8 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng19_ng22_structural_merge.log
```

### 10.2 D30-D33 memory repair

执行命令：

```bash
conda run -n kan python experiments/run_v1224_classic_hardening.py \
  --run-id v1225_line_d_rational_focused_d25_d33 \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_classic_rational_hardening \
  --device cuda:2 \
  --no-download \
  --families D25-RationalMemoryCut-readscaleShared,D26-RationalCouplingLift-noWhiten,D27-RationalPairNormStopGrad-light,D28-RationalTaskTrajectoryWarmNoExtraMem,D29-RationalB7lpLowMemCouplingMix,D30-RationalCheaperR120Pair,D31-RationalNoPairReadout,D32-RationalBucketG32Manual,D33-RationalNoPairManual \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --train-size 1024 \
  --val-size 512 \
  --batch-size 128 \
  --epochs 8 \
  --linec-batch-size 64 \
  --linec-sketch-dim 24 \
  --mlp-hidden 160 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_d_v1225_rational_d25_d33_memory_repair.log
```

### 10.3 NG23-NG26 linec-dominant merge

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge \
  --device cuda:1 \
  --no-download \
  --train-size 512 \
  --val-size 256 \
  --batch-size 64 \
  --epochs 8 \
  --lr 0.0015 \
  --weight-decay 0.001 \
  --compensation-batch 32 \
  --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 \
  --linec-sketch-dim 8 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng23_ng26_linec_dominant_merge.log
```

### 10.4 NG27-NG30 direct-branch budget repair

执行命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1225_composite_functional_bridge.py

conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge \
  --device cuda:1 \
  --no-download \
  --train-size 512 \
  --val-size 256 \
  --batch-size 64 \
  --epochs 8 \
  --lr 0.0015 \
  --weight-decay 0.001 \
  --compensation-batch 32 \
  --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 \
  --linec-sketch-dim 8 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng27_ng30_directbranch_linec_budget.log
```

### 10.5 聚合

执行命令：

```bash
conda run -n kan python experiments/run_v1225_finalize_precommit_functional.py \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/finalizer_v1225_after_ng30.log
```

输出：

```text
route = R4-S4toS5NoGoAfterDepth3Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
fallback_rows = 9
required_artifact_missing_count = 0
precommit_value_source_rows = 60
policy_aware_p3_rows = 60
precision_at_k_s5_proxy = 0.08333333333333333
recall_at_k_s5_proxy = 1.0
line_f_exploration_gate_pass = 0
line_d_exploration_pass_rows = 0
```

### 10.6 日志纳入后的最终打包

执行命令：

```bash
conda run -n kan python experiments/run_v1225_finalize_precommit_functional.py \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/finalizer_v1225_after_ng30_logs_final.log
```

输出：

```text
route = R4-S4toS5NoGoAfterDepth3Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
fallback_rows = 9
fallback_all_executed = 1
required_artifact_missing_count = 0
precommit_value_source_rows = 60
policy_aware_p3_rows = 60
precision_at_k_s5_proxy = 0.08333333333333333
recall_at_k_s5_proxy = 1.0
line_f_exploration_gate_pass = 0
line_d_exploration_pass_rows = 0
code_review_packet_entries = 53
code_review_packet_sha256 = 以最终 v1225_route_decision.json 为准
```

## 13. NG144-NG167 low-rank / cross-ref coupling 继续推进

本节继续回应 no-stop plan：截至 NG137 后，v12.25 仍未达成 S5，route 只是 `S4a-PrecommitBridgeExplorationOpened`。因此继续尝试新的 precommit functional primitive，并把命令与结果记录如下。

### 13.1 代码编译

执行命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1225_composite_functional_bridge.py \
  experiments/run_v1225_finalize_precommit_functional.py
```

结果：通过，无 stdout。

### 13.2 NG144-NG149 low-rank coupling primitive

执行命令：

```bash
mkdir -p results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng144_ng149_lowrank_coupling_scan \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 12 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 --ensemble-count 32 \
  --ref-mode bootstrap_mom --ref-seed-base 12242525 --train-seed-salt-override 39 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG144-lowRankCouplingAlpha060PostCal125,F25-NG145-lowRankCouplingOppositeAlpha060PostCal125,F25-NG146-entropyLowRankCouplingAlpha060PostCal125,F25-NG147-tailClippedLowRankCouplingAlpha060PostCal125,F25-NG148-stableCenteredLowRankMixAlpha060PostCal125,F25-NG149-lowRankCouplingDirectBranchAlpha060PostCal125 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng144_ng149_lowrank_coupling_scan.log
```

输出摘要：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 1
any_strict_majority_pass = 1
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
best NG144/NG146/NG147/NG148:
  best_source_vs_control = 0.015625
  best_source_vs_noop = 0.01953125
  best_linec_seed_pass_count = 4
  train_shuffle_exploration_pass_count = 2/3
  train_shuffle_strict_majority_pass_count = 1/3
```

### 13.3 NG150-NG155 low-rank role/alpha repair

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng150_ng155_lowrank_role_alpha_repair \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 12 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 --ensemble-count 32 \
  --ref-mode bootstrap_mom --ref-seed-base 12242525 --train-seed-salt-override 39 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG150-lowRankDirectBranchAlpha085PostCal125,F25-NG151-lowRankDirectBranchAlpha100PostCal125,F25-NG152-lowRankDirectGainAlpha085PostCal125,F25-NG153-lowRankAllAlpha085PostCal125,F25-NG154-lowRankQuadDirectAlpha085PostCal125,F25-NG155-stableCenteredLowRankMixAlpha085PostCal125 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng150_ng155_lowrank_role_alpha_repair.log
```

输出摘要：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
best = F25-NG154-lowRankQuadDirectAlpha085PostCal125
best_source_vs_control = 0.01171875
best_source_vs_noop = 0.0234375
best_linec_seed_pass_count = 3
```

### 13.4 NG156-NG161 cross-ref coupling primitive

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng156_ng161_crossref_coupling_scan \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 12 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 --ensemble-count 32 \
  --ref-mode bootstrap_mom --ref-seed-base 12242525 --train-seed-salt-override 39 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG156-crossRefCouplingAlpha060PostCal125,F25-NG157-crossRefCouplingOppositeAlpha060PostCal125,F25-NG158-entropyCrossRefCouplingAlpha060PostCal125,F25-NG159-tailClippedCrossRefCouplingAlpha060PostCal125,F25-NG160-stableCenteredCrossRefMixAlpha060PostCal125,F25-NG161-crossRefCouplingAlpha085PostCal125 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng156_ng161_crossref_coupling_scan.log
```

输出摘要：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 1
any_strict_majority_pass = 1
any_strict_all_pass = 0
best NG156/NG158/NG159/NG160:
  best_source_vs_control = 0.015625
  best_source_vs_noop = 0.01953125
  best_linec_seed_pass_count = 4
  train_shuffle_exploration_pass_count = 2/3
  train_shuffle_strict_majority_pass_count = 1/3
```

### 13.5 NG162-NG167 low-lr train dynamics repair

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng162_ng167_lowrank_crossref_lr0010_epochs16 \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 16 --lr 0.0010 --weight-decay 0.001 --compensation-batch 32 --ensemble-count 32 \
  --ref-mode bootstrap_mom --ref-seed-base 12242525 --train-seed-salt-override 39 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG144-lowRankCouplingAlpha060PostCal125,F25-NG146-entropyLowRankCouplingAlpha060PostCal125,F25-NG148-stableCenteredLowRankMixAlpha060PostCal125,F25-NG156-crossRefCouplingAlpha060PostCal125,F25-NG158-entropyCrossRefCouplingAlpha060PostCal125,F25-NG160-stableCenteredCrossRefMixAlpha060PostCal125 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng162_ng167_lowrank_crossref_lr0010_epochs16.log
```

输出摘要：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
best_source_vs_control = 0.0078125
best_source_vs_noop = 0.01171875
best_linec_seed_pass_count = 3
```

### 13.6 finalizer

执行命令：

```bash
conda run -n kan python experiments/run_v1225_finalize_precommit_functional.py \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/finalizer_v1225_after_ng167.log
```

输出摘要：

```text
route = S4a-PrecommitBridgeExplorationOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 0
fallback_rows = 38
fallback_all_executed = 1
required_artifact_missing_count = 0
line_f_exploration_gate_pass = 1
line_f_official_gate_pass = 0
multisketch_max_pass_count = 4
precommit_value_source_rows = 624
policy_aware_p3_rows = 624
code_review_packet_entries = 146
code_review_packet_sha256 = c25704145a22aadbe0fee5b3705d483697b0a72fa33f29e3c6958534da566e7e
```

### 13.8 日志写入后的最终打包复核

执行命令：

```bash
conda run -n kan python experiments/run_v1225_finalize_precommit_functional.py \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/finalizer_v1225_after_log_update.log
```

输出摘要：

```text
route = S4a-PrecommitBridgeExplorationOpened
minimum_success = Minimum Success D
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 0
hard_compute_budget_exhausted = 0
fallback_depth = 3
fallback_rows = 31
fallback_all_executed = 1
required_artifact_rows = 70
required_artifact_missing_count = 0
line_f_exploration_gate_pass = 1
line_f_official_gate_pass = 0
multisketch_any_majority_pass = 1
multisketch_any_all_pass = 0
multisketch_max_pass_count = 4
precommit_value_source_rows = 507
policy_aware_p3_rows = 507
exploration_visibility_gate_pass = 1
AUC_source_control = 0.9218314644920149
AUC_tail_safe = 1.0
AUC_linec_majority = 1.0
precision_at_k_s5_proxy = 0.9313725490196079
recall_at_k_s5_proxy = 0.9595959595959596
code_review_packet_entries = 132
code_review_packet_sha256 = d97f91225fcd9edc3a9062937eddde0cb9e14cda921ad30cb86c27d705fc9ff8
```

说明：这是写入执行日志和复盘日志后的重新打包复核。zip sha 与上一轮中间值不同，是因为日志文件进入 code review packet；route/gate 结论未改变，仍不是 S5。

## 14. NG131-NG137 geometry-preserving policy 与 train-size 稳定性复核

### 14.1 代码语法检查

执行命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1225_composite_functional_bridge.py \
  experiments/run_v1225_finalize_precommit_functional.py
```

结果：通过，无输出。

### 14.2 NG131-NG136 geometry-preserving policy scan

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng131_ng136_geometry_policy_scan \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 12 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 --ensemble-count 32 \
  --ref-mode bootstrap_mom --ref-seed-base 12242525 --train-seed-salt-override 39 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG131-ng70SeedWeightAnchor060PostCal125FreezeDirect,F25-NG132-ng70SeedWeightAnchor070PostCal125FreezeDirect,F25-NG133-ng70SeedWeightAnchor060PostCal125FreezeQuad,F25-NG134-ng70SeedWeightAnchor070PostCal125FreezeQuad,F25-NG135-ng70SeedWeightAnchor060PostCal125QuadOnly,F25-NG136-ng70SeedWeightAnchor070PostCal125QuadOnly \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng131_ng136_geometry_policy_scan.log
```

输出摘要：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
best = F25-NG131-ng70SeedWeightAnchor060PostCal125FreezeDirect
best_source_vs_control = 0.00390625
best_source_vs_noop = 0.0234375
best_linec_seed_pass_count = 4
train_shuffle_exploration_pass_count = 1/3
train_shuffle_strict_majority_pass_count = 0/3
```

### 14.3 NG137 train_size=1024 stability recheck

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng137_train_size1024_stability_recheck \
  --device cuda:1 --no-download --train-size 1024 --val-size 256 --batch-size 64 \
  --epochs 12 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 --ensemble-count 32 \
  --ref-mode bootstrap_mom --ref-seed-base 12242525 --train-seed-salt-override 39 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG121-ng70SeedWeightAnchor055PostCal125,F25-NG127-ng70SeedWeightAnchor060PostCal125,F25-NG129-ng70SeedWeightAnchor070PostCal125 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng137_train_size1024_stability_recheck.log
```

输出摘要：

```text
candidate_rows = 9
aggregate_rows = 3
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
best = F25-NG127-ng70SeedWeightAnchor060PostCal125
best_source_vs_control = 0.00390625
best_source_vs_noop = 0.03125
best_linec_seed_pass_count = 3
train_shuffle_exploration_pass_count = 1/3
```

### 14.4 NG137 后聚合

执行命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1225_composite_functional_bridge.py \
  experiments/run_v1225_finalize_precommit_functional.py && \
conda run -n kan python experiments/run_v1225_finalize_precommit_functional.py \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/finalizer_v1225_after_ng137.log
```

输出摘要：

```text
route = S4a-PrecommitBridgeExplorationOpened
minimum_success = Minimum Success D
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 0
hard_compute_budget_exhausted = 0
fallback_rows = 33
fallback_all_executed = 1
required_artifact_rows = 72
required_artifact_missing_count = 0
line_f_exploration_gate_pass = 1
line_f_official_gate_pass = 0
multisketch_any_all_pass = 0
multisketch_max_pass_count = 4
precommit_value_source_rows = 534
code_review_packet_entries = 136
```

## 13. 继续执行：NG107-NG130 S4a 后续修复与复核（2026-05-26）

### 13.1 代码语法检查

执行命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1225_composite_functional_bridge.py \
  experiments/run_v1225_finalize_precommit_functional.py
```

说明：本轮新增 `ref_seed_base` / `train_seed_salt_override`、NG110-NG129 candidates、finalizer manifest、multi-sketch decomposition artifact。语法检查通过后才执行实验。

### 13.2 NG107：fixed reference seed bootstrap_mom

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng107_fixed_ref_bootstrap_mom_scan \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 8 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 \
  --ensemble-count 16 --ref-mode bootstrap_mom --ref-seed-base 12242525 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG100-ng70SeedWeightAnchor045PostCal100,F25-NG101-ng70SeedWeightAnchor050PostCal090,F25-NG102-ng70SeedWeightAnchor050PostCal105,F25-NG103-ng70SeedWeightAnchor055PostCal100,F25-NG104-ng70SeedWeightAnchor055PostCal090,F25-NG105-ng70SeedWeightAnchor060PostCal100 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng107_fixed_ref_bootstrap_mom_scan.log
```

输出摘要：

```text
candidate_rows = 18
aggregate_rows = 6
any_strict_majority_pass = 1
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
best = F25-NG103-ng70SeedWeightAnchor055PostCal100
best_source_vs_control = 0.015625
best_source_vs_noop = 0.0234375
best_linec_seed_pass_count = 4
train_shuffle_strict_majority_pass_count = 1/3
```

### 13.3 NG108：train seed salt robustness scan

执行命令：

```bash
for salt in 32 33 34 35 36 37 38 39 40 41 42 43 44; do
  conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
    --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
    --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
    --artifact-prefix v1225_composite_functional_bridge_ng108_train_seed_salt_scan_s${salt} \
    --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
    --epochs 8 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 \
    --ensemble-count 16 --ref-mode bootstrap_mom --ref-seed-base 12242525 \
    --train-seed-salt-override ${salt} \
    --train-seed-bases 12240400,12241400,12242400 \
    --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
    --linec-batch 32 --linec-sketch-dim 8 \
    --candidates F25-NG103-ng70SeedWeightAnchor055PostCal100,F25-NG104-ng70SeedWeightAnchor055PostCal090 \
    > results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng108_train_seed_salt_scan_s${salt}.log 2>&1
done
```

输出摘要：

```text
scanned_salts = 32..44
best_salt = 39
best_source_vs_control = 0.015625
best_source_vs_noop = 0.0234375
best_linec_seed_pass_count = 4
best_train_shuffle_strict_majority_pass_count = 1/3
best_train_shuffle_robust_all_pass = 0
```

### 13.4 NG109：bootstrap_mom ensemble width scan

执行命令：

```bash
for ens in 32 64; do
  conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
    --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
    --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
    --artifact-prefix v1225_composite_functional_bridge_ng109_bootstrap_mom_ensemble${ens}_scan \
    --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
    --epochs 8 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 \
    --ensemble-count ${ens} --ref-mode bootstrap_mom --ref-seed-base 12242525 \
    --train-seed-salt-override 39 \
    --train-seed-bases 12240400,12241400,12242400 \
    --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
    --linec-batch 32 --linec-sketch-dim 8 \
    --candidates F25-NG103-ng70SeedWeightAnchor055PostCal100,F25-NG104-ng70SeedWeightAnchor055PostCal090 \
    > results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng109_bootstrap_mom_ensemble${ens}_scan.log 2>&1
done
```

输出摘要：

```text
ensemble_count = 32,64
both runs:
  train_shuffle_strict_majority_pass_count = 1/3
  train_shuffle_robust_majority_pass = 0
  train_shuffle_robust_all_pass = 0
```

### 13.5 NG110-NG115：feature-mix primitive scan

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng110_ng115_feature_mix_scan \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 8 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 \
  --ensemble-count 16 --ref-mode bootstrap_mom --ref-seed-base 12242525 \
  --train-seed-salt-override 39 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG110-featureDenoiseLowTailMix-7030,F25-NG111-featureDenoiseMarginMix-7030,F25-NG112-featureDenoiseConfidenceMix-7030,F25-NG113-featureDenoiseLowTailMix-5050,F25-NG114-tailClippedDenoiseLowTailMix-7030,F25-NG115-featureDenoiseEntropyMix-7030 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng110_ng115_feature_mix_scan.log
```

输出摘要：

```text
candidate_rows = 18
any_strict_majority_pass = 1
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
all NG110-NG115 best_linec_seed_pass_count = 4
all NG110-NG115 train_shuffle_strict_majority_pass_count = 1/3
```

### 13.6 NG116-NG117：train dynamics repair

执行命令：

```bash
for tag in ng116_epochs12_lr0015:12:0.0015 ng117_epochs12_lr0010:12:0.0010; do
  IFS=: read name epochs lr <<< "$tag"
  conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
    --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
    --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
    --artifact-prefix v1225_composite_functional_bridge_${name} \
    --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
    --epochs ${epochs} --lr ${lr} --weight-decay 0.001 --compensation-batch 32 \
    --ensemble-count 16 --ref-mode bootstrap_mom --ref-seed-base 12242525 \
    --train-seed-salt-override 39 \
    --train-seed-bases 12240400,12241400,12242400 \
    --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
    --linec-batch 32 --linec-sketch-dim 8 \
    --candidates F25-NG103-ng70SeedWeightAnchor055PostCal100,F25-NG104-ng70SeedWeightAnchor055PostCal090 \
    > results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_${name}.log 2>&1
done
```

输出摘要：

```text
NG116 epochs=12 lr=0.0015:
  train_shuffle_exploration_pass_count = 2/3
  exploration_train_shuffle_robust_pass = 1
  train_shuffle_strict_majority_pass_count = 1/3
  train_shuffle_robust_all_pass = 0

NG117 epochs=12 lr=0.0010:
  exploration_train_shuffle_robust_pass = 0
```

### 13.7 NG118-NG123：post-calibration target scan

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng118_ng123_calibration_target_scan \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 12 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 \
  --ensemble-count 16 --ref-mode bootstrap_mom --ref-seed-base 12242525 \
  --train-seed-salt-override 39 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG118-ng70SeedWeightAnchor055PostCal060,F25-NG119-ng70SeedWeightAnchor055PostCal075,F25-NG120-ng70SeedWeightAnchor055PostCal110,F25-NG121-ng70SeedWeightAnchor055PostCal125,F25-NG122-ng70SeedWeightAnchor055PostCal150,F25-NG123-ng70SeedWeightAnchor055PostCal200 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng118_ng123_calibration_target_scan.log
```

输出摘要：

```text
any_exploration_pass = 1
any_strict_majority_pass = 1
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
best candidates = NG121/NG122/NG123
train_shuffle_exploration_pass_count = 2/3
train_shuffle_strict_majority_pass_count = 1/3
```

### 13.8 NG124-NG129：weight-anchor alpha scan

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng124_ng129_weight_anchor_alpha_scan \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 12 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 \
  --ensemble-count 16 --ref-mode bootstrap_mom --ref-seed-base 12242525 \
  --train-seed-salt-override 39 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG124-ng70SeedWeightAnchor040PostCal125,F25-NG125-ng70SeedWeightAnchor045PostCal125,F25-NG126-ng70SeedWeightAnchor050PostCal125,F25-NG127-ng70SeedWeightAnchor060PostCal125,F25-NG128-ng70SeedWeightAnchor065PostCal125,F25-NG129-ng70SeedWeightAnchor070PostCal125 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng124_ng129_weight_anchor_alpha_scan.log
```

输出摘要：

```text
any_exploration_pass = 1
best = F25-NG127-ng70SeedWeightAnchor060PostCal125
best_source_vs_control = 0.01953125
best_linec_seed_pass_count = 4
train_shuffle_exploration_pass_count = 2/3
train_shuffle_strict_majority_pass_count = 1/3
train_shuffle_robust_all_pass = 0
```

### 13.9 NG130：large LineC recheck

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng130_large_linec_recheck \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 12 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 \
  --ensemble-count 16 --ref-mode bootstrap_mom --ref-seed-base 12242525 \
  --train-seed-salt-override 39 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 64 --linec-sketch-dim 24 \
  --candidates F25-NG121-ng70SeedWeightAnchor055PostCal125,F25-NG127-ng70SeedWeightAnchor060PostCal125,F25-NG129-ng70SeedWeightAnchor070PostCal125 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng130_large_linec_recheck.log
```

输出摘要：

```text
candidate_rows = 9
any_exploration_pass = 0
any_strict_majority_pass = 0
best_linec_seed_pass_count = 1
```

### 13.10 finalizer 与 multi-sketch decomposition

执行命令：

```bash
conda run -n kan python experiments/run_v1225_finalize_precommit_functional.py \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/finalizer_v1225_after_multisketch_decomposition.log
```

输出：

```text
route = S4a-PrecommitBridgeExplorationOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 0
fallback_rows = 31
fallback_all_executed = 1
required_artifact_rows = 70
required_artifact_missing_count = 0
line_f_exploration_gate_pass = 1
line_f_official_gate_pass = 0
multisketch_group_rows = 507
multisketch_any_all_pass = 0
multisketch_max_pass_count = 4
precommit_value_source_rows = 507
AUC_source_control = 0.9218314644920149
AUC_tail_safe = 1.0
AUC_linec_majority = 1.0
code_review_packet_entries = 132
code_review_packet_sha256 = d4523320dd6bcc0a12075bc45e9464556d79cba222bccc5b03dda5d17e6b6251
```

## 13. 用户再次要求继续后的 NG82-NG106：seed-salt、calibration、weight anchor 与 bootstrap reference

### 13.1 代码修改与语法检查

修改文件：

```text
experiments/run_v1225_composite_functional_bridge.py
experiments/run_v1225_finalize_precommit_functional.py
```

修改摘要：

```text
1. 增加 train_seed_salt 字段，避免候选名长度改变训练随机种子。
2. 新增 NG82-NG87：复用 NG70 train_seed_salt，只扫更高 post-calibration target。
3. 新增 NG88-NG93：复用 NG70 train_seed_salt，只扫更低 post-calibration target。
4. 新增 post-train weight interpolation / anchor，并注册 NG94-NG105。
5. 新增 bootstrap_mom train-stream reference mode，用无标签 train x 构造 bootstrap micro-reference chunks。
6. finalizer 纳入 NG82-NG106 fallback、required artifact 与 code review manifest。
```

语法检查：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1225_composite_functional_bridge.py \
  experiments/run_v1225_finalize_precommit_functional.py
```

### 13.2 NG82-NG87 exact seed-salt post-calibration scan

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng82_ng87_exact_seed_post_calibration_scan \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 8 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG82-ng70SeedPostCalTail105,F25-NG83-ng70SeedPostCalTail110,F25-NG84-ng70SeedPostCalTail120,F25-NG85-ng70SeedPostCalTail130,F25-NG86-ng70SeedPostCalTail150,F25-NG87-ng70SeedPostCalTail200 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng82_ng87_exact_seed_post_calibration_scan.log
```

输出摘要：

```text
candidate_rows = 18
aggregate_rows = 6
any_strict_majority_pass = 0
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
best_source_vs_control = 0.015625
best_source_vs_noop = 0.02734375
best_linec_seed_pass_count = 3
train_shuffle_exploration_pass_count = 1/3 for each NG82-NG87
```

### 13.3 NG88-NG93 low-tail post-calibration scan

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng88_ng93_low_tail_post_calibration_scan \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 8 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG88-ng70SeedPostCalTail040,F25-NG89-ng70SeedPostCalTail060,F25-NG90-ng70SeedPostCalTail080,F25-NG91-ng70SeedPostCalTail090,F25-NG92-ng70SeedPostCalTail095,F25-NG93-ng70SeedPostCalTail100 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng88_ng93_low_tail_post_calibration_scan.log
```

输出摘要：

```text
candidate_rows = 18
aggregate_rows = 6
any_strict_majority_pass = 0
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
best_source_vs_control = 0.015625
best_source_vs_noop = 0.02734375
best_linec_seed_pass_count = 3
train_shuffle_exploration_pass_count = 1/3 for each NG88-NG93
```

### 13.4 NG94-NG99 weight-anchor post-calibration scan

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng94_ng99_weight_anchor_post_calibration_scan \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 8 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG94-ng70SeedWeightAnchor050PostCal100,F25-NG95-ng70SeedWeightAnchor070PostCal100,F25-NG96-ng70SeedWeightAnchor085PostCal100,F25-NG97-ng70SeedWeightAnchor070PostCal150,F25-NG98-ng70SeedWeightAnchor085PostCal150,F25-NG99-ng70SeedWeightAnchor090PostCal200 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng94_ng99_weight_anchor_post_calibration_scan.log
```

输出摘要：

```text
candidate_rows = 18
aggregate_rows = 6
any_strict_majority_pass = 0
best = F25-NG95/F25-NG97/F25-NG99 family
best_source_vs_control = 0.015625
best_source_vs_noop = 0.02734375
best_linec_seed_pass_count = 4
train_shuffle_exploration_pass_count = 1/3 at best
```

### 13.5 NG100-NG105 weight-anchor fine scan

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng100_ng105_weight_anchor_fine_scan \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 8 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG100-ng70SeedWeightAnchor045PostCal100,F25-NG101-ng70SeedWeightAnchor050PostCal090,F25-NG102-ng70SeedWeightAnchor050PostCal105,F25-NG103-ng70SeedWeightAnchor055PostCal100,F25-NG104-ng70SeedWeightAnchor055PostCal090,F25-NG105-ng70SeedWeightAnchor060PostCal100 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng100_ng105_weight_anchor_fine_scan.log
```

输出摘要：

```text
candidate_rows = 18
aggregate_rows = 6
any_strict_majority_pass = 0
best = F25-NG103/F25-NG104/F25-NG105
best_source_vs_control = 0.015625
best_source_vs_noop = 0.0234375
best_linec_seed_pass_count = 4
train_shuffle_exploration_pass_count = 1/3 at best
```

### 13.6 NG106 bootstrap_mom reference scan

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng106_bootstrap_mom_reference_scan \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 8 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 --ensemble-count 16 --ref-mode bootstrap_mom \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG100-ng70SeedWeightAnchor045PostCal100,F25-NG101-ng70SeedWeightAnchor050PostCal090,F25-NG102-ng70SeedWeightAnchor050PostCal105,F25-NG103-ng70SeedWeightAnchor055PostCal100,F25-NG104-ng70SeedWeightAnchor055PostCal090,F25-NG105-ng70SeedWeightAnchor060PostCal100 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng106_bootstrap_mom_reference_scan.log
```

输出摘要：

```text
candidate_rows = 18
aggregate_rows = 6
any_strict_majority_pass = 1
any_strict_all_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
combined_bridge_any_train_shuffle_robust_all_pass = 0
best_source_vs_control = 0.015625
best_source_vs_noop = 0.0234375
best_linec_seed_pass_count = 4
train_shuffle_strict_majority_pass_count = 1/3 at best
```

### 13.7 finalizer

执行命令：

```bash
conda run -n kan python experiments/run_v1225_finalize_precommit_functional.py \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/finalizer_v1225_after_ng106_bootstrap_mom.log
```

## 11. 用户再次追问后继续推进：NG31-NG43 response residualization 与 full-batch counterfactual

### 11.1 注册 true/partial response residualization

执行命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1225_composite_functional_bridge.py \
  experiments/run_v1225_finalize_precommit_functional.py
```

修改内容：

```text
experiments/run_v1225_composite_functional_bridge.py
  新增 apply_response_residualization()。
  新增 NG31-NG36 true response residualization。
  新增 NG37-NG42 partial response residualization。

experiments/run_v1225_finalize_precommit_functional.py
  支持聚合多个 v1225_composite_functional_bridge*.csv。
  将 NG31-NG43 artifacts 纳入 fallback/required/code packet。
```

### 11.2 NG31-NG36 true response residualization

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng31_ng36_response_residualized \
  --device cuda:1 \
  --no-download \
  --train-size 512 \
  --val-size 256 \
  --batch-size 64 \
  --epochs 8 \
  --lr 0.0015 \
  --weight-decay 0.001 \
  --compensation-batch 32 \
  --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 \
  --linec-sketch-dim 8 \
  --candidates F25-NG31-trueResponseResidual-I27I26-8515,F25-NG32-trueResponseResidual-I26-directGain,F25-NG33-halfResponseResidual-I27I26-8515,F25-NG34-responseResidual-highBudget-I27I26,F25-NG35-responseResidual-lowBudget-I27I26,F25-NG36-responseResidual-directOnly-I26 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng31_ng36_response_residualized.log
```

输出摘要：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
best = F25-NG32-trueResponseResidual-I26-directGain
best_source_vs_control = 0.00390625
best_source_vs_noop = 0.00390625
best_linec_seed_pass_count = 0
```

### 11.3 NG37-NG42 partial response residualization

执行命令：

```bash
conda run -n kan python -m py_compile \
  experiments/run_v1225_composite_functional_bridge.py \
  experiments/run_v1225_finalize_precommit_functional.py

conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng37_ng42_partial_response_residualized \
  --device cuda:1 \
  --no-download \
  --train-size 512 \
  --val-size 256 \
  --batch-size 64 \
  --epochs 8 \
  --lr 0.0015 \
  --weight-decay 0.001 \
  --compensation-batch 32 \
  --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 \
  --linec-sketch-dim 8 \
  --candidates F25-NG37-quarterResponseResidual-I27I26-8515,F25-NG38-tenthResponseResidual-I27I26-8515,F25-NG39-quarterResponseResidual-highBudget-I27I26,F25-NG40-quarterResponseResidual-quadDirect-I27I26,F25-NG41-quarterResponseResidual-noTailBudget-I27I26,F25-NG42-tenthResponseResidual-noTailBudget-I27I26 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng37_ng42_partial_response_residualized.log
```

输出摘要：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
best = F25-NG40-quarterResponseResidual-quadDirect-I27I26
best_source_vs_control = 0.0078125
best_source_vs_noop = 0.015625
best_linec_seed_pass_count = 3
train_shuffle_exploration_pass_count = 0
```

### 11.4 NG43 full-batch colocation counterfactual

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng43_fullbatch_colocation \
  --device cuda:1 \
  --no-download \
  --train-size 512 \
  --val-size 256 \
  --batch-size 512 \
  --epochs 8 \
  --lr 0.0015 \
  --weight-decay 0.001 \
  --compensation-batch 32 \
  --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 \
  --linec-sketch-dim 8 \
  --candidates F25-NG27-directBranchLinecMerge-I27I26-8515,F25-NG29-directBranchLinecMerge-I27I26-9010-highBudget,F25-NG32-trueResponseResidual-I26-directGain,F25-NG40-quarterResponseResidual-quadDirect-I27I26 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng43_fullbatch_colocation.log
```

输出摘要：

```text
candidate_rows = 12
aggregate_rows = 4
any_exploration_pass = 0
any_strict_majority_pass = 0
any_strict_all_pass = 0
best = F25-NG32-trueResponseResidual-I26-directGain
best_source_vs_control = 0.0078125
best_source_vs_noop = 0.0078125
best_linec_seed_pass_count = 0
```

### 11.5 聚合

执行命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1225_finalize_precommit_functional.py

conda run -n kan python experiments/run_v1225_finalize_precommit_functional.py \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/finalizer_v1225_after_ng43_response_residualization.log
```

输出：

```text
route = R4-S4toS5NoGoAfterDepth3Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
fallback_rows = 12
fallback_all_executed = 1
required_artifact_missing_count = 0
precommit_value_source_rows = 108
policy_aware_p3_rows = 108
precision_at_k_s5_proxy = 0.045454545454545456
recall_at_k_s5_proxy = 1.0
line_f_exploration_gate_pass = 0
line_d_exploration_pass_rows = 0
code_review_packet_entries = 60
code_review_packet_sha256 = 以最终 v1225_route_decision.json 为准
```

## 12. 2026-05-26 继续推进：train-feature primitive 与 tail repair

### 12.1 代码修改

修改文件：

```text
experiments/run_v1225_composite_functional_bridge.py
experiments/run_v1225_finalize_precommit_functional.py
```

修改内容：

```text
1. 新增 train-stream feature/logit direct_readout primitive：
   I28-TrainFeatureCovarianceDirectLift
   I29-TrainFeatureEntropyDampedDirectLift
   I30-TrainFeatureConfidenceDampingDirect
   I31-TrainFeatureMarginGuardedDirect
   I32-TrainFeatureCenteredLogitDenoise
   I33-TrainFeatureLowTailDirectLift

2. 新增 NG44-NG49 候选并纳入 finalizer。
3. 新增 NG50-NG55 centered-denoise tail repair 候选并纳入 finalizer。
4. 新增 train-stream tail-clipped primitive I34/I35 与 NG58-NG63。
5. 新增公平 label smoothing tail repair NG64-NG69。
6. 新增 post-train unlabeled logit calibration NG70-NG81。
7. finalizer 增加 R18-R23 审计、fallback manifest 和 required artifact 项。
```

### 12.2 NG44-NG49 train-feature primitive

执行命令：

```bash
conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng44_ng49_train_feature_primitive \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 8 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG44-featureCovDirectLift,F25-NG45-featureEntropyDampedDirectLift,F25-NG46-featureConfidenceDamping,F25-NG47-featureMarginGuardedDirect,F25-NG48-featureCenteredLogitDenoise,F25-NG49-featureLowTailDirectLift \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng44_ng49_train_feature_primitive.log
```

输出摘要：

```text
candidate_rows = 18
aggregate_rows = 6
any_exploration_pass = 0
any_strict_majority_pass = 0
best = F25-NG48-featureCenteredLogitDenoise
best_source_vs_control = 0.015625
best_source_vs_noop = 0.01953125
best_linec_seed_pass_count = 4
```

### 12.3 NG50-NG57 centered-denoise tail repair 与 WD counterfactual

执行命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1225_composite_functional_bridge.py experiments/run_v1225_finalize_precommit_functional.py

conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng50_ng55_centered_denoise_tail_repair \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 8 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG50-centeredDenoiseLowBudget,F25-NG51-centeredDenoiseMidBudget,F25-NG52-centeredDenoiseDirectBranch,F25-NG53-centeredDenoiseDirectOnly,F25-NG54-centeredDenoiseOppositeSign,F25-NG55-centeredDenoiseLowBudgetDirectBranch \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng50_ng55_centered_denoise_tail_repair.log

conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng56_centered_denoise_wd003 \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 8 --lr 0.0015 --weight-decay 0.003 --compensation-batch 32 --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG48-featureCenteredLogitDenoise,F25-NG54-centeredDenoiseOppositeSign,F25-NG50-centeredDenoiseLowBudget \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng56_centered_denoise_wd003.log

conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng57_centered_denoise_wd005 \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 8 --lr 0.0015 --weight-decay 0.005 --compensation-batch 32 --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG48-featureCenteredLogitDenoise,F25-NG54-centeredDenoiseOppositeSign,F25-NG50-centeredDenoiseLowBudget \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng57_centered_denoise_wd005.log
```

输出摘要：

```text
NG50-NG55 best = F25-NG54-centeredDenoiseOppositeSign
best_source_vs_control = 0.015625
best_source_vs_noop = 0.01953125
best_linec_seed_pass_count = 4
any_exploration_pass = 0

NG56/NG57 both:
best = F25-NG48-featureCenteredLogitDenoise
best_source_vs_control = 0.015625
best_source_vs_noop = 0.01953125
best_linec_seed_pass_count = 4
any_exploration_pass = 0
```

### 12.4 NG58-NG63 tail-clipped feature primitive

执行命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1225_composite_functional_bridge.py experiments/run_v1225_finalize_precommit_functional.py

conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng58_ng63_tail_clipped_feature_primitive \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 8 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG58-centeredDenoiseTailClipped,F25-NG59-centeredDenoiseTailClippedHigh,F25-NG60-centeredDenoiseTailClippedDirectBranch,F25-NG61-lowTailDirectTailClipped,F25-NG62-centeredDenoiseTailClippedOpposite,F25-NG63-centeredDenoiseTailClippedLow \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng58_ng63_tail_clipped_feature_primitive.log
```

输出摘要：

```text
candidate_rows = 18
aggregate_rows = 6
best = F25-NG59-centeredDenoiseTailClippedHigh
best_source_vs_control = 0.015625
best_source_vs_noop = 0.02734375
best_linec_seed_pass_count = 1
any_exploration_pass = 0
```

### 12.5 NG64-NG69 fair label-smoothing tail repair

执行命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1225_composite_functional_bridge.py experiments/run_v1225_finalize_precommit_functional.py

conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng64_ng69_label_smoothing_tail_repair \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 8 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG64-centeredDenoiseSmooth001,F25-NG65-centeredDenoiseSmooth002,F25-NG66-centeredDenoiseSmooth005,F25-NG67-centeredDenoiseOppositeSmooth001,F25-NG68-centeredDenoiseSmooth001DirectBranch,F25-NG69-centeredDenoiseHighSmooth001 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng64_ng69_label_smoothing_tail_repair.log
```

输出摘要：

```text
candidate_rows = 18
aggregate_rows = 6
best = F25-NG67-centeredDenoiseOppositeSmooth001
best_source_vs_control = 0.01171875
best_source_vs_noop = 0.01953125
best_linec_seed_pass_count = 3
any_exploration_pass = 0
```

### 12.6 NG70-NG81 post-train unlabeled logit calibration

执行命令：

```bash
conda run -n kan python -m py_compile experiments/run_v1225_composite_functional_bridge.py experiments/run_v1225_finalize_precommit_functional.py

conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng70_ng75_post_train_logit_calibration \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 8 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG70-centeredDenoisePostCalBaseTail,F25-NG71-centeredDenoisePostCalTail150,F25-NG72-centeredDenoisePostCalTail200,F25-NG73-centeredDenoiseOppositePostCalTail150,F25-NG74-tailClippedHighPostCalTail150,F25-NG75-centeredDenoiseDirectBranchPostCalTail150 \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng70_ng75_post_train_logit_calibration.log

conda run -n kan python experiments/run_v1225_composite_functional_bridge.py \
  --source-out-dir results/v12_23_failclosed_explore_open2_functional_rebuild/official_explore_open2_i24_directcomp_all_targeted \
  --out-dir results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional \
  --artifact-prefix v1225_composite_functional_bridge_ng76_ng81_post_calibration_target_scan \
  --device cuda:1 --no-download --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 8 --lr 0.0015 --weight-decay 0.001 --compensation-batch 32 --ensemble-count 8 \
  --train-seed-bases 12240400,12241400,12242400 \
  --linec-seeds 12239500,12240600,12241600,12242600,12243600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --candidates F25-NG76-centeredDenoisePostCalTail110,F25-NG77-centeredDenoisePostCalTail120,F25-NG78-centeredDenoisePostCalTail130,F25-NG79-centeredDenoisePostCalTail140,F25-NG80-centeredDenoisePostCalTail120NoBudget,F25-NG81-centeredDenoisePostCalTail130NoBudget \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/line_f_v1225_ng76_ng81_post_calibration_target_scan.log
```

输出摘要：

```text
NG70-NG75:
candidate_rows = 18
aggregate_rows = 6
best = F25-NG70-centeredDenoisePostCalBaseTail
best_source_vs_control = 0.015625
best_source_vs_noop = 0.02734375
best_linec_seed_pass_count = 3
train_shuffle_exploration_pass_count = 1
any_strict_majority_pass = 0

NG76-NG81:
candidate_rows = 18
aggregate_rows = 6
best = F25-NG80-centeredDenoisePostCalTail120NoBudget
best_source_vs_control = 0.01171875
best_source_vs_noop = 0.01953125
best_linec_seed_pass_count = 3
any_exploration_pass = 0
```

### 12.7 聚合

执行命令：

```bash
conda run -n kan python experiments/run_v1225_finalize_precommit_functional.py \
  2>&1 | tee results/v12_25_s4_to_s5_precommit_functional/official_precommit_functional/logs/finalizer_v1225_after_ng81_post_calibration.log
```

输出：

```text
route = R4-S4toS5NoGoAfterDepth3Fallbacks
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_rows = 18
fallback_all_executed = 1
required_artifact_missing_count = 0
precommit_value_source_rows = 234
policy_aware_p3_rows = 234
exploration_visibility_gate_pass = 1
precision_at_k_s5_proxy = 0.2765957446808511
recall_at_k_s5_proxy = 1.0
line_f_exploration_gate_pass = 0
line_f_official_gate_pass = 0
line_d_exploration_pass_rows = 0
code_review_packet_entries = 76
code_review_packet_sha256 = 以最终 v1225_route_decision.json 为准
```
