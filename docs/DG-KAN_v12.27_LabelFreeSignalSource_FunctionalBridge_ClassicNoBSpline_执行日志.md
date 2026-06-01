# DG-KAN v12.27 LabelFreeSignalSource FunctionalBridge ClassicNoBSpline 执行日志

生成时间：2026-05-26（Asia/Singapore）

本日志只记录实际执行过的命令、文件与产物路径；不补写未执行命令。

## 1. 计划入口

计划文档：

```text
docs/DG-KAN_v12.27_LabelFreeSignalSource_FunctionalBridge_ClassicNoBSpline_完整计划.md
```

结果目录：

```text
results/v12_27_label_free_signal_source_functional_bridge/official_v1227
```

## 2. 本轮代码修改记录

修改文件：

```text
dgkan/models/fc_purekan_primitives.py
experiments/run_v1218_b320_label_free_ablation.py
experiments/run_v1224_classic_hardening.py
experiments/run_v1227_label_free_signal_source.py
experiments/run_v1227_finalize_label_free_signal_source.py
```

新增日志文件：

```text
docs/DG-KAN_v12.27_LabelFreeSignalSource_FunctionalBridge_ClassicNoBSpline_执行日志.md
docs/DG-KAN_v12.27_LabelFreeSignalSource_FunctionalBridge_ClassicNoBSpline_实验结果复盘.md
```

## 3. 待执行命令

后续 sections 将按实际执行顺序写入：

```text
1. py_compile / registry audit
2. Line A signal-source scout
3. Line A hardening top-4
4. Line A repair depth
5. Line F shadow / official functional bridge
6. Line D Rational D34-D38 memory repair
7. finalizer and final packet recheck
```

## 4. 语法检查与 provenance audit

执行命令：

```bash
python -m py_compile dgkan/models/fc_purekan_primitives.py experiments/run_v1218_b320_label_free_ablation.py experiments/run_v1224_classic_hardening.py experiments/run_v1227_label_free_signal_source.py experiments/run_v1227_finalize_label_free_signal_source.py
```

结果：

```text
py_compile pass
```

执行命令：

```bash
conda run -n kan python -c "from experiments.run_v1218_b320_label_free_ablation import ablation_specs; from experiments.run_v1226_label_free_only_bridge import forbidden_token_present; from experiments.run_v1227_label_free_signal_source import v1227_candidates; ..."
```

结果摘要：

```text
A-S1a..A-S5d: uses_y_for_stats=0, forbidden_token_present=0
F27-G/R/C candidates: forbidden_token_present=0
```

## 5. Line A signal-source scout

执行命令：

```bash
mkdir -p results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_27_label_free_signal_source_functional_bridge/official_v1227 \
  --artifact-prefix v1227_label_free_signal_source_scout \
  --run-id v1227_label_free_signal_source_scout \
  --summary-stage V1227_LABEL_FREE_SIGNAL_SOURCE_SCOUT_SUMMARY \
  --route-stage V1227_LABEL_FREE_SIGNAL_SOURCE_SCOUT_ROUTE \
  --result-scope v1227_signal_source_scout \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --ablation-ids A-S1a-MultiViewStableFrame,A-S1b-MultiViewStablePlusResidual,A-S1c-MultiViewBlockLocalStable,A-S1d-MultiViewLowQuadDirect,A-S2a-TemporalDriftStableFrame,A-S2b-TemporalDriftResidualFrame,A-S2c-TemporalDriftLowRankDirect,A-S3a-BlockLocalCovFrame,A-S3b-BlockLocalEdgeEnergyFrame,A-S3c-BlockLocalStableAugFrame,A-S3d-BlockLocalLowQuadDirect,A-S4a-CrossRandomProjectionFrame,A-S4b-CrossProjectionStableResidual,A-S4c-CrossProjectionLowQuadDirect \
  --train-size 512 --val-size 256 --test-size 256 \
  --epochs 3 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 32 --linec-sketch-dim 8 \
  2>&1 | tee results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs/v1227_linea_signal_source_scout.log
```

产物：

```text
v1227_label_free_signal_source_scout_ablation.csv
v1227_label_free_signal_source_scout_summary.csv
v1227_label_free_signal_source_scout_route.json
```

结果摘要：

```text
rows = 144
summary_rows = 16
best scout candidate = A-S2a-TemporalDriftStableFrame
```

## 6. Line A hardening top-4

选择规则：2 个按 task mean/worst，1 个按 LineC，1 个按 AUC-time。实际选择：

```text
A-S2a-TemporalDriftStableFrame
A-S4c-CrossProjectionLowQuadDirect
A-S4a-CrossRandomProjectionFrame
A-S3b-BlockLocalEdgeEnergyFrame
```

执行命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_27_label_free_signal_source_functional_bridge/official_v1227 \
  --artifact-prefix v1227_label_free_signal_source_hardening \
  --run-id v1227_label_free_signal_source_hardening \
  --summary-stage V1227_LABEL_FREE_SIGNAL_SOURCE_HARDENING_SUMMARY \
  --route-stage V1227_LABEL_FREE_SIGNAL_SOURCE_HARDENING_ROUTE \
  --result-scope v1227_signal_source_hardening_top4 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --ablation-ids A-S2a-TemporalDriftStableFrame,A-S4c-CrossProjectionLowQuadDirect,A-S4a-CrossRandomProjectionFrame,A-S3b-BlockLocalEdgeEnergyFrame \
  --train-size 1024 --val-size 512 --test-size 512 \
  --epochs 8 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 64 --linec-sketch-dim 24 \
  2>&1 | tee results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs/v1227_linea_signal_source_hardening.log
```

结果摘要：

```text
rows = 54
summary_rows = 6
best hardening candidate = A-S3b-BlockLocalEdgeEnergyFrame
```

## 7. Line A signal-source repair depth

因 hardening 后 task-positive candidate 的 AUC-time 与 LineC 仍失败，执行 view-stability / block-stability / low-rank LineC residual repair。

执行命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_27_label_free_signal_source_functional_bridge/official_v1227 \
  --artifact-prefix v1227_label_free_signal_source_repair \
  --run-id v1227_label_free_signal_source_repair \
  --summary-stage V1227_LABEL_FREE_SIGNAL_SOURCE_REPAIR_SUMMARY \
  --route-stage V1227_LABEL_FREE_SIGNAL_SOURCE_REPAIR_ROUTE \
  --result-scope v1227_signal_source_repair_depth3 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --ablation-ids A-S5a-ViewStableLineCResidual,A-S5b-BlockStableReducedDirect,A-S5c-MultiViewBlockLowRank,A-S5d-CrossProjLineCResidual \
  --train-size 1024 --val-size 512 --test-size 512 \
  --epochs 8 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 64 --linec-sketch-dim 24 \
  2>&1 | tee results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs/v1227_linea_signal_source_repair.log
```

结果摘要：

```text
rows = 54
summary_rows = 6
best repair candidate = A-S5a-ViewStableLineCResidual
```

## 8. Line F shadow functional bridge

Line A 没有 near-anchor，因此 Line F 只能作为 shadow diagnostic，不能 promotion。实际 base：

```text
A-S3b-BlockLocalEdgeEnergyFrame
A-S5a-ViewStableLineCResidual
```

执行命令：

```bash
conda run -n kan python experiments/run_v1227_label_free_signal_source.py \
  --out-dir results/v12_27_label_free_signal_source_functional_bridge/official_v1227 \
  --artifact-prefix v1227_functional_bridge_shadow_or_official \
  --base-candidates A-S3b-BlockLocalEdgeEnergyFrame,A-S5a-ViewStableLineCResidual \
  --datasets KMNIST --seeds 0 \
  --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 8 --lr 0.0015 --weight-decay 0.001 \
  --compensation-batch 32 --ensemble-count 8 \
  --train-seed-bases 12270400,12271400,12272400 \
  --linec-seeds 12279500,12280600,12281600,12282600,12283600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --ref-mode bootstrap_mom --ref-seed-base 12272525 --train-seed-salt-override 39 \
  2>&1 | tee results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs/v1227_functional_bridge_shadow_or_official.log
```

结果摘要：

```text
candidate_rows = 54
aggregate_rows = 18
any_exploration_pass = 0
any_strict_majority_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
```

## 9. Line D Rational D34-D38 memory repair

执行命令：

```bash
conda run -n kan python experiments/run_v1224_classic_hardening.py \
  --out-dir results/v12_27_label_free_signal_source_functional_bridge/official_v1227 \
  --artifact-prefix v1227_rational_memory_repair \
  --run-id v1227_rational_memory_repair_D34_D38 \
  --families D34-RationalActivationCheckpointNoReadoutCache,D35-RationalGroupedWorkspaceReuse,D36-RationalReadoutGradRecompute,D37-RationalDenominatorStateFP16Audit,D38-RationalLowMemNoPairLineCRepeat \
  --datasets MNIST,Fashion-MNIST \
  --seeds 0 \
  --train-size 1024 --val-size 512 --epochs 8 --batch-size 128 \
  --linec-batch-size 64 --linec-sketch-dim 24 \
  2>&1 | tee results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs/v1227_rational_memory_repair.log
```

结果摘要：

```text
rows = 10
summary_rows = 5
exploration_pass_rows = 0
```

## 10. Finalizer

执行命令：

```bash
conda run -n kan python experiments/run_v1227_finalize_label_free_signal_source.py \
  2>&1 | tee results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs/v1227_finalize_initial.log
```

首次发现 finalizer 聚合 bug：

```text
1. MLP reference summary 被误纳入 A-S gate。
2. functional variant/multisketch rows 被误计为 candidate rows。
3. code packet 打包时误把正在写入的 zip 自身纳入 members。
```

修复文件：

```text
experiments/run_v1227_finalize_label_free_signal_source.py
```

复核命令：

```bash
python -m py_compile experiments/run_v1227_finalize_label_free_signal_source.py
conda run -n kan python experiments/run_v1227_finalize_label_free_signal_source.py \
  2>&1 | tee results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs/v1227_finalize_aggregate_fix.log
```

最终 route 摘要：

```text
route = R1-LabelFreeSignalSourceMissing
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
```

## 11. 日志写入后的最终打包复核

执行命令：

```bash
conda run -n kan python experiments/run_v1227_finalize_label_free_signal_source.py \
  2>&1 | tee results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs/v1227_finalize_after_logs.log
```

最终复核：

```text
route = R1-LabelFreeSignalSourceMissing
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
code_review_packet_entries = 56
code_review_packet_sha256 = 以最终 v1227_route_decision.json 为准
```

## 12. 用户继续要求后的 A-S6 pseudo-partition / affinity-anchor extension

继续原因：

```text
v12.27 当前 route = R1-LabelFreeSignalSourceMissing
official_success_reached = 0
line_a_near_anchor_pass_count = 0
line_f_exploration_gate_pass = 0
```

新增机制不再排列旧 multiview/block/cross-projection token，而是测试无标签 pseudo-partition / affinity-anchor signal source。

修改文件：

```text
dgkan/models/fc_purekan_primitives.py
experiments/run_v1218_b320_label_free_ablation.py
experiments/run_v1227_finalize_label_free_signal_source.py
```

新增候选：

```text
A-S6a-PseudoPartitionSignalFrame
A-S6b-PseudoPartitionLowQuadDirect
A-S6c-AffinityAnchorSignalFrame
A-S6d-RankConsensusSignalFrame
A-S6e-AffinityPseudoResidual
```

语法/provenance 检查：

```bash
python -m py_compile dgkan/models/fc_purekan_primitives.py experiments/run_v1218_b320_label_free_ablation.py experiments/run_v1227_finalize_label_free_signal_source.py
conda run -n kan python -c "from experiments.run_v1218_b320_label_free_ablation import ablation_specs; ..."
```

结果：

```text
A-S6a..A-S6e uses_y_for_stats=0
forbidden_token_present=0
```

Smoke command：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_27_label_free_signal_source_functional_bridge/official_v1227 \
  --artifact-prefix v1227_depth6_pseudo_affinity_smoke \
  --run-id v1227_depth6_pseudo_affinity_smoke \
  --summary-stage V1227_DEPTH6_PSEUDO_AFFINITY_SMOKE_SUMMARY \
  --route-stage V1227_DEPTH6_PSEUDO_AFFINITY_SMOKE_ROUTE \
  --result-scope v1227_depth6_pseudo_affinity_smoke \
  --datasets MNIST --seeds 0 \
  --ablation-ids A-S6a-PseudoPartitionSignalFrame,A-S6c-AffinityAnchorSignalFrame \
  --train-size 64 --val-size 32 --test-size 32 \
  --epochs 1 --batch-size 32 \
  --measure-linec 1 --linec-batch-size 16 --linec-sketch-dim 4 \
  2>&1 | tee results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs/v1227_depth6_pseudo_affinity_smoke.log
```

Smoke result：

```text
rows = 4
summary_rows = 4
best = A-S6a-PseudoPartitionSignalFrame
```

Scout command：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_27_label_free_signal_source_functional_bridge/official_v1227 \
  --artifact-prefix v1227_depth6_pseudo_affinity_scout \
  --run-id v1227_depth6_pseudo_affinity_scout \
  --summary-stage V1227_DEPTH6_PSEUDO_AFFINITY_SCOUT_SUMMARY \
  --route-stage V1227_DEPTH6_PSEUDO_AFFINITY_SCOUT_ROUTE \
  --result-scope v1227_depth6_pseudo_affinity_scout \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --ablation-ids A-S6a-PseudoPartitionSignalFrame,A-S6b-PseudoPartitionLowQuadDirect,A-S6c-AffinityAnchorSignalFrame,A-S6d-RankConsensusSignalFrame,A-S6e-AffinityPseudoResidual \
  --train-size 512 --val-size 256 --test-size 256 \
  --epochs 3 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 32 --linec-sketch-dim 8 \
  2>&1 | tee results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs/v1227_depth6_pseudo_affinity_scout.log
```

Scout result：

```text
rows = 63
summary_rows = 7
best = A-S6a-PseudoPartitionSignalFrame
```

Hardening command：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_27_label_free_signal_source_functional_bridge/official_v1227 \
  --artifact-prefix v1227_depth6_pseudo_affinity_hardening \
  --run-id v1227_depth6_pseudo_affinity_hardening \
  --summary-stage V1227_DEPTH6_PSEUDO_AFFINITY_HARDENING_SUMMARY \
  --route-stage V1227_DEPTH6_PSEUDO_AFFINITY_HARDENING_ROUTE \
  --result-scope v1227_depth6_pseudo_affinity_hardening_top3 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --ablation-ids A-S6a-PseudoPartitionSignalFrame,A-S6c-AffinityAnchorSignalFrame,A-S6b-PseudoPartitionLowQuadDirect \
  --train-size 1024 --val-size 512 --test-size 512 \
  --epochs 8 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 64 --linec-sketch-dim 24 \
  2>&1 | tee results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs/v1227_depth6_pseudo_affinity_hardening.log
```

Hardening result：

```text
rows = 45
summary_rows = 5
best = A-S6b-PseudoPartitionLowQuadDirect
```

## 13. A-S7 pseudo-affinity geometry guard repair

因 A-S6b task mean 为正但 worst/AUC/LineC 仍失败，继续执行 pseudo view/block geometry guard repair。

新增候选：

```text
A-S7a-PseudoViewMixReducedDirect
A-S7b-PseudoBlockGuardResidual
A-S7c-AffinityPseudoBlockReducedDirect
A-S7d-AffinityRankViewMix
```

Provenance result：

```text
A-S7a..A-S7d uses_y_for_stats=0
forbidden_token_present=0
```

Scout command：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_27_label_free_signal_source_functional_bridge/official_v1227 \
  --artifact-prefix v1227_depth7_pseudo_affinity_repair_scout \
  --run-id v1227_depth7_pseudo_affinity_repair_scout \
  --summary-stage V1227_DEPTH7_PSEUDO_AFFINITY_REPAIR_SCOUT_SUMMARY \
  --route-stage V1227_DEPTH7_PSEUDO_AFFINITY_REPAIR_SCOUT_ROUTE \
  --result-scope v1227_depth7_pseudo_affinity_repair_scout \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --ablation-ids A-S7a-PseudoViewMixReducedDirect,A-S7b-PseudoBlockGuardResidual,A-S7c-AffinityPseudoBlockReducedDirect,A-S7d-AffinityRankViewMix \
  --train-size 512 --val-size 256 --test-size 256 \
  --epochs 3 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 32 --linec-sketch-dim 8 \
  2>&1 | tee results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs/v1227_depth7_pseudo_affinity_repair_scout.log
```

Scout result：

```text
rows = 54
summary_rows = 6
best = A-S7b-PseudoBlockGuardResidual
```

## 14. A-S6 functional shadow recheck

执行命令：

```bash
conda run -n kan python experiments/run_v1227_label_free_signal_source.py \
  --out-dir results/v12_27_label_free_signal_source_functional_bridge/official_v1227 \
  --artifact-prefix v1227_depth6_pseudo_affinity_functional_shadow \
  --base-candidates A-S6b-PseudoPartitionLowQuadDirect,A-S6c-AffinityAnchorSignalFrame \
  --datasets KMNIST --seeds 0 \
  --train-size 512 --val-size 256 --batch-size 64 \
  --epochs 8 --lr 0.0015 --weight-decay 0.001 \
  --compensation-batch 32 --ensemble-count 8 \
  --train-seed-bases 12270400,12271400,12272400 \
  --linec-seeds 12279500,12280600,12281600,12282600,12283600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --ref-mode bootstrap_mom --ref-seed-base 12272525 --train-seed-salt-override 39 \
  2>&1 | tee results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs/v1227_depth6_pseudo_affinity_functional_shadow.log
```

结果：

```text
candidate_rows = 54
aggregate_rows = 18
any_exploration_pass = 0
any_strict_majority_pass = 0
combined_bridge_any_train_shuffle_robust_majority_pass = 0
```

## 15. Depth6/7 聚合复核

执行命令：

```bash
python -m py_compile experiments/run_v1227_finalize_label_free_signal_source.py
conda run -n kan python experiments/run_v1227_finalize_label_free_signal_source.py \
  2>&1 | tee results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs/v1227_finalize_after_depth6_depth7.log
```

聚合结果：

```text
route = R1-LabelFreeSignalSourceMissing
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
fallback_depth = 7
fallback_all_executed = 1
required_artifact_missing_count = 0
line_a_near_anchor_pass_count = 0
functional_candidate_rows = 108
functional_aggregate_rows = 36
```

## 16. Finalizer 幂等性修复与最终打包

问题：

```text
日志写入后再次运行 finalizer 时，v1227_functional_bridge_shadow_or_official.csv 已经包含 Depth 6 functional rows；
finalizer 又读取 v1227_depth6_pseudo_affinity_functional_shadow.csv 并合并，导致 functional rows 重复累计。
```

修复文件：

```text
experiments/run_v1227_finalize_label_free_signal_source.py
```

修复内容：

```text
新增 unique_rows()，对 finalizer 输入 rows 做幂等去重；
修复只影响聚合与 code packet 打包，不改变实验 CSV 中任何单行指标或 gate。
```

验证命令：

```bash
python -m py_compile experiments/run_v1227_finalize_label_free_signal_source.py && \
conda run -n kan python experiments/run_v1227_finalize_label_free_signal_source.py \
  2>&1 | tee results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs/v1227_finalize_after_depth6_depth7_dedupe.log
```

验证结果：

```text
route = R1-LabelFreeSignalSourceMissing
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
fallback_depth = 7
fallback_all_executed = 1
required_artifact_missing_count = 0
line_a_near_anchor_pass_count = 0
functional_candidate_rows = 108
functional_aggregate_rows = 36
precommit_value_source_rows = 108
```

最终日志写入后打包命令：

```bash
conda run -n kan python experiments/run_v1227_finalize_label_free_signal_source.py \
  2>&1 | tee results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs/v1227_finalize_after_depth6_depth7_logs_final.log
```

章节编号整理后最终打包命令：

```bash
conda run -n kan python experiments/run_v1227_finalize_label_free_signal_source.py \
  2>&1 | tee results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs/v1227_finalize_after_depth6_depth7_logs_final4.log
```

最终打包结果：

```text
route = R1-LabelFreeSignalSourceMissing
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 7
fallback_all_executed = 1
required_artifact_missing_count = 0
line_a_near_anchor_pass_count = 0
line_f_exploration_gate_pass = 0
functional_candidate_rows = 108
functional_aggregate_rows = 36
precommit_value_source_rows = 108
code_review_packet_entries = 70
code_review_packet_sha256 = 以最终 v1227_route_decision.json 为准
```

## 17. 用户再次追问后的 stop-contract 只读复核

复核命令：

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('results/v12_27_label_free_signal_source_functional_bridge/official_v1227/v1227_route_decision.json')
r=json.loads(p.read_text())
for k in ['route','minimum_success','official_success_reached','p4_pass','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','line_a_near_anchor_pass_count','line_f_exploration_gate_pass','functional_candidate_rows','functional_aggregate_rows','precommit_value_source_rows','code_review_packet_entries','code_review_packet_sha256']:
    print(f'{k}={r.get(k)}')
PY
```

复核结果：

```text
route = R1-LabelFreeSignalSourceMissing
minimum_success = Minimum Success E
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 7
fallback_all_executed = 1
required_artifact_missing_count = 0
line_a_near_anchor_pass_count = 0
line_f_exploration_gate_pass = 0
functional_candidate_rows = 108
functional_aggregate_rows = 36
precommit_value_source_rows = 108
code_review_packet_entries = 70
```

本次仅做 stop-contract 复核；没有新增实验。原因记录在实验结果复盘第 13 节：当前已经不确定继续扩展同一 pseudo-partition / affinity-anchor / rank-consensus / viewmix / blockguard family 是否有有效机制，继续排列 token 会变成低价值网格搜索。

日志更新后最终打包命令：

```bash
conda run -n kan python experiments/run_v1227_finalize_label_free_signal_source.py \
  2>&1 | tee results/v12_27_label_free_signal_source_functional_bridge/official_v1227/logs/v1227_finalize_after_stop_contract_recheck.log
```
