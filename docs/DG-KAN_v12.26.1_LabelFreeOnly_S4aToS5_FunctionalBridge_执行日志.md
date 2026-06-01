# DG-KAN v12.26.1 LabelFreeOnly S4aToS5 FunctionalBridge 执行日志

生成时间：2026-05-26（Asia/Singapore）

本日志只记录实际执行过的命令、文件路径和产物位置，方便后续复现。不把未执行命令写成已执行。

## 1. 计划读取与代码定位

读取计划：

```bash
sed -n '260,620p' docs/DG-KAN_v12.26.1_LabelFreeOnly_S4aToS5_FunctionalBridge_完整计划.md
sed -n '620,1040p' docs/DG-KAN_v12.26.1_LabelFreeOnly_S4aToS5_FunctionalBridge_完整计划.md
sed -n '1040,1420p' docs/DG-KAN_v12.26.1_LabelFreeOnly_S4aToS5_FunctionalBridge_完整计划.md
```

定位旧代码中 label/probe/init 与 v12.25 functional bridge 路径：

```bash
rg -n "y_for_stats|trainprobe|TrainProbe|probe_dirs|build_y_stats|SimpleFastTaskGeometryKAN|direct_readout|quad_proj|signalBroad|signalBlock" dgkan experiments/run_v1218_b320_label_free_ablation.py experiments/run_v1225_composite_functional_bridge.py experiments/run_v1225_finalize_precommit_functional.py
rg -n "def main|argparse|candidate|CANDIDATE|A51|A70|summary|out" experiments/run_v1218_b320_label_free_ablation.py
rg -n "def run_candidate|def main|argparse|CANDIDATES|apply_train_feature|bootstrap_mom|post_cal|weight_anchor|linec|route|controls|MLP" experiments/run_v1225_composite_functional_bridge.py experiments/run_v1225_finalize_precommit_functional.py
```

结论：旧 A51/A70-A76 虽然 `uses_y_for_stats=0`，但 `init_variant` 仍保留 trainprobe token，不能作为 v12.26.1 official/exploration candidate。

## 2. 代码修改

修改文件：

```text
experiments/run_v1218_b320_label_free_ablation.py
experiments/run_v1226_label_free_only_bridge.py
experiments/run_v1226_finalize_label_free_only.py
```

修改后语法检查：

```bash
python -m py_compile experiments/run_v1218_b320_label_free_ablation.py experiments/run_v1226_label_free_only_bridge.py experiments/run_v1226_finalize_label_free_only.py
```

结果：通过。

## 3. 环境检查

```bash
mkdir -p results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs
conda run -n kan python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
```

输出：

```text
cuda_available= True
device_count= 4
```

## 4. Depth 1 / Line A scout

运行指令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only \
  --artifact-prefix v1226_label_free_base_scout \
  --run-id v1226_label_free_base_scout \
  --summary-stage V1226_LABEL_FREE_BASE_SCOUT_SUMMARY \
  --route-stage V1226_LABEL_FREE_BASE_SCOUT_ROUTE \
  --result-scope v1226_depth1_scout \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --ablation-ids A-LF0-BaseNoProbe,A-LF1-OrthoBank,A-LF2-CovFrame,A-LF3-AugStable,A-LF4-ResidualLowRank,A-LF5-RoleEnergyBalance,A-LF6-CouplingAware,A-LF7-EMACovAdapt \
  --train-size 512 --val-size 256 --test-size 256 \
  --epochs 3 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 32 --linec-sketch-dim 8 \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_linea_scout.log
```

Scout 自动 top-3 只读命令：

```bash
python - <<'PY'
import csv, math
from pathlib import Path
p=Path('results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/v1226_label_free_base_scout_summary.csv')
rows=list(csv.DictReader(p.open()))
def f(x,d=-999):
    try:
        return float(x) if x not in ('',None) and math.isfinite(float(x)) else d
    except Exception:
        return d
c=[r for r in rows if r['candidate_id'].startswith('A-LF')]
for r in sorted(c,key=lambda r:(f(r.get('mean_delta_vs_mlp')), f(r.get('linec_nontearing_pass_rate'))), reverse=True):
    print(r['candidate_id'], 'mean_delta_vs_mlp=', r.get('mean_delta_vs_mlp'), 'worst=', r.get('worst_delta_vs_mlp'), 'auc_time=', r.get('max_AUC_time_ratio_vs_mlp'), 'linec_rate=', r.get('linec_nontearing_pass_rate'))
print('TOP3=', ','.join([r['candidate_id'] for r in sorted(c,key=lambda r:(f(r.get('mean_delta_vs_mlp')), f(r.get('linec_nontearing_pass_rate'))), reverse=True)[:3]]))
PY
```

Scout top-3：

```text
A-LF0-BaseNoProbe,A-LF1-OrthoBank,A-LF4-ResidualLowRank
```

## 5. Depth 1 / Line A hardening

运行指令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only \
  --artifact-prefix v1226_label_free_base_hardening \
  --run-id v1226_label_free_base_hardening_top3 \
  --summary-stage V1226_LABEL_FREE_BASE_HARDENING_SUMMARY \
  --route-stage V1226_LABEL_FREE_BASE_HARDENING_ROUTE \
  --result-scope v1226_depth1_hardening_top3 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --ablation-ids A-LF0-BaseNoProbe,A-LF1-OrthoBank,A-LF4-ResidualLowRank \
  --train-size 1024 --val-size 512 --test-size 512 \
  --epochs 8 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 64 --linec-sketch-dim 24 \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_linea_hardening.log
```

Hardening top2 只读命令：

```bash
python - <<'PY'
import csv, math
from pathlib import Path
p=Path('results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/v1226_label_free_base_hardening_summary.csv')
rows=list(csv.DictReader(p.open()))
def f(x,d=-999):
    try:
        return float(x) if x not in ('',None) and math.isfinite(float(x)) else d
    except Exception:
        return d
for r in [x for x in rows if x['candidate_id'].startswith('A-LF')]:
    print(r['candidate_id'], 'mean_delta_vs_mlp=', r.get('mean_delta_vs_mlp'), 'worst=', r.get('worst_delta_vs_mlp'), 'auc_time=', r.get('max_AUC_time_ratio_vs_mlp'), 'auc_step=', r.get('max_AUC_step_ratio_vs_mlp'), 'linec_rate=', r.get('linec_nontearing_pass_rate'), 'all=', r.get('linec_nontearing_all_pass'))
print('TOP2=', ','.join([r['candidate_id'] for r in sorted([x for x in rows if x['candidate_id'].startswith('A-LF')], key=lambda r:(f(r.get('mean_delta_vs_mlp')), f(r.get('linec_nontearing_pass_rate'))), reverse=True)[:2]]))
PY
```

Hardening top2：

```text
A-LF0-BaseNoProbe,A-LF1-OrthoBank
```

## 6. Depth 1 repair / LineC-oriented hardening

原因：A-LF0/A-LF1 task 接近但 LineC 失败，按计划 4.7 补跑 residual/role-energy/coupling/EMA repair。

运行指令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only \
  --artifact-prefix v1226_label_free_base_linec_repair \
  --run-id v1226_label_free_base_linec_repair \
  --summary-stage V1226_LABEL_FREE_BASE_LINEC_REPAIR_SUMMARY \
  --route-stage V1226_LABEL_FREE_BASE_LINEC_REPAIR_ROUTE \
  --result-scope v1226_depth1_linec_repair \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --ablation-ids A-LF4-ResidualLowRank,A-LF5-RoleEnergyBalance,A-LF6-CouplingAware,A-LF7-EMACovAdapt \
  --train-size 1024 --val-size 512 --test-size 512 \
  --epochs 8 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 64 --linec-sketch-dim 24 \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_linea_linec_repair.log
```

补充语法/registry 检查：

```bash
python -m py_compile experiments/run_v1226_label_free_only_bridge.py
conda run -n kan python -c "import experiments.run_v1226_label_free_only_bridge as x; print(','.join(x.v1226_candidates().keys()))"
```

## 7. Depth 2 / Functional S4a transfer

原因：Line A 未达 near-anchor，但计划禁止早停；继续在 top-2 label-free base 上做 shadow functional transfer diagnostic。

运行指令：

```bash
conda run -n kan python experiments/run_v1226_label_free_only_bridge.py \
  --out-dir results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only \
  --artifact-prefix v1226_functional_depth2_transfer \
  --base-candidates A-LF0-BaseNoProbe,A-LF1-OrthoBank \
  --datasets KMNIST --seeds 0 \
  --candidates F26-D2-directGainTask,F26-D2-lowRankQuadGuard,F26-D2-crossRefQuadGuard,F26-D2-compositeTaskGuard,F26-D2-postCalWeightAnchor,F26-D2-controlResidualComposite \
  --train-size 512 --val-size 256 --epochs 8 --batch-size 64 \
  --train-seed-bases 12260400,12261400,12262400 \
  --linec-seeds 12259500,12260600,12261600,12262600,12263600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --ref-mode bootstrap_mom --ref-seed-base 12262525 --train-seed-salt-override 39 \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_functional_depth2_transfer.log
```

## 8. Depth 3 / Geometry-risk-aware policy

原因：Depth 2 出现 task/control 与 LineC 分离，继续执行计划要求的 veto/downscale/role-switch/sequential policy。

运行指令：

```bash
conda run -n kan python experiments/run_v1226_label_free_only_bridge.py \
  --out-dir results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only \
  --artifact-prefix v1226_functional_depth3_risk_policy \
  --base-candidates A-LF0-BaseNoProbe,A-LF1-OrthoBank \
  --datasets KMNIST --seeds 0 \
  --candidates F26-D3-riskAwareTailMix,F26-D3-roleSwitchFreezeDirect,F26-D3-sequentialGeometryTask,F26-D3-downscaleQuadPolicy,F26-D3-partialResponseRiskVeto \
  --train-size 512 --val-size 256 --epochs 8 --batch-size 64 \
  --train-seed-bases 12260400,12261400,12262400 \
  --linec-seeds 12259500,12260600,12261600,12262600,12263600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --ref-mode bootstrap_mom --ref-seed-base 12262525 --train-seed-salt-override 39 \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_functional_depth3_risk_policy.log
```

## 9. Depth 4 / Primitive replacement

原因：Depth 3 仍未 co-locate task/control 与 LineC majority，继续执行计划要求的机制替换。

运行指令：

```bash
conda run -n kan python experiments/run_v1226_label_free_only_bridge.py \
  --out-dir results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only \
  --artifact-prefix v1226_functional_depth4_primitive_replacement \
  --base-candidates A-LF0-BaseNoProbe,A-LF1-OrthoBank \
  --datasets KMNIST --seeds 0 \
  --candidates F26-D4-roleGainTransport,F26-D4-quadReservoirGuard,F26-D4-lowRankLogitSubspace,F26-D4-unlabeledCovarianceTransport \
  --train-size 512 --val-size 256 --epochs 8 --batch-size 64 \
  --train-seed-bases 12260400,12261400,12262400 \
  --linec-seeds 12259500,12260600,12261600,12262600,12263600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --ref-mode bootstrap_mom --ref-seed-base 12262525 --train-seed-salt-override 39 \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_functional_depth4_primitive_replacement.log
```

## 10. Finalizer 聚合

运行指令：

```bash
python -m py_compile experiments/run_v1226_finalize_label_free_only.py
conda run -n kan python experiments/run_v1226_finalize_label_free_only.py \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_finalize.log
```

首次聚合输出摘要：

```text
route = R1-LabelFreeBaseMissing
minimum_success = Minimum Success E
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 4
fallback_rows = 4
fallback_all_executed = 1
required_artifact_rows = 31
required_artifact_missing_count = 0
forbidden_token_official_count = 0
uses_y_for_stats_official_count = 0
uses_label_for_init_count = 0
uses_ce_for_direction_count = 0
uses_query_batch_for_direction_count = 0
code_semantics_review_pass = 1
```

Line A 聚合摘要：

```text
line_a_scout_rows = 90
line_a_hardening_rows = 45
line_a_candidate_summary_rows = 7
line_a_best_candidate = A-LF4-ResidualLowRank
line_a_best_mean_delta_vs_mlp = -0.008246527777777778
line_a_best_worst_delta_vs_mlp = -0.044921875
line_a_best_linec_rate = 0.2222222222222222
line_a_near_anchor_pass_count = 0
line_a_official_pass_count = 0
```

Functional 聚合摘要：

```text
functional_candidate_rows = 90
functional_aggregate_rows = 30
line_f_exploration_gate_pass = 0
line_f_official_gate_pass = 0
precommit_value_source_rows = 90
support_count = 0
task_positive_rows = 3
linec_majority_rows = 5
s4a_single_positive_rows = 0
linec_multisketch_rows = 1110
linec_multisketch_pass_rows = 79
```

Code packet：

```text
path = results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/v1226_code_review_packet.zip
entries = 57
sha256 = 以最终 v1226_route_decision.json 为准
```

## 11. 日志写入后的最终复核

目的：把本执行日志和实验结果复盘日志纳入 code review packet 后重新打包。

运行指令：

```bash
conda run -n kan python experiments/run_v1226_finalize_label_free_only.py \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_finalize_after_logs.log
```

复核口径：

```text
最终 route、required manifest、code packet entries 和 sha256 以该命令完成后的
results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/v1226_route_decision.json 为准。
```

## 12. 用户继续要求后的 Depth 5 / label-free base geometry redesign

原因：最终 route 仍为 `R1-LabelFreeBaseMissing`，不能写 S5。按复盘建议，不继续盲目扩 functional grid，先重构 label-free base geometry。

代码修改：

```text
experiments/run_v1218_b320_label_free_ablation.py
experiments/run_v1226_finalize_label_free_only.py
```

新增 A-LF8..A-LF16：

```text
A-LF8-LowQuadOrtho
A-LF9-LowQuadBoundQ
A-LF10-LowQuadDirectRead
A-LF11-MultiFrameBank
A-LF12-MultiFrameDirect
A-LF13-ConvexFrameMix
A-LF14-SelfCondStopGrad
A-LF15-RoleCondDirect
A-LF16-FrozenBranchGain
```

语法和 provenance 检查：

```bash
python -m py_compile experiments/run_v1218_b320_label_free_ablation.py experiments/run_v1226_finalize_label_free_only.py experiments/run_v1226_label_free_only_bridge.py
conda run -n kan python -c "from experiments.run_v1218_b320_label_free_ablation import ablation_specs; from experiments.run_v1226_label_free_only_bridge import forbidden_token_present; specs=ablation_specs(784,10); print([k for k in specs if k.startswith('A-LF8') or k.startswith('A-LF16')]); ..."
```

结果：A-LF8..A-LF16 均 `uses_y_for_stats=0`，`forbidden=0`。

Scout 指令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only \
  --artifact-prefix v1226_label_free_base_depth5_geometry_scout \
  --run-id v1226_label_free_base_depth5_geometry_scout \
  --summary-stage V1226_LABEL_FREE_BASE_DEPTH5_GEOMETRY_SCOUT_SUMMARY \
  --route-stage V1226_LABEL_FREE_BASE_DEPTH5_GEOMETRY_SCOUT_ROUTE \
  --result-scope v1226_depth5_geometry_scout \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --ablation-ids A-LF8-LowQuadOrtho,A-LF9-LowQuadBoundQ,A-LF10-LowQuadDirectRead,A-LF11-MultiFrameBank,A-LF12-MultiFrameDirect,A-LF13-ConvexFrameMix,A-LF14-SelfCondStopGrad,A-LF15-RoleCondDirect,A-LF16-FrozenBranchGain \
  --train-size 512 --val-size 256 --test-size 256 \
  --epochs 3 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 32 --linec-sketch-dim 8 \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_linea_depth5_geometry_scout.log
```

Hardening 指令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only \
  --artifact-prefix v1226_label_free_base_depth5_geometry_hardening \
  --run-id v1226_label_free_base_depth5_geometry_hardening_top3 \
  --summary-stage V1226_LABEL_FREE_BASE_DEPTH5_GEOMETRY_HARDENING_SUMMARY \
  --route-stage V1226_LABEL_FREE_BASE_DEPTH5_GEOMETRY_HARDENING_ROUTE \
  --result-scope v1226_depth5_geometry_hardening_top3 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --ablation-ids A-LF13-ConvexFrameMix,A-LF15-RoleCondDirect,A-LF10-LowQuadDirectRead \
  --train-size 1024 --val-size 512 --test-size 512 \
  --epochs 8 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 64 --linec-sketch-dim 24 \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_linea_depth5_geometry_hardening.log
```

## 13. Depth 6 / A-LF10 task-anchor LineC repair

原因：Depth 5 hardening 中 A-LF10 有正 task delta，但 LineC/worst/AUC 仍 fail，因此继续按机制思路叠加 residual/boundQ/covadapt/role-conditioned geometry stabilizer。

代码修改：

```text
experiments/run_v1218_b320_label_free_ablation.py
experiments/run_v1226_finalize_label_free_only.py
```

新增 A-LF17..A-LF22：

```text
A-LF17-LowQuadDirectResidual
A-LF18-LowQuadDirectBoundQ
A-LF19-LowQuadDirectCovAdapt
A-LF20-LowQuadRoleCondDirect
A-LF21-LowQuadConvexDirect
A-LF22-LowQuadSelfCondDirect
```

Scout 指令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only \
  --artifact-prefix v1226_label_free_base_depth6_a_lf10_repair_scout \
  --run-id v1226_label_free_base_depth6_a_lf10_repair_scout \
  --summary-stage V1226_LABEL_FREE_BASE_DEPTH6_A_LF10_REPAIR_SCOUT_SUMMARY \
  --route-stage V1226_LABEL_FREE_BASE_DEPTH6_A_LF10_REPAIR_SCOUT_ROUTE \
  --result-scope v1226_depth6_a_lf10_repair_scout \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --ablation-ids A-LF17-LowQuadDirectResidual,A-LF18-LowQuadDirectBoundQ,A-LF19-LowQuadDirectCovAdapt,A-LF20-LowQuadRoleCondDirect,A-LF21-LowQuadConvexDirect,A-LF22-LowQuadSelfCondDirect \
  --train-size 512 --val-size 256 --test-size 256 \
  --epochs 3 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 32 --linec-sketch-dim 8 \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_linea_depth6_a_lf10_repair_scout.log
```

Hardening 指令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only \
  --artifact-prefix v1226_label_free_base_depth6_a_lf10_repair_hardening \
  --run-id v1226_label_free_base_depth6_a_lf10_repair_hardening_top3 \
  --summary-stage V1226_LABEL_FREE_BASE_DEPTH6_A_LF10_REPAIR_HARDENING_SUMMARY \
  --route-stage V1226_LABEL_FREE_BASE_DEPTH6_A_LF10_REPAIR_HARDENING_ROUTE \
  --result-scope v1226_depth6_a_lf10_repair_hardening_top3 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --ablation-ids A-LF21-LowQuadConvexDirect,A-LF22-LowQuadSelfCondDirect,A-LF19-LowQuadDirectCovAdapt \
  --train-size 1024 --val-size 512 --test-size 512 \
  --epochs 8 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 64 --linec-sketch-dim 24 \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_linea_depth6_a_lf10_repair_hardening.log
```

新 base functional 复验指令：

```bash
conda run -n kan python experiments/run_v1226_label_free_only_bridge.py \
  --out-dir results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only \
  --artifact-prefix v1226_functional_depth6_new_base_transfer \
  --base-candidates A-LF10-LowQuadDirectRead,A-LF21-LowQuadConvexDirect \
  --datasets KMNIST --seeds 0 \
  --candidates F26-D2-controlResidualComposite,F26-D3-downscaleQuadPolicy,F26-D4-lowRankLogitSubspace,F26-D4-unlabeledCovarianceTransport \
  --train-size 512 --val-size 256 --epochs 8 --batch-size 64 \
  --train-seed-bases 12260400,12261400,12262400 \
  --linec-seeds 12259500,12260600,12261600,12262600,12263600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --ref-mode bootstrap_mom --ref-seed-base 12262525 --train-seed-salt-override 39 \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_functional_depth6_new_base_transfer.log
```

## 14. Depth 6 后 finalizer 复核

运行指令：

```bash
conda run -n kan python experiments/run_v1226_finalize_label_free_only.py \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_finalize_after_depth6.log
```

复核口径：

```text
最终 route、manifest 和 code packet sha256 以该命令完成后的 v1226_route_decision.json 为准。
```

## 15. 用户继续要求后的 Depth 7 / label-free signal-frame estimator

原因：Depth 6 后仍为 `R1-LabelFreeBaseMissing`。继续推进时不再沿 low-quad/direct-read 小组合扩展，而是测试更接近“替代 trainprobe signal frame”的无标签输入几何 estimator。

代码修改：

```text
experiments/run_v1218_b320_label_free_ablation.py
experiments/run_v1226_finalize_label_free_only.py
```

新增 A-LF23..A-LF30：

```text
A-LF23-PCAOrthoMix
A-LF24-RandomCotangentStable
A-LF25-BlockLocalAugStable
A-LF26-AugTangentFrame
A-LF27-DriftCotangentBank
A-LF28-LowQuadPCAOrthoDirect
A-LF29-LowQuadAugTangentDirect
A-LF30-LowQuadCotangentDirect
```

审计说明：

```text
没有使用 optframep，因为它会从 supervised optimizer update 形成 projector adaptation，存在 CE-derived direction 风险。
A-LF23..A-LF30 均只使用初始化阶段的 train-stream x 几何，不读 label/CE/query。
```

检查指令：

```bash
python -m py_compile experiments/run_v1218_b320_label_free_ablation.py experiments/run_v1226_finalize_label_free_only.py
conda run -n kan python -c "from experiments.run_v1218_b320_label_free_ablation import ablation_specs; from experiments.run_v1226_label_free_only_bridge import forbidden_token_present; specs=ablation_specs(784,10); ..."
```

检查结果：A-LF23..A-LF30 均 `uses_y_for_stats=0`，`forbidden=0`。

Scout 指令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only \
  --artifact-prefix v1226_label_free_base_depth7_signal_frame_scout \
  --run-id v1226_label_free_base_depth7_signal_frame_scout \
  --summary-stage V1226_LABEL_FREE_BASE_DEPTH7_SIGNAL_FRAME_SCOUT_SUMMARY \
  --route-stage V1226_LABEL_FREE_BASE_DEPTH7_SIGNAL_FRAME_SCOUT_ROUTE \
  --result-scope v1226_depth7_signal_frame_scout \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --ablation-ids A-LF23-PCAOrthoMix,A-LF24-RandomCotangentStable,A-LF25-BlockLocalAugStable,A-LF26-AugTangentFrame,A-LF27-DriftCotangentBank,A-LF28-LowQuadPCAOrthoDirect,A-LF29-LowQuadAugTangentDirect,A-LF30-LowQuadCotangentDirect \
  --train-size 512 --val-size 256 --test-size 256 \
  --epochs 3 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 32 --linec-sketch-dim 8 \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_linea_depth7_signal_frame_scout.log
```

Hardening 指令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only \
  --artifact-prefix v1226_label_free_base_depth7_signal_frame_hardening \
  --run-id v1226_label_free_base_depth7_signal_frame_hardening_top3 \
  --summary-stage V1226_LABEL_FREE_BASE_DEPTH7_SIGNAL_FRAME_HARDENING_SUMMARY \
  --route-stage V1226_LABEL_FREE_BASE_DEPTH7_SIGNAL_FRAME_HARDENING_ROUTE \
  --result-scope v1226_depth7_signal_frame_hardening_top3 \
  --datasets MNIST,Fashion-MNIST,KMNIST \
  --seeds 0,1,2 \
  --ablation-ids A-LF25-BlockLocalAugStable,A-LF24-RandomCotangentStable,A-LF23-PCAOrthoMix \
  --train-size 1024 --val-size 512 --test-size 512 \
  --epochs 8 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 64 --linec-sketch-dim 24 \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_linea_depth7_signal_frame_hardening.log
```

Depth 7 functional 复验指令：

```bash
conda run -n kan python experiments/run_v1226_label_free_only_bridge.py \
  --out-dir results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only \
  --artifact-prefix v1226_functional_depth7_signal_frame_transfer \
  --base-candidates A-LF25-BlockLocalAugStable,A-LF24-RandomCotangentStable \
  --datasets KMNIST --seeds 0 \
  --candidates F26-D2-controlResidualComposite,F26-D3-downscaleQuadPolicy,F26-D4-lowRankLogitSubspace,F26-D4-unlabeledCovarianceTransport \
  --train-size 512 --val-size 256 --epochs 8 --batch-size 64 \
  --train-seed-bases 12260400,12261400,12262400 \
  --linec-seeds 12259500,12260600,12261600,12262600,12263600 \
  --linec-batch 32 --linec-sketch-dim 8 \
  --ref-mode bootstrap_mom --ref-seed-base 12262525 --train-seed-salt-override 39 \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_functional_depth7_signal_frame_transfer.log
```

## 16. Depth 7 后 finalizer 复核

运行指令：

```bash
conda run -n kan python experiments/run_v1226_finalize_label_free_only.py \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_finalize_after_depth7.log
```

## 17. 用户再次追问后的 stop-contract 复核

原因：用户再次询问是否达成目标，并要求未达成则继续；但上一节 route 已经显示 `final_stop_allowed=1`、`hard_compute_budget_exhausted=1`、`fallback_depth=7`。因此本轮先复核计划 stop 条款与最终 route，确认是否还有明确未执行 fallback。

复核计划 stop/fallback 条款：

```bash
rg -n "Depth|fallback|stop|hard|停止|继续|下一步|budget|final_stop|R1|Minimum" \
  docs/DG-KAN_v12.26.1_LabelFreeOnly_S4aToS5_FunctionalBridge_完整计划.md
```

复核 route：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/v1226_route_decision.json")
print(json.dumps(json.loads(p.read_text()), indent=2, ensure_ascii=False))
PY
```

复核结论：

```text
计划 mandatory fallback depth = 1..4。
当前实际 fallback_depth = 7，已超过 mandatory depth。
route = R1-LabelFreeBaseMissing
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
required_artifact_missing_count = 0
line_a_near_anchor_pass_count = 0
line_f_exploration_gate_pass = 0
support_count = 0
```

判断：当前不是“未执行完就退出”，而是 Depth 7 后的 hard-stop no-go。继续排列现有 A-LF projection/PCA/cotangent/augmentation token family 没有明确计划依据；我已经不确定如何在不重新设计新 label-free signal source 的情况下继续做有效修复。

本节写入后重新打包：

```bash
conda run -n kan python experiments/run_v1226_finalize_label_free_only.py \
  2>&1 | tee results/v12_26_1_label_free_only_s4a_to_s5/official_label_free_only/logs/v1226_finalize_after_stop_contract_recheck.log
```
