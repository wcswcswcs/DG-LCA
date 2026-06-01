# DG-KAN v12.31 BasisKernelFirst MLPFunctionalNoGoTest 执行日志

生成时间：2026-05-27（Asia/Singapore）

本日志只记录实际执行过的命令、文件与产物路径；不补写未执行命令。

## 1. 计划入口

计划文档：

```text
docs/DG-KAN_v12.31_BasisKernelFirst_MLPFunctionalNoGoTest_完整计划.md
```

结果目录：

```text
results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231
```

## 2. 本轮代码修改记录

当前已修改/新增文件：

```text
dgkan/diagnostics/basis_workspace.py
dgkan/functional/mlp_functional.py
experiments/run_v1231_basis_kernel_workspace.py
experiments/run_v1231_finalize_basis_kernel_first.py
docs/DG-KAN_v12.31_BasisKernelFirst_MLPFunctionalNoGoTest_执行日志.md
docs/DG-KAN_v12.31_BasisKernelFirst_MLPFunctionalNoGoTest_实验结果复盘.md
```

## 3. 待执行命令

后续 sections 将按实际执行顺序写入：

```text
1. py_compile / provenance import audit
2. smoke checks
3. Line D basis workspace truth audit
4. Line M M-G MLP functional no-go test
5. Line A A-DYN monitor
6. finalizer / route / packet
```

## 4. 语法检查与 provenance import audit

执行命令：

```bash
python -m py_compile dgkan/diagnostics/basis_workspace.py dgkan/functional/mlp_functional.py experiments/run_v1231_basis_kernel_workspace.py experiments/run_v1231_finalize_basis_kernel_first.py experiments/run_v1230_mlp_functional.py experiments/run_v1218_b320_label_free_ablation.py
```

结果：

```text
py_compile pass
```

执行命令：

```bash
python - <<'PY'
from dgkan.diagnostics.basis_workspace import V1231_BASIS_CANDIDATES
...
PY
```

结果：

```text
failed: ModuleNotFoundError: No module named 'torch'
```

解释：系统默认 `python` 没有安装 torch；后续 provenance audit 与实验均使用 `conda run -n kan python`。

复核命令：

```bash
conda run -n kan python -c "from dgkan.diagnostics.basis_workspace import V1231_BASIS_CANDIDATES; from dgkan.functional.mlp_functional import SOURCE_CANDIDATES, CONTROL_IDS; from experiments.run_v1218_b320_label_free_ablation import ablation_specs; from experiments.run_v1226_label_free_only_bridge import forbidden_token_present; ..."
```

结果：

```text
basis_candidates = 22
basis_exact_kernel_sum = 0
mg_candidates = 5
controls = 5
adyn = 5
adyn_uses_y_for_stats = 0
adyn_forbidden = 0
```

## 5. Smoke checks

创建结果目录：

```bash
mkdir -p results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs
```

### 5.1 Basis workspace smoke

执行命令：

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_basis_smoke \
  --run-id v1231_basis_smoke \
  --candidates D-RAT7-FusedGroupRationalNoMaterialize,D-WAV6-HatWaveletNoMaterializeLocal2 \
  --datasets MNIST --seeds 0 \
  --train-size 64 --val-size 32 --batch-size 32 \
  --hardening-epochs 1 --mlp-reference-epochs 1 \
  --linec-batch-size 16 --linec-sketch-dim 4 --linec-seeds 12319500 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_basis_smoke.log
```

结果：

```text
workspace_rows = 2
workspace_gate_pass_rows = 1
workspace_strong_gate_pass_rows = 1
hardening_rows = 2
hardening_executed_rows = 1
family_near_pass_rows = 0
```

### 5.2 M-G MLP functional smoke

执行命令：

```bash
conda run -n kan python experiments/run_v1230_mlp_functional.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_mlp_functional_smoke \
  --candidates M-G1-UnlabeledCloneResponseCov \
  --datasets MNIST --seeds 0 --train-seed-bases 12310400 \
  --train-size 64 --val-size 32 --epochs 1 --batch-size 32 \
  --functional-batch 32 \
  --linec-batch-size 16 --linec-sketch-dim 4 \
  --linec-seeds 12319500,12320600 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_mlp_functional_smoke.log
```

结果：

```text
candidate_rows = 1
control_rows = 6
linec_rows = 4
gate_rows = 1
any_exploration_pass = 0
any_official_pass = 0
```

### 5.3 A-DYN monitor smoke

执行命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_adyn_monitor_smoke \
  --run-id v1231_adyn_monitor_smoke \
  --summary-stage V1231_ADYN_MONITOR_SMOKE_SUMMARY \
  --route-stage V1231_ADYN_MONITOR_SMOKE_ROUTE \
  --result-scope v1231_adyn_monitor_smoke \
  --datasets MNIST --seeds 0 \
  --ablation-ids A-DYN1-LearnableSignalFrameWarmup,A-DYN2-EarlySelfPredictiveFrame \
  --train-size 64 --val-size 32 --test-size 32 \
  --epochs 1 --batch-size 32 \
  --measure-linec 1 --linec-batch-size 16 --linec-sketch-dim 4 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_adyn_monitor_smoke.log
```

结果：

```text
rows = 4
summary_rows = 4
best_label_free_candidate = A-DYN1-LearnableSignalFrameWarmup
smoke_not_official = 1
```

## 6. Line D basis workspace truth audit

### 6.1 首次 official run

执行命令：

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_basis \
  --run-id v1231_basis_workspace_official \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 512 --val-size 256 --batch-size 128 \
  --hardening-epochs 3 --mlp-reference-epochs 1 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  --linec-seeds 12319500,12320600,12321600 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_basis_workspace_official.log
```

首次结果：

```text
workspace_rows = 198
workspace_gate_pass_rows = 140
workspace_strong_gate_pass_rows = 64
hardening_rows = 198
hardening_executed_rows = 140
family_near_pass_rows = 48
```

发现 blocker：

```text
1. MLP task reference 只跑了 1 个 epoch，而 basis hardening 跑 3 个 epoch，task delta 不公平。
2. workspace step 直接测首步，可能混入 first-call compile / lazy allocation。
3. row-level family_near_pass 缺少跨 dataset/seed worst/AUC 聚合，不能作为 FamilyNearPass。
```

修复文件：

```text
experiments/run_v1231_basis_kernel_workspace.py
experiments/run_v1231_finalize_basis_kernel_first.py
```

修复内容：

```text
1. MLP workspace profile 与 MLP task reference 分离；task reference 使用 fresh MLP 并训练同等 hardening_epochs。
2. basis / MLP workspace phase 测量前增加 workspace_warmup_steps，降低首步编译/初始化口径污染。
3. hardening row 只写 task_linec_probe_pass；family_near_pass 固定为 0，由 finalizer 聚合且注明 AUC_time 未测不能 promotion。
4. finalizer route 修正：workspace 已打开但 task/LineC 未闭合时 route 为 R2-BasisTaskLineCNotColocated，同时 minimum_success 保留 S1-BasisWorkspaceOpened。
```

语法复核：

```bash
python -m py_compile experiments/run_v1231_basis_kernel_workspace.py experiments/run_v1231_finalize_basis_kernel_first.py dgkan/diagnostics/basis_workspace.py dgkan/functional/mlp_functional.py
```

结果：

```text
py_compile pass
```

### 6.2 修复后 official rerun

执行命令：

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_basis \
  --run-id v1231_basis_workspace_official_fixed_reference \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 512 --val-size 256 --batch-size 128 \
  --hardening-epochs 3 --mlp-reference-epochs 1 --workspace-warmup-steps 1 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  --linec-seeds 12319500,12320600,12321600 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_basis_workspace_official_fixed_reference.log
```

修复后结果：

```text
workspace_rows = 198
workspace_gate_pass_rows = 36
workspace_strong_gate_pass_rows = 22
hardening_rows = 198
hardening_executed_rows = 36
family_near_pass_rows = 0
```

按 family 摘要：

```text
D-RAT rows=54 workspace_pass=36 strong=22 min_raw=0.7030934708989138 min_incremental=1.3255813953488371 min_step=0.5972796919231077
D-CHE rows=36 workspace_pass=0 strong=0 min_raw=0.9490445859872612 min_incremental=6.819767441860465 min_step=0.7927786921543468
D-FOU rows=36 workspace_pass=0 strong=0 min_raw=0.9228825604042744 min_incremental=3.9244186046511627 min_step=0.7347246970844235
D-RBF rows=36 workspace_pass=0 strong=0 min_raw=0.9103191732027864 min_incremental=3.5456810631229234 min_step=0.9492694811328597
D-WAV rows=36 workspace_pass=0 strong=0 min_raw=0.9233387727886859 min_incremental=3.5456810631229234 min_step=1.1220340571266207
```

Rational hardening probe 摘要：

```text
D-RAT10 executed=6 skipped=3 mean_delta=0.0032552083333333335 worst_delta=-0.0234375 LineC=17/18 aggregate_probe_pass=0
D-RAT11 executed=6 skipped=3 mean_delta=0.0032552083333333335 worst_delta=-0.0234375 LineC=17/18 aggregate_probe_pass=0
D-RAT12 executed=6 skipped=3 mean_delta=0.0032552083333333335 worst_delta=-0.0234375 LineC=17/18 aggregate_probe_pass=0
D-RAT7 executed=6 skipped=3 mean_delta=0.0032552083333333335 worst_delta=-0.0234375 LineC=17/18 aggregate_probe_pass=0
D-RAT8 executed=6 skipped=3 mean_delta=0.0026041666666666665 worst_delta=-0.0234375 LineC=16/18 aggregate_probe_pass=0
D-RAT9 executed=6 skipped=3 mean_delta=-0.15169270833333334 worst_delta=-0.2265625 LineC=3/18 aggregate_probe_pass=0
```

说明：

```text
Rational 打开了 workspace，但因为 3 个 dataset/seed row 仍 workspace skip，且本 runner 不测 AUC_time_ratio，不能写 FamilyNearPass。
```

## 7. Line M M-G MLP functional no-go test

候选：

```text
M-G1-UnlabeledCloneResponseCov
M-G2-RandomCotangentResponseStability
M-G3-InputPerturbationConsistencyTransport
M-G4-HiddenCovarianceTransportWithControlResidual
M-G5-ActivationSubspaceGuardThenTaskNeutral
```

Window 3 执行命令：

```bash
MG_CANDIDATES=M-G1-UnlabeledCloneResponseCov,M-G2-RandomCotangentResponseStability,M-G3-InputPerturbationConsistencyTransport,M-G4-HiddenCovarianceTransportWithControlResidual,M-G5-ActivationSubspaceGuardThenTaskNeutral
conda run -n kan python experiments/run_v1230_mlp_functional.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_mlp_functional_w3 \
  --candidates "$MG_CANDIDATES" \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-seed-bases 12310400,12311400,12312400 \
  --train-size 512 --val-size 256 --epochs 3 --batch-size 128 \
  --functional-batch 128 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  --linec-seeds 12319500,12320600,12321600,12322600,12323600 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_mlp_functional_w3.log
```

Window 3 结果：

```text
candidate_rows = 135
control_rows = 810
linec_rows = 1350
gate_rows = 5
any_exploration_pass = 0
any_official_pass = 0
```

Window 5 执行命令：

```bash
MG_CANDIDATES=M-G1-UnlabeledCloneResponseCov,M-G2-RandomCotangentResponseStability,M-G3-InputPerturbationConsistencyTransport,M-G4-HiddenCovarianceTransportWithControlResidual,M-G5-ActivationSubspaceGuardThenTaskNeutral
conda run -n kan python experiments/run_v1230_mlp_functional.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_mlp_functional_w5 \
  --candidates "$MG_CANDIDATES" \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-seed-bases 12310400,12311400,12312400 \
  --train-size 512 --val-size 256 --epochs 5 --batch-size 128 \
  --functional-batch 128 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  --linec-seeds 12319500,12320600,12321600,12322600,12323600 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_mlp_functional_w5.log
```

Window 5 结果：

```text
candidate_rows = 135
control_rows = 810
linec_rows = 1350
gate_rows = 5
any_exploration_pass = 0
any_official_pass = 0
```

Window 10 执行命令：

```bash
MG_CANDIDATES=M-G1-UnlabeledCloneResponseCov,M-G2-RandomCotangentResponseStability,M-G3-InputPerturbationConsistencyTransport,M-G4-HiddenCovarianceTransportWithControlResidual,M-G5-ActivationSubspaceGuardThenTaskNeutral
conda run -n kan python experiments/run_v1230_mlp_functional.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_mlp_functional_w10 \
  --candidates "$MG_CANDIDATES" \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-seed-bases 12310400,12311400,12312400 \
  --train-size 512 --val-size 256 --epochs 10 --batch-size 128 \
  --functional-batch 128 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  --linec-seeds 12319500,12320600,12321600,12322600,12323600 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_mlp_functional_w10.log
```

Window 10 结果：

```text
candidate_rows = 135
control_rows = 810
linec_rows = 1350
gate_rows = 5
any_exploration_pass = 0
any_official_pass = 0
```

合并命令：

```bash
python - <<'PY'
import csv
from pathlib import Path
out=Path('results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231')
for kind, target in [('candidates','v1231_mlp_functional_candidates.csv'),('controls','v1231_mlp_functional_controls.csv'),('linec','v1231_mlp_functional_linec.csv'),('gate','v1231_mlp_functional_gate.csv')]:
    ...
PY
```

合并结果：

```text
v1231_mlp_functional_candidates.csv = 405 rows
v1231_mlp_functional_controls.csv = 2430 rows
v1231_mlp_functional_linec.csv = 4050 rows
v1231_mlp_functional_gate.csv = 15 rows
exploration_rows = 0
official_rows = 0
```

## 8. Line A A-DYN monitor

执行命令：

```bash
conda run -n kan python experiments/run_v1218_b320_label_free_ablation.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_adyn_monitor \
  --run-id v1231_adyn_monitor \
  --summary-stage V1231_ADYN_MONITOR_SUMMARY \
  --route-stage V1231_ADYN_MONITOR_ROUTE \
  --result-scope v1231_adyn_monitor \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --ablation-ids A-DYN1-LearnableSignalFrameWarmup,A-DYN2-EarlySelfPredictiveFrame,A-DYN3-OptimizerObservableFrameRefresh,A-DYN4-OvercompleteRankGuardFrame,A-DYN5-RoleEnergyBalancedFHQMonitor \
  --train-size 512 --val-size 256 --test-size 256 \
  --epochs 3 --batch-size 128 \
  --measure-linec 1 --linec-batch-size 32 --linec-sketch-dim 8 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_adyn_monitor.log
```

结果：

```text
rows = 63
summary_rows = 7
best_label_free_candidate = A-DYN1-LearnableSignalFrameWarmup
best A-DYN mean_delta_vs_mlp = -0.012152777777777778
best A-DYN worst_delta_vs_mlp = -0.06640625
```

## 9. Finalizer / route / packet

首次执行命令：

```bash
conda run -n kan python experiments/run_v1231_finalize_basis_kernel_first.py \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_finalize_initial.log
```

首次 route：

```text
route = S1-BasisWorkspaceOpened
basis_workspace_pass_count = 36
basis_family_near_pass_count = 0
```

发现 route 逻辑问题：

```text
workspace 已打开但 hardening 已执行且 task/LineC 未闭合，按计划应该落为 R2-BasisTaskLineCNotColocated，并保留 minimum_success=S1-BasisWorkspaceOpened。
```

修复文件：

```text
experiments/run_v1231_finalize_basis_kernel_first.py
```

复核命令：

```bash
python -m py_compile experiments/run_v1231_finalize_basis_kernel_first.py && \
conda run -n kan python experiments/run_v1231_finalize_basis_kernel_first.py \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_finalize_route_fix.log
```

最终 route：

```text
route = R2-BasisTaskLineCNotColocated
minimum_success = S1-BasisWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_rows = 198
basis_workspace_pass_count = 36
basis_workspace_strong_pass_count = 22
basis_family_near_pass_count = 0
basis_hardening_executed_rows = 36
mlp_functional_candidate_rows = 405
mlp_functional_control_rows = 2430
mlp_functional_linec_rows = 4050
mlp_functional_exploration_pass_rows = 0
mlp_functional_official_pass_rows = 0
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
code_review_packet_entries = 78
code_review_packet_sha256 = 以最终 v1231_route_decision.json 为准
```

## 16. 用户再次追问后的最终只读复核与日志整理说明

本次用户再次要求确认 v12.31 是否达成目标。实际执行的只读复核命令：

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_route_decision.json')
r=json.loads(p.read_text())
for k in ['route','minimum_success','official_success_reached','p4_pass','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','basis_family_near_pass_count','rational_workspace_repair_pass_count','rational_tailnorm_workspace_repair_pass_count','rational_auc_near_pass_count','mlp_functional_exploration_pass_rows','line_a_near_anchor_pass_count','provenance_violation_count','code_review_packet_entries','code_review_packet_sha256']:
    print(f'{k}={r.get(k)}')
PY
```

只读复核结果：

```text
route = R2-BasisTaskLineCNotColocated
minimum_success = S1-BasisWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_family_near_pass_count = 0
rational_workspace_repair_pass_count = 45
rational_tailnorm_workspace_repair_pass_count = 26
rational_auc_near_pass_count = 0
mlp_functional_exploration_pass_rows = 0
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
code_review_packet_entries = 117
code_review_packet_sha256 = 1268f6b9771704d9d592406034135f04cbfc7021a5cb28de3b14d62ef8defaea
```

随后尝试整理执行日志后半段章节顺序；该整理命令因文档内嵌 `PY` here-doc 与外层 here-doc 冲突而失败，未写入文件、未改变实验数据。失败期间 shell 误执行了若干文档片段并触发了 finalizer 打包，但没有启动新增训练实验。

确认无遗留训练进程命令：

```bash
ps -ef | grep -E 'v1231_finalize|python - <<|conda run -n kan' | grep -v grep || true
```

结果：

```text
no running process
```

最终 route 再读：

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_route_decision.json')
r=json.loads(p.read_text())
print(r['route'], r['code_review_packet_sha256'])
PY
```

结果：

```text
R2-BasisTaskLineCNotColocated
code_review_packet_sha256 = c0ca4b67c7ac608caabea4b6e32735dd54370363e318dfe4bff2b23273ecedc7
```

本次没有新增实验。原因仍是：

```text
1. fallback_depth = 6 且 fallback_all_executed = 1。
2. final_stop_allowed = 1。
3. basis_family_near_pass_count = 0。
4. rational_auc_near_pass_count = 0。
5. mlp_functional_exploration_pass_rows = 0。
6. 当前已经不确定继续在同一 Rational alias / tailnorm token family 上扩局部组合能形成有效机制。
```

## 14. 用户再次追问后的 stop-contract 只读复核

用户再次要求确认 v12.31 是否达成目标，若未达成则继续。本次先读取最终 route decision，确认是否仍有计划内 fallback 未执行。

复核命令：

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_route_decision.json')
r=json.loads(p.read_text())
for k in ['route','minimum_success','official_success_reached','p4_pass','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','basis_family_near_pass_count','basis_workspace_pass_count','basis_workspace_strong_pass_count','rational_workspace_repair_pass_count','rational_tailnorm_workspace_repair_pass_count','rational_auc_near_pass_count','mlp_functional_exploration_pass_rows','mlp_functional_official_pass_rows','line_a_near_anchor_pass_count','provenance_violation_count','code_review_packet_entries','code_review_packet_sha256']:
    print(f'{k}={r.get(k)}')
PY
```

复核结果：

```text
route = R2-BasisTaskLineCNotColocated
minimum_success = S1-BasisWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_family_near_pass_count = 0
basis_workspace_pass_count = 36
basis_workspace_strong_pass_count = 22
rational_workspace_repair_pass_count = 45
rational_tailnorm_workspace_repair_pass_count = 26
rational_auc_near_pass_count = 0
mlp_functional_exploration_pass_rows = 0
mlp_functional_official_pass_rows = 0
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
code_review_packet_entries = 117
code_review_packet_sha256 = d84e0465cb91044f5564f06b45abc480f414cb24075455d931f93302678c119d
```

本次没有新增实验。原因：

```text
1. 计划内 fallback depth 6 已执行完毕。
2. workspace repair、early trajectory、conservative LR、output-geometry RMS norm/mix 均已尝试。
3. 当前仍无 basis FamilyNearPass、无 MLP functional exploration pass、无 Line A near-anchor。
4. 我现在不确定继续在同一 Rational alias / tailnorm token family 上扩局部组合能形成有效机制。
```

日志写入后最终打包命令：

```bash
conda run -n kan python experiments/run_v1231_finalize_basis_kernel_first.py \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_finalize_after_stop_contract_recheck.log
```

## 15. 用户再次追问后的 stop-contract 只读复核 2

复核命令：

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_route_decision.json')
r=json.loads(p.read_text())
for k in ['route','minimum_success','official_success_reached','p4_pass','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','basis_family_near_pass_count','rational_workspace_repair_pass_count','rational_tailnorm_workspace_repair_pass_count','rational_auc_near_pass_count','mlp_functional_exploration_pass_rows','line_a_near_anchor_pass_count','provenance_violation_count','code_review_packet_entries','code_review_packet_sha256']:
    print(f'{k}={r.get(k)}')
PY
```

复核结果：

```text
route = R2-BasisTaskLineCNotColocated
minimum_success = S1-BasisWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_family_near_pass_count = 0
rational_workspace_repair_pass_count = 45
rational_tailnorm_workspace_repair_pass_count = 26
rational_auc_near_pass_count = 0
mlp_functional_exploration_pass_rows = 0
line_a_near_anchor_pass_count = 0
provenance_violation_count = 0
code_review_packet_entries = 117
code_review_packet_sha256 = 0482bede2c2e1dc59f847ff397a2c8cb65b693ef739bf8e281e2c456962c1769
```

没有新增实验；原因同第 14 节：计划 fallback 已闭合，继续同 family token 扩展已经没有明确机制假设。

日志写入后最终打包命令：

```bash
conda run -n kan python experiments/run_v1231_finalize_basis_kernel_first.py \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_finalize_after_stop_contract_recheck2.log
```

## 13. 用户继续要求后的 Rational tailnorm repair

继续原因：

```text
v12.31 route 仍为 R2-BasisTaskLineCNotColocated
rational_auc_near_pass_count = 0
basis_family_near_pass_count = 0
mlp_functional_exploration_pass_rows = 0
```

本轮不继续排列旧 D-RAT7..D-RAT12，而是测试已有 `dgkan` PrimitiveSpec 中的 label-free output-geometry tail control：

```text
D-RAT13-RationalLogitBatchRMSNormSG
D-RAT14-RationalLogitBatchRMSMixSG025
D-RAT15-RationalLogitBatchRMSMixSG050
```

修改文件：

```text
dgkan/diagnostics/basis_workspace.py
experiments/run_v1231_finalize_basis_kernel_first.py
```

语法/import 检查：

```bash
python -m py_compile dgkan/diagnostics/basis_workspace.py experiments/run_v1231_basis_kernel_workspace.py experiments/run_v1231_rational_auc_hardening.py experiments/run_v1231_finalize_basis_kernel_first.py
conda run -n kan python -c "from dgkan.diagnostics.basis_workspace import V1231_BASIS_CANDIDATES; print(len(V1231_BASIS_CANDIDATES)); print([k for k in V1231_BASIS_CANDIDATES if k.startswith('D-RAT1')])"
```

结果：

```text
py_compile pass
V1231_BASIS_CANDIDATES = 25
D-RAT13/D-RAT14/D-RAT15 present
```

Workspace command：

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_rational_tailnorm_workspace_repair \
  --run-id v1231_rational_tailnorm_workspace_repair \
  --candidates D-RAT13-RationalLogitBatchRMSNormSG,D-RAT14-RationalLogitBatchRMSMixSG025,D-RAT15-RationalLogitBatchRMSMixSG050 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 512 --val-size 256 --batch-size 128 \
  --hardening-epochs 3 --mlp-reference-epochs 1 \
  --workspace-warmup-steps 2 --workspace-profile-steps 3 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  --linec-seeds 12319500,12320600,12321600 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_rational_tailnorm_workspace_repair.log
```

Workspace result：

```text
workspace_rows = 27
workspace_gate_pass_rows = 26
workspace_strong_gate_pass_rows = 10
hardening_executed_rows = 26
family_near_pass_rows = 0
D-RAT13 workspace pass = 8/9, skipped row = Fashion-MNIST seed 0
D-RAT14 workspace pass = 9/9
D-RAT15 workspace pass = 9/9
```

AUC repair command：

```bash
conda run -n kan python experiments/run_v1231_rational_auc_hardening.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_rational_tailnorm_auc_repair_e3_lr15 \
  --run-id v1231_rational_tailnorm_auc_repair_e3_lr15 \
  --workspace-csv results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_rational_tailnorm_workspace_repair_workspace_truth.csv \
  --candidates D-RAT13-RationalLogitBatchRMSNormSG,D-RAT14-RationalLogitBatchRMSMixSG025,D-RAT15-RationalLogitBatchRMSMixSG050 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 512 --val-size 256 --batch-size 128 \
  --epochs 3 --lr 0.0015 --weight-decay 0.001 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  --linec-seeds 12319500,12320600,12321600 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_rational_tailnorm_auc_repair_e3_lr15.log
```

AUC repair result：

```text
candidate_rows = 27
summary_rows = 3
trajectory_rows = 105
linec_rows = 78
candidate_auc_near_pass_rows = 0
D-RAT13: executed=8, skipped=1, mean_delta=-0.07177734375, worst_delta=-0.1171875, max_AUC_time=3.117835794920232, max_CEp99_delta=0.1939530372619629, LineC=6/24
D-RAT14: executed=9, skipped=0, mean_delta=-0.004774305555555556, worst_delta=-0.0390625, max_AUC_time=1.735411371440807, max_CEp99_delta=2.5851240158081055, LineC=25/27
D-RAT15: executed=9, skipped=0, mean_delta=-0.021701388888888888, worst_delta=-0.05078125, max_AUC_time=1.5925863296468374, max_CEp99_delta=2.3133020401000977, LineC=19/27
```

Finalizer after tailnorm command：

```bash
conda run -n kan python experiments/run_v1231_finalize_basis_kernel_first.py \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_finalize_after_tailnorm_repair.log
```

结果：

```text
route = R2-BasisTaskLineCNotColocated
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
rational_workspace_repair_pass_count = 45
rational_tailnorm_workspace_repair_rows = 27
rational_tailnorm_workspace_repair_pass_count = 26
rational_auc_near_pass_count = 0
rational_auc_repair_summary_rows = 23
code_review_packet_entries = 117
```

日志写入后最终打包命令：

```bash
conda run -n kan python experiments/run_v1231_finalize_basis_kernel_first.py \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_finalize_after_tailnorm_repair_execution_log_final.log
```

最终打包复核：

```text
route = R2-BasisTaskLineCNotColocated
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
rational_tailnorm_workspace_repair_pass_count = 26
rational_auc_near_pass_count = 0
code_review_packet_entries = 117
code_review_packet_sha256 = 以最终 v1231_route_decision.json 为准
```

最终 packet 包含日志文本后又执行一次 finalizer：

```bash
conda run -n kan python experiments/run_v1231_finalize_basis_kernel_first.py \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_finalize_after_logs_final2.log
```

结果：

```text
route = R2-BasisTaskLineCNotColocated
minimum_success = S1-BasisWorkspaceOpened
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
code_review_packet_entries = 78
code_review_packet_sha256 = 53583c27db49b00dd811215bcb44d77dc67cff82c3a17a4b0287a03fe9c519b3
```

最终日志占位整理后重新打包命令：

```bash
conda run -n kan python experiments/run_v1231_finalize_basis_kernel_first.py \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_finalize_after_logs_final.log
```

结果：

```text
route = R2-BasisTaskLineCNotColocated
minimum_success = S1-BasisWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
required_artifact_missing_count = 0
code_review_packet_entries = 78
code_review_packet_sha256 = 以最终 v1231_route_decision.json 为准
```

日志写入后最终打包命令：

```bash
conda run -n kan python experiments/run_v1231_finalize_basis_kernel_first.py \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_finalize_after_logs.log
```

最终打包结果：

```text
route = R2-BasisTaskLineCNotColocated
minimum_success = S1-BasisWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 6
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_pass_count = 36
basis_family_near_pass_count = 0
mlp_functional_exploration_pass_rows = 0
line_a_near_anchor_pass_count = 0
code_review_packet_entries = 78
code_review_packet_sha256 = 以最终 v1231_route_decision.json 为准
```

## 13. 用户再次要求后的 Rational tailnorm output-geometry repair

继续原因：

```text
v12.31 仍未达成目标；
route = R2-BasisTaskLineCNotColocated；
rational_auc_near_pass_count = 0；
最接近的 e3_lr15 D-RAT10 仍有 CEp99 tail 坏化。
```

本轮不使用 CEp99 设计 CE/loss direction，而是复用已有 `PrimitiveSpec` 中 label-free stop-gradient logit RMS norm / RMS mix 的 Rational output-geometry paths。

### 13.1 Candidate registry 修改

修改文件：

```text
dgkan/diagnostics/basis_workspace.py
```

新增 candidates：

```text
D-RAT13-RationalLogitBatchRMSNormSG
D-RAT14-RationalLogitBatchRMSMixSG025
D-RAT15-RationalLogitBatchRMSMixSG050
```

映射：

```text
D-RAT13 -> B7md-RationalKAT-...-logitBatchRMSNormSG-...
D-RAT14 -> B7mg-RationalKAT-...-logitBatchRMSMixSG025-...
D-RAT15 -> B7mh-RationalKAT-...-logitBatchRMSMixSG050-...
```

语法/import 检查：

```bash
python -m py_compile dgkan/diagnostics/basis_workspace.py experiments/run_v1231_basis_kernel_workspace.py experiments/run_v1231_rational_auc_hardening.py experiments/run_v1231_finalize_basis_kernel_first.py
conda run -n kan python -c "from dgkan.diagnostics.basis_workspace import V1231_BASIS_CANDIDATES; print(len(V1231_BASIS_CANDIDATES)); print([k for k in V1231_BASIS_CANDIDATES if k.startswith('D-RAT1')])"
```

结果：

```text
py_compile pass
V1231_BASIS_CANDIDATES = 25
D-RAT13..D-RAT15 present
```

### 13.2 Tailnorm workspace repair

执行命令：

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_rational_tailnorm_workspace_repair \
  --run-id v1231_rational_tailnorm_workspace_repair \
  --candidates D-RAT13-RationalLogitBatchRMSNormSG,D-RAT14-RationalLogitBatchRMSMixSG025,D-RAT15-RationalLogitBatchRMSMixSG050 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 512 --val-size 256 --batch-size 128 \
  --hardening-epochs 3 --mlp-reference-epochs 1 \
  --workspace-warmup-steps 2 --workspace-profile-steps 3 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  --linec-seeds 12319500,12320600,12321600 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_rational_tailnorm_workspace_repair.log
```

结果：

```text
workspace_rows = 27
workspace_gate_pass_rows = 26
workspace_strong_gate_pass_rows = 10
hardening_rows = 27
hardening_executed_rows = 26
family_near_pass_rows = 0
```

### 13.3 Tailnorm AUC repair

执行命令：

```bash
conda run -n kan python experiments/run_v1231_rational_auc_hardening.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_rational_tailnorm_auc_repair_e3_lr15 \
  --run-id v1231_rational_tailnorm_auc_repair_e3_lr15 \
  --workspace-csv results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_rational_tailnorm_workspace_repair_workspace_truth.csv \
  --candidates D-RAT13-RationalLogitBatchRMSNormSG,D-RAT14-RationalLogitBatchRMSMixSG025,D-RAT15-RationalLogitBatchRMSMixSG050 \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 512 --val-size 256 --batch-size 128 \
  --epochs 3 --lr 0.0015 --weight-decay 0.001 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  --linec-seeds 12319500,12320600,12321600 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_rational_tailnorm_auc_repair_e3_lr15.log
```

结果：

```text
candidate_rows = 27
summary_rows = 3
trajectory_rows = 105
linec_rows = 78
candidate_auc_near_pass_rows = 0
```

关键 summary：

```text
D-RAT13: executed=8, skipped=1, mean_delta=-0.07177734375, worst=-0.1171875, max_AUC_time=3.117835794920232, max_CEp99_delta=0.1939530372619629, LineC=6/24, near=0
D-RAT14: executed=9, skipped=0, mean_delta=-0.004774305555555556, worst=-0.0390625, max_AUC_time=1.735411371440807, max_CEp99_delta=2.5851240158081055, LineC=25/27, near=0
D-RAT15: executed=9, skipped=0, mean_delta=-0.021701388888888888, worst=-0.05078125, max_AUC_time=1.5925863296468374, max_CEp99_delta=2.3133020401000977, LineC=19/27, near=0
```

结论：

```text
tailnorm output-geometry repair 没有形成 near-pass；
D-RAT13 降低 CEp99 tail 但 workspace/task/AUC/LineC 明显失败；
D-RAT14/D-RAT15 保留部分 LineC，但 worst/AUC/CEp99 仍失败。
```

### 13.4 Finalizer 纳入 tailnorm artifacts

修改文件：

```text
experiments/run_v1231_finalize_basis_kernel_first.py
```

修改内容：

```text
1. required manifest 纳入 v1231_rational_tailnorm_* artifacts。
2. rational_auc_repair_all_summary.csv 同时聚合 rational_auc_repair* 与 rational_tailnorm_auc_repair* summary。
3. route decision 新增 rational_tailnorm_workspace_repair_rows/pass_count。
```

Finalizer 命令：

```bash
conda run -n kan python experiments/run_v1231_finalize_basis_kernel_first.py \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_finalize_after_tailnorm_repair.log
```

结果：

```text
route = R2-BasisTaskLineCNotColocated
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
rational_tailnorm_workspace_repair_rows = 27
rational_tailnorm_workspace_repair_pass_count = 26
rational_auc_near_pass_count = 0
rational_auc_repair_summary_rows = 23
code_review_packet_entries = 117
code_review_packet_sha256 = 8572216518e08de23a7e493c0cb67a4323b0a6e3416c6afddb817ac1a0b3db78
```

日志写入后最终打包命令：

```bash
conda run -n kan python experiments/run_v1231_finalize_basis_kernel_first.py \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_finalize_after_tailnorm_repair_logs.log
```

## 12. 用户继续要求后的 Rational workspace / AUC repair continuation

继续原因：

```text
当前 route = R2-BasisTaskLineCNotColocated
minimum_success = S1-BasisWorkspaceOpened
official_success_reached = 0
basis_family_near_pass_count = 0
mlp_functional_exploration_pass_rows = 0
promotion_allowed = 0
```

本轮不扩展 M-G 单项 functional objective，也不把非 Rational family 直接 task grid 化；按计划优先处理 Rational workspace skipped rows 与 worst/AUC blocker。

### 12.1 多步 workspace profile 代码修复

修改文件：

```text
dgkan/diagnostics/basis_workspace.py
experiments/run_v1231_basis_kernel_workspace.py
```

修改内容：

```text
1. dgkan/diagnostics/basis_workspace.py 新增 measured_phase_window()。
2. WORKSPACE_FIELDS 新增 profile_steps。
3. experiments/run_v1231_basis_kernel_workspace.py 新增 --workspace-profile-steps。
4. MLP workspace profile 与 basis workspace profile 从 measured_phase_step 改为 measured_phase_window。
```

语法检查：

```bash
python -m py_compile dgkan/diagnostics/basis_workspace.py experiments/run_v1231_basis_kernel_workspace.py
```

结果：

```text
py_compile pass
```

### 12.2 Rational multi-step workspace repair

执行命令：

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_rational_workspace_repair \
  --run-id v1231_rational_workspace_repair_multistep \
  --candidates D-RAT7-FusedGroupRationalNoMaterialize,D-RAT8-RecomputeDenominatorBackward,D-RAT10-FusedDenNumReadoutGrad,D-RAT11-LowMemGroupSharedDenomPlusLineC,D-RAT12-RationalWorkspaceMinStrongDiag \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 512 --val-size 256 --batch-size 128 \
  --hardening-epochs 3 --mlp-reference-epochs 1 \
  --workspace-warmup-steps 2 --workspace-profile-steps 3 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  --linec-seeds 12319500,12320600,12321600 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_rational_workspace_repair_multistep.log
```

产物：

```text
v1231_rational_workspace_repair_workspace_truth.csv
v1231_rational_workspace_repair_component_peak.csv
v1231_rational_workspace_repair_microkernel_correctness.csv
v1231_rational_workspace_repair_hardening.csv
v1231_rational_workspace_repair_linec.csv
v1231_rational_workspace_repair_workspace_aggregate.json
```

结果：

```text
workspace_rows = 45
workspace_gate_pass_rows = 45
workspace_strong_gate_pass_rows = 26
hardening_rows = 45
hardening_executed_rows = 45
family_near_pass_rows = 0
```

### 12.3 Rational AUC hardening runner

新增/修改文件：

```text
dgkan/diagnostics/basis_workspace.py
experiments/run_v1231_rational_auc_hardening.py
```

新增核心函数：

```text
train_epoch_timed
finite_mean
trajectory_auc
trajectory_time_auc
```

新增 runner：

```text
experiments/run_v1231_rational_auc_hardening.py
```

语法/import 检查：

```bash
python -m py_compile dgkan/diagnostics/basis_workspace.py experiments/run_v1231_rational_auc_hardening.py
conda run -n kan python -c "from dgkan.diagnostics.basis_workspace import train_epoch_timed, trajectory_auc, trajectory_time_auc; import experiments.run_v1231_rational_auc_hardening as r; print('ok', r.DEFAULT_CANDIDATES.count('D-RAT'))"
```

结果：

```text
py_compile pass
ok 5
```

### 12.4 Rational 8-epoch AUC repair

执行命令：

```bash
conda run -n kan python experiments/run_v1231_rational_auc_hardening.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_rational_auc_repair \
  --run-id v1231_rational_auc_repair \
  --workspace-csv results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_rational_workspace_repair_workspace_truth.csv \
  --candidates D-RAT7-FusedGroupRationalNoMaterialize,D-RAT8-RecomputeDenominatorBackward,D-RAT10-FusedDenNumReadoutGrad,D-RAT11-LowMemGroupSharedDenomPlusLineC,D-RAT12-RationalWorkspaceMinStrongDiag \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 512 --val-size 256 --batch-size 128 \
  --epochs 8 --lr 0.002 --weight-decay 0.001 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  --linec-seeds 12319500,12320600,12321600 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_rational_auc_repair.log
```

发现审计字段问题：

```text
trajectory CSV 的 seed 字段写成了内部 RNG salt；candidate/summary 不受影响，但为复现清晰修复为原始实验 seed。
```

修复文件：

```text
experiments/run_v1231_rational_auc_hardening.py
```

复核命令：

```bash
python -m py_compile experiments/run_v1231_rational_auc_hardening.py
```

结果：

```text
py_compile pass
```

修复后重跑命令：

```bash
conda run -n kan python experiments/run_v1231_rational_auc_hardening.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_rational_auc_repair \
  --run-id v1231_rational_auc_repair \
  --workspace-csv results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_rational_workspace_repair_workspace_truth.csv \
  --candidates D-RAT7-FusedGroupRationalNoMaterialize,D-RAT8-RecomputeDenominatorBackward,D-RAT10-FusedDenNumReadoutGrad,D-RAT11-LowMemGroupSharedDenomPlusLineC,D-RAT12-RationalWorkspaceMinStrongDiag \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 512 --val-size 256 --batch-size 128 \
  --epochs 8 --lr 0.002 --weight-decay 0.001 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  --linec-seeds 12319500,12320600,12321600 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_rational_auc_repair_seedfield_fix.log
```

结果：

```text
candidate_rows = 45
summary_rows = 5
trajectory_rows = 432
linec_rows = 135
candidate_auc_near_pass_rows = 0
```

### 12.5 Rational global early-stop repairs

因 8-epoch trajectory 显示 Rational 在 early epochs 后出现 NLL/CE tail 上升与 KMNIST worst 回落，执行非 dataset-specific 的固定 epoch repair。

4-epoch 命令：

```bash
conda run -n kan python experiments/run_v1231_rational_auc_hardening.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_rational_auc_repair_e4 \
  --run-id v1231_rational_auc_repair_e4 \
  --workspace-csv results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_rational_workspace_repair_workspace_truth.csv \
  --candidates D-RAT7-FusedGroupRationalNoMaterialize,D-RAT8-RecomputeDenominatorBackward,D-RAT10-FusedDenNumReadoutGrad,D-RAT11-LowMemGroupSharedDenomPlusLineC,D-RAT12-RationalWorkspaceMinStrongDiag \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 512 --val-size 256 --batch-size 128 \
  --epochs 4 --lr 0.002 --weight-decay 0.001 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  --linec-seeds 12319500,12320600,12321600 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_rational_auc_repair_e4.log
```

4-epoch 结果：

```text
candidate_rows = 45
summary_rows = 5
trajectory_rows = 216
linec_rows = 135
candidate_auc_near_pass_rows = 0
```

3-epoch 命令：

```bash
conda run -n kan python experiments/run_v1231_rational_auc_hardening.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_rational_auc_repair_e3 \
  --run-id v1231_rational_auc_repair_e3 \
  --workspace-csv results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_rational_workspace_repair_workspace_truth.csv \
  --candidates D-RAT7-FusedGroupRationalNoMaterialize,D-RAT8-RecomputeDenominatorBackward,D-RAT10-FusedDenNumReadoutGrad,D-RAT11-LowMemGroupSharedDenomPlusLineC,D-RAT12-RationalWorkspaceMinStrongDiag \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 512 --val-size 256 --batch-size 128 \
  --epochs 3 --lr 0.002 --weight-decay 0.001 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  --linec-seeds 12319500,12320600,12321600 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_rational_auc_repair_e3.log
```

3-epoch 结果：

```text
candidate_rows = 45
summary_rows = 5
trajectory_rows = 162
linec_rows = 135
candidate_auc_near_pass_rows = 0
```

3-epoch conservative LR 命令：

```bash
conda run -n kan python experiments/run_v1231_rational_auc_hardening.py \
  --out-dir results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231 \
  --artifact-prefix v1231_rational_auc_repair_e3_lr15 \
  --run-id v1231_rational_auc_repair_e3_lr15 \
  --workspace-csv results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/v1231_rational_workspace_repair_workspace_truth.csv \
  --candidates D-RAT7-FusedGroupRationalNoMaterialize,D-RAT8-RecomputeDenominatorBackward,D-RAT10-FusedDenNumReadoutGrad,D-RAT11-LowMemGroupSharedDenomPlusLineC,D-RAT12-RationalWorkspaceMinStrongDiag \
  --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 \
  --train-size 512 --val-size 256 --batch-size 128 \
  --epochs 3 --lr 0.0015 --weight-decay 0.001 \
  --linec-batch-size 32 --linec-sketch-dim 8 \
  --linec-seeds 12319500,12320600,12321600 \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_rational_auc_repair_e3_lr15.log
```

3-epoch conservative LR 结果：

```text
candidate_rows = 45
summary_rows = 5
trajectory_rows = 162
linec_rows = 135
candidate_auc_near_pass_rows = 0
```

### 12.6 Finalizer 纳入 repair artifacts

修改文件：

```text
experiments/run_v1231_finalize_basis_kernel_first.py
```

修改内容：

```text
1. required manifest 纳入 Rational workspace repair 与 e3_lr15 AUC repair artifacts。
2. code review manifest 纳入 measured_phase_window / train_epoch_timed / trajectory_time_auc / run_v1231_rational_auc_hardening.py。
3. route decision 新增 rational_workspace_repair_pass_count / rational_auc_near_pass_count / rational_auc_repair_summary_rows。
4. code packet 纳入 run_v1231_rational_auc_hardening.py。
```

语法/import 检查：

```bash
python -m py_compile experiments/run_v1231_finalize_basis_kernel_first.py dgkan/diagnostics/basis_workspace.py experiments/run_v1231_basis_kernel_workspace.py experiments/run_v1231_rational_auc_hardening.py
conda run -n kan python -c "from dgkan.diagnostics.basis_workspace import measured_phase_window, train_epoch_timed, trajectory_time_auc; import experiments.run_v1231_rational_auc_hardening as r; print('audit_import_ok')"
```

结果：

```text
py_compile pass
audit_import_ok
```

Finalizer 命令：

```bash
conda run -n kan python experiments/run_v1231_finalize_basis_kernel_first.py \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_finalize_after_rational_repairs.log
```

结果：

```text
route = R2-BasisTaskLineCNotColocated
minimum_success = S1-BasisWorkspaceOpened
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_workspace_pass_count = 36
basis_family_near_pass_count = 0
rational_workspace_repair_pass_count = 45
rational_auc_near_pass_count = 0
rational_auc_repair_summary_rows = 20
mlp_functional_exploration_pass_rows = 0
line_a_near_anchor_pass_count = 0
code_review_packet_entries = 106
code_review_packet_sha256 = df9ae117e5e4d0f2bf0b06fa86c999948dc714aa095e1db14ac177d7b87cd393
```

日志写入后最终打包命令：

```bash
conda run -n kan python experiments/run_v1231_finalize_basis_kernel_first.py \
  2>&1 | tee results/v12_31_basis_kernel_first_mlp_functional_no_go/official_v1231/logs/v1231_finalize_after_rational_repairs_logs.log
```

最终打包结果：

```text
route = R2-BasisTaskLineCNotColocated
official_success_reached = 0
p4_pass = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
rational_workspace_repair_pass_count = 45
rational_auc_near_pass_count = 0
code_review_packet_entries = 106
code_review_packet_sha256 = 以最终 v1231_route_decision.json 为准
```
