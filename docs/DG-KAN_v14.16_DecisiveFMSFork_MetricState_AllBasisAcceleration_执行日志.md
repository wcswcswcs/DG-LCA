# DG-KAN v14.16 DecisiveFMSFork MetricState AllBasisAcceleration 执行日志

生成时间：2026-05-30（Asia/Singapore）

## 1. 关键文件

```text
计划文档：
docs/DG-KAN_v14.16_DecisiveFMSFork_MetricState_AllBasisAcceleration_完整计划.md

runner：
experiments/run_v1416_decisive_fms_fork_metric_state_allbasis.py

底层修改：
experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py

official artifacts：
results/v14_16_decisive_fms_fork_metric_state_allbasis/official_v1416
```

## 2. 执行命令

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python -m py_compile experiments/run_v1416_decisive_fms_fork_metric_state_allbasis.py experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
```

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v149_line_d_all_basis_substrate_repair.py --out-dir results/v14_16_decisive_fms_fork_metric_state_allbasis/line_e_v1416_substrate_acceleration --device cuda:0 --candidates D-FOU32-LowFreqIdentityResidualV3,D-FOU33-BandwiseSNRWarmupV2,D-FOU34-PhaseStableBandMixNoHighFreqV2,D-FOU35-NoMaterializeLifetimeV4,D-FOU36-HighFrequencyQuarantineV2,D-RBF30-ActiveCenterOccupancyV3,D-RBF31-WidthConditionIdentityResidualV2,D-RBF32-CompactBumpNoDenseMaterializationV2,D-RBF33-GaussianLocalK4TaskHealthV2,D-RBF34-CenterOccupancyWarmupNoTaskBranch,D-WAV29-TriangularSupportV4,D-WAV30-ScaleOccupancyNoTailTargetV2,D-WAV31-LocalSupportOverlapDampingV2,D-WAV32-LocalTailCoverageAuditV2 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --epochs 1 --linec-seeds 12319500,12319501,12319502
```

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python experiments/run_v1416_decisive_fms_fork_metric_state_allbasis.py --out-dir results/v14_16_decisive_fms_fork_metric_state_allbasis/official_v1416 --line-e-out results/v14_16_decisive_fms_fork_metric_state_allbasis/line_e_v1416_substrate_acceleration --device cuda:0 --datasets MNIST,Fashion-MNIST,KMNIST --seeds 0,1,2 --train-size 256 --val-size 128 --test-size 128 --train-steps 200 --batch-size 32 --streaming-per-example-gradients 1 --fms-update-interval 80 --linec-seeds 12319500,12319501,12319502 --reuse-metric-state-if-present 1 --compute-budgeted-run 1
```

Manifest self-check 修正后，仅重写 manifest / route / 日志：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import json
from pathlib import Path
from experiments import run_v1416_decisive_fms_fork_metric_state_allbasis as r
out=Path('results/v14_16_decisive_fms_fork_metric_state_allbasis/official_v1416')
route=json.loads((out/'v1416_route_decision.json').read_text())
missing=r.write_required_manifest(out)
route['required_artifact_missing_count']=missing
r.write_json(out/'v1416_route_decision.json', route)
args=r.build_argparser().parse_args(['--out-dir', str(out)])
r.write_docs(route, args)
PY
```

补齐执行日志模板后，重新打包 code review packet 并复核 required manifest：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
from pathlib import Path
from experiments import run_v1416_decisive_fms_fork_metric_state_allbasis as r
out=Path('results/v14_16_decisive_fms_fork_metric_state_allbasis/official_v1416')
r.write_code_packet(out)
missing=r.write_required_manifest(out)
print('code_packet_rewritten missing', missing)
PY
```

## 3. 复现说明

```text
1. v14.16 runner 会实际执行 Line A full selector-direction-boundary contrast table。
2. Line B/M 会实际运行 MS1/MS2/MS3 与 matched controls；若 official metric-state artifact 已存在，可复用以避免重复训练。
3. Line C 会读取 v14.15 official artifacts，并在 --compute-budgeted-run 1 时执行 fixed half-budget Line A probe 补齐 leave-time-budget-out。
4. Line E 会优先读取 --line-e-out 中的 v14.16 substrate-only reconfirmation；缺失时才 fallback 到 v14.15 replay。
5. Line D 只在 Line A 或 Line B 有 passing mechanism 时打开；本轮未打开，artifact 记录 not opened reason。
6. promotion_allowed 始终 fail-closed。
7. 若只需复核 manifest self-check，可重写 manifest 和 route，不需要重跑 metric-state 训练。
```

## 4. 关键结果

```text
route = R16-CurrentFMSDefinitionNoCausalValue
Line A = R-A-CurrentFMSDirectionNoGo
Line B = R-B-MetricStateFMSNoGo
Line C = R-C-TransferPredictorNotRobust
Line C leaveout_rows = 63
Line C leaveout_available_rows = 57
Line C complete_feature_count = 7
Line C time_budget_probe_rows = 567
Line D = R-D-DCHENoRealLiteTransfer
Line E = R-E-AllBasisCarrierBlocked
Line E source = v1416_actual_v149_substrate_reconfirmation
Line E rows = 126
Line M = M-ControlsCompleteNoPositive
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

## 5. 用户再次追问后的 Line E actual reconfirmation

复核 v14.16 完整计划第 10 节后，确认 Line E 不能只停在 v14.15 replay；
已按 substrate-only 分支实际执行 D-FOU32..36 / D-RBF30..34 / D-WAV29..32。

新增 / 修改文件：

```text
experiments/run_v1416_decisive_fms_fork_metric_state_allbasis.py
  新增 DEFAULT_LINE_E_OUT / LINE_E_CANDIDATES。
  build_line_e 优先读取 --line-e-out/v149_line_d_substrate_repair_results.csv。
  写入 line_e_source、line_e_rows、v1416_substrate_gate_pass。
```

Line E substrate-only 命令输出：

```text
out_dir = results/v14_16_decisive_fms_fork_metric_state_allbasis/line_e_v1416_substrate_acceleration
route = R8-NonRATSubstrateStillMissing
candidate_rows = 126
linec_rows = 378
best_family = D-FOU
best_family_dataset_seed_pass_count = 0 / 9
exploration_open_family_count = 0
official_fms_eligible_family_count = 0
official_fms_proof_executed = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
```

Line E actual artifacts：

```text
results/v14_16_decisive_fms_fork_metric_state_allbasis/line_e_v1416_substrate_acceleration/v149_line_d_substrate_repair_results.csv
  rows = 126
  D-FOU rows = 45
  D-RBF rows = 45
  D-WAV rows = 36

results/v14_16_decisive_fms_fork_metric_state_allbasis/line_e_v1416_substrate_acceleration/v149_line_d_substrate_repair_linec.csv
  rows = 378

results/v14_16_decisive_fms_fork_metric_state_allbasis/line_e_v1416_substrate_acceleration/v149_line_d_substrate_repair_summary.csv
  rows = 3
  D-FOU family pass = 0/9
  D-RBF family pass = 0/9
  D-WAV family pass = 0/9
```

重新运行 finalizer 后的 official_v1416 Line E：

```text
v1416_e_allbasis_summary.csv rows = 3
v1416_e_fou_substrate.csv rows = 45
v1416_e_rbf_substrate.csv rows = 45
v1416_e_wav_monitor.csv rows = 36
line_e_source = v1416_actual_v149_substrate_reconfirmation
line_e_rows = 126
line_e_best_family = D-FOU
line_e_best_count = 0 / 9
line_e_gate_pass = 0
line_e_route = R-E-AllBasisCarrierBlocked
```

复核命令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv, json
from pathlib import Path
out = Path('results/v14_16_decisive_fms_fork_metric_state_allbasis/official_v1416')
route = json.loads((out/'v1416_route_decision.json').read_text())
print({k: route.get(k) for k in [
  'route','minimum_success','line_e_source','line_e_rows',
  'line_e_best_family','line_e_best_count','line_e_gate_pass',
  'official_s5_reached','promotion_allowed',
  'required_artifact_missing_count',
  'forbidden_information_violation_count',
  'no_action_search_violation_count'
]})
for name in [
  'v1416_required_artifact_manifest.csv',
  'v1416_forbidden_information_audit.csv',
  'v1416_no_action_search_audit.csv'
]:
    rows = list(csv.DictReader((out/name).open(newline='')))
    print(name, len(rows))
PY
```

复核结果：

```text
route = R16-CurrentFMSDefinitionNoCausalValue
minimum_success = S0-ParallelForkExecuted
line_e_source = v1416_actual_v149_substrate_reconfirmation
line_e_rows = 126
line_e_best_family = D-FOU
line_e_best_count = 0
line_e_gate_pass = 0
official_s5_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
forbidden_information_violation_count = 0
no_action_search_violation_count = 0
manifest_missing_sum = 0
forbidden_violation_sum = 0
no_action_violation_sum = 0
```

## 6. 用户再次追问后的计划覆盖矩阵复核

本次没有新增训练；只执行 artifact 覆盖与 audit 复核。

覆盖复核命令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv, json
from pathlib import Path
out=Path('results/v14_16_decisive_fms_fork_metric_state_allbasis/official_v1416')
route=json.loads((out/'v1416_route_decision.json').read_text())
print('ROUTE')
for k in [
  'route','minimum_success','line_a_route','line_b_route','line_c_route',
  'line_d_route','line_e_route','line_m_route','official_s5_reached',
  'promotion_allowed','required_artifact_missing_count',
  'forbidden_information_violation_count','no_action_search_violation_count'
]:
    print(k, route.get(k))
for name, keys in [
 ('v1416_a_current_fms_exit_matrix.csv',
  ['line_a_selector_id','line_a_direction_id','line_a_boundary_id','method','dataset','seed']),
 ('v1416_b_metric_state_results.csv',
  ['method','dataset','seed']),
 ('v1416_b_metric_state_controls.csv',
  ['method','dataset','seed']),
 ('v1416_m_mlp_generic_controls.csv',
  ['method','dataset','seed'])
]:
    rows=list(csv.DictReader((out/name).open(newline='')))
    print(name, 'rows=', len(rows))
    for key in keys:
        print(key, sorted({r.get(key,'') for r in rows}))
rows=list(csv.DictReader((out/'v1416_c_transfer_predictor_leaveout.csv').open(newline='')))
print('Line C rows', len(rows), 'available', sum(int(float(r.get('available_for_gate',0) or 0)) for r in rows))
for split in sorted({r['planned_leaveout_split'] for r in rows}):
    sr=[r for r in rows if r['planned_leaveout_split']==split]
    print(split, len(sr), sum(int(float(r.get('available_for_gate',0) or 0)) for r in sr))
PY
```

覆盖复核结果：

```text
route = R16-CurrentFMSDefinitionNoCausalValue
minimum_success = S0-ParallelForkExecuted
Line A = R-A-CurrentFMSDirectionNoGo
Line B = R-B-MetricStateFMSNoGo
Line C = R-C-TransferPredictorNotRobust
Line D = R-D-DCHENoRealLiteTransfer
Line E = R-E-AllBasisCarrierBlocked
Line M = M-ControlsCompleteNoPositive
official_s5_reached = 0
promotion_allowed = 0

Line A rows = 567
selectors = S0/S1/S4
directions = D0/D1/D2/D3/D4/D5/D6
boundaries = B0/B1/B4

Line B metric-state rows = 27
Line B D-CHE control rows = 54
Line M MLP/generic control rows = 81

Line C leaveout rows = 63
Line C available rows = 57
unavailable features in method/control/time-budget split =
  P2-recovery_lag
  P7-micro_horizon_loss_integral

Line E actual rows = 126
D-FOU/D-RBF/D-WAV family pass = 0/9, 0/9, 0/9

manifest_missing_sum = 0
forbidden_violation_sum = 0
no_action_violation_sum = 0
```

结论：

```text
没有发现 v14.16 当前计划内仍可合法补跑的预注册分支。
当前停止不是 fail-fast，而是 A/B/C/D/E/M/Z 已闭合后的 route decision。
```

## 7. 用户再次追问后的实现级 selector / boundary 复核

本次没有新增训练；检查 Line A selector / boundary 是否实际进入训练逻辑。

代码检索命令：

```bash
rg -n "line_a_selector|line_a_boundary|S1-predictor|NoOp-safe|value-retention|degree-projection-safety|line_a_direction|SameProjection|SameValueRetention|A-D4|A-D5" experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py experiments/run_v1416_decisive_fms_fork_metric_state_allbasis.py
```

关键命中：

```text
experiments/run_v1416_decisive_fms_fork_metric_state_allbasis.py
  run_actual_line_a 设置 local_args.line_a_selector_id / line_a_direction_id / line_a_boundary_id。

experiments/run_v1410_nonrat_fms_transfer_fms_definition_reset.py
  train_model_case 读取 line_a_boundary_id 并执行：
    B1-degree-projection-safety
    B4-value-retention-floor

  train_model_case 读取 line_a_selector_id 并执行：
    S1-predictor-high-score
    S4-NoOp-safe-abstention
```

artifact 复核命令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv, statistics
from collections import defaultdict, Counter
from pathlib import Path
rows=list(csv.DictReader(Path('results/v14_16_decisive_fms_fork_metric_state_allbasis/official_v1416/v1416_a_current_fms_exit_matrix.csv').open(newline='')))
by=defaultdict(list)
for r in rows:
    by[r['line_a_selector_id']].append(float(r.get('line_a_selector_active_fraction',0) or 0))
for k, vals in by.items():
    print(k, 'n=', len(vals), 'min=', min(vals), 'max=', max(vals), 'mean=', statistics.fmean(vals))
print('strict passes', Counter((r['line_a_selector_id'], r['line_a_direction_id'], r['line_a_boundary_id']) for r in rows if int(float(r.get('strict_gate_pass',0) or 0))==1))
print('current FMS source by selector')
for sel in sorted(by):
    vals=[float(r.get('source_vs_best_control',0) or 0) for r in rows if r['line_a_selector_id']==sel and r['line_a_direction_id']=='D1-current-FMS']
    print(sel, len(vals), statistics.fmean(vals), min(vals), max(vals))
PY
```

artifact 复核结果：

```text
selector active fraction:
  S0-always-on:
    n = 189
    min = 1.0
    max = 1.0
    mean = 1.0

  S1-predictor-high-score:
    n = 189
    min = 0.0
    max = 0.015
    mean = 0.010714285714285714

  S4-NoOp-safe-abstention:
    n = 189
    min = 1.0
    max = 1.0
    mean = 1.0

strict passes = 0

current FMS source_vs_best_control:
  S0-always-on:
    rows = 27
    mean = -0.015283273326026069
    min = -0.0378987193107605
    max = 0.011716127395629883

  S1-predictor-high-score:
    rows = 27
    mean = -0.028656482696533203
    min = -0.08471512794494629
    max = 0.019894123077392578

  S4-NoOp-safe-abstention:
    rows = 27
    mean = -0.015283273326026069
    min = -0.0378987193107605
    max = 0.011716127395629883
```

结论：

```text
Line A selector / boundary 已进入训练逻辑，不是 metadata-only。
S1 实际 abstain，但没有改善 current FMS direction。
S4 在当前 stats 下等价于 S0，没有打开 boundary/curriculum value。
没有发现新的合法修复分支。
```

## 8. 用户再次追问后的 Line B / Line M control surface 复核

本次没有新增训练；复核 Line B controls 与 Line M generic controls 的映射。

复核命令：

```bash
/home/chengshun.wang/miniconda3/envs/kan/bin/python - <<'PY'
import csv, statistics
from pathlib import Path
out=Path('results/v14_16_decisive_fms_fork_metric_state_allbasis/official_v1416')
b=list(csv.DictReader((out/'v1416_b_metric_state_results.csv').open(newline='')))
bc=list(csv.DictReader((out/'v1416_b_metric_state_controls.csv').open(newline='')))
m=list(csv.DictReader((out/'v1416_m_mlp_generic_controls.csv').open(newline='')))
def f(r,k):
    try: return float(r.get(k,0) or 0)
    except: return 0.0
print('LineB metric-state rows', len(b), sorted({r['method'] for r in b}))
print('LineB D-CHE control rows', len(bc), sorted({r['method'] for r in bc}))
print('LineM rows', len(m), sorted({r['method'] for r in m}))
for method in sorted({r['method'] for r in b}):
    rs=[r for r in b if r['method']==method]
    print('B', method, 'rows', len(rs),
          'mean_source_vs_best_control', statistics.fmean(f(r,'source_vs_best_control') for r in rs),
          'strict_pass', sum(int(float(r.get('strict_gate_pass',0) or 0)) for r in rs),
          'mean_NLL', statistics.fmean(f(r,'NLL') for r in rs))
for method in ['M8-MLP-MetricState-MS1','M9-MLP-MetricState-MS2']:
    rs=[r for r in m if r['method']==method]
    print('M', method, 'rows', len(rs),
          'mean_source_vs_best_control', statistics.fmean(f(r,'source_vs_best_control') for r in rs),
          'strict_pass', sum(int(float(r.get('strict_gate_pass',0) or 0)) for r in rs),
          'mean_NLL', statistics.fmean(f(r,'NLL') for r in rs))
PY
```

复核结果：

```text
LineB metric-state rows = 27
LineB methods =
  MS1-ParameterMetricState
  MS2-DegreeRoleMetricState
  MS3-BasisGroupMetricState

LineB D-CHE control rows = 54
LineB D-CHE controls =
  C0-D-CHE-AdamW
  C1-D-CHE-AdamW-NoOpMatchedOverhead
  C2-D-CHE-AdamW-RandomMatchedNorm
  C5-D-CHE-AdamW-SameActiveFractionControl
  C6-D-CHE-SameTCRandomDirection
  C7-D-CHE-SameMetricScaleRandomPermutation

LineM rows = 81
LineM MLP metric-state controls =
  M8-MLP-MetricState-MS1
  M9-MLP-MetricState-MS2

MS1-ParameterMetricState:
  rows = 9
  mean_source_vs_best_control = -0.023307979106903076
  strict_pass = 0
  mean_NLL = 0.8350219064288669

MS2-DegreeRoleMetricState:
  rows = 9
  mean_source_vs_best_control = -0.040673659907446966
  strict_pass = 0
  mean_NLL = 0.8523875872294108

MS3-BasisGroupMetricState:
  rows = 9
  mean_source_vs_best_control = -0.0381844507323371
  strict_pass = 0
  mean_NLL = 0.8498983780543009

M8-MLP-MetricState-MS1:
  rows = 9
  mean_source_vs_best_control = -0.18170246150758532
  strict_pass = 0
  mean_NLL = 1.2892921831872728

M9-MLP-MetricState-MS2:
  rows = 9
  mean_source_vs_best_control = -0.12233130799399482
  strict_pass = 0
  mean_NLL = 1.2299210296736822
```

结论：

```text
计划中的 C4 MLP-MetricState control 由 Line M 的 M8/M9 覆盖；
不是 D-CHE control CSV 缺失。
D-CHE MS1/MS2/MS3 和 MLP metric-state controls 都没有 positive gate。
没有发现新的合法补跑分支。
```
