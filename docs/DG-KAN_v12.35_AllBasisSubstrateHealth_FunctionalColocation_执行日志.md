# DG-KAN v12.35 AllBasisSubstrateHealth FunctionalColocation 执行日志

生成时间：2026-05-27（Asia/Singapore）

本日志只记录实际执行过的命令、输入文件、输出文件与直接观察到的结果。smoke / diagnostic / skipped row 不写成 promotion。

## 1. 计划文件

```bash
sed -n '1,240p' docs/DG-KAN_v12.35_AllBasisSubstrateHealth_FunctionalColocation_完整计划.md
sed -n '240,520p' docs/DG-KAN_v12.35_AllBasisSubstrateHealth_FunctionalColocation_完整计划.md
sed -n '520,980p' docs/DG-KAN_v12.35_AllBasisSubstrateHealth_FunctionalColocation_完整计划.md
```

理解到的 v12.35 执行顺序：

```text
Line R: code/provenance/implementation readback
Line S: substrate-health map v2
Line Q: basis response dictionary
Line B: basis-specific channel co-location functional repair
Line N: Non-RAT task-health substrate repair
Line M: MLP functional closure monitor
Line C: Manifold-Channel audit
Line Z: finalizer / route / no-go boundary
```

## 2. 复用的真实输入 artifact

输入目录：

```text
results/v12_34_2_all_basis_substrate_functional_repair/official_v12342
```

检查命令：

```bash
ls -1 results/v12_34_2_all_basis_substrate_functional_repair/official_v12342 | sed -n '1,220p'
python - <<'PY'
import csv, pathlib, json
p=pathlib.Path('results/v12_34_2_all_basis_substrate_functional_repair/official_v12342')
for name in ['v12342_family_substrate_summary.csv','v12342_basis_functional_p3.csv','v12342_basis_functional_nonrat_foreachoff_p3.csv','v12342_mlp_functional_closure.csv','v12342_route_decision.json']:
 print('\n==',name)
 path=p/name
 if name.endswith('.json'):
  print(path.read_text()[:2000])
 else:
  with path.open() as f:
   r=csv.DictReader(f); rows=list(r)
  print('rows',len(rows),'fields',r.fieldnames[:40])
  for row in rows[:3]: print({k:row.get(k) for k in r.fieldnames[:12]})
PY
```

直接观察：

```text
v12342_family_substrate_summary.csv rows = 44
v12342_basis_functional_p3.csv rows = 135
v12342_basis_functional_nonrat_foreachoff_p3.csv rows = 135
v12342_mlp_functional_closure.csv rows = 81
v12342_route_decision.json route = R2-SubstrateButNoFunctionalRepair
```

## 3. 代码修改与语法检查

修改文件：

```text
dgkan/diagnostics/basis_workspace.py
experiments/run_v1231_basis_kernel_workspace.py
experiments/run_v1231_rational_auc_hardening.py
experiments/run_v1235_response_dictionary.py
experiments/run_v1235_finalize_substrate_health_colocation.py
```

语法检查：

```bash
python -m py_compile dgkan/diagnostics/basis_workspace.py experiments/run_v1231_basis_kernel_workspace.py experiments/run_v1231_rational_auc_hardening.py experiments/run_v1235_response_dictionary.py experiments/run_v1235_finalize_substrate_health_colocation.py
```

结果：

```text
py_compile pass
```

## 11. 用户再次追问后的 stop-contract 复核 4

再次读取 route：

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_route_decision.json')
d=json.loads(p.read_text())
keys=['route','route_detail','minimum_success','official_success_reached','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','basis_substrate_health_rows','substrate_health_gate_pass_count','substrate_health_near_pass_count','healthy_base_gate_pass_count','nonrat_substrate_health_pass_count','basis_response_dictionary_rows','basis_response_dictionary_executed_rows','response_dictionary_pass_count','basis_functional_p3_pass_count','basis_functional_p4_pass_count','nonrat_task_health_repair_rows','nonrat_task_health_repair_pass_count','mlp_functional_mj_v3_pass_rows','mlp_functional_no_go_current_family_v3','provenance_violation_count','code_review_surface_incomplete_count','code_review_packet_entries','code_review_packet_sha256']
for k in keys:
    print(f'{k}={d.get(k)}')
PY
```

结果：

```text
route=R2-SubstrateButNoFunctionalRepair
route_detail=R4-RationalOnlySubstrate also applies; R3-NonRATLifetimeTaskHealthSplit also applies; R5-MLPFunctionalNoGo also applies; ResponseDictionaryNoGo
minimum_success=S1-SubstrateHealthPass
official_success_reached=0
promotion_allowed=0
final_stop_allowed=1
hard_compute_budget_exhausted=1
fallback_depth=5
fallback_all_executed=1
required_artifact_missing_count=0
basis_substrate_health_rows=44
substrate_health_gate_pass_count=11
substrate_health_near_pass_count=16
healthy_base_gate_pass_count=0
nonrat_substrate_health_pass_count=0
basis_response_dictionary_rows=270
basis_response_dictionary_executed_rows=195
response_dictionary_pass_count=0
basis_functional_p3_pass_count=0
basis_functional_p4_pass_count=0
nonrat_task_health_repair_rows=40
nonrat_task_health_repair_pass_count=0
mlp_functional_mj_v3_pass_rows=0
mlp_functional_no_go_current_family_v3=1
provenance_violation_count=0
code_review_surface_incomplete_count=0
code_review_packet_entries=35
code_review_packet_sha256=cfba47f617b7a45d5b692b760e6527f861e3e6af414b1e785ae401e7a61420bd
```

判断：v12.35 仍未达成 S5，只达到 S1；P3/P4、response dictionary、Non-RAT task-health repair、MLP M-J v3 closure 全部 0 pass。本次未新增训练实验、未新增 CSV 指标、未改 gate。由于本日志发生追加，随后重跑 finalizer 更新 required manifest 与 code review packet。

```bash
conda run -n kan python experiments/run_v1235_finalize_substrate_health_colocation.py
```

重跑结果：

```text
route = R2-SubstrateButNoFunctionalRepair
minimum_success = S1-SubstrateHealthPass
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
required_artifact_missing_count = 0
response_dictionary_pass_count = 0
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
nonrat_task_health_repair_pass_count = 0
mlp_functional_mj_v3_pass_rows = 0
provenance_violation_count = 0
code_review_surface_incomplete_count = 0
```

注：本段日志追加后再次运行同一 finalizer，最终 `code_review_packet_sha256` 以 `v1235_route_decision.json` 为准。

## 8. 用户再次追问后的 stop-contract 复核

用户再次要求确认 v12.35 是否达成目标，若未达成则继续。本次只读取最终 route / manifest 与计划文件 stop/go 条款，不新增训练实验。

复核命令：

```bash
python - <<'PY'
import json,csv,pathlib
p=pathlib.Path('results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235')
r=json.loads((p/'v1235_route_decision.json').read_text())
keys=['route','route_detail','minimum_success','official_success_reached','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','basis_substrate_health_rows','substrate_health_gate_pass_count','substrate_health_near_pass_count','healthy_base_gate_pass_count','nonrat_substrate_health_pass_count','response_dictionary_pass_count','basis_functional_p3_pass_count','basis_functional_p4_pass_count','nonrat_task_health_repair_pass_count','mlp_functional_mj_v3_pass_rows','provenance_violation_count','code_review_surface_incomplete_count']
for k in keys:
 print(f'{k} = {r.get(k)}')
rows=list(csv.DictReader((p/'v1235_required_artifact_manifest.csv').open()))
print(f'manifest_rows = {len(rows)}')
print(f'missing_rows = {sum(1 for row in rows if row.get("exists") == "0")}')
PY

rg -n "S1-SubstrateHealthPass|S2-ResponseDictionaryPass|S3-BasisFunctionalP3Pass|S4-BasisFunctionalP4Pass|S5-OfficialFunctionalSuccess|If functional P3 fails|If Non-RAT lifetime opens|If MLP closure fails" docs/DG-KAN_v12.35_AllBasisSubstrateHealth_FunctionalColocation_完整计划.md
```

结果：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R4-RationalOnlySubstrate also applies; R3-NonRATLifetimeTaskHealthSplit also applies; R5-MLPFunctionalNoGo also applies; ResponseDictionaryNoGo
minimum_success = S1-SubstrateHealthPass
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 5
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_substrate_health_rows = 44
substrate_health_gate_pass_count = 11
substrate_health_near_pass_count = 16
healthy_base_gate_pass_count = 0
nonrat_substrate_health_pass_count = 0
response_dictionary_pass_count = 0
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
nonrat_task_health_repair_pass_count = 0
mlp_functional_mj_v3_pass_rows = 0
provenance_violation_count = 0
code_review_surface_incomplete_count = 0
manifest_rows = 33
missing_rows = 0
```

计划文件复核点：

```text
S1/S2/S3/S4/S5 route 定义存在。
functional P3 CouplingR2 blocker 后要求 build response dictionary。
Non-RAT lifetime opens but task collapses 后要求 task-health repair。
MLP closure fails 后要求 mark no-go 并减少 MLP functional budget。
```

本次没有新增训练实验、没有新增 CSV 指标、没有改动 gate。原因是上述 fallback 均已经在 v1235 official artifacts 中执行完，并且 `final_stop_allowed=1`。为使 code packet 包含本次复核日志，追加本节后重跑：

```bash
conda run -n kan python experiments/run_v1235_finalize_substrate_health_colocation.py
```

## 10. 用户再次追问后的 stop-contract 复核 3

用户再次要求确认 v12.35 是否达成目标，若未达成则继续。本次再次读取最终 route / manifest，并对照计划 failure-handling 规则：

```bash
python - <<'PY'
import json,csv,pathlib
p=pathlib.Path('results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235')
r=json.loads((p/'v1235_route_decision.json').read_text())
keys=['route','route_detail','minimum_success','official_success_reached','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','basis_substrate_health_rows','substrate_health_gate_pass_count','substrate_health_near_pass_count','healthy_base_gate_pass_count','nonrat_substrate_health_pass_count','basis_response_dictionary_rows','basis_response_dictionary_executed_rows','response_dictionary_pass_count','basis_functional_p3_pass_count','basis_functional_p4_pass_count','nonrat_task_health_repair_pass_count','mlp_functional_mj_v3_pass_rows','provenance_violation_count','code_review_surface_incomplete_count','code_review_packet_sha256']
for k in keys:
 print(f'{k} = {r.get(k)}')
rows=list(csv.DictReader((p/'v1235_required_artifact_manifest.csv').open()))
print(f'manifest_rows = {len(rows)}')
print(f'missing_rows = {sum(1 for row in rows if row.get("exists") == "0")}')
PY

sed -n '940,975p' docs/DG-KAN_v12.35_AllBasisSubstrateHealth_FunctionalColocation_完整计划.md
```

结果：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R4-RationalOnlySubstrate also applies; R3-NonRATLifetimeTaskHealthSplit also applies; R5-MLPFunctionalNoGo also applies; ResponseDictionaryNoGo
minimum_success = S1-SubstrateHealthPass
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 5
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_substrate_health_rows = 44
substrate_health_gate_pass_count = 11
substrate_health_near_pass_count = 16
healthy_base_gate_pass_count = 0
nonrat_substrate_health_pass_count = 0
basis_response_dictionary_rows = 270
basis_response_dictionary_executed_rows = 195
response_dictionary_pass_count = 0
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
nonrat_task_health_repair_pass_count = 0
mlp_functional_mj_v3_pass_rows = 0
provenance_violation_count = 0
code_review_surface_incomplete_count = 0
manifest_rows = 33
missing_rows = 0
```

计划 failure-handling 对照：

```text
P3/CouplingR2 blocker -> 已建立 response dictionary，response_dictionary_pass_count=0。
Non-RAT lifetime opens but task collapses -> 已执行 family-specific task-health repair audit，nonrat_task_health_repair_pass_count=0。
MLP closure fails -> 已标记 MLPFunctionalNoGo v3，不继续 M-J grid。
artifacts missing -> required_artifact_missing_count=0。
implementation readback incomplete -> code_review_surface_incomplete_count=0。
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改代码或 gate。原因仍是：计划内 fallback 全部闭合，且我已经不确定继续在同一 B-RAT/B-FOU/M-J token family 上排列局部变体能形成有效机制。

为使 code packet 包含本节，追加后重跑：

```bash
conda run -n kan python experiments/run_v1235_finalize_substrate_health_colocation.py
```

## 9. 用户再次追问后的 stop-contract 复核 2

用户再次要求确认 v12.35 是否达成目标，若未达成则继续。本次重新读取最终 route / manifest：

```bash
python - <<'PY'
import json,csv,pathlib
p=pathlib.Path('results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235')
r=json.loads((p/'v1235_route_decision.json').read_text())
keys=['route','route_detail','minimum_success','official_success_reached','promotion_allowed','final_stop_allowed','hard_compute_budget_exhausted','fallback_depth','fallback_all_executed','required_artifact_missing_count','basis_substrate_health_rows','substrate_health_gate_pass_count','substrate_health_near_pass_count','healthy_base_gate_pass_count','nonrat_substrate_health_pass_count','response_dictionary_pass_count','basis_functional_p3_pass_count','basis_functional_p4_pass_count','nonrat_task_health_repair_pass_count','mlp_functional_mj_v3_pass_rows','provenance_violation_count','code_review_surface_incomplete_count']
for k in keys:
 print(f'{k} = {r.get(k)}')
rows=list(csv.DictReader((p/'v1235_required_artifact_manifest.csv').open()))
print(f'manifest_rows = {len(rows)}')
print(f'missing_rows = {sum(1 for row in rows if row.get("exists") == "0")}')
PY
```

结果：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R4-RationalOnlySubstrate also applies; R3-NonRATLifetimeTaskHealthSplit also applies; R5-MLPFunctionalNoGo also applies; ResponseDictionaryNoGo
minimum_success = S1-SubstrateHealthPass
official_success_reached = 0
promotion_allowed = 0
final_stop_allowed = 1
hard_compute_budget_exhausted = 1
fallback_depth = 5
fallback_all_executed = 1
required_artifact_missing_count = 0
basis_substrate_health_rows = 44
substrate_health_gate_pass_count = 11
substrate_health_near_pass_count = 16
healthy_base_gate_pass_count = 0
nonrat_substrate_health_pass_count = 0
response_dictionary_pass_count = 0
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
nonrat_task_health_repair_pass_count = 0
mlp_functional_mj_v3_pass_rows = 0
provenance_violation_count = 0
code_review_surface_incomplete_count = 0
manifest_rows = 33
missing_rows = 0
```

本次没有新增训练实验、没有新增 CSV 指标、没有修改代码或 gate。原因仍是：计划要求的 response dictionary、Non-RAT task-health repair、MLP closure no-go、provenance/required artifact 审计均已执行，且我已经不确定继续在同一 B-RAT/B-FOU/M-J token family 上排列局部变体能形成有效机制。

为使 code packet 包含本节，追加后重跑：

```bash
conda run -n kan python experiments/run_v1235_finalize_substrate_health_colocation.py
```


默认 `python` 环境 blocker：

```bash
python - <<'PY'
from dgkan.diagnostics.basis_workspace import V1235_BASIS_CANDIDATES
PY

python experiments/run_v1235_response_dictionary.py --source-dir results/v12_34_2_all_basis_substrate_functional_repair/official_v12342 --out-dir results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235 --artifact-prefix v1235
```

结果：

```text
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'numpy'
```

环境修复：

```bash
conda env list
conda run -n kan python -c "import torch,numpy; print('torch', torch.__version__); print('numpy', numpy.__version__)"
conda run -n kan python -m py_compile dgkan/diagnostics/basis_workspace.py experiments/run_v1231_basis_kernel_workspace.py experiments/run_v1231_rational_auc_hardening.py experiments/run_v1235_response_dictionary.py experiments/run_v1235_finalize_substrate_health_colocation.py
```

结果：

```text
conda env = kan
torch 2.11.0+cu128
numpy 2.4.4
py_compile pass
```

## 4. v1235 registry smoke

```bash
conda run -n kan python -c "from dgkan.diagnostics.basis_workspace import V1235_BASIS_CANDIDATES; from collections import Counter; print('basis_candidates', len(V1235_BASIS_CANDIDATES)); print('families', ','.join(sorted(Counter(c.family for c in V1235_BASIS_CANDIDATES.values())))); print('new_candidates', ','.join([c for c in V1235_BASIS_CANDIDATES if c.startswith(('D-RAT40','D-CHE20','D-FOU20','D-RBF17','D-WAV16'))]))"
```

结果：

```text
basis_candidates 49
families D-CHE,D-FOU,D-RAT,D-RBF,D-WAV
new_candidates D-RAT40-ResponseReadyReadoutCouplingSubstrate,D-CHE20-DegreeNormalizedReadoutHealthSubstrate,D-FOU20-LowFreqIdentityResidualHealthSubstrate,D-RBF17-CompactCapacityK4HealthSubstrate,D-WAV16-SupportStableHatHealthSubstrate
```

CUDA 检查：

```bash
conda run -n kan python -c "import torch; print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
```

结果：

```text
cuda_available True
device_count 4
```

## 5. v1235 runner smoke

Workspace smoke：

```bash
conda run -n kan python experiments/run_v1231_basis_kernel_workspace.py --out-dir results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235 --artifact-prefix v1235_smoke_basis --run-id v1235_smoke_basis --candidate-registry v1235 --candidates D-RAT40-ResponseReadyReadoutCouplingSubstrate,D-FOU20-LowFreqIdentityResidualHealthSubstrate --datasets MNIST --seeds 0 --train-size 64 --val-size 32 --batch-size 16 --hardening-epochs 1 --mlp-reference-epochs 1 --workspace-warmup-steps 1 --workspace-profile-steps 1 --device cuda:0 --data-root data --no-download
```

结果：

```text
workspace_rows = 2
workspace_gate_pass_rows = 1
workspace_strong_gate_pass_rows = 1
hardening_executed_rows = 1
family_near_pass_rows = 0
promotion_allowed = 0
```

AUC/telemetry smoke：

```bash
conda run -n kan python experiments/run_v1231_rational_auc_hardening.py --out-dir results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235 --artifact-prefix v1235_smoke_auc --run-id v1235_smoke_auc --candidate-registry v1235 --candidates D-RAT40-ResponseReadyReadoutCouplingSubstrate,D-FOU20-LowFreqIdentityResidualHealthSubstrate --workspace-csv results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235/v1235_smoke_basis_workspace_truth.csv --datasets MNIST --seeds 0 --train-size 64 --val-size 32 --batch-size 16 --epochs 1 --device cuda:0 --data-root data --no-download
```

结果：

```text
candidate_rows = 2
summary_rows = 2
linec_rows = 3
candidate_auc_near_pass_rows = 0
candidate_auc_official_pass_v1233_rows = 0
promotion_allowed = 0
```

## 6. Line Q response dictionary

```bash
conda run -n kan python experiments/run_v1235_response_dictionary.py --source-dir results/v12_34_2_all_basis_substrate_functional_repair/official_v12342 --out-dir results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235 --artifact-prefix v1235
```

结果：

```text
dictionary_rows = 270
dictionary_executed_rows = 195
response_dictionary_pass_rows = 0
p3_gate_recomputed_pass_rows = 0
summary_rows = 30
promotion_allowed = 0
```

生成：

```text
v1235_basis_response_dictionary.csv
v1235_basis_response_controls.csv
v1235_basis_response_linec.csv
v1235_basis_response_task_tail.csv
v1235_response_colocation_summary.csv
v1235_response_dictionary_route.json
```

## 7. Line Z finalizer

第一次 finalizer 后发现 route JSON 中 `required_artifact_missing_count=1`，但 manifest 已经显示 missing=0。原因是 route 在 code packet 生成前计算了一次 missing count。修复：

```text
experiments/run_v1235_finalize_substrate_health_colocation.py
重新生成 code packet 后再次计算 required_artifact_missing_count。
```

复查语法并重跑：

```bash
conda run -n kan python -m py_compile experiments/run_v1235_finalize_substrate_health_colocation.py
conda run -n kan python experiments/run_v1235_finalize_substrate_health_colocation.py
```

最终结果：

```text
route = R2-SubstrateButNoFunctionalRepair
route_detail = R4-RationalOnlySubstrate also applies; R3-NonRATLifetimeTaskHealthSplit also applies; R5-MLPFunctionalNoGo also applies; ResponseDictionaryNoGo
minimum_success = S1-SubstrateHealthPass
official_success_reached = 0
promotion_allowed = 0
required_artifact_missing_count = 0
basis_substrate_health_rows = 44
substrate_health_gate_pass_count = 11
substrate_health_near_pass_count = 16
healthy_base_gate_pass_count = 0
nonrat_substrate_health_pass_count = 0
basis_response_dictionary_rows = 270
basis_response_dictionary_executed_rows = 195
response_dictionary_pass_count = 0
basis_functional_p3_pass_count = 0
basis_functional_p4_pass_count = 0
nonrat_task_health_repair_pass_count = 0
mlp_functional_mj_v3_pass_rows = 0
provenance_violation_count = 0
code_review_surface_incomplete_count = 0
```

最终复核：

```bash
python - <<'PY'
import json,csv,pathlib
p=pathlib.Path('results/v12_35_all_basis_substrate_health_functional_colocation/official_v1235')
r=json.loads((p/'v1235_route_decision.json').read_text())
print('route', r['route'])
print('minimum_success', r['minimum_success'])
print('official_success_reached', r['official_success_reached'])
print('promotion_allowed', r['promotion_allowed'])
print('required_artifact_missing_count', r['required_artifact_missing_count'])
print('code_review_packet_sha256', r['code_review_packet_sha256'])
rows=list(csv.DictReader((p/'v1235_required_artifact_manifest.csv').open()))
print('manifest_rows', len(rows), 'missing', sum(1 for row in rows if row['exists']=='0'))
PY

conda run -n kan python -m py_compile dgkan/diagnostics/basis_workspace.py experiments/run_v1231_basis_kernel_workspace.py experiments/run_v1231_rational_auc_hardening.py experiments/run_v1235_response_dictionary.py experiments/run_v1235_finalize_substrate_health_colocation.py
```

结果：

```text
route R2-SubstrateButNoFunctionalRepair
minimum_success S1-SubstrateHealthPass
official_success_reached 0
promotion_allowed 0
required_artifact_missing_count 0
code_review_packet_sha256 以最终 v1235_route_decision.json 为准
manifest_rows 33 missing 0
py_compile pass
```
